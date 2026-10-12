"""Run walk-forward backtest with real historical data.

Fetches 2 years of daily OHLCV from Yahoo Finance for configured symbols,
then runs the walk-forward backtester with all enabled strategies.

Usage:
    python run_backtest.py
    python run_backtest.py --symbols SPY QQQ AAPL --days 500
    python run_backtest.py --strategies long_call bull_call_spread iron_condor
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

# ---------------------------------------------------------------------------
# Live-parity reference values (source of truth: the LIVE code paths)
# ---------------------------------------------------------------------------
# - AIT_IC_WING_K / wing floor $2:   src/ait/strategies/iron_condor.py _vol_scaled_width
# - AIT_IC_MIN_CREDIT ($0.70):       src/ait/strategies/iron_condor.py generate_signals
# - AIT_IC_MIN_CREDIT_WIDTH:         credit-to-width gate in iron_condor.py
# - AIT_CREDIT_LOSS_LIMIT (0=off):   R16 parity - live default DISABLED (R6: touch-close beats flat stops)
# - delta target 0.20:               hardcoded in iron_condor.generate_signals
# - DTE band:                        config.yaml options.dte_range (loaded, NOT the code default)
# - TP ladder:                       src/ait/execution/portfolio.py _get_take_profit_targets
# - IV rank floor 15:                AIT_IRON_CONDOR_IV_FLOOR default in iron_condor.py
# - pre-event blackout:              config.yaml risk.pre_event_blackout_days
#
# R16: the "live" column used to be HARDCODED here (dte_band [14,45] while
# config.yaml says [14,30]; a comment claiming a <=4d macro gate that is now 1
# day) and its env-backed entries read THIS process's environment rather than
# the contract the bot itself applies — so a bare `python run_backtest.py`
# compared the run against wing_k=1.0/ratio 0.20 while live runs 1.6/0.10, and
# the manifest systematically UNDERSTATED live-vs-backtest divergence. Every
# live value below is now resolved from the authorities: ait.config.runtime_env
# (the env contract every bot entry point applies) and load_settings().
_LIVE_TP_LADDER = {"dte>20": 0.50, "dte_11_20": 0.40, "dte_6_10": 0.30, "dte<=5": 0.20}


def _live_env() -> dict:
    """The env contract as the BOT resolves it, without mutating this process.

    run_orchestrator.py / ait.main call apply_runtime_env_defaults() at
    startup; a backtest process does not, so reading os.environ here reported
    code defaults as if they were live values. Apply the contract into a
    snapshot, then restore — the engine reads several of these vars, and this
    function must not change what the backtest itself runs.
    """
    saved = dict(os.environ)
    try:
        from ait.config.runtime_env import apply_runtime_env_defaults
        apply_runtime_env_defaults()
        return dict(os.environ)
    except Exception as e:  # noqa: BLE001 — manifest must never break a run
        print(f"  ! parity manifest: live env contract unavailable ({e})")
        return dict(os.environ)
    finally:
        for k in list(os.environ):
            if k not in saved:
                del os.environ[k]
        for k, v in saved.items():
            if os.environ.get(k) != v:
                os.environ[k] = v


def _live_settings():
    """Loaded config.yaml, or None if it cannot be read."""
    try:
        from ait.config.settings import load_settings
        # W68: the LIVE column must show config.yaml alone, even though this process has frozen
        # base+overlay (use_research_config) for everything else.
        return load_settings(ignore_frozen=True)
    except Exception as e:  # noqa: BLE001
        print(f"  ! parity manifest: settings unavailable ({e})")
        return None


def _research_settings():
    """config.yaml + config_research.yaml as THIS run sees it (the frozen snapshot)."""
    try:
        from ait.config.settings import load_settings
        return load_settings()
    except Exception as e:  # noqa: BLE001
        print(f"  ! parity manifest: research settings unavailable ({e})")
        return None


def build_parity_manifest(args: argparse.Namespace) -> dict:
    """Compare the live bot's config/env values against this backtest run.

    Any mismatch is VISIBLE (parity_warnings), never fatal — the point is
    that a backtest silently diverging from the live strategy is worse than
    one that loudly diverges on purpose.
    """
    from ait.backtesting.engine import Backtester
    from ait.config.settings import MLConfig
    from ait.config.runtime_env import contract_float

    env = _live_env()
    settings = _live_settings()
    rsettings = _research_settings()   # backtest column: base + research overlay
    live_dte = list(settings.options.dte_range) if settings else [14, 45]
    live_blackout = (settings.risk.pre_event_blackout_days
                     if settings else None)

    # R20b follow-up: wing_k/range_confidence now default to None on the CLI
    # (resolved by WalkForwardConfig.__post_init__), so report the RESOLVED
    # value here too -- same reasoning as pre_event_blackout_days above.
    if args.wing_k is not None:
        resolved_wing_k = args.wing_k
    else:
        resolved_wing_k = contract_float("AIT_IC_WING_K")
    if args.range_confidence is not None:
        resolved_range_confidence = args.range_confidence
    elif settings is not None:
        resolved_range_confidence = settings.ml.range_min_confidence
    else:
        resolved_range_confidence = MLConfig().range_min_confidence
    # R20b follow-up: same None -> contract_float resolution as wing_k above
    # -- was raw os.environ.get(..., HARDCODED_FALLBACK), which could never
    # pick up a config.yaml override (and ic-min-credit-width's fallback,
    # 0.20, was flatly wrong vs the contract's 0.10).
    resolved_credit_loss_limit = (
        args.credit_loss_limit if args.credit_loss_limit is not None
        else contract_float("AIT_CREDIT_LOSS_LIMIT")
    )
    resolved_ic_min_credit = (
        args.ic_min_credit if args.ic_min_credit is not None
        else contract_float("AIT_IC_MIN_CREDIT")
    )
    resolved_ic_min_credit_width = (
        args.ic_min_credit_width if args.ic_min_credit_width is not None
        else contract_float("AIT_IC_MIN_CREDIT_WIDTH")
    )

    # W64: per-trade budget / wing cap / contract cap. The backtest side is read
    # through the SAME resolvers WalkForwardConfig uses (there is no CLI override
    # for these), so a future change to either side shows up here as a warning.
    from ait.config.settings import (
        BacktestConfig as _BTC, RiskConfig as _RC, PositionConfig as _PC,
        resolve_config_value as _rcv, resolve_max_contracts_per_trade as _rmc,
        resolve_position_size_pct as _rps)
    live_budget_pct = float(settings.risk.max_position_risk_pct
                            if settings else _RC().max_position_risk_pct)
    live_max_contracts = int(settings.positions.max_contracts_per_trade
                             if settings else _PC().max_contracts_per_trade)
    bt_budget_pct = _rps(None, rsettings)
    bt_max_contracts = _rmc(None, rsettings)
    bt_wings = bool(_rcv(None, "backtest", "budget_aware_wings", _BTC, rsettings))

    live = {
        "per_trade_budget_pct": live_budget_pct,
        "budget_aware_wings": True,   # live caps the wing to the budget (iron_condor.py)
        "condor_any_direction": True,  # live IronCondor ignores market_direction (iron_condor.py)
        "max_contracts_per_trade": live_max_contracts,
        # W67: ml.entry_gates_enabled=false -> live observes; the shared fractal gate's
        # cutoffs are read from the same backtest.* fields by the orchestrator.
        "gates_mode": ("enforce" if (settings.ml.entry_gates_enabled if settings else True)
                       else "observe"),
        # W68: live gates credit entries on IV RANK (floor in the condor builder, cap in the
        # risk manager) -- both read the env contract the bot applies.
        "iv_rank_gate": True,
        "iv_rank_cap": contract_float("AIT_IV_RANK_CAP"),
        "fractal_veto_spread": float(settings.backtest.fractal_veto_spread
                                     if settings else _BTC().fractal_veto_spread),
        "penalty_on_range_prob": float(settings.backtest.penalty_on_range_prob
                                       if settings else _BTC().penalty_on_range_prob),
        "wing_k":            float(env.get("AIT_IC_WING_K", "1.0")),
        "range_min_confidence": (settings.ml.range_min_confidence
                                  if settings else MLConfig().range_min_confidence),
        "ic_min_credit":     float(env.get("AIT_IC_MIN_CREDIT", "0.70")),
        "ic_min_credit_width": float(env.get("AIT_IC_MIN_CREDIT_WIDTH", "0.20")),
        "credit_loss_limit": float(env.get("AIT_CREDIT_LOSS_LIMIT", "0")),
        "pre_event_blackout_days": live_blackout,
        "delta_target":      0.20,
        "dte_band":          live_dte,
        "tp_ladder":         dict(_LIVE_TP_LADDER),
        "wing_floor":        2.0,
        "macro_gate_entry":  True,   # orchestrator blocks credit entries within
                                     # risk.pre_event_blackout_days of an event
        "macro_flatten_enabled": env.get("AIT_SKIP_MACRO_EVENTS", "0") == "1",
        "iv_rank_floor":     float(env.get("AIT_IRON_CONDOR_IV_FLOOR", "15")),
    }

    # Backtest TP ladder read from the engine mirror itself, so this manifest
    # catches any future drift in Backtester._credit_take_profit_pct.
    bt_ladder = {
        "dte>20":    Backtester._credit_take_profit_pct(25),
        "dte_11_20": Backtester._credit_take_profit_pct(15),
        "dte_6_10":  Backtester._credit_take_profit_pct(8),
        "dte<=5":    Backtester._credit_take_profit_pct(3),
    }
    backtest = {
        "per_trade_budget_pct": bt_budget_pct,
        "budget_aware_wings": bt_wings,
        "condor_any_direction": bool(_rcv(None, "backtest", "condor_any_direction", _BTC, rsettings)),
        "friction_model": str(_rcv(None, "backtest", "friction_model", _BTC, rsettings)),
        "max_contracts_per_trade": bt_max_contracts,
        "iv_rank_gate": bool(_rcv(None, "backtest", "iv_rank_gate", _BTC, rsettings)),
        "iv_rank_cap": contract_float("AIT_IV_RANK_CAP"),
        "gates_mode": str(_rcv(getattr(args, "gates_mode", None), "backtest", "gates_mode",
                               _BTC, rsettings)),
        "fractal_veto_spread": float(_rcv(None, "backtest", "fractal_veto_spread", _BTC, rsettings)),
        "penalty_on_range_prob": float(_rcv(
            None, "backtest", "penalty_on_range_prob", _BTC, rsettings)),
        "wing_k":            resolved_wing_k,
        "range_min_confidence": resolved_range_confidence,
        "ic_min_credit":     resolved_ic_min_credit,
        "ic_min_credit_width": resolved_ic_min_credit_width,
        "credit_loss_limit": resolved_credit_loss_limit,
        # R16 #7: the walk-forward passes None, so the engine resolves the same
        # loaded settings value live uses — report the resolved number, not None.
        "pre_event_blackout_days": live_blackout,
        "delta_target":      0.20,   # Backtester delta_short default (walk-forward does not override)
        # PR#7 review 2026-08-25: was a stale hardcoded [21, 21] with the note
        # "engine uses fixed DTE = max_hold_days" — the engine's 2026-08-24
        # decoupling resolves entry_dte from options.dte_range[0] instead, so
        # the manifest reported a DTE no run actually priced. Report the same
        # resolution the engine performs.
        "dte_band":          [live_dte[0], live_dte[0]],
        "tp_ladder":         bt_ladder,
        "wing_floor":        args.wing_floor,
        "macro_gate_entry":  args.macro_gate,
        "macro_flatten_enabled": os.environ.get("AIT_SKIP_MACRO_EVENTS", "0") == "1",
        "iv_rank_floor":     args.iv_floor,
    }

    warnings: list[str] = []
    for key, live_val in live.items():
        bt_val = backtest.get(key)
        if key == "dte_band":
            # Engine trades a single fixed DTE; parity holds if it sits inside
            # the live band rather than being equal to it.
            lo, hi = live_val
            if not (lo <= bt_val[0] <= hi and lo <= bt_val[1] <= hi):
                warnings.append(
                    f"dte_band: backtest fixed DTE {bt_val} outside live band {live_val}"
                )
            continue
        if bt_val != live_val:
            warnings.append(f"{key}: live={live_val} backtest={bt_val}")

    # W74: research fidelity. No live counterpart, so it is not part of the
    # live-vs-backtest compare above; it is surfaced as its own warning because
    # 'legacy' still carries the confirmed W72 defects (L2-L6) and a study that
    # does not say which engine it ran is not reproducible.
    _bt_cfg = getattr(rsettings, "backtest", None) if rsettings is not None else None
    _engine_model = getattr(_bt_cfg, "engine_model", "legacy")
    backtest["engine_model"] = _engine_model
    for _k in ("expiry_guard_dte", "max_hold_fraction", "flatten_minutes_before_close"):
        backtest[_k] = getattr(_bt_cfg, _k, None)
    if _engine_model != "realistic":
        warnings.append(
            "engine_model: 'legacy' - entries are priced at the DAILY close with same-day IV, "
            "expiry = entry + N calendar days (can be a weekend), no same-session exit check, "
            "intraday take-profit decided after friction and blind to touches, condors "
            "invalidated on direction (W72 leads L2-L6). 'realistic' fixes them; flip "
            "backtest.engine_model in config_research.yaml after recording a before/after (A10).")

    notes = [
        # R16: this note used to hardcode a 4-day pre-event window; the
        # blackout has been risk.pre_event_blackout_days (currently 1) since
        # settings.py:64, so it is interpolated from the loaded settings now.
        f"macro gate blocks credit entries within {live_blackout} day(s) of an "
        "event, using the hardcoded-2026 economic calendar "
        "(src/ait/data/economic_calendar.py); for pre-2026 backtest windows "
        "days-to-event is always larger, so the gate is effectively INACTIVE — "
        "acceptable, but macro-event losses are NOT simulated there.",
        "live macro FLATTEN (portfolio.py rule 3d) is env-gated behind "
        "AIT_SKIP_MACRO_EVENTS=1; defined-risk condors are EXEMPT in code, so "
        "it governs undefined/assignment-risk arms (strangle, jade_lizard, "
        "CSP/CC); the backtest mirrors that env var exactly.",
        # PR#7 validation 2026-08-25 (F33 second half): this note still
        # described the retired fixed-DTE=max_hold_days convention, directly
        # contradicting the corrected dte_band lines above in the same
        # manifest.
        f"engine prices a fixed synthetic entry DTE = dte_range[0] "
        f"({live_dte[0]}) inside the live dte_range {live_dte}; the hold is "
        "capped separately by max_hold_days (entry_dte/hold-cap decoupled "
        "2026-08-24).",
        "friction_model (backtest column only, no live counterpart: live pays real "
        "fills) selects the research transaction-cost model; see "
        "scripts/fill_vs_model_report.py for the forward check against real fills.",
        "credit exits: flat loss limit + DTE-laddered TP + DTE<=5 close "
        "(mirrors portfolio.py); trailing/breakeven applies to DEBIT only.",
        "the 'live' column is resolved from ait.config.runtime_env (the env "
        "contract every bot entry point applies) + load_settings(), NOT from "
        "this process's environment — a bare backtest run therefore shows the "
        "REAL divergence instead of comparing the run against itself.",
    ]

    return {
        "live": live,
        "backtest": backtest,
        "parity_warnings": warnings,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AIT Walk-Forward Backtester")
    p.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL"],
        help="Symbols to backtest",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "put_credit_spread",
            "bull_call_spread",
            "bear_put_spread",
            "iron_condor",
        ],
        help="Strategies to test",
    )
    p.add_argument("--train-days", type=int, default=365, help="Training window days (calendar)")
    p.add_argument("--test-days", type=int, default=63, help="Test window days")
    p.add_argument("--step-days", type=int, default=63, help="Step between windows")
    p.add_argument("--gap-days", type=int, default=5, help="Purge gap days")
    p.add_argument("--capital", type=float, default=None,
                   help="TOTAL starting equity, one shared pool across symbols. "
                        "Default: config.yaml account.capital (W-capital: the single "
                        "home; was a 50,000 literal here).")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="Min ML confidence. Default resolves from config.yaml "
                        "risk.min_confidence (currently 0.50 via WalkForwardConfig "
                        "-- was a stale hardcoded 0.65 that overrode the config-"
                        "resolved value (R20b follow-up).")
    p.add_argument("--range-confidence", type=float, default=None,
                   help="Min P(in_range) for iron condors (range model). "
                        "Default resolves from config.yaml ml.range_min_confidence "
                        "(currently 0.65) -- was a stale hardcoded 0.55 (R20b follow-up).")
    p.add_argument("--iv-floor", type=float, default=15.0,
                   help="Min IV rank for iron condors (aligned with live "
                        "AIT_IRON_CONDOR_IV_FLOOR default of 15; was 30)")
    # BooleanOptionalAction gives --trailing-stop / --no-trailing-stop.
    # (Old action="store_true" with default=True made it impossible to disable.)
    # Trailing applies to DEBIT trades only — credit trades always use the
    # live-parity flat loss limit + DTE-laddered take-profit.
    p.add_argument("--trailing-stop", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable trailing stops (debit trades only)")
    # --- R6 live-parity knobs (default resolves env > config.yaml > contract
    # default via contract_float, same as --wing-k below -- was raw
    # os.environ.get(..., HARDCODED_FALLBACK), which could never pick up a
    # config.yaml override and, for --ic-min-credit-width, hardcoded the
    # WRONG fallback: 0.20 vs the contract's 0.10 (CONTRACT_DEFAULTS in
    # runtime_env.py), so a bare run silently traded a 2x looser credit/width
    # gate than live. R20b follow-up.) ---
    p.add_argument("--credit-loss-limit", type=float, default=None,
                   help="Flat loss limit for credit trades, as multiple of credit "
                        "received. Default resolves env AIT_CREDIT_LOSS_LIMIT > "
                        "config.yaml backtest.credit_loss_limit > 0 (off per R6/R16).")
    p.add_argument("--ic-min-credit", type=float, default=None,
                   help="Min mid-price total credit for iron condors. Default "
                        "resolves env AIT_IC_MIN_CREDIT > config.yaml "
                        "backtest.ic_min_credit > 0.70.")
    p.add_argument("--ic-min-credit-width", type=float, default=None,
                   help="Min credit/max-width ratio for iron condors. Default "
                        "resolves env AIT_IC_MIN_CREDIT_WIDTH > config.yaml "
                        "backtest.ic_min_credit_width > 0.10 -- was a hardcoded "
                        "0.20, 2x the contract default (R20b follow-up).")
    p.add_argument("--wing-k", type=float, default=None,
                   help="Wing width = wing_k*price*IV*sqrt(DTE/365). Default "
                        "resolves env AIT_IC_WING_K > config.yaml backtest.wing_k "
                        "> 1.6 (contract_float) -- was a stale hardcoded 1.0 "
                        "that shadowed the promoted live value (R20b follow-up).")
    p.add_argument("--wing-floor", type=float, default=None,
                   help="Hard minimum wing width in $ (live enforces $2; old backtest "
                        "default was $5). Default: config backtest.wing_floor_dollars (W70).")
    p.add_argument("--macro-gate", action=argparse.BooleanOptionalAction, default=True,
                   help="Block credit entries <=4 days before FOMC/CPI/NFP/GDP/PCE "
                        "(2026-only hardcoded calendar; inactive for earlier windows)")
    p.add_argument("--compare-exits", action="store_true", help="Compare fixed vs trailing stops")
    p.add_argument("--optimize-per-window", action="store_true", default=False,
                   help="Run Optuna optimization on each training window before testing")
    p.add_argument("--optimize-n-trials", type=int, default=50,
                   help="Number of Optuna trials per walk-forward window (requires --optimize-per-window)")
    # W65 research-history flags. Defaults come from config.yaml backtest.*.
    p.add_argument("--history-days", type=int, default=None,
                   help="Calendar days of history to request. Overrides config "
                        "backtest.history_start_date (which otherwise sets the span).")
    p.add_argument("--table-prefix", default=None,
                   help="Research table prefix (default: config backtest.research_table_prefix, "
                        "'research_'). Pass '' to use the live tables / legacy loader.")
    p.add_argument("--db-path", default="data/historical.db",
                   help="SQLite store holding the research tables (default: data/historical.db).")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Keep only the LAST N walk-forward windows (timing / pilot runs).")
    p.add_argument("--gates-mode", choices=["observe", "enforce"], default=None,
                   help="W67: observe = live parity (range floor + fractal veto recorded as "
                        "would_reject:*, never block); enforce = they block. Default: "
                        "config backtest.gates_mode.")
    return p.parse_args()


async def run_backtest(args: argparse.Namespace) -> None:
    # W68: config.yaml + config_research.yaml is THIS process's configuration (research only;
    # the live bot never loads the overlay). Fails loudly if the overlay file is missing.
    from ait.config.research import use_research_config
    use_research_config()
    if args.capital is None:
        from ait.config.runtime_env import research_capital
        args.capital = research_capital()
    if args.wing_floor is None:  # W70: the CLI default was a literal 2.0
        from ait.config.runtime_env import contract_float
        # the SAME contract call live's iron_condor.py uses (home backtest.wing_floor_dollars)
        args.wing_floor = contract_float("AIT_IC_WING_FLOOR")
    print("=" * 60)
    print("  AIT v2 — WALK-FORWARD BACKTEST")
    print("=" * 60)
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Strategies: {', '.join(args.strategies)}")
    print(f"  Train:      {args.train_days}d  |  Test: {args.test_days}d  |  Step: {args.step_days}d")
    print(f"  Capital:    ${args.capital:,.0f}")
    print(f"  Trailing:   {'ON (debit trades only)' if args.trailing_stop else 'OFF'}")
    print("=" * 60)

    # --- Parameter-parity manifest: live config/env vs this backtest run ---
    # R16: built BEFORE the env override below. Setting AIT_IRON_CONDOR_IV_FLOOR
    # first made the manifest's "live" iv_rank_floor read back this run's own
    # --iv-floor, so that row could never disagree — the manifest was partly
    # comparing the run against itself.
    manifest = build_parity_manifest(args)

    # Set IV floor for iron condor strategy (read by iron_condor.py via env)
    os.environ["AIT_IRON_CONDOR_IV_FLOOR"] = str(args.iv_floor)
    print("\n  PARAMETER-PARITY MANIFEST (live vs backtest):")
    print("  " + json.dumps(manifest, indent=2).replace("\n", "\n  "))
    if manifest["parity_warnings"]:
        print("\n  PARITY WARNINGS (backtest diverges from live — visible, not fatal):")
        for w in manifest["parity_warnings"]:
            print(f"    ! {w}")
    else:
        print("\n  PARITY WARNINGS: none — backtest parameters match live.")

    # --- W65: research history source (IB research tables) ------------------
    # config.yaml backtest.* is the home of these values; the CLI only overrides.
    # A non-empty prefix means "research mode": IB daily bars + IV from the
    # research_* tables, the operator's economic calendar, and (coverage
    # permitting) 5-min VLMC features via the prefixed intraday store.
    try:
        from ait.config.settings import load_settings
        _bcfg = load_settings().backtest
    except Exception as _exc:  # noqa: BLE001
        print(f"  ! config unavailable ({_exc}); research defaults off")
        _bcfg = None
    _prefix = (args.table_prefix if args.table_prefix is not None
               else (_bcfg.research_table_prefix if _bcfg else ""))
    _research = bool(_prefix)
    _start = (None if args.history_days is not None
              else (_bcfg.history_start_date if (_bcfg and _research) else None))
    _hist_days = (args.history_days if args.history_days is not None
                  else (_bcfg.history_days if (_bcfg and _research) else None))
    if _research:
        print(f"  History:    research tables '{_prefix}*' in {args.db_path} | "
              f"{'from ' + str(_start) if _start else str(_hist_days) + ' days'}")
    else:
        print("  History:    legacy loader (5-min resample / Yahoo), live tables")

    cfg = WalkForwardConfig(
        gates_mode=args.gates_mode,
        max_windows=args.max_windows,
        history_start_date=_start,
        history_days=_hist_days,
        vlmc_min_coverage=(_bcfg.vlmc_min_coverage if _bcfg else 0.95),
        macro_event_types=(tuple(_bcfg.macro_event_types) if (_bcfg and _research) else None),
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        gap_days=args.gap_days,
        initial_capital=args.capital,
        min_confidence=args.min_confidence,
        range_min_confidence=args.range_confidence,
        trailing_stop_enabled=args.trailing_stop,
        optimize_per_window=args.optimize_per_window,
        optimize_n_trials=args.optimize_n_trials,
        # R6 live-parity knobs
        wing_k=args.wing_k,
        wing_floor_dollars=args.wing_floor,
        credit_loss_limit_mult=args.credit_loss_limit,
        ic_min_credit=args.ic_min_credit,
        ic_min_credit_width=args.ic_min_credit_width,
        macro_event_gate=args.macro_gate,
    )

    bt = WalkForwardBacktester(
        symbols=args.symbols,
        strategies=args.strategies,
        config=cfg,
        db_path=(Path(args.db_path) if _research else None),
        table_prefix=_prefix,
    )

    # Fetch data
    print(f"\n[1/3] Fetching historical data ({'IB research tables' if _research else 'IB store / Yahoo'})...")
    t0 = time.time()
    data = await bt._fetch_data()
    fetch_time = time.time() - t0

    if not data:
        print("ERROR: No data fetched. Check internet connection.")
        sys.exit(1)

    # Normalize to tz-naive: Yahoo/IB can return tz-aware (America/New_York)
    # indexes, but walkforward normalizes its VIX/SPY context to tz-naive —
    # mixing the two raises "Cannot compare dtypes datetime64[ns] and
    # datetime64[ns, America/New_York]" inside the window loop.
    for sym, df in data.items():
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

    print(f"  Fetched {len(data)} symbols in {fetch_time:.1f}s:")
    for sym, df in data.items():
        print(f"    {sym}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Run walk-forward backtest
    print("\n[2/3] Running walk-forward backtest...")
    t0 = time.time()
    result = await bt.run(data=data)
    run_time = time.time() - t0
    print(f"  Completed in {run_time:.1f}s")
    if len(result.windows) < 3:
        print("\n" + "!" * 60)
        print(f"  WARNING: only {len(result.windows)} walk-forward window(s) produced trades/results. "
              "Check the data span above (windows need train+gap+test days of history\n"
              "  for EVERY symbol) and the log lines walk_forward_too_few_windows / "
              "range_model_absent_blocking_oos.")
        print("!" * 60)

    # Attach the parity manifest to the result object so any downstream JSON
    # serialization of this run (dashboards, experiment writers) carries it.
    result.parity_manifest = manifest

    # Print results
    print(f"\n{result.summary()}")

    # Buy-and-hold benchmark
    print("\n[3/3] Buy-and-hold benchmark:")
    benchmark = bt.benchmark_buy_hold(data)
    for sym, ret in sorted(benchmark.items(), key=lambda x: x[1], reverse=True):
        label = "PORTFOLIO" if sym == "portfolio" else sym
        print(f"    {label:12s} {ret:+.2%}")

    # Strategy vs buy-and-hold comparison
    port_bh = benchmark.get("portfolio", 0)
    strat_ret = result.total_return
    alpha = strat_ret - port_bh
    print(f"\n  Strategy Return:   {strat_ret:+.2%}")
    print(f"  Buy & Hold Return: {port_bh:+.2%}")
    print(f"  Alpha:             {alpha:+.2%}")

    # Per-window details
    if result.windows:
        print(f"\n  WINDOW DETAILS ({len(result.windows)} windows):")
        print(f"  {'#':>3s}  {'Test Period':25s}  {'Trades':>6s}  {'Return':>8s}  {'Win%':>6s}")
        print(f"  {'---':>3s}  {'-' * 25}  {'------':>6s}  {'--------':>8s}  {'------':>6s}")
        for w in result.windows:
            wr = w.backtest_result
            period = f"{w.test_start} to {w.test_end}"
            print(
                f"  {w.window_id:3d}  {period:25s}  {wr.total_trades:6d}  "
                f"{wr.total_return:+8.2%}  {wr.win_rate:5.1%}"
            )

    # Equity curve sample
    curve = result.equity_curve()
    if not curve.empty:
        print(f"\n  Equity curve: {len(curve)} data points")
        print(f"  Start: ${args.capital:,.0f}  ->  End: ${curve['equity'].iloc[-1]:,.0f}")

    # Compare exit modes if requested
    if args.compare_exits and data:
        print("\n" + "=" * 60)
        print("  EXIT MODE COMPARISON (Fixed vs Trailing)")
        print("=" * 60)
        from ait.backtesting.engine import Backtester

        # Use the first symbol's data for comparison
        first_sym = list(data.keys())[0]
        comparison = Backtester.compare_exit_modes(
            data[first_sym],
            args.strategies,
            initial_capital=args.capital,
        )
        print(f"\n  {first_sym} results:")
        print(f"  {'Metric':20s}  {'Fixed':>10s}  {'Trailing':>10s}  {'Delta':>10s}")
        print(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

        fr, tr = comparison["fixed"], comparison["trailing"]
        delta = comparison["delta"]
        for metric, fmt in [
            ("total_return", ".2%"),
            ("win_rate", ".2%"),
            ("sharpe_ratio", ".2f"),
            ("max_drawdown", ".2%"),
            ("profit_factor", ".2f"),
        ]:
            fv = getattr(fr, metric)
            tv = getattr(tr, metric)
            dv = delta[metric]
            print(f"  {metric:20s}  {fv:>10{fmt}}  {tv:>10{fmt}}  {dv:>+10{fmt}}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()

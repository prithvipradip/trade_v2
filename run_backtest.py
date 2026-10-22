"""Run walk-forward backtest with real historical data.

Loads the research history (config.yaml backtest.* : the IB research tables, or the legacy
loader when --table-prefix '' is passed) for the configured symbols, then runs the
walk-forward backtester with the requested strategies.

Usage:
    python run_backtest.py
    python run_backtest.py --symbols SPY QQQ AAPL --history-days 500
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
# - delta target:                    iron_condor.SHORT_DELTA_TARGET (W87a; was a literal 0.20)
# - DTE band:                        config.yaml options.dte_range (loaded, NOT the code default)
# - TP ladder:                       src/ait/execution/exit_policy.py CREDIT_TP_LADDER (shared)
# - IV rank floor 15:                AIT_IRON_CONDOR_IV_FLOOR (env contract; --iv-floor follows it)
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
# W87a: the take-profit ladder rows are SAMPLED at one DTE per rung from the shared authority
# (``execution/exit_policy.CREDIT_TP_LADDER``) for BOTH columns; the live column used to be a hand copy.
_TP_LADDER_PROBE_DTES = {"dte>20": 25, "dte_11_20": 15, "dte_6_10": 8, "dte<=5": 3}


def _tp_ladder_by_rung(fn) -> dict:
    return {rung: fn(dte) for rung, dte in _TP_LADDER_PROBE_DTES.items()}


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
    from ait.config.settings import BacktestConfig, DteSearchConfig, MLConfig, OptionsConfig
    from ait.config.runtime_env import contract_float
    from ait.execution import exit_policy
    from ait.strategies.iron_condor import SHORT_DELTA_TARGET

    env = _live_env()
    settings = _live_settings()
    from ait.config.runtime_env import CONTRACT_DEFAULTS as _CD

    def _live_contract(key: str) -> float:
        """The live bot's value of an env-contract key: its applied env, else the contract
        default (W81: these were literal fallbacks that could drift from CONTRACT_DEFAULTS)."""
        return float(env.get(key, _CD[key]))

    rsettings = _research_settings()   # backtest column: base + research overlay
    live_dte = list(settings.options.dte_range) if settings else list(OptionsConfig().dte_range)   # W90a: schema default
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
        "wing_k":            _live_contract("AIT_IC_WING_K"),
        "range_min_confidence": (settings.ml.range_min_confidence
                                  if settings else MLConfig().range_min_confidence),
        "ic_min_credit":     _live_contract("AIT_IC_MIN_CREDIT"),
        "ic_min_credit_width": _live_contract("AIT_IC_MIN_CREDIT_WIDTH"),
        "credit_loss_limit": _live_contract("AIT_CREDIT_LOSS_LIMIT"),
        "pre_event_blackout_days": live_blackout,
        "delta_target":      SHORT_DELTA_TARGET,   # W87a: the constant iron_condor.py selects with
        "dte_band":          live_dte,
        "tp_ladder":         _tp_ladder_by_rung(exit_policy.credit_take_profit_pct),
        "wing_floor":        _live_contract("AIT_IC_WING_FLOOR"),
        "macro_gate_entry":  True,   # orchestrator blocks credit entries within
                                     # risk.pre_event_blackout_days of an event
        "macro_flatten_enabled": env.get("AIT_SKIP_MACRO_EVENTS", "0") == "1",
        "iv_rank_floor":     _live_contract("AIT_IRON_CONDOR_IV_FLOOR"),
        # W86: the three divergent-from-code-default knobs research reads from the loaded config
        # (settings.DIVERGENCE_CLASSES, research_reads_loaded); shown so an overlay that ever sets
        # a different value appears as a warning instead of passing silently.
        "min_confidence": float(settings.risk.min_confidence if settings else _RC().min_confidence),
        "max_open_positions": int(settings.positions.max_open_positions
                                  if settings else _PC().max_open_positions),
        "max_concurrent_positions": int(settings.backtest.max_concurrent_positions
                                        if settings else _BTC().max_concurrent_positions),
        # W88c: the tree hyperparameters live trains with (research may differ via the overlay / a study)
        "ml_hyperparams": (settings.ml if settings else MLConfig()).hyperparams.model_dump(),
        "ml_meta_label_hyperparams": (settings.ml if settings else MLConfig()).meta_label_hyperparams.model_dump(),
        "ml_features": (settings.ml if settings else MLConfig()).features.model_dump(),   # W88d: indicator windows
    }

    # Backtest TP ladder read from the engine mirror itself, so this manifest
    # catches any future drift in Backtester._credit_take_profit_pct.
    bt_ladder = _tp_ladder_by_rung(Backtester._credit_take_profit_pct)
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
        # W87a: the resolved backtest.delta_short (walk-forward does not override it); was a literal 0.20
        "delta_target":      float(_rcv(None, "backtest", "delta_short", _BTC, rsettings)),
        # PR#7 review 2026-08-25: was a stale hardcoded [21, 21] with the note
        # "engine uses fixed DTE = max_hold_days" — the engine's 2026-08-24
        # decoupling resolves entry_dte from options.dte_range[0] instead, so
        # the manifest reported a DTE no run actually priced. Report the same
        # resolution the engine performs.
        "dte_band":          [live_dte[0], live_dte[0]],
        "tp_ladder":         bt_ladder,
        # W81: report the RESOLVED floor (main() resolves it before calling; a direct call with an
        # un-resolved args used to print None here).
        "wing_floor":        (args.wing_floor if args.wing_floor is not None
                              else contract_float("AIT_IC_WING_FLOOR")),
        "macro_gate_entry":  args.macro_gate,
        "macro_flatten_enabled": os.environ.get("AIT_SKIP_MACRO_EVENTS", "0") == "1",
        # W87a: --iv-floor defaults to None (resolved like the live contract); report the RESOLVED value
        "iv_rank_floor":     (args.iv_floor if args.iv_floor is not None
                              else contract_float("AIT_IRON_CONDOR_IV_FLOOR")),
        "min_confidence": float(_rcv(None, "risk", "min_confidence", _RC, rsettings)),
        "max_open_positions": int(_rcv(None, "positions", "max_open_positions", _PC, rsettings)),
        "max_concurrent_positions": int(_rcv(
            None, "backtest", "max_concurrent_positions", _BTC, rsettings)),
        "ml_hyperparams": (rsettings.ml if rsettings else MLConfig()).hyperparams.model_dump(),
        "ml_meta_label_hyperparams": (rsettings.ml if rsettings else MLConfig()).meta_label_hyperparams.model_dump(),
        "ml_features": (rsettings.ml if rsettings else MLConfig()).features.model_dump(),
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
    backtest["pricing_source"] = getattr(_bt_cfg, "pricing_source", "bs_legacy")
    if backtest["pricing_source"] != "surface":
        warnings.append(
            "pricing_source: 'bs_legacy' - leg IVs are one stored 30-day IV plus a skew with no DTE term. "
            "The real-bar surface cut median option-price error out-of-sample by 19% (1-2 DTE), 46% "
            "(3-7), 43% (8-14) and 49% (15-45) on QQQ (W75); set backtest.pricing_source: surface in "
            "config_research.yaml to use it (QQQ only; other symbols fall back per leg).")
    if _engine_model != "realistic":
        warnings.append(
            "engine_model: 'legacy' - entries are priced at the DAILY close with same-day IV, "
            "expiry = entry + N calendar days (can be a weekend), no same-session exit check, "
            "intraday take-profit decided after friction and blind to touches, condors "
            "invalidated on direction (W72 leads L2-L6). 'realistic' fixes them; flip "
            "backtest.engine_model in config_research.yaml after recording a before/after (A10).")

    # W79a: a DTE study deliberately leaves the live band. Say so, so the numbers are never read
    # as "what the live bot would do today".
    _dte_cfg = getattr(_bt_cfg, "dte_search", None)
    _searching = bool(getattr(args, "search_dte", False)) or bool(getattr(_dte_cfg, "enabled", False))
    backtest["dte_search"] = _searching
    if _searching:
        _rng = (list(getattr(args, "dte_range", None) or [])
                or list(getattr(_dte_cfg, "entry_dte", DteSearchConfig().entry_dte)))
        warnings.append(
            f"dte_search: ON - entry DTE is CHOSEN per window in {_rng} (plus the expiry guard and "
            "hold fraction), while live's options.dte_range validator still refuses 0 and the live "
            "selector takes the three nearest expiries (W80 wires both). A result here is a "
            "research finding about the DTE, not what the live bot would do today.")

    # W71: the research rise veto has no live counterpart (live gates on the rank LEVEL only:
    # floor 15, cap 85). It is a declared research-only deviation until the operator decides.
    _rise = getattr(_bt_cfg, "iv_rank_rise_threshold", None)
    _rise_gate = bool(getattr(_bt_cfg, "iv_rank_rise_gate", True))
    backtest["iv_rank_rise_threshold"] = _rise
    backtest["iv_rank_rise_gate"] = _rise_gate
    if _rise_gate and _rise is not None and float(_rise) <= 1.0:
        warnings.append(
            f"iv_rank_rise_threshold: {float(_rise):g} - RESEARCH-ONLY gate: entries are blocked when "
            "the feature-engine iv_rank (a realized-vol range rank) rose by more than this over 10 "
            "rows. Live has no such veto (it gates on the stored-IV percentile LEVEL, floor 15 / cap "
            "85). Set backtest.iv_rank_rise_gate: false in config_research.yaml to turn it off (the "
            "code and the threshold are kept; the bar for re-enabling it is written there); see "
            "scripts/iv_rank_parity_probe.py and the W71 / W71c PLAN entries.")

    # W81: gates that exist on ONE side only (the 2026-10-13 audit's P1-P3). The registry lives
    # in ait.backtesting.gate_parity; a guard test fails if the engine grows a reason without a row.
    from ait.backtesting.gate_parity import LIVE_ONLY_GATES, one_sided_gate_warnings
    warnings.extend(one_sided_gate_warnings(_bt_cfg, settings))
    backtest["regime_veto_gate"] = bool(getattr(_bt_cfg, "regime_veto_gate", True))
    backtest["max_entry_vol_annual"] = getattr(_bt_cfg, "max_entry_vol_annual", None)
    backtest["ex_div_entry_gate"] = getattr(_bt_cfg, "ex_div_entry_gate", None)   # W84
    backtest["live_only_gates_not_simulated"] = [g.reason for g in LIVE_ONLY_GATES]

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
        default=None,
        help="Symbols to backtest. Default: config backtest.default_symbols (QQQ; W87a -- was a "
             "hard-coded ten-name list that crashed on the first name without dividend rows).",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies to test. Default: config options.strategies (the live selector list, "
             "currently iron_condor only; W87a -- was four hard-coded strategies).",
    )
    # W90b: the window defaults are backtest.walkforward_tuning.{train,test,step,gap}_days of the research config
    # (config_research.yaml; resolved in run_backtest() once the overlay/study is applied). W87a had copied the
    # dataclass defaults here; the dataclass fields are now None sentinels over the same setting.
    p.add_argument("--train-days", type=int, default=None,
                   help="Training window days (calendar). Default: backtest.walkforward_tuning.train_days")
    p.add_argument("--test-days", type=int, default=None,
                   help="Test window days. Default: backtest.walkforward_tuning.test_days")
    p.add_argument("--step-days", type=int, default=None,
                   help="Step between windows. Default: backtest.walkforward_tuning.step_days")
    p.add_argument("--gap-days", type=int, default=None,
                   help="Purge gap days. Default: backtest.walkforward_tuning.gap_days")
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
    p.add_argument("--iv-floor", type=float, default=None,
                   help="Min IV rank for iron condors. Default resolves env "
                        "AIT_IRON_CONDOR_IV_FLOOR > config.yaml backtest.ic_iv_rank_floor > 15 "
                        "(contract_float; W87a: was a 15.0 literal).")
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
                   help="Block credit entries within risk.pre_event_blackout_days (config, "
                        "currently 1) of an FOMC/CPI/NFP/GDP/PCE event (research windows use the "
                        "operator's economic calendar; the legacy 2026-only calendar is inactive "
                        "for earlier windows)")
    p.add_argument("--study-file", default=None, metavar="PATH",
                   help="W87b: a studies/<name>.yaml merged over config.yaml + config_research.yaml for THIS run "
                        "(settings: overrides, search_space: overrides). A pre-registered study must name a "
                        "PLAN.md heading in its rule:.")
    p.add_argument("--no-run-record", action="store_true", default=False,
                   help="W87b: do not write reports/runs/<run_id>/run_record.json (or log to MLflow).")
    p.add_argument("--compare-exits", action="store_true", help="Compare fixed vs trailing stops")
    p.add_argument("--optimize-per-window", action="store_true", default=False,
                   help="Run Optuna optimization on each training window before testing")
    p.add_argument("--optimize-n-trials", type=int, default=None,
                   help="Number of Optuna trials per walk-forward window (requires --optimize-per-window). "
                        "Default: backtest.walkforward_tuning.optimize_n_trials")
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
    p.add_argument("--search-dte", action="store_true", default=False,
                   help="W79a: let Optuna choose the entry DTE (and the expiry guard and hold "
                        "fraction) per window. Requires --optimize-per-window. Ranges: "
                        "backtest.dte_search.")
    p.add_argument("--dte-range", type=int, nargs=2, metavar=("LOW", "HIGH"), default=None,
                   help="W79a: inclusive entry-DTE range for --search-dte (default: backtest."
                        "dte_search.entry_dte = 0 45).")
    p.add_argument("--optimize-strict", action="store_true", default=False,
                   help="W79a: abort when a window's optimisation fails instead of silently "
                        "falling back to the base config.")
    p.add_argument("--gates-mode", choices=["observe", "enforce"], default=None,
                   help="W67: observe = live parity (range floor + fractal veto recorded as "
                        "would_reject:*, never block); enforce = they block. Default: "
                        "config backtest.gates_mode.")
    return p.parse_args()


def _write_run_record(args, study, settings, cfg, result, manifest) -> None:
    """W87b: what produced this number -- one JSON in reports/runs/<run_id>/ (+ MLflow when installed).
    Never fails the run."""
    try:
        from ait.backtesting.run_record import finish_run
        from ait.config.research import research_overlay_path, settings_origin
        from ait.optimization.param_spaces import effective_space, space_hash
        spaces = {}
        if cfg.optimize_per_window:
            for strat in args.strategies:
                try:
                    spaces[strat] = effective_space(strat, ((study.search_space if study else {}) or {}).get(strat))
                except ValueError:
                    continue
        metrics = {k: getattr(result, k, None) for k in
                   ("total_return", "total_trades", "win_rate", "sharpe_ratio", "max_drawdown", "profit_factor")}
        run_dir, mlflow_id = finish_run(
            kind="backtest", name=f"{'-'.join(args.symbols).lower()}_{study.name if study else 'adhoc'}",
            settings=settings, config_path="config.yaml", overlay_path=research_overlay_path("config.yaml"),
            study_path=study.path if study else None,
            study=({"name": study.name, "question": study.question, "rule": study.rule,
                    "settings": study.settings, "search_space": study.search_space} if study else None),
            settings_origin=settings_origin("config.yaml", study),
            cli_args={k: v for k, v in vars(args).items()}, symbols=args.symbols,
            db_path=args.db_path, search_space=spaces or None,
            search_space_hash=space_hash(spaces) if spaces else None,
            seeds={"optimize_seed": getattr(cfg, "optimize_seed", None)},
            metrics={k: v for k, v in metrics.items() if v is not None},
            extra={"parity_manifest": manifest})
        print(f"\n  Run record: {run_dir}" + (f"  (MLflow run {mlflow_id})" if mlflow_id else ""))
    except Exception as exc:  # noqa: BLE001 - a record failure must never fail the research run
        print(f"\n  ! run record not written ({type(exc).__name__}: {exc})")


def resolve_walkforward_args(args: argparse.Namespace, tuning) -> None:
    """W90b: fill the window lengths and the per-window trial count the operator did not pass from
    ``backtest.walkforward_tuning`` of the research view (config_research.yaml / a study); an explicit flag wins."""
    for name in ("train_days", "test_days", "step_days", "gap_days"):
        if getattr(args, name) is None:
            setattr(args, name, int(getattr(tuning, name)))
    if args.optimize_n_trials is None:
        args.optimize_n_trials = int(tuning.optimize_n_trials)


async def run_backtest(args: argparse.Namespace) -> None:
    # W68: config.yaml + config_research.yaml is THIS process's configuration (research only;
    # the live bot never loads the overlay). Fails loudly if the overlay file is missing.
    from ait.config.research import (
        assert_registered, load_study, resolve_universe, use_research_config)
    from ait.config.settings import BacktestConfig
    _study = load_study(args.study_file) if args.study_file else None     # W87b: the third config layer
    if _study is not None:
        assert_registered(_study)
    _research_cfg = use_research_config(study=_study)
    args.symbols, args.strategies = resolve_universe(args.symbols, args.strategies, _research_cfg)
    if args.capital is None:
        # W90a: account.capital of the research view (overlay / study aware); was a raw read of config.yaml
        args.capital = float(_research_cfg.account.capital)
    resolve_walkforward_args(args, _research_cfg.backtest.walkforward_tuning)   # W90b
    if args.wing_floor is None:  # W70: the CLI default was a literal 2.0
        from ait.config.runtime_env import contract_float
        # the SAME contract call live's iron_condor.py uses (home backtest.wing_floor_dollars)
        args.wing_floor = contract_float("AIT_IC_WING_FLOOR")
    if args.iv_floor is None:  # W87a: resolved BEFORE the manifest and the env write below read it
        from ait.config.runtime_env import contract_float
        args.iv_floor = contract_float("AIT_IRON_CONDOR_IV_FLOOR")
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
    os.environ["AIT_IRON_CONDOR_IV_FLOOR"] = str(args.iv_floor)  # resolved above (W87a)
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
        vlmc_min_coverage=(_bcfg.vlmc_min_coverage if _bcfg else BacktestConfig().vlmc_min_coverage),   # W90a: schema default
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
        # W79a: DTE search (None = the loaded backtest.dte_search values)
        dte_search=(True if args.search_dte else None),
        dte_range=(tuple(args.dte_range) if args.dte_range else None),
        optimize_strict=(True if args.optimize_strict else None),
        # R6 live-parity knobs
        wing_k=args.wing_k,
        wing_floor_dollars=args.wing_floor,
        credit_loss_limit_mult=args.credit_loss_limit,
        ic_min_credit=args.ic_min_credit,
        ic_min_credit_width=args.ic_min_credit_width,
        macro_event_gate=args.macro_gate,
        search_space_overrides=(_study.search_space or None) if _study is not None else None,   # W87b
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

    # W79a: a failed per-window optimisation falls back to the base config; never let that pass
    # quietly, and report what a DTE search found.
    _failed = getattr(result, "optimization_failures", 0)
    if _failed:
        print(f"\n  !! {_failed} window optimisation(s) FAILED and ran the BASE config "
              "(see window_optimization_failed in the log); this study did not search them.")
    if cfg.dte_search:
        from ait.backtesting.dte_report import format_report, summarise_study
        _by_window = {}
        for _wid, _per_sym in (getattr(result, "optimization_status", None) or {}).items():
            _trials = [t for _st in (_per_sym or {}).values() if isinstance(_st, dict)
                       for t in _st.get("trials", [])]
            if _trials:
                _by_window[_wid] = _trials
        print("\n" + "\n".join(format_report(summarise_study(_by_window))))

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

    if not args.no_run_record:
        _write_run_record(args, _study, _research_cfg, cfg, result, manifest)

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

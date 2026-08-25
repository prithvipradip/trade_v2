"""Backtesting engine for simulating trading strategies on historical data.

Runs the trading loop day-by-day over an OHLCV DataFrame, applying
ML predictions (if available), strategy selection, risk rules,
and trade simulation with Black-Scholes options pricing.

Supports:
- Debit strategies: long_call, long_put, bull_call_spread, bear_put_spread
- Credit strategies: iron_condor, short_strangle (profit from theta decay)
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ait.backtesting.pricing import (
    OptionType,
    black_scholes_price,
    find_strike_by_delta,
    realized_vol,
)
from ait.backtesting.result import BacktestResult
# R20 #5a: credit TP ladder / DTE close / macro-flatten windows come from the
# shared exit-policy module (pure, import-light) instead of a hand copy of
# live portfolio.py — a live tuning can no longer silently de-sync research.
from ait.execution import exit_policy
from ait.strategies.base import CREDIT_STRATEGIES, SignalDirection
from ait.utils.logging import get_logger
from ait.config.runtime_env import contract_flag, contract_float  # R19: ONE authority for env-contract defaults

log = get_logger("backtesting.engine")

# Strategies that collect premium (short theta).
# R16: this module used to declare its OWN set under the same name with
# different membership from strategies/base.py — see the comment there. The
# membership now lives in base.py (live + backtest-only arms) and is imported,
# so the two can never drift apart again. The imported name stays module-level
# here, so existing `from ait.backtesting.engine import CREDIT_STRATEGIES`
# callers keep working unchanged.
# Strategies that pay premium (long theta)
DEBIT_STRATEGIES = {"long_call", "long_put", "bull_call_spread", "bear_put_spread", "long_straddle"}

# R16 #9: direction-neutral credit structures sharing the direction-gate
# bypass and the five neutral entry risk gates (hurst hard veto,
# trending-down regime veto, range-model gate, 10d realized-vol gate,
# IV-rank-rise veto). jade_lizard and call_credit_spread are backtest-only
# shadow arms selectable ONLY on NEUTRAL signals (_select_strategy), so they
# belong to the same entry population as the condor/strangle arms — they were
# previously missing from every tuple, which made those arms either
# zero-trade (blocked by the directional-confidence gate that NEUTRAL@0.4
# can never pass) or, when ML emitted NEUTRAL>=0.55, completely ungated.
NEUTRAL_CREDIT_GATED = (
    "iron_condor", "iron_butterfly", "wide_wing_condor", "broken_wing_condor",
    "short_strangle", "jade_lizard", "call_credit_spread",
)


# R20b review follow-up: this used to be a private copy of the "explicit >
# config > fallback-class-default" precedence, duplicated independently in
# walkforward.py/optimizer.py/the ML predictors/run_backtest.py. Moved to
# ait.config.settings as the ONE shared implementation (also reused by those
# other call sites); re-imported under the original name so every existing
# `_resolve_setting(...)` call in this file needs no change.
from ait.config.settings import resolve_config_value as _resolve_setting


class Backtester:
    """Simulates trading strategies against historical OHLCV data."""

    def __init__(
        self,
        data: pd.DataFrame,
        strategies: list[str],
        # R20b: initial_capital (and max_concurrent_positions below) stay
        # EXPLICIT constructor defaults per the pre-registration — they are
        # test-harness sizing knobs, not strategy economics: every research
        # entry point (walkforward, optimizer, run_backtest) passes its own
        # capital, and a $10k bare-engine default cannot bias a comparison
        # the way a divergent gate threshold can.
        initial_capital: float = 10_000.0,
        commission_per_contract: float = 0.65,
        slippage_pct: float = 0.01,
        position_size_pct: float = 0.05,
        stop_loss_pct: float = 0.50,
        profit_target_pct: float = 1.00,
        max_hold_days: int = 30,
        # R20b (pre-registered PLAN 2026-08-21): was 0.55, silently shadowing
        # the config with a DIFFERENT value. None -> resolve from
        # load_settings().risk.min_confidence (yaml 0.50). See the resolution
        # block below for why risk.min_confidence is the documented home.
        min_confidence: float | None = None,
        trailing_stop_enabled: bool = False,
        trailing_stop_pct: float = 0.25,
        breakeven_trigger_pct: float = 0.30,
        predictor: Any = None,
        range_predictor: Any = None,
        # R20b: was 0.55 vs live's ml.range_min_confidence = 0.65 (config.yaml;
        # 0.65 beat 0.55 across every backtest metric) — the parity gap the
        # floor sweep exposed. None -> resolve from load_settings().ml.
        range_min_confidence: float | None = None,
        context_bars: int = 0,
        delta_short: float = 0.20,
        delta_long: float = 0.30,
        # R20b: was 0.12 vs config.yaml backtest.iv_floor = 0.20 — a bare
        # engine priced systematically thinner credits than every configured
        # path. None -> resolve from load_settings().backtest.iv_floor.
        iv_floor: float | None = None,
        # R6 parity: live iron_condor._vol_scaled_width enforces min $2, not $5.
        wing_floor_dollars: float = 2.0,
        # R16 #8: None -> resolve env AIT_IC_WING_K (default 1.0), mirroring
        # live iron_condor.py:107 — same pattern as ic_min_credit_width below.
        wing_k: float | None = None,
        delta_iv_scale: float = 0.0,
        skew_factor: float = 1.0,
        hurst_regime_threshold: float = 0.20,
        hurst_regime_penalty: float = 0.10,
        hurst_hard_veto_multiplier: float = 1.5,
        multifractal_max_width: float = 0.50,
        iv_rank_rise_threshold: float = 0.30,
        pct_from_60d_high_threshold: float = -1.0,
        min_edge_over_baseline: float = 0.05,
        features_cache: pd.DataFrame | None = None,
        # R20b: stays EXPLICIT (with initial_capital above) per the
        # pre-registration — harness sizing knob; 1 = the original
        # single-position semantics unit tests depend on.
        max_concurrent_positions: int = 1,
        max_entry_vol_annual: float = 0.80,
        # Options bid-ask spread model (per-leg, IV/DTE-aware)
        spread_base: float = 0.03,
        spread_iv_sensitivity: float = 0.10,
        spread_dte_sensitivity: float = 0.005,
        spread_cap: float = 0.15,
        # Cross-asset market context forwarded from walk-forward engine (Gap Z3)
        market_context: dict | None = None,
        # Earnings-aware skip (Gap Z4): if symbol is given, skip entries near earnings dates
        symbol: str | None = None,
        earnings_skip_days: int = 2,
        # Intraday engine (Fix 1): 5-min execution loop.
        # R20 #4/#5b: None -> resolve from load_settings().backtest (config.yaml
        # authority; BacktestConfig defaults when no config is reachable). The
        # engine used to hardcode '09:30' while BacktestConfig declares '10:30'
        # — the documented Fix 1/Gap H parity value that
        # scripts/export_production_params.py reports as production parity —
        # so every intraday-gated study traded a window the config contract
        # says is forbidden, and the exported params lied about it. '10:30'
        # (skip open volatility) is the historically correct value; '09:30'
        # was constructor drift.
        intraday_store: Any = None,
        scan_interval_minutes: int | None = None,
        entry_window_start_et: str | None = None,
        entry_window_end_et: str | None = None,
        limit_order_timeout_bars: int | None = None,
        # Per-window MetaLabeler for OOS signal filtering (Gap Z1)
        meta_labeler: Any = None,
        # H2 val-split: skip new entries before this date (full df still used for feature warmup)
        eval_start_date: "date | None" = None,
        # --- R6 exit/construction parity with the live bot ---
        # Flat loss limit for CREDIT structures as a multiple of credit
        # received (live portfolio.py reads env AIT_CREDIT_LOSS_LIMIT with
        # default "0" = flat stop DISABLED — R12-B1 evidence: touch-close beat
        # every flat level). None -> resolve from env with the SAME default as
        # live (R16 #6 parity: this used to default to 1.25, so every backtest
        # simulated an active stop the live book does not run).
        credit_loss_limit_mult: float | None = None,
        # Iron condor construction gates (live iron_condor.py R6):
        # minimum mid-price total credit (env AIT_IC_MIN_CREDIT, $0.70) and
        # minimum credit/max-width ratio (env AIT_IC_MIN_CREDIT_WIDTH, 0.20).
        ic_min_credit: float | None = None,
        ic_min_credit_width: float | None = None,
        # Macro-event gate: live orchestrator blocks CREDIT entries <=4 days
        # before FOMC/CPI/NFP/GDP/PCE; live portfolio flattens defined-risk
        # credit <=1 day before (strangles <=5) when AIT_SKIP_MACRO_EVENTS=1.
        # Uses the hardcoded-2026 calendar in ait.data.economic_calendar —
        # for pre-2026 backtest windows the gate is effectively inactive
        # (days-to-event is always > 4), which is a documented limitation.
        macro_event_gate: bool = True,
        # R16 #7: live reads pre_event_blackout_days from LOADED settings
        # (orchestrator.py: self._settings.risk.pre_event_blackout_days), so a
        # yaml override must reach the engine too. None -> resolve from
        # load_settings(), falling back to the RiskConfig default when no
        # config.yaml is reachable from the CWD.
        pre_event_blackout_days: int | None = None,
        # R16 #1 (look-ahead fence): when no predictor is supplied, standalone
        # runs may fall back to loading the LIVE models/ensemble.pkl. Walk-
        # forward OOS windows must NEVER do that (the live artifact is trained
        # on the very future periods being scored) — walkforward passes False,
        # and the window then trades UNGATED (predictor=None), which is the
        # engine's designed fallback and what the ablation studies claimed to
        # measure.
        allow_live_model_fallback: bool = True,
        # DTE/hold-cap decoupling fix (2026-08-24): entry_dte is the REAL
        # time-to-expiration used for BS pricing and decay, fixed near what
        # live actually gets (options_chain.py picks the nearest expiry >=
        # dte_range[0]) — it is NOT Optuna-searched. max_hold_days remains a
        # separate, deliberately-synthetic Optuna-searched hold-period cap
        # (live has no "close after N days held" rule; only DTE/touch/TP
        # exits) for exploring shorter/longer forced-hold horizons. Before
        # this fix both concepts were the SAME variable (self._max_hold_days
        # doubled as the option's simulated expiration), so every prior
        # walk-forward study's "optimal max_hold_days" was silently also
        # picking the simulated DTE. None -> resolve options.dte_range[0]
        # from load_settings() (config.yaml authority; falls back to
        # OptionsConfig's pydantic default when no config.yaml is reachable).
        entry_dte: int | None = None,
        # R20b follow-up: optional pre-loaded Settings, so a caller building
        # many Backtester instances (StrategyOptimizer: one per Optuna trial)
        # can load config.yaml ONCE and thread it through instead of every
        # construction re-reading and re-validating the file from disk.
        # None (default) preserves the original behavior — load it here.
        settings: Any = None,
    ) -> None:
        self._data = self._prepare_data(data)
        self._strategies = strategies
        self._initial_capital = initial_capital
        self._commission = commission_per_contract
        self._slippage_pct = slippage_pct
        self._spread_base = spread_base
        self._spread_iv_sensitivity = spread_iv_sensitivity
        self._spread_dte_sensitivity = spread_dte_sensitivity
        self._spread_cap = spread_cap
        # R16: prefer EMPIRICALLY CALIBRATED per-symbol spreads over the
        # config formula. scripts/calibrate_option_spreads.py fits the model
        # to real observed bid/ask quotes and stores it in
        # option_spread_params — a table that existed, was never populated,
        # and (once populated) was still read by nothing. Measured 2026-08-11
        # on 3,159 quotes, config's base=0.04 overstates the real base by
        # 2-13x (SPY 0.012 / QQQ 0.003 / IWM 0.020), so every simulated
        # credit has been charged too much friction. AIT_BT_CALIBRATED_SPREADS=0
        # restores the config formula for A/B runs.
        self._calibrated_spreads = None
        self._position_size_pct = position_size_pct
        self._stop_loss_pct = stop_loss_pct
        self._profit_target_pct = profit_target_pct
        self._max_hold_days = max_hold_days
        # R20b: _min_confidence / _range_min_confidence / _iv_floor are
        # resolved in the loaded-settings block below (they need _settings).
        self._trailing_stop_enabled = trailing_stop_enabled
        self._trailing_stop_pct = trailing_stop_pct
        self._breakeven_trigger_pct = breakeven_trigger_pct
        self._context_bars = context_bars
        self._range_predictor = range_predictor
        self._delta_short = delta_short
        self._delta_long = delta_long
        self._wing_floor_dollars = wing_floor_dollars
        # R16 #8: mirror the live resolution when wing_k is not explicitly
        # configured (R19 contract: env AIT_IC_WING_K > config.yaml
        # backtest.wing_k > CONTRACT_DEFAULTS 1.6 — the comment here used to
        # say "default 1.0", stale since the 2026-08-04 promotion).
        self._wing_k = (
            float(wing_k) if wing_k is not None
            else contract_float("AIT_IC_WING_K")
        )
        self._delta_iv_scale = delta_iv_scale
        self._skew_factor = skew_factor
        self._hurst_regime_threshold = hurst_regime_threshold
        self._hurst_regime_penalty = hurst_regime_penalty
        self._hurst_hard_veto_multiplier = hurst_hard_veto_multiplier
        self._multifractal_max_width = multifractal_max_width
        self._iv_rank_rise_threshold = iv_rank_rise_threshold
        self._pct_from_60d_high_threshold = pct_from_60d_high_threshold
        self._min_edge_over_baseline = min_edge_over_baseline
        self._features_cache = features_cache
        self._max_concurrent_positions = max_concurrent_positions
        self._max_entry_vol_annual = max_entry_vol_annual
        self._market_context = market_context
        self._symbol = symbol or ""
        # R16: must run AFTER _symbol is set — the calibration is per-symbol.
        if os.environ.get("AIT_BT_CALIBRATED_SPREADS", "1") != "0":
            self._load_calibrated_spreads()
        self._earnings_dates: set[date] = self._load_earnings_dates(symbol, earnings_skip_days)
        self._intraday_store = intraday_store
        # R20: resolve LOADED settings once for every config-backed knob below
        # (extends the R16 #7 pre_event_blackout_days pattern).
        # R20b follow-up: reuse a caller-supplied `settings` (e.g.
        # StrategyOptimizer loads it once and threads it through per Optuna
        # trial) instead of unconditionally re-reading + re-validating
        # config.yaml on every Backtester construction.
        if settings is not None:
            _settings = settings
        else:
            try:
                from ait.config.settings import load_settings as _ls
                _settings = _ls()
            except Exception:  # noqa: BLE001 — no config.yaml -> per-field defaults
                _settings = None
        # R20 #4/#5b + R20b (pre-registered PLAN 2026-08-21): every knob below
        # follows the same explicit > settings.<section>.<field> > config-model
        # default precedence (_resolve_setting); explicit constructor args
        # always win, a partial stub or missing config.yaml degrades to that
        # field's pydantic default.
        #   iv_floor             0.12 -> settings.backtest.iv_floor      (yaml 0.20)
        #   range_min_confidence 0.55 -> settings.ml.range_min_confidence (yaml 0.65)
        #   min_confidence       0.55 -> settings.risk.min_confidence     (yaml 0.50)
        # min_confidence's home is risk.min_confidence because the engine
        # consumes it as the DIRECTIONAL-confidence entry gate (run():
        # `confidence < effective_min_conf` skips the entry, with the
        # neutral-credit bypass) — exactly the gate live reads from
        # settings.risk.min_confidence (orchestrator.py `neutral_only =
        # prediction.confidence < min_confidence`), and the production-params
        # exporter already maps "min_confidence" -> ("risk", "min_confidence")
        # (optimization/results.py). It is NOT ml.range_min_confidence (the
        # range-gate floor, a separate knob resolved below).
        from ait.config.settings import (
            BacktestConfig, ExitConfig, MLConfig, OptionsConfig, RiskConfig,
        )
        # entry_dte: real time-to-expiration for BS pricing/decay, fixed near
        # live's actual chain selection (dte_range[0]) — see constructor
        # comment. Not itself Optuna-searched; max_hold_days (below) stays
        # the searched hold-cap knob.
        self._entry_dte = int(entry_dte) if entry_dte is not None else int(
            _resolve_setting(None, "options", "dte_range", OptionsConfig, _settings)[0])
        self._scan_interval_minutes = int(_resolve_setting(
            scan_interval_minutes, "backtest", "scan_interval_minutes", BacktestConfig, _settings))
        self._entry_window_start_et = str(_resolve_setting(
            entry_window_start_et, "backtest", "entry_window_start_et", BacktestConfig, _settings))
        self._entry_window_end_et = str(_resolve_setting(
            entry_window_end_et, "backtest", "entry_window_end_et", BacktestConfig, _settings))
        self._limit_order_timeout_bars = int(_resolve_setting(
            limit_order_timeout_bars, "backtest", "limit_order_timeout_bars", BacktestConfig, _settings))
        self._iv_floor = float(_resolve_setting(
            iv_floor, "backtest", "iv_floor", BacktestConfig, _settings))
        self._range_min_confidence = float(_resolve_setting(
            range_min_confidence, "ml", "range_min_confidence", MLConfig, _settings))
        self._min_confidence = float(_resolve_setting(
            min_confidence, "risk", "min_confidence", RiskConfig, _settings))
        # R20 #5a: live's TP ladder is gated by exit.time_decay_scaling
        # (portfolio.py _get_take_profit_targets); the engine ran the ladder
        # unconditionally, so flipping the flag moved live to flat 0.50
        # targets while research kept the ladder. No constructor override —
        # always follows the loaded/default config, same as before.
        self._exit_time_decay_scaling = bool(_resolve_setting(
            None, "exit", "time_decay_scaling", ExitConfig, _settings))
        self._meta_labeler = meta_labeler
        self._eval_start_date = eval_start_date

        # R6 parity knobs — None resolves to the SAME env var + default the
        # live bot reads, so an un-configured backtest matches live behavior.
        # R16 #6: default "0" = flat credit stop DISABLED, matching live
        # portfolio.py:413 exactly (R12-B1: flat stops fired through their
        # trigger on gaps and every level underperformed touch-close; live
        # runs touch-close-only, which the engine does NOT yet mirror — a
        # documented structural divergence, not a parity value mismatch).
        self._credit_loss_limit_mult = (
            float(credit_loss_limit_mult) if credit_loss_limit_mult is not None
            else contract_float("AIT_CREDIT_LOSS_LIMIT")
        )
        # R16: short-strike touch stop — live's PRIMARY loss exit, mirrored
        # here at last via daily High/Low (they bracket the true intraday
        # path). ON by default = live parity. AIT_BT_TOUCH_STOP=0 restores
        # the old close-only behaviour for before/after comparisons ONLY.
        self._touch_stop_enabled = (
            os.environ.get("AIT_BT_TOUCH_STOP", "1") != "0")
        self._ic_min_credit = (
            float(ic_min_credit) if ic_min_credit is not None
            else contract_float("AIT_IC_MIN_CREDIT")
        )
        self._ic_min_credit_width = (
            float(ic_min_credit_width) if ic_min_credit_width is not None
            else contract_float("AIT_IC_MIN_CREDIT_WIDTH")
        )
        self._macro_event_gate = macro_event_gate
        self._economic_cal = None
        if macro_event_gate:
            try:
                from ait.data.economic_calendar import EconomicCalendar
                self._economic_cal = EconomicCalendar()
            except Exception:  # noqa: BLE001 — gate is best-effort, never fatal
                self._economic_cal = None

        # R16 #7: resolve the macro blackout window ONCE, from the same source
        # live reads (loaded settings), instead of pinning to RiskConfig()
        # defaults on every entry day.
        self._pre_event_blackout_days = int(_resolve_setting(
            pre_event_blackout_days, "risk", "pre_event_blackout_days", RiskConfig, _settings))

        # R16 #1: explicit, loudly-logged predictor mode. Never silently score
        # research windows with the live (future-trained) artifact.
        self._allow_live_model_fallback = bool(allow_live_model_fallback)
        if predictor is not None:
            self._predictor = predictor
            log.info("ml_predictor_mode", mode="explicit_window_model")
        elif self._allow_live_model_fallback:
            self._predictor = self._load_predictor()
            if self._predictor is not None:
                log.warning(
                    "ml_predictor_mode",
                    mode="LIVE_ARTIFACT_FALLBACK",
                    version=getattr(self._predictor, "model_version", "?"),
                    note="scoring with live models/ensemble.pkl — INVALID for "
                         "walk-forward OOS windows (look-ahead leak); pass "
                         "allow_live_model_fallback=False for research runs",
                )
            else:
                log.info("ml_predictor_mode", mode="ungated_no_model_found")
        else:
            self._predictor = None
            log.info(
                "ml_predictor_mode",
                mode="ungated_fallback_disabled",
                note="no window model and live-artifact fallback disabled — "
                     "direction comes from _simple_direction only",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
        capital = self._initial_capital
        trades: list[dict] = []
        open_positions: list[dict] = []

        dates = self._data.index
        if len(dates) < 2:
            log.warning("insufficient_data", rows=len(dates))
            return BacktestResult(
                initial_capital=self._initial_capital,
                final_capital=self._initial_capital,
            )

        log.info(
            "backtest_start",
            start=str(dates[0].date()),
            end=str(dates[-1].date()),
            bars=len(dates),
            strategies=self._strategies,
        )

        lookback = max(20, self._context_bars)

        for i in range(lookback, len(dates)):
            today = dates[i]
            today_date = today.date() if hasattr(today, "date") else today
            hist = self._data.iloc[: i + 1]
            row = self._data.iloc[i]

            # --- 1. Check exits on open positions ---
            still_open = []
            # Load today's intraday bars once for the whole exit loop (Fix 1c)
            _session_for_exit = None
            if self._intraday_store is not None:
                _sym = self._symbol
                if _sym:
                    _session_for_exit = self._intraday_store.load_intraday_range(
                        symbol=_sym, start_date=today_date, end_date=today_date
                    )

            for pos in open_positions:
                # Thesis re-evaluation: exit early if ML direction strongly contradicts entry (Gap Z6).
                thesis_exit = self._check_thesis_invalidation(pos, hist)
                if thesis_exit is not None:
                    underlying = row["Close"]
                    days_held = (today_date - date.fromisoformat(pos["entry_date"])).days
                    current_val = self._reprice_position(pos, underlying, days_held, hist)
                    thesis_exit["pnl"] = round(self._calc_pnl(pos, current_val), 2)
                    thesis_exit["exit_price"] = round(current_val, 4)
                    pos.update(thesis_exit)
                    capital += pos["pnl"]
                    trades.append(pos)
                    log.debug("thesis_invalidated", strategy=pos["strategy"], pnl=f"{pos['pnl']:.2f}")
                    continue

                # Intraday exit check (Fix 1c): scan 5-min bars for stops/targets.
                intraday_exit = None
                if _session_for_exit is not None and not _session_for_exit.empty:
                    intraday_exit = self._check_intraday_exit(pos, _session_for_exit, today_date)

                exit_info = intraday_exit or self._check_exit(pos, row, today_date, hist)
                if exit_info is not None:
                    pos.update(exit_info)
                    capital += pos["pnl"]
                    trades.append(pos)
                    log.debug(
                        "trade_closed",
                        strategy=pos["strategy"],
                        pnl=f"{pos['pnl']:.2f}",
                        reason=pos["exit_reason"],
                    )
                else:
                    still_open.append(pos)
            open_positions = still_open

            # Track current capital for strategy selection
            self._current_capital = capital

            # --- 2. Generate new signal (skip if at position limit) ---
            if len(open_positions) >= self._max_concurrent_positions:
                continue

            # H2 val-split: skip new entries before eval_start_date (exits still processed above)
            if self._eval_start_date is not None and today_date < self._eval_start_date:
                continue

            direction, confidence, features_df = self._get_direction(hist, market_context=self._market_context)

            # Decision-chain dict — populated as each gate is evaluated;
            # attached to the trade record so the dashboard can render the
            # full entry reasoning without re-running the backtest.
            _entry_decision: dict = {
                "direction_class": direction.value if hasattr(direction, "value") else str(direction),
                "direction_conf": round(float(confidence), 4),
                "range_gate": {"prob": None, "threshold": self._range_min_confidence, "pass": None},
                "vol_gate": {"vol_10d": None, "max": self._max_entry_vol_annual, "pass": True},
                "meta_label": {"take": True, "prob": None, "threshold": 0.5},
                "fractal_gate": {"hurst_spread": 0.0, "threshold": self._hurst_regime_threshold, "pass": True},
                "regime": "range_bound",
                "earnings_skip": False,
                "macro_gate": {"days_to_event": None, "blocked": False},
            }

            # Apply fractal regime gate to confidence for credit strategies.
            # Features are already computed inside _get_direction — reuse them.
            if not features_df.empty:
                last_f = features_df.iloc[-1]
                spread = float(last_f.get("hurst_scale_spread", 0.0))
                mf_w   = float(last_f.get("multifractal_width",  0.0))
                _hurst_pass = spread <= self._hurst_regime_threshold or self._hurst_regime_threshold <= 0
                _entry_decision["fractal_gate"] = {
                    "hurst_spread": round(spread, 4),
                    "threshold": self._hurst_regime_threshold,
                    "pass": _hurst_pass,
                }
                if spread > self._hurst_regime_threshold and self._hurst_regime_threshold > 0:
                    # Hard veto is optional: it only applies when multiplier > 0.
                    # Exp 20 post-mortem showed QQQ hurst_spread rarely drops below
                    # ~0.43 in normal conditions; overly tight thresholds can block all entries.
                    _neutral_strat = bool(set(self._strategies) & set(NEUTRAL_CREDIT_GATED))
                    _veto_base = max(self._hurst_regime_threshold, 0.20)
                    _hard_veto_threshold = _veto_base * self._hurst_hard_veto_multiplier
                    if _neutral_strat and self._hurst_hard_veto_multiplier > 0 and spread > _hard_veto_threshold:
                        _entry_decision["fractal_gate"]["hard_veto"] = True
                        log.debug(
                            "hard_veto_fired",
                            component="backtesting.engine",
                            strategies=self._strategies,
                            hurst_spread=round(spread, 4),
                            hard_veto_threshold=round(_hard_veto_threshold, 4),
                            hurst_regime_threshold=round(self._hurst_regime_threshold, 4),
                        )
                        continue
                    penalty = self._hurst_regime_penalty * (
                        spread / self._hurst_regime_threshold
                    )
                    confidence = max(0.0, confidence - penalty)
                if mf_w > 0 and mf_w > self._multifractal_max_width:
                    confidence = max(0.0, confidence - self._hurst_regime_penalty)

            effective_min_conf = self._min_confidence

            # Iron condors and short strangles are market-neutral: high directional
            # confidence signals a trending regime — exactly when they fail. Skip the
            # direction gate for these strategies; the range model below is the sole
            # entry filter. Directional strategies still require confidence ≥ min_conf.
            _neutral_only = bool(set(self._strategies) & set(NEUTRAL_CREDIT_GATED))
            if not _neutral_only:
                if confidence < effective_min_conf:
                    continue
                if direction == SignalDirection.BEARISH and confidence < effective_min_conf + 0.05:
                    continue

            # MetaLabeler gate (Gap Z1): applied during OOS evaluation when a trained
            # per-window model is provided, exactly mirroring the live orchestrator.
            if self._meta_labeler is not None and self._meta_labeler.is_trained:
                meta_ctx: dict = {"primary_confidence": float(confidence)}
                if not features_df.empty:
                    last_f = features_df.iloc[-1]
                    vol_exp = float(last_f.get("vol_regime_expanding", 0.0)) > 0.5
                    px_sma  = float(last_f.get("price_vs_sma_20", 0.0))
                    meta_ctx.update({
                        "regime_trending_up":   1.0 if (vol_exp and px_sma > 0.02)               else 0.0,
                        "regime_trending_down":  1.0 if (vol_exp and px_sma < -0.05)               else 0.0,
                        "regime_high_vol":       1.0 if (vol_exp and -0.05 <= px_sma <= 0.02)      else 0.0,
                        "regime_range_bound":    0.0 if vol_exp else 1.0,
                        "vix":                   float(last_f.get("vix_level", 0.5)),
                        "iv_rank":               float(last_f.get("iv_rank", 0.5)),
                        "rsi_14":                float(last_f.get("rsi_14", 50.0)),
                        "rsi_7":                 float(last_f.get("rsi_7",  50.0)),
                        "bb_position":           float(last_f.get("bb_position", 0.5)),
                        "volume_sma_20_ratio":   float(last_f.get("volume_sma_20_ratio", 1.0)),
                        "realized_vol_20":       float(last_f.get("realized_vol_20", 0.20)),
                        "atr_pct":               float(last_f.get("atr_pct", 0.01)),
                        "weekly_trend_aligned":  float(last_f.get("weekly_trend_aligned", 0.5)),
                        "volume_confirmation":   float(last_f.get("volume_confirmation", 0.0)),
                        "macd_hist":             float(last_f.get("macd_hist", 0.0)),
                        "price_vs_sma_20":       px_sma,
                        "sma_10_20_cross":       float(last_f.get("sma_10_20_cross", 0.5)),
                    })
                meta_ctx.setdefault("sentiment_score", 0.0)
                meta_ctx.setdefault("hour_of_day", 10)
                try:
                    meta_signal = self._meta_labeler.predict(meta_ctx)
                    if meta_signal is not None:
                        _entry_decision["meta_label"] = {
                            "take": bool(meta_signal.take_trade),
                            "prob": round(float(getattr(meta_signal, "probability", 0.5)), 4),
                            "threshold": 0.5,
                        }
                        if not meta_signal.take_trade:
                            continue
                except Exception:
                    pass  # meta-labeler errors are non-fatal

            strategy = self._select_strategy(direction, hist, confidence, features_df)
            if strategy is None:
                continue

            # Earnings proximity skip (Gap Z4): matches live orchestrator behaviour.
            if today_date in self._earnings_dates:
                log.debug("earnings_skip", date=str(today_date), strategy=strategy)
                continue

            # Macro-event entry gate (live orchestrator credit-block parity).
            # R16 #7: the blackout window is resolved once in __init__ from
            # the LOADED settings (constructor override > load_settings() >
            # RiskConfig default) — the same source live reads — so a yaml
            # override no longer changes live only.
            # LIMITATION: the calendar is hardcoded 2026 + 2027-H1, so for
            # pre-2026 windows days-to-event is always > window and the gate
            # never fires (documented in the parity manifest).
            if self._economic_cal is not None and strategy in CREDIT_STRATEGIES:
                _blackout = self._pre_event_blackout_days
                try:
                    _d2e = self._economic_cal.days_until_next_event(today_date)
                except Exception:  # noqa: BLE001
                    _d2e = None
                _entry_decision["macro_gate"] = {
                    "days_to_event": _d2e,
                    "blocked": _d2e is not None and _d2e <= _blackout,
                }
                if _d2e is not None and _d2e <= _blackout:
                    log.debug(
                        "macro_event_entry_skip",
                        date=str(today_date),
                        strategy=strategy,
                        days_to_event=_d2e,
                    )
                    continue

            # Regime gate: iron condors / short strangles fail in trending-down regimes.
            # Exp 21 post-mortem: 4 trending_down trades, 25% win rate, avg PnL -$195.
            # Both macro dislocations in the dataset (Yen carry unwind Aug-2024,
            # tariff shock Mar-2026) occurred in this regime. Trending_up (100% win)
            # and high_volatility (62% win) maintain positive EV and are not blocked.
            # Exp 22: threshold -0.02 was too loose — blocked profitable 2-4% corrections
            # in W02, causing Optuna to adapt badly. Both structural failures cleared -0.05
            # (Yen carry -6-8%, tariff shock -8-10%); raised to -0.05 for Exp 23.
            if strategy in NEUTRAL_CREDIT_GATED and not features_df.empty:
                _last_f        = features_df.iloc[-1]
                _vol_exp_flag  = float(_last_f.get("vol_regime_expanding", 0.0)) > 0.5
                _px_sma_val    = float(_last_f.get("price_vs_sma_20", 0.0))
                _regime_class  = (
                    "trending_down"   if (_vol_exp_flag and _px_sma_val < -0.05) else
                    "trending_up"     if (_vol_exp_flag and _px_sma_val > 0.02)  else
                    "high_volatility" if  _vol_exp_flag                          else
                    "range_bound"
                )
                _entry_decision["regime_class"] = _regime_class
                if _regime_class == "trending_down":
                    _entry_decision["regime_veto"] = True
                    log.debug(
                        "regime_veto_fired",
                        component="backtesting.engine",
                        strategy=strategy,
                        regime=_regime_class,
                        vol_regime_expanding=round(float(_last_f.get("vol_regime_expanding", 0.0)), 4),
                        price_vs_sma_20=round(_px_sma_val, 4),
                    )
                    continue

            # Range model gate: for neutral credit structures, replace
            # confidence with P(stays in range). Skip if below range threshold.
            if strategy in NEUTRAL_CREDIT_GATED:
                if self._range_predictor is None:
                    # R16 #3: honor the walkforward "range model absent ->
                    # block entries" contract. walkforward raises
                    # range_min_confidence to 1.0 when window range training
                    # FAILED; with no predictor no probability can ever reach
                    # it, so the entry must be blocked HERE. Previously the
                    # whole gate was skipped when the predictor was None and
                    # entries flowed ungated — every recent study (ablation,
                    # wingk, shadow rounds) ran with no range gate at all.
                    if self._range_min_confidence >= 1.0:
                        _entry_decision["range_gate"] = {
                            "prob": None,
                            "threshold": self._range_min_confidence,
                            "pass": False,
                        }
                        log.debug(
                            "range_gate_blocked_no_predictor",
                            date=str(today_date),
                            strategy=strategy,
                            threshold=self._range_min_confidence,
                        )
                        continue
                else:
                    try:
                        rp = self._range_predictor.predict(
                            hist,
                            symbol=self._symbol,
                            market_context=self._market_context,
                            min_edge_override=self._min_edge_over_baseline,
                        )
                        if rp is None or rp.probability_in_range < self._range_min_confidence:
                            continue  # bad range setup → skip
                        _entry_decision["range_gate"] = {
                            "prob": round(float(rp.probability_in_range), 4),
                            "threshold": self._range_min_confidence,
                            "pass": True,
                        }
                        confidence = rp.probability_in_range
                    except Exception:
                        pass

            # Realized-vol entry gate: neutral credit structures cannot profit
            # during high-volatility regimes (e.g. tariff shocks, VIX > 40).
            if strategy in NEUTRAL_CREDIT_GATED:
                recent_close = hist["Close"].iloc[-11:]
                if len(recent_close) >= 11:
                    vol_10d = recent_close.pct_change().std() * (252 ** 0.5)
                    _entry_decision["vol_gate"] = {
                        "vol_10d": round(float(vol_10d), 4),
                        "max": self._max_entry_vol_annual,
                        "pass": vol_10d <= self._max_entry_vol_annual,
                    }
                    if vol_10d > self._max_entry_vol_annual:
                        log.debug(
                            "vol_gate_skip",
                            strategy=strategy,
                            vol_10d=f"{vol_10d:.2%}",
                            max_vol=f"{self._max_entry_vol_annual:.2%}",
                        )
                        continue

            # R12-C: AEKF (OU-Kou-GARCH) direction veto removed — the GARCH
            # family moved to deprecated/research/ and RangePredictor no longer
            # produces ou_jump_state, so the veto could never fire again.

            # Rising IV rank filter: if IV rank has risen by more than iv_rank_rise_threshold
            # over the last 10 days, market is in directional stress — skip iron condor entry.
            # Rise value is always logged (not just on veto) for threshold tuning.
            if strategy in NEUTRAL_CREDIT_GATED and not features_df.empty:
                if "iv_rank" in features_df.columns and len(features_df) >= 11:
                    iv_rank_series = features_df["iv_rank"].iloc[-11:]
                    iv_rank_rise = float(iv_rank_series.iloc[-1]) - float(iv_rank_series.iloc[0])
                    _entry_decision["iv_rank_rise_10d"] = round(iv_rank_rise, 4)
                    if iv_rank_rise > self._iv_rank_rise_threshold:
                        _entry_decision["iv_rank_veto"] = {
                            "rise": round(iv_rank_rise, 4),
                            "threshold": self._iv_rank_rise_threshold,
                        }
                        log.debug(
                            "iv_rank_veto_fired",
                            component="backtesting.engine",
                            strategy=strategy,
                            iv_rank_rise=round(iv_rank_rise, 4),
                            threshold=self._iv_rank_rise_threshold,
                        )
                        continue

            # Drawdown-from-60d-high gate: block iron condor entry when the underlying is
            # in a sustained decline relative to its 60-day rolling high.  Catches slow-grind
            # bear phases that iv_rank_rise misses (W12 tariff shock: rise=0.021 but -16% from high).
            # Default threshold -1.0 disables the gate; Optuna tunes per window in [-0.15, -0.05].
            # R16: this was the ONE neutral-credit gate the string-patched
            # builder rollout missed — every sibling gate above already reads
            # NEUTRAL_CREDIT_GATED, so an activated threshold gave the
            # iron_condor baseline a drawdown veto the butterfly/wide/broken/
            # jade/ccs arms silently skipped, biasing any promotion study.
            if strategy in NEUTRAL_CREDIT_GATED and self._pct_from_60d_high_threshold > -0.99:
                _close_series = hist["Close"]
                _rolling_high = _close_series.rolling(min(60, len(_close_series))).max().iloc[-1]
                if _rolling_high and _rolling_high > 0:
                    _pct_from_high = (float(_close_series.iloc[-1]) - float(_rolling_high)) / float(_rolling_high)
                    _entry_decision["pct_from_60d_high"] = round(_pct_from_high, 4)
                    if _pct_from_high < self._pct_from_60d_high_threshold:
                        _entry_decision["drawdown_veto"] = {
                            "pct_from_high": round(_pct_from_high, 4),
                            "threshold": self._pct_from_60d_high_threshold,
                        }
                        log.debug(
                            "drawdown_veto_fired",
                            component="backtesting.engine",
                            strategy=strategy,
                            pct_from_high=round(_pct_from_high, 4),
                            threshold=self._pct_from_60d_high_threshold,
                        )
                        continue

            # --- 3. Build the trade ---
            # Intraday entry window gate (Fix 1g / Gap D): when 5-min data is available,
            # only enter during the configured ET window and simulate a limit-order fill.
            entry_time_str: str | None = None
            limit_price: float | None = None
            fill_time_str: str | None = None
            if self._intraday_store is not None:
                from ait.data.historical import HistoricalDataStore
                session_bars = self._intraday_store.load_intraday_range(
                    symbol=self._symbol,
                    start_date=today_date,
                    end_date=today_date,
                )
                if not session_bars.empty:
                    # Find first bar in the entry window
                    window_bars = session_bars[
                        session_bars.index.to_series().apply(
                            lambda ts: self._is_in_entry_window(ts)
                        )
                    ]
                    if window_bars.empty:
                        continue  # No bars in window today — skip entry

                    # Use VWAP of first scan bar as limit price (mid-market proxy)
                    first_scan_bar = window_bars.iloc[0]
                    scan_time = window_bars.index[0]
                    entry_time_str = scan_time.isoformat()

                    # Build partial hist up to scan time for feature computation
                    hist_partial = HistoricalDataStore.slice_intraday_up_to(
                        session_bars, scan_time.time()
                    )
                    if hist_partial.empty:
                        limit_price = float(first_scan_bar["Close"])
                    else:
                        limit_price = float(hist_partial["Close"].iloc[-1])

                    # Try to fill limit order on subsequent bars
                    bars_after_scan = window_bars.iloc[1:]
                    filled, bars_waited, fill_time_str = self._try_limit_fill(
                        limit_price, bars_after_scan, self._limit_order_timeout_bars
                    )
                    if not filled:
                        continue  # Limit order expired without fill — skip entry

            # R20 #1: run() has had self._market_context all along (walkforward
            # passes the window-aligned VIX/SPY frames) but never forwarded it
            # here, so _get_iv's priority-2 VIX branch and the R16 per-symbol
            # vol calibration (VXN/VIX 1.228 QQQ, 1.33 IWM) were DEAD at
            # entry-pricing time — every study priced entries off the
            # priority-3 synthetic fallback realized_vol*1.15.
            pos = self._build_position(
                strategy, direction, row, hist, today_date, capital,
                market_context=self._market_context,
            )
            if pos is None:
                continue

            if entry_time_str:
                pos["entry_time"] = entry_time_str
            if limit_price is not None:
                pos["limit_price"] = round(float(limit_price), 4)
            if fill_time_str is not None:
                pos["fill_time"] = fill_time_str

            # Store signal context for MetaLabeler training (Gap Z9)
            pos["entry_confidence"] = round(float(confidence), 4)
            pos["entry_direction"] = direction.value if hasattr(direction, "value") else str(direction)
            if not features_df.empty:
                last_f = features_df.iloc[-1]
                pos["entry_iv_rank"] = round(float(last_f.get("iv_rank", 0.0)), 4)
                pos["entry_vix_level"] = round(float(last_f.get("vix_level", 0.0)), 4)
                vol_expanding = float(last_f.get("vol_regime_expanding", 0.0)) > 0.5
                px_vs_sma = float(last_f.get("price_vs_sma_20", 0.0))
                if vol_expanding:
                    if px_vs_sma > 0.02:
                        pos["entry_regime"] = "trending_up"
                    elif px_vs_sma < -0.05:
                        pos["entry_regime"] = "trending_down"
                    else:
                        pos["entry_regime"] = "high_volatility"
                else:
                    pos["entry_regime"] = "range_bound"
                # Features snapshot for dashboard decision drawer
                pos["features_at_entry"] = {
                    "rsi_14":              round(float(last_f.get("rsi_14", 0.0)), 2),
                    "macd_hist":           round(float(last_f.get("macd_hist", 0.0)), 5),
                    "bb_position":         round(float(last_f.get("bb_position", 0.0)), 3),
                    "atr_pct":             round(float(last_f.get("atr_pct", 0.0)), 4),
                    "realized_vol_20":     round(float(last_f.get("realized_vol_20", 0.0)), 4),
                    "iv_rank":             round(float(last_f.get("iv_rank", 0.0)), 3),
                    "vix_level":           round(float(last_f.get("vix_level", 0.0)), 2),
                    "hurst_wavelet":       round(float(last_f.get("hurst_wavelet", 0.0)), 3),
                    # R12-C: sentiment_composite / put_call_ratio dropped with
                    # the sentiment/flow feature retirement (constant columns).
                }
            else:
                pos["entry_regime"] = "range_bound"  # default when history too short
                pos["features_at_entry"] = {}

            # Finalize decision chain with resolved regime
            _entry_decision["direction_conf"] = round(float(confidence), 4)
            _entry_decision["regime"] = pos.get("entry_regime", "range_bound")
            pos["decision"] = _entry_decision

            # Iron condor leg structure for the dashboard drawer
            if pos.get("strategy") in ("iron_condor", "iron_butterfly", "wide_wing_condor", "broken_wing_condor"):
                ep = pos.get("entry_price", 0.0)  # net credit per share
                # Each spread contributes ep/2 to the net credit.
                # Short leg = ep/2 + wing_cost; long leg = -wing_cost (debit).
                # Using wing_cost = ep*0.05 so net = 2*(0.55-0.05)*ep = ep.
                pos["legs"] = [
                    {"type": "short_put",  "strike": pos.get("short_put_strike"),  "premium": round(ep * 0.55, 4)},
                    {"type": "long_put",   "strike": pos.get("long_put_strike"),   "premium": round(-ep * 0.05, 4)},
                    {"type": "short_call", "strike": pos.get("short_call_strike"), "premium": round(ep * 0.55, 4)},
                    {"type": "long_call",  "strike": pos.get("long_call_strike"),  "premium": round(-ep * 0.05, 4)},
                ]
                contracts = pos.get("contracts", 1)
                pos["credit"]   = round(ep * 100 * contracts, 2)
                pos["max_loss"] = round(pos.get("max_loss_per_share", 0.0) * 100 * contracts, 2)
            elif pos.get("strategy") in ("put_credit_spread", "call_credit_spread"):
                ep = pos.get("entry_price", 0.0)
                pos["legs"] = [
                    {"type": "short", "strike": pos.get("short_put_strike") or pos.get("short_call_strike"), "premium": round(ep * 1.05, 4)},
                    {"type": "long",  "strike": pos.get("long_put_strike")  or pos.get("long_call_strike"),  "premium": round(-ep * 0.05, 4)},
                ]
                contracts = pos.get("contracts", 1)
                pos["credit"]   = round(ep * 100 * contracts, 2)
                pos["max_loss"] = round(pos.get("max_loss_per_share", 0.0) * 100 * contracts, 2)
            elif pos.get("strategy") == "jade_lizard":
                # R16: jade_lizard fell to the else below, so it got legs=[]
                # and NO credit/max_loss keys — result.py:194 then fell back to
                # `abs(pnl)*2` as the risk proxy, computing the arm's
                # capital_utilization / cash_drag_adjusted_return off a P&L
                # multiple instead of margin, and rendering an empty leg drawer.
                # 3 legs: short put + short call + long call wing. Premium split
                # follows the sibling branches' credit-share convention.
                ep = pos.get("entry_price", 0.0)
                pos["legs"] = [
                    {"type": "short_put",  "strike": pos.get("short_put_strike"),  "premium": round(ep * 0.55, 4)},
                    {"type": "short_call", "strike": pos.get("short_call_strike"), "premium": round(ep * 0.50, 4)},
                    {"type": "long_call",  "strike": pos.get("long_call_strike"),  "premium": round(-ep * 0.05, 4)},
                ]
                contracts = pos.get("contracts", 1)
                pos["credit"]   = round(ep * 100 * contracts, 2)
                # Undefined-risk put side: max_loss_per_share is the builder's
                # (short_put_strike - credit) stress figure, NOT a structural cap.
                pos["max_loss"] = round(pos.get("max_loss_per_share", 0.0) * 100 * contracts, 2)
            else:
                pos["legs"] = []

            # Deep-audit BT-M6: entry commission is included in _calc_pnl
            # (subtracted from the trade P&L added back at exit) — debiting
            # capital here as well double-counted it in final_capital.
            n_legs = pos.get("n_legs", 1)
            entry_commission = self._commission * pos["contracts"] * n_legs
            pos["entry_commission"] = entry_commission

            open_positions.append(pos)

        # --- Force-close remaining positions ---
        last_row = self._data.iloc[-1]
        last_date = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]
        for pos in open_positions:
            exit_info = self._force_close(pos, last_row, last_date, self._data)
            pos.update(exit_info)
            capital += pos["pnl"]
            trades.append(pos)

        result = BacktestResult(
            trades=trades,
            initial_capital=self._initial_capital,
            final_capital=round(capital, 2),
            start_date=dates[0].date() if hasattr(dates[0], "date") else dates[0],
            end_date=last_date,
        )

        log.info(
            "backtest_complete",
            total_trades=result.total_trades,
            total_return=f"{result.total_return:.2%}",
            sharpe=f"{result.sharpe_ratio:.2f}",
            max_dd=f"{result.max_drawdown:.2%}",
            win_rate=f"{result.win_rate:.2%}",
        )
        return result

    @classmethod
    def compare_exit_modes(
        cls, data: pd.DataFrame, strategies: list[str], **kwargs: Any
    ) -> dict:
        """Run backtest with both fixed and trailing stops, return comparison.

        R6 parity note: credit structures always use the flat-loss-limit +
        DTE-laddered TP path, so fixed-vs-trailing deltas reflect DEBIT
        trades only.
        """
        shared = {k: v for k, v in kwargs.items() if k != "trailing_stop_enabled"}

        fixed_bt = cls(data, strategies, trailing_stop_enabled=False, **shared)
        trailing_bt = cls(data, strategies, trailing_stop_enabled=True, **shared)

        fixed_result = fixed_bt.run()
        fixed_result.exit_mode = "fixed"
        trailing_result = trailing_bt.run()
        trailing_result.exit_mode = "trailing"

        return {
            "fixed": fixed_result,
            "trailing": trailing_result,
            "delta": {
                "total_return": trailing_result.total_return - fixed_result.total_return,
                "win_rate": trailing_result.win_rate - fixed_result.win_rate,
                "sharpe_ratio": trailing_result.sharpe_ratio - fixed_result.sharpe_ratio,
                "max_drawdown": trailing_result.max_drawdown - fixed_result.max_drawdown,
                "profit_factor": trailing_result.profit_factor - fixed_result.profit_factor,
            },
        }

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        """Ensure data has a DatetimeIndex and required columns."""
        df = data.copy()
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("open", "high", "low", "close", "volume"):
                col_map[col] = lower.capitalize()
        if col_map:
            df = df.rename(columns=col_map)

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.set_index("Date")
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                df.index = pd.to_datetime(df.index)

        df = df.sort_index()
        return df

    # ------------------------------------------------------------------
    # Direction prediction
    # ------------------------------------------------------------------

    def _load_predictor(self) -> Any:
        """Try to load trained DirectionPredictor."""
        try:
            from ait.config.settings import MLConfig
            from ait.ml.ensemble import DirectionPredictor
            predictor = DirectionPredictor(MLConfig())
            if predictor.load_models():
                log.info("ml_predictor_loaded", version=predictor.model_version)
                return predictor
        except Exception as e:
            log.debug("ml_predictor_unavailable", reason=str(e))
        return None

    @staticmethod
    def _load_earnings_dates(symbol: str | None, skip_days: int) -> set[date]:
        """Pre-fetch historical earnings dates from yfinance (Gap Z4).

        Returns the set of all calendar dates within skip_days of any earnings
        announcement, so the main loop can do an O(1) membership test.
        Returns an empty set if symbol is None or yfinance is unavailable.
        """
        if not symbol or skip_days <= 0:
            return set()
        try:
            import yfinance as yf
            from datetime import timedelta
            ticker = yf.Ticker(symbol)
            eds = getattr(ticker, "earnings_dates", None)
            if eds is None or (hasattr(eds, "empty") and eds.empty):
                return set()
            danger_dates: set[date] = set()
            for idx_val in eds.index:
                ed = idx_val.date() if hasattr(idx_val, "date") else idx_val
                for delta in range(-1, skip_days + 1):  # 1 day after through skip_days before
                    danger_dates.add(ed + timedelta(days=delta))
            log.debug("earnings_dates_loaded", symbol=symbol, count=len(danger_dates))
            return danger_dates
        except Exception as e:
            log.debug("earnings_dates_load_failed", symbol=symbol, error=str(e))
            return set()

    def _parse_et_time(self, time_str: str):
        """Parse an HH:MM Eastern Time string into a datetime.time object."""
        from datetime import time as dt_time
        h, m = time_str.split(":")
        return dt_time(int(h), int(m))

    def _is_in_entry_window(self, bar_time) -> bool:
        """Return True if bar_time is within the ET entry window.

        Accepts a datetime.time (assumed already ET) or a timestamp. R5 audit
        C4: intraday bars are stored tz-aware UTC, but this compared the RAW
        clock time against ET strings — only 13:30-15:30 *UTC* bars passed,
        so every backtest entered solely in the first ~2h of the session
        (DST-shifting), regardless of the configured window. Convert
        tz-aware timestamps to America/New_York first.
        """
        from datetime import time as dt_time
        start = self._parse_et_time(self._entry_window_start_et)
        end = self._parse_et_time(self._entry_window_end_et)
        if isinstance(bar_time, dt_time):
            t = bar_time
        else:
            ts = bar_time
            if getattr(ts, "tzinfo", None) is not None:
                from zoneinfo import ZoneInfo
                ts = ts.tz_convert("America/New_York") if hasattr(ts, "tz_convert") \
                    else ts.astimezone(ZoneInfo("America/New_York"))
            t = ts.time()
        return start <= t < end

    def _check_intraday_exit(
        self, pos: dict, session_bars: "pd.DataFrame", current_date: date
    ) -> dict | None:
        """Check intraday 5-min bars for stop-loss or profit-target triggers.

        Returns exit dict if triggered on any bar, else None.
        """
        if session_bars is None or session_bars.empty:
            return None

        entry_price = pos["entry_price"]
        trade_type = pos.get("trade_type", pos.get("position_type", "debit"))
        expiry_str = pos.get("expiry_date") or pos.get("exit_date")
        if not expiry_str:
            return None
        expiry = date.fromisoformat(str(expiry_str)[:10])

        days_held = (current_date - date.fromisoformat(pos["entry_date"])).days
        remaining_dte = max(0, (expiry - current_date).days)
        exit_half_spread = self._options_half_spread(float(pos.get("entry_iv", 0.25)), remaining_dte)

        for row in session_bars.itertuples(index=True):
            underlying = row.Close
            current_val = self._reprice_position(pos, underlying, days_held, None)
            if trade_type == "credit":
                current_val *= (1 + exit_half_spread)
                pnl_pct = (entry_price - current_val) / entry_price if entry_price > 0 else 0.0
            else:
                current_val *= (1 - exit_half_spread)
                pnl_pct = (current_val - entry_price) / entry_price if entry_price > 0 else 0.0

            # Check stop / profit — routed per trade type (R6 exit parity)
            result = self._dispatch_exit_check(pos, trade_type, pnl_pct, current_date)

            if result is not None:
                bar_ts = row.Index
                bar_dt = bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts)
                pnl = self._calc_pnl(pos, current_val)
                result["pnl"] = round(pnl, 2)
                result["exit_price"] = round(current_val, 4)
                result["exit_underlying"] = round(float(underlying), 4)
                result["exit_time"] = bar_dt
                return result

        return None

    def _try_limit_fill(
        self, limit_price: float, session_bars: "pd.DataFrame", timeout_bars: int
    ) -> "tuple[bool, int, str | None]":
        """Simulate limit order fill on subsequent 5-min bars.

        Returns (filled, bars_waited, fill_time_iso). fill_time_iso is the ISO
        timestamp of the bar where Low ≤ limit_price ≤ High, or None if unfilled.
        """
        bars = session_bars.iloc[:timeout_bars]
        mask = (bars["Low"] <= limit_price) & (limit_price <= bars["High"])
        if mask.any():
            first = int(mask.values.argmax())
            ts = bars.index[first]
            fill_ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
            return True, first + 1, fill_ts
        return False, min(len(session_bars), timeout_bars), None

    # R12-C: _load_directional_model removed — no callers; DirectionalModel
    # retired to deprecated/src/directional.py.

    def _get_direction(
        self,
        hist: pd.DataFrame,
        market_context: dict | None = None,
    ) -> tuple[SignalDirection, float, pd.DataFrame]:
        """Get market direction prediction and pre-computed feature matrix.

        Returns (direction, confidence, features_df). The features_df is shared
        with the fractal gate and _select_strategy to avoid double computation.
        market_context (VIX, SPY, macro) is forwarded to the predictor and
        range predictor so they use cross-asset features, matching live inference (Gap Z3).
        """
        from ait.ml.features import FeatureEngine
        _cache_hit = False
        if self._features_cache is not None and not self._features_cache.empty:
            today = pd.Timestamp(hist.index[-1]).normalize()
            mask = self._features_cache.index <= today
            features_df = self._features_cache[mask]
            if features_df.empty:
                features_df = FeatureEngine().compute(hist)
            else:
                _cache_hit = True
        else:
            features_df = FeatureEngine().compute(hist)

        if self._predictor is not None:
            try:
                if _cache_hit and not features_df.empty and hasattr(self._predictor, "predict_from_features"):
                    # Cache available — predict_from_features bypasses the FeatureEngine
                    # re-run inside predict(), reducing OOS cost from O(N) to O(1) per bar.
                    pred = self._predictor.predict_from_features(
                        features_df.iloc[-1], symbol=self._symbol or ""
                    )
                else:
                    pred = self._predictor.predict(hist, market_context=market_context)
                if pred is not None:
                    return pred.direction, pred.confidence, features_df
            except Exception:
                pass
        direction, confidence = self._simple_direction(hist)
        return direction, confidence, features_df

    @staticmethod
    def _quick_rsi(close: np.ndarray, period: int = 14) -> float:
        """Fast RSI calculation from close prices array."""
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    @staticmethod
    def _simple_direction(hist: pd.DataFrame) -> tuple[SignalDirection, float]:
        """Estimate direction from recent returns and trend."""
        close = hist["Close"]
        ret_5 = close.iloc[-1] / close.iloc[-5] - 1 if len(close) >= 5 else 0.0
        ret_20 = close.iloc[-1] / close.iloc[-20] - 1 if len(close) >= 20 else 0.0
        score = 0.6 * ret_5 + 0.4 * ret_20

        threshold = 0.005
        if score > threshold:
            return SignalDirection.BULLISH, min(0.5 + abs(score) * 10, 0.95)
        elif score < -threshold:
            return SignalDirection.BEARISH, min(0.5 + abs(score) * 10, 0.95)
        return SignalDirection.NEUTRAL, 0.4

    # ------------------------------------------------------------------
    # Strategy selection — now IV-aware
    # ------------------------------------------------------------------

    def _select_strategy(
        self,
        direction: SignalDirection,
        hist: pd.DataFrame,
        confidence: float = 0.65,
        features_df: pd.DataFrame | None = None,
    ) -> str | None:
        """Pick a strategy based on direction, IV regime, and available strategies.

        High IV → prefer credit strategies (sell expensive premium)
        Low IV → prefer debit strategies (buy cheap premium)
        """
        available = set(self._strategies)
        # R20 #1: a bare, context-free IV-estimation call sat here with its
        # result UNUSED — dead code that invited a future reader to assume
        # selection was VIX-aware. Removed rather than context-threaded: the
        # regime proxy below is realized-vol based.

        # IV rank proxy: compare current IV to its range
        close_arr = hist["Close"].values
        rv_short = realized_vol(close_arr, window=10)
        rv_long = realized_vol(close_arr, window=60) if len(close_arr) > 61 else rv_short
        iv_regime_high = rv_short > rv_long * 1.1  # Short-term vol elevated

        # BT-M8: iron_condor used to be FORCED whenever present, silently
        # never trading the other requested strategies (and misleading every
        # multi-strategy breakdown). Prefer it on NEUTRAL, but respect the
        # requested strategy set for directional signals.
        if direction == SignalDirection.NEUTRAL:
            candidates = available & CREDIT_STRATEGIES
        elif direction == SignalDirection.BULLISH:
            candidates = available & {"bull_call_spread", "long_call"}
        else:
            candidates = available & {"bear_put_spread", "long_put"}

        if not candidates:
            return None
        if "iron_condor" in candidates:
            return "iron_condor"  # preferred among the eligible set
        return sorted(candidates)[0]

    # ------------------------------------------------------------------
    # Position building — full BS pricing for each strategy type
    # ------------------------------------------------------------------

    def _get_leg_iv(self, base_iv: float, strike: float, underlying: float,
                    option_type: OptionType) -> float:
        """Apply vol skew: OTM puts carry extra IV, OTM calls carry mild extra IV.

        Linear model in log-moneyness space:
        - Put: +1% IV per 10% below ATM (e.g. 20% OTM put → +2% IV)
        - Call: +0.2% IV per 10% above ATM
        skew_factor=0.0 restores flat IV; default 1.0 approximates market skew.
        """
        log_m = np.log(strike / underlying)
        if option_type == OptionType.PUT:
            skew_adj = self._skew_factor * max(0.0, -log_m) * 0.10
        else:
            skew_adj = self._skew_factor * max(0.0, log_m) * 0.02
        return max(base_iv + skew_adj, self._iv_floor)

    def _options_half_spread(self, iv: float, dte: int) -> float:
        """Compute the per-leg half bid-ask spread for an OTM option.

        Models the empirical relationship between IV, DTE, and spread width:
        - Higher IV → wider spreads (market makers demand more edge)
        - Lower DTE → wider spreads (gamma risk increases near expiry)

        Returns the half-spread as a fraction of the option mid price.
        """
        iv_term = self._spread_iv_sensitivity * max(0.0, iv - 0.20)
        dte_term = self._spread_dte_sensitivity * max(0.0, 21 - dte)
        return min(self._spread_cap, self._spread_base + iv_term + dte_term)

    def _get_iv(self, hist: pd.DataFrame, market_context: dict | None = None) -> float:
        """Return IV for the current bar, in priority order:

        1. Stored IBKR daily implied_vol from the backfill (most realistic).
        2. VIX proxy from market_context — accepts scalar or full DataFrame.
           QQQ IV ≈ VIX × 1.10 (QQQ carries a small premium over SPX vol).
        3. Synthetic fallback: realized_vol × 1.15.

        Returns raw estimated IV without flooring. Credit-strategy entry gating
        (iv < iv_floor → no trade) is enforced separately in _build_position so
        that _get_iv remains a pure estimation function.
        """
        # Priority 1: stored IBKR IV
        if "implied_vol" in hist.columns:
            last_iv = hist["implied_vol"].dropna()
            if not last_iv.empty:
                stored = float(last_iv.iloc[-1])
                if stored > 0:
                    return stored

        # Priority 2: VIX proxy from market context (scalar or DataFrame),
        # scaled PER SYMBOL. R16: a flat 1.10 multiplier was applied to every
        # underlying, but VIX measures SPX vol — QQQ and IWM are structurally
        # more volatile. Measured against the real implied indices and 10y
        # realized vol:
        #   VXN/VIX  (NDX vs SPX, implied)  median 1.228   <- QQQ's true anchor
        #   QQQ/SPY  (realized, 10y)        median 1.354
        #   IWM/SPY  (realized, 10y)        median 1.465
        # So 1.10 understated QQQ implied vol by ~12% at the median (and by
        # ~34% on days like 2026-08-11: VIX 15.6 vs VXN 22.9) — biasing every
        # simulated QQQ credit low and every expected-move wing narrow.
        # SPY takes 1.00 (SPY IV ~ VIX by construction). QQQ takes the
        # measured VXN/VIX median. IWM has no free implied index (RVX is not
        # served by yfinance), so it takes its realized ratio shrunk by the
        # same implied-vs-realized factor VXN/QQQ exhibits (1.228/1.354):
        # 1.465 * 0.907 ~ 1.33. Override per symbol via AIT_BT_VOL_MULT_<SYM>.
        if market_context:
            vix = market_context.get("vix_close") or market_context.get("vix")
            if vix is not None:
                mult = self._symbol_vol_multiplier()
                if hasattr(vix, "reindex"):
                    # Full VIX DataFrame — align to current hist and take last value
                    vix_aligned = vix["Close"].reindex(hist.index, method="ffill")
                    if not vix_aligned.empty:
                        vix_val = float(vix_aligned.iloc[-1])
                        if vix_val > 0:
                            return vix_val / 100.0 * mult
                else:
                    vix_val = float(vix)
                    if vix_val > 0:
                        return vix_val / 100.0 * mult

        # Priority 3: synthetic fallback
        close_arr = hist["Close"].values
        rv = realized_vol(close_arr, window=20)
        return rv * 1.15

    # R16: VIX -> per-symbol implied-vol multipliers (see _get_iv). Unknown
    # symbols keep the historical 1.10 so nothing silently changes for
    # underlyings this calibration was never measured on.
    _VOL_MULT_VS_VIX = {"SPY": 1.00, "QQQ": 1.228, "IWM": 1.33, "DIA": 0.98}
    _VOL_MULT_DEFAULT = 1.10

    def _load_calibrated_spreads(self) -> None:
        """R16: adopt fitted per-symbol spread params when they exist.

        Silent no-op when the symbol has no calibration (keeps the config
        formula), so an uncalibrated underlying behaves exactly as before.
        """
        sym = (self._symbol or "").upper()
        if not sym:
            return
        try:
            from ait.data.historical import HistoricalDataStore
            p = HistoricalDataStore().load_spread_params(sym)
            if not p:
                return
            self._spread_base = float(p.get("spread_base", self._spread_base))
            self._spread_iv_sensitivity = float(
                p.get("spread_iv_sensitivity", self._spread_iv_sensitivity))
            self._spread_dte_sensitivity = float(
                p.get("spread_dte_sensitivity", self._spread_dte_sensitivity))
            self._calibrated_spreads = p
            log.info("calibrated_spreads_loaded", symbol=sym,
                     base=round(self._spread_base, 5),
                     iv_sens=round(self._spread_iv_sensitivity, 5),
                     dte_sens=round(self._spread_dte_sensitivity, 6))
        except Exception as e:  # noqa: BLE001 — never block a run on this
            log.debug("calibrated_spreads_unavailable", symbol=sym, error=str(e))

    def _symbol_vol_multiplier(self) -> float:
        """Per-symbol VIX scaling, env-overridable for sensitivity runs."""
        sym = (self._symbol or "").upper()
        env = os.environ.get(f"AIT_BT_VOL_MULT_{sym}")
        if env:
            try:
                return float(env)
            except ValueError:
                pass
        return self._VOL_MULT_VS_VIX.get(sym, self._VOL_MULT_DEFAULT)

    def _build_position(
        self,
        strategy: str,
        direction: SignalDirection,
        row: pd.Series,
        hist: pd.DataFrame,
        today_date: date,
        capital: float,
        market_context: dict | None = None,
    ) -> dict | None:
        """Build a position dict with proper BS pricing for the strategy type."""
        underlying = row["Close"]
        iv = self._get_iv(hist, market_context=market_context)
        dte = self._entry_dte
        t = dte / 365.0
        r = 0.05

        # Entry gate for credit strategies: insufficient IV → premiums too small to
        # justify spread costs and risk. Honest no-trade rather than fake-pricing at floor.
        if strategy in CREDIT_STRATEGIES and iv < self._iv_floor:
            return None

        if strategy in CREDIT_STRATEGIES:
            return self._build_credit_position(strategy, underlying, iv, t, r, dte, today_date, capital)
        else:
            return self._build_debit_position(strategy, direction, underlying, iv, t, r, dte, today_date, capital)

    def _build_debit_position(
        self, strategy: str, direction: SignalDirection,
        S: float, iv: float, t: float, r: float, dte: int,
        today_date: date, capital: float,
    ) -> dict | None:
        """Build a debit (long option/spread) position."""

        if strategy == "long_call":
            strike = find_strike_by_delta(S, t, iv, self._delta_long, OptionType.CALL, r)
            price = black_scholes_price(S, strike, t, r,
                self._get_leg_iv(iv, strike, S, OptionType.CALL), OptionType.CALL)
            opt_type = "call"
        elif strategy == "long_put":
            strike = find_strike_by_delta(S, t, iv, -self._delta_long, OptionType.PUT, r)
            price = black_scholes_price(S, strike, t, r,
                self._get_leg_iv(iv, strike, S, OptionType.PUT), OptionType.PUT)
            opt_type = "put"
        elif strategy == "bull_call_spread":
            long_strike = find_strike_by_delta(S, t, iv, self._delta_long, OptionType.CALL, r)
            short_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            long_price = black_scholes_price(S, long_strike, t, r,
                self._get_leg_iv(iv, long_strike, S, OptionType.CALL), OptionType.CALL)
            short_price = black_scholes_price(S, short_strike, t, r,
                self._get_leg_iv(iv, short_strike, S, OptionType.CALL), OptionType.CALL)
            price = long_price - short_price  # Net debit
            strike = long_strike
            opt_type = "call"
            if price <= 0:
                return None
            return self._finalize_spread_position(
                strategy, "debit", price, S, iv, dte, today_date, capital,
                long_strike=long_strike, short_strike=short_strike, opt_type="call",
            )
        elif strategy == "bear_put_spread":
            long_delta = -self._delta_long
            short_delta = -self._delta_short
            long_strike = find_strike_by_delta(S, t, iv, long_delta, OptionType.PUT, r)
            short_strike = find_strike_by_delta(S, t, iv, short_delta, OptionType.PUT, r)
            long_price = black_scholes_price(S, long_strike, t, r,
                self._get_leg_iv(iv, long_strike, S, OptionType.PUT), OptionType.PUT)
            short_price = black_scholes_price(S, short_strike, t, r,
                self._get_leg_iv(iv, short_strike, S, OptionType.PUT), OptionType.PUT)
            price = long_price - short_price  # Net debit
            strike = long_strike
            opt_type = "put"
            if price <= 0:
                return None
            return self._finalize_spread_position(
                strategy, "debit", price, S, iv, dte, today_date, capital,
                long_strike=long_strike, short_strike=short_strike, opt_type="put",
            )
        elif strategy == "long_strangle":
            # Buy OTM call + buy OTM put — profit from large moves in either direction.
            # Delta is IV-scaled: high IV → go further OTM for cheaper entry on vol expansion.
            iv_scale = max(0.5, min(1.5, self._iv_floor / iv))
            effective_delta = self._delta_long * (1.0 + self._delta_iv_scale * (iv_scale - 1.0))
            effective_delta = max(0.05, min(0.45, effective_delta))
            call_strike = find_strike_by_delta(S, t, iv, effective_delta, OptionType.CALL, r)
            put_strike  = find_strike_by_delta(S, t, iv, -effective_delta, OptionType.PUT, r)

            call_price = black_scholes_price(S, call_strike, t, r,
                self._get_leg_iv(iv, call_strike, S, OptionType.CALL), OptionType.CALL)
            put_price  = black_scholes_price(S, put_strike, t, r,
                self._get_leg_iv(iv, put_strike, S, OptionType.PUT), OptionType.PUT)

            total_cost = (call_price + put_price) * (1 + self._slippage_pct)
            cost_per_contract = total_cost * 100
            if cost_per_contract <= 0 or capital < cost_per_contract:
                return None

            contracts = int(capital * self._position_size_pct / cost_per_contract)
            if contracts < 1:
                if cost_per_contract <= capital * 0.25:
                    contracts = 1
                else:
                    return None

            return {
                "symbol": "SIM", "strategy": "long_strangle",
                "direction": SignalDirection.NEUTRAL.value, "trade_type": "debit",
                "entry_date": str(today_date), "entry_price": round(total_cost, 4),
                "contracts": contracts, "n_legs": 2,
                "long_call_strike": round(call_strike, 0),
                "long_put_strike": round(put_strike, 0),
                "strike": round(S, 0), "option_type": "strangle",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }
        else:
            return None

        # Single-leg sizing
        price *= (1 + self._slippage_pct)  # Buy at ask
        cost_per_contract = price * 100
        if cost_per_contract <= 0 or capital < cost_per_contract:
            return None
        contracts = int(capital * self._position_size_pct / cost_per_contract)
        if contracts < 1:
            return None

        return {
            "symbol": "SIM",
            "strategy": strategy,
            "direction": direction.value,
            "trade_type": "debit",
            "entry_date": str(today_date),
            "entry_price": round(price, 4),
            "contracts": contracts,
            "n_legs": 1,
            "strike": round(strike, 0),
            "option_type": opt_type,
            "entry_iv": round(iv, 4),
            "underlying_at_entry": round(S, 2),
            "expiry_date": str(today_date + timedelta(days=dte)),
            "high_water_mark": 0.0,
        }

    def _log_unaffordable(self, strategy: str, risk_per_contract: float,
                          capital: float) -> None:
        """R16: attribute a zero-trade credit arm to SIZING, not to 'no edge'.

        The `contracts < 1` budget gate is intentional policy (refusing a
        structure whose single-contract risk exceeds position_size_pct of
        capital), and it is shared by every condor-family builder. But it was
        completely silent, so a shadow-tournament arm whose per-contract risk
        never fits the budget at index-ETF prices (jade_lizard's S*0.20*100
        margin is the extreme case) reported n=0 with no error and no reason.
        """
        log.debug(
            "credit_position_unaffordable",
            component="backtesting.engine",
            strategy=strategy,
            risk_per_contract=round(float(risk_per_contract), 2),
            budget=round(float(capital) * self._position_size_pct, 2),
            capital=round(float(capital), 2),
        )

    def _build_credit_position(
        self, strategy: str,
        S: float, iv: float, t: float, r: float, dte: int,
        today_date: date, capital: float,
    ) -> dict | None:
        """Build a credit (short premium) position.

        Iron condor: sell OTM call+put, buy further OTM wings.
        P&L is inverted: collect premium upfront, buy back cheaper later.
        """
        if strategy == "iron_condor":
            short_call_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            short_put_strike = find_strike_by_delta(S, t, iv, -self._delta_short, OptionType.PUT, r)

            # Wings: vol-scaled by wing_k; wing_floor_dollars is the hard safety minimum.
            # wing_k=1.0 → 1-sigma expected move; wing_k=2.0 → 2-sigma (wider, safer).
            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing_width = max(self._wing_k * expected_move, self._wing_floor_dollars)
            long_call_strike = short_call_strike + wing_width
            long_put_strike = short_put_strike - wing_width

            # Price each leg with its own skewed IV
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_put_price = black_scholes_price(S, long_put_strike, t, r,
                self._get_leg_iv(iv, long_put_strike, S, OptionType.PUT), OptionType.PUT)

            # Net credit received at mid prices
            net_credit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
            if net_credit <= 0:
                return None

            # R6 live-parity construction gates (src/ait/strategies/iron_condor.py
            # applies both to the MID-price total credit, before friction):
            # 1. Cost floor — live rejects total_credit < AIT_IC_MIN_CREDIT
            #    ($0.70 default): at a 50% TP the gross must clear ~3x the
            #    round-trip commission/crossing cost.
            if net_credit < self._ic_min_credit:
                return None
            # 2. Credit-to-width gate — live rejects total_credit/max_width <
            #    AIT_IC_MIN_CREDIT_WIDTH (0.20 default). Engine wings are
            #    symmetric, so max_width == wing_width.
            if wing_width > 0 and (net_credit / wing_width) < self._ic_min_credit_width:
                return None

            # Per-leg spread cost: 4 legs × half-spread (sell at bid, buy at ask)
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (short_call_price + short_put_price + long_call_price + long_put_price) / 4
            spread_cost = 4 * half_spread * avg_leg_mid
            net_credit = max(0.0, net_credit - spread_cost)
            if net_credit <= 0:
                return None

            # Max loss = wing width - net credit (per share)
            max_loss_per_share = wing_width - net_credit
            max_loss_per_contract = max_loss_per_share * 100

            # Position sizing based on max loss (margin requirement)
            import math
            if (max_loss_per_contract <= 0 or math.isnan(max_loss_per_contract)
                    or capital < max_loss_per_contract):
                return None
            contracts = int(capital * self._position_size_pct / max_loss_per_contract)
            if contracts < 1:
                self._log_unaffordable(strategy, max_loss_per_contract, capital)
                return None

            return {
                "symbol": "SIM",
                "strategy": "iron_condor",
                "direction": SignalDirection.NEUTRAL.value,
                "trade_type": "credit",
                "entry_date": str(today_date),
                "entry_price": round(net_credit, 4),  # Credit received per share
                "contracts": contracts,
                "n_legs": 4,
                "short_call_strike": round(short_call_strike, 0),
                "short_put_strike": round(short_put_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "long_put_strike": round(long_put_strike, 0),
                "strike": round(S, 0),  # Reference: underlying at entry
                "option_type": "iron_condor",
                "entry_iv": round(iv, 4),
                "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy in ("wide_wing_condor", "broken_wing_condor"):
            # SHADOW ROUND 3 (PLAN 2026-08-04, pre-registered): wings-cost
            # interpolants between the condor (PF 0.97) and the wings-free
            # strangle benchmark (PF 1.30). wide_wing: BOTH wings at 2x the
            # standard k*EM distance; broken_wing: call wing standard, put
            # wing 2x (skew makes the put wing the expensive insurance).
            # REGISTERED DEVIATION from the live ratio gate: the credit floor
            # is held ABSOLUTE at the standard-width equivalent
            # (0.20 * std_width) so the entry population matches the live
            # condor; the ratio floor scales by std/max width.
            short_call_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            short_put_strike = find_strike_by_delta(S, t, iv, -self._delta_short, OptionType.PUT, r)
            expected_move = S * iv * (dte / 365.0) ** 0.5
            std_width = max(self._wing_k * expected_move, self._wing_floor_dollars)
            call_width = std_width * (2.0 if strategy == "wide_wing_condor" else 1.0)
            put_width = std_width * 2.0
            long_call_strike = short_call_strike + call_width
            long_put_strike = short_put_strike - put_width
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_put_price = black_scholes_price(S, long_put_strike, t, r,
                self._get_leg_iv(iv, long_put_strike, S, OptionType.PUT), OptionType.PUT)
            net_credit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
            if net_credit <= 0 or net_credit < self._ic_min_credit:
                return None
            max_width = max(call_width, put_width)
            scaled_floor = self._ic_min_credit_width * (std_width / max_width)
            if max_width > 0 and (net_credit / max_width) < scaled_floor:
                return None
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (short_call_price + short_put_price + long_call_price + long_put_price) / 4
            net_credit = max(0.0, net_credit - 4 * half_spread * avg_leg_mid)
            if net_credit <= 0:
                return None
            max_loss_per_share = max_width - net_credit
            max_loss_per_contract = max_loss_per_share * 100
            import math
            if (max_loss_per_contract <= 0 or math.isnan(max_loss_per_contract)
                    or capital < max_loss_per_contract):
                return None
            contracts = int(capital * self._position_size_pct / max_loss_per_contract)
            if contracts < 1:
                self._log_unaffordable(strategy, max_loss_per_contract, capital)
                return None
            return {
                "symbol": "SIM",
                "strategy": strategy,
                "direction": SignalDirection.NEUTRAL.value,
                "trade_type": "credit",
                "entry_date": str(today_date),
                "entry_price": round(net_credit, 4),
                "contracts": contracts,
                "n_legs": 4,
                "short_call_strike": round(short_call_strike, 0),
                "short_put_strike": round(short_put_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "long_put_strike": round(long_put_strike, 0),
                "strike": round(S, 0),
                "option_type": strategy,
                "entry_iv": round(iv, 4),
                "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "iron_butterfly":
            # SHADOW TOURNAMENT candidate (PLAN 2026-08-03): condor variant
            # with BOTH shorts at-the-money — much larger credit, so the
            # credit/width ratio clears in low IV where condors starve.
            # Same wing logic (wing_k x expected move, floor applies).
            atm = round(S)
            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing_width = max(self._wing_k * expected_move, self._wing_floor_dollars)
            short_call_strike = short_put_strike = float(atm)
            long_call_strike = atm + wing_width
            long_put_strike = atm - wing_width
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_put_price = black_scholes_price(S, long_put_strike, t, r,
                self._get_leg_iv(iv, long_put_strike, S, OptionType.PUT), OptionType.PUT)
            net_credit = (short_call_price + short_put_price) - (long_call_price + long_put_price)
            if net_credit <= 0:
                return None
            if net_credit < self._ic_min_credit:
                return None
            if wing_width > 0 and (net_credit / wing_width) < self._ic_min_credit_width:
                return None
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (short_call_price + short_put_price + long_call_price + long_put_price) / 4
            net_credit = max(0.0, net_credit - 4 * half_spread * avg_leg_mid)
            if net_credit <= 0:
                return None
            max_loss_per_share = wing_width - net_credit
            max_loss_per_contract = max_loss_per_share * 100
            import math
            if (max_loss_per_contract <= 0 or math.isnan(max_loss_per_contract)
                    or capital < max_loss_per_contract):
                return None
            contracts = int(capital * self._position_size_pct / max_loss_per_contract)
            if contracts < 1:
                self._log_unaffordable(strategy, max_loss_per_contract, capital)
                return None
            return {
                "symbol": "SIM", "strategy": "iron_butterfly",
                "direction": SignalDirection.NEUTRAL.value, "trade_type": "credit",
                "entry_date": str(today_date), "entry_price": round(net_credit, 4),
                "contracts": contracts, "n_legs": 4,
                "short_call_strike": round(short_call_strike, 0),
                "short_put_strike": round(short_put_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "long_put_strike": round(long_put_strike, 0),
                "strike": round(S, 0), "option_type": "iron_butterfly",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "put_credit_spread":
            # Sell higher strike put, buy lower strike put = bullish credit spread
            short_put_strike = find_strike_by_delta(S, t, iv, -self._delta_short, OptionType.PUT, r)

            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing = max(self._wing_k * expected_move * 0.5, self._wing_floor_dollars)

            long_put_strike = short_put_strike - wing

            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            long_put_price = black_scholes_price(S, long_put_strike, t, r,
                self._get_leg_iv(iv, long_put_strike, S, OptionType.PUT), OptionType.PUT)

            net_credit = short_put_price - long_put_price
            if net_credit <= 0:
                return None

            # Per-leg spread cost: 2 legs
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (short_put_price + long_put_price) / 2
            net_credit = max(0.0, net_credit - 2 * half_spread * avg_leg_mid)
            if net_credit <= 0:
                return None

            max_loss_per_share = wing - net_credit
            max_loss_per_contract = max_loss_per_share * 100

            if max_loss_per_contract <= 0 or capital < max_loss_per_contract:
                return None

            # Position sizing
            size_pct = self._position_size_pct
            contracts = int(capital * size_pct / max_loss_per_contract)
            if contracts < 1:
                if max_loss_per_contract <= capital * 0.25:
                    contracts = 1
                else:
                    return None

            return {
                "symbol": "SIM",
                "strategy": "put_credit_spread",
                "direction": SignalDirection.BULLISH.value,
                "trade_type": "credit",
                "entry_date": str(today_date),
                "entry_price": round(net_credit, 4),
                "contracts": contracts,
                "n_legs": 2,
                "short_put_strike": round(short_put_strike, 0),
                "long_put_strike": round(long_put_strike, 0),
                "strike": round(short_put_strike, 0),
                "option_type": "put",
                "entry_iv": round(iv, 4),
                "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "call_credit_spread":
            # SHADOW R2: bearish twin of put_credit_spread.
            short_call_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing = max(self._wing_k * expected_move * 0.5, self._wing_floor_dollars)
            long_call_strike = short_call_strike + wing
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            net_credit = short_call_price - long_call_price
            if net_credit <= 0:
                return None
            half_spread = self._options_half_spread(iv, dte)
            net_credit = max(0.0, net_credit - 2 * half_spread *
                             (short_call_price + long_call_price) / 2)
            if net_credit <= 0:
                return None
            max_loss_per_share = wing - net_credit
            max_loss_per_contract = max_loss_per_share * 100
            if max_loss_per_contract <= 0 or capital < max_loss_per_contract:
                return None
            contracts = int(capital * self._position_size_pct / max_loss_per_contract)
            if contracts < 1:
                if max_loss_per_contract <= capital * 0.25:
                    contracts = 1
                else:
                    return None
            return {
                "symbol": "SIM", "strategy": "call_credit_spread",
                "direction": SignalDirection.BEARISH.value, "trade_type": "credit",
                "entry_date": str(today_date), "entry_price": round(net_credit, 4),
                "contracts": contracts, "n_legs": 2,
                "short_call_strike": round(short_call_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "strike": round(short_call_strike, 0), "option_type": "call",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max_loss_per_share, 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "jade_lizard":
            # SHADOW R2 BENCHMARK ONLY (PLAN 2026-08-03): short put + short
            # call spread. Naked put side => NEVER live-promotable at current
            # capital. Sized by the strangle margin convention (S*0.20*100).
            short_put_strike = find_strike_by_delta(S, t, iv, -self._delta_short, OptionType.PUT, r)
            short_call_strike = find_strike_by_delta(S, t, iv, self._delta_short, OptionType.CALL, r)
            expected_move = S * iv * (dte / 365.0) ** 0.5
            wing = max(self._wing_k * expected_move * 0.5, self._wing_floor_dollars)
            long_call_strike = short_call_strike + wing
            short_put_price = black_scholes_price(S, short_put_strike, t, r,
                self._get_leg_iv(iv, short_put_strike, S, OptionType.PUT), OptionType.PUT)
            short_call_price = black_scholes_price(S, short_call_strike, t, r,
                self._get_leg_iv(iv, short_call_strike, S, OptionType.CALL), OptionType.CALL)
            long_call_price = black_scholes_price(S, long_call_strike, t, r,
                self._get_leg_iv(iv, long_call_strike, S, OptionType.CALL), OptionType.CALL)
            net_credit = short_put_price + short_call_price - long_call_price
            if net_credit <= 0:
                return None
            half_spread = self._options_half_spread(iv, dte)
            avg_mid = (short_put_price + short_call_price + long_call_price) / 3
            net_credit = max(0.0, net_credit - 3 * half_spread * avg_mid)
            if net_credit <= 0:
                return None
            margin_per_contract = S * 0.20 * 100
            if capital < margin_per_contract:
                return None
            contracts = int(capital * self._position_size_pct / margin_per_contract)
            if contracts < 1:
                self._log_unaffordable(strategy, margin_per_contract, capital)
                return None
            return {
                "symbol": "SIM", "strategy": "jade_lizard",
                "direction": SignalDirection.NEUTRAL.value, "trade_type": "credit",
                "entry_date": str(today_date), "entry_price": round(net_credit, 4),
                "contracts": contracts, "n_legs": 3,
                "short_put_strike": round(short_put_strike, 0),
                "short_call_strike": round(short_call_strike, 0),
                "long_call_strike": round(long_call_strike, 0),
                "strike": round(S, 0), "option_type": "jade_lizard",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "max_loss_per_share": round(max(short_put_strike - net_credit,
                                                wing - net_credit), 4),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        elif strategy == "short_strangle":
            # Sell OTM call + sell OTM put — no wings (naked short premium).
            # Delta is IV-scaled: high IV → go further OTM for more breathing room.
            iv_scale = max(0.5, min(1.5, self._iv_floor / iv))
            effective_delta = self._delta_short * (1.0 + self._delta_iv_scale * (iv_scale - 1.0))
            effective_delta = max(0.05, min(0.45, effective_delta))
            short_call = find_strike_by_delta(S, t, iv, effective_delta, OptionType.CALL, r)
            short_put  = find_strike_by_delta(S, t, iv, -effective_delta, OptionType.PUT, r)

            call_price = black_scholes_price(S, short_call, t, r,
                self._get_leg_iv(iv, short_call, S, OptionType.CALL), OptionType.CALL)
            put_price  = black_scholes_price(S, short_put, t, r,
                self._get_leg_iv(iv, short_put, S, OptionType.PUT), OptionType.PUT)

            net_credit = call_price + put_price
            if net_credit <= 0:
                return None

            # Per-leg spread cost: 2 legs (both short)
            half_spread = self._options_half_spread(iv, dte)
            avg_leg_mid = (call_price + put_price) / 2
            net_credit = max(0.0, net_credit - 2 * half_spread * avg_leg_mid)
            if net_credit <= 0:
                return None

            # Margin approx: 20% of underlying per strangle (standard naked-option requirement)
            margin_per_contract = S * 0.20 * 100
            contracts = int(capital * self._position_size_pct / margin_per_contract)
            if contracts < 1:
                return None

            return {
                "symbol": "SIM", "strategy": "short_strangle",
                "direction": SignalDirection.NEUTRAL.value, "trade_type": "credit",
                "entry_date": str(today_date), "entry_price": round(net_credit, 4),
                "contracts": contracts, "n_legs": 2,
                "short_call_strike": round(short_call, 0),
                "short_put_strike": round(short_put, 0),
                "strike": round(S, 0), "option_type": "strangle",
                "entry_iv": round(iv, 4), "underlying_at_entry": round(S, 2),
                "expiry_date": str(today_date + timedelta(days=dte)),
                "high_water_mark": 0.0,
            }

        return None

    def _finalize_spread_position(
        self, strategy: str, trade_type: str, net_cost: float,
        S: float, iv: float, dte: int, today_date: date, capital: float,
        long_strike: float, short_strike: float, opt_type: str,
    ) -> dict | None:
        """Finalize a vertical spread position."""
        net_cost *= (1 + self._slippage_pct)  # Slippage on debit
        cost_per_contract = net_cost * 100
        if cost_per_contract <= 0 or capital < cost_per_contract:
            return None
        size_pct = self._position_size_pct
        contracts = int(capital * size_pct / cost_per_contract)
        if contracts < 1:
            # Allow 1 contract only if affordable within 25% of capital
            if cost_per_contract <= capital * 0.25:
                contracts = 1
            else:
                return None

        direction = SignalDirection.BULLISH if "bull" in strategy else SignalDirection.BEARISH

        return {
            "symbol": "SIM",
            "strategy": strategy,
            "direction": direction.value,
            "trade_type": "debit",
            "entry_date": str(today_date),
            "entry_price": round(net_cost, 4),
            "contracts": contracts,
            "n_legs": 2,
            "long_strike": round(long_strike, 0),
            "short_strike": round(short_strike, 0),
            "strike": round(long_strike, 0),
            "option_type": opt_type,
            "entry_iv": round(iv, 4),
            "underlying_at_entry": round(S, 2),
            "max_profit_per_share": round(abs(short_strike - long_strike) - net_cost, 4),
            "expiry_date": str(today_date + timedelta(days=dte)),
            "high_water_mark": 0.0,
        }

    # ------------------------------------------------------------------
    # Exit logic — handles both debit and credit positions
    # ------------------------------------------------------------------

    def _get_current_iv(self, pos: dict, hist: pd.DataFrame | None) -> float:
        """Mean-reverting IV: blends entry IV with current realized vol.

        Weights 70% toward entry IV (forward-looking anchor) and 30% toward
        the current realized vol × 1.15 premium, so vega P&L is non-zero
        without wild swings driven purely by realized vol noise.
        """
        entry_iv = pos.get("entry_iv") or pos.get("iv", self._iv_floor)
        if hist is None or len(hist) < 21:
            return entry_iv
        rv = realized_vol(hist["Close"].values, window=20) * 1.15
        return max(0.70 * entry_iv + 0.30 * rv, self._iv_floor)

    def _reprice_position(self, pos: dict, underlying: float, days_held: int,
                          hist: pd.DataFrame | None = None) -> float:
        """Reprice a position at the current underlying using Black-Scholes.

        Returns the current value per share:
        - For debit positions: current option/spread value (want it to go UP)
        - For credit positions: current cost to buy back (want it to go DOWN)
        """
        iv = self._get_current_iv(pos, hist)
        # Decay to the REAL expiration (entry_dte), not the hold-cap —
        # max_hold_days now closes the position early via its own exit
        # check (_check_exit_*) rather than by fabricating a shorter option.
        dte_remaining = max(self._entry_dte - days_held, 0)
        t = max(dte_remaining / 365.0, 0.0001)
        r = 0.05

        strategy = pos["strategy"]

        if strategy in ("iron_condor", "iron_butterfly", "wide_wing_condor", "broken_wing_condor"):
            sc = black_scholes_price(underlying, pos["short_call_strike"], t, r,
                self._get_leg_iv(iv, pos["short_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r,
                self._get_leg_iv(iv, pos["short_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            lc = black_scholes_price(underlying, pos["long_call_strike"], t, r,
                self._get_leg_iv(iv, pos["long_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            lp = black_scholes_price(underlying, pos["long_put_strike"], t, r,
                self._get_leg_iv(iv, pos["long_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            return (sc + sp) - (lc + lp)

        elif strategy == "put_credit_spread":
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r,
                self._get_leg_iv(iv, pos["short_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            lp = black_scholes_price(underlying, pos["long_put_strike"], t, r,
                self._get_leg_iv(iv, pos["long_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            return sp - lp

        elif strategy == "call_credit_spread":
            sc = black_scholes_price(underlying, pos["short_call_strike"], t, r,
                self._get_leg_iv(iv, pos["short_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            lc = black_scholes_price(underlying, pos["long_call_strike"], t, r,
                self._get_leg_iv(iv, pos["long_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            return sc - lc

        elif strategy == "jade_lizard":
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r,
                self._get_leg_iv(iv, pos["short_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            sc = black_scholes_price(underlying, pos["short_call_strike"], t, r,
                self._get_leg_iv(iv, pos["short_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            lc = black_scholes_price(underlying, pos["long_call_strike"], t, r,
                self._get_leg_iv(iv, pos["long_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            return sp + sc - lc

        elif strategy in ("bull_call_spread", "bear_put_spread"):
            long_strike = pos["long_strike"]
            short_strike = pos["short_strike"]
            if pos["option_type"] == "call":
                long_val = black_scholes_price(underlying, long_strike, t, r,
                    self._get_leg_iv(iv, long_strike, underlying, OptionType.CALL), OptionType.CALL)
                short_val = black_scholes_price(underlying, short_strike, t, r,
                    self._get_leg_iv(iv, short_strike, underlying, OptionType.CALL), OptionType.CALL)
            else:
                long_val = black_scholes_price(underlying, long_strike, t, r,
                    self._get_leg_iv(iv, long_strike, underlying, OptionType.PUT), OptionType.PUT)
                short_val = black_scholes_price(underlying, short_strike, t, r,
                    self._get_leg_iv(iv, short_strike, underlying, OptionType.PUT), OptionType.PUT)
            return long_val - short_val

        elif strategy == "short_strangle":
            sc = black_scholes_price(underlying, pos["short_call_strike"], t, r,
                self._get_leg_iv(iv, pos["short_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            sp = black_scholes_price(underlying, pos["short_put_strike"], t, r,
                self._get_leg_iv(iv, pos["short_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            return sc + sp  # Cost to buy back (want DOWN for profit)

        elif strategy == "long_strangle":
            lc = black_scholes_price(underlying, pos["long_call_strike"], t, r,
                self._get_leg_iv(iv, pos["long_call_strike"], underlying, OptionType.CALL), OptionType.CALL)
            lp = black_scholes_price(underlying, pos["long_put_strike"], t, r,
                self._get_leg_iv(iv, pos["long_put_strike"], underlying, OptionType.PUT), OptionType.PUT)
            return lc + lp  # Current value (want UP for profit)

        else:
            # Single-leg option
            opt_type = OptionType.CALL if pos["option_type"] == "call" else OptionType.PUT
            return black_scholes_price(underlying, pos["strike"], t, r,
                self._get_leg_iv(iv, pos["strike"], underlying, opt_type), opt_type)

    def _check_thesis_invalidation(
        self, pos: dict, hist: pd.DataFrame
    ) -> dict | None:
        """Return exit dict if the ML direction strongly contradicts the entry thesis.

        A NEUTRAL (iron condor / short strangle) entry is invalidated when the
        current ML prediction is STRONGLY directional (confidence ≥ 0.80).
        A directional entry (BULLISH/BEARISH) is invalidated when the prediction
        flips to the opposite direction with confidence ≥ 0.80.

        Returns None if no invalidation, or {"exit_date", "exit_reason"} dict.
        """
        if self._predictor is None:
            return None
        # Only re-evaluate after at least 2 days held (avoid same-bar noise)
        entry_date = date.fromisoformat(pos["entry_date"])
        if not hasattr(hist.index[-1], "date"):
            return None
        current_date = hist.index[-1].date() if hasattr(hist.index[-1], "date") else hist.index[-1]
        if (current_date - entry_date).days < 2:
            return None

        try:
            pred = self._predictor.predict(hist)
        except Exception:
            return None
        if pred is None or pred.confidence < 0.80:
            return None

        entry_dir = pos.get("direction", SignalDirection.NEUTRAL.value)
        pred_dir = pred.direction.value if hasattr(pred.direction, "value") else str(pred.direction)

        invalidated = False
        if entry_dir == SignalDirection.NEUTRAL.value:
            # Iron condor / short strangle: strong directional signal invalidates the range thesis
            if pred_dir in (SignalDirection.BULLISH.value, SignalDirection.BEARISH.value):
                invalidated = True
        elif entry_dir == SignalDirection.BULLISH.value and pred_dir == SignalDirection.BEARISH.value:
            invalidated = True
        elif entry_dir == SignalDirection.BEARISH.value and pred_dir == SignalDirection.BULLISH.value:
            invalidated = True

        if invalidated:
            return {
                "exit_date": str(current_date),
                "exit_time": str(current_date),
                "exit_reason": f"thesis_invalidated:{pred_dir}@{pred.confidence:.2f}",
            }
        return None

    def _check_exit(self, pos: dict, row: pd.Series, current_date: date,
                    hist: pd.DataFrame | None = None) -> dict | None:
        """Check if a position should be exited."""
        underlying = row["Close"]
        entry_date = date.fromisoformat(pos["entry_date"])
        days_held = (current_date - entry_date).days

        current_value = self._reprice_position(pos, underlying, days_held, hist)

        trade_type = pos.get("trade_type", "debit")
        entry_price = pos["entry_price"]

        if trade_type == "credit":
            # Credit position: we received entry_price, now it costs current_value to close
            # Profit when current_value < entry_price (cheaper to buy back)
            # pnl_pct: positive = profitable (value decayed)
            if entry_price > 0:
                pnl_pct = (entry_price - current_value) / entry_price
            else:
                pnl_pct = 0.0
        else:
            # Debit position: we paid entry_price, now it's worth current_value
            if entry_price > 0:
                pnl_pct = (current_value - entry_price) / entry_price
            else:
                pnl_pct = 0.0

        # Apply per-leg bid-ask spread at exit (scales with current IV and remaining DTE)
        exit_iv = float(pos.get("entry_iv", 0.25))
        expiry = date.fromisoformat(pos["expiry_date"])
        remaining_dte = max(0, (expiry - current_date).days)
        exit_half_spread = self._options_half_spread(exit_iv, remaining_dte)
        if trade_type == "credit":
            current_value *= (1 + exit_half_spread)  # Buy back at ask
        else:
            current_value *= (1 - exit_half_spread)  # Sell at bid

        result = self._dispatch_exit_check(pos, trade_type, pnl_pct, current_date, row)

        if result is not None:
            # R16 touch-stop: live transacts AT the touched strike intraday, not
            # at the close. Reprice the structure at that level so a pierce that
            # recovered by the bell is booked at what live would actually have
            # paid — the whole point of modelling the touch.
            if result.get("exit_reason") == "touch_stop":
                _tu = float(result.pop("touch_underlying"))
                current_value = self._reprice_position(pos, _tu, days_held, hist)
                current_value *= (1 + exit_half_spread)
                underlying = _tu

            # Calculate actual P&L
            pnl = self._calc_pnl(pos, current_value)
            result["pnl"] = round(pnl, 2)
            result["exit_price"] = round(current_value, 4)
            result["exit_underlying"] = round(float(underlying), 4)
            if "exit_time" not in result:
                result["exit_time"] = result.get("exit_date")  # EOD exit — no intraday timestamp
            return result

        return None

    def _dispatch_exit_check(
        self, pos: dict, trade_type: str, pnl_pct: float, current_date: date,
        row: pd.Series | None = None
    ) -> dict | None:
        """Route exit logic by trade type (R6 exit parity with live).

        CREDIT structures mirror live portfolio.py check_position(): ONE flat
        loss limit at credit_loss_limit_mult x credit received plus the
        DTE-laddered take-profit — NO trailing stop, NO breakeven tier, NO
        vol-adjusted stops. DEBIT structures keep the existing behaviour
        (trailing/breakeven when enabled, else fixed stop + profit target).
        """
        if trade_type == "credit":
            return self._check_exit_credit(pos, pnl_pct, current_date, row)
        if self._trailing_stop_enabled:
            return self._check_exit_trailing(pos, pnl_pct, current_date)
        return self._check_exit_fixed(pos, pnl_pct, current_date)

    @staticmethod
    def _credit_take_profit_pct(dte: int | None, time_decay_scaling: bool = True) -> float:
        """DTE-laddered take-profit target as a fraction of credit received.

        R20 #5a: the ladder (0.50/0.40/0.30/0.20 + breakpoints) used to be a
        hand copy of portfolio.py _get_take_profit_targets — now imported from
        the shared ait.execution.exit_policy authority. Stays a staticmethod
        with a defaulted flag because run_backtest.py calls it UNBOUND for its
        drift-check table; the exit path passes self._exit_time_decay_scaling.
        """
        return exit_policy.credit_take_profit_pct(
            dte, time_decay_scaling=time_decay_scaling
        )

    def _check_exit_credit(self, pos: dict, pnl_pct: float, current_date: date,
                           row: pd.Series | None = None) -> dict | None:
        """Credit-structure exits — R6 parity with live portfolio.py.

        pnl_pct is fraction of credit received (positive = value decayed).
        Order mirrors live check_position(): (0) SHORT-STRIKE TOUCH,
        (1) flat loss limit, (2) DTE-laddered take-profit, (3)
        expiry-approaching close at DTE<=5, (4) macro-event flatten.
        """
        # 0. R16: SHORT-STRIKE TOUCH — live's PRIMARY loss exit, and until now
        # the engine's single largest structural divergence. Live watches spot
        # every 30s and closes the moment price reaches a short strike; the
        # engine only ever saw the daily CLOSE, so an intraday pierce that
        # recovered by the bell was scored as an untouched winner. Every study
        # to date (wing_k, shadow R1-R3, ablations) therefore measured a
        # DIFFERENT exit policy than the one running live — biased toward
        # optimism precisely on the days that hurt.
        # Daily High/Low bracket the true intraday path, so they detect the
        # touch without intraday bars. The exit is priced AT the touched
        # strike (the level live would have transacted at), not at the close.
        if row is not None and self._touch_stop_enabled:
            _sp = pos.get("short_put_strike")
            _sc = pos.get("short_call_strike")
            try:
                _low = float(row["Low"]) if "Low" in row else None
                _high = float(row["High"]) if "High" in row else None
            except (TypeError, ValueError):
                _low = _high = None
            _touched = None
            if _sp and _low is not None and _low <= float(_sp):
                _touched = float(_sp)
            elif _sc and _high is not None and _high >= float(_sc):
                _touched = float(_sc)
            if _touched is not None:
                return {"exit_date": str(current_date),
                        "exit_reason": "touch_stop",
                        "touch_underlying": _touched}
        # HWM still tracked for journaling/analysis parity (live persists it
        # even though credit exits no longer trail off it).
        pos["high_water_mark"] = max(pos.get("high_water_mark", 0.0), pnl_pct)

        expiry = date.fromisoformat(pos["expiry_date"])
        remaining_dte = (expiry - current_date).days

        # 1. Flat loss limit: one stop at -mult x credit (env
        # AIT_CREDIT_LOSS_LIMIT). R16 #6 parity with live portfolio.py:413-414:
        # default 0 = flat stop DISABLED (mult <= 0 mirrors live's -999
        # sentinel); the wings cap the true tail, and R12-B1 showed touch-close
        # beats every flat level. Env override still re-arms the stop.
        if self._credit_loss_limit_mult > 0 and pnl_pct <= -self._credit_loss_limit_mult:
            return {"exit_date": str(current_date), "exit_reason": "credit_loss_limit"}

        # 2. DTE-laddered take-profit (shared exit_policy ladder — R20 #5a
        # also honors exit.time_decay_scaling exactly as live does; the flag
        # off collapses to live's flat 0.50 short target).
        if pnl_pct >= self._credit_take_profit_pct(
                remaining_dte, self._exit_time_decay_scaling):
            return {"exit_date": str(current_date), "exit_reason": "take_profit_short"}

        # 2b. Max-hold-days cap — RESEARCH-ONLY knob (Optuna-searched): live
        # has no "close after N days held" rule, only the DTE/touch/TP exits
        # in this function. 2026-08-24 fix: before this, max_hold_days WAS
        # entry_dte (the same variable), so this trigger never existed — the
        # position simply couldn't outlive its own fabricated expiration.
        # Now entry_dte is fixed near live's real chain selection and
        # max_hold_days independently forces an earlier close, letting
        # Optuna explore shorter/longer synthetic hold horizons.
        held = (current_date - date.fromisoformat(pos["entry_date"])).days
        if held >= self._max_hold_days:
            return {"exit_date": str(current_date), "exit_reason": "max_hold_reached"}

        # 3. Expiry-approaching close — live closes any position at
        # DTE <= exit_policy.EXPIRY_APPROACHING_DTE (portfolio.py rule 3a;
        # rule 3 assignment risk at DTE<=1 is subsumed).
        if remaining_dte <= exit_policy.EXPIRY_APPROACHING_DTE:
            reason = "expiry" if current_date >= expiry else "expiry_approaching"
            return {"exit_date": str(current_date), "exit_reason": reason}

        # 4. Macro-event flatten (portfolio.py rule 3d parity). PLAN
        # 2026-08-04: defined-risk EXEMPT — condors hold through events
        # (wings cap the surprise; the vol crush is the payoff). Only
        # undefined/assignment-risk strategies keep the early exit.
        # R20 #5a: the strategy list AND per-strategy windows (strangle-class
        # 5 days, CSP/CC 1) come from the shared exit_policy table — the R16
        # jade_lizard omission was exactly the hand-copy drift this kills.
        # 2026+2027-H1 calendar: inactive for pre-2026 windows.
        _flat_window = exit_policy.macro_flatten_window_days(pos.get("strategy"))
        if (self._economic_cal is not None
                and contract_flag("AIT_SKIP_MACRO_EVENTS")
                and _flat_window is not None):
            try:
                _d2e = self._economic_cal.days_until_next_event(current_date)
            except Exception:  # noqa: BLE001
                _d2e = None
            if _d2e is not None and _d2e <= _flat_window:
                return {"exit_date": str(current_date), "exit_reason": "macro_event_flatten"}

        return None

    def _check_exit_fixed(self, pos: dict, pnl_pct: float, current_date: date) -> dict | None:
        """Fixed stop-loss / take-profit (DEBIT trades only since R6 parity)."""
        stop = self._stop_loss_pct

        # Stop loss
        if pnl_pct <= -stop:
            return {"exit_date": str(current_date), "exit_reason": "stop_loss"}

        # Profit target
        target = self._profit_target_pct
        if pnl_pct >= target:
            return {"exit_date": str(current_date), "exit_reason": "profit_target"}

        # Max-hold-days cap (research-only knob; see _check_exit_credit)
        held = (current_date - date.fromisoformat(pos["entry_date"])).days
        if held >= self._max_hold_days:
            return {"exit_date": str(current_date), "exit_reason": "max_hold_reached"}

        # Expiry
        expiry = date.fromisoformat(pos["expiry_date"])
        if current_date >= expiry:
            return {"exit_date": str(current_date), "exit_reason": "expiry"}

        return None

    def _check_exit_trailing(self, pos: dict, pnl_pct: float, current_date: date) -> dict | None:
        """Dynamic trailing stop."""
        pos["high_water_mark"] = max(pos.get("high_water_mark", 0.0), pnl_pct)
        hwm = pos["high_water_mark"]

        if hwm < self._breakeven_trigger_pct:
            effective_stop = -self._stop_loss_pct
            stop_label = "stop_loss"
        else:
            effective_stop = max(0.0, hwm - self._trailing_stop_pct)
            stop_label = "breakeven_stop" if effective_stop == 0.0 else "trailing_stop"

        if pnl_pct <= effective_stop:
            return {"exit_date": str(current_date), "exit_reason": stop_label}

        # Max-hold-days cap (research-only knob; see _check_exit_credit)
        held = (current_date - date.fromisoformat(pos["entry_date"])).days
        if held >= self._max_hold_days:
            return {"exit_date": str(current_date), "exit_reason": "max_hold_reached"}

        expiry = date.fromisoformat(pos["expiry_date"])
        if current_date >= expiry:
            return {"exit_date": str(current_date), "exit_reason": "expiry"}

        return None

    def _calc_pnl(self, pos: dict, current_value: float) -> float:
        """Calculate P&L including commissions.

        For debit positions: PnL = (current_value - entry_price) * 100 * contracts
        For credit positions: PnL = (entry_price - current_value) * 100 * contracts
        """
        contracts = pos["contracts"]
        entry_price = pos["entry_price"]
        trade_type = pos.get("trade_type", "debit")
        n_legs = pos.get("n_legs", 1)

        if trade_type == "credit":
            raw_pnl = (entry_price - current_value) * 100 * contracts
        else:
            raw_pnl = (current_value - entry_price) * 100 * contracts

        exit_commission = self._commission * contracts * n_legs
        total_commission = pos.get("entry_commission", 0) + exit_commission

        return raw_pnl - total_commission

    def _force_close(self, pos: dict, last_row: pd.Series, last_date: date,
                     hist: pd.DataFrame | None = None) -> dict:
        """Force-close a position at end of backtest."""
        entry_date = date.fromisoformat(pos["entry_date"])
        days_held = (last_date - entry_date).days
        current_value = self._reprice_position(pos, last_row["Close"], days_held, hist)

        trade_type = pos.get("trade_type", "debit")
        exit_iv = float(pos.get("entry_iv", 0.25))
        expiry = date.fromisoformat(pos["expiry_date"])
        remaining_dte = max(0, (expiry - last_date).days)
        exit_half_spread = self._options_half_spread(exit_iv, remaining_dte)
        if trade_type == "credit":
            current_value *= (1 + exit_half_spread)
        else:
            current_value *= (1 - exit_half_spread)

        pnl = self._calc_pnl(pos, current_value)
        return {
            "exit_date": str(last_date),
            "exit_time": str(last_date),
            "exit_price": round(current_value, 4),
            "pnl": round(pnl, 2),
            "exit_reason": "backtest_end",
        }

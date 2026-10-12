"""Walk-forward backtester with multi-symbol support.

Trains the ML model on historical data, then tests on unseen future data,
sliding the window forward. This is the gold standard for validating
trading strategies — it prevents overfitting by never testing on training data.

Usage:
    from ait.backtesting.walkforward import WalkForwardBacktester
    bt = WalkForwardBacktester(
        symbols=["SPY", "QQQ", "AAPL"],
        strategies=["long_call", "bull_call_spread", "iron_condor"],
    )
    result = await bt.run()
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ait.backtesting.engine import Backtester
from ait.backtesting.learner import BacktestLearner
from ait.backtesting.range_bank import RangeModelBank
from ait.backtesting.result import BacktestResult
from ait.config.settings import MLConfig
from ait.ml.ensemble import DirectionPredictor
from ait.strategies.base import SignalDirection
from ait.utils.logging import get_logger
from ait.utils.time import today_et


# R12-C: _range_model_worker (spawn-isolated GARCH/MS-GARCH/OU training) was
# removed with the GARCH family retirement to deprecated/research/. Range
# models are ML-only (XGB/LGBM) and train in-process — the RNG-contamination
# isolation existed solely for the parametric members' scipy/numpy RNG use.

log = get_logger("backtesting.walkforward")


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest."""

    train_days: int = 365        # ~1 year training window (calendar days)
    test_days: int = 63          # ~3 months test window
    step_days: int = 63          # == test_days: NON-overlapping by default (BT-M5:
                                 # 21d step x 63d test simulated every date ~3x,
                                 # triple-counting trades in pooled metrics and
                                 # compounding overlapping windows in total_return)
    gap_days: int = 5            # Purge gap between train and test
    # W-capital: TOTAL starting equity, ONE shared pool across all symbols.
    # None -> __post_init__ resolves config.yaml account.capital via
    # runtime_env.research_capital() (was a 100_000 literal; BT-H3 gave every
    # symbol its own full-capital sleeve, superseded by the pool allocator).
    initial_capital: float | None = None
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.03  # 3% realistic for multi-leg options
    # W64: None -> resolve_position_size_pct: backtest.position_size_pct if a
    # sizing study set one, else risk.max_position_risk_pct (live's per-trade
    # budget). Was a 0.05 literal while live funded 0.0133-0.03.
    position_size_pct: float | None = None
    # W64 live parity: cap the condor wing to the per-trade budget like live,
    # and cap contracts per trade at positions.max_contracts_per_trade (live: 1).
    # None -> config (backtest.budget_aware_wings / positions.max_contracts_per_trade).
    budget_aware_wings: bool | None = None
    max_contracts_per_trade: int | None = None
    condor_any_direction: bool | None = None   # W66: None -> backtest.condor_any_direction
    # W66 friction model v2: None -> backtest.friction_* (see BacktestConfig)
    friction_model: str | None = None
    friction_entry_capture: float | None = None
    friction_exit_capture: float | None = None
    friction_stress_multiplier: float | None = None
    # W67 gate parity (None -> backtest.gates_mode / fractal_veto_spread / penalty_on_range_prob).
    # fractal_veto_spread: an explicit legacy hurst_hard_veto_multiplier > 0 is honoured by
    # translating it once with the original formula (see ait.ml.regime_gate.resolve_veto_spread).
    gates_mode: str | None = None
    iv_rank_gate: bool | None = None          # W68: None -> backtest.iv_rank_gate
    fractal_veto_spread: float | None = None
    penalty_on_range_prob: float | None = None
    # W65 research-history knobs (None = legacy behaviour; the research entry points
    # resolve them from config.yaml backtest.* and pass them explicitly).
    history_start_date: "date | None" = None   # trim data + start windows here
    history_days: int | None = None            # None -> legacy train+test+100
    vlmc_min_coverage: float = 0.95            # 5-min day coverage needed for VLMC
    macro_event_types: "tuple[str, ...] | None" = None  # None -> hardcoded live calendar
    max_windows: int | None = None             # keep only the LAST N windows (timing/pilot runs)
    wing_floor_dollars: float | None = None   # W70: None -> backtest.wing_floor_dollars (2.0; R6 parity: live iron_condor min wing is $2, not $5)
    # R20b (pre-registered PLAN 2026-08-21): was a hardcoded 1.0 — the frozen
    # 2026-07 default that shadowed the PROMOTED live value (contract 1.6 since
    # the 2026-08-04 wide-wing promotion). None -> __post_init__ resolves via
    # contract_float("AIT_IC_WING_K"): env > config.yaml backtest.wing_k > 1.6.
    # Explicit values always win.
    wing_k: float | None = None
    # ML ABLATION (pre-registered PLAN 2026-08-03): False = gate-stack-only
    # arm — NO window models trained (direction AND range predictors both
    # read None) and the engine runs ungated (its designed fallback).
    # R16 #4: this flag previously gated only the DIRECTION predictor; the
    # range predictor still trained (and would have gated entries whenever
    # its training succeeded), making "gate_only" arms mislabeled. Range
    # training now honors the flag in _run_single_window AND
    # _pretrain_range_models (status "disabled_by_config", predictor None,
    # entries deliberately NOT blocked — unlike a FAILED training, which
    # blocks OOS IC entries via range_min_confidence=1.0 + engine R16 #3).
    train_window_models: bool = True
    # R20b follow-up: was a hardcoded 0.12 while live/engine resolve
    # settings.backtest.iv_floor (config.yaml 0.20) — the same "constructor
    # default silently shadows config" defect range_min_confidence was fixed
    # for below, just missed for this field. None -> __post_init__ resolves
    # load_settings().backtest.iv_floor (BacktestConfig default when no
    # config.yaml is reachable). Explicit values always win.
    iv_floor: float | None = None
    delta_iv_scale: float | None = None      # W70: None -> backtest.delta_iv_scale
    stop_loss_pct: float | None = None      # None -> __post_init__ resolves
    profit_target_pct: float | None = None  # settings.backtest baselines
    max_hold_days: int | None = None
    # R20b follow-up: was a hardcoded 0.55 while live/engine resolve
    # settings.risk.min_confidence (config.yaml 0.50) — same defect class as
    # iv_floor above. None -> __post_init__ resolves load_settings()
    # .risk.min_confidence (RiskConfig default when no config.yaml is
    # reachable). Explicit values always win.
    min_confidence: float | None = None
    # R20b (pre-registered PLAN 2026-08-21): was a hardcoded 0.55 while live
    # gates on ml.range_min_confidence = 0.65 (config.yaml; 0.65 beat 0.55
    # across every backtest metric) — the parity gap the range-floor sweep
    # exposed. None -> __post_init__ resolves load_settings().ml
    # .range_min_confidence (MLConfig default 0.65 when no config.yaml is
    # reachable). Explicit values always win.
    range_min_confidence: float | None = None  # threshold for range model on iron condors
    trailing_stop_enabled: bool = True
    trailing_stop_pct: float | None = None   # W70: None -> exit.trailing_stop_pct
    breakeven_trigger_pct: float | None = None   # W70: None -> exit.breakeven_trigger_pct
    max_concurrent_positions: int = 1
    max_entry_vol_annual: float | None = None    # W70: None -> backtest.max_entry_vol_annual
    optimize_per_window: bool = False
    optimize_n_trials: int = 50
    optimize_patience: int = 0       # 0 = disabled; N = stop after N non-improving trials
    optimize_min_trades: int | None = None   # None -> backtest.optimize_min_trades (W67: was a 10 literal)
    optimize_hard_min_trades: int | None = None          # W67: None -> backtest.*
    optimize_below_floor_gradient: float | None = None   # W67: None -> backtest.*
    optimize_seed: int = 42          # TPESampler seed — fix for reproducibility
    range_threshold_pct: float = 0.05  # Target move % for range model; links to strategy profitability
    hurst_regime_threshold: float | None = None
    hurst_regime_penalty: float | None = None
    hurst_hard_veto_multiplier: float = 0.0   # 0=disabled; Exp 20 post-mortem: QQQ spread never < 0.43, any multiplier blocks all entries
    multifractal_max_width: float | None = None
    # R20b review follow-up: was a hardcoded 0.30 while StrategyOptimizer's
    # OWN copy of this field was already migrated to the None-sentinel +
    # config-resolution pattern this PR gave every sibling field -- config.yaml
    # edits to backtest.iv_rank_rise_threshold were inert on the default
    # (non-optimized) walk-forward path, which forwards window_cfg's own
    # frozen value straight into the OOS Backtester. None -> __post_init__
    # resolves load_settings().backtest.iv_rank_rise_threshold.
    iv_rank_rise_threshold: float | None = None  # Exp 20: suppress IC entry when IV rank rose > this over last 10 days
    pct_from_60d_high_threshold: float = -1.0 # Exp 27: suppress IC entry when price is this far below 60d rolling high (-1.0 = disabled)
    # R20b review follow-up: same defect class as iv_rank_rise_threshold above.
    min_edge_over_baseline: float | None = None  # Exp 28: min weighted CV edge for range predictor to activate (0.0 = always use, higher = stricter quality gate)
    # Intraday execution params (Fix 1 / Gap H).
    # R20 #4/#5b: these four were PHANTOM — declared here (with their own
    # '09:30' fork of the entry window) but never forwarded to Backtester, so
    # setting them did nothing and every OOS run traded the engine's old
    # hardcoded 09:30-15:30 window while config declared 10:30. They are now
    # forwarded at BOTH Backtester construction sites; None = defer to the ONE
    # config-backed source (settings.backtest / config.yaml, default 10:30),
    # which the engine resolves itself.
    scan_interval_minutes: int | None = None    # how often to scan for signals during a session
    entry_window_start_et: str | None = None    # earliest allowed entry time (ET)
    entry_window_end_et: str | None = None      # latest allowed entry time (ET)
    limit_order_timeout_bars: int | None = None # cancel limit order after N 5-min bars without fill
    # Options spread model params (Fix 5 / Gap E)
    # W70: None -> backtest.spread_* (these four config keys were read by NOTHING; the
    # walk-forward carried its own literals 0.03 / 0.10 / 0.005 / 0.15).
    spread_base: float | None = None             # base half-spread per leg ($)
    spread_iv_sensitivity: float | None = None   # additional spread per unit IV above 0.20
    spread_dte_sensitivity: float | None = None  # additional spread per DTE below 21
    spread_cap: float | None = None              # maximum half-spread per leg ($)
    # R6 exit/construction parity knobs (None -> engine resolves the SAME env
    # var + default the live bot reads; see Backtester.__init__):
    credit_loss_limit_mult: float | None = None   # env AIT_CREDIT_LOSS_LIMIT (live default 0
                                                  # = flat stop DISABLED; R12-B1 + R16 #6)
    ic_min_credit: float | None = None            # env AIT_IC_MIN_CREDIT (live $0.70)
    # R20b: comment said "live 0.20" — stale since the 2026-08-04 promotion
    # (contract 0.10). None -> __post_init__ resolves
    # contract_float("AIT_IC_MIN_CREDIT_WIDTH"): env > config.yaml
    # backtest.ic_min_credit_width > 0.10. Explicit values always win.
    ic_min_credit_width: float | None = None      # env AIT_IC_MIN_CREDIT_WIDTH (live 0.10)
    macro_event_gate: bool = True                 # block credit entries pre macro event
                                                  # (2026+2027-H1 calendar; inactive earlier)
    pre_event_blackout_days: int | None = None    # R16 #7: None -> engine resolves from
                                                  # load_settings().risk (live parity),
                                                  # falling back to the RiskConfig default
    # Research-only credit take-profit ladder, rows (dte_floor, long, short)
    # as in exit_policy.CREDIT_TP_LADDER. None = the live ladder. A dataclass
    # field (not a module patch) so it pickles into ProcessPoolExecutor window
    # workers. scripts/take_profit_study.py is the consumer.
    credit_tp_ladder: tuple | None = None
    # Research-only entry DTE for BS pricing/decay. None = the engine's
    # options.dte_range[0] (14). Live picks among every expiry in
    # dte_range and measured a median 18 DTE (14-25) over its first 14
    # condors, so studies keyed to DTE (the take-profit ladder) set it.
    entry_dte: int | None = None
    # W79a: the short-DTE exit knobs and the searched short delta, all research-only and None =
    # the engine resolves from settings.backtest (so nothing moves unless a study sets them).
    # Before W79a none of these reached a trial or the OOS run: a searched delta_short was
    # applied to nothing (lead L1) and a study could not hold a fixed guard / fraction / flatten.
    expiry_guard_dte: int | None = None
    max_hold_fraction: float | None = None
    flatten_minutes_before_close: int | None = None
    delta_short: float | None = None
    # W79a: search the entry DTE and its exit dimensions per window (DteSearchConfig supplies the
    # ranges). None -> backtest.dte_search.enabled. Requires optimize_per_window.
    dte_search: bool | None = None
    dte_range: "tuple[int, int] | None" = None     # overrides dte_search.entry_dte (CLI --dte-range)
    optimize_strict: bool | None = None            # None -> backtest.dte_search.strict
    optimize_multivariate: bool | None = None      # None -> backtest.dte_search.multivariate
    # Research-only post-event blackout (scripts/post_event_study.py): block
    # credit entries for N calendar days after each date in post_event_dates
    # (ISO strings, so the config pickles into window workers). None = off.
    post_event_blackout_days: int | None = None
    post_event_dates: tuple | None = None
    optimize_n_jobs: int = 1              # Number of walk-forward WINDOWS to run in parallel via
                                          # ProcessPoolExecutor. Despite the "optimize_" prefix this
                                          # controls WINDOW-LEVEL parallelism, NOT Optuna trial-level
                                          # parallelism. It applies equally to optimized runs
                                          # (optimize_per_window=True) and ablation runs
                                          # (optimize_per_window=False). Optuna's own trial-level
                                          # n_jobs is always 1 (threading is GIL-bound; window-level
                                          # process parallelism is the effective speedup path).
                                          # Set via --wf-n-jobs CLI flag (default: 1 = sequential).
    optimize_val_split: bool = False      # H2: score Optuna objective on held-out 20% val slice

    def __post_init__(self) -> None:
        """R20b (pre-registered PLAN 2026-08-21): research defaults migrate to
        config resolution. R20 proved every prior absolute was priced without
        volatility data, so protecting old-study reproduction protects
        disavowed numbers — a bare WalkForwardConfig() now measures the
        CURRENT operating contract, not frozen 2026-07 literals.

        None is the sentinel meaning "resolve from the contract/config";
        explicit values always win (a historical reproduction must pass them).
        """
        if self.credit_tp_ladder is not None:
            from ait.execution.exit_policy import validate_ladder
            self.credit_tp_ladder = validate_ladder(self.credit_tp_ladder)
        if self.entry_dte is not None:
            self.entry_dte = int(self.entry_dte)
        if self.post_event_blackout_days is not None:
            self.post_event_blackout_days = int(self.post_event_blackout_days)
            if self.post_event_blackout_days < 0:
                raise ValueError("post_event_blackout_days must be >= 0")
        if self.post_event_dates is not None:
            self.post_event_dates = tuple(str(d)[:10] for d in self.post_event_dates)
        # W79a: entry_dte (and the guard / fraction / flatten / delta_short baselines) now reach
        # every trial Backtester through StrategyOptimizer, so they are no longer refused here.
        _research_only = [name for name in ("credit_tp_ladder", "post_event_blackout_days")
                          if getattr(self, name) is not None]
        if _research_only and self.optimize_per_window:
            # StrategyOptimizer trials build their own Backtester without
            # these overrides, so trials would score the config defaults while
            # the OOS run used the override. Refuse the combination.
            raise ValueError(
                f"{', '.join(_research_only)} not threaded into per-window "
                "Optuna trials; run with optimize_per_window=False")
        # W79a: DTE-search switches resolve from the loaded settings (research overlay).
        if self.max_hold_fraction is not None and not 0.0 < float(self.max_hold_fraction) <= 1.0:
            raise ValueError(f"max_hold_fraction must be in (0, 1], got {self.max_hold_fraction}")
        if self.dte_range is not None:
            _lo, _hi = (int(x) for x in self.dte_range)
            if not 0 <= _lo <= _hi <= 45:
                raise ValueError(f"dte_range must satisfy 0 <= low <= high <= 45, got {self.dte_range}")
            self.dte_range = (_lo, _hi)
        try:
            from ait.config.settings import DteSearchConfig as _DSC, load_settings as _ls
            _ds = _ls().backtest.dte_search
        except Exception:  # noqa: BLE001 - no config.yaml -> the model defaults
            from ait.config.settings import DteSearchConfig as _DSC
            _ds = _DSC()
        if self.dte_search is None:
            self.dte_search = bool(_ds.enabled)
        if self.optimize_strict is None:
            self.optimize_strict = bool(_ds.strict)
        if self.optimize_multivariate is None:
            self.optimize_multivariate = bool(_ds.multivariate)
        if self.dte_search and not self.optimize_per_window:
            raise ValueError("dte_search requires optimize_per_window=True (the DTE is chosen "
                             "per window by Optuna)")
        if self.initial_capital is None:
            from ait.config.runtime_env import research_capital
            self.initial_capital = research_capital()
        if self.wing_k is None:
            from ait.config.runtime_env import contract_float
            self.wing_k = contract_float("AIT_IC_WING_K")
        if self.ic_min_credit_width is None:
            from ait.config.runtime_env import contract_float
            self.ic_min_credit_width = contract_float("AIT_IC_MIN_CREDIT_WIDTH")
        # R20b review follow-up: load settings ONCE for every config-backed
        # field below via the shared resolve_config_value helper (also used
        # by engine.py/optimizer.py/the ML predictors) instead of each field
        # independently calling load_settings() (was up to 3 redundant reads
        # per WalkForwardConfig construction, now 1).
        from ait.config.settings import (
            BacktestConfig, ExitConfig, MLConfig, RiskConfig, load_settings,
            resolve_config_value,
        )
        try:
            _settings = load_settings()
        except Exception:  # noqa: BLE001 — no config.yaml -> per-field model defaults
            _settings = None
        from ait.config.settings import (
            resolve_max_contracts_per_trade, resolve_position_size_pct)
        self.position_size_pct = resolve_position_size_pct(
            self.position_size_pct, _settings)
        self.max_contracts_per_trade = resolve_max_contracts_per_trade(
            self.max_contracts_per_trade, _settings)
        self.budget_aware_wings = bool(resolve_config_value(
            self.budget_aware_wings, "backtest", "budget_aware_wings",
            BacktestConfig, _settings))
        self.condor_any_direction = bool(resolve_config_value(
            self.condor_any_direction, "backtest", "condor_any_direction",
            BacktestConfig, _settings))
        self.friction_model = str(resolve_config_value(
            self.friction_model, "backtest", "friction_model", BacktestConfig, _settings))
        self.friction_entry_capture = float(resolve_config_value(
            self.friction_entry_capture, "backtest", "friction_entry_capture",
            BacktestConfig, _settings))
        self.friction_exit_capture = float(resolve_config_value(
            self.friction_exit_capture, "backtest", "friction_exit_capture",
            BacktestConfig, _settings))
        self.friction_stress_multiplier = float(resolve_config_value(
            self.friction_stress_multiplier, "backtest", "friction_stress_multiplier",
            BacktestConfig, _settings))
        self.gates_mode = str(resolve_config_value(
            self.gates_mode, "backtest", "gates_mode", BacktestConfig, _settings))
        self.iv_rank_gate = bool(resolve_config_value(
            self.iv_rank_gate, "backtest", "iv_rank_gate", BacktestConfig, _settings))
        self.penalty_on_range_prob = float(resolve_config_value(
            self.penalty_on_range_prob, "backtest", "penalty_on_range_prob",
            BacktestConfig, _settings))
        if self.fractal_veto_spread is None and self.hurst_hard_veto_multiplier > 0:
            from ait.ml.regime_gate import resolve_veto_spread
            _thr = float(resolve_config_value(
                self.hurst_regime_threshold, "backtest", "hurst_regime_threshold",
                BacktestConfig, _settings))
            self.fractal_veto_spread = resolve_veto_spread(
                _thr, self.hurst_hard_veto_multiplier, None)
        self.fractal_veto_spread = float(resolve_config_value(
            self.fractal_veto_spread, "backtest", "fractal_veto_spread",
            BacktestConfig, _settings))
        self.optimize_min_trades = int(resolve_config_value(
            self.optimize_min_trades, "backtest", "optimize_min_trades",
            BacktestConfig, _settings))
        self.optimize_hard_min_trades = int(resolve_config_value(
            self.optimize_hard_min_trades, "backtest", "optimize_hard_min_trades",
            BacktestConfig, _settings))
        self.optimize_below_floor_gradient = float(resolve_config_value(
            self.optimize_below_floor_gradient, "backtest", "optimize_below_floor_gradient",
            BacktestConfig, _settings))
        self.range_min_confidence = float(resolve_config_value(
            self.range_min_confidence, "ml", "range_min_confidence", MLConfig, _settings))
        self.iv_floor = float(resolve_config_value(
            self.iv_floor, "backtest", "iv_floor", BacktestConfig, _settings))
        self.min_confidence = float(resolve_config_value(
            self.min_confidence, "risk", "min_confidence", RiskConfig, _settings))
        self.stop_loss_pct = float(resolve_config_value(
            self.stop_loss_pct, "backtest", "stop_loss_pct", BacktestConfig, _settings))
        self.profit_target_pct = float(resolve_config_value(
            self.profit_target_pct, "backtest", "profit_target_pct", BacktestConfig, _settings))
        self.max_hold_days = int(resolve_config_value(
            self.max_hold_days, "backtest", "max_hold_days", BacktestConfig, _settings))
        self.hurst_regime_threshold = float(resolve_config_value(
            self.hurst_regime_threshold, "backtest", "hurst_regime_threshold",
            BacktestConfig, _settings))
        self.hurst_regime_penalty = float(resolve_config_value(
            self.hurst_regime_penalty, "backtest", "hurst_regime_penalty",
            BacktestConfig, _settings))
        self.multifractal_max_width = float(resolve_config_value(
            self.multifractal_max_width, "backtest", "multifractal_max_width",
            BacktestConfig, _settings))
        # R20b review follow-up: same defect class as the three above -- was
        # a frozen literal even though StrategyOptimizer's own copy of these
        # two fields was already migrated (optimizer.py __init__).
        self.iv_rank_rise_threshold = float(resolve_config_value(
            self.iv_rank_rise_threshold, "backtest", "iv_rank_rise_threshold",
            BacktestConfig, _settings))
        self.min_edge_over_baseline = float(resolve_config_value(
            self.min_edge_over_baseline, "backtest", "min_edge_over_baseline",
            BacktestConfig, _settings))
        # W70: the remaining literals, now config-resolved (values unchanged at today's
        # config.yaml except wing_floor_dollars, which was already 2.0 here).
        # W70: the SAME contract call live's iron_condor.py uses (AIT_IC_WING_FLOOR, home
        # backtest.wing_floor_dollars) — one key, live AND research.
        from ait.config.runtime_env import contract_float as _cf
        self.wing_floor_dollars = (float(self.wing_floor_dollars)
                                   if self.wing_floor_dollars is not None
                                   else _cf("AIT_IC_WING_FLOOR"))
        self.delta_iv_scale = float(resolve_config_value(
            self.delta_iv_scale, "backtest", "delta_iv_scale", BacktestConfig, _settings))
        self.max_entry_vol_annual = float(resolve_config_value(
            self.max_entry_vol_annual, "backtest", "max_entry_vol_annual", BacktestConfig, _settings))
        self.trailing_stop_pct = float(resolve_config_value(
            self.trailing_stop_pct, "exit", "trailing_stop_pct", ExitConfig, _settings))
        self.breakeven_trigger_pct = float(resolve_config_value(
            self.breakeven_trigger_pct, "exit", "breakeven_trigger_pct", ExitConfig, _settings))
        for _f in ("spread_base", "spread_iv_sensitivity", "spread_dte_sensitivity", "spread_cap"):
            setattr(self, _f, float(resolve_config_value(
                getattr(self, _f), "backtest", _f, BacktestConfig, _settings)))


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""

    window_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    backtest_result: BacktestResult
    model_accuracy: float = 0.0
    # W-capital: why candidate trades never reached the shared pool, by reason
    # (unaffordable | portfolio_risk | max_open_positions | no_contracts).
    pool_rejections: dict = field(default_factory=dict)
    # R16 #3: per-symbol range-model training status for this window
    # ("ok" | "disabled_by_config" | failure reasons — anything not
    # ok/disabled means OOS IC entries were BLOCKED for that symbol).
    range_model_status: dict = field(default_factory=dict)
    # R20 #3: per-symbol MetaLabeler status for this window ("ok" = trained
    # and GATING OOS entries | "disabled_by_config" = ablation arm, no meta
    # gate | "not_trained" = enabled but training failed/short — no gate).
    # R16 #4 gated direction+range on train_window_models but missed this
    # third window model, so "ML-free" arms could silently carry a trained
    # XGB entry gate.
    meta_labeler_status: dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward backtest results."""

    windows: list[WindowResult] = field(default_factory=list)
    symbol_results: dict[str, BacktestResult] = field(default_factory=dict)
    strategy_results: dict[str, dict] = field(default_factory=dict)
    initial_capital: float = 10_000.0
    config: WalkForwardConfig | None = None
    # R16 #3: {window_id: {symbol: status}} for EVERY generated window
    # (including zero-trade ones) so the summary can surface how many
    # windows actually had a trained range model — the shipped ablation /
    # shadow studies ran with range training failing in 32/32 windows and
    # nothing in the summary said so.
    range_training_status: dict = field(default_factory=dict)
    # R20 #3: {window_id: {symbol: status}} for the window MetaLabeler,
    # surfaced in summary() exactly like range_training_status — an arm
    # labelled "gates only / ML-free" that ran WITH a trained meta gate is a
    # mislabeled study (the R16 #4 failure mode through the third model).
    meta_training_status: dict = field(default_factory=dict)
    # W66: {symbol: {reason: days}} summed over every window (zero-trade windows included)
    entry_funnel: dict = field(default_factory=dict)
    # W67: {window_id: {symbol: {"usable", "best_value", "best_n_trades"}}} (empty = no Optuna)
    optimization_status: dict = field(default_factory=dict)

    @property
    def optimization_failures(self) -> int:
        """W79a: (window, symbol) optimisations that FAILED and fell back to the base config."""
        return sum(1 for per_sym in self.optimization_status.values()
                   for st in (per_sym or {}).values() if isinstance(st, dict) and st.get("failed"))

    @property
    def total_trades(self) -> int:
        return sum(w.backtest_result.total_trades for w in self.windows)

    @property
    def total_return(self) -> float:
        if not self.windows:
            return 0.0
        # Chain returns across windows
        equity = self.initial_capital
        for w in self.windows:
            equity *= (1 + w.backtest_result.total_return)
        return (equity - self.initial_capital) / self.initial_capital

    @property
    def win_rate(self) -> float:
        all_trades = []
        for w in self.windows:
            all_trades.extend(w.backtest_result.trades)
        if not all_trades:
            return 0.0
        return sum(1 for t in all_trades if t.get("pnl", 0) > 0) / len(all_trades)

    @property
    def sharpe_ratio(self) -> float:
        all_pnls = []
        for w in self.windows:
            all_pnls.extend(t.get("pnl", 0) for t in w.backtest_result.trades)
        if len(all_pnls) < 2:
            return 0.0
        mean = np.mean(all_pnls)
        std = np.std(all_pnls, ddof=1)
        if std == 0:
            return 0.0
        from ait.backtesting.result import annualization_factor
        _all_t = [t for w in self.windows for t in w.backtest_result.trades]
        return float((mean / std) * annualization_factor(_all_t))

    @property
    def max_drawdown(self) -> float:
        equity = self.initial_capital
        peak = equity
        max_dd = 0.0
        for w in self.windows:
            for t in w.backtest_result.trades:
                equity += t.get("pnl", 0)
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
        return max_dd

    @property
    def profit_factor(self) -> float:
        all_trades = []
        for w in self.windows:
            all_trades.extend(w.backtest_result.trades)
        gross_wins = sum(t["pnl"] for t in all_trades if t.get("pnl", 0) > 0)
        gross_losses = abs(sum(t["pnl"] for t in all_trades if t.get("pnl", 0) <= 0))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    @property
    def avg_window_return(self) -> float:
        if not self.windows:
            return 0.0
        returns = [w.backtest_result.total_return for w in self.windows]
        return float(np.mean(returns))

    @property
    def consistency(self) -> float:
        """Fraction of windows that were profitable."""
        if not self.windows:
            return 0.0
        profitable = sum(1 for w in self.windows if w.backtest_result.total_return > 0)
        return profitable / len(self.windows)

    def _all_trades(self) -> list[dict]:
        out = []
        for w in self.windows:
            out.extend(w.backtest_result.trades)
        return out

    @property
    def sortino_ratio(self) -> float:
        """Downside-only volatility ratio — better for option-selling skew."""
        all_pnls = np.array([t.get("pnl", 0) for t in self._all_trades()])
        if len(all_pnls) < 2:
            return 0.0
        mean_pnl = float(all_pnls.mean())
        downside = all_pnls[all_pnls < 0]
        if len(downside) < 2:
            return float("inf") if mean_pnl > 0 else 0.0
        ds_std = float(downside.std(ddof=1))
        if ds_std == 0:
            return 0.0
        from ait.backtesting.result import annualization_factor
        _all_t = [t for w in self.windows for t in w.backtest_result.trades]
        return (mean_pnl / ds_std) * annualization_factor(_all_t)

    @property
    def avg_win(self) -> float:
        wins = [t["pnl"] for t in self._all_trades() if t.get("pnl", 0) > 0]
        return float(np.mean(wins)) if wins else 0.0

    @property
    def avg_loss(self) -> float:
        losses = [abs(t["pnl"]) for t in self._all_trades() if t.get("pnl", 0) <= 0]
        return float(np.mean(losses)) if losses else 0.0

    @property
    def win_loss_ratio(self) -> float:
        if self.avg_loss == 0:
            return float("inf") if self.avg_win > 0 else 0.0
        return self.avg_win / self.avg_loss

    @property
    def expectancy(self) -> float:
        wr = self.win_rate
        return wr * self.avg_win - (1 - wr) * self.avg_loss

    @property
    def best_trade(self) -> float:
        trades = self._all_trades()
        return max((t.get("pnl", 0) for t in trades), default=0.0)

    @property
    def worst_trade(self) -> float:
        trades = self._all_trades()
        return min((t.get("pnl", 0) for t in trades), default=0.0)

    @property
    def capital_utilization(self) -> float:
        """Avg % of initial capital deployed across the backtest period."""
        trades = self._all_trades()
        if not trades or not self.windows:
            return 0.0
        first_date = self.windows[0].test_start
        last_date = self.windows[-1].test_end
        total_days = max(1, (last_date - first_date).days)

        from datetime import datetime, date as _date
        def to_d(d):
            if d is None: return None
            if isinstance(d, _date) and not isinstance(d, datetime): return d
            if hasattr(d, "date"): return d.date()
            if isinstance(d, str):
                try: return datetime.fromisoformat(d.split("T")[0]).date()
                except Exception: return None
            return None

        capital_days = 0.0
        for t in trades:
            entry = to_d(t.get("entry_date"))
            exit_ = to_d(t.get("exit_date"))
            risk = t.get("max_loss") or t.get("cost") or abs(t.get("pnl", 0)) * 2
            if entry and exit_ and risk:
                hold = max(1, (exit_ - entry).days)
                capital_days += risk * hold
        return capital_days / (self.initial_capital * total_days) if self.initial_capital > 0 else 0.0

    @property
    def cash_drag_adjusted_return(self) -> float:
        """Total return + idle-cash T-bill yield adjustment."""
        if not self.windows:
            return self.total_return
        first_date = self.windows[0].test_start
        last_date = self.windows[-1].test_end
        years = max(0.01, (last_date - first_date).days / 365.25)
        idle_pct = max(0, 1 - self.capital_utilization)
        return self.total_return + (0.05 * idle_pct * years)

    @property
    def raroc(self) -> float:
        """Return on capital actually deployed."""
        util = self.capital_utilization
        return self.total_return / util if util > 0 else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  WALK-FORWARD BACKTEST RESULTS",
            "=" * 60,
            f"  Windows:           {len(self.windows)}",
            f"  Total Trades:      {self.total_trades}",
            f"  Total Return:      {self.total_return:.2%}",
            f"  Cash-Drag Adj Ret: {self.cash_drag_adjusted_return:.2%}  (idle cash @ 5%)",
            "-" * 60,
            "  RISK-ADJUSTED",
            f"  Sharpe Ratio:      {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio:     {self.sortino_ratio:.2f}  (downside-only vol)",
            f"  Max Drawdown:      {self.max_drawdown:.2%}",
            "-" * 60,
            "  TRADE QUALITY",
            f"  Win Rate:          {self.win_rate:.2%}",
            f"  Avg Win:           ${self.avg_win:,.2f}",
            f"  Avg Loss:          ${self.avg_loss:,.2f}",
            f"  Win/Loss Ratio:    {self.win_loss_ratio:.2f}  (>1 = winners bigger)",
            f"  Expectancy/Trade:  ${self.expectancy:,.2f}",
            f"  Best Trade:        ${self.best_trade:,.2f}",
            f"  Worst Trade:       ${self.worst_trade:,.2f}",
            f"  Profit Factor:     {self.profit_factor:.2f}",
            "-" * 60,
            "  CAPITAL EFFICIENCY",
            f"  Utilization:       {self.capital_utilization:.1%}  (avg % deployed)",
            f"  RAROC:             {self.raroc:.1%}  (return on deployed)",
            "-" * 60,
            "  CONSISTENCY",
            f"  Profitable Windows:{self.consistency:.0%}",
            f"  Avg Window Return: {self.avg_window_return:.2%}",
            "-" * 60,
        ]

        # R16 #3: surface per-window range-model training status. A study
        # whose range model never trained is NOT evidence about the range
        # gate — say so in the result text, loudly.
        if self.range_training_status:
            _status_counts: dict[str, int] = {}
            for _per_sym in self.range_training_status.values():
                for _s in _per_sym.values():
                    _status_counts[_s] = _status_counts.get(_s, 0) + 1
            _total = sum(_status_counts.values())
            _ok = _status_counts.get("ok", 0)
            lines.append("  RANGE MODEL TRAINING (per window-symbol)")
            lines.append(f"  Trained ok:        {_ok}/{_total}")
            for _s in sorted(_status_counts):
                if _s == "ok":
                    continue
                _note = (
                    "entries ungated (ablation arm)"
                    if _s == "disabled_by_config"
                    else "OOS IC entries BLOCKED"
                )
                lines.append(
                    f"    {_s}: {_status_counts[_s]}/{_total}  [{_note}]"
                )
            if _ok == 0 and _total > 0 and "disabled_by_config" not in _status_counts:
                lines.append(
                    "  WARNING: range model trained in ZERO windows — this "
                    "run says NOTHING about the range gate."
                )
            lines.append("-" * 60)

        # R20 #3: surface the meta-labeler gate per window-symbol, mirroring
        # the range block above. "ok" means OOS entries WERE meta-gated in
        # that window; anything else means they were not — either is fine,
        # but a study must SAY which apparatus actually ran.
        if self.meta_training_status:
            _m_counts: dict[str, int] = {}
            for _per_sym in self.meta_training_status.values():
                for _s in _per_sym.values():
                    _m_counts[_s] = _m_counts.get(_s, 0) + 1
            _m_total = sum(_m_counts.values())
            _m_ok = _m_counts.get("ok", 0)
            lines.append("  META-LABEL GATE (per window-symbol)")
            lines.append(f"  Trained ok:        {_m_ok}/{_m_total}  [OOS entries meta-gated]")
            for _s in sorted(_m_counts):
                if _s == "ok":
                    continue
                _note = (
                    "no meta gate (ablation arm)"
                    if _s == "disabled_by_config"
                    else "no meta gate (training failed/insufficient trades)"
                )
                lines.append(
                    f"    {_s}: {_m_counts[_s]}/{_m_total}  [{_note}]"
                )
            lines.append("-" * 60)

        if self.strategy_results:
            lines.append("  STRATEGY BREAKDOWN:")
            for strat, data in sorted(
                self.strategy_results.items(),
                key=lambda x: x[1].get("total_pnl", 0),
                reverse=True,
            ):
                lines.append(
                    f"    {strat:25s} | trades={data['trades']:3d} | "
                    f"win={data['win_rate']:.0%} | pnl=${data['total_pnl']:,.0f}"
                )

        if self.symbol_results:
            lines.append("  SYMBOL BREAKDOWN:")
            for sym, result in sorted(
                self.symbol_results.items(),
                key=lambda x: x[1].total_return,
                reverse=True,
            ):
                lines.append(
                    f"    {sym:8s} | return={result.total_return:.2%} | "
                    f"trades={result.total_trades:3d} | sharpe={result.sharpe_ratio:.2f}"
                )

        if self.optimization_status:
            studies = [(w, sym, st) for w, per in sorted(self.optimization_status.items())
                       for sym, st in sorted(per.items())]
            starved = [x for x in studies if not x[2]["usable"]]
            lines.append(f"  OPTIMIZATION: {len(studies) - len(starved)} of {len(studies)} "
                         f"per-window studies found a trading trial"
                         + ("" if not starved else
                            f"  --  {len(starved)} NEVER did (their 'best params' are not a "
                            f"result): " + ", ".join(f"w{w}/{sym}" for w, sym, _ in starved[:12])
                            + (" ..." if len(starved) > 12 else "")))
        if self.entry_funnel:
            lines.append("  ENTRY FUNNEL (days, summed over windows; why entries did not happen):")
            for sym in sorted(self.entry_funnel):
                f = self.entry_funnel[sym]
                days, taken = f.get("days_evaluated", 0), f.get("entries_taken", 0)
                top = sorted(((k, v) for k, v in f.items()
                              if k not in ("days_evaluated", "entries_taken")
                              and not k.startswith(("builder:", "would_reject:"))),
                             key=lambda kv: -kv[1])
                wr_ = sorted(((k[13:], v) for k, v in f.items() if k.startswith("would_reject:")),
                             key=lambda kv: -kv[1])
                det = sorted(((k[8:], v) for k, v in f.items() if k.startswith("builder:")),
                             key=lambda kv: -kv[1])
                lines.append(f"    {sym:6s} days={days} taken={taken} | " +
                             ", ".join(f"{k}={v}" for k, v in top[:6]))
                if wr_:
                    lines.append("           gates in observe mode would also have rejected: " +
                                 ", ".join(f"{k}={v}" for k, v in wr_[:5]))
                if det:
                    lines.append("           builder detail: " +
                                 ", ".join(f"{k}={v}" for k, v in det[:5]))

        lines.append("=" * 60)
        return "\n".join(lines)

    def equity_curve(self) -> pd.DataFrame:
        """Generate equity curve DataFrame from all trades across windows."""
        rows = []
        equity = self.initial_capital

        for w in self.windows:
            for t in w.backtest_result.trades:
                equity += t.get("pnl", 0)
                rows.append({
                    "date": t.get("exit_date", ""),
                    "equity": equity,
                    "pnl": t.get("pnl", 0),
                    "strategy": t.get("strategy", ""),
                    "symbol": t.get("symbol", ""),
                    "window": w.window_id,
                })

        if not rows:
            return pd.DataFrame(columns=["date", "equity", "pnl", "strategy", "symbol", "window"])

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)


def _json_numpy_default(o: object) -> object:
    """Custom JSON default handler that converts numpy scalars to Python native types."""
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _build_optuna_window_data(
    results: list,
    strategies: list[str],
    window_id: int,
    symbol: str,
    n_trials_requested: int = 50,
    patience: int = 0,
) -> dict:
    """Serialize Optuna trial data from one or more per-strategy OptimizationResult
    objects into a flat dict for the dashboard (Layer 2b).

    Returns ``{"trials": [...], "meta": {...}}``.
    """
    trials_out = []
    trial_offset = 0
    early_stopped = False
    stop_reason = ""

    for res in results:
        study = getattr(res, "study", None)
        if study is None:
            trial_offset += n_trials_requested
            continue
        for t in study.trials:
            dur = None
            if t.datetime_start and t.datetime_complete:
                dur = round((t.datetime_complete - t.datetime_start).total_seconds(), 1)
            trials_out.append({
                "number":       trial_offset + t.number,
                "state":        t.state.name,
                "value":        round(float(t.value), 4) if t.value is not None else None,
                "params":       t.params,
                "sharpe":       t.user_attrs.get("sharpe"),
                "win_rate":     t.user_attrs.get("win_rate"),
                "max_drawdown": t.user_attrs.get("max_drawdown"),
                "n_trades":     t.user_attrs.get("n_trades"),
                "duration_s":   dur,
                "intermediate": round(float(t.value), 4) if t.value is not None else None,
            })
        trial_offset += len(study.trials)
        if getattr(res, "early_stopped", False):
            early_stopped = True
            stop_reason = getattr(res, "stop_reason", "")

    if not early_stopped:
        n_pruned = sum(1 for t in trials_out if t["state"] == "PRUNED")
        suffix = f" {n_pruned} trial(s) pruned by MedianPruner." if n_pruned else ""
        stop_reason = f"Completed all {len(trials_out)} trials.{suffix}"

    study_names = [f"wf_w{window_id}_{symbol}_{s}" for s in strategies]
    meta = {
        "study_name":       ", ".join(study_names),
        "n_trials_requested": n_trials_requested * len(strategies),
        "n_trials_run":     len(trials_out),
        "status":           "early_stopped" if early_stopped else "completed",
        "stop_reason":      stop_reason,
        "sampler":          "TPESampler(seed=42)",
        "pruner":           "MedianPruner(n_warmup_steps=1)",
        "patience":         patience,
    }
    return {"trials": trials_out, "meta": meta}


_TIMESERIES_FEAT_KEYS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "sma_20", "sma_50", "bb_upper", "bb_lower", "bb_position",
    "atr_pct", "realized_vol_20", "iv_rank", "vix_level",
    "hurst_wavelet",
    # R12-C: sentiment_composite / put_call_ratio removed with the
    # sentiment/flow feature retirement (constant columns).
]


def _save_window_timeseries(
    test_df: "pd.DataFrame",
    full_context_df: "pd.DataFrame",
    predictor: "Any",
    range_predictor: "Any",
    vix_ctx: "dict | None",
    progress_dir: "Path",
    symbol: str,
    window_id: int,
    precomputed_features: "pd.DataFrame | None" = None,
) -> None:
    """Compute features + ML predictions for each test-period bar and append them
    to ``timeseries_bars.json`` in the experiment directory (Layer 2c).

    Features are computed ONCE on the full context+OOS window (O(1) instead of
    O(N²) FeatureEngine calls). Per-bar ML predictions use predict_from_features()
    which bypasses FeatureEngine entirely — no redundant feature computation.

    The file is keyed by experiment (one file per run, not per window) so the
    dashboard can load all windows in a single request.
    """
    try:
        import json as _json
        from ait.ml.features import FeatureEngine

        # Use pre-computed features if provided (shared from OOS Backtester setup),
        # otherwise compute once on full context+OOS. O(1) instead of O(N²).
        if precomputed_features is not None and not precomputed_features.empty:
            feat_df = precomputed_features
        else:
            feat_df = FeatureEngine().compute(full_context_df, market_context=vix_ctx)
        if feat_df.empty:
            return

        # Align to OOS period only via index reindex (safe against minor index gaps)
        oos_feat_df = feat_df.reindex(test_df.index)

        bars_out = []
        for ts, row in test_df.iterrows():
            date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            bar: dict = {
                "time":   date_str,
                "symbol": symbol,
                "open":   round(float(row.get("Open", 0)), 4),
                "high":   round(float(row.get("High", 0)), 4),
                "low":    round(float(row.get("Low", 0)), 4),
                "close":  round(float(row.get("Close", 0)), 4),
                "volume": int(row.get("Volume", 0)),
                "window": window_id,
            }

            feat_row = oos_feat_df.loc[ts] if ts in oos_feat_df.index else None

            # Features for this bar (O(1) lookup — no date-string scan)
            if feat_row is not None and not feat_row.isna().all():
                for k in _TIMESERIES_FEAT_KEYS:
                    v = feat_row.get(k) if k in feat_row.index else None
                    bar[k] = round(float(v), 4) if v is not None and not _isnan(v) else None
                vol20 = feat_row.get("volume_sma_20_ratio")
                bar["volume_ratio"] = round(float(vol20), 4) if vol20 is not None and not _isnan(vol20) else None

            # ML predictions via predict_from_features — O(1), no FeatureEngine re-run
            bar["dir_class"] = None
            bar["dir_conf"] = None
            bar["p_up"] = None
            bar["p_down"] = None
            bar["p_neutral"] = None
            bar["range_prob"] = None
            bar["meta_take"] = None

            if feat_row is not None:
                try:
                    if predictor is not None and predictor.is_trained:
                        sig = predictor.predict_from_features(feat_row, symbol=symbol)
                        if sig is not None:
                            bar["dir_class"] = sig.direction.value if hasattr(sig.direction, "value") else str(sig.direction)
                            bar["dir_conf"] = round(float(sig.confidence), 4)
                            bar["p_up"] = round(float(sig.probabilities.get("bullish", 0)), 4)
                            bar["p_down"] = round(float(sig.probabilities.get("bearish", 0)), 4)
                            bar["p_neutral"] = round(float(sig.probabilities.get("neutral", 0)), 4)
                except Exception:
                    pass
                try:
                    if range_predictor is not None and range_predictor.is_trained:
                        rp = range_predictor.predict_from_features(feat_row, symbol=symbol)
                        if rp is not None:
                            bar["range_prob"] = round(float(rp.probability_in_range), 4)
                except Exception:
                    pass

            bars_out.append(bar)

        if not bars_out:
            return

        # Merge into the experiment-wide timeseries file (append / replace window entries)
        ts_path = progress_dir / "timeseries_bars.json"
        existing: list = []
        if ts_path.exists():
            try:
                existing = _json.loads(ts_path.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        existing = [b for b in existing if not (b.get("window") == window_id and b.get("symbol") == symbol)]
        existing.extend(bars_out)
        existing.sort(key=lambda b: (b.get("time", ""), b.get("symbol", "")))
        ts_path.write_text(_json.dumps(existing), encoding="utf-8")

    except Exception:
        pass  # timeseries save is best-effort; never break the backtest


def _isnan(v: object) -> bool:
    try:
        import math
        return math.isnan(float(v))  # type: ignore[arg-type]
    except Exception:
        return False


def _window_task_mp(args: tuple) -> tuple:
    """Worker for ProcessPoolExecutor window-level parallelism.

    Must live at module level (not a closure) so it is picklable.
    Each call creates a fresh WalkForwardBacktester with no cross-window state:
    warm-starting and BacktestLearner adaptation are disabled.

    Accepts an optional pre-trained range_predictor per symbol (dict keyed by
    symbol) so statistical models are not re-trained inside the worker.  When
    provided, _run_single_window skips _train_window_range_model entirely.
    """
    (window_id, train_start, train_end, test_start, test_end,
     data, vix_full, spy_full, config, symbols, strategies, db_path, progress_dir,
     pretrained_range, table_prefix, vlmc_enabled, research_calendar) = args

    bt = WalkForwardBacktester(
        symbols=symbols,
        strategies=strategies,
        config=config,
        db_path=db_path,
        progress_dir=progress_dir,
        table_prefix=table_prefix,   # W65: a worker used to lose the research prefix
    )
    bt._vlmc_enabled = vlmc_enabled
    bt._research_calendar = research_calendar
    wr, best_params, meta_status = bt._run_single_window(
        window_id=window_id,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        data=data,
        vix_full=vix_full,
        spy_full=spy_full,
        learner=None,
        prev_best_params=None,
        prev_oos=None,
        pretrained_range=pretrained_range,
    )
    return (window_id, wr, best_params, meta_status,
            bt._funnel_by_window.get(window_id, {}),
            bt._opt_status_by_window.get(window_id, {}))


class WalkForwardBacktester:
    """Walk-forward backtester with multi-symbol, multi-strategy support.

    Slides a train/test window across history:
    1. Train ML model on [train_start, train_end]
    2. Skip gap_days (purge)
    3. Backtest on [test_start, test_end]
    4. Slide forward by step_days and repeat

    Supports:
    - Multiple symbols (aggregated or per-symbol results)
    - Multiple strategies (with per-strategy breakdown)
    - Buy-and-hold benchmark comparison
    - Equity curve generation
    """

    def __init__(
        self,
        symbols: list[str],
        strategies: list[str],
        config: WalkForwardConfig | None = None,
        db_path: "Path | None" = None,
        progress_dir: "Path | None" = None,
        table_prefix: str = "",
    ) -> None:
        self._symbols = symbols
        self._strategies = strategies
        self._config = config or WalkForwardConfig()
        self._db_path = db_path
        self._table_prefix = table_prefix
        # W65: set by _check_vlmc_coverage() once the data span is known. True keeps
        # the legacy behaviour (a db_path means "use the intraday store").
        self._vlmc_enabled = True
        # W66: {window_id: {symbol: entry_funnel}} — kept for EVERY window, including the
        # zero-trade windows whose WindowResult is None (they are exactly where the
        # funnel explains the emptiness).
        self._funnel_by_window: dict[int, dict[str, dict[str, int]]] = {}
        # W67: {window_id: {symbol: {"usable": bool, "best_value": float, "best_n_trades": int}}}
        # for every per-window Optuna study, so a run whose studies never found a trading
        # trial says so in its summary instead of looking like an optimized result.
        self._opt_status_by_window: dict[int, dict[str, dict]] = {}
        # W68: {symbol: IV-rank series by date} over each symbol's FULL history (live's 252-obs
        # lookback), filled lazily by _ensure_iv_rank_series; empty when the gate is off.
        self._iv_rank_series: dict[str, pd.Series] = {}
        # W65: research macro calendar (built in run() when macro_event_types is set).
        self._research_calendar = None
        self._progress_dir = Path(progress_dir) if progress_dir else None
        self._global_best_params: dict | None = None
        self._global_best_score: float = -1.0
        # R20b review follow-up: load settings ONCE for the whole run instead
        # of every per-window Backtester()/StrategyOptimizer() construction
        # independently re-reading + re-validating config.yaml (same
        # redundant-reload class the settings= param on Backtester/
        # StrategyOptimizer was built to eliminate for Optuna trials —
        # WalkForwardConfig.__post_init__ already loads its own copy above
        # for its config-backed fields, this is the SEPARATE copy every
        # window's Backtester/StrategyOptimizer construction used to trigger).
        try:
            from ait.config.settings import load_settings as _ls3
            self._settings = _ls3()
        except Exception:  # noqa: BLE001 — no config.yaml -> per-field defaults downstream
            self._settings = None
        # R12-C: enable_msgarch/enable_oujump flags removed — the GARCH family
        # is retired to deprecated/research/; range models are ML-only.

    async def run(self, data: dict[str, pd.DataFrame] | None = None) -> WalkForwardResult:
        """Run walk-forward backtest.

        Args:
            data: Pre-loaded data as {symbol: OHLCV DataFrame}.
                  If None, fetches from Yahoo Finance.
        """
        if data is None:
            data = await self._fetch_data()

        if not data:
            log.error("no_data_for_backtest")
            return WalkForwardResult(initial_capital=self._config.initial_capital)

        # R16 #5 (defensive): normalize any tz-aware symbol frame to tz-naive.
        # VIX/SPY context frames are normalized below, and reindexing a
        # tz-aware frame against a naive index raises ("Cannot compare dtypes
        # datetime64[ns] and datetime64[ns, America/New_York]") on the FIRST
        # window. Shallow-copy so callers' frames are untouched.
        _normalized: dict[str, pd.DataFrame] = {}
        for _sym, _df in data.items():
            if isinstance(_df.index, pd.DatetimeIndex) and _df.index.tz is not None:
                _df = _df.copy(deep=False)
                _df.index = _df.index.tz_localize(None)
            _normalized[_sym] = _df
        data = _normalized

        # R16 #3: per-window range-model training status, recorded for EVERY
        # window (zero-trade windows included) and surfaced in the summary.
        self._range_status_by_window: dict[int, dict[str, str]] = {}
        # R20 #3: same for the window MetaLabeler (sequential mode records
        # here; parallel workers report through WindowResult.meta_labeler_status
        # and are merged below).
        self._meta_status_by_window: dict[int, dict[str, str]] = {}

        # Load VIX and SPY data for the full backtest period (yfinance fallback when not in DB).
        # VIX: Priority-2 IV proxy and live feature for the MetaLabeler.
        # SPY: cross-asset features (relative strength, momentum, correlation) for the ML model.
        from ait.data.market_data import load_daily_ohlcv as _load_ohlcv
        try:
            _ctx_days = self._context_days()
            _vix_full: pd.DataFrame = _load_ohlcv(
                "^VIX", days=_ctx_days, db_path=self._db_path,
                table_prefix=self._table_prefix,
            )
            # yfinance may return tz-aware index; normalize to tz-naive to match OHLCV data
            if not _vix_full.empty and _vix_full.index.tz is not None:
                _vix_full.index = _vix_full.index.tz_localize(None)
        except Exception:
            _vix_full = pd.DataFrame()
        try:
            _spy_full: pd.DataFrame = _load_ohlcv(
                "SPY", days=_ctx_days, db_path=self._db_path,
                table_prefix=self._table_prefix,
                start_date=self._config.history_start_date,
            )
            if not _spy_full.empty and _spy_full.index.tz is not None:
                _spy_full.index = _spy_full.index.tz_localize(None)
        except Exception:
            _spy_full = pd.DataFrame()

        # Generate walk-forward windows
        data = self._apply_research_calendar(data)
        windows = self._generate_windows(data)
        windows = self._limit_windows(windows)
        self._check_vlmc_coverage(data)
        if len(windows) < 3:
            _spans = {k: (str(v.index[0].date()), str(v.index[-1].date()), len(v))
                      for k, v in data.items() if not v.empty}
            log.error("walk_forward_too_few_windows", windows=len(windows),
                      minimum=3, spans=_spans,
                      note="the common date range of all symbols is too short for "
                           "train_days + gap + test_days; run the research backfill "
                           "(scripts/backfill_historical_data.py --mode research) or "
                           "lengthen history (backtest.history_start_date / history_days)")
        log.info("walk_forward_windows", count=len(windows))

        # Run each window — parallel or sequential depending on optimize_n_jobs
        window_results = []

        if self._config.optimize_n_jobs > 1:
            # Window-level parallelism via ProcessPoolExecutor (bypasses the GIL).
            # Each worker process creates a fresh WalkForwardBacktester, so there is
            # no cross-window state: warm-starting and BacktestLearner adaptation are
            # disabled.  Progress JSON files are written independently per window with
            # no conflicts (each window has a unique file name).
            #
            # Range models are pre-trained sequentially here in the parent
            # process BEFORE the ProcessPoolExecutor launches, eliminating CPU
            # contention with Optuna workers. (R12-C: this used to also train
            # the MS-GARCH/OU statistical members — now ML-only.)
            pretrained_range_by_window: list[dict] = self._pretrain_range_models(
                windows, data, _vix_full
            )
            # R16 #3: statuses are known here in the parent process (workers
            # cannot report back through instance state).
            for _i, _per_symbol in enumerate(pretrained_range_by_window):
                self._range_status_by_window[_i + 1] = {
                    _sym: _tup[1] for _sym, _tup in _per_symbol.items()
                }

            from concurrent.futures import ProcessPoolExecutor

            args_list = [
                (
                    i + 1, train_start, train_end, test_start, test_end,
                    data, _vix_full, _spy_full, self._config,
                    self._symbols, self._strategies,
                    self._db_path, self._progress_dir,
                    pretrained_range_by_window[i],
                    self._table_prefix, self._vlmc_enabled, self._research_calendar,
                )
                for i, (train_start, train_end, test_start, test_end) in enumerate(windows)
            ]
            with ProcessPoolExecutor(max_workers=self._config.optimize_n_jobs) as executor:
                mp_results = list(executor.map(_window_task_mp, args_list))

            for _wid, wr, curr_best_params, _meta_status, _fun, _opt in sorted(mp_results, key=lambda x: x[0]):
                self._funnel_by_window[_wid] = _fun
                self._opt_status_by_window[_wid] = _opt
                # R20b review follow-up: merge the per-symbol meta-labeler
                # status independently of whether the window produced trades
                # — a zero-trade parallel window (wr is None) is precisely
                # where a failed/disabled gate needs to stay visible in the
                # summary instead of silently vanishing with the WindowResult.
                if _meta_status:
                    self._meta_status_by_window.setdefault(_wid, {}).update(_meta_status)
                if wr is not None:
                    window_results.append(wr)
                    if curr_best_params is not None:
                        br = wr.backtest_result
                        score = br.win_rate * (min(1.0, br.total_trades / 5) ** 0.5)
                        if score > self._global_best_score:
                            self._global_best_score = score
                            self._global_best_params = curr_best_params

        else:
            # Sequential: warm-starting and BacktestLearner adaptation are active.
            learner = BacktestLearner(base_confidence=self._config.min_confidence)
            prev_best_params: dict | None = None
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
                prev_oos = window_results[-1].backtest_result if window_results else None
                wr, curr_best_params, _meta_status = self._run_single_window(
                    window_id=i + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    data=data,
                    vix_full=_vix_full,
                    spy_full=_spy_full,
                    learner=learner,
                    prev_best_params=prev_best_params,
                    prev_oos=prev_oos,
                )
                if wr is not None:
                    window_results.append(wr)
                    if curr_best_params is not None:
                        br = wr.backtest_result
                        score = br.win_rate * (min(1.0, br.total_trades / 5) ** 0.5)
                        if score > self._global_best_score:
                            self._global_best_score = score
                            self._global_best_params = curr_best_params
                    learning_summary = learner.process_window(wr.backtest_result.trades, i + 1)
                    if learning_summary.get("adaptations"):
                        log.info(
                            "self_learning_adapted",
                            window=i + 1,
                            adaptations=[a["change"] for a in learning_summary["adaptations"]],
                        )
                    if curr_best_params is not None:
                        prev_best_params = curr_best_params

            log.info("self_learning_final_state", summary=learner.summary())

        # R20 #3: merge meta-labeler statuses — sequential windows recorded
        # into the instance dict; parallel workers can only report through the
        # WindowResult they return (zero-trade parallel windows are therefore
        # absent, a documented limitation matching how their other per-symbol
        # metadata is lost too).
        _meta_status_all: dict[int, dict[str, str]] = {
            k: dict(v) for k, v in self._meta_status_by_window.items()
        }
        for _wr in window_results:
            if _wr.meta_labeler_status:
                _meta_status_all.setdefault(_wr.window_id, {}).update(
                    _wr.meta_labeler_status
                )

        # Build aggregated results
        result = WalkForwardResult(
            windows=window_results,
            initial_capital=self._config.initial_capital,
            config=self._config,
            range_training_status=dict(self._range_status_by_window),
            meta_training_status=_meta_status_all,
            entry_funnel=self._aggregate_funnel(),
            optimization_status=self._opt_status_by_window,
        )

        # Compute per-symbol results
        result.symbol_results = self._compute_symbol_results(window_results, data)

        # Compute per-strategy breakdown
        result.strategy_results = self._compute_strategy_results(window_results)

        log.info(
            "walk_forward_complete",
            windows=len(window_results),
            total_trades=result.total_trades,
            total_return=f"{result.total_return:.2%}",
            sharpe=f"{result.sharpe_ratio:.2f}",
        )

        return result

    @staticmethod
    def _resolve_oos_range_min_conf(
        range_predictor, status: str, configured: float
    ) -> float:
        """R16 #3/#4: decide the OOS range-gate threshold for a window.

        - Training ATTEMPTED but failed (any status other than "ok" /
          "disabled_by_config" with no predictor): return 1.0 — combined with
          the engine-side block (predictor None + threshold >= 1.0 skips the
          entry) this HONORS the "range model absent -> block entries"
          contract that engine.py previously broke by skipping the gate.
        - Training disabled by config (gate-stack-only ablation arm): run
          UNGATED at the configured threshold — that is the arm's documented
          semantics.
        - Trained ok: configured threshold.
        """
        if range_predictor is None and status not in ("ok", "disabled_by_config"):
            return 1.0
        return configured

    def _run_single_window(
        self,
        window_id: int,
        train_start: "date",
        train_end: "date",
        test_start: "date",
        test_end: "date",
        data: "dict[str, pd.DataFrame]",
        vix_full: "pd.DataFrame",
        spy_full: "pd.DataFrame | None" = None,
        learner: "BacktestLearner | None" = None,
        prev_best_params: "dict | None" = None,
        prev_oos: "BacktestResult | None" = None,
        pretrained_range: "dict | None" = None,
    ) -> "tuple[WindowResult | None, dict | None, dict]":
        """Run optimization, ML training, and backtest for one window.

        Returns (window_result, best_params, meta_labeler_status). window_result
        is None when no trades are generated, but meta_labeler_status is always
        populated (per-symbol) — parallel workers build a fresh instance per
        window and only communicate back through this return value, so a
        zero-trade window (precisely where a failed/disabled gate needs
        visibility) must not be silently dropped along with the WindowResult.
        Pass learner=None in parallel mode — the method then uses static
        config defaults (no cross-window adaptation).
        """
        log.info(
            "running_window",
            window=window_id,
            train=f"{train_start} to {train_end}",
            test=f"{test_start} to {test_end}",
        )
        self._ensure_iv_rank_series(data)

        if learner is not None:
            learned_config = learner.get_config()
            effective_min_conf = learned_config["min_confidence"]
            active_strategies = [
                s for s in self._strategies if learner.is_strategy_enabled(s)
            ]
            if not active_strategies:
                active_strategies = self._strategies
        else:
            effective_min_conf = self._config.min_confidence
            active_strategies = list(self._strategies)

        all_symbol_trades: list = []
        model_accuracy = 0.0
        active_symbols = 0
        curr_best_params: dict | None = None
        _optuna_meta: dict | None = None
        _model_weights: dict = {}  # initialised here so it's always bound even if loop is empty
        _range_status_local: dict[str, str] = {}  # R16 #3: symbol -> range training status
        _meta_status_local: dict[str, str] = {}   # R20 #3: symbol -> meta-labeler status
        # W-capital: every symbol's engine run is seeded with the WHOLE pool so its
        # own affordability gate sees the budget live sees (a percentage of the
        # total NLV, not of a sleeve -- splitting by symbol count makes condors
        # unaffordable). The pool itself is enforced AFTER the per-symbol runs by
        # allocate_pool (backtesting/pool.py), which replays all symbols' trades
        # chronologically against ONE equity. Supersedes the BT-H3 per-symbol sleeves.
        per_symbol_capital = self._config.initial_capital

        for symbol, df in data.items():
            if learner is not None and not learner.is_symbol_allowed(symbol):
                log.info("learner_skipping_symbol", symbol=symbol, window=window_id)
                continue

            train_df = df[str(train_start):str(train_end)]
            test_df = df[str(test_start):str(test_end)]

            if len(train_df) < 50 or len(test_df) < 5:
                continue

            active_symbols += 1

            # Train ML models BEFORE Optuna so every trial evaluation uses the same
            # entry signal as the final OOS backtest (Change C, Exp 10).
            # Previously models were trained after optimization, so Optuna evaluated
            # params against _simple_direction fallback with no range gate — a
            # different signal than what ran in OOS.
            predictor = None if not self._config.train_window_models                 else self._train_window_model(
                train_df, symbol, window_id,
                vix_full=vix_full, spy_full=spy_full,
            )

            # Use pre-trained range predictor when provided (parallel mode with
            # _pretrain_range_models()).  Fall back to training in-process when
            # running sequentially or when pre-training was skipped.
            # R16 #4: train_window_models=False disables the range predictor
            # too (previously only the direction predictor honored the flag,
            # so a "gate_only" ablation arm could run WITH a range gate).
            if not self._config.train_window_models:
                _range_result = (None, "disabled_by_config", 0.05)
            elif pretrained_range is not None and symbol in pretrained_range:
                _range_result = pretrained_range[symbol]
            else:
                _range_result = self._train_window_range_model(
                    train_df, symbol, window_id,
                    # PR#7 review 2026-08-25: was self._config.max_hold_days
                    # (30) — trained 30-day containment labels for ~9-day
                    # reachable trades. See _range_label_horizon.
                    max_hold_days=self._range_label_horizon(),
                )
            # Guard: tests may monkeypatch _train_window_range_model to return None
            if _range_result is None or not isinstance(_range_result, tuple):
                range_predictor, _range_model_status, _range_threshold = None, "skipped", 0.05
            else:
                range_predictor, _range_model_status, _range_threshold = _range_result

            # W78: per-horizon range models for this (window, symbol). The model trained /
            # pre-trained above is the BASE bucket (so the default configuration is
            # unchanged); any other entry DTE (a trial, or a window config that differs) is
            # trained lazily and cached here.
            _base_bucket = self._range_bucket()
            _range_bank = RangeModelBank(
                train_fn=lambda _b, _tr=train_df, _sym=symbol, _wid=window_id:
                    self._train_range_bucket(_tr, _sym, _wid, _b),
                bucket_fn=self._range_bucket,
            )
            _range_bank.seed(_base_bucket,
                             (range_predictor, _range_model_status, _range_threshold))

            # R16 #3: record status for the WindowResult and run summary
            # (instance dict is guarded — tests may call _run_single_window
            # directly, bypassing run()).
            _range_status_local[symbol] = _range_model_status
            if hasattr(self, "_range_status_by_window"):
                self._range_status_by_window.setdefault(window_id, {})[symbol] = _range_model_status

            # Build market context for the training slice (VIX + SPY) so features_cache
            # has real cross-asset values instead of neutral defaults.
            _vix_train_ctx_opt: dict | None = None
            _ctx_opt: dict = {}
            if not vix_full.empty:
                _vix_t_opt = vix_full.reindex(train_df.index, method="ffill").dropna(how="all")
                if not _vix_t_opt.empty:
                    _ctx_opt["vix"] = _vix_t_opt
            if spy_full is not None and not spy_full.empty:
                _spy_t_opt = spy_full.reindex(train_df.index, method="ffill").dropna(how="all")
                if not _spy_t_opt.empty:
                    _ctx_opt["spy"] = _spy_t_opt
            if _ctx_opt:
                _vix_train_ctx_opt = _ctx_opt

            window_cfg = self._config
            if self._config.optimize_per_window:
                new_cfg, best_params, _optuna_meta = self._optimize_window_params(
                    train_df, symbol, window_id, active_strategies,
                    prior_oos_result=prev_oos,
                    prior_best_params=prev_best_params,
                    range_predictor=range_predictor,
                    range_bank=_range_bank,
                    vix_ctx=_vix_train_ctx_opt,
                )
                window_cfg = new_cfg or self._config
                if best_params is not None:
                    curr_best_params = best_params
            # W78: the OOS run is gated by the model for the window config's OWN entry DTE
            # (None -> the base bucket, so today's behaviour).
            _oos_bucket, _oos_range = self._pick_range_model(_range_bank, window_cfg)
            if _oos_bucket != _base_bucket:
                range_predictor, _range_model_status, _range_threshold = _oos_range
                _range_status_local[symbol] = _range_model_status
            if predictor and predictor.is_trained:
                model_accuracy = max(
                    model_accuracy,
                    max(predictor.cv_scores.values()) if predictor.cv_scores else 0.0,
                )

            # Collect per-model fitted weights + CV scores for window JSON export.
            _model_weights: dict = {}
            if range_predictor is not None and range_predictor.fitted_weights:
                rp_sym = getattr(range_predictor, "_symbol_models", {}).get(symbol, {})
                # R12-C: GARCH/MS-GARCH/OU-Jump metadata keys removed — the
                # family is retired to deprecated/research/ and RangePredictor
                # no longer emits their states.
                _model_weights["range_predictor"] = {
                    "status":          _range_model_status,
                    "threshold_pct":   round(_range_threshold, 4),
                    "fitted_weights":  dict(range_predictor.fitted_weights),
                    "cv_scores":       dict(rp_sym.get("cv_scores", {})),
                    "in_range_rate":   rp_sym.get("in_range_rate"),
                }
            else:
                _model_weights["range_predictor"] = {
                    "status":        _range_model_status,
                    "threshold_pct": round(_range_threshold, 4),
                }
            if predictor is not None and predictor.fitted_weights:
                _model_weights["direction_predictor"] = {
                    "fitted_weights": dict(predictor.fitted_weights),
                    "cv_scores": dict(predictor.cv_scores or {}),
                }

            _meta_artifact = (
                self._progress_dir / f"window_{window_id:03d}" / symbol
                if self._progress_dir else None
            )
            _vix_train_ctx: dict | None = None
            _ctx_train: dict = {}
            if not vix_full.empty:
                _vix_t = vix_full.reindex(train_df.index, method="ffill").dropna(how="all")
                if not _vix_t.empty:
                    _ctx_train["vix"] = _vix_t
            if spy_full is not None and not spy_full.empty:
                _spy_t = spy_full.reindex(train_df.index, method="ffill").dropna(how="all")
                if not _spy_t.empty:
                    _ctx_train["spy"] = _spy_t
            if _ctx_train:
                _vix_train_ctx = _ctx_train

            # R20 #3: honor train_window_models for the THIRD window model.
            # R16 #4 gated direction + range on this flag but the MetaLabeler
            # was still trained (shadow backtest) and passed to the OOS
            # Backtester, which gates entries whenever it is_trained — so a
            # "gate-stack-only / ML-free" ablation arm could silently run
            # WITH a trained XGB entry gate (the mislabeled-arm failure mode
            # of the vacuous 08-03 ablation, recurring). False => no meta
            # training AND no meta gating; status recorded like range's.
            if not self._config.train_window_models:
                meta_labeler = None
                _meta_status = "disabled_by_config"
            else:
                meta_labeler = self._train_window_meta_labeler(
                    train_df=train_df,
                    symbol=symbol,
                    window_id=window_id,
                    predictor=predictor,
                    window_cfg=window_cfg,
                    artifact_dir=_meta_artifact,
                    vix_ctx=_vix_train_ctx,
                )
                _meta_status = "ok" if meta_labeler is not None else "not_trained"
            _meta_status_local[symbol] = _meta_status
            if hasattr(self, "_meta_status_by_window"):
                self._meta_status_by_window.setdefault(window_id, {})[symbol] = _meta_status

            # Prepend training data context so ML features can be computed.
            # iv_rank uses vol_20.rolling(252) — needs 252 bars to be meaningful.
            # Capped at actual train_df length so early windows still work.
            context_bars = min(252, len(train_df))
            test_with_context = pd.concat([train_df.tail(context_bars), test_df])

            if learner is not None:
                strategy_mults = [
                    learner.get_strategy_multiplier(s) for s in active_strategies
                ]
                avg_mult = sum(strategy_mults) / len(strategy_mults) if strategy_mults else 1.0
                effective_position_size = self._config.position_size_pct * min(avg_mult, 2.0)
            else:
                effective_position_size = self._config.position_size_pct

            confidence_threshold = (
                window_cfg.min_confidence
                if self._config.optimize_per_window
                else effective_min_conf
            )

            _vix_ctx: dict | None = None
            _ctx_oos: dict = {}
            if not vix_full.empty:
                _vix_w = vix_full.reindex(test_with_context.index, method="ffill").dropna(how="all")
                if not _vix_w.empty:
                    _ctx_oos["vix"] = _vix_w
            if spy_full is not None and not spy_full.empty:
                _spy_w = spy_full.reindex(test_with_context.index, method="ffill").dropna(how="all")
                if not _spy_w.empty:
                    _ctx_oos["spy"] = _spy_w
            if _ctx_oos:
                _vix_ctx = _ctx_oos

            # When the range model FAILED to train, block all OOS IC entries by
            # raising range_min_confidence to an unreachable value. Without a range
            # gate, iron condors have no entry signal and should not trade.
            # (The engine now enforces this contract too — R16 #3: predictor
            # None + threshold >= 1.0 blocks the entry instead of silently
            # skipping the gate.) "disabled_by_config" (R16 #4 ablation arm)
            # deliberately runs UNGATED at the configured threshold.
            # R20b: stub-config fallback is the CONFIG contract
            # (ml.range_min_confidence, default 0.65), no longer the retired
            # 0.55 literal. Real WalkForwardConfigs resolve in __post_init__
            # and never hit this fallback.
            _cfg_range_min = getattr(window_cfg, "range_min_confidence", None)
            if _cfg_range_min is None:
                _cfg_range_min = float(MLConfig().range_min_confidence)
            _oos_range_min_conf = self._resolve_oos_range_min_conf(
                range_predictor, _range_model_status, _cfg_range_min,
            )
            if _oos_range_min_conf >= 1.0:
                log.warning(
                    "range_model_absent_blocking_oos",
                    window=window_id, symbol=symbol,
                    status=_range_model_status,
                    action="setting range_min_confidence=1.0 to block OOS IC entries",
                )

            # Pre-compute OOS feature matrix once — shared by the OOS Backtester,
            # timeseries export, and both model evaluators.  Eliminates ~120
            # redundant FeatureEngine calls per window (2 per bar × 60 bars in
            # the old path where no features_cache was passed to the Backtester).
            _oos_feat_cache: "pd.DataFrame | None" = None
            try:
                from ait.ml.features import FeatureEngine as _FE_oos
                _c = _FE_oos().compute(test_with_context, market_context=_vix_ctx)
                _oos_feat_cache = _c if not _c.empty else None
            except Exception:
                pass

            # Change D: pass intraday_store to OOS Backtester so entries are
            # gated by the config-backed ET entry window (settings.backtest,
            # default 10:30–15:30 — R20 #5b: the old comment's "09:30–15:30"
            # described engine-default drift, not configuration) and limit-fill
            # simulation, matching the live-trading execution model. Also
            # enables capture of limit_price and fill_time on every OOS trade.
            _oos_intraday_store = self._make_intraday_store()

            bt = Backtester(
                data=test_with_context,
                context_bars=context_bars,
                strategies=active_strategies,
                initial_capital=per_symbol_capital,
                features_cache=_oos_feat_cache,
                commission_per_contract=window_cfg.commission_per_contract,
                slippage_pct=window_cfg.slippage_pct,
                position_size_pct=effective_position_size,
                budget_aware_wings=window_cfg.budget_aware_wings,
                max_contracts_per_trade=window_cfg.max_contracts_per_trade,
                condor_any_direction=window_cfg.condor_any_direction,
                friction_model=window_cfg.friction_model,
                friction_entry_capture=window_cfg.friction_entry_capture,
                friction_exit_capture=window_cfg.friction_exit_capture,
                friction_stress_multiplier=window_cfg.friction_stress_multiplier,
                gates_mode=window_cfg.gates_mode,
                iv_rank_gate=window_cfg.iv_rank_gate,
                iv_rank_series=self._iv_rank_series.get(symbol),
                fractal_veto_spread=window_cfg.fractal_veto_spread,
                penalty_on_range_prob=window_cfg.penalty_on_range_prob,
                economic_calendar=self._research_calendar,
                stop_loss_pct=window_cfg.stop_loss_pct,
                profit_target_pct=window_cfg.profit_target_pct,
                max_hold_days=window_cfg.max_hold_days,
                min_confidence=confidence_threshold,
                trailing_stop_enabled=window_cfg.trailing_stop_enabled,
                trailing_stop_pct=window_cfg.trailing_stop_pct,
                breakeven_trigger_pct=window_cfg.breakeven_trigger_pct,
                predictor=predictor,
                range_predictor=range_predictor,
                range_min_confidence=_oos_range_min_conf,
                hurst_regime_threshold=getattr(window_cfg, "hurst_regime_threshold", 0.20),
                hurst_regime_penalty=getattr(window_cfg, "hurst_regime_penalty", 0.10),
                hurst_hard_veto_multiplier=getattr(window_cfg, "hurst_hard_veto_multiplier", 1.5),
                multifractal_max_width=getattr(window_cfg, "multifractal_max_width", 0.50),
                iv_rank_rise_threshold=getattr(window_cfg, "iv_rank_rise_threshold", 0.30),
                pct_from_60d_high_threshold=getattr(window_cfg, "pct_from_60d_high_threshold", -1.0),
                min_edge_over_baseline=getattr(window_cfg, "min_edge_over_baseline", 0.05),
                iv_floor=window_cfg.iv_floor,
                wing_floor_dollars=window_cfg.wing_floor_dollars,
                wing_k=window_cfg.wing_k,
                # R6 parity knobs — flow through so run_backtest CLI overrides reach the engine
                credit_loss_limit_mult=getattr(window_cfg, "credit_loss_limit_mult", None),
                ic_min_credit=getattr(window_cfg, "ic_min_credit", None),
                ic_min_credit_width=getattr(window_cfg, "ic_min_credit_width", None),
                macro_event_gate=getattr(window_cfg, "macro_event_gate", True),
                # R16 #7: None -> engine resolves from loaded settings (live parity)
                pre_event_blackout_days=getattr(window_cfg, "pre_event_blackout_days", None),
                credit_tp_ladder=getattr(window_cfg, "credit_tp_ladder", None),
                entry_dte=getattr(window_cfg, "entry_dte", None),
                # W79a: the short-DTE exit knobs and the (possibly searched) delta reach the run
                expiry_guard_dte=getattr(window_cfg, "expiry_guard_dte", None),
                max_hold_fraction=getattr(window_cfg, "max_hold_fraction", None),
                flatten_minutes_before_close=getattr(window_cfg, "flatten_minutes_before_close", None),
                delta_short=getattr(window_cfg, "delta_short", None),
                db_path=self._db_path,  # W52a: the engine reads THIS store
                post_event_blackout_days=getattr(window_cfg, "post_event_blackout_days", None),
                post_event_dates=getattr(window_cfg, "post_event_dates", None),
                # R16 #1: NEVER score an OOS window with the live (future-
                # trained) models/ensemble.pkl. If the window model failed to
                # train, the window trades ungated — the documented fallback.
                allow_live_model_fallback=False,
                delta_iv_scale=window_cfg.delta_iv_scale,
                max_concurrent_positions=window_cfg.max_concurrent_positions,
                max_entry_vol_annual=window_cfg.max_entry_vol_annual,
                spread_base=window_cfg.spread_base,
                spread_iv_sensitivity=window_cfg.spread_iv_sensitivity,
                spread_dte_sensitivity=window_cfg.spread_dte_sensitivity,
                spread_cap=window_cfg.spread_cap,
                meta_labeler=meta_labeler,
                market_context=_vix_ctx,
                intraday_store=_oos_intraday_store,
                # R20 #4: forward the intraday execution knobs — they were
                # declared on WalkForwardConfig but never passed, so the
                # engine always ran its own defaults regardless of config.
                # None defers to the engine's settings.backtest resolution.
                scan_interval_minutes=getattr(window_cfg, "scan_interval_minutes", None),
                entry_window_start_et=getattr(window_cfg, "entry_window_start_et", None),
                entry_window_end_et=getattr(window_cfg, "entry_window_end_et", None),
                limit_order_timeout_bars=getattr(window_cfg, "limit_order_timeout_bars", None),
                symbol=symbol,
                # R20b review follow-up: reuse settings loaded once in
                # __init__ instead of every window's OOS Backtester
                # independently re-reading + re-validating config.yaml.
                settings=self._settings,
            )
            result = bt.run()
            self._funnel_by_window.setdefault(window_id, {})[symbol] = dict(result.entry_funnel)

            # Evaluate range predictor on OOS data — calibration bins + Brier/AUROC
            # per model (train/test generalisation gap visible in the dashboard).
            if range_predictor is not None and range_predictor.is_trained:
                _oos_scores = self._evaluate_range_model_oos(
                    range_predictor, test_df, symbol,
                    threshold_pct=_range_threshold,
                    # PR#7 review 2026-08-25: OOS eval must score the same
                    # horizon the model was trained on (see
                    # _range_label_horizon) — was max_hold_days (30).
                    horizon_days=_oos_bucket,
                    precomputed_features=_oos_feat_cache,
                )
                if _oos_scores and "range_predictor" in _model_weights:
                    _model_weights["range_predictor"]["oos_scores"] = _oos_scores

            # Evaluate directional predictor on OOS data — confidence calibration
            # bins per member (predicted confidence vs actual accuracy).
            if predictor is not None and predictor.is_trained:
                _dir_oos = self._evaluate_direction_model_oos(
                    predictor, test_df, symbol,
                    horizon_days=self._config.max_hold_days,
                    precomputed_features=_oos_feat_cache,
                )
                if _dir_oos and "direction_predictor" in _model_weights:
                    _model_weights["direction_predictor"]["oos_scores"] = _dir_oos

            for t in result.trades:
                t["symbol"] = symbol
                # W-capital: allocate_pool sizes with the SAME effective fraction
                # this symbol's engine used (learner multipliers included); the
                # allocator strips this private key from its output.
                t["_pool_size_pct"] = effective_position_size
            all_symbol_trades.extend(result.trades)
            # (sorted chronologically after the loop — BT-M9: drawdown was
            # computed on symbol-grouped, non-chronological order)

            # Save per-bar timeseries for dashboard (Layer 2c)
            if self._progress_dir is not None:
                _save_window_timeseries(
                    test_df=test_df,
                    full_context_df=test_with_context,
                    predictor=predictor,
                    range_predictor=range_predictor,
                    vix_ctx=_vix_ctx,
                    progress_dir=self._progress_dir,
                    symbol=symbol,
                    window_id=window_id,
                    precomputed_features=_oos_feat_cache,
                )

        all_symbol_trades.sort(
            key=lambda t: str(t.get("entry_date") or t.get("entry_time") or ""))
        if all_symbol_trades:
            # W-capital: ONE shared pool. Replay every symbol's candidate trades
            # chronologically against a single equity, with live's per-trade
            # budget, aggregate-risk and open-position gates (backtesting/pool.py).
            # Supersedes Deep-audit BT-H3 (full-capital sleeve per symbol, window
            # capital = capital x N) and R17 #14 (N = active symbols): the
            # window's capital base is now exactly the configured pool, so the
            # reported return is on the money the account actually has.
            from ait.backtesting.pool import allocate_pool
            _pos = getattr(self._settings, "positions", None)
            _pool = allocate_pool(
                all_symbol_trades,
                pool_capital=self._config.initial_capital,
                position_size_pct=self._config.position_size_pct,
                max_portfolio_risk_pct=float(
                    getattr(_pos, "max_portfolio_risk_pct", 0.20)),
                max_open_positions=int(getattr(_pos, "max_open_positions", 10)),
            )
            log.info("window_pool_allocation", window=window_id,
                     pool=self._config.initial_capital,
                     candidates=len(all_symbol_trades),
                     accepted=len(_pool.accepted),
                     rejections=dict(_pool.rejections),
                     peak_open_risk=round(_pool.peak_open_risk, 2))
            all_symbol_trades = _pool.accepted
            total_pnl = sum(t.get("pnl", 0) for t in all_symbol_trades)
            window_capital = self._config.initial_capital
            window_result = WindowResult(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                backtest_result=BacktestResult(
                    trades=all_symbol_trades,
                    initial_capital=window_capital,
                    final_capital=window_capital + total_pnl,
                    start_date=test_start,
                    end_date=test_end,
                ),
                model_accuracy=model_accuracy,
                pool_rejections=dict(_pool.rejections),
                range_model_status=dict(_range_status_local),
                meta_labeler_status=dict(_meta_status_local),  # R20 #3
            )
            self._write_window_progress(window_id, window_result, curr_best_params,
                                         optuna_meta=_optuna_meta, model_weights=_model_weights)
            return window_result, curr_best_params, dict(_meta_status_local)
        else:
            self._write_window_progress(
                window_id, None, curr_best_params,
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                optuna_meta=_optuna_meta, model_weights=_model_weights,
            )
            return None, curr_best_params, dict(_meta_status_local)

    def _write_window_progress(
        self,
        window_id: int,
        window_result: "WindowResult | None",
        best_params: "dict | None" = None,
        optuna_meta: "dict | None" = None,
        model_weights: "dict | None" = None,
        **date_kwargs,
    ) -> None:
        if self._progress_dir is None:
            return
        self._progress_dir.mkdir(parents=True, exist_ok=True)

        if window_result is not None:
            br = window_result.backtest_result
            strategy_counts: dict[str, int] = {}
            for t in br.trades:
                s = t.get("strategy", "unknown")
                strategy_counts[s] = strategy_counts.get(s, 0) + 1

            # Per-trade detail: intraday-aware fields + dashboard enrichment (Layer 2a)
            def _hold_days(t: dict) -> int:
                try:
                    from datetime import date as _date
                    return (_date.fromisoformat(str(t["exit_date"])) - _date.fromisoformat(str(t["entry_date"]))).days
                except Exception:
                    return 0

            trades_detail = [
                {
                    "symbol":           t.get("symbol", ""),
                    "strategy":         t.get("strategy", ""),
                    "entry_date":       str(t.get("entry_date", "")),
                    "entry_time":       str(t.get("entry_time", t.get("entry_date", ""))),
                    "exit_date":        str(t.get("exit_date", "")),
                    "exit_time":        str(t.get("exit_time",  t.get("exit_date", ""))),
                    "exit_reason":      t.get("exit_reason", ""),
                    "pnl":              round(t.get("pnl", 0), 2),
                    "entry_confidence": t.get("entry_confidence"),
                    "entry_regime":     t.get("entry_regime", ""),
                    # Layer 2a additions
                    "contracts":        t.get("contracts"),
                    "n_legs":           t.get("n_legs"),
                    "hold_days":        _hold_days(t),
                    "entry_price":      t.get("entry_price"),
                    "limit_price":      t.get("limit_price"),
                    "fill_time":        t.get("fill_time"),
                    "exit_price":       t.get("exit_price"),
                    "entry_iv_rank":    t.get("entry_iv_rank"),
                    "entry_vix_level":  t.get("entry_vix_level"),
                    "credit":           t.get("credit"),
                    "max_loss":         t.get("max_loss"),
                    "return_pct":       round(t["pnl"] / t["max_loss"], 4)
                                        if t.get("max_loss") and t["max_loss"] != 0 else None,
                    "legs":             t.get("legs", []),
                    "decision":         t.get("decision", {}),
                    "features_at_entry": t.get("features_at_entry", {}),
                }
                for t in br.trades
            ]

            payload = {
                "window": window_id,
                "test_start": str(window_result.test_start),
                "test_end": str(window_result.test_end),
                "pnl": round(br.final_capital - br.initial_capital, 2),
                "return_pct": round(br.total_return * 100, 4),
                "trades": br.total_trades,
                "win_rate": round(br.win_rate, 4),
                "sharpe": round(br.sharpe_ratio, 4),
                "max_drawdown": round(br.max_drawdown * 100, 4),
                "strategies": strategy_counts,
                "best_params": best_params or {},
                "trades_detail": trades_detail,
            }
        else:
            payload = {
                "window": window_id,
                "test_start": str(date_kwargs.get("test_start", "")),
                "test_end": str(date_kwargs.get("test_end", "")),
                "pnl": 0.0,
                "return_pct": 0.0,
                "trades": 0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "strategies": {},
                "best_params": best_params or {},
                "note": "no_trades",
            }

        # Optuna trial history (Layer 2b) — only present when optimize_per_window=True
        if optuna_meta is not None:
            payload["optuna_trials"] = optuna_meta.get("trials", [])
            payload["optuna_meta"] = optuna_meta.get("meta", {})

        # Per-model fitted ensemble weights + CV scores
        if model_weights:
            payload["model_weights"] = model_weights

        path = self._progress_dir / f"window_{window_id:03d}.json"
        path.write_text(json.dumps(payload, indent=2, default=_json_numpy_default), encoding="utf-8")
        log.info("window_progress_written", window=window_id, path=str(path),
                 pnl=payload["pnl"], trades=payload["trades"])

    def _optimize_window_params(
        self,
        train_df: pd.DataFrame,
        symbol: str,
        window_id: int,
        strategies: list[str],
        prior_oos_result: "BacktestResult | None" = None,
        prior_best_params: "dict | None" = None,
        range_predictor: "Any | None" = None,
        vix_ctx: "dict | None" = None,
        range_bank: "RangeModelBank | None" = None,
    ) -> "tuple[WalkForwardConfig | None, dict | None, dict | None]":
        """Run Optuna on the training slice and return (updated_config, best_params, optuna_meta).

        Runs one Optuna study per strategy (per-strategy optimization) so each
        strategy's ~12D space gets full coverage at the trial budget rather than
        a shared 48D joint space. Warm-starts each strategy's study from its own
        subset of prior params.
        Returns (None, None) on any failure (caller falls back to original config).
        """
        try:
            from ait.ml.features import FeatureEngine
            from ait.optimization.optimizer import StrategyOptimizer

            try:
                # Pass intraday_store+symbol so VLMC features appear in features_cache,
                # matching the training path used by _train_window_model (Gap B fix).
                _intraday_store = self._make_intraday_store()
                features_cache = FeatureEngine().compute(
                    train_df, intraday_store=_intraday_store, symbol=symbol,
                    market_context=vix_ctx,
                )
            except Exception:
                features_cache = None

            # Per-strategy optimization: one Optuna study per strategy.
            # Each study seeds from its own subset of prior/global-best params.
            all_best_params: dict = {}
            best_values: dict[str, float] = {}
            _all_optuna_results: list = []

            for strategy in strategies:
                warm: dict | None = None
                source: str = "cold_start"

                if prior_best_params and prior_oos_result is not None:
                    if (prior_oos_result.win_rate >= 0.75
                            and prior_oos_result.total_trades >= 3):
                        subset = {k: v for k, v in prior_best_params.items()
                                  if k.startswith(f"{strategy}__")}
                        warm = subset or None
                        source = "prior_window"
                    elif self._global_best_params is not None:
                        subset = {k: v for k, v in self._global_best_params.items()
                                  if k.startswith(f"{strategy}__")}
                        warm = subset or None
                        source = "global_best"
                elif self._global_best_params is not None:
                    subset = {k: v for k, v in self._global_best_params.items()
                              if k.startswith(f"{strategy}__")}
                    warm = subset or None
                    source = "global_best_no_prior"

                log.info(
                    "per_strategy_optimization_starting",
                    window=window_id, strategy=strategy, warm_start=source,
                )
                opt = StrategyOptimizer(
                    symbols=[symbol],
                    strategies=[strategy],
                    n_trials=self._config.optimize_n_trials,
                    n_jobs=1,
                    objective="composite",
                    study_name=f"wf_w{window_id}_{symbol}_{strategy}",
                    initial_capital=self._config.initial_capital,
                    features_cache=features_cache,
                    position_size_pct=self._config.position_size_pct,
                    budget_aware_wings=self._config.budget_aware_wings,
                    max_contracts_per_trade=self._config.max_contracts_per_trade,
                    condor_any_direction=self._config.condor_any_direction,
                    friction_model=self._config.friction_model,
                    friction_entry_capture=self._config.friction_entry_capture,
                    friction_exit_capture=self._config.friction_exit_capture,
                    friction_stress_multiplier=self._config.friction_stress_multiplier,
                    gates_mode=self._config.gates_mode,
                    iv_rank_gate=self._config.iv_rank_gate,
                    iv_rank_series=self._iv_rank_series,
                    fractal_veto_spread=self._config.fractal_veto_spread,
                    penalty_on_range_prob=self._config.penalty_on_range_prob,
                    hard_min_trades=self._config.optimize_hard_min_trades,
                    below_floor_gradient=self._config.optimize_below_floor_gradient,
                    economic_calendar=self._research_calendar,
                    wing_floor_dollars=self._config.wing_floor_dollars,
                    wing_k=self._config.wing_k,
                    iv_floor=self._config.iv_floor,
                    delta_iv_scale=self._config.delta_iv_scale,
                    patience=self._config.optimize_patience,
                    min_trades=self._config.optimize_min_trades,
                    max_concurrent_positions=self._config.max_concurrent_positions,
                    max_entry_vol_annual=self._config.max_entry_vol_annual,
                    # R20 #2: baselines for the searched-but-previously-dropped
                    # gate params — trials now score the values they suggest.
                    iv_rank_rise_threshold=self._config.iv_rank_rise_threshold,
                    min_edge_over_baseline=self._config.min_edge_over_baseline,
                    # PR#7 review 2026-08-25: without these three, trials
                    # trained at loaded-YAML baselines while the OOS
                    # Backtester ran the caller's (possibly explicit) config
                    # values — train/OOS parity break for any run that
                    # overrides them.
                    stop_loss_pct=self._config.stop_loss_pct,
                    profit_target_pct=self._config.profit_target_pct,
                    min_confidence=self._config.min_confidence,
                    seed=self._config.optimize_seed,
                    intraday_store=_intraday_store,
                    symbol=symbol,
                    range_predictor=range_predictor,
                    # W78: per-trial horizon (None provider = the single model above).
                    range_predictor_provider=(
                        range_bank.predictor_for if range_bank is not None else None),
                    # W79a: the DTE baselines and the opt-in search block (L1: delta_short too).
                    entry_dte=self._config.entry_dte,
                    expiry_guard_dte=self._config.expiry_guard_dte,
                    max_hold_fraction=self._config.max_hold_fraction,
                    flatten_minutes_before_close=self._config.flatten_minutes_before_close,
                    delta_short=self._config.delta_short,
                    dte_search=self._dte_search_block(),
                    multivariate=bool(self._config.optimize_multivariate),
                    val_split=self._config.optimize_val_split,
                    # R20b review follow-up: trial objectives priced with the
                    # same training-window VIX/SPY context as features_cache
                    # above, instead of the synthetic realized-vol fallback.
                    market_context=vix_ctx,
                    # R20b review follow-up: reuse settings loaded once in
                    # WalkForwardBacktester.__init__ instead of every
                    # per-window, per-strategy StrategyOptimizer independently
                    # re-reading + re-validating config.yaml.
                    settings=self._settings,
                )
                res = opt.run(data={symbol: train_df}, prior_params=warm)
                self._opt_status_by_window.setdefault(window_id, {})[symbol] = {
                    "usable": bool(res.usable), "best_value": float(res.best_value),
                    "best_n_trades": res.best_n_trades}
                all_best_params.update(res.best_params)
                best_values[strategy] = res.best_value
                # Collect trial data for dashboard (Layer 2b)
                _all_optuna_results.append(res)

            # Merge Optuna trial data across per-strategy studies into a single
            # flat list for the dashboard.  Each trial gets a unique number offset
            # so they don't collide when multiple strategies are optimized.
            _optuna_meta = _build_optuna_window_data(
                _all_optuna_results, strategies, window_id, symbol,
                n_trials_requested=self._config.optimize_n_trials,
                patience=self._config.optimize_patience,
            )

            # W79a: keep the compact DTE trials with the window's optimisation status, so the
            # report works in sequential AND parallel mode (the status dict already travels
            # back from the window workers). Only for a DTE study: other runs are unchanged.
            if getattr(self._config, "dte_search", False):
                _dte_names = ("entry_dte", "expiry_guard_dte", "max_hold_fraction",
                              "flatten_minutes_before_close")
                _st = self._opt_status_by_window.setdefault(window_id, {}).setdefault(symbol, {})
                _st["trials"] = [
                    {"params": {k.rpartition("__")[2]: v for k, v in t["params"].items()
                                if k.rpartition("__")[2] in _dte_names},
                     "value": t["value"], "n_trades": t["n_trades"], "state": t["state"]}
                    for t in _optuna_meta["trials"]]

            # Build an updated config from merged best params, falling back to originals
            import dataclasses
            cfg_dict = dataclasses.asdict(self._config)
            for key, val in all_best_params.items():
                _, _, param_name = key.partition("__")
                if param_name in cfg_dict:
                    cfg_dict[param_name] = val
            new_cfg = WalkForwardConfig(**cfg_dict)
            log.info(
                "window_params_optimized",
                window=window_id,
                symbol=symbol,
                best_values=best_values,
            )
            return new_cfg, all_best_params, _optuna_meta
        except Exception as e:
            log.warning("window_optimization_failed", window=window_id, symbol=symbol, error=str(e))
            # W79a: a failed optimisation used to vanish into a fall-back to the BASE config, so
            # a study that searched nothing looked identical to one that did. It is now a
            # counted status (travels through the parallel workers with the rest of
            # optimization_status) and, under optimize_strict, an error.
            self._opt_status_by_window.setdefault(window_id, {})[symbol] = {
                "usable": False, "failed": True, "error": f"{type(e).__name__}: {e}"}
            if getattr(self._config, "optimize_strict", False):
                raise
            return None, None, None

    def _evaluate_range_model_oos(
        self,
        range_predictor: "Any",
        test_df: pd.DataFrame,
        symbol: str,
        threshold_pct: float = 0.05,
        horizon_days: int = 30,
        precomputed_features: "pd.DataFrame | None" = None,
    ) -> dict:
        """Evaluate range predictor accuracy on OOS test data.

        Computes per-model probability scoring metrics (Brier score, log loss,
        Brier skill score, realized-vol MAE) and balanced accuracy for the ML
        members.  All metrics are stored under ``oos_scores`` in the window
        JSON so train/test generalisation gaps are visible per model.
        (R12-C: GARCH/MS-GARCH/OU rolling scoring and AEKF diagnostics removed
        with the family's retirement to deprecated/research/.)

        Returns empty dict if evaluation is not possible.
        """
        import numpy as _np

        try:
            # ----------------------------------------------------------------
            # 1. Build OOS range labels and realized-vol series
            # ----------------------------------------------------------------
            if len(test_df) <= horizon_days + 5:
                return {}

            close = test_df["Close"].values
            oos_returns = _np.diff(_np.log(close))
            n_close = len(close)

            labels_dict: dict[int, int] = {}
            rvol_realized: dict[int, float] = {}
            for t in range(n_close - horizon_days):
                base   = close[t]
                future = close[t + 1: t + 1 + horizon_days]
                max_dev = float(_np.max(_np.abs(future / base - 1)))
                labels_dict[t] = 1 if max_dev < threshold_pct else 0
                # Realized annualised vol over the horizon window
                fwd_rets = _np.diff(_np.log(close[t: t + 1 + horizon_days]))
                rvol_realized[t] = (
                    float(_np.std(fwd_rets) * _np.sqrt(252))
                    if len(fwd_rets) > 1 else float("nan")
                )

            if not labels_dict:
                return {}

            label_idx = list(labels_dict.keys())
            y_true     = _np.array([labels_dict[i] for i in label_idx])
            rvol_true  = _np.array([rvol_realized[i] for i in label_idx])
            base_rate  = float(_np.mean(y_true))

            if len(_np.unique(y_true)) < 2:
                return {}

            # ----------------------------------------------------------------
            # 2. ML model scoring (XGBoost / LightGBM)
            # ----------------------------------------------------------------
            rp_sym       = getattr(range_predictor, "_symbol_models", {}).get(symbol, {})
            models       = rp_sym.get("models", getattr(range_predictor, "_models", {}))
            scaler       = rp_sym.get("scaler", getattr(range_predictor, "_scaler", None))
            feature_names = rp_sym.get(
                "feature_names", getattr(range_predictor, "_feature_names", [])
            )
            weights      = rp_sym.get("fitted_weights") or getattr(range_predictor, "_fitted_weights", None) or {}

            ml_scores:    dict[str, dict] = {}
            ensemble_proba = _np.zeros(len(y_true))
            total_weight   = sum(weights.get(n, 0.5) for n in models) if models else 1.0

            if models and scaler is not None and feature_names:
                if precomputed_features is not None and not precomputed_features.empty:
                    features = precomputed_features.reindex(test_df.index)
                else:
                    try:
                        from ait.ml.features import FeatureEngine as _FE
                        features = _FE().compute(test_df)
                    except Exception:
                        features = None
                if features is not None and not features.empty and len(features) > horizon_days:
                    # Align feature rows to labellable timesteps
                    valid_idx = [i for i in label_idx if i < len(features)]
                    if valid_idx:
                        feat_rows = features.iloc[valid_idx]
                        y_true_ml = _np.array([labels_dict[i] for i in valid_idx])
                    else:
                        feat_rows, y_true_ml = features.iloc[label_idx], y_true
                    if not any(f not in feat_rows.columns for f in feature_names):
                        X = feat_rows[feature_names]
                        try:
                            X_sc = pd.DataFrame(
                                scaler.transform(X.values), columns=feature_names,
                            )
                            for name, model in models.items():
                                try:
                                    proba = model.predict_proba(X_sc)[:, 1]
                                    ml_scores[name] = self._prob_score_metrics(
                                        y_true_ml, proba, base_rate
                                    )
                                    w = weights.get(name, 0.5) / total_weight if total_weight > 0 else 0.5
                                    ensemble_proba += w * proba[:len(ensemble_proba)]
                                except Exception:
                                    continue
                        except Exception:
                            pass

            # Ensemble score over ML members
            if ml_scores:
                ml_scores["ensemble_ml"] = self._prob_score_metrics(
                    y_true, ensemble_proba, base_rate
                )

            # ----------------------------------------------------------------
            # 3. Statistical model rolling OOS scoring — REMOVED (R12-C).
            # GARCH/MS-GARCH/OU forecasts + AEKF diagnostics retired to
            # deprecated/research/; RangePredictor emits no *_state keys.
            # ----------------------------------------------------------------
            stat_scores: dict[str, dict] = {}

            # ----------------------------------------------------------------
            # 4. Full ensemble combined score
            # ----------------------------------------------------------------
            all_scores = {**ml_scores, **stat_scores}
            if all_scores:
                # Build combined weighted ensemble proba
                combined_proba = _np.zeros(len(y_true))
                combined_w     = 0.0
                for name, w in weights.items():
                    m_scores = all_scores.get(name, {})
                    if "p_scores" in m_scores and w > 0:
                        combined_proba += w * _np.asarray(m_scores["p_scores"])
                        combined_w += w
                if combined_w > 0:
                    combined_proba /= combined_w
                    all_scores["ensemble"] = self._prob_score_metrics(
                        y_true, combined_proba, base_rate
                    )

            # Strip raw p_scores array; calibration_bins (pre-computed) are kept
            def _strip_arrays(d: dict) -> dict:
                return {k: v for k, v in d.items() if k != "p_scores"}

            result: dict = {
                "n_samples":   len(y_true),
                "base_rate":   round(base_rate, 4),
                "ml":          {k: _strip_arrays(v) for k, v in ml_scores.items()},
                "statistical": {k: _strip_arrays(v) for k, v in stat_scores.items()},
            }
            if "ensemble" in all_scores:
                result["ensemble"] = _strip_arrays(all_scores["ensemble"])

            log.debug(
                "range_oos_evaluated",
                symbol=symbol,
                n_samples=len(y_true),
                ml_models=list(ml_scores.keys()),
                stat_models=list(stat_scores.keys()),
            )
            return result

        except Exception as e:
            log.debug("range_oos_eval_failed", symbol=symbol, error=str(e))
            return {}

    @staticmethod
    def _prob_score_metrics(
        y_true: "np.ndarray",
        p_scores: "np.ndarray",
        base_rate: float,
        rvol_pred: "np.ndarray | None" = None,
        rvol_true: "np.ndarray | None" = None,
    ) -> dict:
        """Compute probability scoring metrics for one model's OOS predictions.

        Metrics:
          brier_score      — mean squared probability error: E[(p - y)²]
          brier_skill      — 1 − Brier / Brier_baseline  (+ = better than climatology)
          log_loss         — mean negative log-likelihood: −E[y·log(p) + (1−y)·log(1−p)]
          auroc            — Area Under ROC Curve (rank discrimination)
          balanced_acc     — balanced accuracy at 0.5 decision threshold
          mean_confidence  — mean(max(p, 1−p)); how assertive the model is
          rvol_mae         — mean |σ_predicted − σ_realized| annualised (if provided)
          rvol_bias        — mean(σ_predicted − σ_realized); + = over-estimates vol
        """
        import numpy as _np
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score

        p = _np.asarray(p_scores, dtype=float)
        y = _np.asarray(y_true,   dtype=float)
        p = _np.clip(p, 1e-7, 1.0 - 1e-7)

        n = len(y)
        result: dict = {"p_scores": p.tolist()}  # kept internally; stripped before JSON

        # Brier score
        brier = float(_np.mean((p - y) ** 2))
        brier_base = float(base_rate * (1.0 - base_rate))
        brier_skill = float(1.0 - brier / brier_base) if brier_base > 0 else float("nan")
        result["brier_score"]  = round(brier, 5)
        result["brier_skill"]  = round(brier_skill, 4)

        # Log loss (cross-entropy)
        ll = float(-_np.mean(y * _np.log(p) + (1.0 - y) * _np.log(1.0 - p)))
        result["log_loss"] = round(ll, 5)

        # AUROC
        try:
            result["auroc"] = round(float(roc_auc_score(y, p)), 4)
        except Exception:
            result["auroc"] = float("nan")

        # Balanced accuracy at 0.5 threshold
        try:
            preds = (p >= 0.5).astype(int)
            result["balanced_acc"] = round(float(balanced_accuracy_score(y.astype(int), preds)), 4)
        except Exception:
            result["balanced_acc"] = float("nan")

        # Mean confidence (assertiveness)
        result["mean_confidence"] = round(float(_np.mean(_np.maximum(p, 1.0 - p))), 4)

        # Realized vol MAE and bias
        if rvol_pred is not None and rvol_true is not None:
            rp = _np.asarray(rvol_pred, dtype=float)
            rt = _np.asarray(rvol_true, dtype=float)
            finite = _np.isfinite(rp) & _np.isfinite(rt)
            if finite.sum() >= 2:
                result["rvol_mae"]  = round(float(_np.mean(_np.abs(rp[finite] - rt[finite]))), 5)
                result["rvol_bias"] = round(float(_np.mean(rp[finite] - rt[finite])), 5)

        # Calibration bins: predicted probability vs empirical frequency.
        # Format: [{p, actual, n}, ...] — same shape the dashboard ReliabilityChart
        # expects so export.py can pass them through directly.
        nbins = 9
        bins = []
        for b in range(nbins):
            lo, hi = b / nbins, (b + 1) / nbins
            mask = (p >= lo) & (p < hi)
            n_bin = int(mask.sum())
            if n_bin > 0:
                bins.append({
                    "p":      round((lo + hi) / 2, 3),
                    "actual": round(float(y[mask].mean()), 3),
                    "n":      n_bin,
                })
        result["calibration_bins"] = bins

        return result

    def _evaluate_direction_model_oos(
        self,
        predictor: "Any",
        test_df: "pd.DataFrame",
        symbol: str,
        horizon_days: int = 30,
        return_threshold_pct: float = 0.02,
        precomputed_features: "pd.DataFrame | None" = None,
    ) -> dict:
        """Evaluate directional predictor calibration on OOS test data.

        Computes per-member confidence calibration bins (predicted confidence vs
        actual accuracy) and overall OOS AUROC for XGBoost and LightGBM.
        Returns empty dict when evaluation is not possible.

        Calibration semantics: bins predicted confidence p (= max class probability)
        against the fraction of predictions at that confidence level that were
        actually correct (1-vs-rest accuracy).  Stored as
        ``calibration_bins: [{p, actual, n}]`` so the dashboard ReliabilityChart
        can render them directly without synthetic approximation.
        """
        import numpy as _np

        try:
            sym_data = getattr(predictor, "_symbol_models", {}).get(symbol, {})
            models       = sym_data.get("models") or getattr(predictor, "_models", {})
            scaler       = sym_data.get("scaler") or getattr(predictor, "_scaler", None)
            feature_names = sym_data.get("feature_names") or getattr(predictor, "_feature_names", [])

            if not models or scaler is None or not feature_names:
                return {}
            if len(test_df) < horizon_days + 20:
                return {}

            if precomputed_features is not None and not precomputed_features.empty:
                features = precomputed_features.reindex(test_df.index)
            else:
                from ait.ml.features import FeatureEngine as _FE
                features = _FE().compute(test_df)
            if features is None or features.empty:
                return {}

            # Build forward-return labels for each bar (3-class)
            close = test_df["Close"].values
            n_label = len(close) - horizon_days
            if n_label < 10:
                return {}

            y_class = []  # 0=bearish, 1=neutral, 2=bullish
            for t in range(n_label):
                fwd_ret = (close[t + horizon_days] - close[t]) / close[t]
                if fwd_ret > return_threshold_pct:
                    y_class.append(2)
                elif fwd_ret < -return_threshold_pct:
                    y_class.append(0)
                else:
                    y_class.append(1)
            y_class_arr = _np.array(y_class)

            # Align features to labellable bars
            feat_rows = features.iloc[:n_label]
            if any(f not in feat_rows.columns for f in feature_names):
                return {}

            X = feat_rows[feature_names]
            try:
                X_sc = pd.DataFrame(scaler.transform(X.values), columns=feature_names)
            except Exception:
                return {}

            member_scores: dict[str, dict] = {}
            ensemble_conf = _np.zeros(n_label)
            total_w = sum(
                getattr(predictor, "_fitted_weights", {}).get(n, 0.5) for n in models
            ) or 1.0

            for name, model in models.items():
                try:
                    proba = model.predict_proba(X_sc)          # (n, 3)
                    conf  = _np.max(proba, axis=1)             # max-class confidence
                    pred  = _np.argmax(proba, axis=1)          # predicted class index
                    correct = (pred == y_class_arr).astype(float)

                    # Confidence calibration bins
                    nbins, calib = 8, []
                    for b in range(nbins):
                        lo, hi = b / nbins, (b + 1) / nbins
                        mask = (conf >= lo) & (conf < hi)
                        n_bin = int(mask.sum())
                        if n_bin > 0:
                            calib.append({
                                "p":      round((lo + hi) / 2, 3),
                                "actual": round(float(correct[mask].mean()), 3),
                                "n":      n_bin,
                            })

                    # Per-class AUROC (one-vs-rest, macro)
                    from sklearn.metrics import roc_auc_score
                    try:
                        auroc = round(float(roc_auc_score(
                            y_class_arr, proba, multi_class="ovr", average="macro"
                        )), 4)
                    except Exception:
                        auroc = None

                    overall_acc = round(float(correct.mean()), 4)
                    w = getattr(predictor, "_fitted_weights", {}).get(name, 0.5)
                    member_scores[name] = {
                        "calibration_bins": calib,
                        "auroc": auroc,
                        "accuracy": overall_acc,
                        "n_samples": n_label,
                    }
                    ensemble_conf += (w / total_w) * conf

                except Exception:
                    continue

            if not member_scores:
                return {}

            # Ensemble confidence calibration
            all_preds = _np.array([
                _np.argmax(
                    sum(
                        (getattr(predictor, "_fitted_weights", {}).get(n, 0.5) / total_w)
                        * models[n].predict_proba(X_sc)
                        for n in models
                    ),
                    axis=1,
                )
            ])
            ensemble_correct = (all_preds[0] == y_class_arr).astype(float)
            nbins, ens_calib = 8, []
            for b in range(nbins):
                lo, hi = b / nbins, (b + 1) / nbins
                mask = (ensemble_conf >= lo) & (ensemble_conf < hi)
                n_bin = int(mask.sum())
                if n_bin > 0:
                    ens_calib.append({
                        "p":      round((lo + hi) / 2, 3),
                        "actual": round(float(ensemble_correct[mask].mean()), 3),
                        "n":      n_bin,
                    })

            return {
                "n_samples":       n_label,
                "horizon_days":    horizon_days,
                "return_threshold": return_threshold_pct,
                "members":         member_scores,
                "ensemble": {
                    "calibration_bins": ens_calib,
                    "accuracy": round(float(ensemble_correct.mean()), 4),
                    "n_samples": n_label,
                },
            }

        except Exception as e:
            log.debug("direction_oos_eval_failed", symbol=symbol, error=str(e))
            return {}

    @staticmethod
    def _adaptive_range_threshold(
        train_df: pd.DataFrame,
        horizon_days: int,
        lookback_days: int = 60,
        multiplier: float = 1.25,
        low_clip: float = 0.02,
        high_clip: float = 0.15,
    ) -> float:
        """Derive a regime-adaptive range threshold from recent realized volatility.

        Uses the last `lookback_days` of training data to estimate 1-day realized
        vol, then scales to the horizon via √(horizon/252). The 1.25× multiplier
        places the threshold at ~1.25 standard deviations — an iron condor survives
        ~85% of moves under a normal distribution at this width.

        Clipped to [low_clip, high_clip] to prevent degenerate extremes.

        Example outputs for QQQ:
          Quiet bull run (rvol≈15%): threshold ≈ 4.3% for 21d horizon
          Normal regime (rvol≈20%): threshold ≈ 5.7%
          High-vol spike (rvol≈35%): threshold ≈ 10.1%
        """
        recent = train_df["Close"].pct_change().tail(lookback_days).dropna()
        if len(recent) < 10:
            return 0.05  # insufficient data — fall back to fixed default
        rvol_annual = float(recent.std() * np.sqrt(252))
        raw = rvol_annual * np.sqrt(horizon_days / 252) * multiplier
        return float(np.clip(raw, low_clip, high_clip))

    def _pretrain_range_models(
        self,
        windows: "list[tuple]",
        data: "dict[str, pd.DataFrame]",
        vix_full: "pd.DataFrame",
    ) -> "list[dict]":
        """Train range predictors sequentially for all windows before Optuna starts.

        Returns a list (one entry per window) of dicts keyed by symbol, each
        containing the trained RangePredictor (or None on failure) plus the
        status string and threshold float — the same tuple _train_window_range_model
        returns, but wrapped in a dict so multiple symbols can be pre-trained.

        Running this sequentially in the parent process before ProcessPoolExecutor
        launches means:
          - No CPU contention with Optuna workers (they haven't started yet)
          - No subprocess timeout risk from resource starvation
          - Optuna's RNG state is untouched (statistical fitting runs before Optuna)
          - The spawn-subprocess isolation is no longer needed for this phase
        """
        if not self._config.train_window_models:
            # R16 #4: gate-stack-only arm — the flag disables ALL window
            # models, so skip range pre-training entirely (it previously ran
            # regardless, leaking a trained range gate into "ML-free" arms).
            log.info(
                "range_pretraining_disabled_by_config",
                windows=len(windows),
                reason="train_window_models=False",
            )
            return [
                {symbol: (None, "disabled_by_config", 0.05) for symbol in data}
                for _ in windows
            ]

        results: list[dict] = []
        n_windows = len(windows)
        for i, (train_start, train_end, _test_start, _test_end) in enumerate(windows):
            window_id = i + 1
            per_symbol: dict[str, "tuple[Any | None, str, float]"] = {}
            for symbol, df in data.items():
                train_df = df[str(train_start):str(train_end)]
                if len(train_df) < 50:
                    per_symbol[symbol] = (None, "insufficient_train_data", 0.05)
                    continue
                log.info(
                    "pretraining_range_model",
                    window=window_id,
                    symbol=symbol,
                    n_windows=n_windows,
                )
                # Train directly in-process — Optuna hasn't started, no RNG risk.
                # PR#7 review 2026-08-25: was self._config.max_hold_days (30)
                # for both label and threshold horizons — see
                # _range_label_horizon.
                per_symbol[symbol] = self._train_window_range_model_inprocess(
                    train_df=train_df,
                    symbol=symbol,
                    window_id=window_id,
                    max_hold_days=self._range_label_horizon(),
                    threshold_pct=self._adaptive_range_threshold(
                        train_df, horizon_days=self._range_label_horizon()
                    ),
                )
            results.append(per_symbol)
        return results

    def _range_label_horizon(self, entry_dte: "int | None" = None,
                             guard_dte: "int | None" = None,
                             hold_fraction: "float | None" = None) -> int:
        """Label/eval horizon for the range model: the REACHABLE trade horizon.

        PR#7 review 2026-08-25: both range-training paths passed
        self._config.max_hold_days (resolves to 30) as RangePredictor's
        horizon_days, but since the engine's entry_dte decoupling positions
        expire at options.dte_range[0] (14) and credit structures close at
        EXPIRY_APPROACHING_DTE (5) — real holds cap at ~9 calendar days, and
        optimized IC holds search 1-9. The model was trained to predict
        30-day containment (a strictly harder question) and reused to gate
        much shorter trades. Derive the horizon from what a trade can
        actually reach; max_hold_days remains a cap, never an extender.
        """
        from ait.config.settings import OptionsConfig
        from ait.ml.range_spec import horizon_for
        if entry_dte is None:
            entry_dte = getattr(self._config, "entry_dte", None)   # W78: a study's own DTE
        if entry_dte is None:
            try:
                _opts = (self._settings.options if self._settings is not None
                         else OptionsConfig())
            except Exception:  # noqa: BLE001 — partial test stubs -> field defaults
                _opts = OptionsConfig()
            entry_dte = int(_opts.dte_range[0])
        # W78: ONE definition shared with the live spec (ait.ml.range_spec.horizon_for).
        # guard / hold-fraction fields arrive with W79; getattr keeps partial configs valid.
        # W79a: a window / trial may carry its OWN guard and fraction (they are searched);
        # None falls back to the study's base values.
        return horizon_for(
            int(entry_dte),
            guard_dte=(guard_dte if guard_dte is not None
                       else getattr(self._config, "expiry_guard_dte", None)),
            hold_fraction=(hold_fraction if hold_fraction is not None
                           else getattr(self._config, "max_hold_fraction", None)),
            max_hold_days=int(self._config.max_hold_days),
        )

    def _range_bucket(self, entry_dte: "int | None" = None, guard_dte: "int | None" = None,
                      hold_fraction: "float | None" = None) -> int:
        """W78: the horizon whose range model serves ``entry_dte`` (None = the base config).

        The base configuration keeps its EXACT horizon (what it has always trained at); any
        other DTE whose reachable horizon differs is rounded UP to the next bucket so a study
        that searches the DTE trains a handful of models per window, not one per DTE.
        """
        from ait.ml.range_spec import bucket_for
        base = self._range_label_horizon()
        if entry_dte is None and guard_dte is None and hold_fraction is None:
            return base
        h = self._range_label_horizon(
            None if entry_dte is None else int(entry_dte), guard_dte, hold_fraction)
        return base if h == base else bucket_for(h)

    def _pick_range_model(self, bank: "RangeModelBank", window_cfg: "Any") -> "tuple[int, tuple]":
        """W78: (bucket, (predictor, status, threshold)) that gates this window's OOS run: the
        model for the window config's OWN entry DTE (None -> the base bucket)."""
        bucket = self._range_bucket(
            getattr(window_cfg, "entry_dte", None),
            getattr(window_cfg, "expiry_guard_dte", None),
            getattr(window_cfg, "max_hold_fraction", None))
        return bucket, bank.get(bucket)

    def _dte_search_block(self) -> "Any | None":
        """W79a: the DteSearchConfig handed to each window's optimizer (None = not searching).

        The ranges come from the loaded settings; the per-study switches on the walk-forward
        config (enable, ``dte_range``) win."""
        if not getattr(self._config, "dte_search", False):
            return None
        from ait.config.settings import DteSearchConfig
        try:
            base = self._settings.backtest.dte_search
        except Exception:  # noqa: BLE001 - partial stubs -> the model defaults
            base = DteSearchConfig()
        upd: dict = {"enabled": True}
        if getattr(self._config, "dte_range", None) is not None:
            upd["entry_dte"] = list(self._config.dte_range)
        return base.model_copy(update=upd)

    def _train_range_bucket(self, train_df: pd.DataFrame, symbol: str, window_id: int,
                            bucket: int) -> "tuple[Any | None, str, float]":
        """Train the range model for a non-base bucket (artifacts under models/research/h<N>)."""
        if not self._config.train_window_models:
            return None, "disabled_by_config", 0.05
        res = self._train_window_range_model(
            train_df, symbol, window_id, max_hold_days=int(bucket),
            model_subdir=f"h{int(bucket)}",
        )
        return res

    def _train_window_range_model(
        self,
        train_df: pd.DataFrame,
        symbol: str,
        window_id: int,
        max_hold_days: int = 30,
        model_subdir: "str | None" = None,
    ) -> "tuple[Any | None, str, float]":
        """Train range predictor on this window's training data.

        In parallel mode this is only called as a fallback (when pretrained_range
        is not supplied).  In sequential mode it runs directly in-process since
        Optuna has not started yet and there is no RNG contamination risk.

        Returns (predictor, status, threshold_used).
        """
        threshold_pct = self._adaptive_range_threshold(train_df, horizon_days=max_hold_days)
        return self._train_window_range_model_inprocess(
            train_df, symbol, window_id, max_hold_days, threshold_pct,
            model_subdir=model_subdir,
        )

    def _train_window_range_model_inprocess(
        self,
        train_df: pd.DataFrame,
        symbol: str,
        window_id: int,
        max_hold_days: int,
        threshold_pct: float,
        model_subdir: "str | None" = None,
    ) -> "tuple[Any | None, str, float]":
        """Train the ML-only (XGB/LGBM) range model in-process.

        R12-C: the spawn-isolated variant and the MS-GARCH/OU statistical
        members are retired (deprecated/research/); ML-only training does not
        touch the numpy/scipy global RNG that Optuna's sampler depends on.
        """
        try:
            from ait.ml.range_predictor import RESEARCH_MODEL_DIR, RangePredictor
            intraday_store = self._make_intraday_store()
            rp = RangePredictor(
                threshold_pct=threshold_pct,
                horizon_days=max_hold_days,
                # R7/R10 artifact hygiene: research runs save under
                # models/research/ — NEVER the live models/range.pkl.
                # W78: a non-base horizon saves under its own subdirectory so two horizons
                # never overwrite each other's range.pkl (base keeps today's path).
                model_dir=(RESEARCH_MODEL_DIR / model_subdir if model_subdir
                           else RESEARCH_MODEL_DIR),
            )
            accs = rp.train(train_df, symbol=symbol, intraday_store=intraday_store)
            if accs and rp.is_trained:
                avg = sum(accs.values()) / len(accs)
                log.info(
                    "window_range_model_trained",
                    window=window_id, symbol=symbol,
                    mode="inprocess",
                    accuracy=f"{avg:.3f}",
                    threshold_pct=f"{threshold_pct:.3f}",
                )
                return rp, "ok", threshold_pct
            return None, "training_returned_no_accuracy", threshold_pct
        except Exception as e:
            reason = str(e)
            log.warning(
                "range_model_train_failed",
                window=window_id, symbol=symbol,
                error=reason,
                threshold_pct=f"{threshold_pct:.3f}",
                action="OOS IC entries will be blocked — no range gate available",
            )
            return None, f"exception: {reason}", threshold_pct

    def _train_window_model(
        self,
        train_df: pd.DataFrame,
        symbol: str,
        window_id: int,
        vix_full: "pd.DataFrame | None" = None,
        spy_full: "pd.DataFrame | None" = None,
    ) -> "DirectionPredictor | None":
        """Train ML model on a training window's data.

        Returns a trained DirectionPredictor, or None if training fails.
        Each window gets a fresh model to prevent data leakage.
        """
        try:
            intraday_store = self._make_intraday_store()

            _train_ctx: dict = {}
            if vix_full is not None and not vix_full.empty:
                _vix_t = vix_full.reindex(train_df.index, method="ffill").dropna(how="all")
                if not _vix_t.empty:
                    _train_ctx["vix"] = _vix_t
            if spy_full is not None and not spy_full.empty:
                _spy_t = spy_full.reindex(train_df.index, method="ffill").dropna(how="all")
                if not _spy_t.empty:
                    _train_ctx["spy"] = _spy_t
            _market_ctx = _train_ctx if _train_ctx else None

            ml_config = MLConfig()
            # R16 #2: window models are throwaway research artifacts — NEVER
            # persist them. DirectionPredictor.train() used to save
            # unconditionally to the CWD-relative models/ dir, clobbering the
            # LIVE ensemble.pkl (the bot served an IWM-only window model for
            # 2.5 market hours on 2026-08-03).
            predictor = DirectionPredictor(ml_config, persist_artifacts=False)
            accuracies = predictor.train(
                train_df, symbol=symbol, intraday_store=intraday_store,
                market_context=_market_ctx,
            )

            if accuracies:
                avg_acc = sum(accuracies.values()) / len(accuracies)
                log.info(
                    "window_model_trained",
                    window=window_id,
                    symbol=symbol,
                    accuracy=f"{avg_acc:.3f}",
                    models=list(accuracies.keys()),
                )
                return predictor

        except Exception as e:
            log.debug("window_model_training_failed", symbol=symbol, window=window_id, error=str(e))

        return None

    def _train_window_meta_labeler(
        self,
        train_df: pd.DataFrame,
        symbol: str,
        window_id: int,
        predictor: "DirectionPredictor | None",
        window_cfg: "WalkForwardConfig",
        artifact_dir: "Path | None" = None,
        vix_ctx: "dict | None" = None,
    ) -> "MetaLabeler | None":
        """Train MetaLabeler on trade outcomes from the training window (Gap Z1 long-term).

        Runs a shadow Backtester on the TRAINING data using the just-trained direction
        predictor, collects closed trades with their entry context, looks up the
        corresponding FeatureEngine rows, and trains the MetaLabeler on those outcomes.
        Saves 'meta_labeler.pkl' alongside model.pkl in the window artifact directory.

        Returns a trained MetaLabeler, or None if insufficient data / training failed.
        """
        from ait.ml.features import FeatureEngine
        from ait.ml.meta_label import MetaLabeler

        try:
            # Build the feature matrix for the training window (same call as training path)
            intraday_store = self._make_intraday_store()

            features_df = FeatureEngine().compute(
                train_df, intraday_store=intraday_store, symbol=symbol
            )

            # Shadow Backtester: run on training data to get labelled trade outcomes.
            # Use a reduced context prefix (same 60-bar convention as the OOS run).
            context_bars = 60
            if len(train_df) <= context_bars:
                return None

            train_head = train_df.head(len(train_df) - context_bars)  # actual training slice
            train_with_ctx = train_df  # full training window already has its own context

            shadow_bt = Backtester(
                data=train_with_ctx,
                # W66: the shadow labeller was built WITHOUT a symbol, so it silently priced
                # trades with the config spread formula and a 1.10 vol multiplier while the
                # OOS run used the symbol's calibrated spreads / friction model v2 / vol
                # multiplier -- meta-labels learned from a different cost model than the
                # one they gate. Same symbol as the OOS Backtester now.
                symbol=symbol,
                context_bars=0,
                strategies=["iron_condor"],  # meta-labeler targets the primary strategy
                initial_capital=window_cfg.initial_capital,
                commission_per_contract=window_cfg.commission_per_contract,
                slippage_pct=window_cfg.slippage_pct,
                position_size_pct=window_cfg.position_size_pct,
                budget_aware_wings=window_cfg.budget_aware_wings,
                max_contracts_per_trade=window_cfg.max_contracts_per_trade,
                condor_any_direction=window_cfg.condor_any_direction,
                friction_model=window_cfg.friction_model,
                friction_entry_capture=window_cfg.friction_entry_capture,
                friction_exit_capture=window_cfg.friction_exit_capture,
                friction_stress_multiplier=window_cfg.friction_stress_multiplier,
                gates_mode=window_cfg.gates_mode,
                iv_rank_gate=window_cfg.iv_rank_gate,
                iv_rank_series=self._iv_rank_series.get(symbol),
                fractal_veto_spread=window_cfg.fractal_veto_spread,
                penalty_on_range_prob=window_cfg.penalty_on_range_prob,
                economic_calendar=self._research_calendar,
                stop_loss_pct=window_cfg.stop_loss_pct,
                profit_target_pct=window_cfg.profit_target_pct,
                max_hold_days=window_cfg.max_hold_days,
                min_confidence=window_cfg.min_confidence,
                trailing_stop_enabled=window_cfg.trailing_stop_enabled,
                trailing_stop_pct=window_cfg.trailing_stop_pct,
                breakeven_trigger_pct=window_cfg.breakeven_trigger_pct,
                predictor=predictor,
                features_cache=features_df,
                iv_floor=window_cfg.iv_floor,
                wing_floor_dollars=window_cfg.wing_floor_dollars,
                wing_k=window_cfg.wing_k,
                # R6 parity knobs — shadow backtest must label trades under the
                # same exit/construction rules as the OOS run
                credit_loss_limit_mult=getattr(window_cfg, "credit_loss_limit_mult", None),
                ic_min_credit=getattr(window_cfg, "ic_min_credit", None),
                ic_min_credit_width=getattr(window_cfg, "ic_min_credit_width", None),
                macro_event_gate=getattr(window_cfg, "macro_event_gate", True),
                pre_event_blackout_days=getattr(window_cfg, "pre_event_blackout_days", None),
                credit_tp_ladder=getattr(window_cfg, "credit_tp_ladder", None),
                entry_dte=getattr(window_cfg, "entry_dte", None),
                # W79a: the short-DTE exit knobs and the (possibly searched) delta reach the run
                expiry_guard_dte=getattr(window_cfg, "expiry_guard_dte", None),
                max_hold_fraction=getattr(window_cfg, "max_hold_fraction", None),
                flatten_minutes_before_close=getattr(window_cfg, "flatten_minutes_before_close", None),
                delta_short=getattr(window_cfg, "delta_short", None),
                db_path=self._db_path,  # W52a: the engine reads THIS store
                post_event_blackout_days=getattr(window_cfg, "post_event_blackout_days", None),
                post_event_dates=getattr(window_cfg, "post_event_dates", None),
                # R16 #1: shadow labelling runs with the window predictor
                # (possibly None) — never with the live artifact.
                allow_live_model_fallback=False,
                max_concurrent_positions=window_cfg.max_concurrent_positions,
                max_entry_vol_annual=window_cfg.max_entry_vol_annual,
                spread_base=window_cfg.spread_base,
                spread_iv_sensitivity=window_cfg.spread_iv_sensitivity,
                spread_dte_sensitivity=window_cfg.spread_dte_sensitivity,
                spread_cap=window_cfg.spread_cap,
                # R20 #4: same intraday knobs as the OOS Backtester (inert
                # here — no intraday_store — but the shadow must never fork
                # its own execution model if a store is ever wired in).
                scan_interval_minutes=getattr(window_cfg, "scan_interval_minutes", None),
                entry_window_start_et=getattr(window_cfg, "entry_window_start_et", None),
                entry_window_end_et=getattr(window_cfg, "entry_window_end_et", None),
                limit_order_timeout_bars=getattr(window_cfg, "limit_order_timeout_bars", None),
                market_context=vix_ctx,
                # R20b review follow-up: reuse settings loaded once in
                # __init__ instead of every shadow-labeling Backtester
                # independently re-reading + re-validating config.yaml.
                settings=self._settings,
            )
            shadow_result = shadow_bt.run()

            if not shadow_result.trades:
                log.debug(
                    "meta_labeler_no_shadow_trades",
                    window=window_id,
                    symbol=symbol,
                )
                return None

            meta_labeler = MetaLabeler()
            training_df = meta_labeler.build_training_data_from_backtest(
                trades=shadow_result.trades,
                features_df=features_df,
            )

            if training_df.empty:
                return None

            stats = meta_labeler.train(training_df)
            if not stats:
                return None

            log.info(
                "meta_labeler_window_trained",
                window=window_id,
                symbol=symbol,
                trades=len(shadow_result.trades),
                accuracy=f"{stats.get('accuracy', 0):.3f}",
                precision=f"{stats.get('precision', 0):.3f}",
            )

            if artifact_dir is not None:
                from pathlib import Path as _Path
                meta_labeler.save_to_path(_Path(artifact_dir) / "meta_labeler.pkl")

            return meta_labeler

        except Exception as e:
            log.debug(
                "meta_labeler_training_failed",
                window=window_id,
                symbol=symbol,
                error=str(e),
            )
            return None

    def benchmark_buy_hold(self, data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Compute buy-and-hold return for each symbol as a benchmark."""
        results = {}
        for symbol, df in data.items():
            if len(df) < 2:
                continue
            start_price = df["Close"].iloc[0]
            end_price = df["Close"].iloc[-1]
            results[symbol] = (end_price - start_price) / start_price
        results["portfolio"] = float(np.mean(list(results.values()))) if results else 0.0
        return results

    def _apply_research_calendar(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """W65: build the research macro calendar and keep the gate CONSISTENT.

        When `macro_event_types` is set and `{prefix}econ_events` has rows, the macro
        gate uses the operator's CSV calendar instead of the hardcoded 2026-27 list.
        Windows before the calendar's first date would have no gate at all (a mix of
        gated and ungated windows inside one run), so the data is trimmed to the
        calendar's span with an ERROR rather than silently mixed. No table / no rows
        -> WARNING and the legacy hardcoded calendar (pre-W65 behaviour)."""
        cfg = self._config
        if cfg.macro_event_types is None or self._db_path is None or not self._table_prefix:
            return data
        from ait.data.economic_calendar import build_research_calendar
        from ait.data.historical import HistoricalDataStore
        try:
            cal = build_research_calendar(self._db_path, self._table_prefix, cfg.macro_event_types)
            span = HistoricalDataStore(self._db_path, self._table_prefix).econ_events_span()
        except Exception as exc:  # noqa: BLE001
            log.warning("research_calendar_unavailable", error=str(exc))
            return data
        if cal is None or span is None:
            log.warning("research_calendar_empty", table_prefix=self._table_prefix,
                        note="no econ_events rows; macro gate uses the hardcoded live "
                             "calendar (2026+ only). Load the CSV with "
                             "scripts/load_econ_calendar_csv.py")
            return data
        self._research_calendar = cal
        first = pd.Timestamp(span[0])
        trimmed = {}
        for sym, df in data.items():
            idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
            if len(df) and idx[0] < first:
                log.error("macro_calendar_not_covering_span", symbol=sym,
                          data_start=str(idx[0].date()), calendar_start=span[0],
                          note="data trimmed to the calendar span so the macro gate is on "
                               "in every window")
                df = df[idx >= first]
            trimmed[sym] = df
        log.info("research_macro_calendar", event_types=list(cfg.macro_event_types),
                 span=span[:2], rows=span[2])
        return trimmed

    def _ensure_iv_rank_series(self, data: "dict[str, pd.DataFrame]") -> None:
        """W68: IV rank over each symbol's FULL history, once per instance (the parallel path
        builds one instance per window, so this also runs inside each worker).

        Ranking the full series, not the window slice, gives every day live's 252-observation
        lookback (a slice would restart the warm-up each window); row t only ever uses
        observations at or before t, so this introduces no look-ahead."""
        if not self._config.iv_rank_gate or self._iv_rank_series:
            return
        from ait.data.iv_rank import iv_rank_series
        for sym, df in data.items():
            if df is not None and "implied_vol" in df.columns:
                self._iv_rank_series[sym] = iv_rank_series(df["implied_vol"])

    def _aggregate_funnel(self) -> dict[str, dict[str, int]]:
        """W66: sum the per-window entry funnels into {symbol: {reason: days}}."""
        total: dict[str, dict[str, int]] = {}
        for per_symbol in self._funnel_by_window.values():
            for sym, funnel in per_symbol.items():
                acc = total.setdefault(sym, {})
                for reason, n in funnel.items():
                    acc[reason] = acc.get(reason, 0) + int(n)
        return total

    def _limit_windows(self, windows: list) -> list:
        """W65: keep only the most recent `max_windows` windows (timing/pilot runs)."""
        n = self._config.max_windows
        if n and len(windows) > n:
            log.info("walk_forward_windows_limited", total=len(windows), kept=n)
            return windows[-n:]
        return windows

    def _make_intraday_store(self):
        """W65: the (prefixed) intraday store, or None. One place for the five sites
        that used to build it inline, so the VLMC coverage guard can switch ALL of
        them off together (never VLMC in some windows and not in others)."""
        if self._db_path is None or not self._vlmc_enabled:
            return None
        from ait.data.historical import HistoricalDataStore
        return HistoricalDataStore(db_path=self._db_path, table_prefix=self._table_prefix)

    def _check_vlmc_coverage(self, data: dict[str, pd.DataFrame]) -> None:
        """W65: VLMC/intraday-fractal features must exist for EVERY window or none.

        The features are left-joined per day from 5-min bars (NaN where a day has
        none), so partial coverage would put a feature in some windows only — a
        distribution shift inside one run. Below `vlmc_min_coverage` (per symbol, over
        the whole data span) the run trains daily-only, loudly."""
        if self._db_path is None:
            return
        from ait.data.historical import HistoricalDataStore
        store = HistoricalDataStore(db_path=self._db_path, table_prefix=self._table_prefix)
        per: dict[str, float] = {}
        for sym, df in data.items():
            if df.empty:
                continue
            first, last = df.index[0].date(), df.index[-1].date()
            n = store.intraday_day_count(sym, first, last)
            per[sym] = round(n / max(1, len(df)), 3)
        threshold = self._config.vlmc_min_coverage
        if per and min(per.values()) < threshold:
            self._vlmc_enabled = False
            log.warning("vlmc_disabled_low_intraday_coverage", coverage=per,
                        threshold=threshold, table_prefix=self._table_prefix,
                        note="daily-only features this run; backfill 5-min bars with "
                             "scripts/backfill_historical_data.py --mode intraday "
                             "--what-to-show TRADES --table-prefix " + self._table_prefix)
        else:
            log.info("vlmc_enabled", coverage=per, threshold=threshold)

    def _generate_windows(self, data: dict[str, pd.DataFrame]) -> list[tuple[date, date, date, date]]:
        """Generate walk-forward train/test date windows."""
        # Find common date range across all symbols
        min_date = max(df.index[0].date() if hasattr(df.index[0], "date") else df.index[0]
                       for df in data.values())
        max_date = min(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
                       for df in data.values())

        cfg = self._config
        windows = []
        current = min_date

        while True:
            train_start = current
            train_end = train_start + timedelta(days=cfg.train_days)
            test_start = train_end + timedelta(days=cfg.gap_days)
            test_end = test_start + timedelta(days=cfg.test_days)

            if test_end > max_date:
                break

            windows.append((train_start, train_end, test_start, test_end))
            current += timedelta(days=cfg.step_days)

        return windows

    def _context_days(self) -> int:
        """W65: days of VIX/SPY context to load — the legacy train+900, or the whole
        research span plus feature warm-up when a history span is configured."""
        cfg = self._config
        if cfg.history_start_date is not None:
            span = (today_et() - cfg.history_start_date).days
        else:
            span = cfg.history_days or 0
        return max(cfg.train_days + 900, span + 400)

    async def _fetch_data(self) -> dict[str, pd.DataFrame]:
        """Load daily OHLCV from IB store (fallback: Yahoo Finance)."""
        from ait.data.market_data import load_daily_ohlcv

        data = {}
        cfg = self._config
        fetch_days = cfg.train_days + cfg.test_days + 100
        start = cfg.history_start_date
        if start is not None:
            fetch_days = max(fetch_days, (today_et() - start).days + 30)
        elif cfg.history_days:
            fetch_days = max(fetch_days, cfg.history_days)

        for symbol in self._symbols:
            try:
                loop = asyncio.get_running_loop()
                # W65: the research kwargs are passed ONLY when set, so a legacy run
                # (no prefix, no start date) makes the exact pre-W65 call.
                _kw: dict = {}
                if self._table_prefix:
                    _kw["table_prefix"] = self._table_prefix
                if start is not None:
                    _kw["start_date"] = start
                df = await loop.run_in_executor(
                    None,
                    lambda s=symbol: load_daily_ohlcv(
                        s, days=fetch_days, db_path=self._db_path, **_kw
                    ),
                )
                if df is not None and start is not None and not df.empty:
                    _idx = df.index.tz_localize(None) if df.index.tz is not None else df.index
                    df = df[_idx >= pd.Timestamp(start)]
                if df is not None and len(df) > 100:
                    # R16 #5: load_daily_ohlcv can return a tz-aware
                    # (America/New_York) index while run() normalizes VIX/SPY
                    # context to tz-naive — the mix crashed
                    # vix_full.reindex(train_df.index) on the first window,
                    # so run(data=None) (the dashboard runner) never worked.
                    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                        df = df.copy(deep=False)
                        df.index = df.index.tz_localize(None)
                    data[symbol] = df
                    log.info("data_fetched", symbol=symbol, rows=len(df))
            except Exception as e:
                log.warning("data_fetch_failed", symbol=symbol, error=str(e))

        return data

    def _compute_symbol_results(
        self, windows: list[WindowResult], data: dict[str, pd.DataFrame]
    ) -> dict[str, BacktestResult]:
        """Aggregate results per symbol across all windows."""
        symbol_trades: dict[str, list] = {}

        for w in windows:
            for t in w.backtest_result.trades:
                sym = t.get("symbol", "unknown")
                symbol_trades.setdefault(sym, []).append(t)

        results = {}
        # BT-M7 / W-capital: the denominator is the ONE shared pool. Each symbol's
        # P&L is attributed against the account it was drawn from (the BT-H3
        # sleeve model is superseded by allocate_pool), so per-symbol returns
        # add up to the pooled return instead of each reading as its own account.
        per_symbol_capital = self._config.initial_capital
        for sym, trades in symbol_trades.items():
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            results[sym] = BacktestResult(
                trades=trades,
                initial_capital=per_symbol_capital,
                final_capital=per_symbol_capital + total_pnl,
            )

        return results

    @staticmethod
    def _compute_strategy_results(windows: list[WindowResult]) -> dict[str, dict]:
        """Aggregate results per strategy across all windows."""
        strat_trades: dict[str, list] = {}

        for w in windows:
            for t in w.backtest_result.trades:
                strat = t.get("strategy", "unknown")
                strat_trades.setdefault(strat, []).append(t)

        results = {}
        for strat, trades in strat_trades.items():
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            results[strat] = {
                "trades": len(trades),
                "wins": wins,
                "losses": len(trades) - wins,
                "win_rate": wins / len(trades) if trades else 0,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(trades) if trades else 0,
            }

        return results

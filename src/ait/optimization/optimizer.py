"""StrategyOptimizer — Bayesian hyperparameter search via Optuna.

Wraps the existing Backtester to explore strategy parameter spaces and ML
hyperparameter spaces using TPE sampling with median pruning.

Usage:
    opt = StrategyOptimizer(
        symbols=["SPY", "QQQ"],
        strategies=["iron_condor"],
        n_trials=100,
        objective="composite",
    )
    result = opt.run()
    print(result.summary())
    result.save()
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from typing import Any

import optuna
import pandas as pd

from ait.optimization.objectives import OBJECTIVES
from ait.optimization.param_spaces import (
    DTE_SEARCH_REPLACES, FRACTAL_GATE_SPACE, ML_SPACES, STRATEGY_SPACES, dte_search_space,
    effective_space, live_baselines, override_warnings, space_hash,
)
from ait.optimization.results import OptimizationResult
from ait.utils.logging import get_logger

log = get_logger("optimization.optimizer")

optuna.logging.set_verbosity(optuna.logging.WARNING)

_FETCH_WEEKEND_HOLIDAY_BUFFER_DAYS = 30
_FETCH_MIN_PERIOD_DAYS = 90
_MIN_BACKTEST_ROWS = 60


class _EarlyStopCallback:
    """Stop an Optuna study after `patience` consecutive non-improving trials."""

    def __init__(self, patience: int) -> None:
        self._patience = patience
        self._best = float("-inf")
        self._no_improve = 0
        self.triggered_at: int | None = None  # trial number that triggered the stop

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_value > self._best:
            self._best = study.best_value
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self._patience:
            self.triggered_at = trial.number
            study.stop()

    @property
    def stop_reason(self) -> str:
        if self.triggered_at is not None:
            return (
                f"Early-stopped: {self._patience} consecutive non-improving trials "
                f"(patience reached at trial {self.triggered_at})."
            )
        return ""


def _top_rejections(funnel: dict | None, n: int = 3) -> list[str]:
    """W67: the n most frequent top-level entry rejections of a trial, "reason=days" strings,
    stored on the trial so a starved study explains itself."""
    if not funnel:
        return []
    top = sorted(((k, v) for k, v in funnel.items()
                  if k not in ("days_evaluated", "entries_taken")
                  and not k.startswith(("builder:", "would_reject:"))), key=lambda kv: -kv[1])
    return [f"{k}={v}" for k, v in top[:n]]


class StrategyOptimizer:
    """Bayesian strategy + ML hyperparameter optimizer."""

    def __init__(
        self,
        symbols: list[str],
        strategies: list[str],
        n_trials: int = 100,
        n_jobs: int = 1,
        objective: str = "sharpe_ratio",
        study_name: str | None = None,
        storage: str | None = None,
        optimize_ml: bool = False,
        initial_capital: float | None = None,     # W90a: None -> account.capital (was a stale 50_000)
        train_days: int = 365,
        features_cache: "pd.DataFrame | None" = None,
        # W64: None -> backtest.position_size_pct, else risk.max_position_risk_pct
        # (live's per-trade budget). Was a 0.05 literal; live funded 0.0133-0.03.
        position_size_pct: float | None = None,
        # W64 live parity (see Backtester): None -> config.
        budget_aware_wings: bool | None = None,
        max_contracts_per_trade: int | None = None,
        economic_calendar: Any = None,   # W65: research macro calendar (None = hardcoded)
        condor_any_direction: bool | None = None,   # W66 (None -> config)
        friction_model: str | None = None,           # W66 friction v2 (None -> config)
        friction_entry_capture: float | None = None,
        friction_exit_capture: float | None = None,
        friction_stress_multiplier: float | None = None,
        gates_mode: str | None = None,               # W67 gate parity (None -> config)
        iv_rank_gate: bool | None = None,            # W68 IV-rank parity (None -> config)
        iv_rank_rise_gate: bool | None = None,       # W71c research-only rise veto on/off (None -> config)
        regime_veto_gate: bool | None = None,        # W81 research-only regime veto on/off (None -> config)
        ex_div_entry_gate: str | None = None,        # W84 live's ex-div entry block twin (None -> config)
        ex_div_entry_k_days: int | None = None,      # W84 K for last_k_days (None -> config); not searched
        iv_rank_series: dict | None = None,          # W68 {symbol: rank by date}, full history
        fractal_veto_spread: float | None = None,
        penalty_on_range_prob: float | None = None,
        hard_min_trades: int | None = None,          # W67 graded objective (None -> config)
        below_floor_gradient: float | None = None,
        table_prefix: str = "",         # W65: research table prefix for the standalone data fetch
        db_path: Any = None,             # W65: store holding the research tables
        history_start_date: Any = None,  # W65: trim the standalone fetch to this date
        wing_floor_dollars: float | None = None,   # W70: None -> backtest.wing_floor_dollars (was 5.0)
        # R20b review follow-up: was a hardcoded 1.0 that shadowed the
        # promoted live/config value (1.6) for any direct optimizer caller
        # (run_optimizer.py never overrides it) -- same defect class as the
        # other frozen literals below. None resolves from _bt_cfg.wing_k
        # (set below from load_settings().backtest) once _bt_cfg exists.
        wing_k: float | None = None,
        # R20b review follow-up: was a hardcoded 0.20 literal, shadowing the
        # config/live value for any direct caller (run_optimizer.py never
        # overrides it) -- same defect class as wing_k above. None resolves
        # from _bt_cfg.iv_floor once _bt_cfg exists.
        iv_floor: float | None = None,
        delta_iv_scale: float | None = None,       # W70: None -> backtest.delta_iv_scale
        patience: int = 0,
        min_trades: int = 10,
        max_concurrent_positions: int | None = None,   # W86: None -> backtest.max_concurrent_positions
        max_entry_vol_annual: float | None = None,   # W70: None -> backtest.max_entry_vol_annual
        # R20 #2: baseline values for the two iron_condor gate params Optuna
        # searches (param_spaces Exp 26/28). They were absent from bt_kwargs,
        # and the trial-override loop only applies params already present
        # there — so both suggestions were silently DROPPED every trial: the
        # objective was flat noise along those dimensions while walkforward
        # still applied the arbitrary "best" values OOS. Threaded from the
        # caller (walkforward passes its config) like wing_k/iv_floor.
        # R20b (pre-registered PLAN 2026-08-21): defaults were literals
        # (0.30 / 0.05); None now resolves from load_settings().backtest
        # (iv_rank_rise_threshold / min_edge_over_baseline). Explicit caller
        # values always win.
        # Keyword-only from here: inserted mid-signature, a positional
        # caller would otherwise silently misbind into these two params.
        *,
        iv_rank_rise_threshold: float | None = None,
        min_edge_over_baseline: float | None = None,
        # PR#7 review 2026-08-25: walkforward threads wing/IV/gate overrides
        # into every per-window optimizer, but NOT these three non-searched
        # trial baselines — an explicitly configured historical run trained
        # IC trials at the loaded-YAML values and then evaluated OOS with the
        # caller's explicit values (train/OOS parity break, same class as the
        # R20 #2 dropped-dimension bug). None resolves from config exactly as
        # before; explicit caller values now win, matching wing_k/iv_floor.
        stop_loss_pct: float | None = None,
        profit_target_pct: float | None = None,
        min_confidence: float | None = None,
        # W87a: per-study overrides of single search ranges, ``{strategy: {dimension: [low, high]}}``
        # (a study file's ``search_space:``). Validated here, at construction, by ``effective_space``.
        # None = the code defaults, unchanged.
        search_space_overrides: "dict | None" = None,
        seed: int | None = None,                 # W90a: None -> backtest.walkforward_tuning.optimize_seed (was 42)
        intraday_store: "Any | None" = None,
        symbol: str | None = None,
        range_predictor: "Any | None" = None,
        # W78/W79a: optional per-trial hook ``(entry_dte, expiry_guard_dte, max_hold_fraction)
        # -> predictor`` (each None = the study's base). When set, each trial's Backtester is
        # gated by the range model for THAT trial's reachable horizon; None keeps the single
        # model above for every trial, as before.
        range_predictor_provider: "Any | None" = None,
        val_split: bool = False,
        # R20b review follow-up: walkforward builds vix_ctx for the training
        # window and threads it into features_cache above, but every trial's
        # Backtester was still constructed without it — trials priced entries
        # with the synthetic realized-vol fallback while the selected params
        # are evaluated OOS with real VIX pricing (same defect class as the
        # engine.py entry-pricing fix). Threaded through to _run_backtest.
        market_context: "dict | None" = None,
        # R20b review follow-up: optional pre-loaded Settings, mirroring
        # Backtester's own `settings=` param -- lets a caller building many
        # StrategyOptimizer instances (WalkForwardBacktester: one per window
        # per strategy) load config.yaml ONCE and thread it through instead
        # of every construction re-reading + re-validating the file from
        # disk. None (default) preserves the original behavior — load it here.
        settings: "Any | None" = None,
        # W79a: baselines for the DTE dimensions (None = the engine resolves them from
        # settings.backtest, exactly as before) plus the opt-in search block. A searched value
        # overrides its baseline through the same loop as every other param.
        entry_dte: int | None = None,
        expiry_guard_dte: int | None = None,
        max_hold_fraction: float | None = None,
        flatten_minutes_before_close: int | None = None,
        delta_short: float | None = None,        # L1: the searched value must also be the OOS one
        dte_search: "Any | None" = None,         # DteSearchConfig | None
        multivariate: bool = False,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVES)}")

        self._symbols = symbols
        self._strategies = strategies
        # W87a: the EFFECTIVE per-strategy spaces (defaults + validated overrides), built once. STRATEGY_SPACES
        # itself is never mutated. Every read of a strategy's space below goes through ``_space``.
        self._space_overrides = dict(search_space_overrides or {})
        _stray = sorted(set(self._space_overrides) - set(strategies))
        if _stray:
            raise ValueError(f"search_space overrides for {_stray}, which this study does not run "
                             f"(strategies: {list(strategies)})")
        self._spaces = {st: effective_space(st, self._space_overrides.get(st))
                        for st in strategies if st in STRATEGY_SPACES}
        for _st, _over in self._space_overrides.items():          # W87b: a range that excludes the live value
            for _w in override_warnings(_st, _over, live_baselines(_st)):
                log.warning("search_space_excludes_live_value", detail=_w)
        self._n_trials = n_trials
        self._n_jobs = n_jobs
        self._objective_name = objective
        self._study_name = study_name or f"ait_{'_'.join(strategies[:2])}_{objective}"
        self._storage = storage
        self._optimize_ml = optimize_ml
        self._initial_capital_arg = initial_capital       # resolved below, once the settings are loaded
        self._train_days = train_days
        self._features_cache = features_cache
        # self._wing_floor_dollars / _delta_iv_scale / _max_entry_vol_annual resolved below (W70)
        self._economic_calendar = economic_calendar
        self._table_prefix = table_prefix
        self._db_path = db_path
        self._history_start_date = history_start_date
        # W65: params removed from the search because they cannot affect the result
        # (see _maybe_drop_inert_params); filled in run() once data is known.
        self._dropped_params: set[str] = set()
        # self._iv_floor resolved below once _bt_cfg exists (R20b review follow-up)
        self._patience = patience
        self._min_trades = min_trades
        # R20b (pre-registered PLAN 2026-08-21): resolve the config-backed
        # trial BASELINES once, from the same loaded settings live reads
        # (mirrors Backtester's R20 None->load_settings pattern). These feed
        # the non-searched engine params in _run_backtest's bt_kwargs, which
        # were frozen literals — an operator/config change could never reach
        # a trial backtest. Guarded so a missing config.yaml (or a partial
        # stub in tests) degrades to the BacktestConfig field defaults.
        # R20b review follow-up: reuse a caller-supplied `settings` (e.g.
        # WalkForwardBacktester loads it once and threads it through per
        # window/strategy) instead of unconditionally re-reading it here.
        if settings is not None:
            self._settings = settings
            _bt_cfg = self._settings.backtest
        else:
            try:
                from ait.config.settings import load_settings as _ls
                self._settings = _ls()
                _bt_cfg = self._settings.backtest
            except Exception:  # noqa: BLE001 — no config.yaml -> field defaults
                self._settings = None
                _bt_cfg = None
        if _bt_cfg is None:
            from ait.config.settings import BacktestConfig as _BTC
            _bt_cfg = _BTC()
        self._bt_baselines = _bt_cfg
        # W88b: research constants that were module literals (validation split, inert-parameter probe,
        # objective weights) come from backtest.*; a stub settings object degrades to the schema defaults.
        from ait.config.settings import ObjectivesConfig as _OC, WalkForwardTuningConfig as _WTC
        self._tuning = getattr(_bt_cfg, "walkforward_tuning", None) or _WTC()
        from ait.config.settings import resolve_account_capital as _rac
        self._initial_capital = _rac(self._initial_capital_arg, self._settings)   # W90a: account.capital, not 50_000
        self._objectives_cfg = getattr(_bt_cfg, "objectives", None) or _OC()
        # W64: per-trade budget + live-parity switches, resolved once the loaded
        # settings exist (explicit arg > config > model default).
        from ait.config.settings import (
            BacktestConfig as _BTC2, resolve_config_value as _rcv,
            resolve_max_contracts_per_trade as _rmc,
            resolve_position_size_pct as _rps)
        self._position_size_pct = _rps(position_size_pct, self._settings)
        self._max_contracts_per_trade = _rmc(max_contracts_per_trade, self._settings)
        self._budget_aware_wings = bool(_rcv(
            budget_aware_wings, "backtest", "budget_aware_wings", _BTC2, self._settings))
        self._condor_any_direction = bool(_rcv(
            condor_any_direction, "backtest", "condor_any_direction", _BTC2, self._settings))
        self._friction_model = str(_rcv(
            friction_model, "backtest", "friction_model", _BTC2, self._settings))
        self._friction_entry_capture = float(_rcv(
            friction_entry_capture, "backtest", "friction_entry_capture", _BTC2, self._settings))
        self._friction_exit_capture = float(_rcv(
            friction_exit_capture, "backtest", "friction_exit_capture", _BTC2, self._settings))
        self._friction_stress_multiplier = float(_rcv(
            friction_stress_multiplier, "backtest", "friction_stress_multiplier", _BTC2,
            self._settings))
        self._gates_mode = str(_rcv(gates_mode, "backtest", "gates_mode", _BTC2, self._settings))
        self._iv_rank_gate = bool(_rcv(
            iv_rank_gate, "backtest", "iv_rank_gate", _BTC2, self._settings))
        self._iv_rank_rise_gate = bool(_rcv(
            iv_rank_rise_gate, "backtest", "iv_rank_rise_gate", _BTC2, self._settings))
        self._regime_veto_gate = bool(_rcv(
            regime_veto_gate, "backtest", "regime_veto_gate", _BTC2, self._settings))   # W81
        # W86 (operator, 2026-10-21): trials follow the config's concurrency (was a hard-coded 1).
        self._max_concurrent_positions = int(_rcv(
            max_concurrent_positions, "backtest", "max_concurrent_positions", _BTC2, self._settings))
        self._ex_div_entry_gate = str(_rcv(
            ex_div_entry_gate, "backtest", "ex_div_entry_gate", _BTC2, self._settings))     # W84
        self._ex_div_entry_k_days = int(_rcv(
            ex_div_entry_k_days, "backtest", "ex_div_entry_k_days", _BTC2, self._settings))  # W84
        # W70: were literals 5.0 / 0.0 / 0.80 (and trailing 0.25, breakeven 0.30, delta 0.20 /
        # 0.30 in _run_backtest's bt_kwargs); the wing floor moves 5.0 -> 2.0 (live parity).
        # W70: the SAME contract call live uses (AIT_IC_WING_FLOOR) — live AND research.
        from ait.config.runtime_env import contract_float as _cf
        self._wing_floor_dollars = (float(wing_floor_dollars) if wing_floor_dollars is not None
                                    else _cf("AIT_IC_WING_FLOOR"))
        self._delta_iv_scale = float(_rcv(
            delta_iv_scale, "backtest", "delta_iv_scale", _BTC2, self._settings))
        self._max_entry_vol_annual = float(_rcv(
            max_entry_vol_annual, "backtest", "max_entry_vol_annual", _BTC2, self._settings))
        from ait.config.settings import ExitConfig as _EXC
        self._trailing_stop_pct = float(_rcv(
            None, "exit", "trailing_stop_pct", _EXC, self._settings))
        self._breakeven_trigger_pct = float(_rcv(
            None, "exit", "breakeven_trigger_pct", _EXC, self._settings))
        self._delta_short = float(_rcv(delta_short, "backtest", "delta_short", _BTC2, self._settings))
        self._entry_dte = None if entry_dte is None else int(entry_dte)
        self._expiry_guard_dte = None if expiry_guard_dte is None else int(expiry_guard_dte)
        self._max_hold_fraction = None if max_hold_fraction is None else float(max_hold_fraction)
        self._flatten_minutes = (None if flatten_minutes_before_close is None
                                 else int(flatten_minutes_before_close))
        self._dte_search = dte_search
        self._multivariate = bool(multivariate)
        self._delta_long = float(_rcv(None, "backtest", "delta_long", _BTC2, self._settings))
        self._iv_rank_series = iv_rank_series or {}
        self._fractal_veto_spread = float(_rcv(
            fractal_veto_spread, "backtest", "fractal_veto_spread", _BTC2, self._settings))
        self._penalty_on_range_prob = float(_rcv(
            penalty_on_range_prob, "backtest", "penalty_on_range_prob", _BTC2, self._settings))
        self._hard_min_trades = int(_rcv(
            hard_min_trades, "backtest", "optimize_hard_min_trades", _BTC2, self._settings))
        self._below_floor_gradient = float(_rcv(
            below_floor_gradient, "backtest", "optimize_below_floor_gradient", _BTC2,
            self._settings))
        # R20b review follow-up: same None -> config-resolution pattern as
        # iv_rank_rise_threshold/min_edge_over_baseline below — a direct
        # caller that omits wing_k now gets the config/live value (1.6)
        # instead of the frozen 1.0 literal.
        self._wing_k = float(wing_k) if wing_k is not None else float(_bt_cfg.wing_k)
        # R20b review follow-up: same None -> config-resolution pattern as
        # wing_k above -- a direct caller that omits iv_floor now gets the
        # config/live value instead of the frozen 0.20 literal.
        self._iv_floor = float(iv_floor) if iv_floor is not None else float(_bt_cfg.iv_floor)
        self._iv_rank_rise_threshold = (
            float(iv_rank_rise_threshold) if iv_rank_rise_threshold is not None
            else float(_bt_cfg.iv_rank_rise_threshold)
        )
        self._min_edge_over_baseline = (
            float(min_edge_over_baseline) if min_edge_over_baseline is not None
            else float(_bt_cfg.min_edge_over_baseline)
        )
        # PR#7 review 2026-08-25: explicit caller values (walkforward threads
        # its config-resolved copies) win over the config baselines so trial
        # training and OOS evaluation see the SAME numbers.
        self._stop_loss_pct = (
            float(stop_loss_pct) if stop_loss_pct is not None
            else float(_bt_cfg.stop_loss_pct)
        )
        self._profit_target_pct = (
            float(profit_target_pct) if profit_target_pct is not None
            else float(_bt_cfg.profit_target_pct)
        )
        # R20b follow-up: min_confidence in _run_backtest's bt_kwargs was a
        # hardcoded 0.55 even though the sibling literals above were migrated
        # to config resolution in this same PR -- same defect class, lives on
        # RiskConfig (not BacktestConfig) since that's min_confidence's
        # documented home (see engine.py's resolution comment). Reuses
        # self._settings (loaded once above), not a second independent read.
        if min_confidence is not None:
            self._min_confidence = float(min_confidence)
        else:
            try:
                self._min_confidence = float(self._settings.risk.min_confidence)
            except Exception:  # noqa: BLE001 — no config.yaml -> field default
                from ait.config.settings import RiskConfig as _RC
                self._min_confidence = float(_RC().min_confidence)
        self._seed = int(seed if seed is not None else self._tuning.optimize_seed)
        self._intraday_store = intraday_store
        self._symbol = symbol
        self._range_predictor = range_predictor
        self._range_predictor_provider = range_predictor_provider
        self._val_split = val_split
        self._market_context = market_context
        self._data: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        prior_params: dict | None = None,
    ) -> OptimizationResult:
        """Fetch data if needed, create/resume Optuna study, and run optimization.

        Args:
            data: Optional preloaded OHLCV data keyed by symbol. When provided,
                this data is used as-is and no fetch is performed.
            prior_params: Optional best params from the previous window. When
                provided (and the prior window out-of-sample result was strong),
                the first trial is seeded with these values (warm-start).
        """
        if data is not None:
            self._data = data
        elif not self._data:
            self._data = self._fetch_data()

        if not self._data:
            raise RuntimeError("No market data fetched — check internet connection or symbol names.")

        self._maybe_drop_inert_params()
        if prior_params and self._dropped_params:
            prior_params = {k: v for k, v in prior_params.items() if k not in self._dropped_params}

        study = optuna.create_study(
            direction="maximize",
            study_name=self._study_name,
            storage=self._storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                seed=self._seed, multivariate=self._multivariate, group=self._multivariate),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )

        # W87a: a stored study (``storage=`` + ``load_if_exists``) that was created over a DIFFERENT search
        # space would silently mix two distributions (Optuna only rejects a changed TYPE). Refuse it; a
        # study created before this guard has no hash and is adopted.
        _space_hash = self.search_space_hash()
        _stored = study.user_attrs.get("search_space_hash")
        if _stored is not None and _stored != _space_hash:
            raise ValueError(
                f"study {self._study_name!r} was created over search space {_stored}, this run's space is "
                f"{_space_hash}: use a new study name (or storage) for a changed search space")
        study.set_user_attr("search_space_hash", _space_hash)

        if prior_params:
            study.enqueue_trial(prior_params)

        callbacks = []
        early_stop_cb: _EarlyStopCallback | None = None
        if self._patience > 0:
            early_stop_cb = _EarlyStopCallback(self._patience)
            callbacks.append(early_stop_cb)

        log.info(
            "optuna_study_starting",
            study_name=self._study_name,
            n_trials=self._n_trials,
            objective=self._objective_name,
            existing_trials=len(study.trials),
            patience=self._patience,
            min_trades=self._min_trades,
            warm_start=prior_params is not None,
        )

        study.optimize(
            self._objective_fn,
            n_trials=self._n_trials,
            n_jobs=self._n_jobs,
            callbacks=callbacks or None,
        )

        early_stopped = early_stop_cb is not None and early_stop_cb.triggered_at is not None
        stop_reason = (
            early_stop_cb.stop_reason if early_stopped
            else f"Completed all {len(study.trials)} trials."
        )
        if early_stopped and early_stop_cb is not None:
            n_pruned = len([t for t in study.trials if t.state.name == "PRUNED"])
            if n_pruned:
                stop_reason += f" {n_pruned} trial(s) pruned by MedianPruner."

        result = OptimizationResult(study)
        best_n = study.best_trial.user_attrs.get("n_trades", 0)
        result.best_n_trades = int(best_n)
        result.usable = int(best_n) >= self._hard_min_trades
        if not result.usable:
            log.warning(
                "optuna_no_usable_trial", study_name=self._study_name,
                best_value=study.best_value, best_n_trades=int(best_n),
                hard_min_trades=self._hard_min_trades, n_trials=len(study.trials),
                top_rejections=study.best_trial.user_attrs.get("entry_funnel_top"),
                note="no trial reached the trade floor: the reported best params are NOT an "
                     "optimization result")
        result.stop_reason = stop_reason
        result.early_stopped = early_stopped

        log.info(
            "optuna_study_complete",
            best_value=study.best_value,
            best_params=study.best_params,
            early_stopped=early_stopped,
        )
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _objective_fn(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)

        try:
            result = self._run_backtest(params)
        except Exception as e:
            log.debug("trial_backtest_failed", trial=trial.number, error=str(e))
            raise optuna.TrialPruned()

        value = OBJECTIVES[self._objective_name](result, self._objectives_cfg)

        # Two-tier trade-count guard — prevents degenerate low-sample Sharpe inflation.
        # Tier 1: hard floor — fewer than 3 trades is always a loser regardless of score.
        # Tier 2: quadratic penalty between 3 and min_trades — (actual/min)^2 multiplier.
        #   Linear (×0.5 at 5 trades) leaves overfit trials competitive; quadratic (×0.25)
        #   pushes them below healthy 10-15 trade results.
        # W67: the floor and the below-floor score are config knobs, and the score is
        # GRADED (-100 at 0 trades rising by `below_floor_gradient` as trades approach the
        # floor) instead of a flat -100 -- a flat plateau gave TPE nothing to climb, and 28
        # of 30 W65 studies never left it.
        trial.set_user_attr("n_trades", int(result.total_trades))
        trial.set_user_attr("entry_funnel_top", _top_rejections(result.entry_funnel))
        if result.total_trades < self._hard_min_trades:
            return -100.0 + self._below_floor_gradient * (
                result.total_trades / max(self._hard_min_trades, 1))
        if self._min_trades > 0 and result.total_trades < self._min_trades:
            _f = (result.total_trades / self._min_trades) ** 2
            # R5 audit: multiplying a NEGATIVE score by a <1 factor shrinks
            # the loss toward zero — degenerate 3-9 trade configs outranked
            # honest 12-trade losers. Penalize magnitude symmetrically.
            value = value * _f if value > 0 else value / _f

        # Store per-trial metrics for dashboard Optuna tab (Layer 2b).
        trial.set_user_attr("sharpe",        round(float(result.sharpe_ratio), 4))
        trial.set_user_attr("win_rate",      round(float(result.win_rate), 4))
        trial.set_user_attr("max_drawdown",  round(float(result.max_drawdown), 4))
        trial.set_user_attr("n_trades",      int(result.total_trades))

        # Report intermediate value for pruning (single-step — step=0)
        trial.report(value, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return value

    def _maybe_drop_inert_params(self) -> None:
        """W65: drop iron_condor__wing_k from the search when the W64 budget cap makes
        it inert.

        With budget_aware_wings the wing is min(wing_k x expected move, budget/90).
        At a small account (the $5k / 5% budget -> a ~$3 wing) the cap binds for every
        wing_k in the searched range, so Optuna would spend trials on a dimension that
        cannot change any result (pure noise). Decided from the loaded data at the LOW
        end of IV (smallest expected move), so it only drops when the cap certainly
        binds; at a larger capital the cap stops binding and wing_k is searched again.
        """
        self._dropped_params = set()
        # W67: with a condor-only strategy set the fractal gate's penalty cannot move any
        # result (condors skip the directional-confidence gate) unless it also acts on the
        # range probability, and the veto is an absolute cutoff, not derived from the
        # threshold. Searching the three FRACTAL_GATE_SPACE dimensions is then pure noise.
        from ait.backtesting.engine import NEUTRAL_CREDIT_GATED
        if (self._strategies and set(self._strategies) <= set(NEUTRAL_CREDIT_GATED)
                and self._penalty_on_range_prob == 0):
            _fractal = {f"{st}__{name}" for st in self._strategies
                        for name in FRACTAL_GATE_SPACE}
            self._dropped_params |= _fractal
            log.info("optimizer_dropped_inert_param", param="fractal_gate_space",
                     dropped=sorted(_fractal),
                     note="condor-only strategy set and penalty_on_range_prob=0: the fractal "
                          "penalty cannot change any entry")
        # W71c: with the research-only rise veto switched OFF the threshold cannot change a result,
        # and a searched value would otherwise be applied through the override loop (the dimension
        # is searched 0.25-0.60 whatever the baseline says). IRON_CONDOR_SPACE keeps the dimension
        # for the day the gate is turned back on.
        if not self._iv_rank_rise_gate:
            _rise = {f"{st}__iv_rank_rise_threshold" for st in self._strategies
                     if "iv_rank_rise_threshold" in self._space(st)}
            if _rise:
                self._dropped_params |= _rise
                log.info("optimizer_dropped_inert_param", param="iv_rank_rise_threshold",
                         dropped=sorted(_rise),
                         note="backtest.iv_rank_rise_gate is false: the veto is off, so the "
                              "threshold cannot change any entry")
        if not self._budget_aware_wings or "iron_condor" not in self._strategies:
            return
        spec = self._space("iron_condor").get("wing_k")
        if not spec or not self._data:
            return
        import math
        from ait.strategies.iron_condor import IronCondor
        k_min = float(spec[1])
        affordable = IronCondor.affordable_width(self._initial_capital * self._position_size_pct)
        try:
            dte = int(self._settings.options.dte_range[0]) if self._settings else self._tuning.inert_probe_dte_fallback
        except Exception:  # noqa: BLE001
            dte = self._tuning.inert_probe_dte_fallback
        # W79a: the inertness test must hold at the SMALLEST DTE the study can pick, not the
        # configured band's low end (the expected move, hence the wing target, shrinks with DTE).
        if self._entry_dte is not None:
            dte = self._entry_dte
        if dte_search_space(self._dte_search):
            dte = max(1, int(self._dte_search.entry_dte[0]))
        smallest_target = None
        for df in self._data.values():
            if df is None or df.empty:
                return
            spot = float(df["Close"].iloc[-1])
            iv = self._tuning.inert_probe_iv_fallback
            if "implied_vol" in df.columns and df["implied_vol"].notna().any():
                iv = float(df["implied_vol"].dropna().quantile(self._tuning.inert_probe_iv_quantile))
            target = k_min * spot * iv * math.sqrt(dte / 365.0)
            smallest_target = target if smallest_target is None else min(smallest_target, target)
        if smallest_target is not None and affordable < smallest_target:
            self._dropped_params |= {"iron_condor__wing_k"}
            log.info("optimizer_dropped_inert_param", param="iron_condor__wing_k",
                     affordable_wing=round(affordable, 2),
                     smallest_searched_wing=round(smallest_target, 2),
                     note="the per-trade budget caps the wing below every searched "
                          "wing_k, so the dimension cannot change any result")

    def _space(self, strategy: str) -> dict[str, tuple]:
        """The effective space of one strategy (defaults + this study's validated overrides)."""
        return self._spaces.get(strategy, {})

    def effective_search_space(self) -> dict[str, dict[str, tuple]]:
        """Every dimension a trial would actually sample, ``{strategy: {name: spec}}``: the effective
        spaces after the DTE merge and without the dimensions ``_maybe_drop_inert_params`` removed. This is
        what ``search_space_hash`` hashes and what a run record stores."""
        out: dict[str, dict[str, tuple]] = {}
        for strategy in self._strategies:
            space = self._space(strategy)
            if strategy == "iron_condor":
                _extra = dte_search_space(self._dte_search)
                if _extra:
                    space = {k: v for k, v in space.items() if k not in DTE_SEARCH_REPLACES}
                    space = {**space, **_extra}
            out[strategy] = {n: s for n, s in space.items()
                             if f"{strategy}__{n}" not in self._dropped_params}
        if self._optimize_ml:
            for model, space in ML_SPACES.items():
                out[f"ml:{model}"] = dict(space)
        return out

    def search_space_hash(self) -> str:
        """Short stable hash of ``effective_search_space()`` (see ``param_spaces.space_hash``)."""
        return space_hash(self.effective_search_space())

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {}

        # Strategy-level params — merge spaces for all requested strategies
        for strategy in self._strategies:
            space = self._space(strategy)
            if strategy == "iron_condor":
                _extra = dte_search_space(self._dte_search)
                if _extra:    # W79a: the hold fraction replaces the day cap
                    space = {k: v for k, v in space.items() if k not in DTE_SEARCH_REPLACES}
                    space = {**space, **_extra}
            for name, spec in space.items():
                key = f"{strategy}__{name}"
                if key in params or key in self._dropped_params:
                    continue
                # W79a (flatness check): the expiry guard only exempts entries AT OR BELOW it, so
                # for a trial whose entry DTE is above the searched range the dimension is flat
                # (identical trades at every value). Do not sample it there: a define-by-run
                # conditional, which TPE handles, instead of spending trials on a no-op.
                if (strategy == "iron_condor" and name == "expiry_guard_dte"
                        and f"{strategy}__entry_dte" in params
                        and params[f"{strategy}__entry_dte"] > int(spec[2])):
                    continue
                params[key] = self._suggest_one(trial, key, spec)

        # ML hyperparams (optional)
        if self._optimize_ml:
            for model, space in ML_SPACES.items():
                for name, spec in space.items():
                    key = f"{model}__{name}"
                    params[key] = self._suggest_one(trial, key, spec)

        return params

    @staticmethod
    def _suggest_one(trial: optuna.Trial, name: str, spec: tuple) -> Any:
        kind = spec[0]
        low, high = spec[1], spec[2]
        kwargs: dict = spec[3] if len(spec) > 3 else {}

        if kind == "int":
            return trial.suggest_int(name, int(low), int(high))
        if kind == "float":
            return trial.suggest_float(name, float(low), float(high), **kwargs)
        if kind == "categorical":
            return trial.suggest_categorical(name, list(spec[1]))
        raise ValueError(f"Unknown param type '{kind}'")

    def _run_backtest(self, params: dict) -> Any:
        """Run a single-symbol backtest with the suggested params.

        When val_split=True (H2 mode): the training window is split 80/20.
        Optuna sees only the first 80% (train split) during optimisation; the
        objective is scored on the held-out last 20% (val split). This prevents
        the optimizer from overfitting risk-management parameters to the
        specific price path of the full training window.
        """
        from ait.backtesting.engine import Backtester

        # Use first available symbol for speed (full multi-symbol adds overhead)
        symbol = self._symbols[0]
        df = self._data.get(symbol)
        if df is None or len(df) < _MIN_BACKTEST_ROWS:
            raise ValueError(f"Insufficient data for {symbol}")

        eval_start_date = None
        if self._val_split and len(df) >= _MIN_BACKTEST_ROWS * 2:
            # H2: split 80/20. Full df is passed to Backtester for feature warmup;
            # eval_start_date restricts new entries to the val slice only.
            split_idx = max(_MIN_BACKTEST_ROWS, int(len(df) * (1.0 - self._tuning.optimize_val_split_ratio)))
            val_start = df.index[split_idx]
            eval_start_date = val_start.date() if hasattr(val_start, "date") else val_start
            if len(df) - split_idx < _MIN_BACKTEST_ROWS // 2:
                eval_start_date = None  # val slice too short — fall back to full window

        # Extract Backtester-compatible params (drop strategy__ prefix).
        # Only parameters that Backtester.__init__ actually accepts are included
        # so Optuna optimises values that genuinely influence the objective.
        # R20 #2 INVARIANT: every param a STRATEGY_SPACES entry searches MUST
        # have a baseline key here (or a derived-param handler below) — the
        # override loop applies a trial suggestion only when its bare name is
        # already present, so a missing key silently turns that search
        # dimension into noise (iv_rank_rise_threshold /
        # min_edge_over_baseline were dropped this way for every Exp 26/28
        # trial). test_r20_research_validity pins the full-space coverage.
        # R20b (pre-registered PLAN 2026-08-21): the frozen literals
        # (stop_loss_pct 0.35 / profit_target_pct 0.50 / max_hold_days 30 /
        # hurst 0.20+0.10 / multifractal 0.50) now come from
        # load_settings().backtest (resolved once in __init__), so trial
        # baselines track the operating config instead of 2026-07 snapshots.
        _bl = self._bt_baselines
        bt_kwargs: dict[str, Any] = {
            "initial_capital":       self._initial_capital,
            # PR#7 review 2026-08-25: were float(_bl.stop_loss_pct)/(_bl.
            # profit_target_pct) — always the loaded YAML, ignoring explicit
            # caller overrides that the OOS Backtester WOULD honor.
            "stop_loss_pct":         self._stop_loss_pct,
            "profit_target_pct":     self._profit_target_pct,
            "min_confidence":        self._min_confidence,
            "position_size_pct":     self._position_size_pct,
            "budget_aware_wings":    self._budget_aware_wings,
            "economic_calendar":     self._economic_calendar,
            "condor_any_direction":  self._condor_any_direction,
            "friction_model":        self._friction_model,
            "friction_entry_capture": self._friction_entry_capture,
            "friction_exit_capture": self._friction_exit_capture,
            "friction_stress_multiplier": self._friction_stress_multiplier,
            # W67: the SAME gate configuration the OOS evaluation uses. Before this the
            # trial Backtester inherited the bare engine default (hurst veto at 1.5x), a
            # gate the OOS run never applied -- Optuna optimized under the wrong rules.
            "gates_mode":            self._gates_mode,
            "iv_rank_gate":          self._iv_rank_gate,
            "iv_rank_rise_gate":     self._iv_rank_rise_gate,    # W71c
            "regime_veto_gate":      self._regime_veto_gate,     # W81
            "ex_div_entry_gate":     self._ex_div_entry_gate,    # W84
            "ex_div_entry_k_days":   self._ex_div_entry_k_days,  # W84
            "fractal_veto_spread":   self._fractal_veto_spread,
            "penalty_on_range_prob": self._penalty_on_range_prob,
            "max_contracts_per_trade": self._max_contracts_per_trade,
            "trailing_stop_pct":     self._trailing_stop_pct,
            "breakeven_trigger_pct": self._breakeven_trigger_pct,
            "max_hold_days":         int(_bl.max_hold_days),
            "delta_short":           self._delta_short,
            "delta_long":            self._delta_long,
            "iv_floor":              self._iv_floor,
            "wing_floor_dollars":    self._wing_floor_dollars,
            "wing_k":                    self._wing_k,
            "delta_iv_scale":            self._delta_iv_scale,
            "max_concurrent_positions":  self._max_concurrent_positions,
            "max_entry_vol_annual":      self._max_entry_vol_annual,
            "hurst_regime_threshold":    float(_bl.hurst_regime_threshold),
            "hurst_regime_penalty":      float(_bl.hurst_regime_penalty),
            "multifractal_max_width":    float(_bl.multifractal_max_width),
            "iv_rank_rise_threshold":    self._iv_rank_rise_threshold,
            "min_edge_over_baseline":    self._min_edge_over_baseline,
            # W79a: DTE dimensions. None = the engine resolves from settings.backtest (as
            # before); a searched value replaces the baseline in the override loop below.
            "entry_dte":                    self._entry_dte,
            "expiry_guard_dte":             self._expiry_guard_dte,
            "max_hold_fraction":            self._max_hold_fraction,
            "flatten_minutes_before_close": self._flatten_minutes,
        }
        # Override with trial params that match Backtester signatures
        for key, val in params.items():
            _, _, param_name = key.partition("__")
            if param_name in bt_kwargs:
                bt_kwargs[param_name] = val

        # Derived params: trailing_stop_fraction → trailing_stop_pct relative to profit target.
        # Ensures trailing stop is always a coherent fraction of profit_target, not independent.
        for key, val in params.items():
            _, _, param_name = key.partition("__")
            if param_name == "trailing_stop_fraction":
                bt_kwargs["trailing_stop_pct"] = val * bt_kwargs["profit_target_pct"]

        # H2: pass full df for feature warmup; eval_start_date restricts new entries
        # to the val slice. In non-val_split mode eval_start_date is None (no restriction).
        bt = Backtester(
            data=df,
            strategies=self._strategies,
            features_cache=self._features_cache,
            symbol=self._symbol or symbol,
            # W66: the store this run was pointed at (W52a), so the trial engines read the
            # same calibrated spreads / friction model v2 as the OOS engine instead of the
            # cwd-default historical.db.
            db_path=self._db_path,
            iv_rank_series=self._iv_rank_series.get(self._symbol or symbol),
            range_predictor=(
                self._range_predictor_provider(
                    bt_kwargs.get("entry_dte"), bt_kwargs.get("expiry_guard_dte"),
                    bt_kwargs.get("max_hold_fraction"))
                if self._range_predictor_provider is not None else self._range_predictor),
            intraday_store=self._intraday_store,
            eval_start_date=eval_start_date,
            # R16: Optuna trials must never score with the live future-trained
            # artifact (same look-ahead class as the walkforward OOS fence).
            allow_live_model_fallback=False,
            # R20b follow-up: reuse the settings loaded once in __init__ —
            # every trial otherwise re-read + re-validated config.yaml from
            # disk (BT-E1: N windows x M trials redundant reloads/run).
            settings=self._settings,
            # R20b review follow-up: forward the training-window VIX/SPY
            # context so trial entries price with the same VIX branch as the
            # OOS evaluation, instead of the synthetic realized-vol fallback.
            market_context=self._market_context,
            **bt_kwargs,
        )
        return bt.run()

    def _fetch_data(self) -> dict[str, pd.DataFrame]:
        """Load daily OHLCV from IB store (fallback: Yahoo Finance)."""
        from ait.data.market_data import load_daily_ohlcv

        data = {}
        requested_days = max(1, int(self._train_days))
        fetch_days = max(
            requested_days + _FETCH_WEEKEND_HOLIDAY_BUFFER_DAYS,
            _FETCH_MIN_PERIOD_DAYS,
        )
        for symbol in self._symbols:
            try:
                df = load_daily_ohlcv(
                    symbol, days=fetch_days, db_path=self._db_path,
                    table_prefix=self._table_prefix, start_date=self._history_start_date,
                ) if self._table_prefix else load_daily_ohlcv(symbol, days=fetch_days)
                if len(df) > 0:
                    df = df.tail(requested_days)
                    if len(df) >= _MIN_BACKTEST_ROWS:
                        data[symbol] = df
                        log.info(
                            "data_fetched_for_optimizer",
                            symbol=symbol,
                            rows=len(df),
                            requested_days=requested_days,
                        )
            except Exception as e:
                log.warning("data_fetch_failed", symbol=symbol, error=str(e))
        return data

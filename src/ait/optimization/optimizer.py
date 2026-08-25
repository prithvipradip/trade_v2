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
from ait.optimization.param_spaces import ML_SPACES, STRATEGY_SPACES
from ait.optimization.results import OptimizationResult
from ait.utils.logging import get_logger

log = get_logger("optimization.optimizer")

optuna.logging.set_verbosity(optuna.logging.WARNING)

_FETCH_WEEKEND_HOLIDAY_BUFFER_DAYS = 30
_FETCH_MIN_PERIOD_DAYS = 90
_MIN_BACKTEST_ROWS = 60
_VAL_SPLIT_RATIO = 0.20   # fraction of training window reserved as validation holdout (H2)


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
        initial_capital: float = 50_000.0,
        train_days: int = 365,
        features_cache: "pd.DataFrame | None" = None,
        position_size_pct: float = 0.05,
        wing_floor_dollars: float = 5.0,
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
        delta_iv_scale: float = 0.0,
        patience: int = 0,
        min_trades: int = 10,
        max_concurrent_positions: int = 1,
        max_entry_vol_annual: float = 0.80,
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
        seed: int = 42,
        intraday_store: "Any | None" = None,
        symbol: str | None = None,
        range_predictor: "Any | None" = None,
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
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{objective}'. Choose from: {list(OBJECTIVES)}")

        self._symbols = symbols
        self._strategies = strategies
        self._n_trials = n_trials
        self._n_jobs = n_jobs
        self._objective_name = objective
        self._study_name = study_name or f"ait_{'_'.join(strategies[:2])}_{objective}"
        self._storage = storage
        self._optimize_ml = optimize_ml
        self._initial_capital = initial_capital
        self._train_days = train_days
        self._features_cache = features_cache
        self._position_size_pct = position_size_pct
        self._wing_floor_dollars = wing_floor_dollars
        # self._iv_floor resolved below once _bt_cfg exists (R20b review follow-up)
        self._delta_iv_scale = delta_iv_scale
        self._patience = patience
        self._min_trades = min_trades
        self._max_concurrent_positions = max_concurrent_positions
        self._max_entry_vol_annual = max_entry_vol_annual
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
        self._seed = seed
        self._intraday_store = intraday_store
        self._symbol = symbol
        self._range_predictor = range_predictor
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

        study = optuna.create_study(
            direction="maximize",
            study_name=self._study_name,
            storage=self._storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self._seed),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )

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

        value = OBJECTIVES[self._objective_name](result)

        # Two-tier trade-count guard — prevents degenerate low-sample Sharpe inflation.
        # Tier 1: hard floor — fewer than 3 trades is always a loser regardless of score.
        # Tier 2: quadratic penalty between 3 and min_trades — (actual/min)^2 multiplier.
        #   Linear (×0.5 at 5 trades) leaves overfit trials competitive; quadratic (×0.25)
        #   pushes them below healthy 10-15 trade results.
        _HARD_MIN_TRADES = 3
        if result.total_trades < _HARD_MIN_TRADES:
            return -100.0
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

    def _suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {}

        # Strategy-level params — merge spaces for all requested strategies
        for strategy in self._strategies:
            space = STRATEGY_SPACES.get(strategy, {})
            for name, spec in space.items():
                key = f"{strategy}__{name}"
                if key in params:
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
            split_idx = max(_MIN_BACKTEST_ROWS, int(len(df) * (1.0 - _VAL_SPLIT_RATIO)))
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
            "trailing_stop_pct":     0.25,
            "breakeven_trigger_pct": 0.30,
            "max_hold_days":         int(_bl.max_hold_days),
            "delta_short":           0.20,
            "delta_long":            0.30,
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
            range_predictor=self._range_predictor,
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
                df = load_daily_ohlcv(symbol, days=fetch_days)
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

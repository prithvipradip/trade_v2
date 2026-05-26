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


class _EarlyStopCallback:
    """Stop an Optuna study after `patience` consecutive non-improving trials."""

    def __init__(self, patience: int) -> None:
        self._patience = patience
        self._best = float("-inf")
        self._no_improve = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_value > self._best:
            self._best = study.best_value
            self._no_improve = 0
        else:
            self._no_improve += 1
        if self._no_improve >= self._patience:
            study.stop()


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
        wing_k: float = 1.0,
        iv_floor: float = 0.20,
        delta_iv_scale: float = 0.0,
        patience: int = 0,
        min_trades: int = 10,
        max_concurrent_positions: int = 1,
        max_entry_vol_annual: float = 0.80,
        seed: int = 42,
        intraday_store: "Any | None" = None,
        symbol: str | None = None,
        range_predictor: "Any | None" = None,
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
        self._wing_k = wing_k
        self._iv_floor = iv_floor
        self._delta_iv_scale = delta_iv_scale
        self._patience = patience
        self._min_trades = min_trades
        self._max_concurrent_positions = max_concurrent_positions
        self._max_entry_vol_annual = max_entry_vol_annual
        self._seed = seed
        self._intraday_store = intraday_store
        self._symbol = symbol
        self._range_predictor = range_predictor
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
        if self._patience > 0:
            callbacks.append(_EarlyStopCallback(self._patience))

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

        log.info(
            "optuna_study_complete",
            best_value=study.best_value,
            best_params=study.best_params,
        )
        return OptimizationResult(study)

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
            value *= (result.total_trades / self._min_trades) ** 2

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
        """Run a single-symbol backtest with the suggested params."""
        from ait.backtesting.engine import Backtester

        # Use first available symbol for speed (full multi-symbol adds overhead)
        symbol = self._symbols[0]
        df = self._data.get(symbol)
        if df is None or len(df) < _MIN_BACKTEST_ROWS:
            raise ValueError(f"Insufficient data for {symbol}")

        # Extract Backtester-compatible params (drop strategy__ prefix).
        # Only parameters that Backtester.__init__ actually accepts are included
        # so Optuna optimises values that genuinely influence the objective.
        bt_kwargs: dict[str, Any] = {
            "initial_capital":       self._initial_capital,
            "stop_loss_pct":         0.35,
            "profit_target_pct":     0.50,
            "min_confidence":        0.55,
            "position_size_pct":     self._position_size_pct,
            "trailing_stop_pct":     0.25,
            "breakeven_trigger_pct": 0.30,
            "max_hold_days":         30,
            "delta_short":           0.20,
            "delta_long":            0.30,
            "iv_floor":              self._iv_floor,
            "wing_floor_dollars":    self._wing_floor_dollars,
            "wing_k":                    self._wing_k,
            "delta_iv_scale":            self._delta_iv_scale,
            "max_concurrent_positions":  self._max_concurrent_positions,
            "max_entry_vol_annual":      self._max_entry_vol_annual,
            "hurst_regime_threshold":    0.20,
            "hurst_regime_penalty":      0.10,
            "multifractal_max_width":    0.50,
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

        bt = Backtester(
            data=df,
            strategies=self._strategies,
            features_cache=self._features_cache,
            symbol=self._symbol or symbol,
            range_predictor=self._range_predictor,
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

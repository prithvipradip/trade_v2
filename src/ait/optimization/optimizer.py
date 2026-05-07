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
        self._data: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, data: dict[str, pd.DataFrame] | None = None) -> OptimizationResult:
        """Fetch data if needed, create/resume Optuna study, and run optimization.

        Args:
            data: Optional preloaded OHLCV data keyed by symbol. When provided,
                this data is used as-is and no fetch is performed.
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
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )

        log.info(
            "optuna_study_starting",
            study_name=self._study_name,
            n_trials=self._n_trials,
            objective=self._objective_name,
            existing_trials=len(study.trials),
        )

        study.optimize(
            self._objective_fn,
            n_trials=self._n_trials,
            n_jobs=self._n_jobs,
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
        if df is None or len(df) < 60:
            raise ValueError(f"Insufficient data for {symbol}")

        # Extract Backtester-compatible params (drop strategy__ prefix).
        # Only parameters that Backtester.__init__ actually accepts are included
        # so Optuna optimises values that genuinely influence the objective.
        bt_kwargs: dict[str, Any] = {
            "initial_capital":       self._initial_capital,
            "stop_loss_pct":         0.35,
            "profit_target_pct":     0.50,
            "min_confidence":        0.55,
            "position_size_pct":     0.05,
            "trailing_stop_pct":     0.25,
            "breakeven_trigger_pct": 0.30,
            "max_hold_days":         30,
            "delta_short":           0.20,
            "delta_long":            0.30,
            "iv_floor":              0.10,
        }
        # Override with trial params that match Backtester signatures
        for key, val in params.items():
            _, _, param_name = key.partition("__")
            if param_name in bt_kwargs:
                bt_kwargs[param_name] = val

        bt = Backtester(
            data=df,
            strategies=self._strategies,
            **bt_kwargs,
        )
        return bt.run()

    def _fetch_data(self) -> dict[str, pd.DataFrame]:
        """Fetch historical OHLCV for all symbols from Yahoo Finance."""
        import yfinance as yf

        data = {}
        requested_days = max(1, int(self._train_days))
        period_days = max(requested_days + 30, 90)  # buffer for weekends/holidays
        for symbol in self._symbols:
            try:
                df = yf.Ticker(symbol).history(period=f"{period_days}d", interval="1d")
                if df is not None and len(df) > 0:
                    df = df.tail(requested_days)
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    if len(df) >= 60:
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

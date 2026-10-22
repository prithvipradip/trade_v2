"""Objective functions for Optuna strategy optimization.

Each function receives a BacktestResult (or WalkForwardResult) and returns a
scalar that Optuna will try to *maximise*.

Usage:
    from ait.optimization.objectives import OBJECTIVES
    value = OBJECTIVES["composite"](backtest_result)
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from ait.config.settings import ObjectivesConfig

if TYPE_CHECKING:
    from ait.backtesting.result import BacktestResult


def _cfg(cfg: "ObjectivesConfig | None") -> ObjectivesConfig:
    """W88b: the weights come from ``backtest.objectives`` (the optimizer passes its own copy; a bare call
    reads the loaded settings, else the schema defaults)."""
    if cfg is not None:
        return cfg
    try:
        from ait.config.settings import load_settings
        return load_settings().backtest.objectives
    except Exception:  # noqa: BLE001 - no config.yaml -> the schema defaults
        return ObjectivesConfig()


def _sharpe(r: "BacktestResult", cfg: "ObjectivesConfig | None" = None) -> float:
    return r.sharpe_ratio


def _composite(r: "BacktestResult", cfg: "ObjectivesConfig | None" = None) -> float:
    c = _cfg(cfg)
    return c.sharpe_weight * r.sharpe_ratio + c.win_rate_weight * r.win_rate - c.drawdown_penalty * abs(r.max_drawdown)


def _profit_factor(r: "BacktestResult", cfg: "ObjectivesConfig | None" = None) -> float:
    pf = r.profit_factor
    # Cap inf to a large finite value so Optuna's pruner can compare
    return min(pf, _cfg(cfg).profit_factor_cap)


def _win_rate(r: "BacktestResult", cfg: "ObjectivesConfig | None" = None) -> float:
    return r.win_rate


OBJECTIVES: dict[str, Callable] = {
    "sharpe_ratio":  _sharpe,
    "composite":     _composite,
    "profit_factor": _profit_factor,
    "win_rate":      _win_rate,
}

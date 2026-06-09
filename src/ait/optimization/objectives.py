"""Objective functions for Optuna strategy optimization.

Each function receives a BacktestResult (or WalkForwardResult) and returns a
scalar that Optuna will try to *maximise*.

Usage:
    from ait.optimization.objectives import OBJECTIVES
    value = OBJECTIVES["composite"](backtest_result)
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ait.backtesting.result import BacktestResult


def _sharpe(r: "BacktestResult") -> float:
    return r.sharpe_ratio


def _composite(r: "BacktestResult") -> float:
    return 0.4 * r.sharpe_ratio + 0.4 * r.win_rate - 0.2 * abs(r.max_drawdown)


def _profit_factor(r: "BacktestResult") -> float:
    pf = r.profit_factor
    # Cap inf to a large finite value so Optuna's pruner can compare
    return min(pf, 10.0)


def _win_rate(r: "BacktestResult") -> float:
    return r.win_rate


OBJECTIVES: dict[str, Callable] = {
    "sharpe_ratio":  _sharpe,
    "composite":     _composite,
    "profit_factor": _profit_factor,
    "win_rate":      _win_rate,
}

"""Tests for walk-forward backtester."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.walkforward import (
    WalkForwardBacktester,
    WalkForwardConfig,
    WalkForwardResult,
    WindowResult,
)
from ait.backtesting.result import BacktestResult


def _make_ohlcv(days: int = 500, start_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    returns = np.random.normal(0.0005, 0.015, days)
    close = start_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, days)))
    open_ = close * (1 + np.random.normal(0, 0.002, days))
    volume = np.random.randint(1_000_000, 10_000_000, days)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


class TestWalkForwardConfig:

    def test_defaults(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.train_days == 365
        assert cfg.test_days == 63
        assert cfg.step_days == 21
        assert cfg.gap_days == 5

    def test_custom_config(self) -> None:
        cfg = WalkForwardConfig(train_days=120, test_days=30, step_days=10)
        assert cfg.train_days == 120


class TestWalkForwardResult:

    def test_empty_result(self) -> None:
        result = WalkForwardResult()
        assert result.total_trades == 0
        assert result.total_return == 0.0
        assert result.win_rate == 0.0
        assert result.consistency == 0.0

    def test_with_windows(self) -> None:
        windows = [
            WindowResult(
                window_id=1,
                train_start=date(2023, 1, 1),
                train_end=date(2023, 12, 31),
                test_start=date(2024, 1, 5),
                test_end=date(2024, 3, 31),
                backtest_result=BacktestResult(
                    trades=[
                        {"pnl": 100, "strategy": "long_call", "symbol": "SPY", "exit_date": "2024-01-15"},
                        {"pnl": -50, "strategy": "iron_condor", "symbol": "QQQ", "exit_date": "2024-02-01"},
                        {"pnl": 75, "strategy": "long_call", "symbol": "SPY", "exit_date": "2024-02-15"},
                    ],
                    initial_capital=10000,
                    final_capital=10125,
                ),
            ),
        ]
        result = WalkForwardResult(windows=windows, initial_capital=10000)
        assert result.total_trades == 3
        assert result.win_rate == pytest.approx(2 / 3)
        assert result.total_return > 0

    def test_summary_format(self) -> None:
        result = WalkForwardResult(initial_capital=10000)
        summary = result.summary()
        assert "WALK-FORWARD" in summary
        assert "Total Trades" in summary

    def test_equity_curve(self) -> None:
        windows = [
            WindowResult(
                window_id=1,
                train_start=date(2023, 1, 1),
                train_end=date(2023, 6, 30),
                test_start=date(2023, 7, 5),
                test_end=date(2023, 9, 30),
                backtest_result=BacktestResult(
                    trades=[
                        {"pnl": 100, "strategy": "long_call", "symbol": "SPY", "exit_date": "2023-07-15"},
                        {"pnl": -30, "strategy": "long_put", "symbol": "QQQ", "exit_date": "2023-08-01"},
                    ],
                    initial_capital=10000,
                    final_capital=10070,
                ),
            ),
        ]
        result = WalkForwardResult(windows=windows, initial_capital=10000)
        curve = result.equity_curve()
        assert len(curve) == 2
        assert "equity" in curve.columns
        assert "date" in curve.columns

    def test_strategy_breakdown(self) -> None:
        windows = [
            WindowResult(
                window_id=1,
                train_start=date(2023, 1, 1),
                train_end=date(2023, 6, 30),
                test_start=date(2023, 7, 5),
                test_end=date(2023, 9, 30),
                backtest_result=BacktestResult(
                    trades=[
                        {"pnl": 100, "strategy": "long_call", "symbol": "SPY"},
                        {"pnl": -30, "strategy": "iron_condor", "symbol": "SPY"},
                    ],
                    initial_capital=10000,
                    final_capital=10070,
                ),
            ),
        ]
        result = WalkForwardResult(windows=windows, initial_capital=10000)
        strat_results = WalkForwardBacktester._compute_strategy_results(windows)
        assert "long_call" in strat_results
        assert "iron_condor" in strat_results
        assert strat_results["long_call"]["wins"] == 1


class TestWalkForwardBacktester:

    def test_generate_windows(self) -> None:
        data = {"SPY": _make_ohlcv(500)}
        cfg = WalkForwardConfig(train_days=200, test_days=50, step_days=25, gap_days=5)
        bt = WalkForwardBacktester(["SPY"], ["long_call"], config=cfg)
        windows = bt._generate_windows(data)

        assert len(windows) >= 1
        for train_start, train_end, test_start, test_end in windows:
            # Gap between train and test
            assert (test_start - train_end).days >= cfg.gap_days
            # Test window correct size
            assert (test_end - test_start).days == cfg.test_days

    def test_benchmark_buy_hold(self) -> None:
        data = {"SPY": _make_ohlcv(252)}
        bt = WalkForwardBacktester(["SPY"], ["long_call"])
        benchmark = bt.benchmark_buy_hold(data)

        assert "SPY" in benchmark
        assert "portfolio" in benchmark
        assert isinstance(benchmark["SPY"], float)

    def test_run_with_data(self) -> None:
        import asyncio

        data = {
            "SPY": _make_ohlcv(1000, start_price=450),
            "QQQ": _make_ohlcv(1000, start_price=380),
        }
        cfg = WalkForwardConfig(
            train_days=350,
            test_days=63,
            step_days=63,
            gap_days=5,
            initial_capital=50000,
        )
        bt = WalkForwardBacktester(
            ["SPY", "QQQ"],
            ["long_call", "bull_call_spread", "iron_condor"],
            config=cfg,
        )
        result = asyncio.run(bt.run(data=data))

        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) >= 1

    def test_max_drawdown(self) -> None:
        windows = [
            WindowResult(
                window_id=1,
                train_start=date(2023, 1, 1),
                train_end=date(2023, 6, 30),
                test_start=date(2023, 7, 5),
                test_end=date(2023, 9, 30),
                backtest_result=BacktestResult(
                    trades=[
                        {"pnl": 200},
                        {"pnl": -500},
                        {"pnl": 100},
                    ],
                    initial_capital=10000,
                    final_capital=9800,
                ),
            ),
        ]
        result = WalkForwardResult(windows=windows, initial_capital=10000)
        assert result.max_drawdown > 0

    def test_consistency(self) -> None:
        windows = [
            WindowResult(
                window_id=i,
                train_start=date(2023, 1, 1),
                train_end=date(2023, 6, 30),
                test_start=date(2023, 7, 5),
                test_end=date(2023, 9, 30),
                backtest_result=BacktestResult(
                    trades=[{"pnl": 100 if i % 2 == 0 else -100}],
                    initial_capital=10000,
                    final_capital=10100 if i % 2 == 0 else 9900,
                ),
            )
            for i in range(4)
        ]
        result = WalkForwardResult(windows=windows, initial_capital=10000)
        assert result.consistency == 0.5  # Half profitable

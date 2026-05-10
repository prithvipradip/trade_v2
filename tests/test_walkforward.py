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

    def test_wing_k_default_is_1(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.wing_k == pytest.approx(1.0)

    def test_delta_iv_scale_default_is_0(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.delta_iv_scale == pytest.approx(0.0)

    def test_max_concurrent_positions_default_is_3(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.max_concurrent_positions == 3

    def test_max_entry_vol_annual_default(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.max_entry_vol_annual == pytest.approx(0.80)


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


# ---------------------------------------------------------------------------
# Fix regression tests — capital sizing and features cache
# ---------------------------------------------------------------------------

def _make_spy_ohlcv(n: int = 60, start: float = 400.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-03", periods=n, freq="B")
    prices = pd.Series(start + pd.Series(range(n)) * 0.2, index=idx)
    return pd.DataFrame({
        "Open":   prices * 0.999,
        "High":   prices * 1.005,
        "Low":    prices * 0.995,
        "Close":  prices,
        "Volume": 1_000_000,
    }, index=idx)


def test_walkforward_config_default_capital_is_100k() -> None:
    cfg = WalkForwardConfig()
    assert cfg.initial_capital == 100_000.0


class TestFeaturesCache:
    """Backtester features_cache parameter wires up correctly."""

    def test_features_cache_skips_recompute(self, monkeypatch) -> None:
        from ait.backtesting.engine import Backtester
        from ait.ml.features import FeatureEngine

        df = _make_spy_ohlcv(60)
        # Build a minimal synthetic cache with the same index as df so every bar hits
        fake_cache = pd.DataFrame(
            {"hurst_wavelet": 0.5, "psd_beta": -1.5},
            index=df.index,
        )
        assert not fake_cache.empty

        calls: list[int] = []
        original = FeatureEngine.compute
        monkeypatch.setattr(
            FeatureEngine, "compute",
            lambda self, d: calls.append(1) or original(self, d),
        )

        bt = Backtester(
            data=df, strategies=["iron_condor"],
            initial_capital=100_000, iv_floor=0.20, features_cache=fake_cache,
        )
        bt.run()
        assert len(calls) == 0, "FeatureEngine.compute must not be called when features_cache is provided"

    def test_features_cache_fallback_when_none(self, monkeypatch) -> None:
        from ait.backtesting.engine import Backtester
        from ait.ml.features import FeatureEngine

        df = _make_spy_ohlcv(60)

        calls: list[int] = []
        original = FeatureEngine.compute
        monkeypatch.setattr(
            FeatureEngine, "compute",
            lambda self, d: calls.append(1) or original(self, d),
        )

        bt = Backtester(
            data=df, strategies=["iron_condor"],
            initial_capital=100_000, iv_floor=0.20, features_cache=None,
        )
        bt.run()
        assert len(calls) > 0, "FeatureEngine.compute must be called when no cache is provided"


# ---------------------------------------------------------------------------
# wing_k dynamic sizing tests
# ---------------------------------------------------------------------------

class TestWingKDynamicSizing:
    """Verify wing_k drives wing width from vol, not the static floor."""

    def _make_df(self, price: float = 500.0, n: int = 60) -> pd.DataFrame:
        idx = pd.date_range("2023-01-03", periods=n, freq="B")
        p = pd.Series([price] * n, index=idx)
        return pd.DataFrame({
            "Open": p, "High": p * 1.01, "Low": p * 0.99, "Close": p, "Volume": 1_000_000,
        })

    def test_wing_k_1_produces_vol_scaled_width(self):
        """wing_k=1.0 → wing_width ≈ expected_move, well above wing_floor_dollars."""
        import datetime
        from ait.backtesting.engine import Backtester

        bt = Backtester(
            data=self._make_df(500.0), strategies=["iron_condor"],
            initial_capital=100_000, iv_floor=0.25, wing_floor_dollars=5.0, wing_k=1.0,
        )
        pos = bt._build_credit_position(
            "iron_condor", S=500.0, iv=0.25, t=30 / 365, r=0.05, dte=30,
            today_date=datetime.date(2023, 3, 1), capital=100_000,
        )
        assert pos is not None
        wing_used = pos["long_call_strike"] - pos["short_call_strike"]
        # expected_move ≈ 500 × 0.25 × √(30/365) ≈ 16.4 → well above $5 floor
        assert wing_used > 5.0, f"wing_width={wing_used} should exceed floor of $5"
        assert wing_used > 10.0, f"expected_move ~16, got {wing_used}"

    def test_wing_floor_is_hard_minimum(self):
        """wing_k=0.001 (near zero) → wing_width falls back to wing_floor_dollars."""
        import datetime
        from ait.backtesting.engine import Backtester

        bt = Backtester(
            data=self._make_df(500.0), strategies=["iron_condor"],
            initial_capital=100_000, iv_floor=0.20, wing_floor_dollars=5.0, wing_k=0.001,
        )
        pos = bt._build_credit_position(
            "iron_condor", S=500.0, iv=0.20, t=30 / 365, r=0.05, dte=30,
            today_date=datetime.date(2023, 3, 1), capital=100_000,
        )
        if pos is not None:
            wing_used = pos["long_call_strike"] - pos["short_call_strike"]
            assert wing_used >= 5.0, "Wing must never go below wing_floor_dollars"

    def test_higher_wing_k_produces_wider_wings(self):
        """wing_k=2.0 should produce wider wings than wing_k=0.5."""
        import datetime
        from ait.backtesting.engine import Backtester

        today = datetime.date(2023, 3, 1)
        kwargs = dict(S=500.0, iv=0.25, t=30 / 365, r=0.05, dte=30,
                      today_date=today, capital=100_000)

        bt_narrow = Backtester(data=self._make_df(500.0), strategies=["iron_condor"],
                               initial_capital=100_000, iv_floor=0.20,
                               wing_floor_dollars=1.0, wing_k=0.5)
        bt_wide = Backtester(data=self._make_df(500.0), strategies=["iron_condor"],
                             initial_capital=100_000, iv_floor=0.20,
                             wing_floor_dollars=1.0, wing_k=2.0)

        pos_narrow = bt_narrow._build_credit_position("iron_condor", **kwargs)
        pos_wide = bt_wide._build_credit_position("iron_condor", **kwargs)

        if pos_narrow and pos_wide:
            wing_narrow = pos_narrow["long_call_strike"] - pos_narrow["short_call_strike"]
            wing_wide = pos_wide["long_call_strike"] - pos_wide["short_call_strike"]
            assert wing_wide > wing_narrow, (
                f"wide wing_k=2.0 ({wing_wide}) should exceed narrow wing_k=0.5 ({wing_narrow})"
            )


# ---------------------------------------------------------------------------
# New strategy tests
# ---------------------------------------------------------------------------

class TestNewStrategies:
    """Verify short_strangle and long_strangle build and reprice correctly."""

    def _make_df(self, price: float = 450.0, n: int = 60) -> pd.DataFrame:
        idx = pd.date_range("2023-01-03", periods=n, freq="B")
        p = pd.Series([price] * n, index=idx)
        return pd.DataFrame({
            "Open": p, "High": p * 1.01, "Low": p * 0.99, "Close": p, "Volume": 1_000_000,
        })

    def test_short_strangle_builds_non_none(self):
        import datetime
        from ait.backtesting.engine import Backtester

        # position_size_pct=0.50 so capital×50% = $50k > margin_per_contract ($9k) → ≥1 contract
        bt = Backtester(data=self._make_df(), strategies=["short_strangle"],
                        initial_capital=100_000, iv_floor=0.20, position_size_pct=0.50)
        pos = bt._build_credit_position(
            "short_strangle", S=450.0, iv=0.20, t=30 / 365, r=0.05, dte=30,
            today_date=datetime.date(2023, 3, 1), capital=100_000,
        )
        assert pos is not None
        assert pos["strategy"] == "short_strangle"
        assert "short_call_strike" in pos
        assert "short_put_strike" in pos
        assert pos["trade_type"] == "credit"

    def test_short_strangle_reprice_positive(self):
        """Cost to buy back a short strangle must be positive."""
        import datetime
        from ait.backtesting.engine import Backtester

        bt = Backtester(data=self._make_df(), strategies=["short_strangle"],
                        initial_capital=100_000, iv_floor=0.20, position_size_pct=0.50)
        pos = bt._build_credit_position(
            "short_strangle", S=450.0, iv=0.20, t=30 / 365, r=0.05, dte=30,
            today_date=datetime.date(2023, 3, 1), capital=100_000,
        )
        assert pos is not None
        current_val = bt._reprice_position(pos, underlying=450.0, days_held=5)
        assert current_val > 0

    def test_long_strangle_builds_non_none(self):
        import datetime
        from ait.backtesting.engine import Backtester
        from ait.strategies.base import SignalDirection

        bt = Backtester(data=self._make_df(), strategies=["long_strangle"],
                        initial_capital=100_000, iv_floor=0.20)
        pos = bt._build_debit_position(
            "long_strangle", direction=SignalDirection.NEUTRAL,
            S=450.0, iv=0.20, t=30 / 365, r=0.05, dte=30,
            today_date=datetime.date(2023, 3, 1), capital=100_000,
        )
        assert pos is not None
        assert pos["strategy"] == "long_strangle"
        assert "long_call_strike" in pos
        assert "long_put_strike" in pos
        assert pos["trade_type"] == "debit"

    def test_long_strangle_reprice_increases_after_large_move(self):
        """Long strangle value should increase when underlying moves significantly."""
        import datetime
        from ait.backtesting.engine import Backtester
        from ait.strategies.base import SignalDirection

        bt = Backtester(data=self._make_df(), strategies=["long_strangle"],
                        initial_capital=100_000, iv_floor=0.20)
        pos = bt._build_debit_position(
            "long_strangle", direction=SignalDirection.NEUTRAL,
            S=450.0, iv=0.20, t=30 / 365, r=0.05, dte=30,
            today_date=datetime.date(2023, 3, 1), capital=100_000,
        )
        assert pos is not None
        val_flat = bt._reprice_position(pos, underlying=450.0, days_held=1)
        val_moved = bt._reprice_position(pos, underlying=490.0, days_held=1)
        assert val_moved > val_flat, "Long strangle should gain value after large underlying move"

    def test_short_strangle_delta_scales_with_iv(self):
        """Higher current IV → lower effective delta → further OTM strikes."""
        import datetime
        from ait.backtesting.engine import Backtester

        today = datetime.date(2023, 3, 1)
        df = self._make_df(450.0)

        bt = Backtester(data=df, strategies=["short_strangle"],
                        initial_capital=100_000, iv_floor=0.20,
                        delta_short=0.20, delta_iv_scale=1.0, position_size_pct=0.50)

        pos_low_iv = bt._build_credit_position(
            "short_strangle", S=450.0, iv=0.20,
            t=30 / 365, r=0.05, dte=30, today_date=today, capital=100_000,
        )
        pos_high_iv = bt._build_credit_position(
            "short_strangle", S=450.0, iv=0.40,
            t=30 / 365, r=0.05, dte=30, today_date=today, capital=100_000,
        )

        if pos_low_iv and pos_high_iv:
            assert pos_high_iv["short_call_strike"] >= pos_low_iv["short_call_strike"], (
                "High IV should push short call strike further OTM (higher)"
            )

    def test_delta_iv_scale_zero_is_static(self):
        """delta_iv_scale=0 vs 1: at high IV, scale=1 places strikes further OTM than scale=0."""
        import datetime
        from ait.backtesting.engine import Backtester

        today = datetime.date(2023, 3, 1)
        df = self._make_df(450.0)

        # scale=0: effective_delta = delta_short = 0.20 regardless of IV
        bt_static = Backtester(data=df, strategies=["short_strangle"],
                               initial_capital=100_000, iv_floor=0.20,
                               delta_short=0.20, delta_iv_scale=0.0, position_size_pct=0.50)
        # scale=1: high IV → effective_delta = 0.20 × (0.20/0.40) = 0.10 → further OTM
        bt_scaled = Backtester(data=df, strategies=["short_strangle"],
                               initial_capital=100_000, iv_floor=0.20,
                               delta_short=0.20, delta_iv_scale=1.0, position_size_pct=0.50)

        kwargs = dict(S=450.0, iv=0.40, t=30 / 365, r=0.05, dte=30,
                      today_date=today, capital=100_000)
        pos_static = bt_static._build_credit_position("short_strangle", **kwargs)
        pos_scaled = bt_scaled._build_credit_position("short_strangle", **kwargs)

        if pos_static and pos_scaled:
            assert pos_scaled["short_call_strike"] >= pos_static["short_call_strike"], (
                "delta_iv_scale=1 at 2× iv_floor should push call strike further OTM than scale=0"
            )

    def test_multi_strategy_run_includes_all_strategies(self):
        """Walk-forward with full strategy list must complete without error."""
        import asyncio

        strategies = [
            "iron_condor", "put_credit_spread", "short_strangle",
            "bull_call_spread", "bear_put_spread", "long_strangle",
        ]
        data = {"SPY": _make_ohlcv(1000, start_price=450)}
        cfg = WalkForwardConfig(
            train_days=350, test_days=63, step_days=63, gap_days=5,
            initial_capital=100_000, wing_k=1.0,
        )
        bt = WalkForwardBacktester(["SPY"], strategies, config=cfg)
        result = asyncio.run(bt.run(data=data))

        assert isinstance(result, WalkForwardResult)
        assert result.total_trades >= 0
        executed = {t["strategy"] for w in result.windows for t in w.backtest_result.trades}
        assert executed.issubset(set(strategies) | {""}), (
            f"Unexpected strategies found: {executed - set(strategies)}"
        )


# ---------------------------------------------------------------------------
# Multi-position tests
# ---------------------------------------------------------------------------

class TestMultiPosition:

    def _make_df(self, price: float = 500.0, n: int = 100) -> pd.DataFrame:
        idx = pd.date_range("2023-01-03", periods=n, freq="B")
        p = pd.Series([price] * n, index=idx)
        return pd.DataFrame({"Open": p, "High": p * 1.01, "Low": p * 0.99,
                              "Close": p, "Volume": 1_000_000})

    def test_single_limit_produces_result(self) -> None:
        """max_concurrent_positions=1 must complete without error."""
        from ait.backtesting.engine import Backtester
        bt = Backtester(data=self._make_df(), strategies=["iron_condor"],
                        initial_capital=100_000, iv_floor=0.20,
                        max_concurrent_positions=1)
        result = bt.run()
        assert result is not None

    def test_multi_position_allows_more_trades(self) -> None:
        """max_concurrent_positions=3 produces >= trades as max_concurrent_positions=1."""
        import asyncio
        data = {"SPY": _make_ohlcv(1000, start_price=450)}
        cfg_single = WalkForwardConfig(
            train_days=350, test_days=63, step_days=63, gap_days=5,
            initial_capital=100_000, max_concurrent_positions=1,
        )
        cfg_multi = WalkForwardConfig(
            train_days=350, test_days=63, step_days=63, gap_days=5,
            initial_capital=100_000, max_concurrent_positions=3,
        )
        r_single = asyncio.run(WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg_single).run(data=data))
        r_multi  = asyncio.run(WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg_multi).run(data=data))
        assert r_multi.total_trades >= r_single.total_trades


# ---------------------------------------------------------------------------
# Realized-vol gate tests
# ---------------------------------------------------------------------------

class TestVolGate:

    def _make_volatile_df(self, base: float = 500.0, n: int = 120,
                          daily_vol: float = 0.04) -> pd.DataFrame:
        """Synthetic data with configurable daily vol (~64% annualized at 4%)."""
        np.random.seed(42)
        prices = [base]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + np.random.normal(0, daily_vol)))
        prices_s = pd.Series(prices)
        idx = pd.date_range("2023-01-03", periods=n, freq="B")
        return pd.DataFrame({"Open": prices_s, "High": prices_s * 1.01,
                              "Low": prices_s * 0.99, "Close": prices_s,
                              "Volume": 1_000_000}, index=idx)

    def test_strict_gate_blocks_entries_in_high_vol(self) -> None:
        """max_entry_vol_annual=0.30 should block most entries when daily_vol=4%."""
        from ait.backtesting.engine import Backtester
        bt = Backtester(data=self._make_volatile_df(daily_vol=0.04),
                        strategies=["iron_condor"], initial_capital=100_000,
                        iv_floor=0.20, max_entry_vol_annual=0.30)
        result = bt.run()
        # 4% daily ≈ 63% annualized >> 30% gate → nearly all entries blocked
        assert result.total_trades < 3

    def test_permissive_gate_allows_entries_in_low_vol(self) -> None:
        """max_entry_vol_annual=0.90 should allow entries when daily_vol=1%."""
        from ait.backtesting.engine import Backtester
        bt = Backtester(data=self._make_volatile_df(daily_vol=0.01),
                        strategies=["iron_condor"], initial_capital=100_000,
                        iv_floor=0.20, max_entry_vol_annual=0.90)
        result = bt.run()
        assert result is not None  # must not crash; entries may or may not occur

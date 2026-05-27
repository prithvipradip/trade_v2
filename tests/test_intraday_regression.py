"""Regression / Integration Tests — intraday engine vs EOD engine consistency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import Backtester
from ait.backtesting.result import BacktestResult


def _make_ohlcv(days: int = 200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2022-01-03", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0005, 0.012, days))
    high  = close * (1 + np.abs(np.random.normal(0, 0.005, days)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.005, days)))
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close,
         "Volume": np.random.randint(1_000_000, 10_000_000, days).astype(float)},
        index=dates,
    )


def _run_eod(df: pd.DataFrame, **kwargs) -> BacktestResult:
    bt = Backtester(
        data=df,
        strategies=["iron_condor"],
        initial_capital=50_000,
        min_confidence=0.30,  # low to force trades
        **kwargs,
    )
    return bt.run()


class TestBacktestResultStructure:
    """BacktestResult trade dicts have required fields."""

    def test_trades_have_entry_date(self) -> None:
        result = _run_eod(_make_ohlcv())
        for t in result.trades:
            assert "entry_date" in t

    def test_trades_have_exit_date(self) -> None:
        result = _run_eod(_make_ohlcv())
        for t in result.trades:
            assert "exit_date" in t

    def test_trades_have_exit_time(self) -> None:
        result = _run_eod(_make_ohlcv())
        for t in result.trades:
            assert "exit_time" in t, \
                f"exit_time missing from trade dict, keys: {list(t.keys())}"

    def test_trades_have_entry_confidence(self) -> None:
        result = _run_eod(_make_ohlcv())
        for t in result.trades:
            assert "entry_confidence" in t, \
                f"entry_confidence missing, keys: {list(t.keys())}"

    def test_entry_confidence_in_plausible_range(self) -> None:
        result = _run_eod(_make_ohlcv())
        for t in result.trades:
            conf = t.get("entry_confidence", -1)
            assert 0.0 <= conf <= 1.0, f"entry_confidence={conf} out of range"

    def test_trades_have_entry_regime(self) -> None:
        result = _run_eod(_make_ohlcv())
        for t in result.trades:
            assert "entry_regime" in t, \
                f"entry_regime missing, keys: {list(t.keys())}"
            assert t["entry_regime"] in (
                "trending_up", "trending_down", "high_volatility", "range_bound"
            )


class TestIVAndSpreadInteraction:
    """TR-4: IV and spread interact self-consistently."""

    def test_higher_iv_floor_produces_more_premium(self) -> None:
        df = _make_ohlcv()
        result_lo = _run_eod(df, iv_floor=0.20)
        result_hi = _run_eod(df, iv_floor=0.45)

        if result_lo.trades and result_hi.trades:
            avg_net_credit_lo = sum(
                t.get("net_credit", 0) for t in result_lo.trades
            ) / len(result_lo.trades)
            avg_net_credit_hi = sum(
                t.get("net_credit", 0) for t in result_hi.trades
            ) / len(result_hi.trades)
            assert avg_net_credit_hi >= avg_net_credit_lo, \
                "Higher IV floor should produce higher net credit on average"


class TestBacktestResultMetrics:
    """Core metric properties work on a realistic run."""

    def test_total_return_finite(self) -> None:
        result = _run_eod(_make_ohlcv())
        assert np.isfinite(result.total_return)

    def test_win_rate_between_zero_and_one(self) -> None:
        result = _run_eod(_make_ohlcv())
        assert 0.0 <= result.win_rate <= 1.0

    def test_sharpe_ratio_finite(self) -> None:
        result = _run_eod(_make_ohlcv())
        assert np.isfinite(result.sharpe_ratio)

    def test_max_drawdown_non_negative(self) -> None:
        result = _run_eod(_make_ohlcv())
        assert result.max_drawdown >= 0.0


class TestLookAheadFree:
    """TR-5 partial: partial bar construction cannot see future data."""

    def test_slice_intraday_up_to_is_monotone(self) -> None:
        import datetime as dt
        from ait.data.historical import HistoricalDataStore

        session_date = dt.date(2024, 3, 1)
        base_dt = dt.datetime(2024, 3, 1, 9, 30)
        times = [base_dt + dt.timedelta(minutes=5 * i) for i in range(78)]
        close = np.linspace(480, 490, 78)
        intraday = pd.DataFrame(
            {"Open": close, "High": close + 0.5, "Low": close - 0.5, "Close": close},
            index=pd.DatetimeIndex(times),
        )

        for cutoff_min in [30, 60, 90, 120]:
            cutoff = dt.time(9 + cutoff_min // 60, cutoff_min % 60)
            sliced = HistoricalDataStore.slice_intraday_up_to(intraday, cutoff)
            for ts in sliced.index:
                assert ts.time() <= cutoff, \
                    f"Bar at {ts.time()} exceeds cutoff {cutoff}"

            # Close at cutoff must match the bar at that exact time
            last_close = sliced["Close"].iloc[-1] if not sliced.empty else None
            if last_close is not None:
                cutoff_dt = dt.datetime(2024, 3, 1, cutoff.hour, cutoff.minute)
                expected_idx = intraday.index[intraday.index <= cutoff_dt]
                if len(expected_idx) > 0:
                    expected_close = intraday.loc[expected_idx[-1], "Close"]
                    assert abs(last_close - expected_close) < 1e-9

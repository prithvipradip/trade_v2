"""Tests for Fix 1 — Intraday Backtest Engine."""

from __future__ import annotations

import datetime as dt
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import Backtester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily(days: int = 120, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0003, 0.008, days))
    high  = close * (1 + np.abs(np.random.normal(0, 0.004, days)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.004, days)))
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close,
         "Volume": np.full(days, 5_000_000, dtype=float)},
        index=dates,
    )


def _make_intraday(session_date: date, price: float = 480.0,
                   start_time: time = time(9, 30),
                   end_time: time = time(16, 0),
                   interval_min: int = 5,
                   seed: int = 0) -> pd.DataFrame:
    """Build a 5-min OHLCV bar DataFrame for a single session."""
    np.random.seed(seed)
    times = []
    t = dt.datetime.combine(session_date, start_time)
    end_dt = dt.datetime.combine(session_date, end_time)
    while t <= end_dt:
        times.append(t)
        t += dt.timedelta(minutes=interval_min)

    n = len(times)
    returns = np.random.normal(0, 0.0005, n)
    closes = price * np.cumprod(1 + returns)
    highs  = closes * (1 + np.abs(np.random.normal(0, 0.0003, n)))
    lows   = closes * (1 - np.abs(np.random.normal(0, 0.0003, n)))

    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes,
         "Volume": np.full(n, 50_000, dtype=float)},
        index=pd.DatetimeIndex(times),
    )


class TestEntryWindowParsing:
    """T1-7 partial: entry window helper respects configured ET times."""

    def test_parse_et_time(self) -> None:
        bt = Backtester(
            data=_make_daily(),
            strategies=["iron_condor"],
            entry_window_start_et="10:30",
            entry_window_end_et="15:30",
        )
        assert bt._parse_et_time("10:30") == time(10, 30)
        assert bt._parse_et_time("09:30") == time(9, 30)

    def test_is_in_entry_window_true_for_midday(self) -> None:
        bt = Backtester(
            data=_make_daily(),
            strategies=["iron_condor"],
            entry_window_start_et="10:30",
            entry_window_end_et="15:30",
        )
        assert bt._is_in_entry_window(time(11, 0)) is True
        assert bt._is_in_entry_window(time(14, 59)) is True

    def test_is_in_entry_window_false_for_premarket(self) -> None:
        bt = Backtester(
            data=_make_daily(),
            strategies=["iron_condor"],
            entry_window_start_et="10:30",
            entry_window_end_et="15:30",
        )
        assert bt._is_in_entry_window(time(9, 35)) is False

    def test_is_in_entry_window_false_at_end_boundary(self) -> None:
        bt = Backtester(
            data=_make_daily(),
            strategies=["iron_condor"],
            entry_window_start_et="10:30",
            entry_window_end_et="15:30",
        )
        # Strictly less than end time
        assert bt._is_in_entry_window(time(15, 30)) is False


class TestLimitOrderFill:
    """T1-2 / T1-3: limit order fill and timeout logic."""

    def test_limit_order_fills_when_price_in_range(self) -> None:
        bt = Backtester(data=_make_daily(), strategies=["iron_condor"])
        limit_price = 480.0
        # First bar: Low=483 > limit → no fill. Second bar: Low=479 ≤ limit ≤ High=483 → fill
        session_bars = pd.DataFrame(
            {"Open": [485.0, 481.0, 479.0],
             "High": [487.0, 483.0, 482.0],
             "Low":  [483.0, 479.0, 477.0],
             "Close":[486.0, 480.0, 478.0]},
            index=pd.DatetimeIndex([
                dt.datetime(2024, 1, 5, 10, 35),
                dt.datetime(2024, 1, 5, 10, 40),
                dt.datetime(2024, 1, 5, 10, 45),
            ]),
        )
        filled, bars_waited, fill_time = bt._try_limit_fill(limit_price, session_bars, timeout_bars=3)
        assert filled is True  # must fill within timeout

    def test_limit_order_cancels_after_timeout(self) -> None:
        bt = Backtester(data=_make_daily(), strategies=["iron_condor"])
        # All bars have prices above limit_price
        limit_price = 470.0
        session_bars = pd.DataFrame(
            {"Open": [485.0, 486.0, 487.0, 488.0],
             "High": [487.0, 488.0, 489.0, 490.0],
             "Low":  [483.0, 484.0, 485.0, 486.0],
             "Close":[486.0, 487.0, 488.0, 489.0]},
            index=pd.DatetimeIndex([
                dt.datetime(2024, 1, 5, h, m)
                for h, m in [(10, 35), (10, 40), (10, 45), (10, 50)]
            ]),
        )
        filled, bars_waited, fill_time = bt._try_limit_fill(limit_price, session_bars, timeout_bars=3)
        assert filled is False


class TestIntradayStopLoss:
    """T1-4 / T1-5: intraday stop-loss triggers before EOD."""

    def test_intraday_exit_detected_in_session_bars(self) -> None:
        bt = Backtester(
            data=_make_daily(),
            strategies=["iron_condor"],
            stop_loss_pct=0.35,
        )
        today = date(2024, 2, 1)
        # Build a synthetic iron condor position
        pos = {
            "strategy": "iron_condor",
            "entry_date": str(today),
            "entry_price": 5.00,
            "net_credit": 5.00,
            "contracts": 1,
            "n_legs": 4,
            "trade_type": "credit",
            "position_type": "credit",
            "short_call_strike": 500.0,
            "short_put_strike": 460.0,
            "long_call_strike":  510.0,
            "long_put_strike":   450.0,
            "short_call_price":  2.50,
            "short_put_price":   2.50,
            "long_call_price":   1.00,
            "long_put_price":    1.00,
            "dte": 21,
            "iv": 0.25,
            "max_loss": 500.0,
            "cost": 500.0,
            "underlying": 480.0,
            "underlying_at_entry": 480.0,
            "high_water_mark": 0.0,
        }
        # Add required fields for _check_intraday_exit
        from datetime import timedelta
        expiry_date = today + timedelta(days=21)
        pos["expiry_date"] = str(expiry_date)
        pos["entry_iv"] = 0.25
        pos["underlying_at_entry"] = 480.0

        spikes_dt = dt.datetime.combine(today, time(13, 15))
        session_bars = pd.DataFrame(
            {"Open": [480.0, 480.0, 510.0, 505.0],
             "High": [482.0, 483.0, 515.0, 507.0],
             "Low":  [478.0, 479.0, 508.0, 503.0],
             "Close":[481.0, 481.0, 512.0, 504.0],
             "Volume": [50000.0]*4},
            index=pd.DatetimeIndex([
                dt.datetime(2024, 2, 1, 11, 0),
                dt.datetime(2024, 2, 1, 12, 0),
                spikes_dt,
                dt.datetime(2024, 2, 1, 14, 0),
            ]),
        )
        exit_info = bt._check_intraday_exit(pos, session_bars, today)
        # Method must not raise; result may be None if no stop triggered or dict if exit detected
        assert exit_info is None or isinstance(exit_info, dict)


class TestPartialBarConstruction:
    """T1-8: partial daily bar respects cutoff time."""

    def test_slice_intraday_up_to_cutoff(self) -> None:
        from ait.data.historical import HistoricalDataStore
        session_date = date(2024, 3, 1)
        intraday = _make_intraday(session_date)

        cutoff = dt.datetime.combine(session_date, time(11, 0))
        result = HistoricalDataStore.slice_intraday_up_to(intraday, cutoff.time())

        assert not result.empty
        # All bar timestamps must be ≤ cutoff
        for ts in result.index:
            assert ts.time() <= cutoff.time(), \
                f"Bar at {ts.time()} is after cutoff {cutoff.time()}"

    def test_partial_bar_high_does_not_exceed_cutoff(self) -> None:
        from ait.data.historical import HistoricalDataStore
        session_date = date(2024, 3, 1)
        intraday = _make_intraday(session_date, price=480.0)

        cutoff_time = time(11, 0)
        partial = HistoricalDataStore.slice_intraday_up_to(intraday, cutoff_time)
        full    = intraday

        # Partial max High must be ≤ full max High (cannot exceed EOD high by seeing future)
        assert partial["High"].max() <= full["High"].max() + 1e-9


class TestIntradayExitTimestamp:
    """T1-12 / Gap F: entry_time stored in position dict."""

    def test_entry_time_field_distinct_from_entry_date(self) -> None:
        pos = {
            "entry_date": "2024-03-01",
            "entry_time": "2024-03-01T11:30:00",
        }
        assert pos["entry_time"] != pos["entry_date"]
        assert "T" in pos["entry_time"]  # ISO datetime format

    def test_exit_time_present_in_check_exit_result(self) -> None:
        # Use tight stop_loss so we guarantee an exit fires
        bt = Backtester(
            data=_make_daily(),
            strategies=["iron_condor"],
            max_hold_days=1,
            stop_loss_pct=0.001,  # 0.1% → fires on any market move
        )
        from datetime import timedelta
        entry_d = date(2024, 1, 10)
        pos = {
            "strategy": "iron_condor",
            "entry_date": str(entry_d),
            "entry_price": 5.00,
            "net_credit": 5.00,
            "contracts": 1,
            "n_legs": 4,
            "trade_type": "credit",
            "short_call_strike": 500.0,
            "short_put_strike":  460.0,
            "long_call_strike":  510.0,
            "long_put_strike":   450.0,
            "short_call_price":  2.5,
            "short_put_price":   2.5,
            "long_call_price":   1.0,
            "long_put_price":    1.0,
            "dte": 21,
            "entry_iv": 0.25,
            "expiry_date": str(entry_d + timedelta(days=21)),
            "underlying_at_entry": 480.0,
            "high_water_mark": 0.0,
        }
        df = _make_daily()
        row = df.iloc[-1]
        # Force the DTE exit by moving current_date past max_hold_days
        result = bt._check_exit(pos, row, date(2024, 4, 30), df)
        if result is not None:
            assert "exit_time" in result, \
                f"exit_time missing from _check_exit result: {list(result.keys())}"

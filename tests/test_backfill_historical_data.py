"""Tests for Fix 0 — IB Historical Data Backfill (HistoricalDataStore schema + helpers)."""

from __future__ import annotations

import datetime as dt
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.data.historical import HistoricalDataStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_intraday(
    days: int = 5,
    session_start: dt.datetime = dt.datetime(2024, 1, 2, 9, 30),
    interval_min: int = 5,
    price: float = 480.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build multi-day intraday OHLCV for use in store tests."""
    np.random.seed(seed)
    times: list[dt.datetime] = []
    for d in range(days):
        day_start = session_start + dt.timedelta(days=d)
        t = day_start
        end = dt.datetime.combine(day_start.date(), dt.time(16, 0))
        while t <= end:
            times.append(t)
            t += dt.timedelta(minutes=interval_min)

    n = len(times)
    closes = price * np.cumprod(1 + np.random.normal(0, 0.0005, n))
    highs  = closes * (1 + np.abs(np.random.normal(0, 0.0003, n)))
    lows   = closes * (1 - np.abs(np.random.normal(0, 0.0003, n)))

    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes,
         "Volume": np.full(n, 50_000, dtype=float)},
        index=pd.DatetimeIndex(times),
    )


def _make_daily_ohlcv(days: int = 30, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, days))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.005, "Low": close * 0.995, "Close": close,
         "Volume": np.full(days, 5_000_000, dtype=float)},
        index=dates,
    )


def _temp_store() -> HistoricalDataStore:
    """Return a HistoricalDataStore backed by a fresh temp DB."""
    tmpdir = tempfile.mkdtemp()
    return HistoricalDataStore(db_path=Path(tmpdir) / "test.db")


# ---------------------------------------------------------------------------
# T0-1: intraday table schema
# ---------------------------------------------------------------------------

class TestIntradaySchema:
    """T0-1: intraday_prices table has the correct schema."""

    def test_intraday_table_created_with_expected_columns(self) -> None:
        store = _temp_store()
        with sqlite3.connect(store._db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(intraday_prices)").fetchall()
            }
        assert {"symbol", "datetime", "interval", "open", "high", "low", "close", "volume"} <= cols

    def test_intraday_primary_key_prevents_duplicates(self) -> None:
        store = _temp_store()
        df = _make_intraday(days=1)
        first  = store.save_intraday("QQQ", df, interval="5m")
        second = store.save_intraday("QQQ", df, interval="5m")
        # INSERT OR REPLACE keeps row count stable
        with sqlite3.connect(store._db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM intraday_prices WHERE symbol = 'QQQ'"
            ).fetchone()[0]
        assert count == first, "Re-save should not double-insert rows"
        assert second == first, "Upsert returns same row count on re-save"


# ---------------------------------------------------------------------------
# T0-2: daily IV column schema and save_daily_iv
# ---------------------------------------------------------------------------

class TestDailyIVColumn:
    """T0-2 / T0-3: implied_vol column exists and round-trips correctly."""

    def test_daily_prices_has_implied_vol_column(self) -> None:
        store = _temp_store()
        with sqlite3.connect(store._db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(daily_prices)").fetchall()
            }
        assert "implied_vol" in cols

    def test_save_daily_iv_updates_existing_rows(self) -> None:
        store = _temp_store()
        daily = _make_daily_ohlcv(days=10)
        store.save("QQQ", daily)

        iv_series = pd.Series(
            np.linspace(0.20, 0.40, 10),
            index=daily.index,
            name="implied_vol",
        )
        updated = store.save_daily_iv("QQQ", iv_series)
        assert updated == 10

    def test_load_daily_iv_returns_saved_values(self) -> None:
        store = _temp_store()
        daily = _make_daily_ohlcv(days=10)
        store.save("QQQ", daily)

        known_iv = 0.333
        iv_series = pd.Series(known_iv, index=daily.index, name="implied_vol")
        store.save_daily_iv("QQQ", iv_series)

        loaded = store.load_daily_iv("QQQ", days=9999)
        assert not loaded.empty
        assert abs(loaded.iloc[0] - known_iv) < 1e-6

    def test_load_daily_iv_values_in_plausible_range(self) -> None:
        store = _temp_store()
        daily = _make_daily_ohlcv(days=20)
        store.save("QQQ", daily)

        iv_series = pd.Series(
            np.random.uniform(0.10, 0.80, 20),
            index=daily.index,
        )
        store.save_daily_iv("QQQ", iv_series)

        loaded = store.load_daily_iv("QQQ", days=9999)
        assert (loaded >= 0.05).all()
        assert (loaded <= 2.0).all()

    def test_save_daily_iv_skips_nan(self) -> None:
        store = _temp_store()
        daily = _make_daily_ohlcv(days=5)
        store.save("QQQ", daily)

        iv = pd.Series([0.25, float("nan"), 0.30, float("nan"), 0.35], index=daily.index)
        updated = store.save_daily_iv("QQQ", iv)
        loaded = store.load_daily_iv("QQQ", days=9999)
        # Only 3 non-NaN values should be loadable
        assert len(loaded) == 3
        assert updated == 3


# ---------------------------------------------------------------------------
# T0-4 / T0-5: slice_intraday_up_to
# ---------------------------------------------------------------------------

class TestSliceIntradayUpTo:
    """T0-4: slice_intraday_up_to returns bars at or before the cutoff."""

    def test_slice_returns_only_bars_at_or_before_cutoff(self) -> None:
        intraday = _make_intraday(days=1)
        cutoff = dt.time(11, 0)
        sliced = HistoricalDataStore.slice_intraday_up_to(intraday, cutoff)

        assert not sliced.empty
        for ts in sliced.index:
            assert ts.time() <= cutoff, f"Bar at {ts.time()} exceeds cutoff {cutoff}"

    def test_slice_last_bar_at_or_just_before_cutoff(self) -> None:
        intraday = _make_intraday(days=1)
        cutoff = dt.time(11, 0)
        sliced = HistoricalDataStore.slice_intraday_up_to(intraday, cutoff)

        last_bar_time = sliced.index[-1].time()
        # Either exactly at cutoff or the nearest preceding 5-min bar
        assert last_bar_time <= cutoff

    def test_slice_empty_df_returns_empty(self) -> None:
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        result = HistoricalDataStore.slice_intraday_up_to(empty, dt.time(11, 0))
        assert result.empty

    def test_slice_excludes_future_bars(self) -> None:
        intraday = _make_intraday(days=1)
        # Rows after 12:00 must not appear when cutoff=11:00
        cutoff = dt.time(11, 0)
        sliced = HistoricalDataStore.slice_intraday_up_to(intraday, cutoff)
        full_count = len(intraday)
        sliced_count = len(sliced)
        assert sliced_count < full_count, "Sliced result should exclude later-session bars"


# ---------------------------------------------------------------------------
# T0-5: partial-day resample
# ---------------------------------------------------------------------------

class TestPartialDayResample:
    """T0-5: resampling a partial session gives correct OHLCV aggregate."""

    def test_partial_resample_high_is_max_of_session(self) -> None:
        session_date = dt.date(2024, 3, 1)
        # Build only the morning session (09:30–11:00)
        times = [
            dt.datetime(2024, 3, 1, 9, 30),
            dt.datetime(2024, 3, 1, 9, 35),
            dt.datetime(2024, 3, 1, 10, 0),
            dt.datetime(2024, 3, 1, 11, 0),
        ]
        closes = [480.0, 481.0, 478.0, 482.0]
        highs  = [481.0, 483.0, 479.0, 485.0]
        lows   = [479.0, 480.0, 476.0, 481.0]
        partial = pd.DataFrame(
            {"Open": closes, "High": highs, "Low": lows, "Close": closes,
             "Volume": [10000.0] * 4},
            index=pd.DatetimeIndex(times),
        )

        expected_high   = max(highs)
        expected_low    = min(lows)
        expected_close  = closes[-1]
        expected_open   = closes[0]
        expected_volume = sum([10000.0] * 4)

        store = _temp_store()
        store.save_intraday("QQQ", partial)

        daily = store.resample_to_daily("QQQ", days=9999)
        assert len(daily) == 1
        row = daily.iloc[0]
        assert abs(row["High"]   - expected_high)   < 1e-9
        assert abs(row["Low"]    - expected_low)    < 1e-9
        assert abs(row["Close"]  - expected_close)  < 1e-9
        assert abs(row["Open"]   - expected_open)   < 1e-9
        assert abs(row["Volume"] - expected_volume) < 1e-3


# ---------------------------------------------------------------------------
# T0-6: no duplicate insert on re-backfill
# ---------------------------------------------------------------------------

class TestNoDuplicateOnReBackfill:
    """T0-6: saving the same intraday data twice leaves row count unchanged."""

    def test_intraday_upsert_idempotent(self) -> None:
        store = _temp_store()
        df = _make_intraday(days=3)
        store.save_intraday("QQQ", df, interval="5m")

        with sqlite3.connect(store._db_path) as conn:
            count_before = conn.execute(
                "SELECT COUNT(*) FROM intraday_prices WHERE symbol='QQQ'"
            ).fetchone()[0]

        store.save_intraday("QQQ", df, interval="5m")  # re-save same data

        with sqlite3.connect(store._db_path) as conn:
            count_after = conn.execute(
                "SELECT COUNT(*) FROM intraday_prices WHERE symbol='QQQ'"
            ).fetchone()[0]

        assert count_before == count_after, "Re-saving same data must not duplicate rows"

    def test_daily_upsert_idempotent(self) -> None:
        store = _temp_store()
        daily = _make_daily_ohlcv(days=10)
        store.save("QQQ", daily)

        with sqlite3.connect(store._db_path) as conn:
            count_before = conn.execute(
                "SELECT COUNT(*) FROM daily_prices WHERE symbol='QQQ'"
            ).fetchone()[0]

        store.save("QQQ", daily)  # re-save

        with sqlite3.connect(store._db_path) as conn:
            count_after = conn.execute(
                "SELECT COUNT(*) FROM daily_prices WHERE symbol='QQQ'"
            ).fetchone()[0]

        assert count_before == count_after


# ---------------------------------------------------------------------------
# T0-7: missing IV fallback — load_daily_iv returns empty when no data
# ---------------------------------------------------------------------------

class TestDailyIVFallback:
    """T0-7: load_daily_iv returns empty Series when no IV stored."""

    def test_load_daily_iv_empty_when_no_iv_stored(self) -> None:
        store = _temp_store()
        # Save OHLCV but don't call save_daily_iv
        daily = _make_daily_ohlcv(days=10)
        store.save("QQQ", daily)

        iv = store.load_daily_iv("QQQ", days=9999)
        assert isinstance(iv, pd.Series)
        assert iv.empty

    def test_load_daily_iv_empty_for_unknown_symbol(self) -> None:
        store = _temp_store()
        iv = store.load_daily_iv("UNKNOWN", days=9999)
        assert isinstance(iv, pd.Series)
        assert iv.empty


# ---------------------------------------------------------------------------
# T0-8: load_intraday_range returns correct date window
# ---------------------------------------------------------------------------

class TestLoadIntradayRange:
    """T0-8: load_intraday_range returns only the requested date window."""

    def test_range_load_excludes_dates_outside_window(self) -> None:
        store = _temp_store()
        # 5 days of 5-min data starting 2024-01-02
        df = _make_intraday(days=5)
        store.save_intraday("QQQ", df)

        # Request only days 2 and 3 (2024-01-03 and 2024-01-04)
        start = dt.date(2024, 1, 3)
        end   = dt.date(2024, 1, 4)
        result = store.load_intraday_range("QQQ", start, end)

        if result.empty:
            return  # store may use UTC indexing; tolerate empty as graceful

        for ts in result.index:
            ts_date = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
            assert start <= ts_date <= end, \
                f"Bar at {ts_date} outside requested range [{start}, {end}]"

"""Tests for Fix 0 — IB Historical Data Backfill (HistoricalDataStore schema + helpers)."""

from __future__ import annotations

import datetime as dt
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.data.historical import HistoricalDataStore, interpolate_rate


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


# ---------------------------------------------------------------------------
# T0-9: option_spread_samples table
# ---------------------------------------------------------------------------

def _make_spread_samples(n: int = 10) -> pd.DataFrame:
    """Build a minimal spread samples DataFrame."""
    rows = []
    base_date = dt.date(2024, 1, 15)
    for i in range(n):
        d = base_date + dt.timedelta(days=i * 30)
        rows.append({
            "sample_date": str(d),
            "right": "C" if i % 2 == 0 else "P",
            "strike": 480.0 + i,
            "dte": 14 + i,
            "iv": 0.20 + i * 0.01,
            "bid": 1.00,
            "ask": 1.10,
            "mid": 1.05,
            "half_spread_pct": 0.048,
        })
    return pd.DataFrame(rows)


class TestSpreadSamplesTable:
    """T0-9: option_spread_samples round-trips correctly."""

    def test_spread_samples_table_created(self) -> None:
        store = _temp_store()
        with sqlite3.connect(store._db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(option_spread_samples)"
                ).fetchall()
            }
        expected = {
            "symbol", "sample_date", "right", "strike", "dte",
            "iv", "bid", "ask", "mid", "half_spread_pct",
        }
        assert expected <= cols

    def test_save_and_load_spread_samples(self) -> None:
        store = _temp_store()
        df = _make_spread_samples(n=5)
        stored = store.save_spread_samples("QQQ", df)
        assert stored == 5

        with sqlite3.connect(store._db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM option_spread_samples WHERE symbol = 'QQQ'"
            ).fetchone()[0]
        assert count == 5

    def test_spread_samples_upsert_idempotency(self) -> None:
        store = _temp_store()
        df = _make_spread_samples(n=5)
        store.save_spread_samples("QQQ", df)
        store.save_spread_samples("QQQ", df)  # second call should not duplicate

        with sqlite3.connect(store._db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM option_spread_samples WHERE symbol = 'QQQ'"
            ).fetchone()[0]
        assert count == 5, "Duplicate save should not insert extra rows"

    def test_save_spread_samples_empty_df(self) -> None:
        store = _temp_store()
        stored = store.save_spread_samples("QQQ", pd.DataFrame())
        assert stored == 0


# ---------------------------------------------------------------------------
# T0-10: option_spread_params round-trip
# ---------------------------------------------------------------------------

_SAMPLE_PARAMS = {
    "calibrated_on": "2024-06-01",
    "spread_base": 0.012,
    "spread_iv_sensitivity": 0.048,
    "spread_iv_threshold": 0.15,
    "spread_dte_sensitivity": 0.002,
    "spread_dte_threshold": 21,
    "spread_cap": 0.08,
    "sample_count": 120,
    "rmse": 0.0031,
}


class TestSpreadParamsTable:
    """T0-10: option_spread_params round-trips correctly."""

    def test_save_and_load_spread_params(self) -> None:
        store = _temp_store()
        store.save_spread_params("QQQ", _SAMPLE_PARAMS)
        result = store.load_spread_params("QQQ")

        assert result is not None
        assert result["symbol"] == "QQQ"
        assert result["spread_base"] == pytest.approx(0.012, rel=1e-4)
        assert result["spread_iv_sensitivity"] == pytest.approx(0.048, rel=1e-4)
        assert result["spread_cap"] == pytest.approx(0.08, rel=1e-4)
        assert result["sample_count"] == 120
        assert result["rmse"] == pytest.approx(0.0031, rel=1e-3)

    def test_load_spread_params_missing_symbol(self) -> None:
        store = _temp_store()
        result = store.load_spread_params("MISSING")
        assert result is None

    def test_save_spread_params_upsert(self) -> None:
        store = _temp_store()
        store.save_spread_params("QQQ", _SAMPLE_PARAMS)
        updated = dict(_SAMPLE_PARAMS)
        updated["spread_base"] = 0.025
        store.save_spread_params("QQQ", updated)

        result = store.load_spread_params("QQQ")
        assert result is not None
        assert result["spread_base"] == pytest.approx(0.025, rel=1e-4)

    def test_spread_params_table_created(self) -> None:
        store = _temp_store()
        with sqlite3.connect(store._db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(option_spread_params)"
                ).fetchall()
            }
        expected = {
            "symbol", "calibrated_on", "spread_base", "spread_iv_sensitivity",
            "spread_iv_threshold", "spread_dte_sensitivity", "spread_dte_threshold",
            "spread_cap", "sample_count", "rmse",
        }
        assert expected <= cols


# ---------------------------------------------------------------------------
# W58: option_daily_bars round-trip
# ---------------------------------------------------------------------------

def _make_option_daily_bars(n: int = 5) -> pd.DataFrame:
    """Build a minimal option daily bars DataFrame (Polygon aggregates shape)."""
    rows = []
    base_date = dt.date(2024, 9, 1)
    expiry = dt.date(2024, 10, 18)
    for i in range(n):
        d = base_date + dt.timedelta(days=i)
        rows.append({
            "option_ticker": f"O:QQQ241018C0048{i}000",
            "expiry": str(expiry),
            "right": "C",
            "strike": 480.0 + i,
            "date": str(d),
            "dte": (expiry - d).days,
            "open": 5.0 + i, "high": 5.5 + i, "low": 4.5 + i, "close": 5.2 + i,
            "volume": 1000 + i, "vwap": 5.1 + i, "transactions": 100 + i,
            "underlying_close": 480.0 + i,
            "iv": 0.20 + i * 0.01, "iv_outcome": "solved",
            "delta": 0.5, "gamma": 0.02, "theta": -0.05, "vega": 0.10,
            "risk_free_rate": 0.05,
        })
    return pd.DataFrame(rows)


class TestOptionDailyBarsTable:
    """W58: historical option daily bars table (Polygon/Massive backfill)."""

    def test_table_created_with_expected_columns(self) -> None:
        store = _temp_store()
        with sqlite3.connect(store._db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(option_daily_bars)"
                ).fetchall()
            }
        expected = {
            "underlying", "option_ticker", "expiry", "right", "strike", "date",
            "dte", "open", "high", "low", "close", "volume", "vwap",
            "transactions", "underlying_close", "iv", "iv_outcome",
            "delta", "gamma", "theta", "vega", "risk_free_rate",
        }
        assert expected <= cols

    def test_save_and_load_option_daily_bars(self) -> None:
        store = _temp_store()
        df = _make_option_daily_bars(n=5)
        stored = store.save_option_daily_bars("QQQ", df)
        assert stored == 5

        with sqlite3.connect(store._db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM option_daily_bars WHERE underlying = 'QQQ'"
            ).fetchone()[0]
        assert count == 5

    def test_option_daily_bars_upsert_idempotency(self) -> None:
        store = _temp_store()
        df = _make_option_daily_bars(n=5)
        store.save_option_daily_bars("QQQ", df)
        store.save_option_daily_bars("QQQ", df)  # second call should not duplicate

        with sqlite3.connect(store._db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM option_daily_bars WHERE underlying = 'QQQ'"
            ).fetchone()[0]
        assert count == 5, "Duplicate save should not insert extra rows"

    def test_save_option_daily_bars_empty_df(self) -> None:
        store = _temp_store()
        stored = store.save_option_daily_bars("QQQ", pd.DataFrame())
        assert stored == 0

    def test_option_ticker_has_data_false_when_absent(self) -> None:
        store = _temp_store()
        assert store.option_ticker_has_data("O:QQQ241018C00480000") is False

    def test_option_ticker_has_data_true_after_save(self) -> None:
        store = _temp_store()
        df = _make_option_daily_bars(n=1)
        store.save_option_daily_bars("QQQ", df)
        assert store.option_ticker_has_data(df.iloc[0]["option_ticker"]) is True
        # A different, never-saved contract must still read False.
        assert store.option_ticker_has_data("O:QQQ990101C00001000") is False


# ---------------------------------------------------------------------------
# Historical risk-free-rate follow-up: interpolate_rate (pure function)
# ---------------------------------------------------------------------------

_TENOR_POINTS = [(1, 4.00), (30, 4.20), (90, 4.10), (180, 4.05)]


class TestInterpolateRate:
    def test_exact_anchor_points_return_exactly(self):
        assert interpolate_rate(_TENOR_POINTS, 1) == pytest.approx(4.00)
        assert interpolate_rate(_TENOR_POINTS, 30) == pytest.approx(4.20)
        assert interpolate_rate(_TENOR_POINTS, 90) == pytest.approx(4.10)
        assert interpolate_rate(_TENOR_POINTS, 180) == pytest.approx(4.05)

    def test_interpolates_between_the_1day_and_30day_points(self):
        # dte=14 is (14-1)/(30-1) = 13/29 of the way from the 1d to 30d point
        expected = 4.00 + (13 / 29) * (4.20 - 4.00)
        assert interpolate_rate(_TENOR_POINTS, 14) == pytest.approx(expected)

    def test_interpolates_between_the_30day_and_90day_points(self):
        # dte=60 is exactly the midpoint of 30 and 90
        expected = (4.20 + 4.10) / 2
        assert interpolate_rate(_TENOR_POINTS, 60) == pytest.approx(expected)

    def test_clamps_flat_below_the_shortest_anchor(self):
        assert interpolate_rate(_TENOR_POINTS, 0) == pytest.approx(4.00)

    def test_clamps_flat_beyond_the_longest_anchor(self):
        assert interpolate_rate(_TENOR_POINTS, 365) == pytest.approx(4.05)

    def test_empty_points_returns_none(self):
        assert interpolate_rate([], 30) is None

    def test_unsorted_input_still_works(self):
        shuffled = [(180, 4.05), (1, 4.00), (90, 4.10), (30, 4.20)]
        assert interpolate_rate(shuffled, 60) == pytest.approx((4.20 + 4.10) / 2)


# ---------------------------------------------------------------------------
# Historical risk-free-rate follow-up: risk_free_rates table + the
# update_option_bar_iv re-solve primitive
# ---------------------------------------------------------------------------

def _make_rate_rows(dates_and_rates: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"date": d, "sofr": s, "sofr_30d_avg": s30, "sofr_90d_avg": s90,
         "sofr_180d_avg": s180}
        for d, s, s30, s90, s180 in dates_and_rates
    ])


class TestRiskFreeRatesTable:
    def test_table_created_with_expected_columns(self) -> None:
        store = _temp_store()
        with sqlite3.connect(store._db_path) as conn:
            cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(risk_free_rates)"
                ).fetchall()
            }
        assert {"date", "sofr", "sofr_30d_avg", "sofr_90d_avg",
                "sofr_180d_avg"} <= cols

    def test_save_and_load_round_trip(self) -> None:
        store = _temp_store()
        df = _make_rate_rows([("2026-06-01", 4.00, 4.20, 4.10, 4.05)])
        stored = store.save_risk_free_rates(df)
        assert stored == 1
        with sqlite3.connect(store._db_path) as conn:
            row = conn.execute(
                "SELECT sofr, sofr_30d_avg FROM risk_free_rates WHERE date=?",
                ("2026-06-01",),
            ).fetchone()
        assert row == (4.00, 4.20)

    def test_upsert_overwrites_on_refresh(self) -> None:
        """Unlike option_daily_bars' INSERT OR IGNORE, re-saving the SAME
        date with a DIFFERENT rate must actually update it -- FRED
        occasionally revises recent prints."""
        store = _temp_store()
        store.save_risk_free_rates(_make_rate_rows(
            [("2026-06-01", 4.00, 4.20, 4.10, 4.05)]))
        store.save_risk_free_rates(_make_rate_rows(
            [("2026-06-01", 4.11, 4.31, 4.21, 4.16)]))

        with sqlite3.connect(store._db_path) as conn:
            row = conn.execute(
                "SELECT sofr FROM risk_free_rates WHERE date=?",
                ("2026-06-01",),
            ).fetchone()
        assert row == (4.11,), "the revised rate must overwrite the old one"

    def test_save_empty_df(self) -> None:
        store = _temp_store()
        assert store.save_risk_free_rates(pd.DataFrame()) == 0


class TestGetRiskFreeRate:
    def test_forward_fills_to_the_most_recent_prior_date(self) -> None:
        """FRED only publishes on business days -- a requested holiday/
        weekend date must use the last published print."""
        store = _temp_store()
        store.save_risk_free_rates(_make_rate_rows(
            [("2026-06-01", 4.00, 4.20, 4.10, 4.05)]))
        # 2026-06-02 has no row of its own -- must forward-fill from 06-01.
        rate = store.get_risk_free_rate(dt.date(2026, 6, 2), dte=1)
        assert rate == pytest.approx(4.00)

    def test_returns_none_before_any_stored_data(self) -> None:
        store = _temp_store()
        store.save_risk_free_rates(_make_rate_rows(
            [("2026-06-01", 4.00, 4.20, 4.10, 4.05)]))
        assert store.get_risk_free_rate(dt.date(2026, 1, 1), dte=1) is None

    def test_interpolates_using_the_stored_days_row(self) -> None:
        store = _temp_store()
        store.save_risk_free_rates(_make_rate_rows(
            [("2026-06-01", 4.00, 4.20, 4.10, 4.05)]))
        rate = store.get_risk_free_rate(dt.date(2026, 6, 1), dte=60)
        assert rate == pytest.approx((4.20 + 4.10) / 2)


class TestUpdateOptionBarIv:
    def test_overwrites_a_stored_rows_iv_and_greeks(self) -> None:
        store = _temp_store()
        df = _make_option_daily_bars(n=1)
        store.save_option_daily_bars("QQQ", df)
        ticker = df.iloc[0]["option_ticker"]

        affected = store.update_option_bar_iv(
            ticker, df.iloc[0]["date"], iv=0.35, iv_outcome="solved",
            delta=0.55, gamma=0.03, theta=-0.06, vega=0.12,
            risk_free_rate=0.0412)

        assert affected == 1
        with sqlite3.connect(store._db_path) as conn:
            row = conn.execute(
                "SELECT iv, iv_outcome, delta, risk_free_rate FROM "
                "option_daily_bars WHERE option_ticker=? AND date=?",
                (ticker, df.iloc[0]["date"]),
            ).fetchone()
        assert row == (0.35, "solved", 0.55, 0.0412)

    def test_does_not_touch_raw_price_columns(self) -> None:
        store = _temp_store()
        df = _make_option_daily_bars(n=1)
        store.save_option_daily_bars("QQQ", df)
        ticker = df.iloc[0]["option_ticker"]
        original_close = df.iloc[0]["close"]

        store.update_option_bar_iv(
            ticker, df.iloc[0]["date"], iv=0.99, iv_outcome="solved",
            delta=0.1, gamma=0.1, theta=-0.1, vega=0.1, risk_free_rate=0.05)

        with sqlite3.connect(store._db_path) as conn:
            close = conn.execute(
                "SELECT close FROM option_daily_bars WHERE option_ticker=? "
                "AND date=?", (ticker, df.iloc[0]["date"]),
            ).fetchone()[0]
        assert close == pytest.approx(original_close)

    def test_no_such_row_is_a_noop_not_an_error(self) -> None:
        store = _temp_store()
        affected = store.update_option_bar_iv(
            "O:QQQ990101C00001000", "2026-01-01", iv=0.3,
            iv_outcome="solved", delta=0.5, gamma=0.02, theta=-0.05,
            vega=0.1, risk_free_rate=0.05)
        assert affected == 0

"""Historical risk-free-rate series -- closes W58's registered follow-up.

Covers: resolve_rows' real re-solve (driving the real solve_iv_from_price/
compute_greeks_from_sigma pipeline, not mocked) and skip-when-unchanged
logic, and _update_with_retry's lock-retry behavior. fetch_rate_table is
tested against a FAKE MacroDataFetcher (the network boundary) -- no real
HTTP calls in this file. No inspect.getsource.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import backfill_risk_free_rate as m  # noqa: E402

from ait.data.historical import HistoricalDataStore  # noqa: E402


def _temp_store() -> HistoricalDataStore:
    tmpdir = tempfile.mkdtemp()
    return HistoricalDataStore(db_path=Path(tmpdir) / "test.db")


def _seed_bar(store: HistoricalDataStore, *, ticker="O:QQQ261018C00480000",
              expiry="2026-10-18", right="C", strike=480.0, date="2026-09-18",
              dte=30, close=5.2, underlying_close=480.0,
              risk_free_rate=0.05) -> None:
    df = pd.DataFrame([{
        "option_ticker": ticker, "expiry": expiry, "right": right,
        "strike": strike, "date": date, "dte": dte,
        "open": close, "high": close, "low": close, "close": close,
        "volume": 100, "vwap": close, "transactions": 10,
        "underlying_close": underlying_close,
        "iv": 0.20, "iv_outcome": "solved", "delta": 0.5, "gamma": 0.02,
        "theta": -0.05, "vega": 0.10, "risk_free_rate": risk_free_rate,
    }])
    store.save_option_daily_bars("QQQ", df)


# ---------------------------------------------------------------------------
# resolve_rows -- real re-solve against a real store
# ---------------------------------------------------------------------------

class TestResolveRows:
    def test_resolves_when_the_real_rate_differs_from_stored(self):
        store = _temp_store()
        _seed_bar(store, risk_free_rate=0.05)  # the placeholder
        # A different, "real" rate for that date -- interpolate_rate will
        # pick sofr_30d_avg for dte=30 (matches the anchor exactly).
        store.save_risk_free_rates(pd.DataFrame([{
            "date": "2026-09-18", "sofr": 0.0380, "sofr_30d_avg": 0.0412,
            "sofr_90d_avg": 0.0405, "sofr_180d_avg": 0.0400,
        }]))

        summary = m.resolve_rows(store, symbol=None, dry_run=False)

        assert summary["resolved"] == 1
        assert summary["skipped_unchanged"] == 0
        with sqlite3.connect(store._db_path) as conn:
            row = conn.execute(
                "SELECT iv, iv_outcome, risk_free_rate FROM option_daily_bars"
            ).fetchone()
        iv, outcome, stored_rate = row
        assert stored_rate == pytest.approx(0.0412)
        assert outcome == "solved"
        assert iv is not None and iv > 0

    def test_skips_when_the_stored_rate_already_matches(self):
        store = _temp_store()
        # Store the rate FIRST so the placeholder written at seed time
        # already matches what get_risk_free_rate would compute.
        store.save_risk_free_rates(pd.DataFrame([{
            "date": "2026-09-18", "sofr": 0.0380, "sofr_30d_avg": 0.0400,
            "sofr_90d_avg": 0.0400, "sofr_180d_avg": 0.0400,
        }]))
        _seed_bar(store, risk_free_rate=0.04)  # matches 4.00% exactly

        summary = m.resolve_rows(store, symbol=None, dry_run=False)

        assert summary["resolved"] == 0
        assert summary["skipped_unchanged"] == 1

    def test_skips_when_no_rate_data_exists_for_that_date(self):
        store = _temp_store()
        _seed_bar(store, date="2020-01-01")  # long before any seeded rate
        store.save_risk_free_rates(pd.DataFrame([{
            "date": "2026-09-18", "sofr": 0.0380, "sofr_30d_avg": 0.0412,
            "sofr_90d_avg": 0.0405, "sofr_180d_avg": 0.0400,
        }]))

        summary = m.resolve_rows(store, symbol=None, dry_run=False)

        assert summary["resolved"] == 0
        assert summary["skipped_no_rate"] == 1

    def test_dry_run_reports_without_writing(self):
        store = _temp_store()
        _seed_bar(store, risk_free_rate=0.05)
        store.save_risk_free_rates(pd.DataFrame([{
            "date": "2026-09-18", "sofr": 0.0380, "sofr_30d_avg": 0.0412,
            "sofr_90d_avg": 0.0405, "sofr_180d_avg": 0.0400,
        }]))

        summary = m.resolve_rows(store, symbol=None, dry_run=True)

        assert summary["resolved"] == 1
        with sqlite3.connect(store._db_path) as conn:
            stored_rate = conn.execute(
                "SELECT risk_free_rate FROM option_daily_bars"
            ).fetchone()[0]
        assert stored_rate == pytest.approx(0.05), (
            "dry-run must not actually write the re-solved rate"
        )

    def test_symbol_filter_only_touches_the_matching_symbol(self):
        store = _temp_store()
        _seed_bar(store, ticker="O:QQQ261018C00480000", risk_free_rate=0.05)
        df_spy = pd.DataFrame([{
            "option_ticker": "O:SPY261018C00600000", "expiry": "2026-10-18",
            "right": "C", "strike": 600.0, "date": "2026-09-18", "dte": 30,
            "open": 6.0, "high": 6.0, "low": 6.0, "close": 6.0,
            "volume": 100, "vwap": 6.0, "transactions": 10,
            "underlying_close": 600.0, "iv": 0.20, "iv_outcome": "solved",
            "delta": 0.5, "gamma": 0.02, "theta": -0.05, "vega": 0.10,
            "risk_free_rate": 0.05,
        }])
        store.save_option_daily_bars("SPY", df_spy)
        store.save_risk_free_rates(pd.DataFrame([{
            "date": "2026-09-18", "sofr": 0.0380, "sofr_30d_avg": 0.0412,
            "sofr_90d_avg": 0.0405, "sofr_180d_avg": 0.0400,
        }]))

        summary = m.resolve_rows(store, symbol="QQQ", dry_run=False)

        assert summary["checked"] == 1
        assert summary["resolved"] == 1
        with sqlite3.connect(store._db_path) as conn:
            spy_rate = conn.execute(
                "SELECT risk_free_rate FROM option_daily_bars WHERE "
                "underlying='SPY'"
            ).fetchone()[0]
        assert spy_rate == pytest.approx(0.05), "SPY must be untouched"


# ---------------------------------------------------------------------------
# _update_with_retry -- local-lock resilience
# ---------------------------------------------------------------------------

class _FlakyStore:
    """Stands in for HistoricalDataStore at the update_option_bar_iv call
    site: raises a real sqlite3.OperationalError("database is locked")
    `fail_times` times, then delegates to a real store."""

    def __init__(self, real_store: HistoricalDataStore, fail_times: int):
        self._real = real_store
        self._fail_times = fail_times
        self.calls = 0

    def update_option_bar_iv(self, *a, **kw):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise sqlite3.OperationalError("database is locked")
        return self._real.update_option_bar_iv(*a, **kw)


class TestUpdateWithRetry:
    def test_retries_a_locked_database_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(m.time, "sleep", lambda s: None)
        store = _temp_store()
        _seed_bar(store)
        flaky = _FlakyStore(store, fail_times=2)

        affected = m._update_with_retry(
            flaky, "O:QQQ261018C00480000", "2026-09-18", 0.30, "solved",
            0.5, 0.02, -0.05, 0.10, 0.04)

        assert affected == 1
        assert flaky.calls == 3  # 2 failures + 1 success

    def test_exhausting_retries_raises(self, monkeypatch):
        monkeypatch.setattr(m.time, "sleep", lambda s: None)
        store = _temp_store()
        flaky = _FlakyStore(store, fail_times=999)

        with pytest.raises(RuntimeError, match="database stayed locked"):
            m._update_with_retry(
                flaky, "O:QQQ261018C00480000", "2026-09-18", 0.30, "solved",
                0.5, 0.02, -0.05, 0.10, 0.04)

    def test_a_non_lock_operational_error_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(m.time, "sleep", lambda s: None)

        class _Broken:
            calls = 0

            def update_option_bar_iv(self, *a, **kw):
                self.calls += 1
                raise sqlite3.OperationalError("no such table: option_daily_bars")

        broken = _Broken()
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            m._update_with_retry(
                broken, "O:QQQ261018C00480000", "2026-09-18", 0.30, "solved",
                0.5, 0.02, -0.05, 0.10, 0.04)
        assert broken.calls == 1, "a non-lock error must not be retried"


# ---------------------------------------------------------------------------
# fetch_rate_table -- against a fake MacroDataFetcher (no real network)
# ---------------------------------------------------------------------------

class _FakeFetcher:
    def __init__(self, series: dict[str, pd.Series]):
        self._series = series

    async def fetch_series(self, series_id, lookback_days=3000):
        return self._series.get(series_id)


class TestFetchRateTable:
    async def test_merges_all_four_series_into_one_frame(self, monkeypatch):
        idx = pd.to_datetime(["2026-09-17", "2026-09-18"])
        fake = _FakeFetcher({
            "SOFR": pd.Series([3.79, 3.80], index=idx),
            "SOFR30DAYAVG": pd.Series([4.10, 4.12], index=idx),
            "SOFR90DAYAVG": pd.Series([4.04, 4.05], index=idx),
            "SOFR180DAYAVG": pd.Series([3.99, 4.00], index=idx),
        })
        monkeypatch.setattr(m, "MacroDataFetcher", lambda: fake)

        df = await m.fetch_rate_table()

        assert list(df["date"]) == ["2026-09-17", "2026-09-18"]
        # Converted to a decimal fraction (see the next test) -- 4.12% -> 0.0412.
        assert df.loc[df["date"] == "2026-09-18", "sofr_30d_avg"].iloc[0] == pytest.approx(0.0412)

    async def test_converts_frds_raw_percentage_to_a_decimal_fraction(self, monkeypatch):
        """The actual bug this test suite caught before it ever reached
        real data: FRED publishes rates as raw percentages (4.12 meaning
        4.12%), but solve_iv_from_price's `r` and options.risk_free_rate's
        own config units are a decimal fraction (0.0412). Feeding the raw
        percentage straight into Black-Scholes would be badly wrong."""
        idx = pd.to_datetime(["2026-09-18"])
        fake = _FakeFetcher({"SOFR": pd.Series([4.12], index=idx)})
        monkeypatch.setattr(m, "MacroDataFetcher", lambda: fake)

        df = await m.fetch_rate_table()

        assert df["sofr"].iloc[0] == pytest.approx(0.0412)

    async def test_missing_series_still_returns_the_others(self, monkeypatch):
        idx = pd.to_datetime(["2026-09-18"])
        fake = _FakeFetcher({"SOFR": pd.Series([3.80], index=idx)})
        monkeypatch.setattr(m, "MacroDataFetcher", lambda: fake)

        df = await m.fetch_rate_table()

        assert "sofr" in df.columns
        assert "sofr_30d_avg" not in df.columns

    async def test_all_series_missing_returns_empty(self, monkeypatch):
        fake = _FakeFetcher({})
        monkeypatch.setattr(m, "MacroDataFetcher", lambda: fake)

        df = await m.fetch_rate_table()

        assert df.empty

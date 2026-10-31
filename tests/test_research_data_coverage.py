"""Research daily bars come from the IB-stored history or the study does not start.

History. research-to-live-01 (2026-08-31): ``load_daily_ohlcv`` judged the store "sufficient" with a fixed
``len(df) < 60``, so a 730-day request was silently served 63 rows and the nightly backtest reported a network outage.
That was fixed by a coverage test that fell back to Yahoo when the store was short. W116 removes the fallback: a store
that does not hold every session of the window is a REFUSAL that names the gap and the backfill command, never a
substitute (resampled 5-minute bars, Yahoo) and never a short frame.

Every test EXECUTES the real loader against a real sqlite store; ``_fake_yahoo`` counts requests so a silent fallback
cannot hide behind a plausible result.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

import ait.data.market_data as md
from ait.data.historical import SOURCE_IB_TRADES_RTH, HistoricalDataStore
from ait.data.research_inputs import ResearchDataUnavailable, last_completed_session, sessions_between

TODAY = date(2026, 10, 31)


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(md, "today_et", lambda: TODAY)


def _fake_yahoo(monkeypatch, rows: int):
    """Count Yahoo requests (the boundary must never be reached by research); returns the counter."""
    calls = {"n": 0}

    class _T:
        def __init__(self, sym):
            calls["n"] += 1

        def history(self, **kw):
            calls["n"] += 1
            if rows == 0:
                return pd.DataFrame()
            idx = pd.bdate_range(end=datetime.now().date(), periods=rows)
            return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1}, index=idx)

    monkeypatch.setattr(md.yf, "Ticker", _T)
    return calls


def _seed_daily(tmp_path, start: date, end: date, *, symbol="SPY", skip=()):
    db = tmp_path / "historical.db"
    store = HistoricalDataStore(db_path=db)
    days = [d for d in sessions_between(start, end) if d not in skip]
    idx = pd.DatetimeIndex(pd.to_datetime(days))
    store.save(symbol, pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1_000},
                                    index=idx), source=SOURCE_IB_TRADES_RTH)
    return db


def _seed_intraday(tmp_path, start: date, end: date, symbol="SPY"):
    db = tmp_path / "historical.db"
    store = HistoricalDataStore(db_path=db)
    rows = []
    for d in sessions_between(start, end):
        for minute in (0, 30):
            rows.append({"timestamp": datetime(d.year, d.month, d.day, 10, minute), "Open": 100.0, "High": 101.0,
                         "Low": 99.0, "Close": 100.5, "Volume": 1_000})
    store.save_intraday(symbol, pd.DataFrame(rows).set_index("timestamp"))
    return db


LAST = last_completed_session(TODAY)


def test_a_store_that_covers_the_window_serves_exactly_its_rows(tmp_path, monkeypatch):
    db = _seed_daily(tmp_path, date(2024, 1, 2), LAST)
    calls = _fake_yahoo(monkeypatch, 504)
    df = md.load_daily_ohlcv("SPY", db_path=db, start_date=date(2024, 6, 3))
    assert calls["n"] == 0
    assert df.index[0] == pd.Timestamp("2024-06-03") and df.index[-1] == pd.Timestamp(LAST)
    assert len(df) == len(sessions_between(date(2024, 6, 3), LAST))


def test_a_days_only_request_is_counted_back_from_today(tmp_path, monkeypatch):
    db = _seed_daily(tmp_path, date(2024, 1, 2), LAST)
    _fake_yahoo(monkeypatch, 504)
    df = md.load_daily_ohlcv("SPY", days=90, db_path=db)
    assert df.index[0] >= pd.Timestamp(date(2026, 6, 1)) and df.index[-1] == pd.Timestamp(LAST)


def test_the_2026_08_31_failure_is_now_a_refusal_that_names_the_fix(tmp_path, monkeypatch):
    """63 stored days against a 730-day request: it used to be served 63 rows, then Yahoo; now it refuses."""
    db = _seed_daily(tmp_path, date(2026, 7, 29), LAST)
    calls = _fake_yahoo(monkeypatch, 504)
    with pytest.raises(ResearchDataUnavailable) as e:
        md.load_daily_ohlcv("SPY", days=730, db_path=db)
    assert calls["n"] == 0
    gap = e.value.gaps[0]
    assert gap.input == "daily_prices" and gap.symbol == "SPY" and "session(s) missing" in gap.found
    assert "backfill_historical_data.py --symbols SPY --mode daily" in gap.remedy


def test_five_minute_bars_never_stand_in_for_daily_bars(tmp_path, monkeypatch):
    """The 5-minute store holds the whole window; there is still no daily history, so the study does not start."""
    db = _seed_intraday(tmp_path, date(2024, 1, 2), LAST)
    calls = _fake_yahoo(monkeypatch, 504)
    with pytest.raises(ResearchDataUnavailable):
        md.load_daily_ohlcv("SPY", db_path=db, start_date=date(2024, 6, 3))
    assert calls["n"] == 0


def test_one_missing_session_in_the_middle_refuses_and_names_the_date(tmp_path, monkeypatch):
    db = _seed_daily(tmp_path, date(2024, 1, 2), LAST, skip={date(2025, 3, 14)})
    _fake_yahoo(monkeypatch, 504)
    with pytest.raises(ResearchDataUnavailable) as e:
        md.load_daily_ohlcv("SPY", db_path=db, start_date=date(2024, 6, 3))
    assert "first 2025-03-14, last 2025-03-14" in e.value.gaps[0].found


def test_a_store_that_stops_before_the_last_completed_session_refuses(tmp_path, monkeypatch):
    db = _seed_daily(tmp_path, date(2024, 1, 2), date(2026, 10, 27))
    _fake_yahoo(monkeypatch, 504)
    with pytest.raises(ResearchDataUnavailable) as e:
        md.load_daily_ohlcv("SPY", db_path=db, start_date=date(2024, 6, 3))
    assert "first 2026-10-28, last 2026-10-30" in e.value.gaps[0].found


def test_an_empty_store_refuses_it_does_not_return_an_empty_frame(tmp_path, monkeypatch):
    db = tmp_path / "historical.db"
    HistoricalDataStore(db_path=db)
    _fake_yahoo(monkeypatch, 0)
    with pytest.raises(ResearchDataUnavailable):
        md.load_daily_ohlcv("SPY", days=730, db_path=db)


def test_a_missing_database_file_refuses(tmp_path):
    with pytest.raises(ResearchDataUnavailable) as e:
        md.load_daily_ohlcv("SPY", days=730, db_path=tmp_path / "nope.db")
    assert e.value.gaps[0].input == "history_database"


def test_a_yahoo_tagged_row_in_the_window_refuses(tmp_path, monkeypatch):
    import sqlite3
    db = _seed_daily(tmp_path, date(2024, 1, 2), LAST)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE daily_prices SET source='YAHOO_UNADJ' WHERE symbol='SPY' AND date='2025-06-02'")
    _fake_yahoo(monkeypatch, 504)
    with pytest.raises(ResearchDataUnavailable) as e:
        md.load_daily_ohlcv("SPY", db_path=db, start_date=date(2024, 6, 3))
    assert any("YAHOO_UNADJ" in g.found for g in e.value.gaps)


def test_a_request_with_neither_start_nor_days_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        md.load_daily_ohlcv("SPY", db_path=tmp_path / "x.db")


def test_a_run_pinned_in_the_past_is_judged_against_its_own_end_not_today(tmp_path, monkeypatch):
    """The pre-W116 span check measured every window to today, so a study pinned to 2025-12-31 read as 'short' more
    each day and fell through to the legacy path, which ignores the pin."""
    db = _seed_daily(tmp_path, date(2024, 1, 2), date(2025, 12, 31))
    calls = _fake_yahoo(monkeypatch, 504)
    df = md.load_daily_ohlcv("SPY", db_path=db, start_date=date(2024, 1, 2), end_date=date(2025, 12, 31))
    assert calls["n"] == 0
    assert df.index[0] == pd.Timestamp("2024-01-02") and df.index[-1] == pd.Timestamp("2025-12-31")


def test_a_days_only_request_pinned_in_the_past_counts_back_from_its_pin(tmp_path):
    db = _seed_daily(tmp_path, date(2022, 1, 3), date(2025, 12, 31))
    df = md.load_daily_ohlcv("SPY", days=730, db_path=db, end_date=date(2025, 12, 31))
    assert df.index[-1] == pd.Timestamp("2025-12-31") and df.index[0] >= pd.Timestamp("2023-11-01")


def test_stored_iv_over_a_window_pinned_in_the_past_is_not_cut_off(tmp_path):
    """load_daily_iv measured its cutoff from NOW, so the IV of a window pinned in 2023 was dropped and the engine
    silently priced from a VIX proxy."""
    db = _seed_daily(tmp_path, date(2022, 1, 3), date(2023, 12, 29))
    store = HistoricalDataStore(db_path=db)
    idx = pd.DatetimeIndex(pd.to_datetime(sessions_between(date(2022, 1, 3), date(2023, 12, 29))))
    store.save_daily_iv("SPY", pd.Series(0.21, index=idx))
    df = md.load_daily_ohlcv("SPY", db_path=db, start_date=date(2022, 1, 3), end_date=date(2023, 12, 29))
    assert df["implied_vol"].notna().all() and df["implied_vol"].iloc[0] == pytest.approx(0.21)

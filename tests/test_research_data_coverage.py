"""research-to-live-01 (R23 register) — MATERIALIZED 2026-08-31.

``load_daily_ohlcv`` judged the IB store "sufficient" with a fixed
``len(df) < 60``, independent of the window the caller asked for. The
intraday store crossed 60 trading days around 2026-08-26, so from then on a
research entry point asking for 730 days silently received ~63 and the
nightly backtest died with a misattributed "No data fetched. Check internet
connection." (reports/backtest_20260831_164000.json, exit_code 1) — the
audit predicted this date almost exactly.

Every test EXECUTES the real resolver against a real sqlite store; the Yahoo
boundary is stubbed so the suite stays offline and deterministic.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

import ait.data.market_data as md


def _seed_store(tmp_path, trading_days: int):
    """A real HistoricalDataStore holding `trading_days` of 5-min bars."""
    from ait.data.historical import HistoricalDataStore
    db = tmp_path / "historical.db"
    store = HistoricalDataStore(db_path=db)
    idx = pd.bdate_range(end=datetime.now().date(), periods=trading_days)
    rows = []
    for d in idx:
        for minute in (0, 30):
            ts = datetime(d.year, d.month, d.day, 10, minute)
            rows.append({"timestamp": ts, "Open": 100.0, "High": 101.0,
                         "Low": 99.0, "Close": 100.5, "Volume": 1_000})
    store.save_intraday("SPY", pd.DataFrame(rows).set_index("timestamp"))
    return db


def _fake_yahoo(monkeypatch, rows: int):
    """Stub the Yahoo boundary; returns a frame of `rows` daily bars."""
    calls = {"n": 0}

    class _T:
        def __init__(self, sym): pass
        def history(self, **kw):
            calls["n"] += 1
            if rows == 0:
                return pd.DataFrame()
            idx = pd.bdate_range(end=datetime.now().date(), periods=rows)
            return pd.DataFrame(
                {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                 "Volume": 1}, index=idx)

    monkeypatch.setattr(md.yf, "Ticker", _T)
    return calls


def test_short_store_falls_back_for_a_two_year_request(tmp_path, monkeypatch):
    """THE 2026-08-31 FAILURE: 63 stored days cleared the old `< 60` test, so a
    730-day request was served 63 rows and the backtest reported it as a
    network outage."""
    db = _seed_store(tmp_path, 63)
    calls = _fake_yahoo(monkeypatch, 504)
    df = md.load_daily_ohlcv("SPY", days=730, db_path=db)
    assert calls["n"] == 1, "Yahoo fallback never fired (the shipped defect)"
    assert len(df) > 400


def test_store_serving_a_full_window_is_still_preferred(tmp_path, monkeypatch):
    """Our own data stays first choice when it genuinely covers the request."""
    db = _seed_store(tmp_path, 420)
    calls = _fake_yahoo(monkeypatch, 504)
    df = md.load_daily_ohlcv("SPY", days=600, db_path=db)
    assert calls["n"] == 0, "fell back despite sufficient stored coverage"
    # exact count comes from the store itself (resample drops the partial
    # current day), so derive it rather than pinning a brittle literal
    from ait.data.historical import HistoricalDataStore
    expected = len(HistoricalDataStore(db_path=db).resample_to_daily("SPY", days=600))
    assert len(df) == expected


def test_short_request_still_served_from_the_store(tmp_path, monkeypatch):
    """A 90-day ask is satisfiable from 63 stored days via the absolute floor."""
    db = _seed_store(tmp_path, 63)
    calls = _fake_yahoo(monkeypatch, 504)
    df = md.load_daily_ohlcv("SPY", days=90, db_path=db)
    assert calls["n"] == 0
    from ait.data.historical import HistoricalDataStore
    expected = len(HistoricalDataStore(db_path=db).resample_to_daily("SPY", days=90))
    assert len(df) == expected >= 60


def test_coverage_still_short_after_fallback_is_logged_not_hidden(
        tmp_path, monkeypatch, caplog):
    """A short frame must never look healthy — silently studying 63 days is
    worse than refusing to study."""
    db = _seed_store(tmp_path, 63)
    _fake_yahoo(monkeypatch, 70)          # Yahoo also comes up short
    events: list[tuple[str, dict]] = []
    real = md.log.warning
    monkeypatch.setattr(md.log, "warning",
                        lambda e, **kw: (events.append((e, kw)), real(e, **kw))[0])
    md.load_daily_ohlcv("SPY", days=730, db_path=db)
    assert any(e == "daily_ohlcv_coverage_short" for e, _ in events), events


def test_empty_store_and_dead_yahoo_returns_empty_not_garbage(
        tmp_path, monkeypatch):
    db = _seed_store(tmp_path, 2)
    _fake_yahoo(monkeypatch, 0)
    df = md.load_daily_ohlcv("SPY", days=730, db_path=db)
    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 2

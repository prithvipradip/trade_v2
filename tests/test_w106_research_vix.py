"""Research reads the VIX it stores from IB, never Yahoo's.

W106 item 4 made research read ``^VIX`` from ``daily_iv`` (source IB_INDEX_TRADES, stored /100) on the index scale and
fall back to Yahoo only when the store was short. The review follow-up (2026-10-29) added a staleness check because a
VIX store that lags the symbols was forward-filled onto every later test day. W116 removes the fallback altogether:
the loader raises ``ResearchDataUnavailable`` when the stored series does not hold every session of the run's window,
which covers both the short store and the stale tail with one rule.
"""
from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
import pytest

import ait.data.market_data as md
from ait.backtesting.walkforward import WalkForwardBacktester
from ait.data.historical import HistoricalDataStore
from ait.data.research_inputs import ResearchDataUnavailable, last_completed_session, sessions_between
from tests.test_research_data_coverage import _fake_yahoo

TODAY = date(2026, 10, 31)
LAST = last_completed_session(TODAY)


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    monkeypatch.setattr(md, "today_et", lambda: TODAY)


def _seed_vix(tmp_path, start: date, end: date, *, symbol="VIX"):
    db = tmp_path / "historical.db"
    store = HistoricalDataStore(db_path=db)
    idx = pd.DatetimeIndex(pd.to_datetime(sessions_between(start, end)))
    bars = pd.DataFrame({"Open": 0.150, "High": 0.160, "Low": 0.140, "Close": 0.155}, index=idx)
    from ait.data.historical import SOURCE_IB_VIX_INDEX
    store.save_daily_iv_bars(symbol, bars, source=SOURCE_IB_VIX_INDEX)
    return db


def test_research_vix_comes_from_the_store_on_the_index_scale(tmp_path, monkeypatch):
    db = _seed_vix(tmp_path, date(2024, 1, 2), LAST)
    calls = _fake_yahoo(monkeypatch, rows=300)
    df = md.load_daily_ohlcv("^VIX", days=365, db_path=db)
    assert calls["n"] == 0, "Yahoo was asked although the store covers the window"
    assert df.index[-1] == pd.Timestamp(LAST) and df["Close"].iloc[-1] == pytest.approx(15.5)
    assert {"Open", "High", "Low", "Close", "Volume"} <= set(df.columns)


def test_a_short_vix_store_refuses_with_the_vix_backfill_command(tmp_path, monkeypatch):
    db = _seed_vix(tmp_path, date(2026, 9, 1), LAST)                       # ~40 sessions against a year
    calls = _fake_yahoo(monkeypatch, rows=300)
    with pytest.raises(ResearchDataUnavailable) as e:
        md.load_daily_ohlcv("^VIX", days=365, db_path=db)
    assert calls["n"] == 0
    gap = e.value.gaps[0]
    assert gap.input == "vix_family" and gap.symbol == "VIX" and "--mode vix --symbols VIX" in gap.remedy


def test_a_vix_store_that_stops_before_the_window_ends_refuses(tmp_path, monkeypatch):
    """The VIX family has its own writer and can lag the symbols (it once ended 2026-08-25): a stale tail would be
    forward-filled onto every later test day. The loader no longer accepts it."""
    db = _seed_vix(tmp_path, date(2024, 1, 2), date(2026, 8, 25))
    _fake_yahoo(monkeypatch, rows=300)
    with pytest.raises(ResearchDataUnavailable) as e:
        md.load_daily_ohlcv("^VIX", days=365, db_path=db)
    assert "first 2026-08-26" in e.value.gaps[0].found


def test_a_vix_run_pinned_in_the_past_is_judged_against_its_own_end(tmp_path, monkeypatch):
    db = _seed_vix(tmp_path, date(2023, 1, 3), date(2025, 12, 31))
    calls = _fake_yahoo(monkeypatch, rows=300)
    df = md.load_daily_ohlcv("^VIX", days=730, db_path=db, end_date=date(2025, 12, 31))
    assert calls["n"] == 0 and df.index[-1] == pd.Timestamp("2025-12-31")


def test_the_other_vix_family_members_are_read_from_their_own_series(tmp_path):
    db = _seed_vix(tmp_path, date(2024, 1, 2), LAST, symbol="VIX9D")
    df = md.load_daily_ohlcv("^VIX9D", days=365, db_path=db)
    assert df["Close"].iloc[-1] == pytest.approx(15.5)


def test_the_walk_forward_asks_for_the_stored_vix_over_the_runs_pinned_end(tmp_path, monkeypatch):
    seen: dict = {}

    def _rec(symbol, **kw):
        seen[symbol] = kw
        return pd.DataFrame()

    monkeypatch.setattr(md, "load_daily_ohlcv", _rec)
    from ait.backtesting.walkforward import WalkForwardConfig
    cfg = WalkForwardConfig(history_end_date=date(2025, 12, 31), history_start_date=date(2024, 1, 2))
    wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg, db_path=tmp_path / "h.db")
    wf._generate_windows = lambda d: []                                    # type: ignore[method-assign]
    idx = pd.bdate_range("2024-01-02", periods=150)
    spy = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)
    asyncio.run(wf.run(data={"SPY": spy}))
    assert seen["^VIX"]["end_date"] == date(2025, 12, 31) and "research" not in seen["^VIX"]


# --------------------------------------------------------------------------- the walk-forward's context loader

def _frame(end: str, close: float, tz=None) -> pd.DataFrame:
    idx = pd.bdate_range(end=end, periods=300, tz=tz)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 0.0}, index=idx)


def test_the_context_loader_returns_the_stored_frame_untouched_apart_from_tz(tmp_path):
    seen = {}

    def loader(symbol, **kw):
        seen.update(kw)
        return _frame("2026-10-27", 30.0, tz="America/New_York")

    wf = WalkForwardBacktester(["SPY"], ["iron_condor"], db_path=tmp_path / "h.db")
    vix = wf._load_vix_context(loader, 400, {"SPY": _frame("2026-10-27", 1.0)})
    assert vix.index.tz is None and vix["Close"].iloc[-1] == 30.0 and seen["days"] == 400


def test_the_context_loader_lets_a_refusal_through(tmp_path):
    def loader(symbol, **kw):
        raise ResearchDataUnavailable([], context="vix")

    wf = WalkForwardBacktester(["SPY"], ["iron_condor"], db_path=tmp_path / "h.db")
    with pytest.raises(ResearchDataUnavailable):
        wf._load_vix_context(loader, 400, {})

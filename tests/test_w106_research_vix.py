"""W106 item 4: research reads the VIX it stores from IB, not Yahoo.

`load_daily_ohlcv` sent every `^` symbol to the legacy path, which ends at Yahoo, and
the walk-forward's `^VIX` request never asked for the research path at all. Since W91
the store holds the VIX family from IB (`daily_iv`, source IB_INDEX_TRADES, stored /100).
The research path now builds `^VIX` from that series on the index scale (VIX 15, not
0.15, which is what every consumer's `vix["Close"]` expects) and falls back to Yahoo
only when the store is short, with a warning.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd
import pytest

import ait.data.market_data as md
from ait.backtesting.walkforward import WalkForwardBacktester
from ait.data.historical import HistoricalDataStore
from tests.test_research_data_coverage import _fake_yahoo


def _seed_vix(tmp_path, days: int):
    db = tmp_path / "historical.db"
    store = HistoricalDataStore(db_path=db)
    idx = pd.bdate_range(end=datetime.now().date(), periods=days)
    bars = pd.DataFrame({"Open": 0.150, "High": 0.160, "Low": 0.140, "Close": 0.155}, index=idx)
    store.save_daily_iv_bars("VIX", bars)
    return db


def test_research_vix_comes_from_the_store_on_the_index_scale(tmp_path, monkeypatch):
    db = _seed_vix(tmp_path, 300)
    calls = _fake_yahoo(monkeypatch, rows=300)
    df = md.load_daily_ohlcv("^VIX", days=365, db_path=db, research=True)
    assert calls["n"] == 0, "Yahoo was asked although the store covers the window"
    assert 250 < len(df) <= 300                                       # the requested window, from the store
    assert df["Close"].iloc[-1] == pytest.approx(15.5)
    assert {"Open", "High", "Low", "Close", "Volume"} <= set(df.columns)


def test_a_short_store_falls_back_to_yahoo(tmp_path, monkeypatch):
    db = _seed_vix(tmp_path, 20)
    calls = _fake_yahoo(monkeypatch, rows=300)
    df = md.load_daily_ohlcv("^VIX", days=365, db_path=db, research=True)
    assert calls["n"] == 1 and len(df) == 300


def test_the_live_path_is_unchanged(tmp_path, monkeypatch):
    db = _seed_vix(tmp_path, 300)
    calls = _fake_yahoo(monkeypatch, rows=300)
    md.load_daily_ohlcv("^VIX", days=365, db_path=db)          # no research flag, empty prefix
    assert calls["n"] == 1


def test_the_walk_forward_asks_for_the_research_vix(tmp_path, monkeypatch):
    seen: dict = {}

    def _rec(symbol, **kw):
        seen[symbol] = kw
        return pd.DataFrame()

    monkeypatch.setattr(md, "load_daily_ohlcv", _rec)
    wf = WalkForwardBacktester(["SPY"], ["iron_condor"], db_path=tmp_path / "h.db")
    wf._generate_windows = lambda d: []                                    # type: ignore[method-assign]
    idx = pd.bdate_range("2024-01-02", periods=150)
    spy = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)
    asyncio.run(wf.run(data={"SPY": spy}))
    assert seen["^VIX"].get("research") is True

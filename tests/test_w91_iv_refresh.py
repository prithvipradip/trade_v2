"""W91: the shared IB implied-vol refresh and the append-only daily save.

Covers: parse_ohlc_bars, refresh_daily_iv against a FAKE IB (rows land in daily_iv and
daily_prices.implied_vol, today's mid-session bar is never stored, a re-run is a no-op, an existing
reading is never rewritten, one failing symbol does not stop the rest, the VIX family is stored /100
with its own source tag), HistoricalDataStore.save(overwrite=False), and that a failing IV refresh does
not abort the supervisor's retrain script. No inspect.getsource.
"""
from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from ait.data.historical import (
    SOURCE_IB_OPTION_IV,
    SOURCE_IB_TRADES_RTH,
    SOURCE_IB_VIX_INDEX,
    HistoricalDataStore,
)
from ait.data.iv_refresh import VIX_FAMILY, parse_ohlc_bars, refresh_daily_iv, summary_line
from ait.utils.time import ET


def _store() -> HistoricalDataStore:
    return HistoricalDataStore(db_path=Path(tempfile.mkdtemp()) / "t.db")


def _bar(day, o, h, l, c):  # noqa: E741
    return SimpleNamespace(date=day, open=o, high=h, low=l, close=c)


def _prices(store, symbol, days):
    df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1},
                      index=pd.to_datetime(days))
    store.save(symbol, df, source=SOURCE_IB_TRADES_RTH)


class _FakeIB:
    """qualify/request at the broker boundary. `bars` maps symbol -> list of bars; `fail` raises."""

    def __init__(self, bars, fail=()):
        self.bars, self.fail = bars, set(fail)
        self.requests = []

    async def qualifyContractsAsync(self, contract):
        contract._sym = contract.symbol
        return [contract]

    async def reqHistoricalDataAsync(self, contract, **kw):
        self.requests.append((contract._sym, kw["whatToShow"]))
        if contract._sym in self.fail:
            raise RuntimeError("pacing violation")
        return self.bars.get(contract._sym, [])


NOW_AFTER_CLOSE = datetime(2026, 10, 23, 17, 0, tzinfo=ET)
NOW_MIDDAY = datetime(2026, 10, 23, 10, 25, tzinfo=ET)


class TestParse:
    def test_skips_non_positive_and_scales(self):
        df = parse_ohlc_bars([_bar("2026-10-21", 1, 2, 0.5, 1.5), _bar("2026-10-22", 1, 1, 1, 0),
                              _bar("2026-10-23", "x", 1, 1, 1)], scale=0.01)
        assert list(df.index.strftime("%Y-%m-%d")) == ["2026-10-21"]
        assert df["Close"].iloc[0] == pytest.approx(0.015)

    def test_empty(self):
        assert parse_ohlc_bars([]).empty


class TestRefresh:
    def _run(self, ib, store, symbols=("QQQ",), now=NOW_AFTER_CLOSE, **kw):
        return asyncio.run(refresh_daily_iv(ib, store, list(symbols), now=now, pause_secs=0, **kw))

    def test_writes_daily_iv_and_the_close_into_daily_prices(self):
        store = _store()
        _prices(store, "QQQ", ["2026-10-21", "2026-10-22"])
        ib = _FakeIB({"QQQ": [_bar("2026-10-21", .2, .3, .1, .25), _bar("2026-10-22", .25, .3, .2, .26)]})
        out = self._run(ib, store, vix_family=False)
        assert out["QQQ"]["error"] is None and out["QQQ"]["latest"] == "2026-10-22"
        bars = store.load_daily_iv_bars("QQQ")
        assert len(bars) == 2
        iv = store.load_daily_iv("QQQ", days=30)
        assert float(iv.iloc[-1]) == pytest.approx(0.26)
        assert ("QQQ", "OPTION_IMPLIED_VOLATILITY") in ib.requests

    def test_todays_mid_session_bar_is_never_stored(self):
        store = _store()
        ib = _FakeIB({"QQQ": [_bar("2026-10-22", .2, .3, .1, .25), _bar("2026-10-23", .2, .3, .1, .99)]})
        out = self._run(ib, store, now=NOW_MIDDAY, vix_family=False)
        assert out["QQQ"]["latest"] == "2026-10-22"
        dates = list(store.load_daily_iv_bars("QQQ").index.strftime("%Y-%m-%d"))
        assert dates == ["2026-10-22"]

    def test_rerun_is_a_noop_and_never_rewrites_a_stored_reading(self):
        store = _store()
        _prices(store, "QQQ", ["2026-10-22"])
        ib = _FakeIB({"QQQ": [_bar("2026-10-22", .2, .3, .1, .25)]})
        self._run(ib, store, vix_family=False)
        ib2 = _FakeIB({"QQQ": [_bar("2026-10-22", .9, .9, .9, .9)]})     # a later, different print
        out = self._run(ib2, store, vix_family=False)
        assert out["QQQ"]["rows"] == 0
        assert float(store.load_daily_iv("QQQ", days=30).iloc[-1]) == pytest.approx(0.25)
        assert float(store.load_daily_iv_bars("QQQ")["Close"].iloc[-1]) == pytest.approx(0.25)

    def test_one_failing_symbol_does_not_stop_the_rest(self):
        store = _store()
        ib = _FakeIB({"QQQ": [_bar("2026-10-22", .2, .3, .1, .25)], "IWM": [_bar("2026-10-22", .2, .3, .1, .3)]},
                     fail={"SPY"})
        out = self._run(ib, store, symbols=("QQQ", "SPY", "IWM"), vix_family=False)
        assert out["SPY"]["error"] and "pacing" in out["SPY"]["error"]
        assert out["QQQ"]["rows"] > 0 and out["IWM"]["rows"] > 0
        assert "SPY:ERR" in summary_line(out)

    def test_vix_family_is_stored_divided_by_100_with_its_own_source(self):
        store = _store()
        ib = _FakeIB({v: [_bar("2026-10-22", 20.0, 21.0, 19.0, 20.5)] for v in VIX_FAMILY})
        out = self._run(ib, store, symbols=(), vix_family=True)
        assert set(out) == set(VIX_FAMILY)
        bars = store.load_daily_iv_bars("VIX3M")
        assert float(bars["Close"].iloc[-1]) == pytest.approx(0.205)
        assert ("VIX3M", "TRADES") in ib.requests
        import sqlite3
        with sqlite3.connect(store._db_path) as c:
            src = c.execute("SELECT DISTINCT source FROM daily_iv WHERE symbol='VIX3M'").fetchall()
        assert src == [(SOURCE_IB_VIX_INDEX,)]

    def test_source_tag_on_stock_iv(self):
        store = _store()
        ib = _FakeIB({"QQQ": [_bar("2026-10-22", .2, .3, .1, .25)]})
        self._run(ib, store, vix_family=False)
        import sqlite3
        with sqlite3.connect(store._db_path) as c:
            src = c.execute("SELECT DISTINCT source FROM daily_iv WHERE symbol='QQQ'").fetchall()
        assert src == [(SOURCE_IB_OPTION_IV,)]


class TestAppendOnlySave:
    def test_existing_row_is_left_alone_and_a_new_date_is_inserted(self):
        store = _store()
        first = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100]},
                             index=pd.to_datetime(["2026-10-20"]))
        store.save("QQQ", first, source="IB_TRADES_RTH")
        again = pd.DataFrame({"Open": [9.0, 3.0], "High": [9.0, 3.0], "Low": [9.0, 3.0],
                              "Close": [9.0, 3.0], "Volume": [9, 30]},
                             index=pd.to_datetime(["2026-10-20", "2026-10-21"]))
        store.save("QQQ", again, source="YAHOO_UNADJ", overwrite=False)
        df = store.load("QQQ")
        assert float(df.loc["2026-10-20", "Close"]) == 1.5 and int(df.loc["2026-10-20", "Volume"]) == 100
        assert float(df.loc["2026-10-21", "Close"]) == 3.0
        import sqlite3
        with sqlite3.connect(store._db_path) as c:
            tags = dict(c.execute("SELECT date, source FROM daily_prices WHERE symbol='QQQ'").fetchall())
        assert tags["2026-10-20"] == "IB_TRADES_RTH" and tags["2026-10-21"] == "YAHOO_UNADJ"

    def test_a_price_less_iv_only_row_is_filled_but_a_priced_row_is_not(self):
        """The IV writers create a skeleton row (NULL prices) for today before its bar exists; the append-only
        save must still fill it, or that date would have no prices forever (found 2026-10-24)."""
        store = _store()
        store.save_daily_iv("QQQ", pd.Series([0.25], index=pd.to_datetime(["2026-10-23"])), source="IB_OPTION_IV")
        import sqlite3
        with sqlite3.connect(store._db_path) as c:
            assert c.execute("SELECT close FROM daily_prices WHERE date='2026-10-23'").fetchone() == (None,)
        bar = pd.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100]},
                           index=pd.to_datetime(["2026-10-23"]))
        store.save("QQQ", bar, source="IB_TRADES_RTH", overwrite=False)
        with sqlite3.connect(store._db_path) as c:
            row = c.execute("SELECT close, volume, implied_vol, source FROM daily_prices WHERE date='2026-10-23'").fetchone()
        assert row == (1.5, 100, 0.25, "IB_TRADES_RTH")            # prices filled, the stored IV kept
        store.save("QQQ", bar.assign(Close=9.0, Volume=9), source="YAHOO_UNADJ", overwrite=False)
        with sqlite3.connect(store._db_path) as c:
            assert c.execute("SELECT close, volume FROM daily_prices WHERE date='2026-10-23'").fetchone() == (1.5, 100)

    def test_default_still_overwrites(self):
        store = _store()
        a = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1]},
                         index=pd.to_datetime(["2026-10-20"]))
        store.save("QQQ", a)
        b = a.assign(Close=2.0)
        store.save("QQQ", b)
        assert float(store.load("QQQ").loc["2026-10-20", "Close"]) == 2.0


class TestRetrainScriptSurvivesAnIvFailure:
    def test_iv_refresh_raising_does_not_abort_the_retrain(self, monkeypatch, tmp_path):
        import ait.orchestration.master as M
        from ait.ml.trainer import ModelTrainer

        monkeypatch.setattr(M, "LOGS_DIR", tmp_path)
        launched = {}

        class _Captured(Exception):
            pass

        def _popen(argv, **kw):
            launched["argv"] = argv
            raise _Captured()

        monkeypatch.setattr(M.subprocess, "Popen", _popen)
        M.retrain_models()
        script = launched["argv"][-1]

        trained = {"called": False}

        async def _train(self, symbols):
            trained["called"] = True
            return {}

        class _Svc:
            def __init__(self, *a, **kw):
                pass

            async def get_historical(self, symbol, days=0, **kw):
                return pd.DataFrame({"Close": [1.0]})

        ib = MagicMock()
        ib.disconnect = AsyncMock()
        monkeypatch.setattr(ModelTrainer, "train_all_symbols", _train)
        monkeypatch.setattr("ait.orchestration.retrain_ib.connect_from_settings", AsyncMock(return_value=ib))
        monkeypatch.setattr("ait.data.market_data.MarketDataService", _Svc)
        monkeypatch.setattr("ait.data.historical.HistoricalDataStore", MagicMock())
        monkeypatch.setattr("ait.data.iv_refresh.refresh_daily_iv", AsyncMock(side_effect=RuntimeError("boom")))
        monkeypatch.chdir(Path(__file__).resolve().parents[1])
        exec(compile(script, "<supervisor retrain script>", "exec"), {"__name__": "__main__"})
        assert trained["called"] is True
        ib.disconnect.assert_awaited()

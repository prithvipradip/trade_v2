"""W89: the daily retrain gets its own IB session and never stores Yahoo bars.

Covers: MarketDataService's opt-out of the Yahoo daily fallback, connect_for_retrain's client-id
and port rules (fake client at the broker boundary), the supervisor's retrain script aborting with
exit 3 BEFORE any training/write when IB data is missing, drop_incomplete_session (pinned non-ET
TZ), and the trainer persisting no partial bar. No inspect.getsource.
"""
from __future__ import annotations

import asyncio
import time as _time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

import ait.orchestration.master as M
from ait.config.settings import IBKREnvConfig, MLConfig
from ait.data.market_data import MarketDataService
from ait.orchestration.retrain_ib import (
    RetrainIBUnavailable,
    connect_for_retrain,
    connect_from_settings,
    refusal_reason,
)
from ait.utils.time import ET, drop_incomplete_session

REPO = Path(__file__).resolve().parents[1]


def _frame(dates):
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1},
                        index=pd.to_datetime(dates))


# ---------------------------------------------------------------------------
# MarketDataService: the Yahoo daily fallback can be switched off
# ---------------------------------------------------------------------------

class TestDailyYahooFallbackOptOut:
    def _svc(self, fallback):
        svc = MarketDataService(None, daily_yahoo_fallback=fallback)
        calls = {"yahoo": 0}

        async def _yahoo(symbol, days, interval):
            calls["yahoo"] += 1
            return _frame(["2026-10-20"])

        svc._get_yahoo_historical = _yahoo
        return svc, calls

    def test_off_returns_none_and_never_calls_yahoo(self):
        svc, calls = self._svc(False)
        assert asyncio.run(svc.get_historical("QQQ", days=504)) is None
        assert calls["yahoo"] == 0

    def test_default_still_falls_back_to_yahoo(self):
        svc, calls = self._svc(True)
        df = asyncio.run(svc.get_historical("QQQ", days=504))
        assert df is not None and calls["yahoo"] == 1
        assert df.attrs["source"] == "YAHOO_UNADJ"

    def test_off_does_not_affect_intraday(self):
        svc, calls = self._svc(False)

        async def _intra(symbol, interval, days):
            calls["yahoo"] += 1
            return _frame(["2026-10-20"])

        svc._get_yahoo_intraday = _intra
        assert asyncio.run(svc.get_intraday("QQQ", "1h", 3)) is not None


# ---------------------------------------------------------------------------
# connect_for_retrain
# ---------------------------------------------------------------------------

class _FakeIB:
    def __init__(self, cfg, ok=True):
        self.cfg, self._ok = cfg, ok

    async def connect(self):
        return self._ok


def _env(port=7497, base=1):
    return IBKREnvConfig(ibkr_port=port, ibkr_client_id=base)


class TestConnectForRetrain:
    def _go(self, **kw):
        kw.setdefault("client_id", 94)
        kw.setdefault("short_dte_client_id", 91)
        kw.setdefault("listening", lambda p: True)
        kw.setdefault("client_factory", lambda cfg: _FakeIB(cfg))
        return asyncio.run(connect_for_retrain(_env(), **kw))

    def test_connects_with_its_own_id_and_port(self):
        client = self._go()
        assert client.cfg.ibkr_client_id == 94 and client.cfg.ibkr_port == 7497

    def test_uses_the_listening_twin_port(self):
        client = self._go(listening=lambda p: p == 4002)
        assert client.cfg.ibkr_port == 4002          # TWS closed, Gateway paper open

    @pytest.mark.parametrize("cid", [1, 101, 105, 91])
    def test_refuses_reserved_ids(self, cid):
        with pytest.raises(RetrainIBUnavailable, match="collides"):
            self._go(client_id=cid)

    def test_nothing_listening(self):
        with pytest.raises(RetrainIBUnavailable, match="down"):
            self._go(listening=lambda p: False)

    def test_connect_failure(self):
        with pytest.raises(RetrainIBUnavailable, match="connection failed"):
            self._go(client_factory=lambda cfg: _FakeIB(cfg, ok=False))

    def test_refusal_reason_free_id(self):
        assert refusal_reason(94, 1, 91) is None


class TestConnectFromSettings:
    def test_reads_the_ids_from_config(self):
        from ait.config.settings import load_settings
        st = load_settings(REPO / "config.yaml")
        client = asyncio.run(connect_from_settings(
            st, _env(), listening=lambda p: True, client_factory=lambda cfg: _FakeIB(cfg)))
        assert client.cfg.ibkr_client_id == st.ml.retrain_ib_client_id == 94


class TestSettingsKey:
    def test_default_and_bounds(self):
        assert MLConfig().retrain_ib_client_id == 94
        with pytest.raises(Exception):
            MLConfig(retrain_ib_client_id=1)
        with pytest.raises(Exception):
            MLConfig(retrain_ib_clientid=94)         # a typo'd key is a startup error


# ---------------------------------------------------------------------------
# The supervisor's retrain script: abort BEFORE any write when IB data is missing
# ---------------------------------------------------------------------------

def _script(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "LOGS_DIR", tmp_path)
    launched = {}

    class _Captured(Exception):
        pass

    def _popen(argv, **kw):
        launched["argv"] = argv
        raise _Captured()

    monkeypatch.setattr(M.subprocess, "Popen", _popen)
    M.retrain_models()
    return launched["argv"][-1]


def _run_script(monkeypatch, script, *, connect, history):
    from ait.ml.trainer import ModelTrainer

    trained = {"called": False}

    async def _train(self, symbols):
        trained["called"] = True
        return {}

    class _Svc:
        def __init__(self, ibkr, **kw):
            trained["fallback"] = kw.get("daily_yahoo_fallback")

        async def get_historical(self, symbol, days=0, **kw):
            return history(symbol)

    monkeypatch.setattr(ModelTrainer, "train_all_symbols", _train)
    monkeypatch.setattr("ait.orchestration.retrain_ib.connect_from_settings", connect)
    monkeypatch.setattr("ait.data.market_data.MarketDataService", _Svc)
    monkeypatch.setattr("ait.data.historical.HistoricalDataStore", MagicMock())
    monkeypatch.chdir(REPO)
    code = None
    try:
        exec(compile(script, "<supervisor retrain script>", "exec"), {"__name__": "__main__"})
    except SystemExit as e:
        code = e.code
    return code, trained


class TestRetrainScriptAbortsWithoutIB:
    def test_trains_when_every_series_comes_from_ib(self, monkeypatch, tmp_path):
        script = _script(monkeypatch, tmp_path)
        ib = MagicMock()
        ib.disconnect = AsyncMock()
        code, trained = _run_script(monkeypatch, script, connect=AsyncMock(return_value=ib),
                                    history=lambda s: _frame(["2026-10-20"]))
        assert code is None and trained["called"] is True
        assert trained["fallback"] is False          # Yahoo is OFF for the retrain
        ib.disconnect.assert_awaited()

    def test_exit_3_when_ib_is_unavailable(self, monkeypatch, tmp_path):
        script = _script(monkeypatch, tmp_path)
        code, trained = _run_script(
            monkeypatch, script,
            connect=AsyncMock(side_effect=RetrainIBUnavailable("TWS down")),
            history=lambda s: _frame(["2026-10-20"]))
        assert code == 3 and trained["called"] is False

    def test_exit_3_before_training_when_one_series_is_missing(self, monkeypatch, tmp_path):
        script = _script(monkeypatch, tmp_path)
        ib = MagicMock()
        ib.disconnect = AsyncMock()
        code, trained = _run_script(
            monkeypatch, script, connect=AsyncMock(return_value=ib),
            history=lambda s: None if s == "^VIX" else _frame(["2026-10-20"]))
        assert code == 3 and trained["called"] is False
        ib.disconnect.assert_awaited()               # the session is released even on abort


# ---------------------------------------------------------------------------
# drop_incomplete_session (pinned non-ET TZ) and the trainer's save path
# ---------------------------------------------------------------------------

@pytest.fixture
def tokyo_tz():
    import os
    old = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Tokyo"
    _time.tzset()
    yield
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    _time.tzset()


class TestDropIncompleteSession:
    DF = _frame(["2026-10-21", "2026-10-22"])

    def test_mid_session_drops_today(self, tokyo_tz):
        now = datetime(2026, 10, 22, 10, 25, tzinfo=ET)
        assert list(drop_incomplete_session(self.DF, now).index.strftime("%Y-%m-%d")) == ["2026-10-21"]

    def test_just_before_1605_still_drops(self, tokyo_tz):
        now = datetime(2026, 10, 22, 16, 4, tzinfo=ET)
        assert len(drop_incomplete_session(self.DF, now)) == 1

    def test_after_1605_keeps_today(self, tokyo_tz):
        now = datetime(2026, 10, 22, 16, 5, tzinfo=ET)
        assert len(drop_incomplete_session(self.DF, now)) == 2

    def test_non_datetime_index_passes_through(self):
        df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
        assert drop_incomplete_session(df, datetime(2026, 10, 22, 10, 0, tzinfo=ET)) is df

    def test_empty_and_none_pass_through(self):
        assert drop_incomplete_session(None) is None
        assert drop_incomplete_session(self.DF.iloc[0:0]).empty


class TestTrainerStoresNoPartialBar:
    def test_save_gets_a_frame_without_today(self, monkeypatch):
        from ait.ml.trainer import ModelTrainer
        import ait.ml.trainer as T

        monkeypatch.setattr(T, "drop_incomplete_session",
                            lambda df: df[df.index < pd.Timestamp("2026-10-22")])
        df = _frame(["2026-10-21", "2026-10-22"])
        df.attrs["source"] = "IB_TRADES_RTH"
        saved = {}
        store = MagicMock()
        store.save = lambda sym, d, source=None, overwrite=True: saved.update(
            sym=sym, rows=len(d), source=source, overwrite=overwrite)
        md = MagicMock()
        md.get_historical = AsyncMock(return_value=df)
        predictor = MagicMock()
        predictor.train.return_value = {}
        trainer = ModelTrainer(MLConfig(), predictor, md, store)
        monkeypatch.setattr(trainer, "_fetch_market_context", AsyncMock(return_value={}))
        asyncio.run(trainer.train_all_symbols(["QQQ"]))
        assert saved == {"sym": "QQQ", "rows": 1, "source": "IB_TRADES_RTH", "overwrite": False}

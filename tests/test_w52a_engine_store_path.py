"""W52a: the research engine reads the store it was pointed at, and says so
loudly when it falls back.

Registered by the 2026-09-22 post-FOMC study: a walk-forward run from a
checkout without the live data directory silently used an auto-created
EMPTY historical.db — Yahoo bars, no stored IBKR implied vol, no spread
calibration — and reported a different harness (n 87 / PF 0.83 vs n 50 /
PF 0.45, same code, same dates). Two causes: ``_load_calibrated_spreads``
opened ``HistoricalDataStore()`` at the cwd-relative default, ignoring the
``db_path`` the walk-forward had been given; and the daily loader's
empty-store fallback was an info line.

Now: ``Backtester(db_path=...)`` exists and is threaded from both
walk-forward construction sites; the daily loader WARNS (not infos) when
the store holds nothing at all for the symbol and Yahoo is the only source;
the engine WARNS when it could not load settings and is running on model
defaults.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import ait.backtesting.engine as engine_mod
import ait.backtesting.walkforward as wf_mod
import ait.data.market_data as md
from ait.backtesting.engine import Backtester
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.data.historical import HistoricalDataStore

from tests.test_r20_research_validity import (  # noqa: F401 — shared fixtures
    _engine_kwargs,
    _recorder_factory,
    _run_window,
    synth_df,
    synth_features,
)
from tests.test_w2_orchestrator_gates import LogCapture

PARAMS = {
    "calibrated_on": "2026-09-01", "spread_base": 0.0123,
    "spread_iv_sensitivity": 0.4, "spread_iv_threshold": 0.25,
    "spread_dte_sensitivity": 0.001, "spread_dte_threshold": 10,
    "spread_cap": 0.08, "sample_count": 500, "rmse": 0.003,
}


class TestTheEngineReadsTheStoreItWasGiven:
    def test_calibrated_spreads_come_from_db_path(self, tmp_path, monkeypatch,
                                                  synth_df, synth_features):
        store_path = tmp_path / "elsewhere" / "historical.db"
        HistoricalDataStore(db_path=store_path).save_spread_params("SPY", PARAMS)
        (tmp_path / "cwd").mkdir()
        monkeypatch.chdir(tmp_path / "cwd")
        monkeypatch.setenv("AIT_BT_CALIBRATED_SPREADS", "1")

        bt = Backtester(**_engine_kwargs(synth_df, synth_features),
                        symbol="SPY", db_path=store_path)
        assert bt._calibrated_spreads, "the calibration at db_path was not loaded"
        assert bt._spread_base == pytest.approx(0.0123)
        assert not (tmp_path / "cwd" / "data" / "historical.db").exists(), (
            "the engine still created/opened a cwd-relative store")

    def test_without_db_path_the_default_store_is_used(self, tmp_path, monkeypatch,
                                                       synth_df, synth_features):
        """The pre-fix behaviour for callers that pass nothing is kept —
        but it is a choice now, not the only path."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AIT_BT_CALIBRATED_SPREADS", "1")
        bt = Backtester(**_engine_kwargs(synth_df, synth_features), symbol="SPY")
        assert not bt._calibrated_spreads


class TestTheWalkForwardThreadsIt:
    def test_oos_backtester_receives_db_path(self, tmp_path, synth_df, monkeypatch):
        db = tmp_path / "h.db"
        cfg = WalkForwardConfig(train_window_models=False)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg, db_path=db)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink, "no OOS Backtester was constructed"
        assert sink[-1].get("db_path") == db

    def test_meta_label_shadow_backtester_receives_db_path(self, tmp_path, synth_df, monkeypatch):
        db = tmp_path / "h.db"
        cfg = WalkForwardConfig()
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg, db_path=db)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        wf._train_window_meta_labeler(
            train_df=synth_df, symbol="SPY", window_id=1,
            predictor=None, window_cfg=cfg)
        assert sink, "shadow Backtester was not constructed"
        assert sink[-1].get("db_path") == db


def _yahoo_frame(n: int = 700) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                         "Close": 100.0, "Volume": 1e6, "Dividends": 0.0,
                         "Stock Splits": 0.0}, index=idx)


class TestTheFallbackIsLoud:
    def test_an_empty_store_warns_that_yahoo_is_the_only_source(self, tmp_path, monkeypatch):
        cap = LogCapture(md.log)
        monkeypatch.setattr(md, "log", cap)

        class _Ticker:
            def __init__(self, symbol): pass
            def history(self, **kw): return _yahoo_frame()
        monkeypatch.setattr(md.yf, "Ticker", _Ticker)

        empty = tmp_path / "empty.db"
        df = md.load_daily_ohlcv("SPY", days=730, db_path=empty)
        assert len(df) > 0
        warns = [kw for lvl, e, kw in cap.events
                 if lvl == "warning" and e == "daily_ohlcv_store_empty_yahoo_only"]
        assert warns, "an empty store fell back to Yahoo without a warning"
        assert warns[0]["symbol"] == "SPY"
        assert str(empty) in str(warns[0]["db_path"])
        assert "implied_vol" in df.columns and df["implied_vol"].isna().all()

    def test_a_partial_store_keeps_the_info_line(self, tmp_path, monkeypatch):
        """Short-but-not-empty coverage is the R23 case, not this one."""
        cap = LogCapture(md.log)
        monkeypatch.setattr(md, "log", cap)

        class _Ticker:
            def __init__(self, symbol): pass
            def history(self, **kw): return _yahoo_frame()
        monkeypatch.setattr(md.yf, "Ticker", _Ticker)
        store = HistoricalDataStore(db_path=tmp_path / "partial.db")
        monkeypatch.setattr(store.__class__, "resample_to_daily",
                            lambda self, symbol, days: _yahoo_frame(20).iloc[:, :5])

        md.load_daily_ohlcv("SPY", days=730, db_path=tmp_path / "partial.db")
        assert "daily_ohlcv_store_empty_yahoo_only" not in cap.names()
        assert "daily_ohlcv_yahoo_fallback" in cap.names()

    def test_the_engine_warns_when_it_runs_on_model_defaults(self, monkeypatch,
                                                              synth_df, synth_features):
        cap = LogCapture(engine_mod.log)
        monkeypatch.setattr(engine_mod, "log", cap)

        def _no_config(*a, **k):
            raise FileNotFoundError("Config file not found: config.yaml")
        import ait.config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "load_settings", _no_config)

        Backtester(**_engine_kwargs(synth_df, synth_features))
        warns = [kw for lvl, e, kw in cap.events
                 if lvl == "warning" and e == "engine_settings_unavailable_using_defaults"]
        assert warns, "the engine fell back to pydantic defaults silently"
        assert "cwd" in warns[0]

    def test_no_warning_when_settings_are_supplied(self, monkeypatch, synth_df, synth_features):
        from ait.config.settings import Settings
        cap = LogCapture(engine_mod.log)
        monkeypatch.setattr(engine_mod, "log", cap)
        Backtester(**_engine_kwargs(synth_df, synth_features), settings=Settings())
        assert "engine_settings_unavailable_using_defaults" not in cap.names()

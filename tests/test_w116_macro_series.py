"""W116 WP6: research reads the stored FRED macro series (one session of lag) instead of passing none.

Until now research handed ``FeatureEngine`` no ``macros``, so the nine macro columns were constants in every research
model while live trains and scores on the real series. Each test executes the real code path against a real sqlite file.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.data import macro_store as ms
from ait.data import research_inputs as ri
from ait.data.historical import HistoricalDataStore
from ait.data.macro import FRED_SERIES

ROOT = Path(__file__).resolve().parent.parent
WINDOW = ri.Window(date(2024, 3, 4), date(2024, 6, 28))


def _obs(start="2024-01-02", end="2024-07-03", base=4.0):
    idx = pd.bdate_range(start, end)
    return pd.Series(base + np.arange(len(idx)) * 0.01, index=idx)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "h.db"
    HistoricalDataStore(db_path=path)
    for k, sid in enumerate(ms.SERIES_IDS):
        ms.save_macro_series(path, sid, _obs(base=1.0 + k), "2024-07-04T00:00:00")
    return path


def _req(db, **kw):
    base = dict(window=WINDOW, symbols=(), db_path=db, max_null_iv_run=0, max_dividend_gap_days=0, needs_iv=False,
                macro_series=ms.SERIES_IDS, max_macro_gap_days=5, max_macro_tail_days=10)
    base.update(kw)
    return ri.ResearchRequest(**base)


class TestParse:
    def test_fred_missing_marker_becomes_nan_and_the_columns_are_found(self):
        s = ms.parse_fred_csv("observation_date,DGS10\n2024-01-02,4.1\n2024-01-03,.\n2024-01-04,4.3\n")
        assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-04"]
        assert s.iloc[0] == 4.1 and np.isnan(s.iloc[1])

    def test_an_unrecognised_file_is_an_error(self):
        with pytest.raises(ValueError):
            ms.parse_fred_csv("a,b\n1,2\n")


class TestTheLag:
    def test_the_value_known_at_a_sessions_open_is_the_previous_observation(self):
        obs = pd.Series({pd.Timestamp("2024-03-01"): 1.0, pd.Timestamp("2024-03-04"): 2.0, pd.Timestamp("2024-03-05"): 3.0})
        lagged = ms.lag_one_session(obs, ri.Window(date(2024, 3, 4), date(2024, 3, 6)))
        assert lagged.loc["2024-03-04"] == 1.0         # Monday sees Friday's
        assert lagged.loc["2024-03-05"] == 2.0         # Tuesday sees Monday's
        assert lagged.loc["2024-03-06"] == 3.0
        assert pd.Timestamp("2024-03-03") not in lagged.index

    def test_no_value_ever_comes_from_the_session_itself_or_later(self):
        obs = _obs()
        lagged = ms.lag_one_session(obs, ri.Window(date(2024, 3, 4), date(2024, 6, 28)))
        for d, v in lagged.items():
            assert v == obs[obs.index < d].iloc[-1], d

    def test_a_fred_holiday_carries_the_last_value_forward(self):
        obs = pd.Series({pd.Timestamp("2024-05-24"): 5.0, pd.Timestamp("2024-05-28"): 6.0})     # Monday 05-27 is Memorial Day
        lagged = ms.lag_one_session(obs, ri.Window(date(2024, 5, 28), date(2024, 5, 29)))
        assert lagged.loc["2024-05-28"] == 5.0 and lagged.loc["2024-05-29"] == 6.0


class TestTheCheck:
    def test_complete_series_pass(self, db):
        assert ri.check_research_inputs(_req(db)).ok

    def test_a_missing_table_is_a_gap_with_the_backfill_command(self, tmp_path):
        path = tmp_path / "e.db"
        HistoricalDataStore(db_path=path)
        (g,) = ri.check_research_inputs(_req(path)).gaps
        assert g.input == "macro_series" and "backfill_macro_series.py" in g.remedy

    def test_one_missing_series_is_named(self, db):
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM macro_series WHERE series_id='DTWEXBGS'")
        (g,) = ri.check_research_inputs(_req(db)).gaps
        assert g.symbol == "DTWEXBGS" and g.found == "no rows"

    def test_a_stale_tail_is_a_gap_but_a_week_late_dollar_index_is_not(self, db):
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM macro_series WHERE series_id='DGS10' AND date>'2024-06-10'")
            conn.execute("DELETE FROM macro_series WHERE series_id='DTWEXBGS' AND date>'2024-06-21'")
        gaps = ri.check_research_inputs(_req(db)).gaps
        assert [g.symbol for g in gaps] == ["DGS10"] and "the last observation is 2024-06-10" in gaps[0].found

    def test_a_hole_inside_the_window_is_a_gap(self, db):
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM macro_series WHERE series_id='DGS2' AND date>'2024-04-01' AND date<'2024-04-15'")
        (g,) = ri.check_research_inputs(_req(db)).gaps
        assert g.symbol == "DGS2" and "gap(s) longer than 5 days" in g.found

    def test_a_series_that_starts_late_is_a_gap(self, db):
        with sqlite3.connect(db) as conn:
            conn.execute("DELETE FROM macro_series WHERE series_id='T10Y2Y' AND date<'2024-03-20'")
        (g,) = ri.check_research_inputs(_req(db, window=ri.Window(date(2024, 3, 4), date(2024, 6, 28)),
                                             vix_window=ri.Window(date(2024, 3, 4), date(2024, 6, 28)))).gaps
        assert g.symbol == "T10Y2Y" and "the first observation is" in g.found

    def test_a_foreign_tag_is_a_gap(self, db):
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE macro_series SET source='MANUAL' WHERE series_id='DGS2' AND date='2024-04-02'")
        (g,) = ri.check_research_inputs(_req(db)).gaps
        assert "MANUAL" in g.found

    def test_not_checked_when_no_macro_series_are_needed(self, db):
        assert "macro_series" not in ri.check_research_inputs(_req(db, macro_series=())).checked


class TestTheLoader:
    def test_it_returns_the_four_lagged_series_by_their_friendly_names(self, db):
        out = ms.load_macro_series(db, WINDOW)
        assert set(out) == set(FRED_SERIES)
        assert all(s.index[0] == pd.Timestamp("2024-03-04") for s in out.values())

    def test_it_refuses_when_the_store_does_not_cover_the_window(self, tmp_path):
        path = tmp_path / "e.db"
        HistoricalDataStore(db_path=path)
        with pytest.raises(ri.ResearchDataUnavailable):
            ms.load_macro_series(path, WINDOW)


class TestTheFeaturesSeeTheSeries:
    def test_the_macro_columns_are_no_longer_constants(self, db):
        from ait.ml.features import FeatureEngine
        idx = pd.bdate_range("2024-03-04", "2024-06-28")
        frame = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e6}, index=idx)
        engine = FeatureEngine()
        constant = engine._add_macro(frame.copy(), None)
        real = engine._add_macro(frame.copy(), {"macros": ms.load_macro_series(db, WINDOW)})
        assert constant["us_2y_yield_level"].nunique() == 1
        assert real["us_2y_yield_level"].nunique() > 10
        # session D carries the observation of D-1: D's own level is never used
        obs = _obs(base=1.0)
        d = pd.Timestamp("2024-04-10")
        assert real.loc[d, "us_2y_yield_level"] == pytest.approx(float(obs[obs.index < d].iloc[-1]))


class TestTheRequestBuilder:
    def test_the_default_features_need_the_macro_series(self):
        from ait.backtesting.preflight import macro_features_used
        from ait.config.settings import load_settings
        assert macro_features_used(load_settings()) is True

    def test_a_model_without_macro_columns_does_not(self):
        from ait.backtesting.preflight import macro_features_used
        from ait.config.settings import load_settings
        s = load_settings()
        use = {k: False for k in s.ml.features.macro.use.model_dump()} if hasattr(s.ml.features.macro.use, "model_dump") \
            else {k: False for k in s.ml.features.macro.use}
        macro = s.ml.features.macro.model_copy(update={"use": type(s.ml.features.macro.use)(**use)
                                                       if hasattr(s.ml.features.macro.use, "model_dump") else use})
        s2 = s.model_copy(update={"ml": s.ml.model_copy(update={"features": s.ml.features.model_copy(update={"macro": macro})})})
        assert macro_features_used(s2) is False


class TestTheBackfillScript:
    def _script(self):
        spec = importlib.util.spec_from_file_location("_w116_macro_backfill", ROOT / "scripts/backfill_macro_series.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_w116_macro_backfill"] = mod
        spec.loader.exec_module(mod)
        return mod

    CSV = "observation_date,{sid}\n" + "\n".join(f"2024-03-{d:02d},{3 + d / 100}" for d in range(1, 29)) + "\n"

    def test_it_stores_every_series_and_a_dry_run_stores_nothing(self, tmp_path, monkeypatch, capsys):
        mod = self._script()
        monkeypatch.setattr(mod, "fetch", lambda sid, timeout=30.0: self.CSV.format(sid=sid))
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db)
        assert mod.main(["--db-path", str(db), "--dry-run", "--since", "2024-01-01"]) == 0
        with sqlite3.connect(db) as conn:
            assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='macro_series'").fetchone()
        assert mod.main(["--db-path", str(db), "--since", "2024-01-01"]) == 0
        with sqlite3.connect(db) as conn:
            rows = dict(conn.execute("SELECT series_id, COUNT(*) FROM macro_series GROUP BY 1").fetchall())
            tags = {t for (t,) in conn.execute("SELECT DISTINCT source FROM macro_series")}
        assert set(rows) == set(ms.SERIES_IDS) and all(n == 28 for n in rows.values()) and tags == {"FRED_CSV"}

    def test_a_series_that_fails_to_download_fails_the_run_but_not_the_others(self, tmp_path, monkeypatch, capsys):
        mod = self._script()

        def _fetch(sid, timeout=30.0):
            if sid == "DGS10":
                raise OSError("network down")
            return self.CSV.format(sid=sid)
        monkeypatch.setattr(mod, "fetch", _fetch)
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db)
        assert mod.main(["--db-path", str(db), "--since", "2024-01-01"]) == 1
        with sqlite3.connect(db) as conn:
            stored = {s for (s,) in conn.execute("SELECT DISTINCT series_id FROM macro_series")}
        assert stored == set(ms.SERIES_IDS) - {"DGS10"}

"""W91: scripts/unify_history_tables.py -- merge the research_* tables into the shared set, verify, archive.

Builds a small database with BOTH table sets through the real HistoricalDataStore (so the schemas are the
production ones), then drives the real merge / verify / archive code. No inspect.getsource.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from ait.data.historical import SOURCE_IB_OPTION_IV, SOURCE_IB_TRADES_RTH, HistoricalDataStore

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("unify_mod", REPO / "scripts" / "unify_history_tables.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["unify_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


U = _load()


def _frame(rows):
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({"Open": [r[1] for r in rows], "High": [r[1] for r in rows], "Low": [r[1] for r in rows],
                         "Close": [r[1] for r in rows], "Volume": [r[2] for r in rows]}, index=idx)


@pytest.fixture
def db():
    path = Path(tempfile.mkdtemp()) / "h.db"
    live = HistoricalDataStore(db_path=path)
    res = HistoricalDataStore(db_path=path, table_prefix="research_")
    # live: QQQ with zero volume (old rows), one live-only IB day, another symbol
    live.save("QQQ", _frame([("2026-10-06", 100.0, 0), ("2026-10-07", 101.0, 0), ("2026-10-09", 103.0, 50)]),
              source=None)
    live.save("AAPL", _frame([("2026-10-06", 200.0, 0)]))
    # research (IB): real volume, one date live lacks (10-08), same dates as live otherwise
    res.save("QQQ", _frame([("2026-10-06", 100.01, 1000), ("2026-10-07", 101.0, 1100),
                            ("2026-10-08", 102.0, 1200)]), source=SOURCE_IB_TRADES_RTH)
    # IV: live has one reading research lacks; research has readings live lacks
    import sqlite3 as _s
    with _s.connect(path) as c:
        c.execute("UPDATE daily_prices SET implied_vol=0.5, iv_source='LIVE' WHERE symbol='QQQ' AND date='2026-10-06'")
        c.execute("UPDATE research_daily_prices SET implied_vol=0.2, iv_source=? WHERE symbol='QQQ' "
                  "AND date='2026-10-07'", (SOURCE_IB_OPTION_IV,))
    live.save_daily_iv_bars("VIX", _frame([("2026-10-06", 0.15, 0)])[["Open", "High", "Low", "Close"]])
    res.save_daily_iv_bars("QQQ", _frame([("2026-10-07", 0.2, 0)])[["Open", "High", "Low", "Close"]],
                           source=SOURCE_IB_OPTION_IV)
    res.save_dividends("QQQ", [("2026-09-22", 0.5)], "YAHOO_DIVIDENDS", "2026-10-01T00:00:00")
    return path


def _con(path):
    return sqlite3.connect(path)


class TestMerge:
    def test_plan_counts_before_anything_is_written(self, db):
        with _con(db) as c:
            plan = {p["shared"]: p for p in (U.plan_table(c, r, s) for r, s in U._pairs(c))}
        dp = plan["daily_prices"]
        assert dp["research_rows"] == 3 and dp["to_insert"] == 1 and dp["overlap"] == 2
        assert dp["overlap_differs"] == 2            # volume / source differ on both overlapping dates
        assert plan["dividends"]["to_insert"] == 1

    def test_research_wins_on_overlap_live_only_rows_are_kept(self, db):
        with _con(db) as c:
            for r, s in U._pairs(c):
                U.merge_table(c, r, s)
            rows = {d: (cl, vol, src) for d, cl, vol, src in c.execute(
                "SELECT date, close, volume, source FROM daily_prices WHERE symbol='QQQ'")}
            assert rows["2026-10-06"] == (100.01, 1000, SOURCE_IB_TRADES_RTH)     # research replaced live
            assert rows["2026-10-08"][0] == 102.0                                    # research-only date added
            assert rows["2026-10-09"] == (103.0, 50, None)                           # live-only date kept untouched
            assert c.execute("SELECT COUNT(*) FROM daily_prices WHERE symbol='AAPL'").fetchone()[0] == 1
            assert c.execute("SELECT COUNT(*) FROM daily_iv WHERE symbol='VIX'").fetchone()[0] == 1  # live-only kept
            assert c.execute("SELECT COUNT(*) FROM daily_iv WHERE symbol='QQQ'").fetchone()[0] == 1
            assert c.execute("SELECT COUNT(*) FROM dividends").fetchone()[0] == 1

    def test_implied_vol_keeps_live_when_research_has_none_and_takes_research_when_it_has_one(self, db):
        with _con(db) as c:
            for r, s in U._pairs(c):
                U.merge_table(c, r, s)
            iv = {d: (v, src) for d, v, src in c.execute(
                "SELECT date, implied_vol, iv_source FROM daily_prices WHERE symbol='QQQ'")}
        assert iv["2026-10-06"] == (0.5, "LIVE")                      # research NULL -> live kept
        assert iv["2026-10-07"] == (0.2, SOURCE_IB_OPTION_IV)         # research has it -> research wins

    def test_merge_is_idempotent(self, db):
        with _con(db) as c:
            for r, s in U._pairs(c):
                U.merge_table(c, r, s)
            first = c.execute("SELECT COUNT(*), SUM(close), SUM(volume) FROM daily_prices").fetchone()
            for r, s in U._pairs(c):
                U.merge_table(c, r, s)
            assert c.execute("SELECT COUNT(*), SUM(close), SUM(volume) FROM daily_prices").fetchone() == first

    def test_empty_research_table_is_a_noop(self, db):
        with _con(db) as c:
            assert U.merge_table(c, "research_intraday_iv", "intraday_iv") == 0


class TestVerify:
    def test_ok_after_merge_and_not_ok_before(self, db):
        with _con(db) as c:
            assert U.verify(c)["ok"] is False                       # research rows are missing from the shared table
            for r, s in U._pairs(c):
                U.merge_table(c, r, s)
            rep = U.verify(c)
        assert rep["ok"] is True, rep["problems"]
        assert rep["closes"]["QQQ"]["sha256_equal"] is True

    def test_detects_a_row_removed_after_the_merge(self, db):
        with _con(db) as c:
            for r, s in U._pairs(c):
                U.merge_table(c, r, s)
            c.execute("DELETE FROM daily_prices WHERE symbol='QQQ' AND date='2026-10-07'")
            c.commit()
            rep = U.verify(c)
        assert rep["ok"] is False and any("missing" in p or "differs" in p for p in rep["problems"])

    def test_flags_an_unknown_source_tag(self, db):
        with _con(db) as c:
            for r, s in U._pairs(c):
                U.merge_table(c, r, s)
            c.execute("UPDATE daily_prices SET source='SOMETHING_ELSE' WHERE symbol='AAPL'")
            c.commit()
            assert any("SOMETHING_ELSE" in p for p in U.verify(c)["problems"])


class TestArchive:
    def test_archive_then_drop(self, db):
        with _con(db) as c:
            for r, s in U._pairs(c):
                U.merge_table(c, r, s)
            names = U.archive(c)
            tables = U._tables(c)
            assert "research_daily_prices" not in tables and "archive_w91_research_daily_prices" in tables
            assert "research_daily_prices" in names
            dropped = U.drop_archives(c)
            assert dropped and not any(t.startswith(U.ARCHIVE) for t in U._tables(c))
            assert c.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0] > 0   # shared data untouched


class TestCli:
    def test_dry_run_writes_nothing_but_a_report(self, db, tmp_path):
        before = _con(db).execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
        rc = U.main(["--db-path", str(db), "--report-dir", str(tmp_path)])
        assert rc == 0
        assert _con(db).execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0] == before
        rep = json.loads(next(tmp_path.glob("unify_history_*.json")).read_text())
        assert rep["mode"] == "dry-run" and rep["plan"]

    def test_apply_backs_up_merges_and_verifies(self, db, tmp_path):
        rc = U.main(["--db-path", str(db), "--apply", "--report-dir", str(tmp_path), "--backup-dir", str(tmp_path)])
        assert rc == 0
        assert list(tmp_path.glob("historical_pre_w91_*.db"))
        rep = json.loads(next(tmp_path.glob("unify_history_*.json")).read_text())
        assert rep["verify"]["ok"] is True and rep["mode"] == "apply"

    def test_archive_refuses_until_verification_passes(self, db, tmp_path):
        rc = U.main(["--db-path", str(db), "--archive", "--report-dir", str(tmp_path)])
        assert rc == 4
        assert "research_daily_prices" in U._tables(_con(db))

    def test_refuses_to_write_the_real_db_while_the_supervisor_runs(self, tmp_path, monkeypatch):
        (tmp_path / "data").mkdir()
        real = tmp_path / "data" / "historical.db"
        HistoricalDataStore(db_path=real)
        monkeypatch.setattr(U, "ROOT", tmp_path)
        monkeypatch.setattr(U, "_supervisor_running", lambda: True)
        assert U.main(["--db-path", str(real), "--apply", "--report-dir", str(tmp_path)]) == 3
        assert U.main(["--db-path", str(real), "--apply", "--allow-running", "--report-dir", str(tmp_path),
                       "--backup-dir", str(tmp_path)]) == 0

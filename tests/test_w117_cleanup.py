"""W117: scripts/clean_history_db.py -- list and delete rows of data/historical.db, reviewably.

Every test runs the real script code against a real sqlite file built by the production HistoricalDataStore (so the
schemas are the production ones). The bot guard is exercised with a really flocked file, a really touched heartbeat
and a process scan that is replaced only where the process table itself is the thing under test.
"""
from __future__ import annotations

import ast
import fcntl
import importlib.util
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from ait.data.historical import SOURCE_IB_TRADES_RTH, SOURCE_YAHOO_UNADJ, HistoricalDataStore
from ait.data.research_inputs import registered_tables

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("clean_history_mod", REPO / "scripts" / "clean_history_db.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["clean_history_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load()

ROWS = [  # symbol, date, source
    ("QQQ", "2026-10-05", SOURCE_IB_TRADES_RTH), ("QQQ", "2026-10-06", SOURCE_IB_TRADES_RTH),
    ("QQQ", "2026-10-07", SOURCE_IB_TRADES_RTH), ("QQQ", "2026-10-02", None),
    ("AAPL", "2026-10-05", None), ("AAPL", "2026-10-06", None), ("AAPL", "2026-10-07", SOURCE_YAHOO_UNADJ),
    ("SPY", "2026-10-05", "SOMETHING_ELSE"),
]


@pytest.fixture
def paths(tmp_path):
    db = tmp_path / "h.db"
    HistoricalDataStore(db_path=db)
    with sqlite3.connect(db) as c:
        c.executemany("INSERT INTO daily_prices (symbol, date, close, source) VALUES (?,?,100.0,?)", ROWS)
        c.execute("INSERT INTO intraday_prices (symbol, datetime, interval, close, source) "
                  "VALUES ('QQQ','2026-10-06 09:30:00','5min',1.0,?)", (SOURCE_IB_TRADES_RTH,))
        c.execute("INSERT INTO intraday_prices (symbol, datetime, interval, close, source) "
                  "VALUES ('QQQ','2026-10-07 09:30:00','5min',1.0,?)", (SOURCE_IB_TRADES_RTH,))
        c.execute("CREATE TABLE archive_w91_research_daily_prices AS SELECT * FROM daily_prices")
        c.execute("INSERT INTO option_spread_samples (symbol, sample_date, [right], strike, dte, iv, bid, ask, mid, "
                  "half_spread_pct) VALUES ('QQQ','2026-10-05','C',500,3,0.2,1.0,1.2,1.1,0.1)")
    return {"db": db, "backups": tmp_path / "backups", "reports": tmp_path / "reports"}


def _run(paths, *argv, capsys=None):
    full = ["--db-path", str(paths["db"]), "--report-dir", str(paths["reports"])]
    full += list(argv)
    if argv[0] == "delete":
        full += ["--backup-dir", str(paths["backups"])]
    return C.main(full)


def _count(db, table, where="1=1"):
    with sqlite3.connect(db) as c:
        return c.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


def _report(paths):
    files = sorted(paths["reports"].glob("clean_history_db_*.json"))
    assert files, "no report written"
    return json.loads(files[-1].read_text())


def _snapshot(db):
    with sqlite3.connect(db) as c:
        return {t: c.execute(f"SELECT * FROM {t} ORDER BY 1,2,3").fetchall()
                for (t,) in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


class TestList:
    def test_rows_by_table_symbol_and_tag_with_flags(self, paths, capsys):
        before = _snapshot(paths["db"])
        assert _run(paths, "list") == 0
        assert _snapshot(paths["db"]) == before
        rep = _report(paths)
        tables = {t["table"]: t for t in rep["tables"]}
        groups = {(g["symbol"], g["source"]): g for g in tables["daily_prices"]["groups"]}
        assert groups[("QQQ", SOURCE_IB_TRADES_RTH)]["rows"] == 3
        assert groups[("QQQ", SOURCE_IB_TRADES_RTH)]["first"] == "2026-10-05"
        assert groups[("QQQ", SOURCE_IB_TRADES_RTH)]["last"] == "2026-10-07"
        assert groups[("AAPL", "UNRECORDED")]["flags"] == ["UNRECORDED"]
        assert groups[("SPY", "SOMETHING_ELSE")]["flags"] == ["FOREIGN_SOURCE"]
        assert groups[("AAPL", SOURCE_YAHOO_UNADJ)]["flags"] == []
        assert "ARCHIVE_TABLE" in tables["archive_w91_research_daily_prices"]["flags"]
        assert tables["option_spread_samples"]["deletable"] is False
        assert tables["option_spread_samples"]["groups"] == []
        out = capsys.readouterr().out
        assert "daily_prices: 8 rows" in out and "FOREIGN_SOURCE" in out

    def test_a_table_without_a_source_column_is_listed_without_a_tag(self, paths):
        with sqlite3.connect(paths["db"]) as c:
            c.execute("INSERT INTO option_daily_bars (underlying, option_ticker, expiry, [right], strike, date, dte) "
                      "VALUES ('QQQ','O:QQQ261016C00500000','2026-10-16','C',500,'2026-10-05',11)")
        assert _run(paths, "list") == 0
        t = {t["table"]: t for t in _report(paths)["tables"]}["option_daily_bars"]
        assert t["groups"] == [{"symbol": "QQQ", "source": None, "rows": 1, "first": "2026-10-05",
                                "last": "2026-10-05", "flags": []}]
        assert _run(paths, "delete", "--table", "option_daily_bars", "--symbol", "QQQ", "--apply") == 0
        assert _count(paths["db"], "option_daily_bars") == 0
        assert _run(paths, "delete", "--table", "option_daily_bars", "--source", "X") == 2

    def test_the_list_opens_the_file_read_only(self, paths, monkeypatch):
        seen = []
        real = sqlite3.connect
        monkeypatch.setattr(C.sqlite3, "connect",
                            lambda target, *a, **k: seen.append(str(target)) or real(target, *a, **k))
        assert _run(paths, "list") == 0
        assert seen and all("mode=ro" in s for s in seen)


class TestDryRun:
    def test_changes_nothing_and_reports_exact_counts(self, paths, capsys):
        before = _snapshot(paths["db"])
        assert _run(paths, "delete", "--table", "daily_prices", "--symbol", "AAPL", "--source-null") == 0
        assert _snapshot(paths["db"]) == before
        assert not paths["backups"].exists()
        rep = _report(paths)
        assert rep["mode"] == "dry-run" and rep["rows_matched"] == 2 and rep["rows_kept"] == 6
        assert rep["table_rows"] == 8 and rep["table"] == "daily_prices"
        assert rep["breakdown"] == [{"symbol": "AAPL", "source": "UNRECORDED", "rows": 2,
                                     "first": "2026-10-05", "last": "2026-10-06"}]
        assert "dry run: nothing deleted" in capsys.readouterr().out

    def test_the_explicit_dry_run_flag_is_the_same(self, paths):
        assert _run(paths, "delete", "--table", "daily_prices", "--symbol", "QQQ", "--dry-run") == 0
        assert _count(paths["db"], "daily_prices") == 8


class TestApply:
    def test_deletes_exactly_the_selected_rows_and_the_backup_restores_them(self, paths):
        before = _snapshot(paths["db"])
        assert _run(paths, "delete", "--table", "daily_prices", "--symbol", "AAPL", "--source-null",
                    "--apply") == 0
        assert _count(paths["db"], "daily_prices") == 6
        assert _count(paths["db"], "daily_prices", "symbol='AAPL'") == 1                # the YAHOO row stays
        assert _count(paths["db"], "daily_prices", "symbol='AAPL' AND source IS NULL") == 0
        after = _snapshot(paths["db"])
        assert {t: r for t, r in after.items() if t != "daily_prices"} == \
               {t: r for t, r in before.items() if t != "daily_prices"}
        assert set(before["daily_prices"]) - set(after["daily_prices"]) == {
            r for r in before["daily_prices"] if r[0] == "AAPL" and r[-2] is None}
        rep = _report(paths)
        assert rep["mode"] == "apply" and rep["rows_deleted"] == 2 and rep["table_rows_after"] == 6
        backups = list(paths["backups"].glob("historical_pre_clean_*.db"))
        assert len(backups) == 1 and rep["backup"] == str(backups[0])
        assert _snapshot(backups[0]) == before                 # the backup restores every deleted row
        import shutil
        shutil.copy(backups[0], paths["db"])
        assert _snapshot(paths["db"]) == before

    @pytest.mark.parametrize("flags,expected", [
        (["--before", "2026-10-06"], 4),          # strictly before: 10-02, 10-05 (QQQ, AAPL, SPY)
        (["--after", "2026-10-06"], 2),           # strictly after: 10-07 (QQQ, AAPL)
        (["--before", "2026-10-07", "--after", "2026-10-05"], 2),      # 10-06 only
        (["--symbol", "QQQ", "--source", SOURCE_IB_TRADES_RTH], 3),
        (["--source", "SOMETHING_ELSE"], 1),
    ])
    def test_the_filters_combine_with_and_and_dates_are_exclusive(self, paths, flags, expected):
        assert _run(paths, "delete", "--table", "daily_prices", *flags, "--apply") == 0
        assert _count(paths["db"], "daily_prices") == 8 - expected

    def test_a_date_bound_compares_the_day_part_of_a_datetime_column(self, paths):
        assert _run(paths, "delete", "--table", "intraday_prices", "--before", "2026-10-07", "--apply") == 0
        assert _count(paths["db"], "intraday_prices") == 1
        assert _count(paths["db"], "intraday_prices", "datetime LIKE '2026-10-07%'") == 1

    def test_all_rows_empties_only_the_named_table(self, paths):
        assert _run(paths, "delete", "--table", "archive_w91_research_daily_prices", "--all-rows",
                    "--apply") == 0
        assert _count(paths["db"], "archive_w91_research_daily_prices") == 0
        assert _count(paths["db"], "daily_prices") == 8

    def test_a_selection_that_matches_nothing_deletes_nothing_and_takes_no_backup(self, paths):
        assert _run(paths, "delete", "--table", "daily_prices", "--symbol", "ZZZZ", "--apply") == 0
        assert _count(paths["db"], "daily_prices") == 8
        assert not paths["backups"].exists()

    def test_one_transaction_a_failure_midway_leaves_the_table_untouched(self, paths, monkeypatch):
        with sqlite3.connect(paths["db"]) as c:
            c.execute("CREATE TRIGGER boom BEFORE DELETE ON daily_prices WHEN old.symbol='SPY' "
                      "BEGIN SELECT RAISE(ABORT, 'no'); END")
        before = _snapshot(paths["db"])
        rc = _run(paths, "delete", "--table", "daily_prices", "--before", "2027-01-01", "--apply")
        assert rc == 1
        assert _snapshot(paths["db"]) == before

    def test_a_selection_that_changes_after_the_plan_aborts(self, paths):
        with pytest.raises(RuntimeError, match="selection changed"):
            C.apply_delete(paths["db"], "daily_prices", '"symbol" = ?', ["QQQ"], 99, paths["backups"])
        assert _count(paths["db"], "daily_prices") == 8


class TestRefusals:
    def test_an_unscoped_delete_is_refused_even_with_apply(self, paths, capsys):
        assert _run(paths, "delete", "--table", "daily_prices") == 3
        assert _run(paths, "delete", "--table", "daily_prices", "--apply") == 3
        assert "REFUSED" in capsys.readouterr().out
        assert _count(paths["db"], "daily_prices") == 8 and not paths["backups"].exists()

    def test_all_rows_cannot_be_combined_with_a_filter(self, paths):
        assert _run(paths, "delete", "--table", "daily_prices", "--all-rows", "--symbol", "QQQ",
                    "--apply") == 2
        assert _count(paths["db"], "daily_prices") == 8

    @pytest.mark.parametrize("table", ["option_spread_samples", "risk_free_rates", "sqlite_master",
                                       "daily_prices; DROP TABLE daily_prices", "daily_prices--",
                                       "daily_prices WHERE 1=1", '"daily_prices"', "research_daily_prices",
                                       "archive_w91_research_", "../h", "", "DAILY_PRICES"])
    def test_a_table_outside_the_allow_list_is_refused(self, paths, table):
        before = _snapshot(paths["db"])
        assert _run(paths, "delete", "--table", table, "--all-rows", "--apply") == 3
        assert _snapshot(paths["db"]) == before

    def test_an_allowed_table_missing_from_the_database_is_not_found(self, paths):
        assert _run(paths, "delete", "--table", "macro_series", "--all-rows") == 2
        assert _run(paths, "delete", "--table", "archive_w91_research_nothing", "--all-rows") == 2

    def test_the_allow_list_is_the_registry_plus_the_archives(self):
        for t in registered_tables():
            assert C.is_allowed(t)
        assert C.is_allowed("archive_w91_research_intraday_prices")
        assert not C.is_allowed("option_spread_samples") and not C.is_allowed("executions")

    def test_a_filter_the_table_cannot_take_is_a_bad_argument(self, paths):
        assert _run(paths, "delete", "--table", "daily_prices", "--source", "UNRECORDED") == 2
        assert _run(paths, "delete", "--table", "official_econ_calendar", "--symbol", "QQQ") == 2
        assert _run(paths, "delete", "--table", "daily_prices", "--source", "X", "--source-null") == 2
        assert _run(paths, "delete", "--table", "daily_prices", "--before", "not-a-date") == 2

    def test_a_missing_database_is_exit_2(self, paths):
        paths["db"].unlink()
        assert _run(paths, "list") == 2


class TestTheBotGuard:
    """The guard applies to the real data/historical.db only; a test points the script's notion of 'real' at the
    temporary database and its lock / heartbeat paths at temporary files."""

    @pytest.fixture
    def real(self, paths, tmp_path, monkeypatch):
        monkeypatch.setattr(C, "REAL_DB", paths["db"])
        monkeypatch.setattr(C, "LOCK_PATH", tmp_path / "orchestrator.lock")
        monkeypatch.setattr(C, "HEARTBEAT_PATH", tmp_path / "bot_heartbeat")
        monkeypatch.setattr(C, "_scan", lambda **kw: [])
        return paths

    def _apply(self, paths):
        return _run(paths, "delete", "--table", "daily_prices", "--symbol", "QQQ", "--apply")

    def test_it_passes_when_nothing_runs_and_deletes(self, real):
        assert self._apply(real) == 0
        assert _count(real["db"], "daily_prices", "symbol='QQQ'") == 0
        assert _report(real)["guard_checked"] is True

    def test_a_held_supervisor_lock_refuses(self, real, tmp_path, capsys):
        lock = tmp_path / "orchestrator.lock"
        lock.write_text("1234")
        fh = open(lock, "r+")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            assert self._apply(real) == 3
        finally:
            fh.close()
        assert "supervisor holds" in capsys.readouterr().out
        assert _count(real["db"], "daily_prices") == 8 and not real["backups"].exists()
        lock_free = self._apply(real)                              # released: the same call now goes through
        assert lock_free == 0

    def test_a_fresh_heartbeat_refuses_at_any_hour_and_a_stale_one_does_not(self, real, tmp_path):
        hb = tmp_path / "bot_heartbeat"
        hb.write_text("x")
        assert self._apply(real) == 3
        old = time.time() - C.HEARTBEAT_MAX_AGE_S - 60
        os.utime(hb, (old, old))
        assert self._apply(real) == 0

    @pytest.mark.parametrize("scan_result,code", [(None, 3), ([4242], 3), ([], 0)])
    def test_a_failed_or_positive_process_scan_refuses(self, real, monkeypatch, scan_result, code):
        monkeypatch.setattr(C, "_scan", lambda **kw: scan_result)
        assert self._apply(real) == code

    def test_the_scan_asks_for_the_bot_and_the_supervisor(self, real, monkeypatch):
        asked = []
        monkeypatch.setattr(C, "_scan", lambda **kw: asked.append(kw) or [])
        assert self._apply(real) == 0
        assert {"module": "ait.main"} in asked and {"script": "run_orchestrator.py"} in asked

    def test_a_second_scan_that_fails_refuses_even_if_the_first_was_clean(self, real, monkeypatch):
        monkeypatch.setattr(C, "_scan", lambda **kw: [] if "module" in kw else None)
        assert self._apply(real) == 3

    def test_an_unreadable_lock_refuses(self, real, tmp_path, monkeypatch):
        (tmp_path / "orchestrator.lock").write_text("1")
        os.chmod(tmp_path / "orchestrator.lock", 0o000)
        try:
            if os.access(tmp_path / "orchestrator.lock", os.R_OK):
                pytest.skip("running as a user that ignores file modes")
            assert self._apply(real) == 3
        finally:
            os.chmod(tmp_path / "orchestrator.lock", 0o644)

    def test_the_dry_run_and_list_never_need_the_guard(self, real, monkeypatch):
        monkeypatch.setattr(C, "_scan", lambda **kw: None)
        (real["db"].parent / "bot_heartbeat").write_text("x")
        assert _run(real, "list") == 0
        assert _run(real, "delete", "--table", "daily_prices", "--symbol", "QQQ") == 0
        assert _report(real)["guard_checked"] is False

    def test_there_is_no_way_to_override_the_guard(self):
        parser = C._parser()
        text = parser.format_help() + "".join(
            sp.format_help() for a in parser._actions if hasattr(a, "choices") and a.choices
            for sp in a.choices.values() if hasattr(sp, "format_help"))
        assert "allow-running" not in text and "force" not in text

    def test_a_copy_of_the_database_is_not_checked(self, paths, monkeypatch):
        monkeypatch.setattr(C, "_scan", lambda **kw: None)
        assert _run(paths, "delete", "--table", "daily_prices", "--symbol", "QQQ", "--apply") == 0


class TestReport:
    def test_the_report_names_what_was_selected_and_done(self, paths):
        assert _run(paths, "delete", "--table", "daily_prices", "--symbol", "QQQ", "--before", "2026-10-07",
                    "--source", SOURCE_IB_TRADES_RTH, "--apply") == 0
        rep = _report(paths)
        assert rep["db"] == str(paths["db"]) and rep["command"] == "delete" and rep["at"]
        assert rep["filters"] == {"symbol": "QQQ", "source": SOURCE_IB_TRADES_RTH, "source_null": False,
                                  "before": "2026-10-07", "after": None, "all_rows": False}
        assert (rep["rows_matched"], rep["rows_deleted"], rep["table_rows"], rep["table_rows_after"]) == (2, 2, 8, 6)
        assert rep["breakdown"][0]["symbol"] == "QQQ" and Path(rep["backup"]).is_file()
        assert "dotenv" not in json.dumps(rep).lower()


class TestNoOtherHistoryDeletes:
    """Class guard: the only code that deletes rows of a history table is scripts/clean_history_db.py (tests build
    and trim their own fixtures). A new `DELETE FROM <history table>` elsewhere fails here and should go through
    that tool, or be added to ALLOWED with the reason."""

    ALLOWED = {"scripts/clean_history_db.py"}

    @staticmethod
    def _history_tables(tmp_path_factory) -> set[str]:
        path = tmp_path_factory.mktemp("hist") / "h.db"
        HistoricalDataStore(db_path=path)
        with sqlite3.connect(path) as c:
            names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return names | {"macro_series"}

    @staticmethod
    def _sql_strings(path: Path):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                yield node.lineno, node.value
            elif isinstance(node, ast.JoinedStr):
                yield node.lineno, "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)

    def test_no_delete_from_a_history_table_outside_the_cleanup_tool(self, tmp_path_factory):
        tables = self._history_tables(tmp_path_factory)
        pat = re.compile(r"\bDELETE\s+FROM\s+([\"`\[]?[\w{}.]+)", re.IGNORECASE)
        offenders = []
        for root in ("src", "scripts"):
            for path in sorted((REPO / root).rglob("*.py")):
                rel = path.relative_to(REPO).as_posix()
                if rel in self.ALLOWED:
                    continue
                for line, text in self._sql_strings(path):
                    if rel == "src/ait/data/historical.py" and "WHERE event_date >=" in text:
                        # save_official_econ_calendar(replace_from=...): the calendar build drops the stored
                        # future rows and re-inserts the rebuilt ones in ONE transaction, so a release the agency
                        # moved does not leave its old date behind. A rebuild, not a cleanup.
                        continue
                    for m in pat.finditer(text):
                        name = m.group(1).strip("\"`[")
                        base = name.replace("research_", "", 1) if name.startswith("research_") else name
                        if "{" in name or name in tables or base in tables or name.startswith("archive_w91_"):
                            offenders.append(f"{rel}:{line}: DELETE FROM {name}")
        assert not offenders, "history rows are deleted only by scripts/clean_history_db.py:\n" + "\n".join(offenders)

    def test_the_guard_sees_a_planted_offender(self, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("def f(con, t):\n    con.execute('DELETE FROM daily_prices WHERE 1')\n"
                       "    con.execute(f'DELETE FROM {t}')\n")
        found = [t for _, text in self._sql_strings(bad) for t in re.findall(r"DELETE\s+FROM\s+(\S+)", text)]
        assert found == ["daily_prices", "{}"]

"""W38 (R26 F5) — SQL readers must bucket by the ET day, not the UTC day.

`w21-sqlite-date-normalises-aware-timestamps-to-utc-in-the-dashboard`.

SQLite's date functions convert an offset-bearing ISO string to UTC first and
leave a naive one alone. W21 made every stored trade timestamp aware, which
silently changed what seven dashboard queries mean:

    date('2026-09-13T21:05:00')        -> '2026-09-13'    the ET day
    date('2026-09-13T21:05:00-04:00')  -> '2026-09-14'    the UTC day
    strftime('%w', ...)                -> '0' vs '1'      Sunday vs Monday

So a close at or after 20:00 EDT landed in tomorrow's P&L row, tomorrow's
calendar cell and tomorrow's weekday bucket, while `master.py`'s LIKE counters
and every `parse_trade_ts` caller still called it today. Two irreconcilable
daily series over one table, and nothing raised.

W21 enumerated its readers by grepping `fromisoformat` and walking its AST.
That method is blind to SQL by construction, which is why this class survived
the wave that was supposed to find it — hence the class guard at the bottom.

These tests execute the REAL query fragments the dashboard runs (imported as
constants, never read as source text) against real SQLite.
"""

from __future__ import annotations

import ast
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from ait.dashboard.app import _ET_DAY_WINDOW_SQL, _ET_ENTRY_WEEKDAY_SQL

SRC = Path(__file__).resolve().parents[1] / "src"

#: A close at 21:05 ET — inside the 20:00-24:00 ET band where the ET day and
#: the UTC day disagree. 2026-09-13 is a Sunday; the UTC day is Monday.
LATE_ET_CLOSE = "2026-09-13T21:05:00-04:00"
ITS_ET_DAY = "2026-09-13"


@pytest.fixture
def con():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE trades (trade_id TEXT, entry_time TEXT, exit_time TEXT)")
    c.execute("INSERT INTO trades VALUES ('T1', ?, ?)", (LATE_ET_CLOSE, LATE_ET_CLOSE))
    c.commit()
    return c


class TestTheWindowBucketsByTheETDay:

    def test_a_late_et_close_falls_in_its_own_day(self, con):
        """The window the dashboard actually runs, on the real engine."""
        rows = con.execute(
            f"SELECT trade_id FROM trades WHERE {_ET_DAY_WINDOW_SQL}",
            (ITS_ET_DAY, ITS_ET_DAY),
        ).fetchall()
        assert rows == [("T1",)], (
            f"a 21:05 ET close is missing from its own ET day {ITS_ET_DAY}"
        )

    def test_it_does_not_leak_into_the_next_day(self, con):
        rows = con.execute(
            f"SELECT trade_id FROM trades WHERE {_ET_DAY_WINDOW_SQL}",
            ("2026-09-14", "2026-09-14"),
        ).fetchall()
        assert rows == [], "the close was counted on the following day too"

    def test_the_retired_expression_really_did_misbucket(self, con):
        """NEGATIVE CONTROL: without it, the two tests above prove nothing.

        Asserts the pre-W38 SQL puts the same row on the WRONG day — if SQLite
        ever stopped normalising to UTC, these tests would pass for a reason
        unrelated to the fix and should be re-derived rather than trusted.
        """
        old = "date(COALESCE(exit_time, entry_time)) BETWEEN ? AND ?"
        assert con.execute(
            f"SELECT trade_id FROM trades WHERE {old}", (ITS_ET_DAY, ITS_ET_DAY)
        ).fetchall() == [], "premise moved: date() no longer normalises to UTC"
        assert con.execute(
            f"SELECT trade_id FROM trades WHERE {old}", ("2026-09-14", "2026-09-14")
        ).fetchall() == [("T1",)]

    def test_a_naive_row_is_unaffected(self, con):
        """CONTROL: the fix must not move rows that predate the migration."""
        con.execute("INSERT INTO trades VALUES ('LEGACY', ?, ?)",
                    ("2026-09-13T21:05:00", "2026-09-13T21:05:00"))
        rows = con.execute(
            f"SELECT trade_id FROM trades WHERE {_ET_DAY_WINDOW_SQL}",
            (ITS_ET_DAY, ITS_ET_DAY),
        ).fetchall()
        assert sorted(rows) == [("LEGACY",), ("T1",)], (
            "aware and naive rows for the same ET instant no longer agree"
        )


class TestTheWeekdayBucket:

    def test_the_entry_weekday_is_the_et_weekday(self, con):
        got = con.execute(
            f"SELECT {_ET_ENTRY_WEEKDAY_SQL} FROM trades").fetchone()[0]
        expected = date.fromisoformat(ITS_ET_DAY).strftime("%w")
        assert got == expected, (
            f"weekday bucket is {got!r}, but {ITS_ET_DAY} is {expected!r} in ET"
        )

    def test_the_retired_expression_really_did_shift_the_weekday(self, con):
        """NEGATIVE CONTROL. substr alone yields no weekday, so this site
        needed strftime OVER substr rather than the substr used elsewhere."""
        old = con.execute(
            "SELECT strftime('%w', entry_time) FROM trades").fetchone()[0]
        expected = date.fromisoformat(ITS_ET_DAY).strftime("%w")
        assert old != expected, "premise moved: strftime no longer shifts"


class TestNoSqlReaderBucketsOnARawTimestamp:
    """CLASS GUARD — the check W21's `fromisoformat` walk could not make.

    Any SQL date function applied DIRECTLY to a migrated column reintroduces
    the defect. Scans string literals, because that is where SQL lives.
    """

    #: Columns W21 made timezone-aware. `executions.exec_time` is deliberately
    #: absent: it stores true UTC, a different convention.
    MIGRATED = ("entry_time", "exit_time", "claimed_at")
    FUNCS = ("date(", "strftime(", "julianday(", "datetime(")

    def _offenders(self) -> list[str]:
        hits = []
        for path in sorted(SRC.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                sql = node.value.lower()
                for func in self.FUNCS:
                    start = 0
                    while (i := sql.find(func, start)) != -1:
                        start = i + len(func)
                        arg = sql[start:start + 120]
                        # `substr(col,1,10)` inside the call is the fix, not a hit.
                        if "substr(" in arg.split(")")[0]:
                            continue
                        if any(c in arg.split(")")[0] for c in self.MIGRATED):
                            hits.append(
                                f"{path.relative_to(SRC).as_posix()}:{node.lineno} "
                                f"{func}...{arg.split(')')[0][:60]}")
        return sorted(set(hits))

    def test_no_module_buckets_on_a_raw_migrated_column(self):
        assert self._offenders() == [], (
            "SQL date functions applied to an aware timestamp bucket by the "
            "UTC day:\n  " + "\n  ".join(self._offenders())
        )

    def test_the_guard_actually_scans_the_tree(self):
        """Anti-vacuity: a broken glob would pass by finding nothing."""
        files = list(SRC.rglob("*.py"))
        assert len(files) > 50, f"the scan found only {len(files)} files"
        assert (SRC / "ait" / "dashboard" / "app.py") in files

    def test_the_guard_catches_the_pattern_it_is_for(self):
        """Anti-vacuity: prove the detector fires on the retired expression."""
        tree = ast.parse(
            'q = "SELECT x FROM trades WHERE '
            'date(COALESCE(exit_time, entry_time)) BETWEEN ? AND ?"'
        )
        found = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                sql = node.value.lower()
                i = sql.find("date(")
                arg = sql[i + 5:i + 125].split(")")[0]
                found = any(c in arg for c in self.MIGRATED) and "substr(" not in arg
        assert found, "the detector does not recognise the defect it exists for"

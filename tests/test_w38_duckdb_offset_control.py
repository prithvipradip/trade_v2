"""W38 — pin the DuckDB behaviour W21 cleared the analytics mirror on.

W21's entry (PLAN.md:3484-3488) says, verbatim:

    VERIFIED, NOT ASSUMED: the DuckDB mirror is unaffected. Its `entry_time` /
    `exit_time` are `TIMESTAMP` (not `TIMESTAMPTZ`) and DuckDB DROPS the offset
    on ingest — probed 09-16, an aware string and a naive one land as the
    identical naive value. Both sides of its hold-time subtraction stay naive
    ET wall-clock, so duckdb_analytics.py is consistent and deliberately
    untouched.

That is true — re-measured here on duckdb 1.5.2. But it was a ONE-TIME probe
with nothing standing behind it, and the safety rests entirely on library
behaviour this project does not control. Flipping that column to `TIMESTAMPTZ`,
or ingesting through a path that parses Python-side, would shift every hold
time and every date bucket in the analytics mirror by 4-5 hours, silently, with
no test anywhere to notice.

So the claim becomes executable. These tests do not assert that the current
behaviour is *desirable* — only that the allowlist entry W21 wrote still
describes reality.
"""

from __future__ import annotations

import ast
from pathlib import Path

import duckdb
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

AWARE = "2026-09-13T21:05:00-04:00"
NAIVE = "2026-09-13T21:05:00"


class TestDuckDBStillDropsTheOffset:

    def test_an_aware_and_a_naive_string_land_identically(self):
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t (id INT, ts TIMESTAMP)")
        con.execute("INSERT INTO t VALUES (1, ?), (2, ?)", [AWARE, NAIVE])
        rows = con.execute("SELECT ts FROM t ORDER BY id").fetchall()

        assert rows[0][0] == rows[1][0], (
            f"DuckDB now distinguishes the two: {rows[0][0]} vs {rows[1][0]}. "
            f"W21's allowlist entry for the analytics mirror no longer holds — "
            f"every hold time and date bucket in it has shifted."
        )
        assert rows[0][0].tzinfo is None, (
            "the mirror's timestamps are aware now, so the hold-time "
            "subtraction has one aware and one naive side"
        )

    def test_the_hold_time_subtraction_agrees_for_both_conventions(self):
        """The actual arithmetic duckdb_analytics does, on both row shapes."""
        con = duckdb.connect(":memory:")
        con.execute("CREATE TABLE t (id INT, entry_time TIMESTAMP, exit_time TIMESTAMP)")
        con.execute(
            "INSERT INTO t VALUES (1, ?, ?), (2, ?, ?)",
            ["2026-09-13T09:05:00-04:00", AWARE, "2026-09-13T09:05:00", NAIVE],
        )
        hours = [
            (e - s).total_seconds() / 3600
            for s, e in con.execute(
                "SELECT entry_time, exit_time FROM t ORDER BY id").fetchall()
        ]
        assert hours[0] == hours[1] == 12.0, (
            f"aware and naive rows produce different hold times: {hours}"
        )


class TestTheMirrorSchemaIsStillNaive:
    """If the column type changes, the control above stops being the question."""

    def test_the_trade_columns_are_timestamp_not_timestamptz(self):
        src = (SRC / "ait" / "monitoring" / "duckdb_analytics.py").read_text(
            encoding="utf-8")
        assert "TIMESTAMPTZ" not in src.upper(), (
            "duckdb_analytics declares a TIMESTAMPTZ column — W21's allowlist "
            "entry assumed TIMESTAMP, and offsets would now be preserved"
        )

    def test_the_module_parses_within_the_tree(self):
        """Anti-vacuity for the scan above."""
        path = SRC / "ait" / "monitoring" / "duckdb_analytics.py"
        assert path.exists()
        assert isinstance(ast.parse(path.read_text(encoding="utf-8")), ast.Module)

"""The ONE reader of the stored official economic calendar (``official_econ_calendar``).

Shared by live and research. ``scripts/build_official_calendar.py`` writes the table (through
``HistoricalDataStore.save_official_econ_calendar``); two consumers read it:

* the live ``EconomicCalendar()`` (macro-event entry guards), through ``read_official_rows`` on a READ-ONLY
  connection that never creates a file or a table, so the live process cannot alter the history store by
  looking at it;
* the research gate (``build_research_calendar``) and the research preflight
  (``ait.data.research_inputs``), through the summary helpers below on the preflight's own read-only
  connection.

Every raw SQL statement on the table outside the store (DDL, writer, research loader) lives in this module; a
test (``tests/test_w117_official_calendar.py``) fails when another module under ``src/`` names the table.
Pure otherwise: no network, no writes.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import quote

OFFICIAL_TABLE = "official_econ_calendar"
# the five engine event types (the same vocabulary as EventType and backtest.macro_event_types)
OFFICIAL_EVENT_TYPES: tuple[str, ...] = ("FOMC", "CPI", "NFP", "GDP", "PCE")


class OfficialCalendarUnavailable(RuntimeError):
    """The stored official calendar cannot be read (no file, no table, no rows, or a database error)."""


@dataclass(frozen=True)
class OfficialRow:
    """One stored official release: the reconciled date of one event type."""

    event_date: date
    event_time_et: str
    event_type: str
    source: str
    source_url: str
    fetched_at: str
    sha256: str


def table_name(table_prefix: str = "") -> str:
    return f"{table_prefix}{OFFICIAL_TABLE}"


def read_official_rows(db_path: "str | Path", table_prefix: str = "") -> list[OfficialRow]:
    """Every stored row, ascending by date, read through a READ-ONLY connection (``mode=ro``: a missing file
    is an error, never created). Raises ``OfficialCalendarUnavailable`` when the file, the table or the rows
    are missing or the database cannot be read; the caller decides the fail direction (live falls back to its
    declared list and alerts; research refuses)."""
    path = Path(db_path)
    if not path.is_file():
        raise OfficialCalendarUnavailable(f"history database not found: {path}")
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        try:
            raw = conn.execute(
                f"SELECT event_date, event_time_et, event_type, source, source_url, fetched_at, sha256 "
                f"FROM {table_name(table_prefix)} ORDER BY event_date, event_type").fetchall()
        finally:
            conn.close()
        rows = [OfficialRow(date.fromisoformat(r[0]), *r[1:]) for r in raw]
    except (sqlite3.Error, ValueError) as exc:
        raise OfficialCalendarUnavailable(f"{type(exc).__name__}: {exc}") from exc
    if not rows:
        raise OfficialCalendarUnavailable(f"{table_name(table_prefix)} has no rows")
    return rows


# ---------------------------------------------------------------------------------------------
# summaries for the research preflight (a connection is passed in; nothing is opened here)
# ---------------------------------------------------------------------------------------------

def span_summary(conn: sqlite3.Connection, table: str) -> "tuple[str | None, str | None, int]":
    """(first date, last date, rows) of the whole table."""
    first, last, n = conn.execute(f"SELECT MIN(event_date), MAX(event_date), COUNT(*) FROM {table}").fetchone()
    return first, last, int(n)


def type_summary(conn: sqlite3.Connection, table: str, event_type: str, start: date, end: date
                 ) -> "tuple[str | None, str | None, int, int]":
    """(first, last, rows, rows inside start..end) of one event type."""
    first, last, n = conn.execute(
        f"SELECT MIN(event_date), MAX(event_date), COUNT(*) FROM {table} WHERE event_type=?",
        (event_type,)).fetchone()
    inside = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE event_type=? AND event_date>=? AND event_date<=?",
        (event_type, str(start), str(end))).fetchone()[0]
    return first, last, int(n), int(inside)


def foreign_sources(conn: sqlite3.Connection, table: str, event_types: "tuple[str, ...]",
                    allowed: "tuple[str, ...]") -> "list[tuple[str, int]]":
    """[(source tag, rows)] of the given event types whose tag is not one of ``allowed``."""
    marks = ",".join("?" * len(event_types))
    amarks = ",".join("?" * len(allowed))
    return [(str(s), int(n)) for s, n in conn.execute(
        f"SELECT source, COUNT(*) FROM {table} WHERE event_type IN ({marks}) "
        f"AND source NOT IN ({amarks}) GROUP BY source", (*event_types, *allowed))]

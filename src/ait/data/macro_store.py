"""Stored macro series (the FRED yields and the dollar index) for research.

W116. The live bot and the daily retrain fetch these four FRED series over the network
(``ait.data.macro.MacroDataFetcher``); research used to pass none, so its nine macro features were constants while the
models that trade were trained and scored on real values. Research now reads them from the history database, stored by
``scripts/backfill_macro_series.py`` from the same FRED series, tagged ``FRED_CSV``.

The series are joined with ONE SESSION OF LAG: FRED publishes a day's value the next morning, so the value known at
the open of session D is the one observed at D-1 (the live bot's inference fetch has the same property). A table that
is missing or short is a refusal (``ResearchDataUnavailable``), never a constant.

The table is created by the backfill script, not by the history store's constructor (which the live process also
runs); the reader opens the database read-only.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from ait.data.macro import FRED_SERIES
from ait.data.research_inputs import (
    ResearchDataUnavailable, ResearchRequest, Window, check_research_inputs, sessions_between)

SOURCE_FRED_CSV = "FRED_CSV"
MACRO_TABLE = "macro_series"
CREATE_SQL = (f"CREATE TABLE IF NOT EXISTS {{table}} (series_id TEXT NOT NULL, date TEXT NOT NULL, value REAL, "
              f"source TEXT NOT NULL, fetched_at TEXT, PRIMARY KEY (series_id, date))")
# the FRED ids the macro features read, by the friendly names ``FeatureEngine._add_macro`` looks up
SERIES_IDS: tuple[str, ...] = tuple(FRED_SERIES.values())


def parse_fred_csv(csv_text: str) -> pd.Series:
    """A FRED ``fredgraph.csv`` body -> the observations as a float Series indexed by date ('.' = FRED's missing marker
    becomes NaN). Raises ValueError when the file has no date and value columns."""
    from io import StringIO
    df = pd.read_csv(StringIO(csv_text))
    date_col = next((c for c in df.columns if c.lower() in ("date", "observation_date")), None)
    value_col = next((c for c in df.columns if c != date_col), None)
    if date_col is None or value_col is None:
        raise ValueError(f"not a FRED csv: columns {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    return pd.to_numeric(df.set_index(date_col)[value_col], errors="coerce").astype(float)


def save_macro_series(db_path: "Path | str", series_id: str, observations: pd.Series, fetched_at: str,
                      table_prefix: str = "") -> int:
    """Upsert one series' non-null observations (index = observation date). Used by the backfill script only."""
    table = f"{table_prefix}{MACRO_TABLE}"
    rows = [(series_id, pd.Timestamp(d).date().isoformat(), float(v), SOURCE_FRED_CSV, fetched_at)
            for d, v in observations.dropna().items()]
    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_SQL.format(table=table))
        conn.executemany(f"INSERT OR REPLACE INTO {table} (series_id, date, value, source, fetched_at) "
                         f"VALUES (?,?,?,?,?)", rows)
    return len(rows)


def macro_request(window: Window, db_path: "Path | str", table_prefix: str = "") -> ResearchRequest:
    """A request that checks the four macro series only."""
    from ait.config.invariants_research import RESEARCH_MAX_MACRO_GAP_DAYS, RESEARCH_MAX_MACRO_TAIL_DAYS
    return ResearchRequest(window=window, symbols=(), db_path=db_path, max_null_iv_run=0, max_dividend_gap_days=0,
                           max_macro_gap_days=RESEARCH_MAX_MACRO_GAP_DAYS,
                           max_macro_tail_days=RESEARCH_MAX_MACRO_TAIL_DAYS, table_prefix=table_prefix,
                           needs_iv=False, macro_series=SERIES_IDS)


def lag_one_session(observations: pd.Series, window: Window) -> pd.Series:
    """The value known at each session's open: the newest observation dated STRICTLY BEFORE the session (FRED
    publishes a day's value the next morning), indexed by session date, for the sessions of ``window``. A FRED holiday
    simply carries the previous value; a session with no earlier observation is dropped."""
    obs = observations.dropna().sort_index()
    sessions = pd.DatetimeIndex(pd.to_datetime(sessions_between(window.start, window.end)))
    pos = obs.index.searchsorted(sessions, side="left") - 1
    keep = pos >= 0
    return pd.Series(obs.to_numpy()[pos[keep]], index=sessions[keep], dtype=float)


def load_macro_series(db_path: "Path | str", window: Window, table_prefix: str = "") -> dict[str, pd.Series]:
    """{friendly name: lagged series} over ``window`` for ``FeatureEngine``'s ``market_context["macros"]``.

    Raises ``ResearchDataUnavailable`` (with the backfill command) when the stored series do not cover the window."""
    report = check_research_inputs(macro_request(window, db_path, table_prefix))
    if not report.ok:
        raise ResearchDataUnavailable(report.gaps, context="macro series")
    table = f"{table_prefix}{MACRO_TABLE}"
    # a month of look-back in front of the window so the first session still has a previous observation
    lo = (pd.Timestamp(window.start) - pd.Timedelta(days=31)).date().isoformat()
    out: dict[str, pd.Series] = {}
    with sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True) as conn:
        for friendly, series_id in FRED_SERIES.items():
            rows = conn.execute(f"SELECT date, value FROM {table} WHERE series_id=? AND date>=? AND date<=? "
                                f"AND value IS NOT NULL ORDER BY date", (series_id, lo, str(window.end))).fetchall()
            obs = pd.Series({pd.Timestamp(d): v for d, v in rows}, dtype=float)
            out[friendly] = lag_one_session(obs, window).rename(series_id)
    return out

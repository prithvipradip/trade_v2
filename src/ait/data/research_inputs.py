"""The research data contract: what a study needs, whether the store has it, and what to run when it does not.

W116. Research must never substitute or mix data sources. When an input a study needs is missing, the study
REFUSES TO START and says exactly what is missing and which command fixes it; it does not fall back to Yahoo,
to resampled 5-minute bars, to a hard-coded calendar or to a modelled stand-in. This module holds

* the types of that refusal (``DataGap``, ``ResearchDataUnavailable``),
* the ONE rule for "which sessions does this run need" (``required_window``), used by the research loaders, the
  walk-forward, the optimizer and the preflight alike, so the preflight can never check a different window than
  the run loads,
* the registry of inputs and their checks (added with the preflight).

Pure on purpose: resolved config values, a database path and a table prefix come in as ARGUMENTS and a read-only
connection goes to the history database. It imports nothing from the broker, the strategies or the engine, and it
never reads the research overlay (nothing under ``src/ait`` may know it; see the W68 guard in tests).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

# Calendar days of slack in front of a request that names only ``days``: holidays and a short first week must not
# make an otherwise complete window fail. The research loaders have always queried ``days + 30``.
WINDOW_BUFFER_DAYS = 30
# Look-back used to find the previous session around a weekend or holiday run (the longest NYSE closure is 4 days).
_SESSION_LOOKBACK_DAYS = 14


@dataclass(frozen=True)
class DataGap:
    """One missing or unusable input: what was needed, what was found, and the command that fixes it."""

    input: str                 # registry name, e.g. "daily_prices"
    symbol: str | None         # None for an input that is not per-symbol
    required: str              # what the run needs, in words ("every NYSE session 2024-01-02..2026-10-30")
    found: str                 # what the store holds
    remedy: str                # the command (or the config decision) that closes the gap

    def line(self) -> str:
        who = f"{self.input}[{self.symbol}]" if self.symbol else self.input
        return f"{who}: need {self.required}; found {self.found}. Fix: {self.remedy}"


class ResearchDataUnavailable(RuntimeError):
    """A study cannot start: one or more inputs are missing. ``gaps`` lists every one, not just the first."""

    def __init__(self, gaps: "list[DataGap] | tuple[DataGap, ...]", context: str = "") -> None:
        self.gaps = tuple(gaps)
        self.context = context
        head = f"research data unavailable{f' ({context})' if context else ''}: {len(self.gaps)} gap(s)"
        super().__init__("\n".join([head, *(f"  - {g.line()}" for g in self.gaps)]))


@dataclass(frozen=True)
class Window:
    """The sessions a run needs: ``start`` and ``end`` are calendar dates, both inclusive."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"research window ends before it starts: {self.start}..{self.end}")

    @property
    def span_days(self) -> int:
        return max(1, (self.end - self.start).days)


@lru_cache(maxsize=1)
def _calendar():
    import pandas_market_calendars as mcal
    return mcal.get_calendar("NYSE")


def sessions_between(start: date, end: date) -> list[date]:
    """Every NYSE session from ``start`` to ``end`` inclusive (the library calendar, no hard-coded list)."""
    if end < start:
        return []
    return [d.date() for d in _calendar().schedule(start_date=start, end_date=end).index]


def last_session_on_or_before(day: date) -> date:
    """The latest NYSE session that is ``day`` or earlier."""
    found = sessions_between(day - timedelta(days=_SESSION_LOOKBACK_DAYS), day)
    if not found:
        raise ValueError(f"no NYSE session within {_SESSION_LOOKBACK_DAYS} days before {day}")
    return found[-1]


def last_completed_session(today: date) -> date:
    """The latest session STRICTLY before ``today``.

    Today's bar is never required: the loaders drop the current session as partial (a daily bar stored mid-session
    is incomplete), so a run on a Saturday needs Friday and a run on a Monday morning needs Friday."""
    return last_session_on_or_before(today - timedelta(days=1))


def window_start(days: int, anchor: date) -> date:
    """First calendar day of a request that names ``days`` and no start date, counted back from ``anchor``."""
    return anchor - timedelta(days=int(days) + WINDOW_BUFFER_DAYS)


def required_window(*, today: date, days: int | None = None, start_date: date | None = None,
                    end_date: date | None = None) -> Window:
    """The window a run needs, from the same three values the loaders take.

    * end: ``end_date`` when the run is pinned, else the last completed session; never later than the last
      completed session (a pin in the future cannot be satisfied), and snapped back to a session.
    * start: ``start_date`` when given, else ``days`` back from the run's own end: the pin when there is one, else
      today (the loaders have always counted a days-only request back from today; the extra days in front of it
      absorb a short first week).
    """
    cap = last_completed_session(today)
    end = last_session_on_or_before(min(end_date, cap)) if end_date is not None else cap
    if start_date is None:
        if days is None:
            raise ValueError("required_window needs start_date or days")
        start_date = window_start(days, end_date if end_date is not None else today)
    return Window(start_date, end)


def prices_request(symbol: str, window: Window, db_path: "Path | str", table_prefix: str = "") -> "ResearchRequest":
    """A request that checks one symbol's daily bars only (what the loaders verify before they read)."""
    return ResearchRequest(window=window, symbols=(symbol,), db_path=db_path, max_null_iv_run=0,
                           max_dividend_gap_days=0, table_prefix=table_prefix, needs_iv=False)


def index_request(db_symbol: str, window: Window, db_path: "Path | str", table_prefix: str = "") -> "ResearchRequest":
    """A request that checks one stored index series (a VIX-family member) only."""
    return ResearchRequest(window=window, symbols=(), db_path=db_path, max_null_iv_run=0, max_dividend_gap_days=0,
                           table_prefix=table_prefix, vix_symbols=(db_symbol,), needs_iv=False)


@dataclass
class PreflightReport:
    """The outcome of a preflight: the window checked, every input that was checked, every gap, and the
    declared, counted facts the run record stamps (``notes``: e.g. the sessions that will be no-entry days)."""

    window: Window
    checked: list[str] = field(default_factory=list)
    gaps: list[DataGap] = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.gaps

    def as_dict(self) -> dict:
        return {"window": [str(self.window.start), str(self.window.end)], "ok": self.ok,
                "checked": list(self.checked), "notes": self.notes,
                "gaps": [{"input": g.input, "symbol": g.symbol, "required": g.required, "found": g.found,
                          "remedy": g.remedy} for g in self.gaps]}


@dataclass(frozen=True)
class ResearchRequest:
    """Everything the checks need, resolved by the caller from the run's settings (this module reads no config).

    ``context_window`` is the longer span the VIX / SPY context is loaded over (it starts before ``window`` for the
    feature warm-up). ``symbols`` are the traded symbols; ``context_symbols`` are loaded for prices only."""

    window: Window
    symbols: tuple[str, ...]
    db_path: "Path | str"
    max_null_iv_run: int
    max_dividend_gap_days: int
    max_event_gap_days: int = 0
    max_macro_gap_days: int = 0
    max_macro_tail_days: int = 0
    table_prefix: str = ""
    context_window: "Window | None" = None
    vix_window: "Window | None" = None
    context_symbols: tuple[str, ...] = ()
    vix_symbols: tuple[str, ...] = ()
    needs_iv: bool = True
    ex_div_gate: bool = False
    macro_gate: bool = False
    macro_event_types: tuple[str, ...] = ()
    earnings_gate: bool = False
    earnings_exempt: tuple[str, ...] = ()
    friction_per_leg: bool = False
    friction_bands: bool = False
    intraday: bool = False
    intraday_min_coverage: float = 1.0
    surface: bool = False
    surface_min_coverage: float = 1.0
    macro_series: tuple[str, ...] = ()          # FRED series ids the macro features read (empty = not needed)


# --------------------------------------------------------------------------- helpers

_BACKFILL = "python scripts/backfill_historical_data.py"


def _years(req: ResearchRequest, win: "Window | None" = None) -> int:
    w = win or req.window
    from ait.utils.time import today_et
    return max(1, -(-(today_et() - w.start).days // 365) + 1)


def _ro(db_path: "Path | str") -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _span(missing: "list[date]") -> str:
    return f"{len(missing)} session(s) missing, first {missing[0]}, last {missing[-1]}"


def _no_table(name: str, table: str, symbol: "str | None", required: str, remedy: str) -> DataGap:
    return DataGap(name, symbol, required, f"table {table} does not exist", remedy)


def _daily_gaps(conn, req, notes, symbols: tuple[str, ...], win: Window, table: str, symbol_col: str,
                date_col: str, value_col: str, tag_col: str, allowed_tag: str, name: str,
                remedy_for) -> list[DataGap]:
    """Every NYSE session in ``win`` has a non-null ``value_col`` row for each symbol, all tagged ``allowed_tag``."""
    if not symbols:
        return []
    if not _has_table(conn, table):
        return [_no_table(name, table, s, f"{table} rows for {win.start}..{win.end}", remedy_for(s)) for s in symbols]
    need = sessions_between(win.start, win.end)
    out = []
    for sym in symbols:
        rows = conn.execute(
            f"SELECT {date_col}, {tag_col} FROM {table} WHERE {symbol_col}=? AND {date_col}>=? AND {date_col}<=? "
            f"AND {value_col} IS NOT NULL", (sym, str(win.start), str(win.end))).fetchall()
        have = {date.fromisoformat(d) for d, _ in rows}
        missing = [d for d in need if d not in have]
        required = f"every NYSE session {win.start}..{win.end} ({len(need)})"
        if missing:
            out.append(DataGap(name, sym, required, f"{len(have)} of {len(need)}; {_span(missing)}", remedy_for(sym)))
        foreign: dict = {}
        for _, tag in rows:
            if tag != allowed_tag:
                foreign[tag or "UNRECORDED"] = foreign.get(tag or "UNRECORDED", 0) + 1
        if foreign:
            out.append(DataGap(name, sym, f"only rows tagged {allowed_tag}",
                               "rows tagged " + ", ".join(f"{t} ({n})" for t, n in sorted(foreign.items())),
                               remedy_for(sym)))
    return out


# --------------------------------------------------------------------------- the checks

def _check_prices(conn, req, notes) -> list[DataGap]:
    from ait.data.historical import SOURCE_IB_TRADES_RTH
    t = f"{req.table_prefix}daily_prices"
    out = []
    # the traded symbols over the window, the context symbols over the context window (prices only)
    for symbols, win in ((req.symbols, req.window),
                         (tuple(s for s in req.context_symbols if s not in req.symbols),
                          req.context_window or req.window)):
        out += _daily_gaps(conn, req, notes, symbols, win, t, "symbol", "date", "close", "source",
                           SOURCE_IB_TRADES_RTH, "daily_prices",
                           lambda s, w=win: f"{_BACKFILL} --symbols {s} --mode daily --years {_years(req, w)}")
    return out


def _check_iv(conn, req, notes) -> list[DataGap]:
    t = f"{req.table_prefix}daily_prices"
    if not _has_table(conn, t):
        return []                                                  # reported by the prices check
    need = sessions_between(req.window.start, req.window.end)
    out, no_entry = [], {}
    for sym in req.symbols:
        rows = conn.execute(f"SELECT date FROM {t} WHERE symbol=? AND date>=? AND date<=? AND implied_vol IS NOT NULL",
                            (sym, str(req.window.start), str(req.window.end))).fetchall()
        have = {date.fromisoformat(d) for (d,) in rows}
        remedy = f"{_BACKFILL} --symbols {sym} --mode iv --years {_years(req)}"
        required = (f"stored implied vol on every session {req.window.start}..{req.window.end}, "
                    f"gaps of at most {req.max_null_iv_run} consecutive session(s)")
        if not have:
            out.append(DataGap("implied_vol", sym, required, "no stored implied vol in the window", remedy))
            continue
        missing = [d for d in need if d not in have]
        run = longest = 0
        for d in need:
            run = run + 1 if d not in have else 0
            longest = max(longest, run)
        if longest > req.max_null_iv_run:
            out.append(DataGap("implied_vol", sym, required,
                               f"a run of {longest} consecutive session(s) without it ({_span(missing)})", remedy))
        elif missing:
            no_entry[sym] = [str(d) for d in missing]
    if no_entry:
        notes["iv_no_entry_sessions"] = no_entry
    return out


def _check_vix(conn, req, notes) -> list[DataGap]:
    from ait.data.historical import SOURCE_IB_VIX_INDEX
    win = req.vix_window or req.context_window or req.window
    return _daily_gaps(conn, req, notes, req.vix_symbols, win, f"{req.table_prefix}daily_iv", "symbol", "date",
                       "iv_close", "source", SOURCE_IB_VIX_INDEX, "vix_family",
                       lambda s: f"{_BACKFILL} --mode vix --symbols {s} --years {_years(req, win)}")


def _check_dividends(conn, req, notes) -> list[DataGap]:
    from ait.data.historical import SOURCE_YAHOO_DIVIDENDS
    t = f"{req.table_prefix}dividends"
    out = []
    for sym in req.symbols:
        remedy = f"python scripts/backfill_dividends.py --symbols {sym}"
        required = (f"stored dividends for {sym}, the last ex-date within {req.max_dividend_gap_days} days of "
                    f"{req.window.end}, tagged {SOURCE_YAHOO_DIVIDENDS}")
        if not _has_table(conn, t):
            out.append(_no_table("dividends", t, sym, required, remedy))
            continue
        row = conn.execute(f"SELECT MIN(ex_date), MAX(ex_date), COUNT(*) FROM {t} WHERE symbol=?", (sym,)).fetchone()
        if not row or not row[2]:
            out.append(DataGap("dividends", sym, required, "no rows", remedy))
            continue
        last = date.fromisoformat(row[1])
        if (req.window.end - last).days > req.max_dividend_gap_days:
            out.append(DataGap("dividends", sym, required,
                               f"{row[2]} rows, the last ex-date is {last} ({(req.window.end - last).days} days "
                               f"before the window ends)", remedy))
        tags = {tg for (tg,) in conn.execute(f"SELECT DISTINCT source FROM {t} WHERE symbol=?", (sym,))}
        if tags - {SOURCE_YAHOO_DIVIDENDS}:
            out.append(DataGap("dividends", sym, f"rows tagged {SOURCE_YAHOO_DIVIDENDS}",
                               "rows tagged " + ", ".join(sorted(str(x) for x in tags)), remedy))
    return out


def _check_econ(conn, req, notes) -> list[DataGap]:
    """The macro gate reads the stored OFFICIAL calendar (Track D): the Fed / BLS / BEA pages (FRED where it fills
    a gap), never the Investing.com export (that stays stored as the reconciliation baseline and for its numeric
    actual / forecast / previous columns)."""
    from ait.data.historical import OFFICIAL_CALENDAR_SOURCES
    from ait.data.official_calendar import foreign_sources, span_summary, table_name, type_summary
    t = table_name(req.table_prefix)
    remedy = "python scripts/build_official_calendar.py --apply"
    tags = ", ".join(OFFICIAL_CALENDAR_SOURCES)
    if not _has_table(conn, t):
        return [_no_table("official_econ_calendar", t, None, "the stored official economic calendar", remedy)]
    out = []
    first, _last, n = span_summary(conn, t)
    # the calendar must reach back to the window's start: an earlier session has no gate at all
    if not n or date.fromisoformat(first) > req.window.start:
        out.append(DataGap("official_econ_calendar", None,
                           f"an official calendar that starts on or before {req.window.start}",
                           f"{n} rows starting {first}" if n else "no rows", remedy + f" --since {req.window.start}"))
    for etype in req.macro_event_types:
        required = (f"{etype} events inside {req.window.start}..{req.window.end}, the last no more than "
                    f"{req.max_event_gap_days} days before the window ends, tagged one of {tags}")
        tfirst, tlast, tn, inside = type_summary(conn, t, etype, req.window.start, req.window.end)
        if not tn:
            out.append(DataGap("official_econ_calendar", etype, required, "no rows", remedy))
            continue
        newest = date.fromisoformat(tlast)
        if not inside or (req.window.end - newest).days > req.max_event_gap_days:
            out.append(DataGap("official_econ_calendar", etype, required,
                               f"{tn} rows spanning {tfirst}..{tlast} ({inside} inside the window)", remedy))
    foreign = foreign_sources(conn, t, tuple(req.macro_event_types), OFFICIAL_CALENDAR_SOURCES)
    if foreign:
        out.append(DataGap("official_econ_calendar", None, f"rows tagged one of {tags}",
                           "rows tagged " + ", ".join(f"{s} ({k})" for s, k in foreign), remedy))
    return out


def _check_earnings(conn, req, notes) -> list[DataGap]:
    t = f"{req.table_prefix}earnings_schedule"
    out, exempt = [], []
    for sym in req.symbols:
        if sym in req.earnings_exempt:
            exempt.append(sym)
            continue
        remedy = (f"python scripts/build_research_calendars.py --what earnings, or list {sym} in "
                  f"backtest.earnings_exempt_symbols if it has no earnings")
        required = f"stored earnings dates for {sym} covering {req.window.start}..{req.window.end}"
        if not _has_table(conn, t):
            out.append(_no_table("earnings", t, sym, required, remedy))
            continue
        row = conn.execute(f"SELECT MIN(event_session), MAX(event_session), COUNT(*) FROM {t} WHERE company_key=?",
                           (sym,)).fetchone()
        if not row or not row[2]:
            out.append(DataGap("earnings", sym, required, "no rows", remedy))
        elif date.fromisoformat(row[0]) > req.window.start or date.fromisoformat(row[1]) < req.window.end:
            out.append(DataGap("earnings", sym, required, f"{row[2]} rows spanning {row[0]}..{row[1]}", remedy))
    if exempt:
        notes["earnings_exempt"] = exempt
    return out


def _check_friction(conn, req, notes) -> list[DataGap]:
    out, no_bands = [], []
    for sym in req.symbols:
        remedy = (f"python scripts/calibrate_option_spreads.py --source ib-bars --symbols {sym} "
                  f"--dte-targets 10 21 35 --client-id 91")
        # the engine reads the friction tables WITHOUT the research prefix
        if not _has_table(conn, "option_leg_spread_params") or not conn.execute(
                "SELECT 1 FROM option_leg_spread_params WHERE symbol=?", (sym,)).fetchone():
            out.append(DataGap("friction_params", sym, f"a per-leg friction model for {sym}", "none stored", remedy))
        if req.friction_bands and (not _has_table(conn, "option_leg_spread_band_params") or not conn.execute(
                "SELECT 1 FROM option_leg_spread_band_params WHERE symbol=?", (sym,)).fetchone()):
            no_bands.append(sym)
    if no_bands:
        notes["friction_bands_extrapolated"] = no_bands        # declared: the all-DTE model, floored, is used
    return out


def _check_intraday(conn, req, notes) -> list[DataGap]:
    t = f"{req.table_prefix}intraday_prices"
    need = len(sessions_between(req.window.start, req.window.end))
    out = []
    for sym in req.symbols:
        remedy = (f"{_BACKFILL} --symbols {sym} --mode intraday, or set backtest.intraday_features to false "
                  f"(the study then runs on daily bars only and says so)")
        required = f"5-minute bars on at least {req.intraday_min_coverage:.0%} of the {need} sessions"
        if not _has_table(conn, t):
            out.append(_no_table("intraday_prices", t, sym, required, remedy))
            continue
        days = conn.execute(
            f"SELECT COUNT(DISTINCT substr(datetime,1,10)) FROM {t} WHERE symbol=? AND interval='5m' "
            f"AND substr(datetime,1,10)>=? AND substr(datetime,1,10)<=?",
            (sym, str(req.window.start), str(req.window.end))).fetchone()[0]
        if days < req.intraday_min_coverage * need:
            out.append(DataGap("intraday_prices", sym, required, f"{days} sessions ({days / need:.0%})", remedy))
    return out


def _check_option_bars(conn, req, notes) -> list[DataGap]:
    need = len(sessions_between(req.window.start, req.window.end))
    out = []
    sessions = sessions_between(req.window.start, req.window.end)
    for sym in req.symbols:
        # option bars exist for QQQ only; the other symbols fall to the second remedy
        remedy = (f"python scripts/backfill_option_history.py --symbol {sym}, or set backtest.pricing_source to "
                  f"bs_legacy (the legacy single-IV pricing)")
        required = f"option bars on at least {req.surface_min_coverage:.0%} of the {need} sessions"
        if not _has_table(conn, "option_daily_bars"):                 # the surface reads it WITHOUT the prefix
            out.append(_no_table("option_daily_bars", "option_daily_bars", sym, required, remedy))
            continue
        stored = {r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM option_daily_bars WHERE underlying=? AND date>=? AND date<=?",
            (sym, str(req.window.start), str(req.window.end)))}
        days = len(stored)
        if days < req.surface_min_coverage * need:
            missing = [d for d in sessions if str(d) not in stored]
            found = f"{days} sessions" + (f", first missing {missing[0]}" if missing else "")
            out.append(DataGap("option_daily_bars", sym, required, found, remedy))
    return out


def _check_macro_series(conn, req, notes) -> list[DataGap]:
    from datetime import timedelta
    t = f"{req.table_prefix}macro_series"
    win = req.vix_window or req.context_window or req.window
    remedy = "python scripts/backfill_macro_series.py"
    if not _has_table(conn, t):
        return [_no_table("macro_series", t, None, "the stored FRED macro series", remedy)]
    out = []
    lo = str(win.start - timedelta(days=31))
    for sid in req.macro_series:
        required = (f"{sid} observations from on or before {win.start} to within {req.max_macro_tail_days} days of "
                    f"{win.end}, never more than {req.max_macro_gap_days} days apart, tagged FRED_CSV")
        rows = conn.execute(f"SELECT date, source FROM {t} WHERE series_id=? AND date>=? AND date<=? "
                            f"AND value IS NOT NULL ORDER BY date", (sid, lo, str(win.end))).fetchall()
        if not rows:
            out.append(DataGap("macro_series", sid, required, "no rows", remedy))
            continue
        dates = [date.fromisoformat(d) for d, _ in rows]
        problems = []
        if dates[0] > win.start:
            problems.append(f"the first observation is {dates[0]}")
        if (win.end - dates[-1]).days > req.max_macro_tail_days:
            problems.append(f"the last observation is {dates[-1]}")
        inside = [d for d in dates if d >= win.start] or dates[-1:]
        gaps = [(a, b) for a, b in zip(inside, inside[1:]) if (b - a).days > req.max_macro_gap_days]
        if gaps:
            problems.append(f"{len(gaps)} gap(s) longer than {req.max_macro_gap_days} days, first {gaps[0][0]}..{gaps[0][1]}")
        foreign = sorted({s or "UNRECORDED" for _, s in rows} - {"FRED_CSV"})
        if foreign:
            problems.append("rows tagged " + ", ".join(foreign))
        if problems:
            out.append(DataGap("macro_series", sid, required, "; ".join(problems), remedy))
    return out


@dataclass(frozen=True)
class ResearchInput:
    """One registered input: the tables it reads (and whether the research prefix applies to them), when a
    request needs it, and its check."""

    name: str
    tables: tuple[tuple[str, bool], ...]            # (table, uses_prefix)
    applies: Callable[[ResearchRequest], bool]
    check: Callable[..., list]


REGISTRY: tuple[ResearchInput, ...] = (
    ResearchInput("daily_prices", (("daily_prices", True),), lambda r: bool(r.symbols), _check_prices),
    ResearchInput("implied_vol", (("daily_prices", True),), lambda r: r.needs_iv and bool(r.symbols), _check_iv),
    ResearchInput("vix_family", (("daily_iv", True),), lambda r: bool(r.vix_symbols), _check_vix),
    ResearchInput("dividends", (("dividends", True),), lambda r: r.ex_div_gate, _check_dividends),
    ResearchInput("official_econ_calendar", (("official_econ_calendar", True),),
                  lambda r: r.macro_gate and bool(r.macro_event_types), _check_econ),
    ResearchInput("earnings", (("earnings_schedule", True),), lambda r: r.earnings_gate, _check_earnings),
    ResearchInput("friction_params", (("option_leg_spread_params", False), ("option_leg_spread_band_params", False)),
                  lambda r: r.friction_per_leg, _check_friction),
    ResearchInput("intraday_prices", (("intraday_prices", True),), lambda r: r.intraday, _check_intraday),
    ResearchInput("option_daily_bars", (("option_daily_bars", False),), lambda r: r.surface, _check_option_bars),
    ResearchInput("macro_series", (("macro_series", True),), lambda r: bool(r.macro_series), _check_macro_series),
)


def registered_tables() -> set[str]:
    """Every (unprefixed) table name some registered input reads."""
    return {table for entry in REGISTRY for table, _ in entry.tables}


def check_research_inputs(req: ResearchRequest) -> PreflightReport:
    """Check every input the request needs against the history database, read-only; never raises on a gap."""
    report = PreflightReport(req.window)
    if not Path(req.db_path).is_file():
        report.gaps.append(DataGap("history_database", None, f"the history database {req.db_path}", "file not found",
                                   f"{_BACKFILL} --symbols {' '.join(req.symbols) or '<symbols>'} --mode research"))
        return report
    conn = _ro(req.db_path)
    try:
        for entry in REGISTRY:
            if not entry.applies(req):
                continue
            report.checked.append(entry.name)
            report.gaps += entry.check(conn, req, report.notes)
    finally:
        conn.close()
    return report


def preflight_or_raise(req: ResearchRequest, context: str = "") -> PreflightReport:
    """``check_research_inputs``, but a gap raises ``ResearchDataUnavailable`` listing every gap."""
    report = check_research_inputs(req)
    if not report.ok:
        raise ResearchDataUnavailable(report.gaps, context)
    return report


def format_report(report: PreflightReport) -> str:
    """The preflight as text: the window, what was checked, then every gap (or OK) with its fix command."""
    lines = [f"research data preflight: window {report.window.start}..{report.window.end}",
             "checked: " + (", ".join(report.checked) or "nothing")]
    for key, value in sorted(report.notes.items()):
        lines.append(f"declared: {key} = {value}")
    if report.ok:
        lines.append("OK: every input the study needs is stored")
    else:
        lines.append(f"REFUSED: {len(report.gaps)} gap(s)")
        lines += [f"  - {g.line()}" for g in report.gaps]
    return "\n".join(lines)

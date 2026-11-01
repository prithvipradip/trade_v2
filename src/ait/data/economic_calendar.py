"""Economic calendar — prevents holding iron condors through major macro events.

Major economic releases (FOMC, CPI, NFP, GDP, PCE) cause massive IV crush
and gap risk that can blow through iron condor wings in minutes. This module
lets the bot:
  - Skip opening new positions on event days and the day before
  - Close or avoid iron condors that would span these events

Track D: ``EconomicCalendar()`` (live) reads the stored OFFICIAL calendar, the
``official_econ_calendar`` table that ``scripts/build_official_calendar.py`` fills from the
Fed / BLS / BEA release pages (FRED as the cross-check) — the same table the research gate reads. The
hard-coded lists below are the DECLARED FALLBACK: used when the table is missing, unreadable, incomplete or
stale (log.critical + the orchestrator's calendar page), and to extend a type past the table's last date
(`_fallback_tail`). They carry the official dates verified on 2026-10-31; what could not be verified is
named in UNVERIFIED_FALLBACK.

Sources of the fallback lists:
  - FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm
  - CPI/NFP: bls.gov/schedule/news_release/ (+ the release archive for past dates)
  - GDP/PCE: bea.gov/news/schedule/full (the 2027 schedule is not published yet)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from pathlib import Path

from ait.utils.logging import get_logger
from ait.utils.time import today_et

log = get_logger("data.economic_calendar")


class EventType(Enum):
    """Categories of market-moving economic events."""

    FOMC = "FOMC"          # Federal Open Market Committee rate decision
    CPI = "CPI"            # Consumer Price Index (inflation)
    NFP = "NFP"            # Non-Farm Payrolls (jobs report)
    GDP = "GDP"            # Gross Domestic Product (advance estimates only)
    PCE = "PCE"            # Personal Consumption Expenditures (Fed's preferred inflation gauge)


@dataclass(frozen=True)
class EconomicEvent:
    """A single scheduled economic release."""

    date: date
    event_type: EventType
    description: str


# ---------------------------------------------------------------------------
# 2026 hardcoded calendar
# ---------------------------------------------------------------------------
# FOMC: statement released on the *second* day of each two-day meeting at 2:00 PM ET.
# We use the second day (decision day) as the event date.
_FOMC_2026 = [
    date(2026, 1, 28),   # Jan 27-28
    date(2026, 3, 18),   # Mar 17-18
    date(2026, 4, 29),   # Apr 28-29
    date(2026, 6, 17),   # Jun 16-17
    date(2026, 7, 29),   # Jul 28-29
    date(2026, 9, 16),   # Sep 15-16
    date(2026, 10, 28),  # Oct 27-28
    date(2026, 12, 9),   # Dec 8-9
]

# CPI: released at 8:30 AM ET, typically second week of the month.
# Source: BLS schedule (bls.gov/schedule/news_release/cpi.htm) and release archive. Track D: the Jan-2026
# CPI is 02-13 (BLS moved it after the 2025 shutdown; this list said 02-11).
_CPI_2026 = [
    date(2026, 1, 13),   # Dec 2025 data
    date(2026, 2, 13),   # Jan 2026 data (BLS reschedule)
    date(2026, 3, 11),   # Feb 2026 data
    date(2026, 4, 10),   # Mar 2026 data
    date(2026, 5, 12),   # Apr 2026 data
    date(2026, 6, 10),   # May 2026 data
    date(2026, 7, 14),   # Jun 2026 data
    date(2026, 8, 12),   # Jul 2026 data
    date(2026, 9, 11),   # Aug 2026 data
    date(2026, 10, 14),  # Sep 2026 data
    date(2026, 11, 10),  # Oct 2026 data
    date(2026, 12, 10),  # Nov 2026 data
]

# NFP (Employment Situation): released at 8:30 AM ET, typically first Friday.
# Source: BLS schedule (bls.gov/schedule/news_release/empsit.htm) and release archive. Track D: the
# Jan-2026 report is 02-11 (BLS rescheduled it after the 2025 shutdown; this list said 02-06).
_NFP_2026 = [
    date(2026, 1, 9),    # Dec 2025 data
    date(2026, 2, 11),   # Jan 2026 data (BLS reschedule)
    date(2026, 3, 6),    # Feb 2026 data
    date(2026, 4, 3),    # Mar 2026 data
    date(2026, 5, 8),    # Apr 2026 data
    date(2026, 6, 5),    # May 2026 data
    date(2026, 7, 2),    # Jun 2026 data
    date(2026, 8, 7),    # Jul 2026 data
    date(2026, 9, 4),    # Aug 2026 data
    date(2026, 10, 2),   # Sep 2026 data
    date(2026, 11, 6),   # Oct 2026 data
    date(2026, 12, 4),   # Nov 2026 data
]

# GDP (Advance Estimate only — the first read is the big market mover).
# Second and third estimates rarely move markets significantly.
# Source: BEA schedule (bea.gov/news/schedule/full). Track D: Q4-2025 advance was delayed to 02-20
# (the shutdown); it was missing from this list.
_GDP_2026 = [
    date(2026, 2, 20),   # Q4 2025 advance (shutdown-delayed)
    date(2026, 4, 30),   # Q1 2026 advance
    date(2026, 7, 30),   # Q2 2026 advance
    date(2026, 10, 29),  # Q3 2026 advance
    # Q4 2026 advance is in Jan 2027 (_GDP_2027; the BEA 2027 schedule is not published yet)
]

# PCE (Personal Income and Outlays — contains core PCE price index).
# Source: BEA schedule (bea.gov/news/schedule/full), ET release time 8:30 AM except 01-22 (10:00 AM).
# Track D: BEA's 2026 dates follow the post-shutdown catch-up, NOT the "last Friday" pattern this list
# used to assume (9 of its 12 dates were wrong). The two shutdown-combined releases are one event each.
# Some dates coincide with GDP releases (BEA bundles them).
_PCE_2026 = [
    date(2026, 1, 22),   # Oct + Nov 2025 data (combined release)
    date(2026, 2, 20),   # Dec 2025 data (same day as the Q4 2025 GDP advance)
    date(2026, 3, 13),   # Jan 2026 data
    date(2026, 4, 9),    # Feb 2026 data
    date(2026, 4, 30),   # Mar 2026 data (same day as GDP advance Q1)
    date(2026, 5, 28),   # Apr 2026 data
    date(2026, 6, 25),   # May 2026 data
    date(2026, 7, 30),   # Jun 2026 data (same day as GDP advance Q2)
    date(2026, 8, 26),   # Jul 2026 data
    date(2026, 9, 30),   # Aug 2026 data
    date(2026, 10, 29),  # Sep 2026 data (same day as GDP advance Q3)
    date(2026, 11, 25),  # Oct 2026 data
    date(2026, 12, 23),  # Nov 2026 data
]

# ---------------------------------------------------------------------------
# 2027 H1 calendar (R16 #10)
# ---------------------------------------------------------------------------
# The 2026 tables end 2026-12-23 (PCE), so from 12-24 every macro guard went
# silently blind while standard 14-45 DTE entries span the first 2027 NFP/CPI.
#
# VERIFIED against the official pages on 2026-10-31 (Track D): FOMC (the Fed's calendar page), CPI and NFP
# (the BLS schedule pages). NOT YET PUBLISHED by BEA, so still APPROXIMATE (the agency's TYPICAL pattern,
# and BEA's 2026 dates showed that pattern is unreliable): _GDP_2027 and _PCE_2027, named in
# UNVERIFIED_FALLBACK. Replace them from bea.gov/news/schedule/full once BEA publishes the 2027 schedule
# (scripts/build_official_calendar.py then loads them into the table, which wins over these lists).
_FOMC_2027 = [
    date(2027, 1, 27),   # Jan 26-27 (tentative)
    date(2027, 3, 17),   # Mar 16-17 (tentative)
    date(2027, 4, 28),   # Apr 27-28 (tentative)
    date(2027, 6, 9),    # Jun 8-9 (tentative; this list said 06-16)
]

_CPI_2027 = [
    date(2027, 1, 13),   # Dec 2026 data
    date(2027, 2, 11),   # Jan 2027 data (BLS schedule; this list said 02-10)
    date(2027, 3, 10),   # Feb 2027 data
    date(2027, 4, 13),   # Mar 2027 data
    date(2027, 5, 12),   # Apr 2027 data
    date(2027, 6, 10),   # May 2027 data
]

_NFP_2027 = [
    date(2027, 1, 8),    # Dec 2026 data (BLS: Jan 1 = holiday Friday -> 2nd Friday)
    date(2027, 2, 5),    # Jan 2027 data (first Friday)
    date(2027, 3, 5),    # Feb 2027 data (first Friday)
    date(2027, 4, 2),    # Mar 2027 data (first Friday)
    date(2027, 5, 7),    # Apr 2027 data (first Friday)
    date(2027, 6, 4),    # May 2027 data (first Friday)
]

_GDP_2027 = [
    date(2027, 1, 28),   # Q4 2026 advance (UNVERIFIED: BEA has not published its 2027 schedule)
    date(2027, 4, 29),   # Q1 2027 advance (UNVERIFIED)
]

_PCE_2027 = [
    date(2027, 1, 29),   # Dec 2026 data (last Friday)
    date(2027, 2, 26),   # Jan 2027 data (last Friday)
    date(2027, 3, 26),   # Feb 2027 data (last Friday; Good Friday — markets
                         # closed, BEA typically still releases; VERIFY)
    date(2027, 4, 30),   # Mar 2027 data (last Friday)
    date(2027, 5, 28),   # Apr 2027 data (last Friday)
    date(2027, 6, 25),   # May 2027 data (last Friday)
]

# The fallback dates that are NOT confirmed by an official page (2026-10-31): tests/test_w117 pins that every
# other fallback date equals the official schedule. Drop a list from here only when BEA's 2027 schedule
# confirms it (the builder then stores the real dates and the table wins over these anyway).
UNVERIFIED_FALLBACK: tuple[str, ...] = ("_GDP_2027", "_PCE_2027")


def _build_event_list() -> list[EconomicEvent]:
    """Construct the full event list from hardcoded dates."""
    events: list[EconomicEvent] = []

    for d in _FOMC_2026 + _FOMC_2027:
        events.append(EconomicEvent(d, EventType.FOMC, f"FOMC rate decision {d}"))
    for d in _CPI_2026 + _CPI_2027:
        events.append(EconomicEvent(d, EventType.CPI, f"CPI release {d}"))
    for d in _NFP_2026 + _NFP_2027:
        events.append(EconomicEvent(d, EventType.NFP, f"NFP jobs report {d}"))
    for d in _GDP_2026 + _GDP_2027:
        events.append(EconomicEvent(d, EventType.GDP, f"GDP advance estimate {d}"))
    for d in _PCE_2026 + _PCE_2027:
        events.append(EconomicEvent(d, EventType.PCE, f"PCE release {d}"))

    events.sort(key=lambda e: e.date)
    return events


# Pre-built for fast lookups
_ALL_EVENTS = _build_event_list()
_EVENT_DATES: frozenset[date] = frozenset(e.date for e in _ALL_EVENTS)

_DESCRIPTIONS = {
    EventType.FOMC: "FOMC rate decision", EventType.CPI: "CPI release", EventType.NFP: "NFP jobs report",
    EventType.GDP: "GDP advance estimate", EventType.PCE: "PCE release"}
# The oldest the table's newest fetch may be before the live calendar stops trusting it (days). Dates move
# (BLS and BEA rescheduled releases after the 2025 shutdown), so a calendar nobody has rebuilt for a quarter is
# treated as unverified; the builder is run by the operator (RUNBOOK "Economic calendar").
TABLE_MAX_AGE_DAYS = 90


def default_live_db_path() -> Path:
    """The history database the live bot opens: the SAME default as ``HistoricalDataStore()`` (bot/orchestrator
    builds it with no argument), resolved here so the orchestrator needs no change."""
    from ait.data.historical import DB_PATH
    return DB_PATH


def _fallback_tail(events: list[EconomicEvent]) -> list[EconomicEvent]:
    """Hard-coded events of a type that fall AFTER the table's last event of that type. The table is
    authoritative up to its own horizon (BEA publishes a year at a time, BLS about fourteen months); beyond it
    an APPROXIMATE date still blocks entries on the likeliest day, which is the safer direction than leaving
    the guard blind (an extra skipped day costs a day, a missed release costs a gap through a condor)."""
    last: dict[EventType, date] = {}
    for e in events:
        last[e.event_type] = max(e.date, last.get(e.event_type, date.min))
    return [e for e in _ALL_EVENTS if e.date > last.get(e.event_type, date.min)]


def _load_live_table(db_path: Path, today: date) -> "tuple[list[EconomicEvent] | None, str]":
    """(events, reason): the stored official calendar as events, or (None, why it cannot be used).

    FAIL DIRECTION (stated, not incidental): any problem reading or trusting the table returns None and the
    caller uses the declared hard-coded fallback. That is never LESS protective than before this table existed
    (the guards still see every macro event on their list) and it never blocks an exit: the calendar only
    gates ENTRIES, and nothing here raises. Unusable means: no file / table / rows, a type with no rows
    (an incomplete build), a row tagged by a source that is not an official one, or a newest fetch older than
    TABLE_MAX_AGE_DAYS."""
    from ait.data.historical import OFFICIAL_CALENDAR_SOURCES
    from ait.data.official_calendar import OFFICIAL_EVENT_TYPES, OfficialCalendarUnavailable, read_official_rows
    try:
        rows = read_official_rows(db_path)
    except OfficialCalendarUnavailable as exc:
        return None, f"unreadable: {exc}"
    except Exception as exc:  # noqa: BLE001 - the live calendar must never raise into the entry loop
        return None, f"unreadable: {type(exc).__name__}: {exc}"
    missing = sorted(set(OFFICIAL_EVENT_TYPES) - {r.event_type for r in rows})
    if missing:
        return None, f"incomplete: no rows for {', '.join(missing)}"
    foreign = sorted({r.source for r in rows} - set(OFFICIAL_CALENDAR_SOURCES))
    if foreign:
        return None, f"rows tagged by a non-official source: {', '.join(foreign)}"
    try:
        newest = max(date.fromisoformat(r.fetched_at[:10]) for r in rows)
    except ValueError:
        return None, "unreadable fetched_at timestamps"
    age = (today - newest).days
    if age > TABLE_MAX_AGE_DAYS:
        return None, f"stale: last fetched {newest} ({age} days ago, limit {TABLE_MAX_AGE_DAYS})"
    events = [EconomicEvent(r.event_date, EventType[r.event_type], f"{_DESCRIPTIONS[EventType[r.event_type]]} "
                                                                   f"{r.event_date}") for r in rows]
    return events, ""


class EconomicCalendar:
    """Provides macro event awareness for the trading bot.

    The bot should NOT hold iron condors (or open new positions) through
    major economic events. IV crush + gap risk makes this negative EV.

    Usage:
        cal = EconomicCalendar()
        if cal.should_skip_trading(today_et()):
            log.info("skipping trades — macro event window")
            return
    """

    # R16 #10: fire the staleness alarm while there is still runway. The old
    # A10 alarm only triggered when a check was made with a year-2027 date —
    # i.e. AFTER the guards had already gone blind (last 2026 event was
    # 12-23; entries from 12-24 sailed unprotected into Jan-2027 NFP/CPI).
    # Now it fires as soon as fewer than STALENESS_ALARM_DAYS of known future
    # events remain, which is before a fresh 14-45 DTE entry can outlive the
    # calendar.
    STALENESS_ALARM_DAYS = 30

    def __init__(self, events: "list[EconomicEvent] | None" = None, *,
                 db_path: "Path | None" = None) -> None:
        # W65: `events=None` (live, the default) reads the stored official calendar; research passes a
        # calendar built from the same table (build_research_calendar) so past windows are gated on what
        # actually happened. Track D: `db_path` (default: the bot's own historical.db, resolved here) names
        # the store the live calendar reads, READ-ONLY.
        self._degraded = False
        if events is not None:
            self._source = "explicit"
            self._events = sorted(events, key=lambda e: e.date)
            self._event_dates = frozenset(e.date for e in self._events)
        else:
            path = Path(db_path) if db_path is not None else default_live_db_path()
            loaded, reason = _load_live_table(path, today_et())
            if loaded is None:
                self._source = "fallback"
                self._degraded = True
                self._events = _ALL_EVENTS
                self._event_dates = _EVENT_DATES
                self._alert_fallback(path, reason)
            else:
                tail = _fallback_tail(loaded)
                self._source = "table"
                self._events = sorted(loaded + tail, key=lambda e: e.date)
                self._event_dates = frozenset(e.date for e in self._events)
                log.info("economic_calendar_loaded", source="official_econ_calendar", path=str(path),
                         rows=len(loaded), first=str(min(e.date for e in loaded)),
                         last=str(max(e.date for e in loaded)),
                         fallback_tail={t.value: sum(e.event_type is t for e in tail)
                                        for t in EventType if any(e.event_type is t for e in tail)})
        self._last_event_date: date = (
            max(e.date for e in self._events) if self._events else date.min
        )

    _exhausted_warned = False
    _fallback_alerted = False

    @property
    def source(self) -> str:
        """Where the events came from: 'table' (official_econ_calendar), 'fallback' (the declared
        hard-coded lists, the table being unusable) or 'explicit' (a list the caller passed: research)."""
        return self._source

    @property
    def exhausted_warned(self) -> bool:
        """R17: True once `_warn_if_nearly_exhausted` has logged the once-per-process staleness alarm, OR
        when this calendar is running on the hard-coded fallback (Track D). Polled by the orchestrator
        (whose callers are async, unlike this class's) to actually page a human -- the log.critical() alone
        never reached Telegram. The orchestrator's page text names the exhaustion; the log line
        economic_calendar_table_unusable carries the actual reason."""
        return EconomicCalendar._exhausted_warned or self._degraded

    @classmethod
    def _alert_fallback(cls, path: Path, reason: str) -> None:
        """log.critical (once per process) that the live calendar is on the hard-coded fallback."""
        if cls._fallback_alerted:
            return
        cls._fallback_alerted = True
        log.critical(
            "economic_calendar_table_unusable",
            path=str(path), reason=reason, fallback="hardcoded_lists",
            msg="the stored official economic calendar cannot be used; macro-event guards run on the "
                "hard-coded fallback lists (verified 2026-10-31, 2027 GDP/PCE unverified). Rebuild it: "
                "python scripts/build_official_calendar.py --apply (RUNBOOK 'Economic calendar'), then "
                "restart the bot.")

    def _warn_if_nearly_exhausted(self, check_date: date) -> None:
        """log.critical (once per process) when the calendar has fewer than STALENESS_ALARM_DAYS of
        future events left."""
        if EconomicCalendar._exhausted_warned:
            return
        remaining = (self._last_event_date - check_date).days
        if remaining < self.STALENESS_ALARM_DAYS:
            EconomicCalendar._exhausted_warned = True
            log.critical(
                "economic_calendar_nearly_exhausted",
                last_known_event=str(self._last_event_date),
                days_of_coverage_left=remaining,
                source=self._source,
                msg="the economic calendar has <"
                    f"{self.STALENESS_ALARM_DAYS} days of future events -- "
                    "macro-event guards are about to go BLIND. Rebuild the stored calendar "
                    "(python scripts/build_official_calendar.py --apply) and, past the horizon the "
                    "agencies have published, extend the _FOMC/_CPI/_NFP/_GDP/_PCE fallback tables "
                    "(2027 GDP/PCE are UNVERIFIED approximations).",
            )

    def is_event_day(self, check_date: date | None = None) -> bool:
        """Return True if the given date has a major economic release."""
        check_date = check_date or today_et()
        self._warn_if_nearly_exhausted(check_date)
        return check_date in self._event_dates

    def is_pre_event_day(self, check_date: date | None = None) -> bool:
        """Return True if *tomorrow* (next trading day) has a major release.

        For simplicity we check the next calendar day. Since most releases
        are on weekdays, and markets are closed on weekends, this is
        conservative (we might skip a Friday before a Monday event, which
        is exactly what we want — no weekend gap risk into an event).
        """
        check_date = check_date or today_et()
        tomorrow = check_date + timedelta(days=1)
        # Also check Monday if today is Friday
        if check_date.weekday() == 4:  # Friday
            monday = check_date + timedelta(days=3)
            return tomorrow in self._event_dates or monday in self._event_dates
        return tomorrow in self._event_dates

    def should_skip_trading(self, check_date: date | None = None) -> bool:
        """Return True if the bot should avoid opening new positions.

        Skips on:
        - Event days (release already happened or happening, IV crush in play)
        - Pre-event days (holding overnight into the event is the risk)
        """
        check_date = check_date or today_et()
        skip = self.is_event_day(check_date) or self.is_pre_event_day(check_date)
        if skip:
            events = self.get_events_on(check_date)
            upcoming = self.get_events_on(check_date + timedelta(days=1))
            if check_date.weekday() == 4:
                upcoming += self.get_events_on(check_date + timedelta(days=3))
            all_relevant = events + upcoming
            names = ", ".join(f"{e.event_type.value}" for e in all_relevant)
            log.info(
                "economic_calendar_skip",
                date=str(check_date),
                events=names or "pre-event window",
            )
        return skip

    def get_events_on(self, check_date: date | None = None) -> list[EconomicEvent]:
        """Get all economic events on a specific date."""
        check_date = check_date or today_et()
        return [e for e in self._events if e.date == check_date]

    def get_upcoming_events(self, days: int = 7) -> list[EconomicEvent]:
        """Get all events in the next N calendar days (inclusive of today)."""
        today = today_et()
        cutoff = today + timedelta(days=days)
        return [e for e in self._events if today <= e.date <= cutoff]

    def days_until_next_event(self, check_date: date | None = None) -> int | None:
        """Return days until the next macro event, or None if no future events.

        NOTE (R16 #10): a None return means the calendar is EXHAUSTED, not
        that no event is coming — consumers (orchestrator blackout, engine
        macro gate) treat None as "no event", so the staleness alarm below is
        the guard rail against silently trading blind.
        """
        check_date = check_date or today_et()
        self._warn_if_nearly_exhausted(check_date)
        for event in self._events:
            if event.date >= check_date:
                return (event.date - check_date).days
        return None

    def is_safe_to_hold_through_expiry(
        self, entry_date: date, expiry_date: date
    ) -> bool:
        """Check if there are NO macro events between entry and expiry.

        Use this to decide whether an iron condor can safely be held
        to expiration without macro event risk.
        """
        self._warn_if_nearly_exhausted(entry_date)
        for event in self._events:
            if entry_date < event.date <= expiry_date:
                return False
        return True

    def get_events_between(
        self, start: date, end: date
    ) -> list[EconomicEvent]:
        """Get all events between start and end (inclusive)."""
        return [e for e in self._events if start <= e.date <= end]


# ---------------------------------------------------------------------------
# W65: Investing.com CSV classification (the reconciliation baseline; no longer the gate source)
# ---------------------------------------------------------------------------
# The ONE definition of which CSV rows are the five engine event types, shared by
# scripts/load_econ_calendar_csv.py (ingest), scripts/build_official_calendar.py (the reconciliation
# baseline) and the tests. Names are the exact Investing.com event strings of the US releases
# (country == "United States"). Track D: the research GATE reads official_econ_calendar instead.
ENGINE_EVENT_NAMES: dict[str, str] = {
    "Fed Interest Rate Decision": "FOMC",
    "U.S. Consumer Price Index (CPI) MoM": "CPI",
    "U.S. Nonfarm Payrolls": "NFP",
    "U.S. Core PCE Price Index MoM": "PCE",
    "U.S. Gross Domestic Product (GDP) QoQ": "GDP",
}
_GDP_STAGE_GAP_DAYS = 60   # advance/second/third estimates of one quarter are ~30 days apart


def derive_engine_event_types(df):
    """Return a Series (aligned to df.index) of FOMC/CPI/NFP/PCE/GDP or None.

    Combined releases (two reference periods on one date, e.g. Oct+Nov NFP after the
    2025 shutdown) simply map to the same event date. For GDP only the ADVANCE
    estimate counts (the engine's hardcoded list is "advance estimates only"): the
    first release of each reference-period group, where a new group starts when a
    release of that reference period comes > 60 days after the previous one."""
    import pandas as pd

    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    us = df["country"].eq("United States") & df["type"].eq("event")
    for name, etype in ENGINE_EVENT_NAMES.items():
        if etype == "GDP":
            continue
        out[us & df["event"].eq(name)] = etype
    # W106: a CPI release with no MoM headline row (2025-12-18, after the shutdown)
    # is still a CPI date: tag its first headline-CPI row of any kind.
    cpi_rows = us & df["event"].astype(str).str.startswith("U.S. Consumer Price Index (CPI)")
    tagged_dates = set(df.loc[out.eq("CPI"), "event_date"])
    untagged = df[cpi_rows & ~df["event_date"].isin(tagged_dates)].drop_duplicates("event_date")
    out.loc[untagged.index] = "CPI"
    gdp = df[us & df["event"].eq("U.S. Gross Domestic Product (GDP) QoQ")].copy()
    if not gdp.empty:
        gdp["_d"] = pd.to_datetime(gdp["event_date"])
        gdp = gdp.sort_values("_d")
        for _, grp in gdp.groupby(gdp["reference_period"].fillna("")):
            prev = None
            for idx, d in zip(grp.index, grp["_d"]):
                if prev is None or (d - prev).days > _GDP_STAGE_GAP_DAYS:
                    out[idx] = "GDP"
                prev = d
    return out


def build_research_calendar(db_path, table_prefix: str, event_types) -> "EconomicCalendar | None":
    """EconomicCalendar from `{prefix}official_econ_calendar`, or None when the table is empty.

    Track D: the research gate reads the SAME stored official calendar the live bot reads (the Fed / BLS /
    BEA pages, FRED where it filled a gap), no longer the Investing.com export. NO hard-coded fallback: a
    table without rows returns None and the caller (strict research) refuses. One EconomicEvent per
    (date, type): the engine gates on dates, so a combined release is a single event."""
    from ait.data.historical import HistoricalDataStore

    store = HistoricalDataStore(db_path=db_path, table_prefix=table_prefix)
    df = store.load_official_econ_calendar(event_types=list(event_types))
    if df.empty:
        return None
    events = []
    for d, t in sorted({(date.fromisoformat(r.event_date), r.event_type) for r in df.itertuples()}):
        events.append(EconomicEvent(d, EventType[t], f"{t} (research calendar)"))
    return EconomicCalendar(events=events)

"""Economic calendar — prevents holding iron condors through major macro events.

Major economic releases (FOMC, CPI, NFP, GDP, PCE) cause massive IV crush
and gap risk that can blow through iron condor wings in minutes. This module
provides a hardcoded 2026 calendar of known release dates so the bot can:
  - Skip opening new positions on event days and the day before
  - Close or avoid iron condors that would span these events

Sources:
  - FOMC: federalreserve.gov/monetarypolicy/fomccalendars.htm
  - CPI/NFP: bls.gov/schedule/news_release/
  - GDP/PCE: bea.gov/news/schedule
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from ait.utils.logging import get_logger

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
# Source: BLS schedule (bls.gov/schedule/news_release/cpi.htm)
_CPI_2026 = [
    date(2026, 1, 13),   # Dec 2025 data
    date(2026, 2, 11),   # Jan 2026 data
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
# Source: BLS schedule (bls.gov/schedule/news_release/empsit.htm)
_NFP_2026 = [
    date(2026, 1, 9),    # Dec 2025 data
    date(2026, 2, 6),    # Jan 2026 data
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
# Source: BEA schedule (bea.gov/news/schedule)
_GDP_2026 = [
    date(2026, 4, 30),   # Q1 2026 advance
    date(2026, 7, 30),   # Q2 2026 advance
    date(2026, 10, 29),  # Q3 2026 advance
    # Q4 2026 advance is in Jan 2027
]

# PCE (Personal Income and Outlays — contains core PCE price index).
# Released at 8:30 AM ET, typically last week of the month.
# Source: BEA schedule (bea.gov/news/schedule)
# Note: Some dates coincide with GDP releases (BEA bundles them).
_PCE_2026 = [
    date(2026, 1, 30),   # Dec 2025 data
    date(2026, 2, 27),   # Jan 2026 data
    date(2026, 3, 27),   # Feb 2026 data
    date(2026, 4, 30),   # Mar 2026 data (same day as GDP advance Q1)
    date(2026, 5, 29),   # Apr 2026 data
    date(2026, 6, 26),   # May 2026 data
    date(2026, 7, 31),   # Jun 2026 data
    date(2026, 8, 28),   # Jul 2026 data
    date(2026, 9, 25),   # Aug 2026 data
    date(2026, 10, 30),  # Sep 2026 data
    date(2026, 11, 25),  # Oct 2026 data
    date(2026, 12, 23),  # Nov 2026 data
]


def _build_event_list() -> list[EconomicEvent]:
    """Construct the full event list from hardcoded dates."""
    events: list[EconomicEvent] = []

    for d in _FOMC_2026:
        events.append(EconomicEvent(d, EventType.FOMC, f"FOMC rate decision {d}"))
    for d in _CPI_2026:
        events.append(EconomicEvent(d, EventType.CPI, f"CPI release {d}"))
    for d in _NFP_2026:
        events.append(EconomicEvent(d, EventType.NFP, f"NFP jobs report {d}"))
    for d in _GDP_2026:
        events.append(EconomicEvent(d, EventType.GDP, f"GDP advance estimate {d}"))
    for d in _PCE_2026:
        events.append(EconomicEvent(d, EventType.PCE, f"PCE release {d}"))

    events.sort(key=lambda e: e.date)
    return events


# Pre-built for fast lookups
_ALL_EVENTS = _build_event_list()
_EVENT_DATES: frozenset[date] = frozenset(e.date for e in _ALL_EVENTS)


class EconomicCalendar:
    """Provides macro event awareness for the trading bot.

    The bot should NOT hold iron condors (or open new positions) through
    major economic events. IV crush + gap risk makes this negative EV.

    Usage:
        cal = EconomicCalendar()
        if cal.should_skip_trading(date.today()):
            log.info("skipping trades — macro event window")
            return
    """

    def __init__(self) -> None:
        self._events = _ALL_EVENTS
        self._event_dates = _EVENT_DATES

    def is_event_day(self, check_date: date | None = None) -> bool:
        """Return True if the given date has a major economic release."""
        check_date = check_date or date.today()
        return check_date in self._event_dates

    def is_pre_event_day(self, check_date: date | None = None) -> bool:
        """Return True if *tomorrow* (next trading day) has a major release.

        For simplicity we check the next calendar day. Since most releases
        are on weekdays, and markets are closed on weekends, this is
        conservative (we might skip a Friday before a Monday event, which
        is exactly what we want — no weekend gap risk into an event).
        """
        check_date = check_date or date.today()
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
        check_date = check_date or date.today()
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
        check_date = check_date or date.today()
        return [e for e in self._events if e.date == check_date]

    def get_upcoming_events(self, days: int = 7) -> list[EconomicEvent]:
        """Get all events in the next N calendar days (inclusive of today)."""
        today = date.today()
        cutoff = today + timedelta(days=days)
        return [e for e in self._events if today <= e.date <= cutoff]

    def days_until_next_event(self, check_date: date | None = None) -> int | None:
        """Return days until the next macro event, or None if no future events."""
        check_date = check_date or date.today()
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
        for event in self._events:
            if entry_date < event.date <= expiry_date:
                return False
        return True

    def get_events_between(
        self, start: date, end: date
    ) -> list[EconomicEvent]:
        """Get all events between start and end (inclusive)."""
        return [e for e in self._events if start <= e.date <= end]

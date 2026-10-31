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
never reads the research overlay (the W68 guard: nothing under ``src/ait`` may know ``use_research_config``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache

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


def window_start(days: int, end: date) -> date:
    """First calendar day of a request that names ``days`` and no start date, counted back from ``end``."""
    return end - timedelta(days=int(days) + WINDOW_BUFFER_DAYS)


def required_window(*, today: date, days: int | None = None, start_date: date | None = None,
                    end_date: date | None = None) -> Window:
    """The window a run needs, from the same three values the loaders take.

    * end: ``end_date`` when the run is pinned, else the last completed session; never later than the last
      completed session (a pin in the future cannot be satisfied), and snapped back to a session.
    * start: ``start_date`` when given, else ``days`` back from the end (``window_start``).
    """
    cap = last_completed_session(today)
    end = last_session_on_or_before(min(end_date, cap)) if end_date is not None else cap
    if start_date is None:
        if days is None:
            raise ValueError("required_window needs start_date or days")
        start_date = window_start(days, end)
    return Window(start_date, end)


@dataclass
class PreflightReport:
    """The outcome of a preflight: the window checked, every input that was checked, and every gap."""

    window: Window
    checked: list[str] = field(default_factory=list)
    gaps: list[DataGap] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.gaps

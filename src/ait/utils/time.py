"""Market hours and trading calendar utilities.

All times are in US/Eastern (ET) timezone for consistency with US equity markets.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

# W38: ET and the stored-timestamp convention live in a stdlib-only leaf so the
# supervisor and reporting-only consumers can import them without this module's
# calendar stack. Re-exported here; every existing caller is unchanged.
from ait.utils.tz import ET, parse_trade_ts as _parse_trade_ts  # noqa: F401
UTC = ZoneInfo("UTC")

# NYSE calendar for market hours
_nyse = mcal.get_calendar("NYSE")

# Regular trading hours
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Pre/post market analysis windows
PRE_MARKET_START = time(9, 0)   # Start analysis 30 min before open
POST_MARKET_END = time(16, 15)  # Final reconciliation 15 min after close


def get_market_close(d: date | None = None) -> time:
    """Return the market close time for a given date.

    NYSE has early-close days (13:00 ET) such as the day before Independence
    Day, the day after Thanksgiving, and Christmas Eve.  The
    ``pandas_market_calendars`` library knows about these dates, so we query
    the NYSE schedule and extract the close time when available.

    Falls back to the regular-session close (16:00 ET) if the schedule is
    empty or lacks a ``market_close`` column.
    """
    d = d or now_et().date()
    schedule = _nyse.schedule(start_date=d, end_date=d)

    if not schedule.empty and "market_close" in schedule.columns:
        close_ts = schedule.iloc[0]["market_close"]
        # Convert to ET and extract the time component
        return close_ts.tz_convert(ET).time()

    return MARKET_CLOSE


def now_et() -> datetime:
    """Current time in Eastern timezone."""
    return datetime.now(ET)


def today_et() -> date:
    """The current ET calendar date — this codebase's date convention.

    W28: the single authority replacing 49 `date.today()` reads across src/.
    `date.today()` returns the HOST's date, and on this box (CEST = ET+6) that
    is tomorrow's date from 18:00 ET onward — so an evening restart ran startup
    reconcile and the launch coherence check against the wrong day.

    Measured, because the honest severity is narrower than the raw count
    suggests and narrower than W26's ledger entry implied: the whole RTH session
    (09:30-16:00 ET = 15:30-22:00 CEST) falls inside the band where the host
    date and the ET date AGREE, so the live scan/exit loops were never wrong on
    this host. That correctness is ACCIDENTAL — it holds only because the offset
    happens to be +6. A host at ET+13 would put RTH itself across the date line.
    Named rather than open-coded as `now_et().date()` so intent is explicit at
    the call site and the class guard can forbid `.today()` outright, with no
    allowlist.
    """
    return now_et().date()


def parse_trade_ts(value: str) -> datetime:
    """Parse a stored trade timestamp and return it as an AWARE ET datetime.

    W21: `trades.entry_time` / `trades.exit_time` (and the `open_positions` /
    `exit_outbox` mirrors) were written offset-free by `datetime.now()`, so the
    whole corpus is ET only by accident of the host. Writers now emit
    `now_et().isoformat()`, but a naive legacy row must keep parsing correctly
    and must keep meaning ET — otherwise a half-migrated DB silently shifts
    every trading-day and hour-of-day bucket downstream.

    Naive input is localised to ET; aware input is CONVERTED to ET, so `.date()`
    and `.hour` are ET calendar semantics for every caller regardless of which
    offset the row happens to carry. Raises ValueError/TypeError exactly as
    `datetime.fromisoformat` does, so existing caller guards still apply.

    W38: the implementation moved to `ait.utils.tz`, a stdlib-only leaf, so the
    supervisor and go_live can share it without paying for this module's market
    calendar (pandas_market_calendars -> pandas -> numpy at import, which is
    what killed the supervisor in the incident W34 fixed). Re-exported here so
    every existing caller and every `from ait.utils.time import ...` is
    unchanged. This is a delegation, not a copy — there is one implementation.
    """
    return _parse_trade_ts(value)


def is_market_open() -> bool:
    """Check if the market is currently open for regular trading."""
    now = now_et()
    today = now.date()

    # Check if today is a trading day
    schedule = _nyse.schedule(start_date=today, end_date=today)
    if schedule.empty:
        return False

    # Check if current time is within RTH
    current_time = now.time()
    return MARKET_OPEN <= current_time < get_market_close(today)


def is_trading_day(d: date | None = None) -> bool:
    """Check if a given date is a valid trading day."""
    d = d or now_et().date()
    schedule = _nyse.schedule(start_date=d, end_date=d)
    return not schedule.empty


def next_market_open(now: datetime | None = None) -> datetime:
    """Get the next market open time.

    W24: accepts an optional `now` so a caller making several related decisions
    in one breath (see MarketScheduler.wait_for_phase) can pass a SINGLE frozen
    instant instead of each call deriving its own fresh now_et() — which is
    exactly what let a caller observe "today's open already happened" here
    while a near-simultaneous read elsewhere still said "not yet", producing a
    spurious ~1-hour wait at the open itself (confirmed live 2026-09-16).
    """
    now = now or now_et()
    d = now.date()

    # If market hasn't opened yet today and today is a trading day
    if is_trading_day(d) and now.time() < MARKET_OPEN:
        return datetime.combine(d, MARKET_OPEN, tzinfo=ET)

    # Find next trading day
    for i in range(1, 10):
        candidate = d + timedelta(days=i)
        if is_trading_day(candidate):
            return datetime.combine(candidate, MARKET_OPEN, tzinfo=ET)

    raise RuntimeError("Could not find next trading day within 10 days")


def time_to_market_close() -> timedelta | None:
    """Time remaining until market close. Returns None if market is closed."""
    if not is_market_open():
        return None
    now = now_et()
    close_dt = datetime.combine(now.date(), get_market_close(now.date()), tzinfo=ET)
    return close_dt - now


def minutes_since_open() -> int | None:
    """Minutes elapsed since market open. Returns None if market is closed."""
    if not is_market_open():
        return None
    now = now_et()
    open_dt = datetime.combine(now.date(), MARKET_OPEN, tzinfo=ET)
    return int((now - open_dt).total_seconds() / 60)


def trading_days_between(start: date, end: date) -> int:
    """Count trading days between two dates (exclusive of end)."""
    schedule = _nyse.schedule(start_date=start, end_date=end)
    return len(schedule)


def get_recent_trading_days(n: int) -> list[date]:
    """Get the last N trading days up to today."""
    today = now_et().date()
    # Look back extra days to account for weekends/holidays
    start = today - timedelta(days=int(n * 1.5) + 10)
    schedule = _nyse.schedule(start_date=start, end_date=today)
    dates = [d.date() for d in schedule.index[-n:]]
    return dates


def next_session_on_or_after(d: date, max_ahead: int = 10) -> date:
    """W74: the first NYSE session on or after ``d`` (``d`` itself when it trades).

    Listed expiries never fall on a weekend or holiday; an option "N days out"
    that lands on one expires on the next session instead. ``max_ahead`` bounds
    the lookup (the longest NYSE closure is far shorter) and a miss raises
    rather than inventing a date.
    """
    sched = _nyse.schedule(start_date=d, end_date=d + timedelta(days=max_ahead))
    if sched.empty:
        raise ValueError(f"no NYSE session within {max_ahead} days of {d}")
    return sched.index[0].date()


def time_to_expiry_years(now: datetime, expiry: date, close: time | None = None) -> float:
    """W74: fractional calendar years from ``now`` to the expiry's closing bell.

    PM-settled ETF options stop trading at the session close, 13:00 ET on an
    early-close day (``get_market_close``), so a 0-DTE option held at 10:30 has
    ~5.5 hours of life, not zero and not a whole day. ``now`` must be tz-aware.
    Floors at 0 once the bell has passed. ``close`` overrides the lookup (tests).
    """
    if now.tzinfo is None:
        raise ValueError("time_to_expiry_years needs a timezone-aware 'now'")
    bell = datetime.combine(expiry, close if close is not None else get_market_close(expiry), tzinfo=ET)
    return max((bell - now.astimezone(ET)).total_seconds(), 0.0) / (365.0 * 86400.0)

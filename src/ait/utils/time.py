"""Market hours and trading calendar utilities.

All times are in US/Eastern (ET) timezone for consistency with US equity markets.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

ET = ZoneInfo("US/Eastern")
UTC = ZoneInfo("UTC")

# NYSE calendar for market hours
_nyse = mcal.get_calendar("NYSE")

# Regular trading hours
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# Pre/post market analysis windows
PRE_MARKET_START = time(9, 0)   # Start analysis 30 min before open
POST_MARKET_END = time(16, 15)  # Final reconciliation 15 min after close


def now_et() -> datetime:
    """Current time in Eastern timezone."""
    return datetime.now(ET)


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
    return MARKET_OPEN <= current_time < MARKET_CLOSE


def is_trading_day(d: date | None = None) -> bool:
    """Check if a given date is a valid trading day."""
    d = d or now_et().date()
    schedule = _nyse.schedule(start_date=d, end_date=d)
    return not schedule.empty


def next_market_open() -> datetime:
    """Get the next market open time."""
    now = now_et()
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
    close_dt = datetime.combine(now.date(), MARKET_CLOSE, tzinfo=ET)
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

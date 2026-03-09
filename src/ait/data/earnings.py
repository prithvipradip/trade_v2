"""Earnings calendar — prevents entering options positions near earnings.

IV crush after earnings can destroy option premium in minutes.
This module fetches earnings dates and blocks trades that would be
holding options through an earnings announcement.

Uses Yahoo Finance (free, no API key) as the primary source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ait.data.cache import TTLCache
from ait.utils.logging import get_logger

log = get_logger("data.earnings")


@dataclass
class EarningsInfo:
    """Earnings date information for a symbol."""

    symbol: str
    next_earnings_date: date | None
    is_confirmed: bool = False  # Whether the date is confirmed vs estimated


class EarningsCalendar:
    """Fetches and caches earnings dates to protect against IV crush."""

    def __init__(self, buffer_days_before: int = 2, buffer_days_after: int = 1) -> None:
        self._buffer_before = buffer_days_before
        self._buffer_after = buffer_days_after
        # Cache earnings dates for 6 hours (they don't change often)
        self._cache = TTLCache(default_ttl=21600, max_size=200)

    def get_next_earnings(self, symbol: str) -> EarningsInfo:
        """Get the next earnings date for a symbol."""
        cached = self._cache.get(f"earnings_{symbol}")
        if cached is not None:
            return cached

        info = self._fetch_earnings(symbol)
        self._cache.set(f"earnings_{symbol}", info)
        return info

    def is_near_earnings(self, symbol: str, check_date: date | None = None) -> bool:
        """Check if a date is within the earnings danger zone.

        Danger zone = [earnings - buffer_before, earnings + buffer_after]
        Default: 2 days before through 1 day after earnings.
        """
        check_date = check_date or date.today()
        info = self.get_next_earnings(symbol)

        if info.next_earnings_date is None:
            return False  # Unknown earnings date — allow trading

        days_until = (info.next_earnings_date - check_date).days

        # In danger zone: between -buffer_after and +buffer_before days from earnings
        if -self._buffer_after <= days_until <= self._buffer_before:
            log.info(
                "near_earnings",
                symbol=symbol,
                earnings_date=info.next_earnings_date.isoformat(),
                days_until=days_until,
            )
            return True

        return False

    def would_hold_through_earnings(
        self, symbol: str, entry_date: date, expiry_date: date
    ) -> bool:
        """Check if an options position would span an earnings date.

        Returns True if earnings falls between entry and expiry dates.
        """
        info = self.get_next_earnings(symbol)
        if info.next_earnings_date is None:
            return False

        # Check if earnings date is between entry and expiry
        return entry_date <= info.next_earnings_date <= expiry_date

    def _fetch_earnings(self, symbol: str) -> EarningsInfo:
        """Fetch next earnings date from Yahoo Finance."""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            cal = ticker.calendar

            if cal is not None and not cal.empty:
                # yfinance returns earnings date in different formats
                if hasattr(cal, "iloc"):
                    # DataFrame format
                    earnings_date = None
                    for col in cal.columns:
                        val = cal.iloc[0][col] if len(cal) > 0 else None
                        if isinstance(val, (datetime, date)):
                            earnings_date = val if isinstance(val, date) else val.date()
                            break
                    if earnings_date:
                        return EarningsInfo(symbol=symbol, next_earnings_date=earnings_date)

                # Try the earnings_dates attribute instead
                if hasattr(ticker, "earnings_dates") and ticker.earnings_dates is not None:
                    eds = ticker.earnings_dates
                    if not eds.empty:
                        future_dates = [
                            d.date() if hasattr(d, "date") else d
                            for d in eds.index
                            if (d.date() if hasattr(d, "date") else d) >= date.today()
                        ]
                        if future_dates:
                            return EarningsInfo(
                                symbol=symbol,
                                next_earnings_date=min(future_dates),
                            )

            log.debug("no_earnings_data", symbol=symbol)
            return EarningsInfo(symbol=symbol, next_earnings_date=None)

        except Exception as e:
            log.warning("earnings_fetch_failed", symbol=symbol, error=str(e))
            return EarningsInfo(symbol=symbol, next_earnings_date=None)

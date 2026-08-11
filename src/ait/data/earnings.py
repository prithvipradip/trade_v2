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
from ait.utils.time import now_et

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
        """Get the next earnings date for a symbol.

        Deep-audit DATA-M5: the synchronous yfinance fetch used to run
        directly on the asyncio event loop — a cache miss froze position
        monitoring, stops, and exit fills for the whole fetch. When called
        from inside a running loop we now NEVER block: kick a background
        thread to populate the cache and return "unknown" for this tick
        (same trade-through behavior as a failed fetch); the next tick gets
        the cached answer. Sync callers (tests/CLI) fetch inline as before.
        """
        cached = self._cache.get(f"earnings_{symbol}")
        if cached is not None:
            return cached

        import asyncio as _aio
        import threading as _th
        try:
            _aio.get_running_loop()
            inflight = getattr(self, "_inflight", None)
            if inflight is None:
                inflight = self._inflight = set()
            if symbol not in inflight:
                inflight.add(symbol)

                def _bg():
                    try:
                        info = self._fetch_earnings(symbol)
                        self._cache.set(f"earnings_{symbol}", info)
                    except Exception as e:  # noqa: BLE001
                        log.debug("earnings_bg_fetch_failed", symbol=symbol, error=str(e))
                    finally:
                        inflight.discard(symbol)

                _th.Thread(target=_bg, daemon=True).start()
            return EarningsInfo(symbol=symbol, next_earnings_date=None)
        except RuntimeError:
            info = self._fetch_earnings(symbol)
            self._cache.set(f"earnings_{symbol}", info)
            return info

    def is_near_earnings(self, symbol: str, check_date: date | None = None) -> bool:
        """Check if a date is within the earnings danger zone.

        Danger zone = [earnings - buffer_before, earnings + buffer_after]
        Default: 2 days before through 1 day after earnings.
        """
        # R17: was date.today() (server-local clock) instead of this
        # codebase's ET convention -- a multi-hour window around midnight
        # UTC could misclassify a same-day earnings date as "already past".
        check_date = check_date or now_et().date()
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

            # yfinance calendar can be a dict or DataFrame depending on version
            cal_empty = (
                cal is None
                or (isinstance(cal, dict) and not cal)
                or (hasattr(cal, "empty") and cal.empty)
            )
            if not cal_empty:
                # R7 CRITICAL FIX: the old code returned the FIRST datetime in
                # the dict — usually 'Dividend Date'/'Ex-Dividend Date', NOT
                # earnings. In production AAPL's "next earnings" was a past
                # dividend date and AMD's was 1995 — all three earnings guards
                # were dead for single names. Parse 'Earnings Date' by KEY,
                # reject past dates, and fall through to earnings_dates.
                def _to_date(v):
                    if isinstance(v, list) and v:
                        v = v[0]
                    if isinstance(v, datetime):
                        return v.date()
                    if isinstance(v, date):
                        return v
                    return None

                if isinstance(cal, dict):
                    for key in ("Earnings Date", "earnings date", "EarningsDate"):
                        if key in cal:
                            d = _to_date(cal[key])
                            if d is not None and d >= now_et().date():
                                return EarningsInfo(symbol=symbol, next_earnings_date=d)
                            break  # key present but past/unparseable -> earnings_dates fallback

                elif hasattr(cal, "iloc") and len(cal) > 0:
                    # DataFrame format: same rule — earnings columns only
                    for col in cal.columns:
                        if "earnings" not in str(col).lower():
                            continue
                        d = _to_date(cal.iloc[0][col])
                        if d is not None and d >= now_et().date():
                            return EarningsInfo(symbol=symbol, next_earnings_date=d)

                # Try the earnings_dates attribute instead
                if hasattr(ticker, "earnings_dates") and ticker.earnings_dates is not None:
                    eds = ticker.earnings_dates
                    if not eds.empty:
                        future_dates = [
                            d.date() if hasattr(d, "date") else d
                            for d in eds.index
                            if (d.date() if hasattr(d, "date") else d) >= now_et().date()
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

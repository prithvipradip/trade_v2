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
from ait.utils.time import now_et, today_et

log = get_logger("data.earnings")


@dataclass
class EarningsInfo:
    """Earnings date information for a symbol."""

    symbol: str
    next_earnings_date: date | None
    is_confirmed: bool = False  # Whether the date is confirmed vs estimated
    # W44 (fail-direction-03): True = a real answer -- a confirmed date, or a
    # CONFIRMED ABSENCE of one (every ETF in the current universe resolves
    # here). False = the fetch failed, or hasn't completed yet (the
    # background-thread cold-miss placeholder) -- the date is genuinely NOT
    # known and must not be treated the same as "confirmed no earnings".
    data_available: bool = True


class EarningsCalendar:
    """Fetches and caches earnings dates to protect against IV crush."""

    def __init__(
        self,
        buffer_days_before: int = 2,
        buffer_days_after: int = 1,
        refuse_entries_when_data_unavailable: bool = True,
        lkg_max_age_hours: int = 168,
    ) -> None:
        self._buffer_before = buffer_days_before
        self._buffer_after = buffer_days_after
        self._refuse_when_unavailable = refuse_entries_when_data_unavailable
        self._lkg_max_age_s = lkg_max_age_hours * 3600
        # Cache earnings dates for 6 hours (they don't change often). W44:
        # only ever populated with data_available=True answers via
        # _remember() below -- a fetch failure must not poison this for the
        # full TTL (concurrency-4 made TTLCache itself safe for the
        # background thread in get_next_earnings to write into concurrently
        # with main-thread reads).
        self._cache = TTLCache(default_ttl=21600, max_size=200)
        # W44 (fail-direction-03): last successful fetch per symbol, kept
        # independent of the TTL cache above so a later failure can fall
        # back to it instead of reporting bare "unknown".
        # symbol -> (info, monotonic time.time() of that successful fetch)
        self._last_good: dict[str, tuple[EarningsInfo, float]] = {}

    def get_next_earnings(self, symbol: str) -> EarningsInfo:
        """Get the next earnings date for a symbol.

        Deep-audit DATA-M5: the synchronous yfinance fetch used to run
        directly on the asyncio event loop — a cache miss froze position
        monitoring, stops, and exit fills for the whole fetch. When called
        from inside a running loop we now NEVER block: kick a background
        thread to populate the cache and return immediately for this tick
        (the last-known-good if one is still fresh -- W44 -- else
        "unknown"); the next tick gets the cached answer. Sync callers
        (tests/CLI) fetch inline as before.
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
                        self._remember(symbol, info)
                    except Exception as e:  # noqa: BLE001
                        # W44: was log.debug -- a silent debug line is a
                        # large part of why this whole class of failure was
                        # invisible until this wave.
                        log.warning("earnings_bg_fetch_failed", symbol=symbol, error=str(e))
                    finally:
                        inflight.discard(symbol)

                _th.Thread(target=_bg, daemon=True).start()
            return self._unavailable(symbol)
        except RuntimeError:
            info = self._fetch_earnings(symbol)
            self._remember(symbol, info)
            # W44: the LKG-or-unavailable policy applies uniformly to a sync
            # failure too, not just the async cold-miss placeholder above --
            # there is nothing sync-specific about "should a failure reuse
            # the last good answer".
            return info if info.data_available else self._unavailable(symbol)

    def _remember(self, symbol: str, info: EarningsInfo) -> None:
        """W44: cache and record last-known-good, but ONLY a real answer.

        A fetch failure (data_available=False) must never be written to the
        6-hour TTL cache -- that was the original defect (a transient
        failure cached as "confirmed no earnings" for the full TTL). A
        failure falls through to _unavailable() instead, which serves the
        existing last-known-good if one is still fresh enough.
        """
        if info.data_available:
            self._cache.set(f"earnings_{symbol}", info)
            self._last_good[symbol] = (info, time.time())

    def _unavailable(self, symbol: str) -> EarningsInfo:
        """W44: no fresh cached answer -- fetch is pending or just failed.

        Serve the last-known-good if it's still within lkg_max_age_hours
        (this is what makes the policy free for the ETF-only universe: their
        LKG says "confirmed no earnings", so a later outage keeps allowing
        them). Otherwise the symbol is genuinely unknown.
        """
        lkg = self._last_good.get(symbol)
        if lkg is not None:
            info, fetched_at = lkg
            age_s = time.time() - fetched_at
            if age_s <= self._lkg_max_age_s:
                log.warning(
                    "earnings_using_last_known_good",
                    symbol=symbol,
                    age_hours=round(age_s / 3600, 1),
                )
                return info
        return EarningsInfo(symbol=symbol, next_earnings_date=None, data_available=False)

    def is_near_earnings(self, symbol: str, check_date: date | None = None) -> bool:
        """Check if a date is within the earnings danger zone.

        Danger zone = [earnings - buffer_before, earnings + buffer_after]
        Default: 2 days before through 1 day after earnings.
        """
        # R17: was `date.today()` (server-local clock) instead of this
        # codebase's ET convention -- a multi-hour window around midnight
        # UTC could misclassify a same-day earnings date as "already past".
        # W28 generalised that one-site fix into today_et(); this call was
        # already correct, so it only adopts the shared name.
        check_date = check_date or today_et()
        info = self.get_next_earnings(symbol)

        # W44 (fail-direction-03): a symbol with no usable answer (never
        # fetched, or the fetch failed with no fresh last-known-good) used
        # to build the identical shape as "confirmed no earnings" and
        # allow trading either way. The two are now distinguishable;
        # refuse per config, since an unprotected single-name entry is
        # exactly the risk this gate exists to prevent. Operator-tunable
        # via earnings.refuse_entries_when_data_unavailable.
        if not info.data_available:
            return self._refuse_when_unavailable

        if info.next_earnings_date is None:
            return False  # Confirmed no upcoming earnings — allow trading

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
        # W44 (fail-direction-03): see the matching comment in
        # is_near_earnings -- same distinction, same operator-tunable policy.
        if not info.data_available:
            return self._refuse_when_unavailable
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

            # W44: confirmed absence -- a REAL answer. data_available stays
            # True (the dataclass default): every symbol in the live
            # ETF-only universe resolves here, and this is what lets a LATER
            # fetch failure fall back to "no earnings" instead of "unknown".
            log.debug("no_earnings_data", symbol=symbol)
            return EarningsInfo(symbol=symbol, next_earnings_date=None)

        except Exception as e:
            # W44 (fail-direction-03): this used to return the IDENTICAL
            # shape as the confirmed-absence case above, so a transient
            # fetch failure was cached for 6h and read downstream as
            # "definitely no earnings" -- indistinguishable from a real
            # answer. data_available=False keeps it out of the cache
            # (_remember) and routes callers through the last-known-good
            # fallback (_unavailable) instead.
            log.warning("earnings_fetch_failed", symbol=symbol, error=str(e))
            return EarningsInfo(symbol=symbol, next_earnings_date=None, data_available=False)

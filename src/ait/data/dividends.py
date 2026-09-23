"""Dividend calendar — prevents holding a short call through early-assignment
risk around ex-dividend dates, and blocks new entries whose hold horizon
would straddle one.

Early assignment on a short call is most likely right before the underlying
goes ex-dividend, when the dividend an assignor would capture exceeds the
option's remaining extrinsic (time) value. TLT pays monthly, SPY/QQQ/IWM
quarterly — all four names in the live universe carry this risk on a regular
cadence, unlike earnings (an ETF-only universe has no per-company earnings
date at all).

Structurally this module is EARNINGS.PY's twin — same cache/last-known-good/
fail-open-exit-fail-closed-entry shape — with one addition: a dividend needs
an AMOUNT, not just a date, to compare against extrinsic value.

Uses Yahoo Finance (free, no API key), like earnings.py. `ticker.calendar`
(earnings.py's own primary source) was checked directly against live SPY/TLT
output before writing this and returns an EMPTY dict for both — yfinance's
fundamentals-calendar endpoint 404s for ETFs ("No fundamentals data found").
`ticker.dividends` — a real historical Series of past ex-dividend dates and
per-share amounts — is what this module fetches instead; there is no "next"
entry in a historical series, so the next date is projected forward from the
average interval between recent payments (this works for both SPY/QQQ/IWM's
quarterly cadence and TLT's monthly one without hardcoding either).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, timedelta

from ait.data.cache import TTLCache
from ait.utils.logging import get_logger
from ait.utils.time import now_et, today_et

log = get_logger("data.dividends")

# How many recent payments to average the cadence over. 4 covers a full year
# of quarterly payers and a third of a year of monthly ones — enough to
# smooth a single irregular gap (a delayed declaration, a holiday shift)
# without the average going stale relative to the underlying's real schedule.
_HISTORY_WINDOW = 4
# Quarterly fallback when a symbol has fewer than 2 historical payments to
# derive a real interval from (a genuinely new fund, or a data gap).
_DEFAULT_INTERVAL_DAYS = 91
# Safety bound on the forward-projection loop below — a payment history so
# stale that 8 projected intervals still land in the past means the data is
# not trustworthy, not that the next date is 2+ years out.
_MAX_PROJECTION_STEPS = 8


@dataclass
class DividendInfo:
    """Next-ex-dividend information for a symbol."""

    symbol: str
    next_ex_div_date: date | None
    expected_amount: float | None
    # Same tri-state meaning as earnings.py's EarningsInfo.data_available:
    # True = a real answer (a projected date+amount, or a confirmed absence
    # of dividend history). False = the fetch failed or hasn't completed —
    # genuinely unknown, must not be treated as "no dividend".
    data_available: bool = True


class DividendCalendar:
    """Fetches and caches ex-dividend dates to protect against early
    assignment. Mirrors EarningsCalendar's shape exactly (earnings.py)."""

    def __init__(
        self,
        refuse_entries_when_data_unavailable: bool = True,
        lkg_max_age_hours: int = 168,
        persist=None,
    ) -> None:
        self._refuse_when_unavailable = refuse_entries_when_data_unavailable
        self._lkg_max_age_s = lkg_max_age_hours * 3600
        # W52a: same as EarningsCalendar — the last-known-good is written
        # through to the bot state (dividend_lkg_<symbol>) so a restart does
        # not begin with none and refuse on the first ticks.
        self._persist = persist
        # Same 6h TTL as EarningsCalendar — dividend schedules change even
        # less often than earnings dates.
        self._cache = TTLCache(default_ttl=21600, max_size=200)
        # symbol -> (info, monotonic time.time() of that successful fetch)
        self._last_good: dict[str, tuple[DividendInfo, float]] = {}

    def get_next_ex_div(self, symbol: str) -> DividendInfo:
        """Get the next expected ex-dividend date+amount for a symbol.

        Same non-blocking shape as EarningsCalendar.get_next_earnings: never
        run the synchronous yfinance fetch directly on the asyncio event
        loop. When called from inside a running loop, kick a background
        thread and return the last-known-good (or "unknown") for this tick;
        the next tick gets the cached answer. Sync callers (tests/CLI) fetch
        inline.
        """
        cached = self._cache.get(f"div_{symbol}")
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
                        info = self._fetch_dividend(symbol)
                        self._remember(symbol, info)
                    except Exception as e:  # noqa: BLE001
                        log.warning("dividend_bg_fetch_failed", symbol=symbol, error=str(e))
                    finally:
                        inflight.discard(symbol)

                _th.Thread(target=_bg, daemon=True).start()
            return self._unavailable(symbol)
        except RuntimeError:
            info = self._fetch_dividend(symbol)
            self._remember(symbol, info)
            return info if info.data_available else self._unavailable(symbol)

    def _remember(self, symbol: str, info: DividendInfo) -> None:
        """Cache and record last-known-good, but ONLY a real answer — a
        fetch failure must never be cached as "no dividend" for the full
        TTL, same lesson earnings.py's W44 already applied here."""
        if info.data_available:
            self._cache.set(f"div_{symbol}", info)
            fetched_at = time.time()
            self._last_good[symbol] = (info, fetched_at)
            self._persist_lkg(symbol, info, fetched_at)

    def _persist_lkg(self, symbol: str, info: DividendInfo, fetched_at: float) -> None:
        """W52a: write a REAL answer through to the bot state (best effort)."""
        if self._persist is None:
            return
        try:
            self._persist.set_state(f"dividend_lkg_{symbol}", json.dumps({
                "symbol": symbol,
                "next_ex_div_date": (info.next_ex_div_date.isoformat()
                                     if info.next_ex_div_date else None),
                "expected_amount": (float(info.expected_amount)
                                    if info.expected_amount is not None else None),
                "fetched_at": fetched_at,
            }))
        except Exception as e:  # noqa: BLE001
            log.warning("dividend_lkg_persist_failed", symbol=symbol, error=str(e))

    def _load_persisted_lkg(self, symbol: str) -> tuple[DividendInfo, float] | None:
        """W52a: the previous process's last real answer, or None."""
        if self._persist is None:
            return None
        try:
            raw = self._persist.get_state(f"dividend_lkg_{symbol}", "")
            if not raw:
                return None
            row = json.loads(raw)
            d = row.get("next_ex_div_date")
            amt = row.get("expected_amount")
            info = DividendInfo(
                symbol=symbol,
                next_ex_div_date=date.fromisoformat(d) if d else None,
                expected_amount=float(amt) if amt is not None else None,
                data_available=True,
            )
            return info, float(row["fetched_at"])
        except Exception as e:  # noqa: BLE001
            log.warning("dividend_lkg_unreadable", symbol=symbol, error=str(e))
            return None

    def _unavailable(self, symbol: str) -> DividendInfo:
        """No fresh cached answer — fetch is pending or just failed. Serve
        the last-known-good if still within lkg_max_age_hours."""
        lkg = self._last_good.get(symbol)
        if lkg is None:
            # W52a: cold start — hydrate from the previous process's answer.
            lkg = self._load_persisted_lkg(symbol)
            if lkg is not None:
                self._last_good[symbol] = lkg
        if lkg is not None:
            info, fetched_at = lkg
            age_s = time.time() - fetched_at
            if age_s <= self._lkg_max_age_s:
                log.warning(
                    "dividend_using_last_known_good",
                    symbol=symbol,
                    age_hours=round(age_s / 3600, 1),
                )
                return info
        return DividendInfo(symbol=symbol, next_ex_div_date=None,
                            expected_amount=None, data_available=False)

    def is_near_ex_div(self, symbol: str, check_date: date | None = None,
                       days_before: int = 1) -> bool:
        """Is check_date within days_before days of the next ex-dividend
        date? Mirrors EarningsCalendar.is_near_earnings's danger-zone shape,
        one-sided (before only — the risk is pre-ex-div, not post)."""
        check_date = check_date or today_et()
        info = self.get_next_ex_div(symbol)
        if not info.data_available:
            return self._refuse_when_unavailable
        if info.next_ex_div_date is None:
            return False
        days_until = (info.next_ex_div_date - check_date).days
        return 0 <= days_until <= days_before

    def would_hold_through_ex_div(
        self, symbol: str, entry_date: date, expiry_date: date
    ) -> bool:
        """Check if an options position would span an ex-dividend date.

        Mirrors EarningsCalendar.would_hold_through_earnings exactly —
        same entry_date <= date <= expiry_date shape, same
        data_available fail-closed-for-entries policy.
        """
        info = self.get_next_ex_div(symbol)
        if not info.data_available:
            return self._refuse_when_unavailable
        if info.next_ex_div_date is None:
            return False
        return entry_date <= info.next_ex_div_date <= expiry_date

    def _fetch_dividend(self, symbol: str) -> DividendInfo:
        """Fetch dividend history from Yahoo Finance and project the next
        ex-dividend date + amount forward from the recent cadence.

        ticker.calendar (earnings.py's primary source) is empty for ETFs —
        verified directly against live SPY/TLT output, not assumed.
        ticker.dividends is real history with no "next" entry, so the next
        date is estimated as the last payment date plus the average
        interval between the last few payments, and the expected amount is
        the last payment's own amount (amounts drift quarter to quarter —
        measured $1.797-$1.993 across SPY's last 4 — so the most recent
        actual payment is a better estimate than an annual-rate/4 or /12
        guess that would also need to know the payment frequency in advance).
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            history = ticker.dividends
            if history is None or history.empty or len(history) < 2:
                log.debug("no_dividend_history", symbol=symbol)
                return DividendInfo(symbol=symbol, next_ex_div_date=None,
                                    expected_amount=None)

            recent = history.tail(_HISTORY_WINDOW)
            dates = [d.date() if hasattr(d, "date") else d for d in recent.index]
            gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
            avg_interval = (sum(gaps) / len(gaps)) if gaps else _DEFAULT_INTERVAL_DAYS

            last_date = dates[-1]
            last_amount = float(recent.iloc[-1])
            projected = last_date + timedelta(days=round(avg_interval))

            # A stale-relative-to-today projection isn't "no dividend" (the
            # W44 lesson from earnings.py, applied here) — keep stepping
            # forward by the same interval, bounded, rather than reporting a
            # date already in the past.
            today = now_et().date()
            steps = 0
            while projected < today and steps < _MAX_PROJECTION_STEPS:
                projected = projected + timedelta(days=round(avg_interval))
                steps += 1

            return DividendInfo(symbol=symbol, next_ex_div_date=projected,
                                expected_amount=last_amount)

        except Exception as e:  # noqa: BLE001
            log.warning("dividend_fetch_failed", symbol=symbol, error=str(e))
            return DividendInfo(symbol=symbol, next_ex_div_date=None,
                                expected_amount=None, data_available=False)

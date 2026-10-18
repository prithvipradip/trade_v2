"""Pattern Day Trader (PDT) protection.

US regulation: accounts under $25k are limited to 3 day trades
in a rolling 5-business-day window. Violating this freezes the
account for 90 days.

A "day trade" = opening AND closing the same position on the same day.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date
from ait.utils.time import now_et, datetime

from ait.bot.state import StateManager
from ait.config.invariants import PDT_MAX_DAY_TRADES, PDT_WINDOW_DAYS
from ait.config.settings import AccountConfig
from ait.utils.logging import get_logger
from ait.utils.time import get_recent_trading_days

log = get_logger("risk.pdt")

# W70: the values live in ait.config.invariants (a regulatory rule, not a preference);
# these names stay because callers and tests import them from here.
MAX_DAY_TRADES = PDT_MAX_DAY_TRADES
WINDOW_DAYS = PDT_WINDOW_DAYS


@dataclass
class PDTStatus:
    """Current PDT status."""

    enabled: bool
    day_trades_used: int
    day_trades_remaining: int
    can_day_trade: bool
    window_resets: str  # Date when oldest day trade falls off


class PDTGuard:
    """Tracks day trades and prevents PDT violations.

    A day trade is counted when a position is OPENED and CLOSED
    on the same trading day.
    """

    def __init__(self, config: AccountConfig, state: StateManager) -> None:
        self._enabled = config.pdt_protection and config.pdt_account_under_25k
        self._state = state
        # Each entry: (date_str, symbol) of a day trade
        self._day_trades: deque[tuple[str, str]] = deque()
        # fail-direction-09 (blindspot_composition_hunt_20260825): the guard
        # used to convert its OWN failures into "allow". Two health flags now
        # carry that state instead:
        #   _state_unreadable  — the stored counter could not be parsed at
        #     load (sticky: nothing else can repair it in-process);
        #   _calendar_degraded — the market-calendar helper failed or came
        #     back empty on THIS pass (recomputed every check, so a calendar
        #     that recovers restores normal counting).
        # While enabled, either one BLOCKS the day trade. A PDT violation is
        # the one gate whose consequence (90-day account freeze) cannot be
        # unwound, so an unknown count must never read as zero.
        self._state_unreadable = False
        self._calendar_degraded = False
        self._load_state()

    def _load_state(self) -> None:
        """Load day trade history from persistent state."""
        import json

        try:
            stored = self._state.get_state("pdt_day_trades", "[]")
            trades = json.loads(stored)
            self._day_trades = deque(
                [(t["date"], t["symbol"]) for t in trades]
            )
            self._state_unreadable = False
        except Exception as e:  # noqa: BLE001 — corrupt JSON, wrong shape, or
            # an unreadable/locked state store. fail-direction-09: this used
            # to reset to an EMPTY deque with no log, i.e. "0 day trades
            # used", i.e. always-allow.
            self._day_trades = deque()
            self._state_unreadable = True
            log.error(
                "pdt_state_unreadable_fail_closed",
                error=str(e),
                enabled=self._enabled,
                detail=("day-trade history could not be read; day trades are "
                        "BLOCKED until pdt_day_trades is repaired"),
            )

        self._purge_old_trades()

    def _save_state(self) -> None:
        """Persist day trade history."""
        import json

        trades = [{"date": d, "symbol": s} for d, s in self._day_trades]
        self._state.set_state("pdt_day_trades", json.dumps(trades))

    def _purge_old_trades(self) -> None:
        """Remove day trades older than the 5-day window."""
        trading_days = self._window_days()
        if not trading_days:
            return  # degraded: keep every recorded trade (see _window_days)

        cutoff = trading_days[0].isoformat()
        while self._day_trades and self._day_trades[0][0] < cutoff:
            self._day_trades.popleft()

    def record_day_trade(self, symbol: str) -> None:
        """Record that a day trade occurred."""
        if not self._enabled:
            return

        # W41 (R26 regress-live-05): a load that failed left _day_trades as an
        # EMPTY deque (see _load_state) while _state_unreadable stayed sticky
        # true. _save_state() used to write unconditionally, so the FIRST day
        # trade recorded after a merely-transient unreadable load persisted
        # that empty deque plus today's one entry OVER the real 5-day history
        # — a locked store for one read became permanent history loss, not a
        # blocked trade. can_day_trade() already fails closed on
        # _state_unreadable; the store itself must fail closed too.
        #
        # Retry-and-MERGE, not retry-and-replace: the lock may already be gone
        # by the time a trade closes, so re-read first — but a bare re-read
        # would DISCARD any trade this same process recorded in-memory while
        # still unreadable (suppressed from the store, per the branch below,
        # but real for THIS process's own can_day_trade() count). A second
        # record_day_trade call straddling the recovery moment would then
        # silently drop the first trade from the persisted history — the
        # exact under-count class this finding is about, one call later.
        # Fold any not-yet-persisted entries back in once a real read
        # succeeds, instead of overwriting them.
        if self._state_unreadable:
            _pending = list(self._day_trades)  # recorded here, never saved
            self._load_state()
            if not self._state_unreadable:
                _known = set(self._day_trades)
                for _entry in _pending:
                    if _entry not in _known:
                        self._day_trades.append(_entry)
                        _known.add(_entry)

        today = now_et().date().isoformat()  # ET-pinned (deep-audit SR-M6)
        self._day_trades.append((today, symbol))
        if self._state_unreadable:
            log.warning(
                "pdt_history_write_suppressed_store_unverified",
                symbol=symbol,
                detail=("day trade recorded in-memory for this process, but "
                        "NOT persisted — the store is still unreadable, and "
                        "writing now would overwrite real history with an "
                        "incomplete deque"),
            )
        else:
            self._save_state()

        remaining = MAX_DAY_TRADES - self._count_in_window()
        log.warning(
            "day_trade_recorded",
            symbol=symbol,
            remaining=remaining,
        )

    def _window_days(self) -> list:
        """The rolling window's trading days, or [] when the calendar could
        not answer. fail-direction-09: an exception or an EMPTY calendar is a
        guard failure, not "no trades in window" — it sets _calendar_degraded
        so can_day_trade()/get_status() fail CLOSED instead of counting zero.
        """
        try:
            days = get_recent_trading_days(WINDOW_DAYS)
        except Exception as e:  # noqa: BLE001
            self._calendar_degraded = True
            log.error("pdt_calendar_unavailable_fail_closed", error=str(e),
                      enabled=self._enabled)
            return []
        if not days:
            self._calendar_degraded = True
            log.error("pdt_calendar_empty_fail_closed", enabled=self._enabled,
                      detail=("no trading days returned for the rolling "
                              "window; the day-trade count cannot be trusted"))
            return []
        return list(days)

    def _degraded(self) -> bool:
        """True when the guard cannot trust its own inputs."""
        return self._state_unreadable or self._calendar_degraded

    def can_day_trade(self) -> bool:
        """Check if we can make another day trade without violating PDT."""
        if not self._enabled:
            return True

        self._calendar_degraded = False  # re-evaluated by the calls below
        self._purge_old_trades()
        used = self._count_in_window()
        if self._degraded():
            # fail-direction-09: corrupt counter / broken calendar BLOCKS.
            log.error(
                "pdt_guard_degraded_day_trade_blocked",
                state_unreadable=self._state_unreadable,
                calendar_degraded=self._calendar_degraded,
                day_trades_known=len(self._day_trades),
            )
            return False
        return used < MAX_DAY_TRADES

    def get_status(self) -> PDTStatus:
        """Get current PDT status."""
        self._calendar_degraded = False  # re-evaluated by the calls below
        self._purge_old_trades()
        used = self._count_in_window()
        remaining = MAX_DAY_TRADES - used

        # When does the oldest trade fall off?
        reset_date = ""
        if self._day_trades:
            from datetime import timedelta

            oldest = datetime.strptime(self._day_trades[0][0], "%Y-%m-%d").date()
            # It falls off after 5 trading days
            trading_days = get_recent_trading_days(WINDOW_DAYS + 5)
            for td in trading_days:
                if td > oldest:
                    reset_date = td.isoformat()
                    break

        return PDTStatus(
            enabled=self._enabled,
            day_trades_used=used,
            day_trades_remaining=max(0, remaining),
            # fail-direction-09: mirror can_day_trade() — a degraded guard
            # reports "cannot day trade" rather than an optimistic remaining.
            can_day_trade=(remaining > 0
                           and not (self._enabled and self._degraded())),
            window_resets=reset_date,
        )

    def would_be_day_trade(self, symbol: str, entry_date: date) -> bool:
        """Check if closing a position entered today would be a day trade.

        Call this BEFORE entering a trade to warn the user that closing
        it today would consume a day trade.
        """
        return entry_date == now_et().date()  # ET-pinned (deep-audit SR-M6)

    def _count_in_window(self) -> int:
        """Count day trades in the rolling 5-day window.

        fail-direction-09: an empty/failed calendar used to return 0 — the
        most permissive answer possible. It now counts EVERY recorded trade
        as in-window (the conservative reading) and leaves _calendar_degraded
        set so the caller blocks regardless.
        """
        trading_days = self._window_days()
        if not trading_days:
            return len(self._day_trades)

        cutoff = trading_days[0].isoformat()
        return sum(1 for d, _ in self._day_trades if d >= cutoff)

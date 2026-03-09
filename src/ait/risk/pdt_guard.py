"""Pattern Day Trader (PDT) protection.

US regulation: accounts under $25k are limited to 3 day trades
in a rolling 5-business-day window. Violating this freezes the
account for 90 days.

A "day trade" = opening AND closing the same position on the same day.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime

from ait.bot.state import StateManager
from ait.config.settings import AccountConfig
from ait.utils.logging import get_logger
from ait.utils.time import get_recent_trading_days

log = get_logger("risk.pdt")

MAX_DAY_TRADES = 3
WINDOW_DAYS = 5


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
        self._load_state()

    def _load_state(self) -> None:
        """Load day trade history from persistent state."""
        import json

        stored = self._state.get_state("pdt_day_trades", "[]")
        try:
            trades = json.loads(stored)
            self._day_trades = deque(
                [(t["date"], t["symbol"]) for t in trades]
            )
        except (json.JSONDecodeError, KeyError):
            self._day_trades = deque()

        self._purge_old_trades()

    def _save_state(self) -> None:
        """Persist day trade history."""
        import json

        trades = [{"date": d, "symbol": s} for d, s in self._day_trades]
        self._state.set_state("pdt_day_trades", json.dumps(trades))

    def _purge_old_trades(self) -> None:
        """Remove day trades older than the 5-day window."""
        trading_days = get_recent_trading_days(WINDOW_DAYS)
        if not trading_days:
            return

        cutoff = trading_days[0].isoformat()
        while self._day_trades and self._day_trades[0][0] < cutoff:
            self._day_trades.popleft()

    def record_day_trade(self, symbol: str) -> None:
        """Record that a day trade occurred."""
        if not self._enabled:
            return

        today = date.today().isoformat()
        self._day_trades.append((today, symbol))
        self._save_state()

        remaining = MAX_DAY_TRADES - self._count_in_window()
        log.warning(
            "day_trade_recorded",
            symbol=symbol,
            remaining=remaining,
        )

    def can_day_trade(self) -> bool:
        """Check if we can make another day trade without violating PDT."""
        if not self._enabled:
            return True

        self._purge_old_trades()
        return self._count_in_window() < MAX_DAY_TRADES

    def get_status(self) -> PDTStatus:
        """Get current PDT status."""
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
            can_day_trade=remaining > 0,
            window_resets=reset_date,
        )

    def would_be_day_trade(self, symbol: str, entry_date: date) -> bool:
        """Check if closing a position entered today would be a day trade.

        Call this BEFORE entering a trade to warn the user that closing
        it today would consume a day trade.
        """
        return entry_date == date.today()

    def _count_in_window(self) -> int:
        """Count day trades in the rolling 5-day window."""
        trading_days = get_recent_trading_days(WINDOW_DAYS)
        if not trading_days:
            return 0

        cutoff = trading_days[0].isoformat()
        return sum(1 for d, _ in self._day_trades if d >= cutoff)

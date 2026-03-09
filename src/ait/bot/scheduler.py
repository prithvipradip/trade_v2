"""Market-aware scheduler.

Manages the bot's daily lifecycle:
- Pre-market: Load data, train models, prepare signals
- Market open: Run trading loop
- Post-market: Reconcile positions, generate daily report
- Off-hours: Sleep until next trading day
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from enum import Enum

from ait.utils.logging import get_logger
from ait.utils.time import (
    ET,
    MARKET_CLOSE,
    MARKET_OPEN,
    POST_MARKET_END,
    PRE_MARKET_START,
    is_market_open,
    is_trading_day,
    next_market_open,
    now_et,
)

log = get_logger("bot.scheduler")


class TradingPhase(str, Enum):
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    POST_MARKET = "post_market"
    OFF_HOURS = "off_hours"


class MarketScheduler:
    """Determines what the bot should be doing based on current time."""

    def get_current_phase(self) -> TradingPhase:
        """Get the current trading phase."""
        now = now_et()

        if not is_trading_day():
            return TradingPhase.OFF_HOURS

        current_time = now.time()

        if current_time < PRE_MARKET_START:
            return TradingPhase.OFF_HOURS
        elif current_time < MARKET_OPEN:
            return TradingPhase.PRE_MARKET
        elif current_time < MARKET_CLOSE:
            return TradingPhase.MARKET_OPEN
        elif current_time < POST_MARKET_END:
            return TradingPhase.POST_MARKET
        else:
            return TradingPhase.OFF_HOURS

    async def wait_for_phase(self, target: TradingPhase) -> None:
        """Sleep until the target phase begins."""
        while self.get_current_phase() != target:
            now = now_et()

            if target == TradingPhase.PRE_MARKET:
                if is_trading_day():
                    target_time = datetime.combine(now.date(), PRE_MARKET_START, tzinfo=ET)
                    if now >= target_time:
                        # Today's pre-market passed, wait for tomorrow
                        next_open = next_market_open()
                        wait = (next_open - timedelta(minutes=30) - now).total_seconds()
                    else:
                        wait = (target_time - now).total_seconds()
                else:
                    next_open = next_market_open()
                    wait = (next_open - timedelta(minutes=30) - now).total_seconds()

            elif target == TradingPhase.MARKET_OPEN:
                if is_trading_day() and now.time() < MARKET_OPEN:
                    target_time = datetime.combine(now.date(), MARKET_OPEN, tzinfo=ET)
                    wait = (target_time - now).total_seconds()
                else:
                    next_open = next_market_open()
                    wait = (next_open - now).total_seconds()

            else:
                wait = 60  # Check every minute for other phases

            wait = max(1, min(wait, 3600))  # Clamp: 1s to 1hr
            log.info(
                "waiting_for_phase",
                target=target.value,
                wait_seconds=int(wait),
                current_phase=self.get_current_phase().value,
            )
            await asyncio.sleep(wait)

    def seconds_until_close(self) -> float:
        """Seconds until market close. Returns 0 if market is closed."""
        if not is_market_open():
            return 0

        now = now_et()
        close = datetime.combine(now.date(), MARKET_CLOSE, tzinfo=ET)
        return max(0, (close - now).total_seconds())

    def should_avoid_new_trades(self) -> bool:
        """Check if it's too close to market close for new entries.

        Don't enter new positions in the last 15 minutes.
        """
        remaining = self.seconds_until_close()
        return 0 < remaining < 900  # Last 15 minutes

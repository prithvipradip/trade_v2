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
    get_market_close,
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

    def __init__(self, settings=None) -> None:
        # W35 (R26 entry-blackout-cutoff-900s): the no-new-entries window
        # before the close was a bare `900` in should_avoid_new_trades.
        # config.yaml trading.no_new_entries_minutes_before_close is its home;
        # a scheduler built bare keeps the historical 15 minutes.
        _minutes = getattr(getattr(settings, "trading", None),
                           "no_new_entries_minutes_before_close", 15)
        self._no_new_entries_s = int(_minutes) * 60

    def get_current_phase(self, now: datetime | None = None) -> TradingPhase:
        """Get the current trading phase.

        W24: accepts an optional `now` so callers making several related
        decisions together (wait_for_phase, its log line) can share ONE
        instant instead of each derivation re-reading the clock — see
        wait_for_phase's docstring for the race that motivated this.
        """
        now = now or now_et()

        if not is_trading_day(now.date()):
            return TradingPhase.OFF_HOURS

        current_time = now.time()
        # W34 (R26 w24-get-current-phase-ignores-its-own-now): bare
        # get_market_close() reads the REAL clock's date, so the caller's
        # `now` — W24's whole mechanism — was ignored for the close boundary:
        # 13:30 ET on a 13:00 half-day read MARKET_OPEN unless today happened
        # to be that half-day.
        close = get_market_close(now.date())
        # On half-days the post-market window is 15 minutes after actual close
        close_dt = datetime.combine(now.date(), close, tzinfo=ET)
        post_market_end = (close_dt + timedelta(minutes=15)).time()

        if current_time < PRE_MARKET_START:
            return TradingPhase.OFF_HOURS
        elif current_time < MARKET_OPEN:
            return TradingPhase.PRE_MARKET
        elif current_time < close:
            return TradingPhase.MARKET_OPEN
        elif current_time < post_market_end:
            return TradingPhase.POST_MARKET
        else:
            return TradingPhase.OFF_HOURS

    async def wait_for_phase(self, target: TradingPhase) -> None:
        """Sleep until the target phase begins.

        W24: every decision below shares ONE `now`, captured fresh at the top
        of each loop iteration, and the phase is re-checked against THAT now
        before any wait is computed. Before this fix, get_current_phase() (the
        while condition), next_market_open() and the log line's current_phase
        each called now_et() independently — up to five unsynchronized clock
        reads per iteration. When two of them landed on opposite sides of the
        MARKET_OPEN boundary, the function could observe "already open" for
        the log line while having already committed to a wait computed a
        moment earlier when it still looked like "not yet" — and that earlier
        read's branch treats "today's open must be in the past" as certain,
        handing off to next_market_open() for TOMORROW's open: a ~1-hour
        sleep, entered at the single moment the bot most needs to be running.
        Confirmed live 2026-09-16, 09:30:00 ET:
            {"target": "market_open", "wait_seconds": 3600,
             "current_phase": "market_open", "event": "waiting_for_phase"}
        — current_phase was already "market_open" in the very log line that
        committed to a 3600s sleep. Recovered only because the supervisor's
        independent heartbeat-staleness check happened to tick 7 seconds
        later; not a safeguard this function knows about or should rely on.

        With a single shared `now`, that contradiction is structurally
        impossible: if `now` already satisfies target, every branch below is
        skipped entirely via the early return, so a stale wait can never be
        computed, logged, or slept on.
        """
        while True:
            now = now_et()
            if self.get_current_phase(now) == target:
                return

            if target == TradingPhase.PRE_MARKET:
                if is_trading_day(now.date()):
                    target_time = datetime.combine(now.date(), PRE_MARKET_START, tzinfo=ET)
                    if now >= target_time:
                        # Today's pre-market passed, wait for tomorrow
                        next_open = next_market_open(now)
                        wait = (next_open - timedelta(minutes=30) - now).total_seconds()
                    else:
                        wait = (target_time - now).total_seconds()
                else:
                    next_open = next_market_open(now)
                    wait = (next_open - timedelta(minutes=30) - now).total_seconds()

            elif target == TradingPhase.MARKET_OPEN:
                if is_trading_day(now.date()) and now.time() < MARKET_OPEN:
                    target_time = datetime.combine(now.date(), MARKET_OPEN, tzinfo=ET)
                    wait = (target_time - now).total_seconds()
                else:
                    next_open = next_market_open(now)
                    wait = (next_open - now).total_seconds()

            else:
                wait = 60  # Check every minute for other phases

            wait = max(1, min(wait, 3600))  # Clamp: 1s to 1hr
            log.info(
                "waiting_for_phase",
                target=target.value,
                wait_seconds=int(wait),
                current_phase=self.get_current_phase(now).value,
            )
            await asyncio.sleep(wait)

    def seconds_until_close(self) -> float:
        """Seconds until market close. Returns 0 if market is closed."""
        if not is_market_open():
            return 0

        now = now_et()
        close = datetime.combine(now.date(), get_market_close(now.date()), tzinfo=ET)
        return max(0, (close - now).total_seconds())

    def should_avoid_new_trades(self) -> bool:
        """Check if it's too close to market close for new entries.

        Don't enter new positions in the last 15 minutes.
        """
        remaining = self.seconds_until_close()
        return 0 < remaining < self._no_new_entries_s

"""R14 Tier-1 item 3 — staleness gate on exit inputs.

The short-strike TOUCH stop is a credit position's primary early-loss cap, and
it is the ONE exit rule that acts directly on the underlying's price. It read
that price with no quality check at all: `Quote.timestamp` (fixed to carry the
exchange tick time in audit A2) was written but never read, and
`DataQualityValidator` had existed, instantiated, with zero call sites.

A frozen feed returns the same number forever with a stale timestamp, so a
single unvalidated read could fire the touch on a breach that had long since
reversed — or, on a real breach seen through a dead feed, look identical to a
healthy tick. This gate classifies the quote (fresh / degraded / frozen /
missing) and requires corroboration before firing on anything but a fresh one,
while NEVER going silent on a confirmed touch (a missed real breach costs the
wing width or, on a strangle, is unbounded; a false one costs the spread).

Harness: real PortfolioManager via __new__, only collaborators mocked — the
same pattern as test_credit_exits.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.config.settings import ExitConfig
from ait.data.market_data import Quote
from ait.execution.portfolio import PortfolioManager

# A $2-wide condor; short put 98, short call 102.
CONDOR_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])


@dataclass
class FakeTrade:
    entry_price: float = 1.00
    quantity: int = 1
    contract_type: str = "spread"
    strategy: str = "iron_condor"
    symbol: str = "SPY"
    trade_id: str = "T-CREDIT"
    direction: object = None
    legs: str = CONDOR_LEGS
    expiry: str | None = None
    strike: float | None = None
    entry_time: str = "2026-07-01T10:00:00"


class _QuoteFeed:
    """Scriptable quote source. Each call returns the next quote; `advance`
    controls whether the tick time moves (a frozen feed repeats a timestamp)."""

    def __init__(self, spot, bid=None, ask=None, advance=True, base_ts=None):
        self.spot = spot
        self.bid = bid
        self.ask = ask
        self.advance = advance
        # The validator scores staleness against wall-clock now(). An advancing
        # feed anchors a couple seconds back so every tick reads genuinely
        # fresh. A NON-advancing feed anchors well past the 180s staleness bound
        # so it reads degraded-by-age on the first look and frozen-by-repeat
        # thereafter — the deterministic stand-in for a feed that died minutes
        # ago (a feed that only just froze still, correctly, reads fresh once).
        if base_ts is not None:
            self._ts = base_ts
        elif advance:
            self._ts = datetime.now() - timedelta(seconds=2)
        else:
            self._ts = datetime.now() - timedelta(seconds=600)
        self.calls = 0

    async def get_quote(self, symbol):
        self.calls += 1
        if self.advance:
            self._ts = self._ts + timedelta(seconds=1)
        b = self.bid if self.bid is not None else round(self.spot - 0.01, 2)
        a = self.ask if self.ask is not None else round(self.spot + 0.01, 2)
        return Quote(symbol=symbol, bid=b, ask=a, last=self.spot,
                     volume=1_000_000, timestamp=self._ts)


def _manager(feed: _QuoteFeed, unrealized=10.0, touch_ticks=2,
             notify=None) -> PortfolioManager:
    mgr = PortfolioManager.__new__(PortfolioManager)
    mgr._ibkr = MagicMock()
    mgr._ibkr.ib.portfolio.return_value = []
    mgr._state = MagicMock()
    mgr._state.get_high_water_mark.return_value = 0.0
    mgr._market_data = MagicMock()
    mgr._market_data.get_quote = feed.get_quote
    mgr._exit_config = ExitConfig(touch_confirm_ticks=touch_ticks)
    mgr._earnings = None
    mgr._economic_cal = None
    mgr._notify_cb = notify

    async def _vol_mult(symbol):
        return 1.0
    mgr._get_volatility_stop_multiplier = _vol_mult
    mgr._pdt_guard = MagicMock()
    mgr._pdt_guard.would_be_day_trade.return_value = False
    mgr._option_position_unrealized = lambda trade, marks: unrealized
    mgr._get_position_delta = lambda trade: None
    return mgr


def _condor(dte: int = 20) -> FakeTrade:
    t = FakeTrade(expiry=(date.today() + timedelta(days=dte)).isoformat())
    t.direction = SimpleNamespace(value="neutral")
    return t


@pytest.fixture(autouse=True)
def _touch_on(monkeypatch):
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
    monkeypatch.delenv("AIT_SKIP_MACRO_EVENTS", raising=False)


# --------------------------------------------------------------------------
# Fresh quote — the gate must be invisible; a single touch fires immediately.
# --------------------------------------------------------------------------

async def test_fresh_quote_touch_fires_on_first_tick():
    feed = _QuoteFeed(spot=97.5)  # through the 98 put, advancing timestamps
    mgr = _manager(feed)
    status = await mgr._evaluate_position(_condor())
    assert status.should_exit
    assert "short_strike_touch" in status.exit_reason
    assert "agreeing ticks" not in status.exit_reason  # fast path, no suffix


async def test_fresh_quote_inside_strikes_does_not_exit():
    feed = _QuoteFeed(spot=100.0)
    mgr = _manager(feed)
    status = await mgr._evaluate_position(_condor())
    assert not status.should_exit


# --------------------------------------------------------------------------
# Frozen feed — timestamp never advances. Corroboration required; still fires.
# --------------------------------------------------------------------------

async def test_frozen_feed_touch_needs_confirmation_then_fires():
    feed = _QuoteFeed(spot=97.5, advance=False)  # touched, but never ticks
    mgr = _manager(feed, touch_ticks=2)
    trade = _condor()

    first = await mgr._evaluate_position(trade)
    assert not first.should_exit  # one frozen print is not enough

    second = await mgr._evaluate_position(trade)
    assert second.should_exit  # a second agreeing look confirms it
    assert "short_strike_touch" in second.exit_reason
    assert "frozen" in second.exit_reason


async def test_frozen_feed_pages_once():
    notes = []

    async def _notify(msg):
        notes.append(msg)

    feed = _QuoteFeed(spot=97.5, advance=False)
    mgr = _manager(feed, touch_ticks=3, notify=_notify)
    trade = _condor()
    for _ in range(4):
        await mgr._evaluate_position(trade)
    assert sum("STALE EXIT FEED" in n for n in notes) == 1


async def test_frozen_but_never_touched_never_exits():
    """A dead feed reading a safe price must NOT manufacture an exit."""
    feed = _QuoteFeed(spot=100.0, advance=False)
    mgr = _manager(feed, touch_ticks=2)
    trade = _condor()
    for _ in range(5):
        status = await mgr._evaluate_position(trade)
        assert not status.should_exit


# --------------------------------------------------------------------------
# Degraded quote (validator fails: crossed / wide) — same corroboration rule.
# --------------------------------------------------------------------------

async def test_degraded_wide_spread_needs_confirmation():
    # Advancing timestamps (not frozen) but a grossly wide spread => degraded.
    feed = _QuoteFeed(spot=97.5, bid=90.0, ask=105.0, advance=True)
    mgr = _manager(feed, touch_ticks=2)
    trade = _condor()
    first = await mgr._evaluate_position(trade)
    assert not first.should_exit
    second = await mgr._evaluate_position(trade)
    assert second.should_exit
    assert "agreeing ticks" in second.exit_reason


async def test_confirm_ticks_one_restores_fire_on_single_bad_print():
    """touch_confirm_ticks=1 is the documented escape hatch back to pre-R14
    behaviour: fire even on a lone frozen print."""
    feed = _QuoteFeed(spot=97.5, advance=False)
    mgr = _manager(feed, touch_ticks=1)
    status = await mgr._evaluate_position(_condor())
    assert status.should_exit
    assert "short_strike_touch" in status.exit_reason


async def test_recovery_inside_strikes_resets_the_streak():
    """A frozen touch that then pulls back inside must not carry its partial
    confirmation into a later, unrelated breach."""
    trade = _condor()

    touched = _QuoteFeed(spot=97.5, advance=False)
    mgr = _manager(touched, touch_ticks=2)
    s1 = await mgr._evaluate_position(trade)
    assert not s1.should_exit  # 1/2 toward confirmation

    # Same manager, spot recovers inside the strikes -> streak should reset.
    mgr._market_data.get_quote = _QuoteFeed(spot=100.0, advance=False).get_quote
    s2 = await mgr._evaluate_position(trade)
    assert not s2.should_exit

    # Breach again, frozen: this must be treated as the FIRST agreeing tick,
    # not the second — i.e. still no exit.
    mgr._market_data.get_quote = _QuoteFeed(spot=97.5, advance=False).get_quote
    s3 = await mgr._evaluate_position(trade)
    assert not s3.should_exit


# --------------------------------------------------------------------------
# Missing underlying quote — must NOT disable the position wholesale.
# --------------------------------------------------------------------------

async def test_missing_quote_keeps_dte_safety_exit_alive():
    """The old code returned None (no PositionStatus) when spot was missing,
    killing the DTE/expiry safety exit too. An option's P&L comes from leg
    marks, not spot; only touch/assignment need spot."""
    async def _no_quote(symbol):
        return None

    feed = _QuoteFeed(spot=0)
    mgr = _manager(feed, unrealized=5.0)
    mgr._market_data.get_quote = _no_quote
    status = await mgr._evaluate_position(_condor(dte=3))  # inside DTE<=5
    assert status is not None
    assert status.should_exit
    assert "expiry_approaching" in status.exit_reason


async def test_missing_quote_does_not_touch():
    async def _no_quote(symbol):
        return None

    feed = _QuoteFeed(spot=0)
    mgr = _manager(feed, unrealized=10.0)
    mgr._market_data.get_quote = _no_quote
    status = await mgr._evaluate_position(_condor(dte=20))
    assert status is not None
    assert not status.should_exit  # far DTE, healthy P&L, no spot -> hold


async def test_missing_quote_still_takes_profit_from_marks():
    """P&L-driven exits read leg marks, not spot, so they must survive a
    missing underlying quote."""
    async def _no_quote(symbol):
        return None

    feed = _QuoteFeed(spot=0)
    mgr = _manager(feed, unrealized=55.0)  # +55% of credit
    mgr._market_data.get_quote = _no_quote
    status = await mgr._evaluate_position(_condor(dte=20))
    assert status.should_exit
    assert "take_profit_short" in status.exit_reason

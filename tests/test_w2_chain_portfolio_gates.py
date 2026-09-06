"""W2 - registered defects from the blind-spot composition hunt (2026-08-25).

Two gates that were failing OPEN in the two places a credit book cannot
afford one:

  fail-direction-06-liquidity-gate-passes-quoteless-contracts
  external-contracts-02-zero-quote-reads-as-max-liquidity
      A contract with no live market (bid/ask 0.0 or NaN) scored
      (ask - bid) / last = 0.0 -- the TIGHTEST spread in the chain -- so on
      this profile (min_volume=0, min_oi=10 with the IBKR unknown-OI
      carve-out, i.e. the spread is the only active criterion) a phantom
      quote outranked every real market and its stale `last` print became
      the condor's credit.

  fail-direction-04-touch-stop-except-fails-open-at-debug
      The short-strike touch block swallowed every exception at log.debug.
      The flat credit stop is disabled by contract default, so the touch
      stop is the ONLY loss exit on a credit structure: one malformed legs
      row silently switched loss protection off for the life of the trade,
      re-failing every 30s tick with nothing but a DEBUG line.

Everything here EXECUTES the real code paths (real OptionContract /
OptionsChain objects, the real PortfolioManager exit evaluation). No
inspect.getsource assertions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.bot.state import TradeStatus
from ait.config.settings import ExitConfig, OptionsConfig
from ait.data.market_data import Quote
from ait.data.options_chain import (
    OptionContract,
    OptionsChain,
    _liquidity_thresholds,
)
from ait.execution import portfolio as portfolio_mod
from ait.execution.portfolio import PortfolioManager

EXPIRY = date.today() + timedelta(days=21)

# The live profile the finding's probe ran under (config.yaml:79-81 / .env):
# volume floor 0, OI floor 10, max spread 40%.
PROD_LIQ = {"AIT_LIQ_MIN_VOLUME": "0", "AIT_LIQ_MIN_OI": "10",
            "AIT_LIQ_MAX_SPREAD": "0.40"}
PROD_CFG = OptionsConfig(min_volume=0, min_open_interest=10,
                         max_bid_ask_spread_pct=0.40)


@pytest.fixture
def liq_env(monkeypatch):
    """Resolve the production liquidity thresholds for this test.

    _liquidity_thresholds is lru_cached process-wide, so it is cleared on the
    way in AND on the way out -- leaving this env cached would change how
    every later test file's contracts are scored.
    """
    for k, v in PROD_LIQ.items():
        monkeypatch.setenv(k, v)
    _liquidity_thresholds.cache_clear()
    yield
    _liquidity_thresholds.cache_clear()


def _contract(**kw) -> OptionContract:
    """A real OptionContract; defaults are a healthy, genuinely quoted call."""
    fields = dict(
        symbol="SPY", expiry=EXPIRY, strike=650.0, right="C",
        bid=1.80, ask=1.90, last=1.85, volume=500, open_interest=250,
        implied_vol=0.18, delta=0.20, gamma=0.01, theta=-0.05, vega=0.10,
        con_id=123456, source="ibkr",
    )
    fields.update(kw)
    return OptionContract(**fields)


# ---------------------------------------------------------------------------
# fail-direction-06 / external-contracts-02 -- contract-level liquidity gate
# ---------------------------------------------------------------------------

class TestQuotelessContractsAreIlliquid:

    def test_no_quote_contract_is_rejected(self, liq_env):
        """The finding's exact probe: bid=0, ask=0, a stale last=2.50."""
        c = _contract(bid=0.0, ask=0.0, last=2.50, volume=0, open_interest=0)
        assert c.has_two_sided_quote is False
        assert c.is_liquid is False
        # Pre-fix this was 0.000 -- the tightest "spread" in the chain.
        assert c.spread_pct > PROD_CFG.max_bid_ask_spread_pct

    def test_zero_bid_is_rejected(self, liq_env):
        """One-sided market: an offer with no bid is not a market."""
        c = _contract(bid=0.0, ask=1.90, last=1.85)
        assert c.is_liquid is False
        assert c.spread_pct > PROD_CFG.max_bid_ask_spread_pct

    def test_zero_ask_is_rejected(self, liq_env):
        c = _contract(bid=1.80, ask=0.0, last=1.85)
        assert c.is_liquid is False

    def test_nan_ask_is_rejected(self, liq_env):
        """ib_insync hands back NaN for "no tick yet"; every NaN comparison is
        False, so `ask > 0` read a missing quote as "not positive" in one
        place while arithmetic propagated it in another."""
        c = _contract(bid=1.80, ask=float("nan"), last=1.85)
        assert c.is_liquid is False
        assert not math.isnan(c.spread_pct)
        assert c.spread_pct > PROD_CFG.max_bid_ask_spread_pct

    def test_nan_bid_is_rejected(self, liq_env):
        c = _contract(bid=float("nan"), ask=1.90, last=1.85)
        assert c.is_liquid is False

    def test_crossed_book_is_rejected(self, liq_env):
        """ask < bid gives a NEGATIVE spread, which reads as tighter than any
        threshold -- the same fail direction as the quote-less case."""
        c = _contract(bid=2.00, ask=1.50, last=1.75)
        assert c.spread_pct > PROD_CFG.max_bid_ask_spread_pct
        assert c.is_liquid is False

    def test_healthy_wide_but_quoted_contract_is_still_accepted(self, liq_env):
        """The genuinely quoted contract must behave exactly as before: this
        one is WIDE (1.60/2.20 = 31.6% of mid) but real, and 31.6% < 40%."""
        c = _contract(bid=1.60, ask=2.20, last=1.90, volume=0, open_interest=0)
        assert c.has_two_sided_quote is True
        assert c.spread_pct == pytest.approx(0.60 / 1.90)
        assert c.is_liquid is True

    def test_tight_quoted_contract_is_still_accepted(self, liq_env):
        c = _contract(bid=1.80, ask=1.90, last=1.85)
        assert c.spread_pct == pytest.approx(0.10 / 1.85)
        assert c.is_liquid is True

    def test_quoteless_no_longer_outranks_a_real_market(self, liq_env):
        """The ranking claim in the finding: the no-market contract literally
        scored better than every quoted one."""
        quoteless = _contract(bid=0.0, ask=0.0, last=2.50)
        quoted = _contract(strike=655.0, bid=1.80, ask=1.90, last=1.85)
        assert quoteless.spread_pct > quoted.spread_pct

    def test_wide_quoted_contract_beyond_the_cap_still_rejected(self, liq_env):
        """Unchanged behaviour: a real but ~62%-wide market fails the cap."""
        c = _contract(bid=1.00, ask=1.90, last=1.45)
        assert c.is_liquid is False

    def test_known_thin_oi_still_rejected_on_a_real_quote(self, liq_env):
        """The quote gate must not shadow the OI gate (R17 behaviour kept)."""
        c = _contract(bid=1.80, ask=1.90, last=1.85, open_interest=3,
                      source="yahoo_delayed")
        assert c.is_liquid is False


# ---------------------------------------------------------------------------
# fail-direction-06 -- chain-level filter must agree with is_liquid
# ---------------------------------------------------------------------------

class TestFilterLiquidDropsQuotelessContracts:

    def _chain(self, calls, puts=()):
        return OptionsChain(symbol="SPY", underlying_price=650.0, expiry=EXPIRY,
                            calls=list(calls), puts=list(puts))

    def test_quoteless_call_dropped_quoted_kept(self, liq_env):
        quoteless = _contract(strike=660.0, bid=0.0, ask=0.0, last=2.50)
        quoted = _contract(strike=650.0, bid=1.80, ask=1.90, last=1.85)
        filtered = self._chain([quoteless, quoted]).filter_liquid(PROD_CFG)
        assert [c.strike for c in filtered.calls] == [650.0]

    def test_nan_quoted_put_dropped(self, liq_env):
        nan_put = _contract(strike=640.0, right="P", bid=float("nan"),
                            ask=float("nan"), last=1.20)
        good_put = _contract(strike=645.0, right="P", bid=1.10, ask=1.30,
                             last=1.20)
        filtered = self._chain([], [nan_put, good_put]).filter_liquid(PROD_CFG)
        assert [p.strike for p in filtered.puts] == [645.0]

    def test_both_filters_agree_on_every_contract(self, liq_env):
        """R16 invariant: is_liquid (contract) and filter_liquid (chain) must
        never disagree -- including on the quote-less shapes."""
        contracts = [
            _contract(strike=650.0, bid=1.80, ask=1.90, last=1.85),
            _contract(strike=655.0, bid=0.0, ask=0.0, last=2.50),
            _contract(strike=660.0, bid=0.0, ask=1.90, last=1.85),
            _contract(strike=665.0, bid=float("nan"), ask=float("nan"), last=1.85),
            _contract(strike=670.0, bid=1.60, ask=2.20, last=1.90),
        ]
        filtered = self._chain(contracts).filter_liquid(PROD_CFG)
        kept = {c.strike for c in filtered.calls}
        assert kept == {c.strike for c in contracts if c.is_liquid}
        assert kept == {650.0, 670.0}


# ---------------------------------------------------------------------------
# fail-direction-04 -- the touch stop must never fail silently
# ---------------------------------------------------------------------------

CONDOR_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])

# The finding's probe shape: a SELL leg with no 'strike' key -> KeyError at
# the parse, inside the block that used to swallow it at DEBUG.
MALFORMED_LEGS = json.dumps([
    {"right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"right": "C", "action": "SELL", "expiry": "2026-12-18"},
])


@dataclass
class FakeTrade:
    trade_id: str = "T-BROKEN"
    symbol: str = "SPY"
    legs: str = MALFORMED_LEGS
    entry_price: float = 1.00
    quantity: int = 1
    contract_type: str = "spread"
    strategy: str = "iron_condor"
    direction: object = None
    expiry: str | None = None
    strike: float | None = None
    entry_time: str = "2026-07-01T10:00:00"
    status: object = TradeStatus.FILLED

    def __post_init__(self):
        if self.direction is None:
            self.direction = SimpleNamespace(value="neutral")
        if self.expiry is None:
            self.expiry = (date.today() + timedelta(days=20)).isoformat()


class _RecordingLog:
    """Stands in for the module logger; records (level, event, kwargs)."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict]] = []

    def __getattr__(self, level):
        def _emit(event, **kw):
            self.calls.append((level, event, kw))
        return _emit

    def events(self, level=None):
        return [e for lvl, e, _ in self.calls if level is None or lvl == level]

    def kwargs_for(self, event):
        return [kw for _, e, kw in self.calls if e == event]


class _Feed:
    """Advancing (therefore 'fresh') quote source, per symbol."""

    def __init__(self, spots: dict[str, float]):
        self.spots = spots
        self._ts = datetime.now() - timedelta(seconds=2)

    async def get_quote(self, symbol):
        self._ts = self._ts + timedelta(seconds=1)
        spot = self.spots[symbol]
        return Quote(symbol=symbol, bid=round(spot - 0.01, 2),
                     ask=round(spot + 0.01, 2), last=spot, volume=1_000_000,
                     timestamp=self._ts)


def _manager(feed: _Feed, notify=None, trades=()) -> PortfolioManager:
    """Real PortfolioManager, collaborators mocked (test_credit_exits pattern)."""
    mgr = PortfolioManager.__new__(PortfolioManager)
    mgr._ibkr = MagicMock()
    mgr._ibkr.ib.portfolio.return_value = []
    mgr._state = MagicMock()
    mgr._state.get_high_water_mark.return_value = 0.0
    mgr._state.get_open_trades.return_value = list(trades)
    mgr._market_data = MagicMock()
    mgr._market_data.get_quote = feed.get_quote
    mgr._exit_config = ExitConfig(touch_confirm_ticks=2)
    mgr._earnings = None
    mgr._economic_cal = None
    mgr._notify_cb = notify
    mgr._touch_confirm = {}
    mgr._touch_fail_alerted = set()

    async def _vol_mult(symbol):
        return 1.0

    mgr._get_volatility_stop_multiplier = _vol_mult
    mgr._pdt_guard = MagicMock()
    mgr._pdt_guard.would_be_day_trade.return_value = False
    mgr._option_position_unrealized = lambda trade, marks: 10.0
    mgr._get_position_delta = lambda trade: None
    return mgr


@pytest.fixture(autouse=True)
def _touch_stop_on(monkeypatch):
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
    monkeypatch.delenv("AIT_SKIP_MACRO_EVENTS", raising=False)


@pytest.fixture
def rec_log(monkeypatch):
    log = _RecordingLog()
    monkeypatch.setattr(portfolio_mod, "log", log)
    return log


class TestTouchStopFailureIsSurfaced:

    async def test_malformed_legs_logs_error_not_debug(self, rec_log):
        mgr = _manager(_Feed({"SPY": 100.0}))
        trade = FakeTrade()
        status = await mgr._evaluate_position(trade)

        errors = [kw for lvl, ev, kw in rec_log.calls
                  if lvl == "error" and ev == "touch_stop_evaluation_failed"]
        assert len(errors) == 1
        assert errors[0]["trade_id"] == "T-BROKEN"
        assert errors[0]["error_type"] == "KeyError"
        # Pre-fix the ONLY evidence was a DEBUG line.
        assert "touch_check_failed" not in rec_log.events("debug")
        # The rest of the evaluation still produced a status for this trade.
        assert status is not None and status.trade_id == "T-BROKEN"

    async def test_malformed_legs_pages_once_per_outage(self, rec_log):
        pages = []

        async def _notify(msg):
            pages.append(msg)

        mgr = _manager(_Feed({"SPY": 100.0}), notify=_notify)
        trade = FakeTrade()
        for _ in range(4):  # four 30s ticks
            await mgr._evaluate_position(trade)

        assert len(pages) == 1
        assert "TOUCH STOP NOT EVALUATING" in pages[0]
        assert trade.trade_id in pages[0]
        # ...but every tick is still recorded in the log.
        assert len(rec_log.kwargs_for("touch_stop_evaluation_failed")) == 4

    async def test_recovery_then_new_outage_pages_again(self, rec_log):
        pages = []

        async def _notify(msg):
            pages.append(msg)

        mgr = _manager(_Feed({"SPY": 100.0}), notify=_notify)
        trade = FakeTrade()
        await mgr._evaluate_position(trade)
        assert len(pages) == 1

        trade.legs = CONDOR_LEGS          # row repaired: block evaluates clean
        await mgr._evaluate_position(trade)
        assert len(pages) == 1

        trade.legs = MALFORMED_LEGS       # a NEW outage must page again
        await mgr._evaluate_position(trade)
        assert len(pages) == 2

    async def test_report_pass_logs_but_does_not_page(self, rec_log):
        """A4 invariant: a read-only summary must not fire alerts or spend the
        once-per-outage page."""
        pages = []

        async def _notify(msg):
            pages.append(msg)

        mgr = _manager(_Feed({"SPY": 100.0}), notify=_notify)
        await mgr._evaluate_position(FakeTrade(), None, persist=False)

        assert pages == []
        assert len(rec_log.kwargs_for("touch_stop_evaluation_failed")) == 1

    async def test_missing_notifier_does_not_break_the_exit_path(self, rec_log):
        """No notifier wired (the default in several rigs): the failure must
        still log and the position must still be evaluated."""
        mgr = _manager(_Feed({"SPY": 100.0}), notify=None)
        status = await mgr._evaluate_position(FakeTrade())
        assert status is not None
        assert len(rec_log.kwargs_for("touch_stop_evaluation_failed")) == 1

    async def test_unparseable_legs_json_also_surfaces(self, rec_log):
        mgr = _manager(_Feed({"SPY": 100.0}))
        status = await mgr._evaluate_position(FakeTrade(legs="{not json"))
        errs = rec_log.kwargs_for("touch_stop_evaluation_failed")
        assert len(errs) == 1 and errs[0]["error_type"] == "JSONDecodeError"
        assert status is not None


class TestOneBadRowDoesNotKillTheLoop:

    async def test_second_position_still_evaluated_and_exits(self, rec_log):
        """The broken row is evaluated FIRST; the healthy QQQ condor behind it
        must still get its touch stop -- spot 97.50 is through the 98 short
        put, and a fresh quote fires on the first tick."""
        broken = FakeTrade(trade_id="T-BROKEN", symbol="SPY",
                           legs=MALFORMED_LEGS)
        healthy = FakeTrade(trade_id="T-HEALTHY", symbol="QQQ",
                            legs=CONDOR_LEGS)
        feed = _Feed({"SPY": 100.0, "QQQ": 97.50})
        mgr = _manager(feed, trades=[broken, healthy])

        statuses = await mgr.check_positions()

        assert [s.trade_id for s in statuses] == ["T-BROKEN", "T-HEALTHY"]
        healthy_status = statuses[1]
        assert healthy_status.should_exit
        assert "short_strike_touch" in healthy_status.exit_reason

        errs = rec_log.kwargs_for("touch_stop_evaluation_failed")
        assert [kw["trade_id"] for kw in errs] == ["T-BROKEN"]

    async def test_healthy_position_alone_is_unaffected(self, rec_log):
        """Control: no failure, no error log, touch still fires as before."""
        healthy = FakeTrade(trade_id="T-OK", symbol="QQQ", legs=CONDOR_LEGS)
        mgr = _manager(_Feed({"QQQ": 97.50}), trades=[healthy])

        statuses = await mgr.check_positions()

        assert len(statuses) == 1 and statuses[0].should_exit
        assert rec_log.kwargs_for("touch_stop_evaluation_failed") == []

"""W33 (R26 fix-set F1): the W7 exit rebuild, corrected.

W7 rebuilt exit pricing after the 08-31 incident and shipped with 20/20 green
tests. The 09-11 closes then showed seven consecutive band-rejected exits in
sixteen minutes. R26 traced that to seven defects, none of which W7's tests
could see because each one called a helper in isolation with a mark present:

  1. the broker's band reject never reached the parser. ib_insync 0.9.86
     classes error 202 as a WARNING and never appends it to ``trade.log``;
     ``ib.errorEvent`` is the only route and nothing subscribed to it. No
     ``exit_band_*`` key was ever written, so every band rule below was inert.
  2. the band clamp placed AT the band ("at or more aggressive than 1.745"
     rejects 1.75 and 1.745 alike).
  3. the no-quote fallback used the CEILING as the order price (1.5 x mark,
     0.94-1.14 above the band on 7 of 7 live attempts) and re-placed the same
     price every backoff.
  4. a legitimate 0.00 bid on a sold-back wing aborted the whole leg synthesis.
  5. a one-sided leg was priced from the wrong side of its spread.
  6. at low marks the ceiling overwrote a marketable synthesized limit with an
     unmarketable one.
  7. a stale mark (>900s) still priced the exit at the raw wing width.

Every test drives the real code: the real ib_insync ``Wrapper``, the real
``IBKRClient``, the real ``TradeExecutor.check_fills``, the real
``_close_multi_leg``. Numbers are the 09-11 session's own.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from ib_insync import Contract, Order, OrderStatus, Trade

import ait.bot.orchestrator as orch_mod
from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.broker.ibkr_client import IBKRClient
from ait.config.settings import load_settings
from ait.execution.executor import TradeExecutor

NAN = float("nan")

# The verbatim 09-11 rejection (logs/ait.log, 13:31:59Z).
IBKR_202_0911 = ("Order Canceled - reason:We cannot accept an order at a limit "
                 "price at or more aggressive than 3.69. Please submit your "
                 "order using a limit price that is closer to the current "
                 "market price of 3.17.")


# --------------------------------------------------------------------------
# 1. the band reject reaches the parser
# --------------------------------------------------------------------------
def _client() -> IBKRClient:
    c = IBKRClient(SimpleNamespace(host="127.0.0.1", port=4002, client_id=1,
                                   account="DU0", timeout=5, readonly=False))
    c._connected = True
    c._ib.isConnected = lambda: True
    return c


def _working_order(client: IBKRClient, order_id: int) -> Trade:
    """A Trade exactly as placeOrder leaves it in the wrapper's cache."""
    w = client.ib.wrapper
    order = Order(orderId=order_id, clientId=w.clientId, action="BUY",
                  orderType="LMT", lmtPrice=4.79, totalQuantity=1)
    trade = Trade(Contract(secType="BAG", symbol="SPY"), order,
                  OrderStatus(orderId=order_id, status="Submitted"), [], [])
    w.trades[w.orderKey(order.clientId, order.orderId, order.permId)] = trade
    return trade


def _broker_rejects(client: IBKRClient, order_id: int, text: str) -> None:
    """What the wire delivers for a band reject: error 202, then Cancelled."""
    w = client.ib.wrapper
    w.error(order_id, 202, text, "")
    w.orderStatus(order_id, "Cancelled", 0, 1, 0.0, 0, 0, 0.0, w.clientId,
                  "", 0.0)


class TestTheBandRejectReachesTheParser:
    def test_ib_insync_really_does_drop_the_202_from_the_order_log(self):
        """The premise, pinned: if a later ib_insync starts logging warnings
        on the order, this fails and the capture below can be simplified."""
        c = _client()
        t = _working_order(c, 77)
        _broker_rejects(c, 77, IBKR_202_0911)
        assert TradeExecutor.parse_price_band_reject(t.log) == (None, None)

    def test_the_client_captures_the_reject_by_order_id(self):
        c = _client()
        _working_order(c, 77)
        _broker_rejects(c, 77, IBKR_202_0911)
        msgs = c.order_error_messages(77)
        assert TradeExecutor.parse_price_band_reject(msgs) == (3.69, 3.17)
        assert c.order_error_messages(78) == []

    def test_feed_status_chatter_is_not_kept(self):
        c = _client()
        c.ib.wrapper.error(-1, 2104, "Market data farm connection is OK", "")
        c.ib.wrapper.error(-1, 1100, "Connectivity lost", "")
        assert c.order_error_messages(-1) == []

    def test_the_capture_is_bounded(self):
        c = _client()
        for oid in range(1, 1200):
            c.ib.wrapper.error(oid, 202, IBKR_202_0911, "")
        assert len(c._order_errors) <= 256
        assert c.order_error_messages(1199)          # newest survives
        assert c.order_error_messages(1) == []       # oldest evicted

    def test_a_cancelled_exit_persists_the_band_end_to_end(self, tmp_path):
        """wire -> Wrapper.error -> errorEvent -> client -> check_fills ->
        bot_state. At HEAD the key was never written."""
        st = StateManager(db_path=tmp_path / "ait_state.db")
        with sqlite3.connect(st._db_path) as conn:
            conn.execute(
                "INSERT INTO trades (trade_id, symbol, strategy, direction, "
                "status, entry_time, entry_price, quantity, contract_type) "
                "VALUES ('T-SPY','SPY','iron_condor','short','closing',?,"
                "5.32,1,'iron_condor')", (datetime.now().isoformat(),))
        c = _client()
        ex = TradeExecutor(c, st, MagicMock())
        _working_order(c, 77)
        ex.register_exit_order(order_id=77, trade_id="T-SPY",
                               exit_reason="take_profit", estimated_pnl=211.0)
        _broker_rejects(c, 77, IBKR_202_0911)
        # live shape: the cancel is observed ~31s after placement (check_fills
        # ignores exit orders younger than 10s)
        ex._pending_exit_orders[77].submitted_at -= 31

        asyncio.run(ex.check_fills())

        blob = json.loads(st.get_state("exit_band_T-SPY", "") or "{}")
        assert blob.get("band") == 3.69 and blob.get("market") == 3.17, blob
        assert 77 not in ex._pending_exit_orders


# --------------------------------------------------------------------------
# the _close_multi_leg rig
# --------------------------------------------------------------------------
LEGS = [
    {"strike": 600, "right": "P", "action": "BUY", "expiry": "20261016"},
    {"strike": 639, "right": "P", "action": "SELL", "expiry": "20261016"},
    {"strike": 700, "right": "C", "action": "SELL", "expiry": "20261016"},
    {"strike": 739, "right": "C", "action": "BUY", "expiry": "20261016"},
]
# conIds are handed out in leg order, so after the reversal:
#   1 = long put  -> SELL   2 = short put  -> BUY
#   3 = short call -> BUY   4 = long call  -> SELL


class _Store:
    """The state calls exit pricing makes, over plain dicts."""

    def __init__(self):
        self.kv: dict[str, str] = {}
        self.mark: dict | None = None

    def get_state(self, key, default=""):
        return self.kv.get(key, default)

    def set_state(self, key, value):
        self.kv[key] = value

    def get_position_mark(self, trade_id):
        return self.mark

    def update_trade_status(self, trade_id, status, from_statuses=None):
        # W45 (bot-day-05): _close_multi_leg/_close_single_leg CAS to CLOSING
        # immediately before placing. These pricing tests don't exercise that
        # transition itself, so always succeed, matching a real FILLED trade.
        return True


def _rig(*, leg_quotes, bag=(NAN, NAN), entry=5.32, mark_cost=None,
         mark_age_s=5, band=None, market=None, prior_attempts=0):
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._settings = load_settings()

    def _ticker(contract):
        cid = getattr(contract, "conId", None)
        if cid in leg_quotes:
            q = leg_quotes[cid]
            return None if q is None else SimpleNamespace(bid=q[0], ask=q[1])
        if bag is None:                                     # the BAG
            return None
        return SimpleNamespace(bid=bag[0], ask=bag[1])

    ib = MagicMock()
    ib.ticker = _ticker
    ibkr = MagicMock()
    ibkr.ib = ib
    n = {"i": 0}

    async def _qualify(contract):
        n["i"] += 1
        return SimpleNamespace(conId=n["i"])

    ibkr.qualify_contract = AsyncMock(side_effect=_qualify)
    ibkr.place_order = AsyncMock(
        return_value=SimpleNamespace(order=SimpleNamespace(orderId=42)))
    o._ibkr = ibkr

    st = _Store()
    if mark_cost is not None:
        st.mark = {
            "unrealized_pnl": (entry - mark_cost) * 100.0,
            "pnl_pct": 0.0,
            "mark_time": (datetime.now()
                          - timedelta(seconds=mark_age_s)).isoformat(),
        }
    if band is not None or market is not None:
        st.kv["exit_band_T-1"] = json.dumps(
            {"band": band, "market": market, "at": datetime.now().isoformat()})
    o._state = st
    o._send_notification = AsyncMock()
    o._alert_gate = lambda *a, **k: True
    if prior_attempts:
        import time as _time
        o._exit_noquote_attempts = {
            "T-1": {"n": prior_attempts, "at": _time.monotonic()}}
    trade = SimpleNamespace(trade_id="T-1", symbol="SPY",
                            strategy="iron_condor",
                            contract_type="iron_condor", quantity=1,
                            entry_price=entry, expiry="20261016")
    return o, ibkr, trade


@pytest.fixture(autouse=True)
def _instant_polls(monkeypatch):
    real = asyncio.sleep

    async def _no_wait(_s):
        await real(0)

    monkeypatch.setattr(orch_mod.asyncio, "sleep", _no_wait)


def _placed(o, ibkr, trade, legs=LEGS):
    asyncio.run(o._close_multi_leg(trade, legs))
    if not ibkr.place_order.called:
        return None
    return ibkr.place_order.call_args.args[1].lmtPrice


# Four live legs whose ask-side cost to close is 2.20 (1.40 + 1.30 - .30 - .20).
QUOTED = {1: (0.30, 0.40), 2: (1.20, 1.40), 3: (1.10, 1.30), 4: (0.20, 0.30)}
# The 09-11 shape: the bought-back short call never offers, so synthesis
# cannot price the structure and the no-quote branch is taken.
ONE_DEAD_BUY_LEG = {1: (0.30, 0.40), 2: (1.20, 1.40), 3: (NAN, NAN),
                    4: (0.20, 0.30)}


# --------------------------------------------------------------------------
# 2. the band clamp lands strictly under the band
# --------------------------------------------------------------------------
class TestTheBandCapIsStrictlyUnderTheBand:
    @pytest.mark.parametrize("band,market", [(1.745, 1.42), (2.00, 1.70)])
    def test_a_quoted_exit_is_placed_under_the_band(self, band, market):
        o, ibkr, trade = _rig(leg_quotes=QUOTED, mark_cost=1.40,
                              band=band, market=market)
        lmt = _placed(o, ibkr, trade)
        assert lmt is not None and lmt < band, (
            f"IBKR rejects 'at or more aggressive than {band}'; placed {lmt}")
        assert lmt >= band - 0.02, "clamped further under the band than a tick"

    def test_the_no_quote_fallback_is_placed_under_the_band_too(self):
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=3.19,
                              band=3.25, market=3.17, prior_attempts=3)
        lmt = _placed(o, ibkr, trade)
        assert lmt is not None and lmt < 3.25, lmt


class TestACapturedBandCannotPinALaterExit:
    """Bands were never captured before W33, so W7's one-hour lifetime was
    never exercised. Live, it would cap a stop below a moved market."""

    def _blob(self, age_s, band=3.69, market=3.17):
        return json.dumps({"band": band, "market": market,
                           "at": (datetime.now()
                                  - timedelta(seconds=age_s)).isoformat()})

    def test_a_band_expires_within_minutes_not_an_hour(self):
        o, _, trade = _rig(leg_quotes=QUOTED, mark_cost=3.19)
        o._state.kv["exit_band_T-1"] = self._blob(age_s=60)
        assert o._exit_band_cap(trade) == 3.69
        o._state.kv["exit_band_T-1"] = self._blob(age_s=6 * 60)
        assert o._exit_band_cap(trade) is None

    def test_a_band_the_mark_has_risen_through_is_ignored(self):
        """13:31 take-profit rejected at band 3.69; by 13:33 the structure is
        marked 4.50 and the legs offer 4.90 — a stop must be able to pay it."""
        legs = {1: (0.30, 0.40), 2: (2.60, 2.80), 3: (2.50, 2.70),
                4: (0.20, 0.30)}                       # ask-side cost 5.00
        o, ibkr, trade = _rig(leg_quotes=legs, mark_cost=4.50)
        o._state.kv["exit_band_T-1"] = self._blob(age_s=120)
        lmt = _placed(o, ibkr, trade)
        assert lmt is not None and lmt > 3.69, lmt


# --------------------------------------------------------------------------
# 3. the no-quote fallback starts at the mark and steps; the ceiling is a cap
# --------------------------------------------------------------------------
class TestTheNoQuoteFallbackStepsFromTheMark:
    def test_first_attempt_is_near_the_mark_not_at_the_ceiling(self):
        """09-11 13:31:59Z: mark 3.19 -> BUY LMT 4.79 against a 3.69 band."""
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=3.19)
        lmt = _placed(o, ibkr, trade)
        assert lmt is not None and lmt <= 3.59, lmt
        assert lmt > 3.19, "must still cross above the mark to be marketable"

    def test_each_unfilled_attempt_steps_toward_the_offer(self):
        """Four consecutive exits of the SAME position on one orchestrator."""
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=3.19)
        prices = [_placed(o, ibkr, trade) for _ in range(4)]
        assert prices == sorted(prices) and len(set(prices)) == 4, (
            f"HEAD re-placed the identical limit every backoff: {prices}")
        assert prices[0] == pytest.approx(3.29)
        assert prices[3] == pytest.approx(3.59)

    def test_a_new_exit_episode_starts_from_the_anchor_again(self):
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=3.19,
                              prior_attempts=5)
        o._exit_noquote_attempts["T-1"]["at"] -= 2 * 3600
        assert _placed(o, ibkr, trade) == pytest.approx(3.29)

    def test_stepping_never_passes_the_ceiling(self):
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=3.19,
                              prior_attempts=500)
        ceiling = o._exit_price_ceiling(3.19, 39.0, 0.10)
        assert _placed(o, ibkr, trade) == pytest.approx(ceiling)

    def test_all_seven_live_attempts_would_have_cleared_the_band(self):
        live = [(3.19, 3.69), (3.19, 3.65), (3.10, 3.61), (3.0764, 3.59),
                (3.0555, 3.64), (3.1103, 3.71), (3.1403, 3.68)]
        for mark, band in live:
            o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=mark)
            lmt = _placed(o, ibkr, trade)
            assert lmt < band, (mark, band, lmt)


# --------------------------------------------------------------------------
# 4 + 5. leg synthesis prices every leg from its conservative side
# --------------------------------------------------------------------------
def _synth(quotes, spec, credit_buyback=True):
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    ib = MagicMock()
    ib.ticker = lambda c: (None if quotes.get(c.conId) is None else
                           SimpleNamespace(bid=quotes[c.conId][0],
                                           ask=quotes[c.conId][1]))
    o._ibkr = SimpleNamespace(ib=ib)
    legs = [{"conId": cid, "action": act, "ratio": 1,
             "_contract": SimpleNamespace(conId=cid)} for cid, act in spec]
    return asyncio.run(o._synthesize_combo_cost(
        legs, poll_rounds=1, credit_buyback=credit_buyback))


CONDOR = [(1, "SELL"), (2, "BUY"), (3, "BUY"), (4, "SELL")]


class TestLegSynthesisIsConservative:
    def test_a_worthless_sold_back_wing_does_not_abort_the_synthesis(self):
        """09-11, con_id 909134039: bid 0.00, no ask, seven times running."""
        q = {1: (0.0, NAN), 2: (1.20, 1.40), 3: (1.10, 1.30), 4: (0.01, 0.02)}
        assert _synth(q, CONDOR) == pytest.approx(1.40 + 1.30 - 0.0 - 0.01)

    def test_a_zero_bid_wing_is_sold_at_zero_not_at_its_ask(self):
        """HEAD mapped bid 0.00 to 'missing' and then sold the wing at its
        ASK, understating the cost to close."""
        q = {1: (0.0, 0.05), 2: (1.20, 1.40), 3: (1.10, 1.30), 4: (0.0, 0.05)}
        assert _synth(q, CONDOR) == pytest.approx(2.70)

    def test_a_bidless_wing_contributes_nothing_to_a_credit_buyback(self):
        """IBKR's -1 says "nobody bids": we receive nothing for that wing."""
        for bidless in ((-1.0, 0.05), (-1.0, NAN), (-1.0, -1.0)):
            q = {1: bidless, 2: (1.20, 1.40), 3: (1.10, 1.30), 4: (0.20, 0.30)}
            assert _synth(q, CONDOR) == pytest.approx(1.40 + 1.30 - 0.20), bidless

    def test_missing_data_is_not_a_quote(self):
        """NaN / a silent ticker is the feed saying nothing, not the market
        saying zero — a dead feed must not synthesize a price from half the
        legs."""
        for silent in ((NAN, 0.05), (NAN, NAN), None):
            q = {1: silent, 2: (1.20, 1.40), 3: (1.10, 1.30), 4: (0.20, 0.30)}
            assert _synth(q, CONDOR) is None, silent

    def test_a_long_structure_is_never_priced_at_zero(self):
        """Closing a long straddle SELLS both legs: there the sold legs are
        the value, so 'bidless = 0' would give the position away."""
        q = {1: (-1.0, 0.30), 2: (-1.0, 0.40)}
        assert _synth(q, [(1, "SELL"), (2, "SELL")],
                      credit_buyback=False) is None

    def test_a_leg_we_must_buy_is_never_priced_off_its_bid(self):
        """HEAD: short leg 1.00 x (no ask) -> bought back 'at' 1.00."""
        q = {1: (0.40, 0.55), 2: (1.00, NAN)}
        assert _synth(q, [(1, "SELL"), (2, "BUY")]) is None

    def test_fully_quoted_legs_are_unchanged(self):
        assert _synth(QUOTED, CONDOR) == pytest.approx(2.20)


# --------------------------------------------------------------------------
# 6. a marketable quote survives a low mark
# --------------------------------------------------------------------------
LOW_MARK_LEGS = {1: (0.02, 0.06), 2: (0.30, 0.40), 3: (0.62, 0.72),
                 4: (0.02, 0.06)}     # ask-side cost to close: 1.08


class TestALowMarkDoesNotUnpriceAMarketableExit:
    def test_the_synthesized_limit_stays_marketable(self):
        """HEAD: ceiling(0.60) = 0.90 overwrote the 1.18 synthesized limit."""
        o, ibkr, trade = _rig(leg_quotes=LOW_MARK_LEGS, mark_cost=0.60,
                              entry=6.35)
        lmt = _placed(o, ibkr, trade)
        assert lmt is not None and lmt >= 1.08, lmt

    def test_headroom_is_config_homed(self):
        s = load_settings()
        assert 0.10 <= s.exit.exit_min_ceiling_headroom <= 2.0

    def test_the_garbage_quote_guard_still_binds(self):
        """R13's incident shape: a 9.90 ask on a structure marked near 1.00."""
        o, ibkr, trade = _rig(leg_quotes={}, bag=(0.90, 9.90), mark_cost=1.00)
        lmt = _placed(o, ibkr, trade)
        assert lmt is not None and lmt <= 1.00 + max(
            0.5, o._settings.exit.exit_min_ceiling_headroom) + 0.001, lmt

    def test_the_incident_ceilings_are_unchanged(self):
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._settings = load_settings()
        assert o._exit_price_ceiling(1.4006, 39.0, 0.10) == pytest.approx(2.10, abs=0.01)
        assert o._exit_price_ceiling(8.76, 55.0, 0.10) == pytest.approx(13.14, abs=0.01)
        assert o._exit_price_ceiling(100.0, 60.0, 0.10) == 60.0


# --------------------------------------------------------------------------
# 7. a stale mark never prices the exit at the wing width
# --------------------------------------------------------------------------
class TestAStaleMarkNeverPricesAtTheWingWidth:
    def test_sixteen_minute_old_mark_is_an_anchor_not_a_blank(self):
        """HEAD: a mark 16 min old -> BUY LMT 39.00 (the raw width); the
        same position marked two minutes fresher -> 0.90."""
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=0.80,
                              mark_age_s=16 * 60, entry=6.35)
        lmt = _placed(o, ibkr, trade)
        assert lmt == pytest.approx(0.90), lmt       # last mark + the cross
        assert o._send_notification.await_count == 1

    def test_a_stale_anchor_may_step_past_its_own_ceiling_but_not_the_blind_cap(self):
        """The market may have moved since the mark: the cap widens to what a
        blind exit may pay (k x entry credit), never to the wing width."""
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=0.80,
                              mark_age_s=16 * 60, entry=6.35,
                              prior_attempts=5000)
        lmt = _placed(o, ibkr, trade)
        k = o._settings.exit.exit_blind_credit_multiple
        assert lmt == pytest.approx(min(39.0, k * 6.35)), lmt
        assert lmt < 39.0

    def test_no_mark_at_all_is_anchored_on_the_entry_credit(self):
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=None,
                              entry=5.32)
        lmt = _placed(o, ibkr, trade)
        assert lmt == pytest.approx(5.42), (
            f"HEAD placed the structural max loss (39.00); got {lmt}")
        assert o._send_notification.await_count == 1

    def test_nothing_to_anchor_on_refuses_and_pages(self):
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=None,
                              entry=0.0)
        assert _placed(o, ibkr, trade) is None
        assert o._send_notification.await_count == 1


# --------------------------------------------------------------------------
# riders: the fill-quality tuple and the poll telemetry
# --------------------------------------------------------------------------
class TestExitQualityIsNotInheritedFromAnotherTrade:
    def test_the_no_quote_branch_does_not_keep_a_previous_exits_numbers(self):
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=3.19,
                              bag=None)           # R26 probe B: nothing ticks
        o._last_exit_quality = (11.11, 22.22, 33.33)
        _placed(o, ibkr, trade)
        q = o._last_exit_quality
        assert q[1] == 0.0 and q[2] == 0.0, q          # no quote was seen
        assert q[0] == pytest.approx(3.19, abs=0.01)   # this trade's mark


class TestTheExitPollIsInstrumented:
    def test_what_the_bag_and_the_legs_actually_showed_is_logged(self):
        import structlog
        o, ibkr, trade = _rig(leg_quotes=ONE_DEAD_BUY_LEG, mark_cost=3.19)
        with structlog.testing.capture_logs() as logs:
            _placed(o, ibkr, trade)
        poll = [e for e in logs if e.get("event") == "combo_exit_quote_poll"]
        assert len(poll) == 1, [e.get("event") for e in logs]
        assert poll[0]["bag"] == "nan/nan"
        assert poll[0]["legs"]["3"] == "nan/nan"
        assert poll[0]["legs"]["2"] == "1.2/1.4"

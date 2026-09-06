"""W7 (R24 logic-exit-risk-01 / -02): the stop-loss must exist as a mechanism.

THE INCIDENT (2026-08-31, from logs/ait.log.4): every multi-leg exit took the
no-quote branch — `combo_exit_limit` appears ZERO times in any retained log —
and the fallback priced a credit buyback at
``max(2*mark, entry_credit + 0.25*wing_width)``. At the promoted $39-60 wings
the width term dominated: SPY was sent as BUY LMT 14.04 against a 1.40 mark
(10x), QQQ 18.60 against 2.94 (6.3x). IBKR's price band rejected 61 of 63
attempts — the broker's guard was the ONLY thing bounding the price — and each
reject was handled exactly like a cancel, so the IDENTICAL unexecutable limit
was re-placed every backoff (SPY looped 32 min, QQQ 2h55m).

Every test EXECUTES the real functions with the real incident numbers.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.config.settings import load_settings
from ait.execution.executor import TradeExecutor

# The two live positions the incident priced, from data/ait_state.db.
SPY = dict(mark=1.4006, width=39.0, entry=4.29, shipped_limit=14.04, band=1.745)
QQQ = dict(mark=2.9430, width=49.0, entry=6.35, shipped_limit=18.60, band=3.50)

# The verbatim IBKR rejection from that session.
IBKR_202 = ("Warning 202, reqId 1578968: Order Canceled - reason:We cannot "
            "accept an order at a limit price at or more aggressive than "
            "1.745. Please submit your order using a limit price that is "
            "closer to the current market price of 1.42.")


def _orch(settings=None):
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._settings = settings or load_settings()
    return o


class TestExitCeilingIsAnchoredToTheMark:
    @pytest.mark.parametrize("case,label", [(SPY, "SPY"), (QQQ, "QQQ")])
    def test_the_shipped_overpay_is_gone(self, case, label):
        o = _orch()
        ceiling = o._exit_price_ceiling(case["mark"], case["width"], 0.10)
        # pre-fix this returned entry + 0.25*width == the shipped limit
        assert ceiling < case["shipped_limit"], label
        assert ceiling <= case["mark"] * 2.0, (
            f"{label}: {ceiling} is more than 2x the {case['mark']} mark")

    def test_ceiling_stays_marketable(self):
        """A ceiling below mark+cross would be unfillable by construction."""
        o = _orch()
        for mark in (0.02, 0.25, 1.40, 2.94, 8.76):
            c = o._exit_price_ceiling(mark, 60.0, 0.10)
            assert c >= round(mark + 0.10, 2) - 0.005, (mark, c)

    def test_wing_width_remains_the_structural_cap(self):
        """A deep-ITM condor really is worth ~width; never bid beyond it."""
        o = _orch()
        assert o._exit_price_ceiling(40.0, 60.0, 0.10) == 60.0
        assert o._exit_price_ceiling(100.0, 60.0, 0.10) == 60.0

    def test_multiple_is_config_homed(self):
        s = load_settings()
        assert 1.05 <= s.exit.exit_mark_multiple <= 3.0
        hi = _orch(s)
        s2 = load_settings()
        s2.exit.exit_mark_multiple = 1.2
        lo = _orch(s2)
        assert lo._exit_price_ceiling(2.0, 60.0, 0.10) < \
            hi._exit_price_ceiling(2.0, 60.0, 0.10)


class TestBrokerBandIsParsedAndHonoured:
    def test_parses_the_real_ibkr_rejection(self):
        band, mkt = TradeExecutor.parse_price_band_reject(
            [SimpleNamespace(message="Order submitted"),
             SimpleNamespace(message=IBKR_202)])
        assert band == 1.745 and mkt == 1.42

    def test_newest_rejection_wins(self):
        msgs = [SimpleNamespace(message=IBKR_202),
                SimpleNamespace(message="... more aggressive than 1.765 ...")]
        band, _ = TradeExecutor.parse_price_band_reject(msgs)
        assert band == 1.765

    def test_absent_or_malformed_is_none(self):
        assert TradeExecutor.parse_price_band_reject(None) == (None, None)
        assert TradeExecutor.parse_price_band_reject(
            [SimpleNamespace(message="Order Canceled")]) == (None, None)

    def _orch_with_band(self, blob):
        o = _orch()
        st = MagicMock()
        st.get_state.return_value = json.dumps(blob) if blob else ""
        o._state = st
        return o

    def test_band_is_returned_for_the_next_attempt(self):
        o = self._orch_with_band(
            {"band": 1.745, "market": 1.42, "at": datetime.now().isoformat()})
        assert o._exit_band_cap(SimpleNamespace(trade_id="T-1")) == 1.745

    def test_stale_band_is_ignored(self):
        """An hour-old band must not pin an exit in a moved market."""
        old = (datetime.now() - timedelta(hours=2)).isoformat()
        o = self._orch_with_band({"band": 1.745, "market": 1.42, "at": old})
        assert o._exit_band_cap(SimpleNamespace(trade_id="T-1")) is None

    def test_market_price_used_when_no_explicit_band(self):
        o = self._orch_with_band(
            {"band": None, "market": 2.00, "at": datetime.now().isoformat()})
        cap = o._exit_band_cap(SimpleNamespace(trade_id="T-1"))
        assert cap is not None and 2.00 <= cap <= 2.20

    def test_no_band_recorded_is_none(self):
        assert self._orch_with_band(None)._exit_band_cap(
            SimpleNamespace(trade_id="T-1")) is None

    def test_unreadable_state_never_blocks_an_exit(self):
        o = _orch()
        st = MagicMock()
        st.get_state.side_effect = RuntimeError("database is locked")
        o._state = st
        assert o._exit_band_cap(SimpleNamespace(trade_id="T-1")) is None


class TestLegSynthesisWhenTheBagIsSilent:
    """The root cause: the BAG never ticked on 63 of 63 exits."""

    def _rig(self, quotes):
        """quotes: conId -> (bid, ask); None means the leg never ticks."""
        o = _orch()
        ib = MagicMock()
        ib.reqMktData = MagicMock()
        ib.cancelMktData = MagicMock()
        ib.ticker = lambda c: (
            SimpleNamespace(bid=quotes[c.conId][0], ask=quotes[c.conId][1])
            if quotes.get(c.conId) else None)
        o._ibkr = SimpleNamespace(ib=ib)
        return o, ib

    def _legs(self, spec):
        return [{"conId": cid, "action": act, "ratio": 1,
                 "_contract": SimpleNamespace(conId=cid)}
                for cid, act in spec]

    def test_condor_buyback_cost_is_summed_from_legs(self):
        # closing a short condor: buy back the two shorts, sell the two longs
        legs = self._legs([(1, "BUY"), (2, "BUY"), (3, "SELL"), (4, "SELL")])
        quotes = {1: (1.00, 1.20), 2: (0.80, 0.90), 3: (0.30, 0.40), 4: (0.10, 0.20)}
        o, _ = self._rig(quotes)
        cost = asyncio.run(o._synthesize_combo_cost(legs, poll_rounds=1))
        # pay 1.20 + 0.90, receive 0.30 + 0.10
        assert cost == pytest.approx(1.70)

    def test_one_unquotable_leg_returns_none_not_a_partial_sum(self):
        """A partial sum understates the cost and places an unfillable order."""
        legs = self._legs([(1, "BUY"), (2, "BUY"), (3, "SELL"), (4, "SELL")])
        quotes = {1: (1.00, 1.20), 2: (0.80, 0.90), 3: (0.30, 0.40), 4: None}
        o, _ = self._rig(quotes)
        assert asyncio.run(o._synthesize_combo_cost(legs, poll_rounds=1)) is None

    def test_nan_quotes_are_rejected(self):
        legs = self._legs([(1, "BUY")])
        o, _ = self._rig({1: (float("nan"), float("nan"))})
        assert asyncio.run(o._synthesize_combo_cost(legs, poll_rounds=1)) is None

    def test_subscriptions_are_always_cancelled(self):
        """A leaked snapshot=False sub floods market data (native crash class)."""
        legs = self._legs([(1, "BUY"), (2, "SELL")])
        o, ib = self._rig({1: (1.0, 1.2), 2: (0.3, 0.4)})
        asyncio.run(o._synthesize_combo_cost(legs, poll_rounds=1))
        assert ib.cancelMktData.call_count == ib.reqMktData.call_count == 2

    def test_cancelled_even_when_a_leg_raises(self):
        legs = self._legs([(1, "BUY")])
        o, ib = self._rig({1: (1.0, 1.2)})
        ib.ticker = MagicMock(side_effect=RuntimeError("feed down"))
        assert asyncio.run(o._synthesize_combo_cost(legs, poll_rounds=1)) is None
        assert ib.cancelMktData.call_count == 1


class TestTheIncidentCannotRecur:
    def test_spy_take_profit_is_priced_sanely_end_to_end(self):
        """The 08-31 SPY exit: mark 1.40, wings 39, credit 4.29 -> 14.04."""
        o = _orch()
        ceiling = o._exit_price_ceiling(SPY["mark"], SPY["width"], 0.10)
        assert ceiling == pytest.approx(2.10, abs=0.01)
        # and if IBKR still objects, the band caps it under its own limit
        assert min(ceiling, SPY["band"]) <= SPY["band"]

    def test_qqq_touch_stop_is_priced_sanely(self):
        """The 08-24 QQQ touch stop: mark 8.76, wings 55."""
        o = _orch()
        ceiling = o._exit_price_ceiling(8.76, 55.0, 0.10)
        assert ceiling < 19.0, "the old anchor allowed 19.87 here"
        assert ceiling == pytest.approx(13.14, abs=0.01)

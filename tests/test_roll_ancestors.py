"""W54a: StateManager.count_roll_ancestors -- how many times an OPEN position
has already been rolled, walking rolled_from_trade_id BACKWARD to its
origin. Deliberately NOT go_live.py's fold_roll_chains, confirmed by direct
read this session that function only walks CLOSED rows FORWARD for
historical accounting and drops any chain ending at a still-open trade --
exactly the case this exists for.

Real temp-DB StateManager throughout -- never mocked. No inspect.getsource.
"""
from __future__ import annotations

import json

from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
from ait.execution.roll_booking import cas_roll_closing_trade, seed_roll_opening_trade
from ait.execution.roll_selector import RollTarget
from ait.data.options_chain import OptionContract

LEGS = json.dumps([
    {"strike": 745.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 735.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
])


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _trade(trade_id) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id, symbol="SPY", strategy="iron_condor",
        direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
        entry_time="2026-09-27T10:00:00", entry_price=1.50, quantity=1,
        contract_type="iron_condor", legs=LEGS,
    )


def _roll_target(short=720.0, long=710.0, expiry="2027-01-15") -> RollTarget:
    return RollTarget(
        new_short=OptionContract(symbol="SPY", expiry=expiry, strike=short, right="P",
                                 bid=0, ask=0, last=0, volume=0, open_interest=0,
                                 implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        new_long=OptionContract(symbol="SPY", expiry=expiry, strike=long, right="P",
                                bid=0, ask=0, last=0, volume=0, open_interest=0,
                                implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        expiry=expiry,
    )


def _roll(state, closing_trade_id, new_id_suffix) -> TradeRecord:
    """Perform one roll hop: CAS closing_trade_id to CLOSING, seed a fresh
    opening trade linked to it (mirrors _execute_roll's own sequence,
    minus placement), then promote the new trade straight to FILLED (a
    plain status transition, standing in for a real book_roll_fill) so the
    NEXT hop in a multi-hop chain can CAS it in turn."""
    closing = state.get_trade_by_id(closing_trade_id)
    assert cas_roll_closing_trade(state, closing_trade_id) is True
    closing = state.get_trade_by_id(closing_trade_id)
    opening_legs = [
        {"strike": 720.0, "right": "P", "action": "SELL", "expiry": "2027-01-15"},
        {"strike": 710.0, "right": "P", "action": "BUY", "expiry": "2027-01-15"},
    ]
    opened = seed_roll_opening_trade(
        state, closing, opening_legs, _roll_target(), 1, entry_price_estimate=0.75,
    )
    assert state.transition(opened.trade_id, (TradeStatus.PENDING,), TradeStatus.FILLED)
    return state.get_trade_by_id(opened.trade_id)


class TestCountRollAncestors:
    def test_a_never_rolled_trade_has_depth_zero(self, tmp_path):
        st = _state(tmp_path)
        a = _trade("T-A")
        st.record_trade(a)
        assert st.count_roll_ancestors("T-A") == 0

    def test_a_missing_trade_id_has_depth_zero(self, tmp_path):
        st = _state(tmp_path)
        assert st.count_roll_ancestors("T-DOES-NOT-EXIST") == 0

    def test_one_hop_has_depth_one(self, tmp_path):
        st = _state(tmp_path)
        st.record_trade(_trade("T-A"))
        b = _roll(st, "T-A", "B")
        assert st.count_roll_ancestors(b.trade_id) == 1
        # The origin itself is still depth 0 -- ancestors are counted FROM
        # the queried trade, not a property of the chain as a whole.
        assert st.count_roll_ancestors("T-A") == 0

    def test_three_hop_chain_has_depth_two_at_the_open_leaf(self, tmp_path):
        st = _state(tmp_path)
        st.record_trade(_trade("T-A"))
        b = _roll(st, "T-A", "B")
        c = _roll(st, b.trade_id, "C")
        assert st.count_roll_ancestors(c.trade_id) == 2
        assert st.count_roll_ancestors(b.trade_id) == 1

    def test_depth_is_bounded_by_max_depth(self, tmp_path):
        st = _state(tmp_path)
        st.record_trade(_trade("T-A"))
        current = _roll(st, "T-A", "B")
        for _ in range(4):
            current = _roll(st, current.trade_id, "next")
        # 5 real hops from the origin, but bounded at max_depth=2.
        assert st.count_roll_ancestors(current.trade_id, max_depth=2) == 2
        assert st.count_roll_ancestors(current.trade_id, max_depth=20) == 5

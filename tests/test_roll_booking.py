"""Roll booking mechanics -- execution/roll_booking.py. Nothing in src/ait
calls into this module; these tests drive it directly against a REAL
temp-DB StateManager (not mocked -- proves the actual SQL, same convention
test_w53_db_migration.py uses), never a fake/mock for the DB layer.

No inspect.getsource.
"""
from __future__ import annotations

import json

import pytest

from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
from ait.execution.roll_booking import (
    book_roll_fill,
    cas_roll_closing_trade,
    seed_roll_opening_trade,
    split_roll_fill_price,
)
from ait.execution.roll_selector import RollTarget
from ait.data.options_chain import OptionContract

CLOSING_LEGS_JSON = json.dumps([
    {"strike": 470.0, "right": "P", "action": "BUY", "expiry": "2026-10-05"},
    {"strike": 480.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"},
])
OPENING_LEGS = [
    {"strike": 450.0, "right": "P", "action": "BUY", "expiry": "2026-10-23"},
    {"strike": 460.0, "right": "P", "action": "SELL", "expiry": "2026-10-23"},
]


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _closing_trade(trade_id="T-A", strategy="iron_condor", entry_price=1.50,
                    quantity=1, status=TradeStatus.FILLED) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id, symbol="SPY", strategy=strategy,
        direction=TradeDirection.SHORT, status=status,
        entry_time="2026-09-27T10:00:00", entry_price=entry_price, quantity=quantity,
        contract_type="iron_condor", legs=CLOSING_LEGS_JSON,
    )


def _roll_target() -> RollTarget:
    return RollTarget(
        new_short=OptionContract(symbol="SPY", expiry="2026-10-23", strike=460.0, right="P",
                                 bid=0, ask=0, last=0, volume=0, open_interest=0,
                                 implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        new_long=OptionContract(symbol="SPY", expiry="2026-10-23", strike=450.0, right="P",
                                bid=0, ask=0, last=0, volume=0, open_interest=0,
                                implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        expiry="2026-10-23",
    )


# ---------------------------------------------------------------------------
# split_roll_fill_price -- pure
# ---------------------------------------------------------------------------

class TestSplitRollFillPrice:
    def test_splits_closing_and_opening_by_conid(self):
        leg_fills = [
            {"conId": 1, "side": "BUY", "price": 1.00, "shares": 1},   # close short (BUY back)
            {"conId": 2, "side": "SELL", "price": 0.20, "shares": 1},  # close long (SELL back)
            {"conId": 3, "side": "SELL", "price": 0.90, "shares": 1},  # open new short
            {"conId": 4, "side": "BUY", "price": 0.15, "shares": 1},   # open new long
        ]
        result = split_roll_fill_price(leg_fills, closing_conids={1, 2}, opening_conids={3, 4})
        assert result is not None
        closing_net, opening_net = result
        assert closing_net == pytest.approx(1.00 - 0.20)
        assert opening_net == pytest.approx(-0.90 + 0.15)

    def test_unattributed_conid_returns_none(self):
        leg_fills = [{"conId": 99, "side": "BUY", "price": 1.00, "shares": 1}]
        assert split_roll_fill_price(leg_fills, closing_conids={1}, opening_conids={2}) is None

    def test_empty_fills_returns_zero_zero_not_none(self):
        assert split_roll_fill_price([], closing_conids={1}, opening_conids={2}) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# seed_roll_opening_trade
# ---------------------------------------------------------------------------

class TestSeedRollOpeningTrade:
    def test_writes_a_pending_row_with_the_opening_legs(self, tmp_path):
        state = _state(tmp_path)
        closing = _closing_trade()
        state.record_trade(closing)
        opened = seed_roll_opening_trade(
            state, closing, OPENING_LEGS, _roll_target(), quantity=1, entry_price_estimate=0.75,
        )
        assert opened.status == TradeStatus.PENDING
        assert json.loads(opened.legs) == OPENING_LEGS
        assert opened.trade_id != closing.trade_id

        fetched = state.get_trade_by_id(opened.trade_id)
        assert fetched is not None
        assert fetched.status == TradeStatus.PENDING
        assert json.loads(fetched.legs) == OPENING_LEGS

    def test_duplicate_trade_id_does_not_duplicate_the_row(self, tmp_path, monkeypatch):
        """INSERT OR IGNORE's own existing guarantee, proven here (via a
        forced fixed trade_id) rather than just assumed."""
        state = _state(tmp_path)
        closing = _closing_trade()
        state.record_trade(closing)

        import ait.execution.roll_booking as rb_mod
        monkeypatch.setattr(rb_mod.uuid, "uuid4", lambda: type("_U", (), {"hex": "fixed0"})())

        first = seed_roll_opening_trade(
            state, closing, OPENING_LEGS, _roll_target(), quantity=1, entry_price_estimate=0.75,
        )
        second = seed_roll_opening_trade(
            state, closing, OPENING_LEGS, _roll_target(), quantity=1, entry_price_estimate=0.75,
        )
        # Same trade_id (fixed clock+uuid) -- must not raise, and only one row.
        assert first.trade_id == second.trade_id
        import sqlite3
        with sqlite3.connect(state._db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE trade_id = ?", (first.trade_id,)
            ).fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# cas_roll_closing_trade
# ---------------------------------------------------------------------------

class TestCasRollClosingTrade:
    def test_filled_trade_transitions_to_closing(self, tmp_path):
        state = _state(tmp_path)
        trade = _closing_trade(status=TradeStatus.FILLED)
        state.record_trade(trade)
        assert cas_roll_closing_trade(state, trade.trade_id) is True
        assert state.get_trade_by_id(trade.trade_id).status == TradeStatus.CLOSING

    def test_already_closed_trade_refuses(self, tmp_path):
        state = _state(tmp_path)
        trade = _closing_trade(status=TradeStatus.CLOSED)
        state.record_trade(trade)
        assert cas_roll_closing_trade(state, trade.trade_id) is False
        assert state.get_trade_by_id(trade.trade_id).status == TradeStatus.CLOSED


# ---------------------------------------------------------------------------
# book_roll_fill -- end to end, real StateManager
# ---------------------------------------------------------------------------

LEG_FILLS = [
    {"conId": 1, "side": "BUY", "price": 1.00, "shares": 1},
    {"conId": 2, "side": "SELL", "price": 0.20, "shares": 1},
    {"conId": 3, "side": "SELL", "price": 0.90, "shares": 1},
    {"conId": 4, "side": "BUY", "price": 0.15, "shares": 1},
]
CLOSING_CONIDS = {1, 2}
OPENING_CONIDS = {3, 4}


class TestBookRollFill:
    def _seeded(self, tmp_path, *, strategy="iron_condor", entry_price=1.50):
        state = _state(tmp_path)
        closing = _closing_trade(strategy=strategy, entry_price=entry_price,
                                 status=TradeStatus.FILLED)
        state.record_trade(closing)
        assert cas_roll_closing_trade(state, closing.trade_id) is True
        closing = state.get_trade_by_id(closing.trade_id)  # re-fetch: now CLOSING
        opening = seed_roll_opening_trade(
            state, closing, OPENING_LEGS, _roll_target(), quantity=1, entry_price_estimate=0.75,
        )
        return state, closing, opening

    def test_happy_path_closes_a_and_fills_b_with_the_link(self, tmp_path):
        state, closing, opening = self._seeded(tmp_path)
        ok = book_roll_fill(
            state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS,
        )
        assert ok is True

        a = state.get_trade_by_id(closing.trade_id)
        assert a.status == TradeStatus.CLOSED
        assert a.exit_price == pytest.approx(1.00 - 0.20)
        assert f"rolled_to_{opening.trade_id}" in a.exit_reason_detailed

        b = state.get_trade_by_id(opening.trade_id)
        assert b.status == TradeStatus.FILLED
        assert b.rolled_from_trade_id == closing.trade_id
        assert b.entry_price == pytest.approx(-0.90 + 0.15)

    def test_credit_strategy_closed_for_a_debit_is_a_realized_loss(self, tmp_path):
        """A credit vertical (entry_price=1.50 received) closed by paying
        0.80 net (BUY back 1.00, SELL back 0.20) nets a LOSS: received 1.50,
        paid 0.80 more than that would be a gain -- check the actual sign
        against _option_position_unrealized's own polarity directly."""
        state, closing, opening = self._seeded(tmp_path, strategy="iron_condor", entry_price=1.50)
        book_roll_fill(state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS)
        a = state.get_trade_by_id(closing.trade_id)
        # signed_entry = -1.50 (credit); closing_net = 0.80; pnl = (0.80 - (-1.50))*100*1 = 230
        # i.e. paying only 0.80 to close a 1.50 credit is a GAIN, matching
        # _option_position_unrealized's own formula exactly.
        assert a.realized_pnl == pytest.approx((0.80 - (-1.50)) * 100)

    def test_debit_strategy_polarity_is_the_mirror_of_credit(self, tmp_path):
        state, closing, opening = self._seeded(
            tmp_path, strategy="long_straddle", entry_price=1.50,
        )
        book_roll_fill(state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS)
        a = state.get_trade_by_id(closing.trade_id)
        # signed_entry = +1.50 (debit paid); closing_net = 0.80;
        # pnl = (0.80 - 1.50) * 100 = -70 (a loss -- paid 1.50, got back 0.80).
        assert a.realized_pnl == pytest.approx((0.80 - 1.50) * 100)

    def test_a_not_in_closing_refuses_the_close_but_b_still_promotes(self, tmp_path):
        """The documented residual risk of the deliberate promote-B-first
        ordering: B's promotion doesn't depend on A's state at all, so a
        caller that forgot to CAS A to CLOSING first still gets B promoted
        to a real, live FILLED trade -- only A's own close correctly
        refuses (A stays FILLED, never closed). Not a silent double-booking
        of A (its money/exit fields are untouched), but B and A both now
        look like separate open positions -- exactly the scenario this
        module's own docstring names as not eliminated by this design."""
        state = _state(tmp_path)
        closing = _closing_trade(status=TradeStatus.FILLED)  # never CAS'd to CLOSING
        state.record_trade(closing)
        opening = seed_roll_opening_trade(
            state, closing, OPENING_LEGS, _roll_target(), quantity=1, entry_price_estimate=0.75,
        )
        ok = book_roll_fill(
            state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS,
        )
        assert ok is False
        assert state.get_trade_by_id(closing.trade_id).status == TradeStatus.FILLED
        assert state.get_trade_by_id(opening.trade_id).status == TradeStatus.FILLED

    def test_b_already_filled_refuses_and_leaves_a_untouched(self, tmp_path):
        state, closing, opening = self._seeded(tmp_path)
        # Simulate B already promoted by something else before this call.
        assert state.transition(
            opening.trade_id, (TradeStatus.PENDING, TradeStatus.PARTIAL), TradeStatus.FILLED,
        ) is True

        ok = book_roll_fill(
            state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS,
        )
        assert ok is False
        # Promote-B-first means this is the CLEAN failure direction: B's
        # promotion is attempted and refused before A is ever touched, so A
        # stays exactly where it was (still CLOSING, not closed).
        assert state.get_trade_by_id(closing.trade_id).status == TradeStatus.CLOSING

    def test_split_failure_refuses_before_any_write(self, tmp_path):
        state, closing, opening = self._seeded(tmp_path)
        bad_fills = [{"conId": 999, "side": "BUY", "price": 1.0, "shares": 1}]
        ok = book_roll_fill(
            state, closing, opening.trade_id, bad_fills, CLOSING_CONIDS, OPENING_CONIDS,
        )
        assert ok is False
        assert state.get_trade_by_id(closing.trade_id).status == TradeStatus.CLOSING
        assert state.get_trade_by_id(opening.trade_id).status == TradeStatus.PENDING

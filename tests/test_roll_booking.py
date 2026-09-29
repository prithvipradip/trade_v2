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
    _persistable_legs,
    _promote_and_link_opening_trade,
    book_roll_fill,
    cas_roll_closing_trade,
    parse_roll_order_ref,
    roll_order_ref,
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
# W57: a plain-vertical roll carries no other side -- most existing tests
# use this (carried_legs=[], carried_value=0.0). A dedicated section below
# covers a 4-leg IC's carried call vertical.
NO_CARRIED: list[dict] = []


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _closing_trade(trade_id="T-A", strategy="iron_condor", entry_price=1.50,
                    quantity=1, status=TradeStatus.FILLED, capital_at_risk=0.0) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id, symbol="SPY", strategy=strategy,
        direction=TradeDirection.SHORT, status=status,
        entry_time="2026-09-27T10:00:00", entry_price=entry_price, quantity=quantity,
        contract_type="iron_condor", legs=CLOSING_LEGS_JSON, capital_at_risk=capital_at_risk,
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
# roll_order_ref / parse_roll_order_ref -- the reconciler's marker
# ---------------------------------------------------------------------------

class TestRollOrderRef:
    def test_round_trips(self):
        ref = roll_order_ref("T-A", "T-B")
        assert parse_roll_order_ref(ref) == ("T-A", "T-B")

    def test_plain_trade_id_is_not_a_roll_ref(self):
        """A normal exit's order_ref is a bare trade_id -- must never be
        misread as a roll (no colons to split on)."""
        assert parse_roll_order_ref("T-A") is None

    def test_empty_string_is_not_a_roll_ref(self):
        assert parse_roll_order_ref("") is None


# ---------------------------------------------------------------------------
# _persistable_legs -- W57 bug 3 (unserializable legs)
# ---------------------------------------------------------------------------

class TestPersistableLegs:
    def test_strips_contract_but_keeps_everything_else(self):
        legs = [{"strike": 460.0, "right": "P", "action": "SELL", "expiry": "2026-10-23",
                 "conId": 111, "ratio": 1, "_contract": object()}]
        out = _persistable_legs(legs)
        assert out == [{"strike": 460.0, "right": "P", "action": "SELL",
                        "expiry": "2026-10-23", "conId": 111, "ratio": 1}]
        # The real regression guard: json.dumps must not raise on the
        # ORIGINAL legs' worth of data once passed through this.
        json.dumps(out)

    def test_legs_without_contract_pass_through_unchanged(self):
        assert _persistable_legs(OPENING_LEGS) == OPENING_LEGS


# ---------------------------------------------------------------------------
# seed_roll_opening_trade
# ---------------------------------------------------------------------------

class TestSeedRollOpeningTrade:
    def test_writes_a_pending_row_with_the_opening_legs_only(self, tmp_path):
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

    def test_rolled_from_trade_id_is_set_at_seed_time_not_deferred_to_promotion(self, tmp_path):
        """W57: record_trade()'s own INSERT doesn't carry the column
        (state.py:340-369) -- must be set by an explicit follow-up write,
        and must be visible immediately, before any fill lands."""
        state = _state(tmp_path)
        closing = _closing_trade()
        state.record_trade(closing)
        opened = seed_roll_opening_trade(
            state, closing, OPENING_LEGS, _roll_target(), quantity=1, entry_price_estimate=0.75,
        )
        fetched = state.get_trade_by_id(opened.trade_id)
        assert fetched.rolled_from_trade_id == closing.trade_id

    def test_serializes_legs_straight_out_of_qualify_roll_legs(self, tmp_path):
        """W57 bug 3, reproduced directly: qualify_roll_legs' own output
        shape (each leg carrying a live, non-JSON-serializable '_contract')
        must not raise here."""
        state = _state(tmp_path)
        closing = _closing_trade()
        state.record_trade(closing)
        qualified_opening = [
            {**leg, "conId": 900 + i, "ratio": 1, "_contract": object()}
            for i, leg in enumerate(OPENING_LEGS)
        ]
        opened = seed_roll_opening_trade(
            state, closing, qualified_opening, _roll_target(),
            quantity=1, entry_price_estimate=0.75,
        )
        fetched = state.get_trade_by_id(opened.trade_id)
        stored = json.loads(fetched.legs)
        assert all("_contract" not in leg for leg in stored)
        assert {leg["conId"] for leg in stored} == {900, 901}

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
# _promote_and_link_opening_trade -- the atomic-write fix's own proof
# ---------------------------------------------------------------------------

class TestPromoteAndLinkOpeningTrade:
    def _pending_opening(self, tmp_path):
        state = _state(tmp_path)
        closing = _closing_trade()
        state.record_trade(closing)
        opening = seed_roll_opening_trade(
            state, closing, OPENING_LEGS, _roll_target(), quantity=1, entry_price_estimate=0.75,
        )
        return state, closing, opening

    def test_promotes_links_and_sets_legs_together(self, tmp_path):
        state, closing, opening = self._pending_opening(tmp_path)
        merged = json.dumps(OPENING_LEGS)
        ok = _promote_and_link_opening_trade(
            state, opening.trade_id, 0.65, 1, closing.trade_id, merged,
        )
        assert ok is True
        b = state.get_trade_by_id(opening.trade_id)
        assert b.status == TradeStatus.FILLED
        assert b.rolled_from_trade_id == closing.trade_id
        assert b.entry_price == pytest.approx(0.65)
        assert json.loads(b.legs) == OPENING_LEGS

    def test_refused_when_not_pending_or_partial_leaves_everything_untouched(self, tmp_path):
        """Proves the CAS is refused wholesale on a wrong status -- the
        legs write (this test's own new field, on top of status +
        rolled_from_trade_id) must never land on a non-PENDING/PARTIAL row
        either. rolled_from_trade_id is already set at SEED time (not
        deferred to promotion -- see TestSeedRollOpeningTrade), so a
        refused promotion here leaves it exactly as seed_roll_opening_trade
        left it, not None."""
        state, closing, opening = self._pending_opening(tmp_path)
        state.transition(opening.trade_id, (TradeStatus.PENDING,), TradeStatus.CANCELLED)
        original_legs = state.get_trade_by_id(opening.trade_id).legs

        ok = _promote_and_link_opening_trade(
            state, opening.trade_id, 0.65, 1, closing.trade_id, json.dumps(CARRIED_CALL_LEGS),
        )
        assert ok is False
        b = state.get_trade_by_id(opening.trade_id)
        assert b.status == TradeStatus.CANCELLED
        assert b.rolled_from_trade_id == closing.trade_id  # set at seed, untouched by the refusal
        assert b.legs == original_legs  # NOT overwritten with the attempted merged legs

    def test_single_atomic_update_statement(self, tmp_path, monkeypatch):
        """The direct regression guard for the bug this fix closes: the
        status CAS, the rolled_from_trade_id write, and the legs write must
        be ONE statement, not several (the prior design's real crash
        window)."""
        state, closing, opening = self._pending_opening(tmp_path)

        import sqlite3 as real_sqlite3
        executed_sql: list[str] = []
        real_connect = real_sqlite3.connect

        class _SpyConn:
            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *args, **kwargs):
                executed_sql.append(sql)
                return self._conn.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._conn, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._conn.__exit__(*exc)

        def _spy_connect(*args, **kwargs):
            return _SpyConn(real_connect(*args, **kwargs))

        import ait.execution.roll_booking as rb_mod
        monkeypatch.setattr(rb_mod.sqlite3, "connect", _spy_connect)

        _promote_and_link_opening_trade(
            state, opening.trade_id, 0.65, 1, closing.trade_id, json.dumps(OPENING_LEGS),
        )

        updates = [s for s in executed_sql if s.strip().upper().startswith("UPDATE")]
        assert len(updates) == 1, f"expected exactly one UPDATE, got {len(updates)}: {updates}"
        assert "status" in updates[0] and "rolled_from_trade_id" in updates[0]
        assert "legs" in updates[0]


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

CARRIED_CALL_LEGS = [
    {"strike": 520.0, "right": "C", "action": "SELL", "expiry": "2026-10-05"},
    {"strike": 530.0, "right": "C", "action": "BUY", "expiry": "2026-10-05"},
]


class TestBookRollFill:
    def _seeded(self, tmp_path, *, strategy="iron_condor", entry_price=1.50, capital_at_risk=0.0):
        state = _state(tmp_path)
        closing = _closing_trade(strategy=strategy, entry_price=entry_price,
                                 status=TradeStatus.FILLED, capital_at_risk=capital_at_risk)
        state.record_trade(closing)
        # record_trade's own INSERT doesn't carry capital_at_risk at all
        # (state.py:349-367) -- only set_trade_capital_at_risk's separate
        # UPDATE does, matching how a real entry populates it live.
        if capital_at_risk:
            state.set_trade_capital_at_risk(closing.trade_id, capital_at_risk)
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
            NO_CARRIED, 0.0,
        )
        assert ok is True

        a = state.get_trade_by_id(closing.trade_id)
        assert a.status == TradeStatus.CLOSED
        assert a.exit_price == pytest.approx(1.00 - 0.20)
        assert f"rolled_to_{opening.trade_id}" in a.exit_reason_detailed

        b = state.get_trade_by_id(opening.trade_id)
        assert b.status == TradeStatus.FILLED
        assert b.rolled_from_trade_id == closing.trade_id
        assert b.entry_price == pytest.approx(abs(-0.90 + 0.15))
        assert json.loads(b.legs) == OPENING_LEGS

    def test_capital_at_risk_defaults_to_as_own(self, tmp_path):
        state, closing, opening = self._seeded(tmp_path, capital_at_risk=190.0)
        book_roll_fill(state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS,
                       OPENING_CONIDS, NO_CARRIED, 0.0)
        b = state.get_trade_by_id(opening.trade_id)
        assert b.capital_at_risk == pytest.approx(190.0)

    def test_capital_at_risk_explicit_override(self, tmp_path):
        state, closing, opening = self._seeded(tmp_path, capital_at_risk=190.0)
        book_roll_fill(
            state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS,
            NO_CARRIED, 0.0, capital_at_risk=250.0,
        )
        b = state.get_trade_by_id(opening.trade_id)
        assert b.capital_at_risk == pytest.approx(250.0)

    def test_credit_strategy_closed_for_less_than_its_credit_is_a_gain(self, tmp_path):
        """W57 bug 2 (fixed): a credit vertical (entry_price=1.50 received)
        bought back for a net 0.80 debit (BUY back 1.00, SELL back 0.20) is
        a GAIN of $70 -- received 1.50, paid back only 0.80. The pre-fix
        formula booked +$230 here (entry/exit sign inverted); this is the
        corrected value, matching _calculate_realized_pnl's own credit
        formula (executor.py:2708-2710) with carried_value=0."""
        state, closing, opening = self._seeded(tmp_path, strategy="iron_condor", entry_price=1.50)
        book_roll_fill(state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS,
                       OPENING_CONIDS, NO_CARRIED, 0.0)
        a = state.get_trade_by_id(closing.trade_id)
        assert a.realized_pnl == pytest.approx((1.50 - 0.80) * 100)

    def test_debit_strategy_polarity_is_the_mirror_of_credit(self, tmp_path):
        state, closing, opening = self._seeded(
            tmp_path, strategy="long_straddle", entry_price=1.50,
        )
        book_roll_fill(state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS,
                       OPENING_CONIDS, NO_CARRIED, 0.0)
        a = state.get_trade_by_id(closing.trade_id)
        # Debit: paid 1.50 to open, got back 0.80 on close -- a $70 LOSS.
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
            NO_CARRIED, 0.0,
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
            NO_CARRIED, 0.0,
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
            NO_CARRIED, 0.0,
        )
        assert ok is False
        assert state.get_trade_by_id(closing.trade_id).status == TradeStatus.CLOSING
        assert state.get_trade_by_id(opening.trade_id).status == TradeStatus.PENDING

    def test_opening_trade_missing_refuses(self, tmp_path):
        """book_roll_fill now reads B's seeded legs from the DB (to merge
        with carried_legs) -- a B row that doesn't exist must refuse
        cleanly, not raise."""
        state, closing, _opening = self._seeded(tmp_path)
        ok = book_roll_fill(
            state, closing, "T-DOES-NOT-EXIST", LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS,
            NO_CARRIED, 0.0,
        )
        assert ok is False


# ---------------------------------------------------------------------------
# book_roll_fill with a carried vertical -- W57 bug 1 (orphaned vertical)
# ---------------------------------------------------------------------------

class TestBookRollFillWithCarriedVertical:
    def _seeded(self, tmp_path, *, entry_price=1.50):
        state = _state(tmp_path)
        closing = _closing_trade(entry_price=entry_price, status=TradeStatus.FILLED)
        state.record_trade(closing)
        assert cas_roll_closing_trade(state, closing.trade_id) is True
        closing = state.get_trade_by_id(closing.trade_id)
        opening = seed_roll_opening_trade(
            state, closing, OPENING_LEGS, _roll_target(), quantity=1, entry_price_estimate=0.75,
        )
        return state, closing, opening

    def test_carried_vertical_is_attached_to_b_not_orphaned(self, tmp_path):
        state, closing, opening = self._seeded(tmp_path)
        ok = book_roll_fill(
            state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS,
            CARRIED_CALL_LEGS, carried_value=-0.75,
        )
        assert ok is True
        b = state.get_trade_by_id(opening.trade_id)
        stored_legs = json.loads(b.legs)
        assert len(stored_legs) == 4
        assert {(leg["strike"], leg["right"]) for leg in stored_legs} == {
            (450.0, "P"), (460.0, "P"), (520.0, "C"), (530.0, "C"),
        }

    def test_pnl_with_carried_vertical_matches_hand_computed_cash_flow(self, tmp_path):
        """The worked example from PLAN.md's W57 entry: $1.50 credit IC,
        rolled put side bought back at 0.80 net debit, carried call
        vertical marked at -0.30 (still a $0.30 liability). A must realize
        +$40, and B's entry basis must be 1.05 (0.75 new credit + the 0.30
        inherited from the carried side) -- both cash flows sum correctly
        across the chain (the carried mark cancels: A books +0.30 worth of
        it via exit_price, B absorbs the same 0.30 into its entry)."""
        state, closing, opening = self._seeded(tmp_path)
        ok = book_roll_fill(
            state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS,
            CARRIED_CALL_LEGS, carried_value=-0.30,
        )
        assert ok is True
        a = state.get_trade_by_id(closing.trade_id)
        assert a.realized_pnl == pytest.approx(40.0)
        b = state.get_trade_by_id(opening.trade_id)
        assert b.entry_price == pytest.approx(1.05)

    def test_no_carried_legs_is_pnl_neutral_versus_the_plain_case(self, tmp_path):
        """carried_value=0.0 with an empty carried_legs list must reduce
        exactly to the plain-vertical-roll numbers above -- the carried
        machinery must not perturb a roll that has nothing to carry."""
        state, closing, opening = self._seeded(tmp_path)
        book_roll_fill(
            state, closing, opening.trade_id, LEG_FILLS, CLOSING_CONIDS, OPENING_CONIDS,
            [], 0.0,
        )
        a = state.get_trade_by_id(closing.trade_id)
        assert a.realized_pnl == pytest.approx((1.50 - 0.80) * 100)

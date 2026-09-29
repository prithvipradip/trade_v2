"""W57: reconciler.py must not fight a working roll order.

A roll order's orderRef is "ROLL:<A>:<B>" (roll_booking.roll_order_ref) --
never equal to trade_id verbatim, since it names TWO trades. Before this
wave the reconciler's exact-orderRef matching (_working_exit_order_for /
_cancel_working_entry_order) could not recognise it, which meant:
  - A working roll order tagged for A got re-adopted as a plain exit
    (closing_exit_orders -> executor.adopt_exit_order), double-booking the
    eventual fill against book_roll_fill's own booking.
  - Or, if the symbol/leg-key fallback also missed it, A got reverted
    CLOSING -> FILLED mid-roll (a position REVERSAL risk: the exit monitor
    would then try to close an already-in-flight position again).
  - The every-cycle stale-pending sweep would book a roll child B "never
    filled" after 30 minutes even while its roll order is still working,
    and would refund an entry-budget slot B never consumed.

Every test drives the real reconciler code path against a REAL temp-DB
StateManager (never mocked) and tests.fakes' FakeIB/FakeIBKRClient (never a
MagicMock stand-in for option marks/quotes). No inspect.getsource.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
from ait.execution.reconciler import PositionReconciler
from ait.execution.roll_booking import (
    cas_roll_closing_trade,
    roll_order_ref,
    seed_roll_opening_trade,
)
from ait.execution.roll_selector import RollTarget
from ait.data.options_chain import OptionContract
from ait.utils.time import now_et
from tests.fakes import FakeIB, FakeIBKRClient, _option_position

CONDOR_LEGS = [
    {"strike": 745.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 735.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 775.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 785.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
]
OPENING_LEGS = [
    {"strike": 720.0, "right": "P", "action": "SELL", "expiry": "2027-01-15"},
    {"strike": 710.0, "right": "P", "action": "BUY", "expiry": "2027-01-15"},
]


def _condor_positions(legs=CONDOR_LEGS) -> list:
    return [
        _option_position("SPY", leg["strike"], leg["right"],
                         leg["expiry"].replace("-", ""),
                         -1.0 if leg["action"] == "SELL" else 1.0)
        for leg in legs
    ]


def _broker_order(ref="", sec="BAG", status="Submitted", order_id=11):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol="SPY", secType=sec, strike=0.0, right="P",
                                 lastTradeDateOrContractMonth="20261218"),
        order=SimpleNamespace(orderId=order_id, orderRef=ref),
        orderStatus=SimpleNamespace(status=status),
    )


# ---------------------------------------------------------------------------
# _working_roll_order_for -- pure, mirrors test_r17_risk_data_fixes.py's
# TestReconcilerOrderRefDisambiguation for the plain-exit case
# ---------------------------------------------------------------------------

def _bare_reconciler():
    return PositionReconciler.__new__(PositionReconciler)


def _bare_trade(trade_id):
    return SimpleNamespace(trade_id=trade_id, symbol="SPY", strategy="iron_condor",
                           legs="[]", strike=None, expiry=None)


class TestWorkingRollOrderFor:
    def test_matches_on_the_closing_side_of_a_roll_ref(self):
        rec = _bare_reconciler()
        trade_a = _bare_trade("T-A")
        working = [(111, {"SPY:BAG"}, "SPY", roll_order_ref("T-A", "T-B"))]
        assert rec._working_roll_order_for(trade_a, working) == 111

    def test_does_not_match_on_the_opening_side(self):
        """The opening side (B) is recognised by _cancel_working_entry_order
        instead (B is PENDING, not CLOSING) -- this method is the CLOSING
        (A) side's own check."""
        rec = _bare_reconciler()
        trade_b = _bare_trade("T-B")
        working = [(111, {"SPY:BAG"}, "SPY", roll_order_ref("T-A", "T-B"))]
        assert rec._working_roll_order_for(trade_b, working) is None

    def test_a_plain_exit_ref_is_not_a_roll_match(self):
        rec = _bare_reconciler()
        trade_a = _bare_trade("T-A")
        working = [(111, {"SPY:BAG"}, "SPY", "T-A")]
        assert rec._working_roll_order_for(trade_a, working) is None

    def test_no_working_orders_returns_none(self):
        rec = _bare_reconciler()
        assert rec._working_roll_order_for(_bare_trade("T-A"), []) is None


# ---------------------------------------------------------------------------
# _cancel_working_entry_order recognises a roll order tagged for B (the
# opening/PENDING side)
# ---------------------------------------------------------------------------

def _rec_with_orders(orders):
    rec = _bare_reconciler()
    rec._ibkr = MagicMock()
    rec._ibkr.foreign_open_order_ids = set()
    ib = MagicMock()
    ib.openTrades.return_value = orders
    ib.cancelOrder = MagicMock()
    rec._ibkr.ib = ib
    return rec, ib


class TestCancelRecognisesARollOrderForB:
    def test_cancels_a_roll_order_tagged_for_this_trade_as_opening_side(self):
        mine = _broker_order(ref=roll_order_ref("T-A", "T-B"), sec="BAG")
        rec, ib = _rec_with_orders([mine])
        assert rec._cancel_working_entry_order(_bare_trade("T-B")) is True
        ib.cancelOrder.assert_called_once_with(mine.order)

    def test_does_not_claim_a_roll_order_tagged_for_a_different_opening_trade(self):
        other = _broker_order(ref=roll_order_ref("T-A", "T-OTHER"), sec="BAG")
        rec, ib = _rec_with_orders([other])
        assert rec._cancel_working_entry_order(_bare_trade("T-B")) is False
        ib.cancelOrder.assert_not_called()


# ---------------------------------------------------------------------------
# reconcile() end to end -- the CLOSING branch must not fight a roll
# ---------------------------------------------------------------------------

def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _closing_trade_a(trade_id="T-A") -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id, symbol="SPY", strategy="iron_condor",
        direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
        entry_time="2026-09-27T10:00:00", entry_price=4.24, quantity=1,
        contract_type="iron_condor", legs=json.dumps(CONDOR_LEGS),
    )


def _roll_target() -> RollTarget:
    return RollTarget(
        new_short=OptionContract(symbol="SPY", expiry="2027-01-15", strike=720.0, right="P",
                                 bid=0, ask=0, last=0, volume=0, open_interest=0,
                                 implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        new_long=OptionContract(symbol="SPY", expiry="2027-01-15", strike=710.0, right="P",
                                bid=0, ask=0, last=0, volume=0, open_interest=0,
                                implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        expiry="2027-01-15",
    )


@pytest.fixture(autouse=True)
def _cwd_tmp(tmp_path, monkeypatch):
    """The reconciler writes data/HALT_UNTRACKED relative to cwd -- same
    fixture shape test_w50_qty_ledger_fork.py/test_w41 use."""
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


class TestReconcileClosingBranchWithARollInFlight:
    async def test_a_working_roll_order_keeps_a_closing_not_reverted(self, tmp_path):
        st = _state(tmp_path)
        a = _closing_trade_a()
        st.record_trade(a)
        assert cas_roll_closing_trade(st, a.trade_id) is True

        ib = FakeIB()
        ib.positions_list = _condor_positions()  # A's legs still live at the broker
        ib.open_trades = [_broker_order(ref=roll_order_ref(a.trade_id, "T-B"), sec="BAG")]
        rec = PositionReconciler(FakeIBKRClient(ib), st)

        result = await rec.reconcile()

        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.CLOSING
        assert result.closing_roll_orders.get(a.trade_id) == 11
        assert a.trade_id not in result.closing_exit_orders

    async def test_a_plain_working_exit_order_is_unaffected_by_the_roll_check(self, tmp_path):
        """R17 behaviour must survive unchanged: a plain exit order (bare
        trade_id orderRef) still lands in closing_exit_orders, not
        closing_roll_orders."""
        st = _state(tmp_path)
        a = _closing_trade_a()
        st.record_trade(a)
        assert cas_roll_closing_trade(st, a.trade_id) is True

        ib = FakeIB()
        ib.positions_list = _condor_positions()
        ib.open_trades = [_broker_order(ref=a.trade_id, sec="BAG")]
        rec = PositionReconciler(FakeIBKRClient(ib), st)

        result = await rec.reconcile()

        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.CLOSING
        assert result.closing_exit_orders.get(a.trade_id) == 11
        assert a.trade_id not in result.closing_roll_orders

    async def test_no_working_order_at_all_still_reverts_to_filled(self, tmp_path):
        """The existing self-heal (an early-crash recovery, unrelated to
        rolls) must survive unchanged: no working order of ANY kind ->
        CLOSING is promoted back to FILLED."""
        st = _state(tmp_path)
        a = _closing_trade_a()
        st.record_trade(a)
        assert cas_roll_closing_trade(st, a.trade_id) is True

        ib = FakeIB()
        ib.positions_list = _condor_positions()
        ib.open_trades = []
        rec = PositionReconciler(FakeIBKRClient(ib), st)

        await rec.reconcile()

        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.FILLED


# ---------------------------------------------------------------------------
# _sweep_stale_pending -- a roll child B that never filled must not refund
# a budget slot it never spent, and must restore its parent A
# ---------------------------------------------------------------------------

def _backdate(st, trade_id, minutes):
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "UPDATE trades SET entry_time = ? WHERE trade_id = ?",
            ((now_et() - timedelta(minutes=minutes)).isoformat(), trade_id),
        )


class TestSweepOnAnAbandonedRollChild:
    def _seed_roll(self, tmp_path):
        st = _state(tmp_path)
        a = _closing_trade_a()
        st.record_trade(a)
        assert cas_roll_closing_trade(st, a.trade_id) is True
        a = st.get_trade_by_id(a.trade_id)
        b = seed_roll_opening_trade(
            st, a, OPENING_LEGS, _roll_target(), quantity=1, entry_price_estimate=0.75,
        )
        _backdate(st, b.trade_id, minutes=45)  # older than STALE_PENDING_MINUTES=30
        return st, a, b

    def test_stale_roll_child_books_never_filled_no_refund_parent_restored(self, tmp_path):
        st, a, b = self._seed_roll(tmp_path)
        entry_date = now_et().date()
        stats = st.get_daily_stats(entry_date)
        stats.trades_taken = 2
        st.update_daily_stats(stats)

        ib = FakeIB()
        ib.positions_list = []  # B's new legs never went live -- the roll never filled
        ib.open_trades = []     # and no roll order is working either
        rec = PositionReconciler(FakeIBKRClient(ib), st)

        closed = rec._sweep_stale_pending()

        assert closed == 1
        assert st.get_trade_by_id(b.trade_id).status == TradeStatus.CLOSED
        assert st.get_trade_by_id(b.trade_id).exit_reason_detailed == "stale_pending_never_filled"
        # The budget slot B never spent must not be refunded.
        assert st.get_daily_stats(entry_date).trades_taken == 2
        # A, parked CLOSING when the roll was seeded, must come back to FILLED
        # -- or check_positions (FILLED/PARTIAL only) never manages it again.
        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.FILLED

    def test_a_plain_stale_pending_is_still_refunded_unchanged(self, tmp_path):
        """Regression guard: the roll-child branch must not swallow the
        ordinary refund path for a ROW WITH NO rolled_from_trade_id."""
        st = _state(tmp_path)
        plain = TradeRecord(
            trade_id="T-PLAIN", symbol="SPY", strategy="iron_condor",
            direction=TradeDirection.SHORT, status=TradeStatus.PENDING,
            entry_time=(now_et() - timedelta(minutes=45)).isoformat(),
            entry_price=4.24, quantity=1, contract_type="iron_condor",
            legs=json.dumps(CONDOR_LEGS),
        )
        st.record_trade(plain)
        entry_date = now_et().date()
        stats = st.get_daily_stats(entry_date)
        stats.trades_taken = 2
        st.update_daily_stats(stats)

        ib = FakeIB()
        ib.positions_list = []
        ib.open_trades = []
        rec = PositionReconciler(FakeIBKRClient(ib), st)

        closed = rec._sweep_stale_pending()

        assert closed == 1
        assert st.get_trade_by_id("T-PLAIN").status == TradeStatus.CLOSED
        assert st.get_daily_stats(entry_date).trades_taken == 1

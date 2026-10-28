"""W112: every post-entry path at N lots (the cap stays 1; this is what must hold before it is raised).

Two kinds of test. PINS: paths that were already right at N (close BAG size, realized P&L, expiry intrinsic,
quantity adoption) are fixed in place so a later edit cannot quietly assume one lot. GAPS: five defects that
exist only at N >= 2, each written to fail on the code as it stood before the fix.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.execution.executor import BAG_CON_ID, TradeExecutor
from ait.execution.reconciler import PositionReconciler
from ait.execution.portfolio import PositionAction
from ait.utils.time import now_et
from tests.test_exit_order_safety import (  # noqa: F401  (the autouse fixture must be in scope)
    CONDOR_LEGS, FakeTrade, _fast_sleep, _orchestrator, _quote,
)
from tests.test_position_action_output import FakeCondorTrade, _condor, _manager
from tests.test_w101c_expiry_booking import _book, _monday, _rec, _trade  # noqa: F401
from tests.test_w73_live_dte_and_clock_fixes import _freeze  # noqa: F401

TID = "T-W112"


def _state(tmp_path, quantity: int, legs: str = "[1,2,3,4]") -> StateManager:
    st = StateManager(db_path=tmp_path / "ait_state.db")
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, status, "
            "entry_time, entry_price, quantity, contract_type, legs) "
            "VALUES (?,?,?,?,?,?,?,?,'iron_condor',?)",
            (TID, "SPY", "iron_condor", "short", "filled", now_et().isoformat(), 2.50, quantity, legs))
    return st


def _ledger(st, rows):
    """rows: (con_id, shares) with a known commission; each gets its own execution id."""
    with sqlite3.connect(st._db_path) as conn:
        for i, (con_id, shares) in enumerate(rows):
            conn.execute(
                "INSERT INTO executions (exec_id, order_id, trade_id, symbol, con_id, shares, commission) "
                "VALUES (?,?,?,?,?,?,?)", (f"E{i}", 1, TID, "SPY", con_id, shares, 0.9))


# ---------------------------------------------------------------------------------------------------------
# PINS: already right at N lots
# ---------------------------------------------------------------------------------------------------------
class TestAlreadyRightAtNLots:
    @pytest.mark.parametrize("lots", [2, 3])
    async def test_the_close_combo_is_sized_from_the_trade_row(self, lots):
        orch, place_order, _ = _orchestrator(_quote(bid=0.85, ask=0.95))
        await orch._close_multi_leg(FakeTrade(quantity=lots), CONDOR_LEGS)
        assert place_order.call_args.args[1].totalQuantity == lots

    @pytest.mark.parametrize("lots", [2, 3])
    def test_realized_pnl_scales_with_the_lots_credit_and_debit(self, lots):
        ex = TradeExecutor.__new__(TradeExecutor)
        ex._state = MagicMock()
        credit = SimpleNamespace(trade_id="C", strategy="iron_condor", entry_price=2.50, quantity=lots,
                                 contract_type="iron_condor", legs="[1,2,3,4]")
        debit = SimpleNamespace(trade_id="D", strategy="bull_call_spread", entry_price=1.40, quantity=lots,
                                contract_type="spread", legs="[1,2]")
        gross_credit = (2.50 - 0.50) * 100 * lots
        gross_debit = (2.40 - 1.40) * 100 * lots
        # commission is the only thing that is not exactly proportional; it is per lot too
        assert ex._calculate_realized_pnl(credit, 0.50) == pytest.approx(gross_credit, abs=lots * 10.0)
        assert ex._calculate_realized_pnl(debit, 2.40) == pytest.approx(gross_debit, abs=lots * 10.0)
        one = ex._calculate_realized_pnl(SimpleNamespace(**{**credit.__dict__, "quantity": 1}), 0.50)
        many = ex._calculate_realized_pnl(credit, 0.50)
        assert many == pytest.approx(one * lots, rel=0.1)

    @pytest.mark.parametrize("lots", [2, 3])
    def test_an_expired_condor_books_the_settlement_value_times_the_lots(self, lots):
        trade = _trade()
        trade.quantity = lots
        kw = _book(_rec(settlement=704.4), trade)
        assert kw["realized_pnl"] == pytest.approx(-168.0 * lots)


# ---------------------------------------------------------------------------------------------------------
# GAP 1: multi-leg partial exits
# ---------------------------------------------------------------------------------------------------------
class _TwoLotVertical(FakeCondorTrade):
    pass


def _debit_vertical(lots: int, dte: int = 25):
    t = FakeCondorTrade(entry_price=2.00, quantity=lots, strategy="bull_call_spread", contract_type="spread",
                        trade_id="T-VERT",
                        expiry=(date.today().replace(year=date.today().year)).isoformat())
    from datetime import timedelta
    t.expiry = (date.today() + timedelta(days=dte)).isoformat()
    t.direction = SimpleNamespace(value="long")
    return t


class TestAMultiLegPositionIsManagedWhole:
    @pytest.mark.parametrize("lots", [2, 3])
    async def test_a_debit_vertical_in_the_partial_band_asks_for_no_partial_exit(self, lots, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        # +60% of the debit: above partial_exit_levels[0] (0.50), below the debit take-profit target
        mgr = _manager(spot=100.0, unrealized=0.60 * 2.00 * lots * 100)
        status = await mgr._evaluate_position(_debit_vertical(lots))
        assert not status.should_exit
        assert status.partial_exit_quantity == 0
        assert status.action == PositionAction.HOLD

    async def test_a_condor_at_two_lots_asks_for_none_either(self, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        mgr = _manager(spot=100.0, unrealized=10.0)
        trade = _condor(dte=20)
        trade.quantity = 2
        status = await mgr._evaluate_position(trade)
        assert status.partial_exit_quantity == 0

    async def test_a_single_leg_position_still_takes_its_partial_exit(self, monkeypatch):
        """The ladder is untouched where it is implemented."""
        from tests.test_position_action_output import _long_call
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        mgr = _manager(spot=100.0, unrealized=400.0)
        status = await mgr._evaluate_position(_long_call(dte=25))
        assert status.partial_exit_quantity > 0


# ---------------------------------------------------------------------------------------------------------
# GAP 2: commission completeness is counted in contracts, not rows
# ---------------------------------------------------------------------------------------------------------
class TestTheCommissionTrueUpWaitsForEveryContract:
    @staticmethod
    def _orch(st):
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._state = st
        o._find_trade_by_id = lambda tid: st.get_trade_by_id(tid)
        return o

    def test_entry_rows_alone_do_not_complete_a_two_lot_ledger(self, tmp_path):
        """A 2-lot condor whose four legs each filled in TWO entry executions has 8 commissioned rows (the
        count of a finished 1-lot round trip) but only the entry's 8 contracts of the 16 it will have."""
        st = _state(tmp_path, quantity=2)
        _ledger(st, [(c, 1) for c in (11, 12, 13, 14)] * 2)
        o = self._orch(st)
        complete, rows, expected = o._commission_ledger_complete(TID, st.get_trade_by_id(TID))
        assert rows == 8 and expected == 8
        assert complete is False

    def test_the_booking_time_true_up_defers_on_entry_rows_only_and_runs_on_the_full_ledger(
            self, tmp_path, monkeypatch):
        from tests.test_w37_commission_estimate_is_remembered import _cost_model, _executor
        st = _state(tmp_path, quantity=2)
        _cost_model(monkeypatch, 0.65)
        booked = _executor(st)._calculate_realized_pnl(st.get_trade_by_id(TID), exit_price=0.50)
        o = self._orch(st)
        _ledger(st, [(c, 1) for c in (11, 12, 13, 14)] * 2)
        assert o._true_up_commission(TID, booked) == pytest.approx(booked)
        assert TID in st.pending_trueup_trade_ids()
        with sqlite3.connect(st._db_path) as conn:
            for i, c in enumerate((11, 12, 13, 14) * 2):
                conn.execute("INSERT INTO executions (exec_id, order_id, trade_id, symbol, con_id, shares, "
                             "commission) VALUES (?,?,?,?,?,?,?)", (f"Y{i}", 1, TID, "SPY", c, 1, 0.9))
        assert o._true_up_commission(TID, booked) != pytest.approx(booked)       # now it trues up

    def test_the_full_round_trip_completes_it(self, tmp_path):
        st = _state(tmp_path, quantity=2)
        _ledger(st, [(c, 1) for c in (11, 12, 13, 14)] * 4)
        complete, _, _ = self._orch(st)._commission_ledger_complete(TID, st.get_trade_by_id(TID))
        assert complete is True

    def test_a_bag_summary_row_never_counts_toward_it(self, tmp_path):
        st = _state(tmp_path, quantity=2)
        _ledger(st, [(c, 1) for c in (11, 12, 13, 14)] * 2 + [(BAG_CON_ID, 2)] * 2)
        complete, _, _ = self._orch(st)._commission_ledger_complete(TID, st.get_trade_by_id(TID))
        assert complete is False

    def test_one_lot_is_what_it_always_was(self, tmp_path):
        st = _state(tmp_path, quantity=1)
        o = self._orch(st)
        _ledger(st, [(c, 1) for c in (11, 12, 13, 14)])
        assert o._commission_ledger_complete(TID, st.get_trade_by_id(TID))[0] is False
        _ledger_more = [(c, 1) for c in (11, 12, 13, 14)]
        with sqlite3.connect(st._db_path) as conn:
            for i, (con_id, shares) in enumerate(_ledger_more):
                conn.execute(
                    "INSERT INTO executions (exec_id, order_id, trade_id, symbol, con_id, shares, commission) "
                    "VALUES (?,?,?,?,?,?,?)", (f"X{i}", 1, TID, "SPY", con_id, shares, 0.9))
        assert o._commission_ledger_complete(TID, st.get_trade_by_id(TID))[0] is True

    def test_a_row_without_a_recorded_size_counts_as_one_contract(self, tmp_path):
        st = _state(tmp_path, quantity=1)
        with sqlite3.connect(st._db_path) as conn:
            for i in range(8):
                conn.execute("INSERT INTO executions (exec_id, order_id, trade_id, symbol, commission) "
                             "VALUES (?,?,?,?,?)", (f"N{i}", 1, TID, "SPY", 0.9))
        assert self._orch(st)._commission_ledger_complete(TID, st.get_trade_by_id(TID))[0] is True

    def test_lots_already_sold_in_exit_slices_still_count_as_lots_traded(self, tmp_path):
        st = _state(tmp_path, quantity=1)
        with sqlite3.connect(st._db_path) as conn:
            conn.execute(
                "INSERT INTO open_positions (position_id, trade_id, symbol, contract_type, quantity, "
                "entry_price, entry_time, high_water_mark, partial_exits) VALUES (?,?,?,?,?,?,?,?,?)",
                (f"pos-{TID}", TID, "SPY", "iron_condor", 1, 2.50, now_et().isoformat(), 0.0, "[]"))
        st.record_partial_exit(TID, quantity=1, price=1.0, pnl=10.0)
        # the row says 1 lot left; the ledger of an original 2-lot trade has 16 contracts when complete
        _ledger(st, [(c, 1) for c in (11, 12, 13, 14)] * 2)
        complete, _, _ = self._orch(st)._commission_ledger_complete(TID, st.get_trade_by_id(TID))
        assert complete is False


class TestThePostMarketTrueUpWaitsToo:
    def test_a_two_lot_ledger_with_entry_rows_only_is_not_settled_and_the_full_one_is(self, tmp_path, monkeypatch):
        from tests.test_w37_commission_estimate_is_remembered import _cost_model, _executor
        st = _state(tmp_path, quantity=2)
        _cost_model(monkeypatch, 0.65)
        trade = st.get_trade_by_id(TID)
        booked = _executor(st)._calculate_realized_pnl(trade, exit_price=0.50)
        st.close_trade(TID, exit_price=0.50, realized_pnl=booked)
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._state = st
        o._find_trade_by_id = lambda tid: st.get_trade_by_id(tid)
        st.set_state(f"trueup_pending_{TID}", "2026-10-28")
        _ledger(st, [(c, 1) for c in (11, 12, 13, 14)] * 2)                 # the entry's 8 contracts only
        o._settle_deferred_trueups()
        assert TID in st.pending_trueup_trade_ids()
        assert st.get_trade_by_id(TID).realized_pnl == pytest.approx(booked)
        with sqlite3.connect(st._db_path) as conn:
            for i, c in enumerate((11, 12, 13, 14) * 2):
                conn.execute("INSERT INTO executions (exec_id, order_id, trade_id, symbol, con_id, shares, "
                             "commission) VALUES (?,?,?,?,?,?,?)", (f"X{i}", 1, TID, "SPY", c, 1, 0.9))
        o._settle_deferred_trueups()
        assert TID not in st.pending_trueup_trade_ids()


# ---------------------------------------------------------------------------------------------------------
# GAP 3: expiry after exit slices must not count them twice
# ---------------------------------------------------------------------------------------------------------
class TestExpiryAfterExitSlices:
    @staticmethod
    def _items(pnl_each):
        keys = [("P", 704.0), ("P", 707.0), ("C", 769.0), ("C", 772.0)]
        return [SimpleNamespace(
            contract=SimpleNamespace(secType="OPT", symbol="QQQ", strike=k, right=r,
                                     lastTradeDateOrContractMonth="20261023"),
            position=0, realizedPNL=pnl_each) for r, k in keys]

    def test_the_remaining_lot_is_valued_at_settlement_and_the_broker_sum_is_not_booked(self):
        rec = _rec(settlement=704.4, portfolio=self._items(-100.0))      # a broker figure that covers both lots
        rec._state.get_partial_exits.return_value = [{"quantity": 1, "price": 0.4, "pnl": 52.0}]
        trade = _trade()
        trade.quantity = 1                                               # the slice already left the row
        kw = _book(rec, trade)
        assert kw["exit_reason_detailed"] == "reconciler_expired_intrinsic"
        assert kw["realized_pnl"] == pytest.approx(-168.0)              # one lot at settlement, once

    def test_without_slices_the_brokers_figure_is_used_as_before(self):
        rec = _rec(settlement=704.4, portfolio=self._items(25.0))
        rec._state.get_partial_exits.return_value = []
        kw = _book(rec)
        assert kw["exit_reason_detailed"] == "reconciler_ibkr_realized"
        assert kw["realized_pnl"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------------------------------------
# GAP 4: a fill known only from the executions ledger books the lots that filled
# ---------------------------------------------------------------------------------------------------------
class TestLedgerOnlyFillUnits:
    @staticmethod
    def _executor(st):
        ex = TradeExecutor.__new__(TradeExecutor)
        ex._state = st
        return ex

    @staticmethod
    def _pending(legs=4):
        return SimpleNamespace(trade_id=TID, signal=SimpleNamespace(legs=[object()] * legs), contracts=2)

    def test_the_bag_rows_are_the_combo_units(self, tmp_path):
        st = _state(tmp_path, quantity=2)
        _ledger(st, [(BAG_CON_ID, 1), (11, 1), (12, 1), (13, 1), (14, 1)])
        assert self._executor(st)._ledger_units(TID, self._pending()) == 1.0

    def test_the_bag_rows_are_enough_when_leg_rows_are_missing(self, tmp_path):
        st = _state(tmp_path, quantity=2)
        _ledger(st, [(BAG_CON_ID, 2), (11, 2)])
        assert self._executor(st)._ledger_units(TID, self._pending()) == 2.0

    def test_without_bag_rows_it_is_the_smallest_leg_total(self, tmp_path):
        st = _state(tmp_path, quantity=3)
        _ledger(st, [(11, 3), (12, 2), (13, 1), (13, 1), (14, 3)])       # the 13 leg has filled 2 of 3
        assert self._executor(st)._ledger_units(TID, self._pending()) == 2.0

    def test_a_missing_leg_is_unknown_not_a_guess(self, tmp_path):
        st = _state(tmp_path, quantity=2)
        _ledger(st, [(11, 1), (12, 1), (13, 1)])
        assert self._executor(st)._ledger_units(TID, self._pending()) == 0.0

    def test_the_recovered_quantity_is_what_filled(self, tmp_path):
        st = _state(tmp_path, quantity=2)
        _ledger(st, [(11, 1), (12, 1), (13, 1), (14, 1)])
        ex = self._executor(st)
        ex._ibkr = MagicMock()
        ex._ibkr.ib.fills.return_value = []
        ex._perm_maps = lambda: ({}, {})
        assert ex._get_filled_quantity(7, [], self._pending()) == 1

    def test_a_stub_store_gives_no_evidence_of_a_size(self):
        ex = TradeExecutor.__new__(TradeExecutor)
        ex._state = MagicMock()
        assert ex._ledger_units(TID, self._pending()) == 0.0


# ---------------------------------------------------------------------------------------------------------
# GAP 5: the placement-time stamp does not overwrite the fill's restatement
# ---------------------------------------------------------------------------------------------------------
class TestThePlacementStampKeepsTheRestatement:
    pytestmark = pytest.mark.asyncio

    @pytest.fixture
    def rig(self, tmp_path, monkeypatch):
        from tests.test_hot_path_smoke import build_smoke_orchestrator
        monkeypatch.chdir(tmp_path)
        smoke = build_smoke_orchestrator()
        try:
            yield smoke
        finally:
            smoke.restore()

    async def _place(self, rig, trade_row):
        orch = rig.orch
        from ait.risk.manager import TradeValidation

        async def _validate(request):
            return TradeValidation(approved=True, reason="stubbed", position_size=2, max_risk=760.0)
        orch._risk_manager.validate_trade = _validate
        orch._learning.adaptor.get_strategy_multiplier = lambda strategy: 1.0

        async def _vix():
            return 16.0
        orch._market_data.get_vix = _vix
        orch._find_trade_by_id = lambda tid: trade_row
        from tests.test_hot_path_smoke import iron_condor_signal
        sig = iron_condor_signal()
        sig.quantity = 1
        await orch._try_execute(sig, 0.72, None, None)
        await rig.drain_notifications()

    async def test_a_restatement_that_landed_first_is_kept(self, rig):
        rig.orch._state.set_state("trade_maxloss_T-SMOKE-1", "555.0")
        await self._place(rig, SimpleNamespace(capital_at_risk=555.0))
        assert float(rig.orch._state.get_state("trade_maxloss_T-SMOKE-1")) == pytest.approx(555.0)

    async def test_without_one_the_placement_figure_is_stamped_as_before(self, rig):
        await self._place(rig, SimpleNamespace(capital_at_risk=0.0))
        assert float(rig.orch._state.get_state("trade_maxloss_T-SMOKE-1")) == pytest.approx(2 * 380.0)

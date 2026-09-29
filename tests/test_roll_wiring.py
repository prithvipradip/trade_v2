"""W61: wire ROLL end-to-end -- executor.py's place_roll_combo + check_fills'
roll-recognition block, and orchestrator.py's _execute_roll + _trading_cycle's
roll dispatch/exclusion (_partition_roll_positions).

Gated off by CondorDefenseConfig.execute_rolls (False by default, required IN
ADDITION to enabled=True) -- every gate test proves ZERO calls into the roll
machinery when off, not just "no exception".

Real StateManager where DB state matters (never mocked); tests.fakes.FakeIB/
FakeIBKRClient for the broker boundary (never MagicMock for option marks/
fills, per this codebase's own house rule). No inspect.getsource.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
from ait.config.settings import AssignmentResponseConfig, CondorDefenseConfig
from ait.execution.executor import BAG_CON_ID, TradeExecutor
from ait.execution.portfolio import PositionAction, PositionStatus
from ait.execution.roll_booking import parse_roll_order_ref, roll_order_ref
from ait.execution.roll_selector import RollTarget
from ait.data.options_chain import OptionContract
from tests.fakes import FakeIB, FakeIBKRClient

CONDOR_LEGS = [
    {"strike": 745.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 735.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 775.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 785.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
]


def _trade(trade_id="T-A", entry_price=1.50, quantity=1, legs=None) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id, symbol="SPY", strategy="iron_condor",
        direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
        entry_time="2026-09-27T10:00:00", entry_price=entry_price, quantity=quantity,
        contract_type="iron_condor", legs=json.dumps(legs if legs is not None else CONDOR_LEGS),
    )


def _pos(trade_id="T-A", action=PositionAction.ROLL, roll_side="P",
         should_exit=False, partial_exit_quantity=0) -> PositionStatus:
    return PositionStatus(
        trade_id=trade_id, symbol="SPY", strategy="iron_condor", quantity=1,
        entry_price=1.50, current_price=1.50, unrealized_pnl=0.0, pnl_pct=0.0,
        high_water_mark=0.0, dte=20, should_exit=should_exit, exit_reason="",
        partial_exit_quantity=partial_exit_quantity, action=action, roll_side=roll_side,
    )


# ---------------------------------------------------------------------------
# executor.py -- place_roll_combo
# ---------------------------------------------------------------------------

def _prepared(order_ref="ROLL:T-A:PENDING"):
    return SimpleNamespace(
        bag=SimpleNamespace(symbol="SPY"),
        order=SimpleNamespace(orderRef=order_ref),
        closing_legs=[{"conId": 1}, {"conId": 2}],
        opening_legs=[{"conId": 3}, {"conId": 4}],
    )


def _bare_executor(place_order_result=None) -> TradeExecutor:
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._ibkr = MagicMock()
    ex._ibkr.place_order = AsyncMock(return_value=place_order_result)
    ex._state = MagicMock()
    ex._pending_roll_orders = {}
    ex._pending_orders = {}
    ex._pending_exit_orders = {}
    ex._order_trade_map = {}
    return ex


class TestPlaceRollComboGates:
    async def test_market_closed_refuses(self, monkeypatch):
        monkeypatch.delenv("AIT_ALLOW_AFTER_HOURS", raising=False)
        monkeypatch.setattr("ait.utils.time.is_market_open", lambda: False)
        ex = _bare_executor(SimpleNamespace(order=SimpleNamespace(orderId=555)))
        ok = await ex.place_roll_combo(_prepared(), "T-A", "T-B", [], 0.0)
        assert ok is False
        ex._ibkr.place_order.assert_not_awaited()

    async def test_calendar_exception_fails_closed(self, monkeypatch):
        monkeypatch.delenv("AIT_ALLOW_AFTER_HOURS", raising=False)

        def _boom():
            raise RuntimeError("calendar unavailable")
        monkeypatch.setattr("ait.utils.time.is_market_open", _boom)
        ex = _bare_executor(SimpleNamespace(order=SimpleNamespace(orderId=555)))
        ok = await ex.place_roll_combo(_prepared(), "T-A", "T-B", [], 0.0)
        assert ok is False
        ex._ibkr.place_order.assert_not_awaited()

    async def test_after_hours_override_bypasses_the_market_hours_check(self, monkeypatch):
        monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
        monkeypatch.setattr("ait.utils.time.is_market_open", lambda: False)
        ex = _bare_executor(SimpleNamespace(order=SimpleNamespace(orderId=555)))
        ok = await ex.place_roll_combo(_prepared(), "T-A", "T-B", [], 0.0)
        assert ok is True

    async def test_rate_limit_shares_state_with_prior_entry_placements(self, monkeypatch):
        """The SAME self._order_times list execute_signal populates -- proof
        it's shared state, not a parallel counter that would let a roll
        bypass the malfunction backstop entries are already subject to."""
        monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
        ex = _bare_executor(SimpleNamespace(order=SimpleNamespace(orderId=555)))
        import time as _t
        ex._order_times = [_t.time()] * 8  # 8 entries already placed this window
        ok = await ex.place_roll_combo(_prepared(), "T-A", "T-B", [], 0.0)
        assert ok is False
        ex._ibkr.place_order.assert_not_awaited()

    async def test_placement_failure_returns_false_and_registers_nothing(self, monkeypatch):
        monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
        ex = _bare_executor(place_order_result=None)  # broker refused
        ok = await ex.place_roll_combo(_prepared(), "T-A", "T-B", [], 0.0)
        assert ok is False
        assert ex._pending_roll_orders == {}


class TestPlaceRollComboHappyPath:
    async def test_registers_pending_roll_order_with_correct_shape(self, monkeypatch):
        monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
        ex = _bare_executor(SimpleNamespace(order=SimpleNamespace(orderId=555)))
        carried = [{"strike": 520.0, "right": "C", "action": "SELL"}]
        ok = await ex.place_roll_combo(_prepared(), "T-A", "T-B", carried, -0.75)
        assert ok is True
        pr = ex._pending_roll_orders[555]
        assert pr.closing_trade_id == "T-A"
        assert pr.opening_trade_id == "T-B"
        assert pr.closing_conids == {1, 2}
        assert pr.opening_conids == {3, 4}
        assert pr.carried_legs == carried
        assert pr.carried_value == -0.75

    async def test_ledger_attribution_map_set_to_closing_trade(self, monkeypatch):
        monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
        ex = _bare_executor(SimpleNamespace(order=SimpleNamespace(orderId=555)))
        await ex.place_roll_combo(_prepared(), "T-A", "T-B", [], 0.0)
        assert ex._order_trade_map[555] == "T-A"

    async def test_pending_orders_and_pending_exit_orders_untouched(self, monkeypatch):
        """Full isolation: check_fills' entry/exit loops iterate their OWN
        dicts directly, never broker orders broadly -- a roll order never
        added to either must never be visited by them."""
        monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
        ex = _bare_executor(SimpleNamespace(order=SimpleNamespace(orderId=555)))
        await ex.place_roll_combo(_prepared(), "T-A", "T-B", [], 0.0)
        assert ex._pending_orders == {}
        assert ex._pending_exit_orders == {}


# ---------------------------------------------------------------------------
# executor.py -- check_fills' roll-recognition block
# ---------------------------------------------------------------------------

def _fill(con_id: int, side: str, price: float, shares: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(
        execution=SimpleNamespace(execId=f"e{con_id}", side=side, shares=shares, price=price),
        commissionReport=SimpleNamespace(commission=0.65, realizedPNL=0.0),
        contract=SimpleNamespace(conId=con_id, secType="OPT"),
    )


def _bag_row() -> SimpleNamespace:
    return SimpleNamespace(
        execution=SimpleNamespace(execId="ebag", side="BOT", shares=1.0, price=-0.75),
        commissionReport=SimpleNamespace(commission=0.0, realizedPNL=0.0),
        contract=SimpleNamespace(conId=BAG_CON_ID, secType="BAG"),
    )


def _broker_trade(order_id: int, status: str, fills: list) -> SimpleNamespace:
    return SimpleNamespace(
        order=SimpleNamespace(orderId=order_id, permId=order_id * 10),
        orderStatus=SimpleNamespace(status=status, filled=1, remaining=0),
        contract=SimpleNamespace(symbol="SPY", secType="BAG"),
        fills=fills,
    )


def _real_executor(state, ib) -> TradeExecutor:
    return TradeExecutor(ibkr_client=FakeIBKRClient(ib), state=state,
                         circuit_breaker=MagicMock(), order_timeout=240)


@pytest.fixture(autouse=True)
def _cwd_tmp(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


def _seed_roll_in_flight(tmp_path):
    """A real A (CLOSING) + B (PENDING, rolled_from_trade_id=A) pair, exactly
    as _execute_roll would have left them right before a fill lands."""
    from ait.execution.roll_booking import cas_roll_closing_trade, seed_roll_opening_trade

    st = StateManager(db_path=tmp_path / "ait_state.db")
    a = _trade()
    st.record_trade(a)
    assert cas_roll_closing_trade(st, a.trade_id) is True
    a = st.get_trade_by_id(a.trade_id)
    roll_target = RollTarget(
        new_short=OptionContract(symbol="SPY", expiry="2027-01-15", strike=720.0, right="P",
                                 bid=0, ask=0, last=0, volume=0, open_interest=0,
                                 implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        new_long=OptionContract(symbol="SPY", expiry="2027-01-15", strike=710.0, right="P",
                                bid=0, ask=0, last=0, volume=0, open_interest=0,
                                implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        expiry="2027-01-15",
    )
    opening_legs = [
        {"strike": 720.0, "right": "P", "action": "SELL", "expiry": "2027-01-15", "conId": 3},
        {"strike": 710.0, "right": "P", "action": "BUY", "expiry": "2027-01-15", "conId": 4},
    ]
    b = seed_roll_opening_trade(st, a, opening_legs, roll_target, 1, entry_price_estimate=0.75)
    return st, a, b


class TestCheckFillsRollRecognition:
    async def test_filled_roll_order_books_via_book_roll_fill(self, tmp_path):
        st, a, b = _seed_roll_in_flight(tmp_path)
        ib = FakeIB()
        ib.all_trades = [_broker_trade(555, "Filled", [
            _fill(1, "BOT", 1.00), _fill(2, "SLD", 0.20),
            _fill(3, "SLD", 0.90), _fill(4, "BOT", 0.15),
            _bag_row(),
        ])]
        ex = _real_executor(st, ib)
        from ait.execution.roll_booking import PendingRollOrder
        ex._pending_roll_orders[555] = PendingRollOrder(
            order_id=555, closing_trade_id=a.trade_id, opening_trade_id=b.trade_id,
            closing_conids={1, 2}, opening_conids={3, 4}, carried_legs=[], carried_value=0.0,
        )

        await ex.check_fills()

        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.CLOSED
        assert st.get_trade_by_id(b.trade_id).status == TradeStatus.FILLED
        assert 555 not in ex._pending_roll_orders

    async def test_bag_summary_row_excluded_from_leg_fills(self, tmp_path):
        """The BAG row's price (-0.75, a combo-level summary) must never be
        treated as a leg fill -- if it were, split_roll_fill_price would
        refuse (conId not in either set) and the roll would never book."""
        st, a, b = _seed_roll_in_flight(tmp_path)
        ib = FakeIB()
        ib.all_trades = [_broker_trade(555, "Filled", [
            _fill(1, "BOT", 1.00), _fill(2, "SLD", 0.20),
            _fill(3, "SLD", 0.90), _fill(4, "BOT", 0.15),
            _bag_row(),
        ])]
        ex = _real_executor(st, ib)
        from ait.execution.roll_booking import PendingRollOrder
        ex._pending_roll_orders[555] = PendingRollOrder(
            order_id=555, closing_trade_id=a.trade_id, opening_trade_id=b.trade_id,
            closing_conids={1, 2}, opening_conids={3, 4}, carried_legs=[], carried_value=0.0,
        )

        await ex.check_fills()

        # Booked successfully -- proves the BAG row didn't poison the split.
        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.CLOSED

    async def test_side_translation_bot_sld_to_buy_sell(self, tmp_path):
        """W61's own named risk: ib_insync fills carry 'BOT'/'SLD', not
        'BUY'/'SELL'. Verified via the actual booked P&L, which is only
        correct if the translation happened."""
        st, a, b = _seed_roll_in_flight(tmp_path)
        ib = FakeIB()
        ib.all_trades = [_broker_trade(555, "Filled", [
            _fill(1, "BOT", 1.00), _fill(2, "SLD", 0.20),
            _fill(3, "SLD", 0.90), _fill(4, "BOT", 0.15),
        ])]
        ex = _real_executor(st, ib)
        from ait.execution.roll_booking import PendingRollOrder
        ex._pending_roll_orders[555] = PendingRollOrder(
            order_id=555, closing_trade_id=a.trade_id, opening_trade_id=b.trade_id,
            closing_conids={1, 2}, opening_conids={3, 4}, carried_legs=[], carried_value=0.0,
        )

        await ex.check_fills()

        # closing_net = 1.00 - 0.20 = 0.80; entry=1.50 (credit) -> pnl = 70.
        # Only correct if BOT->BUY/SLD->SELL translated; a raw pass-through
        # would make split_roll_fill_price treat every leg as "not BUY" (the
        # str(side).upper() check only matches "BUY" literally), inverting
        # every sign and producing a materially different (wrong) P&L.
        a_row = st.get_trade_by_id(a.trade_id)
        assert a_row.realized_pnl == pytest.approx((1.50 - 0.80) * 100)

    async def test_still_working_order_is_left_alone(self, tmp_path):
        st, a, b = _seed_roll_in_flight(tmp_path)
        ib = FakeIB()
        ib.all_trades = [_broker_trade(555, "Submitted", [])]
        ex = _real_executor(st, ib)
        from ait.execution.roll_booking import PendingRollOrder
        ex._pending_roll_orders[555] = PendingRollOrder(
            order_id=555, closing_trade_id=a.trade_id, opening_trade_id=b.trade_id,
            closing_conids={1, 2}, opening_conids={3, 4}, carried_legs=[], carried_value=0.0,
        )

        await ex.check_fills()

        assert 555 in ex._pending_roll_orders
        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.CLOSING
        assert st.get_trade_by_id(b.trade_id).status == TradeStatus.PENDING

    async def test_booking_failure_still_removes_the_order_not_retried(self, tmp_path):
        st, a, b = _seed_roll_in_flight(tmp_path)
        ib = FakeIB()
        # Unattributed conId -- split_roll_fill_price refuses -> booking fails.
        ib.all_trades = [_broker_trade(555, "Filled", [_fill(999, "BOT", 1.00)])]
        ex = _real_executor(st, ib)
        from ait.execution.roll_booking import PendingRollOrder
        ex._pending_roll_orders[555] = PendingRollOrder(
            order_id=555, closing_trade_id=a.trade_id, opening_trade_id=b.trade_id,
            closing_conids={1, 2}, opening_conids={3, 4}, carried_legs=[], carried_value=0.0,
        )

        await ex.check_fills()

        assert 555 not in ex._pending_roll_orders
        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.CLOSING  # untouched


# ---------------------------------------------------------------------------
# orchestrator.py -- _partition_roll_positions (pure)
# ---------------------------------------------------------------------------

class TestPartitionRollPositions:
    def test_rolls_disabled_everything_falls_through_to_remaining(self):
        positions = [_pos("T-1", action=PositionAction.ROLL), _pos("T-2", action=PositionAction.HOLD)]
        rolls, remaining = TradingOrchestrator._partition_roll_positions(positions, False)
        assert rolls == []
        assert remaining == positions

    def test_rolls_enabled_splits_by_action(self):
        roll_pos = _pos("T-1", action=PositionAction.ROLL)
        hold_pos = _pos("T-2", action=PositionAction.HOLD)
        rolls, remaining = TradingOrchestrator._partition_roll_positions(
            [roll_pos, hold_pos], True)
        assert rolls == [roll_pos]
        assert remaining == [hold_pos]

    def test_already_exiting_or_partial_never_included_either_way(self):
        exiting = _pos("T-1", action=PositionAction.ROLL, should_exit=True)
        partial = _pos("T-2", action=PositionAction.ROLL, partial_exit_quantity=1)
        for enabled in (False, True):
            rolls, remaining = TradingOrchestrator._partition_roll_positions(
                [exiting, partial], enabled)
            assert rolls == []
            assert remaining == []


# ---------------------------------------------------------------------------
# orchestrator.py -- _execute_roll
# ---------------------------------------------------------------------------

def _orch(state, executor, chain_service, settings=None) -> TradingOrchestrator:
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._state = state
    o._executor = executor
    o._ibkr = MagicMock()
    o._options_chain = chain_service
    o._settings = settings or SimpleNamespace(assignment_response=AssignmentResponseConfig())
    o._watchdog = MagicMock()
    return o


def _roll_target():
    return RollTarget(
        new_short=OptionContract(symbol="SPY", expiry="2027-01-15", strike=720.0, right="P",
                                 bid=0, ask=0, last=0, volume=0, open_interest=0,
                                 implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        new_long=OptionContract(symbol="SPY", expiry="2027-01-15", strike=710.0, right="P",
                                bid=0, ask=0, last=0, volume=0, open_interest=0,
                                implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        expiry="2027-01-15",
    )


def _prepared_combo(mid_net=-0.75, carried_legs=None, carried_value=0.0):
    return SimpleNamespace(
        bag=SimpleNamespace(), order=SimpleNamespace(orderRef=""),
        closing_legs=[{"conId": 1}, {"conId": 2}],
        opening_legs=[
            {"strike": 720.0, "right": "P", "action": "SELL", "expiry": "2027-01-15", "conId": 3},
            {"strike": 710.0, "right": "P", "action": "BUY", "expiry": "2027-01-15", "conId": 4},
        ],
        carried_legs=carried_legs or [], carried_value=carried_value, mid_net=mid_net,
        natural_net=mid_net,
    )


class TestExecuteRoll:
    async def test_no_target_found_writes_nothing(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        a = _trade()
        st.record_trade(a)
        chain = MagicMock()
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target", AsyncMock(return_value=None))
        o = _orch(st, MagicMock(), chain)
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)

        await o._execute_roll(_pos(a.trade_id))

        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.FILLED  # untouched

    async def test_prepare_failure_writes_nothing(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        a = _trade()
        st.record_trade(a)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=_roll_target()))
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo",
                            AsyncMock(return_value=None))
        o = _orch(st, MagicMock(), MagicMock())
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)

        await o._execute_roll(_pos(a.trade_id))

        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.FILLED

    async def test_cas_refused_writes_nothing_further(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        a = _trade()
        st.record_trade(a)
        st.transition(a.trade_id, (TradeStatus.FILLED,), TradeStatus.CLOSING)  # already claimed
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=_roll_target()))
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo",
                            AsyncMock(return_value=_prepared_combo()))
        o = _orch(st, MagicMock(), MagicMock())
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)

        await o._execute_roll(_pos(a.trade_id))

        # Only A pre-existed; no B row was ever seeded.
        import sqlite3
        with sqlite3.connect(st._db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert count == 1

    async def test_debit_cap_is_the_negated_roll_min_credit(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        a = _trade()
        st.record_trade(a)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=_roll_target()))
        captured = {}

        async def _capture(*args, **kwargs):
            captured.update(kwargs)
            return None  # short-circuit after capturing
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo", _capture)
        settings = SimpleNamespace(
            assignment_response=AssignmentResponseConfig(roll_min_credit=0.20))
        o = _orch(st, MagicMock(), MagicMock(), settings=settings)
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)

        await o._execute_roll(_pos(a.trade_id))

        assert captured["debit_cap"] == pytest.approx(-0.20)

    async def test_happy_path_seeds_b_and_places(self, tmp_path, monkeypatch):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        a = _trade()
        st.record_trade(a)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=_roll_target()))
        prepared = _prepared_combo()
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo",
                            AsyncMock(return_value=prepared))
        executor = MagicMock()
        executor.place_roll_combo = AsyncMock(return_value=True)
        o = _orch(st, executor, MagicMock())
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)

        await o._execute_roll(_pos(a.trade_id))

        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.CLOSING
        opened = [t for t in st.get_open_trades() if t.trade_id != a.trade_id]
        assert len(opened) == 1
        b = opened[0]
        assert b.status == TradeStatus.PENDING
        assert b.rolled_from_trade_id == a.trade_id
        # The real orderRef, stamped with B's real trade_id, not the
        # placeholder "PENDING" prepare_roll_combo was called with.
        parsed = parse_roll_order_ref(prepared.order.orderRef)
        assert parsed == (a.trade_id, b.trade_id)
        executor.place_roll_combo.assert_awaited_once()

    async def test_placement_failure_does_not_revert_a_or_b(self, tmp_path, monkeypatch):
        """No new recovery path invented -- the reconciler's own
        stale-pending sweep already handles an abandoned roll attempt."""
        st = StateManager(db_path=tmp_path / "ait_state.db")
        a = _trade()
        st.record_trade(a)
        monkeypatch.setattr("ait.bot.orchestrator.select_roll_target",
                            AsyncMock(return_value=_roll_target()))
        monkeypatch.setattr("ait.bot.orchestrator.prepare_roll_combo",
                            AsyncMock(return_value=_prepared_combo()))
        executor = MagicMock()
        executor.place_roll_combo = AsyncMock(return_value=False)
        o = _orch(st, executor, MagicMock())
        o._find_trade_record = lambda tid: st.get_trade_by_id(tid)

        await o._execute_roll(_pos(a.trade_id))

        assert st.get_trade_by_id(a.trade_id).status == TradeStatus.CLOSING
        opened = [t for t in st.get_open_trades() if t.trade_id != a.trade_id]
        assert len(opened) == 1 and opened[0].status == TradeStatus.PENDING


# ---------------------------------------------------------------------------
# CondorDefenseConfig.execute_rolls -- the gate itself
# ---------------------------------------------------------------------------

class TestExecuteRollsGate:
    def test_defaults_to_false(self):
        assert CondorDefenseConfig().execute_rolls is False

    def test_requires_both_flags(self):
        both_off = CondorDefenseConfig(enabled=False, execute_rolls=False)
        enabled_only = CondorDefenseConfig(enabled=True, execute_rolls=False)
        execute_only = CondorDefenseConfig(enabled=False, execute_rolls=True)
        both_on = CondorDefenseConfig(enabled=True, execute_rolls=True)
        for cfg in (both_off, enabled_only, execute_only):
            assert not (cfg.enabled and cfg.execute_rolls)
        assert both_on.enabled and both_on.execute_rolls

"""R27 G1 - the exit wedge.

Every exit builder CASes the trade to CLOSING on the line before its own
place_order call (W45). IBKRClient.place_order returns None when the broker
cannot be reached or placeOrder raises; before R27 G1 that None left the row
CLOSING with no broker order and no tracker, outside portfolio.check_positions,
until the next full reconcile, and paged nobody. The fix reverts the row
(CAS from CLOSING only), owes a reconcile and pages.

Every test drives the real builders / _execute_exit on the W45 rig. No
inspect.getsource.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ait.bot.state import TradeDirection, TradeRecord, TradeStatus
from ait.utils.time import now_et
from tests.test_hot_path_smoke import build_smoke_orchestrator

CONDOR_LEGS = [
    {"strike": 745.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 735.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 775.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 785.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
]
QUALIFIED = SimpleNamespace(conId=1, symbol="SPY", secType="OPT")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    smoke = build_smoke_orchestrator()
    try:
        yield smoke
    finally:
        smoke.restore()


def _seed(rig, *, strategy="iron_condor", contract_type="iron_condor", legs=None,
          strike=None, status=TradeStatus.FILLED, trade_id=None) -> TradeRecord:
    record = TradeRecord(
        trade_id=trade_id or f"T-G1-{strategy}-{status.value}",
        symbol="SPY", strategy=strategy,
        direction=TradeDirection.SHORT, status=status,
        entry_time=now_et().isoformat(), entry_price=1.20, quantity=1,
        contract_type=contract_type, strike=strike, expiry="2026-12-18",
        legs=json.dumps(legs if legs is not None else []),
    )
    rig.orch._state.record_trade(record)
    return record


def _placement_fails(rig) -> list:
    """place_order returns None, as IBKRClient.place_order does when the
    Gateway is gone or placeOrder raises. Records the calls."""
    calls: list = []

    async def _none(contract, order):
        calls.append(order)
        return None

    rig.orch._ibkr.place_order = _none
    return calls


def _status(rig, trade_id):
    return rig.orch._state.get_trade_by_id(trade_id).status


async def _pages(rig, trade_id):
    await rig.drain_notifications()
    return [n for n in rig.notifications if "EXIT NOT SENT" in n and "SPY" in n]


class TestMultiLeg:
    async def test_failed_placement_reverts_owes_a_reconcile_and_pages(self, rig):
        trade = _seed(rig, legs=CONDOR_LEGS)
        rig.ibkr.ib.quote = (-1.30, -1.10)
        calls = _placement_fails(rig)
        rig.orch._reconcile_pending = False

        result = await rig.orch._close_multi_leg(trade, CONDOR_LEGS)

        assert result is None
        assert len(calls) == 1, "the builder did reach placement"
        assert _status(rig, trade.trade_id) == TradeStatus.FILLED, (
            "an exit that never left the client must not leave the row CLOSING")
        assert rig.orch._reconcile_pending is True
        assert len(await _pages(rig, trade.trade_id)) == 1


class TestSingleLeg:
    async def test_sell_to_close_branch(self, rig):
        trade = _seed(rig, strategy="long_call", contract_type="call", strike=750.0)
        rig.ibkr.ib.quote = (2.00, 2.20)
        _placement_fails(rig)

        assert await rig.orch._close_single_leg(trade, QUALIFIED, "SELL") is None
        assert _status(rig, trade.trade_id) == TradeStatus.FILLED
        assert rig.orch._reconcile_pending is True
        assert len(await _pages(rig, trade.trade_id)) == 1

    async def test_buy_to_close_branch(self, rig):
        trade = _seed(rig, strategy="cash_secured_put", contract_type="put", strike=740.0)
        rig.ibkr.ib.quote = (1.80, 2.00)
        _placement_fails(rig)

        assert await rig.orch._close_single_leg(trade, QUALIFIED, "BUY") is None
        assert _status(rig, trade.trade_id) == TradeStatus.FILLED
        assert len(await _pages(rig, trade.trade_id)) == 1


class TestPriorStatusRestored:
    async def test_partial_comes_back_as_partial(self, rig):
        trade = _seed(rig, legs=CONDOR_LEGS, status=TradeStatus.PARTIAL)
        rig.ibkr.ib.quote = (-1.30, -1.10)
        _placement_fails(rig)

        await rig.orch._close_multi_leg(trade, CONDOR_LEGS)

        assert _status(rig, trade.trade_id) == TradeStatus.PARTIAL


class TestThroughExecuteExit:
    async def test_the_position_is_monitored_again_and_the_next_exit_places(self, rig):
        trade = _seed(rig, legs=CONDOR_LEGS, trade_id="T-G1-EXEC")
        rig.ibkr.ib.quote = (-1.30, -1.10)

        async def _live(_t):
            return "live"
        rig.orch._reconciler.position_liveness = _live
        pos = SimpleNamespace(trade_id=trade.trade_id, symbol="SPY",
                              exit_reason="touch_stop", unrealized_pnl=-120.0,
                              current_price=None, dte=10)

        real_place = rig.orch._ibkr.place_order
        _placement_fails(rig)
        await rig.orch._execute_exit(pos)

        assert _status(rig, trade.trade_id) == TradeStatus.FILLED, (
            "after a failed exit the row must be back where check_positions looks")
        assert not rig.orch._executor._pending_exit_orders

        # Gateway back: the next exit goes out and is tracked.
        rig.orch._ibkr.place_order = real_place
        rig.orch._exit_attempts = {}
        await rig.orch._execute_exit(pos)

        assert _status(rig, trade.trade_id) == TradeStatus.CLOSING
        tracked = [p.trade_id for p in rig.orch._executor._pending_exit_orders.values()]
        assert tracked == [trade.trade_id]


class TestCasGuard:
    async def test_a_row_moved_meanwhile_is_not_resurrected(self, rig):
        trade = _seed(rig, legs=CONDOR_LEGS)
        rig.ibkr.ib.quote = (-1.30, -1.10)

        async def _closed_meanwhile(contract, order):
            # Something else (a reconcile booking the trade gone) moved the row.
            rig.orch._state.update_trade_status(
                trade.trade_id, TradeStatus.CLOSED, from_statuses=(TradeStatus.CLOSING,))
            return None
        rig.orch._ibkr.place_order = _closed_meanwhile

        await rig.orch._close_multi_leg(trade, CONDOR_LEGS)

        assert _status(rig, trade.trade_id) == TradeStatus.CLOSED


class TestPageThrottle:
    async def test_two_failures_one_page(self, rig):
        trade = _seed(rig, legs=CONDOR_LEGS)
        rig.ibkr.ib.quote = (-1.30, -1.10)
        _placement_fails(rig)

        await rig.orch._close_multi_leg(trade, CONDOR_LEGS)
        await rig.orch._close_multi_leg(trade, CONDOR_LEGS)

        assert _status(rig, trade.trade_id) == TradeStatus.FILLED
        assert len(await _pages(rig, trade.trade_id)) == 1


class TestSuccessUnchanged:
    async def test_a_placed_exit_stays_closing_and_pages_nothing(self, rig):
        trade = _seed(rig, legs=CONDOR_LEGS)
        rig.ibkr.ib.quote = (-1.30, -1.10)
        rig.orch._reconcile_pending = False

        result = await rig.orch._close_multi_leg(trade, CONDOR_LEGS)

        assert result is not None
        assert _status(rig, trade.trade_id) == TradeStatus.CLOSING
        assert rig.orch._reconcile_pending is False
        assert await _pages(rig, trade.trade_id) == []

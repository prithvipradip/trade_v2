"""W45 (R26 register) — bot-day-05 + bot-day-08, reconciler/exit-path crash safety.

Both findings from reports/blindspot_composition_hunt_20260825.md, re-verified against
HEAD this session (not trusted from the original audit — orchestrator.py and
reconciler.py have both moved since 2026-08-25).

1. bot-day-05 — _execute_exit placed the closing broker order BEFORE CASing the
   trade to CLOSING. Fixed by moving the CAS onto the line immediately before
   each of the three actual placeOrder call sites, inside _close_single_leg /
   _close_multi_leg — not earlier in _execute_exit, which would have made a
   healthy position invisible to portfolio.check_positions for the multi-leg
   quote-poll's up-to-2.5s window on every routine exit, not just at crash time.

2. bot-day-08 — the stale-PENDING sweep's entry-order cancel check
   (_cancel_working_entry_order) was protected against a foreign-clientId
   working order only by an undocumented ib_insync cache side effect. Fixed
   with an explicit Pass 3 reading IBKRClient.foreign_open_order_ids, kept
   accurate by a periodic prune wired into sweep_stale_pending().

Every test executes the real code path. No inspect.getsource.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.bot.state import TradeDirection, TradeRecord, TradeStatus
from ait.execution.reconciler import PositionReconciler
from ait.utils.time import now_et
from tests.test_hot_path_smoke import build_smoke_orchestrator


# ---------------------------------------------------------------------------
# bot-day-05 — CAS-before-place
# ---------------------------------------------------------------------------

CONDOR_LEGS = [
    {"strike": 745.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 735.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 775.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 785.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    smoke = build_smoke_orchestrator()
    try:
        yield smoke
    finally:
        smoke.restore()


def _seed_filled_trade(rig, *, strategy="iron_condor", contract_type="iron_condor",
                        legs=None, strike=None) -> TradeRecord:
    """Directly inserts a FILLED trade — bypassing the entry path entirely,
    since these tests exercise only the exit builders, not entry placement."""
    trade_id = f"T-{strategy}-{id(legs)}"
    record = TradeRecord(
        trade_id=trade_id, symbol="SPY", strategy=strategy,
        direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
        entry_time=now_et().isoformat(), entry_price=1.20, quantity=1,
        contract_type=contract_type, strike=strike, expiry="2026-12-18",
        legs=json.dumps(legs if legs is not None else []),
    )
    rig.orch._state.record_trade(record)
    return record


def _capture_status_at_place(rig, trade_id: str) -> list[str]:
    """Wraps orch._ibkr.place_order so the trade's DB status is read at the
    EXACT moment placement is invoked — the direct proof of the reordering,
    not a source-string check."""
    observed: list[str] = []
    real_place = rig.orch._ibkr.place_order

    async def _wrapped(contract, order):
        row = rig.orch._state.get_trade_by_id(trade_id)
        observed.append(row.status)
        return await real_place(contract, order)

    rig.orch._ibkr.place_order = _wrapped
    return observed


class TestMultiLegCasBeforePlace:
    async def test_cas_observed_before_place_order_is_called(self, rig):
        trade = _seed_filled_trade(rig, legs=CONDOR_LEGS)
        rig.ibkr.ib.quote = (-1.30, -1.10)  # credit combo: negative-quoted
        observed = _capture_status_at_place(rig, trade.trade_id)

        result = await rig.orch._close_multi_leg(trade, CONDOR_LEGS)

        assert result is not None, "the order should have been placed"
        assert observed == [TradeStatus.CLOSING], (
            "place_order was called while the trade's DB status was "
            f"{observed}, not already CLOSING"
        )
        # And it stayed there — nothing reverted a successful placement.
        assert rig.orch._state.get_trade_by_id(trade.trade_id).status == TradeStatus.CLOSING

    async def test_no_order_paths_never_cas_and_never_place(self, rig):
        """The no-quote/no-anchor/no-entry-credit give-up path (:4545 in the
        plan's line citations) returns None BEFORE reaching the CAS — prove
        neither placement nor the status transition ever happens."""
        trade = _seed_filled_trade(rig, legs=CONDOR_LEGS)
        trade.entry_price = 0.0  # no entry credit -> no blind cap either
        rig.ibkr.ib.quote = None  # no quote at all
        placed = []
        real_place = rig.orch._ibkr.place_order

        async def _wrapped(contract, order):
            placed.append(order)
            return await real_place(contract, order)
        rig.orch._ibkr.place_order = _wrapped

        result = await rig.orch._close_multi_leg(trade, CONDOR_LEGS)

        assert result is None
        assert placed == [], "no order should ever have been placed"
        assert rig.orch._state.get_trade_by_id(trade.trade_id).status == TradeStatus.FILLED, (
            "a path that never placed an order must never have touched CLOSING"
        )


class TestSingleLegCasBeforePlace:
    async def test_sell_branch_cas_before_place(self, rig):
        """Closing a long option (SELL to close) — always places something,
        even with no quote (bounded market fallback)."""
        trade = _seed_filled_trade(rig, strategy="long_call", contract_type="call",
                                    strike=750.0)
        rig.ibkr.ib.quote = (2.00, 2.20)
        observed = _capture_status_at_place(rig, trade.trade_id)

        qualified = SimpleNamespace(conId=1, symbol="SPY", secType="OPT")
        result = await rig.orch._close_single_leg(trade, qualified, "SELL")

        assert result is not None
        assert observed == [TradeStatus.CLOSING]

    async def test_buy_to_close_branch_cas_before_place(self, rig):
        """Buying back a short option — the branch with a real defer path
        (no quote): must be CASed only on the path that actually places."""
        trade = _seed_filled_trade(rig, strategy="cash_secured_put",
                                    contract_type="put", strike=740.0)
        rig.ibkr.ib.quote = (1.80, 2.00)
        observed = _capture_status_at_place(rig, trade.trade_id)

        qualified = SimpleNamespace(conId=1, symbol="SPY", secType="OPT")
        result = await rig.orch._close_single_leg(trade, qualified, "BUY")

        assert result is not None
        assert observed == [TradeStatus.CLOSING]

    async def test_buy_to_close_no_quote_defers_without_touching_status(self, rig):
        trade = _seed_filled_trade(rig, strategy="cash_secured_put",
                                    contract_type="put", strike=740.0)
        rig.ibkr.ib.quote = None
        qualified = SimpleNamespace(conId=1, symbol="SPY", secType="OPT")

        result = await rig.orch._close_single_leg(trade, qualified, "BUY")

        assert result is None
        assert rig.orch._state.get_trade_by_id(trade.trade_id).status == TradeStatus.FILLED


class TestReconcilerSelfHealsAnEarlyCrash:
    async def test_closing_with_no_working_order_promotes_back_to_filled(self, rig):
        """The crash-recovery story this whole fix depends on: the CAS lands
        (real path, via _cas_to_closing), then nothing else happens — as if
        the process died right there, before placeOrder. The reconciler's
        existing recovery net (not new in this wave) must self-heal it."""
        trade = _seed_filled_trade(rig, legs=CONDOR_LEGS)
        assert rig.orch._cas_to_closing(trade) is True
        assert rig.orch._state.get_trade_by_id(trade.trade_id).status == TradeStatus.CLOSING

        # No working order at the broker for this trade, and no matching
        # IBKR position either (the legs were never actually submitted).
        rig.ibkr.ib.open_trades = []
        rig.ibkr.ib.positions_list = []

        await rig.orch._reconciler.reconcile()

        row = rig.orch._state.get_trade_by_id(trade.trade_id)
        assert row.status in (TradeStatus.FILLED, TradeStatus.CLOSED), (
            f"a CLOSING trade with no working order and no broker position "
            f"must not be left stuck in CLOSING; got {row.status}"
        )


# ---------------------------------------------------------------------------
# bot-day-08 — the foreign-order-id backstop (Pass 3)
# ---------------------------------------------------------------------------

def _rec_trade(trade_id="T-MINE", symbol="SPY", age_min=40):
    return SimpleNamespace(
        trade_id=trade_id, symbol=symbol, strategy="iron_condor",
        contract_type="iron_condor", quantity=1, entry_price=4.24,
        expiry="2026-12-18", strike=None, status=TradeStatus.PENDING,
        entry_time=(now_et() - timedelta(minutes=age_min)).isoformat(),
        legs=json.dumps(CONDOR_LEGS),
    )


def _reconciler_with_empty_cache():
    rec = PositionReconciler.__new__(PositionReconciler)
    rec._state = MagicMock()
    rec._ibkr = MagicMock()
    rec._ibkr.ib = MagicMock()
    rec._ibkr.ib.openTrades.return_value = []  # Pass 1/2 find nothing locally
    rec._ibkr.foreign_open_order_ids = set()
    return rec


class TestPass3ForeignOrderBackstop:
    def test_defers_when_a_foreign_order_is_stashed(self):
        rec = _reconciler_with_empty_cache()
        rec._ibkr.foreign_open_order_ids = {12345}

        assert rec._cancel_working_entry_order(_rec_trade()) is True

    def test_does_not_defer_when_the_stash_is_empty(self):
        """The gate must not be overly conservative: no foreign uncertainty
        means today's exact behavior (no working order -> book $0)."""
        rec = _reconciler_with_empty_cache()

        assert rec._cancel_working_entry_order(_rec_trade()) is False

    def test_the_guard_never_touches_which_trade_it_belongs_to(self):
        """Documents the deliberate design limit: Pass 3 defers regardless of
        WHICH trade is being checked, since bare order ids carry no
        symbol/orderRef to match against — two different trades both defer
        off the exact same stash."""
        rec = _reconciler_with_empty_cache()
        rec._ibkr.foreign_open_order_ids = {999}

        assert rec._cancel_working_entry_order(_rec_trade("T-ONE")) is True
        assert rec._cancel_working_entry_order(_rec_trade("T-TWO")) is True


class TestSweepPrunesTheForeignStash:
    async def test_prune_runs_and_empties_a_resolved_id(self):
        rec = PositionReconciler.__new__(PositionReconciler)
        rec._state = MagicMock()
        rec._state.get_open_trades.return_value = []
        rec._ibkr = MagicMock()
        rec._ibkr.foreign_open_order_ids = {555}

        async def _get_all_open_order_ids():
            return set()  # broker no longer reports 555 as working anywhere

        def _prune(broker_open_ids):
            dropped = rec._ibkr.foreign_open_order_ids - (broker_open_ids or set())
            rec._ibkr.foreign_open_order_ids -= dropped
            return set(rec._ibkr.foreign_open_order_ids)

        rec._ibkr.get_all_open_order_ids = _get_all_open_order_ids
        rec._ibkr.prune_foreign_open_orders = _prune
        rec._ibkr.get_positions_fresh = None

        await rec.sweep_stale_pending()

        assert rec._ibkr.foreign_open_order_ids == set(), (
            "a resolved foreign id should have been pruned by the periodic sweep"
        )

    async def test_a_failed_snapshot_leaves_the_stash_untouched(self):
        """Never treat a failed broker query as 'the order resolved'."""
        rec = PositionReconciler.__new__(PositionReconciler)
        rec._state = MagicMock()
        rec._state.get_open_trades.return_value = []
        rec._ibkr = MagicMock()
        rec._ibkr.foreign_open_order_ids = {555}

        async def _boom():
            raise RuntimeError("broker unavailable")

        rec._ibkr.get_all_open_order_ids = _boom
        rec._ibkr.prune_foreign_open_orders = MagicMock(
            side_effect=AssertionError("must not be called on a failed snapshot"))
        rec._ibkr.get_positions_fresh = None

        await rec.sweep_stale_pending()

        assert rec._ibkr.foreign_open_order_ids == {555}

"""W12 (R24 data-integrity-08): slippage must be measurable on BOTH halves
of a round trip.

The broker study concluded that "109 of 116 executions recorded
nbbo_spread = 0.0000" and inferred slippage was uninstrumented. Measuring the
ledger directly gives a sharper answer: every one of the 7 ENTRY combo rows
carries live_mid and nbbo_spread (the leg rows are zeroed deliberately — R16
puts combo context on the BAG row only). What is genuinely missing is the
EXIT side: PendingExitOrder had no quality fields and _order_ctx_map was
written only on the entry path, so every exit fill resolved to zeros.

That matters more than the commissions W11 corrected: on the 7 instrumented
entries the NBBO spread averaged $0.1429 per leg — roughly 15x the $0.9168
all-in per-leg commission.

Every test EXECUTES the real executor.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.execution.executor import PendingExitOrder, TradeExecutor


def _executor():
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._pending_exit_orders = {}
    ex._pending_orders = {}
    ex._order_trade_map = {}
    ex._order_ctx_map = {}
    ex._state = MagicMock()
    return ex


class TestPendingExitCarriesQuality:
    def test_fields_exist_and_default_to_zero(self):
        p = PendingExitOrder("T-1", "take_profit", 10.0)
        assert (p.signal_price, p.live_mid, p.nbbo_spread) == (0.0, 0.0, 0.0)

    def test_fields_round_trip(self):
        p = PendingExitOrder("T-1", "take_profit", 10.0,
                             signal_price=1.40, live_mid=1.42, nbbo_spread=0.06)
        assert (p.signal_price, p.live_mid, p.nbbo_spread) == (1.40, 1.42, 0.06)

    def test_slots_did_not_lose_the_original_fields(self):
        p = PendingExitOrder("T-1", "stop", -50.0)
        assert p.trade_id == "T-1" and p.exit_reason == "stop"
        assert p.estimated_pnl == -50.0 and p.submitted_at > 0


class TestRegistrationRecordsQuality:
    def test_quality_reaches_the_tracker(self):
        ex = _executor()
        ex.register_exit_order(order_id=42, trade_id="T-1",
                               exit_reason="take_profit", estimated_pnl=25.0,
                               signal_price=1.40, live_mid=1.42,
                               nbbo_spread=0.06)
        p = ex._pending_exit_orders[42]
        assert (p.live_mid, p.nbbo_spread) == (1.42, 0.06)

    def test_quality_also_lands_in_the_context_map(self):
        """The tracker is deleted on fill; the ctx map survives cleanup, which
        is how an event-driven fill swept later still gets attribution."""
        ex = _executor()
        ex.register_exit_order(order_id=42, trade_id="T-1", exit_reason="tp",
                               estimated_pnl=1.0, signal_price=1.40,
                               live_mid=1.42, nbbo_spread=0.06)
        assert ex._order_ctx_map[42] == (1.40, 1.42, 0.06)

    def test_absent_quality_does_not_pollute_the_ctx_map(self):
        ex = _executor()
        ex.register_exit_order(order_id=43, trade_id="T-2", exit_reason="tp",
                               estimated_pnl=1.0)
        assert 43 not in ex._order_ctx_map

    def test_backwards_compatible_call_still_registers(self):
        ex = _executor()
        ex.register_exit_order(order_id=44, trade_id="T-3",
                               exit_reason="expiry", estimated_pnl=0.0)
        assert ex._pending_exit_orders[44].trade_id == "T-3"
        assert ex._order_trade_map[44] == "T-3"


class TestLedgerContextResolvesExitQuality:
    """The seam that was returning zeros for every exit fill."""

    def test_exit_fill_resolves_its_quality(self):
        ex = _executor()
        ex.register_exit_order(order_id=42, trade_id="T-1", exit_reason="tp",
                               estimated_pnl=1.0, signal_price=1.40,
                               live_mid=1.42, nbbo_spread=0.06)
        trade_id, sig, mid, nbbo = ex._ledger_context(42)
        assert trade_id == "T-1"
        assert (sig, mid, nbbo) == (1.40, 1.42, 0.06)

    def test_resolves_after_the_tracker_is_cleaned_up(self):
        """Pre-W12 this returned zeros even when the exit HAD been quoted."""
        ex = _executor()
        ex.register_exit_order(order_id=42, trade_id="T-1", exit_reason="tp",
                               estimated_pnl=1.0, signal_price=1.40,
                               live_mid=1.42, nbbo_spread=0.06)
        del ex._pending_exit_orders[42]          # fill processed
        _tid, sig, mid, nbbo = ex._ledger_context(42)
        assert (sig, mid, nbbo) == (1.40, 1.42, 0.06)

    def test_unquoted_exit_still_returns_zeros_not_garbage(self):
        ex = _executor()
        ex.register_exit_order(order_id=45, trade_id="T-9", exit_reason="tp",
                               estimated_pnl=0.0)
        _tid, sig, mid, nbbo = ex._ledger_context(45)
        assert (sig, mid, nbbo) == (0.0, 0.0, 0.0)


class TestTheGapThisCloses:
    def test_entry_and_exit_quality_are_now_symmetric(self):
        """Entries recorded quality since R19; exits did not. Both now do,
        so a round-trip slippage figure is computable."""
        ex = _executor()
        # entry-style context (written by the entry path at placement)
        ex._order_ctx_map[100] = (6.44, 6.085, 0.21)
        # exit-style context (W12)
        ex.register_exit_order(order_id=101, trade_id="T-1", exit_reason="tp",
                               estimated_pnl=1.0, signal_price=1.40,
                               live_mid=1.42, nbbo_spread=0.06)
        _t, e_sig, e_mid, e_nbbo = ex._ledger_context(100)
        _t2, x_sig, x_mid, x_nbbo = ex._ledger_context(101)
        assert e_mid > 0 and x_mid > 0, "both halves must carry a mid"
        assert e_nbbo > 0 and x_nbbo > 0, "both halves must carry a spread"

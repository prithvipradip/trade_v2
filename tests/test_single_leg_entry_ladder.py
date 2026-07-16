"""R14: single-leg entries must escalate toward marketable like combos do.

The bug: single-leg option entries placed one PASSIVE sub-mid limit and never
escalated (combos ladder 0.25 -> 0.60 -> 1.00 of the marketable offset). So a
single-leg order sat below the ask and was reconciled stale_pending_never_filled
even on a tight, healthy quote — observed live as a long put resting at 3.28
against a 3.32 ask, cancelled unfilled after 4 minutes.

The fix arms the SAME ladder machinery combos use, via
single_leg_entry_ladder(bid, ask) + the stash fields _execute_single_leg now
sets, so _reprice_pending_entries steps the order to the ask by 90s.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.execution.executor import (
    TradeExecutor,
    _ladder_limit,
    single_leg_entry_ladder,
)


# --------------------------------------------------------------------------
# single_leg_entry_ladder — the pricing/offset math
# --------------------------------------------------------------------------

class TestSingleLegEntryLadder:
    def test_full_step_is_marketable(self):
        """The whole point: at frac=1.0 the limit must reach/cross the ask, or
        the order can never fill. base=mid, so mid+offset must be >= ask."""
        bid, ask = 3.26, 3.32
        base, offset, cap = single_leg_entry_ladder(bid, ask)
        assert base == pytest.approx(3.29)          # mid
        full = _ladder_limit(base, offset, 1.0, is_credit=False)
        assert full >= ask                            # marketable
        assert cap == pytest.approx(full)             # cap == marketable ceiling

    def test_step0_starts_near_mid_for_improvement(self):
        """Step 0 (0.25x) should sit below the ask on a wider spread — price
        improvement early, marketable only if it has to escalate."""
        bid, ask = 1.00, 1.40           # wide 40c spread, mid 1.20
        base, offset, _ = single_leg_entry_ladder(bid, ask)
        step0 = _ladder_limit(base, offset, 0.25, is_credit=False)
        assert step0 < ask               # improvement vs paying the ask
        assert step0 >= base             # but at/above mid (a real bid to fill)

    def test_ladder_is_monotonic_up_to_marketable(self):
        bid, ask = 2.00, 2.10
        base, offset, cap = single_leg_entry_ladder(bid, ask)
        s0 = _ladder_limit(base, offset, 0.25, is_credit=False)
        s1 = _ladder_limit(base, offset, 0.60, is_credit=False)
        s2 = _ladder_limit(base, offset, 1.00, is_credit=False)
        assert s0 < s1 < s2
        assert s2 >= ask and s2 == pytest.approx(cap)

    def test_cap_never_overshoots_marketable(self):
        """The debit cap == the 1.0x marketable price, so the reprice ladder's
        min(limit, cap) can reach the ask but never pay through it unbounded."""
        base, offset, cap = single_leg_entry_ladder(0.80, 0.95)
        assert cap == pytest.approx(_ladder_limit(base, offset, 1.0, is_credit=False))


# --------------------------------------------------------------------------
# _execute_single_leg — arms the ladder stash for a debit long option
# --------------------------------------------------------------------------

def _executor() -> TradeExecutor:
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._ibkr = MagicMock()
    ex._ibkr.qualify_contract = AsyncMock(
        side_effect=lambda c: SimpleNamespace(conId=1, contract=c))
    ex._ibkr.place_order = AsyncMock(
        return_value=SimpleNamespace(order=SimpleNamespace(orderId=11)))
    ex._state = MagicMock()
    # start the stash zeroed, like execute_signal does before dispatch
    ex._last_base_mag = 0.0
    ex._last_full_offset = 0.0
    ex._last_debit_cap = 0.0
    return ex


def _put_signal(bid, ask, strategy="long_put"):
    contract = SimpleNamespace(expiry="2026-08-21", strike=290.0, right="P",
                               bid=bid, ask=ask)
    return SimpleNamespace(
        symbol="IWM", strategy_name=strategy, action="BUY",
        entry_price=(bid + ask) / 2 if bid and ask else 3.0,
        contract=contract, legs=None)


class TestExecuteSingleLegArmsLadder:
    async def test_debit_long_put_arms_ladder_and_prices_at_step0(self):
        ex = _executor()
        sig = _put_signal(3.26, 3.32)
        await ex._execute_single_leg(sig, 1, "T-SL")
        # stash armed so execute_signal registers a ladder-eligible PendingOrder
        assert ex._last_base_mag == pytest.approx(3.29)
        assert ex._last_full_offset > 0
        assert ex._last_debit_cap >= 3.32
        # the placed order is the step-0 ladder price (marketable-escalating),
        # NOT a passive sub-mid rest
        order = ex._ibkr.place_order.call_args.args[1]
        assert order.orderType == "LMT"
        assert order.lmtPrice >= 3.29          # at/above mid, escalates from here

    async def test_no_quote_falls_back_and_does_not_arm(self):
        """No bid/ask: keep the plain entry-price limit, ladder stays disarmed
        (base stays 0 -> _reprice_pending_entries skips it)."""
        ex = _executor()
        sig = _put_signal(0, 0)
        await ex._execute_single_leg(sig, 1, "T-SL")
        assert ex._last_base_mag == 0.0
        assert ex._last_full_offset == 0.0

    async def test_wide_spread_still_rejected(self):
        """The >15% stale/illiquid guard is unchanged — a 40% spread is refused
        before any ladder arming."""
        ex = _executor()
        sig = _put_signal(1.00, 1.60)          # 60c on 1.30 mid = 46%
        result = await ex._execute_single_leg(sig, 1, "T-SL")
        assert result is None
        ex._ibkr.place_order.assert_not_called()
        ex._state.transition.assert_called_once()


# --------------------------------------------------------------------------
# _reprice_pending_entries — the escalation actually fires for single-leg
# --------------------------------------------------------------------------

class TestSingleLegLadderEscalates:
    async def test_armed_single_leg_order_escalates_to_marketable(self):
        """End-to-end of the fix: a single-leg PendingOrder armed by the helper
        gets stepped toward the ask by _reprice_pending_entries once it ages —
        the exact escalation the old passive order never got."""
        from ait.execution.executor import PendingOrder

        base, offset, cap = single_leg_entry_ladder(3.26, 3.32)
        pend = PendingOrder(trade_id="T-SL", signal=MagicMock(), contracts=1,
                            base_price=base, full_offset=offset, is_credit=False)
        pend.debit_cap = cap
        pend.submitted_at = 0.0            # force it "old" so the ladder steps it

        ex = TradeExecutor.__new__(TradeExecutor)
        ex._pending_orders = {11: pend}
        order = SimpleNamespace(orderId=11, lmtPrice=0.0)
        placed = SimpleNamespace(order=order, contract=SimpleNamespace())
        ex._ibkr = MagicMock()
        ex._ibkr.ib.openTrades.return_value = [placed]
        ex._ibkr.ib.placeOrder = MagicMock()

        import time as _t
        real = _t.time
        try:
            _t.time = lambda: 1000.0        # age = 1000s -> final ladder step
            await ex._reprice_pending_entries()
        finally:
            _t.time = real

        # repriced to the marketable step, capped at the ask+buffer ceiling
        assert order.lmtPrice >= 3.32
        assert order.lmtPrice == pytest.approx(cap)
        ex._ibkr.ib.placeOrder.assert_called_once()

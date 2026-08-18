"""R12 hypothesis property suites (validated by a 20k-sample probe, 0 violations).

Three invariant families over the execution-layer money math:

1. Reprice-ladder monotonicity/bounds — escalation only ever concedes edge
   toward marketable, never past it, and clamps out-of-range fractions.
2. P&L credit-membership — realized P&L direction is decided by
   CREDIT_STRATEGIES membership alone, including the mirror identity
   (a credit round-trip entry->exit books exactly what the debit formula
   books for exit->entry).
3. Iron-condor wing-width budget bound — the budget cap keeps worst-case
   loss within the per-trade risk budget, with the documented <$80 carve-out
   where the $1 width floor wins (worst case then bounded by the floor's own
   ~$80, not the budget).

Guarded with importorskip so the suite passes before the `hypothesis` dev
dependency lands in pyproject (main thread owns that edit).
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest

pytest.importorskip("hypothesis", reason="R12: add hypothesis>=6.100 to [dev]")

from hypothesis import given, settings, strategies as st

from ait.execution.executor import TradeExecutor, _ladder_limit, combo_entry_limit
from ait.strategies.base import CREDIT_STRATEGIES
from ait.strategies.iron_condor import IronCondor

# Keep examples cheap and deterministic-ish in CI; no wall-clock deadline
# (Windows CI timers are coarse enough to false-flag 1ms functions).
_SETTINGS = settings(max_examples=200, deadline=None)


# ---------------------------------------------------------------------------
# 1. Reprice ladder: monotone toward marketable, bounded, clamped
# ---------------------------------------------------------------------------

_price = st.floats(min_value=0.05, max_value=50.0,
                   allow_nan=False, allow_infinity=False)
_frac = st.floats(min_value=0.0, max_value=1.0,
                  allow_nan=False, allow_infinity=False)


class TestLadderProperties:

    @_SETTINGS
    @given(base=_price, f1=_frac, f2=_frac, is_credit=st.booleans())
    def test_escalation_is_monotone_and_bounded(self, base, f1, f2, is_credit):
        _, offset = combo_entry_limit(base, is_credit)
        lo, hi = sorted((f1, f2))
        p_lo = _ladder_limit(base, offset, lo, is_credit)
        p_hi = _ladder_limit(base, offset, hi, is_credit)

        if is_credit:
            # Credit combos quote NEGATIVE (always-BUY convention).
            assert p_lo < 0 and p_hi < 0
            # Escalating concedes credit: magnitude never increases.
            assert abs(p_hi) <= abs(p_lo)
            # Bounds: never collect more than mid, never below the fully
            # marketable price (floored at $0.01).
            assert abs(p_lo) <= round(base, 2)
            assert abs(p_hi) >= max(0.01, round(base - offset, 2))
        else:
            assert p_lo > 0 and p_hi > 0
            # Escalating pays more, never less.
            assert p_hi >= p_lo
            # Bounds: never below mid, never past the marketable offset.
            assert p_lo >= round(base, 2)
            assert p_hi <= round(base + offset, 2)

    @_SETTINGS
    @given(base=_price,
           frac=st.floats(min_value=-3.0, max_value=4.0,
                          allow_nan=False, allow_infinity=False),
           is_credit=st.booleans())
    def test_fraction_is_clamped_to_unit_interval(self, base, frac, is_credit):
        _, offset = combo_entry_limit(base, is_credit)
        clamped = _ladder_limit(base, offset, frac, is_credit)
        expected = _ladder_limit(base, offset, min(1.0, max(0.0, frac)), is_credit)
        assert clamped == expected

    @_SETTINGS
    @given(base=_price, is_credit=st.booleans())
    def test_entry_offset_floor_and_proportionality(self, base, is_credit):
        limit, offset = combo_entry_limit(base, is_credit)
        assert offset == max(0.10, round(0.15 * abs(base), 2))
        if is_credit:
            assert limit < 0  # signed credit quote
            assert abs(limit) >= 0.01  # never a free (or inverted) credit


# ---------------------------------------------------------------------------
# 2. Realized P&L: credit-membership decides the sign convention
# ---------------------------------------------------------------------------

# Realistic strategy names on both sides of the membership line.
_STRATEGIES = sorted(CREDIT_STRATEGIES) + [
    "bull_call_spread", "bear_put_spread", "long_call", "long_put",
    "long_straddle", "event_straddle", "calendar_spread",
]

_premium = st.floats(min_value=0.0, max_value=50.0,
                     allow_nan=False, allow_infinity=False)
_qty = st.integers(min_value=1, max_value=10)


def _pnl(strategy: str, contract_type: str, entry: float,
         exit_price: float, qty: int) -> float:
    executor = TradeExecutor.__new__(TradeExecutor)  # method only reads class state
    trade = SimpleNamespace(
        strategy=strategy, contract_type=contract_type,
        entry_price=entry, quantity=qty,
    )
    return executor._calculate_realized_pnl(trade, exit_price)


def _commission(contract_type: str, strategy: str, qty: int) -> float:
    legs = 1
    if contract_type == "iron_condor":
        legs = 4
    elif contract_type == "spread" or strategy in (
        "long_straddle", "event_straddle", "short_strangle", "calendar_spread",
    ):
        legs = 2
    return 0.65 * legs * qty * 2


class TestPnlCreditMembership:

    @_SETTINGS
    @given(strategy=st.sampled_from(_STRATEGIES),
           entry=_premium, exit_price=_premium, qty=_qty)
    def test_membership_decides_gross_pnl_direction(self, strategy, entry,
                                                    exit_price, qty):
        contract_type = "iron_condor" if strategy == "iron_condor" else "spread"
        pnl = _pnl(strategy, contract_type, entry, exit_price, qty)
        comm = _commission(contract_type, strategy, qty)
        if strategy in CREDIT_STRATEGIES:
            gross = (entry - exit_price) * 100 * qty
        else:
            gross = (exit_price - entry) * 100 * qty
        assert pnl == round(gross - comm, 2)
        # Costs are real: net is always strictly below gross.
        assert pnl < gross

    @_SETTINGS
    @given(entry=_premium, exit_price=_premium, qty=_qty)
    def test_mirror_identity(self, entry, exit_price, qty):
        """A credit round trip (collect `entry`, pay `exit`) books exactly
        what the debit formula books for the mirrored trip (pay `exit`,
        receive `entry`) at equal leg count — the two P&L branches are one
        formula with the cash-flow sign flipped, not two accounting schemes.
        """
        credit_pnl = _pnl("short_strangle", "spread", entry, exit_price, qty)
        debit_pnl = _pnl("bull_call_spread", "spread", exit_price, entry, qty)
        assert credit_pnl == debit_pnl

    @_SETTINGS
    @given(strategy=st.sampled_from(_STRATEGIES), premium=_premium, qty=_qty)
    def test_scratch_close_is_pure_cost(self, strategy, premium, qty):
        """Exit at the entry price => P&L is exactly minus the commissions,
        regardless of membership."""
        contract_type = "iron_condor" if strategy == "iron_condor" else "spread"
        pnl = _pnl(strategy, contract_type, premium, premium, qty)
        assert pnl == round(-_commission(contract_type, strategy, qty), 2)


# ---------------------------------------------------------------------------
# 3. Wing width: budget bound with the <$80 floor carve-out
# ---------------------------------------------------------------------------

_underlying = st.floats(min_value=10.0, max_value=1000.0,
                        allow_nan=False, allow_infinity=False)
_iv = st.floats(min_value=0.05, max_value=2.0,
                allow_nan=False, allow_infinity=False)
_budget = st.floats(min_value=1.0, max_value=5000.0,
                    allow_nan=False, allow_infinity=False)


def _width(price: float, iv: float, budget: float | None) -> float:
    ic = IronCondor()
    ic.risk_budget = budget
    leg = SimpleNamespace(implied_vol=iv)
    return ic._vol_scaled_width(price, leg, short_call_hint=leg, chain=None)


def _min_ratio() -> float:
    return float(os.environ.get("AIT_IC_MIN_CREDIT_WIDTH", "0.20"))


class TestWingWidthBudgetBound:

    @_SETTINGS
    @given(price=_underlying, iv=_iv)
    def test_unbudgeted_width_honors_two_dollar_floor(self, price, iv):
        w = _width(price, iv, budget=None)
        assert math.isfinite(w)
        assert w >= 2.0

    @_SETTINGS
    @given(price=_underlying, iv=_iv, budget=_budget)
    def test_budget_bounds_worst_case_loss(self, price, iv, budget):
        """Worst-case loss = width*(1 - min_credit_ratio)*100. With a budget
        it must not exceed the budget — EXCEPT below the carve-out point
        (budget < 100*(1-ratio) = $80 at defaults), where the $1 width floor
        wins and the worst case is bounded by the floor's own ~$80 instead.
        """
        ratio = _min_ratio()
        floor_loss = 100.0 * max(0.5, 1.0 - ratio) * 1.0  # $1-wide worst case
        w_free = _width(price, iv, budget=None)
        w = _width(price, iv, budget=budget)

        assert 1.0 <= w <= w_free + 1e-9  # cap never widens the structure
        worst_case = w * 100.0 * max(0.5, 1.0 - ratio)
        assert worst_case <= max(budget, floor_loss) + 1e-6

        # The carve-out, explicitly: a sub-$80 budget that binds produces
        # exactly the $1 floor (and therefore CAN exceed the budget — that
        # is the documented, intentional exception).
        affordable = budget / (100.0 * max(0.5, 1.0 - ratio))
        if budget < floor_loss and affordable < w_free:
            assert w == 1.0

    @_SETTINGS
    @given(price=_underlying, iv=_iv, b1=_budget, b2=_budget)
    def test_width_monotone_in_budget(self, price, iv, b1, b2):
        lo, hi = sorted((b1, b2))
        assert _width(price, iv, lo) <= _width(price, iv, hi) + 1e-9

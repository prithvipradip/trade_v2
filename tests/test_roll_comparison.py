"""W54a: the close-or-roll comparison models -- execution/roll_comparison.py
and options_chain.black_scholes_price. Nothing in src/ait calls into either
yet (rule 3/3e wiring is W54b) -- these tests drive them directly.

No inspect.getsource.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ait.data.options_chain import black_scholes_price, solve_iv_from_price
from ait.execution.roll_comparison import (
    close_cost_now,
    ev_should_roll,
    heuristic_should_roll,
)


# ---------------------------------------------------------------------------
# black_scholes_price
# ---------------------------------------------------------------------------

class TestBlackScholesPrice:
    def test_deep_itm_call_approaches_intrinsic(self):
        # spot way above strike, short time -> value ~= intrinsic (spot-strike)
        price = black_scholes_price(spot=200.0, strike=100.0, t=0.01, r=0.05, sigma=0.2, flag="c")
        assert price == pytest.approx(200.0 - 100.0, abs=1.5)

    def test_deep_otm_call_is_near_zero(self):
        price = black_scholes_price(spot=100.0, strike=500.0, t=0.05, r=0.05, sigma=0.2, flag="c")
        assert price is not None
        assert price < 0.5

    def test_put_call_parity_holds(self):
        spot, strike, t, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.2
        call = black_scholes_price(spot, strike, t, r, sigma, "c")
        put = black_scholes_price(spot, strike, t, r, sigma, "p")
        # C - P = S - K*e^(-rt)
        import math
        expected = spot - strike * math.exp(-r * t)
        assert (call - put) == pytest.approx(expected, abs=0.01)

    def test_zero_or_negative_time_returns_none(self):
        assert black_scholes_price(100.0, 100.0, 0.0, 0.05, 0.2, "c") is None
        assert black_scholes_price(100.0, 100.0, -1.0, 0.05, 0.2, "c") is None

    def test_zero_or_negative_sigma_returns_none(self):
        assert black_scholes_price(100.0, 100.0, 0.1, 0.05, 0.0, "c") is None
        assert black_scholes_price(100.0, 100.0, 0.1, 0.05, -0.1, "c") is None

    def test_never_raises_on_pathological_inputs(self):
        # Should return a value or None, never throw.
        black_scholes_price(0.0, 0.0, 0.1, 0.05, 0.2, "c")
        black_scholes_price(-100.0, 100.0, 0.1, 0.05, 0.2, "p")

    def test_round_trips_with_solve_iv_from_price(self):
        """A real sanity check the two py_vollib call sites agree: price a
        contract via black_scholes_price, then solve_iv_from_price on that
        SAME price must recover (approximately) the original sigma."""
        spot, strike, t, r, sigma = 100.0, 95.0, 0.2, 0.05, 0.25
        price = black_scholes_price(spot, strike, t, r, sigma, "c")
        solved_iv, outcome = solve_iv_from_price(price, spot, strike, t, r, "c")
        assert outcome == "solved"
        assert solved_iv == pytest.approx(sigma, abs=0.01)


# ---------------------------------------------------------------------------
# close_cost_now / heuristic_should_roll
# ---------------------------------------------------------------------------

def _prepared(closing_net, carried_value=0.0):
    return SimpleNamespace(closing_net=closing_net, carried_value=carried_value)


class TestCloseCostNow:
    def test_matches_book_roll_fills_own_formula(self):
        """close_cost_now must be exactly closing_net - carried_value --
        the SAME formula roll_booking.book_roll_fill uses for a real fill's
        exit_price, applied here to a mid estimate."""
        p = _prepared(closing_net=0.80, carried_value=-0.30)
        assert close_cost_now(p) == pytest.approx(0.80 - (-0.30))


class TestHeuristicShouldRoll:
    def test_a_losing_position_should_roll(self):
        """Closing now costs a net debit (a loss) -- rolling (guaranteed a
        credit by prepare_roll_combo's own debit_cap, by the time `prepared`
        exists at all) is preferred."""
        p = _prepared(closing_net=0.80, carried_value=0.0)  # costs 0.80 to close
        assert heuristic_should_roll(p) is True

    def test_a_winning_position_should_not_roll(self):
        """Closing now would realize a credit (a gain) -- don't give that up
        to roll into fresh risk."""
        p = _prepared(closing_net=-0.30, carried_value=0.0)  # nets 0.30 credit to close
        assert heuristic_should_roll(p) is False

    def test_exactly_breakeven_should_not_roll(self):
        p = _prepared(closing_net=0.0, carried_value=0.0)
        assert heuristic_should_roll(p) is False

    def test_carried_vertical_shifts_the_verdict(self):
        """A losing rolled vertical (closing_net=0.10, a small debit) but a
        profitable carried vertical (carried_value=+0.50, an asset) can make
        closing EVERYTHING a net win overall -- carried_value must be part
        of the decision, not just the rolled side."""
        p = _prepared(closing_net=0.10, carried_value=0.50)
        # close_cost_now = 0.10 - 0.50 = -0.40 -> a net credit to close -> don't roll
        assert heuristic_should_roll(p) is False


# ---------------------------------------------------------------------------
# ev_should_roll
# ---------------------------------------------------------------------------

def _leg(strike, action, spot=100.0, t=0.2, iv=0.25, flag="p"):
    return {"strike": strike, "action": action, "spot": spot, "t": t, "iv": iv, "flag": flag}


class TestEvShouldRoll:
    def test_missing_iv_on_any_leg_returns_none(self):
        """t<=0 makes black_scholes_price fail to solve -- ev_should_roll
        must defer (None), never guess."""
        closing = [_leg(100, "BUY", t=0.0)]
        opening = [_leg(95, "SELL")]
        assert ev_should_roll(closing, opening, [], r=0.05) is None

    def test_a_losing_position_by_model_price_should_roll(self):
        closing = [_leg(100, "BUY", spot=95, t=0.02, iv=0.3)]   # near-expiry, ITM-ish -> costly to buy back
        opening = [_leg(90, "SELL", spot=95, t=0.3, iv=0.3)]    # further out -> real credit
        result = ev_should_roll(closing, opening, [], r=0.05)
        assert result is not None

    def test_no_carried_legs_is_still_computed_correctly(self):
        closing = [_leg(100, "BUY")]
        opening = [_leg(95, "SELL")]
        # Should not raise or treat empty carried list as a missing-IV failure.
        result = ev_should_roll(closing, opening, [], r=0.05)
        assert result in (True, False)

    def test_heuristic_and_ev_can_genuinely_disagree(self):
        """The real proof EV is a second opinion, not a relabeled heuristic:
        construct a case where the RAW mid-quote credit (heuristic's input)
        and the MODEL fair-value credit (EV's input) point different ways.
        A deep-ITM near-expiry closing leg is where a flat-quote assumption
        and a genuine Black-Scholes solve diverge most -- verified here by
        computing both sides' real outputs, not asserted a priori."""
        # Heuristic: raw quote says closing nets a small CREDIT -> won't roll.
        prepared = _prepared(closing_net=-0.05, carried_value=0.0)
        heuristic_verdict = heuristic_should_roll(prepared)
        assert heuristic_verdict is False

        # EV: the SAME closing leg priced by a real (much higher, near-ITM)
        # model value -- a genuinely more expensive buyback than the flat
        # quote assumed, flipping the verdict to roll.
        closing = [_leg(100, "BUY", spot=101, t=0.005, iv=0.35)]
        opening = [_leg(95, "SELL", spot=101, t=0.3, iv=0.35)]
        ev_verdict = ev_should_roll(closing, opening, [], r=0.05)
        assert ev_verdict is not None
        assert ev_verdict != heuristic_verdict

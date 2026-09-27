"""W55 (synthetic per-leg delta for the monitor loop): the two pure functions
`solve_iv_from_price` / `compute_greeks_from_sigma` (options_chain.py), plus
the proof `_calculate_greeks`'s own behavior didn't change when its pass-1
solve was refactored to call the first one.

Quotes/Greeks are generated with py_vollib itself at a known sigma, so
expected answers are exact -- same convention test_w35_chain_iv_after_the_solve.py
uses. Every test drives the real functions, no mocking of py_vollib itself.

No inspect.getsource.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

pytest.importorskip("py_vollib")
from py_vollib.black_scholes import black_scholes  # noqa: E402
from py_vollib.black_scholes.greeks.analytical import (  # noqa: E402
    delta as bs_delta,
    gamma as bs_gamma,
    theta as bs_theta,
    vega as bs_vega,
)

from ait.data.options_chain import (  # noqa: E402
    MIN_PHYSICAL_IV,
    compute_greeks_from_sigma,
    solve_iv_from_price,
)

SPOT, STRIKE, T, R = 500.0, 500.0, 30 / 365.0, 0.05
SIGMA = 0.22


class TestSolveIvFromPrice:
    def test_a_real_quote_solves_to_the_known_sigma(self):
        price = black_scholes("c", SPOT, STRIKE, T, R, SIGMA)
        iv, outcome = solve_iv_from_price(price, SPOT, STRIKE, T, R, "c")
        assert outcome == "solved"
        assert iv == pytest.approx(SIGMA, abs=0.005)

    def test_puts_solve_too(self):
        price = black_scholes("p", SPOT, STRIKE, T, R, SIGMA)
        iv, outcome = solve_iv_from_price(price, SPOT, STRIKE, T, R, "p")
        assert outcome == "solved"
        assert iv == pytest.approx(SIGMA, abs=0.005)

    def test_a_price_below_intrinsic_is_an_arbitrage_failure_not_other(self):
        """Deep ITM call priced at pure noise (below strike-spot intrinsic) --
        the quote is impossible, not a convergence miss. W19/w17-07's own
        distinction, now shared code."""
        iv, outcome = solve_iv_from_price(0.01, SPOT, 400.0, T, R, "c")
        assert outcome == "failed_arbitrage"
        assert iv is None

    def test_a_price_above_the_maximum_is_also_an_arbitrage_failure(self):
        iv, outcome = solve_iv_from_price(600.0, SPOT, 400.0, T, R, "c")
        assert outcome == "failed_arbitrage"
        assert iv is None

    def test_a_solve_below_the_physical_iv_floor_is_failed_other(self):
        """A genuine near-zero-vol price solves to exactly 0.0 (py_vollib's
        own convergence, not an arbitrage bound) -- below MIN_PHYSICAL_IV,
        so it must be reported as a failure, not accepted as a real IV."""
        tiny_vol_price = black_scholes("c", SPOT, STRIKE, T, R, 0.001)
        iv, outcome = solve_iv_from_price(tiny_vol_price, SPOT, STRIKE, T, R, "c")
        assert outcome == "failed_other"
        assert iv is None

    def test_outcome_is_always_one_of_the_three_named_strings(self):
        for price in (0.01, 5.0, 600.0):
            _, outcome = solve_iv_from_price(price, SPOT, 400.0, T, R, "c")
            assert outcome in ("solved", "failed_arbitrage", "failed_other")


class TestComputeGreeksFromSigma:
    def test_matches_py_vollib_directly_for_a_call(self):
        result = compute_greeks_from_sigma(SPOT, STRIKE, T, R, SIGMA, "c")
        assert result is not None
        d, g, th, v = result
        assert d == pytest.approx(bs_delta("c", SPOT, STRIKE, T, R, SIGMA))
        assert g == pytest.approx(bs_gamma("c", SPOT, STRIKE, T, R, SIGMA))
        assert th == pytest.approx(bs_theta("c", SPOT, STRIKE, T, R, SIGMA) / 365.0)
        assert v == pytest.approx(bs_vega("c", SPOT, STRIKE, T, R, SIGMA) / 100.0)

    def test_matches_py_vollib_directly_for_a_put(self):
        result = compute_greeks_from_sigma(SPOT, STRIKE, T, R, SIGMA, "p")
        assert result is not None
        d, g, th, v = result
        assert d == pytest.approx(bs_delta("p", SPOT, STRIKE, T, R, SIGMA))
        assert -1.0 <= d <= 0.0   # a put's delta is never positive

    def test_a_call_deltas_bounded_zero_to_one(self):
        result = compute_greeks_from_sigma(SPOT, STRIKE, T, R, SIGMA, "c")
        assert result is not None
        assert 0.0 <= result[0] <= 1.0

    def test_t_zero_raises_inside_py_vollib_and_is_reported_as_none(self):
        """A genuine py_vollib failure mode (confirmed empirically: t=0 -> a
        NaN-to-int ValueError deep inside py_lets_be_rational) -- never
        reachable via either real caller (both floor t at 0.001), but proves
        the function's own 'all four or nothing' contract on ANY exception."""
        assert compute_greeks_from_sigma(SPOT, STRIKE, 0.0, R, SIGMA, "c") is None


class TestCalculateGreeksRefactorChangedNothing:
    """The pass-1 solve loop in _calculate_greeks now calls solve_iv_from_price
    instead of inlining the same py_vollib call. This is covered end-to-end by
    the full existing suite (test_w35/w19/r16-data-fixes/w36, re-run unchanged
    as this wave's regression proof) -- this one extra check just pins the
    NEW module-level function is genuinely what gets called, not a second,
    silently-diverging copy."""

    def test_calculate_greeks_and_solve_iv_from_price_agree_on_the_same_quote(self):
        import ait.data.options_chain as OC
        from ait.utils.time import today_et

        expiry = today_et() + timedelta(days=30)
        price = black_scholes("c", SPOT, STRIKE, 30 / 365.0, R, SIGMA)
        contract = OC.OptionContract(
            symbol="SPY", expiry=expiry, strike=STRIKE, right="C",
            bid=round(price - 0.02, 4), ask=round(price + 0.02, 4), last=price,
            volume=100, open_interest=200, implied_vol=0.0, delta=0.0)
        chain = OC.OptionsChain(symbol="SPY", underlying_price=SPOT, expiry=expiry,
                                calls=[contract], puts=[])
        OC.OptionsChainService.__new__(OC.OptionsChainService)._calculate_greeks(chain)

        standalone_iv, outcome = solve_iv_from_price(price, SPOT, STRIKE, 30 / 365.0, R, "c")
        assert outcome == "solved"
        assert contract.implied_vol == pytest.approx(standalone_iv)

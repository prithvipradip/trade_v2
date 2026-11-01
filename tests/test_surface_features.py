"""Surface features (research only): 25-delta risk reversal / butterfly, term slope, forward variance.

Synthetic smiles with a KNOWN shape, so every claim is checkable against the generating function. CI never needs the
private research DB. The study-level tests (labels, folds, bootstrap, canaries) live in test_surface_feature_study.py.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from ait.backtesting.iv_surface import IVSurface, SmileFit
from ait.backtesting.pricing import OptionType, bs_delta
from ait.backtesting.surface_features import (
    bf25, forward_variance, in_range, rr25_norm, smile_atm_iv, smile_is_usable, strike_for_delta, term_slope, years)

AS_OF = date(2025, 5, 2)
SPOT = 500.0
T30 = 30 / 365.0
D25 = 0.25                                  # the study's risk-reversal delta (PLAN.md): passed explicitly, no default in the module


def smile(a=0.20, b=0.0, c=0.0, *, m_lo=-0.12, m_hi=0.12, expiry=None, as_of=AS_OF, n=40, rmse=0.002, r=0.04):
    return SmileFit(as_of=as_of, expiry=expiry or as_of + timedelta(days=30), a=a, b=b, c=c, n=n,
                    m_lo=m_lo, m_hi=m_hi, rmse=rmse, r=r)


# --------------------------------------------------------------------------- strike for delta

@pytest.mark.parametrize("right", ["P", "C"])
@pytest.mark.parametrize("target", [0.10, 0.25, 0.40])
def test_strike_for_delta_round_trips(right, target):
    s = smile(b=-0.30, c=1.5)
    got = strike_for_delta(s, SPOT, T30, target, right)
    assert got is not None
    strike, iv = got
    opt = OptionType.PUT if right == "P" else OptionType.CALL
    assert abs(bs_delta(SPOT, strike, T30, s.r, iv, opt)) == pytest.approx(target, abs=1e-6)
    assert iv == pytest.approx(s.iv_at(math.log(strike / SPOT), clamp_margin=0.0))


def test_put_strike_is_below_spot_and_call_strike_above():
    s = smile()
    assert strike_for_delta(s, SPOT, T30, 0.25, "P")[0] < SPOT < strike_for_delta(s, SPOT, T30, 0.25, "C")[0]


def test_strike_for_delta_is_none_outside_the_fitted_range():
    # 25-delta at 20% IV / 30 days sits near m = -0.04; a smile fitted only to -0.01 cannot place it
    s = smile(m_lo=-0.01, m_hi=0.01)
    assert strike_for_delta(s, SPOT, T30, 0.25, "P") is None
    assert strike_for_delta(s, SPOT, T30, 0.25, "C") is None


def test_strike_for_delta_is_none_without_data_on_that_side():
    s = smile(m_lo=0.01, m_hi=0.12)            # calls only
    assert strike_for_delta(s, SPOT, T30, 0.25, "P") is None
    assert strike_for_delta(s, SPOT, T30, 0.25, "C") is not None


def test_strike_for_delta_rejects_bad_inputs():
    s = smile()
    assert strike_for_delta(s, 0.0, T30, 0.25, "P") is None
    assert strike_for_delta(s, SPOT, 0.0, 0.25, "P") is None
    assert strike_for_delta(s, SPOT, T30, 0.0, "P") is None
    assert strike_for_delta(s, SPOT, T30, 0.5, "P") is None


def test_strike_for_delta_is_none_when_the_smile_goes_non_positive():
    s = smile(a=0.05, b=+1.5, c=0.0)           # IV = 0.05 + 1.5 m is negative from m = -0.033 down: the put edge is unusable
    assert strike_for_delta(s, SPOT, T30, 0.25, "P") is None


# --------------------------------------------------------------------------- risk reversal / butterfly

def test_flat_smile_has_zero_risk_reversal_and_butterfly_up_to_forward_drift():
    s = smile(a=0.20)
    # equal-|delta| put and call strikes are not symmetric in moneyness (drift), but a flat smile gives the same IV
    assert rr25_norm(s, SPOT, T30, D25) == pytest.approx(0.0, abs=1e-12)
    assert bf25(s, SPOT, T30, D25) == pytest.approx(0.0, abs=1e-12)


def test_put_skew_gives_positive_risk_reversal():
    assert rr25_norm(smile(b=-0.30), SPOT, T30, D25) > 0
    assert rr25_norm(smile(b=+0.30), SPOT, T30, D25) < 0


def test_risk_reversal_matches_the_generating_function():
    s = smile(a=0.20, b=-0.30, c=1.5)
    mp = math.log(strike_for_delta(s, SPOT, T30, 0.25, "P")[0] / SPOT)
    mc = math.log(strike_for_delta(s, SPOT, T30, 0.25, "C")[0] / SPOT)
    want = ((s.a + s.b * mp + s.c * mp * mp) - (s.a + s.b * mc + s.c * mc * mc)) / s.a
    assert rr25_norm(s, SPOT, T30, D25) == pytest.approx(want, rel=1e-9)


def test_risk_reversal_is_comparable_across_vol_levels_where_the_raw_slope_is_not():
    # The same smile at half the vol: IV and the moneyness axis both scale by 0.5 (a -> 0.5a, b -> b, c -> 2c), so the
    # 25-delta strikes sit at half the log-moneyness. The delta-based number is unchanged (up to rate drift) ...
    high = smile(a=0.20, b=-0.30, c=1.5, m_lo=-0.12, m_hi=0.12)
    low = smile(a=0.10, b=-0.30, c=3.0, m_lo=-0.06, m_hi=0.06)
    assert rr25_norm(low, SPOT, T30, D25) == pytest.approx(rr25_norm(high, SPOT, T30, D25), rel=0.05)
    # ... whereas the stored slope b alone says the two regimes differ, or agree, for the wrong reason: doubling a and
    # b together (a different shape in delta space) moves the normalised number a lot. This is why the study reads the
    # skew at a fixed delta and not from b.
    twice = smile(a=0.20, b=-0.60, c=1.5)
    assert rr25_norm(twice, SPOT, T30, D25) > 1.5 * rr25_norm(high, SPOT, T30, D25)


def test_convexity_gives_positive_butterfly():
    assert bf25(smile(c=2.0), SPOT, T30, D25) > 0
    assert bf25(smile(c=0.0), SPOT, T30, D25) == pytest.approx(0.0, abs=1e-12)


def test_features_are_nan_when_a_wing_is_outside_the_fitted_range():
    narrow = smile(b=-0.30, c=1.5, m_lo=-0.02, m_hi=0.02)
    assert math.isnan(rr25_norm(narrow, SPOT, T30, D25))
    assert math.isnan(bf25(narrow, SPOT, T30, D25))


def test_features_are_nan_when_the_money_is_not_inside_the_fitted_range():
    otm_puts_only = smile(b=-0.30, m_lo=-0.12, m_hi=-0.01)
    assert math.isnan(smile_atm_iv(otm_puts_only))
    assert math.isnan(rr25_norm(otm_puts_only, SPOT, T30, D25))


def test_clamp_margin_is_never_used_to_stretch_the_range():
    # a smile whose range ends just short of the 25-delta call strike: the surface's own clamp_margin (0.01) would
    # let iv_at reach it; the feature must still say NaN
    s = smile(b=-0.30, m_lo=-0.12, m_hi=0.03)
    got = strike_for_delta(s, SPOT, T30, 0.25, "C")
    if got is not None:                         # 25-delta call is inside +0.03 for this IV/maturity; tighten until it is not
        s = smile(b=-0.30, m_lo=-0.12, m_hi=math.log(got[0] / SPOT) - 1e-4)
    assert math.isnan(rr25_norm(s, SPOT, T30, D25))


# --------------------------------------------------------------------------- quality gate

def test_smile_is_usable_checks_points_and_fit_error():
    assert smile_is_usable(smile(n=40, rmse=0.01), min_points=10, max_rmse=0.03)
    assert not smile_is_usable(smile(n=5), min_points=10, max_rmse=0.03)
    assert not smile_is_usable(smile(rmse=0.05), min_points=10, max_rmse=0.03)
    assert not smile_is_usable(None, min_points=10, max_rmse=0.03)


def test_in_range_is_inclusive():
    s = smile(m_lo=-0.1, m_hi=0.1)
    assert in_range(s, -0.1) and in_range(s, 0.1) and not in_range(s, 0.1001)


# --------------------------------------------------------------------------- term slope

def _surface(*fits):
    return IVSurface(list(fits))


def test_term_slope_is_short_minus_long_atm():
    e7, e30 = AS_OF + timedelta(days=7), AS_OF + timedelta(days=30)
    surf = _surface(smile(a=0.25, expiry=e7), smile(a=0.20, expiry=e30))
    assert term_slope(surf, AS_OF, SPOT, 7, 30) == pytest.approx(0.05)               # backwardation
    flipped = _surface(smile(a=0.18, expiry=e7), smile(a=0.22, expiry=e30))
    assert term_slope(flipped, AS_OF, SPOT, 7, 30) == pytest.approx(-0.04)           # contango


def test_term_slope_interpolates_in_total_variance_between_listed_expiries():
    e5, e10, e40 = (AS_OF + timedelta(days=d) for d in (5, 10, 40))
    surf = _surface(smile(a=0.20, expiry=e5), smile(a=0.22, expiry=e10), smile(a=0.24, expiry=e40))
    iv7 = surf.iv(AS_OF, AS_OF + timedelta(days=7), SPOT, SPOT)
    assert 0.20 < iv7 < 0.22
    assert term_slope(surf, AS_OF, SPOT, 7, 30) == pytest.approx(iv7 - surf.iv(AS_OF, AS_OF + timedelta(days=30), SPOT, SPOT))


def test_term_slope_is_nan_without_any_surface():
    assert math.isnan(term_slope(_surface(), AS_OF, SPOT, 7, 30))


def test_term_slope_ignores_smiles_dated_after_as_of():
    """The look-ahead guard: adding or changing smiles dated AFTER as_of cannot move a feature computed at as_of."""
    e7, e30 = AS_OF + timedelta(days=7), AS_OF + timedelta(days=30)
    known = [smile(a=0.25, expiry=e7), smile(a=0.20, expiry=e30)]
    later = AS_OF + timedelta(days=1)
    future_a = [smile(a=0.60, expiry=e7, as_of=later), smile(a=0.05, expiry=e30, as_of=later)]
    future_b = [smile(a=0.01, expiry=e7, as_of=later), smile(a=0.90, expiry=e30, as_of=later)]
    base = term_slope(_surface(*known), AS_OF, SPOT, 7, 30)
    assert term_slope(_surface(*known, *future_a), AS_OF, SPOT, 7, 30) == base
    assert term_slope(_surface(*known, *future_b), AS_OF, SPOT, 7, 30) == base


# --------------------------------------------------------------------------- forward variance

def test_forward_variance_of_a_flat_term_structure_is_that_variance():
    near = smile(a=0.20, expiry=AS_OF + timedelta(days=7))
    far = smile(a=0.20, expiry=AS_OF + timedelta(days=30))
    assert forward_variance(near, far) == pytest.approx(0.20 ** 2)


def test_forward_variance_matches_the_formula():
    near = smile(a=0.30, expiry=AS_OF + timedelta(days=7))
    far = smile(a=0.22, expiry=AS_OF + timedelta(days=30))
    t1, t2 = 7 / 365, 30 / 365
    assert forward_variance(near, far) == pytest.approx((0.22 ** 2 * t2 - 0.30 ** 2 * t1) / (t2 - t1))
    assert years(AS_OF, near.expiry) == pytest.approx(t1)


def test_forward_variance_is_nan_for_calendar_arbitrage_and_bad_inputs():
    near = smile(a=0.60, expiry=AS_OF + timedelta(days=7))
    far = smile(a=0.10, expiry=AS_OF + timedelta(days=30))                   # total variance falls: negative forward
    assert math.isnan(forward_variance(near, far))
    assert math.isnan(forward_variance(far, near))                           # not ordered
    other_day = smile(a=0.2, expiry=AS_OF + timedelta(days=30), as_of=AS_OF + timedelta(days=1))
    assert math.isnan(forward_variance(near, other_day))                     # different as-of dates
    otm_only = smile(m_lo=-0.12, m_hi=-0.01, expiry=AS_OF + timedelta(days=30))
    assert math.isnan(forward_variance(smile(expiry=AS_OF + timedelta(days=7)), otm_only))

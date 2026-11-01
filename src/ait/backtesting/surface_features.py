"""Features read off the fitted IV smile (research only): ATM level, 25-delta risk reversal and butterfly, term slope,
forward variance.

Why. The surface (``iv_surface.IVSurface``) stores one quadratic smile per (date, expiry): ``iv(m) = a + b*m + c*m^2``
in log-moneyness ``m = ln(K/S)``. The numbers a trader reads off a smile (skew as a 25-delta risk reversal, wing
richness as a 25-delta butterfly, the slope between two maturities) are not stored; this module derives them, once,
so a study and any later gate or model feature share one definition.

Rules every function here follows:
  * a quadratic is only trusted INSIDE the strikes it was fitted on. A delta strike that falls outside
    ``[m_lo, m_hi]`` gives NaN, never a clamped or extrapolated number (the surface's own ``clamp_margin`` is
    deliberately NOT used: it exists so a pricing lookup near the edge does not fail, not so a feature can be read
    past the data);
  * no function reads a date later than the ``as_of`` it is given (the surface lookup already enforces this for
    ``IVSurface.iv``; the smile-level functions take the smile as an argument and cannot reach other dates);
  * time is calendar days / 365, the same convention as ``IVSurface.iv``.

Pure functions, no IO.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from ait.backtesting.iv_surface import IVSurface, SmileFit
from ait.backtesting.pricing import OptionType, bs_delta

NAN = float("nan")
_BISECT_ITERATIONS = 60


def years(as_of: date, expiry: date) -> float:
    """Calendar-day time to expiry in years, floored at one day (as ``IVSurface.iv`` does)."""
    return max((expiry - as_of).days, 1) / 365.0


def in_range(smile: SmileFit, m: float) -> bool:
    return smile.m_lo <= m <= smile.m_hi


def smile_atm_iv(smile: SmileFit) -> float:
    """ATM IV of one smile, or NaN when the fitted strikes do not straddle the money (then ATM would be an
    extrapolation)."""
    if not in_range(smile, 0.0):
        return NAN
    iv = smile.iv_at(0.0, clamp_margin=0.0)
    return iv if iv > 0 else NAN


def smile_is_usable(smile: SmileFit | None, *, min_points: int, max_rmse: float) -> bool:
    """A smile supports features only if it has enough points and fits its own points well enough."""
    return smile is not None and smile.n >= min_points and smile.rmse <= max_rmse


def strike_for_delta(smile: SmileFit, spot: float, t: float, abs_delta: float, right: str) -> tuple[float, float] | None:
    """(strike, smile IV at that strike) whose Black-Scholes |delta|, priced with the smile's own IV, equals
    ``abs_delta``; ``None`` when no such strike lies inside the fitted range (or the smile's IV is not positive there).

    ``right`` is "P" or "C". Bisection on log-moneyness over the side of the money the right trades OTM on.
    """
    if spot <= 0 or t <= 0 or not 0.0 < abs_delta < 0.5:
        return None
    opt = OptionType.PUT if right == "P" else OptionType.CALL
    lo, hi = (smile.m_lo, min(0.0, smile.m_hi)) if right == "P" else (max(0.0, smile.m_lo), smile.m_hi)
    if not lo < hi:
        return None

    def gap(m: float) -> float | None:
        iv = smile.iv_at(m, clamp_margin=0.0)
        if iv <= 0:
            return None
        return abs(bs_delta(spot, spot * math.exp(m), t, smile.r, iv, opt)) - abs_delta

    g_lo, g_hi = gap(lo), gap(hi)
    if g_lo is None or g_hi is None or g_lo * g_hi > 0:
        return None
    for _ in range(_BISECT_ITERATIONS):
        mid = 0.5 * (lo + hi)
        g_mid = gap(mid)
        if g_mid is None:
            return None
        if g_lo * g_mid <= 0:
            hi = mid
        else:
            lo, g_lo = mid, g_mid
    m = 0.5 * (lo + hi)
    iv = smile.iv_at(m, clamp_margin=0.0)
    return (spot * math.exp(m), iv) if iv > 0 else None


def _wing_ivs(smile: SmileFit, spot: float, t: float, abs_delta: float) -> tuple[float, float] | None:
    put = strike_for_delta(smile, spot, t, abs_delta, "P")
    call = strike_for_delta(smile, spot, t, abs_delta, "C")
    if put is None or call is None:
        return None
    return put[1], call[1]


def rr25_norm(smile: SmileFit, spot: float, t: float, abs_delta: float) -> float:
    """(IV of the ``abs_delta`` (25-delta in the study) put - IV of the same-delta call) / ATM IV. Positive = put skew. NaN when either wing strike
    or the ATM point is outside the fitted range."""
    atm = smile_atm_iv(smile)
    wings = _wing_ivs(smile, spot, t, abs_delta)
    if math.isnan(atm) or wings is None:
        return NAN
    return (wings[0] - wings[1]) / atm


def bf25(smile: SmileFit, spot: float, t: float, abs_delta: float) -> float:
    """0.5 * (IV ``abs_delta`` put + IV ``abs_delta`` call) - ATM IV: how much richer the wings are than the money. NaN as in
    ``rr25_norm``."""
    atm = smile_atm_iv(smile)
    wings = _wing_ivs(smile, spot, t, abs_delta)
    if math.isnan(atm) or wings is None:
        return NAN
    return 0.5 * (wings[0] + wings[1]) - atm


def term_slope(surface: IVSurface, as_of: date, spot: float, short_days: int, long_days: int) -> float:
    """ATM IV at ``as_of + short_days`` minus ATM IV at ``as_of + long_days`` from the smiles known at ``as_of``
    (``IVSurface.iv`` interpolates in total variance between listed expiries). Positive = backwardation. NaN when
    either maturity cannot be priced."""
    short = surface.iv(as_of, as_of + timedelta(days=short_days), spot, spot)
    long_ = surface.iv(as_of, as_of + timedelta(days=long_days), spot, spot)
    if short is None or long_ is None:
        return NAN
    return short - long_


def forward_variance(near: SmileFit, far: SmileFit) -> float:
    """Annualised forward variance between two expiries of the SAME as-of date: (s2^2 T2 - s1^2 T1) / (T2 - T1) from
    their ATM IVs. NaN when the expiries are not ordered, an ATM point is not inside its range, or the result is
    negative (calendar arbitrage in the fit, which is a data problem, not a signal)."""
    if near.as_of != far.as_of or far.expiry <= near.expiry:
        return NAN
    s1, s2 = smile_atm_iv(near), smile_atm_iv(far)
    if math.isnan(s1) or math.isnan(s2):
        return NAN
    t1, t2 = years(near.as_of, near.expiry), years(far.as_of, far.expiry)
    fv = (s2 * s2 * t2 - s1 * s1 * t1) / (t2 - t1)
    return fv if fv >= 0 else NAN

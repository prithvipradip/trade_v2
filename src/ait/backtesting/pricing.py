"""Black-Scholes pricing primitives for the backtest engine.

Extracted from the retired options_sim.py (R12 Tier-C, 2026-07-13): the
engine only ever consumed these stateless functions. The simulator class
half (OptionsSimulator / SimulatedOption / OptionPosition / SpreadPosition,
including the loaded 5x-premium short-side max-loss trap) lives in
deprecated/src/options_sim.py and must not be resurrected without review.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.stats import norm

from ait.config.settings import DEFAULT_RISK_FREE_RATE  # W88b: one schema default for r


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


def black_scholes_price(
    S: float, K: float, t: float, r: float, sigma: float,
    option_type: OptionType,
) -> float:
    """Black-Scholes option price.

    Args:
        S: Underlying price
        K: Strike price
        t: Time to expiry in years
        r: Risk-free rate
        sigma: Implied volatility (annualized)
        option_type: CALL or PUT
    """
    if t <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # At expiry or invalid inputs: intrinsic value only
        if option_type == OptionType.CALL:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option_type == OptionType.CALL:
        price = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return max(price, 0.0)


def bs_delta(
    S: float, K: float, t: float, r: float, sigma: float,
    option_type: OptionType,
) -> float:
    """Black-Scholes delta."""
    if t <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    if option_type == OptionType.CALL:
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def find_strike_by_delta(
    S: float, t: float, sigma: float, target_delta: float,
    option_type: OptionType, r: float = DEFAULT_RISK_FREE_RATE,
) -> float:
    """Find the strike price for a target delta using bisection.

    Args:
        S: Underlying price
        t: Time to expiry in years
        sigma: Implied volatility
        target_delta: Target delta (e.g., 0.30 for OTM call, -0.30 for OTM put)
        option_type: CALL or PUT
    """
    # Search range: 20% below to 20% above underlying
    lo, hi = S * 0.80, S * 1.20

    for _ in range(50):  # Bisection iterations
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, t, r, sigma, option_type)

        if option_type == OptionType.CALL:
            if d > target_delta:
                lo = mid  # Strike too low, move up (further OTM)
            else:
                hi = mid
        else:
            if d < target_delta:  # target_delta is negative for puts
                hi = mid  # Strike too high, move down (further OTM)
            else:
                lo = mid

    # Round to nearest dollar
    return round((lo + hi) / 2, 0)


def skewed_leg_iv(
    base_iv: float, strike: float, underlying: float, option_type: OptionType,
    skew_factor: float, skew_slope_per_pct_otm: float, iv_floor: float,
    put_skew_coefficient: float, call_skew_coefficient: float,
) -> float:
    """Per-leg IV with the engine's linear-in-log-moneyness skew.

    W72: the ONE implementation. ``Backtester._get_leg_iv`` and the parity
    replay (``ait.backtesting.replay``) both call this, so the replay can never
    price a leg differently from the engine. Legacy (pre-W5) put term
    ``skew_factor * otm * put_skew_coefficient`` plus the W5 multiplicative put steepening
    ``base_iv * skew_slope_per_pct_otm * (otm * 100)``; calls keep the mild
    ``skew_factor * max(0, ln(K/S)) * call_skew_coefficient``. ``skew_slope_per_pct_otm=0``
    reproduces the pre-W5 output exactly (see the engine docstring).

    W90c: the two coefficients (literals 0.10 / 0.02 until now) are ``backtest.engine_tuning`` settings and are
    REQUIRED arguments, so no caller can silently price with a stale copy of them.
    """
    log_m = np.log(strike / underlying)
    if option_type == OptionType.PUT:
        otm = max(0.0, -log_m)
        skew_adj = skew_factor * otm * put_skew_coefficient
        # otm * 100 = distance in PERCENTAGE POINTS OTM.
        skew_adj += base_iv * skew_slope_per_pct_otm * (otm * 100.0)
    else:
        skew_adj = skew_factor * max(0.0, log_m) * call_skew_coefficient
    return max(base_iv + skew_adj, iv_floor)


def ic_leg_marks(
    S: float, short_call: float, short_put: float, long_call: float, long_put: float,
    t: float, r: float, base_iv: float,
    skew_factor: float, skew_slope_per_pct_otm: float, iv_floor: float,
    put_skew_coefficient: float, call_skew_coefficient: float,
) -> tuple[float, float, float, float]:
    """The four iron-condor leg mids ``(short call, short put, long call, long put)``.

    W72: single owner of condor leg pricing for the engine's entry build, its
    daily re-mark (``_reprice_ic_legs``) and the parity replay. Each leg is
    priced with its own skewed IV (``skewed_leg_iv``) via ``black_scholes_price``;
    the call order below is the historical one and is part of the contract.
    """
    def _p(strike: float, opt: OptionType) -> float:
        return black_scholes_price(
            S, strike, t, r,
            skewed_leg_iv(base_iv, strike, S, opt, skew_factor,
                          skew_slope_per_pct_otm, iv_floor,
                          put_skew_coefficient, call_skew_coefficient), opt)

    sc = _p(short_call, OptionType.CALL)
    sp = _p(short_put, OptionType.PUT)
    lc = _p(long_call, OptionType.CALL)
    lp = _p(long_put, OptionType.PUT)
    return sc, sp, lc, lp


def expected_move(S: float, iv: float, t_years: float) -> float:
    """One-standard-deviation dollar move over ``t_years``: ``S * iv * sqrt(t)``.

    W74: takes time in YEARS (fractional), so a 0-DTE position with hours left
    has a small positive move instead of the whole-day ``max(dte, 0)`` form's
    hard zero, which made the condor builder reject every 0-DTE entry with
    ``em_unavailable``. Returns 0.0 for non-positive spot, IV or time.
    """
    if S <= 0.0 or iv <= 0.0 or t_years <= 0.0:
        return 0.0
    return float(S * iv * np.sqrt(t_years))


def ic_leg_marks_from_ivs(
    S: float, short_call: float, short_put: float, long_call: float, long_put: float,
    t: float, r: float, iv_short_call: float, iv_short_put: float,
    iv_long_call: float, iv_long_put: float,
) -> tuple[float, float, float, float]:
    """Condor leg mids ``(short call, short put, long call, long put)`` at EXPLICIT per-leg IVs.

    W75: the surface pricer supplies each leg's IV from the real-bar smile of the
    leg's own expiry instead of a base IV plus a DTE-free skew. Same Black-Scholes,
    same leg order as ``ic_leg_marks`` (the contract callers rely on).
    """
    return (
        black_scholes_price(S, short_call, t, r, iv_short_call, OptionType.CALL),
        black_scholes_price(S, short_put, t, r, iv_short_put, OptionType.PUT),
        black_scholes_price(S, long_call, t, r, iv_long_call, OptionType.CALL),
        black_scholes_price(S, long_put, t, r, iv_long_put, OptionType.PUT),
    )


def realized_vol(close_prices: np.ndarray, window: int = 20) -> float:
    """Calculate annualized realized volatility from close prices."""
    if len(close_prices) < window + 1:
        return 0.25  # Default 25%
    log_returns = np.diff(np.log(close_prices[-window - 1:]))
    return float(np.std(log_returns) * np.sqrt(252))

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
    option_type: OptionType, r: float = 0.05,
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


def realized_vol(close_prices: np.ndarray, window: int = 20) -> float:
    """Calculate annualized realized volatility from close prices."""
    if len(close_prices) < window + 1:
        return 0.25  # Default 25%
    log_returns = np.diff(np.log(close_prices[-window - 1:]))
    return float(np.std(log_returns) * np.sqrt(252))

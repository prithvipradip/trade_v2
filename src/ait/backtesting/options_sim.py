"""Realistic options price simulation for backtesting.

Uses Black-Scholes pricing with historical underlying prices and
realized volatility to generate accurate option premiums, Greeks,
and P&L instead of crude price-ratio proxies.

Key improvements over the 2% proxy:
- Real delta-based strike selection (0.30 delta OTM)
- Time decay (theta) properly modeled
- IV changes affect option prices (vega)
- Spreads priced as net debit/credit with proper max loss
- Bid-ask spread simulation (tighter for liquid, wider for illiquid)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.stats import norm


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class SimulatedOption:
    """A simulated option contract for backtesting."""

    strike: float
    option_type: OptionType
    entry_price: float
    entry_delta: float
    entry_iv: float
    dte_at_entry: int
    underlying_at_entry: float

    def price_at(
        self,
        underlying: float,
        days_held: int,
        iv_change: float = 0.0,
    ) -> float:
        """Calculate option price at a future point using Black-Scholes.

        Args:
            underlying: Current underlying price
            days_held: Days since entry
            iv_change: Change in IV since entry (e.g., +0.05 = 5% IV increase)
        """
        dte_remaining = max(self.dte_at_entry - days_held, 0)
        t = max(dte_remaining / 365.0, 0.0001)
        sigma = max(self.entry_iv + iv_change, 0.05)
        r = 0.05

        price = black_scholes_price(
            S=underlying,
            K=self.strike,
            t=t,
            r=r,
            sigma=sigma,
            option_type=self.option_type,
        )
        return max(price, 0.0)


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
    if t <= 0:
        # At expiry: intrinsic value only
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


def simulate_bid_ask(mid_price: float, iv: float) -> tuple[float, float]:
    """Simulate realistic bid-ask spread.

    Higher IV = wider spreads. Cheaper options = wider relative spreads.
    """
    # Base spread: ~2% for liquid ATM, wider for OTM/cheap options
    base_spread_pct = 0.02 + 0.01 * max(iv - 0.20, 0)  # Wider in high vol
    min_spread = 0.05  # Minimum $0.05 spread

    half_spread = max(mid_price * base_spread_pct / 2, min_spread / 2)
    bid = max(mid_price - half_spread, 0.01)
    ask = mid_price + half_spread

    return bid, ask


@dataclass
class OptionPosition:
    """Tracks a simulated option position through time."""

    option: SimulatedOption
    contracts: int
    direction: str  # "long" or "short"
    entry_cost: float  # Total cost to enter (per contract * 100 * contracts)
    strategy_name: str
    symbol: str

    @property
    def max_loss(self) -> float:
        """Maximum possible loss for this position."""
        if self.direction == "long":
            return self.entry_cost  # Can only lose what you paid
        # Short options: theoretically unlimited, but cap at 5x premium
        return self.entry_cost * 5

    def current_value(
        self, underlying: float, days_held: int, iv_change: float = 0.0,
    ) -> float:
        """Current market value of the position."""
        price = self.option.price_at(underlying, days_held, iv_change)
        value = price * 100 * self.contracts
        if self.direction == "short":
            # Short position: we collected premium, now owe the current value
            return self.entry_cost - value  # Positive if price dropped (good for shorts)
        return value

    def pnl(
        self, underlying: float, days_held: int, iv_change: float = 0.0,
    ) -> float:
        """P&L of the position."""
        current = self.current_value(underlying, days_held, iv_change)
        if self.direction == "long":
            return current - self.entry_cost
        return current  # For shorts, current_value already accounts for P&L


@dataclass
class SpreadPosition:
    """A multi-leg option spread position."""

    legs: list[OptionPosition]
    strategy_name: str
    symbol: str
    net_debit: float  # Positive = debit spread, negative = credit spread
    max_loss: float
    max_profit: float

    def pnl(
        self, underlying: float, days_held: int, iv_change: float = 0.0,
    ) -> float:
        """Combined P&L of all legs."""
        return sum(leg.pnl(underlying, days_held, iv_change) for leg in self.legs)


class OptionsSimulator:
    """Simulates option trades with realistic Black-Scholes pricing.

    Used by the backtesting engine to replace the crude price proxy
    with proper options pricing including Greeks and time decay.
    """

    def __init__(self, target_dte: int = 30, risk_free_rate: float = 0.05):
        self._target_dte = target_dte
        self._r = risk_free_rate

    def price_single_option(
        self,
        underlying: float,
        iv: float,
        option_type: OptionType,
        target_delta: float = 0.30,
        dte: int | None = None,
    ) -> SimulatedOption:
        """Create a simulated option at a target delta.

        Args:
            underlying: Current underlying price
            iv: Implied volatility (annualized)
            option_type: CALL or PUT
            target_delta: Target delta for strike selection
            dte: Days to expiry (defaults to self._target_dte)
        """
        dte = dte or self._target_dte
        t = dte / 365.0

        # Find strike for target delta
        if option_type == OptionType.PUT:
            target_delta = -abs(target_delta)
        else:
            target_delta = abs(target_delta)

        strike = find_strike_by_delta(underlying, t, iv, target_delta, option_type, self._r)

        # Calculate entry price
        entry_price = black_scholes_price(underlying, strike, t, self._r, iv, option_type)
        entry_delta = bs_delta(underlying, strike, t, self._r, iv, option_type)

        return SimulatedOption(
            strike=strike,
            option_type=option_type,
            entry_price=entry_price,
            entry_delta=entry_delta,
            entry_iv=iv,
            dte_at_entry=dte,
            underlying_at_entry=underlying,
        )

    def create_long_call(
        self, underlying: float, iv: float, capital: float,
        delta: float = 0.30, dte: int | None = None,
    ) -> OptionPosition | None:
        """Create a long call position."""
        opt = self.price_single_option(underlying, iv, OptionType.CALL, delta, dte)
        cost_per_contract = opt.entry_price * 100
        if cost_per_contract <= 0 or capital < cost_per_contract:
            return None

        contracts = max(1, int(capital * 0.05 / cost_per_contract))  # 5% position sizing
        return OptionPosition(
            option=opt, contracts=contracts, direction="long",
            entry_cost=cost_per_contract * contracts,
            strategy_name="long_call", symbol="",
        )

    def create_long_put(
        self, underlying: float, iv: float, capital: float,
        delta: float = 0.30, dte: int | None = None,
    ) -> OptionPosition | None:
        """Create a long put position."""
        opt = self.price_single_option(underlying, iv, OptionType.PUT, delta, dte)
        cost_per_contract = opt.entry_price * 100
        if cost_per_contract <= 0 or capital < cost_per_contract:
            return None

        contracts = max(1, int(capital * 0.05 / cost_per_contract))
        return OptionPosition(
            option=opt, contracts=contracts, direction="long",
            entry_cost=cost_per_contract * contracts,
            strategy_name="long_put", symbol="",
        )

    def create_bull_call_spread(
        self, underlying: float, iv: float, capital: float,
        long_delta: float = 0.40, short_delta: float = 0.20,
        dte: int | None = None,
    ) -> SpreadPosition | None:
        """Create a bull call spread (debit spread).

        Buy lower strike call (higher delta), sell higher strike call (lower delta).
        Max loss = net debit. Max profit = strike width - net debit.
        """
        dte = dte or self._target_dte
        long_opt = self.price_single_option(underlying, iv, OptionType.CALL, long_delta, dte)
        short_opt = self.price_single_option(underlying, iv, OptionType.CALL, short_delta, dte)

        net_debit = (long_opt.entry_price - short_opt.entry_price) * 100
        if net_debit <= 0 or capital < net_debit:
            return None

        strike_width = abs(short_opt.strike - long_opt.strike) * 100
        max_profit = strike_width - net_debit

        long_pos = OptionPosition(long_opt, 1, "long", long_opt.entry_price * 100, "bull_call_spread", "")
        short_pos = OptionPosition(short_opt, 1, "short", short_opt.entry_price * 100, "bull_call_spread", "")

        return SpreadPosition(
            legs=[long_pos, short_pos],
            strategy_name="bull_call_spread", symbol="",
            net_debit=net_debit, max_loss=net_debit, max_profit=max_profit,
        )

    def create_bear_put_spread(
        self, underlying: float, iv: float, capital: float,
        long_delta: float = 0.40, short_delta: float = 0.20,
        dte: int | None = None,
    ) -> SpreadPosition | None:
        """Create a bear put spread (debit spread).

        Buy higher strike put (higher delta), sell lower strike put (lower delta).
        """
        dte = dte or self._target_dte
        long_opt = self.price_single_option(underlying, iv, OptionType.PUT, long_delta, dte)
        short_opt = self.price_single_option(underlying, iv, OptionType.PUT, short_delta, dte)

        net_debit = (long_opt.entry_price - short_opt.entry_price) * 100
        if net_debit <= 0 or capital < net_debit:
            return None

        strike_width = abs(long_opt.strike - short_opt.strike) * 100
        max_profit = strike_width - net_debit

        long_pos = OptionPosition(long_opt, 1, "long", long_opt.entry_price * 100, "bear_put_spread", "")
        short_pos = OptionPosition(short_opt, 1, "short", short_opt.entry_price * 100, "bear_put_spread", "")

        return SpreadPosition(
            legs=[long_pos, short_pos],
            strategy_name="bear_put_spread", symbol="",
            net_debit=net_debit, max_loss=net_debit, max_profit=max_profit,
        )

    def create_iron_condor(
        self, underlying: float, iv: float, capital: float,
        call_delta: float = 0.20, put_delta: float = 0.20,
        wing_width_pct: float = 0.03, dte: int | None = None,
    ) -> SpreadPosition | None:
        """Create an iron condor (credit spread).

        Sell OTM call + sell OTM put, buy further OTM wings for protection.
        Max profit = net credit. Max loss = wing width - net credit.
        """
        dte = dte or self._target_dte
        t = dte / 365.0

        # Short legs (closer to ATM)
        short_call = self.price_single_option(underlying, iv, OptionType.CALL, call_delta, dte)
        short_put = self.price_single_option(underlying, iv, OptionType.PUT, put_delta, dte)

        # Long legs (wings, further OTM)
        wing_width = underlying * wing_width_pct
        long_call_strike = short_call.strike + wing_width
        long_put_strike = short_put.strike - wing_width

        long_call_price = black_scholes_price(underlying, long_call_strike, t, self._r, iv, OptionType.CALL)
        long_put_price = black_scholes_price(underlying, long_put_strike, t, self._r, iv, OptionType.PUT)

        # Net credit = premiums received - premiums paid
        net_credit = (short_call.entry_price + short_put.entry_price - long_call_price - long_put_price) * 100

        if net_credit <= 0:
            return None

        max_loss = wing_width * 100 - net_credit
        if capital < max_loss:
            return None

        long_call_opt = SimulatedOption(long_call_strike, OptionType.CALL, long_call_price,
                                         0.0, iv, dte, underlying)
        long_put_opt = SimulatedOption(long_put_strike, OptionType.PUT, long_put_price,
                                        0.0, iv, dte, underlying)

        legs = [
            OptionPosition(short_call, 1, "short", short_call.entry_price * 100, "iron_condor", ""),
            OptionPosition(short_put, 1, "short", short_put.entry_price * 100, "iron_condor", ""),
            OptionPosition(long_call_opt, 1, "long", long_call_price * 100, "iron_condor", ""),
            OptionPosition(long_put_opt, 1, "long", long_put_price * 100, "iron_condor", ""),
        ]

        return SpreadPosition(
            legs=legs, strategy_name="iron_condor", symbol="",
            net_debit=-net_credit, max_loss=max_loss, max_profit=net_credit,
        )

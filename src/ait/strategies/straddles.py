"""Volatility strategies: long straddles and short strangles.

Long straddle: buy ATM call + ATM put — profits from big moves in either direction.
Short strangle: sell OTM call + OTM put — profits from low volatility/range-bound.

Best in: Long straddle when IV is low and big move expected.
         Short strangle when IV is high and range-bound expected.
"""

from __future__ import annotations

import pandas as pd

from ait.data.options_chain import OptionsChain
from ait.strategies.base import Signal, SignalDirection, Strategy


class LongStraddle(Strategy):
    """Buy ATM call + ATM put — bet on a big move in either direction.

    Max loss: Total premium paid (both legs).
    Max profit: Unlimited (directionally).
    Best before: Earnings, FOMC, major catalysts.
    """

    @property
    def name(self) -> str:
        return "long_straddle"

    @property
    def direction_bias(self) -> SignalDirection | None:
        return None  # Direction agnostic

    def generate_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
    ) -> list[Signal]:
        # Straddles need low IV (cheap premium) with expected vol expansion
        if iv_rank > 40:
            return []  # Don't buy expensive straddles

        liquid_calls = self._filter_liquid(chain.calls)
        liquid_puts = self._filter_liquid(chain.puts)

        if not liquid_calls or not liquid_puts:
            return []

        # Find ATM strike
        atm_strike = chain.get_atm_strike()

        atm_call = self._find_strike_near(liquid_calls, atm_strike)
        atm_put = self._find_strike_near(liquid_puts, atm_strike)

        if not atm_call or not atm_put:
            return []

        total_premium = atm_call.mid + atm_put.mid
        if total_premium <= 0:
            return []

        # Breakevens: strike ± total premium
        # Need underlying to move more than total premium in either direction
        max_loss = total_premium * 100

        # Target: 50% gain on total premium
        take_profit = round(total_premium * 1.50, 2)
        stop_loss = round(total_premium * 0.50, 2)

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=confidence,
                contract=atm_call,
                action="BUY",
                quantity=1,
                legs=[
                    {"contract": atm_call, "action": "BUY", "ratio": 1},
                    {"contract": atm_put, "action": "BUY", "ratio": 1},
                ],
                entry_price=total_premium,
                max_loss=max_loss,
                max_profit=max_loss * 3,  # Target 3:1 on big moves
                stop_loss=stop_loss,
                take_profit=take_profit,
                iv_rank=iv_rank,
                underlying_price=chain.underlying_price,
                expiry=chain.expiry,
            )
        ]


class ShortStrangle(Strategy):
    """Sell OTM call + OTM put — bet on range-bound market.

    Max profit: Total premium collected.
    Max loss: Undefined (theoretically unlimited on call side).
    Best in: High IV, stable/range-bound markets.

    WARNING: Undefined risk — requires margin and careful sizing.
    """

    @property
    def name(self) -> str:
        return "short_strangle"

    @property
    def direction_bias(self) -> SignalDirection | None:
        return None

    def generate_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
    ) -> list[Signal]:
        # Only sell strangles in neutral markets with high IV
        if market_direction != SignalDirection.NEUTRAL:
            return []

        if iv_rank < 50:
            return []  # Need high IV to collect enough premium

        liquid_calls = self._filter_liquid(chain.calls)
        liquid_puts = self._filter_liquid(chain.puts)

        if not liquid_calls or not liquid_puts:
            return []

        # Sell far OTM (delta ~0.15) for higher probability
        short_call = self._find_strike_by_delta(liquid_calls, 0.15)
        short_put = self._find_strike_by_delta(liquid_puts, 0.15)

        if not short_call or not short_put:
            return []

        # Verify structure
        if short_put.strike >= short_call.strike:
            return []

        total_credit = short_call.mid + short_put.mid
        if total_credit <= 0:
            return []

        # Undefined risk — estimate max loss as 3x credit for risk management
        estimated_max_loss = total_credit * 3 * 100
        max_profit = total_credit * 100

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=confidence,
                contract=short_put,
                action="SELL",
                quantity=1,
                legs=[
                    {"contract": short_call, "action": "SELL", "ratio": 1},
                    {"contract": short_put, "action": "SELL", "ratio": 1},
                ],
                entry_price=total_credit,
                max_loss=estimated_max_loss,
                max_profit=max_profit,
                stop_loss=round(total_credit * 2.0, 2),  # Close at 2x loss
                take_profit=round(total_credit * 0.50, 2),  # Take profit at 50%
                iv_rank=iv_rank,
                underlying_price=chain.underlying_price,
                expiry=chain.expiry,
            )
        ]

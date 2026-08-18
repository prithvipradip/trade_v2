"""Covered strategies: covered calls and cash-secured puts.

Income-generating strategies that involve stock ownership or cash collateral.
Best in: Moderate IV, slightly directional or neutral markets.

Note: These require stock positions or sufficient cash.
Covered call = own 100 shares + sell 1 call.
Cash-secured put = have cash to buy 100 shares + sell 1 put.
"""

from __future__ import annotations

import pandas as pd

from ait.data.options_chain import OptionsChain
from ait.strategies.base import Signal, SignalDirection, Strategy


class CoveredCall(Strategy):
    """Sell covered calls against existing stock positions.

    Requires: 100 shares of the underlying per contract.
    Profit: Premium collected + stock gains up to strike.
    Risk: Stock drops (mitigated by premium collected).
    """

    @property
    def name(self) -> str:
        return "covered_call"

    @property
    def direction_bias(self) -> SignalDirection:
        return SignalDirection.BULLISH

    def generate_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
    ) -> list[Signal]:
        # Covered calls work in neutral-to-slightly-bullish markets
        if market_direction == SignalDirection.BEARISH:
            return []

        # Higher IV = more premium = better for selling
        if iv_rank < 30:
            return []

        liquid_calls = self._filter_liquid(chain.calls)
        if not liquid_calls:
            return []

        # Sell OTM call (delta ~0.25-0.30) — less likely to be called away
        target = self._find_strike_by_delta(liquid_calls, 0.30)
        if not target:
            return []

        premium = target.mid
        if premium <= 0:
            return []

        # Max profit: premium + (strike - current price) if stock rises to strike
        price = chain.underlying_price
        upside = max(0, target.strike - price)
        max_profit = (premium + upside) * 100

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.BULLISH,
                confidence=confidence,
                contract=target,
                action="SELL",
                quantity=1,
                entry_price=premium,
                max_loss=price * 100,  # Stock goes to 0 (extreme)
                max_profit=max_profit,
                iv_rank=iv_rank,
                underlying_price=price,
                expiry=chain.expiry,
            )
        ]


class CashSecuredPut(Strategy):
    """Sell cash-secured puts to collect premium or buy stock at a discount.

    Requires: Cash equal to strike × 100 per contract.
    Profit: Premium collected if stock stays above strike.
    Risk: Must buy stock at strike (mitigated: you wanted to own it).
    """

    @property
    def name(self) -> str:
        return "cash_secured_put"

    @property
    def direction_bias(self) -> SignalDirection:
        return SignalDirection.BULLISH

    def generate_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
    ) -> list[Signal]:
        if market_direction == SignalDirection.BEARISH:
            return []

        if iv_rank < 30:
            return []

        liquid_puts = self._filter_liquid(chain.puts)
        if not liquid_puts:
            return []

        # Sell OTM put (delta ~0.25) — strike below current price
        target = self._find_strike_by_delta(liquid_puts, 0.25)
        if not target:
            return []

        premium = target.mid
        if premium <= 0:
            return []

        # Max loss: strike price × 100 - premium (stock goes to 0)
        max_loss = (target.strike - premium) * 100

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.BULLISH,
                confidence=confidence,
                contract=target,
                action="SELL",
                quantity=1,
                entry_price=premium,
                max_loss=max_loss,
                max_profit=premium * 100,
                iv_rank=iv_rank,
                underlying_price=chain.underlying_price,
                expiry=chain.expiry,
            )
        ]

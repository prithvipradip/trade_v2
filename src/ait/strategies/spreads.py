"""Vertical spread strategies: bull call spreads and bear put spreads.

Defined-risk directional strategies that cost less than naked options.
Best in: Moderate directional conviction with defined risk tolerance.

Bull call spread = buy lower strike call + sell higher strike call (net debit)
Bear put spread = buy higher strike put + sell lower strike put (net debit)
"""

from __future__ import annotations

import pandas as pd

from ait.data.options_chain import OptionsChain, OptionContract
from ait.strategies.base import Signal, SignalDirection, Strategy


class BullCallSpread(Strategy):
    """Bull call spread — defined risk bullish play.

    Buy ATM/slightly-OTM call, sell further OTM call.
    Max loss = net debit paid.
    Max profit = width of spread - net debit.
    """

    @property
    def name(self) -> str:
        return "bull_call_spread"

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
        if market_direction != SignalDirection.BULLISH:
            return []

        liquid_calls = self._filter_liquid(chain.calls)
        if len(liquid_calls) < 2:
            return []

        # Long leg: slightly OTM (delta ~0.45)
        long_leg = self._find_strike_by_delta(liquid_calls, 0.45)
        if not long_leg:
            return []

        # Short leg: further OTM (delta ~0.25)
        # Must be higher strike than long leg
        candidates = [c for c in liquid_calls if c.strike > long_leg.strike]
        short_leg = self._find_strike_by_delta(candidates, 0.25)
        if not short_leg:
            return []

        # Calculate spread pricing
        net_debit = long_leg.mid - short_leg.mid
        if net_debit <= 0:
            return []  # Should be a debit spread

        spread_width = short_leg.strike - long_leg.strike
        max_profit = (spread_width - net_debit) * 100
        max_loss = net_debit * 100

        if max_profit <= 0:
            return []

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.BULLISH,
                confidence=confidence,
                contract=long_leg,
                action="BUY",
                quantity=1,
                legs=[
                    {"contract": long_leg, "action": "BUY", "ratio": 1},
                    {"contract": short_leg, "action": "SELL", "ratio": 1},
                ],
                entry_price=net_debit,
                max_loss=max_loss,
                max_profit=max_profit,
                stop_loss=round(net_debit * 0.50, 2),  # Exit at 50% loss of debit
                take_profit=round(net_debit + (spread_width - net_debit) * 0.75, 2),  # 75% of max
                iv_rank=iv_rank,
                underlying_price=chain.underlying_price,
                expiry=chain.expiry,
            )
        ]


class BearPutSpread(Strategy):
    """Bear put spread — defined risk bearish play.

    Buy ATM/slightly-OTM put, sell further OTM put.
    Max loss = net debit paid.
    Max profit = width of spread - net debit.
    """

    @property
    def name(self) -> str:
        return "bear_put_spread"

    @property
    def direction_bias(self) -> SignalDirection:
        return SignalDirection.BEARISH

    def generate_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
    ) -> list[Signal]:
        if market_direction != SignalDirection.BEARISH:
            return []

        liquid_puts = self._filter_liquid(chain.puts)
        if len(liquid_puts) < 2:
            return []

        # Long leg: slightly OTM put (delta ~0.45)
        long_leg = self._find_strike_by_delta(liquid_puts, 0.45)
        if not long_leg:
            return []

        # Short leg: further OTM put (delta ~0.25)
        # Must be lower strike than long leg
        candidates = [p for p in liquid_puts if p.strike < long_leg.strike]
        short_leg = self._find_strike_by_delta(candidates, 0.25)
        if not short_leg:
            return []

        net_debit = long_leg.mid - short_leg.mid
        if net_debit <= 0:
            return []

        spread_width = long_leg.strike - short_leg.strike
        max_profit = (spread_width - net_debit) * 100
        max_loss = net_debit * 100

        if max_profit <= 0:
            return []

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.BEARISH,
                confidence=confidence,
                contract=long_leg,
                action="BUY",
                quantity=1,
                legs=[
                    {"contract": long_leg, "action": "BUY", "ratio": 1},
                    {"contract": short_leg, "action": "SELL", "ratio": 1},
                ],
                entry_price=net_debit,
                max_loss=max_loss,
                max_profit=max_profit,
                stop_loss=round(net_debit * 0.50, 2),
                take_profit=round(net_debit + (spread_width - net_debit) * 0.75, 2),
                iv_rank=iv_rank,
                underlying_price=chain.underlying_price,
                expiry=chain.expiry,
            )
        ]

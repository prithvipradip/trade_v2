"""Directional strategies: long calls and long puts.

Simple directional plays when ML predicts strong moves.
Best in: Low IV environments (cheap premium), strong directional signals.
"""

from __future__ import annotations

import pandas as pd

from ait.data.options_chain import OptionsChain, OptionContract
from ait.strategies.base import Signal, SignalDirection, Strategy


class LongCall(Strategy):
    """Buy calls for bullish directional plays."""

    @property
    def name(self) -> str:
        return "long_call"

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

        # Prefer low IV (cheap options)
        if iv_rank > 60:
            return []

        liquid_calls = self._filter_liquid(chain.calls)
        if not liquid_calls:
            return []

        # Target: slightly OTM call (delta ~0.35-0.45 for good leverage)
        target = self._find_strike_by_delta(liquid_calls, 0.40)
        if not target:
            return []

        # Stop loss at 50% of premium, take profit at 100% gain
        entry = target.mid
        if entry <= 0:
            return []

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.BULLISH,
                confidence=confidence,
                contract=target,
                action="BUY",
                quantity=1,
                entry_price=entry,
                max_loss=entry * 100,  # Full premium
                max_profit=entry * 2 * 100,  # Target 100% gain
                stop_loss=round(entry * 0.50, 2),
                take_profit=round(entry * 2.0, 2),
                iv_rank=iv_rank,
                underlying_price=chain.underlying_price,
                expiry=chain.expiry,
            )
        ]


class LongPut(Strategy):
    """Buy puts for bearish directional plays."""

    @property
    def name(self) -> str:
        return "long_put"

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

        if iv_rank > 60:
            return []

        liquid_puts = self._filter_liquid(chain.puts)
        if not liquid_puts:
            return []

        target = self._find_strike_by_delta(liquid_puts, 0.40)
        if not target:
            return []

        entry = target.mid
        if entry <= 0:
            return []

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.BEARISH,
                confidence=confidence,
                contract=target,
                action="BUY",
                quantity=1,
                entry_price=entry,
                max_loss=entry * 100,
                max_profit=entry * 2 * 100,
                stop_loss=round(entry * 0.50, 2),
                take_profit=round(entry * 2.0, 2),
                iv_rank=iv_rank,
                underlying_price=chain.underlying_price,
                expiry=chain.expiry,
            )
        ]

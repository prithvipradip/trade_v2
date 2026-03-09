"""Strategy selector — picks the best strategy based on market conditions.

Considers: market regime, ML direction, IV rank, and available strategies.
This is the "brain" that decides WHAT to trade, not just WHETHER to trade.
"""

from __future__ import annotations

import pandas as pd

from ait.config.settings import OptionsConfig
from ait.data.options_chain import OptionsChain
from ait.strategies.base import Signal, SignalDirection, Strategy
from ait.strategies.covered import CashSecuredPut, CoveredCall
from ait.strategies.iron_condor import IronCondor
from ait.strategies.long_options import LongCall, LongPut
from ait.strategies.spreads import BearPutSpread, BullCallSpread
from ait.strategies.straddles import LongStraddle, ShortStrangle
from ait.utils.logging import get_logger

log = get_logger("strategies.selector")

# Map strategy names to classes
STRATEGY_MAP: dict[str, type[Strategy]] = {
    "long_call": LongCall,
    "long_put": LongPut,
    "covered_call": CoveredCall,
    "cash_secured_put": CashSecuredPut,
    "bull_call_spread": BullCallSpread,
    "bear_put_spread": BearPutSpread,
    "iron_condor": IronCondor,
    "long_straddle": LongStraddle,
    "short_strangle": ShortStrangle,
}


class StrategySelector:
    """Selects and ranks strategies based on market conditions."""

    def __init__(self, config: OptionsConfig) -> None:
        self._config = config
        self._strategies: list[Strategy] = []

        for name in config.strategies:
            cls = STRATEGY_MAP.get(name)
            if cls:
                self._strategies.append(cls())
            else:
                log.warning("unknown_strategy", name=name)

        log.info("strategies_loaded", count=len(self._strategies),
                 names=[s.name for s in self._strategies])

    def generate_all_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
    ) -> list[Signal]:
        """Run all enabled strategies and return ranked signals.

        Strategies self-filter based on market conditions — a bullish strategy
        won't generate signals in a bearish market.
        """
        all_signals: list[Signal] = []

        for strategy in self._strategies:
            try:
                signals = strategy.generate_signals(
                    symbol=symbol,
                    chain=chain,
                    market_direction=market_direction,
                    confidence=confidence,
                    iv_rank=iv_rank,
                    historical_data=historical_data,
                )
                all_signals.extend(signals)
            except Exception as e:
                log.warning(
                    "strategy_signal_error",
                    strategy=strategy.name,
                    symbol=symbol,
                    error=str(e),
                )

        # Rank signals by quality
        ranked = self._rank_signals(all_signals)

        log.info(
            "signals_generated",
            symbol=symbol,
            direction=market_direction.value,
            iv_rank=iv_rank,
            total_signals=len(ranked),
            strategies_with_signals=[s.strategy_name for s in ranked],
        )

        return ranked

    def _rank_signals(self, signals: list[Signal]) -> list[Signal]:
        """Rank signals by quality score.

        Higher score = better opportunity.
        """
        if not signals:
            return []

        def score(s: Signal) -> float:
            points = 0.0

            # Confidence (most important) — 0-40 points
            points += s.confidence * 40

            # Risk/reward ratio — 0-30 points
            if s.risk_reward > 0:
                points += min(s.risk_reward * 10, 30)

            # Defined risk bonus — 10 points
            if s.is_defined_risk:
                points += 10

            # IV rank alignment:
            # High IV → selling strategies get bonus
            # Low IV → buying strategies get bonus
            is_selling = s.action == "SELL" or "short" in s.strategy_name or "iron" in s.strategy_name
            if is_selling and s.iv_rank > 50:
                points += 10  # Selling in high IV = good
            elif not is_selling and s.iv_rank < 30:
                points += 10  # Buying in low IV = good

            # Liquidity — penalize wide spreads
            if s.contract and s.contract.spread_pct > 0.05:
                points -= 5

            return points

        return sorted(signals, key=score, reverse=True)

    def get_recommended_strategies(
        self, market_direction: SignalDirection, iv_rank: float
    ) -> list[str]:
        """Get a list of recommended strategies given current conditions.

        Useful for logging/dashboard — shows what the selector is considering.
        """
        recommendations = []

        if market_direction == SignalDirection.BULLISH:
            if iv_rank < 30:
                recommendations.extend(["long_call", "bull_call_spread"])
            elif iv_rank > 50:
                recommendations.extend(["bull_call_spread", "cash_secured_put"])
            else:
                recommendations.extend(["bull_call_spread", "covered_call"])

        elif market_direction == SignalDirection.BEARISH:
            if iv_rank < 30:
                recommendations.extend(["long_put", "bear_put_spread"])
            else:
                recommendations.extend(["bear_put_spread"])

        elif market_direction == SignalDirection.NEUTRAL:
            if iv_rank > 50:
                recommendations.extend(["iron_condor", "short_strangle"])
            else:
                recommendations.extend(["iron_condor"])

        return [r for r in recommendations if r in [s.name for s in self._strategies]]

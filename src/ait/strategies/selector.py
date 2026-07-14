"""Strategy selector — picks the best strategy based on market conditions.

Considers: market regime, ML direction, IV rank, and available strategies.
This is the "brain" that decides WHAT to trade, not just WHETHER to trade.
"""

from __future__ import annotations

import pandas as pd

from ait.config.settings import OptionsConfig
from ait.data.options_chain import OptionsChain
from ait.strategies.base import Signal, SignalDirection, Strategy
from ait.strategies.iron_condor import IronCondor
from ait.strategies.long_options import LongCall, LongPut
from ait.strategies.spreads import BearPutSpread, BullCallSpread
from ait.strategies.straddles import LongStraddle, ShortStrangle
from ait.utils.logging import get_logger

log = get_logger("strategies.selector")

# Map strategy names to classes.
# R12-C: covered_call / cash_secured_put (covered.py), calendar_spread
# (calendar.py) and event_straddle (event_straddle.py) retired to
# deprecated/src/ — never traded live; covered.py could emit naked calls
# when the share-ownership premise wasn't checked.
STRATEGY_MAP: dict[str, type[Strategy]] = {
    "long_call": LongCall,
    "long_put": LongPut,
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

        names = [s.name for s in self._strategies]
        log.info("strategies_loaded", count=len(self._strategies), names=names)

    def generate_calendar_signals(
        self,
        symbol: str,
        chains: list[OptionsChain],
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
    ) -> list[Signal]:
        """R12-C stub: CalendarSpread retired to deprecated/src/calendar.py.

        Kept because the orchestrator still calls this unconditionally; it
        already returned [] whenever calendar_spread wasn't configured (it
        never was in config.yaml), so this is behavior-identical.
        """
        return []

    def generate_event_straddle_signals(
        self,
        symbol: str,
        chains: list[OptionsChain],
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        economic_cal=None,
    ) -> list[Signal]:
        """R12-C stub: EventStraddle retired to deprecated/src/event_straddle.py.

        Kept because the orchestrator still calls this unconditionally; it
        already returned [] whenever event_straddle wasn't configured (it
        never was in config.yaml), so this is behavior-identical.
        """
        return []

    def generate_all_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
        risk_budget: float | None = None,
    ) -> list[Signal]:
        """Run all enabled strategies and return ranked signals.

        Strategies self-filter based on market conditions — a bullish strategy
        won't generate signals in a bearish market.
        """
        all_signals: list[Signal] = []

        for strategy in self._strategies:
            try:
                # R7: hand every strategy the account's per-trade budget so
                # defined-risk construction can fit the funded account, not
                # just the paper one.
                strategy.risk_budget = risk_budget
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

    # Strategies that profit from selling premium (short vega)
    SELLING_STRATEGIES = frozenset({
        "iron_condor", "short_strangle",
    })
    # Strategies that profit from buying premium (long vega)
    BUYING_STRATEGIES = frozenset({
        "long_call", "long_put", "long_straddle",
    })

    def _rank_signals(self, signals: list[Signal]) -> list[Signal]:
        """Rank signals by quality score.

        Higher score = better opportunity. Uses IV-aware scoring.
        """
        if not signals:
            return []

        def score(s: Signal) -> float:
            points = 0.0

            # IRON CONDOR PRIORITY — always preferred (theta collection, direction-neutral)
            # This matches the backtest engine logic that proved +311% returns
            if s.strategy_name == "iron_condor":
                points += 50  # Massive bonus — iron condors are our primary strategy

            # Confidence (most important) — 0-40 points
            points += s.confidence * 40

            # Risk/reward ratio — 0-30 points
            if s.risk_reward > 0:
                points += min(s.risk_reward * 10, 30)

            # Defined risk bonus — 10 points
            if s.is_defined_risk:
                points += 10

            # IV rank alignment (smooth scoring, not binary):
            # Selling strategies score higher as IV rank increases
            # Buying strategies score higher as IV rank decreases
            is_selling = s.strategy_name in self.SELLING_STRATEGIES
            is_buying = s.strategy_name in self.BUYING_STRATEGIES
            iv = s.iv_rank  # 0-100

            if is_selling:
                # Linear bonus: 0 pts at IV rank 30, +15 pts at IV rank 80+
                points += max(0, min(15, (iv - 30) * 0.30))
            elif is_buying:
                # Linear bonus: 0 pts at IV rank 50, +15 pts at IV rank 10
                points += max(0, min(15, (50 - iv) * 0.375))

            # Spread strategies (directional but defined risk) get moderate IV bonus
            if s.strategy_name in ("bull_call_spread", "bear_put_spread"):
                # Best in moderate IV (25-50), penalize extremes
                iv_dist_from_ideal = abs(iv - 37.5)
                points += max(0, 8 - iv_dist_from_ideal * 0.2)

            # Liquidity — penalize wide spreads
            if s.contract and s.contract.spread_pct > 0.05:
                points -= 5

            return points

        return sorted(signals, key=score, reverse=True)

    def get_recommended_strategies(
        self, market_direction: SignalDirection, iv_rank: float
    ) -> list[str]:
        """Get a list of recommended strategies given current conditions.

        Uses IV rank to determine optimal strategy type:
        - Low IV (< 30): Buy premium (long options, debit spreads)
        - Mid IV (30-50): Defined-risk directional (spreads)
        - High IV (> 50): Sell premium (condors, strangles, covered)
        """
        recommendations = []
        enabled = {s.name for s in self._strategies}

        if market_direction == SignalDirection.BULLISH:
            if iv_rank < 30:
                recommendations.extend(["long_call", "bull_call_spread"])
            else:
                recommendations.extend(["bull_call_spread"])

        elif market_direction == SignalDirection.BEARISH:
            if iv_rank < 30:
                recommendations.extend(["long_put", "bear_put_spread"])
            elif iv_rank > 50:
                recommendations.extend(["bear_put_spread"])
            else:
                recommendations.extend(["bear_put_spread", "long_put"])

        elif market_direction == SignalDirection.NEUTRAL:
            if iv_rank > 50:
                recommendations.extend(["iron_condor", "short_strangle"])
            elif iv_rank > 30:
                recommendations.extend(["iron_condor"])
            else:
                recommendations.extend(["long_straddle"])

        return [r for r in recommendations if r in enabled]

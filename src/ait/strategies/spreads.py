"""Vertical spread strategies: bull call spreads and bear put spreads.

Defined-risk directional strategies that cost less than naked options.
Best in: Moderate directional conviction with defined risk tolerance.

Bull call spread = buy lower strike call + sell higher strike call (net debit)
Bear put spread = buy higher strike put + sell lower strike put (net debit)

W108 4a: the width is fitted to the per-trade budget (``risk_budget``), as the condor's is. A debit
spread's max loss is its DEBIT, not its width, so the fit is priced (ait.risk.width_sizing
.fitted_debit_width, the rule the research engine calls): the delta-placed spread is kept when its
mid debit x 100 fits the budget; otherwise the short leg moves inward to the widest listed strike
whose debit fits; none fits = no signal. The long leg never moves.
"""

from __future__ import annotations

import pandas as pd

from ait.data.options_chain import OptionsChain, OptionContract
from ait.risk.width_sizing import fitted_debit_width  # W108 4a: shared with backtesting/engine.py
from ait.strategies.base import Signal, SignalDirection, Strategy
from ait.utils.logging import get_logger

log = get_logger("strategies.spreads")


def _budget_fitted_short_leg(
    strategy: str,
    symbol: str,
    long_leg: OptionContract,
    short_leg: OptionContract,
    candidates: list[OptionContract],
    risk_budget: float | None,
) -> OptionContract | None:
    """W108 4a: the short leg after the budget fit, or None when no listed width fits the budget.

    The debit of each listed short strike between the long leg and the delta-placed short leg is
    the MID difference, the number gate 6b later checks as ``signal.max_loss`` (debit x 100).
    Unchanged when there is no budget or the delta-placed spread already fits it.
    """
    by_width = {
        abs(c.strike - long_leg.strike): c for c in candidates
        if abs(c.strike - long_leg.strike) <= abs(short_leg.strike - long_leg.strike)
    }
    natural = abs(short_leg.strike - long_leg.strike)
    width = fitted_debit_width(
        natural, risk_budget, {w: long_leg.mid - c.mid for w, c in by_width.items()})
    if width is None:
        # W36 lesson: a refusal is never silent.
        log.info("debit_spread_rejected_no_width_fits_budget", strategy=strategy, symbol=symbol,
                 risk_budget=round(float(risk_budget or 0.0), 2), natural_width=round(natural, 2),
                 natural_debit=round(long_leg.mid - short_leg.mid, 4))
        return None
    return short_leg if width == natural else by_width[width]


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
        # W108 4a: fit the width to the per-trade budget (the short leg moves inward only).
        short_leg = _budget_fitted_short_leg(
            self.name, symbol, long_leg, short_leg, candidates, self.risk_budget)
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
        # W108 4a: fit the width to the per-trade budget (the short leg moves inward only).
        short_leg = _budget_fitted_short_leg(
            self.name, symbol, long_leg, short_leg, candidates, self.risk_budget)
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

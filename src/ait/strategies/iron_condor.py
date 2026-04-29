"""Iron condor strategy — premium selling in range-bound markets.

Sell OTM put spread + sell OTM call spread simultaneously.
Profits when underlying stays between the short strikes.

Best in: High IV (expensive premium), neutral/range-bound markets.
Risk: Defined — max loss = width of wider spread - net credit.
"""

from __future__ import annotations

import pandas as pd

from ait.data.options_chain import OptionsChain, OptionContract
from ait.strategies.base import Signal, SignalDirection, Strategy


class IronCondor(Strategy):
    """Iron condor — sell premium on both sides."""

    @property
    def name(self) -> str:
        return "iron_condor"

    @property
    def direction_bias(self) -> SignalDirection | None:
        return None  # Direction neutral

    def generate_signals(
        self,
        symbol: str,
        chain: OptionsChain,
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
        historical_data: pd.DataFrame | None = None,
    ) -> list[Signal]:
        # Iron condors work in ANY direction — they profit from theta decay
        # regardless of whether the market goes up, down, or sideways.
        # Only skip in extreme low IV where premium isn't worth the risk.
        # IV rank floor — read from env so we can backtest different values.
        # Default 20 (loosened from 30 for 2-week data-collection window —
        # need 30+ paper trades to validate the bot before tightening).
        # Research suggests IV rank > 50 is ideal but trades too rarely.
        import os
        iv_floor = float(os.environ.get("AIT_IRON_CONDOR_IV_FLOOR", "20"))
        if iv_rank < iv_floor:
            return []

        liquid_calls = self._filter_liquid(chain.calls)
        liquid_puts = self._filter_liquid(chain.puts)

        if len(liquid_calls) < 2 or len(liquid_puts) < 2:
            return []

        price = chain.underlying_price

        # Put side (below price):
        # Sell put at delta ~0.20, buy put 1-2 strikes lower
        short_put = self._find_strike_by_delta(liquid_puts, 0.20)
        if not short_put:
            return []

        # Buy protection at least 2 strikes below short put for meaningful width
        long_put_candidates = sorted(
            [p for p in liquid_puts if p.strike < short_put.strike],
            key=lambda p: p.strike, reverse=True,
        )
        if len(long_put_candidates) < 2:
            return []
        long_put = long_put_candidates[1]  # Skip adjacent, take 2nd strike down

        # Call side (above price):
        # Sell call at delta ~0.20, buy call 1-2 strikes higher
        short_call = self._find_strike_by_delta(liquid_calls, 0.20)
        if not short_call:
            return []

        # Buy protection at least 2 strikes above short call for meaningful width
        long_call_candidates = sorted(
            [c for c in liquid_calls if c.strike > short_call.strike],
            key=lambda c: c.strike,
        )
        if len(long_call_candidates) < 2:
            return []
        long_call = long_call_candidates[1]  # Skip adjacent, take 2nd strike up

        # Verify structure: long_put < short_put < price < short_call < long_call
        if not (long_put.strike < short_put.strike < price < short_call.strike < long_call.strike):
            return []

        # Calculate pricing
        put_credit = short_put.mid - long_put.mid
        call_credit = short_call.mid - long_call.mid
        total_credit = put_credit + call_credit

        if total_credit <= 0:
            return []

        # Max loss = wider spread width - total credit
        put_width = short_put.strike - long_put.strike
        call_width = long_call.strike - short_call.strike
        max_width = max(put_width, call_width)
        max_loss = (max_width - total_credit) * 100

        if max_loss <= 0:
            return []

        max_profit = total_credit * 100

        # Exit at 50% of max profit (take profit early)
        take_profit = round(total_credit * 0.50, 2)
        # Stop loss at 2x credit received
        stop_loss = round(total_credit * 2.0, 2)

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=confidence,
                contract=short_put,  # Primary reference leg
                action="SELL",
                quantity=1,
                legs=[
                    {"contract": long_put, "action": "BUY", "ratio": 1},
                    {"contract": short_put, "action": "SELL", "ratio": 1},
                    {"contract": short_call, "action": "SELL", "ratio": 1},
                    {"contract": long_call, "action": "BUY", "ratio": 1},
                ],
                entry_price=total_credit,
                max_loss=max_loss,
                max_profit=max_profit,
                stop_loss=stop_loss,
                take_profit=take_profit,
                iv_rank=iv_rank,
                underlying_price=price,
                expiry=chain.expiry,
            )
        ]

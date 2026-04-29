"""Calendar spread strategy — for LOW IV environments.

Sells a near-term option, buys a far-dated option at the SAME STRIKE.
Profits from theta decay differential (short option decays faster than
long option) AND from IV expansion (long vega).

Best in: LOW IV (cheap premium to buy), expected pin near strike.
Risk: Defined — max loss = net debit paid.

Why complementary to iron condors:
  Iron condors WANT high IV (sell expensive premium)
  Calendars     WANT low IV  (buy cheap premium that will rise)
"""

from __future__ import annotations

from datetime import date

from ait.data.options_chain import OptionsChain, OptionContract
from ait.strategies.base import Signal, SignalDirection
from ait.utils.logging import get_logger

log = get_logger("strategies.calendar")


class CalendarSpread:
    """Sell near-term option, buy same-strike far-dated option."""

    name = "calendar_spread"
    direction_bias = None  # Direction neutral

    def generate_signals(
        self,
        symbol: str,
        chains: list[OptionsChain],
        market_direction: SignalDirection,
        confidence: float,
        iv_rank: float,
    ) -> list[Signal]:
        """Generate calendar spread signals from multiple expiry chains.

        Different from other strategies: needs at least 2 expiries (chains).
        """
        # Calendars LOVE low IV — opposite of iron condors. Skip if IV too high.
        if iv_rank > 50:
            return []

        if len(chains) < 2:
            return []  # Need at least 2 expiries

        # Sort chains by expiry (near → far)
        chains_sorted = sorted(
            [c for c in chains if c.expiry is not None],
            key=lambda c: c.expiry,
        )
        if len(chains_sorted) < 2:
            return []

        today = date.today()
        # Pick near = 21-35 DTE, far = 50-70 DTE
        near_chain = None
        far_chain = None
        for c in chains_sorted:
            dte = (c.expiry - today).days
            if near_chain is None and 14 <= dte <= 40:
                near_chain = c
            elif far_chain is None and 45 <= dte <= 80:
                far_chain = c
            if near_chain and far_chain:
                break

        if not near_chain or not far_chain:
            return []

        price = near_chain.underlying_price
        if price <= 0:
            return []

        # Use calls for the calendar (could be either, calls work fine).
        # Find ATM strike — closest to current price.
        near_calls = self._filter_liquid(near_chain.calls)
        far_calls = self._filter_liquid(far_chain.calls)
        if not near_calls or not far_calls:
            return []

        # Match strikes between near and far
        common_strikes = set(c.strike for c in near_calls) & set(c.strike for c in far_calls)
        if not common_strikes:
            return []

        # ATM strike (closest to current price)
        atm_strike = min(common_strikes, key=lambda s: abs(s - price))

        near_call = next((c for c in near_calls if c.strike == atm_strike), None)
        far_call = next((c for c in far_calls if c.strike == atm_strike), None)
        if not near_call or not far_call:
            return []

        # Net debit = buy far - sell near (must be positive)
        debit = far_call.mid - near_call.mid
        if debit <= 0:
            return []

        # Max loss = the debit paid. Max profit hard to estimate analytically
        # (depends on IV and time path), but typically ~3x the debit if pinned.
        max_loss = debit * 100
        max_profit_est = debit * 100 * 3  # rough estimate

        legs = [
            {"strike": atm_strike, "right": "C", "action": "SELL",
             "expiry": near_call.expiry.isoformat()},
            {"strike": atm_strike, "right": "C", "action": "BUY",
             "expiry": far_call.expiry.isoformat()},
        ]

        return [
            Signal(
                symbol=symbol,
                strategy_name=self.name,
                direction=SignalDirection.NEUTRAL,
                confidence=confidence,
                contract=near_call,
                action="BUY",  # Net debit position
                quantity=1,
                legs=legs,
                entry_price=debit,
                max_loss=max_loss,
                max_profit=max_profit_est,
                stop_loss=round(debit * 0.5, 2),  # 50% loss stop
                take_profit=round(debit * 1.0, 2),  # 100% profit target
                iv_rank=iv_rank,
                underlying_price=price,
                expiry=near_call.expiry,
            )
        ]

    @staticmethod
    def _filter_liquid(contracts: list[OptionContract]) -> list[OptionContract]:
        """Drop illiquid options."""
        out = []
        for c in contracts:
            if not c.bid or not c.ask:
                continue
            if c.bid <= 0 or c.ask <= 0:
                continue
            if (c.ask - c.bid) / max(c.mid, 0.01) > 0.20:  # Wide spread
                continue
            out.append(c)
        return out

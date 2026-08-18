"""Portfolio Greeks hedging — rebalances delta exposure.

When portfolio delta drifts beyond safe thresholds, this module
generates hedging orders (typically SPY shares or options) to
bring delta back within limits.

Only hedges with SPY — liquid, tight spreads, and inversely
correlated with most positions.
"""

from __future__ import annotations

from dataclasses import dataclass

from ait.risk.manager import PortfolioGreeks
from ait.utils.logging import get_logger

log = get_logger("risk.hedging")


@dataclass
class HedgeRecommendation:
    """A recommended hedging action."""

    action: str  # "BUY" or "SELL"
    symbol: str  # Always "SPY" for now
    quantity: int  # Number of shares
    reason: str
    current_delta: float
    target_delta: float


class DeltaHedger:
    """Monitors and hedges portfolio delta exposure."""

    def __init__(
        self,
        max_delta_pct: float = 0.30,
        hedge_trigger_pct: float = 0.50,
        hedge_symbol: str = "SPY",
    ) -> None:
        """
        Args:
            max_delta_pct: Maximum allowed portfolio delta as % of account value
            hedge_trigger_pct: Trigger hedging when delta > this % of max allowed
            hedge_symbol: Symbol to hedge with (SPY by default)
        """
        self._max_delta_pct = max_delta_pct
        self._trigger_pct = hedge_trigger_pct
        self._hedge_symbol = hedge_symbol

    def check_hedge_needed(
        self,
        portfolio_greeks: PortfolioGreeks,
        account_value: float,
        spy_price: float,
    ) -> HedgeRecommendation | None:
        """Check if portfolio delta needs hedging.

        Returns a hedge recommendation or None if no action needed.
        """
        if account_value <= 0 or spy_price <= 0:
            return None

        max_delta_value = account_value * self._max_delta_pct
        trigger_delta = max_delta_value * self._trigger_pct
        current_delta = portfolio_greeks.delta

        # Check if delta exceeds trigger threshold
        if abs(current_delta) <= trigger_delta:
            return None

        # Calculate how many SPY shares to hedge
        # Each SPY share = 1 delta point
        target_delta = 0.0  # Hedge back to neutral
        delta_to_hedge = current_delta - target_delta

        # Round to nearest lot of 10 shares
        shares = int(abs(delta_to_hedge) / 10) * 10
        if shares < 10:
            return None  # Not worth hedging small amounts

        action = "SELL" if delta_to_hedge > 0 else "BUY"

        recommendation = HedgeRecommendation(
            action=action,
            symbol=self._hedge_symbol,
            quantity=shares,
            reason=(
                f"Portfolio delta ${current_delta:.0f} exceeds "
                f"trigger ${trigger_delta:.0f} (max ${max_delta_value:.0f})"
            ),
            current_delta=current_delta,
            target_delta=target_delta,
        )

        log.info(
            "hedge_recommended",
            action=action,
            shares=shares,
            current_delta=current_delta,
            trigger=trigger_delta,
            max_delta=max_delta_value,
        )

        return recommendation

    def calculate_hedge_cost(
        self, recommendation: HedgeRecommendation, price: float
    ) -> float:
        """Estimate the cost of a hedge."""
        return recommendation.quantity * price

"""Volatility-adjusted position sizing with Kelly criterion.

Determines how many contracts to trade based on:
- Account size and max position percentage
- Current volatility of the underlying
- ML model confidence
- Strategy type (spreads risk less than naked options)
"""

from __future__ import annotations

from dataclasses import dataclass

from ait.config.settings import PositionConfig, RiskConfig
from ait.utils.logging import get_logger

log = get_logger("risk.sizer")


@dataclass
class PositionSize:
    """Recommended position size with reasoning."""

    contracts: int
    max_risk_dollars: float
    confidence_adjustment: float
    volatility_adjustment: float
    reason: str


class PositionSizer:
    """Calculate position sizes based on risk parameters."""

    # Strategy risk multipliers — how much of the max position to use
    STRATEGY_RISK = {
        "long_call": 1.0,        # Full premium at risk
        "long_put": 1.0,
        "covered_call": 0.5,     # Shares + short call
        "cash_secured_put": 0.5,
        "bull_call_spread": 0.6, # Defined risk (max loss = net debit)
        "bear_put_spread": 0.6,
        "iron_condor": 0.4,      # Defined risk, both sides
        "long_straddle": 1.2,    # Double premium
        "short_strangle": 0.8,   # Undefined risk, margin required
    }

    def __init__(self, position_config: PositionConfig, risk_config: RiskConfig) -> None:
        self._pos_config = position_config
        self._risk_config = risk_config

    # Strategies that benefit from high IV (selling premium)
    _SELLING_STRATEGIES = frozenset({
        "covered_call", "cash_secured_put", "iron_condor", "short_strangle",
    })

    def calculate(
        self,
        account_value: float,
        option_price: float,
        confidence: float,
        implied_vol: float,
        strategy: str,
        underlying_price: float,
        iv_rank: float = 50.0,
    ) -> PositionSize:
        """Calculate recommended position size.

        Args:
            account_value: Total account value (net liquidation)
            option_price: Price per contract (mid price)
            confidence: ML model confidence (0.0 to 1.0)
            implied_vol: Implied volatility of the option
            strategy: Strategy name (e.g., "bull_call_spread")
            underlying_price: Current price of underlying
            iv_rank: IV rank percentile 0-100 (used for strategy-aware sizing)

        Returns:
            PositionSize with recommended contracts and reasoning
        """
        if account_value <= 0 or option_price <= 0:
            return PositionSize(0, 0, 0, 0, "invalid inputs")

        # Base: max position as percentage of account
        max_position_value = account_value * self._pos_config.max_position_pct
        cost_per_contract = option_price * 100  # Options are 100 shares

        # Confidence adjustment: scale down for low confidence
        # At min_confidence, use 50% of max. At 1.0, use 100%.
        min_conf = self._risk_config.min_confidence
        conf_adj = 0.5 + 0.5 * ((confidence - min_conf) / (1.0 - min_conf))
        conf_adj = max(0.3, min(1.0, conf_adj))

        # Volatility adjustment: smooth linear scale-down in high-vol environments
        # IV <= 20% → full size (1.0), IV >= 60% → half size (0.5)
        if implied_vol <= 0.20:
            vol_adj = 1.0
        elif implied_vol >= 0.60:
            vol_adj = 0.5
        else:
            # Linear interpolation: 1.0 at 20% → 0.5 at 60%
            vol_adj = 1.0 - 0.5 * ((implied_vol - 0.20) / 0.40)

        # IV rank alignment: boost size when IV conditions favor the strategy
        # Selling strategies benefit from high IV rank (more premium to collect)
        # Buying strategies benefit from low IV rank (cheaper options)
        iv_rank_adj = 1.0
        if strategy in self._SELLING_STRATEGIES:
            # High IV rank → boost up to 1.2x; Low IV rank → reduce to 0.8x
            iv_rank_adj = 0.8 + 0.4 * (iv_rank / 100.0)
        else:
            # Low IV rank → boost up to 1.2x; High IV rank → reduce to 0.8x
            iv_rank_adj = 1.2 - 0.4 * (iv_rank / 100.0)
        iv_rank_adj = max(0.8, min(1.2, iv_rank_adj))

        # Strategy adjustment
        strategy_adj = self.STRATEGY_RISK.get(strategy, 1.0)

        # Calculate contracts
        adjusted_max = max_position_value * conf_adj * vol_adj * strategy_adj * iv_rank_adj
        max_contracts = int(adjusted_max / cost_per_contract)

        # Scale cap with account size: 10 contracts per $100k, minimum 1
        account_scale_cap = max(1, int(account_value / 10_000))
        contracts = max(1, min(max_contracts, account_scale_cap))

        # Max risk: for spreads, risk is the net debit (option_price already is net debit).
        # For naked options, risk is the full premium paid.
        max_risk = contracts * cost_per_contract

        reason = (
            f"account=${account_value:.0f}, "
            f"max_pos={self._pos_config.max_position_pct:.0%}, "
            f"conf_adj={conf_adj:.2f}, vol_adj={vol_adj:.2f}, "
            f"iv_rank_adj={iv_rank_adj:.2f}, strat_adj={strategy_adj:.1f}"
        )

        log.debug(
            "position_sized",
            contracts=contracts,
            max_risk=max_risk,
            confidence=confidence,
            implied_vol=implied_vol,
            strategy=strategy,
        )

        return PositionSize(
            contracts=contracts,
            max_risk_dollars=max_risk,
            confidence_adjustment=conf_adj,
            volatility_adjustment=vol_adj,
            reason=reason,
        )

    def max_contracts_for_budget(
        self, budget: float, option_price: float
    ) -> int:
        """Simple calculation: how many contracts can we afford."""
        if option_price <= 0:
            return 0
        return int(budget / (option_price * 100))

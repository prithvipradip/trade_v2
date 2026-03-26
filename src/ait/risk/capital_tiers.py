"""Capital-tier strategy selection.

Automatically selects appropriate strategies based on account size.
Small accounts can't afford iron condors on $600 stocks, so we start
with credit spreads and graduate to iron condors as capital grows.

Tiers based on research (Option Alpha, tastytrade studies):
  - Micro  ($0-$2k):   $1-wide credit spreads, 1-2 positions max
  - Small  ($2k-$5k):  $2-5 wide spreads + small iron condors, 2-3 positions
  - Medium ($5k-$25k): Full iron condors + spreads, 3-5 positions
  - Large  ($25k+):    No PDT limits, full strategy set, 5+ positions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ait.utils.logging import get_logger

log = get_logger("risk.capital_tiers")


class CapitalTier(str, Enum):
    MICRO = "micro"      # $0 - $2,000
    SMALL = "small"      # $2,000 - $5,000
    MEDIUM = "medium"    # $5,000 - $25,000
    LARGE = "large"      # $25,000+


@dataclass
class TierConfig:
    """Strategy constraints for a capital tier."""
    tier: CapitalTier
    min_capital: float
    max_capital: float

    # Strategy selection
    allowed_strategies: list[str]
    prefer_strategy: str           # Default/preferred strategy

    # Position sizing
    max_risk_per_trade_pct: float  # Max % of account to risk per trade
    max_positions: int             # Max concurrent positions
    cash_reserve_pct: float        # Keep this % in cash always

    # ML confidence
    min_confidence: float          # Higher for directional, lower for neutral

    # Spread sizing
    max_wing_width: float          # Max width in dollars
    min_wing_width: float          # Min width in dollars

    # Exit rules
    stop_loss_pct: float           # Tighter for directional trades
    profit_target_pct: float       # Take profits

    # Underlying filters
    max_underlying_price: float    # Skip stocks above this price
    preferred_underlyings: list[str]  # Best tickers for this tier


# ---------------------------------------------------------------------------
# Tier definitions — backed by research
# ---------------------------------------------------------------------------

TIERS = {
    CapitalTier.MICRO: TierConfig(
        tier=CapitalTier.MICRO,
        min_capital=0,
        max_capital=2_000,
        allowed_strategies=["bull_call_spread", "bear_put_spread"],
        prefer_strategy="bull_call_spread",  # Debit spreads — risk/reward favors us
        max_risk_per_trade_pct=0.10,    # 10% = $70 per trade on $700
        max_positions=2,
        cash_reserve_pct=0.40,
        min_confidence=0.65,            # Debit spreads profit at 40%+ win rate
        max_wing_width=2.0,
        min_wing_width=1.0,
        stop_loss_pct=0.25,             # Tight stops — can't afford big losses
        profit_target_pct=0.50,
        max_underlying_price=700.0,     # SPY $1-wide spreads are ~$65-80 risk
        preferred_underlyings=["SPY"],  # SPY only — 63% ML accuracy, everything else is coin flip
    ),
    CapitalTier.SMALL: TierConfig(
        tier=CapitalTier.SMALL,
        min_capital=2_000,
        max_capital=5_000,
        allowed_strategies=["bull_call_spread", "bear_put_spread", "iron_condor"],
        prefer_strategy="iron_condor",
        max_risk_per_trade_pct=0.07,
        max_positions=3,
        cash_reserve_pct=0.35,
        min_confidence=0.65,            # Iron condors need less conviction
        max_wing_width=5.0,
        min_wing_width=2.0,
        stop_loss_pct=0.35,
        profit_target_pct=0.50,
        max_underlying_price=300.0,
        preferred_underlyings=["SPY", "QQQ", "IWM", "AMD", "AAPL"],
    ),
    CapitalTier.MEDIUM: TierConfig(
        tier=CapitalTier.MEDIUM,
        min_capital=5_000,
        max_capital=25_000,
        allowed_strategies=[
            "bull_call_spread", "bear_put_spread", "iron_condor",
            "long_call", "long_put",
        ],
        prefer_strategy="iron_condor",
        max_risk_per_trade_pct=0.05,
        max_positions=5,
        cash_reserve_pct=0.30,
        min_confidence=0.55,
        max_wing_width=10.0,
        min_wing_width=5.0,
        stop_loss_pct=0.35,
        profit_target_pct=0.50,
        max_underlying_price=700.0,
        preferred_underlyings=["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMD"],
    ),
    CapitalTier.LARGE: TierConfig(
        tier=CapitalTier.LARGE,
        min_capital=25_000,
        max_capital=float("inf"),
        allowed_strategies=[
            "bull_call_spread", "bear_put_spread", "iron_condor",
            "long_call", "long_put", "long_straddle", "short_strangle",
        ],
        prefer_strategy="iron_condor",
        max_risk_per_trade_pct=0.04,
        max_positions=8,
        cash_reserve_pct=0.25,
        min_confidence=0.55,
        max_wing_width=25.0,
        min_wing_width=5.0,
        stop_loss_pct=0.35,
        profit_target_pct=0.50,
        max_underlying_price=float("inf"),
        preferred_underlyings=[
            "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT",
            "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL",
        ],
    ),
}


class CapitalTierManager:
    """Determines strategy constraints based on current account capital."""

    def __init__(self):
        self._current_tier: CapitalTier | None = None
        self._last_capital: float = 0

    def get_tier(self, capital: float) -> CapitalTier:
        """Determine the capital tier for a given account balance."""
        if capital >= 25_000:
            tier = CapitalTier.LARGE
        elif capital >= 5_000:
            tier = CapitalTier.MEDIUM
        elif capital >= 2_000:
            tier = CapitalTier.SMALL
        else:
            tier = CapitalTier.MICRO

        # Log tier changes
        if tier != self._current_tier:
            if self._current_tier is not None:
                direction = "upgraded" if capital > self._last_capital else "downgraded"
                log.info("capital_tier_changed",
                         old_tier=self._current_tier.value,
                         new_tier=tier.value,
                         capital=f"${capital:,.2f}",
                         direction=direction)
            else:
                log.info("capital_tier_initialized",
                         tier=tier.value,
                         capital=f"${capital:,.2f}")
            self._current_tier = tier
            self._last_capital = capital

        return tier

    def get_config(self, capital: float) -> TierConfig:
        """Get the full tier configuration for current capital."""
        tier = self.get_tier(capital)
        return TIERS[tier]

    def filter_strategies(self, strategies: list[str], capital: float) -> list[str]:
        """Filter strategies to only those affordable at current capital."""
        config = self.get_config(capital)
        return [s for s in strategies if s in config.allowed_strategies]

    def filter_universe(self, symbols: list[str], capital: float) -> list[str]:
        """Filter universe to preferred underlyings for current capital tier."""
        config = self.get_config(capital)
        preferred = set(config.preferred_underlyings)
        # Keep preferred symbols that are in the universe, maintaining order
        filtered = [s for s in symbols if s in preferred]
        return filtered if filtered else symbols[:4]  # Fallback: first 4

    def get_wing_width(self, underlying_price: float, capital: float) -> float:
        """Get appropriate wing width for current capital and underlying."""
        config = self.get_config(capital)
        # Scale wing width with underlying price, clamped to tier limits
        ideal = round(underlying_price / 50)  # ~2% of stock price
        return max(config.min_wing_width, min(ideal, config.max_wing_width))

    def get_max_risk(self, capital: float) -> float:
        """Get max dollar risk per trade."""
        config = self.get_config(capital)
        available = capital * (1 - config.cash_reserve_pct)
        return available * config.max_risk_per_trade_pct

    def can_trade_symbol(self, symbol: str, underlying_price: float, capital: float) -> bool:
        """Check if we can afford to trade this symbol at current capital."""
        config = self.get_config(capital)
        if underlying_price > config.max_underlying_price:
            return False
        # Check if minimum spread ($1 wide) is affordable
        min_risk = config.min_wing_width * 100  # $100 per $1 width
        max_risk = self.get_max_risk(capital)
        return max_risk >= min_risk

    def get_position_count_limit(self, capital: float) -> int:
        """Max concurrent positions for current capital."""
        return self.get_config(capital).max_positions

    def summary(self, capital: float) -> dict:
        """Return a summary of current tier constraints."""
        config = self.get_config(capital)
        return {
            "tier": config.tier.value,
            "capital": f"${capital:,.2f}",
            "strategies": config.allowed_strategies,
            "preferred": config.prefer_strategy,
            "max_positions": config.max_positions,
            "max_risk_per_trade": f"${self.get_max_risk(capital):,.2f}",
            "wing_width_range": f"${config.min_wing_width}-${config.max_wing_width}",
            "cash_reserve": f"{config.cash_reserve_pct:.0%}",
            "universe": config.preferred_underlyings,
        }

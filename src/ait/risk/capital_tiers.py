"""Capital-tier universe selection.

R12-C simplification (2026-07-13): this module once carried a full parallel
risk system (per-tier wing widths, stop/profit rules, risk-per-trade math,
strategy filtering, affordability checks) that NOTHING consumed — the real
sizing/stops live in risk.manager / position_sizer / the exit engine. The
orchestrator consumes exactly two things:

  * ``get_config(capital)``      -> tier metadata for the capital_tier_active log line
  * ``filter_universe(symbols, capital)`` -> NLV-appropriate underlyings

Everything else was deleted; the dead logic is in git history (this file
pre-R12) if it is ever wanted again.

Tier boundaries based on research (Option Alpha, tastytrade studies):
  - Micro  ($0-$2k)    - Small ($2k-$5k)
  - Medium ($5k-$25k)  - Large ($25k+)
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
    """What the live system actually reads per tier: identity for the log
    line (tier / allowed_strategies / max_positions are logged, not enforced
    here) and the preferred-underlyings universe filter."""

    tier: CapitalTier
    allowed_strategies: list[str]
    max_positions: int
    preferred_underlyings: list[str]


# ---------------------------------------------------------------------------
# NLV -> tier -> universe table
# ---------------------------------------------------------------------------

TIERS = {
    CapitalTier.MICRO: TierConfig(
        tier=CapitalTier.MICRO,
        allowed_strategies=["bull_call_spread", "bear_put_spread"],
        max_positions=2,
        preferred_underlyings=["SPY"],  # SPY only — 63% ML accuracy, everything else is coin flip
    ),
    CapitalTier.SMALL: TierConfig(
        tier=CapitalTier.SMALL,
        allowed_strategies=["bull_call_spread", "bear_put_spread", "iron_condor"],
        max_positions=3,
        preferred_underlyings=["SPY", "QQQ", "IWM", "AMD", "AAPL",
                               "GLD", "TLT", "XLE"],  # cheap underlyings suit small accounts (2026-07-07, $3k CAD launch plan)
    ),
    CapitalTier.MEDIUM: TierConfig(
        tier=CapitalTier.MEDIUM,
        allowed_strategies=[
            "bull_call_spread", "bear_put_spread", "iron_condor",
            "long_call", "long_put",
        ],
        max_positions=5,
        preferred_underlyings=["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMD",
                               "GLD", "TLT", "XLE"],  # deep-audit SR-H2: tier filter was silently deleting the R2.10 diversifiers
    ),
    CapitalTier.LARGE: TierConfig(
        tier=CapitalTier.LARGE,
        allowed_strategies=[
            "bull_call_spread", "bear_put_spread", "iron_condor",
            "long_call", "long_put", "long_straddle", "short_strangle",
        ],
        max_positions=8,
        preferred_underlyings=[
            "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT",
            "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL",
            "GLD", "TLT", "XLE",  # deep-audit SR-H2
        ],
    ),
}


class CapitalTierManager:
    """Maps current account capital to a tier; filters the trading universe."""

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
        """Get the tier configuration for current capital."""
        tier = self.get_tier(capital)
        return TIERS[tier]

    def filter_universe(self, symbols: list[str], capital: float) -> list[str]:
        """Filter universe to preferred underlyings for current capital tier."""
        config = self.get_config(capital)
        preferred = set(config.preferred_underlyings)
        # Keep preferred symbols that are in the universe, maintaining order
        filtered = [s for s in symbols if s in preferred]
        return filtered if filtered else symbols[:4]  # Fallback: first 4

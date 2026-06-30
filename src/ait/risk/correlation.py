"""Position correlation guard — prevents correlated position stacking.

If you're long AAPL calls and MSFT calls, you're essentially making
the same bet twice. This module tracks correlations and blocks
new trades that are too correlated with existing positions.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from ait.data.cache import TTLCache
from ait.utils.logging import get_logger

log = get_logger("risk.correlation")

# Known high-correlation groups (fallback when historical data unavailable)
SECTOR_GROUPS = {
    "mega_tech": {"AAPL", "MSFT", "GOOGL", "META", "AMZN"},
    "semiconductors": {"NVDA", "AMD", "INTC", "AVGO", "TSM"},
    "index": {"SPY", "QQQ", "IWM", "DIA"},
    "ev_momentum": {"TSLA", "RIVN", "LCID"},
}


class CorrelationGuard:
    """Prevents stacking correlated positions."""

    def __init__(
        self,
        max_correlation: float = 0.75,
        lookback_days: int = 60,
        max_correlated_positions: int = 1,
    ) -> None:
        self._max_corr = max_correlation
        self._lookback = lookback_days
        # Max positions correlated above the threshold allowed at once. Set to
        # 1 (conservative, by user choice 2026-06-29): only ONE position per
        # correlated cluster — so at most one index bet (SPY/QQQ/IWM/DIA all
        # correlate ~0.95) at a time. This avoids stacking the same bet. It's
        # viable now that combo exits actually fill (marketable-exit fix): the
        # bot closes a position and rotates to the next rather than freezing.
        # Raise this to allow diversified premium across correlated names.
        self._max_correlated = max_correlated_positions
        self._corr_cache = TTLCache(default_ttl=3600, max_size=500)  # 1hr cache
        self._price_data: dict[str, pd.Series] = {}

    def update_price_data(self, symbol: str, prices: pd.Series) -> None:
        """Store price data for correlation calculation."""
        self._price_data[symbol] = prices

    def check_correlation(
        self, new_symbol: str, open_symbols: list[str]
    ) -> tuple[bool, str]:
        """Check if a new trade is too correlated with existing positions.

        Returns (allowed, reason).
        """
        if not open_symbols:
            return True, "no open positions"

        if new_symbol in open_symbols:
            # Same symbol — allow (duplicate check is in risk manager)
            return True, "same symbol handled by duplicate check"

        # Count how many existing positions are correlated above the threshold.
        # Block only once that count reaches the cap, so a few correlated
        # positions are allowed (diversified premium) but not unlimited stacking.
        correlated = [
            existing for existing in open_symbols
            if (c := self._get_correlation(new_symbol, existing)) is not None
            and abs(c) > self._max_corr
        ]
        if len(correlated) >= self._max_correlated:
            reason = (
                f"{new_symbol} correlated with {len(correlated)} open positions "
                f"{correlated} (cap {self._max_correlated})"
            )
            log.info("correlation_block", new=new_symbol,
                     correlated_count=len(correlated), cap=self._max_correlated)
            return False, reason

        return True, "correlation check passed"

    def get_portfolio_correlation_matrix(
        self, symbols: list[str]
    ) -> dict[str, dict[str, float]]:
        """Get correlation matrix for a list of symbols."""
        matrix: dict[str, dict[str, float]] = {}
        for s1 in symbols:
            matrix[s1] = {}
            for s2 in symbols:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                else:
                    corr = self._get_correlation(s1, s2)
                    matrix[s1][s2] = corr if corr is not None else 0.0
        return matrix

    def _get_correlation(self, sym1: str, sym2: str) -> float | None:
        """Get correlation between two symbols."""
        # Check cache
        cache_key = f"corr_{min(sym1, sym2)}_{max(sym1, sym2)}"
        cached = self._corr_cache.get(cache_key)
        if cached is not None:
            return cached

        # Try to calculate from price data
        if sym1 in self._price_data and sym2 in self._price_data:
            corr = self._calculate_correlation(
                self._price_data[sym1], self._price_data[sym2]
            )
            if corr is not None:
                self._corr_cache.set(cache_key, corr)
                return corr

        # Fallback: check sector groups
        corr = self._sector_correlation(sym1, sym2)
        self._corr_cache.set(cache_key, corr)
        return corr

    @staticmethod
    def _calculate_correlation(prices1: pd.Series, prices2: pd.Series) -> float | None:
        """Calculate rolling correlation from price series."""
        if len(prices1) < 20 or len(prices2) < 20:
            return None

        # Align on common index
        combined = pd.DataFrame({"a": prices1, "b": prices2}).dropna()
        if len(combined) < 20:
            return None

        # Use log returns for correlation
        returns = combined.pct_change().dropna()
        if len(returns) < 10:
            return None

        corr = float(returns["a"].corr(returns["b"]))
        return corr if not np.isnan(corr) else None

    @staticmethod
    def _sector_correlation(sym1: str, sym2: str) -> float:
        """Estimate correlation based on sector groupings."""
        for group_name, members in SECTOR_GROUPS.items():
            if sym1 in members and sym2 in members:
                if group_name == "index":
                    return 0.95  # Index ETFs are very correlated
                return 0.80  # Same sector

        # SPY/QQQ correlate with everything tech
        if sym1 in SECTOR_GROUPS["index"] or sym2 in SECTOR_GROUPS["index"]:
            other = sym2 if sym1 in SECTOR_GROUPS["index"] else sym1
            if other in SECTOR_GROUPS["mega_tech"] or other in SECTOR_GROUPS["semiconductors"]:
                return 0.70

        return 0.30  # Default: low correlation for unrelated symbols

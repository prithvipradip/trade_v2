"""Data quality validation — prevents trading on bad data.

Checks for:
- Stale quotes (older than threshold)
- Price outliers (sudden unrealistic jumps)
- Invalid bid-ask spreads
- Missing or zero prices
- Volume anomalies

Never trade on data you can't trust.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ait.utils.logging import get_logger

log = get_logger("data.quality")


@dataclass
class QuoteQuality:
    """Quality assessment of a market quote."""

    symbol: str
    is_valid: bool
    issues: list[str]
    staleness_seconds: float = 0.0
    spread_pct: float = 0.0


class DataQualityValidator:
    """Validates market data before it's used for trading decisions."""

    def __init__(
        self,
        max_staleness_seconds: float = 30.0,
        max_spread_pct: float = 0.15,
        max_price_jump_pct: float = 0.10,
    ) -> None:
        self._max_staleness = max_staleness_seconds
        self._max_spread_pct = max_spread_pct
        self._max_price_jump = max_price_jump_pct

        # Track last known good prices for outlier detection
        self._last_prices: dict[str, float] = {}
        self._last_timestamps: dict[str, float] = {}

    def validate_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        last: float,
        volume: int = 0,
        timestamp: float | None = None,
    ) -> QuoteQuality:
        """Validate a market quote. Returns quality assessment."""
        issues = []
        now = time.time()
        ts = timestamp or now

        # 1. Check staleness
        staleness = now - ts
        if staleness > self._max_staleness:
            issues.append(f"stale quote ({staleness:.0f}s old, max {self._max_staleness:.0f}s)")

        # 2. Check for zero/negative prices
        if last <= 0:
            issues.append("zero or negative last price")
        if bid < 0 or ask < 0:
            issues.append("negative bid or ask")

        # 3. Check bid-ask validity
        if bid > 0 and ask > 0:
            if bid > ask:
                issues.append(f"crossed market (bid {bid:.2f} > ask {ask:.2f})")

            mid = (bid + ask) / 2
            if mid > 0:
                spread_pct = (ask - bid) / mid
                if spread_pct > self._max_spread_pct:
                    issues.append(f"wide spread ({spread_pct:.1%} > {self._max_spread_pct:.0%})")
            else:
                spread_pct = 0.0
        else:
            spread_pct = 0.0

        # 4. Check for price outliers vs last known
        if symbol in self._last_prices and last > 0:
            prev = self._last_prices[symbol]
            if prev > 0:
                change_pct = abs(last - prev) / prev
                if change_pct > self._max_price_jump:
                    issues.append(
                        f"price jump ({change_pct:.1%} from ${prev:.2f} to ${last:.2f})"
                    )

        # 5. Check volume (warn only, don't invalidate)
        if volume == 0 and last > 0:
            log.debug("zero_volume", symbol=symbol)

        # Update tracking
        if last > 0:
            self._last_prices[symbol] = last
            self._last_timestamps[symbol] = ts

        is_valid = len(issues) == 0

        if not is_valid:
            log.warning("quote_quality_failed", symbol=symbol, issues=issues)

        return QuoteQuality(
            symbol=symbol,
            is_valid=is_valid,
            issues=issues,
            staleness_seconds=staleness,
            spread_pct=spread_pct,
        )

    def validate_option_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        last: float,
        open_interest: int,
        volume: int,
        delta: float | None = None,
    ) -> QuoteQuality:
        """Validate an options quote with options-specific checks."""
        quality = self.validate_quote(symbol, bid, ask, last, volume)

        # Options-specific checks
        if open_interest < 10:
            quality.issues.append(f"very low open interest ({open_interest})")
            quality.is_valid = False

        if delta is not None and abs(delta) > 1.0:
            quality.issues.append(f"invalid delta ({delta:.2f})")
            quality.is_valid = False

        if bid == 0 and ask > 0 and last > 0:
            quality.issues.append("zero bid (no market)")
            quality.is_valid = False

        return quality

    def validate_historical(
        self,
        symbol: str,
        prices: list[float],
        min_points: int = 20,
    ) -> bool:
        """Validate historical price data quality."""
        if len(prices) < min_points:
            log.warning("insufficient_history", symbol=symbol, points=len(prices), required=min_points)
            return False

        # Check for too many zeros or NaNs
        import math
        valid_prices = [p for p in prices if p > 0 and not math.isnan(p)]
        if len(valid_prices) < len(prices) * 0.9:
            log.warning("too_many_invalid_prices", symbol=symbol,
                       valid=len(valid_prices), total=len(prices))
            return False

        # Check for flat prices (stuck feed)
        unique_prices = set(valid_prices[-20:])
        if len(unique_prices) <= 1:
            log.warning("flat_price_data", symbol=symbol)
            return False

        return True

    def reset_tracking(self, symbol: str | None = None) -> None:
        """Reset price tracking for a symbol or all symbols."""
        if symbol:
            self._last_prices.pop(symbol, None)
            self._last_timestamps.pop(symbol, None)
        else:
            self._last_prices.clear()
            self._last_timestamps.clear()

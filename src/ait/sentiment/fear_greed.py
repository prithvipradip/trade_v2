"""Market fear/greed indicators from free sources.

Combines multiple signals into a fear/greed score:
- VIX level (primary fear gauge)
- Put/Call ratio (options market sentiment)
- Market breadth (advance/decline proxy from returns)
- Recent momentum (short-term trend)

All data from Yahoo Finance — no paid APIs needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ait.data.cache import TTLCache
from ait.data.market_data import MarketDataService
from ait.utils.logging import get_logger

log = get_logger("sentiment.fear_greed")


@dataclass
class FearGreedReading:
    """Market fear/greed composite reading."""

    score: float  # -1.0 (extreme fear) to +1.0 (extreme greed)
    vix_signal: float
    momentum_signal: float
    breadth_signal: float
    label: str  # "extreme_fear", "fear", "neutral", "greed", "extreme_greed"


class FearGreedIndicator:
    """Composite market fear/greed indicator."""

    def __init__(self, market_data: MarketDataService, cache_ttl: int = 300) -> None:
        self._market_data = market_data
        self._cache = TTLCache(default_ttl=cache_ttl)

    async def get_reading(self) -> FearGreedReading | None:
        """Get current fear/greed reading."""
        cached = self._cache.get("fear_greed")
        if cached:
            return cached

        signals = []

        # 1. VIX signal
        vix_signal = await self._vix_signal()
        if vix_signal is not None:
            signals.append(("vix", vix_signal, 0.40))  # 40% weight

        # 2. Market momentum (SPY)
        momentum = await self._momentum_signal()
        if momentum is not None:
            signals.append(("momentum", momentum, 0.35))  # 35% weight

        # 3. Market breadth proxy
        breadth = await self._breadth_signal()
        if breadth is not None:
            signals.append(("breadth", breadth, 0.25))  # 25% weight

        if not signals:
            return None

        # Weighted average
        total_weight = sum(w for _, _, w in signals)
        composite = sum(s * w for _, s, w in signals) / total_weight

        # Label
        if composite <= -0.5:
            label = "extreme_fear"
        elif composite <= -0.2:
            label = "fear"
        elif composite <= 0.2:
            label = "neutral"
        elif composite <= 0.5:
            label = "greed"
        else:
            label = "extreme_greed"

        reading = FearGreedReading(
            score=composite,
            vix_signal=next((s for n, s, _ in signals if n == "vix"), 0.0),
            momentum_signal=next((s for n, s, _ in signals if n == "momentum"), 0.0),
            breadth_signal=next((s for n, s, _ in signals if n == "breadth"), 0.0),
            label=label,
        )

        self._cache.set("fear_greed", reading)

        log.info(
            "fear_greed_reading",
            score=f"{composite:.2f}",
            label=label,
            vix_signal=reading.vix_signal,
        )
        return reading

    async def _vix_signal(self) -> float | None:
        """VIX-based fear/greed signal.

        VIX < 15: greed (+0.5 to +1.0)
        VIX 15-20: neutral (0.0 to +0.5)
        VIX 20-30: fear (-0.5 to 0.0)
        VIX > 30: extreme fear (-1.0 to -0.5)
        """
        vix = await self._market_data.get_vix()
        if vix is None:
            return None

        if vix < 12:
            return 1.0
        elif vix < 15:
            return 0.5 + (15 - vix) / 6  # 0.5 to 1.0
        elif vix < 20:
            return (20 - vix) / 10  # 0.0 to 0.5
        elif vix < 30:
            return -(vix - 20) / 20  # -0.5 to 0.0
        elif vix < 40:
            return -0.5 - (vix - 30) / 20  # -0.5 to -1.0
        else:
            return -1.0

    async def _momentum_signal(self) -> float | None:
        """SPY momentum-based signal."""
        df = await self._market_data.get_historical("SPY", days=30)
        if df is None or len(df) < 20:
            return None

        close = df["Close"]
        ret_5 = float(close.pct_change(5).iloc[-1])
        ret_20 = float(close.pct_change(20).iloc[-1])

        # Blend short and medium momentum
        signal = (ret_5 * 0.6 + ret_20 * 0.4) * 10  # Scale up
        return float(np.clip(signal, -1.0, 1.0))

    async def _breadth_signal(self) -> float | None:
        """Market breadth proxy — how many symbols are positive.

        Uses a small basket of major stocks as a proxy.
        """
        breadth_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA"]
        positive = 0
        total = 0

        for sym in breadth_symbols:
            df = await self._market_data.get_historical(sym, days=10)
            if df is not None and len(df) >= 5:
                ret = float(df["Close"].pct_change(5).iloc[-1])
                if ret > 0:
                    positive += 1
                total += 1

        if total == 0:
            return None

        # Scale from [0, 1] to [-1, 1]
        return (positive / total) * 2 - 1

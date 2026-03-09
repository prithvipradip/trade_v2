"""Market regime detection.

Classifies the current market into regimes:
- TRENDING_UP: Strong bullish trend
- TRENDING_DOWN: Strong bearish trend
- RANGE_BOUND: Choppy/sideways market
- HIGH_VOLATILITY: Crisis/fear mode
- LOW_VOLATILITY: Complacent/calm market

Each regime influences which strategies the bot considers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ait.utils.logging import get_logger

log = get_logger("ml.regime")


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeAnalysis:
    """Result of regime detection."""

    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    trend_strength: float  # -1.0 (strong down) to +1.0 (strong up)
    volatility_level: float  # Annualized realized volatility
    vix_level: float | None
    details: dict[str, float]


class RegimeDetector:
    """Detects the current market regime from price data and VIX."""

    def analyze(
        self,
        df: pd.DataFrame,
        vix: float | None = None,
    ) -> RegimeAnalysis:
        """Analyze market regime from recent price data.

        Args:
            df: OHLCV DataFrame with at least 60 rows.
            vix: Current VIX level (optional, improves accuracy).
        """
        if df is None or len(df) < 60:
            return RegimeAnalysis(
                regime=MarketRegime.RANGE_BOUND,
                confidence=0.0,
                trend_strength=0.0,
                volatility_level=0.0,
                vix_level=vix,
                details={},
            )

        close = df["Close"]

        # --- Trend Analysis ---
        trend_strength = self._measure_trend(close)

        # --- Volatility Analysis ---
        vol = self._measure_volatility(close)

        # --- Regime Classification ---
        regime, confidence = self._classify(trend_strength, vol, vix)

        details = {
            "trend_strength": trend_strength,
            "realized_vol_20d": vol,
            "sma_20_slope": float(close.rolling(20).mean().pct_change(5).iloc[-1]),
            "rsi_14": float(self._rsi(close, 14).iloc[-1]),
            "price_vs_sma_50": float((close.iloc[-1] - close.rolling(50).mean().iloc[-1]) / close.rolling(50).mean().iloc[-1]),
        }

        log.info(
            "regime_detected",
            regime=regime.value,
            confidence=f"{confidence:.2f}",
            trend=f"{trend_strength:.2f}",
            vol=f"{vol:.2f}",
            vix=vix,
        )

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=vol,
            vix_level=vix,
            details=details,
        )

    def _measure_trend(self, close: pd.Series) -> float:
        """Measure trend strength from -1.0 to +1.0.

        Uses multiple timeframe MA slopes and price position.
        """
        scores = []

        # SMA slopes (normalized)
        for period in [10, 20, 50]:
            sma = close.rolling(period).mean()
            slope = sma.pct_change(5).iloc[-1]
            # Clamp to [-1, 1]
            scores.append(max(-1, min(1, slope * 100)))

        # Price vs SMAs
        for period in [20, 50]:
            sma_val = close.rolling(period).mean().iloc[-1]
            deviation = (close.iloc[-1] - sma_val) / sma_val
            scores.append(max(-1, min(1, deviation * 10)))

        # Recent momentum (20-day return)
        ret_20 = close.pct_change(20).iloc[-1]
        scores.append(max(-1, min(1, ret_20 * 5)))

        return float(np.mean(scores))

    def _measure_volatility(self, close: pd.Series) -> float:
        """Measure annualized realized volatility (20-day)."""
        log_returns = np.log(close / close.shift(1)).dropna()
        return float(log_returns.tail(20).std() * np.sqrt(252))

    def _classify(
        self,
        trend: float,
        vol: float,
        vix: float | None,
    ) -> tuple[MarketRegime, float]:
        """Classify regime based on trend and volatility."""

        # High volatility override (VIX > 30 or realized vol > 40%)
        if (vix and vix > 30) or vol > 0.40:
            return MarketRegime.HIGH_VOLATILITY, min(0.9, vol)

        # Low volatility (VIX < 15 or realized vol < 12%)
        if (vix and vix < 15) or vol < 0.12:
            # Still check for trend within low vol
            if abs(trend) > 0.3:
                if trend > 0:
                    return MarketRegime.TRENDING_UP, abs(trend)
                else:
                    return MarketRegime.TRENDING_DOWN, abs(trend)
            return MarketRegime.LOW_VOLATILITY, 0.7

        # Strong trend
        if trend > 0.35:
            return MarketRegime.TRENDING_UP, min(0.95, abs(trend))
        elif trend < -0.35:
            return MarketRegime.TRENDING_DOWN, min(0.95, abs(trend))

        # Range-bound
        return MarketRegime.RANGE_BOUND, max(0.5, 1.0 - abs(trend) * 2)

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

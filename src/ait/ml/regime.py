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

from ait.config.settings import RegimeConfig
from ait.utils.logging import get_logger

log = get_logger("ml.regime")


class MarketRegime(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    IV_CRUSH = "iv_crush"  # Post-catalyst vol compression
    MEAN_REVERTING = "mean_reverting"  # Choppy with autocorrelation < 0


@dataclass
class RegimeAnalysis:
    """Result of regime detection."""

    regime: MarketRegime
    confidence: float  # 0.0 to 1.0
    trend_strength: float  # -1.0 (strong down) to +1.0 (strong up)
    volatility_level: float  # Annualized realized volatility
    vix_level: float | None
    iv_rank: float  # 0.0 to 1.0, current vol vs 252-day range
    hurst_exponent: float  # < 0.5 = mean-reverting, > 0.5 = trending
    details: dict[str, float]


def regime_forces_neutral(analysis: RegimeAnalysis, threshold: float) -> bool:
    """W93: True when the live entry rule must force the direction to NEUTRAL: the market is classed HIGH_VOLATILITY with a
    confidence AT OR ABOVE ``threshold`` (``ml.regime_override_confidence``).

    It was a strict ``>`` inlined in the orchestrator. The classifier's high-vol confidence floor is exactly the shipped
    threshold (0.80), so a day triggered by VIX alone with realized vol at or under 40% scored 0.80 and ``0.80 > 0.80`` never
    fired the override, although the R5 audit made the floor precisely so that any VIX / vol trigger would clear the gate. ``>=``
    is the behaviour the floor was built for (operator decision, 2026-10-23). Iron condors ignore the direction, so this only
    matters for direction-sensitive strategies."""
    return analysis.regime == MarketRegime.HIGH_VOLATILITY and analysis.confidence >= threshold


class RegimeDetector:
    """Detects the current market regime from price data and VIX."""

    def __init__(self, config: RegimeConfig | None = None) -> None:
        # W69: the classification thresholds are config (`regime:`); None ->
        # the model defaults, which are the pre-W69 literals.
        self._cfg = config or RegimeConfig()

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
                iv_rank=0.5,
                hurst_exponent=0.5,
                details={},
            )

        close = df["Close"]

        # --- Trend Analysis ---
        trend_strength = self._measure_trend(close)

        # --- Volatility Analysis ---
        vol = self._measure_volatility(close)

        # --- IV Rank (vol percentile over 252 days) ---
        iv_rank = self._measure_iv_rank(close)

        # --- Hurst Exponent (mean-reversion vs trending) ---
        hurst = self._estimate_hurst(close)

        # --- Regime Classification ---
        regime, confidence = self._classify(trend_strength, vol, vix, iv_rank, hurst)

        details = {
            "trend_strength": trend_strength,
            "realized_vol_20d": vol,
            "iv_rank": iv_rank,
            "hurst_exponent": hurst,
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
            iv_rank=f"{iv_rank:.2f}",
            hurst=f"{hurst:.2f}",
            vix=vix,
        )

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=vol,
            vix_level=vix,
            iv_rank=iv_rank,
            hurst_exponent=hurst,
            details=details,
        )

    def _measure_trend(self, close: pd.Series) -> float:
        """Measure trend strength from -1.0 to +1.0.

        Uses multiple timeframe MA slopes and price position.
        """
        c = self._cfg
        scores = []

        # SMA slopes (normalized). W92: the periods, look-back and scale are regime.trend_* (were 10/20/50, 5, 100)
        for period in c.trend_sma_periods:
            sma = close.rolling(period).mean()
            slope = sma.pct_change(c.trend_slope_lookback).iloc[-1]
            # Clamp to [-1, 1]
            scores.append(max(-1, min(1, slope * c.trend_slope_scale)))

        # Price vs SMAs (was 20/50, scale 10)
        for period in c.trend_deviation_periods:
            sma_val = close.rolling(period).mean().iloc[-1]
            deviation = (close.iloc[-1] - sma_val) / sma_val
            scores.append(max(-1, min(1, deviation * c.trend_deviation_scale)))

        # Recent momentum (was the 20-day return, scale 5)
        ret_n = close.pct_change(c.trend_momentum_days).iloc[-1]
        scores.append(max(-1, min(1, ret_n * c.trend_momentum_scale)))

        return float(np.mean(scores))

    def _measure_volatility(self, close: pd.Series) -> float:
        """Measure annualized realized volatility (20-day)."""
        log_returns = np.log(close / close.shift(1)).dropna()
        return float(log_returns.tail(20).std() * np.sqrt(252))

    def _measure_iv_rank(self, close: pd.Series) -> float:
        """IV rank proxy: current 20-day vol percentile over 252 days."""
        log_returns = np.log(close / close.shift(1)).dropna()
        vol_20 = log_returns.rolling(20).std() * np.sqrt(252)
        vol_20 = vol_20.dropna()

        if len(vol_20) < 60:
            return 0.5

        current = float(vol_20.iloc[-1])
        lookback = vol_20.tail(min(252, len(vol_20)))
        vol_min = float(lookback.min())
        vol_max = float(lookback.max())
        vol_range = vol_max - vol_min

        if vol_range <= 0:
            return 0.5

        return max(0.0, min(1.0, (current - vol_min) / vol_range))

    def _estimate_hurst(self, close: pd.Series, max_lag: int = 20) -> float:
        """Estimate Hurst exponent via rescaled range (R/S) method.

        H < 0.5 → mean-reverting (good for iron condors, range strategies)
        H = 0.5 → random walk
        H > 0.5 → trending (good for directional strategies)
        """
        try:
            log_returns = np.log(close / close.shift(1)).dropna().values
            if len(log_returns) < max_lag * 2:
                return 0.5

            lags = range(2, max_lag + 1)
            rs_values = []

            for lag in lags:
                rs_list = []
                for start in range(0, len(log_returns) - lag, lag):
                    chunk = log_returns[start:start + lag]
                    mean_adj = chunk - chunk.mean()
                    cumdev = np.cumsum(mean_adj)
                    r = cumdev.max() - cumdev.min()
                    s = chunk.std()
                    if s > 0:
                        rs_list.append(r / s)
                if rs_list:
                    rs_values.append((np.log(lag), np.log(np.mean(rs_list))))

            if len(rs_values) < 3:
                return 0.5

            x = np.array([v[0] for v in rs_values])
            y = np.array([v[1] for v in rs_values])
            hurst = float(np.polyfit(x, y, 1)[0])
            return max(0.0, min(1.0, hurst))
        except Exception:
            return 0.5

    def _classify(
        self,
        trend: float,
        vol: float,
        vix: float | None,
        iv_rank: float = 0.5,
        hurst: float = 0.5,
    ) -> tuple[MarketRegime, float]:
        """Classify regime based on trend, volatility, IV rank, and Hurst exponent."""

        c = self._cfg
        # High volatility override (VIX > 30 or realized vol > 40% by default)
        if (vix and vix > c.high_vol_vix) or vol > c.high_vol_realized:
            # R5 audit HIGH: confidence used to be the raw vol fraction
            # (VIX-35 crash with 25% realized vol -> 0.25), but the live
            # consumers gate defensive action at >0.70/0.80 — the crash
            # override only woke at ~2008-level realized vol. Scale so any
            # classification at the trigger threshold clears the gates:
            # vol 0.40 -> 0.80; VIX-triggered entries floor at 0.80.
            conf = max(c.high_vol_confidence_floor,
                       min(c.high_vol_confidence_cap, vol / c.high_vol_confidence_vol_scale))      # W92: was 0.80 / 0.9 / 0.50
            return MarketRegime.HIGH_VOLATILITY, conf

        # IV Crush: vol was recently high but is rapidly compressing
        # (iv_rank dropping from high, vol trend negative)
        if (iv_rank < c.iv_crush_iv_rank_max and vol < c.iv_crush_vol_max
                and abs(trend) < c.iv_crush_trend_max):
            return MarketRegime.IV_CRUSH, c.iv_crush_confidence

        # Mean-reverting: Hurst < 0.4 with no strong trend
        if hurst < c.mean_revert_hurst_max and abs(trend) < c.mean_revert_trend_max:
            return MarketRegime.MEAN_REVERTING, max(c.mean_revert_confidence_floor, 1.0 - hurst)

        # Low volatility (VIX < 15 or realized vol < 12%)
        if (vix and vix < c.low_vol_vix) or vol < c.low_vol_realized:
            if abs(trend) > c.low_vol_trend_breakout:
                if trend > 0:
                    return MarketRegime.TRENDING_UP, abs(trend)
                else:
                    return MarketRegime.TRENDING_DOWN, abs(trend)
            return MarketRegime.LOW_VOLATILITY, c.low_vol_confidence

        # Strong trend
        if trend > c.strong_trend:
            return MarketRegime.TRENDING_UP, min(c.trend_confidence_cap, abs(trend))
        elif trend < -c.strong_trend:
            return MarketRegime.TRENDING_DOWN, min(c.trend_confidence_cap, abs(trend))

        # Range-bound
        return MarketRegime.RANGE_BOUND, max(c.range_confidence_floor, 1.0 - abs(trend) * 2)

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

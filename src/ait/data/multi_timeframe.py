"""Multi-timeframe analysis — combines 5-min, daily, and weekly signals.

Three timeframes, each serving a different purpose:
- Weekly (5-day resampled): Trend direction and major support/resistance
- Daily: Signal generation (existing ML pipeline)
- Intraday (5-min): Entry timing and momentum confirmation

The key insight: when all three timeframes agree, the probability
of a successful trade is significantly higher.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ait.utils.logging import get_logger

log = get_logger("data.multi_timeframe")


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""

    timeframe: str  # "weekly", "daily", "intraday"
    trend: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 to 1.0
    momentum: float  # Rate of change
    rsi: float  # RSI value
    volume_confirms: bool  # Volume supports the move


@dataclass
class MultiTimeframeAnalysis:
    """Combined analysis across all timeframes."""

    weekly: TimeframeSignal
    daily: TimeframeSignal
    intraday: TimeframeSignal | None  # None if no intraday data

    @property
    def alignment(self) -> float:
        """How aligned are the timeframes? 1.0 = all agree, 0.0 = all disagree."""
        signals = [self.weekly, self.daily]
        if self.intraday:
            signals.append(self.intraday)

        # Count agreement
        trends = [s.trend for s in signals]
        if len(set(trends)) == 1:
            return 1.0  # Perfect alignment
        elif "neutral" in trends:
            # Neutral counts as partial agreement
            non_neutral = [t for t in trends if t != "neutral"]
            if len(set(non_neutral)) <= 1:
                return 0.7
        elif len(set(trends)) == 2:
            return 0.3  # Two different trends
        return 0.0  # All different

    @property
    def dominant_trend(self) -> str:
        """The trend that most timeframes agree on, weighted by timeframe importance."""
        weights = {"weekly": 0.4, "daily": 0.4, "intraday": 0.2}
        scores = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}

        for signal in [self.weekly, self.daily, self.intraday]:
            if signal is None:
                continue
            w = weights.get(signal.timeframe, 0.2)
            scores[signal.trend] += w * signal.strength

        return max(scores, key=scores.get)

    @property
    def confidence_boost(self) -> float:
        """How much to boost ML confidence based on timeframe alignment.

        Returns a value between -0.15 and +0.15.
        - All aligned with volume: +0.15
        - Partial alignment: +0.05
        - Disagreement: -0.10
        """
        if self.alignment >= 0.9:
            # Full alignment — check volume confirmation
            vol_confirms = sum(1 for s in [self.weekly, self.daily, self.intraday]
                             if s and s.volume_confirms)
            if vol_confirms >= 2:
                return 0.15
            return 0.10
        elif self.alignment >= 0.6:
            return 0.05
        elif self.alignment >= 0.3:
            return 0.0
        else:
            return -0.10

    @property
    def entry_timing_score(self) -> float:
        """Score for entry timing based on intraday data.

        High score = good time to enter. Uses intraday RSI and momentum.
        Returns 0.0 to 1.0.
        """
        if self.intraday is None:
            return 0.5  # Neutral when no intraday data

        signal = self.intraday

        # For bullish entries, we want intraday pullback (lower RSI = better entry)
        if self.dominant_trend == "bullish":
            # RSI 30-45 = great entry, 45-55 = ok, >55 = chasing
            if signal.rsi < 35:
                rsi_score = 1.0
            elif signal.rsi < 50:
                rsi_score = 0.7
            elif signal.rsi < 60:
                rsi_score = 0.4
            else:
                rsi_score = 0.1

        elif self.dominant_trend == "bearish":
            # For bearish entries, want intraday bounce (higher RSI = better entry)
            if signal.rsi > 65:
                rsi_score = 1.0
            elif signal.rsi > 50:
                rsi_score = 0.7
            elif signal.rsi > 40:
                rsi_score = 0.4
            else:
                rsi_score = 0.1
        else:
            rsi_score = 0.5

        # Volume confirmation adds to score
        vol_bonus = 0.2 if signal.volume_confirms else 0.0

        return min(1.0, rsi_score + vol_bonus)


class MultiTimeframeAnalyzer:
    """Analyzes price data across multiple timeframes.

    Accepts daily and optionally intraday DataFrames,
    produces a unified multi-timeframe analysis.
    """

    def analyze(
        self,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame | None = None,
    ) -> MultiTimeframeAnalysis:
        """Run multi-timeframe analysis.

        Args:
            daily_df: OHLCV DataFrame with daily bars (≥50 rows)
            intraday_df: Optional OHLCV DataFrame with 5-min bars (recent day)
        """
        weekly = self._analyze_weekly(daily_df)
        daily = self._analyze_daily(daily_df)

        intraday = None
        if intraday_df is not None and len(intraday_df) >= 20:
            intraday = self._analyze_intraday(intraday_df)

        return MultiTimeframeAnalysis(
            weekly=weekly,
            daily=daily,
            intraday=intraday,
        )

    def _analyze_weekly(self, df: pd.DataFrame) -> TimeframeSignal:
        """Analyze weekly trend from daily data (5-day resampling)."""
        close = df["Close"]
        volume = df["Volume"]

        # Weekly SMA (20-day ≈ 4 weeks)
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()

        # Trend direction
        price_vs_sma = float((close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1])
        sma_slope = float((sma_20.iloc[-1] - sma_20.iloc[-5]) / sma_20.iloc[-5]) if len(sma_20) >= 5 else 0

        if price_vs_sma > 0.01 and sma_slope > 0:
            trend = "bullish"
        elif price_vs_sma < -0.01 and sma_slope < 0:
            trend = "bearish"
        else:
            trend = "neutral"

        # Strength based on distance from moving average
        strength = min(1.0, abs(price_vs_sma) * 10)

        # Weekly RSI (using 5-day smoothed)
        weekly_close = close.rolling(5).mean().dropna()
        rsi = self._rsi(weekly_close, 14)

        # Volume: is weekly volume above average?
        weekly_vol = volume.rolling(5).sum()
        avg_weekly_vol = weekly_vol.rolling(4).mean()
        vol_confirms = bool(weekly_vol.iloc[-1] > avg_weekly_vol.iloc[-1]) if len(avg_weekly_vol) > 0 else False

        # Momentum (25-day ROC ≈ 5-week)
        momentum = float(close.pct_change(25).iloc[-1]) if len(close) > 25 else 0.0

        return TimeframeSignal(
            timeframe="weekly",
            trend=trend,
            strength=strength,
            momentum=momentum,
            rsi=rsi,
            volume_confirms=vol_confirms,
        )

    def _analyze_daily(self, df: pd.DataFrame) -> TimeframeSignal:
        """Analyze daily trend."""
        close = df["Close"]
        volume = df["Volume"]

        sma_10 = close.rolling(10).mean()
        sma_20 = close.rolling(20).mean()

        # Trend from MA crossover and price position
        above_sma10 = close.iloc[-1] > sma_10.iloc[-1]
        above_sma20 = close.iloc[-1] > sma_20.iloc[-1]
        sma_cross = sma_10.iloc[-1] > sma_20.iloc[-1]

        if above_sma10 and above_sma20 and sma_cross:
            trend = "bullish"
        elif not above_sma10 and not above_sma20 and not sma_cross:
            trend = "bearish"
        else:
            trend = "neutral"

        # Strength
        price_vs_sma = float((close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1])
        strength = min(1.0, abs(price_vs_sma) * 15)

        # RSI-14
        rsi = self._rsi(close, 14)

        # Volume confirmation
        vol_sma = volume.rolling(20).mean()
        vol_confirms = bool(volume.iloc[-1] > vol_sma.iloc[-1])

        # 5-day momentum
        momentum = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0

        return TimeframeSignal(
            timeframe="daily",
            trend=trend,
            strength=strength,
            momentum=momentum,
            rsi=rsi,
            volume_confirms=vol_confirms,
        )

    def _analyze_intraday(self, df: pd.DataFrame) -> TimeframeSignal:
        """Analyze intraday trend from 5-min bars."""
        close = df["Close"]
        volume = df["Volume"]

        # VWAP proxy (cumulative volume-weighted price)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        cum_vol = volume.cumsum()
        vwap = (typical_price * volume).cumsum() / cum_vol.replace(0, np.nan)

        # Trend: price vs VWAP
        price_vs_vwap = float((close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1])

        # Short-term EMA
        ema_9 = close.ewm(span=9, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()
        ema_cross = ema_9.iloc[-1] > ema_21.iloc[-1]

        if price_vs_vwap > 0.001 and ema_cross:
            trend = "bullish"
        elif price_vs_vwap < -0.001 and not ema_cross:
            trend = "bearish"
        else:
            trend = "neutral"

        strength = min(1.0, abs(price_vs_vwap) * 50)

        # Intraday RSI (faster, 7-period on 5-min)
        rsi = self._rsi(close, 7)

        # Volume: current bar vs intraday average
        vol_avg = volume.mean()
        vol_confirms = bool(volume.iloc[-1] > vol_avg) if vol_avg > 0 else False

        # Short-term momentum (last 12 bars = 1 hour)
        momentum = float(close.pct_change(12).iloc[-1]) if len(close) > 12 else 0.0

        return TimeframeSignal(
            timeframe="intraday",
            trend=trend,
            strength=strength,
            momentum=momentum,
            rsi=rsi,
            volume_confirms=vol_confirms,
        )

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> float:
        """Calculate current RSI value."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if not np.isnan(val) else 50.0

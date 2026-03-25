"""Tests for multi-timeframe analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ait.data.multi_timeframe import (
    MultiTimeframeAnalysis,
    MultiTimeframeAnalyzer,
    TimeframeSignal,
)
from ait.ml.features import FeatureEngine


def _make_daily(days: int = 100, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic daily OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=days, freq="B")

    if trend == "up":
        base_return = 0.001
    elif trend == "down":
        base_return = -0.001
    else:
        base_return = 0.0

    returns = np.random.normal(base_return, 0.012, days)
    close = 100.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.005, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, days)))
    open_ = close * (1 + np.random.normal(0, 0.002, days))
    volume = np.random.randint(500_000, 5_000_000, days)

    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


def _make_intraday(bars: int = 78, trend: str = "up") -> pd.DataFrame:
    """Generate synthetic 5-min intraday OHLCV data (1 day ≈ 78 bars)."""
    np.random.seed(42)
    idx = pd.date_range("2024-06-01 09:30", periods=bars, freq="5min")

    if trend == "up":
        drift = 0.0001
    elif trend == "down":
        drift = -0.0001
    else:
        drift = 0.0

    returns = np.random.normal(drift, 0.001, bars)
    close = 450.0 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.001, bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.001, bars)))
    open_ = close * (1 + np.random.normal(0, 0.0005, bars))
    volume = np.random.randint(10_000, 500_000, bars)

    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=idx)


class TestTimeframeSignal:

    def test_creation(self) -> None:
        sig = TimeframeSignal(
            timeframe="daily", trend="bullish",
            strength=0.7, momentum=0.02, rsi=55.0,
            volume_confirms=True,
        )
        assert sig.timeframe == "daily"
        assert sig.trend == "bullish"


class TestMultiTimeframeAnalysis:

    def test_full_alignment(self) -> None:
        weekly = TimeframeSignal("weekly", "bullish", 0.8, 0.02, 55, True)
        daily = TimeframeSignal("daily", "bullish", 0.7, 0.01, 60, True)
        intraday = TimeframeSignal("intraday", "bullish", 0.6, 0.005, 50, False)

        analysis = MultiTimeframeAnalysis(weekly, daily, intraday)
        assert analysis.alignment == 1.0
        assert analysis.dominant_trend == "bullish"
        assert analysis.confidence_boost > 0

    def test_partial_alignment_with_neutral(self) -> None:
        weekly = TimeframeSignal("weekly", "bullish", 0.8, 0.02, 55, True)
        daily = TimeframeSignal("daily", "neutral", 0.3, 0.0, 50, False)
        intraday = TimeframeSignal("intraday", "bullish", 0.6, 0.005, 50, True)

        analysis = MultiTimeframeAnalysis(weekly, daily, intraday)
        assert analysis.alignment == 0.7  # Neutral counts as partial

    def test_disagreement(self) -> None:
        weekly = TimeframeSignal("weekly", "bullish", 0.8, 0.02, 55, True)
        daily = TimeframeSignal("daily", "bearish", 0.7, -0.01, 40, False)

        analysis = MultiTimeframeAnalysis(weekly, daily, None)
        assert analysis.alignment <= 0.3
        assert analysis.confidence_boost <= 0

    def test_no_intraday(self) -> None:
        weekly = TimeframeSignal("weekly", "bullish", 0.8, 0.02, 55, True)
        daily = TimeframeSignal("daily", "bullish", 0.7, 0.01, 60, True)

        analysis = MultiTimeframeAnalysis(weekly, daily, None)
        assert analysis.alignment == 1.0
        assert analysis.entry_timing_score == 0.5  # Default when no intraday

    def test_entry_timing_bullish_pullback(self) -> None:
        weekly = TimeframeSignal("weekly", "bullish", 0.8, 0.02, 55, True)
        daily = TimeframeSignal("daily", "bullish", 0.7, 0.01, 60, True)
        intraday = TimeframeSignal("intraday", "bullish", 0.5, 0.001, 32, True)

        analysis = MultiTimeframeAnalysis(weekly, daily, intraday)
        # RSI 32 = great pullback entry for bullish
        assert analysis.entry_timing_score >= 0.9

    def test_entry_timing_chasing(self) -> None:
        weekly = TimeframeSignal("weekly", "bullish", 0.8, 0.02, 55, True)
        daily = TimeframeSignal("daily", "bullish", 0.7, 0.01, 60, True)
        intraday = TimeframeSignal("intraday", "bullish", 0.5, 0.003, 72, False)

        analysis = MultiTimeframeAnalysis(weekly, daily, intraday)
        # RSI 72 = chasing, bad entry
        assert analysis.entry_timing_score < 0.3

    def test_confidence_boost_with_volume(self) -> None:
        weekly = TimeframeSignal("weekly", "bearish", 0.8, -0.02, 35, True)
        daily = TimeframeSignal("daily", "bearish", 0.7, -0.01, 30, True)
        intraday = TimeframeSignal("intraday", "bearish", 0.6, -0.005, 65, True)

        analysis = MultiTimeframeAnalysis(weekly, daily, intraday)
        # All aligned + volume confirms on ≥2 timeframes
        assert analysis.confidence_boost == 0.15


class TestMultiTimeframeAnalyzer:

    def test_analyze_daily_only(self) -> None:
        daily = _make_daily(100, trend="up")
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(daily)

        assert result.weekly is not None
        assert result.daily is not None
        assert result.intraday is None

    def test_analyze_with_intraday(self) -> None:
        daily = _make_daily(100, trend="up")
        intraday = _make_intraday(78, trend="up")
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(daily, intraday)

        assert result.intraday is not None
        assert result.intraday.timeframe == "intraday"

    def test_analyze_downtrend(self) -> None:
        daily = _make_daily(100, trend="down")
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(daily)

        # With strong downtrend data, weekly should detect it
        assert result.weekly.trend in ("bearish", "neutral")

    def test_rsi_in_valid_range(self) -> None:
        daily = _make_daily(100)
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(daily)

        assert 0 <= result.weekly.rsi <= 100
        assert 0 <= result.daily.rsi <= 100

    def test_short_intraday_ignored(self) -> None:
        daily = _make_daily(100)
        short_intraday = _make_intraday(5)  # Too few bars
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(daily, short_intraday)

        assert result.intraday is None

    def test_strength_bounded(self) -> None:
        daily = _make_daily(100)
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(daily)

        assert 0 <= result.weekly.strength <= 1.0
        assert 0 <= result.daily.strength <= 1.0


class TestIntradayFeatures:

    def test_compute_intraday_features(self) -> None:
        intraday = _make_intraday(78)
        engine = FeatureEngine()
        features = engine.compute_intraday_features(intraday)

        assert "intraday_vwap_position" in features
        assert "intraday_rsi" in features
        assert "intraday_momentum_1h" in features
        assert "intraday_atr_pct" in features
        assert "intraday_vol_ratio" in features
        assert "intraday_range_compression" in features

    def test_intraday_rsi_in_range(self) -> None:
        intraday = _make_intraday(78)
        engine = FeatureEngine()
        features = engine.compute_intraday_features(intraday)

        assert 0 <= features["intraday_rsi"] <= 100

    def test_empty_intraday(self) -> None:
        engine = FeatureEngine()
        assert engine.compute_intraday_features(None) == {}
        assert engine.compute_intraday_features(pd.DataFrame()) == {}

    def test_short_intraday(self) -> None:
        short = _make_intraday(10)  # Too few bars
        engine = FeatureEngine()
        assert engine.compute_intraday_features(short) == {}

    def test_vol_ratio_positive(self) -> None:
        intraday = _make_intraday(78)
        engine = FeatureEngine()
        features = engine.compute_intraday_features(intraday)
        assert features["intraday_vol_ratio"] > 0

    def test_range_compression_bounded(self) -> None:
        intraday = _make_intraday(78)
        engine = FeatureEngine()
        features = engine.compute_intraday_features(intraday)
        assert 0 <= features["intraday_range_compression"] <= 1.5

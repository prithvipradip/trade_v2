"""Feature engineering for ML models.

Computes technical indicators, volatility metrics, and market microstructure
features from price data. All features are calculated from real data only.

Features are organized into groups:
- Momentum (RSI, MACD, rate of change)
- Volatility (ATR, Bollinger Bands, realized vol)
- Volume (OBV, volume ratio, VWAP proxy)
- Trend (moving averages, ADX proxy)
- Options-specific (IV rank, put/call ratio — when available)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ait.utils.logging import get_logger

log = get_logger("ml.features")


class FeatureEngine:
    """Computes features from OHLCV data for ML models."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features from OHLCV DataFrame.

        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume]
                indexed by date.

        Returns:
            DataFrame with original columns plus computed features.
            Rows with NaN (from lookback periods) are dropped.
        """
        if df is None or len(df) < 50:
            log.warning("insufficient_data_for_features", rows=len(df) if df is not None else 0)
            return pd.DataFrame()

        features = df.copy()

        # --- Momentum Features ---
        features = self._add_momentum(features)

        # --- Volatility Features ---
        features = self._add_volatility(features)

        # --- Volume Features ---
        features = self._add_volume(features)

        # --- Trend Features ---
        features = self._add_trend(features)

        # --- Price Action Features ---
        features = self._add_price_action(features)

        # Drop rows with NaN from lookback calculations
        features = features.dropna()

        log.debug("features_computed", rows=len(features), columns=len(features.columns))
        return features

    def get_feature_names(self) -> list[str]:
        """Get list of all feature column names (excluding OHLCV)."""
        return [
            # Momentum
            "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist",
            "roc_5", "roc_10", "roc_20",
            # Volatility
            "atr_14", "atr_pct", "bb_upper", "bb_lower", "bb_width",
            "bb_position", "realized_vol_20", "realized_vol_10",
            "high_low_range",
            # Volume
            "volume_sma_20_ratio", "obv_change", "volume_trend",
            # Trend
            "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
            "sma_10_slope", "sma_20_slope",
            "price_vs_sma_20", "price_vs_sma_50",
            "sma_10_20_cross",
            # Price action
            "daily_return", "gap", "body_size", "upper_wick", "lower_wick",
            "consecutive_up", "consecutive_down",
        ]

    # --- Feature Groups ---

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]

        # RSI
        df["rsi_14"] = self._rsi(close, 14)
        df["rsi_7"] = self._rsi(close, 7)

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Rate of change
        df["roc_5"] = close.pct_change(5)
        df["roc_10"] = close.pct_change(10)
        df["roc_20"] = close.pct_change(20)

        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(14).mean()
        df["atr_pct"] = df["atr_14"] / close  # ATR as % of price

        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
        df["bb_position"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Realized volatility
        log_returns = np.log(close / close.shift(1))
        df["realized_vol_20"] = log_returns.rolling(20).std() * np.sqrt(252)
        df["realized_vol_10"] = log_returns.rolling(10).std() * np.sqrt(252)

        # High-low range
        df["high_low_range"] = (high - low) / close

        return df

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        volume = df["Volume"]
        close = df["Close"]

        # Volume relative to 20-day average
        vol_sma20 = volume.rolling(20).mean()
        df["volume_sma_20_ratio"] = volume / vol_sma20.replace(0, np.nan)

        # OBV (On-Balance Volume) change
        obv = (np.sign(close.diff()) * volume).cumsum()
        df["obv_change"] = obv.pct_change(5)

        # Volume trend (5-day slope)
        df["volume_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()

        return df

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]

        # Moving averages
        df["sma_10"] = close.rolling(10).mean()
        df["sma_20"] = close.rolling(20).mean()
        df["sma_50"] = close.rolling(50).mean()
        df["ema_12"] = close.ewm(span=12, adjust=False).mean()
        df["ema_26"] = close.ewm(span=26, adjust=False).mean()

        # MA slopes (rate of change of the average)
        df["sma_10_slope"] = df["sma_10"].pct_change(5)
        df["sma_20_slope"] = df["sma_20"].pct_change(5)

        # Price vs MAs
        df["price_vs_sma_20"] = (close - df["sma_20"]) / df["sma_20"]
        df["price_vs_sma_50"] = (close - df["sma_50"]) / df["sma_50"]

        # MA crossover signal
        df["sma_10_20_cross"] = (df["sma_10"] > df["sma_20"]).astype(float)

        return df

    def _add_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

        # Daily return
        df["daily_return"] = c.pct_change()

        # Gap (open vs previous close)
        df["gap"] = (o - c.shift(1)) / c.shift(1)

        # Candle body and wicks
        body = (c - o).abs()
        full_range = (h - l).replace(0, np.nan)
        df["body_size"] = body / full_range
        df["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / full_range
        df["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / full_range

        # Consecutive up/down days
        up = (c > c.shift(1)).astype(int)
        down = (c < c.shift(1)).astype(int)
        df["consecutive_up"] = self._consecutive_count(up)
        df["consecutive_down"] = self._consecutive_count(down)

        return df

    # --- Utilities ---

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _consecutive_count(binary_series: pd.Series) -> pd.Series:
        """Count consecutive 1s in a binary series."""
        groups = binary_series.ne(binary_series.shift()).cumsum()
        return binary_series.groupby(groups).cumsum()

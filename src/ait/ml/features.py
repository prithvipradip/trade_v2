"""Feature engineering for ML models.

Computes technical indicators, volatility metrics, and market microstructure
features from price data. All features are calculated from real data only.

Features are organized into groups:
- Momentum (RSI, MACD, rate of change)
- Volatility (ATR, Bollinger Bands, realized vol, IV rank)
- Volume (OBV, volume ratio, VWAP proxy)
- Trend (moving averages, ADX proxy)
- Options-specific (IV rank, vol ratio, vol regime)
- Seasonality (day-of-week, month effects)
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

        # --- Multi-Timeframe Features ---
        features = self._add_multi_timeframe(features)

        # --- IV & Volatility Regime Features ---
        features = self._add_iv_features(features)

        # --- Market Structure Features ---
        features = self._add_market_structure(features)

        # --- Seasonality Features ---
        features = self._add_seasonality(features)

        # Drop rows with NaN from lookback calculations
        features = features.dropna()

        log.debug("features_computed", rows=len(features), columns=len(features.columns))
        return features

    def compute_intraday_features(self, intraday_df: pd.DataFrame) -> dict[str, float]:
        """Compute features from intraday (5-min) data for entry timing.

        Returns a flat dict of intraday features that can be appended
        to the daily feature set for enhanced predictions.
        """
        if intraday_df is None or len(intraday_df) < 20:
            return {}

        close = intraday_df["Close"]
        volume = intraday_df["Volume"]
        high = intraday_df["High"]
        low = intraday_df["Low"]

        features = {}

        # VWAP and price position relative to it
        typical = (high + low + close) / 3
        cum_vol = volume.cumsum()
        vwap = (typical * volume).cumsum() / cum_vol.replace(0, np.nan)
        features["intraday_vwap_position"] = float(
            (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1]
        ) if vwap.iloc[-1] > 0 else 0.0

        # Intraday RSI (7-period on 5-min bars)
        features["intraday_rsi"] = float(self._rsi(close, 7).iloc[-1])

        # Intraday momentum (last 12 bars = 1 hour)
        features["intraday_momentum_1h"] = float(close.pct_change(12).iloc[-1]) if len(close) > 12 else 0.0

        # Intraday volatility (ATR as % of price)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        features["intraday_atr_pct"] = float(atr.iloc[-1] / close.iloc[-1]) if close.iloc[-1] > 0 else 0.0

        # Volume profile: current vs session average
        features["intraday_vol_ratio"] = float(
            volume.iloc[-1] / volume.mean()
        ) if volume.mean() > 0 else 1.0

        # Price range compression (low range = breakout incoming)
        recent_range = (high.tail(12).max() - low.tail(12).min()) / close.iloc[-1]
        session_range = (high.max() - low.min()) / close.iloc[-1]
        features["intraday_range_compression"] = float(
            recent_range / session_range
        ) if session_range > 0 else 1.0

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
            # Multi-timeframe
            "weekly_trend_aligned", "weekly_rsi",
            "weekly_momentum", "volume_confirmation",
            # IV & Volatility Regime
            "iv_rank", "vol_ratio", "vol_trend", "vol_of_vol",
            "vol_regime_expanding", "vol_mean_reversion_signal",
            # Seasonality
            "day_of_week", "month_of_year",
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

    def _add_multi_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe features by resampling daily data to weekly.

        Cross-timeframe alignment is a strong confirmation signal —
        when daily and weekly trends agree, the signal is more reliable.
        """
        close = df["Close"]
        volume = df["Volume"]

        # Weekly SMA trend (resample daily to weekly-equivalent with 5-day rolling)
        weekly_close = close.rolling(5).mean()  # Simulated weekly close
        weekly_sma = weekly_close.rolling(4).mean()  # ~4-week trend (20-day)
        weekly_sma_prev = weekly_sma.shift(5)

        # Weekly trend alignment: 1 = weekly trend up and daily up, -1 = both down, 0 = mixed
        daily_trend = (close > df.get("sma_20", close.rolling(20).mean())).astype(float)
        weekly_trend = (weekly_sma > weekly_sma_prev).astype(float)
        df["weekly_trend_aligned"] = daily_trend * weekly_trend + (1 - daily_trend) * (1 - weekly_trend)

        # Weekly RSI (using 5-day smoothed close)
        df["weekly_rsi"] = self._rsi(weekly_close, 14)

        # Weekly momentum (5-week / 25-day rate of change)
        df["weekly_momentum"] = close.pct_change(25)

        # Volume confirmation: above-average volume on trend-direction days
        avg_volume = volume.rolling(20).mean()
        daily_return = close.pct_change()
        # Volume confirmation = 1 if volume is above average AND price moved in trend direction
        vol_above_avg = (volume > avg_volume).astype(float)
        trend_direction = (daily_return > 0).astype(float)  # 1 = up day
        df["volume_confirmation"] = vol_above_avg * (
            trend_direction * daily_trend + (1 - trend_direction) * (1 - daily_trend)
        )

        return df

    def _add_iv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add implied volatility proxy and vol regime features.

        Since we use OHLCV data (no live IV feed during feature computation),
        we derive IV-proxy features from realized volatility patterns.
        These capture the same dynamics: vol rank, vol expansion/compression,
        and mean-reversion signals that drive options pricing.
        """
        close = df["Close"]
        log_returns = np.log(close / close.shift(1))

        # Rolling realized vol at multiple horizons
        vol_5 = log_returns.rolling(5).std() * np.sqrt(252)
        vol_10 = log_returns.rolling(10).std() * np.sqrt(252)
        vol_20 = log_returns.rolling(20).std() * np.sqrt(252)
        vol_60 = log_returns.rolling(60).std() * np.sqrt(252)

        # IV Rank proxy: where is current 20-day vol relative to rolling range?
        # min_periods=20 so feature is valid after just 40 total rows (20 for vol_20 + 20 more)
        vol_252_min = vol_20.rolling(252, min_periods=20).min()
        vol_252_max = vol_20.rolling(252, min_periods=20).max()
        vol_range = (vol_252_max - vol_252_min).replace(0, np.nan)
        df["iv_rank"] = ((vol_20 - vol_252_min) / vol_range).clip(0, 1)

        # Vol ratio: short-term vs long-term (expansion when > 1, compression when < 1)
        df["vol_ratio"] = (vol_10 / vol_60.replace(0, np.nan)).clip(0, 3)

        # Vol trend: is volatility rising or falling? (5-day slope of 20-day vol)
        df["vol_trend"] = vol_20.pct_change(5).clip(-1, 1)

        # Volatility of volatility: how unstable is vol itself?
        df["vol_of_vol"] = vol_20.rolling(20).std().clip(0, 1)

        # Vol regime: expanding (short-term > long-term) = 1, compressing = 0
        df["vol_regime_expanding"] = (vol_5 > vol_20).astype(float)

        # Mean-reversion signal: when vol is extreme, expect reversion
        # High iv_rank (>0.8) → expect vol to drop → sell premium
        # Low iv_rank (<0.2) → expect vol to rise → buy premium
        vol_zscore = ((vol_20 - vol_60) / vol_60.replace(0, np.nan)).clip(-3, 3)
        df["vol_mean_reversion_signal"] = -vol_zscore  # Negative = expect reversion down

        return df

    def _add_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market structure features for better directional prediction.

        These capture regime, breadth, and fear dynamics that pure price
        action features miss.
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # --- Put/Call proxy: ratio of down-volume to up-volume ---
        # Can't get real P/C ratio from OHLCV, but this captures the same fear dynamic
        price_change = close.diff()
        up_vol = volume.where(price_change > 0, 0)
        down_vol = volume.where(price_change < 0, 0)
        up_vol_ma = up_vol.rolling(10, min_periods=5).sum()
        down_vol_ma = down_vol.rolling(10, min_periods=5).sum()
        df["put_call_proxy"] = (down_vol_ma / up_vol_ma.replace(0, np.nan)).clip(0, 3).fillna(1.0)

        # --- VIX term structure proxy: short vol vs long vol ---
        # Contango (short < long) = calm, backwardation (short > long) = fear
        log_returns = np.log(close / close.shift(1))
        vol_5 = log_returns.rolling(5).std() * np.sqrt(252)
        vol_20 = log_returns.rolling(20).std() * np.sqrt(252)
        vol_60 = log_returns.rolling(60).std() * np.sqrt(252)
        df["vix_term_structure"] = (vol_5 / vol_60.replace(0, np.nan)).clip(0, 3).fillna(1.0)
        # > 1 = backwardation (fear), < 1 = contango (calm)

        # --- Skew proxy: downside vs upside realized moves ---
        # Real skew measures OTM put vs OTM call IV; this proxies it
        ret = close.pct_change()
        downside_vol = ret.where(ret < 0, 0).rolling(20, min_periods=10).std() * np.sqrt(252)
        upside_vol = ret.where(ret > 0, 0).rolling(20, min_periods=10).std() * np.sqrt(252)
        df["skew_proxy"] = (downside_vol / upside_vol.replace(0, np.nan)).clip(0, 3).fillna(1.0)
        # > 1 = more downside fear, < 1 = upside dominance

        # --- Sector rotation proxy: relative performance short vs long term ---
        # If recent returns >> long-term returns, momentum is strong
        ret_5 = close.pct_change(5)
        ret_60 = close.pct_change(60)
        df["momentum_divergence"] = (ret_5 - ret_60).clip(-0.2, 0.2)

        # --- Bear market indicator: price vs SMA200 ---
        if len(close) >= 200:
            sma_200 = close.rolling(200, min_periods=100).mean()
            df["above_sma200"] = (close > sma_200).astype(float)
            df["distance_sma200"] = ((close - sma_200) / sma_200).clip(-0.3, 0.3)
        else:
            df["above_sma200"] = 1.0
            df["distance_sma200"] = 0.0

        # --- Range compression: narrowing range often precedes big moves ---
        atr_5 = (high - low).rolling(5).mean()
        atr_20 = (high - low).rolling(20).mean()
        df["range_compression"] = (atr_5 / atr_20.replace(0, np.nan)).clip(0, 3).fillna(1.0)

        return df

    def _add_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add day-of-week and month seasonality features."""
        idx = df.index
        if hasattr(idx, "dayofweek"):
            df["day_of_week"] = idx.dayofweek / 4.0  # Normalize 0-1 (Mon=0, Fri=1)
            df["month_of_year"] = idx.month / 12.0  # Normalize 0-1
        else:
            df["day_of_week"] = 0.5
            df["month_of_year"] = 0.5
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

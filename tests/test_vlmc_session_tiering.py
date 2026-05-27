"""Tests for Fix 2 — VLMC Feature Session Alignment & Tiering."""

from __future__ import annotations

import datetime as dt
from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from ait.ml.features import FeatureEngine


def _make_daily(days: int = 100, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0005, 0.010, days))
    high  = close * (1 + np.abs(np.random.normal(0, 0.004, days)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.004, days)))
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close,
         "Volume": np.full(days, 5_000_000, dtype=float)},
        index=dates,
    )


def _make_intraday(session_date: date, n_bars: int = 78,
                   price: float = 480.0, seed: int = 0) -> pd.DataFrame:
    """Build 5-min bars for a single day (09:30 to ~16:00)."""
    np.random.seed(seed)
    start_dt = dt.datetime.combine(session_date, time(9, 30))
    times = [start_dt + dt.timedelta(minutes=5 * i) for i in range(n_bars)]
    close = price * np.cumprod(1 + np.random.normal(0, 0.0005, n_bars))
    high  = close * (1 + np.abs(np.random.normal(0, 0.0003, n_bars)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.0003, n_bars)))
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close,
         "Volume": np.full(n_bars, 50_000, dtype=float)},
        index=pd.DatetimeIndex(times),
    )


class TestFeatureEngineVLMCConstant:
    """T2-x / Gap A: VLMC_FEATURE_NAMES class constant."""

    def test_vlmc_feature_names_is_class_attribute(self) -> None:
        assert hasattr(FeatureEngine, "VLMC_FEATURE_NAMES")
        assert isinstance(FeatureEngine.VLMC_FEATURE_NAMES, list)

    def test_vlmc_names_are_unique(self) -> None:
        names = FeatureEngine.VLMC_FEATURE_NAMES
        assert len(names) == len(set(names)), "VLMC_FEATURE_NAMES has duplicates"

    def test_known_intraday_features_in_vlmc_list(self) -> None:
        names = FeatureEngine.VLMC_FEATURE_NAMES
        # These must be present — they are the original 6 intraday features
        for expected in ["intraday_vwap_position", "intraday_rsi",
                         "intraday_momentum_1h", "intraday_atr_pct",
                         "intraday_vol_ratio", "intraday_range_compression"]:
            assert expected in names, f"{expected} missing from VLMC_FEATURE_NAMES"

    def test_session_structure_features_in_vlmc_list(self) -> None:
        names = FeatureEngine.VLMC_FEATURE_NAMES
        for expected in ["power_hour_momentum", "session_vwap_position",
                         "closing_imbalance"]:
            assert expected in names, f"{expected} missing from VLMC_FEATURE_NAMES"


class TestGetFeatureNamesAPI:
    """T2-x / Gap A: get_feature_names() API contract."""

    def test_default_excludes_vlmc(self) -> None:
        fe = FeatureEngine()
        names = fe.get_feature_names()
        for vf in FeatureEngine.VLMC_FEATURE_NAMES:
            assert vf not in names

    def test_include_vlmc_true_adds_all_vlmc_names(self) -> None:
        fe = FeatureEngine()
        names = fe.get_feature_names(include_vlmc=True)
        for vf in FeatureEngine.VLMC_FEATURE_NAMES:
            assert vf in names

    def test_no_duplicates_with_vlmc(self) -> None:
        fe = FeatureEngine()
        names = fe.get_feature_names(include_vlmc=True)
        assert len(names) == len(set(names)), "Duplicate feature names with include_vlmc=True"


class TestSliceIntradayUpTo:
    """T0-4 / Gap D: slice_intraday_up_to() helper."""

    def test_slice_returns_only_bars_at_or_before_cutoff(self) -> None:
        from ait.data.historical import HistoricalDataStore
        session_date = date(2024, 2, 5)
        intraday = _make_intraday(session_date, n_bars=78)
        cutoff = time(11, 0)
        sliced = HistoricalDataStore.slice_intraday_up_to(intraday, cutoff)
        assert not sliced.empty
        for ts in sliced.index:
            assert ts.time() <= cutoff

    def test_slice_empty_when_cutoff_before_first_bar(self) -> None:
        from ait.data.historical import HistoricalDataStore
        session_date = date(2024, 2, 5)
        intraday = _make_intraday(session_date, n_bars=10)
        # Cutoff before market open (09:00)
        cutoff = time(9, 0)
        sliced = HistoricalDataStore.slice_intraday_up_to(intraday, cutoff)
        assert sliced.empty

    def test_slice_all_bars_when_cutoff_is_eod(self) -> None:
        from ait.data.historical import HistoricalDataStore
        session_date = date(2024, 2, 5)
        intraday = _make_intraday(session_date, n_bars=78)
        cutoff = time(16, 0)
        sliced = HistoricalDataStore.slice_intraday_up_to(intraday, cutoff)
        assert len(sliced) == len(intraday)


class TestHurstWaveletShortSession:
    """T2-6: hurst_wavelet_intraday falls back to 0.5 for very short sessions."""

    def test_short_session_returns_default(self) -> None:
        session_date = date(2024, 2, 5)
        # Only 20 bars — less than minimum needed for wavelet decomposition
        short_session = _make_intraday(session_date, n_bars=20)
        fe = FeatureEngine()
        if not hasattr(fe, "_hurst_wavelet"):
            pytest.skip("FeatureEngine._hurst_wavelet is unavailable")
        result, _ = fe._hurst_wavelet(short_session["Close"].to_numpy())
        assert result == pytest.approx(0.5, abs=0.01)
        assert np.isfinite(result)


class TestComputeBaseFeatures:
    """Core: compute() always returns base features without error."""

    def test_compute_returns_dataframe(self) -> None:
        df = _make_daily()
        fe = FeatureEngine()
        result = fe.compute(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_no_all_nan_columns_in_base_features(self) -> None:
        df = _make_daily(days=100)
        fe = FeatureEngine()
        result = fe.compute(df)
        critical = ["rsi_14", "bb_position", "macd_hist", "atr_pct",
                    "price_vs_sma_20", "sma_10_20_cross"]
        for col in critical:
            if col in result.columns:
                non_nan = result[col].dropna()
                assert len(non_nan) > 0, f"{col} is all-NaN"

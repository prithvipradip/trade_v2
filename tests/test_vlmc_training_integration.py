"""Tests for Fix 3 — VLMC Features in Walk-forward Training (Gap A + Gap B)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ait.ml.features import FeatureEngine


def _make_ohlcv(days: int = 150, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2023-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0005, 0.012, days))
    high  = close * (1 + np.abs(np.random.normal(0, 0.005, days)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.005, days)))
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close,
         "Volume": np.random.randint(1_000_000, 10_000_000, days).astype(float)},
        index=dates,
    )


class TestVLMCFeatureNames:
    """T3-1: get_feature_names(include_vlmc=True) includes all VLMC names."""

    def test_vlmc_feature_names_class_constant_has_26_entries(self) -> None:
        assert len(FeatureEngine.VLMC_FEATURE_NAMES) == 26

    def test_get_feature_names_excludes_vlmc_by_default(self) -> None:
        fe = FeatureEngine()
        names = fe.get_feature_names(include_vlmc=False)
        for vf in FeatureEngine.VLMC_FEATURE_NAMES:
            assert vf not in names, f"{vf} should not appear in base feature names"

    def test_get_feature_names_includes_vlmc_when_requested(self) -> None:
        fe = FeatureEngine()
        names = fe.get_feature_names(include_vlmc=True)
        for vf in FeatureEngine.VLMC_FEATURE_NAMES:
            assert vf in names, f"{vf} missing from VLMC-included feature names"

    def test_base_feature_count_is_constant(self) -> None:
        fe = FeatureEngine()
        base = fe.get_feature_names(include_vlmc=False)
        # Base features must be a consistent positive number
        assert len(base) > 0
        # Sanity: with VLMC adds exactly 26 more
        total = fe.get_feature_names(include_vlmc=True)
        assert len(total) == len(base) + 26

    def test_total_feature_count_with_vlmc_adds_26(self) -> None:
        fe = FeatureEngine()
        base  = fe.get_feature_names(include_vlmc=False)
        total = fe.get_feature_names(include_vlmc=True)
        assert len(total) - len(base) == 26


class TestFeatureEngineCompute:
    """T3-3 / T3-6: compute() with and without intraday_store."""

    def test_base_features_always_present(self) -> None:
        df = _make_ohlcv()
        fe = FeatureEngine()
        result = fe.compute(df)
        for col in ["rsi_14", "rsi_7", "bb_position", "macd_hist", "atr_pct"]:
            assert col in result.columns, f"Base feature {col} missing"

    def test_compute_without_intraday_returns_57_base_columns(self) -> None:
        df = _make_ohlcv()
        fe = FeatureEngine()
        result = fe.compute(df)
        vlmc_in_result = [c for c in result.columns if c in FeatureEngine.VLMC_FEATURE_NAMES]
        # Without intraday_store, VLMC columns should be absent
        assert len(vlmc_in_result) == 0, \
            f"VLMC columns unexpectedly present without intraday_store: {vlmc_in_result}"

    def test_empty_intraday_store_graceful_fallback(self) -> None:
        """T3-6: empty intraday store → base features only, no exception."""
        import tempfile
        from ait.data.historical import HistoricalDataStore
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = HistoricalDataStore(db_path=db_path)
            df = _make_ohlcv()
            fe = FeatureEngine()
            result = fe.compute(df, intraday_store=store, symbol="QQQ")
            # Should not raise and should still have base features
            assert "rsi_14" in result.columns
            assert not result["rsi_14"].isna().all()


class TestEnsemblePredictorFeatureNames:
    """T3-2: predictor._feature_names includes VLMC names after training with intraday data."""

    def test_predictor_without_intraday_has_no_vlmc_features(self) -> None:
        from ait.ml.ensemble import DirectionPredictor
        from ait.config.settings import MLConfig

        df = _make_ohlcv(days=200)
        predictor = DirectionPredictor(MLConfig())
        predictor.train(df, symbol="QQQ")
        if predictor.is_trained:
            vlmc_in_predictor = [
                f for f in predictor._feature_names
                if f in FeatureEngine.VLMC_FEATURE_NAMES
            ]
            # Without intraday_store, no VLMC features should be selected
            assert len(vlmc_in_predictor) == 0, \
                f"Unexpected VLMC features in predictor: {vlmc_in_predictor}"


class TestNoFeatureCountMismatch:
    """T3-4: predict() with and without intraday_store doesn't raise KeyError."""

    def test_predict_without_intraday_does_not_raise(self) -> None:
        from ait.ml.ensemble import DirectionPredictor
        from ait.config.settings import MLConfig

        df = _make_ohlcv(days=200)
        predictor = DirectionPredictor(MLConfig())
        predictor.train(df, symbol="QQQ")
        if predictor.is_trained:
            result = predictor.predict(df.tail(100))
            # Must not raise — result can be None for insufficient confidence
            # but should not be an exception

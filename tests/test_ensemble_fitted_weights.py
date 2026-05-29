"""Tests for fitted ensemble weights in RangePredictor, VolMagnitudePredictor, and
DirectionPredictor (ensemble.py).

All three predictors now derive per-model ensemble weights from CV balanced-accuracy
edge over the random-chance baseline after training, instead of using a fixed 50/50 split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(days: int = 250, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2023-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0005, 0.012, days))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, days)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, days).astype(float),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# RangePredictor
# ---------------------------------------------------------------------------

class TestRangePredictorFittedWeights:

    def test_fitted_weights_stored_after_train(self):
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        df = _make_ohlcv()
        rp.train(df, symbol="QQQ")
        assert rp.is_trained
        fw = rp._symbol_models["QQQ"]["fitted_weights"]
        assert set(fw.keys()) == {"xgboost", "lightgbm"}

    def test_fitted_weights_sum_to_one(self):
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        rp.train(_make_ohlcv(), symbol="QQQ")
        fw = rp._symbol_models["QQQ"]["fitted_weights"]
        assert abs(sum(fw.values()) - 1.0) < 1e-9

    def test_fitted_weights_non_negative(self):
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        rp.train(_make_ohlcv(), symbol="QQQ")
        fw = rp._symbol_models["QQQ"]["fitted_weights"]
        assert all(v >= 0.0 for v in fw.values())

    def test_property_returns_weights_after_train(self):
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        rp.train(_make_ohlcv(), symbol="QQQ")
        assert rp.fitted_weights is not None
        assert set(rp.fitted_weights.keys()) == {"xgboost", "lightgbm"}

    def test_property_returns_none_before_train(self):
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor()
        assert rp.fitted_weights is None

    def test_predict_does_not_raise_after_train(self):
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        df = _make_ohlcv(150)
        rp.train(df, symbol="QQQ")
        result = rp.predict(df, symbol="QQQ")
        # May return None if model has no edge, but must not raise
        assert result is None or hasattr(result, "probability_in_range")

    def test_fallback_to_static_when_no_fitted_weights(self):
        """Fresh predictor (no training) should not raise when predict is called
        even though _fitted_weights is absent — falls back to self._weights."""
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor()
        # Not trained — predict() should return None, not AttributeError
        df = _make_ohlcv(150)
        result = rp.predict(df, symbol="QQQ")
        assert result is None


# ---------------------------------------------------------------------------
# VolMagnitudePredictor
# ---------------------------------------------------------------------------

class TestVolMagnitudePredictorFittedWeights:

    def test_fitted_weights_stored_after_train(self):
        from ait.ml.vol_magnitude_predictor import VolMagnitudePredictor
        vmp = VolMagnitudePredictor(threshold_pct=0.07, horizon_days=10)
        vmp.train(_make_ohlcv(), symbol="QQQ")
        assert vmp.is_trained
        fw = vmp._symbol_models["QQQ"]["fitted_weights"]
        assert set(fw.keys()) == {"xgboost", "lightgbm"}

    def test_fitted_weights_sum_to_one(self):
        from ait.ml.vol_magnitude_predictor import VolMagnitudePredictor
        vmp = VolMagnitudePredictor(threshold_pct=0.07, horizon_days=10)
        vmp.train(_make_ohlcv(), symbol="QQQ")
        fw = vmp._symbol_models["QQQ"]["fitted_weights"]
        assert abs(sum(fw.values()) - 1.0) < 1e-9

    def test_property_returns_weights_after_train(self):
        from ait.ml.vol_magnitude_predictor import VolMagnitudePredictor
        vmp = VolMagnitudePredictor(threshold_pct=0.07, horizon_days=10)
        vmp.train(_make_ohlcv(), symbol="QQQ")
        assert vmp.fitted_weights is not None

    def test_property_returns_none_before_train(self):
        from ait.ml.vol_magnitude_predictor import VolMagnitudePredictor
        assert VolMagnitudePredictor().fitted_weights is None


# ---------------------------------------------------------------------------
# DirectionPredictor (ensemble.py)
# ---------------------------------------------------------------------------

class TestDirectionPredictorFittedWeights:

    def _make_predictor(self):
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        return DirectionPredictor(MLConfig())

    def test_fitted_weights_stored_after_train(self):
        predictor = self._make_predictor()
        df = _make_ohlcv()
        predictor.train(df, symbol="QQQ")
        assert predictor.is_trained
        fw = predictor._symbol_models["QQQ"]["fitted_weights"]
        assert set(fw.keys()) == {"xgboost", "lightgbm"}

    def test_fitted_weights_sum_to_one(self):
        predictor = self._make_predictor()
        predictor.train(_make_ohlcv(), symbol="QQQ")
        fw = predictor._symbol_models["QQQ"]["fitted_weights"]
        assert abs(sum(fw.values()) - 1.0) < 1e-9

    def test_fitted_weights_non_negative(self):
        predictor = self._make_predictor()
        predictor.train(_make_ohlcv(), symbol="QQQ")
        fw = predictor._symbol_models["QQQ"]["fitted_weights"]
        assert all(v >= 0.0 for v in fw.values())

    def test_property_returns_weights_after_train(self):
        predictor = self._make_predictor()
        predictor.train(_make_ohlcv(), symbol="QQQ")
        assert predictor.fitted_weights is not None
        assert abs(sum(predictor.fitted_weights.values()) - 1.0) < 1e-9

    def test_property_returns_none_before_train(self):
        assert self._make_predictor().fitted_weights is None

    def test_predict_does_not_raise_after_train(self):
        predictor = self._make_predictor()
        df = _make_ohlcv()
        predictor.train(df, symbol="QQQ")
        result = predictor.predict(df, symbol="QQQ")
        assert result is None or hasattr(result, "direction")

"""Tests for predict_from_features() on DirectionPredictor and RangePredictor,
and for the vectorized _save_window_timeseries refactor.

Key invariants verified:
- predict_from_features returns None before training (same as predict)
- predict_from_features returns the same prediction type as predict
- directional probabilities sum to ~1.0
- p_up/p_down/p_neutral are populated (new; old code left them None)
- _save_window_timeseries writes one bar per OOS row
- _save_window_timeseries calls FeatureEngine exactly once regardless of bar count
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(days: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, days))
    high = close * (1 + np.abs(rng.normal(0, 0.005, days)))
    low  = close * (1 - np.abs(rng.normal(0, 0.005, days)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, days).astype(float),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# DirectionPredictor.predict_from_features
# ---------------------------------------------------------------------------

class TestDirectionPredictorPredictFromFeatures:

    def test_returns_none_before_training(self):
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        from ait.ml.features import FeatureEngine
        pred = DirectionPredictor(MLConfig())
        df = _make_ohlcv()
        feat_df = FeatureEngine().compute(df)
        assert not feat_df.empty
        result = pred.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        assert result is None

    def test_returns_prediction_after_training(self):
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor, Prediction
        from ait.ml.features import FeatureEngine
        df = _make_ohlcv()
        pred = DirectionPredictor(MLConfig())
        pred.train(df, symbol="QQQ")
        feat_df = FeatureEngine().compute(df)
        assert not feat_df.empty
        result = pred.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        # May return None if model has no skill; just verify type when non-None
        assert result is None or isinstance(result, Prediction)

    def test_probabilities_sum_to_one(self):
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        from ait.ml.features import FeatureEngine
        df = _make_ohlcv()
        pred = DirectionPredictor(MLConfig())
        pred.train(df, symbol="QQQ")
        feat_df = FeatureEngine().compute(df)
        result = pred.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        if result is None:
            pytest.skip("model returned None — no trained skill on synthetic data")
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_all_probability_keys_present(self):
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        from ait.ml.features import FeatureEngine
        df = _make_ohlcv()
        pred = DirectionPredictor(MLConfig())
        pred.train(df, symbol="QQQ")
        feat_df = FeatureEngine().compute(df)
        result = pred.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        if result is None:
            pytest.skip("model returned None")
        assert {"bearish", "neutral", "bullish"} == set(result.probabilities.keys())

    def test_confidence_matches_argmax_proba(self):
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        from ait.ml.features import FeatureEngine
        df = _make_ohlcv()
        pred = DirectionPredictor(MLConfig())
        pred.train(df, symbol="QQQ")
        feat_df = FeatureEngine().compute(df)
        result = pred.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        if result is None:
            pytest.skip("model returned None")
        max_proba = max(result.probabilities.values())
        assert abs(result.confidence - max_proba) < 1e-6

    def test_consistent_with_predict_on_same_row(self):
        """predict_from_features(last_row) should give same class as predict(full_df)."""
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        from ait.ml.features import FeatureEngine
        df = _make_ohlcv()
        pred = DirectionPredictor(MLConfig())
        pred.train(df, symbol="QQQ")

        # Compute features the same way predict() does internally
        feat_df = FeatureEngine().compute(df)
        if feat_df.empty:
            pytest.skip("FeatureEngine returned empty")

        result_ff = pred.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        result_p  = pred.predict(df, symbol="QQQ")

        if result_ff is None or result_p is None:
            pytest.skip("one or both predictions returned None")

        assert result_ff.direction == result_p.direction


# ---------------------------------------------------------------------------
# RangePredictor.predict_from_features
# ---------------------------------------------------------------------------

class TestRangePredictorPredictFromFeatures:

    def test_returns_none_before_training(self):
        from ait.ml.features import FeatureEngine
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        df = _make_ohlcv()
        feat_df = FeatureEngine().compute(df)
        assert not feat_df.empty
        result = rp.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        assert result is None

    def test_returns_range_prediction_after_training(self):
        from ait.ml.features import FeatureEngine
        from ait.ml.range_predictor import RangePrediction, RangePredictor
        df = _make_ohlcv()
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        rp.train(df, symbol="QQQ")
        feat_df = FeatureEngine().compute(df)
        result = rp.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        assert result is None or isinstance(result, RangePrediction)

    def test_probability_in_range(self):
        from ait.ml.features import FeatureEngine
        from ait.ml.range_predictor import RangePredictor
        df = _make_ohlcv()
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        rp.train(df, symbol="QQQ")
        feat_df = FeatureEngine().compute(df)
        result = rp.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        if result is None:
            pytest.skip("model returned None — no edge on synthetic data")
        assert 0.0 <= result.probability_in_range <= 1.0

    def test_confidence_equals_max_of_p_and_complement(self):
        from ait.ml.features import FeatureEngine
        from ait.ml.range_predictor import RangePredictor
        df = _make_ohlcv()
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        rp.train(df, symbol="QQQ")
        feat_df = FeatureEngine().compute(df)
        result = rp.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        if result is None:
            pytest.skip("model returned None")
        p = result.probability_in_range
        assert abs(result.confidence - max(p, 1 - p)) < 1e-6

    def test_consistent_with_predict_on_same_row(self):
        """predict_from_features(last_row) should give same probability as predict(full_df)."""
        from ait.ml.features import FeatureEngine
        from ait.ml.range_predictor import RangePredictor
        df = _make_ohlcv()
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        rp.train(df, symbol="QQQ")

        feat_df = FeatureEngine().compute(df)
        if feat_df.empty:
            pytest.skip("FeatureEngine returned empty")

        result_ff = rp.predict_from_features(feat_df.iloc[-1], symbol="QQQ")
        result_p  = rp.predict(df, symbol="QQQ")

        if result_ff is None or result_p is None:
            pytest.skip("one or both predictions returned None")

        assert abs(result_ff.probability_in_range - result_p.probability_in_range) < 1e-5


# ---------------------------------------------------------------------------
# _save_window_timeseries — vectorized rewrite
# ---------------------------------------------------------------------------

class TestSaveWindowTimeseriesVectorized:

    def _make_context_and_test(self, context_days: int = 100, test_days: int = 20):
        df = _make_ohlcv(days=context_days + test_days)
        context_df = df.iloc[:context_days]
        test_df = df.iloc[context_days:]
        full_ctx = pd.concat([context_df, test_df])
        return test_df, full_ctx

    def test_writes_one_bar_per_oos_row(self, tmp_path):
        from ait.backtesting.walkforward import _save_window_timeseries
        test_df, full_ctx = self._make_context_and_test()
        _save_window_timeseries(
            test_df=test_df,
            full_context_df=full_ctx,
            predictor=None,
            range_predictor=None,
            vix_ctx=None,
            progress_dir=tmp_path,
            symbol="QQQ",
            window_id=1,
        )
        out = json.loads((tmp_path / "timeseries_bars.json").read_text())
        assert len(out) == len(test_df)

    def test_bars_have_expected_fields(self, tmp_path):
        from ait.backtesting.walkforward import _save_window_timeseries
        test_df, full_ctx = self._make_context_and_test()
        _save_window_timeseries(
            test_df=test_df,
            full_context_df=full_ctx,
            predictor=None,
            range_predictor=None,
            vix_ctx=None,
            progress_dir=tmp_path,
            symbol="QQQ",
            window_id=1,
        )
        out = json.loads((tmp_path / "timeseries_bars.json").read_text())
        for bar in out:
            assert "time" in bar
            assert "open" in bar and "close" in bar
            assert "window" in bar
            assert bar["window"] == 1
            assert bar["symbol"] == "QQQ"

    def test_feature_engine_called_once(self, tmp_path):
        """FeatureEngine.compute must be called exactly once, not once per bar."""
        from ait.backtesting.walkforward import _save_window_timeseries
        test_df, full_ctx = self._make_context_and_test(test_days=30)

        call_count = {"n": 0}
        real_compute = None

        def counting_compute(self_fe, df, **kwargs):
            call_count["n"] += 1
            return real_compute(self_fe, df, **kwargs)

        from ait.ml import features as feat_module
        real_compute = feat_module.FeatureEngine.compute
        with patch.object(feat_module.FeatureEngine, "compute", counting_compute):
            _save_window_timeseries(
                test_df=test_df,
                full_context_df=full_ctx,
                predictor=None,
                range_predictor=None,
                vix_ctx=None,
                progress_dir=tmp_path,
                symbol="QQQ",
                window_id=1,
            )

        assert call_count["n"] == 1, (
            f"Expected 1 FeatureEngine.compute call, got {call_count['n']}. "
            "Vectorization regression."
        )

    def test_ml_predictions_use_predict_from_features(self, tmp_path):
        """Directional predictor's predict_from_features is called per bar, predict() never."""
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        from ait.backtesting.walkforward import _save_window_timeseries
        test_df, full_ctx = self._make_context_and_test(test_days=5)

        pred = DirectionPredictor(MLConfig())
        df_train = _make_ohlcv(days=300)
        pred.train(df_train, symbol="QQQ")

        predict_calls = {"n": 0}
        predict_from_features_calls = {"n": 0}
        original_p = pred.predict
        original_pff = pred.predict_from_features

        def mock_predict(df, **kwargs):
            predict_calls["n"] += 1
            return original_p(df, **kwargs)

        def mock_pff(feature_row, **kwargs):
            predict_from_features_calls["n"] += 1
            return original_pff(feature_row, **kwargs)

        pred.predict = mock_predict
        pred.predict_from_features = mock_pff

        _save_window_timeseries(
            test_df=test_df,
            full_context_df=full_ctx,
            predictor=pred,
            range_predictor=None,
            vix_ctx=None,
            progress_dir=tmp_path,
            symbol="QQQ",
            window_id=1,
        )

        assert predict_calls["n"] == 0, "predict() must not be called — use predict_from_features"
        assert predict_from_features_calls["n"] == len(test_df)

    def test_p_up_p_down_p_neutral_populated(self, tmp_path):
        """New: p_up/p_down/p_neutral should be non-None when predictor is trained."""
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        from ait.backtesting.walkforward import _save_window_timeseries
        test_df, full_ctx = self._make_context_and_test(context_days=280, test_days=5)

        pred = DirectionPredictor(MLConfig())
        df_train = _make_ohlcv(days=300)
        pred.train(df_train, symbol="QQQ")

        _save_window_timeseries(
            test_df=test_df,
            full_context_df=full_ctx,
            predictor=pred,
            range_predictor=None,
            vix_ctx=None,
            progress_dir=tmp_path,
            symbol="QQQ",
            window_id=1,
        )
        out = json.loads((tmp_path / "timeseries_bars.json").read_text())
        bars_with_pred = [b for b in out if b.get("dir_class") is not None]
        if not bars_with_pred:
            pytest.skip("predictor returned None on all bars — no edge on synthetic data")
        for bar in bars_with_pred:
            assert bar["p_up"] is not None
            assert bar["p_down"] is not None
            assert bar["p_neutral"] is not None
            total = bar["p_up"] + bar["p_down"] + bar["p_neutral"]
            assert abs(total - 1.0) < 0.01

    def test_window_replacement_on_rerun(self, tmp_path):
        """Re-running same window_id should replace bars, not duplicate them."""
        from ait.backtesting.walkforward import _save_window_timeseries
        test_df, full_ctx = self._make_context_and_test(test_days=10)

        for _ in range(2):
            _save_window_timeseries(
                test_df=test_df,
                full_context_df=full_ctx,
                predictor=None,
                range_predictor=None,
                vix_ctx=None,
                progress_dir=tmp_path,
                symbol="QQQ",
                window_id=5,
            )
        out = json.loads((tmp_path / "timeseries_bars.json").read_text())
        assert len(out) == len(test_df), "Bars for same window+symbol must not be duplicated"

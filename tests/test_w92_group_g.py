"""W92: group G, the last live-shared numbers in the literal census, plus a drift detector that really throttles and can be switched off.

* ``ml.drift_*`` (switch default OFF, window 50, accuracy floor 40%, high-confidence cutoff 80%, calibration gap 25 points, 20 / 5
  minimum samples, check interval 10 that was stored and never read), ``ml.model_versions_kept`` (5);
* ``meta_label.{min_trades_for_training, min_class_trades, cv_max_splits, cv_min_rows_per_split}`` (30 / 5 / 5 / 10);
* ``regime.*``: the seven trend-score numbers and the eight confidence outputs of ``RegimeDetector``.
Values equal the old literals, so the regime classifier reproduces a golden record recorded BEFORE the change.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from ait.config.settings import MetaLabelConfig, MLConfig, RegimeConfig, load_settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "data"))
from _w92_regime_inputs import replay  # noqa: E402

CONFIG_TEXT = (ROOT / "config.yaml").read_text()
GUIDE_TEXT = (ROOT / "docs" / "RESEARCH_GUIDE.md").read_text()

OLD = {
    "ml": {"drift_detection_enabled": False, "drift_window_size": 50, "drift_accuracy_threshold": 0.40, "drift_high_confidence": 0.80,
           "drift_calibration_threshold": 0.25, "drift_min_samples": 20, "drift_min_high_confidence_samples": 5,
           "drift_check_interval": 10, "model_versions_kept": 5},
    "meta_label": {"min_trades_for_training": 30, "min_class_trades": 5, "cv_max_splits": 5, "cv_min_rows_per_split": 10},
    "regime": {"trend_sma_periods": [10, 20, 50], "trend_slope_lookback": 5, "trend_slope_scale": 100.0,
               "trend_deviation_periods": [20, 50], "trend_deviation_scale": 10.0, "trend_momentum_days": 20,
               "trend_momentum_scale": 5.0, "high_vol_confidence_floor": 0.80, "high_vol_confidence_cap": 0.90,
               "high_vol_confidence_vol_scale": 0.50, "iv_crush_confidence": 0.70, "mean_revert_confidence_floor": 0.60,
               "low_vol_confidence": 0.70, "trend_confidence_cap": 0.95, "range_confidence_floor": 0.50},
}


class TestDefaultsAndDocs:
    def test_defaults_equal_the_old_literals_in_schema_and_config_yaml(self):
        s = load_settings()
        for section, fields in OLD.items():
            for name, old in fields.items():
                assert getattr(getattr(s, section), name) == old, (section, name)
        assert (MLConfig().drift_detection_enabled, MetaLabelConfig().min_class_trades, RegimeConfig().trend_slope_scale) == (False, 5, 100.0)

    def test_every_new_setting_is_in_config_yaml_with_a_comment_and_in_the_guide(self):
        """Each setting is explained in config.yaml (a comment on its line or the line above) AND named in RESEARCH_GUIDE.md."""
        lines = CONFIG_TEXT.splitlines()
        for section, fields in OLD.items():
            start = next(i for i, l in enumerate(lines) if l.startswith(f"{section}:"))
            end = next((i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith((" ", "#"))), len(lines))
            block = lines[start:end]
            for name in fields:
                idx = next((i for i, l in enumerate(block) if re.match(rf"\s+{name}:", l)), None)
                assert idx is not None, f"{section}.{name} is not in config.yaml"
                documented = "#" in block[idx].split(":", 1)[1] or block[idx - 1].lstrip().startswith("#")
                assert documented, f"{section}.{name} has no comment in config.yaml"
                assert f"`{name}`" in GUIDE_TEXT or f"`{section}.{name}`" in GUIDE_TEXT, (
                    f"{section}.{name} is not explained in docs/RESEARCH_GUIDE.md")

    def test_the_config_blocks_point_at_the_guide(self):
        assert CONFIG_TEXT.count('docs/RESEARCH_GUIDE.md, "Drift detection"') >= 1
        assert CONFIG_TEXT.count('docs/RESEARCH_GUIDE.md, "The regime classifier"') >= 1

    def test_the_drift_numbers_must_be_consistent(self):
        with pytest.raises(ValueError, match="drift_min_samples"):
            MLConfig(drift_window_size=20, drift_min_samples=30)

    def test_the_high_vol_floor_cannot_exceed_the_cap(self):
        with pytest.raises(ValueError, match="high_vol_confidence_floor"):
            RegimeConfig(high_vol_confidence_floor=0.95, high_vol_confidence_cap=0.9)


class TestRegime:
    @staticmethod
    def _detector(**kw):
        from ait.ml.regime import RegimeDetector
        return RegimeDetector(RegimeConfig(**kw))

    def test_the_classifier_reproduces_the_golden_record(self):
        want = json.loads((ROOT / "tests" / "data" / "w92_regime_golden.json").read_text())
        assert replay(self._detector()) == want                     # recorded before the numbers became settings

    @staticmethod
    def _expected_trend(close, sma_periods, lookback, slope_scale, dev_periods, dev_scale, mom_days, mom_scale):
        clamp = lambda v: max(-1, min(1, v))                        # noqa: E731
        scores = [clamp(close.rolling(p).mean().pct_change(lookback).iloc[-1] * slope_scale) for p in sma_periods]
        for p in dev_periods:
            m = close.rolling(p).mean().iloc[-1]
            scores.append(clamp((close.iloc[-1] - m) / m * dev_scale))
        scores.append(clamp(close.pct_change(mom_days).iloc[-1] * mom_scale))
        return float(np.mean(scores))

    def test_the_trend_score_follows_every_trend_setting(self):
        rng = np.random.default_rng(4)
        # gentle drift and low noise, so no term saturates at +/-1 (a clamped term would hide a changed setting)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.002, 300))))
        base = dict(sma_periods=[10, 20, 50], lookback=5, slope_scale=100.0, dev_periods=[20, 50], dev_scale=10.0, mom_days=20, mom_scale=5.0)
        assert self._detector()._measure_trend(close) == pytest.approx(self._expected_trend(close, **base), abs=1e-12)
        moves = {"trend_sma_periods": ("sma_periods", [5, 30]), "trend_slope_lookback": ("lookback", 10),
                 "trend_slope_scale": ("slope_scale", 20.0), "trend_deviation_periods": ("dev_periods", [10]),
                 "trend_deviation_scale": ("dev_scale", 2.0), "trend_momentum_days": ("mom_days", 60), "trend_momentum_scale": ("mom_scale", 1.0)}
        for field, (arg, value) in moves.items():
            got = self._detector(**{field: value})._measure_trend(close)
            want = self._expected_trend(close, **{**base, arg: value})
            assert got == pytest.approx(want, abs=1e-12), field
            assert got != pytest.approx(self._detector()._measure_trend(close), abs=1e-9), f"{field} did not move the score"

    @pytest.mark.parametrize("overrides,kwargs,expect", [
        ({"high_vol_confidence_floor": 0.55}, dict(trend=0.0, vol=0.1, vix=35.0), 0.55),            # vix trigger, low vol -> the floor
        ({"high_vol_confidence_floor": 0.5, "high_vol_confidence_cap": 0.65}, dict(trend=0.0, vol=0.9, vix=None), 0.65),   # the cap
        ({"high_vol_confidence_floor": 0.5, "high_vol_confidence_vol_scale": 1.0}, dict(trend=0.0, vol=0.6, vix=None), 0.6),
        ({"iv_crush_confidence": 0.42}, dict(trend=0.0, vol=0.1, vix=None, iv_rank=0.1), 0.42),
        ({"low_vol_confidence": 0.43}, dict(trend=0.0, vol=0.1, vix=None), 0.43),
        ({"trend_confidence_cap": 0.61}, dict(trend=0.99, vol=0.2, vix=None), 0.61),
        ({"trend_confidence_cap": 0.62}, dict(trend=-0.99, vol=0.2, vix=None), 0.62),                 # the down side too
        ({"range_confidence_floor": 0.45}, dict(trend=0.3, vol=0.2, vix=None), 0.45),
    ])
    def test_each_confidence_output_is_the_setting(self, overrides, kwargs, expect):
        base = dict(iv_rank=0.5, hurst=0.5)
        _, conf = self._detector(**overrides)._classify(**{**base, **kwargs})
        assert conf == pytest.approx(expect)

    def test_the_mean_revert_floor_binds_when_the_hurst_band_is_widened(self):
        # at the default hurst band (< 0.4) 1 - hurst always exceeds 0.6; widen it so the floor can bind
        regime, conf = self._detector(mean_revert_hurst_max=0.5, mean_revert_confidence_floor=0.8)._classify(
            0.0, 0.2, None, 0.5, 0.45)
        assert regime.value == "mean_reverting" and conf == pytest.approx(0.8)

    def test_the_high_vol_floor_and_the_entry_gate_are_a_must_agree_pair(self):
        """A VIX-triggered high-vol day with realized vol <= 40% scores exactly the floor (0.80); the live gate is a strict '>' 0.80."""
        _, conf = self._detector()._classify(0.0, 0.25, 35.0, 0.5, 0.5)
        assert conf == pytest.approx(0.80) and not conf > load_settings().ml.regime_override_confidence
        _, conf = self._detector()._classify(0.0, 0.45, None, 0.5, 0.5)
        assert conf > load_settings().ml.regime_override_confidence


class TestDrift:
    @staticmethod
    def _run(detector, n=30, right=1.0, conf=0.6):
        for i in range(n):
            detector.record_prediction(f"T{i}", "bullish", conf)
            detector.record_outcome(f"T{i}", "bullish" if i < n * right else "bearish")
        return detector

    def test_the_switch_defaults_off_and_a_disabled_detector_is_inert(self):
        from ait.ml.drift import DriftDetector
        assert load_settings().ml.drift_detection_enabled is False
        d = DriftDetector(enabled=False)
        self._run(d, 30, right=0.0)                               # 100% wrong, would drift if it were on
        r = d.check_drift()
        assert (r.is_drifting, r.should_retrain, r.samples) == (False, False, 0)
        assert d.pending_count == 0 and d.completed_count == 0
        on = self._run(DriftDetector(), 30, right=0.0)            # a bare detector is active
        assert on.check_drift().should_retrain is True
        # switching an active detector off silences it at once, even with data already recorded
        on._enabled = False
        r2 = on.check_drift()
        assert (r2.is_drifting, r2.should_retrain, r2.samples) == (False, False, 0)

    def test_the_trainer_passes_the_switch_and_every_number(self):
        from unittest.mock import MagicMock

        from ait.ml.trainer import ModelTrainer
        off = ModelTrainer(MLConfig(), MagicMock(), MagicMock(), MagicMock()).drift_detector
        assert off._enabled is False
        cfg = MLConfig(drift_detection_enabled=True, drift_check_interval=3, drift_min_samples=25, drift_min_high_confidence_samples=7,
                       drift_window_size=60)
        d = ModelTrainer(cfg, MagicMock(), MagicMock(), MagicMock()).drift_detector
        assert (d._enabled, d._check_interval, d._min_samples, d._min_high_conf_samples, d._window_size) == (True, 3, 25, 7, 60)

    def test_the_check_is_throttled_to_one_full_evaluation_per_interval(self):
        from ait.ml.drift import DriftDetector
        d = DriftDetector(check_interval=10)
        self._run(d, 25, right=1.0)
        first = d.check_drift()
        assert first.is_drifting is False and first.samples == 25
        # make it drift by adding wrong outcomes, but fewer than 10 NEW predictions: the cached report comes back
        for i in range(5):
            d.record_prediction(f"X{i}", "bullish", 0.6)
            d.record_outcome(f"X{i}", "bearish")
        assert d.check_drift() is first
        for i in range(5, 40):
            d.record_prediction(f"X{i}", "bullish", 0.6)
            d.record_outcome(f"X{i}", "bearish")
        later = d.check_drift()
        assert later is not first and later.is_drifting is True
        every = DriftDetector(check_interval=1)
        self._run(every, 25, right=1.0)
        a = every.check_drift()
        every.record_prediction("Z", "bullish", 0.6)
        assert every.check_drift() is not a                        # 1 = evaluate on every new prediction

    def test_acknowledging_a_retrain_resets_the_throttle(self):
        from ait.ml.drift import DriftDetector
        d = DriftDetector(check_interval=100)
        self._run(d, 25, right=0.0)
        assert d.check_drift().should_retrain is True
        d.acknowledge_retrain()
        self._run(d, 25, right=1.0)
        assert d.check_drift().is_drifting is False                # evaluated at once, not served from the cache

    def test_minimum_samples(self):
        from ait.ml.drift import DriftDetector
        assert self._run(DriftDetector(min_samples=40), 30, right=0.0).check_drift().is_drifting is False
        assert self._run(DriftDetector(min_samples=25), 30, right=0.0).check_drift().is_drifting is True

    def test_minimum_high_confidence_samples(self):
        """Accuracy is fine (100%); only the calibration test can fire, and only with enough high-confidence outcomes."""
        from ait.ml.drift import DriftDetector

        def run(min_hc):
            d = DriftDetector(high_confidence=0.8, calibration_threshold=0.05, accuracy_threshold=0.2, min_high_confidence_samples=min_hc)
            for i in range(30):
                high = i < 6                                       # 6 high-confidence predictions, all wrong
                d.record_prediction(f"T{i}", "bullish", 0.9 if high else 0.5)
                d.record_outcome(f"T{i}", "bearish" if high else "bullish")
            return d.check_drift()
        assert run(5).is_drifting is True and run(7).is_drifting is False

    def test_the_four_thresholds_each_change_the_verdict(self):
        from ait.ml.drift import DriftDetector
        # 50% right, confidence 0.9 throughout
        def verdict(**kw):
            d = DriftDetector(**kw)
            for i in range(40):
                d.record_prediction(f"T{i}", "bullish", 0.9)
                d.record_outcome(f"T{i}", "bullish" if i % 2 == 0 else "bearish")
            return d.check_drift()
        assert verdict(accuracy_threshold=0.40, calibration_threshold=0.60).is_drifting is False
        assert verdict(accuracy_threshold=0.60).is_drifting is True                       # accuracy floor
        assert verdict(accuracy_threshold=0.2, calibration_threshold=0.25, high_confidence=0.9).is_drifting is True    # 90% claimed, 50% right
        assert verdict(accuracy_threshold=0.2, calibration_threshold=0.25, high_confidence=0.7).is_drifting is False   # cohort expects 70%: gap .2
        small = verdict(window_size=20, min_samples=20)
        assert small.samples == 20                                                       # the window caps what is scored


class TestVersionsAndMetaLabeler:
    def test_model_versions_kept(self, tmp_path):
        from ait.ml.ensemble import DirectionPredictor
        for kept, left in ((2, 2), (5, 5)):
            d = tmp_path / f"k{kept}"
            d.mkdir()
            for v in range(7):
                (d / f"ensemble_v-2026010{v}.pkl").write_bytes(b"x")
            p = DirectionPredictor(MLConfig(model_versions_kept=kept), model_dir=d, persist_artifacts=False)
            p._prune_old_versions()
            assert sorted(x.name for x in d.glob("ensemble_v-*.pkl")) == [f"ensemble_v-2026010{v}.pkl" for v in range(7 - left, 7)]
        p._prune_old_versions(keep=1)                                       # an explicit argument still wins
        assert len(list((tmp_path / "k5").glob("ensemble_v-*.pkl"))) == 1

    def test_saving_prunes_to_the_setting(self, tmp_path):
        from ait.ml.ensemble import DirectionPredictor
        p = DirectionPredictor(MLConfig(model_versions_kept=2), model_dir=tmp_path, persist_artifacts=True)
        for v in range(4):
            p._model_version = f"v-2026010{v}-000000"
            assert p._save_models() is True
        assert len(list(tmp_path.glob("ensemble_v-*.pkl"))) == 2

    def _frame(self, n, pos):
        rng = np.random.default_rng(7)
        return pd.DataFrame({"primary_confidence": rng.uniform(0.5, 0.95, n), "vix": rng.uniform(12, 30, n),
                             "iv_rank": rng.uniform(0, 1, n), "sentiment_score": rng.uniform(-1, 1, n), "rsi_14": rng.uniform(20, 80, n),
                             "profitable": rng.permutation([1] * pos + [0] * (n - pos))})

    def test_the_meta_labeler_floors_follow_the_config(self):
        from ait.ml.meta_label import MetaLabeler
        low = MetaLabelConfig(min_feature_coverage=5)
        with capture_logs() as ev:
            assert MetaLabeler(meta_label_config=low).train(self._frame(20, 10)) == {}
        assert any(e["event"] == "insufficient_trades_for_meta_label" and e["required"] == 30 for e in ev)
        with capture_logs() as ev_req:
            MetaLabeler(meta_label_config=MetaLabelConfig(min_feature_coverage=5, min_trades_for_training=25)).train(self._frame(20, 10))
        assert any(e["event"] == "insufficient_trades_for_meta_label" and e["required"] == 25 for e in ev_req)
        with capture_logs() as ev2:
            MetaLabeler(meta_label_config=MetaLabelConfig(min_feature_coverage=5, min_trades_for_training=10)).train(self._frame(20, 10))
        assert not any(e["event"] == "insufficient_trades_for_meta_label" for e in ev2)
        with capture_logs() as ev3:
            MetaLabeler(meta_label_config=MetaLabelConfig(min_feature_coverage=5, min_trades_for_training=10)).train(self._frame(40, 3))
        assert any(e["event"] == "meta_label_class_imbalance" for e in ev3)                         # 3 profitable < the default 5
        with capture_logs() as ev4:
            MetaLabeler(meta_label_config=MetaLabelConfig(min_feature_coverage=5, min_trades_for_training=10, min_class_trades=2)).train(self._frame(40, 3))
        assert not any(e["event"] == "meta_label_class_imbalance" for e in ev4)

    def test_the_cross_validation_fold_rule_follows_the_config(self, monkeypatch):
        import ait.ml.meta_label as ml
        seen = []

        class Spy(ml.TimeSeriesSplit):
            def __init__(self, n_splits=5, **kw):
                seen.append(n_splits)
                super().__init__(n_splits=n_splits, **kw)
        monkeypatch.setattr(ml, "TimeSeriesSplit", Spy)
        for cfg, rows, want in ((MetaLabelConfig(min_feature_coverage=5), 80, 5),
                                (MetaLabelConfig(min_feature_coverage=5, cv_max_splits=3), 80, 3),
                                (MetaLabelConfig(min_feature_coverage=5, cv_min_rows_per_split=20), 80, 4),
                                (MetaLabelConfig(min_feature_coverage=5, cv_min_rows_per_split=100), 80, 2)):
            seen.clear()
            ml.MetaLabeler(meta_label_config=cfg).train(self._frame(rows, rows // 2))
            assert seen and seen[0] == want, (cfg.cv_max_splits, cfg.cv_min_rows_per_split, seen)

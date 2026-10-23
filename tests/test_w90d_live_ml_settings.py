"""W90d: the live models' labels, validation folds, class-balance guards and retrain constants are ``ml.*`` settings.

Range band (+/-5%) and its edge floor, the vol-magnitude move (+/-7% over 30 days) and edge floor, the direction label
(+/-1% over 5 days), the walk-forward folds (5 / gap 5 / min 20 val / min 50 train for the direction model, 4 / gap
max(5, horizon) for the binary models), the class-balance guards (0.05 / 0.98 / 10 rows) and the retrain's trials, seed and
rollback margin were literals in ``ml/*.py``. SHARED by live and research, so they are in ``config.yaml``; defaults equal
the old literals. Each setting is shown to MOVE behaviour.
"""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from ait.config.settings import MLConfig, TreeHyperparamsConfig, load_settings

OLD = {
    "range_threshold_pct": 0.05, "range_min_edge_over_baseline": 0.05, "vol_magnitude_threshold_pct": 0.07,
    "vol_magnitude_horizon_days": 30, "vol_magnitude_min_edge_over_baseline": 0.10, "direction_label_threshold_pct": 0.01,
    "direction_label_horizon_days": 5, "direction_cv_splits": 5, "direction_cv_gap_days": 5, "cv_min_val_rows": 20,
    "cv_min_train_rows": 50, "binary_cv_splits": 4, "binary_cv_min_gap_days": 5, "label_balance_min": 0.05,
    "label_balance_max": 0.98, "min_class_samples": 10, "retrain_optuna_trials": 50, "retrain_optuna_seed": 42,
    "rollback_margin": 0.05,
}
SMALL_TREES = TreeHyperparamsConfig(n_estimators=10, max_depth=2)


def _ml(**kw) -> MLConfig:
    return load_settings().ml.model_copy(update={"hyperparams": SMALL_TREES, **kw})


def _walk(n=320, vol=0.015, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, vol, n)))
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close, "Volume": 1e6}, index=idx)


def test_defaults_equal_the_old_literals_in_schema_and_config_yaml():
    schema, loaded = MLConfig(), load_settings().ml
    for name, old in OLD.items():
        assert getattr(schema, name) == old == getattr(loaded, name), name


def test_label_balance_limits_returns_each_setting():
    from ait.ml.hyperparams import label_balance_limits
    assert label_balance_limits(_ml(label_balance_min=0.2, label_balance_max=0.9, min_class_samples=33)) == (0.2, 0.9, 33)
    assert label_balance_limits() == (0.05, 0.98, 10)


def test_the_balance_limits_must_be_ordered():
    with pytest.raises(ValueError, match="label_balance_min"):
        MLConfig(label_balance_min=0.6, label_balance_max=0.55)


class TestRangeSpec:
    def test_the_live_band_is_the_setting(self):
        from ait.ml import range_spec as rs
        s = SimpleNamespace(options=SimpleNamespace(dte_range=[21, 45]), ml=SimpleNamespace(range_threshold_pct=0.08))
        assert rs.live_range_spec(s) == (0.08, rs.horizon_for(21))
        # a settings stub without the field degrades to the LOADED value, not a copy of the number
        s2 = SimpleNamespace(options=SimpleNamespace(dte_range=[21, 45]))
        assert rs.live_range_spec(s2)[0] == load_settings().ml.range_threshold_pct

    def test_the_alias_is_the_schema_default(self):
        from ait.ml import range_spec as rs
        assert rs.LIVE_RANGE_THRESHOLD_PCT == MLConfig().range_threshold_pct == 0.05
        with pytest.raises(AttributeError):
            rs.NOT_A_THING  # noqa: B018

    def test_without_a_config_it_is_the_schema_default(self, monkeypatch):
        import ait.config.settings as st
        from ait.ml import range_spec as rs
        monkeypatch.setattr(st, "load_settings", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no config")))
        assert rs.live_range_spec()[0] == 0.05


class TestRangePredictor:
    def _rp(self, **kw):
        from ait.ml.range_predictor import RangePredictor
        return RangePredictor(**kw)

    def test_band_and_edge_follow_the_ml_config_and_explicit_wins(self, tmp_path):
        moved = _ml(range_threshold_pct=0.08, range_min_edge_over_baseline=0.12)
        p = self._rp(ml_config=moved, model_dir=tmp_path)
        assert (p._threshold, p._spec_threshold, p._min_edge_over_baseline) == (0.08, 0.08, 0.12)
        q = self._rp(ml_config=moved, model_dir=tmp_path, threshold_pct=0.03, min_edge_over_baseline=0.2)
        assert (q._threshold, q._min_edge_over_baseline) == (0.03, 0.2)
        d = self._rp(model_dir=tmp_path)
        assert (d._threshold, d._min_edge_over_baseline) == (0.05, 0.05)

    def test_the_training_floor_is_the_ml_config_and_explicit_wins(self, tmp_path):
        assert self._rp(ml_config=_ml(min_training_samples=77), model_dir=tmp_path)._min_training_samples == 77
        assert self._rp(ml_config=_ml(min_training_samples=77), model_dir=tmp_path, min_training_samples=5)._min_training_samples == 5

    def test_the_edge_floor_gates_predictions(self, tmp_path):
        """CV balanced accuracy 0.58 = edge 0.08: dropped under a 0.12 floor, let through under 0.05 (then it fails on the
        missing scaler, which is logged; the dropped one never gets there)."""
        def run(floor):
            p = self._rp(ml_config=_ml(range_min_edge_over_baseline=floor), model_dir=tmp_path)
            p._symbol_models = {"SPY": {"cv_scores": {"xgboost": 0.58}, "models": {}, "scaler": None, "feature_names": []}}
            with capture_logs() as events:
                try:
                    p.predict_from_features(pd.Series(dtype=float), "SPY")
                except Exception:        # noqa: BLE001 - only reaching (or not reaching) the scaler matters here
                    pass
            return any(e["event"] == "range_scaling_failed" for e in events)
        assert run(0.12) is False and run(0.05) is True

    def test_the_walk_forward_folds_follow_the_ml_config(self, tmp_path):
        p = self._rp(ml_config=_ml(binary_cv_splits=3, binary_cv_min_gap_days=12), model_dir=tmp_path, horizon_days=4)
        splits = p._walk_forward_split(400)
        assert len(splits) == 3
        assert all(v[0] - t[-1] - 1 == 12 for t, v in splits)            # gap = max(12, horizon 4)
        assert len(self._rp(model_dir=tmp_path, horizon_days=4)._walk_forward_split(400)) == 4
        long_h = self._rp(ml_config=_ml(binary_cv_min_gap_days=12), model_dir=tmp_path, horizon_days=20)
        assert all(v[0] - t[-1] - 1 == 20 for t, v in long_h._walk_forward_split(400))   # the horizon still wins when longer

    def test_class_balance_guards_follow_the_ml_config(self, tmp_path):
        df = _walk()
        with capture_logs() as hard:
            out = self._rp(ml_config=_ml(min_class_samples=5000), model_dir=tmp_path, horizon_days=5).train(df)
        assert out == {} and any(e["event"] == "range_single_class" for e in hard)
        with capture_logs() as band:
            self._rp(ml_config=_ml(label_balance_min=0.45, label_balance_max=0.55), model_dir=tmp_path, horizon_days=5).train(df)
        assert any(e["event"] == "range_imbalanced" for e in band)
        with capture_logs() as quiet:
            self._rp(ml_config=_ml(), model_dir=tmp_path, horizon_days=5).train(df)
        assert not any(e["event"] in ("range_single_class", "range_imbalanced") for e in quiet)

    def test_a_pre_spec_artifact_is_compared_as_the_old_band_whatever_the_setting(self, tmp_path):
        """range.pkl files written before the spec was recorded have no threshold/horizon key and were trained at +/-5%/30d."""
        (tmp_path / "range.pkl").write_bytes(pickle.dumps({"symbol_models": {}}))
        same = self._rp(ml_config=_ml(), model_dir=tmp_path, horizon_days=30)
        same.load_models()
        assert same._spec_mismatch is False
        moved = self._rp(ml_config=_ml(range_threshold_pct=0.06), model_dir=tmp_path, horizon_days=30)
        moved.load_models()
        assert moved._spec_mismatch is True                              # a changed setting flags the old artifact for retraining


class TestVolMagnitudePredictor:
    def _vp(self, **kw):
        from ait.ml.vol_magnitude_predictor import VolMagnitudePredictor
        return VolMagnitudePredictor(**kw)

    def test_move_horizon_and_edge_follow_the_ml_config_and_explicit_wins(self, tmp_path):
        moved = _ml(vol_magnitude_threshold_pct=0.09, vol_magnitude_horizon_days=12, vol_magnitude_min_edge_over_baseline=0.2)
        p = self._vp(ml_config=moved, model_dir=tmp_path)
        assert (p._threshold, p._horizon, p._min_edge_over_baseline) == (0.09, 12, 0.2)
        q = self._vp(ml_config=moved, model_dir=tmp_path, threshold_pct=0.04, horizon_days=7)
        assert (q._threshold, q._horizon) == (0.04, 7)
        d = self._vp(model_dir=tmp_path)
        assert (d._threshold, d._horizon, d._min_edge_over_baseline) == (0.07, 30, 0.10)

    def test_the_edge_floor_gates_predictions(self, tmp_path):
        """A model whose CV balanced accuracy is 0.58 (edge 0.08) is dropped under a 0.10 floor and used under 0.05."""
        def run(floor):
            p = self._vp(ml_config=_ml(vol_magnitude_min_edge_over_baseline=floor), model_dir=tmp_path)
            p._symbol_models = {"SPY": {"cv_scores": {"xgboost": 0.58}, "models": {}, "scaler": None, "feature_names": []}}
            with capture_logs() as events:
                try:
                    p.predict(_walk(60), "SPY")
                except Exception:        # noqa: BLE001 - past the gate there is no real model; only the gate matters here
                    pass
            return any(e["event"] == "vol_mag_no_edge" for e in events)
        assert run(0.10) is True and run(0.05) is False

    def test_folds_and_balance_follow_the_ml_config(self, tmp_path):
        p = self._vp(ml_config=_ml(binary_cv_splits=3, binary_cv_min_gap_days=40), model_dir=tmp_path, horizon_days=10)
        splits = p._walk_forward_split(500)
        assert len(splits) == 3 and all(v[0] - t[-1] - 1 == 40 for t, v in splits)
        with capture_logs() as hard:
            out = self._vp(ml_config=_ml(min_class_samples=5000), model_dir=tmp_path, threshold_pct=0.03, horizon_days=5).train(_walk())
        assert out == {} and any(e["event"] == "vol_mag_single_class" for e in hard)
        with capture_logs() as band:
            self._vp(ml_config=_ml(label_balance_min=0.01, label_balance_max=0.3), model_dir=tmp_path,
                     threshold_pct=0.03, horizon_days=5).train(_walk())
        assert any(e["event"] == "vol_mag_imbalanced" for e in band)

    def test_a_pre_spec_artifact_is_compared_as_the_old_move(self, tmp_path):
        (tmp_path / "vol_magnitude.pkl").write_bytes(pickle.dumps({"symbol_models": {}}))
        same = self._vp(ml_config=_ml(), model_dir=tmp_path, threshold_pct=0.07, horizon_days=30)
        same.load_models()
        assert same._spec_mismatch is False
        moved = self._vp(ml_config=_ml(), model_dir=tmp_path, threshold_pct=0.09, horizon_days=30)
        moved.load_models()
        assert moved._spec_mismatch is True


class TestDirectionEnsemble:
    def _dp(self, **kw):
        from ait.ml.ensemble import DirectionPredictor
        return DirectionPredictor(_ml(**kw), persist_artifacts=False)

    def test_label_threshold_and_horizon(self):
        close = pd.Series(np.cumprod(1 + np.array([0.004] * 40)))        # a steady +0.4%/day climb: +2.0% over 5 days
        base = self._dp()._create_labels(close)
        assert set(base.dropna().unique()) == {2.0} and base.iloc[-5:].isna().all() and base.iloc[:-5].notna().all()
        wide = self._dp(direction_label_threshold_pct=0.05)._create_labels(close)
        assert set(wide.dropna().unique()) == {1.0}                      # +2% no longer clears a 5% band
        short = self._dp(direction_label_horizon_days=2)._create_labels(close)
        assert short.iloc[-2:].isna().all() and short.iloc[:-2].notna().all()        # the horizon sets how many rows are unknowable
        assert set(short.dropna().unique()) == {1.0}                         # +0.8% over 2 days is neutral under the 1% band

    def test_walk_forward_geometry(self):
        base = self._dp()._walk_forward_split(400)
        assert len(base) == 5 and all(v[0] - t[-1] - 1 == 5 for t, v in base)
        assert all(len(v) == max(20, 400 // 7) for _, v in base)
        moved = self._dp(direction_cv_splits=3, direction_cv_gap_days=9, cv_min_val_rows=60)._walk_forward_split(400)
        assert len(moved) == 3 and all(v[0] - t[-1] - 1 == 9 for t, v in moved) and all(len(v) == max(60, 400 // 5) for _, v in moved)
        # a bigger training minimum drops the earliest fold (its training part is 110 rows) ...
        assert len(self._dp(cv_min_train_rows=120)._walk_forward_split(400)) == 4
        # ... and with fewer than two usable folds the split falls back to sklearn's TimeSeriesSplit (no purge gap)
        fallback = self._dp(cv_min_train_rows=300)._walk_forward_split(400)
        assert all(v[0] - t[-1] - 1 == 0 for t, v in fallback)
        # the smallest validation fold: 400 // 7 = 57 rows unless the floor is higher
        assert all(len(v) == 100 for _, v in self._dp(cv_min_val_rows=100)._walk_forward_split(400))
        # explicit arguments win
        assert len(self._dp()._walk_forward_split(400, n_splits=2, gap=1)) == 2

    def test_a_partial_instance_uses_the_schema_defaults(self):
        from ait.ml.ensemble import DirectionPredictor
        bare = DirectionPredictor.__new__(DirectionPredictor)
        assert len(bare._walk_forward_split(400)) == 5


class TestTrainer:
    def test_rollback_margin(self):
        from ait.ml.trainer import ModelTrainer
        prev, new = {"a": 0.60}, {"a": 0.53}
        assert ModelTrainer._should_rollback(prev, new) is True               # 7 points down: past the default 5
        assert ModelTrainer._should_rollback(prev, new, 0.10) is False        # ... but not past a 10-point margin
        assert ModelTrainer._should_rollback(prev, {"a": 0.57}, 0.02) is True

    def test_the_rollback_margin_follows_the_trainer_config(self):
        from ait.ml.trainer import ModelTrainer
        t = ModelTrainer.__new__(ModelTrainer)
        t._config = _ml(rollback_margin=0.20)
        assert t._ml("rollback_margin") == 0.20
        t._config = SimpleNamespace()
        assert t._ml("rollback_margin") == 0.05                              # a stub config degrades to the schema

    def test_the_default_margin_is_the_loaded_setting(self, monkeypatch):
        import ait.config.settings as st
        from ait.ml.trainer import ModelTrainer
        s = load_settings()
        monkeypatch.setattr(st, "_FROZEN", s.model_copy(update={"ml": s.ml.model_copy(update={"rollback_margin": 0.20})}))
        assert ModelTrainer._should_rollback({"a": 0.60}, {"a": 0.53}) is False

    def test_the_hyperparameter_search_uses_trials_seed_and_proxy_seed(self, monkeypatch, tmp_path):
        import optuna
        import xgboost

        import ait.ml.trainer as tr
        from ait.ml.ensemble import DirectionPredictor

        seen: dict = {"seeds": [], "random_states": []}
        real_sampler, real_clf = optuna.samplers.TPESampler, xgboost.XGBClassifier

        class SpySampler(real_sampler):
            def __init__(self, *a, **k):
                seen["seeds"].append(k.get("seed"))
                super().__init__(*a, **k)

        class SpyClf:
            def __init__(self, **kw):
                seen["random_states"].append(kw.get("random_state"))
                raise RuntimeError("stop: only the constructor arguments matter")

        monkeypatch.setattr(optuna.samplers, "TPESampler", SpySampler)
        monkeypatch.setattr(xgboost, "XGBClassifier", SpyClf)
        store = SimpleNamespace(load=lambda symbol: _walk(260, 0.01))
        counts = {}
        real_create = optuna.create_study

        def create(**kw):
            study = real_create(**kw)
            orig = study.optimize

            def opt(fn, n_trials=None, **k):
                counts["n"] = n_trials
                return orig(fn, n_trials=1, **k)          # one pruned trial is enough: the constructor arguments are captured
            study.optimize = opt
            return study
        monkeypatch.setattr(optuna, "create_study", create)

        def run(**ml_kw):
            seen["seeds"].clear(); seen["random_states"].clear(); counts.clear()
            cfg = _ml(retrain_optuna_trials=7, retrain_optuna_seed=11, **ml_kw)
            t = tr.ModelTrainer(cfg, DirectionPredictor(cfg, persist_artifacts=False), None, store)
            t._apply_optimized_hyperparams(["SPY"])
            return dict(counts), list(seen["seeds"]), list(seen["random_states"])

        n, seeds, states = run(hyperparams=TreeHyperparamsConfig(n_estimators=10, max_depth=2, random_state=99))
        assert n == {"n": 7} and seeds == [11] and states and set(states) == {99}      # trials, sampler seed, proxy seed
        _, _, unseeded = run(hyperparams=TreeHyperparamsConfig(n_estimators=10, max_depth=2, random_state=None))
        assert set(unseeded) == {None}

    def test_train_all_symbols_rolls_back_by_the_configured_margin(self):
        """A retrain that costs 7 points of CV accuracy is rolled back under the default 5-point margin, kept under 10."""
        import asyncio

        import ait.ml.trainer as tr

        def run(margin):
            calls = []
            pred = SimpleNamespace(model_version="v1", cv_scores={"a": 0.60}, train=lambda df, symbol, market_context: {"gbm": 0.53},
                                   rollback=lambda v: calls.append(v))
            async def ctx():
                return {}
            async def hist(symbol, days):
                return _walk(30)
            t = tr.ModelTrainer.__new__(tr.ModelTrainer)
            t._config = _ml(rollback_margin=margin)
            t._predictor, t._range_predictor, t._vol_mag_predictor = pred, None, None
            t._market_data = SimpleNamespace(get_historical=hist)
            t._store = SimpleNamespace(save=lambda *a, **k: None)
            t._drift_detector = SimpleNamespace(acknowledge_retrain=lambda: None)
            t.check_range_spec = lambda: True
            t._fetch_market_context = ctx
            asyncio.run(t.train_all_symbols(["SPY"]))
            return calls
        assert run(0.05) == ["v1"] and run(0.10) == []

    def test_train_all_symbols_defers_the_trial_count_to_the_search(self, monkeypatch):
        import asyncio

        import ait.ml.trainer as tr
        t = tr.ModelTrainer.__new__(tr.ModelTrainer)
        got = {}
        monkeypatch.setattr(t, "_apply_optimized_hyperparams", lambda symbols, n_trials=None: got.setdefault("n", n_trials), raising=False)
        t._predictor = SimpleNamespace(model_version="v", cv_scores={})
        with pytest.raises(Exception):
            asyncio.run(t.train_all_symbols(["SPY"], optimize_hyperparams=True))
        assert got["n"] is None                                              # resolved to ml.retrain_optuna_trials inside the search


class TestOrchestratorWiring:
    def test_the_real_init_builds_the_vol_magnitude_predictor_from_the_settings(self, tmp_path, monkeypatch):
        """Through the hot-path rig's REAL TradingOrchestrator.__init__ against a copy of config.yaml with the two keys moved."""
        from tests.test_hot_path_smoke import CONFIG_PATH, build_smoke_orchestrator
        cfg = tmp_path / "config.yaml"
        text = CONFIG_PATH.read_text()
        assert "vol_magnitude_threshold_pct: 0.07" in text and "vol_magnitude_horizon_days: 30" in text
        cfg.write_text(text.replace("vol_magnitude_threshold_pct: 0.07", "vol_magnitude_threshold_pct: 0.09")
                       .replace("vol_magnitude_horizon_days: 30", "vol_magnitude_horizon_days: 12"))
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.chdir(sandbox)
        rig = build_smoke_orchestrator(cfg)
        try:
            vm = rig.orch._vol_mag_predictor
            assert (vm._threshold, vm._horizon) == (0.09, 12)
        finally:
            rig.restore()

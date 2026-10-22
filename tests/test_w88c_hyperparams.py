"""W88c: model hyperparameters are settings, recorded with every model, and shown in the parity manifest.

Two halves, as in W88b: AT DEFAULTS the models are built exactly as before (tests/test_w87a_hygiene.py pins the
recorded constructor keywords), and a CHANGED field reaches the constructor of every family that is supposed to read it.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ait.config.settings import MLConfig, TreeHyperparamsConfig, load_settings
from ait.ml import hyperparams as hp

ROOT = Path(__file__).resolve().parents[1]

FIELDS = {"n_estimators": 77, "max_depth": 2, "learning_rate": 0.2, "subsample": 0.5, "colsample_bytree": 0.6,
          "random_state": 7}


def _ohlcv(days: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, days))
    return pd.DataFrame({"Open": close, "High": close * (1 + abs(rng.normal(0, .005, days))),
                         "Low": close * (1 - abs(rng.normal(0, .005, days))), "Close": close,
                         "Volume": rng.integers(1_000_000, 10_000_000, days).astype(float)}, index=idx)


@pytest.fixture
def recorded(monkeypatch):
    import lightgbm
    import xgboost
    seen = {"xgb": [], "lgb": []}
    real_x, real_l = xgboost.XGBClassifier, lightgbm.LGBMClassifier
    monkeypatch.setattr(xgboost, "XGBClassifier", lambda **kw: (seen["xgb"].append(dict(kw)), real_x(**kw))[1])
    monkeypatch.setattr(lightgbm, "LGBMClassifier", lambda **kw: (seen["lgb"].append(dict(kw)), real_l(**kw))[1])
    return seen


def _meta_frame():
    from ait.ml.meta_label import META_FEATURES
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f: rng.normal(size=80) for f in META_FEATURES})
    df["profitable"] = (rng.random(80) > 0.5).astype(int)
    return df


def _frozen(monkeypatch, **ml):
    import ait.config.settings as st
    s = load_settings()
    monkeypatch.setattr(st, "_FROZEN", s.model_copy(update={"ml": s.ml.model_copy(update=ml)}))


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

class TestSchema:
    def test_defaults_are_the_old_literals(self):
        m = MLConfig()
        assert m.hyperparams.model_dump() == dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                                                  colsample_bytree=0.8, random_state=42)
        assert m.meta_label_hyperparams.model_dump() == dict(n_estimators=100, max_depth=3, learning_rate=0.05,
                                                             subsample=0.8, colsample_bytree=0.8, random_state=None)

    def test_config_yaml_lists_them_equal_to_the_defaults(self):
        s = load_settings().ml
        assert s.hyperparams == MLConfig().hyperparams and s.meta_label_hyperparams == MLConfig().meta_label_hyperparams

    @pytest.mark.parametrize("kw", [{"n_estimators": 5}, {"max_depth": 0}, {"learning_rate": 0.0}, {"subsample": 1.5},
                                    {"colsample_bytree": 0.0}, {"random_state": -1}, {"bogus": 1}])
    def test_bad_values_fail_at_load(self, kw):
        with pytest.raises(ValidationError):
            TreeHyperparamsConfig(**kw)

    def test_the_view_shows_them_with_their_origin(self):
        from ait.config.research import settings_origin
        got = settings_origin("config.yaml")
        assert got["ml.hyperparams.n_estimators"] == (200, "config.yaml")
        assert got["ml.meta_label_hyperparams.random_state"] == (None, "config.yaml")

    def test_a_study_can_change_them_for_one_run(self, tmp_path):
        from ait.config.research import load_research_settings
        p = tmp_path / "s.yaml"
        p.write_text("name: s\nquestion: q\nrule: exploratory\nsettings: {ml: {hyperparams: {max_depth: 3}}}\n")
        s = load_research_settings("config.yaml", p)
        assert s.ml.hyperparams.max_depth == 3 and s.ml.hyperparams.n_estimators == 200


# --------------------------------------------------------------------------
# the builders
# --------------------------------------------------------------------------

class TestBuilders:
    def test_every_field_reaches_every_builder(self):
        cfg = TreeHyperparamsConfig(**FIELDS)
        for kw in (hp.xgb_multiclass_kwargs(cfg), hp.lgbm_multiclass_kwargs(cfg), hp.xgb_binary_kwargs(2.0, cfg),
                   hp.lgbm_binary_kwargs(cfg), hp.meta_label_xgb_kwargs(2.0, cfg)):
            for k, v in FIELDS.items():
                assert kw[k] == v, (k, kw)

    def test_a_null_seed_leaves_the_key_out(self):
        cfg = TreeHyperparamsConfig(random_state=None)
        assert "random_state" not in hp.xgb_binary_kwargs(1.0, cfg)

    def test_the_default_resolves_the_loaded_settings(self, monkeypatch):
        _frozen(monkeypatch, hyperparams=TreeHyperparamsConfig(max_depth=9))
        assert hp.xgb_multiclass_kwargs()["max_depth"] == 9
        assert hp.xgb_multiclass_kwargs(TreeHyperparamsConfig(max_depth=3))["max_depth"] == 3      # explicit wins

    def test_the_meta_label_profile_is_separate(self, monkeypatch):
        _frozen(monkeypatch, hyperparams=TreeHyperparamsConfig(max_depth=9))
        kw = hp.meta_label_xgb_kwargs(2.0)
        assert kw["max_depth"] == 3 and kw["n_estimators"] == 100 and "random_state" not in kw

    def test_scale_pos_weight_is_passed_through(self):
        assert hp.xgb_binary_kwargs(3.5)["scale_pos_weight"] == 3.5
        assert hp.meta_label_xgb_kwargs(1.25)["scale_pos_weight"] == 1.25


# --------------------------------------------------------------------------
# the families read their config
# --------------------------------------------------------------------------

class TestFamiliesReadTheirConfig:
    def test_the_direction_ensemble(self, recorded):
        from ait.ml.ensemble import DirectionPredictor
        cfg = MLConfig(hyperparams=TreeHyperparamsConfig(**FIELDS))        # every field changed at once
        DirectionPredictor(cfg, persist_artifacts=False).train(_ohlcv(), symbol="QQQ")
        for k, v in FIELDS.items():
            assert recorded["xgb"][0][k] == v and recorded["lgb"][0][k] == v, k

    @pytest.mark.parametrize("mod, cls, thr", [("range_predictor", "RangePredictor", 0.05),
                                              ("vol_magnitude_predictor", "VolMagnitudePredictor", 0.04)])
    def test_the_binary_predictors_by_constructor(self, recorded, mod, cls, thr):
        import importlib
        predictor = getattr(importlib.import_module(f"ait.ml.{mod}"), cls)(
            threshold_pct=thr, horizon_days=10, hyperparams=TreeHyperparamsConfig(**FIELDS))
        predictor.train(_ohlcv(), symbol="QQQ")
        for k, v in FIELDS.items():
            assert recorded["xgb"][0][k] == v and recorded["lgb"][0][k] == v, k

    def test_the_binary_predictors_by_loaded_settings(self, recorded, monkeypatch):
        from ait.ml.range_predictor import RangePredictor
        _frozen(monkeypatch, hyperparams=TreeHyperparamsConfig(max_depth=3, n_estimators=50))
        RangePredictor(threshold_pct=0.05, horizon_days=10).train(_ohlcv(), symbol="QQQ")
        assert recorded["xgb"][0]["max_depth"] == 3 and recorded["lgb"][0]["n_estimators"] == 50

    def test_the_meta_labeler_by_constructor(self, recorded):
        from ait.ml.meta_label import MetaLabeler
        fields = {k: v for k, v in FIELDS.items() if k != "random_state"}
        MetaLabeler(hyperparams=TreeHyperparamsConfig(random_state=None, **fields)).train(_meta_frame())
        for k, v in fields.items():
            assert recorded["xgb"][0][k] == v, k
        assert "random_state" not in recorded["xgb"][0]

    def test_the_meta_labeler_can_be_seeded(self, recorded):
        from ait.ml.meta_label import MetaLabeler
        MetaLabeler(hyperparams=TreeHyperparamsConfig(random_state=11)).train(_meta_frame())
        assert recorded["xgb"][0]["random_state"] == 11

    def test_the_meta_labeler_by_loaded_settings(self, recorded, monkeypatch):
        from ait.ml.meta_label import MetaLabeler
        _frozen(monkeypatch, meta_label_hyperparams=TreeHyperparamsConfig(n_estimators=33, max_depth=2, random_state=None))
        MetaLabeler().train(_meta_frame())
        assert recorded["xgb"][0]["n_estimators"] == 33 and recorded["xgb"][0]["max_depth"] == 2

    def test_a_stub_ensemble_config_degrades_to_the_loaded_settings(self, recorded):
        from types import SimpleNamespace

        from ait.ml.ensemble import DirectionPredictor
        real = MLConfig()
        stub = SimpleNamespace(**{k: getattr(real, k) for k in ("ensemble_weights", "min_training_samples", "lookback_days",
                                                                "retrain_interval_days")})
        p = DirectionPredictor(stub, persist_artifacts=False)
        assert p._hyperparams() == load_settings().ml.hyperparams


# --------------------------------------------------------------------------
# the model records what it was trained with
# --------------------------------------------------------------------------

class TestArtifactSpec:
    @pytest.fixture
    def warnings_seen(self, monkeypatch):
        import ait.utils.logging as ulog
        seen = []

        class Log:
            def warning(self, event, **kw):
                seen.append((event, kw))

            def __getattr__(self, name):
                return lambda *a, **k: None
        real = ulog.get_logger
        monkeypatch.setattr(ulog, "get_logger", lambda name: Log() if name == "ml.hyperparams" else real(name))
        return seen

    def test_the_spec_and_its_differences(self):
        a = hp.model_spec(TreeHyperparamsConfig())
        b = hp.model_spec(TreeHyperparamsConfig(max_depth=3, n_estimators=10))
        assert a["hyperparams"]["max_depth"] == 5 and a["feature_spec_hash"] is None
        assert hp.spec_differences(a, a) == [] and hp.spec_differences(None, a) == [] and hp.spec_differences({}, a) == []
        diffs = hp.spec_differences(a, b)
        assert len(diffs) == 2 and any("max_depth" in d for d in diffs)
        assert hp.spec_differences({**a, "feature_spec_hash": "abc"}, a) != []         # the W88d hook

    def test_warn_logs_only_on_a_real_difference(self, warnings_seen):
        a = hp.model_spec(TreeHyperparamsConfig())
        assert hp.warn_on_spec_mismatch("x", a, a) is False and hp.warn_on_spec_mismatch("x", None, a) is False
        assert warnings_seen == []
        assert hp.warn_on_spec_mismatch("x", a, hp.model_spec(TreeHyperparamsConfig(max_depth=3))) is True
        assert warnings_seen[0][0] == "model_spec_mismatch" and warnings_seen[0][1]["artifact"] == "x"

    def test_the_ensemble_saves_the_spec_and_warns_when_the_config_moved(self, tmp_path, warnings_seen):
        from ait.ml.ensemble import DirectionPredictor
        DirectionPredictor(MLConfig(), model_dir=tmp_path).train(_ohlcv(), symbol="QQQ")
        saved = pickle.loads((tmp_path / "ensemble.pkl").read_bytes())
        assert saved["spec"] == hp.model_spec(TreeHyperparamsConfig())
        same = DirectionPredictor(MLConfig(), model_dir=tmp_path)
        assert same.load_models() and same._hp_spec_mismatch is False and warnings_seen == []
        moved = DirectionPredictor(MLConfig(hyperparams=TreeHyperparamsConfig(max_depth=3)), model_dir=tmp_path)
        assert moved.load_models() is True                                  # served, not refused
        assert moved._hp_spec_mismatch is True and warnings_seen[0][0] == "model_spec_mismatch"

    def test_a_legacy_artifact_without_a_spec_loads_silently(self, tmp_path, warnings_seen):
        from ait.ml.ensemble import DirectionPredictor
        DirectionPredictor(MLConfig(), model_dir=tmp_path).train(_ohlcv(), symbol="QQQ")
        data = pickle.loads((tmp_path / "ensemble.pkl").read_bytes())
        data.pop("spec")
        (tmp_path / "ensemble.pkl").write_bytes(pickle.dumps(data))
        p = DirectionPredictor(MLConfig(hyperparams=TreeHyperparamsConfig(max_depth=3)), model_dir=tmp_path)
        assert p.load_models() is True and warnings_seen == []

    def test_the_real_live_ensemble_artifact_still_loads(self, warnings_seen):
        live = ROOT / "models" / "ensemble.pkl"
        if not live.exists():
            pytest.skip("no live artifact in this checkout")
        from ait.ml.ensemble import DirectionPredictor
        p = DirectionPredictor(MLConfig(), model_dir=live.parent, persist_artifacts=False)
        assert p.load_models() is True and warnings_seen == []            # pre-W88c artifact: no spec, accepted

    @pytest.mark.parametrize("mod, cls, file, thr", [("range_predictor", "RangePredictor", "range.pkl", 0.05),
                                                    ("vol_magnitude_predictor", "VolMagnitudePredictor",
                                                     "vol_magnitude.pkl", 0.04)])
    def test_the_binary_predictors_round_trip_the_spec(self, tmp_path, warnings_seen, mod, cls, file, thr):
        import importlib
        C = getattr(importlib.import_module(f"ait.ml.{mod}"), cls)
        a = C(threshold_pct=thr, horizon_days=10, model_dir=tmp_path)
        a.train(_ohlcv(), symbol="QQQ")
        a._save_models(tmp_path)
        assert pickle.loads((tmp_path / file).read_bytes())["spec"] == hp.model_spec(TreeHyperparamsConfig())
        assert C(threshold_pct=thr, horizon_days=10, model_dir=tmp_path).load_models() and warnings_seen == []
        moved = C(threshold_pct=thr, horizon_days=10, model_dir=tmp_path, hyperparams=TreeHyperparamsConfig(max_depth=3))
        assert moved.load_models() and warnings_seen and warnings_seen[0][0] == "model_spec_mismatch"

    def test_the_meta_labeler_load_warns_on_a_moved_profile_and_stays_silent_on_a_legacy_one(self, tmp_path,
                                                                                            monkeypatch, warnings_seen):
        import ait.ml.meta_label as ml
        monkeypatch.setattr(ml, "MODEL_DIR", tmp_path)
        m = ml.MetaLabeler()
        m.train(_meta_frame())
        m.save_to_path(tmp_path / "meta_label.pkl")
        assert ml.MetaLabeler().load_model() is True and warnings_seen == []
        moved = ml.MetaLabeler(hyperparams=TreeHyperparamsConfig(n_estimators=50, max_depth=3, random_state=None))
        assert moved.load_model() is True and warnings_seen[0][0] == "model_spec_mismatch"
        warnings_seen.clear()
        legacy = pickle.loads((tmp_path / "meta_label.pkl").read_bytes())
        legacy.pop("spec")
        (tmp_path / "meta_label.pkl").write_bytes(pickle.dumps(legacy))
        assert moved.load_model() is True and warnings_seen == []

    def test_the_meta_labeler_saves_the_spec(self, tmp_path):
        from ait.ml.meta_label import MetaLabeler
        m = MetaLabeler()
        m.train(_meta_frame())
        m.save_to_path(tmp_path / "m.pkl")
        saved = pickle.loads((tmp_path / "m.pkl").read_bytes())
        assert saved["spec"]["hyperparams"]["n_estimators"] == 100 and saved["spec"]["hyperparams"]["random_state"] is None


# --------------------------------------------------------------------------
# manifest and run record
# --------------------------------------------------------------------------

def _manifest(monkeypatch, research_ml=None):
    import run_backtest as rb
    s = load_settings()
    rs = s if research_ml is None else s.model_copy(update={"ml": s.ml.model_copy(update=research_ml)})
    monkeypatch.setattr(rb, "_research_settings", lambda: rs)
    monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
    return rb.build_parity_manifest(rb.parse_args())


class TestManifestAndRecord:
    def test_equal_hyperparameters_show_no_warning(self, monkeypatch):
        m = _manifest(monkeypatch)
        assert m["live"]["ml_hyperparams"] == m["backtest"]["ml_hyperparams"] == MLConfig().hyperparams.model_dump()
        assert not [w for w in m["parity_warnings"] if w.startswith("ml_")]

    def test_a_research_difference_is_a_warning(self, monkeypatch):
        m = _manifest(monkeypatch, {"hyperparams": TreeHyperparamsConfig(max_depth=3)})
        warned = [w for w in m["parity_warnings"] if w.startswith("ml_hyperparams")]
        assert len(warned) == 1 and "max_depth" in warned[0]
        m2 = _manifest(monkeypatch, {"meta_label_hyperparams": TreeHyperparamsConfig(n_estimators=50, max_depth=3, random_state=None)})
        assert any(w.startswith("ml_meta_label_hyperparams") for w in m2["parity_warnings"])

    def test_the_run_record_seed_follows_the_settings(self):
        from ait.backtesting import run_record as rr
        s = load_settings()
        moved = s.model_copy(update={"ml": s.ml.model_copy(update={"hyperparams": TreeHyperparamsConfig(random_state=5)})})
        assert rr.seed_inventory(None, moved)["ml_random_state"] == 5
        assert rr.seed_inventory(None, s)["ml_random_state"] == 42
        assert rr.seed_inventory()["ml_random_state"] == 42

"""W98: every model feature has an on/off switch (``ml.features.<group>.use``) listed in ``config.yaml``.

What this file proves:

* at the default switches nothing moved: the model feature list (with and without the 26 intraday names), its order and the
  spec hash are the ones recorded BEFORE the change (``tests/data/w98_golden.json``), and ``compute()`` is untouched;
* ``config.yaml`` lists every feature the code can produce (a class guard: the next feature added to the catalogue cannot be
  missing from the file), and the catalogue is the feature list the engine really computes;
* a switch removes / adds exactly its column from the inputs of every model (trained through the real code path), is keyed by
  role so a window change never renames it, and enters the hash only when it differs from its default;
* a typo, a feature in the wrong group, or an all-off selection is a load error; a study can switch features for one run.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from ait.config import feature_names as fn
from ait.config.settings import FeaturesConfig, MLConfig, load_settings
from ait.ml.feature_spec import FeatureSpec, default_spec_hash, derive_names, spec_from_config
from ait.ml.features import FeatureEngine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "data"))
import _w88d_golden_inputs as gi  # noqa: E402
import _w97_golden_inputs as g97  # noqa: E402

GOLD = json.loads((ROOT / "tests" / "data" / "w98_golden.json").read_text())
ALL = [(g, r) for g, roles in fn.CATALOGUE.items() for r in roles]


def _cfg(use: dict | None = None, **windows) -> FeaturesConfig:
    """FeaturesConfig with ``use={"group.feature": bool}`` and ``windows={"group": {"field": value}}`` applied."""
    d = FeaturesConfig().model_dump()
    for key, val in (use or {}).items():
        grp, feat = key.split(".")
        d[grp]["use"][feat] = val
    for grp, fields in windows.items():
        d[grp].update(fields)
    return FeaturesConfig(**d)


def _spec(use: dict | None = None, **windows) -> FeatureSpec:
    return FeatureSpec.from_config(_cfg(use, **windows))


def _names(spec: FeatureSpec, vlmc: bool = False) -> list[str]:
    return spec.feature_names(list(fn.VLMC_NAMES) if vlmc else None)


@pytest.fixture
def fast_fractal(monkeypatch):
    monkeypatch.setattr(FeatureEngine, "_add_fractal_features", g97.stub_fractal)


@pytest.fixture(scope="module")
def frame():
    return gi.frames()["calm_700"]


# --------------------------------------------------------------------------------------------------------------------
# 1. at the defaults nothing changed (recorded BEFORE the change)
# --------------------------------------------------------------------------------------------------------------------

class TestDefaultsAreTheOldBehaviour:
    def test_the_feature_lists_and_the_hash_equal_the_recorded_ones(self):
        spec = spec_from_config(None)
        assert _names(spec) == GOLD["names"]
        assert _names(spec, vlmc=True) == GOLD["names_vlmc"]
        assert spec.spec_hash() == default_spec_hash() == GOLD["hash"] == "6e0e3f8f922f6b88"

    def test_the_engine_and_the_loaded_config_agree(self):
        fe = FeatureEngine()
        assert fe.get_feature_names() == GOLD["names"] and fe.get_feature_names(include_vlmc=True) == GOLD["names_vlmc"]
        assert fe.spec_hash() == GOLD["hash"]
        assert load_settings().ml.features.model_dump() == FeaturesConfig().model_dump()      # config.yaml == the defaults

    def test_the_template_is_derived_from_the_catalogue(self):
        assert len(fn._BASE_TEMPLATE) == len(GOLD["names"]) == 75
        names = derive_names(FeaturesConfig())
        assert [n.format_map(names) for n in fn._BASE_TEMPLATE] == GOLD["names"]

    def test_the_vlmc_class_attribute_is_the_catalogue_list(self):
        assert FeatureEngine.VLMC_FEATURE_NAMES == list(fn.VLMC_NAMES) and len(fn.VLMC_NAMES) == 26


# --------------------------------------------------------------------------------------------------------------------
# 2. the catalogue is complete and config.yaml lists all of it (class guards)
# --------------------------------------------------------------------------------------------------------------------

class TestTheCatalogueIsComplete:
    def test_config_yaml_lists_every_feature_of_every_group(self):
        live = yaml.safe_load((ROOT / "config.yaml").read_text())["ml"]["features"]
        assert set(live) == set(fn.CATALOGUE)
        for g, roles in fn.CATALOGUE.items():
            assert list(live[g]["use"]) == list(roles), f"ml.features.{g}.use must list the catalogue in order"
            assert live[g]["use"] == fn.DEFAULT_USE[g], g              # config.yaml ships the defaults

    def test_every_group_is_a_features_field_with_a_use_map(self):
        cfg = FeaturesConfig()
        for g in fn.CATALOGUE:
            assert getattr(cfg, g).use == fn.DEFAULT_USE[g], g
        assert set(fn.CATALOGUE) == set(FeaturesConfig.model_fields)

    def test_use_switches_returns_every_group_in_catalogue_order(self):
        cfg = FeaturesConfig()
        assert list(fn.use_switches(cfg)) == list(fn.CATALOGUE)
        assert all(fn.use_switches(cfg)[g] is getattr(cfg, g).use for g in fn.CATALOGUE)

    def test_a_feature_belongs_to_exactly_one_group(self):
        daily = [r for g, roles in fn.CATALOGUE.items() if g != "intraday" for r in roles]
        assert len(daily) == len(set(daily))
        assert not set(daily) & set(fn.VLMC_NAMES)

    def test_the_only_features_off_by_default_are_the_five_unused_columns(self):
        off = [(g, r) for g, u in fn.DEFAULT_USE.items() for r, on in u.items() if not on]
        assert off == [("market_structure", r) for r in
                       ("put_call_proxy", "vix_term_structure", "skew_proxy", "momentum_divergence", "range_compression")]

    def test_every_daily_catalogue_name_is_a_column_the_engine_computes(self, fast_fractal, frame):
        """The catalogue is not a wish list: each name (at the default windows) is a column of a real compute()."""
        names = derive_names(FeaturesConfig())
        cols = set(FeatureEngine().compute(frame).columns)
        missing = [names.get(r, r) for g, roles in fn.CATALOGUE.items() if g != "intraday" for r in roles
                   if names.get(r, r) not in cols]
        # the cross-asset / macro columns need market context; without it they are absent or neutral-filled by design
        assert not [m for m in missing if not m.startswith(("vix", "rel_", "spy_", "correlation", "us_", "yield", "dxy"))], missing

    def test_window_roles_are_exactly_the_window_bearing_columns(self):
        roles = set(derive_names(FeaturesConfig())) - set(fn._RAW_ROLES)
        assert fn._WINDOW_ROLES == roles and len(roles) == 27
        daily = {r for g, rs in fn.CATALOGUE.items() if g != "intraday" for r in rs}
        assert fn._WINDOW_ROLES <= daily


# --------------------------------------------------------------------------------------------------------------------
# 3. a switch removes / adds exactly its column
# --------------------------------------------------------------------------------------------------------------------

class TestSwitches:
    @pytest.mark.parametrize("group, feature", [(g, r) for g, r in ALL if g != "intraday"])
    def test_flipping_a_daily_feature_adds_or_removes_exactly_its_column(self, group, feature):
        on = fn.DEFAULT_USE[group][feature]
        spec = _spec({f"{group}.{feature}": not on})
        col = spec.names.get(feature, feature)
        base, got = GOLD["names"], _names(spec)
        assert (col in got) is (not on)
        assert [n for n in got if n != col] == [n for n in base if n != col]            # nothing else moved
        assert len(got) == len(base) + (1 if not on else -1)

    @pytest.mark.parametrize("feature", fn.VLMC_NAMES)
    def test_switching_an_intraday_feature_off_removes_it_only_when_intraday_is_used(self, feature):
        spec = _spec({f"intraday.{feature}": False})
        assert _names(spec) == GOLD["names"]                                             # not requested: no change
        assert _names(spec, vlmc=True) == [n for n in GOLD["names_vlmc"] if n != feature]

    def test_a_switched_on_column_is_placed_in_its_groups_slot(self):
        got = _names(_spec({"market_structure.skew_proxy": True}))
        i = got.index("skew_proxy")
        assert got[i - 1] == "distance_sma200" and got[i + 1] == "daily_return"

    def test_a_switch_is_keyed_by_role_and_follows_a_window_change(self):
        spec = _spec({"momentum.rsi_long": False}, momentum={"rsi_long": 21})
        got = _names(spec)
        assert "rsi_21" not in got and "rsi_14" not in got and "rsi_7" in got
        assert "rsi_21" in _names(_spec(momentum={"rsi_long": 21}))

    def test_a_partial_use_map_takes_the_defaults_for_the_rest(self):
        cfg = FeaturesConfig(fractal={"use": {"hurst_psd_divergence": False}})
        assert cfg.fractal.use == {**fn.DEFAULT_USE["fractal"], "hurst_psd_divergence": False}

    def test_the_column_is_still_computed_when_its_switch_is_off(self, fast_fractal, frame):
        """Switches choose model INPUTS; the frame (the meta-labeler's and the gates' source) is not touched."""
        base = FeatureEngine(spec_from_config(None)).compute(frame)
        off = FeatureEngine(_spec({"momentum.rsi_long": False, "trend.cross": False})).compute(frame)
        pd.testing.assert_frame_equal(base, off)
        assert "rsi_14" in off.columns and "rsi_14" not in FeatureEngine(_spec({"momentum.rsi_long": False})).get_feature_names()

    def test_a_switched_on_unused_column_is_a_real_computed_column(self, fast_fractal, frame):
        f = FeatureEngine(_spec({"market_structure.skew_proxy": True})).compute(frame)
        assert f["skew_proxy"].nunique() > 20


# --------------------------------------------------------------------------------------------------------------------
# 4. the hash: only a switch that differs from its default counts
# --------------------------------------------------------------------------------------------------------------------

class TestHash:
    def test_a_switch_at_its_default_does_not_move_the_hash(self):
        assert _spec({"momentum.rsi_long": True, "market_structure.skew_proxy": False}).spec_hash() == GOLD["hash"]

    def test_every_switch_changes_the_hash_when_flipped_and_no_two_collide(self):
        seen = {GOLD["hash"]}
        for g, r in ALL:
            h = _spec({f"{g}.{r}": not fn.DEFAULT_USE[g][r]}).spec_hash()
            assert h not in seen, (g, r)
            seen.add(h)

    def test_the_hash_separates_a_window_from_a_switch(self):
        assert _spec({"fractal.hurst_short": False}).spec_hash() != _spec(fractal={"hurst_short_window": 61}).spec_hash()

    def test_two_switches_combine_and_order_does_not_matter(self):
        a = _spec({"market_structure.skew_proxy": True, "fractal.hurst_short": False}).spec_hash()
        b = _spec({"fractal.hurst_short": False, "market_structure.skew_proxy": True}).spec_hash()
        assert a == b != _spec({"market_structure.skew_proxy": True}).spec_hash()


# --------------------------------------------------------------------------------------------------------------------
# 5. validation
# --------------------------------------------------------------------------------------------------------------------

class TestValidation:
    def test_an_unknown_feature_is_a_load_error_that_lists_the_valid_names(self):
        with pytest.raises(ValidationError, match="unknown feature.*hurst_psd_divergnce.*hurst_psd_divergence"):
            FeaturesConfig(fractal={"use": {"hurst_psd_divergnce": False}})

    def test_a_feature_in_the_wrong_group_is_refused(self):
        with pytest.raises(ValidationError, match="unknown feature"):
            FeaturesConfig(fractal={"use": {"rsi_long": False}})

    def test_a_column_name_is_not_a_role(self):
        with pytest.raises(ValidationError, match="unknown feature"):
            FeaturesConfig(momentum={"use": {"rsi_14": False}})

    def test_every_daily_feature_off_is_refused(self):
        d = FeaturesConfig().model_dump()
        for g, roles in fn.CATALOGUE.items():
            if g != "intraday":
                d[g]["use"] = {r: False for r in roles}
        with pytest.raises(ValidationError, match="every daily feature is switched off"):
            FeaturesConfig(**d)

    def test_a_switch_must_be_a_boolean(self):
        with pytest.raises(ValidationError):
            FeaturesConfig(fractal={"use": {"hurst_short": "sometimes"}})


# --------------------------------------------------------------------------------------------------------------------
# 6. the models train on the switched list (real code path) and a stale model is refused
# --------------------------------------------------------------------------------------------------------------------

@pytest.fixture
def criticals(monkeypatch):
    import ait.utils.logging as ulog
    seen = []

    class Log:
        def critical(self, event, **kw):
            seen.append((event, kw))

        def __getattr__(self, name):
            return lambda *a, **k: None
    real = ulog.get_logger
    monkeypatch.setattr(ulog, "get_logger", lambda name: Log() if name == "ml.hyperparams" else real(name))
    return seen


def _frozen(monkeypatch, cfg: FeaturesConfig):
    import ait.config.settings as st
    s = load_settings()
    monkeypatch.setattr(st, "_FROZEN", s.model_copy(update={"ml": s.ml.model_copy(update={"features": cfg})}))


@pytest.mark.usefixtures("fast_fractal")
class TestModels:
    OFF = {"momentum.rsi_long": False, "fractal.hurst_psd_divergence": False}
    ON = {"market_structure.skew_proxy": True}

    def test_the_direction_ensemble(self, tmp_path, criticals):
        from ait.ml.ensemble import DirectionPredictor
        cfg = _cfg({**self.OFF, **self.ON})
        DirectionPredictor(MLConfig(features=cfg), model_dir=tmp_path).train(gi.frames()["calm_700"].iloc[:300], symbol="QQQ")
        saved = pickle.loads((tmp_path / "ensemble.pkl").read_bytes())
        names = saved["feature_names"]
        assert "rsi_14" not in names and "hurst_psd_divergence" not in names and "skew_proxy" in names
        assert "rsi_7" in names and saved["spec"]["feature_spec_hash"] == FeatureSpec.from_config(cfg).spec_hash()
        # the same config serves it; the default config refuses it (trained under other inputs)
        assert DirectionPredictor(MLConfig(features=cfg), model_dir=tmp_path).load_models() is True and criticals == []
        stale = DirectionPredictor(MLConfig(), model_dir=tmp_path)
        assert stale.load_models() is False and criticals[0][0] == "model_feature_spec_mismatch"

    @pytest.mark.parametrize("mod, cls, thr", [("range_predictor", "RangePredictor", 0.03),
                                              ("vol_magnitude_predictor", "VolMagnitudePredictor", 0.03)])
    def test_the_binary_predictors_train_on_the_switched_list(self, tmp_path, monkeypatch, mod, cls, thr):
        import importlib
        C = getattr(importlib.import_module(f"ait.ml.{mod}"), cls)
        _frozen(monkeypatch, _cfg({**self.OFF, **self.ON}))
        p = C(threshold_pct=thr, horizon_days=10, model_dir=tmp_path)
        p.train(gi.frames()["calm_700"].iloc[:300], symbol="QQQ")
        names = p._feature_names
        assert "rsi_14" not in names and "hurst_psd_divergence" not in names and "skew_proxy" in names

    def test_the_tuner_reads_the_same_list(self, monkeypatch):
        _frozen(monkeypatch, _cfg(self.OFF))
        got = FeatureEngine().get_feature_names()
        assert "rsi_14" not in got and len(got) == 73

    def test_a_default_model_is_not_refused_at_the_default_switches(self, tmp_path, criticals):
        from ait.ml.ensemble import DirectionPredictor
        DirectionPredictor(MLConfig(), model_dir=tmp_path).train(gi.frames()["calm_700"].iloc[:300], symbol="QQQ")
        assert DirectionPredictor(MLConfig(features=_cfg({"momentum.rsi_long": True})), model_dir=tmp_path).load_models() is True
        assert criticals == []


# --------------------------------------------------------------------------------------------------------------------
# 7. research: a study switches features for one run, and the run says which
# --------------------------------------------------------------------------------------------------------------------

class TestResearch:
    def _study(self, tmp_path, body: str) -> Path:
        p = tmp_path / "s.yaml"
        p.write_text(f"name: s\nquestion: q\nrule: exploratory\nsettings: {body}\n")
        return p

    def test_a_study_switches_a_feature_and_a_window_in_one_run(self, tmp_path):
        from ait.config.research import load_research_settings
        p = self._study(tmp_path, "{ml: {features: {fractal: {use: {hurst_psd_divergence: false}}, "
                                  "market_structure: {use: {skew_proxy: true}}, momentum: {rsi_long: 21}}}}")
        s = load_research_settings("config.yaml", p)
        use = s.ml.features.fractal.use
        assert use["hurst_psd_divergence"] is False and use["hurst_short"] is True       # only the difference was written
        assert s.ml.features.market_structure.use["skew_proxy"] is True
        names = _names(spec_from_config(s.ml.features))
        assert "rsi_21" in names and "skew_proxy" in names and "hurst_psd_divergence" not in names

    def test_a_typo_in_a_study_is_refused(self, tmp_path):
        from ait.config.research import load_research_settings
        p = self._study(tmp_path, "{ml: {features: {fractal: {use: {hurst_psd_divergnce: false}}}}}")
        with pytest.raises(Exception, match="unknown feature"):
            load_research_settings("config.yaml", p)

    def test_the_study_origin_view_shows_a_switch_and_its_layer(self, tmp_path):
        from ait.config.research import settings_origin
        p = self._study(tmp_path, "{ml: {features: {market_structure: {use: {skew_proxy: true}}}}}")
        got = settings_origin("config.yaml", study=p)
        assert got["ml.features.market_structure.use.skew_proxy"][0] is True
        assert got["ml.features.market_structure.use.skew_proxy"][1] == "study"

    def test_the_overlay_does_not_copy_shared_feature_keys(self):
        research = yaml.safe_load((ROOT / "config_research.yaml").read_text())
        assert "features" not in (research.get("ml") or {})

    def test_the_run_record_carries_the_resolved_feature_list_and_the_hash(self):
        from ait.backtesting import run_record as rr
        cfg = _cfg({"momentum.rsi_long": False, "market_structure.skew_proxy": True})
        s = load_settings()
        moved = s.model_copy(update={"ml": s.ml.model_copy(update={"features": cfg})})
        rec = rr.build_run_record(kind="backtest", name="x", settings=moved, config_path=ROOT / "config.yaml", overlay_path=None)
        assert rec["feature_spec_hash"] == FeatureSpec.from_config(cfg).spec_hash() != GOLD["hash"]
        assert "rsi_14" not in rec["feature_list"] and "skew_proxy" in rec["feature_list"]
        base = rr.build_run_record(kind="backtest", name="x", settings=None, config_path=ROOT / "config.yaml", overlay_path=None)
        assert base["feature_list"] == GOLD["names_vlmc"] and base["feature_spec_hash"] == GOLD["hash"]

    def test_the_parity_manifest_warns_when_research_switches_a_feature(self, monkeypatch):
        import run_backtest as rb
        s = load_settings()
        moved = s.model_copy(update={"ml": s.ml.model_copy(update={"features": _cfg({"market_structure.skew_proxy": True})})})
        for research, expect in ((s, False), (moved, True)):
            monkeypatch.setattr(rb, "_research_settings", lambda r=research: r)
            monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
            m = rb.build_parity_manifest(rb.parse_args())
            assert any(w.startswith("ml_features") for w in m["parity_warnings"]) is expect

    def test_the_template_documents_the_switches(self):
        text = (ROOT / "studies" / "_template.yaml").read_text()
        assert "use: {hurst_psd_divergence: false}" in text and "skew_proxy: true" in text

"""W97: the remaining ``ml/features.py`` windows are ``ml.features`` settings; the spec hash is unchanged at the defaults.

Five new groups under ``ml.features`` (``market_structure``, ``cross_asset``, ``macro``, ``fractal``, ``intraday``; 50
settings, defaults = the old literals). What this file proves:

* at the defaults nothing moved: the extended golden record (macro columns at real values, short contexts, every
  intraday minimum-bar branch; recorded BEFORE the change) and the pinned default spec hash, plus the real artifacts;
* the hash rule: the new groups enter the hash ONLY when a value differs from its default;
* every setting MOVES exactly the columns it should (oracles from the raw series), and the fourteen model columns whose
  name embeds a window are renamed with it;
* the validators, the bounds, the config files, the walk-forward context check that now reads ``longest_lookback()``.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ait.config.settings import FeaturesConfig, load_settings
from ait.ml import hyperparams as hp
from ait.ml.feature_spec import FeatureSpec, default_spec_hash, derive_names, spec_from_config
from ait.ml.features import FeatureEngine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "data"))
import _w97_golden_inputs as g97  # noqa: E402

GOLD = json.loads((ROOT / "tests" / "data" / "w97_golden.json").read_text())
PINNED_DEFAULT_HASH = "6e0e3f8f922f6b88"          # default_spec_hash() on main BEFORE W97; recorded in models/*.pkl

# the 50 settings and their old literals
DEFAULTS = {
    "market_structure": {"put_call_window": 10, "put_call_min_periods": 5, "term_fast": 5, "term_slow": 60, "skew_window": 20,
                         "skew_min_periods": 10, "divergence_short": 5, "divergence_long": 60, "long_sma_window": 200,
                         "long_sma_min_periods": 100, "range_fast": 5, "range_slow": 20},
    "cross_asset": {"vix_change_short": 5, "vix_change_long": 20, "vix_z_window": 60, "vix_z_min_periods": 20, "vix_term_fast": 5,
                    "vix_term_slow": 20, "rel_short": 5, "rel_mid": 20, "rel_long": 60, "spy_momentum_days": 10,
                    "spy_rsi_period": 14, "corr_window": 20, "corr_min_periods": 10, "min_context_rows": 20},
    "macro": {"change_days": 20, "dxy_change_short": 5, "dxy_change_long": 20, "dxy_center": 100.0, "dxy_scale": 20.0},
    "fractal": {"hurst_short_window": 60, "hurst_long_window": 180, "hurst_min_len": 50, "psd_min_rows": 60, "min_rows": 50,
                "mfdfa_min_returns": 500},
    "intraday": {"hour_bars": 12, "rsi_period": 7, "atr_period": 14, "closing_bars": 3, "min_bars": 20, "session_min_rows": 10,
                 "load_buffer_days": 10, "session_min_bars": 3, "hurst_min_bars": 78, "scale_spread_min_bars": 200,
                 "wavelet_min_bars": 40, "psd_min_bars": 100, "mfdfa_min_returns": 500},
}
ALL_FIELDS = [(g, f) for g, d in DEFAULTS.items() for f in d]
# one valid alternative per setting (each is valid on its own: no ordered pair or min_periods rule is broken)
NEW_VALUES = {
    "market_structure": {"put_call_window": 12, "put_call_min_periods": 3, "term_fast": 4, "term_slow": 50, "skew_window": 25,
                         "skew_min_periods": 5, "divergence_short": 4, "divergence_long": 50, "long_sma_window": 150,
                         "long_sma_min_periods": 60, "range_fast": 4, "range_slow": 25},
    "cross_asset": {"vix_change_short": 4, "vix_change_long": 15, "vix_z_window": 50, "vix_z_min_periods": 10, "vix_term_fast": 4,
                    "vix_term_slow": 25, "rel_short": 4, "rel_mid": 15, "rel_long": 50, "spy_momentum_days": 8,
                    "spy_rsi_period": 10, "corr_window": 25, "corr_min_periods": 5, "min_context_rows": 700},
    "macro": {"change_days": 15, "dxy_change_short": 4, "dxy_change_long": 15, "dxy_center": 95.0, "dxy_scale": 25.0},
    "fractal": {"hurst_short_window": 40, "hurst_long_window": 150, "hurst_min_len": 60, "psd_min_rows": 80, "min_rows": 60,
                "mfdfa_min_returns": 150},
    "intraday": {"hour_bars": 6, "rsi_period": 5, "atr_period": 10, "closing_bars": 4, "min_bars": 30, "session_min_rows": 6,
                 "load_buffer_days": 30, "session_min_bars": 78, "hurst_min_bars": 100, "scale_spread_min_bars": 300,
                 "wavelet_min_bars": 100, "psd_min_bars": 200, "mfdfa_min_returns": 800},
}

# role -> the column name at the defaults (the fourteen model columns that follow a window)
OLD_NAMES = {
    "above_sma_long": "above_sma200", "distance_sma_long": "distance_sma200", "vix_change_short": "vix_change_5d",
    "vix_change_long": "vix_change_20d", "rel_short": "rel_strength_5d", "rel_mid": "rel_strength_20d",
    "rel_long": "rel_strength_60d", "spy_momentum": "spy_momentum_10d", "spy_rsi": "spy_rsi_14",
    "corr_spy": "correlation_spy_20d", "yield_curve_change": "yield_curve_change_20d", "us10y_change": "us_10y_change_20d",
    "dxy_change_short": "dxy_change_5d", "dxy_change_long": "dxy_change_20d",
}


def _cfg(**changes) -> FeaturesConfig:
    """FeaturesConfig with ``{"group.field": value}`` applied."""
    d = FeaturesConfig().model_dump()
    for key, val in changes.items():
        grp, fld = key.split(".")
        d[grp][fld] = val
    return FeaturesConfig(**d)


def _spec(**changes) -> FeatureSpec:
    return FeatureSpec.from_config(_cfg(**changes))


def _fe(**changes) -> FeatureEngine:
    return FeatureEngine(_spec(**changes))


@pytest.fixture
def fast_fractal(monkeypatch):
    """The fractal group is ~95% of a compute(); tests that are not about it replace it with constants."""
    monkeypatch.setattr(FeatureEngine, "_add_fractal_features", g97.stub_fractal)


def _frozen(monkeypatch, cfg: FeaturesConfig):
    import ait.config.settings as st
    s = load_settings()
    moved = s.model_copy(update={"ml": s.ml.model_copy(update={"features": cfg})})
    monkeypatch.setattr(st, "_FROZEN", moved)
    return moved


# --------------------------------------------------------------------------------------------------------------------
# 1. at the defaults nothing changed
# --------------------------------------------------------------------------------------------------------------------

class TestDefaultsAreTheOldBehaviour:
    def test_the_default_spec_hash_is_the_pinned_value_recorded_in_the_live_models(self):
        assert default_spec_hash() == PINNED_DEFAULT_HASH == GOLD["default_spec_hash"]
        assert spec_from_config(None).spec_hash() == _spec().spec_hash() == PINNED_DEFAULT_HASH

    @pytest.mark.parametrize("name", sorted(GOLD["frames"]))
    def test_the_extended_feature_frames_equal_the_golden_record(self, name):
        got = g97.compute_frames()[name]
        want = GOLD["frames"][name]
        assert list(got.columns) == want["columns"] and list(got.shape) == want["shape"]
        assert [str(i.date()) for i in got.tail(g97.TAIL).index] == want["tail_index"]
        np.testing.assert_allclose(got.tail(g97.TAIL).to_numpy(dtype="float64"),
                                   np.array(want["values"], dtype="float64"), rtol=1e-9, atol=1e-12, equal_nan=True)

    def test_every_intraday_minimum_bar_branch_equals_the_golden_record(self):
        got = g97.compute_intraday()
        assert sorted(got) == sorted(GOLD["intraday"])
        for n, want in GOLD["intraday"].items():
            assert sorted(got[n]) == sorted(want), n
            for k, v in want.items():
                assert got[n][k] == pytest.approx(v, rel=1e-9, abs=1e-12), (n, k)

    def test_the_golden_frames_exercise_the_new_columns(self):
        """A golden record of constants would prove nothing: the macro / cross-asset columns must carry real values."""
        f = GOLD["frames"]["macro_full"]
        cols = f["columns"]
        for name in ("dxy_change_5d", "dxy_level_norm", "yield_curve_change_20d", "us_10y_change_20d", "vix_zscore",
                     "rel_strength_60d", "distance_sma200", "skew_proxy", "put_call_proxy", "range_compression"):
            vals = {round(r[cols.index(name)], 9) for r in f["values"] if r[cols.index(name)] is not None}
            assert len(vals) > 20, name

    def test_the_loaded_config_lists_every_setting_at_its_default(self):
        live = yaml.safe_load((ROOT / "config.yaml").read_text())["ml"]["features"]
        loaded = load_settings().ml.features
        def windows(d):          # W98: every group also holds its ``use`` switches (tests/test_w98_feature_switches.py)
            return {k: v for k, v in d.items() if k != "use"}
        for grp, fields in DEFAULTS.items():
            assert windows(live[grp]) == fields, grp
            assert windows(loaded.model_dump()[grp]) == fields, grp
            assert windows(FeaturesConfig().model_dump()[grp]) == fields, grp

    def test_the_overlay_does_not_copy_shared_feature_keys(self):
        research = yaml.safe_load((ROOT / "config_research.yaml").read_text())
        assert "features" not in (research.get("ml") or {}), "shared keys are diff-only in the overlay"


# --------------------------------------------------------------------------------------------------------------------
# 2. the hash rule
# --------------------------------------------------------------------------------------------------------------------

class TestHashRule:
    @pytest.mark.parametrize("group,field", ALL_FIELDS)
    def test_changing_any_new_setting_changes_the_hash(self, group, field):
        h = _spec(**{f"{group}.{field}": NEW_VALUES[group][field]}).spec_hash()
        assert h != PINNED_DEFAULT_HASH

    def test_different_changes_give_different_hashes_and_reverting_restores_the_pinned_one(self):
        a = _spec(**{"macro.change_days": 15}).spec_hash()
        b = _spec(**{"macro.change_days": 25}).spec_hash()
        c = _spec(**{"cross_asset.spy_rsi_period": 10}).spec_hash()
        d = _spec(**{"macro.change_days": 15, "cross_asset.spy_rsi_period": 10}).spec_hash()
        assert len({PINNED_DEFAULT_HASH, a, b, c, d}) == 5
        assert _spec(**{"macro.change_days": 20}).spec_hash() == PINNED_DEFAULT_HASH
        assert _spec(**{"momentum.rsi_long": 21}).spec_hash() not in {PINNED_DEFAULT_HASH, a, b, c, d}   # a W88d window still counts

    def test_a_group_at_its_defaults_adds_nothing_to_the_canonical_string(self):
        """The same config built by value or from model_dump (explicit defaults) hashes identically."""
        explicit = FeaturesConfig(**FeaturesConfig().model_dump())
        assert FeatureSpec.from_config(explicit).spec_hash() == PINNED_DEFAULT_HASH

    @pytest.mark.parametrize("name", ["ensemble.pkl", "range.pkl", "vol_magnitude.pkl", "meta_label.pkl"])
    def test_the_real_live_artifacts_load_at_the_defaults_and_are_refused_when_a_w97_window_changes(self, name):
        live = ROOT / "models" / name
        if not live.exists():
            pytest.skip("no live artifact in this checkout")
        stored = pickle.loads(live.read_bytes()).get("spec")
        assert hp.feature_spec_refused(name, stored, default_spec_hash()) is False
        assert hp.feature_spec_refused(name, stored, _spec(**{"cross_asset.rel_long": 90}).spec_hash()) is True
        assert hp.feature_spec_refused(name, stored, _spec(**{"intraday.hour_bars": 6}).spec_hash()) is True

    def test_a_model_trained_under_a_changed_w97_window_is_refused_by_the_defaults(self, tmp_path, monkeypatch):
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        live = ROOT / "models" / "ensemble.pkl"
        if not live.exists():
            pytest.skip("no live artifact in this checkout")
        data = pickle.loads(live.read_bytes())
        data["spec"] = {**(data.get("spec") or {}), "feature_spec_hash": _spec(**{"macro.dxy_center": 95.0}).spec_hash()}
        (tmp_path / "ensemble.pkl").write_bytes(pickle.dumps(data))
        assert DirectionPredictor(MLConfig(), model_dir=tmp_path, persist_artifacts=False).load_models() is False


# --------------------------------------------------------------------------------------------------------------------
# 3. schema, validators, names
# --------------------------------------------------------------------------------------------------------------------

class TestSchema:
    def test_fifty_settings_in_five_groups_all_explained(self):
        assert len(ALL_FIELDS) == 50
        for grp, fields in DEFAULTS.items():
            model = type(getattr(FeaturesConfig(), grp))
            assert set(model.model_fields) - {"use"} == set(fields), grp          # W98: ``use`` is the switch map, not a window
            for f in fields:
                assert "W97" in (model.model_fields[f].description or ""), (grp, f)

    @pytest.mark.parametrize("changes", [
        {"market_structure.term_fast": 60}, {"market_structure.term_fast": 61},
        {"market_structure.divergence_short": 60}, {"market_structure.range_fast": 20}, {"market_structure.range_fast": 21},
        {"cross_asset.vix_change_short": 20}, {"cross_asset.vix_term_fast": 20}, {"cross_asset.rel_short": 20},
        {"cross_asset.rel_mid": 60}, {"cross_asset.rel_short": 30, "cross_asset.rel_mid": 25},
        {"macro.dxy_change_short": 20}, {"macro.dxy_change_short": 21},
        {"fractal.hurst_short_window": 180}, {"fractal.hurst_short_window": 181},
        {"market_structure.put_call_min_periods": 11}, {"market_structure.skew_min_periods": 21},
        {"market_structure.long_sma_min_periods": 201}, {"cross_asset.vix_z_min_periods": 61},
        {"cross_asset.corr_min_periods": 21},
    ])
    def test_ordered_pairs_and_min_periods_are_enforced(self, changes):
        with pytest.raises(ValueError):
            _cfg(**changes)

    @pytest.mark.parametrize("group,field,bad", [
        ("market_structure", "long_sma_window", 19), ("cross_asset", "vix_z_window", 4), ("cross_asset", "min_context_rows", 0),
        ("macro", "dxy_scale", 0.0), ("macro", "dxy_center", -1.0), ("fractal", "hurst_min_len", 15), ("fractal", "min_rows", 19),
        ("fractal", "mfdfa_min_returns", 99), ("intraday", "hour_bars", 1), ("intraday", "hour_bars", 79),
        ("intraday", "closing_bars", 1), ("intraday", "session_min_bars", 0), ("intraday", "load_buffer_days", -1),
    ])
    def test_bounds(self, group, field, bad):
        with pytest.raises(ValueError):
            _cfg(**{f"{group}.{field}": bad})

    def test_a_minimum_equal_to_its_window_is_allowed(self):
        assert _cfg(**{"market_structure.put_call_min_periods": 10}).market_structure.put_call_min_periods == 10
        assert _cfg(**{"cross_asset.corr_min_periods": 20}).cross_asset.corr_min_periods == 20

    def test_unknown_keys_are_a_startup_error(self):
        with pytest.raises(ValueError):
            FeaturesConfig(macro={"change_dayz": 20})
        with pytest.raises(ValueError):
            FeaturesConfig(not_a_group={})


class TestNames:
    def test_the_fourteen_names_at_the_defaults_are_the_old_ones(self):
        names = derive_names(FeaturesConfig())
        for role, col in OLD_NAMES.items():
            assert names[role] == col, role
        assert len(OLD_NAMES) == 14

    @pytest.mark.parametrize("field,value,renames", [
        ("market_structure.long_sma_window", 150, {"above_sma200": "above_sma150", "distance_sma200": "distance_sma150"}),
        ("cross_asset.vix_change_short", 4, {"vix_change_5d": "vix_change_4d"}),
        ("cross_asset.vix_change_long", 15, {"vix_change_20d": "vix_change_15d"}),
        ("cross_asset.rel_short", 4, {"rel_strength_5d": "rel_strength_4d"}),
        ("cross_asset.rel_mid", 15, {"rel_strength_20d": "rel_strength_15d"}),
        ("cross_asset.rel_long", 90, {"rel_strength_60d": "rel_strength_90d"}),
        ("cross_asset.spy_momentum_days", 8, {"spy_momentum_10d": "spy_momentum_8d"}),
        ("cross_asset.spy_rsi_period", 10, {"spy_rsi_14": "spy_rsi_10"}),
        ("cross_asset.corr_window", 25, {"correlation_spy_20d": "correlation_spy_25d"}),
        ("macro.change_days", 15, {"yield_curve_change_20d": "yield_curve_change_15d", "us_10y_change_20d": "us_10y_change_15d"}),
        ("macro.dxy_change_short", 4, {"dxy_change_5d": "dxy_change_4d"}),
        ("macro.dxy_change_long", 15, {"dxy_change_20d": "dxy_change_15d"}),
    ])
    def test_names_follow_the_windows_and_keep_their_position(self, field, value, renames):
        base = FeatureEngine().get_feature_names()
        moved = _fe(**{field: value}).get_feature_names()
        assert len(moved) == len(base) and len(set(moved)) == len(moved)
        for old, new in renames.items():
            assert base.index(old) == moved.index(new), (old, new)
        assert [c for c in base if c not in renames] == [c for c in moved if c not in renames.values()]

    def test_the_intraday_names_are_a_fixed_list(self):
        for hb in (6, 12, 30):
            assert _fe(**{"intraday.hour_bars": hb}).get_feature_names(include_vlmc=True) == \
                FeatureEngine().get_feature_names(include_vlmc=True)


# --------------------------------------------------------------------------------------------------------------------
# 4. every window moves what it should
# --------------------------------------------------------------------------------------------------------------------

BASE_DF, CTX = g97.full_context()


def _oracle_cross(df, ctx, kind, n, lo=None, hi=None):
    close = df["Close"]
    vix = ctx["vix"]["Close"].reindex(df.index, method="ffill")
    spy = ctx["spy"]["Close"].reindex(df.index, method="ffill")
    if kind == "vix_change":
        return vix.pct_change(n).clip(-0.5, 0.5)
    if kind == "rel":
        return (close.pct_change(n) - spy.pct_change(n)).clip(lo, hi)
    if kind == "spy_mom":
        return spy.pct_change(n).clip(-0.15, 0.15)
    if kind == "spy_rsi":
        return FeatureEngine._rsi(spy, n) / 100.0
    if kind == "corr":
        return close.pct_change().rolling(n, min_periods=10).corr(spy.pct_change()).clip(-1, 1).fillna(0)
    raise AssertionError(kind)


def _oracle_macro(df, ctx, kind, n):
    m = {k: v.reindex(df.index, method="ffill") for k, v in ctx["macros"].items()}
    if kind == "curve":
        return m["yield_curve"].diff(n).clip(-2, 2).fillna(0)
    if kind == "y10":
        return m["us_10y_yield"].diff(n).clip(-2, 2).fillna(0)
    if kind == "dxy5":
        return m["dxy"].pct_change(n).clip(-0.1, 0.1).fillna(0)
    if kind == "dxy20":
        return m["dxy"].pct_change(n).clip(-0.2, 0.2).fillna(0)
    raise AssertionError(kind)


def _aligned(frame, series):
    pos = BASE_DF.index.get_indexer(pd.DatetimeIndex(frame.index))
    assert (pos >= 0).all()
    return series.to_numpy(dtype="float64")[pos]


# field, value, {new column: oracle(df, ctx)}
RENAMED = [
    ("market_structure.long_sma_window", 150, {
        "distance_sma150": lambda d, c: ((d["Close"] - d["Close"].rolling(150, min_periods=100).mean())
                                         / d["Close"].rolling(150, min_periods=100).mean()).clip(-0.3, 0.3),
        "above_sma150": lambda d, c: (d["Close"] > d["Close"].rolling(150, min_periods=100).mean()).astype(float)}),
    ("cross_asset.vix_change_short", 4, {"vix_change_4d": lambda d, c: _oracle_cross(d, c, "vix_change", 4)}),
    ("cross_asset.vix_change_long", 15, {"vix_change_15d": lambda d, c: _oracle_cross(d, c, "vix_change", 15)}),
    ("cross_asset.rel_short", 4, {"rel_strength_4d": lambda d, c: _oracle_cross(d, c, "rel", 4, -0.2, 0.2)}),
    ("cross_asset.rel_mid", 15, {"rel_strength_15d": lambda d, c: _oracle_cross(d, c, "rel", 15, -0.3, 0.3)}),
    ("cross_asset.rel_long", 50, {"rel_strength_50d": lambda d, c: _oracle_cross(d, c, "rel", 50, -0.5, 0.5)}),
    ("cross_asset.spy_momentum_days", 8, {"spy_momentum_8d": lambda d, c: _oracle_cross(d, c, "spy_mom", 8)}),
    ("cross_asset.spy_rsi_period", 10, {"spy_rsi_10": lambda d, c: _oracle_cross(d, c, "spy_rsi", 10)}),
    ("cross_asset.corr_window", 25, {"correlation_spy_25d": lambda d, c: _oracle_cross(d, c, "corr", 25)}),
    ("macro.change_days", 15, {"yield_curve_change_15d": lambda d, c: _oracle_macro(d, c, "curve", 15),
                               "us_10y_change_15d": lambda d, c: _oracle_macro(d, c, "y10", 15)}),
    ("macro.dxy_change_short", 4, {"dxy_change_4d": lambda d, c: _oracle_macro(d, c, "dxy5", 4)}),
    ("macro.dxy_change_long", 15, {"dxy_change_15d": lambda d, c: _oracle_macro(d, c, "dxy20", 15)}),
]

# field, value, the columns whose VALUES change (the name does not)
VALUE_ONLY = [
    ("market_structure.put_call_window", 12, {"put_call_proxy"}),
    ("market_structure.term_fast", 4, {"vix_term_structure"}),
    ("market_structure.term_slow", 50, {"vix_term_structure"}),
    ("market_structure.skew_window", 25, {"skew_proxy"}),
    ("market_structure.divergence_short", 4, {"momentum_divergence"}),
    ("market_structure.divergence_long", 50, {"momentum_divergence"}),
    ("market_structure.range_fast", 4, {"range_compression"}),
    ("market_structure.range_slow", 25, {"range_compression"}),
    ("cross_asset.vix_z_window", 50, {"vix_zscore"}),
    ("cross_asset.vix_term_fast", 4, {"vix_term_spread"}),
    ("cross_asset.vix_term_slow", 25, {"vix_term_spread"}),
    ("macro.dxy_center", 95.0, {"dxy_level_norm"}),
    ("macro.dxy_scale", 25.0, {"dxy_level_norm"}),
]


@pytest.mark.usefixtures("fast_fractal")
class TestEveryWindowChangesTheExpectedColumns:
    @pytest.fixture(scope="class")
    def base(self):
        mp = pytest.MonkeyPatch()
        mp.setattr(FeatureEngine, "_add_fractal_features", g97.stub_fractal)
        try:
            yield FeatureEngine().compute(BASE_DF, market_context=CTX)
        finally:
            mp.undo()

    @pytest.mark.parametrize("field,value,oracles", RENAMED, ids=[r[0] for r in RENAMED])
    def test_a_renamed_column_holds_the_new_window(self, field, value, oracles, base):
        f = _fe(**{field: value}).compute(BASE_DF, market_context=CTX)
        for col, oracle in oracles.items():
            assert col in f.columns, col
            np.testing.assert_allclose(f[col].to_numpy(dtype="float64"), _aligned(f, oracle(BASE_DF, CTX)),
                                       rtol=1e-9, atol=1e-12, err_msg=col)
        gone = [c for c in base.columns if c not in f.columns]
        assert len(gone) == len(oracles) and not [c for c in oracles if c in base.columns]

    @pytest.mark.parametrize("field,value,changed", VALUE_ONLY, ids=[r[0] for r in VALUE_ONLY])
    def test_a_window_changes_exactly_its_columns(self, field, value, changed, base):
        f = _fe(**{field: value}).compute(BASE_DF, market_context=CTX)
        assert list(f.columns) == list(base.columns) and len(f) == len(base)
        moved = {c for c in base.columns
                 if not np.allclose(base[c].to_numpy(dtype=float), f[c].to_numpy(dtype=float), rtol=1e-12, atol=1e-12, equal_nan=True)}
        assert moved == changed

    def test_the_minimum_context_rows_switch_the_cross_asset_columns_to_neutral(self, base):
        neutral = _fe(**{"cross_asset.min_context_rows": 700}).compute(BASE_DF, market_context=CTX)     # 700 rows is not > 700
        for c in ("vix_zscore", "vix_term_spread", "vix_change_5d", "vix_change_20d", "rel_strength_5d", "rel_strength_20d",
                  "rel_strength_60d", "spy_momentum_10d", "correlation_spy_20d"):
            assert neutral[c].nunique() == 1 and neutral[c].iloc[0] == 0.0, c
        assert neutral["vix_level"].iloc[0] == 0.5 and (neutral["spy_rsi_14"] == 0.5).all()
        assert base["vix_zscore"].nunique() > 20
        almost = _fe(**{"cross_asset.min_context_rows": 699}).compute(BASE_DF, market_context=CTX)       # 700 > 699: used
        np.testing.assert_allclose(almost["vix_zscore"].to_numpy(), base["vix_zscore"].to_numpy())

    def test_the_long_sma_threshold_is_the_window(self):
        short = BASE_DF.iloc[:150]
        f = _fe(**{"market_structure.long_sma_window": 150, "market_structure.long_sma_min_periods": 100}).compute(short)
        assert f["distance_sma150"].nunique() > 1                     # 150 rows >= 150: computed
        g = _fe().compute(short)
        assert g["distance_sma200"].nunique() == 1 and g["distance_sma200"].iloc[0] == 0.0 and g["above_sma200"].iloc[0] == 1.0

    def test_the_neutral_branches_write_the_renamed_columns(self):
        """When a feed is absent the neutral value goes into the column under its CURRENT (renamed) name."""
        renamed = {"market_structure.long_sma_window": 150, "market_structure.long_sma_min_periods": 100,
                   "cross_asset.vix_change_short": 4, "cross_asset.vix_change_long": 15, "cross_asset.rel_short": 4,
                   "cross_asset.rel_mid": 15, "cross_asset.rel_long": 50, "cross_asset.spy_momentum_days": 8,
                   "cross_asset.spy_rsi_period": 10, "cross_asset.corr_window": 25, "macro.change_days": 15,
                   "macro.dxy_change_short": 4, "macro.dxy_change_long": 15}
        fe = _fe(**renamed)
        # (a) no market context at all, and a frame shorter than the long SMA window
        short = fe.compute(BASE_DF.iloc[:140])
        assert (short["above_sma150"] == 1.0).all() and (short["distance_sma150"] == 0.0).all()
        for c in ("vix_change_4d", "vix_change_15d", "rel_strength_4d", "rel_strength_15d", "rel_strength_50d",
                  "spy_momentum_8d", "correlation_spy_25d", "yield_curve_change_15d", "us_10y_change_15d", "dxy_change_4d",
                  "dxy_change_15d"):
            assert (short[c] == 0.0).all(), c
        assert (short["spy_rsi_10"] == 50.0).all()                         # the no-context neutral RSI
        # (b) VIX / SPY present but too short to use, and no macros
        thin = {"vix": CTX["vix"].iloc[:15], "spy": CTX["spy"].iloc[:15]}
        out = fe.compute(BASE_DF, market_context=thin)
        assert (out["spy_rsi_10"] == 0.5).all()                            # the short-context neutral RSI
        for c in ("vix_change_4d", "vix_change_15d", "rel_strength_4d", "spy_momentum_8d", "correlation_spy_25d",
                  "yield_curve_change_15d", "us_10y_change_15d", "dxy_change_4d", "dxy_change_15d"):
            assert (out[c] == 0.0).all(), c

    def test_the_long_sma_minimum_periods_set_the_first_row_kept(self, base):
        f = _fe(**{"market_structure.long_sma_min_periods": 60}).compute(BASE_DF, market_context=CTX)
        assert len(f) > len(base) and f.index[0] < base.index[0]

    @pytest.mark.parametrize("group,field,lo,hi,call", [
        ("market_structure", "put_call_min_periods", 3, 5, "ms"), ("market_structure", "skew_min_periods", 5, 10, "ms"),
        ("cross_asset", "vix_z_min_periods", 10, 20, "ca"), ("cross_asset", "corr_min_periods", 5, 10, "ca"),
    ])
    def test_a_minimum_periods_changes_the_first_rows(self, group, field, lo, hi, call):
        """A min_periods only matters at the start of a frame (the warm-up rows are dropped from a full compute())."""
        def first_rows(value):
            fe = _fe(**{f"{group}.{field}": value})
            df = BASE_DF.iloc[:40].copy()
            out = fe._add_market_structure(df) if call == "ms" else fe._add_cross_asset(df, {k: v for k, v in CTX.items()})
            return out
        a, b = first_rows(lo), first_rows(hi)
        col = {"put_call_min_periods": "put_call_proxy", "skew_min_periods": "skew_proxy", "vix_z_min_periods": "vix_zscore",
               "corr_min_periods": "correlation_spy_20d"}[field]
        assert not np.allclose(a[col].to_numpy(), b[col].to_numpy(), equal_nan=True)


class TestFractalWindows:
    @pytest.fixture(scope="class")
    def frames(self):
        df = BASE_DF.iloc[:260]
        cache = {}

        def run(**changes):
            key = tuple(sorted(changes.items()))
            if key not in cache:
                cache[key] = _fe(**changes)._add_fractal_features(df.copy())
            return cache[key]
        return run

    COLS = ["hurst_wavelet", "hurst_fit_r2", "psd_beta", "psd_fit_r2", "hurst_psd_divergence", "hurst_short", "hurst_long",
            "hurst_scale_spread", "multifractal_width", "multifractal_asymmetry"]

    @pytest.mark.parametrize("changes,expected", [
        ({"fractal.hurst_short_window": 40}, {"hurst_short", "hurst_scale_spread"}),
        ({"fractal.hurst_long_window": 150}, {"hurst_long", "hurst_scale_spread"}),
        ({"fractal.hurst_min_len": 60}, {"hurst_wavelet", "hurst_fit_r2", "hurst_psd_divergence"}),
        ({"fractal.psd_min_rows": 80}, {"psd_beta", "psd_fit_r2", "hurst_psd_divergence"}),
        ({"fractal.min_rows": 60}, {"hurst_wavelet", "hurst_fit_r2", "hurst_psd_divergence"}),
        ({"fractal.mfdfa_min_returns": 150}, {"multifractal_width", "multifractal_asymmetry"}),
    ])
    def test_each_fractal_window_moves_its_columns(self, frames, changes, expected):
        base, moved = frames(), frames(**changes)
        got = {c for c in self.COLS if not np.allclose(base[c].to_numpy(float), moved[c].to_numpy(float), rtol=1e-12, atol=1e-12)}
        assert got == expected

    def test_the_hurst_windows_are_the_trailing_bars_of_the_log_price(self, frames):
        df = BASE_DF.iloc[:260]
        lp = np.log(df["Close"].to_numpy())
        for short, long_ in ((60, 180), (40, 150)):
            f = frames(**({} if (short, long_) == (60, 180) else {"fractal.hurst_short_window": short, "fractal.hurst_long_window": long_}))
            fe = FeatureEngine()
            assert f["hurst_short"].iloc[-1] == pytest.approx(fe._hurst_wavelet(lp[-short:])[0])
            assert f["hurst_long"].iloc[-1] == pytest.approx(fe._hurst_wavelet(lp[-long_:])[0])

    def test_the_expanding_start_is_min_rows(self, frames):
        f = frames(**{"fractal.min_rows": 60})
        assert (f["hurst_wavelet"].iloc[:59] == 0.0).all() and f["hurst_wavelet"].iloc[59] != 0.0
        g = frames()
        assert (g["hurst_wavelet"].iloc[:49] == 0.0).all() and g["hurst_wavelet"].iloc[49] != 0.0

    def test_below_min_rows_the_whole_group_is_zero(self):
        short = BASE_DF.iloc[:55]
        out = _fe(**{"fractal.min_rows": 60})._add_fractal_features(short.copy())
        assert (out[self.COLS] == 0.0).all().all()
        assert _fe()._add_fractal_features(short.copy())["hurst_wavelet"].iloc[-1] != 0.0

    def test_the_estimator_minimums(self):
        series = np.cumsum(np.random.default_rng(3).normal(0, 1, 70))
        fe, strict = FeatureEngine(), _fe(**{"fractal.hurst_min_len": 80, "fractal.psd_min_rows": 80})
        assert fe._hurst_wavelet(series)[0] != 0.5 and strict._hurst_wavelet(series) == (0.5, 0.0)
        assert fe._psd_features(series)[0] != 2.0 and strict._psd_features(series) == (2.0, 0.0)

    def test_multiscale_hurst_uses_the_two_windows_and_their_fallbacks(self):
        series = np.cumsum(np.random.default_rng(4).normal(0, 1, 120))
        fe = _fe(**{"fractal.hurst_short_window": 40, "fractal.hurst_long_window": 100})
        s, l_ = fe._multiscale_hurst(series)
        assert s == pytest.approx(fe._hurst_wavelet(series[-40:])[0]) and l_ == pytest.approx(fe._hurst_wavelet(series[-100:])[0])
        s2, l2 = fe._multiscale_hurst(series[:80])            # shorter than the long window: only the short one
        assert l2 == 0.0 and s2 == pytest.approx(fe._hurst_wavelet(series[:80][-40:])[0])
        assert fe._multiscale_hurst(series[:30]) == (0.0, 0.0)


class TestIntradayWindows:
    FRAME = g97.intraday_frame()

    def _all(self, fe):
        return {n: fe.compute_intraday_features(self.FRAME.iloc[-n:]) for n in (702, 250, 150, 90, 45, 25)}

    @pytest.fixture(scope="class")
    def base(self):
        return self._all(FeatureEngine())

    def _moved(self, base, **changes):
        got = self._all(_fe(**changes))
        out = set()
        for n in base:
            if sorted(got[n]) != sorted(base[n]):
                out.add(f"keys@{n}")
            for k in got[n]:
                if k in base[n] and not np.isclose(got[n][k], base[n][k], rtol=1e-12, atol=1e-12):
                    out.add(k)
        return out

    @pytest.mark.parametrize("field,value,expected", [
        ("rsi_period", 5, {"intraday_rsi"}),
        ("atr_period", 10, {"intraday_atr_pct"}),
        ("closing_bars", 4, {"closing_imbalance"}),
        ("min_bars", 30, {"keys@25"}),
        ("hurst_min_bars", 100, {"hurst_wavelet_intraday"}),
        ("wavelet_min_bars", 100, {"wavelet_L3_energy"}),
        ("psd_min_bars", 200, {"psd_beta_intraday"}),
        ("mfdfa_min_returns", 800, {"mfdfa_width_intraday"}),
        ("session_min_bars", 78, {"closing_imbalance", "closing_range_position", "power_hour_momentum", "power_hour_vol_accel",
                                  "power_hour_vwap_cross", "session_high_timing", "session_low_timing",
                                  "session_volume_front_load", "session_volume_shape", "session_vwap_position",
                                  "session_vwap_q1", "session_vwap_q2", "session_vwap_q3"}),
    ])
    def test_each_bar_count_moves_its_features(self, base, field, value, expected):
        assert self._moved(base, **{f"intraday.{field}": value}) == expected

    def test_hour_bars_moves_the_hourly_momentum_and_the_power_hour(self, base):
        got = self._moved(base, **{"intraday.hour_bars": 6})
        assert {"intraday_momentum_1h", "power_hour_momentum", "power_hour_vol_accel"} <= got <= {
            "intraday_momentum_1h", "power_hour_momentum", "power_hour_vol_accel", "intraday_range_compression"}
        df = self.FRAME
        assert _fe(**{"intraday.hour_bars": 6}).compute_intraday_features(df)["intraday_momentum_1h"] == \
            pytest.approx(float(df["Close"].pct_change(6).iloc[-1]))

    def test_the_intraday_mfdfa_minimum_overrides_the_daily_one_when_lower(self):
        frame = self.FRAME.iloc[-400:]                                   # 399 returns: below the daily 500
        assert FeatureEngine().compute_intraday_features(frame)["mfdfa_width_intraday"] == 0.0
        lowered = _fe(**{"intraday.mfdfa_min_returns": 300}).compute_intraday_features(frame)["mfdfa_width_intraday"]
        assert lowered != 0.0

    def test_the_scale_spread_minimum_gates_the_multiscale_hurst(self, monkeypatch):
        monkeypatch.setattr(FeatureEngine, "_multiscale_hurst", lambda self, s: (0.2, 0.6))
        frame = self.FRAME.iloc[-250:]
        assert FeatureEngine().compute_intraday_features(frame)["hurst_scale_spread_intraday"] == pytest.approx(0.4)
        assert _fe(**{"intraday.scale_spread_min_bars": 300}).compute_intraday_features(frame)["hurst_scale_spread_intraday"] == 0.0

    def test_the_per_day_session_minimum_and_the_fallback_load_buffer(self, monkeypatch):
        seen: list[int] = []
        monkeypatch.setattr(FeatureEngine, "compute_intraday_features",
                            lambda self, session: seen.append(len(session)) or {"intraday_rsi": 50.0})

        def session(day, n):
            idx = pd.date_range(f"{day}T14:30:00+00:00", periods=n, freq="5min")
            return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)

        class Store:
            def __init__(self):
                self.days = None

            def load_intraday(self, symbol, days):
                self.days = days
                return pd.concat([session("2024-06-03", 12), session("2024-06-04", 8)])

        features = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2024-06-03", "2024-06-04"]))
        store = Store()
        FeatureEngine()._merge_intraday_features(features, store, "SPY")
        assert seen == [12] and store.days == 2 + 10                                  # the 8-bar day is below 10; buffer 10
        seen.clear()
        FeatureEngine(_spec(**{"intraday.session_min_rows": 6, "intraday.load_buffer_days": 30}))._merge_intraday_features(
            features, store, "SPY")
        assert seen == [12, 8] and store.days == 2 + 30


# --------------------------------------------------------------------------------------------------------------------
# 5. longest_lookback and the walk-forward context
# --------------------------------------------------------------------------------------------------------------------

class TestLongestLookback:
    def test_252_at_the_defaults(self):
        assert _spec().longest_lookback() == 252 == FeatureEngine().spec.longest_lookback()

    @pytest.mark.parametrize("changes,expected", [
        ({"market_structure.long_sma_window": 400, "market_structure.long_sma_min_periods": 100}, 400),
        ({"fractal.hurst_long_window": 300}, 300),
        ({"cross_asset.rel_long": 320}, 320),
        ({"cross_asset.vix_z_window": 330}, 330),
        ({"market_structure.term_slow": 340}, 340),
        ({"market_structure.divergence_long": 350}, 350),
        ({"trend.sma_slow": 360}, 360),
        ({"vol_regime.slow": 370}, 370),
        ({"vol_regime.iv_rank_lookback": 380}, 380),
    ])
    def test_each_long_window_can_be_the_longest(self, changes, expected):
        assert _spec(**changes).longest_lookback() == expected

    def test_the_walk_forward_refuses_a_context_shorter_than_the_longest_window(self, monkeypatch):
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        _frozen(monkeypatch, _cfg())
        WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))       # 252 >= 252
        _frozen(monkeypatch, _cfg(**{"fractal.hurst_long_window": 300}))
        with pytest.raises(ValueError, match="longest"):
            WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))
        _frozen(monkeypatch, _cfg(**{"market_structure.long_sma_window": 400, "market_structure.long_sma_min_periods": 100}))
        with pytest.raises(ValueError, match="oos_feature_context_bars"):
            WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))


# --------------------------------------------------------------------------------------------------------------------
# 6. class guards: the converted sites carry no bare window
# --------------------------------------------------------------------------------------------------------------------

class TestNoBareWindowsLeft:
    @pytest.mark.parametrize("func,forbidden", [
        ("_add_market_structure", {10, 5, 60, 20, 200, 100}),
        ("_add_cross_asset", {5, 10, 14, 20, 60}),
        ("_add_macro", {5, 20, 100}),
        ("_add_fractal_features", {49, 50, 500}),
        ("_hurst_wavelet", {50}),
        ("_psd_features", {60}),
        ("_multiscale_hurst", {60, 180}),
        ("_mfdfa_features", {500}),
        ("compute_intraday_features", {7, 12, 14, 20, 40, 78, 100, 200, 500}),
        ("_merge_intraday_features", {10}),
    ])
    def test_the_function_no_longer_carries_the_old_window_literals(self, func, forbidden):
        import ast
        tree = ast.parse((ROOT / "src" / "ait" / "ml" / "features.py").read_text())
        node = next(f for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == "FeatureEngine"
                    for f in c.body if isinstance(f, ast.FunctionDef) and f.name == func)
        found = {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                 and not isinstance(n.value, bool)}
        assert not (found & forbidden), f"{func} still carries {sorted(found & forbidden)}"

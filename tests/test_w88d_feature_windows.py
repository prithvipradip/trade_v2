"""W88d: the indicator windows are settings, the feature columns / names / meta-label keys follow them, and a model trained under
other windows is refused.

Two halves, as in W88b/c: AT THE DEFAULT WINDOWS everything is identical to the pre-W88d code (golden record captured before the change:
``tests/data/w88d_feature_golden.*``), and a CHANGED window changes exactly the expected columns (below) and is followed by every consumer.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.config.settings import FeaturesConfig, MLConfig, load_settings
from ait.ml import feature_spec as fs
from ait.ml import hyperparams as hp
from ait.ml.feature_spec import FeatureSpec, default_spec_hash, derive_names, spec_from_config
from ait.ml.features import FeatureEngine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "data"))
import _w88d_golden_inputs as gi  # noqa: E402

GOLDEN = json.loads((ROOT / "tests" / "data" / "w88d_feature_golden.json").read_text())
GOLDEN_NPZ = np.load(ROOT / "tests" / "data" / "w88d_feature_golden.npz")

# field -> (new value, {old column: new column}, columns that keep their name but change value)
EFFECTS = {
    "momentum.rsi_long": (21, {'rsi_14': 'rsi_21'}, []),
    "momentum.rsi_short": (4, {'rsi_7': 'rsi_4'}, []),
    "momentum.macd_fast": (8, {'ema_12': 'ema_8'}, ['macd', 'macd_hist', 'macd_signal']),
    "momentum.macd_slow": (30, {'ema_26': 'ema_30'}, ['macd', 'macd_hist', 'macd_signal']),
    "momentum.macd_signal": (5, {}, ['macd_hist', 'macd_signal']),
    "momentum.roc_short": (3, {'roc_5': 'roc_3'}, []),
    "momentum.roc_mid": (12, {'roc_10': 'roc_12'}, []),
    "momentum.roc_long": (30, {'roc_20': 'roc_30'}, []),
    "volatility.atr_window": (10, {'atr_14': 'atr_10'}, ['atr_pct']),
    "volatility.bb_window": (25, {}, ['bb_lower', 'bb_pct_above_upper', 'bb_pct_below_lower', 'bb_position', 'bb_upper', 'bb_width']),
    "volatility.bb_std": (2.5, {}, ['bb_lower', 'bb_pct_above_upper', 'bb_pct_below_lower', 'bb_position', 'bb_upper', 'bb_width']),
    "volatility.rv_long": (30, {'realized_vol_20': 'realized_vol_30'}, ['iv_rank', 'vol_mean_reversion_signal', 'vol_of_vol', 'vol_regime_expanding', 'vol_trend']),
    "volatility.rv_short": (8, {'realized_vol_10': 'realized_vol_8'}, ['vol_ratio']),
    "volume.volume_avg_window": (30, {'volume_sma_20_ratio': 'volume_sma_30_ratio'}, ['volume_confirmation', 'volume_trend']),
    "volume.volume_fast_window": (3, {}, ['volume_trend']),
    "volume.obv_period": (8, {}, ['obv_change']),
    "trend.sma_fast": (8, {'sma_10': 'sma_8', 'sma_10_20_cross': 'sma_8_20_cross', 'sma_10_slope': 'sma_8_slope'}, []),
    "trend.sma_mid": (30, {'price_vs_sma_20': 'price_vs_sma_30', 'sma_10_20_cross': 'sma_10_30_cross', 'sma_20': 'sma_30', 'sma_20_slope': 'sma_30_slope'}, ['volume_confirmation', 'weekly_trend_aligned']),
    "trend.sma_slow": (80, {'price_vs_sma_50': 'price_vs_sma_80', 'sma_50': 'sma_80'}, []),
    "trend.slope_period": (3, {}, ['sma_10_slope', 'sma_20_slope']),
    "multi_timeframe.weekly_close_window": (7, {}, ['weekly_rsi', 'weekly_trend_aligned']),
    "multi_timeframe.weekly_trend_window": (6, {}, ['weekly_trend_aligned']),
    "multi_timeframe.weekly_trend_shift": (3, {}, ['weekly_trend_aligned']),
    "multi_timeframe.weekly_rsi_period": (10, {}, ['weekly_rsi']),
    "multi_timeframe.weekly_momentum_period": (15, {}, ['weekly_momentum']),
    "vol_regime.fast": (4, {}, ['vol_regime_expanding']),
    "vol_regime.slow": (80, {}, ['vol_mean_reversion_signal', 'vol_ratio']),
    "vol_regime.iv_rank_lookback": (120, {}, ['iv_rank']),
    "vol_regime.iv_rank_min_periods": (30, {}, []),
    "vol_regime.vol_trend_period": (3, {}, ['vol_trend']),
    "vol_regime.vol_of_vol_window": (30, {}, ['vol_of_vol']),
}



_FRACTAL_COLUMNS = ["hurst_wavelet", "hurst_fit_r2", "psd_beta", "psd_fit_r2", "hurst_psd_divergence", "hurst_short",
                    "hurst_long", "hurst_scale_spread", "multifractal_width", "multifractal_asymmetry"]


def _stub_fractal(self, df):
    """The fractal group is ~95% of a compute() and is fixed (not a W88d window); tests that are not the golden
    comparison replace it with constants so the window logic is exercised in a fraction of the time."""
    for c in _FRACTAL_COLUMNS:
        df[c] = 0.5
    return df


@pytest.fixture(scope="class")
def fast_fractal():
    mp = pytest.MonkeyPatch()
    mp.setattr(FeatureEngine, "_add_fractal_features", _stub_fractal)
    yield
    mp.undo()


_REAL = {}


def _real_features() -> dict:
    """compute_all() with the REAL fractal group, once per session (it takes ~1 min)."""
    if "frames" not in _REAL:
        _REAL["frames"] = gi.compute_all()
    return _REAL["frames"]


def _oracle(col: str, cfg: FeaturesConfig, df: pd.DataFrame) -> pd.Series:
    """What a window-bearing column SHOULD hold, computed from the column's own name and the raw OHLCV only."""
    import re
    close, volume = df["Close"], df["Volume"]
    if m := re.fullmatch(r"rsi_(\d+)", col):
        return FeatureEngine._rsi(close, int(m[1]))
    if m := re.fullmatch(r"roc_(\d+)", col):
        return close.pct_change(int(m[1]))
    if m := re.fullmatch(r"ema_(\d+)", col):
        return close.ewm(span=int(m[1]), adjust=False).mean()
    if m := re.fullmatch(r"atr_(\d+)", col):
        tr = pd.concat([df["High"] - df["Low"], (df["High"] - close.shift(1)).abs(), (df["Low"] - close.shift(1)).abs()],
                       axis=1).max(axis=1)
        return tr.rolling(int(m[1])).mean()
    if m := re.fullmatch(r"realized_vol_(\d+)", col):
        return np.log(close / close.shift(1)).rolling(int(m[1])).std() * np.sqrt(252)
    if m := re.fullmatch(r"volume_sma_(\d+)_ratio", col):
        avg = volume.rolling(int(m[1])).mean()
        return (volume / avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    if m := re.fullmatch(r"sma_(\d+)", col):
        return close.rolling(int(m[1])).mean()
    if m := re.fullmatch(r"sma_(\d+)_slope", col):
        return close.rolling(int(m[1])).mean().pct_change(cfg.trend.slope_period)
    if m := re.fullmatch(r"price_vs_sma_(\d+)", col):
        sma = close.rolling(int(m[1])).mean()
        return (close - sma) / sma
    if m := re.fullmatch(r"sma_(\d+)_(\d+)_cross", col):
        return (close.rolling(int(m[1])).mean() > close.rolling(int(m[2])).mean()).astype(float)
    raise AssertionError(f"no oracle for {col}")


def _cfg(**changes) -> FeaturesConfig:
    """FeaturesConfig with ``{"group.field": value}`` applied."""
    d = FeaturesConfig().model_dump()
    for key, val in changes.items():
        grp, fld = key.split(".")
        d[grp][fld] = val
    return FeaturesConfig(**d)


def _spec(**changes) -> FeatureSpec:
    return FeatureSpec.from_config(_cfg(**changes))


def _frozen(monkeypatch, cfg: FeaturesConfig):
    import ait.config.settings as st
    s = load_settings()
    monkeypatch.setattr(st, "_FROZEN", s.model_copy(update={"ml": s.ml.model_copy(update={"features": cfg})}))


def _ohlcv(days: int = 260, seed: int = 42) -> pd.DataFrame:
    return gi.frames()["calm_700"].iloc[:days] if days <= 700 else gi.frames()["calm_700"]


# --------------------------------------------------------------------------
# 1. at the default windows nothing changed (golden record, captured BEFORE the change)
# --------------------------------------------------------------------------

class TestDefaultsAreTheOldBehaviour:
    @pytest.mark.parametrize("name", sorted(GOLDEN["frames"]))
    def test_the_feature_frames_equal_the_golden_record(self, name):
        got = _real_features()[name]
        want = GOLDEN["frames"][name]
        assert list(got.columns) == want["columns"] and list(got.shape) == want["shape"]
        assert [str(i.date()) for i in got.tail(gi.TAIL).index] == want["tail_index"]
        np.testing.assert_allclose(got.tail(gi.TAIL).to_numpy(dtype="float64"), GOLDEN_NPZ[name], rtol=1e-9, atol=1e-12,
                                   equal_nan=True)

    def test_the_feature_name_lists_equal_the_golden_record(self):
        assert gi.name_lists() == GOLDEN["names"]

    def test_the_meta_training_frame_equals_the_golden_record(self):
        feats = _real_features()["calm_700"]
        got = gi.meta_training_frame(feats)
        want = GOLDEN["meta_training"]
        assert list(got.columns) == want["columns"]
        np.testing.assert_allclose(got.round(10).to_numpy(dtype="float64"), np.array(want["rows"], dtype="float64"),
                                   rtol=1e-9, atol=1e-12)

    def test_the_engine_meta_context_and_entry_snapshot_equal_the_golden_record(self):
        got, want = gi.engine_meta_contexts(), GOLDEN["engine"]
        assert got["n_contexts"] == want["n_contexts"]
        assert got["snapshot"] == want["snapshot"]
        for g, w in zip(got["contexts"], want["contexts"]):
            assert sorted(g) == sorted(w)
            for k in w:
                assert g[k] == pytest.approx(w[k], rel=1e-9, abs=1e-12), k

    def test_the_default_spec_hash_is_the_assumed_hash_of_a_hashless_model(self):
        assert spec_from_config(None).spec_hash() == default_spec_hash() == fs.default_feature_spec().spec_hash()


# --------------------------------------------------------------------------
# 2. schema and spec
# --------------------------------------------------------------------------

@pytest.mark.usefixtures("fast_fractal")
class TestSchemaAndSpec:
    def test_defaults_are_the_old_literals(self):
        c = FeaturesConfig()
        assert (c.momentum.rsi_long, c.momentum.rsi_short, c.momentum.macd_fast, c.momentum.macd_slow,
                c.momentum.macd_signal) == (14, 7, 12, 26, 9)
        assert (c.momentum.roc_short, c.momentum.roc_mid, c.momentum.roc_long) == (5, 10, 20)
        assert (c.volatility.atr_window, c.volatility.bb_window, c.volatility.bb_std, c.volatility.rv_long,
                c.volatility.rv_short) == (14, 20, 2.0, 20, 10)
        assert (c.volume.volume_avg_window, c.volume.volume_fast_window, c.volume.obv_period) == (20, 5, 5)
        assert (c.trend.sma_fast, c.trend.sma_mid, c.trend.sma_slow, c.trend.slope_period) == (10, 20, 50, 5)
        assert (c.multi_timeframe.weekly_close_window, c.multi_timeframe.weekly_trend_window,
                c.multi_timeframe.weekly_trend_shift, c.multi_timeframe.weekly_rsi_period,
                c.multi_timeframe.weekly_momentum_period) == (5, 4, 5, 14, 25)
        assert (c.vol_regime.fast, c.vol_regime.slow, c.vol_regime.iv_rank_lookback, c.vol_regime.iv_rank_min_periods,
                c.vol_regime.vol_trend_period, c.vol_regime.vol_of_vol_window) == (5, 60, 252, 20, 5, 20)

    def test_config_yaml_lists_them_equal_to_the_defaults(self):
        assert load_settings().ml.features == MLConfig().features

    def test_the_thirteen_window_bearing_names_at_the_defaults(self):
        n = derive_names(FeaturesConfig())
        assert [n[k] for k in ("rsi_long", "rsi_short", "roc_short", "roc_mid", "roc_long", "rv_long", "rv_short",
                               "volume_ratio", "slope_fast", "slope_mid", "px_vs_mid", "px_vs_slow", "cross")] == [
            "rsi_14", "rsi_7", "roc_5", "roc_10", "roc_20", "realized_vol_20", "realized_vol_10",
            "volume_sma_20_ratio", "sma_10_slope", "sma_20_slope", "price_vs_sma_20", "price_vs_sma_50", "sma_10_20_cross"]
        assert (n["sma_fast"], n["sma_mid"], n["sma_slow"], n["ema_fast"], n["ema_slow"], n["atr_raw"]) == (
            "sma_10", "sma_20", "sma_50", "ema_12", "ema_26", "atr_14")

    def test_names_follow_the_windows(self):
        n = _spec(**{"momentum.rsi_long": 21, "trend.sma_mid": 30, "volatility.rv_long": 25}).names
        assert (n["rsi_long"], n["px_vs_mid"], n["cross"], n["rv_long"]) == (
            "rsi_21", "price_vs_sma_30", "sma_10_30_cross", "realized_vol_25")

    def test_the_meta_feature_list_follows_the_windows_and_keeps_its_order(self):
        base = spec_from_config(None).meta_feature_names()
        moved = _spec(**{"momentum.rsi_long": 21, "momentum.rsi_short": 5, "volume.volume_avg_window": 30,
                         "volatility.rv_long": 25, "trend.sma_mid": 30}).meta_feature_names()
        assert len(base) == len(moved) == 20
        diff = {b: m for b, m in zip(base, moved) if b != m}
        assert diff == {"rsi_14": "rsi_21", "rsi_7": "rsi_5", "volume_sma_20_ratio": "volume_sma_30_ratio",
                        "realized_vol_20": "realized_vol_25", "price_vs_sma_20": "price_vs_sma_30",
                        "sma_10_20_cross": "sma_10_30_cross"}

    @pytest.mark.parametrize("changes, match", [
        ({"momentum.rsi_short": 14}, "strictly increasing"), ({"momentum.macd_fast": 30}, "strictly increasing"),
        ({"trend.sma_slow": 20}, "strictly increasing"), ({"volatility.rv_short": 25}, "strictly increasing"),
        ({"vol_regime.fast": 12}, "strictly increasing"), ({"vol_regime.slow": 15}, "strictly increasing"),
        ({"volume.volume_fast_window": 40}, "strictly increasing"),
        ({"vol_regime.iv_rank_min_periods": 300}, "iv_rank_min_periods"),
        ({"momentum.roc_short": 10, "momentum.roc_mid": 11, "momentum.roc_long": 11}, "strictly increasing"),
    ])
    def test_bad_windows_fail_at_load(self, changes, match):
        with pytest.raises(Exception, match=match):
            _cfg(**changes)

    def test_derived_names_cannot_collide(self):
        with pytest.raises(ValueError, match="collide"):
            fs.validate_names({"a": "rsi_14", "b": "rsi_14"})
        with pytest.raises(ValueError, match="fixed"):
            fs.validate_names({"a": "macd"})

    def test_the_hash_is_stable_and_sensitive_to_every_window(self):
        base = spec_from_config(None).spec_hash()
        assert _spec().spec_hash() == base and len(base) == 16
        hashes = {key: _spec(**{key: EFFECTS[key][0]}).spec_hash() for key in EFFECTS}
        assert base not in hashes.values() and len(set(hashes.values())) == len(EFFECTS)

    def test_the_hash_changes_with_the_feature_set_version(self, monkeypatch):
        base = spec_from_config(None).spec_hash()
        monkeypatch.setattr(fs, "FEATURE_SET_VERSION", fs.FEATURE_SET_VERSION + 1)
        assert spec_from_config(None).spec_hash() != base

    def test_the_default_spec_follows_the_loaded_settings_and_is_cached(self, monkeypatch):
        a = fs.default_feature_spec()
        assert fs.default_feature_spec() is a                                    # cached
        _frozen(monkeypatch, _cfg(**{"momentum.rsi_long": 21}))
        b = fs.default_feature_spec()
        assert b.names["rsi_long"] == "rsi_21" and fs.default_feature_spec() is b
        assert FeatureEngine().get_feature_names()[0] == "rsi_21"

    def test_a_stub_settings_object_degrades_to_the_defaults(self, monkeypatch):
        import ait.config.settings as st
        from types import SimpleNamespace
        monkeypatch.setattr(st, "_FROZEN", SimpleNamespace())
        assert fs.default_feature_spec().spec_hash() == default_spec_hash()


# --------------------------------------------------------------------------
# 3. every window changes exactly the columns it should
# --------------------------------------------------------------------------

@pytest.mark.usefixtures("fast_fractal")
class TestEveryWindowChangesTheExpectedColumns:
    @pytest.fixture(scope="class")
    def ctx(self, fast_fractal):
        df = gi.frames()["calm_700"]
        return df, FeatureEngine(spec_from_config(None)).compute(df)

    @pytest.mark.parametrize("key", sorted(EFFECTS))
    def test_the_window(self, ctx, key):
        df, base = ctx
        value, renamed, changed = EFFECTS[key]
        f = FeatureEngine(_spec(**{key: value})).compute(df)
        gone, new = set(base.columns) - set(f.columns), set(f.columns) - set(base.columns)
        assert gone == set(renamed) and new == set(renamed.values()), key
        idx = base.index.intersection(f.index)
        moved = sorted(c for c in base.columns if c in f.columns
                       and not np.array_equal(base.loc[idx, c].to_numpy(float), f.loc[idx, c].to_numpy(float),
                                              equal_nan=True))
        assert moved == sorted(changed), key
        cfg = _cfg(**{key: value})
        for col in renamed.values():                    # the renamed column holds the NEW window, checked independently
            np.testing.assert_allclose(f[col].to_numpy(float), _oracle(col, cfg, df).loc[f.index].to_numpy(float),
                                       rtol=1e-9, atol=1e-12, err_msg=f"{key}: {col}")

    def test_a_renamed_column_holds_the_new_window(self, ctx):
        df, _ = ctx
        f = FeatureEngine(_spec(**{"momentum.rsi_long": 21})).compute(df)
        direct = FeatureEngine._rsi(df["Close"], 21)
        np.testing.assert_allclose(f["rsi_21"].to_numpy(), direct.loc[f.index].to_numpy())

    def test_the_feature_name_list_matches_the_computed_columns(self, ctx):
        df, _ = ctx
        for key in ("momentum.rsi_long", "trend.sma_mid", "volume.volume_avg_window", "volatility.rv_long"):
            fe = FeatureEngine(_spec(**{key: EFFECTS[key][0]}))
            f = fe.compute(df)
            assert [c for c in fe.get_feature_names() if c not in f.columns
                    and c.split("_")[0] not in ("vix", "rel", "spy", "correlation", "us", "yield", "dxy")
                    and not c.startswith(("hurst", "psd", "multifractal"))] == [], key

    def test_the_engine_exposes_its_spec_and_hash(self):
        fe = FeatureEngine(_spec(**{"momentum.rsi_long": 21}))
        assert fe.spec.names["rsi_long"] == "rsi_21" and fe.spec_hash() != FeatureEngine(spec_from_config(None)).spec_hash()
        assert FeatureEngine().spec_hash() == spec_from_config(None).spec_hash()

    def test_iv_rank_min_periods_matters_once_it_exceeds_the_warm_up(self, ctx):
        df, base = ctx
        f = FeatureEngine(_spec(**{"vol_regime.iv_rank_min_periods": 200})).compute(df)
        assert len(f) < len(base)                  # inert at small values: the warm-up rows are dropped anyway


# --------------------------------------------------------------------------
# 4. consumers follow the windows
# --------------------------------------------------------------------------

WINDOWS = {"momentum.rsi_long": 21, "momentum.rsi_short": 5, "volume.volume_avg_window": 30, "volatility.rv_long": 25,
           "trend.sma_mid": 30}


def _moved_settings():
    cfg = _cfg(**WINDOWS)
    s = load_settings()
    return s.model_copy(update={"ml": s.ml.model_copy(update={"features": cfg})}), FeatureSpec.from_config(cfg)


@pytest.mark.usefixtures("fast_fractal")
class TestConsumersFollowTheWindows:
    def test_the_engine_meta_context_and_entry_snapshot_use_the_derived_names(self):
        settings, spec = _moved_settings()
        got = gi.engine_meta_contexts(settings=settings, spec=spec)
        keys = set(got["contexts"][-1])
        assert {"rsi_21", "rsi_5", "volume_sma_30_ratio", "realized_vol_25", "price_vs_sma_30", "sma_10_30_cross"} <= keys
        assert not keys & {"rsi_14", "rsi_7", "volume_sma_20_ratio", "realized_vol_20", "price_vs_sma_20", "sma_10_20_cross"}
        assert "rsi_21" in got["snapshot"] and "realized_vol_25" in got["snapshot"] and "rsi_14" not in got["snapshot"]
        last = got["contexts"][-1]
        assert last["price_vs_sma_30"] != 0.0 and last["rsi_21"] != 50.0 and last["realized_vol_25"] != 0.2   # real values, not defaults
        # and the keys are exactly the spec's meta feature names (what the meta-labeler trains on)
        assert keys == set(spec.meta_feature_names()) | {"primary_confidence", "hour_of_day", "sentiment_score"} - set()

    def test_the_meta_labeler_trains_and_predicts_on_the_derived_columns(self):
        from ait.ml.meta_label import MetaLabeler
        _, spec = _moved_settings()
        feats = FeatureEngine(spec).compute(gi.frames()["calm_700"])
        m = MetaLabeler(feature_spec=spec)
        assert m._meta_features == spec.meta_feature_names()
        data = m.build_training_data_from_backtest(gi.fixed_trades(feats), feats)
        assert {"rsi_21", "rsi_5", "price_vs_sma_30", "sma_10_30_cross", "volume_sma_30_ratio",
                "realized_vol_25"} <= set(data.columns)
        assert not set(data.columns) & {"rsi_14", "sma_10_20_cross", "price_vs_sma_20", "realized_vol_20"}
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({f: rng.normal(size=80) for f in spec.meta_feature_names()})
        frame["profitable"] = (rng.random(80) > 0.5).astype(int)
        m.train(frame)
        assert m.is_trained and "rsi_21" in m._feature_names and "rsi_14" not in m._feature_names

    def test_the_live_orchestrator_snapshot_helper(self):
        from ait.ml.meta_label import live_technical_features
        _, spec = _moved_settings()
        feats = FeatureEngine(spec).compute(gi.frames()["calm_700"])
        out = live_technical_features(feats.iloc[-1], spec.names)
        assert set(out) == {"rsi_21", "rsi_5", "bb_position", "volume_sma_30_ratio", "realized_vol_25", "atr_pct",
                            "weekly_trend_aligned", "volume_confirmation", "macd_hist", "price_vs_sma_30",
                            "sma_10_30_cross"}
        assert out["rsi_21"] == pytest.approx(float(feats.iloc[-1]["rsi_21"]))
        default = live_technical_features(FeatureEngine().compute(gi.frames()["calm_700"]).iloc[-1],
                                          spec_from_config(None).names)
        assert "rsi_14" in default and len(default) == 11

    def test_the_entry_signals_snapshot_uses_the_spec_names(self, monkeypatch):
        """The orchestrator's persisted snapshot is read back by the meta-labeler: both must use the same names."""
        _frozen(monkeypatch, _cfg(**WINDOWS))
        from ait.bot.orchestrator import TradingOrchestrator as T
        feats = FeatureEngine().compute(gi.frames()["calm_700"])
        o = T.__new__(T)
        o._entry_feature_snap = {"QQQ": feats.iloc[-1]}
        out = json.loads(o._entry_signals_json("QQQ"))
        assert "rsi_21" in out and "price_vs_sma_30" in out and "rsi_14" not in out

    def test_the_walk_forward_holds_its_runs_spec(self, monkeypatch):
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        cfg = WalkForwardConfig(train_window_models=False)
        assert WalkForwardBacktester(symbols=["QQQ"], strategies=["iron_condor"], config=cfg)._fspec.spec_hash() == default_spec_hash()
        _frozen(monkeypatch, _cfg(**WINDOWS))
        assert WalkForwardBacktester(symbols=["QQQ"], strategies=["iron_condor"], config=cfg)._fspec.spec_hash() == (
            _spec(**WINDOWS).spec_hash())

    def test_the_dashboard_timeseries_keep_their_stable_keys_with_the_new_windows(self, tmp_path, monkeypatch):
        from ait.backtesting.walkforward import _save_window_timeseries
        _frozen(monkeypatch, _cfg(**WINDOWS))
        frame = gi.frames()["calm_700"]
        feats = FeatureEngine().compute(frame)
        _save_window_timeseries(frame.tail(30), frame, None, None, None, tmp_path, "QQQ", 1, precomputed_features=feats)
        bars = json.loads((tmp_path / "timeseries_bars.json").read_text())
        assert len(bars) == 30
        last, row = bars[-1], feats.loc[pd.Timestamp(bars[-1]["time"])]
        assert last["rsi_14"] == pytest.approx(round(float(row["rsi_21"]), 4))
        assert last["sma_20"] == pytest.approx(round(float(row["sma_30"]), 4))
        assert last["realized_vol_20"] == pytest.approx(round(float(row["realized_vol_25"]), 4))
        assert last["volume_ratio"] == pytest.approx(round(float(row["volume_sma_30_ratio"]), 4))

    def test_the_dashboard_series_keep_their_stable_keys(self):
        from ait.dashboard.walkforward.export import _feature_rename
        _, spec = _moved_settings()
        assert _feature_rename(spec.names) == {"volume_sma_30_ratio": "volume_ratio", "rsi_21": "rsi_14",
                                               "sma_30": "sma_20", "sma_50": "sma_50", "realized_vol_25": "realized_vol_20"}
        assert _feature_rename(spec_from_config(None).names)["volume_sma_20_ratio"] == "volume_ratio"


# --------------------------------------------------------------------------
# 5. a model trained under other windows is not served
# --------------------------------------------------------------------------

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


def _df():
    return gi.frames()["calm_700"].iloc[:300]


@pytest.mark.usefixtures("fast_fractal")
class TestModelsTrainedUnderOtherWindowsAreRefused:
    def test_the_ensemble(self, tmp_path, criticals):
        from ait.ml.ensemble import DirectionPredictor
        moved = MLConfig(features=_cfg(**{"momentum.rsi_long": 21}))
        DirectionPredictor(moved, model_dir=tmp_path).train(_df(), symbol="QQQ")
        saved = pickle.loads((tmp_path / "ensemble.pkl").read_bytes())
        assert saved["spec"]["feature_spec_hash"] == _spec(**{"momentum.rsi_long": 21}).spec_hash()
        assert "rsi_21" in saved["feature_names"]
        assert DirectionPredictor(moved, model_dir=tmp_path).load_models() is True and criticals == []
        stale = DirectionPredictor(MLConfig(), model_dir=tmp_path)
        assert stale.load_models() is False and not stale.is_trained
        assert criticals[0][0] == "model_feature_spec_mismatch" and criticals[0][1]["artifact"] == "ensemble"

    @pytest.mark.parametrize("mod, cls, file, thr", [("range_predictor", "RangePredictor", "range.pkl", 0.05),
                                                    ("vol_magnitude_predictor", "VolMagnitudePredictor", "vol_magnitude.pkl", 0.04)])
    def test_the_binary_predictors(self, tmp_path, criticals, mod, cls, file, thr, monkeypatch):
        import importlib
        C = getattr(importlib.import_module(f"ait.ml.{mod}"), cls)
        a = C(threshold_pct=thr, horizon_days=10, model_dir=tmp_path)
        a.train(_df(), symbol="QQQ")
        a._save_models(tmp_path)
        assert pickle.loads((tmp_path / file).read_bytes())["spec"]["feature_spec_hash"] == default_spec_hash()
        assert C(threshold_pct=thr, horizon_days=10, model_dir=tmp_path).load_models() is True and criticals == []
        _frozen(monkeypatch, _cfg(**{"trend.sma_mid": 30}))                    # the config moved after the model was trained
        assert C(threshold_pct=thr, horizon_days=10, model_dir=tmp_path).load_models() is False
        assert criticals[0][1]["artifact"] in ("range", "vol_magnitude")

    def test_the_meta_labeler(self, tmp_path, criticals, monkeypatch):
        import ait.ml.meta_label as ml
        monkeypatch.setattr(ml, "MODEL_DIR", tmp_path)
        spec = _spec(**{"momentum.rsi_long": 21})
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({f: rng.normal(size=80) for f in spec.meta_feature_names()})
        frame["profitable"] = (rng.random(80) > 0.5).astype(int)
        m = ml.MetaLabeler(feature_spec=spec)
        m.train(frame)
        m.save_to_path(tmp_path / "meta_label.pkl")
        assert ml.MetaLabeler(feature_spec=spec).load_model() is True and criticals == []
        assert ml.MetaLabeler().load_model() is False                         # default windows: refused
        assert criticals[0][1]["artifact"] == "meta_label"

    def test_a_model_with_no_recorded_hash_is_assumed_to_carry_the_default_windows(self, tmp_path, criticals):
        from ait.ml.ensemble import DirectionPredictor
        DirectionPredictor(MLConfig(), model_dir=tmp_path).train(_df(), symbol="QQQ")
        data = pickle.loads((tmp_path / "ensemble.pkl").read_bytes())
        data.pop("spec")                                                      # a pre-W88c artifact
        (tmp_path / "ensemble.pkl").write_bytes(pickle.dumps(data))
        assert DirectionPredictor(MLConfig(), model_dir=tmp_path).load_models() is True and criticals == []
        moved = DirectionPredictor(MLConfig(features=_cfg(**{"momentum.rsi_long": 21})), model_dir=tmp_path)
        assert moved.load_models() is False                                    # ... and refused under other windows
        data["spec"] = {"hyperparams": {}, "feature_spec_hash": None}          # a W88c-era artifact
        (tmp_path / "ensemble.pkl").write_bytes(pickle.dumps(data))
        assert DirectionPredictor(MLConfig(), model_dir=tmp_path).load_models() is True

    @pytest.mark.parametrize("name", ["ensemble.pkl", "range.pkl", "vol_magnitude.pkl", "meta_label.pkl"])
    def test_the_real_live_artifacts_still_load_at_the_default_windows_and_not_at_others(self, name, criticals):
        live = ROOT / "models" / name
        if not live.exists():
            pytest.skip("no live artifact in this checkout")
        stored = pickle.loads(live.read_bytes()).get("spec")
        assert hp.feature_spec_refused(name, stored, default_spec_hash()) is False
        assert hp.feature_spec_refused(name, stored, _spec(**{"momentum.rsi_long": 21}).spec_hash()) is True

    def test_the_real_live_ensemble_loads_through_the_predictor(self, criticals):
        live = ROOT / "models" / "ensemble.pkl"
        if not live.exists():
            pytest.skip("no live artifact in this checkout")
        from ait.ml.ensemble import DirectionPredictor
        assert DirectionPredictor(MLConfig(), model_dir=live.parent, persist_artifacts=False).load_models() is True
        assert criticals == []

    def test_the_engine_drops_a_predictor_computed_under_other_windows(self):
        from ait.backtesting.engine import Backtester
        from ait.ml.ensemble import DirectionPredictor
        settings, spec = _moved_settings()
        data = _df()
        default_predictor = DirectionPredictor(MLConfig(), persist_artifacts=False)
        kept = Backtester(data=data, strategies=["iron_condor"], settings=load_settings(), predictor=default_predictor)
        assert kept._predictor is default_predictor
        dropped = Backtester(data=data, strategies=["iron_condor"], settings=settings, predictor=default_predictor)
        assert dropped._predictor is None                       # would have predicted on zero-filled columns
        match = Backtester(data=data, strategies=["iron_condor"], settings=settings,
                           predictor=DirectionPredictor(settings.ml, persist_artifacts=False))
        assert match._predictor is not None


# --------------------------------------------------------------------------
# 6. visibility
# --------------------------------------------------------------------------

@pytest.mark.usefixtures("fast_fractal")
class TestVisibility:
    def test_the_manifest_warns_when_research_differs_from_live(self, monkeypatch):
        import run_backtest as rb
        s = load_settings()
        moved = s.model_copy(update={"ml": s.ml.model_copy(update={"features": _cfg(**{"momentum.rsi_long": 21})})})
        for research, expect in ((s, False), (moved, True)):
            monkeypatch.setattr(rb, "_research_settings", lambda r=research: r)
            monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
            m = rb.build_parity_manifest(rb.parse_args())
            assert m["live"]["ml_features"] == s.ml.features.model_dump()
            assert any(w.startswith("ml_features") for w in m["parity_warnings"]) is expect

    def test_the_run_record_carries_the_hash(self):
        from ait.backtesting import run_record as rr
        settings, spec = _moved_settings()
        assert rr.build_run_record(kind="backtest", name="x", settings=settings, config_path=ROOT / "config.yaml",
                                   overlay_path=None)["feature_spec_hash"] == spec.spec_hash()
        assert rr.build_run_record(kind="backtest", name="x", settings=None, config_path=ROOT / "config.yaml",
                                   overlay_path=None)["feature_spec_hash"] == default_spec_hash()

    def test_the_effective_view_shows_them_with_their_origin(self):
        from ait.config.research import settings_origin
        got = settings_origin("config.yaml")
        assert got["ml.features.momentum.rsi_long"] == (14, "config.yaml")
        assert got["ml.features.vol_regime.slow"] == (60, "config.yaml")

    def test_a_study_can_change_a_window_for_one_run(self, tmp_path):
        from ait.config.research import load_research_settings
        p = tmp_path / "s.yaml"
        p.write_text("name: s\nquestion: q\nrule: exploratory\nsettings: {ml: {features: {momentum: {rsi_long: 21}}}}\n")
        s = load_research_settings("config.yaml", p)
        assert s.ml.features.momentum.rsi_long == 21 and s.ml.features.momentum.rsi_short == 7

    def test_a_bad_window_in_a_study_is_refused(self, tmp_path):
        from ait.config.research import load_research_settings
        p = tmp_path / "s.yaml"
        p.write_text("name: s\nquestion: q\nrule: exploratory\nsettings: {ml: {features: {momentum: {rsi_short: 30}}}}\n")
        with pytest.raises(Exception, match="strictly increasing"):
            load_research_settings("config.yaml", p)


# --------------------------------------------------------------------------
# 7. import order (a class guard: W88d's first version created a cycle only when ait.data was imported first)
# --------------------------------------------------------------------------

class TestEveryEntryPointImportsFirst:
    """``Settings`` is built at import time, so anything its validators import must not pull in ``ait.ml`` (whose package init
    imports the broker, which imports settings). Each of these used to work as the FIRST import of a fresh interpreter."""

    @pytest.mark.parametrize("module", ["ait.config.settings", "ait.data", "ait.broker", "ait.strategies", "ait.ml",
                                        "ait.main", "ait.backtesting.engine", "ait.optimization.optimizer"])
    def test_as_the_first_import(self, module):
        import subprocess
        r = subprocess.run([sys.executable, "-I", "-c", f"import {module}"], capture_output=True, text=True, cwd=ROOT,
                           env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}, timeout=300)
        assert r.returncode == 0, r.stderr[-800:]

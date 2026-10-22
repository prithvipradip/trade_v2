"""W87a: research hygiene, behaviour-identical.

  * the model hyperparameters now come from ONE module and every estimator is built with exactly the
    keyword set the three predictors used before
  * the engine's IV proxy and the replay's mirror share one definition and agree on a grid
  * the four study scripts build the SAME ``Backtester`` keywords as before, through the shared harness helper
  * the parity manifest's live column reads its authorities (TP ladder, IV floor, short delta)
  * walk-forward defaults follow config (commission, CLI window lengths)
  * the search space can be overridden per study without touching the defaults, validated, hashed, and
    used at every point the optimizer reads it
Results-moving items (slippage, default symbols / strategies) are in ``test_w87a_results_moving.py``.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"w87a_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ohlcv(days: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, days))
    return pd.DataFrame({"Open": close, "High": close * (1 + abs(rng.normal(0, .005, days))),
                         "Low": close * (1 - abs(rng.normal(0, .005, days))), "Close": close,
                         "Volume": rng.integers(1_000_000, 10_000_000, days).astype(float)}, index=idx)


# --------------------------------------------------------------------------
# 1. model hyperparameters: one owner, byte-identical keyword sets
# --------------------------------------------------------------------------

XGB_MULTI = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 objective="multi:softprob", num_class=3, eval_metric="mlogloss", verbosity=0, n_jobs=-1,
                 random_state=42)
LGB_MULTI = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 objective="multiclass", num_class=3, metric="multi_logloss", verbose=-1, n_jobs=-1,
                 class_weight="balanced", random_state=42, deterministic=True)
XGB_BINARY = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  objective="binary:logistic", eval_metric="logloss", verbosity=0, n_jobs=-1, random_state=42)
LGB_BINARY = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  objective="binary", metric="binary_logloss", verbose=-1, n_jobs=-1,
                  class_weight="balanced", random_state=42, deterministic=True)


@pytest.fixture
def recorded(monkeypatch):
    """Record the keywords every XGBClassifier / LGBMClassifier is constructed with (and still build it)."""
    import lightgbm
    import xgboost
    seen = {"xgb": [], "lgb": []}
    real_x, real_l = xgboost.XGBClassifier, lightgbm.LGBMClassifier
    monkeypatch.setattr(xgboost, "XGBClassifier", lambda **kw: (seen["xgb"].append(dict(kw)), real_x(**kw))[1])
    monkeypatch.setattr(lightgbm, "LGBMClassifier", lambda **kw: (seen["lgb"].append(dict(kw)), real_l(**kw))[1])
    return seen


class TestHyperparameters:
    def test_the_builders_equal_the_old_literals(self):
        from ait.ml import hyperparams as hp
        assert hp.xgb_multiclass_kwargs() == XGB_MULTI
        assert hp.lgbm_multiclass_kwargs() == LGB_MULTI
        assert hp.xgb_binary_kwargs(2.5) == {**XGB_BINARY, "scale_pos_weight": 2.5}
        assert hp.lgbm_binary_kwargs() == LGB_BINARY

    def test_builders_return_fresh_dicts(self):
        from ait.ml import hyperparams as hp
        a = hp.xgb_multiclass_kwargs()
        a["n_estimators"] = 1
        assert hp.xgb_multiclass_kwargs()["n_estimators"] == 200 and hp.TREE_BASE["n_estimators"] == 200

    def test_the_direction_ensemble_builds_exactly_the_old_models(self, recorded):
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        DirectionPredictor(MLConfig(), persist_artifacts=False).train(_ohlcv(), symbol="QQQ")
        assert recorded["xgb"][0] == XGB_MULTI and recorded["lgb"][0] == LGB_MULTI

    def test_tuned_hyperparameters_still_override_the_defaults(self, recorded):
        from ait.config.settings import MLConfig
        from ait.ml.ensemble import DirectionPredictor
        p = DirectionPredictor(MLConfig(), persist_artifacts=False)
        p._xgb_kwargs, p._lgbm_kwargs = {"n_estimators": 77}, {"num_leaves": 9}
        p.train(_ohlcv(), symbol="QQQ")
        assert recorded["xgb"][0]["n_estimators"] == 77 and recorded["lgb"][0]["num_leaves"] == 9
        assert recorded["xgb"][0]["max_depth"] == 5

    @pytest.mark.parametrize("mod, cls, thr", [("range_predictor", "RangePredictor", 0.05),
                                              ("vol_magnitude_predictor", "VolMagnitudePredictor", 0.04)])
    def test_the_binary_predictors_build_exactly_the_old_models(self, recorded, mod, cls, thr):
        import importlib
        predictor = getattr(importlib.import_module(f"ait.ml.{mod}"), cls)(threshold_pct=thr, horizon_days=10)
        predictor.train(_ohlcv(), symbol="QQQ")
        first = dict(recorded["xgb"][0])
        assert first.pop("scale_pos_weight") > 0 and first == XGB_BINARY
        assert recorded["lgb"][0] == LGB_BINARY

    def test_the_models_that_differ_on_purpose_keep_their_own_profile(self):
        """meta_label (100 trees, depth 3, no seed) and the trainer's search objective must not be folded in."""
        src = (ROOT / "src" / "ait" / "ml" / "meta_label.py").read_text()
        assert "n_estimators=100" in src and "max_depth=3" in src and "random_state" not in src.split(
            "def train")[1].split("def ")[0]

    def test_the_binary_guards_are_shared(self):
        from ait.ml import hyperparams as hp
        from ait.ml import range_predictor as rp
        from ait.ml import vol_magnitude_predictor as vp
        assert (hp.POSITIVE_RATE_MIN, hp.POSITIVE_RATE_MAX, hp.MIN_CLASS_SAMPLES) == (0.05, 0.98, 10)
        for m in (rp, vp):
            assert m.POSITIVE_RATE_MIN is hp.POSITIVE_RATE_MIN or m.POSITIVE_RATE_MIN == 0.05
            assert m.POSITIVE_RATE_MAX == 0.98 and m.MIN_CLASS_SAMPLES == 10


# --------------------------------------------------------------------------
# 2. the IV proxy: one definition, engine == replay
# --------------------------------------------------------------------------

class TestIvModel:
    @staticmethod
    def _engine(floor):
        from ait.backtesting.engine import Backtester
        bt = Backtester.__new__(Backtester)
        bt._iv_floor = floor
        return bt

    def test_constants(self):
        from ait.backtesting import iv_model as m
        assert (m.SYNTHETIC_IV_PREMIUM, m.REMARK_ENTRY_WEIGHT, m.REMARK_RV_WEIGHT, m.REMARK_MIN_CLOSES,
                m.RV_WINDOW) == (1.15, 0.70, 0.30, 21, 20)
        assert m.VOL_MULT_VS_VIX == {"SPY": 1.00, "QQQ": 1.228, "IWM": 1.33, "DIA": 0.98}
        assert (m.VOL_MULT_DEFAULT, m.HALF_SPREAD_REF_IV, m.HALF_SPREAD_REF_DTE) == (1.10, 0.20, 21)

    def test_engine_and_replay_agree_on_a_grid(self):
        from ait.backtesting.replay import engine_current_iv
        rng = np.random.default_rng(7)
        for n in (5, 20, 21, 22, 60, 130):
            closes = 400 * np.cumprod(1 + rng.normal(0, 0.012, n))
            for entry in (0.08, 0.2, 0.55):
                for floor in (0.05, 0.2, 0.9):
                    hist = pd.DataFrame({"Close": closes})
                    got = self._engine(floor)._get_current_iv({"entry_iv": entry}, hist)
                    assert got == engine_current_iv(entry, list(closes), floor), (n, entry, floor)

    def test_the_shared_function_matches_the_old_inline_arithmetic(self):
        from ait.backtesting import iv_model as m
        from ait.backtesting.pricing import realized_vol
        closes = 400 * np.cumprod(1 + np.random.default_rng(3).normal(0, 0.012, 60))
        rv = realized_vol(closes, window=20) * 1.15
        assert m.synthetic_iv(closes) == rv
        assert m.remark_iv(0.3, closes, 0.05) == max(0.70 * 0.3 + 0.30 * rv, 0.05)
        assert m.remark_iv(0.3, closes[:20], 0.05) == 0.3                 # fewer than 21 closes

    def test_the_engine_fallback_iv_is_the_shared_one(self):
        from ait.backtesting import iv_model as m
        hist = pd.DataFrame({"Close": 400 * np.cumprod(1 + np.random.default_rng(4).normal(0, 0.012, 60))})
        assert self._engine(0.05)._get_iv(hist, None) == m.synthetic_iv(hist["Close"].values)

    def test_engine_aliases_and_the_replay_rate_stay_importable(self):
        from ait.backtesting import iv_model as m
        from ait.backtesting.engine import Backtester
        from ait.backtesting.replay import ENGINE_RISK_FREE_RATE
        assert Backtester._VOL_MULT_VS_VIX == m.VOL_MULT_VS_VIX and Backtester._VOL_MULT_DEFAULT == 1.10
        assert ENGINE_RISK_FREE_RATE == 0.05

    @pytest.mark.parametrize("iv, dte", [(0.15, 30), (0.2, 21), (0.35, 10), (0.6, 1)])
    def test_the_legacy_half_spread_is_unchanged(self, iv, dte):
        from ait.backtesting.engine import Backtester
        bt = Backtester.__new__(Backtester)
        bt._spread_base, bt._spread_iv_sensitivity, bt._spread_dte_sensitivity, bt._spread_cap = (
            0.03, 0.10, 0.005, 0.5)
        want = min(0.03 + 0.10 * max(0.0, iv - 0.20) + 0.005 * max(0.0, 21 - dte), 0.5)
        assert bt._options_half_spread(iv, dte) == pytest.approx(want)


# --------------------------------------------------------------------------
# 3. the four study scripts: the same Backtester keywords as before
# --------------------------------------------------------------------------

COMMON = dict(strategies=["iron_condor"], initial_capital=100_000.0, macro_event_gate=False,
              allow_live_model_fallback=False, condor_any_direction=True, symbol="QQQ",
              friction_model="per_leg_usd", gates_mode="observe", max_contracts_per_trade=1,
              max_concurrent_positions=1, hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0,
              iv_floor=0.05, ic_min_credit=0.30, ic_min_credit_width=0.05, entry_window_start_et="10:30",
              entry_window_end_et="15:30", scan_interval_minutes=60, engine_model="realistic",
              friction_short_dte=True, pricing_source="bs_legacy")
OBJECTS = {"data", "features_cache", "intraday_store", "db_path", "iv_rank_series"}


class _FakeRD:
    def __init__(self, db, symbol):
        self.symbol = symbol

    def daily_frame(self):
        f = _ohlcv(450)
        f["implied_vol"] = 0.2
        return f


class _FakeBT:
    last: dict = {}

    def __init__(self, **kw):
        type(self).last = kw
        self._leg_spread_model, self._friction_bands = object(), object()
        self._regime_veto_gate = kw.get("regime_veto_gate")
        self._ex_div_mode, self._ex_dates = kw.get("ex_div_entry_gate"), [date(2024, 1, 1)]
        self._funnel = {}

    def run(self):
        return SimpleNamespace(trades=[])


@pytest.fixture
def fake_engine(monkeypatch):
    import ait.backtesting.engine as engine_mod
    import ait.backtesting.replay as replay_mod
    monkeypatch.setattr(engine_mod, "Backtester", _FakeBT)
    monkeypatch.setattr(replay_mod, "ResearchData", _FakeRD)
    return _FakeBT


def _scalars(kw: dict) -> dict:
    return {k: v for k, v in kw.items() if k not in OBJECTS}


class TestStudyScriptsBuildTheSameEngine:
    START, END = "2024-01-02", "2024-06-28"

    def test_gate_parity_ab(self, fake_engine):
        m = _load_script("gate_parity_ab")
        m.run_arm({"symbol": "QQQ", "arm": "baseline", "start": self.START, "end": self.END, "entry_dte": 14,
                   "regime_veto_gate": True, "vol_cap": None})
        kw = fake_engine.last
        assert _scalars(kw) == {**COMMON, "entry_dte": 14, "iv_rank_gate": True, "iv_rank_rise_gate": False,
                                "regime_veto_gate": True}
        assert OBJECTS - {"iv_rank_series"} <= set(kw) and "iv_rank_series" in kw
        m.run_arm({"symbol": "QQQ", "arm": "no_vol", "start": self.START, "end": self.END, "entry_dte": 7,
                   "regime_veto_gate": False, "vol_cap": 10.0})
        assert _scalars(fake_engine.last) == {**COMMON, "entry_dte": 7, "iv_rank_gate": True,
                                              "iv_rank_rise_gate": False, "regime_veto_gate": False,
                                              "max_entry_vol_annual": 10.0}

    def test_ex_div_gate_ab(self, fake_engine):
        m = _load_script("ex_div_gate_ab")
        m.run_arm({"symbol": "QQQ", "entry_dte": 14, "arm": "live", "mode": "live", "k": 5,
                   "start": self.START, "end": self.END})
        assert _scalars(fake_engine.last) == {**COMMON, "entry_dte": 14, "iv_rank_gate": True,
                                              "iv_rank_rise_gate": False, "ex_div_entry_gate": "live",
                                              "ex_div_entry_k_days": 5}

    def test_iv_rank_parity_probe(self, fake_engine):
        m = _load_script("iv_rank_parity_probe")
        m._arm({"symbol": "QQQ", "threshold": 0.3, "gate": True, "start": self.START, "end": self.END})
        assert _scalars(fake_engine.last) == {**COMMON, "entry_dte": 14, "iv_rank_gate": True,
                                              "iv_rank_rise_gate": True, "iv_rank_rise_threshold": 0.3}

    def test_dte_dimension_flatness(self, fake_engine, monkeypatch):
        m = _load_script("dte_dimension_flatness")
        monkeypatch.setattr(m, "_ReadOnlyIntraday", lambda db, sym: "ro-store")
        m._run_one({"symbol": "QQQ", "start": self.START, "end": self.END, "dte": 3,
                    "kwargs": {"expiry_guard_dte": 0}, "dim": "guard", "value": 0})
        kw = fake_engine.last
        assert _scalars(kw) == {**COMMON, "entry_dte": 3, "iv_rank_rise_gate": False, "expiry_guard_dte": 0}
        assert kw["features_cache"] is None and "iv_rank_gate" not in kw and "iv_rank_series" not in kw
        assert kw["intraday_store"] == "ro-store"

    def test_the_helper_itself(self):
        from ait.backtesting import study_harness as sh
        kw = sh.study_engine_kwargs(data="d", symbol="QQQ", db_path=Path("x.db"), intraday_store="s",
                                    entry_dte=14, iv_rank_series="ivr", extra_flag=1)
        assert _scalars(kw) == {**COMMON, "entry_dte": 14, "iv_rank_rise_gate": False, "iv_rank_gate": True,
                                "extra_flag": 1}
        assert (sh.STUDY_IC_MIN_CREDIT, sh.STUDY_IC_MIN_CREDIT_WIDTH, sh.STUDY_IV_FLOOR) == (0.30, 0.05, 0.05)

    def test_the_study_floors_are_looser_than_live_and_say_so(self):
        """The difference from live is the point of naming these: pin it so nobody 'tidies' it away."""
        from ait.backtesting import study_harness as sh
        from ait.config.runtime_env import CONTRACT_DEFAULTS
        assert sh.STUDY_IC_MIN_CREDIT < float(CONTRACT_DEFAULTS["AIT_IC_MIN_CREDIT"])
        assert sh.STUDY_IC_MIN_CREDIT_WIDTH < float(CONTRACT_DEFAULTS["AIT_IC_MIN_CREDIT_WIDTH"])

    def test_assert_friction_loaded(self):
        from ait.backtesting import study_harness as sh
        sh.assert_friction_loaded(SimpleNamespace(_leg_spread_model=1, _friction_bands=1))
        with pytest.raises(AssertionError, match="per-leg friction"):
            sh.assert_friction_loaded(SimpleNamespace(_leg_spread_model=None, _friction_bands=1))
        with pytest.raises(AssertionError, match="bands"):
            sh.assert_friction_loaded(SimpleNamespace(_leg_spread_model=1, _friction_bands=None))


# --------------------------------------------------------------------------
# 4. search-space overrides: validated, hashed, used at every read point
# --------------------------------------------------------------------------

class TestSearchSpaceOverrides:
    def test_no_override_is_the_default_and_a_new_dict(self):
        from ait.optimization.param_spaces import IRON_CONDOR_SPACE, effective_space
        e = effective_space("iron_condor")
        assert e == IRON_CONDOR_SPACE and e is not IRON_CONDOR_SPACE

    def test_an_override_replaces_one_range_and_never_mutates_the_defaults(self):
        from ait.optimization.param_spaces import IRON_CONDOR_SPACE, STRATEGY_SPACES, effective_space
        before = dict(IRON_CONDOR_SPACE)
        e = effective_space("iron_condor", {"wing_k": [1.0, 2.0], "max_hold_days": [2, 8]})
        assert e["wing_k"] == ("float", 1.0, 2.0) and e["max_hold_days"] == ("int", 2, 8)
        assert {k: v for k, v in e.items() if k not in ("wing_k", "max_hold_days")} == {
            k: v for k, v in before.items() if k not in ("wing_k", "max_hold_days")}
        assert IRON_CONDOR_SPACE == before and STRATEGY_SPACES["iron_condor"] is IRON_CONDOR_SPACE

    def test_a_fractal_override_stays_inside_its_strategy(self):
        """FRACTAL_GATE_SPACE is spread into two spaces at import: an override must not leak across them."""
        from ait.optimization.param_spaces import SHORT_STRANGLE_SPACE, effective_space
        e = effective_space("iron_condor", {"hurst_regime_threshold": [0.1, 0.2]})
        assert e["hurst_regime_threshold"] == ("float", 0.1, 0.2)
        assert effective_space("short_strangle")["hurst_regime_threshold"] == \
            SHORT_STRANGLE_SPACE["hurst_regime_threshold"]

    @pytest.mark.parametrize("strategy, over, match", [
        ("nope", {}, "unknown strategy"),
        ("iron_condor", {"nonsense": [1, 2]}, "not a dimension"),
        ("iron_condor", {"entry_dte": [7, 14]}, "settings.backtest.dte_search"),
        ("iron_condor", {"wing_k": 1.5}, r"\[low, high\]"),
        ("iron_condor", {"wing_k": [1.0]}, r"\[low, high\]"),
        ("iron_condor", {"wing_k": [True, 2.0]}, r"\[low, high\]"),
        ("iron_condor", {"wing_k": ["a", "b"]}, r"\[low, high\]"),
        ("iron_condor", {"wing_k": [2.0, 1.0]}, "low must be below high"),
        ("iron_condor", {"wing_k": [1.0, 1.0]}, "low must be below high"),
        ("iron_condor", {"max_hold_days": [1.5, 8]}, "whole-number"),
        ("iron_condor", {"delta_short": [0.01, 0.2]}, "hard limits"),
        ("iron_condor", {"delta_short": [0.1, 0.6]}, "hard limits"),
        ("iron_condor", {"wing_k": [0.0, 2.0]}, "hard limits"),
    ])
    def test_bad_overrides_fail_loudly_and_name_the_key(self, strategy, over, match):
        from ait.optimization.param_spaces import effective_space
        with pytest.raises(ValueError, match=match):
            effective_space(strategy, over)

    def test_the_error_names_the_study_key(self):
        from ait.optimization.param_spaces import effective_space
        with pytest.raises(ValueError, match=r"search_space\.iron_condor\.wing_k"):
            effective_space("iron_condor", {"wing_k": [3.0, 1.0]})

    def test_a_range_that_excludes_the_live_value_warns(self):
        from ait.optimization.param_spaces import override_warnings
        base = {"wing_k": 1.6, "delta_short": 0.20}
        assert override_warnings("iron_condor", {"wing_k": [1.0, 2.0]}, base) == []
        w = override_warnings("iron_condor", {"wing_k": [0.3, 1.2], "delta_short": [0.1, 0.15]}, base)
        assert len(w) == 2 and "excludes the live value 1.6" in w[0]

    def test_the_hash_is_stable_and_sensitive(self):
        from ait.optimization.param_spaces import effective_space, space_hash
        a = {"iron_condor": effective_space("iron_condor")}
        assert space_hash(a) == space_hash({"iron_condor": effective_space("iron_condor")})
        b = {"iron_condor": effective_space("iron_condor", {"wing_k": [1.0, 2.0]})}
        assert space_hash(a) != space_hash(b)
        reordered = {"iron_condor": dict(reversed(list(a["iron_condor"].items())))}
        assert space_hash(a) == space_hash(reordered)                    # dict order is not content
        assert len(space_hash(a)) == 16
        assert space_hash({"iron_condor": effective_space("iron_condor")}) != space_hash(
            {"put_credit_spread": effective_space("put_credit_spread")})


class TestOptimizerUsesTheEffectiveSpace:
    @staticmethod
    def _opt(**kw):
        from ait.optimization.optimizer import StrategyOptimizer
        return StrategyOptimizer(symbols=["SPY"], strategies=kw.pop("strategies", ["iron_condor"]),
                                 n_trials=1, **kw)

    def test_trials_sample_inside_the_overridden_range(self):
        import optuna
        opt = self._opt(search_space_overrides={"iron_condor": {"wing_k": [1.0, 2.0]}})
        study = optuna.create_study()
        vals = [opt._suggest_params(study.ask())["iron_condor__wing_k"] for _ in range(40)]
        assert all(1.0 <= v <= 2.0 for v in vals) and max(vals) > 1.3
        default = self._opt()
        study2 = optuna.create_study()
        dv = [default._suggest_params(study2.ask())["iron_condor__wing_k"] for _ in range(60)]
        assert min(dv) < 1.0                     # the default range reaches below the override's floor

    def test_the_inert_wing_k_check_reads_the_overridden_floor(self):
        """At $5k the default floor (wing_k 0.3 -> ~3.5 wing) is above the affordable ~2.8 wing, so the
        dimension is dropped; an override that lowers the floor to 0.2 makes it searchable again."""
        df = _ohlcv(300)
        df["implied_vol"] = 0.15
        df["Close"] = 400.0
        kw = dict(initial_capital=5_000.0, position_size_pct=0.05, budget_aware_wings=True, entry_dte=14)
        dropped = self._opt(**kw)
        dropped._data = {"SPY": df}
        dropped._maybe_drop_inert_params()
        assert "iron_condor__wing_k" in dropped._dropped_params
        kept = self._opt(search_space_overrides={"iron_condor": {"wing_k": [0.2, 2.0]}}, **kw)
        kept._data = {"SPY": df}
        kept._maybe_drop_inert_params()
        assert "iron_condor__wing_k" not in kept._dropped_params

    def test_the_effective_space_and_hash_follow_overrides_and_drops(self):
        from ait.optimization.param_spaces import space_hash
        a, b = self._opt(), self._opt(search_space_overrides={"iron_condor": {"wing_k": [1.0, 2.0]}})
        assert a.effective_search_space()["iron_condor"]["wing_k"] == ("float", 0.3, 2.0)
        assert b.effective_search_space()["iron_condor"]["wing_k"] == ("float", 1.0, 2.0)
        assert a.search_space_hash() != b.search_space_hash()
        assert b.search_space_hash() == space_hash(b.effective_search_space())
        b._dropped_params = {"iron_condor__wing_k"}
        assert "wing_k" not in b.effective_search_space()["iron_condor"]

    def test_overrides_are_validated_at_construction(self):
        with pytest.raises(ValueError, match="hard limits"):
            self._opt(search_space_overrides={"iron_condor": {"delta_short": [0.01, 0.2]}})
        with pytest.raises(ValueError, match="does not run"):
            self._opt(search_space_overrides={"put_credit_spread": {"wing_k": [1.0, 2.0]}})

    def test_a_stored_study_over_another_space_is_refused(self, tmp_path):
        import optuna
        url = f"sqlite:///{tmp_path / 'o.db'}"
        stale = optuna.create_study(study_name="s", storage=url, direction="maximize")
        stale.set_user_attr("search_space_hash", "deadbeefdeadbeef")
        opt = self._opt(study_name="s", storage=url)
        with pytest.raises(ValueError, match="new study name"):
            opt.run(data={"SPY": _ohlcv(300)})

    def test_a_study_without_a_stored_hash_is_adopted_and_stamped(self, tmp_path, monkeypatch):
        import optuna
        url = f"sqlite:///{tmp_path / 'o.db'}"
        optuna.create_study(study_name="s", storage=url, direction="maximize")      # pre-W87 study: no attr
        opt = self._opt(study_name="s", storage=url)
        monkeypatch.setattr(optuna.study.Study, "optimize", lambda self, *a, **k: None)
        try:
            opt.run(data={"SPY": _ohlcv(300)})
        except Exception:                                                           # no trials -> result build
            pass
        got = optuna.load_study(study_name="s", storage=url).user_attrs["search_space_hash"]
        assert got == opt.search_space_hash()


# --------------------------------------------------------------------------
# 4. the research CLI / manifest read their authorities, not copies
# --------------------------------------------------------------------------

def _manifest(monkeypatch, *, rsettings=None, argv=("run_backtest.py",)):
    import run_backtest as rb
    from ait.config.settings import load_settings
    monkeypatch.setattr(rb, "_research_settings", lambda: rsettings or load_settings())
    old = sys.argv
    sys.argv = list(argv)
    try:
        return rb.build_parity_manifest(rb.parse_args())      # NOT through main(): args unresolved
    finally:
        sys.argv = old


class TestManifestReadsItsAuthorities:
    RUNGS = {"dte>20": 0.50, "dte_11_20": 0.40, "dte_6_10": 0.30, "dte<=5": 0.20}

    def test_the_hand_copied_live_ladder_is_gone_and_both_columns_agree(self, monkeypatch):
        import run_backtest as rb
        assert not hasattr(rb, "_LIVE_TP_LADDER")
        m = _manifest(monkeypatch)
        assert m["live"]["tp_ladder"] == self.RUNGS == m["backtest"]["tp_ladder"]

    def test_the_live_column_follows_the_shared_ladder(self, monkeypatch):
        from ait.execution import exit_policy
        flatter = ((20, 1.00, 0.45), (10, 0.75, 0.40), (5, 0.50, 0.30),
                   (exit_policy.TERMINAL_DTE_FLOOR, 0.25, 0.20))
        monkeypatch.setattr(exit_policy, "CREDIT_TP_LADDER", flatter)
        m = _manifest(monkeypatch)
        assert m["live"]["tp_ladder"]["dte>20"] == 0.45          # read from the authority, not a copy
        assert m["live"]["tp_ladder"] == m["backtest"]["tp_ladder"]

    def test_the_delta_rows_come_from_the_builder_constant_and_the_config(self, monkeypatch):
        from ait.config.settings import load_settings
        from ait.strategies import iron_condor
        m = _manifest(monkeypatch)
        assert m["live"]["delta_target"] == iron_condor.SHORT_DELTA_TARGET == 0.20
        assert m["backtest"]["delta_target"] == load_settings().backtest.delta_short
        s = load_settings()
        other = s.model_copy(update={"backtest": s.backtest.model_copy(update={"delta_short": 0.25})})
        m2 = _manifest(monkeypatch, rsettings=other)
        assert m2["backtest"]["delta_target"] == 0.25
        assert any(w.startswith("delta_target") for w in m2["parity_warnings"])
        monkeypatch.setattr(iron_condor, "SHORT_DELTA_TARGET", 0.31)
        assert _manifest(monkeypatch)["live"]["delta_target"] == 0.31

    def test_the_builder_selects_with_the_named_delta(self, monkeypatch):
        """The constant is what the live builder passes to the strike search (executes generate_signals)."""
        from datetime import date, timedelta

        from ait.data.options_chain import OptionContract, OptionsChain
        from ait.strategies import iron_condor
        from ait.strategies.base import SignalDirection

        def c(strike, right, delta):
            return OptionContract(symbol="SPY", expiry=date.today() + timedelta(days=30), strike=strike,
                                  right=right, bid=1.0, ask=1.2, last=1.1, volume=100, open_interest=200,
                                  implied_vol=0.2, delta=delta)
        chain = OptionsChain(symbol="SPY", underlying_price=100.0, expiry=date.today() + timedelta(days=30),
                             calls=[c(100.0, "C", .2), c(105.0, "C", .2)],
                             puts=[c(95.0, "P", -.2), c(90.0, "P", -.2)])
        seen = []
        strat = iron_condor.IronCondor.__new__(iron_condor.IronCondor)
        monkeypatch.setattr(iron_condor.IronCondor, "_find_strike_by_delta",
                            lambda self, contracts, target: seen.append(target))
        monkeypatch.setattr(iron_condor, "SHORT_DELTA_TARGET", 0.27)
        strat.generate_signals("SPY", chain, SignalDirection.NEUTRAL, confidence=0.6, iv_rank=50.0)
        assert seen and set(seen) == {0.27}


class TestIvFloorFollowsTheContract:
    def test_the_cli_default_is_unset_and_the_manifest_reports_the_resolved_floor(self, monkeypatch):
        import run_backtest as rb
        old = sys.argv
        sys.argv = ["run_backtest.py"]
        try:
            assert rb.parse_args().iv_floor is None
        finally:
            sys.argv = old
        m = _manifest(monkeypatch)
        assert m["backtest"]["iv_rank_floor"] == 15.0 and m["live"]["iv_rank_floor"] == 15.0

    def test_an_environment_change_reaches_the_backtest_column(self, monkeypatch):
        monkeypatch.setenv("AIT_IRON_CONDOR_IV_FLOOR", "22")
        m = _manifest(monkeypatch)
        assert m["backtest"]["iv_rank_floor"] == 22.0            # never None (the W81 wing_floor defect)

    def test_an_explicit_flag_wins(self, monkeypatch):
        m = _manifest(monkeypatch, argv=("run_backtest.py", "--iv-floor", "30"))
        assert m["backtest"]["iv_rank_floor"] == 30.0
        assert any(w.startswith("iv_rank_floor") for w in m["parity_warnings"])

    def test_run_backtest_resolves_it_before_the_manifest_and_the_env_write(self, monkeypatch):
        import asyncio

        import run_backtest as rb
        import ait.config.research as research
        seen = {}

        class Stop(Exception):
            pass

        def spy(args):
            seen["iv_floor"] = args.iv_floor
            raise Stop

        from ait.config.settings import load_settings
        monkeypatch.setattr(research, "use_research_config", lambda *a, **k: load_settings())
        monkeypatch.setattr(rb, "build_parity_manifest", spy)
        monkeypatch.delenv("AIT_IRON_CONDOR_IV_FLOOR", raising=False)
        monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
        with pytest.raises(Stop):
            asyncio.run(rb.run_backtest(rb.parse_args()))
        assert seen["iv_floor"] == 15.0

    def test_the_cli_window_defaults_are_the_walk_forward_defaults(self, monkeypatch):
        import run_backtest as rb
        from ait.backtesting.walkforward import WalkForwardConfig
        monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
        a = rb.parse_args()
        d = WalkForwardConfig()
        assert (a.train_days, a.test_days, a.step_days, a.gap_days) == (
            d.train_days, d.test_days, d.step_days, d.gap_days)


class TestCommissionFollowsTheConfig:
    def test_the_walk_forward_default_is_the_sentinel(self):
        from ait.backtesting.walkforward import WalkForwardConfig
        assert WalkForwardConfig().commission_per_contract is None
        assert WalkForwardConfig(commission_per_contract=0.5).commission_per_contract == 0.5

    def test_a_config_change_reaches_the_engine_a_window_builds(self):
        from ait.backtesting.engine import Backtester
        from ait.backtesting.walkforward import WalkForwardConfig
        from ait.config.settings import load_settings
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(
            update={"commission_per_contract": 0.90})})
        wf = WalkForwardConfig()
        bt = Backtester(data=_ohlcv(60), strategies=["iron_condor"], settings=moved,
                        commission_per_contract=wf.commission_per_contract)
        assert bt._commission == 0.90
        assert Backtester(data=_ohlcv(60), strategies=["iron_condor"], settings=s,
                          commission_per_contract=wf.commission_per_contract)._commission == 0.65


class TestLiveBaselinesLieInsideTheSearchRanges:
    def test_each_live_value_is_reachable_by_the_default_search(self):
        from ait.optimization.param_spaces import STRATEGY_SPACES, live_baselines
        base = live_baselines("iron_condor")
        assert set(base) == {"wing_k", "delta_short"}
        for name, value in base.items():
            kind, lo, hi = STRATEGY_SPACES["iron_condor"][name]
            assert lo <= value <= hi, f"live {name}={value} is outside the default range [{lo}, {hi}]"

    def test_the_values_come_from_their_owners(self, monkeypatch):
        from ait.optimization.param_spaces import live_baselines
        from ait.strategies import iron_condor
        monkeypatch.setenv("AIT_IC_WING_K", "1.2")
        monkeypatch.setattr(iron_condor, "SHORT_DELTA_TARGET", 0.18)
        assert live_baselines() == {"wing_k": 1.2, "delta_short": 0.18}
        assert live_baselines("put_credit_spread") == {}


class TestExcludedLiveValueIsWarnedAboutByTheOptimizer:
    def _opt(self, monkeypatch, over):
        import ait.optimization.optimizer as om
        seen = []
        monkeypatch.setattr(om.log, "warning", lambda ev, **kw: seen.append((ev, kw)))
        om.StrategyOptimizer(symbols=["QQQ"], strategies=["iron_condor"], n_trials=1,
                             search_space_overrides=over)
        return seen

    def test_a_range_below_the_live_wing_k_warns(self, monkeypatch):
        seen = self._opt(monkeypatch, {"iron_condor": {"wing_k": [0.5, 1.2]}})
        assert [e for e, _ in seen] == ["search_space_excludes_live_value"]
        assert "wing_k" in seen[0][1]["detail"] and "1.6" in seen[0][1]["detail"]

    def test_a_range_that_keeps_the_live_values_is_silent(self, monkeypatch):
        assert self._opt(monkeypatch, {"iron_condor": {"wing_k": [1.0, 2.0], "delta_short": [0.16, 0.24]}}) == []

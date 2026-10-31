"""W79a: the DTE-search plumbing (research only).

Real code paths on synthetic fixtures. Layers:

  * ``DteSearchConfig`` / the research overlay (D1 and D8 are pinned here)
  * ``param_spaces.dte_search_space`` and the optimizer's merge, baselines and provider
  * ``WalkForwardConfig`` / ``WalkForwardBacktester``: the refusal that used to swallow a
    searched DTE, forwarding to the OOS and shadow engines, counted failures, routing of the
    per-window range model by the window's OWN guard and fraction
  * ``dte_report``: what a DTE study found
"""
from __future__ import annotations

import dataclasses
import pickle
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import optuna
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

import ait.backtesting.engine as engine_mod
import ait.backtesting.walkforward as wf_mod
from ait.backtesting import dte_report as rep
from ait.backtesting.walkforward import (
    WalkForwardBacktester, WalkForwardConfig, WalkForwardResult)
from ait.config.settings import BacktestConfig, DteSearchConfig
from ait.optimization import param_spaces as ps
from ait.optimization.optimizer import StrategyOptimizer

from tests.test_r20_research_validity import (  # noqa: F401 - shared fixtures
    _recorder_factory, _run_window, synth_df, synth_features)

ROOT = Path(__file__).resolve().parents[1]

# the IC space as of W78: W79a must not change what every other study searches
IC_SPACE_KEYS = {
    "delta_short", "max_hold_days", "wing_k", "iv_rank_rise_threshold",
    "min_edge_over_baseline", "hurst_regime_threshold", "hurst_regime_penalty",
    "multifractal_max_width",
}


def _on(**kw) -> DteSearchConfig:
    return DteSearchConfig(enabled=True, **kw)


# --------------------------------------------------------------------------
# config + overlay
# --------------------------------------------------------------------------

class TestDteSearchConfig:
    def test_defaults_are_off_and_cover_the_operators_range(self):
        c = DteSearchConfig()
        assert c.enabled is False and c.entry_dte == [0, 45] and c.guard_dte == [0, 5]
        assert c.hold_fraction == [0.05, 1.0]
        assert c.search_flatten is False and c.multivariate is False and c.strict is False
        assert BacktestConfig().dte_search.enabled is False

    @pytest.mark.parametrize("field,bad", [
        ("entry_dte", [10, 5]), ("entry_dte", [0, 46]), ("entry_dte", [-1, 10]), ("entry_dte", [5]),
        ("guard_dte", [3, 1]), ("hold_fraction", [0.0, 1.0]), ("hold_fraction", [0.5, 1.1]),
        ("hold_fraction", [0.9, 0.5]), ("flatten_minutes", [2, 60]), ("flatten_minutes", [90, 30]),
    ])
    def test_bad_ranges_are_refused(self, field, bad):
        with pytest.raises(ValidationError):
            DteSearchConfig(**{field: bad})

    def test_a_typo_is_a_startup_error(self):
        with pytest.raises(ValidationError):
            DteSearchConfig(entry_dtes=[0, 10])


class TestOverlayDecisions:
    """The operator's 2026-10-12 decisions D1 and D8 live in the research overlay ONLY."""

    def _y(self, name):
        return yaml.safe_load((ROOT / name).read_text()) or {}

    def test_d1_and_d8_are_in_the_research_overlay(self):
        bt = self._y("config_research.yaml")["backtest"]
        assert bt["engine_model"] == "realistic"
        assert bt["friction_short_dte"] is True

    def test_d2_is_still_open(self):
        assert "pricing_source" not in self._y("config_research.yaml")["backtest"]

    def test_the_live_config_carries_neither(self):
        bt = (self._y("config.yaml").get("backtest") or {})
        assert "engine_model" not in bt and "friction_short_dte" not in bt

    def test_loaded_research_view(self):
        from ait.config.research import use_research_config
        from ait.config.settings import load_settings
        use_research_config()
        b = load_settings().backtest
        assert (b.engine_model, b.friction_short_dte, b.pricing_source) == (
            "realistic", True, "bs_legacy")
        assert b.dte_search.enabled is False            # a per-study switch, never the overlay


# --------------------------------------------------------------------------
# search space
# --------------------------------------------------------------------------

class TestSpace:
    def test_the_default_iron_condor_space_is_unchanged(self):
        assert set(ps.IRON_CONDOR_SPACE) == IC_SPACE_KEYS
        assert ps.IRON_CONDOR_SPACE["max_hold_days"][:3] == ("int", 1, 9)

    def test_disabled_or_missing_adds_nothing(self):
        assert ps.dte_search_space(None) == {}
        assert ps.dte_search_space(DteSearchConfig()) == {}

    def test_enabled_adds_exactly_the_configured_dimensions(self):
        sp = ps.dte_search_space(_on(entry_dte=[0, 30], guard_dte=[0, 3], hold_fraction=[0.2, 0.9]))
        assert sp == {"entry_dte": ("int", 0, 30), "expiry_guard_dte": ("int", 0, 3),
                      "max_hold_fraction": ("float", 0.2, 0.9)}
        assert "flatten_minutes_before_close" not in sp

    def test_flatten_only_when_asked(self):
        sp = ps.dte_search_space(_on(search_flatten=True, flatten_minutes=[30, 90]))
        assert sp["flatten_minutes_before_close"] == ("int", 30, 90)

    def test_the_credit_floors_are_never_searched(self):
        sp = ps.dte_search_space(_on(search_flatten=True))
        assert not {"ic_min_credit", "ic_min_credit_width"} & set(sp)


# --------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------

def _opt(**kw):
    return StrategyOptimizer(symbols=["SPY"], strategies=kw.pop("strategies", ["iron_condor"]),
                             n_trials=1, **kw)


def _suggest(opt):
    study = optuna.create_study(direction="maximize")
    return opt._suggest_params(study.ask())


class TestOptimizerMerge:
    def test_off_keeps_todays_space(self):
        keys = {k.split("__", 1)[1] for k in _suggest(_opt())}
        assert "max_hold_days" in keys
        assert not {"entry_dte", "expiry_guard_dte", "max_hold_fraction"} & keys

    def test_on_adds_the_dte_block_and_replaces_the_day_cap(self):
        got = _suggest(_opt(dte_search=_on(entry_dte=[0, 4], guard_dte=[1, 4], hold_fraction=[0.3, 0.8])))
        keys = {k.split("__", 1)[1] for k in got}
        assert {"entry_dte", "expiry_guard_dte", "max_hold_fraction"} <= keys
        assert "max_hold_days" not in keys
        assert 0 <= got["iron_condor__entry_dte"] <= 4
        assert 1 <= got["iron_condor__expiry_guard_dte"] <= 4
        assert 0.3 <= got["iron_condor__max_hold_fraction"] <= 0.8

    def test_guard_is_sampled_only_where_it_can_bind(self):
        # flatness check: expiry_guard_dte is FLAT for entries above the searched guard range
        o = _opt(dte_search=_on(entry_dte=[0, 45], guard_dte=[0, 5]))
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=1))
        seen_with, seen_without = 0, 0
        for _ in range(60):
            got = o._suggest_params(study.ask())
            dte, has_guard = got["iron_condor__entry_dte"], "iron_condor__expiry_guard_dte" in got
            assert has_guard == (dte <= 5), (dte, has_guard)
            seen_with += has_guard
            seen_without += not has_guard
        assert seen_with and seen_without                       # both branches were exercised

    def test_guard_always_present_when_every_entry_can_use_it(self):
        o = _opt(dte_search=_on(entry_dte=[0, 3], guard_dte=[0, 5]))
        study = optuna.create_study(direction="maximize")
        assert all("iron_condor__expiry_guard_dte" in o._suggest_params(study.ask()) for _ in range(10))

    def test_other_strategies_are_untouched(self):
        got = _suggest(_opt(strategies=["short_strangle"], dte_search=_on()))
        keys = {k.split("__", 1)[1] for k in got}
        assert "max_hold_days" in keys and "entry_dte" not in keys

    def test_multivariate_switches_the_sampler(self, monkeypatch):
        seen = {}
        real = optuna.samplers.TPESampler

        def spy(**kw):
            seen.update(kw)
            return real(**kw)
        monkeypatch.setattr(optuna.samplers, "TPESampler", spy)
        o = _opt(multivariate=True)
        o._data = {"SPY": pd.DataFrame({"Close": [1.0] * 5})}
        o._objective_fn = lambda trial: 0.0
        o._maybe_drop_inert_params = lambda: None
        o._n_trials = 1
        try:
            o.run(data=o._data)
        except Exception:  # noqa: BLE001 - only the sampler construction matters here
            pass
        assert seen.get("multivariate") is True and seen.get("group") is True


class TestTrialKwargs:
    @staticmethod
    def _run(monkeypatch, synth_df, params, **opt_kw):
        sink: list = []
        monkeypatch.setattr(engine_mod, "Backtester", _recorder_factory(sink))
        opt = _opt(**opt_kw)
        opt._data = {"SPY": synth_df}
        opt._run_backtest(params)
        assert len(sink) == 1
        return sink[0]

    def test_searched_values_reach_the_trial_engine(self, monkeypatch, synth_df):
        kw = self._run(monkeypatch, synth_df, {
            "iron_condor__entry_dte": 3, "iron_condor__expiry_guard_dte": 2,
            "iron_condor__max_hold_fraction": 0.4, "iron_condor__delta_short": 0.17})
        assert (kw["entry_dte"], kw["expiry_guard_dte"], kw["max_hold_fraction"]) == (3, 2, 0.4)
        assert kw["delta_short"] == pytest.approx(0.17)

    def test_unsearched_baselines_still_reach_every_trial(self, monkeypatch, synth_df):
        kw = self._run(monkeypatch, synth_df, {}, entry_dte=18, expiry_guard_dte=5,
                       max_hold_fraction=0.5, flatten_minutes_before_close=60, delta_short=0.25)
        assert (kw["entry_dte"], kw["expiry_guard_dte"], kw["max_hold_fraction"],
                kw["flatten_minutes_before_close"]) == (18, 5, 0.5, 60)
        assert kw["delta_short"] == pytest.approx(0.25)

    def test_nothing_set_means_the_engine_resolves_its_own(self, monkeypatch, synth_df):
        kw = self._run(monkeypatch, synth_df, {})
        assert kw["entry_dte"] is None and kw["expiry_guard_dte"] is None
        assert kw["max_hold_fraction"] is None and kw["flatten_minutes_before_close"] is None

    def test_the_provider_gets_the_trials_dte_guard_and_fraction(self, monkeypatch, synth_df):
        asked = []
        self._run(monkeypatch, synth_df, {
            "iron_condor__entry_dte": 3, "iron_condor__expiry_guard_dte": 5,
            "iron_condor__max_hold_fraction": 0.5},
            range_predictor_provider=lambda e, g, f: (asked.append((e, g, f)), "m")[1])
        assert asked == [(3, 5, 0.5)]


class TestInertWingUsesTheSmallestDte:
    @staticmethod
    def _opt_small(monkeypatch, **kw):
        from ait.strategies.iron_condor import IronCondor
        monkeypatch.setattr(IronCondor, "affordable_width", staticmethod(lambda budget: 0.5))
        # W108 S2: a 0.5 ceiling under the default $2 wing floor binds at every k and every DTE (the engine's
        # free wing is max(k x EM, floor)), so the rule now drops wing_k there whatever the DTE; a floor under
        # the ceiling keeps this test about the DTE alone.
        kw.setdefault("wing_floor_dollars", 0.1)
        o = _opt(**kw)
        idx = pd.bdate_range("2024-01-02", periods=40)
        o._data = {"SPY": pd.DataFrame({"Close": 100.0}, index=idx)}
        o._budget_aware_wings = True
        return o

    def test_dropped_at_the_configured_band_but_kept_when_a_short_dte_is_searchable(self, monkeypatch):
        # wing target at 14 DTE ~0.88 > the 0.5 affordable wing (cap binds: dimension inert);
        # at 1 DTE ~0.24 < 0.5 (cap does not bind: dimension matters).
        off = self._opt_small(monkeypatch)
        off._maybe_drop_inert_params()
        assert "iron_condor__wing_k" in off._dropped_params
        on = self._opt_small(monkeypatch, dte_search=_on(entry_dte=[0, 45]))
        on._maybe_drop_inert_params()
        assert "iron_condor__wing_k" not in on._dropped_params


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------

def _wfc(**kw):
    base = dict(train_window_models=False)
    base.update(kw)
    return WalkForwardConfig(**base)


class TestWalkForwardConfig:
    def test_defaults(self):
        c = _wfc()
        assert c.expiry_guard_dte is None and c.max_hold_fraction is None
        assert c.flatten_minutes_before_close is None and c.delta_short is None
        assert c.dte_search is False and c.optimize_strict is False and c.optimize_multivariate is False

    def test_search_requires_per_window_optimisation(self):
        with pytest.raises(ValueError, match="optimize_per_window"):
            _wfc(dte_search=True)
        assert _wfc(dte_search=True, optimize_per_window=True).dte_search is True

    def test_validation(self):
        with pytest.raises(ValueError, match="max_hold_fraction"):
            _wfc(max_hold_fraction=1.5)
        with pytest.raises(ValueError, match="dte_range"):
            _wfc(dte_range=(10, 3))
        assert _wfc(dte_range=(1, 7)).dte_range == (1, 7)

    def test_settings_supply_the_switches_when_unset(self, monkeypatch):
        import ait.config.settings as st
        real = st.load_settings

        def patched(*a, **k):
            s = real(*a, **k)
            s.backtest.dte_search = DteSearchConfig(enabled=True, strict=True, multivariate=True)
            return s
        monkeypatch.setattr(st, "load_settings", patched)
        c = WalkForwardConfig(optimize_per_window=True, train_window_models=False)
        assert (c.dte_search, c.optimize_strict, c.optimize_multivariate) == (True, True, True)
        assert WalkForwardConfig(optimize_per_window=True, dte_search=False,
                                 train_window_models=False).dte_search is False

    def test_pickles_for_window_workers(self):
        c = _wfc(optimize_per_window=True, dte_search=True, dte_range=(0, 10), expiry_guard_dte=3,
                 max_hold_fraction=0.5, flatten_minutes_before_close=60, delta_short=0.2)
        assert pickle.loads(pickle.dumps(c)) == c


class TestSearchBlock:
    def test_none_when_not_searching(self):
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc())
        assert wf._dte_search_block() is None

    def test_range_override_wins(self):
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(
            optimize_per_window=True, dte_search=True, dte_range=(2, 9)))
        blk = wf._dte_search_block()
        assert blk.enabled and blk.entry_dte == [2, 9]
        assert blk.guard_dte == [0, 5]            # everything else from the settings


class TestForwarding:
    KNOBS = dict(expiry_guard_dte=3, max_hold_fraction=0.5, flatten_minutes_before_close=45,
                 delta_short=0.22)

    def test_oos_engine_receives_the_knobs(self, synth_df, monkeypatch):
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(entry_dte=7, **self.KNOBS))
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        got = sink[-1]
        assert got["entry_dte"] == 7
        for k, v in self.KNOBS.items():
            assert got[k] == v, k

    def test_meta_label_shadow_engine_receives_them(self, synth_df, monkeypatch):
        cfg = _wfc(entry_dte=7, **self.KNOBS)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        wf._train_window_meta_labeler(train_df=synth_df, symbol="SPY", window_id=1,
                                      predictor=None, window_cfg=cfg)
        for k, v in self.KNOBS.items():
            assert sink[-1][k] == v, k


class TestRangeHorizonFollowsTheStudy:
    @staticmethod
    def _wf(**cfg):
        wf = WalkForwardBacktester.__new__(WalkForwardBacktester)
        from ait.config.settings import load_settings
        wf._settings = load_settings()
        wf._config = SimpleNamespace(max_hold_days=30, **cfg)
        return wf

    def test_the_base_guard_applies_unless_the_trial_carries_its_own(self):
        wf = self._wf(expiry_guard_dte=5)
        assert wf._range_label_horizon(3) == 3                  # exempt: lives to expiry
        assert wf._range_label_horizon(3, guard_dte=0) == 1     # the trial's guard wins: closes at once
        assert self._wf()._range_label_horizon(3) == 1          # no guard anywhere: rules apply

    def test_the_trials_fraction_wins_over_the_base(self):
        wf = self._wf(expiry_guard_dte=10, max_hold_fraction=1.0)
        assert wf._range_label_horizon(10) == 10
        assert wf._range_label_horizon(10, hold_fraction=0.5) == 5


class TestTrialsAreRetainedForTheReport:
    @staticmethod
    def _fake_optimizer(monkeypatch):
        import ait.optimization.optimizer as om

        def trial(n, dte, value, trades, state="COMPLETE"):
            return SimpleNamespace(
                number=n, state=SimpleNamespace(name=state), value=value,
                params={"iron_condor__entry_dte": dte, "iron_condor__wing_k": 0.5,
                        "iron_condor__expiry_guard_dte": 2},
                user_attrs={"n_trades": trades}, datetime_start=None, datetime_complete=None)

        class Fake:
            def __init__(self, **kw):
                pass

            def run(self, **kw):
                return SimpleNamespace(
                    study=SimpleNamespace(trials=[trial(0, 3, 1.5, 12), trial(1, 9, 0.2, 8)]),
                    usable=True, best_value=1.5, best_n_trades=12, early_stopped=False,
                    stop_reason="", best_params={"iron_condor__entry_dte": 3})
        monkeypatch.setattr(om, "StrategyOptimizer", Fake)

    def test_a_dte_study_keeps_the_compact_trials_with_the_window_status(self, synth_df, monkeypatch):
        self._fake_optimizer(monkeypatch)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(
            optimize_per_window=True, dte_search=True))
        cfg, best, meta = wf._optimize_window_params(synth_df, "SPY", 4, ["iron_condor"])
        assert cfg.entry_dte == 3 and best == {"iron_condor__entry_dte": 3}
        trials = wf._opt_status_by_window[4]["SPY"]["trials"]
        assert [t["params"]["entry_dte"] for t in trials] == [3, 9]
        assert set(trials[0]["params"]) == {"entry_dte", "expiry_guard_dte"}   # compact: no wing_k
        assert trials[0]["value"] == 1.5 and trials[0]["n_trades"] == 12
        # and the report consumes exactly that shape
        assert rep.summarise_window(trials)["best_dte"] == 3

    def test_a_normal_study_does_not_carry_trials(self, synth_df, monkeypatch):
        self._fake_optimizer(monkeypatch)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(optimize_per_window=True))
        wf._optimize_window_params(synth_df, "SPY", 4, ["iron_condor"])
        assert "trials" not in wf._opt_status_by_window[4]["SPY"]


class TestFailureIsLoud:
    @staticmethod
    def _failing_optimizer(monkeypatch):
        import ait.optimization.optimizer as om

        class Boom:
            def __init__(self, **kw):
                pass

            def run(self, **kw):
                raise RuntimeError("study exploded")
        monkeypatch.setattr(om, "StrategyOptimizer", Boom)

    def test_a_failed_window_is_counted_and_falls_back(self, synth_df, monkeypatch):
        self._failing_optimizer(monkeypatch)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(optimize_per_window=True))
        out = wf._optimize_window_params(synth_df, "SPY", 3, ["iron_condor"])
        assert out == (None, None, None)
        st = wf._opt_status_by_window[3]["SPY"]
        assert st["failed"] is True and "study exploded" in st["error"]

    def test_strict_mode_raises(self, synth_df, monkeypatch):
        self._failing_optimizer(monkeypatch)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(
            optimize_per_window=True, optimize_strict=True))
        with pytest.raises(RuntimeError, match="study exploded"):
            wf._optimize_window_params(synth_df, "SPY", 1, ["iron_condor"])

    def test_result_counts_failures(self):
        r = WalkForwardResult(optimization_status={
            1: {"SPY": {"usable": True}, "QQQ": {"usable": False, "failed": True}},
            2: {"SPY": {"usable": False, "failed": True}}})
        assert r.optimization_failures == 2
        assert WalkForwardResult().optimization_failures == 0


class TestSearchedDteReachesTheWindow:
    """A searched DTE must become the window config, reach the OOS engine, and pick the range
    model for THAT trade's reachable horizon (the guard is searched too)."""

    def test_end_to_end(self, synth_df, monkeypatch):
        cfg = _wfc(optimize_per_window=True, dte_search=True, train_window_models=True)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        chosen = dataclasses.replace(
            cfg, entry_dte=3, expiry_guard_dte=5, max_hold_fraction=0.5, delta_short=0.22)
        monkeypatch.setattr(wf, "_optimize_window_params",
                            lambda *a, **k: (chosen, {"iron_condor__entry_dte": 3}, None))
        trained = []
        stub = SimpleNamespace(is_trained=False, fitted_weights=None, _symbol_models={})
        monkeypatch.setattr(
            wf, "_train_window_range_model",
            lambda df, sym, wid, max_hold_days=30, model_subdir=None:
                (trained.append((max_hold_days, model_subdir)), (stub, "ok", 0.05))[1])
        monkeypatch.setattr(wf, "_train_window_model", lambda *a, **k: None)
        monkeypatch.setattr(wf, "_train_window_meta_labeler", lambda *a, **k: None)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        oos = sink[-1]
        assert (oos["entry_dte"], oos["expiry_guard_dte"], oos["max_hold_fraction"]) == (3, 5, 0.5)
        assert oos["delta_short"] == pytest.approx(0.22)
        # base model (the study's configured horizon) + the window's own bucket: 3 DTE, guard 5
        # is exempt, fraction 0.5 -> ceil(0.5 * 3) = 2 bars -> bucket 2, own artifact directory
        assert trained[0][1] is None
        assert (2, "h2") in trained


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _trial(dte, value, n=20, state="COMPLETE", prefix="iron_condor__"):
    return {"params": {f"{prefix}entry_dte": dte}, "value": value, "n_trades": n, "state": state}


class TestReport:
    def test_bands(self):
        assert [rep.band_label(d) for d in (0, 2, 3, 7, 8, 14, 15, 45, 46)] == [
            "0-2", "0-2", "3-7", "3-7", "8-14", "8-14", "15-45", "15-45", "46+"]

    def test_window_summary(self):
        trials = ([_trial(1, 2.0), _trial(2, 1.5), _trial(10, 0.4), _trial(20, -0.2)]
                  + [_trial(30, -1.0)] * 6)
        s = rep.summarise_window(trials, top_frac=0.1)
        assert s["best_dte"] == 1 and s["best_value"] == 2.0 and s["n_trials"] == 10
        assert s["top_n"] == 1 and s["top_bands"] == ["0-2"]
        assert s["by_band"]["15-45"]["n"] == 7 and s["by_band"]["8-14"]["best"] == 0.4

    def test_unscored_trials_are_ignored(self):
        trials = [_trial(3, 1.0), _trial(4, None), _trial(5, 2.0, state="PRUNED"),
                  _trial(6, float("nan")), {"params": {}, "value": 1.0, "state": "COMPLETE"}]
        s = rep.summarise_window(trials)
        assert s["n_trials"] == 1 and s["best_dte"] == 3
        assert rep.summarise_window([]) is None and rep.summarise_window([_trial(1, None)]) is None

    def test_candidate_bands_need_two_thirds_of_windows(self):
        mk = lambda dtes: [_trial(d, 1.0 - 0.01 * i) for i, d in enumerate(dtes)]
        study = rep.summarise_study({
            1: mk([1, 2, 12, 30, 31, 32, 33, 34, 35, 36]),     # top decile: 0-2
            2: mk([2, 12, 30, 31, 32, 33, 34, 35, 36, 37]),    # 0-2
            3: mk([12, 2, 30, 31, 32, 33, 34, 35, 36, 37])})   # 8-14
        assert study["n_windows"] == 3
        assert study["candidate_bands"] == ["0-2"]             # 3/3 vs 2/3 boundary: 0-2 has 2/3
        assert study["top_decile_band_share"]["0-2"] == pytest.approx(2 / 3)
        assert study["best_dte_by_window"] == {1: 1, 2: 2, 3: 12}

    def test_prefixless_params_and_format(self):
        study = rep.summarise_study({1: [_trial(5, 1.0, prefix="")]})
        lines = rep.format_report(study)
        assert any("candidate bands" in l for l in lines) and any("w1=5" in l for l in lines)
        assert any("NONE" in l or "3-7" in l for l in lines)

    def test_empty_study_says_so(self):
        assert "no window" in " ".join(rep.format_report(rep.summarise_study({})))

    def test_module_is_import_light(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(rep))
        top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        assert top <= {"__future__", "math", "statistics", "collections", "typing"}

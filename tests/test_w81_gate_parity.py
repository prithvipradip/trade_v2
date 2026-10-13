"""W81: one-sided entry gates are visible, the regime veto has a switch, research r is config.

Closes the three findings of the 2026-10-13 session-end audit that were mechanical:

  P2  the research-only trending-down regime veto had hard-coded thresholds and no switch
  P4  the engine priced with a bare ``r = 0.05`` at three sites (live reads options.risk_free_rate)
  P5  the manifest's wing_floor row could print None and its live column carried literal fallbacks

and adds the guard that makes the NEXT one-sided gate impossible to add silently: every reason the
engine can reject on is registered in ``ait.backtesting.gate_parity`` with its live twin (or the
declared deviation), and every active research-only gate is announced by the manifest.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import optuna  # noqa: F401  (the optimizer module imports it)
import pandas as pd
import pytest

import ait.backtesting.engine as engine_mod
import ait.backtesting.gate_parity as gp
import ait.backtesting.walkforward as wf_mod
from ait.backtesting.engine import Backtester, research_regime
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.config.runtime_env import CONTRACT_DEFAULTS
from ait.config.settings import BacktestConfig, OptionsConfig
from ait.optimization.optimizer import StrategyOptimizer

from tests.test_r20_research_validity import (  # noqa: F401 - shared fixtures
    _engine_kwargs, _recorder_factory, _run_window, synth_df, synth_features)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ait"


# --------------------------------------------------------------------------
# the registry: no engine reason without a row, no row without a reason
# --------------------------------------------------------------------------

class TestRegistry:
    @staticmethod
    def _reasons():
        return gp.engine_reject_reasons((SRC / "backtesting" / "engine.py").read_text())

    def test_every_engine_reason_is_registered(self):
        reasons, _dyn = self._reasons()
        assert reasons, "the AST scan found no _reject calls (the guard would pass vacuously)"
        unknown = sorted(r for r in reasons if gp.classify(r) is None)
        assert not unknown, (
            f"engine rejection reasons without a gate_parity row: {unknown}. Register each "
            "with its live twin, or as RESEARCH_ONLY with its switch, and make the manifest "
            "announce it (one_sided_gate_warnings).")

    def test_no_reason_is_passed_dynamically(self):
        _reasons, dynamic = self._reasons()
        assert dynamic == [], f"_reject called with a non-literal reason at lines {dynamic}"

    def test_no_registry_row_has_rotted(self):
        reasons, _dyn = self._reasons()
        stale = sorted(r for r in gp.ENGINE_REJECT_REASONS if r not in reasons)
        assert not stale, f"registered reasons the engine no longer emits: {stale}"

    def test_builder_details_classify_as_builder_none(self):
        row = gp.classify("builder:credit_floor")
        assert row is gp.ENGINE_REJECT_REASONS["builder_none"] and row.kind == gp.MECHANICS

    def test_a_new_reason_would_be_caught(self):
        reasons, dyn = gp.engine_reject_reasons(
            "class B:\n    def f(self):\n        self._reject('brand_new_gate')\n"
            "        self._would_reject(reason_var)\n")
        assert reasons == {"brand_new_gate"} and dyn == [4]
        assert gp.classify("brand_new_gate") is None

    def test_research_only_rows_name_a_real_switch(self):
        fields = set(BacktestConfig.model_fields)
        for row in gp.ENGINE_REJECT_REASONS.values():
            if row.kind == gp.RESEARCH_ONLY and row.switch:
                assert row.switch in fields, (row.reason, row.switch)
        for row in gp.LIVE_ONLY_GATES:
            if row.switch:
                section, _, key = row.switch.partition(".")
                assert section == "risk" and key, row

    def test_module_is_pure_stdlib(self):
        tree = ast.parse((SRC / "backtesting" / "gate_parity.py").read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0:
                mods.add((n.module or "").split(".")[0])
        assert mods <= {"__future__", "ast", "dataclasses", "typing"}, mods


# --------------------------------------------------------------------------
# the manifest announces every active one-sided gate
# --------------------------------------------------------------------------

def _cfg(**kw):
    base = dict(max_entry_vol_annual=0.80, regime_veto_gate=True, regime_down_threshold=-0.05,
                pct_from_60d_high_threshold=-1.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _live(halt=28.0):
    return SimpleNamespace(risk=SimpleNamespace(credit_vix_halt=halt))


class TestOneSidedWarnings:
    def _starts(self, ws):
        return [w.split(":")[0] for w in ws]

    def test_today_is_announced(self):
        ws = gp.one_sided_gate_warnings(_cfg(), _live())
        assert self._starts(ws) == ["vol_gate", "regime_veto", "vix_halt"]
        assert "0.8" in ws[0] and "28" in ws[2] and "5%" in ws[1]

    def test_regime_veto_off_is_silent_but_the_others_remain(self):
        ws = gp.one_sided_gate_warnings(_cfg(regime_veto_gate=False), _live())
        assert self._starts(ws) == ["vol_gate", "vix_halt"]

    def test_drawdown_gate_only_when_enabled(self):
        assert "pct_from_60d_high_threshold" not in self._starts(
            gp.one_sided_gate_warnings(_cfg(), _live()))
        ws = gp.one_sided_gate_warnings(_cfg(pct_from_60d_high_threshold=-0.07), _live())
        assert "pct_from_60d_high_threshold" in self._starts(ws)

    def test_unavailable_inputs_give_no_guess(self):
        assert gp.one_sided_gate_warnings(None, None) == []
        assert self._starts(gp.one_sided_gate_warnings(_cfg(), None)) == ["vol_gate", "regime_veto"]

    def test_a_disabled_vol_cap_is_silent(self):
        ws = gp.one_sided_gate_warnings(_cfg(max_entry_vol_annual=10.0), _live())
        assert "vol_gate" not in self._starts(ws)

    def test_every_research_only_gate_with_a_switch_can_be_announced(self):
        """The registry and the warnings agree: each RESEARCH_ONLY reason that has a backtest
        switch is either announced here or by its own named manifest warning."""
        announced_here = {"vol_gate", "regime_veto", "pct_from_high"}
        announced_elsewhere = {"iv_rank_rise"}           # run_backtest.py W71 / W71c warning
        exempt = {"post_event"}                          # inert unless a study sets it; no switch
        registered = {r for r, row in gp.ENGINE_REJECT_REASONS.items()
                      if row.kind == gp.RESEARCH_ONLY}
        assert registered == announced_here | announced_elsewhere | exempt, (
            "a new RESEARCH_ONLY gate needs a manifest warning (one_sided_gate_warnings) and "
            "an entry here")


class TestManifest:
    @staticmethod
    def _manifest(monkeypatch, *, gate=True, env=None):
        import run_backtest as rb
        from ait.config.settings import load_settings
        s = load_settings()
        s = s.model_copy(update={"backtest": s.backtest.model_copy(
            update={"regime_veto_gate": gate})})
        monkeypatch.setattr(rb, "_research_settings", lambda: s)
        if env is not None:
            monkeypatch.setattr(rb, "_live_env", lambda: dict(env))
        old = sys.argv
        sys.argv = ["run_backtest.py"]
        try:
            return rb.build_parity_manifest(rb.parse_args())       # NOT through main(): args unresolved
        finally:
            sys.argv = old

    def test_one_sided_gates_reach_the_manifest(self, monkeypatch):
        m = self._manifest(monkeypatch)
        starts = [w.split(":")[0] for w in m["parity_warnings"]]
        assert {"vol_gate", "regime_veto", "vix_halt"} <= set(starts)
        assert m["backtest"]["regime_veto_gate"] is True
        assert m["backtest"]["live_only_gates_not_simulated"] == [
            g.reason for g in gp.LIVE_ONLY_GATES]

    def test_switching_the_veto_off_removes_only_its_warning(self, monkeypatch):
        m = self._manifest(monkeypatch, gate=False)
        starts = [w.split(":")[0] for w in m["parity_warnings"]]
        assert "regime_veto" not in starts and "vol_gate" in starts
        assert m["backtest"]["regime_veto_gate"] is False

    def test_wing_floor_is_resolved_not_none(self, monkeypatch):
        m = self._manifest(monkeypatch)
        assert m["backtest"]["wing_floor"] is not None
        assert m["backtest"]["wing_floor"] == m["live"]["wing_floor"]
        assert not [w for w in m["parity_warnings"] if w.startswith("wing_floor")]

    def test_live_column_falls_back_to_the_contract_not_a_literal(self, monkeypatch):
        """An env snapshot WITHOUT the keys: the live column must show CONTRACT_DEFAULTS (it used
        to fall back to literals: wing_k 1.0 and a 0.20 credit ratio, vs the contract's 1.6 / 0.10)."""
        m = self._manifest(monkeypatch, env={})
        for key, field in (("AIT_IC_WING_K", "wing_k"), ("AIT_IC_WING_FLOOR", "wing_floor"),
                           ("AIT_IC_MIN_CREDIT", "ic_min_credit"),
                           ("AIT_IC_MIN_CREDIT_WIDTH", "ic_min_credit_width"),
                           ("AIT_CREDIT_LOSS_LIMIT", "credit_loss_limit"),
                           ("AIT_IRON_CONDOR_IV_FLOOR", "iv_rank_floor")):
            assert m["live"][field] == pytest.approx(float(CONTRACT_DEFAULTS[key])), field


# --------------------------------------------------------------------------
# regime: one definition, one switch, thresholds in config
# --------------------------------------------------------------------------

def _old_chain(vol_exp: bool, px: float) -> str:
    """The pre-W81 literal chain (the veto and entry_regime sites)."""
    return ("trending_down" if (vol_exp and px < -0.05) else
            "trending_up" if (vol_exp and px > 0.02) else
            "high_volatility" if vol_exp else "range_bound")


class TestRegimeClass:
    def test_schema_defaults_equal_the_former_literals(self):
        b = BacktestConfig()
        assert b.regime_veto_gate is True
        assert b.regime_down_threshold == -0.05 and b.regime_up_threshold == 0.02

    def test_bounds(self):
        with pytest.raises(Exception):
            BacktestConfig(regime_down_threshold=0.01)
        with pytest.raises(Exception):
            BacktestConfig(regime_up_threshold=-0.01)

    @pytest.mark.parametrize("vol_exp", [True, False])
    def test_equal_to_the_old_chain_on_a_grid(self, vol_exp):
        for i in range(-200, 201):
            px = i / 1000.0
            assert research_regime(vol_exp, px, -0.05, 0.02) == _old_chain(vol_exp, px), px

    def test_boundaries_are_strict(self):
        assert research_regime(True, -0.05, -0.05, 0.02) == "high_volatility"
        assert research_regime(True, 0.02, -0.05, 0.02) == "high_volatility"
        assert research_regime(True, -0.0501, -0.05, 0.02) == "trending_down"
        assert research_regime(False, -0.5, -0.05, 0.02) == "range_bound"

    def test_thresholds_move_the_class(self):
        assert research_regime(True, -0.10, -0.05, 0.02) == "trending_down"
        assert research_regime(True, -0.10, -0.20, 0.02) == "high_volatility"
        assert research_regime(True, 0.03, -0.05, 0.05) == "high_volatility"


def _trending_down(synth_features: pd.DataFrame) -> pd.DataFrame:
    f = synth_features.copy()
    f["vol_regime_expanding"] = 1.0
    f["price_vs_sma_20"] = -0.10
    return f


class TestEngineSwitch:
    def _run(self, synth_df, synth_features, **over):
        kw = _engine_kwargs(synth_df, synth_features,
                            features_cache=_trending_down(synth_features), **over)
        bt = Backtester(**kw)
        return bt, bt.run()

    def test_gate_on_vetoes_trending_down_days(self, synth_df, synth_features):
        bt, res = self._run(synth_df, synth_features, regime_veto_gate=True)
        assert bt._funnel.get("regime_veto", 0) > 0
        # the only entries are on rows before the feature frame exists (class not computable)
        assert all(t["decision"].get("regime_class") != "trending_down" for t in res.trades)

    def test_gate_off_vetoes_nothing_and_still_records_the_class(self, synth_df, synth_features):
        _on, res_on = self._run(synth_df, synth_features, regime_veto_gate=True)
        bt, res = self._run(synth_df, synth_features, regime_veto_gate=False)
        assert bt._funnel.get("regime_veto", 0) == 0
        assert len(res.trades) > len(res_on.trades), "with the veto off the scenario takes more entries"
        classed = [t for t in res.trades if t["decision"].get("regime_class") == "trending_down"]
        assert classed, "entries on trending_down days must exist and be recorded"
        for t in classed:
            d = t["decision"]
            assert d["regime_veto_suppressed"] is True
            assert d["regime"] == "trending_down"           # entry_regime, same definition
            assert "regime_veto" not in d

    def test_a_threshold_below_the_move_removes_the_class(self, synth_df, synth_features):
        bt, res = self._run(synth_df, synth_features, regime_veto_gate=True,
                            regime_down_threshold=-0.20)
        assert bt._funnel.get("regime_veto", 0) == 0
        classes = {t["decision"].get("regime_class") for t in res.trades}
        assert "high_volatility" in classes and "trending_down" not in classes

    def test_the_entry_regime_and_the_veto_class_always_agree(self, synth_df, synth_features):
        bt, res = self._run(synth_df, synth_features, regime_veto_gate=False,
                            regime_down_threshold=-0.20, regime_up_threshold=0.05)
        classed = [t["decision"] for t in res.trades if "regime_class" in t["decision"]]
        assert classed, "no entry carried a regime class"
        for d in classed:
            assert d["regime"] == d["regime_class"] == "high_volatility"

    @pytest.mark.parametrize("down, expect_down, expect_hv", [(-0.05, 1.0, 0.0), (-0.20, 0.0, 1.0)])
    def test_meta_label_context_reads_the_configured_thresholds(
            self, synth_df, synth_features, down, expect_down, expect_hv):
        seen: list[dict] = []

        class Meta:
            is_trained = True

            def predict(self, ctx):
                seen.append(dict(ctx))
                return None

        bt = Backtester(**_engine_kwargs(
            synth_df, synth_features, features_cache=_trending_down(synth_features),
            regime_veto_gate=False, regime_down_threshold=down, meta_labeler=Meta()))
        bt.run()
        # rows before the cached feature frame are computed live; only the forced rows count
        forced = [c for c in seen if c.get("price_vs_sma_20") == pytest.approx(-0.10)]
        assert forced, "the meta-labeler never saw a forced trending-down row"
        assert all(c["regime_trending_down"] == expect_down and c["regime_high_vol"] == expect_hv
                   and c["regime_trending_up"] == 0.0 and c["regime_range_bound"] == 0.0
                   for c in forced)

    def test_unset_resolves_from_settings_and_bare_engines_keep_the_literals(self, synth_df,
                                                                            synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        b = BacktestConfig()
        assert (bt._regime_veto_gate, bt._regime_down, bt._regime_up) == (
            b.regime_veto_gate, b.regime_down_threshold, b.regime_up_threshold)
        bare = Backtester.__new__(Backtester)           # partial engines (tests) stay legacy
        assert (bare._regime_veto_gate, bare._regime_down, bare._regime_up,
                bare._risk_free_rate) == (True, -0.05, 0.02, 0.05)


# --------------------------------------------------------------------------
# walk-forward + optimizer forward the switch
# --------------------------------------------------------------------------

def _wfc(**kw):
    return WalkForwardConfig(train_window_models=False, **kw)


class TestForwarding:
    def test_resolution(self):
        assert _wfc().regime_veto_gate is True                    # settings default (no overlay)
        assert _wfc(regime_veto_gate=False).regime_veto_gate is False

    def test_the_engines_receive_it(self, synth_df, monkeypatch):
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(regime_veto_gate=False))
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink and sink[-1]["regime_veto_gate"] is False
        sink.clear()
        wf._train_window_meta_labeler(train_df=synth_df, symbol="SPY", window_id=1,
                                      predictor=None, window_cfg=wf._config)
        assert sink and sink[-1]["regime_veto_gate"] is False

    def test_the_window_optimizer_receives_it(self, synth_df, monkeypatch):
        import ait.optimization.optimizer as om
        seen = {}

        class Spy:
            def __init__(self, **kw):
                seen.update(kw)

            def run(self, **kw):
                raise RuntimeError("stop here")
        monkeypatch.setattr(om, "StrategyOptimizer", Spy)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(
            regime_veto_gate=False, optimize_per_window=True))
        wf._optimize_window_params(synth_df, "SPY", 1, ["iron_condor"])
        assert seen["regime_veto_gate"] is False

    def test_trial_engines_carry_it(self, monkeypatch, synth_df):
        sink: list = []
        monkeypatch.setattr(engine_mod, "Backtester", _recorder_factory(sink))
        opt = StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1,
                                regime_veto_gate=False)
        opt._data = {"SPY": synth_df}
        opt._run_backtest({})
        assert sink and sink[0]["regime_veto_gate"] is False


# --------------------------------------------------------------------------
# r comes from options.risk_free_rate
# --------------------------------------------------------------------------

class TestRiskFreeRate:
    def test_default_is_the_config_value(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._risk_free_rate == OptionsConfig().risk_free_rate == 0.05

    def test_explicit_value_reaches_the_engine_and_changes_prices(self, synth_df, synth_features):
        lo = Backtester(**_engine_kwargs(synth_df, synth_features, risk_free_rate=0.05))
        hi = Backtester(**_engine_kwargs(synth_df, synth_features, risk_free_rate=0.25))
        assert (lo._risk_free_rate, hi._risk_free_rate) == (0.05, 0.25)
        a = [round(float(t.get("pnl", 0.0)), 4) for t in lo.run().trades]
        b = [round(float(t.get("pnl", 0.0)), 4) for t in hi.run().trades]
        assert a and b and a != b, "r must change the modelled prices"

    def test_no_bare_literal_rate_assignment_left_in_the_engine(self):
        """AST (not a source-string grep): no `r = 0.05`-shaped assignment inside any method."""
        tree = ast.parse((SRC / "backtesting" / "engine.py").read_text())
        hits = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for n in ast.walk(fn):
                if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                        and isinstance(n.value.value, float)
                        and any(isinstance(t, ast.Name) and t.id in ("r", "rate", "risk_free")
                                for t in n.targets)):
                    hits.append(n.lineno)
        assert hits == [], f"literal risk-free-rate assignment(s) at engine.py lines {hits}"

    def test_replay_prices_with_the_same_rate(self):
        from ait.backtesting.replay import ENGINE_RISK_FREE_RATE, PricingParams
        bc = BacktestConfig()
        assert PricingParams.from_backtest_config(bc).r == ENGINE_RISK_FREE_RATE
        assert PricingParams.from_backtest_config(bc, risk_free_rate=0.031).r == pytest.approx(0.031)

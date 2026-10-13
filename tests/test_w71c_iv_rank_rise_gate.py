"""W71c: the research IV-rank RISE veto has an explicit on/off switch (decision D9: OFF).

  * schema default True (bare engines unchanged); the research overlay sets False and carries the
    written re-enable bar; the live config carries neither
  * engine: gate OFF vetoes nothing but still records the rise
  * walk-forward config resolves it and forwards it to the OOS engine, the shadow engine and the
    optimizer
  * optimizer: gate OFF drops the threshold from the search (the trap: a searched threshold used
    to override the baseline and silently re-enable the gate), ON keeps it
  * manifest: the RESEARCH-ONLY line appears only while the gate is ON
"""
from __future__ import annotations

import sys
from pathlib import Path

import optuna
import pandas as pd
import pytest
import yaml

import ait.backtesting.engine as engine_mod
import ait.backtesting.walkforward as wf_mod
from ait.backtesting.engine import Backtester
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.config.settings import BacktestConfig
from ait.optimization.optimizer import StrategyOptimizer
from ait.optimization.param_spaces import IRON_CONDOR_SPACE

from tests.test_r20_research_validity import (  # noqa: F401 - shared fixtures
    _engine_kwargs, _recorder_factory, _run_window, synth_df, synth_features)

ROOT = Path(__file__).resolve().parents[1]


def _yaml(name):
    return yaml.safe_load((ROOT / name).read_text()) or {}


# --------------------------------------------------------------------------
# schema + overlay
# --------------------------------------------------------------------------

class TestSwitchAndTheWrittenBar:
    def test_schema_default_is_on_so_nothing_else_moves(self):
        assert BacktestConfig().iv_rank_rise_gate is True
        assert BacktestConfig().iv_rank_rise_threshold == 0.30

    def test_the_research_overlay_turns_it_off_and_keeps_the_threshold(self):
        bt = _yaml("config_research.yaml")["backtest"]
        assert bt["iv_rank_rise_gate"] is False
        assert "iv_rank_rise_threshold" not in bt          # the 0.30 and its range are KEPT

    def test_the_live_config_carries_neither(self):
        live = _yaml("config.yaml").get("backtest") or {}
        assert "iv_rank_rise_gate" not in live

    def test_loaded_research_view(self):
        from ait.config.research import use_research_config
        from ait.config.settings import load_settings
        use_research_config()
        b = load_settings().backtest
        assert b.iv_rank_rise_gate is False and b.iv_rank_rise_threshold == pytest.approx(0.30)

    def test_the_bar_is_written_next_to_the_key(self):
        """The re-enable bar lives in the overlay itself, not only in the ledger (operator's ask)."""
        text = (ROOT / "config_research.yaml").read_text()
        for phrase in ("W71c", "WHAT:", "WHY OFF:", "WHAT WAS KEPT:", "RE-ENABLE BAR:",
                       "MULTI-WINDOW", "OUT-OF-SAMPLE", "live's level gates", "port it to live",
                       "iv_rank_parity_probe.py --ab"):
            assert phrase in text, phrase


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

def _ramped(synth_features: pd.DataFrame) -> pd.DataFrame:
    """iv_rank sawtooth: +0.5 over every 10 rows for half of the days (a veto at 0.30)."""
    f = synth_features.copy()
    f["iv_rank"] = [(i % 20) / 20.0 for i in range(len(f))]
    return f


class TestEngine:
    def _run(self, synth_df, synth_features, **over):
        kw = _engine_kwargs(synth_df, synth_features, iv_rank_rise_threshold=0.30,
                            features_cache=_ramped(synth_features), **over)
        bt = Backtester(**kw)
        return bt, bt.run()

    def test_gate_on_vetoes(self, synth_df, synth_features):
        bt, res = self._run(synth_df, synth_features, iv_rank_rise_gate=True)
        assert bt._funnel.get("iv_rank_rise", 0) > 0

    def test_gate_off_vetoes_nothing_and_takes_at_least_as_many(self, synth_df, synth_features):
        on, res_on = self._run(synth_df, synth_features, iv_rank_rise_gate=True)
        off, res_off = self._run(synth_df, synth_features, iv_rank_rise_gate=False)
        assert off._funnel.get("iv_rank_rise", 0) == 0
        assert len(res_off.trades) >= len(res_on.trades)
        assert len(res_off.trades) > len(res_on.trades) or on._funnel.get("iv_rank_rise", 0) > 0

    def test_the_rise_is_still_recorded_when_off(self, synth_df, synth_features):
        _, res = self._run(synth_df, synth_features, iv_rank_rise_gate=False)
        assert res.trades, "scenario must take entries"
        # recorded once 11 feature rows exist (the first few entries precede that)
        rises = [t["decision"]["iv_rank_rise_10d"] for t in res.trades
                 if t.get("decision", {}).get("iv_rank_rise_10d") is not None]
        assert rises, "the rise must still be written to the entry decision with the gate off"
        assert all(abs(abs(r) - 0.5) < 1e-6 for r in rises)        # the sawtooth: +/-0.5 over 10 rows
        assert any(r > 0.30 for r in rises)                        # days the gate would have vetoed

    def test_unset_resolves_from_settings(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._iv_rank_rise_gate is BacktestConfig().iv_rank_rise_gate     # schema default

    def test_a_high_threshold_with_the_gate_on_still_never_vetoes(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features, iv_rank_rise_gate=True,
                                         iv_rank_rise_threshold=10.0,
                                         features_cache=_ramped(synth_features)))
        bt.run()
        assert bt._funnel.get("iv_rank_rise", 0) == 0                          # backward compatible


# --------------------------------------------------------------------------
# walk-forward + optimizer (the trap)
# --------------------------------------------------------------------------

def _wfc(**kw):
    return WalkForwardConfig(train_window_models=False, **kw)


class TestWalkForward:
    def test_resolution(self):
        assert _wfc().iv_rank_rise_gate is True                       # settings default (no overlay)
        assert _wfc(iv_rank_rise_gate=False).iv_rank_rise_gate is False

    def test_the_engines_receive_it(self, synth_df, monkeypatch):
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(iv_rank_rise_gate=False))
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink and sink[-1]["iv_rank_rise_gate"] is False
        sink.clear()
        wf._train_window_meta_labeler(train_df=synth_df, symbol="SPY", window_id=1,
                                      predictor=None, window_cfg=wf._config)
        assert sink and sink[-1]["iv_rank_rise_gate"] is False

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
            iv_rank_rise_gate=False, optimize_per_window=True))
        wf._optimize_window_params(synth_df, "SPY", 1, ["iron_condor"])
        assert seen["iv_rank_rise_gate"] is False


class TestOptimizerTrap:
    @staticmethod
    def _opt(**kw):
        return StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1, **kw)

    @staticmethod
    def _suggested(opt):
        opt._data = {"SPY": pd.DataFrame({"Close": [100.0] * 30},
                                         index=pd.bdate_range("2024-01-02", periods=30))}
        opt._maybe_drop_inert_params()
        return opt._suggest_params(optuna.create_study().ask())

    def test_the_space_still_defines_the_dimension(self):
        assert IRON_CONDOR_SPACE["iv_rank_rise_threshold"] == ("float", 0.25, 0.60)

    def test_gate_on_searches_the_threshold(self):
        opt = self._opt(iv_rank_rise_gate=True)
        got = self._suggested(opt)
        assert "iron_condor__iv_rank_rise_threshold" in got
        assert "iron_condor__iv_rank_rise_threshold" not in opt._dropped_params

    def test_gate_off_does_not_search_it(self):
        opt = self._opt(iv_rank_rise_gate=False)
        got = self._suggested(opt)
        assert "iron_condor__iv_rank_rise_threshold" not in got
        assert "iron_condor__iv_rank_rise_threshold" in opt._dropped_params

    def test_bt_kwargs_carry_the_gate_and_a_searched_threshold_cannot_re_enable_it(
            self, monkeypatch, synth_df):
        sink: list = []
        monkeypatch.setattr(engine_mod, "Backtester", _recorder_factory(sink))
        opt = self._opt(iv_rank_rise_gate=False)
        opt._data = {"SPY": synth_df}
        # even if a threshold WERE supplied (a stale warm-start, a direct call), the engine is told
        # the gate is off, so the value is ignored there
        opt._run_backtest({"iron_condor__iv_rank_rise_threshold": 0.4})
        kw = sink[0]
        assert kw["iv_rank_rise_gate"] is False and kw["iv_rank_rise_threshold"] == pytest.approx(0.4)

    def test_warm_start_params_for_the_dropped_dimension_are_filtered(self):
        opt = self._opt(iv_rank_rise_gate=False)
        opt._data = {"SPY": pd.DataFrame({"Close": [100.0] * 30},
                                         index=pd.bdate_range("2024-01-02", periods=30))}
        opt._maybe_drop_inert_params()
        prior = {"iron_condor__iv_rank_rise_threshold": 0.4, "iron_condor__delta_short": 0.2}
        kept = {k: v for k, v in prior.items() if k not in opt._dropped_params}
        assert kept == {"iron_condor__delta_short": 0.2}


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------

class TestManifest:
    @staticmethod
    def _manifest(monkeypatch, *, gate, threshold=0.30):
        import run_backtest as rb
        from ait.config.settings import load_settings
        s = load_settings()
        s = s.model_copy(update={"backtest": s.backtest.model_copy(update={
            "iv_rank_rise_gate": gate, "iv_rank_rise_threshold": threshold})})
        monkeypatch.setattr(rb, "_research_settings", lambda: s)
        old = sys.argv
        sys.argv = ["run_backtest.py"]
        try:
            return rb.build_parity_manifest(rb.parse_args())
        finally:
            sys.argv = old

    def test_warns_only_while_on(self, monkeypatch):
        on = self._manifest(monkeypatch, gate=True)
        hits = [w for w in on["parity_warnings"] if w.startswith("iv_rank_rise_threshold")]
        assert len(hits) == 1 and "iv_rank_rise_gate: false" in hits[0]
        off = self._manifest(monkeypatch, gate=False)
        assert not [w for w in off["parity_warnings"] if w.startswith("iv_rank_rise_threshold")]
        assert off["backtest"]["iv_rank_rise_gate"] is False

    def test_an_ineffective_threshold_is_silent_even_with_the_gate_on(self, monkeypatch):
        m = self._manifest(monkeypatch, gate=True, threshold=10.0)
        assert not [w for w in m["parity_warnings"] if w.startswith("iv_rank_rise_threshold")]

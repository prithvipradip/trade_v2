"""W52a: the research engine applies the live expected-move band to condor
short strikes, from the same contract keys live reads — and records where
its strikes actually sit.

Registered by the 2026-09-22 post-FOMC study as an arm the harness could
not test: live rejects a condor unless each short strike sits within
[AIT_IC_EM_MIN, AIT_IC_EM_MAX] x expected move of spot (iron_condor.py's
R12-B gate; config home backtest.ic_em_min/max since W36). The engine
placed shorts by delta and never checked.

What writing the gate found: the engine's whole-dollar 0.20-delta
placement sits at ~0.90-1.00 EM on the call side and ~0.68-0.73 EM on the
put side (the lognormal drift pulls puts in; the $1 strike grid does the
rest), NOT the ~0.84 both sides a flat-normal sketch predicts. On the live
store every one of the 57 baseline condors is inside the band (gate on and
off: n 57 / PF 0.628, to the cent — reports/post_event_study/
entry_dte_18_{w52a_check,em_gate}), so the gate is ON by default and the
recorded baselines stand. On a $100 synthetic series with EM under ~5 the
$1 grid can push a put below 0.6 EM; the gate then rejects, as live would —
that case is kept executable below.
"""
from __future__ import annotations

from datetime import date

import pytest

import ait.backtesting.engine as engine_mod
from ait.backtesting.engine import Backtester
from ait.config.runtime_env import contract_float

from tests.test_r20_research_validity import (  # noqa: F401 — shared fixtures
    _engine_kwargs,
    synth_df,
    synth_features,
)
from tests.test_w2_orchestrator_gates import LogCapture

TODAY = date(2024, 6, 3)


def _condor(bt: Backtester, S: float = 100.0, iv: float = 0.20, dte: int = 18):
    return bt._build_credit_position(
        "iron_condor", S, iv, dte / 365.0, 0.05, dte, TODAY, 100_000.0)


class TestTheBandIsTheLiveContract:
    def test_defaults_follow_the_contract_keys_and_the_gate_is_on(self, synth_df, synth_features, monkeypatch):
        monkeypatch.delenv("AIT_BT_EM_GATE", raising=False)
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._ic_em_min == contract_float("AIT_IC_EM_MIN")
        assert bt._ic_em_max == contract_float("AIT_IC_EM_MAX")
        assert (bt._ic_em_min, bt._ic_em_max) == (0.6, 1.3)
        assert bt._em_gate is True

    def test_explicit_arguments_win(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features),
                        ic_em_min=0.9, ic_em_max=1.1, em_gate=False)
        assert (bt._ic_em_min, bt._ic_em_max, bt._em_gate) == (0.9, 1.1, False)

    def test_the_env_switch_turns_the_gate_off(self, synth_df, synth_features, monkeypatch):
        monkeypatch.setenv("AIT_BT_EM_GATE", "0")
        assert Backtester(**_engine_kwargs(synth_df, synth_features))._em_gate is False


class TestWhereDeltaPlacementActuallySits:
    @pytest.mark.parametrize("S,iv,dte", [
        (100.0, 0.20, 18), (450.0, 0.12, 21), (200.0, 0.35, 14), (600.0, 0.15, 18),
    ])
    def test_the_em_ratios_are_recorded(self, synth_df, synth_features, S, iv, dte):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        pos = _condor(bt, S, iv, dte)
        assert pos is not None
        assert 0.85 <= pos["em_ratio_call"] <= 1.05, pos["em_ratio_call"]
        assert 0.60 <= pos["em_ratio_put"] <= 0.80, pos["em_ratio_put"]
        em = S * iv * (dte / 365.0) ** 0.5
        assert pos["em_ratio_call"] == pytest.approx((pos["short_call_strike"] - S) / em, abs=1e-3)
        assert pos["em_ratio_put"] == pytest.approx((S - pos["short_put_strike"]) / em, abs=1e-3)


class TestTheGateBites:
    def test_a_band_above_the_placement_rejects_the_condor(self, synth_df, synth_features, monkeypatch):
        cap = LogCapture(engine_mod.log)
        monkeypatch.setattr(engine_mod, "log", cap)
        bt = Backtester(**_engine_kwargs(synth_df, synth_features), ic_em_min=1.1)
        assert _condor(bt) is None
        skips = cap.kwargs_for("condor_rejected_em_band")
        assert skips and skips[0]["em_min"] == 1.1

    def test_a_band_below_the_placement_rejects_the_condor(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features), ic_em_max=0.8)
        assert _condor(bt) is None

    def test_the_same_band_with_the_gate_off_builds_it(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features), ic_em_max=0.8, em_gate=False)
        assert _condor(bt) is not None

    def test_a_gated_run_opens_nothing_where_the_plain_run_trades(self, synth_df, synth_features):
        kw = _engine_kwargs(synth_df, synth_features)
        n_plain = len(Backtester(**kw, em_gate=False).run().trades)
        assert n_plain > 0, "the synthetic series must produce entries for this test to mean anything"
        assert len(Backtester(**kw, ic_em_min=1.1).run().trades) == 0


class TestTheGridArtifactOnASyntheticSeries:
    def test_gate_off_is_the_pre_w52a_run_plus_the_recorded_ratios(self, synth_df, synth_features):
        kw = _engine_kwargs(synth_df, synth_features)
        off = Backtester(**kw, em_gate=False).run()
        wide = Backtester(**kw, ic_em_min=0.0, ic_em_max=99.0).run()
        assert len(off.trades) == len(wide.trades) > 0
        assert off.trades == wide.trades
        assert off.final_capital == pytest.approx(wide.final_capital)

    def test_on_a_100_dollar_series_the_live_band_removes_grid_snapped_puts(self, synth_df, synth_features):
        """Kept executable: at EM under ~5 the $1 grid can push a put below
        0.6 EM and the gate rejects it, as live would. The live universe
        (EM 9-20) never reaches this — see the module docstring."""
        kw = _engine_kwargs(synth_df, synth_features)
        off = Backtester(**kw, em_gate=False).run()
        on = Backtester(**kw).run()
        assert 0 < len(on.trades) < len(off.trades), (len(on.trades), len(off.trades))
        outside = [t for t in off.trades
                   if not (0.6 <= t["em_ratio_put"] <= 1.3 and 0.6 <= t["em_ratio_call"] <= 1.3)]
        assert outside, "every gate-off trade sat inside the band, yet the gate removed some"
        assert all(t["em_ratio_put"] < 0.6 for t in outside), "the artifact is on the put side"

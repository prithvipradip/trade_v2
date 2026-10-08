"""Wing budget study: the research-only max-loss cap and contract cap must be
inert by default, must cap exactly as the live builder does, and must reach
every window worker.

Prompted by 2026-10-08: the operator asked for a $2,000 maximum loss per
condor (the QQQ condor entered 09-28 carried $4,995). Live already narrows
wings to fit `risk.max_position_risk_pct x NLV`
(iron_condor._vol_scaled_width + the `max_loss > risk_budget` reject); the
engine had no such cap, and it sizes contracts by `position_size_pct`, so
narrower wings would silently have become MORE contracts. These tests cover
the injection path only, same shape as tests/test_post_event_study_injection.py.
"""
from __future__ import annotations

import pickle
from datetime import date

import pytest

import ait.backtesting.walkforward as wf_mod
from ait.backtesting.engine import Backtester
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

from tests.test_r20_research_validity import (  # noqa: F401 — shared fixtures
    _engine_kwargs,
    _recorder_factory,
    _run_window,
    synth_df,
    synth_features,
)

TODAY = date(2024, 6, 3)


def _condor(bt: Backtester, S: float = 450.0, iv: float = 0.20, dte: int = 18,
            capital: float = 100_000.0):
    return bt._build_credit_position(
        "iron_condor", S, iv, dte / 365.0, 0.05, dte, TODAY, capital)


class TestInertByDefault:
    def test_engine_defaults_are_off(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._max_loss_cap is None and bt._contracts_cap is None
        pos = _condor(bt)
        assert pos is not None and pos["wing_capped"] is False

    def test_a_cap_above_the_structure_changes_nothing(self, synth_df, synth_features):
        kw = _engine_kwargs(synth_df, synth_features)
        plain = Backtester(**kw).run()
        huge = Backtester(**kw, max_loss_per_contract_dollars=10_000_000.0).run()
        assert len(plain.trades) == len(huge.trades) > 0
        assert plain.trades == huge.trades

    def test_config_defaults_are_off(self):
        cfg = WalkForwardConfig()
        assert cfg.max_loss_per_contract_dollars is None and cfg.contracts_cap is None


class TestTheCapMirrorsLive:
    def test_wing_is_narrowed_to_the_affordable_width(self, synth_df, synth_features):
        """Live: width <= budget / (100 * max(0.5, 1 - credit/width floor))."""
        kw = _engine_kwargs(synth_df, synth_features, ic_min_credit_width=0.10)
        wide = _condor(Backtester(**kw))
        assert wide["wing_width"] > 22.5, "precondition: the uncapped wing is wider than the cap"
        capped = _condor(Backtester(**kw, max_loss_per_contract_dollars=1990.0))
        assert capped is not None and capped["wing_capped"] is True
        assert capped["wing_width"] == pytest.approx(1990.0 / (100 * 0.90), abs=0.01)
        assert capped["max_loss_per_contract"] <= 1990.0

    def test_a_structure_still_over_budget_is_rejected(self, synth_df, synth_features):
        """A $50 budget makes the affordable width $0.50; live (and the engine)
        floor the wing at $1, so the built condor's max loss (~$1 x 100 less a
        tiny credit) exceeds the budget and must be refused, as live refuses
        `max_loss > risk_budget`."""
        kw = _engine_kwargs(synth_df, synth_features)          # credit floor 0.0
        assert _condor(Backtester(**kw, max_loss_per_contract_dollars=50.0)) is None
        assert _condor(Backtester(**kw, max_loss_per_contract_dollars=400.0)) is not None

    def test_contracts_are_capped(self, synth_df, synth_features):
        kw = _engine_kwargs(synth_df, synth_features)
        many = _condor(Backtester(**kw), capital=10_000_000.0)
        assert many["contracts"] > 1, "precondition: sizing would pick several contracts"
        one = _condor(Backtester(**kw, contracts_cap=1), capital=10_000_000.0)
        assert one["contracts"] == 1

    def test_a_capped_run_trades_narrower_condors(self, synth_df, synth_features):
        kw = _engine_kwargs(synth_df, synth_features)
        plain = Backtester(**kw).run()
        capped = Backtester(**kw, max_loss_per_contract_dollars=150.0, contracts_cap=1).run()
        assert plain.trades and capped.trades
        assert max(t["max_loss_per_contract"] for t in capped.trades) <= 150.0
        assert any(t["wing_capped"] for t in capped.trades)


class TestWalkForwardThreading:
    def test_refused_with_per_window_optimization(self):
        with pytest.raises(ValueError, match="max_loss_per_contract_dollars"):
            WalkForwardConfig(max_loss_per_contract_dollars=1990.0, optimize_per_window=True)
        with pytest.raises(ValueError, match="contracts_cap"):
            WalkForwardConfig(contracts_cap=1, optimize_per_window=True)

    @pytest.mark.parametrize("bad", [{"max_loss_per_contract_dollars": 0},
                                     {"max_loss_per_contract_dollars": -5},
                                     {"contracts_cap": 0}])
    def test_nonsense_values_are_refused(self, bad):
        with pytest.raises(ValueError):
            WalkForwardConfig(**bad)

    def test_survives_the_process_pool_boundary(self):
        cfg = pickle.loads(pickle.dumps(WalkForwardConfig(
            max_loss_per_contract_dollars=1990, contracts_cap=1)))
        assert cfg.max_loss_per_contract_dollars == 1990.0 and cfg.contracts_cap == 1

    def test_oos_backtester_receives_it(self, synth_df, monkeypatch):
        cfg = WalkForwardConfig(train_window_models=False,
                                max_loss_per_contract_dollars=1990.0, contracts_cap=1)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink, "no OOS Backtester was constructed"
        assert sink[-1].get("max_loss_per_contract_dollars") == 1990.0
        assert sink[-1].get("contracts_cap") == 1

    def test_meta_label_shadow_backtester_receives_it(self, synth_df, monkeypatch):
        cfg = WalkForwardConfig(max_loss_per_contract_dollars=2643.0, contracts_cap=1)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        wf._train_window_meta_labeler(
            train_df=synth_df, symbol="SPY", window_id=1,
            predictor=None, window_cfg=cfg)
        assert sink, "shadow Backtester was not constructed"
        assert sink[-1].get("max_loss_per_contract_dollars") == 2643.0

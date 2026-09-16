"""Take-profit study: the credit TP ladder is injectable for RESEARCH only.

Prompted by the 2026-09-11 closes (SPY/QQQ took profit at the 40% DTE-14
tier; real IBKR mid-marks reached ~50% by that afternoon). Two winners cannot
settle a threshold, so scripts/take_profit_study.py sweeps ladder variants
through walk-forward. That needs the ladder to reach the engine in every
window, including windows run in a ProcessPoolExecutor, without touching live.

Every test EXECUTES the real code. None reads source text.
"""
from __future__ import annotations

import pickle
from datetime import date

import pytest

import ait.backtesting.walkforward as wf_mod
from ait.backtesting.engine import Backtester
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.execution import exit_policy

from tests.test_r20_research_validity import (  # noqa: F401 — shared fixtures
    _engine_kwargs,
    _recorder_factory,
    _run_window,
    synth_df,
    synth_features,
)

T = -(10 ** 9)  # exit_policy.TERMINAL_DTE_FLOOR; literal so pre-change runs fail per test
HIGHER_14DTE = ((20, 1.00, 0.50), (10, 0.75, 0.50), (5, 0.50, 0.30), (T, 0.25, 0.20))
FLAT_50 = ((T, 1.00, 0.50),)


# ------------------------------------------------------------ live is untouched
class TestDefaultLadderUnchanged:
    @pytest.mark.parametrize("dte,expected", [(25, 0.50), (14, 0.40), (8, 0.30), (3, 0.20)])
    def test_default_matches_the_live_ladder(self, dte, expected):
        assert exit_policy.credit_take_profit_pct(dte) == expected
        assert exit_policy.credit_take_profit_pct(dte, ladder=None) == expected

    def test_flag_off_still_collapses_to_flat_default(self):
        """Live parity: exit.time_decay_scaling=false means flat 0.50, and a
        research ladder must not quietly override that."""
        assert exit_policy.credit_take_profit_pct(
            14, time_decay_scaling=False, ladder=HIGHER_14DTE) == 0.50

    def test_engine_default_is_the_shared_ladder(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._credit_tp_ladder is None


# ------------------------------------------------------------ the override works
class TestCustomLadder:
    def test_custom_ladder_changes_the_target(self):
        assert exit_policy.credit_take_profit_pct(14, ladder=HIGHER_14DTE) == 0.50
        assert exit_policy.credit_take_profit_pct(3, ladder=FLAT_50) == 0.50

    def test_no_take_profit_variant_never_fires(self):
        never = ((T, 1.00, exit_policy.NO_TAKE_PROFIT),)
        assert 1.0 < exit_policy.credit_take_profit_pct(14, ladder=never)

    @pytest.mark.parametrize("bad", [
        (),                                            # empty
        ((10, 0.75, 0.40),),                           # no terminal row: DTE<=10 falls through
        ((5, 0.5, 0.30), (10, 0.75, 0.40), (T, 0.25, 0.20)),  # floors not descending
        ((T, 1.00, 0.0),),                             # non-positive target
    ])
    def test_malformed_ladders_are_refused(self, bad):
        with pytest.raises(ValueError):
            exit_policy.validate_ladder(bad)

    def test_lists_from_json_are_accepted(self):
        assert exit_policy.validate_ladder([[T, 1.0, 0.5]]) == FLAT_50

    def test_terminal_floor_constant_matches_the_live_ladder(self):
        assert exit_policy.CREDIT_TP_LADDER[-1][0] == exit_policy.TERMINAL_DTE_FLOOR == T


class TestEngineExitDecision:
    """45% of credit captured at DTE 14: the live ladder closes (40% tier),
    a ladder with a 50% DTE-14 tier holds."""

    POS = {"expiry_date": "2024-06-17", "entry_date": "2024-06-01",
           "strategy": "iron_condor"}
    TODAY = date(2024, 6, 3)  # DTE 14

    def test_default_ladder_closes(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        out = bt._check_exit_credit(dict(self.POS), 0.45, self.TODAY)
        assert out is not None and out["exit_reason"] == "take_profit_short"

    def test_higher_tier_holds(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(
            synth_df, synth_features, credit_tp_ladder=HIGHER_14DTE))
        assert bt._check_exit_credit(dict(self.POS), 0.45, self.TODAY) is None

    def test_higher_tier_still_closes_at_its_own_target(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(
            synth_df, synth_features, credit_tp_ladder=HIGHER_14DTE))
        out = bt._check_exit_credit(dict(self.POS), 0.51, self.TODAY)
        assert out is not None and out["exit_reason"] == "take_profit_short"


# ------------------------------------------------------------ walk-forward threading
class TestWalkForwardThreading:
    def test_config_normalises_and_validates(self):
        cfg = WalkForwardConfig(credit_tp_ladder=[[T, 1.0, 0.5]])
        assert cfg.credit_tp_ladder == FLAT_50
        with pytest.raises(ValueError):
            WalkForwardConfig(credit_tp_ladder=((10, 0.75, 0.40),))

    def test_default_config_leaves_the_ladder_alone(self):
        assert WalkForwardConfig().credit_tp_ladder is None

    def test_refused_with_per_window_optimization(self):
        """Optuna trials build their own Backtester and would score the
        DEFAULT ladder while OOS ran the custom one: refuse, don't diverge."""
        with pytest.raises(ValueError, match="optimize_per_window"):
            WalkForwardConfig(credit_tp_ladder=FLAT_50, optimize_per_window=True)

    def test_survives_the_process_pool_boundary(self):
        """ProcessPoolExecutor pickles the config into each window worker
        (_window_task_mp). A module-level patch would not cross; a field does."""
        cfg = pickle.loads(pickle.dumps(WalkForwardConfig(credit_tp_ladder=HIGHER_14DTE)))
        assert cfg.credit_tp_ladder == HIGHER_14DTE

    def test_oos_backtester_receives_it(self, synth_df, monkeypatch):
        cfg = WalkForwardConfig(train_window_models=False, credit_tp_ladder=HIGHER_14DTE)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink, "no OOS Backtester was constructed"
        assert sink[-1].get("credit_tp_ladder") == HIGHER_14DTE

    def test_meta_label_shadow_backtester_receives_it(self, synth_df, monkeypatch):
        """The shadow run labels training trades for the MetaLabeler; labels
        made under a different exit rule than the OOS run are mislabeled."""
        cfg = WalkForwardConfig(credit_tp_ladder=HIGHER_14DTE)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        wf._train_window_meta_labeler(
            train_df=synth_df, symbol="SPY", window_id=1,
            predictor=None, window_cfg=cfg)
        assert sink, "shadow Backtester was not constructed"
        assert sink[-1].get("credit_tp_ladder") == HIGHER_14DTE


# ------------------------------------------------------------ entry DTE parity
class TestEntryDteThreading:
    """Live condors enter at a median 18 DTE (14-25 measured over 14 trades)
    while the engine defaulted every window to dte_range[0] = 14. The ladder
    tiers are keyed to DTE, so the study needs entry_dte to reach the engine."""

    def test_default_defers_to_config(self):
        assert WalkForwardConfig().entry_dte is None

    def test_oos_backtester_receives_it(self, synth_df, monkeypatch):
        cfg = WalkForwardConfig(train_window_models=False, entry_dte=18)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink and sink[-1].get("entry_dte") == 18

    def test_meta_label_shadow_backtester_receives_it(self, synth_df, monkeypatch):
        cfg = WalkForwardConfig(entry_dte=18)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        wf._train_window_meta_labeler(
            train_df=synth_df, symbol="SPY", window_id=1,
            predictor=None, window_cfg=cfg)
        assert sink and sink[-1].get("entry_dte") == 18

    def test_refused_with_per_window_optimization(self):
        with pytest.raises(ValueError, match="optimize_per_window"):
            WalkForwardConfig(entry_dte=18, optimize_per_window=True)

    def test_engine_prices_the_requested_dte(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features, entry_dte=18))
        assert bt._entry_dte == 18

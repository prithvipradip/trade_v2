"""W108 S2: `_maybe_drop_inert_params` drops `iron_condor__wing_k` exactly when the engine makes it inert.

The engine's condor wing at `wing_k = k` is `free(k) = max(k x EM, wing_floor)`, replaced by the budget
ceiling `max(1, round(affordable))` whenever `affordable < free(k)` (`budget_aware_wings`). `free` never falls
as k rises, so the ceiling binds at every k of the searched range iff `affordable < max(k_min x EM, floor)`.
Three defects of the W65 rule, each tested here against the rule AND, where it matters, against the real
engine builder (a structure built at k_min and at k_max):

* the wing floor was missing: a ceiling below the $2 floor (3% of $5k) builds the same condor at every k, but
  the rule kept searching whenever `k_min x EM` fell under the ceiling (short DTE, a DTE study);
* the probe read every loaded symbol, while the trials backtest only `symbols[0]`;
* a FALSE drop: every other credit builder reads the same engine `wing_k` with no budget cap, so with one of
  them in the study the dimension is not inert.

Negative controls keep the dimension where it does change a structure.
"""
from __future__ import annotations

from datetime import date

import optuna
import pandas as pd
import pytest
from structlog.testing import capture_logs

import ait.backtesting.engine as engine_mod
from ait.backtesting.engine import Backtester
from ait.config.settings import DteSearchConfig
from ait.optimization.optimizer import StrategyOptimizer
from ait.strategies.iron_condor import IronCondor
from tests.test_r20_research_validity import (  # noqa: F401  (module fixtures)
    _engine_kwargs, _recorder_factory, synth_df, synth_features)

WING = "iron_condor__wing_k"
FLOOR = 2.0                      # the shipped AIT_IC_WING_FLOOR, passed explicitly so the tests do not read it
CAPITAL = 5_000.0
K_MIN, K_MAX = 0.30, 2.00        # IRON_CONDOR_SPACE["wing_k"]
TODAY = date(2024, 6, 3)
# Credit builders other than the condor that read the engine's ONE wing_k with no budget cap.
# (put_credit_spread also reads it but carries its own `wing_k` dimension, which overrides the condor's in
# _run_backtest when it comes later: a separate, pre-existing collision, not asserted here.)
UNCAPPED_WING_READERS = ["iron_butterfly", "wide_wing_condor", "broken_wing_condor",
                         "call_credit_spread", "jade_lizard"]


def _frame(spot: float, iv: float) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=160)
    return pd.DataFrame({"Open": spot, "High": spot, "Low": spot, "Close": spot, "Volume": 1e6,
                         "implied_vol": iv}, index=idx)


def _opt(pct: float, data: dict, *, symbols=None, strategies=("iron_condor",), **kw) -> StrategyOptimizer:
    kw.setdefault("entry_dte", 7)
    o = StrategyOptimizer(symbols=list(symbols or data), strategies=list(strategies), n_trials=1,
                          initial_capital=CAPITAL, position_size_pct=pct, budget_aware_wings=True,
                          wing_floor_dollars=FLOOR, penalty_on_range_prob=0.05, **kw)
    o._data = data
    o._maybe_drop_inert_params()
    return o


def _em(spot: float, iv: float, dte: int) -> float:
    return spot * iv * (dte / 365.0) ** 0.5


def _structure(bt: Backtester, strategy: str, capital: float, S=100.0, iv=0.20, dte=7):
    """What the REAL engine builder builds (None = refused)."""
    p = bt._build_credit_position(strategy, S, iv, dte / 365.0, 0.05, dte, TODAY, capital)
    if p is None:
        return None
    return tuple(p.get(k) for k in ("wing_width", "short_call_strike", "short_put_strike",
                                    "long_call_strike", "long_put_strike"))


def _built_at(synth_df, synth_features, strategy, k, pct):  # noqa: F811
    bt = Backtester(**_engine_kwargs(synth_df, synth_features, strategies=[strategy], budget_aware_wings=True,
                                     position_size_pct=pct, initial_capital=CAPITAL, wing_k=k,
                                     wing_floor_dollars=FLOOR))
    return _structure(bt, strategy, CAPITAL)


# --------------------------------------------------------------------------------------------------------
# (a) the ceiling below the wing floor
# --------------------------------------------------------------------------------------------------------
class TestTheCeilingBelowTheWingFloor:
    # $100 / $150 budgets (2% / 3% of $5k): ceiling 1.11 / 1.67 at the contract's 0.10 credit/width
    @pytest.mark.parametrize("pct", [0.02, 0.03])
    def test_dropped_although_k_min_x_em_is_under_the_ceiling(self, pct):
        afford = IronCondor.affordable_width(CAPITAL * pct)
        k_min_em = K_MIN * _em(100.0, 0.20, 7)                            # 0.83
        assert k_min_em < afford < FLOOR, "precondition: only the floor makes the ceiling bind at k_min"
        o = _opt(pct, {"IWM": _frame(100.0, 0.20)})
        assert WING in o._dropped_params

    def test_dropped_in_a_dte_study_that_reaches_one_day(self):
        o = _opt(0.03, {"IWM": _frame(100.0, 0.20)}, entry_dte=None,
                 dte_search=DteSearchConfig(enabled=True, entry_dte=[0, 45]))
        assert WING in o._dropped_params

    def test_the_trials_do_not_sample_it_and_the_recorded_space_omits_it(self):
        o = _opt(0.03, {"IWM": _frame(100.0, 0.20)})
        assert WING not in o._suggest_params(optuna.create_study().ask())
        assert "wing_k" not in o.effective_search_space()["iron_condor"]
        assert "delta_short" in o.effective_search_space()["iron_condor"]

    def test_negative_control_a_ceiling_above_the_floor_and_above_k_min_x_em_keeps_it(self):
        afford = IronCondor.affordable_width(CAPITAL * 0.05)
        assert FLOOR < afford and K_MIN * _em(100.0, 0.20, 7) < afford
        o = _opt(0.05, {"IWM": _frame(100.0, 0.20)})
        assert WING not in o._dropped_params
        assert WING in o._suggest_params(optuna.create_study().ask())

    def test_engine_truth_the_dropped_config_builds_one_structure_the_kept_one_two(
            self, synth_df, synth_features):  # noqa: F811
        lo, hi = (_built_at(synth_df, synth_features, "iron_condor", k, 0.03) for k in (K_MIN, K_MAX))
        assert lo is not None and lo == hi            # 3%: the same 2-wide condor at k_min and at k_max
        lo5, hi5 = (_built_at(synth_df, synth_features, "iron_condor", k, 0.05) for k in (K_MIN, K_MAX))
        assert lo5 is not None and hi5 is not None and lo5[0] < hi5[0]   # 5%: 2-wide at k_min, 3-wide at k_max


# --------------------------------------------------------------------------------------------------------
# (b) only the symbol the trials trade
# --------------------------------------------------------------------------------------------------------
class TestOnlyTheTradedSymbolIsProbed:
    DATA = {"SPY": _frame(700.0, 0.15), "IWM": _frame(100.0, 0.15)}

    def test_preconditions(self):
        afford = IronCondor.affordable_width(CAPITAL * 0.05)
        assert K_MIN * _em(700.0, 0.15, 14) > afford                       # inert on SPY
        assert max(K_MIN * _em(100.0, 0.15, 14), FLOOR) < afford           # not inert on IWM

    def test_an_untraded_symbol_does_not_keep_the_dimension(self):
        o = _opt(0.05, dict(self.DATA), symbols=["SPY", "IWM"], entry_dte=14)
        assert WING in o._dropped_params

    def test_negative_control_the_traded_symbol_decides(self):
        o = _opt(0.05, dict(self.DATA), symbols=["IWM", "SPY"], entry_dte=14)
        assert WING not in o._dropped_params

    def test_the_probed_symbol_is_the_one_the_trial_backtests(self, monkeypatch):
        sink: list = []
        monkeypatch.setattr(engine_mod, "Backtester", _recorder_factory(sink))
        o = _opt(0.05, dict(self.DATA), symbols=["SPY", "IWM"], entry_dte=14)
        o._run_backtest({})
        assert sink[0]["data"] is o._data["SPY"]


# --------------------------------------------------------------------------------------------------------
# (c) a wing_k shared with another credit builder is never dropped
# --------------------------------------------------------------------------------------------------------
class TestASharedWingKIsNotDropped:
    SPY = {"SPY": _frame(700.0, 0.15)}

    def test_precondition_the_condor_alone_drops_it(self):
        assert WING in _opt(0.05, dict(self.SPY), entry_dte=14)._dropped_params

    @pytest.mark.parametrize("other", UNCAPPED_WING_READERS)
    def test_another_credit_builder_keeps_it(self, other):
        o = _opt(0.05, dict(self.SPY), strategies=("iron_condor", other), entry_dte=14)
        assert WING not in o._dropped_params
        assert WING in o._suggest_params(optuna.create_study().ask())

    def test_negative_control_a_debit_strategy_does_not(self):
        o = _opt(0.05, dict(self.SPY), strategies=("iron_condor", "bull_call_spread"), entry_dte=14)
        assert WING in o._dropped_params

    def test_engine_truth_the_butterfly_changes_with_wing_k(self, synth_df, synth_features):  # noqa: F811
        lo, hi = (_built_at(synth_df, synth_features, "iron_butterfly", k, 0.05) for k in (K_MIN, K_MAX))
        assert lo != hi


# --------------------------------------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------------------------------------
class TestTheDropIsLogged:
    @staticmethod
    def _event(**kw):
        with capture_logs() as ev:
            _opt(**kw)
        return [e for e in ev if e.get("event") == "optimizer_dropped_inert_param" and e.get("param") == WING]

    def test_the_floor_case_names_the_floor(self):
        (e,) = self._event(pct=0.03, data={"IWM": _frame(100.0, 0.20)})
        assert e["binds"] == "wing_floor" and e["wing_floor"] == FLOOR
        assert e["affordable_wing"] == pytest.approx(IronCondor.affordable_width(CAPITAL * 0.03), abs=0.01)
        assert e["searched_range"] == [K_MIN, K_MAX]

    def test_the_expected_move_case_names_the_symbol_and_the_probe(self):
        (e,) = self._event(pct=0.05, data={"SPY": _frame(700.0, 0.15), "IWM": _frame(100.0, 0.15)},
                           symbols=["SPY", "IWM"], entry_dte=14)
        assert e["binds"] == "expected_move" and e["symbol"] == "SPY" and e["probe_dte"] == 14
        assert e["smallest_searched_wing"] == pytest.approx(K_MIN * _em(700.0, 0.15, 14), abs=0.01)

    def test_nothing_is_logged_when_it_is_kept(self):
        assert self._event(pct=0.05, data={"IWM": _frame(100.0, 0.20)}) == []

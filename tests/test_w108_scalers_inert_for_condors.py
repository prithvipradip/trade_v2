"""W108 phase 3c: where the sizing scalers can and cannot change a condor's lot count.

The shared lot rule (`ait.risk.lot_sizing`) decides viability on the UNSCALED budget and lets a scaler
(clamped at 1) only throttle the count, `max(1, floor(budget x scaler / risk))`. Driving the real engine
builder with the scalers ON and a deliberately small scaler:

* at the $5,000 simulated account, every budget up to 20% builds ONE lot with or without scalers: the
  budget ceiling sets the wing, so one lot risks most of the budget and there is nothing to throttle;
* on a large account the free (market) width binds below the ceiling, one lot risks a fraction of the
  budget, and the scaler DOES cut the count (3 and 7 lots -> 1 here, scaler 0.02). That is the configuration the 3c study
  is about; with the shipped `positions.max_contracts_per_trade: 1` it is capped to one lot either way;
* in every case the scaler never turns a viable condor into a refusal, or the reverse.
"""
from __future__ import annotations

from datetime import date

import pytest

from ait.backtesting.engine import Backtester
from ait.config.settings import PositionSizingConfig, Settings
from tests.test_r20_research_validity import _engine_kwargs, synth_df, synth_features  # noqa: F401

TODAY = date(2024, 6, 3)
LOW_SCALER_CTX = {"strategy": "iron_condor", "confidence": 0.50, "iv": 0.45, "leg_iv": 0.45,
                  "iv_rank": 5.0, "losing_days": 4, "vix": 32.0}


def _built(bt: Backtester, capital: float):
    return bt._build_credit_position("iron_condor", 450.0, 0.20, 18 / 365.0, 0.05, 18, TODAY, capital)


def _pair(synth_df, synth_features, budget, capital):
    kw = _engine_kwargs(synth_df, synth_features, budget_aware_wings=True, position_size_pct=budget,
                        initial_capital=capital)
    off = Backtester(**kw)
    on = Backtester(**kw, settings=Settings(
        position_sizing=PositionSizingConfig(scaled_strategies=["iron_condor"])))
    assert on._scalers_on and not off._scalers_on
    on._size_ctx = dict(LOW_SCALER_CTX)
    p_off, p_on = _built(off, capital), _built(on, capital)
    assert (p_off is None) == (p_on is None)       # the scaler never decides viability
    return on, p_off, p_on


@pytest.mark.parametrize("budget", [0.05, 0.10, 0.15, 0.20])
def test_at_the_5k_account_a_condor_is_one_lot_with_or_without_scalers(synth_df, synth_features, budget):
    on, p_off, p_on = _pair(synth_df, synth_features, budget, 5_000.0)
    if p_off is None:
        return
    assert on._last_scaler < 1.0, "precondition: the scaler is actually throttling"
    assert p_off["contracts"] == p_on["contracts"] == 1
    assert p_off["wing_width"] == p_on["wing_width"]
    assert p_off["max_loss_per_contract"] > 0.5 * 5_000.0 * budget   # most of the budget in one lot


@pytest.mark.parametrize("budget,unscaled", [(0.10, 3), (0.20, 7)])
def test_on_a_large_account_the_scaler_cuts_the_count(synth_df, synth_features, budget, unscaled):
    on, p_off, p_on = _pair(synth_df, synth_features, budget, 100_000.0)
    assert p_off["contracts"] == unscaled
    assert 1 <= p_on["contracts"] < p_off["contracts"]
    assert p_off["wing_width"] == p_on["wing_width"]       # same structure, fewer lots

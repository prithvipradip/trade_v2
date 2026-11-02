"""The research engine's range gate must fail CLOSED when the range model raises during ``predict``.

Registered in PLAN.md ("FINDING ... the range gate silently disappears when the range model raises"). The engine used to
wrap the gate in ``try: ... except Exception: pass``, so an error inside ``predict`` removed the gate for that entry: the
entry flowed on as if the model had approved it and no rejection, would-reject or log line recorded it. A model that is
absent, abstains or predicts under the floor was already blocked; an error is now blocked the same way, under the name
``range_gate_error`` (``gates_mode: enforce``) or recorded as a would-reject (``observe``).

Real ``Backtester`` on synthetic data with a fake range model, the pattern of ``test_w67_regime_gate.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from ait.backtesting.engine import Backtester
from ait.backtesting.gate_parity import TWIN, classify


@pytest.fixture(scope="module")
def synth_df() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=160)
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(idx))))
    return pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996,
                         "Close": close, "Volume": 1e6}, index=idx)


@pytest.fixture(scope="module")
def synth_features(synth_df) -> pd.DataFrame:
    from ait.ml.features import FeatureEngine
    return FeatureEngine().compute(synth_df)


class FakeRange:
    """What the gate reads from a range model: ``predict(...)`` returning an object with ``probability_in_range``."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def predict(self, hist, **kw):
        if self.mode == "raise":
            raise RuntimeError("model blew up")
        if self.mode == "abstain":
            return None
        if self.mode == "malformed":
            return SimpleNamespace()                     # no probability_in_range: fails after predict returned
        p = 0.90 if self.mode == "high" else 0.20
        return SimpleNamespace(probability_in_range=p, threshold_pct=5.0, horizon_days=14, direction=None, confidence=p)


@pytest.fixture
def run(synth_df, synth_features):
    def _run(mode: str | None, gates_mode: str):
        return Backtester(
            data=synth_df, strategies=["iron_condor"], initial_capital=100_000, macro_event_gate=False,
            allow_live_model_fallback=False, hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0,
            iv_rank_rise_threshold=10.0, ic_min_credit=0.0, ic_min_credit_width=0.0, iv_floor=0.01,
            features_cache=synth_features, gates_mode=gates_mode,
            range_predictor=None if mode is None else FakeRange(mode)).run()
    return _run


def _pnls(res) -> list:
    return [t["pnl"] for t in res.trades]


class TestEnforce:
    def test_a_raising_model_blocks_every_entry_like_an_abstaining_one(self, run):
        err, abstain = run("raise", "enforce"), run("abstain", "enforce")
        assert err.total_trades == 0
        assert err.entry_funnel["range_gate_error"] == abstain.entry_funnel["range_gate_below_threshold"] > 0
        assert err.entry_funnel.get("range_gate_below_threshold", 0) == 0

    def test_a_malformed_prediction_also_fails_closed(self, run):
        res = run("malformed", "enforce")
        assert res.total_trades == 0 and res.entry_funnel["range_gate_error"] > 0

    def test_the_other_outcomes_are_unchanged(self, run):
        low, abstain, high, nothing = (run("low", "enforce"), run("abstain", "enforce"), run("high", "enforce"),
                                       run(None, "enforce"))
        for blocked in (low, abstain):
            assert blocked.total_trades == 0
            assert blocked.entry_funnel["range_gate_below_threshold"] > 0
            assert "range_gate_error" not in blocked.entry_funnel
        assert high.total_trades > 0 and "range_gate_error" not in high.entry_funnel
        assert not any(k.startswith("range_gate") for k in high.entry_funnel)
        assert nothing.total_trades == high.total_trades            # no model and a floor below 1.0: gate absent by design


class TestObserve:
    def test_a_raising_model_is_a_would_reject_and_the_entry_goes_ahead(self, run):
        err, nothing = run("raise", "observe"), run(None, "observe")
        assert err.entry_funnel["would_reject:range_gate_error"] > 0
        assert "range_gate_error" not in err.entry_funnel           # observe never blocks
        assert err.total_trades > 0 and _pnls(err) == _pnls(nothing)

    def test_the_error_is_recorded_on_the_entry_decision(self, run):
        err = run("raise", "observe")
        gates = [t["decision"]["range_gate"] for t in err.trades if t.get("decision")]
        assert gates and all(g["pass"] is False and g["error"] == "RuntimeError" for g in gates)


class TestVisibility:
    def test_the_first_error_per_engine_is_logged_once_as_a_warning(self, run):
        with capture_logs() as events:
            res = run("raise", "enforce")
        mine = [e for e in events if e.get("event") == "range_gate_predict_failed"]
        assert res.entry_funnel["range_gate_error"] > 1             # many failing days ...
        warnings = [e for e in mine if e["log_level"] == "warning"]
        assert len(warnings) == 1 and "model blew up" in warnings[0]["error"]      # ... one warning, not one per day

    def test_the_new_reason_is_in_the_gate_registry(self):
        row = classify("range_gate_error")
        assert row is not None and row.kind == TWIN

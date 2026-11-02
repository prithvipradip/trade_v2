"""The research engine's meta-label gate fails OPEN on purpose, exactly as live: a meta-labeler that cannot answer vetoes nothing.

This pins the design so it cannot change silently (the range gate beside it is the opposite on purpose: no verdict from
the range model fails CLOSED, see test_range_gate_fails_closed.py). Only a verdict "do not take this trade" blocks an
entry, under the funnel name ``meta_label``. A real failure inside ``MetaLabeler.predict`` is logged where it happens
(``meta_label_predict_failed``) and returns no verdict; the engine's own handler only guards a malformed return value.

Real ``Backtester`` on synthetic data with a fake trained meta-labeler, the pattern of test_range_gate_fails_closed.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from ait.backtesting.engine import Backtester
from ait.ml.meta_label import MetaLabeler


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


class FakeMeta:
    """What the engine reads from a trained meta-labeler: ``is_trained`` and ``predict(context)``."""

    is_trained = True

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def predict(self, context):
        if self.mode == "raise":
            raise RuntimeError("meta-labeler blew up")
        if self.mode == "none":
            return None
        if self.mode == "malformed":
            return SimpleNamespace()                         # no take_trade: fails after predict returned
        take = self.mode == "take"
        return SimpleNamespace(take_trade=take, probability=0.9 if take else 0.1)


@pytest.fixture
def run(synth_df, synth_features):
    def _run(meta: str | None):
        return Backtester(
            data=synth_df, strategies=["iron_condor"], initial_capital=100_000, macro_event_gate=False,
            allow_live_model_fallback=False, hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0,
            iv_rank_rise_threshold=10.0, ic_min_credit=0.0, ic_min_credit_width=0.0, iv_floor=0.01,
            features_cache=synth_features, meta_labeler=None if meta is None else FakeMeta(meta)).run()
    return _run


def _pnls(res) -> list:
    return [t["pnl"] for t in res.trades]


class TestOnlyAnExplicitSkipBlocks:
    def test_the_baseline_trades(self, run):
        assert run(None).total_trades > 0

    @pytest.mark.parametrize("mode", ["raise", "none", "malformed", "take"])
    def test_an_error_no_verdict_or_a_take_verdict_changes_nothing(self, run, mode):
        base, res = run(None), run(mode)
        assert res.total_trades == base.total_trades and _pnls(res) == _pnls(base)
        assert "meta_label" not in res.entry_funnel

    def test_a_skip_verdict_blocks_every_entry(self, run):
        res = run("skip")
        assert res.total_trades == 0 and res.entry_funnel["meta_label"] > 0


class TestTheRealMetaLabelerLogsItsOwnFailure:
    def test_a_failure_inside_predict_is_a_warning_and_no_verdict(self):
        class _Broken:
            def transform(self, X):
                raise ValueError("scaler broke")

        m = MetaLabeler()
        m._trained = True
        m._feature_names = ["a", "b"]
        m._scaler = _Broken()
        with capture_logs() as events:
            assert m.predict({"a": 1.0}) is None
        warnings = [e for e in events if e.get("event") == "meta_label_predict_failed"]
        assert len(warnings) == 1 and warnings[0]["log_level"] == "warning" and "scaler broke" in warnings[0]["error"]

    def test_an_untrained_labeler_has_no_verdict(self):
        m = MetaLabeler()
        m._trained = False
        assert m.predict({"a": 1.0}) is None

"""Tests for Layer 2 dashboard enrichment.

Verifies that walk-forward window JSON files contain the new fields added for
the dashboard: enriched trades_detail (legs, decision, features_at_entry) and
Optuna trial history.

The WalkForwardBacktester is expensive to run, so two module-scoped fixtures
share a single run across all tests that need the same configuration:
  - wf_plain:  no per-window Optuna (fast ablation path)
  - wf_optuna: optimize_per_window=True, n_trials=3

Skip unless RUN_DASHBOARD_TESTS=1 env var is set (full walk-forward run can
take several minutes).

Run:
    RUN_DASHBOARD_TESTS=1 pytest tests/test_walkforward_dashboard.py -v
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.walkforward import (
    WalkForwardBacktester,
    WalkForwardConfig,
    _build_optuna_window_data,
    _isnan,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_DASHBOARD_TESTS"),
    reason="set RUN_DASHBOARD_TESTS=1 to run (full walk-forward run, several minutes)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(days: int = 200, start_price: float = 450.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    ret = rng.normal(0.0005, 0.012, days)
    close = start_price * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.005, days)))
    low  = close * (1 - np.abs(rng.normal(0, 0.005, days)))
    open_ = close * (1 + rng.normal(0, 0.002, days))
    vol  = rng.integers(20_000_000, 60_000_000, days)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def _run_wf(tmp_path: Path, optimize: bool = False, n_trials: int = 3) -> list[dict]:
    """Run a minimal walk-forward and return parsed window JSONs."""
    import asyncio
    df = _make_ohlcv()
    cfg = WalkForwardConfig(
        train_days=120,
        test_days=20,
        step_days=20,
        gap_days=2,
        initial_capital=50_000.0,
        optimize_per_window=optimize,
        optimize_n_trials=n_trials,
        optimize_patience=0,
        optimize_seed=0,
    )
    bt = WalkForwardBacktester(
        symbols=["SIM"],
        strategies=["iron_condor"],
        config=cfg,
        progress_dir=tmp_path,
    )
    asyncio.run(bt.run(data={"SIM": df}))

    windows = []
    for p in sorted(tmp_path.glob("window_*.json")):
        windows.append(json.loads(p.read_text()))
    return windows


# ---------------------------------------------------------------------------
# Module-scoped fixtures — one real WalkForwardBacktester run per config,
# shared across all tests that use the same configuration.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def wf_plain(tmp_path_factory):
    """Non-optimized walk-forward run, shared by all tests that don't need Optuna."""
    return _run_wf(tmp_path_factory.mktemp("wf_plain"))


@pytest.fixture(scope="module")
def wf_optuna(tmp_path_factory):
    """Optimized walk-forward run (3 trials), shared by all Optuna-related tests."""
    return _run_wf(tmp_path_factory.mktemp("wf_optuna"), optimize=True, n_trials=3)


# ---------------------------------------------------------------------------
# Layer 2a — enriched trades_detail
# ---------------------------------------------------------------------------

class TestTradesDetailEnrichment:
    def test_hold_days_in_trades_detail(self, wf_plain):
        trades = [t for w in wf_plain for t in w.get("trades_detail", [])]
        if not trades:
            pytest.skip("no trades generated with this random seed — adjust _make_ohlcv")
        for t in trades:
            assert "hold_days" in t
            assert isinstance(t["hold_days"], int)
            assert t["hold_days"] >= 0

    def test_contracts_in_trades_detail(self, wf_plain):
        trades = [t for w in wf_plain for t in w.get("trades_detail", [])]
        if not trades:
            pytest.skip("no trades")
        for t in trades:
            assert "contracts" in t

    def test_n_legs_in_trades_detail(self, wf_plain):
        trades = [t for w in wf_plain for t in w.get("trades_detail", [])]
        if not trades:
            pytest.skip("no trades")
        for t in trades:
            assert "n_legs" in t

    def test_entry_iv_rank_in_trades_detail(self, wf_plain):
        trades = [t for w in wf_plain for t in w.get("trades_detail", [])]
        if not trades:
            pytest.skip("no trades")
        for t in trades:
            assert "entry_iv_rank" in t

    def test_entry_vix_level_in_trades_detail(self, wf_plain):
        trades = [t for w in wf_plain for t in w.get("trades_detail", [])]
        if not trades:
            pytest.skip("no trades")
        for t in trades:
            assert "entry_vix_level" in t

    def test_legs_field_present(self, wf_plain):
        trades = [t for w in wf_plain for t in w.get("trades_detail", [])]
        if not trades:
            pytest.skip("no trades")
        for t in trades:
            assert "legs" in t
            assert isinstance(t["legs"], list)

    def test_iron_condor_has_four_legs(self, wf_plain):
        ic_trades = [
            t for w in wf_plain for t in w.get("trades_detail", [])
            if t.get("strategy") == "iron_condor" and t.get("legs")
        ]
        if not ic_trades:
            pytest.skip("no iron condor trades with legs")
        for t in ic_trades:
            assert len(t["legs"]) == 4, f"Expected 4 legs, got {len(t['legs'])}"
            leg_types = {leg["type"] for leg in t["legs"]}
            assert leg_types == {"short_put", "long_put", "short_call", "long_call"}

    def test_decision_field_present(self, wf_plain):
        trades = [t for w in wf_plain for t in w.get("trades_detail", [])]
        if not trades:
            pytest.skip("no trades")
        for t in trades:
            assert "decision" in t
            assert isinstance(t["decision"], dict)

    def test_decision_has_required_keys(self, wf_plain):
        trades = [
            t for w in wf_plain for t in w.get("trades_detail", [])
            if t.get("decision")
        ]
        if not trades:
            pytest.skip("no trades with decision")
        required = {"direction_class", "range_gate", "vol_gate", "meta_label",
                    "fractal_gate", "regime"}
        for t in trades:
            assert required.issubset(t["decision"].keys()), (
                f"Missing decision keys: {required - t['decision'].keys()}"
            )

    def test_decision_range_gate_has_subkeys(self, wf_plain):
        trades = [
            t for w in wf_plain for t in w.get("trades_detail", [])
            if t.get("decision", {}).get("range_gate")
        ]
        if not trades:
            pytest.skip("no trades with range_gate")
        for t in trades:
            rg = t["decision"]["range_gate"]
            assert "threshold" in rg
            assert "pass" in rg

    def test_features_at_entry_present(self, wf_plain):
        trades = [t for w in wf_plain for t in w.get("trades_detail", [])]
        if not trades:
            pytest.skip("no trades")
        for t in trades:
            assert "features_at_entry" in t
            assert isinstance(t["features_at_entry"], dict)

    def test_features_at_entry_has_rsi(self, wf_plain):
        trades = [
            t for w in wf_plain for t in w.get("trades_detail", [])
            if t.get("features_at_entry")
        ]
        if not trades:
            pytest.skip("no trades with features_at_entry")
        for t in trades:
            assert "rsi_14" in t["features_at_entry"]

    def test_features_at_entry_has_hurst(self, wf_plain):
        trades = [
            t for w in wf_plain for t in w.get("trades_detail", [])
            if t.get("features_at_entry")
        ]
        if not trades:
            pytest.skip("no trades with features_at_entry")
        for t in trades:
            assert "hurst_wavelet" in t["features_at_entry"]

    def test_return_pct_none_without_layer2a_max_loss(self, wf_plain):
        # return_pct requires max_loss which is now stored — if max_loss is present
        # and nonzero, return_pct should be computed; if max_loss is None, it's None.
        trades = [t for w in wf_plain for t in w.get("trades_detail", [])]
        if not trades:
            pytest.skip("no trades")
        for t in trades:
            assert "return_pct" in t
            if t.get("max_loss") is not None and t["max_loss"] != 0:
                assert t["return_pct"] is not None
            else:
                assert t["return_pct"] is None


# ---------------------------------------------------------------------------
# Layer 2b — Optuna trial history
# ---------------------------------------------------------------------------

class TestOptunaTrialExport:
    def test_optuna_trials_present_when_optimizing(self, wf_optuna):
        active = [w for w in wf_optuna if w.get("trades", 0) > 0 or w.get("best_params")]
        if not active:
            pytest.skip("no active windows")
        for w in active:
            assert "optuna_trials" in w, f"window {w['window']} missing optuna_trials"

    def test_optuna_trials_absent_when_not_optimizing(self, wf_plain):
        for w in wf_plain:
            assert "optuna_trials" not in w

    def test_optuna_meta_present_when_optimizing(self, wf_optuna):
        active = [w for w in wf_optuna if "optuna_trials" in w]
        if not active:
            pytest.skip("no optimized windows")
        for w in active:
            assert "optuna_meta" in w

    def test_optuna_meta_has_status(self, wf_optuna):
        active = [w for w in wf_optuna if w.get("optuna_meta")]
        if not active:
            pytest.skip("no optuna_meta")
        for w in active:
            assert w["optuna_meta"]["status"] in ("completed", "early_stopped")

    def test_optuna_trial_schema(self, wf_optuna):
        all_trials = [t for w in wf_optuna for t in w.get("optuna_trials", [])]
        if not all_trials:
            pytest.skip("no trials")
        required = {"number", "state", "value", "params"}
        for t in all_trials:
            assert required.issubset(t.keys()), f"Missing trial keys: {required - t.keys()}"

    def test_optuna_trial_user_attrs_present(self, wf_optuna):
        complete = [
            t for w in wf_optuna for t in w.get("optuna_trials", [])
            if t.get("state") == "COMPLETE"
        ]
        if not complete:
            pytest.skip("no COMPLETE trials")
        for t in complete:
            assert "n_trades" in t, "n_trades missing from complete trial"


# ---------------------------------------------------------------------------
# Layer 2b — _build_optuna_window_data helper (unit)
# ---------------------------------------------------------------------------

class TestBuildOptunaWindowData:
    def _stub_result(self, n_complete: int = 3, n_pruned: int = 1, early_stopped: bool = False):
        """Build a minimal fake OptimizationResult-like object."""
        import optuna
        study = optuna.create_study(direction="maximize")

        class _FakeTrial:
            def __init__(self, number, value, state_name):
                self.number = number
                self.value = value
                self.params = {"iron_condor__stop_loss_pct": 0.4 + number * 0.01}
                self.user_attrs = {
                    "sharpe": 1.2,
                    "win_rate": 0.6,
                    "max_drawdown": 0.05,
                    "n_trades": 8,
                }
                self.datetime_start = None
                self.datetime_complete = None

                class _State:
                    name = state_name
                self.state = _State()

        class _FakeStudy:
            def __init__(self):
                self.trials = (
                    [_FakeTrial(i, 0.3 + i * 0.05, "COMPLETE") for i in range(n_complete)]
                    + [_FakeTrial(n_complete + i, None, "PRUNED") for i in range(n_pruned)]
                )

        class _FakeResult:
            def __init__(self):
                self.study = _FakeStudy()
                self.early_stopped = early_stopped
                self.stop_reason = "Early-stopped: patience reached at trial 3." if early_stopped else ""

        return _FakeResult()

    def test_trials_flattened(self):
        res = self._stub_result(n_complete=3, n_pruned=1)
        out = _build_optuna_window_data([res], ["iron_condor"], 1, "QQQ")
        assert len(out["trials"]) == 4

    def test_complete_and_pruned_states(self):
        res = self._stub_result(n_complete=3, n_pruned=1)
        out = _build_optuna_window_data([res], ["iron_condor"], 1, "QQQ")
        states = {t["state"] for t in out["trials"]}
        assert "COMPLETE" in states
        assert "PRUNED" in states

    def test_meta_status_completed(self):
        res = self._stub_result(early_stopped=False)
        out = _build_optuna_window_data([res], ["iron_condor"], 1, "QQQ")
        assert out["meta"]["status"] == "completed"

    def test_meta_status_early_stopped(self):
        res = self._stub_result(early_stopped=True)
        out = _build_optuna_window_data([res], ["iron_condor"], 1, "QQQ")
        assert out["meta"]["status"] == "early_stopped"

    def test_stop_reason_populated(self):
        res = self._stub_result(early_stopped=True)
        out = _build_optuna_window_data([res], ["iron_condor"], 1, "QQQ")
        assert len(out["meta"]["stop_reason"]) > 0

    def test_trial_offset_for_multiple_strategies(self):
        r1 = self._stub_result(n_complete=2, n_pruned=0)
        r2 = self._stub_result(n_complete=2, n_pruned=0)
        out = _build_optuna_window_data([r1, r2], ["iron_condor", "put_credit_spread"], 1, "QQQ")
        numbers = [t["number"] for t in out["trials"]]
        assert len(numbers) == len(set(numbers)), "trial numbers must be unique across strategies"

    def test_empty_results_list(self):
        out = _build_optuna_window_data([], ["iron_condor"], 1, "QQQ")
        assert out["trials"] == []
        assert "meta" in out


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class TestIsNan:
    def test_nan_float(self):
        import math
        assert _isnan(math.nan) is True

    def test_normal_float(self):
        assert _isnan(1.5) is False

    def test_none(self):
        assert _isnan(None) is False

    def test_string(self):
        assert _isnan("abc") is False

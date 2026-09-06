"""R21b — PR#7 review round of 2026-08-25 (4 findings, all fallout from the
entry_dte/max_hold_days decoupling), each verified against source before
fixing:

1. param_spaces: 7 non-IC spaces still searched max_hold_days from 14 up —
   unreachable (debit expires at entry_dte=14, credit closes at DTE<=5,
   ~day 9), so those Optuna dimensions were flat noise.
2. optimizer: explicit walkforward overrides for stop_loss_pct /
   profit_target_pct / min_confidence never reached trial baselines —
   trials trained on YAML values, OOS evaluated on caller values.
3. walkforward: range model trained/evaluated at horizon max_hold_days (30)
   while reachable IC holds cap at ~9 days.
4. run_backtest parity manifest still reported the retired fixed-DTE-21
   convention.

Every test EXECUTES the real resolution path — no source-string assertions.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from ait.config.settings import load_settings
from ait.execution.exit_policy import EXPIRY_APPROACHING_DTE
from ait.optimization.param_spaces import STRATEGY_SPACES
from ait.optimization.optimizer import StrategyOptimizer
from ait.backtesting.walkforward import WalkForwardBacktester
from ait.strategies.base import CREDIT_STRATEGIES


def _entry_dte() -> int:
    return int(load_settings().options.dte_range[0])


# ---------------------------------------------------------------- finding 1
class TestHoldDimensionReachable:
    def test_every_space_hold_range_is_reachable(self):
        entry_dte = _entry_dte()
        credit_horizon = entry_dte - int(EXPIRY_APPROACHING_DTE)
        for strategy, space in STRATEGY_SPACES.items():
            if "max_hold_days" not in space:
                continue
            kind, lo, hi = space["max_hold_days"][:3]
            assert kind == "int"
            limit = (credit_horizon if strategy in CREDIT_STRATEGIES
                     else entry_dte)
            # pre-fix: e.g. long_call searched 14-60 against a 14-day expiry,
            # short_strangle 14-40 against a ~9-day close horizon.
            assert hi <= limit, (
                f"{strategy}: max_hold_days upper bound {hi} exceeds its "
                f"reachable horizon {limit} — dimension is flat noise above it")
            assert 1 <= lo < hi


# ---------------------------------------------------------------- finding 2
class TestOptimizerBaselineParity:
    def _opt(self, **kw):
        return StrategyOptimizer(
            symbols=["SPY"], strategies=["iron_condor"], n_trials=1, **kw)

    def test_explicit_overrides_win(self):
        # pre-fix: TypeError (params did not exist), and bt_kwargs read the
        # YAML baselines regardless of what the caller configured.
        opt = self._opt(stop_loss_pct=0.42, profit_target_pct=0.61,
                        min_confidence=0.63)
        assert opt._stop_loss_pct == 0.42
        assert opt._profit_target_pct == 0.61
        assert opt._min_confidence == 0.63

    def test_defaults_resolve_from_config(self):
        s = load_settings()
        opt = self._opt()
        assert opt._stop_loss_pct == float(s.backtest.stop_loss_pct)
        assert opt._profit_target_pct == float(s.backtest.profit_target_pct)
        assert opt._min_confidence == float(s.risk.min_confidence)


# ---------------------------------------------------------------- finding 3
class TestRangeLabelHorizon:
    def _wf(self, max_hold_days=30, settings="load"):
        wf = WalkForwardBacktester.__new__(WalkForwardBacktester)
        wf._settings = load_settings() if settings == "load" else settings
        wf._config = SimpleNamespace(max_hold_days=max_hold_days)
        return wf

    def test_horizon_is_reachable_not_baseline(self):
        # pre-fix: 30 (the max_hold_days baseline) was passed everywhere.
        expected = min(30, _entry_dte() - int(EXPIRY_APPROACHING_DTE))
        assert self._wf()._range_label_horizon() == expected
        assert self._wf()._range_label_horizon() < 30

    def test_max_hold_days_still_caps(self):
        assert self._wf(max_hold_days=4)._range_label_horizon() == 4

    def test_no_settings_degrades_to_field_defaults(self):
        from ait.config.settings import OptionsConfig
        expected = min(
            30, int(OptionsConfig().dte_range[0]) - int(EXPIRY_APPROACHING_DTE))
        assert self._wf(settings=None)._range_label_horizon() == expected


# ---------------------------------------------------------------- finding 4
class TestParityManifestDte:
    def test_manifest_reports_engine_resolution(self):
        import run_backtest as rb
        old_argv = sys.argv
        sys.argv = ["run_backtest.py"]
        try:
            manifest = rb.build_parity_manifest(rb.parse_args())
        finally:
            sys.argv = old_argv
        entry_dte = _entry_dte()
        # pre-fix: hardcoded [21, 21] ("engine uses fixed DTE =
        # max_hold_days" — a convention the engine no longer has).
        assert manifest["backtest"]["dte_band"] == [entry_dte, entry_dte]

    def test_manifest_matches_actual_engine(self):
        import pandas as pd
        from ait.backtesting.engine import Backtester
        bt = Backtester(
            data=pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]),
            strategies=["iron_condor"])
        import run_backtest as rb
        old_argv = sys.argv
        sys.argv = ["run_backtest.py"]
        try:
            manifest = rb.build_parity_manifest(rb.parse_args())
        finally:
            sys.argv = old_argv
        assert manifest["backtest"]["dte_band"] == [bt._entry_dte] * 2

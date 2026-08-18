"""R16 core-cluster fixes — each test fails against the pre-fix code.

Covers: condor thesis-check exemption (the VIX-spike flatten that undid the
hold-through decision), sector-cluster correlation floor, identity-based
is_defined_risk, sweep CAS narrowing, shared runtime env contract, and the
exit-pricing mark anchor.
"""
from __future__ import annotations

import inspect
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager, TradeStatus
from ait.risk.correlation import CorrelationGuard
from ait.strategies.base import UNDEFINED_RISK_STRATEGIES


class TestCondorThesisExempt:
    async def test_condor_never_thesis_flattened(self):
        # R16: any VIX print / direction flip must NOT flatten a defined-risk
        # neutral structure — the last path that undid hold-through-events.
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        pos = SimpleNamespace(trade_id="T-C", symbol="QQQ",
                              strategy="iron_condor")
        invalidated, reason = await orch._check_thesis_valid(pos)
        assert invalidated is False and reason == ""


class TestSectorClusterFloor:
    def test_index_etfs_always_cluster(self):
        assert CorrelationGuard._same_sector_cluster("SPY", "IWM")
        assert CorrelationGuard._same_sector_cluster("QQQ", "SPY")
        assert not CorrelationGuard._same_sector_cluster("SPY", "GLD")

    def test_third_index_blocked_even_when_measured_corr_dips(self):
        g = CorrelationGuard.__new__(CorrelationGuard)
        g._max_corr = 0.75
        g._max_correlated = 2
        g._get_correlation = lambda a, b: 0.70  # the 0.76->0.70 dip scenario
        allowed, reason = g.check_correlation("IWM", ["QQQ", "SPY"])
        assert allowed is False


class TestDefinedRiskIdentity:
    def test_strangle_estimate_is_not_defined_risk(self):
        assert "short_strangle" in UNDEFINED_RISK_STRATEGIES
        from ait.strategies.base import Signal
        kwargs = {}
        for f in __import__("dataclasses").fields(Signal):
            if (f.default is __import__("dataclasses").MISSING
                    and f.default_factory is __import__("dataclasses").MISSING):
                kwargs[f.name] = f.type and None
        # build minimally: only fields without defaults need values
        sig = SimpleNamespace(strategy_name="short_strangle", max_loss=500.0)
        assert Signal.is_defined_risk.fget(sig) is False
        sig2 = SimpleNamespace(strategy_name="iron_condor", max_loss=500.0)
        assert Signal.is_defined_risk.fget(sig2) is True


class TestSweepCasNarrowed:
    def test_close_trade_accepts_from_statuses(self):
        assert "from_statuses" in inspect.signature(
            StateManager.close_trade).parameters

    def test_sweep_passes_pending_only(self):
        from ait.execution import reconciler
        src = inspect.getsource(reconciler.PositionReconciler._sweep_stale_pending)
        assert "from_statuses=(TradeStatus.PENDING,)" in src
        # and the sweep must consult working orders before booking fiction
        assert "_cancel_working_entry_order" in src


class TestRuntimeEnvContract:
    def test_undefined_risk_gate_defaults_closed(self, monkeypatch):
        for k in ("AIT_ALLOW_UNDEFINED_RISK", "AIT_IC_WING_K",
                  "AIT_IC_MIN_CREDIT_WIDTH", "AIT_SKIP_MACRO_EVENTS"):
            monkeypatch.delenv(k, raising=False)
        from ait.config.runtime_env import apply_runtime_env_defaults
        apply_runtime_env_defaults()
        import os
        assert os.environ["AIT_ALLOW_UNDEFINED_RISK"] == "0"
        assert os.environ["AIT_IC_WING_K"] == "1.6"
        assert os.environ["AIT_IC_MIN_CREDIT_WIDTH"] == "0.10"
        assert os.environ["AIT_SKIP_MACRO_EVENTS"] == "1"

    def test_bare_main_applies_contract(self):
        import ait.main as m
        src = inspect.getsource(m)
        assert "apply_runtime_env_defaults()" in src


class TestExitMarkAnchor:
    def test_marked_cost_math(self):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        state = MagicMock()
        state.get_position_mark.return_value = {
            "unrealized_pnl": 340.0, "pnl_pct": 0.53,
            "mark_time": datetime.now().isoformat()}
        orch._state = state
        t = SimpleNamespace(trade_id="T-M", entry_price=6.44, quantity=1)
        assert orch._marked_cost_to_close(t) == pytest.approx(3.04)

    def test_stale_mark_returns_none(self):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        state = MagicMock()
        state.get_position_mark.return_value = {
            "unrealized_pnl": 340.0, "pnl_pct": 0.53,
            "mark_time": "2026-08-01T10:00:00"}
        orch._state = state
        t = SimpleNamespace(trade_id="T-M", entry_price=6.44, quantity=1)
        assert orch._marked_cost_to_close(t) is None

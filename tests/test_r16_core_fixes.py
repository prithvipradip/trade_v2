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

    def test_sweep_passes_pending_only(self, tmp_path):
        """CENSUS-4 (2026-09-27): three real behaviors, none previously
        covered -- every existing sweep test (test_w26_*, test_r19_state_ml_
        fixes.py) stubs _cancel_working_entry_order=False and never checks
        from_statuses at all."""
        import json
        from datetime import timedelta
        from unittest.mock import MagicMock

        from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
        from ait.execution.reconciler import NEVER_FILLED_EXIT_REASON, PositionReconciler
        from ait.utils.time import now_et
        from tests.test_w26_budget_and_cluster_cooldown import (
            CONDOR_LEGS, _pending_trade, _reconciler, _state_with,
        )

        # (a) THE from_statuses NARROWING, against a REAL StateManager --
        # not a MagicMock told what to return, the genuine CAS SQL. A trade
        # that races to FILLED between the sweep's snapshot and its close
        # attempt must survive as FILLED, not get force-closed to fiction.
        db = tmp_path / "state.db"
        state = StateManager(db_path=db)
        stale_entry = (now_et() - timedelta(minutes=40)).isoformat()
        state.record_trade(TradeRecord(
            trade_id="T-RACE", symbol="SPY", strategy="iron_condor",
            direction=TradeDirection.SHORT, status=TradeStatus.PENDING,
            entry_time=stale_entry, entry_price=1.50, quantity=1,
            contract_type="iron_condor", legs=json.dumps(CONDOR_LEGS)))
        # The race: a concurrent writer promotes it to FILLED before the
        # sweep's own close_trade CAS attempt runs.
        assert state.transition("T-RACE", (TradeStatus.PENDING,), TradeStatus.FILLED)

        rec = PositionReconciler.__new__(PositionReconciler)
        rec._state = state
        rec._ibkr = MagicMock()
        rec._live_ibkr_leg_keys = lambda: {("XXX", "P", 1.0, "20261218")}
        rec._cancel_working_entry_order = lambda tr: False
        rec._refund_entry_budget = lambda tr: None
        # get_open_trades() would no longer return T-RACE (it's FILLED, not
        # PENDING) -- simulate the exact race by handing the sweep a stale
        # PENDING snapshot directly, same technique reconciler.py's own
        # comment describes ("a concurrent writer may have moved the trade
        # since get_open_trades()").
        stale_snapshot = TradeRecord(
            trade_id="T-RACE", symbol="SPY", strategy="iron_condor",
            direction=TradeDirection.SHORT, status=TradeStatus.PENDING,
            entry_time=stale_entry, entry_price=1.50, quantity=1,
            contract_type="iron_condor", legs=json.dumps(CONDOR_LEGS))
        state.get_open_trades = lambda: [stale_snapshot]

        booked = rec._sweep_stale_pending()
        assert booked == 0, "a row that raced to FILLED must not be counted as closed"
        after = state.get_trade_by_id("T-RACE")
        assert after.status == TradeStatus.FILLED, (
            f"from_statuses=(PENDING,) must refuse the CAS once the row is "
            f"no longer PENDING -- got {after.status}"
        )

        # (b) A WORKING entry order must be deferred, not booked as fiction.
        today = now_et().date()
        trade_b = _pending_trade(trade_id="T-WORKING")
        state_b = _state_with(trade_b, {today: 1})
        rec_b = _reconciler(state_b)
        rec_b._cancel_working_entry_order = lambda tr: True
        assert rec_b._sweep_stale_pending() == 0
        state_b.close_trade.assert_not_called()

        # (c) Control: a genuinely stale PENDING trade with no working order
        # still closes normally -- proves (a)/(b) didn't just disable the
        # sweep entirely.
        trade_c = _pending_trade(trade_id="T-STALE")
        state_c = _state_with(trade_c, {today: 1})
        rec_c = _reconciler(state_c)
        assert rec_c._sweep_stale_pending() == 1
        assert state_c.close_trade.call_args.kwargs["exit_reason_detailed"] == (
            NEVER_FILLED_EXIT_REASON
        )
        assert state_c.close_trade.call_args.kwargs["from_statuses"] == (
            TradeStatus.PENDING,
        )


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
        # R19c: config.yaml now seeds this (yaml 0.10 -> "0.1"); compare
        # numerically — the SEMANTIC contract is what matters.
        assert float(os.environ["AIT_IC_MIN_CREDIT_WIDTH"]) == 0.10
        assert os.environ["AIT_SKIP_MACRO_EVENTS"] == "1"

    def test_bare_main_applies_contract(self):
        """CENSUS-3 (2026-09-28): apply_runtime_env_defaults() is called at
        MODULE level in main.py (not inside main()/if __name__), so the
        honest proof is a bare `import ait.main` in a clean subprocess with
        the contract keys stripped from its env first -- not grepping
        main.py's source text for the literal call."""
        import json
        import os
        import subprocess
        import sys
        import textwrap
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        strip = ("KMP_DUPLICATE_LIB_OK", "OMP_NUM_THREADS",
                 "AIT_ALLOW_UNDEFINED_RISK", "AIT_SKIP_MACRO_EVENTS",
                 "AIT_IC_WING_K")
        env = {k: v for k, v in os.environ.items() if k not in strip}
        env["PYTHONPATH"] = str(repo_root / "src")
        code = textwrap.dedent(f"""
            import os
            import ait.main  # noqa: F401 -- the bare import IS the claim
            import json
            print(json.dumps({{k: os.environ.get(k) for k in {strip!r}}}))
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=repo_root, env=env,
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        got = json.loads(result.stdout.strip().splitlines()[-1])
        assert got == {
            "KMP_DUPLICATE_LIB_OK": "TRUE",
            "OMP_NUM_THREADS": "1",
            "AIT_ALLOW_UNDEFINED_RISK": "0",
            "AIT_SKIP_MACRO_EVENTS": "1",
            "AIT_IC_WING_K": "1.6",
        }, (
            f"bare `import ait.main` did not apply the runtime env contract: "
            f"{got}"
        )


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

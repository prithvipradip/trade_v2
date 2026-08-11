"""R17 minor fixes — external review findings, each test fails against the
pre-fix code where a behavior change was made:

  - counterfactual.py record_skip stopped recording repeats of the same
    (symbol, strategy, reason) forever, because the only code that reset
    the dedup flag (evaluate_outcomes) was deleted in an earlier round.
    Now dedups per-day instead.
  - main.py hard-cancelled orchestrator.run() on shutdown, bypassing
    _shutdown()'s notify-drain entirely. Now stops gracefully first, with
    a bounded grace window before falling back to cancel.

(strategies/base.py + selector.py's risk_budget mutation and the Sortino
inf/None inconsistency in monitoring/analytics.py + duckdb_analytics.py
are documentation-only in this round — no behavior change, no new tests.)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest


class TestCounterfactualDedupResetsDaily:
    def _tracker(self, tmp_path, monkeypatch):
        import ait.learning.counterfactual as mod
        monkeypatch.setattr(mod, "STATE_FILE", tmp_path / "counterfactual.json")
        return mod.CounterfactualTracker()

    def test_same_combo_recorded_again_the_next_day(self, tmp_path, monkeypatch):
        tracker = self._tracker(tmp_path, monkeypatch)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        from ait.learning.counterfactual import SkippedTrade
        tracker._skipped.append(SkippedTrade(
            timestamp=yesterday, symbol="SPY", strategy="iron_condor",
            direction="neutral", confidence=0.7, entry_price=450.0,
            reject_reason="risk_limit",
        ))
        # Pre-fix: outcome_checked never becomes True (evaluate_outcomes is
        # gone), so this combo would be silently dropped forever.
        tracker.record_skip("SPY", "iron_condor", "neutral", 0.7, 450.0, "risk_limit")
        assert tracker.total_count == 2

    def test_same_day_repeats_still_deduped(self, tmp_path, monkeypatch):
        tracker = self._tracker(tmp_path, monkeypatch)
        for _ in range(5):
            tracker.record_skip("SPY", "iron_condor", "neutral", 0.7, 450.0, "risk_limit")
        assert tracker.total_count == 1


class TestGracefulShutdownSequencing:
    async def test_stop_awaited_before_cancel_on_fast_exit(self):
        from ait.main import _graceful_shutdown_bot_task

        calls: list[str] = []

        class _FakeOrch:
            async def stop(self):
                calls.append("stop")

        async def _fast_run():
            await asyncio.sleep(0.01)  # exits well within the grace window

        task = asyncio.create_task(_fast_run())
        await _graceful_shutdown_bot_task(_FakeOrch(), task)

        assert calls == ["stop"]
        assert task.done()
        assert not task.cancelled()

    async def test_cancels_after_timeout_if_still_running(self, monkeypatch):
        import ait.main as mod
        monkeypatch.setattr(mod, "GRACEFUL_SHUTDOWN_TIMEOUT", 0.05)

        calls: list[str] = []

        class _FakeOrch:
            async def stop(self):
                calls.append("stop")

        async def _slow_run():
            await asyncio.sleep(10)  # never exits on its own within the window

        task = asyncio.create_task(_slow_run())
        await mod._graceful_shutdown_bot_task(_FakeOrch(), task)

        assert calls == ["stop"]
        assert task.cancelled()

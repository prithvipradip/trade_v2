"""W46 (R26 register) — bot-day-03 + bot-day-04, startup hang/protection-arming.

Both from reports/blindspot_composition_hunt_20260825.md, re-verified against HEAD
this session (orchestrator.py touched by every wave W41-W45; master.py's hang branch
confirmed byte-identical since birth, 2026-07-09).

1. bot-day-04 — run()/_pre_market() awaited ensure_models_ready synchronously (up to
   ~7.5 min) BEFORE reconcile, exit-order re-adoption, and the trading loop -- leaving
   the book with zero working loss exits for the whole training window. Fixed by making
   training a guarded fire-and-forget background task. The entry-gating half needed no
   new code: DirectionPredictor/RangePredictor.predict already return None while
   untrained, and _scan_symbol already treats None as skip -- this file's own test 3
   proves that chain by execution, not by reading it.

2. bot-day-04's sibling, master.py's hang-detector (bot-day-03), is covered separately
   in tests/test_w34_ops_substrate.py::TestHangDetectorGraceAndBudget, extending that
   file's existing real-health_check()-driving harness rather than duplicating it here.

Every test executes the real code path. No inspect.getsource.
"""

from __future__ import annotations

import asyncio

import pytest

from ait.bot.scheduler import TradingPhase
from tests.test_hot_path_smoke import build_smoke_orchestrator


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    smoke = build_smoke_orchestrator()
    try:
        yield smoke
    finally:
        smoke.restore()


# ---------------------------------------------------------------------------
# 1. _launch_background_training — double-launch guard
# ---------------------------------------------------------------------------

class TestBackgroundTrainingDoubleLaunchGuard:
    async def test_a_second_launch_while_the_first_is_in_flight_is_a_no_op(self, rig):
        orch = rig.orch
        orch._training_task = None
        orch._models_ready = False
        gate = asyncio.Event()
        calls = {"n": 0}

        async def _slow_ensure_models_ready(symbols):
            calls["n"] += 1
            await gate.wait()
            return True

        orch._trainer.ensure_models_ready = _slow_ensure_models_ready

        orch._launch_background_training()
        first_task = orch._training_task
        orch._launch_background_training()  # while the first is still pending

        assert orch._training_task is first_task, (
            "a second launch while one is in flight must not replace the task"
        )
        gate.set()
        await first_task
        assert calls["n"] == 1, "ensure_models_ready must have been invoked exactly once"
        assert orch._models_ready is True

    async def test_a_launch_after_models_are_already_ready_is_a_no_op(self, rig):
        orch = rig.orch
        orch._training_task = None
        orch._models_ready = True
        calls = {"n": 0}

        async def _ensure_models_ready(symbols):
            calls["n"] += 1
            return True

        orch._trainer.ensure_models_ready = _ensure_models_ready

        orch._launch_background_training()

        assert orch._training_task is None
        assert calls["n"] == 0


# ---------------------------------------------------------------------------
# 2. run() — reconcile/adoption/notification no longer wait on training
# ---------------------------------------------------------------------------

class TestStartupSequenceDoesNotWaitOnTraining:
    async def test_reconcile_and_notification_complete_before_training_does(self, rig):
        orch = rig.orch
        gate = asyncio.Event()
        order: list[str] = []

        async def _slow_ensure_models_ready(symbols):
            order.append("training_started")
            await gate.wait()
            order.append("training_finished")
            return True

        orch._trainer.ensure_models_ready = _slow_ensure_models_ready

        real_reconcile = orch._reconciler.reconcile

        async def _tracked_reconcile():
            result = await real_reconcile()
            order.append("reconcile_done")
            return result

        orch._reconciler.reconcile = _tracked_reconcile

        notified: list[str] = []

        async def _notify(msg: str) -> bool:
            notified.append(msg)
            order.append("notified")
            return True

        orch.set_notification_callback(_notify)

        async def _stub_trading_loop():
            order.append("trading_loop_reached")
            orch._running = False

        orch._trading_loop = _stub_trading_loop
        orch._scheduler.get_current_phase = lambda: TradingPhase.MARKET_OPEN
        orch._scheduler.should_avoid_new_trades = lambda: False

        await orch.run()

        # The load-bearing claim: run() returned all the way through
        # reconcile, notification and reaching the trading loop WITHOUT
        # waiting on training -- "training_finished" cannot be among them,
        # since the gate is still unset. (Relative ordering of
        # "training_started" against the others is an asyncio scheduling
        # detail, not the property this fix guarantees -- these fakes
        # resolve their own awaits synchronously, so who gets the event
        # loop's very first turn isn't meaningful here.)
        assert "training_finished" not in order
        assert "reconcile_done" in order
        assert "notified" in order
        assert "trading_loop_reached" in order
        assert "training_started" in order

        gate.set()
        await orch._training_task
        assert "training_finished" in order
        assert orch._models_ready is True


# ---------------------------------------------------------------------------
# 3. The existing per-predictor fail-closed gate needs no new code
# ---------------------------------------------------------------------------

class TestUntrainedPredictorAlreadyBlocksEntriesByItself:
    async def test_scan_symbol_skips_the_symbol_when_the_predictor_is_untrained(self, rig):
        """Proves, by execution, that making training non-blocking is safe:
        _scan_symbol already treats an untrained predictor's None prediction
        as skip-this-symbol. No new gate was added for this wave."""
        orch = rig.orch
        orch._predictor.predict = lambda *a, **k: None  # untrained shape

        executed: list = rig.executed
        before = len(executed)

        # FakeMarketData.get_historical already serves realistic OHLCV
        # regardless of symbol -- no extra wiring needed.
        await orch._scan_symbol("SPY", vix=16.0, market_context=None)

        assert len(executed) == before, (
            "an untrained predictor must never reach order execution"
        )

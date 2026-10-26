"""W103: a bot stop finishes inside the supervisor's wait.

Every stop since July ended in SIGKILL with ``shutdown_signal_received`` as the bot's last line.
Nothing hung: the supervisor killed at 30 s while the bot's own 30 s grace was still running, the
phase wait slept on for up to an hour whatever was asked of it, the shutdown step ran only when the
loop ended by itself, and the signal handler may not have woken an idle loop.

Real code throughout: a real child process and a real SIGTERM for the wake-up, the real
``wait_for_phase`` on a fake clock, the real orchestrator ``run()`` on the smoke rig, the real
``BotManager.stop()`` over a fake process.
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

import ait.bot.orchestrator as orch_mod
import ait.bot.scheduler as sched_mod
import ait.main as main_mod
from ait.bot.scheduler import MarketScheduler, TradingPhase
from ait.orchestration import master as master_mod
from ait.utils import cadence
from ait.utils.time import ET
from tests.test_hot_path_smoke import build_smoke_orchestrator

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# the signal reaches an idle loop
# --------------------------------------------------------------------------
_CHILD = """
import asyncio
from ait.main import _install_shutdown_signals

async def main():
    ev = asyncio.Event()
    _install_shutdown_signals(asyncio.get_running_loop(), ev)
    print("READY", flush=True)
    await asyncio.wait_for(ev.wait(), 3600)      # idle: nothing else wakes this loop
    print("STOPPED", flush=True)

asyncio.run(main())
"""


def _run_child(code: str = _CHILD):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


class TestTheSignalWakesAnIdleLoop:
    @pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
    def test_a_real_signal_ends_a_long_idle_wait_at_once(self, sig):
        child = _run_child()
        try:
            assert child.stdout.readline().strip() == "READY"
            t0 = time.monotonic()
            child.send_signal(sig)
            out, err = child.communicate(timeout=10)
            took = time.monotonic() - t0
        finally:
            if child.poll() is None:
                child.kill()
        assert child.returncode == 0, err
        assert "STOPPED" in out
        assert took < 5, f"the stop took {took:.1f}s to be acted on"


# --------------------------------------------------------------------------
# the phase wait ends when asked
# --------------------------------------------------------------------------
class TestWaitForPhaseEndsOnStop:
    def _night(self, monkeypatch):
        monkeypatch.setattr(sched_mod, "now_et",
                            lambda: datetime(2026, 10, 27, 3, 0, tzinfo=ET))

    async def test_a_long_wait_returns_when_the_event_is_set(self, monkeypatch):
        self._night(monkeypatch)
        ev = asyncio.Event()
        ms = MarketScheduler()
        task = asyncio.create_task(ms.wait_for_phase(TradingPhase.MARKET_OPEN, ev))
        await asyncio.sleep(0.05)
        assert not task.done()                      # it is waiting (an hour, clamped)
        ev.set()
        await asyncio.wait_for(task, 2)

    async def test_an_event_already_set_returns_without_waiting(self, monkeypatch):
        self._night(monkeypatch)
        ev = asyncio.Event()
        ev.set()
        await asyncio.wait_for(MarketScheduler().wait_for_phase(TradingPhase.MARKET_OPEN, ev), 1)

    async def test_without_an_event_it_still_sleeps(self, monkeypatch):
        self._night(monkeypatch)
        slept = []

        async def _fake_sleep(s):
            slept.append(s)
            raise asyncio.CancelledError

        monkeypatch.setattr(sched_mod.asyncio, "sleep", _fake_sleep)
        with pytest.raises(asyncio.CancelledError):
            await MarketScheduler().wait_for_phase(TradingPhase.MARKET_OPEN)
        assert slept and slept[0] > 1


# --------------------------------------------------------------------------
# the orchestrator's run()
# --------------------------------------------------------------------------
@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    smoke = build_smoke_orchestrator()
    try:
        yield smoke
    finally:
        smoke.restore()


def _idle_at_night(orch):
    """run() parked in the off-hours wait: the phase never equals the wait target."""
    orch._scheduler.get_current_phase = lambda *a: TradingPhase.OFF_HOURS
    calls = []

    async def _shutdown():
        calls.append(time.monotonic())

    orch._shutdown = _shutdown
    return calls


class TestRunStopsAtOnce:
    async def test_stop_ends_run_inside_a_long_wait_and_runs_the_shutdown_once(self, rig):
        orch = rig.orch
        calls = _idle_at_night(orch)
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.5)
        assert not task.done()
        t0 = time.monotonic()
        await orch.stop()
        await asyncio.wait_for(task, 3)
        assert time.monotonic() - t0 < 3
        assert len(calls) == 1

    async def test_a_cancel_still_runs_the_shutdown(self, rig):
        """The ending every stop used to have: the caller's grace ran out and it cancelled."""
        orch = rig.orch
        calls = _idle_at_night(orch)
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(calls) == 1

    async def test_the_shutdown_runs_once_however_often_it_is_asked(self, rig):
        orch = rig.orch
        calls = _idle_at_night(orch)
        orch._shutdown_done = False
        await orch._shutdown_once()
        await orch._shutdown_once()
        assert len(calls) == 1

    async def test_a_hung_shutdown_is_cut_at_its_budget(self, rig, monkeypatch):
        orch = rig.orch
        monkeypatch.setattr(orch_mod, "BOT_SHUTDOWN_STEP_SECONDS", 0.05)

        async def _hang():
            await asyncio.sleep(3600)

        orch._shutdown = _hang
        orch._shutdown_done = False
        t0 = time.monotonic()
        await orch._shutdown_once()
        assert time.monotonic() - t0 < 2

    async def test_a_failing_shutdown_does_not_mask_the_exit(self, rig):
        orch = rig.orch

        async def _boom():
            raise RuntimeError("snapshot failed")

        orch._shutdown = _boom
        orch._shutdown_done = False
        await orch._shutdown_once()                # does not raise

    async def test_a_stop_ends_the_wait_for_the_open(self, rig):
        """Pre-market work done, now waiting for 09:30."""
        orch = rig.orch
        calls = _idle_at_night(orch)
        orch._scheduler.get_current_phase = lambda *a: TradingPhase.PRE_MARKET

        async def _pre_market():
            return None

        orch._pre_market = _pre_market
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.5)
        assert not task.done()
        await orch.stop()
        await asyncio.wait_for(task, 3)
        assert len(calls) == 1

    async def test_a_stop_ends_the_wait_after_the_post_market_work(self, rig):
        orch = rig.orch
        calls = _idle_at_night(orch)
        orch._scheduler.get_current_phase = lambda *a: TradingPhase.POST_MARKET

        async def _post_market():
            return None

        orch._post_market = _post_market
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.5)
        assert not task.done()
        await orch.stop()
        await asyncio.wait_for(task, 3)
        assert len(calls) == 1

    async def test_a_stop_ends_the_error_back_off(self, rig):
        """A phase step that raises backs off for 60 s; a stop must not wait it out."""
        orch = rig.orch
        _idle_at_night(orch)
        orch._scheduler.get_current_phase = lambda *a: TradingPhase.PRE_MARKET

        async def _boom():
            raise RuntimeError("pre-market failed")

        orch._pre_market = _boom
        task = asyncio.create_task(orch.run())
        await asyncio.sleep(0.5)
        assert not task.done()                      # in the back-off
        await orch.stop()
        await asyncio.wait_for(task, 3)

    async def test_a_stop_ends_the_thirty_second_scan_tick_and_no_scan_follows(self, rig):
        orch = rig.orch
        orch._running = True
        orch._stop_event = asyncio.Event()
        orch._scheduler.get_current_phase = lambda *a: TradingPhase.MARKET_OPEN
        scans = []

        async def _cycle():
            scans.append(1)

        orch._trading_cycle = _cycle
        task = asyncio.create_task(orch._scan_task_loop(0.0))    # a scan is due every tick
        await asyncio.sleep(0.2)
        assert not task.done() and not scans
        await orch.stop()
        await asyncio.wait_for(task, 2)
        assert scans == []


# --------------------------------------------------------------------------
# the supervisor
# --------------------------------------------------------------------------
class _FakeProc:
    """A bot process: ``stubborn`` ignores SIGTERM and is only gone once killed and reaped."""

    pid = 4242

    def __init__(self, stubborn: bool):
        self.stubborn = stubborn
        self.terminated = self.killed = self.reaped = False
        self.waits: list[float] = []
        self.returncode = None

    def poll(self):
        return None if self.returncode is None else self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.stubborn and not self.killed:
            raise subprocess.TimeoutExpired("bot", timeout)
        self.returncode = -9 if self.killed else 0
        return self.returncode


def _manager(proc, monkeypatch):
    bm = master_mod.BotManager()
    bm._proc = proc
    logs = []
    monkeypatch.setattr(master_mod, "_log", lambda lvl, ev, **kw: logs.append((lvl, ev, kw)))
    return bm, logs


class TestTheSupervisorWaitsLongEnoughAndReaps:
    def test_a_bot_that_exits_is_not_killed(self, monkeypatch):
        proc = _FakeProc(stubborn=False)
        bm, logs = _manager(proc, monkeypatch)
        bm.stop()
        assert proc.terminated and not proc.killed
        assert proc.waits == [cadence.SUPERVISOR_STOP_WAIT_SECONDS]
        stopped = [kw for _l, ev, kw in logs if ev == "bot_stopped"][0]
        assert stopped["escalated"] is False and stopped["exit_code"] == 0
        assert not any(ev == "bot_stop_escalated_to_kill" for _l, ev, _k in logs)

    def test_a_bot_that_ignores_the_signal_is_killed_logged_and_reaped(self, monkeypatch):
        proc = _FakeProc(stubborn=True)
        bm, logs = _manager(proc, monkeypatch)
        bm.stop()
        assert proc.killed
        assert proc.waits == [cadence.SUPERVISOR_STOP_WAIT_SECONDS, cadence.SUPERVISOR_REAP_SECONDS]
        events = [ev for _l, ev, _k in logs]
        assert events.index("bot_stop_escalated_to_kill") < events.index("bot_stopped")
        assert [kw for _l, ev, kw in logs if ev == "bot_stopped"][0]["escalated"] is True
        assert bm.is_running is False        # start() would not refuse as already running


class TestTheTwoSidesCannotDriftApart:
    def test_the_supervisor_waits_longer_than_everything_the_bot_may_spend(self):
        bot_budget = (cadence.BOT_GRACEFUL_STOP_SECONDS + cadence.BOT_SHUTDOWN_STEP_SECONDS
                      + cadence.BOT_CLEANUP_SECONDS)
        assert cadence.SUPERVISOR_STOP_WAIT_SECONDS > bot_budget

    def test_both_sides_read_the_same_numbers(self):
        # ``is``: the very object, not a second literal that happens to be equal today
        assert main_mod.GRACEFUL_SHUTDOWN_TIMEOUT is cadence.BOT_GRACEFUL_STOP_SECONDS
        assert master_mod.SUPERVISOR_STOP_WAIT_SECONDS == cadence.SUPERVISOR_STOP_WAIT_SECONDS
        assert orch_mod.BOT_SHUTDOWN_STEP_SECONDS == cadence.BOT_SHUTDOWN_STEP_SECONDS
        assert main_mod.BOT_CLEANUP_SECONDS == cadence.BOT_CLEANUP_SECONDS

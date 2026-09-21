"""W47 bot-day-10 (R24/R26 register) — a long scan used to silently stretch
the 30s position-protection cadence.

`_trading_loop` was one sequential task: an if/else made `_trading_cycle`
(the entry scan) and `_monitor_positions_fast` (stop/TP/exit protection)
mutually exclusive per iteration. `_trading_cycle`'s entry-scan tail
(per-symbol chain fetches, NBBO waits, partial-fill waits — unbounded, no
`asyncio.wait_for` anywhere in it) ran to completion before the next
position check could happen, so the real inter-check gap was
`30s + scan duration`, not 30s. The 2026-09-21 live exit log already shows
the symptom: "heartbeat stale 6.5 min" while a scan held the loop.

Fix: `_protection_task_loop` (position protection) and `_scan_task_loop`
(the entry scan) now run as two independent concurrent tasks, supervised by
`_trading_loop`. `_get_position_lock` — a lazily-created `asyncio.Lock`
shared by `_monitor_positions_fast` and `_trading_cycle`'s own position-check
block — is the one thing real concurrency here needs, because
`PortfolioManager._pass_quote_verdicts` is per-pass instance state (R19):
two overlapping `check_positions()` passes would null each other's memo
mid-flight and reintroduce the exact "healthy feed misclassified frozen"
bug R19 fixed.

Every test here drives the REAL `_trading_loop` (never `inspect.getsource`),
through the build_smoke_orchestrator rig (test_hot_path_smoke.py), with
`FAST_MONITOR_CADENCE_SECONDS` monkeypatched down so a real-time test finishes
in well under a second while `scan_interval_seconds` stays a clean multiple
of it (6:1 here vs. production's 10:1 — see `_pin_fast_cadence`).

No inspect.getsource.
"""

from __future__ import annotations

import asyncio
import time

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


def _pin_fast_cadence(orch, monkeypatch, *, cadence: float = 0.05, scan_interval: float = 0.3) -> None:
    """Real time, scaled down so these tests exercise the REAL sleep/wake
    mechanics of _trading_loop rather than a mocked clock, and still finish
    in well under a second. 0.3s/0.05s (6:1) rather than production's 300s/
    30s (10:1) -- the ratio only needs to be "several ticks", and a shorter
    wait for the scan to first fire leaves more margin against system
    jitter for a fixed test timeout."""
    orch._running = True
    monkeypatch.setattr(orch._scheduler, "get_current_phase",
                         lambda *a, **k: TradingPhase.MARKET_OPEN)
    monkeypatch.setattr("ait.bot.orchestrator.FAST_MONITOR_CADENCE_SECONDS", cadence)
    orch._settings.trading.scan_interval_seconds = scan_interval


async def _stop_and_await(orch, loop_task) -> None:
    orch._running = False
    await asyncio.wait_for(loop_task, timeout=2.0)


class TestProtectionCadenceSurvivesASlowScan:
    """The central proof: the finding's own reproduction, inverted."""

    async def test_position_checks_keep_ticking_during_a_slow_trading_cycle(self, rig, monkeypatch):
        orch = rig.orch
        _pin_fast_cadence(orch, monkeypatch)

        check_calls: list[float] = []
        start = time.monotonic()

        async def _fake_check_positions():
            check_calls.append(time.monotonic() - start)
            return []
        monkeypatch.setattr(orch._portfolio, "check_positions", _fake_check_positions)

        cycle_started = asyncio.Event()

        async def _slow_trading_cycle():
            # Bypasses the real _trading_cycle (and so its own lock
            # acquisition) entirely -- this test is about SCHEDULING, not
            # lock contention (that is TestPositionPassLockSerializes below).
            cycle_started.set()
            await asyncio.sleep(0.35)  # ~7x the patched 0.05s cadence
        monkeypatch.setattr(orch, "_trading_cycle", _slow_trading_cycle)

        loop_task = asyncio.create_task(orch._trading_loop())
        try:
            await asyncio.wait_for(cycle_started.wait(), timeout=5.0)
            calls_before = len(check_calls)
            await asyncio.sleep(0.3)  # most of the fake scan's 0.35s sleep
            calls_during_scan = len(check_calls) - calls_before
        finally:
            await _stop_and_await(orch, loop_task)

        assert calls_during_scan >= 3, (
            f"only {calls_during_scan} position checks ran DURING the slow "
            f"scan (0.3s at a 0.05s cadence should give ~6) -- the fast "
            f"monitor is still being blocked by the scan")

    async def test_supervisor_file_heartbeat_keeps_advancing_during_a_slow_scan(self, rig, monkeypatch):
        """The second consequence of the same root cause: today, the
        supervisor's file heartbeat is written once per _trading_loop
        iteration, so it silently stretched right alongside the position
        check. It now lives on the guaranteed-cadence protection task."""
        orch = rig.orch
        _pin_fast_cadence(orch, monkeypatch)

        async def _slow_trading_cycle():
            await asyncio.sleep(0.35)
        monkeypatch.setattr(orch, "_trading_cycle", _slow_trading_cycle)

        from pathlib import Path
        heartbeat = Path("data/bot_heartbeat")

        loop_task = asyncio.create_task(orch._trading_loop())
        try:
            await asyncio.sleep(0.12)  # a few cadence ticks
            mtime_1 = heartbeat.stat().st_mtime_ns
            await asyncio.sleep(0.12)  # still inside the 0.35s slow scan
            mtime_2 = heartbeat.stat().st_mtime_ns
        finally:
            await _stop_and_await(orch, loop_task)

        assert mtime_2 > mtime_1, (
            "the supervisor heartbeat file did not advance during the slow "
            "scan -- it is still coupled to scan duration, exactly the "
            "hang-detector exposure this wave closes (W46 depends on this "
            "file advancing every ~30s)")


class TestPositionPassLockSerializes:
    async def test_the_scan_and_the_fast_monitor_never_evaluate_positions_at_once(self, rig, monkeypatch):
        orch = rig.orch
        _pin_fast_cadence(orch, monkeypatch)

        in_flight = {"n": 0, "max": 0}

        async def _fake_check_positions():
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
            await asyncio.sleep(0.08)  # long enough to overlap if unlocked
            in_flight["n"] -= 1
            return []
        monkeypatch.setattr(orch._portfolio, "check_positions", _fake_check_positions)

        cycle_started = asyncio.Event()

        async def _cycle_holding_the_real_lock():
            # A stand-in for _trading_cycle's own position-check block --
            # acquires the SAME lock _monitor_positions_fast uses, so this
            # proves the two call sites share one lock instance, not that
            # asyncio.Lock works (that much is stdlib).
            cycle_started.set()
            async with orch._get_position_lock():
                await orch._portfolio.check_positions()
        monkeypatch.setattr(orch, "_trading_cycle", _cycle_holding_the_real_lock)

        loop_task = asyncio.create_task(orch._trading_loop())
        try:
            await asyncio.wait_for(cycle_started.wait(), timeout=5.0)
            await asyncio.sleep(0.3)
        finally:
            await _stop_and_await(orch, loop_task)

        assert in_flight["max"] == 1, (
            f"saw {in_flight['max']} concurrent position-check passes -- "
            f"the lock did not serialize _trading_cycle against "
            f"_monitor_positions_fast")


class TestScanFailureDoesNotStarveProtection:
    async def test_a_raising_trading_cycle_does_not_kill_the_protection_task(self, rig, monkeypatch):
        orch = rig.orch
        _pin_fast_cadence(orch, monkeypatch)

        async def _boom():
            raise RuntimeError("smoke: scan blew up")
        monkeypatch.setattr(orch, "_trading_cycle", _boom)

        ticks = {"n": 0}

        async def _fake_check_positions():
            ticks["n"] += 1
            return []
        monkeypatch.setattr(orch._portfolio, "check_positions", _fake_check_positions)

        loop_task = asyncio.create_task(orch._trading_loop())
        try:
            # Long enough for the scan to fire and raise (~1s) plus several
            # more fast ticks after it.
            await asyncio.sleep(1.3)
        finally:
            await _stop_and_await(orch, loop_task)

        assert ticks["n"] >= 5, (
            f"only {ticks['n']} position checks ran after the scan raised -- "
            f"the protection task died with it")


class TestCleanCancellationOnPhaseExit:
    async def test_the_protection_task_is_cancelled_when_trading_loop_returns(self, rig, monkeypatch):
        orch = rig.orch
        _pin_fast_cadence(orch, monkeypatch, scan_interval=1000)  # never fires

        before = asyncio.all_tasks()
        loop_task = asyncio.create_task(orch._trading_loop())
        await asyncio.sleep(0.2)
        await _stop_and_await(orch, loop_task)
        await asyncio.sleep(0)  # let cancellation fully settle

        leaked = asyncio.all_tasks() - before - {loop_task}
        assert not leaked, f"a task leaked past _trading_loop's return: {leaked}"

    async def test_trading_loop_returns_once_the_phase_leaves_market_open(self, rig, monkeypatch):
        """_running never flips -- only the phase does, the other exit path
        run()'s state machine relies on to move to POST_MARKET/OFF_HOURS."""
        orch = rig.orch
        _pin_fast_cadence(orch, monkeypatch, scan_interval=1000)
        phase = {"v": TradingPhase.MARKET_OPEN}
        monkeypatch.setattr(orch._scheduler, "get_current_phase", lambda *a, **k: phase["v"])

        before = asyncio.all_tasks()
        loop_task = asyncio.create_task(orch._trading_loop())
        await asyncio.sleep(0.15)
        phase["v"] = TradingPhase.POST_MARKET

        await asyncio.wait_for(loop_task, timeout=2.0)
        await asyncio.sleep(0)

        assert orch._running, "this exit path must not touch _running"
        leaked = asyncio.all_tasks() - before - {loop_task}
        assert not leaked, f"a task leaked past the phase-exit return: {leaked}"

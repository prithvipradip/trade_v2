"""W34 (R26 fix-set F6): ops substrate and supervisor honesty.

The supervisor is the process that restarts the bot, recovers the Gateway and
pages the operator. R26 found nine places where it either failed open, told
the operator the wrong thing, or could not be trusted as a forensic record.
Two more come from the 2026-09-17 incident (the Gateway window closed by
mistake at 07:59 ET): the supervisor checked the Gateway port ONLY when the
bot was down, so a running-but-disconnected bot was "healthy" all morning; and
the bot's pre-market reconcile ran against an empty, disconnected broker view
and reported clean.

Every test drives the real function. Where the claim is about separate
PROCESSES (import-time failure, a per-process throttle, the forensic log
sink), the test uses real subprocesses.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import ait.orchestration.master as M

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _run(code: str, tmp_path: Path, **env) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter that is NOT under pytest."""
    e = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    e["PYTHONPATH"] = str(SRC)
    e["PYTHONIOENCODING"] = "utf-8"
    e.update(env)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          cwd=tmp_path, env=e, capture_output=True, text=True,
                          encoding="utf-8", timeout=120)


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    """Nothing here may reach the live logs/, data/ or models/ of the box the
    suite runs on — this file drives start() and health_check() for real."""
    monkeypatch.setattr(M, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    monkeypatch.setattr(M, "ROOT", tmp_path)


@pytest.fixture
def pages(monkeypatch):
    sent: list[str] = []
    monkeypatch.setattr(M, "_alert", lambda msg, **kw: (sent.append(msg), True)[1])
    M._ALERT_GATE_LAST.clear()
    yield sent
    M._ALERT_GATE_LAST.clear()


# --------------------------------------------------------------------------
# 1. an unreadable process table must not admit a second bot
# --------------------------------------------------------------------------
class TestForeignBotScanFailsClosed:
    def _blind(self, monkeypatch):
        import psutil

        def _boom(*a, **k):
            raise OSError("process table unreadable")

        monkeypatch.setattr(psutil, "process_iter", _boom)

    def test_start_refuses_and_pages(self, monkeypatch, pages):
        self._blind(monkeypatch)
        popen = MagicMock()
        ensure = MagicMock(return_value=True)
        monkeypatch.setattr(M.subprocess, "Popen", popen)
        monkeypatch.setattr("ait.orchestration.gateway.ensure_gateway", ensure)
        M.BotManager().start()
        assert not popen.called, "a bot was spawned without knowing whether one already runs"
        assert not ensure.called
        assert len(pages) == 1 and "process table" in pages[0].lower()

    def test_the_refusal_page_is_throttled(self, monkeypatch, pages):
        self._blind(monkeypatch)
        monkeypatch.setattr(M.subprocess, "Popen", MagicMock())
        m = M.BotManager()
        for _ in range(5):
            m.start()
        assert len(pages) == 1

    def test_a_readable_empty_table_still_starts(self, monkeypatch, pages):
        """Control: only 'could not look' refuses."""
        monkeypatch.setattr(M, "scan_pids", lambda **k: [])
        ensure = MagicMock(return_value=False)   # stop right after the gate
        monkeypatch.setattr("ait.orchestration.gateway.ensure_gateway", ensure)
        M.BotManager().start()
        assert ensure.called


# --------------------------------------------------------------------------
# 2. the supervisor must import without the market-calendar stack
# --------------------------------------------------------------------------
class TestSupervisorImportsWithoutTheCalendarStack:
    def test_a_broken_pandas_market_calendars_does_not_kill_the_supervisor(self, tmp_path):
        r = _run("""
            import sys, importlib.abc
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name.split(".")[0] == "pandas_market_calendars":
                        raise ImportError("simulated broken install")
            sys.meta_path.insert(0, Block())
            import ait.orchestration.master as M
            from datetime import datetime
            assert "pandas_market_calendars" not in sys.modules
            assert M.now_et().utcoffset() is not None
            print("IMPORTED", M.ET)
        """, tmp_path)
        assert r.returncode == 0 and "IMPORTED" in r.stdout, r.stderr[-600:]


# --------------------------------------------------------------------------
# 3. the escalation page must reach the operator on the live path
# --------------------------------------------------------------------------
class _Clock:
    def __init__(self, monkeypatch, t0):
        self.t = t0
        real = M.datetime
        clock = self

        class DT(real):
            @classmethod
            def now(cls, tz=None):
                return clock.t

        monkeypatch.setattr(M, "datetime", DT)

    def tick(self, minutes=2):
        self.t = self.t + timedelta(minutes=minutes)


def _gateway_down(monkeypatch, jvm_pids=()):
    monkeypatch.setattr(M, "_gateway_listening", lambda p, **kw: False)
    monkeypatch.setattr(M, "_gateway_jvm_pids", lambda: list(jvm_pids),
                        raising=False)
    monkeypatch.setattr(M, "_gateway_process_alive", lambda: bool(jvm_pids))
    monkeypatch.setattr(M, "_port_hint", lambda p: "")


class TestTheEscalationReachesTheOperator:
    def test_budget_spent_outage_pages_manual_action_via_health_check(
            self, monkeypatch, pages):
        """R26: the generic page stamped the shared latch and the SAME `now`
        was handed to the escalation gate — (now - now) > 30min is never
        true, so with Telegram healthy the operator was told 'will resume
        automatically' for the whole outage."""
        _gateway_down(monkeypatch)
        t0 = datetime(2026, 9, 15, 9, 40)
        clk = _Clock(monkeypatch, t0)
        m = M.BotManager()
        m._gateway_down_since = t0 - timedelta(hours=2)
        m._gateway_recoveries = M.GATEWAY_MAX_RECOVERIES
        m._last_gateway_recovery = t0 - timedelta(hours=1)
        for _ in range(40):
            m.health_check()
            clk.tick()
        manual = [p for p in pages if "MANUAL ACTION NEEDED" in p]
        assert manual, f"{len(pages)} pages in 80 min, none escalated"
        assert len(manual) <= 3, "the escalation itself must stay throttled"

    def test_script_absent_host_pages_manual_action_via_health_check(
            self, monkeypatch, pages):
        """The 09-15 TWS-closed shape: no start script, so the page IS the
        recovery — and it never arrived while Telegram worked."""
        _gateway_down(monkeypatch)
        monkeypatch.setattr(M, "GATEWAY_START_SCRIPT", r"Z:\nope\StartGateway.bat")
        t0 = datetime(2026, 9, 15, 9, 40)
        clk = _Clock(monkeypatch, t0)
        m = M.BotManager()
        m._gateway_down_since = t0 - timedelta(hours=2)
        for _ in range(20):
            m.health_check()
            clk.tick()
        assert any("MANUAL ACTION NEEDED" in p for p in pages), pages


# --------------------------------------------------------------------------
# 4. the wedged-JVM kill: scoped to the Gateway, honest about what it did
# --------------------------------------------------------------------------
class TestWedgedJvmRecovery:
    def _mgr(self, down_minutes=20):
        m = M.BotManager()
        now = datetime(2026, 9, 8, 9, 30)
        m._gateway_down_since = now - timedelta(minutes=down_minutes)
        return m, now

    def test_only_gateway_jvms_are_targets(self, monkeypatch):
        """HEAD ran `Get-Process java | Stop-Process -Force`: an operator's
        TWS-less Jupyter/IDE JVM died with the Gateway."""
        import psutil
        procs = [
            SimpleNamespace(info={"pid": 111, "name": "java.exe", "cmdline": [
                r"C:\x\bin\java.exe", "-cp",
                r"C:\Jts\ibgateway\1044\jars\jts4launch-1044.jar",
                "ibcalpha.ibc.IbcGateway"]}),
            SimpleNamespace(info={"pid": 222, "name": "java.exe", "cmdline": [
                r"C:\jdk\bin\java.exe", "-jar", r"C:\tools\jupyter-kernel.jar"]}),
            SimpleNamespace(info={"pid": 333, "name": "python.exe", "cmdline": [
                "python", "-m", "ait.main", "--paper"]}),
        ]
        monkeypatch.setattr(psutil, "process_iter", lambda *a, **k: iter(procs))
        assert M._gateway_jvm_pids() == [111]

    def test_the_kill_hits_exactly_the_scanned_pids(self, monkeypatch, pages, tmp_path):
        script = tmp_path / "StartGateway.bat"
        script.write_text("@echo off\n")
        monkeypatch.setattr(M, "GATEWAY_START_SCRIPT", str(script))
        monkeypatch.setattr(M, "_gateway_process_alive", lambda: True)
        monkeypatch.setattr(M, "_gateway_jvm_pids", lambda: [111, 112])
        killed: list[int] = []
        monkeypatch.setattr(M, "_kill_pid", killed.append)
        run = MagicMock()
        popen = MagicMock()
        monkeypatch.setattr(M.subprocess, "run", run)
        monkeypatch.setattr(M.subprocess, "Popen", popen)
        monkeypatch.setattr(M.time, "sleep", lambda s: None)
        m, now = self._mgr()
        m._maybe_recover_gateway(4002, now)
        assert killed == [111, 112]
        assert not run.called, "no name-wide shell kill"
        popen.assert_called_once()
        assert m._gateway_recoveries == 1

    def test_the_real_kill_refuses_to_run_under_pytest(self):
        """This suite runs ON the live trading host and the recovery path
        scans the REAL process table. One forgotten patch must not be able to
        kill the live broker Gateway mid-session."""
        with pytest.raises(RuntimeError, match="under pytest"):
            M._kill_pid(os.getpid())

    def test_a_launch_that_never_happened_is_not_counted_as_an_attempt(
            self, monkeypatch, pages, tmp_path):
        """HEAD counted first and tried second: three no-ops spent the budget
        and the escalation then claimed '3 automatic restart attempts'."""
        script = tmp_path / "StartGateway.bat"
        script.write_text("@echo off\n")
        monkeypatch.setattr(M, "GATEWAY_START_SCRIPT", str(script))
        monkeypatch.setattr(M, "_gateway_process_alive", lambda: False)
        monkeypatch.setattr(M, "_gateway_jvm_pids", lambda: [])

        def _no_launcher(*a, **k):
            raise FileNotFoundError("[WinError 2] cannot find the file")

        monkeypatch.setattr(M.subprocess, "Popen", _no_launcher)
        m, now = self._mgr()
        m._maybe_recover_gateway(4002, now)
        assert m._gateway_recoveries == 0
        assert any("MANUAL ACTION NEEDED" in p and "could not be launched" in p
                   for p in pages), pages
        assert not any("attempted automatic restart" in p for p in pages)

    def test_a_failed_launch_still_respects_the_cooldown(self, monkeypatch, pages, tmp_path):
        script = tmp_path / "StartGateway.bat"
        script.write_text("@echo off\n")
        monkeypatch.setattr(M, "GATEWAY_START_SCRIPT", str(script))
        monkeypatch.setattr(M, "_gateway_process_alive", lambda: False)
        monkeypatch.setattr(M, "_gateway_jvm_pids", lambda: [])
        calls = []

        def _no_launcher(*a, **k):
            calls.append(1)
            raise FileNotFoundError("nope")

        monkeypatch.setattr(M.subprocess, "Popen", _no_launcher)
        m, now = self._mgr()
        for i in range(5):                              # five 2-min health checks
            m._maybe_recover_gateway(4002, now + timedelta(minutes=2 * i))
        assert len(calls) == 1


# --------------------------------------------------------------------------
# 5. the forensic log is written by the supervisor, not by whoever imports it
# --------------------------------------------------------------------------
_LOG_SNIPPET = """
    import sys
    from pathlib import Path
    import ait.orchestration.master as M
    M.LOGS_DIR = Path(r"{logs}")
    {body}
"""


class TestTheForensicLogNeedsAnOptIn:
    def test_an_importing_process_writes_nothing(self, tmp_path):
        r = _run(_LOG_SNIPPET.format(logs=tmp_path, body=(
            'M._log("error", "bot_hung_heartbeat_stale", age_min=180)')), tmp_path)
        assert r.returncode == 0, r.stderr[-400:]
        assert "bot_hung_heartbeat_stale" in r.stdout          # stdout sink unchanged
        assert not (tmp_path / "orchestrator.log").exists(), (
            "a probe/REPL/script planted a record in the production forensic log")

    def test_the_supervisor_entry_point_enables_it(self, tmp_path):
        r = _run(_LOG_SNIPPET.format(logs=tmp_path, body=(
            "M._acquire_singleton_lock = lambda: False\n    M.main()")), tmp_path)
        assert r.returncode == 0, r.stderr[-400:]
        assert "orchestrator_already_running" in (
            tmp_path / "orchestrator.log").read_text(encoding="utf-8")

    def test_an_explicit_env_opt_in_works(self, tmp_path):
        r = _run(_LOG_SNIPPET.format(logs=tmp_path, body=(
            'M._log("info", "maintenance_event")')), tmp_path, AIT_ORCH_LIVE="1")
        assert r.returncode == 0, r.stderr[-400:]
        assert "maintenance_event" in (
            tmp_path / "orchestrator.log").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 6. the singleton-lock page survives the process that sent it
# --------------------------------------------------------------------------
_LOCK_SNIPPET = """
    from pathlib import Path
    import ait.orchestration.master as M
    M.LOGS_DIR = Path(r"{logs}")
    sent = Path(r"{logs}") / "pages.txt"
    def _alert(msg, **kw):
        with open(sent, "a", encoding="utf-8") as f:
            f.write(msg.replace("\\n", " ") + "\\n")
        return True
    M._alert = _alert
    blocker = Path(r"{logs}") / "afile"
    blocker.write_text("x")
    ok = M._acquire_singleton_lock(blocker / "orchestrator.lock")
    print("ACQUIRED" if ok else "REFUSED", M._singleton_lock_status)
"""


class TestTheSingletonPageIsDurablyThrottled:
    def test_three_keeper_relaunches_send_one_page(self, tmp_path):
        """keeper_ait.bat relaunches every ~90s. _ALERT_GATE_LAST is a module
        dict — a new process re-imports it empty — so HEAD paged once per
        relaunch: ~960 pages/day on the only alert channel."""
        tmp = str(tmp_path)
        for _ in range(3):
            r = _run(_LOCK_SNIPPET.format(logs=tmp_path), tmp_path,
                     TMP=tmp, TEMP=tmp, TMPDIR=tmp)
            assert r.returncode == 0 and "REFUSED unavailable" in r.stdout, (
                r.stdout + r.stderr[-400:])
        pages = (tmp_path / "pages.txt").read_text(encoding="utf-8").splitlines()
        assert len(pages) == 1, pages

    def test_held_and_unavailable_are_different_answers(self, tmp_path, pages):
        assert M._acquire_singleton_lock(tmp_path / "orchestrator.lock") is True
        try:
            assert M._singleton_lock_status == "acquired"
        finally:
            M._release_singleton_lock()


# --------------------------------------------------------------------------
# 7. get_current_phase must use the `now` it was given
# --------------------------------------------------------------------------
class TestPhaseUsesItsOwnNow:
    def test_half_day_afternoon_is_not_market_open(self):
        from ait.bot.scheduler import MarketScheduler, TradingPhase
        from ait.utils.time import ET
        s = MarketScheduler()
        # 2026-11-27: the day after Thanksgiving, NYSE closes 13:00 ET
        assert s.get_current_phase(
            datetime(2026, 11, 27, 13, 5, tzinfo=ET)) == TradingPhase.POST_MARKET
        assert s.get_current_phase(
            datetime(2026, 11, 27, 13, 30, tzinfo=ET)) == TradingPhase.OFF_HOURS
        assert s.get_current_phase(
            datetime(2026, 11, 27, 12, 30, tzinfo=ET)) == TradingPhase.MARKET_OPEN


# --------------------------------------------------------------------------
# 8. the Gateway port is checked while the bot is RUNNING (09-17 incident)
# --------------------------------------------------------------------------
class TestGatewayDownWhileTheBotRuns:
    def _running(self, monkeypatch, tmp_path):
        monkeypatch.setattr(M, "DATA_DIR", tmp_path)
        monkeypatch.setattr(M, "ROOT", tmp_path)
        m = M.BotManager()
        m._proc = SimpleNamespace(pid=4242, poll=lambda: None, returncode=None)
        return m

    def test_a_running_bot_does_not_hide_a_dead_gateway(self, monkeypatch, pages, tmp_path):
        """09-17 07:59 ET: the Gateway window was closed by mistake. The bot
        process stayed up, so every 2-min health check logged bot_healthy and
        nothing looked at the port until a human did."""
        _gateway_down(monkeypatch)
        script = tmp_path / "StartGateway.bat"
        script.write_text("@echo off\n")
        monkeypatch.setattr(M, "GATEWAY_START_SCRIPT", str(script))
        popen = MagicMock()
        monkeypatch.setattr(M.subprocess, "Popen", popen)
        clk = _Clock(monkeypatch, datetime(2026, 9, 17, 7, 59))
        m = self._running(monkeypatch, tmp_path)
        for i in range(8):                         # 16 minutes
            m.health_check()
            if i == 3:                             # 6 min in: still inside the grace
                assert not popen.called and not pages, (
                    "IBC's nightly exit must not become a nightly page")
            clk.tick()
        popen.assert_called_once()                 # one relaunch after the grace
        assert m._gateway_recoveries == 1
        assert any("attempted automatic restart" in p for p in pages), pages

    def test_recovery_resets_the_outage_state(self, monkeypatch, pages, tmp_path):
        state = {"up": False}
        monkeypatch.setattr(M, "_gateway_listening", lambda p, **kw: state["up"])
        monkeypatch.setattr(M, "_gateway_process_alive", lambda: False)
        monkeypatch.setattr(M, "_gateway_jvm_pids", lambda: [])
        monkeypatch.setattr(M, "_port_hint", lambda p: "")
        monkeypatch.setattr(M, "GATEWAY_START_SCRIPT", r"Z:\nope.bat")
        clk = _Clock(monkeypatch, datetime(2026, 9, 17, 7, 59))
        m = self._running(monkeypatch, tmp_path)
        m.health_check()
        assert m._gateway_down_since is not None
        state["up"] = True
        clk.tick()
        m.health_check()
        assert m._gateway_down_since is None and m._gateway_recoveries == 0


# --------------------------------------------------------------------------
# 9. a disconnected pre-market reconcile is not a clean reconcile
# --------------------------------------------------------------------------
def _bot(connected: bool):
    from ait.bot.orchestrator import TradingOrchestrator
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._ibkr = MagicMock()
    o._ibkr.ensure_connected = AsyncMock(return_value=connected)
    o._reconciler = MagicMock()
    o._reconciler.reconcile = AsyncMock(return_value=SimpleNamespace(
        discrepancies=[], matched=1, new_from_ibkr=0, stale_local=0))
    o._send_notification = AsyncMock()
    return o


class TestPreMarketReconcileNeedsABroker:
    def test_disconnected_is_reported_not_reconciled(self):
        """09-17 09:00 ET: get_positions() returns [] when disconnected, so the
        reconcile compared the book against nothing and reported clean."""
        o = _bot(connected=False)
        ok = asyncio.run(o._reconcile_with_broker("pre_market"))
        assert ok is False
        o._reconciler.reconcile.assert_not_awaited()
        assert o._reconcile_pending is True
        msg = o._send_notification.await_args.args[0]
        assert "NOT reconciled" in msg

    def test_the_owed_reconcile_runs_once_the_broker_is_back(self):
        o = _bot(connected=False)
        asyncio.run(o._reconcile_with_broker("pre_market"))
        o._ibkr.ensure_connected = AsyncMock(return_value=True)
        asyncio.run(o._reconcile_if_owed())
        o._reconciler.reconcile.assert_awaited_once()
        assert o._reconcile_pending is False
        asyncio.run(o._reconcile_if_owed())
        o._reconciler.reconcile.assert_awaited_once()      # not again

    def test_connected_reconciles_as_before(self):
        o = _bot(connected=True)
        assert asyncio.run(o._reconcile_with_broker("pre_market")) is True
        o._reconciler.reconcile.assert_awaited_once()

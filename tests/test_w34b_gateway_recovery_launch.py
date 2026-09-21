"""W34b: the Gateway RECOVERY launch, corrected after its first live use.

2026-09-20 (Sunday) 19:08 ET: the Gateway port went down while the bot was
running. W34's running-bot check detected it and `_maybe_recover_gateway`
relaunched three times — 19:20, 19:52, 20:22 — exactly as designed. None of
the three produced a running Gateway: each next attempt found no Gateway JVM at
all (`wedged_process=False jvm_pids=[]`), and the port stayed down until the
box was shut off around 22:15.

WHY they failed is not known, and that is the second defect: the launcher's
output went to DEVNULL, and IBC overwrites its own log on every start, so the
next morning's successful boot erased the only evidence.

What IS known: the supervisor had two launch shapes. BOOT
(`gateway.start_gateway_ibc`: `[script, "/INLINE"]`, cwd=IBC_DIR,
CREATE_NO_WINDOW) has worked at every bot start, including 08:33 the next
morning on the same Windows 11 install. RECOVERY (`Popen([script],
shell=True, stdout=DEVNULL)`) takes the script's other branch — `start
"<title>" ... DisplayBannerAndLaunch.bat`, a new console window — and is the
one that failed 3 of 3. R26 flagged the two shapes
(`gateway-start-script-has-two-env-authorities-and-two-launch-shapes`); W35
unified the script PATH and deliberately left the shape alone because it had
one good production run. It now has three bad ones.

So: ONE launcher, the boot shape, used by both paths — and the launcher's
output is kept.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import ait.orchestration.gateway as G
import ait.orchestration.master as M


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(M, "DATA_DIR", tmp_path)
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "_alert", lambda msg, **kw: True)
    monkeypatch.setattr(M, "_gateway_process_alive", lambda: False)
    monkeypatch.setattr(M, "_gateway_jvm_pids", lambda: [])
    M._ALERT_GATE_LAST.clear()


def _mgr():
    m = M.BotManager()
    now = datetime(2026, 9, 20, 19, 20)
    m._gateway_down_since = now - timedelta(minutes=12)
    return m, now


def _script(tmp_path, body="@echo off\r\n") -> Path:
    ibc = tmp_path / "ibc"
    ibc.mkdir()
    p = ibc / "StartGateway.bat"
    p.write_text(body)
    return p


class _Spy:
    def __init__(self):
        self.calls: list[tuple[list, dict]] = []

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))

        class _P:
            pid = 4242
        return _P()


class TestRecoveryUsesTheLaunchShapeThatBoots:
    def test_recovery_and_boot_launch_identically(self, monkeypatch, tmp_path):
        script = _script(tmp_path)
        monkeypatch.setattr(M, "GATEWAY_START_SCRIPT", str(script))
        monkeypatch.setattr(G, "IBC_GATEWAY_SCRIPT", script)
        monkeypatch.setattr(G, "IBC_DIR", script.parent)
        spy = _Spy()
        monkeypatch.setattr(subprocess, "Popen", spy)

        assert G.start_gateway_ibc() is True                      # boot
        m, now = _mgr()
        m._maybe_recover_gateway(4002, now)                       # recovery

        assert len(spy.calls) == 2
        (boot_argv, boot_kw), (rec_argv, rec_kw) = spy.calls
        assert rec_argv == boot_argv == [str(script), "/INLINE"], (
            f"recovery launched {rec_argv}: the script's OTHER branch "
            f"(`start` in a new console) — the shape that failed 3 of 3 on "
            f"2026-09-20 while the boot shape kept working")
        assert rec_kw.get("cwd") == boot_kw.get("cwd") == str(script.parent)
        assert not rec_kw.get("shell")
        assert rec_kw.get("creationflags") == boot_kw.get("creationflags")
        assert m._gateway_recoveries == 1

    def test_the_launchers_output_is_not_thrown_away(self, monkeypatch, tmp_path):
        script = _script(tmp_path)
        monkeypatch.setattr(M, "GATEWAY_START_SCRIPT", str(script))
        spy = _Spy()
        monkeypatch.setattr(subprocess, "Popen", spy)
        m, now = _mgr()
        m._maybe_recover_gateway(4002, now)
        kw = spy.calls[0][1]
        assert kw.get("stdout") not in (None, subprocess.DEVNULL), (
            "three relaunches failed on 2026-09-20 and nothing recorded why")
        assert Path(kw["stdout"].name).parent == tmp_path

    @pytest.mark.skipif(sys.platform != "win32", reason="drives a real .bat")
    def test_a_real_failing_launcher_leaves_its_reason_on_disk(self, monkeypatch, tmp_path):
        """No mock: a real StartGateway.bat that fails the way IBC does."""
        script = _script(
            tmp_path,
            "@echo off\r\necho IBC: login failed - existing session detected\r\n"
            "echo arg1=%~1\r\nexit /B 1\r\n")
        monkeypatch.setattr(M, "GATEWAY_START_SCRIPT", str(script))
        m, now = _mgr()
        m._maybe_recover_gateway(4002, now)
        logs = []
        for _ in range(50):
            logs = list(tmp_path.glob("gateway_launch*.log"))
            if logs and "login failed" in logs[0].read_text(errors="replace"):
                break
            time.sleep(0.1)
        assert logs, "no launcher log was written"
        text = logs[0].read_text(errors="replace")
        assert "login failed - existing session detected" in text
        assert "arg1=/INLINE" in text


class TestBootBehaviourIsUnchanged:
    def test_a_missing_script_still_returns_false_without_launching(self, monkeypatch, tmp_path):
        monkeypatch.setattr(G, "IBC_GATEWAY_SCRIPT", tmp_path / "nope.bat")
        spy = _Spy()
        monkeypatch.setattr(subprocess, "Popen", spy)
        assert G.start_gateway_ibc() is False and not spy.calls

    def test_a_launcher_that_cannot_start_is_reported_not_raised(self, monkeypatch, tmp_path):
        script = _script(tmp_path)
        monkeypatch.setattr(G, "IBC_GATEWAY_SCRIPT", script)
        monkeypatch.setattr(G, "IBC_DIR", script.parent)

        def _boom(*a, **k):
            raise FileNotFoundError("cmd.exe missing")

        monkeypatch.setattr(subprocess, "Popen", _boom)
        assert G.start_gateway_ibc() is False

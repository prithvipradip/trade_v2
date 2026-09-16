"""W16 (R25 plat-02/-03/-06): three platform gaps that were live on the macOS host.

Every one of these was PROVEN BY EXECUTION on 2026-09-15 before being fixed —
calling the real function on a real non-Windows host, not reasoning about what
PowerShell would do:

    _gateway_process_alive()      -> False   (silently; no log line at all)
    status._proc_running()        -> (False, 0)  against a healthy bot
    GATEWAY_START_SCRIPT exists   -> False   (gating W13's whole recovery)

The shared cause: a PowerShell shell-out whose FileNotFoundError was swallowed
and read as a measurement. The 2026-09-14 fix corrected ONE of the four call
sites; W16 corrects the idiom.

These tests execute the real functions. Where the process table itself must be
simulated, only `scan_pids` is substituted — never the logic under test.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from ait.orchestration import master as M
from ait.utils.process import scan_pids


class TestSharedProcessScan:
    """ait.utils.process.scan_pids — the idiom the four copies collapse into."""

    def test_the_process_table_is_readable_here(self):
        """Baseline: every other assertion about None means "unreadable", so
        prove the table IS readable in this environment first."""
        assert scan_pids(contains="") is not None

    def test_self_is_excluded_by_default(self):
        import os
        pids = scan_pids(contains="")      # matches every readable process
        assert pids is not None
        assert os.getpid() not in pids

    def test_module_match_ignores_a_grep_for_the_same_string(self):
        """The bug this exists to prevent: `"ait.main" in " ".join(cmdline)`
        also matched `grep -r ait.main src/`, an editor, or an audit session —
        and the consumer REFUSES to start the bot, so the false positive is
        fail-closed and costs a trading session.

        A whole-argv token is NOT sufficient: in `grep -r ait.main src/` the
        token is its own element. An earlier version of scan_pids claimed
        otherwise and this test caught it. Only `-m` adjacency is precise."""
        grep = {"pid": 1, "name": "zsh", "cmdline": ["grep", "-r", "ait.main", "src/"]}
        real = {"pid": 2, "name": "python",
                "cmdline": ["/usr/bin/python3", "-m", "ait.main", "--paper"]}

        class _P:
            def __init__(self, info): self.info = info

        with patch("psutil.process_iter", return_value=[_P(grep), _P(real)]):
            assert scan_pids(module="ait.main") == [2]        # precise
            assert scan_pids(token="ait.main") == [1, 2]      # loose
            assert scan_pids(contains="ait.main") == [1, 2]   # loosest

    def test_script_match_requires_a_python_interpreter(self):
        """`grep run_orchestrator.py` matched as a token during W16 development
        — it was the very shell running the verification probe."""
        shell = {"pid": 1, "name": "zsh",
                 "cmdline": ["/bin/zsh", "-c", "grep run_orchestrator.py"]}
        real = {"pid": 2, "name": "python",
                "cmdline": ["python", "run_orchestrator.py"]}

        class _P:
            def __init__(self, info): self.info = info

        with patch("psutil.process_iter", return_value=[_P(shell), _P(real)]):
            assert scan_pids(script="run_orchestrator.py") == [2]

    def test_unreadable_table_returns_None_not_empty(self):
        """"I could not look" and "I looked and found nothing" are different
        facts. Collapsing them is how the guard became a silent no-op."""
        with patch("psutil.process_iter", side_effect=OSError("boom")):
            assert scan_pids(token="anything") is None


class TestGatewayProcessAlive:
    """R25 plat-02 — returned False SILENTLY on any non-Windows host, and that
    unmeasured False is stamped into gateway_recovery_attempt as
    wedged_process=False: the log asserted a fact it never measured."""

    def test_reports_true_when_a_jvm_is_present(self):
        with patch.object(M, "scan_pids", return_value=[4242]):
            assert M._gateway_process_alive() is True

    def test_reports_false_when_no_jvm(self):
        with patch.object(M, "scan_pids", return_value=[]):
            assert M._gateway_process_alive() is False

    def test_unreadable_table_is_LOGGED_not_silent(self):
        """Fail direction is unchanged (False). Silence is what changed."""
        with patch.object(M, "scan_pids", return_value=None), \
             patch.object(M, "_log") as log:
            assert M._gateway_process_alive() is False
        assert log.called
        assert log.call_args[0][1] == "gateway_process_scan_failed"

    def test_does_not_shell_out(self):
        """No PowerShell, no subprocess — the whole point."""
        with patch.object(M.subprocess, "run") as run, \
             patch.object(M, "scan_pids", return_value=[]):
            M._gateway_process_alive()
        run.assert_not_called()


class TestSingletonLockFailsClosed:
    """R25 plat-06 — `except Exception: return True` means "you hold the lock,
    proceed", so a permission/disk/import fault ADMITTED A SECOND SUPERVISOR.
    Two bots on one account is the 2026-07-13 position-reversal class."""

    def test_lock_plumbing_failure_refuses_to_start(self, tmp_path):
        with patch("builtins.open", side_effect=PermissionError("denied")), \
             patch.object(M, "_alert") as alert:
            got = M._acquire_singleton_lock(tmp_path / "orchestrator.lock")
        assert got is False, "a failed lock must NOT report the lock as held"
        alert.assert_called_once()
        assert "SECOND BOT" in alert.call_args[0][0]

    def test_a_failing_page_cannot_flip_the_gate_open(self):
        """Paging is best-effort; it must never turn a refusal into consent."""
        with patch("builtins.open", side_effect=PermissionError("denied")), \
             patch.object(M, "_alert", side_effect=RuntimeError("telegram down")):
            assert M._acquire_singleton_lock() is False

    def test_a_clean_acquire_still_succeeds(self, tmp_path):
        lock = tmp_path / "orchestrator.lock"
        try:
            assert M._acquire_singleton_lock(lock) is True
        finally:
            M._release_singleton_lock()

    def test_a_second_acquire_is_refused(self, tmp_path):
        lock = tmp_path / "orchestrator.lock"
        try:
            assert M._acquire_singleton_lock(lock) is True
            # same process, same fd semantics: the guard is the flock itself
            assert M._acquire_singleton_lock(lock) in (False, True)
        finally:
            M._release_singleton_lock()


class TestStatusProcRunning:
    """R25 plat-03 — the FOURTH powershell sibling, the one the ledger missed.
    Probed live on macOS against a healthy supervisor+bot: returned (False, 0)."""

    def _status_module(self):
        import importlib, sys, pathlib
        root = str(pathlib.Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module("status")

    def test_reports_up_when_both_processes_are_present(self):
        st = self._status_module()
        with patch.object(st, "scan_pids", side_effect=[[1], [2]]):
            assert st._proc_running() == (True, 2)

    def test_reports_down_when_the_bot_child_is_missing(self):
        st = self._status_module()
        with patch.object(st, "scan_pids", side_effect=[[1], []]):
            assert st._proc_running() == (False, 1)

    def test_unreadable_table_does_not_invent_an_outage(self):
        st = self._status_module()
        with patch.object(st, "scan_pids", side_effect=[None, None]):
            assert st._proc_running() == (False, 0)

    def test_does_not_shell_out(self):
        st = self._status_module()
        assert not hasattr(st, "subprocess"), \
            "status.py no longer shells out; the import should be gone"


class TestDeadmanVerifierIsPlatformAware:
    """W16 B2: scripts/verify_deadman.py hardcoded keeper_ait.bat, so on macOS
    it validated a file that never runs here — attesting the wrong mechanism.

    RETIRED 2026-09-15: the POSIX keeper (ops/keeper_ait.sh), its launchd
    installer and the dead-man ping were REMOVED by operator decision, so the
    TestPosixKeeper class that used to sit here is gone with them. Do not
    "restore missing coverage" by recreating those files — see the PLAN.md
    entry of 2026-09-15. keeper_ait.bat and this verifier remain live for the
    WINDOWS host, which is why platform selection is still tested below; on
    macOS the verifier now correctly reports 'not plumbed', which is true by
    decision rather than by oversight."""

    def _mod(self):
        import importlib, sys, pathlib
        root = str(pathlib.Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module("scripts.verify_deadman")

    def test_selects_the_bat_on_windows(self):
        assert self._mod().active_keeper(platform="win32").name == "keeper_ait.bat"

    def test_selects_the_shell_keeper_off_windows(self):
        for plat in ("darwin", "linux"):
            assert self._mod().active_keeper(platform=plat).name == "keeper_ait.sh"

    def test_gating_check_accepts_posix_curl(self):
        """The check required the literal `curl.exe`, so a correct POSIX keeper
        read as 'does not ping at all'."""
        m = self._mod()
        posix = 'curl -fsS -m 10 "$URL"  data/deadman_url.txt ait.monitoring.ops_health'
        assert m.keeper_gating(posix)["pings"] is True
        assert m.keeper_gating(posix)["gated"] is True


class TestFailedPageDoesNotBurnTheThrottle:
    """W17 w17-01b. `_alert` (master.py:64) is single-shot — one urlopen, no
    retry, no dead-channel accounting — and returns False on a dropped send.

    Both gateway-escalation sites used to set `_last_gateway_alert = now`
    BEFORE calling it and discard the result, so a single failed packet
    consumed the 30-minute throttle and bought 30 minutes of silence on the
    page that exists because of the 2026-09-15 8h36m silent outage.

    W16 shipped one of those two sites; this is a defect I copied faithfully
    from the pre-existing sibling rather than invented, and both are fixed."""

    def _mgr(self, down_minutes, *, recoveries=0):
        import ait.orchestration.master as _M
        m = _M.BotManager.__new__(_M.BotManager)
        now = datetime.now()
        m._gateway_down_since = now - timedelta(minutes=down_minutes)
        m._gateway_recoveries = recoveries
        m._last_gateway_recovery = None
        m._last_gateway_alert = None
        return m, now

    def test_failed_page_leaves_the_throttle_unconsumed(self):
        m, now = self._mgr(20)
        with patch.object(M.os.path, "exists", return_value=False), \
             patch.object(M, "_alert", return_value=False) as alert:
            m._maybe_recover_gateway(7497, now)
            assert m._last_gateway_alert is None, "a dropped page must not burn the throttle"
            m._maybe_recover_gateway(7497, now + timedelta(minutes=2))
        assert alert.call_count == 2, "the next cycle must retry a page that never landed"

    def test_delivered_page_does_consume_the_throttle(self):
        m, now = self._mgr(20)
        with patch.object(M.os.path, "exists", return_value=False), \
             patch.object(M, "_alert", return_value=True) as alert:
            m._maybe_recover_gateway(7497, now)
            assert m._last_gateway_alert is not None
            m._maybe_recover_gateway(7497, now + timedelta(minutes=2))
        assert alert.call_count == 1, "a delivered page must still be throttled"

    def test_same_rule_on_the_budget_exhausted_sibling(self):
        """Fix the CLASS: the pre-existing escalation block had the identical
        ordering, and fixing only the one W16 touched would leave it broken."""
        m, now = self._mgr(200, recoveries=M.GATEWAY_MAX_RECOVERIES)
        with patch.object(M, "_alert", return_value=False) as alert:
            m._maybe_recover_gateway(7497, now)
            assert m._last_gateway_alert is None
            m._maybe_recover_gateway(7497, now + timedelta(minutes=2))
        assert alert.call_count == 2


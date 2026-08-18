"""R16 ops fixes — supervisor (master.py) pure-python behavior.

Covers the R16 audit fixes that are testable without processes or network:
  #66  single-instance mutex (atomic, self-releasing — no TOCTOU)
  #54  per-key alert cooldowns (_alert_gate) — a new, DIFFERENT critical
       must not be starved by an unrelated earlier alert
  #29  digest sent-marker + startup catch-up window
  #59  missed-slot catch-up for the 16:30 report and Sunday backtest
       (backtest deferred, never in market hours)
  #56  _log pytest guard — test runs must not pollute the production
       orchestrator.log with fake ERROR events
  #15  TELEGRAM_DEAD marker: read every health cycle, hourly probe,
       cleared on a successful send
  #10  BotManager.start() refuses to spawn a second bot beside a
       pre-existing ait.main process

All tests monkeypatch DATA_DIR/LOGS_DIR/REPORTS_DIR to tmp_path and stub
_alert — nothing here touches the live bot, DBs, or Telegram.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from ait.orchestration import master as _master


@pytest.fixture(autouse=True)
def _clean_gate_state():
    """Per-key cooldown state and the module lock handle are process-global;
    isolate every test."""
    _master._ALERT_GATE_LAST.clear()
    yield
    _master._ALERT_GATE_LAST.clear()
    _master._release_singleton_lock()


# ==========================================================================
# 1. single-instance mutex  (R16 #66)
# ==========================================================================

class TestSingletonMutex:
    def test_acquire_release_reacquire(self, tmp_path):
        lock = tmp_path / "orch.lock"
        assert _master._acquire_singleton_lock(lock) is True
        _master._release_singleton_lock()
        assert _master._acquire_singleton_lock(lock) is True

    def test_second_acquire_same_process_fails(self, tmp_path):
        lock = tmp_path / "orch.lock"
        assert _master._acquire_singleton_lock(lock) is True
        # a second handle (even in the same process) must NOT get the lock —
        # this is the atomicity the check-then-launch bat guards lacked
        assert _master._acquire_singleton_lock(lock) is False

    def test_cross_process_exclusion_and_release_on_death(self, tmp_path):
        lock = tmp_path / "orch.lock"
        holder_code = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "from ait.orchestration import master as m\n"
            "ok = m._acquire_singleton_lock(Path(sys.argv[1]))\n"
            "print('LOCKED' if ok else 'FAIL', flush=True)\n"
            "time.sleep(60)\n"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", holder_code, str(lock)],
            stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout.readline().strip() == "LOCKED"
            # held by the other process -> we must lose the race
            assert _master._acquire_singleton_lock(lock) is False
        finally:
            proc.kill()
            proc.wait()
        # msvcrt region locks die with the process — no stale-pidfile
        # detection needed; retry briefly while the OS cleans up
        got = False
        for _ in range(20):
            got = _master._acquire_singleton_lock(lock)
            if got:
                break
            time.sleep(0.25)
        assert got is True


# ==========================================================================
# 2. per-key alert cooldown  (R16 #54)
# ==========================================================================

class TestAlertGate:
    def test_first_call_passes(self):
        assert _master._alert_gate("a", 3600, _now=1000.0) is True

    def test_same_key_throttled_within_interval(self):
        assert _master._alert_gate("a", 3600, _now=1000.0) is True
        assert _master._alert_gate("a", 3600, _now=1100.0) is False

    def test_different_key_not_starved(self):
        # THE finding: component A paging at 10:00 must not silence a NEW,
        # DIFFERENT critical from component B at 10:20
        assert _master._alert_gate("ibkr", 3600, _now=1000.0) is True
        assert _master._alert_gate("market_data", 3600, _now=2200.0) is True

    def test_expired_interval_reopens(self):
        assert _master._alert_gate("a", 3600, _now=1000.0) is True
        assert _master._alert_gate("a", 3600, _now=4601.0) is True

    def test_failed_send_still_consumes_the_slot(self):
        # the gate records the ATTEMPT, so a dead channel is probed once per
        # interval, not on every 2-min health cycle
        assert _master._alert_gate("probe", 3600, _now=1000.0) is True
        assert _master._alert_gate("probe", 3600, _now=1001.0) is False


# ==========================================================================
# 3. digest sent-marker + catch-up window  (R16 #29)
# ==========================================================================

class TestDigestCatchup:
    WED = datetime(2026, 8, 5, 10, 0)  # Wednesday

    def test_due_on_weekday_morning_without_marker(self, tmp_path):
        assert _master._digest_catchup_due(self.WED, tmp_path) is True

    def test_not_due_when_marker_exists(self, tmp_path):
        (tmp_path / "digest_sent_20260805").write_text("sent")
        assert _master._digest_catchup_due(self.WED, tmp_path) is False

    def test_window_boundaries(self, tmp_path):
        # before 09:35 the cron still fires; at/after 16:00 the PM digest
        # (16:05) is imminent with the same content
        assert _master._digest_catchup_due(datetime(2026, 8, 5, 9, 34), tmp_path) is False
        assert _master._digest_catchup_due(datetime(2026, 8, 5, 9, 35), tmp_path) is True
        assert _master._digest_catchup_due(datetime(2026, 8, 5, 15, 59), tmp_path) is True
        assert _master._digest_catchup_due(datetime(2026, 8, 5, 16, 0), tmp_path) is False

    def test_not_due_on_weekend(self, tmp_path):
        assert _master._digest_catchup_due(datetime(2026, 8, 8, 10, 0), tmp_path) is False
        assert _master._digest_catchup_due(datetime(2026, 8, 9, 10, 0), tmp_path) is False

    def test_daily_digest_writes_marker_only_on_confirmed_send(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(_master, "DATA_DIR", tmp_path)
        monkeypatch.setattr(_master, "_mirror_content_age_hours", lambda: 1.0)
        stale = tmp_path / "digest_sent_20250101"
        stale.write_text("old")
        today = datetime.now().strftime("%Y%m%d")

        # failed send -> no marker (next boot's catch-up retries)
        monkeypatch.setattr(_master, "_alert", lambda m: False)
        _master.daily_digest()
        assert not (tmp_path / f"digest_sent_{today}").exists()
        assert stale.exists()  # pruning only happens alongside a real send

        # confirmed send -> marker written, stale markers pruned
        monkeypatch.setattr(_master, "_alert", lambda m: True)
        _master.daily_digest()
        assert (tmp_path / f"digest_sent_{today}").exists()
        assert not stale.exists()


# ==========================================================================
# 4. missed-slot catch-up: 16:30 report + Sunday backtest  (R16 #59)
# ==========================================================================

class TestDailyReportCatchup:
    def test_due_after_1635_when_artifact_missing(self, tmp_path):
        assert _master._daily_report_catchup_due(
            datetime(2026, 8, 5, 17, 0), tmp_path) is True

    def test_not_due_when_artifact_exists(self, tmp_path):
        (tmp_path / "daily_20260805.json").write_text("{}")
        assert _master._daily_report_catchup_due(
            datetime(2026, 8, 5, 17, 0), tmp_path) is False

    def test_not_due_before_1635_or_weekend(self, tmp_path):
        assert _master._daily_report_catchup_due(
            datetime(2026, 8, 5, 16, 20), tmp_path) is False
        assert _master._daily_report_catchup_due(
            datetime(2026, 8, 9, 17, 0), tmp_path) is False


class TestBacktestCatchup:
    FRI = datetime(2026, 8, 7, 11, 0)  # Friday; last due slot Sun Aug 2 20:00

    def _artifact(self, tmp_path, mtime: datetime) -> Path:
        f = tmp_path / "backtest_20260803_120000.json"
        f.write_text("{}")
        ts = mtime.timestamp()
        os.utime(f, (ts, ts))
        return f

    def test_due_when_no_artifacts(self, tmp_path):
        assert _master._backtest_catchup_due(self.FRI, tmp_path) is True

    def test_not_due_when_artifact_after_slot(self, tmp_path):
        self._artifact(tmp_path, datetime(2026, 8, 3, 12, 0))
        assert _master._backtest_catchup_due(self.FRI, tmp_path) is False

    def test_due_when_artifact_predates_slot(self, tmp_path):
        self._artifact(tmp_path, datetime(2026, 7, 30, 12, 0))
        assert _master._backtest_catchup_due(self.FRI, tmp_path) is True

    def test_sunday_before_2000_uses_previous_week_slot(self, tmp_path):
        # Sunday 19:00: today's slot hasn't happened yet — due only if the
        # newest artifact predates LAST week's slot
        self._artifact(tmp_path, datetime(2026, 8, 3, 12, 0))
        assert _master._backtest_catchup_due(
            datetime(2026, 8, 9, 19, 0), tmp_path) is False
        self._artifact(tmp_path, datetime(2026, 8, 1, 12, 0))
        assert _master._backtest_catchup_due(
            datetime(2026, 8, 9, 19, 0), tmp_path) is True

    def test_run_at_deferred_10_min_off_hours(self):
        assert _master._backtest_catchup_run_at(
            datetime(2026, 8, 4, 20, 0)) == datetime(2026, 8, 4, 20, 10)
        assert _master._backtest_catchup_run_at(
            datetime(2026, 8, 8, 12, 0)) == datetime(2026, 8, 8, 12, 10)

    def test_run_at_never_lands_in_market_hours(self):
        # mid-session boot -> pushed past the 16:30 report slot
        assert _master._backtest_catchup_run_at(
            datetime(2026, 8, 4, 10, 0)) == datetime(2026, 8, 4, 16, 40)
        # pre-open boot whose +10min deferral would land at the bell
        assert _master._backtest_catchup_run_at(
            datetime(2026, 8, 4, 9, 10)) == datetime(2026, 8, 4, 16, 40)


# ==========================================================================
# 5. _log pytest guard  (R16 #56)
# ==========================================================================

class TestLogPytestGuard:
    def test_pytest_run_never_touches_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_master, "LOGS_DIR", tmp_path)
        assert os.environ.get("PYTEST_CURRENT_TEST")  # guard input present
        # the exact fake-forensics event class from the finding
        _master._log("error", "bot_launch_failed_immediately", exit_code=1)
        assert not (tmp_path / "orchestrator.log").exists()

    def test_production_run_appends(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(_master, "LOGS_DIR", tmp_path)
        monkeypatch.delenv("PYTEST_CURRENT_TEST")
        _master._log("info", "real_event", pid=42)
        content = (tmp_path / "orchestrator.log").read_text()
        assert "orchestrator.real_event" in content
        assert "pid=42" in content
        # stdout sink is unconditional in both modes
        assert "orchestrator.real_event" in capsys.readouterr().out


# ==========================================================================
# 6. TELEGRAM_DEAD marker surfacing  (R16 #15)
# ==========================================================================

class TestTelegramDeadMarker:
    def test_no_marker_no_probe(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_master, "DATA_DIR", tmp_path)
        sent = []
        monkeypatch.setattr(_master, "_alert", lambda m: sent.append(m) or True)
        _master._check_telegram_dead()
        assert not sent

    def test_failed_probe_keeps_marker_and_hourly_gate(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_master, "DATA_DIR", tmp_path)
        marker = tmp_path / "TELEGRAM_DEAD"
        marker.write_text("5 consecutive dropped alerts")
        sent = []
        monkeypatch.setattr(_master, "_alert", lambda m: sent.append(m) is None and False)
        _master._check_telegram_dead()
        assert marker.exists()  # send failed -> channel still dead
        assert len(sent) == 1
        # next health cycle (2 min later): critical logged, but NO re-probe
        _master._check_telegram_dead()
        assert len(sent) == 1

    def test_successful_probe_clears_marker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_master, "DATA_DIR", tmp_path)
        marker = tmp_path / "TELEGRAM_DEAD"
        marker.write_text("5 consecutive dropped alerts")
        monkeypatch.setattr(_master, "_alert", lambda m: True)
        # pretend the last probe was over an hour ago
        _master._ALERT_GATE_LAST["telegram_dead_probe"] = time.time() - 3601
        _master._check_telegram_dead()
        assert not marker.exists()


# ==========================================================================
# 7. foreign-bot refusal in BotManager.start()  (R16 #10)
# ==========================================================================

class TestForeignBotRefusal:
    def test_refuses_beside_preexisting_bot(self, monkeypatch):
        import ait.orchestration.gateway as gw
        mgr = _master.BotManager()
        monkeypatch.setattr(_master, "_scan_foreign_bot_pids", lambda: [9999])
        alerts = []
        monkeypatch.setattr(_master, "_alert", lambda m: alerts.append(m) or True)
        gateway_calls = []
        monkeypatch.setattr(gw, "ensure_gateway",
                            lambda port: gateway_calls.append(port) or True)
        mgr.start()
        assert mgr._proc is None          # nothing spawned
        assert not gateway_calls          # refused before any side effects
        assert any("REFUSING" in a for a in alerts)
        # health loop re-enters every 2 min: refusal repeats, page is gated
        mgr.start()
        assert sum("REFUSING" in a for a in alerts) == 1

    def test_proceeds_when_no_foreign_bot(self, monkeypatch):
        import ait.orchestration.gateway as gw
        mgr = _master.BotManager()
        monkeypatch.setattr(_master, "_scan_foreign_bot_pids", lambda: [])
        alerts = []
        monkeypatch.setattr(_master, "_alert", lambda m: alerts.append(m) or True)
        gateway_calls = []
        # gateway unavailable -> start() aborts AFTER the scan guard, without
        # ever spawning — proves the guard let a clean start proceed
        monkeypatch.setattr(gw, "ensure_gateway",
                            lambda port: gateway_calls.append(port) and False)
        mgr.start()
        assert gateway_calls == [int(os.environ.get("IBKR_PORT", "4002"))]
        assert mgr._proc is None
        assert not any("REFUSING" in a for a in alerts)

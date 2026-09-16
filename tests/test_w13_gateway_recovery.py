"""W13 (R24 ops-substrate-01): a wedged IB Gateway must not be an absorbing state.

MATERIALIZED TWICE, both times costing a trading session:
  * 2026-07-01 — 5h10m down, 3h15m of it inside RTH.
  * 2026-09-08 — a Gateway JVM started at 08:56 and NEVER opened port 4002.
    The supervisor logged `bot_restart_deferred_gateway_down` every two
    minutes until a human intervened at 12:17: three and a half hours of a
    trading day with two open condors and no exit engine running. The JVM
    EXISTED, so no process-level check saw anything wrong — only the port was
    dead, and nothing was watching the port except the deferral itself.

The supervisor now attempts recovery, but conservatively: a grace period (IB's
own nightly maintenance recovers unaided and must not be fought), a cooldown,
a bounded budget, then escalation to a human.

Every test EXECUTES the real BotManager method.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ait.orchestration import master as M


def _mgr(down_minutes: float, *, recoveries: int = 0,
         last_recovery_minutes_ago: float | None = None):
    """A real BotManager with only the gateway-recovery state populated."""
    m = M.BotManager.__new__(M.BotManager)
    now = datetime.now()
    m._gateway_down_since = now - timedelta(minutes=down_minutes)
    m._gateway_recoveries = recoveries
    m._last_gateway_recovery = (
        None if last_recovery_minutes_ago is None
        else now - timedelta(minutes=last_recovery_minutes_ago))
    m._last_gateway_alert = None
    return m, now


class TestGracePeriodProtectsNightlyMaintenance:
    def test_short_outage_does_not_intervene(self):
        """IB takes the Gateway down 30-60 min nightly and brings it back.
        Fighting that would thrash a healthy system."""
        m, now = _mgr(down_minutes=2)
        with patch.object(M.subprocess, "Popen") as popen:
            m._maybe_recover_gateway(4002, now)
        popen.assert_not_called()
        assert m._gateway_recoveries == 0

    def test_intervenes_once_past_the_grace_period(self):
        m, now = _mgr(down_minutes=20)
        with patch.object(M, "_gateway_process_alive", return_value=False), \
             patch.object(M.os.path, "exists", return_value=True), \
             patch.object(M.subprocess, "Popen") as popen, \
             patch.object(M, "_alert"):
            m._maybe_recover_gateway(4002, now)
        popen.assert_called_once()
        assert m._gateway_recoveries == 1

    def test_grace_boundary_is_the_configured_threshold(self):
        just_under = M.GATEWAY_RECOVERY_AFTER_S / 60.0 - 0.5
        m, now = _mgr(down_minutes=just_under)
        with patch.object(M.subprocess, "Popen") as popen:
            m._maybe_recover_gateway(4002, now)
        popen.assert_not_called()


class TestWedgedProcessIsKilledFirst:
    def test_live_but_wedged_jvm_is_killed_before_relaunch(self):
        """THE 2026-09-08 SHAPE: a JVM that never opened the port holds it
        against a relaunch, so starting on top of it cannot work."""
        m, now = _mgr(down_minutes=20)
        with patch.object(M, "_gateway_process_alive", return_value=True), \
             patch.object(M.os.path, "exists", return_value=True), \
             patch.object(M.subprocess, "run") as run, \
             patch.object(M.subprocess, "Popen") as popen, \
             patch.object(M.time, "sleep"), patch.object(M, "_alert"):
            m._maybe_recover_gateway(4002, now)
        assert run.called, "wedged JVM was not killed"
        assert "Stop-Process" in " ".join(run.call_args[0][0])
        popen.assert_called_once()

    def test_absent_process_is_simply_started(self):
        m, now = _mgr(down_minutes=20)
        with patch.object(M, "_gateway_process_alive", return_value=False), \
             patch.object(M.os.path, "exists", return_value=True), \
             patch.object(M.subprocess, "run") as run, \
             patch.object(M.subprocess, "Popen") as popen, \
             patch.object(M, "_alert"):
            m._maybe_recover_gateway(4002, now)
        run.assert_not_called()
        popen.assert_called_once()


class TestCooldownAndBudget:
    def test_cooldown_blocks_a_rapid_second_attempt(self):
        """A Gateway that cannot come up (expired login, 2FA prompt) must not
        be thrashed every polling cycle."""
        m, now = _mgr(down_minutes=30, recoveries=1,
                      last_recovery_minutes_ago=2)
        with patch.object(M.subprocess, "Popen") as popen:
            m._maybe_recover_gateway(4002, now)
        popen.assert_not_called()

    def test_attempt_allowed_once_the_cooldown_expires(self):
        m, now = _mgr(down_minutes=90, recoveries=1,
                      last_recovery_minutes_ago=45)
        with patch.object(M, "_gateway_process_alive", return_value=False), \
             patch.object(M.os.path, "exists", return_value=True), \
             patch.object(M.subprocess, "Popen") as popen, \
             patch.object(M, "_alert"):
            m._maybe_recover_gateway(4002, now)
        popen.assert_called_once()
        assert m._gateway_recoveries == 2

    def test_budget_exhausted_escalates_to_a_human(self):
        """Looping forever IS the original defect. When automation is out of
        ideas it must say so."""
        m, now = _mgr(down_minutes=200,
                      recoveries=M.GATEWAY_MAX_RECOVERIES,
                      last_recovery_minutes_ago=60)
        with patch.object(M.subprocess, "Popen") as popen, \
             patch.object(M, "_alert") as alert:
            m._maybe_recover_gateway(4002, now)
        popen.assert_not_called()
        alert.assert_called_once()
        msg = alert.call_args[0][0]
        assert "MANUAL ACTION NEEDED" in msg
        assert "2FA" in msg or "login" in msg


class TestSafety:
    def test_missing_start_script_is_reported_AND_escalated(self):
        """R25 plat-01. This test used to assert only what did NOT happen —
        no Popen, counter untouched — and so it PERMITTED total silence.

        That silence was the defect: the early return sat before the counter
        increment, so `_gateway_recoveries` never moved and the escalation
        branch was unreachable on any host without the start script. On
        2026-09-15 TWS was closed by accident and the supervisor deferred for
        8h36m behind 134 `gateway_recovery_unavailable` lines and zero pages.

        The two original assertions still hold — there is genuinely nothing to
        launch and no attempt was made. What was missing is the page."""
        m, now = _mgr(down_minutes=20)
        with patch.object(M.os.path, "exists", return_value=False), \
             patch.object(M.subprocess, "Popen") as popen, \
             patch.object(M, "_alert") as alert:
            m._maybe_recover_gateway(4002, now)
        popen.assert_not_called()
        assert m._gateway_recoveries == 0      # no attempt was made
        alert.assert_called_once()             # but a human MUST be told
        msg = alert.call_args[0][0]
        assert "MANUAL ACTION NEEDED" in msg
        assert "4002" in msg

    def test_unavailable_recovery_page_is_throttled(self):
        """The health loop runs every 2 min; the page must not ride it.
        Shares the 30-minute throttle the budget-exhausted branch uses."""
        m, now = _mgr(down_minutes=20)
        with patch.object(M.os.path, "exists", return_value=False), \
             patch.object(M.subprocess, "Popen"), \
             patch.object(M, "_alert") as alert:
            m._maybe_recover_gateway(4002, now)
            m._maybe_recover_gateway(4002, now + timedelta(minutes=2))
            m._maybe_recover_gateway(4002, now + timedelta(minutes=10))
        assert alert.call_count == 1

    def test_unavailable_recovery_pages_again_once_throttle_expires(self):
        """A still-down gateway half an hour later deserves a fresh page."""
        m, now = _mgr(down_minutes=20)
        with patch.object(M.os.path, "exists", return_value=False), \
             patch.object(M.subprocess, "Popen"), \
             patch.object(M, "_alert") as alert:
            m._maybe_recover_gateway(4002, now)
            m._maybe_recover_gateway(4002, now + timedelta(minutes=31))
        assert alert.call_count == 2

    def test_a_failing_restart_never_kills_the_supervisor(self):
        """The supervisor is the last line of defence; recovery must not
        become the thing that takes it down."""
        m, now = _mgr(down_minutes=20)
        with patch.object(M, "_gateway_process_alive", return_value=False), \
             patch.object(M.os.path, "exists", return_value=True), \
             patch.object(M.subprocess, "Popen",
                          side_effect=OSError("boom")), \
             patch.object(M, "_alert"):
            m._maybe_recover_gateway(4002, now)   # must not raise

    def test_budget_and_timer_reset_after_recovery(self):
        """A later, unrelated outage deserves fresh attempts."""
        m, _ = _mgr(down_minutes=200, recoveries=3)
        # the healthy branch clears both
        m._gateway_down_since = None
        m._gateway_recoveries = 0
        assert m._gateway_recoveries == 0 and m._gateway_down_since is None


class TestTheOutageThatMotivatedThis:
    def test_three_and_a_half_hours_would_have_been_ten_minutes(self):
        """2026-09-08: down 08:56-12:17 (201 min) with a wedged JVM. Under
        the old code: zero interventions. Now: an attempt at the grace
        boundary."""
        m, now = _mgr(down_minutes=201)
        with patch.object(M, "_gateway_process_alive", return_value=True), \
             patch.object(M.os.path, "exists", return_value=True), \
             patch.object(M.subprocess, "run"), \
             patch.object(M.subprocess, "Popen") as popen, \
             patch.object(M.time, "sleep"), patch.object(M, "_alert"):
            m._maybe_recover_gateway(4002, now)
        popen.assert_called_once()
        assert M.GATEWAY_RECOVERY_AFTER_S <= 900, (
            "grace period must be well inside a trading session")

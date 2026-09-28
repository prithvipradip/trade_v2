"""Tests for health monitoring watchdog."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import ait.monitoring.watchdog as _wd
from ait.monitoring.watchdog import ComponentStatus, Watchdog


@pytest.fixture
def watchdog():
    return Watchdog(heartbeat_timeout=5.0, max_memory_mb=1000.0, error_threshold=3)


class TestHeartbeat:
    """Test heartbeat monitoring."""

    def test_initial_status_unknown(self, watchdog):
        watchdog.register_component("test")
        health = watchdog.get_health()
        assert health.components["test"].status == ComponentStatus.UNKNOWN

    def test_heartbeat_sets_healthy(self, watchdog):
        watchdog.heartbeat("test")
        health = watchdog.get_health()
        assert health.components["test"].status == ComponentStatus.HEALTHY

    def test_auto_register_on_heartbeat(self, watchdog):
        watchdog.heartbeat("new_component")
        assert "new_component" in watchdog.get_health().components


class TestErrorTracking:
    """Test error recording and status degradation."""

    def test_error_degrades_status(self, watchdog):
        watchdog.register_component("test")
        watchdog.heartbeat("test")

        # Threshold/2 errors → degraded
        watchdog.record_error("test", "error 1")
        watchdog.record_error("test", "error 2")
        health = watchdog.get_health()
        assert health.components["test"].status == ComponentStatus.DEGRADED

    def test_many_errors_marks_down(self, watchdog):
        watchdog.register_component("test")

        for i in range(3):
            watchdog.record_error("test", f"error {i}")

        health = watchdog.get_health()
        assert health.components["test"].status == ComponentStatus.DOWN

    def test_heartbeat_does_not_reset_errors(self, watchdog):
        # R5 audit: this test used to assert error_count == 0 after a
        # heartbeat — enshrining the exact bug Round 3 fixed (a hot error
        # loop that also heartbeats could never trip the watchdog). The
        # fixed behavior: heartbeat refreshes liveness, error count persists.
        watchdog.register_component("test")
        watchdog.record_error("test", "error")
        watchdog.record_error("test", "error")
        watchdog.heartbeat("test")

        health = watchdog.get_health()
        assert health.components["test"].error_count == 2


class TestOverallHealth:
    """Test overall health status computation."""

    def test_all_healthy(self, watchdog, monkeypatch):
        """Pinned clock AND pinned memory: get_health() compares real
        time.time() against the fixture's tight 5.0s heartbeat_timeout, and
        separately reads the REAL pytest process's RSS against the
        fixture's max_memory_mb=1000.0 -- both real, shared resources a
        loaded full-suite run can push past a margin this test never meant
        to exercise (W30's original diagnosis named only the clock; a
        second full-lane run after that fix still failed here, this time on
        `high_memory_usage` at memory_mb=1001.0 -- the same class of bug,
        one mechanism deeper). Neither mechanism is what THIS test is
        about -- that's TestPerComponentHeartbeatTimeout's job for the
        clock, with its own deliberate real-clock offsets (untouched: this
        patch replaces watchdog.py's own `time`/`_get_memory_mb`
        references on this ONE instance, not global state)."""
        monkeypatch.setattr(_wd, "time", SimpleNamespace(time=lambda: 1_700_000_000.0))
        monkeypatch.setattr(watchdog, "_get_memory_mb", lambda: 100.0)
        watchdog.heartbeat("a")
        watchdog.heartbeat("b")
        health = watchdog.get_health()
        assert health.status == ComponentStatus.HEALTHY

    def test_one_down_marks_overall_down(self, watchdog, monkeypatch):
        """Same clock- and memory-pinning as test_all_healthy, same reason."""
        monkeypatch.setattr(_wd, "time", SimpleNamespace(time=lambda: 1_700_000_000.0))
        monkeypatch.setattr(watchdog, "_get_memory_mb", lambda: 100.0)
        watchdog.heartbeat("a")
        for i in range(3):
            watchdog.record_error("b", f"err {i}")

        health = watchdog.get_health()
        assert health.status == ComponentStatus.DOWN

    def test_uptime_tracked(self, watchdog):
        health = watchdog.get_health()
        assert health.uptime_seconds >= 0


class TestLatency:
    """Test latency tracking."""

    def test_record_latency(self, watchdog):
        watchdog.record_latency("api", 150.0)
        health = watchdog.get_health()
        assert health.components["api"].latency_ms == 150.0


class TestRecovery:
    """Test auto-recovery."""

    @pytest.mark.asyncio
    async def test_recovery_callback(self, watchdog):
        recovered = []

        async def recover():
            recovered.append(True)

        watchdog.register_component("test")
        watchdog.on_recovery("test", recover)

        for i in range(3):
            watchdog.record_error("test", f"err {i}")

        result = await watchdog.check_and_recover()
        assert "test" in result
        assert len(recovered) == 1


class TestSummary:
    """Test health summary."""

    def test_get_summary(self, watchdog):
        watchdog.heartbeat("ibkr")
        watchdog.heartbeat("trading_loop")
        summary = watchdog.get_summary()

        assert "ibkr" in summary
        assert "trading_loop" in summary
        assert "healthy" in summary


class TestPerComponentHeartbeatTimeout:
    """Live incident 2026-09-17: market_data heartbeats at most once per
    SYMBOL inside a full scan cycle (scan_interval_seconds, 300s on the live
    book) -- never on the 30s fast-loop tick that keeps trading_loop/ibkr
    fresh against the shared 120s default. Measured on the live book:
    consecutive heartbeats landed 201-434s apart, so EVERY gap exceeded the
    shared timeout and check_and_recover (running every ~5th fast-tick) paged
    "market_data is DOWN and recovery failed" on an otherwise perfectly
    healthy scan loop -- repeatedly, because market_data has no recovery
    callback to ever clear it (see TestNoRecoveryCallbackRepagesUntilHealthy).
    """

    def test_negative_control_the_shared_default_would_have_alarmed(self, watchdog):
        """Reproduces the live defect exactly, without the fix.

        `watchdog` (the fixture) uses the OLD single-timeout shape: no
        per-component override. A gap smaller than the SMALLEST one actually
        measured live (201s) must already read DOWN against the 5.0s fixture
        timeout, proving the shared-timeout behaviour this wave replaces was
        real, not hypothetical.
        """
        watchdog.heartbeat("market_data")
        watchdog._components["market_data"].last_heartbeat = time.time() - 201
        health = watchdog.get_health()
        assert health.components["market_data"].status == ComponentStatus.DOWN, (
            "premise broken: a 201s gap no longer trips the shared default, "
            "so this control no longer demonstrates the live defect"
        )

    def test_a_component_with_a_wider_registered_timeout_tolerates_its_own_cadence(self):
        """The fix. Sized to the real evidence: every live gap fell inside
        900s (3x the 300s scan interval); none came close to it."""
        wd = Watchdog(heartbeat_timeout=120.0)
        wd.register_component("market_data", heartbeat_timeout=900.0)
        wd.heartbeat("market_data")

        wd._components["market_data"].last_heartbeat = time.time() - 434  # worst live gap
        health = wd.get_health()

        assert health.components["market_data"].status == ComponentStatus.HEALTHY, (
            "the real worst-case gap measured on the live book (434s) must not "
            "trip a component explicitly given margin for it"
        )

    def test_the_override_still_catches_a_genuine_stall(self):
        """A guard against over-correcting into never alerting at all."""
        wd = Watchdog(heartbeat_timeout=120.0)
        wd.register_component("market_data", heartbeat_timeout=900.0)
        wd.heartbeat("market_data")

        wd._components["market_data"].last_heartbeat = time.time() - 1000
        health = wd.get_health()

        assert health.components["market_data"].status == ComponentStatus.DOWN

    def test_unregistered_components_are_unaffected_by_the_new_parameter(self):
        """trading_loop/ibkr must keep exactly today's behaviour: no override
        was requested for them, so the shared instance default still applies
        and a real stall on their own (much faster) cadence still pages."""
        wd = Watchdog(heartbeat_timeout=120.0)
        wd.register_component("trading_loop")
        wd.register_component("market_data", heartbeat_timeout=900.0)
        wd.heartbeat("trading_loop")
        wd.heartbeat("market_data")

        # The exact live worst-case gap: harmless for market_data (asserted
        # above), but trading_loop's own cadence is 30s, so 434s is a real
        # stall and must still be caught.
        wd._components["trading_loop"].last_heartbeat = time.time() - 434
        wd._components["market_data"].last_heartbeat = time.time() - 434
        health = wd.get_health()

        assert health.components["trading_loop"].status == ComponentStatus.DOWN
        assert health.components["market_data"].status == ComponentStatus.HEALTHY

    def test_register_component_still_accepts_the_old_single_argument_call(self):
        """Backward compatibility: every existing call site
        (register_component("name")) must keep working unmodified."""
        wd = Watchdog(heartbeat_timeout=120.0)
        wd.register_component("ibkr")  # no keyword arg, exactly as today
        assert wd._components["ibkr"].heartbeat_timeout is None

    def test_heartbeat_auto_registration_does_not_set_an_override(self):
        """heartbeat() on a never-registered component auto-registers it --
        that path must keep resolving to the instance default, not silently
        acquire a timeout from nowhere."""
        wd = Watchdog(heartbeat_timeout=120.0)
        wd.heartbeat("never_registered")
        assert wd._components["never_registered"].heartbeat_timeout is None


class TestNoRecoveryCallbackRepagesUntilHealthy:
    """market_data (and any component with no recovery callback) has nothing
    that can ever mark it recovered except a fresh heartbeat -- confirms the
    fix in TestPerComponentHeartbeatTimeout is what actually stops the page,
    not a change to the alert/recovery machinery itself (there is none)."""

    @pytest.mark.asyncio
    async def test_a_down_component_with_no_recovery_callback_alerts(self):
        alerts = []

        async def alert_cb(msg):
            alerts.append(msg)

        wd = Watchdog(heartbeat_timeout=120.0)
        wd.set_alert_callback(alert_cb)
        wd.register_component("market_data")  # no on_recovery(), no override
        wd.heartbeat("market_data")
        wd._components["market_data"].last_heartbeat = time.time() - 200

        await wd.check_and_recover()

        assert len(alerts) == 1
        assert "market_data is DOWN and recovery failed" in alerts[0]

    @pytest.mark.asyncio
    async def test_the_sized_override_prevents_the_alert_for_the_same_gap(self):
        """The end-to-end proof: identical scenario, only the registration
        differs, and the live alert simply does not fire."""
        alerts = []

        async def alert_cb(msg):
            alerts.append(msg)

        wd = Watchdog(heartbeat_timeout=120.0)
        wd.set_alert_callback(alert_cb)
        wd.register_component("market_data", heartbeat_timeout=900.0)
        wd.heartbeat("market_data")
        wd._components["market_data"].last_heartbeat = time.time() - 200

        await wd.check_and_recover()

        assert alerts == []

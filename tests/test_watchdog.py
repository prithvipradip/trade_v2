"""Tests for health monitoring watchdog."""

from __future__ import annotations

import asyncio
import time

import pytest

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

    def test_heartbeat_resets_errors(self, watchdog):
        watchdog.register_component("test")
        watchdog.record_error("test", "error")
        watchdog.record_error("test", "error")
        watchdog.heartbeat("test")

        health = watchdog.get_health()
        assert health.components["test"].status == ComponentStatus.HEALTHY
        assert health.components["test"].error_count == 0


class TestOverallHealth:
    """Test overall health status computation."""

    def test_all_healthy(self, watchdog):
        watchdog.heartbeat("a")
        watchdog.heartbeat("b")
        health = watchdog.get_health()
        assert health.status == ComponentStatus.HEALTHY

    def test_one_down_marks_overall_down(self, watchdog):
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

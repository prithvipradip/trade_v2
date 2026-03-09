"""Health monitoring and watchdog — keeps the bot alive and healthy.

Monitors:
- IBKR connection health
- Trading loop heartbeat
- Memory usage
- API response times
- Error rates

Provides auto-recovery for transient failures.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum

from ait.utils.logging import get_logger

log = get_logger("monitoring.watchdog")


class ComponentStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health of a single component."""

    name: str
    status: ComponentStatus = ComponentStatus.UNKNOWN
    last_heartbeat: float = 0.0
    error_count: int = 0
    last_error: str = ""
    latency_ms: float = 0.0


@dataclass
class HealthStatus:
    """Overall system health."""

    status: ComponentStatus
    components: dict[str, ComponentHealth] = field(default_factory=dict)
    memory_mb: float = 0.0
    uptime_seconds: float = 0.0
    trading_loop_alive: bool = False
    ibkr_connected: bool = False


class Watchdog:
    """Monitors system health and triggers recovery actions."""

    def __init__(
        self,
        heartbeat_timeout: float = 120.0,
        max_memory_mb: float = 500.0,
        error_threshold: int = 10,
    ) -> None:
        self._heartbeat_timeout = heartbeat_timeout
        self._max_memory_mb = max_memory_mb
        self._error_threshold = error_threshold

        self._components: dict[str, ComponentHealth] = {}
        self._start_time = time.time()
        self._recovery_callbacks: dict[str, list] = {}
        self._alert_callback = None

    def register_component(self, name: str) -> None:
        """Register a component to monitor."""
        self._components[name] = ComponentHealth(name=name)

    def heartbeat(self, component: str) -> None:
        """Record a heartbeat from a component."""
        if component not in self._components:
            self.register_component(component)
        self._components[component].last_heartbeat = time.time()
        self._components[component].status = ComponentStatus.HEALTHY
        self._components[component].error_count = 0

    def record_error(self, component: str, error: str) -> None:
        """Record an error for a component."""
        if component not in self._components:
            self.register_component(component)

        comp = self._components[component]
        comp.error_count += 1
        comp.last_error = error

        if comp.error_count >= self._error_threshold:
            comp.status = ComponentStatus.DOWN
            log.critical("component_down", component=component, errors=comp.error_count)
        elif comp.error_count >= self._error_threshold // 2:
            comp.status = ComponentStatus.DEGRADED
            log.warning("component_degraded", component=component, errors=comp.error_count)

    def record_latency(self, component: str, latency_ms: float) -> None:
        """Record API latency for a component."""
        if component not in self._components:
            self.register_component(component)
        self._components[component].latency_ms = latency_ms

        if latency_ms > 5000:
            log.warning("high_latency", component=component, latency_ms=latency_ms)

    def set_alert_callback(self, callback) -> None:
        """Set async callback for health alerts."""
        self._alert_callback = callback

    def on_recovery(self, component: str, callback) -> None:
        """Register a recovery callback for a component."""
        self._recovery_callbacks.setdefault(component, []).append(callback)

    def get_health(self) -> HealthStatus:
        """Get current system health status."""
        now = time.time()

        # Check heartbeat timeouts
        for comp in self._components.values():
            if comp.last_heartbeat > 0:
                age = now - comp.last_heartbeat
                if age > self._heartbeat_timeout:
                    comp.status = ComponentStatus.DOWN

        # Memory check
        memory_mb = self._get_memory_mb()

        # Overall status
        statuses = [c.status for c in self._components.values()]
        if ComponentStatus.DOWN in statuses:
            overall = ComponentStatus.DOWN
        elif ComponentStatus.DEGRADED in statuses:
            overall = ComponentStatus.DEGRADED
        elif all(s == ComponentStatus.HEALTHY for s in statuses) and statuses:
            overall = ComponentStatus.HEALTHY
        else:
            overall = ComponentStatus.UNKNOWN

        # Memory warning
        if memory_mb > self._max_memory_mb:
            overall = ComponentStatus.DEGRADED
            log.warning("high_memory_usage", memory_mb=memory_mb, limit_mb=self._max_memory_mb)

        return HealthStatus(
            status=overall,
            components=dict(self._components),
            memory_mb=memory_mb,
            uptime_seconds=now - self._start_time,
            trading_loop_alive=self._is_component_alive("trading_loop"),
            ibkr_connected=self._is_component_alive("ibkr"),
        )

    async def check_and_recover(self) -> list[str]:
        """Check health and trigger recovery for down components.

        Returns list of components that recovery was attempted for.
        """
        health = self.get_health()
        recovered = []

        for name, comp in health.components.items():
            if comp.status == ComponentStatus.DOWN:
                callbacks = self._recovery_callbacks.get(name, [])
                for cb in callbacks:
                    try:
                        log.info("attempting_recovery", component=name)
                        if asyncio.iscoroutinefunction(cb):
                            await cb()
                        else:
                            cb()
                        recovered.append(name)
                        comp.error_count = 0
                        comp.status = ComponentStatus.HEALTHY
                        log.info("recovery_successful", component=name)
                    except Exception as e:
                        log.error("recovery_failed", component=name, error=str(e))

                # Alert if recovery didn't work
                if name not in recovered and self._alert_callback:
                    try:
                        await self._alert_callback(
                            f"ALERT: {name} is DOWN and recovery failed. "
                            f"Errors: {comp.error_count}, Last: {comp.last_error}"
                        )
                    except Exception:
                        pass

        return recovered

    def get_summary(self) -> str:
        """Get a human-readable health summary."""
        health = self.get_health()
        lines = [f"System: {health.status.value} | Memory: {health.memory_mb:.0f}MB | Uptime: {health.uptime_seconds/3600:.1f}h"]

        for name, comp in health.components.items():
            line = f"  {name}: {comp.status.value}"
            if comp.error_count > 0:
                line += f" (errors: {comp.error_count})"
            if comp.latency_ms > 0:
                line += f" (latency: {comp.latency_ms:.0f}ms)"
            lines.append(line)

        return "\n".join(lines)

    def _is_component_alive(self, component: str) -> bool:
        """Check if a specific component is alive."""
        comp = self._components.get(component)
        if not comp:
            return False
        return comp.status in (ComponentStatus.HEALTHY, ComponentStatus.DEGRADED)

    @staticmethod
    def _get_memory_mb() -> float:
        """Get current process memory usage in MB."""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / (1024 * 1024)  # macOS returns bytes
        except (ImportError, AttributeError):
            return 0.0

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
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from typing import TYPE_CHECKING

from ait.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ait.monitoring.ops_health import HealthStatePublisher

log = get_logger("monitoring.watchdog")

#: How many recent component errors the watchdog keeps for the dashboard's
#: "Recent Errors" panel.  Bounded because it is serialised into bot_state.
MAX_RECENT_ERRORS = 50


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
        *,
        state_publisher: HealthStatePublisher | None = None,
    ) -> None:
        self._heartbeat_timeout = heartbeat_timeout
        self._max_memory_mb = max_memory_mb
        self._error_threshold = error_threshold

        self._components: dict[str, ComponentHealth] = {}
        self._start_time = time.time()
        self._recovery_callbacks: dict[str, list] = {}
        self._alert_callback = None
        # W6 string-contracts-5 / db-contracts-6: the dashboard's System Health
        # tab was built against a watchdog state-publishing contract whose
        # writer half never existed, so every panel took its fallback branch on
        # every render and the error panel's fallback was a green
        # st.success('No errors logged').  The health this class already
        # computes in memory is now PERSISTED under exactly the bot_state keys
        # the dashboard reads.  Publishing is best-effort and fully isolated:
        # it can never raise into the trading loop, never creates a database,
        # and is off by default under pytest (see HealthStatePublisher).
        # Imported lazily so `python -m ait.monitoring.ops_health` (the CLI
        # keeper_ait.bat calls) does not re-enter a half-initialised module.
        self._recent_errors: deque[dict] = deque(maxlen=MAX_RECENT_ERRORS)
        if state_publisher is None:
            from ait.monitoring.ops_health import HealthStatePublisher as _HSP
            state_publisher = _HSP()
        self._state_publisher = state_publisher

    def register_component(self, name: str) -> None:
        """Register a component to monitor."""
        self._components[name] = ComponentHealth(name=name)

    def heartbeat(self, component: str) -> None:
        """Record a heartbeat from a component."""
        if component not in self._components:
            self.register_component(component)
        self._components[component].last_heartbeat = time.time()
        # R8: a heartbeat proves LIVENESS, not health — during the 07-10
        # incident the looping (alive) bot heartbeat every 30s, wiping the
        # DOWN status its own errors had just set. Status only recovers
        # when errors stop crossing the threshold (note_success below).
        if self._components[component].error_count < self._error_threshold:
            self._components[component].status = ComponentStatus.HEALTHY
        # Deep-audit BC-H2a: heartbeat used to zero error_count — with a 30s
        # heartbeat and a 10-error threshold the counter oscillated 0<->1 and
        # could NEVER trip. Errors now persist until explicit recovery.
        # W6: the 30s heartbeat is the only call that runs on a fixed cadence,
        # so it is what keeps the dashboard's health channel FRESH (throttled
        # to one write/minute inside the publisher). Without it the channel
        # would only refresh on the ~5-minute check_and_recover tick and the
        # dashboard would render "STALE" through healthy sessions.
        self._publish_state()

    def note_success(self, component: str) -> None:
        """R8: consecutive-error semantics — a clean pass resets the counter
        (the lifetime counter meant 10 sporadic errors across a session put a
        component permanently DOWN, flooding component_down logs)."""
        comp = self._components.get(component)
        if comp is not None:
            comp.error_count = 0
            comp.status = ComponentStatus.HEALTHY
            # W6: a clean pass is exactly when a red component turns green;
            # publish it so the dashboard is not stuck showing the last error.
            self._publish_state()

    def record_error(self, component: str, error: str) -> None:
        """Record an error for a component."""
        if component not in self._components:
            self.register_component(component)

        comp = self._components[component]
        comp.error_count += 1
        comp.last_error = error
        # W6 string-contracts-5: the dashboard's Recent Errors panel had NO
        # producer at all, so its fallback (a green "No errors logged") was the
        # only thing it ever rendered — including through the documented
        # full-day outage. This ring buffer is that producer.
        self._recent_errors.append({
            "time": datetime.now().isoformat(timespec="seconds"),
            "component": component,
            "error": str(error)[:500],
            "error_count": comp.error_count,
        })

        if comp.error_count >= self._error_threshold:
            comp.status = ComponentStatus.DOWN
            log.critical("component_down", component=component, errors=comp.error_count)
        elif comp.error_count >= self._error_threshold // 2:
            comp.status = ComponentStatus.DEGRADED
            log.warning("component_degraded", component=component, errors=comp.error_count)

        # A component that just went DEGRADED/DOWN must reach the operator
        # surface immediately, not on the next throttle window.
        self._publish_state(
            force=comp.status in (ComponentStatus.DEGRADED, ComponentStatus.DOWN)
        )

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
        """Get current system health status (and refresh the bot_state channel)."""
        health = self._compute_health()
        self._publish_state(health)
        return health

    def recent_errors(self) -> list[dict]:
        """Bounded, newest-last list of component errors seen this session."""
        return list(self._recent_errors)

    def _publish_state(self, health: HealthStatus | None = None, *,
                       force: bool = False) -> None:
        """Persist health into bot_state. Best-effort; never raises."""
        pub = self._state_publisher
        if pub is None or not pub.enabled:
            return
        if not force and not pub.due():
            return
        try:
            if health is None:
                health = self._compute_health()
            pub.publish(health, list(self._recent_errors), force=force)
        except Exception:  # noqa: BLE001 — monitoring must never kill the loop
            pass

    def _compute_health(self) -> HealthStatus:
        """Compute health WITHOUT publishing (the publish path calls this)."""
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

        if recovered:
            # A recovery flips components back to HEALTHY in memory; push that
            # to the operator surface immediately instead of leaving a red
            # component on screen for the rest of the throttle window.
            self._publish_state(force=True)
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
        """Get current process memory usage in MB.

        W6: delegates to ops_health.process_memory so the number rendered in
        get_summary() and the number written to bot_state['system_memory_usage']
        come from ONE code path (they were two, and the resource-module
        fallback here divided KB by 1024*1024 — 1024x too small)."""
        from ait.monitoring.ops_health import process_memory
        return float(process_memory().get("rss_mb") or 0.0)

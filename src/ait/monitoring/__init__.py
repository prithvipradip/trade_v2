"""Monitoring package.

ops_health is NOT imported here: keeper_ait.bat runs
`python -m ait.monitoring.ops_health liveness`, and an eager package-level
import makes runpy warn that the module was already in sys.modules.  Import it
directly (`from ait.monitoring.ops_health import ...`).
"""

from ait.monitoring.watchdog import HealthStatus, Watchdog

__all__ = ["Watchdog", "HealthStatus"]

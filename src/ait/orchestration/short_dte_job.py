"""W76: run the 0-6 DTE friction collector from the supervisor, unattended.

Why this exists. IB keeps no bid/ask history for an expired contract, so the short-DTE
collection (``scripts/calibrate_option_spreads.py --source ib-bars --short-dte``) has to
run once per RTH session or that day's data is gone for good. The operator is not always at
the machine at 15:30 ET; the supervisor already lives all day. This module is the whole job,
with every external dependency injected so it is exercised in tests without a broker, a
clock, a calendar or a subprocess; ``master.collect_short_dte_friction`` is a thin wrapper.

What it never does: place or touch an order, use the bot's client id (or its fallback
ladder), run when the broker port is closed, or run twice at once.

Outcomes (``JobOutcome.status``):
    skipped_not_trading_day | skipped_gateway_down | refused_client_id   (nothing was started)
    ok | empty | failed | timeout | idle_timeout                          (a run was started)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

# Statuses that deserve a Telegram message: the day's data is missing or the run is unhealthy.
ALERT_STATUSES = frozenset({
    "skipped_gateway_down", "refused_client_id", "empty", "failed", "timeout", "idle_timeout",
})
STATUS_FILE_NAME = "short_dte_collection_status.json"


@dataclass
class JobOutcome:
    status: str
    detail: str = ""
    stored: int = 0
    planned: int = 0
    bands: dict[str, dict[str, str]] = field(default_factory=dict)  # {symbol: {band: "7/15"}}
    log_path: str | None = None

    @property
    def alert_worthy(self) -> bool:
        return self.status in ALERT_STATUSES


def reserved_client_ids(base: int) -> set[int]:
    """The bot's own ids: its configured base plus the +100..+104 fallback ladder
    (``IBKRClient.connect`` on error 326)."""
    return {int(base)} | {int(base) + 100 + i for i in range(5)}


def build_command(cfg: Any, *, python: str, script: Path, status_path: Path,
                  min_sessions: int) -> list[str]:
    """The collector's argv. The port is NOT passed: the script resolves it from IBKR_PORT
    (.env), the same value the bot and the gateway probe below use."""
    return [
        python, "-u", str(script),
        "--source", "ib-bars", "--short-dte",
        "--symbols", *cfg.short_dte_symbols,
        "--client-id", str(cfg.short_dte_client_id),
        "--short-dte-max", str(cfg.short_dte_max_dte),
        "--short-dte-duration", str(cfg.short_dte_duration),
        "--min-sessions", str(int(min_sessions)),
        "--status-file", str(status_path),
    ]


def _tail(path: Path, n: int = 6) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return " | ".join(x.strip() for x in lines[-n:] if x.strip())


def _read_status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _summarise(status: dict) -> tuple[int, int, dict[str, dict[str, str]]]:
    stored = planned = 0
    bands: dict[str, dict[str, str]] = {}
    for sym, blk in (status.get("symbols") or {}).items():
        stored += int(blk.get("stored") or 0)
        planned += int(blk.get("planned") or 0)
        bands[sym] = {b: f"{v.get('sessions', 0)}/{v.get('needed', '?')}"
                      for b, v in (blk.get("bands") or {}).items()}
    return stored, planned, bands


def _stop(proc: Any, sleep: Callable[[float], None]) -> None:
    """terminate, then kill: never leave a collector holding a broker client id."""
    try:
        proc.terminate()
    except Exception:  # noqa: BLE001
        pass
    for _ in range(10):
        if proc.poll() is not None:
            return
        sleep(0.5)
    try:
        proc.kill()
    except Exception:  # noqa: BLE001
        pass


def run_collection(
    cfg: Any, *, root: Path, data_dir: Path, logs_dir: Path, min_sessions: int,
    ibkr_port: int, base_client_id: int, is_trading_day: Callable[[date], bool],
    gateway_listening: Callable[[int], bool], today: date,
    popen: Callable[..., Any] = subprocess.Popen, python: str | None = None,
    clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
    poll_s: float = 5.0,
) -> JobOutcome:
    """Run one collection and classify it. Never raises for an expected failure."""
    if not is_trading_day(today):
        return JobOutcome("skipped_not_trading_day", f"{today} is not an NYSE session")

    cid = int(cfg.short_dte_client_id)
    if cid in reserved_client_ids(base_client_id):
        return JobOutcome(
            "refused_client_id",
            f"short_dte_client_id={cid} collides with the bot's ids "
            f"{sorted(reserved_client_ids(base_client_id))}")

    if not gateway_listening(int(ibkr_port)):
        return JobOutcome("skipped_gateway_down",
                          f"nothing is listening on the IB port {ibkr_port} (TWS/Gateway down)")

    status_path = Path(data_dir) / STATUS_FILE_NAME
    try:  # a stale file from an earlier run must never be mistaken for this run's result
        status_path.unlink()
    except FileNotFoundError:
        pass
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"short_dte_collect_{today:%Y%m%d}.log"
    cmd = build_command(cfg, python=python or sys.executable,
                        script=Path(root) / "scripts" / "calibrate_option_spreads.py",
                        status_path=status_path, min_sessions=min_sessions)

    started = clock()
    timeout_s = float(cfg.short_dte_timeout_minutes) * 60.0
    idle_s = float(cfg.short_dte_idle_minutes) * 60.0
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} start: "
                   f"{' '.join(cmd[2:])}\n")
        logf.flush()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        proc = popen(cmd, cwd=str(root), stdout=logf, stderr=subprocess.STDOUT, env=env)
        last_size, last_change = -1, clock()
        while proc.poll() is None:
            now = clock()
            if now - started > timeout_s:
                _stop(proc, sleep)
                return JobOutcome("timeout", f"exceeded {cfg.short_dte_timeout_minutes} min",
                                  log_path=str(log_path))
            try:
                size = log_path.stat().st_size
            except OSError:
                size = last_size
            if size != last_size:
                last_size, last_change = size, now
            elif now - last_change > idle_s:
                _stop(proc, sleep)
                return JobOutcome("idle_timeout",
                                  f"no output for {cfg.short_dte_idle_minutes} min",
                                  log_path=str(log_path))
            sleep(poll_s)
        rc = proc.poll()

    status = _read_status(status_path)
    if rc != 0 or status is None:
        why = (status or {}).get("error") or _tail(log_path) or f"exit code {rc}, no status file"
        return JobOutcome("failed", f"exit={rc}: {why}", log_path=str(log_path))
    stored, planned, bands = _summarise(status)
    if not status.get("ok", False):
        return JobOutcome("failed", str(status.get("error") or "status ok=false"),
                          stored=stored, planned=planned, bands=bands, log_path=str(log_path))
    if stored == 0:
        return JobOutcome("empty", f"planned {planned} contracts, stored 0 samples",
                          stored=0, planned=planned, bands=bands, log_path=str(log_path))
    return JobOutcome("ok", f"stored {stored} samples over {planned} planned contracts",
                      stored=stored, planned=planned, bands=bands, log_path=str(log_path))

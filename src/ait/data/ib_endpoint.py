"""W94: the record of which IB endpoint the supervisor really uses vs what .env names.

The supervisor writes data/ib_endpoint.json (no file I/O in the broker layer); the status page and the
daily digest read it, so a wrong IBKR_PORT in .env keeps being visible instead of being silently
papered over by the fallback in ait.data.ib_ports. Kept apart from ib_ports, which is pinned as a pure,
import-light module (tests/test_w82_lifecycle_capture.py::test_module_is_pure_stdlib).
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from ait.data.ib_ports import endpoint_label

ENDPOINT_FILE_NAME = "ib_endpoint.json"


def read_endpoint_state(path: Path) -> dict | None:
    """The stored {configured, using, fallback, since}, or None when absent/unreadable."""
    try:
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        return rec if isinstance(rec, dict) else None
    except (OSError, ValueError):
        return None


def write_endpoint_state(path: Path, configured: int, using: int,
                         now: datetime | None = None) -> bool:
    """Record (configured, using); rewrite only when that pair changed. True when written.

    `since` is the time the current pair first held. Atomic (temp file + replace)."""
    prev = read_endpoint_state(path)
    if prev and prev.get("configured") == int(configured) and prev.get("using") == int(using):
        return False
    rec = {"configured": int(configured), "using": int(using),
           "fallback": int(configured) != int(using),
           "since": (now or datetime.now()).isoformat(timespec="seconds")}
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return True


def endpoint_mismatch_text(rec: dict | None) -> str:
    """Empty when in sync/unknown, else one line telling the operator to fix IBKR_PORT."""
    if not rec or not rec.get("fallback"):
        return ""
    c, u = int(rec["configured"]), int(rec["using"])
    return (f".env IBKR_PORT={c} ({endpoint_label(c)}) but {u} ({endpoint_label(u)}) is what is up "
            f"- using {u}; fix .env (python scripts/sync_ibkr_port.py)")

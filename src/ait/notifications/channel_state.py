"""Is the alert channel delivering? One small file both processes write (W104).

A page that never arrives is the worst silent failure: the system keeps running
and nobody is told. Before this module a Telegram outage left only a warning in a
log (``alert_send_failed`` / ``telegram_send_failed``), the same line for a
revoked token, a rate limit and a lost network, and no surface a person looks at.

``data/alert_channel.json`` exists only while the channel is failing:
``{"failing_since", "kind", "count", "source", "last_failure"}``. A successful
send from either process deletes it. ``status.py`` shows it, so the web status
page and ``python status.py`` say "alerts failing since ...".

Kinds: ``auth`` (HTTP 401/403: token revoked or wrong chat), ``rate`` (429),
``transient`` (anything else: network, 5xx), ``no_creds`` (token or chat id not
configured).

Pure stdlib with no ``ait`` imports: the supervisor imports it and must not pull
in anything heavy (see the note at the top of orchestration/master.py).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

#: Repo root / data. Tests point STATE_PATH at a temp file.
_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "alert_channel.json"
STATE_PATH: Path = _DEFAULT_PATH

AUTH = "auth"
RATE = "rate"
TRANSIENT = "transient"
NO_CREDS = "no_creds"


def classify_status(status: int | None) -> str:
    """The failure kind for an HTTP status."""
    if status in (401, 403):
        return AUTH
    if status == 429:
        return RATE
    return TRANSIENT


def _active() -> bool:
    """Tests never touch the real file: under pytest only a patched path writes."""
    return not ("PYTEST_CURRENT_TEST" in os.environ and STATE_PATH == _DEFAULT_PATH)


def read() -> dict | None:
    """The failing-channel record, or None when the channel is healthy / unknown."""
    try:
        rec = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return rec if isinstance(rec, dict) and rec.get("failing_since") else None
    except (OSError, ValueError):
        return None


def _write(rec: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(STATE_PATH.parent), prefix=".alert_channel.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        os.replace(tmp, STATE_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_failure(source: str, kind: str) -> None:
    """A send failed. Keeps the first failure's time, counts the rest. Never raises."""
    if not _active():
        return
    try:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        cur = read() or {}
        _write({
            "failing_since": cur.get("failing_since") or now,
            "kind": kind,
            "count": int(cur.get("count", 0)) + 1,
            "source": source,
            "last_failure": now,
        })
    except Exception:  # noqa: BLE001 — reporting the channel must never break a sender
        pass


def record_success(source: str) -> None:
    """A send was accepted: the channel works. Never raises."""
    if not _active():
        return
    try:
        if STATE_PATH.exists():
            STATE_PATH.unlink()
    except Exception:  # noqa: BLE001
        pass


def marker_path() -> Path:
    """Where the bot writes TELEGRAM_DEAD: beside the state file (absolute), or the
    old cwd-relative path when the state file is sandboxed away under pytest."""
    return STATE_PATH.parent / "TELEGRAM_DEAD" if _active() else Path("data/TELEGRAM_DEAD")

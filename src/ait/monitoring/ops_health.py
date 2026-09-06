"""Shared authority for AIT's operator-facing health surfaces (W6).

Every helper in here exists because an operator surface reported GREEN or
ZERO when the truth was "this channel was never wired".  A false all-clear is
strictly worse than an error, because the operator stops looking.

The surfaces that share this module:

* ``ait.dashboard.app``  - Streamlit System Health / Trade Intelligence tabs
* ``web_logs.py``        - the localhost:8502 live log viewer
* ``scripts/verify_deadman.py`` - the external dead-man arming check
* ``ait.monitoring.watchdog``   - the writer half of the health channel
* ``status.py``          - MUST call :func:`native_crash_report` (see
  log-contracts-3); it still greps ``bot_stdout.log``, where faulthandler has
  not written since ``src/ait/main.py`` moved the sink to ``logs/fatal.log``.

Sections
--------
1. ANSI-robust structured-log parsing        (log-contracts-6)
2. Bounded tail reads + incremental counters (log-contracts-1, log-contracts-4)
3. Native crash counting                     (log-contracts-3)
4. Bot liveness evidence                     (bot-day-02)
5. bot_state health channel (write + read)   (string-contracts-5, db-contracts-6)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    # parsing
    "ANSI_RE",
    "strip_ansi",
    "parse_event_line",
    "parse_timestamp",
    "event_local_date",
    # tailing
    "DEFAULT_TAIL_BYTES",
    "tail_text",
    "tail_lines",
    "rotated_siblings",
    "local_midnight",
    "DayEventCounts",
    "IncrementalEventCounter",
    "count_events_today",
    # crashes
    "FATAL_HEADERS",
    "count_native_crashes",
    "native_crash_report",
    # liveness
    "LivenessVerdict",
    "HEARTBEAT_MAX_AGE_S",
    "RTH_WARMUP_S",
    "bot_liveness",
    # health channel
    "HEALTH_CHANNEL_KEY",
    "HEALTH_ERRORS_KEY",
    "HEALTH_MEMORY_KEY",
    "COMPONENT_KEY_PREFIX",
    "NON_COMPONENT_KEYS",
    "ChannelState",
    "HealthStatePublisher",
    "read_channel_state",
    "process_memory",
]


# ---------------------------------------------------------------------------
# 1. ANSI-robust structured-log parsing  (log-contracts-6)
# ---------------------------------------------------------------------------
#
# setup_logging() picks ConsoleRenderer when sys.stdout.isatty() and feeds the
# SAME rendered string to the file handler, so a bot started from an
# interactive console writes ANSI-coloured key=value lines into the production
# ait.log for the whole session.  Rather than fight the renderer (a file we do
# not own), every consumer parses all three shapes the pipeline can produce:
#
#   JSON            {"component": ..., "event": "ml_prediction", ...}
#   console + ANSI  \x1b[2m<ts>\x1b[0m [\x1b[32m\x1b[1minfo ...] <event> ... k=v
#   console plain   2026-08-26 07:09:35 [info     ] config_default_divergence  k=v
#
# The plain-console shape is not hypothetical either: logs/bot_stdout.log
# line 1 in production is exactly that (structlog's pre-setup_logging default).

ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")

_CONSOLE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)\s+\[(?P<level>[A-Za-z]+)\s*\]\s+(?P<rest>.*)$"
)
# event is the first bracket-free, whitespace-free token; the optional
# [logger] block follows; everything after that is key=value pairs.
_CONSOLE_REST_RE = re.compile(
    r"^(?P<event>[^\s\[\]]+)\s*(?:\[(?P<logger>[^\]]*)\]\s*)?(?P<kv>.*)$"
)
_KV_RE = re.compile(r"(?P<k>[A-Za-z_][\w.]*)=(?P<v>'[^']*'|\"[^\"]*\"|\S+)")


def strip_ansi(text: str) -> str:
    """Remove SGR/CSI escape sequences from ``text``."""
    return ANSI_RE.sub("", text or "")


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def parse_event_line(line: str) -> dict[str, Any] | None:
    """Parse one log line into a field dict, or ``None`` if it is not a record.

    The returned dict always carries ``event``/``level``/``timestamp``/
    ``component``/``logger`` keys (possibly empty strings) plus every
    structured field the producer bound.  ``_fmt`` records which rendering the
    line used ("json" | "console").  Keys starting with ``_`` are metadata and
    should be skipped by display code.
    """
    if not line:
        return None
    raw = line.strip()
    if not raw:
        return None

    # --- JSON rendering (production, non-tty) ---------------------------
    if raw[0] == "{" and raw[-1] == "}":
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict) and "event" in data:
            rec = dict(data)
            rec.setdefault("level", "")
            rec.setdefault("component", "")
            rec.setdefault("logger", "")
            rec.setdefault("timestamp", "")
            rec["_fmt"] = "json"
            return rec

    # --- ConsoleRenderer rendering, with or without ANSI ----------------
    plain = strip_ansi(raw).strip()
    m = _CONSOLE_RE.match(plain)
    if not m:
        return None
    rest = _CONSOLE_REST_RE.match(m.group("rest").strip())
    if not rest:
        return None
    rec: dict[str, Any] = {
        "event": rest.group("event"),
        "level": m.group("level").lower(),
        "timestamp": m.group("ts"),
        "logger": rest.group("logger") or "",
        "component": "",
        "_fmt": "console",
    }
    for kv in _KV_RE.finditer(rest.group("kv") or ""):
        rec[kv.group("k")] = _unquote(kv.group("v"))
    return rec


def parse_timestamp(raw: Any) -> datetime | None:
    """Parse a log timestamp into an aware datetime in the LOCAL zone.

    time-authority-1: producers stamp UTC (``TimeStamper(fmt='iso')`` defaults
    to ``utc=True``) while every consumer compares against a LOCAL date.  A
    naive stamp is read as local; a UTC stamp is converted.  Doing the
    conversion here is what keeps "today" counters honest across the 20:00 ET
    UTC-midnight rollover.
    """
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    s = s.replace(",", ".")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.fromisoformat(s[:26])
        except ValueError:
            return None
    try:
        return dt.astimezone()
    except (OSError, OverflowError, ValueError):
        return None


def event_local_date(rec: dict[str, Any]) -> date | None:
    """Local calendar date of a parsed record (None when unstamped)."""
    dt = parse_timestamp(rec.get("timestamp"))
    return dt.date() if dt else None


# ---------------------------------------------------------------------------
# 2. Bounded tail reads + incremental day counters  (log-contracts-1, -4)
# ---------------------------------------------------------------------------

DEFAULT_TAIL_BYTES = 512 * 1024
DEFAULT_SEED_BYTES = 64 * 1024 * 1024


def tail_text(path: str | Path, max_bytes: int = DEFAULT_TAIL_BYTES) -> str:
    """Read at most ``max_bytes`` from the END of ``path``.

    log-contracts-4: web_logs.py used ``f.readlines()[-200:]`` on a 76 MB file
    every 5 s per browser tab.  Seeking makes the read O(max_bytes) instead of
    O(file), and dropping the first (partial) line keeps the parser honest.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return ""
    try:
        with open(p, "rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # discard the partial line we landed inside
            data = fh.read()
    except OSError:
        return ""
    return data.decode("utf-8", "replace")


def tail_lines(path: str | Path, max_bytes: int = DEFAULT_TAIL_BYTES) -> list[str]:
    """Lines from a bounded tail read of ``path``."""
    return tail_text(path, max_bytes).splitlines()


def local_midnight(now: datetime | None = None) -> datetime:
    """Start of the current LOCAL day, as an aware datetime."""
    now = (now or datetime.now()).astimezone()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def rotated_siblings(
    path: str | Path,
    max_backups: int = 10,
    newer_than: datetime | None = None,
) -> list[Path]:
    """``[log.N, ..., log.1, log]`` oldest-first, existing files only.

    log-contracts-1: ait.log rotates several times during one RTH session, so
    "today" lives mostly in ``ait.log.1``..``ait.log.4``.  A consumer that
    reads only the live file reports ZERO activity for a bot that predicted
    112 times.  ``newer_than`` (normally local midnight) drops backups that
    cannot contain today's records.
    """
    p = Path(path)
    out: list[Path] = []
    for i in range(max_backups, 0, -1):
        cand = p.with_name(p.name + f".{i}")
        try:
            st = cand.stat()
        except OSError:
            continue
        if newer_than is not None:
            mtime = datetime.fromtimestamp(st.st_mtime).astimezone()
            if mtime < newer_than:
                continue
        out.append(cand)
    if p.exists():
        out.append(p)
    return out


@dataclass
class DayEventCounts:
    """Result of one :meth:`IncrementalEventCounter.refresh`."""

    day: date
    counts: dict[str, int]
    last_record: dict[str, Any] | None = None
    #: newest record seen today PER event name — so "last ML prediction" does
    #: not disappear the moment a different event is logged after it.
    last_by_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    partial: bool = False
    rotations: int = 0
    bytes_scanned: int = 0
    sources: list[str] = field(default_factory=list)

    def total(self) -> int:
        return sum(self.counts.values())


class IncrementalEventCounter:
    """Counts named events for the current LOCAL day without re-reading the file.

    Seeds once from today's rotated backups plus a byte-capped tail of the live
    file, then only ever reads the bytes appended since the previous refresh.
    Handles rotation (size shrank) and local-midnight rollover.

    This replaces both halves of log-contracts-4: the 200-line display window
    that hid a trade from 20 minutes ago, and the full 76 MB ``readlines()``
    every 5 s per browser tab.
    """

    def __init__(
        self,
        path: str | Path,
        events: Sequence[str],
        *,
        seed_max_bytes: int = DEFAULT_SEED_BYTES,
        min_interval_s: float = 5.0,
        include_rotations: bool = True,
        max_backups: int = 10,
    ) -> None:
        self.path = Path(path)
        self.events = tuple(events)
        self.seed_max_bytes = int(seed_max_bytes)
        self.min_interval_s = float(min_interval_s)
        self.include_rotations = include_rotations
        self.max_backups = max_backups

        self._day: date | None = None
        self._offset = 0
        self._counts: dict[str, int] = {e: 0 for e in self.events}
        self._last_record: dict[str, Any] | None = None
        self._last_by_event: dict[str, dict[str, Any]] = {}
        self._partial = False
        self._rotations = 0
        self._bytes = 0
        self._sources: list[str] = []
        self._last_refresh = 0.0
        self._seeded = False

    # -- internals ------------------------------------------------------
    def _reset_day(self, day: date) -> None:
        self._day = day
        self._counts = {e: 0 for e in self.events}
        self._last_record = None
        self._last_by_event = {}
        self._partial = False
        self._rotations = 0
        self._bytes = 0
        self._sources = []
        self._seeded = False

    def _consume(self, text: str, day: date) -> None:
        for line in text.splitlines():
            rec = parse_event_line(line)
            if rec is None:
                continue
            ev = str(rec.get("event", ""))
            if ev not in self._counts:
                continue
            when = event_local_date(rec)
            # An unstamped record cannot be attributed to a day; skip it rather
            # than inflate today's count.
            if when != day:
                continue
            self._counts[ev] += 1
            self._last_record = rec
            self._last_by_event[ev] = rec

    def _seed(self, day: date) -> None:
        midnight = datetime.combine(day, datetime.min.time()).astimezone()
        files: list[Path] = []
        if self.include_rotations:
            files = rotated_siblings(self.path, self.max_backups, newer_than=midnight)
        elif self.path.exists():
            files = [self.path]
        for f in files:
            if f == self.path:
                continue
            text = tail_text(f, self.seed_max_bytes)
            self._bytes += len(text)
            self._sources.append(f.name)
            self._consume(text, day)
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        text = tail_text(self.path, self.seed_max_bytes)
        self._bytes += len(text)
        if self.path.exists():
            self._sources.append(self.path.name)
        if size > self.seed_max_bytes:
            self._partial = True
        self._consume(text, day)
        self._offset = size
        self._seeded = True

    # -- public ---------------------------------------------------------
    def refresh(self, *, now: datetime | None = None, force: bool = False) -> DayEventCounts:
        """Fold newly-appended bytes into the running day counts."""
        now = (now or datetime.now()).astimezone()
        day = now.date()
        monotonic = time.monotonic()
        if (
            self._seeded
            and not force
            and self._day == day
            and (monotonic - self._last_refresh) < self.min_interval_s
        ):
            return self.snapshot()
        self._last_refresh = monotonic

        if self._day != day or not self._seeded:
            if self._day != day:
                self._reset_day(day)
            self._seed(day)
            return self.snapshot()

        try:
            size = self.path.stat().st_size
        except OSError:
            return self.snapshot()
        if size < self._offset:
            # Rotated or truncated under us: the new file starts at 0.  Records
            # written to the old file after our last read are unrecoverable, so
            # say so rather than silently under-reporting.
            self._rotations += 1
            self._partial = True
            self._offset = 0
        if size == self._offset:
            return self.snapshot()
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self._offset)
                data = fh.read()
        except OSError:
            return self.snapshot()
        self._offset = self._offset + len(data)
        self._bytes += len(data)
        self._consume(data.decode("utf-8", "replace"), day)
        return self.snapshot()

    def snapshot(self) -> DayEventCounts:
        return DayEventCounts(
            day=self._day or date.today(),
            counts=dict(self._counts),
            last_record=dict(self._last_record) if self._last_record else None,
            last_by_event={k: dict(v) for k, v in self._last_by_event.items()},
            partial=self._partial,
            rotations=self._rotations,
            bytes_scanned=self._bytes,
            sources=list(self._sources),
        )


def count_events_today(
    path: str | Path,
    events: Sequence[str],
    *,
    now: datetime | None = None,
    seed_max_bytes: int = DEFAULT_SEED_BYTES,
    include_rotations: bool = True,
) -> DayEventCounts:
    """One-shot version of :class:`IncrementalEventCounter` (CLI / test use)."""
    counter = IncrementalEventCounter(
        path, events, seed_max_bytes=seed_max_bytes,
        min_interval_s=0.0, include_rotations=include_rotations,
    )
    return counter.refresh(now=now, force=True)


# ---------------------------------------------------------------------------
# 3. Native crash counting  (log-contracts-3)
# ---------------------------------------------------------------------------
#
# src/ait/main.py:43-48 points faulthandler at logs/fatal.log ("dumps used to
# go to stderr -> bot_stdout.log, where the rotation cycle destroyed every one
# of them").  status.py still counts the header in bot_stdout.log and so
# reports "native crashes: 0" forever, beside "keeper relaunches: 8".

FATAL_HEADERS = ("Windows fatal exception", "Fatal Python error")
FATAL_LOG = Path("logs") / "fatal.log"
LEGACY_CRASH_LOGS = (Path("logs") / "bot_stdout.log",)


def _count_headers(text: str) -> int:
    return sum(1 for ln in text.splitlines()
               if any(h in ln for h in FATAL_HEADERS))


def native_crash_report(
    fatal_log: str | Path = FATAL_LOG,
    legacy_logs: Iterable[str | Path] = LEGACY_CRASH_LOGS,
    *,
    max_bytes: int = 8 * 1024 * 1024,
    max_backups: int = 3,
) -> dict[str, Any]:
    """Count faulthandler crash dumps across the CURRENT and legacy sinks.

    Returns ``{"count", "fatal_count", "legacy_count", "channel_wired",
    "last_write", "sources", "detail"}``.

    ``channel_wired`` is False when ``logs/fatal.log`` does not exist at all -
    that is the difference between "no crashes" and "nothing would record a
    crash", and it must never be rendered as a plain 0.
    """
    fatal_path = Path(fatal_log)
    wired = fatal_path.exists()
    sources: list[str] = []
    fatal_count = 0
    last_write: str | None = None
    if wired:
        sources.append(str(fatal_path))
        fatal_count = _count_headers(tail_text(fatal_path, max_bytes))
        for i in range(1, max_backups + 1):
            back = fatal_path.with_name(fatal_path.name + f".{i}")
            if back.exists():
                sources.append(str(back))
                fatal_count += _count_headers(tail_text(back, max_bytes))
        try:
            st = fatal_path.stat()
            if st.st_size > 0:
                last_write = datetime.fromtimestamp(
                    st.st_mtime).isoformat(timespec="seconds")
        except OSError:
            pass

    legacy_count = 0
    for lg in legacy_logs:
        for cand in rotated_siblings(lg, max_backups):
            sources.append(str(cand))
            legacy_count += _count_headers(tail_text(cand, max_bytes))

    if not wired:
        detail = (f"crash channel NOT WIRED - {fatal_path} does not exist; "
                  "faulthandler has nowhere to dump (see src/ait/main.py:43-48)")
    elif fatal_count or legacy_count:
        detail = (f"{fatal_count} in {fatal_path.name}, "
                  f"{legacy_count} in legacy stdout logs")
    else:
        detail = f"crash channel live ({fatal_path}); no dumps recorded"

    return {
        "count": fatal_count + legacy_count,
        "fatal_count": fatal_count,
        "legacy_count": legacy_count,
        "channel_wired": wired,
        "last_write": last_write,
        "sources": sources,
        "detail": detail,
    }


def count_native_crashes(
    fatal_log: str | Path = FATAL_LOG,
    legacy_logs: Iterable[str | Path] = LEGACY_CRASH_LOGS,
    **kw: Any,
) -> int:
    """Total faulthandler dumps across current + legacy sinks."""
    return int(native_crash_report(fatal_log, legacy_logs, **kw)["count"])


# ---------------------------------------------------------------------------
# 4. Bot liveness evidence  (bot-day-02)
# ---------------------------------------------------------------------------
#
# keeper_ait.bat pings healthchecks.io whenever a process-table regex matches,
# so a Telegram-dead + gateway-down outage (master alive, bot dead: see
# master.py:575-584 'bot_restart_deferred_gateway_down') shows GREEN on the one
# monitor built for dead alert channels.
#
# Window justification (the finding warns that over-sensitivity buys alert
# fatigue):
#   * cadence: orchestrator.py:1021-1034 touches data/bot_heartbeat once per
#     30 s market-loop iteration, and DELIBERATELY stops while an error streak
#     is live (R8) - so staleness already means "not functioning", not merely
#     "quiet".
#   * threshold: 900 s == 30 missed beats == exactly master.py:499's own
#     bot_hung_heartbeat_stale trigger.  Matching it means the dead-man can
#     never fire before the supervisor's own restart+alert would have, so this
#     adds zero new alert surface.
#   * scope: enforced ONLY while the market is open.  The heartbeat is written
#     only in the MARKET_OPEN phase, so demanding freshness overnight would
#     page every single evening.
#   * warmup: the first 900 s after 09:30 are exempt, because a keeper
#     relaunch at the bell has to finish startup training before the loop's
#     first heartbeat (bot-day-04).

HEARTBEAT_MAX_AGE_S = 900.0
RTH_WARMUP_S = 900.0
DEFAULT_HEARTBEAT = Path("data") / "bot_heartbeat"


@dataclass(frozen=True)
class LivenessVerdict:
    """Whether there is EVIDENCE the bot is functioning, not merely present."""

    ok: bool
    state: str
    detail: str
    heartbeat_age_s: float | None = None
    rth: bool = False

    @property
    def should_ping(self) -> bool:
        """True when a dead-man success ping is honest."""
        return self.ok


def _default_market_open() -> tuple[bool, int | None]:
    """(market_open, minutes_since_open) - never raises."""
    try:
        from ait.utils.time import is_market_open, minutes_since_open
        return bool(is_market_open()), minutes_since_open()
    except Exception:  # noqa: BLE001 - a calendar import must not gate alerting
        return False, None


def bot_liveness(
    *,
    now: datetime | None = None,
    heartbeat_path: str | Path = DEFAULT_HEARTBEAT,
    max_age_s: float = HEARTBEAT_MAX_AGE_S,
    warmup_s: float = RTH_WARMUP_S,
    market_open: bool | None = None,
    minutes_since_open: int | None = None,
) -> LivenessVerdict:
    """Decide whether the bot is demonstrably ALIVE AND TRADING.

    ``market_open``/``minutes_since_open`` are injectable so callers (and
    tests) do not pay for the NYSE calendar; when omitted they come from
    ``ait.utils.time``.  A calendar failure degrades to "not RTH" - that
    preserves today's behaviour instead of inventing a new page.
    """
    now = now or datetime.now()
    if market_open is None:
        market_open, cal_minutes = _default_market_open()
        if minutes_since_open is None:
            minutes_since_open = cal_minutes

    hb = Path(heartbeat_path)
    age: float | None = None
    try:
        age = max(0.0, now.timestamp() - hb.stat().st_mtime)
    except OSError:
        age = None

    if not market_open:
        return LivenessVerdict(
            ok=True,
            state="not_rth",
            detail=("market closed - the trading loop legitimately stops "
                    "touching data/bot_heartbeat outside RTH"),
            heartbeat_age_s=age,
            rth=False,
        )

    if minutes_since_open is not None and minutes_since_open * 60.0 < warmup_s:
        return LivenessVerdict(
            ok=True,
            state="warmup",
            detail=(f"within the first {warmup_s / 60:.0f} min of RTH - a keeper "
                    "relaunch at the bell is still arming (bot-day-04)"),
            heartbeat_age_s=age,
            rth=True,
        )

    if age is None:
        return LivenessVerdict(
            ok=False,
            state="heartbeat_missing",
            detail=(f"{hb} missing during RTH - no evidence the trading loop "
                    "ever ran; process existence is NOT liveness"),
            heartbeat_age_s=None,
            rth=True,
        )

    if age > max_age_s:
        return LivenessVerdict(
            ok=False,
            state="heartbeat_stale",
            detail=(f"trading-loop heartbeat {age / 60:.0f} min stale during RTH "
                    f"(limit {max_age_s / 60:.0f} min) - bot down, hung, or in an "
                    "error streak; positions are unmanaged"),
            heartbeat_age_s=age,
            rth=True,
        )

    return LivenessVerdict(
        ok=True,
        state="heartbeat_fresh",
        detail=f"trading-loop heartbeat {age:.0f}s old during RTH",
        heartbeat_age_s=age,
        rth=True,
    )


# ---------------------------------------------------------------------------
# 5. bot_state health channel  (string-contracts-5, db-contracts-6)
# ---------------------------------------------------------------------------
#
# The dashboard's System Health tab was built against a watchdog
# state-publishing contract whose WRITER half was never implemented: every
# panel took its fallback branch on every render, and the error panel's
# fallback was an affirmative green st.success('No errors logged').  This is
# that writer half, plus the reader-side freshness check the panels need to
# tell "no errors" from "no error channel".

HEALTH_CHANNEL_KEY = "watchdog_channel"
HEALTH_ERRORS_KEY = "watchdog_errors"
HEALTH_MEMORY_KEY = "system_memory_usage"
COMPONENT_KEY_PREFIX = "watchdog_"
#: watchdog_* keys that are NOT per-component rows (the dashboard's
#: ``LIKE 'watchdog_%'`` component query must exclude these).
NON_COMPONENT_KEYS = frozenset({HEALTH_CHANNEL_KEY, HEALTH_ERRORS_KEY})

DEFAULT_STATE_DB = Path("data") / "ait_state.db"
DEFAULT_PUBLISH_INTERVAL_S = 60.0
MAX_ERRORS_RETAINED = 50
#: A channel older than this is historical, not current.  15 min covers the
#: watchdog's worst-case publish cadence (check_and_recover every ~5 min) with
#: 3x headroom.
CHANNEL_MAX_AGE_S = 900.0


def process_memory() -> dict[str, Any]:
    """RSS/VMS for this process in MB (``None`` where unavailable)."""
    try:
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        vms = getattr(info, "vms", None)
        return {
            "rss_mb": round(info.rss / (1024 * 1024), 1),
            "vms_mb": round(vms / (1024 * 1024), 1) if vms else None,
        }
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"rss_mb": round(usage.ru_maxrss / 1024, 1), "vms_mb": None}
    except Exception:  # noqa: BLE001
        return {"rss_mb": None, "vms_mb": None}


def _under_pytest() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules


def _status_str(value: Any) -> str:
    return str(getattr(value, "value", value) or "unknown")


def _hb_iso(epoch: Any) -> str:
    try:
        epoch = float(epoch or 0.0)
    except (TypeError, ValueError):
        return ""
    if epoch <= 0:
        return ""
    return datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


class HealthStatePublisher:
    """Writes the watchdog's in-memory health into bot_state.

    Deliberately uses a plain sqlite connection rather than StateManager:
    StateManager.__init__ opens the DuckDB mirror READ-WRITE (multiprocess-db-1)
    and a monitoring heartbeat must never take that lock.

    Safety rails, because this runs inside the live trading loop:
      * never creates a database - if ``bot_state`` is absent the publisher
        disables itself for that call;
      * throttled to ``min_interval_s`` (forced writes bypass it);
      * every failure is swallowed and recorded in :attr:`last_error`;
      * when ``db_path`` is left to the default, publishing is OFF under
        pytest so a unit test can never write the production ledger.  Tests
        that exercise the real write path pass an explicit ``db_path``.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        min_interval_s: float = DEFAULT_PUBLISH_INTERVAL_S,
        max_errors: int = MAX_ERRORS_RETAINED,
        enabled: bool | None = None,
    ) -> None:
        explicit = db_path is not None
        self.db_path = Path(db_path) if explicit else Path(
            os.environ.get("AIT_STATE_DB", str(DEFAULT_STATE_DB)))
        self.min_interval_s = float(min_interval_s)
        self.max_errors = int(max_errors)
        if enabled is None:
            env = os.environ.get("AIT_HEALTH_STATE")
            if env is not None:
                enabled = env.strip().lower() not in ("0", "false", "no", "off")
            else:
                enabled = True if explicit else not _under_pytest()
        self.enabled = bool(enabled)
        self.last_error: str | None = None
        self.writes = 0
        self._last_publish = 0.0

    # -- internals ------------------------------------------------------
    def _connect(self) -> sqlite3.Connection | None:
        if not self.db_path.exists():
            self.last_error = f"state db missing: {self.db_path}"
            return None
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='bot_state'"
        ).fetchone()
        if row is None:
            conn.close()
            self.last_error = f"no bot_state table in {self.db_path}"
            return None
        return conn

    @staticmethod
    def _put(conn: sqlite3.Connection, key: str, value: Any, when: str) -> None:
        conn.execute(
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value if isinstance(value, str) else json.dumps(value), when),
        )

    # -- public ---------------------------------------------------------
    def due(self, *, now_monotonic: float | None = None) -> bool:
        """True when the throttle window has elapsed and a publish would write.

        Lets a caller skip building a HealthStatus it would only throw away.
        """
        if not self.enabled:
            return False
        now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
        return (now_monotonic - self._last_publish) >= self.min_interval_s

    def publish(
        self,
        health: Any,
        errors: Sequence[dict[str, Any]] = (),
        *,
        now: datetime | None = None,
        force: bool = False,
    ) -> bool:
        """Persist ``health`` under exactly the keys the dashboard reads.

        Returns True when a write happened.  Never raises.
        """
        if not self.enabled:
            return False
        monotonic = time.monotonic()
        if not force and (monotonic - self._last_publish) < self.min_interval_s:
            return False
        conn = None
        try:
            conn = self._connect()
            if conn is None:
                return False
            when = (now or datetime.now()).isoformat(timespec="seconds")
            components = dict(getattr(health, "components", {}) or {})
            with conn:
                for name, comp in components.items():
                    self._put(conn, f"{COMPONENT_KEY_PREFIX}{name}", {
                        "status": _status_str(getattr(comp, "status", "unknown")),
                        "last_heartbeat": _hb_iso(getattr(comp, "last_heartbeat", 0.0)),
                        "error_count": int(getattr(comp, "error_count", 0) or 0),
                        "last_error": str(getattr(comp, "last_error", "") or ""),
                        "latency_ms": round(
                            float(getattr(comp, "latency_ms", 0.0) or 0.0), 1),
                    }, when)
                self._put(conn, f"{COMPONENT_KEY_PREFIX}system", {
                    "status": _status_str(getattr(health, "status", "unknown")),
                    "last_heartbeat": when,
                    "uptime_hours": round(
                        float(getattr(health, "uptime_seconds", 0.0) or 0.0) / 3600.0, 2),
                    "trading_loop_alive": bool(getattr(health, "trading_loop_alive", False)),
                    "ibkr_connected": bool(getattr(health, "ibkr_connected", False)),
                }, when)
                mem = process_memory()
                mem_mb = getattr(health, "memory_mb", None)
                if mem.get("rss_mb") is None and mem_mb:
                    mem["rss_mb"] = round(float(mem_mb), 1)
                mem["updated_at"] = when
                self._put(conn, HEALTH_MEMORY_KEY, mem, when)
                # ALWAYS written, even empty: key presence is what tells the
                # dashboard "an error channel exists" vs "nothing reports here".
                self._put(conn, HEALTH_ERRORS_KEY,
                          list(errors)[-self.max_errors:], when)
                self._put(conn, HEALTH_CHANNEL_KEY, {
                    "updated_at": when,
                    "pid": os.getpid(),
                    "source": "ait.monitoring.watchdog",
                    "components": sorted(components),
                }, when)
            self.writes += 1
            self._last_publish = monotonic
            self.last_error = None
            return True
        except Exception as exc:  # noqa: BLE001 - monitoring must never kill the loop
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass


@dataclass(frozen=True)
class ChannelState:
    """Is the bot_state health channel wired, and is what it says current?"""

    wired: bool
    fresh: bool
    updated_at: str | None
    age_s: float | None
    detail: str

    @property
    def trustworthy(self) -> bool:
        return self.wired and self.fresh


def read_channel_state(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    max_age_s: float = CHANNEL_MAX_AGE_S,
) -> ChannelState:
    """Read the health-channel beacon.  NEVER reports green on an absent key."""
    now = now or datetime.now()
    try:
        row = conn.execute(
            "SELECT value, updated_at FROM bot_state WHERE key = ?",
            (HEALTH_CHANNEL_KEY,),
        ).fetchone()
    except Exception:  # noqa: BLE001
        row = None
    if row is None:
        return ChannelState(
            wired=False, fresh=False, updated_at=None, age_s=None,
            detail=("health channel NOT WIRED - nothing has ever published "
                    f"bot_state['{HEALTH_CHANNEL_KEY}']. This is not an "
                    "all-clear; read logs/ait.log directly."),
        )
    value, updated_at = row[0], row[1]
    stamp = updated_at
    try:
        parsed = json.loads(value) if isinstance(value, str) else {}
        if isinstance(parsed, dict) and parsed.get("updated_at"):
            stamp = parsed["updated_at"]
    except (ValueError, TypeError):
        pass
    age: float | None = None
    dt = parse_timestamp(stamp)
    if dt is not None:
        age = (now.astimezone() - dt).total_seconds()
    fresh = age is not None and age <= max_age_s
    if fresh:
        detail = f"health channel live (published {stamp})"
    elif age is None:
        detail = f"health channel present but unstamped (raw {stamp!r})"
    else:
        detail = (f"health channel STALE - last publish {stamp} "
                  f"({age / 60:.0f} min ago). Values below are historical, not "
                  "current; expected while the bot is outside market hours.")
    return ChannelState(wired=True, fresh=fresh, updated_at=stamp,
                        age_s=age, detail=detail)


# ---------------------------------------------------------------------------
# CLI - used by keeper_ait.bat / scripts to gate the dead-man ping
# ---------------------------------------------------------------------------

def _cli(argv: Sequence[str]) -> int:
    cmd = argv[0] if argv else "liveness"
    if cmd == "liveness":
        verdict = bot_liveness()
        print(f"liveness: {'OK' if verdict.ok else 'NOT-OK'} "
              f"[{verdict.state}] {verdict.detail}")
        return 0 if verdict.ok else 1
    if cmd == "crashes":
        rep = native_crash_report()
        print(f"native crashes: {rep['count']}  ({rep['detail']})")
        return 0
    print("usage: python -m ait.monitoring.ops_health [liveness|crashes]")
    return 2


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(_cli(sys.argv[1:]))

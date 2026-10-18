"""Master orchestrator — schedules and monitors all AIT v2 subsystems.

Manages:
- Trading bot lifecycle (start, health check, auto-restart)
- ML model retraining (daily pre-market + weekly deep retrain)
- Walk-forward backtesting (weekly, tracks strategy health)
- Performance reporting (daily P&L summary, weekly strategy report)
- Log aggregation and alerting

Runs as a persistent daemon alongside the trading bot.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from zoneinfo import ZoneInfo

from ait.utils.process import scan_pids

# W34 (R26 w23-module-scope-time-import): W23 added
# `from ait.utils.time import ET, now_et` here. That module builds the NYSE
# calendar at import (pandas_market_calendars -> pandas -> numpy), so a broken
# scientific stack — the classic ABI break after an upgrade — killed the
# SUPERVISOR at import: no bot restart, no Gateway recovery, no page, while
# the bot may be holding condors. This file states the rule itself, twice
# (_assume_market_open_on_clock, _in_post_market_window): resolve ET through
# stdlib zoneinfo. Same key as ait.utils.time.ET, so it is the same object.
ET = ZoneInfo("US/Eastern")


def now_et() -> datetime:
    """Current ET time — stdlib only (see the note above)."""
    return datetime.now(ET)


# W38: the stored-timestamp convention, from the stdlib-only leaf. NOT from
# `ait.utils.time` — that module's NYSE calendar build at import is precisely
# what W34 removed from this file, because a broken scientific stack then kills
# the SUPERVISOR. `ait.utils.tz` imports nothing outside the standard library
# and a guard test keeps it that way.
from ait.utils.tz import parse_trade_ts as _as_et  # noqa: E402

# Project root
ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — standalone (no structlog dependency so orchestrator stays light)
# ---------------------------------------------------------------------------

# W34 (R26 probe-output-planted-in-production-forensic-log): the file sink was
# gated ONLY on PYTEST_CURRENT_TEST, so any probe, script or REPL that imported
# this module and drove BotManager appended to logs/orchestrator.log. The
# 2026-09-16 record asserts a 3.5h hang, a fabricated 09:06-10:10 Gateway outage
# and a tmp-fixture singleton failure — none of which happened. The sink is
# now OFF unless the process is the supervisor (main() enables it) or sets
# AIT_ORCH_LIVE=1 on purpose. stdout is unchanged.
_FILE_SINK_ENABLED = False


def _enable_file_sink() -> None:
    global _FILE_SINK_ENABLED
    _FILE_SINK_ENABLED = True


def _log(level: str, event: str, **kw):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extras = " | ".join(f"{k}={v}" for k, v in kw.items())
    line = f"[{ts}] {level.upper():5s} orchestrator.{event}"
    if extras:
        line += f" | {extras}"
    print(line, flush=True)
    # R16 #56: pytest drives BotManager/_verify_launch with mock procs
    # (pid=123) and every run planted fake ERROR launch-failure lines in the
    # production forensic log — indistinguishable from real 22:xx launch
    # failures during incident forensics. Under pytest, drop the file sink
    # (the print above still reaches captured stdout for assertions).
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if not (_FILE_SINK_ENABLED or os.environ.get("AIT_ORCH_LIVE") == "1"):
        return
    # W35: utf-8, explicitly. Both readers (status.py, web_logs.py) decode
    # this file as utf-8; the writer used the host's locale encoding, so on
    # this Windows box any non-ASCII character in an error text (an em dash in
    # a config error, a path) was written as cp1252 and read back as U+FFFD —
    # or raised in a strict reader. Same host-move class as W20/W21.
    with open(LOGS_DIR / "orchestrator.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Cached Telegram creds (read once); None until resolved, () if unavailable.
_TG_CREDS: tuple | None = None


def _alert(message: str) -> bool:
    """Send a supervisor-level Telegram alert (best-effort, synchronous).

    The supervisor is the only thing that survives a bot crash, so dead-bot
    and restart-exhaustion notifications must originate here, not in the bot.
    R16: returns True only when Telegram accepted the send — the digest
    sent-marker (#29) and the TELEGRAM_DEAD recovery probe (#15) both need
    to know whether the message actually left the box.
    """
    global _TG_CREDS
    if _TG_CREDS is None:
        try:
            from ait.config.settings import load_settings
            s = load_settings()
            _TG_CREDS = (s.api_keys.telegram_bot_token, s.api_keys.telegram_chat_id)
        except Exception as e:  # noqa: BLE001
            _log("warn", "alert_creds_unavailable", error=str(e))
            # R8: caching () permanently silenced ALL supervisor alerts after
            # one transient settings failure. Leave None so the next alert
            # retries the load.
            _TG_CREDS = None
            return False
    if not _TG_CREDS or not _TG_CREDS[0] or not _TG_CREDS[1]:
        return False
    token, chat_id = _TG_CREDS
    try:
        import urllib.request
        import urllib.parse
        data = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": f"AIT SUPERVISOR: {message}",
            "disable_web_page_preview": "true",
        }).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data)
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as e:  # noqa: BLE001
        import re as _re
        _log("warn", "alert_send_failed",
             error=_re.sub(r"/bot[^/\s]+", "/bot***", str(e))[:300])  # never leak the token via exception URLs
        return False


# R16 #54: per-key alert cooldowns. A single shared timestamp for a class of
# alerts lets one noisy failure starve a NEW, DIFFERENT critical for the whole
# window (the exact defect in the bot watchdog's global 1h cooldown). Every
# throttled supervisor alert site keys its own cooldown through this gate.
_ALERT_GATE_LAST: dict[str, float] = {}


def _page_stamp_dirs() -> list[Path]:
    """Where a durable page stamp may live. NOT data/: the faults that need a
    durable throttle (permissions, a full disk) are faults OF data/."""
    import tempfile
    return [LOGS_DIR, Path(tempfile.gettempdir())]


def _durable_alert_gate(key: str, interval_s: float) -> bool:
    """W34: _alert_gate for a page sent by a process that then EXITS.

    _ALERT_GATE_LAST is a module dict; a relaunched process re-imports it
    empty. keeper_ait.bat relaunches a supervisor that cannot take its lock
    every ~90s, so the SINGLETON LOCK page went out once per relaunch — ~960
    a day on the only alert channel, while the whole stack was down. The
    stamp is a file's mtime. If no stamp can be written anywhere, page: loud
    beats silent.
    """
    now = time.time()
    name = f".ait_page_{key}.stamp"
    dirs = _page_stamp_dirs()
    for d in dirs:
        try:
            f = Path(d) / name
            if f.exists() and (now - f.stat().st_mtime) < interval_s:
                return False
        except OSError:
            continue
    for d in dirs:
        try:
            (Path(d) / name).write_text(str(now))
            break
        except OSError:
            continue
    return True


def _alert_gate(key: str, interval_s: float, _now: float | None = None) -> bool:
    """True if an alert keyed `key` may fire now; records the attempt.

    Records the attempt time whether or not the caller's send then succeeds,
    so a dead channel is probed once per interval, not every cycle.
    """
    now = time.time() if _now is None else _now
    last = _ALERT_GATE_LAST.get(key)
    if last is not None and now - last < interval_s:
        return False
    _ALERT_GATE_LAST[key] = now
    return True



# ops-substrate-01 (R24, materialized TWICE): when IB Gateway is not listening
# the supervisor deferred the bot start indefinitely and never tried to bring
# the Gateway back. On 2026-07-01 that cost 5h10m (3h15m inside RTH); on
# 2026-09-08 a Gateway java process started at 08:56 and never opened port
# 4002, so the supervisor logged bot_restart_deferred_gateway_down every two
# minutes until 12:17 — three and a half hours of a trading day with open
# positions and no exit engine. The process EXISTED, so nothing detected it.
GATEWAY_RECOVERY_AFTER_S = 600.0     # 10 min down before we intervene
GATEWAY_RECOVERY_COOLDOWN_S = 1800.0  # at most one attempt per 30 min
GATEWAY_MAX_RECOVERIES = 3           # then escalate to a human, do not loop

# W46 (bot-day-03): the hang-detector below (health_check's heartbeat-age
# branch) had zero startup grace, so a boot slower than its own 15-min
# staleness threshold (pre-W46: synchronous model training, up to ~7.5 min
# per trainer.py's own R9 comment, plus reconcile/broker-probe overhead)
# read as "hung" and got killed on the very tick that would have written the
# first fresh heartbeat -- and the retry repeated the same slow boot, so it
# was killed again next tick, forever. 20 min: comfortably above that
# worst-case boot cost even after W46's own bot-day-04 fix (which removes
# training from the blocking path but not the event loop entirely -- a fit
# call can still hold it in bursts), without leaving a genuine hang
# unaddressed for the better part of a session.
BOT_RESTART_GRACE_S = 1200.0
# W35 (R26 gateway-start-script-has-two-env-authorities): the BOOT launcher
# (orchestration/gateway.py) locates IBC through IBC_DIR; this RECOVERY
# launcher had its own unrelated default, so a relocated IBC booted but could
# never be recovered. IBC_DIR is the one authority; AIT_GATEWAY_START_SCRIPT
# remains an explicit override.
GATEWAY_START_SCRIPT = (
    os.environ.get("AIT_GATEWAY_START_SCRIPT")
    or str(Path(os.environ.get("IBC_DIR", r"C:\IBC")) / "StartGateway.bat"))


def _gateway_process_alive() -> bool:
    """Is an IB Gateway JVM running at all?

    Distinguishes "Gateway not started" from "Gateway started but wedged" —
    the 2026-09-08 shape, where killing the stuck JVM is REQUIRED before a
    relaunch can bind the port.

    R25 plat-02 (W16 A2): this was a `powershell Get-Process java` shell-out,
    so on any non-Windows host FileNotFoundError('powershell') was caught and
    read as "no JVM running" — SILENTLY. It is the only one of the four
    Windows-only siblings with no log line at all, and its unmeasured False is
    stamped into the gateway_recovery_attempt record as wedged_process=False:
    the log asserted a fact it never measured. A psutil name scan finds the TWS
    JVM on macOS as readily as the Gateway JVM on Windows.

    The fail direction is UNCHANGED (False), but it is no longer silent —
    "no JVM" and "could not look" imply different recovery actions.
    """
    pids = scan_pids(name_contains="java")
    if pids is None:
        _log("warning", "gateway_process_scan_failed",
             detail="process table unreadable; treating as no JVM")
        return False
    return bool(pids)

#: What marks a JVM as IB's. The live Gateway's command line carries
#: C:\Jts\ibgateway\<ver>\jars\... and, under IBC, ibcalpha.ibc.IbcGateway.
_GATEWAY_JVM_MARKERS = ("ibgateway", "ibcalpha", "\\jts\\", "/jts/",
                        "trader workstation")


def _gateway_jvm_pids() -> list[int] | None:
    """W34 (R26 wedged-jvm-kill): PIDs of IB Gateway/TWS JVMs — and only those.

    The wedged-JVM kill was `Get-Process java | Stop-Process -Force`: every
    JVM on the box, none of the pids the scan had just found, PowerShell-only.
    None = the process table could not be read.
    """
    try:
        import psutil
        pids: list[int] = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                name = (p.info.get("name") or "").lower()
                cmd = " ".join(p.info.get("cmdline") or []).lower()
                pid = p.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                continue
            if "java" in name and any(m in cmd for m in _GATEWAY_JVM_MARKERS):
                pids.append(pid)
        return pids
    except Exception:  # noqa: BLE001
        return None


def _kill_pid(pid: int) -> None:
    """Kill one process by pid (cross-platform, no shell).

    Refuses under pytest. The suite runs ON the live trading host, and this
    path scans the REAL process table: a test that forgets one patch would
    otherwise kill the live broker Gateway mid-session. Tests replace this
    function; nothing in a test run may reach the real one.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError(f"refusing to kill pid {pid} under pytest")
    import psutil
    proc = psutil.Process(pid)
    proc.kill()
    proc.wait(timeout=10)


def _gateway_listening(port: int, host: str = "127.0.0.1") -> bool:
    """Cheap check: is something accepting connections on the Gateway port?

    Used to distinguish "IB Gateway is down" (external, transient — don't
    spend restart budget) from "the bot itself keeps dying" (a real fault).
    """
    import socket
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Single-instance mutex (R16 #66)
# ---------------------------------------------------------------------------
# The keeper's 90s loop and the 07:30 AIT-Bot-Start task both guard with a
# check-then-launch process-table query — no lock — so two masters can pass
# the guard inside the same ~1-2s window (TOCTOU). An OS-level exclusive byte
# lock is atomic and self-releasing: the lock dies with the process, so there
# is no stale-pidfile problem to detect. msvcrt.locking() on Windows (the
# production platform); fcntl.flock() on POSIX (CI runs ubuntu-latest, and
# this also lets the mechanism be exercised on a macOS dev box) — both are
# per-open-handle exclusive locks that release when the holding process
# dies, so a second handle in the SAME process still conflicts with the
# first (the atomicity guard the check-then-launch .bat lacked) and cross-
# process exclusion behaves identically either way.

_singleton_lock_handle = None  # module-held: lock lives exactly as long as this process
# W34: WHY the last acquisition returned what it did. "held" (another
# supervisor has it) and "unavailable" (the lock plumbing itself failed) both
# return False, and main() logged both as orchestrator_already_running — a
# false forensic line pointing away from the real cause one line above it.
_singleton_lock_status = "unknown"   # acquired | held | unavailable


# W18: the IB API ports, with what each one means. This is the same set
# .env's own inline comment documents.
_IB_PORTS = {
    4002: "IB Gateway paper",
    4001: "IB Gateway live",
    7497: "TWS paper",
    7496: "TWS live",
}


def _port_hint(configured: int) -> str:
    """If the configured port is dead but another IB port answers, name it.

    W18: after the operator moved from TWS to IB Gateway, .env still said
    IBKR_PORT=7497 while Gateway listened on 4002. The supervisor deferred the
    bot for 9h47m behind a bare `bot_restart_deferred_gateway_down | port=7497`
    — the mismatch was measurable the entire time and nothing looked. Returns a
    leading-space fragment for splicing into an alert, or "" when there is
    nothing to add.
    """
    try:
        for port, label in _IB_PORTS.items():
            if port != configured and _gateway_listening(port):
                return (f" NOTE: nothing is on {configured}, but {port} ({label}) IS "
                        f"open — check IBKR_PORT in .env matches the app you are running.")
    except Exception:  # noqa: BLE001 — a hint must never break the alert it decorates
        pass
    return ""


def _acquire_singleton_lock(lock_path: Path | None = None) -> bool:
    """Atomically claim the single-orchestrator lock. False = already held."""
    global _singleton_lock_handle, _singleton_lock_status
    path = lock_path or (DATA_DIR / "orchestrator.lock")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+")
        fh.seek(0)  # locking() operates from the fd's current position
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            _singleton_lock_status = "held"
            return False
        try:  # informational only — who holds it (the LOCK is the guard)
            fh.truncate(0)
            fh.write(f"{os.getpid()}\n")
            fh.flush()
        except OSError:
            pass
        _singleton_lock_handle = fh
        _singleton_lock_status = "acquired"
        return True
    except Exception as e:  # noqa: BLE001
        _singleton_lock_status = "unavailable"
        # R25 plat-06 (W16 A4): this returned True — and True means "you hold
        # the lock, proceed". A permission fault, a full disk, or a missing
        # fcntl therefore ADMITTED A SECOND SUPERVISOR: two supervisors means
        # two bots on one account, the class behind the 2026-07-13 triple-fill
        # that built four accidental reverse iron condors.
        #
        # The old comment argued "lock plumbing must never keep the bot down".
        # W15 settled that trade for the sibling gates and it settles this one:
        # a stopped bot is visible, recoverable, and costs opportunity; a
        # second bot placing duplicate orders costs money and corrupts the
        # book. Fail CLOSED — and page, so the refusal is never silent.
        _log("critical", "singleton_lock_error_fail_closed", error=str(e))
        try:
            # W34: durable throttle — this process exits right after paging
            # and the keeper relaunches it every ~90s.
            if _durable_alert_gate("singleton_lock_unavailable", 1800):
                _alert(f"SINGLETON LOCK UNAVAILABLE ({type(e).__name__}: {e}). "
                       f"Refusing to start the supervisor rather than risk a "
                       f"SECOND BOT on the same account. MANUAL ACTION NEEDED — "
                       f"check data/ permissions and free disk space.")
        except Exception:  # noqa: BLE001 — paging must never flip the gate open
            pass
        return False


def _release_singleton_lock() -> None:
    """Explicit release (tests / graceful paths); process death also releases."""
    global _singleton_lock_handle
    if _singleton_lock_handle is None:
        return
    try:
        _singleton_lock_handle.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(_singleton_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(_singleton_lock_handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        _singleton_lock_handle.close()
    except OSError:
        pass
    _singleton_lock_handle = None


# ---------------------------------------------------------------------------
# Process-table scans (R16 #10 / #57)
# ---------------------------------------------------------------------------

def _scan_foreign_bot_pids() -> list[int] | None:
    """R16 #10 (+ macOS fix 2026-09-14): PIDs of any pre-existing 'ait.main'
    trading-bot process.

    An ait.main child orphaned by a master-only death survives (no Job
    object), the keeper's guard used to match only 'run_orchestrator', and a
    relaunched master would spawn a SECOND bot trading the same account/DB
    (the clientId fallback defeats IBKR's duplicate-id rejection).

    Originally shelled out to PowerShell/Get-CimInstance (Windows-only): on
    any non-Windows host that raised FileNotFoundError('powershell'), caught
    by the except below and silently read as "no foreign bot found" — the
    guard was a standing no-op on macOS/Linux, every 2-min health cycle, not
    just at startup. psutil (already a pinned dependency) replaces it
    cross-platform, no shell dependency.

    W16 A3: the scan itself now lives in ait.utils.process.scan_pids, shared
    with status.py — the 2026-09-14 fix was applied to this call site rather
    than to the idiom, so status.py kept its PowerShell copy and went on
    reporting the bot DOWN while it was UP on any non-Windows host.

    The match is now `-m` ADJACENCY, not a substring of the joined command
    line. `"ait.main" in " ".join(cmdline)` also matched `grep -r ait.main
    src/`, an editor with the file open, or an audit session — and the consumer
    of this answer REFUSES to start the trading bot, so a false positive is
    fail-closed and costs a session. A whole-argv token is not enough either:
    in `grep -r ait.main src/` the token is its own element (a W16 test caught
    exactly that). The real bot is spawned at master.py:585 as
    [sys.executable, "-m", "ait.main", "--paper"], so `-m` adjacency is exact.

    W34 (R26 foreign-bot-scan-unreadable-process-table-permits-a-second-bot):
    this failed OPEN — an unreadable process table returned [] ("no foreign
    bot"), and nothing downstream self-limits (ait.main takes no lock), so any
    psutil fault admitted a second bot on the account. W15/W16 settled this
    trade for the sibling gates: a bot that does not start is visible and
    costs opportunity; two bots cost money and corrupt the book. None now
    means "could not look" and the caller refuses. A transient fault costs one
    2-minute health cycle.
    """
    pids = scan_pids(module="ait.main")
    if pids is None:
        _log("warning", "foreign_bot_scan_failed",
             detail="process table unreadable; refusing to assume none")
        return None
    return pids


def _dashboard_cfg():
    """W69: the dashboard section of the supervisor's configuration.

    The three web-service ports and `enabled` were literals here (8501 in
    three places, 8502, 8503) while `dashboard.port` / `dashboard.enabled` in
    config.yaml were read by nothing. load_settings() returns this process's
    frozen settings (W35), so a config.yaml edited after start never changes
    what the supervisor supervises. A settings failure falls back to the
    DashboardConfig MODEL defaults — the same values — and never blocks the
    supervisor (W35: "never block the supervisor on config").
    """
    from ait.config.settings import DashboardConfig, load_settings
    try:
        return load_settings().dashboard
    except Exception as e:  # noqa: BLE001
        _log("warning", "dashboard_config_unavailable", error=str(e))
        return DashboardConfig()


def _data_collection_cfg():
    """W76: the supervisor-run data-collection section (default: everything OFF).

    Same contract as `_dashboard_cfg`: this process's frozen settings, and a settings
    failure falls back to the MODEL defaults (collection disabled) rather than blocking
    the supervisor."""
    from ait.config.settings import DataCollectionConfig, load_settings
    try:
        return load_settings().data_collection
    except Exception as e:  # noqa: BLE001
        _log("warning", "data_collection_config_unavailable", error=str(e))
        return DataCollectionConfig()


def _clear_dashboard_port(port: int) -> bool:
    """R16 #57: free the dashboard port before (re)launching Streamlit.

    An orphaned Streamlit from an ungracefully-dead supervisor holds the port
    (8501 by default; W69: dashboard.port, so the caller passes it), so
    every relaunch died within ~1s ('Port 8501 is not available') and the
    health loop blind-respawned a doomed child every 2 minutes (663 restarts
    on 08-04 alone). Kill the orphan first. This is the ONE sanctioned
    process-kill in the supervisor and it is scoped hard: only a process that
    is LISTENING on the dashboard port AND whose command line is a streamlit
    launch — never ait.main, never run_orchestrator, never any other python.
    Returns True when the port is free to bind."""
    if not _gateway_listening(port):
        return True  # nothing listening — free
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"Get-NetTCPConnection -State Listen -LocalPort {port} "
             "-ErrorAction SilentlyContinue | "
             "Select-Object -ExpandProperty OwningProcess -Unique"],
            capture_output=True, text=True, timeout=30)
        pids = [int(tok) for tok in result.stdout.split() if tok.strip().isdigit()]
    except Exception as e:  # noqa: BLE001
        _log("warning", "dashboard_port_scan_failed", port=port, error=str(e))
        return False
    for pid in pids:
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").CommandLine"],
                capture_output=True, text=True, timeout=30)
            cmdline = (r.stdout or "").strip()
        except Exception as e:  # noqa: BLE001
            _log("warning", "dashboard_port_owner_query_failed", pid=pid, error=str(e))
            continue
        low = cmdline.lower()
        if "streamlit" not in low or "ait.main" in low or "run_orchestrator" in low:
            # NOT ours to kill — leave it and report; the caller skips the
            # doomed spawn instead of crash-looping against the holder.
            _log("error", "dashboard_port_held_foreign", pid=pid,
                 cmdline=cmdline[:160])
            continue
        _log("warn", "orphan_dashboard_killing", pid=pid, port=port)
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                           capture_output=True, text=True, timeout=15)
        except Exception as e:  # noqa: BLE001
            _log("warning", "orphan_dashboard_kill_failed", pid=pid, error=str(e))
    # give the OS a moment to release the listener
    for _ in range(10):
        if not _gateway_listening(port):
            return True
        time.sleep(0.5)
    return False


def _assume_market_open_on_clock(now: datetime | None = None) -> bool:
    """Calendar-FREE conservative answer to "is the market open?".

    Used only when the NYSE calendar is unavailable. A weekday 09:30-16:00 ET
    is treated as "may be open". Resolves ET through stdlib ``zoneinfo``, not
    ``ait.utils.time`` — the latter imports ``pandas_market_calendars`` at
    module scope, so in the very scenario that calls this it may be the thing
    that is broken.

    W20d — on the DIRECTION of this heuristic, because it looks like the one
    W15 deleted: W15 removed ``weekday() < 5`` from the pre-event blackout
    because counting a holiday as a session made that gate fail OPEN. Here the
    same heuristic errs the other way. Every caller asks "is it open?" in order
    to be MORE cautious when the answer is yes, so a holiday wrongly counted as
    a session only defers a restart or keeps hang-detection armed. Never
    reuse it anywhere the true branch is the permissive one.
    """
    from zoneinfo import ZoneInfo as _ZI
    _et = _ZI("US/Eastern")
    n = now or datetime.now(_et)
    n = n.replace(tzinfo=_et) if n.tzinfo is None else n.astimezone(_et)
    if n.weekday() >= 5:
        return False
    mins = n.hour * 60 + n.minute
    return (9 * 60 + 30) <= mins < (16 * 60)


def _in_post_market_window(now: datetime | None = None,
                           grace_minutes: int = 20) -> bool:
    """R16: is NOW inside the bot's own POST_MARKET phase (+grace)?

    The bot's scheduler treats close -> close+15min as POST_MARKET and runs the
    whole EOD pipeline on entry, with no already-ran-today guard. A supervisor
    restart landing in that window therefore re-runs it. Mirrors
    bot/scheduler.py:52-63 (honouring early-close days via get_market_close)
    and adds 5 minutes of slack so a restart cannot land on the boundary.

    If the market calendar is unavailable it falls back to the regular-session
    clock (16:00 **ET**) rather than a blanket True — fail closed INSIDE the
    window, but never wedge the model-reload restart forever on a calendar
    outage.

    W20b: that fallback previously read the BOX's local clock, and this
    docstring claimed "16:00 ET, the box's local time" — true on the old ET
    host, false since the move to CEST (= ET+6). Measured on the CEST box, it
    protected 10:00-10:15 ET by accident and allowed a mid-RTH restart at
    09:05, 11:05, 13:05 and 15:05 ET.
    """
    try:
        from datetime import timedelta as _td
        from ait.utils.time import ET, get_market_close, is_trading_day, now_et
        n = now or now_et()
        if n.tzinfo is None:
            n = n.replace(tzinfo=ET)
        if not is_trading_day(n.date()):
            return False
        close_dt = datetime.combine(n.date(), get_market_close(n.date()), tzinfo=ET)
        return close_dt <= n < close_dt + _td(minutes=15 + grace_minutes)
    except Exception as e:  # noqa: BLE001
        _log("warning", "post_market_window_check_failed", error=str(e))
        # W20b: resolve ET from stdlib zoneinfo, NOT ait.utils.time — the
        # latter imports pandas_market_calendars at module scope, so in the
        # very scenario that lands us here (a broken calendar package) it may
        # be unavailable. zoneinfo has no such dependency. A naive `now` is
        # treated as ET, matching the primary path above.
        from zoneinfo import ZoneInfo as _ZI
        _et = _ZI("US/Eastern")
        n2 = now or datetime.now(_et)
        n2 = n2.replace(tzinfo=_et) if n2.tzinfo is None else n2.astimezone(_et)
        mins = n2.hour * 60 + n2.minute
        return (16 * 60) <= mins < (16 * 60 + 15 + grace_minutes)


# ---------------------------------------------------------------------------
# Bot process management
# ---------------------------------------------------------------------------

class BotManager:
    """Manages the trading bot subprocess."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._log_handle = None
        self._restarts = 0
        self._max_restarts = 10
        self._last_restart: datetime | None = None
        self._last_gateway_alert: datetime | None = None
        # W34 (R26 gateway-escalation-page-starved-by-shared-throttle): the
        # MANUAL ACTION pages shared _last_gateway_alert with the generic
        # "unreachable ... will resume automatically" page. health_check
        # stamped that latch and then handed the SAME `now` to the escalation
        # gate, so (now - now) > 30min was never true: with Telegram healthy
        # the operator got the wrong instruction for the whole outage.
        self._last_gateway_escalation: datetime | None = None
        # ops-substrate-01: a Gateway that never opens its port is an
        # ABSORBING state — see _maybe_recover_gateway.
        self._gateway_down_since: datetime | None = None
        self._gateway_recoveries = 0
        self._last_gateway_recovery: datetime | None = None


    def _maybe_recover_gateway(self, port: int, now: datetime) -> None:
        """ops-substrate-01: restart a wedged IB Gateway instead of waiting.

        Materialized twice. 2026-07-01: 5h10m down, 3h15m inside RTH.
        2026-09-08: a Gateway JVM started 08:56 and NEVER opened port 4002;
        the supervisor logged bot_restart_deferred_gateway_down every two
        minutes until a human intervened at 12:17 — 3.5 hours of a trading
        day with open positions and no exit engine. The JVM existed, so no
        process-level check saw anything wrong; only the PORT was dead.

        Deliberately conservative, because restarting a broker connection is
        not free:
          * a GRACE PERIOD first — IB's own nightly maintenance takes the
            Gateway down for 30-60 min and recovers on its own; we must not
            fight it. 10 minutes is well inside a trading session but long
            enough to ignore a blip.
          * a COOLDOWN between attempts, so a Gateway that cannot come up
            (expired credentials, 2FA prompt) is not thrashed.
          * a BUDGET, then escalate. Looping forever is how the original
            defect behaved; a human must be told when automation is out of
            ideas.
          * the JVM is KILLED FIRST when one exists — the 09-08 shape was a
            live-but-wedged process, and a relaunch cannot bind the port
            while it holds it.
        """
        down_for = (now - (self._gateway_down_since or now)).total_seconds()
        if down_for < GATEWAY_RECOVERY_AFTER_S:
            return
        if (self._last_gateway_recovery is not None
                and (now - self._last_gateway_recovery).total_seconds()
                < GATEWAY_RECOVERY_COOLDOWN_S):
            return
        if self._gateway_recoveries >= GATEWAY_MAX_RECOVERIES:
            if (self._last_gateway_escalation is None
                    or (now - self._last_gateway_escalation) > timedelta(minutes=30)):
                # W17 w17-01b: consume the throttle only on a DELIVERED page.
                # _alert (:64) is single-shot — one urlopen, no retry, no
                # dead-channel accounting — and returns False on a dropped
                # send. Setting the timestamp FIRST meant one failed packet
                # bought 30 minutes of silence on the page that exists
                # because of an 8h36m silent outage.
                if _alert(f"IB Gateway STILL down on port {port} after "
                          f"{GATEWAY_MAX_RECOVERIES} automatic restart attempts "
                          f"({down_for / 60:.0f} min). MANUAL ACTION NEEDED — "
                          f"check for a login/2FA prompt on the Gateway window."):
                    self._last_gateway_escalation = now
            return

        script = GATEWAY_START_SCRIPT
        if not os.path.exists(script):
            # R25 plat-01: this returned SILENTLY, and the return sits BEFORE
            # the counter increment below — so _gateway_recoveries never moved
            # and the escalation above ("MANUAL ACTION NEEDED") was unreachable
            # on any host without the start script. Materialized 2026-09-15:
            # TWS was closed by accident at 07:33 and the supervisor deferred
            # for 8h36m — 134 of these log lines, ZERO pages, ~40 min of it
            # inside RTH. Nothing can automate away a broker app a human
            # closed, so for this shape the page IS the recovery.
            # The counter is deliberately NOT incremented (no attempt was
            # made); this shares the 30-min throttle above so the 2-min health
            # loop cannot spam.
            _log("warn", "gateway_recovery_unavailable", script=script)
            if (self._last_gateway_escalation is None
                    or (now - self._last_gateway_escalation) > timedelta(minutes=30)):
                # W17 w17-01b: consume the throttle only on a DELIVERED page.
                # _alert (:64) is single-shot — one urlopen, no retry, no
                # dead-channel accounting — and returns False on a dropped
                # send. Setting the timestamp FIRST meant one failed packet
                # bought 30 minutes of silence on the page that exists
                # because of an 8h36m silent outage.
                if _alert(f"IB Gateway/TWS has not answered on port {port} for "
                          f"{down_for / 60:.0f} min, and NO automated recovery is "
                          f"configured on this host ({script} not found). "
                          f"MANUAL ACTION NEEDED — start TWS/Gateway; the bot "
                          f"resumes on its own once the port answers."):
                    self._last_gateway_escalation = now
            return

        # W34 (R26 wedged-jvm-kill): the counter used to move BEFORE the try,
        # so a kill/launch that raised still spent budget — three no-ops then
        # produced an escalation claiming "3 automatic restart attempts". The
        # COOLDOWN applies to a failure too (else a broken launcher is retried
        # every 2-minute health check); the BUDGET counts only real launches.
        self._last_gateway_recovery = now
        wedged = _gateway_process_alive()
        jvm_pids = (_gateway_jvm_pids() or []) if wedged else []
        _log("warn", "gateway_recovery_attempt",
             attempt=self._gateway_recoveries + 1,
             down_minutes=round(down_for / 60),
             wedged_process=wedged, jvm_pids=jvm_pids, script=script)
        try:
            if jvm_pids:
                # A JVM that never opened the port holds it against a
                # relaunch. Kill exactly the Gateway JVMs that were scanned —
                # never every `java` on the box.
                for _pid in jvm_pids:
                    _kill_pid(_pid)
                time.sleep(5)
            # W34b: the SAME launcher the boot path uses, and keep its output.
            # This was Popen([script], shell=True, stdout=DEVNULL) — the
            # script's `start`-a-new-console branch. First real outage,
            # 2026-09-20: three relaunches, none produced a Gateway, and
            # nothing recorded why (see gateway.launch_ibc_gateway).
            from ait.orchestration.gateway import launch_ibc_gateway
            _launched = launch_ibc_gateway(script, log_dir=LOGS_DIR)
            _log("info", "gateway_recovery_launched",
                 pid=getattr(_launched, "pid", None),
                 output="logs/gateway_launch_*.log")
        except Exception as e:  # noqa: BLE001 — recovery must never kill the supervisor
            _log("error", "gateway_recovery_failed", error=str(e))
            if _alert_gate("gateway_launch_failed", 1800):
                _alert(f"IB Gateway has been down {down_for / 60:.0f} min and the "
                       f"automatic restart could not be launched "
                       f"({type(e).__name__}: {e}). MANUAL ACTION NEEDED — "
                       f"start the Gateway by hand.")
            return
        self._gateway_recoveries += 1
        _alert(f"IB Gateway was down {down_for / 60:.0f} min — attempted "
               f"automatic restart ({self._gateway_recoveries}/"
               f"{GATEWAY_MAX_RECOVERIES}). The bot resumes on its own "
               f"once port {port} answers.")

    def _note_gateway_up(self) -> None:
        """Gateway is answering: clear the outage state so a LATER outage
        re-alerts and gets a fresh recovery budget."""
        self._last_gateway_alert = None
        self._last_gateway_escalation = None
        if self._gateway_down_since is not None:
            _log("info", "gateway_recovered",
                 down_seconds=round(
                     (datetime.now() - self._gateway_down_since).total_seconds()),
                 recoveries_used=self._gateway_recoveries)
            self._gateway_down_since = None
            self._gateway_recoveries = 0

    def _check_gateway_while_bot_runs(self) -> None:
        """W34 (2026-09-17 incident): the Gateway port was examined ONLY on
        the bot-DOWN branch. At 07:59 ET the Gateway window was closed by
        mistake; the bot process stayed up, so every health check logged
        bot_healthy and recovery never ran — a running bot that cannot mark,
        trade or exit is not healthy. No page of its own before the grace
        period (IBC's nightly exit would make it a nightly page); after it,
        _maybe_recover_gateway relaunches and pages exactly as it does when
        the bot is down.
        """
        try:
            gw_port = int(os.environ.get("IBKR_PORT", "4002"))
            if _gateway_listening(gw_port):
                self._note_gateway_up()
                return
            now = datetime.now()
            if self._gateway_down_since is None:
                self._gateway_down_since = now
            _log("warn", "gateway_down_while_bot_running", port=gw_port,
                 down_seconds=round(
                     (now - self._gateway_down_since).total_seconds()))
            self._maybe_recover_gateway(gw_port, now)
        except Exception as e:  # noqa: BLE001 — never break the health loop
            _log("warning", "gateway_check_while_running_failed", error=str(e))

    @property
    def is_running(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def start(self):
        if self.is_running:
            _log("info", "bot_already_running", pid=self._proc.pid)
            return

        # R16 #10: REFUSE to spawn a second trading bot. is_running only knows
        # about OUR OWN Popen — an ait.main orphaned by a previous master's
        # death is invisible to it, and spawning alongside it means two bots
        # trading the same account/DB (the clientId fallback lets the second
        # one connect). Refuse loudly instead; the operator must resolve the
        # orphan (it is a live trading process — never auto-kill a bot).
        foreign = _scan_foreign_bot_pids()
        if foreign is None:
            _log("critical", "bot_start_refused_process_table_unreadable")
            if _alert_gate("foreign_bot_scan_unreadable", 1800):
                _alert(
                    "REFUSING to start the trading bot: the process table "
                    "could not be read, so I cannot rule out a bot that is "
                    "already running — and two bots on one account is the "
                    "07-13 duplicate-order class. Retrying every health "
                    "cycle; if this persists, check psutil/permissions on "
                    "the host."
                )
            return
        if foreign:
            _log("critical", "bot_start_refused_foreign_bot_running",
                 pids=foreign)
            if _alert_gate("foreign_bot_refusal", 1800):
                _alert(
                    f"REFUSING to start the trading bot: a pre-existing "
                    f"ait.main process (PID {foreign}) is already running — "
                    f"starting another would mean TWO bots trading the same "
                    f"account. Kill the orphan manually (verify the PID) and "
                    f"the supervisor will start a fresh bot on the next "
                    f"health cycle."
                )
            return

        # Ensure IB Gateway is running before starting bot
        from ait.orchestration.gateway import ensure_gateway
        gw_port = int(os.environ.get("IBKR_PORT", "4002"))
        if not ensure_gateway(port=gw_port, log_dir=LOGS_DIR):
            _log("error", "bot_start_aborted", reason="gateway_not_available")
            # Alert at most once per 30 min so a long Gateway outage pings
            # once rather than every 2-min health check.
            now = datetime.now()
            if (self._last_gateway_alert is None
                    or (now - self._last_gateway_alert) > timedelta(minutes=30)):
                # W18: consume the throttle only on a DELIVERED page. W17's
                # w17-01b fixed this ordering in _maybe_recover_gateway and
                # claimed the class; it covered 2 of the 4 sites sharing
                # _last_gateway_alert. This is one of the two it missed.
                if _alert(f"cannot start bot — IB Gateway not available on port "
                          f"{gw_port}.{_port_hint(gw_port)} Is Gateway/TWS logged in?"):
                    self._last_gateway_alert = now
            return

        _log("info", "bot_starting")
        # Size-capped rotation for the bot's stdout sink. This file grew to
        # 2 GB unbounded (audit 2026-07-07 item 1.3): raw append, never
        # rotated, and mtime-based cleanup can never touch a continuously
        # written file. A subprocess stdout fd can't use RotatingFileHandler,
        # so rotate here at start time — the only moment the handle is closed
        # (Windows cannot rename an open file). Bot restarts are frequent
        # enough that this bounds growth in practice.
        # R5 audit F10: the crash-restart path (health_check -> start())
        # opened a new handle without closing the previous one — the leaked
        # handle kept the file open, every rotation rename failed (WinError
        # 32), and rotation had NEVER succeeded (98MB log, destroyed crash
        # dumps, the 2GB incident). Close before rotating.
        if getattr(self, "_log_handle", None):
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        self._rotate_stdout_log()
        self._log_handle = open(LOGS_DIR / "bot_stdout.log", "a")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "ait.main", "--paper"],
            cwd=str(ROOT),
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self._last_restart = datetime.now()
        _log("info", "bot_started", pid=self._proc.pid)
        self._verify_launch()

    _LAUNCH_GRACE_S = 8.0  # a Popen that dies within this window never came up

    def _verify_launch(self) -> bool:
        """R14 #11: confirm the child actually STAYED up after Popen.

        `start()` used to Popen and return, assuming success. A process that
        launches and dies a second later (bad import, port bind, Gateway drop
        mid-init) then read as 'started' until the NEXT 2-minute health cycle —
        the keeper's whole reason to exist, blind for 2 minutes each time. Poll
        briefly: if the child exits inside the grace window, it never came up,
        so alert immediately (with the stdout tail hint) instead of reporting a
        healthy PID. Returns True if it survived the window."""
        deadline = time.time() + self._LAUNCH_GRACE_S
        while time.time() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                code = self._proc.returncode if self._proc else "no_proc"
                _log("error", "bot_launch_failed_immediately", exit_code=code)
                _alert(
                    f"BOT RELAUNCH FAILED: the process exited within "
                    f"{self._LAUNCH_GRACE_S:.0f}s of launch (exit_code={code}). "
                    f"It is NOT running — check logs/bot_stdout.log. The keeper "
                    f"will keep retrying."
                )
                return False
            time.sleep(0.5)
        _log("info", "bot_launch_verified", pid=self._proc.pid)
        return True

    _STDOUT_LOG_MAX_BYTES = 50 * 1024 * 1024  # rotate above 50 MB
    _STDOUT_LOG_BACKUPS = 2                    # keep .1 and .2

    def _rotate_stdout_log(self) -> None:
        """Shift bot_stdout.log -> .1 -> .2 when over the size cap."""
        base = LOGS_DIR / "bot_stdout.log"
        try:
            if not base.exists() or base.stat().st_size <= self._STDOUT_LOG_MAX_BYTES:
                return
            oldest = LOGS_DIR / f"bot_stdout.log.{self._STDOUT_LOG_BACKUPS}"
            if oldest.exists():
                oldest.unlink()
            for i in range(self._STDOUT_LOG_BACKUPS - 1, 0, -1):
                src = LOGS_DIR / f"bot_stdout.log.{i}"
                if src.exists():
                    src.rename(LOGS_DIR / f"bot_stdout.log.{i + 1}")
            base.rename(LOGS_DIR / "bot_stdout.log.1")
            _log("info", "stdout_log_rotated")
        except Exception as e:  # rotation must never block a bot start
            _log("warning", "stdout_log_rotate_failed", error=str(e))

    def stop(self):
        if not self.is_running:
            return
        _log("info", "bot_stopping", pid=self._proc.pid)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None
        _log("info", "bot_stopped")

    def health_check(self):
        """Check if bot is alive, restart if crashed."""
        if self.is_running:
            _log("debug", "bot_healthy", pid=self._proc.pid)
            self._check_gateway_while_bot_runs()
            # R6 (user-approved): HANG detection — process-exists is not
            # process-alive. The bot touches data/bot_heartbeat every ~30s
            # while its trading loop runs; stale >15 min DURING MARKET HOURS
            # means a hung event loop (stops/TPs silently off on the whole
            # book) — the exact failure class all three supervision layers
            # were blind to (R5-D1). Restart + alert.
            try:
                _hb = DATA_DIR / "bot_heartbeat"
                try:
                    from ait.utils.time import is_market_open as _imo_hb
                    _mkt_open = _imo_hb()
                except Exception as _ce:  # noqa: BLE001
                    # W20d, same class as the marker site below: a calendar
                    # fault here escaped to the OUTER handler and silently
                    # disarmed hang detection for the whole outage — a hung
                    # event loop mid-session would then go unnoticed, which is
                    # the exact failure class (R5-D1) this check exists for.
                    # Fall back to the calendar-free clock proxy so the check
                    # stays armed through a weekday RTH.
                    _log("warning", "heartbeat_market_check_failed",
                         error=str(_ce))
                    _mkt_open = _assume_market_open_on_clock()
                if _mkt_open and _hb.exists():
                    _now_hb = datetime.now()
                    _age_min = (_now_hb - datetime.fromtimestamp(
                        _hb.stat().st_mtime)).total_seconds() / 60
                    if _age_min > 15:
                        # W46 (bot-day-03): a startup-anchored grace before
                        # this branch may even fire. Without it, a boot whose
                        # first heartbeat write takes longer than 15 min got
                        # killed on THIS tick, and the retry repeated the
                        # same slow boot -- killed again next tick, forever.
                        # _last_restart is None only when this supervisor
                        # instance never called start() itself; that gets NO
                        # grace, falling straight through to today's
                        # behavior (a genuinely long-stale heartbeat with no
                        # restart-time reference has nothing to be lenient
                        # about).
                        if (self._last_restart is not None
                                and (_now_hb - self._last_restart).total_seconds()
                                < BOT_RESTART_GRACE_S):
                            _log("info", "hang_check_grace_skipped",
                                 age_min=round(_age_min),
                                 since_restart_s=int(
                                     (_now_hb - self._last_restart).total_seconds()))
                            return
                        _log("error", "bot_hung_heartbeat_stale",
                             age_min=round(_age_min))
                        if _alert_gate("bot_hung_heartbeat_stale", 1800.0):
                            _alert(f"BOT HUNG: trading-loop heartbeat {_age_min:.0f} "
                                   f"min stale during market hours — restarting the "
                                   f"bot. Positions were unmanaged for that window.")
                        # W46: spend the SAME restart budget every other
                        # recovery path in this file uses, instead of
                        # bypassing it via the `return` below -- a genuine
                        # hang (not just a slow-but-healthy boot, which the
                        # grace above already exempts) must eventually
                        # escalate to a human, not loop forever every 2 min
                        # with the keeper reporting green throughout.
                        if self._restarts >= self._max_restarts:
                            _log("error", "bot_max_restarts_reached",
                                 max=self._max_restarts)
                            _alert(f"BOT DOWN — gave up after "
                                   f"{self._max_restarts} restart attempts "
                                   f"(last cause: hung heartbeat). Manual "
                                   f"intervention needed; check bot_stdout.log.")
                            return
                        # W34 lesson (see _maybe_recover_gateway): the budget
                        # counts only real attempts. No wrapping try/except
                        # here on purpose -- if stop()/start() raises, this
                        # increment is never reached and the exception
                        # propagates to the existing outer handler below,
                        # exactly as it already does today.
                        self.stop()
                        self.start()
                        self._restarts += 1
                        return
            except Exception as _e:  # noqa: BLE001
                _log("warning", "heartbeat_check_failed", error=str(_e))
            # Fresh-models marker (audit item 3.1): the daily retrain wrote new
            # models the running bot won't load until its 7-day timer. One
            # controlled restart picks them up. GUARDS (audit R2): (a) never
            # restart during market hours — a slow retrain can outlive the
            # 7:30 slot and finish mid-session (only an IDLE timeout bounds
            # it); the marker simply waits until the market is closed.
            # (b) only restart if the marker was actually deleted, else a
            # locked file would restart the bot every 2 minutes forever.
            marker = ROOT / "models" / ".retrained"
            if marker.exists():
                try:
                    from ait.utils.time import is_market_open
                    market_open = is_market_open()
                except Exception as _e:  # noqa: BLE001
                    # W20d: this was a bare `market_open = False`, which FAILS
                    # OPEN — "I don't know" unlocked the restart branch below,
                    # so a single calendar fault restarts the LIVE bot
                    # mid-session with positions open. Probed 2026-09-16 on
                    # this CEST box: with the calendar faulted, 09:05, 10:05,
                    # 11:05, 13:05 and 15:05 ET ALL took the restart branch.
                    # Note W20b alone does not close this — making
                    # _in_post_market_window timezone-correct fixes the
                    # function and leaves the hole, because returning False
                    # mid-RTH is the *right* answer to "are we post-market?".
                    # The hole is here, in the caller.
                    _log("warning", "fresh_models_market_check_failed",
                         error=str(_e))
                    market_open = _assume_market_open_on_clock()
                if market_open:
                    _log("info", "fresh_models_restart_deferred_market_open")
                elif _in_post_market_window():
                    # R16: "market closed" is true the SECOND the bell rings,
                    # so the first post-16:00 health check consumed the marker
                    # and the replacement bot started inside the bot's own
                    # POST_MARKET phase (close -> close+15min, scheduler.py:
                    # 52-63). Its run() then re-ran _post_market: a second EOD
                    # RECON + daily summary to Telegram, a second reconcile and
                    # a second learning cycle, ~80s after the first — observed
                    # on 5 of the last 8 sessions (07-28, 07-29, 07-31, 08-03,
                    # 08-05) and a straight re-run of the R13 duplicate-page
                    # class. Wait until the post-market window has closed; the
                    # marker survives to the next 2-minute check.
                    _log("info", "fresh_models_restart_deferred_post_market")
                else:
                    unlinked = False
                    try:
                        marker.unlink()
                        unlinked = True
                    except Exception as e:
                        _log("warning", "retrain_marker_unlink_failed", error=str(e))
                    if unlinked:
                        _log("info", "bot_restart_for_fresh_models")
                        self.stop()
                        self.start()
                        return
            # Reset the restart budget once the bot has been continuously
            # healthy for a sustained stretch. Without this, restart attempts
            # spent during a transient overnight Gateway outage are never
            # forgiven — the bot recovers and runs fine for hours, but the
            # next single blip trips max_restarts and the supervisor gives up
            # permanently (exactly what happened on 2026-06-16, FOMC eve).
            if (self._restarts > 0 and self._last_restart
                    and (datetime.now() - self._last_restart) > timedelta(minutes=30)):
                _log("info", "restart_budget_reset", was=self._restarts,
                     healthy_minutes=int((datetime.now() - self._last_restart).total_seconds() / 60))
                self._restarts = 0
            return

        exit_code = self._proc.returncode if self._proc else "never_started"
        _log("warn", "bot_down", exit_code=exit_code, restarts=self._restarts)

        # If IB Gateway itself is unreachable, the bot can't start through no
        # fault of its own — IB's daily maintenance can take it down for 30-60
        # min overnight. Do NOT spend restart budget on this: keep retrying
        # indefinitely and alert once. Spending budget here is what repeatedly
        # tripped max_restarts and left the bot dead for whole sessions
        # (incl. FOMC eve 2026-06-16 and FOMC morning 2026-06-17).
        gw_port = int(os.environ.get("IBKR_PORT", "4002"))
        if not _gateway_listening(gw_port):
            _log("info", "bot_restart_deferred_gateway_down", port=gw_port,
                 other_ib_port_open=(_port_hint(gw_port) or "none"))
            now = datetime.now()
            if self._gateway_down_since is None:
                self._gateway_down_since = now
            if (self._last_gateway_alert is None
                    or (now - self._last_gateway_alert) > timedelta(minutes=30)):
                # W18: same missed throttle-ordering site as above.
                if _alert(f"IB Gateway unreachable on port {gw_port} (likely daily "
                          f"maintenance).{_port_hint(gw_port)} Bot start deferred; "
                          f"will resume automatically when Gateway returns. "
                          f"No budget spent."):
                    self._last_gateway_alert = now
            # ops-substrate-01: deferring FOREVER is the absorbing state. After
            # a grace period, try to bring the Gateway back ourselves.
            self._maybe_recover_gateway(gw_port, now)
            return

        # Gateway is up — a fresh start attempt may have cleared whatever was
        # wrong; reset the gateway-alert latches so the next outage re-alerts,
        # and the outage timer + recovery budget so it gets fresh attempts.
        self._note_gateway_up()

        # Reset restart counter if last restart was >30 min ago — a transient
        # blip shouldn't burn the budget toward a permanent give-up.
        if self._last_restart and (datetime.now() - self._last_restart) > timedelta(minutes=30):
            self._restarts = 0

        if self._restarts >= self._max_restarts:
            _log("error", "bot_max_restarts_reached", max=self._max_restarts)
            _alert(
                f"BOT DOWN — gave up after {self._max_restarts} restart attempts "
                f"(last exit_code={exit_code}). Manual intervention needed; "
                f"check bot_stdout.log."
            )
            return

        self._restarts += 1
        _log("info", "bot_restarting", attempt=self._restarts)
        # Warn early — by restart #3 something is genuinely wrong, not a blip.
        if self._restarts >= 3:
            _alert(
                f"bot restarting (attempt {self._restarts}/{self._max_restarts}, "
                f"exit_code={exit_code}). Will alert again if it gives up."
            )
        self.start()


class WebServiceManager:
    """Manages the dashboard (Streamlit) and web log viewer (Flask) subprocesses."""

    def __init__(self):
        self._dashboard_proc: subprocess.Popen | None = None
        self._weblog_proc: subprocess.Popen | None = None
        self._status_proc: subprocess.Popen | None = None
        self._dashboard_log = None
        self._weblog_log = None
        self._status_log = None

    def start(self):
        """Start all web services."""
        self._start_dashboard()
        self._start_weblog()
        self._start_status()

    def _start_status(self):
        """Lightweight live status dashboard (dashboard.status_port, 8503 default).

        Reliable real-time view of bot health, positions, today's activity,
        and real-exit P&L — reads the DB/logs directly, no IBKR clientId
        conflict. Separate from the heavy Streamlit app (dashboard.port).
        """
        if self._status_proc and self._status_proc.poll() is None:
            return
        status_port = _dashboard_cfg().status_port
        _log("info", "status_dashboard_starting", port=status_port)
        log_path = LOGS_DIR / "status_server.log"
        self._status_log = open(log_path, "a")
        self._status_proc = subprocess.Popen(
            # W69: --port hands the child THIS process's configured port.
            [sys.executable, str(ROOT / "status_server.py"), "--port", str(status_port)],
            cwd=str(ROOT),
            stdout=self._status_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        _log("info", "status_dashboard_started", pid=self._status_proc.pid,
             url=f"http://localhost:{status_port}")

    def _start_dashboard(self):
        if self._dashboard_proc and self._dashboard_proc.poll() is None:
            return
        cfg = _dashboard_cfg()
        if not cfg.enabled:
            # W69: dashboard.enabled was read by nothing. False now means the
            # supervisor does not launch (or health-check-respawn) Streamlit.
            _log("info", "dashboard_disabled_by_config", key="dashboard.enabled")
            return
        port = cfg.port
        # R16 #57: an orphaned Streamlit (from an ungracefully-dead previous
        # supervisor) holding the dashboard port (8501 by default) made every
        # relaunch die in ~1s and the health loop respawned a doomed child
        # every 2 min, forever, silently. Clear the port first; if a
        # NON-streamlit process holds it, skip the spawn entirely (alert,
        # gated) instead of crash-looping against it.
        if not _clear_dashboard_port(port):
            _log("error", "dashboard_start_skipped_port_held", port=port)
            if _alert_gate("dashboard_port_held", 3600):
                _alert(f"dashboard NOT started: port {port} is held by a process "
                       "the supervisor won't kill — see orchestrator.log "
                       "(dashboard_port_held_foreign) for the holder.")
            return
        _log("info", "dashboard_starting", port=port)
        log_path = LOGS_DIR / "dashboard.log"
        # R16: rotate BEFORE reopening — this is the one moment no child holds
        # the handle, mirroring BotManager._rotate_stdout_log at bot start.
        _rotate_if_oversized(log_path, _ROTATE_TARGETS["dashboard.log"])
        # R5 F10 pattern: close the previous handle before reopening — the
        # crash-restart path leaked one open handle per restart (663/day
        # during the 08-04 loop).
        if self._dashboard_log:
            try:
                self._dashboard_log.close()
            except Exception:
                pass
        self._dashboard_log = open(log_path, "a")
        self._dashboard_proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run",
             str(ROOT / "src" / "ait" / "dashboard" / "app.py"),
             f"--server.port={port}", "--server.headless=true",
             # localhost ONLY (deep-audit S2): streamlit's default binds all
             # interfaces, LAN-exposing the P&L dashboard with no auth.
             "--server.address=127.0.0.1",
             "--browser.gatherUsageStats=false"],
            cwd=str(ROOT),
            stdout=self._dashboard_log,
            stderr=subprocess.STDOUT,
        )
        _log("info", "dashboard_started", pid=self._dashboard_proc.pid, log=str(log_path))

    def _start_weblog(self):
        if self._weblog_proc and self._weblog_proc.poll() is None:
            return
        weblog_port = _dashboard_cfg().weblog_port
        _log("info", "weblog_starting", port=weblog_port)
        log_path = LOGS_DIR / "weblog.log"
        _rotate_if_oversized(log_path, _ROTATE_TARGETS["weblog.log"])  # R16
        self._weblog_log = open(log_path, "a")
        self._weblog_proc = subprocess.Popen(
            # W69: --port hands the child THIS process's configured port.
            [sys.executable, str(ROOT / "web_logs.py"), "--port", str(weblog_port)],
            cwd=str(ROOT),
            stdout=self._weblog_log,
            stderr=subprocess.STDOUT,
        )
        _log("info", "weblog_started", pid=self._weblog_proc.pid, log=str(log_path))

    def health_check(self):
        """Restart web services if they crashed."""
        if self._dashboard_proc and self._dashboard_proc.poll() is not None:
            _log("warn", "dashboard_crashed", restarting=True)
            self._start_dashboard()
        if self._weblog_proc and self._weblog_proc.poll() is not None:
            _log("warn", "weblog_crashed", restarting=True)
            self._start_weblog()
        if self._status_proc and self._status_proc.poll() is not None:
            _log("warn", "status_dashboard_crashed", restarting=True)
            self._start_status()

    def stop(self):
        for name, proc in [("dashboard", self._dashboard_proc), ("weblog", self._weblog_proc),
                           ("status", self._status_proc)]:
            if proc and proc.poll() is None:
                _log("info", f"{name}_stopping", pid=proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        for handle in (self._dashboard_log, self._weblog_log):
            if handle:
                try:
                    handle.close()
                except Exception:
                    pass
        self._dashboard_proc = None
        self._weblog_proc = None
        self._dashboard_log = None
        self._weblog_log = None
        _log("info", "web_services_stopped")


# ---------------------------------------------------------------------------
# Scheduled tasks
# ---------------------------------------------------------------------------

def run_backtest():
    """Weekly walk-forward backtest to track strategy health."""
    _log("info", "backtest_starting")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"backtest_{ts}.json"

    try:
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "run_backtest.py"),
                # 2026-08-03: align the weekly HEALTH backtest with what
                # live actually trades (IC-only, index ETFs, k=0.8) — it was
                # still testing the retired AAPL/MSFT/NVDA mixed-strategy
                # universe, so its Sunday signal tracked a phantom strategy.
                # This is drift TRACKING, not decision-making: params stay
                # locked by the pre-registered rules regardless of results.
                "--symbols", "SPY", "QQQ", "IWM",
                "--strategies", "iron_condor",
                # W-capital: no --capital here -- run_backtest resolves
                # config.yaml account.capital (was a hardcoded "50000").
                # W65: run_backtest now defaults to the IB research history
                # (2024-01-01 -> today, 10 windows, ~12 min measured) which would
                # blow this job's 600 s timeout; the health signal only needs the
                # recent windows (4 -> ~5 min). Drift TRACKING, not decision-making.
                "--max-windows", "4",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )

        # Save full output
        output_path = LOGS_DIR / f"backtest_{ts}.log"
        with open(output_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        # Parse key metrics from output
        metrics = _parse_backtest_output(result.stdout)
        metrics["timestamp"] = ts
        metrics["exit_code"] = result.returncode

        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)

        _log("info", "backtest_complete",
             total_return=metrics.get("total_return", "?"),
             sharpe=metrics.get("sharpe", "?"),
             win_rate=metrics.get("win_rate", "?"),
             report=str(report_path))

        # Append to tracking history
        _append_health_metric(metrics)

    except subprocess.TimeoutExpired:
        _log("error", "backtest_timeout")
    except Exception as e:
        _log("error", "backtest_failed", error=str(e))


def retrain_models():
    """Deep model retrain — full walk-forward on all symbols."""
    _log("info", "retrain_starting")
    try:
        # Stream output live so user can see progress (and we don't silently
        # time out with no clue what happened). Log to file AND terminal.
        log_path = LOGS_DIR / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", """
import sys
sys.path.insert(0, 'src')
from ait.config.settings import load_settings
from ait.data.market_data import MarketDataService
from ait.data.historical import HistoricalDataStore
from ait.ml.ensemble import DirectionPredictor
from ait.ml.range_predictor import RangePredictor
from ait.ml.vol_magnitude_predictor import VolMagnitudePredictor
from ait.ml.trainer import ModelTrainer
import asyncio

settings = load_settings('config.yaml')
predictor = DirectionPredictor(settings.ml)
# W35: this was RangePredictor(threshold_pct=0.05, horizon_days=30) — bare
# literals. The live bot builds its range model from ait.ml.range_spec (0.05,
# horizon = the reachable hold, 9d), so every daily retrain rebuilt range.pkl
# at a spec the live gate's confidence floor was never calibrated for, and the
# bot's spec guard flagged its own fresh artifact each morning. One authority.
from ait.ml.range_spec import live_range_spec
_range_threshold, _range_horizon = live_range_spec(settings)
range_pred = RangePredictor(threshold_pct=_range_threshold, horizon_days=_range_horizon)
vol_mag_pred = VolMagnitudePredictor(threshold_pct=0.07, horizon_days=30)
market_data = MarketDataService(None)   # W65: no IB client here -> daily bars from Yahoo; no Polygon
historical = HistoricalDataStore()
trainer = ModelTrainer(
    settings.ml, predictor, market_data, historical,
    range_predictor=range_pred,
    vol_mag_predictor=vol_mag_pred,
)

async def train():
    symbols = list(settings.trading.universe)
    print(f"[retrain] Training {len(symbols)} symbols...", flush=True)
    results = await trainer.train_all_symbols(symbols)
    for sym, acc in results.items():
        print(f"Trained: {sym} -> {acc}", flush=True)

asyncio.run(train())
"""],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # OpenMP pinning (audit 2026-07-07 item 3.1): xgboost+lightgbm in
            # an unpinned subprocess is the exact native-crash class already
            # seen live (access violation in xgboost core.update). Also stops
            # full-core saturation while the trading bot runs.
            env={**os.environ,
                 "PYTHONIOENCODING": "utf-8",
                 "KMP_DUPLICATE_LIB_OK": "TRUE",
                 "OMP_NUM_THREADS": "1",
                 "OPENBLAS_NUM_THREADS": "1"},
        )

        # Idle timeout: kill only if no output for 5 minutes (truly hung).
        # Legitimate slow training on older hardware stays alive as long as
        # it's producing progress output.
        import threading, time
        IDLE_LIMIT = 300  # 5 min of silence = assumed hung
        last_output = {"ts": time.time()}
        timed_out = {"flag": False}

        def _idle_killer():
            while proc.poll() is None:
                time.sleep(15)
                if time.time() - last_output["ts"] > IDLE_LIMIT:
                    timed_out["flag"] = True
                    proc.kill()
                    return

        killer_thread = threading.Thread(target=_idle_killer, daemon=True)
        killer_thread.start()

        # Stream output live, save to log file, print to terminal
        with open(log_path, "w") as log_file:
            for line in proc.stdout:
                last_output["ts"] = time.time()  # reset idle timer on activity
                line = line.rstrip()
                if line:
                    print(f"[retrain] {line}", flush=True)
                    log_file.write(line + "\n")
                    log_file.flush()
        proc.wait()

        if timed_out["flag"]:
            _log("error", "retrain_timeout_idle", log=str(log_path),
                 idle_limit_seconds=IDLE_LIMIT)
            _alert(f"model retrain HUNG (no output {IDLE_LIMIT}s) — killed. "
                   f"Models are stale. Log: {log_path.name}")
            return

        if proc.returncode != 0:
            # A hard native crash (OpenMP/DLL segfault) exits nonzero without
            # hanging — previously logged silently, leaving stale models with
            # no one the wiser (audit item 3.1).
            _log("error", "retrain_failed_nonzero", exit_code=proc.returncode,
                 log=str(log_path))
            _alert(f"model retrain FAILED (exit {proc.returncode}) — models "
                   f"stale. Log: {log_path.name}")
            return

        _log("info", "retrain_complete", exit_code=proc.returncode, log=str(log_path))
        # Schedule-unification (audit item 3.1): fresh models were written to
        # disk, but the live bot only reloads on its own 7-day timer — daily
        # retrains sat unused. Drop a marker; BotManager.health_check restarts
        # the bot once (pre-market, harmless) so it loads today's models.
        try:
            (ROOT / "models" / ".retrained").write_text(datetime.now().isoformat())
        except Exception as e:
            _log("warning", "retrain_marker_failed", error=str(e))
    except Exception as e:
        _log("error", "retrain_failed", error=str(e))


def daily_report():
    """Generate daily P&L and performance summary."""
    _log("info", "daily_report_starting")
    try:
        ts = datetime.now().strftime("%Y%m%d")
        report_path = REPORTS_DIR / f"daily_{ts}.json"
        # R16 #58: the child used to PRINT the JSON and the parent saved raw
        # stdout — but TradeAnalytics() emits a structlog line to stdout
        # first ('duckdb_initialized ...'), so every daily_*.json since
        # 07-02 began with a log line and json.load failed on all of them.
        # json.dump straight to the report file instead; stdout stays
        # log-only and never touches the artifact.
        result = subprocess.run(
            [sys.executable, "-c", """
import sys, json
sys.path.insert(0, 'src')
from ait.monitoring.analytics import TradeAnalytics

analytics = TradeAnalytics()
metrics = analytics.get_performance(lookback_days=1)
payload = metrics.__dict__ if hasattr(metrics, '__dict__') else metrics
with open(sys.argv[1], 'w') as f:
    json.dump(payload, f, indent=2, default=str)
""", str(report_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Deep-audit C1: this called a method that never existed
        # (get_performance_metrics), wrote the empty stdout, and logged
        # success — the 4:30 report has been an empty file since day one.
        # Gate on the child actually succeeding.
        if result.returncode != 0:
            _log("error", "daily_report_child_failed",
                 returncode=result.returncode,
                 stderr=(result.stderr or "")[:400])
            return

        # R16 #58: and gate on the artifact PARSING — 'complete' was logged
        # for 22 straight invalid files; never again without a json.load.
        try:
            with open(report_path) as f:
                json.load(f)
        except (OSError, ValueError) as e:
            _log("error", "daily_report_invalid_json", report=str(report_path),
                 error=str(e)[:200])
            return

        _log("info", "daily_report_complete", report=str(report_path))
    except Exception as e:
        _log("error", "daily_report_failed", error=str(e))


def publish_shared_logs():
    """Shared per-machine logs (2026-09-29, agreed by both operators): export
    this box's bot events for today (and any of the last 4 weekdays missed while
    the box was off) to the `shared-logs` orphan branch, as
    `<AIT_LOG_OWNER>_logs/YYYY-MM-DD.jsonl.gz`. Runs the publisher as a SUBPROCESS
    with a hard timeout so git/network trouble can never stall the scheduler;
    never touches the live tree's own branch or files (see shared_logs.py)."""
    from ait.orchestration.shared_logs import resolve_owner
    if not resolve_owner():
        _log("DEBUG", "shared_logs_publish_skipped_no_owner")
        return
    try:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "publish_shared_logs.py")],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600)
        tail = (r.stdout or r.stderr or "").strip()[-400:]
        _log("INFO" if r.returncode == 0 else "WARNING",
             "shared_logs_published" if r.returncode == 0 else "shared_logs_publish_failed",
             rc=r.returncode, result=tail)
    except Exception as e:  # noqa: BLE001 — publishing must never hurt the supervisor
        _log("WARNING", "shared_logs_publish_failed", error=str(e))


def fetch_shared_logs():
    """W70 follow-up: refresh `origin/shared-logs` in this checkout, 5 minutes after the
    16:45 publish, so the OTHER box's published days are readable locally without anyone
    running read_shared_logs.py first.

    A fetch only (no checkout/merge; see shared_logs.fetch_shared_branch). Deliberately NOT
    gated on AIT_LOG_OWNER: reading a peer's logs does not require publishing your own.
    Logs the newest published day per owner (a stalled box shows up as a stale date); no
    alert — a quiet peer is information, not a page. Never raises."""
    try:
        from ait.orchestration.shared_logs import fetch_shared_branch
        res = fetch_shared_branch(ROOT)
        if res.ok:
            _log("INFO", "shared_logs_fetched", head=res.head, advanced=res.advanced,
                 new_commits=res.new_commits, latest_by_owner=res.latest_by_owner)
        else:
            _log("WARNING", "shared_logs_fetch_failed", error=res.error)
    except Exception as e:  # noqa: BLE001 — must never hurt the supervisor
        _log("WARNING", "shared_logs_fetch_failed", error=str(e))


def collect_short_dte_friction():
    """W76: run the 0-6 DTE friction collector, unattended, after the close.

    A thin wrapper: the whole job (guards, command, idle/absolute timeout, outcome
    classification) lives in ait.orchestration.short_dte_job and is tested without a
    broker. Failure-only alerts, rate-limited to one per day; a healthy run is logged only
    (progress is in data/short_dte_collection_status.json). Never raises: a collection
    problem must never hurt the supervisor."""
    try:
        from ait.config.settings import IBKREnvConfig, load_settings
        from ait.orchestration.short_dte_job import run_collection
        from ait.utils.time import is_trading_day, today_et

        cfg = _data_collection_cfg()
        env = IBKREnvConfig()
        port = int(os.environ.get("IBKR_PORT") or env.ibkr_port)
        base_id = int(os.environ.get("IBKR_CLIENT_ID") or env.ibkr_client_id)
        try:
            min_sessions = int(load_settings().backtest.friction_min_sessions_per_band)
        except Exception:  # noqa: BLE001
            min_sessions = 15
        _log("info", "short_dte_collection_starting", symbols=",".join(cfg.short_dte_symbols),
             port=port, client_id=cfg.short_dte_client_id)
        out = run_collection(
            cfg, root=ROOT, data_dir=DATA_DIR, logs_dir=LOGS_DIR, min_sessions=min_sessions,
            ibkr_port=port, base_client_id=base_id, is_trading_day=is_trading_day,
            gateway_listening=_gateway_listening, today=today_et())
        _log("INFO" if out.status in ("ok", "skipped_not_trading_day") else "WARNING",
             f"short_dte_collection_{out.status}", detail=out.detail, stored=out.stored,
             planned=out.planned, bands=out.bands, log=out.log_path)
        if out.alert_worthy and _alert_gate("short_dte_collection", 20 * 3600):
            _alert(f"Short-DTE friction collection {out.status}: {out.detail}. "
                   f"Today's samples may be missing for good (IB keeps nothing for expired "
                   f"contracts). Log: {out.log_path or 'n/a'}")
    except Exception as e:  # noqa: BLE001 - must never hurt the supervisor
        _log("WARNING", "short_dte_collection_failed", error=str(e))


def backup_state_db():
    """INST-1 (institutional audit): the entire real track record — the whole
    basis of the go/no-go decision — lives in ONE unreplicated SQLite file on
    the box with the known process-killer. Online-safe daily backup via
    sqlite3's .backup API; keeps 14 dailies locally + mirrors the latest to a
    second directory outside the repo (sync that folder to cloud for off-box)."""
    import sqlite3 as _sq
    try:
        src = DATA_DIR / "ait_state.db"
        if not src.exists():
            return
        bdir = DATA_DIR / "backups"
        bdir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d")
        dest = bdir / f"ait_state.{ts}.db"
        con = _sq.connect(str(src))
        out = _sq.connect(str(dest))
        try:
            con.backup(out)
            # R6: verify the snapshot — a corrupt backup discovered at
            # restore time is unrecoverable. integrity_check + row parity on
            # the table that IS the track record.
            _ok = out.execute("PRAGMA integrity_check").fetchone()[0]
            _n_src = con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            _n_dst = out.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            if _ok != "ok" or _n_src != _n_dst:
                raise RuntimeError(
                    f"backup verification failed: integrity={_ok}, "
                    f"trades {_n_src} -> {_n_dst}")
        finally:
            # R13 (human-factors): `with sqlite3.connect(...)` manages the
            # TRANSACTION, not the connection — the destination file could be
            # un-finalized when copy2 read it, and on 07-14 the mirror was
            # found a full run stale (missing all of Monday's closes) while
            # every check read green. Close BEFORE the mirror copy.
            con.close()
            out.close()
        # prune to 14 most recent
        snaps = sorted(bdir.glob("ait_state.*.db"))
        for old_f in snaps[:-14]:
            old_f.unlink(missing_ok=True)
        # mirror latest outside the repo tree (second failure domain)
        mirror_dir = Path.home() / "Documents" / "ait_backups"
        mirror_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        import hashlib
        mirror = mirror_dir / "ait_state.latest.db"
        shutil.copy2(dest, mirror)
        # R13: verify the mirror BY CONTENT — copy2 back-dates the mtime, so
        # the RUNBOOK's timestamp check could not catch a stale/partial copy.
        # A mismatch raises into the existing BACKUP FAILED alert path.
        def _sha(p):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        if _sha(dest) != _sha(mirror):
            raise RuntimeError("mirror hash mismatch after copy — "
                               f"{mirror} does not match {dest}")
        _log("info", "state_db_backed_up", dest=str(dest))
    except Exception as e:  # noqa: BLE001
        _log("error", "state_db_backup_failed", error=str(e))
        _alert(f"STATE DB BACKUP FAILED: {e} — the track record is unprotected.")


def _newest_trade_ts(db_path: Path) -> str | None:
    """Newest entry_time/exit_time in a state DB, read-only. None if unusable."""
    import sqlite3 as _sq
    if not db_path.exists():
        return None
    try:
        con = _sq.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT MAX(ts) FROM ("
                "  SELECT MAX(entry_time) ts FROM trades"
                "  UNION ALL SELECT MAX(exit_time) FROM trades"
                ")"
            ).fetchone()
        finally:
            con.close()
        return str(row[0]) if row and row[0] else None
    except Exception as e:  # noqa: BLE001
        _log("warning", "newest_trade_ts_failed", db=str(db_path), error=str(e))
        return None


def _mirror_lag() -> tuple[str, float | None]:
    """R16: is the off-box mirror BEHIND the live DB? (label, lag_hours).

    _mirror_content_age_hours below measures now - newest trade IN the mirror,
    which is last-TRADE recency, not mirror freshness: during any no-trade
    stretch (entries blacked out pre-NFP, positions holding) the digest reported
    an ever-growing "mirror 69h stale" while the nightly hash-verified copy was
    <16h old — training the operator to ignore the line — and a genuinely stale
    mirror during a quiet week read identically to a healthy one.

    The real question is whether the mirror's content MATCHES the source, which
    is exactly the 07-14 failure signature. Labels:
      "current"  — mirror newest trade == live newest trade
      "behind"   — mirror is older than live, lag_hours = the gap
      "missing"  — mirror absent/unreadable
      "unknown"  — live DB unreadable, so no comparison is possible
    """
    mirror = Path.home() / "Documents" / "ait_backups" / "ait_state.latest.db"
    if not mirror.exists():
        return ("missing", None)
    m_ts = _newest_trade_ts(mirror)
    s_ts = _newest_trade_ts(DATA_DIR / "ait_state.db")
    if m_ts is None:
        return ("missing", None)
    if s_ts is None:
        return ("unknown", None)
    if m_ts == s_ts:
        return ("current", 0.0)
    try:
        # W38 (R26 `w21-mirror-lag-is-an-unmigrated-reader`): these were bare
        # `fromisoformat` calls, so once W21 made the live corpus aware while a
        # mirror still held naive rows (or the reverse, after a restore),
        # subtracting them raised TypeError into the except below and the
        # digest lost the NUMBER — degrading to a bare "mirror BEHIND live"
        # exactly when the mirror IS behind, so the operator could not tell a
        # one-run lag from the multi-day 07-14 failure this check exists to
        # catch. W21's function-scoped AST guard structurally could not see it:
        # this function's own body never names a migrated column — they live in
        # `_newest_trade_ts`.
        lag = (_as_et(s_ts) - _as_et(m_ts)).total_seconds() / 3600
    except Exception:  # noqa: BLE001
        return ("behind", None)
    if lag <= 0:
        # Mirror newer than source (post-restore) — not a staleness problem.
        return ("current", 0.0)
    return ("behind", lag)


def _mirror_content_age_hours() -> float | None:
    """R14 #11: how stale is the off-box mirror's CONTENT (not its file mtime)?

    copy2 back-dates the mirror's mtime, so an mtime check reads green even
    when the mirror is a full run behind (the 07-14 incident). Derive freshness
    from the newest trade timestamp inside the mirror DB, compared to now.
    Returns hours, or None if the mirror is absent/unreadable/empty — the
    caller surfaces that as a MISSING alarm rather than a false all-clear.
    """
    import sqlite3 as _sq
    mirror = Path.home() / "Documents" / "ait_backups" / "ait_state.latest.db"
    if not mirror.exists():
        return None
    try:
        con = _sq.connect(f"file:{mirror}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT MAX(ts) FROM ("
                "  SELECT MAX(entry_time) ts FROM trades"
                "  UNION ALL SELECT MAX(exit_time) FROM trades"
                ")"
            ).fetchone()
        finally:
            con.close()
        if not row or not row[0]:
            return None
        # W21: MAX(entry_time)/MAX(exit_time) are now timezone-aware, so both
        # the parse and the clock they are subtracted from must be. Left naive
        # this raised TypeError into the `except` below, which returns None —
        # and the caller reads None as a MISSING alarm, so the mirror-age check
        # would have alarmed permanently instead of measuring anything.
        from ait.utils.time import now_et, parse_trade_ts
        newest = parse_trade_ts(str(row[0]))
        return max(0.0, (now_et() - newest).total_seconds() / 3600)
    except Exception as e:  # noqa: BLE001
        _log("warning", "mirror_content_age_failed", error=str(e))
        return None


def daily_digest():
    """R6 (user-approved): one-line liveness digest to Telegram (09:35 and
    16:05 ET). The ABSENCE of this message is itself an alarm — the Jul 8->9
    18-hour outage was silent because every alert path died with the machine.
    Pairs with the keeper's external dead-man ping (data/deadman_url.txt)."""
    import sqlite3 as _sq
    try:
        parts = []
        src = DATA_DIR / "ait_state.db"
        if src.exists():
            con = _sq.connect(str(src))
            today = datetime.now().strftime("%Y-%m-%d")
            # R8: NULL-safe. W3 string-contracts-1/-4: the hand-rolled
            # trio is replaced by the ONE shared authority, which also
            # excludes the reconciler's $0 needs-manual-review sentinels.
            from ait.reporting.go_live import not_real_close_sql
            real = not_real_close_sql()
            t = con.execute(
                f"SELECT COUNT(*), COALESCE(SUM(realized_pnl),0) FROM trades "
                f"WHERE status='closed' AND exit_time LIKE ?||'%' {real}",
                (today,)).fetchone()
            e = con.execute(
                "SELECT COUNT(*) FROM trades WHERE entry_time LIKE ?||'%'",
                (today,)).fetchone()
            o = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(unrealized_pnl),0) "
                "FROM open_positions").fetchone()
            # W22: the reminder cadence the operator chose — folded into
            # this existing digest rather than a new schedule or extra
            # Telegram traffic. Declared positions are otherwise silent
            # (by design, they're ignored for management); this is the
            # only place they resurface periodically.
            declared = {r[0][len("manual_position_answer_"):] for r in con.execute(
                "SELECT key FROM bot_state WHERE key LIKE "
                "'manual_position_answer_%' AND value='yes'")}
            prompted = {r[0][len("manual_position_prompt_msgid_"):] for r in con.execute(
                "SELECT key FROM bot_state WHERE key LIKE "
                "'manual_position_prompt_msgid_%'")}
            answered = {r[0][len("manual_position_answer_"):] for r in con.execute(
                "SELECT key FROM bot_state WHERE key LIKE "
                "'manual_position_answer_%'")}
            awaiting_reply = prompted - answered
            con.close()
            parts.append(f"open {o[0]} (unreal ${o[1]:+,.0f})")
            parts.append(f"today {e[0]} entries / {t[0]} closes ${t[1]:+,.0f}")
            if declared:
                parts.append(f"{len(declared)} manual position(s) declared, still open")
            if awaiting_reply:
                parts.append(f"{len(awaiting_reply)} manual position(s) awaiting your reply")
        hb = DATA_DIR / "bot_heartbeat"
        if hb.exists():
            age = (datetime.now() - datetime.fromtimestamp(
                hb.stat().st_mtime)).total_seconds() / 60
            parts.append(f"heartbeat {age:.0f}m")
        # R14 #11: report the MIRROR's CONTENT, not a local snapshot's mtime.
        # The mirror (off-box, the real recovery copy) is what silently went a
        # full run stale on 07-14 while every mtime check read green — and
        # copy2 back-dates its mtime, so mtime can't detect it.
        # R16: the PRIMARY number is now the mirror-vs-SOURCE lag, which is
        # what "stale" actually means. now - newest-trade measured last-TRADE
        # recency, so a quiet market read as an ever-worsening staleness alarm
        # (69h "stale" on 08-07 against a mirror copied 08-06 17:00) while a
        # genuinely stale mirror in a quiet week looked identical. Wall-clock
        # content age is kept as the secondary token.
        m_state, m_lag = _mirror_lag()
        if m_state == "current":
            parts.append("mirror current")
        elif m_state == "behind":
            parts.append(f"mirror BEHIND live by {m_lag:.0f}h"
                         if m_lag is not None else "mirror BEHIND live")
        elif m_state == "unknown":
            parts.append("mirror ?(live db unreadable)")
        else:
            # Mirror unreadable/absent — say so loudly rather than silently
            # falling back to a green-looking local mtime.
            parts.append("mirror MISSING")
        m_age = _mirror_content_age_hours()
        if m_age is not None:
            parts.append(f"last trade {m_age:.0f}h ago")
        # R16 #29: record a sent-marker so the startup catch-up knows whether
        # today's digest actually went out. Only on a CONFIRMED send — a
        # failed/credless send leaves no marker, so the next boot retries.
        if _alert("DIGEST: " + " | ".join(parts)):
            try:
                marker = DATA_DIR / f"digest_sent_{datetime.now().strftime('%Y%m%d')}"
                for _old in DATA_DIR.glob("digest_sent_*"):
                    if _old != marker:
                        _old.unlink()
                marker.write_text(datetime.now().isoformat())
            except Exception as _me:  # noqa: BLE001
                _log("warning", "digest_marker_failed", error=str(_me))
    except Exception as e:  # noqa: BLE001
        _log("warning", "daily_digest_failed", error=str(e))


def _digest_catchup_due(now: datetime, data_dir: Path | None = None) -> bool:
    """R16 #29: should a supervisor boot send the missed 09:35 digest now?

    True on a weekday boot inside 09:35-16:00 ET with no digest_sent marker
    for today. Before 09:35 the cron will fire normally; at/after 16:00 the
    16:05 PM digest is imminent and carries the same content."""
    if now.weekday() >= 5:
        return False
    if not ((9, 35) <= (now.hour, now.minute) < (16, 0)):
        return False
    d = data_dir if data_dir is not None else DATA_DIR
    return not (d / f"digest_sent_{now.strftime('%Y%m%d')}").exists()


def _max_concurrent_car(rows) -> float:
    """D2 (pinned 2026-07-16): max CONCURRENT sum of capital_at_risk over the
    trades' [entry_time, exit_time) windows — the economically meaningful
    drawdown denominator (what was actually at risk when the loss happened).

    W3: the implementation now lives in ait.reporting.go_live so status.py's
    verdict and this scorecard share one definition; this stays as the
    row-shaped adapter (sqlite3.Row / dict) the scorecard and
    tests/test_d2_concurrent_risk.py call.
    """
    from ait.reporting.go_live import max_concurrent_car

    def _car(r):
        try:
            return r["car"] if "car" in r.keys() else 0
        except AttributeError:      # plain dict
            return r.get("car", 0)

    return max_concurrent_car(
        [(r["entry_time"], r["exit_time"], _car(r)) for r in rows])


def weekly_scorecard():
    """R7/W3: go-live-gate scorecard to Telegram (Friday 16:10 ET).

    W3 (money-flow-04 / policy-vs-impl-1/-2/-3/-5): this SCHEDULED surface is
    the one the operator actually receives, and it used to grade the RETIRED
    all-strategy metric while the pinned R19d verdict metric (IRON CONDOR
    closes only) lived solely in status.py — on the live book the two sat on
    opposite sides of the PF>1.3 gate. It now calls the SAME
    ait.reporting.go_live.compute_go_live_verdict() status.py calls and
    renders the SAME format_verdict_lines(), so a divergence between the two
    operator surfaces is no longer expressible. The all-strategy book-level
    line is kept underneath, clearly labelled as book honesty, exactly as
    status.py presents it.

    Every one of gate 1's five criteria is either computed AS PINNED or
    printed UNAVAILABLE with a reason — never a differently-defined number
    wearing the gate's label (the old message rendered a LIFETIME DOLLAR MEAN
    of |fill - mid| under the label "gate: median <=8% of credit").
    """
    import sqlite3 as _sq
    from ait.reporting.go_live import (
        compute_go_live_verdict,
        format_pace_line,
        format_verdict_lines,
        not_real_close_sql,
    )
    try:
        src = DATA_DIR / "ait_state.db"
        if not src.exists():
            return
        # W3 string-contracts-1/-4: one authority for "not a real close" —
        # also excludes the reconciler's $0 needs-manual-review sentinels,
        # which passed the old trio filter and counted as real losing closes.
        real = not_real_close_sql()

        # ---- the gate block (SHARED with status.py) ------------------------
        # policy-vs-impl-5: an exception here used to delete the ENTIRE gate
        # readout silently. It now prints a loud failure line instead.
        try:
            v = compute_go_live_verdict(src)
            gate_lines = format_verdict_lines(v) + [format_pace_line(v)]
        except Exception as _ge:  # noqa: BLE001
            _log("error", "weekly_scorecard_gate_readout_failed",
                 error=f"{type(_ge).__name__}: {_ge}")
            gate_lines = [
                f"!! GATE READOUT FAILED: {type(_ge).__name__}: {_ge}",
                "!! the go-live verdict is MISSING, not passing — do not read "
                "its absence as green.",
            ]

        # ---- TCA / attribution context (NOT gate criteria) ----------------
        # R11 (R10 adoption): the capture layer existed with zero aggregation.
        # W3: the slippage GATE is criterion [4] above, computed as the pinned
        # statistic; what remains here is plain fill/commission bookkeeping
        # with no gate label attached to it.
        con2 = _sq.connect(str(src)); con2.row_factory = _sq.Row
        try:
            xr = con2.execute(
                "SELECT COUNT(*) n, SUM(CASE WHEN live_mid > 0 THEN 1 ELSE 0 END) m, "
                "COALESCE(SUM(commission),0) c FROM executions").fetchone()
            exec_line = (f"executions: {xr['n']} fills "
                         f"({xr['m'] or 0} with a live mid), "
                         f"${xr['c']:.2f} commissions")
            att = con2.execute(
                f"SELECT COALESCE(exit_reason_detailed,'?') r, COUNT(*) n, "
                f"SUM(realized_pnl) p FROM trades WHERE status='closed' {real} "
                f"GROUP BY 1 ORDER BY 3").fetchall()
            attrib = " | ".join(
                f"{(a['r'].split(' ')[0])[:22]}: {a['n']}x ${a['p']:+,.0f}"
                for a in att[:6])
            fr = con2.execute(
                "SELECT SUM(CASE WHEN status='closed' AND COALESCE(exit_reason_detailed,'') "
                "LIKE '%never_filled%' THEN 1 ELSE 0 END) dead, COUNT(*) total "
                "FROM trades WHERE entry_time >= '2026-07-06'").fetchone()
            fill_line = (f"entry fill rate: {fr['total'] - fr['dead']}/{fr['total']}"
                         if fr and fr["total"] else "")
        except Exception as _e:  # noqa: BLE001
            exec_line, attrib, fill_line = f"(exec stats unavailable: {_e})", "", ""
        finally:
            con2.close()

        body = ["GO-LIVE SCORECARD (since 07-06 reset)"]
        body += gate_lines
        body += [exec_line, f"by exit reason: {attrib}"]
        if fill_line:
            body.append(fill_line)
        body.append("see docs/GAP_AUDIT_R7.md for gate definitions")
        _alert(chr(10).join(body))
    except Exception as e:  # noqa: BLE001
        _log("warning", "weekly_scorecard_failed", error=str(e))
        # W3: a failed scorecard used to be a SILENT Friday — the operator saw
        # no message and had no way to tell "nothing to report" from "the gate
        # readout crashed". Say so loudly.
        try:
            _alert(f"GO-LIVE SCORECARD: GATE READOUT FAILED — "
                   f"{type(e).__name__}: {e} (no verdict was computed; "
                   f"run `python status.py` for the gates)")
        except Exception:  # noqa: BLE001
            pass


# R16: keeper.log (22.7 MB), dashboard.log (22 MB) and weblog.log are all
# append-only with NO rotation anywhere — the keeper .bat only ever `>>`s, and
# mtime-based cleanup can never touch a continuously-appended file. The logs
# directory was at 438 MB and growing unbounded on a machine that has to stay
# up for the bot. Same shift-and-cap pattern as _rotate_stdout_log, run daily
# (not monthly) because the keeper's tight-loop bug can add 144k lines in a
# single day.
_ROTATE_TARGETS = {
    "keeper.log": 20 * 1024 * 1024,
    "dashboard.log": 20 * 1024 * 1024,
    "weblog.log": 20 * 1024 * 1024,
    "orchestrator.log": 20 * 1024 * 1024,
}
_ROTATE_BACKUPS = 2


def _rotate_if_oversized(path: Path, max_bytes: int,
                         backups: int = _ROTATE_BACKUPS) -> bool:
    """Shift path -> .1 -> .2 when over the cap. Never raises.

    A rename can legitimately fail on Windows while a child process holds the
    file open (dashboard/weblog stdout); that is logged and retried next run —
    those two are also rotated at their own (re)start, where the handle is
    closed.
    """
    try:
        if not path.exists() or path.stat().st_size <= max_bytes:
            return False
        oldest = path.with_name(f"{path.name}.{backups}")
        if oldest.exists():
            oldest.unlink()
        for i in range(backups - 1, 0, -1):
            src = path.with_name(f"{path.name}.{i}")
            if src.exists():
                src.rename(path.with_name(f"{path.name}.{i + 1}"))
        path.rename(path.with_name(f"{path.name}.1"))
        _log("info", "log_rotated", file=path.name)
        return True
    except Exception as e:  # noqa: BLE001 — rotation must never break the loop
        _log("warning", "log_rotate_failed", file=path.name, error=str(e))
        return False


def rotate_oversized_logs():
    """Daily size cap for the append-only logs nothing else rotates."""
    rotated = 0
    for name, cap in _ROTATE_TARGETS.items():
        if _rotate_if_oversized(LOGS_DIR / name, cap):
            rotated += 1
    _log("info", "log_rotation_sweep", rotated=rotated)


def cleanup_old_logs():
    """Remove logs and reports older than 30 days; cap rotated backups.

    mtime-based deletion can never touch a continuously-appended file (its
    mtime is always "now"), so rotation is handled at bot start
    (BotManager._rotate_stdout_log); here we only prune rotated backups and
    genuinely stale files.
    """
    # R16: size-cap the append-only logs first (keeper/dashboard/weblog/
    # orchestrator) — mtime deletion below can never reach them.
    rotate_oversized_logs()
    cutoff = datetime.now() - timedelta(days=30)
    removed = 0
    for d in [LOGS_DIR, REPORTS_DIR]:
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.name in _ROTATE_TARGETS:
                # Active append-only logs: rotated above by size, exempt from
                # mtime deletion (their mtime is always "now" anyway).
                continue
            try:
                # Rotated stdout backups: keep only the configured count
                # regardless of age (they can be 50 MB each).
                if f.name.startswith("bot_stdout.log."):
                    suffix = f.name.rsplit(".", 1)[-1]
                    if suffix.isdigit() and int(suffix) > BotManager._STDOUT_LOG_BACKUPS:
                        f.unlink()
                        removed += 1
                        continue
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if mtime < cutoff:
                    f.unlink()
                    removed += 1
            except Exception:
                pass
    _log("info", "cleanup_complete", removed=removed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_backtest_output(output: str) -> dict:
    """Extract key metrics from backtest stdout."""
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if "Total Return:" in line:
            metrics["total_return"] = line.split(":")[-1].strip()
        elif "Sharpe Ratio:" in line:
            metrics["sharpe"] = line.split(":")[-1].strip()
        elif "Win Rate:" in line:
            metrics["win_rate"] = line.split(":")[-1].strip()
        elif "Max Drawdown:" in line:
            metrics["max_drawdown"] = line.split(":")[-1].strip()
        elif "Total Trades:" in line:
            metrics["total_trades"] = line.split(":")[-1].strip()
        elif "Profit Factor:" in line:
            metrics["profit_factor"] = line.split(":")[-1].strip()
        elif "Start:" in line and "End:" in line:
            metrics["equity_summary"] = line.strip()
    return metrics


def _append_health_metric(metrics: dict):
    """Append backtest metrics to health tracking file."""
    health_file = DATA_DIR / "strategy_health.jsonl"
    DATA_DIR.mkdir(exist_ok=True)
    with open(health_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def _daily_report_catchup_due(now: datetime, reports_dir: Path | None = None) -> bool:
    """R16 #59: should a supervisor boot run the missed 16:30 daily report?

    True on a weekday boot at/after 16:35 with no daily_YYYYMMDD.json for
    today. The 16:35 threshold leaves the cron slot 5 min of margin so a boot
    at 16:3x cannot double-run alongside the scheduled job."""
    if now.weekday() >= 5:
        return False
    if (now.hour, now.minute) < (16, 35):
        return False
    d = reports_dir if reports_dir is not None else REPORTS_DIR
    return not (d / f"daily_{now.strftime('%Y%m%d')}.json").exists()


def _backtest_catchup_due(now: datetime, reports_dir: Path | None = None) -> bool:
    """R16 #59: was the weekly Sunday 20:00 backtest slot missed?

    True when the newest backtest_*.json predates the most recent Sunday
    20:00 slot (or none exists at all — the observed state: exactly one run
    since March). `now` is naive local time, matching the artifact mtimes it
    is compared against.

    W20: this said "naive local time (box runs ET)" — false since the CEST
    move, and harmless only because BOTH sides of the comparison are naive
    local. The Sunday-20:00 slot is therefore the BOX's 20:00, not ET's;
    immaterial for a weekly catch-up, but do not read an ET guarantee into
    this function."""
    d = reports_dir if reports_dir is not None else REPORTS_DIR
    # Most recent Sunday-20:00 slot at or before `now` (Mon=0..Sun=6).
    due = (now - timedelta(days=(now.weekday() + 1) % 7)).replace(
        hour=20, minute=0, second=0, microsecond=0)
    if due > now:  # it IS Sunday, before 20:00 — last due slot was last week's
        due -= timedelta(days=7)
    snaps = list(d.glob("backtest_*.json"))
    if not snaps:
        return True
    newest = max(datetime.fromtimestamp(f.stat().st_mtime) for f in snaps)
    return newest < due


def _backtest_catchup_run_at(now: datetime) -> datetime:
    """R16 #59: when to run the backtest catch-up.

    Deferred 10 min after start (never synchronously in main() — a stuck
    subprocess there would block the whole supervisor, finding #69), and
    never inside market hours: the 10-min full-core backtest has no business
    running beside the live bot. If the deferred slot would land in (or
    within 15 min of) the 9:30-16:00 session, push it to 16:40 ET — after
    the 16:30 report slot."""
    run_at = now + timedelta(minutes=10)
    if run_at.weekday() < 5 and (9, 15) <= (run_at.hour, run_at.minute) < (16, 30):
        run_at = run_at.replace(hour=16, minute=40, second=0, microsecond=0)
    return run_at


def _retrain_catchup_run_at(now: datetime) -> datetime:
    """W23: mirrors _backtest_catchup_run_at's shape for the same reason —
    a small, independently-testable unit rather than an inline expression at
    the call site. `now` MUST be aware ET here: this value becomes the
    scheduler's run_date directly, and W23 probed that a naive datetime
    passed through the "date" alias is mislabeled with the scheduler's own
    offset rather than converted (confirmed live: retrain_catchup registered
    "2026-09-14 21:31:45 EDT", actually ran 2026-09-15 03:31:45 -- 6h late,
    to the second). No market-hours dodge needed here (unlike the backtest
    catch-up) -- the model-reload restart this unblocks already defers
    until market close on its own."""
    return now + timedelta(minutes=2)


def _check_telegram_dead():
    """R16 #15: the bot writes data/TELEGRAM_DEAD after 5 consecutive dropped
    sends — and until now NOTHING read it (the writer's own comment claimed
    the digest surfaced it; it never did), so a dead alert channel stayed
    invisible until the operator noticed prolonged silence. Every health
    cycle: log critical while the marker exists, probe the channel at most
    once per hour, and clear the marker the moment a send goes through."""
    marker = DATA_DIR / "TELEGRAM_DEAD"
    try:
        if not marker.exists():
            return
        detail = ""
        try:
            detail = marker.read_text()[:120]
        except OSError:
            pass
        _log("critical", "telegram_dead_marker_present", detail=detail)
        if not _alert_gate("telegram_dead_probe", 3600):
            return
        if _alert("Telegram channel RECOVERED — the bot had marked it dead "
                  f"({detail or 'no detail'}). Alerts dropped during the "
                  "outage were NOT re-delivered; check logs for "
                  "notification_dropped_after_retries."):
            marker.unlink()
            _log("info", "telegram_dead_marker_cleared")
    except Exception as e:  # noqa: BLE001 — health loop must never die on this
        _log("warning", "telegram_dead_check_failed", error=str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _register_recurring_jobs(scheduler, bot, web) -> None:
    """W43 (R26 `w23-et-cron-tests-build-their-own-crontrigger`): the ten
    unconditional cron/interval jobs, extracted from `main()` so a test can
    get back a REAL scheduler with real jobs registered on it without
    fighting `main()`'s side effects (the singleton lock, `bot.start()` /
    `web.start()`, signal handlers, the startup Telegram alert).

    Deliberately does NOT include the four catch-up blocks
    (backup/retrain/digest/report) that follow these in `main()` — each of
    those can synchronously fire a real side effect (`backup_state_db()`,
    `daily_digest()`, `daily_report()`) when its own due-check is true, which
    it can be by DEFAULT in an empty test sandbox (e.g. the backup catch-up's
    `not _snaps` is true when no backup exists yet). Mixing that into a
    function a test calls directly would risk a real Telegram send from a
    unit test. The one catch-up job this file's own tests need to reach —
    the backtest date job — is `_maybe_schedule_backtest_catchup` below,
    which registers a future job and does not run anything synchronously.
    """
    # Health check: every 2 minutes (bot + web services)
    def combined_health_check():
        bot.health_check()
        web.health_check()
        # R16 #15: surface a dead Telegram channel (bot-written marker that
        # nothing read); probes hourly, clears itself on a successful send.
        _check_telegram_dead()

    scheduler.add_job(combined_health_check, "interval", minutes=2, id="health_check")

    # Daily model retrain: 7:30 AM ET on trading days (Mon-Fri)
    scheduler.add_job(retrain_models,
                      CronTrigger(day_of_week="mon-fri", hour=7, minute=30, timezone=ET),
                      id="daily_retrain")

    # Daily performance report: 4:30 PM ET
    scheduler.add_job(backup_state_db, CronTrigger(day_of_week="mon-fri", hour=17, minute=0, timezone=ET),
                      id="db_backup", name="State DB backup")
    scheduler.add_job(weekly_scorecard,
                      CronTrigger(day_of_week="fri", hour=16, minute=10, timezone=ET),
                      id="weekly_scorecard")
    scheduler.add_job(daily_digest,
                      CronTrigger(day_of_week="mon-fri", hour=9, minute=35, timezone=ET),
                      id="digest_am")
    scheduler.add_job(daily_digest,
                      CronTrigger(day_of_week="mon-fri", hour=16, minute=5, timezone=ET),
                      id="digest_pm")
    scheduler.add_job(daily_report,
                      CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=ET),
                      id="daily_report")
    # Shared per-machine logs: after the post-market run and the 16:30 report.
    scheduler.add_job(publish_shared_logs,
                      CronTrigger(day_of_week="mon-fri", hour=16, minute=45, timezone=ET),
                      id="shared_logs_publish")
    # W70 follow-up: refresh origin/shared-logs 5 minutes after the publish (fetch only;
    # not gated on AIT_LOG_OWNER). See fetch_shared_logs().
    scheduler.add_job(fetch_shared_logs,
                      CronTrigger(day_of_week="mon-fri", hour=16, minute=50, timezone=ET),
                      id="shared_logs_fetch")
    # W76: supervisor-run 0-6 DTE friction collection. Registered ONLY when
    # data_collection.short_dte_enabled (default off); a process runs on the settings it
    # started with, so toggling it takes a supervisor restart. See collect_short_dte_friction().
    _dc = _data_collection_cfg()
    if _dc.short_dte_enabled:
        _dc_h, _dc_m = (int(x) for x in _dc.short_dte_time_et.split(":"))
        scheduler.add_job(collect_short_dte_friction,
                          CronTrigger(day_of_week="mon-fri", hour=_dc_h, minute=_dc_m, timezone=ET),
                          id="short_dte_collection", max_instances=1)
        _log("info", "short_dte_collection_scheduled", time_et=_dc.short_dte_time_et,
             symbols=",".join(_dc.short_dte_symbols), client_id=_dc.short_dte_client_id)

    # Weekly deep backtest: Sunday 8 PM ET
    scheduler.add_job(run_backtest,
                      CronTrigger(day_of_week="sun", hour=20, minute=0, timezone=ET),
                      id="weekly_backtest")

    # Monthly log cleanup: 1st of month at midnight
    scheduler.add_job(cleanup_old_logs,
                      CronTrigger(day=1, hour=0, minute=0, timezone=ET),
                      id="monthly_cleanup")
    # R16: DAILY size cap for the append-only logs (keeper/dashboard/weblog/
    # orchestrator). Monthly was far too slow for a keeper.log that can gain
    # 144k lines in one day when the .bat pacing collapses.
    scheduler.add_job(rotate_oversized_logs,
                      CronTrigger(hour=0, minute=5, timezone=ET),
                      id="daily_log_rotation")


def _maybe_schedule_backtest_catchup(scheduler, now_bt: datetime | None = None) -> None:
    """R16 #59: weekly-backtest catch-up — register a ONE-TIME date job when
    the box missed its Sunday 20:00 ET slot. Registration only: this never
    calls `run_backtest()` itself, so it is safe to call directly in a test.

    W43: extracted so the date job's `run_date` (which W23 hardened to come
    from the AWARE `now_et()` clock, deliberately different from the naive
    `now_bt` the due-check uses — see the comment inline below, preserved
    verbatim) can be asserted on directly instead of via
    `inspect.getsource(main)`. `now_bt` is injectable for the same reason
    every other catch-up helper in this file takes `now: datetime` explicitly
    (`_backtest_catchup_due`, `_digest_catchup_due`, ...) rather than reading
    the clock itself; defaults to `datetime.now()` so `main()`'s own call
    (below) is unchanged.
    """
    if now_bt is None:
        # W20: naive local, matching the file mtimes it is compared against.
        # The old comment said "(=ET box)" — false since the CEST move. The
        # CODE is still right: both sides of the comparison are naive local.
        now_bt = datetime.now()
    try:
        if _backtest_catchup_due(now_bt):
            # W23: a SECOND, deliberately DIFFERENT clock for the run-at
            # computation — _backtest_catchup_run_at's market-hours check
            # needs real ET hours (a naive host-local hour compared against
            # 9:15/16:30 ET thresholds is checking the wrong clock), and its
            # return value becomes the scheduler's run_date directly, which
            # W23 probed must be aware or apscheduler mislabels it. Do not
            # "simplify" this back to reusing now_bt — the due-check above
            # and the run-at computation below need genuinely different
            # clocks, on purpose.
            _bt_at = _backtest_catchup_run_at(now_et())
            scheduler.add_job(run_backtest, "date", run_date=_bt_at,
                              id="backtest_catchup",
                              name="Weekly backtest catch-up")
            _log("info", "startup_catchup_backtest_scheduled",
                 run_at=_bt_at.strftime("%Y-%m-%d %H:%M"))
    except Exception as _e:  # noqa: BLE001
        _log("warning", "startup_backtest_check_failed", error=str(_e))


def main():
    # R16 #66: atomic single-instance gate BEFORE any side effects — the
    # keeper's 90s loop and the 07:30 scheduled task can both pass their
    # check-then-launch guards in the same window; the OS lock cannot race.
    _enable_file_sink()   # W34: this process IS the supervisor
    # W35: run on the configuration we started with (see freeze_settings) — a
    # deploy that adds a config.yaml key must not break this process's pages.
    try:
        from ait.config.settings import freeze_settings, load_settings
        freeze_settings(load_settings())
    except Exception as e:  # noqa: BLE001 — never block the supervisor on config
        _log("warning", "settings_freeze_failed", error=str(e))
    if not _acquire_singleton_lock():
        if _singleton_lock_status == "unavailable":
            _log("critical", "orchestrator_lock_unavailable", action="exiting")
        else:
            _log("info", "orchestrator_already_running", action="exiting")
        return
    _log("info", "orchestrator_starting", pid=os.getpid())
    _alert("supervisor started — bot + web services coming up.")

    bot = BotManager()
    web = WebServiceManager()
    # Deep-audit OPS-C2: default misfire_grace_time is 1 SECOND — on this
    # frequently-restarted box the 7:30 retrain / 16:30 report / Sunday
    # backtest were silently dropped whenever the process was down or busy
    # across the cron second. Give jobs an hour of grace and coalesce.
    scheduler = BlockingScheduler(
        timezone="US/Eastern",
        job_defaults={"misfire_grace_time": 3600, "coalesce": True},
    )
    shutdown = Event()

    def graceful_shutdown(signum, frame):
        _log("info", "shutdown_signal", signal=signum)
        shutdown.set()
        bot.stop()
        web.stop()
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # --- Start the trading bot + web services ---
    bot.start()
    web.start()

    # --- Schedule tasks ---

    # W43: the ten unconditional recurring jobs, extracted so a test can
    # register real jobs on a real scheduler without main()'s other side
    # effects (see _register_recurring_jobs's own docstring).
    _register_recurring_jobs(scheduler, bot, web)

    # R6: catch-up backup at supervisor start when the newest snapshot is
    # >24h old — a machine-down day silently skips the 17:00 slot (observed
    # Jul 8: the 18h outage covered it and nothing noticed).
    try:
        _snaps = sorted((DATA_DIR / "backups").glob("ait_state.*.db"))
        if not _snaps or (datetime.now() - datetime.fromtimestamp(
                _snaps[-1].stat().st_mtime)) > timedelta(hours=24):
            _log("info", "startup_catchup_backup")
            backup_state_db()
    except Exception as _e:  # noqa: BLE001
        _log("warning", "startup_backup_check_failed", error=str(_e))
    # R15 follow-up: the 07:30 retrain cron only fires if this PROCESS is
    # alive at 07:30 — and the box routinely comes up ~08:45-09:00 (no
    # auto-logon, U2), so the retrain silently missed EVERY late start
    # (last ran 07-17; models went stale for days, and the meta_label.pkl
    # test-clobber persisted because nothing rewrote it). Same catch-up
    # pattern as the backup above: a weekday start after 07:35 with no
    # retrain log for today runs one. Safe intraday: the .retrained
    # model-reload restart already defers until market close.
    # R16: DEFERRED, not synchronous. retrain_models() is a multi-minute
    # blocking child (bounded only by its own idle killer) and it ran inside
    # main() BEFORE scheduler.start() — so on every late boot the supervisor
    # had no health checks, no bot-down detection and no dead-man ping for the
    # whole retrain, exactly when the box has just come up. Same one-shot
    # date-job pattern as the weekly-backtest catch-up (#69).
    try:
        from ait.utils.time import now_et as _cnet
        _now_c = _cnet()
        if (_now_c.weekday() < 5 and (_now_c.hour, _now_c.minute) >= (7, 35)
                and not list(LOGS_DIR.glob(
                    f"retrain_{_now_c.strftime('%Y%m%d')}_*.log"))):
            _rt_at = _retrain_catchup_run_at(_now_c)  # _now_c is already aware ET
            scheduler.add_job(retrain_models, "date", run_date=_rt_at,
                              id="retrain_catchup",
                              name="Daily retrain catch-up")
            _log("info", "startup_catchup_retrain_scheduled",
                 run_at=_rt_at.strftime("%Y-%m-%d %H:%M"))
    except Exception as _e:  # noqa: BLE001
        _log("warning", "startup_retrain_check_failed", error=str(_e))
    # R16 #29: digest catch-up — the 09:35 digest ('absence IS the alarm')
    # was silently skipped on exactly the outage mornings it was built to
    # expose (boots at 13:43 on 08-05, 11:44 on 08-06: no digest either day).
    # Same catch-up pattern as backup/retrain above: trading-day boot inside
    # 09:35-16:00 with no digest_sent marker for today -> send one now.
    try:
        from ait.utils.time import now_et as _dnet
        if _digest_catchup_due(_dnet()):
            _log("info", "startup_catchup_digest")
            daily_digest()
    except Exception as _e:  # noqa: BLE001
        _log("warning", "startup_digest_check_failed", error=str(_e))
    # R16 #59: same missed-slot catch-up for the 16:30 report (a boot after
    # 16:30 skips the cron entirely — in-memory jobstore, grace only applies
    # while the process is alive).
    try:
        from ait.utils.time import now_et as _rpnet
        if _daily_report_catchup_due(_rpnet()):
            _log("info", "startup_catchup_daily_report")
            daily_report()
    except Exception as _e:  # noqa: BLE001
        _log("warning", "startup_daily_report_check_failed", error=str(_e))

    # R16 #59: weekly-backtest catch-up — fired ONCE since March (Apr 26);
    # the box is routinely off Sunday 20:00. Unlike the report, this one is
    # DEFERRED 10 min (never synchronously in main(), #69) and never runs
    # during market hours: pushed past 16:30 ET if it would land in-session.
    # W43: registration extracted to _maybe_schedule_backtest_catchup.
    _maybe_schedule_backtest_catchup(scheduler)

    _log("info", "scheduler_ready", jobs=len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        _log("info", "job_registered", id=job.id, trigger=str(job.trigger))

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        _log("info", "orchestrator_shutting_down")
        bot.stop()
        web.stop()


if __name__ == "__main__":
    main()

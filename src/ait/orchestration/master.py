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
    with open(LOGS_DIR / "orchestrator.log", "a") as f:
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
# lock is atomic and self-releasing: msvcrt region locks die with the process,
# so there is no stale-pidfile problem to detect.

_singleton_lock_handle = None  # module-held: lock lives exactly as long as this process


def _acquire_singleton_lock(lock_path: Path | None = None) -> bool:
    """Atomically claim the single-orchestrator lock. False = already held."""
    global _singleton_lock_handle
    import msvcrt
    path = lock_path or (DATA_DIR / "orchestrator.lock")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "a+")
        fh.seek(0)  # locking() operates from the fd's current position
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            fh.close()
            return False
        try:  # informational only — who holds it (the LOCK is the guard)
            fh.truncate(0)
            fh.write(f"{os.getpid()}\n")
            fh.flush()
        except OSError:
            pass
        _singleton_lock_handle = fh
        return True
    except Exception as e:  # noqa: BLE001 — lock plumbing must never keep the bot down
        _log("warning", "singleton_lock_error", error=str(e))
        return True


def _release_singleton_lock() -> None:
    """Explicit release (tests / graceful paths); process death also releases."""
    global _singleton_lock_handle
    if _singleton_lock_handle is None:
        return
    import msvcrt
    try:
        _singleton_lock_handle.seek(0)
        msvcrt.locking(_singleton_lock_handle.fileno(), msvcrt.LK_UNLCK, 1)
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

def _scan_foreign_bot_pids() -> list[int]:
    """R16 #10: PIDs of any pre-existing 'ait.main' trading-bot process.

    An ait.main child orphaned by a master-only death survives (no Job
    object), the keeper's guard used to match only 'run_orchestrator', and a
    relaunched master would spawn a SECOND bot trading the same account/DB
    (the clientId fallback defeats IBKR's duplicate-id rejection). Same
    Get-CimInstance approach as the keeper (wmic is deprecated, A11)."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'ait\\.main' } | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=30)
        return [int(tok) for tok in result.stdout.split() if tok.strip().isdigit()]
    except Exception as e:  # noqa: BLE001 — a failed scan must not block a start
        _log("warning", "foreign_bot_scan_failed", error=str(e))
        return []


def _clear_dashboard_port(port: int = 8501) -> bool:
    """R16 #57: free the dashboard port before (re)launching Streamlit.

    An orphaned Streamlit from an ungracefully-dead supervisor holds 8501, so
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


def _in_post_market_window(now: datetime | None = None,
                           grace_minutes: int = 20) -> bool:
    """R16: is NOW inside the bot's own POST_MARKET phase (+grace)?

    The bot's scheduler treats close -> close+15min as POST_MARKET and runs the
    whole EOD pipeline on entry, with no already-ran-today guard. A supervisor
    restart landing in that window therefore re-runs it. Mirrors
    bot/scheduler.py:52-63 (honouring early-close days via get_market_close)
    and adds 5 minutes of slack so a restart cannot land on the boundary.

    If the market calendar is unavailable it falls back to the regular-session
    clock (16:00 ET, the box's local time) rather than a blanket True — fail
    closed INSIDE the window, but never wedge the model-reload restart forever
    on a calendar outage.
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
        n2 = now or datetime.now()
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
        if not ensure_gateway(port=gw_port):
            _log("error", "bot_start_aborted", reason="gateway_not_available")
            # Alert at most once per 30 min so a long Gateway outage pings
            # once rather than every 2-min health check.
            now = datetime.now()
            if (self._last_gateway_alert is None
                    or (now - self._last_gateway_alert) > timedelta(minutes=30)):
                self._last_gateway_alert = now
                _alert(f"cannot start bot — IB Gateway not available on port {gw_port}. "
                       f"Is Gateway/TWS logged in?")
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
            # R6 (user-approved): HANG detection — process-exists is not
            # process-alive. The bot touches data/bot_heartbeat every ~30s
            # while its trading loop runs; stale >15 min DURING MARKET HOURS
            # means a hung event loop (stops/TPs silently off on the whole
            # book) — the exact failure class all three supervision layers
            # were blind to (R5-D1). Restart + alert.
            try:
                _hb = DATA_DIR / "bot_heartbeat"
                from ait.utils.time import is_market_open as _imo_hb
                if _imo_hb() and _hb.exists():
                    _age_min = (datetime.now() - datetime.fromtimestamp(
                        _hb.stat().st_mtime)).total_seconds() / 60
                    if _age_min > 15:
                        _log("error", "bot_hung_heartbeat_stale",
                             age_min=round(_age_min))
                        _alert(f"BOT HUNG: trading-loop heartbeat {_age_min:.0f} "
                               f"min stale during market hours — restarting the bot. "
                               f"Positions were unmanaged for that window.")
                        self.stop()
                        self.start()
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
                except Exception:
                    market_open = False
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
            _log("info", "bot_restart_deferred_gateway_down", port=gw_port)
            now = datetime.now()
            if (self._last_gateway_alert is None
                    or (now - self._last_gateway_alert) > timedelta(minutes=30)):
                self._last_gateway_alert = now
                _alert(f"IB Gateway unreachable on port {gw_port} (likely daily "
                       f"maintenance). Bot start deferred; will resume "
                       f"automatically when Gateway returns. No budget spent.")
            return

        # Gateway is up — a fresh start attempt may have cleared whatever was
        # wrong; reset the gateway-alert latch so the next outage re-alerts.
        self._last_gateway_alert = None

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
        """Lightweight live status dashboard (http://localhost:8503).

        Reliable real-time view of bot health, positions, today's activity,
        and real-exit P&L — reads the DB/logs directly, no IBKR clientId
        conflict. Separate from the heavy Streamlit app on 8501.
        """
        if self._status_proc and self._status_proc.poll() is None:
            return
        _log("info", "status_dashboard_starting", port=8503)
        log_path = LOGS_DIR / "status_server.log"
        self._status_log = open(log_path, "a")
        self._status_proc = subprocess.Popen(
            [sys.executable, str(ROOT / "status_server.py")],
            cwd=str(ROOT),
            stdout=self._status_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        _log("info", "status_dashboard_started", pid=self._status_proc.pid,
             url="http://localhost:8503")

    def _start_dashboard(self):
        if self._dashboard_proc and self._dashboard_proc.poll() is None:
            return
        # R16 #57: an orphaned Streamlit (from an ungracefully-dead previous
        # supervisor) holding 8501 made every relaunch die in ~1s and the
        # health loop respawned a doomed child every 2 min, forever, silently.
        # Clear the port first; if a NON-streamlit process holds it, skip the
        # spawn entirely (alert, gated) instead of crash-looping against it.
        if not _clear_dashboard_port(8501):
            _log("error", "dashboard_start_skipped_port_held", port=8501)
            if _alert_gate("dashboard_port_held", 3600):
                _alert("dashboard NOT started: port 8501 is held by a process "
                       "the supervisor won't kill — see orchestrator.log "
                       "(dashboard_port_held_foreign) for the holder.")
            return
        _log("info", "dashboard_starting", port=8501)
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
             "--server.port=8501", "--server.headless=true",
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
        _log("info", "weblog_starting", port=8502)
        log_path = LOGS_DIR / "weblog.log"
        _rotate_if_oversized(log_path, _ROTATE_TARGETS["weblog.log"])  # R16
        self._weblog_log = open(log_path, "a")
        self._weblog_proc = subprocess.Popen(
            [sys.executable, str(ROOT / "web_logs.py")],
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
                "--capital", "50000",
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
range_pred = RangePredictor(threshold_pct=0.05, horizon_days=30)
vol_mag_pred = VolMagnitudePredictor(threshold_pct=0.07, horizon_days=30)
market_data = MarketDataService(None, polygon_api_key=settings.api_keys.polygon_api_key)
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
        lag = (datetime.fromisoformat(s_ts)
               - datetime.fromisoformat(m_ts)).total_seconds() / 3600
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
        newest = datetime.fromisoformat(str(row[0]))
        return max(0.0, (datetime.now() - newest).total_seconds() / 3600)
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
            real = ("AND COALESCE(exit_reason_detailed,'') NOT LIKE '%migrated%' "
                    "AND COALESCE(exit_reason_detailed,'') NOT LIKE '%pending%' "
                    "AND COALESCE(exit_reason_detailed,'') NOT LIKE '%never_filled%'")  # R8: NULL-safe like sibling sites
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
            con.close()
            parts.append(f"open {o[0]} (unreal ${o[1]:+,.0f})")
            parts.append(f"today {e[0]} entries / {t[0]} closes ${t[1]:+,.0f}")
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

    Event-sweep: +car at entry, -car at exit (open trades never subtract).
    Rows without a positive car contribute nothing — the D1 backfill filled
    all derivable ones, and the referee flags coverage holes.
    """
    ev = []
    for r in rows:
        car = r["car"] if "car" in r.keys() else 0
        if not car or car <= 0:
            continue
        ev.append((r["entry_time"] or "", +car))
        ev.append((r["exit_time"] or "9999", -car))
    peak = cur = 0.0
    for _, d in sorted(ev):
        cur += d
        peak = max(peak, cur)
    return peak


def weekly_scorecard():
    """R7: go-live-gate scorecard to Telegram (Friday 16:10 ET). The five
    gate criteria, computed since the 2026-07-06 reset, on real closes only.
    DD is reported vs max concurrent DEPLOYED RISK, not paper NLV (a 100%
    loss of deployed risk used to read as ~2% 'drawdown')."""
    import sqlite3 as _sq
    try:
        src = DATA_DIR / "ait_state.db"
        if not src.exists():
            return
        con = _sq.connect(str(src)); con.row_factory = _sq.Row
        real = ("AND COALESCE(exit_reason_detailed,'') NOT LIKE '%migrated%' "
                "AND COALESCE(exit_reason_detailed,'') NOT LIKE '%pending%' "
                "AND COALESCE(exit_reason_detailed,'') NOT LIKE '%never_filled%'")
        rows = con.execute(
            f"SELECT realized_pnl, entry_time, exit_time, "
            f"COALESCE(capital_at_risk, 0) car, COALESCE(commission,0) comm "
            f"FROM trades WHERE status='closed' {real} "
            f"ORDER BY COALESCE(exit_time, entry_time)").fetchall()
        n = len(rows)
        wins = sum(1 for r in rows if r["realized_pnl"] > 0)
        gp = sum(r["realized_pnl"] for r in rows if r["realized_pnl"] > 0)
        gl = abs(sum(r["realized_pnl"] for r in rows if r["realized_pnl"] < 0))
        pf = (gp / gl) if gl > 0 else float("inf")
        tot = sum(r["realized_pnl"] for r in rows)
        comm = sum(r["comm"] for r in rows)
        # drawdown vs deployed risk: equity curve of realized P&L; base =
        # max concurrent capital_at_risk (approx: max single-day sum of open
        # trades' car; fallback to open book's current car sum, then $1k).
        peak = dd = cum = 0.0
        for r in rows:
            cum += r["realized_pnl"]
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        # D2 (DECIDED 2026-07-16, PLAN.md): the DD base is the MAX CONCURRENT
        # deployed risk over the whole window — not the current open book.
        # The old base (open-book car floored at $1,000) collapsed to the
        # floor after any flatten, inflating DD% (13.8% FAIL vs the true
        # 7.4% PASS across the 8% gate — the exact ambiguity D2 settles).
        # Criteria pinned BEFORE the sample grows; do not revisit with
        # results visible.
        open_rows = con.execute(
            "SELECT t.entry_time, t.exit_time, COALESCE(t.capital_at_risk,0) car "
            "FROM trades t JOIN open_positions o ON o.trade_id=t.trade_id").fetchall()
        base = max(_max_concurrent_car(list(rows) + list(open_rows)), 1000.0)
        con.close()
        # R11 (R10 adoption): TCA + attribution read-outs — the capture layer
        # existed with zero aggregation; the go-live "stable slippage" gate
        # now has a NUMBER: median entry slippage <= 8% of credit over the
        # trailing 20 fills, no worsening trend (PLAN.md gate 1).
        con2 = _sq.connect(str(src)); con2.row_factory = _sq.Row
        try:
            xr = con2.execute(
                "SELECT COUNT(*) n, SUM(CASE WHEN live_mid > 0 THEN 1 ELSE 0 END) m, "
                "COALESCE(SUM(commission),0) c FROM executions").fetchone()
            exec_line = f"executions: {xr['n']} fills, ${xr['c']:.2f} commissions"
            slip = con2.execute(
                "SELECT AVG(ABS(price - live_mid)) FROM executions "
                "WHERE live_mid > 0").fetchone()[0]
            if slip is not None:
                exec_line += f", avg |fill-mid| ${slip:.2f} (gate: median <=8% of credit)"
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

        pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        _alert(
            f"GO-LIVE SCORECARD (since 07-06 reset)\n"
            f"closes: {n}/50 | PF: {pf_s} (gate >1.3) | win rate: "
            f"{(wins / n * 100) if n else 0:.0f}%\n"
            f"realized: ${tot:+,.0f} (commissions recorded: ${comm:,.0f})\n"
            f"max DD: ${dd:,.0f} = {dd / base * 100:.1f}% of deployed risk "
            f"~${base:,.0f} (gate <8%)\n"
            f"{exec_line}\n"
            f"by exit reason: {attrib}\n"
            f"{fill_line}\n"
            f"pace: {'on track' if n >= 1 else 'no closes yet'} — see "
            f"docs/GAP_AUDIT_R7.md for gate definitions"
        )
    except Exception as e:  # noqa: BLE001
        _log("warning", "weekly_scorecard_failed", error=str(e))


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
    since March). `now` is naive local time (box runs ET), matching the
    artifact mtimes it is compared against."""
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

def main():
    # R16 #66: atomic single-instance gate BEFORE any side effects — the
    # keeper's 90s loop and the 07:30 scheduled task can both pass their
    # check-then-launch guards in the same window; the OS lock cannot race.
    if not _acquire_singleton_lock():
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
                      CronTrigger(day_of_week="mon-fri", hour=7, minute=30),
                      id="daily_retrain")

    # Daily performance report: 4:30 PM ET
    scheduler.add_job(backup_state_db, CronTrigger(day_of_week="mon-fri", hour=17, minute=0),
                      id="db_backup", name="State DB backup")
    scheduler.add_job(weekly_scorecard,
                      CronTrigger(day_of_week="fri", hour=16, minute=10),
                      id="weekly_scorecard")
    scheduler.add_job(daily_digest,
                      CronTrigger(day_of_week="mon-fri", hour=9, minute=35),
                      id="digest_am")
    scheduler.add_job(daily_digest,
                      CronTrigger(day_of_week="mon-fri", hour=16, minute=5),
                      id="digest_pm")
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
            _rt_at = datetime.now() + timedelta(minutes=2)
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
    scheduler.add_job(daily_report,
                      CronTrigger(day_of_week="mon-fri", hour=16, minute=30),
                      id="daily_report")
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

    # Weekly deep backtest: Sunday 8 PM ET
    scheduler.add_job(run_backtest,
                      CronTrigger(day_of_week="sun", hour=20, minute=0),
                      id="weekly_backtest")
    # R16 #59: weekly-backtest catch-up — fired ONCE since March (Apr 26);
    # the box is routinely off Sunday 20:00. Unlike the report, this one is
    # DEFERRED 10 min (never synchronously in main(), #69) and never runs
    # during market hours: pushed past 16:30 ET if it would land in-session.
    try:
        _now_bt = datetime.now()  # naive local (=ET box), matches file mtimes
        if _backtest_catchup_due(_now_bt):
            _bt_at = _backtest_catchup_run_at(_now_bt)
            scheduler.add_job(run_backtest, "date", run_date=_bt_at,
                              id="backtest_catchup",
                              name="Weekly backtest catch-up")
            _log("info", "startup_catchup_backtest_scheduled",
                 run_at=_bt_at.strftime("%Y-%m-%d %H:%M"))
    except Exception as _e:  # noqa: BLE001
        _log("warning", "startup_backtest_check_failed", error=str(_e))

    # Monthly log cleanup: 1st of month at midnight
    scheduler.add_job(cleanup_old_logs,
                      CronTrigger(day=1, hour=0, minute=0),
                      id="monthly_cleanup")
    # R16: DAILY size cap for the append-only logs (keeper/dashboard/weblog/
    # orchestrator). Monthly was far too slow for a keeper.log that can gain
    # 144k lines in one day when the .bat pacing collapses.
    scheduler.add_job(rotate_oversized_logs,
                      CronTrigger(hour=0, minute=5),
                      id="daily_log_rotation")

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

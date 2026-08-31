"""W6 — operator surfaces must never report GREEN or ZERO for an unwired channel.

Findings under test (reports/relationship_defect_hunt_20260825.md and
reports/blindspot_composition_hunt_20260825.md):

  string-contracts-5 / db-contracts-6  System Health tab read five bot_state
                                       keys with no writer; the error panel's
                                       fallback was a green "No errors logged"
  log-contracts-1 / -4                 today-counters read a fixed 200-line
                                       window of one file and matched a LOCAL
                                       date against UTC stamps
  log-contracts-6                      an interactive-console start writes ANSI
                                       ConsoleRenderer lines no consumer parses
  log-contracts-3                      "native crashes: 0" while faulthandler
                                       writes to logs/fatal.log
  dead-surface-3                       ML Direction Accuracy can never populate
  dead-surface-4                       "Avg Capture Efficiency -1653%" is
                                       dollars divided by a percentage
  bot-day-02                           the dead-man ping attests the SUPERVISOR

Every test below EXECUTES the real function against real temp artefacts: a
StateManager-created sqlite DB, log files produced by the real structlog
pipeline (including a genuine ANSI ConsoleRenderer render), and a genuine
faulthandler crash dump produced by crashing a real child process.  Each
behaviour is asserted twice: positively (the surface reports the TRUE value)
and negatively (an absent channel reports "not wired" — never green, never a
bare zero).
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from ait.bot.state import StateManager
from ait.monitoring.ops_health import (
    CHANNEL_MAX_AGE_S,
    HEARTBEAT_MAX_AGE_S,
    RTH_WARMUP_S,
    HealthStatePublisher,
    IncrementalEventCounter,
    bot_liveness,
    count_events_today,
    native_crash_report,
    parse_event_line,
    read_channel_state,
    strip_ansi,
    tail_lines,
)
from ait.monitoring.watchdog import ComponentStatus, Watchdog

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Import the two root-level surfaces under test (not importable as packages)
# ---------------------------------------------------------------------------

def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


web_logs = _load("w6_web_logs", ROOT / "web_logs.py")
verify_deadman = _load("w6_verify_deadman", ROOT / "scripts" / "verify_deadman.py")
dash = _load("w6_dash_app", ROOT / "src" / "ait" / "dashboard" / "app.py")


# ---------------------------------------------------------------------------
# Fixtures — real artefacts, no mocks of the code under test
# ---------------------------------------------------------------------------

@pytest.fixture
def state_db(tmp_path) -> Path:
    """A real ledger DB, schema created by the production StateManager."""
    db = tmp_path / "ait_state.db"
    StateManager(db)
    return db


@pytest.fixture
def conn(state_db):
    c = sqlite3.connect(state_db)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


@pytest.fixture
def publisher(state_db) -> HealthStatePublisher:
    # Explicit db_path -> the publisher is enabled even under pytest, so the
    # REAL write path runs (the auto path stays off so no test can ever write
    # the production ledger).
    return HealthStatePublisher(state_db, min_interval_s=0.0)


_EMIT_SCRIPT = textwrap.dedent(
    """
    import io, logging, sys
    from pathlib import Path
    out = Path(sys.argv[1])

    from ait.config.settings import LoggingConfig
    from ait.utils.logging import setup_logging, get_logger

    def emit(target, tty):
        class Out(io.StringIO):
            def isatty(self):
                return tty
        real, sys.stdout = sys.stdout, Out()
        try:
            setup_logging(LoggingConfig(level="DEBUG", file=str(target)))
            log = get_logger("bot.orchestrator")
            log.info("ml_prediction", symbol="SPY", direction="up",
                     confidence="0.623")
            log.info("ml_prediction", symbol="IWM", direction="down",
                     confidence="0.551")
            log.info("signals_generated", symbol="QQQ", total_signals=3)
            log.info("trade_executed", trade_id="T-1", symbol="QQQ",
                     strategy="iron_condor", contracts=1)
            log.info("order_placed", symbol="QQQ", action="BUY", order_id=42)
            log.error("trading_cycle_error", error="boom bang")
        finally:
            for h in list(logging.getLogger().handlers):
                h.close()
                logging.getLogger().removeHandler(h)
            sys.stdout = real

    emit(out / "json.log", False)
    emit(out / "ansi.log", True)
    """
)


@pytest.fixture(scope="session")
def real_log_files(tmp_path_factory) -> dict[str, Path]:
    """Log files written by the REAL ait.utils.logging pipeline.

    Produced in a child process so the production setup_logging genuinely runs
    (it reconfigures structlog and the root logger globally) without leaking
    that configuration into the rest of the suite.  ``ansi.log`` is the
    log-contracts-6 case: stdout.isatty() -> ConsoleRenderer -> ANSI escapes in
    the production log file.
    """
    out = tmp_path_factory.mktemp("w6_real_logs")
    script = out / "emit.py"
    script.write_text(_EMIT_SCRIPT, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(script), str(out)],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    files = {"json": out / "json.log", "ansi": out / "ansi.log"}
    for path in files.values():
        assert path.exists() and path.stat().st_size > 0, proc.stderr
    return files


@pytest.fixture(scope="session")
def real_fatal_dump(tmp_path_factory) -> Path:
    """A genuine faulthandler crash dump from a real child-process fault."""
    out = tmp_path_factory.mktemp("w6_fatal")
    script = out / "crash.py"
    script.write_text(
        "import faulthandler, sys\n"
        "f = open(sys.argv[1], 'a')\n"
        "faulthandler.enable(file=f)\n"
        "faulthandler._read_null()\n",
        encoding="utf-8",
    )
    target = out / "fatal.log"
    subprocess.run([sys.executable, str(script), str(target)],
                   capture_output=True)
    text = target.read_text(encoding="utf-8", errors="replace")
    assert "fatal" in text.lower(), text
    return target


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _utc_stamp(dt: datetime) -> str:
    """Render a local datetime the way the production TimeStamper does (UTC Z)."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _json_line(event: str, when: datetime, **fields) -> str:
    payload = {"component": "bot.orchestrator", **fields, "event": event,
               "level": "info", "logger": "__main__",
               "timestamp": _utc_stamp(when)}
    return json.dumps(payload)


# ===========================================================================
# string-contracts-5 / db-contracts-6 — the System Health writer half
# ===========================================================================

class TestHealthChannelWriter:

    def test_watchdog_persists_component_health_under_the_dashboard_keys(
            self, publisher, conn):
        """POSITIVE: the keys the dashboard reads are actually written."""
        wd = Watchdog(error_threshold=4, state_publisher=publisher)
        wd.register_component("trading_loop")
        wd.register_component("ibkr")
        wd.heartbeat("trading_loop")
        wd.heartbeat("ibkr")

        keys = {r[0] for r in conn.execute("SELECT key FROM bot_state")}
        assert "watchdog_trading_loop" in keys
        assert "watchdog_ibkr" in keys
        assert "watchdog_system" in keys
        assert "system_memory_usage" in keys
        assert "watchdog_errors" in keys
        assert "watchdog_channel" in keys

        stored = json.loads(conn.execute(
            "SELECT value FROM bot_state WHERE key='watchdog_trading_loop'"
        ).fetchone()[0])
        assert stored["status"] == ComponentStatus.HEALTHY.value
        assert stored["last_heartbeat"]

    def test_component_panel_reports_the_true_down_status(self, publisher, conn):
        """POSITIVE: a DOWN component reaches the operator surface as DOWN."""
        wd = Watchdog(error_threshold=2, state_publisher=publisher)
        wd.register_component("trading_loop")
        wd.record_error("trading_loop", "ibkr disconnected")
        wd.record_error("trading_loop", "ibkr disconnected")
        assert wd.get_health().components["trading_loop"].status is ComponentStatus.DOWN

        panel = dash.component_status_panel(conn)
        assert panel["state"] == dash.PANEL_OK
        by_name = {c["name"]: c for c in panel["components"]}
        assert by_name["Trading Loop"]["status"] == "down"
        assert by_name["Trading Loop"]["icon"] == "🔴"
        assert by_name["Trading Loop"]["last_error"] == "ibkr disconnected"

    def test_component_panel_says_not_wired_on_an_untouched_ledger(self, conn):
        """NEGATIVE: no writer -> NOT WIRED, and never a green/empty all-clear."""
        panel = dash.component_status_panel(conn)
        assert panel["state"] == dash.PANEL_NOT_WIRED
        assert panel["components"] == []
        assert "NOT WIRED" in panel["message"]
        assert "not an all-clear" in panel["message"].lower()

    def test_stale_channel_never_shows_a_green_component_dot(
            self, publisher, conn):
        """NEGATIVE: a healthy dot published hours ago is history, not health."""
        wd = Watchdog(state_publisher=publisher)
        wd.register_component("trading_loop")
        wd.heartbeat("trading_loop")
        old = datetime.now() - timedelta(seconds=CHANNEL_MAX_AGE_S + 600)
        publisher.publish(wd.get_health(), [], now=old, force=True)

        panel = dash.component_status_panel(conn)
        assert panel["state"] == dash.PANEL_STALE
        assert panel["components"], "components must still be listed"
        assert all(c["icon"] != "🟢" for c in panel["components"])
        assert "STALE" in panel["message"]

    def test_memory_panel_reports_real_numbers_then_not_wired(
            self, publisher, conn, state_db):
        """POSITIVE + NEGATIVE for bot_state['system_memory_usage']."""
        wd = Watchdog(state_publisher=publisher)
        wd.heartbeat("trading_loop")
        panel = dash.memory_panel(conn)
        assert panel["state"] == dash.PANEL_OK
        assert panel["memory"]["rss_mb"] > 0

        conn.execute("DELETE FROM bot_state WHERE key='system_memory_usage'")
        conn.commit()
        blank = dash.memory_panel(conn)
        assert blank["state"] == dash.PANEL_NOT_WIRED
        assert "NOT WIRED" in blank["message"]

    def test_publisher_never_creates_a_database(self, tmp_path):
        """NEGATIVE: monitoring must not conjure a ledger out of a bad path."""
        missing = tmp_path / "nope" / "ait_state.db"
        pub = HealthStatePublisher(missing, min_interval_s=0.0)
        wd = Watchdog(state_publisher=pub)
        wd.heartbeat("trading_loop")
        assert pub.publish(wd.get_health(), [], force=True) is False
        assert not missing.exists()
        assert "missing" in (pub.last_error or "")

    def test_publisher_refuses_a_db_without_bot_state(self, tmp_path):
        """NEGATIVE: never write schema into somebody else's sqlite file."""
        other = tmp_path / "other.db"
        sqlite3.connect(other).close()
        pub = HealthStatePublisher(other, min_interval_s=0.0)
        wd = Watchdog(state_publisher=pub)
        assert pub.publish(wd.get_health(), [], force=True) is False
        assert "bot_state" in (pub.last_error or "")

    def test_default_publisher_is_disabled_under_pytest(self):
        """NEGATIVE: an auto-path publisher must never touch data/ait_state.db."""
        assert HealthStatePublisher().enabled is False


class TestErrorPanel:
    """The worst offender: st.success('No errors logged') on an absent channel."""

    def test_recorded_errors_are_reported(self, publisher, conn):
        """POSITIVE: real watchdog errors reach the panel."""
        wd = Watchdog(error_threshold=10, state_publisher=publisher)
        wd.record_error("trading_loop", "fast_monitor: connection reset")
        wd.record_error("ibkr", "socket closed")

        panel = dash.error_panel(conn)
        assert panel["state"] == dash.PANEL_OK
        assert len(panel["errors"]) == 2
        assert panel["errors"][0]["error"] == "fast_monitor: connection reset"
        assert panel["source"] == "watchdog_errors"

    def test_absent_channel_is_not_wired_never_green(self, conn):
        """NEGATIVE: this is the exact false all-clear from the finding."""
        panel = dash.error_panel(conn)
        assert panel["state"] == dash.PANEL_NOT_WIRED
        assert "NOT WIRED" in panel["message"]
        assert "not an all-clear" in panel["message"].lower()
        assert "No errors logged" not in panel["message"]

    def test_green_only_when_channel_is_live_and_empty(self, publisher, conn):
        """POSITIVE: the single legitimate green branch."""
        wd = Watchdog(state_publisher=publisher)
        wd.register_component("trading_loop")
        wd.heartbeat("trading_loop")

        panel = dash.error_panel(conn)
        assert panel["state"] == dash.PANEL_EMPTY
        assert "live" in panel["message"]

    def test_empty_but_stale_channel_is_not_green(self, publisher, conn):
        """NEGATIVE: an empty list from a channel that stopped publishing."""
        wd = Watchdog(state_publisher=publisher)
        wd.register_component("trading_loop")
        old = datetime.now() - timedelta(hours=6)
        publisher.publish(wd.get_health(), [], now=old, force=True)

        panel = dash.error_panel(conn)
        assert panel["state"] == dash.PANEL_STALE
        assert panel["errors"] == []
        assert "not an all-clear" in panel["message"].lower()

    def test_error_log_key_wins_when_a_future_writer_appears(
            self, publisher, conn):
        """POSITIVE: the primary key is still honoured, no dashboard change."""
        wd = Watchdog(state_publisher=publisher)
        wd.record_error("trading_loop", "watchdog error")
        conn.execute(
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) "
            "VALUES ('error_log', ?, ?)",
            (json.dumps([{"time": "2026-08-31T10:00:00", "error": "from orchestrator"}]),
             _iso(datetime.now())),
        )
        conn.commit()
        panel = dash.error_panel(conn)
        assert panel["source"] == "error_log"
        assert panel["errors"][0]["error"] == "from orchestrator"


class TestModelAndMetaLabelPanels:

    def test_model_info_reads_the_real_trade_context_version(self, conn):
        """POSITIVE: a dead bot_state key replaced by a column that IS written."""
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, status, "
            "entry_time, entry_price, quantity, contract_type) "
            "VALUES ('T-1','QQQ','iron_condor','neutral','closed',"
            "'2026-08-24T10:00:40', 6.12, 1, 'option')")
        conn.execute(
            "INSERT INTO trade_context (trade_id, entry_direction, model_version) "
            "VALUES ('T-1','neutral','v-20260824-082050')")
        conn.commit()

        panel = dash.model_info_panel(conn)
        assert panel["state"] == dash.PANEL_OK
        assert panel["source"] == "trade_context.model_version"
        assert panel["model"]["model_version"] == "v-20260824-082050"

    def test_model_info_says_not_wired_with_no_version_anywhere(self, conn):
        """NEGATIVE."""
        panel = dash.model_info_panel(conn)
        assert panel["state"] == dash.PANEL_NOT_WIRED
        assert "NOT WIRED" in panel["message"]

    def test_meta_label_says_not_wired_not_not_yet_trained(self, conn):
        """NEGATIVE: the old copy asserted a training fact it could not know."""
        panel = dash.meta_label_panel(conn)
        assert panel["state"] == dash.PANEL_NOT_WIRED
        assert "NOT WIRED" in panel["message"]
        assert "not yet trained" not in panel["message"].lower()

    def test_meta_label_renders_real_stats_when_a_writer_exists(self, conn):
        """POSITIVE."""
        conn.execute(
            "INSERT INTO bot_state (key, value, updated_at) VALUES "
            "('meta_label_stats', ?, ?)",
            (json.dumps({"accuracy": 0.61, "precision": 0.58, "trades_used": 42}),
             _iso(datetime.now())))
        conn.commit()
        panel = dash.meta_label_panel(conn)
        assert panel["state"] == dash.PANEL_OK
        assert panel["stats"]["trades_used"] == 42


# ===========================================================================
# log-contracts-6 — ANSI-rendered production logs must stay parseable
# ===========================================================================

class TestAnsiRobustParsing:

    def test_production_json_lines_parse(self, real_log_files):
        """POSITIVE: baseline — the normal, non-tty rendering."""
        lines = real_log_files["json"].read_text(encoding="utf-8").splitlines()
        events = [parse_event_line(ln) for ln in lines]
        names = [e["event"] for e in events if e]
        assert "ml_prediction" in names
        assert "trade_executed" in names
        first = next(e for e in events if e and e["event"] == "ml_prediction")
        assert first["symbol"] == "SPY"
        assert first["_fmt"] == "json"

    def test_ansi_console_lines_are_unparseable_as_json_but_parse_here(
            self, real_log_files):
        """POSITIVE + the proof the fix matters (log-contracts-6)."""
        raw = real_log_files["ansi"].read_text(encoding="utf-8")
        assert "\x1b[" in raw, "fixture must contain real ANSI escapes"
        assert '"timestamp": "' not in raw, "ConsoleRenderer, not JSONRenderer"

        lines = [ln for ln in raw.splitlines() if ln.strip()]
        # The pre-W6 consumer was json.loads-only: it saw nothing at all.
        for line in lines:
            with pytest.raises(ValueError):
                json.loads(line)

        events = [parse_event_line(ln) for ln in lines]
        assert all(e is not None for e in events)
        names = [e["event"] for e in events]
        assert names.count("ml_prediction") == 2
        assert "trading_cycle_error" in names

        pred = next(e for e in events if e["event"] == "ml_prediction")
        assert pred["symbol"] == "SPY"
        assert pred["confidence"] == "0.623"
        assert pred["component"] == "bot.orchestrator"
        assert pred["_fmt"] == "console"

        err = next(e for e in events if e["event"] == "trading_cycle_error")
        assert err["level"] == "error"
        # value with spaces is repr-quoted by ConsoleRenderer
        assert err["error"] == "boom bang"

    def test_ansi_counters_match_json_counters(self, real_log_files):
        """POSITIVE: the two renderings yield identical today-counts."""
        events = ("ml_prediction", "trade_executed", "signals_generated")
        j = count_events_today(real_log_files["json"], events,
                               include_rotations=False)
        a = count_events_today(real_log_files["ansi"], events,
                               include_rotations=False)
        assert j.counts == a.counts == {
            "ml_prediction": 2, "trade_executed": 1, "signals_generated": 1}

    def test_web_logs_renders_ansi_events_without_escape_sequences(
            self, real_log_files):
        """POSITIVE: the viewer classifies ANSI lines instead of dumping them."""
        line = next(ln for ln in
                    real_log_files["ansi"].read_text(encoding="utf-8").splitlines()
                    if "trade_executed" in strip_ansi(ln))
        html, category, event, symbol = web_logs.parse_log_line(line)
        assert event == "trade_executed"
        assert category == "trade"
        assert symbol == "QQQ"
        assert "\x1b" not in html


# ===========================================================================
# log-contracts-1 / -4 — the 200-line window, the 76 MB re-read, UTC vs local
# ===========================================================================

class TestDayCounters:

    def _write_day(self, path: Path, *, trades: int, noise: int,
                   when: datetime) -> None:
        lines = [_json_line("trade_executed", when, symbol="QQQ", trade_id=f"T-{i}")
                 for i in range(trades)]
        lines += [_json_line("scan_symbol_timing", when, symbol="SPY",
                             elapsed=0.4 + i) for i in range(noise)]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_counts_a_trade_buried_behind_the_old_200_line_window(self, tmp_path):
        """POSITIVE: log-contracts-4's headline symptom."""
        log = tmp_path / "bot_stdout.log"
        now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        self._write_day(log, trades=3, noise=5000, when=now)

        # Reproduce the pre-W6 reader to prove the window really did hide it.
        old_window = log.read_text(encoding="utf-8").splitlines()[-200:]
        assert not any("trade_executed" in ln for ln in old_window)

        snap = count_events_today(log, ("trade_executed",),
                                  include_rotations=False)
        assert snap.counts["trade_executed"] == 3

    def test_refresh_reads_only_newly_appended_bytes(self, tmp_path):
        """POSITIVE: no more full re-read every 5 s per browser tab."""
        log = tmp_path / "bot_stdout.log"
        now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
        self._write_day(log, trades=1, noise=4000, when=now)

        counter = IncrementalEventCounter(
            log, ("trade_executed",), min_interval_s=0.0,
            include_rotations=False)
        first = counter.refresh(force=True)
        assert first.counts["trade_executed"] == 1
        seeded_bytes = first.bytes_scanned
        assert seeded_bytes >= log.stat().st_size * 0.9

        with open(log, "a", encoding="utf-8") as fh:
            fh.write(_json_line("trade_executed", now, symbol="IWM") + "\n")
        second = counter.refresh(force=True)
        assert second.counts["trade_executed"] == 2
        # The incremental read must be a tiny fraction of the file.
        assert (second.bytes_scanned - seeded_bytes) < 1000

    def test_counts_todays_rotated_backups(self, tmp_path):
        """POSITIVE: log-contracts-1 — today lives mostly in ait.log.1..N."""
        log = tmp_path / "ait.log"
        now = datetime.now().replace(hour=11, minute=0, second=0, microsecond=0)
        (tmp_path / "ait.log.1").write_text(
            "\n".join(_json_line("ml_prediction", now, symbol="SPY")
                      for _ in range(27)) + "\n", encoding="utf-8")
        (tmp_path / "ait.log.2").write_text(
            "\n".join(_json_line("ml_prediction", now, symbol="QQQ")
                      for _ in range(28)) + "\n", encoding="utf-8")
        log.write_text(_json_line("ml_prediction", now, symbol="IWM") + "\n",
                       encoding="utf-8")

        live_only = count_events_today(log, ("ml_prediction",),
                                       include_rotations=False)
        assert live_only.counts["ml_prediction"] == 1  # the old, wrong answer

        snap = count_events_today(log, ("ml_prediction",))
        assert snap.counts["ml_prediction"] == 56
        assert "ait.log.2" in snap.sources and "ait.log" in snap.sources

    def test_utc_stamps_bucket_into_the_local_day(self, tmp_path):
        """NEGATIVE: a UTC stamp whose LOCAL day is yesterday is not 'today'."""
        offset = datetime.now().astimezone().utcoffset()
        if offset is None or offset == timedelta(0):
            pytest.skip("needs a non-UTC local zone to distinguish the days")

        now_local = datetime.now().astimezone()
        today_local = now_local.replace(hour=12, minute=0, second=0, microsecond=0)
        # A moment that is a different LOCAL day but whose UTC date string
        # still contains today's local date -> the old substring match counted it.
        other_local = today_local - timedelta(days=1) if offset < timedelta(0) \
            else today_local + timedelta(days=1)
        # Land it inside the UTC window that carries today's local date.
        other_local = other_local.replace(
            hour=1 if offset < timedelta(0) else 22)

        log = tmp_path / "bot_stdout.log"
        log.write_text(
            _json_line("trade_executed", today_local, symbol="SPY") + "\n"
            + _json_line("trade_executed", other_local, symbol="QQQ") + "\n",
            encoding="utf-8")

        snap = count_events_today(log, ("trade_executed",),
                                  include_rotations=False)
        assert snap.counts["trade_executed"] == 1
        assert snap.last_record["symbol"] == "SPY"

    def test_rotation_mid_session_keeps_counting(self, tmp_path):
        """POSITIVE: the counter survives the file being rotated under it."""
        log = tmp_path / "bot_stdout.log"
        now = datetime.now().replace(hour=13, minute=0, second=0, microsecond=0)
        log.write_text(
            "\n".join(_json_line("trade_executed", now, symbol="SPY")
                      for _ in range(5)) + "\n", encoding="utf-8")
        counter = IncrementalEventCounter(log, ("trade_executed",),
                                          min_interval_s=0.0,
                                          include_rotations=False)
        assert counter.refresh(force=True).counts["trade_executed"] == 5

        # RotatingFileHandler rolls the file over and starts a fresh, smaller one.
        log.replace(tmp_path / "bot_stdout.log.1")
        log.write_text(_json_line("trade_executed", now, symbol="IWM") + "\n",
                       encoding="utf-8")
        snap = counter.refresh(force=True)
        assert snap.counts["trade_executed"] == 6
        assert snap.rotations == 1
        assert snap.partial is True


class TestWebLogsViewer:

    def test_api_reports_todays_trades_from_outside_the_display_window(
            self, tmp_path, monkeypatch):
        """POSITIVE: end-to-end through the real Flask endpoint."""
        log = tmp_path / "bot_stdout.log"
        now = datetime.now().replace(hour=10, minute=30, second=0, microsecond=0)
        lines = [_json_line("trade_executed", now, symbol="QQQ", trade_id="T-1"),
                 _json_line("ml_prediction", now, symbol="SPY", confidence="0.6"),
                 _json_line("order_placed", now, symbol="QQQ", order_id=1),
                 _json_line("order_placed", now, symbol="QQQ", order_id=2)]
        lines += [_json_line("scan_symbol_timing", now, symbol="SPY")
                  for _ in range(3000)]
        log.write_text("\n".join(lines) + "\n", encoding="utf-8")

        orch = tmp_path / "orchestrator.log"
        orch.write_text(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] DEBUG "
            "orchestrator.bot_healthy | pid=1234\n", encoding="utf-8")

        monkeypatch.setattr(web_logs, "LOG_FILE", log)
        monkeypatch.setattr(web_logs, "ORCH_LOG", orch)
        monkeypatch.setattr(web_logs, "_COUNTER", IncrementalEventCounter(
            log, web_logs.COUNTED_EVENTS, min_interval_s=0.0,
            include_rotations=False))

        with web_logs.app.test_request_context("/api/logs?filter=all"):
            payload = web_logs.api_logs()

        assert payload["trades_today"] == "1"
        assert payload["orders_today"] == "2"
        assert payload["predictions_today"] == "1"
        assert payload["last_signal"] == "SPY"
        assert payload["status"] == "Healthy"
        # ...and the trade is genuinely outside the old 200-line window.
        assert not any("trade_executed" in ln for ln in
                       log.read_text(encoding="utf-8").splitlines()[-200:])

    def test_display_read_is_byte_bounded_not_whole_file(self, tmp_path):
        """POSITIVE: tail_lines never reads the whole 76 MB file."""
        log = tmp_path / "big.log"
        chunk = ("x" * 200 + "\n") * 20000  # ~4 MB
        log.write_text(chunk, encoding="utf-8")
        lines = tail_lines(log, 64 * 1024)
        assert 0 < len(lines) < 400
        assert log.stat().st_size > 3_000_000

    def test_status_marks_a_stale_supervisor_log(self, tmp_path):
        """NEGATIVE: 'Healthy' from a three-hour-old line is a false all-clear."""
        orch = tmp_path / "orchestrator.log"
        old = datetime.now() - timedelta(hours=3)
        orch.write_text(f"[{old:%Y-%m-%d %H:%M:%S}] DEBUG "
                        "orchestrator.bot_healthy | pid=1\n", encoding="utf-8")
        status, age = web_logs.orchestrator_status(orch)
        assert "STALE" in status
        assert status != "Healthy"
        assert age > 3600

    def test_status_healthy_when_the_supervisor_is_current(self, tmp_path):
        """POSITIVE."""
        orch = tmp_path / "orchestrator.log"
        orch.write_text(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] DEBUG "
                        "orchestrator.bot_healthy | pid=1\n", encoding="utf-8")
        status, age = web_logs.orchestrator_status(orch)
        assert status == "Healthy"
        assert age < 60

    def test_status_reports_no_supervisor_log_rather_than_unknown_green(
            self, tmp_path):
        """NEGATIVE: an absent supervisor log is named, not silently 'Unknown'."""
        status, age = web_logs.orchestrator_status(tmp_path / "missing.log")
        assert status == "No supervisor log"
        assert age is None


# ===========================================================================
# log-contracts-3 — native crashes land in fatal.log, not bot_stdout.log
# ===========================================================================

class TestNativeCrashCounting:

    def test_counts_a_genuine_faulthandler_dump(self, real_fatal_dump):
        """POSITIVE: a real child-process access violation is counted as 1."""
        report = native_crash_report(real_fatal_dump, legacy_logs=())
        assert report["channel_wired"] is True
        assert report["count"] == 1
        assert report["fatal_count"] == 1
        assert report["last_write"]

    def test_the_pre_w6_consumer_would_have_reported_zero(
            self, tmp_path, real_fatal_dump):
        """NEGATIVE: reproduce status.py's bot_stdout.log grep on the same crash."""
        stdout_log = tmp_path / "bot_stdout.log"
        stdout_log.write_text("ordinary line\nanother line\n", encoding="utf-8")
        legacy_only = sum(1 for ln in stdout_log.read_text().splitlines()
                          if "Windows fatal exception" in ln)
        assert legacy_only == 0
        assert native_crash_report(real_fatal_dump,
                                   legacy_logs=(stdout_log,))["count"] == 1

    def test_still_counts_legacy_stdout_dumps(self, tmp_path):
        """POSITIVE: pre-move crashes in bot_stdout.log are not lost."""
        fatal = tmp_path / "fatal.log"
        fatal.write_text("", encoding="utf-8")
        stdout_log = tmp_path / "bot_stdout.log"
        stdout_log.write_text(
            "Windows fatal exception: access violation\n\n"
            "Current thread 0x00001234 (most recent call first):\n"
            "  File \"run_orchestrator.py\", line 1 in <module>\n",
            encoding="utf-8")
        report = native_crash_report(fatal, legacy_logs=(stdout_log,))
        assert report["count"] == 1
        assert report["legacy_count"] == 1

    def test_missing_fatal_log_reports_not_wired_never_a_clean_zero(
            self, tmp_path):
        """NEGATIVE: 'no crashes' and 'nothing records crashes' differ."""
        report = native_crash_report(tmp_path / "absent.log", legacy_logs=())
        assert report["channel_wired"] is False
        assert report["count"] == 0
        assert "NOT WIRED" in report["detail"]

        panel = dash.crash_panel(fatal_log=tmp_path / "absent.log",
                                 legacy_logs=())
        assert panel["state"] == dash.PANEL_NOT_WIRED
        assert "NOT WIRED" in panel["message"]

    def test_crash_panel_reports_the_true_count(self, real_fatal_dump):
        """POSITIVE: the dashboard surface shows 1, not 0."""
        panel = dash.crash_panel(fatal_log=real_fatal_dump, legacy_logs=())
        assert panel["state"] == dash.PANEL_OK
        assert panel["report"]["count"] == 1


# ===========================================================================
# bot-day-02 — the dead-man must attest the BOT, not the supervisor
# ===========================================================================

class TestBotLiveness:

    def _heartbeat(self, tmp_path: Path, age_s: float) -> Path:
        hb = tmp_path / "bot_heartbeat"
        hb.write_text(datetime.now().isoformat(), encoding="utf-8")
        import os
        stamp = datetime.now().timestamp() - age_s
        os.utime(hb, (stamp, stamp))
        return hb

    def test_fresh_heartbeat_during_rth_is_alive(self, tmp_path):
        """POSITIVE."""
        hb = self._heartbeat(tmp_path, 30)
        v = bot_liveness(heartbeat_path=hb, market_open=True,
                         minutes_since_open=120)
        assert v.ok is True
        assert v.state == "heartbeat_fresh"

    def test_stale_heartbeat_during_rth_fails_the_deadman(self, tmp_path):
        """NEGATIVE: the Telegram-dead + gateway-down outage from the finding."""
        hb = self._heartbeat(tmp_path, HEARTBEAT_MAX_AGE_S + 120)
        v = bot_liveness(heartbeat_path=hb, market_open=True,
                         minutes_since_open=180)
        assert v.ok is False
        assert v.state == "heartbeat_stale"
        assert "unmanaged" in v.detail

    def test_missing_heartbeat_during_rth_fails_the_deadman(self, tmp_path):
        """NEGATIVE: process existence is not liveness."""
        v = bot_liveness(heartbeat_path=tmp_path / "absent", market_open=True,
                         minutes_since_open=180)
        assert v.ok is False
        assert v.state == "heartbeat_missing"

    def test_outside_rth_a_stale_heartbeat_does_not_page(self, tmp_path):
        """NEGATIVE (alert fatigue): the loop legitimately stops overnight."""
        hb = self._heartbeat(tmp_path, 12 * 3600)
        v = bot_liveness(heartbeat_path=hb, market_open=False)
        assert v.ok is True
        assert v.state == "not_rth"

    def test_rth_warmup_window_does_not_page(self, tmp_path):
        """NEGATIVE (alert fatigue): a keeper relaunch at the bell is arming."""
        v = bot_liveness(heartbeat_path=tmp_path / "absent", market_open=True,
                         minutes_since_open=int(RTH_WARMUP_S // 60) - 1)
        assert v.ok is True
        assert v.state == "warmup"

    def test_threshold_matches_the_supervisors_own_hang_detector(self, tmp_path):
        """The 15-minute window is master.py:499's bot_hung_heartbeat_stale."""
        assert HEARTBEAT_MAX_AGE_S == 900.0
        just_inside = self._heartbeat(tmp_path, HEARTBEAT_MAX_AGE_S - 60)
        assert bot_liveness(heartbeat_path=just_inside, market_open=True,
                            minutes_since_open=120).ok is True
        just_outside = self._heartbeat(tmp_path, HEARTBEAT_MAX_AGE_S + 60)
        assert bot_liveness(heartbeat_path=just_outside, market_open=True,
                            minutes_since_open=120).ok is False


class TestVerifyDeadman:

    def test_ping_target_switches_to_the_fail_endpoint(self):
        """POSITIVE + NEGATIVE for the healthchecks.io endpoint choice."""
        url = "https://hc-ping.com/abc"
        assert verify_deadman.ping_target(url, True) == url
        assert verify_deadman.ping_target(url, False) == url + "/fail"

    def test_the_real_keeper_is_gated(self):
        """POSITIVE (was ...is_detected_as_ungated): the shipped keeper_ait.bat
        now gates its ping on real liveness evidence.

        W6/bot-day-02: this test originally asserted `gated is False` because
        keeper_ait.bat was outside the fixing agent's ownership — it PINNED the
        live defect (the ping fired on a process match, attesting the
        supervisor rather than the bot, so a Telegram-dead + gateway-down
        outage showed green for hours). The keeper was gated during
        integration, so the assertion is inverted: this now guards against a
        REGRESSION back to an ungated ping."""
        keeper = (ROOT / "keeper_ait.bat").read_text(encoding="utf-8",
                                                     errors="ignore")
        gating = verify_deadman.keeper_gating(keeper)
        assert gating["pings"] is True
        assert gating["gated"] is True
        assert verify_deadman.LIVENESS_MODULE in keeper
        # the /fail branch is what makes an outage alert NOW rather than
        # waiting out the check's grace period
        assert "/fail" in keeper

    def test_a_gated_keeper_is_recognised(self):
        """POSITIVE: the suggested patch satisfies the check."""
        patched = ("if exist data\\deadman_url.txt curl.exe\n"
                   + verify_deadman.KEEPER_PATCH)
        gating = verify_deadman.keeper_gating(patched)
        assert gating["pings"] is True
        assert gating["gated"] is True

    def test_main_refuses_a_green_ping_when_the_bot_is_not_alive(
            self, tmp_path, monkeypatch, capsys):
        """NEGATIVE: the whole point — no success ping during an outage."""
        from ait.monitoring.ops_health import LivenessVerdict

        url_file = tmp_path / "deadman_url.txt"
        url_file.write_text("https://hc-ping.com/abc\n", encoding="utf-8")
        sent: list[str] = []

        monkeypatch.setattr(verify_deadman, "URL_FILE", url_file)
        monkeypatch.setattr(verify_deadman, "KEEPER", tmp_path / "keeper.bat")
        monkeypatch.setattr(verify_deadman, "bot_liveness", lambda **kw:
                            LivenessVerdict(ok=False, state="heartbeat_stale",
                                            detail="45 min stale", rth=True))
        monkeypatch.setattr(verify_deadman, "_curl",
                            lambda curl, target: sent.append(target) or 0)

        rc = verify_deadman.main([])
        assert rc == 3
        assert sent == ["https://hc-ping.com/abc/fail"]
        out = capsys.readouterr().out
        assert "NOT ALIVE" in out
        assert "a green ping would be a lie" in out

    def test_main_pings_green_when_the_bot_is_demonstrably_alive(
            self, tmp_path, monkeypatch):
        """POSITIVE."""
        from ait.monitoring.ops_health import LivenessVerdict

        url_file = tmp_path / "deadman_url.txt"
        url_file.write_text("https://hc-ping.com/abc\n", encoding="utf-8")
        sent: list[str] = []

        monkeypatch.setattr(verify_deadman, "URL_FILE", url_file)
        monkeypatch.setattr(verify_deadman, "KEEPER", tmp_path / "keeper.bat")
        monkeypatch.setattr(verify_deadman, "bot_liveness", lambda **kw:
                            LivenessVerdict(ok=True, state="heartbeat_fresh",
                                            detail="30s old", rth=True))
        monkeypatch.setattr(verify_deadman, "_curl",
                            lambda curl, target: sent.append(target) or 0)

        assert verify_deadman.main([]) == 0
        assert sent == ["https://hc-ping.com/abc"]

    def test_read_deadman_url_handles_a_missing_file(self, tmp_path):
        """NEGATIVE."""
        assert verify_deadman.read_deadman_url(tmp_path / "nope.txt") is None


# ===========================================================================
# dead-surface-4 — capture efficiency was dollars / percent
# ===========================================================================

class TestCaptureEfficiency:

    #: The exact production row the finding cites (data/ait_state.db).
    NVDA = {"symbol": "NVDA", "strategy": "iron_condor",
            "exit_reason_detailed": "stop_loss", "peak_pnl_pct": 0.0077968,
            "realized_pnl": -127.69, "direction_correct": -1,
            "entry_price": 2.81, "quantity": 1, "contract_type": "option"}

    def test_units_are_consistent_and_the_old_number_is_gone(self):
        """POSITIVE: the -16376% artefact is replaced by a real ratio."""
        df = pd.DataFrame([self.NVDA])
        old = (self.NVDA["realized_pnl"] / (self.NVDA["peak_pnl_pct"] * 100)) * 100
        # the shipped, dimensionally broken value the finding measured (-16376%)
        assert -16400 < old < -16300

        out = dash.add_capture_efficiency(df)
        cost_basis = 2.81 * 1 * 100
        expected_realized_pct = -127.69 / cost_basis
        expected_capture = expected_realized_pct / self.NVDA["peak_pnl_pct"] * 100

        assert out["cost_basis"].iloc[0] == pytest.approx(cost_basis)
        assert out["realized_pnl_pct"].iloc[0] == pytest.approx(
            expected_realized_pct)
        assert out["capture_pct"].iloc[0] == pytest.approx(expected_capture)
        assert out["capture_pct"].iloc[0] != pytest.approx(old)

    def test_capture_is_invariant_to_position_size(self):
        """POSITIVE: the actual defect — 'doubling size doubled efficiency'."""
        one = dict(self.NVDA, quantity=1, realized_pnl=-127.69)
        two = dict(self.NVDA, quantity=2, realized_pnl=-255.38)
        out = dash.add_capture_efficiency(pd.DataFrame([one, two]))
        assert out["capture_pct"].iloc[0] == pytest.approx(
            out["capture_pct"].iloc[1])

        old_one = one["realized_pnl"] / (one["peak_pnl_pct"] * 100) * 100
        old_two = two["realized_pnl"] / (two["peak_pnl_pct"] * 100) * 100
        assert old_two == pytest.approx(old_one * 2)  # the bug, reproduced

    def test_well_captured_winner_reads_near_100_percent(self):
        """POSITIVE: a trade that gave back nothing scores ~100%."""
        row = {"symbol": "SPY", "strategy": "iron_condor",
               "exit_reason_detailed": "take_profit", "peak_pnl_pct": 0.683443,
               "realized_pnl": 280.36, "direction_correct": -1,
               "entry_price": 4.29, "quantity": 1, "contract_type": "option"}
        out = dash.add_capture_efficiency(pd.DataFrame([row]))
        assert 90 < out["capture_pct"].iloc[0] < 100

    def test_rows_without_a_cost_basis_are_dropped_and_declared(self):
        """NEGATIVE: never fabricate a ratio out of a zero basis."""
        bad = dict(self.NVDA, entry_price=0.0)
        panel = dash.capture_efficiency_panel(pd.DataFrame([bad]))
        assert panel["state"] == dash.PANEL_NOT_WIRED
        assert "not computable" in panel["message"]
        assert panel["avg_capture"] is None

    def test_panel_headlines_the_median_and_flags_outliers(self):
        """POSITIVE: a near-zero peak cannot hijack the headline number."""
        rows = [
            self.NVDA,
            {"symbol": "SPY", "strategy": "iron_condor",
             "exit_reason_detailed": "tp", "peak_pnl_pct": 0.683443,
             "realized_pnl": 280.36, "direction_correct": -1,
             "entry_price": 4.29, "quantity": 1, "contract_type": "option"},
            {"symbol": "QQQ", "strategy": "iron_condor",
             "exit_reason_detailed": "tp", "peak_pnl_pct": 0.402278,
             "realized_pnl": 205.65, "direction_correct": -1,
             "entry_price": 5.44, "quantity": 1, "contract_type": "option"},
        ]
        panel = dash.capture_efficiency_panel(pd.DataFrame(rows))
        assert panel["state"] == dash.PANEL_OK
        assert 80 < panel["median_capture"] < 110
        assert panel["avg_capture"] < panel["median_capture"]
        assert panel["outliers"] == 1
        assert "median is the headline" in panel["message"]

    def test_empty_input_is_empty_not_not_wired(self):
        """NEGATIVE boundary: 'no closes yet' is a real, honest empty."""
        panel = dash.capture_efficiency_panel(pd.DataFrame())
        assert panel["state"] == dash.PANEL_EMPTY


# ===========================================================================
# dead-surface-3 — ML Direction Accuracy can never populate
# ===========================================================================

class TestDirectionAccuracy:

    def _close(self, conn, trade_id, direction_correct, pnl):
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, status, "
            "entry_time, entry_price, quantity, contract_type, realized_pnl, "
            "exit_reason_detailed, peak_pnl_pct, direction_correct) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trade_id, "SPY", "long_call", "long", "closed",
             "2026-08-24T10:00:00", 2.50, 1, "option", pnl,
             "take_profit", 0.3, direction_correct))
        conn.commit()

    def _exit_data(self, conn) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT symbol, strategy, exit_reason_detailed, peak_pnl_pct, "
            "realized_pnl, direction_correct, entry_price, quantity, "
            "contract_type FROM trades WHERE status='closed'", conn)

    def test_all_minus_one_reports_not_recorded_not_no_data_yet(self, conn):
        """NEGATIVE: the production state — 28 closes, all at the DDL default."""
        for i in range(5):
            self._close(conn, f"T-{i}", -1, 10.0)

        panel = dash.direction_accuracy_panel(self._exit_data(conn))
        assert panel["state"] == dash.PANEL_NOT_WIRED
        assert panel["unset"] == 5
        assert "NOT RECORDED" in panel["message"]
        assert "close_trade" in panel["message"]
        assert "no direction accuracy data yet" not in panel["message"].lower()

    def test_computes_the_true_accuracy_once_the_column_is_written(self, conn):
        """POSITIVE: the panel works the moment a writer appears."""
        self._close(conn, "T-a", 1, 50.0)
        self._close(conn, "T-b", 1, -20.0)
        self._close(conn, "T-c", 0, -30.0)
        self._close(conn, "T-d", -1, 5.0)

        panel = dash.direction_accuracy_panel(self._exit_data(conn))
        assert panel["state"] == dash.PANEL_OK
        assert panel["correct"] == 2
        assert panel["total"] == 3
        assert panel["accuracy"] == pytest.approx(200 / 3)
        assert panel["right_but_lost"] == 1
        assert panel["unset"] == 1
        assert "1 further closed trade" in panel["message"]

    def test_no_rows_at_all_reports_not_recorded(self, conn):
        """NEGATIVE boundary."""
        panel = dash.direction_accuracy_panel(self._exit_data(conn))
        assert panel["state"] == dash.PANEL_NOT_WIRED
        assert "NOT RECORDED" in panel["message"]


# ===========================================================================
# The rendered tab itself — the strongest form of "never green when unwired"
# ===========================================================================

class _FakeColumn:
    def __init__(self, sink):
        self.sink = sink

    def metric(self, label, value, *a, **kw):
        self.sink.append(("metric", str(label), str(value)))


class _FakeStreamlit:
    """Records every widget call so a test can assert what the operator sees."""

    def __init__(self):
        self.calls: list[tuple] = []

    def _rec(self, name):
        def fn(*args, **kwargs):
            self.calls.append((name,) + tuple(str(a)[:2000] for a in args))
        return fn

    def __getattr__(self, name):
        if name == "columns":
            def columns(n, *a, **kw):
                count = n if isinstance(n, int) else len(n)
                return [_FakeColumn(self.calls) for _ in range(count)]
            return columns
        return self._rec(name)

    def text(self, kind) -> str:
        return " ".join(str(part) for call in self.calls
                        if call[0] == kind for part in call[1:])


@pytest.fixture
def fake_st(monkeypatch):
    fake = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    return fake


class TestSystemHealthTabRendering:

    def test_unwired_ledger_never_renders_a_green_success(self, conn, fake_st):
        """NEGATIVE: the shipped tab called st.success('No errors logged')."""
        dash._tab_system_health(conn)
        successes = [c for c in fake_st.calls if c[0] == "success"]
        # The only success allowed here is the bot-liveness verdict, which is
        # evidence-based; the error channel must NOT produce one.
        assert all("errors" not in " ".join(c[1:]).lower() for c in successes)
        warnings = fake_st.text("warning")
        assert "NOT WIRED" in warnings
        assert "Error channel NOT WIRED" in warnings
        assert "Component Status NOT WIRED" in warnings
        assert "No errors logged" not in " ".join(
            " ".join(c[1:]) for c in fake_st.calls)

    def test_live_empty_error_channel_is_the_only_green(
            self, publisher, conn, fake_st):
        """POSITIVE: a wired, current, error-free channel does say so."""
        wd = Watchdog(state_publisher=publisher)
        wd.register_component("trading_loop")
        wd.heartbeat("trading_loop")

        dash._tab_system_health(conn)
        successes = fake_st.text("success")
        assert "No errors recorded" in successes
        assert "live" in successes
        assert "Error channel NOT WIRED" not in fake_st.text("warning")

    def test_recorded_errors_reach_the_rendered_tab(
            self, publisher, conn, fake_st):
        """POSITIVE: a real error is displayed, not swallowed."""
        wd = Watchdog(error_threshold=2, state_publisher=publisher)
        wd.record_error("trading_loop", "gateway refused connection")
        wd.record_error("trading_loop", "gateway refused connection")

        dash._tab_system_health(conn)
        rendered = " ".join(" ".join(c[1:]) for c in fake_st.calls)
        assert "gateway refused connection" in rendered
        assert "No errors recorded" not in rendered


# ===========================================================================
# Cross-cutting: the channel-state reader itself
# ===========================================================================

class TestChannelState:

    def test_absent_key_is_never_trustworthy(self, conn):
        """NEGATIVE."""
        state = read_channel_state(conn)
        assert state.wired is False
        assert state.trustworthy is False
        assert "NOT WIRED" in state.detail

    def test_fresh_publish_is_trustworthy(self, publisher, conn):
        """POSITIVE."""
        wd = Watchdog(state_publisher=publisher)
        wd.heartbeat("trading_loop")
        state = read_channel_state(conn)
        assert state.wired is True and state.fresh is True
        assert state.trustworthy is True

    def test_old_publish_is_wired_but_not_fresh(self, publisher, conn):
        """NEGATIVE."""
        wd = Watchdog(state_publisher=publisher)
        old = datetime.now() - timedelta(seconds=CHANNEL_MAX_AGE_S + 60)
        publisher.publish(wd.get_health(), [], now=old, force=True)
        state = read_channel_state(conn)
        assert state.wired is True
        assert state.fresh is False
        assert state.trustworthy is False
        assert "STALE" in state.detail

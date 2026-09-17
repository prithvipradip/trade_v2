"""W21: stored trade timestamps are timezone-aware, on any host.

`trades.entry_time` / `trades.exit_time` (plus the `open_positions` and
`exit_outbox` mirrors) were written by naive `datetime.now()`. Every row in the
corpus was written on the old ET host, so it was ET *by accident* — the moment
the box moved to CEST the same column would have started meaning something
else, silently, with no marker in the data to say which rows were which.

WHY THIS FILE PINS A HOST TIMEZONE. Same lesson W20 paid for: a timezone test
that passes explicit naive datetimes exercises only the arithmetic and never
the `datetime.now()` path where the host clock enters. Every writer test here
pins a real non-ET `TZ` and drives the no-argument path, and
`TestTheHarnessItself` fails loudly if the pin ever stops taking effect.

WHAT MAKES A HALF-MIGRATION DANGEROUS, and why the class guard below is not
redundant with the unit tests: `reconciler.py`'s stale-PENDING sweep wraps its
only arithmetic site in `except (ValueError, TypeError): continue`. Mixing
naive and aware values raises TypeError *there*, gets swallowed, and the sweep
degrades to a no-op with nothing in the log. The failure mode of a partial
migration is silence, so "we changed every site we found" has to be enforced
mechanically rather than asserted.
"""
from __future__ import annotations

import ast
import os
import sqlite3
import time as _time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import ait.bot.state as _state_mod
import ait.execution.executor as _executor_mod
import ait.execution.reconciler as _reconciler_mod
import ait.utils.time as _time_mod
from ait.bot.state import StateManager
from ait.utils.time import ET, now_et, parse_trade_ts

_real_datetime = datetime

SRC = Path(__file__).resolve().parents[1] / "src"

# The columns whose convention W21 fixed, and the tables that mirror them.
MIGRATED_COLUMNS = ("entry_time", "exit_time")


# The pinned host zone. TZ + tzset moves the REAL process clock, which is the
# strongest form of this pin — but tzset is POSIX-only and Windows ignores TZ,
# so on the Windows box (the one that trades) every test in this file errored
# in setup AND teardown (R26: w21-tz-pin-fixture-calls-tzset). The seam below
# models the host instead: in every module that writes these columns,
# `datetime.now()` with no argument returns the pinned zone's NAIVE wall
# clock, while `datetime.now(tz)` stays a genuine conversion — so a writer
# that regresses to a naive `datetime.now()` stores Berlin wall time and the
# offset assertions catch it, on any host. POSIX still gets the real tzset.
_HOST: dict = {"zone": None}
_CLOCK_MODULES = (_state_mod, _executor_mod, _reconciler_mod, _time_mod)


class _HostClock(_real_datetime):
    @classmethod
    def now(cls, tz=None):
        real = _real_datetime.now(timezone.utc)
        if tz is None:
            zone = _HOST["zone"]
            return (real.astimezone(zone) if zone else real.astimezone()).replace(tzinfo=None)
        return real.astimezone(tz)


@pytest.fixture
def host_tz(monkeypatch):
    """Pin the process's LOCAL timezone, the way the box's own clock is set."""
    def _set(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        if hasattr(_time, "tzset"):
            _time.tzset()
        _HOST["zone"] = ZoneInfo(name)
        for mod in _CLOCK_MODULES:
            monkeypatch.setattr(mod, "datetime", _HostClock)
    yield _set
    _HOST["zone"] = None
    monkeypatch.undo()
    if hasattr(_time, "tzset"):
        _time.tzset()


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _seed(st, trade_id="T-1", status="filled"):
    """Seed a trade the way the real writer now does — an AWARE entry_time."""
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, "
            "status, entry_time, entry_price, quantity, contract_type) "
            "VALUES (?,?,?,?,?,?,?,1,'iron_condor')",
            (trade_id, "QQQ", "iron_condor", "short", status,
             now_et().isoformat(), 1.90))


class TestTheHarnessItself:
    """If the TZ pin silently no-ops, every writer test below is vacuous."""

    def test_tz_pin_actually_moves_the_naive_local_clock(self, host_tz):
        host_tz("Europe/Berlin")
        berlin = _state_mod.datetime.now()      # the clock the WRITERS see
        host_tz("US/Eastern")
        eastern = _state_mod.datetime.now()
        drift_h = round((berlin - eastern).total_seconds() / 3600)
        assert drift_h in (5, 6), (
            f"TZ pin did not take effect (drift {drift_h}h); the CEST/ET gap "
            "is 6h in summer, 5h in winter"
        )

    def test_the_seam_leaves_aware_conversion_genuine(self, host_tz):
        """`datetime.now(tz)` must stay a real conversion or the writers'
        `now_et()` would be pinned too and every offset assertion vacuous."""
        host_tz("Europe/Berlin")
        assert _time_mod.datetime.now(ET).utcoffset() == _real_datetime.now(ET).utcoffset()

    @pytest.mark.skipif(not hasattr(_time, "tzset"), reason="POSIX only")
    def test_tz_pin_also_moves_the_real_process_clock(self, host_tz):
        """Where the platform allows it the pin is the real thing, which is
        what proves the seam models the host rather than replacing it."""
        host_tz("Europe/Berlin")
        berlin = _real_datetime.now()
        host_tz("US/Eastern")
        eastern = _real_datetime.now()
        assert round((berlin - eastern).total_seconds() / 3600) in (5, 6)


class TestParseTradeTs:
    """The one place legacy-naive and migrated-aware rows are reconciled."""

    def test_naive_legacy_row_means_ET_not_the_host_clock(self, host_tz):
        """The whole point: a pre-W21 row is ET, wherever it is later read."""
        host_tz("Europe/Berlin")
        got = parse_trade_ts("2026-04-23T09:35:17.585857")
        assert got.utcoffset() == timedelta(hours=-4)
        assert (got.hour, got.minute) == (9, 35)

    def test_winter_row_gets_EST_not_a_hardcoded_EDT(self):
        """Offsets are derived per row, so the DST boundary is not a cliff."""
        assert parse_trade_ts("2026-01-15T09:35:00").utcoffset() == timedelta(hours=-5)
        assert parse_trade_ts("2026-07-15T09:35:00").utcoffset() == timedelta(hours=-4)

    def test_already_aware_row_is_preserved_as_an_instant(self):
        raw = "2026-04-23T09:35:17.585857-04:00"
        assert parse_trade_ts(raw) == datetime.fromisoformat(raw)

    def test_a_foreign_offset_is_converted_to_ET_wall_clock(self):
        """A row written on a CEST host must still bucket into the ET day."""
        got = parse_trade_ts("2026-09-16T15:35:00+02:00")
        assert (got.hour, got.date().isoformat()) == (9, "2026-09-16")

    def test_round_trip_is_idempotent(self):
        once = parse_trade_ts("2026-04-23T09:35:17.585857").isoformat()
        assert parse_trade_ts(once).isoformat() == once

    def test_unparseable_still_raises_ValueError(self):
        """Callers guard on ValueError/TypeError; that contract must hold."""
        with pytest.raises(ValueError):
            parse_trade_ts("not-a-timestamp")


class TestWritersEmitETOnAnyHost:
    """Drive the real StateManager with the host clock pinned off-ET."""

    def test_exit_time_carries_the_ET_offset_not_the_hosts(self, tmp_path, host_tz):
        host_tz("Europe/Berlin")
        st = _state(tmp_path)
        _seed(st)
        assert st.close_trade("T-1", 2.5, -150.0, exit_reason_detailed="stop_loss")
        with sqlite3.connect(st._db_path) as conn:
            raw = conn.execute(
                "SELECT exit_time FROM trades WHERE trade_id='T-1'").fetchone()[0]
        parsed = datetime.fromisoformat(raw)
        assert parsed.tzinfo is not None, f"exit_time still naive: {raw!r}"
        assert parsed.utcoffset() == now_et().utcoffset(), (
            f"exit_time carries the host offset, not ET: {raw!r}")

    def test_open_position_entry_time_is_aware(self, tmp_path, host_tz):
        host_tz("Europe/Berlin")
        st = _state(tmp_path)
        st.insert_open_position("T-2", "QQQ", "iron_condor", 1, 1.90)
        with sqlite3.connect(st._db_path) as conn:
            raw = conn.execute(
                "SELECT entry_time FROM open_positions "
                "WHERE trade_id='T-2'").fetchone()[0]
        assert datetime.fromisoformat(raw).tzinfo is not None, raw

    def test_outbox_cutoff_still_selects_rows_the_writer_produced(
            self, tmp_path, host_tz):
        """The string-compare hazard, pinned.

        `pending_exit_bookings` compares `exit_time < ?` as TEXT. If the stored
        value and the cutoff ever disagree about offsets, the comparison
        mis-selects with no error — on a CEST box a naive row sorts ABOVE an ET
        cutoff, so every pending booking silently goes unclaimed.
        """
        host_tz("Europe/Berlin")
        st = _state(tmp_path)
        _seed(st)
        st.close_trade("T-1", 2.5, -150.0, exit_reason_detailed="stop_loss")
        with sqlite3.connect(st._db_path) as conn:
            conn.execute(
                "UPDATE exit_outbox SET exit_time = ? WHERE trade_id = 'T-1'",
                ((now_et() - timedelta(minutes=30)).isoformat(),))
        assert [r["trade_id"] for r in st.pending_exit_bookings()] == ["T-1"]


class TestTheClassGuard:
    """Cheap guard against the NEXT instance, not just the ones W21 found.

    W18 [0c] and W20 both recorded the same failure: fixing the instances you
    know about and declaring the class closed. These scan the tree instead.
    """

    def test_no_naive_now_writes_a_migrated_column(self):
        """`entry_time=datetime.now()...` must not reappear in any writer."""
        offenders = []
        for path in SRC.rglob("*.py"):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if "datetime.now()" not in line:
                    continue
                if any(f"{c}=" in line or f"{c} =" in line for c in MIGRATED_COLUMNS):
                    offenders.append(f"{path.relative_to(SRC).as_posix()}:{i}: {line.strip()}")
        assert not offenders, "naive writes to a migrated column:\n" + "\n".join(offenders)

    def test_no_bare_fromisoformat_in_a_function_that_touches_these_columns(self):
        """FUNCTION scope, not line scope — the line-level check was unsound.

        W21's migration enumerated readers by grepping `fromisoformat` lines
        that NAMED entry_time/exit_time. Two readers select the column in SQL
        and then parse `_row[0]`, so the name never appears on the parse line:
        `orchestrator._post_stop_cooldown_until` (a live entry gate, which then
        raised TypeError into a bare except and failed CLOSED) and
        `master._mirror_content_age_hours` (which failed to None, read by its
        caller as a permanent MISSING alarm). Both were found by the suite,
        not by reading — so the guard now asks a question a grep cannot:
        does this FUNCTION touch these columns and parse a timestamp by hand?
        """
        allowed = {
            # Slices [:10] for a DATE, which no offset can perturb, and also
            # handles backtest rows whose exit_date is a plain date.
            "ait/dashboard/walkforward/export.py",
            # Parses trade.expiry (a date), not a migrated column.
            "ait/execution/portfolio.py",
            # The DuckDB mirror's columns are TIMESTAMP, and DuckDB DROPS the
            # offset on ingest (verified 2026-09-16: an aware and a naive
            # string land identically). Both sides of its subtraction stay
            # naive ET wall-clock, so it is consistent and deliberately
            # untouched.
            "ait/monitoring/duckdb_analytics.py",
            # The helper itself.
            "ait/utils/time.py",
        }
        offenders = []
        for path in SRC.rglob("*.py"):
            # POSIX form on every host: the allowlist and the prefix test below
            # are written with "/", and str() of a WindowsPath gives "\\".
            rel = path.relative_to(SRC).as_posix()
            # The backtest engine keeps its own entry_date/expiry_date corpus.
            if rel in allowed or rel.startswith("ait/backtesting/"):
                continue
            text = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            lines = text.split("\n")
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = "\n".join(lines[node.lineno - 1:node.end_lineno])
                if not any(c in body for c in MIGRATED_COLUMNS):
                    continue
                for n in ast.walk(node):
                    if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "fromisoformat":
                        offenders.append(f"{rel}:{n.lineno} in {node.name}()")
        assert not offenders, (
            "hand-rolled timestamp parse inside a function that touches a "
            "migrated column (use parse_trade_ts, or allowlist with a "
            "reason):\n" + "\n".join(offenders))

    def test_the_sweep_guard_is_not_silent(self):
        """The swallowed TypeError is what would hide a half-migration.

        Assert structurally that the sweep's except body logs rather than
        bare-`continue`ing, so a future tidy-up cannot quietly restore the
        silence.
        """
        tree = ast.parse((SRC / "ait/execution/reconciler.py").read_text(encoding="utf-8"))
        guards = [
            h for node in ast.walk(tree) if isinstance(node, ast.Try)
            for h in node.handlers
            if any(isinstance(n, ast.Name) and n.id == "parse_trade_ts"
                   for n in ast.walk(node))
        ]
        assert guards, "the stale-PENDING parse guard vanished"
        for h in guards:
            assert any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "warning"
                       for n in ast.walk(h)), (
                f"reconciler.py:{h.lineno} swallows a parse failure silently")


class TestTheCorpusIsUniformlyAware:
    """The post-condition that makes W21 a convention change, not a restatement."""

    def test_live_db_has_no_naive_trade_timestamps(self):
        # R26 (w21-backfill-was-never-applied-to-the-live-corpus): this asserts
        # on the LIVE state DB, which stays naive until the operator runs the
        # F5 restate with the bot stopped — on the Windows box 30 naive rows
        # plus the first aware fill of 2026-09-17. A test that is red until a
        # deploy step runs teaches everyone to read red as normal, so it only
        # runs when asked for as that deploy check.
        if os.environ.get("AIT_CHECK_LIVE_CORPUS") != "1":
            pytest.skip("deploy check: set AIT_CHECK_LIVE_CORPUS=1 after "
                        "scripts/restate_w21_tz_aware.py --apply (F5)")
        db = Path(__file__).resolve().parents[1] / "data" / "ait_state.db"
        if not db.exists():
            pytest.skip("no live state DB on this box")
        naive = []
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT trade_id, entry_time, exit_time FROM trades"):
                for col in MIGRATED_COLUMNS:
                    if row[col] and parse_trade_ts(row[col]) and \
                            datetime.fromisoformat(row[col]).tzinfo is None:
                        naive.append(f"{row['trade_id']}.{col}={row[col]}")
        finally:
            conn.close()
        assert not naive, (
            "naive rows remain — run scripts/restate_w21_tz_aware.py --apply:\n"
            + "\n".join(naive))


def test_et_zone_is_the_one_the_helper_uses():
    """Guards against a second, divergent ET definition creeping in."""
    assert ET is ZoneInfo("US/Eastern")

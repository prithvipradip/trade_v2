"""W38 (R26 F5 #2/#3) — the W21 restatement script, tested for the first time.

`scripts/restate_w21_tz_aware.py` rewrites timestamp values in the live track
record. R26 found it had ZERO executing coverage — `grep -rn "restate_w21"
tests/` matched only an assertion MESSAGE — and that its post-condition could
not fail:

    if row[col] and parse_trade_ts(row[col]).tzinfo is None:

`parse_trade_ts` localises naive input and converts aware input, so it can
never return naive. `bad` stayed 0, the success line always printed, and
`main()` returned 0 for a run that wrote nothing. The one automated statement
an operator would rely on to confirm the money-bearing corpus really migrated
was a constant.

Untested until now, and each one load-bearing: the HOST_MOVE abort (the only
thing standing between a convention change and a silent restatement of the
trading record), idempotency, the snapshot, and the per-row EST/EDT offset
derivation.

Every test drives the REAL `main()` against a throwaway DB. Nothing here reads
the live `data/ait_state.db`.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "restate_w21_tz_aware.py"


def _load():
    """Import the script by path — scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("restate_w21_tz_aware", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def script():
    return _load()


def _db(tmp_path: Path, rows: list[tuple[str, str, str | None]]) -> Path:
    """A minimal state DB carrying only what the script's TARGETS touch."""
    path = tmp_path / "ait_state.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE trades (trade_id TEXT PRIMARY KEY, "
                "entry_time TEXT, exit_time TEXT)")
    con.execute("CREATE TABLE open_positions (position_id TEXT PRIMARY KEY, "
                "entry_time TEXT)")
    con.execute("CREATE TABLE exit_outbox (trade_id TEXT PRIMARY KEY, "
                "exit_time TEXT, claimed_at TEXT)")
    con.executemany("INSERT INTO trades VALUES (?,?,?)", rows)
    con.commit()
    con.close()
    return path


def _entry_times(path: Path) -> dict[str, str]:
    con = sqlite3.connect(path)
    out = dict(con.execute("SELECT trade_id, entry_time FROM trades").fetchall())
    con.close()
    return out


class TestThePostConditionCanFail:
    """R26 `w21-restate-post-condition-is-unsatisfiable`."""

    def test_a_migration_that_wrote_nothing_is_reported_as_failure(
        self, script, tmp_path, capsys
    ):
        """THE defect, in one test: writes silently reverted must exit non-zero.

        A trigger reverts every UPDATE the script issues, so `--apply` runs,
        reports the writes it believes it made, and leaves the corpus naive.
        Pre-fix the post-check called parse_trade_ts (never naive), printed
        "every stored timestamp is timezone-aware" and returned 0.
        """
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857", None)])
        con = sqlite3.connect(path)
        con.execute(
            "CREATE TRIGGER revert AFTER UPDATE OF entry_time ON trades "
            "BEGIN UPDATE trades SET entry_time = OLD.entry_time "
            "WHERE trade_id = NEW.trade_id; END"
        )
        con.commit()
        con.close()

        rc = script.main(["--apply"], db_path=path)
        out = capsys.readouterr().out

        assert rc == 1, "a migration that wrote nothing reported success"
        assert "POST-CHECK FAILED" in out
        assert _entry_times(path)["T1"] == "2026-04-23T09:35:17.585857", (
            "premise broken: the trigger did not revert the write, so this "
            "test is no longer reproducing the silent-failure case"
        )

    def test_an_unparseable_row_is_reported_not_raised(self, script, tmp_path, capsys):
        """A crash in the post-check would abandon the run AFTER the writes."""
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857", None)])
        rc = script.main(["--apply"], db_path=path)
        assert rc == 0

        con = sqlite3.connect(path)
        con.execute("UPDATE trades SET exit_time = 'not-a-timestamp'")
        con.commit()
        con.close()

        rc = script.main(["--apply"], db_path=path)
        out = capsys.readouterr().out
        assert rc == 1
        assert "UNVOUCHABLE" in out and "unparseable" in out

    def test_a_real_migration_passes_the_post_check(self, script, tmp_path, capsys):
        """CONTROL: the post-check must still be satisfiable."""
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857", None)])
        rc = script.main(["--apply"], db_path=path)
        out = capsys.readouterr().out

        assert rc == 0
        assert "post-check: every stored timestamp is timezone-aware" in out
        assert _entry_times(path)["T1"].endswith("-04:00")


class TestTheHostMoveAbort:
    """The guard between a convention change and a silent restatement.

    A naive row written on a non-ET host is NOT ET, so relabelling it ET would
    move the instant. The script must refuse the whole run rather than migrate
    the vouchable rows around it.
    """

    def test_a_row_written_after_the_move_aborts_the_run(self, script, tmp_path, capsys):
        path = _db(tmp_path, [
            ("T1", "2026-04-23T09:35:17.585857", None),   # vouchable
            ("T2", "2026-09-15T10:00:00.000000", None),   # after HOST_MOVE
        ])
        before = path.read_bytes()

        rc = script.main(["--apply"], db_path=path)
        out = capsys.readouterr().out

        assert rc == 1
        assert "UNVOUCHABLE" in out and "written on/after the host move" in out
        assert path.read_bytes() == before, (
            "the abort still wrote to the database — the vouchable row was "
            "migrated around the unvouchable one"
        )

    def test_the_boundary_is_the_configured_date(self, script, tmp_path):
        """The guard must follow HOST_MOVE, not a date baked into the logic."""
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857", None)])
        assert script.main(["--apply"], db_path=path, host_move="2026-01-01") == 1, (
            "a row before the default boundary but after an earlier one was "
            "migrated anyway"
        )
        assert _entry_times(path)["T1"] == "2026-04-23T09:35:17.585857"


class TestTheWriteItself:

    def test_offsets_are_derived_per_row_not_hardcoded(self, script, tmp_path):
        """EST and EDT rows each get their own offset.

        A single hardcoded offset would silently move every row on the other
        side of the DST boundary by an hour.
        """
        path = _db(tmp_path, [
            ("WINTER", "2026-01-15T10:00:00.000000", None),
            ("SUMMER", "2026-07-15T10:00:00.000000", None),
        ])
        assert script.main(["--apply"], db_path=path) == 0

        times = _entry_times(path)
        assert times["WINTER"].endswith("-05:00"), times["WINTER"]
        assert times["SUMMER"].endswith("-04:00"), times["SUMMER"]

    def test_the_instant_is_preserved(self, script, tmp_path):
        """A convention change must not move the wall-clock reading."""
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857", None)])
        script.main(["--apply"], db_path=path)

        got = datetime.fromisoformat(_entry_times(path)["T1"])
        assert (got.year, got.month, got.day, got.hour, got.minute) == (
            2026, 4, 23, 9, 35)

    def test_a_dry_run_writes_nothing(self, script, tmp_path, capsys):
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857", None)])
        before = path.read_bytes()

        rc = script.main([], db_path=path)
        out = capsys.readouterr().out

        assert rc == 0
        assert "dry run" in out
        assert path.read_bytes() == before

    def test_it_is_idempotent(self, script, tmp_path, capsys):
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857", None)])
        assert script.main(["--apply"], db_path=path) == 0
        after_first = _entry_times(path)["T1"]
        capsys.readouterr()

        assert script.main(["--apply"], db_path=path) == 0
        out = capsys.readouterr().out
        assert "nothing to do" in out
        assert _entry_times(path)["T1"] == after_first


class TestTheSnapshot:

    def test_apply_snapshots_the_pre_migration_state(self, script, tmp_path):
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857", None)])
        assert script.main(["--apply"], db_path=path) == 0

        snaps = list((tmp_path / "backups").glob("ait_state.pre_w21_tz.*.db"))
        assert len(snaps) == 1, f"expected one snapshot, got {snaps}"
        assert _entry_times(snaps[0])["T1"] == "2026-04-23T09:35:17.585857", (
            "the snapshot holds the POST-migration state — it is useless as a "
            "rollback"
        )

    def test_the_nothing_to_do_path_takes_no_snapshot(self, script, tmp_path):
        """Documented so the live box's no-op is not mistaken for a backup.

        On a corpus that is already aware the script returns at "nothing to do"
        BEFORE the snapshot, so running it leaves no artifact at all.
        """
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857-04:00", None)])
        assert script.main(["--apply"], db_path=path) == 0
        assert not (tmp_path / "backups").exists()


class TestTheSeamDoesNotChangeOperatorBehaviour:
    """CONTROL: the defaults must still be the command-line behaviour."""

    def test_omitting_argv_still_reads_the_command_line(self, script, tmp_path, capsys):
        """No argv means sys.argv, which under pytest carries no --apply.

        Behavioural rather than introspective: the claim is that an operator
        running the script bare still gets a dry run, not that the signature
        has a particular shape.
        """
        path = _db(tmp_path, [("T1", "2026-04-23T09:35:17.585857", None)])
        before = path.read_bytes()

        assert script.main(db_path=path) == 0
        assert "dry run" in capsys.readouterr().out
        assert path.read_bytes() == before

    def test_the_default_database_is_the_live_one(self, script):
        assert script.DB.name == "ait_state.db"
        assert script.DB.parent.name == "data"

    def test_host_move_default_is_unchanged(self, script):
        assert script.HOST_MOVE == "2026-09-14"

"""W21 timestamp restatement: make stored trade timestamps timezone-aware.

`trades.entry_time` / `trades.exit_time` (and the `open_positions` /
`exit_outbox` mirrors) were written by naive `datetime.now()`, so their meaning
was whatever the host's clock happened to be. Every existing row was written on
the old ET host, so the corpus is uniformly ET *by accident* — see PLAN.md
"SESSION CLOSE" [X1]/[X2]/[X3].

While that remains true this is a CONVENTION CHANGE, not a restatement: each
naive value is already ET, so stamping the correct ET offset onto it preserves
the instant exactly. Once a row has been written on a non-ET host that is no
longer true and a pre/post boundary is needed instead. This script therefore
REFUSES to run if it finds a row it cannot vouch for — any naive row stamped
on or after HOST_MOVE below. (W38: this said "see --before"; there is no such
flag and never was. HOST_MOVE is a module constant.)

Offsets are derived per row from the ET zone, not hardcoded, so EST and EDT
rows are each stamped correctly.

Idempotent: a second run finds nothing to do.

Usage:
    python scripts/restate_w21_tz_aware.py           # dry run
    python scripts/restate_w21_tz_aware.py --apply   # snapshot + write
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "ait_state.db"
BACKUPS = ROOT / "data" / "backups"

sys.path.insert(0, str(ROOT / "src"))
from ait.utils.time import parse_trade_ts  # noqa: E402

# table -> (primary key column, timestamp columns)
TARGETS = {
    "trades": ("trade_id", ("entry_time", "exit_time")),
    "open_positions": ("position_id", ("entry_time",)),
    "exit_outbox": ("trade_id", ("exit_time", "claimed_at")),
}

# Every row written before the host moved to CEST is ET by construction. A
# naive row stamped after this is NOT vouchable and must be restated by hand
# with a pre/post boundary rather than silently relabelled ET.
HOST_MOVE = "2026-09-14"


def main(argv: list[str] | None = None, *, db_path: Path | None = None,
         host_move: str = HOST_MOVE) -> int:
    """W38: `argv`/`db_path`/`host_move` are injection seams for the tests.

    Defaults reproduce the command-line behaviour exactly, so `__main__` and
    every operator invocation are unchanged. They exist because R26 found this
    script had ZERO test coverage — including the HOST_MOVE abort, which is the
    only thing standing between a convention change and a silent restatement of
    the trading record — and `DB`/`sys.argv` offered nothing to inject.
    """
    apply = "--apply" in (sys.argv[1:] if argv is None else argv)
    db_path = db_path or DB
    backups = db_path.parent / "backups"
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    pending: list[tuple[str, str, str, str, str]] = []
    unvouchable: list[tuple[str, str, str]] = []

    try:
        for table, (pk, cols) in TARGETS.items():
            for row in db.execute(f"SELECT * FROM {table}"):  # noqa: S608
                for col in cols:
                    raw = row[col]
                    if not raw:
                        continue
                    try:
                        if datetime.fromisoformat(raw).tzinfo is not None:
                            continue  # already aware
                    except (ValueError, TypeError):
                        unvouchable.append((table, row[pk], f"{col}={raw!r} unparseable"))
                        continue
                    if raw >= host_move:
                        unvouchable.append(
                            (table, row[pk], f"{col}={raw} written on/after the host move"))
                        continue
                    pending.append((table, pk, row[pk], col, parse_trade_ts(raw).isoformat()))

        for table, key, why in unvouchable:
            print(f"UNVOUCHABLE  {table}[{key}]: {why}")
        if unvouchable:
            print(f"\nABORT: {len(unvouchable)} row(s) cannot be relabelled ET safely. "
                  "This is the restatement case, not the convention-change case — "
                  "handle those rows explicitly before rerunning.")
            return 1

        if not pending:
            print("nothing to do — every stored timestamp is already timezone-aware")
            return 0

        for table, _pk, key, col, new in pending:
            print(f"{'APPLY' if apply else 'DRY  '}  {table}[{key}].{col} -> {new}")
        print(f"\n{len(pending)} value(s) across "
              f"{len({t for t, *_ in pending})} table(s)")

        if not apply:
            print("\ndry run — rerun with --apply to write")
            return 0

        backups.mkdir(parents=True, exist_ok=True)
        snap = backups / f"ait_state.pre_w21_tz.{datetime.now():%Y%m%d_%H%M%S}.db"
        shutil.copy2(db_path, snap)
        print(f"\nsnapshot: {snap}")

        with db:
            for table, pk, key, col, new in pending:
                db.execute(
                    f"UPDATE {table} SET {col} = ? WHERE {pk} = ?",  # noqa: S608
                    (new, key))
        print(f"wrote {len(pending)} value(s)")

        # Post-condition: the corpus must now be uniformly aware, which is what
        # makes this a convention change with no pre/post boundary.
        #
        # W38 (R26 w21-restate-post-condition-is-unsatisfiable): this read
        # `parse_trade_ts(row[col]).tzinfo is None`, and parse_trade_ts can
        # NEVER return naive — it localises naive input and converts aware
        # input. So `bad` stayed 0, the success line always printed, and main()
        # returned 0 even for a run that wrote nothing: the one automated
        # statement an operator would rely on to confirm the money-bearing
        # corpus really migrated was a constant. Read the STORED text instead,
        # which is what the W21 test file already does.
        bad = 0
        for table, (pk, cols) in TARGETS.items():
            for row in db.execute(f"SELECT * FROM {table}"):  # noqa: S608
                for col in cols:
                    if not row[col]:
                        continue
                    try:
                        if datetime.fromisoformat(row[col]).tzinfo is None:
                            print(f"POST-CHECK FAILED {table}[{row[pk]}].{col}")
                            bad += 1
                    except (ValueError, TypeError):
                        # Reported, not raised: a crash here would abandon the
                        # run AFTER the writes, leaving no verdict at all.
                        print(f"POST-CHECK UNPARSEABLE {table}[{row[pk]}].{col}"
                              f" = {row[col]!r}")
                        bad += 1
        print("post-check: every stored timestamp is timezone-aware"
              if not bad else f"post-check: {bad} FAILURES")
        return 1 if bad else 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

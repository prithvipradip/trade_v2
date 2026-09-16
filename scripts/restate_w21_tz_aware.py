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
REFUSES to run if it finds a row it cannot vouch for (see --before).

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


def main() -> int:
    apply = "--apply" in sys.argv
    db = sqlite3.connect(DB)
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
                    if raw >= HOST_MOVE:
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

        BACKUPS.mkdir(parents=True, exist_ok=True)
        snap = BACKUPS / f"ait_state.pre_w21_tz.{datetime.now():%Y%m%d_%H%M%S}.db"
        shutil.copy2(DB, snap)
        print(f"\nsnapshot: {snap}")

        with db:
            for table, pk, key, col, new in pending:
                db.execute(
                    f"UPDATE {table} SET {col} = ? WHERE {pk} = ?",  # noqa: S608
                    (new, key))
        print(f"wrote {len(pending)} value(s)")

        # Post-condition: the corpus must now be uniformly aware, which is what
        # makes this a convention change with no pre/post boundary.
        bad = 0
        for table, (pk, cols) in TARGETS.items():
            for row in db.execute(f"SELECT * FROM {table}"):  # noqa: S608
                for col in cols:
                    if row[col] and parse_trade_ts(row[col]).tzinfo is None:
                        print(f"POST-CHECK FAILED {table}[{row[pk]}].{col}")
                        bad += 1
        print("post-check: every stored timestamp is timezone-aware"
              if not bad else f"post-check: {bad} FAILURES")
        return 1 if bad else 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

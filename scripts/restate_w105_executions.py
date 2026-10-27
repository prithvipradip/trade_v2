"""W105 executions restatement: tag the trade-less rows and fix five wrong fill times.

An audit of `data/ait_state.db` (36 executions, 2026-09-14..09-29) found:

* 31 rows with `trade_id = ''`. None is a bot trade: 8 are the operator's own TWS orders
  (order id 0, clientId 0) and 23 are the 2026-09-29 W53 probe (clientId 97, orderRef
  `W53PROBE-*`, refs taken from reports/w53_probe/2026-09-29/*.jsonl). Neither the client id
  nor the order reference was stored then, so a trade-less row could not be told from a bot
  fill we failed to attribute. This script tags them `manual` / `other` (new columns
  `client_id`, `order_ref`, `origin`). It never deletes a row and never gives one a trade id.
* 5 rows (the bot's only real fill, 2026-09-28) carry `exec_time` 15:54:15Z. ib_insync
  converted the gateway's zone-less time with the host's zone (CEST, 2 h); the true time is
  17:54:15Z (the restart replay in logs/ait.log.9:76141 logged it, and the bot's own entry
  time is 13:54:16 ET).

Refuses to run unless the database still matches what was audited (the same row counts per
order, the same current `exec_time` on the five rows). Idempotent: a second run finds nothing
to do. Run it with the bot stopped.

Usage:
    python scripts/restate_w105_executions.py           # dry run, opens the database read-only
    python scripts/restate_w105_executions.py --apply   # snapshot to data/backups/, then write
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "ait_state.db"

#: (order_id, perm_id) -> (origin, client_id, order_ref, rows audited).
#: Manual orders: TWS reports order id 0 and client id 0, with no order reference.
AUDITED: dict[tuple[int, int], tuple[str, int, str, int]] = {
    (0, 1178063164): ("manual", 0, "", 3),            # 2026-09-14 calendar spread (BAG + 2 legs)
    (0, 1779141601): ("manual", 0, "", 5),            # 2026-09-29 iron condor (BAG + 4 legs)
    (2276, 1779141650): ("other", 97, "W53PROBE-open-1790698798", 3),
    (2954, 1779141674): ("other", 97, "W53PROBE-flatten-1790699206", 3),
    (3758, 1779141699): ("other", 97, "W53PROBE-open-1790699955", 3),
    (4436, 1779141735): ("other", 97, "W53PROBE-flatten-1790700314", 3),
    (5240, 1779141795): ("other", 97, "W53PROBE-open-1790706776", 3),
    (5910, 1779141815): ("other", 97, "W53PROBE-roll-1790707000", 5),
    (5918, 1779141845): ("other", 97, "W53PROBE-flatten-1790707151", 3),
}
EXPECTED_TRADELESS = sum(v[3] for v in AUDITED.values())          # 31

WRONG_TIME = "2026-09-28T15:54:15+00:00"
RIGHT_TIME = "2026-09-28T17:54:15+00:00"
TIME_FIX_IDS = tuple(f"000182cd.6aba316c.0{i}.01" for i in range(1, 6))

NEW_COLUMNS = (
    "ALTER TABLE executions ADD COLUMN client_id INTEGER",
    "ALTER TABLE executions ADD COLUMN order_ref TEXT",
    "ALTER TABLE executions ADD COLUMN origin TEXT",
)


def _columns(db) -> set[str]:
    return {r[1] for r in db.execute("PRAGMA table_info(executions)")}


def main(argv: list[str] | None = None, *, db_path: Path | None = None) -> int:
    """`argv`/`db_path` are injection seams for the tests; defaults reproduce the CLI."""
    apply = "--apply" in (sys.argv[1:] if argv is None else argv)
    db_path = db_path or DB
    if apply:
        db = sqlite3.connect(db_path)
    else:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)      # a dry run cannot write
    db.row_factory = sqlite3.Row
    try:
        cols = _columns(db)
        have_new = {"client_id", "order_ref", "origin"} <= cols

        # ---- the trade-less rows must be exactly the audited ones -------------------
        tradeless = db.execute(
            "SELECT exec_id, order_id, perm_id" + (", origin" if "origin" in cols else "") +
            " FROM executions WHERE COALESCE(trade_id,'') = ''").fetchall()
        groups: dict[tuple[int, int], list[str]] = {}
        already = 0
        for r in tradeless:
            if "origin" in r.keys() and r["origin"] is not None:
                already += 1
                continue
            groups.setdefault((int(r["order_id"] or 0), int(r["perm_id"] or 0)), []).append(r["exec_id"])

        problems: list[str] = []
        for key, ids in groups.items():
            if key not in AUDITED:
                problems.append(f"unaudited trade-less rows for order/perm {key}: {len(ids)}")
            elif len(ids) != AUDITED[key][3]:
                problems.append(f"order/perm {key}: {len(ids)} rows, audited {AUDITED[key][3]}")
        if not already and groups and sum(len(v) for v in groups.values()) != EXPECTED_TRADELESS:
            problems.append(f"{sum(len(v) for v in groups.values())} trade-less rows, audited {EXPECTED_TRADELESS}")

        # ---- the five wrong fill times ----------------------------------------------
        time_rows = {r["exec_id"]: r["exec_time"] for r in db.execute(
            f"SELECT exec_id, exec_time FROM executions WHERE exec_id IN ({','.join('?' * len(TIME_FIX_IDS))})",
            TIME_FIX_IDS)}
        to_fix: list[str] = []
        for eid in TIME_FIX_IDS:
            cur = time_rows.get(eid)
            if cur is None:
                continue                                              # not in this database
            if cur == WRONG_TIME:
                to_fix.append(eid)
            elif cur != RIGHT_TIME:
                problems.append(f"{eid}: exec_time {cur!r} is neither the audited wrong nor the right value")

        for p in problems:
            print(f"UNVOUCHABLE  {p}")
        if problems:
            print("\nABORT: the database no longer matches the audit this script was written from. "
                  "Re-audit the rows before changing anything.")
            return 1

        n_tag = sum(len(v) for v in groups.values())
        if not n_tag and not to_fix:
            print("nothing to do: every audited row is already tagged and every fill time is right")
            return 0

        for key, ids in sorted(groups.items()):
            origin, client_id, ref, _n = AUDITED[key]
            print(f"{'APPLY' if apply else 'DRY  '}  tag {len(ids)} row(s) of order/perm {key}: "
                  f"origin={origin} client_id={client_id} order_ref={ref!r}")
        for eid in to_fix:
            print(f"{'APPLY' if apply else 'DRY  '}  {eid}.exec_time {WRONG_TIME} -> {RIGHT_TIME}")
        if not have_new:
            print(f"{'APPLY' if apply else 'DRY  '}  add the columns client_id, order_ref, origin")
        print(f"\n{n_tag} row(s) to tag, {len(to_fix)} fill time(s) to fix")

        if not apply:
            print("\ndry run: rerun with --apply to write (stop the bot first)")
            return 0

        backups = db_path.parent / "backups"
        backups.mkdir(parents=True, exist_ok=True)
        snap = backups / f"ait_state.pre_w105_executions.{datetime.now():%Y%m%d_%H%M%S}.db"
        shutil.copy2(db_path, snap)
        print(f"\nsnapshot: {snap}")

        with db:
            if not have_new:
                for ddl in NEW_COLUMNS:
                    db.execute(ddl)
            for key, ids in groups.items():
                origin, client_id, ref, _n = AUDITED[key]
                for eid in ids:
                    db.execute(
                        "UPDATE executions SET origin = ?, client_id = ?, order_ref = ? "
                        "WHERE exec_id = ? AND COALESCE(trade_id,'') = '' AND origin IS NULL",
                        (origin, client_id, ref, eid))
            for eid in to_fix:
                db.execute("UPDATE executions SET exec_time = ? WHERE exec_id = ? AND exec_time = ?",
                           (RIGHT_TIME, eid, WRONG_TIME))
        print(f"wrote {n_tag} tag(s) and {len(to_fix)} fill time(s)")

        left = db.execute("SELECT COUNT(*) FROM executions WHERE COALESCE(trade_id,'') = '' "
                          "AND origin IS NULL").fetchone()[0]
        if left:
            print(f"POST-CONDITION FAILED: {left} trade-less row(s) still untagged")
            return 1
        print("post-condition ok: every trade-less row is tagged")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())

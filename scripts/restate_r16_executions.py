"""R16 books restatement: heal the 18 legacy execution rows that poison the
go-live slippage gate.

Audit R16 (books dim, verified): `_sweep_executions` wrote the combo pricing
context (signal_price / live_mid / nbbo_spread) onto EVERY leg row of a BAG
order, and stored the BAG row's price with the broker's SIGNED convention
while `trades.entry_price` is unsigned. The referee's slippage query

    abs(price - live_mid) / entry_price

therefore compared a per-leg price against a whole-combo mid, reporting 49.9%
(median 65.1%) slippage where the SPY condor's true figure is ~1.62%.

The writer is fixed (leg rows get no combo context; the BAG row stores a
magnitude), but `record_execution`'s upsert never overwrites `price` and only
overwrites `live_mid` when the incoming value is > 0, so existing rows cannot
self-heal — the gate keeps reporting a BREAK until ~20 new fills roll them out
of the referee's trailing window.

This script fixes them in place:
  * non-BAG (per-leg) rows  -> zero signal_price / live_mid / nbbo_spread
  * BAG (combo) rows        -> price = abs(price)

Both are lossless for P&L: no consumer reconstructs P&L from these columns
(realized P&L comes from commissionReports and the trades table), and the leg
rows' raw broker `price` is left untouched.

Usage:
    python scripts/restate_r16_executions.py           # dry run (default)
    python scripts/restate_r16_executions.py --apply   # write + provenance
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "ait_state.db"

# Single authority: the same generic combo conId the fixed writer uses. Do NOT
# re-declare it here — a drifting copy is exactly the class of bug being
# restated (verified against the live ledger: the BAG row for the 08-10 QQQ
# condor is con_id 28812380, price -5.44 = the combo credit, while the leg
# rows carry real 9-digit option conIds).
sys.path.insert(0, str(ROOT / "src"))
from ait.execution.executor import BAG_CON_ID  # noqa: E402


def main() -> int:
    apply = "--apply" in sys.argv
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    try:
        legs = con.execute(
            "SELECT COUNT(*) c FROM executions WHERE con_id != ? "
            "AND (COALESCE(live_mid,0) != 0 OR COALESCE(signal_price,0) != 0 "
            "     OR COALESCE(nbbo_spread,0) != 0)", (BAG_CON_ID,)).fetchone()["c"]
        bags = con.execute(
            "SELECT COUNT(*) c FROM executions WHERE con_id = ? AND price < 0",
            (BAG_CON_ID,)).fetchone()["c"]
        print(f"leg rows carrying combo context : {legs}")
        print(f"BAG rows with signed price      : {bags}")

        sample = con.execute(
            "SELECT exec_id, trade_id, con_id, price, live_mid, signal_price "
            "FROM executions WHERE COALESCE(live_mid,0) != 0 "
            "ORDER BY exec_time DESC LIMIT 5").fetchall()
        print("\nsample rows currently feeding the gate:")
        for r in sample:
            print(f"  {r['trade_id']} con={r['con_id']} price={r['price']} "
                  f"live_mid={r['live_mid']} signal={r['signal_price']}")

        if not legs and not bags:
            print("\nnothing to restate")
            return 0
        if not apply:
            print(f"\nDRY RUN: would clear context on {legs} leg rows and "
                  f"abs() price on {bags} BAG rows. Re-run with --apply.")
            return 0

        con.execute(
            "UPDATE executions SET live_mid = 0, signal_price = 0, "
            "nbbo_spread = 0 WHERE con_id != ? "
            "AND (COALESCE(live_mid,0) != 0 OR COALESCE(signal_price,0) != 0 "
            "     OR COALESCE(nbbo_spread,0) != 0)", (BAG_CON_ID,))
        con.execute(
            "UPDATE executions SET price = ABS(price) WHERE con_id = ? "
            "AND price < 0", (BAG_CON_ID,))
        con.execute(
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) "
            "VALUES ('r16_executions_restatement', ?, ?)",
            (f"cleared combo context on {legs} leg rows; abs() price on "
             f"{bags} BAG rows (R16 slippage-gate semantics)",
             datetime.now().isoformat()))
        con.commit()
        print(f"\nAPPLIED: {legs} leg rows cleared, {bags} BAG prices "
              f"normalized. Provenance in bot_state.")
        print("Re-run scripts/shadow_referee.py to confirm the gate clears.")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())

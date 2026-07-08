#!/usr/bin/env python
"""One-off migration: correct historical realized_pnl in SQLite + DuckDB.

Background (2026-06-11 audit): three accounting bugs corrupted trades.realized_pnl:
1. Sign inversion — 2-leg debit positions (long/event straddles, debit spreads)
   were classified contract_type="spread" and run through the credit formula
   (entry - exit) instead of (exit - entry).
2. Reconciler closes stamped exit_price=0.0 and assumed "expired worthless"
   with a credit list missing short_strangle/covered_call, recording expired
   strangle wins as full losses.
3. Real exits whose combo fill price came back 0/None persisted the
   portfolio's estimated_pnl, which was computed from the UNDERLYING STOCK
   price vs the option premium — fantasy numbers like +$67k on a $2.4k straddle.

Correction rules (deterministic, per closed trade):
  A) exit_price == 0 and exit_reason_detailed == ''        -> reconciler close
       expired by exit date -> worthless assumption, CORRECT sign by
       CREDIT_STRATEGIES, flag 'migrated_expired_worthless_estimate'
       not expired           -> unrecoverable, pnl=0, flag 'migrated_unknown_exit'
  B) exit_price == 0 and exit_reason_detailed != ''        -> real exit, fill
       price never recovered -> unrecoverable, pnl=0,
       flag '<original>|migrated_unrecoverable_exit'
  C) exit_price > 0                                        -> recompute with the
       corrected credit/debit formula, flag '<original>|migrated_recomputed'

Both DBs are backed up next to the originals before any write.
Run with --dry-run to preview without writing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ait.strategies.base import CREDIT_STRATEGIES  # noqa: E402

SQLITE_DB = ROOT / "data" / "ait_state.db"
DUCK_DB = ROOT / "data" / "ait_analytics.duckdb"


def _legs_per_side(contract_type: str, strategy: str) -> int:
    if contract_type == "iron_condor":
        return 4
    if contract_type == "spread" or strategy in (
        "long_straddle", "event_straddle", "short_strangle", "calendar_spread",
    ):
        return 2
    return 1


def _commission(contract_type: str, strategy: str, qty: int) -> float:
    return 0.65 * _legs_per_side(contract_type, strategy) * qty * 2


def corrected_pnl(row: dict) -> tuple[float, str]:
    """Return (corrected_realized_pnl, new_exit_reason_detailed)."""
    entry = row["entry_price"] or 0.0
    exit_p = row["exit_price"] or 0.0
    qty = row["quantity"] or 1
    strategy = row["strategy"] or ""
    ctype = row["contract_type"] or ""
    reason = row["exit_reason_detailed"] or ""
    mult = 1 if ctype == "stock" else 100
    comm = _commission(ctype, strategy, qty)
    is_credit = strategy in CREDIT_STRATEGIES

    if exit_p > 0:  # rule C — real exit price on record, just fix the formula
        if ctype == "stock":
            pnl = (exit_p - entry) * qty
        elif is_credit:
            pnl = (entry - exit_p) * mult * qty
        else:
            pnl = (exit_p - entry) * mult * qty
        return round(pnl - comm, 2), f"{reason}|migrated_recomputed".lstrip("|")

    if reason:  # rule B — real exit, fill price never recovered
        return 0.0, f"{reason}|migrated_unrecoverable_exit"

    # rule A — reconciler close at stamped 0.0
    expired = False
    if row["expiry"] and row["exit_time"]:
        try:
            expired = (
                datetime.fromisoformat(row["expiry"]).date()
                <= datetime.fromisoformat(row["exit_time"]).date()
            )
        except ValueError:
            pass
    # If still unexpired as of today, the position truly vanished mid-life
    if not expired and row["expiry"]:
        try:
            expired = datetime.fromisoformat(row["expiry"]).date() <= datetime.now().date()
        except ValueError:
            pass

    if expired and entry > 0:
        signed = entry if is_credit else -entry
        return round(signed * mult * qty - comm, 2), "migrated_expired_worthless_estimate"
    return 0.0, "migrated_unknown_exit"


def _already_migrated(conn) -> bool:
    """Deep-audit BT-H1: this script is NOT idempotent — a second run pushes
    already-migrated expired-worthless wins through Rule B and zeroes them,
    then propagates the zeros to DuckDB. Refuse to run twice."""
    row = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE exit_reason_detailed LIKE '%migrated%'"
    ).fetchone()
    return bool(row and row[0] > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        shutil.copy2(SQLITE_DB, SQLITE_DB.with_suffix(f".db.bak_{stamp}"))
        if DUCK_DB.exists():
            shutil.copy2(DUCK_DB, DUCK_DB.with_suffix(f".duckdb.bak_{stamp}"))
        print(f"Backups written with suffix .bak_{stamp}")

    con = sqlite3.connect(SQLITE_DB)
    if _already_migrated(conn):
        print("REFUSING TO RUN: migration flags already present in trades "
              "(re-running would zero migrated wins — deep-audit BT-H1).")
        return
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM trades WHERE status='closed' ORDER BY entry_time"
    )]

    print(f"\n{'entry':<12} {'sym':<5} {'strategy':<18} {'old_pnl':>10} {'new_pnl':>10}  rule")
    by_strategy_old: dict[str, float] = {}
    by_strategy_new: dict[str, float] = {}
    updates = []
    for row in rows:
        new_pnl, new_reason = corrected_pnl(row)
        old_pnl = row["realized_pnl"] or 0.0
        by_strategy_old[row["strategy"]] = by_strategy_old.get(row["strategy"], 0) + old_pnl
        by_strategy_new[row["strategy"]] = by_strategy_new.get(row["strategy"], 0) + new_pnl
        if abs(new_pnl - old_pnl) > 0.01 or new_reason != (row["exit_reason_detailed"] or ""):
            updates.append((new_pnl, new_reason, row["trade_id"]))
            print(f"{row['entry_time'][:10]:<12} {row['symbol']:<5} {row['strategy']:<18} "
                  f"{old_pnl:>10.0f} {new_pnl:>10.0f}  {new_reason.split('|')[-1]}")

    print(f"\n=== totals by strategy (old -> new) ===")
    total_old = total_new = 0.0
    for strat in sorted(by_strategy_old):
        o, n = by_strategy_old[strat], by_strategy_new.get(strat, 0)
        total_old += o
        total_new += n
        print(f"  {strat:<20} ${o:>12,.0f} -> ${n:>12,.0f}")
    print(f"  {'TOTAL':<20} ${total_old:>12,.0f} -> ${total_new:>12,.0f}")
    print(f"\n{len(updates)} of {len(rows)} closed trades need correction")

    if args.dry_run:
        print("\nDRY RUN — nothing written.")
        return

    with con:
        con.executemany(
            "UPDATE trades SET realized_pnl = ?, exit_reason_detailed = ? WHERE trade_id = ?",
            updates,
        )
    print("SQLite updated.")

    # Rebuild the DuckDB copy from corrected SQLite (it held a divergent
    # 210-row copy; full rebuild from source of truth)
    try:
        from ait.monitoring.duckdb_analytics import DuckDBAnalytics
        duck = DuckDBAnalytics()
        with duck._get_conn() as dc:
            dc.execute("DELETE FROM trades")
        fresh = [dict(r) for r in con.execute("SELECT * FROM trades WHERE status='closed'")]
        for row in fresh:
            duck.ingest_trade(row)
        with duck._get_conn() as dc:
            n = dc.execute("SELECT COUNT(*), SUM(realized_pnl) FROM trades").fetchone()
        print(f"DuckDB rebuilt: {n[0]} rows, total realized_pnl ${n[1]:,.2f}")
    except Exception as e:  # noqa: BLE001
        print(f"DuckDB rebuild FAILED ({e}) — SQLite is corrected; rerun DuckDB step manually.")

    con.close()
    print("Done.")


if __name__ == "__main__":
    main()

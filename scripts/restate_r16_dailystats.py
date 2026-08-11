"""R16 books restatement: backfill the missing 2026-07-22 daily_stats row.

Audit R16 (books dim, verified): the restated straddle loss
(T-20260707-093745-d65049, realized -575.33, closed 07-22 via manual flatten)
never reached daily_stats — restate_d1.py recomputed with a bare UPDATE that
silently no-ops when no row exists for the date. Every consumer summing
daily_stats overstates cumulative P&L by exactly $575.33.

Usage:
    python scripts/restate_r16_dailystats.py           # dry run
    python scripts/restate_r16_dailystats.py --apply   # write + provenance
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "ait_state.db"
DATE = "2026-07-22"
TRADE = "T-20260707-093745-d65049"


def main() -> int:
    apply = "--apply" in sys.argv
    db = sqlite3.connect(DB)
    try:
        row = db.execute(
            "SELECT * FROM daily_stats WHERE date=?", (DATE,)).fetchone()
        pnl = db.execute(
            "SELECT realized_pnl FROM trades WHERE trade_id=?",
            (TRADE,)).fetchone()
        if pnl is None:
            print(f"ABORT: {TRADE} not found")
            return 1
        pnl = float(pnl[0])
        print(f"trade {TRADE}: realized {pnl}; existing {DATE} row: {row}")
        if row is not None:
            print("row already present — nothing to do")
            return 0
        # R17: was hardcoded won=0/lost=1 regardless of the fetched pnl's
        # sign. Harmless for this specific trade (a confirmed loss), but
        # the label should be derived, not assumed.
        won, lost = (1, 0) if pnl >= 0 else (0, 1)
        if not apply:
            print(f"DRY RUN: would INSERT daily_stats({DATE}, "
                  f"won={won}, lost={lost}, total_pnl={pnl})")
            return 0
        db.execute(
            "INSERT OR REPLACE INTO daily_stats (date, trades_taken, "
            "trades_won, trades_lost, total_pnl, max_drawdown, "
            "day_trades_count, circuit_breaker_triggered) "
            "VALUES (?, 0, ?, ?, ?, 0.0, 0, 0)", (DATE, won, lost, pnl))
        db.execute(
            "INSERT OR REPLACE INTO bot_state (key, value, updated_at) "
            "VALUES ('r16_dailystats_0722_backfill', ?, ?)",
            (f"{TRADE} realized {pnl} closed {DATE}; restate_d1 UPDATE "
             f"no-oped on missing row (R16 books finding)",
             datetime.now().isoformat()))
        db.commit()
        total = db.execute(
            "SELECT ROUND(SUM(total_pnl),2) FROM daily_stats").fetchone()[0]
        print(f"APPLIED. cumulative daily_stats total_pnl now: {total}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())

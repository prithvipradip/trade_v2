"""D1 RESTATEMENT (decision 2026-07-16) — one-time scoreboard correction.

Replaces booked exit P&L / commissions with the broker's own numbers, using
EXACTLY the shadow referee's math (scripts/shadow_referee.py):
  - true realized P&L  = sum of IBKR per-leg realizedPNL over CLOSING fill
    groups (groups where any leg carries a nonzero realizedPNL);
  - true commission    = sum of commissions over ALL fill groups for the
    trade (phantom re-closes cost real money too);
  - exit_price         = |combo price| of the FIRST closing fill group.

Also backfills trades.capital_at_risk where derivable (D2 dependency):
  - defined-risk multi-leg (condor/spread): (max wing width - |entry|) * 100 * qty
  - debit structures (straddle/long option): |entry| * 100 * qty
  - short strangle (undefined risk): 3x credit convention (strategies'
    documented floor) — labeled as an estimate in the log.

Writes provenance to bot_state key 'd1_restatement' (old->new JSON per trade).
Mirrors the same UPDATEs into the DuckDB analytics copy when reachable.

USAGE:
  python scripts/restate_d1.py            # DRY RUN — prints, changes nothing
  python scripts/restate_d1.py --apply    # applies (take a backup first;
                                          #  run with the bot STOPPED)
"""
import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "ait_state.db"
MIRROR = ROOT / "data" / "ait_analytics.duckdb"

BAG_CONID = 28812380
# W3/string-contracts-4: membership from the ONE authority.
from ait.reporting.go_live import NOT_REAL_CLOSE_PATTERNS as NOT_REAL
PNL_EPS = 0.005  # below this, booked == broker; skip


def real_closed_trades(cur):
    q = ("SELECT * FROM trades WHERE status='closed' "
         + " ".join(f"AND COALESCE(exit_reason_detailed,'') NOT LIKE '{p}'"
                    for p in NOT_REAL)
         + " ORDER BY COALESCE(exit_time, entry_time)")
    return [dict(r) for r in cur.execute(q)]


def fill_groups(cur, trade_id):
    rows = [dict(r) for r in cur.execute(
        "SELECT * FROM executions WHERE trade_id=? ORDER BY exec_time, exec_id",
        (trade_id,)) if r["con_id"] != BAG_CONID]
    groups = {}
    for r in rows:
        seg = r["exec_id"].split(".")
        key = seg[1] if len(seg) > 1 else r["exec_id"]
        groups.setdefault(key, []).append(r)
    out = []
    for key, legs in sorted(groups.items(), key=lambda kv: kv[1][0]["exec_time"]):
        out.append({
            "key": key,
            "closing": any(abs(l["realized_pnl"] or 0) > 1e-6 for l in legs),
            "combo_price": round(sum(
                l["price"] if l["side"] == "BOT" else -l["price"]
                for l in legs), 4),
            "commission": round(sum(l["commission"] or 0 for l in legs), 6),
            "broker_realized": round(sum(l["realized_pnl"] or 0 for l in legs), 2),
        })
    return out


def derive_car(trade) -> tuple[float, str]:
    """capital_at_risk from the trade's own structure. Returns (car, how)."""
    try:
        legs = json.loads(trade["legs"]) if trade["legs"] else []
    except (ValueError, TypeError):
        legs = []
    qty = max(1, int(trade["quantity"] or 1))
    entry = abs(trade["entry_price"] or 0)
    strat = trade["strategy"] or ""

    by_right = {}
    for lg in legs:
        try:
            by_right.setdefault(str(lg["right"]).upper()[:1], []).append(float(lg["strike"]))
        except (KeyError, TypeError, ValueError):
            pass
    widths = [abs(s[0] - s[1]) for s in by_right.values() if len(s) == 2]

    if strat in ("iron_condor", "bull_call_spread", "bear_put_spread") and widths:
        # defined-risk: width - credit (credit strategies) or debit paid
        if strat == "iron_condor":
            return round((max(widths) - entry) * 100 * qty, 2), "wings-credit"
        return round(entry * 100 * qty, 2), "net-debit"
    if strat in ("long_straddle", "long_call", "long_put", "event_straddle"):
        return round(entry * 100 * qty, 2), "premium-paid"
    if strat == "short_strangle":
        return round(entry * 3 * 100 * qty, 2), "3x-credit (estimate)"
    return 0.0, "underivable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    mode = "rw" if a.apply else "ro"
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode={mode}", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    closed = real_closed_trades(cur)
    pnl_changes, car_changes, provenance = [], [], {}

    for t in closed:
        gs = fill_groups(cur, t["trade_id"])
        closing = [g for g in gs if g["closing"]]
        if closing:  # verifiable against the broker
            true_pnl = round(sum(g["broker_realized"] for g in closing), 2)
            true_comm = round(sum(g["commission"] for g in gs), 2)
            true_exit = abs(closing[0]["combo_price"])
            old_pnl = round(t["realized_pnl"] or 0, 2)
            old_comm = round(t["commission"] or 0, 2)
            if abs(true_pnl - old_pnl) > PNL_EPS or abs(true_comm - old_comm) > PNL_EPS:
                pnl_changes.append((t, old_pnl, true_pnl, old_comm, true_comm, true_exit))
                provenance[t["trade_id"]] = {
                    "pnl": [old_pnl, true_pnl], "comm": [old_comm, true_comm],
                    "exit_price": [t["exit_price"], true_exit],
                }
        car = t["capital_at_risk"] or 0
        if car <= 0:
            new_car, how = derive_car(t)
            if new_car > 0:
                car_changes.append((t, new_car, how))
                provenance.setdefault(t["trade_id"], {})["car"] = [car, new_car, how]

    # ---- report
    print(f"D1 RESTATEMENT {'APPLY' if a.apply else 'DRY RUN'} — {len(closed)} real closes")
    print("\nP&L / commission corrections (booked -> broker):")
    for t, op, np_, oc, nc, ex in pnl_changes:
        print(f"  {t['symbol']:5} {t['trade_id'][-10:]}  pnl {op:+8.2f} -> {np_:+8.2f}"
              f"   comm {oc:5.2f} -> {nc:5.2f}   exit_price -> {ex}")
    if not pnl_changes:
        print("  (none)")

    print("\ncapital_at_risk backfill:")
    for t, car, how in car_changes:
        print(f"  {t['symbol']:5} {t['trade_id'][-10:]}  {t['strategy']:16} -> ${car:,.2f} ({how})")
    if not car_changes:
        print("  (none)")

    # ---- aggregates before/after
    def agg(pnls):
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        return (len(pnls), sum(1 for p in pnls if p > 0), round(sum(pnls), 2),
                round(gp / gl, 3) if gl else float("inf"))

    before = agg([t["realized_pnl"] or 0 for t in closed])
    after_map = {t["trade_id"]: (t["realized_pnl"] or 0) for t in closed}
    for t, _, np_, *_ in pnl_changes:
        after_map[t["trade_id"]] = np_
    after = agg(list(after_map.values()))
    print(f"\naggregates: BEFORE n={before[0]} wins={before[1]} total={before[2]:+.2f} PF={before[3]}")
    print(f"            AFTER  n={after[0]} wins={after[1]} total={after[2]:+.2f} PF={after[3]}")

    if not a.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply (bot stopped, backup taken).")
        con.close()
        return

    # ---- apply
    for t, _, np_, _, nc, ex in pnl_changes:
        cur.execute("UPDATE trades SET realized_pnl=?, commission=?, exit_price=? "
                    "WHERE trade_id=?", (np_, nc, ex, t["trade_id"]))
    for t, car, _ in car_changes:
        cur.execute("UPDATE trades SET capital_at_risk=? WHERE trade_id=?",
                    (car, t["trade_id"]))
    # recompute daily_stats for affected close dates from restated trades
    dates = sorted({(t["exit_time"] or "")[:10] for t, *_ in pnl_changes if t["exit_time"]})
    for d in dates:
        row = cur.execute(
            "SELECT COALESCE(SUM(realized_pnl),0) p, "
            "SUM(CASE WHEN realized_pnl>0 THEN 1 ELSE 0 END) w, "
            "SUM(CASE WHEN realized_pnl<=0 THEN 1 ELSE 0 END) l "
            "FROM trades WHERE status='closed' AND exit_time LIKE ?||'%' "
            + " ".join(f"AND COALESCE(exit_reason_detailed,'') NOT LIKE '{p}'"
                       for p in NOT_REAL), (d,)).fetchone()
        # R16: bare UPDATE silently no-ops when the date row doesn't exist —
        # that's how the 07-22 straddle loss vanished from cumulative daily
        # P&L. INSERT OR REPLACE like state.update_daily_stats does.
        cur.execute(
            "INSERT OR REPLACE INTO daily_stats (date, trades_taken, "
            "trades_won, trades_lost, total_pnl, max_drawdown, "
            "day_trades_count, circuit_breaker_triggered) VALUES (?, "
            "COALESCE((SELECT trades_taken FROM daily_stats WHERE date=?),0), "
            "?, ?, ?, "
            "COALESCE((SELECT max_drawdown FROM daily_stats WHERE date=?),0), "
            "COALESCE((SELECT day_trades_count FROM daily_stats WHERE date=?),0), "
            "COALESCE((SELECT circuit_breaker_triggered FROM daily_stats WHERE date=?),0))",
            (d, d, row["w"], row["l"], row["p"], d, d, d))
        print(f"daily_stats[{d}] recomputed: pnl={row['p']:+.2f} W{row['w']}/L{row['l']}")
    cur.execute(
        "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?,?,?)",
        ("d1_restatement", json.dumps(provenance), datetime.now().isoformat()))
    con.commit()
    con.close()
    print("\nSQLite applied + provenance stored under bot_state['d1_restatement'].")

    # ---- mirror (best effort; bot must be stopped or this will be locked)
    try:
        import duckdb
        m = duckdb.connect(str(MIRROR))
        for t, _, np_, _, nc, ex in pnl_changes:
            m.execute("UPDATE trades SET realized_pnl=?, commission=?, exit_price=? "
                      "WHERE trade_id=?", [np_, nc, ex, t["trade_id"]])
        for t, car, _ in car_changes:
            m.execute("UPDATE trades SET capital_at_risk=? WHERE trade_id=?",
                      [car, t["trade_id"]])
        m.close()
        print("DuckDB mirror updated to match.")
    except Exception as e:  # noqa: BLE001
        print(f"MIRROR NOT UPDATED ({e}) — it will re-sync at next ingest, or "
              f"referee check [9] flags it.")


if __name__ == "__main__":
    main()

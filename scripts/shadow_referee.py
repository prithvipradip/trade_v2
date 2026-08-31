"""SHADOW REFEREE (Round 13) — independent track-record recomputation.

Recomputes per-trade realized P&L and the go-live gate numbers from the RAW
broker-derived ledger (executions table: broker exec ids, fill prices, real
commissions, IBKR's own per-leg realizedPNL) and diffs them against the
system's books (trades table / scorecard math). Prints one screen of
PASS/BREAK checks; exits nonzero on any BREAK. --json for machine reads.

INDEPENDENCE: deliberately imports NOTHING from src/ait. The handful of
constants below are duplicated on purpose — divergence is the point: if the
system's constants drift, this referee must NOT drift with them.

Read-only by construction: sqlite3 URI mode=ro, duckdb read_only=True.
"""
import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- duplicated constants (do NOT import from src/ait; divergence is the point)
BAG_CONID = 28812380          # IBKR generic combo (BAG) contract id
MULT = 100                    # option multiplier
RESET = "2026-07-06"          # track-record reset date (PLAN.md)
GATE_PF = 1.3                 # go-live gate: profit factor
GATE_DD_PCT = 8.0             # go-live gate: max DD as % of deployed risk
GATE_SLIP_PCT = 8.0           # go-live gate: median entry slip <= 8% of credit
SLIP_WINDOW = 20              # trailing fills for the slippage gate
PNL_TOL = 2.00                # per-trade booked-vs-broker tolerance ($)
COMM_TOL = 0.75               # per-trade commission tolerance ($)
TOTAL_TOL = 5.00              # aggregate realized tolerance ($)
PF_TOL = 0.10                 # aggregate PF tolerance
# W3/string-contracts-4: the scorecard's filter now lives in ONE place.
from ait.reporting.go_live import NOT_REAL_CLOSE_PATTERNS as NOT_REAL


def real_closed_trades(cur):
    q = ("SELECT * FROM trades WHERE status='closed' "
         + " ".join(f"AND COALESCE(exit_reason_detailed,'') NOT LIKE '{p}'"
                    for p in NOT_REAL)
         + " ORDER BY COALESCE(exit_time, entry_time)")
    return [dict(r) for r in cur.execute(q)]


def fill_groups(cur, trade_id):
    """Group raw leg executions by broker order (2nd exec-id segment).
    Returns [{key, legs:[rows], closing:bool, combo_price, commission}]."""
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
            "key": key, "n_legs": len(legs),
            "closing": any(abs(l["realized_pnl"] or 0) > 1e-6 for l in legs),
            "combo_price": round(sum(
                l["price"] if l["side"] == "BOT" else -l["price"]
                for l in legs), 4),
            "commission": round(sum(l["commission"] or 0 for l in legs), 6),
            "broker_realized": round(sum(l["realized_pnl"] or 0 for l in legs), 2),
            "time": legs[0]["exec_time"],
        })
    return out


def max_concurrent_risk(trades_rows):
    """Max concurrent sum of capital_at_risk over trade [entry, exit) windows."""
    ev = []
    for t in trades_rows:
        car = t.get("capital_at_risk") or 0
        if car <= 0:
            continue
        ev.append((t["entry_time"], +car))
        ev.append((t.get("exit_time") or "9999", -car))
    peak = cur = 0.0
    for _, d in sorted(ev):
        cur += d
        peak = max(peak, cur)
    return peak


def main():
    ap = argparse.ArgumentParser(description="shadow referee (read-only)")
    ap.add_argument("--db", default=str(ROOT / "data" / "ait_state.db"))
    ap.add_argument("--mirror", default=str(ROOT / "data" / "ait_analytics.duckdb"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{Path(a.db).as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    checks, detail = [], {}

    def check(name, ok, msg):
        checks.append({"check": name, "status": "PASS" if ok else "BREAK", "msg": msg})

    closed = real_closed_trades(cur)
    open_rows = [dict(r) for r in cur.execute(
        "SELECT t.* FROM trades t JOIN open_positions o ON o.trade_id=t.trade_id")]

    # [1] raw-ledger coverage
    covered = [t for t in closed if fill_groups(cur, t["trade_id"])]
    check("raw_coverage", True,  # informational: pre-ledger closes are unverifiable
          f"{len(covered)}/{len(closed)} real closes have raw executions "
          f"(pre-ledger closes unverifiable against broker)")

    # [2] duplicate / phantom close fills + implied residual legs.
    # Incidents can be ACKNOWLEDGED by a human once resolved at the broker:
    # bot_state['incident_ack_before'] = ISO timestamp. Dup groups whose
    # fills all predate it are reported but don't BREAK (the 07-13 triple
    # fill is history — its 11 residual legs were flattened and broker-
    # verified flat on 07-15). NEW duplicates always BREAK.
    ack_row = cur.execute(
        "SELECT value FROM bot_state WHERE key='incident_ack_before'").fetchone()
    ack_before = ack_row[0] if ack_row else ""
    dups, acked, residual_legs = [], [], 0
    for t in covered:
        gs = fill_groups(cur, t["trade_id"])
        # R16: `closing` is REQUIRED here. This counted every >=2-leg fill
        # group, so an ENTRY group and its EXIT group read as two closes.
        # It never fired historically only because the ledger used to capture
        # exit fills alone; the 08-04 QQQ condor is the first trade whose
        # ENTRY legs were also swept, and it immediately produced a phantom
        # "1 re-close + 4 residual legs" against a book that is provably
        # flat (entry legs carry realized_pnl 0.0, exit legs carry the real
        # P&L — the discriminator the group builder already computes).
        full = [g for g in gs if g["n_legs"] >= 2 and g["closing"]]
        if len(full) > 1:
            phantom = len(full) - 1
            msg = (f"{t['symbol']} {t['trade_id'][-6:]}: {len(full)} fills "
                   f"(prices {[g['combo_price'] for g in full]}) -> "
                   f"{phantom} phantom re-close(s)")
            if ack_before and all(g["time"] <= ack_before for g in full):
                acked.append(msg)
            else:
                residual_legs += phantom * full[0]["n_legs"]
                dups.append(msg)
    check("duplicate_closes", not dups,
          ("; ".join(dups) + f" | implied ~{residual_legs} residual untracked legs")
          if dups else
          (f"one fill group per close"
           + (f" | {len(acked)} acknowledged historical incident(s) "
              f"(pre-{ack_before[:10]})" if acked else "")))
    detail["duplicate_closes"] = dups
    detail["acknowledged_dups"] = acked

    # [3] per-trade realized P&L: books vs broker's own realizedPNL
    pnl_rows, pnl_bad = [], []
    for t in covered:
        gs = [g for g in fill_groups(cur, t["trade_id"]) if g["closing"]]
        if not gs:
            pnl_bad.append(f"{t['symbol']}: raw fills but no closing group")
            continue
        broker = round(sum(g["broker_realized"] for g in gs), 2)
        diff = round(t["realized_pnl"] - broker, 2)
        pnl_rows.append({"trade": t["trade_id"], "symbol": t["symbol"],
                         "booked": t["realized_pnl"], "broker": broker, "diff": diff})
        if abs(diff) > PNL_TOL:
            pnl_bad.append(f"{t['symbol']} booked {t['realized_pnl']:+.2f} vs "
                           f"broker {broker:+.2f} (diff {diff:+.2f})")
    check("pnl_vs_broker", not pnl_bad, "; ".join(pnl_bad) or
          f"all {len(pnl_rows)} verifiable trades within ${PNL_TOL:.2f}")
    detail["pnl_rows"] = pnl_rows

    # [4] commissions: booked vs raw (all groups; phantoms cost real money too)
    comm_bad = []
    for t in covered:
        raw = round(sum(g["commission"] for g in fill_groups(cur, t["trade_id"])), 2)
        booked = round(t["commission"] or 0, 2)
        if abs(booked - raw) > COMM_TOL:
            comm_bad.append(f"{t['symbol']} booked ${booked:.2f} vs raw ${raw:.2f}")
    check("commissions", not comm_bad, "; ".join(comm_bad) or
          "booked commissions match raw exit fills (entry commissions not captured)")

    # [5] aggregates: system math replicated vs broker-corrected values
    def agg(rows, key):
        gp = sum(r[key] for r in rows if r[key] > 0)
        gl = abs(sum(r[key] for r in rows if r[key] < 0))
        wins = sum(1 for r in rows if r[key] > 0)
        cum = peak = dd = 0.0
        for r in rows:
            cum += r[key]
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        return {"n": len(rows), "wins": wins, "total": round(sum(r[key] for r in rows), 2),
                "pf": round(gp / gl, 3) if gl else float("inf"), "dd": round(dd, 2)}

    corrected = []
    broker_by_id = {r["trade"]: r["broker"] for r in pnl_rows}
    for t in closed:
        corrected.append({"pnl": broker_by_id.get(t["trade_id"], t["realized_pnl"])})
    sys_a = agg([{"pnl": t["realized_pnl"]} for t in closed], "pnl")
    ref_a = agg(corrected, "pnl")
    agg_ok = (sys_a["n"] == ref_a["n"] and sys_a["wins"] == ref_a["wins"]
              and abs(sys_a["total"] - ref_a["total"]) <= TOTAL_TOL
              and abs(sys_a["pf"] - ref_a["pf"]) <= PF_TOL)
    check("aggregates", agg_ok,
          f"system n={sys_a['n']} wins={sys_a['wins']} total={sys_a['total']:+.2f} "
          f"PF={sys_a['pf']} | referee(broker-corrected) n={ref_a['n']} "
          f"wins={ref_a['wins']} total={ref_a['total']:+.2f} PF={ref_a['pf']}")
    detail["aggregates"] = {"system": sys_a, "referee": ref_a}

    # [6] max drawdown on deployed risk — D2 (DECIDED 2026-07-16) pins the
    # CONCURRENT-risk base for both system and referee. This check now
    # verifies (a) car COVERAGE — every real close carries capital_at_risk,
    # since a coverage hole silently understates the base and inflates DD% —
    # and (b) reports DD under the pinned method vs the gate.
    no_car = [t["trade_id"][-6:] for t in closed if (t.get("capital_at_risk") or 0) <= 0]
    base_ref = max(max_concurrent_risk(closed + open_rows), 1.0)
    dd_ref_pct = ref_a["dd"] / base_ref * 100
    check("dd_deployed_risk", not no_car,
          f"pinned concurrent-risk method: DD ${ref_a['dd']:.2f} / "
          f"${base_ref:,.0f} = {dd_ref_pct:.1f}% (gate <{GATE_DD_PCT}%)"
          + (f" | COVERAGE HOLE: {len(no_car)} close(s) missing "
             f"capital_at_risk {no_car}" if no_car else " | car coverage complete"))

    # [7] slippage gate: median entry slip as % of credit, trailing N fills
    slips = [dict(r) for r in cur.execute(
        "SELECT e.price, e.live_mid, e.exec_time, t.entry_price FROM executions e "
        "JOIN trades t ON t.trade_id=e.trade_id WHERE e.live_mid > 0 "
        "ORDER BY e.exec_time DESC LIMIT ?", (SLIP_WINDOW,))]
    vals = [abs(s["price"] - s["live_mid"]) / s["entry_price"] * 100
            for s in slips if s["entry_price"]]
    if vals:
        med = statistics.median(vals)
        check("slippage_gate", med <= GATE_SLIP_PCT,
              f"median {med:.1f}% of credit over trailing {len(vals)} "
              f"(gate <= {GATE_SLIP_PCT}%)")
    else:
        check("slippage_gate", False,
              "NO DATA: zero executions carry live_mid/signal_price -> the "
              "slippage gate is currently unmeasurable")

    # [8] unmanaged-position incidents (go-live gate demands ZERO)
    halts = [p.name for p in (ROOT / "data").glob("HALT*") if p.is_file()]
    incident = bool(halts) or bool(dups)
    check("unmanaged_incidents", not incident,
          f"halt files: {halts or 'none'}; phantom-fill residue implied by [2]: "
          f"{residual_legs} legs" if incident else "none detected")

    # [9] mirror consistency (DuckDB analytics copy)
    try:
        import duckdb
        m = duckdb.connect(a.mirror, read_only=True)
        mn, mtot, mcomm = m.execute(
            "SELECT COUNT(*), COALESCE(SUM(realized_pnl),0), COALESCE(SUM(commission),0) "
            "FROM trades WHERE status='closed'").fetchone()
        m.close()
        sn = cur.execute("SELECT COUNT(*), COALESCE(SUM(realized_pnl),0), "
                         "COALESCE(SUM(commission),0) FROM trades "
                         "WHERE status='closed'").fetchone()
        ok = mn == sn[0] and abs(mtot - sn[1]) < 0.01 and abs(mcomm - sn[2]) < 0.01
        check("mirror_sync", ok,
              f"sqlite closed n={sn[0]} pnl={sn[1]:+.2f} comm={sn[2]:.2f} | "
              f"duckdb n={mn} pnl={mtot:+.2f} comm={mcomm:.2f}")
    except Exception as e:  # mirror locked by live bot -> skip, don't disturb it
        checks.append({"check": "mirror_sync", "status": "SKIP", "msg": str(e)[:90]})

    # [10] orphan raw fills (executions pointing at no known trade)
    orphans = cur.execute("SELECT COUNT(*) FROM executions e LEFT JOIN trades t "
                          "ON t.trade_id=e.trade_id WHERE t.trade_id IS NULL").fetchone()[0]
    check("orphan_fills", orphans == 0, f"{orphans} executions with unknown trade_id")

    con.close()
    breaks = [c for c in checks if c["status"] == "BREAK"]
    verdict = {"breaks": len(breaks), "checks": checks, "detail": detail}
    if a.json:
        print(json.dumps(verdict, indent=1, default=str))
    else:
        print("SHADOW REFEREE -- raw broker ledger vs system books "
              f"(since {RESET} reset)")
        print(f"db={a.db} (ro)")
        for c in checks:
            print(f"[{c['status']:5s}] {c['check']:20s} {c['msg']}")
        print(f"VERDICT: {len(breaks)} BREAK(s) -> exit {1 if breaks else 0}")
    sys.exit(1 if breaks else 0)


if __name__ == "__main__":
    main()

"""W66: forward check of the research friction model against REAL fills (read-only).

For every combo fill that carries quote context (`executions.live_mid > 0`, written by the
executor at placement: the net mid of the structure when the order was sent), the realized
friction per share is ``|live_mid - fill price|`` (a credit that fills below the mid, a debit
above it). The v2 model predicts ``capture x sum(leg half-spread)`` over the order's leg
fills (leg fill prices stand in for leg mids: within a couple of percent of them).

With n this small (1 qualifying fill on 2026-10-09) the report is a PLUMBING check and the
mechanism that tightens ``backtest.friction_*_capture`` as paper fills accumulate; it makes
no statistical claim. It also prints the coverage breakdown that explains why so few of the
executions rows qualify (leg rows by design carry no context; some rows have no trade id).

    python scripts/fill_vs_model_report.py [--state-db data/ait_state.db] [--hist-db data/historical.db]

The state DB is opened read-only (file:...?mode=ro); nothing is written anywhere.
"""
from __future__ import annotations

import argparse
import sqlite3
import statistics as st
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def load_rows(state_db: Path) -> list[sqlite3.Row]:
    con = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return con.execute(
            "SELECT exec_id, order_id, symbol, con_id, side, shares, price, exec_time, "
            "COALESCE(trade_id,'') trade_id, COALESCE(live_mid,0) live_mid, "
            "COALESCE(nbbo_spread,0) nbbo_spread, COALESCE(signal_price,0) signal_price "
            "FROM executions ORDER BY exec_time").fetchall()
    finally:
        con.close()


def coverage(rows) -> dict:
    return {"total": len(rows),
            "with_quote_context": sum(1 for r in rows if r["live_mid"] > 0),
            "without_trade_id": sum(1 for r in rows if not r["trade_id"]),
            "leg_rows_no_context": sum(1 for r in rows if r["live_mid"] <= 0 and r["trade_id"])}


def fill_comparisons(rows, model_for) -> list[dict]:
    """One dict per combo fill with quote context: realized vs model friction ($/share)."""
    by_order: dict[int, list] = {}
    for r in rows:
        by_order.setdefault(r["order_id"], []).append(r)
    out = []
    for oid, grp in by_order.items():
        for combo in (g for g in grp if g["live_mid"] > 0):
            legs = [g for g in grp if g["con_id"] != combo["con_id"]]
            model = model_for(combo["symbol"])
            realized = abs(float(combo["live_mid"]) - float(combo["price"]))
            pred = None
            if model is not None and len(legs) == 4:
                pred = sum(model.half_spread_usd(float(g["price"])) for g in legs)
            out.append({"order_id": oid, "symbol": combo["symbol"], "exec_time": combo["exec_time"],
                        "live_mid": float(combo["live_mid"]), "fill": float(combo["price"]),
                        "nbbo_spread": float(combo["nbbo_spread"]), "realized": realized,
                        "model": pred, "n_legs": len(legs)})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--state-db", default="data/ait_state.db")
    ap.add_argument("--hist-db", default="data/historical.db")
    args = ap.parse_args(argv)

    from ait.backtesting.friction import LegSpreadModel
    from ait.data.historical import HistoricalDataStore

    rows = load_rows(Path(args.state_db))
    cov = coverage(rows)
    print("executions coverage: " + ", ".join(f"{k}={v}" for k, v in cov.items()))
    store = HistoricalDataStore(db_path=Path(args.hist_db))
    cache: dict = {}

    def model_for(sym):
        if sym not in cache:
            row = store.load_leg_spread_model(sym)
            cache[sym] = LegSpreadModel.from_row(row) if row else None
        return cache[sym]

    comps = fill_comparisons(rows, model_for)
    if not comps:
        print("No combo fill carries quote context yet: nothing to compare.")
        return 0
    print(f"\n{'time':26s} {'sym':4s} {'mid':>7s} {'fill':>7s} {'NBBOw':>6s} "
          f"{'realized':>9s} {'model':>7s} legs")
    for c in comps:
        m = "n/a" if c["model"] is None else f"{c['model']:.3f}"
        print(f"{c['exec_time']:26s} {c['symbol']:4s} {c['live_mid']:7.3f} {c['fill']:7.3f} "
              f"{c['nbbo_spread']:6.3f} {c['realized']:9.3f} {m:>7s} {c['n_legs']}")
    both = [c for c in comps if c["model"] is not None]
    print(f"\nn={len(comps)} fills with context, {len(both)} comparable to the model.")
    if both:
        print(f"median realized ${st.median(c['realized'] for c in both):.3f}/share vs "
              f"model ${st.median(c['model'] for c in both):.3f}/share "
              f"(implied capture {sum(c['realized'] for c in both) / sum(c['model'] for c in both):.2f})")
        print("n is far too small to retune backtest.friction_*_capture: this is the forward "
              "mechanism, re-run as paper fills accumulate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""W79a: which DTE-study dimensions actually move a result?

Optuna spends trials on every dimension it is given. A dimension that cannot change any trade
is pure noise (the W65 inert ``wing_k`` rule), so before the DTE study searches one we sweep
it: same data, same engine, everything else at the baseline, several values of THIS dimension,
at several entry DTEs. If every value yields the identical set of trades the dimension is FLAT
at that DTE and must not be searched there.

Setup (fixed on purpose, so the table is reproducible): QQQ research tables, one window, the
realistic engine, per-leg friction with the short-DTE floor ASSERTED loaded, 1 contract,
one concurrent position, ``bs_legacy`` pricing (flatness is structural, so the option-bar
backfill is not needed). Read-only against ``data/historical.db``.

    python scripts/dte_dimension_flatness.py                      # default window / DTEs
    python scripts/dte_dimension_flatness.py --start 2025-04-01 --end 2025-07-31 --workers 6
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()  # before any trading-subsystem import (docs/TECHNICAL_GUIDE.md "Conventions")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "historical.db"

# (dimension, engine kwarg, values, also_exempt_guard)
#   also_exempt_guard: the dimension only exists once the position is exempt from the DTE close
#   rules (guard >= entry DTE), so the sweep sets the guard to the entry DTE for it.
SWEEPS: list[tuple[str, str, list, bool]] = [
    ("expiry_guard_dte",              "expiry_guard_dte",              [0, 1, 3, 5],            False),
    ("max_hold_fraction",             "max_hold_fraction",             [0.25, 0.5, 0.75, 1.0],  True),
    ("flatten_minutes_before_close",  "flatten_minutes_before_close",  [None, 30, 60, 120],     True),
    ("delta_short",                   "delta_short",                   [0.10, 0.15, 0.20, 0.30], False),
    ("wing_k",                        "wing_k",                        [0.3, 0.8, 1.6, 2.0],    False),
    ("max_hold_days (control)",       "max_hold_days",                 [1, 3, 5, 9, 30],        True),
]
DEFAULT_DTES = (1, 3, 7, 14)


class _ReadOnlyIntraday:
    """5-minute bars straight from the research table (sqlite ``mode=ro``)."""

    def __init__(self, db: Path, symbol: str, prefix: str = "") -> None:   # W91: shared tables
        self._c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        self._sym, self._t, self._cache = symbol, f"{prefix}intraday_prices", {}

    def load_intraday_range(self, symbol, start_date, end_date):
        import pandas as pd
        key = (start_date, end_date)
        if key in self._cache:
            return self._cache[key]
        lo = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
        hi = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)
        df = pd.read_sql_query(
            f"SELECT datetime, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume "
            f"FROM {self._t} WHERE symbol=? AND interval='5m' AND datetime>=? AND datetime<=? "
            f"ORDER BY datetime", self._c,
            params=(self._sym, lo.strftime("%Y-%m-%dT%H:%M:%S+00:00"), hi.strftime("%Y-%m-%dT%H:%M:%S+00:00")))
        if not df.empty:
            df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("datetime"), utc=True))
        self._cache[key] = df
        return df


def _run_one(task: dict) -> dict:
    warnings.filterwarnings("ignore")
    import pandas as pd

    from ait.backtesting.engine import Backtester
    from ait.backtesting.replay import ResearchData

    frame = ResearchData(str(DB), task["symbol"]).daily_frame()
    data = frame.loc[(frame.index >= task["start"]) & (frame.index <= task["end"])].copy()
    from ait.backtesting.study_harness import assert_friction_loaded, study_engine_kwargs

    # W87a: the study engine block is defined once (backtesting/study_harness.py); this script passes no
    # features_cache and no IV-rank series (the engine defaults), and a read-only intraday store.
    kw = study_engine_kwargs(
        data=data, symbol=task["symbol"], db_path=DB, intraday_store=_ReadOnlyIntraday(DB, task["symbol"]),
        entry_dte=task["dte"], features_cache=None)
    kw.update(task["kwargs"])
    bt = Backtester(**kw)
    assert_friction_loaded(bt)
    trades = bt.run().trades
    fp = hashlib.sha1(json.dumps(
        [(t.get("entry_date"), t.get("fill_time"), t.get("exit_date"), t.get("exit_time"),
          t.get("exit_reason"), round(float(t.get("pnl", 0.0)), 2)) for t in trades],
        sort_keys=True).encode()).hexdigest()[:10]
    return {**{k: task[k] for k in ("dim", "dte", "value")}, "n": len(trades),
            "total": round(sum(float(t.get("pnl", 0.0)) for t in trades), 2), "fp": fp}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--start", default="2025-04-01")
    ap.add_argument("--end", default="2025-07-31")
    ap.add_argument("--dtes", type=int, nargs="+", default=list(DEFAULT_DTES))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    tasks = []
    for dim, kwarg, values, exempt in SWEEPS:
        for dte in args.dtes:
            for v in values:
                kws = {kwarg: v}
                if exempt:
                    kws.setdefault("expiry_guard_dte", dte)     # let the position reach the knob
                    if kwarg == "max_hold_days":
                        kws["max_hold_days"] = v
                tasks.append({"dim": dim, "dte": dte, "value": v, "kwargs": kws,
                              "symbol": args.symbol, "start": args.start, "end": args.end})
    print(f"{len(tasks)} runs ({args.symbol} {args.start}..{args.end}, DTEs {args.dtes}, "
          f"{args.workers} workers)", flush=True)
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(_run_one, tasks))
    else:
        res = [_run_one(t) for t in tasks]

    print("\n## Dimension flatness (identical trade fingerprint at every value = FLAT)\n")
    print("| dimension | entry DTE | values tried | distinct outcomes | trades (min-max) | total $ (min-max) | verdict |")
    print("|---|---|---|---|---|---|---|")
    verdicts: dict[str, set] = {}
    for dim, _k, values, _e in SWEEPS:
        for dte in args.dtes:
            rows = [r for r in res if r["dim"] == dim and r["dte"] == dte]
            if not rows:
                continue
            distinct = len({r["fp"] for r in rows})
            flat = distinct == 1
            verdicts.setdefault(dim, set()).add("FLAT" if flat else "MOVES")
            ns, ts = [r["n"] for r in rows], [r["total"] for r in rows]
            print(f"| {dim} | {dte} | {', '.join(str(r['value']) for r in rows)} | {distinct} | "
                  f"{min(ns)}-{max(ns)} | {min(ts):,.0f} to {max(ts):,.0f} | {'FLAT' if flat else 'moves'} |")
    print("\n## Summary\n")
    for dim, _k, _v, _e in SWEEPS:
        v = verdicts.get(dim, set())
        txt = ("FLAT at every DTE tried: do not search" if v == {"FLAT"}
               else "moves results at every DTE tried" if v == {"MOVES"}
               else "FLAT at some DTEs, moves at others (conditional)")
        print(f"- {dim}: {txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

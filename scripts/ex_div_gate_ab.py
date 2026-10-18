#!/usr/bin/env python
"""W84: what does live's ex-dividend entry block cost or protect? (F1 measurement, F2 / F3 evidence)

Live refuses a call-selling entry whose [entry, expiry] spans the projected ex-dividend date
(W51). This runs the realistic engine per symbol and entry DTE with the research twin in each
mode (``backtest.ex_div_entry_gate``):

  off        no block (research before W84)
  live       live's rule: ONE projected ex-date (parity)
  exact      any real ex-date in the window (F3: declared dates instead of the projection)
  last_k_K   a real ex-date within K days of expiry (F2: block only the end of the call's life)

Real engine runs, so a blocked entry frees its slot for a later one (unlike the post-hoc filter
of the 2026-10-16 probe).

    python scripts/ex_div_gate_ab.py --workers 3            # QQQ SPY IWM, entry DTE 14 and 7

Read-only: ``data/historical.db`` is opened read-only by the engine's loaders except for the
schema statements every HistoricalDataStore open runs. Needs ``research_dividends``
(scripts/backfill_dividends.py); the engine raises without it. Setup as gate_parity_ab.py
(W81): ``bs_legacy`` pricing, per-leg friction with the short-DTE floor ASSERTED loaded,
1 contract, ``condor_any_direction``, rise veto off.
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()  # before any trading-subsystem import (CLAUDE.md)

import argparse  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "historical.db"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def arms(k_values: list[int]) -> dict[str, tuple[str, int]]:
    """{arm name: (mode, k)}; K only matters for last_k_days."""
    out = {"off": ("off", 5), "live": ("live", 5), "exact": ("exact", 5)}
    out.update({f"last_k_{k}": ("last_k_days", int(k)) for k in k_values})
    return out


def run_arm(task: dict) -> dict:
    """One engine run. Module-level so a process pool can pickle it."""
    warnings.filterwarnings("ignore")
    from ait.backtesting.engine import Backtester
    from ait.backtesting.replay import ResearchData
    from ait.data.iv_rank import iv_rank_series
    from ait.ml.features import FeatureEngine

    rd = ResearchData(str(DB), task["symbol"])
    frame = rd.daily_frame()
    data = frame.loc[task["start"]:task["end"]].copy()
    bt = Backtester(
        data=data, strategies=["iron_condor"], initial_capital=100_000.0, macro_event_gate=False,
        allow_live_model_fallback=False,
        features_cache=FeatureEngine().compute(frame[OHLCV].loc["2023-01-01":]),
        condor_any_direction=True, intraday_store=rd, symbol=task["symbol"],
        friction_model="per_leg_usd", db_path=DB, gates_mode="observe", max_contracts_per_trade=1,
        max_concurrent_positions=1, hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0,
        iv_floor=0.05, ic_min_credit=0.30, ic_min_credit_width=0.05, entry_window_start_et="10:30",
        entry_window_end_et="15:30", scan_interval_minutes=60, engine_model="realistic",
        friction_short_dte=True, pricing_source="bs_legacy", entry_dte=int(task["entry_dte"]),
        iv_rank_gate=True, iv_rank_series=iv_rank_series(frame["implied_vol"]),
        iv_rank_rise_gate=False, ex_div_entry_gate=task["mode"], ex_div_entry_k_days=task["k"])
    assert bt._leg_spread_model is not None and bt._friction_bands is not None, \
        "per-leg friction model did NOT load"
    assert bt._ex_div_mode == task["mode"], "ex-div mode did not reach the engine"
    if task["mode"] != "off":
        assert bt._ex_dates, "the ex-dividend calendar did NOT load"
    trades = bt.run().trades
    pnls = [float(t.get("pnl", 0.0)) for t in trades]
    return {"symbol": task["symbol"], "entry_dte": task["entry_dte"], "arm": task["arm"],
            "n": len(pnls), "total": sum(pnls), "worst": min(pnls, default=0.0),
            "win": (sum(p > 0 for p in pnls) / len(pnls)) if pnls else 0.0,
            "blocked": int(bt._funnel.get("ex_div_entry", 0)),
            "ex_dates": len(bt._ex_dates or [])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY", "IWM"])
    ap.add_argument("--start", default="2024-01-02")
    ap.add_argument("--end", default="2026-10-08")
    ap.add_argument("--entry-dte", nargs="+", type=int, default=[14, 7])
    ap.add_argument("--k", nargs="+", type=int, default=[3, 5, 7], help="K values for last_k_days")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    arm_map = arms(args.k)
    tasks = [{"symbol": s, "entry_dte": d, "arm": a, "mode": m, "k": k,
              "start": args.start, "end": args.end}
             for s in args.symbols for d in args.entry_dte for a, (m, k) in arm_map.items()]
    print(f"{len(tasks)} runs ({args.start}..{args.end}, realistic engine, 1 contract)", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        res = list(ex.map(run_arm, tasks))

    print("\n## Ex-dividend entry block A/B (statistics only; QQQ first)\n")
    print("| symbol | entry DTE | arm | trades | win % | total $ | vs off $ | worst trade $ "
          "| entries blocked |")
    print("|---|---|---|---|---|---|---|---|---|")
    base = {(r["symbol"], r["entry_dte"]): r["total"] for r in res if r["arm"] == "off"}
    for r in res:
        d = r["total"] - base[(r["symbol"], r["entry_dte"])]
        print(f"| {r['symbol']} | {r['entry_dte']} | {r['arm']} | {r['n']} | {r['win']:.0%} "
              f"| {r['total']:,.0f} | {d:+,.0f} | {r['worst']:,.0f} | {r['blocked']} |")
    print("\nEx-dates loaded per symbol: "
          + ", ".join(sorted({f"{r['symbol']} {r['ex_dates']}" for r in res if r["ex_dates"]})))
    print("Differences of a few hundred $ over ~100 trades are inside the noise; the parity "
          "question is answered by the live arm, the F2 / F3 question by the others.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

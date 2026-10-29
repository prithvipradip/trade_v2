#!/usr/bin/env python
"""W81: what do the one-sided entry gates cost or earn? (decisions D10 / D11 evidence)

The 2026-10-13 audit found three gates on one side of the live/research line only:
  P1 the research realized-vol cap (``max_entry_vol_annual``, 0.80)        research only
  P2 the research trending-down regime veto (``regime_veto_gate``)         research only
  P3 live's VIX halt (``risk.credit_vix_halt``, 28)                        live only

This script measures them on the realistic engine, per symbol, in four arms:

  baseline  research as configured today (regime veto ON, vol cap 0.80)
  no_regime regime veto OFF
  no_vol    vol cap OFF
  neither   both OFF  (research closest to live on these two gates)

and then applies live's VIX halt AFTER THE FACT to every arm (drop the trades whose entry day
the previous VIX close was at or above the halt). That is an approximation: removing a trade
frees its slot for a later entry, which this does not simulate; it measures how much of each
arm's P&L sits on days live would have refused.

    python scripts/gate_parity_ab.py --workers 3                 # QQQ SPY IWM, 2024-01-02..2026-10-08

Read-only: ``data/historical.db`` is opened ``mode=ro``; VIX closes come from Yahoo (network).
Prints statistics only. Setup (fixed so the table is reproducible, as in
``iv_rank_parity_probe.py --ab``): ``bs_legacy`` pricing, per-leg friction with the short-DTE
floor ASSERTED loaded, 1 contract, ``--entry-dte`` (default 14), ``condor_any_direction``,
rise veto off (D9).
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()  # before any trading-subsystem import (docs/TECHNICAL_GUIDE.md "Conventions")

import argparse  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "historical.db"
NO_VOL_CAP = 10.0            # a cap no annualized 10-day realized vol reaches

ARMS: dict[str, dict] = {
    "baseline":  {"regime_veto_gate": True,  "vol_cap": None},
    "no_regime": {"regime_veto_gate": False, "vol_cap": None},
    "no_vol":    {"regime_veto_gate": True,  "vol_cap": NO_VOL_CAP},
    "neither":   {"regime_veto_gate": False, "vol_cap": NO_VOL_CAP},
}


def run_arm(task: dict) -> dict:
    """One engine run. Module-level so a process pool can pickle it."""
    warnings.filterwarnings("ignore")
    from ait.backtesting.engine import Backtester
    from ait.backtesting.study_harness import assert_friction_loaded, study_engine_kwargs, study_inputs

    rd, data, features, ivr = study_inputs(DB, task["symbol"], task["start"], task["end"])
    extra = {}
    if task["vol_cap"] is not None:
        extra["max_entry_vol_annual"] = float(task["vol_cap"])
    # W87a: the study engine block is defined once (backtesting/study_harness.py)
    bt = Backtester(**study_engine_kwargs(
        data=data, symbol=task["symbol"], db_path=DB, intraday_store=rd, entry_dte=int(task["entry_dte"]),
        features_cache=features, iv_rank_series=ivr,
        regime_veto_gate=bool(task["regime_veto_gate"]), **extra))
    assert_friction_loaded(bt)
    assert bt._regime_veto_gate is bool(task["regime_veto_gate"]), "regime switch did not reach the engine"
    trades = bt.run().trades
    f = dict(bt._funnel)
    return {"symbol": task["symbol"], "arm": task["arm"],
            "trades": [(str(t.get("entry_date"))[:10], float(t.get("pnl", 0.0))) for t in trades],
            "regime_veto_days": f.get("regime_veto", 0), "vol_gate_days": f.get("vol_gate", 0)}


def vix_prev_close(start: str, end: str):
    """Series: previous session's VIX close, indexed by session date (Yahoo)."""
    import pandas as pd
    import yfinance as yf

    lo = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    hi = (pd.Timestamp(end) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    df = yf.download("^VIX", start=lo, end=hi, progress=False, auto_adjust=False)
    s = df["Close"]
    s = s.iloc[:, 0] if hasattr(s, "columns") else s
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.astype(float).shift(1)


def summarise(trades: list[tuple[str, float]], prev_vix, halt: float) -> dict:
    """Pure: trades [(entry_date, pnl)] + previous-close VIX series -> statistics."""
    import pandas as pd

    n = len(trades)
    total = sum(p for _d, p in trades)
    worst = min((p for _d, p in trades), default=0.0)
    on_halt = []
    for d, p in trades:
        ts = pd.Timestamp(d)
        v = prev_vix.reindex(prev_vix.index.union([ts])).ffill().get(ts)
        if v is not None and v == v and float(v) >= halt:
            on_halt.append(p)
    kept = [p for _d, p in trades]
    for p in on_halt:
        kept.remove(p)
    return {"n": n, "total": total, "worst": worst, "win": (sum(p > 0 for _d, p in trades) / n) if n else 0.0,
            "halt_n": len(on_halt), "halt_total": sum(on_halt),
            "kept_n": len(kept), "kept_total": sum(kept)}


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from study_cli import add_study_args, load_study_config, study_pool, write_study_record
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY", "IWM"])
    ap.add_argument("--start", default="2024-01-02")
    ap.add_argument("--end", default="2026-10-08")
    ap.add_argument("--entry-dte", type=int, default=14,
                    help="entry DTE of the study (the gates sit behind the position-slot and IV-rank "
                         "gates, so their reach depends on it: run 1 / 3 / 7 for the short-DTE case)")
    ap.add_argument("--halt", type=float, default=None,
                    help="VIX halt level (default: risk.credit_vix_halt from config.yaml)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--dump", default=None,
                    help="write the raw per-arm results (entry date, pnl per trade) to this JSON file")
    add_study_args(ap)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    study = load_study_config(args.study_file)

    halt = args.halt
    if halt is None:
        from ait.config.settings import load_settings
        halt = float(load_settings(ignore_frozen=True).risk.credit_vix_halt)

    tasks = [{"symbol": s, "arm": a, "start": args.start, "end": args.end,
              "entry_dte": args.entry_dte, **ARMS[a]}
             for s in args.symbols for a in args.arms]
    print(f"{len(tasks)} runs ({args.start}..{args.end}, realistic engine, entry DTE {args.entry_dte}, "
          f"1 contract; "
          f"VIX halt {halt:g} applied after the fact)", flush=True)
    with study_pool(args.workers, args.study_file) as ex:
        res = list(ex.map(run_arm, tasks))
    write_study_record("gate_parity_ab", args, study, res, symbols=args.symbols)
    prev = vix_prev_close(args.start, args.end)
    window = prev.loc[args.start:args.end]
    n_halt_days = int((window >= halt).sum())
    # The halt filter is only meaningful if the VIX series loaded; say so loudly instead of
    # silently reporting zero halted trades from an empty download.
    print(f"VIX previous-close series: {len(window)} sessions in the window, {n_halt_days} at or "
          f"above {halt:g} ({', '.join(sorted({d.strftime('%Y-%m') for d in window[window >= halt].index}))})")
    if len(window) < 100:
        print("WARNING: the VIX series is (nearly) empty; the halt columns below are meaningless")
    if args.dump:
        import json
        Path(args.dump).write_text(json.dumps(res, default=str))

    print(f"\n## Gate parity A/B at entry DTE {args.entry_dte} (statistics only)\n")
    print("| symbol | arm | trades | win % | total $ | worst trade $ | days regime-vetoed | days vol-gated "
          "| trades on halt days | their $ | total $ if halted | trades kept |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res:
        s = summarise(r["trades"], prev, halt)
        print(f"| {r['symbol']} | {r['arm']} | {s['n']} | {s['win']:.0%} | {s['total']:,.0f} | {s['worst']:,.0f} "
              f"| {r['regime_veto_days']} | {r['vol_gate_days']} | {s['halt_n']} | {s['halt_total']:,.0f} "
              f"| {s['kept_total']:,.0f} | {s['kept_n']} |")
    print("\nThe halt columns are a post-hoc filter (no slot re-use); differences of a few $k over "
          "~100 trades per arm are inside the noise. The decision rests on parity, not on the sign.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

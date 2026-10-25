"""Wing budget study (walk-forward): what does a smaller max loss per condor cost?

QUESTION (operator, 2026-10-08). The QQQ condor entered 09-28 could lose at
most $4,995. Cap every condor's max loss at ~$2,000 instead — does the
strategy still work? Ahmed set a per-trade cap of 1.33% of NLV (~$2,643) on the
shared branch on 09-29 for a different reason (one max-loss trade must not trip
the daily breaker alone); the same harness answers both.

HOW THE CAP WORKS (live, unchanged): the risk budget is
risk.max_position_risk_pct x NLV; iron_condor._vol_scaled_width narrows the
wings to budget / (100 x (1 - credit/width floor)) and rejects a built condor
whose max loss still exceeds the budget. The engine had no such cap; this study
adds it research-only (Backtester.max_loss_per_contract_dollars, inert unless
set), plus contracts_cap=1 because live trades 1 lot and the engine otherwise
sizes by position_size_pct — narrower wings would silently become MORE
contracts and the comparison would measure leverage, not structure.

VARIANTS (all 1 lot; dollar caps are the live NLV ~$198.7k x pct):
  base_1lot      no cap                     (sanity partner)
  cap_huge_1lot  $10,000,000 cap            SANITY: must equal base_1lot exactly
  live_3pct      $5,962  (0.03, live today) BASELINE for the rule
  ahmed_1_33pct  $2,643  (0.0133, remote)
  user_1pct      $1,990  (0.01, the $2,000 ask)
Dollar caps are fixed: on 2023 prices (SPY ~$400) wings were narrower anyway,
so the caps bind mostly on recent windows. The share of entries where a cap
bound is reported per variant.

HARNESS: identical to scripts/post_event_study.py (SPY/QQQ/IWM, iron_condor
only, train126/test63/step63/gap5, ungated entries, W5 cost model, config
wing_k / credit floor, entry DTE 18, live data store via --data-root, aborts
without stored IBKR IV). Absolute level is NOT trusted (this harness puts the
live ruleset at PF ~0.6 while the live book is PF 2.2); read RELATIVE effects.

DECISION RULE (fixed before any run; do not edit after seeing results).
This is a risk reduction, not an edge search, so a cap may cost a little.
Against live_3pct, a cap is ACCEPTABLE only if ALL hold:
  1. both n >= 30;
  2. variant n >= 0.85 x baseline n (the over-budget reject and the credit
     floor may remove entries; pace is the binding constraint);
  3. PF >= baseline PF - 0.10;
  4. max drawdown $ <= baseline max drawdown $.
Otherwise COSTLY, with the failing conditions named. Any live change is the
operator's call and starts a new sample epoch.

Usage: python scripts/wing_budget_study.py --data-root <live tree> [--n-jobs 2] [--entry-dte 18]
Writes reports/wing_budget_study/entry_dte_<n>/<variant>.json, results.json, results.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig  # noqa: E402
from ait.config.settings import load_settings  # noqa: E402
from post_event_study import SYMBOLS, load_data, lower_priority, metrics  # noqa: E402

OUT_BASE = ROOT / "reports" / "wing_budget_study"
MIN_TRADES, MIN_N_RATIO, PF_TOLERANCE = 30, 0.85, 0.10
BASELINE = "live_3pct"

VARIANTS: dict[str, float | None] = {
    "base_1lot": None,
    "cap_huge_1lot": 10_000_000.0,
    "live_3pct": 5962.0,
    "ahmed_1_33pct": 2643.0,
    "user_1pct": 1990.0,
}


def structure_stats(result) -> dict:
    trades = result._all_trades()
    if not trades:
        return {}
    widths = [float(t.get("wing_width", 0) or 0) for t in trades]
    mls = [float(t.get("max_loss_per_contract", 0) or 0) for t in trades]
    credits = [float(t.get("entry_price", 0) or 0) for t in trades]
    pnls = [float(t.get("pnl", 0) or 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "cap_bound_pct": round(100 * sum(1 for t in trades if t.get("wing_capped")) / len(trades), 1),
        "mean_wing_width": round(sum(widths) / len(widths), 2),
        "mean_max_loss": round(sum(mls) / len(mls), 0),
        "max_max_loss": round(max(mls), 0),
        "mean_credit": round(sum(credits) / len(credits), 3),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "worst_trade": round(min(pnls), 2),
        "contracts_max": max(int(t.get("contracts", 1) or 1) for t in trades),
    }


async def run_variant(name, cap, bc, data, n_jobs, entry_dte, out, db_path) -> dict:
    cfg = WalkForwardConfig(
        train_days=126, test_days=63, step_days=63, gap_days=5,
        train_window_models=False,
        initial_capital=None,  # account.capital of the loaded settings (WalkForwardConfig resolves None)
        position_size_pct=bc.position_size_pct,
        wing_floor_dollars=bc.wing_floor_dollars,
        iv_floor=bc.iv_floor,
        delta_iv_scale=bc.delta_iv_scale,
        spread_base=bc.spread_base,
        spread_iv_sensitivity=bc.spread_iv_sensitivity,
        spread_dte_sensitivity=bc.spread_dte_sensitivity,
        spread_cap=bc.spread_cap,
        entry_dte=entry_dte,
        max_loss_per_contract_dollars=cap,
        contracts_cap=1,
        optimize_n_jobs=n_jobs,
    )
    bt = WalkForwardBacktester(SYMBOLS, ["iron_condor"], config=cfg, db_path=db_path,
                               progress_dir=out / name)
    t0 = time.time()
    result = await bt.run(data=data)
    m = {"variant": name, "max_loss_cap": cap, "entry_dte": entry_dte,
         "run_seconds": round(time.time() - t0), **metrics(result), **structure_stats(result)}
    (out / f"{name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"[wb] {name}: n={m['n']} PF={m['pf']} net={m['net']} DD$={m['max_dd_dollars']} "
          f"width={m.get('mean_wing_width')} maxloss~{m.get('mean_max_loss')} "
          f"bound={m.get('cap_bound_pct')}% ({m['run_seconds']}s)", flush=True)
    return m


def apply_rule(rows: list[dict]) -> tuple[str, list[str]]:
    by = {r["variant"]: r for r in rows if "error" not in r}
    notes: list[str] = []
    a, b = by.get("base_1lot"), by.get("cap_huge_1lot")
    if a and b:
        keys = ("n", "net", "pf", "max_dd_dollars")
        diff = {k: (a[k], b[k]) for k in keys if a[k] != b[k]}
        if diff:
            return f"VOID: cap_huge_1lot does not reproduce base_1lot {diff}", notes
        notes.append("sanity: a non-binding cap reproduces the uncapped 1-lot run exactly")
    base = by.get(BASELINE)
    if base is None:
        return "VOID: baseline failed", notes
    if base["n"] < MIN_TRADES:
        return f"INCONCLUSIVE: baseline n={base['n']} < {MIN_TRADES}", notes
    verdicts = []
    for name in ("ahmed_1_33pct", "user_1pct"):
        r = by.get(name)
        if r is None:
            continue
        checks = {
            "n>=30": r["n"] >= MIN_TRADES,
            f"n>={MIN_N_RATIO:.2f}x base": r["n"] >= MIN_N_RATIO * base["n"],
            f"pf>=base-{PF_TOLERANCE}": r["pf"] >= base["pf"] - PF_TOLERANCE,
            "dd$<=base": r["max_dd_dollars"] <= base["max_dd_dollars"],
        }
        failed = [k for k, ok in checks.items() if not ok]
        v = "ACCEPTABLE" if not failed else "COSTLY (fails " + ", ".join(failed) + ")"
        notes.append(f"{name}: {v}")
        verdicts.append(f"{name} {v.split(' (')[0]}")
    return "; ".join(verdicts) or "no cap variants ran", notes


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--entry-dte", type=int, default=18)
    ap.add_argument("--data-root", default=str(ROOT))
    args = ap.parse_args()
    db_path = Path(args.data_root) / "data" / "historical.db"
    os.chdir(args.data_root)
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    settings = load_settings(str(ROOT / "config.yaml"))
    lower_priority()
    out = OUT_BASE / f"entry_dte_{args.entry_dte}"
    out.mkdir(parents=True, exist_ok=True)
    bc = settings.backtest
    data, prov = load_data(db_path)
    rows = []
    for name in names:
        try:
            rows.append(await run_variant(name, VARIANTS[name], bc, data, args.n_jobs,
                                          args.entry_dte, out, db_path))
        except Exception as e:  # noqa: BLE001
            print(f"[wb] {name} FAILED: {e}", flush=True)
            rows.append({"variant": name, "error": str(e)[:300]})
    verdict, notes = apply_rule(rows)
    (out / "results.json").write_text(json.dumps(
        {"rows": rows, "verdict": verdict, "notes": notes, "data_store": str(db_path),
         "data": prov}, indent=1), encoding="utf-8")
    hdr = ("| variant | cap $ | n | win% | net $ | PF | max DD $ | avg win | avg loss | worst "
           "| touch | TP | wing $ | max loss ~$ | credit | cap bound % |")
    lines = [hdr, "|" + "---|" * 16]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['variant']} | ERROR {r['error'][:60]} |" + " |" * 14)
            continue
        er = r.get("exit_reasons", {})
        lines.append(
            f"| {r['variant']} | {r['max_loss_cap'] or '-'} | {r['n']} | {r['win_rate_pct']} "
            f"| {r['net']:,.0f} | {r['pf']} | {r['max_dd_dollars']:,.0f} | {r.get('avg_win')} "
            f"| {r.get('avg_loss')} | {r.get('worst_trade')} | {er.get('touch_stop', 0)} "
            f"| {er.get('take_profit_short', 0)} | {r.get('mean_wing_width')} "
            f"| {r.get('mean_max_loss')} | {r.get('mean_credit')} | {r.get('cap_bound_pct')} |")
    lines += ["", f"**Verdict:** {verdict}", ""] + [f"- {n}" for n in notes]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

"""Credit take-profit ladder study (walk-forward).

Question: would a different take-profit rule than the live DTE ladder
(50% > 20 DTE / 40% 11-20 / 30% 6-10 / 20% <= 5) have earned more?
Prompted by the 2026-09-11 closes: SPY and QQQ took profit at the 40% tier at
DTE 14, and real IBKR mid-marks reached ~50% by that afternoon. Two winners
cannot settle a threshold (hindsight only shows the trades that kept going our
way), so this sweeps the rule across the full history instead.

HARNESS (live parity, the variable under test is the exit only)
  - SPY/QQQ/IWM, iron_condor only, one position per symbol.
  - wing_k, credit/width floor, spreads, commission + fees ($0.9168/leg) and
    the 14-DTE entry all resolve from config.yaml exactly as a bare run does.
  - train_window_models=False: live runs ml.entry_gates_enabled=false
    (observe mode), so OOS entries are ungated here too.
  - train126 / test63 / step63 / gap5. Deliberately NON-overlapping
    (wingk_study used step42): with ungated entries, overlapping test windows
    simulate the same dates twice and would double-count identical trades.

DECISION RULE (fixed before any run; do not edit after seeing results)
  A variant replaces the live ladder only if ALL hold:
    1. its trade count n >= 30 and the baseline's n >= 30;
    2. PF > baseline PF + 0.10;
    3. net P&L per position-day >= baseline (holding longer consumes entry
       slots; pace is the binding constraint on the live verdict);
    4. max drawdown <= baseline max drawdown + 1.0 percentage point.
  Several qualify -> highest PF. None qualify -> keep the live ladder.
  Any change is a new live sample epoch and needs operator approval.

SANITY CHECK
  "live_ladder_explicit" passes CREDIT_TP_LADDER through the injection path
  and must reproduce "live_ladder" (ladder=None) exactly. A mismatch means the
  override path itself changes results, and the study is void.

ENTRY DTE SENSITIVITY (added after the first run, rule unchanged)
  The first run used the engine default entry DTE, options.dte_range[0] = 14.
  Live condors measured a median 18 DTE (14-25 over 14 trades), and the
  ladder tiers are keyed to DTE, so a 14-DTE harness spends only 3 days in
  the 11-20 tier. --entry-dte 18 reruns every variant at the live median.
  Both runs are reported in full under the same decision rule.

LIMITATIONS
  - Exits are checked once per simulated day, so intraday give-back (and
    intraday gains, like 2026-09-11's afternoon) is invisible in both
    directions.
  - Positions still open at a window's end are force-closed (backtest_end);
    longer-hold variants hit that more often. The count is reported.

Writes reports/take_profit_study/<run>/<variant>.json, <variant>_summary.txt,
results.json and results.md, where <run> is entry_dte_config or entry_dte_<n>.

Usage: python scripts/take_profit_study.py [--variants a,b] [--n-jobs 2] [--entry-dte 18]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig  # noqa: E402
from ait.config.settings import load_settings  # noqa: E402
from ait.execution.exit_policy import (  # noqa: E402
    CREDIT_TP_LADDER,
    NO_TAKE_PROFIT,
    TERMINAL_DTE_FLOOR as T,
)

OUT_BASE = ROOT / "reports" / "take_profit_study"
SYMBOLS = ["SPY", "QQQ", "IWM"]
MIN_TRADES = 30

VARIANTS: dict[str, tuple | None] = {
    "live_ladder": None,
    "live_ladder_explicit": CREDIT_TP_LADDER,
    "flat_40": ((T, 1.00, 0.40),),
    "flat_50": ((T, 1.00, 0.50),),
    "flat_60": ((T, 1.00, 0.60),),
    "ladder_plus_10": ((20, 1.00, 0.60), (10, 0.75, 0.50), (5, 0.50, 0.40), (T, 0.25, 0.30)),
    "ladder_14dte_50": ((20, 1.00, 0.50), (10, 0.75, 0.50), (5, 0.50, 0.30), (T, 0.25, 0.20)),
    "no_take_profit": ((T, 1.00, NO_TAKE_PROFIT),),
}


def _ladder_json(ladder):
    if ladder is None:
        return None
    return [[f, lt, "inf" if st == NO_TAKE_PROFIT else st] for f, lt, st in ladder]


def metrics(result) -> dict:
    trades = sorted(result._all_trades(), key=lambda t: str(t.get("exit_date", "")))
    pnls = [float(t.get("pnl", 0.0) or 0.0) for t in trades]
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    hold_days = []
    for t in trades:
        try:
            hold_days.append((date.fromisoformat(str(t["exit_date"])[:10])
                              - date.fromisoformat(str(t["entry_date"])[:10])).days)
        except (KeyError, ValueError):
            pass
    position_days = sum(max(d, 1) for d in hold_days)
    cum = peak = dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    reasons: dict[str, int] = {}
    for t in trades:
        r = str(t.get("exit_reason", "unknown"))
        reasons[r] = reasons.get(r, 0) + 1
    n = len(pnls)
    return {
        "n": n,
        "windows": len(result.windows),
        "wins": sum(1 for p in pnls if p > 0),
        "win_rate_pct": round(100 * sum(1 for p in pnls if p > 0) / n, 1) if n else 0.0,
        "net": round(sum(pnls), 2),
        "pf": round(gp / gl, 3) if gl else (float("inf") if gp else 0.0),
        "avg_trade": round(sum(pnls) / n, 2) if n else 0.0,
        "max_dd_dollars": round(dd, 2),
        "max_dd_pct": round(100 * result.max_drawdown, 2),
        "mean_hold_days": round(sum(hold_days) / len(hold_days), 2) if hold_days else 0.0,
        "position_days": position_days,
        "net_per_position_day": round(sum(pnls) / position_days, 3) if position_days else 0.0,
        "closes_per_window": round(n / len(result.windows), 2) if result.windows else 0.0,
        "exit_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


async def run_variant(name, ladder, bc, data, n_jobs, entry_dte, out) -> dict:
    cfg = WalkForwardConfig(
        train_days=126, test_days=63, step_days=63, gap_days=5,
        train_window_models=False,
        initial_capital=None,  # account.capital of the loaded settings (WalkForwardConfig resolves None)
        position_size_pct=None,  # W108b: resolves to risk.max_position_risk_pct (backtest.position_size_pct is retired)
        wing_floor_dollars=bc.wing_floor_dollars,
        iv_floor=bc.iv_floor,
        delta_iv_scale=bc.delta_iv_scale,
        spread_base=bc.spread_base,
        spread_iv_sensitivity=bc.spread_iv_sensitivity,
        spread_dte_sensitivity=bc.spread_dte_sensitivity,
        spread_cap=bc.spread_cap,
        credit_tp_ladder=ladder,
        entry_dte=entry_dte,
        optimize_n_jobs=n_jobs,
    )
    bt = WalkForwardBacktester(SYMBOLS, ["iron_condor"], config=cfg,
                               db_path=ROOT / "data" / "historical.db",
                               progress_dir=out / name)
    result = await bt.run(data=data)
    m = {"variant": name, "ladder": _ladder_json(ladder), "entry_dte": entry_dte,
         **metrics(result)}
    (out / f"{name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    try:
        (out / f"{name}_summary.txt").write_text(result.summary(), encoding="utf-8")
    except Exception:  # noqa: BLE001 — the summary is a convenience copy
        pass
    print(f"[tp] {name}: n={m['n']} PF={m['pf']} net={m['net']} "
          f"net/pos-day={m['net_per_position_day']} DD%={m['max_dd_pct']}", flush=True)
    return m


def load_data():
    # Naive-ET indexes: the engine slices windows with naive timestamps
    # (same normalisation as wingk_study).
    from ait.data.market_data import load_daily_ohlcv
    data = {}
    for sym in SYMBOLS:
        df = load_daily_ohlcv(sym, days=1500)
        if df is not None and len(df):
            if getattr(df.index, "tz", None) is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None)
            data[sym] = df
            print(f"[tp] data {sym}: {len(df)} rows "
                  f"{df.index[0].date()}..{df.index[-1].date()}", flush=True)
    return data


def apply_rule(rows: list[dict]) -> tuple[str, list[str]]:
    by = {r["variant"]: r for r in rows if "error" not in r}
    base = by.get("live_ladder")
    notes: list[str] = []
    if base is None:
        return "VOID: baseline failed", notes
    explicit = by.get("live_ladder_explicit")
    if explicit is not None:
        keys = ("n", "net", "pf", "max_dd_dollars")
        if any(explicit[k] != base[k] for k in keys):
            return ("VOID: live_ladder_explicit does not reproduce live_ladder "
                    f"({ {k: (base[k], explicit[k]) for k in keys} })"), notes
        notes.append("sanity: explicit ladder reproduces the default run exactly")
    if base["n"] < MIN_TRADES:
        return f"INCONCLUSIVE: baseline n={base['n']} < {MIN_TRADES}", notes
    qualifiers = []
    for name, r in by.items():
        if name in ("live_ladder", "live_ladder_explicit"):
            continue
        checks = {
            "n>=30": r["n"] >= MIN_TRADES,
            "pf>base+0.10": r["pf"] > base["pf"] + 0.10,
            "net/pos-day>=base": r["net_per_position_day"] >= base["net_per_position_day"],
            "dd<=base+1pp": r["max_dd_pct"] <= base["max_dd_pct"] + 1.0,
        }
        failed = [k for k, ok in checks.items() if not ok]
        notes.append(f"{name}: {'QUALIFIES' if not failed else 'fails ' + ', '.join(failed)}")
        if not failed:
            qualifiers.append(r)
    if not qualifiers:
        return "KEEP the live ladder (no variant clears every condition)", notes
    best = max(qualifiers, key=lambda r: r["pf"])
    return (f"CANDIDATE {best['variant']} (PF {best['pf']} vs {base['pf']}); "
            "live change needs operator approval and starts a new sample epoch"), notes


def lower_priority():
    """This runs on the same machine as the live bot: yield CPU to it."""
    try:
        import psutil
        p = psutil.Process()
        if sys.platform == "win32":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception as e:  # noqa: BLE001
        print(f"[tp] could not lower priority: {e}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--entry-dte", type=int, default=None,
                    help="default: engine resolves options.dte_range[0]")
    args = ap.parse_args()
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    settings = load_settings(str(ROOT / "config.yaml"))
    if not settings.exit.time_decay_scaling:
        raise SystemExit("exit.time_decay_scaling is false: live uses flat targets "
                         "and the engine ignores any ladder; this study does not apply")
    lower_priority()
    out = OUT_BASE / (f"entry_dte_{args.entry_dte}" if args.entry_dte else "entry_dte_config")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[tp] entry_dte={args.entry_dte or 'config default'} -> {out}", flush=True)
    bc = settings.backtest
    data = load_data()
    rows = []
    for name in names:
        try:
            rows.append(await run_variant(name, VARIANTS[name], bc, data, args.n_jobs,
                                          args.entry_dte, out))
        except Exception as e:  # noqa: BLE001
            print(f"[tp] {name} FAILED: {e}", flush=True)
            rows.append({"variant": name, "error": str(e)[:300]})

    verdict, notes = apply_rule(rows)
    (out / "results.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict, "notes": notes}, indent=1),
        encoding="utf-8")

    header = ("| variant | n | win% | net $ | PF | avg $ | max DD $ | DD % | hold d "
              "| net $/pos-day | touch | TP | backtest_end |")
    lines = [header, "|" + "---|" * 13]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['variant']} | ERROR: {r['error'][:80]} |" + " |" * 11)
            continue
        er = r["exit_reasons"]
        lines.append(
            f"| {r['variant']} | {r['n']} | {r['win_rate_pct']} | {r['net']:,.0f} | {r['pf']} "
            f"| {r['avg_trade']} | {r['max_dd_dollars']:,.0f} | {r['max_dd_pct']} "
            f"| {r['mean_hold_days']} | {r['net_per_position_day']} "
            f"| {er.get('touch_stop', 0)} | {er.get('take_profit_short', 0)} "
            f"| {er.get('backtest_end', 0)} |")
    lines += ["", f"**Verdict:** {verdict}", ""] + [f"- {n}" for n in notes]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

"""Per-trade budget ladder (walk-forward): 5% / 7.5% / 10% / 15% / 20% of the pool at wing_k 1.6 and 1 lot.

QUESTION. The narrow-wing study (scripts/narrow_wing_study.py) came back VOID on a wrong wiring
expectation, but its exploratory reading showed the shipped 5% / 3-wide setup as the worst budget tried
in the model and the 20% / 11-wide regime as the best. The per-trade percentage was reverted from 20% to
5% on a risk argument, not on return evidence. This study asks the question properly.

PRE-REGISTERED in PLAN.md "W109b AS PLANNED" (2026-10-28), before this script existed or any run. The
rule below is a copy; the PLAN entry is the authority and is not edited after any result. Operator
decisions: a per-trade loss cap of 10% of the pool (levels above it are run and reported, never
supported), a drawdown ceiling of 15%, a reference arm for the width question.

HARNESS (identical to narrow_wing_study). SPY / QQQ / IWM, iron_condor only, train_window_models=False,
train 126 / test 63 / step 63 / gap 5, the stored-IBKR-IV store, use_research_config() then
gates_mode="observe", per-leg friction fitted, a $5,000 pool with the W108 pool gates, entry_dte 18 and
25, 1 lot, scalers off. wing_k and the lot cap are NOT varied: at wing_k 1.6 the budget ceiling decides
the width and one ceiling-width lot already uses 75-90% of the budget.

VARIANTS (budget, wing_k, lots 1). base 5% / 1.6 = BASELINE; sanity_default (wing_k None) must reproduce
it; candidates b075 7.5%, b10 10%; above the cap, reported only: b15 15%, b20 20%; reference only:
ref20_k055 (20%, wing_k 0.55, a 7-8-wide, to separate a higher budget from a wider wing).

DECISION RULE (fixed before any run; do not edit after seeing results). Candidates are b075 and b10.
Per DTE, against base, a candidate is SUPPORTED only if at BOTH DTE:
  1. n >= 30 and n >= 0.85 x base n;
  2. PF >= base PF + 0.10 and net per position-day >= base;
  3. max drawdown <= 15% of the pool and <= 1.25 x base's, and the largest single-trade structural loss
     <= 10% of the pool;
  4. >= 60% of traded windows with window net >= base's AND a positive point estimate of the
     block-bootstrapped net difference (80% interval reported);
  5. friction share <= base + 10 points (entry crossing + round-trip commissions over gross credit; the
     exit crossing is not separable);
  6. ABSOLUTE: net > 0 and PF >= 1.0 (a configuration that loses money in the model is never supported,
     even if it beats a worse base);
  7. a survivor repeats at friction_stress_multiplier 2.0 (base rerun at 2.0 too) and still passes 1-3
     and 6.
Dose-response label over the whole ladder 5% -> 20%: PF non-decreasing in >= 3 of the 4 adjacent pairs at
each DTE; a candidate passing 1-7 while the ladder is not monotone is SINGLE-POINT (exploratory), not
supported. Verdicts: BUDGET RAISE SUPPORTED to <level> (the supported candidate with the highest net per
position-day; ties to the lower budget) / SINGLE-POINT / NO BUDGET WITHIN THE CAP BEATS 5% (keep 5%) /
VOID. Above-cap and reference rows are printed in the same table, marked informational.

WIRING CHECKS (corrected from the narrow-wing study). VOID: the per-leg friction model is missing for a
symbol; sanity_default differs from base; mean wing is not non-decreasing along the ladder, or b20 is not
strictly wider than base; mean max loss is not non-decreasing along the ladder. FLAGGED: identical
results between two rungs (inert); runs that errored.

LIMITS (printed in results.md). Black-Scholes + skew from stored daily IV (half the days carry stored
IV), not quotes; about 50-100 trades per variant and 14-22 windows, so only large effects are visible;
the absolute level is not trusted (an earlier run of this harness under the legacy cost model put the
shipped ruleset at PF 0.18 against the live book's 2.2); read RELATIVE effects only. A result licenses a
config PROPOSAL to the operator (a new sample epoch if accepted), never a live change.

Usage: python scripts/budget_ladder_study.py [--data-root <tree>] [--entry-dtes 18,25] [--n-jobs 2]
       [--variants a,b,...] [--skip-stress] [--out <dir>]
Writes <out>/entry_dte_<n>/<variant>.json, <out>/stress_2x/..., results.json, results.md.
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

from narrow_wing_study import (  # noqa: E402  (the committed W109 helpers; that script is not edited)
    DD_BASE_MULT, FRICTION_MARGIN, MIN_N_RATIO, MIN_TRADES, MIN_WINDOW_SHARE, PF_MARGIN,
    STRESS_MULTIPLIER, Variant, build_config, dd_pct, friction_preflight, paired_window_stats,
    trade_stats, window_nets,
)

OUT_BASE = ROOT / "reports" / "budget_ladder_study"
BASELINE, SANITY = "base", "sanity_default"

# decision-rule constants (pre-registered; the ones shared with the narrow-wing study are imported)
DD_POOL_CEILING_PCT = 15.0
MAX_SINGLE_LOSS_POOL_PCT = 10.0
MIN_ABSOLUTE_PF = 1.0
MONOTONE_MIN_PAIRS = 3

VARIANTS: dict[str, Variant] = {
    "base": Variant(0.05, 1.6, 1, "base"),
    "sanity_default": Variant(0.05, None, 1, "sanity"),
    "b075": Variant(0.075, 1.6, 1, "candidate"),
    "b10": Variant(0.10, 1.6, 1, "candidate"),
    "b15": Variant(0.15, 1.6, 1, "above_cap"),
    "b20": Variant(0.20, 1.6, 1, "above_cap"),
    "ref20_k055": Variant(0.20, 0.55, 1, "reference"),
}
LADDER = ["base", "b075", "b10", "b15", "b20"]


def candidates() -> list[str]:
    return [n for n, v in VARIANTS.items() if v.role == "candidate"]


# --------------------------------------------------------------------------- pure rule

def criteria(row: dict, base: dict, paired: dict) -> dict[str, bool]:
    """The pre-registered criteria 1-6 of one candidate row against the baseline row (same DTE)."""
    bf, rf = base.get("friction_share"), row.get("friction_share")
    return {
        "1_n": row["n"] >= MIN_TRADES and row["n"] >= MIN_N_RATIO * base["n"],
        "2_pf_and_rate": row["pf"] >= base["pf"] + PF_MARGIN
        and row["net_per_position_day"] >= base["net_per_position_day"],
        "3_risk": dd_pct(row) <= DD_POOL_CEILING_PCT and dd_pct(row) <= DD_BASE_MULT * dd_pct(base)
        and 100.0 * float(row["max_single_loss"]) / float(row["pool"]) <= MAX_SINGLE_LOSS_POOL_PCT,
        "4_windows": paired["share_ge"] >= MIN_WINDOW_SHARE and paired["point"] > 0,
        "5_friction": (rf is not None and bf is not None and rf <= bf + FRICTION_MARGIN),
        "6_absolute": row["net"] > 0 and row["pf"] >= MIN_ABSOLUTE_PF,
    }


def criteria_stress(row: dict, base: dict) -> dict[str, bool]:
    """Criterion 7: a survivor still passes 1-3 and 6 against the baseline rerun at the stressed friction."""
    c = criteria(row, base, {"share_ge": 1.0, "point": 1.0})
    return {k: c[k] for k in ("1_n", "2_pf_and_rate", "3_risk", "6_absolute")}


def monotone(rows: dict[str, dict]) -> tuple[bool, int, int]:
    """PF non-decreasing in at least MONOTONE_MIN_PAIRS adjacent pairs of the ladder (needs every rung)."""
    if any(n not in rows or "error" in rows[n] for n in LADDER):
        return False, 0, 0
    pfs = [rows[n]["pf"] for n in LADDER]
    ups = sum(1 for a, b in zip(pfs, pfs[1:]) if b >= a)
    return ups >= MONOTONE_MIN_PAIRS, ups, len(pfs) - 1


def wiring(rows_by_dte: dict[int, dict[str, dict]], friction_ok: bool) -> tuple[list[str], list[str]]:
    """(void reasons, flags)."""
    void: list[str] = []
    flags: list[str] = []
    if not friction_ok:
        void.append("the fitted per-leg friction model is missing for a symbol (legacy % model would run)")
    for dte, rows in sorted(rows_by_dte.items()):
        for name, r in rows.items():
            if "error" in r:
                flags.append(f"DTE {dte}: {name} errored: {r['error']}")
        base, san = rows.get(BASELINE), rows.get(SANITY)
        if base and san and "error" not in base and "error" not in san:
            diff = {k: (base[k], san[k]) for k in ("n", "net", "pf", "max_dd_dollars") if base[k] != san[k]}
            if diff:
                void.append(f"DTE {dte}: {SANITY} does not reproduce {BASELINE}: {diff}")
        ladder = [rows[n] for n in LADDER if n in rows and "error" not in rows[n]]
        if len(ladder) == len(LADDER):
            wings = [r["mean_wing"] for r in ladder]
            losses = [r["mean_max_loss"] for r in ladder]
            if any(b < a for a, b in zip(wings, wings[1:])) or not wings[-1] > wings[0]:
                void.append(f"DTE {dte}: mean wing is not non-decreasing along the ladder with b20 wider "
                            f"than base: {wings} (the budget is not wired)")
            if any(b < a for a, b in zip(losses, losses[1:])):
                void.append(f"DTE {dte}: mean max loss is not non-decreasing along the ladder: {losses}")
        names = sorted(n for n in rows if "error" not in rows[n])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                va, vb = VARIANTS[a], VARIANTS[b]
                if (va.budget, va.wing_k) == (vb.budget, vb.wing_k) or {a, b} == {BASELINE, SANITY}:
                    continue
                if all(rows[a][k] == rows[b][k] for k in ("n", "net", "pf", "max_dd_dollars")):
                    flags.append(f"DTE {dte}: {a} and {b} have identical results (inert rung)")
    return void, flags


def stage_a(rows_by_dte: dict[int, dict[str, dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name in candidates():
        per_dte: dict[int, dict] = {}
        ok = True
        for dte, rows in sorted(rows_by_dte.items()):
            if name not in rows or BASELINE not in rows or "error" in rows[name] or "error" in rows[BASELINE]:
                ok = False
                per_dte[dte] = {"missing": True}
                continue
            paired = paired_window_stats(rows[BASELINE]["window_nets"], rows[name]["window_nets"])
            c = criteria(rows[name], rows[BASELINE], paired)
            per_dte[dte] = {"criteria": c, "paired": paired}
            ok = ok and all(c.values())
        out[name] = {"pass": ok, "per_dte": per_dte}
    return out


def stage_b(stress_by_dte: dict[int, dict[str, dict]], survivors: list[str]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for name in survivors:
        ok = bool(stress_by_dte)
        for rows in stress_by_dte.values():
            if name not in rows or BASELINE not in rows or "error" in rows[name] or "error" in rows[BASELINE]:
                ok = False
                continue
            ok = ok and all(criteria_stress(rows[name], rows[BASELINE]).values())
        out[name] = ok
    return out


def verdict(rows_by_dte, stress_by_dte, friction_ok) -> dict:
    void, flags = wiring(rows_by_dte, friction_ok)
    if void:
        return {"verdict": "VOID", "void": void, "flags": flags, "supported": [], "notes": []}
    a = stage_a(rows_by_dte)
    survivors = [n for n, d in a.items() if d["pass"]]
    notes: list[str] = []
    if survivors and not stress_by_dte:
        notes.append("survivors of criteria 1-6 were NOT stress-tested (--skip-stress): not supported")
        supported: list[str] = []
    else:
        b = stage_b(stress_by_dte, survivors)
        supported = [n for n in survivors if b.get(n)]
        for n in survivors:
            if not b.get(n):
                notes.append(f"{n}: passes 1-6 but fails the stressed repeat (criterion 7)")
    for n, d in a.items():
        if not d["pass"]:
            failed = sorted({k for dd in d["per_dte"].values() for k, ok in dd.get("criteria", {}).items() if not ok})
            notes.append(f"{n}: fails {', '.join(failed) if failed else 'a missing run'}")
    mono = {dte: monotone(rows) for dte, rows in sorted(rows_by_dte.items())}
    mono_ok = bool(mono) and all(m[0] for m in mono.values())
    for dte, (ok, ups, pairs) in mono.items():
        notes.append(f"dose-response DTE {dte}: PF non-decreasing in {ups} of {pairs} adjacent pairs "
                     f"({'monotone' if ok else 'NOT monotone'})")
    if supported and not mono_ok:
        v = "SINGLE-POINT"
    elif supported:
        def rate(n):
            return sum(rows[n]["net_per_position_day"] for rows in rows_by_dte.values()) / len(rows_by_dte)
        best = sorted(supported, key=lambda n: (-rate(n), VARIANTS[n].budget))[0]
        v = f"BUDGET RAISE SUPPORTED to {VARIANTS[best].budget:.1%} ({best})"
    else:
        v = "NO BUDGET WITHIN THE CAP BEATS 5%"
    return {"verdict": v, "void": [], "flags": flags, "supported": supported, "stage_a": a,
            "monotone": {str(k): list(m) for k, m in mono.items()}, "notes": notes}


# --------------------------------------------------------------------------- running

async def run_variant(name, v, entry_dte, bc, data, n_jobs, out: Path, db_path, pool, stress=None) -> dict:
    from ait.backtesting.walkforward import WalkForwardBacktester
    from post_event_study import SYMBOLS, metrics
    cfg = build_config(v, entry_dte, n_jobs, bc, stress)
    bt = WalkForwardBacktester(SYMBOLS, ["iron_condor"], config=cfg, db_path=db_path,
                               progress_dir=out / name)
    t0 = time.time()
    result = await bt.run(data=data)
    trades = result._all_trades()
    losses = [float(t.get("max_loss_per_contract", 0) or 0) for t in trades]
    pnls = [float(t.get("pnl", 0) or 0) for t in trades]
    row = {"variant": name, "entry_dte": entry_dte, "stress": stress, "pool": pool,
           "budget": v.budget, "wing_k": v.wing_k, "lots": v.lots, "role": v.role,
           "run_seconds": round(time.time() - t0), **metrics(result), **trade_stats(trades),
           "max_single_loss": round(max(losses), 2) if losses else 0.0,
           "worst_trade": round(min(pnls), 2) if pnls else 0.0,
           "window_nets": window_nets([(w.window_id, w.backtest_result.trades) for w in result.windows])}
    row["dd_pool_pct"] = round(dd_pct(row), 2)
    (out / f"{name}.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
    print(f"[bl] dte{entry_dte}{'' if stress is None else f' x{stress}'} {name}: n={row['n']} PF={row['pf']} "
          f"net={row['net']} DD%={row['dd_pool_pct']} wing={row['mean_wing']} "
          f"maxloss~{row['mean_max_loss']} ({row['run_seconds']}s)", flush=True)
    return row


def render(rows_by_dte, stress_by_dte, result: dict) -> str:
    lines = ["# Per-trade budget ladder (W109b)", ""]
    hdr = ("| variant | role | budget | wing_k | n | win% | net $ | PF | DD $ | DD % pool | net/pos-day "
           "| mean wing | max loss ~$ | largest loss $ | worst trade $ | credit | friction |")
    for title, by in (("", rows_by_dte), (" stressed (friction x2)", stress_by_dte)):
        for dte, rows in sorted(by.items()):
            lines += [f"## Entry DTE {dte}{title}", "", hdr, "|" + "---|" * 17]
            for name, r in rows.items():
                if "error" in r:
                    lines.append(f"| {name} | ERROR {r['error'][:60]} |" + " |" * 15)
                    continue
                note = "" if r["role"] in ("base", "sanity", "candidate") else " (outside the operator's cap: informational)"
                lines.append(
                    f"| {name} | {r['role']}{note} | {r['budget']:.1%} | {r['wing_k']} | {r['n']} | {r['win_rate_pct']} "
                    f"| {r['net']:,.0f} | {r['pf']} | {r['max_dd_dollars']:,.0f} | {r['dd_pool_pct']} "
                    f"| {r['net_per_position_day']} | {r['mean_wing']} | {r['mean_max_loss']} | {r['max_single_loss']} "
                    f"| {r['worst_trade']} | {r['mean_credit']} | {r['friction_share']} |")
            lines.append("")
    lines += [f"**Verdict: {result['verdict']}**", ""]
    for key in ("void", "flags", "notes"):
        for item in result.get(key, []):
            lines.append(f"- {key.upper()}: {item}")
    if result.get("supported"):
        lines.append(f"- SUPPORTED: {', '.join(result['supported'])}")
    lines += ["", "Limits: Black-Scholes + skew from stored daily IV, not quotes; about 50-100 trades per variant and "
              "14-22 windows (only large effects are visible); the absolute level is not trusted; read RELATIVE "
              "effects only; friction share excludes the exit crossing. Rows marked informational are outside the "
              "operator's 10% per-trade cap and cannot be supported.", ""]
    return "\n".join(lines) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--entry-dtes", default="18,25")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--data-root", default=str(ROOT))
    ap.add_argument("--skip-stress", action="store_true")
    ap.add_argument("--out", default=str(OUT_BASE))
    args = ap.parse_args()
    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    if BASELINE not in names:
        names.insert(0, BASELINE)
    dtes = [int(x) for x in args.entry_dtes.split(",") if x.strip()]
    db_path = Path(args.data_root) / "data" / "historical.db"
    os.chdir(args.data_root)

    from ait.config.research import use_research_config
    use_research_config()                      # config.yaml + config_research.yaml (research only)
    from ait.config.settings import load_settings
    settings = load_settings()
    bc, pool = settings.backtest, float(settings.account.capital)
    from post_event_study import load_data, lower_priority
    lower_priority()
    out = Path(args.out)
    friction_ok = friction_preflight(db_path)
    data, prov = load_data(db_path)

    rows_by_dte: dict[int, dict[str, dict]] = {}
    for dte in dtes:
        d_out = out / f"entry_dte_{dte}"
        d_out.mkdir(parents=True, exist_ok=True)
        rows_by_dte[dte] = {}
        for name in names:
            try:
                rows_by_dte[dte][name] = await run_variant(
                    name, VARIANTS[name], dte, bc, data, args.n_jobs, d_out, db_path, pool)
            except Exception as e:  # noqa: BLE001 — a failed variant is reported, never hidden
                print(f"[bl] dte{dte} {name} FAILED: {e}", flush=True)
                rows_by_dte[dte][name] = {"variant": name, "error": str(e)[:300]}

    clean = {k: {n: r for n, r in v.items() if "error" not in r} for k, v in rows_by_dte.items()}
    stress_by_dte: dict[int, dict[str, dict]] = {}
    if not args.skip_stress:
        survivors = [n for n, d in stage_a(clean).items() if d["pass"]]
        print(f"[bl] stress repeat for survivors of criteria 1-6: {survivors}", flush=True)
        if survivors:
            for dte in dtes:
                s_out = out / "stress_2x" / f"entry_dte_{dte}"
                s_out.mkdir(parents=True, exist_ok=True)
                stress_by_dte[dte] = {}
                for name in [BASELINE] + survivors:
                    stress_by_dte[dte][name] = await run_variant(
                        name, VARIANTS[name], dte, bc, data, args.n_jobs, s_out, db_path, pool,
                        stress=STRESS_MULTIPLIER)

    result = verdict(clean, stress_by_dte, friction_ok)
    for dte, rows in rows_by_dte.items():          # carry run errors into the report as flags
        for n, r in rows.items():
            if "error" in r:
                result.setdefault("flags", []).append(f"DTE {dte}: {n} errored: {r['error']}")
    (out / "results.json").write_text(json.dumps(
        {"rows": rows_by_dte, "stress_rows": stress_by_dte, "result": result, "data": prov,
         "pool": pool, "entry_dtes": dtes}, indent=1, default=str), encoding="utf-8")
    md = render(rows_by_dte, stress_by_dte, result)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    asyncio.run(main())

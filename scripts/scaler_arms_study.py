"""Sizing scalers on a multi-lot condor book (walk-forward): W108 phase 3c.

QUESTION. On a free-width-bound account with a lot cap above 1, does turning the condor's scalers on
(`position_sizing.scaled_strategies: [iron_condor]`) improve risk-adjusted results against the unscaled
budget-only lot count? (At the $5,000 simulated account, or with the shipped 1-lot cap, the scalers cannot
change a condor at all: tests/test_w108_scalers_inert_for_condors.py.)

PRE-REGISTERED in PLAN.md "W108 phase 3c AS PLANNED" (2026-10-29), before this script existed or any run.
The rule below is a copy; the PLAN entry is the authority and is not edited after any result.

HARNESS (as scripts/budget_ladder_study.py). SPY / QQQ / IWM, iron_condor only, train 126 / test 63 /
step 63 / gap 5, train_window_models=False, the stored-IBKR-IV store, use_research_config() then
gates_mode="observe", per-leg friction fitted, entry DTE 18 and 25, wing_k from config. Pool $200,000,
budget 3% (the Windows box's live values).

ARMS. unscaled_3 (lot cap 3, scalers off) = BASELINE; scaled_3 (lot cap 3, scalers on) = CANDIDATE;
one_lot (lot cap 1, scalers off) = reference; scaled_1 (lot cap 1, scalers on) = SANITY.

WIRING (VOID if any fails). Friction model present for all three symbols; scaled_1 identical to one_lot
(n, net, PF, max DD); unscaled_3 mean lots > 1.0; scaled_3 mean lots < unscaled_3 mean lots; scaled_3 and
unscaled_3 have the same number of trades.

DECISION RULE (fixed before any run). scaled_3 is SUPPORTED only if at BOTH DTE:
  1. n >= 30;
  2. max drawdown $ <= unscaled_3's;
  3. if unscaled_3 net > 0: net / max DD $ >= 1.10 x unscaled_3's; else scaled_3 net > unscaled_3 net;
  4. scaled_3 net > 0;
  5. a repeat at friction_stress_multiplier 2.0 (both arms) still passes 2-4.
Verdicts: SCALERS SUPPORTED FOR MULTI-LOT / NOT SUPPORTED / VOID.

LIMITS. Black-Scholes + skew from stored daily IV, not quotes; 50-100 trades per arm; the absolute level
is not trusted, read RELATIVE effects. A result licenses a config PROPOSAL, never a live change.

Usage: python scripts/scaler_arms_study.py [--data-root <tree>] [--entry-dtes 18,25] [--n-jobs 2]
       [--skip-stress] [--out <dir>]
Writes <out>/entry_dte_<n>/<arm>.json, <out>/stress_2x/..., results.json, results.md.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from narrow_wing_study import STRESS_MULTIPLIER, Variant, friction_preflight, run_variant  # noqa: E402

OUT_BASE = ROOT / "reports" / "scaler_arms_study"
POOL = 200_000.0
BUDGET = 0.03
BASELINE, CANDIDATE, REFERENCE, SANITY = "unscaled_3", "scaled_3", "one_lot", "scaled_1"
MIN_TRADES = 30
RATIO_MARGIN = 1.10


@dataclass(frozen=True)
class Arm:
    lots: int
    scaled: bool
    role: str


ARMS: dict[str, Arm] = {
    BASELINE: Arm(3, False, "base"),
    CANDIDATE: Arm(3, True, "candidate"),
    REFERENCE: Arm(1, False, "reference"),
    SANITY: Arm(1, True, "sanity"),
}


# --------------------------------------------------------------------------- the rule (pure)

def _same_outcome(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k) for k in ("n", "net", "pf", "max_dd_dollars"))


def wiring(rows_by_dte: dict[int, dict[str, dict]], friction_ok: bool) -> list[str]:
    void = [] if friction_ok else ["per-leg friction model missing for a symbol"]
    for dte, rows in rows_by_dte.items():
        missing = [n for n in ARMS if n not in rows]
        if missing:
            void.append(f"DTE {dte}: arm(s) did not run: {missing}")
            continue
        b, c, r, s = rows[BASELINE], rows[CANDIDATE], rows[REFERENCE], rows[SANITY]
        if not _same_outcome(s, r):
            void.append(f"DTE {dte}: {SANITY} differs from {REFERENCE} (scalers changed a one-lot book)")
        if not (b.get("mean_lots") or 0) > 1.0:
            void.append(f"DTE {dte}: {BASELINE} mean lots {b.get('mean_lots')} <= 1 (multi-lot never exercised)")
        if not (c.get("mean_lots") or 0) < (b.get("mean_lots") or 0):
            void.append(f"DTE {dte}: {CANDIDATE} mean lots {c.get('mean_lots')} not below {BASELINE}'s")
        if c.get("n") != b.get("n"):
            void.append(f"DTE {dte}: trade counts differ ({c.get('n')} vs {b.get('n')}): scalers changed viability")
    return void


def _ratio(row: dict) -> float:
    dd = float(row.get("max_dd_dollars") or 0)
    return float("inf") if dd <= 0 else float(row["net"]) / dd


def criteria(cand: dict, base: dict, *, with_n: bool = True) -> dict[str, bool]:
    if float(base["net"]) > 0:
        better = _ratio(cand) >= RATIO_MARGIN * _ratio(base)
    else:
        better = float(cand["net"]) > float(base["net"])
    out = {"dd<=base": float(cand["max_dd_dollars"]) <= float(base["max_dd_dollars"]),
           "risk-adjusted better": better,
           "net>0": float(cand["net"]) > 0}
    if with_n:
        out = {"n>=30": int(cand["n"]) >= MIN_TRADES, **out}
    return out


def verdict(rows_by_dte, stress_by_dte, friction_ok) -> dict:
    void = wiring(rows_by_dte, friction_ok)
    if void:
        return {"verdict": "VOID", "void": void, "per_dte": {}, "notes": []}
    per_dte = {dte: criteria(rows[CANDIDATE], rows[BASELINE]) for dte, rows in rows_by_dte.items()}
    notes: list[str] = []
    passed = all(all(c.values()) for c in per_dte.values())
    if passed:
        if not stress_by_dte:
            notes.append("criteria 1-4 pass but the stressed repeat was skipped: NOT SUPPORTED")
            passed = False
        else:
            for dte, rows in stress_by_dte.items():
                c = criteria(rows[CANDIDATE], rows[BASELINE], with_n=False)
                if not all(c.values()):
                    notes.append(f"DTE {dte} stressed: fails {[k for k, ok in c.items() if not ok]}")
                    passed = False
    for dte, c in per_dte.items():
        failed = [k for k, ok in c.items() if not ok]
        if failed:
            notes.append(f"DTE {dte}: fails {failed}")
    return {"verdict": "SCALERS SUPPORTED FOR MULTI-LOT" if passed else "NOT SUPPORTED",
            "void": [], "per_dte": per_dte, "notes": notes}


def render(rows_by_dte, stress_by_dte, result: dict) -> str:
    lines = ["# Sizing scalers on a multi-lot condor book (W108 phase 3c)", "",
             f"Pool ${POOL:,.0f}, budget {BUDGET:.0%}. Verdict: **{result['verdict']}**", ""]
    hdr = "| DTE | stress | arm | role | lots cap | scaled | n | net $ | PF | max DD $ | net/DD | mean lots |"
    lines += [hdr, "|" + "---|" * 12]
    for label, block in (("1x", rows_by_dte), ("2x", stress_by_dte)):
        for dte, rows in block.items():
            for name, r in rows.items():
                a = ARMS[name]
                lines.append(f"| {dte} | {label} | {name} | {a.role} | {a.lots} | {a.scaled} | {r.get('n')} | "
                             f"{r.get('net')} | {r.get('pf')} | {r.get('max_dd_dollars')} | "
                             f"{_ratio(r):.2f} | {r.get('mean_lots')} |")
    lines += [""] + [f"- VOID: {v}" for v in result.get("void", [])] + [f"- {n}" for n in result.get("notes", [])]
    lines += ["", "Limits: Black-Scholes + skew from stored daily IV, not quotes; 50-100 trades per arm; read "
              "RELATIVE effects only. A result licenses a config proposal, never a live change.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- running

def arm_settings(base_settings, arm: Arm):
    """The settings one arm runs under: the pool, and the scalers on or off for the condor."""
    ps = base_settings.position_sizing.model_copy(
        update={"scaled_strategies": ["iron_condor"] if arm.scaled else []})
    acct = base_settings.account.model_copy(update={"capital": POOL})
    return base_settings.model_copy(update={"position_sizing": ps, "account": acct})


async def run_arm(name, dte, base_settings, data, n_jobs, out: Path, db_path, stress=None) -> dict:
    from ait.config.settings import freeze_settings
    arm = ARMS[name]
    freeze_settings(arm_settings(base_settings, arm))      # the walk-forward hands these to its workers
    row = await run_variant(name, Variant(BUDGET, None, arm.lots, arm.role), dte, base_settings.backtest,
                            data, n_jobs, out, db_path, POOL, stress=stress)
    row["scaled"] = arm.scaled
    return row


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-dtes", default="18,25")
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--data-root", default=str(ROOT))
    ap.add_argument("--skip-stress", action="store_true")
    ap.add_argument("--out", default=str(OUT_BASE))
    args = ap.parse_args()
    dtes = [int(x) for x in args.entry_dtes.split(",") if x.strip()]
    db_path = Path(args.data_root) / "data" / "historical.db"
    os.chdir(args.data_root)

    from ait.config.research import use_research_config
    base_settings = use_research_config()
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
        for name in ARMS:
            try:
                rows_by_dte[dte][name] = await run_arm(name, dte, base_settings, data, args.n_jobs, d_out, db_path)
            except Exception as e:  # noqa: BLE001 - a failed arm is reported (and VOIDs the run), never hidden
                print(f"[sa] dte{dte} {name} FAILED: {e}", flush=True)

    stress_by_dte: dict[int, dict[str, dict]] = {}
    first = verdict(rows_by_dte, {}, friction_ok)
    if not args.skip_stress and first["verdict"] != "VOID" and all(
            all(c.values()) for c in first["per_dte"].values()):
        for dte in dtes:
            s_out = out / "stress_2x" / f"entry_dte_{dte}"
            s_out.mkdir(parents=True, exist_ok=True)
            stress_by_dte[dte] = {n: await run_arm(n, dte, base_settings, data, args.n_jobs, s_out, db_path,
                                                   stress=STRESS_MULTIPLIER) for n in (BASELINE, CANDIDATE)}

    result = verdict(rows_by_dte, stress_by_dte, friction_ok)
    (out / "results.json").write_text(json.dumps(
        {"rows": rows_by_dte, "stress_rows": stress_by_dte, "result": result, "data": prov,
         "pool": POOL, "budget": BUDGET, "entry_dtes": dtes}, indent=1, default=str), encoding="utf-8")
    md = render(rows_by_dte, stress_by_dte, result)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    asyncio.run(main())

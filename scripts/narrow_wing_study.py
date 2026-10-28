"""Narrow-wing and multi-lot study (walk-forward): does a narrower wing and/or more than one lot beat the shipped condor?

QUESTION (operator, 2026-10-27). At the $5k account the shipped configuration is 5% per trade,
``wing_k`` 1.6 (the budget ceiling makes it a 3-wide condor) and 1 lot. Does a NARROWER wing and/or
MORE THAN ONE LOT beat it? The answer decides whether Phase 2 (live multi-lot: the builders requesting
lots, multi-leg scale-out skipped, multi-lot cases for the W101-B exit pricing and the W101-C expiry
booking, a staged paper ramp, section-3 files) is worth building. At 5% a 3-wide risks about $200
against a $250 budget, so live is 1 lot whatever the rules say; multi-lot appears only at a higher
per-trade percentage with narrow wings.

PRE-REGISTERED in PLAN.md "W109 AS PLANNED" (2026-10-28), before this script existed or any run.
The rule below is a copy; the PLAN entry is the authority and is not edited after any result.

HARNESS (fixed). SPY / QQQ / IWM, iron_condor only, ``train_window_models=False``, walk-forward
train 126 / test 63 / step 63 / gap 5, the stored-IBKR-IV store (``data/historical.db``), one shared
pool of ``account.capital`` ($5,000) with the W108 pool gates on, ``entry_dte`` 18 (the live median)
and 25 (robustness). ``use_research_config()`` first (calibrated ``friction_model: per_leg_usd``,
IV-rank gate, ``condor_any_direction false``), then ``gates_mode="observe"`` explicitly: live runs
observe, and the overlay's ``enforce`` with no trained models would reject every entry. That is the
one deliberate deviation. Budget (``position_size_pct``), ``wing_k`` and ``max_contracts_per_trade``
are set per variant; scalers are not a variable (off: the shared rule ``lots_for_budget``).

VARIANTS (budget, wing_k, max lots). ``base`` 5% / 1.6 / 1 is the BASELINE (the shipped config);
``sanity_default`` is ``base`` with ``wing_k=None`` and must reproduce it exactly. ``ref20_k160_1``
(20%, the W100 regime) is a REFERENCE and never a candidate.

WIRING CHECKS. VOID: the fitted per-leg friction model is missing for a symbol; ``sanity_default`` does
not reproduce ``base``; a ``b5_k010_*`` variant is not narrower than ``base`` or ``b10_k160_1`` is not
wider (the knobs are not wired). FLAGGED (reported, not void): a 3-lot variant that never took more
than 1 lot ("multi-lot never triggered": it cannot meet the lot-share criterion), and variants with
different parameters but identical results ("inert knob", itself a finding).

DECISION RULE (fixed before any run; do not edit after seeing results). Per DTE, against ``base``, a
candidate is WORTH PURSUING only if at BOTH DTE levels:
  1. n >= 30 and n >= 0.85 x base n;
  2. PF >= base PF + 0.10 and net per position-day >= base;
  3. max drawdown <= 8% of the pool (the go-live gate) and <= 1.25 x base's;
  4. in >= 60% of the walk-forward windows that traded its window net >= base's, AND the point
     estimate of the net difference from a block bootstrap over windows is positive (the 80% interval
     is reported);
  5. friction share <= base + 10 points;
  6. a survivor of 1-5 repeats at ``friction_stress_multiplier`` 2.0 (``base`` rerun at 2.0 too) and
     still passes 1-3 at both DTE.
A multi-lot candidate is WORTH PHASE 2 only if it passes the above AND beats its own 1-lot twin (same
budget, same wing_k) on net at both DTE with max DD <= 1.25 x the twin's AND takes more than 1 lot in
>= 25% of its trades. Verdicts: PHASE 2 JUSTIFIED / NARROW-ONLY (a narrower wing_k or a higher budget
at 1 lot beats base; no Phase 2) / NOTHING BEATS BASELINE (keep 5%, 1.6, 1 lot) / VOID.

DEVIATION recorded in advance of the run: criterion 5's "friction share" is computed as (entry
crossing + round-trip commissions) / gross credit. The exit crossing is inside the exit price and is not
a separate field of a closed trade, so it is not in the share (a lower bound on the true friction).

LIMITS (printed in results.md). About 55-70 trades per variant over four years and about 14 windows:
only large effects are visible and "no difference" is the likely, valid answer. Prices are
Black-Scholes plus skew from stored daily IV (half the days carry stored IV), not real quotes. W64
measured real 2-wide credits of $0.62-$0.68, under the $0.70 floor, while the model gives $0.72-$0.77,
so 2-wide arms are biased favourably (the stress repeat bounds it). The absolute level is not trusted
(an earlier run of this harness under the legacy cost model put the shipped ruleset at PF 0.18 against
the live book's 2.2); read RELATIVE effects only. A result licenses a Phase 2 plan-mode pass or a config
proposal to the operator, never a live change.

Usage: python scripts/narrow_wing_study.py [--data-root <tree>] [--entry-dtes 18,25] [--n-jobs 2]
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT_BASE = ROOT / "reports" / "narrow_wing_study"
STRESS_MULTIPLIER = 2.0
BASELINE = "base"
SANITY = "sanity_default"

# decision-rule constants (pre-registered)
MIN_TRADES, MIN_N_RATIO = 30, 0.85
PF_MARGIN = 0.10
DD_POOL_CEILING_PCT, DD_BASE_MULT = 8.0, 1.25
MIN_WINDOW_SHARE = 0.60
FRICTION_MARGIN = 0.10
MIN_MULTILOT_SHARE_PCT = 25.0
BOOT_N, BOOT_SEED, BOOT_INTERVAL = 2000, 20261028, (10, 90)


@dataclass(frozen=True)
class Variant:
    budget: float          # fraction of the pool risked per trade (risk.max_position_risk_pct analogue)
    wing_k: float | None   # None = the config default
    lots: int              # max_contracts_per_trade
    role: str              # base | sanity | candidate | reference


VARIANTS: dict[str, Variant] = {
    "base": Variant(0.05, 1.6, 1, "base"),
    "sanity_default": Variant(0.05, None, 1, "sanity"),
    "b5_k015_1": Variant(0.05, 0.15, 1, "candidate"),
    "b5_k010_1": Variant(0.05, 0.10, 1, "candidate"),
    "b5_k010_3": Variant(0.05, 0.10, 3, "candidate"),
    "b10_k160_1": Variant(0.10, 1.6, 1, "candidate"),
    "b10_k030_1": Variant(0.10, 0.30, 1, "candidate"),
    "b10_k015_1": Variant(0.10, 0.15, 1, "candidate"),
    "b10_k010_1": Variant(0.10, 0.10, 1, "candidate"),
    "b10_k015_3": Variant(0.10, 0.15, 3, "candidate"),
    "b10_k010_3": Variant(0.10, 0.10, 3, "candidate"),
    "ref20_k160_1": Variant(0.20, 1.6, 1, "reference"),
}


def twin_of(name: str) -> str | None:
    """The 1-lot variant with the same budget and wing_k as a multi-lot variant (None for a 1-lot one)."""
    v = VARIANTS[name]
    if v.lots <= 1:
        return None
    for other, o in VARIANTS.items():
        if o.lots == 1 and o.budget == v.budget and o.wing_k == v.wing_k and o.role == "candidate":
            return other
    return None


def candidates() -> list[str]:
    return [n for n, v in VARIANTS.items() if v.role == "candidate"]


# --------------------------------------------------------------------------- pure statistics

def trade_stats(trades: list[dict]) -> dict:
    """Structure and lot statistics of a list of closed trades (dicts as the engine records them)."""
    if not trades:
        return {"mean_wing": None, "mean_max_loss": None, "mean_credit": None, "mean_lots": None,
                "lots_gt1_pct": 0.0, "contracts_max": 0, "friction_share": None}
    lots = [int(t.get("contracts", 1) or 1) for t in trades]
    wings = [float(t.get("wing_width", 0) or 0) for t in trades]
    mls = [float(t.get("max_loss_per_contract", 0) or 0) for t in trades]
    credits = [float(t.get("entry_price", 0) or 0) for t in trades]
    gross = cost = 0.0
    for t, n in zip(trades, lots):
        mid = t.get("mid_credit")
        if mid is None:
            continue
        gross += float(mid) * 100.0 * n
        cost += (float(mid) - float(t.get("entry_price", 0) or 0)) * 100.0 * n
        cost += 2.0 * float(t.get("entry_commission", 0) or 0)       # entry + a symmetric exit commission
    return {
        "mean_wing": round(sum(wings) / len(wings), 2),
        "mean_max_loss": round(sum(mls) / len(mls), 0),
        "mean_credit": round(sum(credits) / len(credits), 3),
        "mean_lots": round(sum(lots) / len(lots), 3),
        "lots_gt1_pct": round(100.0 * sum(1 for n in lots if n > 1) / len(lots), 1),
        "contracts_max": max(lots),
        "friction_share": round(cost / gross, 4) if gross > 0 else None,
    }


def window_nets(windows: list[tuple[str, list[dict]]]) -> dict[str, float]:
    """Net P&L per walk-forward window that traded (a window with no trade has no key)."""
    out: dict[str, float] = {}
    for wid, trades in windows:
        if trades:
            out[str(wid)] = round(sum(float(t.get("pnl", 0) or 0) for t in trades), 2)
    return out


def paired_window_stats(base_nets: dict[str, float], var_nets: dict[str, float],
                        n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> dict:
    """Window-paired comparison: share of traded windows where the variant >= base, and a block
    bootstrap (windows resampled with replacement) of the total net difference."""
    wins = sorted(set(base_nets) | set(var_nets))
    if not wins:
        return {"windows": 0, "share_ge": 0.0, "point": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    diffs = np.array([var_nets.get(w, 0.0) - base_nets.get(w, 0.0) for w in wins], dtype=float)
    share = float(np.mean([var_nets.get(w, 0.0) >= base_nets.get(w, 0.0) for w in wins]))
    rng = np.random.default_rng(seed)
    sums = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).sum(axis=1)
    lo, hi = np.percentile(sums, BOOT_INTERVAL)
    return {"windows": len(wins), "share_ge": round(share, 4), "point": round(float(diffs.sum()), 2),
            "ci_lo": round(float(lo), 2), "ci_hi": round(float(hi), 2)}


def dd_pct(row: dict) -> float:
    return 100.0 * float(row["max_dd_dollars"]) / float(row["pool"])


def criteria(row: dict, base: dict, paired: dict) -> dict[str, bool]:
    """The pre-registered criteria 1-5 of one candidate row against the baseline row (same DTE)."""
    bf, rf = base.get("friction_share"), row.get("friction_share")
    return {
        "1_n": row["n"] >= MIN_TRADES and row["n"] >= MIN_N_RATIO * base["n"],
        "2_pf_and_rate": row["pf"] >= base["pf"] + PF_MARGIN
        and row["net_per_position_day"] >= base["net_per_position_day"],
        "3_drawdown": dd_pct(row) <= DD_POOL_CEILING_PCT and dd_pct(row) <= DD_BASE_MULT * dd_pct(base),
        "4_windows": paired["share_ge"] >= MIN_WINDOW_SHARE and paired["point"] > 0,
        "5_friction": (rf is not None and bf is not None and rf <= bf + FRICTION_MARGIN),
    }


def criteria_1_to_3(row: dict, base: dict) -> dict[str, bool]:
    c = criteria(row, base, {"share_ge": 1.0, "point": 1.0})
    return {k: c[k] for k in ("1_n", "2_pf_and_rate", "3_drawdown")}


# --------------------------------------------------------------------------- checks and verdict

def wiring(rows_by_dte: dict[int, dict[str, dict]], friction_ok: bool) -> tuple[list[str], list[str]]:
    """(void reasons, flags). VOID reasons end the study; flags are reported."""
    void: list[str] = []
    flags: list[str] = []
    if not friction_ok:
        void.append("the fitted per-leg friction model is missing for a symbol (legacy % model would run)")
    for dte, rows in sorted(rows_by_dte.items()):
        base, san = rows.get(BASELINE), rows.get(SANITY)
        if base and san:
            diff = {k: (base[k], san[k]) for k in ("n", "net", "pf", "max_dd_dollars") if base[k] != san[k]}
            if diff:
                void.append(f"DTE {dte}: {SANITY} does not reproduce {BASELINE}: {diff}")
        if base:
            bw = base.get("mean_wing")
            for name in ("b5_k010_1", "b5_k010_3"):
                r = rows.get(name)
                if r and bw is not None and r.get("mean_wing") is not None and not r["mean_wing"] < bw:
                    void.append(f"DTE {dte}: {name} mean wing {r['mean_wing']} is not narrower than "
                                f"{BASELINE} {bw} (wing_k not wired)")
            r = rows.get("b10_k160_1")
            if r and bw is not None and r.get("mean_wing") is not None and not r["mean_wing"] > bw:
                void.append(f"DTE {dte}: b10_k160_1 mean wing {r['mean_wing']} is not wider than "
                            f"{BASELINE} {bw} (the budget is not wired)")
        for name, r in rows.items():
            if VARIANTS[name].lots > 1 and r.get("contracts_max", 0) < 2:
                flags.append(f"DTE {dte}: {name} multi-lot never triggered (max lots {r.get('contracts_max')})")
        names = sorted(rows)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                va, vb = VARIANTS[a], VARIANTS[b]
                if (va.budget, va.wing_k, va.lots) == (vb.budget, vb.wing_k, vb.lots):
                    continue
                key = ("n", "net", "pf", "max_dd_dollars")
                if all(rows[a][k] == rows[b][k] for k in key) and {a, b} != {BASELINE, SANITY}:
                    flags.append(f"DTE {dte}: {a} and {b} have identical results (inert knob)")
    return void, flags


def stage_a(rows_by_dte: dict[int, dict[str, dict]]) -> dict[str, dict]:
    """Criteria 1-5 for every candidate at every DTE; passes only if every DTE passes every criterion."""
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
    """Criterion 6: a survivor still passes 1-3 against the baseline rerun at the stressed friction."""
    out: dict[str, bool] = {}
    for name in survivors:
        ok = bool(stress_by_dte)
        for rows in stress_by_dte.values():
            if name not in rows or BASELINE not in rows or "error" in rows[name]:
                ok = False
                continue
            ok = ok and all(criteria_1_to_3(rows[name], rows[BASELINE]).values())
        out[name] = ok
    return out


def multilot_ok(name: str, rows_by_dte: dict[int, dict[str, dict]]) -> tuple[bool, list[str]]:
    """A multi-lot candidate must beat its own 1-lot twin and actually use the extra lots."""
    twin = twin_of(name)
    why: list[str] = []
    if twin is None:
        return False, ["no 1-lot twin"]
    for dte, rows in sorted(rows_by_dte.items()):
        r, t = rows.get(name), rows.get(twin)
        if not r or not t:
            why.append(f"DTE {dte}: variant or twin missing")
            continue
        if not r["net"] > t["net"]:
            why.append(f"DTE {dte}: net {r['net']} does not beat the 1-lot twin {twin} {t['net']}")
        if not dd_pct(r) <= DD_BASE_MULT * dd_pct(t):
            why.append(f"DTE {dte}: max DD {dd_pct(r):.2f}% > {DD_BASE_MULT}x the twin's {dd_pct(t):.2f}%")
        if not r.get("lots_gt1_pct", 0.0) >= MIN_MULTILOT_SHARE_PCT:
            why.append(f"DTE {dte}: only {r.get('lots_gt1_pct')}% of trades took more than 1 lot")
    return (not why), why


def verdict(rows_by_dte, stress_by_dte, friction_ok) -> dict:
    void, flags = wiring(rows_by_dte, friction_ok)
    if void:
        return {"verdict": "VOID", "void": void, "flags": flags, "worth": [], "notes": []}
    a = stage_a(rows_by_dte)
    survivors = [n for n, d in a.items() if d["pass"]]
    notes: list[str] = []
    if survivors and not stress_by_dte:
        notes.append("survivors of criteria 1-5 were NOT stress-tested (--skip-stress): not WORTH PURSUING")
        worth: list[str] = []
    else:
        b = stage_b(stress_by_dte, survivors)
        worth = [n for n in survivors if b.get(n)]
        for n in survivors:
            if not b.get(n):
                notes.append(f"{n}: passes 1-5 but fails the stressed repeat (criterion 6)")
    for n, d in a.items():
        if not d["pass"]:
            failed = sorted({k for dd in d["per_dte"].values() for k, ok in dd.get("criteria", {}).items()
                             if not ok})
            notes.append(f"{n}: fails {', '.join(failed) if failed else 'a missing run'}")
    phase2 = []
    for n in worth:
        if VARIANTS[n].lots > 1:
            ok, why = multilot_ok(n, rows_by_dte)
            (phase2.append(n) if ok else notes.append(f"{n}: worth pursuing but not for Phase 2 ({'; '.join(why)})"))
    if phase2:
        v = "PHASE 2 JUSTIFIED"
    elif worth:
        v = "NARROW-ONLY"
    else:
        v = "NOTHING BEATS BASELINE"
    return {"verdict": v, "void": [], "flags": flags, "worth": worth, "phase2": phase2,
            "stage_a": a, "notes": notes}


# --------------------------------------------------------------------------- running

def build_config(v: Variant, entry_dte: int, n_jobs: int, bc, stress: float | None):
    from ait.backtesting.walkforward import WalkForwardConfig
    return WalkForwardConfig(
        train_days=126, test_days=63, step_days=63, gap_days=5,
        train_window_models=False,
        initial_capital=None,                      # account.capital of the loaded settings ($5,000)
        position_size_pct=v.budget,
        wing_k=v.wing_k,
        max_contracts_per_trade=v.lots,
        wing_floor_dollars=bc.wing_floor_dollars,
        iv_floor=bc.iv_floor,
        delta_iv_scale=bc.delta_iv_scale,
        spread_base=bc.spread_base,
        spread_iv_sensitivity=bc.spread_iv_sensitivity,
        spread_dte_sensitivity=bc.spread_dte_sensitivity,
        spread_cap=bc.spread_cap,
        entry_dte=entry_dte,
        gates_mode="observe",
        friction_stress_multiplier=stress,
        optimize_n_jobs=n_jobs,
    )


async def run_variant(name, v, entry_dte, bc, data, n_jobs, out: Path, db_path, pool, stress=None) -> dict:
    from ait.backtesting.walkforward import WalkForwardBacktester
    from post_event_study import SYMBOLS, metrics
    cfg = build_config(v, entry_dte, n_jobs, bc, stress)
    bt = WalkForwardBacktester(SYMBOLS, ["iron_condor"], config=cfg, db_path=db_path,
                               progress_dir=out / name)
    t0 = time.time()
    result = await bt.run(data=data)
    trades = result._all_trades()
    row = {"variant": name, "entry_dte": entry_dte, "stress": stress, "pool": pool,
           "budget": v.budget, "wing_k": v.wing_k, "lots": v.lots, "role": v.role,
           "run_seconds": round(time.time() - t0), **metrics(result), **trade_stats(trades),
           "window_nets": window_nets([(w.window_id, w.backtest_result.trades) for w in result.windows])}
    row["dd_pool_pct"] = round(dd_pct(row), 2)
    (out / f"{name}.json").write_text(json.dumps(row, indent=1), encoding="utf-8")
    print(f"[nw] dte{entry_dte}{'' if stress is None else f' x{stress}'} {name}: n={row['n']} "
          f"PF={row['pf']} net={row['net']} DD%={row['dd_pool_pct']} wing={row['mean_wing']} "
          f"lots={row['mean_lots']} (>1: {row['lots_gt1_pct']}%) ({row['run_seconds']}s)", flush=True)
    return row


def friction_preflight(db_path: Path) -> bool:
    from ait.data.historical import HistoricalDataStore
    from post_event_study import SYMBOLS
    store = HistoricalDataStore(db_path=db_path)
    ok = True
    for sym in SYMBOLS:
        row = store.load_leg_spread_model(sym)
        print(f"[nw] friction model {sym}: " + (f"{row['form']} (n={row.get('n')}, "
              f"calibrated {row.get('calibrated_on')})" if row else "MISSING"), flush=True)
        ok = ok and bool(row)
    return ok


def render(rows_by_dte, stress_by_dte, result: dict) -> str:
    lines = ["# Narrow-wing and multi-lot study (W109)", ""]
    hdr = ("| variant | budget | wing_k | lots | n | win% | net $ | PF | DD $ | DD % pool | net/pos-day "
           "| mean wing | max loss ~$ | credit | mean lots | >1 lot % | friction |")
    for title, by in (("", rows_by_dte), (" stressed (friction x2)", stress_by_dte)):
        for dte, rows in sorted(by.items()):
            lines += [f"## Entry DTE {dte}{title}", "", hdr, "|" + "---|" * 17]
            for name, r in rows.items():
                if "error" in r:
                    lines.append(f"| {name} | ERROR {r['error'][:60]} |" + " |" * 15)
                    continue
                lines.append(
                    f"| {name} | {r['budget']:.0%} | {r['wing_k']} | {r['lots']} | {r['n']} | {r['win_rate_pct']} "
                    f"| {r['net']:,.0f} | {r['pf']} | {r['max_dd_dollars']:,.0f} | {r['dd_pool_pct']} "
                    f"| {r['net_per_position_day']} | {r['mean_wing']} | {r['mean_max_loss']} | {r['mean_credit']} "
                    f"| {r['mean_lots']} | {r['lots_gt1_pct']} | {r['friction_share']} |")
            lines.append("")
    lines += [f"**Verdict: {result['verdict']}**", ""]
    for key in ("void", "flags", "notes"):
        for item in result.get(key, []):
            lines.append(f"- {key.upper()}: {item}")
    if result.get("worth"):
        lines.append(f"- WORTH PURSUING: {', '.join(result['worth'])}")
    if result.get("phase2"):
        lines.append(f"- WORTH PHASE 2: {', '.join(result['phase2'])}")
    lines += ["", "Limits: ~55-70 trades per variant and ~14 windows (only large effects are visible); prices are "
              "Black-Scholes + skew from stored daily IV, not quotes; 2-wide arms are biased favourably "
              "(W64: real 2-wide credits $0.62-$0.68 vs the model's $0.72-$0.77); read RELATIVE effects only; "
              "friction share excludes the exit crossing.", ""]
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
    for needed in (BASELINE,):
        if needed not in names:
            names.insert(0, needed)
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
                print(f"[nw] dte{dte} {name} FAILED: {e}", flush=True)
                rows_by_dte[dte][name] = {"variant": name, "error": str(e)[:300]}

    stress_by_dte: dict[int, dict[str, dict]] = {}
    if not args.skip_stress:
        survivors = [n for n, d in stage_a({k: {n: r for n, r in v.items() if "error" not in r}
                                            for k, v in rows_by_dte.items()}).items() if d["pass"]]
        print(f"[nw] stress repeat for survivors of criteria 1-5: {survivors}", flush=True)
        if survivors:
            for dte in dtes:
                s_out = out / "stress_2x" / f"entry_dte_{dte}"
                s_out.mkdir(parents=True, exist_ok=True)
                stress_by_dte[dte] = {}
                for name in [BASELINE] + survivors:
                    stress_by_dte[dte][name] = await run_variant(
                        name, VARIANTS[name], dte, bc, data, args.n_jobs, s_out, db_path, pool,
                        stress=STRESS_MULTIPLIER)

    clean = {k: {n: r for n, r in v.items() if "error" not in r} for k, v in rows_by_dte.items()}
    result = verdict(clean, stress_by_dte, friction_ok)
    (out / "results.json").write_text(json.dumps(
        {"rows": rows_by_dte, "stress_rows": stress_by_dte, "result": result, "data": prov,
         "pool": pool, "entry_dtes": dtes}, indent=1, default=str), encoding="utf-8")
    md = render(rows_by_dte, stress_by_dte, result)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    asyncio.run(main())

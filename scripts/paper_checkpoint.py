"""Paper checkpoint: compare the paper account's real iron-condor closes with the walk-forward model.

PRE-REGISTERED in PLAN.md "W110 AS PLANNED" (2026-10-28), before the paper restart: the epoch, the
counted closes, the model figures, the checkpoints and the decision classes are fixed there, and this
script implements them without choosing anything new. Where the PLAN text and this file disagree the PLAN
text wins and this file is wrong.

READ-ONLY. ``data/ait_state.db`` is opened ``file:...?mode=ro``; nothing is written except the markdown
report under ``reports/paper_checkpoint/`` (statistics only, git-ignored).

POPULATION (the same definition the go-live scorecard uses). ``trades`` rows with strategy iron_condor and
status closed, minus the not-real closes (``ait.reporting.go_live.not_real_close_sql``), rolled chains folded
into one outcome each (``fold_roll_chains``), whose ORIGIN was entered at or after ``--epoch-start``
(default 2026-10-28, the earliest possible paper restart). The weekly scorecard is the whole-DB reference; this is the same
population restricted to the epoch.

EXIT-REASON MAPPING (live ``exit_reason_detailed`` -> the model's categories). The model's exits are
``touch_stop``, ``take_profit_short``, ``expiry_approaching`` and ``backtest_end`` (a forced close at a
walk-forward window end, which live never has and which is excluded from the comparison):
  ``short_strike_touch*``                    -> touch_stop
  ``take_profit_short*`` / ``take_profit*``  -> take_profit
  ``expiry_approaching*``                    -> expiry_approach
  contains ``expired`` (reconciler / settlement bookings) -> expiry_settled (live only)
  anything else (stop_loss, trailing_stop, condor_defense_*, assignment_risk_*, delta_breach, ...) -> other,
  listed verbatim in the report so the mapping can be corrected before the decision checkpoint.

CHECKPOINTS (W110). C1, a data-capture audit with NO decision, at 10 real closes or 12 weeks from the epoch
start. C2, the decision, at 25 real closes or 26 weeks; if fewer than 15 real closes exist then it is
extended until 15 closes or 52 weeks; still fewer than 15 at 52 weeks = PACE-LIMITED. Decision classes at C2
(n >= 15): MODEL TOO PESSIMISTIC AT 3-WIDE (net > 0 and PF >= 1.0); MODEL CONSISTENT (net < 0 and
PF <= 0.6); otherwise INDETERMINATE (continue to the 50-close go-live preliminary read). No go-live gate is
changed by anything printed here.

LIMITS. The model is Black-Scholes + skew from stored daily IV and its force-closed window ends are an
artefact, so the comparison uses the non-forced exits only. The live friction share excludes the exit
crossing (it is inside the exit price and not separable), as the model's does. A PF measured on about 25
closes has a wide interval; the 80% bootstrap interval is printed with it.

Usage: python scripts/paper_checkpoint.py [--checkpoint C1|C2] [--epoch-start 2026-10-28] [--now ISO]
       [--db data/ait_state.db] [--out reports/paper_checkpoint]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EPOCH_START_DEFAULT = "2026-10-28"
OUT_BASE = ROOT / "reports" / "paper_checkpoint"

# --- pre-registered thresholds (PLAN.md W110) -----------------------------------------------------
C1_CLOSES, C1_WEEKS = 10, 12
C2_CLOSES, C2_WEEKS, C2_EXT_WEEKS = 25, 26, 52
MIN_DECISION_CLOSES = 15
PF_PESSIMISTIC = 1.0        # real net > 0 and PF >= this -> the model is too pessimistic
PF_CONSISTENT = 0.6         # real net < 0 and PF <= this -> the model is consistent
BOOT_N, BOOT_SEED, BOOT_INTERVAL, PF_CAP = 2000, 20261029, (10, 90), 99.0

# --- the model figures (W109b `base`, entry DTE 18 / entry DTE 25), as written in PLAN.md W110 ----
MODEL = {
    "win_rate_pct": (39.4, 37.5),
    "pf": (0.41, 0.34),
    "mean_credit": (0.855, 0.838),
    "mean_wing": (3.0, 3.0),
    "mean_max_loss": (215.0, 216.0),
    "mean_hold_days": (8.5, 11.2),
    "friction_share_pct": (14.4, 15.3),
    "touch_stop_pct": (45.7, 51.5),
    "take_profit_pct": (49.4, 43.9),
    "expiry_approach_pct": (4.9, 4.5),
}

FILLED_STATUSES = ("filled", "partial", "closing")
# the columns this script reads; tests assert the real schema (ait.bot.state) still has them
TRADE_COLUMNS = ("trade_id", "symbol", "strategy", "status", "entry_time", "exit_time", "entry_price",
                 "quantity", "realized_pnl", "commission", "legs", "exit_reason_detailed",
                 "capital_at_risk", "rolled_from_trade_id")
EXECUTION_COLUMNS = ("trade_id", "price", "live_mid", "exec_time")


# --------------------------------------------------------------------------- pure functions

def classify_exit(reason: str | None) -> str:
    """Map a live ``exit_reason_detailed`` onto the model's exit categories (see the module docstring)."""
    r = (reason or "").strip().lower()
    if r.startswith("short_strike_touch"):
        return "touch_stop"
    if r.startswith("take_profit"):
        return "take_profit"
    if r.startswith("expiry_approaching"):
        return "expiry_approach"
    if "expired" in r:
        return "expiry_settled"
    return "other"


def parse_width(legs_json: str | None) -> float | None:
    """Wing width of a four-leg condor from ``trades.legs`` (None unless exactly two puts and two calls)."""
    try:
        legs = json.loads(legs_json or "[]")
        puts = [leg for leg in legs if leg.get("right") == "P"]
        calls = [leg for leg in legs if leg.get("right") == "C"]
        if len(legs) != 4 or len(puts) != 2 or len(calls) != 2:
            return None
        sp = max(float(leg["strike"]) for leg in puts if leg.get("action") == "SELL")
        lp = min(float(leg["strike"]) for leg in puts if leg.get("action") == "BUY")
        sc = min(float(leg["strike"]) for leg in calls if leg.get("action") == "SELL")
        lc = max(float(leg["strike"]) for leg in calls if leg.get("action") == "BUY")
    except (ValueError, TypeError, KeyError):
        return None
    width = max(sp - lp, lc - sc)
    return width if width > 0 else None


def _days_between(a: str | None, b: str | None) -> float | None:
    try:
        return (datetime.fromisoformat(str(b)) - datetime.fromisoformat(str(a))).total_seconds() / 86400.0
    except (ValueError, TypeError):
        return None


def profit_factor(pnls) -> float | None:
    """Gross profit / gross loss; None when there is no losing close (undefined, 'inf')."""
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    return gp / gl if gl > 0 else None


def bootstrap_pf_interval(pnls, n_boot: int = BOOT_N, seed: int = BOOT_SEED) -> tuple[float, float] | None:
    """80% interval of the PF with the closes resampled with replacement (seeded; an undefined PF is capped)."""
    arr = np.asarray(list(pnls), dtype=float)
    if len(arr) < 2:
        return None
    rng = np.random.default_rng(seed)
    s = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))]
    gp = np.where(s > 0, s, 0.0).sum(axis=1)
    gl = -np.where(s < 0, s, 0.0).sum(axis=1)
    pf = np.where(gl > 0, gp / np.where(gl > 0, gl, 1.0), np.where(gp > 0, PF_CAP, 0.0))
    lo, hi = np.percentile(np.minimum(pf, PF_CAP), BOOT_INTERVAL)
    return float(lo), float(hi)


def epoch_closes(rows: list[dict], epoch_start: str) -> tuple[list[dict], int]:
    """Real iron-condor closes whose ORIGIN was entered at/after the epoch, rolled chains folded.
    ``rows`` must already be the go-live population (status closed, not-real closes removed)."""
    from ait.reporting.go_live import fold_roll_chains
    folded, removed = fold_roll_chains(rows)
    n = len(epoch_start)
    keep = [r for r in folded if r.get("strategy") == "iron_condor"
            and str(r.get("entry_time") or "")[:n] >= epoch_start]
    return keep, removed


def metrics(closes: list[dict], executions: dict[str, float] | None = None) -> dict:
    """Descriptive statistics of the epoch's real closes. ``executions`` maps trade_id -> live mid at placement."""
    pnls = [float(c.get("realized_pnl") or 0.0) for c in closes]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    widths = [w for w in (parse_width(c.get("legs")) for c in closes) if w]
    credits = [float(c["entry_price"]) for c in closes if (c.get("entry_price") or 0) > 0]
    holds = [d for d in (_days_between(c.get("entry_time"), c.get("exit_time")) for c in closes) if d is not None]
    cars = [float(c["capital_at_risk"]) for c in closes if (c.get("capital_at_risk") or 0) > 0]
    cats: dict[str, int] = {}
    for c in closes:
        k = classify_exit(c.get("reason", c.get("exit_reason_detailed")))
        cats[k] = cats.get(k, 0) + 1
    pf = profit_factor(pnls)
    mid_gross = cost = 0.0
    for c in closes:                                            # live friction share (exit crossing not separable)
        mid = (executions or {}).get(c.get("trade_id"))
        if mid and (c.get("entry_price") or 0) > 0:
            q = int(c.get("quantity") or 1)
            mid_gross += mid * 100.0 * q
            cost += (mid - float(c["entry_price"])) * 100.0 * q + float(c.get("commission") or 0.0)
    return {
        "n": n, "wins": len(wins), "win_rate_pct": round(100.0 * len(wins) / n, 1) if n else None,
        "net": round(sum(pnls), 2), "pf": pf, "pf_interval": bootstrap_pf_interval(pnls),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
        "mean_hold_days": round(sum(holds) / len(holds), 2) if holds else None,
        "mean_wing": round(sum(widths) / len(widths), 2) if widths else None,
        "mean_credit": round(sum(credits) / len(credits), 3) if credits else None,
        "credit_over_width": round(sum(credits) / len(credits) / (sum(widths) / len(widths)), 3)
        if credits and widths else None,
        "mean_capital_at_risk": round(sum(cars) / len(cars), 0) if cars else None,
        "friction_share_pct": round(100.0 * cost / mid_gross, 1) if mid_gross > 0 else None,
        "exit_counts": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "exit_share_pct": {k: round(100.0 * v / n, 1) for k, v in cats.items()} if n else {},
    }


def fill_stats(trade_rows: list[dict]) -> dict:
    """Entries placed / filled / cancelled / rejected / pending among iron-condor rows (any status)."""
    from ait.reporting.go_live import is_real_close
    placed = len(trade_rows)
    counts = {"filled": 0, "cancelled": 0, "rejected": 0, "pending": 0}
    for r in trade_rows:
        st = str(r.get("status") or "").lower()
        reason = r.get("exit_reason_detailed") or r.get("reason") or ""
        if st in FILLED_STATUSES or (st == "closed" and is_real_close(reason)):
            counts["filled"] += 1
        elif st == "cancelled" or (st == "closed" and not is_real_close(reason)):
            counts["cancelled"] += 1
        elif st == "rejected":
            counts["rejected"] += 1
        else:
            counts["pending"] += 1
    decided = placed - counts["pending"]
    return {"placed": placed, **counts, "fill_rate_pct": round(100.0 * counts["filled"] / decided, 1) if decided else None}


def slip_stats(rows: list[dict]) -> dict:
    """Median |fill - live mid| as % of the credit (the go-live statistic) over the given entry executions."""
    vals = [abs(float(r["price"]) - float(r["live_mid"])) / float(r["entry_price"]) * 100.0
            for r in rows if (r.get("live_mid") or 0) > 0 and (r.get("entry_price") or 0) > 0]
    return {"n": len(vals), "median_pct": round(statistics.median(vals), 1) if vals else None}


def weeks_between(epoch_start: str, now: datetime) -> float:
    start = datetime.fromisoformat(epoch_start if "T" in epoch_start else epoch_start + "T00:00:00")
    if start.tzinfo is None and now.tzinfo is not None:
        start = start.replace(tzinfo=now.tzinfo)
    return max(0.0, (now - start).total_seconds() / (7 * 86400.0))


def checkpoint_status(which: str, n_closes: int, weeks: float) -> str:
    """NOT DUE / DUE / EXTENDED / PACE-LIMITED, per the W110 schedule."""
    if which == "C1":
        return "DUE" if (n_closes >= C1_CLOSES or weeks >= C1_WEEKS) else "NOT DUE"
    if n_closes < C2_CLOSES and weeks < C2_WEEKS:
        return "NOT DUE"
    if n_closes >= MIN_DECISION_CLOSES:
        return "DUE"
    return "PACE-LIMITED" if weeks >= C2_EXT_WEEKS else "EXTENDED"


def decide(n: int, net: float, pf: float | None) -> str:
    """The C2 decision class (n >= 15 real closes); ``pf`` None = no losing close."""
    if n < MIN_DECISION_CLOSES:
        return "INDETERMINATE (fewer than 15 real closes)"
    if net > 0 and (pf is None or pf >= PF_PESSIMISTIC):
        return "MODEL TOO PESSIMISTIC AT 3-WIDE"
    if net < 0 and pf is not None and pf <= PF_CONSISTENT:
        return "MODEL CONSISTENT"
    return "INDETERMINATE"


def capture_audit(closes: list[dict], executions: dict[str, float]) -> dict:
    """C1: how many closes lack a field the comparison needs, and which exit reasons the mapping calls 'other'."""
    def missing(pred):
        return sum(1 for c in closes if not pred(c))
    other: dict[str, int] = {}
    for c in closes:
        r = c.get("reason", c.get("exit_reason_detailed")) or ""
        if classify_exit(r) == "other":
            other[r or "(empty)"] = other.get(r or "(empty)", 0) + 1
    return {
        "closes": len(closes),
        "missing_entry_price": missing(lambda c: (c.get("entry_price") or 0) > 0),
        "missing_legs_4": missing(lambda c: parse_width(c.get("legs")) is not None),
        "missing_realized_pnl": missing(lambda c: c.get("realized_pnl") is not None),
        "missing_commission": missing(lambda c: c.get("commission") is not None),
        "missing_capital_at_risk": missing(lambda c: (c.get("capital_at_risk") or 0) > 0),
        "missing_exit_reason": missing(lambda c: bool((c.get("reason") or c.get("exit_reason_detailed") or "").strip())),
        "missing_entry_live_mid": missing(lambda c: (executions.get(c.get("trade_id")) or 0) > 0),
        "unmapped_exit_reasons": other,
    }


# --------------------------------------------------------------------------- database (read-only)

def open_ro(db: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def load(con: sqlite3.Connection) -> tuple[list[dict], list[dict], dict[str, float], list[dict]]:
    """(go-live closed population, every iron_condor trade row, entry live-mid by trade, entry executions)."""
    from ait.reporting.go_live import not_real_close_sql
    cols = ("trade_id, symbol, strategy, status, entry_time, exit_time, entry_price, quantity, realized_pnl, "
            "commission, legs, exit_reason_detailed, COALESCE(capital_at_risk,0) capital_at_risk, "
            "rolled_from_trade_id, COALESCE(exit_reason_detailed,'') reason")
    closed = [dict(r) for r in con.execute(
        f"SELECT {cols} FROM trades WHERE status='closed' AND strategy='iron_condor' {not_real_close_sql()} "
        "ORDER BY COALESCE(exit_time, entry_time)")]
    allic = [dict(r) for r in con.execute(f"SELECT {cols} FROM trades WHERE strategy='iron_condor'")]
    mids: dict[str, float] = {}
    execs = [dict(r) for r in con.execute(
        "SELECT e.trade_id, e.price, e.live_mid, e.exec_time, t.entry_price, t.entry_time "
        "FROM executions e JOIN trades t ON t.trade_id = e.trade_id "
        "WHERE t.strategy='iron_condor' ORDER BY e.exec_time")]
    for e in execs:
        if (e["live_mid"] or 0) > 0 and e["trade_id"] not in mids:
            mids[e["trade_id"]] = float(e["live_mid"])
    return closed, allic, mids, execs


# --------------------------------------------------------------------------- report

def _fmt(x, nd=2):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render(which: str, epoch_start: str, now: datetime, weeks: float, status: str, m: dict, fills: dict,
           pre: dict, slip: dict, audit: dict, decision: str | None, removed: int) -> str:
    pf = "inf" if m["pf"] is None and m["n"] else _fmt(m["pf"])
    iv = m["pf_interval"]
    L = [f"# Paper checkpoint {which} (W110)", "",
         f"epoch start {epoch_start}; as of {now.isoformat(timespec='minutes')}; {weeks:.1f} weeks; "
         f"**{which}: {status}**", "",
         "## Pace and fills (read this first)", "",
         f"- entries placed {fills['placed']}, filled {fills['filled']}, cancelled {fills['cancelled']}, "
         f"rejected {fills['rejected']}, pending {fills['pending']}; fill rate {_fmt(fills['fill_rate_pct'], 1)}%",
         f"- before the epoch (context): placed {pre['placed']}, filled {pre['filled']}, cancelled {pre['cancelled']}",
         f"- real closes {m['n']} ({m['n'] / weeks:.2f} per week; go-live pace target 3-4 per week)" if weeks > 0
         else f"- real closes {m['n']}",
         f"- median entry slip {_fmt(slip['median_pct'], 1)}% of the credit over {slip['n']} fills (go-live gate 8%)", "",
         "## Real closes against the model (entry DTE 18 / DTE 25)", "",
         "| metric | real | model |", "|---|---|---|",
         f"| closes | {m['n']} | |",
         f"| win rate % | {_fmt(m['win_rate_pct'], 1)} | {MODEL['win_rate_pct'][0]} / {MODEL['win_rate_pct'][1]} |",
         f"| net $ | {_fmt(m['net'])} | |",
         f"| profit factor | {pf}" + (f" (80% interval {iv[0]:.2f}-{iv[1]:.2f})" if iv else "")
         + f" | {MODEL['pf'][0]} / {MODEL['pf'][1]} |",
         f"| avg win / avg loss $ | {_fmt(m['avg_win'])} / {_fmt(m['avg_loss'])} | |",
         f"| mean credit $ | {_fmt(m['mean_credit'], 3)} | {MODEL['mean_credit'][0]} / {MODEL['mean_credit'][1]} |",
         f"| mean wing $ | {_fmt(m['mean_wing'])} | {MODEL['mean_wing'][0]} |",
         f"| credit / width | {_fmt(m['credit_over_width'], 3)} | about 0.28 |",
         f"| mean capital at risk $ | {_fmt(m['mean_capital_at_risk'], 0)} | {MODEL['mean_max_loss'][0]:.0f} |",
         f"| mean hold days | {_fmt(m['mean_hold_days'])} | {MODEL['mean_hold_days'][0]} / {MODEL['mean_hold_days'][1]} |",
         f"| friction share % (entry slip + commissions over gross credit) | {_fmt(m['friction_share_pct'], 1)} "
         f"| {MODEL['friction_share_pct'][0]} / {MODEL['friction_share_pct'][1]} |",
         f"| touch-stop share % of exits | {m['exit_share_pct'].get('touch_stop', 0.0)} "
         f"| {MODEL['touch_stop_pct'][0]} / {MODEL['touch_stop_pct'][1]} (non-forced) |",
         f"| take-profit share % | {m['exit_share_pct'].get('take_profit', 0.0)} "
         f"| {MODEL['take_profit_pct'][0]} / {MODEL['take_profit_pct'][1]} |",
         f"| expiry-approach share % | {m['exit_share_pct'].get('expiry_approach', 0.0)} "
         f"| {MODEL['expiry_approach_pct'][0]} / {MODEL['expiry_approach_pct'][1]} |", "",
         f"exit counts (live): {m['exit_counts']}", ""]
    if removed:
        L.append(f"(rolled-chain legs folded away: {removed})")
    if which == "C1":
        L += ["## Data-capture audit (C1, no decision)", ""]
        for k, v in audit.items():
            L.append(f"- {k}: {v}")
        L.append("")
    else:
        L += [f"## Decision (C2): **{decision}**", ""]
    L += ["Model figures are the W109b `base` rows (Black-Scholes + skew, forced window-end closes excluded from the "
          "exit shares); a PF on a few dozen closes has a wide interval; nothing here changes a go-live gate.", ""]
    return "\n".join(L) + "\n"


def run(db: Path, which: str, epoch_start: str, now: datetime) -> tuple[str, dict]:
    con = open_ro(db)
    try:
        closed, allic, mids, execs = load(con)
    finally:
        con.close()
    closes, removed = epoch_closes(closed, epoch_start)
    n = len(epoch_start)
    in_epoch = [r for r in allic if str(r.get("entry_time") or "")[:n] >= epoch_start]
    before = [r for r in allic if str(r.get("entry_time") or "")[:n] < epoch_start]
    m = metrics(closes, mids)
    fills, pre = fill_stats(in_epoch), fill_stats(before)
    slip = slip_stats([e for e in execs if str(e.get("entry_time") or "")[:n] >= epoch_start])
    weeks = weeks_between(epoch_start, now)
    status = checkpoint_status(which, m["n"], weeks)
    decision = decide(m["n"], m["net"], m["pf"]) if which == "C2" and status == "DUE" else (
        f"{status} (no decision)" if which == "C2" else None)
    audit = capture_audit(closes, mids) if which == "C1" else {}
    text = render(which, epoch_start, now, weeks, status, m, fills, pre, slip, audit, decision, removed)
    return text, {"metrics": m, "fills": fills, "status": status, "decision": decision, "weeks": weeks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=str(ROOT / "data" / "ait_state.db"))
    ap.add_argument("--epoch-start", default=EPOCH_START_DEFAULT)
    ap.add_argument("--checkpoint", choices=("C1", "C2"), default="C1")
    ap.add_argument("--now", default=None, help="ISO timestamp (default: the host clock)")
    ap.add_argument("--out", default=str(OUT_BASE))
    args = ap.parse_args(argv)
    now = datetime.fromisoformat(args.now) if args.now else datetime.now().astimezone()
    text, _ = run(Path(args.db), args.checkpoint, args.epoch_start, now)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{now.date().isoformat()}_{args.checkpoint}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

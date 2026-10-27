"""Post-FOMC entry blackout study (walk-forward).

QUESTION. Should iron-condor entries be blocked for N days AFTER a major
macro event, the way they are already blocked 1 day BEFORE one? Prompted by
2026-09-17: a QQQ condor entered the morning after FOMC — IV freshly crushed,
credit 4.99 on a 46-wide, IV rank 16.3 against a floor of 15 — and stopped out
two sessions later for -$503, the largest loss in the sample. In the live
record it is the ONLY entry made 1 day after an event (13 closes), and a 1-day
post-event buffer would have removed exactly that trade and nothing else —
which is what a rule fitted to one loss looks like. So: the whole history.

WHY FOMC ONLY. ait.data.economic_calendar knows 2026-2027 only, so the
engine's live pre-event gate is inactive on every pre-2026 window (documented
limitation). A post-event buffer tested through that calendar would be a
test of one year. This study passes its own FOMC decision days 2022-2026 to a
research-only engine gate (post_event_dates / post_event_blackout_days —
inert unless set; the live gate is untouched). FOMC is the event the trade
was after and the one whose dates are fixed years ahead on the Fed's
published schedule (federalreserve.gov/monetarypolicy/fomccalendars.htm).
The 2022-2025 dates below are transcribed from that schedule, not fetched;
the 2026 dates are read from the live calendar module. If a date is wrong the
buffer is off by that day for that meeting — verify against the source
before acting on a verdict.

HARNESS (live parity, the variable under test is the post-event gate only)
  - Identical to scripts/take_profit_study.py: SPY/QQQ/IWM, iron_condor only,
    train126/test63/step63/gap5, non-overlapping test windows, ungated
    entries (observe mode), the W5-corrected cost model, config wing_k and
    credit floor, entry DTE 18 (the live median).
  - The pre-event gate keeps its live behaviour (1 day before, 2026-2027
    calendar only) in every variant.

DECISION RULE (fixed before any run; do not edit after seeing results)
  A post-FOMC buffer of N days is adopted only if ALL hold against the
  no-buffer baseline:
    1. baseline n >= 30 and variant n >= 30;
    2. variant n >= 0.85 x baseline n (a buffer removes entries; pace is the
       binding constraint on the live verdict, so it may not cost more than
       15% of closes);
    3. PF > baseline PF + 0.10;
    4. net P&L per position-day >= baseline;
    5. max drawdown <= baseline max drawdown + 1.0 percentage point.
  Several qualify -> the SMALLEST N (fewest entries removed); ties -> higher
  PF. None qualify -> no buffer. Any change is a new live sample epoch and
  needs operator approval.

SANITY CHECK (within the run). "no_buffer_dates_only" passes the FOMC dates
with the blackout unset and must reproduce "no_buffer" exactly (n, PF, net,
max DD): the injection path is inert unless the window is set. A mismatch
voids the study. The take-profit study's live_ladder run at the same DTE is
ALSO reported, advisory only: load_daily_ohlcv takes a trailing 1500-day
window, so that run's data (..2026-09-14) and this one's differ by the
sessions since, and its windows do not line up with these.

WHAT THE HARNESS CANNOT TEST (registered, not hidden)
  - The live IV-rank floor: the engine's `iv_rank` feature is a realized-vol
    range rank (0-1), not the implied-vol percentile live gates on, so an
    "IV floor" arm would not measure the live rule. The engine has no IV
    history at all — it prices at a VIX-scaled realized vol.
  - The short-strike expected-move band: the engine places shorts by delta
    (0.20), which sits at ~0.84 expected moves by construction; an EM-min
    arm would need a different strike-placement rule, i.e. a different
    strategy.
  - Absolute level. The take-profit study measured this harness's live
    ruleset at PF 0.58 (DTE 14) / 0.45 (DTE 18) with touch stops on 40-47%
    of trades, while the live record stands at PF 1.79 with 2 touch stops in
    13. The simulation and the live book disagree on level; this study reads
    only the RELATIVE effect of the buffer, holding everything else fixed.

DATA SOURCE — GUARDED. The engine prices 2024-07 onward with the IBKR
implied vol stored in data/historical.db and reads its spread calibration
from the same file; load_daily_ohlcv falls back to Yahoo (no IV) for
anything the store lacks, and HistoricalDataStore CREATES an empty store if
the path is absent. A run from a checkout without the live data directory
therefore succeeds and returns a different harness (synthetic IV everywhere:
measured 2026-09-22, n 87 / PF 0.83 vs n 50 / PF 0.45 on the live store,
same code, same dates). --data-root names the tree whose data/ to use; the
run aborts unless every symbol carries stored implied vol on >= 400 rows.

Writes reports/post_event_study/entry_dte_<n>/<variant>.json, <variant>_summary.txt,
removed_by_<variant>.json (the baseline trades the buffer took away and their
P&L), results.json and results.md.

Usage: python scripts/post_event_study.py [--variants a,b] [--n-jobs 2] [--entry-dte 18]
                                         [--data-root <tree with data/historical.db>]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig  # noqa: E402
from ait.config.settings import load_settings  # noqa: E402

SYMBOLS = ["SPY", "QQQ", "IWM"]
OUT_BASE = ROOT / "reports" / "post_event_study"
MIN_STORED_IV_ROWS = 400
TP_BASE = ROOT / "reports" / "take_profit_study"
MIN_TRADES = 30
MIN_N_RATIO = 0.85

# FOMC decision days (second day of each two-day meeting), transcribed from
# the Fed's published schedule. 2026 is read from the live calendar module.
FOMC_HISTORICAL = [
    date(2022, 7, 27), date(2022, 9, 21), date(2022, 11, 2), date(2022, 12, 14),
    date(2023, 2, 1), date(2023, 3, 22), date(2023, 5, 3), date(2023, 6, 14),
    date(2023, 7, 26), date(2023, 9, 20), date(2023, 11, 1), date(2023, 12, 13),
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
]

VARIANTS: dict[str, int | None] = {
    "no_buffer": None,      # baseline = live behaviour
    "no_buffer_dates_only": None,   # sanity: dates without a window must be inert
    "post_fomc_1d": 1,      # blocks the event day and the day after (the 09-17 shape)
    "post_fomc_2d": 2,
    "post_fomc_3d": 3,
}


def fomc_dates() -> tuple[str, ...]:
    from ait.data.economic_calendar import _FOMC_2026
    all_dates = sorted(set(FOMC_HISTORICAL) | set(_FOMC_2026))
    return tuple(d.isoformat() for d in all_dates)


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


def trade_keys(result) -> dict[tuple, dict]:
    out = {}
    for t in result._all_trades():
        key = (str(t.get("entry_date", ""))[:10], str(t.get("symbol", "")),
               round(float(t.get("pnl", 0.0) or 0.0), 2))
        out[key] = {"entry_date": key[0], "exit_date": str(t.get("exit_date", ""))[:10],
                    "symbol": key[1], "pnl": key[2], "exit_reason": t.get("exit_reason")}
    return out


async def run_variant(name, n_days, bc, data, n_jobs, entry_dte, out, db_path) -> tuple[dict, object]:
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
        entry_dte=entry_dte,
        post_event_blackout_days=n_days,
        post_event_dates=fomc_dates() if (n_days is not None or name == "no_buffer_dates_only") else None,
        optimize_n_jobs=n_jobs,
    )
    bt = WalkForwardBacktester(SYMBOLS, ["iron_condor"], config=cfg,
                               db_path=db_path,
                               progress_dir=out / name)
    t0 = time.time()
    result = await bt.run(data=data)
    m = {"variant": name, "post_fomc_blackout_days": n_days, "entry_dte": entry_dte,
         "run_seconds": round(time.time() - t0), **metrics(result)}
    (out / f"{name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    try:
        (out / f"{name}_summary.txt").write_text(result.summary(), encoding="utf-8")
    except Exception:  # noqa: BLE001 — the summary is a convenience copy
        pass
    print(f"[pe] {name}: n={m['n']} PF={m['pf']} net={m['net']} "
          f"net/pos-day={m['net_per_position_day']} DD%={m['max_dd_pct']} "
          f"({m['run_seconds']}s)", flush=True)
    return m, result


def load_data(db_path: Path) -> tuple[dict, dict]:
    """Frames per symbol + provenance. Aborts on a store without IBKR IV."""
    from ait.data.market_data import load_daily_ohlcv
    if not db_path.exists():
        raise SystemExit(f"[pe] no data store at {db_path}; pass --data-root")
    data, prov = {}, {}
    for sym in SYMBOLS:
        df = load_daily_ohlcv(sym, days=1500, db_path=db_path)
        if df is None or not len(df):
            raise SystemExit(f"[pe] no daily bars for {sym}")
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
        iv_rows = int(df["implied_vol"].notna().sum()) if "implied_vol" in df.columns else 0
        if iv_rows < MIN_STORED_IV_ROWS:
            raise SystemExit(
                f"[pe] {sym}: only {iv_rows} rows carry stored implied vol (need "
                f">= {MIN_STORED_IV_ROWS}); this is not the live-parity store — "
                f"pass --data-root <tree with the IBKR-backfilled data/historical.db>")
        data[sym] = df
        prov[sym] = {"rows": len(df), "first": str(df.index[0].date()),
                     "last": str(df.index[-1].date()), "stored_iv_rows": iv_rows}
        print(f"[pe] data {sym}: {len(df)} rows {prov[sym]['first']}..{prov[sym]['last']} "
              f"| stored IV rows {iv_rows}", flush=True)
    return data, prov


def take_profit_reference(base: dict, entry_dte) -> str:
    """Advisory: the take-profit study's live_ladder at this DTE (its data window
    ended 2026-09-14; this run's is later, so windows differ — see docstring)."""
    run = f"entry_dte_{entry_dte}" if entry_dte else "entry_dte_config"
    ref_path = TP_BASE / run / "live_ladder.json"
    if not ref_path.exists():
        return f"take-profit reference: none at {ref_path}"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    keys = ("n", "pf", "net", "max_dd_dollars")
    return ("take-profit study live_ladder (advisory, different data window): "
            + ", ".join(f"{k} {ref[k]} vs {base[k]} here" for k in keys))


def apply_rule(rows: list[dict], entry_dte) -> tuple[str, list[str]]:
    by = {r["variant"]: r for r in rows if "error" not in r}
    base = by.get("no_buffer")
    notes: list[str] = []
    if base is None:
        return "VOID: baseline failed", notes
    explicit = by.get("no_buffer_dates_only")
    if explicit is not None:
        keys = ("n", "net", "pf", "max_dd_dollars")
        diff = {k: (base[k], explicit[k]) for k in keys if explicit[k] != base[k]}
        if diff:
            return f"VOID: no_buffer_dates_only does not reproduce no_buffer {diff}", notes
        notes.append("sanity: dates without a window reproduce the baseline exactly")
    notes.append(take_profit_reference(base, entry_dte))
    if base["n"] < MIN_TRADES:
        return f"INCONCLUSIVE: baseline n={base['n']} < {MIN_TRADES}", notes
    qualifiers = []
    for name, r in by.items():
        if name in ("no_buffer", "no_buffer_dates_only"):
            continue
        checks = {
            "n>=30": r["n"] >= MIN_TRADES,
            f"n>={MIN_N_RATIO:.2f}x base": r["n"] >= MIN_N_RATIO * base["n"],
            "pf>base+0.10": r["pf"] > base["pf"] + 0.10,
            "net/pos-day>=base": r["net_per_position_day"] >= base["net_per_position_day"],
            "dd<=base+1pp": r["max_dd_pct"] <= base["max_dd_pct"] + 1.0,
        }
        failed = [k for k, ok in checks.items() if not ok]
        notes.append(f"{name}: {'QUALIFIES' if not failed else 'fails ' + ', '.join(failed)}")
        if not failed:
            qualifiers.append(r)
    if not qualifiers:
        return "NO BUFFER (no variant clears every condition)", notes
    best = min(qualifiers, key=lambda r: (r["post_fomc_blackout_days"], -r["pf"]))
    return (f"CANDIDATE {best['variant']} (PF {best['pf']} vs {base['pf']}, n {best['n']} vs "
            f"{base['n']}); live change needs operator approval and starts a new sample epoch"), notes


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
        print(f"[pe] could not lower priority: {e}", flush=True)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--entry-dte", type=int, default=18,
                    help="live median entry DTE (take_profit_study ran 14 and 18)")
    ap.add_argument("--data-root", default=str(ROOT),
                    help="tree whose data/historical.db holds the IBKR daily IV backfill "
                         "and the spread calibration (default: this checkout)")
    ap.add_argument("--tag", default="",
                    help="suffix for the output directory (a re-run under a different "
                         "engine switch, e.g. AIT_BT_EM_GATE=1, must not overwrite the "
                         "recorded results)")
    args = ap.parse_args()
    db_path = Path(args.data_root) / "data" / "historical.db"
    # Two engine paths resolve the store RELATIVE TO CWD regardless of db_path
    # (Backtester._load_calibrated_spreads -> HistoricalDataStore() default;
    # load_settings() inside Backtester when no settings are passed), so run
    # from the data root. Outputs go to OUT_BASE, which is absolute.
    import os
    os.chdir(args.data_root)
    print(f"[pe] cwd -> {os.getcwd()}", flush=True)
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    settings = load_settings(str(ROOT / "config.yaml"))
    lower_priority()
    out = OUT_BASE / (f"entry_dte_{args.entry_dte}" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)
    print(f"[pe] entry_dte={args.entry_dte} -> {out} | FOMC dates: {len(fomc_dates())} "
          f"({fomc_dates()[0]}..{fomc_dates()[-1]})", flush=True)
    bc = settings.backtest
    data, prov = load_data(db_path)
    print(f"[pe] data store: {db_path}", flush=True)
    rows = []
    base_trades = None
    for name in names:
        try:
            m, result = await run_variant(name, VARIANTS[name], bc, data, args.n_jobs,
                                          args.entry_dte, out, db_path)
            rows.append(m)
            keys = trade_keys(result)
            if name == "no_buffer":
                base_trades = keys
            elif name != "no_buffer_dates_only" and base_trades is not None:
                removed = [v for k, v in base_trades.items() if k not in keys]
                added = [v for k, v in keys.items() if k not in base_trades]
                m["removed_trades"] = len(removed)
                m["removed_pnl"] = round(sum(t["pnl"] for t in removed), 2)
                m["added_trades"] = len(added)
                m["added_pnl"] = round(sum(t["pnl"] for t in added), 2)
                (out / f"removed_by_{name}.json").write_text(
                    json.dumps({"removed": removed, "added": added}, indent=1), encoding="utf-8")
                (out / f"{name}.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[pe] {name} FAILED: {e}", flush=True)
            rows.append({"variant": name, "error": str(e)[:300]})

    verdict, notes = apply_rule(rows, args.entry_dte)
    (out / "results.json").write_text(
        json.dumps({"rows": rows, "verdict": verdict, "notes": notes,
                    "data_store": str(db_path), "data": prov}, indent=1),
        encoding="utf-8")

    header = ("| variant | n | win% | net $ | PF | avg $ | max DD $ | DD % | hold d "
              "| net $/pos-day | touch | TP | removed (P&L) | added (P&L) |")
    lines = [header, "|" + "---|" * 14]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['variant']} | ERROR: {r['error'][:80]} |" + " |" * 12)
            continue
        er = r.get("exit_reasons", {})
        rem = f"{r.get('removed_trades', '-')} ({r.get('removed_pnl', '-')})" if "removed_trades" in r else "-"
        add = f"{r.get('added_trades', '-')} ({r.get('added_pnl', '-')})" if "added_trades" in r else "-"
        lines.append(
            f"| {r['variant']} | {r['n']} | {r['win_rate_pct']} | {r['net']:,.0f} | {r['pf']} "
            f"| {r['avg_trade']} | {r['max_dd_dollars']:,.0f} | {r['max_dd_pct']} "
            f"| {r['mean_hold_days']} | {r['net_per_position_day']} "
            f"| {er.get('touch_stop', 0)} | {er.get('take_profit_short', 0)} | {rem} | {add} |")
    lines += ["", f"**Verdict:** {verdict}", ""] + [f"- {n}" for n in notes]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

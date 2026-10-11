#!/usr/bin/env python
"""W72 parity replay: run known real trades through the research engine's pricing
and exit code and report where research and reality disagree.

    python scripts/parity_replay.py                       # defaults below
    python scripts/parity_replay.py --trades data/replay/manual_trades_20260817_20261009.json

Read-only: data/historical.db and data/ait_state.db are opened ``mode=ro``; the
engine used for the exit scan runs with legacy friction and calibrated spreads
off, so no store is touched. Outputs go under data/replay/out/<YYYYMMDD>/
(git-ignored): legs.csv + trades.csv carry per-trade prices and must stay local;
summary.md carries statistics only.

The tolerance block is ``parity:`` in config_research.yaml (ParityConfig).
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()  # before any trading-subsystem import (CLAUDE.md)

import argparse  # noqa: E402
import csv  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _fmt(x, nd=3):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def _pct(x):
    return "n/a" if x is None else f"{100 * x:+.1f}%"


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trades", default=str(ROOT / "data/replay/manual_trades_20260817_20261009.json"))
    ap.add_argument("--historical-db", default=str(ROOT / "data/historical.db"))
    ap.add_argument("--state-db", default=str(ROOT / "data/ait_state.db"))
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--out-dir", default=None,
                    help="default: data/replay/out/<YYYYMMDD> (git-ignored)")
    args = ap.parse_args(argv)

    from ait.backtesting import replay as rp
    from ait.config.research import use_research_config
    from ait.config.settings import load_settings

    use_research_config()
    settings = load_settings()
    params = rp.PricingParams.from_backtest_config(settings.backtest)
    pc = settings.parity
    edges = list(pc.dte_band_upper_edges)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "data/replay/out" / date.today().strftime("%Y%m%d")
    # Privacy guard (checked BEFORE anything is created): per-trade prices must
    # never be written somewhere this repo's git can track.
    try:
        rel = out_dir.resolve().relative_to(ROOT)
        if rel.parts[:2] != ("data", "replay"):
            print(f"refusing to write per-trade outputs outside data/replay: {rel}", file=sys.stderr)
            return 2
    except ValueError:
        pass  # outside the repo entirely: not tracked by this repo's git
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = [t for t in rp.load_trades(args.trades) if True]
    data = rp.ResearchData(args.historical_db, args.symbol)
    friction = data.leg_spread_model()

    trade_rows: list[dict] = []
    leg_rows: list[dict] = []
    verdict_rows: list[dict] = []
    for t in trades:
        if t.entry_time is None:
            continue
        entry = rp.entry_replay(t, data, params, friction)
        exit_ = rp.exit_replay(t, data, params, friction, entry)
        scan = rp.engine_exit_scan(t, data, entry) if t.condor_roles() is not None else None
        ctx = rp.load_bot_exec_context(args.state_db, t.bot_trade_id) if t.bot_trade_id else None
        band = rp.band_label(t.entry_dte, edges)

        row = {
            "id": t.id, "origin": t.origin, "structure": t.structure, "entry_time": t.entry_time.isoformat(),
            "entry_dte": t.entry_dte, "band": band, "eligible": t.parity_eligible,
            "qty": t.quantity, "still_open": t.still_open, "actual_credit": round(t.actual_credit, 4),
        }
        for v in rp.VARIANTS:
            e = entry.get(v)
            for k in ("S", "iv", "credit_mid", "friction_total", "credit_fill", "err", "err_pct"):
                row[f"{v}_{k}"] = None if e is None else round(float(e[k]), 5)
            if e is not None:
                verdict_rows.append({"band": band, "variant": v, "err_pct": e["err_pct"],
                                     "eligible": t.parity_eligible})
                for i, l in enumerate(t.legs):
                    leg_rows.append({
                        "trade": t.id, "leg": i, "variant": v, "side": l.side, "right": l.right,
                        "strike": l.strike, "expiry": l.expiry.isoformat(), "band": band,
                        "moneyness_pct": round(abs(l.strike / e["S"] - 1.0) * 100, 3),
                        "actual_open": l.open_price, "model_mid": round(e["leg_mid"][i], 4),
                        "model_fill": round(e["leg_fill"][i], 4), "err": round(e["leg_err"][i], 4),
                        "model_leg_iv": round(e["leg_model_iv"][i], 5),
                        "actual_iv": (None if e["leg_actual_iv"][i] is None
                                      else round(e["leg_actual_iv"][i], 5)),
                        "base_iv_used": round(e["iv"], 5),
                        # market leg IV minus the IV the model priced THAT leg at
                        # (base IV + the engine's skew): the pricing error in vol points
                        "iv_gap": (None if e["leg_actual_iv"][i] is None
                                   else round(e["leg_actual_iv"][i] - e["leg_model_iv"][i], 5)),
                    })
        if exit_ is not None:
            row["actual_exit_value"] = round(exit_["actual_exit_value"], 4)
            for v in rp.VARIANTS:
                x = exit_.get(v)
                row[f"{v}_exit_err"] = None if x is None else round(x["err"], 4)
        if scan is not None:
            row.update(scan)
        row["actual_exit_codes"] = ";".join(sorted({l.close_code for l in t.legs if l.close_code}))
        row["actual_last_close"] = max((l.close_time for l in t.legs if l.close_time), default=None)
        if ctx:
            lm, fill = ctx.get("live_mid"), ctx.get("fill")
            row["bot_live_mid"] = lm
            row["bot_signal_price"] = ctx.get("signal_price")
            row["bot_realized_friction"] = None if lm is None or fill is None else round(abs(lm - fill), 4)
        trade_rows.append(row)

    def _write(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        cols: list[str] = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        with open(out_dir / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    _write("trades.csv", trade_rows)
    _write("legs.csv", leg_rows)

    verdicts = rp.summarise_bands(
        verdict_rows, edges, error_max=pc.credit_error_pct_max,
        bias_max=pc.credit_bias_pct_max, min_trades=pc.min_trades_per_band)

    L: list[str] = []
    L.append(f"# Parity replay summary ({date.today().isoformat()})")
    L.append("")
    L.append("Statistics only; per-trade prices are in trades.csv / legs.csv (git-ignored).")
    L.append("Error = model - actual; positive = research is RICHER than reality.")
    L.append("")
    L.append(f"- symbol {args.symbol}; structures replayed {len(trade_rows)} "
             f"(verdict-eligible: {sum(1 for r in trade_rows if r['eligible'])})")
    L.append(f"- friction model: {'fitted per-leg ' + friction.form if friction else 'ABSENT (no half-spread applied)'}")
    L.append(f"- tolerances: median |err| <= {pc.credit_error_pct_max:.0%}, |median err| <= "
             f"{pc.credit_bias_pct_max:.0%}, min trades per band {pc.min_trades_per_band}")
    L.append("")
    L.append("## Entry credit parity by DTE band (relative error of the modelled fill vs the actual fill)")
    L.append("")
    L.append("| band | variant | n | median abs err | median signed err | verdict |")
    L.append("|---|---|---|---|---|---|")
    for v in verdicts:
        L.append(f"| {v.band} | {v.variant} | {v.n} | {_pct(v.median_abs_pct)} | "
                 f"{_pct(v.median_signed_pct)} | {v.verdict} |")
    L.append("")
    L.append("## Leg-level entry price error ($ per share, modelled fill - actual fill), variant 'clean'")
    L.append("")
    L.append("| moneyness (|K/S-1|) | n legs | median signed | median abs |")
    L.append("|---|---|---|---|")
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 100)):
        xs = [r["err"] for r in leg_rows if r["variant"] == "clean" and lo <= r["moneyness_pct"] < hi]
        lab = f"{lo}-{hi}%" if hi < 100 else f"{lo}%+"
        L.append(f"| {lab} | {len(xs)} | {_fmt(_median(xs))} | {_fmt(_median([abs(x) for x in xs]))} |")
    L.append("")
    L.append("## Implied-IV gap by DTE band: IV the market priced each leg at minus the IV the model "
             "priced that leg at, i.e. stored IV + engine skew (vol points; negative = the model is "
             "RICHER than the market)")
    L.append("")
    L.append("| band | variant | side | n legs | median gap | median abs gap |")
    L.append("|---|---|---|---|---|---|")
    for band in rp.band_order(edges):
        for v in rp.VARIANTS:
            for right, lab in (("C", "calls"), ("P", "puts")):
                xs = [r["iv_gap"] for r in leg_rows if r["band"] == band and r["variant"] == v
                      and r["right"] == right and r["iv_gap"] is not None
                      and any(t.id == r["trade"] and t.parity_eligible for t in trades)]
                if xs:
                    L.append(f"| {band} | {v} | {lab} | {len(xs)} | {100*_median(xs):+.1f} | "
                             f"{100*_median([abs(x) for x in xs]):.1f} |")
    L.append("")
    L.append("## Exit decision parity (engine's own _check_exit vs what happened)")
    L.append("")
    scanned = [r for r in trade_rows if "engine_exit_reason" in r]
    same_day = [r for r in scanned if r.get("engine_exit_date") and r.get("actual_last_close")
                and r["engine_exit_date"] == str(r["actual_last_close"])[:10]]
    L.append(f"- condors scanned: {len(scanned)}; engine fires an exit on the actual exit date: {len(same_day)}")
    reasons: dict[str, int] = {}
    for r in scanned:
        reasons[str(r.get("engine_exit_reason"))] = reasons.get(str(r.get("engine_exit_reason")), 0) + 1
    L.append(f"- engine exit reasons: {reasons}")
    early = [r for r in scanned if r.get("engine_exit_date") and r.get("actual_last_close")
             and r["engine_exit_date"] < str(r["actual_last_close"])[:10]]
    L.append(f"- engine would have exited EARLIER than the actual last close in {len(early)} of {len(scanned)}")
    L.append("")
    L.append("## Exit value parity (same-instant trader closes only)")
    L.append("")
    for v in rp.VARIANTS:
        xs = [r.get(f"{v}_exit_err") for r in trade_rows]
        xs = [x for x in xs if x is not None]
        L.append(f"- {v}: n={len(xs)} median signed error {_fmt(_median(xs))} $/share")
    L.append("")
    L.append("## Bot execution context (live friction actually paid)")
    L.append("")
    for r in trade_rows:
        if r.get("bot_realized_friction") is not None:
            model = r.get("clean_friction_total")
            L.append(f"- bot trade: realized |live_mid - fill| = {r['bot_realized_friction']:.3f} $/share "
                     f"vs modelled friction {model if model is not None else 'n/a'}")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")

    print("\n".join(L))
    print(f"\nwritten: {out_dir}/ (trades.csv, legs.csv, summary.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

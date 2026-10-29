#!/usr/bin/env python
"""W71: is the research IV-rank RISE veto a gate the live bot actually has?

The research engine blocks a neutral credit entry when the FeatureEngine ``iv_rank`` (a
realized-vol range rank, 0-1) rose by more than ``iv_rank_rise_threshold`` over 10 rows. Live has
no such veto: it gates on the LEVEL of the stored-IV percentile (floor 15, cap 85). This probe
measures, per symbol, how the two ranks relate and how much of what the research veto blocks is
a day live would have traded anyway.

    python scripts/iv_rank_parity_probe.py                 # series comparison only (seconds)
    python scripts/iv_rank_parity_probe.py --ab --workers 6   # + realistic-engine A/B of the veto

Read-only: ``data/historical.db`` is opened ``mode=ro`` (no migrations, nothing written). Prints
statistics only. The A/B uses ``bs_legacy`` pricing, per-leg friction with the short-DTE floor
ASSERTED loaded, 1 contract, entry DTE 14 and ``condor_any_direction`` (as every short-DTE study
so far, so a trade count exists to compare).
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()  # before any trading-subsystem import (docs/TECHNICAL_GUIDE.md "Conventions")

import argparse  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "historical.db"
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def _series(symbol: str, db: Path, prefix: str = "archive_w91_research_"):
    """(ohlcv+iv frame, research IV, live-store IV, feature rank, S_research, S_live)."""
    import pandas as pd

    from ait.backtesting.replay import ResearchData
    from ait.data.iv_rank import iv_rank_series
    from ait.ml.features import FeatureEngine

    # W91: this W71 probe compares the research_* table with the live one; after the tables were merged the two are
    # identical by construction, so it only has meaning on a database that still has both sets. W106: after the W91
    # archive the old set is `archive_w91_research_*` (the default); `--research-prefix research_` before it.
    frame = ResearchData(str(db), symbol, prefix=prefix).daily_frame()
    research_iv = frame["implied_vol"]
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as c:
        live = pd.read_sql_query(
            "SELECT date, implied_vol FROM daily_prices WHERE symbol = ? ORDER BY date",
            c, params=(symbol,))
    live_iv = pd.Series(live["implied_vol"].values, index=pd.to_datetime(live["date"]), dtype=float)
    # The feature rank comes from FeatureEngine's own IV-proxy block (a function of Close alone), so
    # it IS the series the engine's rise veto reads; calling the block directly skips the other
    # ~100 features, which take tens of minutes over 4,000 rows. A test pins it to compute().
    f = FeatureEngine()._add_iv_features(frame[OHLCV].copy())["iv_rank"].astype(float).dropna()
    return frame, research_iv, live_iv, f, iv_rank_series(research_iv), iv_rank_series(live_iv)


def _fmt(x, nd=2):
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def compare(symbols: list[str], db: Path, threshold: float, since: str,
            prefix: str = "archive_w91_research_") -> None:
    import pandas as pd

    from ait.backtesting import iv_rank_compare as cmp

    print(f"\n## 1. Research rise veto (feature rank rise > {threshold:g} over {cmp.rise_rows()} rows) "
          f"vs live's level gates (floor {cmp.live_floor():g} / cap {cmp.live_cap():g}), research IV\n")
    print("| symbol | span | days | spearman(F, S) | opposite side | veto days | already blocked by live | "
          "EXTRA (live would trade) | rise on stored IV | both fire |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows2, rows3 = [], []
    for sym in symbols:
        frame, riv, liv, f, s_res, s_live = _series(sym, db, prefix)
        for label, mask in (("full history", slice(None)), (f"{since}+", slice(since, None))):
            r = cmp.veto_vs_live(f.loc[mask], s_res.loc[mask], threshold)
            print(f"| {sym} | {label} | {r['n_days']} | {_fmt(r['spearman_feature_vs_stored'])} | "
                  f"{r['side_disagreement_days']} ({r['side_disagreement_days'] / max(r['n_days'], 1):.0%}) | "
                  f"{r['research_veto_days']} ({r['research_veto_share']:.1%}) | {r['already_blocked_by_live']} | "
                  f"{r['extra_restriction_days']} ({_fmt(r['extra_share_of_tradeable_days'], 3)} of tradeable days) | "
                  f"{r['rise_on_stored_iv_days']} | {r['veto_agreement']['both']} |")
        r2 = cmp.veto_vs_live(f.loc[since:], s_live.loc[since:], threshold)
        rows2.append((sym, r2))
        both = pd.concat([riv, liv], axis=1, keys=["research", "live"]).dropna()
        both = both.loc[since:]
        rk = pd.concat([s_res, s_live], axis=1, keys=["research", "live"]).dropna().loc[since:]
        rows3.append((sym, len(liv.dropna()), len(both),
                      float((both["research"] - both["live"]).abs().median()) if len(both) else None,
                      float((both["research"] - both["live"]).abs().quantile(0.95)) if len(both) else None,
                      int(((both["research"] - both["live"]).abs() > 0.005).sum()) if len(both) else None,
                      float(both["research"].corr(both["live"])) if len(both) > 10 else None,
                      float(rk["research"].corr(rk["live"], method="spearman")) if len(rk) > 10 else None))

    print(f"\n## 2. The same on LIVE's stored-IV series ({since}+; this is the series the live gates read)\n")
    print("| symbol | days | spearman(F, S_live) | veto days | already blocked by live | EXTRA | live level-block days |")
    print("|---|---|---|---|---|---|---|")
    for sym, r in rows2:
        print(f"| {sym} | {r['n_days']} | {_fmt(r['spearman_feature_vs_stored'])} | {r['research_veto_days']} | "
              f"{r['already_blocked_by_live']} | {r['extra_restriction_days']} | {r['live_level_block_days']} |")

    print(f"\n## 3. Is the research IV the same quantity as live's stored IV? (overlap {since}+)\n")
    print("| symbol | live observations (all) | overlapping days | median abs diff (vol pts) | p95 abs diff "
          "(vol pts) | days differing by > 0.5 vol pt | level corr | rank corr (spearman) |")
    print("|---|---|---|---|---|---|---|---|")
    for sym, n_live, n_ov, mad, p95, nbig, lc, rc in rows3:
        print(f"| {sym} | {n_live} | {n_ov} | {_fmt(None if mad is None else mad * 100)} | "
              f"{_fmt(None if p95 is None else p95 * 100)} | {nbig} | {_fmt(lc, 3)} | {_fmt(rc, 3)} |")


def _arm(task: dict) -> dict:
    warnings.filterwarnings("ignore")
    from ait.backtesting.engine import Backtester
    from ait.backtesting.study_harness import assert_friction_loaded, study_engine_kwargs, study_inputs

    # features from a year of warm-up before the window, as the walk-forward does per window
    rd, data, features, ivr = study_inputs(DB, task["symbol"], task["start"], task["end"])
    # W87a: the study engine block is defined once (backtesting/study_harness.py); entry DTE fixed at 14
    bt = Backtester(**study_engine_kwargs(
        data=data, symbol=task["symbol"], db_path=DB, intraday_store=rd, entry_dte=14,
        features_cache=features, iv_rank_series=ivr,
        iv_rank_rise_gate=bool(task["gate"]), iv_rank_rise_threshold=task["threshold"]))
    assert_friction_loaded(bt)
    tr = bt.run().trades
    pnl = [float(t.get("pnl", 0.0)) for t in tr]
    f = dict(bt._funnel)
    return {"symbol": task["symbol"], "threshold": task["threshold"], "gate": task["gate"], "n": len(tr),
            "win": (sum(p > 0 for p in pnl) / len(pnl)) if pnl else 0.0, "total": round(sum(pnl), 2),
            "rise_days": f.get("iv_rank_rise", 0), "rank_floor_days": f.get("iv_rank_floor", 0),
            "rank_cap_days": f.get("iv_rank_cap", 0), "days": f.get("days_evaluated", 0)}


def ab(symbols: list[str], threshold: float, start: str, end: str, workers: int) -> None:
    tasks = [{"symbol": s, "threshold": threshold, "gate": g, "start": start, "end": end}
             for s in symbols for g in (True, False)]
    print(f"\n## 4. Realistic-engine A/B of the rise veto ({start}..{end}, entry DTE 14, 1 contract)\n")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_arm, tasks))
    print("| symbol | veto | trades | win % | total $ | days vetoed by the rise gate | days at live floor | days at live cap |")
    print("|---|---|---|---|---|---|---|---|")
    for r in res:
        arm = f"ON {r['threshold']:g}" if r["gate"] else "OFF"
        print(f"| {r['symbol']} | {arm} | {r['n']} | {r['win']:.0%} | {r['total']:,.0f} | {r['rise_days']} | "
              f"{r['rank_floor_days']} | {r['rank_cap_days']} |")
    print("\nA difference of a few $k over ~60-120 trades per arm is inside the noise; the decision rests on parity.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbols", nargs="+", default=["QQQ", "SPY", "IWM"])
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--since", default="2024-04-22", help="start of the live-store overlap")
    ap.add_argument("--ab", action="store_true", help="also run the engine A/B (minutes)")
    ap.add_argument("--ab-start", default="2024-01-02")
    ap.add_argument("--ab-end", default="2026-10-08")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--research-prefix", default="archive_w91_research_",
                    help="W106: the pre-W91 research table set to compare with the live one "
                         "(archive_w91_research_ after the W91 archive, research_ before it)")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    table = f"{args.research_prefix}daily_prices"
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as c:
        present = c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if not present:
        print(f"{table} is not in {args.db}: this probe compares the pre-W91 research tables with the live "
              f"ones and needs a database that still has them (see --research-prefix).", file=sys.stderr)
        return 2
    compare(args.symbols, args.db, args.threshold, args.since, args.research_prefix)
    if args.ab:
        ab(args.symbols, args.threshold, args.ab_start, args.ab_end, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())

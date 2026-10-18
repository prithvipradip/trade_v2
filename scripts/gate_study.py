"""W67: entry-day outcome study for the fractal regime gate (read-only on the databases).

Walk-forward OOS yields ~10-30 trades per run: far too few to tell two gate cutoffs apart.
This study instead asks the question of EVERY eligible entry day. It runs the bare engine in
``gates_mode="observe"`` (no ML range gate, no fractal veto -- live's behaviour today) with
overlapping hypothetical condors (``max_concurrent_positions`` large; capital scaled so the
per-trade budget, wing and contract count are the same as the $5k construction), so each
eligible day produces one condor with its full-life P&L and its entry-time fractal features.
It then reports conditional P&L by fractal bucket and what P&L would have been had a given
veto cutoff been enforced, with a CLUSTERED (monthly block) bootstrap because overlapping
positions are strongly autocorrelated.

Verdict rule (fixed before looking at any number): a cutoff is "supported" only if keeping
the days at or below it beats the days above it by more than the bootstrap interval on the
POOLED sample AND the per-symbol sign agrees on every symbol that has >= MIN_EXCLUDED days
above the cutoff. Otherwise: "not distinguishable at this n". It never reports a tuned cutoff.

The overlapping-positions construct is a research device, NOT a portfolio simulation.
The range-model probability arm (needs a per-window trained predictor) is not part of this
script: it is registered in PLAN.md for a later wave.

    python scripts/gate_study.py [--symbols SPY QQQ IWM] [--any-direction] [--n-boot 2000]
Writes reports/gate_study_<date>.csv (one row per hypothetical condor).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
warnings.filterwarnings("ignore")

MIN_EXCLUDED = 8          # per-symbol days above a cutoff needed for its sign to count
CUTOFF_QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90)
OVERLAP_SLOTS = 40        # hypothetical condors that may be open at once


def block_bootstrap_diff(df: pd.DataFrame, keep: pd.Series, n_boot: int = 2000,
                         seed: int = 7, cluster_col: str = "month") -> dict:
    """Mean(P&L | keep) - mean(P&L | ~keep) with a cluster bootstrap 90% interval.

    Whole clusters (entry months) are resampled so the interval reflects the real
    information content of overlapping positions, not n independent trades."""
    keep = keep.astype(bool).reindex(df.index)
    kept, excl = df[keep], df[~keep]
    if len(kept) == 0 or len(excl) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_kept": int(len(kept)), "n_excl": int(len(excl))}
    point = float(kept["pnl"].mean() - excl["pnl"].mean())
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    groups = {c: g for c, g in df.assign(_keep=keep).groupby(cluster_col)}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        samp = pd.concat([groups[c] for c in pick])
        k, e = samp[samp["_keep"]], samp[~samp["_keep"]]
        if len(k) and len(e):
            diffs.append(k["pnl"].mean() - e["pnl"].mean())
    lo, hi = (np.percentile(diffs, [5, 95]) if diffs else (float("nan"), float("nan")))
    return {"diff": point, "lo": float(lo), "hi": float(hi),
            "n_kept": int(len(kept)), "n_excl": int(len(excl))}


def cutoff_verdict(df: pd.DataFrame, cutoff: float, feature: str = "hurst_spread",
                   n_boot: int = 2000, min_excluded: int = MIN_EXCLUDED) -> dict:
    """Is vetoing days with ``feature > cutoff`` supported? (rule in the module docstring)"""
    pooled = block_bootstrap_diff(df, df[feature] <= cutoff, n_boot=n_boot)
    per_symbol = {}
    signs_ok = True
    for sym, g in df.groupby("symbol"):
        r = block_bootstrap_diff(g, g[feature] <= cutoff, n_boot=200)
        per_symbol[sym] = r
        if r["n_excl"] >= min_excluded and not (r["diff"] > 0):
            signs_ok = False
    supported = bool(pooled["lo"] > 0 and signs_ok and pooled["n_excl"] >= min_excluded)
    return {"cutoff": float(cutoff), "pooled": pooled, "per_symbol": per_symbol,
            "signs_agree": signs_ok, "supported": supported,
            "verdict": "SUPPORTED" if supported else "not distinguishable at this n"}


def bucket_table(df: pd.DataFrame, feature: str, q: int = 4) -> pd.DataFrame:
    bins = pd.qcut(df[feature], q=q, duplicates="drop")
    g = df.groupby(bins, observed=True)["pnl"]
    out = g.agg(n="size", mean_pnl="mean", win_pct=lambda x: 100.0 * (x > 0).mean(),
                avg_win=lambda x: x[x > 0].mean() if (x > 0).any() else 0.0,
                avg_loss=lambda x: x[x <= 0].mean() if (x <= 0).any() else 0.0)
    return out.round(2)


def collect_trades(symbols, db_path: Path, table_prefix: str, start: str,
                   any_direction: bool) -> pd.DataFrame:
    from ait.backtesting.engine import Backtester
    from ait.config.runtime_env import apply_runtime_env_defaults, research_capital
    apply_runtime_env_defaults()
    from ait.config.research import use_research_config
    use_research_config()   # W68: config.yaml + config_research.yaml (research only)
    from ait.config.settings import load_settings
    from ait.data.economic_calendar import build_research_calendar
    from ait.data.market_data import load_daily_ohlcv
    from ait.ml.features import FeatureEngine

    settings = load_settings()
    cap = float(research_capital())
    pct = float(settings.risk.max_position_risk_pct)
    cal = build_research_calendar(db_path, table_prefix, list(settings.backtest.macro_event_types))
    rows = []
    for sym in symbols:
        df = load_daily_ohlcv(sym, days=1100, db_path=db_path, table_prefix=table_prefix,
                              start_date=date.fromisoformat(start))
        feats = FeatureEngine().compute(df)
        bt = Backtester(
            data=df, strategies=["iron_condor"],
            # scale capital so budget (capital x pct) is the real $5k construction's
            initial_capital=cap * OVERLAP_SLOTS, position_size_pct=pct / OVERLAP_SLOTS,
            max_concurrent_positions=OVERLAP_SLOTS, budget_aware_wings=True,
            max_contracts_per_trade=1, wing_floor_dollars=2.0, symbol=sym,
            economic_calendar=cal, features_cache=feats, db_path=db_path,
            allow_live_model_fallback=False, condor_any_direction=any_direction,
            friction_model="per_leg_usd", gates_mode="observe", fractal_veto_spread=0.0,
            # W68: live's IV-rank floor/cap, so the study sees the days live would trade
            iv_rank_gate=bool(settings.backtest.iv_rank_gate))
        res = bt.run()
        for t in res.trades:
            fg = (t.get("decision") or {}).get("fractal_gate", {})
            rows.append({"symbol": sym, "entry_date": t["entry_date"], "pnl": float(t["pnl"]),
                         "exit_reason": t.get("exit_reason"),
                         "entry_price": t.get("entry_price"),
                         "hurst_spread": fg.get("hurst_spread"),
                         "multifractal_width": fg.get("multifractal_width"),
                         "entry_iv": t.get("entry_iv")})
        print(f"[{sym}] {len(res.trades)} hypothetical condors; funnel "
              f"{ {k: v for k, v in res.entry_funnel.items() if not k.startswith('builder:')} }")
    out = pd.DataFrame(rows).dropna(subset=["hurst_spread"])
    out["month"] = out["entry_date"].astype(str).str[:7]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--db-path", default="data/historical.db")
    ap.add_argument("--table-prefix", default="research_")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--any-direction", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    df = collect_trades(args.symbols, Path(args.db_path), args.table_prefix, args.start,
                        args.any_direction)
    if df.empty:
        print("no trades collected")
        return 1
    print(f"\n{len(df)} hypothetical condors, {df['month'].nunique()} entry months "
          f"(overlapping positions: a research construct, not a portfolio simulation)")
    print(f"overall: mean ${df.pnl.mean():.1f}  win {100 * (df.pnl > 0).mean():.0f}%  "
          f"total ${df.pnl.sum():,.0f}")
    for feat in ("hurst_spread", "multifractal_width"):
        print(f"\n--- by {feat} quartile (pooled) ---")
        print(bucket_table(df, feat).to_string())
    print("\n--- veto-cutoff sweep on hurst_spread (kept = at or below cutoff) ---")
    print(f"{'quantile':>8s} {'cutoff':>7s} {'n_kept':>6s} {'n_excl':>6s} {'diff$':>7s} "
          f"{'90% interval':>16s}  verdict")
    for q in CUTOFF_QUANTILES:
        c = float(df["hurst_spread"].quantile(q))
        v = cutoff_verdict(df, c, n_boot=args.n_boot)
        p = v["pooled"]
        print(f"{q:8.2f} {c:7.3f} {p['n_kept']:6d} {p['n_excl']:6d} {p['diff']:7.1f} "
              f"[{p['lo']:6.1f},{p['hi']:6.1f}]  {v['verdict']}"
              + ("" if v["signs_agree"] else "  (per-symbol signs disagree)"))
    out = _root / "reports" / f"gate_study_{date.today():%Y%m%d}_{'anydir' if args.any_direction else 'neutral'}.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nper-trade table: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

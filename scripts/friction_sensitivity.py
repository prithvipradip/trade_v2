"""W66: how much does the transaction-cost model matter? (and the payoff-shape A/B)

Runs the BARE research engine (features computed ONCE per symbol, ~7 s per variant) on the
research history and sweeps:

  1. friction multiples of the fitted per-leg dollar model (0, 0.25, 0.5, 1, 2) next to the
     legacy % model, and reports the BREAK-EVEN multiple (where total P&L crosses zero);
  2. with --exit-ab: the payoff-shape experiment on the exit policy (take-profit fraction of
     credit x touch stop on/off) at the fitted friction, to be PRESENTED to the operator.
     No live setting changes: this is evidence for a decision, not a decision.

Read-only on the databases. Needs the research tables (W65) and a fitted
option_leg_spread_params row per symbol (scripts/calibrate_option_spreads.py --source ib-bars).

    python scripts/friction_sensitivity.py --symbols SPY QQQ IWM [--any-direction] [--exit-ab]
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import warnings
from datetime import date
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

warnings.filterwarnings("ignore")

MULTIPLES = (0.0, 0.25, 0.5, 1.0, 2.0)
TP_FRACTIONS = (0.25, 0.35, 0.50, 0.75, None)  # None = hold to the DTE exit (no take-profit)


def summarize(trades: list[dict]) -> dict:
    pnl = [t["pnl"] for t in trades]
    n = len(pnl)
    return {"n": n, "win": (100.0 * sum(p > 0 for p in pnl) / n) if n else 0.0,
            "pnl": float(sum(pnl)), "avg": float(st.mean(pnl)) if n else 0.0,
            "avg_win": float(st.mean([p for p in pnl if p > 0])) if any(p > 0 for p in pnl) else 0.0,
            "avg_loss": float(st.mean([p for p in pnl if p <= 0])) if any(p <= 0 for p in pnl) else 0.0}


def break_even_multiple(points: list[tuple[float, float]]) -> float | None:
    """Linear interpolation of the friction multiple at which total P&L crosses zero.

    `points` = [(multiple, pnl)] ascending in multiple. None when P&L never changes sign
    inside the swept range (reported as '>max' / '<min' by the caller)."""
    for (m0, p0), (m1, p1) in zip(points, points[1:]):
        if p0 == 0:
            return m0
        if (p0 > 0) != (p1 > 0) and p1 != p0:
            return m0 + (0 - p0) * (m1 - m0) / (p1 - p0)
    return None


def _engine_kwargs(sym, df, feats, cal, args, **over):
    kw = dict(data=df, strategies=["iron_condor"], initial_capital=args.capital,
              position_size_pct=args.budget_pct, budget_aware_wings=True,
              max_contracts_per_trade=1, wing_floor_dollars=2.0,
              allow_live_model_fallback=False, symbol=sym, economic_calendar=cal,
              hurst_hard_veto_multiplier=0.0, features_cache=feats,
              condor_any_direction=args.any_direction,
              iv_rank_gate=args.iv_rank_gate, db_path=Path(args.db_path))
    kw.update(over)
    return kw


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM"])
    ap.add_argument("--db-path", default="data/historical.db")
    ap.add_argument("--table-prefix", default="research_")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--capital", type=float, default=None,
                    help="default: config account.capital")
    ap.add_argument("--budget-pct", type=float, default=None,
                    help="default: config risk.max_position_risk_pct")
    ap.add_argument("--any-direction", action="store_true",
                    help="condor_any_direction (live-like selector)")
    ap.add_argument("--exit-ab", action="store_true", help="also run the payoff-shape A/B")
    ap.add_argument("--iv-rank-gate", action=argparse.BooleanOptionalAction, default=None,
                    help="live's IV-rank floor/cap (default: config backtest.iv_rank_gate)")
    args = ap.parse_args(argv)

    from ait.config.runtime_env import apply_runtime_env_defaults, research_capital
    apply_runtime_env_defaults()
    from ait.config.research import use_research_config
    use_research_config()   # W68: config.yaml + config_research.yaml (research only)
    from ait.backtesting.engine import Backtester
    from ait.config.settings import load_settings
    from ait.data.economic_calendar import build_research_calendar
    from ait.data.historical import HistoricalDataStore
    from ait.data.market_data import load_daily_ohlcv
    from ait.ml.features import FeatureEngine

    settings = load_settings()
    args.capital = float(args.capital or research_capital())
    if args.iv_rank_gate is None:
        args.iv_rank_gate = bool(settings.backtest.iv_rank_gate)
    args.budget_pct = float(args.budget_pct or settings.risk.max_position_risk_pct)
    db = Path(args.db_path)
    cal = build_research_calendar(db, args.table_prefix, list(settings.backtest.macro_event_types))
    store = HistoricalDataStore(db_path=db)
    print(f"capital=${args.capital:,.0f} budget={args.budget_pct:.1%} any_direction="
          f"{args.any_direction} start={args.start}")

    totals: dict[float, float] = {}
    for sym in args.symbols:
        model = store.load_leg_spread_model(sym)
        if model is None:
            print(f"[{sym}] NO fitted friction model: run calibrate_option_spreads.py "
                  f"--source ib-bars first")
            continue
        df = load_daily_ohlcv(sym, days=1100, db_path=db, table_prefix=args.table_prefix,
                              start_date=date.fromisoformat(args.start))
        feats = FeatureEngine().compute(df)
        print(f"\n=== {sym}  model={model['form']} a={model['a']:.4f} b={model['b']:.4f} "
              f"(n={model['n']}, sessions={model['n_sessions']}, cvMAE=${model['cv_mae']:.4f}) ===")
        print(f"{'variant':26s} {'n':>3s} {'win%':>5s} {'pnl$':>8s} {'avg$':>7s} "
              f"{'avgW$':>7s} {'avgL$':>7s}")
        pts = []
        legacy = summarize(Backtester(**_engine_kwargs(sym, df, feats, cal, args)).run().trades)
        print(f"{'legacy % model':26s} {legacy['n']:3d} {legacy['win']:5.0f} {legacy['pnl']:8.0f} "
              f"{legacy['avg']:7.1f} {legacy['avg_win']:7.1f} {legacy['avg_loss']:7.1f}")
        for m in MULTIPLES:
            r = summarize(Backtester(**_engine_kwargs(
                sym, df, feats, cal, args, friction_model="per_leg_usd",
                friction_stress_multiplier=m)).run().trades)
            pts.append((m, r["pnl"]))
            totals[m] = totals.get(m, 0.0) + r["pnl"]
            print(f"{'per_leg_usd x' + format(m, 'g'):26s} {r['n']:3d} {r['win']:5.0f} "
                  f"{r['pnl']:8.0f} {r['avg']:7.1f} {r['avg_win']:7.1f} {r['avg_loss']:7.1f}")
        be = break_even_multiple(pts)
        print(f"break-even friction multiple: "
              f"{'%.2fx' % be if be is not None else 'not crossed in 0..2x (P&L %s)' % ('>0' if pts[-1][1] > 0 else '<0')}")

        if args.exit_ab:
            print(f"\n  exit-policy A/B on {sym} at the fitted friction (x1):")
            print(f"  {'TP frac of credit':20s} {'touch':>5s} {'n':>3s} {'win%':>5s} "
                  f"{'pnl$':>8s} {'avgW$':>7s} {'avgL$':>7s}")
            for touch in ("1", "0"):
                os.environ["AIT_BT_TOUCH_STOP"] = touch
                for frac in TP_FRACTIONS:
                    short = float("inf") if frac is None else frac
                    ladder = [(20, 1.0, short), (10, 0.75, short), (5, 0.5, short),
                              (-(10 ** 9), 0.25, short)]
                    r = summarize(Backtester(**_engine_kwargs(
                        sym, df, feats, cal, args, friction_model="per_leg_usd",
                        credit_tp_ladder=ladder)).run().trades)
                    print(f"  {('hold to DTE exit' if frac is None else format(frac, '.0%')):20s} "
                          f"{'on' if touch == '1' else 'off':>5s} {r['n']:3d} {r['win']:5.0f} "
                          f"{r['pnl']:8.0f} {r['avg_win']:7.1f} {r['avg_loss']:7.1f}")
            os.environ["AIT_BT_TOUCH_STOP"] = "1"

    if totals:
        print("\nALL SYMBOLS total P&L by friction multiple: "
              + "  ".join(f"x{m:g}=${v:,.0f}" for m, v in sorted(totals.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())

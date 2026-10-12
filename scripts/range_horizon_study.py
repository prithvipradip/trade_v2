#!/usr/bin/env python
"""W78: does the research range model carry information at each horizon?

For each rolling (train, test) window of a symbol's daily history and each label horizon h,
this trains a ``RangePredictor`` exactly as the walk-forward does (adaptive threshold for
that horizon, ML-only, per-window) and scores it OUT OF SAMPLE:

  * AUROC and Brier skill vs climatology (does the probability rank in-range days at all?)
  * the in-range rate when P(in range) >= the live floor, and the share of days that pass it
  * a reliability table (P deciles vs realised in-range rate)
  * the floor that would hold the SAME pass-rate as the floor does at the base horizon
  * seconds to train one model (the cost W79 pays per bucket per window)

Two comparisons decide items the W78 plan deliberately did not change:
  * unit slip: the horizon the engine uses today (calendar DTE as bars) vs the same DTE
    converted to trading days (``round(h_cal * 5 / 7)``)
  * short horizons: is there any skill at 1-3 bars (the 0-DTE / 1-DTE question)?

Read-only: ``data/historical.db`` is opened ``mode=ro``; model artifacts go to a temporary
directory that is deleted afterwards. Nothing under ``models/`` or ``data/`` is written.

    python scripts/range_horizon_study.py --symbol QQQ --max-windows 8
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()  # before any trading-subsystem import (CLAUDE.md)

import argparse  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HORIZONS = (1, 2, 3, 5, 6, 9, 14, 18, 25, 30)
# calendar-day DTE -> (horizon the engine uses today, horizon if converted to trading days)
UNIT_SLIP_DTES = (14, 30)


def _windows(n: int, train: int, test: int, step: int, max_windows: int) -> list[tuple[int, int, int, int]]:
    out, start = [], 0
    while start + train + test <= n:
        out.append((start, start + train, start + train, start + train + test))
        start += step
    return out[-max_windows:] if max_windows else out


def _labels(close, h: int, thr: float):
    import numpy as np
    n = len(close)
    y = np.full(n, np.nan)
    for t in range(n - h):
        base = close[t]
        y[t] = 1.0 if np.max(np.abs(close[t + 1: t + 1 + h] / base - 1.0)) < thr else 0.0
    return y


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--historical-db", default=str(ROOT / "data/historical.db"))
    ap.add_argument("--train-days", type=int, default=252)
    ap.add_argument("--test-days", type=int, default=63)
    ap.add_argument("--step-days", type=int, default=63)
    ap.add_argument("--max-windows", type=int, default=8, help="most recent N windows (0 = all)")
    ap.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    ap.add_argument("--floor", type=float, default=None, help="range floor (default ml.range_min_confidence)")
    ap.add_argument("--base-horizon", type=int, default=9)
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from ait.backtesting.replay import ResearchData
    from ait.backtesting.walkforward import WalkForwardBacktester
    from ait.config.research import use_research_config
    from ait.config.settings import load_settings
    from ait.ml.features import FeatureEngine
    from ait.ml.range_predictor import RangePredictor

    use_research_config()
    floor = args.floor if args.floor is not None else float(load_settings().ml.range_min_confidence)
    frame = ResearchData(args.historical_db, args.symbol).daily_frame()
    ohlcv = frame[["Open", "High", "Low", "Close", "Volume"]].dropna()
    wins = _windows(len(ohlcv), args.train_days, args.test_days, args.step_days, args.max_windows)
    if not wins:
        print(f"not enough history for {args.symbol}: {len(ohlcv)} bars", file=sys.stderr)
        return 1
    print(f"{args.symbol}: {len(ohlcv)} bars, {len(wins)} windows (train {args.train_days} / test "
          f"{args.test_days} / step {args.step_days}), floor {floor}")

    tmp = Path(tempfile.mkdtemp(prefix="w78_range_study_"))
    per_h: dict[int, dict] = {h: {"auc": [], "skill": [], "pass": [], "prec": [], "secs": [], "thr": [],
                                  "p": [], "y": [], "base": [], "win": [], "skips": {}}
                              for h in args.horizons}
    try:
        for wi, (a, b, c, d) in enumerate(wins, 1):
            train_df, test_df = ohlcv.iloc[a:b], ohlcv.iloc[c:d]
            ctx = ohlcv.iloc[max(0, a):d]
            feats = FeatureEngine().compute(ctx)
            for h in args.horizons:
                thr = WalkForwardBacktester._adaptive_range_threshold(train_df, horizon_days=h)
                rp = RangePredictor(threshold_pct=thr, horizon_days=h, model_dir=tmp / f"w{wi}_h{h}")
                t0 = time.perf_counter()
                accs = rp.train(train_df, symbol=args.symbol, intraday_store=None)
                secs = time.perf_counter() - t0
                if not accs or not rp.is_trained:
                    sk = per_h[h]["skips"]
                    sk["not_trained"] = sk.get("not_trained", 0) + 1
                    continue
                close = test_df["Close"].to_numpy(dtype=float)
                y = _labels(close, h, thr)
                ps, ys = [], []
                n_none = n_nofeat = 0
                for k in range(len(test_df) - h):
                    idx = test_df.index[k]
                    if idx not in feats.index:
                        n_nofeat += 1
                        continue
                    pred = rp.predict_from_features(feats.loc[idx], symbol=args.symbol)
                    if pred is None:
                        n_none += 1      # the ensemble ABSTAINED (no member has CV edge)
                        continue
                    ps.append(float(pred.probability_in_range))
                    ys.append(y[k])
                if len(ys) < 10 or len(set(ys)) < 2:
                    sk = per_h[h]["skips"]
                    if len(ys) >= 10:
                        why = "single_class_oos"
                    elif n_none >= max(1, n_nofeat) and n_none > 0:
                        why = "abstained_no_cv_edge"
                    else:
                        why = "no_feature_rows"
                    sk[why] = sk.get(why, 0) + 1
                    continue
                p, yy = np.asarray(ps), np.asarray(ys)
                base = float(yy.mean())
                brier = float(np.mean((p - yy) ** 2))
                bb = base * (1 - base)
                r = per_h[h]
                r["auc"].append(float(roc_auc_score(yy, p)))
                r["skill"].append(1.0 - brier / bb if bb > 0 else float("nan"))
                r["pass"].append(float((p >= floor).mean()))
                r["prec"].append(float(yy[p >= floor].mean()) if (p >= floor).any() else float("nan"))
                r["secs"].append(secs)
                r["thr"].append(thr)
                r["p"].extend(ps)
                r["y"].extend(ys)
                r["base"].append(base)
                r["win"].append(wi)
            print(f"  window {wi}/{len(wins)} done", flush=True)

        def med(xs):
            xs = [x for x in xs if x == x]
            return statistics.median(xs) if xs else float("nan")

        common = set(range(1, len(wins) + 1))
        for h in args.horizons:
            common &= set(per_h[h]["win"])
        print(f"\nwindows where EVERY horizon trained and scored: {len(common)} of {len(wins)}")
        for h in args.horizons:
            print(f"  h={h}: scored {len(per_h[h]['win'])}/{len(wins)}  skipped: {per_h[h]['skips'] or '-'}")

        print("\n## Out-of-sample skill by label horizon (medians across windows)\n")
        print("| horizon (bars) | windows | threshold | base in-range | AUROC | AUROC share>0.5 | Brier skill | "
              f"pass-rate @{floor:.2f} | in-range when passed | train s |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for h in args.horizons:
            r = per_h[h]
            if not r["auc"]:
                print(f"| {h} | 0 | - | - | - | - | - | - | - | - |")
                continue
            share = sum(1 for a in r["auc"] if a > 0.5) / len(r["auc"])
            print(f"| {h} | {len(r['auc'])} | {med(r['thr']):.3f} | {med(r['base']):.3f} | {med(r['auc']):.3f} | "
                  f"{share:.2f} | {med(r['skill']):+.3f} | {med(r['pass']):.2f} | {med(r['prec']):.3f} | "
                  f"{med(r['secs']):.1f} |")

        base_h = args.base_horizon
        if per_h.get(base_h, {}).get("p"):
            target = float((np.asarray(per_h[base_h]["p"]) >= floor).mean())
            print(f"\n## Floor holding the base horizon's pass-rate ({target:.2f} at {floor:.2f}, h={base_h})\n")
            print("| horizon | pass-rate @ floor | floor for same pass-rate | in-range when passed (equal-rate) |")
            print("|---|---|---|---|")
            for h in args.horizons:
                p = np.asarray(per_h[h]["p"])
                y = np.asarray(per_h[h]["y"])
                if len(p) < 20:
                    continue
                eq = float(np.quantile(p, 1.0 - target)) if target > 0 else 1.0
                sel = p >= eq
                print(f"| {h} | {float((p >= floor).mean()):.2f} | {eq:.3f} | "
                      f"{float(y[sel].mean()) if sel.any() else float('nan'):.3f} |")

        print("\n## Reliability (pooled windows): P(in range) decile -> realised in-range rate\n")
        for h in args.horizons:
            p = np.asarray(per_h[h]["p"])
            y = np.asarray(per_h[h]["y"])
            if len(p) < 50:
                continue
            qs = np.quantile(p, np.linspace(0, 1, 6))
            row = []
            for lo, hi in zip(qs[:-1], qs[1:]):
                m = (p >= lo) & (p <= hi)
                row.append(f"[{lo:.2f},{hi:.2f}] {y[m].mean():.2f} (n={int(m.sum())})")
            print(f"- h={h}: " + "  ".join(row))

        print("\n## Unit slip: calendar days as bars (engine today) vs trading-day conversion\n")
        print("| DTE | h today | h converted | AUROC today | AUROC conv | threshold today | threshold conv | "
              "pass today | pass conv |")
        print("|---|---|---|---|---|---|---|---|---|")
        from ait.ml.range_spec import horizon_for
        for dte in UNIT_SLIP_DTES:
            h0 = horizon_for(dte)
            h1 = max(1, round(h0 * 5 / 7))
            a0, a1 = per_h.get(h0), per_h.get(h1)
            if not a0 or not a1 or not a0["auc"] or not a1["auc"]:
                print(f"| {dte} | {h0} | {h1} | (horizon not in --horizons) | | | | | |")
                continue
            print(f"| {dte} | {h0} | {h1} | {med(a0['auc']):.3f} | {med(a1['auc']):.3f} | {med(a0['thr']):.3f} | "
                  f"{med(a1['thr']):.3f} | {med(a0['pass']):.2f} | {med(a1['pass']):.2f} |")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

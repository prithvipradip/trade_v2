#!/usr/bin/env python
"""W119 S1: does any block of the fractal columns add out-of-sample skill to the range model?

The question, the arms, the windows, the decision rule and the wiring checks are in PLAN.md
"W119 S1 AS PLANNED" (written before this script). Each arm is a study file
``studies/fractal_ablation_<arm>.yaml`` whose only content is the arm's ``ml.features.<group>.use.<column>: false``
switches. Like ``scripts/range_horizon_study.py`` (W78) this is read-only: ``data/historical.db`` is opened
``mode=ro`` and model artifacts go to a temporary directory.

    python scripts/fractal_ablation_study.py --dry-run     # coverage, arms, counts and one training time; NO model score

Only ``--dry-run`` exists so far. The scoring, the statistics and the canaries are step (3) of the registered order
of work and are written after the dry run has been read.
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()  # before any trading-subsystem import (docs/TECHNICAL_GUIDE.md "Conventions")

import argparse  # noqa: E402
import shutil  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "studies"
SYMBOLS = ("SPY", "QQQ", "IWM")

# Registered in PLAN.md "W119 S1 AS PLANNED"; tests/test_w119_fractal_ablation_arms.py pins the study files to this table.
DAILY_FRACTAL = ("hurst_wavelet", "hurst_fit_r2", "psd_beta", "psd_fit_r2", "hurst_psd_divergence",
                 "hurst_short", "hurst_long", "hurst_scale_spread", "multifractal_width", "multifractal_asymmetry")
INTRADAY_FRACTAL = ("hurst_wavelet_intraday", "hurst_scale_spread_intraday", "wavelet_L3_energy",
                    "wavelet_L4_energy", "wavelet_L5_energy", "psd_beta_intraday", "mfdfa_width_intraday")
# arm -> (stage, {group: columns switched off})
ARMS: dict[str, tuple[str, dict[str, tuple[str, ...]]]] = {
    "a0": ("S1a", {}),
    "a1": ("S1a", {"fractal": DAILY_FRACTAL}),
    "a2": ("S1a", {"fractal": ("hurst_wavelet", "hurst_fit_r2")}),
    "a3": ("S1a", {"fractal": ("psd_beta", "psd_fit_r2", "hurst_psd_divergence")}),
    "a4": ("S1a", {"fractal": ("hurst_short", "hurst_long", "hurst_scale_spread")}),
    "a5": ("S1a", {"fractal": ("multifractal_width", "multifractal_asymmetry")}),
    "a6": ("S1a", {"fractal": ("hurst_scale_spread", "multifractal_width")}),
    "b0": ("S1b", {}),
    "b1": ("S1b", {"intraday": INTRADAY_FRACTAL}),
    "b2": ("S1b", {"fractal": DAILY_FRACTAL, "intraday": INTRADAY_FRACTAL}),
}
STAGE_START = {"S1a": "2012-01-01", "S1b": "2022-10-12"}       # first frame row (a year of warm-up before the first test)
STAGE_TEST_FROM = {"S1a": "2013-01-01", "S1b": "2023-10-12"}   # a window counts only if its test period starts on or after
HORIZONS = (5, 9, 30)
TRAIN_ROWS, TEST_ROWS, STEP_ROWS = 252, 63, 63
BUDGET_SECONDS = 6 * 3600
MIN_WINDOWS = 20


def windows(n: int, train: int = TRAIN_ROWS, test: int = TEST_ROWS, step: int = STEP_ROWS) -> list[tuple[int, int, int, int]]:
    """Rolling (train_start, train_end, test_start, test_end) row indices with non-overlapping test periods."""
    out, start = [], 0
    while start + train + test <= n:
        out.append((start, start + train, start + train, start + train + test))
        start += step
    return out


def study_path(arm: str) -> Path:
    return STUDIES / f"fractal_ablation_{arm}.yaml"


def removed_columns(arm: str) -> tuple[str, ...]:
    return tuple(c for cols in ARMS[arm][1].values() for c in cols)


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def dry_run(args: argparse.Namespace) -> int:
    import pandas as pd

    from ait.backtesting.replay import ResearchData
    from ait.backtesting.walkforward import WalkForwardBacktester
    from ait.config.research import load_research_settings, use_research_config
    from ait.ml.feature_spec import FeatureSpec
    from ait.ml.features import FeatureEngine
    from ait.ml.range_predictor import RangePredictor

    warnings.filterwarnings("ignore")
    db = Path(args.historical_db)
    print("# W119 S1 dry run (no model score is computed)\n")

    # ---- 1. arms: expected feature lists -------------------------------------------------------------
    print("## Arms: feature list, count and spec hash (from each arm's study file)\n")
    print("| arm | stage | removed | features | spec hash | study file |")
    print("|---|---|---|---|---|---|")
    base_names: dict[str, list[str]] = {}
    for arm, (stage, _) in ARMS.items():
        path = study_path(arm)
        if not path.exists():
            print(f"| {arm} | {stage} | - | MISSING | - | {path.name} |")
            continue
        s = load_research_settings(study=path)
        spec = FeatureSpec.from_config(s.ml.features)
        names = spec.feature_names(FeatureEngine.VLMC_FEATURE_NAMES if stage == "S1b" else None)
        base = base_names.setdefault(stage, names) if arm in ("a0", "b0") else base_names.get(stage, names)
        gone = [n for n in base if n not in names]
        ok = "OK" if sorted(gone) == sorted(removed_columns(arm)) else f"MISMATCH (expected {sorted(removed_columns(arm))})"
        print(f"| {arm} | {stage} | {len(gone)} ({ok}) | {len(names)} | {spec.spec_hash()} | {path.name} |")

    # The arm files are merged over the research view above; only now freeze the plain research view for the rest
    # (a frozen configuration ignores any later overlay, which is why the arms are read first).
    settings = use_research_config()

    # ---- 2. windows and coverage -----------------------------------------------------------------------
    print("\n## Windows (train {} / test {} / step {} rows)\n".format(TRAIN_ROWS, TEST_ROWS, STEP_ROWS))
    print("| stage | symbol | frame rows | windows (test from) | first test | last test |")
    print("|---|---|---|---|---|---|")
    frames: dict[tuple[str, str], tuple] = {}
    for stage in ("S1a", "S1b"):
        for sym in SYMBOLS:
            frame = ResearchData(str(db), sym).daily_frame()
            ohlcv = frame[["Open", "High", "Low", "Close", "Volume"]].dropna().loc[STAGE_START[stage]:]
            wins = [w for w in windows(len(ohlcv)) if ohlcv.index[w[2]] >= pd.Timestamp(STAGE_TEST_FROM[stage])]
            frames[(stage, sym)] = (ohlcv, wins)
            first = ohlcv.index[wins[0][2]].date() if wins else "-"
            last = ohlcv.index[wins[-1][3] - 1].date() if wins else "-"
            print(f"| {stage} | {sym} | {len(ohlcv)} | {len(wins)} | {first} | {last} |")
    for stage in ("S1a", "S1b"):
        n = min(len(frames[(stage, s)][1]) for s in SYMBOLS)
        print(f"\n{stage}: {n} windows per symbol -> " + ("sufficient" if n >= MIN_WINDOWS else
              f"INSUFFICIENT (fewer than {MIN_WINDOWS}; every arm of the stage is reported as INSUFFICIENT)"))

    print("\n## Input coverage\n")
    con = _ro(db)
    try:
        spy = pd.read_sql("select date from daily_prices where symbol='SPY'", con, parse_dates=["date"]).date
        for stage in ("S1a",):
            cal = set(spy[(spy >= STAGE_TEST_FROM[stage]) & (spy >= STAGE_START[stage])])
            for sym in ("VIX", "VIX3M"):
                d = pd.read_sql("select date from daily_iv where symbol=?", con, params=(sym,), parse_dates=["date"]).date
                miss = cal - set(d)
                print(f"- {sym} in daily_iv over the {stage} span: {len(cal) - len(miss)} of {len(cal)} SPY sessions "
                      f"(missing {len(miss)}; first stored {d.min().date()})")
        vlmc_min = float(getattr(settings.backtest, "vlmc_min_coverage", 0.95))
        for sym in SYMBOLS:
            days = {r[0] for r in con.execute(
                "select distinct substr(datetime,1,10) from intraday_prices where symbol=?", (sym,))}
            ohlcv = frames[("S1b", sym)][0]
            span = ohlcv.loc["2023-10-12":]
            cov = float(span.index.strftime("%Y-%m-%d").isin(days).mean()) if len(span) else 0.0
            print(f"- 5-minute coverage of the S1b span, {sym}: {cov:.3f} (VLMC needs >= {vlmc_min:.2f}: "
                  f"{'ON' if cov >= vlmc_min else 'OFF'})")
    finally:
        con.close()
    print("- macros and the walk-forward's VIX / SPY context are checked by wiring check W-1, not here")

    # ---- 3. one feature computation and ONE training (no score) -------------------------------------------
    print("\n## Timing probe (one window, SPY, S1a; no score is computed or printed)\n")
    ohlcv, wins = frames[("S1a", "SPY")]
    a, b, c, d = wins[0]
    ctx = ohlcv.iloc[a:d]
    t0 = time.perf_counter()
    feats = FeatureEngine().compute(ctx)
    t_feat = time.perf_counter() - t0
    print(f"- features on a {len(ctx)}-row context: {t_feat:.1f}s, {len(feats)} usable rows after the warm-up")
    train_df = ohlcv.iloc[a:b]
    tmp = Path(tempfile.mkdtemp(prefix="w119_s1_dry_"))
    try:
        times = {}
        for h in HORIZONS:
            thr = WalkForwardBacktester._adaptive_range_threshold(train_df, horizon_days=h)
            rp = RangePredictor(threshold_pct=thr, horizon_days=h, model_dir=tmp / f"h{h}")
            t0 = time.perf_counter()
            rp.train(train_df, symbol="SPY", intraday_store=None)
            times[h] = time.perf_counter() - t0
            print(f"- one training at h={h} (threshold {thr:.3f}): {times[h]:.1f}s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- 4. projected time against the budget -------------------------------------------------------------
    per_window = sum(times.values()) + t_feat * len(HORIZONS)
    total = 0.0
    for arm, (stage, _) in ARMS.items():
        total += per_window * sum(len(frames[(stage, s)][1]) for s in SYMBOLS)
    print(f"\n## Projected time\n\nOne run over all arms, symbols, windows and horizons {HORIZONS}: about {total / 3600:.1f} h "
          f"(budget {BUDGET_SECONDS / 3600:.0f} h per run; the stress repeat doubles it).")
    if total > BUDGET_SECONDS:
        print("Over budget: the registered reductions apply in order (horizons -> {5, 30}; S1a start 2013 -> 2016; "
              "stress repeat on the most recent 20 windows).")
    else:
        print("Within budget for the base run.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="coverage, arms, counts and one training time; computes no score")
    ap.add_argument("--historical-db", default=str(ROOT / "data/historical.db"))
    args = ap.parse_args(argv)
    if not args.dry_run:
        print("only --dry-run exists so far: the scoring path is step (3) of PLAN.md 'W119 S1 AS PLANNED'", file=sys.stderr)
        return 2
    return dry_run(args)


if __name__ == "__main__":
    sys.exit(main())

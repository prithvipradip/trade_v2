#!/usr/bin/env python
"""W119 S1: does any block of the fractal columns add out-of-sample skill to the range model?

The question, the arms, the windows, the decision rule and the wiring checks are in PLAN.md
"W119 S1 AS PLANNED" (written before this script). Each arm is a study file
``studies/fractal_ablation_<arm>.yaml`` whose only content is the arm's ``ml.features.<group>.use.<column>: false``
switches. Like ``scripts/range_horizon_study.py`` (W78) this is read-only: ``data/historical.db`` is opened
``mode=ro`` and model artifacts go to a temporary directory.

    python scripts/fractal_ablation_study.py --dry-run                        # coverage, arms, counts, one training time; NO score
    python scripts/fractal_ablation_study.py --run --arm a2 [--seed 42] [--label base]   # one arm: scores per (symbol, window, horizon)
    python scripts/fractal_ablation_study.py --run --arm a0 --canary leaked --label canary_leaked   # a wiring-check arm
    python scripts/fractal_ablation_study.py --check-asof                     # wiring check W-5
    python scripts/fractal_ablation_study.py --check-engine                   # wiring check W-1 (one real walk-forward window)
    python scripts/fractal_ablation_study.py --report --label base            # statistics, verdicts, wiring checks

PLAN.md "W119 S1 ADDENDUM 1" (QQQ is the primary verdict, SPY and IWM replicate) and "ADDENDUM 2" (the S1a start, the S1b
window rule, the canary numbers) amend the registered rule before any scoring. One process per arm: the arm's study file is
read BEFORE the settings freeze (a frozen configuration ignores any later overlay).
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()  # before any trading-subsystem import (docs/TECHNICAL_GUIDE.md "Conventions")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
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
# First frame row of each stage. S1a starts 2016-02-01 (ADDENDUM 2: the stored DTWEXBGS series has a 6-day gap, 2016-01-21..27,
# which the walk-forward's strict macro loader refuses, so no window may contain it; this is the registered "2013 -> 2016" reduction).
# S1b starts at the first 5-minute day so that every train frame has full VLMC coverage ("VLMC for every window or none").
STAGE_START = {"S1a": "2016-02-01", "S1b": "2023-10-12"}
STAGE_TEST_FROM = {"S1a": "2016-02-01", "S1b": "2023-10-12"}   # a window counts only if its test period starts on or after
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


# --------------------------------------------------------------------------------------------------------------------
# Scoring path (step 3 of the registered order of work)
# --------------------------------------------------------------------------------------------------------------------
OUT_DIR = ROOT / "reports" / "fractal_ablation_study"
END_DATE = "2026-10-30"                  # the last stored session when scoring was written; every run of the study uses the same end
PRIMARY = "QQQ"                          # ADDENDUM 1: QQQ is the primary verdict, SPY and IWM replicate
THRESHOLD = 0.01                         # a block changes skill only if the mean Brier-skill change is at least this (chosen)
ALPHA = 0.10                             # one-sided, Holm-adjusted
BOOT_B, BOOT_BLOCK, BOOT_SEED = 2000, 2, 20261103
CANARY_CANARY_ARM = "a5"                 # W-3: both columns are constant 0.0 in a standard frame, so removing them must not matter
W3_MAX_ABS_DELTA = 0.005
LEAK_MIN_GAIN = 0.10                     # W-4 (ADDENDUM 2): the LEAKED canary must raise the mean Brier skill by at least this
SHUFFLE_MAX_GAIN = 0.005                 # W-4: the SHUFFLED canary must not raise it by more than this (and not significantly)
ASOF_TOLERANCE = 1e-9                    # W-5


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def labels(close, h: int, thr: float):
    """The engine's range label: 1 if the max |return| over the next h closes stays below the threshold."""
    import numpy as np
    n = len(close)
    y = np.full(n, np.nan)
    for t in range(n - h):
        y[t] = 1.0 if np.max(np.abs(close[t + 1: t + 1 + h] / close[t] - 1.0)) < thr else 0.0
    return y


def market_context(vix, spy, macros, index) -> dict | None:
    """The walk-forward's slicing of the VIX / SPY / macro context onto a frame (walkforward.py, `_ctx_oos`)."""
    ctx: dict = {}
    if vix is not None and not vix.empty:
        v = vix.reindex(index, method="ffill").dropna(how="all")
        if not v.empty:
            ctx["vix"] = v
    if spy is not None and not spy.empty:
        sp = spy.reindex(index, method="ffill").dropna(how="all")
        if not sp.empty:
            ctx["spy"] = sp
    if macros:
        ctx["macros"] = macros
    return ctx or None


_CANARY: dict = {"kind": None, "h": None}


def _install_canary(kind: str | None) -> None:
    """W-4: wrap FeatureEngine.compute so a canary changes the VALUES of an existing fractal column (names stay, so the model
    uses it). leaked: hurst_scale_spread := the label's own future excursion (cannot be known at t); shuffled: the ten daily
    fractal columns row-shuffled within the frame. Applies identically to train and test frames."""
    if not kind:
        return
    import numpy as np

    from ait.ml.features import FeatureEngine
    orig = FeatureEngine.compute

    def compute(self, df, *a, **kw):
        out = orig(self, df, *a, **kw)
        if out is None or out.empty:
            return out
        out = out.copy()
        if kind == "leaked":
            h = int(_CANARY["h"])
            close = out["Close"].to_numpy(dtype=float)
            leak = np.zeros(len(close))
            for t in range(len(close) - h):
                leak[t] = float(np.max(np.abs(close[t + 1: t + 1 + h] / close[t] - 1.0)))
            out["hurst_scale_spread"] = leak
        elif kind == "shuffled":
            rng = np.random.default_rng(BOOT_SEED)
            perm = rng.permutation(len(out))
            for c in DAILY_FRACTAL:
                if c in out.columns:
                    out[c] = out[c].to_numpy()[perm]
        return out

    FeatureEngine.compute = compute


def run_arm(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from ait.backtesting.replay import ResearchData
    from ait.backtesting.walkforward import WalkForwardBacktester
    from ait.config.research import load_study, use_research_config
    from ait.data.macro_store import load_macro_series
    from ait.data.market_data import load_daily_ohlcv
    from ait.data.research_inputs import Window
    from ait.ml.feature_spec import FeatureSpec
    from ait.ml.features import FeatureEngine
    from ait.ml.range_predictor import RangePredictor

    warnings.filterwarnings("ignore")
    arm = args.arm
    stage, _groups = ARMS[arm]
    study = load_study(study_path(arm))
    if args.seed is not None:           # the stress repeat: only the model seed changes
        study = dataclasses.replace(study, settings=_deep_merge(
            study.settings or {}, {"ml": {"hyperparams": {"random_state": int(args.seed)}}}))
    settings = use_research_config(str(ROOT / "config.yaml"), study=study)     # the study is merged BEFORE the freeze
    spec = FeatureSpec.from_config(settings.ml.features)
    names = spec.feature_names(FeatureEngine.VLMC_FEATURE_NAMES if stage == "S1b" else None)
    _install_canary(args.canary)

    import datetime as dt
    db = Path(args.historical_db)
    start = pd.Timestamp(STAGE_START[stage]).date()
    end = pd.Timestamp(args.end_date).date()
    vix = load_daily_ohlcv("^VIX", db_path=db, start_date=start, end_date=end)
    spy = load_daily_ohlcv("SPY", db_path=db, start_date=start, end_date=end)
    macros = load_macro_series(db, Window(start, end), "")
    symbols = args.symbols or list(SYMBOLS)
    horizons = args.horizons or list(HORIZONS)
    tmp = Path(tempfile.mkdtemp(prefix=f"w119_s1_{arm}_"))
    records: list[dict] = []
    t_start = time.perf_counter()
    try:
        for sym in symbols:
            frame = ResearchData(str(db), sym).daily_frame()
            ohlcv = frame[["Open", "High", "Low", "Close", "Volume"]].dropna().loc[str(start):str(end)]
            wins = [(i, w) for i, w in enumerate(windows(len(ohlcv)), 1)
                    if ohlcv.index[w[2]] >= pd.Timestamp(STAGE_TEST_FROM[stage])]
            if args.max_windows:
                wins = wins[-args.max_windows:]
            store = ResearchData(str(db), sym) if stage == "S1b" else None
            fe = FeatureEngine()
            for wi, (a, b, c, d) in wins:
                train_df, test_df, ctx = ohlcv.iloc[a:b], ohlcv.iloc[c:d], ohlcv.iloc[a:d]
                tr_ctx = market_context(vix, spy, macros, train_df.index)
                oos_ctx = market_context(vix, spy, macros, ctx.index)
                feats = None if args.canary == "leaked" else fe.compute(ctx, market_context=oos_ctx, intraday_store=store, symbol=sym)
                for h in horizons:
                    _CANARY["h"] = h
                    thr = WalkForwardBacktester._adaptive_range_threshold(train_df, horizon_days=h)
                    rec = {"symbol": sym, "window": wi, "test_start": str(test_df.index[0].date()),
                           "test_end": str(test_df.index[-1].date()), "h": h, "threshold": round(float(thr), 5)}
                    rp = RangePredictor(threshold_pct=thr, horizon_days=h, model_dir=tmp / f"{sym}_{wi}_{h}")
                    accs = rp.train(train_df, symbol=sym, market_context=tr_ctx, intraday_store=store)
                    if not accs or not rp.is_trained:
                        records.append({**rec, "status": "not_trained"})
                        continue
                    if feats is None:
                        feats_h = fe.compute(ctx, market_context=oos_ctx, intraday_store=store, symbol=sym)
                    else:
                        feats_h = feats
                    y = labels(test_df["Close"].to_numpy(dtype=float), h, thr)
                    ps, ys, n_none, n_nofeat = [], [], 0, 0
                    for k in range(len(test_df) - h):
                        idx = test_df.index[k]
                        if idx not in feats_h.index:
                            n_nofeat += 1
                            continue
                        pred = rp.predict_from_features(feats_h.loc[idx], symbol=sym)
                        if pred is None:
                            n_none += 1
                            continue
                        ps.append(float(pred.probability_in_range))
                        ys.append(float(y[k]))
                    if len(ys) < 10 or len(set(ys)) < 2:
                        why = ("single_class_oos" if len(ys) >= 10 else
                               "abstained_no_cv_edge" if n_none >= max(1, n_nofeat) and n_none > 0 else "no_feature_rows")
                        records.append({**rec, "status": why, "n_abstain": n_none})
                        continue
                    p, yy = np.asarray(ps), np.asarray(ys)
                    base = float(yy.mean())
                    bb = base * (1.0 - base)
                    records.append({**rec, "status": "scored", "n_scored": len(yy), "n_abstain": n_none, "base_rate": base,
                                    "auroc": float(roc_auc_score(yy, p)),
                                    "brier_skill": float(1.0 - np.mean((p - yy) ** 2) / bb) if bb > 0 else float("nan")})
                print(f"  {arm} {sym} window {wi} done ({time.perf_counter() - t_start:.0f}s)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = OUT_DIR / args.label
    out.mkdir(parents=True, exist_ok=True)
    meta = {"arm": arm, "stage": stage, "seed": args.seed, "canary": args.canary, "end_date": args.end_date,
            "spec_hash": spec.spec_hash(), "n_features": len(names), "features": names,
            "random_state": settings.ml.hyperparams.random_state, "symbols": symbols, "horizons": horizons,
            "max_windows": args.max_windows, "seconds": round(time.perf_counter() - t_start, 1)}
    (out / f"{arm}{'_' + args.canary if args.canary else ''}.json").write_text(json.dumps({"meta": meta, "records": records}, indent=1))
    scored = sum(1 for r in records if r["status"] == "scored")
    print(f"{arm}: {scored} scored of {len(records)} (symbol, window, horizon) cells in {meta['seconds']}s -> {out}")
    return 0


# ---- statistics (pure functions; tests/test_w119_fractal_ablation_arms.py) -------------------------------------------
def block_bootstrap_means(x, block: int = BOOT_BLOCK, B: int = BOOT_B, seed: int = BOOT_SEED):
    """Circular moving-block bootstrap of the mean of a window-ordered series."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    n = len(x)
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return x[idx.reshape(B, -1)[:, :n]].mean(axis=1)


def one_sided_p(deltas, null: float, direction: int, seed: int = BOOT_SEED) -> float:
    """P* of the bootstrap mean being on the wrong side of `null`: direction -1 tests 'mean < null' (p = share of means >= null),
    +1 tests 'mean > null' (p = share <= null). Add-one smoothing so a p is never exactly 0."""
    import numpy as np
    m = block_bootstrap_means(deltas, seed=seed)
    wrong = (m >= null) if direction < 0 else (m <= null)
    return float((wrong.sum() + 1) / (len(m) + 1))


def holm(pvals: list[float]) -> list[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def classify(mean: float, p_adj: float, n_windows: int) -> str:
    """KEEP: removal lowers skill by >= THRESHOLD; REMOVAL OK: removal raises it by >= THRESHOLD; INSUFFICIENT below MIN_WINDOWS."""
    if n_windows < MIN_WINDOWS:
        return "INSUFFICIENT"
    if mean <= -THRESHOLD and p_adj < ALPHA:
        return "KEEP"
    if mean >= THRESHOLD and p_adj < ALPHA:
        return "REMOVAL OK"
    return "NOT SHOWN"


def _load_arm(label_dir: Path, arm: str, canary: str | None = None):
    path = label_dir / f"{arm}{'_' + canary if canary else ''}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _skill_table(doc: dict):
    """{(symbol, window, h): brier_skill} for the SCORED cells of an arm file."""
    return {(r["symbol"], r["window"], r["h"]): r["brier_skill"] for r in doc["records"] if r["status"] == "scored"}


def _paired(base: dict, arm: dict, symbols: list[str], h: int, windows_ok: set | None = None):
    """Per calendar window: arm - baseline Brier skill averaged over `symbols`, using windows where every symbol is scored in both."""
    import numpy as np
    wins = sorted({k[1] for k in base if k[2] == h})
    out_w, out_d = [], []
    for w in wins:
        ds = []
        for sym in symbols:
            kb, ka = (sym, w, h), (sym, w, h)
            if kb in base and ka in arm and base[kb] == base[kb] and arm[ka] == arm[ka]:
                ds.append(arm[ka] - base[kb])
            else:
                ds = None
                break
        if ds:
            out_w.append(w)
            out_d.append(float(np.mean(ds)))
    return np.asarray(out_w), np.asarray(out_d)


def report(args: argparse.Namespace) -> int:
    import numpy as np
    label_dir = OUT_DIR / args.label
    docs = {arm: _load_arm(label_dir, arm) for arm in ARMS}
    have = {a for a, d in docs.items() if d}
    if "a0" not in have and "b0" not in have:
        print(f"no baseline arm file in {label_dir}", file=sys.stderr)
        return 1
    lines: list[str] = [f"# W119 S1 report: {args.label}\n"]
    voids: list[str] = []

    # W-2: every arm's recorded feature set matches the registered removal
    lines.append("## Wiring check W-2: recorded feature sets\n")
    for stage, base_arm in (("S1a", "a0"), ("S1b", "b0")):
        if base_arm not in have:
            continue
        base_feats = set(docs[base_arm]["meta"]["features"])
        for arm in (a for a, (st, _) in ARMS.items() if st == stage and a in have):
            gone = base_feats - set(docs[arm]["meta"]["features"])
            ok = gone == set(removed_columns(arm))
            lines.append(f"- {arm}: {len(gone)} removed, spec hash {docs[arm]['meta']['spec_hash']}: {'OK' if ok else 'MISMATCH -> VOID'}")
            if not ok:
                voids.append(f"W-2 {arm}")

    results = {}
    for stage, base_arm in (("S1a", "a0"), ("S1b", "b0")):
        if base_arm not in have:
            continue
        base = _skill_table(docs[base_arm])
        arms = [a for a, (st, _) in ARMS.items() if st == stage and a != base_arm and a in have]
        lines.append(f"\n## {stage}: Brier-skill change when a block is removed (arm minus baseline), horizons {HORIZONS}\n")
        family = []   # (arm, h) with the primary one-sided p
        cells = {}
        for arm in arms:
            tbl = _skill_table(docs[arm])
            for h in sorted({k[2] for k in base}):
                w, d = _paired(base, tbl, [PRIMARY], h)
                n = len(d)
                if n == 0:
                    cells[(arm, h)] = {"n": 0}
                    continue
                mean = float(d.mean())
                direction = -1 if mean < 0 else +1
                p = one_sided_p(d, null=direction * THRESHOLD, direction=direction)
                cells[(arm, h)] = {"n": n, "mean": mean, "direction": direction, "p": p}
                family.append((arm, h))
        padj = dict(zip(family, holm([cells[k]["p"] for k in family]))) if family else {}
        lines.append("| arm | removed | h | QQQ windows scored | mean change (QQQ) | adj. p | verdict (QQQ) | SPY / IWM change (windows) | replication | pooled 3-symbol |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for arm in arms:
            tbl = _skill_table(docs[arm])
            for h in sorted({k[2] for k in base}):
                c = cells[(arm, h)]
                if c["n"] == 0:
                    lines.append(f"| {arm} | {len(removed_columns(arm))} | {h} | 0 | - | - | INSUFFICIENT | - | - | - |")
                    continue
                verdict = classify(c["mean"], padj[(arm, h)], c["n"])
                rep = []
                reps_ok = []
                for sym in ("SPY", "IWM"):
                    w2, d2 = _paired(base, tbl, [sym], h)
                    if len(d2) >= 2:
                        p2 = one_sided_p(d2, null=0.0, direction=c["direction"])
                        rep.append(f"{sym} {d2.mean():+.4f} ({len(d2)})")
                        reps_ok.append((np.sign(d2.mean()) == c["direction"], p2))
                    else:
                        rep.append(f"{sym} n/a")
                        reps_ok.append((False, 1.0))
                replicated = (all(s for s, _ in reps_ok) and any(p < ALPHA for _, p in reps_ok)) if verdict in ("KEEP", "REMOVAL OK") else None
                wp, dp = _paired(base, tbl, list(SYMBOLS), h)
                pooled = f"{dp.mean():+.4f} ({len(dp)})" if len(dp) else "n/a"
                lines.append(f"| {arm} | {len(removed_columns(arm))} | {h} | {c['n']} | {c['mean']:+.4f} | {padj[(arm, h)]:.3f} | {verdict} | "
                             f"{' / '.join(rep)} | {'-' if replicated is None else ('REPLICATED' if replicated else 'NOT REPLICATED')} | {pooled} |")
                results[f"{arm}_h{h}"] = {"stage": stage, "verdict": verdict, "mean": c["mean"], "p_adj": padj[(arm, h)], "n": c["n"]}
        # W-3: the canary arm
        if stage == "S1a" and CANARY_CANARY_ARM in arms:
            lines.append(f"\n### Wiring check W-3 (arm {CANARY_CANARY_ARM}: two constant columns removed)\n")
            for h in sorted({k[2] for k in base}):
                c = cells.get((CANARY_CANARY_ARM, h), {"n": 0})
                if c["n"] == 0:
                    continue
                ok = abs(c["mean"]) < W3_MAX_ABS_DELTA and padj[(CANARY_CANARY_ARM, h)] >= ALPHA
                lines.append(f"- h={h}: mean change {c['mean']:+.4f}, adj. p {padj[(CANARY_CANARY_ARM, h)]:.3f}: {'OK' if ok else 'FAILED -> S1a VOID'}")
                if not ok:
                    voids.append(f"W-3 h={h}")

    # W-4 canaries (arm a0 run with --canary): compared with the baseline over the SAME windows
    for kind in ("leaked", "shuffled"):
        doc = _load_arm(label_dir, "a0", kind) or _load_arm(OUT_DIR / f"canary_{kind}", "a0", kind)
        if not doc or "a0" not in have:
            continue
        base = _skill_table(docs["a0"])
        tbl = _skill_table(doc)
        lines.append(f"\n### Wiring check W-4: {kind.upper()} canary (QQQ)\n")
        for h in sorted({k[2] for k in tbl}):
            w, d = _paired(base, tbl, [PRIMARY], h)
            if len(d) == 0:
                continue
            gain = float(d.mean())
            p_up = one_sided_p(d, null=0.0, direction=+1)
            ok = gain >= LEAK_MIN_GAIN if kind == "leaked" else (gain <= SHUFFLE_MAX_GAIN and p_up >= ALPHA)
            lines.append(f"- h={h}: mean change {gain:+.4f} over {len(d)} windows, p(gain) {p_up:.3f}: {'OK' if ok else 'FAILED -> VOID'}")
            if not ok:
                voids.append(f"W-4 {kind} h={h}")

    lines.append("\n## Reported only\n")
    for arm in sorted(have):
        doc = docs[arm]
        rec = doc["records"]
        skipped = {}
        for r in rec:
            if r["status"] != "scored":
                skipped[(r["symbol"], r["status"])] = skipped.get((r["symbol"], r["status"]), 0) + 1
        lines.append(f"- {arm}: {sum(1 for r in rec if r['status'] == 'scored')} scored cells; skipped {skipped or '-'}; "
                     f"{doc['meta']['seconds']}s; seed {doc['meta']['random_state']}")
    lines.append("\n## Verdict\n")
    lines.append("VOID: " + ", ".join(voids) if voids else "No wiring check failed (W-1 and W-5 are separate commands; see AS BUILT).")
    text = "\n".join(lines)
    (label_dir / "report.md").write_text(text)
    (label_dir / "results.json").write_text(json.dumps(results, indent=1))
    print(text)
    return 0


def check_asof(args: argparse.Namespace) -> int:
    """W-5: altering every bar dated after a window's last test day leaves that window's features unchanged."""
    import numpy as np
    import pandas as pd

    from ait.backtesting.replay import ResearchData
    from ait.config.research import use_research_config
    from ait.data.macro_store import load_macro_series
    from ait.data.market_data import load_daily_ohlcv
    from ait.data.research_inputs import Window
    from ait.ml.features import FeatureEngine

    warnings.filterwarnings("ignore")
    use_research_config(str(ROOT / "config.yaml"))
    db = Path(args.historical_db)
    start, end = pd.Timestamp(STAGE_START["S1a"]).date(), pd.Timestamp(args.end_date).date()
    vix = load_daily_ohlcv("^VIX", db_path=db, start_date=start, end_date=end)
    spy = load_daily_ohlcv("SPY", db_path=db, start_date=start, end_date=end)
    macros = load_macro_series(db, Window(start, end), "")
    rng = np.random.default_rng(BOOT_SEED)
    worst = 0.0
    for sym in (args.symbols or [PRIMARY]):
        ohlcv = ResearchData(str(db), sym).daily_frame()[["Open", "High", "Low", "Close", "Volume"]].dropna().loc[str(start):str(end)]
        for a, b, c, d in (windows(len(ohlcv))[0], windows(len(ohlcv))[-1]):
            d_plus = min(len(ohlcv), d + 40)
            if d_plus <= d:
                continue
            ctx = ohlcv.iloc[a:d]
            plus = ohlcv.iloc[a:d_plus].copy()
            fut = plus.index[len(ctx):]
            plus.loc[fut, ["Open", "High", "Low", "Close"]] = plus.loc[fut, ["Open", "High", "Low", "Close"]].to_numpy() * \
                (1.0 + 0.3 * rng.standard_normal((len(fut), 1)))
            vix_p, spy_p = vix.copy(), spy.copy()
            for fr in (vix_p, spy_p):
                m = fr.index > ctx.index[-1]
                fr.loc[m, ["Open", "High", "Low", "Close"]] = fr.loc[m, ["Open", "High", "Low", "Close"]].to_numpy() * 1.7
            f0 = FeatureEngine().compute(ctx, market_context=market_context(vix, spy, macros, ctx.index))
            f1 = FeatureEngine().compute(plus, market_context=market_context(vix_p, spy_p, macros, plus.index))
            common = f0.index.intersection(f1.index)
            cols = [col for col in f0.columns if col in f1.columns and col not in ("Open", "High", "Low", "Close", "Volume")]
            diff = float(np.nanmax(np.abs(f0.loc[common, cols].to_numpy(dtype=float) - f1.loc[common, cols].to_numpy(dtype=float))))
            worst = max(worst, diff)
            print(f"{sym} window rows {a}-{d}: {len(common)} rows x {len(cols)} columns compared, max abs difference {diff:.3e}")
    ok = worst <= ASOF_TOLERANCE
    print(f"W-5: {'OK' if ok else 'FAILED -> VOID'} (tolerance {ASOF_TOLERANCE:g}, worst {worst:.3e})")
    return 0 if ok else 1


def check_engine(args: argparse.Namespace) -> int:
    """W-1: run ONE real walk-forward window (QQQ, the research default span), capture what it passes to
    FeatureEngine.compute (the frame index and the market context), and compare the VIX, SPY and macro inputs with the harness's
    own context on the same dates. The features then come from the same code on the same inputs. Writes nothing of its own
    (`run_backtest.py --no-run-record`)."""
    import datetime as dt
    import runpy

    import numpy as np
    import pandas as pd

    from ait.data.macro_store import load_macro_series
    from ait.data.market_data import load_daily_ohlcv
    from ait.data.research_inputs import Window
    from ait.ml.features import FeatureEngine

    captured: list[dict] = []
    orig = FeatureEngine.compute

    def compute(self, df, market_context=None, *a, **kw):
        if market_context is not None and len(captured) < 6:
            captured.append({"index": df.index.copy(), "ctx": {k: (v.copy() if hasattr(v, "copy") else v)
                                                              for k, v in market_context.items()}})
        return orig(self, df, market_context, *a, **kw)

    FeatureEngine.compute = compute
    argv0 = sys.argv
    sys.argv = [str(ROOT / "run_backtest.py"), "--symbols", PRIMARY, "--max-windows", "1", "--no-run-record"]
    try:
        runpy.run_path(str(ROOT / "run_backtest.py"), run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.argv = argv0
        FeatureEngine.compute = orig
    if not captured:
        print("W-1: no FeatureEngine.compute call with a market context was captured -> VOID")
        return 1
    db = Path(args.historical_db)
    first = min(c["index"][0] for c in captured).date()
    start = (pd.Timestamp(first) - pd.Timedelta(days=60)).date()
    end = pd.Timestamp(args.end_date).date()
    vix = load_daily_ohlcv("^VIX", db_path=db, start_date=start, end_date=end)
    spy = load_daily_ohlcv("SPY", db_path=db, start_date=start, end_date=end)
    macros = load_macro_series(db, Window(start, end), "")
    worst = 0.0
    for i, c in enumerate(captured):
        mine = market_context(vix, spy, macros, c["index"])
        for key in ("vix", "spy"):
            if key in c["ctx"]:
                a, b = c["ctx"][key], mine[key]
                common = a.index.intersection(b.index)
                d = float(np.nanmax(np.abs(a.loc[common].to_numpy(dtype=float) - b.loc[common].to_numpy(dtype=float)))) if len(common) else 1e9
                worst = max(worst, d)        # compared on the dates both sides hold (the engine passes a longer context on small frames)
        for name, ser in (c["ctx"].get("macros") or {}).items():
            other = mine["macros"].get(name)
            common = ser.index.intersection(other.index) if other is not None else []
            d = float(np.nanmax(np.abs(ser.loc[common].to_numpy(dtype=float) - other.loc[common].to_numpy(dtype=float)))) if len(common) else 1e9
            worst = max(worst, d)
        print(f"call {i}: frame {c['index'][0].date()}..{c['index'][-1].date()} ({len(c['index'])} rows) compared")
    ok = worst <= ASOF_TOLERANCE
    print(f"W-1: {'OK' if ok else 'FAILED -> VOID'} (engine and harness inputs, worst difference {worst:.3e}, tolerance {ASOF_TOLERANCE:g})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="coverage, arms, counts and one training time; computes no score")
    ap.add_argument("--run", action="store_true", help="score one arm (needs --arm)")
    ap.add_argument("--report", action="store_true", help="statistics, verdicts and wiring checks for --label")
    ap.add_argument("--check-asof", action="store_true", help="wiring check W-5")
    ap.add_argument("--check-engine", action="store_true", help="wiring check W-1 (runs one real walk-forward window)")
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=None, help="ml.hyperparams.random_state for the stress repeat (base run: the config value, 42)")
    ap.add_argument("--canary", choices=("leaked", "shuffled"), default=None, help="W-4 wiring-check arm (run on arm a0)")
    ap.add_argument("--label", default="base", help="output folder under reports/fractal_ablation_study/")
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--horizons", type=int, nargs="+", default=None)
    ap.add_argument("--max-windows", type=int, default=0, help="most recent N windows per symbol (0 = all)")
    ap.add_argument("--end-date", default=END_DATE)
    ap.add_argument("--historical-db", default=str(ROOT / "data/historical.db"))
    args = ap.parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    if args.run:
        if not args.arm:
            ap.error("--run needs --arm")
        return run_arm(args)
    if args.report:
        return report(args)
    if args.check_asof:
        return check_asof(args)
    if args.check_engine:
        return check_engine(args)
    ap.error("choose one of --dry-run, --run, --report, --check-asof")
    return 2


if __name__ == "__main__":
    sys.exit(main())

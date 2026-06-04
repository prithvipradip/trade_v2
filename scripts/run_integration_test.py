"""Full pipeline integration test with real IB data.

Runs end-to-end:
  A. Backfill  — 2-year SPY 5-min bars from IB → test_intraday_prices
  B. Data QC   — coverage, gaps, price sanity
  C. Feature health — NaN rates, range validation for all 18 features
  D. IC decay  — Spearman IC vs forward returns at 1/3/5/10/20-day horizons
  E. Walk-forward — multi-strategy with per-window Optuna optimization
  F. Ablation  — walk-forward WITHOUT per-window optimization (baseline)
  G. Reports   — fractal_report_SPY.html + ic_summary.csv via generate_report()
  H. Document  — RESULTS.md with all metrics

Usage:
    python scripts/run_integration_test.py --symbols SPY --years 2
    python scripts/run_integration_test.py --symbols SPY --years 2 --skip-backfill
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd

TABLE_PREFIX = "test_"
DEFAULT_DB = "data/integration_test.db"
OUTPUT_DIR = "reports/integration_test"
START_DATE = "2023-01-01"   # ≈2 years back from 2025


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_integration_test",
        description="Full pipeline integration test using real IB data.",
    )
    parser.add_argument(
        "--config", default="config.yaml", metavar="PATH",
        help="Config YAML file to load backtest settings from (default: config.yaml).",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["SPY"], metavar="SYMBOL",
        help="Symbols to test (default: SPY).",
    )
    parser.add_argument(
        "--years", type=float, default=2.0,
        help="Years of intraday history to backfill (default: 2).",
    )
    parser.add_argument(
        "--db-path", default=DEFAULT_DB, metavar="PATH",
        help=f"SQLite DB path (default: {DEFAULT_DB}).",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR, metavar="DIR",
        help=f"Output directory for reports (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="IBKR TWS/Gateway host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port", type=int, default=4002,
        help="IBKR port (default: 4002).",
    )
    parser.add_argument(
        "--client-id", type=int, default=88,
        help="IBKR client ID (default: 88).",
    )
    parser.add_argument(
        "--skip-backfill", action="store_true",
        help="Skip the IB backfill step (use existing data in the test DB).",
    )
    parser.add_argument(
        "--skip-walkforward", action="store_true",
        help="Skip walk-forward and ablation (fast mode: QC + features + reports only).",
    )
    parser.add_argument(
        "--wf-trials", type=int, default=None,
        help="Optuna trials per walk-forward window (default: from config optimize_n_trials, fallback 50).",
    )
    parser.add_argument(
        "--wf-patience", type=int, default=None,
        help="Early-stop after N non-improving trials (default: from config optimize_patience, fallback 20; 0=disabled).",
    )
    parser.add_argument(
        "--wf-min-trades", type=int, default=None,
        help="Min trades floor for objective penalty (default: from config optimize_min_trades, fallback 10).",
    )
    parser.add_argument(
        "--wf-n-jobs", type=int, default=None,
        help="Parallel Optuna workers per walk-forward window (default: 1 = sequential).",
    )
    parser.add_argument(
        "--wf-val-split", action="store_true", default=False,
        help="H2: score Optuna objective on held-out 20%% val slice instead of full training window.",
    )
    parser.add_argument(
        "--train-days", type=int, default=365,
        help="Walk-forward training window in calendar days (default: 365).",
    )
    parser.add_argument(
        "--test-days", type=int, default=63,
        help="Walk-forward test window in calendar days (default: 63).",
    )
    parser.add_argument(
        "--step-days", type=int, default=21,
        help="Days to advance each walk-forward window (default: 21).",
    )
    parser.add_argument(
        "--gap-days", type=int, default=5,
        help="Purge gap between train end and test start (default: 5).",
    )
    parser.add_argument(
        "--optuna-seed", type=int, default=42,
        help="TPESampler seed for Optuna reproducibility (default: 42).",
    )
    _DEFAULT_STRATEGIES = [
        "iron_condor", "put_credit_spread", "short_strangle",
        "bull_call_spread", "bear_put_spread", "long_strangle",
    ]
    parser.add_argument(
        "--strategies", nargs="+", default=_DEFAULT_STRATEGIES,
        metavar="STRATEGY",
        help=(
            "Strategies to run per walk-forward window "
            "(default: iron_condor put_credit_spread short_strangle "
            "bull_call_spread bear_put_spread long_strangle)."
        ),
    )
    parser.add_argument(
        "--console-log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=(
            "Minimum log level printed to stdout (default: WARNING). "
            "DEBUG/INFO always captured in logs/ait.log regardless. "
            "Use WARNING to prevent stdout from growing large when running "
            "background experiments."
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Section A — Backfill
# ---------------------------------------------------------------------------

async def _section_a_backfill(args: argparse.Namespace, out: Path) -> int:
    """Fetch 2-year SPY 5-min bars from IB and store in test_intraday_prices."""
    from ib_insync import IB, Stock, util
    from ait.data.historical import HistoricalDataStore

    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=TABLE_PREFIX)
    years = args.years
    chunk_months = 6
    now = datetime.now(tz=timezone.utc)
    total_months = int(years * 12)

    chunks: list[tuple[datetime, str]] = []
    end = now
    remaining = total_months
    while remaining > 0:
        this = min(chunk_months, remaining)
        chunks.append((end, f"{this} M"))
        end = end - timedelta(days=this * 31)
        remaining -= this

    print(f"\n{'='*60}")
    print(f"SECTION A — Backfill ({years:.0f} years, {len(chunks)} chunks per symbol)")
    print(f"  DB: {args.db_path}  table: {TABLE_PREFIX}intraday_prices")
    print(f"{'='*60}")

    ib = IB()
    try:
        await ib.connectAsync(args.host, args.port, clientId=args.client_id, timeout=10)
    except Exception as e:
        print(f"  ERROR connecting to IBKR: {e}")
        return 0

    grand_total = 0
    for sym in args.symbols:
        contract = Stock(sym, "SMART", "USD")
        try:
            qualified_list = await ib.qualifyContractsAsync(contract)
            qualified = qualified_list[0] if qualified_list else None
        except Exception as e:
            print(f"  [{sym}] ERROR qualifying: {e}")
            continue

        if qualified is None:
            print(f"  [{sym}] ERROR: could not qualify contract")
            continue

        total_rows = 0
        for i, (end_dt, duration) in enumerate(chunks, start=1):
            end_str = end_dt.strftime("%Y%m%d %H:%M:%S") + " UTC"
            print(f"  [{sym}] chunk {i}/{len(chunks)}: {duration} ending {end_str}", end="", flush=True)

            bars = None
            for attempt in range(1, 3):          # up to 2 attempts per chunk
                try:
                    bars = await ib.reqHistoricalDataAsync(
                        qualified,
                        endDateTime=end_str,
                        durationStr=duration,
                        barSizeSetting="5 mins",
                        whatToShow="TRADES",
                        useRTH=True,
                        formatDate=1,
                        timeout=120,             # 2-minute timeout per chunk
                    )
                    break                        # success — exit retry loop
                except Exception as e:
                    print(f" (attempt {attempt} ERROR: {e})", end="", flush=True)
                    if attempt < 2:
                        print(f" retrying in 30s…", end="", flush=True)
                        await ib.sleepAsync(30)  # use ib_insync event-loop sleep

            if not bars:
                print(" 0 bars")
                # IB pacing: wait before next request even on failure
                if i < len(chunks):
                    await asyncio.sleep(15)
                continue

            df = util.df(bars).rename(columns={
                "date": "Datetime", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume",
            })
            df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
            df.set_index("Datetime", inplace=True)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            stored = store.save_intraday(sym, df, interval="5m")
            total_rows += stored
            print(f" → {len(df)} bars, {stored} rows upserted")
            # IB pacing: ≥10s between historical data requests to avoid Error 162
            if i < len(chunks):
                await asyncio.sleep(15)

        print(f"  [{sym}] backfill complete: {total_rows} total rows")
        grand_total += total_rows

    ib.disconnect()
    print(f"\nBackfill done. Grand total: {grand_total} rows across {len(args.symbols)} symbol(s).")
    return grand_total


# ---------------------------------------------------------------------------
# Section B — Data Quality
# ---------------------------------------------------------------------------

def _section_b_data_quality(args: argparse.Namespace, out: Path) -> dict:
    """Check coverage, gaps, price sanity. Returns QC metrics dict."""
    from ait.data.historical import HistoricalDataStore

    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=TABLE_PREFIX)

    print(f"\n{'='*60}")
    print("SECTION B — Data Quality")
    print(f"{'='*60}")

    results: dict[str, Any] = {}
    lines = ["DATA QUALITY REPORT", "=" * 60]

    for sym in args.symbols:
        df = store.load_intraday(sym, days=int(args.years * 366) + 10, interval="5m")
        total_bars = len(df)
        results[sym] = {"total_bars": total_bars}

        if df.empty:
            msg = f"[{sym}] NO DATA — backfill may have failed"
            print(f"  {msg}")
            lines.append(msg)
            continue

        # Date range
        date_min = df.index.min()
        date_max = df.index.max()
        date_range_days = (date_max - date_min).days
        trading_dates = sorted(set(df.index.date))
        trading_days = len(trading_dates)

        # Coverage (expected 78 bars per trading session for US equities, RTH only)
        expected_bars = trading_days * 78
        coverage_pct = 100.0 * total_bars / expected_bars if expected_bars > 0 else 0.0

        # Gap analysis: sessions with fewer than 60 bars
        bars_per_day = df.groupby(df.index.date).size()
        sparse_sessions = int((bars_per_day < 60).sum())

        # Price sanity
        neg_close = int((df["Close"] <= 0).sum())
        high_lt_low = int((df["High"] < df["Low"]).sum())

        results[sym].update({
            "date_min": str(date_min.date()),
            "date_max": str(date_max.date()),
            "date_range_days": date_range_days,
            "trading_days": trading_days,
            "coverage_pct": round(coverage_pct, 1),
            "sparse_sessions": sparse_sessions,
            "neg_close_count": neg_close,
            "high_lt_low_count": high_lt_low,
        })

        status = "PASS" if coverage_pct >= 80 and neg_close == 0 and high_lt_low == 0 else "WARN"
        summary = (
            f"[{sym}] [{status}] {total_bars:,} bars | "
            f"{trading_days} sessions | {date_min.date()} → {date_max.date()} | "
            f"coverage={coverage_pct:.1f}% | sparse={sparse_sessions} | "
            f"neg_close={neg_close} | high<low={high_lt_low}"
        )
        print(f"  {summary}")
        lines.append(summary)

    out.mkdir(parents=True, exist_ok=True)
    (out / "data_quality.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Written: {out}/data_quality.txt")
    return results


# ---------------------------------------------------------------------------
# Section C — Feature Health
# ---------------------------------------------------------------------------

def _section_c_feature_health(args: argparse.Namespace, out: Path) -> dict:
    """Compute NaN rates and distribution stats for all 18 features."""
    from ait.data.market_data import load_daily_ohlcv
    from ait.data.historical import HistoricalDataStore
    from ait.ml.features import FeatureEngine
    from ait.diagnostics.fractal_report import FRACTAL_FEATURE_COLS
    from ait.diagnostics.vlmc_report import VLMC_FEATURE_COLS

    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=TABLE_PREFIX)
    engine = FeatureEngine()

    print(f"\n{'='*60}")
    print("SECTION C — Feature Health")
    print(f"{'='*60}")

    lines = ["FEATURE HEALTH REPORT", "=" * 60]
    all_stats: dict[str, Any] = {}

    for sym in args.symbols:
        lines.append(f"\n--- {sym} ---")

        # Fractal features from daily bars (IB store first, Yahoo fallback)
        try:
            daily_df = load_daily_ohlcv(sym, days=int(args.years * 366) + 30,
                                        db_path=Path(args.db_path))
            feat_df = engine.compute(daily_df)
        except Exception as e:
            lines.append(f"  fractal compute ERROR: {e}")
            feat_df = pd.DataFrame()

        sym_stats: dict[str, dict] = {}

        for col in FRACTAL_FEATURE_COLS:
            if col not in feat_df.columns:
                sym_stats[col] = {"present": False}
                lines.append(f"  [MISSING] {col}")
                continue
            series = feat_df[col]
            nan_pct = 100.0 * series.isna().sum() / max(len(series), 1)
            sym_stats[col] = {
                "present": True,
                "nan_pct": round(nan_pct, 2),
                "min": round(float(series.min()), 5),
                "max": round(float(series.max()), 5),
                "mean": round(float(series.mean()), 5),
                "std": round(float(series.std()), 5),
            }
            flag = " [HIGH NAN]" if nan_pct > 5 else ""
            lines.append(
                f"  [fractal] {col}: nan={nan_pct:.1f}%{flag}, "
                f"range=[{series.min():.4f}, {series.max():.4f}], mean={series.mean():.4f}"
            )

        # VLMC features from intraday sessions
        intraday = store.load_intraday(sym, days=int(args.years * 366) + 10, interval="5m")
        vlmc_rows: list[dict] = []
        if not intraday.empty:
            dates = sorted(set(intraday.index.date))
            for d in dates:
                session = intraday[intraday.index.date == d]
                if len(session) < 10:
                    continue
                try:
                    feats = engine.compute_intraday_features(session)
                    vlmc_rows.append({k: feats.get(k, np.nan) for k in VLMC_FEATURE_COLS})
                except Exception:
                    pass

        vlmc_df = pd.DataFrame(vlmc_rows) if vlmc_rows else pd.DataFrame(columns=VLMC_FEATURE_COLS)

        for col in VLMC_FEATURE_COLS:
            if col not in vlmc_df.columns or vlmc_df.empty:
                sym_stats[col] = {"present": False}
                lines.append(f"  [MISSING] {col}")
                continue
            series = vlmc_df[col]
            nan_pct = 100.0 * series.isna().sum() / max(len(series), 1)
            sym_stats[col] = {
                "present": True,
                "nan_pct": round(nan_pct, 2),
                "min": round(float(series.min()), 5),
                "max": round(float(series.max()), 5),
                "mean": round(float(series.mean()), 5),
                "std": round(float(series.std()), 5),
            }
            flag = " [HIGH NAN]" if nan_pct > 5 else ""
            lines.append(
                f"  [vlmc   ] {col}: nan={nan_pct:.1f}%{flag}, "
                f"range=[{series.min():.4f}, {series.max():.4f}], mean={series.mean():.4f}"
            )

        all_stats[sym] = sym_stats

    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_health.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Written: {out}/feature_health.txt")
    return all_stats


# ---------------------------------------------------------------------------
# Section D — IC Decay
# ---------------------------------------------------------------------------

def _section_d_ic_decay(args: argparse.Namespace, out: Path) -> pd.DataFrame:
    """Compute Spearman IC at 1/3/5/10/20-day forward return horizons."""
    from ait.data.market_data import load_daily_ohlcv
    from scipy.stats import spearmanr
    from ait.data.historical import HistoricalDataStore
    from ait.ml.features import FeatureEngine
    from ait.diagnostics.fractal_report import FRACTAL_FEATURE_COLS
    from ait.diagnostics.vlmc_report import VLMC_FEATURE_COLS

    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=TABLE_PREFIX)
    engine = FeatureEngine()
    horizons = [1, 3, 5, 10, 20]

    print(f"\n{'='*60}")
    print("SECTION D — IC Decay Curve")
    print(f"{'='*60}")

    ic_rows: list[dict] = []

    for sym in args.symbols:
        try:
            daily_df = load_daily_ohlcv(sym, days=int(args.years * 366) + 30,
                                        db_path=Path(args.db_path))
            feat_df = engine.compute(daily_df)
        except Exception as e:
            print(f"  [{sym}] feature compute error: {e}")
            continue

        # Build VLMC feature series (one value per day)
        intraday = store.load_intraday(sym, days=int(args.years * 366) + 10, interval="5m")
        vlmc_series: dict[str, pd.Series] = {}
        if not intraday.empty:
            dates = sorted(set(intraday.index.date))
            rows: list[dict] = []
            for d in dates:
                session = intraday[intraday.index.date == d]
                if len(session) < 10:
                    continue
                try:
                    feats = engine.compute_intraday_features(session)
                    row = {k: feats.get(k, np.nan) for k in VLMC_FEATURE_COLS}
                    row["_date"] = pd.Timestamp(d)
                    rows.append(row)
                except Exception:
                    pass
            if rows:
                tmp = pd.DataFrame(rows).set_index("_date")
                for col in VLMC_FEATURE_COLS:
                    if col in tmp.columns:
                        vlmc_series[col] = tmp[col]

        # Compute IC at each horizon for each feature
        close = daily_df["Close"]
        for h in horizons:
            fwd_ret = close.pct_change(h).shift(-h)

            for col in FRACTAL_FEATURE_COLS:
                if col not in feat_df.columns:
                    continue
                x = feat_df[col]
                y = fwd_ret.reindex(x.index)
                valid = x.notna() & y.notna()
                if valid.sum() < 10:
                    continue
                corr, pval = spearmanr(x[valid].values, y[valid].values)
                ic_rows.append({
                    "symbol": sym, "feature": col, "feature_type": "fractal",
                    "horizon_days": h,
                    "ic": round(float(corr) if np.isfinite(corr) else 0.0, 4),
                    "p_value": round(float(pval), 4),
                    "n_obs": int(valid.sum()),
                })

            for col, series in vlmc_series.items():
                # Align to daily date index
                y = fwd_ret.copy()
                y.index = y.index.normalize()
                x_aligned = series.reindex(y.index)
                valid = x_aligned.notna() & y.notna()
                if valid.sum() < 10:
                    continue
                corr, pval = spearmanr(x_aligned[valid].values, y[valid].values)
                ic_rows.append({
                    "symbol": sym, "feature": col, "feature_type": "vlmc",
                    "horizon_days": h,
                    "ic": round(float(corr) if np.isfinite(corr) else 0.0, 4),
                    "p_value": round(float(pval), 4),
                    "n_obs": int(valid.sum()),
                })

    ic_df = pd.DataFrame(ic_rows)
    out.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(out / "ic_decay.csv", index=False)

    # Print summary table
    if not ic_df.empty:
        print(f"\n  IC at 5-day horizon (|IC| ≥ 0.02 highlighted):")
        h5 = ic_df[ic_df["horizon_days"] == 5].sort_values("ic", key=abs, ascending=False)
        for _, row in h5.iterrows():
            flag = " ◀" if abs(row["ic"]) >= 0.02 else ""
            sig = "*" if row["p_value"] < 0.05 else " "
            print(f"  {row['feature_type']:8s} {row['feature']:35s} IC={row['ic']:+.4f} {sig} p={row['p_value']:.3f}{flag}")

    print(f"\n  → Written: {out}/ic_decay.csv ({len(ic_df)} rows)")
    return ic_df


# ---------------------------------------------------------------------------
# Section E — Walk-Forward (with optimization)
# ---------------------------------------------------------------------------

def _fetch_daily_data(args: argparse.Namespace) -> "dict[str, pd.DataFrame]":
    """Load daily OHLCV once (test_intraday_prices table) so E & F share identical date ranges.

    Uses HistoricalDataStore directly with TABLE_PREFIX so the walk-forward receives the full
    backfilled history (test_intraday_prices), not the shorter production intraday_prices table.
    fetch_days covers train_days + test_days + buffer to support long training windows.
    """
    from ait.data.historical import HistoricalDataStore

    fetch_days = max(int(args.years * 366) + 30, args.train_days + args.test_days + 100)
    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=TABLE_PREFIX)
    ticker_data: dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        df = store.resample_to_daily(sym, days=fetch_days)
        if not df.empty:
            ticker_data[sym] = df
    return ticker_data


async def _section_e_walkforward(
    args: argparse.Namespace, out: Path, ticker_data: "dict[str, pd.DataFrame]"
) -> dict:
    """Run walk-forward backtest with per-window Optuna optimization."""
    from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

    from ait.config.settings import load_settings
    _settings = load_settings(args.config)
    _bc = _settings.backtest

    _n_trials   = args.wf_trials     if args.wf_trials     is not None else _bc.optimize_n_trials
    _patience   = args.wf_patience   if args.wf_patience   is not None else _bc.optimize_patience
    _min_trades = args.wf_min_trades if args.wf_min_trades is not None else _bc.optimize_min_trades
    _n_jobs     = args.wf_n_jobs     if args.wf_n_jobs     is not None else 1
    _val_split  = args.wf_val_split

    print(f"\n{'='*60}")
    print(f"SECTION E — Walk-Forward (optimize_per_window=True, trials={_n_trials}, patience={_patience}, min_trades={_min_trades})")
    print(f"{'='*60}")

    config = WalkForwardConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        gap_days=args.gap_days,
        optimize_per_window=True,
        optimize_n_trials=_n_trials,
        optimize_patience=_patience,
        optimize_min_trades=_min_trades,
        optimize_n_jobs=_n_jobs,
        optimize_seed=args.optuna_seed,
        optimize_val_split=_val_split,
        initial_capital=_bc.initial_capital,
        position_size_pct=_bc.position_size_pct,
        wing_floor_dollars=_bc.wing_floor_dollars,
        wing_k=_bc.wing_k,
        iv_floor=_bc.iv_floor,
        delta_iv_scale=_bc.delta_iv_scale,
        spread_base=_bc.spread_base,
        spread_iv_sensitivity=_bc.spread_iv_sensitivity,
        spread_dte_sensitivity=_bc.spread_dte_sensitivity,
        spread_cap=_bc.spread_cap,
    )

    try:
        out.mkdir(parents=True, exist_ok=True)
        bt = WalkForwardBacktester(args.symbols, args.strategies, config=config,
                                   db_path=Path(args.db_path), progress_dir=out,
                                   table_prefix=TABLE_PREFIX)
        result = await bt.run(data=ticker_data if ticker_data else None)

        summary_text = result.summary()
        print(f"\n{summary_text}")

        (out / "walkforward_summary.txt").write_text(summary_text, encoding="utf-8")
        ec = result.equity_curve()
        if not ec.empty:
            ec.to_csv(out / "equity_curve.csv", index=False)
        print(f"  → Written: {out}/walkforward_summary.txt")

        return {
            "total_trades": result.total_trades,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "consistency": result.consistency,
            "n_windows": len(result.windows),
            "profit_factor": result.profit_factor,
        }

    except Exception as e:
        print(f"  Walk-forward ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Section F — Ablation (without intraday/fractal features)
# ---------------------------------------------------------------------------

async def _section_f_ablation(
    args: argparse.Namespace, out: Path, ticker_data: "dict[str, pd.DataFrame]"
) -> dict:
    """Walk-forward baseline: no per-window optimization, default config."""
    from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

    print(f"\n{'='*60}")
    print("SECTION F — Ablation (baseline: no optimization)")
    print(f"{'='*60}")

    from ait.config.settings import load_settings
    _settings = load_settings(args.config)
    _bc = _settings.backtest

    _min_trades = args.wf_min_trades if args.wf_min_trades is not None else _bc.optimize_min_trades

    config = WalkForwardConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        gap_days=args.gap_days,
        optimize_per_window=False,
        optimize_min_trades=_min_trades,
        initial_capital=_bc.initial_capital,
        position_size_pct=_bc.position_size_pct,
        wing_floor_dollars=_bc.wing_floor_dollars,
        wing_k=_bc.wing_k,
        iv_floor=_bc.iv_floor,
        delta_iv_scale=_bc.delta_iv_scale,
        spread_base=_bc.spread_base,
        spread_iv_sensitivity=_bc.spread_iv_sensitivity,
        spread_dte_sensitivity=_bc.spread_dte_sensitivity,
        spread_cap=_bc.spread_cap,
    )

    try:
        bt = WalkForwardBacktester(args.symbols, args.strategies, config=config,
                                   db_path=Path(args.db_path), table_prefix=TABLE_PREFIX)
        result = await bt.run(data=ticker_data if ticker_data else None)

        summary_text = result.summary()
        print(f"\n{summary_text}")

        out.mkdir(parents=True, exist_ok=True)
        (out / "ablation_summary.txt").write_text(summary_text, encoding="utf-8")

        return {
            "total_trades": result.total_trades,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
            "consistency": result.consistency,
        }

    except Exception as e:
        print(f"  Ablation ERROR: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Section G — Diagnostic Reports
# ---------------------------------------------------------------------------

def _section_g_reports(args: argparse.Namespace, out: Path) -> None:
    """Generate fractal + VLMC HTML diagnostic report."""
    from ait.diagnostics.report import generate_report

    end_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print("SECTION G — Diagnostic Reports")
    print(f"{'='*60}")

    try:
        generate_report(
            symbols=args.symbols,
            start=START_DATE,
            end=end_date,
            output_dir=str(out),
            fmt="html",
            db_path=args.db_path,
            table_prefix=TABLE_PREFIX,
        )
        html_files = list(out.glob("fractal_report_*.html"))
        csv_files = list(out.glob("ic_summary.csv"))
        print(f"  HTML reports: {[f.name for f in html_files]}")
        print(f"  IC summary CSV: {[f.name for f in csv_files]}")
    except Exception as e:
        print(f"  Report generation ERROR: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Section H — Document Results
# ---------------------------------------------------------------------------

def _section_h_document(
    args: argparse.Namespace,
    out: Path,
    qc: dict,
    feat_health: dict,
    ic_df: pd.DataFrame,
    wf: dict,
    ablation: dict,
    elapsed_total: float,
) -> None:
    """Write RESULTS.md with all metrics and pass/fail assessment."""

    lines = [
        "# Integration Test Results",
        "",
        f"**Date**: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Symbols**: {', '.join(args.symbols)}",
        f"**Intraday history**: {args.years:.0f} years of 5-min bars from IB",
        f"**Database**: `{args.db_path}` (tables: `{TABLE_PREFIX}intraday_prices`, `{TABLE_PREFIX}daily_prices`)",
        f"**Total run time**: {elapsed_total/60:.1f} minutes",
        "",
        "---",
        "",
        "## A. Data Quality",
        "",
    ]

    for sym, stats in qc.items():
        if "total_bars" not in stats:
            lines.append(f"- **{sym}**: NO DATA")
            continue
        coverage = stats.get("coverage_pct", 0)
        status = "✅ PASS" if coverage >= 80 and stats.get("neg_close_count", 0) == 0 else "⚠️ WARN"
        lines += [
            f"- **{sym}** {status}",
            f"  - Bars: {stats['total_bars']:,} | Sessions: {stats['trading_days']}",
            f"  - Date range: {stats.get('date_min')} → {stats.get('date_max')}",
            f"  - Coverage: {coverage}% (threshold: ≥ 80%)",
            f"  - Sparse sessions (< 60 bars): {stats.get('sparse_sessions', '?')}",
            f"  - Price anomalies: neg_close={stats.get('neg_close_count', 0)}, high<low={stats.get('high_lt_low_count', 0)}",
        ]

    lines += ["", "---", "", "## B. Feature Health (18 features: 5 fractal + 13 VLMC)", ""]
    for sym, feat_stats in feat_health.items():
        lines.append(f"### {sym}")
        missing = [k for k, v in feat_stats.items() if not v.get("present", True)]
        high_nan = [k for k, v in feat_stats.items() if v.get("nan_pct", 0) > 5]
        lines += [
            f"- Missing features: {missing if missing else 'None ✅'}",
            f"- High NaN (> 5%): {high_nan if high_nan else 'None ✅'}",
        ]

    lines += ["", "---", "", "## C. IC Decay Curve", ""]
    if not ic_df.empty:
        for h in [1, 3, 5, 10, 20]:
            h_df = ic_df[ic_df["horizon_days"] == h]
            if h_df.empty:
                continue
            sig = h_df[h_df["p_value"] < 0.05]
            best = h_df.loc[h_df["ic"].abs().idxmax()]
            lines += [
                f"**{h}-day horizon**: {len(sig)}/{len(h_df)} features significant (p<0.05) | "
                f"best IC: {best['feature']} = {best['ic']:+.4f}",
            ]
    else:
        lines.append("IC decay data not available.")

    lines += ["", "---", "", "## D. Walk-Forward (multi-strategy, optimize_per_window=True)", ""]
    if "error" in wf:
        lines.append(f"⚠️ Walk-forward failed: {wf['error']}")
    else:
        sharpe_ok = wf.get("sharpe_ratio", -99) > -1
        trades_ok = wf.get("total_trades", 0) > 0
        wf_status = "✅ PASS" if sharpe_ok and trades_ok else "⚠️ WARN"
        lines += [
            f"{wf_status}",
            f"- Windows: {wf.get('n_windows', '?')}",
            f"- Total trades: {wf.get('total_trades', 0)}",
            f"- Total return: {100*wf.get('total_return', 0):+.1f}%",
            f"- Sharpe ratio: {wf.get('sharpe_ratio', 0):.3f} (threshold: > -1.0)",
            f"- Max drawdown: {100*wf.get('max_drawdown', 0):.1f}%",
            f"- Win rate: {100*wf.get('win_rate', 0):.1f}%",
            f"- Consistency: {100*wf.get('consistency', 0):.1f}% profitable windows",
        ]

    lines += ["", "---", "", "## E. Ablation (baseline: no optimization)", ""]
    if "error" in ablation:
        lines.append(f"⚠️ Ablation failed: {ablation['error']}")
    else:
        wf_sharpe = wf.get("sharpe_ratio", 0)
        ab_sharpe = ablation.get("sharpe_ratio", 0)
        delta = wf_sharpe - ab_sharpe
        lines += [
            f"- Sharpe (optimized): {wf_sharpe:.3f} vs. baseline: {ab_sharpe:.3f} → delta: {delta:+.3f}",
            f"- Return (optimized): {100*wf.get('total_return',0):+.1f}% vs. baseline: {100*ablation.get('total_return',0):+.1f}%",
            f"- Win rate (optimized): {100*wf.get('win_rate',0):.1f}% vs. baseline: {100*ablation.get('win_rate',0):.1f}%",
        ]
        if delta > 0:
            lines.append("- ✅ Per-window optimization improves Sharpe over baseline.")
        else:
            lines.append("- ℹ️ Per-window optimization did not improve Sharpe in this test run.")

    lines += [
        "",
        "---",
        "",
        "## F. Output Files",
        "",
        f"| File | Description |",
        f"|------|-------------|",
        f"| `data_quality.txt` | Coverage, gaps, price sanity per symbol |",
        f"| `feature_health.txt` | NaN rates and ranges for all 18 features |",
        f"| `ic_decay.csv` | Spearman IC at 1/3/5/10/20-day horizons |",
        f"| `walkforward_summary.txt` | Walk-forward metrics and per-window breakdown |",
        f"| `equity_curve.csv` | Trade-level equity curve from walk-forward |",
        f"| `ablation_summary.txt` | Baseline walk-forward (no optimization) |",
        f"| `fractal_report_{args.symbols[0]}.html` | Interactive fractal + VLMC diagnostic plots |",
        f"| `ic_summary.csv` | Aggregated IC from diagnostic report pipeline |",
        f"| `RESULTS.md` | This file |",
        "",
        "---",
        "",
        "## G. Pass/Fail Summary",
        "",
        "| Check | Criterion | Status |",
        "|-------|-----------|--------|",
    ]

    def _pf(ok: bool) -> str:
        return "✅ PASS" if ok else "⚠️ WARN"

    first_sym = args.symbols[0]
    qc_s = qc.get(first_sym, {})
    fh_s = feat_health.get(first_sym, {})

    lines += [
        f"| Data coverage | ≥ 80% | {_pf(qc_s.get('coverage_pct', 0) >= 80)} |",
        f"| No negative prices | 0 anomalies | {_pf(qc_s.get('neg_close_count', 0) == 0)} |",
        f"| All 18 features present | 0 missing | {_pf(not any(not v.get('present', True) for v in fh_s.values()))} |",
        f"| Feature NaN rate | < 10% | {_pf(not any(v.get('nan_pct', 0) > 10 for v in fh_s.values()))} |",
    ]
    if not ic_df.empty:
        h5 = ic_df[(ic_df['horizon_days'] == 5) & (ic_df['p_value'] < 0.10)]
        lines.append(f"| IC significance at 5d | ≥ 3 features p<0.10 | {_pf(len(h5) >= 3)} |")
    if "error" not in wf:
        lines.append(f"| Walk-forward trades | > 0 | {_pf(wf.get('total_trades', 0) > 0)} |")
        lines.append(f"| Walk-forward Sharpe | > -1.0 | {_pf(wf.get('sharpe_ratio', -99) > -1)} |")

    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → Written: {out}/RESULTS.md")


# ---------------------------------------------------------------------------
# Section I — Archive run + MLflow logging
# ---------------------------------------------------------------------------

def _create_run_archive(
    args: argparse.Namespace,
    out: Path,
    wf: dict,
    wf_cfg_fields: dict,
) -> Path | None:
    """Copy integration-test outputs to a versioned reports/runs/ subdirectory
    and log the run to MLflow.  Returns the archive path, or None on error."""
    import shutil
    import subprocess

    try:
        now_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")
        symbol = args.symbols[0] if args.symbols else "UNKNOWN"
        strategies_str = "_".join(sorted(args.strategies)) if args.strategies else "all"
        run_id = f"{symbol}_{wf_cfg_fields.get('train_days',365)}d_{strategies_str}_{now_str}"

        runs_dir = Path("reports/runs") / run_id
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Copy all artifacts from integration_test output dir
        for src in out.iterdir():
            dst = runs_dir / src.name
            if src.is_file():
                shutil.copy2(src, dst)

        # Config snapshot
        cfg_src = Path(args.config)
        if cfg_src.exists():
            shutil.copy2(cfg_src, runs_dir / "config_snapshot.yaml")

        # Git metadata
        try:
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip()
        except Exception:
            git_branch = "unknown"
            git_commit = "unknown"

        # Per-window data from window_*.json files
        windows = []
        for wf_file in sorted(out.glob("window_*.json")):
            try:
                w = json.loads(wf_file.read_text())
                windows.append(w)
            except Exception:
                pass

        # Compute date range from windows
        dates = [w.get("test_start") for w in windows if w.get("test_start")]
        dates += [w.get("test_end") for w in windows if w.get("test_end")]
        backtest_period = f"{min(dates)} to {max(dates)}" if dates else "unknown"

        active_windows = sum(1 for w in windows if w.get("trades", 0) > 0)

        summary = {
            "total_trades":    wf.get("total_trades", 0),
            "total_pnl":       wf.get("total_return", 0.0) * wf_cfg_fields.get("initial_capital", 100_000),
            "total_return_pct": 100 * wf.get("total_return", 0.0),
            "win_rate":        wf.get("win_rate", 0.0),
            "sharpe_ratio":    wf.get("sharpe_ratio", 0.0),
            "max_drawdown_pct": abs(wf.get("max_drawdown", 0.0)),
            "profit_factor":   wf.get("profit_factor", 0.0),
        }

        # Reconstruct the CLI command from args for reproducibility
        cli_parts = [
            "python scripts/run_integration_test.py",
            f"--symbols {' '.join(args.symbols)}",
            f"--config {args.config}",
            f"--years {args.years:.0f}",
            f"--port {args.port}",
            f"--strategies {' '.join(args.strategies)}",
            f"--train-days {args.train_days}",
            f"--test-days {args.test_days}",
            f"--step-days {args.step_days}",
            f"--gap-days {args.gap_days}",
            f"--optuna-seed {args.optuna_seed}",
        ]
        if args.wf_trials is not None:
            cli_parts.append(f"--wf-trials {args.wf_trials}")
        if args.wf_patience is not None:
            cli_parts.append(f"--wf-patience {args.wf_patience}")
        if args.skip_backfill:
            cli_parts.append("--skip-backfill")
        cli_command = " ".join(cli_parts)

        # Serialize search space bounds for the strategies used in this run
        from ait.optimization.param_spaces import STRATEGY_SPACES
        search_space = {
            strat: {k: list(v) for k, v in STRATEGY_SPACES[strat].items()}
            for strat in args.strategies
            if strat in STRATEGY_SPACES
        }

        metadata = {
            "run_id":           run_id,
            "run_date":         datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "symbol":           symbol,
            "strategy":         strategies_str,
            "optimization":     "per_strategy",
            "n_windows":        len(windows),
            "active_windows":   active_windows,
            "train_days":       wf_cfg_fields.get("train_days", 365),
            "test_days":        wf_cfg_fields.get("test_days", 63),
            "step_days":        wf_cfg_fields.get("step_days", 21),
            "gap_days":         wf_cfg_fields.get("gap_days", 5),
            "wf_trials":        wf_cfg_fields.get("optimize_n_trials", 50),
            "optuna_seed":      wf_cfg_fields.get("optuna_seed", 42),
            "initial_capital":  wf_cfg_fields.get("initial_capital", 100_000.0),
            "position_size_pct": wf_cfg_fields.get("position_size_pct", 0.05),
            "search_space":     search_space,
            "backtest_period":  backtest_period,
            "cli_command":      cli_command,
            "git_branch":       git_branch,
            "git_commit":       git_commit,
            "summary":          summary,
            "windows":          windows,
        }

        (runs_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"\n  → Run archived: {runs_dir.resolve()}")
        print(f"    run_id={run_id}, windows={len(windows)}, active={active_windows}, "
              f"trades={wf.get('total_trades', 0)}")

        # MLflow logging (optional — skipped gracefully if mlflow not installed)
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            tracking_uri = (
                __import__("os").environ.get("MLFLOW_TRACKING_URI")
                or "sqlite:///data/mlflow.db"
            )
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(f"walkforward_{symbol}")
            with mlflow.start_run(
                run_name=run_id,
                tags={
                    "run_id": run_id, "symbol": symbol, "strategy": strategies_str,
                    "git_commit": git_commit, "git_branch": git_branch,
                },
            ):
                mlflow.log_params({
                    "symbol":            symbol,
                    "strategy":          strategies_str,
                    "train_days":        wf_cfg_fields.get("train_days", 365),
                    "test_days":         wf_cfg_fields.get("test_days", 63),
                    "step_days":         wf_cfg_fields.get("step_days", 21),
                    "gap_days":          wf_cfg_fields.get("gap_days", 5),
                    "wf_trials":         wf_cfg_fields.get("optimize_n_trials", 50),
                    "optuna_seed":       wf_cfg_fields.get("optuna_seed", 42),
                    "initial_capital":   wf_cfg_fields.get("initial_capital", 100_000.0),
                    "position_size_pct": wf_cfg_fields.get("position_size_pct", 0.05),
                    "optimization":      "per_strategy",
                    "backtest_period":   backtest_period,
                    "search_space":      json.dumps(search_space),
                })
                mlflow.set_tag("cli_command", cli_command)
                mlflow.log_metrics({
                    "total_pnl":        summary["total_pnl"],
                    "total_return_pct": summary["total_return_pct"],
                    "win_rate":         summary["win_rate"],
                    "sharpe_ratio":     summary["sharpe_ratio"],
                    "max_drawdown_pct": summary["max_drawdown_pct"],
                    "total_trades":     float(summary["total_trades"]),
                })
                for w in windows:
                    step = w.get("window", 0)
                    mlflow.log_metrics({
                        "w_pnl":      w.get("pnl", 0.0),
                        "w_trades":   float(w.get("trades", 0)),
                        "w_win_rate": w.get("win_rate", 0.0),
                        "w_sharpe":   w.get("sharpe", 0.0),
                    }, step=step)
                mlflow.log_artifacts(str(runs_dir))
            print(f"    MLflow: logged to experiment 'walkforward_{symbol}'")
        except ImportError:
            pass  # mlflow not installed — silently skip
        except Exception as e:
            print(f"    MLflow logging skipped: {e}")

        return runs_dir

    except Exception as e:
        print(f"\n  Archive ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> int:
    # Configure logging early: DEBUG/INFO go to logs/ait.log; stdout gets
    # console_log_level (default WARNING) to prevent the tmp task-output
    # buffer from filling up during long experiments.
    from ait.config.settings import LoggingConfig
    from ait.utils.logging import setup_logging
    setup_logging(
        LoggingConfig(level="DEBUG", file="logs/ait.log"),
        console_level=getattr(args, "console_log_level", "WARNING"),
    )

    out = Path(args.output_dir)
    t_start = time.time()

    # A — Backfill
    if not args.skip_backfill:
        await _section_a_backfill(args, out)
    else:
        print("\nSECTION A — Backfill SKIPPED (--skip-backfill)")

    # B — Data Quality
    qc = _section_b_data_quality(args, out)

    # C — Feature Health
    feat_health = _section_c_feature_health(args, out)

    # D — IC Decay
    ic_df = _section_d_ic_decay(args, out)

    # E & F — Walk-Forward + Ablation (shared data fetch so window counts match)
    wf: dict = {}
    ablation: dict = {}
    wf_cfg_fields: dict = {}
    if not args.skip_walkforward:
        print("\nFetching daily OHLCV for walk-forward sections…")
        shared_daily_data = _fetch_daily_data(args)
        wf = await _section_e_walkforward(args, out, shared_daily_data)
        ablation = await _section_f_ablation(args, out, shared_daily_data)

        # Capture WalkForwardConfig fields for archival
        from ait.config.settings import load_settings
        _bc = load_settings(args.config).backtest
        _n_trials = args.wf_trials if args.wf_trials is not None else _bc.optimize_n_trials
        wf_cfg_fields = {
            "train_days": args.train_days,
            "test_days":  args.test_days,
            "step_days":  args.step_days,
            "gap_days":   args.gap_days,
            "optimize_n_trials": _n_trials,
            "optuna_seed": args.optuna_seed,
            "initial_capital": _bc.initial_capital,
            "position_size_pct": _bc.position_size_pct,
        }
    else:
        print("\nSECTIONS E & F — Walk-forward SKIPPED (--skip-walkforward)")

    # G — Diagnostic Reports
    _section_g_reports(args, out)

    # H — Document
    elapsed = time.time() - t_start
    _section_h_document(args, out, qc, feat_health, ic_df, wf, ablation, elapsed)

    # I — Archive + MLflow (only when walk-forward ran and produced trades)
    if wf and "error" not in wf and wf.get("total_trades", 0) > 0:
        _create_run_archive(args, out, wf, wf_cfg_fields)

    print(f"\n{'='*60}")
    print(f"Integration test complete in {elapsed/60:.1f} minutes.")
    print(f"Results written to: {out.resolve()}")
    print(f"{'='*60}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())

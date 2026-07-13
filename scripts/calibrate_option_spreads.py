"""Calibrate options bid-ask spread model parameters from real market data.

Algorithm:
  1. Download current options chain via yfinance (default) or IBKR snapshot (--source ibkr).
  2. For each DTE target in --dte-targets, find the nearest available expiry and
     collect all strikes with valid bid/ask quotes.
  3. Compute half_spread_pct = (ask - bid) / (2 × mid) per contract.
  4. Store raw samples in option_spread_samples table.
  5. Fit parametric model:
       half_spread_pct = base + iv_sens × max(0, IV - iv_thresh)
                               + dte_sens × max(0, dte_thresh - DTE)
     using scipy.optimize.curve_fit.
  6. Store fitted params in option_spread_params table.
  7. Optionally auto-write calibrated values into a YAML config file.

Usage:
    # Collect data + fit + update YAML (yfinance, no IBKR needed):
    python scripts/calibrate_option_spreads.py \\
        --symbols QQQ \\
        --db-path data/historical.db \\
        --update-config deprecated/configs/config_QQQ_test.yaml

    # Dry-run (shows request plan only):
    python scripts/calibrate_option_spreads.py --symbols QQQ --dry-run

    # IBKR snapshot mode (requires live/paper account + options data subscription):
    python scripts/calibrate_option_spreads.py \\
        --symbols QQQ --source ibkr \\
        --db-path data/historical.db \\
        --port 4002 --client-id 91 \\
        --update-config deprecated/configs/config_QQQ_test.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


# ---------------------------------------------------------------------------
# Model formula (also imported by tests)
# ---------------------------------------------------------------------------

def half_spread_model(
    X: tuple,
    base: float,
    iv_sens: float,
    iv_thresh: float,
    dte_sens: float,
    dte_thresh: float,
) -> np.ndarray:
    """Parametric half-spread model.

    X is a tuple of (iv_array, dte_array) from curve_fit.
    Returns predicted half_spread_pct values.
    """
    iv_arr, dte_arr = X
    iv_arr = np.asarray(iv_arr, dtype=float)
    dte_arr = np.asarray(dte_arr, dtype=float)
    return (
        base
        + iv_sens * np.maximum(0.0, iv_arr - iv_thresh)
        + dte_sens * np.maximum(0.0, dte_thresh - dte_arr)
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calibrate_option_spreads",
        description="Calibrate option spread model params from real market data.",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["QQQ"], metavar="SYMBOL",
        help="Tickers to calibrate (default: QQQ).",
    )
    parser.add_argument(
        "--dte-targets", nargs="+", type=int, default=[7, 14, 21, 30, 45],
        metavar="DTE",
        help="DTE levels to sample (default: 7 14 21 30 45).",
    )
    parser.add_argument(
        "--source", choices=["yfinance", "ibkr"], default="yfinance",
        help="Market data source (default: yfinance — no subscription needed).",
    )
    parser.add_argument(
        "--max-moneyness", type=float, default=0.30, metavar="FRAC",
        help="Max |strike/spot - 1| to include in fit (default: 0.30 = 30%% OTM).",
    )
    parser.add_argument(
        "--min-mid", type=float, default=0.05, metavar="DOLLARS",
        help="Min option mid price to include (default: $0.05 — filters penny options).",
    )
    parser.add_argument(
        "--db-path", default="data/historical.db", metavar="PATH",
        help="SQLite DB path (default: data/historical.db).",
    )
    # IBKR options (only used with --source ibkr)
    parser.add_argument("--host", default="127.0.0.1", help="IBKR host.")
    parser.add_argument("--port", type=int, default=4002, help="IBKR port (default: 4002).")
    parser.add_argument("--client-id", type=int, default=91, metavar="ID")
    parser.add_argument(
        "--snapshot-timeout", type=float, default=8.0, metavar="SECS",
        help="Seconds to wait for IBKR snapshot per contract (default: 8).",
    )
    parser.add_argument(
        "--pause-secs", type=float, default=1.0, metavar="SECS",
        help="Pause between IBKR requests (default: 1.0 s).",
    )
    parser.add_argument(
        "--update-config", metavar="PATH",
        help="YAML config file to auto-update with calibrated spread values.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print request plan without fetching or storing data.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# yfinance-based calibration (default, no subscription needed)
# ---------------------------------------------------------------------------

def _calibrate_symbol_yfinance(
    symbol: str,
    dte_targets: list[int],
    store,
    dry_run: bool,
    max_moneyness: float = 0.30,
    min_mid: float = 0.05,
) -> int:
    """Collect spread samples from yfinance option chain. Returns rows stored."""
    import yfinance as yf

    today = date.today()
    ticker = yf.Ticker(symbol)

    # Get current spot price for moneyness filter
    try:
        spot_price = float(ticker.fast_info["last_price"])
    except Exception:
        spot_price = None

    # Available expiries as "YYYY-MM-DD" strings
    try:
        expiry_strs = ticker.options
    except Exception as e:
        print(f"  [{symbol}] ERROR fetching option expiries: {e}")
        return 0

    if not expiry_strs:
        print(f"  [{symbol}] No option expiries returned by yfinance.")
        return 0

    expiry_dates = sorted(
        datetime.strptime(es, "%Y-%m-%d").date() for es in expiry_strs
    )
    expiry_dates = [d for d in expiry_dates if d > today]

    if not expiry_dates:
        print(f"  [{symbol}] No future expiries found.")
        return 0

    print(
        f"  [{symbol}] {len(expiry_dates)} future expiries "
        f"(nearest: {expiry_dates[0]}, farthest: {expiry_dates[-1]})"
    )

    if dry_run:
        for dte_target in dte_targets:
            nearest = min(expiry_dates, key=lambda d: abs((d - today).days - dte_target))
            actual_dte = (nearest - today).days
            print(
                f"  [{symbol}] DTE target {dte_target:3d} → expiry {nearest} "
                f"(actual DTE={actual_dte}) → would download full chain"
            )
        return 0

    all_rows: list[dict] = []

    for dte_target in dte_targets:
        nearest = min(expiry_dates, key=lambda d: abs((d - today).days - dte_target))
        actual_dte = (nearest - today).days
        expiry_str = nearest.strftime("%Y-%m-%d")

        try:
            chain = ticker.option_chain(expiry_str)
        except Exception as e:
            print(f"  [{symbol}] DTE {actual_dte} ({expiry_str}): chain fetch error: {e}")
            continue

        rows_this_dte = 0
        for df_side, right in [(chain.calls, "C"), (chain.puts, "P")]:
            for _, row in df_side.iterrows():
                try:
                    bid = float(row.get("bid") or 0)
                    ask = float(row.get("ask") or 0)
                    iv = float(row.get("impliedVolatility") or 0)
                    strike = float(row["strike"])
                except (TypeError, ValueError):
                    continue

                if bid <= 0 or ask <= 0 or ask < bid:
                    continue
                if iv <= 0 or not math.isfinite(iv) or iv > 5.0:
                    continue

                mid = (bid + ask) / 2.0
                if mid < min_mid:
                    continue  # filter penny options with unreliable spreads

                # Filter by moneyness if spot is known
                if spot_price and spot_price > 0:
                    moneyness = abs(strike / spot_price - 1.0)
                    if moneyness > max_moneyness:
                        continue

                half_spread_pct = (ask - bid) / (2.0 * mid)

                all_rows.append({
                    "sample_date": str(today),
                    "right": right,
                    "strike": strike,
                    "dte": actual_dte,
                    "iv": iv,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "half_spread_pct": half_spread_pct,
                })
                rows_this_dte += 1

        print(f"  [{symbol}] DTE {actual_dte:3d} ({expiry_str}): {rows_this_dte} valid bid/ask rows")

    if not all_rows:
        print(f"  [{symbol}] No spread samples collected.")
        return 0

    df_samples = pd.DataFrame(all_rows)
    stored = store.save_spread_samples(symbol, df_samples)
    print(f"  [{symbol}] Stored {stored} new spread samples ({len(all_rows)} collected)")
    return stored


# ---------------------------------------------------------------------------
# IBKR snapshot-based calibration (requires options data subscription)
# ---------------------------------------------------------------------------

async def _calibrate_symbol_ibkr(
    ib,
    symbol: str,
    dte_targets: list[int],
    store,
    snapshot_timeout: float,
    pause_secs: float,
    dry_run: bool,
) -> int:
    """Collect spread samples via IBKR snapshot. Returns rows stored."""
    from ib_insync import Stock, Option

    today = date.today()

    # Qualify underlying
    underlying = Stock(symbol, "SMART", "USD")
    try:
        qualified_list = await ib.qualifyContractsAsync(underlying)
        if not qualified_list:
            print(f"  [{symbol}] ERROR: could not qualify underlying")
            return 0
        qualified_underlying = qualified_list[0]
    except Exception as e:
        print(f"  [{symbol}] ERROR qualifying underlying: {e}")
        return 0

    # Delayed-frozen data works even when market is closed
    ib.reqMarketDataType(4)

    # Get spot price
    spot_price: float | None = None
    try:
        [ticker_ib] = await ib.reqTickersAsync(qualified_underlying)
        sp = ticker_ib.marketPrice()
        if sp and math.isfinite(sp) and sp > 0:
            spot_price = sp
        elif ticker_ib.last and ticker_ib.last > 0:
            spot_price = ticker_ib.last
        elif ticker_ib.close and ticker_ib.close > 0:
            spot_price = ticker_ib.close
    except Exception as e:
        print(f"  [{symbol}] reqTickersAsync error: {e}")

    if not spot_price or not math.isfinite(spot_price) or spot_price <= 0:
        try:
            latest_df = store.load(symbol)
            if not latest_df.empty:
                spot_price = float(latest_df["Close"].iloc[-1])
                print(f"  [{symbol}] Using DB fallback spot price: {spot_price:.2f}")
        except Exception:
            pass

    if not spot_price or spot_price <= 0:
        print(f"  [{symbol}] ERROR: could not get spot price")
        return 0

    print(f"  [{symbol}] Spot price: {spot_price:.2f}")

    # IV estimate for strike computation
    assumed_iv = 0.20
    try:
        iv_series = store.load_daily_iv(symbol, days=30)
        if not iv_series.empty:
            iv_latest = float(iv_series.iloc[-1])
            if 0.01 < iv_latest < 5.0:
                assumed_iv = iv_latest
    except Exception:
        pass

    # Get available option chain
    try:
        opt_params = await ib.reqSecDefOptParamsAsync(
            qualified_underlying.symbol, "",
            qualified_underlying.secType, qualified_underlying.conId,
        )
    except Exception as e:
        print(f"  [{symbol}] reqSecDefOptParams error: {e}")
        return 0

    expiry_strs: set[str] = set()
    for op in opt_params or []:
        if op.exchange in ("SMART", "CBOE"):
            expiry_strs.update(op.expirations)

    expiry_dates = sorted(
        datetime.strptime(es, "%Y%m%d").date() for es in expiry_strs
    )
    expiry_dates = [d for d in expiry_dates if d > today]

    if not expiry_dates:
        print(f"  [{symbol}] No future expiries found")
        return 0

    print(f"  [{symbol}] {len(expiry_dates)} future expiries")

    z_offsets = [0.5, 1.0, 1.5]

    if dry_run:
        for dte_target in dte_targets:
            nearest = min(expiry_dates, key=lambda d: abs((d - today).days - dte_target))
            actual_dte = (nearest - today).days
            print(
                f"  [{symbol}] DTE target {dte_target:3d} → expiry {nearest} "
                f"(DTE={actual_dte}) → {len(z_offsets) * 2} contracts to snapshot"
            )
        return 0

    all_rows: list[dict] = []

    for dte_target in dte_targets:
        nearest = min(expiry_dates, key=lambda d: abs((d - today).days - dte_target))
        actual_dte = (nearest - today).days
        expiry_str = nearest.strftime("%Y%m%d")
        sqrt_t = math.sqrt(max(actual_dte, 1) / 252.0)

        for z in z_offsets:
            call_strike = round(spot_price * math.exp(z * assumed_iv * sqrt_t))
            put_strike = round(spot_price * math.exp(-z * assumed_iv * sqrt_t))

            for right, strike in [("C", call_strike), ("P", put_strike)]:
                contract = Option(symbol, expiry_str, strike, right, "SMART")
                try:
                    qualified = await ib.qualifyContractsAsync(contract)
                    if not qualified:
                        continue
                    opt_contract = qualified[0]
                except Exception:
                    continue

                try:
                    ticker_opt = ib.reqMktData(opt_contract, "106", snapshot=False, regulatorySnapshot=False)
                    deadline = time.time() + snapshot_timeout
                    while time.time() < deadline:
                        await asyncio.sleep(0.5)
                        if ticker_opt.bid and ticker_opt.bid > 0 and ticker_opt.ask and ticker_opt.ask > 0:
                            break
                    ib.cancelMktData(opt_contract)
                except Exception as e:
                    try:
                        ib.cancelMktData(opt_contract)
                    except Exception:
                        pass
                    continue

                bid = ticker_opt.bid if ticker_opt.bid and ticker_opt.bid > 0 else None
                ask = ticker_opt.ask if ticker_opt.ask and ticker_opt.ask > 0 else None
                if bid is None or ask is None or ask < bid:
                    print(f"    [{symbol}] DTE{actual_dte} {right}{strike}: no valid bid/ask")
                    continue

                mid = (bid + ask) / 2.0
                half_spread_pct = (ask - bid) / (2.0 * mid)

                iv_val = assumed_iv
                if ticker_opt.modelGreeks and ticker_opt.modelGreeks.impliedVol:
                    mg = ticker_opt.modelGreeks.impliedVol
                    if mg and math.isfinite(mg) and 0.01 < mg < 5.0:
                        iv_val = float(mg)

                print(
                    f"    [{symbol}] DTE{actual_dte} {right}{strike}: "
                    f"bid={bid:.3f} ask={ask:.3f} iv={iv_val:.3f} "
                    f"half_spread={half_spread_pct:.4f}"
                )

                all_rows.append({
                    "sample_date": str(today),
                    "right": right,
                    "strike": float(strike),
                    "dte": actual_dte,
                    "iv": iv_val,
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "half_spread_pct": half_spread_pct,
                })

                if pause_secs > 0:
                    await asyncio.sleep(pause_secs)

        print(
            f"  [{symbol}] DTE {actual_dte}: "
            f"{len([r for r in all_rows if r['dte'] == actual_dte])} samples"
        )

    if not all_rows:
        print(f"  [{symbol}] No spread samples collected.")
        return 0

    df_samples = pd.DataFrame(all_rows)
    stored = store.save_spread_samples(symbol, df_samples)
    print(f"  [{symbol}] Stored {stored} new samples ({len(all_rows)} collected)")
    return stored


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _fit_spread_model(
    symbol: str,
    store,
    max_half_spread: float = 0.50,
) -> dict | None:
    """Load all stored samples and fit the parametric spread model.

    max_half_spread: exclude samples with half_spread_pct above this threshold.
    This removes deep-OTM penny-option rows that would distort the fit for
    the ITM/near-ATM strikes the iron condor actually trades.

    Returns fitted params dict, or None if insufficient data.
    """
    from scipy.optimize import curve_fit

    with __import__("sqlite3").connect(store._db_path) as conn:
        df = pd.read_sql_query(
            f"SELECT iv, dte, half_spread_pct, mid FROM {store._spread_samples_table} "
            f"WHERE symbol = ? AND half_spread_pct <= ? AND mid >= 0.05",
            conn,
            params=(symbol, max_half_spread),
        )

    if len(df) < 10:
        print(f"  [{symbol}] Insufficient samples for fitting ({len(df)} < 10)")
        return None

    iv_arr = df["iv"].values
    dte_arr = df["dte"].values.astype(float)
    y = df["half_spread_pct"].values

    p0 = [0.02, 0.05, 0.20, 0.002, 21.0]
    bounds = (
        [0.0, 0.0, 0.05, 0.0, 5.0],
        [0.20, 0.50, 0.50, 0.05, 60.0],
    )

    try:
        popt, _ = curve_fit(
            half_spread_model,
            (iv_arr, dte_arr),
            y,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
    except Exception as e:
        print(f"  [{symbol}] curve_fit failed: {e}")
        return None

    base, iv_sens, iv_thresh, dte_sens, dte_thresh = popt
    y_pred = half_spread_model((iv_arr, dte_arr), *popt)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    # Practical spread_cap: 99th percentile, rounded to nearest 0.05
    spread_cap = round(float(np.percentile(y, 99)) * 20) / 20

    return {
        "calibrated_on": date.today().isoformat(),
        "spread_base": round(float(base), 4),
        "spread_iv_sensitivity": round(float(iv_sens), 4),
        "spread_iv_threshold": round(float(iv_thresh), 4),
        "spread_dte_sensitivity": round(float(dte_sens), 5),
        "spread_dte_threshold": int(round(float(dte_thresh))),
        "spread_cap": spread_cap,
        "sample_count": len(df),
        "rmse": round(rmse, 5),
    }


# ---------------------------------------------------------------------------
# YAML auto-update
# ---------------------------------------------------------------------------

def _update_yaml_config(config_path: Path, params: dict) -> None:
    """Overwrite only the 4 spread keys in the YAML backtest section.

    Uses ruamel.yaml to preserve all comments and formatting.
    """
    try:
        from ruamel.yaml import YAML
    except ImportError:
        print("  WARNING: ruamel.yaml not installed. Install with: pip install ruamel.yaml")
        return

    yaml = YAML()
    yaml.preserve_quotes = True

    with open(config_path) as f:
        data = yaml.load(f)

    backtest = data.get("backtest", {})
    backtest["spread_base"] = params["spread_base"]
    backtest["spread_iv_sensitivity"] = params["spread_iv_sensitivity"]
    backtest["spread_dte_sensitivity"] = params["spread_dte_sensitivity"]
    backtest["spread_cap"] = params["spread_cap"]

    with open(config_path, "w") as f:
        yaml.dump(data, f)

    print(f"  → Updated {config_path} with calibrated spread values")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> int:
    from ait.data.historical import HistoricalDataStore

    print(f"Spread calibration — source: {args.source}")
    print(f"  Symbols: {args.symbols}")
    print(f"  DTE targets: {args.dte_targets}\n")

    store = HistoricalDataStore(db_path=Path(args.db_path))

    if args.source == "yfinance":
        for sym in args.symbols:
            print(f"=== {sym} ===")
            stored = _calibrate_symbol_yfinance(
                sym, args.dte_targets, store,
                dry_run=args.dry_run,
                max_moneyness=args.max_moneyness,
                min_mid=args.min_mid,
            )

            if args.dry_run or stored == 0:
                continue

            print(f"\n  [{sym}] Fitting parametric spread model...")
            params = _fit_spread_model(sym, store)
            if params:
                _print_results(sym, params)
                store.save_spread_params(sym, params)
                if args.update_config:
                    _update_yaml_config(Path(args.update_config), params)

    else:  # ibkr
        from ib_insync import IB
        ib = IB()
        print(f"Connecting to IBKR at {args.host}:{args.port} (clientId={args.client_id})...")
        try:
            await ib.connectAsync(args.host, args.port, clientId=args.client_id, timeout=10)
        except Exception as e:
            print(f"ERROR: could not connect to IBKR: {e}")
            return 1

        print("Connected.\n")
        t_start = time.time()

        for sym in args.symbols:
            print(f"=== {sym} ===")
            stored = await _calibrate_symbol_ibkr(
                ib, sym, args.dte_targets, store,
                snapshot_timeout=args.snapshot_timeout,
                pause_secs=args.pause_secs,
                dry_run=args.dry_run,
            )

            if args.dry_run or stored == 0:
                continue

            print(f"\n  [{sym}] Fitting parametric spread model...")
            params = _fit_spread_model(sym, store)
            if params:
                _print_results(sym, params)
                store.save_spread_params(sym, params)
                if args.update_config:
                    _update_yaml_config(Path(args.update_config), params)

        ib.disconnect()
        print(f"\nDone in {time.time() - t_start:.1f}s.")

    return 0


def _print_results(sym: str, params: dict) -> None:
    print(f"\n=== {sym} Spread Calibration Results ===")
    print(f"  Samples: {params['sample_count']}")
    print(
        f"  Fit: base={params['spread_base']:.4f}  "
        f"iv_sens={params['spread_iv_sensitivity']:.4f}  "
        f"iv_thresh={params['spread_iv_threshold']:.3f}  "
        f"dte_sens={params['spread_dte_sensitivity']:.5f}  "
        f"dte_thresh={params['spread_dte_threshold']}"
    )
    print(f"  RMSE: {params['rmse']:.5f}")
    print()
    print("  Calibrated values for YAML config:")
    print(f"    spread_base: {params['spread_base']}")
    print(f"    spread_iv_sensitivity: {params['spread_iv_sensitivity']}")
    print(f"    spread_dte_sensitivity: {params['spread_dte_sensitivity']}")
    print(f"    spread_cap: {params['spread_cap']}")
    print()
    print("  → Saved to option_spread_params (DB)")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())

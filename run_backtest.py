"""Run walk-forward backtest with real historical data.

Fetches 2 years of daily OHLCV from Yahoo Finance for configured symbols,
then runs the walk-forward backtester with all enabled strategies.

Usage:
    python run_backtest.py
    python run_backtest.py --symbols SPY QQQ AAPL --days 500
    python run_backtest.py --strategies long_call bull_call_spread iron_condor
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig

# ---------------------------------------------------------------------------
# Live-parity reference values (source of truth: the LIVE code paths)
# ---------------------------------------------------------------------------
# - AIT_IC_WING_K / wing floor $2:   src/ait/strategies/iron_condor.py _vol_scaled_width
# - AIT_IC_MIN_CREDIT ($0.70):       src/ait/strategies/iron_condor.py generate_signals
# - AIT_IC_MIN_CREDIT_WIDTH (0.20):  credit-to-width gate being added live
# - AIT_CREDIT_LOSS_LIMIT (0=off):   R16 parity - live default DISABLED (R6: touch-close beats flat stops)
# - delta target 0.20:               hardcoded in iron_condor.generate_signals
# - DTE band [14, 45]:               src/ait/config/settings.py OptionsConfig.dte_range
# - TP ladder:                       src/ait/execution/portfolio.py _get_take_profit_targets
# - IV rank floor 15:                AIT_IRON_CONDOR_IV_FLOOR default in iron_condor.py
_LIVE_TP_LADDER = {"dte>20": 0.50, "dte_11_20": 0.40, "dte_6_10": 0.30, "dte<=5": 0.20}


def build_parity_manifest(args: argparse.Namespace) -> dict:
    """Compare the live bot's config/env values against this backtest run.

    Any mismatch is VISIBLE (parity_warnings), never fatal — the point is
    that a backtest silently diverging from the live strategy is worse than
    one that loudly diverges on purpose.
    """
    from ait.backtesting.engine import Backtester

    live = {
        "wing_k":            float(os.environ.get("AIT_IC_WING_K", "1.0")),
        "ic_min_credit":     float(os.environ.get("AIT_IC_MIN_CREDIT", "0.70")),
        "ic_min_credit_width": float(os.environ.get("AIT_IC_MIN_CREDIT_WIDTH", "0.20")),
        "credit_loss_limit": float(os.environ.get("AIT_CREDIT_LOSS_LIMIT", "0")),
        "pre_event_blackout_days": None,  # R16: resolved from loaded settings in engine
        "delta_target":      0.20,
        "dte_band":          [14, 45],
        "tp_ladder":         dict(_LIVE_TP_LADDER),
        "wing_floor":        2.0,
        "macro_gate_entry":  True,   # orchestrator blocks credit entries <=4d pre-event
        "macro_flatten_enabled": os.environ.get("AIT_SKIP_MACRO_EVENTS", "0") == "1",
        "iv_rank_floor":     float(os.environ.get("AIT_IRON_CONDOR_IV_FLOOR", "15")),
    }

    # Backtest TP ladder read from the engine mirror itself, so this manifest
    # catches any future drift in Backtester._credit_take_profit_pct.
    bt_ladder = {
        "dte>20":    Backtester._credit_take_profit_pct(25),
        "dte_11_20": Backtester._credit_take_profit_pct(15),
        "dte_6_10":  Backtester._credit_take_profit_pct(8),
        "dte<=5":    Backtester._credit_take_profit_pct(3),
    }
    backtest = {
        "wing_k":            args.wing_k,
        "ic_min_credit":     args.ic_min_credit,
        "ic_min_credit_width": args.ic_min_credit_width,
        "credit_loss_limit": args.credit_loss_limit,
        "delta_target":      0.20,   # Backtester delta_short default (walk-forward does not override)
        "dte_band":          [21, 21],  # engine uses fixed DTE = max_hold_days
        "tp_ladder":         bt_ladder,
        "wing_floor":        args.wing_floor,
        "macro_gate_entry":  args.macro_gate,
        "macro_flatten_enabled": os.environ.get("AIT_SKIP_MACRO_EVENTS", "0") == "1",
        "iv_rank_floor":     args.iv_floor,
    }

    warnings: list[str] = []
    for key, live_val in live.items():
        bt_val = backtest.get(key)
        if key == "dte_band":
            # Engine trades a single fixed DTE; parity holds if it sits inside
            # the live band rather than being equal to it.
            lo, hi = live_val
            if not (lo <= bt_val[0] <= hi and lo <= bt_val[1] <= hi):
                warnings.append(
                    f"dte_band: backtest fixed DTE {bt_val} outside live band {live_val}"
                )
            continue
        if bt_val != live_val:
            warnings.append(f"{key}: live={live_val} backtest={bt_val}")

    notes = [
        "macro gate uses the hardcoded-2026 economic calendar "
        "(src/ait/data/economic_calendar.py); for pre-2026 backtest windows "
        "days-to-event is always > 4, so the gate is effectively INACTIVE — "
        "acceptable, but macro-event losses are NOT simulated there.",
        "live macro FLATTEN (portfolio.py rule 3d) is env-gated behind "
        "AIT_SKIP_MACRO_EVENTS=1 (currently disabled by default for data "
        "collection); the backtest mirrors that env var exactly.",
        "engine trades a fixed synthetic DTE (max_hold_days=21) inside the "
        "live dte_range [14, 45].",
        "credit exits: flat loss limit + DTE-laddered TP + DTE<=5 close "
        "(mirrors portfolio.py); trailing/breakeven applies to DEBIT only.",
    ]

    return {
        "live": live,
        "backtest": backtest,
        "parity_warnings": warnings,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AIT Walk-Forward Backtester")
    p.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL"],
        help="Symbols to backtest",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "put_credit_spread",
            "bull_call_spread",
            "bear_put_spread",
            "iron_condor",
        ],
        help="Strategies to test",
    )
    p.add_argument("--train-days", type=int, default=365, help="Training window days (calendar)")
    p.add_argument("--test-days", type=int, default=63, help="Test window days")
    p.add_argument("--step-days", type=int, default=63, help="Step between windows")
    p.add_argument("--gap-days", type=int, default=5, help="Purge gap days")
    p.add_argument("--capital", type=float, default=50_000.0, help="Initial capital")
    p.add_argument("--min-confidence", type=float, default=0.65, help="Min ML confidence")
    p.add_argument("--range-confidence", type=float, default=0.55,
                   help="Min P(in_range) for iron condors (range model)")
    p.add_argument("--iv-floor", type=float, default=15.0,
                   help="Min IV rank for iron condors (aligned with live "
                        "AIT_IRON_CONDOR_IV_FLOOR default of 15; was 30)")
    # BooleanOptionalAction gives --trailing-stop / --no-trailing-stop.
    # (Old action="store_true" with default=True made it impossible to disable.)
    # Trailing applies to DEBIT trades only — credit trades always use the
    # live-parity flat loss limit + DTE-laddered take-profit.
    p.add_argument("--trailing-stop", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable trailing stops (debit trades only)")
    # --- R6 live-parity knobs (defaults resolve from the SAME env vars the
    # live bot reads, falling back to the live defaults) ---
    p.add_argument("--credit-loss-limit", type=float,
                   default=float(os.environ.get("AIT_CREDIT_LOSS_LIMIT", "0")),
                   help="Flat loss limit for credit trades, as multiple of credit "
                        "received (live env AIT_CREDIT_LOSS_LIMIT, default 0=off per R6/R16)")
    p.add_argument("--ic-min-credit", type=float,
                   default=float(os.environ.get("AIT_IC_MIN_CREDIT", "0.70")),
                   help="Min mid-price total credit for iron condors "
                        "(live env AIT_IC_MIN_CREDIT, default 0.70)")
    p.add_argument("--ic-min-credit-width", type=float,
                   default=float(os.environ.get("AIT_IC_MIN_CREDIT_WIDTH", "0.20")),
                   help="Min credit/max-width ratio for iron condors "
                        "(live env AIT_IC_MIN_CREDIT_WIDTH, default 0.20)")
    p.add_argument("--wing-k", type=float,
                   default=float(os.environ.get("AIT_IC_WING_K", "1.0")),
                   help="Wing width = wing_k*price*IV*sqrt(DTE/365) "
                        "(live env AIT_IC_WING_K, default 1.0)")
    p.add_argument("--wing-floor", type=float, default=2.0,
                   help="Hard minimum wing width in $ (live enforces $2; "
                        "old backtest default was $5)")
    p.add_argument("--macro-gate", action=argparse.BooleanOptionalAction, default=True,
                   help="Block credit entries <=4 days before FOMC/CPI/NFP/GDP/PCE "
                        "(2026-only hardcoded calendar; inactive for earlier windows)")
    p.add_argument("--compare-exits", action="store_true", help="Compare fixed vs trailing stops")
    p.add_argument("--optimize-per-window", action="store_true", default=False,
                   help="Run Optuna optimization on each training window before testing")
    p.add_argument("--optimize-n-trials", type=int, default=50,
                   help="Number of Optuna trials per walk-forward window (requires --optimize-per-window)")
    return p.parse_args()


async def run_backtest(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("  AIT v2 — WALK-FORWARD BACKTEST")
    print("=" * 60)
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Strategies: {', '.join(args.strategies)}")
    print(f"  Train:      {args.train_days}d  |  Test: {args.test_days}d  |  Step: {args.step_days}d")
    print(f"  Capital:    ${args.capital:,.0f}")
    print(f"  Trailing:   {'ON (debit trades only)' if args.trailing_stop else 'OFF'}")
    print("=" * 60)

    # Set IV floor for iron condor strategy (read by iron_condor.py via env)
    os.environ["AIT_IRON_CONDOR_IV_FLOOR"] = str(args.iv_floor)

    # --- Parameter-parity manifest: live config/env vs this backtest run ---
    manifest = build_parity_manifest(args)
    print("\n  PARAMETER-PARITY MANIFEST (live vs backtest):")
    print("  " + json.dumps(manifest, indent=2).replace("\n", "\n  "))
    if manifest["parity_warnings"]:
        print("\n  PARITY WARNINGS (backtest diverges from live — visible, not fatal):")
        for w in manifest["parity_warnings"]:
            print(f"    ! {w}")
    else:
        print("\n  PARITY WARNINGS: none — backtest parameters match live.")

    cfg = WalkForwardConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        gap_days=args.gap_days,
        initial_capital=args.capital,
        min_confidence=args.min_confidence,
        range_min_confidence=args.range_confidence,
        trailing_stop_enabled=args.trailing_stop,
        optimize_per_window=args.optimize_per_window,
        optimize_n_trials=args.optimize_n_trials,
        # R6 live-parity knobs
        wing_k=args.wing_k,
        wing_floor_dollars=args.wing_floor,
        credit_loss_limit_mult=args.credit_loss_limit,
        ic_min_credit=args.ic_min_credit,
        ic_min_credit_width=args.ic_min_credit_width,
        macro_event_gate=args.macro_gate,
    )

    bt = WalkForwardBacktester(
        symbols=args.symbols,
        strategies=args.strategies,
        config=cfg,
    )

    # Fetch data
    print("\n[1/3] Fetching historical data from Yahoo Finance...")
    t0 = time.time()
    data = await bt._fetch_data()
    fetch_time = time.time() - t0

    if not data:
        print("ERROR: No data fetched. Check internet connection.")
        sys.exit(1)

    # Normalize to tz-naive: Yahoo/IB can return tz-aware (America/New_York)
    # indexes, but walkforward normalizes its VIX/SPY context to tz-naive —
    # mixing the two raises "Cannot compare dtypes datetime64[ns] and
    # datetime64[ns, America/New_York]" inside the window loop.
    for sym, df in data.items():
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

    print(f"  Fetched {len(data)} symbols in {fetch_time:.1f}s:")
    for sym, df in data.items():
        print(f"    {sym}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Run walk-forward backtest
    print("\n[2/3] Running walk-forward backtest...")
    t0 = time.time()
    result = await bt.run(data=data)
    run_time = time.time() - t0
    print(f"  Completed in {run_time:.1f}s")

    # Attach the parity manifest to the result object so any downstream JSON
    # serialization of this run (dashboards, experiment writers) carries it.
    result.parity_manifest = manifest

    # Print results
    print(f"\n{result.summary()}")

    # Buy-and-hold benchmark
    print("\n[3/3] Buy-and-hold benchmark:")
    benchmark = bt.benchmark_buy_hold(data)
    for sym, ret in sorted(benchmark.items(), key=lambda x: x[1], reverse=True):
        label = "PORTFOLIO" if sym == "portfolio" else sym
        print(f"    {label:12s} {ret:+.2%}")

    # Strategy vs buy-and-hold comparison
    port_bh = benchmark.get("portfolio", 0)
    strat_ret = result.total_return
    alpha = strat_ret - port_bh
    print(f"\n  Strategy Return:   {strat_ret:+.2%}")
    print(f"  Buy & Hold Return: {port_bh:+.2%}")
    print(f"  Alpha:             {alpha:+.2%}")

    # Per-window details
    if result.windows:
        print(f"\n  WINDOW DETAILS ({len(result.windows)} windows):")
        print(f"  {'#':>3s}  {'Test Period':25s}  {'Trades':>6s}  {'Return':>8s}  {'Win%':>6s}")
        print(f"  {'---':>3s}  {'-' * 25}  {'------':>6s}  {'--------':>8s}  {'------':>6s}")
        for w in result.windows:
            wr = w.backtest_result
            period = f"{w.test_start} to {w.test_end}"
            print(
                f"  {w.window_id:3d}  {period:25s}  {wr.total_trades:6d}  "
                f"{wr.total_return:+8.2%}  {wr.win_rate:5.1%}"
            )

    # Equity curve sample
    curve = result.equity_curve()
    if not curve.empty:
        print(f"\n  Equity curve: {len(curve)} data points")
        print(f"  Start: ${args.capital:,.0f}  ->  End: ${curve['equity'].iloc[-1]:,.0f}")

    # Compare exit modes if requested
    if args.compare_exits and data:
        print("\n" + "=" * 60)
        print("  EXIT MODE COMPARISON (Fixed vs Trailing)")
        print("=" * 60)
        from ait.backtesting.engine import Backtester

        # Use the first symbol's data for comparison
        first_sym = list(data.keys())[0]
        comparison = Backtester.compare_exit_modes(
            data[first_sym],
            args.strategies,
            initial_capital=args.capital,
        )
        print(f"\n  {first_sym} results:")
        print(f"  {'Metric':20s}  {'Fixed':>10s}  {'Trailing':>10s}  {'Delta':>10s}")
        print(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

        fr, tr = comparison["fixed"], comparison["trailing"]
        delta = comparison["delta"]
        for metric, fmt in [
            ("total_return", ".2%"),
            ("win_rate", ".2%"),
            ("sharpe_ratio", ".2f"),
            ("max_drawdown", ".2%"),
            ("profit_factor", ".2f"),
        ]:
            fv = getattr(fr, metric)
            tv = getattr(tr, metric)
            dv = delta[metric]
            print(f"  {metric:20s}  {fv:>10{fmt}}  {tv:>10{fmt}}  {dv:>+10{fmt}}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_backtest(args))


if __name__ == "__main__":
    main()

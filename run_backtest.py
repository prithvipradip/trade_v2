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
import sys
import time

from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig


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
            "bull_call_spread",
            "bear_put_spread",
            "iron_condor",
        ],
        help="Strategies to test",
    )
    p.add_argument("--train-days", type=int, default=365, help="Training window days (calendar)")
    p.add_argument("--test-days", type=int, default=63, help="Test window days")
    p.add_argument("--step-days", type=int, default=21, help="Step between windows")
    p.add_argument("--gap-days", type=int, default=5, help="Purge gap days")
    p.add_argument("--capital", type=float, default=50_000.0, help="Initial capital")
    p.add_argument("--min-confidence", type=float, default=0.65, help="Min ML confidence")
    p.add_argument("--trailing-stop", action="store_true", default=True, help="Enable trailing stops")
    p.add_argument("--compare-exits", action="store_true", help="Compare fixed vs trailing stops")
    return p.parse_args()


async def run_backtest(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("  AIT v2 — WALK-FORWARD BACKTEST")
    print("=" * 60)
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Strategies: {', '.join(args.strategies)}")
    print(f"  Train:      {args.train_days}d  |  Test: {args.test_days}d  |  Step: {args.step_days}d")
    print(f"  Capital:    ${args.capital:,.0f}")
    print(f"  Trailing:   {'ON' if args.trailing_stop else 'OFF'}")
    print("=" * 60)

    cfg = WalkForwardConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        gap_days=args.gap_days,
        initial_capital=args.capital,
        min_confidence=args.min_confidence,
        trailing_stop_enabled=args.trailing_stop,
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

    print(f"  Fetched {len(data)} symbols in {fetch_time:.1f}s:")
    for sym, df in data.items():
        print(f"    {sym}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    # Run walk-forward backtest
    print("\n[2/3] Running walk-forward backtest...")
    t0 = time.time()
    result = await bt.run(data=data)
    run_time = time.time() - t0
    print(f"  Completed in {run_time:.1f}s")

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

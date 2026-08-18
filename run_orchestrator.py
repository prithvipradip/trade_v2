#!/usr/bin/env python
"""AIT v2 Master Orchestrator

Starts the full trading system:
- Trading bot (auto-restart on crash)
- ML retraining (daily 7:30 AM ET)
- Walk-forward backtest (weekly Sunday 8 PM ET)
- Daily P&L report (4:30 PM ET)
- Health monitoring (every 2 min)
- Log cleanup (monthly)

Usage:
    python run_orchestrator.py                    # Start everything
    python run_orchestrator.py --status           # Show scheduled jobs
    python run_orchestrator.py --backtest         # Run backtest now
    python run_orchestrator.py --retrain          # Retrain models now
    python run_orchestrator.py --refresh-fundamentals  # Refresh equity stats (yfinance)
"""
# R12-C: --fetch-news retired with the sentiment stack (deprecated/src/).
# --refresh-fundamentals KEPT: it only touches ait.data.equity_stats +
# DuckDB (yfinance fundamentals), none of which were retired.

import argparse
import os
import sys
from pathlib import Path

# R16: the protective/economic env contract (OMP crash guards, delayed-data
# mode, wide-wing promotion values, undefined-risk gate, macro protection)
# moved to ait.config.runtime_env so EVERY entry point — this launcher AND a
# bare `python -m ait.main` — resolves identical values. runtime_env is
# import-light by contract, so the KMP/OMP guards land before any
# OpenMP-bundling library loads. .env still loads last, filling unset keys.
sys.path.insert(0, str(Path(__file__).parent / "src"))
from ait.config.runtime_env import apply_runtime_env_defaults  # noqa: E402

apply_runtime_env_defaults()

from ait.orchestration.master import (
    BotManager,
    daily_report,
    main,
    retrain_models,
    run_backtest,
    _log,
)


def _refresh_fundamentals() -> None:
    """Force-refresh equity stats for all configured symbols via yfinance."""
    from ait.config.settings import load_settings
    from ait.monitoring.duckdb_analytics import DuckDBAnalytics
    from ait.data.equity_stats import EquityStatsService

    settings = load_settings()
    symbols = settings.trading.universe
    print(f"Refreshing equity stats for {len(symbols)} symbols: {', '.join(symbols)}")

    analytics = DuckDBAnalytics()
    svc = EquityStatsService(analytics)
    results = svc.refresh_all(symbols)

    ok = sum(results.values())
    print(f"Done: {ok}/{len(symbols)} symbols refreshed successfully.")
    for sym, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {sym:8s}  {status}")


def _fetch_news() -> None:
    """R12-C tombstone: the IB news / FinBERT sentiment pipeline is retired.

    The whole stack (ait.sentiment, ait.data.ib_news, ait.data.fundamentals_db)
    lives in deprecated/src/ — R7/R12 verified it had zero influence on iron
    condor decisions, and FinBERT's torch dependency was implicated in the
    c0000005 crash cluster. Nothing consumes the news/analyst tables anymore.
    """
    print("--fetch-news is retired (R12 Tier-C, 2026-07-13): the sentiment/IB-news")
    print("pipeline moved to deprecated/src/ and no live component reads its output.")
    print("See docs/AUDIT_R12.md Tier C item 3.")
    raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIT v2 Master Orchestrator")
    parser.add_argument("--status", action="store_true", help="Check bot status")
    parser.add_argument("--backtest", action="store_true", help="Run backtest now")
    parser.add_argument("--retrain", action="store_true", help="Retrain ML models now")
    parser.add_argument("--report", action="store_true", help="Generate daily report now")
    parser.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="Force-refresh equity fundamental stats from yfinance",
    )
    parser.add_argument(
        "--fetch-news",
        action="store_true",
        help="RETIRED (R12-C): sentiment/IB-news stack moved to deprecated/src/",
    )
    args = parser.parse_args()

    if args.status:
        bot = BotManager()
        print(f"Bot running: {bot.is_running}")
    elif args.backtest:
        print("Running backtest...")
        run_backtest()
    elif args.retrain:
        print("Retraining models...")
        retrain_models()
    elif args.report:
        print("Generating daily report...")
        daily_report()
    elif args.refresh_fundamentals:
        _refresh_fundamentals()
    elif args.fetch_news:
        _fetch_news()
    else:
        main()

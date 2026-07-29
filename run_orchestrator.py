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

# OpenMP conflict guard. XGBoost and LightGBM each bundle their own OpenMP
# runtime; loading both on Windows causes an access-violation crash inside
# xgboost's native training (observed 2026-06-26: segfault in xgboost
# core.update during vol_magnitude fit, killing the whole process — a crash
# try/except cannot catch). KMP_DUPLICATE_LIB_OK lets the duplicate runtimes
# coexist; single-threaded OpenMP avoids the conflicting parallel path.
# Must be set BEFORE numpy/xgboost/lightgbm import anywhere, and propagates
# to the bot + retrain subprocesses via inherited env.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Market data type: 1=live real-time, 4=delayed-frozen. Live now that the
# account holds Network B/C (US equity consolidated) + OPRA subscriptions
# (added 2026-07-02). Propagates to bot + keeper-relaunched subprocesses.
# Revert to "4" here if the subscription ever lapses.
# 2026-07-28 (user decision: trade IC on delayed until U6 funds live data).
# Type 1 with no subscription = 8,446x Error 354 spam and sporadic greeks;
# explicit delayed-frozen (4) serves clean 15-min data. The economics gates
# (credit/width, delta band, EM) remain the protection; delayed-era closes
# are separable by timestamp once live data starts. Flip back to "1" after
# funding + resubscribe.
os.environ.setdefault("AIT_MARKET_DATA_TYPE", "4")

# Paper phase keeps short strangles enabled for the edge comparison
# (PLAN.md go-live gates); the executor refuses undefined-risk orders unless
# this is "1". REMOVE/SET TO 0 AT GO-LIVE — that's what makes defined-risk-
# only contractual (institutional audit INST-5).
os.environ.setdefault("AIT_ALLOW_UNDEFINED_RISK", "1")

# Macro-event protection ON (user decision 2026-07-08): flattens short-premium
# positions when <=1 day to FOMC/CPI/NFP and blocks new entries around events.
# This code existed since Round 1 but the switch defaulted OFF — the book
# would have held short vol straight through the Jul 28-29 FOMC.
os.environ.setdefault("AIT_SKIP_MACRO_EVENTS", "1")

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load .env early so AIT_LIQ_* and any other env-var overrides reach the bot
# subprocesses spawned by BotManager.
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for raw in _env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if val and val[0] in {'"', "'"}:
            quote = val[0]
            end = val.find(quote, 1)
            if end != -1:
                val = val[1:end]
            else:
                val = val[1:]
        else:
            val = val.split("#", 1)[0].rstrip()
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = val

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

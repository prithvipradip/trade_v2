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
    python run_orchestrator.py              # Start everything
    python run_orchestrator.py --status     # Show scheduled jobs
    python run_orchestrator.py --backtest   # Run backtest now
    python run_orchestrator.py --retrain    # Retrain models now
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ait.orchestration.master import (
    BotManager,
    daily_report,
    main,
    retrain_models,
    run_backtest,
    _log,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIT v2 Master Orchestrator")
    parser.add_argument("--status", action="store_true", help="Check bot status")
    parser.add_argument("--backtest", action="store_true", help="Run backtest now")
    parser.add_argument("--retrain", action="store_true", help="Retrain ML models now")
    parser.add_argument("--report", action="store_true", help="Generate daily report now")
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
    else:
        main()

"""AIT v2 — Main entry point.

Starts the autonomous trading bot with all subsystems.

Usage:
    # With default config
    python -m ait.main

    # With custom config
    python -m ait.main --config /path/to/config.yaml

    # Paper trading mode (overrides config)
    python -m ait.main --paper
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path

from ait.bot.orchestrator import TradingOrchestrator
from ait.broker.ibkr_client import IBKRClient
from ait.config.settings import load_settings
from ait.notifications.telegram import TelegramNotifier
from ait.utils.logging import get_logger, setup_logging

log = get_logger("main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIT v2 - Autonomous Intelligent Trading")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--dashboard-only", action="store_true", help="Run dashboard only")
    return parser.parse_args()


async def run_bot(args: argparse.Namespace) -> None:
    """Initialize and run the trading bot."""
    # Load and validate configuration
    settings = load_settings(args.config)

    if args.paper:
        settings.trading.mode = "paper"

    # Setup logging
    setup_logging(settings.logging)

    log.info(
        "ait_starting",
        version="2.0.0",
        mode=settings.trading.mode,
        universe=settings.trading.universe,
        strategies=settings.options.strategies,
    )

    # Safety check for live trading
    if settings.trading.mode == "live":
        log.critical("LIVE TRADING MODE — real money at risk!")
        print("\n" + "=" * 60)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real money will be used for trades.")
        print("  Press Ctrl+C within 10 seconds to abort.")
        print("=" * 60 + "\n")
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            print("Aborted.")
            return

    # Connect to IBKR
    ibkr = IBKRClient(settings.ibkr)
    connected = await ibkr.connect()

    if not connected:
        log.critical("failed_to_connect_ibkr")
        print("\nFailed to connect to IBKR TWS/Gateway.")
        print("Make sure TWS or IB Gateway is running on "
              f"{settings.ibkr.ibkr_host}:{settings.ibkr.ibkr_port}")
        print("\nTo start paper trading:")
        print("1. Open TWS or IB Gateway")
        print("2. Login with your paper trading credentials")
        print("3. Enable API connections in TWS: Edit -> Global Configuration -> API -> Settings")
        print(f"4. Set socket port to {settings.ibkr.ibkr_port}")
        return

    # Setup notifications
    telegram = TelegramNotifier(
        bot_token=settings.api_keys.telegram_bot_token,
        chat_id=settings.api_keys.telegram_chat_id,
    )

    # Create orchestrator
    orchestrator = TradingOrchestrator(settings, ibkr)
    orchestrator.set_notification_callback(telegram.send)

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    def handle_signal(sig, frame):
        log.info("shutdown_signal_received", signal=sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Run bot with shutdown handler
    try:
        bot_task = asyncio.create_task(orchestrator.run())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        done, pending = await asyncio.wait(
            [bot_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    finally:
        await orchestrator.stop()
        await ibkr.disconnect()
        log.info("ait_shutdown_complete")


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if args.dashboard_only:
        # Launch Streamlit dashboard
        import subprocess
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        subprocess.run(["streamlit", "run", str(dashboard_path)])
        return

    # Run the bot
    try:
        asyncio.run(run_bot(args))
    except KeyboardInterrupt:
        print("\nShutdown complete.")


if __name__ == "__main__":
    main()

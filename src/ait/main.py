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
import faulthandler
import signal
import sys
from pathlib import Path

# R16: apply the shared runtime env contract BEFORE any trading subsystem
# import. A bare `python -m ait.main` used to run with macro protections
# OFF, k=1.0 wings, the 0.20 floor, and the undefined-risk gate open —
# silently different economics from the supervised launch path.
# runtime_env is import-light; the KMP/OMP crash guards land before numpy.
from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()

# Dump the Python stack on a native crash (segfault / access violation) to
# stderr -> bot_stdout.log. The bot was dying every 30-60 min to a c0000005
# access violation in a C-extension (2026-06-24); this names the exact call
# site on the next crash so we can pin the culprit library.
# R5 audit F4: dumps used to go to stderr -> bot_stdout.log, where the
# rotation/truncation cycle destroyed every one of them — the c0000005 call
# site is still unproven after ~30 WER crash reports. Dedicated always-open
# file; falls back to stderr if the logs dir is unwritable.
try:
    from pathlib import Path as _P
    _P("logs").mkdir(exist_ok=True)
    _fatal_log = open("logs/fatal.log", "a")
    faulthandler.enable(file=_fatal_log)
except Exception:
    faulthandler.enable()

from ait.bot.orchestrator import TradingOrchestrator
from ait.broker.ibkr_client import IBKRClient
from ait.config.settings import load_settings
from ait.notifications.telegram import TelegramNotifier
from ait.utils.logging import get_logger, setup_logging

log = get_logger("main")

# R17: exceeds orchestrator._shutdown()'s own 15s notify-task drain, so a
# graceful stop has room to actually send the "bot shutting down" message.
GRACEFUL_SHUTDOWN_TIMEOUT = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIT v2 - Autonomous Intelligent Trading")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--paper", action="store_true", help="Force paper trading mode")
    parser.add_argument("--dashboard-only", action="store_true", help="Run dashboard only")
    return parser.parse_args()


async def _graceful_shutdown_bot_task(orchestrator, bot_task: asyncio.Task) -> None:
    """R17: stop the orchestrator's run() loop gracefully before cancelling.

    run()'s while-loop only reaches its post-loop await self._shutdown()
    (notify-drain, etc.) after self._running goes False AND the current
    phase-processing chunk returns -- hard-cancelling immediately injects
    CancelledError wherever run() happens to be awaiting, unwinding it
    before _shutdown() is ever called. Ask nicely first, with a bounded
    grace window; cancel only if it doesn't exit in time -- same behavior
    as before for a shutdown mid-long-wait, strictly better for the common
    case (mid trading-loop, cadence tens of seconds).
    """
    await orchestrator.stop()
    try:
        await asyncio.wait_for(bot_task, timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
    except asyncio.TimeoutError:
        log.warning("graceful_shutdown_timed_out_cancelling")
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass


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

        if bot_task in pending:
            await _graceful_shutdown_bot_task(orchestrator, bot_task)
            pending.discard(bot_task)

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

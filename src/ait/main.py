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
from ait.bot.state import StateManager
from ait.broker.ibkr_client import IBKRClient
from ait.config.settings import load_settings
from ait.execution import manual_positions
from ait.notifications.telegram import TelegramNotifier
from ait.utils.cadence import BOT_CLEANUP_SECONDS, BOT_GRACEFUL_STOP_SECONDS
from ait.utils.logging import get_logger, setup_logging

log = get_logger("main")

# R17: exceeds orchestrator._shutdown()'s own 15s notify-task drain, so a
# graceful stop has room to actually send the "bot shutting down" message.
# W103: the value lives in ait.utils.cadence beside the supervisor's wait, which
# must exceed this plus the shutdown step and the cleanup below.
GRACEFUL_SHUTDOWN_TIMEOUT = BOT_GRACEFUL_STOP_SECONDS


def _install_shutdown_signals(loop: asyncio.AbstractEventLoop,
                              shutdown_event: asyncio.Event) -> None:
    """SIGINT/SIGTERM set ``shutdown_event``.

    W103: the handler runs between two bytecodes of the main thread and must
    reach the event loop through ``call_soon_threadsafe``, which also wakes the
    selector. A bare ``event.set()`` from the handler only queues the callback;
    an idle loop (a long sleep, a long poll) does not run it until something
    else wakes it. ``signal.signal`` rather than ``loop.add_signal_handler`` so
    the same code works where the latter does not exist (Windows).
    """
    def handle_signal(sig, frame):
        log.info("shutdown_signal_received", signal=sig)
        loop.call_soon_threadsafe(shutdown_event.set)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


async def _cleanup_after_stop(poller_task: asyncio.Task, orchestrator, ibkr) -> None:
    """Poller, detached reconcile tasks, orchestrator, broker connection."""
    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass
    # W30 (R26 F3, finding #6): the poller is cancelled (no NEW answers
    # can be recorded) but any reconcile_now() it already fired via
    # _fire_and_forget is still running detached — wait for it before
    # IBKR disconnects out from under it.
    await manual_positions.drain_background_tasks()
    await orchestrator.stop()
    await ibkr.disconnect()


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

    # Paper or live is decided once, here and in the supervisor, by ait/config/launch_mode.py: live needs config.yaml AND
    # .env TRADING_MODE to say live, AIT_ALLOW_LIVE=1 and a live-account IB endpoint; paper must not point at a live
    # endpoint. `--paper` forces paper but the endpoint check still applies.
    import os
    from ait.config.launch_mode import decide
    decision = decide("paper" if args.paper else settings.trading.mode, os.environ.get("TRADING_MODE"),
                      os.environ.get("AIT_ALLOW_LIVE") == "1",
                      int(settings.ibkr.ibkr_port), settings.ibkr.ibkr_account)
    if decision.error:
        print(f"\n{decision.error}\n", file=sys.stderr)
        sys.exit(1)
    settings.trading.mode = decision.mode

    # W35: this process runs on THIS configuration until it exits — runtime
    # load_settings() calls get a copy of it instead of re-reading a
    # config.yaml that a later deploy may have extended (see freeze_settings).
    from ait.config.settings import freeze_settings
    freeze_settings(settings)

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
        enabled=settings.notifications.telegram.enabled,  # W39: was never read
    )

    # Create orchestrator
    orchestrator = TradingOrchestrator(settings, ibkr)
    orchestrator.set_notification_callback(telegram.send)
    # W22: full TelegramNotifier (buttons/edit/callbacks), not just .send —
    # the reconciler needs it to prompt on an untracked position.
    orchestrator.set_telegram(telegram)

    # Setup graceful shutdown
    shutdown_event = asyncio.Event()

    _install_shutdown_signals(asyncio.get_running_loop(), shutdown_event)

    # W22: inbound Telegram polling (button taps + reply-to corrections)
    # runs as its own task, decoupled from the trading loop's own run()/
    # shutdown semantics below — a separate StateManager instance, same
    # DB file (WAL mode; StateManager already opens a fresh sqlite3
    # connection per call, so concurrent instances are the existing,
    # already-relied-on pattern, not a new risk).
    # on_answer=orchestrator.reconcile_now: an answered prompt lifts
    # HALT_UNTRACKED immediately instead of waiting for the next natural
    # reconcile pass (startup/pre-market/post-market) or a restart.
    poller_task = asyncio.create_task(
        manual_positions.run_poller(
            StateManager(), telegram, settings.api_keys.telegram_chat_id,
            on_answer=orchestrator.reconcile_now))

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
        # W103: bounded, so the supervisor's wait (cadence.SUPERVISOR_STOP_WAIT_SECONDS)
        # always covers it.
        _timed_out = False
        try:
            await asyncio.wait_for(
                _cleanup_after_stop(poller_task, orchestrator, ibkr), BOT_CLEANUP_SECONDS)
        except asyncio.TimeoutError:
            _timed_out = True
            log.warning("shutdown_cleanup_timed_out", budget_s=BOT_CLEANUP_SECONDS)
        log.info("ait_shutdown_complete", cleanup_timed_out=_timed_out)


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

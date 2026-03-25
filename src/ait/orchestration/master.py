"""Master orchestrator — schedules and monitors all AIT v2 subsystems.

Manages:
- Trading bot lifecycle (start, health check, auto-restart)
- ML model retraining (daily pre-market + weekly deep retrain)
- Walk-forward backtesting (weekly, tracks strategy health)
- Performance reporting (daily P&L summary, weekly strategy report)
- Log aggregation and alerting

Runs as a persistent daemon alongside the trading bot.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# Project root
ROOT = Path(__file__).resolve().parents[3]
LOGS_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

LOGS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Logging — standalone (no structlog dependency so orchestrator stays light)
# ---------------------------------------------------------------------------

def _log(level: str, event: str, **kw):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extras = " | ".join(f"{k}={v}" for k, v in kw.items())
    line = f"[{ts}] {level.upper():5s} orchestrator.{event}"
    if extras:
        line += f" | {extras}"
    print(line, flush=True)
    with open(LOGS_DIR / "orchestrator.log", "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Bot process management
# ---------------------------------------------------------------------------

class BotManager:
    """Manages the trading bot subprocess."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._restarts = 0
        self._max_restarts = 10
        self._last_restart: datetime | None = None

    @property
    def is_running(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def start(self):
        if self.is_running:
            _log("info", "bot_already_running", pid=self._proc.pid)
            return

        # Ensure IB Gateway is running before starting bot
        from ait.orchestration.gateway import ensure_gateway
        gw_port = int(os.environ.get("IBKR_PORT", "4002"))
        if not ensure_gateway(port=gw_port):
            _log("error", "bot_start_aborted", reason="gateway_not_available")
            return

        _log("info", "bot_starting")
        bot_log = open(LOGS_DIR / "bot_stdout.log", "a")
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "ait.main", "--paper"],
            cwd=str(ROOT),
            stdout=bot_log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        self._last_restart = datetime.now()
        _log("info", "bot_started", pid=self._proc.pid)

    def stop(self):
        if not self.is_running:
            return
        _log("info", "bot_stopping", pid=self._proc.pid)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        _log("info", "bot_stopped")

    def health_check(self):
        """Check if bot is alive, restart if crashed."""
        if self.is_running:
            _log("debug", "bot_healthy", pid=self._proc.pid)
            return

        exit_code = self._proc.returncode if self._proc else "never_started"
        _log("warn", "bot_down", exit_code=exit_code, restarts=self._restarts)

        # Reset restart counter if last restart was >1 hour ago
        if self._last_restart and (datetime.now() - self._last_restart) > timedelta(hours=1):
            self._restarts = 0

        if self._restarts >= self._max_restarts:
            _log("error", "bot_max_restarts_reached", max=self._max_restarts)
            return

        self._restarts += 1
        _log("info", "bot_restarting", attempt=self._restarts)
        self.start()


# ---------------------------------------------------------------------------
# Scheduled tasks
# ---------------------------------------------------------------------------

def run_backtest():
    """Weekly walk-forward backtest to track strategy health."""
    _log("info", "backtest_starting")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"backtest_{ts}.json"

    try:
        result = subprocess.run(
            [
                sys.executable, str(ROOT / "run_backtest.py"),
                "--symbols", "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
                "--capital", "50000",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )

        # Save full output
        output_path = LOGS_DIR / f"backtest_{ts}.log"
        with open(output_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        # Parse key metrics from output
        metrics = _parse_backtest_output(result.stdout)
        metrics["timestamp"] = ts
        metrics["exit_code"] = result.returncode

        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)

        _log("info", "backtest_complete",
             total_return=metrics.get("total_return", "?"),
             sharpe=metrics.get("sharpe", "?"),
             win_rate=metrics.get("win_rate", "?"),
             report=str(report_path))

        # Append to tracking history
        _append_health_metric(metrics)

    except subprocess.TimeoutExpired:
        _log("error", "backtest_timeout")
    except Exception as e:
        _log("error", "backtest_failed", error=str(e))


def retrain_models():
    """Deep model retrain — full walk-forward on all symbols."""
    _log("info", "retrain_starting")
    try:
        result = subprocess.run(
            [sys.executable, "-c", """
import sys
sys.path.insert(0, 'src')
from ait.config.settings import Settings
from ait.data.market_data import MarketDataService
from ait.data.historical import HistoricalDataStore
from ait.ml.ensemble import DirectionPredictor
from ait.ml.trainer import ModelTrainer
import asyncio

settings = Settings()
predictor = DirectionPredictor(settings.ml)
market_data = MarketDataService(None, polygon_api_key=settings.api_keys.polygon_api_key)
historical = HistoricalDataStore()
trainer = ModelTrainer(settings.ml, predictor, market_data, historical)

async def train():
    symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN", "META", "GOOGL"]
    for sym in symbols:
        try:
            await trainer.train_symbol(sym)
            print(f"Trained: {sym}")
        except Exception as e:
            print(f"Failed {sym}: {e}")

asyncio.run(train())
"""],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        _log("info", "retrain_complete", output=result.stdout[-200:] if result.stdout else "empty")
    except subprocess.TimeoutExpired:
        _log("error", "retrain_timeout")
    except Exception as e:
        _log("error", "retrain_failed", error=str(e))


def daily_report():
    """Generate daily P&L and performance summary."""
    _log("info", "daily_report_starting")
    try:
        result = subprocess.run(
            [sys.executable, "-c", """
import sys, json
sys.path.insert(0, 'src')
from ait.monitoring.analytics import TradeAnalytics

analytics = TradeAnalytics()
metrics = analytics.get_performance_metrics()
print(json.dumps(metrics, indent=2, default=str))
"""],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )

        ts = datetime.now().strftime("%Y%m%d")
        report_path = REPORTS_DIR / f"daily_{ts}.json"
        with open(report_path, "w") as f:
            f.write(result.stdout)

        _log("info", "daily_report_complete", report=str(report_path))
    except Exception as e:
        _log("error", "daily_report_failed", error=str(e))


def cleanup_old_logs():
    """Remove logs and reports older than 30 days."""
    cutoff = datetime.now() - timedelta(days=30)
    removed = 0
    for d in [LOGS_DIR, REPORTS_DIR]:
        for f in d.iterdir():
            if f.is_file() and f.name != "orchestrator.log":
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    if mtime < cutoff:
                        f.unlink()
                        removed += 1
                except Exception:
                    pass
    _log("info", "cleanup_complete", removed=removed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_backtest_output(output: str) -> dict:
    """Extract key metrics from backtest stdout."""
    metrics = {}
    for line in output.splitlines():
        line = line.strip()
        if "Total Return:" in line:
            metrics["total_return"] = line.split(":")[-1].strip()
        elif "Sharpe Ratio:" in line:
            metrics["sharpe"] = line.split(":")[-1].strip()
        elif "Win Rate:" in line:
            metrics["win_rate"] = line.split(":")[-1].strip()
        elif "Max Drawdown:" in line:
            metrics["max_drawdown"] = line.split(":")[-1].strip()
        elif "Total Trades:" in line:
            metrics["total_trades"] = line.split(":")[-1].strip()
        elif "Profit Factor:" in line:
            metrics["profit_factor"] = line.split(":")[-1].strip()
        elif "Start:" in line and "End:" in line:
            metrics["equity_summary"] = line.strip()
    return metrics


def _append_health_metric(metrics: dict):
    """Append backtest metrics to health tracking file."""
    health_file = DATA_DIR / "strategy_health.jsonl"
    DATA_DIR.mkdir(exist_ok=True)
    with open(health_file, "a") as f:
        f.write(json.dumps(metrics) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _log("info", "orchestrator_starting", pid=os.getpid())

    bot = BotManager()
    scheduler = BlockingScheduler(timezone="US/Eastern")
    shutdown = Event()

    def graceful_shutdown(signum, frame):
        _log("info", "shutdown_signal", signal=signum)
        shutdown.set()
        bot.stop()
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    # --- Start the trading bot ---
    bot.start()

    # --- Schedule tasks ---

    # Health check: every 2 minutes
    scheduler.add_job(bot.health_check, "interval", minutes=2, id="health_check")

    # Daily model retrain: 7:30 AM ET on trading days (Mon-Fri)
    scheduler.add_job(retrain_models,
                      CronTrigger(day_of_week="mon-fri", hour=7, minute=30),
                      id="daily_retrain")

    # Daily performance report: 4:30 PM ET
    scheduler.add_job(daily_report,
                      CronTrigger(day_of_week="mon-fri", hour=16, minute=30),
                      id="daily_report")

    # Weekly deep backtest: Sunday 8 PM ET
    scheduler.add_job(run_backtest,
                      CronTrigger(day_of_week="sun", hour=20, minute=0),
                      id="weekly_backtest")

    # Monthly log cleanup: 1st of month at midnight
    scheduler.add_job(cleanup_old_logs,
                      CronTrigger(day=1, hour=0, minute=0),
                      id="monthly_cleanup")

    _log("info", "scheduler_ready", jobs=len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        _log("info", "job_registered", id=job.id, trigger=str(job.trigger))

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        _log("info", "orchestrator_shutting_down")
        bot.stop()


if __name__ == "__main__":
    main()

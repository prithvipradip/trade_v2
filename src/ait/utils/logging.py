"""Structured logging with structlog.

Provides consistent, parseable log output across all AIT components.
Each log line includes timestamp, level, component, and structured key-value data.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

from ait.config.settings import LoggingConfig


def setup_logging(
    config: LoggingConfig,
    console_level: str | None = None,
) -> None:
    """Configure structured logging for the entire application.

    Args:
        config:        LoggingConfig (level controls default console threshold;
                       file handler always records DEBUG).
        console_level: Optional override for the console handler level.
                       When running long experiments, pass "WARNING" to keep
                       stdout quiet (DEBUG/INFO go only to the rotating file).
                       Defaults to config.level when not set.
    """
    log_path = Path(config.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    _console_level = console_level or config.level

    # Standard library logging for file output
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # Remove existing handlers
    for handler in list(root_logger.handlers):
        try:
            handler.close()
        finally:
            root_logger.removeHandler(handler)

    # Console handler — human-readable, filtered to console_level
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, _console_level))
    console.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(console)

    # File handler — rotating logs, always captures DEBUG
    file_handler = RotatingFileHandler(
        config.file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(file_handler)

    # R6: ibapi's wire-protocol DEBUG (every >>>/<<< frame) was ~66% of
    # ait.log — 20MB per ~15min of scanning, so the 10x20MB rotation held
    # ~3h of history (destroyed crash forensics) and status.py's 2MB tail
    # undercounted "today" events on the dashboard. Wire frames are noise;
    # real errors from these libraries still pass at WARNING/INFO.
    logging.getLogger("ibapi").setLevel(logging.WARNING)
    logging.getLogger("ib_insync").setLevel(logging.INFO)
    # R13 (secrets lens): urllib3's DEBUG logs full request URLs — a real
    # FINNHUB_API_KEY landed in 11 log files as `?token=<key>` querystrings
    # (200 occurrences) because the file handler captures DEBUG. Same class
    # as the ibapi flood: third-party wire noise never belongs at DEBUG here.
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.INFO)
    # R16: yfinance/peewee print RAW, non-JSON lines ("Entering get()",
    # "url=...", "response code=200") straight into ait.log, breaking every
    # JSON parser that reads the stream — including the audit tooling and
    # status.py. yfinance ships its own debug-mode logger; silence the whole
    # family to WARNING so ait.log stays machine-readable.
    for _noisy in ("yfinance", "peewee", "urllib3.connectionpool",
                   "matplotlib", "asyncio"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    try:  # yfinance>=0.2 exposes an explicit debug toggle
        import yfinance as _yf
        if hasattr(_yf, "set_tz_cache_location"):  # cheap version probe
            _yf.utils.get_yf_logger().setLevel(logging.WARNING)
    except Exception:  # noqa: BLE001 — never let logging setup break boot
        pass

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if sys.stdout.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(component: str) -> structlog.stdlib.BoundLogger:
    """Get a logger bound to a specific component name."""
    return structlog.get_logger(component=component)

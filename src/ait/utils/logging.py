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
        config:        LoggingConfig (level controls the file handler floor).
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
    root_logger.handlers.clear()

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

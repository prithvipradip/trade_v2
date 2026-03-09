"""Persistent state management using SQLite.

Tracks trades, positions, daily P&L, and bot state across restarts.
This is the bot's memory — survives crashes and restarts.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path

from ait.utils.logging import get_logger

log = get_logger("bot.state")

DB_PATH = Path("data/ait_state.db")


class TradeStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    CLOSED = "closed"


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class TradeRecord:
    """A complete trade record from entry to exit."""

    trade_id: str
    symbol: str
    strategy: str
    direction: TradeDirection
    status: TradeStatus

    # Entry
    entry_time: str  # ISO format
    entry_price: float
    quantity: int
    contract_type: str  # "stock", "call", "put", "spread", "iron_condor"
    strike: float | None = None
    expiry: str | None = None

    # Exit
    exit_time: str | None = None
    exit_price: float | None = None

    # P&L
    realized_pnl: float = 0.0
    commission: float = 0.0

    # Context
    ml_confidence: float = 0.0
    sentiment_score: float = 0.0
    market_regime: str = ""
    notes: str = ""

    # Multi-leg details (JSON for spreads/condors)
    legs: str = "[]"


@dataclass
class DailyStats:
    """Daily trading statistics."""

    date: str
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    day_trades_count: int = 0  # For PDT tracking
    circuit_breaker_triggered: bool = False


class StateManager:
    """Manages persistent bot state in SQLite."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    contract_type TEXT NOT NULL,
                    strike REAL,
                    expiry TEXT,
                    exit_time TEXT,
                    exit_price REAL,
                    realized_pnl REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    ml_confidence REAL DEFAULT 0,
                    sentiment_score REAL DEFAULT 0,
                    market_regime TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    legs TEXT DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    trades_taken INTEGER DEFAULT 0,
                    trades_won INTEGER DEFAULT 0,
                    trades_lost INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    day_trades_count INTEGER DEFAULT 0,
                    circuit_breaker_triggered BOOLEAN DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS open_positions (
                    position_id TEXT PRIMARY KEY,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    contract_type TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    current_price REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    stop_loss REAL,
                    take_profit REAL,
                    legs TEXT DEFAULT '[]',
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                );

                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_date
                ON trades(entry_time);

                CREATE INDEX IF NOT EXISTS idx_trades_status
                ON trades(status);

                CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol);
            """)

    # --- Trade Management ---

    def record_trade(self, trade: TradeRecord) -> None:
        """Insert or update a trade record."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO trades
                   (trade_id, symbol, strategy, direction, status,
                    entry_time, entry_price, quantity, contract_type,
                    strike, expiry, exit_time, exit_price,
                    realized_pnl, commission, ml_confidence,
                    sentiment_score, market_regime, notes, legs)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.trade_id, trade.symbol, trade.strategy,
                    trade.direction.value, trade.status.value,
                    trade.entry_time, trade.entry_price, trade.quantity,
                    trade.contract_type, trade.strike, trade.expiry,
                    trade.exit_time, trade.exit_price, trade.realized_pnl,
                    trade.commission, trade.ml_confidence,
                    trade.sentiment_score, trade.market_regime,
                    trade.notes, trade.legs,
                ),
            )

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        realized_pnl: float,
        commission: float = 0.0,
    ) -> None:
        """Mark a trade as closed with exit details."""
        now = datetime.now().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """UPDATE trades
                   SET status = ?, exit_time = ?, exit_price = ?,
                       realized_pnl = ?, commission = ?
                   WHERE trade_id = ?""",
                (TradeStatus.CLOSED.value, now, exit_price, realized_pnl, commission, trade_id),
            )

    def get_open_trades(self) -> list[TradeRecord]:
        """Get all trades that are not closed/cancelled."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status IN (?, ?, ?)",
                (TradeStatus.PENDING.value, TradeStatus.FILLED.value, TradeStatus.PARTIAL.value),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_trades_for_date(self, d: date) -> list[TradeRecord]:
        """Get all trades entered on a specific date."""
        date_str = d.isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE entry_time LIKE ?",
                (f"{date_str}%",),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_recent_trades(self, n: int = 20) -> list[TradeRecord]:
        """Get the N most recent trades."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    # --- Daily Stats ---

    def update_daily_stats(self, stats: DailyStats) -> None:
        """Update daily statistics."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO daily_stats
                   (date, trades_taken, trades_won, trades_lost,
                    total_pnl, max_drawdown, day_trades_count,
                    circuit_breaker_triggered)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    stats.date, stats.trades_taken, stats.trades_won,
                    stats.trades_lost, stats.total_pnl, stats.max_drawdown,
                    stats.day_trades_count, stats.circuit_breaker_triggered,
                ),
            )

    def get_daily_stats(self, d: date | None = None) -> DailyStats:
        """Get daily stats for a specific date (default: today)."""
        d = d or date.today()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM daily_stats WHERE date = ?",
                (d.isoformat(),),
            ).fetchone()

        if row:
            return DailyStats(**dict(row))
        return DailyStats(date=d.isoformat())

    # --- Bot State (Key-Value) ---

    def set_state(self, key: str, value: str) -> None:
        """Set a key-value pair in bot state."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO bot_state (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, datetime.now().isoformat()),
            )

    def get_state(self, key: str, default: str = "") -> str:
        """Get a value from bot state."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT value FROM bot_state WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    # --- Helpers ---

    @staticmethod
    def _row_to_trade(row: sqlite3.Row) -> TradeRecord:
        d = dict(row)
        d["direction"] = TradeDirection(d["direction"])
        d["status"] = TradeStatus(d["status"])
        return TradeRecord(**d)

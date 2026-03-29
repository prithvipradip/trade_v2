"""Persistent state management using SQLite + DuckDB analytics.

SQLite: operational state (open positions, bot KV store, pending trades).
DuckDB: analytics store (closed trades, daily stats, trade context).
Dual-write on close_trade() and update_daily_stats().
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
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
    CLOSING = "closing"
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

    # Journaling (populated on close)
    exit_reason_detailed: str = ""
    peak_pnl_pct: float = 0.0
    time_to_peak_hours: float = 0.0
    direction_correct: int = -1  # -1 = unknown, 0 = wrong, 1 = correct


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
        self._duck = self._init_duckdb()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
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
                    legs TEXT DEFAULT '[]',
                    exit_reason_detailed TEXT DEFAULT '',
                    peak_pnl_pct REAL DEFAULT 0,
                    time_to_peak_hours REAL DEFAULT 0,
                    direction_correct INTEGER DEFAULT -1
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
                    high_water_mark REAL DEFAULT 0,
                    partial_exits TEXT DEFAULT '[]',
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                );

                CREATE TABLE IF NOT EXISTS trade_context (
                    trade_id TEXT PRIMARY KEY,
                    entry_direction TEXT NOT NULL,
                    entry_confidence REAL DEFAULT 0,
                    entry_regime TEXT DEFAULT '',
                    entry_vix REAL DEFAULT 0,
                    entry_iv_rank REAL DEFAULT 0,
                    entry_sentiment_score REAL DEFAULT 0,
                    entry_signals TEXT DEFAULT '{}',
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

    @staticmethod
    def _init_duckdb():
        """Initialize DuckDB analytics store (lazy — returns None if unavailable)."""
        try:
            from ait.monitoring.duckdb_analytics import DuckDBAnalytics
            duck = DuckDBAnalytics()
            log.info("duckdb_analytics_enabled")
            return duck
        except Exception as e:
            log.warning("duckdb_unavailable", error=str(e))
            return None

    # --- Trade Management ---

    def record_trade(self, trade: TradeRecord) -> None:
        """Insert or update a trade record.

        Uses INSERT OR IGNORE for new trades to avoid overwriting journaling
        columns (exit_reason_detailed, peak_pnl_pct, time_to_peak_hours,
        direction_correct) that are populated on close.
        """
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO trades
                   (trade_id, symbol, strategy, direction, status,
                    entry_time, entry_price, quantity, contract_type,
                    strike, expiry, exit_time, exit_price,
                    realized_pnl, commission, ml_confidence,
                    sentiment_score, market_regime, notes, legs,
                    exit_reason_detailed, peak_pnl_pct,
                    time_to_peak_hours, direction_correct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade.trade_id, trade.symbol, trade.strategy,
                    trade.direction.value, trade.status.value,
                    trade.entry_time, trade.entry_price, trade.quantity,
                    trade.contract_type, trade.strike, trade.expiry,
                    trade.exit_time, trade.exit_price, trade.realized_pnl,
                    trade.commission, trade.ml_confidence,
                    trade.sentiment_score, trade.market_regime,
                    trade.notes, trade.legs,
                    "", 0.0, 0.0, -1,
                ),
            )

    def update_trade_status(self, trade_id: str, status: TradeStatus) -> None:
        """Update the status of an existing trade without touching journaling columns."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE trades SET status = ? WHERE trade_id = ?",
                (status.value, trade_id),
            )

    def close_trade(
        self,
        trade_id: str,
        exit_price: float,
        realized_pnl: float,
        commission: float = 0.0,
        exit_reason_detailed: str = "",
    ) -> None:
        """Mark a trade as closed with exit details and journaling data."""
        now = datetime.now().isoformat()

        # Compute journaling fields
        peak_pnl_pct = self.get_high_water_mark(trade_id)

        # Calculate time to peak (from entry to when HWM was set)
        time_to_peak_hours = 0.0

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """UPDATE trades
                   SET status = ?, exit_time = ?, exit_price = ?,
                       realized_pnl = ?, commission = ?,
                       exit_reason_detailed = ?, peak_pnl_pct = ?,
                       time_to_peak_hours = ?
                   WHERE trade_id = ?""",
                (TradeStatus.CLOSED.value, now, exit_price, realized_pnl,
                 commission, exit_reason_detailed, peak_pnl_pct,
                 time_to_peak_hours, trade_id),
            )

            # Clean up open_positions row now that trade is closed
            conn.execute(
                "DELETE FROM open_positions WHERE trade_id = ?",
                (trade_id,),
            )

            # Dual-write: sync closed trade to DuckDB analytics
            if self._duck:
                try:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute(
                        "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
                    ).fetchone()
                    if row:
                        self._duck.ingest_trade(dict(row))
                except Exception as e:
                    log.warning("duckdb_trade_sync_failed", trade_id=trade_id, error=str(e))

    def insert_open_position(
        self,
        trade_id: str,
        symbol: str,
        contract_type: str,
        quantity: int,
        entry_price: float,
        legs: str = "[]",
    ) -> None:
        """Insert a row into open_positions so HWM / partial-exit tracking works.

        Uses trade_id as position_id.  Called when an entry order fills.
        """
        now = datetime.now().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR IGNORE INTO open_positions
                   (position_id, trade_id, symbol, contract_type, quantity,
                    entry_price, entry_time, current_price, unrealized_pnl,
                    stop_loss, take_profit, legs, high_water_mark, partial_exits)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trade_id, trade_id, symbol, contract_type, quantity,
                    entry_price, now, entry_price, 0.0,
                    None, None, legs, 0.0, "[]",
                ),
            )
        log.info("open_position_inserted", trade_id=trade_id, symbol=symbol)

    def remove_open_position(self, trade_id: str) -> None:
        """Remove a position from open_positions when it is fully closed."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "DELETE FROM open_positions WHERE trade_id = ?",
                (trade_id,),
            )

    def get_open_trades(self) -> list[TradeRecord]:
        """Get all trades that are not closed/cancelled."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status IN (?, ?, ?, ?)",
                (TradeStatus.PENDING.value, TradeStatus.FILLED.value,
                 TradeStatus.PARTIAL.value, TradeStatus.CLOSING.value),
            ).fetchall()
        return [self._row_to_trade(r) for r in rows]

    def get_trade_by_id(self, trade_id: str) -> TradeRecord | None:
        """Get a single trade record by ID (any status)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        return self._row_to_trade(row) if row else None

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
        """Update daily statistics (SQLite + DuckDB)."""
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

        # Dual-write to DuckDB
        if self._duck:
            try:
                self._duck.ingest_daily_stats({
                    "date": stats.date,
                    "trades_taken": stats.trades_taken,
                    "trades_won": stats.trades_won,
                    "trades_lost": stats.trades_lost,
                    "total_pnl": stats.total_pnl,
                    "max_drawdown": stats.max_drawdown,
                    "day_trades_count": stats.day_trades_count,
                    "circuit_breaker_triggered": stats.circuit_breaker_triggered,
                })
            except Exception as e:
                log.warning("duckdb_daily_sync_failed", error=str(e))

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

    # --- High Water Mark & Partial Exits ---

    def update_high_water_mark(self, trade_id: str, hwm: float) -> None:
        """Update the high water mark for a trade's P&L percentage."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE open_positions SET high_water_mark = MAX(high_water_mark, ?) WHERE trade_id = ?",
                (hwm, trade_id),
            )

    def get_high_water_mark(self, trade_id: str) -> float:
        """Get the high water mark P&L pct for a trade."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT high_water_mark FROM open_positions WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        return row[0] if row else 0.0

    def record_partial_exit(
        self,
        trade_id: str,
        quantity: int,
        price: float,
        pnl: float,
        pnl_level: float | None = None,
    ) -> None:
        """Record a partial exit for a trade.

        Args:
            pnl_level: The milestone P&L level that triggered this partial exit.
                       Stored so the same level is not re-triggered.
        """
        raw = self.get_partial_exits(trade_id)
        entry: dict = {
            "quantity": quantity,
            "price": price,
            "pnl": pnl,
            "time": datetime.now().isoformat(),
        }
        if pnl_level is not None:
            entry["pnl_level"] = pnl_level
        raw.append(entry)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE open_positions SET partial_exits = ? WHERE trade_id = ?",
                (json.dumps(raw), trade_id),
            )

    def get_partial_exits(self, trade_id: str) -> list[dict]:
        """Get partial exit history for a trade."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT partial_exits FROM open_positions WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return []
        return []

    def update_trade_quantity(self, trade_id: str, new_quantity: int) -> None:
        """Update remaining quantity after a partial exit."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE trades SET quantity = ? WHERE trade_id = ?",
                (new_quantity, trade_id),
            )
            conn.execute(
                "UPDATE open_positions SET quantity = ? WHERE trade_id = ?",
                (new_quantity, trade_id),
            )

    # --- Trade Context ---

    def save_trade_context(
        self,
        trade_id: str,
        direction: str,
        confidence: float,
        regime: str,
        vix: float,
        iv_rank: float,
        sentiment_score: float,
        signals: str = "{}",
    ) -> None:
        """Save the market context at time of trade entry (SQLite + DuckDB)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO trade_context
                   (trade_id, entry_direction, entry_confidence, entry_regime,
                    entry_vix, entry_iv_rank, entry_sentiment_score, entry_signals)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (trade_id, direction, confidence, regime, vix, iv_rank, sentiment_score, signals),
            )

        # Dual-write to DuckDB
        if self._duck:
            try:
                self._duck.ingest_trade_context({
                    "trade_id": trade_id,
                    "entry_direction": direction,
                    "entry_confidence": confidence,
                    "entry_regime": regime,
                    "entry_vix": vix,
                    "entry_iv_rank": iv_rank,
                    "entry_sentiment_score": sentiment_score,
                    "entry_signals": signals,
                })
            except Exception as e:
                log.warning("duckdb_context_sync_failed", trade_id=trade_id, error=str(e))

    def get_trade_context(self, trade_id: str) -> dict | None:
        """Get the entry context for a trade."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM trade_context WHERE trade_id = ?",
                (trade_id,),
            ).fetchone()
        return dict(row) if row else None

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

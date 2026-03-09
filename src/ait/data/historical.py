"""Historical data storage in SQLite.

Stores and retrieves historical price data for ML training.
Avoids re-downloading data that's already been fetched.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ait.utils.logging import get_logger

log = get_logger("data.historical")

DB_PATH = Path("data/historical.db")


class HistoricalDataStore:
    """SQLite-backed store for historical price data."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prices_symbol
                ON daily_prices(symbol, date)
            """)

    def save(self, symbol: str, df: pd.DataFrame) -> int:
        """Save historical data for a symbol. Returns number of rows inserted."""
        if df is None or df.empty:
            return 0

        rows = []
        for idx, row in df.iterrows():
            dt = idx
            if isinstance(dt, pd.Timestamp):
                dt = dt.date()
            elif isinstance(dt, datetime):
                dt = dt.date()
            rows.append((
                symbol,
                str(dt),
                float(row.get("Open", 0)),
                float(row.get("High", 0)),
                float(row.get("Low", 0)),
                float(row.get("Close", 0)),
                int(row.get("Volume", 0)),
            ))

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO daily_prices
                   (symbol, date, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

        log.debug("historical_data_saved", symbol=symbol, rows=len(rows))
        return len(rows)

    def load(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Load historical data for a symbol."""
        query = "SELECT date, open, high, low, close, volume FROM daily_prices WHERE symbol = ?"
        params: list = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(str(start_date))
        if end_date:
            query += " AND date <= ?"
            params.append(str(end_date))

        query += " ORDER BY date"

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    def get_latest_date(self, symbol: str) -> date | None:
        """Get the most recent date we have data for."""
        with sqlite3.connect(self._db_path) as conn:
            result = conn.execute(
                "SELECT MAX(date) FROM daily_prices WHERE symbol = ?",
                (symbol,),
            ).fetchone()

        if result and result[0]:
            return datetime.strptime(result[0], "%Y-%m-%d").date()
        return None

    def symbols_stored(self) -> list[str]:
        """Get list of all symbols with stored data."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol"
            ).fetchall()
        return [r[0] for r in rows]

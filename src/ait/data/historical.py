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
    """SQLite-backed store for historical price data.

    Args:
        db_path:      Path to the SQLite file (created if absent).
        table_prefix: Prefix applied to all table names, e.g. "test_" →
                      test_daily_prices / test_intraday_prices.  Leave empty
                      for production tables.
    """

    def __init__(self, db_path: Path = DB_PATH, table_prefix: str = "") -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._daily_table = f"{table_prefix}daily_prices"
        self._intraday_table = f"{table_prefix}intraday_prices"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._daily_table} (
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
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._daily_table}_symbol
                ON {self._daily_table}(symbol, date)
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._intraday_table} (
                    symbol   TEXT    NOT NULL,
                    datetime TEXT    NOT NULL,
                    interval TEXT    NOT NULL,
                    open     REAL,
                    high     REAL,
                    low      REAL,
                    close    REAL,
                    volume   INTEGER,
                    PRIMARY KEY (symbol, datetime, interval)
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._intraday_table}_symbol_dt
                ON {self._intraday_table}(symbol, interval, datetime)
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
                f"""INSERT OR REPLACE INTO {self._daily_table}
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
        query = f"SELECT date, open, high, low, close, volume FROM {self._daily_table} WHERE symbol = ?"
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
                f"SELECT MAX(date) FROM {self._daily_table} WHERE symbol = ?",
                (symbol,),
            ).fetchone()

        if result and result[0]:
            return datetime.strptime(result[0], "%Y-%m-%d").date()
        return None

    def symbols_stored(self) -> list[str]:
        """Get list of all symbols with stored data."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT symbol FROM {self._daily_table} ORDER BY symbol"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Intraday data (5-min bars)
    # ------------------------------------------------------------------

    def save_intraday(
        self,
        symbol: str,
        df: pd.DataFrame,
        interval: str = "5m",
    ) -> int:
        """Upsert 5-min bars into intraday table. Returns rows inserted/replaced."""
        if df is None or df.empty:
            return 0

        rows = []
        for idx, row in df.iterrows():
            if isinstance(idx, pd.Timestamp):
                dt_str = idx.isoformat()
            else:
                dt_str = str(idx)
            rows.append((
                symbol,
                dt_str,
                interval,
                float(row.get("Open",   0.0)),
                float(row.get("High",   0.0)),
                float(row.get("Low",    0.0)),
                float(row.get("Close",  0.0)),
                int(row.get("Volume", 0)),
            ))

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT OR REPLACE INTO {self._intraday_table}
                   (symbol, datetime, interval, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

        log.debug("intraday_saved", symbol=symbol, interval=interval, rows=len(rows))
        return len(rows)

    def load_intraday(
        self,
        symbol: str,
        days: int = 7,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Load recent intraday bars. Returns DataFrame with UTC DatetimeIndex."""
        from datetime import timedelta, timezone
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days)).isoformat()

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(
                f"""SELECT datetime, open, high, low, close, volume
                   FROM {self._intraday_table}
                   WHERE symbol = ? AND interval = ? AND datetime >= ?
                   ORDER BY datetime""",
                conn,
                params=(symbol, interval, cutoff),
            )

        if df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.index.name = "Datetime"
        return df

    def resample_to_daily(self, symbol: str, days: int = 730) -> pd.DataFrame:
        """Resample stored 5-min bars to daily OHLCV.

        Returns DataFrame with DatetimeIndex and columns
        [Open, High, Low, Close, Volume], or empty DataFrame if no data stored.
        """
        intraday = self.load_intraday(symbol, days=days, interval="5m")
        if intraday.empty:
            return pd.DataFrame()

        intraday = intraday.copy()
        intraday.index = pd.to_datetime(intraday.index)
        daily = (
            intraday.groupby(intraday.index.date)
            .agg(
                Open=("Open", "first"),
                High=("High", "max"),
                Low=("Low", "min"),
                Close=("Close", "last"),
                Volume=("Volume", "sum"),
            )
        )
        daily.index = pd.to_datetime(daily.index)
        daily.index.name = "Date"
        log.debug("resampled_to_daily", symbol=symbol, rows=len(daily))
        return daily

    def get_latest_intraday_timestamp(
        self,
        symbol: str,
        interval: str = "5m",
    ) -> pd.Timestamp | None:
        """Return the timestamp of the most recent stored intraday bar."""
        with sqlite3.connect(self._db_path) as conn:
            result = conn.execute(
                f"""SELECT MAX(datetime) FROM {self._intraday_table}
                   WHERE symbol = ? AND interval = ?""",
                (symbol, interval),
            ).fetchone()

        if result and result[0]:
            return pd.Timestamp(result[0], tz="UTC")
        return None

    def cleanup_old_intraday(self, keep_days: int = 10) -> None:
        """Delete intraday rows older than keep_days calendar days."""
        from datetime import timedelta, timezone
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=keep_days)).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                f"DELETE FROM {self._intraday_table} WHERE datetime < ?",
                (cutoff,),
            )
        log.debug("intraday_cleanup_done", keep_days=keep_days)

    def row_count_intraday(self, symbol: str, interval: str = "5m") -> int:
        """Return the total number of stored intraday bars for a symbol."""
        with sqlite3.connect(self._db_path) as conn:
            result = conn.execute(
                f"SELECT COUNT(*) FROM {self._intraday_table} WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            ).fetchone()
        return result[0] if result else 0

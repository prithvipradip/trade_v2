"""Historical data storage in SQLite.

Stores and retrieves historical price data for ML training.
Avoids re-downloading data that's already been fetched.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd

from ait.utils.logging import get_logger

def _safe_vol(v) -> int:
    """int() that treats NaN/None/negative sentinel volume as 0."""
    try:
        f = float(v)
        if f != f or f < 0:  # NaN or IB's -1 sentinel
            return 0
        return int(f)
    except (TypeError, ValueError):
        return 0


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
        self._daily_iv_table = f"{table_prefix}daily_iv"
        self._intraday_iv_table = f"{table_prefix}intraday_iv"
        self._spread_samples_table = f"{table_prefix}option_spread_samples"
        self._spread_params_table = f"{table_prefix}option_spread_params"
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
                    implied_vol REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            # Migrate existing tables that pre-date the implied_vol column
            try:
                conn.execute(
                    f"ALTER TABLE {self._daily_table} ADD COLUMN implied_vol REAL"
                )
            except sqlite3.OperationalError:
                pass  # column already exists
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
                    source   TEXT DEFAULT '',
                    PRIMARY KEY (symbol, datetime, interval)
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._intraday_table}_symbol_dt
                ON {self._intraday_table}(symbol, interval, datetime)
            """)
            # Dedicated IV-bar tables (separate from price OHLCV — IBKR's
            # OPTION_IMPLIED_VOLATILITY historical bars are themselves OHLC:
            # start/high/low/last IV over the bar period, not a single
            # scalar). Kept apart from daily_prices/intraday_prices so this
            # doesn't require NULL-OHLCV skeleton rows in the price tables.
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._daily_iv_table} (
                    symbol   TEXT NOT NULL,
                    date     TEXT NOT NULL,
                    iv_open  REAL,
                    iv_high  REAL,
                    iv_low   REAL,
                    iv_close REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._intraday_iv_table} (
                    symbol   TEXT NOT NULL,
                    datetime TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    iv_open  REAL,
                    iv_high  REAL,
                    iv_low   REAL,
                    iv_close REAL,
                    PRIMARY KEY (symbol, datetime, interval)
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._intraday_iv_table}_symbol_dt
                ON {self._intraday_iv_table}(symbol, interval, datetime)
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._spread_samples_table} (
                    symbol          TEXT    NOT NULL,
                    sample_date     TEXT    NOT NULL,
                    right           TEXT    NOT NULL,
                    strike          REAL    NOT NULL,
                    dte             INTEGER NOT NULL,
                    iv              REAL    NOT NULL,
                    bid             REAL    NOT NULL,
                    ask             REAL    NOT NULL,
                    mid             REAL    NOT NULL,
                    half_spread_pct REAL    NOT NULL,
                    PRIMARY KEY (symbol, sample_date, right, strike, dte)
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._spread_params_table} (
                    symbol                TEXT    NOT NULL PRIMARY KEY,
                    calibrated_on         TEXT    NOT NULL,
                    spread_base           REAL    NOT NULL,
                    spread_iv_sensitivity REAL    NOT NULL,
                    spread_iv_threshold   REAL    NOT NULL,
                    spread_dte_sensitivity REAL   NOT NULL,
                    spread_dte_threshold  INTEGER NOT NULL,
                    spread_cap            REAL    NOT NULL,
                    sample_count          INTEGER NOT NULL,
                    rmse                  REAL
                )
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
                _safe_vol(row.get("Volume", 0)),  # int(NaN) aborted the whole batch (deep-audit DATA-L10)
            ))

        with sqlite3.connect(self._db_path) as conn:
            # Deep-audit DATA-H2: INSERT OR REPLACE deletes the whole
            # conflicting row — including implied_vol, which this statement
            # doesn't carry — so the daily retrain's save() nulled every IV
            # the backfill had written, silently killing the IV/IV-rank
            # features. Upsert only the OHLCV columns instead.
            conn.executemany(
                f"""INSERT INTO {self._daily_table}
                   (symbol, date, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, date) DO UPDATE SET
                     open=excluded.open, high=excluded.high,
                     low=excluded.low, close=excluded.close,
                     volume=excluded.volume""",
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

    def save_daily_iv(self, symbol: str, iv_series: "pd.Series") -> int:
        """Upsert implied_vol into daily_prices for a symbol.

        iv_series must be a pandas Series with date or datetime index and
        float values (e.g. 0.25 = 25% IV). Rows that already exist get their
        implied_vol updated; rows that don't exist are inserted with NULL OHLCV
        so the IV is available even before OHLCV data arrives.
        Returns the number of rows upserted.
        """
        if iv_series is None or iv_series.empty:
            return 0

        rows = []
        for idx, val in iv_series.items():
            if pd.isna(val):
                continue
            if isinstance(idx, pd.Timestamp):
                date_str = idx.date().isoformat()
            elif isinstance(idx, datetime):
                date_str = idx.date().isoformat()
            else:
                date_str = str(idx)
            rows.append((symbol, date_str, float(val)))

        with sqlite3.connect(self._db_path) as conn:
            # Insert skeleton row if the date is not yet present, then set IV.
            # INSERT OR IGNORE leaves existing OHLCV untouched; the UPDATE
            # then stamps implied_vol on both new and pre-existing rows.
            conn.executemany(
                f"INSERT OR IGNORE INTO {self._daily_table} (symbol, date) VALUES (?, ?)",
                [(symbol, date_str) for symbol, date_str, _ in rows],
            )
            conn.executemany(
                f"UPDATE {self._daily_table} SET implied_vol = ? "
                "WHERE symbol = ? AND date = ?",
                [(iv, sym, date_str) for sym, date_str, iv in rows],
            )

        log.debug("daily_iv_saved", symbol=symbol, rows=len(rows))
        return len(rows)

    def load_daily_iv(
        self,
        symbol: str,
        days: int = 504,
    ) -> "pd.Series":
        """Load stored implied_vol values for a symbol.

        Returns a Series with DatetimeIndex, values are float IV (or NaN where
        implied_vol was not stored).  Only rows with non-NULL implied_vol are
        returned — callers should left-join against the OHLCV index.
        """
        from datetime import timedelta, timezone
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days + 30)).date().isoformat()

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(
                f"""SELECT date, implied_vol FROM {self._daily_table}
                   WHERE symbol = ? AND date >= ? AND implied_vol IS NOT NULL
                   ORDER BY date""",
                conn,
                params=(symbol, cutoff),
            )

        if df.empty:
            return pd.Series(dtype=float, name="implied_vol")

        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["implied_vol"].rename("implied_vol")

    # ------------------------------------------------------------------
    # Intraday data (5-min bars)
    # ------------------------------------------------------------------

    def save_intraday(
        self,
        symbol: str,
        df: pd.DataFrame,
        interval: str = "5m",
        source: str = "TRADES",
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
                _safe_vol(row.get("Volume", 0)),  # int(NaN) aborted the whole batch (deep-audit DATA-L10)
                source,  # A9: bar semantics tag (TRADES vs MIDPOINT vs YAHOO_ADJ)
            ))

        with sqlite3.connect(self._db_path) as conn:
            # Guarded migration for pre-existing DBs (duplicate-add raises)
            try:
                conn.execute(f"ALTER TABLE {self._intraday_table} ADD COLUMN source TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            # R17: INSERT OR REPLACE let MIDPOINT bars (backfill_historical_data.py's
            # default --what-to-show) silently overwrite real TRADES bars, since
            # `source` isn't part of the primary key. Upsert instead, refusing to
            # downgrade an existing TRADES row unless the incoming row is ALSO
            # TRADES (which always wins, e.g. a genuine re-backfill).
            conn.executemany(
                f"""INSERT INTO {self._intraday_table}
                   (symbol, datetime, interval, open, high, low, close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, datetime, interval) DO UPDATE SET
                       open=excluded.open, high=excluded.high, low=excluded.low,
                       close=excluded.close, volume=excluded.volume, source=excluded.source
                   WHERE {self._intraday_table}.source != 'TRADES'
                      OR excluded.source = 'TRADES'""",
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

    def load_intraday_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """Load intraday bars for a specific date range (inclusive).

        Used by the backfill script and walk-forward engine to fetch historical
        5-min data for a known date window without a rolling-days cutoff.
        """
        start_str = str(start_date)
        end_str = str(end_date) + "T23:59:59"

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(
                f"""SELECT datetime, open, high, low, close, volume
                   FROM {self._intraday_table}
                   WHERE symbol = ? AND interval = ?
                     AND datetime >= ? AND datetime <= ?
                   ORDER BY datetime""",
                conn,
                params=(symbol, interval, start_str, end_str),
            )

        if df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.index.name = "Datetime"
        return df

    @staticmethod
    def slice_intraday_up_to(
        intraday_df: pd.DataFrame,
        cutoff_time: time,
    ) -> pd.DataFrame:
        """Return only intraday bars whose time component ≤ cutoff_time.

        Works on a DataFrame with a DatetimeIndex (timezone-aware or naive).
        Used by the intraday backtest engine to construct partial daily bars
        without look-ahead from future bars in the same session.
        """
        if intraday_df.empty:
            return intraday_df
        times = intraday_df.index.time
        mask = times <= cutoff_time
        return intraday_df.loc[mask]

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
        # A9 (deep-audit DATA-M8): drop TODAY's partial session — a half-day
        # bar (partial High/Low/Volume, mid-session Close) fed live features
        # a bar shape the models never saw in training (train/serve skew).
        try:
            from ait.utils.time import now_et
            _today = pd.Timestamp(now_et().date())
            if len(daily) and daily.index[-1] >= _today:
                daily = daily[daily.index < _today]
        except Exception:  # noqa: BLE001
            pass
        log.debug("resampled_to_daily", symbol=symbol, rows=len(daily))
        return daily

    # ------------------------------------------------------------------
    # IV bar history (dedicated tables — IBKR's OPTION_IMPLIED_VOLATILITY
    # historical data is itself OHLC per bar, distinct from the live
    # single-value self-healing store in daily_prices.implied_vol above).
    # ------------------------------------------------------------------

    def save_daily_iv_bars(self, symbol: str, df: "pd.DataFrame") -> int:
        """Upsert daily IV OHLC bars into the daily_iv table.

        df must have a date/DatetimeIndex and columns Open/High/Low/Close
        (IV values as decimals, e.g. 0.25 = 25%). Returns rows upserted.
        """
        if df is None or df.empty:
            return 0

        rows = []
        for idx, row in df.iterrows():
            dt = idx.date() if isinstance(idx, (pd.Timestamp, datetime)) else idx
            rows.append((
                symbol, str(dt),
                float(row["Open"]), float(row["High"]),
                float(row["Low"]), float(row["Close"]),
            ))

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT INTO {self._daily_iv_table}
                   (symbol, date, iv_open, iv_high, iv_low, iv_close)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, date) DO UPDATE SET
                       iv_open=excluded.iv_open, iv_high=excluded.iv_high,
                       iv_low=excluded.iv_low, iv_close=excluded.iv_close""",
                rows,
            )

        log.debug("daily_iv_bars_saved", symbol=symbol, rows=len(rows))
        return len(rows)

    def load_daily_iv_bars(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> "pd.DataFrame":
        """Load daily IV OHLC bars for a symbol."""
        query = (
            f"SELECT date, iv_open, iv_high, iv_low, iv_close "
            f"FROM {self._daily_iv_table} WHERE symbol = ?"
        )
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
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df.columns = ["Open", "High", "Low", "Close"]
        return df

    def save_intraday_iv_bars(
        self,
        symbol: str,
        df: "pd.DataFrame",
        interval: str = "5m",
    ) -> int:
        """Upsert intraday IV OHLC bars into the intraday_iv table.

        df must have a DatetimeIndex and columns Open/High/Low/Close
        (IV values as decimals). Returns rows upserted.
        """
        if df is None or df.empty:
            return 0

        rows = []
        for idx, row in df.iterrows():
            dt_str = idx.isoformat() if isinstance(idx, (pd.Timestamp, datetime)) else str(idx)
            rows.append((
                symbol, dt_str, interval,
                float(row["Open"]), float(row["High"]),
                float(row["Low"]), float(row["Close"]),
            ))

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT INTO {self._intraday_iv_table}
                   (symbol, datetime, interval, iv_open, iv_high, iv_low, iv_close)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, datetime, interval) DO UPDATE SET
                       iv_open=excluded.iv_open, iv_high=excluded.iv_high,
                       iv_low=excluded.iv_low, iv_close=excluded.iv_close""",
                rows,
            )

        log.debug("intraday_iv_bars_saved", symbol=symbol, interval=interval, rows=len(rows))
        return len(rows)

    def load_intraday_iv_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "5m",
    ) -> "pd.DataFrame":
        """Load intraday IV OHLC bars for a symbol over a date range."""
        start_str = str(start_date)
        end_str = str(end_date) + "T23:59:59"

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(
                f"""SELECT datetime, iv_open, iv_high, iv_low, iv_close
                   FROM {self._intraday_iv_table}
                   WHERE symbol = ? AND interval = ?
                     AND datetime >= ? AND datetime <= ?
                   ORDER BY datetime""",
                conn,
                params=(symbol, interval, start_str, end_str),
            )

        if df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df.set_index("datetime", inplace=True)
        df.columns = ["Open", "High", "Low", "Close"]
        df.index.name = "Datetime"
        return df

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

    # ------------------------------------------------------------------
    # Option spread calibration data
    # ------------------------------------------------------------------

    def save_spread_samples(self, symbol: str, df: pd.DataFrame) -> int:
        """Bulk INSERT OR IGNORE spread samples from DataFrame.

        DataFrame must have columns: sample_date, right, strike, dte, iv,
        bid, ask, mid, half_spread_pct. Returns number of rows inserted.
        """
        if df is None or df.empty:
            return 0

        rows = []
        for _, row in df.iterrows():
            rows.append((
                symbol,
                str(row["sample_date"]),
                str(row["right"]),
                float(row["strike"]),
                int(row["dte"]),
                float(row["iv"]),
                float(row["bid"]),
                float(row["ask"]),
                float(row["mid"]),
                float(row["half_spread_pct"]),
            ))

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT OR IGNORE INTO {self._spread_samples_table}
                   (symbol, sample_date, right, strike, dte, iv, bid, ask, mid, half_spread_pct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

        log.debug("spread_samples_saved", symbol=symbol, rows=len(rows))
        return len(rows)

    def save_spread_params(self, symbol: str, params: dict) -> None:
        """INSERT OR REPLACE fitted spread model params for a symbol."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                f"""INSERT OR REPLACE INTO {self._spread_params_table}
                   (symbol, calibrated_on, spread_base, spread_iv_sensitivity,
                    spread_iv_threshold, spread_dte_sensitivity, spread_dte_threshold,
                    spread_cap, sample_count, rmse)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    symbol,
                    params["calibrated_on"],
                    float(params["spread_base"]),
                    float(params["spread_iv_sensitivity"]),
                    float(params["spread_iv_threshold"]),
                    float(params["spread_dte_sensitivity"]),
                    int(params["spread_dte_threshold"]),
                    float(params["spread_cap"]),
                    int(params["sample_count"]),
                    float(params["rmse"]) if params.get("rmse") is not None else None,
                ),
            )
        log.debug("spread_params_saved", symbol=symbol)

    def load_spread_params(self, symbol: str) -> dict | None:
        """Load fitted spread params for a symbol. Returns None if not calibrated."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"""SELECT symbol, calibrated_on, spread_base, spread_iv_sensitivity,
                           spread_iv_threshold, spread_dte_sensitivity, spread_dte_threshold,
                           spread_cap, sample_count, rmse
                   FROM {self._spread_params_table} WHERE symbol = ?""",
                (symbol,),
            ).fetchone()

        if row is None:
            return None

        return {
            "symbol": row[0],
            "calibrated_on": row[1],
            "spread_base": row[2],
            "spread_iv_sensitivity": row[3],
            "spread_iv_threshold": row[4],
            "spread_dte_sensitivity": row[5],
            "spread_dte_threshold": row[6],
            "spread_cap": row[7],
            "sample_count": row[8],
            "rmse": row[9],
        }

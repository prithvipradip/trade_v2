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

# W65: provenance tags for the `source` / `iv_source` columns on the daily
# tables, defined ONCE and imported everywhere (a typo'd literal at a writer is
# an unreadable provenance column). Legacy rows (before W65) are NULL =
# "unrecorded"; no provider is ever guessed for them. The intraday table keeps
# its own pre-existing vocabulary (TRADES / MIDPOINT / YAHOO_ADJ).
SOURCE_IB_TRADES_RTH = "IB_TRADES_RTH"      # IB daily TRADES bars, regular trading hours
SOURCE_YAHOO_UNADJ = "YAHOO_UNADJ"          # Yahoo daily bars, auto_adjust=False
SOURCE_IB_OPTION_IV = "IB_OPTION_IV"        # IB OPTION_IMPLIED_VOLATILITY daily close/OHLC
SOURCE_IB_VIX_INDEX = "IB_INDEX_TRADES"    # W91: IB daily TRADES of the CBOE VIX-family indices, stored /100
SOURCE_UNRECORDED = "UNRECORDED"            # label used when READING NULL source rows
SOURCE_INVESTING_CSV = "INVESTING_CSV"      # operator-compiled Investing.com economic calendar CSV
SOURCE_IB_BID_ASK_BARS = "IB_BID_ASK_BARS"  # W66: IB hourly BID+ASK bars of listed options
SOURCE_YAHOO_DIVIDENDS = "YAHOO_DIVIDENDS"  # W84: yfinance Ticker.dividends (ex-date, amount/share)
SOURCE_FED_FOMC = "FED_FOMC_PAGES"          # W83a: federalreserve.gov FOMC meeting pages
SOURCE_BLS_ARCHIVE = "BLS_RELEASE_ARCHIVE"  # W83a: bls.gov news-release archive links
# Track D: the official_econ_calendar table (the ONE macro-event calendar live and research both read).
# SOURCE_FED_FOMC above is the FOMC tag; BLS rows (CPI / NFP: archive links AND forward schedule pages) and
# BEA rows (GDP advance / PCE: schedule page AND past-release pages) share one tag per agency, the row's
# source_url says which page. FRED rows are only loaded where no agency page covers the date.
SOURCE_BLS_SCHEDULE = "BLS_SCHEDULE"        # bls.gov release schedule + news-release archive (CPI, NFP)
SOURCE_BEA_SCHEDULE = "BEA_SCHEDULE"        # bea.gov release schedule + past release pages (GDP advance, PCE)
SOURCE_FRED_RELEASE_DATES = "FRED_RELEASE_DATES"   # St. Louis Fed /fred/release/dates (needs FRED_API_KEY)
OFFICIAL_CALENDAR_SOURCES: tuple[str, ...] = (
    SOURCE_FED_FOMC, SOURCE_BLS_SCHEDULE, SOURCE_BEA_SCHEDULE, SOURCE_FRED_RELEASE_DATES)
SOURCE_EDGAR_N30B2 = "EDGAR_N30B2"          # W83a: QQQ trust annual Schedule of Investments
SOURCE_EDGAR_NPORT = "EDGAR_NPORTP"         # W83a: QQQ trust NPORT-P quarterly holdings
SOURCE_EDGAR_8K_202 = "EDGAR_8K_ITEM202"    # W83a: company 8-K Item 2.02 (results of operations)

# Historical risk-free-rate follow-up: the SOFR term structure's tenor
# points in days-to-maturity, in the SAME order get_risk_free_rate's SELECT
# reads its columns (sofr, sofr_30d_avg, sofr_90d_avg, sofr_180d_avg) --
# SOFR itself stands in for the "overnight" (1-day) anchor.
SOFR_TENOR_DAYS: dict[str, int] = {
    "sofr": 1,
    "sofr_30d_avg": 30,
    "sofr_90d_avg": 90,
    "sofr_180d_avg": 180,
}


def interpolate_rate(tenor_points: list[tuple[int, float]], dte: int) -> float | None:
    """Piecewise-linear interpolation across a short-end rate term
    structure, flat-extrapolated beyond the shortest/longest anchor -- the
    same scheme this codebase's own half-built VIX9D/VIX/VIX3M/VIX6M
    term-structure idea uses elsewhere (per the W54 scoping-note entry),
    applied here to the SOFR curve instead. `tenor_points` need not be
    pre-sorted. Returns None if `tenor_points` is empty."""
    points = sorted(tenor_points, key=lambda p: p[0])
    if not points:
        return None
    if dte <= points[0][0]:
        return points[0][1]
    if dte >= points[-1][0]:
        return points[-1][1]
    for (d0, r0), (d1, r1) in zip(points, points[1:]):
        if d0 <= dte <= d1:
            if d1 == d0:
                return r0
            frac = (dte - d0) / (d1 - d0)
            return r0 + frac * (r1 - r0)
    return points[-1][1]  # unreachable given the bounds checks above


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
        # W66: per-leg DOLLAR half-spread samples (hourly IB BID/ASK bars) and the fitted
        # friction model v2. Separate tables, not columns on option_spread_samples: that
        # table's PK (symbol, sample_date, right, strike, dte) holds ONE snapshot per day,
        # an hourly sample would collide on it.
        self._spread_bar_table = f"{table_prefix}option_spread_bar_samples"
        self._leg_spread_params_table = f"{table_prefix}option_leg_spread_params"
        # W76: per-DTE-band friction models. A separate table, not a column on the one
        # above: that table's primary key is `symbol` alone, and rebuilding it would mean
        # dropping a table other (possibly running) processes read.
        self._leg_spread_band_table = f"{table_prefix}option_leg_spread_band_params"
        # W82: which expiries the lifecycle collector has already captured, and at what DTE, so a
        # run knows what is still owed without a state file (a row per capture, never updated).
        self._capture_log_table = f"{table_prefix}option_capture_log"
        self._option_daily_bars_table = f"{table_prefix}option_daily_bars"
        # W106: one row per completed option-history fetch, so a contract that came back
        # empty (never traded, or no underlying close) is not re-fetched on every run.
        self._option_fetch_log_table = f"{table_prefix}option_fetch_log"
        self._risk_free_rates_table = f"{table_prefix}risk_free_rates"
        self._econ_events_table = f"{table_prefix}econ_events"
        # W84: past ex-dividend dates per symbol, for the research ex-dividend entry gate
        # (and W83a). The live bot never reads it: live fetches its own from yfinance.
        self._dividends_table = f"{table_prefix}dividends"
        # W83a: the strike-aware range study's event calendars (research only; the live bot never
        # reads them): scheduled FOMC / CPI / NFP dates, QQQ's reported top holdings, and mega-cap
        # earnings releases.
        self._econ_schedule_table = f"{table_prefix}econ_schedule"
        # Track D: the official-source macro calendar (FOMC / CPI / NFP / GDP advance / PCE). Unlike the
        # tables above it IS read by the live bot (EconomicCalendar, through its own read-only connection).
        self._official_calendar_table = f"{table_prefix}official_econ_calendar"
        self._qqq_holdings_table = f"{table_prefix}qqq_holdings"
        self._earnings_schedule_table = f"{table_prefix}earnings_schedule"
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
            # W65 provenance columns (additive, nullable): who supplied the OHLCV
            # and who supplied implied_vol on each row. Legacy rows stay NULL.
            for _col in ("source", "iv_source"):
                try:
                    conn.execute(
                        f"ALTER TABLE {self._daily_table} ADD COLUMN {_col} TEXT"
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
            # W65: economic-event calendar (operator CSV). Dates are ET release dates.
            # '' (not NULL) in the key columns so the primary key de-duplicates.
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._econ_events_table} (
                    event_date        TEXT NOT NULL,
                    event_time_et     TEXT NOT NULL DEFAULT '',
                    country           TEXT NOT NULL,
                    currency          TEXT,
                    importance        TEXT,
                    event             TEXT NOT NULL,
                    reference_period  TEXT NOT NULL DEFAULT '',
                    type              TEXT,
                    category          TEXT,
                    actual            REAL,
                    forecast          REAL,
                    previous          REAL,
                    unit              TEXT,
                    engine_event_type TEXT,
                    source            TEXT,
                    source_file       TEXT,
                    source_sha256     TEXT,
                    PRIMARY KEY (event_date, event_time_et, country, event, reference_period)
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._econ_events_table}_type
                ON {self._econ_events_table}(engine_event_type, event_date)
            """)
            try:  # W65 provenance column on the daily IV-bar table
                conn.execute(
                    f"ALTER TABLE {self._daily_iv_table} ADD COLUMN source TEXT"
                )
            except sqlite3.OperationalError:
                pass  # column already exists
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
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._spread_bar_table} (
                    symbol            TEXT NOT NULL,
                    bar_time          TEXT NOT NULL,   -- ET, ISO, start of the hourly bar
                    right             TEXT NOT NULL,
                    strike            REAL NOT NULL,
                    expiry            TEXT NOT NULL,
                    dte               INTEGER NOT NULL,
                    bid               REAL NOT NULL,
                    ask               REAL NOT NULL,
                    mid               REAL NOT NULL,
                    half_spread_usd   REAL NOT NULL,
                    underlying_price  REAL,
                    delta             REAL,
                    iv                REAL,
                    role              TEXT,            -- 'short' | 'wing' (delta bucket vs wing strike)
                    source            TEXT,
                    PRIMARY KEY (symbol, bar_time, right, strike, expiry)
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._dividends_table} (
                    symbol      TEXT NOT NULL,
                    ex_date     TEXT NOT NULL,   -- ISO date (ET calendar date of the ex-date)
                    amount      REAL,            -- cash per share
                    source      TEXT NOT NULL,   -- SOURCE_* provenance tag
                    fetched_at  TEXT NOT NULL,   -- ISO timestamp of the fetch
                    PRIMARY KEY (symbol, ex_date)
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._econ_schedule_table} (
                    event_date  TEXT NOT NULL,   -- ISO date (FOMC: the decision day)
                    event       TEXT NOT NULL,   -- FOMC | CPI | NFP
                    source      TEXT NOT NULL,
                    source_url  TEXT NOT NULL,
                    PRIMARY KEY (event_date, event)
                )
            """)
            # Track D. Additive and idempotent like every table here (the live bot also runs this init).
            # One row per (date, type): the reconciled official date; event_time_et '' = not stated.
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._official_calendar_table} (
                    event_date     TEXT NOT NULL,   -- ISO ET release date (FOMC: the decision day)
                    event_time_et  TEXT NOT NULL DEFAULT '',   -- 'HH:MM' ET as the source states it
                    event_type     TEXT NOT NULL,   -- FOMC | CPI | NFP | GDP (advance estimate) | PCE
                    source         TEXT NOT NULL,   -- OFFICIAL_CALENDAR_SOURCES tag
                    source_url     TEXT NOT NULL,
                    fetched_at     TEXT NOT NULL,   -- ISO timestamp of the fetch
                    sha256         TEXT NOT NULL,   -- of the fetched source document
                    PRIMARY KEY (event_date, event_type)
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._qqq_holdings_table} (
                    report_date  TEXT NOT NULL,  -- the holdings' as-of date
                    filed_date   TEXT NOT NULL,  -- when the report became public (weights apply after)
                    form         TEXT NOT NULL,  -- N-30B-2 | NPORT-P
                    accession    TEXT NOT NULL,
                    rank         INTEGER NOT NULL,
                    company_key  TEXT,           -- research_calendars.COMPANIES key (NULL = unmapped)
                    name         TEXT NOT NULL,
                    value_usd    REAL NOT NULL,
                    weight       REAL NOT NULL,  -- value / total value of all holdings
                    source       TEXT NOT NULL,
                    PRIMARY KEY (accession, rank)
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._earnings_schedule_table} (
                    accession      TEXT NOT NULL PRIMARY KEY,
                    company_key    TEXT NOT NULL,
                    cik            INTEGER NOT NULL,
                    filing_date    TEXT NOT NULL,
                    accepted_et    TEXT NOT NULL,   -- ISO datetime, America/New_York
                    event_session  TEXT NOT NULL,   -- first session whose price can react
                    source         TEXT NOT NULL
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._capture_log_table} (
                    symbol          TEXT    NOT NULL,
                    expiry          TEXT    NOT NULL,   -- ISO date of the captured expiry
                    stage           TEXT    NOT NULL,   -- 'A' (at expiry) | 'B' (far coverage)
                    captured_on     TEXT    NOT NULL,   -- ET date of the run
                    dte_at_capture  INTEGER NOT NULL,
                    lookback_days   INTEGER NOT NULL,
                    contracts       INTEGER NOT NULL,   -- contracts requested for this expiry
                    PRIMARY KEY (symbol, expiry, stage, captured_on)
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._leg_spread_params_table} (
                    symbol        TEXT NOT NULL PRIMARY KEY,
                    form          TEXT NOT NULL,
                    a             REAL NOT NULL,
                    b             REAL NOT NULL,
                    n             INTEGER NOT NULL,
                    n_sessions    INTEGER,
                    cv_mae        REAL,
                    cv_mae_json   TEXT,
                    source        TEXT,
                    calibrated_on TEXT NOT NULL
                )
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._leg_spread_band_table} (
                    symbol        TEXT NOT NULL,
                    dte_band      TEXT NOT NULL,
                    form          TEXT NOT NULL,
                    a             REAL NOT NULL,
                    b             REAL NOT NULL,
                    n             INTEGER NOT NULL,
                    n_sessions    INTEGER NOT NULL,
                    cv_mae        REAL,
                    cv_mae_json   TEXT,
                    source        TEXT,
                    calibrated_on TEXT NOT NULL,
                    PRIMARY KEY (symbol, dte_band)
                )
            """)
            # W58: historical per-contract EOD option bars (Polygon/Massive
            # aggregates, trade-based -- NOT reusing option_spread_samples,
            # whose bid/ask/mid columns are NOT NULL and assume a live-quote
            # source this data doesn't have). IV/greeks are precomputed and
            # stored at ingestion (not solved lazily); risk_free_rate is
            # stored per-row so a later re-solve pass (once a real historical
            # rate series exists) can find and update stale rows without
            # re-fetching prices.
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._option_daily_bars_table} (
                    underlying       TEXT    NOT NULL,
                    option_ticker    TEXT    NOT NULL,
                    expiry           TEXT    NOT NULL,
                    right            TEXT    NOT NULL,
                    strike           REAL    NOT NULL,
                    date             TEXT    NOT NULL,
                    dte              INTEGER NOT NULL,
                    open             REAL,
                    high             REAL,
                    low              REAL,
                    close            REAL,
                    volume           INTEGER,
                    vwap             REAL,
                    transactions     INTEGER,
                    underlying_close REAL,
                    iv               REAL,
                    iv_outcome       TEXT,
                    delta            REAL,
                    gamma            REAL,
                    theta            REAL,
                    vega             REAL,
                    risk_free_rate   REAL,
                    PRIMARY KEY (option_ticker, date)
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._option_daily_bars_table}_und_exp_dt
                ON {self._option_daily_bars_table}(underlying, expiry, date)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._option_daily_bars_table}_und_dt
                ON {self._option_daily_bars_table}(underlying, date)
            """)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._option_fetch_log_table} (
                    option_ticker TEXT NOT NULL PRIMARY KEY,
                    underlying    TEXT NOT NULL,
                    expiry        TEXT NOT NULL,
                    fetched_on    TEXT NOT NULL,
                    fetched_from  TEXT NOT NULL,
                    fetched_to    TEXT NOT NULL,
                    bars          INTEGER NOT NULL,
                    rows_saved    INTEGER NOT NULL
                )
            """)
            # Historical risk-free-rate series (SOFR term structure) for
            # re-solving option_daily_bars' IV/greeks once real rates exist,
            # replacing the placeholder options.risk_free_rate config value
            # the backfill used at ingestion. Unlike option_daily_bars, this
            # data SHOULD refresh on re-fetch (FRED occasionally revises
            # recent prints) -- see save_risk_free_rates' upsert.
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._risk_free_rates_table} (
                    date          TEXT NOT NULL PRIMARY KEY,
                    sofr          REAL,
                    sofr_30d_avg  REAL,
                    sofr_90d_avg  REAL,
                    sofr_180d_avg REAL
                )
            """)

    def save(self, symbol: str, df: pd.DataFrame, source: str | None = None,
             overwrite: bool = True) -> int:
        """Save historical data for a symbol. Returns number of rows passed.

        W91: `overwrite=False` makes the save APPEND-ONLY: a (symbol, date) that already has prices is
        left exactly as it is; only new dates (and price-less IV-only rows) are filled. The daily
        retrain uses it so a stored completed session is never rewritten (a later vendor correction or
        a different provider can no longer change history under a registered study). Backfill scripts
        keep the default (explicit overwrite).

        W65: `source` tags the OHLCV provider (SOURCE_* constants). None leaves an
        existing tag untouched (COALESCE below), so an untagged older caller can
        never erase provenance.
        """
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
                source,
            ))

        with sqlite3.connect(self._db_path) as conn:
            # Deep-audit DATA-H2: INSERT OR REPLACE deletes the whole
            # conflicting row — including implied_vol, which this statement
            # doesn't carry — so the daily retrain's save() nulled every IV
            # the backfill had written, silently killing the IV/IV-rank
            # features. Upsert only the OHLCV columns instead.
            # W91 fix (found 2026-10-24): the IV writers (save_daily_iv) insert a price-less "skeleton" row for a
            # date before its bar exists (the bot writes today's IV during the session), so a plain DO NOTHING
            # would leave that date without prices FOREVER. Append-only therefore still fills a row whose close
            # is NULL, and never touches a row that has prices.
            on_conflict = (
                """DO UPDATE SET
                     open=excluded.open, high=excluded.high,
                     low=excluded.low, close=excluded.close,
                     volume=excluded.volume,
                     source=COALESCE(excluded.source, source)"""
                if overwrite else
                f"""DO UPDATE SET
                     open=excluded.open, high=excluded.high,
                     low=excluded.low, close=excluded.close,
                     volume=excluded.volume,
                     source=COALESCE(excluded.source, {self._daily_table}.source)
                   WHERE {self._daily_table}.close IS NULL""")
            conn.executemany(
                f"""INSERT INTO {self._daily_table}
                   (symbol, date, open, high, low, close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, date) {on_conflict}""",
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

    def save_econ_events(self, df: pd.DataFrame, source: str, source_file: str,
                         source_sha256: str) -> int:
        """W65: upsert normalized economic-event rows (see scripts/load_econ_calendar_csv.py).

        Required columns: event_date, event_time_et, country, event, reference_period;
        optional: currency, importance, type, category, actual, forecast, previous, unit,
        engine_event_type. Every row is stamped with source/file/hash."""
        if df is None or df.empty:
            return 0

        def _f(v):
            return None if v is None or v != v else float(v)

        def _t(v):
            return None if v is None or v != v else str(v)

        rows = [(
            str(r["event_date"]), str(r.get("event_time_et") or ""), str(r["country"]),
            _t(r.get("currency")), _t(r.get("importance")), str(r["event"]),
            str(r.get("reference_period") or ""), _t(r.get("type")), _t(r.get("category")),
            _f(r.get("actual")), _f(r.get("forecast")), _f(r.get("previous")),
            _t(r.get("unit")), _t(r.get("engine_event_type")),
            source, source_file, source_sha256,
        ) for _, r in df.iterrows()]
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT OR REPLACE INTO {self._econ_events_table}
                    (event_date, event_time_et, country, currency, importance, event,
                     reference_period, type, category, actual, forecast, previous, unit,
                     engine_event_type, source, source_file, source_sha256)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        log.debug("econ_events_saved", rows=len(rows), source=source)
        return len(rows)

    def load_econ_events(self, engine_event_types: "list[str] | None" = None,
                         start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """W65: stored economic events (all rows, or only the mapped engine types)."""
        q = f"SELECT * FROM {self._econ_events_table} WHERE 1=1"
        params: list = []
        if engine_event_types is not None:
            q += " AND engine_event_type IN (" + ",".join("?" * len(engine_event_types)) + ")"
            params += list(engine_event_types)
        if start:
            q += " AND event_date >= ?"
            params.append(str(start))
        if end:
            q += " AND event_date <= ?"
            params.append(str(end))
        q += " ORDER BY event_date, event_time_et"
        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(q, conn, params=params)
        return df

    def econ_events_span(self) -> "tuple[str, str, int] | None":
        """(first_date, last_date, rows) of the stored events, or None when empty."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"SELECT MIN(event_date), MAX(event_date), COUNT(*) FROM {self._econ_events_table}"
            ).fetchone()
        return None if not row or not row[2] else (row[0], row[1], int(row[2]))

    def save_dividends(self, symbol: str, rows: "list[tuple[date, float | None]]", source: str,
                       fetched_at: str) -> int:
        """W84: upsert (ex_date, amount) rows for a symbol (see scripts/backfill_dividends.py)."""
        if not rows:
            return 0
        payload = [(symbol, str(d), None if a is None or a != a else float(a), source, fetched_at)
                   for d, a in rows]
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT OR REPLACE INTO {self._dividends_table}
                    (symbol, ex_date, amount, source, fetched_at) VALUES (?,?,?,?,?)""",
                payload,
            )
        log.debug("dividends_saved", symbol=symbol, rows=len(payload), source=source)
        return len(payload)

    def load_dividends(self, symbol: str) -> "list[tuple[date, float | None]]":
        """W84: a symbol's stored (ex_date, amount) rows, ascending by ex-date."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT ex_date, amount FROM {self._dividends_table} WHERE symbol=? "
                f"ORDER BY ex_date", (symbol,)).fetchall()
        return [(date.fromisoformat(d), a) for d, a in rows]

    def load_earnings_sessions(self, company_key: str) -> "list[date]":
        """W116: the first session that can react to each stored earnings release of ``company_key``, ascending."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT event_session FROM {self._earnings_schedule_table} WHERE company_key=? "
                f"ORDER BY event_session", (company_key,)).fetchall()
        return [date.fromisoformat(d) for (d,) in rows]

    def dividends_span(self, symbol: str) -> "tuple[str, str, int] | None":
        """(first_ex_date, last_ex_date, rows) for a symbol, or None when it has none."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"SELECT MIN(ex_date), MAX(ex_date), COUNT(*) FROM {self._dividends_table} "
                f"WHERE symbol=?", (symbol,)).fetchone()
        return None if not row or not row[2] else (row[0], row[1], int(row[2]))

    def save_econ_schedule(self, rows: "list[tuple[date, str, str]]", source: str) -> int:
        """W83a: upsert (event_date, event, source_url) rows."""
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO {self._econ_schedule_table} "
                f"(event_date, event, source, source_url) VALUES (?,?,?,?)",
                [(str(d), e, source, u) for d, e, u in rows])
        return len(rows)

    def load_econ_schedule(self, event: str | None = None) -> "list[tuple[date, str]]":
        """W83a: stored (event_date, event) rows, ascending; one event type if given."""
        q = f"SELECT event_date, event FROM {self._econ_schedule_table}"
        params: tuple = ()
        if event:
            q += " WHERE event=?"
            params = (event,)
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(q + " ORDER BY event_date, event", params).fetchall()
        return [(date.fromisoformat(d), e) for d, e in rows]

    def save_official_econ_calendar(self, rows: "list[tuple]", replace_from: date | None = None) -> int:
        """Track D: upsert (event_date, event_time_et, event_type, source, source_url, fetched_at, sha256)
        rows (see scripts/build_official_calendar.py). With ``replace_from``, first delete every stored row
        on or after that date IN THE SAME TRANSACTION: a release the agency has since moved must not leave
        its old date behind as a second event."""
        payload = [(str(r[0]), str(r[1] or ""), str(r[2]), str(r[3]), str(r[4]), str(r[5]), str(r[6]))
                   for r in rows]
        with sqlite3.connect(self._db_path) as conn:
            if replace_from is not None:
                conn.execute(f"DELETE FROM {self._official_calendar_table} WHERE event_date >= ?",
                             (str(replace_from),))
            conn.executemany(
                f"INSERT OR REPLACE INTO {self._official_calendar_table} (event_date, event_time_et, "
                f"event_type, source, source_url, fetched_at, sha256) VALUES (?,?,?,?,?,?,?)", payload)
        return len(payload)

    def load_official_econ_calendar(self, event_types: "list[str] | None" = None) -> pd.DataFrame:
        """Track D: stored official calendar rows (all, or only the given event types), by date."""
        q = f"SELECT * FROM {self._official_calendar_table}"
        params: list = []
        if event_types is not None:
            q += " WHERE event_type IN (" + ",".join("?" * len(event_types)) + ")"
            params += list(event_types)
        with sqlite3.connect(self._db_path) as conn:
            return pd.read_sql_query(q + " ORDER BY event_date, event_type", conn, params=params)

    def official_econ_calendar_span(self) -> "tuple[str, str, int] | None":
        """Track D: (first_date, last_date, rows) of the stored official calendar, or None when empty."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(f"SELECT MIN(event_date), MAX(event_date), COUNT(*) "
                               f"FROM {self._official_calendar_table}").fetchone()
        return None if not row or not row[2] else (row[0], row[1], int(row[2]))

    def save_qqq_holdings(self, rows: "list[tuple]", source: str) -> int:
        """W83a: upsert (report_date, filed_date, form, accession, rank, company_key, name,
        value_usd, weight) rows."""
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO {self._qqq_holdings_table} (report_date, filed_date, form, "
                f"accession, rank, company_key, name, value_usd, weight, source) "
                f"VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(str(r[0]), str(r[1]), *r[2:], source) for r in rows])
        return len(rows)

    def load_qqq_holdings(self) -> pd.DataFrame:
        """W83a: every stored top-holding row (one row per report and rank)."""
        with sqlite3.connect(self._db_path) as conn:
            return pd.read_sql_query(
                f"SELECT * FROM {self._qqq_holdings_table} ORDER BY filed_date, rank", conn)

    def save_earnings_schedule(self, rows: "list[tuple]", source: str) -> int:
        """W83a: upsert (accession, company_key, cik, filing_date, accepted_et, event_session) rows."""
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO {self._earnings_schedule_table} (accession, company_key, cik, "
                f"filing_date, accepted_et, event_session, source) VALUES (?,?,?,?,?,?,?)",
                [(r[0], r[1], int(r[2]), str(r[3]), str(r[4]), str(r[5]), source) for r in rows])
        return len(rows)

    def load_earnings_schedule(self) -> pd.DataFrame:
        """W83a: every stored earnings release, ascending by event session."""
        with sqlite3.connect(self._db_path) as conn:
            return pd.read_sql_query(
                f"SELECT * FROM {self._earnings_schedule_table} ORDER BY event_session, company_key",
                conn)

    def source_summary(self, symbol: str) -> dict[str, tuple[int, str, str]]:
        """W65: {source: (priced_rows, first_date, last_date)} for a symbol's daily
        OHLCV. Rows written before W65 report under SOURCE_UNRECORDED (NULL)."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"""SELECT COALESCE(source, ?), COUNT(*), MIN(date), MAX(date)
                    FROM {self._daily_table}
                    WHERE symbol = ? AND close IS NOT NULL
                    GROUP BY COALESCE(source, ?)""",
                (SOURCE_UNRECORDED, symbol, SOURCE_UNRECORDED),
            ).fetchall()
        return {src: (int(n), str(a), str(b)) for src, n, a, b in rows}

    def symbols_stored(self) -> list[str]:
        """Get list of all symbols with stored data."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                f"SELECT DISTINCT symbol FROM {self._daily_table} ORDER BY symbol"
            ).fetchall()
        return [r[0] for r in rows]

    def save_daily_iv(self, symbol: str, iv_series: "pd.Series",
                      source: str | None = None) -> int:
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
                f"UPDATE {self._daily_table} SET implied_vol = ?, "
                "iv_source = COALESCE(?, iv_source) "
                "WHERE symbol = ? AND date = ?",
                [(iv, source, sym, date_str) for sym, date_str, iv in rows],
            )

        log.debug("daily_iv_saved", symbol=symbol, rows=len(rows))
        return len(rows)

    def load_daily_iv(
        self,
        symbol: str,
        days: int = 504,
        start_date: "date | None" = None,
        end_date: "date | None" = None,
    ) -> "pd.Series":
        """Load stored implied_vol values for a symbol.

        Returns a Series with DatetimeIndex, values are float IV (or NaN where
        implied_vol was not stored).  Only rows with non-NULL implied_vol are
        returned — callers should left-join against the OHLCV index.

        W116: with ``start_date`` the window is explicit (``start_date`` .. ``end_date``). Without it the cutoff is
        ``days + 30`` back from NOW, which is wrong for any window that does not end today: a study pinned in the past
        silently lost the IV at the start of its window (the live callers keep this form).
        """
        from datetime import timedelta, timezone
        if start_date is not None:
            cutoff = str(start_date)
        else:
            cutoff = (datetime.now(tz=timezone.utc) - timedelta(days=days + 30)).date().isoformat()
        query = (f"SELECT date, implied_vol FROM {self._daily_table} "
                 f"WHERE symbol = ? AND date >= ? AND implied_vol IS NOT NULL")
        params: list = [symbol, cutoff]
        if end_date is not None:
            query += " AND date <= ?"
            params.append(str(end_date))
        query += " ORDER BY date"

        with sqlite3.connect(self._db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

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

    def save_daily_iv_bars(self, symbol: str, df: "pd.DataFrame",
                           source: str | None = None) -> int:
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
                source,
            ))

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT INTO {self._daily_iv_table}
                   (symbol, date, iv_open, iv_high, iv_low, iv_close, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(symbol, date) DO UPDATE SET
                       iv_open=excluded.iv_open, iv_high=excluded.iv_high,
                       iv_low=excluded.iv_low, iv_close=excluded.iv_close,
                       source=COALESCE(excluded.source, source)""",
                rows,
            )

        log.debug("daily_iv_bars_saved", symbol=symbol, rows=len(rows))
        return len(rows)

    def iv_dates_present(self, symbol: str, since: date | str) -> tuple[set[str], set[str]]:
        """W91: (dates with a daily_iv row, dates whose daily_prices.implied_vol is set), both >= `since`.

        The append-only IV refresh writes only what is missing, so a stored IV reading is never
        rewritten by a later refresh."""
        with sqlite3.connect(self._db_path) as conn:
            bars = {r[0] for r in conn.execute(
                f"SELECT date FROM {self._daily_iv_table} WHERE symbol = ? AND date >= ?",
                (symbol, str(since)))}
            vals = {r[0] for r in conn.execute(
                f"SELECT date FROM {self._daily_table} WHERE symbol = ? AND date >= ? "
                f"AND implied_vol IS NOT NULL", (symbol, str(since)))}
        return bars, vals

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

    def intraday_day_count(self, symbol: str, start: date, end: date,
                           interval: str = "5m") -> int:
        """W65: number of distinct session days with stored intraday bars in
        [start, end]. Timestamps are stored UTC and an RTH session (13:30-21:00 UTC)
        never crosses a UTC date, so the UTC date prefix IS the session date."""
        from datetime import timedelta
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"""SELECT COUNT(DISTINCT substr(datetime, 1, 10))
                    FROM {self._intraday_table}
                    WHERE symbol = ? AND interval = ?
                      AND datetime >= ? AND datetime < ?""",
                (symbol, interval, str(start), str(end + timedelta(days=1))),
            ).fetchone()
        return int(row[0] or 0)

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

    def save_spread_bar_samples(self, symbol: str, df: pd.DataFrame,
                                source: str | None = None) -> int:
        """W66: INSERT OR IGNORE hourly per-leg spread samples (dollars).

        Columns: bar_time, right, strike, expiry, dte, bid, ask, underlying_price,
        delta, iv, role. ``half_spread_usd`` and ``mid`` are derived here so every
        stored row is internally consistent. Returns rows offered."""
        if df is None or df.empty:
            return 0
        rows = []
        for r in df.to_dict("records"):
            bid, ask = float(r["bid"]), float(r["ask"])
            rows.append((
                symbol, str(r["bar_time"]), str(r["right"]), float(r["strike"]),
                str(r["expiry"]), int(r["dte"]), bid, ask, (bid + ask) / 2.0,
                (ask - bid) / 2.0,
                None if r.get("underlying_price") is None else float(r["underlying_price"]),
                None if r.get("delta") is None else float(r["delta"]),
                None if r.get("iv") is None else float(r["iv"]),
                r.get("role"), source,
            ))
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT OR IGNORE INTO {self._spread_bar_table}
                   (symbol, bar_time, right, strike, expiry, dte, bid, ask, mid,
                    half_spread_usd, underlying_price, delta, iv, role, source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        return len(rows)

    def load_spread_bar_samples(self, symbol: str) -> pd.DataFrame:
        with sqlite3.connect(self._db_path) as conn:
            return pd.read_sql_query(
                f"SELECT * FROM {self._spread_bar_table} WHERE symbol = ? ORDER BY bar_time",
                conn, params=(symbol,))

    def save_leg_spread_model(self, symbol: str, row: dict) -> None:
        """INSERT OR REPLACE the fitted friction model v2 for a symbol."""
        import json
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                f"""INSERT OR REPLACE INTO {self._leg_spread_params_table}
                   (symbol, form, a, b, n, n_sessions, cv_mae, cv_mae_json, source,
                    calibrated_on) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (symbol, str(row["form"]), float(row["a"]), float(row["b"]), int(row["n"]),
                 row.get("n_sessions"), row.get("cv_mae"),
                 json.dumps(row.get("cv_mae_by_form") or {}), row.get("source"),
                 str(row["calibrated_on"])))

    def save_leg_spread_band_model(self, symbol: str, band: str, row: dict) -> None:
        """W76: INSERT OR REPLACE the fitted friction model of one DTE band."""
        import json
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                f"""INSERT OR REPLACE INTO {self._leg_spread_band_table}
                   (symbol, dte_band, form, a, b, n, n_sessions, cv_mae, cv_mae_json, source,
                    calibrated_on) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (symbol, str(band), str(row["form"]), float(row["a"]), float(row["b"]),
                 int(row["n"]), int(row["n_sessions"]), row.get("cv_mae"),
                 json.dumps(row.get("cv_mae_by_form") or {}), row.get("source"),
                 str(row["calibrated_on"])))

    def record_captures(self, symbol: str, rows: list[dict]) -> int:
        """W82: INSERT OR REPLACE one row per (expiry, stage, run date) the lifecycle collector
        requested. Returns rows written."""
        if not rows:
            return 0
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT OR REPLACE INTO {self._capture_log_table}
                   (symbol, expiry, stage, captured_on, dte_at_capture, lookback_days, contracts)
                   VALUES (?,?,?,?,?,?,?)""",
                [(symbol, str(r["expiry"]), str(r["stage"]), str(r["captured_on"]),
                  int(r["dte_at_capture"]), int(r["lookback_days"]), int(r["contracts"]))
                 for r in rows])
        return len(rows)

    def load_capture_log(self, symbol: str) -> pd.DataFrame:
        """W82: every recorded capture for ``symbol`` (empty frame when none)."""
        with sqlite3.connect(self._db_path) as conn:
            return pd.read_sql_query(
                f"SELECT * FROM {self._capture_log_table} WHERE symbol = ? ORDER BY captured_on",
                conn, params=(symbol,))

    def load_leg_spread_band_models(self, symbol: str) -> dict[str, dict]:
        """W76: {dte_band: row} of every stored band model for ``symbol`` (may be empty)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM {self._leg_spread_band_table} WHERE symbol = ?",
                (symbol,)).fetchall()
        return {r["dte_band"]: dict(r) for r in rows}

    def load_leg_spread_model(self, symbol: str) -> dict | None:
        """Fitted friction model v2 for a symbol, or None if never calibrated."""
        with sqlite3.connect(self._db_path) as conn:
            r = conn.execute(
                f"""SELECT form, a, b, n, n_sessions, cv_mae, source, calibrated_on
                   FROM {self._leg_spread_params_table} WHERE symbol = ?""",
                (symbol,)).fetchone()
        if r is None:
            return None
        return {"form": r[0], "a": r[1], "b": r[2], "n": r[3], "n_sessions": r[4],
                "cv_mae": r[5], "source": r[6], "calibrated_on": r[7]}

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

    # ------------------------------------------------------------------
    # W58: historical option daily bars (Polygon/Massive backfill)
    # ------------------------------------------------------------------

    def save_option_daily_bars(self, symbol: str, df: pd.DataFrame) -> int:
        """Bulk INSERT OR IGNORE historical option daily bars from a DataFrame.

        DataFrame must have columns: option_ticker, expiry, right, strike,
        date, dte, open, high, low, close, volume, vwap, transactions,
        underlying_close, iv, iv_outcome, delta, gamma, theta, vega,
        risk_free_rate. Returns number of rows passed (matches
        save_spread_samples's own convention, not rows actually inserted).
        """
        if df is None or df.empty:
            return 0

        def _f(v):
            return None if v is None or v != v else float(v)  # NaN-safe

        def _i(v):
            return None if v is None or v != v else int(v)

        rows = []
        for _, row in df.iterrows():
            rows.append((
                symbol,
                str(row["option_ticker"]),
                str(row["expiry"]),
                str(row["right"]),
                float(row["strike"]),
                str(row["date"]),
                int(row["dte"]),
                _f(row.get("open")), _f(row.get("high")),
                _f(row.get("low")), _f(row.get("close")),
                _i(row.get("volume")), _f(row.get("vwap")),
                _i(row.get("transactions")), _f(row.get("underlying_close")),
                _f(row.get("iv")), row.get("iv_outcome"),
                _f(row.get("delta")), _f(row.get("gamma")),
                _f(row.get("theta")), _f(row.get("vega")),
                _f(row.get("risk_free_rate")),
            ))

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT OR IGNORE INTO {self._option_daily_bars_table}
                   (underlying, option_ticker, expiry, right, strike, date, dte,
                    open, high, low, close, volume, vwap, transactions,
                    underlying_close, iv, iv_outcome, delta, gamma, theta, vega,
                    risk_free_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

        log.debug("option_daily_bars_saved", symbol=symbol, rows=len(rows))
        return len(rows)

    def option_ticker_has_data(self, option_ticker: str) -> bool:
        """Resume-skip check: has this contract already been fetched?

        Safe as a coarse per-contract check because save_option_daily_bars's
        executemany commits or rolls back as one unit (sqlite3's `with
        connect(...)` semantics) -- a crash mid-write can never leave a
        partial batch for one contract, so "any row exists" means "this
        contract's fetch already completed."
        """
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"SELECT 1 FROM {self._option_daily_bars_table} "
                f"WHERE option_ticker = ? LIMIT 1",
                (option_ticker,),
            ).fetchone()
        return row is not None

    def record_option_fetch(self, symbol: str, option_ticker: str, expiry: str, fetched_on: date,
                            fetched_from: date, fetched_to: date, bars: int, rows_saved: int) -> None:
        """W106: note a completed fetch, a real answer from the provider, and the window it asked for
        (the latest one wins)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {self._option_fetch_log_table} "
                f"(option_ticker, underlying, expiry, fetched_on, fetched_from, fetched_to, bars, rows_saved) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (option_ticker, symbol, expiry, fetched_on.isoformat(), fetched_from.isoformat(),
                 fetched_to.isoformat(), int(bars), int(rows_saved)),
            )

    def option_fetch_is_final(self, option_ticker: str, start_date: date) -> bool:
        """W106: is this contract's history already complete for a run starting at `start_date`?
        True only when an earlier fetch was made AFTER the contract expired, reached its expiry, and
        started no later than `start_date`, and the provider returned NO bars: then an empty answer
        stays empty. A fetch over part of the contract's life (a later --start-date, an --end-date
        before expiry) is not final, and neither is one whose bars were all dropped for want of an
        underlying close (bars > 0, rows_saved = 0): it is asked again once those closes exist."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"SELECT 1 FROM {self._option_fetch_log_table} "
                f"WHERE option_ticker = ? AND fetched_on > expiry AND fetched_to >= expiry "
                f"AND fetched_from <= ? AND bars = 0 LIMIT 1",
                (option_ticker, start_date.isoformat()),
            ).fetchone()
        return row is not None

    def get_latest_option_expiry(self, symbol: str) -> date | None:
        """W85: the latest contract expiry stored for `symbol` (None if empty).

        Lets backfill_option_history.py --since-last narrow its contract listing to
        expiries it has not yet seen, instead of re-listing the whole 2-year window.
        """
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"SELECT MAX(expiry) FROM {self._option_daily_bars_table} "
                f"WHERE underlying = ?",
                (symbol,),
            ).fetchone()
        if row and row[0]:
            return datetime.strptime(row[0], "%Y-%m-%d").date()
        return None

    def update_option_bar_iv(
        self,
        option_ticker: str,
        date_: date | str,
        iv: float | None,
        iv_outcome: str | None,
        delta: float | None,
        gamma: float | None,
        theta: float | None,
        vega: float | None,
        risk_free_rate: float,
    ) -> int:
        """Overwrite a stored bar's IV/greeks/risk_free_rate in place -- the
        re-solve primitive (historical risk-free-rate follow-up). Unlike
        save_option_daily_bars' INSERT OR IGNORE (deliberately never
        overwrites, what makes ingestion resume-safe), this DELIBERATELY
        overwrites -- and deliberately never touches the raw price columns.
        Returns rows affected (0 if no such (option_ticker, date) row
        exists -- a no-op, not an error)."""
        with sqlite3.connect(self._db_path) as conn:
            cur = conn.execute(
                f"""UPDATE {self._option_daily_bars_table}
                   SET iv=?, iv_outcome=?, delta=?, gamma=?, theta=?, vega=?,
                       risk_free_rate=?
                   WHERE option_ticker=? AND date=?""",
                (iv, iv_outcome, delta, gamma, theta, vega, risk_free_rate,
                 option_ticker, str(date_)),
            )
        return cur.rowcount

    # ------------------------------------------------------------------
    # Historical risk-free-rate series (SOFR term structure)
    # ------------------------------------------------------------------

    def save_risk_free_rates(self, df: pd.DataFrame) -> int:
        """Upsert daily SOFR term-structure rates. DataFrame must have
        columns: date, sofr, sofr_30d_avg, sofr_90d_avg, sofr_180d_avg (any
        may be NaN/absent for a given date). Unlike option_daily_bars, this
        table SHOULD refresh on re-fetch -- FRED occasionally revises
        recent prints. Returns number of rows passed (matches this
        module's own save_option_daily_bars/save_spread_samples
        convention, not rows actually changed)."""
        if df is None or df.empty:
            return 0

        def _f(v):
            return None if v is None or v != v else float(v)  # NaN-safe

        rows = []
        for _, row in df.iterrows():
            rows.append((
                str(row["date"]), _f(row.get("sofr")),
                _f(row.get("sofr_30d_avg")), _f(row.get("sofr_90d_avg")),
                _f(row.get("sofr_180d_avg")),
            ))

        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                f"""INSERT INTO {self._risk_free_rates_table}
                   (date, sofr, sofr_30d_avg, sofr_90d_avg, sofr_180d_avg)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(date) DO UPDATE SET
                       sofr=excluded.sofr,
                       sofr_30d_avg=excluded.sofr_30d_avg,
                       sofr_90d_avg=excluded.sofr_90d_avg,
                       sofr_180d_avg=excluded.sofr_180d_avg""",
                rows,
            )

        log.debug("risk_free_rates_saved", rows=len(rows))
        return len(rows)

    def get_risk_free_rate(self, as_of: date, dte: int) -> float | None:
        """Real historical risk-free rate for an option with `dte` days to
        expiry, as of `as_of`. Forward-fills to the most recently
        PUBLISHED date at or before `as_of` (FRED only publishes on
        business days; a requested weekend/holiday date uses the last
        published print -- the rate genuinely didn't change on days
        nothing was published, standard practice), then linearly
        interpolates across whichever SOFR tenor points are available on
        that date via interpolate_rate(). Returns None only if there is no
        rate data at or before `as_of` at all."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"SELECT sofr, sofr_30d_avg, sofr_90d_avg, sofr_180d_avg "
                f"FROM {self._risk_free_rates_table} WHERE date <= ? "
                f"ORDER BY date DESC LIMIT 1",
                (str(as_of),),
            ).fetchone()
        if row is None:
            return None
        tenor_points = [
            (days, rate)
            for (days, rate) in zip(SOFR_TENOR_DAYS.values(), row)
            if rate is not None
        ]
        return interpolate_rate(tenor_points, dte)

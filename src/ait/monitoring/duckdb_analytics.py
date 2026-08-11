"""DuckDB-powered analytics engine for fast trade analysis.

DuckDB provides columnar storage and vectorized execution — ideal for
analytics workloads like aggregations, window functions, and time-series
analysis over trade history.

Architecture:
- SQLite remains the primary state store (open positions, bot state, KV)
- DuckDB is the analytics store (closed trades, daily stats, feature snapshots)
- State manager dual-writes to both on trade close and daily stat updates
- Dashboard and learning modules read from DuckDB for heavy queries
"""

from __future__ import annotations

import math
import os
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

from ait.utils.logging import get_logger

log = get_logger("monitoring.duckdb")

DUCK_DB_PATH = Path("data/ait_analytics.duckdb")


def _real_close_sql(prefix: str = "") -> str:
    """SQL predicate excluding non-real closes from closed-trade aggregates.

    Reconciler bookkeeping rows (never_filled / pending / migrated) must not
    count as closes in PF / win-rate / drawdown — mirrors status.py's filter.
    """
    col = f"COALESCE({prefix}exit_reason_detailed, '')"
    return (f"{col} NOT LIKE '%never_filled%' AND {col} NOT LIKE '%pending%' "
            f"AND {col} NOT LIKE '%migrated%'")


def _capital_base() -> float:
    """Equity base for drawdown %. Defaults to 196000 (paper NLV).

    Go-live MUST set AIT_CAPITAL_BASE to the funded amount, otherwise
    drawdown % is computed off the wrong base.
    """
    try:
        from ait.config.runtime_env import capital_base as _cb
        return _cb()
    except Exception:  # noqa: BLE001 - R16: single authority, safe fallback
        try:
            return float(os.environ.get("AIT_CAPITAL_BASE", "196000"))
        except (TypeError, ValueError):
            return 196000.0


def _annualization_factor(close_times: list) -> float:
    """sqrt(trades-per-year) — replicates ait.backtesting.result.annualization_factor.

    sqrt(252) treated every TRADE as one trading DAY, overstating
    Sharpe/Sortino ~4x at typical trade frequency (see result.py BT-H4 note).
    Span runs first→last close time to match exit-time windowing; capped at
    daily sqrt(252).
    """
    try:
        dates = sorted(
            t.date() if isinstance(t, datetime)
            else datetime.fromisoformat(str(t)[:19]).date()
            for t in close_times
            if t is not None
        )
        if len(dates) < 2:
            return 1.0
        span_days = max((dates[-1] - dates[0]).days, 1)
        trades_per_year = len(dates) / (span_days / 365.25)
        return math.sqrt(min(252.0, max(trades_per_year, 1.0)))
    except Exception:
        return 1.0


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance metrics computed by DuckDB."""

    total_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_hold_hours: float = 0.0
    recovery_factor: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    current_streak: int = 0


class DuckDBAnalytics:
    """Columnar analytics engine backed by DuckDB.

    Provides fast aggregations, window functions, and time-series queries
    over trade history. All methods are read-heavy except ingest_*.
    """

    def __init__(self, db_path: Path = DUCK_DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()

    def _get_conn(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        # A11 (deep-audit OPS-R3): DuckDB is single-writer -- read paths
        # (dashboard/report/learning) opening read-write collided with the
        # bot's close-time ingest. Readers pass read_only=True.
        return duckdb.connect(str(self._db_path), read_only=read_only)

    def _init_schema(self) -> None:
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id VARCHAR PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    strategy VARCHAR NOT NULL,
                    direction VARCHAR NOT NULL,
                    status VARCHAR NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    entry_price DOUBLE NOT NULL,
                    quantity INTEGER NOT NULL,
                    contract_type VARCHAR NOT NULL,
                    strike DOUBLE,
                    expiry VARCHAR,
                    exit_time TIMESTAMP,
                    exit_price DOUBLE,
                    realized_pnl DOUBLE DEFAULT 0,
                    commission DOUBLE DEFAULT 0,
                    ml_confidence DOUBLE DEFAULT 0,
                    sentiment_score DOUBLE DEFAULT 0,
                    market_regime VARCHAR DEFAULT '',
                    notes VARCHAR DEFAULT '',
                    legs VARCHAR DEFAULT '[]',
                    exit_reason_detailed VARCHAR DEFAULT '',
                    peak_pnl_pct DOUBLE DEFAULT 0,
                    time_to_peak_hours DOUBLE DEFAULT 0,
                    direction_correct INTEGER DEFAULT -1
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date DATE PRIMARY KEY,
                    trades_taken INTEGER DEFAULT 0,
                    trades_won INTEGER DEFAULT 0,
                    trades_lost INTEGER DEFAULT 0,
                    total_pnl DOUBLE DEFAULT 0,
                    max_drawdown DOUBLE DEFAULT 0,
                    day_trades_count INTEGER DEFAULT 0,
                    circuit_breaker_triggered BOOLEAN DEFAULT false
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_context (
                    trade_id VARCHAR PRIMARY KEY,
                    entry_direction VARCHAR NOT NULL,
                    entry_confidence DOUBLE DEFAULT 0,
                    entry_regime VARCHAR DEFAULT '',
                    entry_vix DOUBLE DEFAULT 0,
                    entry_iv_rank DOUBLE DEFAULT 0,
                    entry_sentiment_score DOUBLE DEFAULT 0,
                    entry_signals VARCHAR DEFAULT '{}'
                )
            """)

            # Feature snapshots for ML analysis — what features looked like at entry
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    trade_id VARCHAR NOT NULL,
                    feature_name VARCHAR NOT NULL,
                    feature_value DOUBLE NOT NULL,
                    PRIMARY KEY (trade_id, feature_name)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS equity_stats (
                    symbol              VARCHAR PRIMARY KEY,
                    updated_at          TIMESTAMP NOT NULL,
                    company_name        VARCHAR DEFAULT '',
                    sector              VARCHAR DEFAULT '',
                    industry            VARCHAR DEFAULT '',
                    country             VARCHAR DEFAULT '',
                    exchange            VARCHAR DEFAULT '',
                    market_cap          BIGINT  DEFAULT 0,
                    pe_ratio            DOUBLE  DEFAULT 0,
                    forward_pe          DOUBLE  DEFAULT 0,
                    pb_ratio            DOUBLE  DEFAULT 0,
                    ps_ratio            DOUBLE  DEFAULT 0,
                    ev_ebitda           DOUBLE  DEFAULT 0,
                    eps                 DOUBLE  DEFAULT 0,
                    book_value_ps       DOUBLE  DEFAULT 0,
                    revenue_ps          DOUBLE  DEFAULT 0,
                    dividend_yield      DOUBLE  DEFAULT 0,
                    dividend_rate       DOUBLE  DEFAULT 0,
                    payout_ratio        DOUBLE  DEFAULT 0,
                    beta                DOUBLE  DEFAULT 0,
                    week52_high         DOUBLE  DEFAULT 0,
                    week52_low          DOUBLE  DEFAULT 0,
                    avg_volume_30d      BIGINT  DEFAULT 0,
                    float_shares        BIGINT  DEFAULT 0,
                    shares_outstanding  BIGINT  DEFAULT 0,
                    analyst_target_mean DOUBLE  DEFAULT 0,
                    analyst_target_high DOUBLE  DEFAULT 0,
                    analyst_target_low  DOUBLE  DEFAULT 0,
                    analyst_rating      VARCHAR DEFAULT '',
                    analyst_count       INTEGER DEFAULT 0
                )
            """)

        log.info("duckdb_initialized", path=str(self._db_path))

    # ------------------------------------------------------------------
    # Ingest (write path — called by StateManager on trade close)
    # ------------------------------------------------------------------

    _TRADE_INGEST_COLS = (
        "trade_id", "symbol", "strategy", "direction", "status",
        "entry_time", "entry_price", "quantity", "contract_type",
        "strike", "expiry", "exit_time", "exit_price",
        "realized_pnl", "commission", "ml_confidence",
        "sentiment_score", "market_regime", "notes", "legs",
        "exit_reason_detailed", "peak_pnl_pct",
        "time_to_peak_hours", "direction_correct",
    )

    def ingest_trade(self, trade: dict) -> None:
        """Upsert a trade record into DuckDB analytics store.

        R8 CRITICAL fix: callers feed dict(SELECT * FROM trades) — after the
        capital_at_risk migration that dict has MORE keys than the INSERT's
        named params and DuckDB raises on the excess, silently (warning-
        swallowed) stopping every future close from reaching analytics.
        Filter to the ingest columns so sqlite-side migrations can never
        break the mirror again.
        """
        trade = {k: trade.get(k) for k in self._TRADE_INGEST_COLS}
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trades VALUES (
                    $trade_id, $symbol, $strategy, $direction, $status,
                    $entry_time, $entry_price, $quantity, $contract_type,
                    $strike, $expiry, $exit_time, $exit_price,
                    $realized_pnl, $commission, $ml_confidence,
                    $sentiment_score, $market_regime, $notes, $legs,
                    $exit_reason_detailed, $peak_pnl_pct,
                    $time_to_peak_hours, $direction_correct
                )
            """, trade)

    def ingest_daily_stats(self, stats: dict) -> None:
        """Upsert daily statistics."""
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_stats VALUES (
                    $date, $trades_taken, $trades_won, $trades_lost,
                    $total_pnl, $max_drawdown, $day_trades_count,
                    $circuit_breaker_triggered
                )
            """, stats)

    def ingest_trade_context(self, context: dict) -> None:
        """Upsert trade entry context."""
        # R8: same excess-parameter armor as ingest_trade (model_version
        # column was already breaking this — 17/17 sync_context_failed).
        _cols = ("trade_id", "entry_direction", "entry_confidence",
                 "entry_regime", "entry_vix", "entry_iv_rank",
                 "entry_sentiment_score", "entry_signals")
        context = {k: context.get(k) for k in _cols}
        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO trade_context VALUES (
                    $trade_id, $entry_direction, $entry_confidence,
                    $entry_regime, $entry_vix, $entry_iv_rank,
                    $entry_sentiment_score, $entry_signals
                )
            """, context)

    def ingest_feature_snapshot(self, trade_id: str, features: dict[str, float]) -> None:
        """Store the feature values at trade entry for later analysis."""
        if not features:
            return
        with self._get_conn() as conn:
            for name, value in features.items():
                conn.execute(
                    "INSERT OR REPLACE INTO feature_snapshots VALUES (?, ?, ?)",
                    [trade_id, name, value],
                )

    # ------------------------------------------------------------------
    # Performance queries (read path)
    # ------------------------------------------------------------------

    def get_performance(self, lookback_days: int = 30) -> PerformanceSnapshot:
        """Compute comprehensive performance metrics using DuckDB."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            # Exit-time window: a trade opened before the lookback window but
            # CLOSED inside it must count (entry-time filtering hid late
            # losers). Order by close time so drawdown/streaks follow
            # realization order; exclude bookkeeping closes.
            rows = conn.execute(f"""
                SELECT realized_pnl, entry_time, exit_time
                FROM trades
                WHERE status = 'closed'
                  AND COALESCE(exit_time, entry_time) >= ?
                  AND {_real_close_sql()}
                ORDER BY COALESCE(exit_time, entry_time)
            """, [cutoff]).fetchall()

        if not rows:
            return PerformanceSnapshot()

        pnls = [r[0] for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        snap = PerformanceSnapshot(
            total_trades=len(pnls),
            total_pnl=total_pnl,
            win_rate=len(wins) / len(pnls) if pnls else 0,
            avg_trade_pnl=total_pnl / len(pnls) if pnls else 0,
            avg_win=sum(wins) / len(wins) if wins else 0,
            avg_loss=sum(losses) / len(losses) if losses else 0,
            largest_win=max(wins) if wins else 0,
            largest_loss=min(losses) if losses else 0,
        )

        # Profit factor
        gross_wins = sum(wins)
        gross_losses = abs(sum(losses))
        snap.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Sharpe & Sortino — annualized by ACTUAL trade frequency, not
        # sqrt(252) (sqrt(252) treated each trade as one trading day; see
        # ait.backtesting.result.annualization_factor)
        if len(pnls) > 1:
            ann = _annualization_factor([r[2] or r[1] for r in rows])
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl > 0:
                snap.sharpe_ratio = (mean_pnl / std_pnl) * ann
            # Sortino — target-0 downside deviation over ALL returns:
            # sqrt(mean(min(r,0)^2)), matching result.py. The old
            # stdev-of-losses-about-their-own-mean gave a consistent
            # -$500 loser downside-dev≈0.
            downside_dev = math.sqrt(sum(min(p, 0.0) ** 2 for p in pnls) / len(pnls))
            if downside_dev > 0:
                snap.sortino_ratio = (mean_pnl / downside_dev) * ann
            elif mean_pnl > 0:
                # R17: see the matching comment in monitoring/analytics.py --
                # inconsistent with export.py's None convention, not unified
                # (no current JSON consumer of this field).
                snap.sortino_ratio = float("inf")

        # Drawdown — dd% measured against the EQUITY high-water mark where
        # equity = CAPITAL_BASE + cumulative P&L. The old cumulative-P&L peak
        # (starting at 0) returned 0% after pure losses and absurd % on small
        # samples. dd$ is still reported alongside.
        base = _capital_base()
        equity = base
        peak = base
        max_dd = 0.0
        max_dd_pct = 0.0
        for p in pnls:
            equity += p
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
            if peak > 0 and dd / peak > max_dd_pct:
                max_dd_pct = dd / peak
        snap.max_drawdown_dollars = max_dd
        snap.max_drawdown_pct = max_dd_pct

        if max_dd > 0:
            snap.recovery_factor = total_pnl / max_dd

        # Streaks
        max_w = max_l = 0
        streak = 0
        for p in pnls:
            if p > 0:
                streak = streak + 1 if streak > 0 else 1
                max_w = max(max_w, streak)
            else:
                streak = streak - 1 if streak < 0 else -1
                max_l = max(max_l, abs(streak))
        snap.consecutive_wins = max_w
        snap.consecutive_losses = max_l
        snap.current_streak = streak

        # Average hold time
        hold_hours = []
        for r in rows:
            if r[1] and r[2]:
                try:
                    entry = datetime.fromisoformat(str(r[1]))
                    exit_ = datetime.fromisoformat(str(r[2]))
                    hold_hours.append((exit_ - entry).total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass
        snap.avg_hold_hours = sum(hold_hours) / len(hold_hours) if hold_hours else 0.0

        return snap

    def get_daily_pnl(self, lookback_days: int = 30) -> list[dict]:
        """Get daily P&L with cumulative running total — uses DuckDB window function."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            rows = conn.execute("""
                SELECT
                    date,
                    total_pnl AS daily_pnl,
                    SUM(total_pnl) OVER (ORDER BY date) AS cumulative_pnl,
                    trades_taken AS trades,
                    trades_won AS wins,
                    trades_lost AS losses
                FROM daily_stats
                WHERE date >= ?
                ORDER BY date
            """, [cutoff]).fetchall()

        return [
            {
                "date": str(r[0]),
                "daily_pnl": r[1],
                "cumulative_pnl": r[2],
                "trades": r[3],
                "wins": r[4],
                "losses": r[5],
            }
            for r in rows
        ]

    def get_strategy_breakdown(self, lookback_days: int = 60) -> list[dict]:
        """Strategy performance breakdown — single DuckDB query."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            # exit-time window + real-close filter (see get_performance)
            rows = conn.execute(f"""
                SELECT
                    strategy,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(realized_pnl), 2) AS total_pnl,
                    ROUND(AVG(realized_pnl), 2) AS avg_pnl,
                    ROUND(
                        CASE WHEN SUM(CASE WHEN realized_pnl <= 0 THEN ABS(realized_pnl) ELSE 0 END) > 0
                        THEN SUM(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END)
                             / SUM(CASE WHEN realized_pnl <= 0 THEN ABS(realized_pnl) ELSE 0 END)
                        ELSE 999.0 END, 2) AS profit_factor,
                    ROUND(AVG(ml_confidence), 3) AS avg_confidence
                FROM trades
                WHERE status = 'closed' AND COALESCE(exit_time, entry_time) >= ?
                    AND {_real_close_sql()}
                GROUP BY strategy
                ORDER BY total_pnl DESC
            """, [cutoff]).fetchall()

        cols = ["strategy", "trades", "wins", "win_rate_pct", "total_pnl",
                "avg_pnl", "profit_factor", "avg_confidence"]
        return [dict(zip(cols, r)) for r in rows]

    def get_symbol_breakdown(self, lookback_days: int = 60) -> list[dict]:
        """Symbol performance breakdown."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            # exit-time window + real-close filter (see get_performance)
            rows = conn.execute(f"""
                SELECT
                    symbol,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(realized_pnl), 2) AS total_pnl,
                    ROUND(AVG(realized_pnl), 2) AS avg_pnl
                FROM trades
                WHERE status = 'closed' AND COALESCE(exit_time, entry_time) >= ?
                    AND {_real_close_sql()}
                GROUP BY symbol
                ORDER BY total_pnl DESC
            """, [cutoff]).fetchall()

        cols = ["symbol", "trades", "wins", "win_rate_pct", "total_pnl", "avg_pnl"]
        return [dict(zip(cols, r)) for r in rows]

    def get_regime_breakdown(self, lookback_days: int = 60) -> list[dict]:
        """Performance breakdown by market regime — new analytics not in SQLite."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            # exit-time window + real-close filter (see get_performance)
            rows = conn.execute(f"""
                SELECT
                    t.market_regime AS regime,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN t.realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    ROUND(SUM(CASE WHEN t.realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(t.realized_pnl), 2) AS total_pnl,
                    ROUND(AVG(t.realized_pnl), 2) AS avg_pnl,
                    ROUND(AVG(tc.entry_iv_rank), 3) AS avg_iv_rank,
                    ROUND(AVG(tc.entry_vix), 1) AS avg_vix
                FROM trades t
                LEFT JOIN trade_context tc ON t.trade_id = tc.trade_id
                WHERE t.status = 'closed'
                    AND COALESCE(t.exit_time, t.entry_time) >= ?
                    AND {_real_close_sql('t.')}
                    AND t.market_regime != ''
                GROUP BY t.market_regime
                ORDER BY total_pnl DESC
            """, [cutoff]).fetchall()

        cols = ["regime", "trades", "wins", "win_rate_pct", "total_pnl",
                "avg_pnl", "avg_iv_rank", "avg_vix"]
        return [dict(zip(cols, r)) for r in rows]

    def get_strategy_regime_matrix(self, lookback_days: int = 90) -> list[dict]:
        """Strategy x Regime performance matrix — which strategies work in which regimes."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            # exit-time window + real-close filter (see get_performance)
            rows = conn.execute(f"""
                SELECT
                    strategy,
                    market_regime AS regime,
                    COUNT(*) AS trades,
                    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(realized_pnl), 2) AS total_pnl
                FROM trades
                WHERE status = 'closed' AND COALESCE(exit_time, entry_time) >= ?
                    AND {_real_close_sql()}
                    AND market_regime != ''
                GROUP BY strategy, market_regime
                HAVING COUNT(*) >= 2
                ORDER BY strategy, total_pnl DESC
            """, [cutoff]).fetchall()

        cols = ["strategy", "regime", "trades", "win_rate_pct", "total_pnl"]
        return [dict(zip(cols, r)) for r in rows]

    def get_hourly_performance(self, lookback_days: int = 60) -> list[dict]:
        """Win rate by hour of day — find bad trading hours."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            # exit-time window + real-close filter (see get_performance);
            # grouping stays on ENTRY hour — the question is entry timing.
            rows = conn.execute(f"""
                SELECT
                    EXTRACT(HOUR FROM entry_time) AS hour,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(realized_pnl), 2) AS total_pnl
                FROM trades
                WHERE status = 'closed' AND COALESCE(exit_time, entry_time) >= ?
                    AND {_real_close_sql()}
                GROUP BY EXTRACT(HOUR FROM entry_time)
                ORDER BY hour
            """, [cutoff]).fetchall()

        cols = ["hour", "trades", "wins", "win_rate_pct", "total_pnl"]
        return [dict(zip(cols, r)) for r in rows]

    def get_rolling_sharpe(self, window_days: int = 20, lookback_days: int = 90) -> list[dict]:
        """Rolling Sharpe ratio over time — uses DuckDB window functions.

        NOTE: SQRT(252) is CORRECT here — this operates on DAILY P&L from
        daily_stats, not per-trade P&L, so daily annualization applies.
        """
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        # R8: the window size is INLINED — a bound '?' inside a named WINDOW
        # gets duplicated by DuckDB's OVER-w expansion (6 params wanted, 2
        # supplied), so this method raised on every call since it was written.
        window_rows = max(0, int(window_days) - 1)

        with self._get_conn(read_only=True) as conn:
            rows = conn.execute(f"""
                WITH daily_returns AS (
                    SELECT date, total_pnl
                    FROM daily_stats
                    WHERE date >= ?
                    ORDER BY date
                )
                SELECT
                    date,
                    total_pnl,
                    AVG(total_pnl) OVER w AS rolling_mean,
                    STDDEV(total_pnl) OVER w AS rolling_std,
                    CASE WHEN STDDEV(total_pnl) OVER w > 0
                         THEN (AVG(total_pnl) OVER w / STDDEV(total_pnl) OVER w) * SQRT(252)
                         ELSE 0 END AS rolling_sharpe
                FROM daily_returns
                WINDOW w AS (ORDER BY date ROWS BETWEEN {window_rows} PRECEDING AND CURRENT ROW)
                ORDER BY date
            """, [cutoff]).fetchall()

        return [
            {
                "date": str(r[0]),
                "daily_pnl": r[1],
                "rolling_mean": r[2],
                "rolling_std": r[3],
                "rolling_sharpe": r[4],
            }
            for r in rows
        ]

    def get_exit_efficiency(self, lookback_days: int = 60) -> list[dict]:
        """Analyze profit capture efficiency per strategy."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            # exit-time window + real-close filter (see get_performance)
            rows = conn.execute(f"""
                SELECT
                    strategy,
                    COUNT(*) AS trades,
                    ROUND(AVG(peak_pnl_pct) * 100, 1) AS avg_peak_pct,
                    ROUND(AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END), 2) AS avg_realized_win,
                    ROUND(AVG(CAST(exit_reason_detailed != '' AS INTEGER)), 2) AS pct_with_exit_reason
                FROM trades
                WHERE status = 'closed' AND COALESCE(exit_time, entry_time) >= ?
                    AND {_real_close_sql()}
                    AND peak_pnl_pct > 0
                GROUP BY strategy
                HAVING COUNT(*) >= 2
                ORDER BY avg_peak_pct DESC
            """, [cutoff]).fetchall()

        cols = ["strategy", "trades", "avg_peak_pct", "avg_realized_win", "pct_with_exit_reason"]
        return [dict(zip(cols, r)) for r in rows]

    def get_confidence_band_analysis(self, lookback_days: int = 60) -> list[dict]:
        """Analyze win rate by ML confidence bands."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            # exit-time window + real-close filter (see get_performance)
            rows = conn.execute(f"""
                SELECT
                    CASE
                        WHEN ml_confidence < 0.60 THEN '0.50-0.60'
                        WHEN ml_confidence < 0.70 THEN '0.60-0.70'
                        WHEN ml_confidence < 0.80 THEN '0.70-0.80'
                        WHEN ml_confidence < 0.90 THEN '0.80-0.90'
                        ELSE '0.90-1.00'
                    END AS confidence_band,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(realized_pnl), 2) AS total_pnl,
                    ROUND(AVG(realized_pnl), 2) AS avg_pnl
                FROM trades
                WHERE status = 'closed' AND COALESCE(exit_time, entry_time) >= ?
                    AND {_real_close_sql()}
                GROUP BY confidence_band
                ORDER BY confidence_band
            """, [cutoff]).fetchall()

        cols = ["confidence_band", "trades", "wins", "win_rate_pct", "total_pnl", "avg_pnl"]
        return [dict(zip(cols, r)) for r in rows]

    def get_iv_rank_analysis(self, lookback_days: int = 90) -> list[dict]:
        """Analyze strategy performance across IV rank quintiles."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn(read_only=True) as conn:
            # exit-time window + real-close filter (see get_performance)
            rows = conn.execute(f"""
                SELECT
                    t.strategy,
                    CASE
                        WHEN tc.entry_iv_rank < 0.20 THEN 'Very Low (0-20)'
                        WHEN tc.entry_iv_rank < 0.40 THEN 'Low (20-40)'
                        WHEN tc.entry_iv_rank < 0.60 THEN 'Mid (40-60)'
                        WHEN tc.entry_iv_rank < 0.80 THEN 'High (60-80)'
                        ELSE 'Very High (80-100)'
                    END AS iv_quintile,
                    COUNT(*) AS trades,
                    ROUND(SUM(CASE WHEN t.realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(t.realized_pnl), 2) AS total_pnl,
                    ROUND(AVG(t.realized_pnl), 2) AS avg_pnl
                FROM trades t
                JOIN trade_context tc ON t.trade_id = tc.trade_id
                WHERE t.status = 'closed'
                    AND COALESCE(t.exit_time, t.entry_time) >= ?
                    AND {_real_close_sql('t.')}
                GROUP BY t.strategy, iv_quintile
                HAVING COUNT(*) >= 2
                ORDER BY t.strategy, iv_quintile
            """, [cutoff]).fetchall()

        cols = ["strategy", "iv_quintile", "trades", "win_rate_pct", "total_pnl", "avg_pnl"]
        return [dict(zip(cols, r)) for r in rows]

    def get_trade_count(self) -> int:
        """Get total number of trades in DuckDB."""
        with self._get_conn(read_only=True) as conn:
            result = conn.execute("SELECT COUNT(*) FROM trades").fetchone()
        return result[0] if result else 0

    def sync_from_sqlite(self, sqlite_path: Path) -> int:
        """Bulk-import trades from SQLite into DuckDB (initial migration).

        Returns the number of trades imported.
        """
        import sqlite3

        if not sqlite_path.exists():
            log.warning("sqlite_not_found", path=str(sqlite_path))
            return 0

        with sqlite3.connect(sqlite_path) as sq_conn:
            sq_conn.row_factory = sqlite3.Row

            # Import trades
            rows = sq_conn.execute("SELECT * FROM trades").fetchall()
            trades = [dict(r) for r in rows]

            # Import daily_stats
            ds_rows = sq_conn.execute("SELECT * FROM daily_stats").fetchall()
            daily = [dict(r) for r in ds_rows]

            # Import trade_context
            tc_rows = sq_conn.execute("SELECT * FROM trade_context").fetchall()
            contexts = [dict(r) for r in tc_rows]

        count = 0
        for t in trades:
            try:
                self.ingest_trade(t)
                count += 1
            except Exception as e:
                log.warning("sync_trade_failed", trade_id=t.get("trade_id"), error=str(e))

        for d in daily:
            try:
                self.ingest_daily_stats(d)
            except Exception as e:
                log.warning("sync_daily_failed", date=d.get("date"), error=str(e))

        for c in contexts:
            try:
                self.ingest_trade_context(c)
            except Exception as e:
                log.warning("sync_context_failed", trade_id=c.get("trade_id"), error=str(e))

        log.info("sqlite_sync_complete", trades=count, daily_stats=len(daily), contexts=len(contexts))
        return count

    # ------------------------------------------------------------------
    # Equity stats (yfinance fundamentals)
    # ------------------------------------------------------------------

    _EQUITY_STATS_COLS = [
        "symbol", "updated_at", "company_name", "sector", "industry",
        "country", "exchange", "market_cap", "pe_ratio", "forward_pe",
        "pb_ratio", "ps_ratio", "ev_ebitda", "eps", "book_value_ps",
        "revenue_ps", "dividend_yield", "dividend_rate", "payout_ratio",
        "beta", "week52_high", "week52_low", "avg_volume_30d",
        "float_shares", "shares_outstanding",
        "analyst_target_mean", "analyst_target_high", "analyst_target_low",
        "analyst_rating", "analyst_count",
    ]

    def _ensure_equity_stats_table(self, conn, table: str) -> None:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                symbol              VARCHAR PRIMARY KEY,
                updated_at          TIMESTAMP NOT NULL,
                company_name        VARCHAR DEFAULT '',
                sector              VARCHAR DEFAULT '',
                industry            VARCHAR DEFAULT '',
                country             VARCHAR DEFAULT '',
                exchange            VARCHAR DEFAULT '',
                market_cap          BIGINT  DEFAULT 0,
                pe_ratio            DOUBLE  DEFAULT 0,
                forward_pe          DOUBLE  DEFAULT 0,
                pb_ratio            DOUBLE  DEFAULT 0,
                ps_ratio            DOUBLE  DEFAULT 0,
                ev_ebitda           DOUBLE  DEFAULT 0,
                eps                 DOUBLE  DEFAULT 0,
                book_value_ps       DOUBLE  DEFAULT 0,
                revenue_ps          DOUBLE  DEFAULT 0,
                dividend_yield      DOUBLE  DEFAULT 0,
                dividend_rate       DOUBLE  DEFAULT 0,
                payout_ratio        DOUBLE  DEFAULT 0,
                beta                DOUBLE  DEFAULT 0,
                week52_high         DOUBLE  DEFAULT 0,
                week52_low          DOUBLE  DEFAULT 0,
                avg_volume_30d      BIGINT  DEFAULT 0,
                float_shares        BIGINT  DEFAULT 0,
                shares_outstanding  BIGINT  DEFAULT 0,
                analyst_target_mean DOUBLE  DEFAULT 0,
                analyst_target_high DOUBLE  DEFAULT 0,
                analyst_target_low  DOUBLE  DEFAULT 0,
                analyst_rating      VARCHAR DEFAULT '',
                analyst_count       INTEGER DEFAULT 0
            )
        """)

    def upsert_equity_stats(self, stats: dict, table: str = "equity_stats") -> None:
        """INSERT OR REPLACE into equity_stats — daily snapshot overwrites previous."""
        with self._get_conn() as conn:
            self._ensure_equity_stats_table(conn, table)
            conn.execute(f"""
                INSERT OR REPLACE INTO {table} VALUES (
                    $symbol, $updated_at, $company_name, $sector, $industry,
                    $country, $exchange, $market_cap, $pe_ratio, $forward_pe,
                    $pb_ratio, $ps_ratio, $ev_ebitda, $eps, $book_value_ps,
                    $revenue_ps, $dividend_yield, $dividend_rate, $payout_ratio,
                    $beta, $week52_high, $week52_low, $avg_volume_30d,
                    $float_shares, $shares_outstanding,
                    $analyst_target_mean, $analyst_target_high, $analyst_target_low,
                    $analyst_rating, $analyst_count
                )
            """, stats)
        log.debug("equity_stats_upserted", symbol=stats.get("symbol"), table=table)

    def get_equity_stats(self, symbol: str, table: str = "equity_stats") -> dict | None:
        """Return latest fundamental snapshot for one symbol, or None."""
        with self._get_conn(read_only=True) as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE symbol = ?", [symbol]
            ).fetchone()
        if row is None:
            return None
        return dict(zip(self._EQUITY_STATS_COLS, row))

    def get_all_equity_stats(self, table: str = "equity_stats") -> list[dict]:
        """Return all equity stats rows (for dashboard display)."""
        with self._get_conn(read_only=True) as conn:
            rows = conn.execute(
                f"SELECT * FROM {table} ORDER BY symbol"
            ).fetchall()
        return [dict(zip(self._EQUITY_STATS_COLS, r)) for r in rows]

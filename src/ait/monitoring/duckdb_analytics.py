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
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb

from ait.utils.logging import get_logger

log = get_logger("monitoring.duckdb")

DUCK_DB_PATH = Path("data/ait_analytics.duckdb")


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

    def _get_conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._db_path))

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

        log.info("duckdb_initialized", path=str(self._db_path))

    # ------------------------------------------------------------------
    # Ingest (write path — called by StateManager on trade close)
    # ------------------------------------------------------------------

    def ingest_trade(self, trade: dict) -> None:
        """Upsert a trade record into DuckDB analytics store."""
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

        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT realized_pnl, entry_time, exit_time
                FROM trades
                WHERE status = 'closed' AND entry_time >= ?
                ORDER BY entry_time
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

        # Sharpe & Sortino
        if len(pnls) > 1:
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl > 0:
                snap.sharpe_ratio = (mean_pnl / std_pnl) * math.sqrt(252)
            downside = [p for p in pnls if p < 0]
            if downside:
                ds_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
                if ds_std > 0:
                    snap.sortino_ratio = (mean_pnl / ds_std) * math.sqrt(252)

        # Drawdown
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        snap.max_drawdown_dollars = max_dd
        snap.max_drawdown_pct = max_dd / peak if peak > 0 else 0.0

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

        with self._get_conn() as conn:
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

        with self._get_conn() as conn:
            rows = conn.execute("""
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
                WHERE status = 'closed' AND entry_time >= ?
                GROUP BY strategy
                ORDER BY total_pnl DESC
            """, [cutoff]).fetchall()

        cols = ["strategy", "trades", "wins", "win_rate_pct", "total_pnl",
                "avg_pnl", "profit_factor", "avg_confidence"]
        return [dict(zip(cols, r)) for r in rows]

    def get_symbol_breakdown(self, lookback_days: int = 60) -> list[dict]:
        """Symbol performance breakdown."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    symbol,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(realized_pnl), 2) AS total_pnl,
                    ROUND(AVG(realized_pnl), 2) AS avg_pnl
                FROM trades
                WHERE status = 'closed' AND entry_time >= ?
                GROUP BY symbol
                ORDER BY total_pnl DESC
            """, [cutoff]).fetchall()

        cols = ["symbol", "trades", "wins", "win_rate_pct", "total_pnl", "avg_pnl"]
        return [dict(zip(cols, r)) for r in rows]

    def get_regime_breakdown(self, lookback_days: int = 60) -> list[dict]:
        """Performance breakdown by market regime — new analytics not in SQLite."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn() as conn:
            rows = conn.execute("""
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
                WHERE t.status = 'closed' AND t.entry_time >= ?
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

        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    strategy,
                    market_regime AS regime,
                    COUNT(*) AS trades,
                    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(realized_pnl), 2) AS total_pnl
                FROM trades
                WHERE status = 'closed' AND entry_time >= ?
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

        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    EXTRACT(HOUR FROM entry_time) AS hour,
                    COUNT(*) AS trades,
                    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    ROUND(SUM(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0 END)
                          / COUNT(*) * 100, 1) AS win_rate_pct,
                    ROUND(SUM(realized_pnl), 2) AS total_pnl
                FROM trades
                WHERE status = 'closed' AND entry_time >= ?
                GROUP BY EXTRACT(HOUR FROM entry_time)
                ORDER BY hour
            """, [cutoff]).fetchall()

        cols = ["hour", "trades", "wins", "win_rate_pct", "total_pnl"]
        return [dict(zip(cols, r)) for r in rows]

    def get_rolling_sharpe(self, window_days: int = 20, lookback_days: int = 90) -> list[dict]:
        """Rolling Sharpe ratio over time — uses DuckDB window functions."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn() as conn:
            rows = conn.execute("""
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
                WINDOW w AS (ORDER BY date ROWS BETWEEN ? PRECEDING AND CURRENT ROW)
                ORDER BY date
            """, [cutoff, window_days - 1]).fetchall()

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

        with self._get_conn() as conn:
            rows = conn.execute("""
                SELECT
                    strategy,
                    COUNT(*) AS trades,
                    ROUND(AVG(peak_pnl_pct) * 100, 1) AS avg_peak_pct,
                    ROUND(AVG(CASE WHEN realized_pnl > 0 THEN realized_pnl ELSE 0 END), 2) AS avg_realized_win,
                    ROUND(AVG(CAST(exit_reason_detailed != '' AS INTEGER)), 2) AS pct_with_exit_reason
                FROM trades
                WHERE status = 'closed' AND entry_time >= ?
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

        with self._get_conn() as conn:
            rows = conn.execute("""
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
                WHERE status = 'closed' AND entry_time >= ?
                GROUP BY confidence_band
                ORDER BY confidence_band
            """, [cutoff]).fetchall()

        cols = ["confidence_band", "trades", "wins", "win_rate_pct", "total_pnl", "avg_pnl"]
        return [dict(zip(cols, r)) for r in rows]

    def get_iv_rank_analysis(self, lookback_days: int = 90) -> list[dict]:
        """Analyze strategy performance across IV rank quintiles."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self._get_conn() as conn:
            rows = conn.execute("""
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
                WHERE t.status = 'closed' AND t.entry_time >= ?
                GROUP BY t.strategy, iv_quintile
                HAVING COUNT(*) >= 2
                ORDER BY t.strategy, iv_quintile
            """, [cutoff]).fetchall()

        cols = ["strategy", "iv_quintile", "trades", "win_rate_pct", "total_pnl", "avg_pnl"]
        return [dict(zip(cols, r)) for r in rows]

    def get_trade_count(self) -> int:
        """Get total number of trades in DuckDB."""
        with self._get_conn() as conn:
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

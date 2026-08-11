"""Trade analytics — comprehensive performance metrics.

Uses DuckDB as primary analytics engine with SQLite fallback.
DuckDB provides columnar vectorized execution for fast aggregations
and window functions over trade history.

Calculates:
- Sharpe ratio, Sortino ratio
- Maximum drawdown and recovery time
- Win rate by strategy, symbol, regime
- Profit factor
- Risk-adjusted returns
- Slippage tracking
"""

from __future__ import annotations

import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from ait.utils.logging import get_logger

log = get_logger("monitoring.analytics")

DB_PATH = Path("data/ait_state.db")

# Non-real closes (reconciler bookkeeping rows: never_filled / pending /
# migrated) must not count as closes in PF / win-rate / drawdown — mirrors
# the filter in status.py.
_REAL_CLOSE_SQL = (
    "COALESCE(exit_reason_detailed, '') NOT LIKE '%never_filled%' "
    "AND COALESCE(exit_reason_detailed, '') NOT LIKE '%pending%' "
    "AND COALESCE(exit_reason_detailed, '') NOT LIKE '%migrated%'"
)


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


def _annualization_factor(trades: list[dict]) -> float:
    """sqrt(trades-per-year) — replicates ait.backtesting.result.annualization_factor.

    sqrt(252) treated every TRADE as one trading DAY, overstating
    Sharpe/Sortino ~4x at typical trade frequency (see result.py BT-H4 note).
    Span runs first→last exit (COALESCE to entry when exit missing) to match
    the exit-time windowing of closed-trade reports; capped at daily sqrt(252).
    """
    try:
        dates = sorted(
            datetime.fromisoformat(str(t.get("exit_time") or t.get("entry_time"))[:19]).date()
            for t in trades
            if (t.get("exit_time") or t.get("entry_time"))
        )
        if len(dates) < 2:
            return 1.0
        span_days = max((dates[-1] - dates[0]).days, 1)
        trades_per_year = len(dates) / (span_days / 365.25)
        return math.sqrt(min(252.0, max(trades_per_year, 1.0)))
    except Exception:
        return 1.0


@dataclass
class PerformanceMetrics:
    """Comprehensive trading performance metrics."""

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
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    current_streak: int = 0  # Positive = winning streak, negative = losing
    recovery_factor: float = 0.0  # total_pnl / max_drawdown


class TradeAnalytics:
    """Computes detailed trading analytics from the trade database.

    Uses DuckDB as primary for read-heavy queries, with SQLite fallback.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._duck = self._init_duck()

    @staticmethod
    def _init_duck():
        try:
            from ait.monitoring.duckdb_analytics import DuckDBAnalytics
            return DuckDBAnalytics()
        except Exception:
            return None

    def get_performance(self, lookback_days: int = 30) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.

        Uses DuckDB when available for faster computation.
        """
        if self._duck:
            try:
                snap = self._duck.get_performance(lookback_days)
                return PerformanceMetrics(
                    total_trades=snap.total_trades,
                    total_pnl=snap.total_pnl,
                    win_rate=snap.win_rate,
                    profit_factor=snap.profit_factor,
                    sharpe_ratio=snap.sharpe_ratio,
                    sortino_ratio=snap.sortino_ratio,
                    max_drawdown_pct=snap.max_drawdown_pct,
                    max_drawdown_dollars=snap.max_drawdown_dollars,
                    avg_trade_pnl=snap.avg_trade_pnl,
                    avg_win=snap.avg_win,
                    avg_loss=snap.avg_loss,
                    largest_win=snap.largest_win,
                    largest_loss=snap.largest_loss,
                    avg_hold_hours=snap.avg_hold_hours,
                    consecutive_wins=snap.consecutive_wins,
                    consecutive_losses=snap.consecutive_losses,
                    current_streak=snap.current_streak,
                    recovery_factor=snap.recovery_factor,
                )
            except Exception as e:
                log.warning("duckdb_fallback", method="get_performance", error=str(e))

        # SQLite fallback
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        trades = self._get_closed_trades(cutoff)

        if not trades:
            return PerformanceMetrics()

        pnls = [t["realized_pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        metrics = PerformanceMetrics(
            total_trades=len(trades),
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
        metrics.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        # Sharpe ratio — annualized by ACTUAL trade frequency, not sqrt(252)
        # (sqrt(252) treated each trade as one trading day; see
        # ait.backtesting.result.annualization_factor)
        if len(pnls) > 1:
            import statistics
            ann = _annualization_factor(trades)
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl > 0:
                metrics.sharpe_ratio = (mean_pnl / std_pnl) * ann

            # Sortino — target-0 downside deviation over ALL returns:
            # sqrt(mean(min(r,0)^2)), matching result.py. The old
            # stdev-of-losses-about-their-own-mean gave a strategy losing a
            # consistent -$500 downside-dev≈0 and an absurd Sortino.
            downside_dev = math.sqrt(sum(min(p, 0.0) ** 2 for p in pnls) / len(pnls))
            if downside_dev > 0:
                metrics.sortino_ratio = (mean_pnl / downside_dev) * ann
            elif mean_pnl > 0:
                # R17: inconsistent with dashboard/walkforward/export.py's
                # _sortino_from_trades, which returns None for the identical
                # "no losing trades" case (float("inf") isn't valid JSON).
                # Not unified here: PerformanceMetrics.sortino_ratio is typed
                # float (not float | None) with no current JSON consumer —
                # widening the type has a ripple this fix doesn't need to
                # take on. Sanitize at whatever boundary first serializes
                # this to JSON, if one is ever added.
                metrics.sortino_ratio = float("inf")

        # Maximum drawdown
        dd_pct, dd_dollars = self._calculate_drawdown(pnls)
        metrics.max_drawdown_pct = dd_pct
        metrics.max_drawdown_dollars = dd_dollars

        # Recovery factor
        if dd_dollars > 0:
            metrics.recovery_factor = total_pnl / dd_dollars

        # Streaks
        max_wins, max_losses, current = self._calculate_streaks(pnls)
        metrics.consecutive_wins = max_wins
        metrics.consecutive_losses = max_losses
        metrics.current_streak = current

        # Average hold time
        metrics.avg_hold_hours = self._calculate_avg_hold_time(trades)

        return metrics

    def get_daily_pnl(self, lookback_days: int = 30) -> list[dict]:
        """Get daily P&L for charting."""
        if self._duck:
            try:
                return self._duck.get_daily_pnl(lookback_days)
            except Exception as e:
                log.warning("duckdb_fallback", method="get_daily_pnl", error=str(e))

        # SQLite fallback
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        if not self._db_path.exists():
            return []

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT date, total_pnl, trades_taken, trades_won, trades_lost
                   FROM daily_stats WHERE date >= ? ORDER BY date""",
                (cutoff,),
            ).fetchall()

        result = []
        cumulative = 0.0
        for r in rows:
            cumulative += r["total_pnl"]
            result.append({
                "date": r["date"],
                "daily_pnl": r["total_pnl"],
                "cumulative_pnl": cumulative,
                "trades": r["trades_taken"],
                "wins": r["trades_won"],
                "losses": r["trades_lost"],
            })

        return result

    def get_strategy_breakdown(self, lookback_days: int = 60) -> list[dict]:
        """Get performance breakdown by strategy."""
        if self._duck:
            try:
                return self._duck.get_strategy_breakdown(lookback_days)
            except Exception as e:
                log.warning("duckdb_fallback", method="get_strategy_breakdown", error=str(e))

        # SQLite fallback
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        trades = self._get_closed_trades(cutoff)

        by_strategy: dict[str, list[float]] = {}
        for t in trades:
            by_strategy.setdefault(t["strategy"], []).append(t["realized_pnl"])

        result = []
        for strategy, pnls in sorted(by_strategy.items()):
            wins = [p for p in pnls if p > 0]
            # Guard the DENOMINATOR: a $0 scratch trade passes any(p<=0)
            # but sums to zero gross loss → ZeroDivisionError.
            gross_losses = abs(sum(p for p in pnls if p <= 0))
            result.append({
                "strategy": strategy,
                "trades": len(pnls),
                "win_rate": len(wins) / len(pnls) if pnls else 0,
                "total_pnl": sum(pnls),
                "avg_pnl": sum(pnls) / len(pnls),
                "profit_factor": (
                    sum(wins) / gross_losses if gross_losses > 0 else float("inf")
                ),
            })

        return sorted(result, key=lambda x: x["total_pnl"], reverse=True)

    def get_symbol_breakdown(self, lookback_days: int = 60) -> list[dict]:
        """Get performance breakdown by symbol."""
        if self._duck:
            try:
                return self._duck.get_symbol_breakdown(lookback_days)
            except Exception as e:
                log.warning("duckdb_fallback", method="get_symbol_breakdown", error=str(e))

        # SQLite fallback
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        trades = self._get_closed_trades(cutoff)

        by_symbol: dict[str, list[float]] = {}
        for t in trades:
            by_symbol.setdefault(t["symbol"], []).append(t["realized_pnl"])

        result = []
        for symbol, pnls in sorted(by_symbol.items()):
            wins = [p for p in pnls if p > 0]
            result.append({
                "symbol": symbol,
                "trades": len(pnls),
                "win_rate": len(wins) / len(pnls) if pnls else 0,
                "total_pnl": sum(pnls),
                "avg_pnl": sum(pnls) / len(pnls),
            })

        return sorted(result, key=lambda x: x["total_pnl"], reverse=True)

    # --- Calculation helpers ---

    @staticmethod
    def _calculate_drawdown(pnls: list[float]) -> tuple[float, float]:
        """Calculate max drawdown from a sequence of P&L values.

        dd% is measured against the EQUITY high-water mark where
        equity = CAPITAL_BASE + cumulative P&L. The old cumulative-P&L-only
        peak (starting at 0) returned 0% after pure losses and absurd
        percentages on small samples. dd$ is still reported alongside.
        """
        if not pnls:
            return 0.0, 0.0

        base = _capital_base()
        equity = base
        peak = base
        max_dd_dollars = 0.0
        max_dd_pct = 0.0

        for pnl in pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd_dollars:
                max_dd_dollars = dd
            if peak > 0 and dd / peak > max_dd_pct:
                max_dd_pct = dd / peak

        return max_dd_pct, max_dd_dollars

    @staticmethod
    def _calculate_streaks(pnls: list[float]) -> tuple[int, int, int]:
        """Calculate max win/loss streaks and current streak."""
        if not pnls:
            return 0, 0, 0

        max_wins = 0
        max_losses = 0
        current = 0

        for pnl in pnls:
            if pnl > 0:
                current = current + 1 if current > 0 else 1
                max_wins = max(max_wins, current)
            else:
                current = current - 1 if current < 0 else -1
                max_losses = max(max_losses, abs(current))

        return max_wins, max_losses, current

    @staticmethod
    def _calculate_avg_hold_time(trades: list[dict]) -> float:
        """Calculate average hold time in hours."""
        hold_times = []
        for t in trades:
            if t.get("entry_time") and t.get("exit_time"):
                try:
                    entry = datetime.fromisoformat(t["entry_time"])
                    exit_ = datetime.fromisoformat(t["exit_time"])
                    hold_times.append((exit_ - entry).total_seconds() / 3600)
                except (ValueError, TypeError):
                    pass

        return sum(hold_times) / len(hold_times) if hold_times else 0.0

    def _get_closed_trades(self, since: str) -> list[dict]:
        """Get closed trades from DB.

        Windows on EXIT time (COALESCE to entry_time when NULL): a trade
        opened before the lookback window but closed inside it must count in
        the report — entry-time filtering hid late losers. Bookkeeping closes
        (never_filled / pending / migrated) are excluded, and rows are ordered
        by close time so drawdown/streaks follow realization order.
        """
        if not self._db_path.exists():
            return []

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""SELECT * FROM trades
                   WHERE status = 'closed'
                     AND COALESCE(exit_time, entry_time) >= ?
                     AND {_REAL_CLOSE_SQL}
                   ORDER BY COALESCE(exit_time, entry_time)""",
                (since,),
            ).fetchall()

        return [dict(r) for r in rows]

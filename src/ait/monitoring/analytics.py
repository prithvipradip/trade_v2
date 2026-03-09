"""Trade analytics — comprehensive performance metrics.

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
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from ait.utils.logging import get_logger

log = get_logger("monitoring.analytics")

DB_PATH = Path("data/ait_state.db")


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
    """Computes detailed trading analytics from the trade database."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path

    def get_performance(self, lookback_days: int = 30) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
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

        # Sharpe ratio (annualized, assuming daily P&L)
        if len(pnls) > 1:
            import statistics
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls)
            if std_pnl > 0:
                metrics.sharpe_ratio = (mean_pnl / std_pnl) * math.sqrt(252)

            # Sortino (downside deviation only)
            downside = [p for p in pnls if p < 0]
            if downside:
                downside_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
                if downside_std > 0:
                    metrics.sortino_ratio = (mean_pnl / downside_std) * math.sqrt(252)

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
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        trades = self._get_closed_trades(cutoff)

        by_strategy: dict[str, list[float]] = {}
        for t in trades:
            by_strategy.setdefault(t["strategy"], []).append(t["realized_pnl"])

        result = []
        for strategy, pnls in sorted(by_strategy.items()):
            wins = [p for p in pnls if p > 0]
            result.append({
                "strategy": strategy,
                "trades": len(pnls),
                "win_rate": len(wins) / len(pnls) if pnls else 0,
                "total_pnl": sum(pnls),
                "avg_pnl": sum(pnls) / len(pnls),
                "profit_factor": (
                    sum(wins) / abs(sum(p for p in pnls if p <= 0))
                    if any(p <= 0 for p in pnls) else float("inf")
                ),
            })

        return sorted(result, key=lambda x: x["total_pnl"], reverse=True)

    def get_symbol_breakdown(self, lookback_days: int = 60) -> list[dict]:
        """Get performance breakdown by symbol."""
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
        """Calculate max drawdown from a sequence of P&L values."""
        if not pnls:
            return 0.0, 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd_dollars = 0.0

        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd_dollars:
                max_dd_dollars = dd

        max_dd_pct = max_dd_dollars / peak if peak > 0 else 0.0
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
        """Get closed trades from DB."""
        if not self._db_path.exists():
            return []

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM trades
                   WHERE status = 'closed' AND entry_time >= ?
                   ORDER BY entry_time""",
                (since,),
            ).fetchall()

        return [dict(r) for r in rows]

"""Trade outcome analyzer — mines past trades for patterns.

Analyzes closed trades to identify:
- Which strategies win/lose and under what conditions
- Optimal confidence thresholds per strategy
- IV rank sweet spots per strategy
- Time-of-day patterns
- Symbol-specific behavior
- Slippage and fill quality
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from ait.utils.logging import get_logger

log = get_logger("learning.analyzer")

DB_PATH = Path("data/ait_state.db")


@dataclass
class StrategyStats:
    """Performance statistics for a single strategy."""

    strategy: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    avg_confidence: float = 0.0
    avg_hold_time_hours: float = 0.0
    best_iv_range: tuple[float, float] = (0.0, 100.0)
    best_confidence_range: tuple[float, float] = (0.0, 1.0)


@dataclass
class SymbolStats:
    """Performance statistics for a single symbol."""

    symbol: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    best_strategy: str = ""


@dataclass
class TradeInsight:
    """A specific learning from trade analysis."""

    category: str  # "strategy", "symbol", "confidence", "timing", "regime"
    insight: str  # Human-readable description
    action: str  # What to change
    confidence: float  # How confident we are in this insight (0-1)
    data: dict = field(default_factory=dict)


class TradeAnalyzer:
    """Analyzes historical trades to extract learnings."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path

    def analyze_all(self, lookback_days: int = 30) -> list[TradeInsight]:
        """Run all analyses and return actionable insights."""
        insights = []
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        trades = self._get_closed_trades(cutoff)
        if len(trades) < 5:
            log.info("insufficient_trades_for_analysis", count=len(trades))
            return insights

        insights.extend(self._analyze_strategies(trades))
        insights.extend(self._analyze_confidence_thresholds(trades))
        insights.extend(self._analyze_symbols(trades))
        insights.extend(self._analyze_regimes(trades))
        insights.extend(self._analyze_hold_times(trades))
        insights.extend(self._analyze_slippage(trades))

        # Sort by confidence (most certain insights first)
        insights.sort(key=lambda x: x.confidence, reverse=True)

        log.info("trade_analysis_complete", insights_count=len(insights), trades_analyzed=len(trades))
        return insights

    def get_strategy_stats(self, lookback_days: int = 60) -> dict[str, StrategyStats]:
        """Get performance stats grouped by strategy."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        trades = self._get_closed_trades(cutoff)

        stats: dict[str, StrategyStats] = {}
        for t in trades:
            strategy = t["strategy"]
            if strategy not in stats:
                stats[strategy] = StrategyStats(strategy=strategy)

            s = stats[strategy]
            s.total_trades += 1
            s.total_pnl += t["realized_pnl"]
            s.avg_confidence += t["ml_confidence"]

            if t["realized_pnl"] > 0:
                s.wins += 1
                s.avg_win += t["realized_pnl"]
            else:
                s.losses += 1
                s.avg_loss += t["realized_pnl"]

            # Hold time
            if t["entry_time"] and t["exit_time"]:
                try:
                    entry = datetime.fromisoformat(t["entry_time"])
                    exit_ = datetime.fromisoformat(t["exit_time"])
                    s.avg_hold_time_hours += (exit_ - entry).total_seconds() / 3600
                except (ValueError, TypeError):
                    pass

        # Compute averages
        for s in stats.values():
            if s.total_trades > 0:
                s.avg_pnl = s.total_pnl / s.total_trades
                s.win_rate = s.wins / s.total_trades
                s.avg_confidence /= s.total_trades
                s.avg_hold_time_hours /= s.total_trades
            if s.wins > 0:
                s.avg_win /= s.wins
            if s.losses > 0:
                s.avg_loss /= s.losses
            gross_wins = s.avg_win * s.wins if s.wins > 0 else 0
            gross_losses = abs(s.avg_loss * s.losses) if s.losses > 0 else 0
            s.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        return stats

    def get_symbol_stats(self, lookback_days: int = 60) -> dict[str, SymbolStats]:
        """Get performance stats grouped by symbol."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        trades = self._get_closed_trades(cutoff)

        stats: dict[str, SymbolStats] = {}
        for t in trades:
            symbol = t["symbol"]
            if symbol not in stats:
                stats[symbol] = SymbolStats(symbol=symbol)

            s = stats[symbol]
            s.total_trades += 1
            s.total_pnl += t["realized_pnl"]
            if t["realized_pnl"] > 0:
                s.wins += 1
            else:
                s.losses += 1

        for s in stats.values():
            if s.total_trades > 0:
                s.win_rate = s.wins / s.total_trades

        return stats

    # --- Analysis methods ---

    def _analyze_strategies(self, trades: list[dict]) -> list[TradeInsight]:
        """Identify winning and losing strategies."""
        insights = []
        by_strategy: dict[str, list[dict]] = {}

        for t in trades:
            by_strategy.setdefault(t["strategy"], []).append(t)

        for strategy, strades in by_strategy.items():
            if len(strades) < 3:
                continue

            wins = sum(1 for t in strades if t["realized_pnl"] > 0)
            win_rate = wins / len(strades)
            total_pnl = sum(t["realized_pnl"] for t in strades)
            avg_pnl = total_pnl / len(strades)

            # Strategy consistently losing
            if win_rate < 0.35 and len(strades) >= 5:
                insights.append(TradeInsight(
                    category="strategy",
                    insight=f"{strategy} has {win_rate:.0%} win rate over {len(strades)} trades (avg P&L: ${avg_pnl:.0f})",
                    action=f"disable_{strategy}",
                    confidence=min(0.9, len(strades) / 20),
                    data={"strategy": strategy, "win_rate": win_rate, "avg_pnl": avg_pnl, "trades": len(strades)},
                ))

            # Strategy consistently winning
            elif win_rate > 0.65 and total_pnl > 0 and len(strades) >= 5:
                insights.append(TradeInsight(
                    category="strategy",
                    insight=f"{strategy} has {win_rate:.0%} win rate, ${total_pnl:.0f} total P&L",
                    action=f"boost_{strategy}",
                    confidence=min(0.9, len(strades) / 20),
                    data={"strategy": strategy, "win_rate": win_rate, "total_pnl": total_pnl},
                ))

        return insights

    def _analyze_confidence_thresholds(self, trades: list[dict]) -> list[TradeInsight]:
        """Find optimal confidence threshold from trade outcomes."""
        insights = []

        # Group by confidence bands
        bands = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
        for low, high in bands:
            band_trades = [t for t in trades if low <= t["ml_confidence"] < high]
            if len(band_trades) < 3:
                continue

            wins = sum(1 for t in band_trades if t["realized_pnl"] > 0)
            win_rate = wins / len(band_trades)
            avg_pnl = sum(t["realized_pnl"] for t in band_trades) / len(band_trades)

            if win_rate < 0.40 and avg_pnl < 0:
                insights.append(TradeInsight(
                    category="confidence",
                    insight=f"Trades at confidence {low:.0%}-{high:.0%} have {win_rate:.0%} win rate, avg P&L ${avg_pnl:.0f}",
                    action=f"raise_min_confidence_to_{high:.2f}",
                    confidence=min(0.85, len(band_trades) / 15),
                    data={"low": low, "high": high, "win_rate": win_rate, "avg_pnl": avg_pnl},
                ))

        return insights

    def _analyze_symbols(self, trades: list[dict]) -> list[TradeInsight]:
        """Identify symbols that consistently lose."""
        insights = []
        by_symbol: dict[str, list[dict]] = {}

        for t in trades:
            by_symbol.setdefault(t["symbol"], []).append(t)

        for symbol, strades in by_symbol.items():
            if len(strades) < 3:
                continue

            total_pnl = sum(t["realized_pnl"] for t in strades)
            wins = sum(1 for t in strades if t["realized_pnl"] > 0)
            win_rate = wins / len(strades)

            if win_rate < 0.30 and total_pnl < 0:
                insights.append(TradeInsight(
                    category="symbol",
                    insight=f"{symbol}: {win_rate:.0%} win rate, ${total_pnl:.0f} total P&L over {len(strades)} trades",
                    action=f"remove_symbol_{symbol}",
                    confidence=min(0.8, len(strades) / 15),
                    data={"symbol": symbol, "win_rate": win_rate, "total_pnl": total_pnl},
                ))

        return insights

    def _analyze_regimes(self, trades: list[dict]) -> list[TradeInsight]:
        """Analyze performance across market regimes."""
        insights = []
        by_regime: dict[str, list[dict]] = {}

        for t in trades:
            regime = t.get("market_regime", "")
            if regime:
                by_regime.setdefault(regime, []).append(t)

        for regime, strades in by_regime.items():
            if len(strades) < 3:
                continue

            total_pnl = sum(t["realized_pnl"] for t in strades)
            wins = sum(1 for t in strades if t["realized_pnl"] > 0)
            win_rate = wins / len(strades)

            if win_rate < 0.35 and total_pnl < 0:
                insights.append(TradeInsight(
                    category="regime",
                    insight=f"Poor performance in {regime} regime: {win_rate:.0%} win rate",
                    action=f"reduce_trading_in_{regime}",
                    confidence=min(0.75, len(strades) / 15),
                    data={"regime": regime, "win_rate": win_rate, "total_pnl": total_pnl},
                ))

        return insights

    def _analyze_hold_times(self, trades: list[dict]) -> list[TradeInsight]:
        """Analyze if we're holding too long or too short."""
        insights = []
        winners = []
        losers = []

        for t in trades:
            if not t.get("entry_time") or not t.get("exit_time"):
                continue
            try:
                entry = datetime.fromisoformat(t["entry_time"])
                exit_ = datetime.fromisoformat(t["exit_time"])
                hours = (exit_ - entry).total_seconds() / 3600
                if t["realized_pnl"] > 0:
                    winners.append(hours)
                else:
                    losers.append(hours)
            except (ValueError, TypeError):
                continue

        if len(winners) >= 3 and len(losers) >= 3:
            avg_win_hold = sum(winners) / len(winners)
            avg_loss_hold = sum(losers) / len(losers)

            if avg_loss_hold > avg_win_hold * 2:
                insights.append(TradeInsight(
                    category="timing",
                    insight=f"Losing trades held {avg_loss_hold:.0f}h avg vs winners {avg_win_hold:.0f}h — cut losers faster",
                    action="tighten_stop_loss",
                    confidence=0.7,
                    data={"avg_win_hold_hours": avg_win_hold, "avg_loss_hold_hours": avg_loss_hold},
                ))

        return insights

    def _analyze_slippage(self, trades: list[dict]) -> list[TradeInsight]:
        """Analyze fill quality (expected vs actual entry price)."""
        # Slippage analysis requires tracking expected vs actual fill prices.
        # The current schema stores entry_price (which is the expected limit price).
        # When we have actual fill data, we can compare. For now, return empty.
        return []

    # --- Data access ---

    def _get_closed_trades(self, since: str) -> list[dict]:
        """Get closed trades since a given date."""
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

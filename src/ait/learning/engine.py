"""Self-learning engine — the brain that makes the bot adapt.

Runs periodically (post-market or on restart) to:
1. Analyze recent trade outcomes
2. Generate insights about what works and what doesn't
3. Apply bounded adaptations to strategy selection, sizing, and risk params
4. Track adaptation history for transparency
5. Retrain ML models with trade outcome feedback

The bot gets smarter over time while staying within safety guardrails.
"""

from __future__ import annotations

import json
from datetime import datetime

from ait.bot.state import StateManager
from ait.learning.analyzer import TradeAnalyzer, TradeInsight
from ait.learning.adaptor import Adaptation, StrategyAdaptor
from ait.utils.logging import get_logger

log = get_logger("learning.engine")


class LearningEngine:
    """Orchestrates the self-learning cycle."""

    def __init__(
        self,
        state: StateManager,
        analyzer: TradeAnalyzer | None = None,
        adaptor: StrategyAdaptor | None = None,
    ) -> None:
        self._state = state
        self._analyzer = analyzer or TradeAnalyzer()
        self._adaptor = adaptor or StrategyAdaptor(state)

    @property
    def adaptor(self) -> StrategyAdaptor:
        """Access the strategy adaptor for querying current overrides."""
        return self._adaptor

    def run_learning_cycle(self, lookback_days: int = 30) -> dict:
        """Run a complete learning cycle.

        Returns summary of what was learned and applied.
        """
        log.info("learning_cycle_starting", lookback_days=lookback_days)

        # 1. Analyze trades
        insights = self._analyzer.analyze_all(lookback_days)

        if not insights:
            log.info("learning_cycle_no_insights")
            return {"insights": 0, "adaptations": 0, "details": []}

        # 2. Apply adaptations
        adaptations = self._adaptor.apply_insights(insights)

        # 3. Log summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "insights": len(insights),
            "adaptations": len(adaptations),
            "details": [
                {
                    "parameter": a.parameter,
                    "old": a.old_value,
                    "new": a.new_value,
                    "reason": a.reason,
                }
                for a in adaptations
            ],
            "top_insights": [
                {"category": i.category, "insight": i.insight, "confidence": i.confidence}
                for i in insights[:5]
            ],
        }

        # 4. Store learning history
        self._record_cycle(summary)

        log.info(
            "learning_cycle_complete",
            insights=len(insights),
            adaptations=len(adaptations),
        )

        return summary

    def get_strategy_performance(self, lookback_days: int = 60) -> dict:
        """Get strategy performance report for dashboard/logging."""
        stats = self._analyzer.get_strategy_stats(lookback_days)
        return {
            name: {
                "trades": s.total_trades,
                "win_rate": f"{s.win_rate:.0%}",
                "total_pnl": f"${s.total_pnl:.0f}",
                "avg_pnl": f"${s.avg_pnl:.0f}",
                "profit_factor": f"{s.profit_factor:.1f}",
            }
            for name, s in stats.items()
        }

    def get_symbol_performance(self, lookback_days: int = 60) -> dict:
        """Get symbol performance report."""
        stats = self._analyzer.get_symbol_stats(lookback_days)
        return {
            name: {
                "trades": s.total_trades,
                "win_rate": f"{s.win_rate:.0%}",
                "total_pnl": f"${s.total_pnl:.0f}",
            }
            for name, s in stats.items()
        }

    def get_current_adaptations(self) -> dict:
        """Get all current learning-based overrides."""
        return self._adaptor.get_all_overrides()

    def get_learning_history(self, n: int = 10) -> list[dict]:
        """Get recent learning cycle results."""
        raw = self._state.get_state("learning_history", "[]")
        try:
            history = json.loads(raw)
            return history[-n:]
        except json.JSONDecodeError:
            return []

    def reset_all_learning(self) -> None:
        """Reset all learned adaptations. Use if bot is performing poorly."""
        self._adaptor.reset()
        self._state.set_state("learning_history", "[]")
        log.info("all_learning_reset")

    def _record_cycle(self, summary: dict) -> None:
        """Append a learning cycle to history."""
        raw = self._state.get_state("learning_history", "[]")
        try:
            history = json.loads(raw)
        except json.JSONDecodeError:
            history = []

        history.append(summary)

        # Keep last 50 cycles
        if len(history) > 50:
            history = history[-50:]

        self._state.set_state("learning_history", json.dumps(history))

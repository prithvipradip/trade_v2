"""Counterfactual analysis — tracks what would have happened with skipped trades.

When the bot rejects a signal (risk limits, meta-label, low confidence, etc.),
we record it here. Later, we check what the actual outcome would have been.
This helps identify:
- Systematic missed opportunities (filter too aggressive)
- Correctly avoided losses (filter working well)
- Strategy-specific filter accuracy

The bot uses this data to tune its filters over time.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ait.utils.logging import get_logger

log = get_logger("learning.counterfactual")

STATE_FILE = Path("data/counterfactual_log.json")


@dataclass
class SkippedTrade:
    """A trade signal that was generated but not executed."""

    timestamp: str
    symbol: str
    strategy: str
    direction: str
    confidence: float
    entry_price: float
    reject_reason: str
    # Filled in later when we check the outcome
    exit_price: float | None = None
    hypothetical_pnl: float | None = None
    would_have_won: bool | None = None
    outcome_checked: bool = False


class CounterfactualTracker:
    """Tracks skipped trades and evaluates what would have happened.

    Usage:
    1. Call record_skip() when a signal is rejected
    2. Call evaluate_outcomes() periodically to check actual prices
    3. Call get_analysis() to see filter effectiveness
    """

    def __init__(self, max_history: int = 500) -> None:
        self._skipped: list[SkippedTrade] = []
        self._max_history = max_history
        self._load_state()

    def record_skip(
        self,
        symbol: str,
        strategy: str,
        direction: str,
        confidence: float,
        entry_price: float,
        reject_reason: str,
    ) -> None:
        """Record a signal that was skipped."""
        record = SkippedTrade(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            strategy=strategy,
            direction=direction,
            confidence=confidence,
            entry_price=entry_price,
            reject_reason=reject_reason,
        )
        self._skipped.append(record)

        # Trim history
        if len(self._skipped) > self._max_history:
            self._skipped = self._skipped[-self._max_history:]

        log.info(
            "counterfactual_recorded",
            symbol=symbol,
            strategy=strategy,
            reason=reject_reason,
            confidence=f"{confidence:.2%}",
        )
        self._save_state()

    def evaluate_outcomes(
        self,
        price_lookup: dict[str, float],
        hold_return_threshold: float = 0.02,
    ) -> int:
        """Evaluate hypothetical outcomes for unchecked skipped trades.

        Args:
            price_lookup: {symbol: current_price} for each symbol
            hold_return_threshold: Assume exit after this % move (default 2%)

        Returns:
            Number of outcomes evaluated.
        """
        evaluated = 0

        for trade in self._skipped:
            if trade.outcome_checked:
                continue

            current_price = price_lookup.get(trade.symbol)
            if current_price is None or current_price <= 0:
                continue

            # Simple model: would the direction have been correct?
            price_change_pct = (current_price - trade.entry_price) / trade.entry_price

            if trade.direction in ("bullish", "long"):
                trade.would_have_won = price_change_pct > hold_return_threshold
                trade.hypothetical_pnl = price_change_pct * trade.entry_price * 100  # per contract
            elif trade.direction in ("bearish", "short"):
                trade.would_have_won = price_change_pct < -hold_return_threshold
                trade.hypothetical_pnl = -price_change_pct * trade.entry_price * 100
            else:
                # Neutral strategies — harder to evaluate
                trade.would_have_won = abs(price_change_pct) < hold_return_threshold
                trade.hypothetical_pnl = 0.0

            trade.exit_price = current_price
            trade.outcome_checked = True
            evaluated += 1

        if evaluated > 0:
            self._save_state()
            log.info("counterfactual_evaluated", count=evaluated)

        return evaluated

    def get_analysis(self) -> dict:
        """Analyze counterfactual outcomes to measure filter effectiveness.

        Returns a summary of how well our filters are working.
        """
        checked = [t for t in self._skipped if t.outcome_checked]
        if not checked:
            return {
                "total_skipped": len(self._skipped),
                "evaluated": 0,
                "filter_accuracy": 0.0,
                "by_reason": {},
                "by_strategy": {},
            }

        # Overall: how many skipped trades would have lost?
        correct_skips = sum(1 for t in checked if not t.would_have_won)
        filter_accuracy = correct_skips / len(checked) if checked else 0.0

        # Break down by reject reason
        by_reason: dict[str, dict] = defaultdict(lambda: {"total": 0, "would_won": 0, "would_lost": 0, "pnl": 0.0})
        for t in checked:
            r = by_reason[t.reject_reason]
            r["total"] += 1
            if t.would_have_won:
                r["would_won"] += 1
            else:
                r["would_lost"] += 1
            r["pnl"] += t.hypothetical_pnl or 0.0

        # Break down by strategy
        by_strategy: dict[str, dict] = defaultdict(lambda: {"total": 0, "would_won": 0, "would_lost": 0, "pnl": 0.0})
        for t in checked:
            s = by_strategy[t.strategy]
            s["total"] += 1
            if t.would_have_won:
                s["would_won"] += 1
            else:
                s["would_lost"] += 1
            s["pnl"] += t.hypothetical_pnl or 0.0

        # Compute accuracy per reason
        reason_stats = {}
        for reason, data in by_reason.items():
            reason_stats[reason] = {
                **data,
                "accuracy": data["would_lost"] / data["total"] if data["total"] > 0 else 0.0,
            }

        strategy_stats = {}
        for strategy, data in by_strategy.items():
            strategy_stats[strategy] = {
                **data,
                "accuracy": data["would_lost"] / data["total"] if data["total"] > 0 else 0.0,
            }

        total_missed_pnl = sum(t.hypothetical_pnl or 0 for t in checked if t.would_have_won)

        return {
            "total_skipped": len(self._skipped),
            "evaluated": len(checked),
            "filter_accuracy": filter_accuracy,
            "correct_skips": correct_skips,
            "missed_opportunities": len(checked) - correct_skips,
            "total_missed_pnl": total_missed_pnl,
            "by_reason": reason_stats,
            "by_strategy": strategy_stats,
        }

    def get_worst_filters(self, min_observations: int = 5) -> list[dict]:
        """Identify filters that are rejecting too many winning trades.

        Returns reject reasons sorted by missed opportunity rate.
        """
        analysis = self.get_analysis()
        worst = []

        for reason, data in analysis.get("by_reason", {}).items():
            if data["total"] >= min_observations:
                miss_rate = data["would_won"] / data["total"] if data["total"] > 0 else 0
                if miss_rate > 0.5:  # More than half the skipped trades would have won
                    worst.append({
                        "reason": reason,
                        "miss_rate": miss_rate,
                        "missed_trades": data["would_won"],
                        "total": data["total"],
                        "missed_pnl": data["pnl"],
                    })

        return sorted(worst, key=lambda x: x["miss_rate"], reverse=True)

    @property
    def pending_count(self) -> int:
        """Number of skipped trades awaiting outcome evaluation."""
        return sum(1 for t in self._skipped if not t.outcome_checked)

    @property
    def total_count(self) -> int:
        return len(self._skipped)

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for t in self._skipped:
            data.append({
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "strategy": t.strategy,
                "direction": t.direction,
                "confidence": t.confidence,
                "entry_price": t.entry_price,
                "reject_reason": t.reject_reason,
                "exit_price": t.exit_price,
                "hypothetical_pnl": t.hypothetical_pnl,
                "would_have_won": t.would_have_won,
                "outcome_checked": t.outcome_checked,
            })
        try:
            STATE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.warning("counterfactual_save_failed", error=str(e))

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
            for item in data:
                self._skipped.append(SkippedTrade(
                    timestamp=item["timestamp"],
                    symbol=item["symbol"],
                    strategy=item["strategy"],
                    direction=item["direction"],
                    confidence=item["confidence"],
                    entry_price=item["entry_price"],
                    reject_reason=item["reject_reason"],
                    exit_price=item.get("exit_price"),
                    hypothetical_pnl=item.get("hypothetical_pnl"),
                    would_have_won=item.get("would_have_won"),
                    outcome_checked=item.get("outcome_checked", False),
                ))
            log.info("counterfactual_state_loaded", count=len(self._skipped))
        except Exception as e:
            log.warning("counterfactual_load_failed", error=str(e))

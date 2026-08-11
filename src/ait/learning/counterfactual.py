"""Counterfactual skip log — records signals the bot generated but rejected.

R12-C simplification (2026-07-13): the evaluation/analysis half
(evaluate_outcomes / get_analysis / get_worst_filters) was deleted. Its
outputs were twice ruled garbage — R7 found a units bug, R9 found the
"filter accuracy" indistinguishable from base rate — and the crude
underlying-moved-2% "win" model was never option-structure P&L. What
remains is the durable record of taken-vs-vetoed signals (record_skip +
JSON storage), which the post-market report and future analyses can read.
The deleted logic is in git history (this file pre-R12) if ever wanted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ait.utils.logging import get_logger

log = get_logger("learning.counterfactual")

STATE_FILE = Path("data/counterfactual_log.json")


@dataclass
class SkippedTrade:
    """A trade signal that was generated but not executed.

    The outcome fields (exit_price / hypothetical_pnl / would_have_won /
    outcome_checked) are retained for on-disk compatibility with existing
    data/counterfactual_log.json rows written before R12-C; nothing fills
    them in anymore.
    """

    timestamp: str
    symbol: str
    strategy: str
    direction: str
    confidence: float
    entry_price: float
    reject_reason: str
    exit_price: float | None = None
    hypothetical_pnl: float | None = None
    would_have_won: bool | None = None
    outcome_checked: bool = False


class CounterfactualTracker:
    """Records skipped trades: call record_skip() when a signal is rejected."""

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
        # R5 audit: _scan_symbol fires every ~5 min — persistently-rejected
        # symbols generated hundreds of duplicate records/day, evicting older
        # records before the 24h min-age let them be scored (evaluation
        # starved on every active day). One record per (symbol, strategy,
        # reason) per day.
        #
        # R17: originally gated on `not outcome_checked`, so once
        # evaluate_outcomes() (the only code that ever flipped that flag)
        # was deleted in R12-C, this became "one record per combo, ever" —
        # a persistently-rejected symbol was recorded once and then silently
        # dropped forever. Gate on same-day instead: still floods nothing,
        # but the log keeps recording across days.
        today = record.timestamp[:10]
        for _t in self._skipped:
            if (_t.timestamp[:10] == today and _t.symbol == symbol
                    and _t.strategy == strategy
                    and _t.reject_reason == reject_reason):
                return
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

    # R12-C: evaluate_outcomes / get_analysis / get_worst_filters deleted —
    # outputs twice ruled misleading (units bug R7; base-rate-
    # indistinguishable R9). record_skip rows above are the kept record.

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

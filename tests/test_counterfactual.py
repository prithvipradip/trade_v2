"""Tests for the counterfactual skip log.

R12-C: the evaluation/analysis half (evaluate_outcomes / get_analysis /
get_worst_filters) was deleted — outputs twice ruled misleading (units bug
R7; base-rate-indistinguishable R9). These tests cover what remains:
record_skip, dedup, history trim, counters, and JSON persistence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ait.learning.counterfactual import CounterfactualTracker, SkippedTrade


class TestSkippedTrade:
    """Test the SkippedTrade data class."""

    def test_defaults(self) -> None:
        t = SkippedTrade(
            timestamp="2025-01-01T00:00:00",
            symbol="SPY",
            strategy="long_call",
            direction="bullish",
            confidence=0.75,
            entry_price=450.0,
            reject_reason="low_confidence",
        )
        assert t.outcome_checked is False
        assert t.hypothetical_pnl is None
        assert t.would_have_won is None


class TestCounterfactualTracker:
    """Test the skip-recording system."""

    @pytest.fixture
    def tracker(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CounterfactualTracker:
        monkeypatch.setattr(
            "ait.learning.counterfactual.STATE_FILE",
            tmp_path / "counterfactual.json",
        )
        return CounterfactualTracker()

    def test_record_skip(self, tracker: CounterfactualTracker) -> None:
        tracker.record_skip(
            symbol="SPY",
            strategy="long_call",
            direction="bullish",
            confidence=0.65,
            entry_price=450.0,
            reject_reason="low_confidence",
        )
        assert tracker.total_count == 1
        assert tracker.pending_count == 1

    def test_duplicate_pending_skip_not_recorded(self, tracker: CounterfactualTracker) -> None:
        """R5 audit: one unevaluated record per (symbol, strategy, reason) —
        a persistently-rejected symbol scanning every ~5 min must not flood
        the log with duplicates."""
        for _ in range(5):
            tracker.record_skip("SPY", "iron_condor", "neutral", 0.7, 450.0, "risk_limit")
        assert tracker.total_count == 1

        # A different reject_reason for the same symbol/strategy IS a new record.
        tracker.record_skip("SPY", "iron_condor", "neutral", 0.7, 450.0, "meta_label_reject")
        assert tracker.total_count == 2

    def test_max_history(self, tracker: CounterfactualTracker) -> None:
        tracker._max_history = 5
        for i in range(10):
            tracker.record_skip(f"SYM{i}", "long_call", "bullish", 0.7, 100.0, "test")
        assert tracker.total_count == 5

    def test_persistence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        state_file = tmp_path / "counterfactual.json"
        monkeypatch.setattr("ait.learning.counterfactual.STATE_FILE", state_file)

        t1 = CounterfactualTracker()
        t1.record_skip("SPY", "long_call", "bullish", 0.7, 450.0, "test")

        t2 = CounterfactualTracker()
        assert t2.total_count == 1

    def test_loads_pre_r12_rows_with_outcome_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rows written before R12-C carry evaluated-outcome fields; loading
        them must not crash and pending_count must respect outcome_checked."""
        import json

        state_file = tmp_path / "counterfactual.json"
        monkeypatch.setattr("ait.learning.counterfactual.STATE_FILE", state_file)
        state_file.write_text(json.dumps([
            {
                "timestamp": "2026-07-01T10:00:00", "symbol": "SPY",
                "strategy": "iron_condor", "direction": "neutral",
                "confidence": 0.7, "entry_price": 450.0,
                "reject_reason": "risk_limit",
                "exit_price": 452.0, "hypothetical_pnl": 0.0,
                "would_have_won": True, "outcome_checked": True,
            },
            {
                "timestamp": "2026-07-12T10:00:00", "symbol": "QQQ",
                "strategy": "iron_condor", "direction": "neutral",
                "confidence": 0.8, "entry_price": 380.0,
                "reject_reason": "low_confidence",
            },
        ]))

        t = CounterfactualTracker()
        assert t.total_count == 2
        assert t.pending_count == 1  # only the unevaluated legacy row

    def test_analysis_methods_removed(self, tracker: CounterfactualTracker) -> None:
        """The R12-C deletion is intentional; resurrection needs re-review."""
        assert not hasattr(tracker, "evaluate_outcomes")
        assert not hasattr(tracker, "get_analysis")
        assert not hasattr(tracker, "get_worst_filters")

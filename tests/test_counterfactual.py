"""Tests for counterfactual analysis tracker."""

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
    """Test the counterfactual tracking system."""

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

    def test_evaluate_bullish_win(self, tracker: CounterfactualTracker) -> None:
        tracker.record_skip(
            symbol="SPY", strategy="long_call", direction="bullish",
            confidence=0.70, entry_price=450.0, reject_reason="risk_limit",
        )
        # Price went up significantly
        evaluated = tracker.evaluate_outcomes({"SPY": 470.0})
        assert evaluated == 1
        assert tracker.pending_count == 0

        analysis = tracker.get_analysis()
        assert analysis["evaluated"] == 1
        assert analysis["missed_opportunities"] == 1  # We missed a winner

    def test_evaluate_bullish_loss(self, tracker: CounterfactualTracker) -> None:
        tracker.record_skip(
            symbol="QQQ", strategy="long_call", direction="bullish",
            confidence=0.60, entry_price=380.0, reject_reason="meta_label_reject",
        )
        # Price went down — our filter was correct
        evaluated = tracker.evaluate_outcomes({"QQQ": 370.0})
        assert evaluated == 1

        analysis = tracker.get_analysis()
        assert analysis["correct_skips"] == 1
        assert analysis["filter_accuracy"] == 1.0

    def test_evaluate_bearish(self, tracker: CounterfactualTracker) -> None:
        tracker.record_skip(
            symbol="SPY", strategy="long_put", direction="bearish",
            confidence=0.75, entry_price=450.0, reject_reason="low_confidence",
        )
        # Price went down — bearish was correct, we missed it
        evaluated = tracker.evaluate_outcomes({"SPY": 430.0})
        assert evaluated == 1

        analysis = tracker.get_analysis()
        assert analysis["missed_opportunities"] == 1

    def test_evaluate_skips_unknown_symbols(self, tracker: CounterfactualTracker) -> None:
        tracker.record_skip(
            symbol="AAPL", strategy="long_call", direction="bullish",
            confidence=0.80, entry_price=170.0, reject_reason="risk_limit",
        )
        # No price available for AAPL
        evaluated = tracker.evaluate_outcomes({"SPY": 450.0})
        assert evaluated == 0
        assert tracker.pending_count == 1

    def test_analysis_by_reason(self, tracker: CounterfactualTracker) -> None:
        # Record skips with different reasons
        tracker.record_skip("SPY", "long_call", "bullish", 0.65, 450.0, "low_confidence")
        tracker.record_skip("QQQ", "long_call", "bullish", 0.70, 380.0, "meta_label_reject")
        tracker.record_skip("AAPL", "iron_condor", "neutral", 0.80, 170.0, "low_confidence")

        tracker.evaluate_outcomes({"SPY": 470.0, "QQQ": 370.0, "AAPL": 170.5})

        analysis = tracker.get_analysis()
        assert "low_confidence" in analysis["by_reason"]
        assert "meta_label_reject" in analysis["by_reason"]

    def test_analysis_by_strategy(self, tracker: CounterfactualTracker) -> None:
        tracker.record_skip("SPY", "long_call", "bullish", 0.65, 450.0, "risk_limit")
        tracker.record_skip("QQQ", "iron_condor", "neutral", 0.80, 380.0, "risk_limit")

        tracker.evaluate_outcomes({"SPY": 470.0, "QQQ": 380.5})

        analysis = tracker.get_analysis()
        assert "long_call" in analysis["by_strategy"]
        assert "iron_condor" in analysis["by_strategy"]

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

    def test_get_worst_filters(self, tracker: CounterfactualTracker) -> None:
        # Record many skips with a bad filter (most would have won)
        for i in range(10):
            tracker.record_skip(f"S{i}", "long_call", "bullish", 0.7, 100.0, "bad_filter")

        # Evaluate: all prices went up (all would have won)
        prices = {f"S{i}": 110.0 for i in range(10)}
        tracker.evaluate_outcomes(prices)

        worst = tracker.get_worst_filters(min_observations=5)
        assert len(worst) >= 1
        assert worst[0]["reason"] == "bad_filter"
        assert worst[0]["miss_rate"] > 0.5

    def test_empty_analysis(self, tracker: CounterfactualTracker) -> None:
        analysis = tracker.get_analysis()
        assert analysis["total_skipped"] == 0
        assert analysis["evaluated"] == 0
        assert analysis["filter_accuracy"] == 0.0

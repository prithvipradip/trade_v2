"""Tests for Thompson sampling strategy selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ait.strategies.thompson import ThompsonSampler, StrategyArm


class TestStrategyArm:
    """Test the individual arm (Beta distribution)."""

    def test_initial_prior(self) -> None:
        arm = StrategyArm(name="test")
        assert arm.alpha == 1.0
        assert arm.beta == 1.0
        assert arm.observations == 0
        assert arm.empirical_win_rate == 0.5

    def test_win_updates(self) -> None:
        arm = StrategyArm(name="test", wins=7, losses=3)
        assert arm.alpha == 8.0
        assert arm.beta == 4.0
        assert arm.observations == 10
        assert arm.empirical_win_rate == pytest.approx(0.7)

    def test_sample_in_range(self) -> None:
        arm = StrategyArm(name="test", wins=10, losses=5)
        for _ in range(100):
            s = arm.sample()
            assert 0 <= s <= 1


class TestThompsonSampler:
    """Test the Thompson sampling strategy selector."""

    @pytest.fixture
    def sampler(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ThompsonSampler:
        """Create sampler with tmp state file."""
        monkeypatch.setattr(
            "ait.strategies.thompson.STATE_FILE",
            tmp_path / "thompson_state.json",
        )
        return ThompsonSampler()

    def test_rank_empty(self, sampler: ThompsonSampler) -> None:
        assert sampler.rank_strategies([]) == []

    def test_select_best_from_candidates(self, sampler: ThompsonSampler) -> None:
        result = sampler.select_best(["iron_condor", "long_call", "bull_call_spread"])
        assert result in ("iron_condor", "long_call", "bull_call_spread")

    def test_rank_returns_all_candidates(self, sampler: ThompsonSampler) -> None:
        candidates = ["a", "b", "c"]
        ranked = sampler.rank_strategies(candidates)
        assert set(ranked) == set(candidates)
        assert len(ranked) == 3

    def test_record_outcome_updates_arm(self, sampler: ThompsonSampler) -> None:
        sampler.record_outcome("iron_condor", won=True, pnl=100.0)
        sampler.record_outcome("iron_condor", won=True, pnl=50.0)
        sampler.record_outcome("iron_condor", won=False, pnl=-75.0)

        stats = sampler.get_stats()
        ic = next(s for s in stats if s["strategy"] == "iron_condor")
        assert ic["wins"] == 2
        assert ic["losses"] == 1
        assert ic["total_pnl"] == pytest.approx(75.0)
        assert ic["observations"] == 3

    def test_winner_gets_selected_more(self, sampler: ThompsonSampler) -> None:
        # Give iron_condor a strong track record
        for _ in range(50):
            sampler.record_outcome("iron_condor", won=True, pnl=10.0)
        for _ in range(50):
            sampler.record_outcome("long_call", won=False, pnl=-10.0)

        # Over many selections, iron_condor should win most
        wins = {"iron_condor": 0, "long_call": 0}
        for _ in range(200):
            best = sampler.select_best(["iron_condor", "long_call"])
            wins[best] += 1

        assert wins["iron_condor"] > wins["long_call"]

    def test_decay_reduces_counts(self, sampler: ThompsonSampler) -> None:
        for _ in range(100):
            sampler.record_outcome("test_strat", won=True, pnl=10.0)

        stats_before = sampler.get_stats()
        wins_before = next(s for s in stats_before if s["strategy"] == "test_strat")["wins"]

        sampler.apply_decay()

        stats_after = sampler.get_stats()
        wins_after = next(s for s in stats_after if s["strategy"] == "test_strat")["wins"]

        assert wins_after < wins_before

    def test_persistence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        state_file = tmp_path / "thompson_state.json"
        monkeypatch.setattr("ait.strategies.thompson.STATE_FILE", state_file)

        # Create, record, and destroy
        s1 = ThompsonSampler()
        s1.record_outcome("bull_call_spread", won=True, pnl=200.0)
        s1.record_outcome("bull_call_spread", won=False, pnl=-50.0)

        # New sampler should load state
        s2 = ThompsonSampler()
        stats = s2.get_stats()
        bcs = next(s for s in stats if s["strategy"] == "bull_call_spread")
        assert bcs["wins"] == 1
        assert bcs["losses"] == 1

    def test_selection_probabilities(self, sampler: ThompsonSampler) -> None:
        # With equal priors, probabilities should be roughly equal
        probs = sampler.get_selection_probabilities(["a", "b", "c"], n_samples=1000)
        assert len(probs) == 3
        for p in probs.values():
            assert 0.15 < p < 0.55  # Roughly equal with some variance

    def test_select_best_none_on_empty(self, sampler: ThompsonSampler) -> None:
        assert sampler.select_best([]) is None

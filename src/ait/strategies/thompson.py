"""Thompson sampling for strategy selection — Bayesian exploration/exploitation.

Instead of always picking the historically best strategy, Thompson sampling
maintains a Beta distribution for each strategy and samples from it.
This naturally balances:
- Exploitation: strategies with high win rates get picked more often
- Exploration: strategies with few observations still get tried

Each strategy has (alpha, beta) parameters:
- alpha = 1 + wins  (success count)
- beta  = 1 + losses (failure count)
- Sample from Beta(alpha, beta) to get estimated win probability
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ait.utils.logging import get_logger

log = get_logger("strategies.thompson")

STATE_FILE = Path("data/thompson_state.json")


@dataclass
class StrategyArm:
    """One arm of the multi-armed bandit."""

    name: str
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0

    @property
    def alpha(self) -> float:
        return 1.0 + self.wins

    @property
    def beta(self) -> float:
        return 1.0 + self.losses

    @property
    def observations(self) -> int:
        return self.wins + self.losses

    @property
    def empirical_win_rate(self) -> float:
        if self.observations == 0:
            return 0.5
        return self.wins / self.observations

    def sample(self) -> float:
        """Draw from Beta(alpha, beta)."""
        return random.betavariate(self.alpha, self.beta)


class ThompsonSampler:
    """Thompson sampling for strategy selection.

    Maintains a Beta distribution per strategy. On each selection:
    1. Sample from each strategy's Beta distribution
    2. Return strategies ranked by sampled value
    3. After trade outcome, update the winning/losing strategy

    Supports decay: older observations gradually lose weight to
    adapt to changing market conditions.
    """

    def __init__(self, decay_factor: float = 0.995) -> None:
        self._arms: dict[str, StrategyArm] = {}
        self._decay_factor = decay_factor
        self._load_state()

    def _ensure_arm(self, strategy: str) -> StrategyArm:
        if strategy not in self._arms:
            self._arms[strategy] = StrategyArm(name=strategy)
        return self._arms[strategy]

    def rank_strategies(self, candidates: list[str]) -> list[str]:
        """Rank candidate strategies by Thompson sampling.

        Returns strategies sorted best-to-worst by sampled value.
        Strategies with no history get a uniform prior (50/50).
        """
        if not candidates:
            return []

        samples = []
        for name in candidates:
            arm = self._ensure_arm(name)
            sampled_value = arm.sample()
            samples.append((name, sampled_value))

        samples.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in samples]

    def select_best(self, candidates: list[str]) -> str | None:
        """Select the single best strategy via Thompson sampling."""
        ranked = self.rank_strategies(candidates)
        return ranked[0] if ranked else None

    def record_outcome(self, strategy: str, won: bool, pnl: float = 0.0) -> None:
        """Record trade outcome for a strategy."""
        arm = self._ensure_arm(strategy)
        if won:
            arm.wins += 1
        else:
            arm.losses += 1
        arm.total_pnl += pnl

        log.info(
            "thompson_update",
            strategy=strategy,
            won=won,
            pnl=f"{pnl:.2f}",
            win_rate=f"{arm.empirical_win_rate:.2%}",
            observations=arm.observations,
        )

        self._save_state()

    def apply_decay(self) -> None:
        """Apply decay to all arms — makes sampler adapt to recent performance.

        Call this periodically (e.g., daily) to gradually forget old outcomes.
        """
        for arm in self._arms.values():
            arm.wins = max(0, int(arm.wins * self._decay_factor))
            arm.losses = max(0, int(arm.losses * self._decay_factor))

        log.info("thompson_decay_applied", arms=len(self._arms))
        self._save_state()

    def get_stats(self) -> list[dict]:
        """Get current state of all arms for display."""
        stats = []
        for arm in sorted(self._arms.values(), key=lambda a: a.empirical_win_rate, reverse=True):
            stats.append({
                "strategy": arm.name,
                "wins": arm.wins,
                "losses": arm.losses,
                "observations": arm.observations,
                "win_rate": arm.empirical_win_rate,
                "total_pnl": arm.total_pnl,
                "alpha": arm.alpha,
                "beta": arm.beta,
            })
        return stats

    def get_selection_probabilities(self, candidates: list[str], n_samples: int = 1000) -> dict[str, float]:
        """Estimate probability each strategy would be selected.

        Runs n_samples Monte Carlo simulations to estimate selection frequency.
        """
        if not candidates:
            return {}

        counts = {name: 0 for name in candidates}
        for _ in range(n_samples):
            best = None
            best_val = -1.0
            for name in candidates:
                arm = self._ensure_arm(name)
                val = arm.sample()
                if val > best_val:
                    best_val = val
                    best = name
            if best:
                counts[best] += 1

        return {name: count / n_samples for name, count in counts.items()}

    def _save_state(self) -> None:
        """Persist arm state to disk."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for name, arm in self._arms.items():
            data[name] = {
                "wins": arm.wins,
                "losses": arm.losses,
                "total_pnl": arm.total_pnl,
            }
        try:
            STATE_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.warning("thompson_save_failed", error=str(e))

    def _load_state(self) -> None:
        """Load arm state from disk."""
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
            for name, vals in data.items():
                arm = self._ensure_arm(name)
                arm.wins = vals.get("wins", 0)
                arm.losses = vals.get("losses", 0)
                arm.total_pnl = vals.get("total_pnl", 0.0)
            log.info("thompson_state_loaded", arms=len(self._arms))
        except Exception as e:
            log.warning("thompson_load_failed", error=str(e))

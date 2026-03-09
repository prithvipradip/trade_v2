"""Strategy adaptor — applies learnings to adjust bot behavior.

Takes insights from TradeAnalyzer and translates them into concrete
configuration changes:
- Disable/enable strategies
- Adjust confidence thresholds
- Modify position sizing multipliers
- Remove underperforming symbols
- Tune stop loss / take profit levels

All changes are bounded by safety limits to prevent runaway adaptation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ait.bot.state import StateManager
from ait.learning.analyzer import TradeInsight
from ait.utils.logging import get_logger

log = get_logger("learning.adaptor")


@dataclass
class Adaptation:
    """A single configuration change applied by the learning system."""

    timestamp: str
    parameter: str  # What was changed
    old_value: str  # Previous value
    new_value: str  # New value
    reason: str  # Why it was changed
    insight_confidence: float  # How confident the insight was


@dataclass
class AdaptationLimits:
    """Safety bounds for self-learning adaptations."""

    # Confidence threshold bounds
    min_confidence_floor: float = 0.50  # Never go below 50%
    min_confidence_ceiling: float = 0.90  # Never go above 90%

    # Position sizing multiplier bounds
    min_strategy_multiplier: float = 0.3  # Never reduce strategy below 30%
    max_strategy_multiplier: float = 2.0  # Never boost strategy above 200%

    # Stop loss bounds
    min_stop_loss_pct: float = 0.25  # Never tighter than 25% loss
    max_stop_loss_pct: float = 0.75  # Never wider than 75% loss

    # Minimum insight confidence to act on
    min_insight_confidence: float = 0.6

    # Maximum adaptations per cycle
    max_adaptations_per_cycle: int = 3


class StrategyAdaptor:
    """Applies trade insights to adapt bot configuration."""

    STATE_KEY = "learning_adaptations"

    def __init__(
        self,
        state: StateManager,
        limits: AdaptationLimits | None = None,
    ) -> None:
        self._state = state
        self._limits = limits or AdaptationLimits()

        # Current overrides (loaded from state)
        self._strategy_multipliers: dict[str, float] = {}
        self._disabled_strategies: set[str] = set()
        self._removed_symbols: set[str] = set()
        self._confidence_override: float | None = None
        self._stop_loss_override: float | None = None

        self._load_state()

    def apply_insights(self, insights: list[TradeInsight]) -> list[Adaptation]:
        """Apply trade insights as configuration adaptations.

        Returns list of adaptations actually applied.
        """
        adaptations: list[Adaptation] = []
        applied = 0

        for insight in insights:
            if applied >= self._limits.max_adaptations_per_cycle:
                break
            if insight.confidence < self._limits.min_insight_confidence:
                continue

            adaptation = self._process_insight(insight)
            if adaptation:
                adaptations.append(adaptation)
                applied += 1
                log.info(
                    "adaptation_applied",
                    parameter=adaptation.parameter,
                    old=adaptation.old_value,
                    new=adaptation.new_value,
                    reason=adaptation.reason,
                )

        if adaptations:
            self._save_state()

        return adaptations

    def get_strategy_multiplier(self, strategy: str) -> float:
        """Get the current sizing multiplier for a strategy.

        Returns 1.0 for default, >1 for boosted, <1 for reduced, 0 for disabled.
        """
        if strategy in self._disabled_strategies:
            return 0.0
        return self._strategy_multipliers.get(strategy, 1.0)

    def is_strategy_enabled(self, strategy: str) -> bool:
        """Check if a strategy is currently enabled."""
        return strategy not in self._disabled_strategies

    def is_symbol_allowed(self, symbol: str) -> bool:
        """Check if a symbol is allowed (not removed by learning)."""
        return symbol not in self._removed_symbols

    def get_confidence_override(self) -> float | None:
        """Get the learned minimum confidence threshold, or None for default."""
        return self._confidence_override

    def get_stop_loss_override(self) -> float | None:
        """Get the learned stop loss percentage, or None for default."""
        return self._stop_loss_override

    def get_all_overrides(self) -> dict:
        """Get all current learning overrides for logging/dashboard."""
        return {
            "strategy_multipliers": dict(self._strategy_multipliers),
            "disabled_strategies": list(self._disabled_strategies),
            "removed_symbols": list(self._removed_symbols),
            "confidence_override": self._confidence_override,
            "stop_loss_override": self._stop_loss_override,
        }

    def reset(self) -> None:
        """Reset all adaptations to defaults."""
        self._strategy_multipliers.clear()
        self._disabled_strategies.clear()
        self._removed_symbols.clear()
        self._confidence_override = None
        self._stop_loss_override = None
        self._save_state()
        log.info("learning_adaptations_reset")

    # --- Insight processors ---

    def _process_insight(self, insight: TradeInsight) -> Adaptation | None:
        """Convert a single insight into an adaptation."""
        action = insight.action

        if action.startswith("disable_"):
            return self._disable_strategy(action[8:], insight)
        elif action.startswith("boost_"):
            return self._boost_strategy(action[6:], insight)
        elif action.startswith("raise_min_confidence_to_"):
            prefix = "raise_min_confidence_to_"
            return self._raise_confidence(float(action[len(prefix):]), insight)
        elif action.startswith("remove_symbol_"):
            return self._remove_symbol(action[14:], insight)
        elif action == "tighten_stop_loss":
            return self._tighten_stop_loss(insight)
        elif action.startswith("reduce_trading_in_"):
            # Regime-based: reduce position sizes during bad regimes
            return self._reduce_regime_exposure(action[18:], insight)

        return None

    def _disable_strategy(self, strategy: str, insight: TradeInsight) -> Adaptation | None:
        """Disable a consistently losing strategy."""
        if strategy in self._disabled_strategies:
            return None

        self._disabled_strategies.add(strategy)
        return Adaptation(
            timestamp=datetime.now().isoformat(),
            parameter=f"strategy.{strategy}.enabled",
            old_value="true",
            new_value="false",
            reason=insight.insight,
            insight_confidence=insight.confidence,
        )

    def _boost_strategy(self, strategy: str, insight: TradeInsight) -> Adaptation | None:
        """Increase sizing multiplier for a winning strategy."""
        current = self._strategy_multipliers.get(strategy, 1.0)
        new_mult = min(current * 1.25, self._limits.max_strategy_multiplier)

        if abs(new_mult - current) < 0.05:
            return None

        self._strategy_multipliers[strategy] = new_mult
        return Adaptation(
            timestamp=datetime.now().isoformat(),
            parameter=f"strategy.{strategy}.multiplier",
            old_value=f"{current:.2f}",
            new_value=f"{new_mult:.2f}",
            reason=insight.insight,
            insight_confidence=insight.confidence,
        )

    def _raise_confidence(self, new_min: float, insight: TradeInsight) -> Adaptation | None:
        """Raise the minimum confidence threshold."""
        new_min = max(self._limits.min_confidence_floor, min(new_min, self._limits.min_confidence_ceiling))
        current = self._confidence_override or 0.65

        if new_min <= current:
            return None

        self._confidence_override = new_min
        return Adaptation(
            timestamp=datetime.now().isoformat(),
            parameter="risk.min_confidence",
            old_value=f"{current:.2f}",
            new_value=f"{new_min:.2f}",
            reason=insight.insight,
            insight_confidence=insight.confidence,
        )

    def _remove_symbol(self, symbol: str, insight: TradeInsight) -> Adaptation | None:
        """Remove a consistently losing symbol from the universe."""
        if symbol in self._removed_symbols:
            return None

        self._removed_symbols.add(symbol)
        return Adaptation(
            timestamp=datetime.now().isoformat(),
            parameter=f"trading.universe.{symbol}",
            old_value="included",
            new_value="excluded",
            reason=insight.insight,
            insight_confidence=insight.confidence,
        )

    def _tighten_stop_loss(self, insight: TradeInsight) -> Adaptation | None:
        """Tighten stop loss based on hold time analysis."""
        current = self._stop_loss_override or 0.50
        new_sl = max(current * 0.85, self._limits.min_stop_loss_pct)

        if abs(new_sl - current) < 0.02:
            return None

        self._stop_loss_override = new_sl
        return Adaptation(
            timestamp=datetime.now().isoformat(),
            parameter="portfolio.stop_loss_pct",
            old_value=f"{current:.2f}",
            new_value=f"{new_sl:.2f}",
            reason=insight.insight,
            insight_confidence=insight.confidence,
        )

    def _reduce_regime_exposure(self, regime: str, insight: TradeInsight) -> Adaptation | None:
        """Log regime-based learning (applied via regime detector weight)."""
        # This is tracked but applied via the regime detector's integration
        log.info("regime_learning", regime=regime, insight=insight.insight)
        return None

    # --- State persistence ---

    def _save_state(self) -> None:
        """Persist learning state to SQLite."""
        data = {
            "strategy_multipliers": self._strategy_multipliers,
            "disabled_strategies": list(self._disabled_strategies),
            "removed_symbols": list(self._removed_symbols),
            "confidence_override": self._confidence_override,
            "stop_loss_override": self._stop_loss_override,
        }
        self._state.set_state(self.STATE_KEY, json.dumps(data))

    def _load_state(self) -> None:
        """Load learning state from SQLite."""
        raw = self._state.get_state(self.STATE_KEY)
        if not raw:
            return

        try:
            data = json.loads(raw)
            self._strategy_multipliers = data.get("strategy_multipliers", {})
            self._disabled_strategies = set(data.get("disabled_strategies", []))
            self._removed_symbols = set(data.get("removed_symbols", []))
            self._confidence_override = data.get("confidence_override")
            self._stop_loss_override = data.get("stop_loss_override")
            log.info("learning_state_loaded", overrides=self.get_all_overrides())
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("learning_state_load_failed", error=str(e))

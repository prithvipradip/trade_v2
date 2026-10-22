"""In-backtest self-learning adapter.

After each walk-forward window, analyzes the trades and generates
configuration adaptations for the next window. This is the "self-learning"
loop for backtesting — no SQLite needed, works entirely in-memory.

Learns:
- Which strategies to boost (high win rate + profitable)
- Which strategies to reduce/disable (losing consistently)
- Optimal confidence threshold (raise if low-confidence trades lose)
- Symbol-specific problems (disable bad symbols)
- Iron condor wing adjustments (if they keep getting blown through)

All changes are bounded by safety limits to prevent runaway adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ait.config.settings import LearnerConfig
from ait.utils.logging import get_logger

log = get_logger("backtesting.learner")

# W88b: every threshold the learner used to hold as a module constant or an inline literal (confidence floor /
# cap, multiplier bounds, trades needed, win-rate cutoffs, step sizes, protected strategies) is now a field of
# ``backtest.learner`` (settings.LearnerConfig), listed in full in the research overlay file. The values below are
# only what a learner with NO state reports as its starting confidence.
_STATE_DEFAULT_CONFIDENCE = 0.65


def _default_config() -> LearnerConfig:
    """``backtest.learner`` from the loaded settings (the frozen research view in a research process), else the
    schema defaults."""
    try:
        from ait.config.settings import load_settings
        return load_settings().backtest.learner
    except Exception:  # noqa: BLE001 - no config.yaml -> the schema defaults
        return LearnerConfig()


@dataclass
class LearnerState:
    """Accumulated learning state across walk-forward windows."""

    # Strategy performance: {strategy: {"wins": int, "losses": int, "pnl": float}}
    strategy_stats: dict[str, dict] = field(default_factory=dict)

    # Symbol performance: {symbol: {"wins": int, "losses": int, "pnl": float}}
    symbol_stats: dict[str, dict] = field(default_factory=dict)

    # Confidence band performance: {band: {"wins": int, "losses": int, "pnl": float}}
    # bands: "low" (0.55-0.65), "medium" (0.65-0.75), "high" (0.75+)
    confidence_stats: dict[str, dict] = field(default_factory=dict)

    # Current adaptations
    strategy_multipliers: dict[str, float] = field(default_factory=dict)
    disabled_strategies: set[str] = field(default_factory=set)
    disabled_symbols: set[str] = field(default_factory=set)
    min_confidence_override: float = _STATE_DEFAULT_CONFIDENCE

    # History of adaptations for transparency
    adaptation_log: list[dict] = field(default_factory=list)

    # Window count
    windows_processed: int = 0


class BacktestLearner:
    """Self-learning adapter for walk-forward backtesting.

    Tracks performance across windows and adapts configuration
    for the next window. Think of it as the bot learning from its mistakes.

    Usage:
        learner = BacktestLearner()
        for window in windows:
            ...run backtest with learner.get_config()...
            learner.process_window(trades)
    """

    def __init__(self, base_confidence: float = _STATE_DEFAULT_CONFIDENCE,
                 config: LearnerConfig | None = None) -> None:
        self._cfg = config if config is not None else _default_config()
        self._state = LearnerState(min_confidence_override=base_confidence)

    def process_window(self, trades: list[dict], window_id: int) -> dict:
        """Ingest trades from a completed window and update learning state.

        Returns a summary of what was learned.
        """
        if not trades:
            return {}

        self._state.windows_processed += 1
        self._accumulate_stats(trades)

        # Only adapt after enough data
        if self._state.windows_processed < self._cfg.windows_before_adapting:
            log.info("learner_accumulating", windows=self._state.windows_processed)
            return {}

        adaptations = []
        adaptations.extend(self._adapt_strategies())
        adaptations.extend(self._adapt_confidence())
        adaptations.extend(self._adapt_symbols())

        if adaptations:
            self._state.adaptation_log.extend(adaptations)
            log.info(
                "learner_adapted",
                window=window_id,
                adaptations=len(adaptations),
                changes=[a["change"] for a in adaptations],
            )

        return {
            "window_id": window_id,
            "adaptations": adaptations,
            "current_config": self.get_config(),
        }

    def get_config(self) -> dict:
        """Get the current adapted configuration to pass to the next Backtester."""
        return {
            "min_confidence": self._state.min_confidence_override,
            "strategy_multipliers": dict(self._state.strategy_multipliers),
            "disabled_strategies": set(self._state.disabled_strategies),
            "disabled_symbols": set(self._state.disabled_symbols),
        }

    def get_strategy_multiplier(self, strategy: str) -> float:
        """Get sizing multiplier for a strategy (1.0 = default)."""
        if strategy in self._state.disabled_strategies:
            return 0.0
        return self._state.strategy_multipliers.get(strategy, 1.0)

    def is_strategy_enabled(self, strategy: str) -> bool:
        return strategy not in self._state.disabled_strategies

    def is_symbol_allowed(self, symbol: str) -> bool:
        return symbol not in self._state.disabled_symbols

    def summary(self) -> str:
        """Human-readable summary of current learning state."""
        lines = ["  LEARNER STATE:"]

        # Strategy stats
        for strat, stats in sorted(self._state.strategy_stats.items()):
            total = stats["wins"] + stats["losses"]
            if total < 2:
                continue
            wr = stats["wins"] / total
            mult = self._state.strategy_multipliers.get(strat, 1.0)
            disabled = " [DISABLED]" if strat in self._state.disabled_strategies else ""
            lines.append(
                f"    {strat:25s} | {total:3d} trades | {wr:.0%} WR | "
                f"${stats['pnl']:,.0f} | mult={mult:.2f}{disabled}"
            )

        if self._state.min_confidence_override != _STATE_DEFAULT_CONFIDENCE:
            lines.append(f"    Min confidence adjusted to: {self._state.min_confidence_override:.2f}")

        if self._state.disabled_symbols:
            lines.append(f"    Disabled symbols: {', '.join(self._state.disabled_symbols)}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _accumulate_stats(self, trades: list[dict]) -> None:
        """Update running statistics from a list of trades."""
        for t in trades:
            pnl = t.get("pnl", 0)
            strategy = t.get("strategy", "unknown")
            symbol = t.get("symbol", "unknown")
            confidence = t.get("entry_confidence", self._cfg.default_entry_confidence)

            # Strategy stats
            s = self._state.strategy_stats.setdefault(
                strategy, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
            )
            s["trades"] += 1
            s["pnl"] += pnl
            if pnl > 0:
                s["wins"] += 1
            else:
                s["losses"] += 1

            # Symbol stats
            sym = self._state.symbol_stats.setdefault(
                symbol, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
            )
            sym["trades"] += 1
            sym["pnl"] += pnl
            if pnl > 0:
                sym["wins"] += 1
            else:
                sym["losses"] += 1

            # Confidence band stats
            band = self._confidence_band(confidence)
            cb = self._state.confidence_stats.setdefault(
                band, {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0}
            )
            cb["trades"] += 1
            cb["pnl"] += pnl
            if pnl > 0:
                cb["wins"] += 1
            else:
                cb["losses"] += 1

    def _adapt_strategies(self) -> list[dict]:
        """Adjust strategy multipliers based on cumulative performance."""
        adaptations = []
        c = self._cfg

        for strategy, stats in self._state.strategy_stats.items():
            total = stats["wins"] + stats["losses"]
            if total < c.min_trades_to_learn:
                continue

            win_rate = stats["wins"] / total
            avg_pnl = stats["pnl"] / total
            current_mult = self._state.strategy_multipliers.get(strategy, 1.0)

            # Already disabled — check for recovery
            if strategy in self._state.disabled_strategies:
                if win_rate > c.reenable_win_rate and avg_pnl > 0 and total >= c.reenable_min_trades:
                    self._state.disabled_strategies.discard(strategy)
                    self._state.strategy_multipliers[strategy] = c.reenable_multiplier  # Re-enable at reduced size
                    adaptations.append({
                        "change": f"re-enabled {strategy} (win_rate={win_rate:.0%})",
                        "type": "strategy_reenable",
                        "strategy": strategy,
                    })
                continue

            # Consistently losing → reduce or disable (protected strategies only reduce, never disable)
            if win_rate < c.reduce_win_rate and avg_pnl < 0 and total >= c.min_trades_to_learn:
                is_protected = strategy in c.protected_strategies
                if (current_mult <= c.strategy_multiplier_min and not is_protected
                        and total >= c.min_trades_to_disable):
                    # Already at minimum and not protected: disable it
                    self._state.disabled_strategies.add(strategy)
                    self._state.strategy_multipliers.pop(strategy, None)
                    adaptations.append({
                        "change": f"disabled {strategy} (win_rate={win_rate:.0%}, avg_pnl=${avg_pnl:.0f})",
                        "type": "strategy_disable",
                        "strategy": strategy,
                    })
                elif not (is_protected and current_mult <= c.strategy_multiplier_min):
                    new_mult = max(current_mult * c.reduce_factor, c.strategy_multiplier_min)
                    if abs(new_mult - current_mult) > c.reduce_min_change:
                        self._state.strategy_multipliers[strategy] = new_mult
                        adaptations.append({
                            "change": f"reduced {strategy} mult: {current_mult:.2f} -> {new_mult:.2f}",
                            "type": "strategy_reduce",
                            "strategy": strategy,
                        })

            # Consistently winning → boost
            elif win_rate > c.boost_win_rate and avg_pnl > 0 and total >= c.boost_min_trades:
                new_mult = min(current_mult * c.boost_factor, c.strategy_multiplier_max)
                if new_mult > current_mult + c.boost_min_change:
                    self._state.strategy_multipliers[strategy] = new_mult
                    adaptations.append({
                        "change": f"boosted {strategy} mult: {current_mult:.2f} -> {new_mult:.2f}",
                        "type": "strategy_boost",
                        "strategy": strategy,
                    })

        return adaptations

    def _adapt_confidence(self) -> list[dict]:
        """Raise min confidence if low-confidence trades consistently lose."""
        adaptations = []
        c = self._cfg

        low_band = self._state.confidence_stats.get("low", {})
        total_low = low_band.get("trades", 0)

        if total_low >= c.min_trades_to_learn:
            low_wr = low_band["wins"] / total_low
            low_avg_pnl = low_band["pnl"] / total_low

            if low_wr < c.confidence_raise_win_rate and low_avg_pnl < 0:
                new_conf = min(self._state.min_confidence_override + c.confidence_raise_step, c.confidence_cap)
                if new_conf > self._state.min_confidence_override + c.confidence_raise_min_change:
                    old = self._state.min_confidence_override
                    self._state.min_confidence_override = new_conf
                    adaptations.append({
                        "change": f"raised min_confidence: {old:.2f} -> {new_conf:.2f} (low-conf WR={low_wr:.0%})",
                        "type": "confidence_raise",
                    })

        return adaptations

    def _adapt_symbols(self) -> list[dict]:
        """Disable symbols that consistently lose."""
        adaptations = []
        c = self._cfg

        for symbol, stats in self._state.symbol_stats.items():
            if symbol in self._state.disabled_symbols:
                continue

            total = stats["wins"] + stats["losses"]
            if total < c.min_trades_to_learn:
                continue

            win_rate = stats["wins"] / total
            avg_pnl = stats["pnl"] / total

            if win_rate < c.symbol_disable_win_rate and avg_pnl < 0 and total >= c.min_trades_to_disable:
                self._state.disabled_symbols.add(symbol)
                adaptations.append({
                    "change": f"disabled symbol {symbol} (win_rate={win_rate:.0%}, avg_pnl=${avg_pnl:.0f})",
                    "type": "symbol_disable",
                    "symbol": symbol,
                })

        return adaptations

    def _confidence_band(self, confidence: float) -> str:
        if confidence < self._cfg.confidence_low_band_upper:
            return "low"
        elif confidence < self._cfg.confidence_mid_band_upper:
            return "medium"
        return "high"

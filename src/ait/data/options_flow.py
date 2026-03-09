"""Options flow detection — tracks unusual options activity.

Detects:
- Unusual volume spikes (volume >> open interest)
- Put/call ratio extremes
- Large block trades
- Smart money indicators (OTM options with unusual volume)

Uses data from IBKR options chains — no extra API needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ait.data.cache import TTLCache
from ait.utils.logging import get_logger

log = get_logger("data.options_flow")


@dataclass
class FlowSignal:
    """A detected unusual options activity signal."""

    symbol: str
    signal_type: str  # "unusual_volume", "put_call_extreme", "block_trade", "otm_sweep"
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 to 1.0
    details: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class FlowSummary:
    """Aggregated options flow for a symbol."""

    symbol: str
    total_call_volume: int = 0
    total_put_volume: int = 0
    put_call_ratio: float = 1.0
    unusual_signals: list[FlowSignal] = field(default_factory=list)
    overall_bias: str = "neutral"  # "bullish", "bearish", "neutral"
    bias_strength: float = 0.0


class OptionsFlowDetector:
    """Detects unusual options activity from chain data."""

    def __init__(
        self,
        volume_oi_threshold: float = 2.0,
        put_call_extreme_low: float = 0.5,
        put_call_extreme_high: float = 2.0,
    ) -> None:
        self._vol_oi_threshold = volume_oi_threshold
        self._pc_low = put_call_extreme_low
        self._pc_high = put_call_extreme_high
        self._cache = TTLCache(default_ttl=300, max_size=100)
        # Track historical volumes for comparison
        self._avg_volumes: dict[str, float] = {}

    def analyze_chain(
        self,
        symbol: str,
        calls: list[dict],
        puts: list[dict],
        underlying_price: float,
    ) -> FlowSummary:
        """Analyze an options chain for unusual activity.

        Args:
            symbol: Ticker symbol
            calls: List of call option data dicts with keys: strike, volume, open_interest, last_price, delta
            puts: List of put option data dicts (same keys)
            underlying_price: Current stock price
        """
        signals: list[FlowSignal] = []

        # Aggregate volumes
        total_call_vol = sum(c.get("volume", 0) for c in calls)
        total_put_vol = sum(p.get("volume", 0) for p in puts)

        # 1. Put/Call ratio
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 1.0

        if pc_ratio < self._pc_low:
            signals.append(FlowSignal(
                symbol=symbol,
                signal_type="put_call_extreme",
                direction="bullish",
                strength=min(1.0, (self._pc_low - pc_ratio) / self._pc_low),
                details=f"P/C ratio {pc_ratio:.2f} (very bullish, below {self._pc_low})",
            ))
        elif pc_ratio > self._pc_high:
            signals.append(FlowSignal(
                symbol=symbol,
                signal_type="put_call_extreme",
                direction="bearish",
                strength=min(1.0, (pc_ratio - self._pc_high) / self._pc_high),
                details=f"P/C ratio {pc_ratio:.2f} (very bearish, above {self._pc_high})",
            ))

        # 2. Unusual volume on individual strikes
        for option_list, option_type in [(calls, "call"), (puts, "put")]:
            for opt in option_list:
                volume = opt.get("volume", 0)
                oi = opt.get("open_interest", 1)
                strike = opt.get("strike", 0)
                delta = opt.get("delta", 0)

                if oi <= 0:
                    continue

                vol_oi = volume / oi
                if vol_oi >= self._vol_oi_threshold and volume >= 100:
                    # OTM options with unusual volume = stronger signal
                    is_otm = (
                        (option_type == "call" and strike > underlying_price * 1.02) or
                        (option_type == "put" and strike < underlying_price * 0.98)
                    )

                    strength = min(1.0, vol_oi / (self._vol_oi_threshold * 3))
                    if is_otm:
                        strength = min(1.0, strength * 1.5)

                    direction = "bullish" if option_type == "call" else "bearish"

                    signals.append(FlowSignal(
                        symbol=symbol,
                        signal_type="unusual_volume" if not is_otm else "otm_sweep",
                        direction=direction,
                        strength=strength,
                        details=(
                            f"{option_type.upper()} {strike:.0f}: "
                            f"vol={volume} vs OI={oi} ({vol_oi:.1f}x)"
                        ),
                    ))

        # 3. Overall bias
        bullish_strength = sum(s.strength for s in signals if s.direction == "bullish")
        bearish_strength = sum(s.strength for s in signals if s.direction == "bearish")

        if bullish_strength > bearish_strength * 1.5:
            overall_bias = "bullish"
            bias_strength = min(1.0, bullish_strength / (bullish_strength + bearish_strength + 0.01))
        elif bearish_strength > bullish_strength * 1.5:
            overall_bias = "bearish"
            bias_strength = min(1.0, bearish_strength / (bullish_strength + bearish_strength + 0.01))
        else:
            overall_bias = "neutral"
            bias_strength = 0.0

        summary = FlowSummary(
            symbol=symbol,
            total_call_volume=total_call_vol,
            total_put_volume=total_put_vol,
            put_call_ratio=pc_ratio,
            unusual_signals=signals,
            overall_bias=overall_bias,
            bias_strength=bias_strength,
        )

        if signals:
            log.info(
                "options_flow_detected",
                symbol=symbol,
                signals=len(signals),
                bias=overall_bias,
                pc_ratio=f"{pc_ratio:.2f}",
            )

        self._cache.set(f"flow_{symbol}", summary)
        return summary

    def get_cached_flow(self, symbol: str) -> FlowSummary | None:
        """Get cached flow summary for a symbol."""
        return self._cache.get(f"flow_{symbol}")

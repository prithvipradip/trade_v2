"""The sizing scalers and the inputs they read: ONE implementation, called by live and by research.

W114. The lot COUNT comes from ``ait.risk.lot_sizing.lots_for_budget``; this module produces its
``scaler`` argument. Until now the six factors lived inline in ``PositionSizer.calculate`` and the research
engine had no scaler at all, so switching the scalers on could not have meant the same thing in both. The
factors, the losing-day streak and the confidence a structure is sized on are defined here and nowhere else.

Pure on purpose: config models in, numbers out, no broker, no engine, no state store (like ``lot_sizing``).

* A scaler only throttles. ``scaler_factors`` may produce a product above 1 (the IV-rank and the
  long-straddle strategy factors reach 1.2); ``lots_for_budget`` clamps it at 1, so the count never
  exceeds floor(budget / risk of one lot) and live and research agree on what gate 6b would allow.
* The scalers act only for a strategy listed in ``position_sizing.scaled_strategies`` (empty today).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable

from ait.config.settings import PositionSizingConfig

# Strategies that benefit from high IV (selling premium): their IV-rank factor rises with the rank.
SELLING_STRATEGIES: frozenset[str] = frozenset({
    "covered_call", "cash_secured_put", "iron_condor", "short_strangle",
    "put_credit_spread", "call_credit_spread",
})

# Strategies whose ENTRY confidence comes from the payoff-matched range model rather than the directional
# one: P(price stays inside the wings) is what a short-premium neutral structure is paid for.
RANGE_GATED_STRATEGIES: frozenset[str] = frozenset({"iron_condor", "short_strangle"})

# How many calendar days back the losing-day streak looks (today excluded). A parameter of the streak rule
# the drawdown tiers read, not a trading preference: the rule is "consecutive losing days that had an
# entry, within the last 10 calendar days".
LOSING_DAYS_LOOKBACK_CALENDAR_DAYS = 10


@dataclass(frozen=True)
class ScalerInputs:
    """What the six factors read. ``None`` means unknown and is neutral (factor 1.0)."""

    confidence: float
    implied_vol: float
    iv_rank: float | None = None
    recent_losing_days: int = 0
    vix: float | None = None


@dataclass(frozen=True)
class ScalerFactors:
    conf: float
    vol: float
    iv_rank: float
    strategy: float
    drawdown: float
    vix: float

    @property
    def product(self) -> float:
        return self.conf * self.vol * self.strategy * self.iv_rank * self.drawdown * self.vix


def tier_factor(value: float, tiers: list[list[float]]) -> float:
    """1.0 below the first row; otherwise the factor of the highest row whose threshold ``value`` has
    reached (rows are validated strictly ascending)."""
    factor = 1.0
    for threshold, row_factor in tiers:
        if value >= threshold:
            factor = row_factor
    return factor


def scaler_factors(strategy: str, inputs: ScalerInputs, sizing: PositionSizingConfig,
                   min_confidence: float) -> ScalerFactors:
    """The six multipliers, moved UNCHANGED from ``PositionSizer.calculate``."""
    # Confidence: scale down for low confidence. At min_confidence use 50% of max, at 1.0 use 100%.
    conf_adj = 0.5 + 0.5 * ((inputs.confidence - min_confidence) / (1.0 - min_confidence))
    conf_adj = max(sizing.confidence_size_floor, min(1.0, conf_adj))

    # Volatility: smooth linear scale-down in high-vol environments. IV <= iv_full_size_max -> 1.0,
    # IV >= iv_floor_size_min -> iv_floor_size_factor. round(.., 12) keeps 0.60 - 0.20 the exact 0.40
    # literal the formula was written with (0.6 - 0.2 is 0.39999..).
    iv_span = round(sizing.iv_floor_size_min - sizing.iv_full_size_max, 12)
    if inputs.implied_vol <= sizing.iv_full_size_max:
        vol_adj = 1.0
    elif inputs.implied_vol >= sizing.iv_floor_size_min:
        vol_adj = sizing.iv_floor_size_factor
    else:
        vol_adj = 1.0 - (1.0 - sizing.iv_floor_size_factor) * (
            (inputs.implied_vol - sizing.iv_full_size_max) / iv_span)

    # IV rank alignment. Selling strategies benefit from a high rank (more premium to collect), buying
    # strategies from a low one. An UNKNOWN rank is neutral (1.0): it used to be rank 50, which equals 1.0
    # only while the min / max band is symmetric about 1.
    if inputs.iv_rank is None:
        iv_rank_adj = 1.0
    else:
        iv_rank_span = round(sizing.iv_rank_adj_max - sizing.iv_rank_adj_min, 12)
        if strategy in SELLING_STRATEGIES:
            iv_rank_adj = sizing.iv_rank_adj_min + iv_rank_span * (inputs.iv_rank / 100.0)
        else:
            iv_rank_adj = sizing.iv_rank_adj_max - iv_rank_span * (inputs.iv_rank / 100.0)
        iv_rank_adj = max(sizing.iv_rank_adj_min, min(sizing.iv_rank_adj_max, iv_rank_adj))

    strategy_adj = sizing.strategy_risk.get(strategy, 1.0)

    # Drawdown throttle: the default tiers halve size after 3+ consecutive losing days, quarter it after 5+.
    drawdown_adj = tier_factor(inputs.recent_losing_days, sizing.drawdown_tiers)

    # VIX regime: shrink positions where gamma risk is amplified. A VIX outage (None) is "regime unknown":
    # calm, no shrink (credit strategies fail closed upstream by design).
    vix_adj = tier_factor(inputs.vix or 0.0, sizing.vix_size_tiers)

    return ScalerFactors(conf=conf_adj, vol=vol_adj, iv_rank=iv_rank_adj, strategy=strategy_adj,
                         drawdown=drawdown_adj, vix=vix_adj)


def lot_scaler(strategy: str, factors: ScalerFactors, sizing: PositionSizingConfig) -> float:
    """The ``scaler`` for ``lots_for_budget``: the factors' product for a strategy listed in
    ``position_sizing.scaled_strategies``, 1.0 for every other strategy."""
    return factors.product if strategy in sizing.scaled_strategies else 1.0


def recent_losing_days(today: date, day_stats: Callable[[date], Any],
                       lookback_days: int = LOSING_DAYS_LOOKBACK_CALENDAR_DAYS) -> int:
    """Consecutive losing days that had an entry, counting back from yesterday (``today`` excluded).

    ``day_stats(d)`` returns an object with ``trades_taken`` and ``total_pnl`` (or ``None`` for an unknown
    day). A day with no entry is skipped (weekends, holidays, idle days); the first day that is not a loss
    ends the streak. This is the rule live reads from its ``daily_stats`` table.
    """
    streak = 0
    for days_back in range(1, lookback_days + 1):
        stats = day_stats(today - timedelta(days=days_back))
        if stats is None or stats.trades_taken == 0:
            continue
        if stats.total_pnl < 0:
            streak += 1
        else:
            break
    return streak


def sizing_confidence(strategy: str, *, gates_on: bool, range_prob: float | None,
                      directional_conf: float, neutral_conf: float,
                      range_gated: frozenset[str] = RANGE_GATED_STRATEGIES) -> float:
    """The confidence a structure is SIZED on.

    A range-gated strategy (iron condor, short strangle) with the entry gates off is sized on the
    configured direction-neutral baseline (``ml.observe_mode_neutral_confidence``): the range model's
    verdict is only logged there. With the gates on it is sized on the range model's probability (after
    the fractal delta) when there is one. Everything else is sized on the directional confidence.
    """
    if strategy in range_gated:
        if not gates_on:
            return float(neutral_conf)
        if range_prob is not None:
            return float(range_prob)
    return float(directional_conf)

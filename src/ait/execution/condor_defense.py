"""W56 (tastytrade-style tested/untested-side condor defense, decision
layer): per open iron condor, classify the TESTED short's synthetic delta
(W55, synthetic_delta.py) and DTE against CondorDefenseConfig's table to
decide hold / roll-untested / roll-tested / close.

Nothing calls into this module yet. `CondorDefenseConfig.enabled` defaults
to False (settings.py) -- a future wave wires check_positions to consult
this, gated on the shared non-boolean PositionStatus/decision-output design
W54 also needs (PLAN.md, W55's own NEXT). Every failure path here returns a
HOLD decision with a reason, never raises -- same fail-safe convention
roll_selector.py's select_roll_target and synthetic_delta.py's
synthetic_leg_greeks/synthetic_vertical_deltas use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum

from ait.bot.state import TradeRecord
from ait.config.settings import CondorDefenseConfig
from ait.execution.roll_selector import _at_risk_vertical
from ait.execution.synthetic_delta import synthetic_vertical_deltas
from ait.utils.logging import get_logger

log = get_logger("execution.condor_defense")


class DefenseAction(str, Enum):
    HOLD = "hold"
    ROLL_UNTESTED = "roll_untested"
    ROLL_TESTED = "roll_tested"
    CLOSE = "close"


@dataclass(frozen=True)
class DefenseDecision:
    """The decision for one open iron condor's tested side, this pass."""

    action: DefenseAction
    reason: str
    tested_side: str | None = None          # "P" / "C"
    tested_short_delta: float | None = None  # |delta|, always >= 0
    dte: int | None = None


_HOLD_DISABLED = DefenseDecision(DefenseAction.HOLD, "disabled")


def classify_tested_delta(
    abs_delta: float, dte: int, config: CondorDefenseConfig,
) -> tuple[DefenseAction, str]:
    """Pure table lookup -- the tested short's |delta| and DTE against
    config's bands, no data access. Kept separate from evaluate_condor_
    defense so band edges are testable with exact numbers alone.

    Order matters: the close threshold is checked first (it applies at any
    DTE and is never side-filtered), then the monitor floor, then the two
    roll bands. A cell neither row covers (e.g. a tested-band delta still
    inside the untested DTE window) matches nothing -- HOLD, reason
    "no_rule_matched", not an escalation to either roll or close.
    """
    if abs_delta >= config.close_delta_threshold:
        return DefenseAction.CLOSE, "close_delta_threshold"
    if abs_delta < config.monitor_delta_max:
        return DefenseAction.HOLD, "below_monitor_band"
    if (abs_delta >= config.roll_tested_delta_min
            and dte >= config.roll_tested_dte_min):
        return DefenseAction.ROLL_TESTED, "roll_tested_band"
    if (config.roll_untested_delta_min <= abs_delta < config.roll_untested_delta_max
            and config.roll_untested_dte_min <= dte <= config.roll_untested_dte_max):
        return DefenseAction.ROLL_UNTESTED, "roll_untested_band"
    return DefenseAction.HOLD, "no_rule_matched"


def _tested_side(trade: TradeRecord, spot: float) -> tuple[str, float, float] | None:
    """(side, short_strike, long_strike) for whichever side is under
    pressure -- the side whose short strike is relatively closer to spot
    (an ITM short comes out negative, so it is always picked over an OTM
    one on the other side). Ties go to the put, where skew makes the real
    delta larger for the same relative distance.

    Deliberately NOT "whichever side has the larger synthetic delta": the
    untested side is often far OTM with a near-zero mark, exactly where the
    W55 IV solve is most likely to fail (failed_other / a sub-floor sigma)
    -- needing both deltas to succeed would blind this exactly when the
    tested side is under real pressure. Only the tested side's own delta is
    ever solved for.

    Returns None if either side of the condor isn't a clean single
    short+long pair (matching _at_risk_vertical's own convention).
    """
    put_vertical = _at_risk_vertical(trade, "P")
    call_vertical = _at_risk_vertical(trade, "C")
    if put_vertical is None or call_vertical is None:
        return None
    put_short, put_long = put_vertical
    call_short, call_long = call_vertical
    if spot <= 0:
        return None

    put_distance = (spot - put_short) / spot
    call_distance = (call_short - spot) / spot
    if put_distance <= call_distance:
        return "P", put_short, put_long
    return "C", call_short, call_long


def evaluate_condor_defense(
    trade: TradeRecord,
    option_marks: dict[tuple, float],
    spot: float,
    dte: int | None,
    config: CondorDefenseConfig,
) -> DefenseDecision:
    """Decide what (if anything) to do about `trade`'s tested side, this
    pass. Never raises -- any failure to gather what's needed is a HOLD
    with a reason naming why, matching every other additive module this
    register has built (roll_selector.py, synthetic_delta.py).
    """
    if not config.enabled:
        return _HOLD_DISABLED

    if trade.strategy != "iron_condor":
        return DefenseDecision(DefenseAction.HOLD, "not_an_iron_condor")

    tested = _tested_side(trade, spot)
    if tested is None:
        return DefenseDecision(DefenseAction.HOLD, "no_clean_condor")
    side, _short_strike, _long_strike = tested

    if dte is None:
        return DefenseDecision(DefenseAction.HOLD, "dte_unavailable", tested_side=side)

    deltas = synthetic_vertical_deltas(trade, side, option_marks, spot)
    tested_greeks = deltas["short"]
    if tested_greeks is None:
        return DefenseDecision(
            DefenseAction.HOLD, "tested_delta_unavailable", tested_side=side, dte=dte)

    abs_delta = abs(tested_greeks.delta)
    action, reason = classify_tested_delta(abs_delta, dte, config)

    if action in (DefenseAction.ROLL_TESTED, DefenseAction.ROLL_UNTESTED):
        wants_tested_roll = action is DefenseAction.ROLL_TESTED
        allowed = (
            config.side == "both"
            or (wants_tested_roll and config.side == "tested_only")
            or (not wants_tested_roll and config.side == "untested_only")
        )
        if not allowed:
            blocked_reason = (
                "roll_tested_blocked_by_side" if wants_tested_roll
                else "roll_untested_blocked_by_side"
            )
            return DefenseDecision(
                DefenseAction.HOLD, blocked_reason,
                tested_side=side, tested_short_delta=abs_delta, dte=dte)

    return DefenseDecision(
        action, reason, tested_side=side, tested_short_delta=abs_delta, dte=dte)

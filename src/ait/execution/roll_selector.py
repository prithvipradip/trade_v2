"""W53 (roll-mechanics groundwork): pick a replacement vertical (short +
paired long) at a later expiry for an at-risk credit position.

Nothing calls into this module yet -- W54 dispatches rule 3/rule 3e's exit
decision through the assignment-response policy (PLAN.md finding [3]f),
which is the first real caller. Every picker here follows the fail-safe
convention iron_condor.py's own strike pickers use: return None rather than
raise when nothing suitable is found.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ait.bot.state import TradeRecord
from ait.config.settings import AssignmentResponseConfig
from ait.data.options_chain import OptionContract, OptionsChainService
from ait.utils.logging import get_logger

log = get_logger("execution.roll_selector")

# W69: the new-short delta target (default 0.20, the same one
# strategies/iron_condor.py's short-strike picker uses) and its tolerance
# (default 0.07, matching Strategy._find_strike_by_delta) moved to
# AssignmentResponseConfig.roll_short_delta_target / _tolerance. A rolled
# vertical looks like a fresh entry at the new expiry, not a replica of the
# old position's now-decayed entry delta.


@dataclass
class RollTarget:
    """A replacement vertical: sell `new_short`, buy `new_long`, at `expiry`
    (ISO date string, matching the trade.legs JSON convention).

    `iv_is_guessed` (W54a) is the target chain's own OptionsChain.
    iv_is_guessed flag, carried through so a caller deciding whether to run
    the EV comparison (execution/roll_comparison.py) can refuse it on a
    chain with no usable IV data -- the SAME chain-level signal iron_
    condor.py's own entry path already reuses, not a new detector."""

    new_short: OptionContract
    new_long: OptionContract
    expiry: str
    iv_is_guessed: bool = False


def _at_risk_vertical(trade: TradeRecord, side: str) -> tuple[float, float] | None:
    """(short_strike, long_strike) for `side` ("P" or "C") from trade.legs,
    or None if that side isn't a clean single short+long pair.

    Same legs-JSON idiom portfolio.py's _short_strikes_from_legs (W52) and
    rule 1b/3e use -- extended here to also return the PAIRED LONG strike,
    which that helper (deliberately scoped to rule 3's own needs) does not.
    """
    try:
        legs = json.loads(trade.legs) if trade.legs else []
    except (ValueError, TypeError):
        return None
    shorts = [float(leg["strike"]) for leg in legs
              if str(leg.get("action", "")).upper() == "SELL"
              and str(leg.get("right", leg.get("type", ""))).upper().startswith(side)]
    longs = [float(leg["strike"]) for leg in legs
             if str(leg.get("action", "")).upper() == "BUY"
             and str(leg.get("right", leg.get("type", ""))).upper().startswith(side)]
    if len(shorts) != 1 or len(longs) != 1:
        return None
    return shorts[0], longs[0]


async def select_roll_target(
    trade: TradeRecord,
    side: str,
    chain_service: OptionsChainService,
    config: AssignmentResponseConfig,
) -> RollTarget | None:
    """Pick a replacement vertical for the at-risk `side` ("P" or "C") of
    `trade`, at a later expiry within
    [config.roll_target_dte_min, config.roll_target_dte_max].

    Never raises. Returns None on anything from a missing/malformed
    trade.legs to an empty chain to no strike clearing the delta/liquidity
    gates -- callers must treat None as "nothing to roll to right now", not
    as an error requiring escalation. `side` is supplied by the caller
    (e.g. rule 3's itm_side, portfolio.py:743) rather than re-derived here,
    since whatever triggers a roll decision already knows which side is
    at risk.
    """
    vertical = _at_risk_vertical(trade, side)
    if vertical is None:
        log.warning("roll_target_no_clean_vertical", trade_id=trade.trade_id,
                    symbol=trade.symbol, side=side)
        return None
    old_short_strike, old_long_strike = vertical
    width = abs(old_short_strike - old_long_strike)

    try:
        chains = await chain_service.get_chain(
            trade.symbol,
            min_dte=config.roll_target_dte_min,
            max_dte=config.roll_target_dte_max,
        )
    except Exception as e:  # noqa: BLE001 -- fail-safe: no roll target, not a crash
        log.warning("roll_target_chain_fetch_failed", trade_id=trade.trade_id,
                    symbol=trade.symbol, error=str(e), error_type=type(e).__name__)
        return None
    if not chains:
        log.warning("roll_target_no_chain", trade_id=trade.trade_id,
                    symbol=trade.symbol, dte_min=config.roll_target_dte_min,
                    dte_max=config.roll_target_dte_max)
        return None

    # Furthest-out expiry in the window: a roll is meant to buy real time
    # before the next DTE<=1 crisis, not just clear the DTE floor.
    chain = max(chains, key=lambda c: c.expiry)
    liquid = chain.filter_liquid()
    candidates = liquid.puts if side == "P" else liquid.calls
    if not candidates:
        log.warning("roll_target_no_liquid_contracts", trade_id=trade.trade_id,
                    symbol=trade.symbol, side=side, expiry=str(chain.expiry))
        return None

    # W69: target/tolerance are assignment_response.roll_short_delta_* config
    # (defaults 0.20 / 0.07 — the former module constants).
    _delta_target = config.roll_short_delta_target
    _delta_tol = config.roll_short_delta_tolerance
    new_short = min(candidates,
                     key=lambda c: abs(abs(c.delta) - _delta_target))
    if abs(abs(new_short.delta) - _delta_target) > _delta_tol:
        log.warning("roll_target_short_outside_delta_tolerance", trade_id=trade.trade_id,
                    symbol=trade.symbol, side=side, delta=new_short.delta)
        return None

    # Same width in dollars as the vertical being replaced -- simplest
    # choice, preserves the position's max-loss shape. A defensive
    # adjustment doesn't need a fresh vol-scaled width recompute
    # (strategies/iron_condor.py's _vol_scaled_width, written for sizing a
    # brand-new entry).
    target_long_strike = (new_short.strike - width if side == "P"
                          else new_short.strike + width)
    new_long = min(candidates, key=lambda c: abs(c.strike - target_long_strike))
    if new_long.strike == new_short.strike:
        log.warning("roll_target_no_wing_available", trade_id=trade.trade_id,
                    symbol=trade.symbol, side=side)
        return None

    return RollTarget(new_short=new_short, new_long=new_long, expiry=str(chain.expiry),
                      iv_is_guessed=bool(getattr(chain, "iv_is_guessed", False)))

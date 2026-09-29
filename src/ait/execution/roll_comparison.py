"""W54a: the close-or-roll comparison models -- given a PREPARED roll combo
(roll_executor.prepare_roll_combo), decide whether rolling is actually
preferable to closing outright right now.

Nothing calls into this module yet: rule 3/rule 3e still always close
(portfolio.py), and wiring them to consult AssignmentResponseConfig.policy
and call either function here is W54b's separately-scoped job -- the same
"build additively, wire later" discipline every prior piece of this arc has
followed (roll_selector.py, roll_executor.py, roll_booking.py).

Both functions are pure and take already-resolved inputs rather than
reaching for a chain or a broker themselves -- portfolio.py (where rule 3/
3e's decision is made) has no broker access, the same reason condor_
defense's own roll verdict stays decision-only there; gathering live inputs
happens in the orchestrator, same as _execute_roll already does for
placement.
"""
from __future__ import annotations

from ait.data.options_chain import black_scholes_price
from ait.execution.roll_executor import PreparedRollCombo
from ait.utils.logging import get_logger

log = get_logger("execution.roll_comparison")


def close_cost_now(prepared: PreparedRollCombo) -> float:
    """The cost to close EVERYTHING right now, in the same "cost to close"
    trade convention exit_price/_calculate_realized_pnl already use
    (executor.py:2695-2713) -- positive means paying a debit to close,
    negative means realizing a credit. Reuses roll_booking.book_roll_fill's
    own exact formula (`closing_net - carried_value`), applied here to a MID
    estimate from `prepared` instead of a real fill, so "what would closing
    cost today" and "what did closing actually cost" are computed the same
    way."""
    return prepared.closing_net - prepared.carried_value


def heuristic_should_roll(prepared: PreparedRollCombo) -> bool:
    """The tastytrade-style rule from PLAN.md's W54 scoping note ([1]/[3]d):
    "roll only on a net credit that doesn't increase max-loss vs closing."

    Both halves of that rule are ALREADY structurally enforced by the time
    `prepared` exists at all: `prepare_roll_combo`'s own debit_cap (fed
    `-assignment_response.roll_min_credit` by the caller) refuses anything
    that isn't a net credit, and `select_roll_target` always rolls to the
    SAME width, preserving max-loss. What's actually left to decide: is
    rolling even the right move, versus just closing and taking today's
    result? Don't roll a position that would already be profitable or
    breakeven if closed now -- only roll one that would realize a loss,
    where the roll's own (already-guaranteed) credit reduces that loss
    rather than locking it in.
    """
    return close_cost_now(prepared) > 0


def ev_should_roll(
    closing_legs_priced: list[dict],
    opening_legs_priced: list[dict],
    carried_legs_priced: list[dict],
    r: float,
) -> bool | None:
    """The same decision as `heuristic_should_roll`, but priced at each
    leg's own MODEL fair value (black_scholes_price, at its solved IV) in
    place of raw market mid quotes -- catches what the scoping note's
    finding [2] named: a roll whose raw market credit looks fine but is
    actually giving away more theoretical time value than it collects.

    Each leg dict: {"strike","action","spot","t","iv","flag"} -- "action"
    is "BUY"/"SELL" in the SAME position-sign convention leg_net_price uses
    elsewhere (BUY+/SELL-); closing_legs_priced use the REVERSED (closing)
    action, matching roll_executor.legs_for_roll's own convention, so the
    three lists compose exactly like PreparedRollCombo's own closing_legs/
    opening_legs/carried_legs.

    Returns None -- defer to the heuristic -- if ANY leg's IV is missing or
    its price fails to solve. Never guesses a fair value from an unsolved
    input.
    """
    def _net(legs: list[dict]) -> float | None:
        total = 0.0
        for leg in legs:
            price = black_scholes_price(
                leg["spot"], leg["strike"], leg["t"], r, leg["iv"], leg["flag"])
            if price is None:
                log.warning("ev_leg_price_unsolved", strike=leg["strike"],
                           flag=leg.get("flag"))
                return None
            sign = 1.0 if str(leg["action"]).upper() == "BUY" else -1.0
            total += sign * price
        return total

    closing_net = _net(closing_legs_priced)
    opening_net = _net(opening_legs_priced)
    carried_net = _net(carried_legs_priced) if carried_legs_priced else 0.0
    if closing_net is None or opening_net is None or carried_net is None:
        return None

    model_close_cost_now = closing_net - carried_net
    return model_close_cost_now > 0

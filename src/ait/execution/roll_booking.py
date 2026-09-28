"""Roll booking mechanics -- the DB-sequencing piece between a filled roll
combo and two updated trade rows. Nothing calls into this module yet:
`executor.py`'s live `check_fills`/`_pending_orders`/`_pending_exit_orders`
loops are untouched -- teaching them to recognize a roll order and dispatch
here is the actual wiring step, deliberately deferred (the single most
sensitive part of this codebase). Every DB write here reuses `state.py`'s
existing exactly-once primitives (`transition`, `close_trade`) rather than
reimplementing them.

The booking machinery this module completes actually lives in executor.py +
state.py, NOT reconciler.py -- reconciler.py is a periodic (not per-tick)
position-diff safety net, confirmed by direct re-read while designing this,
worth stating plainly since an earlier register entry this session assumed
otherwise.

Two real risks this design was built to avoid, both confirmed against
source before writing any code:
  - HALT_UNTRACKED freezes on a position whose legs aren't in local_map (built
    from get_open_trades(), which includes PENDING) on a SECOND sighting.
    seed_roll_opening_trade writes B's PENDING row (with B's real legs)
    BEFORE any roll order would be placed -- the same "DB row before
    placeOrder" ordering CLAUDE.md already documents for entries
    (executor.py:execute_signal) -- so this never trips.
  - go_live.py already excludes CLOSING/PENDING rows from the verdict sample
    (counted only toward the drawdown base) -- an intermediate state during
    a multi-step roll booking does not corrupt the 50-close track record.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime

from ait.bot.state import StateManager, TradeRecord, TradeStatus
from ait.execution.roll_selector import RollTarget
from ait.strategies.base import CREDIT_STRATEGIES
from ait.utils.logging import get_logger
from ait.utils.time import now_et

log = get_logger("execution.roll_booking")


@dataclass
class PendingRollOrder:
    """The shape a future executor.py wiring wave would key a new
    _pending_roll_orders dict by (mirroring PendingOrder/PendingExitOrder's
    own shape), so check_fills could recognize a roll order and call
    book_roll_fill instead of the single-trade paths. Not instantiated or
    consumed by any live code this wave -- the agreed data contract only."""

    order_id: int
    closing_trade_id: str
    opening_trade_id: str
    closing_conids: set[int]
    opening_conids: set[int]


def seed_roll_opening_trade(
    state: StateManager,
    closing_trade: TradeRecord,
    opening_legs: list[dict],
    roll_target: RollTarget,
    quantity: int,
    entry_price_estimate: float,
) -> TradeRecord:
    """Write trade B's PENDING row -- mirrors execute_signal's exact
    record_trade-before-placement ordering (executor.py:447-493), just for
    a roll's opening leg instead of a fresh strategy signal. Must be called
    BEFORE a future wiring wave ever places the roll combo, or B's legs are
    untracked at IBKR until the reconciler's next pass -- see this module's
    own docstring. entry_price_estimate is prepare_roll_combo's mid_net,
    the same "optimistic signal price, corrected once the real fill lands"
    pattern every other entry already uses.
    """
    trade_id = f"T-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    record = TradeRecord(
        trade_id=trade_id,
        symbol=closing_trade.symbol,
        strategy=closing_trade.strategy,
        direction=closing_trade.direction,
        status=TradeStatus.PENDING,
        entry_time=now_et().isoformat(),
        entry_price=entry_price_estimate,
        quantity=quantity,
        contract_type=closing_trade.contract_type,
        strike=None,
        expiry=roll_target.expiry,
        legs=json.dumps(opening_legs),
    )
    state.record_trade(record)
    return record


def cas_roll_closing_trade(state: StateManager, trade_id: str) -> bool:
    """Pre-transition trade A to CLOSING before a future wiring wave places
    the roll combo -- a thin, named wrapper around the SAME CAS primitive
    orchestrator._cas_to_closing already uses for every other exit
    (FILLED/PARTIAL -> CLOSING), so this call site's intent (the same
    ordering every exit already follows) is self-documenting."""
    return state.transition(
        trade_id, (TradeStatus.FILLED, TradeStatus.PARTIAL), TradeStatus.CLOSING,
    )


def split_roll_fill_price(
    leg_fills: list[dict], closing_conids: set[int], opening_conids: set[int],
) -> tuple[float, float] | None:
    """Split a roll combo's per-leg fill rows (the shape _sweep_executions
    already produces per row: {"conId","side","price","shares"}) into
    (closing_net, opening_net) -- the same BUY-positive/SELL-negative
    signed-per-leg-sum convention leg_net_price/_option_position_unrealized
    already use. Returns None if any leg's conId isn't in EITHER set -- a
    data-integrity guard, fail fast rather than silently mis-attribute a
    fill to the wrong side of the roll."""
    closing_net = 0.0
    opening_net = 0.0
    for fill in leg_fills:
        con_id = fill["conId"]
        sign = 1.0 if str(fill["side"]).upper() == "BUY" else -1.0
        contribution = sign * float(fill["price"]) * float(fill.get("shares", 1))
        if con_id in closing_conids:
            closing_net += contribution
        elif con_id in opening_conids:
            opening_net += contribution
        else:
            log.warning("roll_fill_leg_unattributed", con_id=con_id)
            return None
    return round(closing_net, 4), round(opening_net, 4)


def book_roll_fill(
    state: StateManager,
    closing_trade: TradeRecord,
    opening_trade_id: str,
    leg_fills: list[dict],
    closing_conids: set[int],
    opening_conids: set[int],
    commission: float = 0.0,
) -> bool:
    """Book a filled roll combo's single fill as TWO trade updates: promote
    B from PENDING to FILLED (real entry price, rolled_from_trade_id set to
    A's trade_id -- state.py:99, unpopulated by any code path until now)
    FIRST, then close A (via the EXISTING close_trade outbox/claim/
    core_booked exactly-once machinery, CAS narrowed to CLOSING only -- a
    stray fill on a non-CLOSING A is refused, not silently booked).
    Returns True only if BOTH sides land, each logged for the operator to
    reconcile manually on any refusal. The ordering is deliberate: if B's
    promotion fails, A is left untouched in CLOSING (recoverable). If A's
    close then fails (rarer -- B already promoted at that point), B is left
    FILLED while A stays stuck in CLOSING -- a real, smaller residual risk
    this two-phase (non-transactional-across-both-rows) design does not
    eliminate, registered as a known limitation for whoever wires this into
    executor.py's live fill loop, not solved here.
    """
    split = split_roll_fill_price(leg_fills, closing_conids, opening_conids)
    if split is None:
        log.error("roll_booking_split_failed", closing_trade_id=closing_trade.trade_id)
        return False
    closing_net, opening_net = split

    if closing_trade.strategy in CREDIT_STRATEGIES:
        signed_entry = -abs(closing_trade.entry_price)
    else:
        signed_entry = abs(closing_trade.entry_price)
    realized_pnl = round((closing_net - signed_entry) * 100 * closing_trade.quantity, 2)

    # Order matters: promote B FIRST. close_trade and transition are two
    # independent CAS calls (each internally atomic, not a single shared
    # transaction across both trades' rows -- composing them into one true
    # cross-table transaction would mean bypassing state.py's own
    # exactly-once primitives, which this module deliberately reuses
    # rather than reimplements). Promoting B first means a failure there
    # (caught directly by this module's own tests) leaves A untouched in
    # CLOSING -- recoverable, still awaiting its real close. Closing A
    # first and having B's promotion fail would instead leave A already
    # CLOSED while B never existed as a real fill -- the position would
    # have silently vanished. The residual risk this ordering does NOT
    # eliminate (A's close failing AFTER B is already promoted, leaving
    # both a live B and a stuck-CLOSING A) is smaller and more visible --
    # registered as a known limitation for whoever wires this into
    # executor.py's live fill loop, not solved here.
    promoted = state.transition(
        opening_trade_id, (TradeStatus.PENDING, TradeStatus.PARTIAL), TradeStatus.FILLED,
    )
    if not promoted:
        log.error("roll_booking_promote_refused", closing_trade_id=closing_trade.trade_id,
                  opening_trade_id=opening_trade_id)
        return False

    closed = state.close_trade(
        closing_trade.trade_id,
        exit_price=closing_net,
        realized_pnl=realized_pnl,
        commission=commission,
        exit_reason_detailed=f"rolled_to_{opening_trade_id}",
        from_statuses=(TradeStatus.CLOSING,),
    )
    if not closed:
        log.error("roll_booking_close_refused", closing_trade_id=closing_trade.trade_id,
                  opening_trade_id=opening_trade_id)
        return False

    with sqlite3.connect(state._db_path) as conn:
        conn.execute(
            """UPDATE trades SET entry_price = ?, quantity = ?, rolled_from_trade_id = ?
               WHERE trade_id = ? AND status = ?""",
            (opening_net, closing_trade.quantity, closing_trade.trade_id,
             opening_trade_id, TradeStatus.FILLED.value),
        )
        conn.commit()

    log.info("roll_booking_complete", closing_trade_id=closing_trade.trade_id,
             opening_trade_id=opening_trade_id, realized_pnl=realized_pnl)
    return True

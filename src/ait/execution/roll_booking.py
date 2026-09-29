"""Roll booking mechanics -- the DB-sequencing piece between a filled roll
combo and two updated trade rows. Nothing PLACES an order or CALLS this
module yet: `executor.py`'s live `check_fills`/`_pending_orders`/
`_pending_exit_orders` loops are untouched -- teaching them to recognize a
roll order and dispatch to book_roll_fill is the actual placement/fill
wiring step, deliberately deferred to a later wave (the single most
sensitive part of this codebase). Every DB write here reuses `state.py`'s
existing exactly-once primitives (`transition`, `close_trade`) rather than
reimplementing them.

W57 (foundation wave, before that later wiring): `reconciler.py` IS now
roll-aware -- it recognizes a working roll order via `parse_roll_order_ref`
and keeps A CLOSING instead of either re-adopting the order as a plain exit
(double-booking the eventual fill) or reverting A out of CLOSING mid-roll,
and its stale-pending sweep skips the entry-budget refund for a roll child
B. This was necessary NOW, even though placement itself is still deferred,
because `reconciler.py` and the sweep both run every cycle against
WHATEVER rows exist in `trades` -- a wiring wave that starts writing B's
PENDING row and placing real orders without reconciler.py already knowing
the marker would immediately hit both failure modes on the very first live
roll. The booking machinery this module completes still lives in
executor.py + state.py, not reconciler.py -- reconciler.py's role here is
narrowly "don't fight a roll it doesn't own," confirmed by direct re-read
of its CLOSING-revert and stale-pending-sweep logic before writing any of
this, worth stating plainly since an earlier register entry this session
assumed reconciler.py owned booking outright.

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

W57 also fixed three latent bugs found while preparing the NEXT wiring
wave, all confirmed by direct source read and reproduced by a failing test
before the fix (see PLAN.md's W57 entry for the full account):
  1. A roll of one vertical orphaned the condor's OTHER vertical at the
     broker (no trade row managing it) -- fixed via roll_executor.py's new
     carried_legs/carried_value, threaded through seed_roll_opening_trade /
     _promote_and_link_opening_trade / book_roll_fill below.
  2. book_roll_fill's realized_pnl formula had the entry/exit sign
     inverted (a real position-value-vs-trade-cost convention mismatch,
     not a typo) -- see book_roll_fill's own docstring for the corrected
     derivation.
  3. seed_roll_opening_trade json.dumps'd legs straight out of
     qualify_roll_legs, which raises TypeError on the first real call
     (each leg carries a live, non-serializable ib_insync Option under
     '_contract') -- fixed via the new _persistable_legs helper, used at
     every legs write in this module.
"""
from __future__ import annotations

import json
import re
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
    consumed by any live code this wave -- the agreed data contract only.
    `carried_legs`/`carried_value` (W57) are prepare_roll_combo's own
    fields, carried through so the eventual fill handler can pass them to
    book_roll_fill without re-deriving them."""

    order_id: int
    closing_trade_id: str
    opening_trade_id: str
    closing_conids: set[int]
    opening_conids: set[int]
    carried_legs: list[dict]
    carried_value: float


def _persistable_legs(legs: list[dict]) -> list[dict]:
    """Strip qualify_roll_legs' own '_contract' (a live ib_insync Option,
    not JSON-serializable) before a legs list is written to trades.legs.
    W57: seed_roll_opening_trade previously json.dumps'd prepared legs
    straight from qualify_roll_legs and raised TypeError on the first real
    call -- every roll leg list that has passed through qualification must
    go through this before any json.dumps."""
    return [{k: v for k, v in leg.items() if k != "_contract"} for leg in legs]


_ROLL_REF_RE = re.compile(r"^ROLL:(.+):(.+)$")


def roll_order_ref(closing_trade_id: str, opening_trade_id: str) -> str:
    """The orderRef stamped on a placed roll combo -- the single marker
    reconciler.py uses to recognise a working roll order instead of either
    re-adopting it as a plain exit (double-booking the fill) or reverting
    A out of CLOSING mid-roll (R17's existing exact-orderRef match would
    otherwise do one or the other, since it has never seen a 2-trade
    order). Single source: reconciler.py imports parse_roll_order_ref
    rather than re-deriving the format."""
    return f"ROLL:{closing_trade_id}:{opening_trade_id}"


def parse_roll_order_ref(order_ref: str) -> tuple[str, str] | None:
    """(closing_trade_id, opening_trade_id) from a roll_order_ref string,
    or None if `order_ref` isn't one -- fail-safe, so a plain exit's
    order_ref (a bare trade_id, no colons) is never misread as a roll."""
    if not order_ref:
        return None
    m = _ROLL_REF_RE.match(order_ref)
    return (m.group(1), m.group(2)) if m else None


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

    B's PENDING row carries only the NEW (opening) legs -- W57: seeding it
    with A's still-live carried vertical too would make both reconcile.py
    and the stale-pending sweep see those legs as "live" and promote/count
    B against them before the roll ever fills; the carried vertical is
    attached to B only at promotion (_promote_and_link_opening_trade),
    once the roll has actually landed. rolled_from_trade_id is set here,
    at seed time, via an explicit UPDATE -- record_trade()'s own INSERT
    doesn't carry the column (state.py:340-369, same gap capital_at_risk
    has) -- so reconciler.py can recognise "this PENDING row belongs to an
    in-flight roll" the moment it exists, before any fill.
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
        legs=json.dumps(_persistable_legs(opening_legs)),
        rolled_from_trade_id=closing_trade.trade_id,
    )
    state.record_trade(record)
    with state._connect() as conn:
        conn.execute(
            "UPDATE trades SET rolled_from_trade_id = ? WHERE trade_id = ?",
            (closing_trade.trade_id, trade_id),
        )
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


def _promote_and_link_opening_trade(
    state: StateManager, opening_trade_id: str, entry_price: float, quantity: int,
    closing_trade_id: str, legs: str,
) -> bool:
    """Promote B (PENDING/PARTIAL -> FILLED), set its rolled_from_trade_id
    back-link, AND attach A's carried vertical to it (W57: B's PENDING row
    was seeded with only the new legs -- see seed_roll_opening_trade -- so
    this is the first point B's `legs` column reflects the whole post-roll
    position) in ONE atomic statement -- not state.transition() followed by
    separate UPDATEs. The multi-step version left a real crash window: a
    process dying mid-sequence leaves B FILLED with no way back to A, or
    FILLED with only half its real legs, either one breaking the one field
    this wave gives its first real reader (go_live.py's chain folding) or
    orphaning the carried vertical anyway. state.transition() only ever
    sets status, so it can't carry the extra columns -- this is a genuinely
    new primitive, not a reimplementation of an existing one, built with
    the SAME BEGIN IMMEDIATE CAS idiom close_trade already uses."""
    with sqlite3.connect(state._db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """UPDATE trades SET status = ?, entry_price = ?, quantity = ?,
                   rolled_from_trade_id = ?, legs = ?
               WHERE trade_id = ? AND status IN (?, ?)""",
            (TradeStatus.FILLED.value, entry_price, quantity, closing_trade_id, legs,
             opening_trade_id, TradeStatus.PENDING.value, TradeStatus.PARTIAL.value),
        )
        conn.commit()
    return cur.rowcount > 0


def book_roll_fill(
    state: StateManager,
    closing_trade: TradeRecord,
    opening_trade_id: str,
    leg_fills: list[dict],
    closing_conids: set[int],
    opening_conids: set[int],
    carried_legs: list[dict],
    carried_value: float,
    commission: float = 0.0,
    capital_at_risk: float | None = None,
) -> bool:
    """Book a filled roll combo's single fill as TWO trade updates: promote
    B from PENDING to FILLED (real entry price, rolled_from_trade_id set to
    A's trade_id -- state.py:99, unpopulated by any code path until now,
    the carried vertical attached to B's legs -- W57 -- all written
    atomically with the promotion -- see _promote_and_link_opening_trade)
    FIRST, then close A (via the EXISTING close_trade outbox/claim/
    core_booked exactly-once machinery, CAS narrowed to CLOSING only -- a
    stray fill on a non-CLOSING A is refused, not silently booked).
    `capital_at_risk` defaults to A's own (roll_selector.select_roll_target
    deliberately rolls to the SAME width, so A's max-loss is a reasonable
    estimate for B's until a more precise figure is available).

    `carried_legs`/`carried_value` are prepare_roll_combo's own fields
    (PreparedRollCombo.carried_legs/.carried_value, passed through
    PendingRollOrder by a future wiring wave) -- A's OTHER vertical, not
    part of this fill, transferred to B rather than left orphaned at the
    broker. W57: the P&L math below treats it as a mark-to-market leg that
    cancels exactly across the chain -- A realizes it as part of its close,
    B absorbs the same value into its own entry basis -- so
    fold_roll_chains' summed P&L still equals the real cash flows, and B's
    own TP%/stop math sees the sane whole-position entry credit instead of
    just the 2 rolled legs' credit (which a live stop could trip on
    immediately).

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

    opening_trade = state.get_trade_by_id(opening_trade_id)
    if opening_trade is None:
        log.error("roll_booking_opening_trade_missing", closing_trade_id=closing_trade.trade_id,
                  opening_trade_id=opening_trade_id)
        return False
    try:
        seeded_opening_legs = json.loads(opening_trade.legs) if opening_trade.legs else []
    except (ValueError, TypeError):
        log.error("roll_booking_opening_legs_unparseable", opening_trade_id=opening_trade_id)
        return False
    merged_legs = json.dumps(_persistable_legs(carried_legs) + _persistable_legs(seeded_opening_legs))

    # W57 (bug 2, this wave): closing_net/opening_net are TRADE-cost
    # convention (BUY+/SELL-, from the REVERSED closing actions -- see
    # split_roll_fill_price/leg_net_price's own docstrings) -- the cost to
    # buy back/establish the ROLLED vertical alone. carried_value is
    # POSITION-value convention (BUY+/SELL-, from the carried leg's own
    # UNCHANGED actions) for the vertical NOT in this fill. Converting it
    # to the same "cost to close" convention the rest of this codebase's
    # exit_price/_calculate_realized_pnl already use (executor.py:2695-2713)
    # means negating it: -carried_value. A's full exit_price is the rolled
    # vertical's real cost to close PLUS the carried vertical's equivalent
    # cost to close, i.e. closing_net - carried_value -- NOT closing_net
    # alone (the original bug: a $1.50 credit IC bought back for $0.80
    # booked +$230 instead of +$70, because the carried side's mark was
    # dropped from the close entirely and the entry/exit sign was inverted).
    exit_price = round(closing_net - carried_value, 4)
    entry = abs(closing_trade.entry_price)
    if closing_trade.strategy in CREDIT_STRATEGIES:
        realized_pnl = round((entry - exit_price) * 100 * closing_trade.quantity, 2)
    else:
        realized_pnl = round((exit_price - entry) * 100 * closing_trade.quantity, 2)

    # B's entry_price is the SAME position-value convention negated the
    # other way -- trades.entry_price is always the unsigned premium
    # (executor.py:2929-2931) -- so B's whole-position entry basis (new
    # vertical's real credit/debit PLUS the carried vertical's mark at the
    # moment of the roll) is abs(opening_net + carried_value): opening_net
    # is already position-value convention here (opening_legs carry their
    # REAL, non-reversed actions, unlike closing_legs).
    opening_entry_price = abs(round(opening_net + carried_value, 4))

    # Order matters: promote B FIRST. close_trade and this module's own
    # atomic promotion are two independent CAS calls (each internally
    # atomic, not a single shared transaction across both trades' rows --
    # composing them into one true cross-table transaction would mean
    # bypassing state.py's own exactly-once primitives, which this module
    # deliberately reuses rather than reimplements). Promoting B first means
    # a failure there (caught directly by this module's own tests) leaves A
    # untouched in CLOSING -- recoverable, still awaiting its real close.
    # Closing A first and having B's promotion fail would instead leave A
    # already CLOSED while B never existed as a real fill -- the position
    # would have silently vanished. The residual risk this ordering does
    # NOT eliminate (A's close failing AFTER B is already promoted, leaving
    # both a live B and a stuck-CLOSING A) is smaller and more visible --
    # registered as a known limitation for whoever wires this into
    # executor.py's live fill loop, not solved here.
    promoted = _promote_and_link_opening_trade(
        state, opening_trade_id, opening_entry_price, closing_trade.quantity,
        closing_trade.trade_id, merged_legs,
    )
    if not promoted:
        log.error("roll_booking_promote_refused", closing_trade_id=closing_trade.trade_id,
                  opening_trade_id=opening_trade_id)
        return False

    closed = state.close_trade(
        closing_trade.trade_id,
        exit_price=exit_price,
        realized_pnl=realized_pnl,
        commission=commission,
        exit_reason_detailed=f"rolled_to_{opening_trade_id}",
        from_statuses=(TradeStatus.CLOSING,),
    )
    if not closed:
        log.error("roll_booking_close_refused", closing_trade_id=closing_trade.trade_id,
                  opening_trade_id=opening_trade_id)
        return False

    state.set_trade_capital_at_risk(
        opening_trade_id,
        capital_at_risk if capital_at_risk is not None else closing_trade.capital_at_risk,
    )

    log.info("roll_booking_complete", closing_trade_id=closing_trade.trade_id,
             opening_trade_id=opening_trade_id, realized_pnl=realized_pnl)
    return True

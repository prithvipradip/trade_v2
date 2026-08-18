"""Trade executor — handles order lifecycle from signal to fill.

Responsible for:
- Converting signals to IBKR orders (single or combo)
- Submitting orders with proper contract qualification
- Tracking fills, partial fills, and cancellations
- Cancelling stale orders that haven't filled within timeout
- Recording trades in state with accurate fill information
"""

from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime

from ib_insync import Trade, TradeLogEntry

from ait.broker.contracts import ContractBuilder
from ait.broker.ibkr_client import IBKRClient
from ait.broker.orders import OrderBuilder
from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
from ait.risk.circuit_breaker import CircuitBreaker
from ait.strategies.base import CREDIT_STRATEGIES, Signal, SignalDirection
from ait.utils.logging import get_logger

log = get_logger("execution.executor")

# Order timeout: cancel unfilled orders after this many seconds
# Options markets move fast — stale orders get bad fills or block capital
import os as _os_t
# R7: 240s default — the mid-first reprice ladder needs 3 steps x 45s before
# the cancel; 90s cancelled orders before the marketable step could fire.
DEFAULT_ORDER_TIMEOUT = int(_os_t.environ.get("AIT_ENTRY_ORDER_TIMEOUT", "240"))

# R19 (audit finding executor.py:453): terminal fallback for the single-leg
# wide-spread reject. The 0.15 literal used to live inline in
# _execute_single_leg, silently shadowing the EXISTING config field
# options.max_bid_ask_spread_pct (config.yaml = 0.40, deliberately loosened
# because stale-quote spreads run wider on this delayed-data paper account).
# The scanner accepted 40% while the executor vetoed anything over 15% with
# only a WARNING — the wing_k silent-divergence pattern. The value now comes
# from settings when the caller threads it in; this constant only covers call
# sites constructed without settings so nothing breaks.
DEFAULT_MAX_SPREAD_PCT = 0.15

# R16: IBKR's generic combo contract id. Every BAG order reports, alongside its
# per-leg executions, ONE combo-level summary execution carrying this conId and
# the net combo price (negative when the combo nets a credit). Leg rows and the
# combo row are different units and must never be compared to each other — see
# _sweep_executions.
BAG_CON_ID = 28812380


# R5 audit: expected (n_sell, n_buy) leg shape per multi-leg strategy. A combo
# whose SELL/BUY counts disagree with its strategy name is refused — this is
# the only layer between strategy signal dicts and the wire.
_EXPECTED_LEG_SHAPE = {
    "iron_condor": (2, 2),
    "short_strangle": (2, 0),
    "long_straddle": (0, 2),
    "event_straddle": (0, 2),
    "bull_call_spread": (1, 1),
    "bear_put_spread": (1, 1),
    "calendar_spread": (1, 1),
}


def _validate_combo_legs(strategy_name: str, legs: list[dict]) -> str | None:
    """Return an error string if the combo legs are malformed, else None."""
    sells = buys = 0
    for leg in legs:
        action = str(leg.get("action", "")).upper()
        if action not in ("BUY", "SELL"):
            return f"invalid leg action {leg.get('action')!r}"
        if int(leg.get("conId", 0) or 0) <= 0:
            return "leg missing qualified conId"
        ratio = int(leg.get("ratio", 1) or 0)
        if ratio < 1 or ratio > 4:
            return f"suspicious leg ratio {ratio}"
        if action == "SELL":
            sells += 1
        else:
            buys += 1
    expected = _EXPECTED_LEG_SHAPE.get(strategy_name)
    if expected and (sells, buys) != expected:
        return (f"leg shape ({sells} SELL, {buys} BUY) does not match "
                f"{strategy_name} (expects {expected[0]} SELL, {expected[1]} BUY)")
    return None


def _ladder_limit(base_mag: float, full_offset: float, frac: float,
                  is_credit: bool) -> float:
    """R7: limit price at `frac` of the marketable offset from |mid|.

    Signed for the always-BUY combo convention: credit combos are negative
    (we accept `frac*offset` less credit than mid as we escalate), debit
    combos positive (we pay `frac*offset` more than mid).
    """
    off = full_offset * max(0.0, min(1.0, frac))
    if is_credit:
        return -max(0.01, round(base_mag - off, 2))
    return round(base_mag + off, 2)


def single_leg_entry_ladder(bid: float, ask: float) -> tuple[float, float, float]:
    """R14: reprice-ladder state for a single-leg DEBIT (long-option) entry.

    Single-leg entries used a PASSIVE sub-mid limit with NO escalation ladder,
    while combos escalate 0.25 -> 0.60 -> 1.00 of the marketable offset. So a
    single-leg order sat below the ask and was reconciled
    stale_pending_never_filled even on a tight, healthy quote (observed: a long
    put resting at 3.28 against a 3.32 ask, cancelled unfilled after 4 min).

    Arm the SAME ladder machinery combos use: base = mid, full_offset crosses
    from mid to the ask plus a small buffer, so at frac=1.0 the limit is
    marketable (fills against the ask) while step 0 (0.25x) still starts near
    mid for price improvement. Returns (base_mag, full_offset, debit_cap); the
    debit cap is the marketable ceiling so the ladder can reach the ask but
    never overshoot it.
    """
    mid = (bid + ask) / 2.0
    half_spread = max(0.0, (ask - bid) / 2.0)
    buffer = 0.02  # a couple ticks past the ask guarantees marketability at 1.0x
    offset = round(half_spread + buffer, 2)
    base_mag = round(mid, 2)
    debit_cap = round(base_mag + offset, 2)  # ~ ask + buffer
    return base_mag, offset, debit_cap


def combo_entry_limit(entry_price: float, is_credit: bool) -> tuple[float, float]:
    """Marketable limit for a multi-leg combo entry.

    A flat 5c off mid almost never fills a combo — the spread is routinely
    10-20% of mid, so orders sat unfilled and were reconciled as
    stale_pending_never_filled. Cross toward the fill by a PROPORTIONAL
    amount (15% of the quote, min 10c), bounded to a marketable LIMIT (not a
    market order) so the entry price stays capped.

    Returns (limit_price, offset). Credit combos quote NEGATIVE limits
    (always-BUY convention: negative = we collect).
    """
    offset = max(0.10, round(0.15 * abs(entry_price), 2))
    if is_credit:
        # Credit: we collect — accept less credit to get filled.
        return -max(0.01, entry_price - offset), offset
    # Debit: we pay — accept paying more to get filled.
    return entry_price + offset, offset


class PendingOrder:
    """Tracks a pending order with its submission time.

    R7: also carries the reprice-ladder state (base price, full marketable
    offset, current step) and the fill-quality context captured at placement
    (signal price, live combo mid, NBBO spread) for the executions ledger.
    """

    __slots__ = ("trade_id", "signal", "submitted_at", "contracts",
                 "base_price", "full_offset", "is_credit", "step",
                 "live_mid", "nbbo_spread", "debit_cap")

    def __init__(self, trade_id: str, signal: Signal, contracts: int,
                 base_price: float = 0.0, full_offset: float = 0.0,
                 is_credit: bool = False, live_mid: float = 0.0,
                 nbbo_spread: float = 0.0) -> None:
        self.trade_id = trade_id
        self.signal = signal
        self.submitted_at = time.time()
        self.contracts = contracts
        self.base_price = base_price
        self.full_offset = full_offset
        self.is_credit = is_credit
        self.step = 0
        self.live_mid = live_mid
        self.nbbo_spread = nbbo_spread
        # R8: risk-validated per-contract debit ceiling — the reprice ladder
        # must never step a debit limit above what the risk manager approved.
        self.debit_cap = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.submitted_at


class PendingExitOrder:
    """Tracks a pending exit/close order."""

    __slots__ = ("trade_id", "exit_reason", "estimated_pnl", "submitted_at")

    def __init__(self, trade_id: str, exit_reason: str, estimated_pnl: float) -> None:
        self.trade_id = trade_id
        self.exit_reason = exit_reason
        self.estimated_pnl = estimated_pnl
        self.submitted_at = time.time()

    @property
    def age_seconds(self) -> float:
        return time.time() - self.submitted_at


class TradeExecutor:
    """Executes trade signals by placing orders with IBKR."""

    def __init__(
        self,
        ibkr_client: IBKRClient,
        state: StateManager,
        circuit_breaker: CircuitBreaker,
        order_timeout: int = DEFAULT_ORDER_TIMEOUT,
        settings=None,
    ) -> None:
        self._ibkr = ibkr_client
        self._state = state
        self._circuit_breaker = circuit_breaker
        self._order_timeout = order_timeout
        # R19: the executor received NO config, so every threshold it enforced
        # had to be a literal — which is how the single-leg spread gate ended
        # up shadowing options.max_bid_ask_spread_pct. Optional so existing
        # call sites keep working; when settings ARE passed, config wins.
        self._settings = settings
        _opts = getattr(settings, "options", None)
        _cfg_spread = getattr(_opts, "max_bid_ask_spread_pct", None)
        try:
            self._max_spread_pct = (float(_cfg_spread) if _cfg_spread
                                    else DEFAULT_MAX_SPREAD_PCT)
        except (TypeError, ValueError):
            self._max_spread_pct = DEFAULT_MAX_SPREAD_PCT
        self._pending_orders: dict[int, PendingOrder] = {}  # order_id → PendingOrder
        self._pending_exit_orders: dict[int, PendingExitOrder] = {}  # order_id → PendingExitOrder
        # R13: exit orderIds we already asked the broker to cancel while they
        # were still WORKING (stale >300s) — one cancel per order, not per pass.
        self._exit_cancels_requested: set[int] = set()
        # R13: orderId -> (signal_price, live_mid, nbbo_spread) captured at
        # placement; survives pending-dict cleanup (see _sweep_executions).
        self._order_ctx_map: dict[int, tuple[float, float, float]] = {}
        self._order_trade_map: dict[int, str] = {}  # orderId → trade_id (survives cleanup)
        # R19: orderId -> permId, learned at placement and refreshed on every
        # pass while the order is still visible. permId is IBKR's PERMANENT,
        # cross-session order key: ib_insync's wrapper.connectionClosed() calls
        # reset() (0.9.86 wrapper.py), wiping self.trades, and orders recovered
        # afterwards via reqCompletedOrders routinely carry order.orderId=0
        # (they are keyed by permId). Without this map an entry that FILLED
        # during a disconnect matches nothing and reads as "not working".
        self._order_perm_map: dict[int, int] = {}
        self._perm_trade_map: dict[int, str] = {}  # permId → trade_id
        # R12 F2.2: strong references to event-spawned tasks. An unreferenced
        # asyncio task is garbage-collectable mid-flight — the event-driven
        # fill check could silently die between scheduling and execution.
        self._bg_tasks: set = set()
        self._fill_task_pending = False
        self._cf_running = False
        self._cf_rerun = False  # R12 F2.3: rerun-requested flag (no dropped passes)
        # R16: (fetched_at, orderIds | None) of the last AUTHORITATIVE
        # all-clients open-order snapshot. None means the broker could not be
        # asked — every consumer must DEFER on it, never conclude "gone".
        self._broker_open_ids_cache: tuple[float, set[int] | None] | None = None
        # R16: one operator page per (order_id, event); these conditions
        # persist across passes and must not re-page every 30s.
        self._stuck_order_pages: set[tuple[int, str]] = set()
        # R16: exit orderIds ever classified as belonging to ANOTHER clientId.
        # Sticky: the foreign stash gets pruned once the broker stops
        # reporting the order, and a wiped cache erases the clientId evidence
        # — but an exit order whose fill was never observable here must stay
        # on the broker-truth path, never fall back into the 900s revert.
        self._foreign_exit_orders: set[int] = set()

    async def execute_signal(self, signal: Signal, contracts: int) -> str | None:
        """Execute a trade signal. Returns trade_id on success, None on failure."""
        # INST-5 (institutional audit): defined-risk-only as an EXECUTION
        # gate, not a config policy. Undefined-risk (naked short premium)
        # orders are refused unless explicitly allowed — in paper we allow
        # them via run_orchestrator's default so the edge comparison runs;
        # at go-live the env is unset and the wings become a contractual,
        # not configured, loss floor.
        # R8: zero the combo pricing stash — it leaked the PREVIOUS combo's
        # base/offset into single-leg PendingOrders (single-leg path never
        # writes it), arming the reprice ladder with wrong prices.
        self._last_base_mag = 0.0
        self._last_full_offset = 0.0
        self._last_live_mid = 0.0
        self._last_nbbo_spread = 0.0
        self._last_debit_cap = 0.0

        import os as _os
        # R5 audit: covered_call included — no stock-ownership check exists
        # anywhere, so a CC fill would be a NAKED call mislabeled defined-risk.
        if (not signal.is_defined_risk
                or signal.strategy_name in ("short_strangle", "covered_call")) \
                and _os.environ.get("AIT_ALLOW_UNDEFINED_RISK") != "1":
            log.error("undefined_risk_refused_at_executor",
                      strategy=signal.strategy_name, symbol=signal.symbol)
            return None

        # INST-4: hard order-rate backstop. The daily budget bounds INTENTS
        # via the orchestrator; nothing used to bound a logic bug that spams
        # placements. Token bucket: >8 orders in a rolling 60s window is not
        # a strategy, it's a malfunction — refuse and scream.
        import time as _time
        _bucket = getattr(self, "_order_times", None)
        if _bucket is None:
            _bucket = self._order_times = []
        _now = _time.time()
        _bucket[:] = [t for t in _bucket if _now - t < 60]
        if len(_bucket) >= 8:
            log.critical("order_rate_limit_tripped", orders_last_60s=len(_bucket))
            return None
        _bucket.append(_now)

        # GOV (governance audit): market-hours enforcement at the EXECUTOR —
        # the scheduler gates the loop, but nothing stopped any other caller
        # from submitting outside RTH. Last line of defense.
        try:
            from ait.utils.time import is_market_open as _imo
            if not _imo() and _os.environ.get("AIT_ALLOW_AFTER_HOURS") != "1":
                log.error("order_refused_market_closed",
                          strategy=signal.strategy_name, symbol=signal.symbol)
                return None
        except Exception:  # noqa: BLE001 — never let the guard itself block
            pass

        if not await self._ibkr.ensure_connected():
            log.error("execution_failed", reason="IBKR not connected")
            self._circuit_breaker.record_api_failure()
            return None

        trade_id = f"T-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        try:
            # R12 chaos#1 (order-before-row): the PENDING trade row is written
            # BEFORE placeOrder. The old order (placeOrder first, row second)
            # left a crash window where a live broker order had NO local
            # record — the exact untracked-option scenario the reconciler
            # HALTs on. A PENDING row with no broker order is the benign
            # direction: it is swept/cancelled by the reconciler.
            legs_json = json.dumps([
                {
                    "strike": self._leg_fields(leg)[0],
                    "right": self._leg_fields(leg)[1],
                    "action": leg["action"],
                    "expiry": self._leg_fields(leg)[2],
                }
                for leg in signal.legs
            ]) if signal.legs else "[]"

            record = TradeRecord(
                trade_id=trade_id,
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                direction=(
                    TradeDirection.LONG
                    if signal.direction == SignalDirection.BULLISH
                    else TradeDirection.SHORT
                    if signal.direction == SignalDirection.BEARISH
                    else TradeDirection.LONG
                ),
                status=TradeStatus.PENDING,
                entry_time=datetime.now().isoformat(),
                entry_price=signal.entry_price,
                quantity=contracts,
                contract_type=self._get_contract_type(signal),
                strike=signal.contract.strike if signal.contract else None,
                expiry=str(signal.expiry) if signal.expiry else None,
                ml_confidence=signal.confidence,
                market_regime="",
                legs=legs_json,
            )
            self._state.record_trade(record)

            if signal.legs:
                trade = await self._execute_multi_leg(signal, contracts, trade_id)
            else:
                trade = await self._execute_single_leg(signal, contracts, trade_id)

            if trade is None:
                # R12 chaos#1 + Tier-A #1: placement failed/refused — retire
                # the pre-written PENDING row via CAS. CANCELLED is included
                # in the from-set so inner reject paths that already flipped
                # the row don't produce a spurious illegal-transition warning.
                self._state.transition(
                    trade_id,
                    (TradeStatus.PENDING, TradeStatus.CANCELLED),
                    TradeStatus.CANCELLED,
                )
                return None

            # Track pending order for fill monitoring
            if trade.order.orderId:
                self._pending_orders[trade.order.orderId] = PendingOrder(
                    trade_id=trade_id,
                    signal=signal,
                    contracts=contracts,
                    base_price=getattr(self, "_last_base_mag", 0.0),
                    full_offset=getattr(self, "_last_full_offset", 0.0),
                    is_credit=signal.strategy_name in self.CREDIT_STRATEGIES,
                    live_mid=getattr(self, "_last_live_mid", 0.0) or 0.0,
                    nbbo_spread=getattr(self, "_last_nbbo_spread", 0.0) or 0.0,
                )
                self._pending_orders[trade.order.orderId].debit_cap = \
                    getattr(self, "_last_debit_cap", 0.0) or 0.0
                # R7: orderId -> trade_id survives pending-dict cleanup so the
                # executions sweep can attribute late commission reports.
                if not hasattr(self, "_order_trade_map"):
                    self._order_trade_map = {}
                self._order_trade_map[trade.order.orderId] = trade_id
                # R19: capture the permId as soon as IBKR assigns one. It is
                # usually still 0 at placeOrder time (it arrives with the
                # openOrder callback), so _refresh_perm_ids re-reads it on
                # every pass — but capture here too for the fast path.
                self._remember_perm_id(trade.order.orderId,
                                       getattr(trade.order, "permId", 0))
                # R13 (shadow-referee break #7): fill-quality context must
                # ALSO survive cleanup. Since R11's event-driven fills, the
                # pending entry is booked and deleted by the fill event BEFORE
                # the next sweep pass — the sweep then inserted every
                # execution with signal_price/live_mid/nbbo_spread = 0 and
                # the slippage gate had literally no data.
                self._order_ctx_map[trade.order.orderId] = (
                    abs(signal.entry_price or 0),
                    getattr(self, "_last_live_mid", 0.0) or 0.0,
                    getattr(self, "_last_nbbo_spread", 0.0) or 0.0,
                )

            log.info(
                "trade_executed",
                trade_id=trade_id,
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                contracts=contracts,
                entry_price=signal.entry_price,
                order_id=trade.order.orderId,
            )

            self._circuit_breaker.record_api_success()
            return trade_id

        except Exception as e:
            log.error(
                "execution_error",
                trade_id=trade_id,
                symbol=signal.symbol,
                error=str(e),
            )
            # R12 chaos#1: retire the pre-written PENDING row (CAS; no-op with
            # a warning if the row was never written because record_trade
            # itself failed). CANCELLED in the from-set keeps this idempotent.
            try:
                self._state.transition(
                    trade_id,
                    (TradeStatus.PENDING, TradeStatus.CANCELLED),
                    TradeStatus.CANCELLED,
                )
            except Exception:  # noqa: BLE001 — cleanup must not mask the original error
                pass
            self._circuit_breaker.record_api_failure()
            return None

    async def _execute_single_leg(
        self, signal: Signal, contracts: int, trade_id: str
    ) -> Trade | None:
        """Execute a single-leg option order.

        Uses passive pricing when bid/ask is available for price improvement.
        Falls back to standard limit order at signal entry_price.
        """
        if not signal.contract:
            log.error("no_contract_in_signal", trade_id=trade_id)
            return None

        contract = ContractBuilder.option(
            symbol=signal.symbol,
            expiry=signal.contract.expiry,
            strike=signal.contract.strike,
            right=signal.contract.right,
        )

        qualified = await self._ibkr.qualify_contract(contract)
        if not qualified:
            log.error("contract_qualification_failed", trade_id=trade_id)
            return None

        # Try passive pricing if bid/ask available on the contract
        bid = getattr(signal.contract, "bid", 0) or 0
        ask = getattr(signal.contract, "ask", 0) or 0

        # Reject if the bid-ask spread is too wide — a stale/illiquid quote.
        # R19: the ceiling is options.max_bid_ask_spread_pct (config), NOT a
        # literal. The old inline 0.15 shadowed the config field: config.yaml
        # sets 0.40, so the scanner/chain filter admitted quotes the executor
        # then vetoed with only a WARNING, and tightening the config below
        # 0.15 did not tighten the executor either. See DEFAULT_MAX_SPREAD_PCT.
        max_spread_pct = getattr(self, "_max_spread_pct", DEFAULT_MAX_SPREAD_PCT)
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > max_spread_pct:
                log.warning("wide_spread_rejected", trade_id=trade_id,
                            bid=bid, ask=ask, spread_pct=f"{spread_pct:.1%}",
                            max_spread_pct=max_spread_pct)
                # R12 Tier-A #1: CAS PENDING->CANCELLED (row now exists —
                # chaos#1 writes it before placement, so this reject is a
                # real state write, not the historical no-op).
                self._state.transition(
                    trade_id, (TradeStatus.PENDING,), TradeStatus.CANCELLED)
                return None

        is_credit = signal.strategy_name in self.CREDIT_STRATEGIES
        if bid > 0 and ask > 0 and ask > bid and not is_credit:
            # R14: arm the marketable reprice ladder (was a passive sub-mid
            # limit that never escalated -> stale_pending_never_filled). Start
            # near mid for price improvement; _reprice_pending_entries steps it
            # to the ask by 90s if unfilled, honoring debit_cap. These stash
            # fields are read by execute_signal when it registers the
            # PendingOrder (same path combos use).
            base_mag, offset, debit_cap = single_leg_entry_ladder(bid, ask)
            self._last_base_mag = base_mag
            self._last_full_offset = offset
            self._last_debit_cap = debit_cap
            self._last_live_mid = round((bid + ask) / 2.0, 2)
            self._last_nbbo_spread = round(ask - bid, 4)
            init_limit = _ladder_limit(base_mag, offset,
                                       self._LADDER_STEPS[0], is_credit=False)
            order = OrderBuilder.limit(
                action=signal.action, quantity=contracts, limit_price=init_limit)
            log.info(
                "single_leg_entry_laddered",
                trade_id=trade_id, bid=bid, ask=ask,
                base=base_mag, offset=offset, init_limit=init_limit, cap=debit_cap,
            )
        else:
            order = OrderBuilder.limit(
                action=signal.action,
                quantity=contracts,
                limit_price=signal.entry_price,
            )

        # R17: tag the order with its trade_id so the reconciler can match a
        # working order to its trade exactly, instead of by symbol alone.
        order.orderRef = trade_id
        return await self._ibkr.place_order(qualified, order)

    async def _execute_multi_leg(
        self, signal: Signal, contracts: int, trade_id: str
    ) -> Trade | None:
        """Execute a multi-leg combo order (spreads, condors).

        Uses batch contract qualification for speed — qualifies all legs
        in a single IBKR call instead of one-at-a-time.
        """
        if not signal.legs:
            log.error("no_legs_in_signal", trade_id=trade_id)
            return None

        # Build all leg contracts at once. Legs come in two shapes:
        # {"contract": OptionContract, "action": ...} from most strategies,
        # or plain {"strike", "right", "expiry", "action"} dicts from
        # event_straddle/calendar — _leg_fields handles both (the old
        # leg["contract"] access KeyError'd and silently killed every
        # event_straddle/calendar signal).
        ibkr_contracts = []
        for leg in signal.legs:
            strike, right, expiry = self._leg_fields(leg)
            ibkr_contracts.append(ContractBuilder.option(
                symbol=signal.symbol,
                expiry=expiry,
                strike=strike,
                right=right,
            ))

        # Batch qualify all legs in one call
        qualified_list = await self._ibkr.qualify_contracts_batch(ibkr_contracts)

        qualified_legs = []
        for i, qualified in enumerate(qualified_list):
            if not qualified:
                strike, right, _ = self._leg_fields(signal.legs[i])
                log.error(
                    "leg_qualification_failed",
                    trade_id=trade_id,
                    strike=strike,
                    right=right,
                )
                return None

            qualified_legs.append({
                "conId": qualified.conId,
                "action": signal.legs[i]["action"],
                "ratio": signal.legs[i].get("ratio", 1),
            })

        # R5 audit CRITICAL: strategy leg dicts flowed verbatim to the wire —
        # nothing asserted leg-side integrity, and a single reversed leg turns
        # defined-risk into undefined-risk (and can be instantly marketable
        # under the always-BUY/signed-price convention). Validate structure
        # before building the BAG.
        _err = _validate_combo_legs(signal.strategy_name, qualified_legs)
        if _err:
            log.error("combo_legs_rejected", trade_id=trade_id,
                      strategy=signal.strategy_name, reason=_err)
            # R12 Tier-A #1: CAS PENDING->CANCELLED
            self._state.transition(
                trade_id, (TradeStatus.PENDING,), TradeStatus.CANCELLED)
            return None

        combo = ContractBuilder.combo(
            symbol=signal.symbol,
            legs=qualified_legs,
        )

        # Determine credit/debit from strategy cash-flow sign.
        # Leg counts are ambiguous (e.g. iron condor has 2 BUY + 2 SELL but is CREDIT).
        is_credit = signal.strategy_name in self.CREDIT_STRATEGIES

        # GOV-1 (governance audit): the primary strategy (iron condor) had
        # NO live-quote validation — the marketable limit derived purely from
        # the signal-time mid. Fetch the live combo NBBO once and refuse when
        # (a) the combo spread is disorderly, or (b) the signal price has
        # drifted far from the live mid (stale signal / fat finger).
        _live_mid_mag = None
        _live_spread = None
        try:
            self._ibkr.ib.reqMktData(combo, "", False, False)
            try:
                import asyncio as _aio
                await _aio.sleep(1.5)
                _t = self._ibkr.ib.ticker(combo)
                _bid = _t.bid if (_t and _t.bid == _t.bid) else None
                _ask = _t.ask if (_t and _t.ask == _t.ask) else None
            finally:
                try:
                    self._ibkr.ib.cancelMktData(combo)
                except Exception:
                    pass
            if _bid is not None and _ask is not None and _ask > _bid:
                _mid = (_bid + _ask) / 2
                _spread = abs(_ask - _bid)
                if abs(_mid) > 0.05 and _spread / abs(_mid) > 1.0:
                    log.warning("combo_spread_disorderly_rejected",
                                trade_id=trade_id, bid=_bid, ask=_ask)
                    # R12 Tier-A #1: CAS PENDING->CANCELLED
                    self._state.transition(
                        trade_id, (TradeStatus.PENDING,), TradeStatus.CANCELLED)
                    return None
                # combo quotes are signed; compare magnitudes
                if abs(_mid) > 0.05 and abs(abs(signal.entry_price) - abs(_mid)) / abs(_mid) > 0.35:
                    log.warning("combo_signal_price_stale_rejected",
                                trade_id=trade_id,
                                signal_px=signal.entry_price, live_mid=_mid)
                    # R12 Tier-A #1: CAS PENDING->CANCELLED
                    self._state.transition(
                        trade_id, (TradeStatus.PENDING,), TradeStatus.CANCELLED)
                    return None
                # R7: keep the live quote — it becomes the PRICING anchor
                # (was fetched for validation then thrown away) and the
                # fill-quality baseline in the executions ledger.
                _live_mid_mag = abs(_mid)
                _live_spread = _spread
        except Exception as _e:  # noqa: BLE001 — validation must not block on quote hiccups
            log.debug("combo_quote_validation_skipped", error=str(_e))

        # R7: anchor pricing to the LIVE combo mid when we have one — the
        # signal-time price can be minutes old. And start the reprice ladder
        # NEAR MID instead of fully marketable: step 0 concedes only 25% of
        # the marketable offset; check_fills escalates 60% -> 100% if
        # unfilled. Measured entry slippage was ~15% of credit; most of it
        # is recoverable patience.
        import os as _os_l
        _base_mag = _live_mid_mag if _live_mid_mag else abs(signal.entry_price)
        _, aggressive_offset = combo_entry_limit(_base_mag, is_credit)
        _start_frac = float(_os_l.environ.get("AIT_ENTRY_LADDER_START", "0.25"))
        limit_price = _ladder_limit(_base_mag, aggressive_offset,
                                    _start_frac, is_credit)
        # R5 audit: for debit combos the marketable cross (up to +15%) pushed
        # the worst-case cost past the max_loss the risk manager validated —
        # a systematic ~15% understatement. Cap the limit at the validated
        # per-contract loss so the risk check stays truthful.
        #
        # R17: signal.max_loss is ALWAYS a per-contract figure (every
        # strategy builds its signal at quantity=1 -- spreads.py/
        # straddles.py compute `max_loss = net_debit * 100`; SR-M6). limit_price
        # is likewise a per-share, per-contract price, unaffected by how
        # many contracts this order actually sizes to. Dividing by
        # `contracts` a second time shrank the cap by that factor for any
        # multi-contract order, clamping limit_price to an unfillable price
        # and silently killing the entry (reprice ladder re-clamps to the
        # same wrong cap every escalation, times out ~4 min later).
        _debit_cap = 0.0
        if not is_credit and signal.max_loss and contracts > 0:
            _validated = signal.max_loss / 100
            _debit_cap = round(_validated, 2)
            if limit_price > _validated > 0:
                limit_price = round(_validated, 2)
        self._last_debit_cap = _debit_cap
        log.info("combo_entry_priced", strategy=signal.strategy_name,
                 target=signal.entry_price, offset=aggressive_offset,
                 limit=limit_price, is_credit=is_credit)

        order = OrderBuilder.combo_limit(
            action="BUY",
            quantity=contracts,
            # R5 audit: float artifacts (2.07-0.31=1.7599999...) produce
            # off-tick prices IBKR rejects with error 110. Round to cents.
            limit_price=round(limit_price, 2),
        )

        # R7: stash pricing context for the PendingOrder created by the caller
        self._last_base_mag = _base_mag
        self._last_full_offset = aggressive_offset
        self._last_live_mid = _live_mid_mag
        self._last_nbbo_spread = _live_spread

        # R17: tag the order with its trade_id so the reconciler can match a
        # working order to its trade exactly, instead of by symbol alone.
        order.orderRef = trade_id
        return await self._ibkr.place_order(combo, order)

    def attach_fill_events(self) -> None:
        """R11 (R9 latency finding): fill DETECTION measured at ~33s because
        it rode the 30s polling loop. Subscribe to ib_insync's orderStatus
        events so a fill triggers check_fills within ~1s (debounced so a
        4-leg burst coalesces into one pass). Safe: ib_insync fires events on
        the loop thread; we only create a task."""
        import asyncio as _aio

        def _on_order_status(trade) -> None:
            try:
                if trade.orderStatus.status != "Filled":
                    return
                if getattr(self, "_fill_task_pending", False):
                    return
                self._fill_task_pending = True

                async def _run() -> None:
                    try:
                        await _aio.sleep(1.0)  # coalesce leg-fill bursts
                        self._fill_task_pending = False
                        filled, exits = await self.check_fills_safe()
                        if filled or exits:
                            log.info("event_driven_fill_processed",
                                     entries=len(filled), exits=len(exits))
                        # exits booked here still reach daily stats/Thompson
                        # via the orchestrator's completed-exit callback if
                        # one is registered
                        cb = getattr(self, "_completed_exits_cb", None)
                        if cb and exits:
                            await cb(exits)
                    except Exception as _e:  # noqa: BLE001
                        self._fill_task_pending = False
                        log.warning("event_fill_check_failed", error=str(_e))

                # R12 F2.1: create_task failure used to leave
                # _fill_task_pending wedged True FOREVER — every later fill
                # event returned early and event-driven detection silently
                # died for the session. Reset the flag in the except.
                # R12 F2.2: keep a strong reference to the task (unreferenced
                # tasks are GC-cancellable mid-flight); discard on done.
                # get_running_loop (not get_event_loop): ib_insync fires this
                # on the loop thread, and get_event_loop is deprecated /
                # wrong-loop-prone outside it.
                try:
                    _task = _aio.get_running_loop().create_task(_run())
                    _bg = getattr(self, "_bg_tasks", None)
                    if _bg is None:
                        _bg = self._bg_tasks = set()
                    _bg.add(_task)
                    _task.add_done_callback(_bg.discard)
                except Exception as _sched_err:  # noqa: BLE001
                    self._fill_task_pending = False
                    log.warning("event_fill_task_schedule_failed",
                                error=str(_sched_err))
            except Exception:  # noqa: BLE001 — event handler must never raise
                pass

        try:
            self._ibkr.ib.orderStatusEvent += _on_order_status
            log.info("fill_events_attached")
        except Exception as _e:  # noqa: BLE001
            log.warning("fill_events_attach_failed", error=str(_e))

    async def check_fills_safe(self) -> tuple[list[str], list[dict]]:
        """R11: reentrancy guard — the event-driven path and the 30s monitor
        can both want check_fills; concurrent runs would double-process
        completed exits.

        R12 F2.3: the busy-guard used to DROP the concurrent request — a fill
        event arriving during a monitor pass was simply never processed until
        the next 30s poll (exactly the latency the event path exists to
        remove). Now the loser sets a rerun flag and the running pass loops
        until no rerun is requested (bounded, so an event storm can't spin
        this forever — the 30s monitor is still behind us).
        """
        if getattr(self, "_cf_running", False):
            self._cf_rerun = True  # R12 F2.3: queue a rerun instead of dropping
            return [], []
        self._cf_running = True
        try:
            filled_all: list[str] = []
            exits_all: list[dict] = []
            for _pass in range(4):  # initial pass + up to 3 queued reruns
                self._cf_rerun = False
                filled, exits = await self.check_fills()
                filled_all.extend(filled)
                exits_all.extend(exits)
                if not getattr(self, "_cf_rerun", False):
                    break
            return filled_all, exits_all
        finally:
            self._cf_running = False

    async def check_fills(self) -> tuple[list[str], list[dict]]:
        """Check for filled/cancelled/timed-out orders and update state.

        Returns (filled_entry_trade_ids, completed_exits) where each
        completed exit is a dict with trade_id, exit_price, realized_pnl,
        exit_reason.
        """
        filled = []
        cancelled = []

        # CONNECTION GUARD (deep-audit MP-F2b): get_open_orders() and
        # get_all_trades() both return [] when disconnected, which used to
        # make every working order look vanished -> mass mis-cancel. Skip
        # the whole pass on a dead connection; nothing is lost by waiting.
        if not self._ibkr.connected:
            log.warning("check_fills_skipped_disconnected")
            return filled, []

        # R16: classify tracked EXIT orders against the stash / cached clientId
        # BEFORE the stash is pruned — pruning is what erases the evidence,
        # and an exit order whose fill was never observable in this session
        # must stay on the broker-truth path instead of falling back into the
        # 900s revert. Then release stashed ids the broker no longer reports
        # working, so a genuinely dead order stops forcing "pending" forever.
        self._classify_foreign_exit_orders(self._ibkr.get_open_orders())
        await self._prune_foreign_stash()

        # 1. Cancel stale orders that have exceeded timeout
        await self._cancel_stale_orders()

        # 2. Check status of all pending orders
        open_trades = self._ibkr.get_open_orders()
        all_trades = self._ibkr.get_all_trades() if hasattr(self._ibkr, 'get_all_trades') else []

        # R19: refresh permIds while the session can still see these orders.
        # permId is 0 at placeOrder time and the Trade cache is wiped by the
        # next disconnect — this is the only window in which the permanent
        # key of a working order can be learned.
        self._refresh_perm_ids(all_trades)

        for order_id, pending in list(self._pending_orders.items()):
            still_open = any(t.order.orderId == order_id for t in open_trades)

            if still_open:
                continue  # Order is still working

            # Order is no longer open — determine what happened
            status = self._determine_fill_status(order_id, all_trades, pending)

            # R16 (post-reconnect amnesia): a "cancelled" verdict that rests
            # ONLY on the order being absent from this session's caches is not
            # evidence of anything — a reconnect whose open-orders sync timed
            # out silently leaves those caches empty while the order is still
            # working (and its later fill creates no Trade object at all).
            # Confirm against the broker before writing CANCELLED; defer when
            # the broker cannot confirm. A verdict read off an explicit
            # terminal status in all_trades is trustworthy and skips this.
            if status == "cancelled" and not any(
                    t.order.orderId == order_id for t in all_trades):
                if not await self._confirm_order_not_working(order_id, all_trades):
                    continue

            if status == "filled":
                # Get actual fill price if available
                actual_price = self._get_fill_price(order_id, all_trades, pending)
                # R12 F4.4: pass the TRUE filled quantity from orderStatus —
                # a PARTIAL pass may have shrunk trades/open_positions to the
                # partial quantity, and the old FILLED write never restored it.
                true_qty = self._get_filled_quantity(order_id, all_trades, pending)
                self._update_trade_filled(
                    pending, actual_price,
                    filled_qty=true_qty if true_qty > 0 else pending.contracts,
                )
                filled.append(pending.trade_id)
                log.info(
                    "trade_filled",
                    trade_id=pending.trade_id,
                    symbol=pending.signal.symbol,
                    expected_price=pending.signal.entry_price,
                    actual_price=actual_price,
                    slippage=actual_price - pending.signal.entry_price,
                )

            elif status == "partial":
                filled_qty = self._get_filled_quantity(order_id, all_trades, pending)
                partial_price = self._get_fill_price(order_id, all_trades, pending)
                self._update_trade_partial(pending, filled_qty, partial_price)

                # Terminal partial (order cancelled/inactive at IBKR with
                # fills): booking is done — stop tracking so this doesn't
                # re-book every cycle (deep-audit MP-F2a follow-through).
                _terminal = any(
                    t.order.orderId == order_id
                    and t.orderStatus.status.lower() in ("cancelled", "inactive", "apicancelled")
                    for t in all_trades
                )
                if _terminal:
                    del self._pending_orders[order_id]
                    filled.append(pending.trade_id)  # partial contracts are live
                    continue

                # If partial fill has been sitting > 30s, cancel the remainder
                # to avoid orphaned legs and stale prices.
                # R19: this searched open_trades — but the branch is only
                # reachable when the order is ABSENT from open_trades (the
                # `if still_open: continue` above), so the loop never matched
                # and the cancel was never transmitted: a dead safety path
                # that logged a cancellation which never happened, leaving the
                # remainder riding at stale prices until the 240s timeout.
                if pending.age_seconds > 30:
                    log.warning(
                        "partial_fill_cancelling_remainder",
                        trade_id=pending.trade_id,
                        symbol=pending.signal.symbol,
                        filled=filled_qty,
                        requested=pending.contracts,
                        age_seconds=pending.age_seconds,
                    )
                    self._blind_cancel_once(
                        order_id, "partial_remainder_cancel_unsent",
                        open_trades, all_trades,
                        trade_id=pending.trade_id,
                        filled=filled_qty,
                        requested=pending.contracts,
                        detail="could not transmit the remainder cancel; the "
                               "unfilled balance is still working at the "
                               "broker at a stale price",
                    )
                else:
                    log.warning(
                        "trade_partial_fill",
                        trade_id=pending.trade_id,
                        symbol=pending.signal.symbol,
                        filled=filled_qty,
                        requested=pending.contracts,
                    )

            elif status == "cancelled":
                # R19: a cancel verdict on a row that is already PARTIAL is
                # refused (contracts are live) — the trade is handed back as
                # a live position instead, exactly like the terminal-partial
                # branch above.
                if self._update_trade_cancelled(pending):
                    cancelled.append(pending.trade_id)
                    log.info(
                        "trade_cancelled",
                        trade_id=pending.trade_id,
                        symbol=pending.signal.symbol,
                        age_seconds=pending.age_seconds,
                    )
                else:
                    filled.append(pending.trade_id)  # partial contracts are live

            if status in ("filled", "cancelled"):
                del self._pending_orders[order_id]

        if cancelled:
            log.info("orders_cancelled", count=len(cancelled), trade_ids=cancelled)

        # 3. Check pending EXIT orders — finalize CLOSING → CLOSED with real fill price
        completed_exits = []
        for order_id, pending_exit in list(self._pending_exit_orders.items()):
            # R16 (foreign / frozen exit orders): when the exit order belongs
            # to another clientId — a fallback-clientId reconnect, or an order
            # adopted from the previous process at startup reconcile — the
            # local cache is meaningless in BOTH directions, and each
            # direction had its own money bug:
            #   present-forever (ib_insync inserts the foreign order frozen at
            #     'Submitted' and never updates it) pinned the trade in CLOSING
            #     with an order nobody in this session can cancel or reprice,
            #     and made the 900s zombie cap unreachable;
            #   absent (stash fetch failed, or a reconnect wiped the cache)
            #     let that 900s cap revert CLOSING->FILLED while the original
            #     close was still working at the broker — the monitor then
            #     places a SECOND close and both fill (R12 F4.1's incident).
            # Resolve such orders against a fresh all-clients broker snapshot,
            # and DEFER (keep CLOSING, page once) whenever the broker cannot
            # be asked or cannot tell us whether it filled.
            _foreign_exits = getattr(self, "_foreign_exit_orders", None)
            if _foreign_exits is None:
                _foreign_exits = self._foreign_exit_orders = set()
            if (order_id in _foreign_exits
                    or self._is_foreign_session_order(order_id, open_trades)):
                _foreign_exits.add(order_id)
                _ids = await self._broker_open_order_ids()
                if _ids is None:
                    self._page_stuck_order(
                        order_id, "exit_order_foreign_state_unknown",
                        trade_id=pending_exit.trade_id,
                        age_seconds=int(pending_exit.age_seconds),
                        detail="exit order belongs to another clientId and the "
                               "broker cannot be asked; holding CLOSING")
                    continue
                if order_id in _ids:
                    # Still working under the other session. Cross-client
                    # cancels are rejected, so escalate instead of looping on
                    # a cancel that can never land.
                    if pending_exit.age_seconds > 300:
                        self._page_stuck_order(
                            order_id, "exit_order_foreign_uncancellable",
                            trade_id=pending_exit.trade_id,
                            age_seconds=int(pending_exit.age_seconds),
                            detail="a previous clientId's close order is still "
                                   "working; cancel/reprice needs the operator")
                    continue
                # Broker says it is no longer working. The frozen local
                # snapshot cannot say whether it FILLED or died, and guessing
                # is a money bug either way (a phantom close, or a re-armed
                # duplicate close). Hold CLOSING and let reconcile settle it
                # from positions + the broker's own leg P&L.
                self._forget_foreign_order(order_id)
                # Once reconcile HAS settled the trade, stop tracking it so
                # the tracker (and its page) do not live forever.
                try:
                    _tr = self._state.get_trade_by_id(pending_exit.trade_id)
                except Exception:  # noqa: BLE001
                    _tr = None
                if _tr is not None and getattr(_tr, "status", None) != TradeStatus.CLOSING:
                    log.warning("exit_order_foreign_tracker_released",
                                order_id=order_id,
                                trade_id=pending_exit.trade_id,
                                status=str(getattr(_tr, "status", "")))
                    _foreign_exits.discard(order_id)
                    del self._pending_exit_orders[order_id]
                    continue
                self._page_stuck_order(
                    order_id, "exit_order_foreign_gone_unbookable",
                    trade_id=pending_exit.trade_id,
                    age_seconds=int(pending_exit.age_seconds),
                    detail="foreign exit order left the broker's open-order "
                           "book; filled-vs-cancelled is unobservable in this "
                           "session — held CLOSING for reconcile")
                continue

            still_open = any(t.order.orderId == order_id for t in open_trades)
            if still_open:
                # R13 (07-13 incident regression work): a WORKING exit was
                # skipped unconditionally here, which made the >300s stale
                # branch and the 900s zombie cap below UNREACHABLE for the
                # common case — a resting limit the market ran away from.
                # The position wedged in CLOSING (CAS then blocks every new
                # exit) with a stale-priced order at the broker. Now: request
                # ONE cancel at >300s and keep tracking. The next pass gets
                # the terminal verdict: filled -> booked (cancel lost the
                # race, exactly one close), cancelled -> CAS revert to FILLED
                # for a clean marketable re-trigger. Never revert while the
                # broker shows the order working (a halt can reject cancels;
                # reverting would re-arm the duplicate-exit incident).
                if (pending_exit.age_seconds > 300
                        and order_id not in self._exit_cancels_requested):
                    log.warning(
                        "exit_order_stale_working_cancel_requested",
                        trade_id=pending_exit.trade_id,
                        order_id=order_id,
                        age_seconds=int(pending_exit.age_seconds),
                    )
                    try:
                        for t in open_trades:
                            if t.order.orderId == order_id:
                                self._ibkr.ib.cancelOrder(t.order)
                                self._exit_cancels_requested.add(order_id)
                                break
                    except Exception as e:
                        log.warning("exit_stale_cancel_failed",
                                    order_id=order_id, error=str(e))
                continue
            self._exit_cancels_requested.discard(order_id)

            # Give exit orders time to appear in IBKR's trade list before checking
            # Without this, fresh orders get falsely marked as "cancelled"
            if pending_exit.age_seconds < 10:
                continue

            exit_status = self._determine_exit_fill_status(order_id, all_trades)

            if exit_status == "filled":
                actual_exit_price = self._get_exit_fill_price(order_id, all_trades)

                # Calculate realized P&L from actual fill price when known.
                # 0.0 is a real price (closed at worthless); only None means
                # "no fill data" — then fall back to the estimate and FLAG it
                # so downstream analytics can exclude these rows.
                trade_record = self._state.get_trade_by_id(pending_exit.trade_id)
                exit_reason = pending_exit.exit_reason
                if trade_record and actual_exit_price is not None:
                    # Multi-leg DEBIT exits fill via a reversed-legs combo,
                    # whose as-defined price is NEGATIVE (we receive credit).
                    # _calculate_realized_pnl expects exit_price = value
                    # received per contract, so flip the sign. Credit
                    # strategies buy back at a positive as-defined price —
                    # already the "cost to close" the formula expects.
                    if (
                        trade_record.contract_type in ("spread", "iron_condor")
                        and trade_record.strategy not in CREDIT_STRATEGIES
                    ):
                        actual_exit_price = -actual_exit_price
                    realized_pnl = self._calculate_realized_pnl(
                        trade_record, actual_exit_price
                    )
                else:
                    realized_pnl = pending_exit.estimated_pnl
                    exit_reason = f"{exit_reason}|pnl_estimate_no_fill_price"
                    log.warning(
                        "exit_fill_price_missing",
                        trade_id=pending_exit.trade_id,
                        order_id=order_id,
                        estimated_pnl=realized_pnl,
                    )

                # R12 Tier-A #1: close_trade is now a CAS from an open status.
                # A refused close (trade already CLOSED by another path) must
                # NOT emit a completed-exit event — the orchestrator callback
                # would double-book daily stats/Thompson on it.
                closed_ok = self._state.close_trade(
                    trade_id=pending_exit.trade_id,
                    exit_price=actual_exit_price if actual_exit_price is not None else 0.0,
                    realized_pnl=realized_pnl,
                    exit_reason_detailed=exit_reason,
                )
                if closed_ok is not False:  # True or legacy None both count
                    completed_exits.append({
                        "trade_id": pending_exit.trade_id,
                        "exit_price": actual_exit_price if actual_exit_price is not None else 0.0,
                        "realized_pnl": realized_pnl,
                        "exit_reason": exit_reason,
                    })
                    log.info(
                        "exit_order_filled",
                        trade_id=pending_exit.trade_id,
                        actual_exit_price=actual_exit_price,
                        realized_pnl=realized_pnl,
                    )
                else:
                    log.warning(
                        "exit_close_refused_already_terminal",
                        trade_id=pending_exit.trade_id,
                        order_id=order_id,
                    )
                del self._pending_exit_orders[order_id]

            elif exit_status == "partial":
                # R12 F4.3 (exit partial-fill-then-cancel): the exit order
                # died with SOME contracts closed. The old code fell into the
                # "cancelled" branch and reverted the trade to FILLED at its
                # ORIGINAL quantity — the next exit then re-sold the already-
                # closed contracts, flipping the book into a naked short.
                # Book the closed portion (partial exit + quantity reduction,
                # mirroring the entry-partial path) and hand the REMAINDER
                # back to the portfolio monitor via CLOSING->FILLED.
                self._book_partial_exit(order_id, pending_exit, all_trades,
                                        completed_exits)
                del self._pending_exit_orders[order_id]

            elif exit_status == "cancelled":
                # Exit order was cancelled/rejected — revert to FILLED so
                # portfolio manager will re-trigger an exit next cycle.
                # R12 Tier-A #1 (F4.5): CAS CLOSING->FILLED ONLY. The blind
                # UPDATE could resurrect a trade another path had already
                # CLOSED (restart race) — a re-managed ghost position whose
                # "exit" would open a fresh naked position.
                self._state.transition(
                    pending_exit.trade_id,
                    (TradeStatus.CLOSING,),
                    TradeStatus.FILLED,
                )
                log.warning(
                    "exit_order_cancelled",
                    trade_id=pending_exit.trade_id,
                    age_seconds=pending_exit.age_seconds,
                )
                del self._pending_exit_orders[order_id]

            elif exit_status == "pending" and pending_exit.age_seconds > 300:
                # Exit stuck pending 5+ min — request cancel but KEEP TRACKING
                # (audit R2/H1): the old code cancelled, reverted to FILLED and
                # dropped the tracker in one shot. If the cancel raced a fill
                # (order fills as the cancel arrives), the portfolio re-triggered
                # a SECOND exit on an already-closed position — flipping it into
                # a fresh naked position. Now: send cancel, keep the tracker;
                # the next check sees a terminal state (filled -> book properly,
                # cancelled -> revert to FILLED for a clean re-trigger). Hard
                # cap at 900s so a zombie order can't wedge the position in
                # CLOSING forever.
                # R19: same dead path as the partial remainder above — this
                # branch is reachable only when the order is INVISIBLE to this
                # session (the working case returns at `still_open`), so
                # iterating open_trades could never find it and no cancel was
                # ever sent; IBKRClient.cancel_order(int) fails the same way
                # ("no matching trade found"). Cancel by orderId on the wire,
                # and page if even that cannot be transmitted.
                log.warning(
                    "exit_order_stale_pending_cancel_requested",
                    trade_id=pending_exit.trade_id,
                    age_seconds=int(pending_exit.age_seconds),
                )
                self._blind_cancel_once(
                    order_id, "exit_stale_pending_cancel_unsent",
                    open_trades, all_trades,
                    trade_id=pending_exit.trade_id,
                    age_seconds=int(pending_exit.age_seconds),
                    detail="a close order invisible to this session could not "
                           "be cancelled; the position stays CLOSING until "
                           "the broker or the operator resolves it",
                )
                if pending_exit.age_seconds > 900:
                    # R16: the zombie cap consulted NOTHING before reverting —
                    # not the foreign stash, not the broker. If the close
                    # order is in fact still working (under a previous
                    # clientId, or simply invisible after a cache wipe),
                    # reverting to FILLED re-arms the monitor and a second
                    # close goes out alongside the first: both fill and the
                    # condor flips into an inverse structure. Revert ONLY on a
                    # broker snapshot that answered and does not list it.
                    _ids = await self._broker_open_order_ids()
                    if _ids is None or order_id in _ids:
                        self._page_stuck_order(
                            order_id, "exit_zombie_revert_deferred",
                            trade_id=pending_exit.trade_id,
                            age_seconds=int(pending_exit.age_seconds),
                            reason=("broker_unreachable" if _ids is None
                                    else "order_still_working_at_broker"),
                            detail="held CLOSING rather than re-arming a "
                                   "duplicate close",
                        )
                        continue
                    log.error(
                        "exit_order_zombie_reverting",
                        trade_id=pending_exit.trade_id,
                        age_seconds=int(pending_exit.age_seconds),
                        broker_confirmed_not_working=True,
                    )
                    # R12 Tier-A #1 (F4.5): CAS CLOSING->FILLED only — never
                    # resurrect a trade that reached CLOSED via another path.
                    self._state.transition(
                        pending_exit.trade_id,
                        (TradeStatus.CLOSING,),
                        TradeStatus.FILLED,
                    )
                    del self._pending_exit_orders[order_id]

        return filled, completed_exits

    _LADDER_STEPS = (0.25, 0.60, 1.00)   # fractions of the marketable offset
    _LADDER_STEP_SECS = 45               # escalate an unfilled entry every 45s

    async def _reprice_pending_entries(self) -> None:
        """R7: step unfilled entry combos toward marketable (mid-first ladder).

        Step 0 was placed at 25% of the offset; at 45s escalate to 60%, at
        90s to 100% (the old immediate-marketable price). Modification uses
        the same orderId, so queue priority is only lost on the price change.
        """
        for order_id, pending in list(self._pending_orders.items()):
            if not pending.base_price or not pending.full_offset:
                continue
            target_step = min(int(pending.age_seconds // self._LADDER_STEP_SECS),
                              len(self._LADDER_STEPS) - 1)
            if target_step <= pending.step:
                continue
            pending.step = target_step
            new_limit = _ladder_limit(pending.base_price, pending.full_offset,
                                      self._LADDER_STEPS[target_step],
                                      pending.is_credit)
            if not pending.is_credit and pending.debit_cap > 0:
                new_limit = min(new_limit, pending.debit_cap)  # R8: honor risk cap
            try:
                for t in self._ibkr.ib.openTrades():
                    if t.order.orderId == order_id:
                        t.order.lmtPrice = round(new_limit, 2)
                        self._ibkr.ib.placeOrder(t.contract, t.order)
                        log.info("entry_ladder_reprice",
                                 trade_id=pending.trade_id, step=target_step,
                                 limit=round(new_limit, 2))
                        break
            except Exception as e:  # noqa: BLE001
                log.warning("entry_ladder_reprice_failed",
                            trade_id=pending.trade_id, error=str(e))

    @staticmethod
    def _is_bag_row(fill) -> bool:
        """R16: True for the combo-level summary execution of a BAG order
        (IBKR's generic combo conId), as opposed to a per-leg execution."""
        c = getattr(fill, "contract", None)
        if c is None:
            return False
        return ((getattr(c, "conId", 0) or 0) == BAG_CON_ID
                or getattr(c, "secType", "") == "BAG")

    def _sweep_executions(self) -> None:
        """R7: persist every broker fill + commission to the executions
        ledger. Runs each check_fills pass; upserts refresh commissions that
        arrive after the fill. This is the ground truth the PF verdict needs
        (trades.commission was guessed at $0.65/leg before).

        R16 (fill-quality semantics): signal_price/live_mid describe the
        COMBO — one net price for the whole structure — and were stamped onto
        every per-leg row as well. Every slippage consumer then computed
        |per-leg price - combo mid|, e.g. a 2.09 short put against a 4.24
        condor mid = "49.9% slippage", and the go-live gate (median <= 8% of
        credit) was permanently, unfixably red while a genuinely bad fill
        would have been invisible in the same noise. Fill-quality context is
        now written ONLY where the comparison is defined:
          * combo orders: the BAG summary row alone, whose price is the net
            combo price. It is stored as a MAGNITUDE so it shares
            trades.entry_price's unsigned convention — the raw as-defined
            value is negative for a credit, which made ABS(price - live_mid)
            read 198% instead of the true 1.6%. The per-leg rows keep their
            raw signed broker prices (all P&L reconstruction uses those and
            already excludes the BAG conId) but carry no combo context.
          * single-leg orders: unchanged — price and mid are the same unit.
        """
        try:
            for t in self._ibkr.ib.trades():
                oid = t.order.orderId
                t_perm = getattr(t.order, "permId", 0) or 0
                self._remember_perm_id(oid, t_perm)
                is_combo = getattr(t.contract, "secType", "") == "BAG"
                for f in t.fills:
                    if not f.execution or not f.execution.execId:
                        continue
                    # R19: attribute per FILL, not per Trade. A trade
                    # recovered via reqCompletedOrders after a reconnect can
                    # carry order.orderId = 0 (IBKR keys completed orders by
                    # permId), which used to write every one of its fills with
                    # order_id 0 and trade_id '' — the ledger declared "the
                    # ground truth the PF verdict needs" losing commission and
                    # P&L attribution for exactly the fills around a
                    # connection loss. Execution objects carry the true
                    # orderId and permId; prefer whichever key resolves.
                    f_oid = int(getattr(f.execution, "orderId", 0) or 0)
                    f_perm = int(getattr(f.execution, "permId", 0) or 0)
                    row_oid = int(oid or 0) or f_oid
                    row_perm = int(t_perm or 0) or f_perm
                    self._remember_perm_id(row_oid, row_perm)
                    trade_id, sig_price, live_mid, nbbo = self._ledger_context(
                        row_oid, row_perm)
                    px = float(f.execution.price or 0)
                    # R16: see the docstring — combo context belongs on the
                    # combo row only, in the combo row's own unit.
                    if is_combo:
                        if self._is_bag_row(f):
                            px = abs(px)
                            row_sig, row_mid, row_nbbo = sig_price, live_mid, nbbo
                        else:
                            row_sig = row_mid = row_nbbo = 0.0
                    else:
                        row_sig, row_mid, row_nbbo = sig_price, live_mid, nbbo
                    comm = 0.0
                    rpnl = 0.0
                    if f.commissionReport:
                        comm = float(f.commissionReport.commission or 0.0)
                        rp = f.commissionReport.realizedPNL
                        rpnl = float(rp) if rp and rp == rp and abs(rp) < 1e12 else 0.0
                    self._state.record_execution(
                        exec_id=f.execution.execId,
                        order_id=row_oid,
                        perm_id=row_perm,
                        trade_id=trade_id,
                        symbol=t.contract.symbol,
                        con_id=getattr(f.contract, "conId", 0) or 0,
                        side=f.execution.side,
                        shares=float(f.execution.shares or 0),
                        price=px,
                        exec_time=self._normalize_exec_time(f.execution.time),
                        commission=comm,
                        realized_pnl=rpnl,
                        signal_price=row_sig,
                        live_mid=row_mid,
                        nbbo_spread=row_nbbo,
                    )
        except Exception as e:  # noqa: BLE001
            log.debug("executions_sweep_failed", error=str(e))

    def _ledger_context(self, order_id: int,
                        perm_id: int = 0) -> tuple[str, float, float, float]:
        """R19: (trade_id, signal_price, live_mid, nbbo_spread) for a ledger
        row, resolved by orderId and — when that fails — by permId.

        Split out of _sweep_executions so attribution can be done per FILL:
        the orderId on a reconnect-recovered completed order is 0, and the
        permId map (learned while the order was still visible) is then the
        only route back to our trade_id.
        """
        omap = getattr(self, "_order_trade_map", {})
        trade_id = omap.get(order_id, "")
        if not trade_id:
            _p = getattr(self, "_pending_exit_orders", {}).get(order_id)
            trade_id = getattr(_p, "trade_id", "") if _p else ""
        if not trade_id and perm_id:
            trade_id = self._perm_maps()[1].get(int(perm_id), "")
        pend = (getattr(self, "_pending_orders", {}).get(order_id)
                or getattr(self, "_pending_exit_orders", {}).get(order_id))
        # R13 (shadow-referee break #7): the pending entry is gone by the
        # first sweep after an event-driven fill (R11) — fall back to the
        # placement-time context map, which survives cleanup.
        ctx = getattr(self, "_order_ctx_map", {}).get(order_id, (0.0, 0.0, 0.0))
        if pend is not None and hasattr(pend, "signal"):
            sig_price = abs(getattr(pend.signal, "entry_price", 0) or 0)
        else:
            sig_price = ctx[0]
        live_mid = (getattr(pend, "live_mid", 0.0) or 0.0) or ctx[1]
        nbbo = (getattr(pend, "nbbo_spread", 0.0) or 0.0) or ctx[2]
        return trade_id, sig_price, live_mid, nbbo

    @staticmethod
    def _normalize_exec_time(t) -> str:
        """R16: store exec_time as TRUE UTC ISO.

        ib_insync hands back either an aware datetime or a naive gateway
        wall-clock stamp that str() then mislabels '+00:00' — the ledger held
        mixed semantics (+4h on swept rows vs true UTC on restated rows),
        silently skewing every time-based join by 4 hours. Aware -> convert
        to UTC; naive -> localize as LOCAL gateway wall time, then convert.
        """
        if not t:
            return ""
        try:
            from datetime import datetime, timezone
            if isinstance(t, str):
                return t
            if t.tzinfo is not None:
                return t.astimezone(timezone.utc).isoformat()
            return t.astimezone().astimezone(timezone.utc).isoformat()
        except Exception:  # noqa: BLE001
            return str(t)

    async def _cancel_stale_orders(self) -> None:
        """Cancel orders that have been pending longer than the timeout."""
        await self._reprice_pending_entries()
        self._sweep_executions()
        for order_id, pending in list(self._pending_orders.items()):
            if pending.age_seconds > self._order_timeout:
                # R16: a foreign-session order has no Trade object here, so
                # cancel_order() can only fail ("no matching trade found") —
                # it logged an error on every 30s pass for the life of the
                # process. The order is unreachable from this session: say so
                # once and let the stash prune / reconcile resolve it.
                if self._in_foreign_stash(order_id):
                    self._page_stuck_order(
                        order_id, "stale_entry_order_foreign_uncancellable",
                        trade_id=pending.trade_id,
                        age_seconds=int(pending.age_seconds),
                        detail="entry order is working under another clientId; "
                               "this session cannot cancel it")
                    continue
                log.info(
                    "cancelling_stale_order",
                    trade_id=pending.trade_id,
                    symbol=pending.signal.symbol,
                    age_seconds=int(pending.age_seconds),
                    timeout=self._order_timeout,
                )
                try:
                    await self._ibkr.cancel_order(order_id)
                except Exception as e:
                    log.warning("cancel_failed", order_id=order_id, error=str(e))

    # R16: an authoritative all-clients snapshot is a broker round trip; a
    # single check_fills pass can consult it from several places (stash prune,
    # entry cancel verdicts, exit resolution, the 900s zombie cap), so cache
    # it for a few seconds. Passes are 30s apart, so this never hides a change.
    _BROKER_SNAPSHOT_TTL = 5.0

    async def _broker_open_order_ids(self) -> set[int] | None:
        """R16: orderIds working at the BROKER across ALL clientIds, or None
        when the broker could not be asked.

        This is the only trustworthy answer to "is this order still working?".
        ib_insync's caches are not: they are wiped by any disconnect, and
        orders owned by another clientId sit in them frozen at 'Submitted'
        forever (no master clientId is configured, so no updates arrive).
        Both failure modes are indistinguishable from a real disappearance,
        which is how working entries were mass-flipped to CANCELLED and how a
        CLOSING trade could be reverted while its close order was still live.

        None means UNKNOWN. Callers must defer, never act.
        """
        now = time.time()
        cached = getattr(self, "_broker_open_ids_cache", None)
        if cached is not None and (now - cached[0]) < self._BROKER_SNAPSHOT_TTL:
            return cached[1]
        ids: set[int] | None = None
        try:
            getter = getattr(self._ibkr, "get_all_open_order_ids", None)
            if getter is not None:
                ids = await getter()
            else:
                # Legacy/shim client without the strict probe: get_all_open_trades
                # returns None only when it truly cannot answer.
                trades_getter = getattr(self._ibkr, "get_all_open_trades", None)
                if trades_getter is not None:
                    trades = await trades_getter()
                    ids = (None if trades is None else
                           {int(t.order.orderId) for t in trades
                            if getattr(t.order, "orderId", 0)})
        except Exception as e:  # noqa: BLE001 — an unanswered probe is UNKNOWN
            log.warning("broker_open_orders_probe_failed", error=str(e)[:200])
            ids = None
        self._broker_open_ids_cache = (now, ids)
        return ids

    def _in_foreign_stash(self, order_id: int) -> bool:
        """R16: quiet membership test (``_is_foreign_open_order`` logs)."""
        return order_id in (getattr(self._ibkr, "foreign_open_order_ids", None) or set())

    def _note_foreign_order(self, order_id: int, reason: str) -> None:
        """R16: record that this orderId is working at the broker but is NOT
        observable in this session, so every later pass refuses cancel
        verdicts for it instead of re-deriving the discovery."""
        try:
            stash = getattr(self._ibkr, "foreign_open_order_ids", None)
            if stash is None or order_id in stash:
                return
            stash.add(order_id)
            log.warning("foreign_open_order_noted", order_id=order_id, reason=reason)
        except Exception:  # noqa: BLE001
            pass

    def _forget_foreign_order(self, order_id: int) -> None:
        """R16: the broker no longer reports this order working — release it
        so the normal fill/cancel resolution can finish the trade."""
        try:
            stash = getattr(self._ibkr, "foreign_open_order_ids", None)
            if stash is not None and order_id in stash:
                stash.discard(order_id)
                log.warning("foreign_open_order_released", order_id=order_id)
        except Exception:  # noqa: BLE001
            pass

    def _is_foreign_session_order(self, order_id: int, trades) -> bool:
        """R16: True when this order cannot be observed from this session.

        Two independent signals:
          * it is in the foreign stash (a clientId-change reconnect saw it
            working under the previous session), or
          * the cached Trade object carries a DIFFERENT clientId — ib_insync
            builds those from reqAllOpenOrders as snapshots and never updates
            them again, so their 'Submitted' status is meaningless. The stash
            alone missed this case: _should_stash_foreign is skipped on a
            same-id reconnect, and a startup reconcile poisons the cache
            before any stash exists.
        """
        if self._in_foreign_stash(order_id):
            return True
        my_id = getattr(self._ibkr, "client_id", None)
        try:
            my_id = int(my_id) if my_id else 0
        except (TypeError, ValueError):
            my_id = 0
        if my_id <= 0:
            return False
        for t in trades:
            if getattr(t.order, "orderId", None) != order_id:
                continue
            other = getattr(t.order, "clientId", None)
            try:
                other = int(other) if other else 0
            except (TypeError, ValueError):
                other = 0
            if other > 0 and other != my_id:
                self._note_foreign_order(
                    order_id, f"cached trade is a frozen snapshot owned by "
                              f"clientId {other} (this session is {my_id})")
                return True
            return False
        return False

    def _page_stuck_order(self, order_id: int, event: str, **fields) -> None:
        """R16: page the operator ONCE per (order, event). These states
        persist across passes; re-paging every 30s is how alarm fatigue
        buries the page that matters."""
        key = (order_id, event)
        pages = getattr(self, "_stuck_order_pages", None)
        if pages is None:
            pages = self._stuck_order_pages = set()
        if key in pages:
            return
        pages.add(key)
        log.critical(event, order_id=order_id, **fields)

    def _classify_foreign_exit_orders(self, trades) -> None:
        """R16: record which tracked exit orders belong to another clientId.

        Sticky by design and evaluated BEFORE the stash is pruned: once an
        exit order's fill is unobservable from this session, no later cache
        state may quietly return it to the guess-from-cache path."""
        pending_exits = getattr(self, "_pending_exit_orders", None) or {}
        if not pending_exits:
            return
        foreign = getattr(self, "_foreign_exit_orders", None)
        if foreign is None:
            foreign = self._foreign_exit_orders = set()
        for oid in list(pending_exits):
            if oid not in foreign and self._is_foreign_session_order(oid, trades):
                foreign.add(oid)

    async def _prune_foreign_stash(self) -> None:
        """R16: the stash was add-only, so an order that genuinely died kept
        forcing 'pending' for the life of the process. Re-validate it against
        an authoritative snapshot; prune ONLY on a successful answer."""
        if not (getattr(self._ibkr, "foreign_open_order_ids", None) or set()):
            return
        ids = await self._broker_open_order_ids()
        if ids is None:
            return  # unknown -> keep the protective stash intact
        pruner = getattr(self._ibkr, "prune_foreign_open_orders", None)
        if pruner is not None:
            pruner(ids)
            return
        stash = self._ibkr.foreign_open_order_ids
        for oid in sorted(stash - ids):
            self._forget_foreign_order(oid)

    # ------------------------------------------------------------------
    # R19: permId bookkeeping + execution-level fill evidence.
    #
    # Every "did this order die?" question in this file used to be answered
    # with `t.order.orderId == order_id` against THIS SESSION's Trade cache.
    # ib_insync 0.9.86 wipes that cache on any disconnect
    # (wrapper.connectionClosed -> reset()), and orders it recovers afterwards
    # via reqCompletedOrders routinely carry order.orderId = 0 because IBKR
    # keys completed orders by permId. So an entry that FILLED during an
    # outage matched nothing, read as "not working", and was booked CANCELLED
    # — a status neither the portfolio monitor nor the reconciler re-adopts —
    # while four condor legs were LIVE at the broker.
    # ------------------------------------------------------------------

    def _perm_maps(self) -> tuple[dict, dict]:
        """R19: (orderId->permId, permId->trade_id), created on demand.

        Lazily built so instances constructed via ``__new__`` in tests (and
        any legacy pickled/partial executor) never AttributeError inside the
        sweep's blanket except, which would silently drop the whole ledger.
        """
        pm = getattr(self, "_order_perm_map", None)
        if pm is None:
            pm = self._order_perm_map = {}
        tm = getattr(self, "_perm_trade_map", None)
        if tm is None:
            tm = self._perm_trade_map = {}
        return pm, tm

    def _remember_perm_id(self, order_id, perm_id) -> None:
        """R19: record IBKR's PERMANENT order key for a tracked orderId."""
        try:
            oid = int(order_id or 0)
            pid = int(perm_id or 0)
        except (TypeError, ValueError):
            return
        if oid <= 0 or pid <= 0:
            return
        pm, tm = self._perm_maps()
        pm[oid] = pid
        trade_id = getattr(self, "_order_trade_map", {}).get(oid, "")
        if not trade_id:
            _pe = getattr(self, "_pending_exit_orders", {}).get(oid)
            trade_id = getattr(_pe, "trade_id", "") if _pe else ""
        if trade_id:
            tm[pid] = trade_id

    def _refresh_perm_ids(self, trades) -> None:
        """R19: re-read permIds for tracked orders while this session can
        still see them. permId is 0 at placeOrder time (it arrives with the
        openOrder callback), so placement alone never captures it — and after
        the cache wipe it is the only link from a recovered execution back to
        our trade."""
        tracked = set(getattr(self, "_pending_orders", {}))
        tracked |= set(getattr(self, "_pending_exit_orders", {}))
        if not tracked:
            return
        for t in trades or []:
            try:
                oid = int(getattr(t.order, "orderId", 0) or 0)
            except (TypeError, ValueError):
                continue
            if oid in tracked:
                self._remember_perm_id(oid, getattr(t.order, "permId", 0))

    def _broker_fill_evidence(self, order_id: int, pending=None) -> dict | None:
        """R19: did this order actually EXECUTE at the broker?

        "Is the order still working?" is NOT the question a cancel verdict
        needs — a FILLED order is also not working. Executions are the durable
        evidence: ``Fill.execution`` carries the TRUE orderId and permId even
        when the parent (completed-order) Trade reports orderId 0, and
        ib.fills() survives what the Trade cache does not.

        Returns None when there is no evidence of a fill, else a dict with
        ``units`` (COMBO units — a 4-leg condor filling once is 1, not 4),
        ``price`` (as-defined signed net for a combo, unsigned average for a
        single leg; None when unknown), ``source`` and ``n_fills``.
        """
        try:
            oid = int(order_id or 0)
        except (TypeError, ValueError):
            return None
        perm_id = self._perm_maps()[0].get(oid, 0)

        matches: list = []
        try:
            getter = getattr(getattr(self._ibkr, "ib", None), "fills", None)
            for f in (getter() or []) if getter is not None else []:
                ex = getattr(f, "execution", None)
                if ex is None or not getattr(ex, "execId", ""):
                    continue
                try:
                    f_oid = int(getattr(ex, "orderId", 0) or 0)
                    f_pid = int(getattr(ex, "permId", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if f_oid == oid or (perm_id > 0 and f_pid == perm_id):
                    matches.append(f)
        except Exception as e:  # noqa: BLE001 — an unreadable ledger is "unknown"
            log.warning("broker_fill_evidence_probe_failed",
                        order_id=oid, error=str(e)[:200])
            matches = []

        if matches:
            n_legs = 1
            legs = getattr(getattr(pending, "signal", None), "legs", None)
            if legs:
                n_legs = max(1, len(legs))
            bag_units = 0.0
            bag_price: float | None = None
            leg_shares = 0.0
            leg_cost = 0.0
            net = 0.0
            for f in matches:
                ex = f.execution
                shares = float(getattr(ex, "shares", 0) or 0)
                price = float(getattr(ex, "price", 0) or 0)
                if self._is_bag_row(f):
                    bag_units += shares
                    if bag_price is None:
                        bag_price = price
                    continue
                leg_shares += shares
                leg_cost += price * shares
                net += (1.0 if getattr(ex, "side", "") == "SLD" else -1.0) * price * shares
            units = bag_units if bag_units > 0 else (
                leg_shares / n_legs if leg_shares > 0 else 0.0)
            price = bag_price
            if price is None and units > 0 and leg_shares > 0:
                # Same unit convention as _reconstruct_bag_price / _get_fill_price:
                # combos are the as-defined signed net per combo unit, single
                # legs the unsigned share-weighted average.
                price = (-net / units) if n_legs > 1 else (leg_cost / leg_shares)
            return {"units": units, "price": price,
                    "source": "ib_fills", "n_fills": len(matches)}

        # Second source: the on-disk executions ledger. ib_insync's
        # wrapper.reset() clears fills too, and while connectAsync does
        # re-request them (reqExecutionsAsync), that request can time out like
        # any other — the ledger already persisted every fill swept on earlier
        # passes, so it still answers for fills that predate the outage.
        trade_id = getattr(pending, "trade_id", "") or \
            getattr(self, "_order_trade_map", {}).get(oid, "")
        if trade_id:
            n = 0
            try:
                raw = self._state.count_executions(trade_id)
                # Strict: a non-numeric answer (stub/mock state) is NOT
                # evidence — inventing a fill would be as bad as missing one.
                if isinstance(raw, int) and not isinstance(raw, bool):
                    n = raw
                elif isinstance(raw, float):
                    n = int(raw)
            except Exception:  # noqa: BLE001 — no ledger answer is not evidence
                n = 0
            if n > 0:
                return {"units": 0.0, "price": None,
                        "source": "executions_ledger", "n_fills": n}
        return None

    def _cancel_verdict(self, order_id: int, pending=None) -> str:
        """R19: the ONLY place an entry 'cancelled' verdict is minted.

        FAIL-SAFE direction: broker executions outrank cache absence. A hit
        means the contracts are LIVE and the trade must be booked FILLED so
        the portfolio monitor manages it — never CANCELLED, which nothing
        re-adopts.
        """
        evidence = self._broker_fill_evidence(order_id, pending)
        if evidence is not None:
            log.critical(
                "cancel_verdict_overridden_by_broker_fills",
                order_id=order_id,
                trade_id=getattr(pending, "trade_id", ""),
                source=evidence["source"],
                n_fills=evidence["n_fills"],
                units=evidence["units"],
                note="order executed at the broker but is absent from this "
                     "session's trade cache (reconnect amnesia / completed "
                     "order with orderId 0) — booking FILLED, not CANCELLED",
            )
            return "filled"
        # R12 F3.1: never issue a cancel verdict for an order the broker
        # showed WORKING under another clientId.
        if self._is_foreign_open_order(order_id):
            return "pending"
        return "cancelled"

    def _cancel_order_anywhere(self, order_id: int, *trade_lists) -> bool:
        """R19: transmit a cancel for ``order_id`` even when this session
        holds no Trade object for it.

        Two safety paths in check_fills searched ONLY ``open_trades`` for the
        order — yet both branches are reachable exclusively when the order is
        ABSENT from open_trades (they sit after ``if still_open: continue``),
        so the loop could never match and the cancel was never transmitted:
        the 30s partial-remainder cancel logged
        'partial_fill_cancelling_remainder' for a cancellation that never
        happened, and the >300s stale-pending exit cancel was a no-op for
        exactly the cache-invisible orders it exists to clear.
        IBKRClient.cancel_order(int) cannot help either — it resolves the id
        against ib.trades() and fails 'no matching trade found'. The wire
        message needs only the orderId (ib_insync IB.cancelOrder calls
        client.cancelOrder(order.orderId)), so fall back to a bare Order.
        """
        ib = getattr(self._ibkr, "ib", None)
        if ib is None:
            return False
        for lst in trade_lists:
            for t in (lst or []):
                try:
                    if int(getattr(t.order, "orderId", 0) or 0) != int(order_id):
                        continue
                except (TypeError, ValueError):
                    continue
                try:
                    ib.cancelOrder(t.order)
                    return True
                except Exception as e:  # noqa: BLE001 — try the bare-order path
                    log.warning("cancel_via_cached_trade_failed",
                                order_id=order_id, error=str(e)[:200])
                    break
        try:
            from ib_insync import Order as _Order
            bare = _Order()
            bare.orderId = int(order_id)
            perm = self._perm_maps()[0].get(int(order_id), 0)
            if perm:
                bare.permId = perm
            ib.cancelOrder(bare)
            log.warning("cancel_sent_for_cache_invisible_order",
                        order_id=order_id, perm_id=perm,
                        note="no Trade object in this session; cancelled by "
                             "orderId on the wire")
            return True
        except Exception as e:  # noqa: BLE001
            log.error("cancel_transmit_failed", order_id=order_id,
                      error=str(e)[:200])
            return False

    def _blind_cancel_once(self, order_id: int, event: str, *trade_lists,
                           **fields) -> bool:
        """R19: send at most ONE cancel per order for the cache-invisible
        paths, and page LOUDLY when the transmit fails — a dead safety path
        must never sit silently in the code."""
        sent_set = getattr(self, "_blind_cancels_sent", None)
        if sent_set is None:
            sent_set = self._blind_cancels_sent = set()
        if order_id in sent_set:
            return True
        if self._cancel_order_anywhere(order_id, *trade_lists):
            sent_set.add(order_id)
            return True
        self._page_stuck_order(order_id, event, **fields)
        return False

    async def _confirm_order_not_working(self, order_id: int, all_trades: list) -> bool:
        """R16: may a "vanished" order be declared CANCELLED?

        Only when an authoritative all-clients snapshot answered AND does not
        list it AND this session can see orders at all. A same-clientId
        reconnect whose open-orders sync silently timed out (ib_insync only
        LOGS that timeout and still reports a successful connect) leaves
        trades()/openTrades() EMPTY, which the old code read as "every working
        entry was cancelled" — flipping live broker orders to CANCELLED
        locally and untracking their later fills.

        R19: "not working" is not the same question as "did not fill". Ask
        the executions first — a filled order is also not working, and after a
        cache wipe that is exactly how a live 4-leg condor got booked
        CANCELLED. Evidence of a fill is an unconditional refusal.
        """
        _ev = self._broker_fill_evidence(order_id)
        if _ev is not None:
            log.critical(
                "cancel_verdict_refused_order_has_fills",
                order_id=order_id, source=_ev["source"], n_fills=_ev["n_fills"],
                note="broker executions exist for this orderId/permId; it "
                     "filled, it was not cancelled",
            )
            return False
        ids = await self._broker_open_order_ids()
        if ids is None:
            log.warning(
                "cancel_verdict_deferred_broker_unreachable",
                order_id=order_id,
                note="cannot confirm the order is gone; treating as pending",
            )
            return False
        if order_id in ids:
            self._note_foreign_order(
                order_id, "working at the broker but absent from this "
                          "session's trade cache")
            return False
        if not all_trades:
            log.warning(
                "cancel_verdict_deferred_empty_session_snapshot",
                order_id=order_id,
                note="this session's trade cache is empty (post-reconnect "
                     "amnesia); a fill would be invisible here",
            )
            return False
        self._forget_foreign_order(order_id)
        return True

    def _is_foreign_open_order(self, order_id: int) -> bool:
        """R12 F3.1 / chaos#4B: True when this orderId was seen working at the
        broker under ANOTHER clientId (stashed at fallback-reconnect time).
        Such orders are invisible to this session's trade list — their
        absence is NOT evidence of a cancel."""
        foreign = getattr(self._ibkr, "foreign_open_order_ids", None) or set()
        if order_id in foreign:
            log.warning(
                "cancel_verdict_refused_foreign_order",
                order_id=order_id,
                note="order is working under a previous clientId session; "
                     "treating as pending, not cancelled",
            )
            return True
        return False

    def _determine_fill_status(
        self, order_id: int, all_trades: list, pending: PendingOrder
    ) -> str:
        """Determine whether a completed order was filled, partially filled, or cancelled."""
        # Try to find the trade in IBKR's completed trades
        for trade in all_trades:
            if trade.order.orderId == order_id:
                status = trade.orderStatus.status.lower()
                filled_qty = trade.orderStatus.filled or 0
                if status in ("filled",):
                    return "filled"
                elif status in ("cancelled", "inactive", "apicancelled"):
                    # FILLED-QTY FIRST (deep-audit MP-F2a): a cancelled order
                    # with fills (partial-then-cancel-remainder) previously
                    # returned "cancelled", flipping the trade to CANCELLED —
                    # a status neither the portfolio monitor nor the
                    # reconciler ever re-adopts — while the filled contracts
                    # were LIVE at IBKR with no stop/TP/expiry management.
                    if filled_qty > 0:
                        return "partial"
                    # R12 F3.1 / R19: _cancel_verdict is the single gate —
                    # it refuses to cancel an order with broker executions
                    # and refuses for orders working under another clientId.
                    return self._cancel_verdict(order_id, pending)
                elif status in ("submitted", "presubmitted"):
                    return "pending"  # Still working

                remaining = trade.orderStatus.remaining or 0
                if filled_qty > 0 and remaining > 0:
                    return "partial"
                elif filled_qty > 0:
                    return "filled"
                else:
                    return self._cancel_verdict(order_id, pending)

        # NOT-FOUND HARDENING (deep-audit MP-F2b): the entry path used to
        # assume "cancelled" whenever the order wasn't in get_all_trades() —
        # but that list is empty on any connection blip, so a mid-cycle
        # disconnect flipped EVERY working entry to CANCELLED while live at
        # IBKR. Mirror the exit path: treat young/unknown as still pending.
        if pending.age_seconds < 30:
            return "pending"
        # R12 F3.1: an order placed before a fallback-clientId reconnect is
        # ABSENT from this session's trade list by construction — that is the
        # mass-false-CANCELLED path, not a cancel.
        # R19: and an order that FILLED during a disconnect is absent for the
        # same reason (wrapper.reset() wiped the cache; the recovered
        # completed order carries orderId 0) — _cancel_verdict checks the
        # executions before it will say "cancelled".
        return self._cancel_verdict(order_id, pending)

    @staticmethod
    def _reconstruct_bag_price(trade) -> float | None:
        """Side-aware net price for a BAG from per-leg executions.

        (sold proceeds - bought costs) per contract, negated to IBKR's
        as-defined convention (negative = combo nets a credit). Shared by
        entry and exit fill reconstruction (audit R2 — the entry path
        previously lacked this and silently fell back to the signal price).
        """
        # R19: divide by what actually FILLED, not what was ORDERED. `net` is
        # summed over real executions, so pairing it with totalQuantity
        # reconstructs filled/ordered of the true price — a 1-of-2 condor fill
        # came back at -3.06 instead of -6.12 per combo. That value is booked
        # as entry_price (and as the exit price / partial-exit price), so the
        # error propagates into every downstream P&L, stop and target.
        ordered = trade.order.totalQuantity or 0
        try:
            filled_units = float(getattr(trade.orderStatus, "filled", 0) or 0)
        except (AttributeError, TypeError, ValueError):
            filled_units = 0.0
        qty = filled_units if filled_units > 0 else ordered
        if not trade.fills or qty <= 0:
            return None
        net = 0.0
        for f in trade.fills:
            sign = 1.0 if f.execution.side == "SLD" else -1.0
            net += sign * f.execution.price * f.execution.shares
        # UNIT FIX (deep-audit MP-F1): ib_insync `execution.shares` for
        # options is the CONTRACT count, not shares — dividing by 100·qty
        # made every reconstructed combo price 100x too small (a fresh
        # strangle would mark at -9900% and instantly stop out on fiction).
        # price-per-combo-unit = premium net / combo quantity.
        return -net / qty

    def _get_fill_price(
        self, order_id: int, all_trades: list, pending: PendingOrder
    ) -> float:
        """Get the actual fill price for an entry order."""
        for trade in all_trades:
            if trade.order.orderId == order_id:
                avg_price = trade.orderStatus.avgFillPrice
                if avg_price is not None and not math.isnan(avg_price) and avg_price != 0:
                    return avg_price
                # BAG fallback: reconstruct from per-leg executions (mirrors
                # the exit path) instead of masking slippage with the signal
                # price.
                if getattr(trade.contract, "secType", "") == "BAG":
                    bag = self._reconstruct_bag_price(trade)
                    if bag is not None:
                        return bag
                elif trade.fills:
                    total_cost = sum(f.execution.price * f.execution.shares for f in trade.fills)
                    total_shares = sum(f.execution.shares for f in trade.fills)
                    if total_shares > 0:
                        return total_cost / total_shares
                if (trade.orderStatus.filled or 0) > 0:
                    return 0.0

        # R19: an entry recovered from the executions (the order is not in
        # all_trades after a reconnect) still has a REAL price — reconstruct
        # it from the fills instead of booking the optimistic signal price,
        # which would flow into entry_price and every downstream P&L.
        evidence = self._broker_fill_evidence(order_id, pending)
        if evidence is not None and evidence.get("price") is not None:
            log.warning("entry_fill_price_from_broker_executions",
                        trade_id=pending.trade_id, order_id=order_id,
                        price=evidence["price"], source=evidence["source"])
            return float(evidence["price"])

        log.warning("entry_fill_price_missing_using_signal",
                    trade_id=pending.trade_id, order_id=order_id)
        return pending.signal.entry_price  # last resort — flagged above

    def _get_filled_quantity(
        self, order_id: int, all_trades: list, pending: PendingOrder
    ) -> int:
        """Get the actual filled quantity for a partial fill."""
        for trade in all_trades:
            if trade.order.orderId == order_id:
                return int(trade.orderStatus.filled or 0)
        # R19: not in this session's cache — fall back to the executions
        # (COMBO units), so a fill recovered after a reconnect is booked at
        # the quantity that actually filled, never 0.
        evidence = self._broker_fill_evidence(order_id, pending)
        if evidence is not None and evidence.get("units", 0) > 0:
            return int(round(float(evidence["units"])))
        return 0

    def _determine_exit_fill_status(self, order_id: int, all_trades: list) -> str:
        """Determine whether an exit order filled, partially filled, or was cancelled.

        R12 F4.3: mirrors the entry path's FILLED-QTY-FIRST logic. An exit
        cancelled WITH fills used to return a clean "cancelled" — the caller
        reverted the trade to FILLED at its original quantity and the next
        exit OVERSOLD the already-closed contracts (naked short). Now it
        returns "partial" so the closed slice is booked.
        """
        for trade in all_trades:
            if trade.order.orderId == order_id:
                status = trade.orderStatus.status.lower()
                filled_qty = trade.orderStatus.filled or 0
                if status in ("filled",):
                    return "filled"
                elif status in ("cancelled", "inactive", "apicancelled"):
                    # R12 F4.3: filled-qty first — dead order with fills is a
                    # partial exit, not a clean cancel.
                    if filled_qty > 0:
                        return "partial"
                    # R12 F3.1: refuse cancel verdicts for foreign-session orders.
                    return "pending" if self._is_foreign_open_order(order_id) \
                        else "cancelled"
                elif status in ("submitted", "presubmitted"):
                    return "pending"

                remaining = trade.orderStatus.remaining or 0
                if filled_qty > 0 and remaining > 0:
                    return "partial"
                elif filled_qty > 0:
                    return "filled"
                elif self._is_foreign_open_order(order_id):  # R12 F3.1
                    return "pending"
                return "cancelled"

        # Order not found in IBKR trade list yet — assume still pending
        # rather than falsely cancelling fresh orders
        return "pending"

    def _get_exit_fill_price(self, order_id: int, all_trades: list) -> float | None:
        """Get the actual fill price for an exit order.

        Returns the SIGNED as-defined price: combo (BAG) fills are negative
        when the as-defined legs net a credit — the normal case when closing
        a debit position via reversed legs. Single contracts are unsigned
        (IBKR convention). Returns None only when no fill data is available;
        0.0 is a LEGITIMATE price (closed at worthless).
        """
        for trade in all_trades:
            if trade.order.orderId != order_id:
                continue
            avg_price = trade.orderStatus.avgFillPrice
            if avg_price and not math.isnan(avg_price):
                return avg_price
            # Fallback: reconstruct from per-leg executions
            if trade.fills:
                if getattr(trade.contract, "secType", "") == "BAG":
                    bag = self._reconstruct_bag_price(trade)
                    if bag is not None:
                        return bag
                else:
                    total_cost = sum(f.execution.price * f.execution.shares for f in trade.fills)
                    total_shares = sum(f.execution.shares for f in trade.fills)
                    if total_shares > 0:
                        return total_cost / total_shares
            # Order is known to IBKR but reported no usable price; a
            # filled combo can legitimately average to ~0.0
            if (trade.orderStatus.filled or 0) > 0:
                return 0.0
            return None
        return None

    def _book_partial_exit(
        self,
        order_id: int,
        pending_exit: PendingExitOrder,
        all_trades: list,
        completed_exits: list[dict],
    ) -> None:
        """R12 F4.3: book an exit order that terminated with a PARTIAL fill.

        The closed portion gets a real partial-exit record (quantity, price,
        fill-derived P&L — mirroring the orchestrator's single-leg partial
        path: record_partial_exit + update_trade_quantity), and the REMAINING
        contracts revert CLOSING->FILLED so the portfolio monitor re-triggers
        an exit for them. The old behavior reverted the WHOLE trade to FILLED
        at its original quantity — the re-triggered exit then oversold the
        already-closed contracts into a naked short.
        """
        trade_id = pending_exit.trade_id
        trade_record = self._state.get_trade_by_id(trade_id)
        filled_units = 0
        for t in all_trades:
            if t.order.orderId == order_id:
                filled_units = int(t.orderStatus.filled or 0)
                break
        if trade_record is None or filled_units <= 0:
            # No basis to book — safest is a clean re-trigger of the whole
            # exit (CAS so a CLOSED trade can never be resurrected).
            self._state.transition(
                trade_id, (TradeStatus.CLOSING,), TradeStatus.FILLED)
            log.warning("exit_partial_unbookable_reverted",
                        trade_id=trade_id, order_id=order_id,
                        filled_units=filled_units)
            return

        exit_price = self._get_exit_fill_price(order_id, all_trades)
        # Same sign convention as the full-fill path: multi-leg DEBIT exits
        # fill via a reversed-legs combo whose as-defined price is negative.
        if (exit_price is not None
                and trade_record.contract_type in ("spread", "iron_condor")
                and trade_record.strategy not in CREDIT_STRATEGIES):
            exit_price = -exit_price

        if filled_units >= trade_record.quantity:
            # The "partial" verdict covered the whole remaining position
            # (e.g. cancel raced the last fill) — book a normal full close.
            exit_reason = pending_exit.exit_reason
            if exit_price is not None:
                realized_pnl = self._calculate_realized_pnl(trade_record, exit_price)
            else:
                realized_pnl = pending_exit.estimated_pnl
                exit_reason = f"{exit_reason}|pnl_estimate_no_fill_price"
            closed_ok = self._state.close_trade(
                trade_id=trade_id,
                exit_price=exit_price if exit_price is not None else 0.0,
                realized_pnl=realized_pnl,
                exit_reason_detailed=exit_reason,
            )
            if closed_ok is not False:  # R12 Tier-A #1: no event on refused close
                completed_exits.append({
                    "trade_id": trade_id,
                    "exit_price": exit_price if exit_price is not None else 0.0,
                    "realized_pnl": realized_pnl,
                    "exit_reason": exit_reason,
                })
                log.info("exit_order_filled", trade_id=trade_id,
                         actual_exit_price=exit_price, realized_pnl=realized_pnl,
                         via="partial_verdict_full_quantity")
            return

        # True partial: book the closed slice at its real fill price.
        import dataclasses as _dc
        closed_slice = _dc.replace(trade_record, quantity=filled_units)
        partial_pnl = (self._calculate_realized_pnl(closed_slice, exit_price)
                       if exit_price is not None else 0.0)
        self._state.record_partial_exit(
            trade_id=trade_id,
            quantity=filled_units,
            price=exit_price if exit_price is not None else 0.0,
            pnl=partial_pnl,
        )
        remaining = int(trade_record.quantity) - filled_units
        self._state.update_trade_quantity(trade_id, remaining)
        # Remainder goes back to the monitor for a fresh exit at the REDUCED
        # quantity. CAS: CLOSING->FILLED only (F4.5).
        self._state.transition(
            trade_id, (TradeStatus.CLOSING,), TradeStatus.FILLED)
        log.warning(
            "exit_partial_fill_booked",
            trade_id=trade_id,
            closed=filled_units,
            remaining=remaining,
            exit_price=exit_price,
            partial_pnl=partial_pnl,
            exit_reason=pending_exit.exit_reason,
        )

    # Entry cash-flow direction lives in CREDIT_STRATEGIES (strategies/base.py).
    # Everything not listed there is DEBIT — long options, straddles,
    # calendars, and debit spreads (bull_call_spread / bear_put_spread are
    # DEBIT despite having 2 legs like a credit spread).
    CREDIT_STRATEGIES = CREDIT_STRATEGIES

    def _calculate_realized_pnl(self, trade, exit_price: float) -> float:
        """Calculate realized P&L from entry and exit prices.

        Direction is decided by the STRATEGY's cash-flow sign, not leg count:
        a long straddle and a credit spread both have 2 legs but opposite
        P&L formulas. contract_type alone is ambiguous here.
        """
        multiplier = 100  # options multiplier
        entry = trade.entry_price
        qty = trade.quantity

        if trade.contract_type == "stock":
            pnl = (exit_price - entry) * qty
        elif trade.strategy in self.CREDIT_STRATEGIES:
            # Credit: collected `entry` to open, pay `exit_price` to close
            pnl = (entry - exit_price) * multiplier * qty
        else:
            # Debit: paid `entry` to open, receive `exit_price` on close
            pnl = (exit_price - entry) * multiplier * qty

        # Subtract commissions: ~$0.65/contract/side (IBKR tiered pricing).
        # R15 #8: this is an ESTIMATE at fill time; _process_completed_exits
        # trues it up against the executions ledger (same formula via
        # commission_estimate) before any stats/breaker/learning booking.
        # Class-qualified (not self.): tests drive this method through a
        # minimal host shim that only carries CREDIT_STRATEGIES.
        pnl -= TradeExecutor.commission_estimate(trade)

        return round(pnl, 2)

    @staticmethod
    def commission_estimate(trade) -> float:
        """Flat round-trip commission estimate: $0.65/contract/side per leg.
        Single source of truth — the fill-time booking subtracts this, and
        the completed-exit true-up adds it back before subtracting the REAL
        ledger commission. The two must stay the same formula."""
        qty = max(1, int(getattr(trade, "quantity", 1) or 1))
        legs_per_side = 1
        if trade.contract_type == "iron_condor":
            legs_per_side = 4
        elif trade.contract_type == "spread" or trade.strategy in (
            "long_straddle", "event_straddle", "short_strangle", "calendar_spread",
        ):
            legs_per_side = 2
        return 0.65 * legs_per_side * qty * 2  # entry + exit

    def _update_trade_filled(
        self, pending: PendingOrder, actual_price: float,
        filled_qty: int | None = None,
    ) -> None:
        """Update a trade record to FILLED status with actual fill info.

        R12 F4.4: takes the TRUE filled quantity from orderStatus. On the
        PARTIAL->FILLED edge the partial pass had already written the partial
        quantity into trades/open_positions; the old FILLED write only
        flipped status, so the position was managed (stops, exits, P&L) at
        the PARTIAL quantity while the full size was live at the broker.
        """
        signal = pending.signal
        contract_type = self._get_contract_type(signal)
        true_qty = int(filled_qty) if filled_qty else pending.contracts

        # R12 Tier-A #1: CAS — a fill can only land on a PENDING or PARTIAL
        # entry (blind UPDATE could resurrect CANCELLED/CLOSED rows).
        self._state.transition(
            pending.trade_id,
            (TradeStatus.PENDING, TradeStatus.PARTIAL),
            TradeStatus.FILLED,
        )

        # Book the REAL fill into trades.entry_price (audit R2/C1): it was
        # only stored in open_positions while every P&L computation read the
        # optimistic signal price from trades — systematically overstating
        # realized P&L by the entry slippage (measured -$89 across the first
        # 8 marketable fills, up to 14.8% of credit). trades.entry_price is
        # the unsigned premium convention; BAG credit fills come signed.
        if actual_price is not None and actual_price != 0:
            self._state.update_trade_entry_price(pending.trade_id, abs(actual_price))

        # Build legs JSON for open_positions (handles both contract-shaped and
        # plain-dict legs via _leg_fields so dict-shaped legs don't crash here)
        legs_json = json.dumps([
            {
                "strike": strike,
                "right": right,
                "action": leg["action"],
                "expiry": expiry,
            }
            for leg in signal.legs
            for strike, right, expiry in [self._leg_fields(leg)]
        ]) if signal.legs else "[]"

        # Insert into open_positions so HWM / partial-exit tracking works
        self._state.insert_open_position(
            trade_id=pending.trade_id,
            symbol=signal.symbol,
            contract_type=contract_type,
            quantity=true_qty,
            entry_price=actual_price,
            legs=legs_json,
        )
        # R12 F4.4: insert_open_position is INSERT OR IGNORE — after a
        # PARTIAL pass the existing row keeps its partial quantity, and
        # trades.quantity was shrunk by the partial write too. Explicitly
        # settle BOTH tables at the broker-reported filled quantity.
        self._state.update_trade_quantity(pending.trade_id, true_qty)

    def _update_trade_partial(
        self, pending: PendingOrder, filled_qty: int, actual_price: float | None = None
    ) -> None:
        """Update a trade record for a partial ENTRY fill.

        AUDIT R2/C4-C5: this (and _update_trade_cancelled) previously went
        through record_trade, which is INSERT OR IGNORE — a silent NO-OP on
        the already-existing PENDING row. The PARTIAL status never persisted,
        quantity stayed wrong, and no open_positions row existed, so the
        already-filled contracts were a LIVE, UNMANAGED position. Write the
        transitions for real and register the partial position.
        """
        signal = pending.signal
        # R12 Tier-A #1: CAS — PARTIAL can only follow PENDING (or refresh an
        # earlier PARTIAL as more contracts fill).
        self._state.transition(
            pending.trade_id,
            (TradeStatus.PENDING, TradeStatus.PARTIAL),
            TradeStatus.PARTIAL,
        )
        self._state.update_trade_quantity(pending.trade_id, filled_qty)
        if actual_price is not None and actual_price != 0:
            self._state.update_trade_entry_price(pending.trade_id, abs(actual_price))
        # Register the filled portion so PortfolioManager manages it
        # (check_positions iterates FILLED/PARTIAL).
        legs_json = json.dumps([
            {"strike": s, "right": r, "action": leg["action"], "expiry": e}
            for leg in signal.legs
            for s, r, e in [self._leg_fields(leg)]
        ]) if signal.legs else "[]"
        self._state.insert_open_position(
            trade_id=pending.trade_id,
            symbol=signal.symbol,
            contract_type=self._get_contract_type(signal),
            quantity=filled_qty,
            entry_price=actual_price if actual_price is not None else signal.entry_price,
            legs=legs_json,
        )
        log.warning(
            "trade_partial_persisted",
            trade_id=pending.trade_id,
            filled=filled_qty,
            requested=pending.contracts,
        )

    def _update_trade_cancelled(self, pending: PendingOrder) -> bool:
        """Update a trade record to CANCELLED status (real write — see
        _update_trade_partial docstring for the INSERT OR IGNORE no-op bug).

        R12 Tier-A #1: CAS from PENDING only — a cancel verdict must never
        claw back a trade that has since FILLED or CLOSED.

        R19: PARTIAL was removed from the from-set. A partially filled entry
        has CONTRACTS LIVE at the broker and an open_positions row; flipping
        it to CANCELLED orphaned them in a status neither the portfolio
        monitor nor the reconciler re-adopts (the same one-way door the
        filled-qty-first rule exists to avoid) — and the filled-qty-first
        protection cannot fire post-reconnect, when the order row is not in
        all_trades at all. A PARTIAL row therefore resolves to FILLED at the
        already-booked partial quantity: the order is dead, the contracts are
        not. Returns True when the trade was really cancelled.
        """
        try:
            row = self._state.get_trade_by_id(pending.trade_id)
        except Exception:  # noqa: BLE001 — a lookup failure must not book a cancel
            row = None
        if row is not None and getattr(row, "status", None) == TradeStatus.PARTIAL:
            promoted = self._state.transition(
                pending.trade_id,
                (TradeStatus.PARTIAL,),
                TradeStatus.FILLED,
            )
            log.critical(
                "cancel_refused_partial_contracts_live",
                trade_id=pending.trade_id,
                quantity=getattr(row, "quantity", None),
                promoted=promoted,
                note="entry order died after a partial fill — the filled "
                     "contracts stay MANAGED (FILLED at the partial "
                     "quantity) instead of being orphaned as CANCELLED",
            )
            return False
        self._state.transition(
            pending.trade_id,
            (TradeStatus.PENDING,),
            TradeStatus.CANCELLED,
        )
        return True

    def register_exit_order(
        self,
        order_id: int,
        trade_id: str,
        exit_reason: str,
        estimated_pnl: float,
    ) -> None:
        # R8: exit orders were never added to the orderId->trade_id map, so
        # exit-leg fills landed in the executions ledger with trade_id='' and
        # total_commission() summed HALF the round trip.
        if not hasattr(self, "_order_trade_map"):
            self._order_trade_map = {}
        self._order_trade_map[order_id] = trade_id
        """Register an exit order for fill tracking.

        Called by the orchestrator after placing a close order so that
        check_fills() can detect when the exit actually fills and finalise
        the trade with the real fill price.
        """
        self._pending_exit_orders[order_id] = PendingExitOrder(
            trade_id=trade_id,
            exit_reason=exit_reason,
            estimated_pnl=estimated_pnl,
        )
        log.info(
            "exit_order_registered",
            order_id=order_id,
            trade_id=trade_id,
        )

    def adopt_exit_order(self, order_id: int, trade_id: str, reason: str) -> None:
        """R12 F4.1 (CLOSING restart): re-register an exit tracker for a
        working close order discovered at startup reconcile.

        The in-memory exit tracker dies with the process; before this, a
        restart mid-close promoted the CLOSING trade back to FILLED and the
        portfolio placed a SECOND close — when the original order also
        filled, the position REVERSED. The reconciler now keeps such trades
        CLOSING and reports the working orderId; the orchestrator calls this
        to resume tracking (estimated_pnl 0.0 — the real P&L is computed from
        the actual fill when check_fills books the close).
        """
        if order_id in self._pending_exit_orders:
            log.info("exit_order_already_tracked", order_id=order_id,
                     trade_id=trade_id)
            return
        # R16: an order adopted from a PREVIOUS clientId is a frozen snapshot
        # in ib_insync's cache — its status never updates again, so check_fills
        # would see it "working" forever (or "vanished" after a cache wipe).
        # Classify it at adoption so the exit loop resolves it against the
        # broker instead of the cache. The R12 stash cannot cover this: it is
        # skipped on the first connect of a restarted process, which is
        # exactly when adoption happens.
        try:
            if self._is_foreign_session_order(order_id, self._ibkr.ib.trades()):
                if getattr(self, "_foreign_exit_orders", None) is None:
                    self._foreign_exit_orders = set()
                self._foreign_exit_orders.add(order_id)
                log.warning("exit_order_adopted_foreign_session",
                            order_id=order_id, trade_id=trade_id,
                            note="status is unobservable in this session; "
                                 "tracked against reqAllOpenOrders instead")
        except Exception as _e:  # noqa: BLE001 — classification must not block adoption
            log.debug("adopted_exit_foreign_check_failed", error=str(_e))
        self.register_exit_order(
            order_id=order_id,
            trade_id=trade_id,
            exit_reason=reason,
            estimated_pnl=0.0,
        )
        log.warning(
            "exit_order_adopted",
            order_id=order_id,
            trade_id=trade_id,
            reason=reason,
        )

    @property
    def pending_count(self) -> int:
        """Number of orders currently pending fill (entry + exit)."""
        return len(self._pending_orders) + len(self._pending_exit_orders)

    @staticmethod
    def _leg_fields(leg: dict) -> tuple[float, str, str]:
        """(strike, right, expiry) from either leg shape.

        Most strategies emit {"contract": OptionContract, "action": ...};
        event_straddle/calendar emit plain {"strike", "right", "expiry",
        "action"} dicts. Both must execute.
        """
        contract = leg.get("contract")
        if contract is not None:
            return float(contract.strike), str(contract.right), str(contract.expiry)
        return float(leg["strike"]), str(leg["right"]), str(leg["expiry"])

    @staticmethod
    def _get_contract_type(signal: Signal) -> str:
        """Determine contract type from signal."""
        if signal.legs and len(signal.legs) == 4:
            return "iron_condor"
        elif signal.legs and len(signal.legs) == 2:
            return "spread"
        elif signal.contract:
            return "call" if signal.contract.right == "C" else "put"
        return "unknown"

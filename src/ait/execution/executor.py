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
DEFAULT_ORDER_TIMEOUT = 90  # 90 seconds


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
    """Tracks a pending order with its submission time."""

    __slots__ = ("trade_id", "signal", "submitted_at", "contracts")

    def __init__(self, trade_id: str, signal: Signal, contracts: int) -> None:
        self.trade_id = trade_id
        self.signal = signal
        self.submitted_at = time.time()
        self.contracts = contracts

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
    ) -> None:
        self._ibkr = ibkr_client
        self._state = state
        self._circuit_breaker = circuit_breaker
        self._order_timeout = order_timeout
        self._pending_orders: dict[int, PendingOrder] = {}  # order_id → PendingOrder
        self._pending_exit_orders: dict[int, PendingExitOrder] = {}  # order_id → PendingExitOrder

    async def execute_signal(self, signal: Signal, contracts: int) -> str | None:
        """Execute a trade signal. Returns trade_id on success, None on failure."""
        # INST-5 (institutional audit): defined-risk-only as an EXECUTION
        # gate, not a config policy. Undefined-risk (naked short premium)
        # orders are refused unless explicitly allowed — in paper we allow
        # them via run_orchestrator's default so the edge comparison runs;
        # at go-live the env is unset and the wings become a contractual,
        # not configured, loss floor.
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
            if signal.legs:
                trade = await self._execute_multi_leg(signal, contracts, trade_id)
            else:
                trade = await self._execute_single_leg(signal, contracts, trade_id)

            if trade is None:
                return None

            # Record trade in state
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

            # Track pending order for fill monitoring
            if trade.order.orderId:
                self._pending_orders[trade.order.orderId] = PendingOrder(
                    trade_id=trade_id,
                    signal=signal,
                    contracts=contracts,
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

        # Reject if bid-ask spread is too wide (>15% of mid) — indicates stale/illiquid quote
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > 0.15:
                log.warning("wide_spread_rejected", trade_id=trade_id,
                            bid=bid, ask=ask, spread_pct=f"{spread_pct:.1%}")
                self._state.update_trade_status(trade_id, TradeStatus.CANCELLED)
                return None

        if bid > 0 and ask > 0 and ask > bid:
            order = OrderBuilder.passive_limit(
                action=signal.action,
                quantity=contracts,
                bid=bid,
                ask=ask,
            )
            log.info(
                "passive_order",
                trade_id=trade_id,
                bid=bid,
                ask=ask,
                limit=order.lmtPrice,
            )
        else:
            order = OrderBuilder.limit(
                action=signal.action,
                quantity=contracts,
                limit_price=signal.entry_price,
            )

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
            self._state.update_trade_status(trade_id, TradeStatus.CANCELLED)
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
                    self._state.update_trade_status(trade_id, TradeStatus.CANCELLED)
                    return None
                # combo quotes are signed; compare magnitudes
                if abs(_mid) > 0.05 and abs(abs(signal.entry_price) - abs(_mid)) / abs(_mid) > 0.35:
                    log.warning("combo_signal_price_stale_rejected",
                                trade_id=trade_id,
                                signal_px=signal.entry_price, live_mid=_mid)
                    self._state.update_trade_status(trade_id, TradeStatus.CANCELLED)
                    return None
        except Exception as _e:  # noqa: BLE001 — validation must not block on quote hiccups
            log.debug("combo_quote_validation_skipped", error=str(_e))

        limit_price, aggressive_offset = combo_entry_limit(
            signal.entry_price, is_credit
        )
        # R5 audit: for debit combos the marketable cross (up to +15%) pushed
        # the worst-case cost past the max_loss the risk manager validated —
        # a systematic ~15% understatement. Cap the limit at the validated
        # per-contract loss so the risk check stays truthful.
        if not is_credit and signal.max_loss and contracts > 0:
            _validated = signal.max_loss / (100 * contracts)
            if limit_price > _validated > 0:
                limit_price = round(_validated, 2)
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

        return await self._ibkr.place_order(combo, order)

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

        # 1. Cancel stale orders that have exceeded timeout
        await self._cancel_stale_orders()

        # 2. Check status of all pending orders
        open_trades = self._ibkr.get_open_orders()
        all_trades = self._ibkr.get_all_trades() if hasattr(self._ibkr, 'get_all_trades') else []

        for order_id, pending in list(self._pending_orders.items()):
            still_open = any(t.order.orderId == order_id for t in open_trades)

            if still_open:
                continue  # Order is still working

            # Order is no longer open — determine what happened
            status = self._determine_fill_status(order_id, all_trades, pending)

            if status == "filled":
                # Get actual fill price if available
                actual_price = self._get_fill_price(order_id, all_trades, pending)
                self._update_trade_filled(pending, actual_price)
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
                # to avoid orphaned legs and stale prices
                if pending.age_seconds > 30:
                    log.warning(
                        "partial_fill_cancelling_remainder",
                        trade_id=pending.trade_id,
                        symbol=pending.signal.symbol,
                        filled=filled_qty,
                        requested=pending.contracts,
                        age_seconds=pending.age_seconds,
                    )
                    try:
                        for t in open_trades:
                            if t.order.orderId == order_id:
                                self._ibkr.ib.cancelOrder(t.order)
                                break
                    except Exception as e:
                        log.warning("partial_cancel_failed", error=str(e))
                else:
                    log.warning(
                        "trade_partial_fill",
                        trade_id=pending.trade_id,
                        symbol=pending.signal.symbol,
                        filled=filled_qty,
                        requested=pending.contracts,
                    )

            elif status == "cancelled":
                self._update_trade_cancelled(pending)
                cancelled.append(pending.trade_id)
                log.info(
                    "trade_cancelled",
                    trade_id=pending.trade_id,
                    symbol=pending.signal.symbol,
                    age_seconds=pending.age_seconds,
                )

            if status in ("filled", "cancelled"):
                del self._pending_orders[order_id]

        if cancelled:
            log.info("orders_cancelled", count=len(cancelled), trade_ids=cancelled)

        # 3. Check pending EXIT orders — finalize CLOSING → CLOSED with real fill price
        completed_exits = []
        for order_id, pending_exit in list(self._pending_exit_orders.items()):
            still_open = any(t.order.orderId == order_id for t in open_trades)
            if still_open:
                continue

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

                self._state.close_trade(
                    trade_id=pending_exit.trade_id,
                    exit_price=actual_exit_price if actual_exit_price is not None else 0.0,
                    realized_pnl=realized_pnl,
                    exit_reason_detailed=exit_reason,
                )
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
                del self._pending_exit_orders[order_id]

            elif exit_status == "cancelled":
                # Exit order was cancelled/rejected — revert to FILLED so
                # portfolio manager will re-trigger an exit next cycle.
                self._state.update_trade_status(
                    pending_exit.trade_id, TradeStatus.FILLED,
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
                log.warning(
                    "exit_order_stale_pending_cancel_requested",
                    trade_id=pending_exit.trade_id,
                    age_seconds=int(pending_exit.age_seconds),
                )
                try:
                    for t in open_trades:
                        if t.order.orderId == order_id:
                            self._ibkr.ib.cancelOrder(t.order)
                            break
                except Exception:
                    pass
                if pending_exit.age_seconds > 900:
                    log.error(
                        "exit_order_zombie_reverting",
                        trade_id=pending_exit.trade_id,
                        age_seconds=int(pending_exit.age_seconds),
                    )
                    self._state.update_trade_status(
                        pending_exit.trade_id, TradeStatus.FILLED,
                    )
                    del self._pending_exit_orders[order_id]

        return filled, completed_exits

    async def _cancel_stale_orders(self) -> None:
        """Cancel orders that have been pending longer than the timeout."""
        for order_id, pending in list(self._pending_orders.items()):
            if pending.age_seconds > self._order_timeout:
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
                    return "partial" if filled_qty > 0 else "cancelled"
                elif status in ("submitted", "presubmitted"):
                    return "pending"  # Still working

                remaining = trade.orderStatus.remaining or 0
                if filled_qty > 0 and remaining > 0:
                    return "partial"
                elif filled_qty > 0:
                    return "filled"
                else:
                    return "cancelled"

        # NOT-FOUND HARDENING (deep-audit MP-F2b): the entry path used to
        # assume "cancelled" whenever the order wasn't in get_all_trades() —
        # but that list is empty on any connection blip, so a mid-cycle
        # disconnect flipped EVERY working entry to CANCELLED while live at
        # IBKR. Mirror the exit path: treat young/unknown as still pending.
        if pending.age_seconds < 30:
            return "pending"
        return "cancelled"

    @staticmethod
    def _reconstruct_bag_price(trade) -> float | None:
        """Side-aware net price for a BAG from per-leg executions.

        (sold proceeds - bought costs) per contract, negated to IBKR's
        as-defined convention (negative = combo nets a credit). Shared by
        entry and exit fill reconstruction (audit R2 — the entry path
        previously lacked this and silently fell back to the signal price).
        """
        qty = trade.order.totalQuantity or 0
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
        return 0

    def _determine_exit_fill_status(self, order_id: int, all_trades: list) -> str:
        """Determine whether an exit order filled or was cancelled."""
        for trade in all_trades:
            if trade.order.orderId == order_id:
                status = trade.orderStatus.status.lower()
                if status in ("filled",):
                    return "filled"
                elif status in ("cancelled", "inactive", "apicancelled"):
                    return "cancelled"
                elif status in ("submitted", "presubmitted"):
                    return "pending"

                filled_qty = trade.orderStatus.filled or 0
                if filled_qty > 0:
                    return "filled"
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

        # Subtract commissions: ~$0.65/contract/side (IBKR tiered pricing)
        # Multi-leg strategies pay per leg both entering and exiting
        legs_per_side = 1
        if trade.contract_type == "iron_condor":
            legs_per_side = 4
        elif trade.contract_type == "spread" or trade.strategy in (
            "long_straddle", "event_straddle", "short_strangle", "calendar_spread",
        ):
            legs_per_side = 2
        commission = 0.65 * legs_per_side * qty * 2  # entry + exit
        pnl -= commission

        return round(pnl, 2)

    def _update_trade_filled(self, pending: PendingOrder, actual_price: float) -> None:
        """Update a trade record to FILLED status with actual fill info."""
        signal = pending.signal
        contract_type = self._get_contract_type(signal)

        # Update trade status to FILLED (record_trade uses INSERT OR IGNORE,
        # so use update_trade_status for existing rows)
        self._state.update_trade_status(pending.trade_id, TradeStatus.FILLED)

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
            quantity=pending.contracts,
            entry_price=actual_price,
            legs=legs_json,
        )

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
        self._state.update_trade_status(pending.trade_id, TradeStatus.PARTIAL)
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

    def _update_trade_cancelled(self, pending: PendingOrder) -> None:
        """Update a trade record to CANCELLED status (real write — see
        _update_trade_partial docstring for the INSERT OR IGNORE no-op bug)."""
        self._state.update_trade_status(pending.trade_id, TradeStatus.CANCELLED)

    def register_exit_order(
        self,
        order_id: int,
        trade_id: str,
        exit_reason: str,
        estimated_pnl: float,
    ) -> None:
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

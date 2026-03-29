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
import time
import uuid
from datetime import datetime

from ib_insync import Trade, TradeLogEntry

from ait.broker.contracts import ContractBuilder
from ait.broker.ibkr_client import IBKRClient
from ait.broker.orders import OrderBuilder
from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
from ait.risk.circuit_breaker import CircuitBreaker
from ait.strategies.base import Signal, SignalDirection
from ait.utils.logging import get_logger

log = get_logger("execution.executor")

# Order timeout: cancel unfilled orders after this many seconds
DEFAULT_ORDER_TIMEOUT = 300  # 5 minutes


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
                    "strike": leg["contract"].strike,
                    "right": leg["contract"].right,
                    "action": leg["action"],
                    "expiry": str(leg["contract"].expiry),
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

        # Build all leg contracts at once
        ibkr_contracts = []
        for leg in signal.legs:
            opt_contract = leg["contract"]
            ibkr_contracts.append(ContractBuilder.option(
                symbol=signal.symbol,
                expiry=opt_contract.expiry,
                strike=opt_contract.strike,
                right=opt_contract.right,
            ))

        # Batch qualify all legs in one call
        qualified_list = await self._ibkr.qualify_contracts_batch(ibkr_contracts)

        qualified_legs = []
        for i, qualified in enumerate(qualified_list):
            if not qualified:
                leg = signal.legs[i]
                opt_contract = leg["contract"]
                log.error(
                    "leg_qualification_failed",
                    trade_id=trade_id,
                    strike=opt_contract.strike,
                    right=opt_contract.right,
                )
                return None

            qualified_legs.append({
                "conId": qualified.conId,
                "action": signal.legs[i]["action"],
                "ratio": signal.legs[i].get("ratio", 1),
            })

        combo = ContractBuilder.combo(
            symbol=signal.symbol,
            legs=qualified_legs,
        )

        # Determine if this is a credit spread by checking the legs:
        # if more legs are selling than buying, it's a net credit trade.
        # For debit spreads (bull call, bear put), entry_price is positive debit.
        # For credit spreads (iron condor, short strangle), IBKR expects negative limit.
        if signal.legs:
            sell_count = sum(1 for leg in signal.legs if leg["action"] == "SELL")
            buy_count = len(signal.legs) - sell_count
            is_credit = sell_count > buy_count
        else:
            is_credit = signal.action == "SELL"
        limit_price = -signal.entry_price if is_credit else signal.entry_price

        order = OrderBuilder.combo_limit(
            action="BUY",
            quantity=contracts,
            limit_price=limit_price,
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
                self._update_trade_partial(pending, filled_qty)
                log.warning(
                    "trade_partial_fill",
                    trade_id=pending.trade_id,
                    symbol=pending.signal.symbol,
                    filled=filled_qty,
                    requested=pending.contracts,
                )
                # Don't remove from pending — will keep checking

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

            exit_status = self._determine_exit_fill_status(order_id, all_trades)

            if exit_status == "filled":
                actual_exit_price = self._get_exit_fill_price(order_id, all_trades)
                realized_pnl = pending_exit.estimated_pnl

                self._state.close_trade(
                    trade_id=pending_exit.trade_id,
                    exit_price=actual_exit_price,
                    realized_pnl=realized_pnl,
                    exit_reason_detailed=pending_exit.exit_reason,
                )
                completed_exits.append({
                    "trade_id": pending_exit.trade_id,
                    "exit_price": actual_exit_price,
                    "realized_pnl": realized_pnl,
                    "exit_reason": pending_exit.exit_reason,
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
                if status in ("filled",):
                    return "filled"
                elif status in ("cancelled", "inactive", "apicancelled"):
                    return "cancelled"
                elif status in ("submitted", "presubmitted"):
                    return "pending"  # Still working

                filled_qty = trade.orderStatus.filled or 0
                remaining = trade.orderStatus.remaining or 0
                if filled_qty > 0 and remaining > 0:
                    return "partial"
                elif filled_qty > 0:
                    return "filled"
                else:
                    return "cancelled"

        # If not found in trades list, assume cancelled (order was rejected or expired)
        return "cancelled"

    def _get_fill_price(
        self, order_id: int, all_trades: list, pending: PendingOrder
    ) -> float:
        """Get the actual fill price for an order."""
        for trade in all_trades:
            if trade.order.orderId == order_id:
                avg_price = trade.orderStatus.avgFillPrice
                if avg_price and avg_price > 0:
                    return avg_price

        return pending.signal.entry_price  # Fallback to expected price

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

        return "cancelled"

    def _get_exit_fill_price(self, order_id: int, all_trades: list) -> float:
        """Get the actual fill price for an exit order."""
        for trade in all_trades:
            if trade.order.orderId == order_id:
                avg_price = trade.orderStatus.avgFillPrice
                if avg_price and avg_price > 0:
                    return avg_price
        return 0.0

    def _update_trade_filled(self, pending: PendingOrder, actual_price: float) -> None:
        """Update a trade record to FILLED status with actual fill info."""
        signal = pending.signal
        contract_type = self._get_contract_type(signal)

        # Update trade status to FILLED (record_trade uses INSERT OR IGNORE,
        # so use update_trade_status for existing rows)
        self._state.update_trade_status(pending.trade_id, TradeStatus.FILLED)

        # Build legs JSON for open_positions
        legs_json = json.dumps([
            {
                "strike": leg["contract"].strike,
                "right": leg["contract"].right,
                "action": leg["action"],
                "expiry": str(leg["contract"].expiry),
            }
            for leg in signal.legs
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

    def _update_trade_partial(self, pending: PendingOrder, filled_qty: int) -> None:
        """Update a trade record for partial fill."""
        signal = pending.signal
        self._state.record_trade(
            TradeRecord(
                trade_id=pending.trade_id,
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                direction=(
                    TradeDirection.LONG
                    if signal.direction != SignalDirection.BEARISH
                    else TradeDirection.SHORT
                ),
                status=TradeStatus.PARTIAL,
                entry_time=datetime.now().isoformat(),
                entry_price=signal.entry_price,
                quantity=filled_qty,
                contract_type=self._get_contract_type(signal),
                notes=f"partial fill: {filled_qty}/{pending.contracts}",
            )
        )

    def _update_trade_cancelled(self, pending: PendingOrder) -> None:
        """Update a trade record to CANCELLED status."""
        signal = pending.signal
        self._state.record_trade(
            TradeRecord(
                trade_id=pending.trade_id,
                symbol=signal.symbol,
                strategy=signal.strategy_name,
                direction=(
                    TradeDirection.LONG
                    if signal.direction != SignalDirection.BEARISH
                    else TradeDirection.SHORT
                ),
                status=TradeStatus.CANCELLED,
                entry_time=datetime.now().isoformat(),
                entry_price=signal.entry_price,
                quantity=0,
                contract_type=self._get_contract_type(signal),
                notes=f"cancelled after {pending.age_seconds:.0f}s (timeout={self._order_timeout}s)",
            )
        )

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
    def _get_contract_type(signal: Signal) -> str:
        """Determine contract type from signal."""
        if signal.legs and len(signal.legs) == 4:
            return "iron_condor"
        elif signal.legs and len(signal.legs) == 2:
            return "spread"
        elif signal.contract:
            return "call" if signal.contract.right == "C" else "put"
        return "unknown"

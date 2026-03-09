"""Trade executor — handles order lifecycle from signal to fill.

Responsible for:
- Converting signals to IBKR orders (single or combo)
- Submitting orders with proper contract qualification
- Tracking fills and partial fills
- Recording trades in state
"""

from __future__ import annotations

import asyncio
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


class TradeExecutor:
    """Executes trade signals by placing orders with IBKR."""

    def __init__(
        self,
        ibkr_client: IBKRClient,
        state: StateManager,
        circuit_breaker: CircuitBreaker,
    ) -> None:
        self._ibkr = ibkr_client
        self._state = state
        self._circuit_breaker = circuit_breaker
        self._pending_trades: dict[int, tuple[str, Signal]] = {}  # order_id → (trade_id, signal)

    async def execute_signal(self, signal: Signal, contracts: int) -> str | None:
        """Execute a trade signal. Returns trade_id on success, None on failure.

        Args:
            signal: The trade signal to execute
            contracts: Number of contracts (may differ from signal.quantity after risk sizing)
        """
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
            import json
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
                    else TradeDirection.LONG  # Neutral strategies default to long
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
                self._pending_trades[trade.order.orderId] = (trade_id, signal)

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
        """Execute a single-leg option order."""
        if not signal.contract:
            log.error("no_contract_in_signal", trade_id=trade_id)
            return None

        # Build IBKR contract
        contract = ContractBuilder.option(
            symbol=signal.symbol,
            expiry=signal.contract.expiry,
            strike=signal.contract.strike,
            right=signal.contract.right,
        )

        # Qualify with IBKR
        qualified = await self._ibkr.qualify_contract(contract)
        if not qualified:
            log.error("contract_qualification_failed", trade_id=trade_id)
            return None

        # Build order — always use limit orders for options
        order = OrderBuilder.limit(
            action=signal.action,
            quantity=contracts,
            limit_price=signal.entry_price,
        )

        return await self._ibkr.place_order(qualified, order)

    async def _execute_multi_leg(
        self, signal: Signal, contracts: int, trade_id: str
    ) -> Trade | None:
        """Execute a multi-leg combo order (spreads, condors)."""
        if not signal.legs:
            log.error("no_legs_in_signal", trade_id=trade_id)
            return None

        # Qualify all leg contracts first
        qualified_legs = []
        for leg in signal.legs:
            opt_contract = leg["contract"]
            ibkr_contract = ContractBuilder.option(
                symbol=signal.symbol,
                expiry=opt_contract.expiry,
                strike=opt_contract.strike,
                right=opt_contract.right,
            )
            qualified = await self._ibkr.qualify_contract(ibkr_contract)
            if not qualified:
                log.error(
                    "leg_qualification_failed",
                    trade_id=trade_id,
                    strike=opt_contract.strike,
                    right=opt_contract.right,
                )
                return None

            qualified_legs.append({
                "conId": qualified.conId,
                "action": leg["action"],
                "ratio": leg.get("ratio", 1),
            })

        # Build combo contract
        combo = ContractBuilder.combo(
            symbol=signal.symbol,
            legs=qualified_legs,
        )

        # Build combo limit order
        # For debit spreads: positive price = paying
        # For credit spreads: negative price = receiving
        is_credit = signal.action == "SELL"
        limit_price = -signal.entry_price if is_credit else signal.entry_price

        order = OrderBuilder.combo_limit(
            action="BUY",  # Always BUY the combo (direction is in the legs)
            quantity=contracts,
            limit_price=limit_price,
        )

        return await self._ibkr.place_order(combo, order)

    async def check_fills(self) -> list[str]:
        """Check for filled orders and update state. Returns list of filled trade_ids."""
        filled = []
        open_trades = self._ibkr.get_open_orders()

        for order_id, (trade_id, signal) in list(self._pending_trades.items()):
            # Check if this order is still open
            still_open = any(t.order.orderId == order_id for t in open_trades)

            if not still_open:
                # Order no longer open — likely filled or cancelled
                # Update trade status
                self._state.record_trade(
                    TradeRecord(
                        trade_id=trade_id,
                        symbol=signal.symbol,
                        strategy=signal.strategy_name,
                        direction=(
                            TradeDirection.LONG
                            if signal.direction != SignalDirection.BEARISH
                            else TradeDirection.SHORT
                        ),
                        status=TradeStatus.FILLED,
                        entry_time=datetime.now().isoformat(),
                        entry_price=signal.entry_price,
                        quantity=signal.quantity,
                        contract_type=self._get_contract_type(signal),
                    )
                )
                filled.append(trade_id)
                del self._pending_trades[order_id]
                log.info("trade_filled", trade_id=trade_id, symbol=signal.symbol)

        return filled

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

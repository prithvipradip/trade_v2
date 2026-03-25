"""IBKR client wrapper around ib_insync.

Handles connection lifecycle, auto-reconnect, and health monitoring.
This is the single point of contact with Interactive Brokers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Union

from ib_insync import IB, Contract, Order, Trade, util

from ait.config.settings import IBKREnvConfig
from ait.utils.logging import get_logger

log = get_logger("broker.ibkr")


class IBKRClient:
    """Manages the IBKR connection with auto-reconnect and health checks."""

    def __init__(self, config: IBKREnvConfig) -> None:
        self._config = config
        self._ib = IB()
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds

        # Wire up disconnect handler
        self._ib.disconnectedEvent += self._on_disconnect

    @property
    def ib(self) -> IB:
        """Direct access to ib_insync IB instance for advanced operations."""
        return self._ib

    @property
    def connected(self) -> bool:
        return self._connected and self._ib.isConnected()

    async def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway."""
        try:
            await self._ib.connectAsync(
                host=self._config.ibkr_host,
                port=self._config.ibkr_port,
                clientId=self._config.ibkr_client_id,
                timeout=15,
                readonly=False,
            )
            self._connected = True
            self._reconnect_attempts = 0

            account = self._config.ibkr_account or (
                self._ib.managedAccounts()[0] if self._ib.managedAccounts() else "unknown"
            )
            log.info(
                "ibkr_connected",
                host=self._config.ibkr_host,
                port=self._config.ibkr_port,
                account=account,
            )
            return True

        except Exception as e:
            log.error("ibkr_connection_failed", error=str(e))
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect from IBKR."""
        if self._ib.isConnected():
            self._ib.disconnect()
            self._connected = False
            log.info("ibkr_disconnected")

    async def ensure_connected(self) -> bool:
        """Ensure we're connected, reconnecting if necessary."""
        if self.connected:
            return True

        log.warning("ibkr_not_connected", action="reconnecting")
        return await self._reconnect()

    async def _reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        while self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))
            log.info(
                "ibkr_reconnecting",
                attempt=self._reconnect_attempts,
                max_attempts=self._max_reconnect_attempts,
                delay_seconds=delay,
            )
            await asyncio.sleep(delay)

            try:
                if self._ib.isConnected():
                    self._ib.disconnect()
                success = await self.connect()
                if success:
                    return True
            except Exception as e:
                log.error("ibkr_reconnect_failed", attempt=self._reconnect_attempts, error=str(e))

        log.critical("ibkr_reconnect_exhausted", attempts=self._max_reconnect_attempts)
        return False

    def _on_disconnect(self) -> None:
        """Handle unexpected disconnection."""
        self._connected = False
        log.warning("ibkr_unexpected_disconnect")

    async def qualify_contract(self, contract: Contract) -> Contract | None:
        """Qualify a contract with IBKR to get full details."""
        if not await self.ensure_connected():
            return None
        try:
            qualified = await self._ib.qualifyContractsAsync(contract)
            return qualified[0] if qualified else None
        except Exception as e:
            log.error("contract_qualification_failed", contract=str(contract), error=str(e))
            return None

    async def qualify_contracts_batch(self, contracts: list[Contract]) -> list[Contract | None]:
        """Qualify multiple contracts in a single batch call.

        Much faster than qualifying one at a time for multi-leg orders.
        Returns a list of qualified contracts (or None for failures)
        in the same order as the input.
        """
        if not contracts:
            return []
        if not await self.ensure_connected():
            return [None] * len(contracts)
        try:
            qualified = await self._ib.qualifyContractsAsync(*contracts)
            # qualifyContractsAsync returns the same contracts with details filled in.
            # Contracts that failed qualification will have conId == 0.
            result = []
            for q in qualified:
                if q.conId and q.conId > 0:
                    result.append(q)
                else:
                    result.append(None)
            return result
        except Exception as e:
            log.error("batch_qualification_failed", count=len(contracts), error=str(e))
            return [None] * len(contracts)

    async def place_order(self, contract: Contract, order: Order) -> Trade | None:
        """Place an order and return the Trade object for tracking."""
        if not await self.ensure_connected():
            log.error("cannot_place_order", reason="not connected")
            return None

        try:
            trade = self._ib.placeOrder(contract, order)
            log.info(
                "order_placed",
                symbol=contract.symbol,
                action=order.action,
                quantity=order.totalQuantity,
                order_type=order.orderType,
                order_id=trade.order.orderId,
            )
            return trade
        except Exception as e:
            log.error(
                "order_placement_failed",
                symbol=contract.symbol,
                error=str(e),
            )
            return None

    async def cancel_order(self, trade_or_id: Union[Trade, int]) -> bool:
        """Cancel a pending order.

        Args:
            trade_or_id: Either a Trade object or an integer order ID.
        """
        if not await self.ensure_connected():
            return False

        if isinstance(trade_or_id, int):
            # Look up the trade by order ID
            order_id = trade_or_id
            matching = [t for t in self._ib.trades() if t.order.orderId == order_id]
            if not matching:
                log.error("order_cancel_failed", order_id=order_id, error="no matching trade found")
                return False
            trade = matching[0]
        else:
            trade = trade_or_id

        try:
            self._ib.cancelOrder(trade.order)
            log.info("order_cancelled", order_id=trade.order.orderId)
            return True
        except Exception as e:
            log.error("order_cancel_failed", order_id=trade.order.orderId, error=str(e))
            return False

    def get_all_trades(self) -> list[Trade]:
        """Get all trades (open and completed) from IBKR."""
        if not self.connected:
            return []
        return self._ib.trades()

    def get_positions(self) -> list:
        """Get all current positions from IBKR."""
        if not self.connected:
            return []
        return self._ib.positions()

    def get_open_orders(self) -> list[Trade]:
        """Get all open/pending orders."""
        if not self.connected:
            return []
        return self._ib.openTrades()

    def get_portfolio(self) -> list:
        """Get portfolio items with market value and P&L."""
        if not self.connected:
            return []
        return self._ib.portfolio()

    async def get_account_values(self) -> dict[str, str]:
        """Get account summary values as a dict."""
        if not await self.ensure_connected():
            return {}
        values = {}
        for av in self._ib.accountValues():
            if av.currency == "USD" or av.currency == "":
                values[av.tag] = av.value
        return values


@asynccontextmanager
async def ibkr_session(config: IBKREnvConfig) -> AsyncGenerator[IBKRClient, None]:
    """Context manager for IBKR connection lifecycle."""
    client = IBKRClient(config)
    try:
        connected = await client.connect()
        if not connected:
            raise ConnectionError("Failed to connect to IBKR TWS/Gateway")
        yield client
    finally:
        await client.disconnect()

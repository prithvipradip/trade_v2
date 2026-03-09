"""Account management — buying power, margin, and account health.

Provides a clean interface to IBKR account data with caching
to avoid excessive API calls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ait.broker.ibkr_client import IBKRClient
from ait.utils.logging import get_logger

log = get_logger("broker.account")


@dataclass
class AccountSnapshot:
    """Point-in-time snapshot of account state."""

    timestamp: float = 0.0
    net_liquidation: float = 0.0
    buying_power: float = 0.0
    available_funds: float = 0.0
    excess_liquidity: float = 0.0
    maintenance_margin: float = 0.0
    initial_margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    cash_balance: float = 0.0


class AccountManager:
    """Manages account data with caching to minimize IBKR API calls."""

    def __init__(self, client: IBKRClient, cache_ttl: int = 30) -> None:
        self._client = client
        self._cache_ttl = cache_ttl
        self._snapshot = AccountSnapshot()
        self._last_fetch = 0.0

    async def get_snapshot(self, force_refresh: bool = False) -> AccountSnapshot:
        """Get current account snapshot, using cache if fresh enough."""
        now = time.time()
        if not force_refresh and (now - self._last_fetch) < self._cache_ttl:
            return self._snapshot

        values = await self._client.get_account_values()
        if not values:
            log.warning("account_fetch_empty", using="cached_values")
            return self._snapshot

        self._snapshot = AccountSnapshot(
            timestamp=now,
            net_liquidation=float(values.get("NetLiquidation", 0)),
            buying_power=float(values.get("BuyingPower", 0)),
            available_funds=float(values.get("AvailableFunds", 0)),
            excess_liquidity=float(values.get("ExcessLiquidity", 0)),
            maintenance_margin=float(values.get("MaintMarginReq", 0)),
            initial_margin=float(values.get("InitMarginReq", 0)),
            unrealized_pnl=float(values.get("UnrealizedPnL", 0)),
            realized_pnl=float(values.get("RealizedPnL", 0)),
            cash_balance=float(values.get("CashBalance", 0)),
        )
        self._last_fetch = now

        log.debug(
            "account_snapshot",
            net_liq=self._snapshot.net_liquidation,
            buying_power=self._snapshot.buying_power,
            unrealized_pnl=self._snapshot.unrealized_pnl,
        )
        return self._snapshot

    async def can_afford(self, estimated_cost: float) -> bool:
        """Check if account has enough buying power for a trade."""
        snapshot = await self.get_snapshot()
        can = snapshot.buying_power >= estimated_cost
        if not can:
            log.warning(
                "insufficient_buying_power",
                required=estimated_cost,
                available=snapshot.buying_power,
            )
        return can

    async def get_net_liquidation(self) -> float:
        """Get net liquidation value (total account value)."""
        snapshot = await self.get_snapshot()
        return snapshot.net_liquidation

    async def get_margin_usage_pct(self) -> float:
        """Get margin usage as a percentage of net liquidation."""
        snapshot = await self.get_snapshot()
        if snapshot.net_liquidation <= 0:
            return 0.0
        return snapshot.maintenance_margin / snapshot.net_liquidation

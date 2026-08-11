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

    # R17: past this age, a stale account snapshot escalates beyond a log
    # line -- every dollar-based risk gate runs on this number.
    ESCALATE_STALE_SECONDS = 900

    def __init__(self, client: IBKRClient, cache_ttl: int = 30) -> None:
        self._client = client
        self._cache_ttl = cache_ttl
        self._snapshot = AccountSnapshot()
        self._last_fetch = 0.0
        self._notify_cb = None  # optional async callable(str), wired by orchestrator
        self._circuit_breaker = None  # optional CircuitBreaker, wired by orchestrator
        self._stale_escalated = False

    async def _handle_stale(self, stale_seconds: float) -> None:
        """R17: log.error alone never escalated -- risk math kept running
        on a fossil snapshot for as long as an outage lasted. Past
        ESCALATE_STALE_SECONDS, trip the existing circuit-breaker halt lever
        and notify, once per outage.
        """
        if stale_seconds > 300:
            log.error(
                "account_data_stale",
                stale_seconds=int(stale_seconds),
                msg="Account data over 5 minutes old — risk calculations unreliable",
            )
        else:
            log.warning("account_fetch_empty", using="cached_values")
            return
        if stale_seconds <= self.ESCALATE_STALE_SECONDS or self._stale_escalated:
            return
        self._stale_escalated = True
        if self._circuit_breaker is not None:
            try:
                self._circuit_breaker.record_api_failure()
            except Exception as e:  # noqa: BLE001 — escalation must not crash the read
                log.warning("account_stale_breaker_failed", error=str(e))
        if self._notify_cb is not None:
            try:
                await self._notify_cb(
                    f"ACCOUNT DATA STALE {int(stale_seconds / 60)} min — risk "
                    f"calculations are running on a fossil snapshot."
                )
            except Exception as e:  # noqa: BLE001
                log.warning("account_stale_notify_failed", error=str(e))

    async def get_snapshot(self, force_refresh: bool = False) -> AccountSnapshot:
        """Get current account snapshot, using cache if fresh enough."""
        now = time.time()
        if not force_refresh and (now - self._last_fetch) < self._cache_ttl:
            return self._snapshot

        values = await self._client.get_account_values()
        if not values:
            # If a prior good snapshot exists, keep it rather than falling
            # through to the position-sum estimate: that estimate is
            # currency-blind (mixes CAD stock and USD options), and
            # get_account_values now intentionally returns {} when FX is
            # unavailable (audit 2026-07-07 item 1.6) — a mis-currencied
            # guess here would defeat that hard-stop. Stale-but-correct
            # beats fresh-but-wrong for risk math.
            if self._last_fetch > 0:
                await self._handle_stale(now - self._last_fetch)
                return self._snapshot
            # Never had a snapshot at all: last-resort estimate from
            # positions (crude, currency-blind — better than zero only at
            # first boot).
            try:
                positions = self._client.ib.positions()
                if positions:
                    total_value = sum(
                        abs(p.position) * p.avgCost for p in positions
                    )
                    if total_value > 0:
                        log.info("account_fallback_from_positions",
                                 estimated_nlv=total_value, positions=len(positions))
                        self._snapshot.net_liquidation = total_value
                        self._snapshot.buying_power = total_value * 0.5
                        self._snapshot.timestamp = now
                        self._last_fetch = now
                        return self._snapshot
            except Exception as e:
                log.debug("position_fallback_failed", error=str(e))

            if self._last_fetch > 0:
                await self._handle_stale(now - self._last_fetch)
            else:
                log.warning("account_fetch_empty", using="cached_values")
            return self._snapshot

        # AIT_SIMULATED_CAPITAL: cap every balance at a target size so the
        # ENTIRE risk stack (capital tiers, position sizing, aggregate caps)
        # behaves as if the account were that small. Purpose: validate the
        # actual go-live configuration (e.g. $2k CAD ~= $1.4k USD -> MICRO
        # tier, SPY credit spreads) on the big paper account. Opt-in via env.
        import os as _os
        _sim = _os.environ.get("AIT_SIMULATED_CAPITAL")
        if _sim:
            try:
                _cap = float(_sim)
                _real_nlv = float(values.get("NetLiquidation", 0)) or _cap
                _scale = min(1.0, _cap / _real_nlv)
                for _k in ("NetLiquidation", "BuyingPower", "AvailableFunds",
                           "ExcessLiquidity", "CashBalance"):
                    if values.get(_k):
                        values[_k] = str(float(values[_k]) * _scale)
                log.info("simulated_capital_active", target=_cap,
                         real_nlv=round(_real_nlv, 0))
            except (TypeError, ValueError):
                pass

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
        self._stale_escalated = False  # R17: outage over — re-arm the escalation latch

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

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
        self._fx_cache: dict[str, float] = {}  # ccy -> units per 1 USD, cached per session

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
        """Connect to IBKR TWS/Gateway.

        Falls back to alternate client IDs on Error 326 ("client id already
        in use"). A bot that restarts faster than IBKR releases the old
        session's client id would otherwise crash-loop forever: connect
        fails -> bot exits -> supervisor restarts -> same id still held.
        """
        base_id = self._config.ibkr_client_id
        # Try the configured id first, then a few deterministic-ish fallbacks.
        # offsets vary per attempt; not random so resume/journaling is stable.
        candidate_ids = [base_id] + [base_id + 100 + i for i in range(5)]
        for client_id in candidate_ids:
            try:
                await self._ib.connectAsync(
                    host=self._config.ibkr_host,
                    port=self._config.ibkr_port,
                    clientId=client_id,
                    timeout=15,
                    readonly=False,
                )
                self._connected = True
                self._reconnect_attempts = 0

                # Market data type. Default 4 = delayed-frozen, because this
                # account has NO live API data entitlement (verified
                # 2026-06-25: type 1 returns nan + Error 10089 "requires
                # additional subscription for API"; only delayed works). Once
                # a real live subscription is active for the API, set
                # AIT_MARKET_DATA_TYPE=1 to switch to real-time.
                import os
                _mdt = int(os.environ.get("AIT_MARKET_DATA_TYPE", "4"))
                self._ib.reqMarketDataType(_mdt)
                log.info("market_data_type_set", type=_mdt)

                account = self._config.ibkr_account or (
                    self._ib.managedAccounts()[0] if self._ib.managedAccounts() else "unknown"
                )
                log.info(
                    "ibkr_connected",
                    host=self._config.ibkr_host,
                    port=self._config.ibkr_port,
                    account=account,
                    client_id=client_id,
                    fallback=client_id != base_id,
                )
                return True

            except Exception as e:
                # Clean up a half-open socket before trying the next id.
                try:
                    if self._ib.isConnected():
                        self._ib.disconnect()
                except Exception:
                    pass
                log.warning("ibkr_connect_attempt_failed",
                            client_id=client_id, error=str(e) or type(e).__name__)
                continue

        log.error("ibkr_connection_failed", tried_ids=candidate_ids)
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

    async def verify_can_trade(self) -> bool:
        """Detect whether the Gateway session can actually place orders.

        The recurring silent killer: Gateway logs in READ-ONLY (session-level,
        not the API checkbox), so every order is rejected with Error 321 while
        the bot still logs 'trade_executed' — nothing fills, and it's invisible
        until end of day. This probes it directly by placing a guaranteed
        non-fillable limit (BUY 1 SPY @ $1, far below market) and checking
        whether Gateway ACCEPTS it (PreSubmitted/Submitted) or rejects it as
        read-only. The probe order is cancelled immediately; on a paper account
        it can never fill. Returns True if orders work, False if read-only.
        """
        if not self.connected:
            return False
        try:
            from ib_insync import LimitOrder, Stock
            spy = Stock("SPY", "SMART", "USD")
            q = await self._ib.qualifyContractsAsync(spy)
            if not q:
                return True  # can't probe; don't false-alarm
            order = LimitOrder("BUY", 1, 1.00, tif="DAY")
            trade = self._ib.placeOrder(q[0], order)
            await asyncio.sleep(2.5)
            msgs = " ".join(le.message for le in trade.log if le.message)
            status = trade.orderStatus.status
            read_only = "Read-Only" in msgs or "read-only" in msgs.lower()
            try:
                self._ib.cancelOrder(order)  # never leave the probe live
            except Exception:
                pass
            if read_only:
                log.critical("trading_blocked_read_only",
                             detail="Gateway session is READ-ONLY — orders rejected (Error 321). "
                                    "Restart IB Gateway and log in as IB API to clear.")
                return False
            if status in ("PreSubmitted", "Submitted", "Filled", "Cancelled"):
                log.info("trading_verified", status=status)
                return True
            return True  # unknown/transient — don't false-alarm
        except Exception as e:  # noqa: BLE001
            log.warning("verify_can_trade_error", error=str(e))
            return True  # probe failed for another reason; don't block

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

    async def _fx_usd_to(self, ccy: str) -> float | None:
        """Fetch the USD->ccy spot rate (units of ccy per 1 USD), cached.

        Used when the account is non-USD base and IBKR sends no ExchangeRate
        rows, so balances can be converted to USD instead of being treated
        1:1. The rate barely moves intraday, so cache it for the session.
        """
        if ccy == "USD":
            return 1.0
        if ccy in self._fx_cache:
            return self._fx_cache[ccy]
        try:
            from ib_insync import Forex
            contract = Forex(f"USD{ccy}")  # e.g. USDCAD => CAD per 1 USD
            await self._ib.qualifyContractsAsync(contract)
            tickers = await self._ib.reqTickersAsync(contract)
            rate = None
            if tickers:
                t = tickers[0]
                for cand in (t.marketPrice(), t.close, t.last, t.bid, t.ask):
                    if cand and cand == cand and cand > 0:  # not None/NaN/<=0
                        rate = float(cand)
                        break
            if rate and rate > 0:
                self._fx_cache[ccy] = rate
                log.info("fx_rate_fetched", pair=f"USD{ccy}", rate=round(rate, 4))
                return rate
            log.warning("fx_rate_unavailable", pair=f"USD{ccy}")
        except Exception as e:  # noqa: BLE001
            log.warning("fx_fetch_failed", ccy=ccy, error=str(e))
        return None

    async def get_account_values(self) -> dict[str, str]:
        """Get account summary values, normalized to USD.

        IBKR reports each tag once per currency. For a CAD-base account,
        NetLiquidation/BuyingPower/etc. arrive in CAD only, while option
        costs downstream are in USD — comparing them directly silently
        skewed every risk limit. Here we pick the authoritative base-currency
        total for each tag and convert it to USD using IBKR's own
        ExchangeRate, so the whole risk layer is denominated consistently.

        ExchangeRate for currency C = units of BASE per 1 unit of C. The USD
        rate (e.g. 1.3976 CAD per USD) converts base totals to USD via
        division. A USD-base account has rate 1.0 and is unchanged.
        """
        if not await self.ensure_connected():
            return {}
        raw = self._ib.accountValues()

        # FX + base-currency detection from the ExchangeRate rows.
        usd_to_base = 1.0
        base_ccy: str | None = None
        for av in raw:
            if av.tag != "ExchangeRate":
                continue
            try:
                rate = float(av.value)
            except ValueError:
                continue
            if av.currency == "USD":
                usd_to_base = rate
            elif av.currency not in ("BASE", "USD", "") and abs(rate - 1.0) < 1e-9:
                base_ccy = av.currency  # base currency has rate 1.0

        # Fallback base-currency detection: some accounts (e.g. this CAD-base
        # paper account, DUN603821) send NO ExchangeRate rows at all, so the
        # loop above leaves base_ccy=None. Without it, every core balance tag
        # (reported only in CAD) is dropped below and NetLiquidation resolves
        # to 0 — which silently zeroes the whole risk layer and blocks ALL
        # trading. Infer the base currency from the currency the core balance
        # tags are actually denominated in.
        if base_ccy is None:
            from collections import Counter
            ccy_counts = Counter(
                av.currency for av in raw
                if av.tag in ("NetLiquidation", "BuyingPower", "AvailableFunds",
                              "ExcessLiquidity", "TotalCashValue", "CashBalance")
                and av.currency not in ("BASE", "USD", "")
            )
            if ccy_counts:
                base_ccy = ccy_counts.most_common(1)[0][0]
                # No ExchangeRate row → fetch the real USD->base spot so
                # balances convert correctly instead of being treated 1:1
                # (which overstated buying power). Fall back to 1.0 (raw, with
                # a warning) only if the FX fetch fails — still better than 0.
                fx = await self._fx_usd_to(base_ccy)
                if fx and fx > 0:
                    usd_to_base = fx
                    log.info("account_base_ccy_inferred_fx", base_ccy=base_ccy,
                             usd_to_base=round(fx, 4))
                else:
                    log.warning("account_base_ccy_inferred_no_fx",
                                base_ccy=base_ccy,
                                note="no ExchangeRate row and FX fetch failed; "
                                     "using base value as USD (approx, overstated)")

        # Collect each tag's base-currency total (the "BASE" row is the
        # authoritative multi-currency total; the base-currency-code row is
        # the fallback). Keep USD-only rows separately for tags that have no
        # base total at all.
        base_vals: dict[str, str] = {}
        usd_vals: dict[str, str] = {}
        for av in raw:
            if av.currency == "BASE":
                base_vals[av.tag] = av.value           # always wins
            elif base_ccy and av.currency == base_ccy:
                base_vals.setdefault(av.tag, av.value)  # only if no BASE row
            elif av.currency == "USD":
                usd_vals[av.tag] = av.value

        values: dict[str, str] = {}
        for tag in set(base_vals) | set(usd_vals):
            if tag in base_vals:
                try:
                    values[tag] = f"{float(base_vals[tag]) / usd_to_base:.2f}"
                except (ValueError, ZeroDivisionError):
                    values[tag] = base_vals[tag]
            else:
                values[tag] = usd_vals[tag]  # already USD

        log.info("account_values_fetched", count=len(values), raw_count=len(raw),
                 net_liquidation=values.get("NetLiquidation", "missing"),
                 base_currency=base_ccy or "USD", usd_to_base=round(usd_to_base, 4))
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

"""Options chain service — fetch, filter, and price options with Greeks.

Fetches options chains from IBKR (primary) with Yahoo Finance fallback.
Calculates Greeks using py_vollib for proper Black-Scholes pricing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ait.broker.ibkr_client import IBKRClient
from ait.config.settings import OptionsConfig
from ait.data.cache import TTLCache
from ait.data.market_data import MarketDataService
from ait.utils.logging import get_logger

log = get_logger("data.options")


@dataclass
class OptionContract:
    """A single option contract with market data and Greeks."""

    symbol: str
    expiry: date
    strike: float
    right: str  # "C" or "P"
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_vol: float
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    # Metadata
    con_id: int = 0  # IBKR contract ID (needed for combo orders)

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_pct(self) -> float:
        if self.mid <= 0:
            return 1.0  # Treat as illiquid
        return (self.ask - self.bid) / self.mid

    @property
    def dte(self) -> int:
        return (self.expiry - date.today()).days

    @property
    def is_liquid(self) -> bool:
        return self.volume >= 50 and self.open_interest >= 100 and self.spread_pct < 0.15


@dataclass
class OptionsChain:
    """Full options chain for a symbol at a specific expiry."""

    symbol: str
    underlying_price: float
    expiry: date
    calls: list[OptionContract]
    puts: list[OptionContract]

    @property
    def dte(self) -> int:
        return (self.expiry - date.today()).days

    def get_atm_strike(self) -> float:
        """Get the at-the-money strike (closest to underlying price)."""
        all_strikes = set()
        for c in self.calls:
            all_strikes.add(c.strike)
        for p in self.puts:
            all_strikes.add(p.strike)
        if not all_strikes:
            return self.underlying_price
        return min(all_strikes, key=lambda s: abs(s - self.underlying_price))

    def filter_by_delta(self, min_delta: float, max_delta: float) -> OptionsChain:
        """Filter contracts by absolute delta range."""
        return OptionsChain(
            symbol=self.symbol,
            underlying_price=self.underlying_price,
            expiry=self.expiry,
            calls=[c for c in self.calls if min_delta <= abs(c.delta) <= max_delta],
            puts=[p for p in self.puts if min_delta <= abs(p.delta) <= max_delta],
        )

    def filter_liquid(self, config: OptionsConfig) -> OptionsChain:
        """Filter to only liquid contracts based on config thresholds."""
        return OptionsChain(
            symbol=self.symbol,
            underlying_price=self.underlying_price,
            expiry=self.expiry,
            calls=[
                c
                for c in self.calls
                if c.volume >= config.min_volume
                and c.open_interest >= config.min_open_interest
                and c.spread_pct <= config.max_bid_ask_spread_pct
            ],
            puts=[
                p
                for p in self.puts
                if p.volume >= config.min_volume
                and p.open_interest >= config.min_open_interest
                and p.spread_pct <= config.max_bid_ask_spread_pct
            ],
        )


class OptionsChainService:
    """Fetches and processes options chains from IBKR and Yahoo Finance."""

    def __init__(
        self,
        ibkr_client: IBKRClient,
        market_data: MarketDataService,
        config: OptionsConfig,
    ) -> None:
        self._ibkr = ibkr_client
        self._market_data = market_data
        self._config = config
        self._cache = TTLCache(default_ttl=120)  # 2 min cache for chains

    async def get_chain(
        self,
        symbol: str,
        min_dte: int | None = None,
        max_dte: int | None = None,
    ) -> list[OptionsChain]:
        """Get options chains for a symbol within DTE range.

        Returns one OptionsChain per expiry date that falls within range.
        """
        min_dte = min_dte or self._config.dte_range[0]
        max_dte = max_dte or self._config.dte_range[1]

        cache_key = f"chain:{symbol}:{min_dte}:{max_dte}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Get underlying price
        price = await self._market_data.get_current_price(symbol)
        if price is None:
            log.warning("cannot_get_chain", symbol=symbol, reason="no underlying price")
            return []

        # Try IBKR first
        chains = await self._get_ibkr_chain(symbol, price, min_dte, max_dte)

        # Fallback to Yahoo
        if not chains:
            chains = await self._get_yahoo_chain(symbol, price, min_dte, max_dte)

        if chains:
            # Calculate Greeks for all contracts
            for chain in chains:
                self._calculate_greeks(chain)

            self._cache.set(cache_key, chains, ttl=120)

        return chains

    async def get_chain_for_expiry(
        self, symbol: str, expiry: date
    ) -> OptionsChain | None:
        """Get a specific expiry's options chain."""
        chains = await self.get_chain(
            symbol,
            min_dte=(expiry - date.today()).days - 1,
            max_dte=(expiry - date.today()).days + 1,
        )
        for chain in chains:
            if chain.expiry == expiry:
                return chain
        return chains[0] if chains else None

    # --- Private methods ---

    async def _get_ibkr_chain(
        self, symbol: str, price: float, min_dte: int, max_dte: int
    ) -> list[OptionsChain]:
        """Fetch options chain from IBKR."""
        if not self._ibkr.connected:
            return []

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            qualified = await self._ibkr.qualify_contract(contract)
            if not qualified:
                return []

            # Get available option chains (use async version to avoid blocking event loop)
            chains_data = await self._ibkr.ib.reqSecDefOptParamsAsync(
                qualified.symbol, "", qualified.secType, qualified.conId
            )

            if not chains_data:
                return []

            # Find SMART exchange chain
            chain_def = None
            for cd in chains_data:
                if cd.exchange == "SMART":
                    chain_def = cd
                    break
            if chain_def is None:
                chain_def = chains_data[0]

            # Filter expiries by DTE range
            today = date.today()
            target_expiries = []
            for exp_str in sorted(chain_def.expirations):
                exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
                dte = (exp_date - today).days
                if min_dte <= dte <= max_dte:
                    target_expiries.append(exp_date)

            if not target_expiries:
                return []

            # Filter strikes around ATM (±20% of price)
            strike_range = price * 0.20
            target_strikes = [
                s for s in chain_def.strikes if abs(s - price) <= strike_range
            ]

            # Fetch option contracts for each expiry
            chains = []
            for exp_date in target_expiries[:3]:  # Limit to 3 closest expiries
                chain = await self._fetch_ibkr_expiry(
                    symbol, qualified, exp_date, target_strikes, price
                )
                if chain:
                    chains.append(chain)

            return chains

        except Exception as e:
            log.warning("ibkr_chain_failed", symbol=symbol, error=str(e))
            return []

    async def _fetch_ibkr_expiry(
        self,
        symbol: str,
        underlying,
        expiry: date,
        strikes: list[float],
        price: float,
    ) -> OptionsChain | None:
        """Fetch a single expiry's chain from IBKR."""
        from ib_insync import Option

        calls = []
        puts = []
        exp_str = expiry.strftime("%Y%m%d")

        # Build option contracts for qualification
        option_contracts = []
        for strike in strikes:
            for right in ("C", "P"):
                opt = Option(symbol, exp_str, strike, right, "SMART")
                option_contracts.append(opt)

        # Qualify in batches to avoid timeout
        batch_size = 50
        for i in range(0, len(option_contracts), batch_size):
            batch = option_contracts[i : i + batch_size]
            try:
                qualified = await self._ibkr.ib.qualifyContractsAsync(*batch)
                for q in qualified:
                    if q.conId == 0:
                        continue
                    # Request market data
                    self._ibkr.ib.reqMktData(q, "", True, False)

                await asyncio.sleep(0.5)

                for q in qualified:
                    if q.conId == 0:
                        continue
                    ticker = self._ibkr.ib.ticker(q)
                    if not ticker:
                        continue

                    contract = OptionContract(
                        symbol=symbol,
                        expiry=expiry,
                        strike=q.strike,
                        right=q.right,
                        bid=ticker.bid if ticker.bid and ticker.bid > 0 else 0.0,
                        ask=ticker.ask if ticker.ask and ticker.ask > 0 else 0.0,
                        last=ticker.last if ticker.last and ticker.last > 0 else 0.0,
                        volume=int(ticker.volume) if ticker.volume else 0,
                        open_interest=0,  # IBKR doesn't provide OI in real-time
                        implied_vol=(
                            ticker.modelGreeks.impliedVol
                            if ticker.modelGreeks
                            else 0.0
                        ),
                        delta=(
                            ticker.modelGreeks.delta if ticker.modelGreeks else 0.0
                        ),
                        gamma=(
                            ticker.modelGreeks.gamma if ticker.modelGreeks else 0.0
                        ),
                        theta=(
                            ticker.modelGreeks.theta if ticker.modelGreeks else 0.0
                        ),
                        vega=(
                            ticker.modelGreeks.vega if ticker.modelGreeks else 0.0
                        ),
                        con_id=q.conId,
                    )

                    if q.right == "C":
                        calls.append(contract)
                    else:
                        puts.append(contract)

            except Exception as e:
                log.debug("ibkr_batch_failed", symbol=symbol, batch=i, error=str(e))

        if not calls and not puts:
            return None

        return OptionsChain(
            symbol=symbol,
            underlying_price=price,
            expiry=expiry,
            calls=sorted(calls, key=lambda c: c.strike),
            puts=sorted(puts, key=lambda p: p.strike),
        )

    def _fetch_yahoo_chain_sync(
        self, symbol: str, price: float, min_dte: int, max_dte: int
    ) -> list[OptionsChain]:
        """Synchronous Yahoo Finance chain fetch (runs in executor)."""
        import math

        import yfinance as yf

        ticker = yf.Ticker(symbol)
        expiry_strings = ticker.options  # List of expiry date strings

        if not expiry_strings:
            return []

        today = date.today()
        chains = []

        def _safe_int(val, default=0):
            try:
                v = float(val) if val is not None else default
                return default if math.isnan(v) else int(v)
            except (TypeError, ValueError):
                return default

        def _safe_float(val, default=0.0):
            try:
                v = float(val) if val is not None else default
                return default if math.isnan(v) else v
            except (TypeError, ValueError):
                return default

        for exp_str in expiry_strings:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if not (min_dte <= dte <= max_dte):
                continue

            try:
                opt_chain = ticker.option_chain(exp_str)
            except Exception:
                continue

            calls = []
            for _, row in opt_chain.calls.iterrows():
                strike = _safe_float(row.get("strike"))
                if not strike or abs(strike - price) > price * 0.20:
                    continue
                calls.append(
                    OptionContract(
                        symbol=symbol,
                        expiry=exp_date,
                        strike=strike,
                        right="C",
                        bid=_safe_float(row.get("bid")),
                        ask=_safe_float(row.get("ask")),
                        last=_safe_float(row.get("lastPrice")),
                        volume=_safe_int(row.get("volume")),
                        open_interest=_safe_int(row.get("openInterest")),
                        implied_vol=_safe_float(row.get("impliedVolatility")),
                    )
                )

            puts = []
            for _, row in opt_chain.puts.iterrows():
                strike = _safe_float(row.get("strike"))
                if not strike or abs(strike - price) > price * 0.20:
                    continue
                puts.append(
                    OptionContract(
                        symbol=symbol,
                        expiry=exp_date,
                        strike=strike,
                        right="P",
                        bid=_safe_float(row.get("bid")),
                        ask=_safe_float(row.get("ask")),
                        last=_safe_float(row.get("lastPrice")),
                        volume=_safe_int(row.get("volume")),
                        open_interest=_safe_int(row.get("openInterest")),
                        implied_vol=_safe_float(row.get("impliedVolatility")),
                    )
                )

            if calls or puts:
                chains.append(
                    OptionsChain(
                        symbol=symbol,
                        underlying_price=price,
                        expiry=exp_date,
                        calls=sorted(calls, key=lambda c: c.strike),
                        puts=sorted(puts, key=lambda p: p.strike),
                    )
                )

            if len(chains) >= 3:
                break

        return chains

    async def _get_yahoo_chain(
        self, symbol: str, price: float, min_dte: int, max_dte: int
    ) -> list[OptionsChain]:
        """Fetch options chain from Yahoo Finance as fallback."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._fetch_yahoo_chain_sync, symbol, price, min_dte, max_dte
            )
        except Exception as e:
            log.warning("yahoo_chain_failed", symbol=symbol, error=str(e))
            return []

    def _calculate_greeks(self, chain: OptionsChain) -> None:
        """Calculate Greeks using Black-Scholes for contracts missing them."""
        try:
            from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
            from py_vollib.black_scholes.implied_volatility import implied_volatility
        except ImportError:
            log.debug("py_vollib_not_available", action="skipping greeks calculation")
            return

        S = chain.underlying_price
        r = 0.05  # Risk-free rate approximation
        t = max(chain.dte / 365.0, 0.001)  # Time to expiry in years

        for contracts in (chain.calls, chain.puts):
            for c in contracts:
                if c.delta != 0:
                    continue  # Already has Greeks (from IBKR)

                flag = "c" if c.right == "C" else "p"
                sigma = c.implied_vol if c.implied_vol > 0 else 0.30  # Default 30% vol

                try:
                    c.delta = delta(flag, S, c.strike, t, r, sigma)
                    c.gamma = gamma(flag, S, c.strike, t, r, sigma)
                    c.theta = theta(flag, S, c.strike, t, r, sigma) / 365.0  # Daily theta
                    c.vega = vega(flag, S, c.strike, t, r, sigma) / 100.0  # Per 1% vol move
                except Exception:
                    pass  # Leave as 0 if calculation fails

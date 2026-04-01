"""Unified market data service with fallback chain.

Data source priority: IBKR → Polygon (free tier) → Yahoo Finance
Each source has proper error handling — if one fails, the next is tried.
NO mock/fake data is ever returned.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from ib_insync import Stock, util

from ait.broker.ibkr_client import IBKRClient
from ait.data.cache import TTLCache
from ait.utils.logging import get_logger

log = get_logger("data.market")


@dataclass
class Quote:
    """Real-time quote for a symbol."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_pct(self) -> float:
        if self.mid <= 0:
            return 0.0
        return (self.ask - self.bid) / self.mid


class MarketDataService:
    """Fetches market data with IBKR → Polygon → Yahoo fallback chain."""

    def __init__(
        self,
        ibkr_client: IBKRClient,
        polygon_api_key: str = "",
        cache_ttl: int = 60,
    ) -> None:
        self._ibkr = ibkr_client
        self._polygon_key = polygon_api_key
        self._cache = TTLCache(default_ttl=cache_ttl)
        self._polygon_client = None

        if polygon_api_key:
            try:
                from polygon import RESTClient

                self._polygon_client = RESTClient(api_key=polygon_api_key)
                log.info("polygon_client_initialized")
            except ImportError:
                log.warning("polygon_package_not_installed")

    async def get_quote(self, symbol: str) -> Quote | None:
        """Get real-time quote. Tries IBKR first, then Yahoo."""
        cached = self._cache.get(f"quote:{symbol}")
        if cached:
            return cached

        # Try IBKR
        quote = await self._get_ibkr_quote(symbol)

        # Fallback to Yahoo
        if quote is None:
            quote = await self._get_yahoo_quote(symbol)

        if quote:
            self._cache.set(f"quote:{symbol}", quote, ttl=15)  # 15s cache for quotes

        return quote

    async def get_historical(
        self,
        symbol: str,
        days: int = 252,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """Get historical OHLCV data.

        Tries Polygon first (better data), then Yahoo Finance.
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        """
        cache_key = f"hist:{symbol}:{days}:{interval}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        df = None

        # Try Polygon (free tier: 2 years of daily data)
        if self._polygon_client and interval == "1d":
            df = await self._get_polygon_historical(symbol, days)

        # Fallback to Yahoo
        if df is None:
            df = await self._get_yahoo_historical(symbol, days, interval)

        if df is not None and not df.empty:
            self._cache.set(cache_key, df, ttl=3600)  # 1hr cache for daily data

        return df

    async def get_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        days: int = 1,
    ) -> pd.DataFrame | None:
        """Get intraday OHLCV data (5-min bars).

        Used for multi-timeframe analysis and entry timing.
        Yahoo Finance provides 5-min data for the last 60 days.
        """
        cache_key = f"intraday:{symbol}:{interval}:{days}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            period = f"{days}d" if days <= 5 else "1mo"
            loop = asyncio.get_running_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            df = await loop.run_in_executor(
                None, lambda: ticker.history(period=period, interval=interval)
            )

            if df is None or df.empty:
                return None

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "Datetime"

            # Cache for 5 minutes (intraday data changes frequently)
            self._cache.set(cache_key, df, ttl=300)
            return df

        except Exception as e:
            log.debug("intraday_fetch_failed", symbol=symbol, interval=interval, error=str(e))
            return None

    async def get_current_price(self, symbol: str) -> float | None:
        """Get the current price for a symbol."""
        quote = await self.get_quote(symbol)
        if quote:
            return quote.mid
        return None

    async def get_vix(self) -> float | None:
        """Get current VIX level."""
        # Try IBKR first — VIX is an index on CBOE, not a stock
        if self._ibkr and self._ibkr.connected:
            try:
                from ib_insync import Index
                contract = Index("VIX", "CBOE", "USD")
                qualified = await self._ibkr.qualify_contract(contract)
                if qualified:
                    self._ibkr.ib.reqMktData(qualified, "", False, False)
                    await asyncio.sleep(0.5)
                    ticker = self._ibkr.ib.ticker(qualified)
                    if ticker and not math.isnan(ticker.last) and ticker.last > 0:
                        return float(ticker.last)
            except Exception as e:
                log.debug("vix_ibkr_failed", error=str(e))

        # Yahoo fallback for VIX
        try:
            loop = asyncio.get_running_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker("^VIX"))
            data = await loop.run_in_executor(None, lambda: ticker.history(period="1d"))
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception as e:
            log.warning("vix_fetch_failed", error=str(e))

        return None

    # --- Private data source methods ---

    async def _get_ibkr_quote(self, symbol: str) -> Quote | None:
        """Get quote from IBKR."""
        if not self._ibkr.connected:
            return None

        try:
            contract = Stock(symbol, "SMART", "USD")
            qualified = await self._ibkr.qualify_contract(contract)
            if not qualified:
                return None

            # Type 4 = delayed-frozen: uses live data when available,
            # falls back to frozen snapshot — avoids "competing session" on paper
            self._ibkr.ib.reqMarketDataType(4)
            self._ibkr.ib.reqMktData(qualified, "", False, False)
            await asyncio.sleep(0.5)  # Brief wait for data

            ticker = self._ibkr.ib.ticker(qualified)
            if ticker:
                bid = ticker.bid if not math.isnan(ticker.bid) else 0.0
                ask = ticker.ask if not math.isnan(ticker.ask) else 0.0
                last = ticker.last if not math.isnan(ticker.last) else 0.0
                volume = int(ticker.volume) if not math.isnan(ticker.volume) else 0

                if last > 0 or bid > 0:
                    return Quote(
                        symbol=symbol,
                        bid=bid if bid > 0 else 0.0,
                        ask=ask if ask > 0 else 0.0,
                        last=last if last > 0 else 0.0,
                        volume=volume,
                        timestamp=datetime.now(),
                    )
        except Exception as e:
            log.debug("ibkr_quote_failed", symbol=symbol, error=str(e))

        return None

    async def _get_polygon_historical(self, symbol: str, days: int) -> pd.DataFrame | None:
        """Get historical data from Polygon free tier."""
        if not self._polygon_client:
            return None

        try:
            end = date.today()
            start = end - timedelta(days=int(days * 1.5))  # Extra days for non-trading days

            loop = asyncio.get_running_loop()
            aggs = await loop.run_in_executor(
                None,
                lambda: list(
                    self._polygon_client.list_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="day",
                        from_=start.strftime("%Y-%m-%d"),
                        to=end.strftime("%Y-%m-%d"),
                        limit=50000,
                    )
                ),
            )

            if not aggs:
                return None

            df = pd.DataFrame(
                [
                    {
                        "Date": pd.Timestamp(a.timestamp, unit="ms"),
                        "Open": a.open,
                        "High": a.high,
                        "Low": a.low,
                        "Close": a.close,
                        "Volume": a.volume,
                    }
                    for a in aggs
                ]
            )
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            return df.tail(days)

        except Exception as e:
            log.debug("polygon_historical_failed", symbol=symbol, error=str(e))
            return None

    async def _get_yahoo_historical(
        self, symbol: str, days: int, interval: str
    ) -> pd.DataFrame | None:
        """Get historical data from Yahoo Finance."""
        try:
            # Map days to yfinance period
            if days <= 5:
                period = "5d"
            elif days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"
            elif days <= 180:
                period = "6mo"
            elif days <= 365:
                period = "1y"
            else:
                period = "2y"

            loop = asyncio.get_running_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            df = await loop.run_in_executor(
                None, lambda: ticker.history(period=period, interval=interval)
            )

            if df is None or df.empty:
                return None

            # Standardize columns
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "Date"
            return df.tail(days)

        except Exception as e:
            log.debug("yahoo_historical_failed", symbol=symbol, error=str(e))
            return None

    async def _get_yahoo_quote(self, symbol: str) -> Quote | None:
        """Get quote from Yahoo Finance (slower, but always available)."""
        try:
            loop = asyncio.get_running_loop()
            ticker = await loop.run_in_executor(None, lambda: yf.Ticker(symbol))
            info = await loop.run_in_executor(None, lambda: ticker.fast_info)

            last = float(info.last_price) if hasattr(info, "last_price") else 0.0
            if last <= 0:
                data = await loop.run_in_executor(None, lambda: ticker.history(period="1d"))
                if data.empty:
                    return None
                last = float(data["Close"].iloc[-1])

            return Quote(
                symbol=symbol,
                bid=0.0,  # Yahoo doesn't provide reliable bid/ask
                ask=0.0,
                last=last,
                volume=int(info.last_volume) if hasattr(info, "last_volume") else 0,
                timestamp=datetime.now(),
            )
        except Exception as e:
            log.debug("yahoo_quote_failed", symbol=symbol, error=str(e))
            return None

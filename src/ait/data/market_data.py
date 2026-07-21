"""Unified market data service with fallback chain.

PRICE BASIS (deep-audit DATA-H1): all Yahoo calls pass auto_adjust=False
so Close is split-adjusted but NOT dividend-adjusted — matching IBKR
TRADES and Polygon. Mixed bases created fake step-returns at ex-div
dates that corrupted features/labels for dividend payers (SPY/GLD/TLT..).

Data source priority: IBKR → Yahoo Finance
Each source has proper error handling — if one fails, the next is tried.
NO mock/fake data is ever returned.

For daily OHLCV used in ML training and backtesting, use load_daily_ohlcv()
which reads from the IB SQLite store (resampled from 5-min bars) and falls
back to Yahoo Finance only when insufficient IB data is available.
"""

from __future__ import annotations

import asyncio
import math
import os
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


def load_daily_ohlcv(
    symbol: str,
    days: int = 730,
    db_path: "Path | None" = None,
) -> pd.DataFrame:
    """Load daily OHLCV — IB SQLite store first, Yahoo Finance fallback.

    Tries to resample stored 5-min bars to daily OHLCV. Falls back to
    Yahoo Finance when fewer than 60 trading days are available in the store.

    Also left-joins stored IBKR implied_vol snapshots (from daily IV backfill)
    onto the returned DataFrame as an `implied_vol` column. Rows without stored
    IV have NaN in that column; callers should check before using.

    Args:
        symbol:  Ticker symbol.
        days:    Calendar days of history requested (default 2 years).
        db_path: Path to SQLite DB; uses production DB_PATH if None.

    Returns:
        DataFrame with DatetimeIndex and [Open, High, Low, Close, Volume, implied_vol].
    """
    from ait.data.historical import HistoricalDataStore, DB_PATH as _DEFAULT_DB

    store = HistoricalDataStore(db_path=db_path or _DEFAULT_DB)
    df = store.resample_to_daily(symbol, days=days)

    if len(df) < 60:
        log.info("daily_ohlcv_yahoo_fallback", symbol=symbol, ib_rows=len(df))
        try:
            start = (date.today() - timedelta(days=days + 30)).isoformat()
            ydf = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=False)
            if not ydf.empty:
                df = ydf[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception as exc:
            log.warning("yahoo_fallback_failed", symbol=symbol, error=str(exc))
            if df.empty:
                return pd.DataFrame()
    else:
        log.info("daily_ohlcv_from_ib_store", symbol=symbol, rows=len(df))

    if df.empty:
        return df

    # Left-join stored IBKR implied_vol snapshots (NaN where not available)
    try:
        iv_series = store.load_daily_iv(symbol, days=days)
        if not iv_series.empty:
            # Align index timezone: df may be tz-naive or tz-aware
            iv_idx = iv_series.index
            df_idx = df.index
            df_tz = getattr(df_idx, "tz", None)
            iv_tz = getattr(iv_idx, "tz", None)
            if df_tz is not None and iv_tz is None:
                iv_idx = iv_idx.tz_localize(df_tz)
            elif df_tz is None and iv_tz is not None:
                iv_idx = iv_idx.tz_localize(None)
            elif df_tz is not None and iv_tz is not None and str(iv_tz) != str(df_tz):
                iv_idx = iv_idx.tz_convert(df_tz)
            iv_aligned = pd.Series(iv_series.values, index=iv_idx, name="implied_vol")
            df = df.join(iv_aligned, how="left")
        else:
            df["implied_vol"] = float("nan")
    except Exception as exc:
        log.debug("iv_join_failed", symbol=symbol, error=str(exc))
        df["implied_vol"] = float("nan")

    return df


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
        # R9 single-flight: per-cache-key asyncio.Lock so N concurrent cache
        # misses for the same key result in ONE network fetch (the rest wait,
        # then hit the freshly-set cache). Created lazily per key.
        self._fetch_locks: dict[str, asyncio.Lock] = {}

        if polygon_api_key:
            try:
                from polygon import RESTClient

                self._polygon_client = RESTClient(api_key=polygon_api_key)
                log.info("polygon_client_initialized")
            except ImportError:
                log.warning("polygon_package_not_installed")

    def _lock_for(self, key: str) -> asyncio.Lock:
        """Return the single-flight lock for a cache key (creating if needed).

        Bounded: when the dict grows past 512 entries, locks nobody currently
        holds are pruned (a pruned-then-recreated lock only risks one
        duplicate fetch, never corruption).
        """
        if len(self._fetch_locks) > 512:
            for k in [k for k, lk in self._fetch_locks.items() if not lk.locked()]:
                self._fetch_locks.pop(k, None)
        return self._fetch_locks.setdefault(key, asyncio.Lock())

    async def get_quote(self, symbol: str) -> Quote | None:
        """Get real-time quote. Tries IBKR first, then Yahoo."""
        cache_key = f"quote:{symbol}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        async with self._lock_for(cache_key):
            # Double-check: another task may have populated the cache while
            # we waited on the single-flight lock.
            cached = self._cache.get(cache_key)
            if cached:
                return cached

            # Try IBKR
            quote = await self._get_ibkr_quote(symbol)

            # Fallback to Yahoo
            if quote is None:
                quote = await self._get_yahoo_quote(symbol)

            if quote:
                self._cache.set(cache_key, quote, ttl=15)  # 15s cache for quotes

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

        async with self._lock_for(cache_key):
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
        days: int = 7,
    ) -> pd.DataFrame | None:
        """Get intraday OHLCV data (5-min bars). IBKR → Yahoo fallback.

        IBKR is the primary source for consistent training/trading data.
        Yahoo Finance is used only when IBKR is unavailable.
        """
        cache_key = f"intraday:{symbol}:{interval}:{days}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        async with self._lock_for(cache_key):
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            df = None

            # Try IBKR first (consistent with live trading data)
            if interval == "5m":
                duration = self._days_to_ibkr_duration(days)
                df = await self._get_ibkr_intraday(symbol, duration=duration)

            # Fallback to Yahoo (max 60 days of 5-min data)
            if df is None:
                df = await self._get_yahoo_intraday(symbol, interval=interval, days=days)

            if df is not None and not df.empty:
                self._cache.set(cache_key, df, ttl=300)

            return df

    async def get_intraday_since(
        self,
        symbol: str,
        since: "pd.Timestamp",
    ) -> pd.DataFrame | None:
        """Get intraday 5-min bars strictly after `since` timestamp.

        Used by the orchestrator for incremental fetching — only downloads
        bars not yet stored in SQLite, rather than re-fetching the full window.
        """
        from datetime import timezone
        now = pd.Timestamp.now(tz=timezone.utc)
        since_utc = since.tz_convert("UTC") if since.tzinfo else since.tz_localize("UTC")
        elapsed_days = max(1, int((now - since_utc).total_seconds() / 86400) + 1)

        duration = self._days_to_ibkr_duration(elapsed_days)
        df = await self._get_ibkr_intraday(symbol, duration=duration)

        if df is None:
            df = await self._get_yahoo_intraday(symbol, interval="5m", days=elapsed_days)

        if df is not None and not df.empty:
            # Return only bars strictly after the cutoff
            if getattr(df.index, "tz", None) is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            return df[df.index > since_utc]

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
                    ticker = None
                    try:
                        await asyncio.sleep(0.5)
                        ticker = self._ibkr.ib.ticker(qualified)
                    finally:
                        # Pair req with cancel — a leaked streaming sub errors
                        # (10091) forever and floods the API thread → crash.
                        try:
                            self._ibkr.ib.cancelMktData(qualified)
                        except Exception:
                            pass
                    if ticker and not math.isnan(ticker.last) and ticker.last > 0:
                        return float(ticker.last)
            except Exception as e:
                log.debug("vix_ibkr_failed", error=str(e))

        # Yahoo fallback for VIX — entire fetch (Ticker construction included)
        # runs in a worker thread so the event loop keeps servicing the risk
        # monitor and ib_insync callbacks. yf.Ticker is created INSIDE the
        # thread: yfinance sessions are not safely shareable across threads.
        def _fetch_vix_sync() -> float | None:
            data = yf.Ticker("^VIX").history(period="1d", auto_adjust=False)
            if data.empty:
                return None
            return float(data["Close"].iloc[-1])

        try:
            vix = await asyncio.wait_for(asyncio.to_thread(_fetch_vix_sync), timeout=30.0)
            if vix is not None:
                return vix
        except Exception as e:
            log.warning("vix_fetch_failed", error=str(e))

        return None

    # --- Private data source methods ---

    @staticmethod
    def _days_to_ibkr_duration(days: int) -> str:
        """Convert a number of calendar days to an IBKR durationStr for 5-min bars."""
        if days <= 14:
            return f"{days} D"
        elif days <= 60:
            return "1 M"
        elif days <= 120:
            return "3 M"
        elif days <= 180:
            return "6 M"
        else:
            return "1 Y"

    async def _get_ibkr_intraday(
        self,
        symbol: str,
        duration: str = "7 D",
        bar_size: str = "5 mins",
    ) -> pd.DataFrame | None:
        """Fetch intraday bars from IBKR via reqHistoricalDataAsync.

        Returns DataFrame with UTC DatetimeIndex and OHLCV columns,
        or None if IBKR is unavailable or returns no data.
        """
        if not self._ibkr or not self._ibkr.connected:
            return None

        try:
            contract = Stock(symbol, "SMART", "USD")
            qualified = await self._ibkr.qualify_contract(contract)
            if not qualified:
                return None

            bars = await self._ibkr.ib.reqHistoricalDataAsync(
                qualified,
                endDateTime="",       # empty = now
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
            if not bars:
                return None

            # R9: the pandas transform (util.df on up to ~20k 5-min bars for
            # "1 Y" duration) blocked the event loop — offload it. This is NOT
            # an ib_insync API call: reqHistoricalDataAsync has already
            # completed and `bars` is a plain snapshot (keepUpToDate=False),
            # so no ib socket/loop state is touched from the worker thread.
            df = await asyncio.to_thread(self._ibkr_bars_to_df, list(bars))
            log.debug(
                "ibkr_intraday_fetched",
                symbol=symbol,
                duration=duration,
                bars=len(df),
            )
            return df

        except Exception as e:
            log.debug("ibkr_intraday_failed", symbol=symbol, duration=duration, error=str(e))
            return None

    @staticmethod
    def _ibkr_bars_to_df(bars: list) -> pd.DataFrame:
        """Pure-pandas conversion of IBKR BarData list → OHLCV DataFrame.

        Runs in a worker thread (see _get_ibkr_intraday) — must not touch
        any ib_insync connection state, only the already-fetched bar list.
        """
        df = util.df(bars)
        df = df.rename(columns={
            "date": "Datetime",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df.set_index("Datetime", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Datetime"
        return df

    async def _get_yahoo_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        days: int = 7,
    ) -> pd.DataFrame | None:
        """Get intraday data from Yahoo Finance (fallback; max 60 days of 5-min data)."""
        try:
            period = f"{min(days, 59)}d" if days <= 59 else "1mo"

            # Self-contained sync fetch + transform, offloaded to a worker
            # thread. yf.Ticker created INSIDE the thread (session safety).
            def _fetch_sync() -> pd.DataFrame | None:
                df = yf.Ticker(symbol).history(
                    period=period, interval=interval, auto_adjust=False
                )
                if df is None or df.empty:
                    return None
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index.name = "Datetime"
                if getattr(df.index, "tz", None) is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                return df

            df = await asyncio.wait_for(asyncio.to_thread(_fetch_sync), timeout=30.0)
            if df is None:
                return None
            log.debug("yahoo_intraday_fetched", symbol=symbol, bars=len(df))
            return df

        except Exception as e:
            log.debug("yahoo_intraday_failed", symbol=symbol, interval=interval, error=str(e))
            return None

    async def _get_ibkr_quote(self, symbol: str) -> Quote | None:
        """Get quote from IBKR."""
        if not self._ibkr.connected:
            return None

        try:
            contract = Stock(symbol, "SMART", "USD")
            qualified = await self._ibkr.qualify_contract(contract)
            if not qualified:
                return None

            # Market data type from env (same knob as ibkr_client): 1=live,
            # 4=delayed-frozen. Live now that Network B/C + OPRA are subscribed.
            self._ibkr.ib.reqMarketDataType(int(os.environ.get("AIT_MARKET_DATA_TYPE", "4")))
            self._ibkr.ib.reqMktData(qualified, "", False, False)
            ticker = None
            try:
                await asyncio.sleep(0.5)  # Brief wait for data
                ticker = self._ibkr.ib.ticker(qualified)
            finally:
                # Pair req with cancel — a leaked streaming sub errors (10091)
                # forever and floods the API thread → native crash.
                try:
                    self._ibkr.ib.cancelMktData(qualified)
                except Exception:
                    pass
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
                        # A2 (deep-audit DATA-M4): exchange tick time when
                        # available — wall-clock stamps made staleness
                        # detection blind (frozen quotes always looked fresh).
                        # R15 #6: normalize to naive LOCAL time. ticker.time is
                        # aware-UTC; the old .replace(tzinfo=None) kept the UTC
                        # wall-clock, while the Yahoo fallback stamps local
                        # now(). Consumers call .timestamp() (interprets naive
                        # as LOCAL), so IBKR quotes read 4h in the future —
                        # never stale to the R14 gate — and an IBKR->Yahoo
                        # source flip looked like time running backwards
                        # ("frozen"). astimezone() first = true local wall.
                        timestamp=(ticker.time.astimezone().replace(tzinfo=None)
                                   if getattr(ticker, "time", None) else datetime.now()),
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

            # Network iteration + DataFrame build both offloaded — the agg
            # list can be thousands of rows and the build blocked the loop.
            def _fetch_sync() -> pd.DataFrame | None:
                aggs = list(
                    self._polygon_client.list_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="day",
                        from_=start.strftime("%Y-%m-%d"),
                        to=end.strftime("%Y-%m-%d"),
                        limit=50000,
                    )
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

            return await asyncio.wait_for(asyncio.to_thread(_fetch_sync), timeout=30.0)

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

            # Self-contained sync fetch + transform in a worker thread.
            # yf.Ticker created INSIDE the thread (session safety).
            def _fetch_sync() -> pd.DataFrame | None:
                df = yf.Ticker(symbol).history(
                    period=period, interval=interval, auto_adjust=False
                )
                if df is None or df.empty:
                    return None
                # Standardize columns
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index.name = "Date"
                return df.tail(days)

            return await asyncio.wait_for(asyncio.to_thread(_fetch_sync), timeout=30.0)

        except Exception as e:
            log.debug("yahoo_historical_failed", symbol=symbol, error=str(e))
            return None

    async def _get_yahoo_quote(self, symbol: str) -> Quote | None:
        """Get quote from Yahoo Finance (slower, but always available)."""
        try:
            # Entire quote fetch is self-contained in one worker thread —
            # fast_info and the history() fallback are both blocking HTTP.
            # yf.Ticker created INSIDE the thread (session safety).
            def _fetch_sync() -> Quote | None:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info

                last = float(info.last_price) if hasattr(info, "last_price") else 0.0
                if last <= 0:
                    data = ticker.history(period="1d", auto_adjust=False)
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

            return await asyncio.wait_for(asyncio.to_thread(_fetch_sync), timeout=30.0)
        except Exception as e:
            log.debug("yahoo_quote_failed", symbol=symbol, error=str(e))
            return None

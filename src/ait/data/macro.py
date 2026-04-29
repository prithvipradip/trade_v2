"""Macro / cross-asset data fetcher.

Pulls free historical data from the St. Louis Fed (FRED). No API key
required for the CSV endpoint. Used as additional features for the ML
models — yield curves and dollar strength have known equity-impact
patterns the technical indicators miss.

Series fetched:
  DGS2   — 2-year Treasury yield
  DGS10  — 10-year Treasury yield
  T10Y2Y — 10y minus 2y spread (yield curve, recession signal)
  DTWEXBGS — Trade-weighted dollar index (broad)

These series cycle with monetary policy and have predictable equity
correlations:
  - Inverted yield curve (T10Y2Y < 0)  → recession risk → equities down
  - Strong dollar → headwinds for multinationals
  - Steepening curve → reflation, equities up
"""

from __future__ import annotations

from datetime import datetime, timedelta
from io import StringIO

import aiohttp
import pandas as pd

from ait.data.cache import TTLCache
from ait.utils.logging import get_logger

log = get_logger("data.macro")

FRED_SERIES = {
    "us_2y_yield":  "DGS2",        # 2-Year Treasury Constant Maturity Rate
    "us_10y_yield": "DGS10",       # 10-Year Treasury
    "yield_curve":  "T10Y2Y",      # 10Y minus 2Y spread
    "dxy":          "DTWEXBGS",    # Trade-weighted dollar (broad)
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


class MacroDataFetcher:
    """Fetches macro time series from FRED. Caches for 1 hour."""

    def __init__(self) -> None:
        # Cache for 1 hour — FRED updates daily
        self._cache = TTLCache(default_ttl=3600, max_size=20)

    async def fetch_series(self, series_id: str, lookback_days: int = 1825) -> pd.Series | None:
        """Fetch a single FRED series. Returns daily indexed pd.Series or None."""
        cached = self._cache.get(series_id)
        if cached is not None:
            return cached

        url = FRED_CSV_URL.format(series_id=series_id)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=15),
                    headers={"User-Agent": "ait-bot/2.0"},
                ) as resp:
                    if resp.status != 200:
                        log.warning("fred_fetch_failed", series=series_id, status=resp.status)
                        return None
                    csv_text = await resp.text()
        except Exception as e:
            log.warning("fred_fetch_error", series=series_id, error=str(e))
            return None

        try:
            df = pd.read_csv(StringIO(csv_text))
            # FRED CSVs have columns "DATE" + the series_id (or "observation_date")
            date_col = next((c for c in df.columns if c.lower() in ("date", "observation_date")), None)
            value_col = next((c for c in df.columns if c != date_col), None)
            if date_col is None or value_col is None:
                return None
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            # Replace "." (FRED's missing value marker) with NaN
            series = pd.to_numeric(df[value_col], errors="coerce")
            # Trim to lookback
            cutoff = datetime.now() - timedelta(days=lookback_days)
            series = series[series.index >= cutoff]
            series.name = series_id
            self._cache.set(series_id, series)
            return series
        except Exception as e:
            log.warning("fred_parse_failed", series=series_id, error=str(e))
            return None

    async def fetch_all(self, lookback_days: int = 1825) -> dict[str, pd.Series]:
        """Fetch all macro series. Returns dict keyed by friendly name."""
        out = {}
        for friendly, series_id in FRED_SERIES.items():
            data = await self.fetch_series(series_id, lookback_days=lookback_days)
            if data is not None and not data.empty:
                out[friendly] = data
        log.info("macro_fetched", count=len(out), series=list(out.keys()))
        return out

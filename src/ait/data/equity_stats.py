"""Equity fundamental statistics via yfinance → DuckDB.

Fetches per-symbol snapshot data (sector, P/E, beta, market cap, analyst
consensus) from yfinance and upserts it into the analytics DuckDB.

Design:
- Snapshot-only: each refresh overwrites the previous row (INSERT OR REPLACE).
- History accumulates by polling daily; we cannot backfill historical P/E ratios.
- Works offline / in backtesting — no live broker connection needed.
"""

from __future__ import annotations

from datetime import datetime

import yfinance as yf

from ait.monitoring.duckdb_analytics import DuckDBAnalytics
from ait.utils.logging import get_logger

log = get_logger("data.equity_stats")

_FIELD_MAP: dict[str, str] = {
    "longName":                     "company_name",
    "sector":                       "sector",
    "industry":                     "industry",
    "country":                      "country",
    "exchange":                     "exchange",
    "marketCap":                    "market_cap",
    "trailingPE":                   "pe_ratio",
    "forwardPE":                    "forward_pe",
    "priceToBook":                  "pb_ratio",
    "priceToSalesTrailing12Months": "ps_ratio",
    "enterpriseToEbitda":           "ev_ebitda",
    "trailingEps":                  "eps",
    "bookValue":                    "book_value_ps",
    "revenuePerShare":              "revenue_ps",
    "dividendYield":                "dividend_yield",
    "dividendRate":                 "dividend_rate",
    "payoutRatio":                  "payout_ratio",
    "beta":                         "beta",
    "fiftyTwoWeekHigh":             "week52_high",
    "fiftyTwoWeekLow":              "week52_low",
    "averageVolume":                "avg_volume_30d",
    "floatShares":                  "float_shares",
    "sharesOutstanding":            "shares_outstanding",
    "targetMeanPrice":              "analyst_target_mean",
    "targetHighPrice":              "analyst_target_high",
    "targetLowPrice":               "analyst_target_low",
    "recommendationKey":            "analyst_rating",
    "numberOfAnalystOpinions":      "analyst_count",
}

_STRING_COLS = {"company_name", "sector", "industry", "country", "exchange", "analyst_rating"}


class EquityStatsService:
    """Fetches yfinance fundamentals and upserts them into the analytics DuckDB."""

    def __init__(self, analytics: DuckDBAnalytics) -> None:
        self._analytics = analytics

    def refresh_symbol(self, symbol: str) -> bool:
        """Fetch one symbol from yfinance and upsert into DuckDB.

        Returns True on success, False if the fetch or upsert failed.
        """
        try:
            info = yf.Ticker(symbol).info
            if not info:
                log.warning("equity_stats_empty_info", symbol=symbol)
                return False
            stats = self._map_fields(symbol, info)
            self._analytics.upsert_equity_stats(stats)
            log.info(
                "equity_stats_refreshed",
                symbol=symbol,
                sector=stats.get("sector", ""),
                pe_ratio=stats.get("pe_ratio", 0),
            )
            return True
        except Exception as e:
            log.warning("equity_stats_fetch_failed", symbol=symbol, error=str(e))
            return False

    def refresh_all(self, symbols: list[str]) -> dict[str, bool]:
        """Refresh all symbols; returns {symbol: success} mapping."""
        results: dict[str, bool] = {}
        for symbol in symbols:
            results[symbol] = self.refresh_symbol(symbol)
        ok = sum(results.values())
        log.info("equity_stats_batch_complete", total=len(symbols), ok=ok, failed=len(symbols) - ok)
        return results

    def _map_fields(self, symbol: str, info: dict) -> dict:
        row: dict = {"symbol": symbol, "updated_at": datetime.utcnow()}
        for yf_key, col in _FIELD_MAP.items():
            raw = info.get(yf_key)
            if col in _STRING_COLS:
                row[col] = raw or ""
            else:
                row[col] = raw if isinstance(raw, (int, float)) and raw == raw else 0
        return row

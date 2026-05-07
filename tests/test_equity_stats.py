"""Tests for Feature 1: equity stats (EquityStatsService + DuckDB)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ait.data.equity_stats import EquityStatsService
from ait.monitoring.duckdb_analytics import DuckDBAnalytics


@pytest.fixture
def analytics(tmp_path: Path) -> DuckDBAnalytics:
    return DuckDBAnalytics(db_path=tmp_path / "test.duckdb")


@pytest.fixture
def service(analytics: DuckDBAnalytics) -> EquityStatsService:
    return EquityStatsService(analytics)


# ---------------------------------------------------------------------------
# DuckDB schema + CRUD
# ---------------------------------------------------------------------------

class TestEquityStatsDuckDB:
    def test_upsert_and_get_single(self, analytics: DuckDBAnalytics):
        row = {
            "symbol": "SPY", "updated_at": datetime.utcnow(),
            "company_name": "SPDR S&P 500 ETF", "sector": "ETF", "industry": "ETF",
            "country": "US", "exchange": "NYSE", "market_cap": 500_000_000_000,
            "pe_ratio": 22.5, "forward_pe": 20.1, "pb_ratio": 4.5, "ps_ratio": 2.1,
            "ev_ebitda": 15.0, "eps": 18.0, "book_value_ps": 110.0, "revenue_ps": 250.0,
            "dividend_yield": 0.013, "dividend_rate": 6.5, "payout_ratio": 0.35,
            "beta": 1.0, "week52_high": 580.0, "week52_low": 420.0,
            "avg_volume_30d": 80_000_000, "float_shares": 900_000_000,
            "shares_outstanding": 910_000_000,
            "analyst_target_mean": 550.0, "analyst_target_high": 600.0,
            "analyst_target_low": 500.0, "analyst_rating": "buy", "analyst_count": 12,
        }
        analytics.upsert_equity_stats(row)
        result = analytics.get_equity_stats("SPY")
        assert result is not None
        assert result["symbol"] == "SPY"
        assert result["sector"] == "ETF"
        assert result["pe_ratio"] == pytest.approx(22.5)
        assert result["analyst_rating"] == "buy"

    def test_get_missing_symbol_returns_none(self, analytics: DuckDBAnalytics):
        assert analytics.get_equity_stats("NONEXISTENT") is None

    def test_upsert_overwrites_on_refresh(self, analytics: DuckDBAnalytics):
        base = {
            "symbol": "QQQ", "updated_at": datetime.utcnow(),
            "company_name": "Invesco QQQ", "sector": "ETF", "industry": "ETF",
            "country": "US", "exchange": "NASDAQ", "market_cap": 200_000_000_000,
            "pe_ratio": 30.0, "forward_pe": 27.0, "pb_ratio": 5.0, "ps_ratio": 3.0,
            "ev_ebitda": 18.0, "eps": 15.0, "book_value_ps": 80.0, "revenue_ps": 200.0,
            "dividend_yield": 0.005, "dividend_rate": 2.0, "payout_ratio": 0.10,
            "beta": 1.15, "week52_high": 500.0, "week52_low": 380.0,
            "avg_volume_30d": 50_000_000, "float_shares": 500_000_000,
            "shares_outstanding": 510_000_000,
            "analyst_target_mean": 450.0, "analyst_target_high": 490.0,
            "analyst_target_low": 400.0, "analyst_rating": "hold", "analyst_count": 5,
        }
        analytics.upsert_equity_stats(base)

        updated = dict(base)
        updated["pe_ratio"] = 35.0
        updated["analyst_rating"] = "strong buy"
        analytics.upsert_equity_stats(updated)

        result = analytics.get_equity_stats("QQQ")
        assert result["pe_ratio"] == pytest.approx(35.0)
        assert result["analyst_rating"] == "strong buy"

    def test_get_all_returns_multiple(self, analytics: DuckDBAnalytics):
        for sym, sector in [("SPY", "ETF"), ("NVDA", "Technology"), ("AMD", "Technology")]:
            row = {
                "symbol": sym, "updated_at": datetime.utcnow(),
                "company_name": sym, "sector": sector, "industry": "x",
                "country": "US", "exchange": "NASDAQ", "market_cap": 1_000_000_000,
                "pe_ratio": 25.0, "forward_pe": 22.0, "pb_ratio": 4.0, "ps_ratio": 2.0,
                "ev_ebitda": 14.0, "eps": 5.0, "book_value_ps": 40.0, "revenue_ps": 80.0,
                "dividend_yield": 0.0, "dividend_rate": 0.0, "payout_ratio": 0.0,
                "beta": 1.2, "week52_high": 200.0, "week52_low": 100.0,
                "avg_volume_30d": 10_000_000, "float_shares": 100_000_000,
                "shares_outstanding": 100_000_000,
                "analyst_target_mean": 180.0, "analyst_target_high": 220.0,
                "analyst_target_low": 140.0, "analyst_rating": "buy", "analyst_count": 30,
            }
            analytics.upsert_equity_stats(row)

        all_rows = analytics.get_all_equity_stats()
        assert len(all_rows) == 3
        symbols = [r["symbol"] for r in all_rows]
        assert "SPY" in symbols and "NVDA" in symbols and "AMD" in symbols


# ---------------------------------------------------------------------------
# EquityStatsService field mapping
# ---------------------------------------------------------------------------

class TestEquityStatsService:
    def _make_info(self, **overrides) -> dict:
        defaults = {
            "longName": "SPDR S&P 500 ETF Trust",
            "sector": "Financial Services",
            "industry": "Asset Management",
            "country": "United States",
            "exchange": "PCX",
            "marketCap": 500_000_000_000,
            "trailingPE": 22.5,
            "forwardPE": 20.0,
            "priceToBook": 4.1,
            "priceToSalesTrailing12Months": 2.5,
            "enterpriseToEbitda": 14.0,
            "trailingEps": 18.2,
            "bookValue": 110.0,
            "revenuePerShare": 250.0,
            "dividendYield": 0.013,
            "dividendRate": 6.5,
            "payoutRatio": 0.35,
            "beta": 1.0,
            "fiftyTwoWeekHigh": 580.0,
            "fiftyTwoWeekLow": 420.0,
            "averageVolume": 80_000_000,
            "floatShares": 900_000_000,
            "sharesOutstanding": 910_000_000,
            "targetMeanPrice": 550.0,
            "targetHighPrice": 600.0,
            "targetLowPrice": 500.0,
            "recommendationKey": "buy",
            "numberOfAnalystOpinions": 12,
        }
        defaults.update(overrides)
        return defaults

    def test_refresh_symbol_success(self, service: EquityStatsService, analytics: DuckDBAnalytics):
        info = self._make_info()
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = info
            ok = service.refresh_symbol("SPY")

        assert ok is True
        row = analytics.get_equity_stats("SPY")
        assert row is not None
        assert row["company_name"] == "SPDR S&P 500 ETF Trust"
        assert row["sector"] == "Financial Services"
        assert row["pe_ratio"] == pytest.approx(22.5)
        assert row["beta"] == pytest.approx(1.0)
        assert row["analyst_rating"] == "buy"
        assert row["analyst_count"] == 12

    def test_refresh_symbol_handles_missing_fields(self, service: EquityStatsService, analytics: DuckDBAnalytics):
        info = {"longName": "Test Corp"}  # Minimal info
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = info
            ok = service.refresh_symbol("TEST")

        assert ok is True
        row = analytics.get_equity_stats("TEST")
        assert row is not None
        assert row["company_name"] == "Test Corp"
        assert row["pe_ratio"] == 0  # Missing fields default to 0
        assert row["sector"] == ""

    def test_refresh_symbol_returns_false_on_exception(self, service: EquityStatsService):
        with patch("yfinance.Ticker", side_effect=Exception("network error")):
            ok = service.refresh_symbol("BROKEN")
        assert ok is False

    def test_refresh_all_returns_per_symbol_status(self, service: EquityStatsService):
        info = self._make_info()
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = info
            results = service.refresh_all(["SPY", "QQQ"])

        assert set(results.keys()) == {"SPY", "QQQ"}
        assert all(results.values())

    def test_nan_field_coerced_to_zero(self, service: EquityStatsService, analytics: DuckDBAnalytics):
        import math
        info = self._make_info(trailingPE=float("nan"), forwardPE=None)
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = info
            service.refresh_symbol("SPY")

        row = analytics.get_equity_stats("SPY")
        assert row["pe_ratio"] == 0
        assert row["forward_pe"] == 0

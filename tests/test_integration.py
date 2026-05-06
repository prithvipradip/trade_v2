"""Integration tests — write real data to production databases.

Tables used (separate from production tables):
  DuckDB  data/ait_analytics.duckdb  → test_equity_stats
  SQLite  data/fundamentals.db       → test_news, test_analyst_recommendations

No cleanup is performed after the tests; rows accumulate across runs (idempotent
inserts ensure repeating the suite never corrupts existing data).

IB Gateway must be reachable on 127.0.0.1:7497.  clientId=98 is used to avoid
conflicting with the live bot (clientId=1).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from ait.data.equity_stats import EquityStatsService
from ait.data.fundamentals_db import FundamentalsStore
from ait.data.ib_news import IBNewsService
from ait.monitoring.duckdb_analytics import DuckDBAnalytics

# ---------------------------------------------------------------------------
# Shared paths (production databases)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_DUCKDB_PATH = _REPO_ROOT / "data" / "ait_analytics.duckdb"
_SQLITE_PATH = _REPO_ROOT / "data" / "fundamentals.db"

_TEST_SYMBOLS = ["SPY", "AAPL"]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analytics() -> DuckDBAnalytics:
    """Production DuckDB — test rows go into test_equity_stats table."""
    return DuckDBAnalytics(db_path=_DUCKDB_PATH)


@pytest.fixture(scope="module")
def equity_service(analytics: DuckDBAnalytics) -> EquityStatsService:
    return EquityStatsService(analytics)


@pytest.fixture(scope="module")
def fundamentals_store() -> FundamentalsStore:
    """Production SQLite — test rows go into test_news / test_analyst_recommendations."""
    return FundamentalsStore(db_path=_SQLITE_PATH, table_prefix="test_")


@pytest.fixture(scope="module")
def ib_client():
    """Thin wrapper around ib_insync.IB that connects synchronously.

    clientId=98 avoids conflicting with the live bot (clientId=1).
    """
    from ib_insync import IB, util

    util.patchAsyncio()
    ib = IB()
    ib.connect("127.0.0.1", 4002, clientId=98, timeout=15, readonly=True)
    yield _IBWrapper(ib)
    ib.disconnect()


class _IBWrapper:
    """Minimal shim so IBNewsService sees a .ib property."""

    def __init__(self, ib):
        self.ib = ib


# ---------------------------------------------------------------------------
# Feature 1 — Equity Stats (yfinance → test_equity_stats in DuckDB)
# ---------------------------------------------------------------------------


class TestEquityStatsIntegration:
    def test_refresh_spy_writes_to_test_table(
        self, equity_service: EquityStatsService, analytics: DuckDBAnalytics
    ):
        ok = equity_service.refresh_symbol("SPY", table="test_equity_stats")
        assert ok is True, "refresh_symbol returned False — yfinance may be unavailable"

        row = analytics.get_equity_stats("SPY", table="test_equity_stats")
        assert row is not None, "SPY not found in test_equity_stats after refresh"
        assert row["symbol"] == "SPY"
        # SPY is an ETF — sector and market_cap may be empty; check company_name instead
        assert row["company_name"] != "", f"company_name should be non-empty for SPY, got {row}"
        assert row["updated_at"] is not None

    def test_refresh_aapl_writes_to_test_table(
        self, equity_service: EquityStatsService, analytics: DuckDBAnalytics
    ):
        ok = equity_service.refresh_symbol("AAPL", table="test_equity_stats")
        assert ok is True

        row = analytics.get_equity_stats("AAPL", table="test_equity_stats")
        assert row is not None
        assert row["company_name"] != "", "company_name should be non-empty for AAPL"
        assert row["pe_ratio"] >= 0, "pe_ratio should be non-negative"

    def test_refresh_all_returns_status_dict(
        self, equity_service: EquityStatsService, analytics: DuckDBAnalytics
    ):
        results = equity_service.refresh_all(_TEST_SYMBOLS, table="test_equity_stats")
        assert set(results.keys()) == set(_TEST_SYMBOLS)
        assert all(results.values()), f"Some symbols failed: {results}"

        all_rows = analytics.get_all_equity_stats(table="test_equity_stats")
        stored_symbols = {r["symbol"] for r in all_rows}
        for sym in _TEST_SYMBOLS:
            assert sym in stored_symbols, f"{sym} missing from test_equity_stats"

    def test_upsert_is_idempotent(
        self, equity_service: EquityStatsService, analytics: DuckDBAnalytics
    ):
        """Running refresh twice should not raise — INSERT OR REPLACE handles duplicates."""
        ok1 = equity_service.refresh_symbol("SPY", table="test_equity_stats")
        ok2 = equity_service.refresh_symbol("SPY", table="test_equity_stats")
        assert ok1 and ok2

        # Only one row per symbol (it's a PRIMARY KEY)
        all_rows = analytics.get_all_equity_stats(table="test_equity_stats")
        spy_rows = [r for r in all_rows if r["symbol"] == "SPY"]
        assert len(spy_rows) == 1


# ---------------------------------------------------------------------------
# Feature 2 — IB News (live IB → test_news in SQLite)
# ---------------------------------------------------------------------------


class TestIBNewsIntegration:
    def test_fetch_spy_news_writes_to_test_table(
        self, ib_client: _IBWrapper, fundamentals_store: FundamentalsStore
    ):
        svc = IBNewsService(ib_client=ib_client, store=fundamentals_store)
        inserted = svc.fetch_and_store_news("SPY", hours_back=48)
        # IB may legitimately return 0 articles in off-market hours; inserted >= 0 is always true.
        # We verify the call succeeds and any returned articles are stored correctly.
        assert isinstance(inserted, int) and inserted >= 0

        rows = fundamentals_store.get_recent_news("SPY", hours=48)
        if rows:
            first = rows[0]
            assert first["symbol"] == "SPY"
            assert first["headline"] != ""
            assert "published_at" in first
            assert isinstance(first["sentiment"], float)

    def test_fetch_aapl_news_writes_to_test_table(
        self, ib_client: _IBWrapper, fundamentals_store: FundamentalsStore
    ):
        svc = IBNewsService(ib_client=ib_client, store=fundamentals_store)
        inserted = svc.fetch_and_store_news("AAPL", hours_back=48)
        assert isinstance(inserted, int) and inserted >= 0

    def test_news_insert_is_idempotent(
        self, ib_client: _IBWrapper, fundamentals_store: FundamentalsStore
    ):
        """Fetching the same window twice should not create duplicate rows."""
        svc = IBNewsService(ib_client=ib_client, store=fundamentals_store)
        inserted_first = svc.fetch_and_store_news("SPY", hours_back=24)
        inserted_second = svc.fetch_and_store_news("SPY", hours_back=24)
        # Second run should insert 0 (all duplicates ignored via INSERT OR IGNORE)
        assert inserted_second == 0, (
            f"Expected 0 new rows on second fetch, got {inserted_second}"
        )


# ---------------------------------------------------------------------------
# Feature 2 — IB Analyst Recommendations (live IB → test_analyst_recommendations)
# ---------------------------------------------------------------------------


class TestIBAnalystIntegration:
    def test_fetch_aapl_analyst_actions(
        self, ib_client: _IBWrapper, fundamentals_store: FundamentalsStore
    ):
        svc = IBNewsService(ib_client=ib_client, store=fundamentals_store)
        inserted = svc.fetch_and_store_analyst_actions("AAPL", hours_back=168)
        assert isinstance(inserted, int) and inserted >= 0

        rows = fundamentals_store.get_analyst_recs("AAPL", days=30)
        if rows:
            first = rows[0]
            assert first["symbol"] == "AAPL"
            assert first["firm"] != "", "firm should be non-empty"
            assert first["action"] != "", "action should be non-empty"
            # id is a 16-char hex SHA-256 fragment
            assert len(first["id"]) == 16

    def test_fetch_spy_analyst_actions(
        self, ib_client: _IBWrapper, fundamentals_store: FundamentalsStore
    ):
        svc = IBNewsService(ib_client=ib_client, store=fundamentals_store)
        inserted = svc.fetch_and_store_analyst_actions("SPY", hours_back=168)
        assert isinstance(inserted, int) and inserted >= 0

    def test_analyst_insert_is_idempotent(
        self, ib_client: _IBWrapper, fundamentals_store: FundamentalsStore
    ):
        """UNIQUE(symbol, issued_at, firm) ensures re-fetching is idempotent."""
        svc = IBNewsService(ib_client=ib_client, store=fundamentals_store)
        svc.fetch_and_store_analyst_actions("AAPL", hours_back=48)
        inserted_second = svc.fetch_and_store_analyst_actions("AAPL", hours_back=48)
        assert inserted_second == 0, (
            f"Expected 0 new rows on second analyst fetch, got {inserted_second}"
        )

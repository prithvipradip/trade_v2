"""Integration tests — write real data to production databases.

Tables used (separate from production tables):
  DuckDB  data/ait_analytics.duckdb  → test_equity_stats
  SQLite  data/fundamentals.db       → test_news, test_analyst_recommendations

No cleanup is performed after the tests; rows accumulate across runs (idempotent
inserts ensure repeating the suite never corrupts existing data).

Mirrors production as closely as possible:
- News fetched via IB Gateway (clientId=98, port 4002) with FinBERT sentiment
  scoring — same scorer used by run_orchestrator.py --fetch-news.
- Sentiment scores on rows inserted before this fix (when lambda _: 0.0 was
  the default) will remain 0.0 due to INSERT OR IGNORE; newly inserted rows
  will carry real FinBERT scores in [-1.0, +1.0].
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ait.data.equity_stats import EquityStatsService
from ait.data.fundamentals_db import FundamentalsStore
from ait.data.ib_news import IBNewsService
from ait.monitoring.duckdb_analytics import DuckDBAnalytics
from ait.sentiment.finbert import FinBERTAnalyzer

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
    """Synchronous IB connection on clientId=98 (avoids conflict with live bot).

    Tries port 4001 (live IB Gateway) first, then 4002 (paper trading).
    Skips the entire module if no Gateway is reachable.
    """
    from ib_insync import IB, util

    util.patchAsyncio()
    ib = IB()
    connected_port = None
    for port in (4001, 4002):
        try:
            ib.connect("127.0.0.1", port, clientId=98, timeout=10, readonly=True)
            connected_port = port
            break
        except Exception:
            continue

    if connected_port is None:
        pytest.skip("Could not connect to IB Gateway on port 4001 or 4002 — ensure Gateway is running.")

    accounts = ib.managedAccounts()
    print(f"\nConnected to IB Gateway on port {connected_port}, account: {accounts}")
    yield _IBWrapper(ib)
    ib.disconnect()


@pytest.fixture(scope="module")
def news_service(ib_client: "_IBWrapper", fundamentals_store: FundamentalsStore) -> IBNewsService:
    """IBNewsService wired with FinBERT — identical to the production setup in
    run_orchestrator.py --fetch-news.  FinBERT loads once for the module."""
    finbert = FinBERTAnalyzer()
    return IBNewsService(
        ib_client=ib_client,
        store=fundamentals_store,
        sentiment_fn=lambda headline: finbert.analyze(headline) or 0.0,
    )


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


# Hint passed to IB's reqHistoricalNews as the start-of-window.
_FETCH_HOURS = 168   # 7 days — a reasonable production-like window
# Read-back window used to verify rows were stored.  Must be wide enough to
# find whatever IB returned (paper accounts can have news 10+ days old).
_READ_HOURS = 720    # 30 days


def _require_news(fetched: int, hours_back: int) -> None:
    """Skip with a clear message if IB returned no articles at all.

    Distinguishes between "IB has no news" (fetched==0, dead feed) and
    "all articles already cached" (fetched>0, inserted==0, normal after first run).
    """
    if fetched == 0:
        pytest.skip(
            f"IB returned 0 articles for the last {hours_back}h — "
            "the account's news feed may be delayed (common on paper accounts) "
            "or reqHistoricalNews is unavailable. Try a live account."
        )


class TestIBNewsIntegration:
    def test_fetch_spy_news_writes_to_test_table(
        self, news_service: IBNewsService, fundamentals_store: FundamentalsStore
    ):
        fetched, _inserted = news_service.fetch_and_store_news("SPY", hours_back=_FETCH_HOURS)
        _require_news(fetched, _FETCH_HOURS)

        rows = fundamentals_store.get_recent_news("SPY", hours=_READ_HOURS)
        assert rows, "IB returned articles but get_recent_news returned empty"
        first = rows[0]
        assert first["symbol"] == "SPY"
        assert first["headline"] != ""
        assert "published_at" in first
        assert isinstance(first["sentiment"], float)
        assert -1.0 <= first["sentiment"] <= 1.0, (
            f"sentiment out of range: {first['sentiment']}"
        )

    def test_fetch_aapl_news_writes_to_test_table(
        self, news_service: IBNewsService, fundamentals_store: FundamentalsStore
    ):
        fetched, _inserted = news_service.fetch_and_store_news("AAPL", hours_back=_FETCH_HOURS)
        _require_news(fetched, _FETCH_HOURS)

        rows = fundamentals_store.get_recent_news("AAPL", hours=_READ_HOURS)
        assert rows, "IB returned articles but get_recent_news returned empty"
        assert all(isinstance(r["sentiment"], float) for r in rows)
        assert all(-1.0 <= r["sentiment"] <= 1.0 for r in rows)

    def test_newly_inserted_news_has_finbert_score(
        self, news_service: IBNewsService, fundamentals_store: FundamentalsStore
    ):
        """Freshly inserted articles must carry real FinBERT scores (not all 0.0).

        Wipes stale test_news rows for SPY so this run inserts fresh articles
        scored by the live FinBERT pipeline — rows from earlier broken runs had
        0.0 scores and would make the assertion vacuously fail.
        """
        with fundamentals_store._connect() as conn:
            conn.execute(f"DELETE FROM {fundamentals_store._news_table} WHERE symbol = 'SPY'")

        fetched, inserted = news_service.fetch_and_store_news("SPY", hours_back=_FETCH_HOURS)
        _require_news(fetched, _FETCH_HOURS)
        assert inserted > 0, f"Expected fresh inserts after clearing table, got {inserted}"

        rows = fundamentals_store.get_recent_news("SPY", hours=_READ_HOURS)
        assert rows, "IB returned articles but get_recent_news returned empty after re-insert"
        scores = [r["sentiment"] for r in rows]
        assert any(s != 0.0 for s in scores), (
            "All sentiment scores are 0.0 — FinBERT may not be running correctly."
        )

    def test_news_insert_is_idempotent(
        self, news_service: IBNewsService, fundamentals_store: FundamentalsStore
    ):
        """Fetching the same window twice should not create duplicate rows."""
        fetched_first, _inserted_first = news_service.fetch_and_store_news("SPY", hours_back=_FETCH_HOURS)
        _require_news(fetched_first, _FETCH_HOURS)
        _fetched_second, inserted_second = news_service.fetch_and_store_news("SPY", hours_back=_FETCH_HOURS)
        assert inserted_second == 0, (
            f"Expected 0 new rows on second fetch, got {inserted_second}"
        )


# ---------------------------------------------------------------------------
# Feature 2 — IB Analyst Recommendations (live IB → test_analyst_recommendations)
# ---------------------------------------------------------------------------


class TestIBAnalystIntegration:
    def test_fetch_aapl_analyst_actions(
        self, news_service: IBNewsService, fundamentals_store: FundamentalsStore
    ):
        fetched, _inserted = news_service.fetch_and_store_analyst_actions("AAPL", hours_back=168)
        _require_news(fetched, 168)

        rows = fundamentals_store.get_analyst_recs("AAPL", days=30)
        assert rows, "IB returned analyst records but get_analyst_recs returned empty"
        first = rows[0]
        assert first["symbol"] == "AAPL"
        assert first["firm"] != "", "firm should be non-empty"
        assert first["action"] != "", "action should be non-empty"
        assert len(first["id"]) == 16

    def test_fetch_spy_analyst_actions(
        self, news_service: IBNewsService, fundamentals_store: FundamentalsStore
    ):
        fetched, _inserted = news_service.fetch_and_store_analyst_actions("SPY", hours_back=168)
        _require_news(fetched, 168)

    def test_analyst_insert_is_idempotent(
        self, news_service: IBNewsService, fundamentals_store: FundamentalsStore
    ):
        """UNIQUE(symbol, issued_at, firm) ensures re-fetching is idempotent."""
        fetched_first, _inserted_first = news_service.fetch_and_store_analyst_actions("AAPL", hours_back=48)
        _require_news(fetched_first, 48)
        _fetched_second, inserted_second = news_service.fetch_and_store_analyst_actions("AAPL", hours_back=48)
        assert inserted_second == 0, (
            f"Expected 0 new rows on second analyst fetch, got {inserted_second}"
        )

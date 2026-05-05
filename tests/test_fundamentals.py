"""Tests for Feature 2: FundamentalsStore (SQLite) and IBNewsService parsing."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ait.data.fundamentals_db import FundamentalsStore
from ait.data.ib_news import IBNewsService


@pytest.fixture
def store(tmp_path: Path) -> FundamentalsStore:
    return FundamentalsStore(db_path=tmp_path / "fundamentals.db")


# ---------------------------------------------------------------------------
# FundamentalsStore — news table
# ---------------------------------------------------------------------------

class TestFundamentalsStoreNews:
    def _make_article(self, article_id: str, symbol: str = "SPY", sentiment: float = 0.3) -> dict:
        return {
            "article_id": article_id,
            "symbol": symbol,
            "provider": "DJ-N",
            "headline": "Markets rise on strong earnings",
            "url": "",
            "published_at": datetime.utcnow().isoformat(),
            "sentiment": sentiment,
        }

    def test_insert_and_retrieve(self, store: FundamentalsStore):
        a = self._make_article("art-001")
        n = store.insert_news([a])
        assert n == 1

        rows = store.get_recent_news("SPY", hours=1)
        assert len(rows) == 1
        assert rows[0]["article_id"] == "art-001"
        assert rows[0]["headline"] == "Markets rise on strong earnings"
        assert rows[0]["sentiment"] == pytest.approx(0.3)

    def test_insert_or_ignore_idempotent(self, store: FundamentalsStore):
        a = self._make_article("art-dup")
        n1 = store.insert_news([a])
        n2 = store.insert_news([a])
        assert n1 == 1
        assert n2 == 0  # Duplicate ignored

        rows = store.get_recent_news("SPY", hours=1)
        assert len(rows) == 1  # Still only one row

    def test_get_recent_news_filters_by_hours(self, store: FundamentalsStore):
        old = self._make_article("art-old")
        old["published_at"] = (datetime.utcnow() - timedelta(hours=48)).isoformat()
        recent = self._make_article("art-new")

        store.insert_news([old, recent])
        rows = store.get_recent_news("SPY", hours=24)
        ids = [r["article_id"] for r in rows]
        assert "art-new" in ids
        assert "art-old" not in ids

    def test_get_recent_news_filters_by_symbol(self, store: FundamentalsStore):
        spy = self._make_article("art-spy", symbol="SPY")
        qqq = self._make_article("art-qqq", symbol="QQQ")
        store.insert_news([spy, qqq])

        rows = store.get_recent_news("SPY", hours=1)
        assert all(r["symbol"] == "SPY" for r in rows)
        assert len(rows) == 1

    def test_insert_empty_list_returns_zero(self, store: FundamentalsStore):
        assert store.insert_news([]) == 0


# ---------------------------------------------------------------------------
# FundamentalsStore — analyst recommendations table
# ---------------------------------------------------------------------------

class TestFundamentalsStoreAnalyst:
    def _make_rec(self, rec_id: str, symbol: str = "AAPL", firm: str = "Barclays") -> dict:
        # Use a recent date so it falls within the default 30-day query window
        recent_date = (datetime.utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
        return {
            "id": rec_id,
            "symbol": symbol,
            "issued_at": recent_date,
            "published_at": datetime.utcnow().isoformat(),
            "firm": firm,
            "action": "reiterated",
            "rating": "Overweight",
            "price_target": 230.0,
            "prior_target": 220.0,
            "raw_text": "Barclays reiterated AAPL with Overweight and $230 target.",
        }

    def test_insert_and_retrieve(self, store: FundamentalsStore):
        r = self._make_rec("rec-001")
        n = store.insert_analyst_rec([r])
        assert n == 1

        rows = store.get_analyst_recs("AAPL", days=30)
        assert len(rows) == 1
        assert rows[0]["firm"] == "Barclays"
        assert rows[0]["rating"] == "Overweight"
        assert rows[0]["price_target"] == pytest.approx(230.0)

    def test_insert_or_ignore_unique_constraint(self, store: FundamentalsStore):
        r1 = self._make_rec("rec-a")
        r2 = self._make_rec("rec-b")  # Different id, same (symbol, issued_at, firm)
        n1 = store.insert_analyst_rec([r1])
        n2 = store.insert_analyst_rec([r2])  # UNIQUE(symbol, issued_at, firm) blocks this
        assert n1 == 1
        assert n2 == 0

    def test_insert_different_firms_both_stored(self, store: FundamentalsStore):
        r1 = self._make_rec("rec-jpm", firm="JP Morgan")
        r2 = self._make_rec("rec-bar", firm="Barclays")
        r2["id"] = "rec-bar2"  # Unique id required
        n = store.insert_analyst_rec([r1, r2])
        assert n == 2
        rows = store.get_analyst_recs("AAPL", days=30)
        firms = {r["firm"] for r in rows}
        assert "JP Morgan" in firms and "Barclays" in firms

    def test_get_analyst_recs_filters_by_days(self, store: FundamentalsStore):
        old = self._make_rec("rec-old", firm="OldFirm")
        old["issued_at"] = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")
        recent = self._make_rec("rec-new", firm="NewFirm")
        recent["id"] = "rec-new2"
        store.insert_analyst_rec([old, recent])

        rows = store.get_analyst_recs("AAPL", days=30)
        firms = [r["firm"] for r in rows]
        assert "NewFirm" in firms
        assert "OldFirm" not in firms


# ---------------------------------------------------------------------------
# IBNewsService — headline prefix stripping
# ---------------------------------------------------------------------------

class TestIBNewsServiceStripPrefix:
    def test_strips_ib_metadata_prefix(self):
        raw = "{A:800015:L:en:K:0.63:C:0.629}!JP Morgan reiterated Apple with Overweight"
        cleaned = IBNewsService._strip_prefix(raw)
        assert cleaned == "JP Morgan reiterated Apple with Overweight"

    def test_passthrough_when_no_prefix(self):
        raw = "Apple reports record quarterly earnings"
        assert IBNewsService._strip_prefix(raw) == raw

    def test_handles_empty_string(self):
        assert IBNewsService._strip_prefix("") == ""

    def test_handles_brace_without_exclamation(self):
        raw = "{incomplete header without bang"
        assert IBNewsService._strip_prefix(raw) == raw


# ---------------------------------------------------------------------------
# IBNewsService — analyst text parsing
# ---------------------------------------------------------------------------

class TestIBNewsServiceParseAnalyst:
    def _parse(self, text: str, symbol: str = "AAPL") -> dict | None:
        return IBNewsService._parse_analyst_text(symbol, datetime(2026, 1, 30, 10, 0), text)

    def test_parses_full_article(self):
        text = (
            "Barclays reiterated Apple (AAPL) coverage with Underweight rating "
            "and price target $239\n"
            "Previous price target: $230\n"
            "Issuance Date: 2026-01-30"
        )
        result = self._parse(text)
        assert result is not None
        assert result["firm"] == "Barclays"
        assert "reiterate" in result["action"]
        assert result["rating"] == "Underweight"
        assert result["price_target"] == pytest.approx(239.0)
        assert result["prior_target"] == pytest.approx(230.0)
        assert result["issued_at"] == "2026-01-30"

    def test_parses_upgrade(self):
        text = (
            "JP Morgan upgraded Tesla (TSLA) to Overweight with price target $350\n"
            "Issuance Date: 2026-01-15"
        )
        result = self._parse(text, "TSLA")
        assert result is not None
        assert "upgraded" in result["action"]
        assert result["rating"] == "Overweight"
        assert result["price_target"] == pytest.approx(350.0)

    def test_returns_none_on_empty_text(self):
        assert self._parse("") is None

    def test_returns_none_when_firm_not_found(self):
        assert self._parse("No firm name here, just noise.") is None

    def test_stable_id_generation(self):
        text = "Barclays reiterated AAPL with Overweight\nIssuance Date: 2026-01-30"
        r1 = self._parse(text)
        r2 = self._parse(text)
        assert r1 is not None and r2 is not None
        assert r1["id"] == r2["id"]  # Deterministic hash

    def test_id_differs_by_firm(self):
        t1 = "Barclays reiterated AAPL with Overweight\nIssuance Date: 2026-01-30"
        t2 = "Goldman reiterated AAPL with Overweight\nIssuance Date: 2026-01-30"
        r1 = self._parse(t1)
        r2 = self._parse(t2)
        assert r1 is not None and r2 is not None
        assert r1["id"] != r2["id"]

    def test_fallback_to_feed_time_when_no_issuance_date(self):
        text = "JP Morgan reiterated NVDA with Buy and price target $900"
        result = self._parse(text, "NVDA")
        assert result is not None
        # Falls back to feed_time.date()
        assert result["issued_at"] == "2026-01-30"

    def test_raw_text_capped_at_4000_chars(self):
        text_base = "Barclays reiterated AAPL with Overweight\nIssuance Date: 2026-01-30\n"
        long_text = text_base + "x" * 5000
        result = self._parse(long_text)
        assert result is not None
        assert len(result["raw_text"]) <= 4000

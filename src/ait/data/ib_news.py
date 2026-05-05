"""IB news and analyst action fetcher.

Calls ib_insync's reqHistoricalNews + reqNewsArticle, strips IB headline
encoding, scores sentiment, and persists to FundamentalsStore.

Provider routing (per plan):
- General news:    BRFG + DJ-N + DJ-RTG + DJ-RTPRO + DJNL
- Analyst actions: BRFUPDN (Briefing.com Analyst Actions only)
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from typing import Callable

from ait.broker.ibkr_client import IBKRClient
from ait.data.fundamentals_db import FundamentalsStore
from ait.utils.logging import get_logger

log = get_logger("data.ib_news")

NEWS_PROVIDERS = "BRFG+DJ-N+DJ-RTG+DJ-RTPRO+DJNL"
ANALYST_PROVIDER = "BRFUPDN"

# Regex patterns for parsing Briefing.com analyst action text
_ACTION_VERBS = r"(?P<action>reiterated?|upgraded?|downgraded?|initiated?|removed?|resumed?|maintained?)"
_FIRM_PATTERN = re.compile(
    r"^(?P<firm>[A-Za-z][A-Za-z &.']+?)\s+" + _ACTION_VERBS,
    re.IGNORECASE | re.MULTILINE,
)
_RATING_PATTERN = re.compile(
    r"\b(?:with|to|at|as)\s+(?P<rating>Overweight|Underweight|Buy|Sell|Neutral|"
    r"Outperform|Underperform|Strong Buy|Strong Sell|Market Perform|Hold|Equal Weight)\b",
    re.IGNORECASE,
)
_TARGET_PATTERN = re.compile(r"price target\s+\$(?P<target>[\d,.]+)", re.IGNORECASE)
_PRIOR_TARGET_PATTERN = re.compile(r"[Pp]revious price target[:\s]+\$(?P<prior>[\d,.]+)")
_ISSUANCE_DATE_PATTERN = re.compile(r"[Ii]ssuance\s+[Dd]ate[:\s]+(?P<date>\d{4}-\d{2}-\d{2})")


class IBNewsService:
    """Fetches IB news and analyst actions and persists them to FundamentalsStore."""

    def __init__(
        self,
        ib_client: IBKRClient,
        store: FundamentalsStore,
        sentiment_fn: Callable[[str], float] | None = None,
    ) -> None:
        self._client = ib_client
        self._store = store
        self._sentiment_fn = sentiment_fn or (lambda _: 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_and_store_news(self, symbol: str, hours_back: int = 24) -> int:
        """Fetch general news for *symbol* and persist via INSERT OR IGNORE.

        Returns the count of newly inserted articles.
        """
        con_id = self._resolve_con_id(symbol)
        if con_id is None:
            return 0

        end = datetime.now()
        start = end - timedelta(hours=hours_back)

        try:
            raw = self._client.ib.reqHistoricalNews(
                con_id,
                NEWS_PROVIDERS,
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
                totalResults=50,
            )
        except Exception as e:
            log.error("ib_news_fetch_failed", symbol=symbol, error=str(e))
            return 0

        rows = []
        for a in raw or []:
            headline = self._strip_prefix(a.headline)
            rows.append({
                "article_id": a.articleId,
                "symbol":     symbol,
                "provider":   a.providerCode,
                "headline":   headline,
                "url":        "",
                "published_at": a.time.isoformat() if hasattr(a.time, "isoformat") else str(a.time),
                "sentiment":  self._sentiment_fn(headline),
            })

        inserted = self._store.insert_news(rows)
        log.info("news_fetched_and_stored", symbol=symbol, fetched=len(rows), inserted=inserted)
        return inserted

    def fetch_and_store_analyst_actions(self, symbol: str, hours_back: int = 168) -> int:
        """Fetch BRFUPDN analyst actions, parse article text, and persist.

        Returns the count of newly inserted records.
        """
        con_id = self._resolve_con_id(symbol)
        if con_id is None:
            return 0

        end = datetime.now()
        start = end - timedelta(hours=hours_back)

        try:
            raw = self._client.ib.reqHistoricalNews(
                con_id,
                ANALYST_PROVIDER,
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
                totalResults=50,
            )
        except Exception as e:
            log.error("analyst_news_fetch_failed", symbol=symbol, error=str(e))
            return 0

        recs = []
        for a in raw or []:
            try:
                article = self._client.ib.reqNewsArticle(a.providerCode, a.articleId, [])
                text = getattr(article, "articleText", "") or ""
                parsed = self._parse_analyst_text(symbol, a.time, text)
                if parsed:
                    recs.append(parsed)
            except Exception as e:
                log.warning("analyst_article_fetch_failed", article_id=a.articleId, error=str(e))

        inserted = self._store.insert_analyst_rec(recs)
        log.info(
            "analyst_actions_fetched_and_stored",
            symbol=symbol,
            fetched=len(recs),
            inserted=inserted,
        )
        return inserted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_con_id(self, symbol: str) -> int | None:
        """Return IB contract conId for a plain equity symbol."""
        from ib_insync import Stock
        try:
            contract = Stock(symbol, "SMART", "USD")
            qualified = self._client.ib.qualifyContracts(contract)
            if qualified:
                return qualified[0].conId
        except Exception as e:
            log.warning("con_id_resolution_failed", symbol=symbol, error=str(e))
        return None

    @staticmethod
    def _strip_prefix(headline: str) -> str:
        """Remove '{A:...:K:...:C:...}!' prefix that IB injects into headlines."""
        if headline.startswith("{") and "!" in headline:
            return headline.split("!", 1)[1]
        return headline

    @staticmethod
    def _parse_analyst_text(symbol: str, feed_time: datetime, text: str) -> dict | None:
        """Parse structured Briefing.com article body into a flat dict.

        Returns None if the essential fields (firm, action, rating) cannot be parsed.
        """
        if not text:
            return None

        # Firm + action verb
        m_firm = _FIRM_PATTERN.search(text)
        if not m_firm:
            return None
        firm = m_firm.group("firm").strip()
        action = m_firm.group("action").lower()  # keep verb as-is (already past or base tense)

        # Rating
        m_rating = _RATING_PATTERN.search(text)
        rating = m_rating.group("rating") if m_rating else ""

        # Price targets
        m_target = _TARGET_PATTERN.search(text)
        price_target = float(m_target.group("target").replace(",", "")) if m_target else 0.0

        m_prior = _PRIOR_TARGET_PATTERN.search(text)
        prior_target = float(m_prior.group("prior").replace(",", "")) if m_prior else 0.0

        # Issuance date (prefer from article body; fallback to feed time)
        m_date = _ISSUANCE_DATE_PATTERN.search(text)
        issued_at = m_date.group("date") if m_date else (
            feed_time.date().isoformat() if hasattr(feed_time, "date") else str(feed_time)[:10]
        )

        published_at = (
            feed_time.isoformat() if hasattr(feed_time, "isoformat") else str(feed_time)
        )

        # Stable primary key: sha256(symbol + issued_at + firm)[:16]
        key_src = f"{symbol}|{issued_at}|{firm}".encode()
        rec_id = hashlib.sha256(key_src).hexdigest()[:16]

        return {
            "id":           rec_id,
            "symbol":       symbol,
            "issued_at":    issued_at,
            "published_at": published_at,
            "firm":         firm,
            "action":       action,
            "rating":       rating,
            "price_target": price_target,
            "prior_target": prior_target,
            "raw_text":     text[:4000],  # cap to avoid huge SQLite rows
        }

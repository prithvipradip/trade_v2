"""IB news and analyst action fetcher.

Calls ib_insync's reqHistoricalNews + reqNewsArticle, strips IB headline
encoding, scores sentiment, and persists to FundamentalsStore.

Provider routing:
- General news:    BRFG + DJ-N + DJ-RT + DJ-RTG + DJNL  (live account set)
- Analyst actions: BRFUPDN (Briefing.com Analyst Actions only)

At startup the service calls reqNewsProviders() and filters _DESIRED_NEWS_PROVIDERS
down to only those actually subscribed on the connected account, avoiding
Error 321 ("Not subscribed for provider X").
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timedelta
from typing import Callable

from ait.broker.ibkr_client import IBKRClient
from ait.data.fundamentals_db import FundamentalsStore
from ait.utils.logging import get_logger

log = get_logger("data.ib_news")

# Full desired set — service filters this to subscribed providers at init time.
_DESIRED_NEWS_PROVIDERS = {"BRFG", "DJ-N", "DJ-RT", "DJ-RTG", "DJ-RTPRO", "DJNL"}
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
        self._news_providers = self._build_provider_string()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_and_store_news(self, symbol: str, hours_back: int = 24) -> tuple[int, int]:
        """Fetch general news for *symbol* and persist via INSERT OR IGNORE.

        Returns (fetched, inserted) — fetched is what IB returned, inserted is
        net-new rows written.  Callers should use *fetched* to detect a dead
        news feed and *inserted* to track incremental ingest.
        """
        con_id = self._resolve_con_id(symbol)
        if con_id is None:
            return 0, 0

        end = datetime.now()
        start = end - timedelta(hours=hours_back)

        t_hist = time.monotonic()
        log.info("req_historical_news_start", symbol=symbol, providers=self._news_providers,
                 hours_back=hours_back)
        try:
            raw = self._client.ib.reqHistoricalNews(
                con_id,
                self._news_providers,
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
                totalResults=50,
            )
        except Exception as e:
            log.error("ib_news_fetch_failed", symbol=symbol, error=str(e),
                      elapsed_s=round(time.monotonic() - t_hist, 2))
            return 0, 0
        log.info("req_historical_news_done", symbol=symbol, article_count=len(raw or []),
                 elapsed_s=round(time.monotonic() - t_hist, 2))

        rows = []
        for i, a in enumerate(raw or []):
            headline = self._strip_prefix(a.headline)
            a_time = a.time if isinstance(a.time, datetime) else datetime.fromisoformat(str(a.time))
            t_sent = time.monotonic()
            sentiment = self._sentiment_fn(headline)
            sent_elapsed = round(time.monotonic() - t_sent, 3)
            if sent_elapsed > 1.0:
                log.warning("slow_sentiment_score", article_index=i, article_id=a.articleId,
                            elapsed_s=sent_elapsed)
            rows.append({
                "article_id": a.articleId,
                "symbol":     symbol,
                "provider":   a.providerCode,
                "headline":   headline,
                "url":        "",
                "published_at": a_time.isoformat(),
                "sentiment":  sentiment,
            })
        log.info("sentiment_scoring_done", symbol=symbol, article_count=len(rows))

        t_insert = time.monotonic()
        inserted = self._store.insert_news(rows)
        log.info("news_fetched_and_stored", symbol=symbol, fetched=len(rows), inserted=inserted,
                 insert_elapsed_s=round(time.monotonic() - t_insert, 3))
        return len(rows), inserted

    def fetch_and_store_analyst_actions(self, symbol: str, hours_back: int = 168) -> tuple[int, int]:
        """Fetch BRFUPDN analyst actions, parse article text, and persist.

        Returns (fetched, inserted) — fetched is parsed records from IB,
        inserted is net-new rows written.
        """
        con_id = self._resolve_con_id(symbol)
        if con_id is None:
            return 0, 0

        end = datetime.now()
        start = end - timedelta(hours=hours_back)

        t_hist = time.monotonic()
        log.info("req_analyst_news_start", symbol=symbol, hours_back=hours_back)
        try:
            raw = self._client.ib.reqHistoricalNews(
                con_id,
                ANALYST_PROVIDER,
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
                totalResults=50,
            )
        except Exception as e:
            log.error("analyst_news_fetch_failed", symbol=symbol, error=str(e),
                      elapsed_s=round(time.monotonic() - t_hist, 2))
            return 0, 0
        log.info("req_analyst_news_done", symbol=symbol, article_count=len(raw or []),
                 elapsed_s=round(time.monotonic() - t_hist, 2))

        recs = []
        for i, a in enumerate(raw or []):
            t_art = time.monotonic()
            log.info("req_news_article_start", symbol=symbol, article_index=i,
                     article_id=a.articleId, provider=a.providerCode)
            try:
                article = self._client.ib.reqNewsArticle(a.providerCode, a.articleId, [])
                text = getattr(article, "articleText", "") or ""
                log.info("req_news_article_done", symbol=symbol, article_index=i,
                         article_id=a.articleId, text_len=len(text),
                         elapsed_s=round(time.monotonic() - t_art, 2))
                parsed = self._parse_analyst_text(symbol, a.time, text)
                if parsed:
                    recs.append(parsed)
            except Exception as e:
                log.warning("analyst_article_fetch_failed", article_id=a.articleId,
                            elapsed_s=round(time.monotonic() - t_art, 2), error=str(e))

        inserted = self._store.insert_analyst_rec(recs)
        log.info(
            "analyst_actions_fetched_and_stored",
            symbol=symbol,
            fetched=len(recs),
            inserted=inserted,
        )
        return len(recs), inserted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_provider_string(self) -> str:
        """Return '+'-joined provider codes filtered to those actually subscribed.

        Calling reqNewsProviders() at init time avoids Error 321 when the
        account lacks a provider in _DESIRED_NEWS_PROVIDERS (e.g. live accounts
        have DJ-RT but not DJ-RTPRO; paper accounts differ again).
        """
        t0 = time.monotonic()
        try:
            available = {p.code for p in self._client.ib.reqNewsProviders()}
        except Exception:
            available = _DESIRED_NEWS_PROVIDERS  # best-effort fallback
        subscribed = _DESIRED_NEWS_PROVIDERS & available
        provider_str = "+".join(sorted(subscribed)) if subscribed else "BRFG"
        log.info("news_providers_active", providers=provider_str, elapsed_s=round(time.monotonic() - t0, 2))
        return provider_str

    def _resolve_con_id(self, symbol: str) -> int | None:
        """Return IB contract conId for a plain equity symbol."""
        from ib_insync import Stock
        t0 = time.monotonic()
        try:
            contract = Stock(symbol, "SMART", "USD")
            qualified = self._client.ib.qualifyContracts(contract)
            if qualified:
                con_id = qualified[0].conId
                log.info("con_id_resolved", symbol=symbol, con_id=con_id, elapsed_s=round(time.monotonic() - t0, 2))
                return con_id
        except Exception as e:
            log.warning("con_id_resolution_failed", symbol=symbol, error=str(e), elapsed_s=round(time.monotonic() - t0, 2))
        return None

    @staticmethod
    def _strip_prefix(headline: str) -> str:
        """Remove '{A:...:K:...:C:...}[!]' prefix that IB injects into headlines."""
        if headline.startswith("{") and "}" in headline:
            after_brace = headline.split("}", 1)[1]
            # Some providers use '}!' separator, others use '}' directly
            return after_brace.lstrip("!")
        return headline

    @staticmethod
    def _parse_analyst_text(symbol: str, feed_time: datetime, text: str) -> dict | None:
        """Parse structured Briefing.com article body into a flat dict.

        Returns None if firm and action cannot be parsed (essential fields).
        Rating is optional and may be empty string when not present in the text.
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

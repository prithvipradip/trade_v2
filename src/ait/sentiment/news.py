"""News sentiment from free sources.

Sources:
- Finnhub free API (60 calls/min): company news headlines
- RSS feeds: Reuters, MarketWatch, CNBC

No mock data — if APIs are unavailable, returns None.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ait.data.cache import TTLCache
from ait.utils.logging import get_logger

log = get_logger("sentiment.news")


@dataclass
class NewsItem:
    """A single news article with sentiment."""

    headline: str
    source: str
    timestamp: datetime
    sentiment: float  # -1.0 to +1.0
    relevance: float  # 0.0 to 1.0


class NewsSentiment:
    """Fetches and analyzes news sentiment for symbols."""

    # RSS feeds for general market news
    RSS_FEEDS = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.reuters.com/reuters/marketsNews",
    ]

    def __init__(self, finnhub_api_key: str = "", cache_ttl: int = 300) -> None:
        self._finnhub_key = finnhub_api_key
        self._cache = TTLCache(default_ttl=cache_ttl)
        self._finnhub_client = None

        if finnhub_api_key:
            try:
                import finnhub

                self._finnhub_client = finnhub.Client(api_key=finnhub_api_key)
                log.info("finnhub_client_initialized")
            except ImportError:
                log.warning("finnhub_package_not_installed")

    async def get_sentiment(self, symbol: str) -> float | None:
        """Get aggregate news sentiment for a symbol.

        Returns: float between -1.0 (very bearish) and +1.0 (very bullish),
                 or None if no news available.
        """
        cached = self._cache.get(f"news_sentiment:{symbol}")
        if cached is not None:
            return cached

        articles = await self._fetch_news(symbol)
        if not articles:
            return None

        # Weight recent articles more
        weighted_sum = 0.0
        weight_total = 0.0
        now = datetime.now()

        for article in articles:
            hours_old = max(1, (now - article.timestamp).total_seconds() / 3600)
            recency_weight = 1.0 / (1.0 + hours_old / 24)  # Decay over 24 hours
            weight = recency_weight * article.relevance

            weighted_sum += article.sentiment * weight
            weight_total += weight

        if weight_total <= 0:
            return None

        score = weighted_sum / weight_total
        self._cache.set(f"news_sentiment:{symbol}", score)

        log.debug(
            "news_sentiment_computed",
            symbol=symbol,
            score=f"{score:.3f}",
            articles=len(articles),
        )
        return score

    async def _fetch_news(self, symbol: str) -> list[NewsItem]:
        """Fetch news from available sources."""
        articles = []

        # Finnhub company news
        if self._finnhub_client:
            finnhub_articles = self._fetch_finnhub(symbol)
            articles.extend(finnhub_articles)

        # RSS feeds for general market sentiment
        rss_articles = self._fetch_rss(symbol)
        articles.extend(rss_articles)

        return articles

    def _fetch_finnhub(self, symbol: str) -> list[NewsItem]:
        """Fetch from Finnhub free API."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            news = self._finnhub_client.company_news(symbol, _from=week_ago, to=today)
            if not news:
                return []

            articles = []
            for item in news[:20]:  # Limit to 20 most recent
                headline = item.get("headline", "")
                if not headline:
                    continue

                # Simple keyword-based sentiment (FinBERT is used separately for deeper analysis)
                sentiment = self._keyword_sentiment(headline)

                articles.append(
                    NewsItem(
                        headline=headline,
                        source=item.get("source", "finnhub"),
                        timestamp=datetime.fromtimestamp(item.get("datetime", 0)),
                        sentiment=sentiment,
                        relevance=0.8,  # Finnhub news is usually relevant
                    )
                )

            return articles

        except Exception as e:
            log.debug("finnhub_fetch_failed", symbol=symbol, error=str(e))
            return []

    def _fetch_rss(self, symbol: str) -> list[NewsItem]:
        """Fetch from RSS feeds and filter for relevant articles."""
        try:
            import feedparser
        except ImportError:
            return []

        articles = []
        for feed_url in self.RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    title = entry.get("title", "")
                    # Only include if symbol or company is mentioned
                    if symbol.upper() not in title.upper():
                        continue

                    published = entry.get("published_parsed")
                    if published:
                        ts = datetime(*published[:6])
                    else:
                        ts = datetime.now()

                    articles.append(
                        NewsItem(
                            headline=title,
                            source="rss",
                            timestamp=ts,
                            sentiment=self._keyword_sentiment(title),
                            relevance=0.5,
                        )
                    )
            except Exception:
                continue

        return articles

    @staticmethod
    def _keyword_sentiment(text: str) -> float:
        """Simple keyword-based sentiment scoring.

        This is a quick heuristic. FinBERT provides better analysis
        but is slower and called separately.
        """
        text_lower = text.lower()

        bullish_words = [
            "surge", "soar", "rally", "gain", "rise", "jump",
            "beat", "exceed", "upgrade", "bullish", "growth",
            "strong", "record", "profit", "breakout", "buy",
        ]
        bearish_words = [
            "crash", "plunge", "drop", "fall", "decline", "loss",
            "miss", "downgrade", "bearish", "weak", "cut",
            "warning", "risk", "fear", "sell", "layoff",
        ]

        bull_count = sum(1 for w in bullish_words if w in text_lower)
        bear_count = sum(1 for w in bearish_words if w in text_lower)

        total = bull_count + bear_count
        if total == 0:
            return 0.0

        return (bull_count - bear_count) / total

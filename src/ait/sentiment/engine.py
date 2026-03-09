"""Sentiment aggregation engine.

Combines multiple sentiment sources into a single score per symbol:
- News sentiment (Finnhub + RSS)
- Fear/Greed indicator (VIX, momentum, breadth)
- FinBERT deep analysis (on headlines)

The final score influences ML confidence and strategy selection.
If sentiment sources fail, trading continues without them (graceful degradation).
"""

from __future__ import annotations

from dataclasses import dataclass

from ait.config.settings import SentimentConfig
from ait.data.cache import TTLCache
from ait.data.market_data import MarketDataService
from ait.sentiment.fear_greed import FearGreedIndicator
from ait.sentiment.finbert import FinBERTAnalyzer
from ait.sentiment.news import NewsSentiment
from ait.utils.logging import get_logger

log = get_logger("sentiment.engine")


@dataclass
class SentimentResult:
    """Aggregated sentiment for a symbol."""

    symbol: str
    composite_score: float  # -1.0 to +1.0
    news_score: float | None
    fear_greed_score: float | None
    finbert_score: float | None
    sources_available: int
    total_sources: int


class SentimentEngine:
    """Aggregates all sentiment sources into a single actionable score."""

    def __init__(
        self,
        config: SentimentConfig,
        market_data: MarketDataService,
        finnhub_api_key: str = "",
    ) -> None:
        self._config = config
        self._cache = TTLCache(default_ttl=config.cache_ttl_seconds)

        # Initialize enabled sources
        self._news: NewsSentiment | None = None
        self._fear_greed: FearGreedIndicator | None = None
        self._finbert: FinBERTAnalyzer | None = None

        if config.sources.news:
            self._news = NewsSentiment(finnhub_api_key=finnhub_api_key, cache_ttl=config.cache_ttl_seconds)

        if config.sources.fear_greed:
            self._fear_greed = FearGreedIndicator(market_data=market_data, cache_ttl=config.cache_ttl_seconds)

        if config.sources.finbert:
            self._finbert = FinBERTAnalyzer(cache_ttl=config.cache_ttl_seconds)

    @property
    def weight(self) -> float:
        """How much sentiment should influence the final signal."""
        return self._config.weight

    async def get_sentiment(self, symbol: str) -> SentimentResult:
        """Get aggregated sentiment for a symbol.

        Always returns a result — if no sources available, returns neutral (0.0).
        """
        cached = self._cache.get(f"sentiment:{symbol}")
        if cached:
            return cached

        scores: list[tuple[float, float]] = []  # (score, weight)
        total_sources = 0
        available_sources = 0

        news_score = None
        fear_greed_score = None
        finbert_score = None

        # 1. News sentiment
        if self._news:
            total_sources += 1
            try:
                news_score = await self._news.get_sentiment(symbol)
                if news_score is not None:
                    scores.append((news_score, 0.35))
                    available_sources += 1
            except Exception as e:
                log.debug("news_sentiment_error", symbol=symbol, error=str(e))

        # 2. Fear/Greed
        if self._fear_greed:
            total_sources += 1
            try:
                reading = await self._fear_greed.get_reading()
                if reading:
                    fear_greed_score = reading.score
                    scores.append((fear_greed_score, 0.40))
                    available_sources += 1
            except Exception as e:
                log.debug("fear_greed_error", error=str(e))

        # 3. FinBERT (use headlines from news source)
        if self._finbert and self._news:
            total_sources += 1
            try:
                # Get headlines and run FinBERT
                articles = await self._news._fetch_news(symbol)
                if articles:
                    headlines = [a.headline for a in articles[:10]]
                    finbert_scores = self._finbert.analyze_batch(headlines)
                    valid_scores = [s for s in finbert_scores if s is not None]
                    if valid_scores:
                        finbert_score = sum(valid_scores) / len(valid_scores)
                        scores.append((finbert_score, 0.25))
                        available_sources += 1
            except Exception as e:
                log.debug("finbert_error", symbol=symbol, error=str(e))

        # Compute weighted composite
        if scores:
            total_weight = sum(w for _, w in scores)
            composite = sum(s * w for s, w in scores) / total_weight
        else:
            composite = 0.0  # Neutral when no data

        result = SentimentResult(
            symbol=symbol,
            composite_score=composite,
            news_score=news_score,
            fear_greed_score=fear_greed_score,
            finbert_score=finbert_score,
            sources_available=available_sources,
            total_sources=total_sources,
        )

        self._cache.set(f"sentiment:{symbol}", result)

        log.info(
            "sentiment_result",
            symbol=symbol,
            composite=f"{composite:.3f}",
            sources=f"{available_sources}/{total_sources}",
        )

        return result

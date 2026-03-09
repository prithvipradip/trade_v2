"""FinBERT sentiment analysis on news headlines.

Uses the ProsusAI/finbert model for financial-domain sentiment.
Runs on CPU — small model (~400MB), loads once and reuses.
"""

from __future__ import annotations

from ait.data.cache import TTLCache
from ait.utils.logging import get_logger

log = get_logger("sentiment.finbert")


class FinBERTAnalyzer:
    """Financial BERT sentiment analysis."""

    def __init__(self, cache_ttl: int = 600) -> None:
        self._pipeline = None
        self._loaded = False
        self._cache = TTLCache(default_ttl=cache_ttl)

    def _load_model(self) -> bool:
        """Lazy-load the FinBERT model (first call only)."""
        if self._loaded:
            return self._pipeline is not None

        try:
            from transformers import pipeline

            log.info("loading_finbert_model")
            self._pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,  # CPU only
                top_k=None,
            )
            self._loaded = True
            log.info("finbert_model_loaded")
            return True

        except Exception as e:
            log.warning("finbert_load_failed", error=str(e))
            self._loaded = True  # Don't retry on failure
            return False

    def analyze(self, text: str) -> float | None:
        """Analyze sentiment of a single text.

        Returns: float between -1.0 (negative) and +1.0 (positive),
                 or None if model unavailable.
        """
        cached = self._cache.get(f"finbert:{text[:100]}")
        if cached is not None:
            return cached

        if not self._load_model():
            return None

        try:
            results = self._pipeline(text[:512])  # FinBERT max 512 tokens

            # results is a list of dicts: [{"label": "positive", "score": 0.9}, ...]
            score = 0.0
            if isinstance(results, list) and results:
                # May be nested list
                items = results[0] if isinstance(results[0], list) else results
                for item in items:
                    label = item["label"].lower()
                    prob = item["score"]
                    if label == "positive":
                        score += prob
                    elif label == "negative":
                        score -= prob
                    # neutral contributes 0

            self._cache.set(f"finbert:{text[:100]}", score)
            return score

        except Exception as e:
            log.debug("finbert_analysis_failed", error=str(e))
            return None

    def analyze_batch(self, texts: list[str]) -> list[float | None]:
        """Analyze sentiment for multiple texts.

        More efficient than calling analyze() in a loop.
        """
        if not texts:
            return []

        if not self._load_model():
            return [None] * len(texts)

        results = []
        for text in texts:
            results.append(self.analyze(text))
        return results

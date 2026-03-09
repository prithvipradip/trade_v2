"""Model trainer — handles periodic retraining of ML models.

Runs weekly (configurable) to retrain the ensemble on fresh data.
Uses walk-forward validation to ensure models don't overfit.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from ait.config.settings import MLConfig
from ait.data.historical import HistoricalDataStore
from ait.data.market_data import MarketDataService
from ait.ml.ensemble import DirectionPredictor
from ait.utils.logging import get_logger

log = get_logger("ml.trainer")


class ModelTrainer:
    """Manages ML model training lifecycle."""

    def __init__(
        self,
        config: MLConfig,
        predictor: DirectionPredictor,
        market_data: MarketDataService,
        historical_store: HistoricalDataStore,
    ) -> None:
        self._config = config
        self._predictor = predictor
        self._market_data = market_data
        self._store = historical_store
        self._last_train_date: date | None = None

    def needs_training(self) -> bool:
        """Check if models need retraining."""
        # First run — always train
        if not self._predictor.is_trained:
            return True

        # Check interval
        if self._last_train_date is None:
            return True

        days_since = (date.today() - self._last_train_date).days
        return days_since >= self._config.retrain_interval_days

    async def train_all_symbols(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Fetch fresh data and train models for all symbols.

        Returns dict of {symbol: {model: accuracy}}.
        """
        results = {}

        for symbol in symbols:
            log.info("training_symbol", symbol=symbol)

            # Fetch and store historical data
            df = await self._market_data.get_historical(
                symbol, days=self._config.lookback_days
            )

            if df is None or df.empty:
                log.warning("no_data_for_training", symbol=symbol)
                continue

            # Save to historical store for future use
            self._store.save(symbol, df)

            # Train ensemble
            accuracies = self._predictor.train(df)
            if accuracies:
                results[symbol] = accuracies

        self._last_train_date = date.today()
        log.info("training_complete", symbols_trained=len(results))
        return results

    async def ensure_models_ready(self, symbols: list[str]) -> bool:
        """Ensure models are loaded or trained. Call on startup."""
        # Try loading existing models
        if self._predictor.load_models():
            if not self.needs_training():
                log.info("using_existing_models")
                return True

        # Need to train
        log.info("training_required")
        results = await self.train_all_symbols(symbols)
        return bool(results)

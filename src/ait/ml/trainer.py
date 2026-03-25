"""Model trainer — handles periodic retraining of ML models.

Runs weekly (configurable) to retrain the ensemble on fresh data.
Uses walk-forward validation to ensure models don't overfit.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from ait.config.settings import MLConfig
from ait.data.historical import HistoricalDataStore
from ait.data.market_data import MarketDataService
from ait.ml.drift import DriftDetector
from ait.ml.ensemble import DirectionPredictor
from ait.utils.logging import get_logger

log = get_logger("ml.trainer")


class ModelTrainer:
    """Manages ML model training lifecycle.

    Integrates with DriftDetector to trigger retraining when the model's
    prediction accuracy degrades below acceptable thresholds.
    """

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
        self._drift_detector = DriftDetector()

    @property
    def drift_detector(self) -> DriftDetector:
        return self._drift_detector

    def needs_training(self) -> bool:
        """Check if models need retraining.

        Triggers on:
        1. First run (no model loaded)
        2. Scheduled interval elapsed
        3. Drift detector signals retraining needed
        """
        # First run — always train
        if not self._predictor.is_trained:
            return True

        # Check interval
        if self._last_train_date is None:
            return True

        days_since = (date.today() - self._last_train_date).days
        if days_since >= self._config.retrain_interval_days:
            return True

        # Drift-triggered retraining
        drift_report = self._drift_detector.check_drift()
        if drift_report.should_retrain:
            log.warning(
                "drift_triggered_retrain",
                accuracy=f"{drift_report.accuracy:.2%}",
                reason=drift_report.reason,
                samples=drift_report.samples,
            )
            return True

        return False

    async def train_all_symbols(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """Fetch fresh data and train models for all symbols.

        Auto-rollback if new model performs significantly worse than previous.
        Returns dict of {symbol: {model: accuracy}}.
        """
        # Save previous scores for comparison
        prev_version = self._predictor.model_version
        prev_scores = dict(self._predictor.cv_scores)

        results = {}
        for symbol in symbols:
            log.info("training_symbol", symbol=symbol)

            df = await self._market_data.get_historical(
                symbol, days=self._config.lookback_days
            )

            if df is None or df.empty:
                log.warning("no_data_for_training", symbol=symbol)
                continue

            self._store.save(symbol, df)
            accuracies = self._predictor.train(df)
            if accuracies:
                results[symbol] = accuracies

        self._last_train_date = date.today()
        self._drift_detector.acknowledge_retrain()

        # Auto-rollback: if new model is significantly worse, revert
        if prev_scores and results:
            new_scores = self._predictor.cv_scores
            if self._should_rollback(prev_scores, new_scores):
                log.warning(
                    "model_performance_degraded",
                    prev_scores=prev_scores,
                    new_scores=new_scores,
                    rolling_back_to=prev_version,
                )
                if prev_version:
                    self._predictor.rollback(prev_version)

        log.info(
            "training_complete",
            symbols_trained=len(results),
            version=self._predictor.model_version,
        )
        return results

    @staticmethod
    def _should_rollback(
        prev_scores: dict[str, float], new_scores: dict[str, float]
    ) -> bool:
        """Check if new model is worse enough to warrant rollback."""
        if not prev_scores or not new_scores:
            return False

        prev_avg = sum(prev_scores.values()) / len(prev_scores)
        new_avg = sum(new_scores.values()) / len(new_scores)

        # Rollback if accuracy dropped by more than 5 percentage points
        return new_avg < prev_avg - 0.05

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

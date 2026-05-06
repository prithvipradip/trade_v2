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
        range_predictor=None,
        vol_mag_predictor=None,
    ) -> None:
        self._config = config
        self._predictor = predictor
        self._market_data = market_data
        self._store = historical_store
        self._last_train_date: date | None = None
        self._drift_detector = DriftDetector()
        self._range_predictor = range_predictor
        self._vol_mag_predictor = vol_mag_predictor

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

    async def train_all_symbols(
        self,
        symbols: list[str],
        optimize_hyperparams: bool = False,
        optimize_n_trials: int = 50,
    ) -> dict[str, dict[str, float]]:
        """Fetch fresh data and train models for all symbols.

        When *optimize_hyperparams* is True, an Optuna search over XGBOOST_SPACE
        and LIGHTGBM_SPACE is run before fitting the final models, and the best
        hyperparams are injected into the predictor's model kwargs.

        Auto-rollback if new model performs significantly worse than previous.
        Returns dict of {symbol: {model: accuracy}}.
        """
        if optimize_hyperparams:
            self._apply_optimized_hyperparams(symbols, n_trials=optimize_n_trials)
        # Save previous scores for comparison
        prev_version = self._predictor.model_version
        prev_scores = dict(self._predictor.cv_scores)

        # Fetch cross-asset data once (shared across all symbols)
        market_context = await self._fetch_market_context()

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
            accuracies = self._predictor.train(
                df, symbol=symbol, market_context=market_context
            )
            if accuracies:
                results[symbol] = accuracies

            # Also train the range predictor (Tier 1 model for iron condors)
            if self._range_predictor is not None:
                try:
                    range_acc = self._range_predictor.train(
                        df, symbol=symbol, market_context=market_context
                    )
                    if range_acc:
                        log.info("range_training_complete", symbol=symbol,
                                 accuracies=range_acc)
                except Exception as e:
                    log.warning("range_training_failed", symbol=symbol, error=str(e))

            # Vol-magnitude predictor (Tier 1 model for long straddles)
            if self._vol_mag_predictor is not None:
                try:
                    vm_acc = self._vol_mag_predictor.train(
                        df, symbol=symbol, market_context=market_context
                    )
                    if vm_acc:
                        log.info("vol_mag_training_complete", symbol=symbol,
                                 accuracies=vm_acc)
                except Exception as e:
                    log.warning("vol_mag_training_failed", symbol=symbol, error=str(e))

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

    async def _fetch_market_context(self) -> dict[str, "pd.DataFrame"]:
        """Fetch VIX and SPY historical data for cross-asset features."""
        import pandas as pd

        context = {}
        lookback = self._config.lookback_days

        def _normalize_index(df: "pd.DataFrame") -> "pd.DataFrame":
            # Strip timezone and normalize to datetime64[ms] so reindex works
            # against symbol DataFrames regardless of what the data source returns.
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_localize(None).astype("datetime64[ms]")
            elif df.index.dtype != "datetime64[ms]":
                df = df.copy()
                df.index = df.index.astype("datetime64[ms]")
            return df

        # Fetch VIX
        try:
            vix_df = await self._market_data.get_historical("^VIX", days=lookback)
            if vix_df is not None and len(vix_df) > 20:
                context["vix"] = _normalize_index(vix_df)
                log.info("market_context_vix", rows=len(vix_df))
        except Exception as e:
            log.warning("market_context_vix_failed", error=str(e))

        # Fetch SPY (for relative strength)
        try:
            spy_df = await self._market_data.get_historical("SPY", days=lookback)
            if spy_df is not None and len(spy_df) > 20:
                context["spy"] = _normalize_index(spy_df)
                log.info("market_context_spy", rows=len(spy_df))
        except Exception as e:
            log.warning("market_context_spy_failed", error=str(e))

        # Fetch macro data (FRED: yield curve + DXY)
        try:
            from ait.data.macro import MacroDataFetcher
            fetcher = MacroDataFetcher()
            macros = await fetcher.fetch_all(lookback_days=lookback * 2)
            if macros:
                context["macros"] = macros
                log.info("market_context_macros", series=list(macros.keys()))
        except Exception as e:
            log.warning("market_context_macros_failed", error=str(e))

        return context

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

    def _apply_optimized_hyperparams(self, symbols: list[str], n_trials: int = 50) -> None:
        """Run a dedicated Optuna search over ML hyperparameter spaces.

        For each trial, XGBoost and LightGBM are constructed with suggested
        hyperparameters and evaluated with walk-forward CV accuracy on a
        representative training DataFrame.  The best params are then injected
        into the predictor's model constructors via ``_xgb_kwargs`` /
        ``_lgbm_kwargs`` so the next ``train()`` call uses them.
        """
        try:
            import optuna

            from ait.optimization.param_spaces import ML_SPACES

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            # Collect one training DataFrame to use as the CV dataset.
            # Prefer the first symbol that has stored historical data.
            train_df: "pd.DataFrame | None" = None
            for symbol in symbols:
                try:
                    df = self._store.load(symbol)
                    if df is not None and len(df) >= self._config.min_training_samples:
                        train_df = df
                        break
                except Exception:
                    continue

            if train_df is None:
                log.warning("ml_hyperparams_skipped", reason="no_training_data_available")
                return

            # Build feature matrix once so each trial only re-fits the models.
            from ait.ml.features import FeatureEngine
            from sklearn.preprocessing import StandardScaler

            engine = FeatureEngine()
            features = engine.compute(train_df)
            if len(features) < self._config.min_training_samples:
                log.warning("ml_hyperparams_skipped", reason="insufficient_features",
                            rows=len(features))
                return

            feature_names = [c for c in engine.get_feature_names() if c in features.columns]
            features["target"] = self._predictor._create_labels(features["Close"])
            features = features.dropna(subset=["target"])

            import numpy as np
            X = features[feature_names].values
            y = features["target"].values.astype(int)
            splits = self._predictor._walk_forward_split(len(X))

            def _objective(trial: "optuna.Trial") -> float:
                scores = []
                for model_name, space in ML_SPACES.items():
                    kwargs: dict = {}
                    for param, spec in space.items():
                        key = f"{model_name}__{param}"
                        kwargs[param] = StrategyOptimizer_suggest_one(trial, key, spec)

                    try:
                        if model_name == "xgboost":
                            from xgboost import XGBClassifier
                            clf = XGBClassifier(
                                **kwargs,
                                objective="multi:softprob",
                                num_class=3,
                                eval_metric="mlogloss",
                                verbosity=0,
                                n_jobs=1,
                                random_state=42,
                            )
                        elif model_name == "lightgbm":
                            from lightgbm import LGBMClassifier
                            clf = LGBMClassifier(
                                **kwargs,
                                objective="multiclass",
                                num_class=3,
                                metric="multi_logloss",
                                verbose=-1,
                                n_jobs=1,
                                random_state=42,
                            )
                        else:
                            continue

                        fold_scores = []
                        for train_idx, val_idx in splits:
                            scaler = StandardScaler()
                            X_tr = scaler.fit_transform(X[train_idx])
                            X_val = scaler.transform(X[val_idx])
                            sw = self._predictor._compute_sample_weights(y[train_idx])
                            clf.fit(X_tr, y[train_idx], sample_weight=sw)
                            fold_scores.append(clf.score(X_val, y[val_idx]))
                        scores.append(float(np.mean(fold_scores)))
                    except Exception as exc:
                        log.debug("ml_trial_fold_failed", error=str(exc))
                        raise optuna.TrialPruned()

                return float(np.mean(scores)) if scores else 0.0

            # Import helper from param_spaces side-car (avoids circular dep)
            from ait.optimization.optimizer import StrategyOptimizer
            StrategyOptimizer_suggest_one = StrategyOptimizer._suggest_one

            study = optuna.create_study(
                direction="maximize",
                study_name="ml_hyperparams",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
            )
            study.optimize(_objective, n_trials=n_trials, n_jobs=1, catch=(Exception,))

            best = study.best_params
            log.info("ml_hyperparams_optimized", best=best, best_cv=study.best_value)

            # Inject best hyperparams so the next train() call uses them.
            xgb_kwargs: dict = {}
            lgbm_kwargs: dict = {}
            for key, val in best.items():
                model, _, param = key.partition("__")
                if model == "xgboost":
                    xgb_kwargs[param] = val
                elif model == "lightgbm":
                    lgbm_kwargs[param] = val

            if xgb_kwargs:
                self._predictor._xgb_kwargs = xgb_kwargs
            if lgbm_kwargs:
                self._predictor._lgbm_kwargs = lgbm_kwargs

        except Exception as e:
            log.warning("ml_hyperparams_optimization_failed", error=str(e))

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

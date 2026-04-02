"""XGBoost + LightGBM ensemble for direction prediction.

Lightweight models that run efficiently on CPU (Apple Silicon).
No TensorFlow or PyTorch needed for inference.

Predicts: BULLISH / BEARISH / NEUTRAL for next trading session.
Output: direction + confidence score (0.0 to 1.0).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ait.config.settings import MLConfig
from ait.ml.features import FeatureEngine
from ait.strategies.base import SignalDirection
from ait.utils.logging import get_logger

log = get_logger("ml.ensemble")

MODEL_DIR = Path("models")


@dataclass
class Prediction:
    """ML model prediction."""

    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    probabilities: dict[str, float]  # {bullish: 0.6, bearish: 0.2, neutral: 0.2}
    features_used: int
    model_version: str = ""


class DirectionPredictor:
    """Ensemble model for predicting market direction."""

    LABELS = {0: SignalDirection.BEARISH, 1: SignalDirection.NEUTRAL, 2: SignalDirection.BULLISH}
    LABEL_MAP = {v: k for k, v in LABELS.items()}

    def __init__(self, config: MLConfig) -> None:
        self._config = config
        self._feature_engine = FeatureEngine()

        # Per-symbol model storage: {symbol: {models, scaler, feature_names, scores}}
        self._symbol_models: dict[str, dict] = {}

        # Legacy single-model (used as fallback and for backtest compatibility)
        self._scaler = StandardScaler()
        self._models: dict[str, object] = {}
        self._trained = False
        self._feature_names: list[str] = []
        self._model_version: str = ""
        self._cv_scores: dict[str, float] = {}

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_trained(self) -> bool:
        return self._trained or bool(self._symbol_models)

    def predict(self, df: pd.DataFrame, symbol: str = "") -> Prediction | None:
        """Predict direction for the latest data point.

        Args:
            df: OHLCV DataFrame with at least 60 rows of history.
            symbol: Symbol name — uses per-symbol model if available.

        Returns:
            Prediction with direction and confidence, or None if model not trained.
        """
        # Select the right model for this symbol
        if symbol and symbol in self._symbol_models:
            sym_data = self._symbol_models[symbol]
            models = sym_data["models"]
            scaler = sym_data["scaler"]
            feature_names = sym_data["feature_names"]
        elif self._trained:
            models = self._models
            scaler = self._scaler
            feature_names = self._feature_names
        else:
            log.warning("prediction_skipped", reason="model not trained", symbol=symbol)
            return None

        features = self._feature_engine.compute(df)
        if features.empty:
            log.warning("prediction_skipped", reason="features_empty",
                        input_rows=len(df))
            return None

        # Use only the last row (current prediction)
        X = features[feature_names].iloc[[-1]]

        try:
            X_scaled = scaler.transform(X.values)
        except Exception as e:
            log.error("feature_scaling_failed", error=str(e), symbol=symbol)
            return None

        # Get predictions from each model
        all_probas = []
        for name, model in models.items():
            weight = self._config.ensemble_weights.get(name, 0.5)
            try:
                proba = model.predict_proba(X_scaled)[0]
                all_probas.append(proba * weight)
            except Exception as e:
                log.warning("model_prediction_failed", model=name, error=str(e))

        if not all_probas:
            return None

        # Weighted average of probabilities
        avg_proba = np.sum(all_probas, axis=0) / sum(self._config.ensemble_weights.values())

        # Get prediction
        pred_class = int(np.argmax(avg_proba))
        confidence = float(avg_proba[pred_class])
        direction = self.LABELS[pred_class]

        probabilities = {
            "bearish": float(avg_proba[0]),
            "neutral": float(avg_proba[1]),
            "bullish": float(avg_proba[2]),
        }

        log.info(
            "prediction",
            direction=direction.value,
            confidence=f"{confidence:.3f}",
            probabilities={k: f"{v:.3f}" for k, v in probabilities.items()},
        )

        return Prediction(
            direction=direction,
            confidence=confidence,
            probabilities=probabilities,
            features_used=len(self._feature_names),
            model_version=self._model_version,
        )

    def train(self, df: pd.DataFrame, symbol: str = "") -> dict[str, float]:
        """Train the ensemble on historical data.

        Args:
            df: OHLCV DataFrame with sufficient history.
            symbol: If provided, stores model per-symbol (prevents cross-contamination).

        Returns:
            Dict of model accuracies.
        """
        features = self._feature_engine.compute(df)
        if len(features) < self._config.min_training_samples:
            log.warning(
                "insufficient_training_data",
                rows=len(features),
                required=self._config.min_training_samples,
            )
            return {}

        # Create labels: next-day return direction
        features["target"] = self._create_labels(features["Close"])
        features = features.dropna(subset=["target"])

        self._feature_names = self._feature_engine.get_feature_names()
        # Only use features that exist in the DataFrame
        self._feature_names = [f for f in self._feature_names if f in features.columns]

        X = features[self._feature_names].values
        y = features["target"].values.astype(int)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = {}

        # Train XGBoost — scaler is fit only on training fold to prevent data leakage
        if "xgboost" in self._config.ensemble_weights:
            acc = self._train_xgboost(X, y, tscv)
            accuracies["xgboost"] = acc

        # Train LightGBM
        if "lightgbm" in self._config.ensemble_weights:
            acc = self._train_lightgbm(X, y, tscv)
            accuracies["lightgbm"] = acc

        # Final scaler fit on all data for inference going forward.
        # Final models are retrained on scaled data so scaler + model are consistent.
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)
        for model in self._models.values():
            model.fit(X_scaled, y)

        self._trained = bool(self._models)
        if self._trained:
            from datetime import datetime
            self._model_version = f"v-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self._cv_scores = accuracies

            # Store per-symbol model (deep copy so next train() doesn't overwrite)
            if symbol:
                import copy
                self._symbol_models[symbol] = {
                    "models": copy.deepcopy(self._models),
                    "scaler": copy.deepcopy(self._scaler),
                    "feature_names": list(self._feature_names),
                    "cv_scores": dict(accuracies),
                    "version": self._model_version,
                }
                log.info("symbol_model_stored", symbol=symbol,
                         accuracies=accuracies, features=len(self._feature_names))

            self._save_models()
            log.info(
                "ensemble_trained",
                version=self._model_version,
                accuracies=accuracies,
                features=len(self._feature_names),
            )

        return accuracies

    def load_models(self, version: str | None = None) -> bool:
        """Load previously trained models from disk.

        Args:
            version: Specific version to load. None = latest (ensemble.pkl).
        """
        if version:
            model_file = MODEL_DIR / f"ensemble_{version}.pkl"
        else:
            model_file = MODEL_DIR / "ensemble.pkl"

        if not model_file.exists():
            return False

        try:
            with open(model_file, "rb") as f:
                data = pickle.load(f)
            self._models = data["models"]
            self._scaler = data["scaler"]
            self._feature_names = data["feature_names"]
            self._model_version = data.get("version", "unknown")
            self._cv_scores = data.get("cv_scores", {})
            self._symbol_models = data.get("symbol_models", {})
            self._trained = True
            log.info("models_loaded", version=self._model_version,
                     models=list(self._models.keys()),
                     per_symbol=list(self._symbol_models.keys()))
            return True
        except Exception as e:
            log.error("model_load_failed", error=str(e))
            return False

    def _save_models(self) -> None:
        """Save trained models to disk with versioning."""
        data = {
            "models": self._models,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "version": self._model_version,
            "cv_scores": self._cv_scores,
            "symbol_models": self._symbol_models,
        }

        # Save as current (ensemble.pkl)
        model_file = MODEL_DIR / "ensemble.pkl"
        try:
            with open(model_file, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            log.error("model_save_failed", error=str(e))
            return

        # Save versioned copy for rollback
        if self._model_version:
            versioned_file = MODEL_DIR / f"ensemble_{self._model_version}.pkl"
            try:
                with open(versioned_file, "wb") as f:
                    pickle.dump(data, f)
                log.info("models_saved", path=str(versioned_file), version=self._model_version)
            except Exception as e:
                log.warning("versioned_save_failed", error=str(e))

            # Prune old versions (keep last 5)
            self._prune_old_versions(keep=5)

    def _prune_old_versions(self, keep: int = 5) -> None:
        """Remove old model versions, keeping only the N most recent."""
        versions = sorted(MODEL_DIR.glob("ensemble_v-*.pkl"))
        if len(versions) > keep:
            for old in versions[:-keep]:
                old.unlink()
                log.debug("pruned_old_model", path=str(old))

    def rollback(self, version: str) -> bool:
        """Roll back to a specific model version. Falls back to latest if target missing."""
        log.info("model_rollback_requested", target_version=version)
        success = self.load_models(version=version)
        if not success:
            log.warning("rollback_target_missing", version=version,
                        hint="Loading latest available model instead")
            success = self.load_models()  # Load latest ensemble.pkl
        return success

    def list_versions(self) -> list[dict]:
        """List available model versions."""
        versions = []
        for f in sorted(MODEL_DIR.glob("ensemble_v-*.pkl")):
            try:
                with open(f, "rb") as fh:
                    data = pickle.load(fh)
                versions.append({
                    "version": data.get("version", f.stem),
                    "cv_scores": data.get("cv_scores", {}),
                    "features": len(data.get("feature_names", [])),
                    "file": str(f),
                })
            except Exception:
                versions.append({"version": f.stem, "file": str(f)})

        return versions

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def cv_scores(self) -> dict[str, float]:
        return self._cv_scores

    def _create_labels(self, close: pd.Series) -> pd.Series:
        """Create 3-class labels from 5-day forward returns.

        Using a 5-day lookahead matches the typical options holding period
        (1-3 weeks) far better than next-day returns. Options need price to
        move meaningfully over several days to overcome theta decay.

        0 = bearish (5-day return < -1.0%)
        1 = neutral (-1.0% to +1.0%)
        2 = bullish (5-day return > +1.0%)

        Narrower neutral band (±1.0% vs old ±1.5%) reduces class imbalance
        and gives the model more directional training signal.
        """
        fwd_return = close.pct_change(5).shift(-5)  # 5-day forward return
        labels = pd.Series(1, index=close.index, dtype=float)  # Default neutral
        labels[fwd_return > 0.01] = 2    # Bullish: >+1.0% in 5 days
        labels[fwd_return < -0.01] = 0   # Bearish: <-1.0% in 5 days
        return labels

    @staticmethod
    def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
        """Compute sample weights to balance class distribution.

        Minority classes (bullish/bearish) get higher weights so the model
        doesn't just learn to always predict neutral.
        """
        from collections import Counter
        counts = Counter(y)
        total = len(y)
        n_classes = len(counts)
        weights = np.ones(len(y))
        for cls, count in counts.items():
            weights[y == cls] = total / (n_classes * count)
        return weights

    def _walk_forward_split(
        self, n_samples: int, n_splits: int = 5, gap: int = 5
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Walk-forward validation splits with purge gap.

        Unlike TimeSeriesSplit, this uses:
        - Expanding training window (all data up to split point)
        - A gap between train and validation to avoid look-ahead bias
        - Fixed-size validation windows

        The gap prevents the model from memorizing recent patterns that
        overlap with the validation period.
        """
        splits = []
        val_size = max(20, n_samples // (n_splits + 2))

        for i in range(n_splits):
            val_end = n_samples - (n_splits - 1 - i) * val_size
            val_start = val_end - val_size
            train_end = val_start - gap  # Purge gap

            if train_end < 50:  # Need minimum training data
                continue

            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))

        # Fallback to regular TimeSeriesSplit if walk-forward yields too few splits
        if len(splits) < 2:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            return list(tscv.split(np.arange(n_samples)))

        return splits

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, tscv) -> float:
        """Train XGBoost model with walk-forward validation."""
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                verbosity=0,
                n_jobs=-1,
            )

            # Walk-forward validation with purge gap
            wf_splits = self._walk_forward_split(len(X))
            scores = []
            for train_idx, val_idx in wf_splits:
                fold_scaler = StandardScaler()
                X_train = fold_scaler.fit_transform(X[train_idx])
                X_val = fold_scaler.transform(X[val_idx])
                sw = self._compute_sample_weights(y[train_idx])
                model.fit(X_train, y[train_idx], sample_weight=sw)
                score = model.score(X_val, y[val_idx])
                scores.append(score)

            # Final model stored (will be refit on fully-scaled data after CV)
            self._models["xgboost"] = model

            avg_acc = float(np.mean(scores))
            log.info("xgboost_trained", cv_accuracy=f"{avg_acc:.3f}", folds=len(scores))
            return avg_acc

        except ImportError:
            log.warning("xgboost_not_installed")
            return 0.0

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, tscv) -> float:
        """Train LightGBM model with walk-forward validation."""
        try:
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multiclass",
                num_class=3,
                metric="multi_logloss",
                verbose=-1,
                n_jobs=-1,
                class_weight="balanced",  # Auto-balance class weights
            )

            wf_splits = self._walk_forward_split(len(X))
            scores = []
            for train_idx, val_idx in wf_splits:
                fold_scaler = StandardScaler()
                X_train = fold_scaler.fit_transform(X[train_idx])
                X_val = fold_scaler.transform(X[val_idx])
                # Pass numpy arrays (no feature names) to avoid sklearn warning
                model.fit(X_train, y[train_idx], feature_name="auto")
                score = model.score(X_val, y[val_idx])
                scores.append(score)

            # Final model stored (will be refit on fully-scaled data after CV)
            self._models["lightgbm"] = model

            avg_acc = float(np.mean(scores))
            log.info("lightgbm_trained", cv_accuracy=f"{avg_acc:.3f}", folds=len(scores))
            return avg_acc

        except ImportError:
            log.warning("lightgbm_not_installed")
            return 0.0

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


class DirectionPredictor:
    """Ensemble model for predicting market direction."""

    LABELS = {0: SignalDirection.BEARISH, 1: SignalDirection.NEUTRAL, 2: SignalDirection.BULLISH}
    LABEL_MAP = {v: k for k, v in LABELS.items()}

    def __init__(self, config: MLConfig) -> None:
        self._config = config
        self._feature_engine = FeatureEngine()
        self._scaler = StandardScaler()
        self._models: dict[str, object] = {}
        self._trained = False
        self._feature_names: list[str] = []

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, df: pd.DataFrame) -> Prediction | None:
        """Predict direction for the latest data point.

        Args:
            df: OHLCV DataFrame with at least 60 rows of history.

        Returns:
            Prediction with direction and confidence, or None if model not trained.
        """
        if not self._trained:
            log.warning("prediction_skipped", reason="model not trained")
            return None

        features = self._feature_engine.compute(df)
        if features.empty:
            return None

        # Use only the last row (current prediction)
        X = features[self._feature_names].iloc[[-1]]

        try:
            X_scaled = self._scaler.transform(X)
        except Exception as e:
            log.error("feature_scaling_failed", error=str(e))
            return None

        # Get predictions from each model
        all_probas = []
        for name, model in self._models.items():
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
        )

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """Train the ensemble on historical data.

        Args:
            df: OHLCV DataFrame with sufficient history.

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

        # Scale features
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = {}

        # Train XGBoost
        if "xgboost" in self._config.ensemble_weights:
            acc = self._train_xgboost(X_scaled, y, tscv)
            accuracies["xgboost"] = acc

        # Train LightGBM
        if "lightgbm" in self._config.ensemble_weights:
            acc = self._train_lightgbm(X_scaled, y, tscv)
            accuracies["lightgbm"] = acc

        self._trained = bool(self._models)
        if self._trained:
            self._save_models()
            log.info("ensemble_trained", accuracies=accuracies, features=len(self._feature_names))

        return accuracies

    def load_models(self) -> bool:
        """Load previously trained models from disk."""
        model_file = MODEL_DIR / "ensemble.pkl"
        if not model_file.exists():
            return False

        try:
            with open(model_file, "rb") as f:
                data = pickle.load(f)
            self._models = data["models"]
            self._scaler = data["scaler"]
            self._feature_names = data["feature_names"]
            self._trained = True
            log.info("models_loaded", models=list(self._models.keys()))
            return True
        except Exception as e:
            log.error("model_load_failed", error=str(e))
            return False

    def _save_models(self) -> None:
        """Save trained models to disk."""
        model_file = MODEL_DIR / "ensemble.pkl"
        try:
            with open(model_file, "wb") as f:
                pickle.dump(
                    {
                        "models": self._models,
                        "scaler": self._scaler,
                        "feature_names": self._feature_names,
                    },
                    f,
                )
            log.info("models_saved", path=str(model_file))
        except Exception as e:
            log.error("model_save_failed", error=str(e))

    def _create_labels(self, close: pd.Series) -> pd.Series:
        """Create 3-class labels from next-day returns.

        0 = bearish (return < -0.5%)
        1 = neutral (-0.5% to +0.5%)
        2 = bullish (return > +0.5%)
        """
        next_return = close.pct_change().shift(-1)
        labels = pd.Series(1, index=close.index, dtype=float)  # Default neutral
        labels[next_return > 0.005] = 2  # Bullish
        labels[next_return < -0.005] = 0  # Bearish
        return labels

    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, tscv) -> float:
        """Train XGBoost model."""
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
                use_label_encoder=False,
                verbosity=0,
                n_jobs=-1,
            )

            # Cross-validate
            scores = []
            for train_idx, val_idx in tscv.split(X):
                model.fit(X[train_idx], y[train_idx])
                score = model.score(X[val_idx], y[val_idx])
                scores.append(score)

            # Final model on all data
            model.fit(X, y)
            self._models["xgboost"] = model

            avg_acc = float(np.mean(scores))
            log.info("xgboost_trained", cv_accuracy=f"{avg_acc:.3f}")
            return avg_acc

        except ImportError:
            log.warning("xgboost_not_installed")
            return 0.0

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, tscv) -> float:
        """Train LightGBM model."""
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
            )

            scores = []
            for train_idx, val_idx in tscv.split(X):
                model.fit(X[train_idx], y[train_idx])
                score = model.score(X[val_idx], y[val_idx])
                scores.append(score)

            model.fit(X, y)
            self._models["lightgbm"] = model

            avg_acc = float(np.mean(scores))
            log.info("lightgbm_trained", cv_accuracy=f"{avg_acc:.3f}")
            return avg_acc

        except ImportError:
            log.warning("lightgbm_not_installed")
            return 0.0

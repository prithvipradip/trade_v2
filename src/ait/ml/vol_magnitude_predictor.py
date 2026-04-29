"""Vol-magnitude prediction model for long straddles.

Long straddles profit when REALIZED vol exceeds IMPLIED vol over the
holding period. (You bought volatility cheap, sold/exercised expensive.)

Direction is irrelevant — what matters is the SIZE of the move.

Binary classifier:
    Y = 1 if max|return over next N days| > implied_vol_threshold
    Y = 0 otherwise

Output: P(big move) — directly usable as long straddle confidence.

Why complementary to range model:
    Range model predicts P(stays in range)   → iron condor confidence
    Vol-magnitude predicts P(breaks out)     → straddle confidence
    Together they cover both sides of the volatility regime question.
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ait.ml.features import FeatureEngine
from ait.utils.logging import get_logger

log = get_logger("ml.vol_magnitude")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VolMagnitudePrediction:
    """Output of vol-magnitude model."""

    probability_big_move: float       # 0.0–1.0
    threshold_pct: float              # the % move that defines "big" (e.g. 0.07 = 7%)
    horizon_days: int                 # the N days lookahead
    confidence: float                 # max(p, 1-p)
    features_used: int = 0
    model_version: str = ""


class VolMagnitudePredictor:
    """Binary classifier: P(realized abs return > threshold over N days).

    Default: ±7% over 30 days. Set higher than range_predictor's ±5% so
    the two models cover non-overlapping regimes (range vs breakout).
    """

    def __init__(
        self,
        threshold_pct: float = 0.07,
        horizon_days: int = 30,
        ensemble_weights: dict[str, float] | None = None,
    ) -> None:
        self._threshold = threshold_pct
        self._horizon = horizon_days
        self._weights = ensemble_weights or {"xgboost": 0.5, "lightgbm": 0.5}
        self._models: dict = {}
        self._symbol_models: dict[str, dict] = {}
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._feature_engine = FeatureEngine()
        self._trained = False
        self._model_version = ""

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def model_version(self) -> str:
        return self._model_version

    def _create_labels(self, close: pd.Series) -> pd.Series:
        """Label = 1 if max abs return over next N days > threshold."""
        labels = pd.Series(np.nan, index=close.index, dtype=float)
        for t in range(len(close) - self._horizon):
            base = close.iloc[t]
            future = close.iloc[t + 1: t + 1 + self._horizon]
            max_dev = float((future / base - 1).abs().max())
            labels.iloc[t] = 1.0 if max_dev > self._threshold else 0.0
        return labels

    def train(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        market_context: dict | None = None,
    ) -> dict[str, float]:
        """Train vol-magnitude model on historical data."""
        features = self._feature_engine.compute(df, market_context=market_context)
        if len(features) < 100:
            log.warning("vol_mag_insufficient_data", rows=len(features), required=100)
            return {}

        features["target"] = self._create_labels(features["Close"])
        features = features.dropna(subset=["target"])

        positive_rate = features["target"].mean()
        if positive_rate < 0.05 or positive_rate > 0.95:
            log.warning(
                "vol_mag_imbalanced", symbol=symbol,
                big_move_pct=f"{positive_rate:.1%}",
                hint=f"Threshold {self._threshold:.0%} may need tuning",
            )

        self._feature_names = self._feature_engine.get_feature_names()
        self._feature_names = [f for f in self._feature_names if f in features.columns]

        X = features[self._feature_names].values
        y = features["target"].values.astype(int)

        accuracies = {}

        if "xgboost" in self._weights:
            acc = self._train_xgb(X, y)
            accuracies["xgboost"] = acc

        if "lightgbm" in self._weights:
            acc = self._train_lgb(X, y)
            accuracies["lightgbm"] = acc

        # Final fit on all data
        self._scaler.fit(X)
        X_scaled = pd.DataFrame(
            self._scaler.transform(X), columns=self._feature_names,
        )
        for model in self._models.values():
            model.fit(X_scaled, y)

        self._trained = bool(self._models)
        if self._trained:
            from datetime import datetime
            self._model_version = f"volmag-v-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            if symbol:
                import copy
                self._symbol_models[symbol] = {
                    "models": copy.deepcopy(self._models),
                    "scaler": copy.deepcopy(self._scaler),
                    "feature_names": list(self._feature_names),
                    "cv_scores": dict(accuracies),
                    "big_move_rate": float(positive_rate),
                    "version": self._model_version,
                }

            self._save_models()
            log.info(
                "vol_mag_trained",
                symbol=symbol,
                accuracies=accuracies,
                big_move_rate=f"{positive_rate:.2%}",
                threshold=self._threshold,
                horizon=self._horizon,
            )

        return accuracies

    def _walk_forward_split(self, n: int, n_splits: int = 4, gap: int = 5):
        splits = []
        fold = n // (n_splits + 1)
        for i in range(n_splits):
            train_end = fold * (i + 1)
            val_start = train_end + gap
            val_end = min(val_start + fold, n)
            if val_end <= val_start:
                continue
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))
        return splits

    def _train_xgb(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                verbosity=0, n_jobs=-1, random_state=42,
            )
            scores = []
            for tr_idx, val_idx in self._walk_forward_split(len(X)):
                fold_scaler = StandardScaler()
                X_tr = pd.DataFrame(
                    fold_scaler.fit_transform(X[tr_idx]), columns=self._feature_names,
                )
                X_val = pd.DataFrame(
                    fold_scaler.transform(X[val_idx]), columns=self._feature_names,
                )
                model.fit(X_tr, y[tr_idx])
                scores.append(model.score(X_val, y[val_idx]))
            self._models["xgboost"] = model
            avg = float(np.mean(scores)) if scores else 0.0
            log.info("vol_mag_xgb_trained", cv_accuracy=f"{avg:.3f}", folds=len(scores))
            return avg
        except ImportError:
            return 0.0

    def _train_lgb(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary", metric="binary_logloss",
                verbose=-1, n_jobs=-1, class_weight="balanced",
                random_state=42, deterministic=True,
            )
            scores = []
            for tr_idx, val_idx in self._walk_forward_split(len(X)):
                fold_scaler = StandardScaler()
                X_tr = pd.DataFrame(
                    fold_scaler.fit_transform(X[tr_idx]), columns=self._feature_names,
                )
                X_val = pd.DataFrame(
                    fold_scaler.transform(X[val_idx]), columns=self._feature_names,
                )
                model.fit(X_tr, y[tr_idx])
                scores.append(model.score(X_val, y[val_idx]))
            self._models["lightgbm"] = model
            avg = float(np.mean(scores)) if scores else 0.0
            log.info("vol_mag_lgb_trained", cv_accuracy=f"{avg:.3f}", folds=len(scores))
            return avg
        except ImportError:
            return 0.0

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        market_context: dict | None = None,
        live_signals: dict | None = None,
    ) -> VolMagnitudePrediction | None:
        """Predict P(big move > threshold over horizon_days)."""
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
            return None

        features = self._feature_engine.compute(
            df, market_context=market_context, live_signals=live_signals,
        )
        if features.empty:
            return None

        X = features[feature_names].iloc[[-1]]
        try:
            X_scaled = pd.DataFrame(
                scaler.transform(X.values), columns=feature_names,
            )
        except Exception as e:
            log.error("vol_mag_scaling_failed", symbol=symbol, error=str(e))
            return None

        weighted_p = 0.0
        total_weight = 0.0
        for name, model in models.items():
            w = self._weights.get(name, 0.5)
            try:
                p = float(model.predict_proba(X_scaled)[0][1])
                weighted_p += w * p
                total_weight += w
            except Exception:
                continue

        if total_weight == 0:
            return None
        p_big_move = weighted_p / total_weight

        return VolMagnitudePrediction(
            probability_big_move=p_big_move,
            threshold_pct=self._threshold,
            horizon_days=self._horizon,
            confidence=max(p_big_move, 1 - p_big_move),
            features_used=len(feature_names),
            model_version=self._model_version,
        )

    def _save_models(self) -> None:
        path = MODEL_DIR / "vol_magnitude.pkl"
        with open(path, "wb") as f:
            pickle.dump({
                "models": self._models,
                "scaler": self._scaler,
                "feature_names": self._feature_names,
                "version": self._model_version,
                "threshold": self._threshold,
                "horizon": self._horizon,
                "symbol_models": self._symbol_models,
            }, f)
        log.info("vol_mag_models_saved", path=str(path), version=self._model_version)

    def load_models(self) -> bool:
        path = MODEL_DIR / "vol_magnitude.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._models = data["models"]
            self._scaler = data["scaler"]
            self._feature_names = data["feature_names"]
            self._model_version = data["version"]
            self._threshold = data.get("threshold", 0.07)
            self._horizon = data.get("horizon", 30)
            self._symbol_models = data.get("symbol_models", {})
            self._trained = bool(self._models)
            log.info("vol_mag_models_loaded", version=self._model_version)
            return True
        except Exception as e:
            log.warning("vol_mag_load_failed", error=str(e))
            return False

"""Directional ML model for credit spread trading.

Separate from the iron condor ensemble which predicts range-bound behavior.
This model focuses on TREND CONTINUATION — predicting strong directional moves
with high precision. Trades less often but with higher conviction.

Key differences from ensemble.py:
- Binary classification (UP vs DOWN) — no neutral class
- 3-day forward return with higher threshold (±2%)
- Momentum/trend features weighted more heavily
- Trained on regime-filtered data (only trending periods)
- Higher precision target (fewer trades, more accurate)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ait.ml.features import FeatureEngine
from ait.strategies.base import SignalDirection
from ait.utils.logging import get_logger

log = get_logger("ml.directional")

MODEL_DIR = Path("models")


@dataclass
class DirectionalSignal:
    direction: SignalDirection
    confidence: float
    trend_strength: float   # How strong is the trend (0-1)
    momentum_score: float   # Momentum alignment (0-1)


class DirectionalModel:
    """High-precision directional predictor for credit spread trades.

    Only fires on strong trends with aligned momentum.
    Designed for small accounts where every trade matters.
    """

    def __init__(self):
        self._feature_engine = FeatureEngine()
        self._scaler = StandardScaler()
        self._model = None
        self._feature_names: list[str] = []
        self._trained = False
        self._model_version = ""
        self._cv_scores: dict[str, float] = {}
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_trained(self) -> bool:
        return self._trained

    def predict(self, df: pd.DataFrame) -> DirectionalSignal | None:
        """Predict direction with high conviction only.

        Returns None if not confident enough — better to skip than guess.
        """
        if not self._trained:
            return None

        features = self._feature_engine.compute(df)
        if features.empty:
            return None

        X = features[self._feature_names].iloc[[-1]]

        try:
            X_scaled = self._scaler.transform(X.values)
        except Exception:
            return None

        proba = self._model.predict_proba(X_scaled)[0]

        # Binary: 0=DOWN, 1=UP
        pred_class = int(np.argmax(proba))
        confidence = float(proba[pred_class])

        # Calculate trend strength from features
        last_row = features.iloc[-1]
        trend_strength = self._calc_trend_strength(last_row)
        momentum_score = self._calc_momentum_score(last_row)

        # Only signal if confidence AND trend AND momentum all align
        if confidence < 0.70:
            return None
        if trend_strength < 0.4:
            return None

        direction = SignalDirection.BULLISH if pred_class == 1 else SignalDirection.BEARISH

        # Boost confidence when trend and momentum strongly agree
        alignment_bonus = min(0.1, trend_strength * momentum_score * 0.15)
        final_confidence = min(0.99, confidence + alignment_bonus)

        log.info("directional_signal",
                 direction=direction.value,
                 confidence=f"{final_confidence:.3f}",
                 trend=f"{trend_strength:.2f}",
                 momentum=f"{momentum_score:.2f}")

        return DirectionalSignal(
            direction=direction,
            confidence=final_confidence,
            trend_strength=trend_strength,
            momentum_score=momentum_score,
        )

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """Train on historical data with directional labels."""
        features = self._feature_engine.compute(df)
        if len(features) < 200:
            log.warning("directional_insufficient_data", rows=len(features))
            return {}

        # Binary labels: UP (1) if 5-day return > +1%, DOWN (0) if < -1%
        # Skip weak/neutral periods — only train on clear directional moves
        close = features["Close"]
        fwd_return = close.pct_change(5).shift(-5)
        features["target"] = np.nan
        features.loc[fwd_return > 0.01, "target"] = 1    # Up move
        features.loc[fwd_return < -0.01, "target"] = 0   # Down move
        # Drop neutral rows — we only learn from directional moves
        features = features.dropna(subset=["target"])

        if len(features) < 100:
            log.warning("directional_too_few_decisive_moves", rows=len(features))
            return {}

        self._feature_names = self._feature_engine.get_feature_names()
        self._feature_names = [f for f in self._feature_names if f in features.columns]

        X = features[self._feature_names].values
        y = features["target"].values.astype(int)

        # Log class balance
        n_up = int(y.sum())
        n_down = len(y) - n_up
        log.info("directional_class_balance", up=n_up, down=n_down, total=len(y))

        # Train XGBoost with precision-focused hyperparams
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = {}

        try:
            import xgboost as xgb

            scores = []
            for train_idx, val_idx in tscv.split(X):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_val = scaler.transform(X[val_idx])

                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    min_child_weight=10,     # More conservative — fewer spurious splits
                    subsample=0.7,
                    colsample_bytree=0.7,
                    reg_alpha=1.0,           # L1 regularization — sparse features
                    reg_lambda=2.0,          # L2 regularization — prevent overfitting
                    scale_pos_weight=n_down / max(n_up, 1),  # Balance classes
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X_train, y[train_idx])
                preds = model.predict(X_val)
                acc = float(np.mean(preds == y[val_idx]))
                scores.append(acc)

            avg_acc = float(np.mean(scores))
            accuracies["xgboost_directional"] = avg_acc
            log.info("directional_trained", cv_accuracy=f"{avg_acc:.3f}", folds=len(scores))

            # Final fit on all data
            self._scaler.fit(X)
            X_scaled = self._scaler.transform(X)
            model.fit(X_scaled, y)
            self._model = model
            self._trained = True

            from datetime import datetime
            self._model_version = f"dir-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self._cv_scores = accuracies
            self._save()

        except ImportError:
            log.warning("xgboost_not_installed")

        return accuracies

    def _calc_trend_strength(self, row: pd.Series) -> float:
        """Calculate trend strength from features (0-1)."""
        signals = []
        # SMA alignment: price > SMA20 > SMA50
        if "price_vs_sma_20" in row.index:
            signals.append(1.0 if row["price_vs_sma_20"] > 0.02 else
                          -1.0 if row["price_vs_sma_20"] < -0.02 else 0.0)
        if "sma_10_20_cross" in row.index:
            signals.append(1.0 if row["sma_10_20_cross"] > 0 else -1.0)
        if "macd_hist" in row.index:
            signals.append(1.0 if row["macd_hist"] > 0 else -1.0)
        if "adx" in row.index:
            # ADX > 25 = trending, > 40 = strong trend
            signals.append(min(1.0, row["adx"] / 40.0) if row["adx"] > 20 else 0.0)

        if not signals:
            return 0.5
        # Strength = how aligned all trend signals are (0 = conflicting, 1 = all agree)
        avg = abs(np.mean(signals))
        return float(min(1.0, avg))

    def _calc_momentum_score(self, row: pd.Series) -> float:
        """Calculate momentum alignment (0-1)."""
        signals = []
        if "rsi_14" in row.index:
            rsi = row["rsi_14"]
            # RSI 50-70 = bullish momentum, 30-50 = bearish momentum
            if rsi > 55:
                signals.append(min(1.0, (rsi - 50) / 30))
            elif rsi < 45:
                signals.append(min(1.0, (50 - rsi) / 30))
            else:
                signals.append(0.0)
        if "volume_sma_20_ratio" in row.index:
            # Volume confirmation: above average = momentum confirmed
            signals.append(min(1.0, max(0.0, row["volume_sma_20_ratio"] - 1.0)))
        if "weekly_trend_aligned" in row.index:
            signals.append(float(row["weekly_trend_aligned"]))

        if not signals:
            return 0.5
        return float(np.mean(signals))

    def _save(self):
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "version": self._model_version,
            "cv_scores": self._cv_scores,
        }
        try:
            path = MODEL_DIR / "directional.pkl"
            with open(path, "wb") as f:
                pickle.dump(data, f)
            log.info("directional_model_saved", path=str(path))
        except Exception as e:
            log.error("directional_save_failed", error=str(e))

    def load(self) -> bool:
        path = MODEL_DIR / "directional.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._scaler = data["scaler"]
            self._feature_names = data["feature_names"]
            self._model_version = data.get("version", "unknown")
            self._cv_scores = data.get("cv_scores", {})
            self._trained = True
            log.info("directional_model_loaded", version=self._model_version)
            return True
        except Exception as e:
            log.error("directional_load_failed", error=str(e))
            return False

"""Meta-labeling model — learns which signals to take vs skip.

The primary model predicts direction (bullish/bearish/neutral).
This secondary model predicts: "given this signal + market context,
will the trade actually be profitable?"

Trained on the bot's own trade history (trade_context + outcomes).
A trade is "good" if it exited with positive realized P&L.

This directly reduces false positives: the primary model might correctly
predict "stock goes up" but the trade still loses due to timing, IV crush,
or unfavorable regime. Meta-labeling catches these patterns.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ait.utils.logging import get_logger

log = get_logger("ml.meta_label")

MODEL_DIR = Path("models")

# Minimum trades needed before the meta-labeler has enough signal
MIN_TRADES_FOR_TRAINING = 30

# Features the meta-labeler uses (all available at trade entry time)
META_FEATURES = [
    "primary_confidence",
    "regime_trending_up",
    "regime_trending_down",
    "regime_high_vol",
    "regime_range_bound",
    "vix",
    "iv_rank",
    "sentiment_score",
    "rsi_14",
    "rsi_7",
    "bb_position",
    "volume_sma_20_ratio",
    "realized_vol_20",
    "atr_pct",
    "weekly_trend_aligned",
    "volume_confirmation",
    "hour_of_day",
    "macd_hist",
    "price_vs_sma_20",
    "sma_10_20_cross",
]


@dataclass
class MetaSignal:
    """Output of the meta-labeling model."""

    take_trade: bool
    probability: float  # P(profitable) — 0.0 to 1.0
    features_used: int


class MetaLabeler:
    """Binary classifier: should we take this trade or skip it?

    Trained on historical trade outcomes from the bot's own journal.
    Uses market context features available at entry time.
    """

    def __init__(self, min_probability: float = 0.50) -> None:
        self._min_probability = min_probability
        self._model = None
        self._scaler = StandardScaler()
        self._trained = False
        self._feature_names: list[str] = []
        self._model_version: str = ""
        self._training_stats: dict = {}

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def training_stats(self) -> dict:
        return self._training_stats

    def predict(self, context: dict) -> MetaSignal | None:
        """Predict whether a trade signal should be taken.

        Args:
            context: Dict with keys matching META_FEATURES.
                     Missing features are filled with 0.

        Returns:
            MetaSignal with take/skip decision, or None if not trained.
        """
        if not self._trained:
            return None

        # Build feature vector
        feature_values = [context.get(f, 0.0) for f in self._feature_names]
        X = np.array([feature_values])

        try:
            X_scaled = self._scaler.transform(X)
            prob_profitable = float(self._model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            log.warning("meta_label_predict_failed", error=str(e))
            return None

        take = prob_profitable >= self._min_probability

        log.info(
            "meta_label_prediction",
            take=take,
            probability=f"{prob_profitable:.3f}",
            threshold=self._min_probability,
        )

        return MetaSignal(
            take_trade=take,
            probability=prob_profitable,
            features_used=len(self._feature_names),
        )

    def train(self, trades_df: pd.DataFrame) -> dict:
        """Train meta-labeler on historical trade outcomes.

        Args:
            trades_df: DataFrame with columns matching META_FEATURES plus
                       'profitable' (1/0) as the target.

        Returns:
            Dict with training metrics.
        """
        if len(trades_df) < MIN_TRADES_FOR_TRAINING:
            log.warning(
                "insufficient_trades_for_meta_label",
                trades=len(trades_df),
                required=MIN_TRADES_FOR_TRAINING,
            )
            return {}

        # Use only features that exist in the data
        available = [f for f in META_FEATURES if f in trades_df.columns]
        if not available:
            log.warning("no_meta_features_available")
            return {}

        self._feature_names = available
        X = trades_df[available].fillna(0).values
        y = trades_df["profitable"].values.astype(int)

        # Class balance check
        n_positive = int(y.sum())
        n_negative = len(y) - n_positive
        if n_positive < 5 or n_negative < 5:
            log.warning(
                "meta_label_class_imbalance",
                profitable=n_positive,
                unprofitable=n_negative,
            )
            return {}

        # Scale features
        self._scaler.fit(X)
        X_scaled = self._scaler.transform(X)

        # Time-series cross-validation
        n_splits = min(5, max(2, len(y) // 10))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,  # Shallow — small dataset, avoid overfit
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=n_negative / max(1, n_positive),
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                verbosity=0,
                n_jobs=-1,
            )

            # Cross-validate
            accuracies = []
            precisions = []
            for train_idx, val_idx in tscv.split(X_scaled):
                model.fit(X_scaled[train_idx], y[train_idx])
                preds = model.predict(X_scaled[val_idx])
                acc = float(np.mean(preds == y[val_idx]))
                accuracies.append(acc)

                # Precision for "take trade" predictions
                take_mask = preds == 1
                if take_mask.sum() > 0:
                    precision = float(np.mean(y[val_idx][take_mask] == 1))
                    precisions.append(precision)

            # Final model on all data
            model.fit(X_scaled, y)
            self._model = model
            self._trained = True

            from datetime import datetime
            self._model_version = f"meta-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            avg_acc = float(np.mean(accuracies)) if accuracies else 0.0
            avg_precision = float(np.mean(precisions)) if precisions else 0.0

            # Feature importance
            importances = dict(zip(self._feature_names, model.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

            self._training_stats = {
                "accuracy": avg_acc,
                "precision": avg_precision,
                "trades_used": len(y),
                "profitable_ratio": n_positive / len(y),
                "features": len(self._feature_names),
                "top_features": {k: round(float(v), 3) for k, v in top_features},
            }

            self._save_model()

            log.info(
                "meta_label_trained",
                version=self._model_version,
                accuracy=f"{avg_acc:.3f}",
                precision=f"{avg_precision:.3f}",
                trades=len(y),
                top_features=[f[0] for f in top_features[:3]],
            )

            return self._training_stats

        except ImportError:
            log.warning("xgboost_not_installed_for_meta_label")
            return {}

    def build_training_data(self, state_manager) -> pd.DataFrame:
        """Build training DataFrame from trade history.

        Joins trade_context (entry features) with trade outcomes (P&L).
        Only uses closed trades with context data.

        Args:
            state_manager: StateManager instance with trade + context tables.

        Returns:
            DataFrame ready for training.
        """
        import sqlite3

        rows = []
        with sqlite3.connect(state_manager._db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Join trades with their entry context
            results = conn.execute("""
                SELECT t.trade_id, t.realized_pnl, t.entry_time,
                       c.entry_direction, c.entry_confidence, c.entry_regime,
                       c.entry_vix, c.entry_iv_rank, c.entry_sentiment_score
                FROM trades t
                JOIN trade_context c ON t.trade_id = c.trade_id
                WHERE t.status = 'closed'
                ORDER BY t.entry_time
            """).fetchall()

        for row in results:
            r = dict(row)
            entry_hour = 10  # Default
            try:
                from datetime import datetime
                entry_hour = datetime.fromisoformat(r["entry_time"]).hour
            except (ValueError, TypeError):
                pass

            regime = r.get("entry_regime", "")
            rows.append({
                "primary_confidence": r.get("entry_confidence", 0),
                "regime_trending_up": 1.0 if regime == "trending_up" else 0.0,
                "regime_trending_down": 1.0 if regime == "trending_down" else 0.0,
                "regime_high_vol": 1.0 if regime == "high_volatility" else 0.0,
                "regime_range_bound": 1.0 if regime == "range_bound" else 0.0,
                "vix": r.get("entry_vix", 0),
                "iv_rank": r.get("entry_iv_rank", 0),
                "sentiment_score": r.get("entry_sentiment_score", 0),
                "hour_of_day": entry_hour,
                "profitable": 1 if r["realized_pnl"] > 0 else 0,
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def load_model(self) -> bool:
        """Load previously trained meta-label model from disk."""
        model_file = MODEL_DIR / "meta_label.pkl"
        if not model_file.exists():
            return False

        try:
            with open(model_file, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._scaler = data["scaler"]
            self._feature_names = data["feature_names"]
            self._model_version = data.get("version", "unknown")
            self._training_stats = data.get("training_stats", {})
            self._trained = True
            log.info("meta_label_loaded", version=self._model_version)
            return True
        except Exception as e:
            log.error("meta_label_load_failed", error=str(e))
            return False

    def _save_model(self) -> None:
        """Save trained meta-label model to disk."""
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "version": self._model_version,
            "training_stats": self._training_stats,
        }

        model_file = MODEL_DIR / "meta_label.pkl"
        try:
            with open(model_file, "wb") as f:
                pickle.dump(data, f)
            log.info("meta_label_saved", version=self._model_version)
        except Exception as e:
            log.error("meta_label_save_failed", error=str(e))

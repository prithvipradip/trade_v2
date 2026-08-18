"""XGBoost + LightGBM ensemble for direction prediction.

Lightweight models that run efficiently on CPU (Apple Silicon).
No TensorFlow or PyTorch needed for inference.

Predicts: BULLISH / BEARISH / NEUTRAL for next trading session.
Output: direction + confidence score (0.0 to 1.0).
"""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Silence LightGBM's benign "X does not have valid feature names" warning
# during walk-forward CV (we pass numpy arrays, not DataFrames)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

from ait.config.settings import MLConfig
from ait.ml.features import FeatureEngine
from ait.strategies.base import SignalDirection
from ait.utils.logging import get_logger

log = get_logger("ml.ensemble")

# R16 #2: absolute, repo-anchored models dir (trade_v2/models). The old
# CWD-relative Path("models") meant any process's CWD decided which
# ensemble.pkl got read/WRITTEN — a walkforward run from repo root clobbered
# the LIVE artifact (the bot served an IWM-only window model for 2.5 market
# hours on 2026-08-03). The pytest conftest fence monkeypatches this module
# attribute to tmp_path; resolve it lazily (self.model_dir) so the fence
# keeps working.
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


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

    def __init__(
        self,
        config: MLConfig,
        model_dir: "Path | str | None" = None,
        persist_artifacts: bool = True,
    ) -> None:
        """Args:
            config: ML configuration.
            model_dir: R16 #2 artifact fence — where models are saved/loaded.
                None -> the module-level MODEL_DIR (absolute repo-anchored
                live models dir; monkeypatched to tmp_path under pytest).
                Research/window training must pass its own directory OR set
                persist_artifacts=False.
            persist_artifacts: False -> train() never writes to disk (used by
                walkforward window training — window models are throwaway and
                previously clobbered the LIVE models/ensemble.pkl).
        """
        self._config = config
        self._feature_engine = FeatureEngine()
        self._model_dir: Path | None = Path(model_dir) if model_dir is not None else None
        self._persist_artifacts = bool(persist_artifacts)

        # Per-symbol model storage: {symbol: {models, scaler, feature_names, scores}}
        self._symbol_models: dict[str, dict] = {}

        # Legacy single-model (used as fallback and for backtest compatibility)
        self._scaler = StandardScaler()
        self._models: dict[str, object] = {}
        self._trained = False
        self._feature_names: list[str] = []
        self._model_version: str = ""
        self._cv_scores: dict[str, float] = {}

        # Optional override dicts injected by ModelTrainer._apply_optimized_hyperparams()
        # so that the next train() call uses Optuna-tuned hyperparameters.
        self._xgb_kwargs: dict = {}
        self._lgbm_kwargs: dict = {}

        if self._persist_artifacts:
            self.model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_dir(self) -> Path:
        """Artifact directory: explicit constructor fence when given, else the
        module default (which the pytest conftest fence monkeypatches)."""
        return self._model_dir if self._model_dir is not None else MODEL_DIR

    @property
    def is_trained(self) -> bool:
        return self._trained or bool(self._symbol_models)

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        market_context: dict | None = None,
        live_signals: dict | None = None,
        intraday_store: "HistoricalDataStore | None" = None,
    ) -> Prediction | None:
        """Predict direction for the latest data point.

        Args:
            df: OHLCV DataFrame with at least 60 rows of history.
            symbol: Symbol name — uses per-symbol model if available.
            market_context: Optional dict with cross-asset data (VIX, SPY).
            intraday_store: Optional store for VLMC intraday feature enrichment.

        Returns:
            Prediction with direction and confidence, or None if model not trained.
        """
        # Select the right model for this symbol
        sym_data: dict | None = None
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

        features = self._feature_engine.compute(
            df, market_context=market_context, live_signals=live_signals,
            intraday_store=intraday_store, symbol=symbol,
        )
        if features.empty:
            log.warning("prediction_skipped", reason="features_empty",
                        input_rows=len(df))
            return None

        # Use only the last row (current prediction)
        # Deep-audit ML-F7: direct column indexing raised KeyError when a
        # VLMC-trained model met a symbol whose intraday store was empty at
        # predict time. Reindex + neutral-fill instead (predict_from_features
        # already did this).
        X = features.reindex(columns=feature_names).fillna(0.0).iloc[[-1]]

        try:
            # Keep as DataFrame with feature names so LightGBM/sklearn agree
            X_scaled = pd.DataFrame(
                scaler.transform(X.values),
                columns=feature_names,
            )
        except Exception as e:
            log.error("feature_scaling_failed", error=str(e), symbol=symbol)
            return None

        # Get predictions from each model — prefer fitted weights when available,
        # fall back to config ensemble_weights.
        fw = (
            (sym_data or {}).get("fitted_weights")
            or getattr(self, "_fitted_weights", None)
        )
        all_probas = []
        total_weight = 0.0
        for name, model in models.items():
            weight = fw.get(name, 0.5) if fw else self._config.ensemble_weights.get(name, 0.5)
            try:
                proba = model.predict_proba(X_scaled)[0]
                all_probas.append(proba * weight)
                total_weight += weight
            except Exception as e:
                log.warning("model_prediction_failed", model=name, error=str(e))

        if not all_probas:
            return None

        # Weighted average of probabilities
        avg_proba = np.sum(all_probas, axis=0) / (total_weight or 1.0)

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
            features_used=len(feature_names),  # per-symbol list, not the global one (deep-audit ML-F7)
            model_version=self._model_version,
        )

    def predict_from_features(
        self,
        feature_row: "pd.Series",
        symbol: str = "",
    ) -> "Prediction | None":
        """Make a prediction from a pre-computed feature row, bypassing FeatureEngine.

        Used by _save_window_timeseries for O(1) per-bar predictions instead of
        re-running FeatureEngine on 252+ rows for each bar.
        """
        sym_data: dict | None = None
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

        try:
            X = pd.DataFrame(
                [feature_row.reindex(feature_names).fillna(0.0).values],
                columns=feature_names,
            )
            X_scaled = pd.DataFrame(
                scaler.transform(X.values),
                columns=feature_names,
            )
        except Exception as e:
            log.error("feature_scaling_failed", error=str(e), symbol=symbol)
            return None

        fw = (
            (sym_data or {}).get("fitted_weights")
            or getattr(self, "_fitted_weights", None)
        )
        all_probas = []
        total_weight = 0.0
        for name, model in models.items():
            weight = fw.get(name, 0.5) if fw else self._config.ensemble_weights.get(name, 0.5)
            try:
                proba = model.predict_proba(X_scaled)[0]
                all_probas.append(proba * weight)
                total_weight += weight
            except Exception as e:
                log.warning("model_prediction_failed", model=name, error=str(e))

        if not all_probas:
            return None

        avg_proba = np.sum(all_probas, axis=0) / (total_weight or 1.0)
        pred_class = int(np.argmax(avg_proba))
        confidence = float(avg_proba[pred_class])
        direction = self.LABELS[pred_class]
        probabilities = {
            "bearish": float(avg_proba[0]),
            "neutral": float(avg_proba[1]),
            "bullish": float(avg_proba[2]),
        }
        return Prediction(
            direction=direction,
            confidence=confidence,
            probabilities=probabilities,
            features_used=len(feature_names),
            model_version=self._model_version,
        )

    def train(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        market_context: dict | None = None,
        intraday_store: "HistoricalDataStore | None" = None,
    ) -> dict[str, float]:
        """Train the ensemble on historical data.

        Args:
            df: OHLCV DataFrame with sufficient history.
            symbol: If provided, stores model per-symbol (prevents cross-contamination).
            market_context: Optional dict with cross-asset data (VIX, SPY).
            intraday_store: Optional store for VLMC intraday feature enrichment.

        Returns:
            Dict of model accuracies.
        """
        features = self._feature_engine.compute(
            df, market_context=market_context,
            intraday_store=intraday_store, symbol=symbol,
        )
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

        # Include VLMC names when intraday_store was used (Gap A fix).
        # The filter below drops any that are absent (e.g. empty store).
        self._feature_names = self._feature_engine.get_feature_names(
            include_vlmc=intraday_store is not None
        )
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
        X_scaled = pd.DataFrame(
            self._scaler.transform(X),
            columns=self._feature_names,
        )
        for model in self._models.values():
            model.fit(X_scaled, y)

        self._trained = bool(self._models)
        if self._trained:
            from datetime import datetime
            self._model_version = f"v-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self._cv_scores = accuracies

            # Fit ensemble weights from per-model CV edge over 0.5 AUROC baseline.
            # AUROC random baseline = 0.5 (one-vs-rest, any number of classes).
            _baseline = 0.5
            _edges = {m: max(0.0, acc - _baseline) for m, acc in accuracies.items()}
            _total = sum(_edges.values())
            fitted_weights = (
                {m: e / _total for m, e in _edges.items()}
                if _total > 0
                else {m: 0.5 for m in accuracies}
            )
            self._fitted_weights: dict[str, float] = fitted_weights

            # Store per-symbol model (deep copy so next train() doesn't overwrite)
            if symbol:
                import copy
                # Extract feature importances (XGB + LGBM agree on this attr)
                importances = {}
                for name, model in self._models.items():
                    if hasattr(model, "feature_importances_"):
                        try:
                            arr = model.feature_importances_
                            importances[name] = dict(zip(self._feature_names, arr.tolist()))
                        except Exception:
                            pass

                self._symbol_models[symbol] = {
                    "models": copy.deepcopy(self._models),
                    "scaler": copy.deepcopy(self._scaler),
                    "feature_names": list(self._feature_names),
                    "cv_scores": dict(accuracies),
                    "fitted_weights": dict(fitted_weights),
                    "feature_importances": importances,
                    "version": self._model_version,
                }
                log.info("symbol_model_stored", symbol=symbol,
                         accuracies=accuracies, fitted_weights=fitted_weights,
                         features=len(self._feature_names))

            # R16 #2: window/research training passes persist_artifacts=False
            # so a train() can NEVER touch the live ensemble.pkl (or churn its
            # versioned rollback copies).
            if self._persist_artifacts:
                self._save_models()
            else:
                log.info(
                    "ensemble_not_persisted",
                    version=self._model_version,
                    reason="persist_artifacts=False (research/window model)",
                )
            log.info(
                "ensemble_trained",
                version=self._model_version,
                accuracies=accuracies,
                fitted_weights=fitted_weights,
                features=len(self._feature_names),
            )

        return accuracies

    @property
    def fitted_weights(self) -> "dict[str, float] | None":
        return getattr(self, "_fitted_weights", None)

    def load_models(self, version: str | None = None) -> bool:
        """Load previously trained models from disk.

        Args:
            version: Specific version to load. None = latest (ensemble.pkl).
        """
        if version:
            model_file = self.model_dir / f"ensemble_{version}.pkl"
        else:
            model_file = self.model_dir / "ensemble.pkl"

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

    def _save_models(self) -> bool:
        """Save trained models to disk with versioning.

        R19: returns whether models/ensemble.pkl now holds THIS in-memory
        model. rollback() must know — a rollback whose write failed leaves the
        degraded artifact on disk for the live bot to load, which has to be
        loud, not silent.
        """
        data = {
            "models": self._models,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "version": self._model_version,
            "cv_scores": self._cv_scores,
            "symbol_models": self._symbol_models,
        }

        # Save as current (ensemble.pkl) — ATOMIC: write tmp then os.replace,
        # so a concurrent reader (live bot) or a second writer (7:30 retrain
        # subprocess vs mid-day bot train) can never observe/produce a
        # half-written pickle. A corrupted-model event was already referenced
        # in adaptor.py (2026-06-23). Audit 2026-07-07 item 3.1.
        import os as _os
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_file = self.model_dir / "ensemble.pkl"
        tmp_file = self.model_dir / "ensemble.pkl.tmp"
        try:
            with open(tmp_file, "wb") as f:
                pickle.dump(data, f)
            _os.replace(tmp_file, model_file)
        except Exception as e:
            log.error("model_save_failed", error=str(e))
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                pass
            return False

        # Save versioned copy for rollback
        if self._model_version:
            versioned_file = self.model_dir / f"ensemble_{self._model_version}.pkl"
            try:
                with open(versioned_file, "wb") as f:
                    pickle.dump(data, f)
                log.info("models_saved", path=str(versioned_file), version=self._model_version)
            except Exception as e:
                log.warning("versioned_save_failed", error=str(e))

            # Prune old versions (keep last 5)
            self._prune_old_versions(keep=5)

        return True

    def _prune_old_versions(self, keep: int = 5) -> None:
        """Remove old model versions, keeping only the N most recent."""
        versions = sorted(self.model_dir.glob("ensemble_v-*.pkl"))
        if len(versions) > keep:
            for old in versions[:-keep]:
                old.unlink()
                log.debug("pruned_old_model", path=str(old))

    def rollback(self, version: str) -> bool:
        """Roll back to a specific model version, IN MEMORY AND ON DISK.

        Returns True only when models/ensemble.pkl now holds `version`.

        R19: this used to load_models() and stop there. The retrain that
        decides to roll back runs in the 7:30 subprocess (master.py) which
        exits seconds later, so the in-memory revert died with it while the
        just-written models/ensemble.pkl still held the DEGRADED model — the
        live bot's next load_models() (orchestrator ensure_models_ready)
        served exactly the model the rollback had rejected. A rollback that
        does not rewrite the artifact is not a rollback.

        The "target missing" fallback also had to change: loading the latest
        ensemble.pkl loads the degraded model itself, so it can never be
        reported as a successful rollback (and must never be re-persisted).
        Keep the load for in-memory continuity, return False, and say so
        loudly — an unprotected retrain is an operator-visible event.
        """
        log.info("model_rollback_requested", target_version=version)
        if not self.load_models(version=version):
            log.critical(
                "model_rollback_target_missing",
                version=version,
                note="rollback artifact absent — the DEGRADED model stays on "
                     "disk and will be loaded by the live bot; needs review",
            )
            self.load_models()  # in-memory continuity only — NOT a rollback
            return False

        # R16 #2 artifact fence still applies: a research/window predictor
        # must never write the live ensemble.pkl, rollback or not.
        if not self._persist_artifacts:
            log.info("model_rollback_not_persisted", version=version,
                     reason="persist_artifacts=False (research/window model)")
            return True

        if not self._save_models():
            log.critical(
                "model_rollback_persist_failed",
                version=version,
                note="rolled back in memory but ensemble.pkl still holds the "
                     "degraded model — the live bot will load it on restart",
            )
            return False

        log.warning("model_rollback_persisted", version=self._model_version,
                    path=str(self.model_dir / "ensemble.pkl"))
        return True

    def list_versions(self) -> list[dict]:
        """List available model versions."""
        versions = []
        for f in sorted(self.model_dir.glob("ensemble_v-*.pkl")):
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

    @cv_scores.setter
    def cv_scores(self, scores: "dict[str, float]") -> None:
        """R19: cv_scores was getter-only, so the R17 auto-rollback fix
        (`self._predictor.cv_scores = {"all_symbol_mean": ...}` in
        trainer.train_all_symbols) raised AttributeError on EVERY retrain and
        its own `except Exception: pass` swallowed it. The rollback decision
        therefore kept comparing the LAST symbol's per-model scores — the
        ML-F9/A8 defect that two rounds of fixes each believed they had
        closed. A setter makes the assignment real; the trainer's `if _vals`
        guard already keeps it from ever being set to an empty dict.
        """
        if not isinstance(scores, dict):
            raise TypeError(
                f"cv_scores must be a dict, got {type(scores).__name__}"
            )
        self._cv_scores = dict(scores)

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
        # Deep-audit ML-F3: initialising to 1 stamped the last 5 rows (whose
        # forward return is unknowable NaN) as NEUTRAL on every retrain —
        # systematically wrong labels at the most regime-relevant rows.
        # NaN-init means dropna(subset=["target"]) removes them instead.
        labels = pd.Series(float("nan"), index=close.index, dtype=float)
        labels[fwd_return > 0.01] = 2    # Bullish: >+1.0% in 5 days
        labels[fwd_return < -0.01] = 0   # Bearish: <-1.0% in 5 days
        labels[(fwd_return <= 0.01) & (fwd_return >= -0.01)] = 1  # Neutral
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

            base_kwargs = dict(
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
                random_state=42,
            )
            base_kwargs.update(self._xgb_kwargs)
            model = XGBClassifier(**base_kwargs)

            # Walk-forward validation with purge gap
            wf_splits = self._walk_forward_split(len(X))
            scores = []
            for train_idx, val_idx in wf_splits:
                fold_scaler = StandardScaler()
                # Wrap in DataFrame with feature names to keep sklearn + models
                # aligned (silences feature-name warnings during CV)
                X_train = pd.DataFrame(
                    fold_scaler.fit_transform(X[train_idx]),
                    columns=self._feature_names,
                )
                X_val = pd.DataFrame(
                    fold_scaler.transform(X[val_idx]),
                    columns=self._feature_names,
                )
                sw = self._compute_sample_weights(y[train_idx])
                model.fit(X_train, y[train_idx], sample_weight=sw)
                # AUROC (one-vs-rest) is immune to class imbalance; raw accuracy
                # on imbalanced folds produces sub-random scores (< 1/3 baseline).
                try:
                    from sklearn.metrics import roc_auc_score
                    proba = model.predict_proba(X_val)
                    score = roc_auc_score(y[val_idx], proba, multi_class="ovr", average="macro")
                except Exception:
                    score = model.score(X_val, y[val_idx])
                scores.append(score)

            # Final model stored (will be refit on fully-scaled data after CV)
            self._models["xgboost"] = model

            avg_acc = float(np.mean(scores))
            log.info("xgboost_trained", cv_auroc=f"{avg_acc:.3f}", folds=len(scores))
            return avg_acc

        except ImportError:
            log.warning("xgboost_not_installed")
            return 0.0

    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, tscv) -> float:
        """Train LightGBM model with walk-forward validation."""
        try:
            from lightgbm import LGBMClassifier

            base_kwargs = dict(
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
                random_state=42,
                deterministic=True,
            )
            base_kwargs.update(self._lgbm_kwargs)
            model = LGBMClassifier(**base_kwargs)

            wf_splits = self._walk_forward_split(len(X))
            scores = []
            for train_idx, val_idx in wf_splits:
                fold_scaler = StandardScaler()
                # Wrap in DataFrame with feature names to keep sklearn + LightGBM
                # aligned — silences "feature names mismatch" warning during CV
                X_train = pd.DataFrame(
                    fold_scaler.fit_transform(X[train_idx]),
                    columns=self._feature_names,
                )
                X_val = pd.DataFrame(
                    fold_scaler.transform(X[val_idx]),
                    columns=self._feature_names,
                )
                model.fit(X_train, y[train_idx])
                # AUROC (one-vs-rest) is immune to class imbalance; raw accuracy
                # on imbalanced folds produces sub-random scores (< 1/3 baseline).
                try:
                    from sklearn.metrics import roc_auc_score
                    proba = model.predict_proba(X_val)
                    score = roc_auc_score(y[val_idx], proba, multi_class="ovr", average="macro")
                except Exception:
                    score = model.score(X_val, y[val_idx])
                scores.append(score)

            # Final model stored (will be refit on fully-scaled data after CV)
            self._models["lightgbm"] = model

            avg_acc = float(np.mean(scores))
            log.info("lightgbm_trained", cv_auroc=f"{avg_acc:.3f}", folds=len(scores))
            return avg_acc

        except ImportError:
            log.warning("lightgbm_not_installed")
            return 0.0

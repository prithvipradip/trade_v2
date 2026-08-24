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

# Research sandbox for backtest/walkforward artifacts — never the live bot's
# models/ (R7/R10, same hygiene rule as range_predictor.RESEARCH_MODEL_DIR).
RESEARCH_MODEL_DIR = MODEL_DIR / "research"


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
        # R20: config-backed training floor (was a hardcoded 100 that
        # shadowed ml.min_training_samples). None -> the config value.
        # Keyword-only: inserted ahead of the original params, a positional
        # caller would otherwise silently misbind (e.g. int(threshold_pct)
        # landing here, truncated to 0, disabling the training floor).
        *,
        min_training_samples: int | None = None,
        threshold_pct: float = 0.07,
        horizon_days: int = 30,
        ensemble_weights: dict[str, float] | None = None,
        model_dir: "Path | str | None" = None,
    ) -> None:
        if min_training_samples is None:
            # R20b follow-up: was a bare MLConfig() (pydantic field default,
            # never reads config.yaml) despite the comment above claiming
            # this resolves "the config value" — load_settings() is the
            # actual config-reading call, same as every sibling resolution
            # block (engine.py/walkforward.py/optimizer.py).
            try:
                from ait.config.settings import load_settings
                min_training_samples = load_settings().ml.min_training_samples
            except Exception:  # noqa: BLE001 — never block construction
                from ait.config.settings import MLConfig
                min_training_samples = MLConfig().min_training_samples
        self._min_training_samples = int(min_training_samples)
        self._threshold = threshold_pct
        self._horizon = horizon_days
        # Artifact directory — live default is models/. Research callers MUST
        # pass RESEARCH_MODEL_DIR so they never clobber the live artifact.
        self._model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
        # Designed spec (immutable) for the load-time spec guard.
        self._spec_threshold = float(threshold_pct)
        self._spec_horizon = int(horizon_days)
        self._spec_mismatch = False
        self._trained_at: "str | None" = None
        self._global_model_symbol: str = ""
        self._fallback_warned: set[str] = set()
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

    @property
    def spec_mismatch(self) -> bool:
        """True when the loaded artifact's threshold/horizon differ from the
        constructor's designed spec (see load_models)."""
        return self._spec_mismatch

    @property
    def trained_at(self) -> "str | None":
        """ISO timestamp of the last training run, persisted in the artifact."""
        return self._trained_at

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
        # Always train at the DESIGNED constructor spec — a loaded pickle may
        # have overwritten these with a foreign spec (see load_models guard).
        self._threshold = self._spec_threshold
        self._horizon = self._spec_horizon
        features = self._feature_engine.compute(df, market_context=market_context)
        if len(features) < self._min_training_samples:
            log.warning("vol_mag_insufficient_data", rows=len(features), required=self._min_training_samples)
            return {}

        features["target"] = self._create_labels(features["Close"])
        features = features.dropna(subset=["target"])

        positive_rate = features["target"].mean()
        n_pos = int(features["target"].sum())
        n_neg = len(features) - n_pos
        _MIN_CLASS_SAMPLES = 10
        if n_pos < _MIN_CLASS_SAMPLES or n_neg < _MIN_CLASS_SAMPLES:
            log.warning(
                "vol_mag_single_class", symbol=symbol,
                big_move_pct=f"{positive_rate:.1%}",
                n_big_move=n_pos, n_small_move=n_neg,
                min_required=_MIN_CLASS_SAMPLES,
                hint=f"±{self._threshold:.0%} threshold produces too few minority-class "
                     f"examples to train reliably — cannot train binary classifier",
            )
            return {}
        if positive_rate < 0.05 or positive_rate > 0.98:
            log.warning(
                "vol_mag_imbalanced", symbol=symbol,
                big_move_pct=f"{positive_rate:.1%}",
                n_big_move=n_pos, n_small_move=n_neg,
                hint=f"Threshold {self._threshold:.0%} may need tuning",
            )

        self._feature_names = self._feature_engine.get_feature_names()
        self._feature_names = [f for f in self._feature_names if f in features.columns]

        X = features[self._feature_names].values
        y = features["target"].values.astype(int)

        accuracies = {}

        _spw = n_neg / n_pos if n_pos > 0 else 1.0

        if "xgboost" in self._weights:
            acc = self._train_xgb(X, y, scale_pos_weight=_spw)
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
            self._trained_at = datetime.now().isoformat()
            if symbol:
                self._global_model_symbol = symbol

            # Fit ensemble weights from per-model CV edge over 0.50 baseline.
            _baseline = 0.50
            _edges = {m: max(0.0, acc - _baseline) for m, acc in accuracies.items()}
            _total = sum(_edges.values())
            fitted_weights = (
                {m: e / _total for m, e in _edges.items()}
                if _total > 0
                else {m: 0.5 for m in accuracies}
            )
            self._fitted_weights: dict[str, float] = fitted_weights

            if symbol:
                import copy
                importances = {}
                for name, model in self._models.items():
                    if hasattr(model, "feature_importances_"):
                        try:
                            arr = model.feature_importances_
                            importances[name] = dict(zip(self._feature_names, arr.tolist()))
                        except Exception:
                            pass
                self._symbol_models[symbol] = {
                    # Real label spec of THIS symbol's model (R9 log fix)
                    "threshold": self._threshold,
                    "horizon": self._horizon,
                    "models": copy.deepcopy(self._models),
                    "scaler": copy.deepcopy(self._scaler),
                    "feature_names": list(self._feature_names),
                    "cv_scores": dict(accuracies),
                    "fitted_weights": dict(fitted_weights),
                    "feature_importances": importances,
                    "big_move_rate": float(positive_rate),
                    "version": self._model_version,
                }

            self._save_models()
            self._spec_mismatch = False
            log.info(
                "vol_mag_trained",
                symbol=symbol,
                accuracies=accuracies,
                fitted_weights=fitted_weights,
                big_move_rate=f"{positive_rate:.2%}",
                threshold=self._threshold,
                horizon=self._horizon,
            )

        return accuracies

    @property
    def fitted_weights(self) -> "dict[str, float] | None":
        return getattr(self, "_fitted_weights", None)

    def _walk_forward_split(self, n: int, n_splits: int = 4, gap: int | None = None):
        # Deep-audit ML-F1: labels look self._horizon (30) days forward and
        # OVERLAP; gap=5 leaked ~25 days of label window into every
        # validation fold, inflating the CV accuracy that gates whether
        # long_straddle trades at all. Purge gap must cover the horizon
        # (same bug-class as the fixed range-model item 2.4).
        if gap is None:
            gap = max(5, int(self._horizon))
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

    def _train_xgb(self, X: np.ndarray, y: np.ndarray, scale_pos_weight: float = 1.0) -> float:
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import balanced_accuracy_score
            model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                verbosity=0, n_jobs=-1, random_state=42,
                scale_pos_weight=scale_pos_weight,
            )
            scores = []
            for tr_idx, val_idx in self._walk_forward_split(len(X)):
                if len(np.unique(y[tr_idx])) < 2:
                    continue
                fold_scaler = StandardScaler()
                X_tr = pd.DataFrame(
                    fold_scaler.fit_transform(X[tr_idx]), columns=self._feature_names,
                )
                X_val = pd.DataFrame(
                    fold_scaler.transform(X[val_idx]), columns=self._feature_names,
                )
                model.fit(X_tr, y[tr_idx])
                preds = model.predict(X_val)
                scores.append(balanced_accuracy_score(y[val_idx], preds))
            self._models["xgboost"] = model
            avg = float(np.mean(scores)) if scores else 0.0
            log.info("vol_mag_xgb_trained", cv_balanced_acc=f"{avg:.3f}", folds=len(scores))
            return avg
        except ImportError:
            return 0.0

    def _train_lgb(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            from lightgbm import LGBMClassifier
            from sklearn.metrics import balanced_accuracy_score
            model = LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary", metric="binary_logloss",
                verbose=-1, n_jobs=-1, class_weight="balanced",
                random_state=42, deterministic=True,
            )
            scores = []
            for tr_idx, val_idx in self._walk_forward_split(len(X)):
                if len(np.unique(y[tr_idx])) < 2:
                    continue
                fold_scaler = StandardScaler()
                X_tr = pd.DataFrame(
                    fold_scaler.fit_transform(X[tr_idx]), columns=self._feature_names,
                )
                X_val = pd.DataFrame(
                    fold_scaler.transform(X[val_idx]), columns=self._feature_names,
                )
                model.fit(X_tr, y[tr_idx])
                preds = model.predict(X_val)
                scores.append(balanced_accuracy_score(y[val_idx], preds))
            self._models["lightgbm"] = model
            avg = float(np.mean(scores)) if scores else 0.0
            log.info("vol_mag_lgb_trained", cv_balanced_acc=f"{avg:.3f}", folds=len(scores))
            return avg
        except ImportError:
            return 0.0

    MIN_EDGE_OVER_BASELINE = 0.10

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        market_context: dict | None = None,
        live_signals: dict | None = None,
    ) -> VolMagnitudePrediction | None:
        """Predict P(big move > threshold over horizon_days)."""
        sym_data = None
        if symbol and symbol in self._symbol_models:
            sym_data = self._symbol_models[symbol]
            models = sym_data["models"]
            scaler = sym_data["scaler"]
            feature_names = sym_data["feature_names"]
        elif self._trained:
            models = self._models
            scaler = self._scaler
            feature_names = self._feature_names
            self._warn_symbol_fallback(symbol)
        else:
            return None

        # Edge-over-baseline check using BALANCED ACCURACY (baselines at 0.50)
        if sym_data is not None:
            cv_scores = sym_data.get("cv_scores", {})
            if cv_scores:
                avg_balanced_acc = sum(cv_scores.values()) / len(cv_scores)
                edge = avg_balanced_acc - 0.50
                if edge < self.MIN_EDGE_OVER_BASELINE:
                    log.debug("vol_mag_no_edge", symbol=symbol,
                              balanced_accuracy=f"{avg_balanced_acc:.2f}",
                              edge=f"{edge:+.2f}")
                    return None

        features = self._feature_engine.compute(
            df, market_context=market_context, live_signals=live_signals,
        )
        if features.empty:
            return None

        # reindex+fill (not direct indexing): an artifact trained before the
        # R12-C feature retirement lists sentiment/flow columns FeatureEngine
        # no longer produces. Constant in training -> 0.0 fill is exact for
        # the tree ensemble (same pattern as ensemble.predict).
        X = features.reindex(columns=feature_names).fillna(0.0).iloc[[-1]]
        try:
            X_scaled = pd.DataFrame(
                scaler.transform(X.values), columns=feature_names,
            )
        except Exception as e:
            log.error("vol_mag_scaling_failed", symbol=symbol, error=str(e))
            return None

        fw = (sym_data or {}).get("fitted_weights") or getattr(self, "_fitted_weights", None)
        weighted_p = 0.0
        total_weight = 0.0
        for name, model in models.items():
            w = fw.get(name, 0.5) if fw else self._weights.get(name, 0.5)
            try:
                p = float(model.predict_proba(X_scaled)[0][1])
                weighted_p += w * p
                total_weight += w
            except Exception:
                continue

        if total_weight == 0:
            return None
        p_big_move = weighted_p / total_weight

        # Report the REAL spec of the model that produced this probability —
        # per-symbol models may carry their own threshold/horizon (R9 log fix).
        if sym_data is not None:
            _thr = float(sym_data.get("threshold", self._threshold))
            _hor = int(sym_data.get("horizon", self._horizon))
        else:
            _thr, _hor = float(self._threshold), int(self._horizon)
        return VolMagnitudePrediction(
            probability_big_move=p_big_move,
            threshold_pct=_thr,
            horizon_days=_hor,
            confidence=max(p_big_move, 1 - p_big_move),
            features_used=len(feature_names),
            model_version=self._model_version,
        )

    def _warn_symbol_fallback(self, symbol: str) -> None:
        """R7 honesty: symbol without its own model served by the global
        fallback (trained on the last symbol trained). Warn once per symbol."""
        if not symbol or symbol in self._symbol_models or symbol in self._fallback_warned:
            return
        self._fallback_warned.add(symbol)
        log.warning(
            "vol_mag_model_symbol_fallback",
            symbol=symbol,
            fallback_trained_on=self._global_model_symbol or "unknown",
            available_symbol_models=sorted(self._symbol_models),
        )

    def _save_models(self, model_dir: "Path | str | None" = None) -> None:
        """Save artifact to model_dir (default: the directory this instance was
        constructed with — live models/ unless a research dir was passed)."""
        from datetime import datetime
        target_dir = Path(model_dir) if model_dir is not None else self._model_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "vol_magnitude.pkl"
        tmp = target_dir / "vol_magnitude.pkl.tmp"
        payload = {
            "models": self._models,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "version": self._model_version,
            "threshold": self._threshold,
            "horizon": self._horizon,
            "symbol_models": self._symbol_models,
            # R9: persisted freshness so needs_training survives restarts
            "trained_at": self._trained_at or datetime.now().isoformat(),
            "trained_symbol": self._global_model_symbol,
        }
        import os as _os
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        _os.replace(tmp, path)
        log.info("vol_mag_models_saved", path=str(path), version=self._model_version,
                 threshold=self._threshold, horizon=self._horizon)

    def load_models(self, model_dir: "Path | str | None" = None) -> bool:
        """Load artifact from model_dir (default: this instance's model dir).

        Spec guard (R7/R10): same contract as RangePredictor.load_models —
        on threshold/horizon mismatch vs the constructor spec, keep the
        pickle's REAL spec, log CRITICAL, and flag needs-retrain. Never
        silently flip the threshold on a model trained with a different one.
        """
        source_dir = Path(model_dir) if model_dir is not None else self._model_dir
        path = source_dir / "vol_magnitude.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            loaded_threshold = float(data.get("threshold", 0.07))
            loaded_horizon = int(data.get("horizon", 30))
            if (abs(loaded_threshold - self._spec_threshold) > 1e-9
                    or loaded_horizon != self._spec_horizon):
                self._spec_mismatch = True
                log.critical(
                    "vol_mag_model_spec_mismatch",
                    designed_threshold=self._spec_threshold,
                    designed_horizon_days=self._spec_horizon,
                    loaded_threshold=loaded_threshold,
                    loaded_horizon_days=loaded_horizon,
                    loaded_version=data.get("version", ""),
                    loaded_symbols=sorted(data.get("symbol_models", {})),
                    path=str(path),
                    action="serving artifact at its REAL spec; needs-retrain "
                           "flagged so the next training pass rebuilds at the "
                           "designed spec",
                )
            else:
                self._spec_mismatch = False

            self._models = data["models"]
            self._scaler = data["scaler"]
            self._feature_names = data["feature_names"]
            self._model_version = data["version"]
            self._threshold = loaded_threshold
            self._horizon = loaded_horizon
            self._symbol_models = data.get("symbol_models", {})
            self._trained_at = data.get("trained_at")
            self._global_model_symbol = data.get("trained_symbol", "")
            self._trained = bool(self._models)
            log.info("vol_mag_models_loaded", version=self._model_version,
                     threshold=self._threshold, horizon=self._horizon,
                     trained_at=self._trained_at, path=str(path))
            return True
        except Exception as e:
            log.warning("vol_mag_load_failed", error=str(e))
            return False

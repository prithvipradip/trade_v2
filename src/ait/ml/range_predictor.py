"""Range prediction model for iron condor strategy.

Iron condors profit when price stays inside a range. Direction-based
ML models (35-42% accuracy) are the wrong tool for this question.

This module trains a BINARY classifier:
    Y = 1 if max|return over next N days| < threshold (stays in range)
    Y = 0 otherwise (breaks out)

Output: P(stays_in_range) — directly usable as iron condor confidence.

Expected accuracy: 65-75% (vs 35-42% for direction prediction).
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

log = get_logger("ml.range_predictor")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

MODEL_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RangePrediction:
    """Output of range model."""

    probability_in_range: float       # 0.0–1.0
    threshold_pct: float              # the ±X% used (e.g., 0.05 = ±5%)
    horizon_days: int                 # the N days lookahead
    confidence: float                 # max(p, 1-p)
    features_used: int = 0
    model_version: str = ""


class RangePredictor:
    """Binary classifier: P(price stays within ±threshold over N days).

    Default: ±5% over 30 days. Matches typical iron-condor wing width and DTE.
    """

    def __init__(
        self,
        threshold_pct: float = 0.05,
        horizon_days: int = 30,
        ensemble_weights: dict[str, float] | None = None,
        enable_garch: bool = False,
        enable_msgarch: bool = False,
        enable_oujump: bool = False,
    ) -> None:
        self._threshold = threshold_pct
        self._horizon = horizon_days
        self._enable_garch = enable_garch
        self._enable_msgarch = enable_msgarch
        self._enable_oujump = enable_oujump
        # Equal prior weights updated by CV edge after training.
        # Caller-supplied ensemble_weights respected as-is for backward compat.
        if ensemble_weights is not None:
            self._weights = ensemble_weights
        else:
            _active = ["xgboost", "lightgbm"]
            if enable_garch:   _active.append("garch")
            if enable_msgarch: _active.append("msgarch")
            if enable_oujump:  _active.append("oujump")
            _w = 1.0 / len(_active)
            self._weights = {m: _w for m in _active}
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

    # --- Label creation ---

    def _create_labels_horizon(self, close: pd.Series, horizon: int) -> pd.Series:
        """Like _create_labels but uses an explicit horizon instead of self._horizon."""
        labels = pd.Series(np.nan, index=close.index, dtype=float)
        for t in range(len(close) - horizon):
            base = close.iloc[t]
            future = close.iloc[t + 1: t + 1 + horizon]
            max_dev = float((future / base - 1).abs().max())
            labels.iloc[t] = 1.0 if max_dev < self._threshold else 0.0
        return labels

    def _create_labels(self, close: pd.Series) -> pd.Series:
        """Label = 1 if the max abs return over the next N days < threshold.

        Mathematically: for each day t, look at closes[t+1..t+N], compute
        max |close[t+i] / close[t] - 1|. If that's below the threshold,
        the stock stayed in range → label 1.
        """
        labels = pd.Series(np.nan, index=close.index, dtype=float)
        for t in range(len(close) - self._horizon):
            base = close.iloc[t]
            future = close.iloc[t + 1: t + 1 + self._horizon]
            max_dev = float((future / base - 1).abs().max())
            labels.iloc[t] = 1.0 if max_dev < self._threshold else 0.0
        return labels

    # --- Training ---

    def train(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        market_context: dict | None = None,
        intraday_store=None,
    ) -> dict[str, float]:
        """Train range model on historical data."""
        features = self._feature_engine.compute(
            df, market_context=market_context,
            intraday_store=intraday_store, symbol=symbol,
        )
        if len(features) < 100:  # Need at least 100 rows for binary classification
            log.warning("range_insufficient_data", rows=len(features), required=100)
            return {}

        # Create binary labels
        features["target"] = self._create_labels(features["Close"])
        features = features.dropna(subset=["target"])

        # Class balance check — hard exit on single-class data before attempting fit.
        # sklearn/XGBoost raise cryptic errors when y has only one unique value;
        # catch it here with a clear diagnostic instead.
        positive_rate = features["target"].mean()
        n_pos = int(features["target"].sum())
        n_neg = len(features) - n_pos
        # Require at least 10 examples of each class — fewer than that and CV folds
        # will routinely produce single-class training splits, causing sklearn to crash.
        _MIN_CLASS_SAMPLES = 10
        if n_pos < _MIN_CLASS_SAMPLES or n_neg < _MIN_CLASS_SAMPLES:
            log.warning(
                "range_single_class", symbol=symbol,
                in_range_pct=f"{positive_rate:.1%}",
                n_in_range=n_pos, n_breakout=n_neg,
                min_required=_MIN_CLASS_SAMPLES,
                hint=f"±{self._threshold:.0%} threshold produces too few minority-class "
                     f"examples to train reliably — cannot train binary classifier",
            )
            return {}
        if positive_rate < 0.05 or positive_rate > 0.98:
            log.warning(
                "range_imbalanced", symbol=symbol,
                in_range_pct=f"{positive_rate:.1%}",
                n_in_range=n_pos, n_breakout=n_neg,
                hint=f"Threshold {self._threshold:.0%} may be too tight/loose",
            )

        self._feature_names = self._feature_engine.get_feature_names()
        self._feature_names = [f for f in self._feature_names if f in features.columns]

        X = features[self._feature_names].values
        y = features["target"].values.astype(int)

        accuracies = {}

        # scale_pos_weight balances XGBoost on imbalanced classes (LightGBM
        # already uses class_weight="balanced"). Value = n_negative / n_positive.
        _spw = n_neg / n_pos if n_pos > 0 else 1.0

        if "xgboost" in self._weights:
            acc = self._train_xgb(X, y, scale_pos_weight=_spw)
            accuracies["xgboost"] = acc

        if "lightgbm" in self._weights:
            acc = self._train_lgb(X, y)
            accuracies["lightgbm"] = acc

        if self._enable_garch and "garch" in self._weights:
            garch_acc = self._train_garch(features["Close"])
            # None = no valid CV folds (insufficient data) → omit from accuracies
            # 0.0–1.0 = valid AUROC score → include, even if 0.5 (no edge still
            # participates in equal-weight tie-breaking with XGB/LGB)
            if garch_acc is not None:
                accuracies["garch"] = garch_acc

        if self._enable_msgarch and "msgarch" in self._weights:
            msgarch_acc = self._train_msgarch(features["Close"])
            if msgarch_acc is not None:
                accuracies["msgarch"] = msgarch_acc

        if self._enable_oujump and "oujump" in self._weights:
            oujump_acc = self._train_oujump(features["Close"])
            if oujump_acc is not None:
                accuracies["oujump"] = oujump_acc

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
            self._model_version = f"range-v-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # Fit ensemble weights from per-model CV edge over 0.50 baseline.
            # Models with no skill get zero weight; proportional to edge otherwise.
            _baseline = 0.50
            _edges = {m: max(0.0, acc - _baseline) for m, acc in accuracies.items()}
            _total = sum(_edges.values())
            n_models = len(accuracies)
            fitted_weights = (
                {m: e / _total for m, e in _edges.items()}
                if _total > 0
                else {m: 1.0 / n_models for m in accuracies}
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
                _garch_state = getattr(self, "_garch_state", None)
                _ms_garch_state = getattr(self, "_ms_garch_state", None)
                _ou_jump_state = getattr(self, "_ou_jump_state", None)
                self._symbol_models[symbol] = {
                    "models": copy.deepcopy(self._models),
                    "scaler": copy.deepcopy(self._scaler),
                    "feature_names": list(self._feature_names),
                    "cv_scores": dict(accuracies),
                    "fitted_weights": dict(fitted_weights),
                    "feature_importances": importances,
                    "in_range_rate": float(positive_rate),
                    "version": self._model_version,
                    # GARCH ensemble member metadata
                    "garch_state":   _garch_state,
                    "garch_variant": (_garch_state or {}).get("selected_variant"),
                    "garch_dist":    (_garch_state or {}).get("selected_dist"),
                    "garch_dist_bic": (_garch_state or {}).get("selected_bic"),
                    "garch_jb_pvalue": (_garch_state or {}).get("jb_pvalue"),
                    "garch_resid_skewness": (_garch_state or {}).get("resid_skewness"),
                    "garch_fallback": (_garch_state or {}).get("fallback_used"),
                    "garch_stable_attempted": (_garch_state or {}).get("garch_stable_attempted"),
                    "garch_stable_converged": (_garch_state or {}).get("garch_stable_converged"),
                    "garch_all_variants": (_garch_state or {}).get("garch_all_variants", {}),
                    # MS-GARCH ensemble member metadata
                    "ms_garch_state":     _ms_garch_state,
                    "ms_garch_converged": (_ms_garch_state or {}).get("converged"),
                    "ms_garch_bic":       (_ms_garch_state or {}).get("bic"),
                    "ms_garch_regime0":   (_ms_garch_state or {}).get("msgarch_state", {}).get("regime0"),
                    "ms_garch_regime1":   (_ms_garch_state or {}).get("msgarch_state", {}).get("regime1"),
                    "ms_garch_transitions": (_ms_garch_state or {}).get("msgarch_state", {}).get("transition"),
                    # OU-Kou-GARCH ensemble member metadata
                    "ou_jump_state":      _ou_jump_state,
                    "ou_jump_converged":  (_ou_jump_state or {}).get("converged"),
                    "ou_jump_bic":        (_ou_jump_state or {}).get("bic"),
                    "ou_jump_direction":  (_ou_jump_state or {}).get("ou_jump_direction"),
                    "ou_jump_confidence": (_ou_jump_state or {}).get("ou_jump_confidence"),
                    "ou_jump_params":     (_ou_jump_state or {}).get("oujump_state", {}).get("params"),
                }

            self._save_models()
            log.info(
                "range_trained",
                symbol=symbol,
                accuracies=accuracies,
                fitted_weights=fitted_weights,
                in_range_rate=f"{positive_rate:.2%}",
                threshold=self._threshold,
                horizon=self._horizon,
            )

        return accuracies

    def _train_garch(self, close: pd.Series) -> "float | None":
        """CV-evaluate GARCH range model. Returns mean AUROC, or None if no valid folds.

        Returns None (not 0.0) when all CV folds fail — caller omits GARCH from
        accuracies dict entirely, producing nan in window JSON (honest: unevaluable).
        Returns 0.0–1.0 for valid AUROC; 0.5 = no edge but still participates in
        equal-weight tie-breaking.

        Mirrors _train_xgb/_train_lgb pattern but operates on the Close price
        series directly (GARCH uses returns, not the feature matrix).
        Stores the full-training-data GARCH state in self._garch_state.
        """
        try:
            from ait.ml.garch_range_predictor import GARCHRangeModel
        except ImportError:
            log.warning("garch_not_available", hint="pip install arch>=7.2")
            return 0.0

        garch = GARCHRangeModel()
        splits = self._walk_forward_split(len(close))

        # CV scored at a SHORT horizon (5d) so P(in range) varies widely across
        # validation days (±5% over 5d: quiet ~0.95, shock day ~0.30 → 3× spread
        # vs 21d: quiet ~0.85, shock ~0.50 → too narrow for AUROC to discriminate).
        # The full-horizon fit below still uses self._horizon (21d) for OOS prediction.
        #
        # Threshold scaled to the CV horizon via sqrt-of-time: a 5% threshold over
        # 21 days corresponds to ~2.4% over 5 days. Using the full 21d threshold
        # with a 5d horizon makes almost every day "in range" → all-positive folds
        # → AUROC undefined (P36). Scale by sqrt(cv_h / full_h) to match.
        _CV_HORIZON = 5
        _cv_threshold = self._threshold * np.sqrt(_CV_HORIZON / max(self._horizon, 1))
        _cv_labels_fn = lambda c: self._create_labels_horizon(c, _CV_HORIZON)
        acc = garch.cv_score(
            close=close,
            horizon_days=_CV_HORIZON,
            threshold_pct=_cv_threshold,
            splits=splits,
            create_labels_fn=_cv_labels_fn,
        )

        # Fit on full training data — stored for predict().
        # Strip non-serialisable objects (arch result, vol kwargs) before storing —
        # the state dict ends up in _symbol_models which gets serialised to window JSON.
        try:
            raw_state = garch.fit(close, self._horizon, self._threshold)
            self._garch_state: dict = {
                k: v for k, v in raw_state.items()
                if k not in ("arch_result", "cts_params")
                # _vol_kwargs is a plain serialisable dict — kept so CV rolling
                # refit uses the same variant spec as the BIC-winning fit.
            }
            if raw_state.get("cts_params") is not None:
                self._garch_state["cts_params"] = raw_state["cts_params"].tolist()
        except Exception as e:
            log.warning("garch_full_fit_failed", error=str(e))
            self._garch_state = {}

        log.info(
            "range_garch_trained",
            cv_auroc="None" if acc is None else f"{acc:.3f}",
            cv_edge="None" if acc is None else f"{acc - 0.5:+.3f}",
            variant=self._garch_state.get("selected_variant"),
            dist=self._garch_state.get("selected_dist"),
            fallback=self._garch_state.get("fallback_used"),
        )
        return acc

    def _train_msgarch(self, close: pd.Series) -> "float | None":
        """CV-evaluate MS-GARCH range model. Returns mean AUROC, or None if no valid folds.

        Same contract as _train_garch(): None means unevaluable (omit from accuracies),
        0.0–1.0 means valid AUROC.  Stores full-training-data MS-GARCH state in
        self._ms_garch_state for use in predict() and window JSON serialisation.
        """
        try:
            from ait.ml.garch_range_predictor import GARCHRangeModel
        except ImportError:
            log.warning("msgarch_not_available")
            return None

        garch = GARCHRangeModel()
        splits = self._walk_forward_split(len(close))

        _CV_HORIZON = 5
        _cv_threshold = self._threshold * np.sqrt(_CV_HORIZON / max(self._horizon, 1))
        _cv_labels_fn = lambda c: self._create_labels_horizon(c, _CV_HORIZON)
        acc = garch.cv_score_msgarch(
            close=close,
            horizon_days=_CV_HORIZON,
            threshold_pct=_cv_threshold,
            splits=splits,
            create_labels_fn=_cv_labels_fn,
        )

        # Fit on full training data via _fit_msgarch (BIC-comparable result dict).
        try:
            import numpy as _np
            returns = _np.diff(_np.log(close.dropna().values))
            ms_result = garch._fit_msgarch(returns, self._horizon, self._threshold)
            # Strip the live object — store only the serialisable state dict.
            ms_result.pop("_msgarch_obj", None)
            ms_result.pop("arch_result", None)
            self._ms_garch_state: dict = ms_result
        except Exception as e:
            log.warning("msgarch_full_fit_failed", error=str(e))
            self._ms_garch_state = {}

        log.info(
            "range_msgarch_trained",
            cv_auroc="None" if acc is None else f"{acc:.3f}",
            cv_edge="None" if acc is None else f"{acc - 0.5:+.3f}",
            converged=self._ms_garch_state.get("converged"),
            bic=self._ms_garch_state.get("bic"),
        )
        return acc

    def _train_oujump(self, close: pd.Series) -> "float | None":
        """CV-evaluate OU-Kou-GARCH model. Returns mean AUROC, or None if no valid folds.

        Same contract as _train_garch() / _train_msgarch():
          None   → unevaluable (omit from accuracies; nan in window JSON)
          float  → valid AUROC in [0, 1]; even 0.5 participates in weighting

        Stores full-training-data state in self._ou_jump_state.
        """
        try:
            from ait.ml.garch_range_predictor import GARCHRangeModel
        except ImportError:
            log.warning("oujump_not_available")
            return None

        garch = GARCHRangeModel()
        splits = self._walk_forward_split(len(close))

        _CV_HORIZON = 5
        _cv_threshold = self._threshold * np.sqrt(_CV_HORIZON / max(self._horizon, 1))
        _cv_labels_fn = lambda c: self._create_labels_horizon(c, _CV_HORIZON)
        acc = garch.cv_score_oujump(
            close=close,
            horizon_days=_CV_HORIZON,
            threshold_pct=_cv_threshold,
            splits=splits,
            create_labels_fn=_cv_labels_fn,
        )

        # Fit on full training data via _fit_oujump (returns BIC-comparable dict)
        try:
            import numpy as _np
            returns = _np.diff(_np.log(close.dropna().values))
            ou_result = garch._fit_oujump(returns, self._horizon, self._threshold)
            ou_result.pop("_oujump_obj", None)
            ou_result.pop("arch_result", None)
            self._ou_jump_state: dict = ou_result
        except Exception as e:
            log.warning("oujump_full_fit_failed", error=str(e))
            self._ou_jump_state = {}

        log.info(
            "range_oujump_trained",
            cv_auroc="None" if acc is None else f"{acc:.3f}",
            cv_edge="None" if acc is None else f"{acc - 0.5:+.3f}",
            converged=self._ou_jump_state.get("converged"),
            bic=self._ou_jump_state.get("bic"),
            direction=self._ou_jump_state.get("ou_jump_direction"),
        )
        return acc

    @property
    def fitted_weights(self) -> "dict[str, float] | None":
        return getattr(self, "_fitted_weights", None)

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

    def _train_xgb(self, X: np.ndarray, y: np.ndarray, scale_pos_weight: float = 1.0) -> float:
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import balanced_accuracy_score
            model = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                verbosity=0, n_jobs=-1,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
            )
            scores = []
            for tr_idx, val_idx in self._walk_forward_split(len(X)):
                if len(np.unique(y[tr_idx])) < 2:
                    continue  # skip fold — training split is single-class
                fold_scaler = StandardScaler()
                X_tr = pd.DataFrame(
                    fold_scaler.fit_transform(X[tr_idx]), columns=self._feature_names,
                )
                X_val = pd.DataFrame(
                    fold_scaler.transform(X[val_idx]), columns=self._feature_names,
                )
                model.fit(X_tr, y[tr_idx])
                # Balanced accuracy: fair for class imbalance, baselines at 0.50
                preds = model.predict(X_val)
                scores.append(balanced_accuracy_score(y[val_idx], preds))
            self._models["xgboost"] = model
            avg = float(np.mean(scores)) if scores else 0.0
            log.info("range_xgb_trained", cv_balanced_acc=f"{avg:.3f}", folds=len(scores))
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
                    continue  # skip fold — training split is single-class
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
            log.info("range_lgb_trained", cv_balanced_acc=f"{avg:.3f}", folds=len(scores))
            return avg
        except ImportError:
            return 0.0

    # --- Prediction ---

    # Minimum predictive edge (accuracy − base_rate) required to use the model.
    # Below this, the model is just classifying based on majority class — no
    # actual prediction skill. Returning None forces the strategy to skip.
    MIN_EDGE_OVER_BASELINE = 0.10

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        market_context: dict | None = None,
        live_signals: dict | None = None,
        intraday_store=None,
    ) -> RangePrediction | None:
        """Predict P(stays in ±threshold% over horizon_days)."""
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
        else:
            return None

        # Edge-over-baseline check — skip predictions when model has no skill.
        # cv_scores stores BALANCED ACCURACY (sensitivity+specificity)/2 which
        # baselines at 0.50 for ANY class distribution. Cleaner than raw accuracy.
        if sym_data is not None:
            cv_scores = sym_data.get("cv_scores", {})
            if cv_scores:
                avg_balanced_acc = sum(cv_scores.values()) / len(cv_scores)
                edge = avg_balanced_acc - 0.50  # balanced acc baseline = 50%
                if edge < self.MIN_EDGE_OVER_BASELINE:
                    log.debug("range_no_edge", symbol=symbol,
                              balanced_accuracy=f"{avg_balanced_acc:.2f}",
                              edge=f"{edge:+.2f}")
                    return None

        features = self._feature_engine.compute(
            df, market_context=market_context, live_signals=live_signals,
            intraday_store=intraday_store, symbol=symbol,
        )
        if features.empty:
            return None

        X = features[feature_names].iloc[[-1]]
        try:
            X_scaled = pd.DataFrame(
                scaler.transform(X.values), columns=feature_names,
            )
        except Exception as e:
            log.error("range_scaling_failed", symbol=symbol, error=str(e))
            return None

        # Weighted ensemble of binary probabilities — prefer fitted weights when available.
        fw = (sym_data or {}).get("fitted_weights") or getattr(self, "_fitted_weights", None)
        weighted_p = 0.0
        total_weight = 0.0
        for name, model in models.items():
            w = fw.get(name, 0.5) if fw else self._weights.get(name, 0.5)
            try:
                p = float(model.predict_proba(X_scaled)[0][1])  # P(class=1)
                weighted_p += w * p
                total_weight += w
            except Exception:
                continue

        # GARCH ensemble contribution
        garch_state = (sym_data or {}).get("garch_state") or getattr(self, "_garch_state", None)
        if garch_state:
            w_garch = fw.get("garch", self._weights.get("garch", 0.0)) if fw else self._weights.get("garch", 0.0)
            if w_garch > 0:
                try:
                    from ait.ml.garch_range_predictor import GARCHRangeModel
                    p_garch = GARCHRangeModel().predict_p_in_range(garch_state)
                    weighted_p += w_garch * p_garch
                    total_weight += w_garch
                except Exception:
                    pass

        # MS-GARCH ensemble contribution
        ms_garch_state = (sym_data or {}).get("ms_garch_state") or getattr(self, "_ms_garch_state", None)
        if ms_garch_state:
            w_ms = fw.get("msgarch", self._weights.get("msgarch", 0.0)) if fw else self._weights.get("msgarch", 0.0)
            if w_ms > 0:
                try:
                    from ait.ml.garch_range_predictor import GARCHRangeModel
                    p_ms = GARCHRangeModel().predict_p_in_range(ms_garch_state)
                    weighted_p += w_ms * p_ms
                    total_weight += w_ms
                except Exception:
                    pass

        # OU-Kou-GARCH ensemble contribution
        ou_jump_state = (sym_data or {}).get("ou_jump_state") or getattr(self, "_ou_jump_state", None)
        if ou_jump_state:
            w_ou = fw.get("oujump", self._weights.get("oujump", 0.0)) if fw else self._weights.get("oujump", 0.0)
            if w_ou > 0:
                try:
                    from ait.ml.garch_range_predictor import GARCHRangeModel
                    p_ou = GARCHRangeModel().predict_p_in_range(ou_jump_state)
                    weighted_p += w_ou * p_ou
                    total_weight += w_ou
                except Exception:
                    pass

        if total_weight == 0:
            return None
        p_in_range = weighted_p / total_weight

        return RangePrediction(
            probability_in_range=p_in_range,
            threshold_pct=self._threshold,
            horizon_days=self._horizon,
            confidence=max(p_in_range, 1 - p_in_range),
            features_used=len(feature_names),
            model_version=self._model_version,
        )

    # --- Persistence ---

    def _save_models(self) -> None:
        path = MODEL_DIR / "range.pkl"
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
        log.info("range_models_saved", path=str(path), version=self._model_version)

    def load_models(self) -> bool:
        path = MODEL_DIR / "range.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._models = data["models"]
            self._scaler = data["scaler"]
            self._feature_names = data["feature_names"]
            self._model_version = data["version"]
            self._threshold = data.get("threshold", 0.05)
            self._horizon = data.get("horizon", 30)
            self._symbol_models = data.get("symbol_models", {})
            self._trained = bool(self._models)
            log.info("range_models_loaded", version=self._model_version)
            return True
        except Exception as e:
            log.warning("range_load_failed", error=str(e))
            return False

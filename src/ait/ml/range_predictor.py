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

# Research sandbox for backtest/walkforward artifacts. R7/R10: a walkforward
# run saved its adaptive-threshold model over models/range.pkl and the live
# bot's range gate silently ran at the research spec (8.67%/21d instead of
# the designed 5%/30d). Research callers MUST pass this as model_dir so they
# can never clobber the live artifact.
RESEARCH_MODEL_DIR = MODEL_DIR / "research"


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

    # R12-C: the GARCH/MS-GARCH/OU-Kou-GARCH ensemble members were retired to
    # deprecated/research/ (2026-07-13). The enable_garch/enable_msgarch/
    # enable_oujump constructor flags are gone — they were hard-False in
    # production and the parametric members never earned CV weight.

    def __init__(
        self,
        threshold_pct: float = 0.05,
        horizon_days: int = 30,
        ensemble_weights: dict[str, float] | None = None,
        min_edge_over_baseline: float = 0.05,
        model_dir: "Path | str | None" = None,
    ) -> None:
        self._threshold = threshold_pct
        self._horizon = horizon_days
        # Artifact directory — live default is models/. Research callers
        # (walkforward/backtest/optimizer) MUST pass RESEARCH_MODEL_DIR so a
        # research run can never overwrite the live artifact (R7/R10).
        self._model_dir = Path(model_dir) if model_dir is not None else MODEL_DIR
        # Designed spec, immutable — the reference for the load-time spec
        # guard. self._threshold/_horizon may be overwritten by a loaded
        # pickle (kept honest and flagged); training always rebuilds at the
        # designed spec.
        self._spec_threshold = float(threshold_pct)
        self._spec_horizon = int(horizon_days)
        self._spec_mismatch = False
        self._trained_at: "str | None" = None
        self._global_model_symbol: str = ""
        self._fallback_warned: set[str] = set()
        # Equal prior weights updated by CV edge after training.
        # Caller-supplied ensemble_weights respected as-is for backward compat.
        if ensemble_weights is not None:
            self._weights = ensemble_weights
        else:
            self._weights = {"xgboost": 0.5, "lightgbm": 0.5}
        self._models: dict = {}
        self._symbol_models: dict[str, dict] = {}
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._feature_engine = FeatureEngine()
        self._trained = False
        self._model_version = ""
        self._min_edge_over_baseline = min_edge_over_baseline

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def spec_mismatch(self) -> bool:
        """True when the loaded artifact's threshold/horizon differ from the
        constructor's designed spec (see load_models). ModelTrainer.needs_training
        treats this as a forced-retrain signal."""
        return self._spec_mismatch

    @property
    def trained_at(self) -> "str | None":
        """ISO timestamp of the last training run, persisted in the artifact
        (R9: lets needs_training recover freshness across restarts)."""
        return self._trained_at

    # --- Label creation ---

    def _create_labels_horizon(
        self, close: pd.Series, horizon: int, threshold: float | None = None,
    ) -> pd.Series:
        """Like _create_labels but uses an explicit horizon instead of self._horizon."""
        threshold_used = self._threshold if threshold is None else float(threshold)
        labels = pd.Series(np.nan, index=close.index, dtype=float)
        for t in range(len(close) - horizon):
            base = close.iloc[t]
            future = close.iloc[t + 1: t + 1 + horizon]
            max_dev = float((future / base - 1).abs().max())
            labels.iloc[t] = 1.0 if max_dev < threshold_used else 0.0
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
        # Always train at the DESIGNED constructor spec. A previously loaded
        # pickle may have overwritten self._threshold/_horizon with a foreign
        # spec (load_models spec guard keeps the pickle's values so served
        # probabilities stay honest); retraining must restore the design so
        # the rebuilt model answers the question this instance was built for.
        self._threshold = self._spec_threshold
        self._horizon = self._spec_horizon
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

        # Deep-audit ML-F5: omitting include_vlmc dropped all 26 intraday
        # VLMC columns that compute() had just built — wasted O(n^2) compute
        # and unrealised feature intent (ensemble does this correctly).
        self._feature_names = self._feature_engine.get_feature_names(
            include_vlmc=intraday_store is not None
        )
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

        # R12-C: GARCH/MS-GARCH/OU-Jump training arms removed — the parametric
        # members live in deprecated/research/garch_range_predictor.py et al.

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
            self._trained_at = datetime.now().isoformat()
            if symbol:
                # The instance-level (global fallback) models now belong to the
                # LAST symbol trained — recorded for fallback honesty (R7: TLT
                # was silently served by a model trained on XLE).
                self._global_model_symbol = symbol

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
                self._symbol_models[symbol] = {
                    "fitted_at": __import__("datetime").date.today().isoformat(),  # A7
                    # Real label spec of THIS symbol's model — predictions must
                    # report these, not the instance defaults (R9 log fix).
                    "threshold": self._threshold,
                    "horizon": self._horizon,
                    "models": copy.deepcopy(self._models),
                    "scaler": copy.deepcopy(self._scaler),
                    "feature_names": list(self._feature_names),
                    "cv_scores": dict(accuracies),
                    "fitted_weights": dict(fitted_weights),
                    "feature_importances": importances,
                    "in_range_rate": float(positive_rate),
                    "version": self._model_version,
                    # R12-C: garch_state / ms_garch_state / ou_jump_state
                    # metadata keys no longer emitted (GARCH family retired
                    # to deprecated/research/).
                }

            self._save_models()
            # A fresh train at the designed spec supersedes any previously
            # flagged artifact mismatch.
            self._spec_mismatch = False
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

    @property
    def fitted_weights(self) -> "dict[str, float] | None":
        return getattr(self, "_fitted_weights", None)

    def _walk_forward_split(self, n: int, n_splits: int = 4, gap: int | None = None):
        # LEAKAGE FIX (audit 2026-07-07 item 2.4): labels look `horizon` days
        # forward and OVERLAP, so a gap smaller than the horizon lets ~
        # (horizon - gap) days of each training fold's label window bleed
        # into validation — inflating CV accuracy, which feeds the
        # edge-vs-threshold gate that decides whether iron condors trade at
        # all. The purge gap must be >= the label horizon.
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

    def predict(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        market_context: dict | None = None,
        live_signals: dict | None = None,
        intraday_store=None,
        min_edge_override: float | None = None,
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
            self._warn_symbol_fallback(symbol)
        else:
            return None

        # Edge-over-baseline check — skip predictions when the ensemble has no skill.
        # Uses fitted weights (which zero out anti-predictive models) so a bad
        # ensemble member can't veto a good one. Falls back to simple average when
        # no fitted weights are available.
        # Threshold: min_edge_override (from Backtester/Optuna) → instance default.
        if sym_data is not None:
            cv_scores = sym_data.get("cv_scores", {})
            if cv_scores:
                fw = sym_data.get("fitted_weights") or {}
                if fw:
                    # Weighted edge: anti-predictive models get weight=0 from CV calibration
                    edge = sum(fw.get(m, 0.0) * (s - 0.50) for m, s in cv_scores.items())
                else:
                    edge = sum(cv_scores.values()) / len(cv_scores) - 0.50
                _threshold = min_edge_override if min_edge_override is not None else self._min_edge_over_baseline
                if edge < _threshold:
                    log.debug("range_no_edge", symbol=symbol,
                              weighted_edge=f"{edge:+.3f}",
                              threshold=f"{_threshold:.3f}")
                    return None

        features = self._feature_engine.compute(
            df, market_context=market_context, live_signals=live_signals,
            intraday_store=intraday_store, symbol=symbol,
        )
        if features.empty:
            return None

        # reindex+fill (not direct indexing): an artifact trained before the
        # R12-C feature retirement lists columns FeatureEngine no longer
        # produces (sentiment/flow). Those were constant in training, so tree
        # models never split on them — 0.0 fill is exact. Same pattern as
        # ensemble.predict and predict_from_features below.
        X = features.reindex(columns=feature_names).fillna(0.0).iloc[[-1]]
        try:
            X_scaled = pd.DataFrame(
                scaler.transform(X.values), columns=feature_names,
            )
        except Exception as e:
            log.error("range_scaling_failed", symbol=symbol, error=str(e))
            return None

        # Weighted ensemble of binary probabilities — prefer fitted weights when available.
        fw = (sym_data or {}).get("fitted_weights") or getattr(self, "_fitted_weights", None)
        def _weight(name: str, default: float) -> float:
            if fw and name in fw:
                return float(fw[name])
            return float(self._weights.get(name, default))

        weighted_p = 0.0
        total_weight = 0.0
        for name, model in models.items():
            w = _weight(name, 0.5)
            try:
                p = float(model.predict_proba(X_scaled)[0][1])  # P(class=1)
                weighted_p += w * p
                total_weight += w
            except Exception:
                continue

        # R12-C: parametric ensemble contributions (GARCH/MS-GARCH/OU-Jump
        # states) removed — the family is retired to deprecated/research/.
        # Artifacts trained before the retirement may still carry *_state
        # keys in sym_data; they are simply ignored (their weights were
        # CV-fitted to ~0 in production anyway).

        if total_weight == 0:
            return None
        p_in_range = weighted_p / total_weight

        _thr, _hor = self._model_spec(sym_data)
        return RangePrediction(
            probability_in_range=p_in_range,
            threshold_pct=_thr,
            horizon_days=_hor,
            confidence=max(p_in_range, 1 - p_in_range),
            features_used=len(feature_names),
            model_version=self._model_version,
        )

    def _model_spec(self, sym_data: "dict | None") -> "tuple[float, int]":
        """Real label spec of the model that produced a prediction.

        R9 log fix: per-symbol models may have been trained at a different
        threshold/horizon than the instance defaults (e.g. walkforward's
        adaptive thresholds). Predictions must report the spec they were
        actually trained on, never the constructor's aspiration.
        """
        if sym_data is not None:
            return (
                float(sym_data.get("threshold", self._threshold)),
                int(sym_data.get("horizon", self._horizon)),
            )
        return float(self._threshold), int(self._horizon)

    def _warn_symbol_fallback(self, symbol: str) -> None:
        """R7 honesty: a symbol without its own model is served by the global
        fallback (trained on whatever symbol trained last, e.g. TLT->XLE).
        Warn once per symbol per process."""
        if not symbol or symbol in self._symbol_models or symbol in self._fallback_warned:
            return
        self._fallback_warned.add(symbol)
        log.warning(
            "range_model_symbol_fallback",
            symbol=symbol,
            fallback_trained_on=self._global_model_symbol or "unknown",
            available_symbol_models=sorted(self._symbol_models),
        )

    def predict_from_features(
        self,
        feature_row: "pd.Series",
        symbol: str = "",
    ) -> "RangePrediction | None":
        """Make a prediction from a pre-computed feature row, bypassing FeatureEngine.

        Used by _save_window_timeseries for O(1) per-bar predictions instead of
        re-running FeatureEngine on 252+ rows for each bar.
        """
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

        if sym_data is not None:
            cv_scores = sym_data.get("cv_scores", {})
            if cv_scores:
                fw = sym_data.get("fitted_weights") or {}
                if fw:
                    edge = sum(fw.get(m, 0.0) * (s - 0.50) for m, s in cv_scores.items())
                else:
                    edge = sum(cv_scores.values()) / len(cv_scores) - 0.50
                if edge < self._min_edge_over_baseline:
                    return None

        try:
            X = pd.DataFrame(
                [feature_row.reindex(feature_names).fillna(0.0).values],
                columns=feature_names,
            )
            X_scaled = pd.DataFrame(
                scaler.transform(X.values), columns=feature_names,
            )
        except Exception as e:
            log.error("range_scaling_failed", symbol=symbol, error=str(e))
            return None

        fw = (sym_data or {}).get("fitted_weights") or getattr(self, "_fitted_weights", None)
        def _weight(name: str, default: float) -> float:
            if fw and name in fw:
                return float(fw[name])
            return float(self._weights.get(name, default))

        weighted_p = 0.0
        total_weight = 0.0
        for name, model in models.items():
            w = _weight(name, 0.5)
            try:
                p = float(model.predict_proba(X_scaled)[0][1])
                weighted_p += w * p
                total_weight += w
            except Exception:
                continue

        # R12-C: parametric ensemble contributions removed (see predict()).

        if total_weight == 0:
            return None
        p_in_range = weighted_p / total_weight

        _thr, _hor = self._model_spec(sym_data)
        return RangePrediction(
            probability_in_range=p_in_range,
            threshold_pct=_thr,
            horizon_days=_hor,
            confidence=max(p_in_range, 1 - p_in_range),
            features_used=len(feature_names),
            model_version=self._model_version,
        )

    # --- Persistence ---

    def _save_models(self, model_dir: "Path | str | None" = None) -> None:
        """Save artifact to model_dir (default: the directory this instance was
        constructed with — live models/ unless a research dir was passed)."""
        from datetime import datetime
        target_dir = Path(model_dir) if model_dir is not None else self._model_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "range.pkl"
        tmp = target_dir / "range.pkl.tmp"
        payload = {
            "models": self._models,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "version": self._model_version,
            "threshold": self._threshold,
            "horizon": self._horizon,
            "symbol_models": self._symbol_models,
            # R9: persisted so ModelTrainer.needs_training can recover model
            # freshness across restarts instead of retraining on every boot.
            "trained_at": self._trained_at or datetime.now().isoformat(),
            # R7: which symbol the instance-level fallback models were fit on.
            "trained_symbol": self._global_model_symbol,
        }
        # Atomic write — a concurrent reader (live bot) can never observe a
        # half-written pickle.
        import os as _os
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        _os.replace(tmp, path)
        log.info("range_models_saved", path=str(path), version=self._model_version,
                 threshold=self._threshold, horizon=self._horizon)

    def load_models(self, model_dir: "Path | str | None" = None) -> bool:
        """Load artifact from model_dir (default: this instance's model dir).

        Spec guard (R7/R10): if the pickle's threshold/horizon differ from the
        constructor's designed spec, the artifact was written by something else
        (e.g. a walkforward's adaptive-threshold model hijacking the live
        pickle). We do NOT silently flip the threshold — the model's
        probabilities only answer the question it was trained on — and we do
        NOT silently serve the designed spec either. The pickle's REAL spec is
        kept (predictions report it honestly), a CRITICAL is logged with both
        specs, and self.spec_mismatch is raised so ModelTrainer.needs_training
        forces a proper retrain at the designed spec.
        """
        source_dir = Path(model_dir) if model_dir is not None else self._model_dir
        path = source_dir / "range.pkl"
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            loaded_threshold = float(data.get("threshold", 0.05))
            loaded_horizon = int(data.get("horizon", 30))
            if (abs(loaded_threshold - self._spec_threshold) > 1e-9
                    or loaded_horizon != self._spec_horizon):
                self._spec_mismatch = True
                log.critical(
                    "range_model_spec_mismatch",
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
            log.info("range_models_loaded", version=self._model_version,
                     threshold=self._threshold, horizon=self._horizon,
                     trained_at=self._trained_at, path=str(path))
            return True
        except Exception as e:
            log.warning("range_load_failed", error=str(e))
            return False

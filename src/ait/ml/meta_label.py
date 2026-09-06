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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ait.utils.logging import get_logger

log = get_logger("ml.meta_label")

# R16: MODEL_DIR was the CWD-relative Path("models") — the same clobber class
# range_predictor.py:37 already fixed by anchoring to the repo. Any script run
# from a different CWD created/read a DIFFERENT models/ tree (and the
# constructor's mkdir silently manufactured it).
MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


def _model_path() -> Path:
    """Resolved at CALL time, never cached at import: tests/conftest.py's
    autouse fence monkeypatches MODEL_DIR, and a module-level constant would
    silently escape it (that fence exists because a pytest run clobbered the
    live artifact once already — R15 #7)."""
    return MODEL_DIR / "meta_label.pkl"


# R16 QUARANTINE, 2026-07-18. models/meta_label.pkl was clobbered by a test run
# and moved aside to meta_label.pkl.clobbered_20260718; nothing has rewritten
# it since, because the daily retrain child (orchestration/master.py
# retrain_models) trains ensemble/range/vol_magnitude ONLY. The gate is inert
# today (config.yaml meta_label.enabled=false -> orchestrator._meta_labeler is
# None, and the gate additionally requires live mode), so this is NOT
# silently re-armed here. What changes: the missing/stale artifact is now LOUD
# and the arming path refuses to trust a degraded rebuild.
#
# REGENERATION PATH (do all three before flipping meta_label.enabled=true):
#   1. Confirm >= MIN_TRADES_FOR_TRAINING closed trades with trade_context
#      rows carrying the FULL META_FEATURES set. build_training_data() below
#      currently supplies only 9 of the 20 (sentiment_score is permanently 0
#      after R12-C retired the sentiment feed) — training on that is exactly
#      the corrupted-input condition the artifact was quarantined for.
#   2. Train explicitly and inspect the stats:
#        python -c "import sys; sys.path.insert(0,'src'); \
#          from ait.ml.meta_label import MetaLabeler; \
#          from ait.execution.state import StateManager; \
#          m=MetaLabeler(); print(m.train(m.build_training_data(StateManager())))"
#      An empty dict means the guards below refused to arm — read the logged
#      reason (meta_label_arm_refused) rather than forcing it.
#   3. Add the trainer to retrain_models() so the artifact cannot silently
#      go stale again, then delete meta_label.pkl.clobbered_20260718.
# Alternative, equally acceptable: DELETE the meta-labeler. It is dead config
# in an iron-condor-only book.
MIN_FEATURE_COVERAGE = 15   # of the 20 META_FEATURES, else refuse to arm
MAX_ARTIFACT_AGE_DAYS = 45  # loaded artifact older than this = stale, refuse

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

        # R16 ARM GUARD: the quarantined 07-18 artifact was trained on a
        # degraded feature set and rejected everything. build_training_data()
        # supplies only 9 of the 20 META_FEATURES today (sentiment_score is
        # permanently 0 after R12-C), so an unattended retrain would recreate
        # exactly that condition — silently, since the caller only checks that
        # load_model() failed. Refuse to arm below the coverage floor, and
        # count a column that is CONSTANT across every row as absent (a
        # retired feed reads as "present" but carries no information).
        live_cols = [
            f for f in available
            if trades_df[f].notna().any() and trades_df[f].nunique(dropna=True) > 1
        ]
        if len(live_cols) < MIN_FEATURE_COVERAGE:
            log.error(
                "meta_label_arm_refused",
                reason="insufficient_feature_coverage",
                informative_features=len(live_cols),
                present_features=len(available),
                required=MIN_FEATURE_COVERAGE,
                expected=len(META_FEATURES),
                dead_or_constant=[f for f in available if f not in live_cols],
                impact="NOT arming the meta-label gate — this is the "
                       "corrupted-training condition meta_label.pkl was "
                       "quarantined for on 2026-07-18. See the REGENERATION "
                       "PATH comment in src/ait/ml/meta_label.py.",
            )
            self._trained = False
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
        # Deep-audit ML-F2: fitting the scaler on ALL rows before CV leaks
        # each fold's mean/variance into training — the reported accuracy/
        # precision of this live trade-veto gate were optimistically biased.
        # CV below scales per fold; the production scaler fits on the full
        # data only AFTER CV (matching ensemble/range/vol_mag).
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
                verbosity=0,
                n_jobs=-1,
            )

            # Cross-validate — scaler fit PER FOLD on the train slice only.
            from sklearn.preprocessing import StandardScaler as _SS
            accuracies = []
            precisions = []
            for train_idx, val_idx in tscv.split(X):
                _fold_scaler = _SS()
                X_tr = _fold_scaler.fit_transform(X[train_idx])
                X_va = _fold_scaler.transform(X[val_idx])
                model.fit(X_tr, y[train_idx])
                preds = model.predict(X_va)
                acc = float(np.mean(preds == y[val_idx]))
                accuracies.append(acc)

                # Precision for "take trade" predictions
                take_mask = preds == 1
                if take_mask.sum() > 0:
                    precision = float(np.mean(y[val_idx][take_mask] == 1))
                    precisions.append(precision)

            # Production scaler + final model fit on ALL data (after CV)
            self._scaler.fit(X)
            X_scaled = self._scaler.transform(X)
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

    def build_training_data_from_backtest(
        self,
        trades: list[dict],
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build MetaLabeler training data from walk-forward backtest trade outcomes.

        For each trade's entry date, looks up the corresponding FeatureEngine row
        and merges with signal-level context stored in the trade dict (entry_confidence,
        entry_regime, entry_iv_rank, entry_vix_level).  Provides all 20 META_FEATURES
        — solving the "9/20 features hardcoded to 0" problem that corrupted live training.

        Args:
            trades: Closed trade dicts from Backtester.run().  Each must have
                'entry_date' (ISO date string) and 'pnl'.  Optional keys:
                'entry_confidence', 'entry_regime', 'entry_iv_rank', 'entry_vix_level'.
            features_df: FeatureEngine DataFrame indexed by DatetimeIndex (or similar
                date-like index).  Must contain the full set of base feature columns.

        Returns:
            DataFrame with META_FEATURES columns plus 'profitable' (1/0).
        """
        if not trades or features_df.empty:
            return pd.DataFrame()

        # Normalise features_df index to plain date objects for O(1) lookup
        try:
            feat_by_date: dict = {}
            for idx, row in features_df.iterrows():
                d = idx.date() if hasattr(idx, "date") else idx
                feat_by_date[d] = row
        except Exception:
            return pd.DataFrame()

        # Helper: derive one-hot regime from stored entry_regime string or features
        def _regime_flags(entry_regime: str, feat_row: "pd.Series | None") -> tuple:
            if entry_regime:
                return (
                    1.0 if entry_regime == "trending_up"   else 0.0,
                    1.0 if entry_regime == "trending_down"  else 0.0,
                    1.0 if entry_regime == "high_volatility" else 0.0,
                    1.0 if entry_regime == "range_bound"    else 0.0,
                )
            if feat_row is not None:
                vol_exp = float(feat_row.get("vol_regime_expanding", 0.0)) > 0.5
                px_sma  = float(feat_row.get("price_vs_sma_20", 0.0))
                if vol_exp:
                    if px_sma > 0.02:  return (1.0, 0.0, 0.0, 0.0)
                    if px_sma < -0.02: return (0.0, 1.0, 0.0, 0.0)
                    return (0.0, 0.0, 1.0, 0.0)
                return (0.0, 0.0, 0.0, 1.0)
            return (0.0, 0.0, 0.0, 1.0)  # default: range_bound

        rows = []
        for trade in trades:
            pnl = trade.get("pnl", None)
            if pnl is None:
                continue

            entry_date_raw = trade.get("entry_date")
            if not entry_date_raw:
                continue
            try:
                from datetime import date as _date
                entry_d = (
                    entry_date_raw
                    if isinstance(entry_date_raw, _date)
                    else _date.fromisoformat(str(entry_date_raw)[:10])
                )
            except (ValueError, TypeError):
                continue

            feat = feat_by_date.get(entry_d)
            entry_regime = trade.get("entry_regime", "")
            reg_tu, reg_td, reg_hv, reg_rb = _regime_flags(entry_regime, feat)

            def _fv(key: str, default: float = 0.0) -> float:
                if feat is not None:
                    v = feat.get(key, default)
                    try:
                        return float(v) if not (v != v) else default  # NaN guard
                    except (TypeError, ValueError):
                        return default
                return default

            rows.append({
                "primary_confidence":   trade.get("entry_confidence", 0.60),
                "regime_trending_up":   reg_tu,
                "regime_trending_down": reg_td,
                "regime_high_vol":      reg_hv,
                "regime_range_bound":   reg_rb,
                "vix":                  trade.get("entry_vix_level", _fv("vix_level", 0.5)),
                "iv_rank":              trade.get("entry_iv_rank",   _fv("iv_rank",   0.5)),
                "sentiment_score":      0.0,  # not available in backtest
                "rsi_14":               _fv("rsi_14",           50.0),
                "rsi_7":                _fv("rsi_7",            50.0),
                "bb_position":          _fv("bb_position",       0.5),
                "volume_sma_20_ratio":  _fv("volume_sma_20_ratio", 1.0),
                "realized_vol_20":      _fv("realized_vol_20",  0.20),
                "atr_pct":              _fv("atr_pct",           0.01),
                "weekly_trend_aligned": _fv("weekly_trend_aligned", 0.5),
                "volume_confirmation":  _fv("volume_confirmation",  0.0),
                "hour_of_day":          10,  # backtest entries are effectively at open
                "macd_hist":            _fv("macd_hist",         0.0),
                "price_vs_sma_20":      _fv("price_vs_sma_20",   0.0),
                "sma_10_20_cross":      _fv("sma_10_20_cross",   0.5),
                "profitable":           1 if pnl > 0 else 0,
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def build_training_data(self, state_manager) -> pd.DataFrame:
        """Build training DataFrame from trade history.

        Joins trade_context (entry features) with trade outcomes (P&L).
        Only uses closed trades with context data.

        Args:
            state_manager: StateManager instance with trade + context tables.

        Returns:
            DataFrame ready for training.
        """
        import json as _json
        import sqlite3

        # W3 string-contracts-1/-4: this query used to be a bare
        # "WHERE t.status = 'closed'" while all 8 sibling consumers filtered
        # out non-real closes — so the trainer learned from $0 rows for
        # signals that NEVER FILLED (6 of 23 rows = 26% of the set, 43% of
        # the negative class on the 2026-08-25 book), teaching the entry gate
        # that those entry-feature contexts lose, and inflating the
        # MIN_TRADES_FOR_TRAINING arming count with fabricated labels. The
        # reconciler's $0 "needs manual review" sentinels are excluded by the
        # same authority.
        from ait.reporting.go_live import not_real_close_sql

        rows = []
        # Legacy rows with no captured snapshot (entry_signals == '{}') must
        # NOT be back-filled with the same fixed defaults for every row —
        # that fabricates artificial variance once even a single real
        # snapshot differs, letting the coverage guard in train() arm on
        # mostly-imputed data. Preserve these as NaN instead so
        # nunique(dropna=True) only reflects genuinely captured values.
        _nan = float("nan")

        def _sig(sig: dict, key: str, default):
            return sig.get(key, default) if sig else _nan

        with sqlite3.connect(state_manager._db_path) as conn:
            conn.row_factory = sqlite3.Row
            # Join trades with their entry context
            results = conn.execute(f"""
                SELECT t.trade_id, t.realized_pnl, t.entry_time,
                       c.entry_direction, c.entry_confidence, c.entry_regime,
                       c.entry_vix, c.entry_iv_rank, c.entry_sentiment_score,
                       c.entry_signals
                FROM trades t
                JOIN trade_context c ON t.trade_id = c.trade_id
                WHERE t.status = 'closed'
                      {not_real_close_sql("t.exit_reason_detailed")}
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

            # R20b review follow-up: entry_signals stores the 11 technical
            # META_FEATURES snapshotted at entry (_entry_signals_json). Parse
            # it so training uses the full 20-feature space instead of only
            # the 9 scalar context columns selected above — the coverage
            # guard (test_r20_research_validity) requires all META_FEATURES.
            try:
                sig = _json.loads(r.get("entry_signals") or "{}")
            except (TypeError, ValueError):
                sig = {}
            if not isinstance(sig, dict):
                sig = {}

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
                "rsi_14":               _sig(sig, "rsi_14", 50.0),
                "rsi_7":                _sig(sig, "rsi_7", 50.0),
                "bb_position":          _sig(sig, "bb_position", 0.5),
                "volume_sma_20_ratio":  _sig(sig, "volume_sma_20_ratio", 1.0),
                "realized_vol_20":      _sig(sig, "realized_vol_20", 0.20),
                "atr_pct":              _sig(sig, "atr_pct", 0.01),
                "weekly_trend_aligned": _sig(sig, "weekly_trend_aligned", 0.5),
                "volume_confirmation":  _sig(sig, "volume_confirmation", 0.0),
                "hour_of_day": entry_hour,
                "macd_hist":            _sig(sig, "macd_hist", 0.0),
                "price_vs_sma_20":      _sig(sig, "price_vs_sma_20", 0.0),
                "sma_10_20_cross":      _sig(sig, "sma_10_20_cross", 0.5),
                "profitable": 1 if r["realized_pnl"] > 0 else 0,
            })

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def load_model(self) -> bool:
        """Load previously trained meta-label model from disk.

        R16: a missing artifact used to return False in complete silence, so
        the 07-18 quarantine has been invisible for weeks and the caller
        (orchestrator.py:383) fell straight through to an unsupervised
        retrain. Missing and STALE are both loud now, and stale refuses to
        arm rather than gating live entries on a months-old model.
        """
        model_file = _model_path()
        if not model_file.exists():
            quarantined = sorted(MODEL_DIR.glob("meta_label.pkl.clobbered_*"))
            log.error(
                "meta_label_artifact_missing",
                path=str(model_file),
                quarantined=[p.name for p in quarantined] or None,
                impact="meta-label gate CANNOT arm; caller must not retrain "
                       "it blind — see the REGENERATION PATH comment at the "
                       "top of src/ait/ml/meta_label.py",
            )
            return False

        try:
            age_days = (
                datetime.now() - datetime.fromtimestamp(model_file.stat().st_mtime)
            ).days
            if age_days > MAX_ARTIFACT_AGE_DAYS:
                log.error(
                    "meta_label_artifact_stale",
                    path=str(model_file),
                    age_days=age_days,
                    max_age_days=MAX_ARTIFACT_AGE_DAYS,
                    impact="refusing to arm the meta-label gate on a stale "
                           "artifact — retrain it deliberately",
                )
                return False
        except OSError as e:  # noqa: BLE001 — stat failure must not arm blind
            log.error("meta_label_artifact_stat_failed",
                      path=str(model_file), error=str(e))
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

    def save_to_path(self, path: "str | Path") -> None:
        """Save trained model to an explicit path (for per-window walk-forward artifacts)."""
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "version": self._model_version,
            "training_stats": self._training_stats,
        }
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(dest, "wb") as f:
                pickle.dump(data, f)
            log.info("meta_label_saved_to_path", path=str(dest))
        except Exception as e:
            log.error("meta_label_save_to_path_failed", path=str(dest), error=str(e))

    def _save_model(self) -> None:
        """Save trained meta-label model to disk."""
        data = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "version": self._model_version,
            "training_stats": self._training_stats,
        }

        model_file = _model_path()
        try:
            with open(model_file, "wb") as f:
                pickle.dump(data, f)
            log.info("meta_label_saved", version=self._model_version)
        except Exception as e:
            log.error("meta_label_save_failed", error=str(e))

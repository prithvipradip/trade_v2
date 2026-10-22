"""W87a: the ONE owner of the tree-model settings the three gradient-boosted predictors share.

Before W87 the same five numbers (200 trees, depth 5, learning rate 0.05, subsample / colsample 0.8, seed 42)
were copied into ``ml/ensemble.py`` (direction, 3-class), ``ml/range_predictor.py`` and
``ml/vol_magnitude_predictor.py`` (binary), for XGBoost and LightGBM alike, so a deliberate change had to be
made in six places and could drift in one of them. They now come from here.

What is deliberately NOT here:

* ``ml/meta_label.py`` (100 trees, depth 3, no seed) and ``ml/trainer.py::_objective`` (``n_jobs=1``, no class
  weights, a search-time proxy) differ ON PURPOSE; folding them into the base would change those models.
* ``scripts/excursion_range_study.py::LGB_PARAMS`` is a pre-registered research setting (W83a amendment 6).

Every function returns a FRESH dict of keyword arguments (the models are imported lazily by their callers), and
the result is identical to the literal dict each call site used before: ``tests/test_w87a_hygiene.py`` pins
that key for key. The direction ensemble's ``dict.update(self._xgb_kwargs)`` (tuned hyperparameters) still
happens AFTER these defaults, at the call site.
"""
from __future__ import annotations

# The five numbers (+ seed) all three predictors share, for XGBoost and LightGBM.
TREE_BASE: dict = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# Quiet, all cores. LightGBM additionally balances the classes and is deterministic.
XGB_COMMON: dict = {"verbosity": 0, "n_jobs": -1}
LGBM_COMMON: dict = {"verbose": -1, "n_jobs": -1, "class_weight": "balanced", "deterministic": True}

# Binary predictors (range, vol-magnitude): the share of rows that must be the positive class for the
# label to be usable, and the minimum rows of the minority class. Warning-only guards, formerly copied.
POSITIVE_RATE_MIN: float = 0.05
POSITIVE_RATE_MAX: float = 0.98
MIN_CLASS_SAMPLES: int = 10


def xgb_multiclass_kwargs() -> dict:
    """Direction ensemble: 3 classes (bearish / neutral / bullish)."""
    return {**TREE_BASE, "objective": "multi:softprob", "num_class": 3,
            "eval_metric": "mlogloss", **XGB_COMMON}


def lgbm_multiclass_kwargs() -> dict:
    return {**TREE_BASE, "objective": "multiclass", "num_class": 3,
            "metric": "multi_logloss", **LGBM_COMMON}


def xgb_binary_kwargs(scale_pos_weight: float) -> dict:
    """Range / vol-magnitude predictors: P(label)."""
    return {**TREE_BASE, "objective": "binary:logistic", "eval_metric": "logloss",
            **XGB_COMMON, "scale_pos_weight": scale_pos_weight}


def lgbm_binary_kwargs() -> dict:
    return {**TREE_BASE, "objective": "binary", "metric": "binary_logloss", **LGBM_COMMON}

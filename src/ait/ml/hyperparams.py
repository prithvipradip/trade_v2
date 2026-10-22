"""W87a/W88c: the ONE owner of how the gradient-boosted predictors build their XGBoost / LightGBM models.

W87a: the same five numbers (200 trees, depth 5, learning rate 0.05, subsample / colsample 0.8, seed 42) were copied
into ``ml/ensemble.py`` (direction, 3-class), ``ml/range_predictor.py`` and ``ml/vol_magnitude_predictor.py``
(binary), for XGBoost and LightGBM alike. They now come from here.

W88c: the numbers themselves are settings. ``ml.hyperparams`` (direction ensemble, range, vol-magnitude) and
``ml.meta_label_hyperparams`` (the meta-labeler's shallower profile) are ``TreeHyperparamsConfig`` models in
``config/settings.py``; research may differ through the research overlay or a study, the parity manifest shows it,
and every saved model records the values it was trained with (``model_spec``). Callers pass their config's
``TreeHyperparamsConfig``; ``None`` resolves the loaded settings (the frozen research view inside a research
process), else the schema defaults.

What is deliberately NOT a setting here: ``n_jobs`` / ``verbosity`` (speed and noise, no effect on the model), each
task's objective and metric (they define the task), LightGBM's ``class_weight="balanced"`` / ``deterministic``,
and ``ml/trainer.py::_objective`` (a search-time proxy with ``n_jobs=1``).

Every builder returns a FRESH dict of keyword arguments (the models are imported lazily by their callers). The direction
ensemble's ``dict.update(self._xgb_kwargs)`` (tuned hyperparameters) still happens AFTER these defaults, at the call
site.
"""
from __future__ import annotations

from ait.config.settings import TreeHyperparamsConfig

# Compatibility alias: the schema DEFAULT of ``ml.hyperparams`` as a plain dict (what the literals used to be).
TREE_BASE: dict = TreeHyperparamsConfig().model_dump()

# Quiet, all cores. LightGBM additionally balances the classes and is deterministic.
XGB_COMMON: dict = {"verbosity": 0, "n_jobs": -1}
LGBM_COMMON: dict = {"verbose": -1, "n_jobs": -1, "class_weight": "balanced", "deterministic": True}

# Binary predictors (range, vol-magnitude): the share of rows that must be the positive class for the
# label to be usable, and the minimum rows of the minority class. Warning-only guards, formerly copied.
POSITIVE_RATE_MIN: float = 0.05
POSITIVE_RATE_MAX: float = 0.98
MIN_CLASS_SAMPLES: int = 10


def default_hyperparams() -> TreeHyperparamsConfig:
    """``ml.hyperparams`` from the loaded settings (the frozen research view in a research process), else the
    schema default."""
    try:
        from ait.config.settings import load_settings
        return load_settings().ml.hyperparams
    except Exception:  # noqa: BLE001 - no config.yaml -> the schema defaults
        return TreeHyperparamsConfig()


def default_meta_label_hyperparams() -> TreeHyperparamsConfig:
    """``ml.meta_label_hyperparams`` from the loaded settings, else the schema default."""
    try:
        from ait.config.settings import load_settings
        return load_settings().ml.meta_label_hyperparams
    except Exception:  # noqa: BLE001
        from ait.config.settings import MLConfig
        return MLConfig().meta_label_hyperparams


def _base(hp: TreeHyperparamsConfig | None) -> dict:
    hp = hp if hp is not None else default_hyperparams()
    base = {"n_estimators": hp.n_estimators, "max_depth": hp.max_depth, "learning_rate": hp.learning_rate,
            "subsample": hp.subsample, "colsample_bytree": hp.colsample_bytree}
    if hp.random_state is not None:                       # null = unseeded: the key is left out entirely
        base["random_state"] = hp.random_state
    return base


def xgb_multiclass_kwargs(hp: TreeHyperparamsConfig | None = None) -> dict:
    """Direction ensemble: 3 classes (bearish / neutral / bullish)."""
    return {**_base(hp), "objective": "multi:softprob", "num_class": 3,
            "eval_metric": "mlogloss", **XGB_COMMON}


def lgbm_multiclass_kwargs(hp: TreeHyperparamsConfig | None = None) -> dict:
    return {**_base(hp), "objective": "multiclass", "num_class": 3,
            "metric": "multi_logloss", **LGBM_COMMON}


def xgb_binary_kwargs(scale_pos_weight: float, hp: TreeHyperparamsConfig | None = None) -> dict:
    """Range / vol-magnitude predictors: P(label)."""
    return {**_base(hp), "objective": "binary:logistic", "eval_metric": "logloss",
            **XGB_COMMON, "scale_pos_weight": scale_pos_weight}


def lgbm_binary_kwargs(hp: TreeHyperparamsConfig | None = None) -> dict:
    return {**_base(hp), "objective": "binary", "metric": "binary_logloss", **LGBM_COMMON}


def meta_label_xgb_kwargs(scale_pos_weight: float, hp: TreeHyperparamsConfig | None = None) -> dict:
    """The meta-labeler: its own (shallower, unseeded) profile, ``ml.meta_label_hyperparams``."""
    hp = hp if hp is not None else default_meta_label_hyperparams()
    return {**_base(hp), "scale_pos_weight": scale_pos_weight, "objective": "binary:logistic",
            "eval_metric": "logloss", "verbosity": 0, "n_jobs": -1}


# --- what a saved model records about how it was trained ------------------------------------------------------

def model_spec(hp: TreeHyperparamsConfig, feature_spec_hash: str | None = None) -> dict:
    """The record stored in a model artifact: the hyperparameters it was trained with (and, from W88d, the hash of
    the feature spec). Plain JSON-able dict so a pickle written today loads under tomorrow's classes."""
    return {"hyperparams": hp.model_dump(), "feature_spec_hash": feature_spec_hash}


def spec_differences(stored: dict | None, current: dict) -> list[str]:
    """Human-readable differences between an artifact's recorded spec and the current one. ``None`` (an artifact
    saved before W88c) is not a difference: legacy artifacts are always accepted."""
    if not stored:
        return []
    from ait.ml.feature_spec import default_spec_hash
    out = []
    for key, now in current.get("hyperparams", {}).items():
        was = (stored.get("hyperparams") or {}).get(key)
        if was != now:
            out.append(f"{key}: model trained with {was!r}, config now {now!r}")
    was_h, now_h = stored.get("feature_spec_hash") or default_spec_hash(), current.get("feature_spec_hash")
    if now_h is not None and was_h != now_h:       # a recorded-hash-less model is assumed to carry the default hash
        out.append(f"feature_spec_hash: model {was_h!r}, config now {now_h!r}")
    return out


def warn_on_spec_mismatch(artifact: str, stored: dict | None, current: dict) -> bool:
    """Log (never refuse) when a loaded model was trained under other hyperparameters than the current config.
    Returns True when they differ. A hyperparameter mismatch is informational: the model is simply the product of
    different settings. (A FEATURE-spec mismatch is different and is refused in W88d.)"""
    diffs = spec_differences(stored, current)
    if diffs:
        from ait.utils.logging import get_logger
        get_logger("ml.hyperparams").warning("model_spec_mismatch", artifact=artifact, differences=diffs)
    return bool(diffs)


def feature_spec_refused(artifact: str, stored: dict | None, current_hash: str) -> bool:
    """True (after logging CRITICAL) when a saved model was trained under other FEATURE WINDOWS than the current ones.

    Unlike a hyperparameter difference this is never served: the model's input columns are named after the windows
    (``rsi_14``), and ``predict`` zero-fills columns it cannot find, so a stale model would predict on silent zeros.
    Callers return False from their ``load_*`` (the normal needs-training path retrains). A model with no recorded
    hash predates W88d and was trained under the DEFAULT windows, so it is compared as such."""
    from ait.ml.feature_spec import default_spec_hash
    was = (stored or {}).get("feature_spec_hash") or default_spec_hash()
    if was == current_hash:
        return False
    from ait.utils.logging import get_logger
    get_logger("ml.hyperparams").critical(
        "model_feature_spec_mismatch", artifact=artifact, model_feature_spec_hash=was, current_feature_spec_hash=current_hash,
        action="REFUSING to serve a model trained under other feature windows; it will be retrained under the "
               "current ones (ml.features)")
    return True

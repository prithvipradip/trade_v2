"""Parameter search spaces for strategies and ML models.

Each space is a dict mapping param_name → tuple:
  (type, low, high)              for int/float
  (type, low, high, kwargs)      for float with log=True etc.

These are consumed by StrategyOptimizer._suggest_params().
"""

from __future__ import annotations

IRON_CONDOR_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.55, 0.80),
    "stop_loss_pct":     ("float", 0.30, 0.70),
    "profit_target_pct": ("float", 0.40, 0.80),
    "trailing_stop_pct": ("float", 0.15, 0.40),
}

LONG_CALL_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.85),
    "stop_loss_pct":     ("float", 0.30, 0.60),
    "profit_target_pct": ("float", 0.60, 1.50),
}

LONG_PUT_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.85),
    "stop_loss_pct":     ("float", 0.30, 0.60),
    "profit_target_pct": ("float", 0.60, 1.50),
}

BULL_CALL_SPREAD_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.85),
    "stop_loss_pct":     ("float", 0.30, 0.65),
    "profit_target_pct": ("float", 0.50, 0.90),
}

BEAR_PUT_SPREAD_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.85),
    "stop_loss_pct":     ("float", 0.30, 0.65),
    "profit_target_pct": ("float", 0.50, 0.90),
}

PUT_CREDIT_SPREAD_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.55, 0.80),
    "stop_loss_pct":     ("float", 0.30, 0.70),
    "profit_target_pct": ("float", 0.40, 0.80),
}

XGBOOST_SPACE: dict[str, tuple] = {
    "n_estimators":      ("int",   50,   400),
    "learning_rate":     ("float", 0.01, 0.30, {"log": True}),
    "max_depth":         ("int",   3,    8),
    "min_child_weight":  ("int",   1,    10),
    "subsample":         ("float", 0.5,  1.0),
    "colsample_bytree":  ("float", 0.5,  1.0),
    "gamma":             ("float", 0.0,  5.0),
}

LIGHTGBM_SPACE: dict[str, tuple] = {
    "n_estimators":      ("int",   50,   400),
    "learning_rate":     ("float", 0.01, 0.30, {"log": True}),
    "num_leaves":        ("int",   20,   150),
    "min_child_samples": ("int",   5,    50),
    "subsample":         ("float", 0.5,  1.0),
    "colsample_bytree":  ("float", 0.5,  1.0),
}

# Keyed by strategy name as used in WalkForwardBacktester / Backtester
STRATEGY_SPACES: dict[str, dict[str, tuple]] = {
    "iron_condor":       IRON_CONDOR_SPACE,
    "long_call":         LONG_CALL_SPACE,
    "long_put":          LONG_PUT_SPACE,
    "bull_call_spread":  BULL_CALL_SPREAD_SPACE,
    "bear_put_spread":   BEAR_PUT_SPREAD_SPACE,
    "put_credit_spread": PUT_CREDIT_SPREAD_SPACE,
}

ML_SPACES: dict[str, dict[str, tuple]] = {
    "xgboost":  XGBOOST_SPACE,
    "lightgbm": LIGHTGBM_SPACE,
}

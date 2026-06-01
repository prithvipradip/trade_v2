"""Parameter search spaces for strategies and ML models.

Each space is a dict mapping param_name → tuple:
  (type, low, high)              for int/float
  (type, low, high, kwargs)      for float with log=True etc.

These are consumed by StrategyOptimizer._suggest_params().
"""

from __future__ import annotations

FRACTAL_GATE_SPACE: dict[str, tuple] = {
    "hurst_regime_threshold": ("float", 0.08, 0.30),
    "hurst_regime_penalty":   ("float", 0.0,  0.25),
    "multifractal_max_width": ("float", 0.30, 0.65),
}

# Per-leg options bid-ask spread model params (applied to all credit strategies)
SPREAD_MODEL_SPACE: dict[str, tuple] = {
    "spread_base":            ("float", 0.02, 0.08),
    "spread_iv_sensitivity":  ("float", 0.05, 0.20),
    "spread_dte_sensitivity": ("float", 0.001, 0.015),
}

IRON_CONDOR_SPACE: dict[str, tuple] = {
    # Excluded from search space (P8/P9):
    #   min_confidence, max_entry_vol_annual, iv_floor — regime gates; in-sample optima
    #   don't generalise OOS (P8: iv_floor creates train/OOS mismatch). Fixed in config.
    #   spread_* — calibrated from real market data; must stay fixed (removing prevents
    #   Optuna from fighting the calibration and recreating P8-style friction mismatch).
    "stop_loss_pct":          ("float", 0.30, 0.70),
    "profit_target_pct":      ("float", 0.30, 0.70),
    # trailing_stop_fraction: fraction of profit_target_pct — optimizer derives the actual
    # trailing_stop_pct as trailing_stop_fraction × profit_target_pct. Keeps trailing stop
    # coherent relative to target rather than as an independent dimension that can cancel it.
    "trailing_stop_fraction": ("float", 0.30, 0.90),
    "delta_short":            ("float", 0.15, 0.30),
    # [14, 40]: 60-day OOS windows give a 20-day end buffer (60-40=20), keeping
    # backtest_end exits rare while still allowing meaningful theta harvesting.
    "max_hold_days":          ("int",   14,   40),
    "wing_k":                 ("float", 0.30, 2.00),
    **FRACTAL_GATE_SPACE,
}

LONG_CALL_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.60),
    "profit_target_pct": ("float", 0.60, 1.50),
    "delta_long":        ("float", 0.25, 0.55),
    "max_hold_days":     ("int",   14,   60),
    "iv_floor":          ("float", 0.08, 0.25),
}

LONG_PUT_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.60),
    "profit_target_pct": ("float", 0.60, 1.50),
    "delta_long":        ("float", 0.25, 0.55),
    "max_hold_days":     ("int",   14,   60),
    "iv_floor":          ("float", 0.08, 0.25),
}

BULL_CALL_SPREAD_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.65),
    "profit_target_pct": ("float", 0.50, 0.90),
    "delta_long":        ("float", 0.30, 0.55),
    "max_hold_days":     ("int",   14,   60),
}

BEAR_PUT_SPREAD_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.65),
    "profit_target_pct": ("float", 0.50, 0.90),
    "delta_long":        ("float", 0.30, 0.55),
    "max_hold_days":     ("int",   14,   60),
}

SHORT_STRANGLE_SPACE: dict[str, tuple] = {
    "min_confidence":       ("float", 0.55, 0.70),
    "stop_loss_pct":        ("float", 0.30, 0.70),
    "profit_target_pct":    ("float", 0.30, 0.70),
    "trailing_stop_fraction": ("float", 0.30, 0.90),
    "delta_short":            ("float", 0.10, 0.25),
    "max_hold_days":          ("int",   14,   40),
    "iv_floor":             ("float", 0.15, 0.40),
    "delta_iv_scale":       ("float", 0.0,  1.0),
    "max_entry_vol_annual": ("float", 0.25, 0.90),
    **FRACTAL_GATE_SPACE,
}

LONG_STRANGLE_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.60),
    "profit_target_pct": ("float", 0.50, 2.00),
    "delta_long":        ("float", 0.10, 0.35),
    "max_hold_days":     ("int",   14,   45),
    "iv_floor":          ("float", 0.15, 0.40),
    "delta_iv_scale":    ("float", 0.0,  1.0),
}

PUT_CREDIT_SPREAD_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.55, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.70),
    "profit_target_pct": ("float", 0.30, 0.70),
    "delta_short":       ("float", 0.15, 0.30),
    "max_hold_days":     ("int",   14,   45),
    "iv_floor":          ("float", 0.08, 0.25),
    "wing_k":            ("float", 0.30, 2.00),
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
    "short_strangle":    SHORT_STRANGLE_SPACE,
    "long_strangle":     LONG_STRANGLE_SPACE,
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

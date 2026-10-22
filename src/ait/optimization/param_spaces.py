"""Parameter search spaces for strategies and ML models.

Each space is a dict mapping param_name → tuple:
  (type, low, high)              for int/float
  (type, low, high, kwargs)      for float with log=True etc.

These are consumed by StrategyOptimizer._suggest_params().
"""

from __future__ import annotations

from ait.config.invariants_research import RESEARCH_DELTA_MAX, RESEARCH_DELTA_MIN  # W88a

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
    # Exp 18 (H1 test): stop_loss_pct, profit_target_pct, trailing_stop_fraction frozen at
    #   ablation defaults (0.35, 0.50, 0.70). These are risk-management constants that
    #   don't vary by market regime — optimising them per window overfit train-path noise.
    #   Only the regime-specific params + fractal gate + iv_rank/edge gates remain in the
    #   search space (8 total — R20: the old "7 total" note miscounted; also note that
    #   until R20 the iv_rank_rise_threshold and min_edge_over_baseline dimensions never
    #   reached the trial Backtester at all, so the Exp 26/28 training-side searches over
    #   them were vacuous — see optimizer._run_backtest bt_kwargs invariant).
    "delta_short":            ("float", 0.15, 0.30),
    # IC credit trades close at DTE<=5, and entry_dte defaults to 14, so holds
    # can last ~9 calendar days before expiry_approaching takes over. Keep the
    # searched range inside that reachable window so this dimension can move
    # trial outcomes instead of becoming flat noise.
    "max_hold_days":          ("int",   1,    9),
    "wing_k":                 ("float", 0.30, 2.00),
    # Exp 26: iv_rank_rise gate threshold tunable per window. Fixed 0.30 was blocking
    # profitable W01/W02 entries (gradual IV drift pre-crash) once context_bars=252
    # made iv_rank meaningful. Range [0.25, 0.60]: 0.25 fires on sustained rises,
    # 0.60 only fires on genuine spikes (Yen-carry / tariff-shock level events).
    # Exp 27 tried widening to 0.70 + adding pct_from_60d_high gate — reverted in
    # Exp 28 (gate was too blunt, over-blocked W06/W07/W08).
    "iv_rank_rise_threshold": ("float", 0.25, 0.60),
    # Exp 28: min weighted CV edge required for range predictor to make predictions.
    # Fixes Bug 1 (engine now passes symbol → MIN_EDGE check actually fires) and
    # Bug 2 (uses fitted weights so anti-predictive members don't veto good ones).
    # Range [0.02, 0.15]: 0.02 = accept almost any skill, 0.15 = require strong edge.
    "min_edge_over_baseline": ("float", 0.02, 0.15),
    **FRACTAL_GATE_SPACE,
}

LONG_CALL_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.60),
    "profit_target_pct": ("float", 0.60, 1.50),
    "delta_long":        ("float", 0.25, 0.55),
    # PR#7 review 2026-08-25: entry_dte is 14 (options.dte_range[0]), so a
    # debit position EXPIRES at day ~14 — every searched value above 14 was
    # identical to hold-to-expiry (a flat/noise Optuna dimension). Bound the
    # range to the reachable horizon so the dimension can move outcomes.
    "max_hold_days":     ("int",   5,    14),
    "iv_floor":          ("float", 0.08, 0.25),
}

LONG_PUT_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.60),
    "profit_target_pct": ("float", 0.60, 1.50),
    "delta_long":        ("float", 0.25, 0.55),
    # PR#7 review 2026-08-25: entry_dte is 14 (options.dte_range[0]), so a
    # debit position EXPIRES at day ~14 — every searched value above 14 was
    # identical to hold-to-expiry (a flat/noise Optuna dimension). Bound the
    # range to the reachable horizon so the dimension can move outcomes.
    "max_hold_days":     ("int",   5,    14),
    "iv_floor":          ("float", 0.08, 0.25),
}

BULL_CALL_SPREAD_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.65),
    "profit_target_pct": ("float", 0.50, 0.90),
    "delta_long":        ("float", 0.30, 0.55),
    # PR#7 review 2026-08-25: entry_dte is 14 (options.dte_range[0]), so a
    # debit position EXPIRES at day ~14 — every searched value above 14 was
    # identical to hold-to-expiry (a flat/noise Optuna dimension). Bound the
    # range to the reachable horizon so the dimension can move outcomes.
    "max_hold_days":     ("int",   5,    14),
}

BEAR_PUT_SPREAD_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.60, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.65),
    "profit_target_pct": ("float", 0.50, 0.90),
    "delta_long":        ("float", 0.30, 0.55),
    # PR#7 review 2026-08-25: entry_dte is 14 (options.dte_range[0]), so a
    # debit position EXPIRES at day ~14 — every searched value above 14 was
    # identical to hold-to-expiry (a flat/noise Optuna dimension). Bound the
    # range to the reachable horizon so the dimension can move outcomes.
    "max_hold_days":     ("int",   5,    14),
}

SHORT_STRANGLE_SPACE: dict[str, tuple] = {
    "min_confidence":       ("float", 0.55, 0.70),
    "stop_loss_pct":        ("float", 0.30, 0.70),
    "profit_target_pct":    ("float", 0.30, 0.70),
    "trailing_stop_fraction": ("float", 0.30, 0.90),
    "delta_short":            ("float", 0.10, 0.25),
    # PR#7 review 2026-08-25: same reachability bound as IRON_CONDOR_SPACE
    # above — entry_dte is 14 (options.dte_range[0]) and credit structures
    # close at DTE<=5, so holds cap at ~9 calendar days; the old 14+ range
    # was entirely unreachable (a flat/noise Optuna dimension every trial).
    "max_hold_days":          ("int",   1,    9),
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
    # PR#7 review 2026-08-25: entry_dte is 14 (options.dte_range[0]), so a
    # debit position EXPIRES at day ~14 — every searched value above 14 was
    # identical to hold-to-expiry (a flat/noise Optuna dimension). Bound the
    # range to the reachable horizon so the dimension can move outcomes.
    "max_hold_days":     ("int",   5,    14),
    "iv_floor":          ("float", 0.15, 0.40),
    "delta_iv_scale":    ("float", 0.0,  1.0),
}

PUT_CREDIT_SPREAD_SPACE: dict[str, tuple] = {
    "min_confidence":    ("float", 0.55, 0.70),
    "stop_loss_pct":     ("float", 0.30, 0.70),
    "profit_target_pct": ("float", 0.30, 0.70),
    "delta_short":       ("float", 0.15, 0.30),
    # PR#7 review 2026-08-25: same reachability bound as IRON_CONDOR_SPACE
    # above — entry_dte is 14 (options.dte_range[0]) and credit structures
    # close at DTE<=5, so holds cap at ~9 calendar days; the old 14+ range
    # was entirely unreachable (a flat/noise Optuna dimension every trial).
    "max_hold_days":     ("int",   1,    9),
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
# W79a: the opt-in DTE dimensions (research only). Not part of IRON_CONDOR_SPACE on purpose: that
# dict feeds every other study and the reachability guard test. ``max_hold_days`` is replaced, not
# supplemented: with a hold fraction set the engine ignores the day cap.
DTE_SEARCH_REPLACES: tuple[str, ...] = ("max_hold_days",)


def dte_search_space(cfg) -> dict[str, tuple]:
    """Extra iron-condor dimensions for a DTE study, from ``BacktestConfig.dte_search``.

    ``cfg`` is a ``DteSearchConfig`` (or None). Returns ``{}`` unless it is enabled, so a
    caller can merge the result unconditionally."""
    if cfg is None or not getattr(cfg, "enabled", False):
        return {}
    space: dict[str, tuple] = {
        "entry_dte":         ("int", int(cfg.entry_dte[0]), int(cfg.entry_dte[1])),
        "expiry_guard_dte":  ("int", int(cfg.guard_dte[0]), int(cfg.guard_dte[1])),
        "max_hold_fraction": ("float", float(cfg.hold_fraction[0]), float(cfg.hold_fraction[1])),
    }
    if getattr(cfg, "search_flatten", False):
        space["flatten_minutes_before_close"] = (
            "int", int(cfg.flatten_minutes[0]), int(cfg.flatten_minutes[1]))
    return space


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


# ---------------------------------------------------------------------------
# W87a: per-study OVERRIDES of the search ranges, validated, and a hash of the effective space
# ---------------------------------------------------------------------------
# The defaults above stay the code-owned, git-versioned, PLAN-recorded spaces. A study (see
# ``studies/*.yaml``, ``search_space:``) may narrow or widen individual dimensions; this layer validates the
# override and returns a NEW dict, so ``STRATEGY_SPACES`` is never mutated (``FRACTAL_GATE_SPACE`` is spread
# into two spaces at import, so mutating it would not even propagate). The optimizer reads the effective
# space at every point it used to read ``STRATEGY_SPACES``.

# Hard limits an OVERRIDE must stay inside (the defaults are not checked against these). Where a limit is the
# engine's own (the 0.05-0.45 delta clamp) the engine silently clamps anything outside, so a wider range
# would search values the backtest cannot honour.
HARD_BOUNDS: dict[str, tuple[float, float]] = {
    "delta_short": (RESEARCH_DELTA_MIN, RESEARCH_DELTA_MAX), "delta_long": (RESEARCH_DELTA_MIN, RESEARCH_DELTA_MAX),
    "wing_k": (0.1, 5.0), "max_hold_days": (1, 45), "max_hold_fraction": (0.0, 1.0),
    "min_confidence": (0.0, 1.0), "stop_loss_pct": (0.01, 1.0), "profit_target_pct": (0.05, 5.0),
    "trailing_stop_fraction": (0.0, 1.0), "iv_floor": (0.0, 1.0), "delta_iv_scale": (0.0, 1.0),
    "max_entry_vol_annual": (0.05, 5.0), "iv_rank_rise_threshold": (0.0, 10.0),
    "min_edge_over_baseline": (0.0, 1.0), "hurst_regime_threshold": (0.0, 1.0),
    "hurst_regime_penalty": (0.0, 1.0), "multifractal_max_width": (0.0, 2.0),
}
_DTE_DIMENSIONS = ("entry_dte", "expiry_guard_dte", "max_hold_fraction", "flatten_minutes_before_close")


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def effective_space(strategy: str, overrides: dict | None = None) -> dict[str, tuple]:
    """``STRATEGY_SPACES[strategy]`` with ``overrides`` ({dimension: [low, high]}) applied, as a NEW dict.

    Raises ``ValueError`` (naming the study key) for: an unknown strategy; a dimension that is not in the
    strategy's space (the DTE dimensions belong to ``backtest.dte_search``); a value that is not two numbers;
    a non-integral bound on an integer dimension; ``low >= high``; a bound outside ``HARD_BOUNDS``. The
    dimension's type and keyword arguments (e.g. ``log=True``) are kept from the default."""
    if strategy not in STRATEGY_SPACES:
        raise ValueError(f"search_space.{strategy}: unknown strategy (known: {sorted(STRATEGY_SPACES)})")
    base = STRATEGY_SPACES[strategy]
    out = dict(base)
    for name, new in (overrides or {}).items():
        where = f"search_space.{strategy}.{name}"
        if name not in base:
            hint = (" (the DTE dimensions are set with settings.backtest.dte_search)"
                    if name in _DTE_DIMENSIONS else "")
            raise ValueError(f"{where}: not a dimension of {strategy} (known: {sorted(base)}){hint}")
        if not (isinstance(new, (list, tuple)) and len(new) == 2 and all(_is_number(v) for v in new)):
            raise ValueError(f"{where}: expected [low, high] numbers, got {new!r}")
        lo, hi = new
        kind = base[name][0]
        if kind == "int":
            if lo != int(lo) or hi != int(hi):
                raise ValueError(f"{where}: integer dimension needs whole-number bounds, got {new!r}")
            lo, hi = int(lo), int(hi)
        else:
            lo, hi = float(lo), float(hi)
        if not lo < hi:
            raise ValueError(f"{where}: low must be below high, got {new!r}")
        limit = HARD_BOUNDS.get(name)
        if limit is not None and not (limit[0] <= lo and hi <= limit[1]):
            raise ValueError(f"{where}: {new!r} is outside the hard limits {list(limit)}")
        out[name] = (kind, lo, hi, *base[name][3:])
    return out


def override_warnings(strategy: str, overrides: dict | None, baselines: dict[str, float]) -> list[str]:
    """Overrides whose new range EXCLUDES the live / baseline value: the optimizer could then never choose
    the setting live trades with. Warnings, not errors (excluding it can be the point of a study)."""
    out = []
    for name, new in (overrides or {}).items():
        base = baselines.get(name)
        if base is not None and not (float(new[0]) <= float(base) <= float(new[1])):
            out.append(f"search_space.{strategy}.{name}: {list(new)} excludes the live value {base}")
    return out


def live_baselines(strategy: str = "iron_condor") -> dict[str, float]:
    """The values LIVE trades a searched dimension at, for ``override_warnings`` and the guard test that
    keeps them inside the default ranges. Only iron_condor has live baselines (the live selector is IC-only);
    each comes from its owner, not a copy: ``AIT_IC_WING_K`` through the env contract, the short delta from
    the constant ``iron_condor.py`` selects with."""
    if strategy != "iron_condor":
        return {}
    from ait.config.runtime_env import contract_float
    from ait.strategies.iron_condor import SHORT_DELTA_TARGET
    return {"wing_k": contract_float("AIT_IC_WING_K"), "delta_short": SHORT_DELTA_TARGET}


def space_hash(spaces: dict[str, dict]) -> str:
    """A short, stable sha256 of an effective space ({strategy: {dimension: spec}}): the same space gives
    the same hash across runs and machines; any change to a range, a type or a keyword changes it."""
    import hashlib
    import json
    canon = {s: {n: list(spec) for n, spec in sorted(dims.items())} for s, dims in sorted(spaces.items())}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, default=str).encode()).hexdigest()[:16]

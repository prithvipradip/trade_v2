"""Unit tests for OUKouGARCH + AdaptiveEKF (ou_jump.py).

Tests cover:
  1.  fit() converges on synthetic OU + GARCH returns
  2.  fit() raises ValueError on too-short input
  3.  forecast_sigma_h() is positive and finite for multiple horizons
  4.  p_in_range() is in [0, 1] and monotone in threshold
  5.  bic() is finite and within reasonable range
  6.  to_state_dict() is JSON-serialisable and has "diagnostics" key
  6b. diagnostics() values are all within expected ranges
  7.  AEKF state is valid: κ_T > 0, state_history shape correct
  8.  direction_signal() returns valid (direction, confidence)
  9.  GARCHRangeModel._fit_oujump() returns correct dict shape
  10. cv_score_oujump() runs without error, returns None or float
  11. GARCHRangeModel.fit() includes "OU-Kou-GARCH" in garch_all_variants
  12. RangePredictor(enable_oujump=True) trains; _symbol_models has ou_jump keys
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ait.ml.ou_jump import AdaptiveEKF, OUKouGARCH, OUKouGARCHParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_ou_returns(n: int = 200, seed: int = 42) -> np.ndarray:
    """Simulate OU + GARCH(1,1) log-returns for testing."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 1e-5, 0.05, 0.90
    kappa, mu_log = 0.15, 0.0   # moderate mean reversion
    log_prices = np.empty(n + 1)
    log_prices[0] = mu_log
    sigma2 = omega / (1.0 - alpha - beta)
    returns = np.empty(n)
    for t in range(n):
        if t > 0:
            sigma2 = max(omega + alpha * returns[t - 1] ** 2 + beta * sigma2, 1e-10)
        drift = kappa * (mu_log - log_prices[t]) / 252.0
        returns[t] = drift + np.sqrt(sigma2) * rng.standard_normal()
        log_prices[t + 1] = log_prices[t] + returns[t]
    return returns


def _fitted_model() -> OUKouGARCH:
    model = OUKouGARCH()
    model.fit(_synthetic_ou_returns())
    return model


# ---------------------------------------------------------------------------
# Test 1: fit() converges on synthetic returns
# ---------------------------------------------------------------------------

def test_fit_convergence():
    model = _fitted_model()
    assert model.is_fitted, "Model should be marked fitted after fit()"
    p = model._params
    assert p is not None
    assert p.kappa > 0, "kappa must be positive"
    assert p.omega > 0, "omega must be positive"
    assert p.alpha + p.beta < 1.0, "GARCH must be stationary (α+β < 1)"
    assert p.lam > 0, "jump intensity must be positive"
    assert 0 < p.p_up < 1, "p_up must be in (0,1)"
    assert p.eta1 > 0 and p.eta2 > 0, "eta1/eta2 must be positive"


# ---------------------------------------------------------------------------
# Test 2: fit() raises ValueError on too-short input
# ---------------------------------------------------------------------------

def test_fit_too_short():
    model = OUKouGARCH()
    with pytest.raises(ValueError, match="requires"):
        model.fit(np.random.standard_normal(20))


# ---------------------------------------------------------------------------
# Test 3: forecast_sigma_h() is positive and finite
# ---------------------------------------------------------------------------

def test_forecast_sigma_h_positive_finite():
    model = _fitted_model()
    for h in [1, 5, 21]:
        sigma = model.forecast_sigma_h(h)
        assert np.isfinite(sigma), f"sigma_h({h}) must be finite"
        assert sigma > 0, f"sigma_h({h}) must be positive"


# ---------------------------------------------------------------------------
# Test 4: p_in_range() in [0,1] and monotone in threshold
# ---------------------------------------------------------------------------

def test_p_in_range_bounds_and_monotone():
    model = _fitted_model()
    horizon = 5
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20]
    probs = [model.p_in_range(horizon, t) for t in thresholds]
    for p in probs:
        assert 0.0 <= p <= 1.0, f"p_in_range out of [0,1]: {p}"
    for i in range(len(probs) - 1):
        assert probs[i] <= probs[i + 1] + 1e-6, (
            f"p_in_range not monotone: {probs[i]:.4f} > {probs[i+1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: bic() is finite and within plausible range
# ---------------------------------------------------------------------------

def test_bic_finite():
    model = _fitted_model()
    b = model.bic()
    assert np.isfinite(b), f"BIC must be finite, got {b}"
    assert b < 1e6, f"BIC suspiciously large: {b}"


# ---------------------------------------------------------------------------
# Test 6: to_state_dict() is JSON-serialisable and has "diagnostics" key
# ---------------------------------------------------------------------------

def test_to_state_dict_json_serialisable():
    model = _fitted_model()
    d = model.to_state_dict()
    assert isinstance(d, dict)

    # Must not raise
    json_str = json.dumps(d)
    assert len(json_str) > 50

    # Required top-level keys
    for key in ("converged", "loglik", "bic", "params", "aekf_final_state",
                "direction", "direction_confidence", "diagnostics"):
        assert key in d, f"to_state_dict missing key: {key}"

    # No numpy arrays
    def _check_no_ndarray(obj, path=""):
        if isinstance(obj, np.ndarray):
            raise AssertionError(f"numpy array found at {path}")
        if isinstance(obj, dict):
            for k, v in obj.items():
                _check_no_ndarray(v, f"{path}.{k}")
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                _check_no_ndarray(v, f"{path}[{i}]")

    _check_no_ndarray(d)


# ---------------------------------------------------------------------------
# Test 6b: diagnostics() values are within expected ranges
# ---------------------------------------------------------------------------

def test_diagnostics_values():
    model = _fitted_model()
    diag = model.diagnostics()

    assert 0.0 <= diag.get("jb_pvalue", 0.5) <= 1.0 or np.isnan(diag.get("jb_pvalue", 0.5))
    assert np.isfinite(diag.get("resid_kurtosis", float("nan"))) or \
           np.isnan(diag.get("resid_kurtosis", float("nan")))

    half_life = diag.get("ou_half_life_days")
    assert half_life is not None and half_life > 0, f"ou_half_life_days should be >0, got {half_life}"

    intensity = diag.get("jump_intensity_annual")
    assert intensity is not None and intensity > 0, f"jump_intensity_annual should be >0"

    persistence = diag.get("diffusion_persistence")
    assert persistence is not None and 0 < persistence < 1, f"diffusion_persistence should be in (0,1)"

    aic = diag.get("aic")
    assert aic is not None and np.isfinite(aic), f"aic should be finite"


# ---------------------------------------------------------------------------
# Test 7: AEKF state is valid
# ---------------------------------------------------------------------------

def test_aekf_state_valid():
    model = _fitted_model()
    assert model._aekf is not None, "AEKF should be set after fit()"

    X_T, kappa_T, mu_T = model._aekf.final_state
    assert np.isfinite(X_T), "X_T must be finite"
    assert kappa_T > 0, f"κ_T must be positive, got {kappa_T}"
    assert np.isfinite(mu_T), "μ_T must be finite"

    sh = model._aekf.state_history
    assert sh.ndim == 2 and sh.shape[1] == 3, f"state_history shape should be (T,3), got {sh.shape}"
    assert sh.shape[0] == len(model._returns), (
        f"state_history length {sh.shape[0]} != n_returns {len(model._returns)}"
    )


# ---------------------------------------------------------------------------
# Test 8: direction_signal() returns valid output
# ---------------------------------------------------------------------------

def test_direction_signal_valid():
    model = _fitted_model()
    direction, confidence = model.direction_signal()
    assert direction in {"BULLISH", "BEARISH"}, f"unexpected direction: {direction}"
    assert 0.0 <= confidence <= 1.0, f"confidence out of [0,1]: {confidence}"


# ---------------------------------------------------------------------------
# Test 9: GARCHRangeModel._fit_oujump() returns correct dict shape
# ---------------------------------------------------------------------------

def test_garch_range_model_fit_oujump():
    from ait.ml.garch_range_predictor import GARCHRangeModel

    rng = np.random.default_rng(7)
    prices = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 300)) * 100)
    returns = np.diff(np.log(prices.values))

    garch = GARCHRangeModel()
    result = garch._fit_oujump(returns, horizon_days=5, threshold_pct=0.05)

    assert "converged" in result
    assert "selected_variant" in result

    if result["converged"]:
        assert result["selected_variant"] == "OU-Kou-GARCH"
        assert np.isfinite(result["bic"]), "BIC must be finite when converged"
        p_val = result["p_in_range_compounding"]
        assert 0.0 <= p_val <= 1.0, f"p_in_range_compounding out of [0,1]: {p_val}"
        assert "oujump_state" in result, "oujump_state key must be present"
        assert result["ou_jump_direction"] in {"BULLISH", "BEARISH"}
        # Live object must be present (stripped later)
        assert "_oujump_obj" in result


# ---------------------------------------------------------------------------
# Test 10: cv_score_oujump() runs without error
# ---------------------------------------------------------------------------

def test_cv_score_oujump_runs():
    from ait.ml.garch_range_predictor import GARCHRangeModel
    from ait.ml.range_predictor import RangePredictor

    rng = np.random.default_rng(13)
    n = 250
    prices = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, n)) * 100)

    garch = GARCHRangeModel()
    fold = n // 5
    splits = []
    for i in range(4):
        tr_end = fold * (i + 1)
        val_start = tr_end + 5
        val_end = min(val_start + fold, n)
        if val_end > val_start:
            splits.append((np.arange(0, tr_end), np.arange(val_start, val_end)))

    rp = RangePredictor(threshold_pct=0.05, horizon_days=5)
    labels_fn = lambda c: rp._create_labels_horizon(c, 5)

    result = garch.cv_score_oujump(
        close=prices, horizon_days=5, threshold_pct=0.05,
        splits=splits, create_labels_fn=labels_fn,
    )
    assert result is None or (0.0 <= result <= 1.0), (
        f"cv_score_oujump returned unexpected value: {result}"
    )


# ---------------------------------------------------------------------------
# Test 11: GARCHRangeModel.fit() includes "OU-Kou-GARCH" in garch_all_variants
# ---------------------------------------------------------------------------

def test_garch_range_fit_includes_oujump():
    from ait.ml.garch_range_predictor import GARCHRangeModel

    rng = np.random.default_rng(99)
    prices = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 300)) * 100)

    garch = GARCHRangeModel()
    state = garch.fit(prices, horizon_days=5, threshold_pct=0.05)

    all_variants = state.get("garch_all_variants", {})
    assert "OU-Kou-GARCH" in all_variants, (
        f"OU-Kou-GARCH should appear in garch_all_variants. Keys: {list(all_variants.keys())}"
    )
    ou_entry = all_variants["OU-Kou-GARCH"]
    assert "converged" in ou_entry
    assert "bic" in ou_entry


# ---------------------------------------------------------------------------
# Test 12: RangePredictor(enable_oujump=True) trains and stores ou_jump keys
# ---------------------------------------------------------------------------

def test_range_predictor_enable_oujump():
    from ait.ml.range_predictor import RangePredictor

    rng = np.random.default_rng(77)
    n = 300
    prices = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
    df = pd.DataFrame({
        "Open":   prices * (1 + rng.normal(0, 0.002, n)),
        "High":   prices * (1 + np.abs(rng.normal(0, 0.005, n))),
        "Low":    prices * (1 - np.abs(rng.normal(0, 0.005, n))),
        "Close":  prices,
        "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })

    rp = RangePredictor(threshold_pct=0.05, horizon_days=5, enable_oujump=True)
    accs = rp.train(df, symbol="TEST")

    # Model might not train if insufficient class balance — just check keys exist
    sym = rp._symbol_models.get("TEST", {})
    if sym:
        # ou_jump keys must be present even if oujump training failed
        assert "ou_jump_state" in sym, "ou_jump_state key missing from _symbol_models"
        assert "ou_jump_direction" in sym, "ou_jump_direction key missing"
        assert "ou_jump_bic" in sym, "ou_jump_bic key missing"

    # Weights must include "oujump" in the prior
    assert "oujump" in rp._weights, "'oujump' key must be in prior weights"

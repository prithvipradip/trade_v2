"""Unit tests for MarkovSwitchingGARCH (msgarch.py).

Tests cover:
  1. fit() converges on synthetic GARCH returns
  2. fit() raises ValueError on too-short input
  3. forecast_sigma_h() is positive and finite
  4. p_in_range() is in [0, 1] and decreases with tighter thresholds
  5. bic() is finite after fitting
  6. to_state_dict() is JSON-serialisable
  7. Properties (regime0_params, regime1_params, transition_matrix, final_regime_probs)
  8. GARCHRangeModel._fit_msgarch() integrates correctly
  9. GARCHRangeModel.cv_score_msgarch() runs without error
 10. GARCHRangeModel.fit() includes MS-GARCH in all_variants
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ait.ml.msgarch import MarkovSwitchingGARCH, MSGARCHParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_returns(n: int = 200, seed: int = 42) -> np.ndarray:
    """Simple GARCH(1,1)-like returns: two-regime by design."""
    rng = np.random.default_rng(seed)
    returns = np.empty(n)
    sigma2 = 0.01
    for t in range(n):
        sigma2 = max(1e-6 + 0.05 * returns[t - 1] ** 2 + 0.90 * sigma2, 1e-10) if t > 0 else 0.01
        returns[t] = rng.normal(0, np.sqrt(sigma2))
    return returns


def _fitted_ms() -> MarkovSwitchingGARCH:
    ms = MarkovSwitchingGARCH()
    ms.fit(_synthetic_returns())
    return ms


# ---------------------------------------------------------------------------
# Test 1: fit() converges
# ---------------------------------------------------------------------------

def test_fit_convergence():
    ms = MarkovSwitchingGARCH()
    ret = _synthetic_returns()
    ms.fit(ret)
    assert ms.is_fitted, "Model should be marked fitted after fit()"
    assert ms._params is not None
    p = ms._params
    # Stationarity: both regimes must have α+β < 1
    assert p.alpha[0] + p.beta[0] < 1.0
    assert p.alpha[1] + p.beta[1] < 1.0
    # All omegas positive
    assert p.omega[0] > 0
    assert p.omega[1] > 0
    # Transition probs in (0, 1)
    assert 0 < p.p00 < 1
    assert 0 < p.p11 < 1


# ---------------------------------------------------------------------------
# Test 2: fit() raises ValueError on short input
# ---------------------------------------------------------------------------

def test_fit_too_short():
    ms = MarkovSwitchingGARCH()
    with pytest.raises(ValueError, match="requires"):
        ms.fit(np.random.standard_normal(20))


# ---------------------------------------------------------------------------
# Test 3: forecast_sigma_h() is positive and finite
# ---------------------------------------------------------------------------

def test_forecast_sigma_positive():
    ms = _fitted_ms()
    for h in [1, 5, 21]:
        sigma = ms.forecast_sigma_h(h)
        assert np.isfinite(sigma), f"sigma_h({h}) should be finite"
        assert sigma > 0, f"sigma_h({h}) should be positive"


# ---------------------------------------------------------------------------
# Test 4: p_in_range() in [0,1] and monotone in threshold
# ---------------------------------------------------------------------------

def test_p_in_range_bounds_and_monotone():
    ms = _fitted_ms()
    horizon = 5
    thresholds = [0.01, 0.03, 0.05, 0.10, 0.20]
    probs = [ms.p_in_range(horizon, t) for t in thresholds]
    for p in probs:
        assert 0.0 <= p <= 1.0, f"p_in_range out of [0,1]: {p}"
    # Wider threshold → higher probability
    for i in range(len(probs) - 1):
        assert probs[i] <= probs[i + 1] + 1e-9, (
            f"p_in_range not monotone: {probs[i]:.4f} > {probs[i+1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 5: bic() is finite
# ---------------------------------------------------------------------------

def test_bic_finite():
    ms = _fitted_ms()
    b = ms.bic()
    assert np.isfinite(b), f"BIC should be finite, got {b}"
    # BIC = -2·loglik + 8·log(200) — rough sanity: should be negative or modest positive
    assert b < 1e6, f"BIC suspiciously large: {b}"


# ---------------------------------------------------------------------------
# Test 6: to_state_dict() is JSON-serialisable
# ---------------------------------------------------------------------------

def test_state_dict_json_serialisable():
    ms = _fitted_ms()
    d = ms.to_state_dict()
    assert isinstance(d, dict)
    # Must not raise
    json_str = json.dumps(d)
    assert len(json_str) > 10


# ---------------------------------------------------------------------------
# Test 7: Properties return correct shapes / types
# ---------------------------------------------------------------------------

def test_properties():
    ms = _fitted_ms()

    r0 = ms.regime0_params
    assert set(r0.keys()) == {"omega", "alpha", "beta", "persistence"}
    assert 0 < r0["persistence"] < 1

    r1 = ms.regime1_params
    assert set(r1.keys()) == {"omega", "alpha", "beta", "persistence"}
    assert 0 < r1["persistence"] < 1

    tm = ms.transition_matrix
    assert abs(tm["p00"] + tm["p01"] - 1.0) < 1e-6
    assert abs(tm["p11"] + tm["p10"] - 1.0) < 1e-6

    fp = ms.final_regime_probs
    assert len(fp) == 2
    assert abs(sum(fp) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Test 8: GARCHRangeModel._fit_msgarch() returns well-formed dict
# ---------------------------------------------------------------------------

def test_garch_range_model_fit_msgarch():
    from ait.ml.garch_range_predictor import GARCHRangeModel
    import pandas as pd

    rng = np.random.default_rng(7)
    prices = np.cumprod(1 + rng.normal(0, 0.01, 300)) * 100
    returns = np.diff(np.log(prices))

    garch = GARCHRangeModel()
    result = garch._fit_msgarch(returns, horizon_days=5, threshold_pct=0.05)

    assert "converged" in result
    if result["converged"]:
        assert result["selected_variant"] == "MS-GARCH"
        assert np.isfinite(result["bic"])
        assert 0 <= result["p_in_range_compounding"] <= 1
        assert "msgarch_state" in result


# ---------------------------------------------------------------------------
# Test 9: cv_score_msgarch() returns float or None — no crashes
# ---------------------------------------------------------------------------

def test_cv_score_msgarch_runs():
    from ait.ml.garch_range_predictor import GARCHRangeModel
    import pandas as pd

    rng = np.random.default_rng(13)
    n = 250
    prices = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, n)) * 100)

    garch = GARCHRangeModel()

    # Build splits manually (mirrors _walk_forward_split)
    fold = n // 5
    splits = []
    for i in range(4):
        tr_end = fold * (i + 1)
        val_start = tr_end + 5
        val_end = min(val_start + fold, n)
        if val_end > val_start:
            splits.append((np.arange(0, tr_end), np.arange(val_start, val_end)))

    from ait.ml.range_predictor import RangePredictor
    rp = RangePredictor(threshold_pct=0.05, horizon_days=5)
    labels_fn = lambda c: rp._create_labels_horizon(c, 5)

    result = garch.cv_score_msgarch(
        close=prices, horizon_days=5, threshold_pct=0.05,
        splits=splits, create_labels_fn=labels_fn,
    )
    assert result is None or (0.0 <= result <= 1.0), f"Unexpected cv_score_msgarch result: {result}"


# ---------------------------------------------------------------------------
# Test 10: GARCHRangeModel.fit() includes MS-GARCH in all_variants
# ---------------------------------------------------------------------------

def test_garch_range_fit_includes_msgarch():
    from ait.ml.garch_range_predictor import GARCHRangeModel
    import pandas as pd

    rng = np.random.default_rng(99)
    prices = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, 300)) * 100)

    garch = GARCHRangeModel()
    state = garch.fit(prices, horizon_days=5, threshold_pct=0.05)

    all_variants = state.get("garch_all_variants", {})
    assert "MS-GARCH" in all_variants, (
        f"MS-GARCH should appear in garch_all_variants. Keys: {list(all_variants.keys())}"
    )
    ms_entry = all_variants["MS-GARCH"]
    assert "converged" in ms_entry
    assert "bic" in ms_entry

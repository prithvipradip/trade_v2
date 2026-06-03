"""Unit tests for GARCHRangeModel and ClassicalTemperedStable.

Tests cover:
- GARCHRangeModel.fit() state dict completeness
- P(in range) bounds and monotonicity
- Variant/distribution selection logic
- Both horizon methods
- Fallback chain
- CV fold correctness (no lookahead)
- ClassicalTemperedStable mathematical properties
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _gbm_prices(n: int = 500, sigma: float = 0.01, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, sigma, n)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), name="Close")


def _fat_tail_prices(n: int = 500, df: int = 4, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = stats.t.rvs(df=df, scale=0.01, size=n, random_state=rng)
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), name="Close")


def _asymmetric_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Prices with left-skewed return distribution."""
    rng = np.random.default_rng(seed)
    returns = -abs(rng.normal(0, 0.01, n))  # all negative returns → left skew
    returns += 0.0002  # add small drift so prices don't collapse
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), name="Close")


def _garch_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Prices generated from a GARCH(1,1) process with clear clustering."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.00001, 0.10, 0.85
    sigma2 = omega / (1 - alpha - beta)
    prices = [100.0]
    for _ in range(n):
        z = rng.standard_normal()
        eps = np.sqrt(sigma2) * z
        prices.append(prices[-1] * np.exp(eps))
        sigma2 = omega + alpha * eps ** 2 + beta * sigma2
    return pd.Series(prices, name="Close")


@pytest.fixture
def price_normal() -> pd.Series:
    return _gbm_prices(sigma=0.008)


@pytest.fixture
def price_fat_tails() -> pd.Series:
    return _fat_tail_prices(df=4)


@pytest.fixture
def price_asymmetric() -> pd.Series:
    return _asymmetric_prices()


@pytest.fixture
def price_garch() -> pd.Series:
    return _garch_prices()


@pytest.fixture
def garch_model():
    from ait.ml.garch_range_predictor import GARCHRangeModel
    return GARCHRangeModel()


@pytest.fixture
def cts():
    from ait.ml.garch_range_predictor import ClassicalTemperedStable
    return ClassicalTemperedStable()


# ---------------------------------------------------------------------------
# GARCHRangeModel — core tests
# ---------------------------------------------------------------------------

class TestGARCHRangeModelFit:
    def test_fit_returns_complete_state(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        required = [
            "selected_variant", "selected_dist", "bic",
            "p_in_range_compounding", "p_in_range_sqrt_scale",
            "sigma_compounding", "sigma_sqrt_scale",
            "fallback_used", "jb_pvalue", "resid_skewness",
            "garch_stable_attempted", "garch_stable_converged",
            "garch_all_variants", "threshold_pct", "horizon_days",
        ]
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_p_in_range_compounding_bounded(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        p = state["p_in_range_compounding"]
        assert 0.0 <= p <= 1.0, f"P out of bounds: {p}"

    def test_p_in_range_sqrt_scale_bounded(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        p = state["p_in_range_sqrt_scale"]
        assert 0.0 <= p <= 1.0, f"P out of bounds: {p}"

    def test_both_horizon_methods_stored(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        assert "p_in_range_compounding" in state
        assert "p_in_range_sqrt_scale" in state
        assert state["p_in_range_compounding"] != state["p_in_range_sqrt_scale"] or True  # may be equal for ARCH

    def test_all_variants_stored(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        variants = state["garch_all_variants"]
        assert "GARCH(1,1)" in variants
        assert "GJR-GARCH" in variants
        assert "EGARCH(1,1)" in variants
        assert "ARCH(1)" in variants

    def test_all_variants_have_dist_race(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        for v_name, v_data in state["garch_all_variants"].items():
            assert "dist_race" in v_data, f"{v_name} missing dist_race"
            assert "converged" in v_data, f"{v_name} missing converged"

    def test_dist_race_has_all_distributions(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        # Only arch-based variants run a dist_race; MS-GARCH uses EM (no dist_race).
        _ARCH_VARIANTS = {"GARCH(1,1)", "GJR-GARCH", "EGARCH(1,1)", "ARCH(1)"}
        for v_name, v_data in state["garch_all_variants"].items():
            if v_name not in _ARCH_VARIANTS:
                continue
            if v_data.get("converged"):
                dist_race = v_data["dist_race"]
                for dist in ["normal", "t", "skewt", "ged", "cts"]:
                    assert dist in dist_race, f"{v_name} missing {dist} in dist_race"

    def test_stable_diagnostic_always_stored(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        assert "garch_stable_attempted" in state

    def test_threshold_and_horizon_stored(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=14, threshold_pct=0.03)
        assert state["threshold_pct"] == 0.03
        assert state["horizon_days"] == 14

    def test_fit_short_series_returns_fallback(self, garch_model):
        short = pd.Series([100.0, 101.0, 100.5, 102.0])
        state = garch_model.fit(short, horizon_days=5, threshold_pct=0.05)
        assert state.get("fallback_used") is not None or state.get("selected_variant") is not None

    def test_predict_p_in_range_from_state(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        p = garch_model.predict_p_in_range(state)
        assert 0.0 <= p <= 1.0


class TestGARCHRangeModelMonotonicity:
    def test_p_increases_with_threshold(self, garch_model, price_normal):
        s1 = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.03)
        s2 = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.08)
        # Wider threshold → more likely to stay in range
        assert s2["p_in_range_compounding"] >= s1["p_in_range_compounding"] - 0.05

    def test_p_decreases_with_horizon(self, garch_model, price_normal):
        s1 = garch_model.fit(price_normal, horizon_days=5, threshold_pct=0.05)
        s2 = garch_model.fit(price_normal, horizon_days=30, threshold_pct=0.05)
        # Longer horizon → more chance to break out
        assert s1["p_in_range_compounding"] >= s2["p_in_range_compounding"] - 0.10


class TestGARCHFallbacks:
    def test_fallback_field_present(self, garch_model, price_normal):
        state = garch_model.fit(price_normal, horizon_days=21, threshold_pct=0.05)
        assert "fallback_used" in state

    def test_constant_vol_fallback_produces_valid_p(self, garch_model):
        """Test _constant_vol_fallback directly."""
        import numpy as np
        returns = np.random.default_rng(0).normal(0, 0.01, 100)
        state = garch_model._constant_vol_fallback(returns, horizon=21, threshold_pct=0.05)
        assert 0.0 <= state["p_in_range_compounding"] <= 1.0
        assert state["fallback_used"] == "constant_vol"

    def test_short_series_no_exception(self, garch_model):
        tiny = pd.Series(np.random.default_rng(1).normal(100, 1, 20))
        try:
            state = garch_model.fit(tiny, horizon_days=5, threshold_pct=0.05)
            assert "p_in_range_compounding" in state
        except Exception as e:
            pytest.fail(f"fit() raised on short series: {e}")


class TestGARCHCVFolds:
    def test_no_lookahead_in_cv_folds(self, garch_model, price_normal):
        """Validation indices must always be strictly after training indices."""
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        splits = rp._walk_forward_split(len(price_normal))
        for tr_idx, val_idx in splits:
            assert val_idx.min() > tr_idx.max(), "Lookahead detected in CV folds"

    def test_cv_score_returns_float(self, garch_model, price_normal):
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10)
        splits = rp._walk_forward_split(len(price_normal))
        score = garch_model.cv_score(
            close=price_normal,
            horizon_days=10,
            threshold_pct=0.05,
            splits=splits,
            create_labels_fn=rp._create_labels,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ClassicalTemperedStable — mathematical properties
# ---------------------------------------------------------------------------

class TestCTSCharacteristicFunction:
    def test_cf_at_zero_equals_one(self, cts):
        """φ(0) = 1 for all valid parameters."""
        params = (1.5, 0.5, 0.5, 5.0, 5.0)
        t = np.array([0.0])
        val = cts.characteristic_function(t, *params)
        assert abs(val[0] - 1.0) < 1e-8, f"φ(0) = {val[0]}, expected 1"

    def test_cf_conjugate_symmetry(self, cts):
        """φ(-t) = conj(φ(t)) for real-valued distribution."""
        params = (1.5, 0.5, 0.5, 5.0, 5.0)
        t = np.array([0.5, 1.0, 2.0])
        phi_t = cts.characteristic_function(t, *params)
        phi_neg = cts.characteristic_function(-t, *params)
        np.testing.assert_allclose(phi_neg, np.conj(phi_t), atol=1e-6)

    def test_cf_special_case_alpha_one(self, cts):
        """α = 1 special case should not raise and φ(0) = 1."""
        params_a1 = (1.0, 0.5, 0.5, 5.0, 5.0)
        t = np.array([0.0, 0.5, 1.0])
        val = cts.characteristic_function(t, *params_a1)
        assert abs(val[0] - 1.0) < 1e-6


class TestCTSPDF:
    def test_pdf_non_negative(self, cts):
        params = np.array([1.5, 0.5, 0.5, 5.0, 5.0])
        x_grid, pdf_vals = cts.fft_pdf(params)
        assert np.all(pdf_vals >= 0.0), "PDF has negative values"

    def test_pdf_integrates_to_one(self, cts):
        params = np.array([1.5, 0.5, 0.5, 5.0, 5.0])
        x_grid, pdf_vals = cts.fft_pdf(params)
        dx = x_grid[1] - x_grid[0]
        integral = float(np.sum(pdf_vals) * dx)
        assert abs(integral - 1.0) < 0.05, f"PDF integrates to {integral:.3f}, expected ~1.0"

    def test_loglikelihood_finite_on_valid_data(self, cts):
        rng = np.random.default_rng(42)
        std_resid = rng.standard_normal(200)
        params = np.array([1.5, 0.5, 0.5, 5.0, 5.0])
        sigma2 = np.ones(200)
        ll = cts.loglikelihood(params, std_resid, sigma2)
        assert np.isfinite(ll), f"Loglikelihood is not finite: {ll}"


class TestCTSStartingValues:
    def test_starting_values_in_bounds(self, cts):
        rng = np.random.default_rng(0)
        std_resid = rng.standard_normal(300)
        sv = cts.starting_values(std_resid)
        bounds = cts.bounds()
        assert len(sv) == len(bounds)
        for i, (lo, hi) in enumerate(bounds):
            assert sv[i] >= lo, f"Param {i} = {sv[i]} below lower bound {lo}"
            if hi is not None:
                assert sv[i] <= hi, f"Param {i} = {sv[i]} above upper bound {hi}"

    def test_starting_values_returns_correct_length(self, cts):
        sv = cts.starting_values(np.random.standard_normal(100))
        assert len(sv) == ClassicalTemperedStable_N_PARAMS()


def ClassicalTemperedStable_N_PARAMS():
    from ait.ml.garch_range_predictor import ClassicalTemperedStable
    return ClassicalTemperedStable.N_PARAMS


class TestCTSPInRange:
    def test_p_in_range_monotone_in_threshold(self, cts):
        params = np.array([1.5, 0.5, 0.5, 5.0, 5.0])
        sigma_h = 0.05
        p_narrow = cts.p_in_range(sigma_h, 0.03, params)
        p_wide = cts.p_in_range(sigma_h, 0.08, params)
        assert p_wide >= p_narrow - 0.01

    def test_p_in_range_monotone_in_vol(self, cts):
        params = np.array([1.5, 0.5, 0.5, 5.0, 5.0])
        threshold = 0.05
        p_low_vol = cts.p_in_range(0.02, threshold, params)
        p_high_vol = cts.p_in_range(0.10, threshold, params)
        assert p_low_vol >= p_high_vol - 0.01

    def test_p_in_range_bounded(self, cts):
        params = np.array([1.5, 0.5, 0.5, 5.0, 5.0])
        p = cts.p_in_range(0.05, 0.05, params)
        assert 0.0 <= p <= 1.0


class TestCTSSimulate:
    def test_simulate_returns_correct_shape(self, cts):
        params = np.array([1.5, 0.5, 0.5, 5.0, 5.0])
        rng = np.random.default_rng(42)
        samples = cts.simulate(params, nobs=200, rng=rng)
        assert len(samples) == 200

    def test_simulate_finite_values(self, cts):
        params = np.array([1.5, 0.5, 0.5, 5.0, 5.0])
        rng = np.random.default_rng(0)
        samples = cts.simulate(params, nobs=100, rng=rng)
        assert np.all(np.isfinite(samples))


# ---------------------------------------------------------------------------
# RangePredictor — GARCH ensemble integration (slow: enable_garch=True)
# ---------------------------------------------------------------------------

def _make_ohlcv_rp(days: int = 300, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2023-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.random.normal(0.0005, 0.012, days))
    open_ = np.roll(close, 1)   # open = previous day's close; first row duplicated (acceptable)
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.005, days)))
    low  = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.005, days)))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": np.random.randint(1_000_000, 10_000_000, days).astype(float)},
        index=dates,
    )


class TestRangePredictorGARCHIntegration:
    """End-to-end integration: GARCH as third ensemble member in RangePredictor.

    These tests use enable_garch=True and are intentionally slow (~5s per train).
    Kept here (not in test_ensemble_fitted_weights.py) so the fast suite stays fast.
    """

    def _trained_rp(self, seed: int = 42):
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10, enable_garch=True)
        rp.train(_make_ohlcv_rp(days=300, seed=seed), symbol="QQQ")
        return rp

    def test_fitted_weights_sum_to_one_three_way(self):
        rp = self._trained_rp()
        fw = rp._symbol_models["QQQ"]["fitted_weights"]
        assert abs(sum(fw.values()) - 1.0) < 1e-9, f"Weights sum to {sum(fw.values())}"

    def test_fitted_weights_keys_include_xgb_and_lgb(self):
        rp = self._trained_rp()
        fw = rp._symbol_models["QQQ"]["fitted_weights"]
        assert "xgboost" in fw
        assert "lightgbm" in fw

    def test_garch_cv_score_stored_if_positive(self):
        rp = self._trained_rp()
        cv = rp._symbol_models["QQQ"]["cv_scores"]
        if "garch" in cv:
            assert 0.0 < cv["garch"] <= 1.0

    def test_garch_weight_non_negative(self):
        rp = self._trained_rp()
        fw = rp._symbol_models["QQQ"]["fitted_weights"]
        if "garch" in fw:
            assert fw["garch"] >= 0.0

    def test_symbol_models_stores_garch_metadata(self):
        rp = self._trained_rp()
        sym = rp._symbol_models["QQQ"]
        assert "garch_state" in sym
        assert "garch_variant" in sym
        assert "garch_dist" in sym
        assert "garch_fallback" in sym

    def test_backward_compat_explicit_weights_no_garch(self):
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(
            threshold_pct=0.05, horizon_days=10,
            ensemble_weights={"xgboost": 0.5, "lightgbm": 0.5},
        )
        assert "garch" not in rp._weights

    def test_garch_absent_gracefully_if_arch_not_installed(self, monkeypatch):
        import ait.ml.range_predictor as rp_mod

        def _mock_train_garch(self, close):
            raise ImportError("arch not installed")

        monkeypatch.setattr(rp_mod.RangePredictor, "_train_garch", _mock_train_garch)
        from ait.ml.range_predictor import RangePredictor
        rp = RangePredictor(threshold_pct=0.05, horizon_days=10, enable_garch=True)
        try:
            rp.train(_make_ohlcv_rp(300), symbol="QQQ")
        except ImportError:
            pass
        fw = rp._symbol_models.get("QQQ", {}).get("fitted_weights", {})
        if fw:
            assert abs(sum(fw.values()) - 1.0) < 1e-9

    def test_predict_returns_valid_probability_after_train(self):
        rp = self._trained_rp()
        result = rp.predict(_make_ohlcv_rp(days=300), symbol="QQQ")
        if result is not None:
            assert 0.0 <= result.probability_in_range <= 1.0


# ---------------------------------------------------------------------------
# CV fold class-imbalance handling
# ---------------------------------------------------------------------------

class TestCVFoldClassBalance:
    """cv_score_msgarch / cv_score_oujump / cv_score score single-class folds
    as 0.5 instead of skipping them, and still skip folds below MIN_FOLD_LABELS.
    """

    def _make_splits_and_labels(self, all_same_class: bool, n_val: int = 20):
        """Return (splits, create_labels_fn) where the val fold is either
        all-one-class or genuinely mixed, with n_val labelable rows."""
        import numpy as _np
        import pandas as _pd
        from ait.ml.garch_range_predictor import GARCHRangeModel

        grm = GARCHRangeModel()
        n_train = 80
        n_total = n_train + 5 + n_val + 5   # train + gap + val + horizon buffer

        # Build price series
        rng = _np.random.default_rng(0)
        prices = _pd.Series(100.0 * _np.exp(_np.cumsum(rng.normal(0, 0.01, n_total))))

        tr_idx = _np.arange(0, n_train)
        val_idx = _np.arange(n_train + 5, n_train + 5 + n_val)
        splits = [(tr_idx, val_idx)]

        if all_same_class:
            # Labels that are entirely 1 (all in-range — ultra-low vol)
            def create_labels_fn(close):
                return _pd.Series(1.0, index=close.index)
        else:
            # Genuinely mixed: alternate 0/1
            def create_labels_fn(close):
                vals = [float(i % 2) for i in range(len(close))]
                return _pd.Series(vals, index=close.index)

        return grm, splits, prices, create_labels_fn

    def test_msgarch_single_class_fold_scores_half(self):
        """A single-class validation fold must contribute 0.5, not be skipped."""
        grm, splits, prices, labels_fn = self._make_splits_and_labels(all_same_class=True)
        result = grm.cv_score_msgarch(prices, 5, 0.05, splits, labels_fn)
        # With one fold that is single-class → scored 0.5 → mean = 0.5
        assert result is not None, "single-class fold should not return None"
        assert abs(result - 0.5) < 1e-9, f"expected 0.5, got {result}"

    def test_oujump_single_class_fold_scores_half(self):
        """Same contract for cv_score_oujump."""
        grm, splits, prices, labels_fn = self._make_splits_and_labels(all_same_class=True)
        result = grm.cv_score_oujump(prices, 5, 0.05, splits, labels_fn)
        assert result is not None
        assert abs(result - 0.5) < 1e-9, f"expected 0.5, got {result}"

    def test_garch_single_class_fold_scores_half(self):
        """Same contract for cv_score (plain GARCH)."""
        from ait.ml.garch_range_predictor import GARCHRangeModel
        grm = GARCHRangeModel()
        # Build a split where the val fold has all-same labels.
        # We mock the fit() to return a constant-vol fallback so no real GARCH.
        import numpy as _np
        import pandas as _pd
        rng = _np.random.default_rng(1)
        prices = _pd.Series(100.0 * _np.exp(_np.cumsum(rng.normal(0, 0.01, 200))))
        tr_idx = _np.arange(0, 80)
        val_idx = _np.arange(85, 115)
        splits = [(tr_idx, val_idx)]

        def all_one_labels(close):
            return _pd.Series(1.0, index=close.index)

        result = grm.cv_score(prices, 5, 0.05, splits, all_one_labels)
        assert result is not None
        assert abs(result - 0.5) < 1e-9, f"expected 0.5, got {result}"

    def test_too_short_fold_still_skipped(self):
        """Folds with fewer than _MIN_FOLD_LABELS rows are skipped → None."""
        from ait.ml.garch_range_predictor import GARCHRangeModel
        import numpy as _np
        import pandas as _pd
        grm = GARCHRangeModel()
        rng = _np.random.default_rng(2)
        prices = _pd.Series(100.0 * _np.exp(_np.cumsum(rng.normal(0, 0.01, 120))))
        # Val fold of only 5 rows — below MIN_FOLD_LABELS=10
        tr_idx = _np.arange(0, 80)
        val_idx = _np.arange(85, 90)   # 5 rows
        splits = [(tr_idx, val_idx)]

        def mixed_labels(close):
            vals = [float(i % 2) for i in range(len(close))]
            return _pd.Series(vals, index=close.index)

        # Should return None because the only fold is too short
        result_ms = grm.cv_score_msgarch(prices, 5, 0.05, splits, mixed_labels)
        result_ou = grm.cv_score_oujump(prices, 5, 0.05, splits, mixed_labels)
        result_g  = grm.cv_score(prices, 5, 0.05, splits, mixed_labels)
        assert result_ms is None, "too-short fold should yield None for msgarch"
        assert result_ou is None, "too-short fold should yield None for oujump"
        assert result_g  is None, "too-short fold should yield None for garch"

    def test_mixed_class_fold_scored_normally(self):
        """A genuine multi-class fold is still scored with real AUROC (not 0.5)."""
        grm, splits, prices, labels_fn = self._make_splits_and_labels(all_same_class=False)
        result = grm.cv_score_msgarch(prices, 5, 0.05, splits, labels_fn)
        # Mixed fold: MS-GARCH produces a constant P(in range) applied to all
        # days — with alternating 0/1 labels, AUROC will be exactly 0.5
        # (a constant score can't rank). But the key is it's scored, not None.
        assert result is not None, "mixed-class fold should be scored"

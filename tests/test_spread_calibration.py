"""Tests for the spread model formula and calibration fitting logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_scripts = Path(__file__).resolve().parent.parent / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from calibrate_option_spreads import half_spread_model


# ---------------------------------------------------------------------------
# T-SM-1: formula correctness
# ---------------------------------------------------------------------------

class TestHalfSpreadModelFormula:
    """T-SM-1: parametric formula produces expected values at known inputs."""

    def test_base_only(self) -> None:
        # When IV == iv_thresh and DTE == dte_thresh, result equals base
        iv = np.array([0.20])
        dte = np.array([21.0])
        result = half_spread_model((iv, dte), base=0.02, iv_sens=0.05, iv_thresh=0.20,
                                   dte_sens=0.003, dte_thresh=21.0)
        assert result == pytest.approx(0.02, rel=1e-6)

    def test_iv_penalty_adds_correctly(self) -> None:
        # IV=0.30, iv_thresh=0.20 → iv contribution = 0.05 × 0.10 = 0.005
        iv = np.array([0.30])
        dte = np.array([21.0])
        result = half_spread_model((iv, dte), base=0.02, iv_sens=0.05, iv_thresh=0.20,
                                   dte_sens=0.003, dte_thresh=21.0)
        assert result == pytest.approx(0.02 + 0.05 * 0.10, rel=1e-6)

    def test_dte_penalty_adds_correctly(self) -> None:
        # DTE=7, dte_thresh=21 → dte contribution = 0.003 × 14 = 0.042
        iv = np.array([0.20])
        dte = np.array([7.0])
        result = half_spread_model((iv, dte), base=0.02, iv_sens=0.05, iv_thresh=0.20,
                                   dte_sens=0.003, dte_thresh=21.0)
        assert result == pytest.approx(0.02 + 0.003 * 14, rel=1e-6)

    def test_no_penalty_when_iv_below_threshold(self) -> None:
        # IV=0.10 < iv_thresh=0.20 → no IV penalty
        iv = np.array([0.10])
        dte = np.array([30.0])  # above dte_thresh, no DTE penalty either
        result = half_spread_model((iv, dte), base=0.03, iv_sens=0.10, iv_thresh=0.20,
                                   dte_sens=0.005, dte_thresh=21.0)
        assert result == pytest.approx(0.03, rel=1e-6)

    def test_no_penalty_when_dte_above_threshold(self) -> None:
        # DTE=45 > dte_thresh=21 → no DTE penalty
        iv = np.array([0.20])
        dte = np.array([45.0])
        result = half_spread_model((iv, dte), base=0.02, iv_sens=0.05, iv_thresh=0.20,
                                   dte_sens=0.003, dte_thresh=21.0)
        assert result == pytest.approx(0.02, rel=1e-6)

    def test_vectorized_input(self) -> None:
        iv = np.array([0.20, 0.30, 0.40])
        dte = np.array([21.0, 14.0, 7.0])
        result = half_spread_model((iv, dte), base=0.01, iv_sens=0.04, iv_thresh=0.20,
                                   dte_sens=0.002, dte_thresh=21.0)
        assert len(result) == 3
        # First row: no penalty
        assert result[0] == pytest.approx(0.01, rel=1e-6)
        # Second row: iv penalty only
        assert result[1] == pytest.approx(0.01 + 0.04 * 0.10 + 0.002 * 7, rel=1e-6)


# ---------------------------------------------------------------------------
# T-SM-2: model fitting recovers known parameters
# ---------------------------------------------------------------------------

class TestFitSyntheticData:
    """T-SM-2: fit recovers known params from synthetic data within 10% error."""

    def _generate_synthetic(
        self,
        n: int = 200,
        true_base: float = 0.015,
        true_iv_sens: float = 0.06,
        true_iv_thresh: float = 0.20,
        true_dte_sens: float = 0.002,
        true_dte_thresh: float = 21.0,
        noise_std: float = 0.001,
        seed: int = 42,
    ) -> tuple:
        rng = np.random.default_rng(seed)
        iv = rng.uniform(0.10, 0.50, n)
        dte = rng.uniform(5.0, 60.0, n)
        y_true = half_spread_model(
            (iv, dte),
            base=true_base,
            iv_sens=true_iv_sens,
            iv_thresh=true_iv_thresh,
            dte_sens=true_dte_sens,
            dte_thresh=true_dte_thresh,
        )
        y_noisy = y_true + rng.normal(0, noise_std, n)
        y_noisy = np.clip(y_noisy, 0.001, None)
        return iv, dte, y_noisy, (true_base, true_iv_sens, true_iv_thresh, true_dte_sens, true_dte_thresh)

    def test_fit_recovers_params_within_10pct(self) -> None:
        from scipy.optimize import curve_fit

        iv, dte, y, true_params = self._generate_synthetic()
        p0 = [0.02, 0.05, 0.20, 0.002, 21.0]
        bounds = ([0.0, 0.0, 0.05, 0.0, 5.0], [0.20, 0.50, 0.50, 0.05, 60.0])

        popt, _ = curve_fit(
            half_spread_model,
            (iv, dte),
            y,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )

        true_base, true_iv_sens, true_iv_thresh, true_dte_sens, true_dte_thresh = true_params
        fitted_base, fitted_iv_sens, fitted_iv_thresh, fitted_dte_sens, fitted_dte_thresh = popt

        assert abs(fitted_base - true_base) / true_base < 0.10, \
            f"base: fitted={fitted_base:.4f} true={true_base:.4f}"
        assert abs(fitted_iv_sens - true_iv_sens) / true_iv_sens < 0.10, \
            f"iv_sens: fitted={fitted_iv_sens:.4f} true={true_iv_sens:.4f}"
        assert abs(fitted_dte_sens - true_dte_sens) / true_dte_sens < 0.15, \
            f"dte_sens: fitted={fitted_dte_sens:.5f} true={true_dte_sens:.5f}"

    def test_rmse_low_on_synthetic_data(self) -> None:
        from scipy.optimize import curve_fit

        iv, dte, y, _ = self._generate_synthetic(noise_std=0.0005)
        p0 = [0.02, 0.05, 0.20, 0.002, 21.0]
        bounds = ([0.0, 0.0, 0.05, 0.0, 5.0], [0.20, 0.50, 0.50, 0.05, 60.0])

        popt, _ = curve_fit(
            half_spread_model, (iv, dte), y,
            p0=p0, bounds=bounds, maxfev=10000,
        )
        y_pred = half_spread_model((iv, dte), *popt)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        assert rmse < 0.005, f"RMSE too high: {rmse:.5f}"

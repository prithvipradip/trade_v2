"""GARCH/ARCH volatility model for range probability estimation.

Implements GARCHRangeModel — a third ensemble member for RangePredictor.
Where XGBoost and LightGBM classify range/breakout from lagged cross-sectional
features, GARCH directly models the conditional variance process and derives
P(stays in ±threshold% over horizon) analytically from the forecasted volatility.

Variant selection: GARCH(1,1), GJR-GARCH, EGARCH(1,1), ARCH(1) — best BIC wins.
Distribution selection: Normal, Student-t, Skewed-t, GED (arch-native), CTS (custom).

See docs/GARCH_METHODOLOGY.md for full mathematical derivation and references.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from scipy import integrate, interpolate, special, stats

from ait.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger("ml.garch_range")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VARIANTS: list[tuple[str, dict]] = [
    ("GARCH(1,1)", dict(vol="GARCH", p=1, o=0, q=1)),
    ("GJR-GARCH",  dict(vol="GARCH", p=1, o=1, q=1)),
    ("EGARCH(1,1)", dict(vol="EGARCH", p=1, q=1)),
    ("ARCH(1)",    dict(vol="ARCH",  p=1)),
]

_ARCH_DISTS: list[str] = ["normal", "t", "skewt", "ged"]

_FFT_GRID_N = 2048       # FFT grid size for CTS PDF inversion
_FFT_GRID_RANGE = 20.0   # Standardised residual range [-20, 20]
_MIN_RETURNS = 60        # Minimum observations to attempt GARCH fit
_CTS_ALPHA_SPECIAL = 1.0  # Singularity in Γ(-α) at α=1


# ---------------------------------------------------------------------------
# Classical Tempered Stable distribution (arch Distribution subclass)
# ---------------------------------------------------------------------------

class ClassicalTemperedStable:
    """CTS innovation distribution for GARCH models.

    Not a full arch Distribution subclass (arch's ABC is complex to extend
    cleanly for FFT-based distributions). Instead used as a standalone
    distribution for: BIC comparison, loglikelihood evaluation, P(in range).

    Parameters (5): alpha ∈ (0.1, 1.99), delta_plus > 0, delta_minus > 0,
                    lambda_plus > 0, lambda_minus > 0.
    Location (mu) absorbed into GARCH mean specification.

    Mathematical reference: Massing (2024) ALEA v21 #59, arXiv:2303.07060v4.
    Characteristic function: Eq. 9.  Cumulants: Eq. 11.
    """

    PARAM_NAMES = ["alpha", "delta_plus", "delta_minus", "lambda_plus", "lambda_minus"]
    N_PARAMS = 5

    def bounds(self) -> list[tuple[float, float]]:
        return [(0.10, 1.99), (1e-6, 20.0), (1e-6, 20.0), (1e-6, 50.0), (1e-6, 50.0)]

    def starting_values(self, std_resid: np.ndarray) -> np.ndarray:
        """Method-of-cumulants starting values using Massing (2024) Eq. 11.

        Matches empirical mean, variance, skewness to CTS cumulant formulas.
        Falls back to sensible defaults when moments are degenerate.
        """
        try:
            m1 = float(np.mean(std_resid))
            m2 = float(np.var(std_resid))
            skew = float(stats.skew(std_resid))

            # Use symmetric CTS as baseline (delta_plus = delta_minus, lambda_plus = lambda_minus)
            alpha = 1.5
            lam = 5.0
            # From κ₂ = Γ(2-α) * δ * 2 / λ^(2-α) = m2 → δ = m2 * λ^(2-α) / (2*Γ(2-α))
            gam2 = special.gamma(2.0 - alpha)
            delta = max(m2 * (lam ** (2.0 - alpha)) / (2.0 * gam2), 1e-4)
            # Asymmetry from skewness: δ+ / δ- encodes skew sign
            if abs(skew) > 0.1:
                ratio = max(0.1, 1.0 - 0.3 * np.sign(skew))
                dp = delta * ratio
                dm = delta * (2.0 - ratio)
            else:
                dp = dm = delta
            return np.array([alpha, dp, dm, lam, lam])
        except Exception:
            return np.array([1.5, 0.5, 0.5, 5.0, 5.0])

    def characteristic_function(
        self,
        t: np.ndarray,
        alpha: float,
        dp: float,
        dm: float,
        lp: float,
        lm: float,
    ) -> np.ndarray:
        """Massing (2024) Eq. 9 — vectorised over t.

        φ_CTS(t) = exp[δ₊ Γ(−α)((λ₊−it)^α − λ₊^α + itαλ₊^{α−1})
                      + δ₋ Γ(−α)((λ₋+it)^α − λ₋^α − itαλ₋^{α−1})]

        Special case α = 1: uses log-based formula to avoid Γ(−1) singularity.
        """
        t = np.asarray(t, dtype=complex)

        if abs(alpha - _CTS_ALPHA_SPECIAL) < 1e-4:
            # α = 1 special case (Massing 2024, below Eq. 9)
            # φ = exp[δ₊((λ₊−it)log(1−it/λ₊)+it) + δ₋((λ₋+it)log(1+it/λ₋)−it)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                term_p = dp * ((lp - 1j * t) * np.log(1.0 - 1j * t / lp) + 1j * t)
                term_m = dm * ((lm + 1j * t) * np.log(1.0 + 1j * t / lm) - 1j * t)
            return np.exp(term_p + term_m)

        gam_neg_alpha = special.gamma(-alpha)  # Γ(−α), defined for non-integer α
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            term_p = dp * gam_neg_alpha * (
                (lp - 1j * t) ** alpha - lp ** alpha + 1j * t * alpha * lp ** (alpha - 1)
            )
            term_m = dm * gam_neg_alpha * (
                (lm + 1j * t) ** alpha - lm ** alpha - 1j * t * alpha * lm ** (alpha - 1)
            )
        return np.exp(term_p + term_m)

    def fft_pdf(self, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse FFT of CF on [-_FFT_GRID_RANGE, +_FFT_GRID_RANGE] grid.

        Returns (x_grid, pdf_values) where pdf_values[i] = f(x_grid[i]).
        """
        alpha, dp, dm, lp, lm = parameters
        n = _FFT_GRID_N
        # Frequency grid for FFT (Nyquist frequency = n/(2*range))
        dt = 2.0 * _FFT_GRID_RANGE / n          # spacing in x domain
        du = 2.0 * np.pi / (n * dt)             # spacing in frequency domain
        u = (np.arange(n) - n // 2) * du        # centred frequency grid

        cf_vals = self.characteristic_function(u, alpha, dp, dm, lp, lm)
        # Inverse FFT: f(x) = (1/2π) ∫ φ(u) e^{-iux} du
        # ifftshift converts centred frequency grid to FFT order before ifft,
        # then fftshift re-centres the output x-grid.
        cf_shifted = np.fft.ifftshift(cf_vals)
        pdf_raw = np.fft.fftshift(np.fft.ifft(cf_shifted).real) * (n * du / (2.0 * np.pi))
        pdf_vals = np.maximum(pdf_raw, 0.0)   # clip numerical negatives

        x_grid = (np.arange(n) - n // 2) * dt
        return x_grid, pdf_vals

    def loglikelihood(
        self,
        parameters: np.ndarray,
        resids: np.ndarray,
        sigma2: np.ndarray,
        individual: bool = False,
    ) -> float | np.ndarray:
        """Log-likelihood: Σ [log f(ε_t/σ_t) − log σ_t].

        PDF f evaluated via FFT inversion, interpolated at each standardised residual.
        """
        try:
            x_grid, pdf_vals = self.fft_pdf(parameters)
            # Avoid log(0) — floor PDF at small positive value
            pdf_safe = np.maximum(pdf_vals, 1e-300)
            interp = interpolate.interp1d(
                x_grid, np.log(pdf_safe),
                kind="linear", bounds_error=False, fill_value=-700.0,
            )
            std_resid = resids / np.sqrt(sigma2)
            lls = interp(std_resid) - 0.5 * np.log(sigma2)
            if individual:
                return lls
            return float(np.sum(lls))
        except Exception:
            return -1e10 if not individual else np.full(len(resids), -1e10)

    def fit(self, std_resid: np.ndarray) -> tuple[np.ndarray | None, float]:
        """MLE on standardised residuals. Returns (params, loglik) or (None, -inf)."""
        from scipy.optimize import minimize

        x0 = self.starting_values(std_resid)
        bounds = self.bounds()
        # Dummy sigma2=1 (already standardised)
        sigma2_ones = np.ones(len(std_resid))

        def neg_ll(params: np.ndarray) -> float:
            ll = self.loglikelihood(params, std_resid, sigma2_ones)
            return -ll if np.isfinite(ll) else 1e10

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-8},
                )
            if res.success and np.isfinite(res.fun):
                return res.x, -res.fun
        except Exception:
            pass
        return None, float("-inf")

    def p_in_range(
        self,
        sigma_h: float,
        threshold_pct: float,
        parameters: np.ndarray,
    ) -> float:
        """Numerical integration of FFT PDF: ∫_{-t/σ}^{+t/σ} f(z) dz."""
        try:
            x_grid, pdf_vals = self.fft_pdf(parameters)
            interp = interpolate.interp1d(
                x_grid, pdf_vals,
                kind="linear", bounds_error=False, fill_value=0.0,
            )
            lo = -threshold_pct / sigma_h
            hi =  threshold_pct / sigma_h
            result, _ = integrate.quad(
                interp, lo, hi, limit=200, epsabs=1e-4, epsrel=1e-4,
            )
            return float(np.clip(result, 0.0, 1.0))
        except Exception:
            return 0.5

    def simulate(self, parameters: np.ndarray, nobs: int, rng: np.random.Generator) -> np.ndarray:
        """Rejection sampling using alpha-stable subordinator as proposal.

        Reference: Massing (2024) §4 — accept stable draw if U < exp(-λ·Y + const).
        Falls back to Normal when rejection rate is too high.
        """
        alpha, dp, dm, lp, lm = parameters
        lam = 0.5 * (lp + lm)   # use average tempering for proposal

        samples = np.zeros(nobs)
        i = 0
        max_attempts = nobs * 50
        attempts = 0

        # Normalisation constant: c = λ^α Γ(-α) δ  (for symmetric proxy)
        delta_proxy = 0.5 * (dp + dm)
        try:
            log_const = np.log(abs(special.gamma(-alpha))) + alpha * np.log(lam) + np.log(delta_proxy)
        except Exception:
            return rng.standard_normal(nobs)

        while i < nobs and attempts < max_attempts:
            attempts += 1
            # Draw from alpha-stable subordinator (one-sided, alpha < 2)
            try:
                from scipy.stats import levy_stable
                v = float(levy_stable.rvs(alpha=min(alpha, 1.99), beta=1.0, scale=1.0, random_state=rng))
            except Exception:
                v = float(rng.standard_normal() ** 2)
            if v <= 0:
                continue
            u = rng.uniform()
            log_accept = -lam * v + log_const
            if np.log(u) < log_accept:
                # Sign: positive with prob dp/(dp+dm), negative otherwise
                sign = 1 if rng.uniform() < dp / (dp + dm) else -1
                samples[i] = sign * v
                i += 1

        if i < nobs:
            # Fallback for remaining: draw from Normal scaled to match variance
            samples[i:] = rng.standard_normal(nobs - i)

        return samples


# ---------------------------------------------------------------------------
# Main GARCH range model
# ---------------------------------------------------------------------------

class GARCHRangeModel:
    """GARCH/ARCH volatility model producing P(price stays in ±threshold% over horizon).

    Fits all (variant × distribution) combinations, selects best BIC, and
    exposes predict_p_in_range() from a stored state dict — no refitting needed
    at prediction time.

    Used by RangePredictor._train_garch() for CV scoring and full training.
    """

    def fit(
        self,
        close: pd.Series,
        horizon_days: int,
        threshold_pct: float,
    ) -> dict:
        """Fit all variants × distributions, select best BIC, return state dict.

        State dict contains everything needed to reproduce P(in range) without
        refitting, plus all metadata stored in the window JSON.
        """
        returns = self._log_returns(close)
        if len(returns) < _MIN_RETURNS:
            log.warning("garch_insufficient_data", n=len(returns), required=_MIN_RETURNS)
            return self._constant_vol_fallback(returns, horizon_days, threshold_pct,
                                               reason="insufficient_data")

        best_bic = float("inf")
        best_state: dict | None = None
        all_variants: dict = {}

        for variant_name, vol_kwargs in _VARIANTS:
            variant_result = self._fit_variant(
                returns, variant_name, vol_kwargs, horizon_days, threshold_pct,
            )
            all_variants[variant_name] = variant_result
            if variant_result["converged"] and variant_result["bic"] < best_bic:
                best_bic = variant_result["bic"]
                best_state = variant_result

        if best_state is None:
            log.warning("garch_all_variants_failed", fallback="constant_vol")
            state = self._constant_vol_fallback(returns, horizon_days, threshold_pct,
                                                reason="all_variants_failed")
        else:
            state = best_state.copy()
            state["fallback_used"] = None

        # JB test on standardised residuals from winning model
        jb_pvalue, resid_skewness = self._residual_diagnostics(state)
        state["jb_pvalue"] = jb_pvalue
        state["resid_skewness"] = resid_skewness

        # α-stable diagnostic (research tracking only)
        stable_diag = self._attempt_stable_diagnostic(state)
        state.update(stable_diag)

        state["garch_all_variants"] = all_variants
        state["threshold_pct"] = threshold_pct
        state["horizon_days"] = horizon_days

        log.info(
            "garch_trained",
            variant=state.get("selected_variant"),
            dist=state.get("selected_dist"),
            bic=f"{state.get('selected_bic', float('nan')):.1f}",
            fallback=state.get("fallback_used"),
            p_compounding=f"{state.get('p_in_range_compounding', float('nan')):.3f}",
        )
        return state

    def predict_p_in_range(self, state: dict) -> float:
        """Return P(in range) from stored state — no refit needed."""
        return float(state.get("p_in_range_compounding", 0.5))

    # ------------------------------------------------------------------
    # Private: variant fitting
    # ------------------------------------------------------------------

    def _fit_variant(
        self,
        returns: np.ndarray,
        variant_name: str,
        vol_kwargs: dict,
        horizon_days: int,
        threshold_pct: float,
    ) -> dict:
        """Fit one GARCH variant with all distributions. Returns best-BIC result dict."""
        try:
            from arch import arch_model
        except ImportError:
            return {"converged": False, "bic": float("inf"), "dist_race": {}}

        best_bic = float("inf")
        best_dist: str | None = None
        best_result = None
        best_cts_params: np.ndarray | None = None
        dist_race: dict = {}

        # --- arch-native distributions ---
        for dist in _ARCH_DISTS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    am = arch_model(returns * 100, mean="Constant", dist=dist, **vol_kwargs)
                    res = am.fit(disp="off", show_warning=False, options={"maxiter": 500})

                if res.convergence_flag != 0:
                    dist_race[dist] = {"bic": None, "converged": False}
                    continue

                bic = float(res.bic)
                dist_race[dist] = {"bic": round(bic, 2), "converged": True}
                if bic < best_bic:
                    best_bic = bic
                    best_dist = dist
                    best_result = res
            except Exception as e:
                dist_race[dist] = {"bic": None, "converged": False, "error": str(e)[:60]}

        # --- CTS distribution (custom) ---
        cts_bic, cts_params = self._fit_cts_to_variant(returns, best_result, vol_kwargs)
        dist_race["cts"] = {"bic": round(cts_bic, 2) if np.isfinite(cts_bic) else None,
                            "converged": np.isfinite(cts_bic)}
        if np.isfinite(cts_bic) and cts_bic < best_bic:
            best_bic = cts_bic
            best_dist = "cts"
            best_cts_params = cts_params

        if best_dist is None:
            return {"converged": False, "bic": float("inf"), "dist_race": dist_race,
                    "selected_dist": None}

        # Compute sigma_h and P(in range) for the winner
        sigma_comp, sigma_sqrt = self._multi_step_sigma(best_result, horizon_days, best_dist,
                                                         best_cts_params, returns, vol_kwargs)
        p_comp = self._p_in_range(sigma_comp, threshold_pct, best_dist, best_result, best_cts_params)
        p_sqrt = self._p_in_range(sigma_sqrt, threshold_pct, best_dist, best_result, best_cts_params)

        return {
            "converged": True,
            "selected_variant": variant_name,
            "selected_dist": best_dist,
            "bic": round(best_bic, 2),
            "sigma_compounding": round(float(sigma_comp), 6),
            "sigma_sqrt_scale": round(float(sigma_sqrt), 6),
            "p_in_range_compounding": round(float(p_comp), 4),
            "p_in_range_sqrt_scale": round(float(p_sqrt), 4),
            "arch_result": best_result,    # kept for std_resid extraction and rolling CV
            "cts_params": best_cts_params,
            "dist_race": dist_race,
            "selected_bic": round(best_bic, 2),
            "_vol_kwargs": vol_kwargs,     # stored so cv_score can refit with same spec
        }

    def _fit_cts_to_variant(
        self,
        returns: np.ndarray,
        arch_result,
        vol_kwargs: dict,
    ) -> tuple[float, np.ndarray | None]:
        """Fit CTS to standardised residuals from the given arch result.

        Returns (BIC, cts_params) or (inf, None) on failure.
        """
        if arch_result is None:
            return float("inf"), None
        try:
            n = len(returns)
            std_resid = (arch_result.resid / np.sqrt(arch_result.conditional_volatility ** 2 + 1e-10)).values
            std_resid = std_resid[np.isfinite(std_resid)]
            if len(std_resid) < _MIN_RETURNS:
                return float("inf"), None

            cts = ClassicalTemperedStable()
            params, loglik = cts.fit(std_resid)
            if params is None or not np.isfinite(loglik):
                return float("inf"), None

            k = ClassicalTemperedStable.N_PARAMS
            bic = -2.0 * loglik + k * np.log(n)
            return float(bic), params
        except Exception:
            return float("inf"), None

    # ------------------------------------------------------------------
    # Private: variance forecasting
    # ------------------------------------------------------------------

    def _multi_step_sigma(
        self,
        arch_result,
        horizon: int,
        dist: str,
        cts_params: np.ndarray | None,
        returns: np.ndarray,
        vol_kwargs: dict,
    ) -> tuple[float, float]:
        """Returns (sigma_compounding, sigma_sqrt_scale) in return units (not ×100)."""
        if arch_result is None:
            # Constant vol fallback
            sigma1 = float(np.std(returns)) if len(returns) > 1 else 0.01
            return sigma1 * np.sqrt(horizon), sigma1 * np.sqrt(horizon)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = arch_result.forecast(horizon=horizon, method="analytic", reindex=False)
            var_h = float(fc.variance.iloc[-1].sum())   # sum of h one-step variances (×100² scale)
            sigma_comp = np.sqrt(max(var_h, 1e-10)) / 100.0   # convert back from %-returns

            # One-step-ahead sigma for sqrt_scale
            var_1 = float(fc.variance.iloc[-1, 0])
            sigma_1 = np.sqrt(max(var_1, 1e-10)) / 100.0
            sigma_sqrt = sigma_1 * np.sqrt(horizon)

            return float(sigma_comp), float(sigma_sqrt)
        except Exception:
            sigma1 = float(np.std(returns)) if len(returns) > 1 else 0.01
            return sigma1 * np.sqrt(horizon), sigma1 * np.sqrt(horizon)

    # ------------------------------------------------------------------
    # Private: P(in range) computation
    # ------------------------------------------------------------------

    def _p_in_range(
        self,
        sigma_h: float,
        threshold_pct: float,
        dist: str,
        arch_result,
        cts_params: np.ndarray | None,
    ) -> float:
        """Compute P(|return over horizon| < threshold) from σ_h and distribution."""
        if sigma_h <= 0 or not np.isfinite(sigma_h):
            return 0.5

        t = threshold_pct / sigma_h   # standardised threshold

        try:
            if dist == "normal":
                return float(2.0 * stats.norm.cdf(t) - 1.0)

            elif dist == "t":
                nu = float(arch_result.params.get("nu", 8.0))
                return float(stats.t.cdf(t, df=nu) - stats.t.cdf(-t, df=nu))

            elif dist == "skewt":
                # Hansen's skewed-t: use scipy skewed-t approximation
                # arch's SkewStudent has params eta (df) and lambda (skew)
                eta = float(arch_result.params.get("eta", 8.0))
                lam = float(arch_result.params.get("lambda", 0.0))
                # Use scipy's nct (non-central t) as approximation for skewed-t
                # nc parameter encodes skewness; scale back to unit variance
                nc = lam * np.sqrt(eta / (eta - 2.0)) if eta > 2 else 0.0
                p_hi = float(stats.nct.cdf(t, df=eta, nc=nc))
                p_lo = float(stats.nct.cdf(-t, df=eta, nc=nc))
                return float(np.clip(p_hi - p_lo, 0.0, 1.0))

            elif dist == "ged":
                nu = float(arch_result.params.get("nu", 2.0))
                # GED = generalised normal: scipy.stats.gennorm with beta=nu
                scale = (special.gamma(1.0 / nu) / special.gamma(3.0 / nu)) ** 0.5
                return float(stats.gennorm.cdf(t, beta=nu, scale=scale) -
                             stats.gennorm.cdf(-t, beta=nu, scale=scale))

            elif dist == "cts" and cts_params is not None:
                cts = ClassicalTemperedStable()
                return cts.p_in_range(sigma_h, threshold_pct, cts_params)

        except Exception:
            pass

        # Fallback: Normal
        return float(2.0 * stats.norm.cdf(t) - 1.0)

    # ------------------------------------------------------------------
    # Private: diagnostics
    # ------------------------------------------------------------------

    def _residual_diagnostics(self, state: dict) -> tuple[float, float]:
        """Compute JB p-value and skewness of standardised residuals from winning model."""
        try:
            arch_result = state.get("arch_result")
            if arch_result is None:
                return float("nan"), float("nan")
            std_resid = arch_result.resid / np.sqrt(arch_result.conditional_volatility ** 2 + 1e-10)
            std_resid = std_resid.dropna().values
            if len(std_resid) < 8:
                return float("nan"), float("nan")
            _, jb_pvalue = stats.jarque_bera(std_resid)
            skewness = float(stats.skew(std_resid))
            return float(jb_pvalue), skewness
        except Exception:
            return float("nan"), float("nan")

    def _attempt_stable_diagnostic(self, state: dict) -> dict:
        """Fit α-stable to standardised residuals for research tracking only.

        Not used in ensemble. Records whether stable distributions are
        competitive across windows (stored in window JSON).
        """
        result = {
            "garch_stable_attempted": False,
            "garch_stable_converged": False,
            "garch_stable_loglik": None,
        }
        try:
            from scipy.stats import levy_stable
            arch_result = state.get("arch_result")
            if arch_result is None:
                return result

            std_resid = arch_result.resid / np.sqrt(arch_result.conditional_volatility ** 2 + 1e-10)
            std_resid = std_resid.dropna().values
            if len(std_resid) < _MIN_RETURNS:
                return result

            result["garch_stable_attempted"] = True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alpha_s, beta_s, loc_s, scale_s = levy_stable.fit(std_resid, floc=0)
                loglik = float(np.sum(levy_stable.logpdf(std_resid, alpha_s, beta_s, loc_s, scale_s)))

            if np.isfinite(loglik):
                result["garch_stable_converged"] = True
                result["garch_stable_loglik"] = round(loglik, 2)
        except Exception:
            pass

        return result

    # ------------------------------------------------------------------
    # Private: fallbacks
    # ------------------------------------------------------------------

    def _constant_vol_fallback(
        self,
        returns: np.ndarray,
        horizon: int,
        threshold_pct: float,
        reason: str = "unknown",
    ) -> dict:
        """Unconditional volatility fallback. Always succeeds."""
        sigma = float(np.std(returns)) if len(returns) > 1 else 0.01
        sigma_h = sigma * np.sqrt(horizon)
        p = float(np.clip(2.0 * stats.norm.cdf(threshold_pct / sigma_h) - 1.0, 0.0, 1.0))
        return {
            "converged": True,
            "selected_variant": "constant_vol",
            "selected_dist": "normal",
            "selected_bic": float("nan"),
            "bic": float("nan"),
            "fallback_used": "constant_vol",
            "fallback_reason": reason,
            "sigma_compounding": round(sigma_h, 6),
            "sigma_sqrt_scale": round(sigma_h, 6),
            "p_in_range_compounding": round(p, 4),
            "p_in_range_sqrt_scale": round(p, 4),
            "arch_result": None,
            "cts_params": None,
            "dist_race": {},
            "garch_all_variants": {},
            "jb_pvalue": float("nan"),
            "resid_skewness": float("nan"),
            "garch_stable_attempted": False,
            "garch_stable_converged": False,
            "garch_stable_loglik": None,
        }

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _log_returns(close: pd.Series) -> np.ndarray:
        """Compute log returns from Close price series."""
        close = close.dropna()
        if len(close) < 2:
            return np.array([])
        return np.diff(np.log(close.values))

    # ------------------------------------------------------------------
    # Walk-forward CV scoring (called by RangePredictor._train_garch)
    # ------------------------------------------------------------------

    def cv_score(
        self,
        close: pd.Series,
        horizon_days: int,
        threshold_pct: float,
        splits: list[tuple[np.ndarray, np.ndarray]],
        create_labels_fn,
    ) -> float:
        """Walk-forward CV balanced accuracy for GARCH P(in range).

        GARCH produces a calibrated probability P(in range) per day via rolling
        forecasts. Scoring it with binary balanced accuracy fails because P(in
        range) at QQQ-like vol levels rarely straddles any fixed threshold —
        it sits in a narrow band (e.g. 0.38–0.75) that always falls entirely
        above or below 0.5, giving mechanically degenerate scores.

        Instead we score with AUROC (Area Under the ROC Curve), which measures
        rank-order discrimination — does higher P(in range) correctly rank
        days that actually stayed in range above days that broke out? AUROC=0.5
        means no skill (same baseline as balanced accuracy=0.5), AUROC=0.6 is
        meaningful, same edge-over-0.5 structure as the fitted-weight formula.

        Rolling per-day forecasts: arch's forecast(start=...) produces a
        separate h-day variance forecast for each validation day, conditioning
        on all returns up to that day, giving genuinely per-day probabilities.
        """
        from sklearn.metrics import roc_auc_score

        scores = []
        for tr_idx, val_idx in splits:
            try:
                tr_close = close.iloc[tr_idx]
                val_close = close.iloc[val_idx]

                # True labels on validation window
                val_labels = create_labels_fn(val_close).dropna()
                if len(val_labels) == 0:
                    continue
                y_true = val_labels.values.astype(int)
                if len(np.unique(y_true)) < 2:
                    continue  # AUROC undefined with one class

                # Fit GARCH on training slice
                state = self.fit(tr_close, horizon_days, threshold_pct)
                arch_result = state.get("arch_result")

                if arch_result is None:
                    # Constant-vol fallback: single P applied to all days
                    p_scores = np.full(len(y_true), self.predict_p_in_range(state))
                else:
                    try:
                        # Rolling forecasts: refit on combined train+val, extract
                        # forecasts starting at first validation day.
                        combined_returns = self._log_returns(
                            pd.concat([tr_close, val_close])
                        ) * 100
                        vol_spec = state.get("_vol_kwargs", dict(vol="GARCH", p=1, q=1))
                        dist = state.get("selected_dist", "normal")
                        from arch import arch_model
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            am = arch_model(combined_returns, mean="Constant",
                                            dist=dist, **vol_spec)
                            res_full = am.fit(disp="off", show_warning=False,
                                              options={"maxiter": 300})

                        start_idx = len(tr_close) - 1
                        fc = res_full.forecast(
                            horizon=horizon_days, start=start_idx,
                            method="analytic", reindex=False,
                        )
                        var_per_day = fc.variance.sum(axis=1).values
                        sigma_per_day = np.sqrt(np.maximum(var_per_day, 1e-10)) / 100.0

                        p_scores = np.array([
                            self._p_in_range(s, threshold_pct, dist, res_full, None)
                            for s in sigma_per_day
                        ])
                        n = min(len(y_true), len(p_scores))
                        p_scores = p_scores[:n]
                        y_true = y_true[:n]
                    except Exception:
                        p_scores = np.full(len(y_true), self.predict_p_in_range(state))

                if len(np.unique(y_true)) < 2:
                    continue
                # AUROC: does higher P(in range) correctly rank in-range days above breakouts?
                auroc = float(roc_auc_score(y_true, p_scores))
                scores.append(auroc)
            except Exception:
                continue

        if not scores:
            return 0.0
        avg = float(np.mean(scores))
        log.info("garch_cv_score", auroc=f"{avg:.3f}", folds=len(scores))
        return avg

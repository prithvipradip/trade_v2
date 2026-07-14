"""OU-Kou-GARCH jump-diffusion model with Adaptive Extended Kalman Filter.

Models log-price X_t = log(S_t) as:
  dX_t = κ(μ - X_t)dt + σ_t dW_t + J_t dN_t

Where:
  - κ(μ - X_t)dt  Ornstein-Uhlenbeck mean reversion
  - σ_t           GARCH(1,1) conditional volatility
  - J_t dN_t      Compound Poisson jumps with Kou double-exponential sizes

The Adaptive EKF jointly tracks latent state [X_t, κ_t, μ_t] and inflates
process noise when innovations exceed 3σ (structural-break detection).

Two outputs:
  - P(in range):  via characteristic function inversion (FFT), for RangePredictor ensemble
  - Direction:    κ_T·(μ_T - X_T) from AEKF final state, BULLISH/BEARISH + confidence

See docs/OU_KOU_GARCH_METHODOLOGY.md for full mathematical derivation and references.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from scipy import integrate, interpolate, optimize, stats

from ait.utils.logging import get_logger

log = get_logger("ml.oujump")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_PARAMS     = 9       # κ, μ, ω, α, β, λ, p_up, η₁, η₂
_MIN_OBS      = 60      # minimum returns for reliable estimation
_MAX_ITER_MLE = 500
_MIN_VAR      = 1e-10   # floor for variance (numerical stability)
_FFT_N        = 2048    # FFT grid size (matches ClassicalTemperedStable)
_FFT_RANGE    = 15.0    # log-price grid half-width
_GAMMA_ADAPT  = 3.0     # AEKF innovation threshold (multiples of predicted std)
_ALPHA_ADAPT  = 5.0     # Q covariance inflation factor on large innovation
_K_MAX_JUMPS  = 10      # Poisson truncation for log-likelihood sum
_DT           = 1.0 / 252  # daily time step


# ---------------------------------------------------------------------------
# Named tuple for fitted parameters
# ---------------------------------------------------------------------------

class OUKouGARCHParams(NamedTuple):
    kappa: float   # mean-reversion speed κ > 0
    mu:    float   # long-run log-price equilibrium
    omega: float   # GARCH intercept ω > 0
    alpha: float   # GARCH ARCH coefficient α ≥ 0
    beta:  float   # GARCH GARCH coefficient β ≥ 0
    lam:   float   # Poisson jump intensity λ > 0  (per day)
    p_up:  float   # P(up jump) ∈ (0, 1)
    eta1:  float   # up-jump rate η₁ > 0  (mean up-jump = 1/η₁ in log-return)
    eta2:  float   # down-jump rate η₂ > 0 (mean down-jump = 1/η₂)


# ---------------------------------------------------------------------------
# Adaptive Extended Kalman Filter
# ---------------------------------------------------------------------------

class AdaptiveEKF:
    """Tracks latent state z = [X_t, κ_t, μ_t] via linearised OU dynamics.

    The 'adaptive' part: when the Kalman innovation |ν_t| exceeds γ standard
    deviations of the predicted state, the process-noise covariance Q is
    inflated by α_adapt. This allows κ and μ to shift rapidly when market
    structure changes, then slowly returns to baseline via a 0.999 decay.

    Reference: Mehra (1970) — innovation-based adaptive Kalman filtering.
    """

    def __init__(
        self,
        x0: float,
        kappa_init: float,
        mu_init: float,
        sigma2_init: float,
        gamma_adapt: float = _GAMMA_ADAPT,
        alpha_adapt: float = _ALPHA_ADAPT,
    ) -> None:
        self._z = np.array([x0, kappa_init, mu_init], dtype=float)
        self._P = np.diag([sigma2_init, 0.01, 0.01])
        self._Q = np.diag([sigma2_init, 1e-6, 1e-6])
        self._gamma = gamma_adapt
        self._alpha = alpha_adapt
        self._state_history: list[tuple[float, float, float]] = []
        self._innovations: list[float] = []

    def update(self, x_obs: float, sigma2_t: float) -> tuple[float, float, float]:
        """Single EKF prediction-update step.

        Args:
            x_obs:    observed log-price at time t
            sigma2_t: GARCH conditional variance at time t (measurement noise R_t)

        Returns:
            (X_filtered, kappa_filtered, mu_filtered)
        """
        X, kappa, mu = self._z[0], self._z[1], self._z[2]

        # -- Jacobian of OU dynamics (Euler linearisation) --
        F = np.eye(3)
        F[0, 0] = 1.0 - kappa * _DT
        F[0, 1] = (mu - X) * _DT
        F[0, 2] = kappa * _DT
        # κ_t and μ_t modelled as random walks (F[1,1]=F[2,2]=1, rest 0)

        # -- Predict --
        z_pred = F @ self._z
        P_pred = F @ self._P @ F.T + self._Q

        # -- Innovation --
        nu = x_obs - z_pred[0]
        S = P_pred[0, 0] + max(sigma2_t, _MIN_VAR)

        # -- Adaptive Q update (Mehra 1970) --
        threshold = self._gamma * np.sqrt(max(P_pred[0, 0], _MIN_VAR))
        if abs(nu) > threshold:
            self._Q = self._Q * self._alpha
        else:
            self._Q = self._Q * 0.999
        # Clip Q diagonal to positive floor
        self._Q = np.maximum(self._Q, np.diag([_MIN_VAR, 1e-12, 1e-12]))

        # -- Kalman gain and update --
        K = P_pred[:, 0] / max(S, _MIN_VAR)   # (3,) gain vector
        self._z = z_pred + K * nu

        H = np.array([[1.0, 0.0, 0.0]])
        I_KH = np.eye(3) - np.outer(K, H[0])
        self._P = I_KH @ P_pred

        # Enforce positive definiteness: symmetrise + eigenvalue-clip
        self._P = (self._P + self._P.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(self._P)
        eigvals = np.maximum(eigvals, 1e-12)
        self._P = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Bound: κ_t must stay positive
        self._z[1] = max(self._z[1], 1e-6)

        state = (float(self._z[0]), float(self._z[1]), float(self._z[2]))
        self._state_history.append(state)
        self._innovations.append(float(nu))

        return state

    def run(
        self,
        log_prices: np.ndarray,
        sigma2_path: np.ndarray,
    ) -> "AdaptiveEKF":
        """Run the AEKF over the full log-price history.

        log_prices and sigma2_path must have the same length T.
        Initialises z[0] from log_prices[0] before starting.
        """
        T = len(log_prices)
        self._z[0] = float(log_prices[0])
        for t in range(T):
            self.update(float(log_prices[t]), float(sigma2_path[t]))
        return self

    @property
    def final_state(self) -> tuple[float, float, float]:
        """(X_T, κ_T, μ_T) at the last observation."""
        if not self._state_history:
            return (self._z[0], self._z[1], self._z[2])
        return self._state_history[-1]

    @property
    def state_history(self) -> np.ndarray:
        """Array of shape (T, 3): columns are [X_t, κ_t, μ_t]."""
        if not self._state_history:
            return np.empty((0, 3))
        return np.array(self._state_history)

    @property
    def innovations(self) -> np.ndarray:
        return np.array(self._innovations)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OUKouGARCH:
    """OU-Kou-GARCH jump-diffusion model with Adaptive Extended Kalman Filter.

    Usage:
        model = OUKouGARCH()
        model.fit(returns_array)
        p = model.p_in_range(5, 0.05)         # P(|5d return| < 5%)
        direction, conf = model.direction_signal()
        diag = model.diagnostics()
        state = model.to_state_dict()          # JSON-serialisable
    """

    def __init__(self) -> None:
        self._params: OUKouGARCHParams | None = None
        self._loglik: float = float("-inf")
        self._n_obs: int = 0
        self._converged: bool = False
        self._n_iter: int = 0
        self._sigma2_path: np.ndarray | None = None   # (T,) GARCH cond. variances
        self._log_prices: np.ndarray | None = None    # (T+1,) log-prices
        self._returns: np.ndarray | None = None       # (T,) log-returns
        self._aekf: AdaptiveEKF | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self, returns: np.ndarray, prices: np.ndarray | None = None
    ) -> "OUKouGARCH":
        """Estimate OU-Kou-GARCH parameters by MLE, then run AEKF.

        Args:
            returns: array of log-returns (already computed by caller)
            prices:  optional raw price series; if provided, log-prices are
                     derived from it. If None, log-prices are cumsum of returns.

        Raises:
            ValueError if len(returns) < _MIN_OBS or if log-likelihood is not finite.
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) < _MIN_OBS:
            raise ValueError(
                f"OUKouGARCH requires ≥{_MIN_OBS} observations, got {len(returns)}"
            )

        self._n_obs = len(returns)
        self._returns = returns

        # Derive log-prices
        if prices is not None:
            lp = np.log(np.asarray(prices, dtype=float))
            # Align: log_prices[0..T], returns[0..T-1], returns[t] = lp[t+1]-lp[t]
            if len(lp) == len(returns):
                log_prices = np.concatenate([[lp[0] - returns[0]], lp])
            else:
                log_prices = lp[:len(returns) + 1]
        else:
            log_prices = np.concatenate([[0.0], np.cumsum(returns)])
        self._log_prices = log_prices   # shape (T+1,)

        # Starting values
        x0 = self._starting_values(returns, log_prices)

        # MLE via L-BFGS-B
        bounds = [
            (1e-4, 20.0),    # kappa
            (None, None),    # mu (unconstrained in log-price space)
            (1e-8, None),    # omega
            (1e-6, 0.90),    # alpha
            (1e-6, 0.90),    # beta
            (1e-4, 5.0),     # lam  (max 5 jumps/day — effectively never reached)
            (0.01, 0.99),    # p_up
            (1.0, 500.0),    # eta1
            (1.0, 500.0),    # eta2
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = optimize.minimize(
                self._neg_loglik,
                x0,
                args=(returns, log_prices),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": _MAX_ITER_MLE, "ftol": 1e-8, "gtol": 1e-6},
            )

        self._converged = bool(result.success)
        self._n_iter = int(result.nit)
        ll = float(-result.fun)

        if not np.isfinite(ll):
            raise ValueError("OUKouGARCH: MLE log-likelihood is not finite")

        self._loglik = ll
        self._params = OUKouGARCHParams(*result.x)

        # Compute GARCH variance path with fitted parameters
        self._sigma2_path = self._garch_variance_path(
            returns, self._params.omega, self._params.alpha, self._params.beta
        )

        # Run AEKF with MLE-estimated starting parameters
        self._aekf = AdaptiveEKF(
            x0=float(log_prices[0]),
            kappa_init=self._params.kappa,
            mu_init=self._params.mu,
            sigma2_init=float(self._sigma2_path[0]),
        )
        self._aekf.run(log_prices[1:], self._sigma2_path)   # log_prices[1..T]

        log.info(
            "oujump_trained",
            converged=self._converged,
            iters=self._n_iter,
            loglik=f"{ll:.2f}",
            kappa=f"{self._params.kappa:.4f}",
            mu=f"{self._params.mu:.4f}",
            lam=f"{self._params.lam:.4f}",
            persistence=f"{self._params.alpha + self._params.beta:.3f}",
            direction=self.direction_signal()[0],
        )
        return self

    def forecast_sigma_h(self, horizon: int) -> float:
        """h-step ahead combined volatility (OU + GARCH + jump contributions).

        Applies OU mean-reversion dampening to the GARCH h-step variance, then
        adds the jump variance accumulation over the horizon.

        Returns σ_h in the same units as input returns (e.g. daily log-return).
        """
        if self._params is None:
            raise RuntimeError("Must call fit() before forecast_sigma_h()")

        p = self._params
        sigma2_last = float(self._sigma2_path[-1])

        # GARCH analytic h-step variance: E[Σ_{j=1}^h σ²_{T+j}]
        persist = p.alpha + p.beta
        if persist >= 1.0 - 1e-6:
            sigma2_garch_h = sigma2_last * horizon
        else:
            # Exact recursion: Σ_{j=1}^h [ω/(1-α-β) + (σ²_last - ω/(1-α-β))·persist^{j-1}]
            # = h·ω/(1-α-β) + (σ²_last - ω/(1-α-β))·(1 - persist^h)/(1 - persist)
            uncond = p.omega / max(1.0 - persist, 1e-8)
            sigma2_garch_h = (
                horizon * uncond
                + (sigma2_last - uncond) * (1.0 - persist ** horizon) / max(1.0 - persist, 1e-8)
            )

        # OU mean-reversion compression factor ξ_h = (1 - e^{-2κh/252}) / (2κ/252)
        # For small κ (near random walk), ξ_h ≈ h/252 (no compression)
        two_kappa_h_dt = 2.0 * p.kappa * horizon * _DT
        if two_kappa_h_dt < 1e-6:
            ou_factor = horizon * _DT
        else:
            ou_factor = (1.0 - np.exp(-two_kappa_h_dt)) / (2.0 * p.kappa)

        # Scale GARCH variance by OU factor (normalised by horizon)
        sigma2_ou = max(sigma2_garch_h, _MIN_VAR) * (ou_factor / max(horizon * _DT, _MIN_VAR))

        # Jump variance: λ·(h/252)·Var(J)
        jump_var_per_jump = 2.0 * p.p_up / p.eta1 ** 2 + 2.0 * (1.0 - p.p_up) / p.eta2 ** 2
        sigma2_jump = p.lam * (horizon * _DT) * jump_var_per_jump

        return float(np.sqrt(max(sigma2_ou + sigma2_jump, _MIN_VAR)))

    def p_in_range(self, horizon: int, threshold_pct: float) -> float:
        """P(|cumulative return over horizon| < threshold) via CF inversion (FFT).

        Uses the characteristic function of X_{t+h}|X_t for the OU-Kou process:
          φ_h(u) = exp(iu·[μ+(X_T-μ)e^{-κh/252}] - u²·σ²_OU(h)/2 + λ(h/252)(E[e^{iuJ}]-1))
        """
        if self._params is None:
            raise RuntimeError("Must call fit() before p_in_range()")

        p = self._params
        X_T = float(self._log_prices[-1])
        sigma2_last = float(self._sigma2_path[-1])

        # OU variance at horizon h
        two_kappa_h_dt = 2.0 * p.kappa * horizon * _DT
        # Deep-audit VL-H: sigma2_last is a DAILY variance; the previous
        # expressions divided the h-day accumulation by an extra 1/_DT
        # (=252), making sigma_h ~16x too small and P(in range) ~= 1.0 —
        # while the CV scoring used the CORRECT path, so a member that
        # scored well contributed a constant ~1.0 live. h-day OU variance
        # = sigma2_daily * (1 - e^{-2 kappa t}) / (2 kappa dt).
        if two_kappa_h_dt < 1e-6:
            sigma2_ou_h = sigma2_last * horizon
        else:
            sigma2_ou_h = sigma2_last * (1.0 - np.exp(-two_kappa_h_dt)) / (2.0 * p.kappa * _DT)
        sigma2_ou_h = max(sigma2_ou_h, _MIN_VAR)

        # OU conditional mean at horizon h
        exp_kh = float(np.exp(-p.kappa * horizon * _DT))
        mean_h = p.mu + (X_T - p.mu) * exp_kh

        # FFT grid centred on mean_h
        grid_half = max(threshold_pct * 10.0, _FFT_RANGE)
        n = _FFT_N
        dt_x = 2.0 * grid_half / n
        du = 2.0 * np.pi / (n * dt_x)
        u_grid = (np.arange(n) - n // 2) * du   # centred frequency grid

        # Evaluate CF on u_grid (vectorised)
        cf_vals = self._ou_kou_cf(
            u_grid, p, mean_h, sigma2_ou_h, horizon
        )

        # IFFT → PDF (same scheme as ClassicalTemperedStable.fft_pdf())
        cf_shifted = np.fft.ifftshift(cf_vals)
        pdf_raw = np.fft.fftshift(np.fft.ifft(cf_shifted).real) * (n * du / (2.0 * np.pi))
        pdf_vals = np.maximum(pdf_raw, 0.0)

        # x-grid centred on mean_h
        x_grid = (np.arange(n) - n // 2) * dt_x + mean_h

        # Integrate PDF from [X_T - threshold] to [X_T + threshold]
        lo = X_T - threshold_pct
        hi = X_T + threshold_pct

        try:
            interp_fn = interpolate.interp1d(
                x_grid, pdf_vals, kind="linear",
                bounds_error=False, fill_value=0.0,
            )
            result, _ = integrate.quad(
                interp_fn, lo, hi, limit=200, epsabs=1e-4, epsrel=1e-4,
            )
            return float(np.clip(result, 0.0, 1.0))
        except Exception:
            # Fallback: Normal approximation from forecast_sigma_h
            sigma_h = self.forecast_sigma_h(horizon)
            t = threshold_pct / max(sigma_h, 1e-8)
            return float(np.clip(2.0 * stats.norm.cdf(t) - 1.0, 0.0, 1.0))

    def direction_signal(self) -> tuple[str, float]:
        """Mean-reversion direction and confidence from AEKF final state.

        Returns:
            (direction, confidence) where direction ∈ {"BULLISH", "BEARISH"}
            and confidence ∈ [0, 1].

        Logic: drift = κ_T·(μ_T - X_T)
          - drift > 0: price is below equilibrium → mean-reversion → BULLISH
          - drift < 0: price is above equilibrium → mean-reversion → BEARISH
          Magnitude normalised to rolling z-score, mapped to [0, 1] confidence.
        """
        if self._aekf is None or self._params is None:
            raise RuntimeError("Must call fit() before direction_signal()")

        X_T, kappa_T, mu_T = self._aekf.final_state
        drift = kappa_T * (mu_T - X_T)

        # Normalise using rolling z-score of historical drifts
        sh = self._aekf.state_history
        if len(sh) > 1:
            historical_drifts = sh[:, 1] * (sh[:, 2] - sh[:, 0])
            std_drift = float(np.std(historical_drifts))
        else:
            std_drift = abs(drift) if abs(drift) > 0 else 1.0
        std_drift = max(std_drift, 1e-8)

        z = float(np.clip(drift / std_drift, -3.0, 3.0))
        confidence = float(np.clip(abs(z) / 3.0, 0.0, 1.0))
        direction = "BULLISH" if drift > 0 else "BEARISH"
        return direction, confidence

    def bic(self) -> float:
        """BIC = -2·loglik + 9·log(n)."""
        if not np.isfinite(self._loglik) or self._n_obs == 0:
            return float("inf")
        return -2.0 * self._loglik + _N_PARAMS * np.log(self._n_obs)

    def diagnostics(self) -> dict:
        """Goodness-of-fit metrics computed from the fitted model.

        Returns a dict of scalar diagnostics for window JSON storage and
        cross-window analysis. Consistent with jb_pvalue / resid_skewness
        recorded by GARCHRangeModel for the arch-based variants.
        """
        if self._params is None or self._returns is None:
            return {}

        p = self._params
        result: dict = {}

        try:
            result["aic"] = round(-2.0 * self._loglik + 2.0 * _N_PARAMS, 4)
            result["bic"] = round(self.bic(), 4)
            result["loglik"] = round(self._loglik, 4)
            result["n_iter"] = self._n_iter
            result["converged"] = self._converged

            # Standardised residuals: remove OU drift + GARCH vol
            T = len(self._returns)
            lp = self._log_prices
            drifts = p.kappa * (p.mu - lp[:T]) * _DT   # (T,) OU drift
            std_resid = (self._returns - drifts) / np.sqrt(
                np.maximum(self._sigma2_path, _MIN_VAR)
            )
            std_resid = std_resid[np.isfinite(std_resid)]

            if len(std_resid) >= 8:
                _, jb_p = stats.jarque_bera(std_resid)
                result["jb_pvalue"] = round(float(jb_p), 6)
                result["resid_skewness"] = round(float(stats.skew(std_resid)), 5)
                result["resid_kurtosis"] = round(float(stats.kurtosis(std_resid)), 5)

                # Ljung-Box on squared residuals (lag=10) for remaining ARCH effects
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox
                    lb = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)
                    lb_pval = float(lb["lb_pvalue"].iloc[0])
                    result["ljung_box_pvalue"] = round(lb_pval, 6)
                except Exception:
                    result["ljung_box_pvalue"] = float("nan")
            else:
                result["jb_pvalue"] = float("nan")
                result["resid_skewness"] = float("nan")
                result["resid_kurtosis"] = float("nan")
                result["ljung_box_pvalue"] = float("nan")

            # Interpretable OU parameters
            result["ou_half_life_days"] = round(252.0 * np.log(2.0) / max(p.kappa, 1e-8), 2)
            result["jump_intensity_annual"] = round(float(p.lam * 252), 4)
            result["jump_mean_up_pct"] = round(100.0 / p.eta1, 4)
            result["jump_mean_down_pct"] = round(100.0 / p.eta2, 4)
            result["diffusion_persistence"] = round(float(p.alpha + p.beta), 5)

            # AEKF parameter stability
            if self._aekf is not None:
                sh = self._aekf.state_history
                if len(sh) > 1:
                    kappa_hist = sh[:, 1]
                    mu_hist = sh[:, 2]
                    kappa_mean = float(np.mean(kappa_hist))
                    kappa_std = float(np.std(kappa_hist))
                    result["aekf_kappa_cv"] = round(
                        kappa_std / max(kappa_mean, 1e-8), 4
                    )
                    mu_range_pct = (float(np.max(mu_hist)) - float(np.min(mu_hist))) * 100.0
                    result["aekf_mu_range"] = round(mu_range_pct, 4)
                else:
                    result["aekf_kappa_cv"] = float("nan")
                    result["aekf_mu_range"] = float("nan")

        except Exception as e:
            log.debug("oujump_diagnostics_failed", error=str(e)[:80])

        return result

    def oos_aekf_diagnostics(
        self,
        oos_returns: np.ndarray,
        horizon_days: int,
        threshold_pct: float,
    ) -> dict:
        """Step the AEKF forward over OOS returns using frozen MLE params.

        Produces per-day innovations, κ path, direction drifts, and realized
        volatility — the raw material for Brier score, direction AUROC, and
        AEKF stability diagnostics in _evaluate_range_model_oos().

        Args:
            oos_returns:   1-D array of OOS log-returns (length T_oos)
            horizon_days:  forecast horizon used to compute realized vol labels
            threshold_pct: ±range threshold used to compute binary range labels

        Returns dict with:
            innovations     — (T_oos,) Kalman innovations ν_t = x_t - x̂_t|t-1
            kappa_path      — (T_oos,) filtered κ_t over OOS
            mu_path         — (T_oos,) filtered μ_t over OOS
            drift_path      — (T_oos,) κ_t·(μ_t - X_t), the direction signal
            sigma_path      — (T_oos,) conditional vol σ_t (GARCH recursion)
            p_scores        — (T_labelable,) P(in range) per labelable day
            y_realized      — (T_labelable,) 1 if |cum-return| < threshold else 0
            rvol_realized   — (T_labelable,) realized σ over next horizon_days
            lb_pvalue       — Ljung-Box p-value on ν_t² (ARCH in innovations)
            lb_acf_pvalue   — Ljung-Box p-value on ν_t (autocorrelation)
        """
        if self._params is None:
            return {}

        oos_returns = np.asarray(oos_returns, dtype=float)
        n = len(oos_returns)
        if n < horizon_days + 1:
            return {}

        p = self._params

        # Initialise AEKF from training end-state
        X_T, kappa_T, mu_T = self._aekf.final_state if self._aekf else (
            float(self._log_prices[-1]) if self._log_prices is not None else p.mu,
            p.kappa, p.mu,
        )
        uncond_var = p.omega / max(1.0 - p.alpha - p.beta, 1e-6)
        sigma2_last = float(self._sigma2_path[-1]) if self._sigma2_path is not None else uncond_var

        aekf = AdaptiveEKF(
            x0=X_T, kappa_init=kappa_T, mu_init=mu_T,
            sigma2_init=sigma2_last,
        )
        aekf._z = np.array([X_T, kappa_T, mu_T])

        innovations  = np.empty(n)
        kappa_path   = np.empty(n)
        mu_path      = np.empty(n)
        drift_path   = np.empty(n)
        sigma_path   = np.empty(n)

        # Reconstruct cumulative log-prices from OOS returns
        log_prices_oos = X_T + np.concatenate([[0.0], np.cumsum(oos_returns)])

        sigma2_t = sigma2_last
        prev_r2  = oos_returns[0] ** 2 if n > 0 else sigma2_last

        for t in range(n):
            sigma2_t = float(
                p.omega + p.alpha * prev_r2 + p.beta * sigma2_t
            )
            sigma2_t = max(sigma2_t, _MIN_VAR)
            prev_r2  = oos_returns[t] ** 2

            x_obs = float(log_prices_oos[t + 1])
            X_f, k_f, m_f = aekf.update(x_obs, sigma2_t)

            innovations[t] = aekf._innovations[-1]
            kappa_path[t]  = k_f
            mu_path[t]     = m_f
            drift_path[t]  = k_f * (m_f - x_obs)
            sigma_path[t]  = float(np.sqrt(sigma2_t))

        # Build labelable slice (can assign horizon_days forward labels)
        n_labelable = n - horizon_days
        p_scores      = np.empty(n_labelable)
        y_realized    = np.empty(n_labelable, dtype=int)
        rvol_realized = np.empty(n_labelable)

        for t in range(n_labelable):
            # P(in range): Normal approx from GARCH+OU h-step sigma
            persist = p.alpha + p.beta
            if persist >= 1.0 - 1e-6:
                sigma2_h = sigma_path[t] ** 2 * horizon_days
            else:
                uncond = p.omega / max(1.0 - persist, 1e-8)
                sigma2_h = (
                    horizon_days * uncond
                    + (sigma_path[t] ** 2 - uncond)
                    * (1.0 - persist ** horizon_days)
                    / max(1.0 - persist, 1e-8)
                )
            two_kappa_h = 2.0 * kappa_path[t] * horizon_days * _DT
            if two_kappa_h < 1e-6:
                ou_f = horizon_days * _DT
            else:
                ou_f = (1.0 - np.exp(-two_kappa_h)) / (2.0 * kappa_path[t])

            jump_var = p.lam * (horizon_days * _DT) * (
                2.0 * p.p_up / p.eta1 ** 2 + 2.0 * (1.0 - p.p_up) / p.eta2 ** 2
            )
            sigma_h = float(np.sqrt(max(
                sigma2_h * (ou_f / max(horizon_days * _DT, _MIN_VAR)) + jump_var,
                _MIN_VAR,
            )))
            t_std = threshold_pct / max(sigma_h, 1e-8)
            p_scores[t] = float(np.clip(2.0 * stats.norm.cdf(t_std) - 1.0, 0.0, 1.0))

            # Realized label and vol
            fwd = oos_returns[t + 1: t + 1 + horizon_days]
            cum_ret = float(np.sum(fwd))
            y_realized[t]    = 1 if abs(cum_ret) < threshold_pct else 0
            rvol_realized[t] = float(np.std(fwd)) * np.sqrt(252) if len(fwd) > 1 else float(np.nan)

        # Innovation diagnostics
        lb_pvalue = lb_acf_pvalue = float("nan")
        if len(innovations) >= 10:
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                lb_sq  = acorr_ljungbox(innovations ** 2, lags=[10], return_df=True)
                lb_acf = acorr_ljungbox(innovations,      lags=[10], return_df=True)
                lb_pvalue     = float(lb_sq["lb_pvalue"].iloc[0])
                lb_acf_pvalue = float(lb_acf["lb_pvalue"].iloc[0])
            except Exception:
                pass

        return {
            "innovations":     innovations,
            "kappa_path":      kappa_path,
            "mu_path":         mu_path,
            "drift_path":      drift_path,
            "sigma_path":      sigma_path,
            "p_scores":        p_scores,
            "y_realized":      y_realized,
            "rvol_realized":   rvol_realized,
            "lb_pvalue":       lb_pvalue,
            "lb_acf_pvalue":   lb_acf_pvalue,
        }

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    def to_state_dict(self) -> dict:
        """JSON-serialisable summary for window storage.

        Never includes numpy arrays, live objects, or non-serialisable types.
        """
        if self._params is None:
            return {"converged": False}

        p = self._params
        direction, dir_conf = self.direction_signal()
        X_T, kappa_T, mu_T = self._aekf.final_state if self._aekf else (0.0, p.kappa, p.mu)

        return {
            "converged":            self._converged,
            "n_iter":               self._n_iter,
            "loglik":               round(self._loglik, 4),
            "bic":                  round(self.bic(), 4),
            "n_obs":                self._n_obs,
            "params": {
                "kappa":   round(float(p.kappa), 6),
                "mu":      round(float(p.mu), 6),
                "omega":   round(float(p.omega), 8),
                "alpha":   round(float(p.alpha), 6),
                "beta":    round(float(p.beta), 6),
                "lambda":  round(float(p.lam), 6),
                "p_up":    round(float(p.p_up), 6),
                "eta1":    round(float(p.eta1), 4),
                "eta2":    round(float(p.eta2), 4),
            },
            "aekf_final_state": {
                "X_T":     round(float(X_T), 6),
                "kappa_T": round(float(kappa_T), 6),
                "mu_T":    round(float(mu_T), 6),
            },
            "direction":             direction,
            "direction_confidence":  round(dir_conf, 6),
            "diagnostics":           self.diagnostics(),
        }

    # ------------------------------------------------------------------
    # Private: log-likelihood
    # ------------------------------------------------------------------

    def _neg_loglik(
        self,
        theta: np.ndarray,
        returns: np.ndarray,
        log_prices: np.ndarray,
    ) -> float:
        """Negative MLE log-likelihood for L-BFGS-B minimisation.

        Uses Gaussian moment-matching for the k-fold jump convolution:
        For k jumps, the conditional return distribution is approximately
        N(drift + k·jump_mean, σ²_t + k·jump_var) by cumulant additivity.
        """
        kappa, mu, omega, alpha, beta, lam, p_up, eta1, eta2 = theta

        # Feasibility checks (bounds enforce most, but guard α+β inside)
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        if kappa <= 0 or lam <= 0 or eta1 <= 0 or eta2 <= 0:
            return 1e10
        if not (0.0 < p_up < 1.0):
            return 1e10

        T = len(returns)
        sigma2_path = self._garch_variance_path(returns, omega, alpha, beta)

        # Kou jump moments
        jump_mean = p_up / eta1 - (1.0 - p_up) / eta2
        jump_var = 2.0 * p_up / eta1 ** 2 + 2.0 * (1.0 - p_up) / eta2 ** 2

        lam_dt = lam * _DT
        loglik = 0.0

        for t in range(T):
            X_prev = float(log_prices[t])
            drift = kappa * (mu - X_prev) * _DT
            r_t = returns[t]
            sigma2_t = max(float(sigma2_path[t]), _MIN_VAR)

            mixture_density = 0.0
            for k in range(_K_MAX_JUMPS + 1):
                pk = float(stats.poisson.pmf(k, lam_dt))
                if pk < 1e-15:
                    continue
                mu_k = drift + k * jump_mean
                var_k = max(sigma2_t + k * jump_var, _MIN_VAR)
                mixture_density += pk * float(stats.norm.pdf(r_t, loc=mu_k, scale=np.sqrt(var_k)))

            loglik += np.log(max(mixture_density, 1e-300))

        return float(-loglik)

    # ------------------------------------------------------------------
    # Private: characteristic function for CF inversion
    # ------------------------------------------------------------------

    @staticmethod
    def _ou_kou_cf(
        u: np.ndarray,
        params: OUKouGARCHParams,
        mean_h: float,
        sigma2_ou_h: float,
        horizon: int,
    ) -> np.ndarray:
        """Characteristic function of X_{t+h}|X_t for the OU-Kou process.

        φ_h(u) = exp(iu·mean_h - u²·σ²_OU(h)/2 + λ·(h/252)·(E[e^{iuJ}]-1))

        Kou (2002) jump CF:  E[e^{iuJ}] = p_up·η₁/(η₁-iu) + (1-p_up)·η₂/(η₂+iu)
        """
        p = params
        u = np.asarray(u, dtype=complex)

        # Diffusion + drift term
        log_cf = 1j * u * mean_h - 0.5 * u ** 2 * sigma2_ou_h

        # Compound Poisson jump term: λ(h/252)(E[e^{iuJ}] - 1)
        lam_h = p.lam * horizon * _DT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Kou CF: rational function of u (numerically stable)
            kou_cf = (
                p.p_up * p.eta1 / (p.eta1 - 1j * u)
                + (1.0 - p.p_up) * p.eta2 / (p.eta2 + 1j * u)
            )
        log_cf += lam_h * (kou_cf - 1.0)

        return np.exp(log_cf)

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _garch_variance_path(
        returns: np.ndarray, omega: float, alpha: float, beta: float
    ) -> np.ndarray:
        """GARCH(1,1) conditional variance recursion.

        Exact copy of the pattern in msgarch.py for consistency.
        Initialised from unconditional variance; floored at _MIN_VAR.
        """
        T = len(returns)
        var = np.empty(T)
        denom = 1.0 - alpha - beta
        var[0] = omega / max(denom, 1e-6) if denom > 0 else omega * 10
        var[0] = max(var[0], _MIN_VAR)
        for t in range(1, T):
            var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
            var[t] = max(var[t], _MIN_VAR)
        return var

    def _starting_values(
        self, returns: np.ndarray, log_prices: np.ndarray
    ) -> np.ndarray:
        """Two-stage starting values for MLE.

        Stage 1: OLS regression X_t ~ a + b·X_{t-1} to initialise κ, μ.
        Stage 2: Residual moments to initialise GARCH and jump parameters.
        """
        T = len(returns)
        lp = log_prices[:T]   # X_{t-1}

        # OLS: X_t = a + b·X_{t-1}  →  b = e^{-κ·dt}, a = μ·(1-b)
        if T >= 10:
            try:
                X_prev = lp
                X_curr = log_prices[1:T + 1]
                b_ols = float(np.cov(X_curr, X_prev)[0, 1] / max(np.var(X_prev), 1e-12))
                b_ols = float(np.clip(b_ols, 0.01, 0.9999))
                kappa_0 = float(-np.log(b_ols) / _DT)
                a_ols = float(np.mean(X_curr) - b_ols * np.mean(X_prev))
                mu_0 = a_ols / max(1.0 - b_ols, 1e-8)
            except Exception:
                kappa_0 = 0.10
                mu_0 = float(np.mean(log_prices))
        else:
            kappa_0 = 0.10
            mu_0 = float(np.mean(log_prices))

        kappa_0 = float(np.clip(kappa_0, 1e-4, 10.0))
        mu_0 = float(np.clip(mu_0, log_prices.min() - 1.0, log_prices.max() + 1.0))

        # GARCH starting values from sample variance
        sigma2_0 = max(float(np.var(returns)), 1e-8)
        alpha_0, beta_0 = 0.05, 0.90
        omega_0 = max(sigma2_0 * (1.0 - alpha_0 - beta_0), 1e-8)

        # Jump starting values: use excess kurtosis to estimate jump intensity
        kurt = float(stats.kurtosis(returns))   # excess kurtosis
        # For Kou model: excess kurtosis ≈ λ·(excess_kurtosis_per_jump)
        # With symmetric params (p_up=0.5, η₁=η₂=50): kurtosis_per_jump = 24
        # λ ≈ kurt / 24 (rough init), clamped to reasonable range
        lam_0 = float(np.clip(max(kurt, 0.0) / 24.0, 0.01, 0.5))

        return np.array([kappa_0, mu_0, omega_0, alpha_0, beta_0,
                         lam_0, 0.50, 50.0, 50.0])

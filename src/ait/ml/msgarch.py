"""Markov-Switching GARCH(1,1) — 2-regime volatility model.

Implements the Gray (1996) / Haas et al. (2004) MS-GARCH model with:
  - 2 independent regimes (calm / stressed), each with own GARCH(1,1) dynamics
  - Hamilton (1989) filter for regime probability inference
  - Kim (1994) backward smoother for full smoothed probabilities
  - EM estimation: E-step (filter+smoother) alternates with M-step (WLS per regime)

Motivation: single-regime GARCH extrapolates volatility persistence into
mean-reversion regimes (post-shock recoveries), causing rank-inversion of
P(in range) predictions. MS-GARCH assigns high probability to the calm regime
after a shock dissipates, correctly predicting lower future vol.

See docs/GARCH_METHODOLOGY.md §6 for full mathematical derivation and references.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from scipy import optimize, stats

from ait.utils.logging import get_logger

log = get_logger("ml.msgarch")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_REGIMES = 2
_N_PARAMS = 8          # ω₀,α₀,β₀, ω₁,α₁,β₁, p₀₀, p₁₁
_MIN_OBS = 60          # minimum returns for reliable EM estimation
_MAX_EM_ITER = 200
_EM_TOL = 1e-6         # log-likelihood convergence tolerance
_MIN_VAR = 1e-10       # floor for variance to avoid division by zero


# ---------------------------------------------------------------------------
# Named tuple for fitted state
# ---------------------------------------------------------------------------

class MSGARCHParams(NamedTuple):
    omega: list[float]      # [ω₀, ω₁]
    alpha: list[float]      # [α₀, α₁]
    beta:  list[float]      # [β₀, β₁]
    p00:   float            # P(S_t=0 | S_{t-1}=0)
    p11:   float            # P(S_t=1 | S_{t-1}=1)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MarkovSwitchingGARCH:
    """2-regime MS-GARCH(1,1) estimated by EM with Hamilton filter.

    Usage:
        ms = MarkovSwitchingGARCH()
        ms.fit(returns_array)          # returns % — already scaled
        p = ms.p_in_range(21, 0.05)   # P(|21d return| < 5%)
        bic = ms.bic(len(returns))
    """

    def __init__(self) -> None:
        self._params: MSGARCHParams | None = None
        self._loglik: float = float("-inf")
        self._smoothed_probs: np.ndarray | None = None   # (T, 2)
        self._filtered_probs: np.ndarray | None = None   # (T, 2)
        self._var_paths: np.ndarray | None = None        # (T, 2) σ²_{t|k}
        self._n_obs: int = 0
        self._converged: bool = False
        self._n_iter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> "MarkovSwitchingGARCH":
        """Fit MS-GARCH via EM. Returns self (for chaining).

        Raises ValueError if convergence fails or data is too short.
        Caller should wrap in try/except.
        """
        returns = np.asarray(returns, dtype=float)
        if len(returns) < _MIN_OBS:
            raise ValueError(
                f"MS-GARCH requires ≥{_MIN_OBS} observations, got {len(returns)}"
            )

        self._n_obs = len(returns)
        params = self._starting_values(returns)

        prev_ll = float("-inf")
        for i in range(_MAX_EM_ITER):
            # E-step: Hamilton filter + Kim smoother
            filtered, ll = self._hamilton_filter(returns, params)
            smoothed = self._smoother(filtered, params)

            # M-step: update GARCH params per regime and transition matrix
            try:
                params = self._m_step(returns, smoothed, params)
            except Exception as e:
                log.debug("msgarch_m_step_failed", iter=i, error=str(e)[:60])
                break

            # Convergence check
            if abs(ll - prev_ll) < _EM_TOL:
                self._converged = True
                self._n_iter = i + 1
                break
            prev_ll = ll

        if not self._converged:
            log.debug("msgarch_not_converged", iters=_MAX_EM_ITER, delta=abs(ll - prev_ll))
            # Accept result anyway if log-likelihood is finite
            if not np.isfinite(ll):
                raise ValueError("MS-GARCH did not converge and log-likelihood is not finite")
            self._n_iter = _MAX_EM_ITER

        self._params = params
        self._loglik = ll
        self._filtered_probs = filtered
        self._smoothed_probs = smoothed
        self._var_paths = self._compute_var_paths(returns, params)

        log.info(
            "msgarch_trained",
            converged=self._converged,
            iters=self._n_iter,
            loglik=f"{ll:.2f}",
            regime0=f"ω={params.omega[0]:.6f} α={params.alpha[0]:.3f} β={params.beta[0]:.3f}",
            regime1=f"ω={params.omega[1]:.6f} α={params.alpha[1]:.3f} β={params.beta[1]:.3f}",
            transitions=f"p00={params.p00:.3f} p11={params.p11:.3f}",
        )
        return self

    def forecast_sigma_h(self, horizon: int) -> float:
        """h-step ahead conditional vol (annualised pct) combining both regimes.

        Propagates regime probabilities forward via transition matrix, then
        compounds per-regime GARCH variance forecasts.

        Returns σ_h in the same units as the input returns (e.g. % per day).
        """
        if self._params is None:
            raise RuntimeError("Must call fit() before forecast_sigma_h()")

        p = self._params
        # Current regime probs from last filtered step
        pi = self._filtered_probs[-1].copy()  # shape (2,)

        # Transition matrix rows: [p00, 1-p00], [1-p11, p11]
        P = np.array([[p.p00, 1 - p.p00],
                      [1 - p.p11, p.p11]])

        # Last conditional variances per regime
        var = self._var_paths[-1].copy()  # shape (2,)

        total_var = 0.0
        for h in range(1, horizon + 1):
            pi = P.T @ pi          # propagate regime probs h steps
            pi = np.clip(pi, 0.0, 1.0)
            pi /= pi.sum()

            # Per-regime one-step GARCH variance forecast
            # E[σ²_{t+h|k}] = ω_k + (α_k+β_k)·E[σ²_{t+h-1|k}]
            var_next = np.array([
                p.omega[k] + (p.alpha[k] + p.beta[k]) * var[k]
                for k in range(_N_REGIMES)
            ])
            var_next = np.maximum(var_next, _MIN_VAR)

            # Combined variance at step h (weighted by regime probs)
            total_var += float(pi @ var_next)
            var = var_next

        sigma_h = np.sqrt(max(total_var, _MIN_VAR))
        return float(sigma_h)

    def p_in_range(self, horizon: int, threshold_pct: float) -> float:
        """P(|cumulative return over horizon| < threshold) under fitted MS-GARCH.

        Uses Normal innovation distribution (symmetric).
        threshold_pct is in the same units as returns (e.g. 0.05 = 5%).
        """
        sigma_h = self.forecast_sigma_h(horizon)
        if sigma_h <= 0 or not np.isfinite(sigma_h):
            return 0.5
        t = threshold_pct / sigma_h
        return float(np.clip(2.0 * stats.norm.cdf(t) - 1.0, 0.0, 1.0))

    def bic(self) -> float:
        """BIC = −2·loglik + k·log(n), k=8 parameters."""
        if not np.isfinite(self._loglik) or self._n_obs == 0:
            return float("inf")
        return -2.0 * self._loglik + _N_PARAMS * np.log(self._n_obs)

    @property
    def is_fitted(self) -> bool:
        return self._params is not None

    @property
    def regime0_params(self) -> dict:
        if self._params is None:
            return {}
        return {
            "omega": round(self._params.omega[0], 8),
            "alpha": round(self._params.alpha[0], 5),
            "beta":  round(self._params.beta[0], 5),
            "persistence": round(self._params.alpha[0] + self._params.beta[0], 5),
        }

    @property
    def regime1_params(self) -> dict:
        if self._params is None:
            return {}
        return {
            "omega": round(self._params.omega[1], 8),
            "alpha": round(self._params.alpha[1], 5),
            "beta":  round(self._params.beta[1], 5),
            "persistence": round(self._params.alpha[1] + self._params.beta[1], 5),
        }

    @property
    def transition_matrix(self) -> dict:
        if self._params is None:
            return {}
        return {
            "p00": round(self._params.p00, 5),
            "p11": round(self._params.p11, 5),
            "p01": round(1 - self._params.p00, 5),
            "p10": round(1 - self._params.p11, 5),
        }

    @property
    def final_regime_probs(self) -> list[float]:
        """Filtered regime probabilities at the last observation."""
        if self._filtered_probs is None:
            return [0.5, 0.5]
        return [round(float(p), 5) for p in self._filtered_probs[-1]]

    def to_state_dict(self) -> dict:
        """Serialisable summary for window JSON storage."""
        if self._params is None:
            return {"converged": False}
        return {
            "converged":         self._converged,
            "n_iter":            self._n_iter,
            "loglik":            round(self._loglik, 4),
            "bic":               round(self.bic(), 4),
            "regime0":           self.regime0_params,
            "regime1":           self.regime1_params,
            "transition":        self.transition_matrix,
            "final_probs":       self.final_regime_probs,
        }

    def load_from_state_dict(self, state: dict) -> bool:
        """Reconstruct a fitted MS-GARCH from a to_state_dict() snapshot.

        Used by roll_msgarch_forecasts() to avoid re-fitting on OOS data.
        Returns True on success, False if the state dict is incomplete.
        """
        try:
            r0 = state.get("regime0") or {}
            r1 = state.get("regime1") or {}
            tr = state.get("transition") or {}
            fp = state.get("final_probs") or [0.5, 0.5]

            self._params = MSGARCHParams(
                omega=[float(r0.get("omega", 1e-6)), float(r1.get("omega", 1e-6))],
                alpha=[float(r0.get("alpha", 0.05)), float(r1.get("alpha", 0.10))],
                beta= [float(r0.get("beta",  0.90)), float(r1.get("beta",  0.85))],
                p00=  float(tr.get("p00", 0.97)),
                p11=  float(tr.get("p11", 0.90)),
            )
            # Initialise filter state at training end-state
            self._current_filtered = np.array([float(fp[0]), float(fp[1])], dtype=float)
            # Initialise per-regime last variance to unconditional variance
            p = self._params
            self._sigma2_last = np.array([
                p.omega[k] / max(1.0 - p.alpha[k] - p.beta[k], 1e-6)
                for k in range(_N_REGIMES)
            ])
            self._loglik   = float(state.get("loglik", 0.0))
            self._converged = bool(state.get("converged", False))
            self._n_iter   = int(state.get("n_iter", 0))
            return True
        except Exception:
            return False

    def step_filter(self, r: float) -> None:
        """Update Hamilton filter state by one OOS observation.

        Updates ``_current_filtered`` (regime probabilities) and
        ``_sigma2_last`` (per-regime conditional variances) using
        frozen MLE parameters.  Called repeatedly over OOS returns
        by roll_msgarch_forecasts().
        """
        if self._params is None or not hasattr(self, "_current_filtered"):
            return

        p = self._params
        P = np.array([[p.p00, 1 - p.p00],
                      [1 - p.p11, p.p11]])

        # Update per-regime conditional variance: σ²_{t|k} = ω_k + α_k·r² + β_k·σ²_{t-1|k}
        r2 = r * r
        self._sigma2_last = np.array([
            max(p.omega[k] + p.alpha[k] * r2 + p.beta[k] * self._sigma2_last[k], _MIN_VAR)
            for k in range(_N_REGIMES)
        ])

        # Predict regime: π_t = P.T @ π_{t-1}
        predicted = P.T @ self._current_filtered
        predicted = np.clip(predicted, 1e-300, None)

        # Update with likelihood
        densities = np.array([
            self._normal_density(r, self._sigma2_last[k]) for k in range(_N_REGIMES)
        ])
        joint = densities * predicted
        total = joint.sum()
        if total > 0 and np.isfinite(total):
            self._current_filtered = joint / total
        else:
            self._current_filtered = predicted

    def p_in_range_from_filter(self, horizon: int, threshold_pct: float) -> float:
        """P(in range) using the current step_filter regime state.

        Used in roll_msgarch_forecasts() after stepping the filter forward
        through OOS observations.
        """
        if self._params is None or not hasattr(self, "_current_filtered"):
            return 0.5
        if not hasattr(self, "_sigma2_last"):
            return self.p_in_range(horizon, threshold_pct)

        p = self._params
        pi = self._current_filtered

        # h-step variance per regime via analytic GARCH recursion
        sigma2_h = 0.0
        for k in range(_N_REGIMES):
            persist = p.alpha[k] + p.beta[k]
            uncond_k = p.omega[k] / max(1.0 - persist, 1e-6)
            if persist >= 1.0 - 1e-6:
                var_k = self._sigma2_last[k] * horizon
            else:
                var_k = (
                    horizon * uncond_k
                    + (self._sigma2_last[k] - uncond_k)
                    * (1.0 - persist ** horizon)
                    / max(1.0 - persist, 1e-8)
                )
            sigma2_h += float(pi[k]) * max(var_k, _MIN_VAR)

        sigma_h = float(np.sqrt(max(sigma2_h, _MIN_VAR)))
        t_std = threshold_pct / max(sigma_h, 1e-8)
        return float(np.clip(2.0 * stats.norm.cdf(t_std) - 1.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Private: Hamilton filter (E-step forward pass)
    # ------------------------------------------------------------------

    def _hamilton_filter(
        self, returns: np.ndarray, params: MSGARCHParams
    ) -> tuple[np.ndarray, float]:
        """Hamilton (1989) forward filter.

        Returns:
            filtered: (T, 2) array of P(S_t=k | r_1..r_t)
            loglik:   scalar total log-likelihood
        """
        T = len(returns)
        P = np.array([[params.p00, 1 - params.p00],
                      [1 - params.p11, params.p11]])

        var_paths = self._compute_var_paths(returns, params)  # (T, 2)

        filtered = np.zeros((T, _N_REGIMES))
        loglik = 0.0

        # Initial regime probs: stationary distribution of transition matrix
        pi = self._stationary_probs(params)

        for t in range(T):
            r = returns[t]
            # Conditional densities under each regime
            densities = np.array([
                self._normal_density(r, var_paths[t, k])
                for k in range(_N_REGIMES)
            ])

            # Predict: P(S_t=k | r_{t-1}) = Σ_j P(S_t=k|S_{t-1}=j) · P(S_{t-1}=j|r_{t-1})
            if t == 0:
                predicted = pi.copy()
            else:
                predicted = P.T @ filtered[t - 1]
            predicted = np.clip(predicted, 1e-300, None)

            # Update: P(S_t=k|r_t) ∝ f(r_t|S_t=k) · P(S_t=k|r_{t-1})
            joint = densities * predicted
            total = joint.sum()
            if total <= 0 or not np.isfinite(total):
                filtered[t] = predicted
                loglik += -1e10  # penalise degenerate steps
            else:
                filtered[t] = joint / total
                loglik += np.log(total)

        return filtered, float(loglik)

    # ------------------------------------------------------------------
    # Private: Kim (1994) smoother (E-step backward pass)
    # ------------------------------------------------------------------

    def _smoother(
        self, filtered: np.ndarray, params: MSGARCHParams
    ) -> np.ndarray:
        """Kim (1994) backward smoother.

        Returns smoothed: (T, 2) array of P(S_t=k | r_1..r_T)
        """
        T = len(filtered)
        P = np.array([[params.p00, 1 - params.p00],
                      [1 - params.p11, params.p11]])

        smoothed = np.zeros((T, _N_REGIMES))
        smoothed[-1] = filtered[-1].copy()

        for t in range(T - 2, -1, -1):
            predicted_next = P.T @ filtered[t]
            predicted_next = np.clip(predicted_next, 1e-300, None)

            for k in range(_N_REGIMES):
                ratio = smoothed[t + 1] / predicted_next
                smoothed[t, k] = filtered[t, k] * np.dot(P[k], ratio)

            # Normalise
            total = smoothed[t].sum()
            if total > 0:
                smoothed[t] /= total
            else:
                smoothed[t] = filtered[t]

        return smoothed

    # ------------------------------------------------------------------
    # Private: M-step
    # ------------------------------------------------------------------

    def _m_step(
        self,
        returns: np.ndarray,
        smoothed: np.ndarray,
        prev_params: MSGARCHParams,
    ) -> MSGARCHParams:
        """Update all parameters using smoothed regime probabilities.

        For each regime k, minimise weighted negative log-likelihood with
        weights = smoothed_probs[:, k].
        Transition matrix updated from smoothed joint probs.
        """
        new_omega = list(prev_params.omega)
        new_alpha = list(prev_params.alpha)
        new_beta  = list(prev_params.beta)

        for k in range(_N_REGIMES):
            weights = smoothed[:, k]
            if weights.sum() < 1.0:
                continue  # too few effective observations for this regime

            result = self._fit_garch_weighted(returns, weights, prev_params, k)
            if result is not None:
                new_omega[k], new_alpha[k], new_beta[k] = result

        # Update transition probabilities from smoothed joint probs
        # P(S_{t+1}=j, S_t=i | Y) via Kim smoother approximation
        new_p00 = self._update_transition(smoothed, from_regime=0, to_regime=0)
        new_p11 = self._update_transition(smoothed, from_regime=1, to_regime=1)

        return MSGARCHParams(
            omega=new_omega, alpha=new_alpha, beta=new_beta,
            p00=new_p00, p11=new_p11,
        )

    def _fit_garch_weighted(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
        prev_params: MSGARCHParams,
        regime: int,
    ) -> tuple[float, float, float] | None:
        """Weighted MLE for GARCH(1,1) params of one regime.

        Minimise Σ_t w_t · [-0.5·log(σ²_t) - 0.5·r²_t/σ²_t]
        subject to ω > 0, α ≥ 0, β ≥ 0, α+β < 1.
        """
        x0 = np.array([
            prev_params.omega[regime],
            prev_params.alpha[regime],
            prev_params.beta[regime],
        ])

        def neg_wll(p: np.ndarray) -> float:
            omega, alpha, beta = p
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
                return 1e10
            var = self._garch_variance_path(returns, omega, alpha, beta)
            var = np.maximum(var, _MIN_VAR)
            ll_terms = -0.5 * (np.log(var) + returns ** 2 / var)
            return -float(np.dot(weights, ll_terms))

        bounds = [(1e-8, None), (1e-6, 0.9), (1e-6, 0.9)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = optimize.minimize(
                neg_wll, x0, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 100, "ftol": 1e-8},
            )

        if res.success and np.isfinite(res.fun):
            omega, alpha, beta = res.x
            if omega > 0 and alpha + beta < 1.0:
                return float(omega), float(alpha), float(beta)
        return None

    def _update_transition(
        self, smoothed: np.ndarray, from_regime: int, to_regime: int
    ) -> float:
        """Estimate P(S_{t+1}=to | S_t=from) from smoothed marginals.

        Uses the approximate formula: sum over t of joint smoothed probs.
        """
        T = len(smoothed)
        if T < 2:
            return 0.97 if from_regime == to_regime else 0.03

        # Approximate joint: P(S_{t+1}=j, S_t=i) ≈ P(S_t=i) · P(S_{t+1}=j|S_t=i)
        # Full Kim smoother would compute exact joint probs; here use marginals
        numerator = 0.0
        denominator = 0.0
        for t in range(T - 1):
            denominator += smoothed[t, from_regime]
            # Contribution toward `to_regime` at t+1
            numerator += smoothed[t, from_regime] * smoothed[t + 1, to_regime]

        if denominator < 1e-10:
            return 0.97 if from_regime == to_regime else 0.03

        val = numerator / denominator
        return float(np.clip(val, 0.01, 0.99))

    # ------------------------------------------------------------------
    # Private: utilities
    # ------------------------------------------------------------------

    def _compute_var_paths(
        self, returns: np.ndarray, params: MSGARCHParams
    ) -> np.ndarray:
        """Compute conditional variance paths for both regimes. Returns (T, 2)."""
        T = len(returns)
        var = np.zeros((T, _N_REGIMES))
        for k in range(_N_REGIMES):
            var[:, k] = self._garch_variance_path(
                returns, params.omega[k], params.alpha[k], params.beta[k]
            )
        return var

    @staticmethod
    def _garch_variance_path(
        returns: np.ndarray, omega: float, alpha: float, beta: float
    ) -> np.ndarray:
        """Compute σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} recursion."""
        T = len(returns)
        var = np.empty(T)
        # Unconditional variance as initial value
        denom = 1.0 - alpha - beta
        var[0] = omega / max(denom, 1e-6) if denom > 0 else omega * 10
        var[0] = max(var[0], _MIN_VAR)
        for t in range(1, T):
            var[t] = omega + alpha * returns[t - 1] ** 2 + beta * var[t - 1]
            var[t] = max(var[t], _MIN_VAR)
        return var

    @staticmethod
    def _normal_density(r: float, var: float) -> float:
        """Gaussian density f(r | σ²) = (2πσ²)^{-0.5} · exp(-r²/2σ²)."""
        var = max(var, _MIN_VAR)
        return float(stats.norm.pdf(r, scale=np.sqrt(var)))

    def _stationary_probs(self, params: MSGARCHParams) -> np.ndarray:
        """Stationary distribution of 2-state Markov chain."""
        p01 = 1.0 - params.p00
        p10 = 1.0 - params.p11
        denom = p01 + p10
        if denom < 1e-10:
            return np.array([0.5, 0.5])
        return np.array([p10 / denom, p01 / denom])

    def _starting_values(self, returns: np.ndarray) -> MSGARCHParams:
        """Initialise parameters: regime 0 = low vol, regime 1 = high vol.

        Strategy:
        - Sort |returns| and split at median to define calm / stressed days
        - Fit simple variance estimates per group
        - Use moderate GARCH params, high persistence in stressed regime
        """
        r2 = returns ** 2
        median_r2 = float(np.median(r2))

        calm_var = float(np.mean(r2[r2 <= median_r2])) or 1e-4
        stressed_var = float(np.mean(r2[r2 > median_r2])) or 1e-3

        # ω = unconditional_var × (1 − α − β)
        alpha0, beta0 = 0.05, 0.70   # calm: low persistence
        alpha1, beta1 = 0.10, 0.85   # stressed: high persistence
        omega0 = max(calm_var * (1 - alpha0 - beta0), 1e-8)
        omega1 = max(stressed_var * (1 - alpha1 - beta1), 1e-8)

        return MSGARCHParams(
            omega=[omega0, omega1],
            alpha=[alpha0, alpha1],
            beta=[beta0, beta1],
            p00=0.97,   # calm is persistent
            p11=0.90,   # stressed is less persistent (mean-reverting)
        )

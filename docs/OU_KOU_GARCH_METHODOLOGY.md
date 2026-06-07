# OU-Kou-GARCH + Adaptive EKF — Methodology Reference

## Overview

This document is the complete mathematical and implementation reference for the **OU-Kou-GARCH** jump-diffusion model with **Adaptive Extended Kalman Filter (AEKF)** used as the 6th BIC competitor in the `GARCHRangeModel` ensemble and optionally as a standalone member of the `RangePredictor` ensemble.

The model produces two outputs:
1. **P(in range)**: Probability that the cumulative log-return over a horizon h stays within ±threshold, derived analytically from the characteristic function of the process.
2. **Direction signal**: BULLISH / BEARISH + confidence, derived from the AEKF's time-varying state estimate of the mean-reversion drift κ_T·(μ_T − X_T).

See also: [GARCH_METHODOLOGY.md](GARCH_METHODOLOGY.md) for the arch-based GARCH/GJR-GARCH/EGARCH/ARCH variants and MS-GARCH.

---

## §1 Motivation

### Why the existing models miss mean reversion

The current ensemble members — XGBoost, LightGBM, GARCH(1,1) variants, and MS-GARCH — model two orthogonal signals:
- **ML models**: cross-sectional feature patterns (RSI, realized vol, Hurst, etc.)
- **GARCH models**: time-series volatility clustering

Neither explicitly models **mean reversion** — the empirically well-documented tendency of equity prices to gravitational pull back toward a long-run equilibrium after deviations. This is the core thesis of iron condor strategies: prices tend to stay range-bound because they mean-revert.

### Why asymmetric jumps matter

Equity log-return distributions have fat left tails: large negative moves occur more frequently and are larger in magnitude than large positive moves. Single-regime GARCH captures the conditional heteroscedasticity but assumes innovations are symmetric (under Normal/Student-t distributions). The Kou (2002) double-exponential jump component explicitly models this asymmetry via separate rate parameters η₁ (up-jump decay) and η₂ (down-jump decay), with the freedom to set η₂ < η₁ (larger mean downside jumps).

### Why an Adaptive EKF

Standard MLE on a fixed window gives static parameter estimates κ̂ and μ̂. But the speed of mean reversion and the level of equilibrium both shift over time (e.g., QQQ's equilibrium log-price shifts as index composition changes; κ increases in choppy low-trend environments and decreases in trending ones). A Kalman filter that tracks [X_t, κ_t, μ_t] as a state vector gives real-time estimates that drive the direction signal.

---

## §2 Model Specification

### 2.1 Continuous-Time SDE

Let S_t be the asset price and X_t = log(S_t) the log-price. The model is:

```
dX_t = κ(μ − X_t) dt + σ_t dW_t + J_t dN_t
```

**Components**:

| Symbol | Type | Description |
|--------|------|-------------|
| κ > 0 | Parameter | Mean-reversion speed (per year) |
| μ | Parameter | Long-run log-price equilibrium |
| σ_t | State variable | GARCH(1,1) conditional volatility (time-varying) |
| W_t | Stochastic process | Standard Brownian motion |
| N_t | Stochastic process | Poisson process with intensity λ > 0 (jump arrivals per day) |
| J_t | Random variable | Kou double-exponential jump size (see §4) |

### 2.2 Discrete-Time Euler Approximation

With daily time step dt = 1/252:

```
r_t = X_t − X_{t-1} = κ(μ − X_{t-1})·dt + σ_t·ε_t + Σ_{i=1}^{N_t} J_{t,i}
```

Where ε_t ~ N(0,1) and N_t ~ Poisson(λ·dt).

---

## §3 GARCH(1,1) Volatility Layer

The conditional variance follows:

```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

**Parameters**: ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (stationarity).

**Unconditional variance**: σ² = ω / (1 − α − β).

**Initialisation**: σ²_0 = unconditional variance (same as `msgarch.py`).

This replaces the constant diffusion coefficient σ from the classical Vasicek/OU model, capturing the empirically observed volatility clustering in equity returns.

---

## §4 Kou (2002) Double-Exponential Jump Distribution

### 4.1 Distribution of a Single Jump

A single jump size J follows an asymmetric double-exponential distribution (Kou 2002):

```
f_J(x) = p_up · η₁ · e^{−η₁x} · 1_{x≥0}  +  (1−p_up) · η₂ · e^{η₂x} · 1_{x<0}
```

**Parameters**:
- p_up ∈ (0,1): probability of an upward jump
- η₁ > 0: rate of upward jumps (mean up-jump = 1/η₁ in log-return units)
- η₂ > 0: rate of downward jumps (mean down-jump = 1/η₂)

**Moments of a single jump**:
```
E[J]   = p_up/η₁ − (1−p_up)/η₂
Var[J] = 2·p_up/η₁² + 2·(1−p_up)/η₂²
```

**Equity calibration**: typically η₂ < η₁ (downside jumps are larger on average) and p_up < 0.5 (more up-jumps but smaller). For QQQ-like assets, a starting point is η₁ ≈ 50 (2% up-jumps), η₂ ≈ 40 (2.5% down-jumps), p_up ≈ 0.45.

### 4.2 Characteristic Function of a Single Jump

The Fourier transform E[e^{iuJ}] is rational in u (one of the key tractability advantages of the Kou model):

```
E[e^{iuJ}] = p_up · η₁/(η₁ − iu) + (1−p_up) · η₂/(η₂ + iu)
```

### 4.3 Compound Poisson Jumps

Over a time period [t, t+dt], the number of jumps N_dt ~ Poisson(λ·dt). The aggregate jump contribution to the return is Σ_{i=1}^{N_dt} J_i.

**Compound Poisson CF**:
```
E[e^{iu·Σ J_i}] = exp(λ·dt·(E[e^{iuJ}] − 1))
```

---

## §5 Maximum Likelihood Estimation

### 5.1 Full Log-Likelihood

At each time step t, the return r_t is a mixture over the number of jumps k:

```
L_t(θ) = Σ_{k=0}^{K_max} P(N_dt=k) · f(r_t | k jumps, θ)
```

Where:
- `P(N_dt=k) = Poisson(k; λ·dt)` — probability of k jumps
- `f(r_t | k jumps)` — conditional density of r_t given k jumps occurred

### 5.2 Gaussian Moment-Matching Approximation

The exact distribution of `Σ_{i=1}^k J_i` is a k-fold convolution of double-exponentials — no closed form. We use **Gaussian moment-matching** (cumulant additivity):

```
For k jumps:
  μ_k   = κ(μ − X_{t-1})·dt + k · E[J]          (mean)
  σ²_k  = σ²_t + k · Var[J]                       (variance)
  
  f(r_t | k jumps) ≈ N(r_t; μ_k, σ²_k)
```

**Why this is accurate**: The k-fold convolution has exact mean and variance given by the formulas above (cumulant additivity). For k ≥ 2, the CLT applies and the Normal approximation is tight. For k=0 (no jump), the density is exact (pure GARCH Normal). K_max = 10 truncates at < 0.001% Poisson weight for λ·dt ≤ 0.05.

**Total log-likelihood**:
```
logL(θ) = Σ_{t=1}^T log(Σ_{k=0}^{10} Poisson(k; λ·dt) · N(r_t; μ_k, σ²_k))
```

The inner sum is floored at 1e-300 before taking the log (numerical underflow prevention).

### 5.3 Parameters and Bounds

| Parameter | Symbol | Bounds | Rationale |
|-----------|--------|--------|-----------|
| Mean reversion speed | κ | (1e-4, 20) | κ > 0 required; 20 ≈ half-life of 8 trading days (very fast) |
| Equilibrium | μ | unconstrained | Log-price in any range |
| GARCH intercept | ω | (1e-8, ∞) | Strictly positive |
| GARCH ARCH | α | (1e-6, 0.90) | Stationarity enforced jointly |
| GARCH GARCH | β | (1e-6, 0.90) | α+β < 0.9999 enforced inside objective |
| Jump intensity | λ | (1e-4, 5.0) | Max 5 jumps/day — never reached in practice |
| Up-jump prob | p_up | (0.01, 0.99) | Avoid degenerate all-up or all-down |
| Up-jump rate | η₁ | (1.0, 500.0) | Mean jump 0.2%–100%; prevents infinite jumps |
| Down-jump rate | η₂ | (1.0, 500.0) | Same |

**Optimiser**: L-BFGS-B (scipy.optimize.minimize), max 500 iterations, ftol=1e-8.

### 5.4 Starting Values (Two-Stage)

**Stage 1 — OLS for κ and μ**:
Regress X_t on X_{t-1}: the slope b = e^{−κ·dt} gives κ = −log(b)/dt; the intercept a = μ(1−b) gives μ = a/(1−b). This provides a warm start that avoids the slow exploration L-BFGS-B would need from random initialisations.

**Stage 2 — Moment matching for jumps**:
The excess kurtosis of returns under the compound Poisson model is approximately λ·kurtosis_per_jump. With symmetric starting params (p_up=0.5, η₁=η₂=50), kurtosis_per_jump ≈ 24. So λ_0 ≈ kurtosis(r)/24, clamped to [0.01, 0.5].

---

## §6 Adaptive Extended Kalman Filter (AEKF)

### 6.1 Purpose

MLE gives population-average (κ̂, μ̂) from the full history. The AEKF tracks how κ and μ evolve through time, which is needed for:
1. The real-time direction signal (uses κ_T and μ_T at the last observation)
2. Detecting structural breaks in mean reversion behaviour (via κ CV diagnostic)

The AEKF is run **after** MLE with the MLE-estimated parameters as initialisations, not jointly. This is a simplification: joint MLE + EKF (extended Kalman smoother in the M-step) is theoretically superior but adds 200+ lines of backward-pass code for what is primarily a direction signal.

### 6.2 State Vector and Dynamics

**State**: `z_t = [X_t, κ_t, μ_t]ᵀ` — log-price + two time-varying OU parameters.

**Why include X_t?** Including the observable log-price as a state component allows P[0,0] (the position variance) to serve as the natural scale for the innovation-detection threshold in the adaptive Q update. An alternative 2-state [κ, μ] formulation would require a separate threshold scale.

**OU Euler dynamics** (linearised Jacobian F at state z = [X, κ, μ]):

```
F = I + dt · [[-κ,   μ-X,   κ  ],
               [ 0,    0,    0  ],
               [ 0,    0,    0  ]]
```

Derivation: `X_{t+1} ≈ X_t + κ(μ-X_t)·dt`, so `∂X_{t+1}/∂X_t = 1 − κ·dt`, `∂X_{t+1}/∂κ = (μ-X)·dt`, `∂X_{t+1}/∂μ = κ·dt`. Both κ and μ are modelled as random walks (∂κ_{t+1}/∂κ_t = 1, etc.).

### 6.3 Prediction Step

```
z_pred = F · z_{t-1}
P_pred = F · P_{t-1} · Fᵀ + Q_t
```

Where Q_t is the adaptive process noise (see §6.4).

### 6.4 Adaptive Q Update (Mehra 1970)

The innovation at time t:
```
ν_t = X_t^{observed} − z_pred[0]
```

Innovation-based Q update:
```
if |ν_t| > γ · √P_pred[0,0]:
    Q_t ← Q_{t-1} · α_adapt      (inflate on structural break)
else:
    Q_t ← Q_{t-1} · 0.999         (slow decay toward Q_0)
Q_t = max(Q_t, diag([1e-10, 1e-12, 1e-12]))   (positive floor)
```

**Default values**: γ=3.0 (3-sigma threshold), α_adapt=5.0 (5× inflation).

**Decay time constant**: τ = −1/log(0.999) ≈ 1000 steps (~4 trading years). This means after a single shock, the inflated Q returns to baseline in ~4 years of data unless further shocks occur. In a 1-year training window, Q remains elevated after a shock — by design, since a single large shock likely indicates a regime shift.

**Rationale**: Mehra (1970) showed that adaptive filters based on the innovation sequence outperform fixed-Q filters when model parameters are non-stationary. Here we use a simplified version: large innovations (price moves > 3σ_predicted) trigger immediate Q inflation.

### 6.5 Update Step

**Measurement model** (we observe X_t directly):
```
H = [1, 0, 0]    (log-price is fully observed)
S_t = P_pred[0,0] + R_t     (innovation variance; R_t = σ²_t from GARCH path)
K_t = P_pred[:,0] / S_t     (Kalman gain, shape (3,))
z_t = z_pred + K_t · ν_t
P_t = (I − K_t·H) · P_pred
```

**Positive-definiteness maintenance**:
After each update, symmetrise P and eigenvalue-clip to ≥ 1e-12:
```
P_t = (P_t + P_tᵀ) / 2
eigvals, V = eigh(P_t)
P_t = V · diag(max(eigvals, 1e-12)) · Vᵀ
```

This prevents accumulated floating-point errors from making P indefinite.

**Constraint**: κ_T = max(κ_T, 1e-6) enforced after every update.

---

## §7 Multi-Step Volatility Forecast

The h-step ahead combined variance is:

```
σ²_h = σ²_OU(h) + σ²_jump(h)
```

### 7.1 GARCH + OU Compression

GARCH analytic h-step forecast (cumulative):
```
σ²_GARCH(h) = Σ_{j=1}^h E[σ²_{T+j}]
             = h·σ²_∞ + (σ²_T − σ²_∞)·(1 − (α+β)^h)/(1 − α+β)
where σ²_∞ = ω/(1 − α − β)
```

OU mean-reversion compression factor:
```
ξ_h = (1 − e^{−2κh/252}) / (2κ/252)
```

For a random walk (κ→0): ξ_h → h/252. For strong reversion (κ→∞): ξ_h → 1/(2κ/252) = 252/(2κ), independent of h.

Combined OU variance:
```
σ²_OU(h) = σ²_GARCH(h) · ξ_h / (h/252)
```

The factor `ξ_h / (h/252)` equals 1 when κ→0 (recovers plain GARCH) and < 1 for κ > 0 (mean reversion compresses long-horizon variance).

### 7.2 Jump Variance Accumulation

Poisson additivity: over horizon h, expected number of jumps = λ·h/252.
```
σ²_jump(h) = λ·(h/252)·Var(J) = λ·(h/252)·[2·p_up/η₁² + 2·(1−p_up)/η₂²]
```

### 7.3 Total Forecast

```
σ_h = √(max(σ²_OU(h) + σ²_jump(h), 1e-10))
```

---

## §8 P(in Range) via Characteristic Function Inversion

### 8.1 Characteristic Function of X_{t+h}|X_t

The exact CF of the conditional log-price distribution (combining OU mean reversion, GARCH variance, and compound Poisson jumps):

```
φ_h(u) = exp(
    iu · [μ + (X_T − μ)·e^{−κh/252}]       ← OU conditional mean
  − u² · σ²_OU(h) / 2                         ← diffusion variance (OU compressed)
  + λ·(h/252) · (E[e^{iuJ}] − 1)              ← compound Poisson jump CF
)
```

Where `σ²_OU(h) = σ²_T · ξ_h` (OU variance at the last GARCH estimate) and the Kou CF is:
```
E[e^{iuJ}] = p_up·η₁/(η₁ − iu) + (1−p_up)·η₂/(η₂ + iu)
```

Note: the CF uses the MLE-estimated parameters (κ, μ) and the AEKF final state for X_T — not the AEKF-tracked κ_T and μ_T — to keep the CF computation fast and stable. The direction signal uses the AEKF parameters.

### 8.2 FFT Inversion

The conditional PDF is recovered by numerical inverse Fourier transform, using the same scheme as `ClassicalTemperedStable.fft_pdf()` in `garch_range_predictor.py`:

```
Grid parameters:
  N     = 2048
  W     = max(10·threshold_pct, 15.0)   (half-width in log-price space)
  dt_x  = 2W/N                           (spacing in x domain)
  du    = 2π/(N·dt_x)                    (spacing in frequency domain)
  u_grid = (j − N/2)·du  for j=0..N-1  (centred frequency grid)
  x_grid = (j − N/2)·dt_x + mean_h      (centred on OU conditional mean)

Inversion:
  CF_shifted = ifftshift(φ_h(u_grid))
  pdf_raw    = fftshift(ifft(CF_shifted)).real · (N·du / 2π)
  pdf_vals   = max(pdf_raw, 0)           (clip numerical negatives)
```

**Grid width W**: set to `max(10·threshold_pct, 15.0)`. For threshold_pct=0.05, W=15 (well within ±15 log-points). For tight thresholds (threshold_pct=0.01), W=max(0.10, 15.0)=15 still. The factor of 10 ensures at least 10× the threshold range is covered.

**P(in range) integration**:
```
P(|X_{T+h} − X_T| < threshold) = ∫_{X_T − threshold}^{X_T + threshold} f_h(x) dx
```
Evaluated via `scipy.integrate.quad` with tolerance 1e-4. Falls back to Normal approximation if integration fails.

---

## §9 Direction Signal from AEKF

The mean-reversion drift at the current state:
```
drift_T = κ_T · (μ_T − X_T)
```

Where (κ_T, μ_T, X_T) are the AEKF final estimates.

**Interpretation**:
- drift_T > 0: price X_T is below equilibrium μ_T → mean-reversion force is upward → **BULLISH**
- drift_T < 0: price X_T is above equilibrium μ_T → mean-reversion force is downward → **BEARISH**

**Confidence normalisation**:
```
Historical drifts: d_t = κ_t · (μ_t − X_t)  for t=1..T  (from AEKF state history)
std_drift = std(d_t)
z = clip(drift_T / std_drift, −3, 3)       (z-score in 3σ range)
confidence = |z| / 3                        (maps [0,3] → [0,1])
```

This normalization means:
- confidence = 1.0: current drift is at the 3σ level of the historical drift distribution (very strong reversion signal)
- confidence = 0.5: current drift is at the 1.5σ level (moderate signal)
- confidence = 0.0: price is exactly at equilibrium (no directional signal)

**Integration with ensemble**: The direction signal is stored in `ou_jump_direction` and `ou_jump_confidence` in `_symbol_models` and the window JSON. It is **not** currently fed into the DirectionPredictor ensemble but is logged for analysis and future integration.

---

## §10 Goodness-of-Fit Diagnostics

All metrics are computed from the fitted model and stored in `to_state_dict()["diagnostics"]` and the window JSON under `ou_jump_*` keys.

### 10.1 Residual Definition

Standardised residuals (removing OU drift and GARCH volatility):
```
ε_t = (r_t − κ(μ − X_{t-1})·dt) / σ_t
```

These residuals should contain the jump component only. Under a well-specified model, ε_t is approximately i.i.d. with zero mean, unit variance, some positive kurtosis (from Kou jumps), and possibly slight left skewness.

### 10.2 Metrics

| Key | Formula | Interpretation |
|-----|---------|----------------|
| `loglik` | Full MLE log-likelihood | Compare across windows (higher = better fit) |
| `bic` | -2·loglik + 9·log(T) | Model selection vs other ensemble members |
| `aic` | -2·loglik + 18 | Comparison with BIC (AIC favours complexity more) |
| `jb_pvalue` | Jarque-Bera p-value on ε_t | p < 0.05: jump model hasn't absorbed all tail risk; increase λ or decrease η |
| `resid_skewness` | Skew of ε_t | Should be near 0 if Kou asymmetry is well-fitted; negative skew = more downside residuals |
| `resid_kurtosis` | Excess kurtosis of ε_t | Expected > 0 (jumps leave kurtosis); compare raw return kurtosis vs residual kurtosis as quality measure |
| `ljung_box_pvalue` | LB Q-test p-value on ε²_t, lag=10 | p < 0.05: remaining ARCH effects — GARCH(1,1) under-specified |
| `ou_half_life_days` | 252·log(2)/κ | Days to halve gap to μ. 5d = strongly reverting; 60d = weak |
| `jump_intensity_annual` | λ·252 | Annualised jump count. >20 = near-daily "jumps" (likely noise) |
| `jump_mean_up_pct` | 100/η₁ | Expected up-jump in % log-return |
| `jump_mean_down_pct` | 100/η₂ | Expected down-jump in % |
| `diffusion_persistence` | α + β | GARCH persistence. >0.95 = highly persistent vol |
| `aekf_kappa_cv` | std(κ_t)/mean(κ_t) | CV > 1: κ highly unstable → mean reversion speed is regime-dependent |
| `aekf_mu_range` | max(μ_t)−min(μ_t) in % | Range of equilibrium tracked by AEKF over the window |

### 10.3 Cross-Window Interpretation

Track the following across training windows (stored in window JSON):
- **`ou_half_life_days`**: short half-life (5–15d) in choppy markets; longer (30–60d) in trending markets. This tracks the regime correctly.
- **`jump_intensity_annual`**: should be 5–15 for daily equity data (weekly to monthly jumps). If > 30, the jump component is absorbing microstructure noise.
- **`aekf_kappa_cv`**: values > 1.0 indicate that AEKF needed to substantially adjust κ across the window — good indicator that the market structure changed mid-window.

---

## §11 BIC and Model Selection

```
BIC = −2·loglik + 9·log(T)
```

With 9 parameters vs MS-GARCH's 8:
- At T=300: BIC penalty = 9·log(300) ≈ 50.9 vs MS-GARCH's 8·log(300) ≈ 45.2
- OU-Kou-GARCH needs log-likelihood improvement of ~2.85 over MS-GARCH to win the BIC race

In low-volatility trending windows, MS-GARCH tends to win (mean reversion is weak, jumps are rare). In post-shock or choppy environments, OU-Kou-GARCH can win because the mean-reversion term correctly compresses the long-horizon variance forecast.

**Cascade**: if OU-Kou-GARCH wins the BIC race inside `GARCHRangeModel.fit()`, it becomes `selected_variant="OU-Kou-GARCH"` and its `p_in_range_compounding` is used as the GARCH ensemble member's contribution. If it loses, it still participates as a standalone `enable_oujump=True` member in `RangePredictor`.

---

## §12 Implementation Notes

- **File**: `src/ait/ml/ou_jump.py`
- **Dependencies**: `numpy`, `scipy` (standard); `statsmodels` (optional, for Ljung-Box)
- **Minimum observations**: 60 (same as MS-GARCH; two-stage OLS init needs ~10)
- **AEKF sequencing**: run after MLE with fitted params as initialisations (not jointly)
- **to_state_dict()**: stores only Python scalars and dicts — no numpy arrays, no live objects
- **Ljung-Box fallback**: if `statsmodels` is not installed, `ljung_box_pvalue` is logged as `nan` (non-fatal)
- **JSON safety**: `_coerce_json()` from `garch_range_predictor.py` is applied to the full state dict before window JSON write

---

## §13 References

1. **Kou, S.G. (2002)**. "A Jump-Diffusion Model for Option Pricing." *Management Science* 48(8): 1086–1101.
   - Original double-exponential jump-diffusion model. Derives the characteristic function used in §8.

2. **Kalman, R.E. (1960)**. "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering* 82(1): 35–45.
   - Foundation of the Kalman filter. The EKF extends this to non-linear systems via Jacobian linearisation.

3. **Mehra, R.K. (1970)**. "On the Identification of Variances and Adaptive Kalman Filtering." *IEEE Transactions on Automatic Control* 15(2): 175–184.
   - Innovation-based adaptive Q update used in the AEKF (§6.4).

4. **Uhlenbeck, G.E. & Ornstein, L.S. (1930)**. "On the Theory of Brownian Motion." *Physical Review* 36(5): 823–841.
   - Original OU process specification; §2 SDE follows this directly.

5. **Bollerslev, T. (1986)**. "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 31(3): 307–327.
   - GARCH(1,1) used as the volatility layer in §3.

6. **Hamilton, J.D. (1994)**. *Time Series Analysis*. Princeton University Press.
   - Standard reference for time-series MLE and Kalman filtering in financial applications.

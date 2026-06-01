# GARCH Volatility Models — Range Predictor Ensemble Member

## Overview

This document is a complete mathematical and implementation reference for the GARCH/ARCH volatility models used as a third member of the range predictor ensemble in the AIT v2 walk-forward backtester.

The range predictor answers one question: **P(price stays within ±threshold% over the next N days)**. XGBoost and LightGBM answer this by learning from lagged cross-sectional features (IV rank, Hurst exponent, realized vol, etc.). The GARCH model answers it differently — by directly modelling the return variance process and computing the probability analytically from the forecasted volatility distribution.

These are genuinely orthogonal signals. GARCH captures volatility clustering; the ML models capture regime patterns in observable features. Together they form a more robust ensemble.

---

## 1. Motivation: Why Volatility is Forecastable

Financial returns exhibit well-documented stylised facts:

1. **Volatility clustering**: large moves tend to follow large moves, quiet periods follow quiet periods. This is the core insight behind ARCH/GARCH.
2. **Fat tails**: the unconditional return distribution has heavier tails than the Normal — excess kurtosis is typically 3–8 for daily equity returns.
3. **Leverage effect**: negative returns tend to increase volatility more than positive returns of the same magnitude. QQQ exhibits this clearly.
4. **Mean reversion in variance**: conditional variance is stationary and reverts to its long-run mean, making multi-step forecasts tractable.

The **ARCH effects test** (Engle 1982) — a Lagrange Multiplier test on squared residuals — formally confirms that variance is time-varying and predictable from past values.

---

## 2. ARCH(p) Model

Proposed by Engle (1982). The conditional variance at time t depends on the p most recent squared residuals:

```
r_t = μ + ε_t
ε_t = σ_t · z_t,   z_t ~ D(0,1)

σ²_t = ω + α₁ε²_{t-1} + α₂ε²_{t-2} + ... + α_p ε²_{t-p}
```

**Parameters**: ω > 0, αᵢ ≥ 0 for all i.  
**Stationarity**: Σαᵢ < 1.  
**Interpretation**: today's variance is a weighted sum of recent squared shocks. No persistence beyond p lags.

**ARCH(1)** is the baseline in this implementation — it always runs and serves as the first fallback when GARCH variants fail to converge.

---

## 3. GARCH(p,q) Model

Bollerslev (1986) generalised ARCH by including lagged conditional variances:

```
σ²_t = ω + Σᵢ αᵢ ε²_{t-i} + Σⱼ βⱼ σ²_{t-j}
```

**GARCH(1,1)** — the workhorse, almost always used in practice:

```
σ²_t = ω + α ε²_{t-1} + β σ²_{t-1}
```

**Parameters**: ω > 0, α ≥ 0, β ≥ 0.  
**Stationarity**: α + β < 1.  
**Long-run (unconditional) variance**: σ̄² = ω / (1 − α − β).  
**Persistence**: α + β close to 1 means shocks decay slowly — high volatility regimes persist.

---

## 4. GJR-GARCH (Asymmetric GARCH)

Glosten, Jagannathan & Runkle (1993). Adds an indicator for negative shocks to capture the leverage effect:

```
σ²_t = ω + α ε²_{t-1} + γ ε²_{t-1} · 𝟙[ε_{t-1} < 0] + β σ²_{t-1}
```

**Leverage parameter** γ ≥ 0: when ε_{t-1} < 0, the effective ARCH coefficient is α + γ instead of α. For equities, γ > 0 is almost always found empirically.

**Stationarity**: α + γ/2 + β < 1.

In the `arch` package: `vol='GARCH', p=1, o=1, q=1` (the `o` parameter controls the asymmetric lag order).

---

## 5. EGARCH (Exponential GARCH)

Nelson (1991). Models log-variance instead of variance — guarantees positivity without parameter constraints:

```
log(σ²_t) = ω + α [|z_{t-1}| − E|z_{t-1}|] + γ z_{t-1} + β log(σ²_{t-1})
```

where z_t = ε_t / σ_t are standardised residuals.

**Key properties**:
- No positivity constraints needed (log variance is unconstrained)
- γ < 0 captures leverage: negative z_{t-1} increases log variance more than positive
- E|z_{t-1}| = √(2/π) for Normal innovations

In the `arch` package: `vol='EGARCH', p=1, q=1`.

---

## 6. Innovation Distributions

The distribution D(0,1) of the standardised residuals z_t = ε_t/σ_t determines how P(in range) is computed from the forecasted volatility.

### 6.1 Normal (Gaussian)

```
f(z) = (1/√(2π)) exp(−z²/2)
P(in range) = 2Φ(threshold/σ_h) − 1
```

Φ is the standard Normal CDF. This is the simplest closed-form expression.

**Selected when**: Jarque-Bera test on standardised residuals fails to reject normality (p ≥ 0.05) AND BIC is lowest.

### 6.2 Student-t (Symmetric)

Hansen (1994), also Bollerslev (1987). Heavier tails than Normal:

```
f(z; ν) = [Γ((ν+1)/2) / (Γ(ν/2)√(π(ν−2)))] · (1 + z²/(ν−2))^{−(ν+1)/2}
```

**Degrees of freedom** ν > 2 (estimated by MLE). As ν → ∞, recovers Normal.

```
P(in range) = F_t(threshold/σ_h; ν) − F_t(−threshold/σ_h; ν)
```

where F_t is the Student-t CDF (`scipy.stats.t.cdf`).

**Selected when**: BIC lower than Normal — typically when excess kurtosis of residuals is significant.

### 6.3 Skewed Student-t (Hansen's Skewed-t)

Hansen (1994). Two-parameter extension adding a skewness parameter:

```
Parameters: η ∈ (2, ∞) (tail thickness), λ ∈ (−1, 1) (skewness)
```

The distribution is constructed by scaling a Student-t differently on each side of the mode. For λ < 0 (left skew), the left tail is heavier — consistent with the empirical distribution of QQQ daily returns.

**P(in range)** computed via `scipy.stats.t` with asymmetric scaling — closed-form, fast.

In the `arch` package: `dist='skewt'`.

**Selected when**: BIC lower than Student-t — typically when skewness of residuals is significant (|skew| > 0.3).

### 6.4 Generalised Error Distribution (GED)

Nelson (1991). Shape parameter κ > 0:

```
f(z; κ) ∝ exp(−(1/2)|z/λ|^κ)
```

where λ is a scale normalisation. κ = 2 recovers Normal, κ = 1 gives Laplace (double-exponential), κ < 2 gives heavier tails.

**P(in range)** via `scipy.stats.gennorm.cdf`. Closed-form.

In the `arch` package: `dist='ged'`.

**Selected when**: BIC lower than Student-t and Skewed-t — alternative fat-tail shape without skewness.

### 6.5 Classical Tempered Stable (CTS)

Kim, Rachev & Chung; Massing (2024, *ALEA* v21 #59, arXiv:2303.07060v4).

The CTS distribution is an infinitely divisible distribution obtained by exponentially tempering the Lévy measure of an α-stable law. It has **all moments finite** (unlike pure stable distributions) while retaining heavy tails and asymmetry.

**Lévy measure**:
```
ν(dx) = [δ₊ x^{−1−α} e^{−λ₊ x} 𝟙{x>0} + δ₋ |x|^{−1−α} e^{−λ₋ |x|} 𝟙{x<0}] dx
```

**Parameters**: α ∈ (0, 2), δ₊ > 0, δ₋ > 0, λ₊ > 0, λ₋ > 0, μ ∈ ℝ. Six parameters total (μ absorbed into GARCH mean spec → 5 estimated).

**Characteristic function** (Massing 2024, Eq. 9):
```
φ_CTS(t) = exp[ itμ
  + δ₊ Γ(−α) ((λ₊ − it)^α − λ₊^α + itαλ₊^{α−1})
  + δ₋ Γ(−α) ((λ₋ + it)^α − λ₋^α − itαλ₋^{α−1}) ]
```

Special case α = 1 uses a log-based formula (Massing 2024, below Eq. 9).

**Cumulants** (Massing 2024, Eq. 11):
```
κ_m = Γ(m − α) [δ₊ λ₊^{α−m} + (−1)^m δ₋ λ₋^{α−m}],   m ∈ ℕ
```

All cumulants are finite. The first four give mean, variance, skewness, kurtosis — used for method-of-cumulants starting values.

**PDF**: No closed form. Computed via inverse FFT of the characteristic function on a grid of 2048 points over [−20, 20], then interpolated for arbitrary x values.

**P(in range)**: Computed via numerical integration (`scipy.integrate.quad`) of the FFT-derived PDF:
```
P(in range) = ∫_{−threshold/σ_h}^{+threshold/σ_h} f(z; θ) dz
```

**Loglikelihood** (required by `arch` Distribution interface):
```
ℓ(θ) = Σ_t [log f(ε_t/σ_t; θ) − log σ_t]
```

where f is evaluated via FFT interpolation.

**Simulation** (Massing 2024, §4): rejection sampling using an α-stable subordinator as the proposal distribution.

**Implementation**: `ClassicalTemperedStable` in [src/ait/ml/garch_range_predictor.py](../src/ait/ml/garch_range_predictor.py), subclassing `arch.univariate.distribution.Distribution`.

---

## 7. Multi-Step Variance Forecasting

Given a fitted GARCH model, we need σ_h — the volatility over the full horizon of h days.

### 7.1 Compounding (default)

Sum h one-step-ahead conditional variance forecasts via the GARCH recursion:

```
σ²_h = Σ_{i=1}^{h} E[σ²_{t+i}]
```

For GARCH(1,1), the recursion is:
```
E[σ²_{t+i}] = σ̄² + (α + β)^{i−1} (σ²_{t+1} − σ̄²)
```

where σ̄² = ω/(1−α−β) is the long-run variance.

The h-day sigma: `σ_h = √(σ²_h)`.

This correctly accounts for mean-reversion: in a high-vol regime, GARCH forecasts variance decaying back toward σ̄² over the horizon. In `arch`: `res.forecast(horizon=h, method='analytic').variance`.

### 7.2 Square-Root Scaling

Simpler approximation that ignores mean-reversion:

```
σ_h = σ_1 × √h
```

where σ_1 is the one-step-ahead conditional volatility.

This is the standard option-market convention (∝ √T scaling of Black-Scholes). It overestimates vol for persistent GARCH processes over long horizons (since mean-reversion brings variance down faster than √h scaling implies) and is included for comparison.

**Both methods are always computed and stored** in the window JSON as `p_in_range_compounding` and `p_in_range_sqrt_scale`. The compounding method is used as the default ensemble contribution. After Exp 13, Brier scores against realised outcomes will determine which is more accurate.

---

## 8. P(in range) Computation

Given σ_h (from either horizon method) and the fitted innovation distribution:

### Symmetric distributions (Normal, Student-t, GED)
```
P(in range) = F(threshold/σ_h) − F(−threshold/σ_h) = 2F(threshold/σ_h) − 1
```

### Skewed Student-t
```
P(in range) = F_skewt(threshold/σ_h; η, λ) − F_skewt(−threshold/σ_h; η, λ)
```

Uses asymmetric CDF with separate scaling for each side of the mode.

### Classical Tempered Stable
```
P(in range) = ∫_{−threshold/σ_h}^{+threshold/σ_h} f_CTS(z; θ) dz
```

via `scipy.integrate.quad` on the FFT-interpolated PDF. Tolerance: `epsabs=1e-4, epsrel=1e-4`.

---

## 9. Model Selection: BIC Grid Search

All combinations of (variant × distribution) are attempted. The combination with the **lowest BIC** wins:

```
BIC = −2 · ℓ(θ̂) + k · log(n)
```

where k = number of parameters, n = number of observations.

BIC penalises complexity more heavily than AIC (`2k` vs `k·log(n)` for large n), which is appropriate here since we want parsimonious models that generalise to OOS windows.

**Grid**:
- 4 variants × 5 distributions = 20 combinations per window
- Each combination records BIC and convergence status in `garch_all_variants` in the window JSON
- `null` BIC = convergence failure

---

## 10. Distribution Validation Diagnostics

Regardless of which distribution wins, the following are always computed and stored:

**Jarque-Bera test** on standardised residuals z_t = ε_t/σ_t:
```
JB = (n/6) [S² + (K−3)²/4]
```
where S = sample skewness, K = sample kurtosis. Under H₀ (normality): JB ~ χ²(2). p-value < 0.05 → reject normality.

**Residual skewness**: sign and magnitude indicate whether left-skew (typical for equities) is present.

**α-Stable diagnostic**: `scipy.stats.levy_stable` is fitted to the standardised residuals as a standalone diagnostic — not used in the ensemble. Records whether stable distributions are competitive across windows (tracked via `garch_stable_attempted`, `garch_stable_converged`, `garch_stable_loglik` in window JSON).

---

## 11. Fallback Chain

```
Best (variant × distribution) by BIC
  ↓ if all GARCH variants fail to converge
ARCH(1) with best distribution
  ↓ if ARCH(1) also fails
Constant volatility: σ_h = std(returns) × √h  [always succeeds]
  ↓ if even constant vol is NaN (zero-return series)
GARCH dropped from ensemble — XGB + LGB weights renormalised to sum to 1.0
```

`fallback_used` in the state dict is `null` (no fallback), `"arch"`, `"constant_vol"`, or `"dropped"`.

---

## 12. CV Integration and Fitted Weights

GARCH participates identically to XGBoost and LightGBM in the fitted-weight system:

### Walk-Forward CV Folds

`_walk_forward_split(n, n_splits=4, gap=5)` produces 4 non-overlapping folds with a 5-row purge gap between train and validation — preventing leakage from autocorrelated returns.

```
fold_size = n // 5
fold i: train = [0, fold_size*(i+1))
        val   = [fold_size*(i+1) + gap, fold_size*(i+2))
```

For GARCH, the Close price series (not the feature matrix) is used in CV — GARCH operates on returns directly.

### Per-Fold Evaluation

For each fold:
1. Compute log returns from training Close prices
2. Fit GARCH (best variant × distribution by BIC on training data)
3. Forecast σ_h for the horizon
4. Compute P(in range) → threshold at 0.5 → binary prediction
5. Compare to actual in-range labels (`_create_labels`) on validation Close prices
6. Measure balanced accuracy (sensitivity + specificity) / 2

### Edge-Over-Baseline Weighting

```
edge_garch  = max(0, cv_balanced_acc_garch  − 0.50)
edge_xgb    = max(0, cv_balanced_acc_xgb    − 0.50)
edge_lgb    = max(0, cv_balanced_acc_lgb    − 0.50)
total_edge  = edge_garch + edge_xgb + edge_lgb

weight_garch = edge_garch / total_edge   (0 if total_edge = 0)
```

Models below 50% balanced accuracy contribute zero weight. The weights are then normalised so they sum to 1.0. This is the same formula used for the XGB/LGB pair in Exp 10+.

---

## 13. Implementation Map

| Mathematical Object | Python Location |
|---|---|
| `GARCHRangeModel.fit()` | [src/ait/ml/garch_range_predictor.py](../src/ait/ml/garch_range_predictor.py) |
| `GARCHRangeModel._fit_one()` | arch `arch_model(...).fit()` wrapper |
| `GARCHRangeModel._best_distribution()` | BIC comparison across 5 distributions |
| `GARCHRangeModel._multi_step_sigma()` | `res.forecast(horizon=h).variance` (compounding) and σ₁×√h (sqrt_scale) |
| `GARCHRangeModel._p_in_range()` | `scipy.stats` CDF calls (Normal/t/skewt/GED) or FFT integration (CTS) |
| `ClassicalTemperedStable._characteristic_function()` | Massing (2024) Eq. 9 |
| `ClassicalTemperedStable._fft_pdf()` | `numpy.fft.ifft` on 2048-point grid |
| `ClassicalTemperedStable.loglikelihood()` | Interpolated log-PDF via `scipy.interpolate` |
| `ClassicalTemperedStable.starting_values()` | Method-of-cumulants via Massing (2024) Eq. 11 |
| `ClassicalTemperedStable.simulate()` | Rejection sampling, Massing (2024) §4 |
| `RangePredictor._train_garch()` | [src/ait/ml/range_predictor.py](../src/ait/ml/range_predictor.py) |
| Window JSON GARCH fields | [src/ait/backtesting/walkforward.py](../src/ait/backtesting/walkforward.py) `_train_window_range_model()` |

---

## 14. Window JSON Schema

```json
"range_predictor": {
  "status":             "ok",
  "cv_scores":          {"xgboost": 0.631, "lightgbm": 0.567, "garch": 0.554},
  "fitted_weights":     {"xgboost": 0.45,  "lightgbm": 0.29,  "garch": 0.26},
  "garch_selected_variant":   "GJR-GARCH",
  "garch_selected_dist":      "CTS",
  "garch_selected_bic":       -1897.4,
  "garch_fallback":           null,
  "garch_jb_pvalue":          0.001,
  "garch_resid_skewness":     -0.47,
  "garch_stable_attempted":   true,
  "garch_stable_converged":   false,
  "garch_stable_loglik":      null,
  "p_in_range_compounding":   0.68,
  "p_in_range_sqrt_scale":    0.65,
  "garch_all_variants": {
    "GARCH(1,1)":  {"selected_dist": "skewt", "bic": -1871.2, "converged": true,
                    "dist_race": {"normal": {"bic": -1843.1, "converged": true}, ...}},
    "GJR-GARCH":   {"selected_dist": "CTS",   "bic": -1897.4, "converged": true, ...},
    "EGARCH(1,1)": {"selected_dist": "skewt", "bic": -1880.1, "converged": true, ...},
    "ARCH(1)":     {"selected_dist": "t",     "bic": -1821.1, "converged": true, ...}
  }
}
```

---

## 15. References

1. **Engle, R.F. (1982)**. "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica* 50(4): 987–1007.

2. **Bollerslev, T. (1986)**. "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics* 31(3): 307–327.

3. **Nelson, D.B. (1991)**. "Conditional Heteroskedasticity in Asset Returns: A New Approach." *Econometrica* 59(2): 347–370.

4. **Glosten, L.R., Jagannathan, R. & Runkle, D.E. (1993)**. "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance* 48(5): 1779–1801.

5. **Hansen, B.E. (1994)**. "Autoregressive Conditional Density Estimation." *International Economic Review* 35(3): 705–730.

6. **Kim, Y.S., Rachev, S.T. & Chung, D.M.** "The Modified Tempered Stable Distribution, GARCH-Models and Option Pricing." Available: https://methods.stat.kit.edu/download/doc_secure1/KimRachevChung.pdf

7. **Massing, T. (2024)**. "Parametric Estimation of Tempered Stable Laws." *ALEA — Latin American Journal of Probability and Mathematical Statistics* 21: 59. arXiv:2303.07060v4. Available: https://alea.impa.br/articles/v21/21-59.pdf

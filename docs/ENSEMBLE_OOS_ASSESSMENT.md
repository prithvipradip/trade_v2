# Ensemble Model OOS Assessment

**Related files:** `src/ait/backtesting/walkforward.py` (`_evaluate_range_model_oos`),
`src/ait/ml/garch_range_predictor.py` (`roll_*_forecasts`),
`src/ait/ml/ou_jump.py` (`oos_aekf_diagnostics`),
`src/ait/ml/msgarch.py` (`step_filter`, `p_in_range_from_filter`)

**Window JSON location:** `model_weights.range_predictor.oos_scores`

---

## 1. Why Balanced Accuracy Is Not Enough

The existing OOS metric was balanced accuracy at a fixed 0.5 decision threshold.  This has two problems:

**Problem 1 — threshold insensitivity.** A model that outputs P=0.72 on every day has 0% skill but perfect balanced accuracy if 72% of days are in-range.  A model that outputs P=0.60 on the best days and P=0.40 on the worst days has genuine rank-discrimination but mediocre balanced accuracy unless the threshold happens to bisect its predictions.

**Problem 2 — calibration is invisible.** A GARCH model that systematically outputs P=0.90 when the true rate is 0.65 is dangerously overconfident.  Balanced accuracy never surfaces this.  An overconfident model's probability gets multiplied by its fitted weight and passed directly into the ensemble — bad calibration propagates to the final P(in range) even when AUROC looks acceptable.

The replacement metrics address both.

---

## 2. Probability Scoring Metrics

These apply to every model that produces a probability sequence `p_t ∈ [0,1]` against a binary label `y_t ∈ {0,1}` (1 = price stayed in range).  The base rate `p̄ = mean(y_t)` is the naive climatology forecast.

### 2.1 Brier Score

```
BS = (1/T) Σ_t (p_t − y_t)²
```

Mean squared error between probability and outcome.  Range: [0, 1].  Lower is better.  A model that always predicts `p = p̄` achieves `BS_ref = p̄(1 − p̄)`.

**Interpretation:** BS < BS_ref means the model is better than climatology.  BS = 0.25 on a balanced dataset (p̄=0.5) means the model has no skill at all.  For QQQ with typical in-range rate ~60%, BS_ref ≈ 0.24; meaningful models should achieve BS < 0.20.

### 2.2 Brier Skill Score

```
BSS = 1 − BS / BS_ref
```

Normalised Brier Score: positive = better than climatology, zero = same, negative = worse.  Makes windows comparable despite different base rates.  A BSS of 0.10 means the model reduced mean squared probability error by 10% vs the naive forecast.

**Why BSS matters here:** Different OOS windows have different in-range rates (W06 was all breakouts; W09 was range-bound).  Raw Brier scores are not comparable across windows without normalisation.

### 2.3 Log Loss (Cross-Entropy)

```
LL = −(1/T) Σ_t [y_t · log(p_t) + (1 − y_t) · log(1 − p_t)]
```

Penalises confident wrong predictions exponentially.  A model that outputs P=0.95 on a day that breaks out contributes `−log(0.05) ≈ 3.0` to the sum — 30× more than a model that outputs P=0.50.

**Why log loss and Brier together:** Brier score is quadratic (gentle penalty for overconfidence); log loss is logarithmic (severe penalty).  A model with good Brier but high log loss is overconfident on specific high-risk days — exactly the failure mode of GARCH in W08/W09.

### 2.4 AUROC

```
AUROC = P(p_in-range > p_breakout)   [averaged over all in-range/breakout pairs]
```

Measures rank discrimination only — does the model rank in-range days above breakout days?  Invariant to calibration (re-scaling monotonically doesn't change AUROC).

**Relationship to AUROC vs Brier:** A model can have AUROC=0.65 but terrible Brier score if it's systematically miscalibrated (e.g., always predicts 0.9 when true rate is 0.65 — it ranks correctly but the probability values are wrong).  Both are needed.

### 2.5 Mean Confidence

```
confidence_t = max(p_t, 1 − p_t) ∈ [0.5, 1.0]
```

Average assertiveness.  A model that always outputs P=0.5 has mean confidence 0.5 — it has no view.  High confidence on correct days is desirable; high confidence on wrong days is the failure mode.  Track this alongside accuracy: `(mean_confidence − 0.5) / (balanced_acc − 0.5)` is an informal overconfidence ratio.

---

## 3. Realised Volatility MAE

For statistical models (GARCH, MS-GARCH, OU-Kou-GARCH) the probability P(in range) flows from a volatility forecast σ̂_h.  Even when P(in range) looks reasonable, the underlying vol forecast may be systematically wrong.  **Realised vol MAE** isolates this.

### 3.1 Definition

```
Realised σ_t = std(r_{t+1}, ..., r_{t+h}) · √252     [annualised]
Predicted σ̂_t = h-step vol forecast at day t         [annualised]

rvol_MAE  = mean |σ̂_t − σ_t|
rvol_bias = mean (σ̂_t − σ_t)   [+ = over-forecast, − = under-forecast]
```

### 3.2 Interpretation

| rvol_bias | Implication |
|-----------|-------------|
| Strongly positive | Model over-estimates vol → P(in range) too low → too few trades entered |
| Near zero | Vol forecast is unbiased |
| Strongly negative | Model under-estimates vol → P(in range) too high → entries in hostile vol regimes |

**GARCH rank inversion (W08/W09):** Post-shock, GARCH's exponential persistence keeps σ̂ high long after realised vol has reverted.  rvol_bias will be strongly positive in W08/W09 for GARCH, near-zero for MS-GARCH and OU-Kou-GARCH.  This is the direct numerical evidence of the rank-inversion problem documented in P28.

---

## 4. Statistical Model Rolling Forecasts

ML models (XGBoost/LightGBM) produce one prediction per OOS day naturally.  Statistical models are different: they are fitted once on training data and produce a single training-end estimate.  To get per-day OOS forecasts, the models must be stepped forward through OOS data.

### 4.1 GARCH Rolling Forecast

The GARCH model is refit on OOS returns alone (frozen spec from training BIC winner, no re-selection).  For each OOS day `t`, `arch.forecast(start=t)` returns the h-step variance forecast conditioning on returns `r_1..r_t`.  This is genuinely out-of-sample because the model parameters are frozen and only the variance path is updated.

**Approximation:** The OOS-only refit discards the training history warmup.  The variance path converges to the correct conditional variance within ~20 observations (the typical GARCH memory).  For 60-day OOS windows this is acceptable; the first ~20 days have slightly elevated uncertainty.

### 4.2 MS-GARCH Step Filter

The Hamilton filter is stepped forward one day at a time using frozen EM-estimated parameters:

```
State at day t:  (π_t, σ²_{t|0}, σ²_{t|1})
                   │       │            │
                   │       └─ Regime-0 variance
                   └─ Regime probabilities
                              └─ Regime-1 variance

Step forward with r_t:
  σ²_{t+1|k} = ω_k + α_k · r_t² + β_k · σ²_{t|k}    [GARCH update per regime]
  π_{t+1}    = P.T @ π_t · f(r_t | regimes)            [Hamilton update]
  normalise π_{t+1}

P(in range)_t = 2·Φ(τ/σ_h) − 1
  where σ²_h = Σ_k π_{t|k} · σ²_{h|k}   [regime-weighted h-step variance]
```

The `load_from_state_dict()` method reconstructs the filter from the `to_state_dict()` snapshot stored in the window JSON.  The filter state at training end is `final_probs` (the filtered regime probabilities at the last training observation).

### 4.3 OU-Kou-GARCH Step Filter

The AEKF is stepped forward through OOS data using frozen MLE parameters.  The GARCH variance path is updated via one-step recursion; the AEKF updates `[X_t, κ_t, μ_t]` using each new OOS log-price as a measurement.

P(in range) at each OOS day uses the Normal approximation (not FFT) for speed:

```
σ²_h,GARCH = Σ_{j=1}^h E[σ²_{T+j}]     [analytic GARCH recursion]
ξ_h = (1 − e^{−2κ_t h/252}) / (2κ_t/252)   [OU compression]
σ²_jump = λ · (h/252) · Var(J)
σ_h = √(σ²_h,GARCH · ξ_h/h + σ²_jump)
P_t = 2·Φ(τ/σ_h) − 1
```

The Normal approximation is used instead of FFT to keep OOS evaluation tractable (FFT is 10–50× slower per call).  The approximation error is small for moderate horizon/threshold ratios.

---

## 5. AEKF OOS Diagnostics

Beyond P(in range), the AEKF produces a rich state sequence `[X_t, κ_t, μ_t]` over the OOS window.  These diagnostics assess whether the filter is tracking meaningfully.

### 5.1 Innovation Whiteness

```
Innovation: ν_t = x_t − x̂_{t|t-1}   [observed vs predicted log-price]
```

A well-functioning filter has white (uncorrelated) innovations.  Two tests:

**Ljung-Box on ν_t (ACF test):** `H0: innovations are serially uncorrelated`.  Low p-value (< 0.05) means the filter is lagging — it predicts the next observation poorly, indicating the process noise Q is too small or the OU dynamics are misspecified.

**Ljung-Box on ν_t² (ARCH test):** `H0: no ARCH effects in squared innovations`.  Low p-value means the filter's measurement noise (GARCH variance) is not absorbing all the heteroskedasticity — the GARCH component may need more lags.

**Innovation outlier rate:** Fraction of `|ν_t| > 2σ_ν`.  Under Gaussian innovations the expected rate is 4.6%.  Higher rates indicate jump contamination (expected for the Kou model) or structural breaks.  Values > 15% suggest the filter is frequently surprised.

### 5.2 κ Tracking Stability

The AEKF tracks time-varying mean-reversion speed κ_t.  Three metrics:

**κ OOS CV (coefficient of variation):** `std(κ_t) / mean(κ_t)`.  High CV (> 0.5) means the reversion speed is varying substantially over the OOS window — a sign of regime instability.  The model is still valid but the direction signal's confidence should be discounted.

**κ OOS min/max:** Direct range of tracked values.  `κ_min < 0.1` (half-life > 694 days) means the model is near-random-walk in some periods; `κ_max > 10` (half-life < 17 days) means strong mean reversion.  Both extremes are meaningful but should be flagged.

### 5.3 Direction Signal Accuracy

The direction signal at each OOS day is `drift_t = κ_t · (μ_t − X_t)`.  Positive drift = BULLISH (price below equilibrium).

**Direction accuracy:** Fraction of OOS days where `sign(drift_t) == sign(realized h-day return)`.  Baseline = 50%.  Values > 55% indicate the OU mean-reversion signal has genuine directional predictive power over this window.

**Direction AUROC:** AUROC treating `|drift_t|` as the score and `(realized_return > 0)` as the positive label.  Measures whether higher-confidence BULLISH signals correctly rank days with positive realized returns above days with negative realized returns.  More robust than accuracy because it doesn't depend on a decision threshold.

**Direction Brier / Brier Skill Score:** Uses `confidence_t = |drift_t| / (3·σ_drift)` (clipped to [0,1]) as the probability estimate for the BULLISH class.  Brier skill score > 0 means the confidence-weighted direction signal is better than a random forecast.

---

## 6. Window JSON Schema

All metrics appear under `model_weights.range_predictor.oos_scores`:

```json
{
  "n_samples":  54,
  "base_rate":  0.63,
  "ml": {
    "xgboost": {
      "brier_score":     0.21340,
      "brier_skill":     0.0811,
      "log_loss":        0.58201,
      "auroc":           0.5812,
      "balanced_acc":    0.5600,
      "mean_confidence": 0.6400
    },
    "lightgbm": { ... },
    "ensemble_ml": { ... }
  },
  "statistical": {
    "garch": {
      "brier_score":  0.23100,
      "brier_skill":  0.0026,
      "log_loss":     0.63400,
      "auroc":        0.4910,
      "balanced_acc": 0.4900,
      "mean_confidence": 0.7200,
      "rvol_mae":     0.04812,
      "rvol_bias":    0.03100
    },
    "msgarch": { ... },
    "oujump":  { ... }
  },
  "ensemble": {
    "brier_score": 0.19800,
    "brier_skill": 0.1527,
    ...
  },
  "aekf": {
    "lb_innovations_sq_pvalue":   0.23140,
    "lb_innovations_acf_pvalue":  0.44100,
    "innovation_outlier_rate":    0.0741,
    "kappa_oos_mean":             0.41200,
    "kappa_oos_cv":               0.3210,
    "kappa_oos_min":              0.10030,
    "kappa_oos_max":              0.89100,
    "direction_accuracy":         0.5556,
    "direction_auroc":            0.5923,
    "direction_brier":            0.24100,
    "direction_brier_skill":      0.0337
  }
}
```

---

## 7. How to Use These Metrics Across Windows

### Diagnosing GARCH rank inversion

Look at `statistical.garch.rvol_bias` in W08 and W09.  If it is strongly positive (e.g., > 0.05 annualised) those windows confirm the rank-inversion hypothesis: GARCH over-forecast vol, P(in range) was too low, AUROC < 0.50.  Compare to `statistical.msgarch.rvol_bias` and `statistical.oujump.rvol_bias` in the same windows — if MS-GARCH and OU-Kou-GARCH have smaller bias, it validates the architectural fix.

### Selecting ensemble weights

The fitted weights use CV AUROC (training time).  The OOS Brier Skill Score is the out-of-sample counterpart.  A model with positive CV AUROC but negative OOS BSS is over-fitting in training.  In future experiments, consider using `max(0, OOS_BSS)` as the weight source instead of (or alongside) CV AUROC to reduce in-sample optimism.

**CV AUROC = None vs 0.5:** A statistical model returns `cv_auroc = None` only when every fold failed both gating checks (see `GARCH_METHODOLOGY.md §12`):

- `None` — model could not be evaluated at all on this window (all folds too short or model fit failed). Model excluded from ensemble; falls back to equal prior if any other statistical model has a finite score.
- `0.5` — model was evaluated but had zero discriminatory skill (all single-class folds, or genuine AUROC=0.5). Model gets weight = 0 (no edge over baseline) but is not excluded.

**Single-class fold handling (Exp 17+ fix):** In low-volatility windows (e.g. W10–W12 with adaptive threshold ~5%), the in-range rate can reach 85–95%, making some CV folds all-positive. Prior to Exp 17's CV fix, these folds were silently dropped, causing `cv_auroc = None` even when the model had genuine signal in the mixed-class folds. The fix (implemented in `GARCHRangeModel._MIN_FOLD_LABELS = 10`) scores single-class folds as 0.5 and only skips folds with fewer than 10 labelable rows.

### AEKF sanity check

Before trusting the OU-Kou-GARCH direction signal in a given window:

1. `lb_innovations_acf_pvalue` > 0.05 — filter is not lagging (innovations are white)
2. `kappa_oos_cv` < 0.5 — reversion speed is stable enough for the signal to be meaningful
3. `direction_auroc` > 0.52 — the direction ranking has at least marginal skill

If any of these fail, the direction signal should be treated as unreliable for that window regardless of training-time BIC.

### Cross-window trend analysis

Plot `brier_skill` across W01–W12 for each model.  A model that has consistently positive BSS in range-bound windows (W03, W07, W09, W11) but negative BSS in trending windows (W05, W06) is a regime-conditional predictor — this is expected and acceptable.  Flat or consistently negative BSS across all windows means the model has no OOS skill.

---

## 8. Implementation Details

### Rolling GARCH forecasts

`GARCHRangeModel.roll_garch_forecasts()` refits on OOS returns alone (frozen spec from training BIC winner) rather than extending the training series.  The approximation is conservative: it ignores training history but avoids look-ahead bias from re-fitting with OOS data included.

### MS-GARCH `step_filter` approximation

The per-regime conditional variance `σ²_{t|k}` in `step_filter()` uses the previous filtered variance (not the regime-weighted blended variance as in the full EM E-step).  This is the standard "frozen filter" approximation used in online Hamilton filter implementations.  It slightly understates uncertainty in the blended forecast but avoids maintaining the full smoother state.

### AEKF OOS vs training

`oos_aekf_diagnostics()` initialises the filter at the training end-state `(X_T, κ_T, μ_T)` from `aekf_final_state` in the stored state dict.  The GARCH variance path initialises at the unconditional variance (not the last training sigma²) because the exact training-end sigma² is not stored.  This produces a slightly warm-up bias in the first ~20 days of OOS that decays as the GARCH recursion converges.

### P(in range) approximation for OU-Kou-GARCH OOS

Full CF-inversion via FFT is ~50ms per call.  At 60 OOS days, that is 3 seconds per window — acceptable.  The Normal approximation is used in `roll_oujump_forecasts()` for speed but the full FFT could be substituted by replacing `stats.norm.cdf(t_std)` with a call to `model.p_in_range()` after each AEKF step.

---

## 9. References

1. **Brier, G.W. (1950)**. "Verification of Forecasts Expressed in Terms of Probability." *Monthly Weather Review* 78(1): 1–3.
2. **Murphy, A.H. (1973)**. "A New Vector Partition of the Probability Score." *Journal of Applied Meteorology* 12(4): 595–600.
3. **Gneiting, T. & Raftery, A.E. (2007)**. "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association* 102(477): 359–378.
4. **Hamilton, J.D. (1989)**. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica* 57(2): 357–384.
5. **Mehra, R.K. (1970)**. "On the Identification of Variances and Adaptive Kalman Filtering." *IEEE Transactions on Automatic Control* 15(2): 175–184.
6. **Engle, R.F. (2002)**. "Dynamic Conditional Correlation." *Journal of Business & Economic Statistics* 20(3): 339–350. (GARCH rolling forecasts background)

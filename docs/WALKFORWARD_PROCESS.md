# Walk-Forward Backtesting & Optimization — Process Reference

> **What this document covers:** The complete mechanics of one walk-forward window, from data slicing through model training, Optuna optimization, and the full entry-gate pipeline during the OOS run. It explains *why* each step exists in the order it does, what can fail, and what the failure modes look like in results.
>
> **Related references:**
> - [GUIDE.md](../GUIDE.md) — system architecture overview, configuration reference, CLI flags, and the 5-step window flow diagram
> - [docs/WALKFORWARD_DASHBOARD.md](WALKFORWARD_DASHBOARD.md) — how to read the dashboard output from a completed run
> - [docs/GARCH_METHODOLOGY.md](GARCH_METHODOLOGY.md) — GARCH/MS-GARCH math and BIC selection
> - [docs/OU_KOU_GARCH_METHODOLOGY.md](OU_KOU_GARCH_METHODOLOGY.md) — OU-Kou-GARCH jump-diffusion and AEKF
> - [docs/ENSEMBLE_OOS_ASSESSMENT.md](ENSEMBLE_OOS_ASSESSMENT.md) — Brier score, AUROC, and calibration metrics

---

## Table of Contents

1. [The Walk-Forward Structure](#1-the-walk-forward-structure)
2. [Step-by-Step: What Happens Inside One Window](#2-step-by-step-what-happens-inside-one-window)
   - 2.1 [Step 1 — Window Slicing](#21-step-1--window-slicing)
   - 2.2 [Step 2 — ML Model Training](#22-step-2--ml-model-training)
   - 2.3 [Step 3 — Optuna Optimization](#23-step-3--optuna-optimization)
   - 2.4 [Step 4 — Meta-Labeler Training](#24-step-4--meta-labeler-training)
   - 2.5 [Step 5 — OOS Evaluation](#25-step-5--oos-evaluation)
3. [The Range Predictor Ensemble in Detail](#3-the-range-predictor-ensemble-in-detail)
4. [The Two-Layer Gate Architecture](#4-the-two-layer-gate-architecture)
   - 4.1 [Layer 1 — The Skill Gate](#41-layer-1--the-skill-gate)
   - 4.2 [Layer 2 — The Entry Gates](#42-layer-2--the-entry-gates)
   - 4.3 [The Training-Failure Hard Block](#43-the-training-failure-hard-block)
   - 4.4 [Full Decision Flow Diagram](#44-full-decision-flow-diagram)
5. [What Optuna Searches (and What It Cannot Touch)](#5-what-optuna-searches-and-what-it-cannot-touch)
6. [Why the Step Ordering Matters](#6-why-the-step-ordering-matters)
7. [Diagnosing Common Failure Patterns](#7-diagnosing-common-failure-patterns)

---

## 1. The Walk-Forward Structure

Walk-forward testing divides the full history into non-overlapping train/test pairs. Each pair is called a **window**. The test period is always genuinely out-of-sample: no data from it enters any training step.

```
Full history (e.g. 2024-05-01 → 2026-05-01)
─────────────────────────────────────────────────────────────

Window 1
├── TRAIN  [day 0   → day 364]   365 trading days
├── GAP    [day 365 → day 369]     5 trading days (information barrier)
└── TEST   [day 370 → day 429]    60 trading days  ← OOS result

Window 2
├── TRAIN  [day 60  → day 424]   365 trading days  (shifted by step_days=60)
├── GAP    [day 425 → day 429]     5 trading days
└── TEST   [day 430 → day 489]    60 trading days  ← OOS result

... (12 windows total at step_days=60 over ~2 years)
```

**The gap** is a 5-day information barrier between the end of training and the start of testing. It prevents the model from learning spurious patterns that exist only at the boundary (e.g., earnings announcements, month-end rebalancing) that would not be present in live trading.

**Key invariant:** The aggregate OOS result (all test windows combined) is the only performance number that matters. In-sample results from training periods are meaningless for evaluating live performance.

See [GUIDE.md §1.19](../GUIDE.md#119-backtesting) for the CLI commands to run a walk-forward experiment.

---

## 2. Step-by-Step: What Happens Inside One Window

The following steps execute sequentially for every window. **Order is not arbitrary** — see §6 for why changing it would corrupt the results.

### 2.1 Step 1 — Window Slicing

The walk-forward engine carves three date ranges from the full history DataFrame:

- `train_df` — the training slice (365 days)
- `gap_df` — the buffer (5 days, discarded)
- `test_df` — the OOS slice (60 days)
- `test_with_context` — `test_df` prepended with `context_bars` (default 252 extra days) so that feature computation at the start of the test period has enough lookback

The `context_bars` extension is important: features like the 50-day SMA or Hurst exponent need historical data to compute. Without context, the first ~50 bars of every test period would have NaN features.

### 2.2 Step 2 — ML Model Training

Two models are trained on `train_df` only. Both are frozen before Optuna runs.

**2a. Direction Predictor** (`src/ait/ml/ensemble.py`)

- **What:** 3-class classifier (BULLISH / NEUTRAL / BEARISH) for 5-day forward price moves
- **How:** XGBoost + LightGBM trained with walk-forward cross-validation and a 5-day purge gap between folds. Each symbol gets its own model pair; symbols with < 100 samples fall back to a universal model.
- **Metric:** Macro one-vs-rest AUROC
- **Role in iron condor:** Not an entry gate. It is passed to the backtester for other strategy types (directional spreads, long straddle) and for thesis re-evaluation during a live iron condor position.

**2b. Range Predictor** (`src/ait/ml/range_predictor.py`)

This is the iron condor's primary entry gate. It predicts `P(price stays within ±threshold% over the next N days)` — a binary classification task that is fundamentally different from direction prediction.

The range predictor is a **4-member ensemble** with CV-weighted blending:

| Member | Type | What it captures |
|--------|------|-----------------|
| XGBoost | Gradient-boosted trees | Cross-sectional feature patterns: IV rank, Hurst, realized vol, RSI, BB position, put-call ratio |
| LightGBM | Gradient-boosted trees | Same features, different inductive bias; better on large training sets |
| MS-GARCH | Markov-Switching GARCH | Volatility regime structure — whether the market is in a low-vol or high-vol regime, and how persistent that regime is |
| OU-Kou-GARCH + AEKF | Jump-diffusion + Kalman Filter | Mean-reversion speed, long-run drift, and jump intensity in the price process |

Each member produces an independent `P(in-range)` estimate. The final ensemble probability is a **weighted average**, where each member's weight is proportional to its CV balanced accuracy on the training fold:

```
P(in-range)_final = Σ_i  w_i × P_i(in-range)
                    where w_i ∝ max(0, CV_balanced_acc_i − 0.50)
```

Members with near-random CV skill (balanced accuracy ≈ 0.50) receive near-zero weight. Members that fail to converge (MS-GARCH, OU-Kou-GARCH numerical failure) receive zero weight and are excluded from the ensemble for that window.

The statistical models (MS-GARCH, OU-Kou-GARCH) are trained in a **subprocess** to isolate their RNG state from the parent process. If the subprocess fails, the engine falls back to the ML-only (XGBoost + LightGBM) ensemble.

> See [GARCH_METHODOLOGY.md](GARCH_METHODOLOGY.md) for MS-GARCH training and BIC selection.
> See [OU_KOU_GARCH_METHODOLOGY.md](OU_KOU_GARCH_METHODOLOGY.md) for the OU-Kou-GARCH process and AEKF.

**What happens when training produces no accuracy:**

If the range predictor's cross-validated balanced accuracy is below any useful threshold — typically because the training slice has too few `in-range` examples or the features are uninformative — training returns `(None, "training_returned_no_accuracy", threshold)`. This triggers the **hard block** described in §4.3.

### 2.3 Step 3 — Optuna Optimization

With both ML models frozen, Optuna searches for the best structural parameters for the strategy on the **training data**. This is described in detail in §5.

### 2.4 Step 4 — Meta-Labeler Training

After Optuna produces `best_params` (`window_cfg`), a meta-labeler is trained on the training data using those exact parameters:

1. A "shadow" backtester runs on `train_df` with `window_cfg` + the frozen direction model
2. Each simulated trade's outcome (profitable = 1, loss = 0) becomes a training label
3. XGBoost trains on `(feature_vector_at_entry, label)` pairs

The meta-labeler answers a different question from the range predictor: not "what will price do?" but "given this exact strategy configuration and this market state, is this particular trade likely to be profitable?" It is a second-order filter.

**Critical dependency on ordering:** The meta-labeler must be trained after Optuna because it needs `window_cfg` to simulate trades and generate labels. If trained before, the labels would reflect a different parameter set, and the meta-labeler would be calibrated to a strategy that doesn't match what actually runs in OOS.

### 2.5 Step 5 — OOS Evaluation

All artifacts from Steps 2–4 are frozen and applied to `test_df` for the first time. No retraining occurs. Every bar in the test window goes through the full entry decision pipeline (§4). P&L from this step is the reported OOS result.

**Timeseries export (Layer 2c):** After the OOS backtester runs, `_save_window_timeseries()` iterates over every bar in `test_df` and writes per-bar feature values and ML predictions to `timeseries_bars.json`. These are what populate the ML Predictions pane in the Walk-Forward Dashboard. The range predictor's Layer 1 skill gate (§4.1) applies here too — if the model has no edge, `range_prob` is written as `null` for every bar.

---

## 3. The Range Predictor Ensemble in Detail

Understanding the ensemble is necessary to interpret what you see in the Predictor Models tab of the dashboard.

**Training sequence for one window:**

```
train_df
    │
    ├── XGBoost.fit(X_train, y_range)     → cv_balanced_acc["xgboost"]
    ├── LightGBM.fit(X_train, y_range)    → cv_balanced_acc["lightgbm"]
    ├── MS-GARCH.fit(returns)             → cv_balanced_acc["msgarch"]   (subprocess)
    └── OU-Kou-GARCH.fit(returns)         → cv_balanced_acc["oujump"]    (subprocess)
                │
    Weight each member by: w_i = max(0, cv_acc_i − 0.50)
    Normalise: w_i ← w_i / Σ w_j
                │
    fitted_weights = {"xgboost": 0.32, "lightgbm": 0.33, "msgarch": 0.35, "oujump": 0.00}
    (example — actual weights vary per window)
```

The fitted weights are stored in `window_NNN.json` under `model_weights.range_predictor.fitted_weights` and visualised in the "Fitted Ensemble Weight" chart.

**The MIN_EDGE gate is applied per-symbol:** The `predict()` and `predict_from_features()` methods look up `_symbol_models["QQQ"]["cv_scores"]` to perform the edge check. This means the gate is evaluated against the symbol-specific CV scores, not a global average. A model that has edge for SPY but not QQQ will gate QQQ entries while still predicting for SPY.

**`min_edge_over_baseline` is Optuna-searchable (Exp 28+):** Rather than a hard-coded 0.10, the minimum edge threshold is now part of the iron condor search space with range [0.02, 0.15]. Optuna finds the value that maximises the training objective per window. A value of 0.02 accepts almost any skill; 0.15 requires strong evidence of predictive power.

---

## 4. The Two-Layer Gate Architecture

There are two completely different kinds of gate in this system. They are often confused because both result in "no trade entered", but they operate at different levels and for different reasons.

### 4.1 Layer 1 — The Skill Gate

**Where it lives:** Inside `RangePredictor.predict()` and `RangePredictor.predict_from_features()` in `src/ait/ml/range_predictor.py`.

**What it does:** Before computing any probability, the predictor checks whether it has demonstrated meaningful predictive skill:

```python
avg_balanced_acc = mean(cv_scores.values())
edge = avg_balanced_acc - 0.50      # balanced-accuracy baseline = 0.50
if edge < self.MIN_EDGE_OVER_BASELINE:   # currently Optuna-tuned [0.02, 0.15]
    return None                          # silently, no exception
```

**What "returning None" means:**
- In the timeseries export: `range_prob` is written as `null` → the blue line on the ML Predictions chart has a gap for every bar in that window.
- In the live OOS backtester: `rp is None` → the range gate check at `engine.py:392` treats this identically to a low-probability prediction and skips the entry.

**Why this gate exists:** A model that achieves balanced accuracy of 0.52 has a very small positive edge, but that edge is not statistically distinguishable from random variation over a 60-bar test period. Emitting probabilities from such a model provides false confidence. A model that says "P(in-range) = 0.72" when it's actually at chance level (0.50) would cause the system to enter iron condors in windows where it has no basis to do so. The Layer 1 gate ensures only models with demonstrated cross-validated skill emit predictions.

**What you see in the dashboard:** The "Windows Gated" KPI tile in the Predictor Models tab counts how many windows had `edge < threshold`. The bar charts in "Member Skill Across Windows" show the CV scores for all trained windows, including gated ones — so you can see the model's training performance even for windows where it never predicted.

See [WALKFORWARD_DASHBOARD.md §5.3](WALKFORWARD_DASHBOARD.md#53-the-quality-gate-why-predictions-go-missing) for the full dashboard interpretation.

### 4.2 Layer 2 — The Entry Gates

Layer 2 gates run inside the OOS backtester (`src/ait/backtesting/engine.py`) at each candidate entry bar, **after** Layer 1 has already passed (the predictor returned a non-None value). Each gate is a sequential check; the first failure skips to the next bar.

**Gate A — Range probability floor (`range_min_confidence`, default 0.55)**

```python
if rp is None or rp.probability_in_range < self._range_min_confidence:
    continue   # skip this bar
```

The raw ensemble `P(in-range)` must clear this floor. Even when the model has skill (Layer 1 passed), individual predictions may be weak — a value of 0.56 is barely above chance for a noisy binary problem. This gate ensures only high-conviction predictions gate entries.

`range_min_confidence` is **not** Optuna-searchable for iron condors (see §5). It is fixed at the config default (0.55).

**Gate B — Realized volatility ceiling (`max_entry_vol_annual`, default 0.80)**

```python
vol_10d = recent_close.pct_change().std() * sqrt(252)
if vol_10d > self._max_entry_vol_annual:
    continue
```

High 10-day realized vol means the iron condor's breakeven range is too narrow relative to expected moves. A condor entered at annualized vol > 80% would need strikes so far out that the credit received is negligible, or so close in that a 1-sigma move hits the short. This gate is fixed at 0.80 and excluded from Optuna (it would overfit to train-period vol spikes).

**Gate C — AEKF direction veto (`aekf_veto_threshold`, default 0.60)**

The OU-Kou-GARCH model's Adaptive Extended Kalman Filter continuously estimates the current mean-reversion drift direction (BULLISH or BEARISH) and a confidence value. If this confidence exceeds the threshold, the market is trending — which is bad for iron condors regardless of what `P(in-range)` says.

```python
if _ou_dir is not None and float(_ou_conf) >= self._aekf_veto_threshold:
    continue   # strongly trending mean-reversion drift → skip condor
```

This gate is orthogonal to `P(in-range)`: the range model works from cross-sectional features (IV rank, RSI, realized vol) while the AEKF works from the time-series dynamics of price itself. A situation where both say "range-bound" is more reliable than either alone. When they conflict (range model says OK, AEKF says trending), the AEKF veto wins.

**Gate D — Rising IV rank filter (`iv_rank_rise_threshold`, Optuna-searchable [0.25, 0.60])**

```python
iv_rank_rise = iv_rank_series.iloc[-1] - iv_rank_series.iloc[0]  # 10-day change
if iv_rank_rise > self._iv_rank_rise_threshold:
    continue
```

A rising IV rank signals the options market is pricing in increasing uncertainty. Even if current IV is moderate, the direction of change matters: selling vol into expanding IV means the options you sold will be marked against you immediately. The threshold is Optuna-tuned per window because the right sensitivity depends on the vol regime of that training slice.

**Gate E — Fractal regime gates (Optuna-searchable)**

Two Hurst-based checks:

```python
# Hard veto for strongly trending markets
if hurst_scale_spread > hurst_regime_threshold * hurst_hard_veto_multiplier:
    continue

# Soft penalty for borderline trending markets
elif hurst_scale_spread > hurst_regime_threshold:
    confidence *= (1.0 - hurst_regime_penalty)

# Multifractal irregularity check
if multifractal_width > multifractal_max_width:
    continue
```

The Hurst exponent measures long-range dependence (H > 0.5 = trending, H < 0.5 = mean-reverting). An iron condor needs the underlying to stay within a range — a strongly trending market (high Hurst spread) violates that assumption. The multifractal width measures irregularity across timescales; wide multifractal spectra indicate hidden tail risk.

All three parameters (`hurst_regime_threshold`, `hurst_regime_penalty`, `multifractal_max_width`) are Optuna-searchable because the right thresholds depend on the vol regime in the training window.

### 4.3 The Training-Failure Hard Block

If the range predictor's training fails entirely (returns `"training_returned_no_accuracy"` or raises an exception), `range_predictor` is set to `None`. The walk-forward engine then forces `range_min_confidence = 1.0` for that window's OOS run:

```python
if range_predictor is None and _range_model_status != "ok":
    _oos_range_min_conf = 1.0   # unreachable → blocks all IC entries
```

A confidence threshold of 1.0 is mathematically unreachable (probabilities are < 1.0 by definition). This completely blocks all iron condor entries for that window. The rationale: without a range gate, there is no basis for entering an iron condor — the strategy's entire edge is predicated on having a reliable estimate of whether the underlying will stay in range.

This is what happened for windows 6 and 7 in the current experiment run: the training data for those windows produced insufficient positive examples for the range model to learn from, training returned no accuracy, and no iron condors were entered in those OOS periods.

### 4.4 Full Decision Flow Diagram

```
TRAINING TIME (train_df)
────────────────────────
  RangePredictor.train()
    │
    ├── XGBoost CV → cv_balanced_acc["xgboost"]
    ├── LightGBM CV → cv_balanced_acc["lightgbm"]
    ├── MS-GARCH fit → cv_balanced_acc["msgarch"]
    └── OU-Kou-GARCH fit → cv_balanced_acc["oujump"]
    │
    edge = mean(cv_scores) − 0.50
    │
    Fit ensemble weights: w_i ∝ max(0, cv_acc_i − 0.50)
    │
    status = "ok"  OR  "training_returned_no_accuracy"

OOS RUN — PER BAR (test_df)
───────────────────────────

  ┌─ status == "ok"? ─────── NO ──→  range_min_confidence = 1.0
  │                                   (hard block: no IC entries this window)
  YES
  │
  ▼
  predict_from_features(feat_row)   ← called for every test bar (timeseries export)
  predict(hist)                     ← called at each entry candidate (backtester)
  │
  ┌─ edge < min_edge_over_baseline? ─ YES ──→  return None  (Layer 1 gate fires)
  │                                             range_prob = null in timeseries
  NO
  │
  ▼
  Compute P(in-range) = Σ w_i × P_i    [Layer 1 passes — prediction emitted]
  range_prob written to timeseries → BLUE LINE on dashboard chart
  │
  AT EACH ENTRY CANDIDATE
  │
  ├── Gate A: P(in-range) ≥ range_min_confidence (0.55)?   NO → skip
  ├── Gate B: vol_10d ≤ max_entry_vol_annual (0.80)?        NO → skip
  ├── Gate C: AEKF direction conf < aekf_veto_threshold?    NO → skip
  ├── Gate D: iv_rank_rise_10d ≤ iv_rank_rise_threshold?    NO → skip
  └── Gate E: hurst / multifractal within bounds?           NO → skip (or penalise)
  │
  ALL PASS
  │
  Meta-labeler filter: P(profitable) ≥ 0.50?              NO → skip
  │
  ENTRY TAKEN ✓
```

**Reading the dashboard in light of this flow:**

| What you observe | What it means |
|---|---|
| Blue line absent for most windows | Layer 1 fired: model had no edge (or training failed) |
| Blue line present, but few trades | Layer 2 filtered most bars: P(in-range) was rarely ≥ 0.55, or vol/AEKF/IV-rise gate was active |
| Many bars with blue line ≥ 0.55, but zero trades | Likely the hard block (range_min_confidence=1.0) from training failure, or all bars failed vol gate |
| "Windows Gated: 10/12" in dashboard | 10 windows never emitted any `range_prob` values (Layer 1); only 2 windows cleared the edge threshold |

---

## 5. What Optuna Searches (and What It Cannot Touch)

Optuna runs on the **training data only** with the ML models frozen. It can only search structural parameters — parameters that can be set before knowing what the market will do.

**Iron condor search space (current as of Exp 28):**

| Parameter | Range | What it controls | Why it's searchable |
|-----------|-------|-----------------|-------------------|
| `delta_short` | [0.15, 0.30] | Short strike placement — how far OTM the sold options are | Optimal distance depends on vol regime |
| `max_hold_days` | [14, 40] | DTE at entry + maximum hold duration | Optimal theta decay window varies by regime |
| `wing_k` | [0.30, 2.00] | Wing width = max loss per contract | Risk/reward trade-off is regime-dependent |
| `iv_rank_rise_threshold` | [0.25, 0.60] | Blocks entry when IV rank rising | Sensitivity to IV drift depends on vol regime |
| `min_edge_over_baseline` | [0.02, 0.15] | Layer 1 skill gate minimum edge | Optimal quality bar varies with training data quality |
| `hurst_regime_threshold` | [0.08, 0.30] | Hard fractal veto threshold | Trending sensitivity varies by period |
| `hurst_regime_penalty` | [0.00, 0.25] | Soft penalty for borderline trending | Grey-zone width is regime-dependent |
| `multifractal_max_width` | [0.30, 0.65] | Blocks irregular price processes | Tail-risk tolerance varies by regime |

**Frozen for iron condor (not searchable):**

| Parameter | Fixed value | Why it cannot be searched |
|-----------|-------------|--------------------------|
| `stop_loss_pct` | 0.35 | Searching it overfit train-path noise (Exp 18 H1 test); risk management constant, not regime-dependent |
| `profit_target_pct` | 0.50 | Same — frozen at ablation default |
| `trailing_stop_fraction` | 0.70 | Same |
| `range_min_confidence` | 0.55 | Searching it drove Optuna to 0-trade OOS solutions (Exp 2–4) |
| `max_entry_vol_annual` | 0.80 | Same |
| `iv_floor` | 0.12 | Searching it created train/OOS mismatch — Optuna found different values for each (Exp P8) |
| `spread_base/iv_sensitivity/dte_sensitivity/cap` | config | Calibrated from real market data; must stay fixed to avoid friction mismatch |
| ML model weights/trees | from Step 2 | Frozen before Optuna starts — same signal for all 200 trials |

**Why the ML models must be frozen during Optuna:**

Each Optuna trial runs the full backtester on every training bar. If the ML models were refit per trial, the training would take ~200× longer and, worse, Optuna would be co-optimizing the model hyperparameters and the strategy parameters jointly — a 200-dimensional problem where interactions cannot be disentangled. More critically: the ML models would be re-trained on training data that includes the target variable the strategy is optimizing, creating a complex leakage path. The correct separation is: train ML models once → freeze → let Optuna search structural params only.

**Warm-start between windows:**

Before each window's Optuna study begins, the engine checks the previous window's OOS result. If `win_rate ≥ 75%` AND `total_trades ≥ 3`, the previous window's best parameters are enqueued as trial #0 (warm start). If that condition fails, the globally best params seen across all prior windows are used instead. Only if neither source exists does the study start cold.

This per-strategy warm-start speeds convergence and prevents early windows (with fewer training examples) from wasting the entire trial budget exploring bad regions.

---

## 6. Why the Step Ordering Matters

The 5-step ordering (Slice → Train ML → Optuna → MetaLabeler → OOS) is not arbitrary. Each ordering dependency has a specific reason:

**ML training before Optuna:**
Optuna must search parameters against the same frozen ML signal that will run in OOS. Before Exp 10, models were trained after Optuna — Optuna optimized against a different signal than what actually ran in OOS. This produced inflated training metrics and poor OOS generalisation. Freezing first ensures consistency.

**Optuna before MetaLabeler:**
The MetaLabeler trains on simulated trade P&L outcomes, which depend on `window_cfg` (stop_loss_pct, profit_target_pct, wing_k, etc.). If MetaLabeler ran before Optuna, it would learn to classify trades under a different parameter set than what actually executes in OOS. The MetaLabeler's labels must reflect the exact same strategy configuration that will run in Step 5.

**Everything before OOS evaluation:**
This is the fundamental invariant of walk-forward testing. Violating it (using any OOS data to inform any training decision, even choosing a threshold) invalidates the OOS result as an unbiased performance estimate.

---

## 7. Diagnosing Common Failure Patterns

### No trades in OOS window

**Check in order:**
1. Was training successful? → `window_NNN.json`: `model_weights.range_predictor.status`
   - `"training_returned_no_accuracy"` → hard block active, no IC entries possible
   - `"ok"` → training succeeded, move to step 2
2. Did Layer 1 fire? → `model_weights.range_predictor.cv_scores` in window JSON
   - Compute `edge = mean(cv_scores) − 0.50`; compare to `min_edge_over_baseline` from best_params
   - If edge < threshold → no predictions emitted, no IC entries possible
3. Did predictions exist but fail Layer 2? → Check timeseries_bars.json for the window
   - If `range_prob` values exist (not null) → predictions were emitted
   - Check the Optuna best_params for `delta_short`, `max_hold_days` — may be set to values that produce zero qualifying candidates in the 60-day OOS window

### OOS return much lower than train return

This is the primary indicator of overfitting. Check:
1. Were `stop_loss_pct`, `profit_target_pct`, `trailing_stop_fraction` in the search space? They should be frozen. If they were searchable, Optuna overfit them to train-path P&L sequences.
2. Is `min_edge_over_baseline` very low (< 0.04)? A very permissive skill gate accepts weakly-trained models that look better in-sample than OOS.
3. Are Optuna's best-trial parameters at the edge of their search bounds? (e.g., `delta_short = 0.15` exactly, `max_hold_days = 40` exactly) → search bounds may be too tight, Optuna is hitting a wall.

### ML Predictions gap (blue line absent) for many windows

This is expected and explained in §4.1. Check the "Windows Gated" KPI in the Predictor Models tab. If most windows are gated, it suggests:
- The training data regime is not producing sufficient `in-range` label diversity for the model to learn meaningful patterns
- The `min_edge_over_baseline` threshold (if set via Optuna or config) may be appropriate, but consider whether the feature set adequately captures range-boundedness

### MS-GARCH or OU-Kou-GARCH showing zero weight in many windows

Convergence failures in the statistical models are normal and expected for certain data regimes. The ensemble automatically handles this by assigning zero weight. If both statistical models consistently fail across all windows, check whether the training slice contains enough return variation for GARCH estimation (very low-vol periods can produce flat likelihood surfaces).

See [GARCH_METHODOLOGY.md §4](GARCH_METHODOLOGY.md) for MS-GARCH convergence diagnostics.

---

*Generated: 2026-06-08. Reflects codebase as of Exp 28 / commit `13d2006`.*

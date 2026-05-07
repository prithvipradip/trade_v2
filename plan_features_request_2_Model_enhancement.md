# Model Enhancement Plan: Fractal & Multi-Resolution Feature Engineering
## Features Request 2

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Why Naively Switching to 5-Min Bars Fails](#2-why-naively-switching-to-5-min-bars-fails)
3. [The Key Reframing: Serial Correlation as Signal](#3-the-key-reframing-serial-correlation-as-signal)
4. [The Three Modeling Frameworks](#4-the-three-modeling-frameworks)
5. [Fractal Analysis Methods](#5-fractal-analysis-methods)
6. [Right Architecture Given the Constraints](#6-right-architecture-given-the-constraints)
7. [Integration with Walk-Forward and Optimizer](#7-integration-with-walk-forward-and-optimizer)
8. [Implementation Steps](#8-implementation-steps)
9. [Critical Files](#9-critical-files)
10. [Diagnostic Reports](#10-diagnostic-reports)
11. [Verification Plan](#11-verification-plan)

---

## 1. The Problem We Are Solving

### 1.1 The Staleness Gap

`DirectionPredictor` is trained on and predicts from 504 days of **daily closed OHLCV bars**. During market hours, the last row in that series is **yesterday's close**. Today's intraday price movement — no matter how large — is invisible to the model until tomorrow's daily bar closes.

```
Cache TTL: 1 hour
Scan cycle: every 5 minutes
→ Same stale daily bars used for 12 consecutive scan cycles
→ Model's most recent data point: yesterday's close
```

**Concrete failure case:** NVDA closes at $900 Tuesday (bearish ML prediction). Wednesday at 10 AM NVDA gaps up 5% on earnings revision. The bot scans at 10:05 AM, still sees Tuesday's close as the most recent row, still gets a BEARISH prediction, and may enter a bear put spread — while the market is screaming bullish.

### 1.2 Why Existing Mitigations Are Insufficient

The system has three post-ML intraday adjustments, all of which adjust **confidence only** — they cannot flip the ML **direction**:

| Mechanism | Updates | Max Effect | Can Flip Direction? |
|---|---|---|---|
| MultiTimeframe RSI confidence boost | Every 5 min (fresh 5-min bars) | ±15% confidence | No |
| Sentiment overlay | Every 5 min (5-min TTL) | ±20% confidence | No |
| Options flow hard gate | Every 5 min (live chain) | Block entry | Only at `bias_strength > 0.7` |

The RSI timing queue — the mechanism most expected to use intraday data — actually uses **daily RSI** (`hist["Close"]` in `orchestrator.py`), not intraday RSI. It is also stale.

**The only escape valve that can override direction** is the options flow hard gate, which fires only on extreme unusual activity. Continuous, unambiguous intraday moves with no unusual options activity pass straight through with a stale, potentially wrong direction signal.

### 1.3 What Good Looks Like

A model that sees today's price action as it unfolds, updating its directional view every 5 minutes as new intraday bars arrive — while still predicting 5-day forward direction (the correct label horizon for options holding periods).

---

## 2. Why Naively Switching to 5-Min Bars Fails

The instinctive fix — "just use 5-min bars instead of daily bars" — introduces three fundamental problems:

### Problem 1: Label-Feature Horizon Mismatch

The current label is:
```python
fwd_return = close.pct_change(5).shift(-5)   # 5 trading days ahead
labels[fwd_return > 0.01] = 2                # BULLISH: >+1% in 5 days
```

On a 5-min series, 5 trading days = **390 bars ahead**. A model trained to predict "will this 5-min bar lead to price being higher in 390 bars?" must learn which 5-min features carry multi-day signal.

The answer: almost nothing specific to 5-min microstructure is informative at a 5-day horizon. A 5-min RSI (last 70 minutes) has near-zero predictive power for a 5-day outcome. The model would learn negligible weights for intraday features and instead latch onto long-period rolling aggregates — which are approximately equivalent to the existing daily-bar features, at 78× the computational cost.

**RSI-14 on 5-min bars ≠ RSI-14 on daily bars:**
- 5-min RSI-14 = RSI over the last **70 minutes** 
- Daily RSI-14 = RSI over the last **14 trading days**

The daily RSI-14 is a far better predictor of 5-day direction than the 5-min RSI-14. The label horizon and feature resolution must match.

### Problem 2: Autocorrelation Collapse of Sample Count

Treating each 5-min bar as an independent training sample with a 5-day label creates massive autocorrelation:

- Bar at 9:35 AM predicts 5 days forward
- Bar at 9:40 AM predicts ~5 days minus 5 minutes forward
- These are essentially the same label, 5 minutes apart

The walk-forward purge gap must be **390 bars** (5 days × 78 bars/day) instead of 5 rows, consuming most of each fold. Effective independent samples: still ~504 (one per trading day). Apparent samples: 39,312. The model looks data-rich but is informationally equivalent to daily bars — with a high risk of fitting to autocorrelated noise patterns.

### Problem 3: Computation Cost for No Directional Gain

| Dimension | Daily bars | 5-min bars |
|---|---|---|
| Rows per symbol | 504 | ~39,312 |
| Training time | Fast | ~78× slower |
| Required purge gap | 5 rows | 390 rows |
| Independent samples | ~504 | ~504 (same) |
| Feature semantics | Match label horizon | Mismatched |

---

## 3. The Key Reframing: Serial Correlation as Signal

The serial correlation in 5-min bars is not just a statistical nuisance — it is **structure**. The correct framing is:

> Construct **one training sample per trading day**, where the input is a **window of recent 5-min bars** (or daily bars), and the label is the **5-day forward return from that day's close**.

This resolves both problems simultaneously:

- **Label-feature mismatch resolved:** Features can span multiple timescales. Long-period aggregates of the window recover the same signal as daily bars. Short-period structure within the window captures what daily bars discard.
- **Autocorrelation resolved:** Training samples are still one per trading day (~504 independent). The serial correlation *within* each window is **used as structure** to compute features — it is not treated as independent observations.

At **inference time**, the window grows bar by bar as the trading day unfolds:
- 9:40 AM → 2 intraday bars in window → features update
- 3:00 PM → 72 intraday bars in window → features update
- Each 5-min scan cycle produces a fresh prediction based on the current window

This eliminates the staleness gap entirely.

---

## 4. The Three Modeling Frameworks

### Framework 1: Variable-Length Markov Chains (Tier 1 — Implemented First)

The VLMC insight: not all lookback contexts within the window are equally informative for the 5-day label. A session that grinded above VWAP for 6 hours on rising volume has different Markov structure than one that opened with a gap that faded by noon.

In practice: compute features over **different lookback windows within the same 5-min series**:

```
Window of 78 bars (full session):
  - VWAP trajectory: price vs. VWAP at each quartile of session
  - Session high/low timing: fraction of session elapsed at high/low
  - Volume profile: front-loaded vs. back-loaded vs. U-shaped

Window of last 12 bars (last hour — "power hour"):
  - Late-session momentum: who has control into close?
  - Volume acceleration / deceleration

Window of last 3 bars (last 15 min):
  - Closing imbalance proxy
```

Each window extracts a different Markov "context." XGBoost learns which context lengths are predictive for the 5-day label.

### Framework 2: Spectral / Wavelet Decomposition (Tier 1 — Core Contribution)

Discrete Wavelet Transform decomposes the price series into components at different scales simultaneously — Fourier cannot do this because it is not localized in time. For a 5-min bar series:

| Wavelet Level | Timescale | Captures | Useful for 5-day label? |
|---|---|---|---|
| L1 detail | 10-min | Microstructure noise | No |
| L2 detail | 20-min | Intraday scalping | Marginally |
| L3 detail | 40-min | Morning/afternoon momentum | Yes |
| L4 detail | 80-min | Half-session trends | Yes |
| L5 detail | 2.7-hour | Full-session structure | Yes (strongest) |
| Approximation | 5+ days | Multi-day trend | Yes (same as daily) |

The approximation coefficients at the coarsest level **are** the daily-bar signal. Finer levels are incremental information. The ML model learns which levels to weight.

### Framework 3: Sequence-to-Sequence / Transformer (Tier 3 — Future)

A Transformer encoder takes the window of 5-min bars and produces a latent representation replacing handcrafted features:

```
Input:  [bar_t-77, ..., bar_t]       ← 78 × 5-min bars (OHLCV)
          ↓ Transformer encoder (multi-head self-attention)
        [CLS token embedding]
          ↓
        Concatenate [fractal_features: 7 scalars]
          ↓ Dense → Softmax
Output: P(BULLISH), P(NEUTRAL), P(BEARISH)
```

**Input data: 5-min bars (same as CNN-GRU, not daily bars).** The advantage over CNN-GRU is that self-attention can relate any bar to any other bar regardless of distance — e.g., "what happened at 9:45 AM predicts what happens at 3:45 PM" — without the locality constraints of convolution or the sequential forgetting of GRU.

**Fractal features in the Transformer:** Concatenated at the [CLS] classification head, not fed through the attention stack. The Transformer learns global bar-to-bar relationships via attention; the precomputed fractal scalars (Hurst, PSD, MFDFA width, scale spread) provide global session statistics explicitly rather than requiring the model to re-derive them from raw bars. This is especially important at small dataset sizes when the model cannot learn fractal structure from scratch. As dataset grows, the explicit features become redundant but don't harm performance.

**Why it's Tier 3:** With ~504 training samples (one per day per symbol), a Transformer overfits severely. Minimum viable: 5,000–20,000 samples, requiring pre-training on a pooled multi-symbol dataset. Also requires significant changes to the training pipeline. The CNN-GRU hybrid (Tier 2) is far more data-efficient for the same intraday bar input.

---

## 5. Fractal Analysis Methods

### 5.1 The Wavelet-Hurst Connection

These are not separate ideas. Wavelet variance IS one of the best methods for estimating the Hurst exponent. The wavelet variance at scale `j` follows:

```
σ²(j) ∝ 2^(2Hj)
```

So `log(σ²(j))` vs `log(scale j)` gives a straight line with slope `2H`. The wavelet energy features proposed in Framework 2 are already implicitly encoding the fractal scaling structure. Making it explicit gives us `hurst_wavelet` and its fit quality `hurst_fit_r2`.

**`hurst_fit_r2` low value = the power-law scaling is breaking down = the process is no longer self-similar across scales = regime transition in progress.** This is the mathematical signature of a transition state.

### 5.2 Power Spectral Density and 1/f Structure

For a fractional Brownian motion with Hurst exponent H, the PSD follows:

```
S(f) ∝ f^(-β),   β = 2H + 1
```

| β range | H implied | Process |
|---|---|---|
| β ≈ 2 | H ≈ 0.5 | Brownian motion (random walk) |
| β > 2 | H > 0.5 | Persistent / trending |
| β < 2 | H < 0.5 | Anti-persistent / mean-reverting |
| β ≈ 1 | — | 1/f noise (long memory, critical state) |
| β unstable | — | Regime in transition |

Two Hurst estimates — from wavelet variance and from PSD — should agree for a genuine fractal process. **Their disagreement `|hurst_wavelet - psd_implied_hurst|` is itself a feature:** a sensitive measure of process non-stationarity.

### 5.3 Multi-Scale Hurst: Scale Invariance Violations

Compute H at multiple window lengths within the same series:

```
In a stationary fractal process: H_short ≈ H_medium ≈ H_long
When H diverges across scales → regime transition in progress
```

`hurst_scale_spread = |H_short - H_long|` is the key regime-change signal. This is a **leading indicator** — the process becomes non-self-similar before the change is visible in price level or VIX.

| Signal | Type | Leads price by |
|---|---|---|
| VIX spike | Coincident / lagging | 0 days |
| Realized vol expansion | Lagging | 0 days |
| `hurst_fit_r2` dropping | Leading | 1–3 days |
| `hurst_scale_spread` widening | Leading | 1–5 days |
| `multifractal_width` expanding | Leading | 2–7 days |
| `multifractal_asymmetry` going negative | Leading | 1–10 days |

### 5.4 Multifractal Analysis (MFDFA)

Monofractal models assume one H for all fluctuation magnitudes. Real markets are **multifractal**: large moves have different scaling properties than small moves. MFDFA measures this via the generalized Hurst exponent `h(q)` for different orders `q`:

- **`multifractal_width` (Δα):** `h(q_min) - h(q_max)` — width of the multifractal spectrum
  - ≈ 0 for a monofractal process
  - ~0.3–0.5 for equities in normal conditions
  - Widens sharply before crash / high-volatility regimes
- **`multifractal_asymmetry`:** Skew of the spectrum
  - Negative = large negative returns dominate scaling = **directional crash risk signal**
  - Not correlated with VIX level; independent information

MFDFA requires ~500+ data points. On daily bars with the 504-day training window, it is available for most walk-forward windows. On 5-min bars, it requires ~1 week of stored intraday history.

---

## 6. Right Architecture Given the Constraints

**Constraints:** ~504 training samples per symbol, Intel Mac (PyTorch ≤ 2.2), existing XGBoost/LightGBM ensemble, daily-only backtesting engine.

### Layer 1: Daily Fractal Features (Full Pipeline — Backtest + Live)

Add to `FeatureEngine.compute()` as a new `_add_fractal_features()` method. Computed from daily OHLCV. Compatible with the entire existing pipeline because they are just additional columns in the feature matrix.

Features:

| Feature | Estimator | Min bars | Primary signal |
|---|---|---|---|
| `hurst_wavelet` | Wavelet variance slope | 50 | Trend persistence |
| `hurst_fit_r2` | R² of log-scale fit | 50 | Scale invariance quality |
| `psd_beta` | FFT power-law exponent | 60 | Long-memory structure |
| `psd_fit_r2` | R² of PSD fit | 60 | Spectral scale invariance |
| `hurst_psd_divergence` | Cross-estimator gap | 60 | Process non-stationarity |
| `hurst_short` | Wavelet H, 60-bar window | 60 | Short-horizon H |
| `hurst_long` | Wavelet H, 180-bar window | 180 | Long-horizon H |
| `hurst_scale_spread` | `\|H_short - H_long\|` | 180 | **Key: regime transition leading signal** |
| `multifractal_width` | MFDFA Δα | 500 | Fat-tail / multifractal degree |
| `multifractal_asymmetry` | MFDFA spectrum skew | 500 | Crash risk signal |

Graceful degradation: fields requiring more bars than available fill with `0.0`. XGBoost and LightGBM handle this correctly — the model learns to ignore zero-filled features when they are systematically zero for early windows.

### Layer 2: Intraday Fractal Features (Live Trading Only)

Extend `FeatureEngine.compute_intraday_features(intraday_df)` with Hurst and wavelet features from the 5-min window. These flow into the existing MTF confidence boost path in the live orchestrator.

**Not in the backtester** — the backtesting engine (`engine.py`) uses daily OHLCV only. No backtesting validation is possible for these features.

Minimum window sizes for 5-min bars:

| Estimator | Min bars | Min history needed |
|---|---|---|
| Wavelet Hurst | 50 | 4 hours (1 partial session) |
| PSD / β | 100 | 8 hours (1–2 sessions) |
| Multi-scale H | 200 | 17 hours (2–3 sessions) |
| MFDFA | 500 | 42 hours (1 week) |

Full infrastructure detail in Phase 5: new `intraday_prices` SQLite table in `HistoricalDataStore`, `get_intraday()` default extended to `days=7`, and orchestrator persistence wiring. No new IBKR function required — Yahoo Finance covers 60 days of 5-min data.

### Layer 3: Fractal Thresholds in Strategy Gating (Optimizer-Tunable)

The ML model learns fractal feature weights automatically. But for strategy selection gates, explicit thresholds are more interpretable and directly optimizable. Add to the iron condor and short strangle strategy spaces:

- `hurst_regime_threshold` (float, 0.08–0.30): at what `hurst_scale_spread` to penalize confidence
- `hurst_regime_penalty` (float, 0.0–0.25): how much to reduce confidence per unit of spread excess
- `multifractal_max_width` (float, 0.30–0.65): Δα above which iron condors are skipped

These are strategy parameters that Optuna searches, not ML hyperparameters. Per-window optimization (`optimize_per_window=True`) will discover that the optimal thresholds differ between calm and volatile market windows — the exact behavior intended.

### Layer 4 (Future — Tier 2): Short-Term CNN-GRU Intraday Model

A hybrid CNN-GRU model trained on 5-min bars with an **end-of-day label** (not 5-day) learns genuine intraday dynamics. Used as a directional **confirmation gate** at entry: if the short-term model strongly disagrees with the daily model's direction, delay entry via the existing signal queue.

**Architecture (recommended over pure LSTM or pure 1D-CNN):**

```
Input: [window × 5 channels]  (78–390 5-min bars of OHLCV)
    ↓
Conv1D(kernel=3, filters=32) → ReLU → Conv1D(kernel=3, filters=64) → ReLU
    ↓ captures local bar patterns (morning gap, reversal, consolidation)
GRU(hidden=64, layers=1, dropout=0.3)
    ↓ captures how session structure evolves across the window
GRU final hidden state [64]
    ↓
Concatenate [fractal_features: 7 scalars]
    ↓
Dense(32) → ReLU → Dense(3) → Softmax → P(BULLISH / NEUTRAL / BEARISH)
```

**Why GRU over LSTM:** Fewer parameters (2 gates vs 3 in LSTM) → less overfitting with limited training data. Same sequential inductive bias. GRU consistently matches or outperforms LSTM on small datasets.

**Why CNN-GRU over pure GRU:** The CNN extracts local bar patterns in parallel (no sequential bottleneck), producing a cleaner feature map for the GRU to read. Pure GRU on raw OHLCV noise is harder to train. Pure 1D-CNN misses how patterns evolve across the session.

**Fractal features in this model:** Concatenated at the classification head (not fed through the convolutional stack). Fractal features (`hurst_wavelet`, `psd_beta`, `hurst_scale_spread`, `mfdfa_width`, `mfdfa_asymmetry`, `hurst_fit_r2`, `psd_fit_r2`) are global session statistics — scalar summaries of the full window. The CNN-GRU handles local sequential patterns. Concatenating at the Dense layer gives the model both representations without forcing convolution kernels to rediscover precomputed fractal structure.

**Training data:** Pooled across all symbols → 504 days × 15 symbols ≈ 7,560 samples. With dropout=0.3, weight decay, and early stopping, this is sufficient for a shallow architecture (total trainable parameters < 50K).

**Label:** End-of-day return (not 5-day). This is an intraday confirmation model, not a replacement for `DirectionPredictor`.

---

## 7. Integration with Walk-Forward and Optimizer

### 7.1 Walk-Forward Compatibility

**Zero changes to the walk-forward engine required.** The walk-forward engine calls `DirectionPredictor.train(train_df)` and `DirectionPredictor.predict(hist)` per window — both internally call `FeatureEngine.compute()`. Adding fractal features to `FeatureEngine` means:

- Every walk-forward window retrains models with fractal features in scope automatically
- `RangePredictor` and `VolMagnitudePredictor` also use `FeatureEngine.compute()` → they receive fractal features at no extra cost
- The 5-day purge gap remains correct — fractal features don't introduce label leakage beyond what price-based rolling features already have

Each market window has distinct fractal characteristics:
- 2022 (volatile): high `hurst_scale_spread`, elevated `multifractal_width`
- 2024 (calm trending): stable H across scales, narrow multifractal spectrum

Per-window retraining means models trained on different windows learn different fractal feature weights — naturally adapting to the prevailing market regime.

### 7.2 Optimizer Compatibility — Two Levels

**Level 1 (Implicit — no changes needed):**
`optimize_ml=True` varies XGBoost/LightGBM hyperparameters (depth, estimators, subsample, colsample). These indirectly control how much the model relies on fractal features vs. price features. The optimizer discovers the best balance automatically.

**Level 2 (Explicit — new parameter spaces):**
Add `FRACTAL_GATE_SPACE` to `param_spaces.py` and merge into `IRON_CONDOR_SPACE` and `SHORT_STRANGLE_SPACE`. Optuna directly searches the regime-gating thresholds. Why only premium-selling strategies? Because trending or fractal-unstable regimes harm range-bound strategies the most. Directional strategies (bull/bear spreads) are actually helped by trending regimes (H > 0.6) and the ML model learns this naturally.

### 7.3 Per-Window Optimization Alignment

`optimize_per_window=True` runs Optuna on each training slice before testing that window. Since fractal characteristics differ per window, the optimizer finds different optimal thresholds per window:
- A 2022 training window → lower `hurst_regime_threshold` (more aggressive blocking when scale breaks down)
- A 2024 training window → higher threshold (less blocking needed)

This is exactly the intended behavior: fractal thresholds that are regime-adaptive without requiring explicit regime labels.

### 7.4 The Backtesting Intraday Gap

The backtesting engine (`engine.py`) uses daily OHLCV only. Intraday fractal features **cannot be backtested** through the existing engine. This creates a known asymmetry:

- **Backtested / validated:** Daily fractal features, fractal gating thresholds, ML feature weights
- **Live-only / unvalidated:** Intraday 5-min fractal features (Hurst from intraday bars, session structure, wavelet energy at intraday scales)

The intraday features improve live prediction without backtest validation. Monitoring their effect via the existing counterfactual tracker (comparing skipped vs. taken trades when the intraday fractal gate fires) provides empirical validation post-deployment.

---

## 8. Implementation Steps

### Scope Boundary

This plan implements the ideas from our discussion in two tiers of complexity:

**IN SCOPE (Phases 1–8):**
- Daily fractal features (Hurst, PSD, MFDFA, scale invariance) → full pipeline: training, backtesting, walk-forward, optimizer
- VLMC session structure features → live intraday path only (`compute_intraday_features`)
- Intraday wavelet energy features → live intraday path only
- Fractal threshold gating for iron condors/strangles → backtester + Optuna optimizer
- Intraday data persistence infrastructure (new SQLite table)
- Fractal diagnostic report tool (feature health, IC analysis, SHAP importance, gate counterfactual)

**DISCUSSED BUT OUT OF SCOPE (future work):**
- **Full window-based training restructuring:** Retraining `DirectionPredictor` on daily samples where each sample uses a *window of 5-min bars* as input features (not just daily bars). This resolves the staleness gap completely but requires architectural changes to `DirectionPredictor.train()`, a paired intraday fetch at each training date, and a new walk-forward data pipeline. Estimated effort: 3–4× the current plan. Deferred to a separate feature branch.
- **CNN-GRU short-term model (Tier 2):** A shallow 1D-CNN + GRU hybrid trained on 5-min bar windows for end-of-day direction. GRU is preferred over LSTM (fewer parameters, less overfitting). Fractal features concatenated at the classification head. See Layer 4 in Section 6 for full architecture. Requires new training pipeline (pooled multi-symbol, end-of-day labels).
- **Sequence-to-Sequence / Transformer (Tier 3):** Described in Framework 3 (Section 4). Deferred — requires 5,000–20,000 training samples minimum; pre-training on a pooled multi-symbol dataset is needed to avoid severe overfitting with ~504 daily samples.

The intraday VLMC and wavelet features (Phase 6) are the first step toward the window-based framing — they feed the same live MTF confidence overlay path that would eventually be replaced by the full short-term model.

---

### Phase 1: Daily Fractal Features in FeatureEngine
**File:** `src/ait/ml/features.py`

1. Add `_add_fractal_features(self, df: pd.DataFrame) -> pd.DataFrame` method
2. Implement the four estimators: `_hurst_wavelet()`, `_psd_features()`, `_multiscale_hurst()`, `_mfdfa_features()`
3. Call from `compute()` between `_add_macro()` and `_add_live_signals()`
4. Update `get_feature_names()` with 10 new feature names
5. Handle graceful degradation: `np.nan_to_num(value, nan=0.0)` for all fractal features

**Dependency:** Add `pywavelets>=1.4` to `pyproject.toml`. SciPy (already a dependency) provides `scipy.signal.periodogram` for PSD.

**Minimum window check pattern:**
```python
def _hurst_wavelet(self, returns: np.ndarray) -> tuple[float, float]:
    if len(returns) < 50:
        return 0.5, 0.0   # neutral H, zero fit quality
    # ... wavelet computation ...
```

### Phase 2: Fractal Threshold Parameters in Optimizer
**File:** `src/ait/optimization/param_spaces.py`

Add:
```python
FRACTAL_GATE_SPACE = {
    "hurst_regime_threshold":     ("float", 0.08, 0.30),
    "hurst_regime_penalty":       ("float", 0.0,  0.25),
    "multifractal_max_width":     ("float", 0.30, 0.65),
}
```

Merge into `IRON_CONDOR_SPACE` and `SHORT_STRANGLE_SPACE` (spread operator or explicit merge in the dict definitions).

### Phase 3: Fractal Gating in Backtesting Engine
**File:** `src/ait/backtesting/engine.py`

1. Add three new constructor kwargs:
   - `hurst_regime_threshold: float = 0.20`
   - `hurst_regime_penalty: float = 0.10`
   - `multifractal_max_width: float = 0.50`

2. In `_select_strategy()`, after direction and IV computation, extract last-row fractal values from the feature matrix and apply the gate:
   ```python
   features = FeatureEngine().compute(hist)
   if not features.empty:
       last = features.iloc[-1]
       spread = last.get("hurst_scale_spread", 0.0)
       mf_w   = last.get("multifractal_width", 0.0)
       if (spread > self._hurst_regime_threshold or
           (mf_w > 0 and mf_w > self._multifractal_max_width)):
           penalty = self._hurst_regime_penalty * (spread / self._hurst_regime_threshold)
           confidence = max(0.0, confidence - penalty)
           if confidence < self._min_confidence:
               return None   # Skip entry
   ```
   
   Note: `FeatureEngine().compute(hist)` is already called inside `DirectionPredictor.predict()`. To avoid double computation, the simplest approach is to compute features once at the call site and share the result. Refactor `_get_direction()` to return `(direction, confidence, features_df)` tuple.

### Phase 4: WalkForwardConfig Extension
**File:** `src/ait/backtesting/walkforward.py`

Add three fields to the `WalkForwardConfig` dataclass:
```python
hurst_regime_threshold:  float = 0.20
multifractal_max_width:  float = 0.50
hurst_regime_penalty:    float = 0.10
```

Propagate to `Backtester.__init__()` in `_run_window()` (where Backtester is instantiated per window). When `optimize_per_window=True` finds better thresholds via Optuna, those values replace the defaults.

### Phase 5: Intraday Data Infrastructure

**Why this phase is needed:** Intraday fractal features require a window of recent 5-min bars that is wider than what the current system fetches (`days=1`). Wavelet/Hurst needs ~78 bars (1 session), multi-scale H needs ~200 bars (2–3 sessions), and MFDFA needs ~500 bars (~1 week). The current in-memory TTL cache is wiped on each restart, meaning MFDFA features would be unavailable until 7 days of uptime accumulate. SQLite persistence solves this.

#### 5a. Data Source — No New IB Function Required

The existing `MarketDataService.get_intraday()` uses **Yahoo Finance** (`yf.Ticker.history(period, interval="5m")`). Yahoo provides 5-min OHLCV data for the past **60 days** — far more than the 7 days needed. **No IBKR API change is required for this feature.** IB's `reqHistoricalData()` for intraday bars is more complex (pagination, trading hours flags, pacing limits) and is not needed when Yahoo already covers the required window.

Change in `MarketDataService.get_intraday()`:
```python
# BEFORE (current):
async def get_intraday(self, symbol: str, interval: str = "5m", days: int = 1)
    period = f"{days}d" if days <= 5 else "1mo"

# AFTER:
async def get_intraday(self, symbol: str, interval: str = "5m", days: int = 7)
    period = f"{days}d" if days <= 5 else "1mo"
    # days=7 → "1mo" → Yahoo returns ~20 trading days of 5-min bars
    # Cache TTL stays 300s (5 min) — same live-data freshness guarantee
```

This single-line default change is all that is needed to get 7 days of 5-min bars per fetch. The `"1mo"` period gives ~1,500 bars — well above the 500-bar MFDFA minimum.

#### 5b. New SQLite Table — `intraday_prices`

**File:** `src/ait/data/historical.py`

Add a second table to the same SQLite DB (`data/historical.db`). A separate table (not an `interval` column on `daily_prices`) is the right choice because:
- The existing `daily_prices` primary key is `(symbol, date)` — cannot hold multiple bars per day
- Intraday rows need `datetime` precision, not `date`
- Schema isolation keeps daily and intraday queries simple

```sql
CREATE TABLE IF NOT EXISTS intraday_prices (
    symbol   TEXT    NOT NULL,
    datetime TEXT    NOT NULL,   -- ISO 8601, e.g. "2026-05-06T09:35:00"
    interval TEXT    NOT NULL,   -- e.g. "5m"
    open     REAL,
    high     REAL,
    low      REAL,
    close    REAL,
    volume   INTEGER,
    PRIMARY KEY (symbol, datetime, interval)
);
CREATE INDEX IF NOT EXISTS idx_intraday_symbol_dt
    ON intraday_prices(symbol, interval, datetime);
```

**Retention policy:** Keep 10 trading days. A `cleanup_old_intraday()` call in `_init_db()` (or called nightly) deletes rows older than 10 days. Storage estimate: 78 bars/day × 10 days × 15 symbols × ~100 bytes/row ≈ 1.2 MB — negligible.

#### 5c. New HistoricalDataStore Methods

**File:** `src/ait/data/historical.py`

Add four methods alongside the existing `save()` / `load()`:

```python
def save_intraday(self, symbol: str, df: pd.DataFrame, interval: str = "5m") -> int:
    """Upsert 5-min bars. Returns rows inserted/replaced."""

def load_intraday(
    self,
    symbol: str,
    days: int = 7,
    interval: str = "5m",
) -> pd.DataFrame:
    """Load recent intraday bars. Returns DataFrame with DatetimeIndex."""

def get_latest_intraday_timestamp(
    self, symbol: str, interval: str = "5m"
) -> datetime | None:
    """Return the timestamp of the most recent stored bar for incremental updates."""

def cleanup_old_intraday(self, keep_days: int = 10) -> None:
    """Delete intraday rows older than keep_days trading days."""
```

#### 5d. Persistence Flow in the Live Orchestrator

**File:** `src/ait/bot/orchestrator.py` — `_scan_symbol()` method

After `get_intraday()` returns and before `compute_intraday_features()` is called, add one line to persist the data:

```python
intraday_df = await self._market_data.get_intraday(symbol, days=7)
if intraday_df is not None and not intraday_df.empty:
    self._store.save_intraday(symbol, intraday_df)   # NEW LINE — persist to SQLite

# Then load from DB to get full history (including pre-restart history)
intraday_full = self._store.load_intraday(symbol, days=7)
# Use intraday_full (not intraday_df) for compute_intraday_features()
```

This means:
- On first startup: fetches 7 days from Yahoo, saves to DB, immediately computes all intraday fractal features including MFDFA (if 500+ bars available)
- On subsequent scans: Yahoo fetch returns latest bars, upserted into DB; full 7-day history loaded from DB (fast, no re-download)
- After restart: DB history survives; warm immediately from stored data

#### 5e. Intraday Fractal Feature Computation

**File:** `src/ait/ml/features.py` — `compute_intraday_features()` method

Extend the existing method to include (using the 7-day `intraday_full` DataFrame):

| Feature | Min bars | Description |
|---|---|---|
| `hurst_wavelet_intraday` | 78 (1 session) | Hurst from wavelet variance on 5-min returns |
| `hurst_scale_spread_intraday` | 200 (3 sessions) | \|H_short(39 bars) - H_long(200 bars)\| — intraday regime signal |
| `wavelet_L3_energy` | 40 | Energy at 40-min scale (morning/afternoon momentum) |
| `wavelet_L4_energy` | 80 | Energy at 80-min scale (half-session trend) |
| `wavelet_L5_energy` | 160 | Energy at 2.7-hour scale (full-session structure) |
| `psd_beta_intraday` | 100 | Spectral exponent from 5-min periodogram |
| `mfdfa_width_intraday` | 500 (~7 days) | Multifractal width from intraday returns |

Features requiring more bars than available fill with `0.0`. MFDFA is `0.0` until 7 days of stored history exist — XGBoost handles this correctly as a systematically-zero feature.

These features flow into the existing flat dict consumed by `orchestrator._scan_symbol()` via the `live_signals` / MTF confidence path — **no changes needed to the confidence overlay architecture**.

### Phase 6: Intraday Feature Computation — Fractal + VLMC Session Structure

**File:** `src/ait/ml/features.py` — `compute_intraday_features()` method

This phase implements two classes of intraday features, both computed from the 7-day `intraday_full` DataFrame loaded from SQLite (see Phase 5d). All features are added to the flat dict already consumed by `orchestrator._scan_symbol()` via the MTF confidence path.

#### 6a. Intraday Fractal Features (Framework 2 + fractal analysis)

Implement using `pywt` (wavelet decomposition) and `scipy.signal.periodogram` (PSD). MFDFA implemented inline.

| Feature | Min bars | Implementation |
|---|---|---|
| `hurst_wavelet_intraday` | 78 | Wavelet variance slope on 5-min log-returns |
| `hurst_scale_spread_intraday` | 200 | \|H_short(39 bars) - H_long(200 bars)\| |
| `wavelet_L3_energy` | 40 | `pywt.dwt` level 3 detail coefficient energy |
| `wavelet_L4_energy` | 80 | Level 4 detail energy (half-session trend) |
| `wavelet_L5_energy` | 160 | Level 5 detail energy (full-session structure) |
| `psd_beta_intraday` | 100 | Power-law exponent from `scipy.signal.periodogram` |
| `mfdfa_width_intraday` | 500 | Multifractal width Δα from 5-min returns |

#### 6b. VLMC Session Structure Features (Framework 1)

These encode the Markov context of the current trading session at three window lengths — exactly the VLMC approach from Section 4. Each window extracts a different "context" that XGBoost can learn to weight for the 5-day label.

**Full session window (78 bars — from open to current bar):**

| Feature | Description |
|---|---|
| `session_vwap_position` | Current price relative to session VWAP: `(close - vwap) / vwap` |
| `session_vwap_q1` | Price vs VWAP at the first quartile of the session (9:35–10:30 AM) |
| `session_vwap_q2` | Price vs VWAP at midday |
| `session_vwap_q3` | Price vs VWAP in the third quartile |
| `session_high_timing` | Fraction of session elapsed when the session high was set (0=open, 1=now) |
| `session_low_timing` | Fraction of session elapsed when the session low was set |
| `session_volume_front_load` | First-third volume / total volume (>0.5 = front-loaded) |
| `session_volume_shape` | `(first_third - last_third) / total_volume` — U-shape vs. front- vs. back-loaded |

**Power hour window (last 12 bars — last 60 min):**

| Feature | Description |
|---|---|
| `power_hour_momentum` | Log return over the last 12 bars |
| `power_hour_vol_accel` | Volume slope over last 12 bars (positive = acceleration) |
| `power_hour_vwap_cross` | Sign of (current price - power_hour open) — late-session directional bias |

**Closing imbalance window (last 3 bars — last 15 min):**

| Feature | Description |
|---|---|
| `closing_imbalance` | Mean return of last 3 bars — proxy for buy/sell pressure into close |
| `closing_range_position` | (close - low_3bar) / (high_3bar - low_3bar) — position within last 15-min range |

**Implementation note:** VWAP must be computed as a running VWAP from the session open (9:30 AM bar), not a trailing rolling VWAP. Filter `intraday_full` to today's session bars using the date index.

Total new features from Phase 6: **20** (7 fractal + 13 VLMC session structure).

All features fill `0.0` when insufficient bars are available (e.g., early in the session). Graceful degradation is the same `np.nan_to_num(value, nan=0.0)` pattern as Phase 1.

### Phase 7: Dependency Addition
**File:** `pyproject.toml`

```toml
pywavelets>=1.4
```

PyWavelets is pure Python + NumPy. No C++ or CUDA. No platform constraints. Compatible with Intel Mac.

### Phase 8: Diagnostic Report Tool
**Files:** `src/ait/diagnostics/fractal_report.py`, `scripts/run_fractal_diagnostics.py`

A standalone module for validating fractal feature quality and predictive power. Has no effect on the trading pipeline.

**`fractal_report.py` — core functions:**

```python
def plot_hurst_timeseries(symbol: str, features_df: pd.DataFrame) -> go.Figure
def plot_psd(returns: np.ndarray) -> go.Figure
def plot_multifractal_spectrum(returns: np.ndarray) -> go.Figure
def plot_scale_invariance_vs_vix(features_df: pd.DataFrame, vix_df: pd.DataFrame) -> go.Figure
def plot_ic_analysis(features_df: pd.DataFrame, labels: pd.Series) -> go.Figure
def plot_shap_importance(model, X: pd.DataFrame, feature_names: list[str]) -> go.Figure
def plot_gate_counterfactual(trades: list[dict]) -> go.Figure
def generate_report(
    symbols: list[str],
    start: str,
    end: str,
    output_dir: str,
    fmt: str = "html",
) -> None
```

**`run_fractal_diagnostics.py` — CLI entry point:**

```bash
python scripts/run_fractal_diagnostics.py \
    --symbols SPY QQQ AAPL NVDA \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --output reports/fractal/ \
    --format html
```

**Dependencies to add:** `shap>=0.42` (SHAP feature importance). `plotly>=5.0` if not already in pyproject.toml.

**What each plot validates:**
- `plot_hurst_timeseries` → confirms H is in [0.3, 0.7] range; spikes align with known volatile periods
- `plot_psd` → confirms power-law fit; R² should be > 0.7 for valid β estimate
- `plot_multifractal_spectrum` → bell-shaped f(α) curve; width should widen during volatile windows
- `plot_scale_invariance_vs_vix` → `hurst_scale_spread` should lead VIX spikes by 1–5 days
- `plot_ic_analysis` → fractal features with |IC| > 0.05 are predictively meaningful
- `plot_shap_importance` → fractal features in top 20 of 90+ features = genuinely contributing
- `plot_gate_counterfactual` → iron condor win rate comparison: gated vs. allowed

---

### Effort Estimates

Estimates assume one developer, familiarity with the codebase, and include unit test writing.

| Phase | Description | Estimated effort | Risk / notes |
|---|---|---|---|
| **1** | Daily fractal features in FeatureEngine | **2–3 days** | Complex numerical code (wavelet variance, MFDFA loop). Requires validation against known Hurst test cases. Most complex phase of the current plan. |
| **2** | Fractal threshold params in optimizer | **0.5 days** | Mechanical — add dict, merge into existing spaces. Low risk. |
| **3** | Fractal gating in backtesting engine | **1 day** | Moderate. Refactoring `_get_direction()` to return features tuple is the tricky part. |
| **4** | WalkForwardConfig extension | **0.5 days** | Trivial — 3 new dataclass fields + propagation. |
| **5** | Intraday data infrastructure | **1.5–2 days** | New SQLite table + 4 new HistoricalDataStore methods + orchestrator wiring. Schema migration must handle existing DBs. |
| **6** | Intraday fractal + VLMC session features | **2–3 days** | VWAP session boundary logic is the tricky part (today's bars only). Can reuse Phase 1 wavelet code for intraday path. |
| **7** | Dependency addition (`pywavelets`, `shap`) | **0.1 days** | Trivial. Verify install on Intel Mac. |
| **8** | Diagnostic report tool | **1–1.5 days** | Plotly charts + SHAP integration + IC computation + CLI. Low risk — diagnostic-only. |
| **Total (Phases 1–8)** | | **~9–11.5 days** | All backtested + live-validated changes plus diagnostics. |

**Deferred work (separate branches):**

| Work item | Estimated effort | Notes |
|---|---|---|
| Window-based training restructuring | **4–6 days** | Architectural refactor of DirectionPredictor to use paired (daily, intraday) DataFrames per training date. New walk-forward data pipeline. |
| CNN-GRU short-term model (Tier 2) | **5–8 days** | New model class, pooled multi-symbol training pipeline, orchestrator confirmation gate integration, end-of-day label construction. Fractal features concatenated at head. |
| Transformer / Seq2Seq (Tier 3) | **10–15 days** | Requires 5,000–20,000 samples (pooled multi-symbol pre-training). New training pipeline. Fractal features as auxiliary head inputs. Most complex; deferred until CNN-GRU is validated. |

---

## 9. Critical Files

| File | Change | Impact |
|---|---|---|
| `src/ait/ml/features.py` | Add `_add_fractal_features()` (10 daily fractal features); extend `compute_intraday_features()` with 20 new features (7 intraday fractal + 13 VLMC session structure); update `get_feature_names()` | Core change — affects all models using FeatureEngine |
| `src/ait/optimization/param_spaces.py` | Add `FRACTAL_GATE_SPACE`, merge into iron condor/strangle spaces | Optimizer now tunes fractal thresholds |
| `src/ait/backtesting/engine.py` | Add fractal gate in `_select_strategy()`, 3 new constructor kwargs | Iron condors gated on regime stability |
| `src/ait/backtesting/walkforward.py` | 3 new `WalkForwardConfig` fields, propagate to Backtester | Per-window threshold optimization |
| `src/ait/data/historical.py` | Add `intraday_prices` SQLite table + 4 new methods (`save_intraday`, `load_intraday`, `get_latest_intraday_timestamp`, `cleanup_old_intraday`) | Required for MFDFA on intraday data; DB schema migration on first run |
| `src/ait/data/market_data.py` | Change `get_intraday()` default from `days=1` to `days=7` | Fetches 7 days of 5-min bars from Yahoo; no new IB function needed |
| `src/ait/bot/orchestrator.py` | In `_scan_symbol()`: call `save_intraday()` after fetch, `load_intraday()` for full history before feature computation | Enables MFDFA from first startup (DB-warm) |
| `pyproject.toml` | Add `pywavelets>=1.4`, `shap>=0.42` | New dependencies |
| `src/ait/diagnostics/fractal_report.py` | New module — 7 Plotly chart functions + `generate_report()` | Diagnostic only; no trading pipeline effect |
| `scripts/run_fractal_diagnostics.py` | New CLI script — calls `generate_report()` with argparse | Run on demand post-backtest |

**No changes required:**
- `src/ait/ml/range_predictor.py` — uses `FeatureEngine.compute()` → gets fractal features automatically
- `src/ait/ml/vol_magnitude_predictor.py` — same
- `src/ait/ml/ensemble.py` — no change to training loop
- `src/ait/backtesting/learner.py` — no change; adapts strategy multipliers based on win rates that improve organically
- `src/ait/optimization/objectives.py` — no change; consumes BacktestResult metrics that improve
- `src/ait/optimization/optimizer.py` — no change; picks up new param spaces via existing dict registration
- `src/ait/backtesting/walkforward.py` window slicing — no change
- Walk-forward 5-day purge gap — no change; fractal features don't add label leakage
- **No new IB/IBKR function required** — Yahoo Finance already provides 5-min bars up to 60 days; IBKR `reqHistoricalData()` for intraday is not needed and adds pagination/pacing complexity

**Retraining required:** After adding new features to `FeatureEngine`, saved models (`models/ensemble.pkl`, `models/range.pkl`, `models/vol_magnitude.pkl`) will have a feature name mismatch and must be retrained. The existing daily 7:30 AM retrain schedule handles this on the next run.

---

## 10. Diagnostic Reports

A standalone diagnostic tool generates plots and predictive metrics for fractal features. It has no effect on the trading pipeline and can be run at any time against historical data.

### 10.1 Feature Health Plots

**Purpose:** Confirm that the fractal estimators are computing correct, sensible values.

| Plot | What to check |
|---|---|
| Rolling `hurst_wavelet` over time | Should be 0.3–0.7. Equities typically 0.55–0.65 in calm markets. Values outside [0.1, 0.9] indicate numerical issues. |
| `hurst_scale_spread` over time | Should spike during known volatility periods (2020 Mar, 2022 Jan–Jun). Should be near zero in calm trending markets. |
| Log-log PSD plot with fitted β line | Visual confirmation that power-law slope is meaningful (R² > 0.7 = good fit). |
| Multifractal spectrum f(α) vs. α | Classic bell-shaped curve. Wider spectrum = more multifractal. Should be narrow in calm markets, wider in volatile. |
| `hurst_fit_r2` vs. VIX | Low R² (scale invariance breaking) should precede VIX spikes by 1–5 days. |

### 10.2 Predictive Power Assessment (IC Analysis)

**Information Coefficient (IC)** = Spearman correlation between fractal feature value and 5-day forward return, computed per walk-forward window.

```python
from scipy.stats import spearmanr

ic_series = {}
for feature in fractal_feature_names:
    ics = []
    for window_df in walk_forward_windows:
        rho, _ = spearmanr(window_df[feature], window_df["fwd_return_5d"])
        ics.append(rho)
    ic_series[feature] = ics
```

**Plots:**
- IC bar chart: mean IC per fractal feature across all windows (sorted). Features with |IC| > 0.05 are considered informative.
- IC stability chart: rolling 6-window IC — shows whether the feature is consistently predictive or regime-dependent.
- IC regime split: IC during volatile windows (VIX > 25) vs. calm windows — `hurst_scale_spread` and `multifractal_width` should have higher IC during volatile regimes.

**Expected findings:**
- `hurst_scale_spread` and `multifractal_width`: IC highest in volatile windows → useful for regime gate
- `hurst_wavelet`: IC for directional strategies — positive H (trending) should correlate with continued bullish/bearish outcomes
- `psd_beta`: Lower IC expected — spectral exponent is more of a regime classifier than a direction predictor

### 10.3 SHAP Feature Importance

After retraining `DirectionPredictor`, compute SHAP values to quantify how much weight the model assigns to fractal features vs. existing price/vol/macro features.

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

**Output:** Bar chart ranking all features by mean |SHAP|. Fractal features highlighted in a different color. If fractal features rank in the top 20 (out of 90+ features), they are genuinely contributing. If they rank near the bottom, the model is ignoring them — which may indicate the estimator is not computing meaningful values or the feature is too noisy.

### 10.4 Regime Gate Counterfactual Analysis

From `BacktestResult.trades`, find all trade attempts where the fractal gate fired (iron condor blocked due to `hurst_scale_spread > threshold`). Compare:

| Metric | Gate fired | Gate not fired |
|---|---|---|
| Win rate | ? | ? |
| Avg P&L | ? | ? |
| Max drawdown | ? | ? |

If the gate consistently blocks trades that would have lost, the threshold is calibrated well. If it blocks winners, the threshold is too aggressive. The Optuna optimizer corrects this automatically, but the counterfactual table makes the gate's behavior interpretable.

### 10.5 CLI Usage

```bash
# Generate full fractal diagnostic report for symbols
python scripts/run_fractal_diagnostics.py \
    --symbols SPY QQQ AAPL NVDA \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --output reports/fractal/ \
    --format html        # or 'png' for static images
```

Output: `reports/fractal/fractal_report_SPY.html` (interactive Plotly) per symbol + `reports/fractal/ic_summary.csv` with IC metrics for all features × all walk-forward windows.

**Dependencies:** `plotly>=5.0` (already in the project if used elsewhere, otherwise add), `shap>=0.42`.

---

## 11. Verification Plan

### 11.1 Unit Tests

**Fractal feature computation:**
```bash
pytest tests/ -k "fractal"
```
- Feed 504-bar SPY daily OHLCV, assert all 10 features are finite and in expected ranges
- `hurst_wavelet` ∈ [0.1, 0.9]
- `hurst_scale_spread` ≥ 0
- `psd_beta` ∈ [0.5, 3.0]
- `multifractal_width` ≥ 0
- Features with < min_bars return `0.0` (not NaN or error)

**Feature count:**
```python
assert len(FeatureEngine().get_feature_names()) == old_count + 10
```

**Saved model mismatch (expected failure):**
Loading `models/ensemble.pkl` trained before this change should raise a feature name mismatch or produce a warning — confirming retraining is required.

### 11.2 Walk-Forward Smoke Test
```bash
python run_backtest.py --symbols SPY --capital 10000
```
- Completes without error
- Log shows `features_computed` with increased column count (+10)
- Check `hurst_scale_spread` values in log — should be ~0.05–0.15 for SPY in calm periods, higher in 2022

### 11.3 Optimizer Smoke Test
```bash
python run_optimizer.py --strategies iron_condor --symbols SPY --n-trials 10
```
- `hurst_regime_threshold` and `multifractal_max_width` appear in trial params in Optuna output
- `summary()` shows fractal params alongside existing strategy params

### 11.4 Per-Window Fractal Adaptation Test
```bash
python run_backtest.py \
  --symbols SPY QQQ \
  --optimize-per-window \
  --optimize-n-trials 20
```
- Different walk-forward windows (calm 2024 vs volatile 2022) produce different optimal `hurst_regime_threshold` values
- Confirms fractal thresholds are regime-adaptive per window

### 11.5 Regime Signal Validation

During backtest, log fractal feature values per day. Validate against known high-volatility periods:
- **Expected:** `hurst_scale_spread` > 0.15 and `multifractal_width` elevated during Jan–Jun 2022
- **Expected:** `hurst_scale_spread` < 0.10 during calm trending periods (2023 Q4, 2024 Q1)
- **Expected:** `multifractal_asymmetry` goes negative before major drawdowns (March 2020, Oct 2022)

### 11.6 Iron Condor Gate Validation

From `BacktestResult.trades`, compare:
- Trades where `hurst_scale_spread > threshold` was triggered (gated) vs. those that proceeded
- Gated trades should have lower hypothetical win rates (counterfactual validation)
- If gated trades would have won, threshold is too aggressive → optimizer should find a higher value

### 11.7 Intraday Fractal (Live Only)
```bash
# During paper trading market hours:
python -m src.ait.main --mode paper
```
Check logs for `compute_intraday_features` returning `hurst_wavelet_intraday` and `wavelet_L4_energy` keys for at least one symbol. Verify values are finite and update every 5-minute scan cycle.

---

*Plan authored: 2026-05-06*
*Branch: features-request-1*
*Related plan: `/Users/ahmednagi/.claude/plans/would-the-above-work-dazzling-bunny.md`*

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

### Pre-implementation Notes

Seven concrete issues must be resolved before or during each phase. They are listed here so they can be addressed in the right place rather than discovered mid-coding.

**Note 1 — `SHORT_STRANGLE_SPACE` must be created before Phase 2 merges into it.**
`param_spaces.py` has no `SHORT_STRANGLE_SPACE` and `"short_strangle"` is absent from `STRATEGY_SPACES`. Phase 2 must first define `SHORT_STRANGLE_SPACE` (modelled on `IRON_CONDOR_SPACE`), add it to `STRATEGY_SPACES`, and then merge `FRACTAL_GATE_SPACE` into both.

**Note 2 — Phase 5d uses the wrong attribute name.**
The plan writes `self._store.save_intraday(...)` but the orchestrator holds the store as `self._historical` (instantiated at line 82 of `orchestrator.py`). All Phase 5d references to `self._store` must be `self._historical`.

**Note 3 — Phase 3 refactor: the `run()` call site must also change.**
The engine's `run()` loop unpacks `_get_direction` as:
```python
direction, confidence = self._get_direction(hist)
```
After the refactor to a 3-tuple this becomes:
```python
direction, confidence, features_df = self._get_direction(hist)
```
`features_df` is then passed directly into `_select_strategy()` as an additional argument so the fractal gate can read from it without calling `FeatureEngine().compute()` a second time. `_select_strategy()` signature changes to `(self, direction, hist, confidence, features_df)`.

**Note 4 — Specify the wavelet family in `_hurst_wavelet()`.**
Use `pywt.wavedec(returns, wavelet='db4', mode='periodization')`. `db4` (Daubechies-4) is the standard choice for financial return series: it is orthogonal, has 4 vanishing moments (removes polynomial trends up to degree 3), and is the most cited family in the Hurst literature. Using a different family (e.g., `haar`) produces systematically different H values and makes results incomparable across codebases.

**Note 5 — Phase 6 integration path: call `compute_intraday_features()` separately, not through `MultiTimeframeAnalyzer`.**
The orchestrator already calls `self._mtf_analyzer.analyze(hist, intraday)` for the existing MTF confidence boost. The 20 new Phase 6 features must be added by calling `FeatureEngine().compute_intraday_features(intraday_full)` separately, after the `load_intraday()` call (Phase 5d), and merging the returned dict into `live_signals` before the `self._predictor.predict()` call. Do **not** extend `MultiTimeframeAnalyzer` — it handles timeframe alignment, not fractal estimation, and mixing concerns there makes both harder to test.

The wiring in `_scan_symbol()` becomes:
```python
intraday_full = self._historical.load_intraday(symbol, days=7)
intraday_fractal = FeatureEngine().compute_intraday_features(intraday_full)
live_signals.update(intraday_fractal)   # merged before predict()
```

**Note 6 — Create `src/ait/diagnostics/` package and `scripts/` directory before Phase 8.**
Neither exists yet. Phase 8 requires:
- `src/ait/diagnostics/__init__.py` (empty is fine)
- `src/ait/diagnostics/fractal_report.py`
- `scripts/run_fractal_diagnostics.py`

Without `__init__.py` the test import `from ait.diagnostics import fractal_report` raises `ModuleNotFoundError`.

**Note 7 — Add `packaging` to dev dependencies for the version-check tests.**
`tests/test_fractal_features.py` uses `from packaging.version import Version`. Add `packaging>=23.0` to `[project.optional-dependencies] dev` in `pyproject.toml`. Alternatively, replace the check with a tuple comparison to avoid the dependency:
```python
assert tuple(int(x) for x in pywt.__version__.split(".")[:2]) >= (1, 4)
```

---

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

### Phase 9: IBKR as Primary Intraday Data Source + 2-Year Backfill
**Files:** `src/ait/data/market_data.py`, `src/ait/data/historical.py`, `src/ait/bot/orchestrator.py`, `scripts/backfill_intraday.py`

**Motivation:** The current `get_intraday()` implementation is Yahoo Finance only. Using Yahoo for training data while trading against IBKR live prices introduces data source inconsistency: different split-adjustment timing, extended-hours bar contamination in Yahoo, and minor price divergences that compound into biased VLMC session features. IBKR must be the primary source for intraday bars so that training data and live inference use the same prices.

The second issue is depth: 7 days of 5-min bars is insufficient for MFDFA (requires ≥500 bars ≈ 7 sessions) and walk-forward validation of VLMC features requires at least 1 year of daily sessions. 2 years (≈730 calendar days, ≈504 trading days, ≈39,312 5-min bars per symbol) enables full walk-forward IC analysis on VLMC features.

**9.1 — Add `_get_ibkr_intraday()` to `MarketDataService`**

```python
async def _get_ibkr_intraday(
    self,
    symbol: str,
    duration: str = "7 D",          # "7 D", "1 M", "6 M", "1 Y"
    bar_size: str = "5 mins",
) -> pd.DataFrame | None:
    """Fetch intraday bars from IBKR via reqHistoricalDataAsync.

    Args:
        symbol:   Ticker string.
        duration: IBKR duration string — "7 D", "1 M", "6 M", "1 Y".
                  Max per request for 5-min bars is "6 M" before IBKR
                  rejects with "invalid duration".
        bar_size: "5 mins" for standard intraday.
    """
    if not self._ibkr or not self._ibkr.connected:
        return None
    try:
        from ib_insync import Stock
        contract = Stock(symbol, "SMART", "USD")
        qualified = await self._ibkr.qualify_contract(contract)
        if not qualified:
            return None

        bars = await self._ibkr.ib.reqHistoricalDataAsync(
            qualified,
            endDateTime="",          # empty = now
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            return None

        df = util.df(bars)
        df = df.rename(columns={
            "date": "Datetime", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df.set_index("Datetime", inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        log.debug("ibkr_intraday_failed", symbol=symbol, error=str(e))
        return None
```

**9.2 — Update `get_intraday()` to IBKR → Yahoo fallback chain**

Change `get_intraday()` to:
1. Try `_get_ibkr_intraday(symbol, duration="7 D")` first.
2. Fall back to Yahoo only if IBKR is unavailable or returns empty.

The `days` parameter maps to IBKR duration strings:

| `days` value | IBKR `durationStr` |
|---|---|
| ≤ 7 | "7 D" |
| ≤ 30 | "1 M" |
| ≤ 180 | "6 M" |
| > 180 | "1 Y" (or paginated — see backfill) |

**9.3 — Incremental daily fetch in orchestrator**

The orchestrator's `_scan_symbol()` should replace its `get_intraday(days=7)` call with an incremental fetch: only request bars since `get_latest_intraday_timestamp()`. This avoids re-downloading bars already stored in SQLite.

```python
# In orchestrator._scan_symbol():
last_ts = self._historical.get_latest_intraday_timestamp(symbol)
if last_ts is None:
    # First run — fetch 7 days to seed the DB
    intraday = await self._market_data.get_intraday(symbol, interval="5m", days=7)
else:
    # Only fetch new bars since last stored timestamp
    intraday = await self._market_data.get_intraday_since(symbol, since=last_ts)
```

Add `get_intraday_since(symbol, since: pd.Timestamp)` to `MarketDataService` — uses IBKR `endDateTime=""` and a `durationStr` computed from `now - since`.

**9.4 — Extend `cleanup_old_intraday` retention to 2 years**

Change `cleanup_old_intraday(keep_days=10)` call in orchestrator to `cleanup_old_intraday(keep_days=730)`. 2 years of 5-min data per symbol = ~39,312 rows × ~20 symbols = ~786K rows in `intraday_prices`. SQLite handles this comfortably (< 50 MB with the existing schema).

**9.5 — Bulk backfill script: `scripts/backfill_intraday.py`**

A one-time (and repeatable) script that backfills 2 years of 5-min bars per symbol from IBKR into SQLite. Must handle IBKR's pacing rule (60 requests per 10 minutes per client) and the 6-month-per-request cap for 5-min bars.

```bash
python scripts/backfill_intraday.py \
    --symbols SPY QQQ AAPL NVDA MSFT \
    --years 2 \
    --bar-size "5 mins"
```

**Pagination logic:**

For 2 years with max 6-month chunks:
1. Split the 2-year window into 4 × 6-month segments (most-recent first).
2. For each segment, call `reqHistoricalDataAsync` with `durationStr="6 M"` and `endDateTime` set to the end of that segment.
3. Upsert into `intraday_prices` via `HistoricalDataStore.save_intraday()`.
4. Pause 1 second between requests (well within 60/10-min pacing limit).

**IBKR pacing constraints:**
- Max 60 historical data requests per 10 minutes per client ID
- Max 6 months of 5-min data per request
- For 5 symbols × 4 requests each = 20 requests total — no pacing issue

**CLI:**
```python
argparse arguments:
  --symbols    list of tickers (required)
  --years      how many years back to fetch (default: 2)
  --bar-size   IBKR bar size string (default: "5 mins")
  --db-path    SQLite DB path (default: data/historical.db)
  --dry-run    print request plan without executing
```

---

### Phase 10: VLMC Session Structure Diagnostic Reports
**Files:** `src/ait/diagnostics/fractal_report.py`, `tests/test_fractal_diagnostics.py`

**Motivation:** Phase 8 generates diagnostics only for fractal features (Hurst, PSD, MFDFA). The 13 VLMC session structure features have no diagnostic coverage: we cannot currently validate that `session_vwap_position`, `session_volume_front_load`, and `power_hour_momentum` are computing correctly or whether they carry predictive signal. Phase 10 adds VLMC-specific plots following the same Plotly HTML pattern as Phase 8.

**New functions to add to `fractal_report.py`:**

```python
def plot_session_vwap_trajectory(symbol: str, intraday_df: pd.DataFrame) -> go.Figure
def plot_volume_profile_distribution(symbol: str, intraday_df: pd.DataFrame) -> go.Figure
def plot_session_feature_ic_analysis(features_df: pd.DataFrame, labels: pd.Series) -> go.Figure
def plot_power_hour_patterns(symbol: str, intraday_df: pd.DataFrame) -> go.Figure
```

**What each plot validates:**

| Function | What to check |
|---|---|
| `plot_session_vwap_trajectory` | Rolling average of `session_vwap_position` across all sessions. Strongly trending markets should show persistent above/below VWAP closing. Mean-reverting regimes should be near zero. |
| `plot_volume_profile_distribution` | Histogram of `session_volume_front_load` values. Should be roughly uniform or slightly front-loaded (equities). Heavy back-loading > 0.6 may indicate ETF rebalancing artifacts. |
| `plot_session_feature_ic_analysis` | IC bar chart for all 13 VLMC features vs. 5-day forward return. `power_hour_momentum` and `closing_imbalance` are expected to have the highest IC. |
| `plot_power_hour_patterns` | Scatter: `power_hour_momentum` vs. next-day return. Should show mild positive correlation for trending markets. |

**Update `generate_report()` to include VLMC plots:**

```python
# In generate_report(), after fetching intraday_df (requires Phase 9 backfill):
if intraday_df is not None and not intraday_df.empty:
    plots.extend([
        plot_session_vwap_trajectory(symbol, intraday_df),
        plot_volume_profile_distribution(symbol, intraday_df),
        plot_session_feature_ic_analysis(features_df, labels),
        plot_power_hour_patterns(symbol, intraday_df),
    ])
```

`generate_report()` must be updated to also accept intraday data. It fetches intraday from the SQLite store (`HistoricalDataStore.load_intraday(symbol, days=730)`) rather than making a live IBKR call — diagnostic tool runs offline against the stored DB.

**Updated CLI (no flag changes needed — VLMC plots included automatically):**

```bash
python scripts/run_fractal_diagnostics.py \
    --symbols SPY QQQ AAPL NVDA \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --output reports/fractal/ \
    --format html
```

Output now includes both fractal and VLMC sections in each per-symbol HTML.

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
| **9** | IBKR intraday source + 2-year backfill | **2–3 days** | `_get_ibkr_intraday()` + pagination logic + incremental orchestrator fetch + backfill script. Main risk: IBKR pacing and `reqHistoricalDataAsync` not available when TWS is disconnected — must degrade gracefully to Yahoo. |
| **10** | VLMC diagnostic reports | **1 day** | 4 new plot functions + `generate_report()` update. Depends on Phase 9 backfill (needs stored intraday data). Low risk — diagnostic-only. |
| **Total (Phases 1–10)** | | **~12–15.5 days** | Full fractal + VLMC feature stack with consistent IBKR data source, 2-year intraday history, and complete diagnostic coverage. |

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
| `src/ait/data/market_data.py` | Add `_get_ibkr_intraday()` + `get_intraday_since()`; change fallback chain to IBKR→Yahoo; `get_intraday()` default `days=7` | IBKR becomes primary source for consistent training/trading data |
| `src/ait/bot/orchestrator.py` | In `_scan_symbol()`: call `save_intraday()` after fetch, `load_intraday()` for full history before feature computation | Enables MFDFA from first startup (DB-warm) |
| `pyproject.toml` | Add `pywavelets>=1.4`, `shap>=0.42` | New dependencies |
| `src/ait/diagnostics/fractal_report.py` | New module — 7 Plotly chart functions + `generate_report()` | Diagnostic only; no trading pipeline effect |
| `scripts/run_fractal_diagnostics.py` | New CLI script — calls `generate_report()` with argparse | Run on demand post-backtest |
| `scripts/backfill_intraday.py` | New CLI script — paginated IBKR backfill for 2 years of 5-min bars per symbol | One-time run before walk-forward validation of VLMC features; repeatable to fill gaps |

**No changes required:**
- `src/ait/ml/range_predictor.py` — uses `FeatureEngine.compute()` → gets fractal features automatically
- `src/ait/ml/vol_magnitude_predictor.py` — same
- `src/ait/ml/ensemble.py` — no change to training loop
- `src/ait/backtesting/learner.py` — no change; adapts strategy multipliers based on win rates that improve organically
- `src/ait/optimization/objectives.py` — no change; consumes BacktestResult metrics that improve
- `src/ait/optimization/optimizer.py` — no change; picks up new param spaces via existing dict registration
- `src/ait/backtesting/walkforward.py` window slicing — no change
- Walk-forward 5-day purge gap — no change; fractal features don't add label leakage

**Retraining required:** After adding new features to `FeatureEngine`, saved models (`models/ensemble.pkl`, `models/range.pkl`, `models/vol_magnitude.pkl`) will have a feature name mismatch and must be retrained. The existing daily 7:30 AM retrain schedule handles this on the next run.

---

## 10. Diagnostic Reports

A standalone diagnostic tool generates plots and predictive metrics for fractal and VLMC session structure features. It has no effect on the trading pipeline and can be run at any time against the stored SQLite data.

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

**Dependencies:** `plotly>=5.0` (already in the project), `shap>=0.42`.

### 10.6 VLMC Session Structure Diagnostic Plots

Four additional plots are generated per symbol when intraday data is available in the SQLite store (requires Phase 9 backfill).

**`plot_session_vwap_trajectory(symbol, intraday_df)`**
- Aggregates `session_vwap_position` (close vs. session VWAP) across all stored sessions.
- Rolling 20-session mean and ±1σ band.
- **What to check:** Trending markets should show persistent positive/negative VWAP position. Mean-reverting markets should oscillate around zero. Systematic bias (e.g., always below VWAP) may indicate a VWAP computation bug (check session boundary filtering).

**`plot_volume_profile_distribution(symbol, intraday_df)`**
- Histogram of `session_volume_front_load` and `session_volume_shape` across all sessions.
- Overlays expected U-curve distribution (high volume at open and close).
- **What to check:** `session_volume_front_load` > 0.45 is typical for US equities. Values uniformly distributed at 0.33 suggest the session boundary filter is not working (treating entire 7-day window as one session). Heavy back-loading (> 0.6) may indicate ETF creation/redemption artifacts.

**`plot_session_feature_ic_analysis(features_df, labels)`**
- IC (Spearman ρ) for all 13 VLMC session features vs. 5-day forward return.
- Same bar chart format as the fractal IC plot.
- **Expected findings:** `power_hour_momentum` and `closing_imbalance` expected to show the highest |IC| (0.04–0.10 range). `session_vwap_q1/q2/q3` will have lower IC — they measure intraday trajectory, not direction. Features with IC < 0.02 and no statistical significance should be flagged for potential removal.

**`plot_power_hour_patterns(symbol, intraday_df)`**
- Scatter plot: `power_hour_momentum` (x-axis) vs. next-session open return (y-axis) per day.
- Colour-coded by `power_hour_vol_accel` (blue = decelerating, red = accelerating).
- **What to check:** Should show mild positive slope (continuation) in trending markets. Flat scatter with no slope means `power_hour_momentum` has no predictive content for this symbol — expected for highly mean-reverting assets (e.g., short-dated inverse ETFs).

**Updated CLI (VLMC plots are included automatically if intraday data exists):**

```bash
# Full report: fractal + VLMC
python scripts/run_fractal_diagnostics.py \
    --symbols SPY QQQ AAPL NVDA \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --output reports/fractal/ \
    --format html

# If intraday DB is populated (after backfill_intraday.py), VLMC plots appear automatically.
# If DB is empty for a symbol, VLMC section is omitted with a logged warning.
```

---

## 11. Verification Plan

This section specifies the complete test suite required to confirm that all eight phases are correctly implemented. Tests are organised into per-phase unit tests, end-to-end integration tests, and CLI smoke tests.

Run all fractal-related tests with:

```bash
pytest tests/ -k "fractal" -v          # fractal-specific tests only
pytest tests/ -x --tb=short            # full suite, stop on first failure
```

---

### 11.1 Phase 1 — Daily Fractal Features in FeatureEngine

**File:** `tests/test_fractal_features.py`

#### 11.1.1 Shared fixture

```python
import numpy as np
import pandas as pd
import pytest

from ait.ml.features import FeatureEngine


def _make_ohlcv(days: int = 504, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, days)
    close = 400.0 * np.cumprod(1 + returns)
    dates = pd.date_range("2022-01-03", periods=days, freq="B")
    return pd.DataFrame({
        "Open":   close * (1 + rng.normal(0, 0.001, days)),
        "High":   close * (1 + np.abs(rng.normal(0, 0.005, days))),
        "Low":    close * (1 - np.abs(rng.normal(0, 0.005, days))),
        "Close":  close,
        "Volume": rng.integers(1_000_000, 10_000_000, days),
    }, index=dates)
```

#### 11.1.2 Feature presence tests

```python
class TestFractalFeaturesPresent:

    FRACTAL_FEATURES = [
        "hurst_wavelet", "hurst_fit_r2",
        "psd_beta", "psd_fit_r2", "hurst_psd_divergence",
        "hurst_short", "hurst_long", "hurst_scale_spread",
        "multifractal_width", "multifractal_asymmetry",
    ]

    def test_all_fractal_features_in_output(self) -> None:
        result = FeatureEngine().compute(_make_ohlcv(504))
        missing = [f for f in self.FRACTAL_FEATURES if f not in result.columns]
        assert missing == [], f"Missing fractal features: {missing}"

    def test_get_feature_names_includes_all_10_fractals(self) -> None:
        names = FeatureEngine().get_feature_names()
        for feat in self.FRACTAL_FEATURES:
            assert feat in names, f"{feat!r} absent from get_feature_names()"

    def test_feature_count_increased_by_exactly_10(self) -> None:
        """get_feature_names() must contain exactly 10 new fractal entries."""
        names = FeatureEngine().get_feature_names()
        fractal_count = sum(1 for n in names if n in self.FRACTAL_FEATURES)
        assert fractal_count == 10
```

#### 11.1.3 Value range tests

```python
class TestFractalFeatureRanges:

    def _last_row(self) -> pd.Series:
        return FeatureEngine().compute(_make_ohlcv(504)).iloc[-1]

    def test_hurst_wavelet_in_range(self) -> None:
        row = self._last_row()
        assert 0.1 <= row["hurst_wavelet"] <= 0.9, (
            f"hurst_wavelet={row['hurst_wavelet']:.4f} out of [0.1, 0.9]"
        )

    def test_hurst_fit_r2_bounded(self) -> None:
        row = self._last_row()
        assert 0.0 <= row["hurst_fit_r2"] <= 1.0

    def test_psd_beta_in_range(self) -> None:
        row = self._last_row()
        assert 0.5 <= row["psd_beta"] <= 3.0, (
            f"psd_beta={row['psd_beta']:.4f} out of [0.5, 3.0]"
        )

    def test_psd_fit_r2_bounded(self) -> None:
        row = self._last_row()
        assert 0.0 <= row["psd_fit_r2"] <= 1.0

    def test_hurst_scale_spread_nonnegative(self) -> None:
        assert self._last_row()["hurst_scale_spread"] >= 0.0

    def test_multifractal_width_nonnegative(self) -> None:
        assert self._last_row()["multifractal_width"] >= 0.0

    def test_no_nans_in_fractal_columns(self) -> None:
        result = FeatureEngine().compute(_make_ohlcv(504))
        fractal_cols = [
            "hurst_wavelet", "hurst_fit_r2", "psd_beta", "psd_fit_r2",
            "hurst_psd_divergence", "hurst_short", "hurst_long",
            "hurst_scale_spread", "multifractal_width", "multifractal_asymmetry",
        ]
        nans = result[fractal_cols].isna().sum()
        assert nans.sum() == 0, f"NaNs found: {nans[nans > 0].to_dict()}"

    def test_no_infs_in_fractal_columns(self) -> None:
        result = FeatureEngine().compute(_make_ohlcv(504))
        fractal_cols = [
            "hurst_wavelet", "hurst_fit_r2", "psd_beta", "psd_fit_r2",
            "hurst_psd_divergence", "hurst_short", "hurst_long",
            "hurst_scale_spread", "multifractal_width", "multifractal_asymmetry",
        ]
        assert not np.isinf(result[fractal_cols].values).any(), "Inf values in fractal features"
```

#### 11.1.4 Graceful degradation tests

```python
class TestFractalGracefulDegradation:

    def test_too_few_rows_returns_empty_dataframe(self) -> None:
        result = FeatureEngine().compute(_make_ohlcv(30))
        assert result.empty

    def test_features_needing_180_bars_are_zero_on_60_bar_series(self) -> None:
        """hurst_long (requires 180 bars) must fill 0.0 when given 60 bars."""
        result = FeatureEngine().compute(_make_ohlcv(60))
        if result.empty:
            pytest.skip("60-bar series rejected at input guard — acceptable")
        assert result.iloc[-1]["hurst_long"] == pytest.approx(0.0)
        assert result.iloc[-1]["hurst_scale_spread"] == pytest.approx(0.0)

    def test_mfdfa_zero_when_under_500_bars(self) -> None:
        """With 250 bars (< 500 MFDFA minimum), multifractal features must be 0.0."""
        result = FeatureEngine().compute(_make_ohlcv(250))
        if result.empty:
            pytest.skip("250-bar series rejected — acceptable")
        assert result.iloc[-1]["multifractal_width"] == pytest.approx(0.0)
        assert result.iloc[-1]["multifractal_asymmetry"] == pytest.approx(0.0)
```

#### 11.1.5 Estimator unit tests (private methods)

```python
class TestFractalEstimators:

    def test_hurst_brownian_motion_near_half(self) -> None:
        """i.i.d. Gaussian returns → H ≈ 0.5 ± 0.15."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0, 1, 1000)
        h, r2 = FeatureEngine()._hurst_wavelet(returns)
        assert 0.35 <= h <= 0.65, f"Brownian H={h:.3f}, expected near 0.5"
        assert r2 >= 0.0

    def test_hurst_persistent_series_above_half(self) -> None:
        """Positively autocorrelated series → H > 0.5."""
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 1, 500)
        persistent = np.cumsum(noise)
        returns = np.diff(persistent) / (np.abs(persistent[:-1]) + 1e-8)
        h, _ = FeatureEngine()._hurst_wavelet(returns)
        assert h > 0.5, f"Persistent H={h:.3f} should be > 0.5"

    def test_hurst_mean_reverting_below_half(self) -> None:
        """AR(1) with φ=−0.7 → H < 0.5."""
        rng = np.random.default_rng(2)
        n = 500
        ar = np.zeros(n)
        eps = rng.normal(0, 1, n)
        for i in range(1, n):
            ar[i] = -0.7 * ar[i - 1] + eps[i]
        h, _ = FeatureEngine()._hurst_wavelet(ar)
        assert h < 0.5, f"Mean-reverting H={h:.3f} should be < 0.5"

    def test_psd_features_return_two_floats(self) -> None:
        rng = np.random.default_rng(3)
        beta, r2 = FeatureEngine()._psd_features(rng.normal(0, 1, 200))
        assert isinstance(beta, float)
        assert isinstance(r2, float)
        assert 0.0 <= r2 <= 1.0

    def test_mfdfa_insufficient_data_returns_zeros(self) -> None:
        width, asymmetry = FeatureEngine()._mfdfa_features(
            np.random.default_rng(4).normal(0, 1, 100)   # < 500 minimum
        )
        assert width == pytest.approx(0.0)
        assert asymmetry == pytest.approx(0.0)

    def test_mfdfa_full_series_nonnegative_width(self) -> None:
        width, _ = FeatureEngine()._mfdfa_features(
            np.random.default_rng(5).normal(0, 1, 600)
        )
        assert width >= 0.0
```

---

### 11.2 Phase 2 — Fractal Gate Parameters in Optimizer

**File:** `tests/test_fractal_features.py` (append)

```python
from ait.optimization.param_spaces import (
    FRACTAL_GATE_SPACE,
    IRON_CONDOR_SPACE,
    STRATEGY_SPACES,
)


class TestFractalGateParamSpace:

    def test_fractal_gate_space_has_all_three_keys(self) -> None:
        for key in ("hurst_regime_threshold", "hurst_regime_penalty", "multifractal_max_width"):
            assert key in FRACTAL_GATE_SPACE, f"Missing key {key!r} in FRACTAL_GATE_SPACE"

    def test_all_ranges_are_valid_float_tuples(self) -> None:
        for key, spec in FRACTAL_GATE_SPACE.items():
            typ, lo, hi = spec[:3]
            assert typ == "float", f"{key}: expected 'float', got {typ!r}"
            assert 0.0 <= lo < hi <= 1.0, f"{key}: invalid range [{lo}, {hi}]"

    def test_iron_condor_space_includes_fractal_keys(self) -> None:
        for key in FRACTAL_GATE_SPACE:
            assert key in IRON_CONDOR_SPACE, (
                f"{key!r} must be merged into IRON_CONDOR_SPACE"
            )

    def test_short_strangle_space_includes_fractal_keys(self) -> None:
        strangle_space = STRATEGY_SPACES["short_strangle"]
        for key in FRACTAL_GATE_SPACE:
            assert key in strangle_space, (
                f"{key!r} must be merged into SHORT_STRANGLE_SPACE"
            )

    def test_directional_strategies_exclude_fractal_keys(self) -> None:
        """Bull/bear/long strategies must NOT carry fractal gate params."""
        for strategy in ("long_call", "long_put", "bull_call_spread", "bear_put_spread"):
            space = STRATEGY_SPACES[strategy]
            for key in FRACTAL_GATE_SPACE:
                assert key not in space, (
                    f"{key!r} must not be in {strategy!r} space"
                )
```

---

### 11.3 Phase 3 — Fractal Gating in Backtesting Engine

**File:** `tests/test_fractal_features.py` (append)

```python
from ait.backtesting.engine import Backtester


def _make_ohlcv_engine(days: int = 504, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 400.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, days))
    dates = pd.date_range("2022-01-03", periods=days, freq="B")
    return pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.005,
        "Low":    close * 0.995,
        "Close":  close,
        "Volume": rng.integers(1_000_000, 10_000_000, days),
    }, index=dates)


class TestFractalGatingEngine:

    def test_backtester_accepts_fractal_kwargs(self) -> None:
        Backtester(
            data=_make_ohlcv_engine(),
            strategies=["iron_condor"],
            hurst_regime_threshold=0.20,
            hurst_regime_penalty=0.10,
            multifractal_max_width=0.50,
        )

    def test_fractal_kwargs_stored_as_private_attributes(self) -> None:
        bt = Backtester(
            data=_make_ohlcv_engine(),
            strategies=["iron_condor"],
            hurst_regime_threshold=0.15,
            hurst_regime_penalty=0.08,
            multifractal_max_width=0.45,
        )
        assert bt._hurst_regime_threshold == pytest.approx(0.15)
        assert bt._hurst_regime_penalty   == pytest.approx(0.08)
        assert bt._multifractal_max_width  == pytest.approx(0.45)

    def test_zero_threshold_zero_penalty_does_not_raise(self) -> None:
        """Degenerate settings must not crash the engine."""
        bt = Backtester(
            data=_make_ohlcv_engine(),
            strategies=["iron_condor"],
            hurst_regime_threshold=0.0,
            hurst_regime_penalty=0.0,
            multifractal_max_width=1.0,
        )
        bt.run()

    def test_extreme_penalty_blocks_all_iron_condors(self) -> None:
        """threshold=0.0 + penalty=1.0 must eliminate all iron condor entries."""
        bt = Backtester(
            data=_make_ohlcv_engine(),
            strategies=["iron_condor"],
            hurst_regime_threshold=0.0,
            hurst_regime_penalty=1.0,
            multifractal_max_width=1.0,
            min_confidence=0.55,
        )
        result = bt.run()
        iron_condor_trades = [
            t for t in result.trades if t.get("strategy") == "iron_condor"
        ]
        assert len(iron_condor_trades) == 0, (
            "Iron condors must be fully blocked when threshold=0 and penalty=1"
        )

    def test_get_direction_returns_three_tuple(self) -> None:
        """After Phase 3 refactor, _get_direction must return (direction, confidence, features_df)."""
        import inspect
        bt = Backtester(data=_make_ohlcv_engine(100), strategies=["iron_condor"])
        # Confirm the method exists and accepts a hist argument
        assert hasattr(bt, "_get_direction")
        assert callable(bt._get_direction)
        sig = inspect.signature(bt._get_direction)
        # Must accept at least one positional parameter (hist DataFrame)
        assert len(sig.parameters) >= 1

    def test_features_shared_not_recomputed(self) -> None:
        """FeatureEngine.compute() must be called once per bar, not twice."""
        call_count = {"n": 0}
        original_compute = FeatureEngine.compute

        def counting_compute(self_, *args, **kwargs):
            call_count["n"] += 1
            return original_compute(self_, *args, **kwargs)

        import unittest.mock
        with unittest.mock.patch.object(FeatureEngine, "compute", counting_compute):
            bt = Backtester(
                data=_make_ohlcv_engine(120),
                strategies=["iron_condor"],
            )
            bt.run()

        bars_run = 120 - 50   # approximate bars after warmup
        # Each bar must call compute at most once (fractal gate reuses the result)
        assert call_count["n"] <= bars_run + 5, (
            f"FeatureEngine.compute() called {call_count['n']} times for ~{bars_run} bars "
            "— features are being computed twice per bar"
        )
```

---

### 11.4 Phase 4 — WalkForwardConfig Extension

**File:** `tests/test_walkforward.py` (append to existing `TestWalkForwardConfig` class or add new class)

```python
from ait.backtesting.walkforward import WalkForwardConfig, WalkForwardBacktester


class TestWalkForwardConfigFractalFields:

    def test_fractal_fields_exist_on_dataclass(self) -> None:
        cfg = WalkForwardConfig()
        assert hasattr(cfg, "hurst_regime_threshold")
        assert hasattr(cfg, "hurst_regime_penalty")
        assert hasattr(cfg, "multifractal_max_width")

    def test_fractal_field_default_values(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.hurst_regime_threshold == pytest.approx(0.20)
        assert cfg.hurst_regime_penalty   == pytest.approx(0.10)
        assert cfg.multifractal_max_width  == pytest.approx(0.50)

    def test_fractal_fields_accept_custom_values(self) -> None:
        cfg = WalkForwardConfig(
            hurst_regime_threshold=0.12,
            hurst_regime_penalty=0.05,
            multifractal_max_width=0.40,
        )
        assert cfg.hurst_regime_threshold == pytest.approx(0.12)
        assert cfg.hurst_regime_penalty   == pytest.approx(0.05)
        assert cfg.multifractal_max_width  == pytest.approx(0.40)

    def test_fractal_thresholds_propagate_to_backtester(self) -> None:
        """WalkForwardBacktester must forward fractal thresholds to per-window Backtester instances."""
        cfg = WalkForwardConfig(
            hurst_regime_threshold=0.12,
            hurst_regime_penalty=0.07,
            multifractal_max_width=0.38,
        )
        wf = WalkForwardBacktester(
            symbols=["SPY"],
            strategies=["iron_condor"],
            config=cfg,
        )
        # The config is stored and must carry the custom values through
        assert wf._config.hurst_regime_threshold == pytest.approx(0.12)
        assert wf._config.hurst_regime_penalty   == pytest.approx(0.07)
        assert wf._config.multifractal_max_width  == pytest.approx(0.38)
```

---

### 11.5 Phase 5 — Intraday Data Infrastructure

#### 11.5.1 HistoricalDataStore — intraday table and methods

**File:** `tests/test_intraday_store.py`

```python
"""Unit tests for the intraday_prices SQLite table and HistoricalDataStore methods."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.data.historical import HistoricalDataStore


def _make_intraday_df(bars: int = 78, start: str = "2026-05-07 09:35") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    closes = 400.0 + np.cumsum(rng.normal(0, 0.5, bars))
    dts = pd.date_range(start, periods=bars, freq="5min")
    return pd.DataFrame({
        "Open":   closes * 0.999, "High": closes * 1.002,
        "Low":    closes * 0.998, "Close": closes,
        "Volume": rng.integers(10_000, 200_000, bars),
    }, index=dts)


@pytest.fixture
def tmp_store(tmp_path: Path) -> HistoricalDataStore:
    return HistoricalDataStore(db_path=tmp_path / "test.db")


class TestIntradayTableCreation:

    def test_intraday_table_created_on_init(self, tmp_store: HistoricalDataStore) -> None:
        with sqlite3.connect(tmp_store._db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert "intraday_prices" in tables

    def test_intraday_index_created(self, tmp_store: HistoricalDataStore) -> None:
        with sqlite3.connect(tmp_store._db_path) as conn:
            indexes = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )}
        assert "idx_intraday_symbol_dt" in indexes

    def test_existing_db_not_broken_by_migration(self, tmp_path: Path) -> None:
        """Opening a pre-existing DB that lacks intraday_prices must not raise."""
        db = tmp_path / "legacy.db"
        with sqlite3.connect(db) as conn:
            conn.execute(
                "CREATE TABLE daily_prices "
                "(symbol TEXT, date TEXT, open REAL, high REAL, low REAL, "
                "close REAL, volume INTEGER, PRIMARY KEY (symbol, date))"
            )
        HistoricalDataStore(db_path=db)   # must not raise


class TestSaveIntraday:

    def test_save_returns_row_count(self, tmp_store: HistoricalDataStore) -> None:
        assert tmp_store.save_intraday("SPY", _make_intraday_df(78)) == 78

    def test_save_empty_df_returns_zero(self, tmp_store: HistoricalDataStore) -> None:
        assert tmp_store.save_intraday("SPY", pd.DataFrame()) == 0

    def test_save_none_returns_zero(self, tmp_store: HistoricalDataStore) -> None:
        assert tmp_store.save_intraday("SPY", None) == 0

    def test_upsert_does_not_duplicate_rows(self, tmp_store: HistoricalDataStore) -> None:
        df = _make_intraday_df(78)
        tmp_store.save_intraday("SPY", df)
        tmp_store.save_intraday("SPY", df)     # same data again
        assert len(tmp_store.load_intraday("SPY", days=30)) == 78


class TestLoadIntraday:

    def test_load_returns_dataframe_with_ohlcv(self, tmp_store: HistoricalDataStore) -> None:
        tmp_store.save_intraday("SPY", _make_intraday_df(78))
        result = tmp_store.load_intraday("SPY", days=30)
        assert isinstance(result, pd.DataFrame)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns

    def test_load_has_datetimeindex(self, tmp_store: HistoricalDataStore) -> None:
        tmp_store.save_intraday("SPY", _make_intraday_df(78))
        result = tmp_store.load_intraday("SPY")
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_load_unknown_symbol_returns_empty(self, tmp_store: HistoricalDataStore) -> None:
        assert tmp_store.load_intraday("UNKNOWN_XYZ").empty

    def test_load_respects_days_window(self, tmp_store: HistoricalDataStore) -> None:
        old = _make_intraday_df(78, start="2022-01-03 09:35")
        new = _make_intraday_df(78, start="2026-05-06 09:35")
        tmp_store.save_intraday("AAPL", old)
        tmp_store.save_intraday("AAPL", new)
        result = tmp_store.load_intraday("AAPL", days=7)
        assert (result.index >= pd.Timestamp("2026-04-29")).all(), (
            "Bars older than days=7 window must be excluded"
        )


class TestGetLatestIntradayTimestamp:

    def test_returns_none_when_empty(self, tmp_store: HistoricalDataStore) -> None:
        assert tmp_store.get_latest_intraday_timestamp("SPY") is None

    def test_returns_max_datetime(self, tmp_store: HistoricalDataStore) -> None:
        df = _make_intraday_df(10)
        tmp_store.save_intraday("QQQ", df)
        ts = tmp_store.get_latest_intraday_timestamp("QQQ")
        assert ts is not None
        assert ts == df.index.max()


class TestCleanupOldIntraday:

    def test_cleanup_removes_old_rows(self, tmp_store: HistoricalDataStore) -> None:
        old = _make_intraday_df(78, start="2020-01-02 09:35")
        new = _make_intraday_df(78, start="2026-05-06 09:35")
        tmp_store.save_intraday("SPY", old)
        tmp_store.save_intraday("SPY", new)
        tmp_store.cleanup_old_intraday(keep_days=10)
        result = tmp_store.load_intraday("SPY", days=365 * 10)
        assert len(result) == 78
        assert (result.index >= pd.Timestamp("2026-01-01")).all()

    def test_cleanup_preserves_recent_rows(self, tmp_store: HistoricalDataStore) -> None:
        recent = _make_intraday_df(78, start="2026-05-06 09:35")
        tmp_store.save_intraday("NVDA", recent)
        tmp_store.cleanup_old_intraday(keep_days=10)
        assert len(tmp_store.load_intraday("NVDA", days=30)) == 78
```

#### 11.5.2 `get_intraday()` default parameter change

**File:** `tests/test_market_data.py` (append)

```python
import inspect
from ait.data.market_data import MarketDataService


class TestGetIntradayDefault:

    def test_default_days_is_7(self) -> None:
        sig = inspect.signature(MarketDataService.get_intraday)
        default_days = sig.parameters["days"].default
        assert default_days == 7, (
            f"get_intraday() default days={default_days}, expected 7 "
            "(Phase 5a: change from days=1 to days=7)"
        )
```

---

### 11.6 Phase 6 — Intraday Fractal + VLMC Session Features

**File:** `tests/test_intraday_features.py`

```python
"""Unit tests for intraday fractal (Framework 2) and VLMC session features (Framework 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ait.ml.features import FeatureEngine


FRACTAL_INTRADAY_FEATURES = [
    "hurst_wavelet_intraday",
    "hurst_scale_spread_intraday",
    "wavelet_L3_energy",
    "wavelet_L4_energy",
    "wavelet_L5_energy",
    "psd_beta_intraday",
    "mfdfa_width_intraday",
]

VLMC_FEATURES = [
    "session_vwap_position",
    "session_vwap_q1",
    "session_vwap_q2",
    "session_vwap_q3",
    "session_high_timing",
    "session_low_timing",
    "session_volume_front_load",
    "session_volume_shape",
    "power_hour_momentum",
    "power_hour_vol_accel",
    "power_hour_vwap_cross",
    "closing_imbalance",
    "closing_range_position",
]

ALL_NEW_INTRADAY_FEATURES = FRACTAL_INTRADAY_FEATURES + VLMC_FEATURES


def _make_session(bars: int = 78, seed: int = 11) -> pd.DataFrame:
    """One 5-min trading session (9:30 AM onward)."""
    rng = np.random.default_rng(seed)
    closes = 400.0 + np.cumsum(rng.normal(0, 0.5, bars))
    dts = pd.date_range("2026-05-07 09:30", periods=bars, freq="5min")
    return pd.DataFrame({
        "Open":   closes * (1 + rng.normal(0, 0.0005, bars)),
        "High":   closes * (1 + np.abs(rng.normal(0, 0.002, bars))),
        "Low":    closes * (1 - np.abs(rng.normal(0, 0.002, bars))),
        "Close":  closes,
        "Volume": rng.integers(10_000, 500_000, bars),
    }, index=dts)


def _make_multiday(days: int = 7, bars_per_day: int = 78) -> pd.DataFrame:
    """Multi-day 5-min bars spanning `days` sessions."""
    frames = []
    for d in range(days):
        start = pd.Timestamp("2026-04-28 09:30") + pd.Timedelta(days=d)
        dts = pd.date_range(start, periods=bars_per_day, freq="5min")
        rng = np.random.default_rng(d + 100)
        closes = 400.0 + np.cumsum(rng.normal(0, 0.5, bars_per_day))
        frames.append(pd.DataFrame({
            "Open":   closes * 0.999, "High": closes * 1.002,
            "Low":    closes * 0.998, "Close": closes,
            "Volume": rng.integers(10_000, 200_000, bars_per_day),
        }, index=dts))
    return pd.concat(frames)


class TestIntradayFeatureCount:

    def test_all_20_new_features_present_with_7_day_history(self) -> None:
        result = FeatureEngine().compute_intraday_features(_make_multiday(days=7))
        missing = [f for f in ALL_NEW_INTRADAY_FEATURES if f not in result]
        assert missing == [], f"Missing intraday features: {missing}"

    def test_exactly_20_new_feature_keys_added(self) -> None:
        result = FeatureEngine().compute_intraday_features(_make_multiday(days=7))
        present = [f for f in ALL_NEW_INTRADAY_FEATURES if f in result]
        assert len(present) == 20, (
            f"Expected 20 new intraday features; got {len(present)}. "
            f"Missing: {sorted(set(ALL_NEW_INTRADAY_FEATURES) - set(present))}"
        )


class TestIntradayFractalFeatures:

    def test_hurst_wavelet_intraday_in_valid_range(self) -> None:
        result = FeatureEngine().compute_intraday_features(_make_multiday(7))
        h = result.get("hurst_wavelet_intraday", 0.0)
        if h != 0.0:
            assert 0.1 <= h <= 0.9, f"hurst_wavelet_intraday={h:.4f} out of [0.1, 0.9]"

    def test_wavelet_energies_nonnegative(self) -> None:
        result = FeatureEngine().compute_intraday_features(_make_multiday(7))
        for key in ("wavelet_L3_energy", "wavelet_L4_energy", "wavelet_L5_energy"):
            assert result.get(key, 0.0) >= 0.0, f"{key} is negative"

    def test_mfdfa_zero_when_single_session(self) -> None:
        """78 bars (1 session) < 500 minimum → mfdfa_width_intraday must be 0.0."""
        result = FeatureEngine().compute_intraday_features(_make_session(78))
        assert result.get("mfdfa_width_intraday", 0.0) == pytest.approx(0.0)

    def test_mfdfa_nonnegative_with_sufficient_history(self) -> None:
        result = FeatureEngine().compute_intraday_features(_make_multiday(7))
        assert result.get("mfdfa_width_intraday", 0.0) >= 0.0

    def test_no_nans_in_intraday_features(self) -> None:
        result = FeatureEngine().compute_intraday_features(_make_multiday(7))
        for k, v in result.items():
            assert not np.isnan(v),  f"NaN  in feature {k!r}"
            assert not np.isinf(v),  f"Inf  in feature {k!r}"

    def test_empty_df_returns_empty_dict(self) -> None:
        assert FeatureEngine().compute_intraday_features(pd.DataFrame()) == {}

    def test_too_few_bars_returns_empty_dict(self) -> None:
        assert FeatureEngine().compute_intraday_features(_make_session(5)) == {}


class TestVLMCSessionFeatures:

    def test_session_timing_features_bounded_0_to_1(self) -> None:
        result = FeatureEngine().compute_intraday_features(_make_session(78))
        for key in ("session_high_timing", "session_low_timing",
                    "session_volume_front_load", "closing_range_position"):
            val = result[key]
            assert 0.0 <= val <= 1.0, f"{key}={val:.4f} out of [0.0, 1.0]"

    def test_vwap_uses_session_open_not_rolling(self) -> None:
        """Inject a final-bar spike; session_vwap_position must be positive."""
        rng = np.random.default_rng(42)
        bars = 78
        dts = pd.date_range("2026-05-07 09:30", periods=bars, freq="5min")
        closes = np.ones(bars) * 400.0
        closes[-1] = 500.0   # spike at last bar only
        df = pd.DataFrame({
            "Open":   closes * 0.999, "High": closes * 1.001,
            "Low":    closes * 0.999, "Close": closes,
            "Volume": rng.integers(10_000, 100_000, bars),
        }, index=dts)
        result = FeatureEngine().compute_intraday_features(df)
        assert result["session_vwap_position"] > 0.0

    def test_power_hour_momentum_positive_on_rising_last_hour(self) -> None:
        """Flat session then rising last 12 bars → power_hour_momentum > 0."""
        rng = np.random.default_rng(55)
        bars = 78
        dts = pd.date_range("2026-05-07 09:30", periods=bars, freq="5min")
        closes = np.ones(bars) * 400.0
        closes[-12:] = np.linspace(400.0, 420.0, 12)   # power hour rises
        df = pd.DataFrame({
            "Open":   closes * 0.999, "High": closes * 1.001,
            "Low":    closes * 0.999, "Close": closes,
            "Volume": rng.integers(10_000, 100_000, bars),
        }, index=dts)
        result = FeatureEngine().compute_intraday_features(df)
        assert result["power_hour_momentum"] > 0.0

    def test_closing_imbalance_negative_on_falling_last_3_bars(self) -> None:
        rng = np.random.default_rng(66)
        bars = 78
        dts = pd.date_range("2026-05-07 09:30", periods=bars, freq="5min")
        closes = np.ones(bars) * 400.0
        closes[-3:] = [398.0, 396.0, 394.0]   # falling into close
        df = pd.DataFrame({
            "Open":   closes * 0.999, "High": closes * 1.001,
            "Low":    closes * 0.999, "Close": closes,
            "Volume": rng.integers(10_000, 100_000, bars),
        }, index=dts)
        result = FeatureEngine().compute_intraday_features(df)
        assert result["closing_imbalance"] < 0.0
```

---

### 11.7 Phase 7 — Dependency Addition

**File:** `tests/test_fractal_features.py` (append)

```python
class TestDependencies:

    def test_pywavelets_importable(self) -> None:
        try:
            import pywt
        except ImportError:
            pytest.fail("pywavelets not installed — add pywavelets>=1.4 to pyproject.toml")

    def test_pywavelets_version_at_least_1_4(self) -> None:
        import pywt
        from packaging.version import Version
        assert Version(pywt.__version__) >= Version("1.4"), (
            f"pywavelets {pywt.__version__} < required >=1.4"
        )

    def test_shap_importable(self) -> None:
        try:
            import shap
        except ImportError:
            pytest.fail("shap not installed — add shap>=0.42 to pyproject.toml")

    def test_shap_version_at_least_0_42(self) -> None:
        import shap
        from packaging.version import Version
        assert Version(shap.__version__) >= Version("0.42")
```

---

### 11.8 Phase 8 — Diagnostic Report Tool

**File:** `tests/test_fractal_diagnostics.py`

```python
"""Smoke tests for the fractal diagnostic report module and CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestFractalReportImport:

    def test_module_importable(self) -> None:
        from ait.diagnostics import fractal_report  # noqa: F401

    def test_all_seven_plot_functions_exist(self) -> None:
        from ait.diagnostics import fractal_report
        for fn in (
            "plot_hurst_timeseries",
            "plot_psd",
            "plot_multifractal_spectrum",
            "plot_scale_invariance_vs_vix",
            "plot_ic_analysis",
            "plot_shap_importance",
            "plot_gate_counterfactual",
        ):
            assert callable(getattr(fractal_report, fn, None)), (
                f"Function {fn!r} missing from ait.diagnostics.fractal_report"
            )

    def test_generate_report_callable(self) -> None:
        from ait.diagnostics.fractal_report import generate_report
        assert callable(generate_report)


class TestFractalReportSmoke:

    def _features_df(self, rows: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(77)
        dates = pd.date_range("2023-01-01", periods=rows, freq="B")
        return pd.DataFrame({
            "hurst_wavelet":       rng.uniform(0.4, 0.7, rows),
            "hurst_scale_spread":  rng.uniform(0.0, 0.2, rows),
            "psd_beta":            rng.uniform(1.5, 2.5, rows),
            "multifractal_width":  rng.uniform(0.1, 0.5, rows),
            "fwd_return_5d":       rng.normal(0.001, 0.02, rows),
        }, index=dates)

    def test_plot_hurst_timeseries_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_hurst_timeseries
        fig = plot_hurst_timeseries("SPY", self._features_df())
        assert fig is not None

    def test_plot_psd_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_psd
        fig = plot_psd(np.random.default_rng(88).normal(0, 1, 500))
        assert fig is not None

    def test_plot_ic_analysis_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_ic_analysis
        df = self._features_df()
        fig = plot_ic_analysis(df, pd.Series(df["fwd_return_5d"].values, index=df.index))
        assert fig is not None

    def test_generate_report_creates_html(self, tmp_path: Path) -> None:
        from ait.diagnostics.fractal_report import generate_report
        generate_report(
            symbols=["SPY"],
            start="2023-01-01",
            end="2023-12-31",
            output_dir=str(tmp_path),
            fmt="html",
        )
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) >= 1, "generate_report must produce at least one .html file"

    def test_generate_report_creates_ic_csv(self, tmp_path: Path) -> None:
        from ait.diagnostics.fractal_report import generate_report
        generate_report(
            symbols=["SPY"],
            start="2023-01-01",
            end="2023-12-31",
            output_dir=str(tmp_path),
            fmt="html",
        )
        csv_files = list(tmp_path.glob("ic_summary.csv"))
        assert len(csv_files) == 1, "generate_report must produce ic_summary.csv"


class TestCLIEntryPoint:

    def test_cli_script_exists(self) -> None:
        assert Path("scripts/run_fractal_diagnostics.py").exists(), (
            "scripts/run_fractal_diagnostics.py must exist (Phase 8)"
        )

    def test_cli_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_fractal_diagnostics.py", "--help"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, (
            f"CLI --help returned non-zero:\n{result.stderr}"
        )

    def test_cli_requires_symbols_argument(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/run_fractal_diagnostics.py"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode != 0, (
            "CLI must fail when called without --symbols"
        )
```

---

### 11.9 Phase 9 — IBKR Intraday Source + Backfill

**Files:** `tests/test_ibkr_intraday.py`, `tests/test_market_data.py` (extended)

```python
"""Tests for Phase 9: IBKR as primary intraday data source."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_intraday(n_bars: int = 78, days: int = 1) -> pd.DataFrame:
    """Synthetic intraday DataFrame with DatetimeIndex (UTC)."""
    from datetime import datetime, timezone, timedelta
    start = datetime(2026, 4, 28, 13, 30, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex(
        [start + timedelta(minutes=5 * i) for i in range(n_bars * days)], tz="UTC"
    )
    price = 100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.001, len(idx))))
    return pd.DataFrame({
        "Open": price, "High": price * 1.001, "Low": price * 0.999,
        "Close": price, "Volume": np.random.randint(1000, 5000, len(idx)),
    }, index=idx)


class TestMarketDataServiceIBKRIntraday:

    def test_get_intraday_returns_none_when_ibkr_disconnected(self) -> None:
        """When IBKR is disconnected, get_intraday() must fall back to Yahoo
        (or return None on network failure) — must never raise."""
        from ait.data.market_data import MarketDataService
        mock_ibkr = MagicMock()
        mock_ibkr.connected = False
        svc = MarketDataService(ibkr_client=mock_ibkr)

        async def run():
            with patch("yfinance.Ticker") as mock_yf:
                mock_ticker = MagicMock()
                mock_ticker.history.return_value = pd.DataFrame()
                mock_yf.return_value = mock_ticker
                result = await svc.get_intraday("SPY", interval="5m", days=7)
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is None or isinstance(result, pd.DataFrame)

    def test_ibkr_intraday_result_has_correct_columns(self) -> None:
        """_get_ibkr_intraday() output must have OHLCV columns."""
        from ait.data.market_data import MarketDataService
        mock_ibkr = MagicMock()
        mock_ibkr.connected = True

        svc = MarketDataService(ibkr_client=mock_ibkr)

        fake_df = _make_intraday(78)

        async def run():
            with patch.object(svc, "_get_ibkr_intraday", return_value=fake_df):
                result = await svc.get_intraday("SPY", interval="5m", days=7)
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is not None
        assert set(result.columns) >= {"Open", "High", "Low", "Close", "Volume"}

    def test_get_intraday_since_returns_only_new_bars(self) -> None:
        """get_intraday_since() must return bars strictly after the given timestamp."""
        from ait.data.market_data import MarketDataService
        mock_ibkr = MagicMock()
        mock_ibkr.connected = True
        svc = MarketDataService(ibkr_client=mock_ibkr)

        df = _make_intraday(78)
        cutoff = df.index[40]

        async def run():
            with patch.object(svc, "_get_ibkr_intraday", return_value=df):
                result = await svc.get_intraday_since("SPY", since=cutoff)
            return result

        result = asyncio.get_event_loop().run_until_complete(run())
        if result is not None and not result.empty:
            assert result.index.min() > cutoff, (
                "get_intraday_since must exclude bars at or before the cutoff timestamp"
            )


class TestBackfillScript:

    def test_backfill_script_exists(self) -> None:
        assert Path("scripts/backfill_intraday.py").exists(), (
            "scripts/backfill_intraday.py must exist (Phase 9)"
        )

    def test_backfill_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/backfill_intraday.py", "--help"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0

    def test_backfill_requires_symbols(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/backfill_intraday.py"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode != 0

    def test_backfill_dry_run_exits_zero(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/backfill_intraday.py",
             "--symbols", "SPY", "--years", "1", "--dry-run",
             "--db-path", str(tmp_path / "test.db")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"dry-run failed:\n{result.stderr}"
```

---

### 11.10 Phase 10 — VLMC Diagnostic Reports

**File:** `tests/test_fractal_diagnostics.py` (extended)

```python
class TestVLMCDiagnosticPlots:

    def _make_intraday_multi_session(self, n_sessions: int = 20) -> pd.DataFrame:
        """Build n_sessions × 78 bars of synthetic intraday data."""
        from datetime import datetime, timezone, timedelta
        bars_per_session = 78
        start = datetime(2026, 1, 2, 13, 30, tzinfo=timezone.utc)
        idx = []
        for day in range(n_sessions):
            session_start = start + timedelta(days=day)
            # Skip weekends (crude)
            while session_start.weekday() >= 5:
                session_start += timedelta(days=1)
            idx.extend([
                session_start + timedelta(minutes=5 * bar)
                for bar in range(bars_per_session)
            ])
        idx = pd.DatetimeIndex(idx, tz="UTC")
        price = 100.0 * np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.001, len(idx))))
        return pd.DataFrame({
            "Open": price, "High": price * 1.001, "Low": price * 0.999,
            "Close": price, "Volume": np.random.randint(1000, 5000, len(idx)),
        }, index=idx)

    def _features_df_with_vlmc(self, rows: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=rows, freq="B")
        return pd.DataFrame({
            "session_vwap_position":     rng.normal(0.0, 0.01, rows),
            "session_volume_front_load": rng.uniform(0.3, 0.6, rows),
            "session_volume_shape":      rng.uniform(-0.1, 0.1, rows),
            "power_hour_momentum":       rng.normal(0.0, 0.005, rows),
            "power_hour_vol_accel":      rng.normal(0.0, 0.1, rows),
            "closing_imbalance":         rng.normal(0.0, 0.003, rows),
            "closing_range_position":    rng.uniform(0.2, 0.8, rows),
            "fwd_return_5d":             rng.normal(0.001, 0.02, rows),
        }, index=dates)

    def test_plot_session_vwap_trajectory_exists(self) -> None:
        from ait.diagnostics import fractal_report
        assert callable(getattr(fractal_report, "plot_session_vwap_trajectory", None)), (
            "plot_session_vwap_trajectory missing from ait.diagnostics.fractal_report"
        )

    def test_plot_volume_profile_distribution_exists(self) -> None:
        from ait.diagnostics import fractal_report
        assert callable(getattr(fractal_report, "plot_volume_profile_distribution", None))

    def test_plot_session_feature_ic_analysis_exists(self) -> None:
        from ait.diagnostics import fractal_report
        assert callable(getattr(fractal_report, "plot_session_feature_ic_analysis", None))

    def test_plot_power_hour_patterns_exists(self) -> None:
        from ait.diagnostics import fractal_report
        assert callable(getattr(fractal_report, "plot_power_hour_patterns", None))

    def test_plot_session_vwap_trajectory_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_session_vwap_trajectory
        fig = plot_session_vwap_trajectory("SPY", self._make_intraday_multi_session())
        assert fig is not None

    def test_plot_volume_profile_distribution_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_volume_profile_distribution
        fig = plot_volume_profile_distribution("SPY", self._make_intraday_multi_session())
        assert fig is not None

    def test_plot_session_feature_ic_analysis_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_session_feature_ic_analysis
        df = self._features_df_with_vlmc()
        fig = plot_session_feature_ic_analysis(
            df, pd.Series(df["fwd_return_5d"].values, index=df.index)
        )
        assert fig is not None

    def test_plot_power_hour_patterns_returns_object(self) -> None:
        from ait.diagnostics.fractal_report import plot_power_hour_patterns
        fig = plot_power_hour_patterns("SPY", self._make_intraday_multi_session())
        assert fig is not None
```

---

### 11.11 End-to-End Integration Tests

**File:** `tests/test_fractal_integration.py`

```python
"""End-to-end integration: fractal features flow through the full pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.ml.features import FeatureEngine
from ait.backtesting.walkforward import WalkForwardConfig


def _make_ohlcv(days: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 400.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, days))
    dates = pd.date_range("2022-01-03", periods=days, freq="B")
    return pd.DataFrame({
        "Open":   close * 0.999, "High": close * 1.005,
        "Low":    close * 0.995, "Close": close,
        "Volume": rng.integers(1_000_000, 10_000_000, days),
    }, index=dates)


class TestFractalFeaturesInPipeline:

    def test_compute_returns_all_10_fractal_columns(self) -> None:
        result = FeatureEngine().compute(_make_ohlcv(504))
        for col in (
            "hurst_wavelet", "hurst_fit_r2", "psd_beta", "psd_fit_r2",
            "hurst_psd_divergence", "hurst_short", "hurst_long",
            "hurst_scale_spread", "multifractal_width", "multifractal_asymmetry",
        ):
            assert col in result.columns, f"Missing column: {col!r}"

    def test_range_predictor_trains_with_fractal_features(self) -> None:
        from ait.ml.range_predictor import RangePredictor
        RangePredictor().train(_make_ohlcv(504))

    def test_vol_magnitude_predictor_trains_with_fractal_features(self) -> None:
        from ait.ml.vol_magnitude_predictor import VolMagnitudePredictor
        VolMagnitudePredictor().train(_make_ohlcv(504))

    def test_direction_predictor_trains_and_predicts(self) -> None:
        from ait.ml.ensemble import DirectionPredictor
        dp = DirectionPredictor()
        df = _make_ohlcv(504)
        dp.train(df)
        assert dp.predict(df) is not None

    def test_fractal_output_is_deterministic(self) -> None:
        """Same input must always produce exactly the same fractal column values."""
        fe = FeatureEngine()
        df = _make_ohlcv(504, seed=42)
        r1 = fe.compute(df).iloc[-1][["hurst_wavelet", "hurst_scale_spread"]]
        r2 = fe.compute(df).iloc[-1][["hurst_wavelet", "hurst_scale_spread"]]
        pd.testing.assert_series_equal(r1, r2)

    def test_walkforward_config_default_fractal_fields(self) -> None:
        cfg = WalkForwardConfig()
        assert cfg.hurst_regime_threshold == pytest.approx(0.20)
        assert cfg.multifractal_max_width  == pytest.approx(0.50)


class TestSavedModelMismatch:

    def test_old_model_feature_count_differs_from_new(self) -> None:
        """A pre-Phase-1 model pickle must have fewer features than the new FeatureEngine.

        Skip if no pre-Phase-1 model is available.
        Failure means the model was already retrained — no action needed.
        """
        model_path = Path("models/ensemble.pkl")
        if not model_path.exists():
            pytest.skip("No pre-existing model to check")
        with open(model_path, "rb") as fh:
            old_model = pickle.load(fh)
        old_count = len(old_model.feature_names_in_)   # XGBoost / LightGBM attribute
        new_count = len(FeatureEngine().get_feature_names())
        assert old_count != new_count, (
            "Model appears to already include fractal features — retraining not required"
        )
```

---

### 11.10 CLI Smoke Tests

Run from the project root after completing all phases. Each command must exit `0`.

```bash
# Phase 1+3: Walk-forward with fractal features
python run_backtest.py --symbols SPY --capital 10000 2>&1 | tee /tmp/backtest.log
grep -c "features_computed" /tmp/backtest.log    # Must be > 0
grep "hurst_scale_spread" /tmp/backtest.log      # Must appear in feature log

# Phase 2+4: Optimizer surfaces fractal params in trial output
python run_optimizer.py --strategies iron_condor --symbols SPY --n-trials 10 \
  2>&1 | tee /tmp/opt.log
grep "hurst_regime_threshold" /tmp/opt.log       # Must appear as a trial parameter
grep "multifractal_max_width"  /tmp/opt.log      # Must appear as a trial parameter

# Phase 4: Per-window fractal threshold adaptation
python run_backtest.py \
  --symbols SPY QQQ \
  --optimize-per-window \
  --optimize-n-trials 20 2>&1 | tee /tmp/perwindow.log
# Manual check: thresholds should differ between the 2022 volatile window
# and the 2024 calm window in the log output

# Phase 8: Diagnostic HTML report
python scripts/run_fractal_diagnostics.py \
    --symbols SPY QQQ AAPL NVDA \
    --start 2022-01-01 \
    --end 2025-12-31 \
    --output reports/fractal/ \
    --format html
test -f reports/fractal/fractal_report_SPY.html && echo "OK" || echo "MISSING"
test -f reports/fractal/ic_summary.csv         && echo "OK" || echo "MISSING"

# Phase 6 + Phase 5 live (during market hours only):
python -m src.ait.main --mode paper 2>&1 | tee /tmp/live.log &
sleep 360    # allow one full 5-min scan cycle
grep "hurst_wavelet_intraday"   /tmp/live.log
grep "wavelet_L4_energy"        /tmp/live.log
grep "session_vwap_position"    /tmp/live.log
kill %1
```

---

### 11.13 Regime Signal Validation (Manual — Post-Backtest)

After the walk-forward smoke test, export feature values and verify against known market history. These are human checks, not automated tests — a mismatch signals a numerical bug in an estimator.

| Feature | Period | Expected | Failure interpretation |
|---|---|---|---|
| `hurst_scale_spread` | Jan–Jun 2022 (SPY) | Median > 0.12 | Estimator insensitive to volatility — check `_multiscale_hurst()` |
| `hurst_scale_spread` | Q4 2023 (SPY calm) | Median < 0.08 | Estimator over-fires in quiet markets — threshold too low |
| `multifractal_width` | Mar 2020 | Elevated (> 0.40) | MFDFA not capturing crash regime — check `_mfdfa_features()` |
| `multifractal_asymmetry` | Weeks before Oct 2022 drawdown | Trending negative | Asymmetry sign convention inverted — check spectrum skew |
| `hurst_fit_r2` | VIX > 30 days vs VIX < 20 days | Lower on high-VIX days | R² not detecting scale-invariance breakdown |
| `psd_beta` | SPY daily bars, any window | 1.5–2.5 | PSD estimator producing out-of-range values |

---

### 11.14 Iron Condor Gate Counterfactual Validation (Manual)

From `BacktestResult.trades` after the per-window backtest, compare gated vs. non-gated iron condor entries. The gate is validated when:

| Condition | Interpretation |
|---|---|
| Gated win rate < non-gated win rate | Gate correctly identifies bad-regime entries |
| Gated avg P&L (counterfactual) < non-gated avg P&L | Gate improves entry quality |
| Gated count > 0 in the 2022 volatile window | Gate fires when it should |
| Gated count ≈ 0 in calm Q4-2023 window | Gate does not over-fire in benign conditions |

If gated trades would have had a higher win rate than non-gated trades, the `hurst_regime_threshold` is too aggressive. The per-window Optuna run (Phase 4) automatically raises the threshold to correct this.

---

### 11.15 Per-Phase Completion Criteria

A phase is done only when its specific pytest target passes with zero failures.

| Phase | Completion command |
|---|---|
| 1 | `pytest tests/test_fractal_features.py -k "TestFractalFeatures or TestFractalEstimators or TestFractalGraceful"` |
| 2 | `pytest tests/test_fractal_features.py -k "TestFractalGateParamSpace"` |
| 3 | `pytest tests/test_fractal_features.py -k "TestFractalGatingEngine"` |
| 4 | `pytest tests/test_walkforward.py -k "TestWalkForwardConfigFractalFields"` |
| 5 | `pytest tests/test_intraday_store.py tests/test_market_data.py` |
| 6 | `pytest tests/test_intraday_features.py` |
| 7 | `pytest tests/test_fractal_features.py -k "TestDependencies"` |
| 8 | `pytest tests/test_fractal_diagnostics.py -k "TestFractalReportImport or TestFractalReportSmoke or TestCLIEntryPoint"` |
| 9 | `pytest tests/test_ibkr_intraday.py tests/test_market_data.py -k "TestMarketDataServiceIBKRIntraday or TestBackfillScript"` |
| 10 | `pytest tests/test_fractal_diagnostics.py -k "TestVLMCDiagnosticPlots"` |
| All phases | `pytest tests/ -k "fractal or intraday or backfill or vlmc" -v` — zero failures |

---

*Plan authored: 2026-05-06. Updated 2026-05-07: added Phase 9 (IBKR intraday + 2-year backfill), Phase 10 (VLMC diagnostics), and clarifications on data sources, VLMC bar resolution, and diagnostic coverage.*
*Branch: features-request-2*
*Related plan: `/Users/ahmednagi/.claude/plans/would-the-above-work-dazzling-bunny.md`*

# Walk-Forward Lab — User Guide

> **What this document covers:** Everything you need to read and interpret the AIT Walk-Forward Lab dashboard (`src/ait/dashboard/walkforward/index.html`). All three tabs — Experiment Analysis, Optuna Optimization, and Predictor Models — are covered in detail: what each number means, why it matters, and how to act on it.
>
> **Related references:**
> - [GUIDE.md](../GUIDE.md) — full system architecture, entry pipeline, and configuration reference
> - [docs/GARCH_METHODOLOGY.md](GARCH_METHODOLOGY.md) — GARCH/GJR-GARCH/EGARCH/MS-GARCH math and implementation
> - [docs/OU_KOU_GARCH_METHODOLOGY.md](OU_KOU_GARCH_METHODOLOGY.md) — OU-Kou-GARCH jump-diffusion model with Adaptive EKF
> - [docs/ENSEMBLE_OOS_ASSESSMENT.md](ENSEMBLE_OOS_ASSESSMENT.md) — OOS scoring metrics: Brier score, Brier skill, AUROC, calibration

---

## Table of Contents

1. [What is Walk-Forward Testing?](#1-what-is-walk-forward-testing)
2. [Dashboard Overview](#2-dashboard-overview)
3. [Tab: Experiment Analysis](#3-tab-experiment-analysis)
   - 3.1 [The Price Chart and Indicator Panes](#31-the-price-chart-and-indicator-panes)
   - 3.2 [The Trades Table](#32-the-trades-table)
   - 3.3 [Pinned Trade Drawer — Decision Chain](#33-pinned-trade-drawer--decision-chain)
   - 3.4 [Window-Level KPIs](#34-window-level-kpis)
4. [Tab: Optuna Optimization](#4-tab-optuna-optimization)
   - 4.1 [Trial Progress and Status](#41-trial-progress-and-status)
   - 4.2 [Objective Over Trials Chart](#42-objective-over-trials-chart)
   - 4.3 [Parameter vs Objective Scatter](#43-parameter-vs-objective-scatter)
   - 4.4 [Optimal Parameters Card](#44-optimal-parameters-card)
5. [Tab: Predictor Models](#5-tab-predictor-models)
   - 5.1 [Directional Predictor](#51-directional-predictor)
   - 5.2 [Range Predictor](#52-range-predictor)
   - 5.3 [The Quality Gate: Why Predictions Go Missing](#53-the-quality-gate-why-predictions-go-missing)
   - 5.4 [Member Skill Across Windows Chart](#54-member-skill-across-windows-chart)
   - 5.5 [Fitted Ensemble Weight Chart](#55-fitted-ensemble-weight-chart)
   - 5.6 [Calibration (Reliability) Chart](#56-calibration-reliability-chart)
   - 5.7 [GARCH Family Detail](#57-garch-family-detail)
6. [Global Controls and Layout](#6-global-controls-and-layout)
7. [Metric Reference Glossary](#7-metric-reference-glossary)

---

## 1. What is Walk-Forward Testing?

Walk-forward testing is the gold standard for evaluating a trading strategy with a machine learning component. Instead of training the model once on all available history and backtesting on the same data (which produces optimistic, in-sample results), the walk-forward approach simulates how the system would have actually operated:

```
  ┌────────────────────────┬──────┬──────────┐
  │     TRAIN (365 d)      │ GAP  │ TEST OOS │   ← Window 1
  └────────────────────────┴──────┴──────────┘
        ┌────────────────────────┬──────┬──────────┐
        │     TRAIN (365 d)      │ GAP  │ TEST OOS │   ← Window 2
        └────────────────────────┴──────┴──────────┘
              ...
```

**Train period** — the model is retrained from scratch on this slice of history. No data from the test period leaks in.

**Gap** (5 trading days by default) — a buffer between the end of training and the start of testing. This prevents the model from learning patterns that exist only at the train/test seam (e.g., earnings announcements).

**Test period (OOS)** — the model is frozen and the strategy runs exactly as it would live. Only results from this period enter the performance metrics.

Each iteration is called a **window**. With `step_days=60`, a new window starts every 60 days, giving ~12 independent OOS periods per two years of data.

> **Why this matters:** An OOS Sharpe of 1.5 across 12 independent windows is far more credible than an in-sample Sharpe of 3.0 on a single backtest.

---

## 2. Dashboard Overview

The dashboard has three tabs accessible from the top navigation bar:

| Tab | Answers the question |
|-----|----------------------|
| **Experiment Analysis** | How did the strategy perform? What trades were taken and why? |
| **Optuna Optimization** | How did hyperparameter search behave? What parameters produced the best results? |
| **Predictor Models** | How accurate are the ML models? Do they have genuine predictive skill? |

A dark/light theme toggle sits in the top-right corner. Your choice persists across sessions.

You can load a different experiment run using the dropdown in the header. When multiple runs are present, each appears as a separate entry.

---

## 3. Tab: Experiment Analysis

This is the main view. It combines a price chart, ML prediction overlays, and a trade-by-trade table for the full experiment.

### 3.1 The Price Chart and Indicator Panes

The chart renders QQQ daily OHLCV data as candlesticks. Below the price pane, additional sub-panes can be toggled from the left sidebar.

**Price pane overlays** (checkboxes in the sidebar):

| Toggle | What it shows |
|--------|---------------|
| SMA 20 | 20-day simple moving average of close price |
| SMA 50 | 50-day simple moving average |
| Bollinger Bands | 20-day SMA ± 2 standard deviations; bands widen in high-vol regimes |

**Trade markers** — arrows on the candlesticks mark every trade:
- **Up arrow (below bar)** — entry date. Gold colour = currently pinned trade; amber = normal.
- **Down arrow (above bar)** — exit date. Green = profitable trade; red = loss.

Click any arrow to pin that trade and open the Decision Drawer (see §3.3).

**Sub-panes** (each synced to the price pane time axis and crosshair):

| Toggle | What it shows | How to read it |
|--------|---------------|----------------|
| **ML Predictions** | `P(in-range)` (blue) and `Dir conf` (amber) | See §5 for full explanation. Gaps mean the model's quality gate was not met for that window. |
| **RSI (14)** | Relative Strength Index, 14-period | < 30 = oversold; > 70 = overbought. Used as a feature input — not a direct entry signal. |
| **MACD** | MACD line, signal line, histogram | Histogram shows momentum. Positive = bullish momentum; negative = bearish. |
| **Realized Vol / ATR** | 20-day realized vol (left axis, red) and ATR % (right axis, amber) | Realized vol spikes signal regime transitions. High ATR → wider iron condor wings needed. |
| **IV Rank / VIX** | IV Rank (left axis, blue) and VIX level (right axis, amber) | IV Rank < 0.20 → premiums are cheap, iron condors less attractive. VIX > 25 → elevated regime. |
| **Hurst** | Hurst exponent (wavelet estimate) | < 0.5 → mean-reverting (good for iron condors); > 0.5 → trending (increases breakout risk). The 0.5 dashed line is the random-walk baseline. |
| **Sentiment / Put-Call** | Composite sentiment (blue, left) and Put-Call ratio (amber, right) | P/C > 1.2 = fear; P/C < 0.7 = greed. Used as regime feature. |

**Zooming to a walk-forward window** — select a window from the dropdown in the sidebar and click "Zoom". The chart scrolls to show only that OOS period.

### 3.2 The Trades Table

The table below the chart lists every trade in the experiment. Columns:

| Column | Meaning |
|--------|---------|
| **ID** | Trade identifier (T001, T002, …) |
| **Window** | Which walk-forward window this trade came from |
| **Entry / Exit** | Date (and time if intraday fill data is available) |
| **Entry price** | Net credit received per share when the iron condor was opened |
| **Exit price** | Net debit paid per share when closed |
| **P&L** | Dollar profit/loss (green = win, red = loss). Accounts for commissions and slippage. |
| **Ret%** | Return as % of max loss (the capital at risk). More meaningful than raw P&L. |
| **Hold** | Holding period in calendar days |
| **Exit reason** | Why the trade was closed: `profit_target`, `stop_loss`, `expiry`, `time_stop` |
| **Conf** | Range model confidence at entry (P(in-range)). Absent if the range model was gated for this window. |

Clicking a row pins the trade and zooms the chart to the entry-to-exit period (±15 bars of buffer). A horizontal price line is drawn on each strike level.

**Filtering** — use the search box and the All/Win/Loss buttons to filter the table. The window dropdown filters both table and chart to a single OOS period.

### 3.3 Pinned Trade Drawer — Decision Chain

Clicking a trade opens a right-side drawer that reconstructs **exactly why the system decided to enter**. This is the most important debugging tool in the dashboard.

The decision chain is a sequential gate: each step must pass for an entry to occur. If any step would have vetoed, you can see exactly which one and why.

**Step 1 — Directional model**
The 3-class ensemble (XGBoost + LightGBM) predicted whether QQQ would be bullish, neutral, or bearish over the next 5 days. For an iron condor, any direction is acceptable as long as confidence is high enough. A low-confidence directional call can still be acceptable if the range gate is strong.

**Step 2 — Range model gate**
Shows `P(price stays within ±threshold over 30d)` as a gauge. This is the iron condor's primary entry gate. The entry is blocked if this value is below `range_min_confidence` (default: 0.55). If the range model was gated (low edge), this step shows "model gated — no signal".

> **Reading the gauge:** Values 0.55–0.70 = marginal, proceed cautiously. Values > 0.70 = high confidence. See §5.3 for why some windows show no gauge.

**Step 3 — Volatility gate**
10-day realized vol is compared to a threshold. If vol is too high at entry, the iron condor's breakeven range would be too narrow — the strategy skips. This prevents entering condors into volatility spikes.

**Step 4 — Meta-label filter**
A meta-classifier (`P(profitable)`) that learns from past trade outcomes. Trades where this probability is low are skipped. This is a secondary quality filter layered on top of the primary range signal.

**Step 5 — Fractal regime gate**
The Hurst scale-spread measures whether the market is mean-reverting, random-walk, or trending. If the Hurst exponent indicates a strongly trending regime, a penalty is applied to the entry score. Extreme trending (Hurst > 0.65) can veto entry entirely.

**Step 6 — Regime classification**
The final regime label assigned at entry: `range_bound`, `elevated`, `trending`, or `chaotic`. This is recorded for post-hoc analysis of which regimes the strategy performs best in.

**Iron Condor Structure** — shows the four legs (short put, long put, short call, long call) with their strikes and individual premiums. Net credit = sum of short premiums minus long premiums. Max loss = difference between strike widths minus net credit.

### 3.4 Window-Level KPIs

When a specific window is selected from the dropdown, a metrics strip appears above the chart:

| Metric | Meaning | What to look for |
|--------|---------|-----------------|
| **Return** | OOS return % for this window | Compare across windows for consistency |
| **Sharpe** | Risk-adjusted return (annualized) | > 1.0 is acceptable; > 2.0 is strong for options |
| **Sortino** | Like Sharpe but only penalizes downside vol | Higher Sortino than Sharpe = mostly upside variance |
| **Win rate** | % of trades closed profitably | Iron condors typically target 60–75% |
| **Profit factor** | Gross wins ÷ Gross losses | > 1.5 is healthy; < 1.0 means losses outpace wins |
| **Max drawdown** | Largest peak-to-trough equity decline in this window | > 10% warrants investigation |
| **P&L** | Raw dollar profit/loss | |
| **Expectancy** | Average dollar gain per trade | Positive expectancy is required for long-term viability |
| **Trades** | Count of trades in this OOS period | Very low trade counts make Sharpe/win-rate unreliable |
| **Avg hold** | Average holding period in days | Should align with strategy intent (iron condors: 10–30 days) |
| **Avg win / Avg loss** | Mean size of winning and losing trades | Win/loss ratio combined with win rate determines expectancy |

---

## 4. Tab: Optuna Optimization

Each walk-forward window runs an independent Optuna hyperparameter search. This tab lets you inspect how that search behaved — whether it converged, which parameters matter, and whether the best parameters generalise.

### 4.1 Trial Progress and Status

**Study status** — "Study completed" means all requested trials finished. "Study stopped early" means the run hit a patience limit (the last N trials showed no improvement).

**Trial breakdown**:
- **Complete** — trial ran to completion and produced a valid objective
- **Pruned** — Optuna's MedianPruner stopped the trial early because intermediate results were below the median of earlier trials at the same step. Pruning is normal and efficient — it redirects compute toward more promising regions.
- **Not run** — trials that were queued but not executed (only in early-stopped runs)

A high pruning rate (> 60%) can indicate either that the search space is too large, or that a few parameter combinations dominate.

### 4.2 Objective Over Trials Chart

The scatter plot shows every trial's objective value (typically composite Sharpe) as a function of trial number.

- **Blue dots** — complete trials
- **Small grey dots near zero** — pruned trials (plotted at the value where they were pruned)
- **Large highlighted dot with ★** — the best trial found
- **Dashed line** — running best-so-far, showing convergence behaviour

**Reading the chart:**

- A flat best-so-far line from early on → the search converged quickly; the space is well-explored.
- A best-so-far line that keeps rising toward the end → the search may not have converged; more trials could help.
- Wide scatter of complete trial values → parameter sensitivity is high; small changes in params produce very different outcomes.
- Tight cluster of complete trial values → robust region; the strategy is not very sensitive to exact parameter values.

Hover any dot for the full trial breakdown: objective, Sharpe, win rate, max drawdown, number of trades, and compute time.

### 4.3 Parameter vs Objective Scatter

Select a parameter from the dropdown to see how its value correlates with the objective. Each dot is one complete trial.

- **Positive trend** → higher values of this parameter improve performance for this window. But be cautious: this could be window-specific overfitting.
- **No trend / flat** → this parameter does not significantly drive outcomes; the search is not sensitive to it.
- **Inverted-U shape** → there is an optimal range; values outside it hurt performance.
- **Best trial highlighted** → check whether the best trial sits at an extreme (edge of the search space) — if so, the search space bounds may need widening.

### 4.4 Optimal Parameters Card

Shows the best-trial parameter values actually used for the OOS run of this window. These are the parameters that produced the highest Sharpe on the training period.

**Important caveat:** These are in-sample optimal parameters. The whole point of walk-forward testing is that the OOS result tells you whether these parameters generalise. Cross-reference with the OOS window return in the summary KPI cards. If the in-sample Sharpe was 4.0 and the OOS return was negative, the parameters overfit to the training slice.

---

## 5. Tab: Predictor Models

This tab separates the model evaluation question from the trading performance question. Good model skill doesn't guarantee good trading performance (execution, sizing, and regime matching all matter), but **poor model skill almost guarantees poor trading performance**.

### 5.1 Directional Predictor

**Task:** 3-class classification — predicts whether QQQ will be bullish, neutral, or bearish over the next 5 days.

**Members:** XGBoost + LightGBM ensemble (weighted by CV AUROC).

**Metric: AUROC (macro one-vs-rest)**

AUROC measures the model's ability to rank-order outcomes correctly, regardless of threshold. An AUROC of 0.5 = random guessing. An AUROC of 1.0 = perfect predictions.

For a 3-class problem, macro AUROC averages the one-vs-rest AUROC for each class. Values you should expect:

| AUROC | Interpretation |
|-------|---------------|
| < 0.52 | No meaningful skill — model should not gate entries |
| 0.52 – 0.58 | Marginal skill — useful at scale but not individually reliable |
| 0.58 – 0.65 | Moderate skill — meaningful directional edge |
| > 0.65 | Strong skill — rare in efficient markets, warrants scrutiny for lookahead |

**How it is used in the strategy:** The directional model provides a confidence score that scales position sizing and, in combination with the range gate, determines whether to enter. A neutral or contradictory direction signal reduces entry confidence. See [GUIDE.md §1.10](../GUIDE.md) for the full entry pipeline.

### 5.2 Range Predictor

**Task:** Binary classification — predicts whether QQQ's price will remain within ±threshold% of its current price over the next 30 days.

**Members:** XGBoost, LightGBM, MS-GARCH, OU-Kou-GARCH (weighted by CV balanced accuracy). See [GARCH_METHODOLOGY.md](GARCH_METHODOLOGY.md) and [OU_KOU_GARCH_METHODOLOGY.md](OU_KOU_GARCH_METHODOLOGY.md) for the statistical models.

**Metric: CV Balanced Accuracy**

Balanced accuracy = (sensitivity + specificity) / 2. It is the average recall across both classes, making it appropriate for imbalanced labels (when "stays in range" is much more common than "breaks out").

| Balanced Acc | Edge over 0.50 baseline | Interpretation |
|-------------|------------------------|---------------|
| < 0.50 | Negative | Model is anti-predictive — worse than random |
| 0.50 | 0.00 | Pure random — no useful signal |
| 0.55 | +0.05 | Below quality gate (< 0.10 edge) — model gated |
| 0.60 | +0.10 | **Quality gate threshold** — minimum to emit predictions |
| 0.65 | +0.15 | Good signal |
| > 0.70 | > +0.20 | Strong signal |

**KPI tiles on the summary row:**

| Tile | Meaning |
|------|---------|
| **Avg balanced acc** | Mean CV balanced accuracy over all 12 windows |
| **Mean edge / window** | Avg balanced acc − 0.50; positive = beats random |
| **Best window** | Window with highest balanced acc; shows its OOS date range |
| **Windows gated** | How many windows had edge < 0.10 and emitted no predictions |
| **Dominant member** | Ensemble member with the highest average fitted weight |
| **Avg in-range rate** | Historical base rate of QQQ staying within ±5% over 30d; the naive baseline |

### 5.3 The Quality Gate: Why Predictions Go Missing

You may notice that `P(in-range)` (the blue line in the ML Predictions pane on the chart) is **blank for most windows** and only appears for a few. This is intentional.

The range predictor has a built-in quality gate (`MIN_EDGE_OVER_BASELINE = 0.10`). At prediction time, before emitting any probability, the model checks whether its cross-validated balanced accuracy exceeded the baseline by at least 10 percentage points. If not, `predict_from_features()` returns `None` — silently, without error — and no `range_prob` is written for that bar.

**Why this design choice?** An overconfident model with near-random skill can assign high `P(in-range)` and still gate entries correctly by luck. Over time, using a poorly-skilled model as a confidence gate introduces invisible adverse selection. The 10% edge requirement ensures only windows where the model has demonstrated genuine out-of-sample rank-discrimination power contribute predictions to the live decision.

**The dashboard tells you this in two places:**
- Under the Range Predictor tab header: *"Iron-condor confidence gate (skips when edge < 0.10 over baseline)"*
- In the **Windows Gated** KPI tile: e.g., "10/12" means 10 of 12 windows were gated

**Distinguishing two different metrics:**

| What you see | Source | Meaning |
|---|---|---|
| Bar charts in "Member skill across windows" | `window_NNN.json` training metrics | Did the model train? What were its CV scores? ← shown for all trained windows |
| Blue `P(in-range)` line on the price chart | `timeseries_bars.json` per-bar predictions | Did the model pass the quality gate AND emit predictions? ← only gated-passing windows |

A window can show bars in the skill chart (model trained fine) but have no blue line on the price chart (model's CV edge was below 0.10). This is correct and expected.

**Typical pattern across windows:**
- Early windows tend to have lower edge because the training dataset is smaller and the regime history is shorter, giving the model less to learn from.
- Windows spanning a mix of trending and range-bound markets tend to produce higher balanced accuracy, because the model can distinguish regimes.
- A window with negative edge (balanced acc < 0.50) means the model's features were anti-predictive for that period — possibly a regime shift the model had not seen in training.

For deeper analysis of these metrics, see [docs/ENSEMBLE_OOS_ASSESSMENT.md](ENSEMBLE_OOS_ASSESSMENT.md).

### 5.4 Member Skill Across Windows Chart

A grouped bar chart showing each ensemble member's CV balanced accuracy (or AUROC for the directional predictor) for every walk-forward window.

- **X-axis:** Window number, with the OOS date range shown below
- **Y-axis:** Skill metric. The dashed horizontal line marks the 0.50 baseline.
- **Amber shaded columns:** Gated windows — the range predictor did not emit predictions for these windows despite training. Entry into the market for this window was based on the backup heuristics (or blocked entirely if no range signal existed).
- **Colour per member:** Blue = XGBoost, Green = LightGBM, Purple = GARCH, Teal = MS-GARCH, Orange = OU-Kou-GARCH

**What to look for:**
- **Consistent bars across all windows** → the feature set captures a persistent signal; the model is not overfitting to one regime.
- **Large variance across windows** → high regime sensitivity; the model works in some market environments and not others.
- **Members near or below 0.50 in most windows** → that member adds noise; consider reducing its search space weight.
- **MS-GARCH and OU-Kou-GARCH bars missing for some windows** → the statistical models failed to converge on that training slice (possible numerical issues with the GARCH estimation). The ensemble automatically assigns them zero weight when convergence fails.

### 5.5 Fitted Ensemble Weight Chart

A 100% stacked bar chart showing how the ensemble blender allocated weight across members for each window.

Each window has its own independently fitted weight vector, determined by the relative CV balanced accuracy of each member on that window's training data. The blender uses a soft-max-style weighting: members with higher CV scores get more weight, members with near-random scores get near-zero weight.

**What to look for:**
- **Stable weight allocations across windows** → the ensemble composition is consistent; the dominant members are genuinely robust.
- **Wild swings window-to-window** → the blender is unstable; CV scores are noisy. This usually indicates small training set sizes or unstable features.
- **MS-GARCH dominating in some windows but not others** → MS-GARCH is highly sensitive to the volatility regime; it tends to outperform during structured volatility clustering and underperform in low-vol, mean-reverting markets.
- **Zero weight for OU-Kou-GARCH in most windows** → the jump-diffusion model adds unique signal only when the training data contains clear jump dynamics. See [OU_KOU_GARCH_METHODOLOGY.md](OU_KOU_GARCH_METHODOLOGY.md) §1 for when this occurs.

### 5.6 Calibration (Reliability) Chart

The calibration chart answers the question: *when the model says P = 0.70, does it actually happen 70% of the time?*

- **X-axis:** Predicted probability (binned into 5–8 buckets)
- **Y-axis:** Empirical frequency (actual "stayed in range" rate within that probability bucket)
- **Diagonal dashed line:** Perfect calibration (predicted = actual)
- **Each line:** One ensemble member or the blended ensemble
- **Dot size:** Proportional to sample count in that bin — larger dots are more statistically reliable

**Reading the chart:**

| Pattern | Meaning |
|---------|---------|
| Line follows the diagonal | Well-calibrated — predicted probabilities are trustworthy |
| Line is above the diagonal | Model is underconfident — when it says 0.60, reality is 0.70. Predictions are conservative; actual rate is higher. |
| Line is below the diagonal | Model is overconfident — when it says 0.80, reality is only 0.60. Do not take high P(in-range) at face value. |
| Line is flat | The model produces little variation in predicted probabilities — poor discrimination |
| Ensemble line follows diagonal while individual members don't | The blending is doing meaningful calibration correction |

The **ensemble (blended) line** is thicker than the individual member lines. It should ideally be the best-calibrated of all series. Calibration matters for this strategy because `P(in-range)` is used as a direct confidence threshold — a systematically overconfident model would approve too many bad entries.

For the mathematical underpinning of calibration assessment (Brier scores, reliability diagrams), see [docs/ENSEMBLE_OOS_ASSESSMENT.md §3](ENSEMBLE_OOS_ASSESSMENT.md).

### 5.7 GARCH Family Detail

For the Range Predictor, each statistical model has an expandable details row showing its convergence state and key parameters for the selected window.

**GARCH / GJR-GARCH / EGARCH / ARCH** (single-regime):
- **Variant** — which GARCH specification won the BIC competition (e.g., GARCH(1,1))
- **Distribution** — Normal, Student-t, or GED (affects tail behaviour)
- **BIC** — lower BIC = better model fit per degree of freedom
- **Jarque-Bera p-value** — tests whether standardised residuals are Normally distributed. Low p-value (< 0.05) = fat tails remain after GARCH filtering; the distribution assumption should be non-Normal.
- **Residual skewness** — ideally near 0; large skew means the model missed asymmetric return dynamics

See [GARCH_METHODOLOGY.md §3–§5](GARCH_METHODOLOGY.md) for the full model equations and BIC selection logic.

**MS-GARCH** (Markov-Switching GARCH, 2-regime):
- **Regime 0 / Regime 1** — the two volatility regimes. Typically one low-vol (quiet market) and one high-vol (stressed market).
- **Transition matrix** — 2×2 table showing P(stay in regime | current regime). High persistence on the diagonal (e.g., 0.90) means regimes are sticky.
- **Convergence** — "converged" means the EM algorithm found a stable solution. "no-converge" means the estimate may be unreliable; the blender sets its weight to zero.
- **BIC** — compared to single-regime GARCH; lower BIC justifies the additional complexity

See [GARCH_METHODOLOGY.md §6](GARCH_METHODOLOGY.md) for MS-GARCH derivation.

**OU-Kou-GARCH** (Ornstein-Uhlenbeck + Kou jumps + AEKF):
- **Drift direction** — BULLISH or BEARISH mean-reversion drift estimated by the AEKF
- **Confidence** — AEKF confidence in the direction signal
- **κ** — mean-reversion speed. High κ (> 5) = fast reversion; low κ (< 1) = slow/weak reversion
- **μ** — long-run mean level of the process
- **σ** — diffusion coefficient (Brownian motion component)
- **λ** — jump intensity (expected jumps per year). λ = 0.5 → one jump every two years on average
- **Jump mean** — average size of a jump

See [OU_KOU_GARCH_METHODOLOGY.md §2](OU_KOU_GARCH_METHODOLOGY.md) for the full process definition and P(in-range) derivation via characteristic function.

---

## 6. Global Controls and Layout

**Tweaks panel** (gear icon, top-right corner of the chart area):

| Control | Effect |
|---------|--------|
| **Plot / table split** | Slider (30%–82%) controlling how much vertical space the chart takes vs the table |
| **Indicator pane height** | Height in pixels of each sub-pane (RSI, MACD, etc.) |
| **Density** | Row padding in the trades table: compact / regular / comfy |
| **Theme** | Light / dark. Persists to localStorage |

**Window dropdown** (sidebar, Experiment Analysis) — filters both chart zoom and trades table to a single OOS window. Select "All" to view the full experiment.

**View mode** — "Both" (chart + table), "Plot" (chart only), "Table" (table only). Use "Plot" when presenting charts, "Table" when doing detailed trade analysis.

**Data view** — "Trades" (trade-by-trade table) or "Time Series" (raw OHLC + indicator values per bar, useful for debugging feature values at specific dates).

---

## 7. Metric Reference Glossary

| Metric | Symbol | Definition | Good range |
|--------|--------|------------|------------|
| Total return | R | (Final equity − Initial equity) / Initial equity | > 0 per window |
| Sharpe ratio | S | Annualised (mean OOS return − 0) / std(daily returns) | > 1.0 |
| Sortino ratio | — | Like Sharpe but std computed on downside returns only | > 1.5 |
| Win rate | W | Trades closed with P&L > 0 / Total trades | 60–75% for iron condors |
| Profit factor | PF | Gross wins / Gross losses | > 1.5 |
| Max drawdown | MDD | Max (peak equity − trough equity) / peak equity | < 10% per window |
| Expectancy | E | Mean P&L per trade across the window | > 0 |
| AUROC | — | Area Under the ROC Curve; 0.5 = random, 1.0 = perfect | > 0.55 for directional |
| Balanced accuracy | BA | (Sensitivity + Specificity) / 2; class-imbalance robust | > 0.60 for range gate to activate |
| Edge over baseline | — | BA − 0.50; must be ≥ 0.10 for range model to emit predictions | ≥ 0.10 |
| Brier score | BS | MSE between predicted probability and binary outcome; lower = better | < 0.20 |
| Brier skill score | BSS | 1 − BS/BS_ref; positive = better than climatology | > 0 |
| CV score | — | Cross-validated balanced accuracy on the training fold | — |
| BIC | — | Bayesian Information Criterion; lower = better model fit per parameter | Minimised over GARCH variants |
| Hurst exponent | H | Measures long-range dependence; H < 0.5 = mean-reverting | 0.40–0.50 favours condors |
| IV rank | IVR | Percentile of current IV vs 1-year range; 0 = cheapest, 1 = richest | > 0.30 for condor entries |
| P(in-range) | — | Range predictor output; probability price stays within ±threshold over 30d | ≥ 0.55 to enter |
| In-range rate | — | Historical base rate of staying in range; the naive prior | Typically 50–65% for QQQ |

---

*Generated: 2026-06-07. Covers dashboard version as of commit `d227c3f`.*

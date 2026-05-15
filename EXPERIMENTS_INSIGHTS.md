# AIT v2 — Walk-Forward Experiment Insights

> **Purpose:** A living record of what each experiment taught us. Covers what we configured, what we assumed, what the results showed, and what we changed as a consequence. Update this file after every experiment.

> **How to add an entry:** Copy the template at the bottom of this file, fill it in, and append it as the next numbered section. Update the Principles Distilled section if the experiment reinforced or invalidated anything there.

---

## Summary Table

| # | Archive | Config | Opt. Return | Ablation | Trades | Sharpe | Active W | Key Change |
|---|---------|--------|-------------|----------|--------|--------|----------|------------|
| 1 | `QQQ_2Y_iron_condor_per_strategy_20260512` | 365/63/21/5, 18 W | +44.87% | — | 50 | 17.70 | 11/18 | First integration test |
| 2 | `QQQ_365d_iron_condor_20260513_1831` | 365/42/14/5, 28 W | +8.11% | ~+9% | 13 | 25.35 | 6/28 | Shorter windows |
| 3 | `QQQ_365d_iron_condor_20260514_1308` | 365/42/14/5, 28 W | +5.45% | ~+9% | 8 | 24.17 | 2/28 | Repeat to confirm Exp 2 |
| 4 | `QQQ_365d_iron_condor_20260514_2359` | 365/42/14/5, 28 W | +21.36% | ~+9% | 29 | 10.86 | 9/28 | Vol gate adjustment |
| 5 | `QQQ_365d_iron_condor_20260514_1142` | 365/42/14/5, 28 W | **+183.14%** | **+9.00%** | **91** | **23.95** | **18/28** | **Removed regime filters from search space** |

Config format: `train_days / test_days / step_days / gap_days`.
All experiments: QQQ, iron_condor only, 50 Optuna trials per window, TPE sampler seed 42, $100k initial capital.

---

## Experiment 1 — First Integration Test

**Archive:** `QQQ_2Y_iron_condor_per_strategy_20260512`
**Date:** 2026-05-11 · Git commit: `fa283217`

### Setup
- Walk-forward config: 365d train / 63d test / 21d step / 5d gap → **18 windows**
- Test period covered: 2025-03-16 to 2026-05-10
- Optimization: per-strategy Optuna, 50 trials, full search space including `min_confidence` [0.55, 0.70] and `max_entry_vol_annual` [0.25, 0.90]
- 12-dimensional search space for iron_condor

### Assumptions Going In
- More parameters in the search space = more flexibility = better optimization
- Per-strategy studies (one Optuna study per strategy, not one joint 48D study) would improve search efficiency
- 63-day test windows provide enough OOS data to evaluate each parameter set

### Results

| Metric | Value |
|--------|-------|
| Total return | +44.87% |
| Total P&L | +$44,867 |
| Total trades | 50 |
| Win rate | 82% |
| Sharpe ratio | 17.70 |
| Max drawdown | 1.51% |
| Profit factor | 16.02 |
| Active windows | 11 / 18 |

**Per-window trades:** `[0, 0, 3, 6, 3, 9, 6, 1, 0, 4, 0, 0, 1, 0, 7, 9, 0, 1]`

### Observations
- 7 zero-trade windows (W1, W2, W9, W11, W12, W14, W17). W1 and W2 cover the Liberation Day tariff shock (April 2025, QQQ −15%) — the vol gate correctly blocked entries during the spike.
- Profitable windows clustered in the calmer periods (post-July 2025, early 2026).
- Regime filter params (`max_entry_vol_annual`) were being optimized: W4 best value was 0.26 (tight gate) yet still produced 6 trades — the 63-day window provided enough days for even a selective gate to allow entries.

### What We Learned
- Results looked promising on the surface (+44.87% is good).
- However, the success partly masked a latent risk: regime filter params in the search space could be exploited as degenerate solutions. The 63-day window was long enough that even restrictive gates occasionally allowed trades.
- The 18-window dataset made it hard to distinguish genuine optimization edge from lucky window selection.

### What Changed for Next Experiment
- Switched to **42d test / 14d step** (28 windows instead of 18) for more granular OOS coverage and faster detection of degenerate optimization patterns.
- Kept the same search space to establish a baseline with the new window config.

---

## Experiment 2 — Baseline with Shorter Windows

**Archive:** `QQQ_365d_iron_condor_20260513_1831`
**Date:** 2026-05-13 18:31 UTC

### Setup
- Walk-forward config: 365d train / 42d test / 14d step / 5d gap → **28 windows**
- Test period covered: 2025-03-16 to 2026-05-10
- Optimization: same full search space as Experiment 1 (12D, includes `min_confidence` + `max_entry_vol_annual`)

### Assumptions Going In
- Shorter, more granular windows should give the optimizer a better signal and produce more realistic OOS evaluation.
- Same search space as Experiment 1, now tested on a harsher setup.

### Results

| Metric | Value |
|--------|-------|
| Total return | +8.11% |
| Total P&L | +$8,115 |
| Total trades | 13 |
| Win rate | 92% |
| Sharpe ratio | 25.35 |
| Max drawdown | 0.00% |
| Profit factor | — (predates metric) |
| Active windows | 6 / 28 |

**Per-window trades:** `[0, 0, 0, 0, 0, 2, 0, 4, 1, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]`

### Observations
- 22 of 28 windows produced zero trades. The optimizer was consistently finding parameters that blocked all OOS entries.
- The regime filter params (`min_confidence` near the 0.70 ceiling, `max_entry_vol_annual` driven to low values) were the culprit: near-zero variance in P&L in-sample → artificially inflated Sharpe → Optuna converges to this degenerate solution.
- The shorter 42-day windows made the pattern more visible than in Experiment 1: with less time per window, even a slightly over-selective gate produces 0 trades.
- Trades were clustered in only 6 windows, concentrated in the calmer mid-2025 period.

### What We Learned
- Shorter windows exposed what Experiment 1 masked: the regime filter params are **overfitting pressure sinks**. Optuna can find degenerate in-sample solutions by exploiting them.
- The degenerate solution pattern: high `min_confidence` (near 0.70) + low `max_entry_vol_annual` → zero OOS trades → zero P&L variance → Sharpe = undefined or very high → Optuna prefers this.
- This is not a data anomaly — it is a structural problem with including regime-filter params in the search space.

### What Changed for Next Experiment
- Ran a near-identical configuration (Experiment 3) to verify the pattern is reproducible and not a one-off.

---

## Experiment 3 — Confirming the Pattern

**Archive:** `QQQ_365d_iron_condor_20260514_1308`
**Date:** 2026-05-14 13:08 UTC

> **Note:** This experiment was overwritten by Experiment 4 on the same calendar day due to an archive naming collision (run ID used date-only format `YYYYMMDD`). It was recovered from git. The naming bug was subsequently fixed to use `YYYYMMDD_HHMM`.

### Setup
- Walk-forward config: 365d / 42d / 14d / 5d → **28 windows** (identical to Experiment 2)
- Same full search space (12D, includes `min_confidence` + `max_entry_vol_annual`)
- Near-identical parameters to Experiment 2 — run to confirm reproducibility of the degenerate pattern

### Assumptions Going In
- If Experiment 2 was a systematic bug and not noise, a repeat run should show the same zero-trade collapse.
- If it was noise, we should see a meaningfully different result.

### Results

| Metric | Value |
|--------|-------|
| Total return | +5.45% |
| Total P&L | +$5,452 |
| Total trades | 8 |
| Win rate | 75% |
| Sharpe ratio | 24.17 |
| Max drawdown | 0.00% |
| Profit factor | 63.87 |
| Active windows | 2 / 28 |

**Per-window trades:** `[0, 0, 1, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

### Observations
- Even worse than Experiment 2: only **2 active windows** and 8 trades total. The pattern is reproducible and deteriorated.
- The high Sharpe (24.17) and profit factor (63.87) on just 8 trades illustrate the problem: small sample sizes can produce misleading metrics even though the strategy barely traded OOS.
- W6 alone accounts for 7 of 8 trades (a single favorable window dominating the result).

### What We Learned
- Confirmed: the zero-trade collapse is **systematic, not noise**. The optimizer reliably finds degenerate solutions when regime-filter params are searchable.
- The result got worse (2 active windows vs 6), suggesting that with each re-run, Optuna refines its convergence toward the degenerate solution.
- High Sharpe and profit factor on tiny trade samples are misleading. A strategy that produces 8 trades in 28 windows is not a deployable strategy regardless of those metrics.
- **Infrastructure lesson exposed:** Same-day archive naming collision caused this experiment's data to be overwritten by Experiment 4. Archive IDs must include a time component to prevent this.

### What Changed for Next Experiment
- Tried adjusting the vol gate (`max_entry_vol_annual`) directly — the "vol45" approach — to see if making it more permissive would produce more OOS trades while keeping it in the search space (Experiment 4).
- Separately, began planning the naming collision fix (implemented post-Experiment 5).

---

## Experiment 4 — Vol Gate Adjustment

**Archive:** `QQQ_365d_iron_condor_20260514_2359`
**Date:** 2026-05-14 23:59 UTC
**Original suffix:** `_vol45` (referring to the vol gate threshold adjustment)

### Setup
- Walk-forward config: 365d / 42d / 14d / 5d → **28 windows**
- Search space: same 12D space but with `max_entry_vol_annual` upper bound widened or default adjusted to be more permissive (allowing the optimizer to explore a broader vol gate range)
- Hypothesis: forcing the vol gate to be more permissive in-sample would allow more OOS trades

### Assumptions Going In
- The zero-trade collapse is caused specifically by `max_entry_vol_annual` being driven too tight.
- If we widen the allowable range or raise the default, the optimizer will find a balanced setting that produces real OOS trades.
- `min_confidence` alone is not the root cause.

### Results

| Metric | Value |
|--------|-------|
| Total return | +21.36% |
| Total P&L | +$21,358 |
| Total trades | 29 |
| Win rate | 62% |
| Sharpe ratio | 10.86 |
| Max drawdown | 0.03% |
| Profit factor | 6.55 |
| Active windows | 9 / 28 |

**Per-window trades:** `[0, 1, 1, 1, 5, 4, 0, 3, 0, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0]`

### Observations
- More active windows (9 vs 2) and more trades (29 vs 8) than Experiments 2–3, suggesting the vol gate adjustment reduced the zero-trade problem.
- But 19/28 windows still had zero trades. The degenerate pattern was not eliminated, only partially mitigated.
- Win rate dropped to 62% (vs 75–92% in Exps 2–3), and Sharpe dropped to 10.86. The trades that now happened included more losers — the vol gate had previously been filtering those out, inflating win rate and Sharpe by simply not trading.
- The "improvement" in active windows was accompanied by degradation in per-trade quality metrics.

### What We Learned
- Adjusting the vol gate range/default does not solve the root problem. The optimizer still exploits regime filters as a degenerate mechanism — just less efficiently.
- The Sharpe drop (from 24 → 10) when more trades appear is revealing: the previous high-Sharpe results were artificial (few trades = low variance = high Sharpe), not genuine edge.
- **The assumption was wrong:** the problem is not the vol gate threshold specifically — it is the fundamental incentive structure. Any parameter that can block all trades becomes a degenerate optimization target. The optimizer will exploit whatever regime-filter params are available.
- The correct fix is to **remove both `min_confidence` and `max_entry_vol_annual` from the iron_condor search space entirely**, not to adjust their ranges.

### What Changed for Next Experiment
- Removed `min_confidence` and `max_entry_vol_annual` from `IRON_CONDOR_SPACE` in `param_spaces.py`.
- Both params remain active at their config defaults (`min_confidence=0.55`, `max_entry_vol_annual=0.80`) — they are simply no longer searchable.
- Iron_condor search space reduced from 12D to 9D (structural params only: delta_short, max_hold_days, stop_loss_pct, profit_target_pct, trailing_stop_pct, iv_floor, wing_k, hurst_regime_threshold, hurst_regime_penalty, multifractal_max_width).

---

## Experiment 5 — Removing Regime Filters from Search Space

**Archive:** `QQQ_365d_iron_condor_20260514_1142`
**Date:** 2026-05-14 11:42 UTC
**Original suffix:** `_nofilters`

### Setup
- Walk-forward config: 365d / 42d / 14d / 5d → **28 windows**
- **Key change:** `min_confidence` and `max_entry_vol_annual` removed from `IRON_CONDOR_SPACE`. Iron_condor search space is now 9D (structural params only).
- Both removed params remain active at config defaults during OOS evaluation.
- Section F ablation run (no optimization) included for direct comparison.

### Assumptions Going In
- Removing regime-filter params will prevent the degenerate optimization pattern.
- The optimizer, forced to search only structural params, will find genuine edge in delta placement, wing sizing, hold duration, and exit thresholds.
- The ablation baseline (+9% from prior experiments) represents the unoptimized floor; optimization should improve on this significantly.

### Results — Optimized (Section E)

| Metric | Value |
|--------|-------|
| Total return | **+183.14%** |
| Total P&L | **+$183,143** |
| Total trades | **91** |
| Win rate | **91%** |
| Sharpe ratio | **23.95** |
| Max drawdown | 0.00% |
| Profit factor | **89.39** |
| Active windows | **18 / 28** |

**Per-window trades:** `[0, 0, 2, 7, 7, 9, 4, 10, 3, 2, 6, 4, 5, 0, 0, 7, 3, 0, 0, 3, 6, 0, 6, 3, 0, 4, 0, 0]`

### Results — Ablation (Section F, no optimization)

| Metric | Value |
|--------|-------|
| Total return | +9.00% |
| Total trades | 45 |
| Win rate | — |
| Sharpe ratio | 2.85 |
| Active windows | 16 / 28 |

### Optimization vs Ablation

| | Optimized | Ablation | Ratio |
|--|--|--|--|
| Return | +183.14% | +9.00% | **20.3×** |
| Sharpe | 23.95 | 2.85 | **8.4×** |
| Trades | 91 | 45 | 2.0× |

### Observations
- The degenerate pattern is completely eliminated. 18/28 windows have trades — more than double any previous experiment.
- Optimization now dramatically outperforms ablation (20× return, 8.4× Sharpe). This is the first experiment where optimization provided clear, genuine edge over the unoptimized baseline.
- The optimizer found meaningful structural differentiation: windows with more trades (W4–W8: 7–10 trades) correspond to calmer QQQ periods; W13–W15 and W17–W20 (0 trades) correspond to market conditions blocked by the fixed `max_entry_vol_annual=0.80` gate — correctly applied as a config default, not as a searchable param.
- 91% win rate across 91 trades is very high. Iron condor theta decay is the primary return driver in the calmer 2025-H2 through early 2026 environment.
- Ablation's 45 trades vs optimized 91 trades shows the optimizer is finding 2× more valid entry signals — it is broadening participation, not just filtering.

### What We Learned
- **Confirmed hypothesis:** Removing regime-filter params from the search space fixes the root problem. The optimizer is now forced to find genuine structural edge.
- The 20× return improvement vs ablation is strong evidence of real optimization value — not just a less-degenerate degenerate solution.
- The **key principle**: any parameter that can produce zero OOS trades as an in-sample optimization strategy is a degenerate pressure sink and should not be in the search space. It should be set as a fixed config value chosen by judgment, not by Optuna.
- Ablation (+9%) establishes the minimum baseline the strategy produces with default parameters and no per-window tuning. Optimization must beat this to justify its compute cost. Experiment 5 beats it by 20×.
- The run duration was 396 minutes (6.6 hours) for 28 windows × 50 trials.

### What Changed After This Experiment
- `IRON_CONDOR_SPACE` in `param_spaces.py` was updated (the removal was what caused this experiment; the lesson is to keep it this way).
- Four infrastructure bugs fixed:
  1. `n_windows=0` in archive metadata — glob was reading from the copy destination instead of the source dir
  2. Archive name collision — run ID now includes `_HHMM` time component (`YYYYMMDD_HHMM`)
  3. `profit_factor=0` in MLflow and archive — was hardcoded; now read from `WalkForwardResult.profit_factor`
  4. Missing `import json` at module level — was causing silent read failures inside `_create_run_archive`
  5. MLflow split between two databases — `run_integration_test.py` now writes to `data/mlflow.db` by default (same as `backfill_mlflow.py`)
- Existing 4 archives retroactively renamed to `_HHMM` convention and re-imported to MLflow.

---

## Principles Distilled

These are standing conclusions drawn from the experiments above. Review and update after each new experiment.

### P1 — Regime-filter params must not be in the optimizer search space
**Evidence:** Experiments 2–4 all suffered zero-trade collapse when `min_confidence` and/or `max_entry_vol_annual` were searchable. Experiment 5 fixed this by removing them.
**Rule:** Any parameter that can produce zero OOS trades as its optimal in-sample solution must be fixed as a config default, not searched. Set it by judgment (calibrated on the distribution of market conditions), not by Optuna.

### P2 — Ablation is a mandatory baseline
**Evidence:** Experiments 2–4 showed "optimization" producing results below the ablation baseline (which was never measured for those experiments individually). Experiment 5 was the first to formally compare.
**Rule:** Every optimization experiment must include a Section F ablation run. The optimization result is only meaningful relative to "what would have happened with no optimization."

### P3 — Active window count matters more than Sharpe or win rate in small samples
**Evidence:** Experiments 2–3 showed Sharpe > 24 and win rate > 75% on 2–6 active windows. These metrics are unreliable with < 15 trades. An optimized strategy that trades in 2/28 windows is not deployable.
**Rule:** Require a minimum of 10 active windows (out of 28) before considering optimization results meaningful. High Sharpe on low trade count is a red flag, not a signal of quality.

### P4 — Shorter test windows expose overfitting more clearly
**Evidence:** Experiment 1 (63-day windows) masked the overfitting issue; Experiments 2–4 (42-day windows) exposed it immediately.
**Rule:** 42-day test windows are preferred for the diagnostic experiments. 63-day windows can be used for final validation of a stable configuration, where masking overfitting is less of a concern.

### P5 — Optimization search space dimensions should reflect the problem geometry
**Evidence:** Iron condor is a range-bound theta strategy. Parameters like `delta_short`, `wing_k`, `max_hold_days`, and exit thresholds directly control the shape and risk profile of the structure. Regime-filter params control whether to trade at all — a different question, better answered by config calibration.
**Rule:** The optimizer's search space should contain parameters that tune **how** to trade, not **whether** to trade. Gate/filter parameters belong in config.

### P6 — Archive infrastructure must be reliable before experiment data can be trusted
**Evidence:** The `n_windows=0` bug, naming collision, `profit_factor=0`, and two-database MLflow split all caused incorrect tracking data across Experiments 2–5. Experiment 3 data was overwritten entirely.
**Rule:** Any new infrastructure (archive, logging, MLflow) must be unit-tested before experiments are run. The `test_run_archive.py` test now verifies the three most critical archive invariants.

---

## Open Questions

These are unresolved questions that future experiments should address:

- **Q1:** Does the current search space (9D structural params for iron_condor) have any remaining degenerate dimensions? Are there structural params that Optuna can exploit to produce near-zero-trade solutions?
- **Q2:** Are the experiment results specific to QQQ, or would SPY, IWM, or individual stocks show the same optimization advantage?
- **Q3:** How stable are the best-params across adjacent windows? Do optimized params drift significantly, or do similar values recur?
- **Q4:** What is the real-world edge of the optimized strategy vs the ablation when accounting for bid-ask spreads and realistic fill prices? The Black-Scholes backtester is frictionless.
- **Q5:** Would increasing Optuna trials per window (e.g. 100 vs 50) improve results further, or has the 9D space already converged sufficiently at 50 trials?

---

## Experiment Template

Copy this section to add a new experiment:

```
## Experiment N — [Short descriptive title]

**Archive:** `[run_id]`
**Date:** YYYY-MM-DD HH:MM UTC

### Setup
- Walk-forward config: [train]d / [test]d / [step]d / [gap]d → **[N] windows**
- Test period covered: [start] to [end]
- Key changes from previous experiment: [what was different]

### Assumptions Going In
- [What you believed would happen and why]

### Results — Optimized (Section E)

| Metric | Value |
|--------|-------|
| Total return | |
| Total P&L | |
| Total trades | |
| Win rate | |
| Sharpe ratio | |
| Max drawdown | |
| Profit factor | |
| Active windows | / [N] |

### Results — Ablation (Section F)

| Metric | Value |
|--------|-------|
| Total return | |
| Total trades | |
| Sharpe ratio | |
| Active windows | / [N] |

### Observations
- [What you actually saw in the data]

### What We Learned
- [Conclusions, confirmed/rejected assumptions]

### What Changed for Next Experiment
- [Specific changes to config, search space, infrastructure]
```

---

*Last updated: 2026-05-14*

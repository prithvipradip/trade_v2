# Research Mode — User Guide

How research mode works, what it contains, and how to run and read a study. Short answers first; the references
(RUNBOOK sections, the config files) are linked where you need the detail.

**Contents:** [1 What it is](#1-what-research-mode-is) · [2 The pieces](#2-the-pieces) · [3 What one run does](#3-what-one-run-does) ·
[4 Configuration](#4-configuration-where-every-value-lives) · [5 The engine](#5-the-engine) · [6 The models](#6-the-models) ·
[7 The learner](#7-the-learner) · [8 The optimizer](#8-the-optimizer) · [9 Studies and run records](#9-studies-and-run-records) ·
[10 Reading results](#10-reading-results) · [11 Research vs live](#11-research-vs-live) · [12 Common tasks](#12-common-tasks) ·
[13 Known limits](#13-known-limits)

---

## 1. What research mode is

**What is it?** An offline simulator of the iron-condor strategy. It replays history, trades it the way the bot would,
and reports what happened. It is how changes are tested before they are promoted to the live bot.

**What does it read?** Historical data in `data/historical.db` (the `research_*` tables: daily prices, daily IV,
5-minute bars, the economic calendar, dividends, option spreads), plus `config.yaml`, `config_research.yaml` and an
optional study file.

**What does it never touch?** The broker, `data/ait_state.db`, `data/ait_analytics.duckdb`, and the live model files in
`models/`. It writes only run records under `reports/runs/`, a research model sandbox under `models/research/`, and (if MLflow is installed) the MLflow database `data/mlflow.db`.
The live bot never loads `config_research.yaml`; a test enforces that.

**Is a research number a forecast?** No. It answers "what would this rule set have done on this history, with these
cost and fill assumptions". See [13 Known limits](#13-known-limits).

---

## 2. The pieces

| Piece | What it is | Where |
|---|---|---|
| `run_backtest.py` | Runs a walk-forward backtest and prints the summary | repo root |
| `run_optimizer.py` | Runs an Optuna search over strategy (and optionally ML) parameters | repo root |
| Study scripts | One-question experiments (gate A/B tests, friction sweeps, parity replays) | `scripts/` |
| Engine | Simulates one symbol over one window: entries, exits, fills, costs | `backtesting/engine.py` |
| Walk-forward | Slides train/test windows, trains models per window, runs the engine out of sample | `backtesting/walkforward.py` |
| Pool | Replays all symbols' trades against one shared account | `backtesting/pool.py` |
| Learner | Adapts size and filters between windows ([section 7](#7-the-learner)) | `backtesting/learner.py` |
| Optimizer | Searches parameters on each training window | `optimization/` |
| Study file | A committed file that changes some settings for one run | `studies/<name>.yaml` |
| Run record | A JSON file saying exactly what produced a number | `reports/runs/<run_id>/` |

---

## 3. What one run does

1. **Load config.** `config.yaml` + `config_research.yaml` (+ the study file, if given), merged and validated.
2. **Load data.** Daily bars and IV for each symbol from the `research_*` tables, from `backtest.history_start_date`.
3. **Cut windows.** Default: train 365 days, a 5-day purge gap, test 63 days, step 63 days (non-overlapping tests).
4. **For each window:**
   1. Train the models on the training slice only: direction ensemble, range model, meta-labeler ([section 6](#6-the-models)).
   2. Optionally run an Optuna search on the training slice ([section 8](#8-the-optimizer)).
   3. Run the engine out of sample on the test slice, one symbol at a time.
   4. Replay all symbols' candidate trades against one shared account (the pool), applying live's per-trade budget,
      aggregate-risk and open-position limits.
   5. Let the learner adapt for the next window ([section 7](#7-the-learner)).
5. **Report.** Totals, per-window rows, the entry funnel, a buy-and-hold benchmark, the parity manifest, and a run record.

Nothing from a test slice ever reaches the training of its own window.

---

## 4. Configuration: where every value lives

**How many layers are there?** Three, merged in this order (later wins):

1. `config.yaml`: the live configuration. Research reads it too, so a live change flows into research.
2. `config_research.yaml`: **research's own file.** It does three jobs:
   - **Diff:** a key that also exists in `config.yaml` appears here only when research deliberately differs from live.
   - **Overlay:** it is applied on top of `config.yaml` by research entry points only.
   - **Tunables:** keys that exist only in research (learner thresholds, objective weights, engine knobs, ...) are listed
     here in full, with comments.
3. `studies/<name>.yaml`: changes for one run only ([section 9](#9-studies-and-run-records)).

**How do I see the value a run will use?** `python scripts/show_research_config.py` prints every setting with the layer that
decides it. `--section backtest` narrows it, `--differs` hides values still at their schema default, `--study-file` adds a study.

**Are some values not configurable?** Yes. Clamps and numerical tolerances (the 0.05–0.45 delta band, the IV-scale clamp,
settlement and probability tolerances) are fixed in `config/invariants_research.py`. They are logged at the start of every research
run and stored in every run record. See RUNBOOK "Research fixed limits".

**What about values the environment overrides?** A few trading values (`backtest.wing_k`, `backtest.ic_*`, `risk.iv_rank_cap`, ...)
are read through the live "env contract", which ignores both overlays. A study may not set them; use the environment variable
or the command-line flag.

**Where are the research-only blocks?** In `config_research.yaml` under `backtest:`:

| Block | What it holds |
|---|---|
| `learner` | Thresholds of the research learner |
| `legacy_engine` | Constants of the engine's dormant paths |
| `objectives` | Weights of the Optuna objectives |
| `surface_tuning` | The acceptance rules of the research IV surface (min volume 10, min price 0.05, |moneyness| 0.10, 5 points, 1% span, IV bounds, clamp margin, 5 stale days); part of the surface cache key, so a changed rule rebuilds it (W90f) |
| `friction_fit` | The minimum data a leg-spread (friction) model needs to be fitted: samples, sessions for cross-validation, training rows, affine points. Offline calibration only (W90f) |
| `spread_tuning` | Sizing numbers of the non-condor builders (put / call credit spreads, iron butterfly, jade lizard, debit structures): max cost as a share of capital (25%) and the wing as a fraction of the expected move (0.5). The iron-condor builder does not read them (W90f) |
| `engine_tuning` | The engine's IV-skew coefficients (`put_skew_coefficient` 0.10, `call_skew_coefficient` 0.02), the look-backs of the realized-vol and IV-rank-rise entry gates, the minimum history before the first decision (W90c), and the fallback volatility used when a position has no entry IV (`fallback_iv`, W90e); the window of the drawdown gate's rolling high (`drawdown_high_lookback_rows`, 60) and the replay's history pad (`replay_lookback_days`, 60) (W95) |
| `walkforward_tuning` | Window lengths (`train_days` / `test_days` / `step_days` / `gap_days`), Optuna trial counts and seed, row floors, context and fetch paddings, out-of-sample diagnostic bins and minimums, the minimum number of windows (`min_windows`) and the window-score shrink (W90e); the test-window feature context (`oos_feature_context_bars`) and three row minimums (`meta_label_min_window_rows`, `min_fetched_rows`, `min_backtest_rows`) (W95); also warm start, adaptive range threshold, validation split, inert-parameter probe. `--train-days`, `--test-days`, `--step-days`, `--gap-days`, `--optimize-n-trials` and `--n-trials` override them for one run (W90b) |

The neutral value a missing meta-label input takes (`ml.meta_label_neutrals`: RSI 50, volatility 0.20, ...) and the meta-labeler's
arm-guard limits (`meta_label.min_feature_coverage`, `meta_label.max_artifact_age_days`) are also shared with live (W90c), so they are in
`config.yaml` too. So are the live models' label and validation constants (the range / vol-magnitude bands, the direction label, the walk-forward
folds, the class-balance guards and the retrain's trials, seed and rollback margin: W90d); RUNBOOK "Model labels, validation folds and retrain constants".

Model hyperparameters (`ml.hyperparams`, `ml.meta_label_hyperparams`) are shared with live, so they are listed in
`config.yaml`; research differs only if you override them. Details: RUNBOOK "Research tunables" and "Model hyperparameters".

### Where the history and row-count minimums are set

Until W95 six research minimums sat in the code as bare numbers (60, 252, 100). They are settings now, in `config_research.yaml`,
and the defaults are the old numbers, so no result moved.

| Setting (under `backtest:`) | Default | What it controls |
|---|---|---|
| `engine_tuning.drawdown_high_lookback_rows` | 60 | Rows of closes in the rolling high the drawdown entry gate measures the fall from. The result keeps the name `pct_from_60d_high`. |
| `engine_tuning.replay_lookback_days` | 60 | Calendar days of history the single-trade replay gives the engine before the entry date. 60 calendar days is about 42 trading rows, so a replayed engine's 60-row gates run on a shorter history than in a full run. |
| `walkforward_tuning.oos_feature_context_bars` | 252 | Bars of the training tail put in front of each test window so the features have their look-back. A shorter context does not fail: it changes the feature values (`iv_rank` below 252 bars, `distance_sma200` below 200), so the model would be scored on inputs it was not trained on. A value below `ml.features.vol_regime.iv_rank_lookback` is refused when the walk-forward starts. Not `backtest.context_bars`, which is the engine's intraday look-back. |
| `walkforward_tuning.meta_label_min_window_rows` | 60 | The meta-labeler is trained for a window only when the training window has more than this many bars. A guard on length, not a context. |
| `walkforward_tuning.min_fetched_rows` | 100 | A symbol is used only when more than this many daily bars were fetched. |
| `walkforward_tuning.min_backtest_rows` | 60 | The optimizer's minimum daily bars: a symbol with fewer is dropped and a trial on fewer raises "insufficient data". With a validation split on, the split needs at least twice this many bars and a validation slice shorter than half of it is dropped. |

The live regime classifier has its own three (`regime.min_history_rows`, `iv_rank_min_rows`, `iv_rank_window_rows`); see "The regime classifier".

### Numbers that are not settings: the literal census

**What is it?** `scripts/literal_census.py` walks the research code (`backtesting/`, `optimization/`, `ml/`, `run_backtest.py`,
`run_optimizer.py`: 37 modules) and lists, per function, the numeric literals written inside it. `tests/data/literal_allowlist.json`
is the accepted list, and `tests/test_w88_literal_census.py` fails when a number appears, changes or disappears without that list being
updated, so a research number cannot be added or removed unseen. It ignores only 0, 1, 2 and -1 (the `TRIVIAL` set in the script), booleans, `round()`
digits, log / `print` arguments, and docstrings and comments. It used to ignore more: 3, 4, 5, 7, 10, 30, 0.5 and 0.25 until W90e, and the unit-like
12, 24, 60, 100, 252, 365 and 1000 until W95. Both lists hid real windows and thresholds (a 5-day lookback, a 60-day high, a 252-bar context, a 100-row
minimum); those are counted now, so a unit conversion (`x 100` shares per contract, `/ 365` days to years, `sqrt(252)`) is listed too and is classified
`structural`.

**Categories.** Every function in the list has one:

| Category | Meaning | Real example | Why it is not a setting |
|---|---|---|---|
| `structural` | a protocol or solver constant | `pricing.py::find_strike_by_delta`: bounds 0.8 / 1.2, 50 iterations | it brackets a root search; it does not change what the model says |
| `fixed_feature_group` | a feature group W88d left fixed on purpose | `ml/features.py::FeatureEngine._mfdfa_features`: scales 8 to 128 | its windows are not in `ml.features`, so not in a model's feature-spec hash; it changes only with a code change and a retrain |
| `epsilon` | a numerical tolerance | `engine.py::Backtester._remark_t`: 0.0001-year floor on time to expiry | it keeps Black-Scholes finite at expiry and has no economic meaning |
| `dormant_path` | legacy code off the iron-condor path | `engine.py::Backtester._quick_rsi`: period 14, neutral 50 | nothing on the iron-condor path reads it (the tunable legacy numbers are in `backtest.legacy_engine`) |
| `display` | formatting only | `run_record.py::flatten`: 400-character cut of an MLflow parameter | it changes how a run is recorded, never the result |
| `registered_followup` | transitional: a research number still in code, to be converted | `ml/features.py::FeatureEngine._add_cross_asset`: the 60-day return and VIX z-score windows | it should be a setting; it is shared with live, so it goes in `ml.features` and changes the feature-spec hash (saved models retrain): planned as W97 |

**Reading an entry.**

```json
"src/ait/backtesting/pricing.py::find_strike_by_delta": {"values": [0.8, 1.2, 50], "category": "structural",
                                                         "note": "bisection bounds and iterations of the strike search"}
```

The key is `file::function` (`Class.method`, or `<module>` for module-level code). `values` is the sorted list of that function's
non-trivial literals, repeats included. Then the category and a one-line note saying what the numbers are.

**You added a number to research code.** The census test fails and names the function. Then either:

1. Make it a setting (preferred). A research-only value goes in the schema and is listed in full in `config_research.yaml`; a value shared
   with live goes in the schema and `config.yaml`. Read it from the loaded settings, then run `python scripts/literal_census.py --update` to drop the entry.
2. If it truly is not a tunable, run `python scripts/literal_census.py --update`, then give the new entry a real category and a note saying
   why. `--update` guesses obvious cases (display helpers, `ml/features.py`) and otherwise adds the entry as `registered_followup` with a
   generated placeholder note; the test rejects that placeholder, so replace both.

**How a new setting is exposed.** Add the field to its block's model in `config/settings.py` and list it in the matching block of
`config_research.yaml`, as in the research-only blocks table above; completeness tests fail if the file misses a field, and
`scripts/show_research_config.py` shows the value and which layer set it.

---

## 5. The engine

**Which engine runs?** `backtest.engine_model: realistic` (set in `config_research.yaml`). It prices and fills at live fidelity:
fill-time spot, previous-session IV, listed session expiries, same-session exits, touch stops, and expiry settlement or assignment.
The older `legacy` engine is kept byte-identical for reproducing old numbers.

**How are options priced?** Black-Scholes from one stored IV per day plus a skew (`pricing_source: bs_legacy`). A fitted IV
surface from real option bars is available (`pricing_source: surface`, QQQ only).

**What does a trade cost?** Commission plus fees per leg, entry and exit, and a fitted half-spread per leg
(`friction_model: per_leg_usd`, with short-DTE bands). Debit structures also pay `backtest.slippage_pct` (iron condors do not).

**What decides whether an entry happens?** A chain of gates, in order. A day that fails one is counted in the **entry funnel**
under that gate's name:

| Funnel name | Meaning |
|---|---|
| `no_strategy` | The selector found no allowed structure |
| `iv_rank_floor` / `iv_rank_cap` / `iv_rank_unknown` | IV rank below the condor floor, above the credit cap, or not measurable (fails closed) |
| `range_gate_below_threshold` | The range model's P(stay in range) is under `ml.range_min_confidence` (only when `gates_mode: enforce`) |
| `macro_event` | An FOMC/CPI/NFP/PCE/GDP event is within `risk.pre_event_blackout_days` |
| `ex_div_entry` | A short call could be assigned early before an ex-dividend date (live's rule) |
| `max_concurrent` | `backtest.max_concurrent_positions` (3) already open |
| `builder:*` | The condor builder refused (credit floor, credit-to-width ratio, affordability, ...) |
| `vol_gate`, `regime_veto`, `iv_rank_rise` | Research-only vetoes (see [section 11](#11-research-vs-live)) |

**What exits a trade?** Take-profit on a DTE ladder (50/40/30/20% of credit), a touch stop, a close at 5 DTE or less, and a hold limit.
The same ladder constants live and research share (`execution/exit_policy.py`).

---

## 6. The models

**Which models are trained per window?** Three, on that window's training slice only:
- **Direction ensemble** (XGBoost + LightGBM): bearish / neutral / bullish.
- **Range model:** P(price stays within ±X% over the horizon). It decides the range gate. One model per horizon bucket when the entry DTE is searched.
- **Meta-labeler:** a second model that vetoes trades the first two like but historically lose.

The vol-magnitude model is live only; research does not train it.

**Do the models gate entries in research?** `gates_mode` decides. `enforce` (research's setting): the range floor blocks entries.
`observe` (live's setting today): the models predict and log but never block. This is a deliberate research-vs-live difference;
the parity manifest shows it.

**Can I change their hyperparameters?** Yes: `ml.hyperparams` (direction, range) and `ml.meta_label_hyperparams`. Every saved model
records the values it was trained with, and loading a model trained under other values logs `model_spec_mismatch`.

**Can I change the indicator windows?** Yes: `ml.features` (RSI, MACD, SMAs, realized vol, ...). The thirteen columns whose name embeds a window are renamed
(`rsi_14` → `rsi_21`), every saved model records a hash of the windows, and a model trained under other windows is refused and retrained. RUNBOOK "Feature windows" lists what is in
scope and what stays fixed.

**Could a research run overwrite the live model?** No. Window models are throw-away (`persist_artifacts=False`) or go to `models/research/`.

### Drift detection (live; off by default)

**What is it?** The live bot compares the direction model's calls with what happened. If the model is clearly getting worse it asks for an
early retrain instead of waiting for the weekly one. Research does not use it. The settings are `ml.drift_*` in `config.yaml`.

**Is it on?** No: `ml.drift_detection_enabled: false`. A prediction is recorded on every scan, but an outcome is recorded only when a
directional trade closes (iron condors, short strangles and long straddles are skipped). An iron-condor-only book therefore never completes a
sample and the detector could never speak, so it is off. Turn it on (`true`, then restart) only if directional strategies trade.

| Setting | Default | Meaning |
|---|---|---|
| `drift_detection_enabled` | `false` | Master switch. Off = records nothing and always reports "no drift". |
| `drift_window_size` | 50 | N: how many of the most recent resolved predictions (trade closed) it looks at. Larger = steadier but slower to notice. |
| `drift_accuracy_threshold` | 0.40 | Drift if the share of those that were right falls below this. Guessing a 3-way direction is right about 33% of the time. |
| `drift_high_confidence` | 0.80 | The "high-confidence group": predictions made with at least this confidence. They should be right about this often. |
| `drift_calibration_threshold` | 0.25 | Drift if that group is right LESS often than it claims by more than this (25 points: claimed 80%, right under 55%). |
| `drift_min_samples` | 20 | No report until this many predictions have an outcome. Must not exceed the window. |
| `drift_min_high_confidence_samples` | 5 | The calibration test needs this many resolved high-confidence predictions; with fewer only accuracy counts. |
| `drift_check_interval` | 10 | The full check runs at most once per this many newly recorded predictions (1 = every call). Until W92 this was documented but not implemented. |

**What happens on drift?** The report says `should_retrain`, the trainer's `needs_training` returns true and the next opportunity retrains; the post-market
job also logs `post_market_drift`. A retrain clears the window and the throttle.

### The regime classifier (live)

**What is it?** From SPY's last `min_history_rows` (60) or more days plus VIX it labels the market as one of seven regimes (high volatility, IV crush, mean-reverting,
low volatility, trending up, trending down, range-bound) and gives a confidence. Research does not use it (the engine has its own regime thresholds,
`backtest.regime_*`). The settings are the `regime:` block of `config.yaml`.

**How is trend measured?** The trend strength (-1 to +1) is the average of three kinds of terms, each clamped to [-1, 1]: the slope of moving averages,
the price's distance from moving averages, and the recent return.

**Where does the live bot use it?**
- **Entry:** a HIGH_VOLATILITY label with confidence at or above `ml.regime_override_confidence` (0.80) forces the direction to NEUTRAL (moot for condors, which ignore the direction).
- **Exit check:** a position entered in a trending regime that now reads as high volatility with confidence above `exit.thesis_regime_shift_confidence` (0.70) is flagged for exit.
- **Record:** the label is stored on each trade as `entry_regime`.

| Setting | Default | Meaning |
|---|---|---|
| `trend_sma_periods` | `[10, 20, 50]` | Moving-average lengths (days) whose slope is scored. |
| `trend_slope_lookback`, `trend_slope_scale` | 5, 100 | A slope is the average's % change over this many days, times this (a 1% move scores 1.0). |
| `trend_deviation_periods`, `trend_deviation_scale` | `[20, 50]`, 10 | Moving-average lengths the price's distance is scored against, times this (10% away scores 1.0). |
| `trend_momentum_days`, `trend_momentum_scale` | 20, 5 | The return over this many days, times this (a 20% return scores 1.0). |
| `high_vol_confidence_floor`, `high_vol_confidence_cap`, `high_vol_confidence_vol_scale` | 0.80, 0.90, 0.50 | HIGH_VOLATILITY reports `max(floor, min(cap, realized vol / scale))`. The only confidence a live rule compares against. |
| `iv_crush_confidence`, `low_vol_confidence` | 0.70, 0.70 | Confidence of those labels (reported only). |
| `mean_revert_confidence_floor` | 0.60 | MEAN_REVERTING reports `max(floor, 1 - Hurst exponent)` (reported only). |
| `trend_confidence_cap` | 0.95 | TRENDING_UP / DOWN report `min(cap, abs(trend strength))` (reported only). |
| `range_confidence_floor` | 0.50 | RANGE_BOUND reports `max(floor, 1 - 2 x abs(trend strength))` (reported only). |
| `min_history_rows` | 60 | Fewer daily bars than this and the classifier makes no call (RANGE_BOUND, confidence 0). It must cover the trend measure's longest look-back (55 at the defaults); a lower value is refused when the settings load. |
| `iv_rank_min_rows`, `iv_rank_window_rows` | 60, 252 | The IV-rank proxy needs at least 60 values of the 20-day realized vol (else it reports a neutral 0.5) and compares today's with the highest and lowest of the last 252. |

**What is the IV rank here?** A price-only proxy: today's 20-day realized volatility as a percentile of its own range over the last
`iv_rank_window_rows` values. It is not the stored-IV rank of `ait.data.iv_rank` and not the `ml.features.vol_regime.iv_rank_lookback`
feature; the three are separate on purpose. Each volatility value needs 20 bars of its own, so at the defaults the proxy stays 0.5 until the history
reaches about 80 bars, although the classifier itself runs from 60 (W95 documented this; the defaults are unchanged).

**Watch the pair.** `high_vol_confidence_floor` and `ml.regime_override_confidence` must agree: keep the override threshold at or below the floor. A day
triggered by VIX alone with realized vol at or under 40% scores exactly the floor, so at 0.80 / 0.80 it reaches the override. (Until W93 the rule was a strict
"above", so that case never fired; PLAN.md W92 recorded it, W93 changed it to "at or above".)

### Saved model versions and the meta-labeler

| Setting | Default | Meaning |
|---|---|---|
| `ml.model_versions_kept` | 5 | Saved direction-model versions kept on disk (older ones deleted on each save); also how far back a rollback can go. |
| `meta_label.min_trades_for_training` | 30 | The meta-labeler (disabled) is not trained on fewer closed trades than this. `min_closes_to_enable` is the separate question of when to trust it. |
| `meta_label.min_class_trades` | 5 | ... and needs at least this many profitable and this many unprofitable trades. |
| `meta_label.cv_max_splits`, `cv_min_rows_per_split` | 5, 10 | Its training cross-validation uses `min(max splits, max(2, rows / rows per split))` folds. |

---

## 7. The learner

**What is it?** `BacktestLearner`, a small in-memory loop in the walk-forward. After each test window it looks at that window's
trades and adapts three things for the next window. It runs in sequential mode only (it is off when `optimize_n_jobs > 1`).

- **Strategy multipliers:** a per-strategy size factor, starting at 1.0. A strategy that keeps losing gets its factor cut. One that keeps winning gets it raised.
- **Confidence floor:** if low-confidence trades keep losing, the minimum entry confidence is raised by 0.05, up to a cap of 0.85.
- **Symbol disabling:** a symbol that keeps losing is switched off for later windows.

**What is a multiplier, and where is it used?** The walk-forward sizes each window's trades at
`position_size_pct × min(average multiplier of the active strategies, 2.0)`. A multiplier of 0.41 means trades are sized at 41% of normal.
In an iron-condor-only run the learner can therefore shrink condor size after losing windows. That is part of what a research number contains.

**What does `reduce_min_change` do?** It is a dead-band. A reduction is applied only if it moves the multiplier by more than 0.03.
This stops tiny adjustments, but it has a side effect at the floor: the multiplier goes 0.41 → 0.33, and the next step would land
on the 0.30 floor, a change of only 0.027, so it is skipped and the multiplier stays at 0.33.

**What do `boost_min_trades` and `reenable_min_trades` do?** They are extra trade-count conditions for boosting a winner (6 trades)
and re-enabling a disabled strategy (10 trades). Only the learner's per-window strategy step reads them, inside a loop that already
skips any strategy with fewer than `min_trades_to_learn` (15) trades. Both are below 15, so they never block anything today.
They only matter if you set them above 15.

**What is "disable a strategy" meant to achieve?** To stop a strategy that has lost across many windows from being traded in the
remaining windows. The sequence is: reduce its size repeatedly, then switch it off; if it later recovers, re-enable it at half size.
Iron condors are exempt from disabling (they are regime-dependent), so for them the end state is just a smaller size.

**Does the disable step ever happen?** Not with the shipped numbers: the multiplier never reaches 0.30 (see `reduce_min_change`), so
strategy disabling is unreachable. Symbol disabling and the confidence raise do work.

**Learner settings** (`backtest.learner`; a test keeps this table equal to the schema defaults):

| Setting | Default | Meaning |
|---|---|---|
| `windows_before_adapting` | 2 | Windows accumulated before any change |
| `min_trades_to_learn` | 15 | Trades before a strategy, symbol or band is judged |
| `min_trades_to_disable` | 25 | Trades before a strategy or symbol is switched off |
| `strategy_multiplier_min` | 0.30 | Floor of the size multiplier |
| `strategy_multiplier_max` | 2.0 | Cap of the size multiplier |
| `reduce_win_rate` | 0.30 | Win rate below which (with a loss) a strategy is reduced |
| `reduce_factor` | 0.80 | Multiplier × this per reduction |
| `reduce_min_change` | 0.03 | Smallest reduction that is applied |
| `boost_win_rate` | 0.58 | Win rate above which (with a profit) a strategy is boosted |
| `boost_factor` | 1.20 | Multiplier × this per boost |
| `boost_min_change` | 0.05 | Smallest boost that is applied |
| `reenable_win_rate` | 0.50 | Win rate above which a disabled strategy returns |
| `reenable_multiplier` | 0.5 | Size it returns at |
| `confidence_cap` | 0.85 | Highest entry confidence the learner may require |
| `confidence_raise_step` | 0.05 | Raise per adaptation |
| `symbol_disable_win_rate` | 0.20 | Win rate below which a symbol is switched off |

---

## 8. The optimizer

**What does it do?** `StrategyOptimizer` searches parameters (wing width, short delta, stops, DTE and so on) with Optuna's TPE sampler
(seed 42). Each trial is one engine run; the best trial's parameters are used for that window's out-of-sample test.

**How is a trial scored?** By the objective: `sharpe_ratio`, `composite`, `profit_factor` or `win_rate`. The composite is
`0.4 × Sharpe + 0.4 × win rate − 0.2 × |max drawdown|` (weights in `backtest.objectives`). Fewer than 3 trades scores on a graded floor, so the
search has something to climb.

**What is the graded floor?** A trial with fewer trades than `backtest.optimize_hard_min_trades` (3) scores
`optimize_below_floor_score + optimize_below_floor_gradient × trades / hard_min`: -100 for zero trades, rising by up to 20 (`optimize_below_floor_gradient`),
so trading always beats not trading and a starved trial always ranks below a real one. The base (`optimize_below_floor_score`, W95) was a literal -100
in the optimizer. Keep it far below any objective's real scores (it is bounded to -1000..-1, and base plus gradient must stay negative). All three
are in `config_research.yaml`.

**What is searched, and can I change it?** The ranges are in `optimization/param_spaces.py`. A study can override one range
(`search_space:` in the study file). It is validated (known dimension, `low < high`, inside the hard limits), hashed, and a stored Optuna
study will not resume over a different space.

**What is warm start?** A window's search starts from the previous window's best parameters if that window's out-of-sample win rate
was at least 0.75 with 3+ trades, else from the best overall (`backtest.walkforward_tuning`).

**Can the entry DTE be searched?** Yes, with `--search-dte` (RUNBOOK "Searching the entry DTE").

---

## 9. Studies and run records

**What is a study?** A committed file `studies/<name>.yaml` that changes settings and/or search ranges for one run:

```yaml
name: w90_wing_k_wider
question: Does a wider wing_k range change the wings Optuna picks?
rule: exploratory
settings: {backtest: {max_concurrent_positions: 1}}
search_space: {iron_condor: {wing_k: [1.0, 2.0]}}
```

Run it with `python run_backtest.py --study-file studies/<name>.yaml` (also `run_optimizer.py`). Commit the file first and never edit it afterwards.

**Does every study need a PLAN.md entry?** No. `rule: exploratory` needs only the file and its run record, and its results may not be quoted as evidence.
A pre-registered study (one whose result could change a gate, a live parameter or a go-live criterion) gets an "AS PLANNED" entry with its pass rule
**before** the run, and its `rule:` names that PLAN heading. The run refuses to start if the heading does not exist.

**What is a run record?** `reports/runs/<run_id>/run_record.json`, written by every `run_backtest` / `run_optimizer` run. It holds the resolved
settings (no secrets), the raw text of the three config files with hashes, the effective `AIT_*` values, the git commit and a dirty flag, data
fingerprints (rows, date span, a hash of each symbol's closes), seeds, the command line, the search space and its hash, the parity manifest, the
fixed limits, the origin of every setting, and the metrics. If MLflow is installed it is also logged there. `--no-run-record` skips it.

---

## 10. Reading results

**What is in the summary?** Windows, total trades, total return, Sharpe and Sortino, max drawdown, win rate, average win/loss, expectancy,
profit factor, capital utilization, and the fraction of profitable windows. Then the **entry funnel** (why days did not trade), a buy-and-hold
benchmark, and per-window rows.

**What should I check before trusting a number?**
1. **Trade count.** A handful of trades is noise. The summary warns when fewer than 3 windows traded.
2. **The parity manifest.** It lists every setting where research differs from live. Read the warnings first.
3. **The funnel.** If one gate removes most days, the result describes the days that were left.
4. **The run record.** Is the tree dirty? Is the data span what you expected?
5. **Comparability.** Numbers made before a research default changed are not comparable (for example, `max_concurrent_positions` went 1 → 3 in W86).

**How is Sharpe annualized?** By the square root of the trades per year the sample actually made, capped at 252 (one trade per trading day).
The cap is a fixed convention (`SHARPE_ANNUALIZATION_CAP` in `backtesting/result.py`), not a setting: it decides how every reported Sharpe and Sortino
is annualized, so a study that could change it would make reported numbers incomparable across runs.

**Why is the default run's return negative or small?** The default run is a smoke test, with a $5,000 pool, enforced ML gates and few windows.
It is not evidence about the strategy.

---

## 11. Research vs live

**Where do they differ on purpose?** The manifest shows each; the main ones:

| Difference | Research | Live |
|---|---|---|
| ML gates (`gates_mode`) | `enforce` | `observe` |
| Selector direction (`condor_any_direction`) | neutral only | any |
| Concurrent positions | 3 (`backtest.max_concurrent_positions`) | per-account limits |
| Research-only vetoes | `vol_gate`, `regime_veto` (inert so far), IV-rank rise (off) | none |
| Live-only gates | not simulated (for example the VIX halt) | active |

**How do I check that research still matches live?** `run_backtest` prints the manifest; `scripts/gate_parity_ab.py` and
`scripts/ex_div_gate_ab.py` measure one-sided gates; `scripts/parity_replay.py` compares modelled entry credits with real trades.
RUNBOOK sections "One-sided entry gates" and "Research-vs-reality parity replay" explain them.

---

## 12. Common tasks

| I want to... | Command |
|---|---|
| Run the default backtest (QQQ, iron condor) | `python run_backtest.py` |
| Pick symbols / limit windows | `python run_backtest.py --symbols QQQ SPY --max-windows 2` |
| Optimize per window | `python run_backtest.py --optimize-per-window --optimize-n-trials 50` |
| Run a recorded study | `python run_backtest.py --study-file studies/<name>.yaml` |
| Run a standalone search | `python run_optimizer.py --n-trials 100 --objective composite` |
| See every value and where it comes from | `python scripts/show_research_config.py --section backtest` |
| Test a gate | `python scripts/gate_parity_ab.py --symbols QQQ` |
| Refresh the data | `scripts/backfill_historical_data.py`, `backfill_intraday.py`, `backfill_dividends.py`, `backfill_option_history.py` |

Without `--symbols` / `--strategies`, a run uses `backtest.default_symbols` (QQQ) and `options.strategies`.

---

## 13. Known limits

- **History is short and one regime.** The research calendar starts 2024-01-01. Results describe that period.
- **Option prices are modelled** (Black-Scholes plus a skew, or a QQQ-only fitted surface), and the spread model is fitted on a limited sample. Real fills differ.
- **Delayed data.** Live market data is delayed (`AIT_MARKET_DATA_TYPE=4`); fills in research are modelled, not measured.
- **One symbol has option history.** QQQ is the only symbol with the option bars an IV surface needs.
- **The learner's disable step is unreachable** with the shipped numbers ([section 7](#7-the-learner)).
- **A single run is one draw.** Compare studies on the same windows and engine, and weigh differences against window-to-window noise.
- **Research never promotes anything.** A finding changes live only through a registered PLAN.md wave and the operator's decision.

---

*Related:* RUNBOOK sections "Studies and run records", "Research tunables", "Research fixed limits", "Model hyperparameters";
`docs/WALKFORWARD_DASHBOARD.md` (reading the dashboard); `PLAN.md` (the decision record).

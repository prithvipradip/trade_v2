# Research Mode — User Guide

For researchers who want to run, trust and read the offline simulator: how research mode works, what it assumes, which settings are its own, and how to run and record a study. Short answers first.
Not for: operating the live bot ([RUNBOOK.md](RUNBOOK.md)), the whole-system picture ([GUIDE.md](GUIDE.md)), the models and features that live and research share ([MODELS.md](MODELS.md)), or what studies have found
([STUDIES_FINDINGS.md](STUDIES_FINDINGS.md)). Terms: [GLOSSARY.md](GLOSSARY.md).

> **Settings named here are in `config.yaml` (shared with live) and `config_research.yaml` (research only); the latest copies are authoritative.** Every setting with its type, allowed range and meaning is listed in
> [reference/SETTINGS_RESEARCH.md](reference/SETTINGS_RESEARCH.md) and [reference/SETTINGS_LIVE.md](reference/SETTINGS_LIVE.md). Numbers in prose below are examples.

**Contents:** [1 What it is](#1-what-research-mode-is) · [2 The pieces](#2-the-pieces) · [3 What one run does](#3-what-one-run-does) · [4 Configuration](#4-configuration-where-every-value-lives) ·
[5 The engine](#5-the-engine) · [6 The models in research](#6-the-models-in-research) · [7 The learner](#7-the-learner) · [8 The optimizer](#8-the-optimizer) · [9 Studies and run records](#9-studies-and-run-records) ·
[10 Research data](#10-research-data-collection-and-calibration) · [11 Research vs live](#11-research-vs-live) · [12 Reading results](#12-reading-results) · [13 Common tasks](#13-common-tasks) ·
[14 Assumptions and limits](#14-assumptions-and-known-limits) · [15 The Walk-Forward Lab](#15-the-walk-forward-lab)

---

## 1. What research mode is

**What is it?** An offline simulator of the iron-condor strategy. It replays history, trades it the way the bot would, and reports what happened. It is how changes are tested before they are promoted to the live bot.

**What does it read?** Historical data in `data/historical.db` (one shared set of tables, used by live and research: daily prices, daily IV, 5-minute bars, the economic calendar, dividends, earnings, QQQ holdings, option bars and
spreads; see the provenance table below), plus `config.yaml`, `config_research.yaml` and an optional study file. `backtest.research_history: true` (research overlay) selects the stored-history path; `backtest.history_end_date` pins the
last day a study may see (the tables grow every day).

**What does it never touch?** The broker, `data/ait_state.db`, `data/ait_analytics.duckdb`, and the live model files in `models/`. It writes only run records under `reports/runs/`, a research model sandbox under `models/research/`, and
(if MLflow is installed) the MLflow database `data/mlflow.db`. The live bot never loads `config_research.yaml`; a test enforces that.

**Is a research number a forecast?** No. It answers "what would this rule set have done on this history, with these cost and fill assumptions". See [14 Assumptions and limits](#14-assumptions-and-known-limits).

### Where each field comes from (the data provenance table)

Every row of history stored in `data/historical.db` carries a `source` tag saying where it came from. The daily retrain only appends completed days and never rewrites a stored one.

| Field | Table | Source tag | Written by |
|---|---|---|---|
| Underlying daily OHLCV (SPY, QQQ, IWM) | `daily_prices` | `IB_TRADES_RTH` | daily retrain (append-only), `backfill_historical_data.py --mode daily` |
| Daily implied vol (close) | `daily_prices.implied_vol` | `IB_OPTION_IV` | `ait.data.iv_refresh` (retrain, `--mode iv`), the bot for today |
| Daily implied vol (OHLC) | `daily_iv` | `IB_OPTION_IV` | `ait.data.iv_refresh` |
| VIX, VIX3M, VIX6M, VIX9D (/100) | `daily_iv` | `IB_INDEX_TRADES` | `ait.data.iv_refresh`, `--mode vix` |
| 5-minute bars | `intraday_prices` | `TRADES` | the bot (it only adds; no process deletes 5-minute bars, and the 730 days it reads for features is a load window, not a retention limit), `backfill_historical_data.py` |
| Option daily bars + solved IV/greeks | `option_daily_bars` | Polygon | `backfill_option_history.py` |
| Option bid/ask spreads | `option_spread_*`, `option_leg_spread_*` | `IB_BID_ASK_BARS` | `calibrate_option_spreads.py` |
| Dividends | `dividends` | `YAHOO_DIVIDENDS` | `backfill_dividends.py` |
| Earnings dates, QQQ holdings, econ schedule | `earnings_schedule`, `qqq_holdings`, `econ_schedule` | EDGAR / Fed / BLS tags | `build_research_calendars.py` |
| Economic events | `econ_events` | `INVESTING_CSV` | `load_econ_calendar_csv.py` |

Rows from before the source tags (old retired symbols in `daily_prices`) carry `NULL`; nothing new is written untagged. The history used to be kept twice (live tables and `research_*` copies); it is one set now.

---

## 2. The pieces

| Piece | What it is | Where |
|---|---|---|
| `run_backtest.py` | Runs a walk-forward backtest and prints the summary | repo root |
| `run_optimizer.py` | Runs an Optuna search over strategy (and optionally ML) parameters | repo root |
| Study scripts | One-question experiments (gate A/B tests, friction sweeps, parity replays); every script, with what it writes and whether it still runs, is in [reference/SCRIPTS.md](reference/SCRIPTS.md) | `scripts/` |
| Engine | Simulates one symbol over one window: entries, exits, fills, costs | `backtesting/engine.py` |
| Walk-forward | Slides train/test windows, trains models per window, runs the engine out of sample | `backtesting/walkforward.py` |
| Pool | Replays all symbols' trades against one shared account | `backtesting/pool.py` |
| Learner | Adapts size and filters between windows ([section 7](#7-the-learner)) | `backtesting/learner.py` |
| Optimizer | Searches parameters on each training window | `optimization/` |
| Study file | A committed file that changes some settings for one run | `studies/<name>.yaml` |
| Run record | A JSON file saying exactly what produced a number | `reports/runs/<run_id>/` |
| Findings register | What studies have found, and whether it may be cited | [STUDIES_FINDINGS.md](STUDIES_FINDINGS.md) |
| Walk-Forward Lab | A browser dashboard for archived runs | [WALKFORWARD_DASHBOARD.md](WALKFORWARD_DASHBOARD.md) |

---

## 3. What one run does

1. **Load config.** `config.yaml` + `config_research.yaml` (+ the study file, if given), merged and validated.
2. **Load data.** Daily bars and IV for each symbol from the shared history tables, from `backtest.history_start_date` (to `backtest.history_end_date` when pinned).
3. **Cut windows.** Default: train 365 days, a 5-day purge gap, test 63 days, step 63 days (non-overlapping tests).
4. **For each window:**
   1. Train the direction ensemble and the range model on the training slice only ([section 6](#6-the-models-in-research)).
   2. Optionally run an Optuna search on the training slice ([section 8](#8-the-optimizer)).
   3. Train the meta-labeler on the training slice, under the parameters the search chose (it learns which trades win under them).
   4. Run the engine out of sample on the test slice, one symbol at a time.
   5. Replay all symbols' candidate trades against one shared account (the pool), applying live's per-trade budget, aggregate-risk and open-position limits.
   6. Let the learner adapt for the next window ([section 7](#7-the-learner)).
5. **Report.** Totals, per-window rows, the entry funnel, a buy-and-hold benchmark, the parity manifest, and a run record.

Nothing from a test slice ever reaches the training of its own window.

---

## 4. Configuration: where every value lives

**How many layers are there?** Three, merged in this order (later wins):

1. `config.yaml`: the live configuration. Research reads it too, so a live change flows into research.
2. `config_research.yaml`: **research's own file.** It does three jobs:
   - **Diff:** a key that also exists in `config.yaml` appears here only when research deliberately differs from live (a test fails on a redundant copy, which would silently shadow a later live change).
   - **Overlay:** it is applied on top of `config.yaml` by research entry points only.
   - **Tunables:** keys that exist only in research (learner thresholds, objective weights, engine knobs, ...) are listed here in full, with comments, and a test fails if a field of those blocks is missing.
3. `studies/<name>.yaml`: changes for one run only ([section 9](#9-studies-and-run-records)).

**How do I see the value a run will use?** `python scripts/show_research_config.py` prints every setting with the layer that decides it (`study` > `overlay` > `config.yaml` > schema default). `--section backtest` narrows it, `--differs` hides
values still at their schema default, `--study-file` adds a study. The same map is stored in every run record.

**What about values the environment overrides?** A few trading values (`backtest.wing_k`, `backtest.ic_*`, `risk.iv_rank_cap`, ...) are read through the live "env contract", which ignores both overlays. A study may not set them; use the
environment variable or the command-line flag.

**Where are the research-only blocks?** In `config_research.yaml` under `backtest:`. A study may override any of them for one run.

| Block | What it holds | Read by |
|---|---|---|
| `learner` | Thresholds of the research learner: trades needed to learn / disable, win-rate cutoffs, step sizes, multiplier and confidence bounds, protected strategies. Not the live learning layer | `backtesting/learner.py` |
| `legacy_engine` | Constants of the engine's dormant paths (fallback direction model, naked-margin fraction, thesis-invalidation confidence, vol-regime ratio); none is on the iron-condor path under `engine_model: realistic` | `backtesting/engine.py` |
| `objectives` | Weights of the Optuna objectives and the profit-factor cap | `optimization/objectives.py` |
| `walkforward_tuning` | Window lengths (`train_days`, `test_days`, `step_days`, `gap_days`), Optuna trial counts and seed, row floors, context and fetch paddings, out-of-sample diagnostic bins and minimums, the minimum number of windows (`min_windows`), the window-score shrink, the test-window feature context (`oos_feature_context_bars`), warm start, adaptive range threshold, validation split, inert-parameter probe. `--train-days`, `--test-days`, `--step-days`, `--gap-days`, `--optimize-n-trials` and `--n-trials` override them for one run | `backtesting/walkforward.py`, `optimization/optimizer.py`, `run_backtest.py`, `run_optimizer.py` |
| `engine_tuning` | The engine's IV-skew coefficients, the look-backs of the realized-vol and IV-rank-rise entry gates, the minimum history before the first decision, the fallback volatility used when a position has no entry IV, the drawdown gate's rolling-high window and the replay's history pad | `backtesting/engine.py`, `pricing.py`, `replay.py` |
| `surface_tuning` | The acceptance rules of the research IV surface (minimum volume and price, moneyness, points, span, IV bounds, clamp margin, stale days); part of the surface cache key, so a changed rule rebuilds it | `backtesting/iv_surface.py`, `engine.py` |
| `friction_fit` | The minimum data a leg-spread (friction) model needs to be fitted: samples, sessions for cross-validation, training rows, affine points. Offline calibration only | `backtesting/friction.py` |
| `spread_tuning` | Sizing numbers of the non-condor builders (put / call credit spreads, iron butterfly, jade lizard, debit structures): max cost as a share of capital and the wing as a fraction of the expected move. The iron-condor builder does not read them | `backtesting/engine.py` |

**Are some values not configurable?** Yes. Clamps and numerical tolerances are fixed in `config/invariants_research.py`. They are logged at the start of every research run and stored in every run record. To change one: edit the
module, run `python scripts/gen_docs.py`, and record the decision in `PLAN.md`; a change moves research numbers.

<!-- BEGIN GENERATED: research_fixed_limits -->
<!-- generated by scripts/gen_docs.py from ait/config/invariants_research.py; edit the code, not this table -->
| Name | Value | Unit | What it does, and why it is fixed | Enforced in |
|---|---|---|---|---|
| `RESEARCH_DELTA_MIN` | 0.05 | \|delta\| | Lower end of the delta band a strike search may be asked for. Below it the strike is far enough OTM that a modelled quote is noise. Also the lower bound the optimizer accepts for delta_short / delta_long overrides (a wider range would search values the engine clamps). | backtesting/engine.py (_build_debit_position, _build_credit_position); optimization/param_spaces.py HARD_BOUNDS |
| `RESEARCH_DELTA_MAX` | 0.45 | \|delta\| | Upper end of the delta band a strike search may be asked for. Above it a leg is no longer out of the money; the structures built from it stop meaning what their names say. | backtesting/engine.py; optimization/param_spaces.py HARD_BOUNDS |
| `IV_SCALE_MIN` | 0.5 | ratio | Lower clamp of the iv_floor / iv scaling applied to a builder's delta. Keeps one stale or extreme IV print from moving a strike by more than a factor of two. | backtesting/engine.py (_build_debit_position, _build_credit_position) |
| `IV_SCALE_MAX` | 1.5 | ratio | Upper clamp of the same scaling. Same reason, other direction. | backtesting/engine.py |
| `SETTLEMENT_LOSS_TOLERANCE` | 0.000001 | dollars | How far a settled gross loss may exceed the defined max loss before it is counted as over_max_loss. A floating-point tolerance, not a modelling choice; loosening it would hide a pricing bug. | backtesting/engine.py (_finish_settlement) |
| `PROB_CLIP_EPS` | 0.0000001 | probability | Probabilities are clipped to [eps, 1 - eps] before log-loss. Numerical safety (log 0); not a modelling choice. | backtesting/walkforward.py (_prob_score_metrics) |
<!-- END GENERATED: research_fixed_limits -->

**Shared settings.** Some `config.yaml` sections are also read by research: model hyperparameters, feature windows and switches, label constants and the meta-labeler's neutral values ([MODELS.md](MODELS.md) lists them). They are diff-only in
`config_research.yaml`: research differs only if you put a different value there (or in a study), and the parity manifest then warns.

**Where the history and row-count minimums are set.** Six research minimums are settings in `config_research.yaml`:

| Setting (under `backtest:`) | What it controls |
|---|---|
| `engine_tuning.drawdown_high_lookback_rows` | Rows of closes in the rolling high the drawdown entry gate measures the fall from. The result keeps the name `pct_from_60d_high`. |
| `engine_tuning.replay_lookback_days` | Calendar days of history the single-trade replay gives the engine before the entry date. Calendar days are fewer trading rows, so a replayed engine's row-count gates run on a shorter history than in a full run. |
| `walkforward_tuning.oos_feature_context_bars` | Bars of the training tail put in front of each test window so the features have their look-back. A shorter context does not fail: it changes the feature values, so the model would be scored on inputs it was not trained on. A value below the feature spec's longest look-back is refused when the walk-forward starts. Not `backtest.context_bars`, which is the engine's intraday look-back. |
| `walkforward_tuning.meta_label_min_window_rows` | The meta-labeler is trained for a window only when the training window has more than this many bars. A guard on length, not a context. |
| `walkforward_tuning.min_fetched_rows` | A symbol is used only when more than this many daily bars were fetched. |
| `walkforward_tuning.min_backtest_rows` | The optimizer's minimum daily bars: a symbol with fewer is dropped and a trial on fewer raises "insufficient data". With a validation split on, the split needs at least twice this many bars. |

**A research number belongs in a setting, not in code.** A test (`tests/test_w88_literal_census.py`) compares the numeric literals of the result-affecting research modules with an accepted list, so a number cannot be added, changed or removed
unseen. How to satisfy it, and what each category means, is a developer topic: [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) ("Research numbers and the literal census").

---

## 5. The engine

**Which engine runs?** `backtest.engine_model: realistic` (set in `config_research.yaml`). It prices and fills at live fidelity: fill-time spot, previous-session IV, listed session expiries, same-session exits, touch stops, and expiry
settlement or assignment. The older `legacy` engine is kept byte-identical for reproducing old numbers.

**How are options priced?** Black-Scholes from one stored IV per day plus a skew (`pricing_source: bs_legacy`). A fitted IV surface from real option bars is available (`pricing_source: surface`, QQQ only).

**What does a trade cost?** Commission plus fees per leg, entry and exit, and a fitted half-spread per leg (`friction_model: per_leg_usd`, with short-DTE bands). Debit structures also pay `backtest.slippage_pct` (iron condors do not).

**What decides whether an entry happens?** A chain of gates, in order. A day that fails one is counted in the **entry funnel** under that gate's name:

| Funnel name | Meaning |
|---|---|
| `no_strategy` | The selector found no allowed structure |
| `iv_rank_floor` / `iv_rank_cap` / `iv_rank_unknown` | IV rank below the condor floor, above the credit cap, or not measurable (fails closed) |
| `range_gate_below_threshold` | The range model's P(stay in range) is under `ml.range_min_confidence` (only when `gates_mode: enforce`) |
| `macro_event` | An FOMC/CPI/NFP/PCE/GDP event is within `risk.pre_event_blackout_days` |
| `ex_div_entry` | A short call could be assigned early before an ex-dividend date (live's rule) |
| `max_concurrent` | `backtest.max_concurrent_positions` already open |
| `builder:*` | The condor builder refused (credit floor, credit-to-width ratio, affordability, ...) |
| `vol_gate`, `regime_veto`, `iv_rank_rise` | Research-only vetoes (see [11 Research vs live](#11-research-vs-live)) |

**What exits a trade?** Take-profit on a DTE ladder, a touch stop, a close at 5 DTE or less, and a hold limit. The same ladder constants live and research share (`execution/exit_policy.py`).

**What does the engine see on a given day (the decision frame)?** With `engine_model: realistic` and intraday entries, the entry gates and models see history through the **previous** session (a 10:30 decision cannot know that day's close),
and an entry counts positions that were still alive at its fill time toward `max_concurrent_positions`. Legacy and daily-close studies are unchanged.

**How is expiry modelled?** `ait.backtesting.settlement` reproduces expiry outcomes (expired, exercised, assigned) from the strikes and the close, with assumptions that are operator-owned settings: `backtest.pin_band_usd`,
`assignment_fee_per_share`, `stock_commission_per_share`; leftover shares are marked at the next session's open. In the realistic engine the settlement path is reached only for a position the take-profit ladder and touch stop did not already
close, so it matters most for studies that disable those exits. Early assignment before expiry is **not** modelled ([14](#14-assumptions-and-known-limits)).

**Range models per horizon.** The walk-forward trains the range gate's model for the horizon the study's entry DTE can actually reach (`ait.ml.range_spec.horizon_for`, shared with the live spec). The configured band's own horizon is the base
model; any other DTE is rounded **up** to a bucket in `RANGE_HORIZON_BUCKETS` and trained once per window, on demand, so a model never answers an easier question than the trade needs. Non-base buckets save under `models/research/h<N>/`;
the live `models/range.pkl` is never touched. `python scripts/range_horizon_study.py --symbol QQQ` measures out-of-sample skill, abstention and reliability per horizon (read-only; artifacts go to a temporary directory).
**Abstention is not a bug:** when no ensemble member has cross-validated edge over the base rate the model returns no probability, and the engine's range gate then rejects (enforce) or would-reject (observe); the study reports how often
that happens per horizon, and it is worth reading before enforcing the gate.

---

## 6. The models in research

**Which models are trained per window?** Three, on that window's training slice only: the **direction ensemble** (XGBoost + LightGBM), the **range model** (one per horizon bucket when the entry DTE is searched) and the **meta-labeler**
(trained last, under the parameters the search chose). The vol-magnitude model is live only; research does not train it. What each model is, its features, labels, validation and hyperparameters: [MODELS.md](MODELS.md).

**Do the models gate entries in research?** `backtest.gates_mode` decides. `enforce` (research's setting): the range floor blocks entries. `observe` (live's shipped mode): the models predict and log but never block. This is a deliberate
research-versus-live difference; the parity manifest shows it. Why live observes: [MODELS.md](MODELS.md) "Gating: observe and enforce".

**Can I change a model setting for one study?** Yes: hyperparameters (`ml.hyperparams`, `ml.meta_label_hyperparams`), feature windows and per-feature switches (`ml.features`) go in the study file (or the overlay). Loading a model trained under other
windows or switches is refused and it is retrained; the run record stores the feature-spec hash and the resolved feature list. A study that changes a feature looks like:

```yaml
settings:
  ml:
    features:
      fractal: {use: {hurst_psd_divergence: false}}      # drop a feature from the model inputs
      market_structure: {use: {skew_proxy: true}}        # try a computed-but-unused one
      momentum: {rsi_long: 21}                           # change a window (rsi_14 becomes rsi_21)
```

**Could a research run overwrite the live model?** No. Window models are throw-away (`persist_artifacts=False`) or go to `models/research/`.

---

## 7. The learner

**What is it?** `BacktestLearner`, a small in-memory loop in the walk-forward. After each test window it looks at that window's trades and adapts three things for the next window. It runs in sequential mode only (it is off when
`optimize_n_jobs > 1`).

- **Strategy multipliers:** a per-strategy size factor, starting at 1.0. A strategy that keeps losing gets its factor cut. One that keeps winning gets it raised.
- **Confidence floor:** if low-confidence trades keep losing, the minimum entry confidence is raised by 0.05, up to a cap of 0.85.
- **Symbol disabling:** a symbol that keeps losing is switched off for later windows.

**What is a multiplier, and where is it used?** The walk-forward sizes each window's trades at `position_size_pct × min(average multiplier of the active strategies, 2.0)`. A multiplier of 0.41 means trades are sized at 41% of normal. In an
iron-condor-only run the learner can therefore shrink condor size after losing windows. That is part of what a research number contains.

**What does `reduce_min_change` do?** It is a dead-band. A reduction is applied only if it moves the multiplier by more than 0.03. This stops tiny adjustments, but it has a side effect at the floor: the multiplier goes 0.41 → 0.33, and the
next step would land on the 0.30 floor, a change of only 0.027, so it is skipped and the multiplier stays at 0.33.

**What do `boost_min_trades` and `reenable_min_trades` do?** They are extra trade-count conditions for boosting a winner (6 trades) and re-enabling a disabled strategy (10 trades). Only the learner's per-window strategy step reads them, inside
a loop that already skips any strategy with fewer than `min_trades_to_learn` (15) trades. Both are below 15, so they never block anything today. They only matter if you set them above 15.

**What is "disable a strategy" meant to achieve?** To stop a strategy that has lost across many windows from being traded in the remaining windows. The sequence is: reduce its size repeatedly, then switch it off; if it later recovers,
re-enable it at half size. Iron condors are exempt from disabling (they are regime-dependent), so for them the end state is just a smaller size.

**Does the disable step ever happen?** Not with the shipped numbers: the multiplier never reaches 0.30 (see `reduce_min_change`), so strategy disabling is unreachable. Symbol disabling and the confidence raise do work.

<!-- BEGIN GENERATED: learner_defaults -->
<!-- generated by scripts/gen_docs.py from ait/config/settings.py and config_research.yaml; edit the code, not this table -->
**Learner settings** (`backtest.learner`)

| Setting | Default | Meaning |
|---|---|---|
| `windows_before_adapting` | 2 | Windows whose trades must be accumulated before any adaptation (the first window only accumulates). |
| `min_trades_to_learn` | 15 | A strategy / symbol / confidence band needs this many trades before it is judged. |
| `min_trades_to_disable` | 25 | Trades needed before a strategy or symbol is disabled outright (more evidence than reducing). |
| `protected_strategies` | [iron_condor] | Strategies the learner may reduce but never disable (condors are regime-dependent). |
| `strategy_multiplier_min` | 0.3 | Floor of a strategy's sizing multiplier after reductions. |
| `strategy_multiplier_max` | 2.0 | Cap of a strategy's sizing multiplier after boosts. |
| `reduce_win_rate` | 0.3 | Win rate below which (with negative average P&L) a strategy is reduced / disabled. |
| `reduce_factor` | 0.8 | Multiplier applied to a losing strategy's size multiplier per adaptation. |
| `reduce_min_change` | 0.03 | A reduction smaller than this (absolute multiplier change) is not applied. |
| `boost_win_rate` | 0.58 | Win rate above which (with positive average P&L) a strategy is boosted. |
| `boost_min_trades` | 6 | Trades needed to boost. |
| `boost_factor` | 1.2 | Multiplier applied to a winning strategy's size multiplier per adaptation. |
| `boost_min_change` | 0.05 | A boost smaller than this (absolute multiplier change) is not applied. |
| `reenable_win_rate` | 0.5 | Win rate above which a disabled strategy is re-enabled. |
| `reenable_min_trades` | 10 | Trades needed to re-enable. |
| `reenable_multiplier` | 0.5 | Size multiplier a re-enabled strategy restarts at. |
| `confidence_cap` | 0.85 | Highest minimum-confidence the learner may raise the entry floor to. |
| `confidence_low_band_upper` | 0.65 | Entry confidence below this is the 'low' band the learner watches. |
| `confidence_mid_band_upper` | 0.75 | Entry confidence below this (and not low) is the 'medium' band. |
| `confidence_raise_win_rate` | 0.35 | If the low band wins less often than this (with negative average P&L), the floor is raised. |
| `confidence_raise_step` | 0.05 | How much the minimum-confidence floor is raised per adaptation. |
| `confidence_raise_min_change` | 0.01 | A raise smaller than this is not applied (the cap was reached). |
| `symbol_disable_win_rate` | 0.2 | Win rate below which (with negative average P&L and min_trades_to_disable trades) a symbol is disabled for the following windows. |
| `default_entry_confidence` | 0.65 | Confidence assumed for a trade record that carries none. |
<!-- END GENERATED: learner_defaults -->

The live learning layer is a different, simpler thing: [GUIDE.md](GUIDE.md) section 10.

---

## 8. The optimizer

**What does it do?** `StrategyOptimizer` searches parameters (wing width, short delta, DTE and so on) with Optuna's TPE sampler (seeded). Each trial is one engine run; the best trial's parameters are used for that window's
out-of-sample test.

**How is a trial scored?** By the objective: `sharpe_ratio`, `composite`, `profit_factor` or `win_rate`. The composite is `0.4 × Sharpe + 0.4 × win rate − 0.2 × |max drawdown|` (weights in `backtest.objectives`). Fewer than 3 trades
scores on a graded floor, so the search has something to climb.

**What is the graded floor?** A trial with fewer trades than `backtest.optimize_hard_min_trades` scores `optimize_below_floor_score + optimize_below_floor_gradient × trades / hard_min`: the base for zero trades, rising by up to the
gradient, so trading always beats not trading and a starved trial always ranks below a real one. Keep the base far below any objective's real scores (it is bounded to -1000..-1, and base plus gradient must stay negative). All three are in
`config_research.yaml`.

**What is searched?** For iron condors, the dimensions below, from `optimization/param_spaces.py`. The range of a dimension is its default; a study can narrow or widen one (`search_space:` in the study file), inside the hard limits.
Everything else is held at its configured value, so that a per-window search varies only the structure and the regime-specific gates and cannot trade away the risk rules: the stops and take-profit (`stop_loss_pct`, `profit_target_pct`,
`trailing_stop_fraction`), the entry confidence and realized-vol gates, the IV-floor and the bid/ask spread model.

<!-- BEGIN GENERATED: search_space_iron_condor -->
<!-- generated by scripts/gen_docs.py from ait/optimization/param_spaces.py; edit the code, not this table -->
| Dimension | Type | Default range | Hard limit for an override | Meaning |
|---|---|---|---|---|
| `delta_short` | float | 0.15 - 0.3 | 0.05 - 0.45 | Target \|delta\| of the two short strikes. Lower sits further out of the money (more room, less credit); higher collects more credit and is tested more often. |
| `max_hold_days` | int | 1 - 9 | 1 - 45 | Calendar days a position may be held. The reachable window is the entry DTE minus the days before expiry at which credit structures are closed, so the range stops at what can actually happen. |
| `wing_k` | float | 0.3 - 2.0 | 0.1 - 5.0 | Scales the wing width: width = wing_k x price x IV x sqrt(DTE/365), so 1.0 is one expected move. Sets the maximum loss per contract. |
| `iv_rank_rise_threshold` | float | 0.25 - 0.6 | 0.0 - 10.0 | Blocks entry when the realized-vol IV rank rose by more than this over 10 days. Searched only while `backtest.iv_rank_rise_gate` is on (it is off in research). |
| `min_edge_over_baseline` | float | 0.02 - 0.15 | 0.0 - 1.0 | The weighted cross-validated edge the range model must show over the base rate before it makes any prediction. |
| `hurst_regime_threshold` | float | 0.08 - 0.3 | 0.0 - 1.0 | Hurst scale spread above which the fractal gate acts (a trending market is bad for a range strategy). |
| `hurst_regime_penalty` | float | 0.0 - 0.25 | 0.0 - 1.0 | Confidence penalty the fractal gate applies when it acts without vetoing. |
| `multifractal_max_width` | float | 0.3 - 0.65 | 0.0 - 2.0 | Largest multifractal-spectrum width allowed before the fractal gate acts (a wide spectrum means an irregular, bursty price process). |
<!-- END GENERATED: search_space_iron_condor -->

An override is validated (known dimension, `low < high`, inside the hard limits), the optimizer stores a hash of the effective space on the Optuna study, and a stored study will not resume over a different space. The run warns when a range
excludes the live value (`wing_k`, `delta_short`). Other strategies have their own spaces in the same file; the ML spaces (`xgboost`, `lightgbm`) cover the trees' hyperparameters for the retrain's optional search.

**What is warm start?** A window's search starts from the previous window's best parameters if that window's out-of-sample win rate was at least `warm_start_min_win_rate` with enough trades, else from the best overall
(`backtest.walkforward_tuning`).

**Can the entry DTE be searched?** Yes, opt-in:

- **Switch:** `python run_backtest.py --optimize-per-window --search-dte [--dte-range 0 45] [--optimize-strict]`. Off by default; `backtest.dte_search` (schema defaults, overlay-overridable) holds the ranges. The standard iron-condor
  space is untouched, so every other study is unchanged.
- **What is searched:** the entry DTE, the expiry guard (sampled only for entries it can exempt) and the hold fraction (which **replaces** `max_hold_days`). The expiry-day flatten is not searched (see [STUDIES_FINDINGS.md](STUDIES_FINDINGS.md)). The credit floors (`ic_min_credit*`) are never searched; run them as sensitivities.
- **Always pass `--optimize-strict` for numbers you will quote.** Without it a window whose optimization fails runs the base config and the summary only reports a count (`optimization_failures`); with it the study stops.
- **Reading the report:** a band is a *candidate* only if it holds a top-decile trial in at least two thirds of windows. Promotion also needs its parity band to PASS (`scripts/parity_replay.py`) and its friction band to be ACTIVE
  (`data/short_dte_collection_status.json`). The report prints a reminder; it does not check either.
- **History limit:** the option-bar history is short (about two years when this was written; check the span of `option_daily_bars`), so a surface-priced study has only a handful of out-of-sample quarters. Use shorter training windows and/or pool symbols, and say so next to any number.

`run_optimizer.py --apply` is refused unless `AIT_ALLOW_APPLY=1` is set (the parameters are in-sample only) and is refused with a study (parameters found under a study's settings are not promotable to the live config). It writes only four named parameters (`risk.min_confidence`, `exit.initial_stop_loss_pct`, `exit.trailing_stop_pct`, `exit.breakeven_trigger_pct`) into the config file named by `--config`, and it saves that file as plain YAML, which **drops every comment in it**: `config.yaml` is heavily commented, so back it up first or apply by hand.

---

## 9. Studies and run records

**What is a study?** A committed file `studies/<name>.yaml` that says what one research run changes relative to `config.yaml` + `config_research.yaml`. `config_research.yaml` stays the stable default for ordinary research runs; editing it
per study would silently change every later default run. The template is [`studies/_template.yaml`](../studies/_template.yaml):

```yaml
name: wing_k_wider
question: Does a wider wing_k range change the wings Optuna picks?
rule: exploratory
settings: {backtest: {max_concurrent_positions: 1}}
search_space: {iron_condor: {wing_k: [1.0, 2.0]}}
```

- **Run it:** `python run_backtest.py --study-file studies/<name>.yaml` (also `run_optimizer.py`). It is `--study-file`, not `--study`: the optimizer already has the Optuna `--study-name`. **Commit the file before the run and never edit it
  afterwards** (a changed study is a new file).
- **`rule:`** is either `exploratory` (looking around; its results may not be quoted as evidence) or the text of a `PLAN.md` heading where the pass/fail rule was written **before** the run. A study whose `rule:` is neither is refused at
  start; a test fails if a study names a heading that does not exist.
- **Does every study need a PLAN.md entry?** No. A **pre-registered** study (its result could change a gate, a live parameter or a go-live criterion) gets an "AS PLANNED" entry with the pass rule and a pointer to the study file
  *before* the run, and a result entry after. An **exploratory** study needs only the file and its record; a decision that leans on one cites it (file and commit) in its own PLAN entry.
- **Refused on purpose:** `ibkr` / `api_keys`; any setting the env contract reads (`backtest.wing_k`, `backtest.ic_*`, `risk.iv_rank_cap`, ...), because the contract reads the raw `config.yaml` and ignores both overlays, so the value
  would be silently ignored (use the environment variable or the CLI flag). Walk-forward window workers and the optimizer's `--n-jobs` receive the parent's resolved settings, so they see the overlay and the study.
- **Defaults for a bare run:** `--symbols` is `backtest.default_symbols` (QQQ: the symbol with historical option prices, so an IV surface can be built) and `--strategies` is `options.strategies` (iron condor); explicit lists win.

**What is a run record?** `reports/runs/<run_id>/run_record.json`, written by every `run_backtest` / `run_optimizer` run (`--no-run-record` skips it; the folder is git-ignored). It holds the resolved settings (no secrets), the raw text of the
three config files with hashes, the effective `AIT_*` values, the git commit and a dirty flag, data fingerprints (rows, date span, a hash of each symbol's closes), the feature-spec hash and the resolved feature list, seeds, the command line,
the search space and its hash, the parity manifest, the fixed limits, the origin of every setting, and the metrics. If MLflow is installed it is also logged (experiment `research_<study name or adhoc>`; tracking URI `MLFLOW_TRACKING_URI`
or `sqlite:///data/mlflow.db`); a failure there never fails a run.

**Where do the results go?** Not here. What a study found, whether it may be cited and which decision it fed are recorded in [STUDIES_FINDINGS.md](STUDIES_FINDINGS.md).

---

## 10. Research data: collection and calibration

**Refreshing the shared history.** `scripts/backfill_historical_data.py` (daily bars, 5-minute bars, implied vol with `--mode iv`, the VIX family with `--mode vix`), `backfill_intraday.py`, `backfill_dividends.py`,
`backfill_option_history.py` (Polygon option bars), `build_research_calendars.py` and `load_econ_calendar_csv.py` fill the tables in the provenance table above. The daily retrain keeps daily bars and implied vol fresh without a new job.

### 10.1 Short-DTE friction collection

The research engine had no real bid/ask sample below about 11 DTE, so every 0-6 DTE leg would be charged an extrapolated half-spread. IB serves historical BID/ASK bars for currently **listed** contracts only (nothing for expired ones), so this
data cannot be backfilled: it has to be collected forward, one RTH session at a time, while the 0-6 DTE contracts are still listed. The collection only requests historical data; it never places an order, and paper Gateway/TWS is fine.

```bash
# 1) see the plan first (no data requested, nothing stored)
python scripts/calibrate_option_spreads.py --source ib-bars --short-dte --symbols QQQ --client-id 91 --dry-run
# 2) collect + fit (tens of minutes per symbol: IB paces historical requests at 60 per 10 minutes)
python scripts/calibrate_option_spreads.py --source ib-bars --short-dte --symbols QQQ --client-id 91
```

- **When: about 15:30 ET, before the close.** After the session ends IB no longer serves that day's expiring contracts, so an after-close run can never capture DTE 0. At 15:30 the entry window's last bar has ended, so one run holds the whole
  expiry day; the DTE-0 contracts are requested first. Bars whose hour had not ended are never stored (`in_progress_bars_skipped` in the summary).
- **Port and client id.** The port defaults to `IBKR_PORT` from `.env` (pass `--port` only to override); a "could not connect" error almost always means TWS/Gateway is not running on that port. The script and the supervisor job accept either
  endpoint of the configured account type (paper 7497 or 4002, live 7496 or 4001) and never probe across account types. Use a client id that is not the bot's (`IBKR_CLIENT_ID` and its fallback ladder): by convention 90 = historical
  backfill, 91 = this collection, the retrain's own id is `ml.retrain_ib_client_id`; two sessions on one id make the second fail.
- **Coverage and target.** The run prints samples and distinct sessions per DTE and, per band (0-2, 3-6, ...), `collecting: N/15 sessions (M more needed)` until the band is `ACTIVE`. The target is `backtest.friction_min_sessions_per_band`
  sessions per band; weekends and holidays do not count, and missing a day only delays the date. A pacing violation backs off 60 s and continues; re-running the same day is safe.
- **Using it.** Set `backtest.friction_short_dte: true` in `config_research.yaml`. A band below the minimum is charged the all-DTE model floored at `friction_extrapolated_floor_usd` per leg, and the run logs `friction_summary` (fitted and
  extrapolated counts). Off by default. Storage: `option_spread_bar_samples` (raw) and `option_leg_spread_band_params` (fits) in `data/historical.db`; the live bot never reads them.

**Lifecycle capture (`--lifecycle`): one run a day, each contract downloaded about twice in its life.** The per-day collector asks every contract in range for its last 7 days, every day, so a contract is downloaded 5-11 times and the overlap is
discarded. IB serves 30 days of hourly BID/ASK bars for any listed option in one request, and what is perishable is the contract at its **expiry**, not each day. Lifecycle capture therefore runs *stage A* (expiries today..today+N, run at 15:30 ET) and
*stage B* (far-listed expiries whose long-DTE tail is not stored yet, nearest first, capped by `--max-far-contracts`; "not stored" is read from the database, so a missed day needs no state file). Strikes are the union, over every
`--thin-every`-th session of the underlying's path, of what a condor would have sold that day. Bands (0-2, 3-6, 7-13, 14-29, 30-45 DTE) are each fitted and gated on the minimum sessions; only 0-6 DTE is extrapolated (floored at the tick) while
unfitted. Dry-run first: `... --short-dte --lifecycle --symbols QQQ --client-id 91 --dry-run`. The capture log is `option_capture_log` in `data/historical.db`.

**Automatic collection (supervisor job).** The supervisor can run the collection Mon-Fri before the close. It is **off by default**: prove it manually once (dry run, then a real run); uncomment the `data_collection:` block at the bottom of
`config.yaml` and set `short_dte_enabled: true` (time defaults to 15:30 ET, symbols `[QQQ]`, client id 91; add `lifecycle: true` for lifecycle capture); **restart the supervisor** (a process runs on the settings it started with; it logs
`short_dte_collection_scheduled` if it took effect); and keep the supervisor and TWS/Gateway running at that time. At the scheduled time it skips non-trading days, refuses to run if the IB port is closed or the client id collides with the
bot's, runs the same script with an idle timeout and a hard cap, and never touches the bot or any order. Progress without opening logs: `data/short_dte_collection_status.json` (counters, samples per DTE, per band `sessions` vs `needed`, and
`active` once the band has a fit), `logs/short_dte_collect_YYYYMMDD.log`, and `short_dte_collection_ok` in `logs/orchestrator.log`. A healthy day is silent; you get at most one Telegram message a day, and only if something is wrong.

### 10.2 Parity replay: research versus reality

`python scripts/parity_replay.py` runs known real trades (`data/replay/manual_trades_*.json`, git-ignored; never commit it) through the research engine's pricing and exit code and reports where research disagrees with what actually filled. Use it
before trusting a DTE band in an Optuna study, and after any change to `backtesting/pricing.py`, `friction.py` or the engine's exit path.

- **Inputs (all read-only):** the replay file, the research tables in `data/historical.db`, and the bot's combo-fill context in `data/ait_state.db`.
- **Outputs:** `data/replay/out/<YYYYMMDD>/`: `summary.md` is statistics only and safe to quote; `trades.csv` and `legs.csv` carry per-trade prices and stay local. The CLI refuses an output directory inside the repo outside `data/replay/`.
- **Reading it:** error = model − actual, so positive means research is *richer* than reality. `engine` = what the engine does today (daily close, same-day IV); `clean` = what was knowable at the fill. A band with fewer trades than
  `parity.min_trades_per_band` is INSUFFICIENT, never PASS. Tolerances are the `parity:` block of `config_research.yaml` (operator-owned).
- **Expiry settlement parity:** `summary.md` also reports, for every structure that reached its expiry bell, whether the modelled leg outcomes (expired / exercised / assigned) and leftover shares match the statement, and the P&L error.
- **Never paste** per-trade prices, P&L or any account identifier into `PLAN.md`: it is shared via git with the other box. Report error statistics by DTE band only.

---

## 11. Research vs live

**Where do they differ on purpose?** The parity manifest that `run_backtest` prints shows each; the main ones:

| Difference | Research | Live |
|---|---|---|
| ML gates (`gates_mode`) | `enforce` | `observe` |
| Selector direction (`condor_any_direction`) | neutral only | any |
| Concurrent positions | `backtest.max_concurrent_positions` | per-account limits |
| Research-only vetoes | `vol_gate`, `regime_veto` (see below), IV-rank rise (off) | none |
| Live-only gates | not simulated (for example the VIX halt and the VIX-tiered credit caps) | active |
| Ex-dividend entry block | `backtest.ex_div_entry_gate` (a twin of live's rule) | active |

### 11.1 One-sided entry gates

Research and live can disagree about whether a condor may open without any parameter differing: a gate that exists on **one side only**. Three are known: the research realized-vol cap (`backtest.max_entry_vol_annual`; the optimizer can search it),
the research trending-down regime veto (`backtest.regime_veto_gate`), and live's VIX halt (`risk.credit_vix_halt`, plus the VIX-tiered credit caps), which research does not simulate.

- **Where it is visible.** `python run_backtest.py` prints one warning per active one-sided gate and records `live_only_gates_not_simulated`. The registry is `ait.backtesting.gate_parity`: every reason the engine can reject on is classified as a
  live twin, a research-only gate (with its switch) or mechanics, and a test fails if the engine grows a reason without a row.
- **The switch.** `backtest.regime_veto_gate` (schema default true). False skips the veto only; the day class is still recorded per entry (`regime_class`; a day the veto would have blocked is flagged `regime_veto_suppressed`). Its thresholds are
  `backtest.regime_down_threshold` and `regime_up_threshold`.
- **Measure it:** `python scripts/gate_parity_ab.py --workers 3 [--entry-dte N]` (writes no rows, and a file only with `--dump`; the VIX closes come from Yahoo). Arms: baseline, regime veto off, vol cap off, both off; then live's VIX halt is applied after the fact (no slot
  re-use). **Re-run it whenever the gate order, the IV-rank cap or the entry-DTE range changes: these gates only matter on the days that reach them.**
- Research prices with `options.risk_free_rate`, the key live's Greeks read.

### 11.2 The IV-rank rise veto

The research engine can block a neutral credit entry when the feature-engine `iv_rank` (a **realized**-vol range rank, 0-1) rose by more than `backtest.iv_rank_rise_threshold` over 10 rows. **Live has no such veto**: it gates on the level of the
stored-IV percentile (condor floor, credit cap; `ait.data.iv_rank`). A test pins that nothing outside the research modules reads the threshold. `python scripts/iv_rank_parity_probe.py [--ab --workers 6]` reports how the two ranks relate, how
many of the veto's days live would have refused anyway, and how many are days live *would* trade. The veto is **off** in research (`backtest.iv_rank_rise_gate: false`): the code, the threshold and its search range are kept, the rise is still
recorded per entry (`iv_rank_rise_10d`), and the optimizer does not search it. The bar for turning it back on: a multi-window walk-forward with the gate on versus off, same windows, engine, pricing and friction, showing an out-of-sample gain beyond
window-to-window noise on days live's level gates would have traded; if it earns its place, port it to live too. The research IV series and live's stored series are the same IB historical IV except on the few days live's own scan wrote.

### 11.3 The ex-dividend entry block

**What live does.** The bot refuses a new position that **sells a call** when the projected next ex-dividend date falls between today and the chosen expiry (log event `trade_blocked_ex_div`): a short call can be assigned early the day before
an ex-date. The projection is `ait.data.dividends.project_next_ex_div` (the last four ex-dates from yfinance, mean gap, stepped to today).

**The research twin.** `backtest.ex_div_entry_gate`: `off` | `live` (live's rule exactly, from the ex-dates known before the entry day) | `exact` (any real ex-date in [entry, expiry]) | `last_k_days` (a real ex-date within
`backtest.ex_div_entry_k_days` of expiry). The schema default is `off`; `config_research.yaml` sets `live`. The manifest warns whenever research differs from live. It reads the `dividends` table in `data/historical.db` and raises without it;
refresh it with `python scripts/backfill_dividends.py --symbols QQQ SPY IWM [--dry-run]` before a study that ends after the last stored ex-date. Measure it: `python scripts/ex_div_gate_ab.py --workers 3 [--entry-dte 14 7] [--k 3 5 7]`
(writes no rows; real engine runs, so a blocked entry frees its slot). The `live` arm answers the parity question; `exact` and `last_k_*` are the evidence for changing live's rule (the operator's call). Research does **not** model early
assignment, so the comparison shows what the block costs, not what it protects against.

### 11.4 Checking that research still matches live

`run_backtest` prints the manifest; `scripts/gate_parity_ab.py` and `scripts/ex_div_gate_ab.py` measure one-sided gates; `scripts/parity_replay.py` compares modelled entry credits with real trades (10.2).

---

## 12. Reading results

**What is in the summary?** Windows, total trades, total return, Sharpe and Sortino, max drawdown, win rate, average win/loss, expectancy, profit factor, capital utilization, and the fraction of profitable windows. Then the **entry
funnel** (why days did not trade), a buy-and-hold benchmark, and per-window rows.

**What should I check before trusting a number?**
1. **Trade count.** A handful of trades is noise. A run with fewer walk-forward windows than `walkforward_tuning.min_windows` (3 by default) aborts, and `run_backtest.py` warns when too few windows produced results.
2. **The parity manifest.** It lists every setting where research differs from live. Read the warnings first.
3. **The funnel.** If one gate removes most days, the result describes the days that were left.
4. **The run record.** Is the tree dirty? Is the data span what you expected?
5. **Comparability.** Numbers made before a research default changed are not comparable (for example, when `max_concurrent_positions` changed).

**How is Sharpe annualized?** By the square root of the trades per year the sample actually made, capped at 252 (one trade per trading day). The cap is a fixed convention (`SHARPE_ANNUALIZATION_CAP` in `backtesting/result.py`), not a setting:
it decides how every reported Sharpe and Sortino is annualized, so a study that could change it would make reported numbers incomparable across runs.

**Why can a default run look poor?** The default run is a smoke test, with a small pool, enforced ML gates and few windows. It is not evidence about the strategy.

---

## 13. Common tasks

| I want to... | Command |
|---|---|
| Run the default backtest (QQQ, iron condor) | `python run_backtest.py` |
| Pick symbols / limit windows | `python run_backtest.py --symbols QQQ SPY --max-windows 2` |
| Optimize per window | `python run_backtest.py --optimize-per-window --optimize-n-trials 50` |
| Run a recorded study | `python run_backtest.py --study-file studies/<name>.yaml` |
| Run a standalone search | `python run_optimizer.py --n-trials 100 --objective composite` |
| Drop, try or re-window a model feature for a study | a study with `ml.features` ([6](#6-the-models-in-research), [MODELS.md](MODELS.md)) |
| See every value and where it comes from | `python scripts/show_research_config.py --section backtest` |
| Test a gate | `python scripts/gate_parity_ab.py --symbols QQQ` |
| Refresh the data | `scripts/backfill_historical_data.py`, `backfill_intraday.py`, `backfill_dividends.py`, `backfill_option_history.py` ([10](#10-research-data-collection-and-calibration)) |
| Look at an archived run in a browser | [WALKFORWARD_DASHBOARD.md](WALKFORWARD_DASHBOARD.md) |

Without `--symbols` / `--strategies`, a run uses `backtest.default_symbols` (QQQ) and `options.strategies`.

---

## 14. Assumptions and known limits

**Modelling assumptions** (what a research number takes for granted):
- **Option prices are modelled**: Black-Scholes from one stored implied vol per day plus a skew, or a QQQ-only fitted surface; real quotes differ.
- **Costs are fitted, not measured per fill**: commission and fees per leg plus a fitted half-spread per leg; short-DTE bands are extrapolated until enough sessions are collected ([10.1](#101-short-dte-friction-collection)).
- **Fills are modelled**: live market data is delayed (`AIT_MARKET_DATA_TYPE=4`), so there is no measured fill to compare except the few parity-replay trades.
- **Decisions see the past only**: entries use history through the previous session (decision frame, [5](#5-the-engine)); the IV used is the previous session's.
- **Expiry is modelled, early assignment is not**: settlement and assignment at expiry follow operator-set assumptions; assignment before expiry (a deep in-the-money short call before an ex-dividend date) is not simulated.
- **One shared account**: all symbols draw on one pool of capital with live's per-trade, aggregate and concurrent-position limits; live-only gates such as the VIX halt are not simulated.
- **The learner's size changes are part of the result** ([7](#7-the-learner)).

**Known limits:**
- **History is short and one regime.** The research calendar starts 2024-01-01. Results describe that period.
- **One symbol has option history.** QQQ is the only symbol with the option bars an IV surface needs, and that history is short (about two years when this was written).
- **The spread model is fitted on a limited sample.** Real fills differ.
- **The learner's disable step is unreachable** with the shipped numbers ([7](#7-the-learner)).
- **A single run is one draw.** Compare studies on the same windows and engine, and weigh differences against window-to-window noise.
- **Research never promotes anything.** A finding changes live only through a registered `PLAN.md` wave and the operator's decision.

---

## 15. The Walk-Forward Lab

A browser dashboard for archived runs (candlesticks with trades, Optuna trials, predictor skill and calibration) has its own guide: [WALKFORWARD_DASHBOARD.md](WALKFORWARD_DASHBOARD.md). Note that it reads the archive format of the older
integration-test pipeline; runs from `run_backtest.py` and `run_optimizer.py` write run records instead (section 9) and do not appear in it today.

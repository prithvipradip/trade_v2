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
| `walkforward_tuning` | Warm start, adaptive range threshold, validation split, inert-parameter probe |

Model hyperparameters (`ml.hyperparams`, `ml.meta_label_hyperparams`) are shared with live, so they are listed in
`config.yaml`; research differs only if you override them. Details: RUNBOOK "Research tunables" and "Model hyperparameters".

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

**Could a research run overwrite the live model?** No. Window models are throw-away (`persist_artifacts=False`) or go to `models/research/`.

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

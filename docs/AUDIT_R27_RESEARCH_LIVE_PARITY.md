# R27 audit — research-to-live parity and component review (2026-11-04)

**Question.** Research exists to vary configurations, settings, predictive models and features. A setting that
shows an edge in backtesting and walk-forward analysis is moved to live: first a paper account, then real
money. For that transfer to mean anything, research must run the same code, on the same inputs, as live. This
audit asks where it does not, and checks the components along the way.

**Result in one paragraph.**
- A lot is already shared: the lot rule, the sizing scalers, the take-profit and DTE-exit constants, the
  ex-dividend entry block and the strict data policy. Two registries (`backtesting/gate_parity.py` and
  `DIVERGENCE_CLASSES` in `config/settings.py`) make most deliberate differences visible.
- The weak links are elsewhere:
  1. What the walk-forward validates is not what gets deployed.
  2. The ML gate research validates is not, input for input, the model live runs.
  3. Several entry and exit rules are second copies that have drifted.
  4. Four rows of `gate_parity.py` that claim parity are false.
- None of this is a live-money bug today. It is a validity problem: an edge measured in research is not yet
  an estimate of what paper or live will earn.

**No code was changed.** The fixes are proposed as waves (section 6) for the operator to order and
pre-register.

---

## 1. Scope and method

- **Read:** the research stack, the live path and the shared pieces.
  - Research stack: `backtesting/engine.py`, `walkforward.py`, `pool.py`, `pricing.py`, `settlement.py`,
    `friction.py`, `replay.py`, `learner.py`, `gate_parity.py`, `optimization/*`, `run_backtest.py`,
    `run_optimizer.py`, `scripts/export_production_params.py`.
  - Live path: `bot/orchestrator.py`, `strategies/iron_condor.py`, `selector.py`, `base.py`, `risk/*`,
    `execution/portfolio.py`, `exit_policy.py`, `condor_defense.py`, `executor.py`.
  - Shared ML and data: `ml/*`, `data/market_data.py`, `macro.py`, `macro_store.py`, `iv_rank.py`,
    `economic_calendar.py`.
  - Config: `config/settings.py`, `config.yaml`, `config_research.yaml`.
  - Plus the PLAN.md history, so that nothing already closed is re-reported.
- **How each finding was established:** three parallel read-only passes (entry path; lifecycle and
  methodology; ML, data and config), then the key claims re-read by hand.
- **The VERIFIED / READ tags:**
  - **VERIFIED:** the cited lines were re-read in this session and show what is claimed.
  - **READ:** from the read-only passes, with a file:line, but not re-read by hand. Re-read the cited code
    before acting on it.
- **Nothing was executed.** No backtest, test suite or study was run. Where a finding says "numbers would
  differ", it gives the direction, not a measured size.

---

## 2. Architecture map: shared or not, step by step

| Step | Live | Research | Status |
|---|---|---|---|
| Daily bars as-of | IB `endDateTime=""`, includes today's partial bar (`data/market_data.py:626`) | previous completed session (`engine.py:1085`, `_dec_lag`) | **DIVERGENT** (B4) |
| Feature frame | `FeatureEngine` with market context + intraday store | `FeatureEngine`; training: intraday, no context; predict: context, no intraday | **SHARED class, DIVERGENT inputs** (B1, B2) |
| Range model | `RangePredictor`, fixed ±5% band, 504-bar span, daily retrain | `RangePredictor`, adaptive 2-15% band, 365-day window | **SHARED class, DIVERGENT spec** (B3, B7) |
| Direction model | ensemble; on failure, no trade | ensemble; on failure, falls back to `_simple_direction` | **SHARED class, DIVERGENT fallback** |
| Multi-timeframe boost | `orchestrator.py` ~2299 | none (W115 planned, never built) | **MISSING** |
| Fractal gate / penalty | `ml.regime_gate.fractal_gate` | same function | **SHARED** |
| Strategy selection | IC on any direction | IC on NEUTRAL only (`condor_any_direction: false`) | **DIVERGENT, undeclared** (C1) |
| Macro blackout | trading sessions + whole pre-event day | calendar days | **DIVERGENT, declared TWIN** (C2) |
| Earnings / ex-div entry | proximity + hold-through / W51 | proximity / `ex_div_blocks` | earnings partial; ex-div **SHARED** |
| IV rank | store percentile, stale ≤5d, realized-vol proxy fallback | `ait.data.iv_rank`, NaN → reject | **SHARED formula, DIVERGENT fallback** (C6) |
| Short strikes | IB smile delta → 0.20, band 0.15-0.30 | Black-Scholes delta at flat ATM IV | **DIVERGENT** (C4) |
| Wing width | per-side IV, budget cap | symmetric, budget cap (`width_sizing`) | cap **SHARED**, formula duplicate |
| Credit floor | `iron_condor.py:357-377` | `engine.py:2888-2896` | duplicate copy, same contract values |
| Expiry choice | best of up to 3 in [14, 30] by credit/width | always `dte_range[0]` | **DIVERGENT** (C5) |
| Entry fill | option-price ladder, scans all day | underlying-price touch test, once a day | **DIVERGENT** (C5) |
| Lot count / scalers | `risk.lot_sizing`, `risk.scalers` | same | **SHARED** (W108, W114) |
| Book caps | risk manager gates 3b-6c | `pool.allocate_pool` | mostly **SHARED**; one-per-symbol and cooldown **MISSING** (C3) |
| Meta-label | bypassed in paper mode; `enabled: false` | applied when windows train models | **DIVERGENT, declared TWIN** (B6, F) |
| Adaptor / Thompson | bypassed in paper mode; on after go-live | research learner (different code) | **DIVERGENT** (A3) |
| Exit constants | `execution/exit_policy.py` | same module | **SHARED** |
| Exit rule order | `portfolio._evaluate_position` | `engine._check_exit_credit` | **DUPLICATE** (D1) |
| Max-hold exit | none | searched 1-9 days | **RESEARCH-ONLY** (A2) |
| Pre-ex-div close, condor defense, rolls | yes (defense / rolls off today) | none | **MISSING** (D2) |
| Exit marks | IB market price; combo ask + $0.10 | Black-Scholes at entry IV or previous-session surface | **DIVERGENT** (D3) |

---

## 3. Headline: what a research edge validates today

1. **A re-optimization process, not the deployed configuration (A1).** Live then runs one snapshot, picked
   by an outcome-conditioned rule.
2. **Parameters live does not read (A2).** `delta_short`, `max_hold_days` and `stop_loss_pct` are searched
   or exported, and live never reads them.
3. **A range gate whose inputs differ from the model it was trained on and from live's (B1-B4).** The 0.65
   floor transfers between different probability scales.
4. **A condor that trades only on neutral days (C1).** Live trades every day the gates pass.
5. **The paper ruleset, not the real-money ruleset (A3).** Going live flips `paper_trading_mode`, which turns
   on three overlays nothing has validated.
6. **Optimistic exits at a touch (D3) and no post-stop cooldown (C3).** Both point the same way: research
   looks better than live.

---

## 4. Findings by component

Severity uses the repo's scale: CRITICAL means it invalidates the research-to-live transfer; HIGH means
numbers differ materially; MEDIUM means they differ in some regimes; LOW is hygiene.

### A. Promotion pipeline

**A1. CRITICAL — the walk-forward validates "re-optimize every quarter"; live runs one frozen snapshot.**
VERIFIED.
- The walk-forward re-runs Optuna for every 63-day window and stitches together the out-of-sample results
  (`walkforward.py` ~1833-1844).
- `scripts/export_production_params.py:88-97` exports `active[-1]["best_params"]`: the last window that had
  any trades. That is a pick conditioned on outcome, and no run scores that fixed configuration out of
  sample.
- Live models are retrained daily on 504 bars (`orchestration/master.py` ~1609); research windows train on
  365 calendar days.
- *Fix:* an out-of-sample run of the one exported configuration, frozen, in the same windows, as the number
  of record for promotion.

**A2. CRITICAL — searched and exported parameters with no live reader.** VERIFIED.
- **`delta_short`:** searched over 0.15-0.30 (`optimization/param_spaces.py:41`). Live uses the constant
  `SHORT_DELTA_TARGET = 0.20` (`strategies/iron_condor.py:29`, used at :248 and :256).
- **`max_hold_days`:** searched over 1-9 (`param_spaces.py:46`) and enforced by the engine
  (`engine.py` ~3919-3948). `execution/portfolio.py` and `exit_policy.py` contain no max-hold rule. An
  out-of-sample profit can therefore rest on an early close that live never makes.
- **`stop_loss_pct`:** maps to `exit.initial_stop_loss_pct`, which credit positions ignore (`portfolio.py`
  ~620). READ.
- *Fix:* the exporter refuses any parameter without a live reader. Remove `max_hold_days` from the condor
  space, or build it live. Give `SHORT_DELTA_TARGET` a config home read by both sides.

**A3. HIGH — `paper_trading_mode` is the parity switch, and the config says to turn it off at go-live.**
VERIFIED.
- `config.yaml` (learning block): `paper_trading_mode: true # ... MUST be set false (or removed) at
  go-live`.
- With it false, live enables:
  - the adaptor confidence override (`orchestrator.py:2248-2252`);
  - the meta-label gate (`:2402`);
  - Thompson re-ranking (`:2693`);
  - adaptor stop, trailing and take-profit overrides (`:3797-3808`).
- The adaptor's symbol, hour and strategy filters (`:1945`, `:1981`, `:2527`) are inert only because W106
  stops adaptations from banking in paper mode.
- Research has its own learner (`backtesting/learner.py`, `backtest.learner`). It is different code with
  different rules, and does not stand in for `learning/adaptor.py`.
- The real-money bot would therefore run a ruleset that neither research nor paper measured.
- *Recommendation:* keep the overlays off at go-live, as a decision recorded in PLAN.md, until a study turns
  each one on in research. Rename the switch to say what it does (for example `learning.overlays_enabled`).

**A4. MEDIUM — the go-live gate.**
- VERIFIED: `reporting/go_live.py:655` has `passed=(s.pf is None or s.pf > GATE_PF)`, so a sample with no
  losing trade passes the profit-factor gate.
- READ:
  - The gate compares paper against fixed thresholds, not against the research expectation for the same
    period. `scripts/paper_checkpoint.py` does that comparison, but outside the gate.
  - The slippage criterion is permanently UNAVAILABLE, because executions carry no entry/exit marker.

### B. ML and features

**B1. CRITICAL — research range model: trained with intraday features, scored without them.** VERIFIED.
- Training: `walkforward.py:3324` calls `rp.train(train_df, symbol=symbol, intraday_store=intraday_store)`.
- Scoring: `engine.py:1335-1340` calls `predict(hist_dec, symbol=..., market_context=...,
  min_edge_override=...)` with no `intraday_store`. The direction model is the same at `engine.py` ~2091.
- `ml/range_predictor.py:510` does `features.reindex(columns=feature_names).fillna(0.0)` and then scales.
  Every intraday column arrives as 0.0, which is off the distribution the model saw in training.
- Live passes the store on both sides (`orchestrator.py` ~2227, ~2542; `ml/trainer.py` ~277).
- The research probabilities that the 0.65 floor and the range studies rest on come from corrupted inputs.
  It bites whenever a window trains with intraday fractal or VLMC features (`vlmc_min_coverage`).

**B2. HIGH — market context differs at training.** READ.
- The research window trains without market context (`walkforward.py:3324`), so the cross-asset and macro
  columns are constants (`ml/features.py` ~776-791, ~885-888).
- The live retrain passes `market_context` (`trainer.py` ~277-279).
- Same class, different model.

**B3. HIGH — range label band differs.** READ, and documented in the `ml/range_spec.py` docstring.
- Research labels "in range" with an adaptive band from realized vol, clipped to 2-15%
  (`walkforward.py` ~3021-3064, ~3290).
- Live uses a fixed `ml.range_threshold_pct = 0.05`.
- The same `ml.range_min_confidence = 0.65` is applied to both, but the two probabilities answer different
  questions.

**B4. HIGH — as-of difference.** VERIFIED.
- Research (realistic engine) decides on the previous completed session (`engine.py:1078-1086`).
- Live builds features from IB daily bars requested with `endDateTime=""` (`data/market_data.py` ~626-643).
  Those include today's in-progress bar.
- `drop_incomplete_session` is applied only when writing to the store (`ml/trainer.py:266`, `data/iv_refresh.py`).
  It is not applied before predict, and not to the frame the live retrain trains on.
- Not look-ahead, but live's features (session shape, closing-hour columns, returns) are computed on a
  half-finished bar that no training row resembles.

**B5. HIGH — macro data timing.** READ.
- Research lags FRED series by one session (`data/macro_store.py`, `lag_one_session`).
- Live training forward-fills raw observation dates onto the same date (`data/macro.py` `fetch_series` →
  `features.py` ~903). Live training rows therefore see values not yet published on that date.
- Live inference is effectively lagged, so live is inconsistent with itself. Research is the consistent side.

**B6. HIGH, latent — meta-labeler inputs on different scales.** READ.
- Live context (`orchestrator.py:2416-2426`): raw VIX (`vix or 0`), the 0-100 store IV rank, `RegimeDetector`
  flags.
- Research context (`engine.py` ~1165-1176, `ml/meta_label.py` ~493-497): `vix_level` (VIX/40), the
  FeatureEngine `iv_rank` (a 0-1 realized-vol rank), `research_regime` flags.
- A research-trained meta model is therefore not portable. This is masked today, because live never applies
  meta-labelling (paper mode, and `meta_label.enabled: false`).

**B7. MEDIUM — training span and warm-up.** READ.
- Research windows train on `train_days: 365` calendar days (~250 bars) with no warm-up prefix. The
  100-period long SMA then drops ~100 rows at `dropna` (`features.py` ~147), and the 252-bar ranks run on
  truncated history.
- Live trains on `ml.lookback_days: 504` bars.
- *Fix:* prepend `spec.longest_lookback()` bars to `train_df`, as is already done for the out-of-sample
  context.

**B8. LOW/MEDIUM — smaller items.** READ.
- **`spy_rsi` neutral value on three scales:** `50.0` with no context (`features.py` ~789), `0.5` with
  context but no SPY (~858), and real values `RSI/100` (~847).
- **Hard-coded vol-magnitude spec:** the daily retrain builds `VolMagnitudePredictor(threshold_pct=0.07,
  horizon_days=30)` (`master.py` ~1619), while the bot reads `ml.vol_magnitude_*`. Equal today; it drifts on
  the first config edit.
- **Model-spec check at load:** `ml/hyperparams.py` `feature_spec_refused` hashes the feature windows and
  switches only. It does not record whether intraday data or market context was used, or the column count.
  Combined with the zero-fill, B1, B2 and the direction-model input mismatch pass silently.
- **Direction CV purge:** `direction_cv_gap_days` is not tied to `direction_label_horizon_days`, and the
  fallback `TimeSeriesSplit` has no gap (`ml/ensemble.py` ~736). The range model does enforce
  `max(gap, horizon)` (`range_predictor.py` ~384).
- **Direction-model inputs differ:** windows train it with intraday features; the live retrain does not
  pass intraday data (`trainer.py` ~268).
- **No stock-split handling:** daily storage is append-only (`historical.py` ~487-555), so the stored
  history and fresh, split-adjusted IB bars disagree after a split. Irrelevant for QQQ/SPY/IWM today.
- **Two homes for the edge threshold:** `backtest.min_edge_over_baseline` and
  `ml.range_min_edge_over_baseline` (0.05 each today). Also, `predict_from_features` ignores
  `min_edge_override`.

### C. Entry path

**C1. HIGH — research opens a condor only on a NEUTRAL direction call; live on any direction.** VERIFIED.
- `config_research.yaml`: `condor_any_direction: false`.
- `engine.py:2176`: `if direction == SignalDirection.NEUTRAL: candidates = available & CREDIT_STRATEGIES`;
  the any-direction branch is behind the switch.
- Live's `IronCondor.generate_signals` ignores `market_direction`.
- The engine's own docstring says the neutral rule removes ~81% of days. Those days are recorded as
  `no_strategy`, which `gate_parity.py` files as MECHANICS ("selector returned nothing").
- It is in fact the largest research-only entry gate, and neither registry shows it.
- The 2026-10-10 operator decision to keep it off stands. It must be registered as RESEARCH_ONLY with its
  switch, or the research condor sample is not the live one.

**C2. HIGH — macro blackout counts different units.** VERIFIED.
- Research blocks when `days_until_next_event(today) <= blackout`, in calendar days (`engine.py:1232-1246`).
- Live has two layers:
  - It converts to trading sessions (`orchestrator.py:3534-3544`, `_sessions_until`; R16 fixed exactly
    this in live).
  - It also skips the whole cycle on event days and the day before (`should_skip_trading`, ~1990).
- Example: a Friday before a Monday CPI gives research d2e = 3, so it trades; live blocks.
- The `macro_event` TWIN row is false. *Fix:* one pure session-based function in
  `data/economic_calendar.py`, called by both sides.

**C3. HIGH — two live gates missing from research and from both registries.** VERIFIED.
- **One condor per symbol:** `risk/manager.py:466` rejects a "duplicate position" (same symbol and
  strategy). Research's engine allows `backtest.max_concurrent_positions` (3) per symbol, and the pool caps
  by cluster only. Research can hold two QQQ condors where live holds one.
- **Post-stop cooldown:** one session for the symbol and 120 minutes for cluster siblings
  (`orchestrator.py` ~136, ~5045). Nothing in `backtesting/` has a cooldown. Research re-enters the day
  after a touch stop, which is the autocorrelated-loss case the cooldown exists for. In the paper
  checkpoint model, touch stops are 46-52% of exits.
- *Fix:* both become pure gates in `pool.allocate_pool` (and in the engine, for the single-symbol path),
  with `POOL_REJECT_REASONS` rows.

**C4. MEDIUM — short strike selection.** READ.
- Research: `find_strike_by_delta(S, t, iv, -delta_short, ...)` at a flat ATM IV (`engine.py` ~2782), then
  prices the legs on the skew (~2856-2866).
- Live: picks the contract nearest 0.20 delta from IB model greeks, which include the smile, within a
  0.15-0.30 band (`strategies/base.py` ~237-255, `iron_condor.py` ~268).
- Under put skew, research's 0.20-delta put sits nearer the money than live's, so research sells a richer
  put than live would.
- *Fix:* pick strikes from surface-implied deltas when `pricing_source: surface`.

**C5. MEDIUM — entry fill and expiry choice.** READ.
- **Fill:** `engine.py` ~2046 fills when the underlying's 5-minute bar range touches the 10:30 underlying
  price within the timeout. It is a test on the underlying, not on the option order, and it conditions
  entries on short-term mean reversion (which favours condors).
- **Live:** works an option-price ladder and rescans every 5 minutes.
- **Expiry:** research always uses `listed_expiry(today, dte_range[0])`. Live ranks up to three expiries in
  [14, 30] by `eff_conf + 0.3 x credit/width` (`orchestrator.py` ~2773; `options_chain.py` ~767).
- `DIVERGENCE_CLASSES["options.dte_range"]` says "research_reads_loaded", which hides that research uses
  only `[0]`.

**C6. MEDIUM — IV-rank fallbacks.** READ.
- Live has two fallbacks research does not:
  - It treats a stored IV up to 5 days old as current.
  - It drops to a realized-vol min-max proxy (`orchestrator.py` ~6440-6500).
- Research returns NaN and rejects `iv_rank_unknown`, so that row's "fails closed" is only true for live's
  hard-failure branch.
- Live also writes today's intraday chain median IV into the `daily_prices.implied_vol` column research
  reads, so recent research dates mix intraday snapshots with backfilled values.

**C7. LOW — unregistered live-only entry gates.** READ.
- The capital-tier universe filter.
- The adaptor hour, symbol and strategy filters.
- The daily trade budget.
- The post-restart settle period (`orchestrator.py` ~1944-1985, ~3584, ~3599).
- All are inert or rarely binding today; they belong in `LIVE_ONLY_GATES` so the next change is seen.

**C8. LOW — engine correctness notes.** READ.
- **Legacy look-ahead:** with intraday data and a non-realistic engine, `_dec_lag = 0` decides a 10:30 fill
  on day D's close. The research overlay sets realistic, so only bare engines are exposed.
- **No intraday bars on a day (realistic):** the entry is silently priced at the daily close with same-day
  IV, and no reason is recorded (`engine.py` ~1479).
- **Shadow builders:** the wide-wing, broken-wing, butterfly and spread builders return None without a
  `builder:` reason, and ignore realistic time and the surface.
- **Stale IV:** non-strict `_get_iv` carries `implied_vol.dropna().iloc[-1]` forward without a staleness
  limit (~2268).

### D. Lifecycle and exits

**D1. HIGH — the exit rule ORDER exists twice.** READ.
- `exit_policy.py` shares the constants: the take-profit ladder, the DTE ≤ 5 close and the macro-flatten
  windows.
- The ordering of touch stop, take-profit, DTE, max-hold, ex-div and defense is hand-written twice:
  `engine._check_exit_credit` (~3837-3993) and `portfolio._evaluate_position` (~476-1209).
- *Fix:* a pure `exit_policy.decide(snapshot, marks, spot, dte, now, cfg) -> Verdict` holding the full
  order. `portfolio.py` feeds it IB marks and `engine.py` feeds it modelled marks.

**D2. MEDIUM — exits research does not simulate.** READ.
- The pre-ex-dividend short-call close (rule 3e, `portfolio.py` ~1021-1084).
- Condor defense, rolls and synthetic delta. They are off live today (`policy: close_only`), so parity holds
  only while they stay off. Turning them on live without a research twin repeats A3.
- The post-stop cooldown (C3).
- The parity replay's reason map (`replay.py` ~713-727) has no row for these live exits.

**D3. HIGH — exit marks ignore same-day IV expansion.** READ.
- Intraday re-marks price the position at entry IV or at the previous session's surface (`engine.py`
  ~2009, ~3435, ~3471).
- On a selloff that triggers the touch stop, that understates the buy-back cost. Live pays the real combo
  ask plus `exit_cross_amount` ($0.10).
- The bias points the same way as C3: research's stops are cheaper than live's.
- *Fix:* scale the exit IV by the same-day move (the skew model already has the slope), or use the day-D
  surface where it is stored. Then calibrate against `fill_vs_model_report`.

### E. Walk-forward and optimizer methodology

**E1. HIGH — Optuna trials are scored with in-sample gate probabilities.** READ.
- The window's range model is fitted on `train_df` (`walkforward.py` ~1766-1788), and the trials are scored
  on that same `train_df` with that model (~2514, ~2538).
- Gate-related thresholds are therefore tuned against overconfident probabilities. `optimize_val_split` is
  off by default, and even when on, the model has already seen the slice.
- *Fix:* score trials with out-of-fold probabilities (the inner CV already produces folds).

**E2. MEDIUM — metric construction.** READ.
- **Sharpe:** computed on per-trade dollar P&L, annualized by √(trades per year), measured between the
  first and last entry (`result.py` ~351-358). Overlap is ignored and short windows are inflated.
- **Max drawdown:** realized-only, walked in entry-date order (`pool.py` ~204, `result.py` ~94-109), so it
  understates the drawdown live would see.
- **Composite objective:** adds `win_rate_weight x win_rate` (`optimization/objectives.py` ~37-39), and the
  global-best warm start ranks by win rate. Both push selection toward high-win-rate, short-gamma
  configurations, which is the profile most exposed to the C3/D3 biases.
- *Fix:* a mark-to-market daily equity curve per run, with Sharpe and drawdown from it.

**E3. MEDIUM — the pool is a post-hoc filter.** READ. `allocate_pool` can only remove per-symbol candidates.
A trade live would have taken in place of one the per-symbol engine skipped (because of its own cap) never
appears.

**E4. MEDIUM — results depend on the worker count.** READ.
- With `optimize_n_jobs > 1`, warm starts and the learner are dropped (`walkforward.py` ~1483-1591), so the
  same study gives different numbers at different worker counts.
- The meta-labeler is unseeded (`ml/hyperparams.py` ~99).
- Record `n_jobs` in the run record as part of the configuration identity.

**E5. LOW — multiple-testing control.** READ. Holm correction exists only in `fractal_ablation_study.py`
and `surface_feature_study.py`. The main guard is pre-registration enforced in code (`study_cli.py`
`assert_registered`). Within a run, best-of-N trials per window has no deflation, which is acceptable only
because A1's frozen out-of-sample run would be the number of record.

**Negative result: no label leakage at the window boundary.** READ.
- Labels are built inside `train_df`, and the last `horizon` rows become NaN and are dropped
  (`range_predictor.py` ~201-236).
- The inner CV purges at least the horizon.
- The outer `gap_days: 5` is shorter than the horizon, but that is harmless given the truncation.

---

## 5. Registry corrections (`backtesting/gate_parity.py`, `config/settings.py`)

| Row | Claims | Reality | Correction |
|---|---|---|---|
| `macro_event` | TWIN | calendar days vs sessions + whole pre-event day (C2) | fix the code (shared function); until then RESEARCH_ONLY-like note |
| `meta_label` | TWIN | live never applies it (paper mode; `enabled: false`); inputs on other scales (B6) | mark as research-only in practice; record the scale gap |
| `iv_rank_unknown` | "fails closed" | live has a stale-value and realized-vol proxy fallback (C6) | state the fallback |
| `no_strategy` | MECHANICS | mostly the neutral-only condor rule (C1) | split out `condor_neutral_only` as RESEARCH_ONLY, switch `condor_any_direction` |
| — (missing) | — | one condor per symbol; post-stop cooldown (C3) | add to `POOL_REJECT_REASONS` / `LIVE_ONLY_GATES` until shared |
| — (missing) | — | C7 gates; the multi-timeframe boost (W115 never built) | add to `LIVE_ONLY_GATES` |
| `DIVERGENCE_CLASSES["options.dte_range"]` | research_reads_loaded | research uses `[0]` only (C5) | deliberate_research_difference |

`DIVERGENCE_CLASSES` compares `config.yaml` with code defaults and skips nested models. It never compares
`config_research.yaml` with `config.yaml`, nor `ml.features.*`, `backtest.learner` or `walkforward_tuning`. The
`settings_origin` diff in `config/research.py` is the actual research-versus-live view. The startup report
should say so rather than imply completeness. Also, the research-only `AIT_BT_*` switches and `AIT_LIQ_*` are
raw `os.environ` reads outside `CONTRACT_DEFAULTS` (`engine.py` ~653, ~791, ~810; `orchestrator.py` ~396).
READ.

---

## 6. Recommendations: same code wherever possible

The guiding rule: **a decision is one pure function, fed by two adapters** (live: IB data; research:
stored data and models). The shared lot rule (W108) and scaler (W114) already prove the pattern. The proposed
waves follow, in recommended order, each to be pre-registered (AS PLANNED) before any code.

- **R27-1 Transfer integrity (A1-A4).**
  - An out-of-sample run of the frozen exported configuration as the promotion number of record.
  - The exporter refuses parameters without a live reader.
  - `max_hold_days` out of the condor space, or built live.
  - `SHORT_DELTA_TARGET` given a config home read by both sides.
  - A recorded decision that go-live keeps the learning overlays off.
  - The `go_live.py` profit-factor gate fixed (no losses is not a pass on its own; require the close count).
- **R27-2 One feature-frame builder (B1-B5, B7, B8).**
  - `ml/features.py` gains `build_feature_frame(df, *, symbol, store, context, as_of, decision_lag)`. It
    drops the incomplete session, applies the decision lag and the one-session macro lag, and always takes
    the intraday store and context. Live train, live predict, window train and engine predict all call it.
  - The model spec records the input configuration (intraday yes/no, context keys, column count), and
    `predict` refuses a mismatch instead of zero-filling.
  - One range-predictor factory with one band rule for both sides. The operator picks fixed or adaptive.
- **R27-3 Shared entry gates (C1-C3, C6, F).**
  - A session-based blackout function.
  - One-per-symbol and the post-stop cooldown as pool and engine gates.
  - `condor_any_direction` registered or switched.
  - The registry rows corrected.
- **R27-4 One exit decision (D1-D3).** `exit_policy.decide()` holds the full rule order. It brings rule 3e
  and condor defense into research, and an exit-IV model scaled by the same-day move.
- **R27-5 Methodology (E1-E4, B6, C4, C5).**
  - Out-of-fold gate probabilities in trials.
  - A mark-to-market daily equity for Sharpe and drawdown.
  - A shared meta-label context builder (raw VIX, 0-100 IV rank, one regime source).
  - Surface-delta strike selection.
  - W115 (the multi-timeframe port).
  - `n_jobs` recorded in the configuration identity.

**Until R27-1 and R27-2 land, an edge found in research is evidence about the research engine, not a
forecast of paper results.** The paper checkpoint (`scripts/paper_checkpoint.py`) remains the honest check.

---

## 7. What is in good shape (keep)

- **One lot rule and one scaler implementation** for live, engine and pool (`risk/lot_sizing.py`,
  `risk/scalers.py`, `risk/width_sizing.py`).
- **Shared exit constants:** take-profit ladder, DTE close and macro-flatten windows (`execution/exit_policy.py`).
- **Ex-dividend entry block:** `backtesting/ex_div.py` mirrors W51.
- **Fractal gate:** one function, `ml/regime_gate.fractal_gate`.
- **Strict research data policy and preflight (W116):** refuse, do not substitute.
- **Research models isolated in `models/research/`**, never overwriting the live `models/*.pkl`.
- **Pre-registration enforced in code (`assert_registered`)** and the AS PLANNED / AS BUILT discipline in
  PLAN.md. This is the strongest protection the project has against selection bias.
- **Causal cross-window learning:** the learner and warm starts read only earlier windows' out-of-sample
  results.
- **Registries with guard tests** (`gate_parity`, `DIVERGENCE_CLASSES`): the right mechanism. This audit's
  corrections are additions to them, not a replacement.

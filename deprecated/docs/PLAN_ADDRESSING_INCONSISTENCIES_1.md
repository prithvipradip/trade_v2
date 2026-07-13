> [!WARNING]
> **STALE — DO NOT TRUST (banner added 2026-07-08, Round 5 audit).**
> This document predates the July 2026 audit rounds. Specifically:
> all quoted performance numbers (Sharpe 8-24, +183% OOS, etc.) came from
> backtest math since proven wrong (sleeve-capital inflation, sqrt(252)
> annualization, window overlap); the credit-sizing formula described here
> was a 4-5x risk understatement bug; the delta gate / hedging described as
> active are DEAD (no greeks feed); config numbers (universe, DTE, caps,
> confidence) no longer match config.yaml. **PLAN.md is the only current
> source of truth.** Structural lessons remain useful; numbers do not.

# Plan: Gap Analysis + Test Cases for ADDRESSING_INCONSISTENCIES_1.md

## Context

`ADDRESSING_INCONSISTENCIES_1.md` defines 6 fixes (Fix 0–6) to close structural
gaps between the walk-forward backtest and the live production bot. This document
surfaces concrete omissions in that plan discovered by reading the actual code
paths involved, adds additional live-vs-backtest gaps (Z1–Z9), identifies code-level
gaps in the existing plan (A–I), adds a complete test suite specification, documents
irreducible residual gaps (R1–R5), and provides a verification checklist.

---

## Implementation Status (branch: features-request-2)

Gaps implemented in this sprint:

| Gap | Status | Key files changed |
|---|---|---|
| **Gap Z9** — `build_training_data()` only 9/20 features | ✅ Done | `src/ait/ml/meta_label.py` — `build_training_data_from_backtest()` populates all 20 META_FEATURES |
| **Gap Z1 (long-term)** — MetaLabeler per-window training | ✅ Done | `src/ait/backtesting/walkforward.py` — `_train_window_meta_labeler()` + wired into run loop |
| **Gap Z1 (OOS gate)** — MetaLabeler applied during OOS evaluation | ✅ Done | `src/ait/backtesting/engine.py` — `meta_labeler` param + gate after min_confidence check |
| **Gap I** — `entry_time`/`exit_time` missing from window JSON | ✅ Done | `walkforward.py` — `_write_window_progress()` now emits `trades_detail` array |
| **Gap F** — `exit_time` missing from position dict exit paths | ✅ Done | `engine.py` — `_check_thesis_invalidation`, `_force_close` both set `exit_time` |
| **Gap Z6** — Thesis re-evaluation exit missing from backtest | ✅ Done | `engine.py` — `_check_thesis_invalidation()` implemented |
| **Gap A** — `get_feature_names()` excludes VLMC | ✅ Done | `features.py` — `get_feature_names(include_vlmc=True)` added |
| **Gap C** — `implied_vol` schema + load path | ✅ Done | `historical.py` — DDL migration, `save_daily_iv`, `load_daily_iv` |
| **Gap D** — `slice_intraday_up_to()` missing | ✅ Done | `historical.py` — static method added |
| **Gap E** — `param_spaces.py` missing spread params | ✅ Done | `param_spaces.py` — `spread_base`, `spread_iv_sensitivity`, etc. |
| **Gap G** — `export_production_params.py` `_PARAM_MAP` incomplete | ✅ Done | `scripts/export_production_params.py` |
| **Gap H** — New params in wrong config class | ✅ Done | `settings.py` — params in `WalkForwardConfig` + `LearningConfig.paper_trading_mode` |

Gaps not yet implemented (pending Fix 0 data backfill or separate sprint):

| Gap | Status |
|---|---|
| Gap Z2 — Thompson Sampling absent from backtest | Pending |
| Gap Z3 — Market context absent from predictors in backtest | Pending |
| Gap Z4 — Earnings skip absent from backtest | Pending |
| Gap Z5 — Hurst/fractal penalty missing from live orchestrator | Pending |
| Gap Z7 — Options flow gate absent from backtest | Pending |
| Gap Z8 — Adaptor exit overrides not gated in paper_trading_mode | Pending |
| Gap B — `features_cache` computed without `intraday_store` in Optuna loop | Pending |

**Test suite:** 9 files, 99 tests passing — see `tests/test_backfill_historical_data.py`, `test_intraday_engine.py`, `test_intraday_regression.py`, `test_iv_model.py`, `test_paper_trading_mode.py`, `test_paper_trading_overlay_parity.py`, `test_spread_model.py`, `test_vlmc_session_tiering.py`, `test_vlmc_training_integration.py`.

---

## Phase 0 — Additional Live-vs-Backtest Gaps Not in ADDRESSING_INCONSISTENCIES_1.md

These were discovered by reading the actual orchestrator signal→execution path and
comparing it gate-by-gate to the backtest engine. They are separate from the data and
feature inconsistencies (Fixes 0–6) and must be addressed for paper-trading P&L to
be comparable to backtest P&L.

---

### Gap Z1 — Meta-labeler in live, absent from backtest (HIGH IMPACT)

**What it does:** The live orchestrator runs a trained `MetaLabeler` at
`orchestrator.py:728–778`. It takes 19 features (primary confidence, regime,
VIX, IV rank, RSI, BB position, volume, trend alignment, MACD, hour of day, etc.)
and makes a binary take/skip decision. If `take_trade = False`, the symbol is
skipped entirely and a counterfactual is logged. In practice this filters
15–30% of signals.

**Backtest:** No meta-labeler. Every signal above `min_confidence` proceeds.

**Impact:** Backtest will have more trades with a lower average quality than live.
The live bot's P&L curve will be smoother (fewer, higher-quality trades); the
backtest's will show more noise. Sharpe and win-rate comparisons will be skewed.

**Interim fix (paper trading validation):**
Add `paper_trading_mode` to disable the meta-labeler gate in live:
```python
if self._meta_labeler is not None and self._meta_labeler.is_trained:
    if not self._settings.learning.paper_trading_mode:
        meta_signal = self._meta_labeler.predict(meta_context)
        if meta_signal is not None and not meta_signal.take_trade:
            return
```

**Chosen long-term fix (selected):** Train meta-labeler per walk-forward window on the
training set's signal outcomes. Include as a trained artifact alongside the direction
predictor. The implementation requires:

1. **Signal outcome labeling:** During each walk-forward window's training pass, record
   every signal above `min_confidence` on the training set. Label each signal with
   `profitable=1` if the corresponding trade closed in profit, `0` otherwise.
   The label set lives entirely in the training period — no OOS leakage.

2. **MetaLabeler feature set (20 features):** Fix `build_training_data()` in
   `src/ait/learning/meta_labeler.py` — currently only 9/20 features are populated;
   RSI, BB, MACD, volume, trend_alignment, hour_of_day, and 4 others are hardcoded to 0.0
   (root cause of the current "trained on corrupted data" disabled state). Rewrite to
   compute all 20 features from historical OHLCV + intraday store (same FeatureEngine
   used for the direction predictor — no new infrastructure needed).

3. **Walk-forward integration (`walkforward.py:_train_window_model()`):**
   After training the direction predictor, run a signal replay on the training window,
   collect labeled signal outcomes, train the MetaLabeler on those outcomes, and save
   it as `meta_labeler.pkl` alongside `model.pkl` and `feature_names.pkl`.

4. **Walk-forward OOS evaluation:** During the OOS evaluation pass, load the trained
   MetaLabeler and apply it as a filter on signals — consistent with how it operates
   in live. Trade count and Sharpe in the OOS result will now reflect meta-labeler
   filtering, directly comparable to live performance.

5. **New files to modify:** `src/ait/learning/meta_labeler.py` (fix `build_training_data()`),
   `src/ait/backtesting/walkforward.py` (add meta-labeler training step and OOS filter),
   `src/ait/backtesting/engine.py` (accept and apply `meta_labeler` param during OOS evaluation).

---

### Gap Z2 — Thompson Sampling in live, absent from backtest (HIGH IMPACT)

**What it does:** `orchestrator.py:913–919` calls
`self._thompson.rank_strategies(strategy_names)` to reorder candidate signals by
Thompson-estimated win rate. The sampler learns from real trade outcomes
(`orchestrator.py:473–477`). A strategy on a losing streak gets demoted; a
strategy on a winning streak gets promoted. Over time the live bot concentrates
on whatever is working.

**Backtest:** Purely deterministic. `_select_strategy()` always picks iron_condor
(hardcoded preference at `engine.py:434–437`). The same strategy is selected for
the same market conditions across every run.

**Impact:** In a walk-forward run dominated by iron condors, the backtest is
self-consistent. In live paper trading, Thompson sampling may shift the bot
away from iron condors after a few losses (or double down after wins). The
strategy mix will diverge from the backtest's fixed iron_condor-only behavior.

**Fix for paper trading validation:**
In `paper_trading_mode`, bypass Thompson ranking:
```python
if not self._settings.learning.paper_trading_mode:
    ranked_names = self._thompson.rank_strategies(strategy_names)
    signals = [sig_map[n] for n in ranked_names if n in sig_map]
```
(Thompson still records outcomes for future use — just doesn't reorder during validation.)

---

### Gap Z3 — Market context absent from range predictor and ML predictor in backtest (MEDIUM IMPACT)

**What it does:** Live orchestrator calls:
```python
range_pred = self._range_predictor.predict(
    hist, market_context=market_context,   # VIX, SPY, macro
    live_signals=live_signals,
    intraday_store=self._historical,
)
```

The range predictor's `FeatureEngine.compute()` call receives cross-asset
features (VIX regime, SPY relative strength, yield curve, DXY). The walk-forward
backtest's range predictor is called as:
```python
rp = self._range_predictor.predict(hist)   # engine.py:199 — no context
```

Same gap exists for the ML direction predictor at `engine.py:369`:
```python
pred = self._predictor.predict(hist)   # no market_context
```

**Impact:** Range predictor and direction predictor in the backtest operate
without cross-asset information that the live bot uses. Features like
`vix_percentile`, `spy_relative_strength`, `yield_curve_slope` are present at
live inference but absent from backtest evaluation.

**Fix:** Pass `market_context` (pre-fetched once per walk-forward window, same
as how the walk-forward's training pass works) to both `predictor.predict()`
and `range_predictor.predict()` inside `engine.py`. The walk-forward already
fetches cross-asset data — it just doesn't forward it to the engine's evaluation.

---

### Gap Z4 — Earnings/economic event skip in live, absent from backtest (MEDIUM IMPACT)

**What it does:** Live orchestrator skips symbols near earnings (`orchestrator.py:718–720`).
Iron condors entered 2–3 days before earnings face IV crush on announcement
followed by a large gap move — worst-case scenario for an iron condor.

**Backtest:** No earnings awareness. Backtest will trade through earnings dates,
collecting what appears to be good premium but facing outsized risk. This
systematically inflates backtest win rate on high-IV pre-earnings periods.

**Fix for paper trading validation:**
Disable earnings skip in live during paper trading for cleaner comparison with
the backtest, OR add earnings date awareness to the backtest engine.

Adding to backtest is the correct long-term fix: the backtest should load a
historical earnings calendar (available from Yahoo Finance via yfinance's
`get_earnings_dates()`) and skip entry signals within N days of earnings.

---

### Gap Z5 — Hurst/fractal regime penalty in backtest, MISSING from live (MEDIUM IMPACT)

**What it does:** The backtest applies a confidence penalty when
`hurst_scale_spread > hurst_regime_threshold` or `multifractal_width > multifractal_max_width`
(`engine.py:167–179`). This makes the backtest MORE conservative in chaotic
fractal regimes.

**Live bot:** Does NOT apply this penalty. It uses the fractal features
(hurst_scale_spread, multifractal_width) as ML input features, but the
explicit confidence penalty gate is absent from the live orchestrator.

**Impact:** Backtest is more selective during chaotic periods than live. During
high-fractal-chaos periods, the live bot will enter trades that the backtest
skips. Backtest win rate is inflated relative to live for this regime.

**Fix:** Add the fractal regime confidence penalty to the live orchestrator's
signal confidence computation, immediately after the ML prediction and before
the min_confidence threshold check. Use the same thresholds exported by Optuna.

---

### Gap Z6 — Thesis re-evaluation (mid-trade exits) in live, absent from backtest (MEDIUM IMPACT)

**What it does:** The live orchestrator checks mid-trade at `orchestrator.py:420–444`
whether the original signal thesis still holds. If the direction predictor
strongly reverses (e.g., entered bearish condor, now strongly bullish), or if
the regime shifts from RANGE_BOUND to HIGH_VOLATILITY, the position is flagged
for early exit (`pos.should_exit = True`).

**Backtest:** Exits ONLY on stop-loss, profit target, DTE expiry, or force-close
at backtest end. No mid-trade thesis invalidation.

**Impact:** Live bot exits some positions early (before stop-loss) when the
market conditions change. Backtest holds until stop or target. For iron condors
where the underlying trends strongly against one wing, early thesis exit may
produce a smaller loss than waiting for the stop to trigger.

**Fix:** Add thesis-invalidation exit logic to the backtest execution layer.
At each intraday execution bar (or daily bar), re-evaluate the ML prediction.
If the current prediction strongly contradicts the position's original direction
(e.g., entered with NEUTRAL signal, now getting STRONGLY_BULLISH with
confidence > 0.80), trigger an exit at the current bar's price.

---

### Gap Z7 — Options flow hard gate in live, absent from backtest (LOWER IMPACT)

**What it does:** `orchestrator.py:809–819` checks if institutional options flow
disagrees with the ML direction by > 0.7 strength. If so, the signal is rejected.
This is a real-money overlay that uses real-time options order flow data.

**Backtest:** No options flow data. Cannot be replicated historically without
sourcing historical flow data (Polygon, CBOE).

**Fix for paper trading:** Disable options flow gate in `paper_trading_mode`
(equivalent to backtest having no flow awareness). Long-term, source historical
flow data from Polygon and add as a backtest input.

---

### Gap Z8 — Adaptor stop/trailing-stop/take-profit overrides in live, absent from backtest (MEDIUM IMPACT)

**What it does:** `StrategyAdaptor` applies not just a confidence override, but also
`get_stop_loss_override()`, `get_trailing_stop_override()`, and
`get_take_profit_override()` in the live orchestrator. These are self-learned per-symbol
adjustments derived from real closed trades. For a symbol with 10 recent losses, the
adaptor may tighten the stop loss from 35% to 20%; for a symbol on a winning streak,
it may widen the profit target.

**Backtest (walk-forward):** None of these overrides are applied. The walk-forward
Backtester always uses the Optuna-optimised fixed values (e.g., `stop_loss_pct=0.35`).

**Impact:** In live paper trading, the adaptor's stop/trailing/take-profit adjustments
will silently change exit behavior from day one of paper trading, even if the adaptor
has no paper-trade history (it may carry over state from prior live runs or be
initialized with defaults that differ from backtest values). Exit timing and P&L
distribution will diverge from the backtest.

**Fix:** Extend `paper_trading_mode = true` scope to freeze ALL adaptor overrides
(not just confidence). In `orchestrator.py`, gate each override lookup:
```python
stop_loss = (
    self._adaptor.get_stop_loss_override(symbol)
    if not self._settings.learning.paper_trading_mode
    else None
) or self._settings.exit.initial_stop_loss_pct
```
Apply the same pattern for `trailing_stop_override` and `take_profit_override`.

---

### Gap Z9 — MetaLabeler `build_training_data()` populates only 9 of 20 features (PREREQUISITE for Gap Z1 fix)

**Root cause of current disabled state:** `src/ait/learning/meta_labeler.py`
`build_training_data()` constructs the 20-feature training matrix, but 11 features
(`rsi`, `bb_position`, `macd_signal`, `volume_ratio`, `trend_alignment`, `hour_of_day`,
and 5 others) are hardcoded to `0.0` because they require OHLCV computation that was
never wired up. The model trains on a matrix that is mostly zeros, learns to reject
everything (since real trades with real feature values all look like outliers), and
is then disabled with the comment "trained on corrupted data, rejects everything."

**Fix (prerequisite for Gap Z1 long-term fix):**
Rewrite `build_training_data()` to compute all 20 features from the same `hist` DataFrame
and `intraday_store` that the direction predictor uses:
- Reuse `FeatureEngine().compute(hist, intraday_store, symbol)` — same call already
  made in `_train_window_model()`; pass the result down to MetaLabeler training.
- Map the 20 MetaLabeler feature names to columns in the FeatureEngine output.
- The signal outcome label (`profitable`) comes from the walk-forward training set's
  simulated trade outcomes (already computed in the Optuna loop — wire through).

---

### Consolidated `paper_trading_mode` scope

Fix 6a in the existing plan adds `paper_trading_mode` only for the confidence
override. Given the above gaps, `paper_trading_mode = true` should disable ALL
live-only overlays to produce a clean backtest-equivalent paper run:

| Overlay | Disabled in paper_trading_mode |
|---|---|
| Adaptor confidence override | Yes (Fix 6a) |
| Adaptor stop_loss_override | Yes (Gap Z8) |
| Adaptor trailing_stop_override | Yes (Gap Z8) |
| Adaptor take_profit_override | Yes (Gap Z8) |
| Meta-labeler gate (interim) | Yes (Gap Z1 interim) → replaced by per-window training |
| Thompson Sampling reranking | Yes (Gap Z2) |
| Options flow hard gate | Yes (Gap Z7) |
| Earnings proximity skip | No — add to backtest instead (Gap Z4) |
| Fractal regime penalty | No — ADD TO LIVE (Gap Z5); should exist in both |

Note on earnings: Earnings skip is a valid real risk-management gate that
SHOULD be added to the backtest rather than disabled in live. It is not a
paper-trading-specific override.

Note on adaptor overrides: In `paper_trading_mode`, the adaptor still observes
outcomes and logs recommendations (for future use). It is read-only — it must not
write any override back to the config, the orchestrator, or the position sizing logic.

---

## Phase 1 — Gaps in the Existing Plan (Code-Level)

### Gap A — `get_feature_names()` does not include VLMC feature names
**Relevant to:** Fix 3

`FeatureEngine.get_feature_names()` (`features.py:385–432`) returns exactly 57
base feature names. It has no knowledge of the 26 VLMC feature names produced
by `_merge_intraday_features()`. The ensemble training code does:

```python
self._feature_names = self._feature_engine.get_feature_names()
self._feature_names = [f for f in self._feature_names if f in features.columns]
```

Even after Fix 0 populates the intraday store and `_merge_intraday_features`
appends VLMC columns, this filtering line will silently drop all 26 of them.
VLMC features still won't train.

**Required addition to Fix 3:** Update `get_feature_names()` to detect and
include VLMC column names when they are present in the computed feature matrix,
or maintain a separate `_vlmc_feature_names` list that is appended when an
`intraday_store` was used.

---

### Gap B — `features_cache` in Optuna loop computed without `intraday_store`
**Relevant to:** Fix 3 / Fix 1e

In `walkforward.py:_optimize_window_params()` (line 701):

```python
features_cache = FeatureEngine().compute(train_df)   # no intraday_store, no symbol
```

This cache is passed to `StrategyOptimizer` → `Backtester`. The predictor was
trained WITH VLMC features (via `_train_window_model` which passes
`intraday_store`). Optuna evaluates the predictor with a feature matrix that
lacks VLMC columns, creating a train/eval mismatch inside each Optuna trial.

**Required additions:**
1. Pass `intraday_store` and `symbol` to `FeatureEngine().compute()` when
   building the features_cache in `_optimize_window_params()`.
2. `Backtester.__init__` needs `intraday_store` and `symbol` constructor params
   so `_get_direction()` can forward them to `FeatureEngine.compute()`.
3. `engine.py:_get_direction()` must pass `intraday_store` and `symbol` to both
   `FeatureEngine().compute(hist)` and `predictor.predict(hist)`.

---

### Gap C — No schema or loading path for `implied_vol` column
**Relevant to:** Fix 0 / Fix 4

The plan says "store as new `implied_vol` column in `daily_prices`" but the
actual `daily_prices` table schema in `historical.py:_init_db()` has no such
column. Three concrete additions are missing from the plan:

1. **Schema migration:** Add `implied_vol REAL` column to `daily_prices` DDL in
   `historical.py`. If the table already exists, `ALTER TABLE` is needed.
2. **Save/load methods:** New `save_daily_iv(symbol, iv_series)` and
   `load_daily_iv(symbol, days)` methods in `HistoricalDataStore`.
3. **Propagation to `hist` DataFrame:** `load_daily_ohlcv()` in `market_data.py`
   currently returns only `[Open, High, Low, Close, Volume]` from Yahoo/Polygon.
   It must be updated to left-join the `implied_vol` series from the SQLite store
   so the DataFrame that reaches `_get_iv()` actually has the column.

---

### Gap D — `resample_to_daily()` needs a `cutoff_time` parameter
**Relevant to:** Fix 1b

Fix 1b specifies:
> Reconstruct the partial daily bar as `resample_5min_to_partial_daily(bars_up_to_T)`

This helper does not exist. The existing `resample_to_daily()` method groups by
`index.date` with no concept of a time cutoff — it always includes all bars up
to the most recent stored bar. For the intraday engine, calling it at scan time
T=11:00 will still return today's bar using only bars up to 11:00 AM, but ONLY
because the caller first slices `intraday_full` to `[bars where time ≤ 11:00]`.
This slicing logic needs to be explicitly specified in the plan as a helper:

```python
def slice_intraday_up_to(intraday_df, cutoff_time: datetime.time) -> pd.DataFrame:
    """Return only intraday bars whose time component ≤ cutoff_time."""
```

Add this to `historical.py` alongside the existing resample methods.

---

### Gap E — `param_spaces.py` not mentioned for new Optuna params
**Relevant to:** Fix 5c

The plan says "add to the Optuna parameter space" for spread params, but the
actual parameter definitions live in `src/ait/optimization/param_spaces.py`
(the `STRATEGY_SPACES` and `ML_SPACES` dicts). `optimizer.py` reads from this
file. The plan must explicitly name `param_spaces.py` as a file to edit, and
specify the strategy scopes for the new params (spread params apply to all
credit strategies: `iron_condor`, `short_strangle`, `put_credit_spread`).

---

### Gap F — Position dict missing `entry_time` / `exit_time` fields
**Relevant to:** Fix 1

The position dict (returned by `_build_position`, `_check_exit`, `_force_close`)
stores `entry_date` and `exit_date` as ISO date strings. For intraday P&L
attribution, forensics, and the execution log, timestamps are needed. The plan
implicitly requires this (it speaks of "fill bar" and "scan time T") but never
adds it as an explicit task.

**Required addition:** Add `entry_time: str` (ISO datetime) and
`exit_time: str` to the position dict. Update `_build_position` to populate
`entry_time` from the current scan timestamp, and `_check_exit` / `_force_close`
to populate `exit_time` from the execution bar timestamp.

---

### Gap G — `export_production_params.py:_PARAM_MAP` not updated for new params
**Relevant to:** Fix 6

Fix 6c says "update `export_production_params.py`" but does not list the new
params that need entries in `_PARAM_MAP`:

```python
"spread_base":              ("backtest", "spread_base"),
"spread_iv_sensitivity":    ("backtest", "spread_iv_sensitivity"),
"spread_dte_sensitivity":   ("backtest", "spread_dte_sensitivity"),
"scan_interval_minutes":    ("backtest", "scan_interval_minutes"),
"entry_window_start_et":    ("backtest", "entry_window_start_et"),
"entry_window_end_et":      ("backtest", "entry_window_end_et"),
```

Without these, the export script silently drops them into the fallback
`("backtest", param_name)` bucket, which may work by coincidence but should be
explicit and tested.

---

### Gap H — `BacktestConfig` vs `WalkForwardConfig` field split
**Relevant to:** Fix 1a

The plan says "add to `BacktestConfig` / `config.yaml`" but the Backtester
constructor params (stop_loss_pct, min_confidence, etc.) actually come from
`WalkForwardConfig` (walkforward.py:40–73), not from the Pydantic
`BacktestConfig` in `settings.py`. The intraday params
(`entry_window_start_et`, `scan_interval_minutes`, etc.) must be added to the
correct dataclass — `WalkForwardConfig` — and the Backtester constructor
signature updated to match.

---

### Gap I — Walk-forward result JSON missing intraday fields
**Relevant to:** Fix 1 / observability

The walk-forward result serialization (per-window JSON) records only aggregate
metrics. With intraday trades, the per-trade list needs:
- `entry_time` (datetime string, not just date)
- `exit_time`
- `scan_index` (which scan within the day triggered the entry)
- `fill_bar_offset` (how many execution bars until fill — measures execution quality)

These fields are needed to diagnose intraday execution quality during paper
trading. Add them to `BacktestResult.trades` and the window JSON serialization.

---

## Phase 2 — Test Cases

### Testing infrastructure notes
- Existing pattern: `_make_ohlcv(days, start_price)` synthetic data helper (seeded RNG)
- Async tests use `@pytest.mark.asyncio` or `asyncio.run()`
- IBKR tests gated by `@pytest.mark.ibkr`
- Expensive tests gated by env var `RUN_INTEGRATION_TESTS=1`
- Test files location: `tests/`

---

### Fix 0 — IB Historical Data Backfill

**File:** `tests/test_backfill_historical_data.py`

```
T0-1: test_backfill_intraday_stores_correct_schema
  - Run backfill for a test symbol against a mock IBKR client
  - Assert intraday_prices table has rows with correct (symbol, datetime, interval, ohlcv)
  - Assert primary key constraint prevents duplicates on re-run

T0-2: test_backfill_daily_iv_column_populated
  - Run IV backfill for a test symbol
  - Assert daily_prices.implied_vol is non-null for all returned rows
  - Assert values are in plausible range (0.05 – 2.0)

T0-3: test_load_daily_ohlcv_includes_iv_column
  - Seed daily_prices with known implied_vol values
  - Call load_daily_ohlcv(symbol)
  - Assert returned DataFrame has 'implied_vol' column
  - Assert values match seeded data

T0-4: test_slice_intraday_up_to_cutoff_time
  - Build synthetic intraday DataFrame with bars from 09:30 to 16:00
  - Call slice_intraday_up_to(df, cutoff_time=time(11, 0))
  - Assert returned DataFrame has only bars ≤ 11:00
  - Assert last bar timestamp is exactly 11:00 or the nearest earlier bar

T0-5: test_resample_to_daily_partial_bar
  - Build synthetic intraday for one day (09:30 – 11:00 only)
  - Call resample_to_daily(symbol) (or the partial helper)
  - Assert returned daily row: Open=first bar open, High=max, Low=min,
    Close=last bar close (not 16:00 close), Volume=sum of partial session
  - Assert row count = 1

T0-6: test_no_duplicate_intraday_on_re-backfill
  - Backfill same date range twice
  - Assert row count is identical after second run (upsert, not double-insert)

T0-7: test_daily_iv_fallback_missing_data
  - Call load_daily_ohlcv for a symbol with no IV data
  - Assert DataFrame returned without implied_vol column OR with NaN values
  - Assert _get_iv() fallback path activates (see Fix 4 tests)
```

---

### Fix 1 — Intraday Backtest Engine

**File:** `tests/test_intraday_engine.py`

```
T1-1: test_intraday_loop_scan_count
  - Build synthetic intraday data (1 trading day, 09:30–16:00, 5-min bars)
  - Configure entry_window_start_et="10:30", entry_window_end_et="15:30",
    scan_interval_minutes=60
  - Run backtest for that day
  - Assert signal evaluation was attempted exactly 5 times
    (10:30, 11:30, 12:30, 13:30, 14:30 — no scan at 15:30 as it equals end)

T1-2: test_limit_order_fills_when_price_in_range
  - Construct synthetic scenario: signal fires at 10:30 with underlying=480
  - Limit posted at mid price P
  - Next execution bar has Low ≤ P ≤ High
  - Assert position is opened with entry_price = P (not market price)

T1-3: test_limit_order_cancels_after_timeout
  - Signal fires, limit posted at price P
  - Next limit_order_timeout_bars bars all have prices above P (no fill)
  - Assert no position is opened (order cancelled)
  - Assert capital unchanged

T1-4: test_intraday_stop_loss_triggers_before_eod
  - Open iron condor at 10:30 (short call at strike K)
  - Construct 5-min bars where underlying spikes above K at 13:15 then
    reverses and closes below K at 16:00
  - Assert position exits at the 13:15 bar (stop triggered intraday)
  - Assert exit_time is 13:15, not 16:00

T1-5: test_eod_exit_would_miss_stop (regression contrast)
  - Same scenario as T1-4 but engine in legacy EOD mode
  - Assert position does NOT exit (EOD close is within profit zone)
  - This confirms intraday checking catches events EOD misses

T1-6: test_intraday_profit_target_triggers_before_eod
  - Open iron condor, condor value drops to profit_target threshold at 12:45
  - Assert position exits at 12:45 bar

T1-7: test_entry_window_gate_respected
  - Configure entry_window_start_et="10:30"
  - Inject a high-confidence signal at 09:35 (before window opens)
  - Assert no position opened at 09:35
  - Inject same signal at 10:30
  - Assert position opens at 10:30

T1-8: test_partial_daily_bar_construction
  - Provide 5-min bars 09:30–11:00 for day D
  - Call the partial bar builder at scan time T=11:00
  - Assert partial bar Close == last 5-min bar close (not EOD close)
  - Assert partial bar High == max High of bars 09:30–11:00
  - Assert partial bar Volume == sum of bars 09:30–11:00

T1-9: test_feature_cache_reused_within_day
  - Spy on FeatureEngine.compute()
  - Run intraday engine for 1 day with 5 scans
  - Assert FeatureEngine.compute() called ≤ 2 times (once for D-1 cache,
    once per partial bar append — NOT 5 full recomputations)

T1-10: test_partial_dte_calculation
  - Build position at scan_time = 11:00 AM with target DTE = 21
  - Assert stored effective_dte ≈ 20.54 (21 - 11/24)
  - Assert t = effective_dte / 365 used in Black-Scholes pricing

T1-11: test_no_second_entry_while_position_open
  - Open iron condor at 10:30
  - Inject another high-confidence signal at 11:30 (same symbol)
  - Assert only 1 position exists (second signal ignored per conflict rule)

T1-12: test_entry_time_stored_in_position_dict
  - Open position at scan time 11:30
  - Assert position dict has entry_time == "2025-01-15T11:30:00"
  - Assert entry_time != entry_date (different granularity)
```

---

### Fix 2 — VLMC Feature Session Alignment

**File:** `tests/test_vlmc_session_tiering.py`

```
T2-1: test_tier_a_features_use_prior_session
  - Build synthetic intraday: complete session for D-1, partial session for D
  - Inject a known power_hour_momentum in D-1's last 12 bars
  - Call compute_intraday_features_tiered(session_d, session_d_minus_1, scan_time)
  - Assert power_hour_momentum in result matches D-1 computation
  - Assert it does NOT match D's partial session computation

T2-2: test_tier_b_features_use_partial_current_session
  - Build synthetic intraday: partial session for D (09:30–11:00)
  - Call compute_intraday_features_tiered with scan_time=11:00
  - Assert intraday_vwap_position reflects D's bars 09:30–11:00 only
  - Assert result is different from what D-1's VWAP would produce

T2-3: test_vlmc_keys_absent_from_live_signals_after_fix
  - After Fix 2c is applied: orchestrator passes intraday_store directly
    to predictor.predict() instead of using live_signals.update(fractal)
  - Mock the orchestrator's _analyze_symbol method and capture live_signals
  - Assert none of the 26 VLMC key names appear in live_signals dict
  - Assert sentiment keys still present (sentiment_composite, fear_greed, etc.)

T2-4: test_unified_path_produces_consistent_features
  - For a given historical date D with known intraday data:
    a. Compute VLMC features via training path (_merge_intraday_features)
    b. Compute VLMC features via inference path (intraday_store in predict)
  - Assert both paths return identical values for the same date

T2-5: test_tier_a_defaults_when_no_prior_session
  - Call compute_intraday_features_tiered when D-1 session is empty/missing
  - Assert Tier A features default to 0.0 (not NaN, not error)
  - Assert Tier B features still computed from D's partial session

T2-6: test_hurst_wavelet_defaults_on_short_session
  - Pass a 30-bar partial session (less than 78 bars)
  - Assert hurst_wavelet_intraday returns its defined default (0.0)
  - Assert no exception is raised

T2-7: test_session_vwap_q3_only_after_75pct_of_session
  - For a partial session at 40% completion (~31 bars of 78)
  - Assert session_vwap_q3 returns 0.0 or a sentinel value
    (cannot be meaningfully computed before 75% of session)
```

---

### Fix 3 — VLMC Features in Walk-forward Training

**File:** `tests/test_vlmc_training_integration.py`

```
T3-1: test_get_feature_names_includes_vlmc_after_compute
  - Create FeatureEngine; call compute() with a real intraday_store (seeded data)
  - Call get_feature_names() after compute
  - Assert all 26 VLMC feature names are present in the returned list
  - Assert total feature count = 57 base + 26 VLMC = 83

T3-2: test_predictor_feature_names_include_vlmc_after_training
  - Train DirectionPredictor (EnsemblePredictor) on synthetic data WITH
    a seeded in-memory intraday store containing QQQ 5-min data
  - Assert predictor._feature_names includes at least 10 VLMC names
  - Assert len(predictor._feature_names) > 57

T3-3: test_features_cache_with_intraday_store_in_optuna_loop
  - Call FeatureEngine().compute(train_df, intraday_store=store, symbol="QQQ")
  - Assign result to features_cache
  - Assert features_cache has VLMC columns (not all-NaN for dates with intraday data)
  - Assert cache can be passed to Backtester and used without KeyError

T3-4: test_no_feature_count_mismatch_train_vs_infer
  - Train predictor with intraday_store (VLMC features present in training)
  - Run predictor.predict() with intraday_store (VLMC via _merge_intraday_features)
  - Assert no KeyError or column mismatch exception
  - Assert prediction is not None

T3-5: test_vlmc_features_nonzero_in_trained_model
  - After training on seeded intraday data, inspect feature importances
  - Assert at least 5 VLMC features have importance > 0
  - (Validates VLMC features are actually reaching the model, not silently zeroed)

T3-6: test_empty_intraday_store_graceful_fallback
  - Create empty intraday store (no rows for QQQ)
  - Call FeatureEngine().compute(hist, intraday_store=empty_store, symbol="QQQ")
  - Assert returned DataFrame has 57 base features (no VLMC)
  - Assert no exception, no NaN injection into base features
```

---

### Fix 4 — Realistic IV from IBKR Historical Data

**File:** `tests/test_iv_model.py`

```
T4-1: test_get_iv_uses_stored_iv_when_column_present
  - Build hist DataFrame with implied_vol column, last value = 0.35
  - Call _get_iv(hist) from the OptionSimulator
  - Assert return value == max(0.35, iv_floor) == 0.35

T4-2: test_get_iv_respects_iv_floor
  - Build hist DataFrame with implied_vol = 0.10 (below floor of 0.20)
  - Call _get_iv(hist)
  - Assert return value == 0.20 (iv_floor applied)

T4-3: test_get_iv_fallback_when_column_absent
  - Build hist DataFrame WITHOUT implied_vol column
  - Call _get_iv(hist)
  - Assert return value == max(rv * 1.15, iv_floor) (synthetic fallback)

T4-4: test_get_iv_fallback_when_value_is_nan
  - Build hist DataFrame with implied_vol column, last value = NaN
  - Call _get_iv(hist)
  - Assert synthetic fallback is used (not NaN returned)

T4-5: test_iron_condor_premium_higher_in_high_iv_regime
  - Price iron condor with iv=0.45 (stress regime)
  - Price same condor with iv=0.20 (normal regime), same underlying + DTE
  - Assert net_credit(iv=0.45) > net_credit(iv=0.20)

T4-6: test_vix_proxy_produces_plausible_iv
  - Build market_context dict with VIX close = 22.0
  - Activate VIX proxy (interim Fix 4b)
  - Assert resulting IV ≈ 22.0 * 1.05 / 100 = 0.231
  - Assert IV > iv_floor

T4-7: test_iv_consistency_backtest_vs_live_same_date
  - For a historical date with known IBKR IV = 0.28 stored in SQLite
  - Confirm _get_iv(hist) returns 0.28 in backtest
  - Confirm live bot would use same source (modelGreeks path — document the
    remaining gap rather than test it, since it's not fully closeable)
```

---

### Fix 5 — Options Bid-Ask Spread Model

**File:** `tests/test_spread_model.py`

```
T5-1: test_half_spread_increases_with_iv
  - sim = OptionSimulator(spread_base=0.03, spread_iv_sensitivity=0.15, ...)
  - Assert sim._options_half_spread(iv=0.20, dte=21) == 0.03
  - Assert sim._options_half_spread(iv=0.40, dte=21) > 0.03
  - Assert sim._options_half_spread(iv=0.60, dte=21) > sim._options_half_spread(iv=0.40, dte=21)

T5-2: test_half_spread_increases_near_expiry
  - Assert sim._options_half_spread(iv=0.25, dte=21) < sim._options_half_spread(iv=0.25, dte=3)

T5-3: test_half_spread_capped_at_spread_cap
  - Set spread_cap = 0.10
  - Assert sim._options_half_spread(iv=2.0, dte=1) == 0.10 (capped)

T5-4: test_per_leg_spread_reduces_net_credit
  - Build iron condor at mid prices (no spread)
  - Build same condor with per-leg spread_base=0.05
  - Assert credit_with_spread < credit_without_spread
  - Assert reduction ≈ 2 * spread_base * (sum of short leg mids)

T5-5: test_exit_spread_increases_total_cost
  - Enter condor at mid, record net_credit
  - Exit condor at full profit (condor value → 0); measure buyback cost
  - Assert round-trip cost > 2 * 1% * net_credit (old model)
  - Assert round-trip cost ≈ 4 * spread_base * leg_mid (new per-leg model)

T5-6: test_spread_params_present_in_param_spaces
  - Import STRATEGY_SPACES from param_spaces.py
  - Assert 'bt__spread_base' in STRATEGY_SPACES['iron_condor']
  - Assert 'bt__spread_iv_sensitivity' in STRATEGY_SPACES['iron_condor']
  - Assert ranges are plausible (low=0.02, high=0.08 for spread_base)

T5-7: test_chase_premium_applied_on_missed_fill
  - Signal fires, limit at mid P, limit_order_timeout_bars elapses without fill
  - Assert position opened at P * (1 + chase_premium_frac * half_spread)
  - Assert entry_price > P (paid up to get filled)
```

---

### Fix 6 — Walk-forward Params → Live Deployment

**File:** `tests/test_paper_trading_mode.py`

```
T6-1: test_paper_trading_mode_blocks_confidence_override
  - Create StrategyAdaptor with a confidence_override set to 0.80
  - Set settings.learning.paper_trading_mode = True
  - Call orchestrator logic that resolves min_confidence
  - Assert resolved value == settings.risk.min_confidence (e.g., 0.65)
  - Assert adaptor override (0.80) is NOT used

T6-2: test_paper_trading_mode_false_allows_override
  - Same setup, paper_trading_mode = False
  - Assert resolved value == 0.80 (adaptor override applied)

T6-3: test_adaptor_still_logs_recommendations_in_paper_mode
  - Enable paper_trading_mode = True
  - Trigger an adaptor insight that would normally raise confidence
  - Assert adaptor.apply_insights() runs without error
  - Assert the adaptation is logged / visible via get_current_adaptations()
  - Assert actual config parameter unchanged

T6-4: test_export_production_params_includes_new_params
  - Run walk-forward with spread_base in best_params
  - Call export_production_params.extract_deployment_params(run_dir)
  - Assert 'bt__spread_base' in returned params
  - Assert 'bt__scan_interval_minutes' in returned params
  - Call apply_params_to_config and assert new keys written to output YAML

T6-5: test_confidence_quantile_recalibration
  - Train predictor on synthetic data; record confidence_scores distribution
  - Compute quantile q at threshold min_confidence=0.65
  - Retrain predictor on different (but similar) synthetic data
  - Apply quantile recalibration to get effective_min_confidence
  - Assert abs(selectivity_at_old_threshold - selectivity_at_new_threshold) < 0.05
  - (Selectivity = fraction of predictions ≥ threshold)

T6-6: test_model_independent_params_unchanged_after_retrain
  - Export params: wing_k=1.2, stop_loss_pct=0.5, min_confidence=0.65
  - Retrain model on different window
  - Assert wing_k and stop_loss_pct unchanged
  - Assert min_confidence was recalibrated (quantile-derived, may differ)
```

---

### Regression / Integration Tests

**File:** `tests/test_intraday_regression.py`

```
TR-1: test_intraday_system_more_conservative_than_eod
  Given identical synthetic data:
  - Run backtest in EOD mode (current engine)
  - Run backtest in intraday mode (new engine)
  - Assert intraday_sharpe <= eod_sharpe  (look-ahead bias removed → lower)
  - Assert intraday_stop_count >= eod_stop_count (intraday stops fire more)
  - Assert intraday_trades_total <= eod_trades_total (missed fills reduce count)

TR-2: test_feature_values_consistent_at_training_vs_inference_time
  For a historical day D with known intraday bars:
  - Compute feature row for D during training pass (via _merge_intraday_features)
  - Compute feature row for D during inference pass (predictor.predict with store)
  - Assert max absolute difference across all VLMC features < 1e-6
  - Assert base features (RSI, ATR, BB) identical in both paths

TR-3: test_walk_forward_completes_with_intraday_data
  - Run a 2-window walk-forward on synthetic intraday data (QQQ-like)
  - Assert no exceptions
  - Assert WalkForwardResult has at least 1 window with trades > 0
  - Assert all 26 VLMC features appear in at least one window's feature names

TR-4: test_iv_spread_interaction_self_consistent
  - High IV day: stored implied_vol = 0.45
  - Assert spread cost increases proportionally with IV (T5-1 linkage)
  - Assert credit collected also increases with IV (T4-5 linkage)
  - Assert net P&L impact is directionally correct:
    high IV → higher raw credit but also higher spread cost

TR-5: test_no_look_ahead_in_partial_bar_construction
  - For scan time T=11:00 on day D, construct partial bar
  - Assert partial bar Close == price at exactly 11:00 (not later)
  - Assert partial bar High does not exceed any bar's High before 11:00
  - Run same scan at T=14:00; Assert partial bar Close != T=11:00 Close
    (new information incorporated)
```

---

### Gap Z Fixes — Live/Backtest Overlay Alignment

**File:** `tests/test_paper_trading_overlay_parity.py`

```
TZ-1: test_paper_mode_disables_meta_labeler
  - Train a MetaLabeler that always returns take_trade=False
  - Set paper_trading_mode = True
  - Run orchestrator._analyze_symbol with a high-confidence signal
  - Assert signal proceeds to execution (meta-labeler not consulted)
  - Set paper_trading_mode = False
  - Assert signal is blocked (meta-labeler applied)

TZ-2: test_paper_mode_disables_thompson_reranking
  - Set Thompson sampler to strongly rank short_strangle above iron_condor
  - Set paper_trading_mode = True
  - Run orchestrator signal path
  - Assert strategy selection follows deterministic priority (iron_condor first)
  - Set paper_trading_mode = False
  - Assert Thompson ranking is applied (short_strangle may come first)

TZ-3: test_market_context_passed_to_backtest_predictors
  - Build a mock market_context with vix=35 (high volatility)
  - Call engine._get_direction(hist, market_context=market_context)
  - Assert FeatureEngine.compute() was called with market_context
  - Assert predictor.predict() was called with market_context
  - Assert range_predictor.predict() was called with market_context

TZ-4: test_fractal_penalty_applied_in_live_bot
  - Build hist with high hurst_scale_spread (> hurst_regime_threshold)
  - Live orchestrator computes confidence = 0.72 from ML
  - Assert that after the fractal penalty is applied, effective confidence < 0.72
  - Assert min_confidence check uses the penalized value

TZ-5: test_fractal_penalty_consistent_backtest_vs_live
  - Apply same hist data to both backtest engine and live confidence path
  - Assert penalty magnitude is identical in both (same threshold, same formula)

TZ-6: test_thesis_invalidation_exit_in_backtest
  - Open a NEUTRAL (iron condor) position at bar 0
  - At bar 5, inject ML prediction: STRONGLY_BULLISH with confidence=0.85
  - Assert position is flagged for early exit at bar 5
  - Assert exit_reason contains "thesis_invalidated"
  - Assert exit_price is bar 5's underlying price, not a later stop-loss price

TZ-7: test_backtest_respects_earnings_skip
  - Build synthetic earnings calendar: earnings on date D+3
  - Inject high-confidence signal on date D (3 days before earnings)
  - Assert no position opened on date D
  - Inject same signal on date D-10 (well before earnings)
  - Assert position opens on D-10

TZ-8: test_paper_mode_disables_options_flow_gate
  - Configure options flow: strong bearish flow contradicting a bullish signal
  - paper_trading_mode = True: assert signal proceeds despite flow disagreement
  - paper_trading_mode = False: assert signal is blocked by flow gate

TZ-9: test_meta_labeler_trained_per_walkforward_window
  - Run walk-forward with meta-labeler training enabled
  - Assert each window's MetaLabeler is trained on that window's training set outcomes
  - Assert MetaLabeler's feature names include primary_confidence + regime + vix
  - Assert MetaLabeler is saved alongside DirectionPredictor in window artifacts

TZ-10: test_all_paper_mode_overlays_disabled_simultaneously
  - Configure paper_trading_mode = True with all overlays active
  - Verify: meta_labeler (interim), thompson, options_flow all bypassed
  - Verify: adaptor confidence_override, stop_loss_override, trailing_stop_override,
    take_profit_override all bypassed (returns None / base config value)
  - Verify: earnings skip NOT bypassed (it is a valid risk gate, added to backtest)
  - Verify: fractal penalty IS applied (it should now exist in both environments)

TZ-11: test_adaptor_exit_overrides_bypassed_in_paper_mode
  - Configure stop_loss_override = 0.20 (tighter than default 0.35)
  - Configure trailing_stop_override = 0.15
  - Configure take_profit_override = 0.40 (lower than default 0.50)
  - Set paper_trading_mode = True
  - Assert orchestrator uses 0.35, 0.25, 0.50 (base config values) for all three
  - Set paper_trading_mode = False
  - Assert orchestrator uses 0.20, 0.15, 0.40 (adaptor overrides applied)

TZ-12: test_meta_labeler_trained_on_walk_forward_training_outcomes
  - Run single walk-forward window with synthetic OHLCV + intraday data (seeded)
  - Assert window artifact directory contains 'meta_labeler.pkl'
  - Load MetaLabeler from artifact; assert it is trained (meta_labeler.is_trained == True)
  - Assert MetaLabeler feature names include all 20 expected features (not just 9)
  - Assert MetaLabeler accuracy on training set > 0.55 (above random for a solvable problem)

TZ-13: test_meta_labeler_applied_during_oos_evaluation
  - Run walk-forward with meta-labeler enabled
  - For the OOS window, assert trade count with meta-labeler ≤ trade count without
  - Assert OOS metrics (sharpe, win_rate) differ between meta-labeled and non-meta-labeled runs
    (if they are identical, meta-labeler is not being applied)

TZ-14: test_meta_labeler_build_training_data_populates_all_features
  - Call MetaLabeler.build_training_data(hist, intraday_store, symbol, trade_outcomes)
  - Assert returned DataFrame has 20 columns (not 9)
  - Assert columns 'rsi', 'bb_position', 'macd_signal', 'volume_ratio',
    'trend_alignment', 'hour_of_day' are NOT all-zero
  - Assert no NaN values in the returned matrix
```

---

## Remaining Residual Gaps After All Fixes

The following gaps will persist even after all planned fixes (Fixes 0–6, Gaps A–I, Z1–Z9)
are implemented. They are **irreducible without major infrastructure additions** that are
out of scope for this sprint. They are documented here as known acceptance criteria for
the paper-trading validation phase — if paper-trading P&L diverges from backtest P&L
in one of these dimensions, these are the first explanations to investigate.

---

### Residual R1 — IV smile: ATM IV vs OTM per-leg IV

**What remains:** `options_sim.py` prices each leg using the ATM IV (from IBKR daily
snapshot or the synthetic fallback). A flat-IV Black-Scholes model applied to OTM
options is systematically wrong because actual options markets exhibit a volatility
smile/skew: OTM puts trade at significantly higher IV than ATM (put skew), and OTM
calls trade slightly lower. For iron condors this means:
- Short puts: backtest underestimates premium collected (OTM put IV > ATM IV).
- Short calls: backtest overestimates premium collected (OTM call IV < ATM IV).
The current `_get_leg_iv()` applies a linear skew adjustment (`skew_factor`) but this is
a rough approximation that is not calibrated to actual market skew.

**Why it cannot be fixed now:** Requires per-strike IV data (full options chain IV surface
per expiry), not just ATM IV snapshots. IBKR API provides this in real time but not
historically at a level that would allow training/calibration.

**Acceptance criterion:** Paper-trade P&L should correlate with backtest P&L, but premium
collected may be ±15% different due to smile. This is tolerable for an iron condor where
the credit is relatively wide.

---

### Residual R2 — European vs American options pricing

**What remains:** The Black-Scholes engine uses European option pricing. QQQ options are
American (can be exercised early). For deeply ITM puts and calls near expiry, the early
exercise premium can be material. For an iron condor where legs are OTM at entry, this
is a second-order effect that only matters if one leg goes deep ITM.

**Acceptance criterion:** Early exercise events are rare for OTM iron condors. Divergence
in this dimension signals a regime where the underlying moved dramatically (the iron condor
was already losing money).

---

### Residual R3 — Expiry cycle mismatch (no weekly/monthly awareness)

**What remains:** The backtest uses `DTE = max_hold_days` (default 30) as a proxy for
DTE regardless of actual expiry cycles. Real QQQ options have weekly expirations (every
Friday) and monthly (third Friday). Entering on Monday with DTE=30 means entering a
~4-week contract; entering on Thursday means entering a ~4-week contract expiring four
Fridays away — not a 30-DTE contract as the backtest assumes. Actual available strikes
and liquidity profiles differ significantly between weekly and monthly expirations.

**Fix (deferred):** Add expiry cycle awareness: at entry time, look up the nearest
expiry that gives DTE ≥ `min_dte` and ≤ `max_dte`. Requires a historical expiry calendar.

---

### Residual R4 — Commission structure underestimation

**What remains:** The backtest applies a flat commission model. Actual IBKR tiered
pricing for options is approximately $0.65/contract for the first 10,000 contracts/month
plus exchange fees ($0.02–0.18/contract) plus regulatory fees ($0.0008/contract). For a
4-leg iron condor with 1 contract per leg, total round-trip commission is typically
$0.77–$0.87 × 8 fills = ~$6.20–$6.96. If the backtest uses a lower flat rate, P&L
will be systematically overstated by ~$1–2 per trade.

**Fix (deferred):** Configure commission as `$0.85/fill` × number of legs × 2 (entry +
exit) and add exchange/regulatory overhead as a fixed `$0.25/trade` line. Document the
calibrated values in `config.yaml`.

---

### Residual R5 — Market impact / liquidity on large position sizes

**What remains:** The backtest assumes zero market impact — orders fill at mid with the
modeled half-spread. In reality, iron condor legs on near-expiry OTM QQQ options can
have 10–50 contracts of open interest. A 2-contract order moving a 20-contract book
can widen the spread and delay fills. The spread model (Fix 5) accounts for bid-ask
width but not market impact from order size.

**Acceptance criterion:** At 1-contract position sizing (current config), market impact
is negligible. If position size scales up (position_size_pct > 5%), this gap becomes
material and the spread model needs a volume-impact term.

---

## Critical Files to Modify (complete list)

| File | Gaps / Changes |
|---|---|
| `src/ait/backtesting/engine.py` | Add intraday loop (Fix 1b), exit checking (1c), `intraday_store`+`symbol` params (Gap B), `entry_time`/`exit_time` (Gap F), partial DTE (1f) |
| `src/ait/backtesting/walkforward.py` | Pass `intraday_store`+`symbol` to features_cache computation (Gap B), add `entry_time`/`exit_time` to result JSON (Gap I) |
| `src/ait/backtesting/options_sim.py` | Add `_options_half_spread()` + per-leg spread application (Fix 5a-b), update `_get_iv()` (Fix 4a) |
| `src/ait/optimization/optimizer.py` | Pass `intraday_store`+`symbol` to Backtester (Gap B) |
| `src/ait/optimization/param_spaces.py` | Add spread params to credit strategy spaces (Gap E / Fix 5c) |
| `src/ait/ml/features.py` | Refactor `_merge_intraday_features` for session tiering (Fix 2a-b), update `get_feature_names()` to include VLMC names (Gap A), add `compute_intraday_features_tiered()` helper |
| `src/ait/ml/ensemble.py` | No logic changes needed; verify `feature_names` selection works with VLMC (Fix 3 verification) |
| `src/ait/data/historical.py` | Add `implied_vol` column to DDL + migration (Gap C), add `save_daily_iv()` + `load_daily_iv()` (Gap C), add `slice_intraday_up_to()` helper (Gap D) |
| `src/ait/data/market_data.py` | Update `load_daily_ohlcv()` to join `implied_vol` from SQLite store (Gap C) |
| `src/ait/bot/orchestrator.py` | Remove VLMC from `live_signals`, pass `intraday_store` to `predict()` (Fix 2c), add `is_in_entry_window()` gate (Fix 1g) |
| `src/ait/learning/adaptor.py` | Add `paper_trading_mode` check to `get_confidence_override()` AND all three exit overrides (Fix 6a, Gap Z8) |
| `src/ait/learning/meta_labeler.py` | Fix `build_training_data()` to populate all 20 features from FeatureEngine (Gap Z9, prerequisite for Gap Z1) |
| `src/ait/config/settings.py` | Add new params to `WalkForwardConfig` (Gap H / Fix 1a), add `paper_trading_mode` to `LearningConfig` (Fix 6a, Gaps Z1-Z2, Z7-Z8) |
| `scripts/export_production_params.py` | Add new params to `_PARAM_MAP` (Gap G / Fix 6c) |
| `scripts/backfill_historical_data.py` | New file: IBKR historical 5-min + daily IV backfill (Fix 0) |
| `src/ait/bot/orchestrator.py` (additional) | Gate meta-labeler + Thompson + flow + exit overrides on `paper_trading_mode` (Gaps Z1, Z2, Z7, Z8); add fractal regime penalty (Gap Z5); add thesis-invalidation exit (Gap Z6) |
| `src/ait/backtesting/engine.py` (additional) | Pass market_context to predictor/range_predictor (Gap Z3); add earnings skip check (Gap Z4); add fractal penalty consistency with live (Gap Z5); add thesis re-evaluation exit (Gap Z6); accept `meta_labeler` param for OOS filtering (Gap Z1) |
| `src/ait/backtesting/walkforward.py` (additional) | Add MetaLabeler training step per window after direction predictor training (Gap Z1); save meta_labeler artifact; apply in OOS evaluation |

---

## Verification: How to Confirm the Full Fix Works

1. **After Fix 0:** Run `pytest tests/test_backfill_historical_data.py -v`; then
   query `SELECT COUNT(*) FROM intraday_prices WHERE symbol='QQQ'` — expect ≥ 100,000 rows.

2. **After Fix 1:** Run `pytest tests/test_intraday_engine.py -v`.
   Run a 1-symbol, 2-window walk-forward with intraday data; confirm it
   completes and produces trades with `entry_time` != `entry_date`.

3. **After Fix 2+3:** Run `pytest tests/test_vlmc_session_tiering.py tests/test_vlmc_training_integration.py -v`.
   Confirm `predictor._feature_names` has >57 features after training.

4. **After Fix 4+5:** Run `pytest tests/test_iv_model.py tests/test_spread_model.py -v`.
   Run the regression test `TR-4` to confirm IV and spread are self-consistent.

5. **After Fix 6:** Run `pytest tests/test_paper_trading_mode.py -v`.
   Run `python scripts/export_production_params.py --symbol QQQ --dry-run`;
   confirm new params appear in output.

6. **Overlay parity:** Run `pytest tests/test_paper_trading_overlay_parity.py -v`.
   Confirm all live-only gates (meta-labeler interim, Thompson, flow, ALL adaptor
   exit overrides) are disabled in `paper_trading_mode`. Confirm fractal regime penalty
   exists identically in both backtest engine and live orchestrator (TZ-5).

7. **MetaLabeler fix verification:**
   - Run `pytest tests/test_paper_trading_overlay_parity.py::test_meta_labeler_build_training_data_populates_all_features -v`
   - Run a 2-window walk-forward; confirm `meta_labeler.pkl` appears in both window artifact dirs.
   - Confirm `predictor._feature_names` in meta_labeler has 20 entries, not 9.

8. **End-to-end regression:** Run `pytest tests/test_intraday_regression.py -v`.
   Confirm `TR-1` (intraday system more conservative than EOD) passes —
   this is the single most important empirical validation that look-ahead bias
   has been removed.

9. **Final parity check (manual):** With `paper_trading_mode = true`, run the
   live bot and the walk-forward backtest on the SAME last 30 trading days.
   Compare:
   - Trade count: should be within ±20% of each other
   - Strategy mix: should be dominated by iron_condor in both
   - Average confidence at entry: should be within ±0.05 of each other
   - Average entry IV: should match closely (both using stored IBKR IV)
   - Average stop loss pct: should be exactly equal (no adaptor exit overrides in paper mode)
   Any large divergences indicate remaining structural gaps to investigate.
   If divergence persists, the Residual R1–R5 gaps (IV smile, expiry cycle, commission)
   are the next candidates to rule out quantitatively.

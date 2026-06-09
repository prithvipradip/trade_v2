# Addressing Backtest ↔ Live Inconsistencies — Implementation Plan

## Overview

This document is the actionable plan derived from the analysis in
`INCONSISTENCIES_BETWEEN_WALKFORWARD_AND_PRODUCTION.md`. Fixes are ordered by
dependency. No fix should be implemented before its prerequisites are met.
The goal is a live bot and walk-forward engine that are structurally identical
in their data, feature, and execution assumptions, so that paper-trading P&L
is a reliable predictor of live P&L.

---

## Implementation Status (branch: features-request-2)

| Fix / Gap | Status | Notes |
|---|---|---|
| Fix 0 — IB Historical Data Backfill | Partial | Schema + save/load methods done; IBKR backfill script (`scripts/backfill_historical_data.py`) stubbed; actual IBKR backfill pending |
| Fix 1 — Intraday Backtest Engine | Partial | Entry window, limit-order fill, intraday stop-loss/profit-target, partial bar construction, `entry_time`/`exit_time` fields implemented; full scan-cadence loop pending Fix 0 data |
| Fix 2 — VLMC Feature Session Alignment | Partial | `slice_intraday_up_to()` helper and `FeatureEngine` tiering stubs in place; full two-tier session compute pending Fix 0 data |
| Fix 3 — VLMC Features in Walk-forward Training | Partial | `get_feature_names(include_vlmc=True)` API added; VLMC columns flow when intraday store has data; pending Fix 0 backfill |
| Fix 4 — Realistic IV from IBKR | Partial | `implied_vol` column in DDL, `save_daily_iv`/`load_daily_iv`, `_get_iv()` uses stored column; IBKR backfill pending |
| Fix 5 — Options Bid-Ask Spread Model | Done | `_options_half_spread()` per-leg spread applied at entry and exit; params in `param_spaces.py` |
| Fix 6a — `paper_trading_mode` flag | Done | `LearningConfig.paper_trading_mode` added; adaptor confidence override gated |
| Fix 6b — Post-retrain confidence recalibration | Pending | — |
| Fix 6c — Model-independent param export | Done | `export_production_params.py` `_PARAM_MAP` updated with new spread + intraday params |
| **Gap Z1** — MetaLabeler per-window training | **Done** | `_train_window_meta_labeler()` in `walkforward.py`; MetaLabeler gate in `engine.py` OOS evaluation |
| **Gap Z9** — MetaLabeler `build_training_data()` only 9/20 features | **Done** | `build_training_data_from_backtest()` populates all 20 META_FEATURES |
| **Gap I** — `entry_time`/`exit_time` missing from walk-forward JSON | **Done** | `trades_detail` array with intraday-aware fields in window progress JSON |
| Gap Z2 — Thompson Sampling in live, absent from backtest | Pending | — |
| Gap Z3 — Market context absent from predictors in backtest | Pending | — |
| Gap Z4 — Earnings skip in live, absent from backtest | Pending | `_earnings_dates` set exists in engine; no calendar loaded yet |
| Gap Z5 — Hurst/fractal penalty missing from live | Pending | Fractal penalty exists in backtest; not yet added to orchestrator |
| Gap Z6 — Thesis re-evaluation absent from backtest | Done | `_check_thesis_invalidation()` implemented in engine |
| Gap Z7 — Options flow gate absent from backtest | Pending | — |
| Gap Z8 — Adaptor exit overrides absent from paper mode | Pending | — |
| Gap A — `get_feature_names()` excludes VLMC | Done | `include_vlmc=True` param added |
| Gap B — `features_cache` computed without `intraday_store` | Pending | — |
| Gap C — `implied_vol` schema + load path | Done | DDL migration, `save_daily_iv`, `load_daily_iv` |
| Gap D — `slice_intraday_up_to()` helper | Done | Static method on `HistoricalDataStore` |
| Gap E — `param_spaces.py` not updated for spread params | Done | `spread_base`, `spread_iv_sensitivity`, etc. added |
| Gap F — `entry_time`/`exit_time` missing from position dict | Done | All exit paths (`_check_exit`, `_check_thesis_invalidation`, `_force_close`) now set `exit_time` |
| Gap G — `export_production_params.py` `_PARAM_MAP` incomplete | Done | New params added |
| Gap H — `BacktestConfig` vs `WalkForwardConfig` field split | Done | New params added to `WalkForwardConfig` |

**Test suite:** 9 test files, 99 tests — all passing (`pytest tests/test_backfill_historical_data.py tests/test_intraday_engine.py tests/test_intraday_regression.py tests/test_iv_model.py tests/test_paper_trading_mode.py tests/test_paper_trading_overlay_parity.py tests/test_spread_model.py tests/test_vlmc_session_tiering.py tests/test_vlmc_training_integration.py`)

---

## Architecture Decision (Incs. 1 + 5)

**Decision: align the backtest TO live, not live to backtest.**

The backtest engine will be upgraded to simulate intraday trading rather than
simplifying the live bot to EOD-signal-only. This is the architecturally
correct long-term direction.

### Signal layer: fixed scan cadence (configurable, default 60 min)

The backtest processes intraday bars and generates signals at a fixed cadence
(e.g., every 60 minutes from `entry_window_start` to `entry_window_end`).
The live bot scans at the same cadence. Both see the same partial daily bar at
each scan time.

### Execution layer: limit-order fill simulation (5-min or 1-hour bars)

After a signal fires, a limit order is posted at the theoretical mid price.
Subsequent bars are scanned at the execution resolution (matching available
data granularity) to simulate fill. Exit conditions (stop-loss, profit target,
DTE expiry) are also checked at execution resolution — not only at EOD.

### Scan time is a fixed configuration parameter

`entry_window_start_et`, `entry_window_end_et`, and `scan_interval_minutes`
must match exactly between the backtest config and the live bot config.
The ML model's confidence distribution is time-of-day dependent: a model
trained on 11:00 AM bars cannot be deployed with a 2:00 PM scan time.
These are quasi-hyperparameters — changing them requires re-running the
walk-forward.

---

## Fix 0 — IB Historical Data Backfill (Prerequisite for everything)

**What:** Backfill two data types for QQQ (and all universe symbols) via
IBKR's historical API:
1. **5-min intraday bars** — for execution simulation and VLMC features.
   Use `reqHistoricalData` with `barSizeSetting="5 mins"`. Store in the
   existing `intraday_prices` SQLite table. Target: 2 years of history.
2. **Daily implied volatility** — for realistic options pricing (Fix 4).
   Use `reqHistoricalData` with `whatToShow="OPTION_IMPLIED_VOLATILITY"`,
   `barSizeSetting="1 day"`. Store as a new `implied_vol` column in the
   `daily_prices` table (or a companion `daily_iv` table).

**Where:** New `scripts/backfill_historical_data.py` script, plus additions
to `src/ait/data/historical.py` for the new `daily_iv` storage and retrieval.

**Verification:** After backfill, confirm:
- `intraday_prices` has ≥ 500 trading days of QQQ 5-min bars
- `daily_prices` has `implied_vol` populated for all symbols and dates

**Effort:** Small-Medium (data pipeline plumbing)

---

## Fix 1 — Intraday Backtest Engine (Incs. 1 + 5)

**Prerequisites:** Fix 0 (intraday bar data)

### 1a. New config parameters

Add to `BacktestConfig` / `config.yaml`:
```yaml
backtest:
  entry_window_start_et: "10:30"   # skip open volatility
  entry_window_end_et:   "15:30"
  scan_interval_minutes: 60        # signal cadence
  execution_bar_size:    "5m"      # fill simulation resolution
  limit_order_timeout_bars: 3      # cancel if unfilled after N exec bars
  time_of_day_feature: true        # add session_fraction feature to ML
```

### 1b. Backtest engine — intraday loop

Replace the current daily loop in `engine.py:run()` with a two-level loop:

**Outer loop:** trading days (as today)
**Inner loop — signal layer:** iterate at `scan_interval_minutes` from
`entry_window_start_et` to `entry_window_end_et`. At each scan time T:
- Slice daily history: all complete daily bars up to D-1 + the partial bar
  of day D (bars from market open through time T)
- Reconstruct the partial daily bar as `resample_5min_to_partial_daily(bars_up_to_T)`
  returning OHLCV: Open=first, High=max, Low=min, Close=last, Volume=sum
- Run `FeatureEngine.compute(hist_with_partial_bar, intraday_store, symbol)`
- Run `predictor.predict()`
- If signal ≥ `min_confidence` and no open position: register a pending limit
  order at the theoretical mid price (BS-priced at current underlying)

**Inner loop — execution layer:** iterate through 5-min bars of day D:
- For each pending limit order: check if the bar's range covers the limit
  price. If so, fill at the limit price. If `limit_order_timeout_bars` bars
  pass without fill, cancel (or fill at market with spread penalty).
- For each open position: check stop-loss and profit-target conditions against
  the current bar's underlying price (see Fix 1c).

### 1c. Intraday exit checking

Current exits use only EOD prices, missing intraday stop-loss hits — a
systematic underestimation of stop frequency for credit strategies.

For open iron condor / short strangle positions, at each execution bar:
- Compute the theoretical condor value at the bar's underlying price using
  Black-Scholes with remaining DTE
- If condor mark-to-market loss ≥ `stop_loss_pct × net_credit`: exit at
  that bar's price
- If condor value ≤ `(1 - profit_target_pct) × net_credit`: exit at that bar

This requires calling `black_scholes_price()` at each execution bar for each
open position leg. For a typical run (few open positions at a time) the cost
is acceptable. Cache IV and interest rate for the day.

### 1d. Multiple intraday signals — conflict resolution rule

If a position is already open: ignore all new entry signals for that symbol
until the position is closed. The iron condor is held for days; intraday
re-entry on a conflicting signal would introduce instability.

Only `max_concurrent_positions` positions are allowed per symbol at any time
(existing behaviour, already enforced).

### 1e. Feature cache optimization

The dominant cost driver of the intraday loop is re-running `FeatureEngine`
at each scan time. Rolling-window features (RSI, ATR, Bollinger Bands) over
the 504-day daily history do not change between scans within the same day —
only the partial bar at position [-1] changes.

Optimization: cache the feature matrix up to D-1 at the start of each trading
day. At each scan time T, append only the recomputed partial bar row rather
than reprocessing the full 504-day window.

Implementation: `Backtester` pre-computes and caches `FeatureEngine.compute()`
for all complete daily bars once per day. At each scan, splice in the updated
partial bar and re-run only the last-row feature computation.

### 1f. Partial DTE correction

When building a position at scan time T on day D with target DTE = N:
```python
fraction_of_day_elapsed = time_T.hour / 24.0
effective_dte = N - fraction_of_day_elapsed
t = max(effective_dte, 0.001) / 365.0
```

Minor for 21-DTE positions but correct.

### 1g. Live bot alignment

The live bot's `orchestrator.py` must use the same `entry_window_start_et`,
`entry_window_end_et`, and `scan_interval_minutes` from config. The existing
`should_avoid_new_trades()` gate (last 15 min) remains; add a complementary
`is_in_entry_window()` check that gates new entries to the configured window.
Skip `entry_window_start_et` entirely for the first 30 minutes after open
(already covered by setting start to 10:30 by default).

**Effort:** Large

---

## Fix 2 — VLMC Feature Session Alignment (Inc. 2)

**Prerequisites:** Fix 0 (intraday bar data), Fix 1 (intraday engine)

### 2a. Session tiering in `_merge_intraday_features`

Split VLMC features into two tiers based on whether they require a complete
trading session to be meaningful:

**Tier A — Temporal/session-complete features (use D-1 complete session):**
- `power_hour_momentum`
- `closing_imbalance`
- `closing_range_position`
- `hurst_wavelet_intraday`

These measure end-of-session structure. At any intraday scan time they are
meaningless or undefined for the current day. Use the prior complete session
(D-1) as a lagged predictor of the next period's outcome. This is consistent
in both training (historical D-1 sessions) and live inference (yesterday's
complete session).

**Tier B — Non-temporal features (use current partial session up to scan time T):**
- `intraday_vwap_position`
- `intraday_rsi`
- `intraday_momentum_1h`
- `session_vwap_position`
- `session_volume_front_load`
- `session_volume_shape`
- `session_high_timing` / `session_low_timing` (partial — valid early in session)
- `mfdfa_width_intraday` (if ≥ 20 bars available)
- `session_vwap_q1` / `session_vwap_q2` (VWAP quartiles up to current bar)
- `power_hour_vol_accel` / `power_hour_vwap_cross` (only after 2 PM, else 0)

### 2b. Implementation in `features.py:_merge_intraday_features`

Refactor the per-day loop to:
1. For each training day D: compute Tier A features from D-1's complete
   session; compute Tier B features from D's session up to the scan time T
   that corresponds to the training signal time.
2. Join both tiers onto the feature row for day D.

Add a helper `compute_intraday_features_tiered(session_d, session_d_minus_1, scan_time)
-> dict[str, float]` that encapsulates this logic.

### 2c. Live inference alignment

In `orchestrator.py`, remove the current separate `compute_intraday_features`
call and `live_signals.update(intraday_fractal)` for VLMC features.
Instead, pass `self._historical` as `intraday_store` to `predictor.predict()`.
The `_merge_intraday_features` path will then handle VLMC features the same
way as training — consistent by construction.

Sentiment features (`sentiment_composite`, `fear_greed`, etc.) remain in
`live_signals` as they are not time-series per-day features.

**Effort:** Medium

---

## Fix 3 — VLMC Features in Walk-forward Training (Inc. 3)

**Prerequisites:** Fix 0 (data), Fix 2 (unified session tiering)

### Current state

`walkforward.py:_train_window_model()` already passes `intraday_store` to
`predictor.train()`. The code is wired. VLMC features are absent from QQQ
training solely because the intraday store has zero QQQ rows. Once Fix 0
(data backfill) is complete, VLMC features will flow into training
automatically without further code changes.

### What to verify after Fix 0

After backfill, confirm that:
1. `FeatureEngine.compute(train_df, intraday_store=store, symbol="QQQ")`
   returns a DataFrame with VLMC columns populated (not all NaN)
2. `predictor._feature_names` includes VLMC column names after training
3. The trained model's feature importances show non-zero weight on at least
   some VLMC features
4. The live inference path (`predictor.predict(..., intraday_store=store)`)
   selects the same feature columns as training

### Eliminate the dual-path problem

After Fix 2c is implemented (passing `intraday_store` directly to
`predictor.predict()`), verify that `_add_live_signals()` no longer receives
any VLMC keys. VLMC features should only enter the model through
`_merge_intraday_features`. Add an assertion or warning log if any VLMC key
names appear in `live_signals`.

**Effort:** Verification + small cleanup (no new logic after Fix 0 + Fix 2)

---

## Fix 4 — Realistic IV from IBKR Historical Data (Inc. 4)

**Prerequisites:** Fix 0 (IV column backfilled in `daily_prices`)

### 4a. Update `_get_iv()` in `options_sim.py`

```python
def _get_iv(self, hist: pd.DataFrame) -> float:
    if "implied_vol" in hist.columns:
        iv = hist["implied_vol"].iloc[-1]
        if pd.notna(iv) and iv > 0:
            return max(float(iv), self._iv_floor)
    # Fallback: synthetic RV-based IV
    rv = realized_vol(hist["Close"].values, window=20)
    return max(rv * 1.15, self._iv_floor)
```

No other changes needed — the rest of the pricing chain already uses the
return value of `_get_iv()`.

### 4b. Interim option (before IB backfill is complete)

VIX is already fetched as a cross-asset feature. As an interim proxy:
`iv = vix_close * 1.05` where `vix_close` is the VIX bar for the same day.
This is available in the `market_context` dict passed to `FeatureEngine` and
can be forwarded to the `Backtester` as a supplementary series.

This interim approach should be clearly flagged in the backtest report output
so results from "VIX-proxy IV" and "IBKR real IV" experiments are not mixed.

**Effort:** Small

---

## Fix 5 — Options Bid-Ask Spread Model (Inc. 6)

**Prerequisites:** Fix 0 (IV data for calibration), Fix 4

### 5a. Replace flat `slippage_pct` with IV/DTE-aware spread model

Replace the global `slippage_pct=0.01` with a per-leg spread function applied
inside `_build_credit_position` and `_build_debit_position`. The function is
parameterized and Optuna-tunable:

```python
def _options_half_spread(self, iv: float, dte: int) -> float:
    """Half bid-ask spread as a fraction of mid price, per leg."""
    iv_premium  = max(0.0, iv - 0.20) * self._spread_iv_sensitivity
    dte_premium = max(0.0, 30 - dte) / 30.0 * self._spread_dte_sensitivity
    return min(self._spread_base + iv_premium + dte_premium, self._spread_cap)
```

Default parameters (pending paper-trading calibration):
- `spread_base = 0.03`   (3% half-spread at normal IV, 30 DTE)
- `spread_iv_sensitivity = 0.15`  (adds ~1.5% per 10pp IV above 20%)
- `spread_dte_sensitivity = 0.03` (adds ~3% for very short DTE)
- `spread_cap = 0.10`    (10% maximum half-spread per leg)

### 5b. Apply spread per-leg, not on net credit

The current code applies slippage to aggregate `net_credit`. Refactor to apply
the half-spread to each leg's individual price at entry and exit:

- **Short leg (selling):** `fill_price = mid * (1 - half_spread)` — you sell
  at the bid
- **Long leg (buying):** `fill_price = mid * (1 + half_spread)` — you buy
  at the ask
- **Exit (closing short leg):** `fill_price = mid * (1 + half_spread)` — you
  buy back at the ask
- **Exit (closing long leg):** `fill_price = mid * (1 - half_spread)` — you
  sell at the bid

This naturally produces higher round-trip cost for strategies with more legs.

### 5c. Expose spread parameters to Optuna

Add to the Optuna parameter space:
```python
"bt__spread_base":           trial.suggest_float("bt__spread_base", 0.02, 0.08),
"bt__spread_iv_sensitivity": trial.suggest_float("bt__spread_iv_sensitivity", 0.05, 0.25),
"bt__spread_dte_sensitivity": trial.suggest_float("bt__spread_dte_sensitivity", 0.01, 0.05),
```

The optimizer will find the spread level consistent with the paper-trading
fill quality observed empirically.

### 5d. Limit order fill quality (complementary)

The intraday execution simulation (Fix 1b) determines whether a limit at mid
is fillable within `limit_order_timeout_bars`. When not filled at mid:
- Chase fill: apply an additional `chase_premium` (e.g., 50% of the half-spread)
  to model the cost of crossing the spread to get filled

**Effort:** Medium

---

## Fix 6 — Walk-forward Params → Live Deployment (Inc. 7)

**Prerequisites:** All prior fixes + a fresh walk-forward run on the new system

### 6a. `paper_trading_mode` flag

Add to `LearningConfig`:
```yaml
learning:
  paper_trading_mode: true   # disables adaptor confidence overrides
```

In `orchestrator.py:676`, respect this flag:
```python
if self._settings.learning.paper_trading_mode:
    min_confidence = self._settings.risk.min_confidence
else:
    min_confidence = adaptor.get_confidence_override() or self._settings.risk.min_confidence
```

The adaptor still runs and logs its recommendations — it just doesn't apply
them. This provides a clean experiment: "does the walk-forward system
reproduce its P&L in paper trading?"

### 6b. Post-retrain confidence recalibration

When the live model retrains (`ModelTrainer.train_all_symbols()`), the
confidence threshold calibrated by Optuna on the old model may no longer be
valid. After each retrain, compute the empirical confidence percentile that
matches the Optuna-selected `min_confidence`'s selectivity:

1. During walk-forward, record the quantile q = percentile_rank(min_confidence,
   all_confidence_scores_on_training_set)
2. Export q alongside min_confidence in `export_production_params.py`
3. After each live retrain, derive new `effective_min_confidence` =
   percentile(q, confidence_scores_on_new_training_set)
4. Use `effective_min_confidence` instead of the raw exported value

This is optional for the initial paper trading phase (use 6a instead) but
should be implemented before going fully live.

### 6c. Model-independent vs model-dependent param export

Update `export_production_params.py` to annotate each exported param as
model-dependent (recalibrate after retrain) or model-independent (freeze):

| Param | Type | Treatment |
|---|---|---|
| `min_confidence` | Model-dependent | Recalibrate per 6b |
| `stop_loss_pct` | Model-independent | Freeze from walk-forward |
| `profit_target_pct` | Model-independent | Freeze |
| `wing_k` | Model-independent | Freeze |
| `max_hold_days` | Model-independent | Freeze |
| `delta_short` | Model-independent | Freeze |
| `iv_floor` | Model-independent | Freeze |

**Effort:** Small-Medium

---

## Implementation Order and Dependencies

```
Fix 0 (IB backfill)
  ├── Fix 4 (real IV)           — unblocked after Fix 0
  ├── Fix 3 (VLMC in training)  — unblocked after Fix 0 + Fix 2
  └── Fix 1 (intraday engine)   — unblocked after Fix 0
        ├── Fix 2 (VLMC tiering) — requires Fix 1 architecture
        │     └── Fix 3 (verify)
        └── Fix 5 (spread model) — requires Fix 1 execution layer + Fix 4

Fix 6 (param export + paper_trading_mode)
  └── requires ALL prior fixes + fresh walk-forward run
```

### Suggested sprint order

| Sprint | Fixes | Goal |
|---|---|---|
| 1 | Fix 0 | Data foundation — unblocks everything |
| 2 | Fix 4, Fix 5a-c | Realistic options pricing and spread model — quick wins, independent of engine rewrite |
| 3 | Fix 1a-f | Intraday engine rewrite — largest change, can be developed in parallel with Sprint 2 |
| 4 | Fix 2 + Fix 3 | VLMC tiering and training alignment — requires Sprint 1 + 3 data and engine |
| 5 | Fix 6 + fresh walk-forward | Paper trading deployment |

---

## New Configuration Parameters Summary

All new parameters should be added to `config.yaml` / `BacktestConfig` with
sensible defaults that preserve existing behaviour when not set:

```yaml
backtest:
  # Intraday simulation (Fix 1)
  entry_window_start_et:     "10:30"
  entry_window_end_et:       "15:30"
  scan_interval_minutes:     60
  execution_bar_size:        "5m"
  limit_order_timeout_bars:  3
  time_of_day_feature:       true

  # Options spread model (Fix 5)
  spread_base:               0.03
  spread_iv_sensitivity:     0.15
  spread_dte_sensitivity:    0.03
  spread_cap:                0.10
  chase_premium_frac:        0.50   # fraction of half-spread applied on missed fills

learning:
  # Paper trading mode (Fix 6)
  paper_trading_mode:        true
  confidence_quantile:       null   # set by export_production_params after walk-forward
```

---

## Blindspots and Residual Risks

1. **IBKR historical 5-min data completeness:** IBKR rate-limits historical
   data requests. The backfill script must paginate requests and handle
   throttling. Verify there are no gaps in the backfilled data before running
   a walk-forward.

2. **IBKR model IV vs contract IV:** The historical `OPTION_IMPLIED_VOLATILITY`
   bars are ATM 30-day IV composites. The live bot uses per-contract
   `modelGreeks.impliedVol` for OTM strikes. These differ, especially for
   far-OTM wings during stress. This gap is not fully eliminated — it is
   significantly reduced compared to the synthetic `rv * 1.15` approach.

3. **ML model is time-of-day dependent:** After the intraday engine is built,
   `entry_window_start_et` and `scan_interval_minutes` become fixed properties
   of the trained model. Changing the scan time requires re-running the full
   walk-forward. Document this constraint clearly.

4. **Walk-forward computational cost:** With a 60-min scan cadence, the
   backtest runs approximately 7-8 signal evaluations per day vs the current 1.
   With the feature cache optimization (Fix 1e), the marginal cost per
   additional scan is low. Benchmark before and after to confirm Optuna
   remains tractable.

5. **Old Exp 4 params are invalidated:** The results from Experiments 1-4
   (`min_confidence`, `wing_k`, `iv_floor` etc.) were calibrated on the pre-fix
   system (EOD signals, synthetic IV, 1% flat slippage). Do not use them as
   starting points for Optuna on the new system. Run with neutral priors first.

6. **Exit resolution on partial intraday data:** For the first walk-forward
   windows after Fix 1, the 5-min historical bars may not extend to the full
   training window start. For days without intraday bars, fall back to the
   existing EOD exit check. Log these fallback cases so you know what fraction
   of the backtest is using intraday vs EOD exit resolution.

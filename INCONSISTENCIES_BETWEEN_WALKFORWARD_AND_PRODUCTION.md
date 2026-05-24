# Backtest ↔ Live Trading Inconsistency Analysis

## Context

Promising iron condor results from experiments (Exp 4: +21.4%, Sharpe 10.86) need to survive into
production (IB paper account). The concern is that several structural differences between the
walk-forward backtest and the live bot could prevent results from being reproduced. This document
catalogues every verified inconsistency, in order of severity.

---

## Inconsistency 1 — Incomplete "today" daily bar during live scanning (Critical)

**Where it matters:** every feature the ML model uses (RSI, realized_vol, ATR, Bollinger Bands, etc.)

**Backtest** (`src/ait/backtesting/engine.py:137`):
```python
hist = self._data.iloc[:i + 1]   # includes today's COMPLETE OHLCV bar
row  = self._data.iloc[i]
```
When the backtest processes day *i*, today's bar is fully resolved: its Open/High/Low/Close/Volume
are the final end-of-day values obtained from a full session of 5-min bars resampled via
`historical.py:resample_to_daily()` (first/max/min/last/sum).

**Live bot** (`src/ait/bot/orchestrator.py:589`):
```python
hist_task = self._market_data.get_historical(symbol, days=504)
```
`get_historical()` calls `load_daily_ohlcv()` which resamples from the SQLite intraday store
including today's incrementally-stored bars. At 11 AM, today's bar has:
- Close = most recent 5-min bar's close (**current mid-day price, not EOD**)
- High/Low = range so far (will widen by close)
- Volume = partial (will be ~35–60% of full-day volume)

All features derived from today's bar — RSI, realized_vol, BB position, ATR — are computed with a
non-final close. The ML model was trained on completed bars but infers on incomplete ones.

**Impact:** Features systematically differ from their trained distribution mid-day. The direction
and confidence values the model outputs are based on data the model never saw during training.

**Fix options:**
- **A (simple, recommended):** In live scanning, always exclude today's bar from `hist` before
  feature computation. Use `hist = hist.iloc[:-1]` if the last index is today's date. Signals
  become "yesterday's EOD" signals — equivalent to the backtest assumption.
- **B (richer):** Build a synthetic "today" bar for feature computation only, and only pass it
  once the session is ≥ 50% complete (after 1 PM ET). Gives fresh intraday signal while avoiding
  severely partial bars.

---

## Inconsistency 2 — VLMC intraday features: full session (backtest) vs partial session (live) (Critical)

**Where it matters:** the 26 VLMC/fractal features merged into the feature matrix.

**Backtest** (`src/ait/ml/features.py:127`):
```python
for d, session in intraday.groupby(intraday.index.date):
    feats = self.compute_intraday_features(session)
```
Loops over historical days; each `session` is a **complete** day (≈78 bars, 9:30–4:00 PM).

**Live bot** (`src/ait/bot/orchestrator.py:649`):
```python
intraday_fractal = _FE().compute_intraday_features(intraday_full)
```
`compute_intraday_features()` isolates "today" by `intraday_df.index[-1].date()` (features.py:283).
At 11 AM this session has ≈30 bars, not 78. Features designed around a full session are
systematically biased:

| Feature | What it measures | Problem mid-day |
|---|---|---|
| `power_hour_momentum` | last 12 bars (≈3–4 PM) | last 12 bars = 10:30–11 AM at 11 AM scan |
| `closing_imbalance` | last 3 bars | closing bars are not yet available |
| `closing_range_position` | position within last 3 bars | same as above |
| `session_vwap_q3` | VWAP at 75% of session (≈2:30 PM) | only 30 bars in → Q3 index is wrong |
| `session_high_timing` | fraction of session where day high occurred | biased toward early session |
| `hurst_wavelet_intraday` | needs ≥78 bars; returns 0.0 if short | degrades to 0.0 default |

The ML model was trained with these features all measured against completed sessions.
In live trading they take different values or degrade to 0 defaults.

**Fix options:**
- **A (clean, recommended):** For today's VLMC features in live mode, use the **previous complete
  session** (yesterday). Replace `compute_intraday_features(intraday_full)` with the session for
  `intraday_full[intraday_full.index.date == yesterday]`. Today's intraday signal only feeds into
  tomorrow's prediction — consistent with how the backtest works.
- **B:** Only compute today's VLMC features if the session is ≥78 bars (post-2 PM). Use
  yesterday's values otherwise.

---

## Inconsistency 3 — VLMC features absent during backtest training, present during live inference (Significant)

**Backtest signal loop** (`src/ait/backtesting/engine.py:369`):
```python
pred = self._predictor.predict(hist)   # no live_signals, no intraday_store
```
The `DirectionPredictor` is trained in the walk-forward optimizer via the same codepath — VLMC
features are **not** in the training feature set.

**Live bot** (`src/ait/bot/orchestrator.py:653–657`):
```python
prediction = self._predictor.predict(
    hist, symbol=symbol,
    market_context=market_context,
    live_signals=live_signals,     # <-- contains 26 intraday VLMC features
)
```
`_add_live_signals()` (features.py:93–96) appends VLMC features as new columns. During training
these default to neutral (0.0). At inference they carry real values.

XGBoost/LightGBM only use features seen during training. If models were trained without VLMC
feature names, those columns are silently ignored at inference — so the 26 VLMC features
contribute **zero signal** despite the code implying they do. The walk-forward Optuna objective
function (used for `min_confidence`, `wing_k` tuning) is also optimised without these features,
meaning the parameters it selects are calibrated to a weaker model.

**Fix:** Pass `intraday_store` to `FeatureEngine().compute()` inside the walk-forward backtest
engine so VLMC features are included during Optuna tuning and OOS evaluation. Requires historical
5-min bars to be available for all walk-forward windows (they should be in `data/historical.db`).

---

## Inconsistency 4 — Synthetic IV (backtest) vs real market IV (live) (Significant)

**Backtest** (`src/ait/backtesting/engine.py:469–474`):
```python
iv = realized_vol(close_arr, window=20) * 1.15
iv = max(iv, self._iv_floor)   # floor: 20%
```
A flat 1.15× multiplier over 20-day realized vol with a 20% floor.

**Live bot** (`src/ait/data/options_chain.py:317`):
```python
impliedVol = contract.modelGreeks.impliedVol   # from IBKR real-time feed
```
Real market IV from IBKR's model Greeks.

The variance risk premium (VRP) is not constant. During stress events (e.g., QQQ's Liberation Day
−15% in April 2025), actual IV/RV ratio can reach 3–5×. The backtest underestimates both the
credit premium collected for iron condors during high-IV regimes and the mark-to-market loss
when IV spikes during the hold period.

**Fix:** No clean fix without adding a live IV data source to the backtest. Mitigation: replace
the flat 1.15× multiplier with a regime-aware VRP estimate (e.g., scale by VIX/RV ratio from
the last 30 days), or pull historical IBKR IV snapshots as a new data source for backtesting.

---

## Inconsistency 5 — Backtest entry at same-day close; live entry mid-day (Significant)

**Backtest:**
Signal uses `hist = data.iloc[:i+1]` which includes today's Close. Entry is assumed at the
close price of day *i* — equivalent to a market-on-close (MOC) order.

**Live bot:**
The trading loop runs every 5 minutes from 9:30 AM to ≈3:45 PM. An iron condor signal at 11 AM
is executed at 11 AM prices (not EOD).

Sub-problems:
1. Today's price action from 11 AM to 4 PM is unknown at entry. The backtest implicitly knows
   the full day's range when deciding to enter (look-ahead on today's Close/High/Low).
2. Option premium at 11 AM differs from EOD premium (theta, intraday IV movement).

**Fix (recommended, pairs with Fix 1A):** Only submit new position orders in the post-market
window (4:00–4:15 PM ET), using the just-closed daily bar. This matches the backtest assumption
exactly and is the approach most likely to reproduce experiment results. The existing
`_avoid_new_trades_gate` (blocks entries in last 15 minutes) would need to be inverted: block
entries before a configurable post-market open time instead.

---

## Inconsistency 6 — Slippage and bid-ask spread model (Medium)

**Backtest** (`src/ait/backtesting/engine.py:~510`): +1% buy / −1% sell uniform slippage per leg.

**Live:** Real options bid-ask spread. For OTM QQQ options this is typically 3–8% mid-to-side.
An iron condor with 4 legs at 5% each way ≈ 20% round-trip slippage — **20× the backtest assumption**.

This alone can explain a substantial performance gap on low-credit iron condors ($0.50–$1.50
collected). The options chain service already filters by `max_spread_pct: 10%` (config.yaml:38)
— feed this observed spread value into the backtest slippage model.

**Fix:** Benchmark real fill prices from paper trading. Replace flat 1% with a per-leg spread
estimate derived from historical bid-ask widths (already available from options chain snapshots).

---

## Inconsistency 7 — Walk-forward Optuna parameters vs daily-retrained live models (Medium)

Walk-forward: one model trained per window (365-day train, 63-day test). Optuna-selected
`min_confidence`, `wing_k` etc. are held fixed across the full 63-day OOS window.

Live: models retrained every morning at 7:30 AM. The live model continuously adapts; the
production parameters exported from walk-forward may not match what the daily-retrained live
model implicitly uses.

**Fix:** Ensure `scripts/export_production_params.py` exports parameters from the most recent
completed walk-forward window, and that the live bot enforces these rather than allowing
self-learning adaptation to override them during the initial deployment phase.

---

## The "Incremental 5-Min Data" Idea

This was discussed earlier. The live orchestrator already implements incremental intraday
fetching every 5 minutes (`orchestrator.py:616–629`). The idea was: use these fresh 5-min bars
to build a running "today's bar" and append it to the 504-day daily history so the
DirectionPredictor sees today's current price movement, not just yesterday's close.

This is **already happening** — Inconsistency 1 above — today's partial bar IS included in
`get_historical()` output because `load_daily_ohlcv()` resamples today's stored intraday bars.
The problem is the backtest doesn't simulate this: it waits for end of day.

So the inconsistency exists and the decision is which side to align:
- **Align live to backtest (recommended for now):** Drop today's partial bar from `hist` in live
  scanning (Fix 1A + Fix 5). Makes the live bot strictly EOD-signal-driven.
- **Align backtest to live:** Add intraday bar simulation to the backtest (larger change, revisit
  after confirming EOD approach works).

---

## Recommended Priority Order

| # | Fix | Effort | Inconsistency addressed |
|---|-----|--------|------------------------|
| 1 | Drop today's incomplete bar from `hist` in live scanning | Small | 1, 5 |
| 2 | Use yesterday's complete session for VLMC features in live | Small | 2 |
| 3 | Pass `intraday_store` to engine/predictor in backtest | Medium | 3 |
| 4 | Replace flat 1% slippage with options-spread-based estimate | Medium | 6 |
| 5 | Regime-aware VRP for synthetic IV | Medium | 4 |

Fixes 1 and 2 together produce a live bot that strictly mirrors the backtest assumption:
EOD signals on complete bars, complete sessions for VLMC features. These are the highest ROI
changes and can be validated quickly with paper trading.

---

## Files Involved

| File | Lines | Relevant to |
|------|-------|-------------|
| `src/ait/bot/orchestrator.py` | 589, 629, 649 | Fixes 1, 2 |
| `src/ait/data/market_data.py` | 55–92 | Fix 1 (`load_daily_ohlcv`) |
| `src/ait/data/historical.py` | 162–198, 229–254 | Fix 1, 2 (intraday store helpers) |
| `src/ait/ml/features.py` | 104–160, 277–383 | Fix 2, 3 (VLMC merge, compute_intraday) |
| `src/ait/backtesting/engine.py` | 134–188, 360–375 | Fix 3 (pass intraday_store) |
| `src/ait/backtesting/options_sim.py` | 469–474 | Fix 5 (IV model) |

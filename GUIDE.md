# AIT v2 — Comprehensive System Guide

> **Who this is for:** Anyone who wants to understand, operate, or extend the AIT v2 autonomous options trading system — from first-time users to developers adding new strategies.

---

## Table of Contents

1. [System Design](#1-system-design)
   - 1.1 [What AIT v2 Does](#11-what-ait-v2-does)
   - 1.2 [High-Level Architecture](#12-high-level-architecture)
   - 1.3 [Component Map](#13-component-map)
   - 1.4 [The Two Trading Loops](#14-the-two-trading-loops)
   - 1.5 [Entry Decision Pipeline](#15-entry-decision-pipeline)
   - 1.6 [Position Monitoring: The 30-Second Loop](#16-position-monitoring-the-30-second-loop)
   - 1.7 [Exit Execution](#17-exit-execution)
   - 1.8 [Market Phases & the Daily Lifecycle](#18-market-phases--the-daily-lifecycle)
   - 1.9 [Trading Strategies](#19-trading-strategies)
   - 1.10 [Machine Learning System](#110-machine-learning-system)
   - 1.11 [Risk Management](#111-risk-management)
   - 1.12 [Order Execution](#112-order-execution)
   - 1.13 [Self-Learning & Adaptation](#113-self-learning--adaptation)
   - 1.14 [Data Sources, Caching & Storage](#114-data-sources-caching--storage)
   - 1.15 [Monitoring & Observability](#115-monitoring--observability)
   - 1.16 [Key Configuration Reference](#116-key-configuration-reference)
   - 1.17 [Technology Stack](#117-technology-stack)
   - 1.18 [Key Design Decisions & Rationale](#118-key-design-decisions--rationale)
   - 1.19 [Backtesting](#119-backtesting)
   - 1.20 [Parameter Optimization](#120-parameter-optimization)

---

# 1. System Design

## 1.1 What AIT v2 Does

AIT v2 (Autonomous Intelligent Trader, version 2) is a fully automated options trading bot that:

- Connects to **Interactive Brokers (IBKR)** to place real options trades
- Scans a universe of ~7 US equities every 5 minutes during market hours
- Uses a **machine learning ensemble** (XGBoost + LightGBM) to predict whether a stock is likely to move up, down, or sideways over the next 5 days
- Selects the **best-fit options strategy** for that prediction (e.g., iron condor if sideways, long call if bullish)
- Manages open positions with **dynamic exit rules** (trailing stops, profit targets, time-decay exits) checked every 30 seconds
- **Retrains its models daily**, learns from past trades nightly, and adapts strategy sizing over time
- Sends real-time alerts via **Telegram** and exposes a **Streamlit dashboard** for live monitoring

The system is designed to run 24/7 on a dedicated machine or cloud VM, requiring no human intervention once configured. It supports both **paper trading** (simulated, risk-free) and **live trading** modes.

---

## 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        run_orchestrator.py                          │
│                      (Master Process — always running)              │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │   BotManager    │  │  APScheduler     │  │ WebServiceManager │  │
│  │ (auto-restart   │  │ (cron jobs:      │  │ Streamlit :8501   │  │
│  │  on crash)      │  │  retrain, report │  │ Flask logs :8502  │  │
│  └────────┬────────┘  │  backtest, etc.) │  └───────────────────┘  │
│           │           └──────────────────┘                         │
└───────────┼─────────────────────────────────────────────────────────┘
            │ spawns
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      src/ait/main.py                                │
│                     (Bot Process — trading brain)                   │
│                                                                     │
│  MarketScheduler ──► PRE_MARKET / MARKET_OPEN / POST_MARKET        │
│                                                                     │
│  TradingOrchestrator                                                │
│  ├─ Data Layer     (MarketDataService, OptionsChainService)         │
│  ├─ ML Layer       (DirectionPredictor, FeatureEngine)              │
│  ├─ Strategy Layer (StrategySelector, 9 strategy classes)           │
│  ├─ Risk Layer     (RiskManager, PositionSizer, CircuitBreaker)     │
│  ├─ Execution Layer(TradeExecutor, PortfolioManager)                │
│  ├─ Learning Layer (LearningEngine, ThompsonSampler)                │
│  └─ State Layer    (SQLite, DuckDB, JSON files)                     │
│                                                                     │
│              ▼ communicates via ib_insync ▼                         │
│        Interactive Brokers Gateway (port 4002/4001)                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design principle:** The orchestrator and the bot are separate processes. If the bot process throws an unhandled exception, the orchestrator's `BotManager` detects the exit code and immediately restarts the bot. The scheduled jobs keep running in the orchestrator regardless of what the bot does.

---

## 1.3 Component Map

| Module | Location | What it does |
|---|---|---|
| **Master Orchestrator** | `run_orchestrator.py` | Top-level process: spawns bot, runs scheduled jobs, serves web UIs |
| **Bot Entry Point** | `src/ait/main.py` | Loads config, connects to IBKR, starts the trading loop |
| **Trading Orchestrator** | `src/ait/bot/` | Main brain: coordinates all components during each scan cycle |
| **Market Scheduler** | `src/ait/bot/scheduler.py` | State machine that knows what phase the market is in |
| **Market Data** | `src/ait/data/market_data.py` | Fetches OHLCV and real-time quotes (IBKR → Polygon → Yahoo fallback) |
| **Options Chain** | `src/ait/data/options_chain.py` | Fetches option chains, filters by liquidity, computes Greeks |
| **Historical Store** | `src/ait/data/historical.py` | SQLite cache for OHLCV data — avoids redundant API calls |
| **Feature Engine** | `src/ait/ml/features.py` | Computes 80+ technical indicators as ML input features |
| **Direction Predictor** | `src/ait/ml/ensemble.py` | XGBoost + LightGBM ensemble → bullish / bearish / neutral + confidence |
| **Range Predictor** | `src/ait/ml/range_predictor.py` | Binary classifier → P(price stays within ±5% over 30 days); replaces confidence for iron condors |
| **Vol Magnitude Predictor** | `src/ait/ml/vol_magnitude_predictor.py` | Binary classifier → P(big move); replaces confidence for long straddles |
| **Regime Detector** | `src/ait/ml/regime.py` | Classifies current market as trending, range-bound, volatile, or calm |
| **Strategy Selector** | `src/ait/strategies/selector.py` | Picks the best strategy given ML direction + IV rank + regime |
| **Thompson Sampler** | `src/ait/strategies/thompson.py` | Bandit algorithm that prefers historically profitable strategies |
| **9 Strategy Classes** | `src/ait/strategies/` | Iron condor, spreads, long options, covered, straddles/strangles |
| **Risk Manager** | `src/ait/risk/manager.py` | Validates every trade through 13+ ordered checks |
| **Position Sizer** | `src/ait/risk/position_sizer.py` | Calculates contracts based on account size, vol, confidence, drawdown |
| **Circuit Breaker** | `src/ait/risk/circuit_breaker.py` | Halts trading after 2% daily loss |
| **PDT Guard** | `src/ait/risk/pdt_guard.py` | Blocks illegal day-trades for accounts under $25k |
| **Correlation Guard** | `src/ait/risk/correlation.py` | Prevents stacking positions in correlated sectors |
| **Capital Tier Manager** | `src/ait/risk/capital_tiers.py` | Scales strategy complexity to account size |
| **Trade Executor** | `src/ait/execution/executor.py` | Places and tracks IBKR orders, handles fills and timeouts |
| **Portfolio Manager** | `src/ait/execution/portfolio.py` | Monitors open positions every 30 sec, triggers exits |
| **Learning Engine** | `src/ait/learning/` | Post-market cycle: analyzes trade history, adapts strategy weights |
| **Counterfactual Logger** | `src/ait/learning/counterfactual.py` | Records trades that were rejected — measures missed opportunity |
| **Sentiment Engine** | `src/ait/sentiment/` | FinBERT + Finnhub + Fear & Greed + IB news → sentiment score |
| **Equity Stats Service** | `src/ait/data/equity_stats.py` | yfinance fundamentals → DuckDB `equity_stats` table (daily refresh) |
| **Fundamentals Store** | `src/ait/data/fundamentals_db.py` | SQLite CRUD for IB news and analyst recommendations |
| **IB News Service** | `src/ait/data/ib_news.py` | Fetches IB news (BRFG/DJ-N) + analyst actions (BRFUPDN), parses and persists |
| **Strategy Optimizer** | `src/ait/optimization/optimizer.py` | Optuna TPE Bayesian search over strategy + ML parameter spaces |
| **Optimization Results** | `src/ait/optimization/results.py` | Top-N trial summary, JSON save, apply best params to config |
| **Analytics** | `src/ait/monitoring/analytics.py` | Win rate, Sharpe, drawdown — tracked per strategy |
| **DuckDB Analytics** | `src/ait/monitoring/duckdb_analytics.py` | Closed trades + equity_stats table; fast analytical queries |
| **Watchdog** | `src/ait/monitoring/watchdog.py` | Monitors memory, latency, error rates, IBKR connection |
| **Dashboard** | `src/ait/dashboard/app.py` | Streamlit web UI on port 8501 |
| **Telegram Notifier** | `src/ait/notifications/` | Sends trade alerts and daily summaries |
| **State Manager** | `src/ait/bot/` | Persists open trades to SQLite; survives bot restarts |

---

## 1.4 The Two Trading Loops

During market hours, the bot runs **two interleaved loops** with different cadences and responsibilities. This is the most important thing to understand about how the system operates in real time.

```python
# From src/ait/bot/orchestrator.py
async def _trading_loop(self):
    scan_interval   = 300   # 5 minutes — look for new trades
    monitor_interval = 30   # 30 seconds — manage open positions

    time_since_scan = 0

    while running and phase == MARKET_OPEN:
        if time_since_scan >= scan_interval:
            await self._trading_cycle()        # full entry pipeline
            time_since_scan = 0
        else:
            await self._monitor_positions_fast()   # exits/stops only
        
        await asyncio.sleep(monitor_interval)
        time_since_scan += monitor_interval
```

The bot wakes up every **30 seconds**. Nine out of ten wakeups it only manages existing positions. Every tenth wakeup (at the 5-minute mark) it does the full scan looking for new entries. These are not separate threads — it is a single async loop alternating between two jobs.

### Responsibilities at Each Cadence

**Every 30 seconds — `_monitor_positions_fast()`**
- Check each open position against exit rules (stops, targets, DTE, delta breach)
- Execute any required full or partial exits
- Check IBKR for fills on pending entry and exit orders
- Check for new SEC 8-K filings on held symbols (every 10th call, ~5 minutes)

**Every 5 minutes — `_trading_cycle()`**
- Run the full entry pipeline for every symbol in the universe
- Also run the 30-second position check (the monitor is a subset of the cycle)
- Re-evaluate trade thesis validity (re-run ML on open positions)
- Check portfolio delta for hedging needs
- Process the signal queue (delayed entries waiting for better timing)

### What This Means in Practice

If you hold an AAPL iron condor and AAPL drops 3% in 2 minutes, the bot will detect the stop breach within at most 30 seconds on the next monitor cycle. It does not need to wait for the 5-minute scan. New entries, however, can only happen at the 5-minute boundary.

---

## 1.5 Entry Decision Pipeline

This section traces exactly what happens during the 5-minute scan for a single symbol — using AAPL as a concrete example throughout.

### Step 1: Parallel Data Fetch

All data for AAPL is fetched concurrently using `asyncio.gather`:

```python
hist, sentiment, iv_rank, intraday = await asyncio.gather(
    market_data.get_historical("AAPL", days=504),
    sentiment_engine.get_sentiment("AAPL"),
    estimate_iv_rank("AAPL"),
    market_data.get_intraday("AAPL", interval="5m", days=1),
)
```

**What is actually fetched vs. served from cache:**

| Data | Cache TTL | Real Fetch Frequency | Notes |
|---|---|---|---|
| Historical OHLCV (504 days) | 1 hour | ~Once per trading day | Reused across all ~10 scan cycles per hour |
| Intraday 5-min bars | 5 minutes | Every scan cycle | Used for multi-timeframe RSI alignment |
| Sentiment (Finnhub + FinBERT) | 5 minutes | Every scan cycle | Most scan cycles get cached value |
| IV rank | Derived from 1-hour OHLCV cache | ~Once per hour | Computed from realized vol percentile |
| VIX level | Per cycle (fetched at top of `_trading_cycle`) | Every 5 minutes | Shared across all symbols in the cycle |
| SPY / market context | Per cycle | Every 5 minutes | Fetched once, passed to all symbol scans |
| Options chain | No cache | Every 5 minutes | The expensive fetch; always fresh |

The 504-day OHLCV history is **not** re-downloaded every 5 minutes. It lives in an in-memory `TTLCache` for one hour. The options chain IS re-fetched every 5 minutes because bid/ask spreads go stale within seconds.

### Step 2: Feature Engineering

`FeatureEngine.compute(hist)` runs on the 504-day OHLCV DataFrame and computes all 80+ features. This is pure CPU work (pandas/numpy arithmetic) and runs in milliseconds — there is no I/O. The options chain data and live bid/ask prices **never enter the ML model** — they are only used for order pricing and liquidity filtering.

Features are organized in groups:

| Group | Examples |
|---|---|
| Momentum | RSI-14, RSI-7, MACD, MACD histogram, ROC-5/10/20 |
| Volatility | ATR-14, Bollinger width/position, realized vol 10d/20d |
| Volume | OBV change, volume/SMA-20 ratio, volume trend |
| Trend | SMA-10/20/50, EMA-12/26, MA slopes, MA crossover signal |
| Price action | Daily return, gap, candle body/wick sizes, consecutive up/down days |
| Multi-timeframe | Weekly trend alignment, weekly RSI, volume confirmation |
| IV & vol regime | IV rank proxy, vol ratio (short/long term), vol trend, vol-of-vol |
| Cross-asset | VIX level/z-score/term spread, SPY relative strength 5d/20d/60d, SPY RSI, correlation with SPY |
| Macro | 2Y/10Y yield levels, yield curve spread/inversion, DXY level/change |
| Live signals | Sentiment composite/news/FinBERT, fear/greed, put/call ratio, flow bias *(see note below)* |
| Seasonality | Day of week, month of year |

> **Note on live signal features:** The 8 sentiment and options flow features (`sentiment_composite`, `fear_greed`, `put_call_ratio`, etc.) are always set to their neutral defaults (0.0 or 1.0) during training. The model therefore learns near-zero weights for them. At inference time, real values are passed in but they have minimal effect on the prediction. In practice, sentiment influences the final trade decision as a **post-ML confidence adjustment** (Step 4 below), not as a model input. See Section 1.18 Design Decision 10 for rationale.

### Step 3: ML Prediction

`DirectionPredictor.predict(hist)` runs the XGBoost + LightGBM ensemble on the last row of features. The models are already loaded in memory; this is a single forward pass taking milliseconds.

- If AAPL has its own per-symbol model (trained on AAPL history only), that is used
- Otherwise falls back to the universal model trained on all symbols
- Output: direction (`BULLISH` / `BEARISH` / `NEUTRAL`) + confidence (0.0–1.0)
- Example: `NEUTRAL, confidence=0.71`

The prediction is blocked if confidence < the configured minimum (default 0.65).

### Step 4: Confidence Adjustments (Three Overlays)

Three adjustments are applied sequentially to the raw ML confidence:

**Sentiment overlay (±20% weight):**
```python
sentiment_adj = sentiment.composite_score * 0.20
final_confidence = clip(ml_confidence + sentiment_adj, 0, 1)
```
A strong positive news day for AAPL (composite_score = +0.5) pushes 0.71 → 0.81. A negative day pushes it down. This is where FinBERT and Finnhub actually matter.

**Multi-timeframe RSI alignment (±0.15 max):**
`MultiTimeframeAnalyzer` computes weekly, daily, and intraday trends. When all three timeframes agree (e.g., all bullish with volume confirming), confidence gets a +0.15 boost. Disagreement between timeframes penalizes by up to −0.10.

**Regime override:**
`RegimeDetector` classifies the current market. If it detects `HIGH_VOLATILITY` with >80% confidence (VIX > 30 or realized vol > 40%), the direction is forced to `NEUTRAL` regardless of what the ML predicted. This prevents directional trades during crisis conditions.

### Step 5: Gate Checks (Ordered, Any Can Abort)

Each gate can reject AAPL for the current cycle without rejecting other symbols:

1. **Confidence floor:** If `final_confidence < 0.65` after adjustments → skip AAPL this cycle. Log as counterfactual.
2. **Earnings proximity:** `EarningsCalendar.is_near_earnings("AAPL")` — if earnings are within 2 days → skip.
3. **Meta-labeler:** Secondary XGBoost trained on the bot's own trade history. Uses primary confidence + regime + VIX + IV rank + technical features to predict P(trade is profitable). If P < 0.50 → skip. Disabled until 30+ closed trades exist (`meta_label.enabled: false` in config).
4. **Options flow hard gate:** `OptionsFlowDetector` analyzes the AAPL chain for unusual put/call volume. If strong flow (bias_strength > 0.7) directly contradicts the ML direction (e.g., heavy put buying against a bullish ML signal) → reject entirely.

### Step 6: Strategy Selection

`StrategySelector.generate_all_signals()` runs all enabled strategies against the filtered AAPL options chain. Each strategy self-filters based on conditions:

- `IronCondor`: requires iv_rank ≥ 15 (configurable via `AIT_IRON_CONDOR_IV_FLOOR`). If AAPL iv_rank=45, generates a signal with short put at ~0.20 delta, short call at ~0.20 delta, wings 1x expected move away.
- `BullCallSpread`: requires direction == BULLISH. If direction is NEUTRAL → no signal.
- `LongCall`: requires direction == BULLISH and iv_rank < 60 → no signal if neutral.

**Tier 1 model gates for iron condors and straddles:**

For iron condors and short strangles, the ML confidence is **replaced** (not just adjusted) by the `RangePredictor` output:
```python
range_pred = range_predictor.predict(hist, symbol="AAPL")
# P(AAPL stays within ±5% over 30 days) = e.g. 0.72
signal.confidence = range_pred.probability_in_range
# If < 0.65 threshold → signal is dropped
```

For long straddles, confidence is replaced by `VolMagnitudePredictor` output (P(big move > 7% over 30 days)).

This is why the Tier 1 models exist: a 3-class direction predictor with 35–42% accuracy is the wrong tool for "will this stay in a range?" A purpose-built binary classifier achieves 65–75% accuracy on that question.

**Signal ranking:**

All generated signals are scored and sorted. The scoring function heavily favors iron condors (hardcoded +50 points, matching backtesting results), then weights by confidence (0–40 points), risk/reward (0–30 points), defined-risk bonus (+10), and IV alignment (±15 points). Thompson sampling then re-ranks, biasing toward strategies with better recent win rates.

### Step 7: Entry Timing Queue

Before going to risk validation, there is an RSI-based delay check:

```python
def _should_queue_signal(signal, hist):
    rsi = compute_rsi(hist["Close"], 14)
    if signal.direction == BULLISH and rsi > 65:
        return True   # Overbought — wait for pullback
    if signal.direction == BEARISH and rsi < 35:
        return True   # Oversold — wait for bounce
    return False
```

If AAPL is overbought (RSI=70) and the signal is bullish, the signal is held in `_signal_queue` for up to 3 scan cycles (~15 minutes). On each subsequent 5-minute scan, the RSI is re-checked. If it drops below 65 within that window, the signal proceeds. If RSI stays elevated for all 3 cycles, the signal expires and is logged as a counterfactual.

### Step 8: Risk Validation (13 Checks)

The signal passes through `RiskManager.validate_trade()`. Checks are ordered cheapest-to-most-expensive:

| # | Check | What it tests | On fail |
|---|---|---|---|
| 1 | Circuit breaker | Daily P&L loss ≥ 2% | REJECT all trades |
| 2 | Confidence floor | ML confidence < 0.65 | REJECT signal |
| 3 | Weekend gap risk | Friday 2:30 PM+ with short_strangle | Require 90% confidence |
| 4 | Daily trade limit | Trades taken today ≥ max_daily_trades | REJECT signal |
| 5 | Position count | Already have 5 open positions | REJECT signal |
| 6 | Duplicate symbol | Already holding AAPL in this strategy | REJECT signal |
| 7 | Correlation guard | Open positions in mega_tech sector | REDUCE size 30% per correlated position |
| 8 | Buying power | Not enough margin/cash | REJECT signal |
| 9 | Per-position max risk | Trade risk > 3% of account | REJECT signal |
| 10 | Symbol concentration | AAPL exposure > 20% of account | REJECT signal |
| 11 | Portfolio delta | Net portfolio delta would exceed ±30% | REJECT signal |
| 12 | Daily loss check | Daily loss at threshold | REJECT signal |
| 13 | PDT guard | Would constitute illegal day-trade | REJECT exit order |

**Position sizing** runs after all checks pass:
```
Base contracts = (account_value × 5%) ÷ (option_cost_per_contract)

× Confidence scalar:   confidence < 0.70 → 0.5×; < 0.80 → 0.75×
× Volatility scalar:   IV > 60% → 0.5×; linear from 1.0x at IV=20%
× IV rank alignment:   selling strategies get +0.4× at IV rank 100; −0.2× at IV rank 0
× Strategy scalar:     iron_condor → 0.4×; long_straddle → 1.2×
× Drawdown scalar:     ≥3 consecutive losing days → 0.5×; ≥5 → 0.25×
× VIX scalar:          VIX ≥ 25 → 0.75×; VIX ≥ 30 → 0.50×
× Correlation haircut: 30% reduction per correlated open position (capped at 70% total reduction)
× Account scale cap:   max 1 contract per $10k of account value
```

### Step 9: Order Placement

`TradeExecutor.execute_signal()`:
1. Qualifies all 4 option contracts with IBKR in a single batch call
2. Builds a 4-leg combo order (all legs as a single atomic transaction)
3. For a credit spread: places limit at `mid − $0.05` (slightly below mid to improve fill probability)
4. Waits up to 90 seconds for fill
5. On fill: writes to SQLite + inserts `open_positions` row for HWM tracking + sends Telegram alert
6. On timeout: cancels order, logs as missed trade

### Step 10: Counterfactual Logging

Any signal that was rejected at any step (confidence, gate checks, risk validation, timing queue expiry) is recorded in `CounterfactualTracker`. The entry price is logged. After the trade would have expired, the bot checks what the actual outcome would have been, allowing the system to quantify whether its filters are blocking good trades or correctly avoiding bad ones.

---

## 1.6 Position Monitoring: The 30-Second Loop

Once an AAPL iron condor is open, the entry pipeline is no longer relevant to it. All management runs in `_monitor_positions_fast()` / `_evaluate_position()`, called every 30 seconds.

### Data Access Pattern

The 30-second loop is deliberately lightweight. It does not re-run the ML model, re-fetch the options chain, or re-run the full feature pipeline. It uses:

- **Current underlying price:** `get_current_price("AAPL")` → calls `get_quote()` which has a 15-second TTL. At most one real IBKR quote request per 15 seconds per symbol.
- **Unrealized P&L:** Estimated from the underlying price movement against the entry credit, not from re-fetching the spread's actual bid/ask. This is an approximation; the real spread value is only updated at the 5-minute scan.
- **Greeks (delta breach check):** Reads from `self._ibkr.ib.portfolio()` — a passive snapshot of what IBKR last reported. Not a live subscription. If IBKR hasn't updated the portfolio item recently, the delta reading may be stale; the check returns `None` (no action) in that case.
- **OHLCV for volatility-adjusted stops:** Uses the 1-hour OHLCV cache that was already populated by the entry scan.

### High Water Mark Tracking

Every 30-second evaluation updates the high water mark:
```sql
UPDATE open_positions SET high_water_mark = MAX(high_water_mark, ?) WHERE trade_id = ?
```
This is persisted to SQLite immediately. If the bot restarts, the trailing stop resumes from the correct peak P&L rather than from zero, preventing premature exits after a restart.

### Dynamic Stop Calculation

The effective stop level has three tiers based on the trade's peak P&L (HWM):

```
Tier 1 — HWM < breakeven_trigger (default 30%):
    effective_stop = −initial_stop_loss_pct (default −50%)
    Example: sold iron condor for $1.80 credit, stop at −$0.90

Tier 2 — HWM just crossed breakeven_trigger:
    effective_stop = MAX(0, HWM − trailing_stop_pct)
    If HWM=30% and trailing=25%: stop = MAX(0, 0.05) = 5%
    The position can no longer lose money once this tier is reached

Tier 3 — HWM well above breakeven:
    effective_stop = HWM − trailing_stop_pct
    Example: HWM=80%, trailing=25% → stop at 55%
    As the position continues profiting, the floor rises with it
```

Stops are also widened for high-volatility underlyings: a 30-day realized vol of 40% (e.g., NVDA) gets a 1.5× multiplier on stop width vs. a low-vol symbol at 10% vol.

### Exit Condition Checks (Priority Order)

Checks run in this exact priority order. The first condition that triggers stops evaluation:

| Priority | Condition | Exit Type | Notes |
|---|---|---|---|
| 1 | `pnl_pct <= effective_stop` | Full exit | Dynamic stop — see above |
| 2 | `pnl_pct >= take_profit` | Full exit | DTE-adjusted target — see below |
| 3 | Short leg ITM, DTE ≤ 1 | Full exit | Assignment risk — force close |
| 4 | DTE ≤ 5 | Full exit | Gamma risk — time decay exit |
| 5 | `\|delta\| > 0.50` | Full exit | Neutral strategy gone directional |
| 6 | Earnings within 2 days AND pnl > 0 | Full exit | IV crush pre-close |
| 7 | Macro event tomorrow (FOMC/CPI/NFP) | Full exit | Only if `AIT_SKIP_MACRO_EVENTS=1` |
| 8 | `pnl_pct >= 50% of credit` | Partial exit (33%) | First profit milestone |
| 9 | `pnl_pct >= 100% of credit` | Partial exit (33%) | Second profit milestone |
| 10 | PDT guard triggered | Block exit | Prevents day-trade violation |

**Partial exits** (items 8 and 9) close a fraction of the position while the rest continues. The system records which P&L levels have been triggered in SQLite (`partial_exits` JSON array) so the same level is never double-triggered.

### Time-Decay Adjusted Take-Profit

The take-profit target is not fixed. As DTE shrinks, the bot becomes less patient and takes profits at lower thresholds:

| DTE Remaining | Long Strategy Target | Short (Credit) Strategy Target |
|---|---|---|
| > 20 days | +100% of debit paid | +50% of credit received |
| 11–20 days | +75% of debit | +40% of credit |
| 6–10 days | +50% of debit | +30% of credit |
| ≤ 5 days | +25% of debit | +20% of credit |

The rationale: with 5 DTE, gamma acceleration makes holding risky, and a position that has already returned 20% of its credit value has done most of what it will ever do in the time remaining.

### Thesis Re-Evaluation (5-Minute Cycle Only)

Once per scan cycle (not the fast 30-second loop), open positions undergo thesis re-evaluation. The original entry context is stored in SQLite at trade open:

```python
state.save_trade_context(
    trade_id=..., direction="neutral", confidence=0.71,
    regime="range_bound", vix=18.5, iv_rank=0.45, ...
)
```

The bot then re-runs the ML and regime detector on fresh data. If any of the following is true, the position is flagged as "thesis invalidated" and closed:

- Direction flipped with >70% confidence (e.g., entered NEUTRAL, now strongly BEARISH)
- Regime shifted from `trending_up/down` to `high_volatility` with >70% confidence
- VIX has spiked more than 30% since entry

---

## 1.7 Exit Execution

When `should_exit=True` on a position, `_execute_exit()` runs in the 30-second monitor loop.

### For a Multi-Leg Position (Iron Condor)

```
_execute_exit()
    ↓
_close_multi_leg()
    ├─ Re-qualify all 4 leg contracts with IBKR
    ├─ Request live market data for the combo contract
    ├─ Wait 0.5s for market data to arrive
    ├─ Calculate current mid price from bid/ask
    │   ├─ If mid available: place limit order at mid (not mid−$0.05 — closing, not opening)
    │   └─ Fallback: place market order if no mid available
    ├─ Set trade status to CLOSING in SQLite
    └─ Register exit order ID with TradeExecutor for fill tracking
```

### Fill Confirmation (30-Second Loop)

The exit order is not considered complete when placed — it is considered complete when filled. `check_fills()` runs every 30 seconds:

1. Scans IBKR's trade list for the registered exit order ID
2. If found as "filled": reads `avgFillPrice` for actual exit price
3. Calculates realized P&L:
   ```python
   pnl = (entry_credit - exit_debit) × 100 × contracts
   pnl -= 0.65 × legs_per_side × contracts × 2  # entry + exit commissions
   ```
4. Calls `state.close_trade()` which:
   - Updates SQLite trade record with exit price, P&L, exit reason
   - Dual-writes the closed trade to DuckDB analytics store
5. Updates daily stats (total P&L, win/loss count)
6. Updates Thompson sampler (win/loss recorded for the iron_condor arm)
7. Updates drift detector (was the ML direction correct for this trade?)
8. Sends Telegram alert with final P&L

### Exit Order Failure Handling

If the exit order is rejected or cancelled by IBKR:
- Status reverts from `CLOSING` back to `FILLED` in SQLite
- On the next 30-second cycle, `_evaluate_position()` will re-detect the exit condition and place a new exit order
- If an exit order sits in `PENDING` status for more than 5 minutes without filling, it is cancelled and retried

---

## 1.8 Market Phases & the Daily Lifecycle

The system uses a state machine to behave differently depending on what time it is:

```
OFF_HOURS (5:00 PM – 9:00 AM ET)
│  Bot sleeps. Watchdog still alive.
│
▼
PRE_MARKET (9:00 – 9:30 AM ET)
│  1. Ensure ML models are trained and ready
│  2. Reconcile open positions against IBKR account
│  3. Run learning cycle (adapt strategy weights from yesterday)
│  4. Validate data quality
│
▼
MARKET_OPEN (9:30 AM – 4:00 PM ET)
│  ┌─ Every 5 minutes: full scan cycle (entry pipeline)
│  └─ Every 30 seconds: position monitoring (exit rules)
│
▼
POST_MARKET (4:00 – 4:15 PM ET)
│  1. Final position reconciliation
│  2. Learning cycle (analyze today's trades, adapt for tomorrow)
│  3. Generate and send daily P&L report
│  4. Log counterfactual outcomes
│
▼
OFF_HOURS → (repeat)

─────────────────────────────────────────────────────
Scheduled Jobs (run by APScheduler, independently):
  07:30 AM ET  Mon–Fri  → Retrain ML models on fresh data
  04:30 PM ET  Mon–Fri  → Generate daily report
  08:00 PM ET  Sunday   → Run walk-forward backtest
  1st of month          → Clean up old logs and temp files
  Every 2 minutes       → Health check + auto-restart bot if crashed
─────────────────────────────────────────────────────
```

---

## 1.9 Trading Strategies

All strategies inherit from `BaseStrategy` and implement a `generate_signals()` method. The system currently supports **9 strategies** organized into 3 groups:

### Group 1: Premium Selling (favor high IV environments)

**Iron Condor** — `strategies/iron_condor.py`
- Simultaneously sell an OTM call spread and an OTM put spread
- Profit if the stock stays in a range until expiration
- Best when: IV rank > 15%, stock expected to stay flat
- Entry requirement: IV rank ≥ 15% (configurable via `AIT_IRON_CONDOR_IV_FLOOR`)
- Confidence used: `RangePredictor` output (P(stays in ±5%) ≥ 0.65), not direction model
- Target: close at 50% of credit; stop at 2× credit received

**Cash-Secured Put** — `strategies/covered.py`
- Sell an OTM put, hold enough cash to buy the stock if assigned

**Short Strangle** — `strategies/straddles.py`
- Sell OTM call + OTM put simultaneously (undefined risk)
- Only used in appropriate account tiers; requires 90% confidence on Friday afternoons

### Group 2: Defined-Risk Spreads (directional with limited loss)

**Bull Call Spread** — `strategies/spreads.py`
- Buy ATM call, sell OTM call (net debit)
- Used when: ML says BULLISH, IV rank is moderate

**Bear Put Spread** — `strategies/spreads.py`
- Buy ATM put, sell lower OTM put (net debit)
- Used when: ML says BEARISH, IV rank is moderate

### Group 3: Long Options & Long Volatility (favor low IV environments)

**Long Call / Long Put** — `strategies/long_options.py`
- Outright directional plays; only when IV rank < 60%

**Long Straddle** — `strategies/straddles.py`
- Buy ATM call + ATM put (net debit)
- Confidence used: `VolMagnitudePredictor` output (P(move > 7%) ≥ 0.65), not direction model
- Used when: low IV rank, volatility expansion expected

**Covered Call** — `strategies/covered.py`
- Own stock + sell OTM call against it

### How the System Picks a Strategy

```
StrategySelector.generate_all_signals()
    │
    ├─ For each enabled strategy:
    │   strategy.generate_signals()    (each self-filters on IV rank, direction, etc.)
    │
    ├─ Collect all signals
    │
    └─ Score and rank signals:
           Iron condor:         +50 pts  (hardcoded priority)
           ML confidence:       0–40 pts
           Risk/reward ratio:   0–30 pts
           Defined risk bonus:  +10 pts
           IV alignment:        0–15 pts (selling higher as IV rises; buying higher as IV falls)
           Wide bid/ask:        −5 pts   (spread_pct > 5%)

    └─ ThompsonSampler re-ranks using historical win rates

    └─ Tier 1 model gates:
           Iron condor/strangle: confidence → RangePredictor P(in range)
           Long straddle:        confidence → VolMagnitudePredictor P(big move)
           If result < 0.65: signal dropped

    └─ RiskManager.validate_trade() — final 13-check validation
```

### Standard Exit Rules (all strategies)

| Rule | Trigger | Action |
|---|---|---|
| Take profit (time-decay adjusted) | P&L reaches 20–50% of max credit (DTE-dependent) | Close entire position |
| Stop loss | Loss reaches 2× credit received or −50% of debit | Close entire position |
| Trailing stop | Price falls 25% from high-water mark (after breakeven trigger) | Close |
| Breakeven lock | P&L reaches +30% | Move stop to entry price |
| Partial exit 1 | P&L reaches +50% of credit | Close 33% of position |
| Partial exit 2 | P&L reaches +100% of credit | Close another 33% |
| Time decay | DTE ≤ 5 days | Close |
| Assignment risk | Short leg deep ITM near expiry | Close |
| Earnings pre-close | Earnings within 2 days AND winning | Close (IV crush risk) |
| Macro event | FOMC / CPI / NFP tomorrow | Close (only if `AIT_SKIP_MACRO_EVENTS=1`) |
| Thesis invalidation | ML direction flipped, regime shifted, VIX spike >30% | Close |

---

## 1.10 Machine Learning System

### What the ML System Predicts

The **Direction Predictor** answers: *"Over the next 5 trading days, will this stock go up ≥ 1%, down ≥ 1%, or stay flat?"*

- Output label: `BULLISH` / `BEARISH` / `NEUTRAL`
- Output confidence: `0.0 – 1.0` (trades only execute if ≥ 0.65)

The **Tier 1 models** answer different, more targeted questions:

| Model | Question | Used For |
|---|---|---|
| `RangePredictor` | P(max\|return\| < 5% over 30 days) | Iron condor confidence (replaces direction confidence) |
| `VolMagnitudePredictor` | P(max\|return\| > 7% over 30 days) | Long straddle confidence (replaces direction confidence) |

The Tier 1 models achieve 65–75% accuracy on their respective binary questions, significantly outperforming the 35–42% accuracy of the 3-class direction predictor when applied to range/breakout prediction.

### Models

Two models run in parallel and their outputs are averaged (50/50):

| Model | Library | Strength |
|---|---|---|
| XGBoost | `xgboost >= 2.0` | Fast, handles sparse features well |
| LightGBM | `lightgbm >= 4.3` | Efficient on large datasets, good with categorical features |

**Per-symbol models:** Each symbol gets its own model pair. If a symbol has insufficient data (< 100 samples), it falls back to a universal model.

### Training Pipeline

```
1. Fetch 504 days (2 years) of daily OHLCV per symbol
2. Compute all 80+ features
3. Label each day: +1 (BULLISH), −1 (BEARISH), 0 (NEUTRAL)
   based on 5-day forward return vs. ±1.0% threshold
4. Walk-forward cross-validation with 5-day purge gap
5. Train XGBoost + LightGBM
6. Evaluate on held-out test window using balanced accuracy
7. If new model accuracy < old model accuracy − 5%: automatic rollback
8. Save to models/ensemble.pkl
```

**Retraining schedule:** Daily at 7:30 AM ET and on-demand via `--retrain` flag.

### The Training / Inference Mismatch: Sentiment Features

The feature set includes 8 sentiment and options flow features (`sentiment_composite`, `sentiment_news`, `sentiment_finbert`, `fear_greed`, `put_call_ratio`, etc.). These features have an important asymmetry:

- **During training:** All 8 features are set to their neutral defaults (0.0 or 1.0) for every row. Historical sentiment scores are not available in the training pipeline.
- **During inference:** Real-time sentiment values are passed in via the `live_signals` dict.

The consequence is that the model learns effectively zero weight for these features during training. At inference time, the values change but the model ignores them. This is not a bug that causes incorrect predictions — it means these features have no effect inside the XGBoost/LightGBM models.

**Why this is not fixed yet:** Sentiment's effect on the trade decision is implemented correctly as a post-ML confidence adjustment (Step 4 in section 1.5), where real sentiment values are applied with proper weighting. This is arguably the right architecture. Wiring historical sentiment into the ML model would require backfilling years of dated sentiment scores per symbol, which is a significant data engineering task. See Section 1.18 Design Decision 10 for a fuller discussion.

**Historical sentiment data options (for future improvement):**

| Source | Coverage | Cost | Format |
|---|---|---|---|
| [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) | 4,775 S&P 500 companies, 1999–2023 | Free (HuggingFace) | CSV, 30GB, ChatGPT sentiment scores |
| Finnhub `company_news` API | Historical headlines, several years | Free tier (60/min) | JSON headlines; requires running FinBERT locally |
| Financial PhraseBank | ~5,000 labelled sentences | Free | Not time-series aligned; useful for FinBERT fine-tuning only |
| Benzinga via Polygon.io | Historical news + analyst ratings | Paid | API |

FNSPID is the most directly applicable — it includes sentiment scores already computed, covers all symbols in the AIT universe, and includes the period most relevant for training. However, research from the FNSPID paper itself showed that "adding sentiment scores *modestly* enhances performance on the transformer-based model." For highly liquid large-cap ETFs and mega-cap stocks (SPY, QQQ, NVDA, AMD), news sentiment is rapidly priced in and the marginal ML gain may not justify the data engineering cost.

### Drift Detection

The `DriftDetector` monitors live prediction accuracy in a rolling window. If accuracy drops more than 5% below baseline, it triggers an out-of-schedule retrain. This protects against market regime changes making old models stale.

---

## 1.11 Risk Management

Every trade signal passes through 13 validation checks before an order is placed. Checks are ordered from cheapest to most expensive computation.

### Validation Pipeline (in order)

| # | Check | What it tests | On fail |
|---|---|---|---|
| 1 | Circuit breaker | Daily P&L loss ≥ 2% → halt trading | REJECT all trades |
| 2 | Confidence floor | ML confidence < 0.65 | REJECT signal |
| 3 | Weekend gap risk | Friday 2:30 PM+ + short_strangle | REJECT if confidence < 90% |
| 4 | Daily trade limit | Trades taken today ≥ max | REJECT signal |
| 5 | Position count | Already have 5 open positions | REJECT signal |
| 6 | Duplicate symbol | Already holding this symbol + strategy | REJECT signal |
| 7 | Correlation guard | Open positions in correlated sectors | REDUCE size 30% per correlated position |
| 8 | Buying power | Not enough margin/cash | REJECT signal |
| 9 | Per-position max risk | Trade risk > 3% of account | REJECT signal |
| 10 | Symbol concentration | Symbol exposure > 20% of account | REJECT signal |
| 11 | Portfolio delta | Net portfolio delta would exceed ±30% | REJECT signal |
| 12 | Daily loss check | Daily loss at threshold | REJECT signal |
| 13 | PDT guard | Would constitute illegal day-trade | REJECT exit order |

### Position Sizing

```
Base = (account_value × 5%) ÷ (option_price_per_contract × 100)

× Confidence scalar:  conf < 0.70 → 0.5×; conf < 0.80 → 0.75×; conf ≥ 0.80 → 1.0×
× Volatility scalar:  IV ≤ 20% → 1.0×; linear to 0.5× at IV ≥ 60%
× IV rank alignment:  selling strats: 0.8× at rank 0 → 1.2× at rank 100
× Strategy scalar:    iron_condor → 0.4×; bull/bear spread → 0.6×; long → 1.0×
× Drawdown scalar:    ≥3 consecutive losing days → 0.5×; ≥5 → 0.25×
× VIX scalar:         VIX ≥ 25 → 0.75×; VIX ≥ 30 → 0.50×
× Correlation haircut: 30% per correlated open position, max 70% total
× Account scale cap:  floor=1, cap=account_value ÷ 10,000
```

### Capital Tiers

| Tier | Account Size | Allowed Strategies | Max Positions | Universe |
|---|---|---|---|---|
| Micro | $0 – $2k | Bull/bear call/put spreads only | 2 | SPY only |
| Small | $2k – $5k | Spreads + small iron condors | 3 | SPY, QQQ, IWM, AMD, AAPL |
| Medium | $5k – $25k | Full iron condors + spreads + long options | 5 | Full 7-symbol universe |
| Large | $25k+ | All strategies, no PDT limits | 8 | Full universe |

---

## 1.12 Order Execution

### Order Lifecycle

```
Signal approved by RiskManager
        │
        ▼
TradeExecutor.execute_signal()
        │
        ├─ Qualify contracts with IBKR (batch call for all legs simultaneously)
        │
        ├─ Build order:
        │   ├─ Single-leg (long call/put): passive limit at mid − (improvement × half-spread)
        │   └─ Multi-leg (spread/condor):  combo limit at mid − $0.05
        │
        ├─ Reject if bid-ask spread > 15% of mid (stale/illiquid quote)
        │
        ├─ Place order via ib_insync
        │
        ├─ Wait up to 90 seconds for fill
        │   ├─ If filled:       record in SQLite + open_positions + send Telegram
        │   ├─ If partial fill: cancel remainder after 30s, record partial
        │   └─ If timeout:      cancel order, log as missed trade
        │
        └─ Track slippage (actual fill − expected mid price)
```

### Why Combo Orders for Multi-Leg Strategies

Spreads and iron condors have 2–4 legs. IBKR combo orders fill all legs as a single atomic transaction — either all legs fill or none do. This prevents **legging risk** (where one leg fills but another doesn't, leaving an unhedged naked position).

### Order Pricing

- **Single-leg entries:** Passive limit — places the order between mid and the natural side (bid for buys). Default improvement_pct=0.20, meaning 20% of the way from mid toward bid.
- **Multi-leg entries:** Limit at mid − $0.05 for credits, mid + $0.05 for debits.
- **Multi-leg exits:** Tries to get current mid from IBKR market data; falls back to market order if mid is unavailable.
- Never uses market orders for multi-leg options entries.

---

## 1.13 Self-Learning & Adaptation

### Learning Cycle (Post-Market)

```
1. TradeAnalyzer reads last 30 days of closed trades from DuckDB
2. Generates TradeInsights:
   ├─ Which strategies are winning / losing?
   ├─ Which symbols are consistently profitable?
   ├─ Is confidence calibration accurate?
   └─ Any patterns in exit timing (peak P&L vs realized)?
3. StrategyAdaptor applies bounded changes:
   ├─ Boost multiplier for winning strategies (max 1.5×)
   ├─ Reduce multiplier for losing strategies (min 0.3×)
   ├─ Iron condor floor: never goes below 0.3×
   ├─ Block specific symbols that have lost consistently
   └─ Adjust confidence floor if calibration is off
4. Saves updated weights to state
5. Changes take effect next trading day
```

### Thompson Sampling

Each strategy has a Beta distribution over its win probability, updated after every closed trade. At selection time, the system samples from each distribution and picks the strategy with the highest sample. This naturally balances exploration (trying underused strategies) vs. exploitation (favoring winners). State persists in `data/thompson_state.json`.

### Counterfactual Logging

When a trade is rejected, the system tracks what would have happened. After the trade's natural holding period passes, it checks actual price movement against the entry price. This data quantifies whether filters are protecting against bad trades (correct rejects) or blocking good ones (missed opportunities).

---

## 1.14 Data Sources, Caching & Storage

### Data Sources (Fallback Chain)

```
1. Interactive Brokers (ib_insync)
   ├─ Real-time quotes (bid/ask/last/volume)
   ├─ Historical OHLCV (daily and 5-min bars)
   ├─ Options chains (all expirations and strikes)
   ├─ News headlines (BRFG, DJ-N, DJ-RTG, DJ-RTPRO, DJNL) → fundamentals.db
   └─ Analyst recommendations (BRFUPDN / Briefing.com) → fundamentals.db
        ↓ on failure
2. Polygon.io (API key required)
   └─ Historical daily data (free tier: 5 calls/minute)
        ↓ on failure
3. Yahoo Finance (yfinance — no key required)
   ├─ Historical OHLCV, fallback for anything above
   └─ Equity fundamentals (P/E, beta, sector, analyst targets) → equity_stats table
```

Options chain data only comes from IBKR. If IBKR disconnects, no new trades can be entered, but the 30-second monitor loop continues to manage existing positions using the last-known underlying price.

### Data Fetch Frequency and Cache TTLs

Understanding what is fetched live vs. served from cache is important for understanding the system's actual I/O load:

| Data Type | Cache TTL | Fetch Frequency During Market Hours | Used For |
|---|---|---|---|
| Daily OHLCV (504 bars) | 1 hour | ~Once per trading day | ML feature computation, IV rank |
| Intraday 5-min bars | 5 minutes | Every scan cycle | Multi-timeframe features, entry timing |
| Real-time quote (underlying) | 15 seconds | At most 4 times/min per symbol | Position monitoring, order pricing |
| Options chain | None (always fresh) | Every 5 minutes | Strategy signal generation |
| VIX level | Per scan cycle | Every 5 minutes | Position sizing, regime detection |
| Sentiment scores | 5 minutes | At most once per scan cycle | Confidence adjustment |
| IBKR account snapshot | 30 seconds | ~Every 30 seconds | Buying power checks |
| Portfolio Greeks | Passive (IBKR push) | When IBKR updates it | Delta breach check |
| FRED macro data (yield curve, DXY) | 1 hour | ~Once per trading day | ML cross-asset features |

**Key implication:** The 504-day OHLCV history, which is the foundation for all ML features, is not re-downloaded every 5 minutes. It is fetched once and cached for one hour. The bot effectively uses the same historical dataset for approximately 10 consecutive scan cycles before refreshing.

### Historical Sentiment Datasets

For future improvement of the training pipeline (see Section 1.10 for context):

| Dataset | Coverage | Access | Sentiment Method | Size |
|---|---|---|---|---|
| [FNSPID](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) | 4,775 S&P 500 companies, 1999–2023 | Free via HuggingFace | ChatGPT-4 | 30 GB |
| Financial PhraseBank | ~5,000 labelled sentences | Free | Human-annotated | Small |
| Finnhub `company_news` | Per-symbol, several years back | Free API (60/min) | Raw headlines only | API |

FNSPID is the most immediately applicable for training — it covers all AIT universe symbols, is pre-scored with sentiment values, and spans years of market conditions. The FNSPID paper found that "adding sentiment scores *modestly* enhances performance on transformer-based models." Given that the sentiment effect in AIT is already captured as a post-ML overlay rather than a model input, the priority for this improvement is medium.

### Storage

| Store | Technology | What's in it |
|---|---|---|
| `data/ait_state.db` | SQLite | Open trades, open_positions (HWM/partials), trade_context, KV store |
| `data/ait_analytics.duckdb` | DuckDB | Closed trades, daily stats, trade context, equity_stats — fast analytical queries |
| `data/historical.db` | SQLite | OHLCV cache — avoids refetching the same data |
| `data/fundamentals.db` | SQLite | IB news headlines (scored) + analyst recommendations (firm/action/rating/target) |
| `data/optuna.db` | SQLite | Optuna study history — created on demand, supports resumable runs |
| `data/thompson_state.json` | JSON | Thompson sampler win/loss counts per strategy |
| `data/counterfactual_log.json` | JSON | Rejected trade outcomes |
| `models/ensemble.pkl` | Pickle | Current trained direction XGBoost + LightGBM models |
| `models/range.pkl` | Pickle | Current trained RangePredictor models |
| `models/vol_magnitude.pkl` | Pickle | Current trained VolMagnitudePredictor models |
| `models/archive/` | Pickle files | Versioned model backups (for rollback) |

**Dual-write on trade close:** When a trade is closed, `StateManager` writes to both SQLite (operational state) and DuckDB (analytics). SQLite is the source of truth for live operation (open positions, pending orders). DuckDB is used for all analytical queries (dashboard, learning engine, reporting).

---

## 1.15 Monitoring & Observability

### Logs

| File | Contents |
|---|---|
| `logs/ait.log` | Main structured log (all events, rotating 10 MB × 5 files) |
| `logs/orchestrator.log` | Master process log (restarts, scheduled jobs) |
| `logs/bot_stdout.log` | Raw stdout capture from bot subprocess |

View logs in real time:
```bash
python tail_logs.py          # color-coded terminal log tail
# or open http://localhost:8502  # Flask web log viewer
```

### Watchdog

The `Watchdog` runs inside the bot process and monitors:
- **IBKR connection** — reconnects automatically on disconnect
- **Trading loop heartbeat** — detects if the main loop hangs
- **Memory usage** — alerts if the process exceeds 500 MB
- **API response times** — flags latency spikes
- **Error rates** — alerts if > 10 errors occur in a window

### Dashboard

The Streamlit dashboard at `http://localhost:8501` shows:
- Current open positions (entry price, current P&L, HWM, Greeks)
- Trade history with filters (by symbol, strategy, date range)
- Win rate and average P&L by strategy
- Portfolio-level P&L chart (cumulative)
- System health status (IBKR connection, last scan time, error count)
- ML confidence calibration (win rate by confidence band)
- Feature importance tab (which features the models are actually using)
- Live vs backtest comparison panel

### Telegram Alerts

The bot sends Telegram messages for:
- Every trade opened (symbol, strategy, strikes, credit/debit, contracts)
- Every trade closed (P&L, reason for exit)
- Partial exits (contracts closed, P&L realized, remaining)
- Daily P&L summary (after market close)
- Circuit breaker triggered
- IBKR disconnection / reconnection
- Self-learning adaptations applied

---

## 1.16 Key Configuration Reference

All user-facing parameters live in `config.yaml`. Below are the most important ones:

### Trading Universe
```yaml
trading:
  universe: [SPY, QQQ, IWM, DIA, NVDA, AMZN, AMD]
  scan_interval_seconds: 300
  max_daily_trades: 5
  trading_hours_only: true
```

### Risk Limits
```yaml
risk:
  max_daily_loss_pct: 0.02        # Circuit breaker: halt at 2% daily loss
  max_consecutive_losses: 3       # Pause after 3 losses in a row
  pause_minutes_after_losses: 30
  min_confidence: 0.65
```

### Position Limits
```yaml
positions:
  max_open_positions: 5
  max_position_pct: 0.05          # 5% of portfolio per position
  max_portfolio_delta: 0.30
  max_portfolio_risk_pct: 0.02
```

### Options Filters
```yaml
options:
  delta_range: [0.20, 0.50]
  dte_range: [14, 45]
  min_open_interest: 100
  min_volume: 50
  max_bid_ask_spread_pct: 0.10
```

### Exit Rules
```yaml
exit:
  trailing_stop_pct: 0.25         # Trail 25% below HWM
  breakeven_trigger_pct: 0.30     # Move stop to entry after +30% P&L
  initial_stop_loss_pct: 0.50     # Fixed stop before breakeven activates
  partial_exit_levels:
    - pnl_pct: 0.50               # At +50% credit: close 33%
      close_pct: 0.33
    - pnl_pct: 1.00               # At +100% credit: close another 33%
      close_pct: 0.33
  time_decay_scaling: true
  volatility_adjusted_stops: true
```

### Iron Condor IV Floor
```bash
# In environment or .env file — not in config.yaml
AIT_IRON_CONDOR_IV_FLOOR=15      # Min IV rank to enter iron condors
AIT_SKIP_MACRO_EVENTS=0          # Set to 1 to flatten positions before FOMC/CPI/NFP
```

### Sentiment — IB News Weight
```yaml
sentiment:
  ib_news_weight: 0.20            # Weight of IB-sourced pre-scored news (0.0 disables)
```
The IB news branch is active when `FundamentalsStore` is provided to `SentimentEngine` and `data/fundamentals.db` contains recent news rows for the symbol.

### Walk-Forward Optimization
```yaml
# These fields are on WalkForwardConfig, not config.yaml
# Pass them as CLI flags to run_backtest.py:
#   --optimize-per-window     enable per-window Optuna tuning
#   --optimize-n-trials 50    trials per window (default 50)
```

---

## 1.17 Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.11 – 3.13 |
| Broker API | ib_insync | ≥ 0.9.86 |
| ML — Gradient Boosting | XGBoost + LightGBM | ≥ 2.0 / ≥ 4.3 |
| ML — Framework | scikit-learn | ≥ 1.4 |
| Options Greeks | py-vollib | ≥ 1.0.1 |
| Sentiment | transformers (FinBERT) + torch | ≥ 4.38 / ≥ 2.2 |
| Data | pandas + numpy + scipy | latest |
| Market calendars | pandas-market-calendars | latest |
| Market data (backup) | yfinance + polygon-api-client | latest |
| Scheduling | APScheduler | ≥ 3.10 |
| Operational state | SQLite (built-in) | — |
| Analytics | DuckDB | ≥ 1.0 |
| Parameter optimization | Optuna | ≥ 3.6 |
| Dashboard | Streamlit + Plotly | ≥ 1.31 / ≥ 5.19 |
| Log viewer | Flask | — |
| Structured logging | structlog | ≥ 24.1 |
| Config models | pydantic | ≥ 2.6 |
| Notifications | Telegram Bot API | — |
| Testing | pytest + pytest-asyncio | — |

**Note on Apple Silicon / Intel Macs:** PyTorch 2.x requires Python 3.11 on Intel Macs. If you are on an Intel Mac, pin Python to 3.11.

---

## 1.18 Key Design Decisions & Rationale

### Decision 1: Separate Orchestrator and Bot Processes

**What:** `run_orchestrator.py` and `src/ait/main.py` run as two separate OS processes.

**Why:** Trading systems must be resilient to crashes. If the bot process throws an unhandled exception, the orchestrator's `BotManager` detects the exit code and immediately restarts the bot. Scheduled jobs (retrain, reports, health checks) keep running regardless.

**How to apply:** Never merge these two processes. Add new background tasks to the orchestrator's APScheduler, not to the bot's main loop.

---

### Decision 2: Two Loops at Different Cadences (5-min / 30-sec)

**What:** The trading loop has two cadences: a 5-minute scan for new entries and a 30-second monitor for position management. They share a single async loop rather than running as separate threads or processes.

**Why:** Entry decisions (ML prediction, options chain analysis, strategy selection, risk validation) are expensive: they involve I/O, model inference, and multi-step computation. Running them every 30 seconds on all symbols would be wasteful and would hammer the IBKR API rate limits. Exit decisions, by contrast, only need the current underlying price — a single lightweight quote request — and can safely run on the faster cadence. The 30-second monitor ensures that a gap down on an open position is caught within 30 seconds, even though the entry scan only runs every 5 minutes.

**Trade-off:** The unrealized P&L displayed during the 30-second loop is estimated from the underlying price movement, not from re-fetching the actual spread bid/ask. This is a deliberate approximation — the true spread value is only updated at the 5-minute scan. For stop/target logic this is accurate enough; for precise P&L reporting it is an estimate.

---

### Decision 3: 13+ Ordered Risk Checks (Fail-Fast Pipeline)

**What:** Every trade signal passes through a fixed sequence of validation checks ordered from cheapest to most expensive computation.

**Why:** Options trading mistakes are expensive and often irreversible within a session. The ordered pipeline means cheap checks (circuit breaker, position count) run first, so the system never wastes time computing Greeks or buying power for a signal that would have been blocked anyway. It also makes the system's behavior auditable.

**How to apply:** When adding new risk rules, insert them in order of computational cost. Never bypass the pipeline even for "obvious" high-confidence signals.

---

### Decision 4: Tier 1 Models Replace Direction Confidence for Non-Directional Strategies

**What:** For iron condors and short strangles, the XGBoost/LightGBM direction confidence is completely replaced by the `RangePredictor` output (P(stays in range)). For long straddles, it is replaced by `VolMagnitudePredictor` output (P(big move)).

**Why:** Iron condors are range strategies — they have nothing to do with direction. A model predicting "up/down/neutral" with 35–42% accuracy on a 3-class problem is the wrong tool for "will this stay in a range?" The `RangePredictor` is a binary classifier targeting exactly the right question and achieves 65–75% balanced accuracy. Using the direction model's confidence for iron condors would be like using a hammer to drive a screw — it technically works sometimes, but the tool doesn't match the task.

**How to apply:** If you add new strategies, decide which question they actually need answered: direction, range, or magnitude. Build or use the Tier 1 model that answers that question, rather than forcing direction confidence to proxy for it.

---

### Decision 5: Thompson Sampling for Strategy Selection

**What:** Strategy selection uses a multi-armed bandit (Thompson sampling) rather than a fixed rotation.

**Why:** Market regimes change. A strategy that works well for 3 months may underperform the next 3. Thompson sampling solves this automatically — strategies that are winning get sampled more often; those that are losing get sampled less. Crucially, it never fully stops trying any strategy, so it detects when a previously poor strategy becomes profitable again.

**How to apply:** Do not delete `data/thompson_state.json` between sessions — it contains the accumulated win/loss history. If you add a new strategy, it starts with a uniform prior and will be explored automatically.

---

### Decision 6: Self-Learning with Bounded Adaptations

**What:** The nightly learning cycle adapts strategy weights with hard bounds: multipliers stay in [0.3×, 1.5×], and iron condors have a floor of 0.3× (never fully disabled).

**Why:** Unconstrained adaptation is dangerous. A bad run of losses (which can be random variance, not a true signal) could permanently disable a profitable strategy. The floors ensure every strategy stays in rotation at a meaningful level.

---

### Decision 7: Per-Symbol ML Models with Universal Fallback

**What:** Each symbol gets its own XGBoost + LightGBM pair. Symbols with < 100 training samples fall back to a universal model.

**Why:** SPY and NVDA have fundamentally different volatility and momentum characteristics. A universal model averages across these differences. Per-symbol models capture stock-specific patterns while the universal fallback means new symbols degrade gracefully rather than failing.

---

### Decision 8: Purge Gap in Walk-Forward Cross-Validation

**What:** Between train and test windows, there is a mandatory 5-day gap (`gap_days=5`).

**Why:** The ML labels are based on 5-day forward returns. Without a gap, the label for day T-4 would use data from T through T+1, overlapping with the test window and inflating apparent accuracy. The 5-day purge matches the label horizon exactly.

**How to apply:** If you change the label horizon, update `gap_days` to match. Setting `gap_days=0` silently inflates backtest performance.

---

### Decision 9: Counterfactual Logging of Rejected Trades

**What:** When the risk manager or any gate rejects a trade signal, the system logs what would have happened if the trade had been taken.

**Why:** Without counterfactual data, you only see trades that were taken. You cannot tell whether a tightened confidence floor is protecting you from bad trades or just reducing trade count with no safety benefit. The counterfactual log lets operators quantify the opportunity cost of each risk rule.

**How to apply:** Review `data/counterfactual_log.json` periodically, especially after tightening a risk limit. The dashboard's Self-Learning tab shows a filter accuracy score summarizing this data.

---

### Decision 10: Sentiment as Post-ML Overlay, Not Model Feature

**What:** Sentiment scores (FinBERT, Finnhub, Fear/Greed) are applied as a confidence adjustment **after** the ML prediction rather than as input features to XGBoost/LightGBM.

**Why — Training/Inference Mismatch:** Historical sentiment scores aligned to specific trading dates are not available in the current training pipeline. If sentiment features are included in the model but always set to 0 during training, the model learns zero weight for them and they have no effect at inference time. The post-ML overlay approach is honest: it uses sentiment where data is actually available (real-time), and the ML model is trained only on data that was available historically.

**Why — Signal Strength:** Research consistently shows that news sentiment provides only modest predictive improvement for highly liquid large-cap stocks (the AIT universe). The VIX-based Fear/Greed indicator already captures most of the macro-sentiment signal and is derived from the same OHLCV data the model already sees. Adding FinBERT headlines as ML features would add training complexity for limited marginal gain.

**What a proper fix would look like:** If historical sentiment is backfilled using FNSPID (1999–2023 coverage) or by running FinBERT offline on archived Finnhub news, the sentiment features could be included in training with real historical values. The expected improvement is modest but real, particularly for event-driven symbols like NVDA and AMD. This is tracked in `TODO.md` under "Sentiment as ML feature."

---

## 1.19 Backtesting

### What is Walk-Forward Backtesting?

A standard ("simple") backtest trains a model on all available data and tests it on the same data — this is meaningless. **Walk-forward backtesting** simulates what would actually happen in live trading:

```
[──── Train (1 year) ────][gap][── Test (3 months) ──]
                                    ↓ slide forward
                    [──── Train (1 year) ────][gap][── Test (3 months) ──]
```

Each test period is genuinely out-of-sample. The aggregate of all test windows gives a realistic estimate of live performance.

### Running the Backtester

```bash
python run_backtest.py                           # Quick defaults
python run_backtest.py --symbols SPY QQQ --capital 25000
python run_backtest.py --iv-floor 20             # Lower = more condor trades
python run_backtest.py --compare-exits           # Fixed vs trailing stops
```

### How the Backtester Works

**Data:** Yahoo Finance (5 years of daily OHLCV). No IBKR connection required.

**Options pricing:** Black-Scholes with `IV = realized_vol × 1.15` to simulate the typical IV premium over realized vol.

**The Tier 1 range model is included:** When backtesting iron condors, the `RangePredictor` is trained per window and gates entries, matching the live system's behavior.

### Interpreting Results

| Metric | What it means | Good threshold |
|---|---|---|
| Win Rate | % of trades profitable | > 55% debit; > 65% credit |
| Profit Factor | Gross wins ÷ gross losses | > 1.5 solid; > 2.0 strong |
| Sortino Ratio | Return / downside-only volatility | > 1.5 |
| RAROC | Return on capital actually deployed | > 50% annually |
| Consistency | % of test windows profitable | > 70% |

### Common Backtesting Mistakes

1. **Over-tuning parameters on backtest results** — if you keep adjusting `--min-confidence` and `--iv-floor` until numbers look good, you are overfitting to historical data.
2. **Setting `--gap-days 0`** — silently inflates results by creating look-ahead bias.
3. **Testing a single symbol** — a strategy that works on SPY alone may fail on NVDA. Always test across the full universe.
4. **Confusing backtest Sharpe with live Sharpe** — Black-Scholes is smooth; real markets have bid-ask gaps and liquidity constraints.

---

## 1.20 Parameter Optimization

The system includes a fully integrated **Optuna-based Bayesian parameter optimizer** (`src/ait/optimization/`). It uses the TPE sampler with MedianPruner to find optimal strategy parameters and ML hyperparameters in far fewer trials than grid search.

### Running the Optimizer

```bash
# Strategy parameters — iron condor + sharpe objective, 100 trials
python run_optimizer.py \
  --strategies iron_condor \
  --symbols SPY QQQ \
  --n-trials 100 \
  --objective sharpe_ratio \
  --storage sqlite:///data/optuna.db   # optional: enables resumable studies

# ML hyperparameters (XGBoost + LightGBM)
python run_optimizer.py \
  --strategies iron_condor \
  --symbols SPY QQQ \
  --optimize-ml \
  --n-trials 50

# Apply best params back to config.yaml
python run_optimizer.py --strategies iron_condor --symbols SPY --n-trials 20 --apply

# Resume a previous study (trials accumulate across runs)
python run_optimizer.py \
  --study-name iron_condor_study \
  --storage sqlite:///data/optuna.db \
  --n-trials 50    # adds 50 more trials to the existing study
```

### Objective Functions

| Objective | Formula | Best when |
|---|---|---|
| `sharpe_ratio` | P&L / σ(P&L) | Risk-adjusted return |
| `composite` | `0.4×sharpe + 0.4×win_rate − 0.2×|max_drawdown|` | Balanced tuning |
| `profit_factor` | gross_wins / gross_losses (capped at 10) | Raw edge focus |
| `win_rate` | % of profitable trades | Consistency focus |

### Parameter Spaces

Each strategy has a defined search space in `param_spaces.py`:

| Parameter (iron condor) | Search Range |
|---|---|
| `delta_min` | float [0.15, 0.30] |
| `delta_max` | float [0.25, 0.45] |
| `dte_min` | int [7, 21] |
| `dte_max` | int [28, 60] |
| `min_confidence` | float [0.55, 0.80] |
| `stop_loss_pct` | float [0.30, 0.70] |
| `profit_target_pct` | float [0.40, 0.80] |
| `iv_floor` | int [10, 30] |

ML model spaces (`xgboost`, `lightgbm`) cover `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, and model-specific params.

### Walk-Forward Integration

Set `optimize_per_window: true` in `WalkForwardConfig` (or pass `--optimize-per-window` to `run_backtest.py`) to run Optuna on each training slice before testing it. Each window finds its own best parameters — this is computationally expensive but produces the most realistic out-of-sample results.

```bash
python run_backtest.py \
  --symbols SPY QQQ \
  --optimize-per-window \
  --optimize-n-trials 50   # Optuna trials per walk-forward window
```

### Manual Tuning (still valid for quick iteration)

**Backtester CLI flags** (no restart needed):

| Parameter | CLI flag | Default | Controls |
|---|---|---|---|
| Min ML confidence | `--min-confidence` | 0.65 | Trade filter threshold |
| Iron condor IV floor | `--iv-floor` | 30.0 | Min IV rank for condors |
| Range model threshold | `--range-confidence` | 0.55 | Min P(in range) for condors |
| Trailing vs fixed stops | `--trailing-stop` / `--compare-exits` | On | Exit style comparison |

```
1. Run baseline: python run_backtest.py --iv-floor 30
2. Run candidate: python run_backtest.py --iv-floor 20
3. Compare: total return, Sharpe/Sortino, consistency, max drawdown
4. If improvement across multiple metrics: update config.yaml
```

---

*This document is a living reference. Last updated: 2026-05-05*

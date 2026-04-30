# AIT v2 — Comprehensive System Guide

> **Who this is for:** Anyone who wants to understand, operate, or extend the AIT v2 autonomous options trading system — from first-time users to developers adding new strategies.

---

## Table of Contents

1. [System Design](#1-system-design)
   - 1.1 [What AIT v2 Does](#11-what-ait-v2-does)
   - 1.2 [High-Level Architecture](#12-high-level-architecture)
   - 1.3 [Component Map](#13-component-map)
   - 1.4 [Data Flow — A Trade From Start to Finish](#14-data-flow--a-trade-from-start-to-finish)
   - 1.5 [Market Phases & the Daily Lifecycle](#15-market-phases--the-daily-lifecycle)
   - 1.6 [Trading Strategies](#16-trading-strategies)
   - 1.7 [Machine Learning System](#17-machine-learning-system)
   - 1.8 [Risk Management](#18-risk-management)
   - 1.9 [Order Execution](#19-order-execution)
   - 1.10 [Self-Learning & Adaptation](#110-self-learning--adaptation)
   - 1.11 [Data Sources & Storage](#111-data-sources--storage)
   - 1.12 [Monitoring & Observability](#112-monitoring--observability)
   - 1.13 [Key Configuration Reference](#113-key-configuration-reference)
   - 1.14 [Technology Stack](#114-technology-stack)
   - 1.15 [Key Design Decisions & Rationale](#115-key-design-decisions--rationale)
   - 1.16 [Backtesting](#116-backtesting)
   - 1.17 [Parameter Optimization](#117-parameter-optimization)

---

---

# 1. System Design

## 1.1 What AIT v2 Does

AIT v2 (Autonomous Intelligent Trader, version 2) is a fully automated options trading bot that:

- Connects to **Interactive Brokers (IBKR)** to place real options trades
- Scans a universe of ~14 US equities every 5 minutes during market hours
- Uses a **machine learning ensemble** (XGBoost + LightGBM) to predict whether a stock is likely to move up, down, or sideways over the next 5 days
- Selects the **best-fit options strategy** for that prediction (e.g., iron condor if sideways, long call if bullish)
- Manages open positions with **dynamic exit rules** (trailing stops, profit targets, time-decay exits)
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

**Key design principle:** The orchestrator and the bot are separate processes. If the bot crashes, the orchestrator restarts it automatically — the master process keeps running no matter what.

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
| **Feature Engine** | `src/ait/ml/features.py` | Computes 60+ technical indicators as ML input features |
| **Direction Predictor** | `src/ait/ml/ensemble.py` | XGBoost + LightGBM ensemble → bullish / bearish / neutral + confidence |
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
| **Sentiment Engine** | `src/ait/sentiment/` | FinBERT + Finnhub news + Fear & Greed → sentiment score |
| **Analytics** | `src/ait/monitoring/analytics.py` | Win rate, Sharpe, drawdown — tracked per strategy |
| **DuckDB Analytics** | `src/ait/monitoring/duckdb_analytics.py` | Fast analytical queries on closed trades |
| **Watchdog** | `src/ait/monitoring/watchdog.py` | Monitors memory, latency, error rates, IBKR connection |
| **Dashboard** | `src/ait/dashboard/app.py` | Streamlit web UI on port 8501 |
| **Telegram Notifier** | `src/ait/notifications/` | Sends trade alerts and daily summaries |
| **State Manager** | `src/ait/bot/` | Persists open trades to SQLite; survives bot restarts |

---

## 1.4 Data Flow — A Trade From Start to Finish

Below is exactly what happens during one 5-minute scan cycle, step by step:

```
1. SCAN TRIGGER (every 5 minutes, 9:30 AM – 4:00 PM ET)
   └─ For each symbol in the universe (e.g., SPY, AAPL, NVDA...):

2. FETCH DATA
   ├─ Real-time quote (bid/ask/last/volume) ← IBKR
   ├─ Historical OHLCV (daily + 5-min bars) ← IBKR → Polygon → Yahoo
   └─ Options chain (all strikes/expirations) ← IBKR

3. COMPUTE FEATURES
   └─ FeatureEngine computes 60+ indicators:
      RSI, MACD, ATR, Bollinger Bands, IV rank,
      VIX level, volume ratios, intraday momentum, etc.

4. ML PREDICTION
   └─ DirectionPredictor runs XGBoost + LightGBM:
      Output → direction: BULLISH / BEARISH / NEUTRAL
               confidence: 0.0 – 1.0 (must be ≥ 0.65 to trade)

5. SENTIMENT OVERLAY (weight: 20%)
   └─ SentimentEngine: FinBERT on recent headlines + Fear & Greed index
      Adjusts effective confidence up or down

6. STRATEGY SELECTION
   └─ StrategySelector considers:
      ├─ ML direction (bullish/bearish/neutral)
      ├─ IV rank (high IV → sell premium; low IV → buy options)
      ├─ Market regime (trending/range-bound/volatile)
      └─ Thompson Sampler (favors historically profitable strategies)
      Output → best strategy signal with target strikes + expiry

7. RISK VALIDATION (13 checks, in order)
   ├─ Circuit breaker open? (daily loss ≥ 2%)      → REJECT
   ├─ Confidence ≥ 0.65?                           → REJECT if not
   ├─ Open positions < 5?                          → REJECT if full
   ├─ Already holding this symbol?                 → REJECT
   ├─ Correlated positions?                        → REDUCE size or REJECT
   ├─ Buying power sufficient?                     → REJECT if not
   ├─ Portfolio delta within limits?               → REJECT if exceeded
   ├─ Not in post-loss cooldown?                   → REJECT if cooling
   └─ PDT day-trade limit OK?                      → REJECT if would breach

8. POSITION SIZING
   └─ PositionSizer calculates contracts:
      Base = account value × 5% ÷ option cost
      × confidence scalar (lower confidence → fewer contracts)
      × volatility scalar (high IV → fewer contracts)
      × drawdown scalar (50% size after 3 losses; 25% after 5)
      × VIX scalar (75% size at VIX ≥ 25; 50% at VIX ≥ 30)

9. ORDER EXECUTION
   ├─ Qualify contracts with IBKR
   ├─ Build combo order (for spreads/condors) or single order
   ├─ Place limit order at mid ± $0.05
   ├─ Wait up to 90 seconds for fill
   └─ On fill: record trade in SQLite + send Telegram alert

10. POSITION MONITORING (every 30 seconds, independently)
    └─ For each open position, check (in priority order):
       ├─ Trailing stop hit? (25% below high-water mark)     → CLOSE
       ├─ Breakeven triggered? (+30% P&L → move stop to 0)
       ├─ Partial exit level? (+50% → close 33%; +100% → close another 33%)
       ├─ Take profit? (50% of credit received)              → CLOSE
       ├─ Time decay exit? (DTE ≤ 5 or held > 21 days)      → CLOSE
       ├─ Assignment risk? (short ITM near expiry)           → CLOSE
       ├─ Delta breach? (|Δ| > 0.50 for neutral strategies)  → CLOSE
       ├─ Earnings tomorrow? (IV crush risk)                 → CLOSE
       └─ Macro event tomorrow? (FOMC/CPI/NFP)              → CLOSE
```

---

## 1.5 Market Phases & the Daily Lifecycle

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
│  ┌─ Every 5 minutes: full scan cycle (steps 1–9 above)
│  └─ Every 30 seconds: position monitoring (step 10 above)
│
▼
POST_MARKET (4:00 – 5:00 PM ET)
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

## 1.6 Trading Strategies

All strategies inherit from `BaseStrategy` and implement a `generate_signals()` method. The system currently supports **9 strategies** organized into 3 groups:

### Group 1: Premium Selling (favor high IV environments)

**Iron Condor** — `strategies/iron_condor.py`
- Simultaneously sell an OTM call spread and an OTM put spread
- Profit if the stock stays in a range until expiration
- Best when: IV rank > 50%, stock expected to stay flat
- Max profit: total credit received × 100
- Max loss: spread width − credit (defined risk)
- Entry requirement: IV rank ≥ 15%
- Target: close at 50% of credit; stop at 2× credit received

**Cash-Secured Put** — `strategies/covered.py`
- Sell an OTM put, hold enough cash to buy the stock if assigned
- Profit if stock stays above the strike at expiration
- Best when: you want to potentially own the stock at a discount

**Short Strangle** — `strategies/straddles.py`
- Sell OTM call + OTM put simultaneously (undefined risk)
- Profit from time decay if stock stays in a wide range
- Higher premium than iron condor, but no protection on the wings
- Only used in appropriate account tiers

### Group 2: Defined-Risk Spreads (directional with limited loss)

**Bull Call Spread** — `strategies/spreads.py`
- Buy ATM call, sell OTM call (net debit)
- Profit if stock rises above the short strike by expiration
- Used when: ML says BULLISH, IV rank is moderate

**Bear Put Spread** — `strategies/spreads.py`
- Buy ATM put, sell lower OTM put (net debit)
- Profit if stock falls below the short strike by expiration
- Used when: ML says BEARISH, IV rank is moderate

### Group 3: Long Options & Long Volatility (favor low IV environments)

**Long Call** — `strategies/long_options.py`
- Buy a call option outright (debit trade)
- Profit if stock rises significantly
- Used when: ML says BULLISH with high confidence, IV rank < 30%

**Long Put** — `strategies/long_options.py`
- Buy a put option outright (debit trade)
- Profit if stock falls significantly
- Used when: ML says BEARISH with high confidence, IV rank < 30%

**Long Straddle** — `strategies/straddles.py`
- Buy ATM call + ATM put simultaneously (net debit)
- Profit if stock makes a big move in either direction
- Used when: low IV rank, volatility expansion expected (pre-earnings)

**Covered Call** — `strategies/covered.py`
- Own stock + sell OTM call against it
- Collect premium; cap upside at the short strike
- Used when: holding stock, want to generate income

### How the System Picks a Strategy

**Relevant files:**
- [`src/ait/strategies/selector.py`](src/ait/strategies/selector.py) — `StrategySelector` class, the primary decision point
- [`src/ait/strategies/thompson.py`](src/ait/strategies/thompson.py) — Thompson sampler for historical win-rate weighting
- [`src/ait/risk/manager.py`](src/ait/risk/manager.py) — final pass validation on the chosen signal

**Call sequence (during each scan cycle):**

```
StrategySelector.generate_all_signals()           # selector.py:86
    │
    ├─ For each enabled strategy:
    │   strategy.generate_signals()               # e.g. IronCondor.generate_signals()
    │   (each strategy self-filters — IronCondor won't emit a signal if IV rank < 15%)
    │
    ├─ Collect all signals from all strategies
    │
    └─ StrategySelector._rank_signals()           # selector.py:144
           Score each signal:
           ├─ Iron condor:        +50 pts (hardcoded priority — best backtest performer)
           ├─ ML confidence:      0–40 pts  (confidence × 40)
           ├─ Risk/reward ratio:  0–30 pts  (capped at 3.0)
           ├─ Defined risk:       +10 pts
           ├─ IV alignment:       0–15 pts  (selling strategies score higher as IV rises;
           │                                 buying strategies score higher as IV falls)
           └─ Wide bid/ask:       −5 pts    (spread_pct > 5%)

ThompsonSampler.sample()                          # thompson.py
    └─ Biases selection toward strategies with higher historical win rates

RiskManager.validate_trade()                      # manager.py
    └─ Final 13-check validation on the top-ranked signal
```

**Recommended strategies by condition** (`StrategySelector.get_recommended_strategies()`, selector.py:199):

```
IV Rank ≥ 50% + Neutral direction  →  Iron Condor (first choice)
IV Rank ≥ 50% + Bullish            →  Cash-Secured Put or Bull Call Spread
IV Rank ≥ 50% + Bearish            →  Bear Put Spread
IV Rank < 30% + Any direction      →  Long Straddle or Long Call/Put
IV Rank 30–50% + Bullish           →  Bull Call Spread
IV Rank 30–50% + Bearish           →  Bear Put Spread
```

The **Thompson Sampler** layers on top of this logic: it tracks each strategy's historical win rate and biases the selection toward strategies that have been profitable recently, balancing exploration (trying underused strategies) with exploitation (repeating winners).

### Standard Exit Rules (all strategies)

| Rule | Trigger | Action |
|---|---|---|
| Take profit | P&L reaches 50% of max credit | Close entire position |
| Stop loss | Loss reaches 2× credit received | Close entire position |
| Trailing stop | Price falls 25% from high-water mark | Close |
| Breakeven lock | P&L reaches +30% | Move stop to entry price |
| Partial exit 1 | P&L reaches +50% of debit paid | Close 33% of position |
| Partial exit 2 | P&L reaches +100% of debit paid | Close another 33% |
| Time decay | DTE ≤ 5 days, or held > 21 days | Close |
| Assignment risk | Short leg deep ITM near expiry | Close |
| Macro event | FOMC / CPI / NFP tomorrow | Close (flatten risk) |
| Earnings | Earnings within 1 day | Close (IV crush risk) |

---

## 1.7 Machine Learning System

### What the ML System Predicts

The ML system answers one question per symbol: **"Over the next 5 trading days, will this stock go up ≥ 1.5%, down ≥ 1.5%, or stay flat?"**

- Output label: `BULLISH` / `BEARISH` / `NEUTRAL`
- Output confidence: `0.0 – 1.0` (trades only execute if ≥ 0.65)

### Models

Two models run in parallel and their outputs are averaged (50/50):

| Model | Library | Strength |
|---|---|---|
| XGBoost | `xgboost >= 2.0` | Fast, handles sparse features well |
| LightGBM | `lightgbm >= 4.3` | Efficient on large datasets, good with categorical features |

**Per-symbol models:** Each of the ~14 symbols in the universe gets its own model pair, trained on that symbol's history. If a symbol has insufficient data, it falls back to a universal model trained on all symbols.

### Input Features (60+)

| Category | Features |
|---|---|
| **Momentum** | RSI 14, RSI 7, MACD, MACD signal, MACD histogram, Rate-of-Change (5/10/20 day) |
| **Volatility** | ATR 14, ATR%, Bollinger upper/lower/width/position, realized vol (10-day, 20-day) |
| **Volume** | On-Balance Volume, volume ratio, VWAP proxy, volume trend, volume surge flag |
| **Trend** | SMA 20, SMA 50, SMA 200, ADX proxy |
| **Price action** | Close > open ratio, gap up/down flag, candle body size |
| **Multi-timeframe** | Intraday RSI, intraday momentum, VWAP position, range compression |
| **Options market** | IV rank, IV vs realized vol ratio, vol regime, vol mean-reversion signal |
| **Market structure** | Put/call ratio, skew proxy, term structure proxy |
| **Cross-asset** | VIX level, VIX change, VIX trend, SPY relative strength, market breadth |
| **Seasonality** | Day of week, month of year |

### Training Pipeline

```
1. Fetch 504 days (2 years) of daily OHLCV per symbol
2. Compute all features
3. Label each day: +1 (BULLISH), -1 (BEARISH), 0 (NEUTRAL)
   based on 5-day forward return vs. ±1.5% threshold
4. Walk-forward cross-validation with a purge gap
   (prevents look-ahead bias between train and validation sets)
5. Train XGBoost + LightGBM
6. Evaluate on held-out test window using balanced accuracy
   (balanced because NEUTRAL dominates the label distribution)
7. If new model's balanced accuracy < old model's − 5%:
   → Automatic rollback to previous model
8. Save winner to models/ensemble.pkl
```

**Retraining schedule:** Daily at 7:30 AM ET (before market open), and on-demand via `--retrain` flag.

### Drift Detection

The `DriftDetector` (`src/ait/ml/drift.py`) monitors live prediction accuracy. If accuracy drops more than 5% below the baseline, it triggers an out-of-schedule retrain. This protects against market regime changes that make old models stale.

---

## 1.8 Risk Management

Every trade signal passes through 13 validation checks before an order is placed. Checks are ordered from cheapest to most expensive computation — the pipeline fails fast on obvious rejections.

### Validation Pipeline (in order)

| # | Check | What it tests | On fail |
|---|---|---|---|
| 1 | Circuit breaker | Daily P&L loss ≥ 2% → halt trading | REJECT all trades |
| 2 | Confidence floor | ML confidence < 0.65 | REJECT signal |
| 3 | Position count | Already have 5 open positions | REJECT signal |
| 4 | Duplicate symbol | Already holding a position in this symbol | REJECT signal |
| 5 | Correlation guard | Open positions in correlated sectors | REDUCE size by 30% per correlated position |
| 6 | Buying power | Not enough margin/cash for this trade | REJECT signal |
| 7 | Portfolio delta | Net portfolio delta would exceed ±30% | REJECT signal |
| 8 | Cooldown period | Still in post-loss pause (30 min after 3 consecutive losses) | REJECT signal |
| 9 | PDT guard | Would constitute an illegal day-trade (< $25k account) | REJECT exit order |

### Position Sizing

The number of contracts is calculated by multiplying several scalars together:

```
Base contracts = (account value × 5%) ÷ (option cost per contract)

× Confidence scalar:  confidence < 0.70 → 0.5×; confidence < 0.80 → 0.75×
× Volatility scalar:  high IV → fewer contracts (options are more expensive)
× Strategy scalar:    iron condor → 0.4×; long calls/puts → 1.0×
× Drawdown scalar:    ≥ 3 consecutive losses → 0.5×; ≥ 5 losses → 0.25×
× VIX scalar:         VIX ≥ 25 → 0.75×; VIX ≥ 30 → 0.50×
```

### Capital Tiers

The system automatically scales strategy complexity based on account size:

| Tier | Account Size | Allowed Strategies | Max Positions | Spread Width |
|---|---|---|---|---|
| Micro | $0 – $2k | Long options only | 2 | $1–2 wide |
| Small | $2k – $5k | Spreads + small condors | 3 | $2–5 wide |
| Medium | $5k – $25k | Full iron condors | 5 | Up to $10 wide |
| Large | $25k+ | All strategies, no PDT limits | 8 | Unlimited |

---

## 1.9 Order Execution

### Order Lifecycle

```
Signal approved by RiskManager
        │
        ▼
TradeExecutor.execute_signal()
        │
        ├─ Qualify contracts with IBKR (verifies option contracts exist)
        │
        ├─ Build order:
        │   ├─ Single-leg (long call/put): LimitOrder at mid price
        │   └─ Multi-leg (spread/condor):  ComboOrder (single IBKR order for all legs)
        │
        ├─ Place order via ib_insync
        │
        ├─ Wait up to 90 seconds for fill
        │   ├─ If filled: record in SQLite + send Telegram alert
        │   ├─ If partial fill: cancel remainder, record partial
        │   └─ If timeout: cancel order, log as missed trade
        │
        └─ Track slippage (actual fill − expected mid price)
```

### Why Combo Orders for Multi-Leg Strategies?

Spreads and iron condors have 2–4 legs. IBKR supports **combo orders** that fill all legs as a single atomic transaction — either all legs fill or none do. This prevents **legging risk** (where one leg fills but the other doesn't, leaving an unhedged position).

### Order Pricing

- Default: limit order at the **mid-price** of the bid/ask spread
- Aggressive: mid ± $0.05 to improve fill probability in fast markets
- The system does not use market orders for multi-leg options (too much slippage risk)

---

## 1.10 Self-Learning & Adaptation

The learning system runs nightly after market close and adapts the bot's behavior based on recent trade outcomes.

### Learning Cycle (Post-Market)

```
1. TradeAnalyzer reads last 30 days of closed trades from DuckDB
2. Generates TradeInsights:
   ├─ Which strategies are winning / losing?
   ├─ Which symbols are consistently profitable?
   ├─ Is confidence calibration accurate? (high-confidence trades winning more?)
   └─ Any patterns in exit timing?
3. StrategyAdaptor applies bounded changes:
   ├─ Boost multiplier for consistently winning strategies (max 1.5×)
   ├─ Reduce multiplier for consistently losing strategies (min 0.3×)
   ├─ Iron condor floor: never goes below 0.3× (always kept in rotation)
   ├─ Block specific symbols that have lost consistently
   └─ Adjust confidence floor if calibration is off
4. Saves updated weights to state
5. Changes take effect next trading day
```

### Thompson Sampling

The Thompson Sampler (`strategies/thompson.py`) implements a **multi-armed bandit** algorithm for strategy selection:

- Each strategy has a Beta distribution over its win probability, updated after every trade
- At selection time, sample from each distribution and pick the strategy with the highest sample
- This naturally balances **exploration** (trying underused strategies) vs. **exploitation** (favoring winners)
- State persists across sessions in `data/thompson_state.json`

### Counterfactual Logging

When a trade is rejected by the risk manager, the system still tracks what would have happened:
- Logs the rejected signal and its parameters
- After the trade would have expired, records whether it would have been profitable
- This data helps identify if risk limits are too conservative (missing good trades) or too loose (rejecting needed protection)

---

## 1.11 Data Sources & Storage

### Data Sources (Fallback Chain)

The system tries each source in order, falling back to the next if a call fails or returns incomplete data:

```
1. Interactive Brokers (ib_insync)
   ├─ Real-time quotes (bid/ask/last/volume)
   ├─ Historical OHLCV (daily and 5-min bars)
   └─ Options chains (all expirations and strikes)

2. Polygon.io (API key required)
   └─ Historical daily data (free tier: 5 calls/minute limit)

3. Yahoo Finance (yfinance — no key required)
   └─ Historical OHLCV, fallback for anything above
```

**Important:** Options chain data only comes from IBKR. If the IBKR connection is lost, the system cannot trade — it will pause until reconnected.

### Data Granularity & Refresh Rates

AIT v2 is a **bar-based system, not a tick-based system.** It does not subscribe to a continuous tick-by-tick price stream. Instead it polls for snapshots and works entirely with OHLCV bar data.

| Data type | Granularity | Used for | Cache TTL |
|---|---|---|---|
| Daily OHLCV bars | 1-day bars, 504 bars (2 years) | ML feature computation, strategy selection | 1 hour |
| Intraday OHLCV bars | 5-minute bars, last 1–5 days | Multi-timeframe features (VWAP, intraday RSI, range compression) | 5 minutes |
| Real-time quote | Snapshot (bid/ask/last/volume) | Current price for position monitoring, order pricing | 15 seconds |
| Options chain | Full chain snapshot | Strike selection, Greeks, liquidity filters | Per scan cycle (not cached) |
| VIX | Snapshot | Position sizing scalar | Per scan cycle |

**When does new data arrive?**
- Every **5 minutes** (the scan cycle): fresh daily bars, fresh 5-min bars, and a new options chain are fetched for each symbol in the universe
- Every **30 seconds** (the position monitor): a fresh real-time quote snapshot is fetched per open position to check exit conditions
- The 15-second quote cache means the system never makes more than 4 quote requests per minute per symbol — important for staying within IBKR's API rate limits

**Why not tick data?** Options strategies have multi-day to multi-week holding periods and are selected based on daily market structure (IV rank, trend, momentum). Intraday noise at the tick level is irrelevant to these decisions. Tick data would add significant infrastructure complexity (streaming connection, tick storage, tick-based feature computation) with no benefit for the strategy types used here.

The 5-minute bar granularity is used only for *entry timing* (the multi-timeframe analyzer checks whether intraday momentum confirms the daily signal before committing), not for the core strategy selection logic.

### Storage

| Store | Technology | What's in it |
|---|---|---|
| `data/ait_state.db` | SQLite | Open trades, KV store, operational state — survives restarts |
| `data/ait_analytics.duckdb` | DuckDB | Closed trades, daily stats — fast analytical queries |
| `data/historical.db` | SQLite | OHLCV cache — avoids refetching the same data |
| `data/thompson_state.json` | JSON | Thompson sampler's per-strategy win/loss counts |
| `data/counterfactual_log.json` | JSON | Rejected trade outcomes |
| `models/ensemble.pkl` | Pickle | Current trained XGBoost + LightGBM models |
| `models/archive/` | Pickle files | Versioned model backups (for rollback) |

---

## 1.12 Monitoring & Observability

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

The `Watchdog` (`src/ait/monitoring/watchdog.py`) runs inside the bot process and monitors:
- **IBKR connection** — reconnects automatically on disconnect
- **Trading loop heartbeat** — detects if the main loop hangs
- **Memory usage** — alerts if the process exceeds 500 MB
- **API response times** — flags latency spikes
- **Error rates** — alerts if > 10 errors occur in a window

### Dashboard

The Streamlit dashboard at `http://localhost:8501` shows:
- Current open positions (entry price, current P&L, Greeks)
- Trade history with filters (by symbol, strategy, date range)
- Win rate and average P&L by strategy
- Portfolio-level P&L chart (cumulative)
- System health status (IBKR connection, last scan time, error count)

### Telegram Alerts

The bot sends Telegram messages for:
- Every trade opened (symbol, strategy, strikes, credit/debit, contracts)
- Every trade closed (P&L, reason for exit)
- Daily P&L summary (after market close)
- Circuit breaker triggered
- IBKR disconnection / reconnection
- System errors

---

## 1.13 Key Configuration Reference

All user-facing parameters live in `config.yaml`. Below are the most important ones:

### Trading Universe
```yaml
trading:
  universe: [SPY, QQQ, AAPL, MSFT, NVDA, AMZN, TSLA, GOOGL, META, JPM, GS, XLE, GLD, TLT]
  scan_interval_seconds: 300      # How often to scan for new trades (5 min)
  max_daily_trades: 5             # Max new trades per day
  trading_hours_only: true        # Only trade during RTH (9:30–4:00 ET)
```

### Risk Limits
```yaml
risk:
  max_daily_loss_pct: 0.02        # Circuit breaker: halt at 2% daily loss
  max_consecutive_losses: 3       # Pause after 3 losses in a row
  pause_minutes_after_losses: 30  # Cooldown duration
  min_confidence: 0.65            # Minimum ML confidence to take a trade
```

### Position Limits
```yaml
positions:
  max_open_positions: 5           # Never hold more than 5 positions at once
  max_position_pct: 0.05          # Each position ≤ 5% of account value
  max_portfolio_delta: 0.30       # Net portfolio delta ≤ ±30%
  max_portfolio_risk_pct: 0.02    # Max 2% account at risk in new trade
```

### Options Filters
```yaml
options:
  delta_range: [0.20, 0.50]       # Only short strikes with delta 0.20–0.50
  dte_range: [14, 45]             # Only expirations 14–45 days out
  min_open_interest: 100          # Minimum OI for liquidity
  min_volume: 50                  # Minimum daily volume for liquidity
  max_bid_ask_spread_pct: 0.10    # Reject if spread > 10% of mid price
```

### Exit Rules
```yaml
exit:
  trailing_stop_pct: 0.25         # Trail stop 25% below high-water mark
  breakeven_trigger_pct: 0.30     # Move stop to entry after +30% P&L
  initial_stop_loss_pct: 0.50     # Initial stop: 50% of premium paid
  partial_exit_levels: [0.50, 1.00]  # Take partial profits at +50%, +100%
```

### ML Settings
```yaml
ml:
  ensemble_weights:
    xgboost: 0.5
    lightgbm: 0.5
  retrain_interval_days: 7        # Retrain at least every 7 days
  lookback_days: 504              # 2 years of training data
  min_training_samples: 100       # Don't train with fewer samples
```

---

## 1.14 Technology Stack

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
| Dashboard | Streamlit + Plotly | ≥ 1.31 / ≥ 5.19 |
| Log viewer | Flask | — |
| Structured logging | structlog | ≥ 24.1 |
| Config models | pydantic | ≥ 2.6 |
| Notifications | Telegram Bot API | — |
| Testing | pytest + pytest-asyncio | — |

**Note on Apple Silicon / Intel Macs:** PyTorch 2.x requires Python 3.11 on Intel Macs. If you are on an Intel Mac, pin Python to 3.11 — newer versions will fail to install `torch`.

---

## 1.15 Key Design Decisions & Rationale

This section explains the *why* behind the most important architectural choices in AIT v2. Understanding these decisions helps you make informed changes without accidentally breaking something that was designed a specific way for a good reason.

---

### Decision 1: Separate Orchestrator and Bot Processes

**What:** `run_orchestrator.py` and `src/ait/main.py` run as two separate OS processes, not in the same process.

**Why:** Trading systems must be resilient to crashes. If the bot process throws an unhandled exception, Python would exit — taking every open position with it, unmonitored. By running the bot as a subprocess under a parent orchestrator, the orchestrator's `BotManager` detects the exit code and immediately restarts the bot. The scheduled jobs (retrain, reports, health checks) keep running in the orchestrator regardless of what the bot does.

**How to apply:** Never merge these two processes into one. If you add new background tasks, add them to the orchestrator's APScheduler, not to the bot's main loop.

---

### Decision 2: 13+ Ordered Risk Checks (Fail-Fast Pipeline)

**What:** Every trade signal passes through a fixed sequence of validation checks. The pipeline is ordered from cheapest to most expensive computation, and stops at the first failure.

**Why:** Options trading mistakes are expensive and often irreversible within a session. A misconfigured position can compound quickly, especially with defined-risk structures like spreads where max loss is locked at entry. The ordered pipeline means that cheap checks (circuit breaker, position count) run first so the system never wastes time computing Greeks or buying power for a signal that would have been blocked anyway. This also makes the system's behavior auditable — if a trade was rejected, the log tells you exactly which check failed.

**How to apply:** When adding new risk rules, insert them in order of computational cost (cheapest first). Never bypass the pipeline even for "obvious" high-confidence signals — the pipeline exists precisely for those cases.

---

### Decision 3: Thompson Sampling for Strategy Selection

**What:** Strategy selection uses a multi-armed bandit (Thompson sampling) rather than a fixed rotation or a rule-based priority list.

**Why:** Market regimes change. A strategy that works well for 3 months may underperform the next 3. A fixed priority list would keep using the same strategy regardless of recent performance. A rule-based system requires manually defining when to switch. Thompson sampling solves this automatically: each strategy maintains a Beta distribution over its win probability, updated after every closed trade. Strategies that are winning get sampled more often; strategies that are losing get sampled less. Crucially, it never fully stops trying any strategy — it always explores, so it can detect when a previously poor strategy has become profitable again.

**How to apply:** The Thompson state persists in `data/thompson_state.json`. Do not delete this file between sessions — it contains the accumulated win/loss history that makes the sampler meaningful. If you add a new strategy, it starts with a uniform prior and will be explored automatically.

---

### Decision 4: Self-Learning with Bounded Adaptations

**What:** The nightly learning cycle adapts strategy weights, but with hard bounds: no strategy multiplier goes below 0.3× or above 1.5×, and iron condors have a floor of 0.3× (never fully disabled).

**Why:** Unconstrained adaptation is dangerous in trading. If the learning system could set a strategy multiplier to 0×, a bad run of losses (which can be random variance, not a true signal) could permanently disable a profitable strategy. The 0.3× floor ensures that every strategy stays in rotation at a meaningful level, so the Thompson sampler still has enough data to evaluate it. The 1.5× cap prevents over-leveraging a strategy that happened to have a lucky streak.

**How to apply:** If you adjust the floors/ceilings in the learning code, be conservative. Widening the bounds gives the system more power but also more rope to hang itself during unusual market conditions.

---

### Decision 5: Per-Symbol ML Models with Universal Fallback

**What:** Each symbol in the universe gets its own XGBoost + LightGBM model pair. If a symbol lacks sufficient training data (fewer than 100 samples), it falls back to a universal model trained on all symbols.

**Why:** Different stocks behave differently. SPY (a broad index ETF) has very different volatility patterns, volume profiles, and momentum characteristics than NVDA (a high-beta semiconductor stock). A single universal model is forced to average across these differences, reducing predictive power for any individual stock. Per-symbol models can capture stock-specific patterns. The universal fallback prevents the system from refusing to trade new symbols entirely — it degrades gracefully rather than failing.

**How to apply:** When adding a new symbol to the universe, the system will initially use the universal model. After the symbol accumulates 100+ days of history and the daily retrain runs, it gets its own model automatically. Do not force-train a per-symbol model on fewer than 100 samples — the model will overfit and perform worse than the universal baseline.

---

### Decision 6: Purge Gap in Walk-Forward Cross-Validation

**What:** Between the training window and the test window in the walk-forward backtester, there is a mandatory 5-day gap (`gap_days=5`) during which no data is used.

**Why:** This is called a "purge gap" and it prevents **look-ahead bias** (also called data leakage). The ML labels are based on 5-day forward returns. If training data ends on day T and the test window starts on day T+1, the label for day T-4 was computed using data from T through T+1 — which overlaps with the test window. This contaminates the test, making the model appear more accurate than it really is. The 5-day purge matches the label horizon, ensuring no training label was computed using test-period prices.

**How to apply:** If you change the label horizon (currently 5 days), update `gap_days` to match. Setting `gap_days=0` will silently cause inflated backtest performance that won't materialize in live trading.

---

### Decision 7: Counterfactual Logging of Rejected Trades

**What:** When the risk manager rejects a trade signal, the system doesn't just discard it. It logs the signal parameters and then tracks what would have happened if the trade had been taken.

**Why:** Risk limits can be either too tight (blocking profitable trades) or too loose (allowing dangerous ones). Without counterfactual data, you only see the trades that were taken. You cannot tell whether a tightened confidence floor is protecting you from bad trades or just reducing your trade count with no safety benefit. The counterfactual log lets the learning system and human operators quantify the opportunity cost of each risk rule, making it possible to calibrate limits with evidence rather than intuition.

**How to apply:** Review `data/counterfactual_log.json` periodically (especially after tightening a risk limit) to assess whether the tightening is eliminating genuinely bad trades or needlessly blocking good ones.

---

### Decision 8: Three-Source Data Fallback Chain

**What:** Market data is fetched from IBKR first, then Polygon.io, then Yahoo Finance, with each source tried in order on failure.

**Why:** Each source has different failure modes. IBKR is the most accurate (real-time, directly from the exchange) but requires a live connection — if the gateway disconnects, all IBKR data fails. Polygon has higher reliability for historical data but has rate limits on the free tier (5 calls/minute) and requires an API key. Yahoo Finance requires no key and rarely goes down, but its data can have adjustments and gaps. The fallback chain means a transient IBKR disconnect doesn't halt all data fetching — it just degrades to the next source.

**How to apply:** The system cannot trade without an IBKR connection (options chains only come from IBKR). The fallback chain is for OHLCV data only. If IBKR disconnects, the bot will pause new trade entries but can continue monitoring open positions using the last known prices.

---

## 1.16 Backtesting

AIT v2 includes a walk-forward backtesting system that lets you validate strategies on historical data before running them live. This section explains what backtesting means in this context, how to run it, and how to interpret the results.

### What is Walk-Forward Backtesting?

A standard ("simple") backtest trains a model on all available data and then tests it on the same data. This is meaningless — of course a model does well on data it was trained on. It tells you nothing about future performance.

**Walk-forward backtesting** simulates what would actually happen in live trading:

```
[──── Train (1 year) ────][gap][── Test (3 months) ──]
                                    ↓ slide forward
                    [──── Train (1 year) ────][gap][── Test (3 months) ──]
                                                        ↓ slide forward
                                [──── Train (1 year) ────][gap][── Test (3 months) ──]
```

- Train the ML model on year 1
- Test it on the next 3 months (data the model has never seen)
- Slide the window forward 1 month
- Repeat

This produces multiple independent test results. Each test period is genuinely out-of-sample, just like live trading. The aggregate of all test windows gives you a realistic estimate of live performance.

### Running the Backtester

```bash
# Quick run — default 10 symbols, default settings
python run_backtest.py

# Custom symbols and capital
python run_backtest.py --symbols SPY QQQ AAPL --capital 25000

# Specific strategies only
python run_backtest.py --strategies iron_condor bull_call_spread

# Tune the walk-forward windows
python run_backtest.py \
  --train-days 365 \      # 1 year training window
  --test-days 63 \        # ~3 months test window
  --step-days 21 \        # Slide 1 month at a time
  --gap-days 5            # 5-day purge gap (matches label horizon)

# Compare fixed stops vs trailing stops
python run_backtest.py --compare-exits

# Adjust iron condor IV requirement
python run_backtest.py --iv-floor 20    # Lower = more condor trades
```

**Full CLI reference:**

| Flag | Default | Description |
|---|---|---|
| `--symbols` | 10 major symbols | Symbols to backtest |
| `--strategies` | 4 strategies | Strategies to test |
| `--train-days` | 365 | Training window in calendar days |
| `--test-days` | 63 | Test window in calendar days (~3 months) |
| `--step-days` | 21 | How far to slide each window (~1 month) |
| `--gap-days` | 5 | Purge gap between train and test |
| `--capital` | $50,000 | Initial capital |
| `--min-confidence` | 0.65 | ML confidence threshold |
| `--range-confidence` | 0.55 | Iron condor range-model threshold |
| `--iv-floor` | 30.0 | Min IV rank for iron condors |
| `--trailing-stop` | On | Enable trailing stops (default: on) |
| `--compare-exits` | Off | Side-by-side fixed vs trailing stop comparison |

### How the Backtester Works Internally

**Data:** Fetched from Yahoo Finance (5 years of daily OHLCV per symbol). No IBKR connection required — you can run the backtester completely offline from live trading.

**Options pricing:** Real options chains are not available historically, so the backtester prices options using Black-Scholes with implied volatility estimated from realized volatility (RV × 1.15 to simulate the typical IV premium). This is an approximation — real IV is higher during stress and lower during quiet periods.

**Entry logic (per trading day):**
1. Compute features from historical OHLCV
2. Run the ML model (trained on the training window) to get direction + confidence
3. If confidence ≥ threshold, select a strategy
4. Price the options using Black-Scholes; calculate position size (5% of capital per trade)
5. Deduct commissions ($0.65/contract/leg) and slippage (3% on multi-leg fills)
6. Enter the position

**Exit logic (checked every day):**
- Take profit at 50% of credit (credit strategies) or 100% of debit paid (debit strategies)
- Stop loss at 35% loss
- Max hold: 21 days (to avoid deep theta decay / gamma risk)
- Trailing stop: after P&L reaches +30%, trail 25% below the high-water mark

**Self-learning between windows:** The backtester includes a `BacktestLearner` that adapts strategy weights between windows based on the previous window's performance. This simulates how the live system's nightly learning cycle works. A strategy that had a bad test window gets a reduced multiplier in the next window.

### Interpreting the Output

The backtester prints a detailed report. Here is how to read each section:

```
============================================================
  WALK-FORWARD BACKTEST RESULTS
============================================================
  Windows:           12          ← Number of train/test cycles run
  Total Trades:      84          ← Trades across all test windows
  Total Return:      +47.3%      ← Chained return across all windows
  Cash-Drag Adj Ret: +49.1%      ← Return + idle-cash T-bill yield
------------------------------------------------------------
  RISK-ADJUSTED
  Sharpe Ratio:      1.51        ← Return / total volatility (>1 is good)
  Sortino Ratio:     2.18        ← Return / downside volatility only (better for options)
  Max Drawdown:      -12.4%      ← Largest peak-to-trough loss
------------------------------------------------------------
  TRADE QUALITY
  Win Rate:          64%         ← % of trades that closed positive
  Avg Win:           $312        ← Average profit per winning trade
  Avg Loss:          $187        ← Average loss per losing trade
  Win/Loss Ratio:    1.67        ← Avg winner 67% bigger than avg loser (>1 is good)
  Expectancy/Trade:  $132        ← Expected $ per trade (positive = edge exists)
  Profit Factor:     2.1         ← Gross wins / gross losses (>1 profitable, >2 strong)
------------------------------------------------------------
  CAPITAL EFFICIENCY
  Utilization:       18%         ← Only 18% of capital was deployed on avg
  RAROC:             263%        ← Return on capital that was actually at risk
------------------------------------------------------------
  CONSISTENCY
  Profitable Windows: 83%        ← 10 of 12 windows made money
  Avg Window Return:  +4.1%      ← Average per 3-month window
```

**Key metrics explained for options trading specifically:**

| Metric | What it means | Good threshold |
|---|---|---|
| **Win Rate** | % of trades profitable | > 55% for debit; > 65% for credit |
| **Profit Factor** | Gross wins ÷ gross losses | > 1.5 is solid; > 2.0 is strong |
| **Sortino Ratio** | Like Sharpe but ignores upside volatility — appropriate for options selling's skewed return profile | > 1.5 |
| **RAROC** | Return on capital actually deployed (not idle) — tells you the real edge per dollar at risk | > 50% annually |
| **Utilization** | How often the bot is deployed. Options strategies sit idle between trades. Low utilization + high RAROC = concentrated edge | Typical: 10–30% |
| **Consistency** | % of test windows that were profitable. High consistency = strategy works across different market conditions | > 70% |
| **Cash-Drag Adjusted Return** | Adds T-bill yield on idle cash — a more honest comparison to buy-and-hold | — |

### Buy-and-Hold Benchmark Comparison

At the end of every backtest run, the system computes what each symbol would have returned if you'd simply bought and held over the same period:

```
  SPY          +31.2%
  QQQ          +44.7%
  AAPL         +28.1%
  PORTFOLIO    +34.7%     ← Equal-weighted buy-and-hold average

  Strategy Return:   +47.3%
  Buy & Hold Return: +34.7%
  Alpha:             +12.6%   ← Strategy outperformance
```

**Important caveat:** This alpha comparison is not apples-to-apples. Options strategies have very different risk profiles than buy-and-hold equities. Options strategies have capped upside (for credit trades), use leverage, and carry assignment/pin risk that buy-and-hold does not. The alpha figure shows outperformance but not the full risk-adjusted picture — use Sharpe and Sortino for that.

### Per-Window Detail Table

```
  WINDOW DETAILS (12 windows):
  #    Test Period                Trades    Return    Win%
  ---  -------------------------  ------  --------  ------
    1  2022-04-15 to 2022-07-14       6    +3.1%    66.7%
    2  2022-05-16 to 2022-08-14       8    -2.4%    37.5%
    3  2022-06-15 to 2022-09-13       7    +6.8%    71.4%
    ...
```

A window with negative returns is not automatically a problem — it may simply correspond to a difficult market period (e.g., a crash). What matters is that the majority of windows are profitable and the drawdown during bad windows is contained.

### Common Backtesting Mistakes

**1. Over-tuning parameters on backtest results.** If you run many backtests and keep adjusting `--min-confidence`, `--iv-floor`, etc. until the numbers look good, you are overfitting to historical data. The parameters should be set based on theory (e.g., "IV rank > 30% means options are expensive enough to sell") and then left fixed.

**2. Ignoring the purge gap.** Setting `--gap-days 0` will inflate results. Always keep it at ≥ 5 to match the label horizon.

**3. Trusting a single symbol backtest.** A strategy that works brilliantly on SPY alone may fail on NVDA. Always backtest across the full universe.

**4. Confusing backtest Sharpe with live Sharpe.** Backtests use Black-Scholes for options pricing, which is smooth and continuous. Real markets have bid-ask spreads, liquidity gaps, and early assignment. Live performance will typically have more variance than the backtest suggests.

---

## 1.17 Parameter Optimization

**Short answer: automated parameter optimization does not currently exist.** All strategy parameters are set manually in `config.yaml` and validated by running the backtester with different values. This section explains what can be tuned, how to do it, and what proper automated optimization would look like.

---

### What Parameters Can Be Tuned

There are two categories of parameters: those exposed via the backtester CLI (fast iteration), and those that require editing `config.yaml` (require a restart).

**Backtester CLI flags** (no restart required, test immediately):

| Parameter | CLI flag | Default | What it controls |
|---|---|---|---|
| Min ML confidence | `--min-confidence` | 0.65 | Trade filter threshold |
| Iron condor IV floor | `--iv-floor` | 30.0 | Min IV rank for condors |
| Range model confidence | `--range-confidence` | 0.55 | Min P(stays in range) for condors |
| Trailing vs fixed stops | `--trailing-stop` / `--compare-exits` | On | Exit style comparison |
| Train window size | `--train-days` | 365 | ML training lookback |
| Test window size | `--test-days` | 63 | Walk-forward test period |

**`config.yaml` parameters** (require bot restart, affect live trading):

| Parameter | Location in config | What it controls |
|---|---|---|
| Delta range for short strikes | `options.delta_range` | How far OTM the short legs are placed |
| DTE range | `options.dte_range` | Expiration window (14–45 days default) |
| Max bid-ask spread | `options.max_bid_ask_spread_pct` | Liquidity filter for option selection |
| Take-profit target | `exit` section | When to close winning positions |
| Stop-loss level | `exit.initial_stop_loss_pct` | When to close losing positions |
| Trailing stop % | `exit.trailing_stop_pct` | How tightly the trailing stop follows the high-water mark |
| Max open positions | `positions.max_open_positions` | Portfolio concentration limit |
| Max daily loss | `risk.max_daily_loss_pct` | Circuit breaker level |
| Scan interval | `trading.scan_interval_seconds` | How often to look for new trades |

---

### Current Manual Tuning Workflow

The recommended process for changing a parameter:

```
1. Identify the parameter you want to test (e.g., lower --iv-floor from 30 to 20)

2. Run a baseline backtest to record current performance:
   python run_backtest.py --iv-floor 30

3. Run the backtest with the candidate value:
   python run_backtest.py --iv-floor 20

4. Compare the key metrics:
   - Did total return improve?
   - Did Sharpe / Sortino improve? (risk-adjusted is more important than raw return)
   - Did consistency (% profitable windows) hold up?
   - Did max drawdown worsen significantly?

5. If the new parameter looks better across multiple metrics:
   - Update config.yaml
   - Restart the bot
```

For exit-rule comparisons specifically, `--compare-exits` runs two full backtests (fixed vs trailing stops) and prints a side-by-side delta table, which is the most efficient way to test exit parameters.

---

### What Automated Optimization Would Require

The system does not include a grid search, Bayesian optimization, or any other automated parameter sweep. Building one would require:

1. **A parameter search loop** that calls `WalkForwardBacktester.run()` for each candidate parameter combination
2. **A stable objective function** — typically the Sharpe ratio or Sortino ratio (not raw return, which can be maximized by taking excessive risk)
3. **An outer validation set** (data held out from all parameter search iterations) to detect whether the chosen parameters overfit to the search period
4. **A runtime budget** — a full walk-forward backtest across 10 symbols takes 30–120 seconds. A grid search over 5 parameters with 5 values each (3,125 combinations) would take 25–100 hours

**Why it hasn't been built:** The system has relatively few parameters compared to a pure algorithmic strategy. The most important levers (ML confidence floor, IV rank floor, exit targets) have clear theoretical interpretations. Setting them based on theory and then confirming with a backtest is less prone to overfitting than a full automated search.

**If you want to add it:** The entry point would be `src/ait/backtesting/walkforward.py` — `WalkForwardBacktester.run()` is already async and returns a structured `WalkForwardResult` object that can be compared programmatically. A simple grid search wrapper could call it in a loop and store results in a DataFrame.

---

### ML Hyperparameters

ML model hyperparameters (XGBoost `n_estimators`, `max_depth`, `learning_rate`, etc.) are a separate concern from strategy parameters. These are currently set to reasonable defaults inside `src/ait/ml/ensemble.py` and are **not exposed in `config.yaml`**.

The `DirectionPredictor.train()` method uses scikit-learn's cross-validation during each retraining cycle to select among a small set of candidate configurations, but this is not a full hyperparameter search — it is a lightweight sanity check. Proper ML hyperparameter tuning (e.g., with Optuna or scikit-learn's `GridSearchCV`) would need to be added to the training pipeline in `src/ait/ml/ensemble.py`.

---

*This document is a living reference. Additional sections (Setup, FAQ, Troubleshooting) will be added below as questions are answered.*

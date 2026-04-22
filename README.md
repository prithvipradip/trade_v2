# AIT v2 — Autonomous Intelligent Trading Bot

An autonomous options trading bot that trades iron condors, credit spreads, and straddles using ML predictions, sentiment analysis, and self-learning. Connects to Interactive Brokers via IB Gateway for live/paper execution.

> ⚠️ **Disclaimer:** This is educational/research software. Trading options involves substantial risk. Test thoroughly on paper accounts before considering live trading. The authors accept no liability for financial losses.

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [Components Explained](#components-explained)
- [Trading Strategies](#trading-strategies)
- [Risk Management](#risk-management)
- [Monitoring & Dashboards](#monitoring--dashboards)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Development](#development)

---

## What It Does

- Scans a configurable universe of stocks/ETFs every 5 minutes during market hours
- Runs ML models (XGBoost + LightGBM ensemble) to predict direction
- Combines ML with sentiment (news, fear/greed, FinBERT) and options flow
- Selects strategy (iron condor, straddle, short strangle, covered call, cash-secured put, spreads) based on IV rank and market regime
- Places orders via IBKR with smart pricing and risk validation
- Monitors positions every 30 seconds, closes on stops/targets/expiry/macro events
- Retrains models daily, runs walk-forward backtests weekly
- Self-learns from past trades to adapt strategy selection

---

## Architecture Overview

```
run_orchestrator.py (master process)
├── BotManager                    ← Auto-restarts bot on crash
│     └── python -m ait.main
│           ├── TradingOrchestrator (the brain)
│           │   ├── MarketScheduler    (pre-market / open / post / off-hours)
│           │   ├── ModelTrainer       (daily retraining)
│           │   ├── SentimentEngine    (news + fear/greed + FinBERT)
│           │   ├── StrategySelector   (picks strategies per IV rank)
│           │   ├── RiskManager        (validates every trade)
│           │   ├── PortfolioManager   (exit logic, every 30s)
│           │   ├── TradeExecutor      (places orders)
│           │   └── LearningEngine     (post-market self-learning)
│           └── IBKRClient (ib_insync → IB Gateway port 4002)
├── WebServiceManager             ← Dashboard + Log viewer
│     ├── Streamlit dashboard (:8501)
│     └── Flask log viewer       (:8502)
└── APScheduler
    ├── health_check (every 2 min)
    ├── daily_retrain (Mon-Fri 7:30 AM ET)
    ├── daily_report  (Mon-Fri 4:30 PM ET)
    ├── weekly_backtest (Sunday 8 PM ET)
    └── monthly_cleanup (1st of month)
```

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|---|---|---|
| Python | 3.11 – 3.13* | Bot runtime |
| Git | Any | Clone the repo |
| Interactive Brokers Gateway | 1044+ | Order execution |
| IBC (IB Controller) | 3.20+ | Gateway auto-start (optional) |

> **\*Intel Mac users: use Python 3.11** — PyTorch dropped Intel Mac wheels from torch 2.3+, and Python 3.12/3.13 Intel Mac users can't install torch. On Python 3.11, torch 2.2.x works. Apple Silicon Macs and Windows/Linux can use any supported Python version.
>
> ```bash
> # Intel Mac install:
> brew install python@3.11
> python3.11 -m venv venv && source venv/bin/activate
> pip install -e .
> ```

### Required Accounts & Keys

| Service | Required? | Purpose | Cost |
|---|---|---|---|
| **Interactive Brokers** | **Yes** | Execute trades | Free to open; paper trading free |
| **OPRA Market Data** | **Yes** | Options quotes | $1.50/mo (waived at $20 commissions) |
| **Polygon.io** | Optional | Historical data | Free tier: 5 calls/min |
| **Finnhub** | Optional | News sentiment | Free tier: 60 calls/min |
| **Telegram Bot** | Optional | Trade alerts | Free |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/trade_v2.git
cd trade_v2
```

### 2. Create Python Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

This installs all packages from `pyproject.toml`:
- **Trading**: `ib_insync`, `yfinance`, `polygon-api-client`
- **ML**: `xgboost`, `lightgbm`, `scikit-learn`, `scipy`
- **Sentiment**: `transformers`, `torch`, `feedparser`, `finnhub-python`
- **Options**: `py-vollib` (Greeks/IV calculations)
- **Dashboard**: `streamlit`, `plotly`, `flask`
- **Infrastructure**: `apscheduler`, `duckdb`, `structlog`, `pydantic`

### 4. Install Interactive Brokers Gateway

1. Download IB Gateway from [interactivebrokers.com/en/trading/ibgateway-stable.php](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php)
2. Install to default location (`C:\Jts\ibgateway\` on Windows)
3. Open Gateway, login to paper account, set API port to **4002**
4. Go to **Configure → Settings → API → Settings**:
   - ✅ Check "Enable ActiveX and Socket Clients"
   - ❌ **UNCHECK "Read-Only API"** (very important — bot can't trade otherwise)
   - Set "Socket port" to **4002**

### 5. Install IBC (Optional — for auto-start)

1. Download from [github.com/IbcAlpha/IBC/releases](https://github.com/IbcAlpha/IBC/releases)
2. Extract to `C:\IBC` (Windows) or `/opt/ibc` (Linux)
3. Edit `C:\IBC\config.ini`:
   ```ini
   IbLoginId=your_paper_username
   IbPassword=your_paper_password
   TradingMode=paper
   OverrideTwsApiPort=4002
   ReadOnlyApi=no
   AcceptNonBrokerageAccountWarning=yes
   ```

---

## Configuration

### Step 1: Copy and Edit `.env`

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```ini
# IBKR TWS/Gateway connection
IBKR_HOST=127.0.0.1
IBKR_PORT=4002                          # 4002=paper gateway, 4001=live gateway
IBKR_CLIENT_ID=1                         # Unique ID for this bot session
IBKR_ACCOUNT=DU1234567                   # Your paper account number

# Polygon.io API key (free at polygon.io)
POLYGON_API_KEY=your_polygon_key_here

# Finnhub API key (free at finnhub.io)
FINNHUB_API_KEY=your_finnhub_key_here

# Telegram alerts (optional)
TELEGRAM_BOT_TOKEN=123456:ABC-your-token
TELEGRAM_CHAT_ID=your_chat_id

# Mode: paper or live
TRADING_MODE=paper
```

#### Getting API Keys

**Polygon.io:**
1. Sign up at [polygon.io](https://polygon.io)
2. Go to Dashboard → API Keys
3. Copy your free tier key

**Finnhub:**
1. Sign up at [finnhub.io](https://finnhub.io)
2. Dashboard → API key

**Telegram Bot:**
1. In Telegram, message [@BotFather](https://t.me/BotFather)
2. Send `/newbot`, follow prompts, copy the bot token
3. Start a chat with your new bot, send any message
4. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
5. Find `"chat":{"id":...}` — that's your `TELEGRAM_CHAT_ID`

### Step 2: Edit `config.yaml`

The main configuration file. Key sections:

#### Trading Universe
```yaml
trading:
  mode: paper                             # "paper" or "live"
  universe: [SPY, QQQ, IWM, DIA, AAPL, MSFT, NVDA, TSLA, AMD, AMZN, META, GOOGL, SOFI, PLTR]
  scan_interval_seconds: 300              # Full scan every 5 min
  max_daily_trades: 5                     # Max new positions per day
  trading_hours_only: true                # RTH only (9:30-16:00 ET)
```

#### Risk Limits
```yaml
risk:
  max_daily_loss_pct: 0.02                # 2% daily loss → circuit breaker
  max_consecutive_losses: 3               # Pause trading after 3 losses
  min_confidence: 0.65                    # Min ML confidence to trade

positions:
  max_open_positions: 5
  max_position_pct: 0.05                  # 5% of account per position
  max_portfolio_delta: 0.30               # Net delta cap
```

#### Strategies
```yaml
options:
  delta_range: [0.20, 0.50]
  dte_range: [14, 45]                     # Days to expiry target
  min_open_interest: 100
  min_volume: 50
  max_bid_ask_spread_pct: 0.10
  strategies:
    - iron_condor      # Neutral, defined risk
    - short_strangle   # Neutral, undefined risk
    - long_straddle    # Neutral, directional breakout
    - cash_secured_put # Bullish
    - covered_call     # Slightly bullish
    - bull_call_spread
    - bear_put_spread
    - long_call
    - long_put
```

#### ML Models
```yaml
ml:
  ensemble_weights:
    xgboost: 0.5
    lightgbm: 0.5
  retrain_interval_days: 7                # Full retrain weekly
  lookback_days: 504                      # 2 years training data
  min_training_samples: 100
```

#### Exit Logic
```yaml
exit:
  trailing_stop_pct: 0.25                 # Trail 25% below HWM
  breakeven_trigger_pct: 0.30             # Move stop to 0 at +30% P&L
  initial_stop_loss_pct: 0.50             # Stop loss before breakeven
  partial_exit_levels:
    - pnl_pct: 0.50                       # At +50% P&L, close 33%
      close_pct: 0.33
    - pnl_pct: 1.00                       # At +100%, close another 33%
      close_pct: 0.33
  time_decay_scaling: true                # Lower TP target as DTE shrinks
  volatility_adjusted_stops: true
```

---

## Running the Bot

### Master Orchestrator (Recommended)

Starts the full system: bot + dashboard + log viewer + scheduler:

```bash
python run_orchestrator.py
```

This launches:
- **Trading bot** (the main process)
- **Streamlit dashboard** on [http://localhost:8501](http://localhost:8501)
- **Web log viewer** on [http://localhost:8502](http://localhost:8502)
- Health checks every 2 min with auto-restart

### Individual Commands

```bash
# Check bot status
python run_orchestrator.py --status

# Force ML retrain
python run_orchestrator.py --retrain

# Generate daily report
python run_orchestrator.py --report

# Run walk-forward backtest
python run_backtest.py --symbols SPY QQQ AAPL --capital 50000

# View live logs (color-coded)
python tail_logs.py

# Just the dashboard (if already running)
streamlit run src/ait/dashboard/app.py
```

### Windows Launcher

Double-click `start_bot.bat` — equivalent to `python run_orchestrator.py`.

---

## Components Explained

### Orchestrator ([src/ait/bot/orchestrator.py](src/ait/bot/orchestrator.py))
The brain. Runs the market-phase state machine: pre-market (retrain, reconcile), market open (scan/trade/monitor), post-market (learning cycle), off-hours (sleep).

### Strategy Selector ([src/ait/strategies/selector.py](src/ait/strategies/selector.py))
Picks strategies based on:
- **Direction** (bullish/bearish/neutral) from ML
- **IV rank** (high IV → sell premium, low IV → buy premium)
- **Market regime** (trending vs range-bound)
- **Thompson sampling** (multi-armed bandit from historical wins)

### ML Ensemble ([src/ait/ml/ensemble.py](src/ait/ml/ensemble.py))
- **XGBoost + LightGBM** (equal weight)
- **60 features**: RSI, MACD, Bollinger, ATR, volume, volatility, IV rank, multi-timeframe, VIX level/change, SPY relative strength, correlation
- **3-class labels**: bullish (>+1% in 5d), bearish (<-1% in 5d), neutral
- **Walk-forward CV** with purge gap
- **Class balancing** (sample weights) to reduce neutral bias
- **Per-symbol models** (plus universal fallback)

### Risk Manager ([src/ait/risk/manager.py](src/ait/risk/manager.py))
Pre-trade validation pipeline:
1. Circuit breaker check
2. Confidence threshold
3. Weekend gap risk (Friday 2:30 PM+ requires 90% conf for short strangles)
4. Daily trade limit
5. Max position count
6. Duplicate position check
7. Correlation guard (blocks stacking correlated positions)
8. Buying power check
9. Per-position max risk (3% of account)
10. Symbol concentration (20% of account max per symbol)
11. Portfolio delta limit
12. Daily loss check
13. Position sizing (volatility-adjusted, drawdown-throttled, VIX-regime-aware)

### Portfolio Manager ([src/ait/execution/portfolio.py](src/ait/execution/portfolio.py))
Checks every 30 seconds for exits:
- **Trailing stop** (trails 25% below HWM once +30% reached)
- **Dynamic take profit** (100% → 25% target as DTE shrinks)
- **Time decay exit** (DTE ≤ 5)
- **Assignment risk** (ITM shorts on expiry day force-close)
- **Delta breach** (|Δ| > 0.50 for neutral strategies)
- **IV crush pre-close** (0-2 days before earnings if winning)
- **Macro event flatten** (1 day before FOMC/CPI/NFP)
- **Partial exits** at +50% / +100% milestones
- **PDT-aware** (blocks day-trade exits if flagged)

### Executor ([src/ait/execution/executor.py](src/ait/execution/executor.py))
Order placement with:
- **Aggressive combo pricing** (mid ± $0.05 for fills)
- **Wide spread rejection** (>15% bid-ask → skip)
- **Stale order cancellation** (90s timeout)
- **Partial fill handling** (cancel remainder after 30s)
- **Slippage tracking** (actual vs expected fill)
- **Commission-adjusted P&L** (~$0.65/contract/side)

### Sentiment Engine ([src/ait/sentiment/engine.py](src/ait/sentiment/engine.py))
Composite sentiment score per symbol (-1.0 to +1.0):
- **News** (weight 35%) from Finnhub headlines
- **Fear & Greed** (weight 40%) from VIX + breadth + momentum
- **FinBERT** (weight 25%) — local NLP model on headlines

### Learning Engine ([src/ait/learning/engine.py](src/ait/learning/engine.py))
Post-market analysis:
- Analyzes last 30 days of trades
- Adapts strategy multipliers (boost winners, reduce losers)
- Never fully disables iron condors (floor at 0.3x)
- Blocks symbols with consistent losses
- Updates confidence floor/ceiling within bounds

---

## Trading Strategies

| Strategy | Market View | Risk Profile | Best When |
|---|---|---|---|
| **Iron Condor** | Neutral | Defined (both sides) | IV rank > 50, range-bound |
| **Short Strangle** | Neutral | Undefined | IV rank > 70, very high premium |
| **Long Straddle** | Volatility breakout | Defined (debit) | Low IV, expected big move |
| **Cash-Secured Put** | Bullish | Undefined (buy stock if assigned) | IV rank > 40, want to own |
| **Covered Call** | Slightly bullish | Capped upside | Own stock, want income |
| **Bull Call Spread** | Bullish | Defined (debit) | Bullish with cost control |
| **Bear Put Spread** | Bearish | Defined (debit) | Bearish with cost control |

The bot auto-selects based on:
- ML direction prediction
- Current IV rank
- Account size (capital tier)
- Self-learning multipliers

---

## Risk Management

Every trade passes through 13+ risk checks. Key protections:

### Portfolio-Level
- **Circuit breaker**: 2% daily loss → halt trading
- **Max open positions**: 5 (configurable)
- **Portfolio delta cap**: 30% of account in aggregate delta

### Position-Level
- **Max risk per trade**: 3% of account
- **Symbol concentration**: 20% of account max per symbol
- **Correlation haircut**: 30% size reduction per correlated open position

### Adaptive Sizing
- **Drawdown throttle**: 50% size after 3 losing days, 25% after 5
- **VIX regime**: 75% at VIX≥25, 50% at VIX≥30
- **Confidence scaling**: Low conf = smaller size

### Event Protection
- **Earnings filter**: Skips trades 2 days before to 1 day after earnings
- **FOMC/CPI/NFP**: Auto-closes short premium 1 day before
- **Assignment protection**: Force-closes ITM shorts on expiry day

---

## Monitoring & Dashboards

### Streamlit Dashboard ([http://localhost:8501](http://localhost:8501))
Tabs:
- **Portfolio Overview**: Account value, open positions, today's P&L
- **Trade History**: All trades with filters
- **Analytics**: Win rate, strategy performance, Sharpe
- **Self-Learning**: Active adaptations, counterfactual trades
- **System Health**: Watchdog, latency, memory

### Web Log Viewer ([http://localhost:8502](http://localhost:8502))
Auto-refreshing colored log stream. Filters: all / trades / predictions / signals / errors.

### Terminal Log Viewer
```bash
python tail_logs.py              # Color-coded live tail
python tail_logs.py --trades     # Only trade events
python tail_logs.py --last 100   # Last 100 lines
```

### Telegram Alerts
If configured, sends:
- Trade placements and exits
- Circuit breaker trips
- Daily P&L summaries
- Critical errors

---

## Troubleshooting

### "Read-Only API" Error (Error 321)
- Open IB Gateway → **Configure → Settings → API → Settings**
- **Uncheck "Read-Only API"**
- Restart Gateway

### "Competing live session" (Error 10197)
- Paper accounts with live subscriptions conflict
- Fix: In [src/ait/broker/ibkr_client.py](src/ait/broker/ibkr_client.py), `reqMarketDataType(4)` (delayed-frozen)
- Or add live data subscription to paper account in IBKR Client Portal

### Combo orders not filling
- Orders sitting at mid-price don't fill on multi-leg spreads
- Bot uses `mid - $0.05` aggressive pricing by default
- Adjust in [src/ait/execution/executor.py](src/ait/execution/executor.py) `aggressive_offset`

### "Delayed market data" warnings (10167)
- Your IBKR subscription doesn't cover those specific exchanges
- OPRA covers most options; NASDAQ stocks may need NASDAQ Basic subscription
- Bot works with delayed data but fills may be suboptimal

### Bot stuck on training
- First run trains models for all 14 symbols (~5-10 min)
- Subsequent runs load from `models/ensemble.pkl`
- Force retrain: `python run_orchestrator.py --retrain`

### Telegram alerts not sending
- Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`
- Test: `curl -X POST "https://api.telegram.org/bot<TOKEN>/sendMessage" -d "chat_id=<CHAT_ID>&text=test"`
- If parse errors: bot uses plain text (no Markdown) since recent fix

### "No module named 'ait'"
- Run `pip install -e .` from project root
- Activate the venv first: `venv\Scripts\activate` (Windows)

### "Could not find a version that satisfies the requirement torch>=2.2"
PyTorch has no pre-built wheel for your platform. Common causes:

1. **Intel Mac + Python 3.12/3.13** — PyTorch dropped Intel Mac wheels from torch 2.3+.
   **Fix:** Use Python 3.11. `brew install python@3.11`, then `python3.11 -m venv venv`.
2. **Python too new** (3.14+ alpha) — use 3.11, 3.12, or 3.13.
3. **Python 32-bit install** — torch requires 64-bit. Check: `python -c "import struct; print(struct.calcsize('P') * 8)"` (should print 64).
4. **Pip is outdated** — upgrade: `python -m pip install --upgrade pip`
5. **ARM Linux/other exotic arch** — install torch separately first from [pytorch.org](https://pytorch.org/get-started/locally/), then `pip install -e .`
6. **Corporate proxy blocking PyPI** — try with `--index-url https://pypi.org/simple/`

**Workaround if torch install is blocked:**
FinBERT (sentiment) is the only thing needing torch. You can disable sentiment in `config.yaml`:
```yaml
sentiment:
  enabled: false
```
Then edit `pyproject.toml` to remove `torch` and `transformers` from dependencies, and reinstall.

### Bot won't start
- Check Gateway is running: `netstat -ano | grep 4002`
- Check logs: `tail logs/orchestrator.log`
- Kill stale Python: `powershell Stop-Process -Name python -Force`

---

## Project Structure

```
trade_v2/
├── README.md                   ← You are here
├── CONTEXT.md                  ← Architecture/history notes
├── config.yaml                 ← Main configuration
├── .env                        ← Secrets (not committed)
├── .env.example                ← Template for .env
├── pyproject.toml              ← Python package config
├── run_orchestrator.py         ← Master entry point
├── run_backtest.py             ← CLI backtester
├── tail_logs.py                ← Terminal log viewer
├── web_logs.py                 ← Flask log viewer (port 8502)
├── start_bot.bat               ← Windows launcher
├── src/ait/
│   ├── main.py                 ← Bot process entry
│   ├── bot/
│   │   ├── orchestrator.py     ← Main brain / trading loop
│   │   ├── scheduler.py        ← Market phase state machine
│   │   └── state.py            ← SQLite state (trades, daily stats)
│   ├── broker/
│   │   ├── ibkr_client.py      ← ib_insync connection
│   │   ├── contracts.py        ← Stock/option/combo builders
│   │   ├── orders.py           ← Market/limit/combo orders
│   │   └── account.py          ← NLV, margin, buying power
│   ├── ml/
│   │   ├── ensemble.py         ← XGBoost + LightGBM predictor
│   │   ├── features.py         ← 60 technical + cross-asset features
│   │   ├── trainer.py          ← Training + drift detection
│   │   ├── regime.py           ← Market regime classifier
│   │   └── meta_label.py       ← Secondary filter (disabled)
│   ├── strategies/
│   │   ├── selector.py         ← Chooses strategies per setup
│   │   ├── iron_condor.py
│   │   ├── straddles.py
│   │   ├── spreads.py
│   │   ├── covered.py          ← CC + CSP
│   │   └── thompson.py         ← Multi-armed bandit
│   ├── risk/
│   │   ├── manager.py          ← Pre-trade validation
│   │   ├── position_sizer.py   ← Contract count calculator
│   │   ├── circuit_breaker.py  ← Daily loss halt
│   │   ├── pdt_guard.py        ← Pattern day trader protection
│   │   ├── correlation.py      ← Correlated position guard
│   │   ├── capital_tiers.py    ← Strategy availability by account size
│   │   └── hedging.py          ← Delta hedge calculator (unused)
│   ├── execution/
│   │   ├── executor.py         ← Order placement + fill tracking
│   │   ├── portfolio.py        ← Exit management (30s loop)
│   │   └── reconciler.py       ← Match local state with IBKR
│   ├── sentiment/
│   │   ├── engine.py           ← Composite sentiment
│   │   ├── news.py             ← Finnhub news
│   │   ├── fear_greed.py       ← VIX/breadth/momentum
│   │   └── finbert.py          ← Local NLP model
│   ├── data/
│   │   ├── market_data.py      ← IBKR → Polygon → Yahoo fallback
│   │   ├── historical.py       ← SQLite OHLCV cache
│   │   ├── options_chain.py    ← Chain fetching + filtering
│   │   ├── options_flow.py     ← Unusual activity detector
│   │   ├── earnings.py         ← Earnings calendar
│   │   ├── economic_calendar.py ← FOMC/CPI/NFP calendar
│   │   ├── multi_timeframe.py  ← Daily + intraday
│   │   ├── quality.py          ← Data validation
│   │   └── cache.py            ← TTL cache
│   ├── learning/
│   │   ├── engine.py           ← Post-market learning cycle
│   │   ├── analyzer.py         ← Trade pattern analysis
│   │   ├── adaptor.py          ← Apply insights as config overrides
│   │   └── counterfactual.py   ← Track skipped trades
│   ├── monitoring/
│   │   ├── analytics.py        ← Performance metrics
│   │   ├── watchdog.py         ← Health monitoring
│   │   └── duckdb_analytics.py ← Analytics database
│   ├── orchestration/
│   │   ├── master.py           ← Master process (scheduler)
│   │   └── gateway.py          ← IBC auto-start
│   ├── dashboard/
│   │   └── app.py              ← Streamlit dashboard
│   ├── notifications/
│   │   └── telegram.py         ← Telegram alerts
│   ├── backtesting/
│   │   ├── engine.py           ← Single-window backtester
│   │   ├── walkforward.py      ← Rolling windows
│   │   ├── options_sim.py      ← Black-Scholes pricing
│   │   ├── learner.py          ← In-backtest learning
│   │   └── result.py           ← Metrics calculator
│   ├── config/
│   │   └── settings.py         ← Pydantic config models
│   └── utils/
│       ├── logging.py          ← Structlog setup
│       └── time.py             ← Market hours utilities
├── tests/                      ← Unit tests
├── data/                       ← Runtime artifacts (gitignored)
│   ├── ait_state.db            ← SQLite trading state
│   ├── ait_analytics.duckdb    ← Analytics database
│   ├── historical.db           ← OHLCV cache
│   ├── counterfactual_log.json
│   └── thompson_state.json
├── models/                     ← Trained models (gitignored)
│   ├── ensemble.pkl            ← Current ensemble
│   └── archive/                ← Versioned backups
├── logs/                       ← Log files (gitignored)
│   ├── ait.log                 ← Main bot log
│   ├── orchestrator.log
│   └── bot_stdout.log
└── reports/                    ← Daily/backtest reports (gitignored)
```

---

## Development

### Running Tests

```bash
pytest                                      # All tests
pytest tests/test_risk.py -v                # Specific file
pytest --cov=src/ait                        # With coverage
```

### Code Style

```bash
ruff check src/                             # Lint
ruff format src/                            # Auto-format
mypy src/                                   # Type check
```

### Adding a New Strategy

1. Create `src/ait/strategies/my_strategy.py` inheriting from `BaseStrategy`
2. Implement `generate_signals(symbol, chain, market_direction, confidence, iv_rank, ...)`
3. Register it in `src/ait/strategies/selector.py`
4. Add to `config.yaml` under `options.strategies`
5. Add risk multiplier in `src/ait/risk/position_sizer.py` `STRATEGY_RISK` dict

### Adding a New ML Feature

1. Edit `src/ait/ml/features.py`:
   - Add computation to `_add_<group>()` method
   - Add feature name to `get_feature_names()` list
2. Rerun training: `python run_orchestrator.py --retrain`
3. Verify via `features=<N>` in log output

### Backtesting Changes

Always backtest before deploying:

```bash
# Quick: 4 symbols, fast
python run_backtest.py --symbols SPY QQQ AAPL MSFT --capital 50000

# Full: all 14 symbols, slower but realistic
python run_backtest.py --symbols SPY QQQ IWM DIA AAPL MSFT NVDA TSLA AMD AMZN META GOOGL SOFI PLTR --capital 50000
```

Results saved to `reports/backtest_<timestamp>.json`.

---

## License

Private repository — for personal/educational use. Not distributed under an open source license. Use at your own risk.

---

## Support

For issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Check `logs/ait.log` for errors
3. Verify IBKR Gateway is connected (green Status Bar)
4. File a GitHub issue with log excerpts

---

**Built for autonomous options trading. Test on paper first. Trade responsibly.**

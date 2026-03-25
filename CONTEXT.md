# AIT v2 — Autonomous Intelligent Trading Bot

## What This Is
A fully autonomous options trading bot that sells theta (iron condors + credit spreads) using ML predictions, sentiment analysis, and self-learning. Starts with $700 CAD and auto-scales strategies as capital grows.

---

## Architecture

```
run_orchestrator.py          ← Master process (start here)
  ├── BotManager             ← Starts/monitors/restarts the trading bot
  │     └── python -m ait.main --paper
  │           ├── TradingOrchestrator    ← The brain
  │           │     ├── MarketScheduler  ← Pre-market / Open / Post-market / Off-hours
  │           │     ├── ModelTrainer     ← XGBoost + LightGBM ensemble
  │           │     ├── SentimentEngine  ← FinBERT + Finnhub + Fear/Greed
  │           │     ├── StrategySelector ← Iron condors preferred, spreads as fallback
  │           │     ├── CapitalTierMgr   ← Auto-scales strategies to account size
  │           │     ├── RiskManager      ← Circuit breaker, PDT guard, position sizing
  │           │     ├── TradeExecutor    ← Places orders via IBKR
  │           │     ├── PortfolioManager ← Manages exits, trailing stops
  │           │     └── LearningEngine   ← Post-market self-learning
  │           └── IBKRClient             ← ib_insync connection to IB Gateway
  ├── APScheduler
  │     ├── health_check     ← Every 2 min (auto-restart if crashed)
  │     ├── daily_retrain    ← Mon-Fri 7:30 AM ET
  │     ├── daily_report     ← Mon-Fri 4:30 PM ET
  │     ├── weekly_backtest  ← Sunday 8 PM ET
  │     └── monthly_cleanup  ← 1st of month
  └── IBC Gateway Manager   ← Auto-starts IB Gateway (needs credentials)
```

## Key Files

| File | Purpose |
|------|---------|
| `run_orchestrator.py` | Master entry point — starts everything |
| `run_backtest.py` | CLI for walk-forward backtesting |
| `tail_logs.py` | Color-coded live log viewer |
| `start_bot.bat` | Windows launcher |
| `config.yaml` | All configuration (strategies, risk, ML, etc.) |
| `.env` | IBKR credentials, API keys |
| `src/ait/main.py` | Bot entry point |
| `src/ait/bot/orchestrator.py` | Trading brain — scan/predict/trade loop |
| `src/ait/bot/scheduler.py` | Market phase management |
| `src/ait/ml/ensemble.py` | XGBoost + LightGBM direction predictor |
| `src/ait/ml/features.py` | 49 technical features (RSI, MACD, BB, vol, etc.) |
| `src/ait/ml/trainer.py` | Model training with drift detection + rollback |
| `src/ait/backtesting/engine.py` | Backtest engine with Black-Scholes options sim |
| `src/ait/backtesting/walkforward.py` | Walk-forward backtester (train 1yr, test 3mo) |
| `src/ait/backtesting/learner.py` | In-backtest self-learning adapter |
| `src/ait/broker/ibkr_client.py` | IBKR connection with auto-reconnect |
| `src/ait/broker/contracts.py` | Contract builders (stocks, options, spreads, condors) |
| `src/ait/broker/orders.py` | Order builders (market, limit, combo, adaptive) |
| `src/ait/broker/account.py` | Account snapshot, margin, buying power |
| `src/ait/risk/capital_tiers.py` | Auto-scales strategies based on account size |
| `src/ait/risk/manager.py` | Risk validation before every trade |
| `src/ait/sentiment/engine.py` | Composite sentiment (FinBERT + news + fear/greed) |
| `src/ait/data/market_data.py` | Polygon → Yahoo fallback data chain |
| `src/ait/data/earnings.py` | Earnings calendar — blocks trades near earnings |
| `src/ait/dashboard/app.py` | Streamlit dashboard at localhost:8501 |
| `src/ait/orchestration/master.py` | Master scheduler (APScheduler) |
| `src/ait/orchestration/gateway.py` | IB Gateway auto-start via IBC |
| `src/ait/learning/engine.py` | Post-market self-learning cycle |

## Strategy Logic

### Iron Condors (primary when capital > $2k)
- Sell OTM call + put, buy wings further out
- Profit from theta decay — stock stays in range = max profit
- Wings at 1x expected move (68% probability of profit)
- Short strikes at ~0.20 delta
- 50% profit target (close when half of credit captured)
- 35% stop loss
- Max hold 21 days

### Credit Spreads (primary when capital < $2k)
- Sell put spread (bullish) or call spread (bearish)
- $1-2 wide for small accounts
- Same exit rules as iron condors

### Capital Tiers (auto-scales)

| Tier | Capital | Strategy | Tickers | Max Positions |
|------|---------|----------|---------|--------------|
| Micro | $0-$2k | $1-2 wide credit spreads | AMD, QQQ, IWM, SOFI, PLTR | 2 |
| Small | $2k-$5k | $2-5 wide spreads + small iron condors | SPY, QQQ, IWM, AMD, AAPL | 3 |
| Medium | $5k-$25k | Full iron condors | All 14 tickers | 5 |
| Large | $25k+ | Full strategy set, no PDT limits | All 14 tickers | 8 |

## ML Pipeline

- **Models**: XGBoost + LightGBM ensemble (50/50 weighted)
- **Features**: 49 technical indicators (RSI, MACD, Bollinger, volume, volatility, iv_rank, etc.)
- **Labels**: 5-day forward return — Bullish (>+1.5%), Bearish (<-1.5%), Neutral
- **Training**: 2 years of daily data (504 trading days), walk-forward cross-validation
- **Retraining**: Daily at 7:30 AM ET, automatic rollback if accuracy degrades
- **Drift detection**: Monitors prediction accuracy, triggers retrain if drifting

## Data Sources

| Source | What | Free Tier |
|--------|------|-----------|
| Yahoo Finance | Historical OHLCV, fallback for everything | Unlimited |
| Polygon.io | Historical daily data (primary) | 5 calls/min |
| Finnhub | News sentiment | 60 calls/min |
| FinBERT | NLP sentiment from news headlines | Local model |
| IBKR | Real-time quotes, order execution | Needs subscription |

## IBKR Setup

- **Account**: DUN603821 (paper)
- **Gateway**: C:\Jts\ibgateway\1044\ibgateway.exe
- **Port**: 4002
- **IBC**: Installed at C:\IBC (needs credentials in config.ini)
- **Trading mode**: Paper (all 4 configs confirm this)

---

## What's Been Done

### Core Engine
- [x] Walk-forward backtester with train/test/step windows
- [x] Black-Scholes options pricing simulation
- [x] Iron condor + vertical spread simulation
- [x] 50% profit target for credit trades
- [x] Stop loss and max hold days
- [x] Trailing stop with breakeven trigger
- [x] Position sizing based on max loss

### ML
- [x] XGBoost + LightGBM ensemble
- [x] 49 technical features
- [x] Walk-forward cross-validation with purge gap
- [x] 5-day forward return labels (±1.5%)
- [x] Model versioning, save/load/rollback
- [x] Drift detection + auto-retrain
- [x] Rollback fallback (if target version missing, loads latest)
- [x] Fixed iv_rank min_periods (20 instead of 60)
- [x] Fixed features_empty bug (300→504 days of history)

### Live Trading
- [x] IBKR connection via ib_insync
- [x] Contract builders (stocks, options, spreads, iron condors)
- [x] Order builders (market, limit, combo, adaptive)
- [x] Account management (NLV, margin, buying power)
- [x] Position reconciliation between bot and IBKR
- [x] Trade execution with slippage tracking

### Self-Learning
- [x] BacktestLearner — adapts between walk-forward windows
- [x] LearningEngine — post-market analysis of real trades
- [x] Strategy multipliers (boost winners, reduce losers)
- [x] Iron condors protected from full disable
- [x] Symbol blocking for consistent losers
- [x] CounterfactualTracker — tracks skipped trades

### Orchestration
- [x] Master orchestrator with APScheduler
- [x] Bot health check every 2 min with auto-restart
- [x] Daily ML retrain (7:30 AM ET)
- [x] Daily P&L report (4:30 PM ET)
- [x] Weekly walk-forward backtest (Sunday 8 PM ET)
- [x] Monthly log cleanup
- [x] IB Gateway auto-start module (needs IBC credentials)

### Risk Management
- [x] Capital tier system (micro/small/medium/large)
- [x] Circuit breaker (halts after consecutive losses)
- [x] PDT guard (pattern day trader protection)
- [x] Correlation guard (don't over-concentrate)
- [x] Position sizer (Kelly fraction)
- [x] Earnings calendar (skip trades near earnings)
- [x] Max daily trades limit

### Monitoring
- [x] Structured logging (structlog → file + console)
- [x] Rotating log files (10MB, 5 backups)
- [x] Color-coded live log viewer (tail_logs.py)
- [x] DuckDB analytics database
- [x] Watchdog with memory/error/latency monitoring
- [x] Streamlit dashboard

### Sentiment
- [x] FinBERT local NLP model
- [x] Finnhub news integration
- [x] Fear & Greed index (VIX-based)
- [x] Composite sentiment score per symbol

### Data
- [x] Polygon → Yahoo fallback chain
- [x] TTL caching for all data
- [x] Multi-timeframe analysis (daily + 5min)
- [x] Options flow detection (unusual activity)

---

## Backtest Results

### Full capital ($50k, 5 symbols, 2022-2026)
- **+311% total return** ($50k → $206k)
- Sharpe 1.89, max drawdown 9.5%
- 54% win rate, profit factor 1.36
- 1,542 trades, all iron condors
- 61% of windows profitable

### Realistic (no NVDA, 3% slippage)
- **+138% total return**
- Sharpe 1.51
- +49.62% alpha over buy-and-hold

### Small account ($700)
- Backtest pending...

---

## What's Next (Priority Order)

### High Priority
1. **[ ] Fund IBKR live account** — Even $1 CAD unlocks market data subscriptions and full paper trading. Currently the main blocker for real-time data.
2. **[ ] Set IBC credentials** — Edit `C:\IBC\config.ini`, fill in `IbLoginId` and `IbPassword` so Gateway auto-starts without manual login.
3. **[ ] Telegram alerts** — .env has empty TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID. Set up a Telegram bot (@BotFather), get token/chat ID, fill in .env. Code already exists in `src/ait/notifications/telegram.py`.
4. **[ ] Verify bot places trades** — ML predictions are now firing. Next market open, confirm trades actually execute on IBKR paper account.

### Medium Priority
5. **[ ] Dashboard upgrade** — Add capital tier display, live ML predictions, current positions table, real-time P&L chart.
6. **[ ] VIX contract fix verification** — Changed from ^VIX (stock) to VIX (index on CBOE). Need to confirm it resolves during market hours.
7. **[ ] Delayed market data type** — Set to type 3 (delayed) for paper account. Verify it eliminates Error 10089 warnings.
8. **[ ] Earnings calendar live test** — Verify the bot actually skips trades near earnings dates.

### Lower Priority
9. **[ ] Meta-labeler tuning** — Secondary ML model filters false positive signals. Needs more trade data to train properly.
10. **[ ] Thompson sampling validation** — Strategy selection via multi-armed bandit. Needs trade history to be useful.
11. **[ ] Add more cheap tickers** — Research other liquid, cheap options for micro tier (RIOT, SNAP, F, LCID, etc.)
12. **[ ] Position adjustment rules** — Roll untested side when tested side delta hits 25. Research showed 86% win rate with adjustments.
13. **[ ] 45 DTE entry experiment** — Backtest 45 DTE entry + 21 DTE exit vs current 21-day hold. Research was SPX-specific, need to validate on our universe.

---

## Research Findings (Data-Backed)

### From 600,000+ trade backtests:
- **50% profit target** is the single best exit rule (we have this)
- **2x credit stop loss** works for wide wings (we use 35% — may need testing)
- **21 DTE time exit** avoids gamma risk (we have max_hold_days=21)
- **45 DTE entry** is optimal for SPX (may not apply to individual stocks)
- **16 delta short strikes** = ~1 standard deviation (matches our expected move approach)
- **IV rank > 50** improves win rate from 48% to 57% (but may conflict with ML)
- **Max 25% portfolio per trade** prevents blowups (we use 5%)

### What didn't work:
- Tastytrade "standard" approach lost -7% to -93% over 11 years on SPX
- Wheel strategy doesn't beat buy-and-hold (17-year study)
- 10+ years of training data hurts ML (markets are non-stationary)

---

## How to Run

```bash
# Start everything (orchestrator + bot + scheduler)
python run_orchestrator.py

# Run backtest
python run_backtest.py --symbols SPY QQQ AMD --capital 700

# Check bot status
python run_orchestrator.py --status

# Force retrain ML models
python run_orchestrator.py --retrain

# Force backtest
python run_orchestrator.py --backtest

# Generate daily report
python run_orchestrator.py --report

# View live logs
python tail_logs.py

# Dashboard
streamlit run src/ait/dashboard/app.py
```

## Configuration

All in `config.yaml`:
- `trading.universe` — symbols to scan
- `trading.mode` — paper | live
- `risk.min_confidence` — ML confidence threshold (0.65)
- `ml.lookback_days` — training data length (504 = 2 years)
- `ml.retrain_interval_days` — days between retrains (7)
- `options.strategies` — allowed strategy types
- `learning.enabled` — self-learning on/off

Secrets in `.env`:
- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`, `IBKR_ACCOUNT`
- `POLYGON_API_KEY`, `FINNHUB_API_KEY`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `TRADING_MODE`

---

*Last updated: 2026-03-25*

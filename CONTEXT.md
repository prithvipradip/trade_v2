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
  │           │     ├── StrategySelector ← Multi-strategy simultaneous execution per window; wing widths and strangle deltas are IV-scaled and Optuna-optimized
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
| `run_optimizer.py` | CLI for Optuna strategy/ML parameter optimization |
| `tail_logs.py` | Color-coded live log viewer |
| `start_bot.bat` | Windows launcher |
| `config.yaml` | All configuration (strategies, risk, ML, etc.) |
| `.env` | IBKR credentials, API keys |
| `src/ait/main.py` | Bot entry point |
| `src/ait/bot/orchestrator.py` | Trading brain — scan/predict/trade loop |
| `src/ait/bot/scheduler.py` | Market phase management |
| `src/ait/ml/ensemble.py` | XGBoost + LightGBM direction predictor |
| `src/ait/ml/features.py` | 81 stationary features (RSI, normalized MACD, BB breach signals, vol, etc.) |
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
| `src/ait/sentiment/engine.py` | Composite sentiment (FinBERT + Finnhub + fear/greed + IB news) |
| `src/ait/data/market_data.py` | `load_daily_ohlcv()` — IB store (5-min resampled) → Yahoo fallback; real-time quotes from IBKR |
| `src/ait/data/earnings.py` | Earnings calendar — blocks trades near earnings |
| `src/ait/data/equity_stats.py` | yfinance equity fundamentals → DuckDB equity_stats table |
| `src/ait/data/fundamentals_db.py` | SQLite CRUD for IB news + analyst recommendations |
| `src/ait/data/ib_news.py` | IB news (BRFG/DJ-N) + analyst actions (BRFUPDN) fetcher; structured timing logs around every blocking IB API call (`reqNewsProviders`, `qualifyContracts`, `reqHistoricalNews`, `reqNewsArticle`, per-article sentiment) |
| `src/ait/optimization/optimizer.py` | StrategyOptimizer — Optuna TPE + MedianPruner |
| `src/ait/optimization/param_spaces.py` | Parameter spaces per strategy and ML model |
| `src/ait/optimization/objectives.py` | Objective functions (sharpe_ratio, composite, etc.) |
| `src/ait/optimization/results.py` | OptimizationResult + summary/save/apply_to_config |
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
- **Features**: 81 stationary technical indicators (all normalized — no raw price levels; MACD divided by close, BB breach signals replace raw BB levels)
- **Labels**: 5-day forward return — Bullish (>+1.5%), Bearish (<-1.5%), Neutral
- **Training**: 2 years of daily data (504 trading days), walk-forward cross-validation
- **Retraining**: Daily at 7:30 AM ET, automatic rollback if accuracy degrades
- **Drift detection**: Monitors prediction accuracy, triggers retrain if drifting

## Data Sources

| Source | What | Notes |
|--------|------|-------|
| IBKR 5-min SQLite store | Primary daily OHLCV (resampled from 5-min bars); intraday VLMC features | Requires IB backfill; 2 years stored locally |
| Yahoo Finance | Daily OHLCV fallback (when IB store has <60 bars); equity fundamentals (P/E, beta, sector) | Unlimited; no key required |
| Finnhub | News sentiment | 60 calls/min |
| FinBERT | NLP sentiment from news headlines | Local model |
| IBKR | Real-time quotes, order execution, options chains, news, analyst actions | Needs subscription |

**IB news providers in use:**
- General news: desired set `{BRFG, DJ-N, DJ-RTG, DJ-RTPRO, DJNL}` — filtered at startup via `reqNewsProviders()` to only subscribed codes (avoids Error 321 on accounts that lack a provider). Active set logged as `news_providers_active`.
- Analyst actions: `BRFUPDN` (Briefing.com) → stored in `data/fundamentals.db/analyst_recommendations`

**News/analyst fetch API contract:** Both `fetch_and_store_news()` and `fetch_and_store_analyst_actions()` return `(fetched, inserted)` — `fetched` is what IB returned; `inserted` is net-new DB rows. Callers use `fetched==0` to detect a dead feed, not `inserted==0` (which is normal after first run, since `INSERT OR IGNORE` deduplicates on subsequent fetches).

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
- [x] 81 stationary technical features (MACD normalized by close, BB breach signals, raw price levels excluded)
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
- [x] FinBERT tokenizer loaded explicitly with `clean_up_tokenization_spaces=True` (suppresses `transformers` FutureWarning)
- [x] Finnhub news integration
- [x] Fear & Greed index (VIX-based)
- [x] Composite sentiment score per symbol
- [x] IB news sentiment integration (pre-scored at ingest, weight 0.20)

### Data
- [x] IB store → Yahoo fallback chain for daily OHLCV (`load_daily_ohlcv()` in `market_data.py`)
- [x] VLMC intraday features merged into ML training via `FeatureEngine.compute(intraday_store=...)` 
- [x] TTL caching for all data
- [x] Multi-timeframe analysis (daily + 5min)
- [x] Options flow detection (unusual activity)
- [x] Equity descriptive stats (yfinance → DuckDB `equity_stats` table, daily refresh)
- [x] IB news archive (BRFG/DJ-N/etc → SQLite `fundamentals.db`)
- [x] IB analyst recommendations (BRFUPDN → SQLite `fundamentals.db`, structured parsing)

### Parameter Optimization
- [x] Optuna `StrategyOptimizer` — Bayesian/TPE with MedianPruner
- [x] Per-strategy parameter spaces (iron_condor, long_call, bull_call_spread, bear_put_spread, put_credit_spread, short_strangle, long_strangle)
- [x] `delta_short` / `delta_long` / `iv_floor` / `wing_floor_dollars` / `max_hold_days` wired into `Backtester.__init__` and searchable by optimizer
- [x] `wing_k` — vol-scaled wing multiplier: `wing = wing_k × price × IV × √(DTE/365)`; Optuna-optimized per window [0.30–2.00]; `wing_floor_dollars` remains as hard minimum floor
- [x] `delta_iv_scale` — IV-driven delta scaling for strangles: 0=static, 1=full response; high IV → lower effective delta → further OTM strikes
- [x] `BacktestConfig` added to `settings.py` — `initial_capital`, `position_size_pct`, `wing_floor_dollars`, `iv_floor`, `wing_k`, `delta_iv_scale` all read from `config.yaml`; no code changes needed to adjust capital size or spread width
- [x] `profit_target_pct` cap at 0.50 for credit trades removed — optimizer can explore the full range
- [x] ML hyperparameter spaces (XGBoost, LightGBM)
- [x] Objective functions: sharpe_ratio, composite, profit_factor, win_rate
- [x] Min-trade penalty (two-tier) — hard floor: < 3 trades always scores −100; quadratic penalty: `(actual/min_trades)²` between 3 and `optimize_min_trades`; prevents degenerate low-sample Sharpe inflation
- [x] Early stopping — `_EarlyStopCallback` halts a window study after `optimize_patience` consecutive non-improving trials (0 = disabled)
- [x] Conditional warm-start — enqueues prior window's best params if OOS `win_rate ≥ 75%` AND `total_trades ≥ 5`; cold-starts otherwise
- [x] `range_threshold_pct` config field; `RangePredictor` `horizon_days` auto-linked to `max_hold_days` per window
- [x] `min_confidence` search range capped at 0.70 (upper bound) across all strategy spaces — prevents Optuna from selecting values of 0.72–0.85 that generate 0 OOS trades in 63-day test windows
- [x] Walk-forward `optimize_per_window` integration
- [x] Resumable studies via SQLite storage (`load_if_exists=True`)
- [x] `run_optimizer.py` CLI
- [x] `OptimizationResult` — summary table, JSON save, apply_to_config

### New Strategies (engine)
- [x] `short_strangle` — sell OTM call + sell OTM put (no wings); IV-scaled delta: high IV → go further OTM; margin modeled as 20% of underlying per side
- [x] `long_strangle` — buy OTM call + buy OTM put; profit from large moves or IV expansion; IV-scaled delta; repriced correctly in `_reprice_position`

### Integration Tests (`tests/test_integration.py`)
- Gated by `RUN_INTEGRATION_TESTS=1` env var — skipped entirely in the normal `pytest` run
- Requires IB Gateway on port 4001 (live) or 4002 (paper)
- Writes to isolated tables (`test_equity_stats`, `test_news`, `test_analyst_recommendations`) — never touches production data
- Typically completes in ~20 seconds: IB API calls (`reqHistoricalNews`, `reqNewsArticle`) each take <0.2s; FinBERT cold model load adds ~7s once per session
- `test_fetch_aapl_analyst_actions_second_fetch` replaced `test_fetch_spy_analyst_actions` — SPY is an ETF and receives no analyst upgrade/downgrade coverage, causing that test to always skip via `_require_news`

### Backtester Modelling Improvements (fixed)
- **Dynamic IV during hold** — `_reprice_position` now calls `_get_current_iv` which blends `entry_iv` (70%) with current `realized_vol × 1.15` (30%); vega P&L is now non-zero
- **Volatility skew** — `_get_leg_iv` applies a linear log-moneyness skew: OTM puts +1% IV per 10% below ATM, OTM calls +0.2% IV per 10% above ATM; controlled by `skew_factor` param (default 1.0, 0.0 = flat)

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

### QQQ integration test (2-yr walk-forward, 2024-2026, multi-strategy)
- **−21% return** (18 windows, 50 Optuna trials each)
- QQQ 2025 had multiple large directional moves (Liberation Day tariff shock in April, +15% recovery, further volatility Q3-Q4) that are hostile to iron condors
- Iron_condor was the only profitable strategy (+$1.6K, ~36-40% win rate) across the test period
- Directional fallback strategies (bear_put_spread, bull_call_spread) lost heavily because ML model directional accuracy is ~22% (too low for reliable directional bets)
- Range predictor gate (52% accuracy) blocks iron_condors in volatile conditions — no fallback to directional strategies (directional fallback was tried and reverted; it worsened total losses by $35K)
- **Key lesson**: Iron condors are the reliable core; directional/volatility strategies require significantly higher ML directional accuracy to be profitable

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
5. **[ ] Dashboard upgrade** — Add capital tier display, live ML predictions, current positions table, real-time P&L chart. Consider adding equity_stats view and analyst recommendation feed.
6. **[ ] VIX contract fix verification** — Changed from ^VIX (stock) to VIX (index on CBOE). Need to confirm it resolves during market hours.
7. **[ ] Delayed market data type** — Set to type 3 (delayed) for paper account. Verify it eliminates Error 10089 warnings.
8. **[ ] Earnings calendar live test** — Verify the bot actually skips trades near earnings dates.
9. **[ ] Run optimizer on production parameters** — Use `run_optimizer.py --n-trials 200 --storage sqlite:///data/optuna.db` to find better iron condor params; apply with `--apply`.

### Lower Priority
10. **[ ] Meta-labeler tuning** — Secondary ML model filters false positive signals. Needs more trade data to train properly.
11. **[ ] Thompson sampling validation** — Strategy selection via multi-armed bandit. Needs trade history to be useful.
12. **[ ] Add more cheap tickers** — Research other liquid, cheap options for micro tier (RIOT, SNAP, F, LCID, etc.)
13. **[ ] Position adjustment rules** — Roll untested side when tested side delta hits 25. Research showed 86% win rate with adjustments.
14. **[ ] 45 DTE entry experiment** — Backtest 45 DTE entry + 21 DTE exit vs current 21-day hold. Research was SPX-specific, need to validate on our universe.
15. **[ ] Wire analyst recommendations into strategy decisions** — `fundamentals_db.get_analyst_recs()` is stored but not yet used in the entry pipeline. Could boost/suppress entries based on recent analyst consensus.

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

# Backtest with per-window Optuna optimization
python run_backtest.py --symbols SPY QQQ --optimize-per-window --optimize-n-trials 50

# Check bot status
python run_orchestrator.py --status

# Force retrain ML models
python run_orchestrator.py --retrain

# Force backtest
python run_orchestrator.py --backtest

# Generate daily report
python run_orchestrator.py --report

# Refresh equity fundamentals (yfinance → DuckDB)
python run_orchestrator.py --refresh-fundamentals

# Fetch IB news + analyst actions → fundamentals.db
python run_orchestrator.py --fetch-news

# Run Optuna strategy parameter optimizer
python run_optimizer.py --strategies iron_condor --symbols SPY QQQ --n-trials 100 --objective sharpe_ratio

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
- `backtest.initial_capital` — starting capital for walk-forward / integration tests (default $100,000)
- `backtest.position_size_pct` — fraction of capital risked per trade on a max-loss basis (default 0.05); raise to 0.20 for accounts < $10k
- `backtest.wing_floor_dollars` — minimum iron condor wing width in dollars (default 5.0); lower to 1.0 for cheap underlyings like MARA, SOFI, RIOT
- `backtest.iv_floor` — minimum synthetic IV used in Black-Scholes pricing (default 0.20); prevents near-zero credits in calm markets

Secrets in `.env`:
- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`, `IBKR_ACCOUNT`
- `POLYGON_API_KEY`, `FINNHUB_API_KEY`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `TRADING_MODE`

---

*Last updated: 2026-05-08*

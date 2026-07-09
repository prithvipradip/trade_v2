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

# Feature Plan: Equity Stats, IB News/Analyst, and Parameter Optimization

## Context

This trading system (AIT v2) is a production options bot that uses XGBoost/LightGBM for direction prediction, 10 options strategies, walk-forward backtesting, and a Thompson-sampling self-learning engine. Three new capabilities have been requested.

**Live IB account findings (verified during planning):**
- 8 active news providers: `BRFG`, `BRFUPDN`, `DJ-N`, `DJ-RTA`, `DJ-RTE`, `DJ-RTG`, `DJ-RTPRO`, `DJNL`
- `BRFUPDN` (Briefing.com Analyst Actions) returns structured analyst rating changes with firm, action, rating, and price target — no need for `reqFundamentalData` for analyst data
- Historical news depth: reliable up to ~90 days; sparse beyond 180 days
- Article text from `reqNewsArticle` is structured and parseable (HTML-entities, line-separated fields)
- Headline format: `{A:...:K:<score>:C:<confidence>}!<clean headline>` — strip prefix to get usable text

**Design decisions:**
- Equity stats → DuckDB `ait_analytics.duckdb`, source = **yfinance** (works offline/in backtest, 80+ fields)
- News → new `data/fundamentals.db` (SQLite), fetched from IB using all providers except BRFUPDN
- Analyst recommendations → same `data/fundamentals.db`, fetched from IB using BRFUPDN + `reqNewsArticle`
- Optimizer → Optuna (Bayesian/TPE with pruning)

---

## Feature 1: Equity Descriptive Statistics (DuckDB via yfinance)

### What/Why

Store per-symbol fundamental data (sector, industry, P/E, beta, market cap, analyst consensus, etc.) in the analytics DuckDB. Snapshot-only at fetch time; history builds by polling daily.

### Schema — add to `DuckDBAnalytics._init_schema()`

```sql
CREATE TABLE IF NOT EXISTS equity_stats (
    symbol          VARCHAR PRIMARY KEY,
    updated_at      TIMESTAMP NOT NULL,
    -- Identity
    company_name    VARCHAR DEFAULT '',
    sector          VARCHAR DEFAULT '',
    industry        VARCHAR DEFAULT '',
    country         VARCHAR DEFAULT '',
    exchange        VARCHAR DEFAULT '',
    -- Valuation
    market_cap      BIGINT  DEFAULT 0,
    pe_ratio        DOUBLE  DEFAULT 0,
    forward_pe      DOUBLE  DEFAULT 0,
    pb_ratio        DOUBLE  DEFAULT 0,
    ps_ratio        DOUBLE  DEFAULT 0,
    ev_ebitda       DOUBLE  DEFAULT 0,
    -- Per-share
    eps             DOUBLE  DEFAULT 0,
    book_value_ps   DOUBLE  DEFAULT 0,
    revenue_ps      DOUBLE  DEFAULT 0,
    -- Dividends
    dividend_yield  DOUBLE  DEFAULT 0,
    dividend_rate   DOUBLE  DEFAULT 0,
    payout_ratio    DOUBLE  DEFAULT 0,
    -- Risk / technicals
    beta            DOUBLE  DEFAULT 0,
    week52_high     DOUBLE  DEFAULT 0,
    week52_low      DOUBLE  DEFAULT 0,
    avg_volume_30d  BIGINT  DEFAULT 0,
    float_shares    BIGINT  DEFAULT 0,
    shares_outstanding BIGINT DEFAULT 0,
    -- Analyst consensus (from yfinance)
    analyst_target_mean  DOUBLE  DEFAULT 0,
    analyst_target_high  DOUBLE  DEFAULT 0,
    analyst_target_low   DOUBLE  DEFAULT 0,
    analyst_rating       VARCHAR DEFAULT '',  -- strong buy/buy/hold/sell/strong sell
    analyst_count        INTEGER DEFAULT 0
)
```

### New file: `src/ait/data/equity_stats.py`

Class: `EquityStatsService`

```python
import yfinance as yf
from datetime import datetime

class EquityStatsService:
    _FIELD_MAP = {
        'longName':                        'company_name',
        'sector':                          'sector',
        'industry':                        'industry',
        'country':                         'country',
        'exchange':                        'exchange',
        'marketCap':                       'market_cap',
        'trailingPE':                      'pe_ratio',
        'forwardPE':                       'forward_pe',
        'priceToBook':                     'pb_ratio',
        'priceToSalesTrailing12Months':    'ps_ratio',
        'enterpriseToEbitda':              'ev_ebitda',
        'trailingEps':                     'eps',
        'bookValue':                       'book_value_ps',
        'revenuePerShare':                 'revenue_ps',
        'dividendYield':                   'dividend_yield',
        'dividendRate':                    'dividend_rate',
        'payoutRatio':                     'payout_ratio',
        'beta':                            'beta',
        'fiftyTwoWeekHigh':               'week52_high',
        'fiftyTwoWeekLow':                'week52_low',
        'averageVolume':                   'avg_volume_30d',
        'floatShares':                     'float_shares',
        'sharesOutstanding':              'shares_outstanding',
        'targetMeanPrice':                'analyst_target_mean',
        'targetHighPrice':                'analyst_target_high',
        'targetLowPrice':                 'analyst_target_low',
        'recommendationKey':              'analyst_rating',
        'numberOfAnalystOpinions':        'analyst_count',
    }

    def __init__(self, analytics: DuckDBAnalytics): ...

    def refresh_all(self, symbols: list[str]) -> None:
        """Fetch yfinance .info for each symbol, upsert into DuckDB."""
        for symbol in symbols:
            try:
                info = yf.Ticker(symbol).info
                stats = self._map_fields(symbol, info)
                self._analytics.upsert_equity_stats(stats)
            except Exception:
                log.warning(f'equity_stats fetch failed for {symbol}')

    def _map_fields(self, symbol: str, info: dict) -> dict:
        row = {'symbol': symbol, 'updated_at': datetime.utcnow()}
        for yf_key, col in self._FIELD_MAP.items():
            row[col] = info.get(yf_key) or 0
        return row
```

### New methods in `DuckDBAnalytics`

```python
def upsert_equity_stats(self, stats: dict) -> None:
    """INSERT OR REPLACE into equity_stats (snapshot replaces on each daily refresh)."""

def get_equity_stats(self, symbol: str) -> dict | None:
    """Return latest snapshot for one symbol."""

def get_all_equity_stats(self) -> list[dict]:
    """Return all symbols (for dashboard display)."""
```

### Update triggers

| Trigger | Where | Cadence |
|---|---|---|
| Pre-market | `src/ait/main.py` `_pre_market()` | Refresh if `updated_at` > 24h stale |
| CLI | `run_orchestrator.py --refresh-fundamentals` | Force-refresh all symbols |
| Scheduled | APScheduler Sunday 8 PM | After weekly walk-forward backtest |

### Files to modify

- [src/ait/monitoring/duckdb_analytics.py](src/ait/monitoring/duckdb_analytics.py) — add schema + 3 methods
- [src/ait/main.py](src/ait/main.py) — instantiate `EquityStatsService`, call in `_pre_market()`
- [run_orchestrator.py](run_orchestrator.py) — add `--refresh-fundamentals` flag

---

## Feature 2: IB News & Analyst Recommendations (`fundamentals.db`)

### What/Why

Persist IB-sourced news and analyst rating changes to a dedicated SQLite DB. `BRFUPDN` (Briefing.com Analyst Actions) returns structured analyst actions (firm, action, rating, target) parseable via `reqNewsArticle`. General news feeds into `SentimentEngine`.

**IB provider routing:**
- News: `BRFG + DJ-N + DJ-RTG + DJ-RTPRO + DJNL` (general market + company news)
- Analyst actions: `BRFUPDN` only
- Fetch depth: 1–7 days for routine; backfill 90 days on first startup

### New database: `data/fundamentals.db`

```sql
CREATE TABLE IF NOT EXISTS news (
    article_id   TEXT PRIMARY KEY,
    symbol       TEXT NOT NULL,
    provider     TEXT NOT NULL,
    headline     TEXT NOT NULL,        -- cleaned (strip {A:...:K:...:C:...}! prefix)
    url          TEXT DEFAULT '',
    published_at TEXT NOT NULL,        -- ISO-8601: when the article was published (from IB n.time)
    fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),  -- when WE stored it in our DB
    sentiment    REAL DEFAULT 0.0      -- scored by FinBERT/keyword scorer at fetch time
);
CREATE INDEX IF NOT EXISTS idx_news_symbol_date
    ON news(symbol, published_at DESC);

CREATE TABLE IF NOT EXISTS analyst_recommendations (
    id              TEXT PRIMARY KEY,  -- sha256(symbol + issued_at + firm)[:16]
    symbol          TEXT NOT NULL,
    issued_at       TEXT NOT NULL,     -- ISO-8601: date the analyst issued the rating (from "Issuance Date:" in article body)
    published_at    TEXT NOT NULL,     -- ISO-8601: when IB published it to the feed (from n.time)
    fetched_at      TEXT NOT NULL DEFAULT (datetime('now')),  -- when WE stored it in our DB
    firm            TEXT DEFAULT '',
    action          TEXT DEFAULT '',   -- reiterated/upgraded/downgraded/initiated/removed
    rating          TEXT DEFAULT '',   -- Underweight/Overweight/Buy/Neutral/Sell
    price_target    REAL DEFAULT 0.0,
    prior_target    REAL DEFAULT 0.0,
    raw_text        TEXT DEFAULT '',   -- full article text for auditing
    UNIQUE(symbol, issued_at, firm)
);
CREATE INDEX IF NOT EXISTS idx_analyst_symbol_date
    ON analyst_recommendations(symbol, issued_at DESC);
```

**Three timestamps per analyst record:**
- `issued_at` — when the analyst actually made the call (parsed from article body)
- `published_at` — when it hit the IB feed wire (from `n.time`)
- `fetched_at` — when our system stored it

### Headline parsing

IB encodes metadata in the headline prefix — strip before storing:
```
{A:800015:L:en:K:0.63:C:0.629}!JP Morgan reiterated Apple (AAPL) with Overweight
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                 Clean headline stored in DB
```

### Article text parsing for analyst actions

`reqNewsArticle(providerCode, articleId, [])` returns structured text:
```
Barclays reiterated Apple (AAPL) coverage with Underweight rating and price target $239
Previous price target: $230
Issuance Date: 2026-01-30
```
Parse with regex: firm, action verb, symbol, rating, target price, prior target, issuance date.

### New files

#### `src/ait/data/fundamentals_db.py` — `FundamentalsStore`

```python
class FundamentalsStore:
    def __init__(self, db_path: Path = Path("data/fundamentals.db")): ...
    def _init_schema(self) -> None: ...

    def insert_news(self, articles: list[dict]) -> int:
        """INSERT OR IGNORE — records are NEVER overwritten after first insertion.
        Idempotent: re-fetching the same date window on a later trigger skips duplicates."""

    def get_recent_news(self, symbol: str, hours: int = 24) -> list[dict]:
        """For SentimentEngine integration."""

    def insert_analyst_rec(self, recs: list[dict]) -> int:
        """INSERT OR IGNORE — UNIQUE(symbol, issued_at, firm) prevents re-insertion
        even when the daily fetch re-fetches overlapping windows. Never overwrites."""

    def get_analyst_recs(self, symbol: str, days: int = 30) -> list[dict]:
        """Recent analyst actions for a symbol."""
```

**Why `INSERT OR IGNORE` (not `INSERT OR REPLACE`):**
News and analyst actions are immutable facts. `INSERT OR REPLACE` would reset `fetched_at` and lose data. `INSERT OR IGNORE` ensures every trigger run is idempotent.

#### `src/ait/data/ib_news.py` — `IBNewsService`

```python
NEWS_PROVIDERS = 'BRFG+DJ-N+DJ-RTG+DJ-RTPRO+DJNL'
ANALYST_PROVIDER = 'BRFUPDN'

class IBNewsService:
    def __init__(self, ib_client: IBKRClient, store: FundamentalsStore,
                 sentiment_fn: Callable[[str], float]): ...

    def fetch_and_store_news(self, symbol: str, hours_back: int = 24) -> int:
        """Fetch general news, score sentiment, persist via INSERT OR IGNORE."""

    def fetch_and_store_analyst_actions(self, symbol: str, hours_back: int = 168) -> int:
        """Fetch BRFUPDN items, call reqNewsArticle for full text,
        parse structured fields, persist via INSERT OR IGNORE."""

    def _strip_prefix(self, headline: str) -> str:
        """Remove '{A:...:K:...:C:...}!' prefix from IB headlines."""
        if headline.startswith('{') and '!' in headline:
            return headline.split('!', 1)[1]
        return headline

    def _parse_analyst_text(self, symbol: str, feed_time, text: str) -> dict | None:
        """Regex-parse Briefing.com article body into structured dict."""
```

### Integration with `SentimentEngine`

Modify `src/ait/sentiment/engine.py` to accept `FundamentalsStore` and add IB news as an additional sentiment source (weight configurable via `SentimentConfig.ib_news_weight`):

```python
if self._fundamentals_store:
    news = self._fundamentals_store.get_recent_news(symbol, hours=24)
    if news:
        ib_score = statistics.mean(n['sentiment'] for n in news)
        scores.append(('ib_news', ib_score, self._config.ib_news_weight))
```

### Update triggers

| Trigger | Frequency | Action |
|---|---|---|
| Pre-market `_pre_market()` | Daily | Last 24h news + last 7d analyst actions for all symbols |
| Trading cycle `_trading_cycle()` | Every 5 min | Last 30 min news for symbols being scored |
| First startup | Once | Backfill 90 days of news + analyst actions for all symbols |
| CLI `--fetch-news` | On demand | Full refresh |

### Files to modify

- [src/ait/main.py](src/ait/main.py) — instantiate `FundamentalsStore`, `IBNewsService`; call in `_pre_market()` + `_trading_cycle()`
- [src/ait/sentiment/engine.py](src/ait/sentiment/engine.py) — accept `FundamentalsStore`, add IB news branch
- [src/ait/config/settings.py](src/ait/config/settings.py) — add `ib_news_weight: float = 0.20` to `SentimentConfig`
- [run_orchestrator.py](run_orchestrator.py) — add `--fetch-news` CLI flag

---

## Feature 3: Optuna Strategy Parameter Optimization Module

### What/Why

No formal hyperparameter optimizer exists. The self-learning engine adapts strategy weights post-market but cannot explore novel parameter combinations systematically. Optuna (Bayesian/TPE) finds better delta ranges, DTE windows, confidence thresholds, and ML hyperparameters in far fewer trials than grid search, with pruning to stop bad trials early.

### New package: `src/ait/optimization/`

```
src/ait/optimization/
├── __init__.py
├── optimizer.py        # StrategyOptimizer (main class)
├── param_spaces.py     # Per-strategy + ML parameter spaces
├── objectives.py       # Objective functions
└── results.py          # OptimizationResult dataclass + reporting
```

### `param_spaces.py`

```python
IRON_CONDOR_SPACE = {
    'delta_min':         ('float', 0.15, 0.30),
    'delta_max':         ('float', 0.25, 0.45),
    'dte_min':           ('int',   7,    21),
    'dte_max':           ('int',   28,   60),
    'min_confidence':    ('float', 0.55, 0.80),
    'stop_loss_pct':     ('float', 0.30, 0.70),
    'profit_target_pct': ('float', 0.40, 0.80),
    'trailing_stop_pct': ('float', 0.15, 0.40),
    'iv_floor':          ('int',   10,   30),
}
LONG_CALL_SPACE = {
    'delta_target':      ('float', 0.40, 0.70),
    'dte_min':           ('int',   14,   30),
    'dte_max':           ('int',   45,   90),
    'min_confidence':    ('float', 0.60, 0.85),
    'stop_loss_pct':     ('float', 0.30, 0.60),
    'profit_target_pct': ('float', 0.60, 1.50),
}
XGBOOST_SPACE = {
    'n_estimators':      ('int',   50,   400),
    'learning_rate':     ('float', 0.01, 0.30, {'log': True}),
    'max_depth':         ('int',   3,    8),
    'min_child_weight':  ('int',   1,    10),
    'subsample':         ('float', 0.5,  1.0),
    'colsample_bytree':  ('float', 0.5,  1.0),
    'gamma':             ('float', 0.0,  5.0),
}
LIGHTGBM_SPACE = {
    'n_estimators':      ('int',   50,   400),
    'learning_rate':     ('float', 0.01, 0.30, {'log': True}),
    'num_leaves':        ('int',   20,   150),
    'min_child_samples': ('int',   5,    50),
    'subsample':         ('float', 0.5,  1.0),
    'colsample_bytree':  ('float', 0.5,  1.0),
}
STRATEGY_SPACES: dict[str, dict] = {
    'iron_condor':    IRON_CONDOR_SPACE,
    'long_call':      LONG_CALL_SPACE,
    # one entry per strategy
}
```

### `objectives.py`

```python
OBJECTIVES = {
    'sharpe_ratio':  lambda r: r.sharpe_ratio,
    'composite':     lambda r: 0.4*r.sharpe_ratio + 0.4*r.win_rate - 0.2*abs(r.max_drawdown),
    'profit_factor': lambda r: r.profit_factor,
    'win_rate':      lambda r: r.win_rate,
}
```

### `optimizer.py` — `StrategyOptimizer`

```python
class StrategyOptimizer:
    def __init__(
        self,
        symbols: list[str],
        strategies: list[str],
        n_trials: int = 100,
        n_jobs: int = 1,
        objective: str = 'sharpe_ratio',
        study_name: str | None = None,
        storage: str | None = None,   # 'sqlite:///data/optuna.db' for persistence
        optimize_ml: bool = False,
        initial_capital: float = 50_000.0,
        train_days: int = 365,
    ): ...

    def run(self) -> OptimizationResult:
        study = optuna.create_study(
            direction='maximize',
            study_name=self._study_name,
            storage=self._storage,
            load_if_exists=True,    # resume interrupted studies
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        study.optimize(self._objective_fn, n_trials=self._n_trials,
                       n_jobs=self._n_jobs)
        return OptimizationResult(study)

    def _objective_fn(self, trial: optuna.Trial) -> float:
        params = self._suggest_params(trial)
        backtester = Backtester(data=self._data, strategies=self._strategies, **params)
        result = backtester.run()
        trial.report(result.sharpe_ratio, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return OBJECTIVES[self._objective](result)
```

### `results.py` — `OptimizationResult`

```python
@dataclass
class OptimizationResult:
    study: optuna.Study

    @property
    def best_params(self) -> dict: ...
    @property
    def best_metrics(self) -> dict: ...

    def summary(self) -> str:
        """Top-N trials as a formatted table."""

    def apply_to_config(self, config_path: str) -> None:
        """Write best params to config.yaml strategy overrides section."""

    def save(self, path: str = 'reports/optimization_result.json') -> None:
        """Persist best params + metrics as JSON."""
```

### CLI: `run_optimizer.py` (new top-level file, matches existing pattern)

```bash
python run_optimizer.py \
  --strategies iron_condor bull_call_spread \
  --symbols SPY QQQ \
  --n-trials 200 \
  --objective composite \
  --n-jobs 4 \
  --train-days 365 \
  --capital 50000 \
  --optimize-ml \
  --storage sqlite:///data/optuna.db \
  --apply    # write best params back to config.yaml
```

### Walk-forward integration

Add two fields to `WalkForwardConfig` in [src/ait/backtesting/walkforward.py](src/ait/backtesting/walkforward.py):

```python
optimize_per_window: bool = False   # enable per-window Optuna search
optimize_n_trials: int = 50         # trials per window (user-configurable, no hardcoded value)
```

Expose as CLI flags on `run_backtest.py`:
```bash
python run_backtest.py --symbols SPY QQQ \
  --optimize-per-window \
  --optimize-n-trials 100
```

When enabled, each training window runs `StrategyOptimizer(n_trials=config.optimize_n_trials)` before its test window. The trained params are then passed to the `Backtester` for that window's test slice.

### ML retraining integration

Add `optimize_hyperparams: bool = False` to `ModelTrainer.retrain()` in [src/ait/ml/trainer.py](src/ait/ml/trainer.py). When enabled, runs `StrategyOptimizer(optimize_ml=True, n_trials=n_trials)` before fitting the final XGBoost/LightGBM models, using the best hyperparams found.

### Optuna study persistence

Store studies in `data/optuna.db` (separate SQLite file, does not pollute trade state). `load_if_exists=True` allows resuming interrupted runs and accumulating trial history across weekly retrains.

---

## Dependency Change

Add to [pyproject.toml](pyproject.toml) `[project.dependencies]`:

```toml
"optuna>=3.6",
```

No other new dependencies — `sqlite3`, `re`, `xml.etree.ElementTree`, `hashlib` are all stdlib.

---

## Critical Files Summary

### New files to create

| File | Purpose |
|---|---|
| [src/ait/data/equity_stats.py](src/ait/data/equity_stats.py) | `EquityStatsService` — yfinance fetch → DuckDB upsert |
| [src/ait/data/fundamentals_db.py](src/ait/data/fundamentals_db.py) | `FundamentalsStore` — SQLite fundamentals.db CRUD |
| [src/ait/data/ib_news.py](src/ait/data/ib_news.py) | `IBNewsService` — IB news + analyst actions → store |
| [src/ait/optimization/\_\_init\_\_.py](src/ait/optimization/__init__.py) | Package init |
| [src/ait/optimization/optimizer.py](src/ait/optimization/optimizer.py) | `StrategyOptimizer` — Optuna study wrapper |
| [src/ait/optimization/param_spaces.py](src/ait/optimization/param_spaces.py) | Parameter spaces per strategy + ML models |
| [src/ait/optimization/objectives.py](src/ait/optimization/objectives.py) | Objective functions (Sharpe, composite, etc.) |
| [src/ait/optimization/results.py](src/ait/optimization/results.py) | `OptimizationResult` + reporting |
| [run_optimizer.py](run_optimizer.py) | CLI entry point (mirrors `run_backtest.py` pattern) |

### Existing files to modify

| File | Change |
|---|---|
| [src/ait/monitoring/duckdb_analytics.py](src/ait/monitoring/duckdb_analytics.py) | Add `equity_stats` table schema + `upsert_equity_stats`, `get_equity_stats`, `get_all_equity_stats` |
| [src/ait/sentiment/engine.py](src/ait/sentiment/engine.py) | Accept `FundamentalsStore`, add IB news sentiment branch |
| [src/ait/config/settings.py](src/ait/config/settings.py) | Add `ib_news_weight: float = 0.20` to `SentimentConfig` |
| [src/ait/main.py](src/ait/main.py) | Instantiate new services; call refresh in pre-market + trading cycle |
| [src/ait/backtesting/walkforward.py](src/ait/backtesting/walkforward.py) | Add `optimize_per_window: bool` and `optimize_n_trials: int` to `WalkForwardConfig` |
| [src/ait/ml/trainer.py](src/ait/ml/trainer.py) | Add `optimize_hyperparams: bool` flag to `retrain()` |
| [run_orchestrator.py](run_orchestrator.py) | Add `--refresh-fundamentals`, `--fetch-news` CLI flags |
| [pyproject.toml](pyproject.toml) | Add `optuna>=3.6` |

---

## Verification Plan

### Feature 1 — Equity Stats

1. `python run_orchestrator.py --refresh-fundamentals`
2. `duckdb data/ait_analytics.duckdb -c "SELECT symbol, sector, pe_ratio, beta, analyst_rating FROM equity_stats;"`
3. Verify non-null values for SPY, QQQ, NVDA, AMZN, AMD, IWM, DIA
4. Re-run after 24h — confirm `updated_at` refreshes

### Feature 2 — News & Analyst Recs

1. `python run_orchestrator.py --fetch-news`
2. `sqlite3 data/fundamentals.db "SELECT symbol, headline, sentiment, published_at, fetched_at FROM news LIMIT 10;"`
3. Confirm headlines have no `{A:...}!` prefix; sentiment ∈ [-1, 1]; both timestamps populated
4. `sqlite3 data/fundamentals.db "SELECT firm, action, rating, price_target, issued_at, published_at, fetched_at FROM analyst_recommendations LIMIT 10;"`
5. Confirm all three timestamps present; firm/rating/target fields parsed correctly
6. Run `--fetch-news` a second time — confirm no duplicate rows inserted (INSERT OR IGNORE working)
7. Place a test trade; confirm logs show `ib_news` source contributing to sentiment score

### Feature 3 — Optuna Optimizer

1. `python run_optimizer.py --strategies iron_condor --symbols SPY --n-trials 20 --objective sharpe_ratio`
2. Confirm summary table printed with top-N trials and metrics
3. Confirm `reports/optimization_result.json` created with `best_params` and `best_metrics` keys
4. `python run_optimizer.py --strategies iron_condor --n-trials 5 --apply` → verify `config.yaml` updated
5. `python run_backtest.py --symbols SPY --optimize-per-window --optimize-n-trials 10` → completes without error
6. `python run_optimizer.py --storage sqlite:///data/optuna.db --n-trials 5` → verify `data/optuna.db` created; re-run with same `--study-name` to confirm it resumes (trial count accumulates)

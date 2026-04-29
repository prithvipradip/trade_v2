# AIT v2 Roadmap

Long-term improvements ranked by impact. Items in `[ACTIVE]` are being worked on now; everything else is the backlog.

---

## Now Active (this session)

- [ ] **SEC EDGAR 8-K real-time monitor** — Material events trigger position-flatten on held names. Free via EDGAR API. ~1 day of work.
- [ ] **Sentiment as ML feature** — Currently FinBERT/Finnhub used only for signal weighting. Wire `composite_score`, `news_score`, `fear_greed_score` into `FeatureEngine` as model inputs.
- [ ] **Options flow as ML feature** — `OptionsFlowDetector` produces `put_call_ratio`, `bias_strength`. Wire these into `FeatureEngine`.

---

## Tier 1 — Change What We Predict (HIGHEST IMPACT)

The model currently predicts 5-day directional outcome (35-42% accuracy on 3-class). Iron condors don't need direction — they need range. Switching the prediction target is higher-impact than any feature.

- [ ] **Range-prediction model** — `P(price stays within ±X% over Y days)`. Direct input for iron-condor sizing. Target: 65-75% accuracy.
- [ ] **Vol-magnitude model** — `P(realized vol > implied vol)`. Direct input for straddle sizing. Target: 60%+ accuracy.

---

## Tier 2 — Better Data Sources

- [ ] **Earnings whisper numbers** — Pre-earnings sentiment shifts (vs official consensus).
- [ ] **Insider trading (SEC Form 4)** — Insider buying/selling filings. Free via EDGAR.
- [ ] **Short interest changes** — Squeeze setups, FINRA bi-weekly data.
- [ ] **Historical IV data** — Properly backtest IV-rank features (currently proxied from realized vol).
- [ ] **Real options chain history** — Backtest with actual bid/ask not Black-Scholes.
- [ ] **NASDAQ Basic subscription ($1.50/mo)** — Eliminate delayed-data slippage on QQQ/AAPL/NVDA. Likely waived once commissions hit $20/mo.

---

## Tier 3 — Model Architecture

- [ ] **Per-strategy ML models** — Separate XGBoost/LightGBM per strategy (iron-condor vs straddle vs CSP). Each strategy needs different features.
- [ ] **Per-VIX-regime models** — Separate models for VIX <20, 20-30, >30. Markets behave fundamentally differently across regimes.
- [ ] **Online learning** — Continuously update model weights from each trade outcome. Currently only retrains weekly from scratch.
- [ ] **Switch 3-class to 2-class** — Drop "neutral", just predict up vs down. Easier problem, higher accuracy possible.

---

## Strategy Selection Improvements

- [ ] **Fix Thompson sampler** — Currently has 0 wins/losses recorded. Ensure outcomes flow back to update beta distributions.
- [ ] **Learn optimal IV-rank thresholds per strategy** — Iron condors at IV>50 historically much better than IV<30. Should be data-driven not hardcoded.
- [ ] **Time-of-day execution learning** — Track which entry hours actually fill best (avoid 9:30-9:45 chaos and 3:30-4:00 gamma).

---

## Risk Management

- [ ] **Tail-risk hedging** — Buy cheap OTM SPY puts (5% OTM, 30 DTE) as portfolio insurance against the -$2,158 worst-case trade. Cost: ~0.5% of account/month.
- [ ] **Per-strategy max-loss caps** — Separate budgets so a single straddle blowup can't wipe iron condor profits.
- [ ] **Position rolling on losers** — Roll credit spreads forward in time when tested side reaches 25 delta. Research shows 86% win rate with adjustments.

---

## Execution

- [ ] **Smart combo pricing** — Walk price down progressively: start mid-$0.05, after 30s try mid-$0.10, then mid-$0.15, then market.
- [ ] **Avoid first 15 min and last 30 min** — Wide spreads at open, gamma risk at close. Add time-of-day filter to entry path.
- [ ] **Direct exchange routing** — Currently SMART. Could try SPY→ARCA, QQQ→NASDAQ direct routing on liquid names.

---

## Operational

- [ ] **Live P&L vs backtest comparison report** — Track gap weekly. Reveals slippage and missed fills.
- [ ] **Surface new metrics on dashboard** — Sortino, RAROC, capital utilization, expectancy.
- [ ] **Per-symbol position size limits** — Prevent single-symbol blowup from killing the account.
- [ ] **Define "deploy live" criteria** — e.g., "after 50 paper trades, win rate >50%, no >$3k single-day loss".
- [ ] **Define kill criterion** — e.g., "if 30-day P&L < -5%, halt trading, force retrain".

---

## Discarded / Low-ROI (do not implement)

- ❌ **Twitter/X firehose** — Too noisy, too slow vs co-located pros. $5k/mo for marginal edge.
- ❌ **Truth Social** — Only relevant during specific political events. Not worth maintaining a scraper.
- ❌ **Stocktwits** — Same issues as Twitter, smaller scale.
- ❌ **Bloomberg Terminal** — $24k/yr. Pro tool — not worth it for a retail bot.
- ❌ **More technical indicator features** — Already have 68. Adding more degrades, not improves accuracy.
- ❌ **Reinforcement learning** — Sample-efficiency too poor for trading data. Backtest-first won't transfer.
- ❌ **Survivorship-bias correction** — Only matters if expanding to small-cap/speculative names. Current 7-symbol universe is all liquid ETFs/megacaps that have not been delisted.

---

## Gold-Standard Engineering Wishlist

Things that would push us from silver/bronze to gold across ML and engineering practices. Roughly ordered by ROI.

### 🟢 Quick Wins (1-3 hours each)

- [ ] **Pin random seeds** — Set `random_state=42` in XGBoost/LightGBM/StandardScaler + `np.random.seed(42)` in feature engine. Makes runs reproducible. Reproducibility 🥉 → 🥇.
- [ ] **Feature importance tracking** — Log `model.feature_importances_` per fold per symbol. Reveals dead-weight features. Feature importance tracking 🥉 → 🥇.
- [ ] **Bid-ask aware slippage in backtest** — Replace flat 1% with bid-ask-spread × contracts model. Uses existing IV data. Slippage 🥈 → 🥇.

### 🟡 Medium Investment (4-8 hours each)

- [ ] **Hyperparameter tuning (Optuna)** — Bayesian search per window for `max_depth`, `learning_rate`, `n_estimators`, `subsample`. May push accuracy 3-5%. RISK: overfitting if nested CV not done correctly. Hyperparameter tuning 🥉 → 🥇.
- [ ] **Model calibration** — Apply isotonic regression / `CalibratedClassifierCV`. Makes "0.65 confidence threshold" actually mean 65% probability. Critical now that we threshold on probability. Model calibration 🥉 → 🥇.
- [ ] **Per-VIX-regime models** — 3 models (low/mid/high VIX) per symbol. Routes prediction to regime-specific model. Specializes per market dynamics. Regime detection 🥈 → 🥇.

### 🔴 Larger Investment (1-2 days each)

- [ ] **True holdout set** — Reserve final 6 months as never-touched. Walk-forward on first 4.5y. Final OOS evaluation on holdout. Out-of-sample evaluation 🥈 → 🥇.
- [ ] **Statistical drift detection** — KS tests on input distributions, PSI scores. Detects concept drift before accuracy degrades. Drift detection 🥈 → 🥇.
- [ ] **Real options chain history** — $79/mo Polygon options addon. Replace Black-Scholes simulation with actual bid/ask. Single biggest backtest realism upgrade. Backtest realism 🥉 → 🥇. (Already in Tier 2 above.)

### Engineering practices (less urgent but tracked)

- [ ] **CI/CD pipeline** — Auto-run tests on PR, pre-commit hooks. CI/CD 🥉 → 🥇.
- [ ] **Test coverage > 80%** — Currently sparse. Add unit tests for risk manager, executor, range predictor. Test coverage 🥉 → 🥇.
- [ ] **Pinned dependency lockfile** — `pip-compile` to freeze exact versions. Prevents "works on my machine" drift. Dependency management 🥈 → 🥇.
- [ ] **Secrets vault** — Move from `.env` plaintext to AWS Secrets Manager / 1Password CLI. Secrets management 🥉 → 🥇.
- [ ] **SLO-based monitoring** — Define p99 latency / uptime SLOs, alert on breach. Monitoring 🥈 → 🥇.
- [ ] **State recovery improvements** — Idempotent retries on every external call (IBKR, Polygon, Telegram). Idempotency 🥈 → 🥇.

### What gold-standard looks like at the end

When all of the above ship, AIT v2 becomes a **research-reproducible, production-grade quant platform** — same caliber as professional firms minus the multi-million-dollar data feeds (Bloomberg/CRSP/etc.).

---

## Last Updated

2026-04-28

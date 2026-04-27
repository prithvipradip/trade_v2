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
- ❌ **More technical indicator features** — Already have 60. Adding more degrades, not improves accuracy.
- ❌ **Reinforcement learning** — Sample-efficiency too poor for trading data. Backtest-first won't transfer.

---

## Last Updated

2026-04-26

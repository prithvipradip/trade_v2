# Glossary

One definition per term, used by every other document. Not for: how a rule works ([GUIDE.md](GUIDE.md)), how a model works ([MODELS.md](MODELS.md)), or any current number (those live in `config.yaml` and the generated tables).
See also: [README.md](../README.md) (document map).

**Contents:** [Options](#options) · [Trade structure and risk](#trade-structure-and-risk) · [Orders and execution](#orders-and-execution) · [Exits](#exits) · [Models and data](#models-and-data) · [Operations](#operations) · [Research](#research)

## Options

| Term | Meaning |
|---|---|
| **Option, call, put** | A contract giving the right (not the obligation) to buy (call) or sell (put) the underlying at the strike price. The seller of the option takes the obligation and receives the premium |
| **Underlying** | The asset the option is written on; here a liquid ETF such as SPY, QQQ or IWM |
| **Strike** | The price in the contract at which the underlying would be bought or sold |
| **Expiry, DTE** | The date the option ends; DTE is the number of calendar days remaining until it |
| **Premium, credit** | Money received for selling an option (or a combination of options); a **credit structure** is opened for a net credit |
| **Mid, bid, ask** | The best price a buyer will pay (bid), the best a seller will accept (ask), and their average (mid) |
| **NBBO** | National best bid and offer: the best quotes across exchanges. For a multi-leg order the **combo NBBO** is the best bid and ask of the combination as one price |
| **Delta** | How much an option's price moves per $1 move in the underlying; roughly the market's probability that it finishes in the money |
| **Implied volatility (IV)** | The volatility the market price of an option implies. High IV means expensive options |
| **IV rank** | The percentage of the last 252 stored daily implied-volatility readings that are below today's (0 to 100); unknown when fewer than 60 readings exist. The system's single definition is `ait.data.iv_rank` |
| **VIX** | The market's implied-volatility index for the S&P 500; used as a measure of market stress |
| **Assignment** | The option buyer exercises, and the seller must deliver or take the stock. A short option can be assigned early, most often around dividends |
| **Ex-dividend** | The date from which a buyer of the stock no longer receives the next dividend; short calls near it carry assignment risk |

## Trade structure and risk

| Term | Meaning |
|---|---|
| **Iron condor** | Four options: a short put and a long put below the price, a short call and a long call above it. It earns the credit if the underlying stays between the two short strikes until expiry, and its loss is limited to the **wing width** minus the credit |
| **Wing, wing width** | The distance between the short and long strike on one side of a condor; it caps the loss on that side |
| **Defined risk / undefined risk** | A defined-risk structure has a worst-case loss known at entry (a condor, a spread). Undefined risk (a short strangle) does not; the executor refuses it unless explicitly allowed |
| **Capital at risk** | The worst-case loss of a position (for a condor: wing width minus credit, times contracts, times 100). **Aggregate** capital at risk sums it across all open positions |
| **Net liquidation (NLV)** | The account's total value as reported by the broker; the denominator of most risk percentages |
| **Concentration** | The share of risk in one symbol or one direction; capped by the risk manager |
| **Circuit breaker** | A persistent halt on new entries after a daily loss, consecutive losses or repeated API failures |
| **PDT (pattern day trader)** | A regulatory rule limiting same-day round trips for margin accounts under $25,000; the PDT guard can hold back an exit that would be an illegal day trade |
| **Gap stress** | The loss the open book would show if every underlying were 10% lower, valued at intrinsic value; printed in the end-of-day reconcile message |

## Orders and execution

| Term | Meaning |
|---|---|
| **Combo order** | One order that trades all legs of a structure together at a net price, so the structure cannot be left half-filled |
| **Reprice ladder** | An unfilled entry limit starts near mid and steps toward the marketable side at fixed intervals, never past the credit floor that justified the trade |
| **Marketable** | A limit price at which the order would fill immediately |
| **Exit outbox** | A queue of closes that were filled but could not yet be booked; it is drained every cycle so each close is booked exactly once |
| **Reconciler, reconcile** | The process that compares the bot's books with the broker's positions and executions, and treats the broker as the truth |
| **Untracked position** | A position at the broker that the bot has no record of (for example, filled during a crash); it freezes new entries until resolved |
| **HALT files** | Empty files in `data/` that freeze entries (`HALT`, `HALT_UNTRACKED`) or ban a symbol (`RESTRICTED.txt`); see [RUNBOOK.md](RUNBOOK.md) |
| **Read-only API** | A Gateway setting (and session state) in which every order is rejected |

## Exits

| Term | Meaning |
|---|---|
| **Touch stop** | The primary loss exit on a credit structure: close when the underlying's price reaches a short strike, rather than waiting for the option's price to rise |
| **Take-profit ladder** | A profit target that shrinks as expiry nears, shared by live trading and research |
| **Time exit** | Closing when the position is at or below a set number of days to expiry, or has been held for a set number of days |
| **Trailing stop, high-water mark (HWM)** | A stop that follows the best profit reached by a position; the HWM is that best value |
| **Flatten** | Close a position (or all of a symbol's positions) at once, for example after a material SEC filing |

## Models and data

| Term | Meaning |
|---|---|
| **Direction ensemble** | An XGBoost plus LightGBM pair that predicts up, down or flat over the label horizon ([MODELS.md](MODELS.md)) |
| **Range model** | A model that predicts whether the underlying stays inside a band; the model that matches a condor's payoff |
| **Vol-magnitude model** | A model that estimates the probability of a large move; used only by long-volatility structures, which are not enabled |
| **Meta-labeler** | A second model, trained on the bot's own closed trades, that asks whether a specific trade should be taken |
| **Observe / enforce** | A gate in *observe* mode logs what it would have done but never blocks; in *enforce* mode it blocks. See [MODELS.md](MODELS.md) "Gating: observe and enforce" |
| **Fail closed** | When a gate lacks the information it needs, it refuses the entry (and says so) instead of letting it through. Exits are never blocked this way |
| **Fractal gate, Hurst exponent** | A regime filter built from the Hurst exponent (a measure of whether a series trends or mean-reverts) over several scales; one shared definition in `ait.ml.regime_gate` |
| **Regime classifier** | A rule-based labeller that puts the market in one of seven regimes (trending up, trending down, range bound, high volatility, low volatility, IV crush, mean reverting) |
| **Drift detector** | A check on whether the direction model's live track record is getting worse; off by default |
| **Feature, feature frame, feature spec** | A feature is one model input column; the frame is all of them for a symbol and date; the spec is the exact list and window lengths, hashed into every saved model so a mismatch is detected |
| **Purge gap** | The rows left out between training and validation data, at least as long as the label horizon, so overlapping labels cannot leak into the score |
| **Walk-forward** | Validation in which every test window lies after its training data, and the window rolls forward through history |
| **AUC** | Area under the ROC curve: 0.5 is no skill, 1.0 is perfect. Used to report a classifier's skill |
| **VLMC** | The code's name for the 26 intraday features (session VWAP position, high/low timing, volume shape, power-hour and closing measures, intraday fractal measures) computed from five-minute bars when they exist |
| **OHLCV** | Open, high, low, close and volume bars |

## Operations

| Term | Meaning |
|---|---|
| **Supervisor** | The master process (`run_orchestrator.py`) that starts and restarts the bot, runs the dashboards and the scheduled jobs |
| **Bot** | The trading process (`python -m ait.main`) |
| **RTH** | Regular trading hours (09:30 to 16:00 Eastern) |
| **ET** | US Eastern time; the only time zone used for scheduling and trade timestamps |
| **Paper / live** | Paper trades against a simulated account at the broker; live trades real money |
| **Go-live gates, scorecard** | The criteria a paper track record must pass before live trading, graded by one function in `reporting/go_live.py` and shown on the status page |
| **Heartbeat** | The file `data/bot_heartbeat`, touched on every protection tick; the supervisor restarts the bot when it goes stale during market hours |
| **Watchdog** | The bot's own monitor of three components (`trading_loop`, `ibkr`, `market_data`), which tries to recover one that goes down and alerts if it cannot |
| **Smoke test** | `scripts/smoke_deploy.py`: a read-only check that the real configuration and database load and the main paths run |

## Research

| Term | Meaning |
|---|---|
| **Backtest** | A replay of the rules over history under stated cost and fill assumptions; not a forecast |
| **Study** | A committed YAML file that changes some settings for one research run; the outcome is recorded in [STUDIES_FINDINGS.md](STUDIES_FINDINGS.md) |
| **Run record** | The file written beside every research run: settings used, code version, data fingerprints, seeds and command line |
| **Overlay** | `config_research.yaml`: the research-only settings and the differences from `config.yaml`, merged over it by research tools only |
| **Parity** | The property that research uses the same logic as live; every deliberate difference is printed in a manifest at the start of a run |
| **Friction** | The per-leg dollar cost of trading that the backtest charges: a fitted half-spread plus commission and fees |
| **Optuna trial** | One evaluation of one candidate parameter set during a search |

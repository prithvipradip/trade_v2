# AIT v2 — System Guide

How the system is built, how a trading day runs, and why it is designed this way. Short answers first; the other documents are linked
where you need the detail. Not for: step-by-step procedures ([RUNBOOK.md](RUNBOOK.md)), model internals ([MODELS.md](MODELS.md)), research how-tos ([RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)) or code changes ([TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md)).
See also: [GETTING_STARTED.md](GETTING_STARTED.md) · [GLOSSARY.md](GLOSSARY.md) · [README.md](../README.md) (document map).

> **Read this first — values in this guide are examples.** Configuration changes often. Every number below (a threshold, a cadence, a
> count, a time of day) is shown to explain how a rule works, not as the current setting. **The current values are in the latest
> [`config.yaml`](../config.yaml) (live and shared settings) and [`config_research.yaml`](../config_research.yaml) (research-only settings).**
> The *allowed range* of a setting is in [`src/ait/config/settings.py`](../src/ait/config/settings.py), and the few values that are
> deliberately not configurable are in `src/ait/config/invariants.py` and `invariants_research.py`. When this guide and a config file
> disagree, the config file is right.

**Contents:** [1 What it is](#1-what-ait-v2-is) · [2 Architecture and processes](#2-architecture-and-processes) · [3 The trading day](#3-the-trading-day) ·
[4 Entry pipeline](#4-entry-pipeline) · [5 Strategies](#5-strategies) · [6 Machine learning](#6-machine-learning) · [7 Risk management](#7-risk-management) ·
[8 Order execution and reconciliation](#8-order-execution-and-reconciliation) · [9 Position management and exits](#9-position-management-and-exits) ·
[10 Learning and adaptation](#10-learning-and-adaptation) · [11 Data and storage](#11-data-and-storage) · [12 Monitoring and operator surfaces](#12-monitoring-and-operator-surfaces) ·
[13 Configuration system](#13-configuration-system) · [14 Research and optimization](#14-research-and-optimization) · [15 Technology and repository layout](#15-technology-and-repository-layout) ·
[16 Design principles](#16-design-principles) · [17 Extending the system](#17-extending-the-system)

---

## 0. Which document answers what

The full map, with every document's reader and job, is in [README.md](../README.md). In short:

| You want to... | Read |
|---|---|
| Install and run it for the first time | [GETTING_STARTED.md](GETTING_STARTED.md) |
| Understand how the system works and why (this document) | `docs/GUIDE.md` |
| Respond to an alert, stop the bot, deploy, run a drill | [RUNBOOK.md](RUNBOOK.md) |
| Understand a model or feature, or change one safely | [MODELS.md](MODELS.md) |
| Run, trust and read a backtest, optimizer or study | [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) |
| See what past studies found | [STUDIES_FINDINGS.md](STUDIES_FINDINGS.md) |
| Read the Walk-Forward Lab dashboard | [WALKFORWARD_DASHBOARD.md](WALKFORWARD_DASHBOARD.md) |
| Change the code, or see which changes need the operator's confirmation | [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) |
| Look up a term, or a setting | [GLOSSARY.md](GLOSSARY.md), [reference/SETTINGS_LIVE.md](reference/SETTINGS_LIVE.md) |
| Find a script: what it does, whether it needs IB, what it writes, when to run it | [reference/SCRIPTS.md](reference/SCRIPTS.md) |
| Find out why something was decided, or what is planned | `PLAN.md` (the decision record; append-only) |

This guide does not repeat procedures (RUNBOOK), model internals (MODELS) or research how-tos (RESEARCH_GUIDE). Where a topic lives there, it says so and links. Terms are defined in [GLOSSARY.md](GLOSSARY.md).

---

## 1. What AIT v2 is

**What is it?** An autonomous options-income bot. It connects to Interactive Brokers (IB Gateway or TWS), scans a small list of liquid
ETFs, and sells **defined-risk premium**: iron condors, which profit if the underlying stays between two strikes until expiry. Machine-learning
models, a risk manager and a set of fail-closed gates decide whether to enter; a fast monitor manages exits.

**What does it trade today?** Whatever `trading.universe` and `options.strategies` say. At the time of writing that is three index ETFs
(for example SPY, QQQ, IWM) and **iron condors only**. Other structures (debit spreads, short strangles, long options, long straddles) still exist in
`src/ait/strategies/` but are not enabled; the live selector emits iron condors.

**Paper or live?** `trading.mode` in `config.yaml` (`paper` or `live`). The account is paper while the bot builds a track record against
the go-live criteria in `PLAN.md`. Live needs `config.yaml` and `.env` to both say live, the environment interlock `AIT_ALLOW_LIVE=1`, and a live-account IB endpoint; the supervisor and the bot refuse to start otherwise, and refuse a paper bot pointed at a live endpoint (`config/launch_mode.py`), because the IB endpoint, not the label, decides which account trades. Going live is a deliberate
multi-part operator step (those switches, the data entitlement, the paper-only learning switch and the other items in the checklist). The list is in [RUNBOOK.md](RUNBOOK.md) "Go-live day checklist".

**What is it not?** It is not a high-frequency system (the entry scan runs every few minutes), not a directional trader, and not
self-modifying beyond small bounded adaptations ([section 10](#10-learning-and-adaptation)). Everything that manages money is in configuration or in a small
set of visible fixed limits.

---

## 2. Architecture and processes

**What runs?** Two long-lived processes plus a few web services, all started and watched by a supervisor.

```
┌────────────────────────────────────────────────────────────────────────┐
│ run_orchestrator.py  →  src/ait/orchestration/master.py   (SUPERVISOR) │
│                                                                        │
│  BotManager         starts / health-checks / restarts the bot          │
│  WebServiceManager  status page :8503 · analytics :8501 · logs :8502   │
│  BlockingScheduler  retrain, digests, report, backup, shared logs ...  │
│  Gateway recovery   restarts a wedged IB Gateway, with a budget        │
└───────────────┬────────────────────────────────────────────────────────┘
                │ spawns  python -m ait.main   (--paper unless live is fully enabled)
                ▼
┌────────────────────────────────────────────────────────────────────────┐
│ src/ait/main.py → bot/orchestrator.py  (TradingOrchestrator, THE BOT)  │
│                                                                        │
│  MarketScheduler  PRE_MARKET / MARKET_OPEN / POST_MARKET / OFF_HOURS   │
│  data  ml  strategies  risk  execution  learning  monitoring  state    │
│                              │                                         │
│                      ib_insync (one session)                           │
└──────────────────────────────┼─────────────────────────────────────────┘
                               ▼
                IB Gateway / TWS   (paper 4002 / 7497, live 4001 / 7496)
```

**Why two processes?** A trading bot must survive its own bugs. The supervisor is deliberately light (its time and path helpers use only the
standard library, so a broken scientific stack cannot take it down), restarts the bot when it exits or hangs, and keeps the scheduled jobs running whatever the
bot does. Add background work to the supervisor's scheduler, not to the bot's loop.

**What does the supervisor do for the bot?**
- **Single instance.** An OS-level lock (`data/orchestrator.lock`) stops a second supervisor, and the supervisor refuses to start a bot when it finds another
  `ait.main` process (two bots on one account is the duplicate-order failure class).
- **Restarts.** If the bot exits, it restarts it with a budget (a handful of attempts; the budget refills after a stretch of healthy running) and pages
  when it gives up.
- **Hang detection.** The bot touches `data/bot_heartbeat` on every protection tick while its loop is healthy (an ongoing error streak deliberately lets it go stale). A heartbeat
  that stays stale during market hours means a hung or failing loop (exits not running), so the supervisor restarts the bot and alerts.
- **Gateway.** If the IB port is closed it defers bot starts without spending the restart budget (IB restarts nightly), and after a grace period tries to
  bring the Gateway back itself (kill a wedged process, relaunch via IBC), with a cooldown and a budget before it escalates to a human.
- **Fresh models.** After a successful retrain it drops `models/.retrained`; the bot is restarted once, never during market hours or the post-market window, to
  load the new files.

**How does the bot connect to IB?** Through [`broker/ibkr_client.py`](../src/ait/broker/ibkr_client.py), the only code that talks to IB. It owns the
connection lifecycle, reconnects, and a client-id fallback (if the configured id is busy it tries others) with a record of orders that belong to *another*
session. The configured port may name TWS while Gateway is what is up (or the reverse): the bot and supervisor use the other endpoint of the *same account type*
and keep the mismatch visible on the status page. Other jobs use their own client ids (the retrain, the short-DTE collector, backfills) so they never collide
with the bot.

**Where does it run?** The reference trading box is Windows (`start_bot.bat`, `keeper_ait.bat` auto-relaunch loop); development and some deploys happen on
macOS (use `nohup`; `setsid` does not exist there). The RUNBOOK uses PowerShell examples for that reason.

### Component map

| Package | What it owns |
|---|---|
| `orchestration/` | The supervisor (`master.py`), IB Gateway/IBC control (`gateway.py`), the retrain's own IB session (`retrain_ib.py`), shared logs, the short-DTE collection job |
| `bot/` | The trading orchestrator (the loop and its phases), the market-phase scheduler, SQLite state (`state.py`) |
| `broker/` | IB connection, account snapshot, order and contract builders |
| `data/` | Market data and option chains, the history store, IV rank and IV refresh, calendars (earnings, dividends, economic, EDGAR 8-K), data-quality checks, IB port/endpoint helpers |
| `ml/` | Features and the feature spec, direction ensemble, range and vol-magnitude models, meta-labeler, regime classifier, drift detector, trainer |
| `strategies/` | Strategy classes and the selector, shared signal types (`base.py`), Thompson sampling |
| `risk/` | Risk manager, position sizer, circuit breaker, PDT guard, correlation guard, capital tiers |
| `execution/` | Executor (orders and fills), portfolio manager (exit rules), exit policy, reconciler, manual positions, condor defense and roll mechanics |
| `learning/` | Trade analyzer, strategy adaptor, learning engine, counterfactual log |
| `monitoring/` | Watchdog, ops-health authority, analytics (DuckDB and SQLite) |
| `reporting/` | The go-live scorecard authority (`go_live.py`) |
| `notifications/` | Telegram |
| `config/` | Pydantic settings, the runtime environment contract, fixed invariants, the research overlay |
| `backtesting/`, `optimization/` | The offline research engine ([section 14](#14-research-and-optimization)) |
| `dashboard/` | Streamlit analytics app and the Walk-Forward Lab |
| `utils/` | ET-aware clocks, cadence constants, logging, process helpers |

Every module, with a one-line description taken from its docstring, is in the generated [reference/MODULES.md](reference/MODULES.md).

Retired subsystems (the news-sentiment stack, options-flow analysis, hedging, the GARCH ensemble members, event straddles) live under `deprecated/` and are not imported by the live tree.

---

## 3. The trading day

**What phases are there?** The bot's [`MarketScheduler`](../src/ait/bot/scheduler.py) is a state machine driven by the NYSE calendar in ET:

```
OFF_HOURS ──► PRE_MARKET ──► MARKET_OPEN ──► POST_MARKET ──► OFF_HOURS
              (before open)   (RTH)           (just after the close)
```

| Phase | What the bot does |
|---|---|
| Pre-market | Reconcile the book against IB, reset daily counters, run the learning cycle, refresh the correlation guard's price data, make sure models are loaded, train the meta-labeler when it is enabled and enough trades exist, read VIX and the regime, decay the Thompson-sampling statistics |
| Market open | Run the two concurrent tasks below |
| Post-market | Final reconcile and the end-of-day reconciliation report, learning cycle, daily summary, pre-stamp tomorrow's mark-to-market baseline, housekeeping |
| Off hours | Wait for the next pre-market |

**What happens at startup?** Order matters, because the book may hold positions the bot must keep protecting:
1. Model loading/training is started in the **background**, so it never delays step 2 (a slow boot used to leave the book without exits).
2. The bot **reconciles with IB first**, re-adopting any working exit orders it finds, then announces itself on Telegram.
3. It checks the API session can really place orders (a "read-only API" session silently rejects every order).
4. It enters the phase loop above.

**What runs during market hours?** Two **independent** asyncio tasks, so a slow scan can never delay position protection:

| Task | Cadence | Does |
|---|---|---|
| Protection | every `FAST_MONITOR_CADENCE_SECONDS` (30 s, `utils/cadence.py`), unconditionally | Evaluate every open position against the exit rules and act, check order fills and book completed exits, run the mark-to-market daily-loss brake, poll SEC 8-K filings for material events (about every 5 minutes), re-probe the read-only API session, write the heartbeat, run the watchdog |
| Entry scan | every `trading.scan_interval_seconds` (e.g. 300 s) after the previous scan *finishes* | The full cycle: circuit breaker, reconcile-if-owed, position check, then the per-symbol entry pipeline ([section 4](#4-entry-pipeline)) |

A shared position lock stops the two from evaluating positions at the same instant. New entries stop `trading.no_new_entries_minutes_before_close` before the close.

**What does the supervisor schedule?** The jobs below are registered in `master.py::_register_recurring_jobs`; the table is generated from that function, so it is always the real schedule.

<!-- BEGIN GENERATED: supervisor_jobs -->
<!-- generated by scripts/gen_docs.py from ait/orchestration/master.py::_register_recurring_jobs; edit the code, not this table -->
| Job | Id | When (ET) | What it does |
|---|---|---|---|
| Health check | `health_check` | every 2 min | Checks the bot and the web services, and probes whether the Telegram channel is dead |
| Model retrain | `daily_retrain` | Mon-Fri 07:30 ET | Daily bars from IB (own session), train, write models, drop the `.retrained` marker. Aborts before any write if IB is unavailable |
| State backup | `db_backup` | Mon-Fri 17:00 ET | Hash-verified copy of `data/ait_state.db`, with an off-box mirror |
| Weekly scorecard | `weekly_scorecard` | Friday 16:10 ET | Go-live criteria progress |
| Morning digest | `digest_am` | Mon-Fri 09:35 ET | Operator summary; the absence of the morning digest is itself an alarm |
| Evening digest | `digest_pm` | Mon-Fri 16:05 ET | Operator summary after the close |
| Daily report | `daily_report` | Mon-Fri 16:30 ET | P&L and performance summary |
| Shared logs (publish) | `shared_logs_publish` | Mon-Fri 16:45 ET | Publishes this box's bot events to the `shared-logs` branch |
| Shared logs (fetch) | `shared_logs_fetch` | Mon-Fri 16:50 ET | Fetches the other box's published events (fetch only; never touches the checkout) |
| Weekly backtest | `weekly_backtest` | Sunday 20:00 ET | Walk-forward run that tracks strategy health |
| Log cleanup | `monthly_cleanup` | day 1 of the month 00:00 ET | Removes old log and temporary files |
| Log rotation | `daily_log_rotation` | daily 00:05 ET | Size cap on the append-only logs |
| Short-DTE friction collection | `short_dte_collection` | optional, weekdays, time from `data_collection.short_dte_time_et` | Research data only; registered only when `data_collection.short_dte_enabled` is true; never touches orders |
<!-- END GENERATED: supervisor_jobs -->

The jobs that matter most have a **catch-up**: a box that was off at the slot runs the missed backup, retrain, digest, daily report or weekly backtest shortly after it starts (the backtest is
never run during market hours). Details and procedures: [RUNBOOK.md](RUNBOOK.md) "Daily 2-minute routine".

---

## 4. Entry pipeline

**What decides whether a trade is opened?** A chain of checks, in order, for every symbol on every scan. Almost every check **fails closed**: if the
information needed to judge is missing, the entry is refused (and, for outages, an operator page is sent). Open positions are never affected by an entry
refusal; exits always keep running.

### 4.1 Per-symbol scan (`_scan_symbol`)

1. **Fetch** ~2 years of daily history and the symbol's **IV rank** concurrently. No IV rank means no entry on that symbol (the old behaviour of substituting a
   neutral 50 hid outages).
2. **Intraday bars** are fetched incrementally into the history store, so the intraday (VLMC) features see every session.
3. **Data-quality** validation of the price series.
4. **Direction model** prediction (XGBoost + LightGBM, [MODELS.md](MODELS.md)). Low directional confidence does *not* veto a condor: it only puts the scan in "neutral-only" mode,
   because a range-bound market is a condor's ideal case.
5. **Regime** classification from the price history and VIX; a strongly high-volatility regime forces the direction to neutral.
6. **Multi-timeframe** adjustment of confidence, then the daily **feature frame** is computed once and reused (fractal gate, meta-labeler, entry snapshot).
7. **Fractal regime gate**: a shared rule that penalises (and optionally vetoes) chaotic, multi-scale price behaviour.
8. **Earnings proximity**: refused near earnings, and refused fail-closed when the earnings data is unavailable.
9. **Meta-labeler** (a second model that asks "should this specific trade be taken?"): acts only when it is enabled, trained, outside paper-trading mode, and the real-close count has reached
   the trust threshold. It is disabled today.
10. **Option chain** fetch, liquidity filtering, and persistence of today's ATM IV (this keeps the IV history fresh as a side effect).
11. **Strategy selection**: every enabled strategy builds its signals from the chain; the learning adaptor can have disabled some.
12. **Range model** (P(price stays inside the band over the hold)). In *enforce* mode it replaces the signal's confidence and a floor applies; in *observe* mode it is
    only logged and stored on the trade ([4.3](#43-observe-vs-enforce)).
13. **Ranking**: signals are ordered (Thompson re-ranking when not in paper mode, defined-risk before undefined-risk), and the top few are collected as candidates.

After all symbols are scanned, candidates are **ranked across the whole universe** (confidence plus a credit-to-width bonus for credit structures) and
executed best-first, so scarce position slots go to the best trade rather than to whichever symbol is first in the config.

### 4.2 Pre-execution checks

Each candidate then passes, in order:

- **Halt files**: `data/HALT` (operator freeze) and `data/HALT_UNTRACKED` (set automatically when a broker position has no local record). Either blocks new entries.
- **Restricted list** (`data/RESTRICTED.txt`): a hard symbol ban; an unreadable file blocks all new entries rather than meaning "no bans".
- **Pre-event blackout**: no new credit entries within N *trading days* of a major macro event (FOMC, CPI, NFP...). If the calendar check itself fails, credit entries are refused and an
  operator page is sent after repeated failures.
- **Post-stop cooldown**: a symbol that was just stopped out (and, for a shorter time, its correlated siblings) cannot be re-entered at once.
- **Trade budget**: a time-of-day cap on how many trades may be opened (a small budget in the first hour, more by midday, the remainder after), bounded by `trading.max_daily_trades`.
- **Settling period**: no entries in the first minutes after a (re)start.
- **First-hour confidence** for directional signals only (neutral structures are exempt).
- **Duplicate-order guards**: three layers (a database cooldown on the same symbol and strategy, working orders at the broker, and the executor's own pending-order tracker) that each refuse when they find a duplicate *or cannot check*.
- **Through-expiry blocks**: an entry whose life would span an earnings date, or (for a short call) an ex-dividend date, is refused; unknown data refuses fail-closed.

Then the **risk manager** validates the trade ([section 7](#7-risk-management)); the learning multiplier may scale the size, in which case the larger trade is **re-validated** at
its real size; and finally the **executor** places it ([section 8](#8-order-execution-and-reconciliation)). On success the bot stores the entry context (regime, VIX, IV rank,
range verdict, feature snapshot) with the trade, sends a Telegram message and updates the daily stats. Every rejection is recorded in the counterfactual log.

### 4.3 Observe vs enforce

`ml.entry_gates_enabled` is the master switch for whether the models can **veto** a live entry. The shipped mode is *observe*: the models predict and log, and the range verdict is stored on each trade, but nothing is blocked. Flipping the
setting is an operator decision after a pre-registered test. What each model does in each mode, and why observe is the shipped mode, is in [MODELS.md](MODELS.md) "Gating: observe and enforce". The research engine enforces the gates on
purpose; the research view of this same chain, with the counters a backtest prints, is the entry funnel in [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md).

---

## 5. Strategies

**What is a strategy?** A class in `src/ait/strategies/` that, given an option chain and market context, may emit a `Signal` (legs, credit or debit, max loss,
confidence). The [`StrategySelector`](../src/ait/strategies/selector.py) asks every enabled strategy and collects the results. Shared definitions live in
[`strategies/base.py`](../src/ait/strategies/base.py): `CREDIT_STRATEGIES` (structures that receive a credit; the union of the live ones and the backtest-only arms) and
`UNDEFINED_RISK_STRATEGIES` (structures whose loss is not capped by wings). Import them; do not re-type the lists.

**How is an iron condor built?** [`IronCondor`](../src/ait/strategies/iron_condor.py) sells an out-of-the-money put spread and call spread together:

- **Gate on IV rank**: no condor below an IV-rank floor (example: 15); the risk manager separately refuses to sell premium above a cap (example: 85), because very rich vol is rich for a reason.
- **Short strikes** near a target delta (example: about 0.20) inside an allowed delta band, and inside an allowed range of expected-move multiples. A feed that returns no usable
  greeks produces no signal.
- **Wing width** is volatility-scaled (`wing_k × price × IV × √(DTE/365)`, with a minimum), then **capped by the per-trade risk budget**, then snapped to the listed strike grid.
- **Economics floors**: an absolute minimum credit and a minimum credit-to-width ratio, so a trade must pay for its costs.
- **Max loss** = widest wing minus credit, per contract.

The values behind these examples are the `backtest.ic_*` / `backtest.wing_*` settings and their environment overrides (see [section 13](#13-configuration-system)).

**Which other strategies exist?** The selector's registry (`STRATEGY_MAP`) also knows bull call and bear put spreads (`spreads.py`), long calls and puts (`long_options.py`), and long straddles and short
strangles (`straddles.py`); none is enabled. Covered calls, cash-secured puts, calendars and event straddles were retired to `deprecated/` (the selector keeps empty stubs for the calendar and event-straddle hooks). Naked
(undefined-risk) shapes such as the short strangle are refused at the executor unless an explicit environment interlock is set ([section 8](#8-order-execution-and-reconciliation)). The research engine has more builders than live does; see [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md).

**What is Thompson sampling doing?** Each strategy has a Beta distribution over its win probability, updated after every close; ranking samples from them to balance trying
and favouring strategies. It is **bypassed in paper mode** (`learning.paper_trading_mode`) so that paper results measure one fixed rule set, and with a single enabled strategy it has nothing to choose
between. State: `data/thompson_state.json`.

---

## 6. Machine learning

**What do the models do in the live system?** Several small models, each answering one question: a direction ensemble (up, down or flat), a range model (will the price stay inside the band the condor sells), a vol-magnitude model (big move), an
optional meta-labeler (should this specific trade be taken), a rule-based regime classifier and an optional drift detector. A strategy is gated by the model that matches its **payoff**: a condor is paid for the price staying put, so it is
gated by the range model, not by direction. In the shipped *observe* mode the models predict and log on every scan but never veto an entry.

**How are they kept fresh?** A supervisor job retrains them every trading morning on its own IB session; the bot loads the new files after a controlled restart ([section 3](#3-the-trading-day)). Models carry a record of the settings they were trained
with and are refused or flagged when those change.

Everything else (features and their switches and windows, labels and validation, hyperparameters, saved models, each model in turn, the regime classifier, the drift detector, how to change a model setting) is in [MODELS.md](MODELS.md).

---

## 7. Risk management

**How is risk controlled?** In layers, from "never trade this" to "size it small". Every layer is independent of the ML models.

### 7.1 The risk manager

[`RiskManager.validate_trade`](../src/ait/risk/manager.py) runs cheap checks first and returns approve / reject plus a size. In order (example values in parentheses):

1. **Circuit breaker** tripped: reject everything.
2. **Confidence** below the minimum (for direction-neutral structures the configured neutral baseline is used).
3. **Weekend-gap rule**: certain undefined-risk shapes need higher confidence late on Friday.
4. **Short-premium guardrails** for credit structures: **VIX unavailable → reject** (fail closed); **VIX at or above a halt level** (e.g. 28) → reject;
   a **cap on simultaneous credit positions that tightens as VIX rises** (e.g. 6 / 4 / 2 by VIX band); an **IV-rank cap** (e.g. 85).
5. **Position count** and **duplicate** (same symbol and strategy) limits.
6. **Per-expiry cap**: no more than N positions sharing one expiry date.
7. **Correlation guard**: positions in correlated symbols count as one cluster (the ETF universe is one cluster).
8. **Buying power**: measured against the capital the position consumes (max loss for credit structures), not the credit received.
9. **Per-trade max loss** as a fraction of net liquidation; **aggregate capital at risk** across all open positions; **per-symbol concentration**.
10. **Portfolio delta** limit. The bot has no live greeks subscription, so this gate is largely inert; the aggregate-risk and concentration gates carry the load.
11. **Daily loss** check.
12. **Position sizing**, then the correlation haircut; a zero-contract verdict is a refusal.

### 7.2 Sizing

[`PositionSizer`](../src/ait/risk/position_sizer.py) starts from a per-position cap and multiplies adjustments, all of them settings under `position_sizing` and `positions`:
confidence, implied volatility (smoothly smaller in high-vol), IV rank (selling gets more room when IV is rich), a per-strategy risk fraction, recent losing days
(drawdown throttle), VIX tiers, and a correlation haircut. Capital tiers (`risk/capital_tiers.py`) choose which universe is affordable at a given account size. The size
of one trade is bounded again by the per-trade risk budget, which is also what the condor builder is handed to size the wings.

### 7.3 Brakes that act on the book

- **Circuit breaker** ([`risk/circuit_breaker.py`](../src/ait/risk/circuit_breaker.py)): trips on daily loss, consecutive losses or repeated API failures. It persists across restarts. It blocks
  *entries only*; exits keep running.
- **Daily mark-to-market brake**: realised plus unrealised loss against a start-of-day baseline can freeze entries for the day.
- **PDT guard** (`risk/pdt_guard.py`): relevant only for margin accounts under the pattern-day-trader threshold; its switches are config (`account.pdt_*`).
- **Operator switches**: `data/HALT`, `data/RESTRICTED.txt`, and the keeper pause file ([RUNBOOK.md](RUNBOOK.md) "Kill switches").

### 7.4 Fixed limits, and the ceilings of tunable settings

**Fixed limits (not configurable, on purpose).** A few backstops are fixed in code in one module, [`config/invariants.py`](../src/ait/config/invariants.py). Each is either a backstop that exists to catch the
system, or its configuration, being wrong; a rule imposed from outside; or the line a configured value is checked against. Putting them in `config.yaml` would let one edit loosen both a gate and its check
(the file is shared between machines through git and edited by people and AI sessions). They are logged at every start (`fixed_limits_in_effect`). To change one: edit the module, run
`python scripts/gen_docs.py`, and record the decision in `PLAN.md`.

<!-- BEGIN GENERATED: fixed_limits -->
<!-- generated by scripts/gen_docs.py from ait/config/invariants.py; edit the code, not this table -->
| Name | Value | Unit | What it does, and why it is fixed | Enforced in |
|---|---|---|---|---|
| `ORDER_RATE_LIMIT_MAX` | 8 | orders | Hard ceiling on orders placed within ORDER_RATE_LIMIT_WINDOW_S (entry and roll placement share one bucket); the next order is refused and logged CRITICAL. A backstop against a MALFUNCTION (a logic bug spamming placements). The daily budget bounds intents; this bounds a bug. If config could raise it, the bug it exists to stop would not be stopped. | execution/executor.py (entry placement and roll placement) |
| `ORDER_RATE_LIMIT_WINDOW_S` | 60 | seconds | The rolling window ORDER_RATE_LIMIT_MAX is counted over. Part of the same backstop as ORDER_RATE_LIMIT_MAX. | execution/executor.py |
| `NBBO_DISORDERLY_SPREAD_RATIO` | 1.0 | ratio | Reject a combo entry whose (ask - bid) is wider than this multiple of its own mid. A backstop against a bad quote (delayed data, crossed or locked market). Distinct from the configurable per-contract options.max_bid_ask_spread_pct filter. | execution/executor.py (combo entry NBBO check) |
| `NBBO_STALE_SIGNAL_DEVIATION` | 0.35 | fraction | Reject a combo entry whose signal price differs from the live mid by more than this fraction. A backstop against acting on a signal priced on a market that has since moved; loosening it lets an order go out at a price unrelated to the market. | execution/executor.py (combo entry NBBO check) |
| `NBBO_MID_FLOOR` | 0.05 | dollars | The two NBBO ratio tests apply only when \|mid\| exceeds this. Below it the ratios blow up and mean nothing; keeps the gates from firing on near-zero mids. | execution/executor.py (combo entry NBBO check) |
| `LIVE_MAX_SPREAD_PCT` | 0.15 | fraction | Maximum bid-ask spread (AIT_LIQ_MAX_SPREAD and options.max_bid_ask_spread_pct) allowed when the bot is started in a non-paper mode. It is the line config values are checked AGAINST. The paper config is deliberately relaxed (0.40); a floor in the same file would let one edit loosen both the setting and its check. | bot/orchestrator.py (GOV-5 live-profile assertion at startup) |
| `LIVE_MIN_VOLUME` | 10 | contracts | Minimum option volume (AIT_LIQ_MIN_VOLUME) allowed when started in a non-paper mode. Same reason as LIVE_MAX_SPREAD_PCT; docs/RUNBOOK.md "Go-live day checklist" lists both numbers. | bot/orchestrator.py (GOV-5 live-profile assertion at startup) |
| `IC_BUDGET_CAPPED_MIN_WING` | 1.0 | dollars | Narrowest wing a budget-capped iron condor may shrink to. Below the configurable wing floor (AIT_IC_WING_FLOOR / backtest.wing_floor_dollars) only when the per-trade budget cannot carry the floor. A physical limit, not a preference: $1 is the narrowest wing the listed-strike grid offers. Lowering it asks for strikes that do not exist; raising it defeats the budget cap on small accounts. Live and the research engine share it. | strategies/iron_condor.py (_vol_scaled_width) and backtesting/engine.py |
| `PDT_MAX_DAY_TRADES` | 3 | trades | Day trades the bot allows itself inside PDT_WINDOW_DAYS trading days. A regulatory rule (FINRA pattern-day-trader, margin accounts under $25,000 are flagged at 4 in 5 business days), not a preference. The SWITCHES are config: account.pdt_protection and account.pdt_account_under_25k. | risk/pdt_guard.py |
| `PDT_WINDOW_DAYS` | 5 | trading days | The rolling window PDT_MAX_DAY_TRADES is counted over. Part of the same regulatory rule. | risk/pdt_guard.py |
| `BROKER_SNAPSHOT_TTL_SECONDS` | 5.0 | seconds | How long the all-clients open-order snapshot is re-used inside and across check_fills passes. A cache lifetime that avoids repeated broker round trips within one pass; an implementation detail of API-call pacing, not a trading preference. Passes are ~30 s apart so it never hides a change. | execution/executor.py (_broker_open_order_ids) |
<!-- END GENERATED: fixed_limits -->

**Ceilings of tunable settings.** These live in `config.yaml` and are tunable, but the **range** each may take is part of the schema in [`config/settings.py`](../src/ait/config/settings.py), so a typo or a bad edit fails at
startup instead of trading. The ranges sit in code on purpose (a limit in the same file as the value lets one edit loosen both). To widen one: edit `settings.py`, run `python scripts/gen_docs.py`, and record the decision
in `PLAN.md`. A change takes effect at the next bot restart, which is the operator's call.

<!-- BEGIN GENERATED: risk_ceilings -->
<!-- generated by scripts/gen_docs.py from ait/config/settings.py bounds and config.yaml; edit the code, not this table -->
| Setting | Shipped value | Allowed range | What it bounds |
|---|---|---|---|
| `risk.max_position_risk_pct` | 0.05 | 0.005 - 0.25 | One structure's maximum loss as a fraction of net liquidation; also the budget the condor builder is handed |
| `risk.max_daily_loss_pct` | 0.045 | 0.005 - 0.2 | The daily-loss breaker; blocks new entries only, exits keep running |
| `positions.max_position_pct` | 0.05 | 0.01 - 0.3 | The position sizer's hard per-position cap |
| `positions.max_portfolio_risk_pct` | 0.2 | 0.005 - 0.5 | Aggregate open capital at risk |
| `risk.max_symbol_concentration_pct` | 0.2 | 0.05 - 0.5 | Exposure in one symbol |
| `backtest.position_size_pct` | 0.05 | 0.01 - 0.25 | Research per-trade size, never looser than live's per-trade range |
<!-- END GENERATED: risk_ceilings -->

**Links enforced at startup** (a violation is a startup error, because the runtime symptom is a silent total stop). The per-trade budget `risk.max_position_risk_pct` must be no larger than the sizer's per-position cap
`positions.max_position_pct`, the aggregate cap `positions.max_portfolio_risk_pct` and the per-symbol cap `risk.max_symbol_concentration_pct`; otherwise the sizer or an aggregate gate would refuse every entry. Raising the
per-trade budget therefore also needs those caps raised with it. Not a startup error: a daily breaker below one full maximum loss. The launch coherence check only reports it by Telegram, because it measures the real condor
builder and the breaker never closes a position (it blocks entries after the loss).

---

## 8. Order execution and reconciliation

### 8.1 Placing an entry

[`TradeExecutor.execute_signal`](../src/ait/execution/executor.py) is the only code that submits an opening order. In order:

1. **Undefined-risk refusal.** A naked shape is refused unless the environment interlock `AIT_ALLOW_UNDEFINED_RISK=1` is set. It is environment-only on purpose, so re-enabling it is a deliberate act, not a config edit.
2. **Order-rate backstop.** More than a few orders in a rolling minute is a malfunction, not a strategy: refuse and log critically.
3. **Market-hours guard** at the executor itself (the last line of defence), failing closed if the calendar cannot answer.
4. **Connected-session check.**
5. **Write the `PENDING` trade row first, then place the order.** A crash between the two leaves a harmless pending row (swept later) instead of a live order nobody knows about.
6. **Qualify all legs in one batch** and validate the combo's leg structure (a reversed leg would turn defined risk into undefined risk).
7. **Live combo quote validation**: reject a disorderly (too wide) combo spread or a signal price that has drifted far from the live mid; an entry with no usable quote is refused.
8. **Price from the live mid**, starting near mid and **stepping toward marketable** on a reprice ladder if unfilled, never past the credit floors that justified the trade.
9. **One combo (BAG) order**, tagged with the trade id, so all legs fill together or not at all (no legging risk).

An unfilled entry is repriced on a timer and finally cancelled; a partial fill has its remainder cancelled so legs are not orphaned. The whole timeline (grace periods,
rests, timeouts, what each floor rests on) is in [section 8.4](#84-order-lifecycle-timeline); do not rely on the example durations here.

### 8.2 Exits and booking

Closing a position is a combo order priced from the current mark with a bounded cross, re-priced on each unfilled attempt, with a ceiling relative to the mark and a band-reject memory so a
broker "price band" message caps later attempts. The trade moves `FILLED → CLOSING → CLOSED` through **compare-and-set transitions**, and the move to `CLOSING` is made *immediately before* the order is
placed so a crash leaves a recoverable state.

**Exactly-once booking.** A close can be discovered by the fill event, the fill sweep, the reconciler or a restart. The money side (the day's stats and the circuit breaker) is written in **one transaction guarded by a
booked-once marker**, so a second discovery books nothing. Each non-money step (the analytics write, the learning and Thompson updates, the Telegram message) is run at most once per success and retried if it failed. Closes that
cannot be booked at once go to an **exit outbox** that is drained every cycle. Commissions are booked as an estimate and **trued up** from the broker's execution reports.

### 8.3 Reconciliation

[`execution/reconciler.py`](../src/ait/execution/reconciler.py) compares the bot's books with the broker's positions and orders:

- **At startup, pre-market and post-market** (and whenever it was owed because the broker was disconnected: a reconcile against a disconnected broker is *not* reported as clean, and at once when the operator answers a Telegram prompt about an unrecognised position). Separately, before any exit the exit path checks the trade's legs against a fresh broker query.
- It promotes or cancels stale `PENDING` rows, books closes it can prove, **re-adopts working exit orders** so a restart cannot cause a duplicate close, and recognises roll orders. A trade whose legs are gone at the broker (for example closed by hand) is **booked by the bot itself**, from the broker's realised P&L when it is usable and as a $0 placeholder that does not count in the go-live verdict otherwise; a partly closed structure is never closed automatically, it is reported.
- A live option position the bot has no record of sets **`data/HALT_UNTRACKED`** and freezes entries until the operator resolves it at the broker.
- A short option **assignment** is detected and reported for the operator to liquidate.
- The **end-of-day report** states clean or the list of breaks, plus a gap-stress number.
- Positions the operator trades by hand can be **declared once via Telegram** ([`execution/manual_positions.py`](../src/ait/execution/manual_positions.py)) so they do not trip the untracked-position freeze.

Operator procedures for all of this are in [RUNBOOK.md](RUNBOOK.md) ("Alert → action", "Resolving a books-versus-broker break", "Manual intervention procedure").

### 8.4 Order lifecycle timeline

Every age below is evaluated on the **30-second fast-monitor tick**, so a duration shorter than the tick takes effect at the next tick. The settings are in `config.yaml`; the table is generated from the schema (defaults
and allowed ranges), and the startup log event `order_timeline_in_effect` prints every effective value.

<!-- BEGIN GENERATED: order_timeline -->
<!-- generated by scripts/gen_docs.py from ait/config/settings.py; edit the code, not this table -->
**Exit order (the position is CLOSING)**

| Age | What happens | Setting | Default | Allowed range |
|---|---|---|---|---|
| 0 to grace | Ignored by the fill checker so the order can appear in IBKR's trade list; a brand-new order would otherwise read as cancelled. | `exit.order_visibility_grace_seconds` | 10.0 | 5.0 - 60.0 |
| rest | Still unfilled: a cancel is requested but the position stays CLOSING so the monitor can re-price. Stop exits (stop loss, short-strike touch, trailing, breakeven) rest less than ordinary exits. | `exit.exit_stop_pending_rest_seconds` / `exit.exit_pending_rest_seconds` | 120.0 / 300.0 | 15.0 - 1800.0 / 30.0 - 1800.0 |
| band TTL | A broker "price band" reject caps later attempts for this long, then stops. | `exit.exit_band_ttl_seconds` | 300.0 | 30.0 - 3600.0 |
| foreign | The order belongs to ANOTHER client id and cannot be cancelled from this session: the operator is paged instead of the cancel being retried. | `exit.foreign_exit_escalation_seconds` | 300.0 | 60.0 - 3600.0 |
| zombie | Still unresolved: the broker is asked; only if it answers and does not list the order is CLOSING reverted to FILLED so a fresh exit can fire (otherwise the position is held CLOSING and the operator is paged). | `exit.zombie_exit_revert_seconds` | 900.0 | 120.0 - 7200.0 |

**Entry order**

| Age | What happens | Setting | Default | Allowed range |
|---|---|---|---|---|
| every step | An unfilled entry combo's limit steps toward marketable (at 0.25, 0.60 and 1.00 of the offset). | `positions.entry_ladder_step_seconds` | 45.0 | 30.0 - 300.0 |
| partial | A partially filled entry has its remainder cancelled so legs are not orphaned. | `positions.partial_fill_cancel_seconds` | 30.0 | 5.0 - 300.0 |
| absent | An entry missing from the trade list is treated as still pending while younger than this (a disconnect makes every working entry look cancelled for a moment). | `positions.absent_order_grace_seconds` | 30.0 | 10.0 - 120.0 |
| timeout | An unfilled entry is cancelled. | `positions.entry_order_timeout_seconds` | 240 | 30 - 3600 |
<!-- END GENERATED: order_timeline -->

**Relations enforced at startup:** the visibility grace is below the stop rest; the stop rest does not exceed the ordinary rest; the foreign-order escalation is at least the ordinary rest; the zombie cap is at least twice the longer rest
(the cancel needs at least as long to resolve as the order was given to fill, or a fill that is merely not yet visible is re-armed as a duplicate close); partial-fill cancel and absent-order grace are below the entry timeout.

**What the floors rest on.** They are engineering margins, not measurements: the order-acknowledgement latency is being logged (`order_ack_latency`, `order_absent_judged`, published to the shared logs) so they can be revisited once
enough paper orders, exits included, exist. Which alerts these timers produce, and what to do, is in [RUNBOOK.md](RUNBOOK.md) ("Alert → action").

---

## 9. Position management and exits

**Who manages an open position?** The protection task evaluates every tracked position every 30 seconds in
[`PortfolioManager`](../src/ait/execution/portfolio.py). The first rule that fires wins; it runs on marks from the broker's portfolio, and on the underlying price where marks are missing.

**What are the exit rules, in the order they are checked?** (Example numbers; the live ones are under `exit:` in `config.yaml` and in `execution/exit_policy.py`.)

| # | Rule | What it does |
|---|---|---|
| 1 | Stop-loss | **Debit structures:** a dynamic stop (initial stop, breakeven, trailing from the high-water mark, widened for volatile underlyings). **Credit structures:** only an optional flat loss limit as a multiple of the credit, **disabled by default**: the short-strike touch rule below is the early loss exit and the wings cap the tail. Needs real marks |
| 1b | **Short-strike touch** | A credit structure is closed when the underlying touches a short strike. This is the **primary loss exit** for condors (the flat credit-loss stop is disabled by default). It uses the underlying price, so it protects even without option marks. On a degraded or frozen quote it waits for corroborating ticks instead of acting on one print |
| 2 | Take profit | A target that **shrinks as expiry nears** (a ladder, for example 50% of credit with plenty of time down to 20% near expiry). Shared with research via `exit_policy.py` |
| 3 | Assignment risk | ITM short strikes at DTE 0–1 must close |
| 3a | Expiry approaching | Close at or below a small DTE (example: 5), paired with the entry DTE floor so each position gets days of theta runway |
| 3b | Delta breach | Present but inert: it needs a live greeks feed the monitor does not have |
| 3c | IV-crush pre-close | Short-premium positions are closed within a couple of days before a known earnings date, whatever their P&L (moot for index ETFs, which have none) |
| 3d | Macro-event flatten | Applies to undefined-risk / assignment-risk shapes; **defined-risk condors are exempt** (the wings cap the surprise and post-event vol crush is the payoff). New credit entries are separately kept out of the days before an event ([4.2](#42-pre-execution-checks)) |
| 3e | Ex-dividend risk | A short call is closed ahead of an ex-dividend date regardless of DTE |
| 4 | Partial exits | Scale-out milestones for strategies that use them; each level fires once (recorded in the database) |
| 5 | PDT veto | Can hold back an exit that would be an illegal day trade |

**Thesis check.** For *directional* strategies the 5-minute cycle also re-evaluates the entry thesis (direction flipped, regime shifted to high volatility, VIX spiked since entry) and exits if it
is invalidated. Iron condors are exempt.

**Condor defense and rolls.** Code exists to classify a condor's tested side by its synthetic delta and DTE and to **roll** an at-risk side to a later expiry (selector, combo builder, exactly-once booking,
reconciler awareness). It is **off by default** (`condor_defense.enabled`, and a separate `execute_rolls`): while off, nothing changes; with `enabled` alone a "close" verdict becomes actionable and
roll verdicts are only reported. The assignment and ex-dividend exits can also be set to roll instead of close (`assignment_response`), but the shipped policy is close-only with rolls not executed. Treat all of this as a staged feature that awaits its paper verdict.

**One exit policy for live and research.** The take-profit ladder, the expiry-approaching close, the macro windows and the expiry-day flatten live in one import-light module so the backtester cannot
drift from the bot.

---

## 10. Learning and adaptation

**What does "self-learning" actually do?** Less than the name suggests, deliberately.

- **Learning cycle** (pre-market and post-market, `learning.enabled`): [`TradeAnalyzer`](../src/ait/learning/analyzer.py) mines the last N days of closed trades from the analytics store (strategy, symbol,
  regime, hour and confidence-band win rates, exit giveback); [`StrategyAdaptor`](../src/ait/learning/adaptor.py) turns strong insights into **bounded** changes: a strategy size multiplier, a disabled
  strategy / symbol / hour, a confidence-threshold override. Bounds are settings (`learning.*`), the number of changes per cycle is capped, and every change is logged and sent to Telegram.
- **Why bounded?** A bad streak is often noise; unbounded adaptation could permanently switch off a good strategy.
- **Paper-trading mode** (`learning.paper_trading_mode`): bypasses the adaptor's overrides, Thompson re-ranking and the meta-label gate, so a paper track record measures *one fixed rule set*
  that stays comparable with the backtest. It must be turned off deliberately at go-live.
- **Counterfactual log** ([`learning/counterfactual.py`](../src/ait/learning/counterfactual.py)): a durable, record-only list of signals the bot generated but rejected (`data/counterfactual_log.json`). The automatic
  evaluation of "what would have happened" was retired after it proved unreliable; the record remains for deliberate studies.
- **Meta-labeler retraining**: when enabled, the secondary model retrains from the bot's own closed trades and their stored entry snapshots.

The offline backtest has its own, separate learner that adapts between walk-forward windows: [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) section 7.

---

## 11. Data and storage

### 11.1 Sources

| Data | Source | Notes |
|---|---|---|
| Quotes, option chains, option marks, account, positions, orders | Interactive Brokers | The only source for option chains and the only broker. Market-data type is an entitlement setting (delayed vs live) |
| Daily bars (training, features) | IB history stored in `historical.db` | The daily retrain takes IB bars only and stores nothing if any series is missing |
| Fallback for quotes and bars | Yahoo Finance, after IB | The market-data service tries IB, then Yahoo, with explicit price-basis rules (unadjusted for dividends) so bars from different sources are not mixed silently. Never mock data |
| Historical option bars | Polygon | Used only by the option-history backfill script for research, not by the live bot |
| Macro (yield curve, dollar) | FRED CSV | Cross-asset features |
| Calendars | Hard-coded economic calendar, earnings and dividend services with last-known-good caches, SEC EDGAR 8-K feed | Unknown data refuses entries rather than allowing them |

Staleness is handled in layers: an in-memory TTL cache for quotes (shorter than the monitor cadence by design, so two consecutive ticks never read the same cached print), data-quality checks, and
fail-closed gates. If IB is unavailable no new trade can be entered, but positions are still monitored from the last known prices.

### 11.2 Stores

| Store | Technology | What is in it |
|---|---|---|
| `data/ait_state.db` | SQLite (WAL) | The operational truth: trades and their status, open-position high-water marks and marks, trade context, key-value state, the executions ledger, the exit outbox |
| `data/ait_analytics.duckdb` | DuckDB | Closed trades, daily stats, trade context, feature snapshots, equity stats; for analytics and learning. Mirrors SQLite on close |
| `data/historical.db` | SQLite | The shared history tables: daily prices and implied vol, 5-minute bars, option bars and spread samples, calendars, dividends. One set of tables for live and research; each row carries a source tag. Table-by-table list: [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) section 1 ("Where each field comes from") |
| `data/` control files | files | `HALT`, `HALT_UNTRACKED`, `RESTRICTED.txt`, `KEEPER_PAUSE`, `bot_heartbeat`, `orchestrator.lock`, `ib_endpoint.json`, `thompson_state.json`, `counterfactual_log.json` |
| `models/` | pickles | Trained models with their spec records, timestamped direction-model versions for rollback, the `.retrained` marker, `research/` sandbox |
| `logs/`, `reports/` | files | Structured logs, supervisor and retrain logs; daily reports and run records (`reports/runs/`) |
| `data/backups/` | files | Hash-verified snapshots of the state database, mirrored off-box |

**Dual write on close.** SQLite is the source of truth for live operation; closing a trade also writes it to DuckDB, which every analytical consumer reads. Never edit either store by hand; ad-hoc scripts open
them read-only. Operator procedures around these stores (backups, catching up missing data) are in [RUNBOOK.md](RUNBOOK.md) ("Daily 2-minute routine", "Data catch-up and maintenance").

---

## 12. Monitoring and operator surfaces

**How do I know what the bot is doing?**

| Surface | Where | Shows |
|---|---|---|
| Live status page | `http://localhost:8503` (`status_server.py`) | Running state, heartbeat, IB endpoint match, positions with unrealised P&L, today's activity, realised P&L from real exits, the go-live scorecard |
| Analytics | `http://localhost:8501` (Streamlit, `dashboard/app.py`) | Tabs: portfolio overview, trade history, analytics, trade intelligence, self-learning, system health, feature importance, backtest |
| Log viewer | `http://127.0.0.1:8502` (`web_logs.py`) | Colour-coded live logs |
| Walk-Forward Lab | `http://localhost:8080`, started by hand | Research runs, Optuna trials, predictor models: [WALKFORWARD_DASHBOARD.md](WALKFORWARD_DASHBOARD.md) |
| Telegram | the operator's chat | Trade opened / closed, alerts, the morning and evening digests, the weekly scorecard, end-of-day reconciliation |
| Logs | `logs/ait.log`, `logs/orchestrator.log`, `logs/bot_stdout.log`, `logs/retrain_*.log`, `logs/fatal.log` | Structured events (snake_case event names plus key-value pairs) |

The ports are settings (`dashboard.*`). All are local-only.

**What keeps the monitors honest?**
- A **watchdog** tracks heartbeats for IB, market data and the trading loop and tries to recover components.
- [`monitoring/ops_health.py`](../src/ait/monitoring/ops_health.py) is the shared authority for operator-facing health, written so that a channel that was never wired reports *unknown*, not green.
- **One scorecard authority.** [`reporting/go_live.py`](../src/ait/reporting/go_live.py) defines what counts as a real close, which strategies count toward the go-live verdict, and each criterion's formula. The
  status page, digest and weekly scorecard all read it, so they cannot disagree. A criterion that cannot be computed as defined is printed as unavailable, never as a differently-defined number.
- **Silence is an alarm.** The morning digest, the heartbeat and the Telegram-dead probe exist because a quiet failure is the dangerous kind.

**Go-live criteria.** The scorecard grades five criteria, defined in `reporting/go_live.py` (the thresholds below are read from it). Only real closes of the strategies in `GO_LIVE_VERDICT_STRATEGIES` count; placeholder rows
(never filled, reconciler-unknown, migrated) never enter any statistic. The decision itself is the operator's, against `PLAN.md`.

<!-- BEGIN GENERATED: go_live_gates -->
<!-- generated by scripts/gen_docs.py from ait/reporting/go_live.py; edit the code, not this table -->
| Criterion | What is measured | Passes when |
|---|---|---|
| [1] sample | Enough real closes | `GO_LIVE_PRELIM_CLOSES` (50) for a preliminary read, `GO_LIVE_FUNDING_CLOSES` (100) before funding (too few cannot settle the profit-factor test) |
| [2] profit factor | Gross profit over gross loss of real closes | at or above `GATE_PF` (1.3) |
| [3] drawdown | Peak-to-trough, as a percentage of deployed risk (not of paper net liquidation) | at or below `GATE_DD_PCT` (8.0) |
| [4] slippage | Median entry slippage as a percentage of the credit, over the trailing `SLIP_WINDOW` (20) fills. Printed as unavailable when entry fills cannot be identified | at or below `GATE_SLIP_PCT` (8.0) |
| [5] incidents | Zero unmanaged-position incidents. Not computable from the state database, so the scorecard prints it as unavailable and points at `scripts/shadow_referee.py` (check 8) | checked by hand |
<!-- END GENERATED: go_live_gates -->

**Deploying safely.** A green test suite is not a safe deploy. After the tests, deploy outside market hours, run `scripts/smoke_deploy.py` (read-only, no IB; it must print `SMOKE PASS`), confirm liveness after
the next open, and tag the verified deploy. Full checklist: [RUNBOOK.md](RUNBOOK.md) "Deploy checklist". A change to `orchestration/master.py` needs a full supervisor restart, not just a bot restart.

---

## 13. Configuration system

**Where do values live?** In `config.yaml` — the *only* home for trading values. Models in [`config/settings.py`](../src/ait/config/settings.py) are strict pydantic classes (`extra="forbid"`): a typo'd key is a **startup
error**, not a silently ignored line. Each field has bounds, validators and a description that records why its default is what it is. Secrets (IB account, API keys, Telegram) come only from `.env`, are
never logged, and are never copied into reports.

**Precedence.** Highest first: an explicit argument at the call site → an environment variable (`.env` or the process) → `config.yaml` → the last-resort safety net `CONTRACT_DEFAULTS` in
[`config/runtime_env.py`](../src/ait/config/runtime_env.py). Code that needs a value uses `resolve_config_value()` or the `contract_*` helpers rather than a local fallback literal. Every entry point calls
`apply_runtime_env_defaults()` *before* importing any trading code (it also sets the OpenMP guards that must precede numpy/xgboost).

**Which files are there?**

| File | Used by | Purpose |
|---|---|---|
| `config.yaml` | live bot, supervisor, research | The live configuration, and the shared settings research also reads |
| `config_research.yaml` | research entry points only | Only the keys where research deliberately differs from live, plus research-only tunables. The live bot never loads it (a test enforces this) |
| `studies/<name>.yaml` | one research run | A committed, named set of changes for one run ([section 14](#14-research-and-optimization)) |
| `.env` | all | Secrets and per-box overrides (git-ignored) |
| `src/ait/config/invariants*.py` | all | Visible fixed limits that are not configurable |

**Which sections does `config.yaml` have?** One per strict model in `settings.py`. The table is generated from the schema (the counts) and from a hand-written description per section, so a new section cannot be added without being described here.

<!-- BEGIN GENERATED: config_sections -->
<!-- generated by scripts/gen_docs.py from ait/config/settings.py and config.yaml; edit the code, not this table -->
| Section | What it controls | Settings | Set in | Explained in |
|---|---|---|---|---|
| `trading` | Symbols scanned, paper or live, scan cadence, the daily trade budget and the no-entry window before the close | 11 | `config.yaml` | [GUIDE.md](GUIDE.md#3-the-trading-day), [GUIDE.md](GUIDE.md#42-pre-execution-checks) |
| `account` | The capital the risk maths uses, simulated capital in live mode, capital tiers and the pattern-day-trader switches | 7 | `config.yaml` | [GUIDE.md](GUIDE.md#7-risk-management) |
| `positions` | Position-count and exposure caps, and the entry order's price ladder and timers | 10 | `config.yaml` | [GUIDE.md](GUIDE.md#71-the-risk-manager), [GUIDE.md](GUIDE.md#84-order-lifecycle-timeline) |
| `position_sizing` | The position sizer's multiplier curves (confidence, volatility, IV rank, per-strategy risk, drawdown, VIX tiers) | 9 | `config.yaml` | [GUIDE.md](GUIDE.md#72-sizing) |
| `risk` | Risk-manager gates: per-trade and aggregate risk, concentration, VIX and IV-rank limits, the circuit breaker and the daily-loss brake | 21 | `config.yaml` | [GUIDE.md](GUIDE.md#7-risk-management) |
| `options` | Option-chain liquidity filters, the entry expiry band and which strategies are enabled | 7 | `config.yaml` | [GUIDE.md](GUIDE.md#4-entry-pipeline), [GUIDE.md](GUIDE.md#5-strategies) |
| `backtest` | The research engine, walk-forward, learner, optimiser, friction and data settings (research only; a few are shared with live) | 189 | `config_research.yaml` | [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md#4-configuration-where-every-value-lives) |
| `ml` | Models: labels, validation, hyperparameters, features with their switches and windows, the retrain, drift and the entry-gate mode | 158 | `config.yaml` | [MODELS.md](MODELS.md) |
| `regime` | The rule-based regime classifier's thresholds | 29 | `config.yaml` | [MODELS.md](MODELS.md#8-the-regime-classifier) |
| `meta_label` | The meta-labeler gate and the guards before it can act | 10 | `config.yaml` | [MODELS.md](MODELS.md#7-the-models-one-by-one) |
| `earnings` | How long a last-known-good earnings answer may be used, the buffer around earnings dates, and whether to refuse entries when the data is unavailable | 4 | `config.yaml` | [GUIDE.md](GUIDE.md#42-pre-execution-checks) |
| `dividends` | The same last-known-good and fail-closed policy for ex-dividend dates | 2 | `config.yaml` | [GUIDE.md](GUIDE.md#42-pre-execution-checks) |
| `assignment_response` | Whether an at-risk short leg is closed or rolled to a later expiry (shipped as close only, rolls not executed) | 9 | `config.yaml` | [GUIDE.md](GUIDE.md#9-position-management-and-exits) |
| `condor_defense` | The staged condor roll-and-defense feature (off by default) | 11 | `config.yaml` | [GUIDE.md](GUIDE.md#9-position-management-and-exits) |
| `exit` | Exit rules: stops, the take-profit ladder, the touch stop, and the exit order's price and timers | 27 | `config.yaml` | [GUIDE.md](GUIDE.md#9-position-management-and-exits), [GUIDE.md](GUIDE.md#84-order-lifecycle-timeline) |
| `learning` | The bounded learning layer and paper-trading mode | 19 | `config.yaml` | [GUIDE.md](GUIDE.md#10-learning-and-adaptation) |
| `sentiment` | Retired; nothing reads it | 1 | `config.yaml` | - |
| `notifications` | The Telegram switch | 1 | `config.yaml` | [GUIDE.md](GUIDE.md#12-monitoring-and-operator-surfaces) |
| `dashboard` | Whether the dashboards start and their ports | 4 | `config.yaml` | [GUIDE.md](GUIDE.md#12-monitoring-and-operator-surfaces) |
| `logging` | Log level, file and rotation | 4 | `config.yaml` | [GUIDE.md](GUIDE.md#12-monitoring-and-operator-surfaces) |
| `sizing_probe` | Test fixtures for the startup sizing self-test (not trading values) | 5 | `config.yaml` | [RUNBOOK.md](RUNBOOK.md#alert--action) (`CAP-GRID INCOHERENT`) |
| `parity` | Tolerances for the research-versus-reality parity replay | 4 | `config_research.yaml` | [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md#102-parity-replay-research-versus-reality) |
| `data_collection` | The scheduled short-DTE friction collection run by the supervisor (research data only, off by default) | 14 | schema default only | [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md#101-short-dte-friction-collection) |
| `ibkr` | Broker connection (from `.env`) | 4 | `.env` | [GETTING_STARTED.md](GETTING_STARTED.md#4-configure) |
| `api_keys` | API keys and the Telegram token (from `.env`) | 4 | `.env` | [GETTING_STARTED.md](GETTING_STARTED.md#4-configure) |
<!-- END GENERATED: config_sections -->

**How do I read a setting?** A dotted name such as `risk.max_position_risk_pct` is section `risk`, key `max_position_risk_pct`. Find that row in [reference/SETTINGS_LIVE.md](reference/SETTINGS_LIVE.md) for its type, allowed range, schema default, shipped value and meaning, or open `config.yaml` and look under the section; the comment beside a value in `config.yaml` says why the shipped value is what it is.

**Settings are frozen at start.** A process runs on the configuration it started with, so a deploy that adds a key cannot break a running process; a change takes effect at the next restart (the supervisor needs its own restart for its own settings).

**Where do I see the effective value?** For research, `python scripts/show_research_config.py` prints every setting with the layer that decides it ([RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) section 4). For live, read `config.yaml`; the bot logs the
effective fixed limits, order timeline and any config-versus-default divergences at start.

**Example values.** A few settings to show what kind of thing lives where. **These are examples; open `config.yaml` for the real values.**

| Area | Setting (section.key) | Example | What it controls |
|---|---|---|---|
| Universe | `trading.universe` | `[SPY, QQQ, IWM]` | Symbols scanned |
| Mode | `trading.mode` | `paper` | Paper vs live |
| Cadence | `trading.scan_interval_seconds` | `300` | Seconds between entry scans |
| Strategies | `options.strategies` | `[iron_condor]` | Enabled structures |
| Expiry window | `options.dte_range` | `[14, 30]` | Entry DTE band |
| Account | `account.capital` | `5000` | The one capital number (live simulates it while `account.simulate_live_capital`) |
| Per-trade risk | `risk.max_position_risk_pct` | `0.05` | Max loss of one structure as a fraction of net liquidation |
| Daily brake | `risk.max_daily_loss_pct` | `0.045` | Entry freeze threshold |
| VIX halt | `risk.credit_vix_halt` | `28` | No new credit entries at or above |
| ML gates | `ml.entry_gates_enabled` | `false` | Observe vs enforce |
| Learning | `learning.paper_trading_mode` | `true` | Fixed rule set in paper |
| Rolls | `condor_defense.enabled` | `false` | Staged feature |
| Dashboards | `dashboard.status_port` | `8503` | Local status page |

**Every setting** (name, type, allowed range, schema default, meaning, and which file lists it) is in the generated [reference/SETTINGS_LIVE.md](reference/SETTINGS_LIVE.md) and [reference/SETTINGS_RESEARCH.md](reference/SETTINGS_RESEARCH.md). Topic documents explain the groups: this guide (fixed limits, risk ceilings, order timeline, section 7.4 and 8.4), [MODELS.md](MODELS.md) (hyperparameters, labels, feature windows and switches, drift, regime) and [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) (research blocks).

---

## 14. Research and optimization

**What is it?** An offline simulator of the iron-condor strategy: a walk-forward backtester (train a window, trade the next one out of sample), a parameter optimizer (Optuna), and a study layer for committed
one-question runs with run records. It reads the shared history tables and the config files, and **never touches the broker, the live databases or the live model files**. A research number answers
"what would this rule set have done on this history, with these cost and fill assumptions"; it is not a forecast, and a research finding changes the live bot only through a registered decision in `PLAN.md`.

Everything about running it, reading it, and the settings it has is in [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md); the Walk-Forward Lab dashboard is in [WALKFORWARD_DASHBOARD.md](WALKFORWARD_DASHBOARD.md). This guide does not repeat them.

---

## 15. Technology and repository layout

**Stack.** Versions below are from `pyproject.toml` and `constraints.txt` at the time of writing; check those files for the current pins.

| Layer | Technology |
|---|---|
| Language | Python 3.13 (`requires-python >=3.13,<3.14`); the trading box runs the frozen set in `constraints.txt` |
| Broker API | `ib_insync` 0.9.86 pinned exactly (the library is archived; it is a load-bearing pair with the Gateway version) |
| ML | XGBoost, LightGBM, scikit-learn; Optuna for searches |
| Options math | `py_vollib` (Black-Scholes greeks and IV), SciPy, PyWavelets for fractal features |
| Data | pandas, NumPy, `pandas_market_calendars`, yfinance, Polygon client, DuckDB |
| Runtime | pydantic and pydantic-settings, PyYAML, structlog, APScheduler, aiohttp |
| UI | Streamlit and Plotly (analytics), Flask (status page and log viewer) |
| Tests | pytest, pytest-asyncio, hypothesis; CI on GitHub Actions |

Installation is in [GETTING_STARTED.md](GETTING_STARTED.md).

**Repository layout.** The layout, ownership of each package, conventions, and the test and quality gates are in [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) sections 1, 2, 5 and 6. In brief:

- `src/ait/` is the package (section 2 lists its sub-packages); `scripts/` holds operator and research tools; `tests/` the test suite; `docs/` the documents; `deprecated/` retired code that is never imported.
- `data/`, `models/`, `logs/` and `reports/` are runtime state and are git-ignored.

---

## 16. Design principles

Why the code looks the way it does. Each is a lesson the system paid for; `PLAN.md` records the incidents.

1. **Separate the supervisor from the bot.** The thing that restarts must not share a failure domain with the thing that trades.
2. **Protection never waits for the scan.** Exits run on their own cadence in their own task; a slow or failing entry scan cannot starve them.
3. **Fail closed on missing information, and say so.** No IV rank, no VIX, unknown earnings or dividend data, an unreadable ban file, an unusable quote: refuse the *entry*, page if it persists, and keep managing positions.
4. **Observe before you enforce.** A new gate first logs what it would have done; it is armed only when the record shows it helps.
5. **Match the model to the payoff.** A condor is gated by a range model, not a direction model.
6. **One owner per fact.** Shared definitions (credit strategies, the go-live verdict, the exit ladder, IV rank, the fractal gate, the feature spec) are imported, never re-typed; duplicated constants have caused silent divergence more than once.
7. **Config is the only home for trading values; fixed limits are visible.** Tunables are in config with bounds and reasons; the few non-configurable backstops live in one module and are logged at start.
8. **Write the row before the order; book each close once.** Compare-and-set state transitions, an outbox for unbooked closes, and a reconciler that treats the broker as the truth.
9. **Bound adaptation.** The system may nudge sizes and restrictions within limits; it may not rewrite itself.
10. **Time is Eastern and explicit.** Use `now_et()` and pass one `now` through a decision; a bare local `datetime.now()` in scheduling or trade timestamps is a known bug class.
11. **Silence is an alarm.** A quiet channel, a stale heartbeat or a missing digest is treated as a fault.
12. **A purge gap between training and validation** equal to the label horizon, so overlapping labels cannot inflate skill.
13. **Research cannot touch live.** Separate config layers, separate model sandbox, read-only against live state, and a parity manifest that lists every deliberate difference.

---

## 17. Extending the system

Adding a feature, strategy, gate, configuration key, scheduled job or alert, and the list of changes that need the operator's confirmation before anyone touches them, are in [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) sections 3 and 4.

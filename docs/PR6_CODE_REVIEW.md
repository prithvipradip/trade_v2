# Review: PR #6 "Audit rounds 1-11" (`feature/event-straddle` → `main`)

**Link:** https://github.com/prithvipradip/trade_v2/pull/6
**Size:** 107 commits, 2026-06-12 → 2026-08-08 (~8 weeks of work; the PR description was written mid-way through and undersells it — it says "60 commits," the branch has grown to 107 since). 141 Python files changed (~18,200 lines). The other ~1,200 changed files are backtest-report JSON artifacts, not code.
**Live config today:** only `iron_condor` is enabled (`config.yaml`). Every other strategy (`short_strangle`, `long_straddle`, the credit/debit spreads) is present in the code but commented out.

---

## What changed, in plain English

Despite the branch's name, this PR does not add an "event straddle" strategy — the opposite, actually. An event-driven straddle was added in an earlier, already-merged PR, traded live for a while, lost money (-55% on one position that needed a manual close because there was no way to exit it), and this PR **removes it**, along with two other strategies (calendar spreads, covered calls) that never worked out. What's left and actually enabled today is a single strategy: **iron condor** on index ETFs (SPY/QQQ/IWM).

Think of this PR as eight weeks of hardening a bot that had already been trading with real money, after a string of incidents. The major themes:

- **The money numbers are now real.** Early on, the bot's P&L bookkeeping didn't match what the broker actually charged and filled at. This PR rebuilds the trade ledger from real broker fills and real commissions, and restarts the track record from 2026-07-06 with the broken history archived (not deleted) rather than mixed in.
- **New risk controls.** The bot now caps how much of the account can be tied up in one stock and in one options expiration date, caps how many "sold premium" positions it can hold at once (tightening automatically when market volatility rises), and adds a same-day loss alarm that watches *unrealized* losses on open positions, not just money already lost — checked every 30 seconds, not just at day-end.
- **How it picks trades changed.** It used to check its watchlist one symbol at a time and take the first workable setup. Now it scores every candidate across the whole watchlist first and only then takes the best one. It also waits for a real, no-lag/no-fake price to enter or exit on, and only blocks trading 1 day before a big economic event instead of 4 (the old rule was blocking roughly half of all trading days).
- **The machine-learning layer is back on, with evidence.** It was briefly turned off to test whether it was actually helping. A clean re-test showed it cuts the worst-case losing streak roughly in half (a ~35% peak drawdown down to ~10%), so it's back to actively screening trades rather than just watching.
- **A lot of "will I actually find out if something breaks" work.** Several real incidents (a bot that silently sat frozen for a whole session, an alert path that had been dead since the bot was first built, duplicate order/fill handling that could double-close a position) drove fixes: the bot detects and recovers from being frozen, pages a human through a working alert channel, can't accidentally run two copies against the same account, and there's now a post-deploy smoke test and a next-morning "is it actually alive" check that run automatically.
- **Housekeeping.** The repo root was cleaned up (25 files down to 12), old configs/docs archived rather than deleted, and this is the first PR in the project's history to run any automated checks (CI) at all.

**Bottom line on trustworthiness:** this is a real, substantial safety upgrade over what was running before — most of the ~70 issues the team's own internal audit found were genuinely fixed. But my independent review below found **20 additional issues that survived into this version**, including two that actively undermine the safety of the one strategy currently trading real money. I'd fix those two before letting this keep trading unattended.

---

## Review findings (20, ranked by importance)

Reviewed by five independent passes over the diff (risk/execution/broker, orchestration/strategies, ML/backtesting, data/monitoring/config, and ops scripts), then I personally re-traced the top ones against the actual code before including them here. Each is marked **CONFIRMED** (I traced the exact trigger in the code) or **PLAUSIBLE** (the bug mechanism is real; I couldn't fully pin down how often it fires in practice).

### Live now — affects the bot as currently configured (iron condor only)

**1. The pre-earnings safety exit for iron condor doesn't actually run.** [`src/ait/execution/portfolio.py:601`](../src/ait/execution/portfolio.py#L601) — CONFIRMED
The code that's supposed to close a position 2 days before an earnings report (to avoid the volatility-crush) sits behind a check that's a chain of "else, try this next check" rules. The check just before it (a delta/directional-risk check) is written so that it always claims the turn for any iron condor position — whether or not delta actually breached — which means the earnings check after it never gets a chance to run. A condor sitting through an earnings report is exactly the scenario this was built to prevent, and it silently doesn't happen. The test file for this code always disables both checks, so nothing caught it.

**2. The "undo a bad model update" safety net is broken and would never trigger.** [`src/ait/ml/trainer.py:231`](../src/ait/ml/trainer.py#L231) — CONFIRMED
After each retraining, the bot is supposed to compare new model accuracy to old and automatically roll back if it got worse. The comparison code has a type-matching bug that makes it silently compare against an empty result every time, so the rollback check can never fire. If a scheduled retrain ever produces a materially worse model, nothing stops it from going live.

**3. The live dashboard can silently under-report today's and lifetime P&L.** [`status.py:150`](../status.py#L150) — CONFIRMED
The query that totals up closed-trade P&L for the dashboard (both the command-line one and the web page) has a filter with a subtle database quirk: any trade whose exit-reason field is blank/unset gets silently excluded from the total rather than included. A sibling piece of code elsewhere in the same file gets this right; this one spot doesn't.

**4. The "we're running out of economic-calendar data" alarm never reaches a person.** [`src/ait/data/economic_calendar.py:239`](../src/ait/data/economic_calendar.py#L239) — CONFIRMED
This was added specifically because the bot's macro-event blackout guards went silently blind once before (calendar data ran out and nobody noticed for weeks). The new alarm only writes to a log file — it was never wired to the Telegram alerts the bot uses for everything else. Unless someone is actively tailing logs, this reproduces the exact failure it was meant to catch.

**5. Your `.env` file can no longer change several safety-critical settings — including the switch to real-time market data.** [`src/ait/config/runtime_env.py:30`](../src/ait/config/runtime_env.py#L30) — CONFIRMED
A new mechanism locks in protective defaults (delayed vs. live data, risk gates, wing sizing) before your `.env` file is read, and `.env` is only allowed to fill in settings that are still blank. That means editing `.env` — the normal, documented way to configure this bot — silently can't change these 7 settings anymore; only a real shell-level environment variable can. One of the frozen settings is the delayed-vs-live market data switch, which the code comments say is meant to be flipped "after funding" — a step that may be close.

**6. The account-size setting used for go-live risk math has no enforcement.** [`src/ait/monitoring/analytics.py:41`](../src/ait/monitoring/analytics.py#L41) (and 2 sibling files) — CONFIRMED mechanism
Drawdown % and return % across the dashboard and analytics are computed against a setting (`AIT_CAPITAL_BASE`) that defaults to the old paper-trading account size and is nowhere validated against your actual funded amount. If it's not explicitly set when the account is funded for real, every percentage-based risk number silently uses the wrong denominator — looking plausible while being wrong.

**7. The earnings-date "don't hold through earnings" check can use the wrong day for a few hours around midnight UTC.** [`src/ait/data/earnings.py:155`](../src/ait/data/earnings.py#L155) — PLAUSIBLE
A new filter correctly rejects using a dividend date as an earnings date, but compares against the server's local clock instead of the Eastern-time convention the rest of the codebase uses. If the server isn't set to US Eastern time (common on cloud hosts), there's a multi-hour window where a same-day earnings date can look like it's already in the past — and "no known earnings date" is treated as safe to hold through.

**8. A historical-data repair tool can silently overwrite good price data with worse data.** [`scripts/backfill_historical_data.py:207`](../scripts/backfill_historical_data.py#L207) — CONFIRMED mechanism
This script's own commit message says it fixes bid/ask-midpoint bars silently overwriting real traded-price bars. In fact it only adds a label — the underlying overwrite is unprevented, and the default mode of the tool is the one that overwrites. Running the documented example command on a symbol that already has good data would replace it with lower-quality data, and nothing downstream can tell the difference. This feeds the same price history your ML features and backtests are built from.

**9. The 20%-of-account same-symbol exposure cap can undercount how much is really at risk.** [`src/ait/risk/manager.py:329`](../src/ait/risk/manager.py#L329) — CONFIRMED mechanism, PLAUSIBLE real-world bite
For iron condors, the cap is supposed to compare 20% of your account against the real dollar risk on a symbol. It actually sums up the *credit collected* on existing same-symbol positions instead of their *max loss* — which for a credit strategy is a much smaller number. Two same-symbol condors already open could look like a fraction of their real combined risk when a third is being sized.

**10. After a restart, the bot can mix up which exit order belongs to which position — if two similar positions on the same symbol are both mid-close.** [`src/ait/execution/reconciler.py:453`](../src/ait/execution/reconciler.py#L453) — PLAUSIBLE
The code that reconnects a restarted bot to its still-working orders can only identify a multi-leg order by the stock symbol, not by which specific position it belongs to. If two same-symbol multi-leg positions are both closing at the moment of a restart, one can end up silently unmanaged until the next check cycle (which does self-heal, but may record an estimated rather than exact fill price).

**11. No automatic stop if the broker's account-value feed goes stale.** [`src/ait/broker/account.py:43`](../src/ait/broker/account.py#L43) — PLAUSIBLE
When IBKR's account-value feed hiccups, the bot now correctly falls back to its last known good numbers instead of crashing — but nothing ever escalates this beyond a log line, even after 5+ minutes. Every dollar-based risk check keeps running against a frozen account balance for as long as the outage lasts.

**12. The post-deploy "is the bot actually alive" check will falsely fail every winter.** [`scripts/check_first_rth_liveness.py:32`](../scripts/check_first_rth_liveness.py#L32) — CONFIRMED, seasonal
This new check hardcodes the Eastern time offset for daylight saving time (EDT, UTC-4). From roughly November through March, the real offset is UTC-5, so every run during that period will misjudge how old a position's price data is by exactly one hour — and, depending on timing, will likely report a false "LIVENESS FAIL" alert.

**13. The automated post-deploy smoke test can raise a false alarm because of the bot's own normal trading activity.** [`scripts/smoke_deploy.py:184`](../scripts/smoke_deploy.py#L184) — CONFIRMED mechanism, PLAUSIBLE trigger
The smoke test checks that it didn't accidentally write to the live database by comparing row counts before and after. If the live bot itself opens or closes a real trade while the smoke test happens to be running, the test misattributes that normal activity to itself and can fail — which could make an operator distrust or roll back a perfectly good deploy.

**14. Backtest returns can be understated once a losing symbol drops out mid-test.** [`src/ait/backtesting/walkforward.py:1298`](../src/ait/backtesting/walkforward.py#L1298) — CONFIRMED mechanism
The capital base a backtest window divides by is fixed at the *original* number of symbols in the test, even for a later window where the self-learning feature has already dropped a symbol for underperforming. This understates the reported return for that window. Direction of the error is conservative (makes a strategy look worse than it is, not better), so it's a "might reject a good strategy" risk rather than a "might approve a bad one" risk.

**15. A data-outage fallback can feed the ML models a falsely "everything is calm" signal.** [`src/ait/ml/features.py:105`](../src/ait/ml/features.py#L105) — CONFIRMED inconsistency, PLAUSIBLE trigger
A new safety net that prevents one broken data column from wiping out an entire day's predictions fills gaps with `0.0`. Elsewhere in the same file, the documented "no VIX data available" neutral value is `0.5`, not `0.0` — so if the VIX feed is present but corrupted (as opposed to simply missing), the fallback tells the model volatility is near-zero, the opposite of neutral, at exactly the moment the data pipeline is unreliable.

**16. A fallback options-data source can let effectively dead contracts pass the liquidity filter.** [`src/ait/data/options_chain.py:97`](../src/ait/data/options_chain.py#L97) — PLAUSIBLE
A fix that correctly stops IBKR's "open interest not yet available" gaps from blocking real trades was applied the same way to the Yahoo Finance fallback path, where a reported `0` genuinely means no open interest. Combined with the live config's minimum-volume check being turned off, a truly illiquid fallback-sourced contract could pass the filter.

### Not live yet — only matters if these currently-disabled strategies are re-enabled

**17. If short strangle is turned back on, its risk exposure will be invisible to the portfolio-wide risk cap.** [`src/ait/bot/orchestrator.py:1453`](../src/ait/bot/orchestrator.py#L1453) — CONFIRMED
A flag that legitimately marks short strangle as "not a defined-risk strategy" (for trade-ranking purposes) is reused by a different function to decide how much capital-at-risk to record for it — and ends up recording **$0** for a position that can realistically lose thousands. The account-wide 20% risk cap would never see that exposure. There's a test for this, but it only checks the case that doesn't reveal the bug.

**18. If long straddle is turned back on, its ML "should I trust this trade" gate isn't controlled by the same master switch as everything else.** [`src/ait/bot/orchestrator.py:1301`](../src/ait/bot/orchestrator.py#L1301), wired via [`src/ait/config/settings.py:195`](../src/ait/config/settings.py#L195) — CONFIRMED
Every other ML-driven trade filter respects a single on/off setting. The one for long straddle (the strategy this branch is nominally named after) doesn't check that setting at all — it's always active whenever the underlying model is trained. Separately, and more importantly: this specific gate has never once been run through a backtest, so there's currently no evidence one way or the other about whether it helps or hurts.

**19. If long straddle is turned back on, its trade outcomes will be mislabeled for the bot's self-learning.** [`src/ait/bot/orchestrator.py:529`](../src/ait/bot/orchestrator.py#L529) — CONFIRMED
A fix that correctly stops the bot from guessing "bullish" or "bearish" from the P&L of a direction-agnostic trade (so it doesn't corrupt the learning signal) covers iron condor and short strangle but was not extended to long straddle, which is equally direction-agnostic (it buys both a call and a put). A profitable straddle from a volatility move would get mislabeled with a directional guess.

**20. A one-off script's win/loss label doesn't match the P&L it fetched.** [`scripts/restate_r16_dailystats.py:44`](../scripts/restate_r16_dailystats.py#L44) — CONFIRMED, negligible impact
Hardcodes "this was a loss" instead of deriving it from the P&L number the script itself just looked up. Low stakes: it's a single-use script already tied to one specific, already-confirmed loss trade, not something that runs repeatedly.

---

### Also noted, not detailed above (lower priority / cosmetic)
- `src/ait/config/settings.py` has stray encoding corruption (a byte-order mark plus ~19 garbled em-dashes) from what looks like a copy-paste mistake — cosmetic only, doesn't break anything, but worth a cleanup pass.
- Sortino ratio's "no losing trades yet" edge case returns `inf` in the live dashboard but `None` in the walk-forward report exporter — inconsistent, not wrong.
- The counterfactual "trades we skipped" log stops recording repeats of the same symbol/strategy/reason after the first one, because the code that was supposed to reset that dedup was removed — this only affects a reporting log, not trading decisions.
- The 2-hour same-symbol re-entry cooldown is checked against the 30 most recent trades account-wide, not per symbol — fine at current trade volume, would quietly stop working if trade frequency grows a lot.
- A strategy's risk budget is set by mutating a shared object rather than passing it as a parameter — safe today because symbols are scanned one at a time, but would be a race condition if that loop is ever parallelized.
- The bot's final "shutting down" notification likely never gets sent on a signal-triggered shutdown (e.g. systemd stop) — an operational nicety, not a money-risk.

---

## What I checked and found solid
The core position state machine (entries, exits, restart recovery), the CAS-guarded status transitions, the price-bounded exit logic, duplicate-instance protection, the day's-worst-loss circuit breaker, currency/FX handling, combo P&L sign conventions, the walk-forward engine's look-ahead-bias fixes (a real prior bug, now fenced), and the bulk of the market-data and watchdog fixes all held up under direct tracing. This PR fixed far more than it broke — the 20 items above are what's left after a genuinely thorough pass, not a sign the whole effort is shaky.

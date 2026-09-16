# W17 — R25 lead probes, batch 1 (2026-09-15)

Converts 5 of R25's 49 single-lens leads into verdicts. Method unchanged from R24/R25:
**two-lens adversarial verification** — a code-reading lens *and* an execution probe; the lead
stands only if both concede. Every probe drove the real function; mocks only at the external edge
(Telegram, the NYSE calendar), never the logic under test.

## Counts — honestly

| | |
|---|---|
| Leads probed | 5 |
| **VERIFIED** | **4** |
| **PARTIAL** (claim stands, stated consequence does not) | **1** |
| REFUTED | 0 |
| **NEW findings surfaced while probing** | **2** (one of them mine, shipped today) |

R24 refuted 28 of 137 at this stage and R25 refuted 6 of its own. Zero refutations here is a
smaller sample, not a better hit rate — batch 1 was deliberately the highest-consequence end of
the list, where claims had already survived the most scrutiny.

**Probe hygiene:** every probe touching supervisor code set `PYTEST_CURRENT_TEST` first, so
`_log`'s file sink stayed suppressed. Verified afterwards: **0 probe lines** reached
`logs/orchestrator.log`. This is the hazard that planted three false lines during W16.

---

## w17-01 · `_alert()` is single-shot, and a failed page burns the throttle — VERIFIED
*(R25 §5 lead 7, promoted to rank 1: the dead-man was retired in W16b, so Telegram is now the
only alerting channel, and A1 — shipped today — rides this exact path)*

**Where:** `src/ait/orchestration/master.py:64-105` (`_alert`), 21 call sites in that file,
including `STATE DB BACKUP FAILED` `:1177` and `GO-LIVE SCORECARD FAILED` `:1480`.

**Claim:** `_alert()` sends once with no retry and no dead-channel accounting, and its callers
discard the `bool` it returns — so a single dropped send is simply lost.

**Evidence (execution probe, real function, urlopen faulted at the network edge):**
```
A. urlopen attempts          : 1          (a retrying impl would be >1)
   _alert returned           : False
B. TELEGRAM_DEAD after 7 failed sends: False   (no marker written)
C. after a FAILED send, _last_gateway_alert set? True
   page attempts in the next 20 min   : 0
```
Contrast `orchestrator.py:4278-4318`, which makes 3 attempts with backoff and writes
`data/TELEGRAM_DEAD` after 5 consecutive definitive failures. None of that exists here.

**Trigger:** any transient Telegram/network failure during a supervisor alert.

**Consequence:** part (C) is the one that matters. `_maybe_recover_gateway` sets
`self._last_gateway_alert = now` *before* calling `_alert()` and ignores the result, so a failed
send still consumes the 30-minute throttle — **one dropped packet buys 30 minutes of silence** on
the page that exists because an 8h36m silent outage happened. With the dead-man retired, nothing
else is watching.

**Suggested fix:** set the throttle only on a successful send (`if _alert(...): self._last_... = now`),
and give supervisor alerts the same retry/dead-channel accounting the bot side already has.

## w17-01b · A1 propagates the throttle-before-send defect — VERIFIED (NEW, and mine)

**Where:** `src/ait/orchestration/master.py`, the escalation block added today, and the
pre-existing sibling at `:422-431`.

**Claim:** the fix I shipped this session for the 8h36m incident consumes its own throttle
regardless of whether the page was delivered.

**Evidence:** probe (C) above — driven against the real `_maybe_recover_gateway`.

**Note:** the pre-existing block at `:422-431` has the identical ordering, so this is a faithfully
copied defect rather than an invented one. It is still a defect, and it is in the code that closes
the incident that motivated W16. **Recommended for immediate fix rather than deferral.**

## w17-02 · A calendar fault makes a dead bot read HEALTHY — VERIFIED (consequence restated)
*(R25 §5 lead 1)*

**Where:** `src/ait/monitoring/ops_health.py:608-614` (`_default_market_open`), routing to
`:646-654`; consumer `src/ait/dashboard/app.py:469-480` (`liveness_panel`).

**Claim:** a calendar-import failure returns `market_open=False`, which takes the `not_rth` branch
and reports `ok=True` — regardless of how stale the heartbeat is.

**Evidence (execution probe, same 6-hour-stale heartbeat in both runs):**
```
calendar healthy, market open : ok=False state=heartbeat_stale
                                "heartbeat 360 min stale during RTH (limit 15 min)"
calendar raises               : ok=True  state=not_rth
                                "market closed - the trading loop legitimately stops..."
```
Driven end-to-end through the real dashboard panel:
```
liveness_panel() normally   : 'stale'
liveness_panel() under fault: 'ok'      (PANEL_OK)
```

**CONSEQUENCE RESTATED.** R25 claimed this would keep the **dead-man ping green**. That consumer
no longer exists — W16b retired the dead-man. But `bot_liveness()` is still consumed by
`dashboard/app.py:475`, so the defect is live with a different victim: **the dashboard's liveness
panel renders green while the bot is six hours dead.** The lead was right about the mechanism and
is now wrong about the blast radius; both are recorded.

**Suggested fix:** an unavailable calendar is *unknown*, not *closed*. Return a third state and
render it as NOT WIRED — the pattern `read_channel_state` already uses at `ops_health.py:924-971`,
which explicitly refuses to show green on missing evidence.

## w17-03 · `combo()` validates nothing, but the exit path cannot currently exploit it — PARTIAL
*(R25 §5 lead 3 — the stated consequence does NOT stand)*

**Where:** `src/ait/broker/contracts.py:90-96`; entry guard `executor.py:652`; exit call
`orchestrator.py:3406`.

**Claim (VERIFIED):** `ContractBuilder.combo()` performs no validation, and the exit path reaches
it without the `_validate_combo_legs` gate the entry path uses.

**Evidence (execution probe — contract objects only, nothing sent):**

| leg defect | entry gate | `combo()` |
|---|---|---|
| action `"HOLD"` | REJECTS | BUILT, `actions=['HOLD','SELL']` |
| `conId=0` | REJECTS | BUILT |
| `ratio=99` | REJECTS | BUILT, `ratios=[99,1]` |
| `ratio=None` | REJECTS | BUILT, `ratios=[None,1]` |
| lowercase `"buy"` | REJECTS | BUILT, `actions=['buy','sell']` |

**Why the consequence does NOT stand:** the exit path's legs are well-formed *by construction*,
not by validation —
`orchestrator.py:3390-3392` aborts on a failed qualification (so `conId` cannot be 0),
the action is always rebuilt as the literal `"SELL"`/`"BUY"`, and `ratio` resolves to the default
`1` because it is never persisted (see w17-03b). R25 implied a malformed leg could reach the wire
via the exit path; **no currently reachable path produces one.**

**What the real risk is:** the invariant holds on three unrelated implementation details staying
true, with nothing enforcing it. A change to leg persistence, qualification, or action handling
would silently put a bad leg on a 4-leg defined-risk structure, and `combo()` has zero tests.

**Suggested fix:** call `_validate_combo_legs` on the exit path too, or move the check inside
`combo()` so both paths inherit it.

## w17-03b · Leg `ratio` is dropped on persistence — VERIFIED (NEW)

**Where:** `src/ait/execution/executor.py:367-375` (writes `legs_json`),
`orchestrator.py:3400` (reads it back).

**Claim:** persisted legs record only `strike`, `right`, `action`, `expiry`. `ratio` is never
written, so every exit rebuilds at `leg.get("ratio", 1)` = 1 regardless of what was sent on entry.

**Consequence:** inert today — every live strategy is 1:1. It becomes a silent structure change
the moment any ratio-spread strategy is enabled: the exit would close a different shape than the
entry opened. Registered now because it is invisible until it is expensive.

## w17-04 · Dashboard "Win Rate" divides close-day wins by entry-day entries — VERIFIED
*(R25 §5 lead 2)*

**Where:** `src/ait/dashboard/app.py:631-642`; writers `orchestrator.py:2620` (entry) and
`:788-791` (close).

**Claim:** the Overview tab reads pre-aggregated `daily_stats`, where the real-close filter cannot
be applied at read time, and computes `Win Rate = trades_won / trades_taken` across two different
populations.

**Evidence:** `trades_taken` increments at **entry** (`:2620`, immediately after the entry
notification). `trades_won`/`trades_lost` increment at **close** (`:788-791`). The expression at
`:640-642` divides one by the other. Applied to the **real rows in `data/ait_state.db`**:
```
2026-04-23  taken=5 won=0 lost=0  -> dashboard shows "Win Rate 0%"
2026-04-24  taken=5 won=0 lost=0  -> dashboard shows "Win Rate 0%"
```
Both are days with five entries and **zero losses**, displayed as a 0% win rate. Boundary cases
from the same expression: `taken=0, won=2` → `"N/A"` (a day of pure wins shows nothing);
`taken=1, won=3` → **`"300%"`**.

**Consequence:** the four headline numbers a human uses to judge the bot are not a win rate. The
real-close asymmetry compounds it: the drain path filters non-real closes
(`orchestrator.py:688-691` consults `is_real_close`), the fill-callback path at `:785-797` does
not — so `$0` sentinel closes can still reach `daily_stats`, which the dashboard cannot filter
because the table is pre-aggregated.

**Suggested fix:** compute the Overview metrics from `trades` with `_REAL_CLOSE_SQL` like every
other panel already does, or split `daily_stats` into entry-day and close-day counters so the
ratio has one population.

## w17-05 · The partial-exit order carries no `orderRef` — VERIFIED
*(R25 §5 lead 4)*

**Where:** `src/ait/bot/orchestrator.py:2849-2850`; invariant documented at
`src/ait/execution/reconciler.py:604-609`.

**Claim:** the partial-exit market order is placed without `orderRef`, breaking the invariant the
reconciler's order matching depends on.

**Evidence:** `OrderBuilder.market()` (`orders.py`) returns a bare `MarketOrder` and sets no
`orderRef`. Every `orderRef` assignment in the tree:
```
executor.py:570, executor.py:798, orchestrator.py:3046, orchestrator.py:3087, orchestrator.py:3674
```
`orchestrator.py:2849` is **not** among them. `reconciler.py:604-609` states: *"Every entry/exit
order has carried its trade_id in orderRef since R17, so: match orderRef == trade_id exactly; fall
back to the symbol/leg heuristic ONLY for orders with no orderRef (pre-R17 orders still in
flight)."*

**Consequence:** the partial exit emits a **post-R17 order that reads as pre-R17**, so the
reconciler falls back to the symbol/leg heuristic — the very matcher R19 replaced because it could
cancel a different trade's working order, "including a live EXIT combo (a stop/TP in flight,
silently killed, and re-killed every ~30s cycle)."

**Reachability note:** the partial-exit ladder is documented as DEAD at 1 contract
(`PLAN.md` dormant inventory), so this is latent until contract sizing exceeds 2. Registered, not
urgent — but it is a correctness gap in a path that wakes up on funding.

**Suggested fix:** `order.orderRef = pos.trade_id` before `place_order`, matching the five sibling
sites.

---

## Proposed fix order

| Wave | Content | Why |
|---|---|---|
| **now** | w17-01b (A1 throttle ordering) | One line, in code shipped today, degrading the fix for the incident that motivated W16 |
| **W18** | w17-01 (`_alert` retry + dead-channel accounting) | Telegram is the only channel since W16b; 21 sites depend on it |
| **W18** | w17-02 (calendar fault → NOT WIRED, not green) | "Green when it should be red" on the only liveness surface left |
| **W19** | w17-04 (Overview metrics) | Wrong numbers on the screen used to judge the bot |
| **W19** | w17-03 / w17-03b / w17-05 | Correctness gaps that are latent today and wake on funding or on a strategy change |

## Batch 2 — re-ranked 2026-09-15

Re-ranked rather than taken in register order, because the board changed underneath the leads:

- **The FIFTH FINDING shipped and deployed** (`df704d7` 22:01; supervisor restarted 22:33:24).
  The register described it as "still uncommitted, not yet deployed" — corrected in place. It is
  running in production, which moves it from queued concern to rank 1.
- **Telegram is the only channel** since W16b retired the dead-man, so anything touching alert
  reliability or silent failure gained weight. w17-01 and w17-02 both turned out to be that class.
- **A2/A3 closed** the `master.py` / `status.py` powershell leads. `gateway.py` and
  `_clear_dashboard_port` were explicitly out of W16 scope and remain open.
- **`ratio`-dropped-on-persistence** was promoted out of the area leads — probed as w17-03b.

| # | Lead | Why here |
|---|---|---|
| **1** | `options_chain.py` trio (a/b/c) | **Live in production.** It alters **strike selection** — strategies choose strikes *by delta*, and delta now derives from a per-contract solved sigma rather than the chain median. Shipped under a data-correctness heading. Also puts a py_vollib Newton solve on the live path per contract, and masks `BelowIntrinsicException` as an ordinary fallback with no counter |
| **2** | `ibkr_client.py:350` | `verify_can_trade` places a **real SPY BUY LMT** with hardcoded symbol/qty/price on **every startup** |
| **3** | `go_live.py:305-310` + `scripts/restate_r16_executions.py:83-87` | The numbers that authorise funding real capital: entry and exit fills pooled with no side filter while the gate comment claims entry-only; `WHERE live_mid > 0` shrinks the sample invisibly; the restate script can zero `live_mid` with no date cutoff and no test coverage |
| **4** | **Fail-open census** (54 live/ops sites, ~25 uncommented) | **Promoted.** w17 probed two members of this class and both VERIFIED — a silent swallow is how the alert path and the liveness gate each failed. With the dead-man gone, silent failure has one less backstop |
| **5** | `_capital_base()` `196000` in `app.py:79-90`, `analytics.py:48-61`, `duckdb_analytics.py:45-58` | The denominator of every return and drawdown percentage, triplicated, each wrapping the declared single authority in `except Exception` |

**Demoted out of the top 5 by probe.** The `.env` `IBKR_ACCOUNT` mismatch is real — `.env` and
`ibkr_client.py:756` name different accounts — but the account **is** enforced at connect
(`_resolve_session_account` `:136`, the "must be one of session_accounts" guard `:149`, and the
`AIT_EXPECT_LIVE_ACCOUNT` paper check `:173-178`), and the bot connected successfully. So this is
**stale comments and test fixtures**, not a trading risk. Reclassified as hygiene: update
`ibkr_client.py:756` and `tests/test_account_currency.py:31` to stop naming an account this
machine does not use.

**Batch 3:** dashboard integrity (five bare-`pass` panels blanking whole analytics sections while
their headers render; `_safe_query` rendering a failed query as an empty book; three independent
real-close implementations), capital chain, orders/contracts dead surface, backtest-display
constants, stale residue (`strategies/base.py:58-61` points at a module the 09-14 fix abandoned),
test isolation.

**Its own wave, not a batch:** triage of the 109 R24 findings + 78 carried-over against today's
source. That is re-verification, not probing — a different shape of work, and folding it into a
lead batch would understate it.

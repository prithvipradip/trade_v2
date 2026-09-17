# Register triage — living document

> **⚠ SUPERSEDED IN PART — read the "R26 CORRECTION" section at the end of this file before
> trusting any CLOSED row.** R26's independent close-out (2026-09-17,
> `reports/full_audit_r26_20260916.md`) re-probed the twenty batch-1 rows and disputes SIX of the
> CLOSED calls below: `concurrency-01`, `concurrency-03`, `regress-live-04`, `logic-exit-risk-02`
> (evidence favours R26), `logic-exit-risk-01` (shared), `ops-substrate-01` (this register's call
> stands, with residuals). The claim "batch-1 fully closed" is RETRACTED. The rows are left as
> written — this is a record of what was concluded and why it was wrong, not a rewrite.

Re-verification of the standing finding register against CURRENT source. Started 2026-09-16,
after the triage had been deferred through W17, W18, W19, W20 and effectively every wave since.

**Posture: PURE TRIAGE, NO FIXES.** Matches R24/R25's own operator-confirmed "REGISTERED, NOT
FIXED". Anything confirmed still-open is recorded and ranked here, feeding later fix waves.
Nothing in the tree was modified while this was written — that is what makes the statuses
trustworthy.

## The universe

| | |
|---|---|
| R24's own standing findings (`full_audit_r24_20260906.md`) | 109 |
| Carried-over OPEN from the prior 123-finding register | 78 |
| R24-marked PARTIAL (fixed, but incompletely) | 19 |
| **Total** | **206** |

**Count discrepancy, stated rather than absorbed:** R24's header says it triaged "123 prior
findings", but its triage section contains **126** rows (78 OPEN + 19 PARTIAL + 28 FIXED +
1 OBSOLETE). Three rows unaccounted for. Not resolved here; flagged so the next batch does not
re-derive it.

## Why this is batched, measured before planning

- Only **22 of 109** R24 findings are mentioned anywhere in PLAN.md after R24 was written.
- **Zero of the 78** carried-over findings are mentioned since.
- Only **17 of 109** have their finding-ID as a fix-marker comment in `src/` or `tests/`.
- **165 of 187 have no trace at all** across ten days and ~18 waves.

The fix-marker convention (`W14 (R24 concurrency-01)` in `state.py`) is real but far from
universal — a lower bound on what's closed, never an upper bound. No grep shortcut exists.

## Status vocabulary

| Status | Meaning | Evidence bar |
|---|---|---|
| `CLOSED` | Defect can no longer occur | Fix in code **and** a test that pins it |
| `CLOSED (untested)` | Old behavior structurally unreachable, but nothing pins it | Show the old path is gone; flag the regression risk |
| `PARTIAL` | Genuinely improved, claim still partly true | Both halves shown |
| `OPEN` | Claim still reads true of today's source | Re-read of the cited `Where:` |
| `OBSOLETE` | Code path no longer exists | Show it's gone |
| `REFUTED` | Original claim was wrong on re-read | Say so plainly |

**The method asymmetry, which is the discipline this whole exercise rests on:** marking OPEN is
cheap; marking CLOSED must be expensive. A wrongly-OPEN finding costs a re-check later; a
wrongly-CLOSED one silently retires a live defect. `CLOSED (untested)` was added during batch 1
because the binary didn't fit reality — see `concurrency-02`.

---

# Batch 1 — the 20 `live_money` + `live_path=True` findings from R24

Assessed 2026-09-16. **All 20 assessed.** 12 CLOSED, 2 CLOSED (untested), 6 OPEN.

## CLOSED (12)

- **CLOSED** `concurrency-01-claim-before-book-is-at-most-once` — W14. `state.py`'s
  `claim_exit_booking` now MARKS the row (`UPDATE ... WHERE claimed_at IS NULL`) instead of the
  committed DELETE-before-booking; the marker comment names the finding. Pinned by
  `tests/test_w14_booking_exactly_once.py` § `concurrency-01`
  (`test_claim_marks_rather_than_deletes`, `test_a_stale_claim_is_re_offered`,
  `test_crash_between_claim_and_booking_is_recovered`).
- **CLOSED** `concurrency-03-cas-gate-misreads-concurrent-promotion-as-terminal` — W14. Pinned
  by `test_w14_booking_exactly_once.py` § `concurrency-03`.
- **CLOSED** `regress-live-01` — W14. The reconciler's $0 not-real-close sentinels no longer
  book as trade outcomes. Pinned by `test_w14_booking_exactly_once.py`
  § `regress-live-01 / pnl-money-01`.
- **CLOSED** `pnl-money-01` — W14, same fix and same test section as `regress-live-01`.
- **CLOSED** `regress-live-04` — W14. Fail-closed breaker no longer auto-resumes and overwrites
  the intact persisted blob. Pinned by `test_w14_booking_exactly_once.py` § `regress-live-04`.
- **CLOSED** `pnl-money-02` — W11. The drain path now applies the identical commission true-up
  as the callback path: `orchestrator.py:729` and `:813` both call `_true_up_commission`
  (defined `:3257`), which the finding's claim said the drain lacked. The W11 comment at the
  call site states the extraction was done precisely so both paths agree.
- **CLOSED** `time-calendar-01` — W15 + W20c. `_sessions_until` now resolves sessions through
  `is_trading_day` (real NYSE calendar) on the primary path, and W20c replaced the fallback's
  `day.weekday() < 5` heuristic — the exact defect claimed — with `trading = False`, which errs
  toward MORE blocking. Both directions of the claim are gone. Covered by
  `tests/test_w20_host_clock_and_calendar_faults.py`.
- **CLOSED** `time-calendar-02` — the post-stop cooldown caller's bare `except Exception: pass`
  is gone. `orchestrator.py:3340+` now logs
  `post_stop_cooldown_check_failed_fail_closed` at ERROR and fails CLOSED (credit strategies
  refused). The claim was a silent fail-OPEN gate; it is now a logged fail-CLOSED one.
- **CLOSED** `regress-live-07` — **OVERLAPPING with `time-calendar-02`, not a duplicate.** Same
  caller, and the one fix closes both, but the claims differ: regress-live-07 claims the
  calendar-library raise plus recurrence (the cutoff cache is written only on success);
  time-calendar-02 claims that surface AND the sqlite busy/locked window AND a malformed
  `exit_time`. R24's own interaction analysis treats them as distinct protections. The logged
  fail-closed caller closes every one of those surfaces, so the status holds on its own evidence.
- **CLOSED** `logic-exit-risk-01` — W7. The `entry_credit + 0.25*wing_width` term that dominated
  the `max()` at $39-60 wings is GONE from both the fallback and the quoted-branch anchor; both
  now route through `_exit_price_ceiling` (orchestrator.py:3413), whose docstring names the
  finding and whose bound is `min(width, max(mark + exit_cross, mult * mark))` — anchored to what
  the structure is currently worth, not to entry credit. Pinned by `tests/test_w7_exit_pricing.py`,
  which replays the real incident numbers (SPY mark 1.4006 / shipped limit 14.04, QQQ 2.9430 /
  18.60) and carries a `TestTheIncidentCannotRecur` class.
- **CLOSED** `logic-exit-risk-02` — W7. The broker's rejection is no longer indistinguishable from
  a cancel: the band price is parsed and honoured, and the re-price is logged as
  `combo_exit_limit_capped_at_broker_band` (orchestrator.py:3679) with a
  `mark_cost + EXIT_CROSS` floor. `tests/test_w7_exit_pricing.py`
  § `TestBrokerBandIsParsedAndHonoured` pins it against the VERBATIM 08-31 IBKR warning text
  (`"...more aggressive than 1.745..."`). Leg-synthesis for the silent-Bag case also landed
  (`combo_exit_limit_from_legs`, :3595), covering the finding's root-cause branch.
- **CLOSED** `ops-substrate-01` — W13, and the ledger carries a dedicated entry for it
  (PLAN.md:1977, "GATEWAY AUTO-RECOVERY (ops-substrate-01, materialized TWICE)"). The absorbing
  state is gone: the bot-down branch now calls `_maybe_recover_gateway(gw_port, now)`
  (master.py:882) under a comment naming the finding — "deferring FOREVER is the absorbing
  state" — with a recovery budget that resets on `gateway_recovered`. Pinned by three files:
  `test_w13_gateway_recovery.py`, `test_w18_gateway_port_and_throttle.py`,
  `test_w16_platform_gaps.py`. The finding's other half (IBC `AutoRestartTime=` empty in
  `C:/IBC/config.ini`) is a Windows-substrate claim that is moot on this macOS host.

## CLOSED (untested) (2)

- **CLOSED (untested)** `concurrency-02-claim-exception-books-anyway-then-drain-double-books` —
  W14. `_claimed = True` no longer exists as executable code anywhere in `src/` (the only
  remaining occurrence is inside the comment describing the old behavior); the except branch now
  logs `exit_booking_claim_failed_deferred_to_drain` and `continue`s, leaving the durable outbox
  row for the drain. The defect is structurally unreachable — a stronger guarantee than a test —
  but **no test exercises the claim-raises path**, so a future edit could reintroduce it
  unnoticed. Recommend a regression test; not a fix, so out of this wave's scope.
- **CLOSED (untested)** `regress-live-02` — **TRUE DUPLICATE of `concurrency-02`**, verified by
  reading both in full: identical `Where:` (orchestrator.py:739-742, `_claimed = True`), identical
  claim, identical probe shape. The fix marker names both
  (`W14 (R24 concurrency-02 / regress-live-02)`). Same evidence, same testing gap.

## OPEN (6)

- **OPEN** `data-integrity-07` — claim still reads true. `orchestrator.py:2695`
  `stats.trades_taken += 1` still fires on every order PLACED, and there is **no decrement
  anywhere**: `trades_taken` appears only at `:2017` (budget read), `:2449` (log), `:2695`
  (increment) and `:3886` (summary), with **zero references in `reconciler.py`** — so the
  stale-pending sweep still cannot give the budget back. The intraday entry budget is still
  consumable by phantom trades.
- **OPEN** `regress-live-06` — claim still reads true. `_credit_floors_hold`
  (`executor.py:1411`) exists but is invoked **only at reprice** (`:1390`, logging
  `credit_floor_breached_at_reprice`). A grep across the whole `_execute_multi_leg` initial
  placement path returns **no floor check at all**, exactly as the finding claimed.
- **OPEN** `params-trading-01` — **the sharpest finding in this batch**, and all three of its
  sub-claims re-read as true today. (a) The sizer's hard viability cap is intact:
  `position_sizer.py:166` still returns 0 contracts on
  `cost_per_contract > max_position_value`. (b) `_launch_size_coherence_check` still never reads
  `max_position_pct` — a grep of the whole function finds no occurrence — so the go-live
  self-test still cannot see the cap that will actually refuse every trade. (c) Worst of the
  three, and worth pulling out: `manager.py:615` is still
  `final_contracts = max(1, int(final_contracts * haircut))`. When the sizer has returned 0 and
  `correlated_count > 0`, that computes `max(1, 0) = 1`, and the `if final_contracts <= 0` guard
  on the next line never fires. **A fail-closed 0-viability verdict is silently converted into a
  live 1-contract position — and only in the correlated case, which is exactly when it is least
  wanted.** The suggested `max(0, ...)` was never applied.
- **OPEN** `logic-entry-01` — claim still reads true, and it is the same self-test as above.
  `_launch_size_coherence_check` (orchestrator.py:2031) still models a minimum viable wing as
  `min_width = max(1.0, min_credit / 0.45)` with
  `min_viable_cost = (min_width - min_credit) * 100` — the $1.56 wing the real builder never
  constructs — and still does not execute `IronCondor.generate_signals` against a real chain.
  So the pre-funding gate can still report COHERENT for budgets that deterministically emit zero
  signals.
- **OPEN** `logic-exit-risk-03` — the primary claim stands: `_post_stop_cooldown_until`
  (orchestrator.py:2221) still queries `WHERE symbol=?` (:2244), with no cluster scoping, while
  `SECTOR_GROUPS['index']` treats SPY/QQQ/IWM as one 0.95 cluster. A SPY touch stop still leaves
  QQQ and IWM free to enter the same move on the next scan. PARTIAL CREDIT, recorded precisely:
  the finding's secondary half — "the adjacent fail-open means a DB hiccup removes the cooldown
  entirely without a log line" — is now CLOSED by the same fix that closed `time-calendar-02`.
- **OPEN (in a CHANGED FORM — the most interesting result of this batch)** `docs-policy-04`.
  The claim was that the RUNBOOK tells the operator to stop the bot processes but never to stop
  `keeper_ait.bat`, so the supervisor rebuilds what was just closed. Re-verifying it moved the
  finding's `Where:`:
  - The RUNBOOK's manual-intervention step 2 (`docs/RUNBOOK.md:41-42`) STILL says "or stop the
    bot processes if you're closing the whole book", and still names no supervisor.
  - `data/HALT` first (:12, :40) genuinely mitigates the ENTRY side — it is durable and survives
    a relaunch. But the finding's hazard is the EXIT engine and the startup reconcile returning
    mid-procedure, and HALT explicitly does **not** stop those: the RUNBOOK's own wording is
    "freezes entries; exits keep managing."
  - The mechanism that relaunches is no longer the one cited. W16b retired the POSIX keeper and
    the host is macOS, so `keeper_ait.bat` is inert — but `master.py` IS the supervisor now, and
    `combined_health_check` runs on a **2-minute interval** (master.py:1959) calling
    `bot.health_check()` → `self.start()`. The operator's "stop the bot processes" is therefore
    undone within ≤2 minutes, by a different component than the finding names.
  - `docs/RUNBOOK.md:93` still offers `keeper_ait.bat` as a restart path, and `keeper_ait.bat`
    is still in the tree — stale on this host.
  So: still OPEN, same hazard, new mechanism, and the doc is now wrong in a second way.

## Batch 1 observations worth carrying forward

1. **The register contains duplicates — one confirmed here, after a self-correction.** I first
   recorded two pairs; reading both in full showed only `concurrency-02`/`regress-live-02` is a
   true duplicate. `time-calendar-02`/`regress-live-07` share a caller and a fix but claim
   different failure surfaces. The lesson outlives the count: the method asymmetry above guards
   against over-calling CLOSED, and "duplicate" is that same hazard in other clothes — collapsing
   two findings retires a claim nobody re-checked. A dedup pass is still worth doing, but it must
   compare `Where:` **and** `Claim:` in full, never a shared code location.
2. **The "already fixed, never marked" rate is high in this tier** — 14 of 20 came back CLOSED
   or CLOSED (untested), from W7, W11, W13, W14, W15 and W20c. That does **not** generalize:
   this batch was chosen as the highest-severity slice, which is precisely where the fix waves
   concentrated. The research_validity and cosmetic tiers have had far less attention and
   should be expected to close at a much lower rate. The corollary is the real lesson — the
   fix-marker convention badly understates what is done, so the register is a poor picture of
   the system's actual safety, in BOTH directions: it lists 14 defects here that no longer
   exist, while `params-trading-01`'s haircut override sat unflagged in the same tier.
3. **Two OPEN findings converge on one function.** `logic-entry-01` and `params-trading-01` both
   fail at `_launch_size_coherence_check` — the self-test the go-live recipe tells the operator
   to trust. It models a wing the builder never builds AND cannot see the sizer cap that will
   refuse the trade. Whatever fix wave takes these should take them together.
3. **`CLOSED (untested)` is a real and recurring category** and the most actionable output of
   this batch: the defect is gone, but nothing stops it coming back.

## Batch 1 verification

**Dangerous-direction spot-check (3 CLOSED rows, per the wave plan).** The bar was not "a test
exists" but "the cited test would FAIL against the pre-fix behavior." All three checked tests
turn out to be constructed as negative controls, with the pre-fix defect named in the assertion
itself:

| Finding | Test | What breaks it pre-fix |
|---|---|---|
| `concurrency-01` | `test_claim_marks_rather_than_deletes` | `assert n == 1, "claim destroyed the obligation (the pre-W14 defect)"` — the committed DELETE leaves `n == 0` |
| `regress-live-01` / `pnl-money-01` | `test_zero_dollar_sentinels_do_not_reset_the_loss_streak` | `assert ... _consecutive_losses == 2, "streak was reset"` — pre-fix `record_trade_result(0.0)` zeroes it |
| `regress-live-04` | `test_the_intact_streak_survives_a_locked_relaunch` | `assert recovered._consecutive_losses == 2, "the streak was destroyed"`, directly below `crashed._persist()  # pre-W14: wrote zeros` |

Test files cited by CLOSED rows, all green on today's tree:
`tests/test_w14_booking_exactly_once.py` **15 passed**;
`tests/test_w7_exit_pricing.py` + `tests/test_w13_gateway_recovery.py` **34 passed**.

**Count reconciliation.** 20 batch-1 rows + 186 untouched = 206 nominal IDs. Within batch 1:
20 assessed = 12 CLOSED + 2 CLOSED (untested) + 6 OPEN, 0 deferred. Two caveats kept visible
rather than absorbed: (a) the **effective** universe is 205 distinct defects, since
`concurrency-02`/`regress-live-02` are one defect under two IDs — and more such pairs are likely
in the untouched 186; (b) the 126-vs-123 discrepancy in R24's own header is still unresolved.

**No-touch proof.** This batch produced a document, not a diff. `src/` and `tests/` were read
only. The full suite at the end returned **1811 passed / 43 skipped / 1 failed** — the failure
being the known local-`.env` `test_r19_config_authority` case — which is byte-for-byte the
baseline PLAN.md recorded at the close of W23. Nothing was touched.

---

## W25 ADDENDUM — registered, not fixed: the cap-grid skew that masked params-trading-01

Recorded during the W25 fix wave (2026-09-16). This is a **new finding**, not a batch-1 row, and
it is registered rather than fixed because changing either knob moves live sizing and wants its
own wave with a backtest.

**Claim.** Two knobs govern the same quantity — "how much capital may one trade put at risk" —
and nothing ties them together:

| knob | value | enforced at | measures |
|---|---|---|---|
| `risk.max_position_risk_pct` | 0.03 | `manager.py:512` (gate 6b) | `request.max_loss` vs NLV |
| `positions.max_position_pct` | 0.05 | `position_sizer.py:166` | per-contract cost vs NLV |

**Why it matters, and it is not the obvious reason.** The ordering `3% < 5%` is the *only* thing
that kept params-trading-01 off the live path. Gate 6b rejects every defined-risk request before
the sizer can return its 0-contract verdict, so the `max(1, ...)` fail-open at `manager.py:615`
was unreachable whenever `max_loss` was set — which, for an iron condor, is always
(`iron_condor.py:318-322` refuses to emit a structure with `max_loss <= 0`). That containment is
**incidental**. Raise `max_position_risk_pct` above `max_position_pct`, or lower the latter, and
the fail-open becomes live on the normal IC path with no other code change. W25 removed the
fail-open itself, so the hazard is closed today; this entry exists so a future wave that tunes
either knob knows what it is standing on.

**Verified by execution, not by reading.** Probe drove the real `RiskManager.validate_trade` with
the real `PositionSizer`, real `CorrelationGuard` and real `config.yaml`:

- $2.1k, IC with `max_loss=165`, one correlated position open → rejected at gate 6b
  (`position risk $165 exceeds 3% account limit $63`); the sizer's 0-verdict never reached :615.
- $2.1k, same request with `max_loss` falsy → gate 6b skipped, sizer returns 0
  (`single contract exceeds max_position_pct cap`), and pre-fix the result was
  **approved=True, position_size=1, max_risk=0.0**.

**Second-order defect in the same evidence, also registered:** on that approval `max_risk` came
back as **0.0**, because `TradeValidation.max_risk` is `size.max_risk_dollars` and the 0-contract
verdict carries zero risk. A position approved as risk-free is invisible to the aggregate
capital-at-risk cap at `manager.py:529`. W25 pins this behaviorally (`approved => max_risk > 0`)
but the underlying coupling — validation risk sourced from a sizer result that may not describe
the approved size — is untouched.

**Status:** OPEN (registered). Not scheduled. Do not tune either knob without reading this entry.

---

## W27 ADDENDUM — registered, not fixed: 21 unregistered `AIT_*` env keys, and the guard that cannot see them

Recorded during the W27 fix wave (2026-09-17). A **new finding**, not a batch-1 row. Registered
rather than fixed by operator decision: W27 does not depend on it (its clamp binds whatever
`AIT_ENTRY_LADDER_START` is set to), and converting 1 of 21 would make the guard look more complete
than it is.

**Claim, measured not estimated.** 22 `AIT_*` keys are read via raw `os.environ.get` in `src/`.
**21 are absent from `CONTRACT_DEFAULTS`.** R19c's rule is "no production module may carry a private
default for a contract key — a promotion must edit exactly one place and take effect everywhere",
and `tests/test_r19_config_authority.py::test_no_raw_environ_reads_of_contract_keys` exists to
enforce it.

**THE GUARD BUILDS ITS REGEX FROM `CONTRACT_DEFAULTS`.** So it polices only keys already registered,
and is structurally blind to every key that was never added — which is the entire population above.
The enforcement's coverage is defined by the thing it is meant to police. This is the same
guard-hole shape as W26 [4]'s `date.today()` finding (W20's naive-clock guard covers
`datetime.now()` and not `date.today()`), and it is worth naming as a pattern: **a class guard keyed
on a registry cannot catch what was never registered.**

**16 carry a competing local fallback literal** — the exact R19c violation:

| key | site | private default |
|---|---|---|
| `AIT_IC_DELTA_MIN` | `ait/strategies/iron_condor.py:203` | `"0.15"` |
| `AIT_IC_DELTA_MAX` | `ait/strategies/iron_condor.py:204` | `"0.30"` |
| `AIT_IC_EM_MIN` | `ait/strategies/iron_condor.py:224` | `"0.6"` |
| `AIT_IC_EM_MAX` | `ait/strategies/iron_condor.py:225` | `"1.3"` |
| `AIT_IRON_CONDOR_IV_FLOOR` | `ait/strategies/iron_condor.py:144` | `"15"` |
| `AIT_CREDIT_TOUCH_STOP` | `ait/execution/portfolio.py:570` | `"1"` |
| `AIT_ENTRY_LADDER_START` | `ait/execution/executor.py:755` | `"0.25"` |
| `AIT_ENTRY_ORDER_TIMEOUT` | `ait/execution/executor.py:38` | `"240"` |
| `AIT_IV_RANK_CAP` | `ait/risk/manager.py:435` | `"85"` |
| `AIT_MAX_PER_EXPIRY` | `ait/risk/manager.py:470` | `"3"` |
| `AIT_LIQ_MAX_SPREAD` | `ait/bot/orchestrator.py:332` | `"0.15"` |
| `AIT_LIQ_MIN_VOLUME` | `ait/bot/orchestrator.py:333` | `"10"` |
| `AIT_BT_CALIBRATED_SPREADS` | `ait/backtesting/engine.py:296` | `"1"` |
| `AIT_BT_TOUCH_STOP` | `ait/backtesting/engine.py:404` | `"1"` |
| `AIT_GATEWAY_START_SCRIPT` | `ait/orchestration/master.py:143` | `r"C:\IBC\StartGateway.bat"` |
| `AIT_STATE_DB` | `ait/monitoring/ops_health.py:789` | `str(DEFAULT_STATE_DB)` |

**5 are bare presence reads** with no competing default, so they are a WEAKER case and should not be
lumped in with the 16 — recorded separately rather than inflating the count:
`AIT_ACCOUNT` (`broker/ibkr_client.py:134`), `AIT_EXPECT_LIVE_ACCOUNT` (`:173`),
`AIT_SIMULATED_CAPITAL` (`broker/account.py:216`), `AIT_ALLOW_AFTER_HOURS`
(`execution/executor.py:340`), `AIT_HEALTH_STATE` (`monitoring/ops_health.py:793`).

**Why it matters beyond hygiene.** The load-bearing ones govern live trading decisions with a
private default that `config.yaml` cannot override and the divergence report cannot see:
`AIT_IC_DELTA_MIN`/`MAX` and `AIT_IRON_CONDOR_IV_FLOOR` decide STRIKE SELECTION and entry gating for
the only live strategy; `AIT_CREDIT_TOUCH_STOP` arms the primary loss exit; `AIT_IV_RANK_CAP` and
`AIT_MAX_PER_EXPIRY` are risk gates; `AIT_ALLOW_AFTER_HOURS` gates market-hours enforcement. The
wing_k incident R19 was written about — four disagreeing sources for one number — is the same shape.

**Status:** OPEN (registered). Wants its own wave, alongside the `date.today()` census from W26 [4].
The two are the same class of problem — an enforcement mechanism whose reach was never measured —
and are probably one wave, not two.

---

## R26 CORRECTION — six CLOSED rows disputed, and the method that produced them

Written 2026-09-17 after R26's independent close-out (`reports/full_audit_r26_20260916.md`,
`reports/r26_integration_20260917.md`) re-probed all twenty batch-1 rows at HEAD. Appended, not
edited: the original rows above stay exactly as written, because a register that quietly rewrites
its own wrong calls teaches nothing.

**Headline: the claim "batch-1 fully closed" is RETRACTED.** Fourteen of twenty rows are agreed.
Six are disputed:

| Row | This register said | Residual that still reproduces | Evidence favours |
|---|---|---|---|
| `concurrency-01` | CLOSED (W14 claim-as-mark, 3 tests) | booking is at-least-once with non-idempotent writes — one −$272.29 stop books as **−$544.58 / 2 losses** after a mid-booking failure | **R26** |
| `concurrency-03` | CLOSED (W14 CAS re-read) | `_update_trade_partial` still treats a promotion as terminal; alert key still single | **R26** |
| `regress-live-04` | CLOSED (W14 persist latch) | `is_tripped` lifts the fail-closed trip after one window **without re-reading the store** | **R26** |
| `logic-exit-risk-02` | CLOSED (W7 band parser) | the parser is fed from `trade.log`, which ib_insync 0.9.86 never populates for a 202 — **the mechanism is dead live** | **R26** |
| `logic-exit-risk-01` | CLOSED (W7 `_exit_price_ceiling`) | stale-mark clause never built: mark >900s + silent BAG → BUY LMT 39.00 on a 1.40 mark | shared |
| `ops-substrate-01` | CLOSED (W13 recovery) | recovery does fire; residuals: no test pins `health_check → launcher`, IBC `AutoRestartTime` empty | **this register**, with residuals |

### Two of these were verified independently here, not taken on trust

**`logic-exit-risk-02` — confirmed, and the refutation was already in our own ledger.**
`grep -rn "errorEvent" src/` returns **nothing**. `parse_price_band_reject` can only read an order's
`trade.log` messages, and ib_insync 0.9.86 classes error 202 as a warning that never lands there.
The mechanism cannot fire live. Worse: `PLAN.md:2469-2476`, written 2026-09-14 — **two days before
this triage** — records exactly that, ending "Fix: capture errorEvent 201/202 per orderId in
IBKRClient and feed it to the parser." The ledger already contained the refutation when this
register marked the row CLOSED.

**The `regress-live-04` test cited at line ~220 of this file cannot fail.** R26's F7 flags it; it is
confirmed here by execution. In
`test_w14_booking_exactly_once.py::test_the_intact_streak_survives_a_locked_relaunch`, the `crashed`
breaker is attached to a **MagicMock**, while the assertion re-reads the **real** `StateManager`.
Probed directly: after `crashed._persist()`, `circuit_breaker_state` in the real store is
**unchanged** — the crash cannot reach the store the assertion reads, so the assertion holds
whatever `_persist()` does. The comment `# pre-W14: wrote zeros` describes something the test does
not exercise. **This register's §5 promoted that test to a dangerous-direction negative control
("the cited test would FAIL against the pre-fix behavior"). That promotion is withdrawn.**

### The method gap — the part worth keeping

This register's §3 set the right bar ("marking CLOSED must be expensive: the fix in code AND a test
that pins it") and then applied it too narrowly. Two failures, both systematic:

1. **It verified the FIX, never re-ran the FINDING.** R26's §2 observes that on every disputed row,
   *the finding's own acceptance test would have caught the residual*. Confirming a fix exists and
   confirming the reported defect no longer reproduces are different acts; only the second closes
   a finding.
2. **It read assertion strings as evidence of what a test exercises.** §5 claimed three rows were
   spot-checked "in the DANGEROUS direction" by quoting their assertion messages
   (`"the streak was destroyed"`). An assertion message is documentation, not behaviour — as the
   `regress-live-04` tautology above shows. The check required is: revert the fix, or trace the
   wiring, and watch the test go red.

Neither is fixed by trying harder. The durable form: **a finding closes when its own reproduction
steps stop reproducing**, run against the current tree — not when the patch and a green test exist.

**Status of the six:** reopened. `concurrency-01`, `concurrency-03`, `regress-live-04` sit in R26's
F2 (owner: R26 author, two tests already failing). `logic-exit-risk-01/-02` sit in F1 (same owner,
gated "before the next take-profit exit"). `ops-substrate-01`'s residuals sit in F6.

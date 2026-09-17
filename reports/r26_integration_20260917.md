# R26 Integration — the completed R24 audit reconciled with Ahmed's W16–W27

**Date:** 2026-09-17 (pre-open, written 08:30 ET). **HEAD:** `8faed40` on `feature/event-straddle`.
**Companion documents:** `reports/full_audit_r26_20260916.md` (the 3,275-line close-out with every
finding block and probe transcript), `reports/register_triage_20260916.md` (Ahmed's batch-1 register),
`reports/full_audit_r25_20260914.md` (his structure audit), `reports/full_audit_r24_20260906.md`.

This document is the single page that says where the two audits agree, where they disagree and who
the evidence favours, what his W21–W27 changed underneath the audit, what is deployed on which box,
and the complete outstanding list with owners. Nothing here is new evidence — every claim traces to
a probe in the R26 report or to a test that was run on this (Windows, live) box today.

---

## 1. The register is fully triaged — for the first time

203 standing findings (109 R24 + 94 August OPEN/PARTIAL), each triaged at `57ba9aa` and attacked by
two independent lenses (code-reading, execution-probe); one tie-break; zero code-vs-probe
disagreements. Survivors re-tested at `870699d` after W21–W25 landed mid-audit.

| Status | Rows | Notes |
|---|---|---|
| FIXED, executing test fails on revert | **8** | `regress-live-01`, `regress-live-07`, `time-calendar-01`, `time-calendar-02`, `pnl-money-01`, `pnl-money-02`, `logic-exit-risk-04`, `fail-direction-02` |
| Claimed CLOSED by his W26/W27 after the audit | **3** | `data-integrity-07`, `logic-exit-risk-03` (graduated cluster window — a deliberate deviation, see §3), `regress-live-06`. His test files pass on this box (126/126 with the tests they could have broken). Not yet independently probed here. |
| PARTIAL | **40** | headline mechanism closed, residual reproduces |
| STILL_OPEN | **152** | 155 minus the three above |

**Live-money still open in some form: 35** (38 minus the three). Ten days and ~20 waves of fixing
closed 11 register rows; the waves closed clusters, not the register.

## 2. Where the two audits disagree, settled by execution

Ahmed's batch-1 triage covered the 20 `live_money` R24 findings. We agree exactly on 14. On six he
recorded CLOSED; the R26 close-out re-probed each at HEAD:

| Finding | His CLOSED rests on | Residual that still reproduces | Evidence favours |
|---|---|---|---|
| `concurrency-01` | W14 claim-as-mark, three tests | booking is now at-least-once with non-idempotent writes: one −$272.29 stop books as **−$544.58 / 2 losses** after a mid-booking failure | mine |
| `concurrency-03` | W14 `_update_trade_filled` CAS re-read | `_update_trade_partial` still treats a promotion as terminal (row reads qty 2 vs 1 live); alert key still single | mine |
| `regress-live-04` | W14 persist latch, three tests | `is_tripped` lifts the fail-closed trip after one window **without re-reading the store**; latch never clears in-process | mine |
| `logic-exit-risk-02` | W7 band parser + `exit_band_<tid>` | the parser is fed from `trade.log`, which ib_insync 0.9.86 **never populates for a 202** — `parse → (None, None)`, `grep errorEvent src/` → nothing; **mechanism is dead live** | mine |
| `logic-exit-risk-01` | W7 `_exit_price_ceiling` (real: 14.04 → 2.10) | stale-mark clause was never built: mark >900s + silent BAG → **BUY LMT 39.00 on a 1.40 mark** | shared |
| `ops-substrate-01` | W13 recovery from the bot-down branch | recovery genuinely fires on the Windows host; residuals: no test pins `health_check → launcher`, IBC `AutoRestartTime` still empty, and part of my objection rested on a false premise about his host | **his**, with residuals |

On no row was his CLOSED simply wrong; on every row the finding's own acceptance test would have
caught the residual. His register still reads CLOSED on all six and "batch-1 fully closed" as of
`f146e72` — this document and the R26 report are the correction.

## 3. His waves W21–W27 against the audit

| Wave | What it did | Audit result | Deployed |
|---|---|---|---|
| **W21** tz-aware trade timestamps | writer stamps `now_et()`; restate script for the corpus | **Backfill never applied on this box: 0 of 30 rows aware.** Restate script returns 0 on failure and has no test. 8 more findings (dashboard UTC bucketing, mirror-lag reader, pace window). **Test file uses POSIX-only `tzset` → 8 failed / 5 errors on Windows** — the same class fixed in his W20 tests the day before. | his box; here since 07:31 |
| **W22** declared manual positions | Telegram ask-once flow; reconciler treats declared legs differently | **5 live-money findings, one root cause: irreversible action on `ib.positions()`'s cache** — his own R25 `arch-01` producing five instances. Declared position invisible to every risk gate despite a docstring claiming otherwise. Bare "yes" recorded against a dead position. | both boxes; dormant while flat |
| **W23** scheduler tz fallback | three bugs | module-scope import makes the supervisor unimportable if `pandas_market_calendars` fails; docstring cites evidence the deployment contradicts | both |
| **W24** market-open race | closed in `wait_for_phase` | `get_current_phase` ignores its own `now` for half-days | both |
| **W25** go-live sizing | haircut guard; coherence check rewrite | `params-trading-01`: the money-losing half **CLOSED, proven by revert**; self-test half PARTIAL. `logic-entry-01`: my residual half **REFUTED** (fake $1.56 wing gone), rest PARTIAL. Breaker arm still fail-open, sized from a $1 probe wing. | both |
| **W26** budget give-back + cluster cooldown | `data-integrity-07`, `logic-exit-risk-03` | Sibling cooldown is **120 min, not one trading day** (`risk.post_stop_cluster_cooldown_minutes`, 0 = old behaviour) — a stated deviation, reasoned (strict scope darkens the whole single-cluster book). Registered, not fixed: 49 real `date.today()` sites. | his box only |
| **W27** credit floor at placement | `regress-live-06` | Floor enforced at placement; **exits exempt** (`is_credit and not is_exit`) — verified, because `_execute_multi_leg` serves exits. Registered, not fixed: 21 `AIT_*` keys outside `CONTRACT_DEFAULTS`, 16 with private literals (superset of `hardcoded-live-04`). | his box only |

Two developers shipped the same Windows-only test defect (`time.tzset`) on consecutive days; the
suite is green on macOS and red on the box that trades.

## 4. New findings: 53, of which 17 live-money

Full blocks in the R26 report §4. The shape:

- **My W7 exit rebuild — 6 live-money (+1 from the backlog), one broken pricing path.** Band text never
  reaches the parser; no-quote fallback prices above IBKR's band; band cap rounds 1.745 → 1.75; mark
  ceiling clamps *below* the ask on cheap positions (mark 0.60 → limit 0.90 vs 1.08 ask — the regime
  of a $2.1k funded account's condors); zero-bid wing aborts synthesis; one-sided leg priced from the
  wrong side; full wing width still reachable on a stale mark. `test_w7_exit_pricing.py` passes 20/20
  with all of them live.
- **His W22 — 5 live-money, one root cause** (§3).
- **My W14 — 2 live-money**, both with a failing test already written: double-book on mid-booking
  exception; breaker never re-reads after a transient unreadable state.
- **W16 / W25 — 2 live-money:** foreign-bot scan fails open on an unreadable process table; W25 breaker
  arm fail-open.
- **Refuted this round:** 7 of the 72 W7–W20 candidates and 9 of the 40 W21–W25 lens outputs.

## 5. Today's live incident (2026-09-17, pre-open)

- 23:45 Gateway exits (IB nightly logoff; `AutoRestartTime` empty). Down all night; the bot's
  alive-but-disconnected state is invisible to the supervisor.
- 07:30 retrain; 07:31:32 scheduled fresh-models bot restart; supervisor's `ensure_gateway()` relaunches
  the Gateway (07:31:35); new bot 10212 starts 07:31:48 **from the current working tree = W21–W25**.
- **07:42:21 the operator closed the Gateway window by mistake** (clean "Normal exit"). The supervisor
  kept logging `bot_healthy` because it watches the bot process, not the port; the bot's disconnect
  handler only flags the socket and reconnects lazily at its next scheduled action. At 08:32:30 the bot woke, computed the next wait (1649 s to the 09:00 pre-market phase) and went back to sleep **without reconnecting** — the first `ensure_connected()` runs when the phase starts at 09:00, thirty minutes before the open.
- 07:52:38 Gateway relaunched via IBC (`Start-Process StartGateway.bat`); port 4002 up.
- This is the `ops-substrate-01` residual seen live: a Gateway that dies while the bot is alive has no
  automatic recovery path until the bot's next cycle. Harmless today (book flat, operator present);
  with open positions it blinds the exit engine.

## 6. Deployment state — two boxes, two stacks

| | This box (Windows, DUN603821, canonical) | Ahmed's box (macOS, separate paper acct) |
|---|---|---|
| Supervisor | `57ba9aa` code (started 09-16 08:19) | W27 (58879) |
| Bot | **W21–W25** (10212, started 07:31:48) | W21–W27 |
| Working tree | `8faed40` (W26/W27 present, not running) | `8faed40` |
| `AIT_SIMULATED_CAPITAL` | **not set** — sizes on real $198k paper NLV | 10000 (provisional probe value) |
| Suite | see §8 | 1882 passed / 1 known local-`.env` failure |

Which box is canonical for the verdict sample is an operator decision (§9 E5).

## 7. Outstanding work — complete list with owners

Fix sets are labelled **F1–F8** (not wave numbers — W26/W27 were consumed while the report was
written). Every fix needs an executing test that fails before the change.

**A. Immediate**
1. Confirm the bot reconnects at 09:00 when the pre-market phase starts (watcher armed to 09:10) — else one supervisor-driven restart. [me]
2. Fix the W21 `tzset`/cp1252 test breakage on Windows (same seam as W20). [me]
3. Decide whether W22's Telegram poller stays enabled here until F3 lands. [operator]
4. Set IBC `AutoRestartTime` so the nightly 23:45 exit restarts itself. [me]
5. Deploy W26/W27 to this box outside RTH (they are in the tree, not the process). [me, with go-ahead]

**B. Fix sets**
- **F2 booking & breaker** [me, ~½ day, two tests already failing]: double-book on mid-booking exception; breaker re-read after transient unreadable; `_update_trade_partial` CAS gate + per-trade alert key.
- **F1 exit rebuild corrective** [me, 1–2 days, before the next take-profit exit]: wire `ib.errorEvent` → band parser; band cap strictly below band; no-quote fallback under the band; zero-bid wing; one-sided leg side; low-mark ceiling with per-attempt stepping; stale-mark wing-width fallback; instrument the exit poll.
- **F3 W22 cache class** [Ahmed]: fix the `ib.positions()` seam once, then the six W22 findings.
- **F4 W25 sizing self-test** [Ahmed]: breaker arm; probe budget; `generate_signals` executed; dead ratio arm; two-knob cap grid.
- **F5 W21 finished** [Ahmed; item 7 by me]: restate script post-condition + tests, then `--apply` with the bot stopped; simulated-capital leak into the reporting denominator; dashboard/mirror/pace readers; Windows test fixture.
- **F6 ops substrate** [me]: foreign-bot scan fail-open; escalation page starved by the shared throttle (test failing); W23 import; wedged-JVM budget; probe output in forensic log; singleton page throttle; half-day phase; **port check on the bot-healthy branch** (today's incident); a test that drives `health_check → launcher`.
- **F7 test honesty & ledger** [both]: W14 intact-streak, W13 reset, W18 `getsource`, W22 background-task, W25 negative controls; the ledger lines resting on `getsource`; his six CLOSED rows.
- **F8 research validity & config homes** [me]: compute `iv_is_guessed`/`atm_iv`/`expected_move` after the per-contract solve; stale exit-quality tuple; r=0.05 / dividend yield; `range.pkl` 9-vs-30 daily flap; 900s cutoff, credit-ratio literal, gateway-script authority; the 21 `AIT_*` keys.

**C. Register live-money items outside F1–F8 (23):** `regress-live-05`, `docs-policy-04`,
`bot-day-03/04/05/08/09/10`, `money-flow-02/05`, `trade-life-partial-slice-pnl-erased`,
`trade-life-promotion-ignores-broker-quantity`, `trade-life-commission-trueup-gate-measures-wrong-quantity`,
`trade-life-credit-econ-floors-signal-time-only`, `fail-direction-03/10`,
`model-vs-reality-ex-div-assignment-void`, `model-vs-reality-margin-model-unimplemented`,
`policy-vs-impl-4/7`, `time-authority-2/4`, `concurrency-4`, `numeric-pairs-07`, `set-membership-6`,
`money-flow-01`.

**D. Research / strategy**
1. **The research re-run** — the engine takes 2 trades where live took ~10, enters at 14 DTE vs live's median 18, touch-stops 22/57; no parameter can be chosen on it until the entry model matches live. Largest open strategic item.
2. Take-profit ladder — parked until D1 (direction: higher/none beats the ladder; every variant loses on the current engine).
3. Blackout inversion — blocks the safest entries, holds the riskiest through events; 33% of sessions closed; touch-rate benefit z=0.6–1.0. Decide after D1, not mid-sample.
4. Range model horizon (9 vs 18-DTE reality) and the `min_edge_over_baseline` flap.
5. IWM dark since 07-17 on IV rank.
6. Pace 1.2 closes/wk vs 3–4 → 50-close read ~30 weeks out; only D3 moves it.

**E. Process**
1. Pull before push; whoever ships a wave runs the suite on the Windows box before calling it green. [operator + Ahmed]
2. PLAN.md conflicts on every concurrent session — per-session append files or pull-then-append. [both]
3. Reconcile his register (batch 2: 186 rows, his dedup pass) with the 203-row triage into one truth. [Ahmed]
4. Untouched by anyone: R25's 49 unverified leads, the fail-open census (12-site shortlist, zero probed), T1-b/T1-c line-by-line, `arch-01..04`.
5. Decide the canonical box and align `AIT_SIMULATED_CAPITAL` and code versions. [operator]

**F. Operator-side**
1. Fund U21959335 (~USD 505) → resubscribe Network B+C+OPRA → live data (still delayed mdType 4).
2. Rotate the Finnhub key; Polygon auth fails daily (`polygon_disabled_auth_failures`).
3. Do Telegram pages reach a place you actually look?
4. Funding: $3k CAD cannot size a viable condor; real figure ≈ $8.6k CAD; $10k simulated is provisional.
5. Box on by 09:25 ET; don't close the Gateway window.

## 8. Suite state on this box at `8faed40`

`python -m pytest tests/ -q -p no:cacheprovider --ignore=tests/test_vlmc_diagnostics.py` at `8faed40`, run 08:26–08:32 ET on the Windows box: **8 failed, 1875 passed, 43 skipped, 5 errors** — all 13 in `tests/test_w21_tz_aware_trade_timestamps.py`, every one `AttributeError: module 'time' has no attribute 'tzset'`. His W26/W27 test files pass here (126/126 together with `test_r21_post_stop_cooldown`, `test_w15_fail_open_gates`, `test_w7_exit_pricing`, `test_w2_executor_risk_gates`). His own fast lane reports 1882 passed on macOS; the 13-test gap is entirely the POSIX-only fixture.

## 9. What this integration does not do

No fix was made to production code. The R26 report's numbers were not changed by the pull except for
the three rows in §1 moving to "claimed CLOSED by W26/W27". His W26/W27 code was verified here only by
running his test files, not by independent probe. R25's 49 leads, the fail-open census, T1-b/T1-c and
`arch-01..04` remain exactly where R25 left them.

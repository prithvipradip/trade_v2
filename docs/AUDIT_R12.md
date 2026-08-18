# Round 12 — Sophistication & Maturity Audit (2026-07-13)

Six lenses no prior round used: concurrency/state-machine, trade-lifecycle craft,
chaos walks, vol/pricing craft, over-engineering, test maturity. This file is the
durable synthesis + decision sheet.

## Tier A — bugs & hardening (assistant scope, no behavior change)

1. **Trade-status state machine** (concurrency agent, NAIVE verdict): six writers do
   blind UPDATEs; reachable illegal edges: CLOSING->FILLED after restart (duplicate
   close -> position REVERSAL), exit partial-fill-then-cancel treated as clean cancel
   (oversell -> naked short), PARTIAL->FILLED loses quantity, CLOSED resurrection.
   FIX: compare-and-swap `transition(from_statuses, to)` in StateManager used by all
   writers; filled-qty-first check in `_determine_exit_fill_status`; rebuild exit
   trackers from broker working orders at startup reconcile.
2. **clientId-fallback blindness** (concurrency + chaos): reconnect on fallback id
   makes all working orders invisible -> mass false-CANCELLED. FIX: reqAllOpenOrders
   guard on fallback + refuse "cancelled" verdicts for orders seen there; first
   untracked sighting appends an EOD-recon discrepancy (debounce delayed the PAGE,
   not just the freeze).
3. **Chaos hardenings** (each <=15 lines): singleton lockfile in master.main
   (double-bot now impossible); IB-server-time drift check hourly + w32time start in
   keeper (clock jump = silent full-session protection kill TODAY; w32time confirmed
   NOT RUNNING); try/except around HWM/mark persist so stops keep firing through
   disk-full/DB-lock; PENDING row written BEFORE placeOrder; keeper skips dead-man
   ping when data/TELEGRAM_DEAD exists (marker currently written and read by NOBODY —
   the R8 comment claimed otherwise, wrong); duckdb-unavailable token in digest.
4. **Event-fill robustness** (concurrency): debounce flag can wedge True forever on
   create_task failure; unreferenced tasks GC-cancellable; busy-guard drops instead
   of queuing a rerun. FIX: flag reset in except, task set + done-callbacks, rerun
   flag.
5. **Test/CI repair** (test agent): CI as shipped CANNOT go green (includes ~1h GARCH
   suite, 30-min timeout, -x). FIX: slow/ibkr markers + default -m "not ibkr and not
   slow" + nightly slow job + drop -x. The 8 event-loop flakes root-caused (sync
   tests using get_event_loop after pytest-asyncio closes it -> replace with
   asyncio.run); test_learning reds are STALE tests enshrining
   pre-PROTECTED_STRATEGIES behavior -> rewrite; adopt the PROVEN lifecycle test
   pattern (real executor+state, shape-faithful SimpleNamespace broker fakes, 3.2s)
   for check_fills/ladder/reconciler; add hypothesis property tests (validated by
   20k-sample probe, 0 violations); shared tests/fakes.py.
6. **DONE during audit**: pytest was clobbering LIVE model artifacts (range.pkl AND
   ensemble.pkl overwritten with synthetic-noise models 07-13 13:12; the R7/R11
   fence covered walkforward only). conftest autouse fixture now redirects MODEL_DIR
   to tmp_path for all tests (verified: offending test no longer touches live
   files); artifacts rebuilt clean from real data 13:57.

## Tier B — strategy decisions (USER)

1. **Stop recalibration** (lifecycle agent, worked example): the 1.25x-credit stop
   fires $4-7 THROUGH the short strike (touch = only 0.49x credit) — cannibalizes
   the touch-and-recover cohort; breakeven win rate ~76% pre-cost. Backtest with the
   parity engine: {no stop / 2.0x / close-at-short-strike-touch}; disable stop when
   width-capped (launch scale: stop = 67% of max loss = pay slippage to save
   nothing).
2. **Enforce the DESIGNED trade** (vol agent): the 07-07 SPY condor filled with its
   short call at 0.49 DELTA (intended 0.20) — and passed every gate BECAUSE of the
   error (credit floors monotonically reward closer-to-ATM; no upper delta bound;
   "the gate stack's equilibrium, not an accident"). FIX: delta band |d| in
   [0.15, 0.30] + expected-move sanity gate (shorts within [0.6, 1.3]xEM) + extract
   ATM IV/EM from every live chain (kills the +/-50% flat-0.20-IV wing error).
   Recommendation: ADOPT — this enforces PLAN's stated strategy, not a new one.
3. **Exit execution**: resting GTC take-profit at the broker (survives bot death +
   monitor blindness) + mid-first ladder for non-urgent exits (stops stay
   marketable). TPs currently cross ask+$0.10 = 6-17% of each win surrendered.
4. **Cadence**: min entry DTE 14 (7-DTE entries are 2-day churn, best case 60%
   cost-dominated); per-expiry position cap 2-3; post-stop re-entry cooldown keyed
   on EXIT time (currently entry time — a stopped symbol can re-enter next scan into
   the same trend); VIX-tiered credit cap 6/4/2.
5. **IV regime**: percentile cap ~85 or term-structure condition (VRP by IV bucket:
   top bucket = -2.3 vol pts; AMD sits 98.8th percentile pre-earnings and the floor
   says TRADE); range gate -> trade-matched P(no-touch of actual short strikes) with
   ML as overlay (gate-vs-P&L correlation measured +0.18; spec choice alone flips
   verdicts).
6. **Two-animals problem**: paper earns the verdict on $22-wide condors; the funded
   account trades $2-wide. Phase-2 (AIT_SIMULATED_CAPITAL) must be the PRIMARY
   evidence. Confirmed keep: mechanical no-adjustment discipline (argued both ways).

## Tier C — deletion slate (USER; zero live-behavior change)

Top-5 by risk-reduction-per-effort (over-engineering agent):

1. **hedging.py + _check_hedging** (~170 LOC): DANGEROUS dead code — auto-execute
   path places SPY stock MARKET orders bypassing ALL executor guardrails (rate
   limit, INST-5, GOV-1, market-hours); one `auto_hedge: true` flip from live.
2. **torch/transformers -> optional extra** (580MB serving hard-disabled FinBERT;
   torch forensically implicated in the c0000005 crash cluster; halves install/CI).
3. **Sentiment stack retirement** (~1,250 LOC: sentiment/ + ib_news + fundamentals_db
   + consult sites): verified zero IC influence; salvage = 30-LOC VIX backwardation
   gate in risk/ if ever wanted.
4. **Signal queue excision** (~100 LOC unreachable post-R7, inside orchestrator).
5. **Dead-code sweep**: options_sim.py (loaded 5x-premium tail trap), directional.py,
   covered.py (naked-call generator), options_flow.py (~1,050 LOC).

Also: GARCH/MSGARCH/OU -> research tree (-2,936 src + ~1,200 test LOC out of live
CI); capital_tiers 250 -> ~40 LOC; counterfactual analysis outputs stripped (twice
ruled garbage), record_skip rows kept; meta_label default-off; drift.py
persist-or-delete. Net: live import graph 38k -> ~20k LOC. None of this changes IC
behavior.

## Sound / validated (no action)

SQLite threading, R11 to_thread work, ib_insync context discipline, reprice-ladder
interleaving, DB-lock and duckdb-corruption failure paths (verified by experiment),
determinism/RNG hygiene in tests, no-adjustment discipline, delta-based strike
selection as skew handling, backtest tree (gate-4 critical path), ops/monitoring
layer.

## Machine facts verified during R12 (user actions, STILL OPEN)

dead-man UNARMED (data/deadman_url.txt missing); AutoAdminLogon NOT set; Windows
Time service NOT RUNNING on a box with a prior 2h clock error.

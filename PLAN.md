# AIT v2 — Improvement Plan (from full repo audit, 2026-07-07)

Mission: autonomous options-income bot (sell premium via iron condors / strangles on
liquid US ETFs/megacaps), ML-gated entries, automated exits, self-learning, hands-off ops.
Paper account DUN603821. Real track record as of audit: **$248.80 banked (5 real closes),
8 open filled positions** — machine works, edge unproven.

Strategic frame: with the learning layer dormant below ~30 real trades, the fastest path
to "is the edge real?" is **maximizing clean trade count while protecting track-record
integrity**. Priorities below rank by that, not code aesthetics.

Full audit reports: four-agent audit (pipeline core, risk/broker, ML/learning, ops/infra),
2026-07-07. Key file:line references inline.

---

## Batch 1 — safety/correctness bugs live right now

| # | Item | Where | Status |
|---|------|-------|--------|
| 1.1 | **Dead thesis-invalidation exit**: `_check_thesis_valid` raises NameError (`symbol` undefined) every call, swallowed by broad except → ML direction-flip exit never runs. Fix var + narrow except. | orchestrator.py:1797,1822 | done |
| 1.2 | **IC gated backwards**: iron condors require ≥0.65 *directional* confidence pre-check — blocks calm markets (ideal for ICs), admits trending ones. Backtest skips this gate for neutral strategies (engine.py:300); live must too. Gate ICs on range-model probability. | orchestrator.py:~730 | done |
| 1.3 | **bot_stdout.log unbounded (2.0 GB)**: raw append sink never rotated; mtime-based cleanup can never delete it. Size-capped rotation + size-based cleanup. | master.py:146,514 | done |
| 1.4 | **Per-position marks never persisted**: `open_positions.unrealized_pnl` stays 0 forever. Add `update_position_mark()` in state.py, call in `_evaluate_position` (marks-present guard), surface per-position P&L in status.py + 8503 dashboard (green/red table). | portfolio.py:240, state.py, status.py:122, status_server.py | done |
| 1.5 | **Reconnect wedge**: after 5 failed reconnects, `_reconnect` returns False forever (counter never resets). Reset counter on exhaustion + alert. | ibkr_client.py:166-189 | done |
| 1.6 | **FX silent overstatement**: failed USDCAD fetch → usd_to_base=1.0 → all CAD-base limits inflated ~39%. Cache last good rate; if none, mark account data unreliable (risk layer treats as no-data). | account.py / ibkr_client.py:401 area | done |
| 1.7 | **keeper_ait.bat untracked** — the crash-recovery script exists only on this machine. git add. | repo root | done |

## Batch 2 — track-record integrity + edge validation

| # | Item | Where | Status |
|---|------|-------|--------|
| 2.1 | **Strangle risk mis-modeled**: `max_loss = 3× credit` understates tail 10-30×; buying-power check inverted for credits (checks affordability of money received); IBKR margin never consulted. Use strike-distance-based tail estimate; check margin/available funds for credit trades. | straddles.py:155, manager.py:204 | done |
| 2.2 | **Partial exits book phantom P&L**: records estimated P&L + underlying price as fill before order fills → DB/IBKR drift. Register order, book on real fill. | orchestrator.py:1407-1447 | done |
| 2.3 | **Size multiplier bypasses risk gates**: learning multiplier applied AFTER validate_trade → gates check 1×, book can execute N×. Apply before validation. | orchestrator.py:1235 | done |
| 2.4 | **Range-model CV leakage**: 30-day overlapping labels with gap=5 → inflated AUROC feeding the trade/no-trade edge gate. gap ≥ horizon. | range_predictor.py:479-491 | done |
| 2.5 | **Thompson decay truncation**: `int(wins*0.995)` zeroes 1-win arms daily → bandit can never accumulate signal at low volume. Decay in float, round on read. | thompson.py:132-133 | done |

## Batch 3 — before trusting results / going live

| # | Item | Where | Status |
|---|------|-------|--------|
| 3.1 | **Retrain pipeline**: 7:30 daily retrain writes models the live bot doesn't load (7-day reload timer) — unify; pin OMP threads in subprocess; atomic pickle writes (tmp+rename). | master.py:396-436, trainer.py:67, ensemble.py:419 | done |
| 3.2 | **Tests for burned paths**: marketable-combo entry pricing + spread-reject gate (executor), BotManager restart budget / gateway-defer. Zero coverage today despite production failures. | tests/ | done |
| 3.3 | **Config consolidation**: hoist behavioral literals (EXIT_CROSS, entry offset, RANGE_MIN_CONFIDENCE, budget tiers, risk cap %s, correlation cap) into settings; delete dead gates (daily-limit on wrong object, concentration reading never-populated field); wire or remove phantom `max_portfolio_risk_pct`. | manager.py, orchestrator.py, executor.py, settings.py | done |
| 3.4 | **Backtest fill realism** (larger project, separate pass): historical option quotes where available, model combo non-fills. Until then treat +311% IC backtest as directional only. | engine.py | deferred |

## Round 2 (2026-07-07 evening) — verification audit: fixes verified + new findings fixed

Adversarial review of Round-1 commits: all major fixes verified clean (neutral-only, FX stop,
reconnect, Thompson floats, config wiring); stress-risk does NOT kill strangles; CC/CSP
blocking is pre-existing (3% cap vs assignment notional), not a regression.

| # | Item | Where | Status |
|---|------|-------|--------|
| R2.1 | **P&L booked on signal price, not real fill** (forensics: all 8 fills adverse, −$89, up to 14.8% of credit — slippage invisible, P&L overstated). Real fill now written to trades.entry_price; BAG-aware entry fill reconstruction. | executor.py, state.py | done |
| R2.2 | **ITM-aware expiry booking** — expired credit was booked full-premium WIN regardless of moneyness (top sample-corruption risk). Now valued at intrinsic via underlying price; refuses to invent numbers (review flag) if price unavailable. | reconciler.py | done |
| R2.3 | **Sweep front-ran fill-promotion** — 30-min PENDING closed as "$0 never filled" before IBKR liveness check could rescue it. Sweep now liveness-gated (promotes if legs live; skips if IBKR unreadable) and runs AFTER promotion. | reconciler.py | done |
| R2.4 | **Partial/cancel status writes were no-ops** (INSERT OR IGNORE on existing row) — partial entry fills were live UNMANAGED positions. Real writes + open_positions registration. | executor.py | done |
| R2.5 | **Exit-cancel double-close race** — stale exit reverted CLOSING→FILLED while the cancel could race a fill → second close = fresh naked position. Now keeps tracking until terminal state (900s zombie cap). | executor.py | done |
| R2.6 | **Assignment detection** — untracked STK position at IBKR now flagged CRITICAL + Telegram (was: one log line, then silent buying-power rot). | reconciler.py, orchestrator.py | done |
| R2.7 | **Stale strangle max_loss restated** — 3 pre-fix positions under-counted aggregate risk 2.7x ($6.6k vs $18.1k true); restated in bot_state + 28 dead keys purged + keys now deleted on close. | one-time data fix, orchestrator.py | done |
| R2.8 | **Short-vol guardrails** — credit-position cap (6) + VIX entry halt (28) for credit strategies; delta gate is dead and daily breaker sees only realized P&L, so these are the gap-day brakes. | manager.py, settings, config | done |
| R2.9 | **Unprotected-state alerts** — prolonged marks-missing (10 ticks ≈5 min → stop/TP silently OFF) and PDT-blocked stop now Telegram once. | portfolio.py, orchestrator.py | done |
| R2.10 | **Sample velocity** — universe +GLD/TLT/XLE (index cluster saturated book at ~5/8 slots), dte_range 14-45 → 7-30 (~2x closes/slot). 100 closes/month was NOT achievable as configured (~15-20). | config.yaml | done |
| R2.11 | **Mid-session restart guard** — fresh-models marker restart now deferred while market open + gated on successful unlink (was: slow retrain → unguarded restart at 10am; locked marker → restart every 2 min). | master.py | done |
| R2.12 | Smalls: partial-exit await 10s→4s (stalled other stops), OI=0 treated as unknown not illiquid (IBKR realtime), GARCH CV gap pinned 5 (over-purge), status.py pre-migration fallback, ait.log 10MB×5→20MB×10 (error spam destroyed forensic evidence), Thompson float fmt. | various | done |
| R2.13 | Retrain-effectiveness + fill-quality watch: verify tomorrow's session — thesis-check runs clean, marks persist, slippage on new entries, GLD/TLT/XLE signals generate. | live verification | todo |
| R2.14 | Deferred: Telegram off hot path w/ retry+2nd channel; covered_call share-ownership precondition (currently unreachable — blocked by 3% cap); ET-pinning for time gates; wmic→tasklist in keeper. | various | todo |

## Round 3 (2026-07-07 night) — LINE-BY-LINE audit of the FULL system (~40k lines, 8 auditors, 100% coverage attested per file)

FIXED (commit refs this batch):
| Area | Fix |
|---|---|
| executor | BAG fill reconstruction was 100x too small (execution.shares = CONTRACTS for options); eager "cancelled" no longer orphans partial fills or mass-cancels on disconnect (filled-qty-first + not-found grace + connection guard); terminal partials stop tracking cleanly |
| reconciler | realized P&L attribution now LEG-aware (was: first same-symbol item ≈ 1/4 of an IC, cross-trade bleed) |
| orchestrator | fast-monitor errors now WARN + feed watchdog (were DEBUG-swallowed for hours); partial exits no longer double-count trades_won; breaker now sees partial P&L; first-hour 0.85 gate exempts range-gated neutral strategies (it structurally banned ICs 9:30-10:30) + ET-pinned; drift not fed fake directions for neutral wins; PDT round-trips now RECORDED (guard was fully inert — zero callers) |
| watchdog | heartbeat no longer zeroes error_count (counter could never trip) |
| circuit_breaker | record_partial_pnl(); ET-pinned daily reset (local-midnight reset could un-trip the daily halt MID-SESSION); consecutive-loss counter reset on auto-resume |
| risk | sizer: credit strategies sized on capital-at-risk not premium (4-5x understatement); min-1 floor no longer forces an unaffordable contract; correlation cap now signed (negative-corr hedges no longer blocked); capital_tiers preferred lists include GLD/TLT/XLE (tier filter was silently deleting the R2.10 diversifiers!) |
| thompson | atomic state save (kill mid-write silently reset the whole bandit) |
| ML | vol_mag CV purge gap >= horizon (same leak-class as fixed range 2.4 — gated straddles on illusory edge); meta-label scaler now per-fold (was fit on ALL data pre-CV); ensemble labels NaN-init (last 5 rows were stamped fake-NEUTRAL every retrain); predict() reindex-safe + honest features_used; range model actually consumes VLMC features; directional CV gap=5 |
| vol models | OU jump P(in-range) horizon variance was 252x too small (returned ~1.0 constant into the IC gate ensemble); GARCH student-t/skew-t thresholds now unit-variance rescaled |
| sentiment | news keyword matching word-boundary anchored ("gain" in "against" flipped signs) |
| learning | analyzer window keys on exit_time (7-30d holds closed in-window were excluded) |
| data | daily save() no longer wipes implied_vol (INSERT OR REPLACE deleted it every retrain — IV features silently dead); int(NaN) volume guards in save paths + chain builder (one NaN killed a whole 50-contract batch); $0.50 strikes allowed (whole-dollar filter deleted ATM strikes on cheap names); earnings fetch never blocks the event loop (background-fills cache) |
| ops | web_logs + streamlit dashboards bound to 127.0.0.1 (were LAN-exposed, no auth); daily report called a NONEXISTENT method since day one (empty file + false success) — fixed + returncode-gated; APScheduler misfire grace 1s→1h (jobs were silently dropped on restarts); orchestrator.log size-capped; Telegram token redacted from error logs; status.py None-guard |
| backtest/offline | fix_pnl_history refuses re-run (2nd run zeroed migrated wins); run_optimizer --apply requires AIT_ALLOW_APPLY=1 (was a direct overfit→live-config pipe); engine entry commission no longer double-debited |

## Round 3-A (2026-07-08 pre-dawn) — "finish everything under A": all remaining code items DONE
A1 MTM daily-loss brake (30s tick, realized + unrealized-vs-SOD, same breaker, alert once);
A2 exchange-time quote timestamps (staleness detection un-blinded); A3 remaining ET-pinning
(budget tiers, daily-stats bucketing incl. StateManager default); A4 get_portfolio_summary
read-only (persist=False — no HWM/marks/streak/alert mutation from reports); A5 watchdog
ibkr+market_data heartbeats wired; A6 chain-median IV fallback (not flat 30%) for greeks;
A7 stale parametric GARCH/MS/OU members skipped past retrain interval; A8 trainer rollback
on all-symbol MEAN + adaptor one-adaptation-per-parameter + counterfactual min-age/zero-
entry guards + analyzer NULL choke point; A9 partial-day bar excluded from live resample +
bar-source tag column (TRADES/MIDPOINT); A10 economic-calendar exhaustion alarm (2027 bomb
defused with a CRITICAL log); A11 Telegram off hot path w/ 3 retries + DuckDB read-only
readers + TTLCache copy-on-get + wmic->PowerShell in keeper/autostart/status.
A12 (open, live-only): observe the sign of a real closing-BAG avgFillPrice on the first
debit-spread close (exit negation depends on it) — grep exit_order_filled that day.

DEFERRED (Round 3 todo — larger design work, ranked):
1. **Backtest credibility overhaul — CORE FIXED 2026-07-07 night**: sleeve-capital aggregation
   (window capital = capital x N symbols; was ~N x inflated), trade-frequency-aware
   Sharpe/Sortino annualization via `annualization_factor()` (was sqrt(252) per trade,
   ~4.6x overstated; Sortino unified to target-0 downside deviation), non-overlapping
   windows by default (step 21->63), chronological trade ordering for drawdown,
   per-symbol returns vs full sleeve, forced-iron_condor removed (preferred on NEUTRAL
   only). STILL OPEN from this cluster: export.py/app.py/analytics/duckdb display-side
   sqrt(252) call sites (5), M10 param-collision in export/apply, M11 learner overlap
   leak (sequential runs), and 3.4 fill realism. Historical run artifacts in reports/
   were produced by the OLD math — re-run before citing any number.
2. **Adjusted-vs-unadjusted price mixing — FIXED at the Yahoo boundary 2026-07-07 night**
   (all 5 yfinance history() sites now auto_adjust=False = split-adjusted, div-unadjusted,
   matching IBKR TRADES / Polygon). NOTE: previously-saved Yahoo-sourced rows in the SQLite
   store and any trained models still carry the old mixed basis — next retrain refreshes
   models; consider a one-time store re-backfill for dividend payers.
3. Quote timestamps = exchange time (staleness detection currently blind); intraday bar-semantics tag (TRADES vs MIDPOINT vs adjusted); partial-day bar exclusion in live features.
4. get_portfolio_summary must stop mutating protection state (EOD false MARKS-MISSING alert risk); watchdog ibkr/market_data components never heartbeated; daily-loss breaker still entry-gated + realized-only.
5. GARCH members frozen at train time (stale up to reload interval); trainer rollback keyed on last symbol only; adaptor same-param compounding within a cycle; counterfactual eval min-elapsed guard; economic_calendar hardcoded 2026-only (year-end time bomb); DuckDB readers read_only; sortino/profit-factor display consistency; analyzer NULL-column guards.
6. MP-F3 verify: sign of a real closing-BAG avgFillPrice (debit-spread exit negation depends on it) — check on first live debit close.

## Deferred / watchlist
- ~~Live data~~ done 2026-07-02 (Network B/C + OPRA, ~$3/mo).
- IBC read-only automation: vmoptions `--add-opens` fixed the crash; box may still need manual
  check→uncheck→Apply toggle after some Gateway restarts. Guardrail alerts via Telegram.
- Options-chain OI=0 vs `min_open_interest` filter interplay; correlation guard gets no price
  data (sector-table fallback only); portfolio delta gate ~0 (no greeks subscription);
  ET-vs-local time in Friday gate + budget tiers; wmic deprecation in keeper.
- Learning layer wakes at ~30 real closes; expect ~1 month at current fill rate.

## Round 4 (2026-07-08 pre-open) — institutional architect audit: lifecycle guarantees
Verdicts: order-lifecycle REFUSE (mid-placement crash window -> live unmanaged position);
unprotected-book REFUSE for strangles / CONDITIONAL PASS for the IC-only funded config (the
LONG WINGS are the broker-side protection — no OCA gold-plating needed at this size); state
architecture REFUSE (no backup, mutable ledger); kill-switch REFUSE; clock CONDITIONAL PASS.
ALL five right-sized remediations SHIPPED (~70 lines):
- INST-1 daily `sqlite3 .backup` of ait_state.db (17:00 ET cron; 14 local snapshots + mirror
  to ~/Documents/ait_backups — SYNC THAT FOLDER TO CLOUD for true off-box). Verified live.
- INST-2 data/HALT file = operator kill switch (blocks new entries, exits keep managing).
- INST-3 untracked live OPTION at broker (mid-placement crash) -> data/HALT_UNTRACKED auto-
  freeze + CRITICAL alert until a human resolves and deletes the file.
- INST-4 executor token-bucket rate limiter (>8 orders/60s = malfunction -> refuse + CRITICAL).
- INST-5 defined-risk-only ENFORCED AT THE EXECUTOR: undefined-risk orders refused unless
  AIT_ALLOW_UNDEFINED_RISK=1 (set by run_orchestrator for the paper phase; REMOVE AT GO-LIVE
  — this is what makes the wings a contractual loss floor, gate 2 structural).
Deferred (documented, right-sized OUT at this account size): broker-side OCA brackets,
event-sourcing ledger rewrite, HA. Immutable-ledger need is partially met by daily snapshots.
Governance-auditor findings (second agent) recorded separately when delivered.

## Go-live gates (decided 2026-07-07 — do NOT relitigate on a winning streak)
Paper phase: ALL strategies stay enabled (incl. short_strangle) — the sample must answer
"which strategies have edge", and paper blowups are data, not losses.

Going LIVE requires ALL of:
1. **Paper verdict**: >= 50-100 real closes with profit factor > 1.3, max drawdown < 8% of
   deployed risk, stable slippage, zero unmanaged-position incidents. Criteria fixed BEFORE
   looking at results.
2. **Defined-risk only**: iron condors (and debit structures) ONLY. NO short strangles live —
   their max_loss is a model estimate, not a contract. Strangles earn live status only after
   a real margin model + 3 months of live consistency at base size.
3. **Tuition sizing — user decision 2026-07-07 (rev 2): $3,000 CAD (~$2.1k USD) initial
   funding → SMALL capital tier** (SPY/QQQ/IWM/AMD/AAPL + GLD/TLT/XLE, iron condors
   preferred — matches the defined-risk-only rule, 3 positions).
   **Phase-switch recipe (target-size paper validation, run BEFORE funding):**
   - .env: AIT_SIMULATED_CAPITAL=2100  (scales the whole risk stack to launch size)
   - config.yaml: risk.max_position_risk_pct: 0.03 -> 0.07  (3% = $63 at $2.1k, below ANY
     real spread's max loss — 7% matches the SMALL tier design; REVERT both when
     switching back to big-account paper)
   - restart bot; verify SMALL-tier universe + 1-2-wide condors + fills; run >= 4 weeks.
   Notes: MICRO/SMALL affordability bug SR-L11 fixed; tier max_underlying_price/wing
   logic confirmed DEAD live (only the name filter runs). Costs at this size: ~$5-13
   per IC round trip vs ~$20-45 targets — thin nets; the $3k stage validates PROCESS
   with real money, not income. 1 contract, daily-loss halt on, VIX halt on, weekly
   human review mandatory.
4. **Backtest credibility fixed first** (Round 3 deferred #1) — no sizing decision may cite
   backtest numbers until the engine's inflation bugs are fixed.
5. Expect live < paper (real fills). Scale only on 3+ months of live evidence.

## Operating rules (learned the hard way)
- Restarting the BOT ≠ restarting the GATEWAY. Entitlements/read-only load at Gateway login.
- ONE live-data slot: live account must stay logged out of mobile/web while bot runs.
- Paper ≠ live fills: paper is optimistic; marketable pricing narrows the gap.
- Secrets: IBC password lives ONLY in C:\IBC\config.ini (outside repo). Never in git/chat/docs.

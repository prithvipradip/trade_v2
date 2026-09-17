# AIT v2 — Plan & Audit Ledger

**Mission:** autonomous options-income bot (iron condors on liquid US ETFs/megacaps),
ML-gated entries, automated exits, hands-off ops. Paper account DUN603821.
**Goal:** a trustworthy real track record answering "does this have edge?" before funding
$3,000 CAD.

**Current state (2026-09-11):**
- **Verdict metric (iron_condor only): 12 closes, 8W-4L, +$1,235.74, PF 3.89.** Gates 2/3/4
  PASS (PF 3.89 > 1.3; maxDD $272 = 3.2% of ~$8,421 deployed risk < 8%; median slip 1.6% of
  credit over 9 fills < 8%). Gate 1 FAILS on sample only: 12/50 prelim, 12/100 funding.
  Book-level incl. retired experiments: 21 closes, 13W-8L, +$933.68, PF 1.82. Book FLAT.
- **2026-09-11 — first two closes under the W7/W11/W12/W14 exit rebuild, both take-profits:**
  SPY +$203.65 @41.6% and QQQ +$241.77 @40.1% (+$445.42). All four waves verified on live
  fills: W7's 1.5x-mark ceiling priced SPY's 7 no-quote attempts at 4.58-4.79 where the old
  formula gave 14.32; W12's leg synthesis fired once (QQQ, synthesized 4.43); W11 trued up
  commission to the broker ledger (SPY est $7.33 vs real $7.35; QQQ $7.34 vs $8.24 — both
  inside the documented $0.62-$1.06/leg routing range); W14 booked each close exactly once
  (exit_outbox empty, daily_stats 2 trades/$445.42, breaker daily_pnl matches).
- **PACE IS THE BINDING CONSTRAINT, and it is a regime not a defect.** 1.2 verdict closes/wk
  over the trailing 28d against a 3-4/wk target — ~30 weeks to the 50-close prelim read. The
  structural ceiling is ~3.1/wk (3 symbols / 6.9-day mean hold), so the gap is entry supply:
  slot utilisation is 55% (16 of 20 sessions at 2 open, never 3). Cause is low IV, confirmed
  09-09: IWM iv_rank 5.95 emits ZERO signals; SPY 19.44 and QQQ 34.92 emit but are rejected on
  `credit_width_ratio` (SPY credit 3.05/width 41; QQQ 4.55/54). The strategy is correctly
  declining to sell cheap vol — but the verdict timeline is hostage to IV mean-reverting.
  NOTE: the "IWM alive since the 08-10 tradingClass fix" claim below is stale in effect —
  IWM's chain, ML and scan all run, but it has produced no signal since 07-17 on IV rank alone.
- **Live config (wide-wing epoch):** iron_condor ONLY, SPY/QQQ/IWM (IWM alive again since the
  08-10 tradingClass fix), k=1.6 wings / 0.10 scaled ratio floor, ML entry gates ON (B1
  confirm: DD 34.6%->10.0% over 11y), pre-event blackout 1 TRADING day, condors hold through
  macro events, PDT off at $198k paper, delayed data (mdType 4) until U6.
- **Reliability state after R16-R18 (~180 defects fixed since 07-16):** dead-man ARMED
  (healthchecks.io, market-hours cron); circuit breaker persists across restarts; exit quotes
  price the raw BAG (the qualify gate was dead code); exits mark-anchored; hot-path smoke
  harness + smoke_deploy dry-run gate + mypy type gate (blocking on attr-defined class);
  suite 1,100 green. R18 fixes are committed but LIVE ONLY AFTER NEXT BOOT.
- **08-11 outage lesson (R18):** a one-line settings typo killed all entry scans for 3 trading
  days behind a green source-string test; Telegram paged 24x unnoticed. Root causes closed
  (execute-don't-read tests, loop amplifier, escalation). OPEN QUESTION for operator: do
  Telegram pages reach a place you actually look?
- **Measurement fidelity (all engine studies now):** touch-stop mirrored via High/Low (sim win
  rates restated ~7pp lower), per-symbol implied vol (VXN-calibrated; flat VIXx1.10 understated
  QQQ ~12%), empirically calibrated spreads (config overstated friction 2-13x). iv_rank now
  self-heals from scan ATM IV (the R16 version never wrote — fixed R18).
- **USER-SIDE OPEN:** U6 fund U21959335 ~USD 505 -> resubscribe Network B+C+OPRA -> I flip to
  live data; merge PR #6 (arms the mypy gate + nightly CI — advisory until merged); U7 repo
  private (username + live account number visible in a public repo); U10 rotate Finnhub key;
  box on by 09:25 ET on trading days (Start-menu shutdowns at ~22:00 are fine).
- 18 audit/fix rounds complete (~465 defects found; R17 was an independent external review).

## INCIDENT 2026-07-13 — the R12 reversal bug fired in the wild, pre-R12-load
Monday's macro-flatten exits **triple-filled**. Ledger proof (executions table): NVDA condor
T-20260706-140054 closed by order 261874 (17:30:44Z, P&L booked −$127.7 net), then the SAME
exit combo re-placed and re-filled as 261926 (17:31:44Z) and 261902 (17:32:26Z, both P&L
$0 in books). First fill closed the condor; the extra fills built an inverse position —
**duplicate close → position REVERSAL, the exact class R12's CAS state machine kills** —
~25 min before the 13:57 retrain reload could load any R12 code. QQQ (261908) and IWM
(261914) each double-filled once. Result: **12 untracked legs at the broker** = 4 accidental
reverse iron condors (long-vol debit, defined risk): IWM 290/288P+310/312C ×1,
QQQ 690/688P+753/755C ×1, NVDA 185/175P+210/220C ×2 — ~$955 premium, expiry 07-24, theta
bleed now that CPI has printed. The untracked-option tripwire worked: HALT_UNTRACKED at
09:00 07-14, entries frozen, exits unaffected. Books are NOT contaminated (each close booked
once, off the first real fill); account NLV vs track-record P&L will diverge by these
positions' outcome until resolved. The duplicate executions DO sit in the executions ledger
under their old trade_ids — the shadow referee (R13) must flag exactly this class.

**Doc map:** `docs/AUDIT_R12.md` (sophistication/maturity decision sheet) ·
`docs/GAP_AUDIT_R7.md` (component gaps) · `docs/BENCHMARKS_R9.md` (measured latency /
accuracy / performance) · `docs/MATURITY_R10.md` (external best-practice matrix) ·
`docs/RUNBOOK.md` (alert→action, deploy checklist, drills). Everything under `deprecated/`
is retired-not-deleted and loaded by nothing.

---

## WHAT'S NEXT (after R13) — the sequence

The mission is a trustworthy verdict on edge. Everything below is ordered by what actually
moves that forward. **The bot cannot build a sample while entries are frozen, so U5 is the
single highest-value action on this page.**

**1. Unfreeze the machine (~~U5~~ ✅ → U6).** U5 done 07-15 (book flattened, entries unfrozen).
U6 re-scoped 07-16: fund the live account ~USD 505 so the data subs can activate. Nothing else
matters until the bot has real option quotes again.
**2. Take the two open decisions (D1, D2)** — both must be settled BEFORE the sample grows,
because both change how results are scored, and "criteria fixed before looking" is the rule.
**3. Close the security exposure (U7 → U8/U9 → U10/U11/U12).** Independent of trading; do in
an evening.
**4. Arm the silent-failure layers (U1, U2, U3, U4).** Long-standing, ~5 min each, and R13
re-confirmed all three are still unarmed — every recovery layer is downstream of them.
**5. Ship the R14 code queue** (below) in out-of-RTH windows while the sample accumulates.
*R14 Tier-1 DONE — exit-price bound + broker-liveness gate (`dae9c64`), exit-input staleness gate
(`935e897`). R14 Tier-3 + item 3b + the capital-at-risk base fix DONE (`081be1b`). 724 tests green,
every guard mutation-checked. All effective on the next bot start; no live position touched while
entries are frozen. **Everything in the R14 code queue that is NOT blocked on a decision is now
shipped.** What remains: Tier 2 (verdict-correctness code — D1/D2) is blocked until you settle D1
and D2; the historical `capital_at_risk` backfill; and two deferred, deliberately-scoped items
(broker `errorEvent`/halt hook, `ib_async` migration). The mission-blocker is still **U5 —
unfreeze entries**; none of this builds the sample.*
**6. Then, and only then**: ML ablation → 50-close preliminary read → 100-close verdict →
Phase-2 sizing → go-live gates.

### OPEN — user actions (only YOU can do these; each verified on the box)

**Blocking the mission**
| # | Action | Why it matters |
|---|---|---|
| ~~U5~~ | ✅ **DONE 07-15.** All 11 orphan legs flattened at the broker (scripted, dedicated clientId, fills verified against a fresh `reqPositions`), book reconciled, `HALT_UNTRACKED` cleared, entries unfrozen, bot did not re-freeze. | Was THE blocker; closed. |
| U6 | **Fund U21959335 with ~USD 505 (≈CAD 700) → re-subscribe Network B ($1.50) + Network C ($1.50) + OPRA ($1.50) → confirm share-with-paper → tell the bot-keeper to restart the Gateway.** Root cause (07-16): the subs LAPSED because the live account is unfunded — IBKR requires USD 500 minimum equity + fees to activate data; re-subscribe was REJECTED "insufficient equity". The competing-session half (10197) is already fixed. The deposit SITS as balance; only $4.50/mo is spent. No free alternative exists (web-verified; OPRA NP floor is $1.25). | **NOW the blocker.** Without OPRA there are no option quotes at all: delayed fallback breaks delta strike selection (18 condor rejects 07-16) and starves fills — zero sample-building even with entries unfrozen. Funding ≠ going live: the bot stays `--paper`; go-live gates unchanged. |

**Security (R13 — do in one evening, outside RTH)**
| # | Action | Why it matters |
|---|---|---|
| U7 | **Make the GitHub repo PRIVATE.** No `gh` on this box → do it in the browser: repo → Settings → General → Danger Zone → Change visibility → Private. | The repo is PUBLIC today: PLAN.md (incl. this unarmed-protections list and the IB-password file location), the account ID, 38 daily P&L reports, a 1.46 GB archive branch — all world-readable. No secret ever entered git history (verified), so nothing needs rotating for this. R13 #9. |
| U8 | **Close the Gateway LAN hole**: `C:\IBC\config.ini` line 699 `AcceptIncomingConnectionAction=accept → reject`, then disable the java.exe Private-profile firewall Allow rules (+ the ibgateway 1048 Allow rule). The bot is unaffected — it connects via 127.0.0.1, which is in TrustedIPs (verified on the live session). | The Gateway API listens on 0.0.0.0:4002 with `ReadOnlyApi=no` and IBC auto-accepting any non-trusted client. The ONLY thing standing between the LAN and an order-placing API is the Wi-Fi being categorized "Public" — one profile flip (or a Private-categorized VPN/vEthernet adapter, and WSL's is already Up) exposes it. R13 #4. |
| U9 | **MySQL is reachable from the Wi-Fi RIGHT NOW**: elevated `Disable-NetFirewallRule -DisplayName 'Port 3306','Port 33060'` (or bind 127.0.0.1 in my.ini). | Its port rules Allow ALL profiles including Public, so the "Public" categorization does NOT protect it (`Test-NetConnection 192.168.2.16 -Port 3306` succeeds). Not part of the bot — pure attack surface on the machine holding the trading credentials. R13 #14. |
| U10 | **Rotate the Finnhub API key** at finnhub.io, then scrub the logs (`logs/*`, grep `?token=`). | The live key sits in 11 log files (200 hits) via urllib3 DEBUG URL logging; confirmed still ACTIVE on 07-14. The class fix (urllib3→INFO) is deployed, so no NEW leaks — but the existing key is burned. R13 #12. |
| U11 | **Tighten the `C:\IBC` ACL**: `icacls C:\IBC /inheritance:r /grant:r "prith:(F)" "SYSTEM:(F)" "Administrators:(F)"`, then restart the Gateway once to confirm it still reads. | The IB login+password file is readable by `BUILTIN\Users` and MODIFIABLE by `Authenticated Users` (inherited from C:\); a second enabled local account exists. Becomes HIGH the day live credentials land there. R13 #13. |
| U12 | Docker Desktop → Settings → General → untick "Expose daemon on tcp://localhost:2375 without TLS". | Unauthenticated Docker Engine API on localhost = a local-privilege-escalation amplifier on the trading box. Localhost-only, so lowest priority. R13 #24. |

**Silent-failure layers (long-standing; R13 re-confirmed all still unarmed)**
| # | Action | Why it matters |
|---|---|---|
| U1 | **Create a healthchecks.io check → put its ping URL in `data/deadman_url.txt`** (~3 min). | Shipped-but-UNARMED since 07-09. The ONLY alert that fires when the whole machine dies. Closes detection for three separate disasters at once (machine down, keeper dead, Telegram dead). The 07-08→07-09 outage was 18 hours of silence. |
| U2 | **Enable Windows auto-logon** (Sysinternals Autologon + lock on logon). | Every recovery layer — keeper, Gateway, bot — starts from Startup shortcuts that need a human logon. A forced reboot at 2pm leaves the book unprotected until someone logs in; the first signal would be a *missing* 16:05 digest. Also required for the scheduled liveness check to run. |
| U3 | **Start the Windows Time service** (`net start w32time & w32tm /resync`). Do it outside RTH — a resync can jump the clock. | w32time is NOT RUNNING on a box that already had a 2-hour clock error. A clock jump silently kills all stop/TP protection AND disarms the hang detector — both read the same wrong clock. |
| U4 | Cloud-sync `~/Documents/ait_backups`. | The mirror exists (and is now hash-verified after the R13 fix) but shares a failure domain with the box. |

### OPEN — decisions (mine to recommend, yours to make; settle BEFORE the sample grows)

| # | Decision | Recommendation |
|---|---|---|
| ~~D1~~ | ✅ **DECIDED + EXECUTED 2026-07-16** (`scripts/restate_d1.py`, backup `ait_state.pre_d1_restatement.db`, bot stopped). All verifiable closes restated to the broker's own numbers (referee math: closing-fill-group realizedPNL + all-group commissions). A 10th close (IWM long_call, −$22.91, thesis-flip same-day) landed between decision and apply, so the final restated record is **n=10, 6W-4L, +$231.99, PF 2.30** (QQQ +6.80 → −15.71 as predicted). daily_stats recomputed for affected dates; provenance in `bot_state['d1_restatement']`; DuckDB mirror matched. **Referee post-apply: 0 BREAKS — books ≡ broker to the cent.** Ongoing path already books real commissions (R7) and duplicates are dead (R12/R14), so the error class does not re-accrue. | Done. |
| ~~D2~~ | ✅ **DECIDED + EXECUTED 2026-07-16**: the DD denominator is PINNED to **max concurrent deployed risk** (`_max_concurrent_car` in master.py, event-sweep over [entry, exit) windows; 6 tests). `capital_at_risk` backfilled on all 5 missing closes (defined-risk exact from legs; strangles 3×-credit convention, labeled estimates). Referee check [6] now verifies car COVERAGE + reports the pinned method: **DD $178.59 / $6,710 = 2.7% (gate <8% → PASS)**. The old open-book/$1k-floor base is gone. Criteria fixed before the sample grows — do not revisit with results visible. | Done. |
| U12 | Docker Desktop → Settings → General → untick "Expose daemon on tcp://localhost:2375 without TLS" | Unauthenticated Docker Engine API on localhost = local-privilege-escalation amplifier on the trading box. Localhost-only, so lowest priority. R13 #24. |

---

## Round 13 (2026-07-14) — security/supply-chain, market-catastrophe, shadow referee
32-agent workflow (5 lenses × adversarial verify), 26 verified findings. Human-factors lens
died on an API error mid-run; rerun in flight. Same-day fixes below are COMMITTED but await
the 16:00 out-of-RTH redeploy — the running bot still has both criticals.

### The two criticals (both in `_evaluate_position`, both shipped by R12, both latent-live)
1. **A function-local `import os` (macro block) killed exit evaluation for EVERY credit
   position** — `os.environ` at the touch-stop guard runs before the local import →
   UnboundLocalError on every credit tick, propagating out of `check_positions`: exits, MTM
   brake, and entry scans all die while any condor is open. Latent only because today's book
   is one debit straddle. FIXED (module import only).
2. **The R12 touch-stop `if` orphaned the exit elif-chain** — take-profit / assignment /
   DTE / delta / earnings exits were structurally unreachable for credit positions. A condor
   could only ever exit via stop or touch. FIXED (explicit `not should_exit` guards).
   Both criticals shipped through the same test hole: **no test had ever driven a credit
   position through the real `_evaluate_position`** — pytest was green while pre-fix all 8
   cells of the new `tests/test_credit_exits.py` matrix ERROR or no-exit. That suite +
   `tests/test_qualify_batch.py` + the 07-13 incident replay in `test_order_lifecycle.py`
   now pin the contracts (18 tests, fail-verified against pre-fix code).

### Shadow referee (`scripts/shadow_referee.py`) — first run: 8 BREAKs
Independent (imports nothing from src/ait), read-only, recomputes the track record from raw
broker executions. Day-one catches: the 07-13 triple-fill (independently); **booked P&L wrong
on all 3 verifiable condors** — track record booked +$280.20 / 7W / PF 3.03 vs
broker-corrected **+$254.90 / 6W / PF 2.64** (QQQ's booked +$6.80 WIN is a real −$15.71 LOSS
— books recorded the phantom re-close price + a flat $0.65/leg commission estimate);
**slippage gate had literally NO data** (every execution row live_mid/signal_price = 0 —
R11's event-driven fills deleted the pending entry before the sweep captured context; FIXED
via placement-time context map + self-healing upsert, measuring starts with the next fill);
**DD-on-deployed-risk methods disagree across the 8% gate line** (system 13.8% FAIL vs
concurrent-risk 7.4% PASS — method must be pinned before more data accrues); DuckDB mirror
commission column stuck at $0 (FIXED: re-ingest on commission update + one-time backfill at
redeploy). **Scoreboard restatement decision pending**: book exits from first-closing-group
+ real commissions (referee method) — do it once, at the redeploy, with a DB backup.

### Also fixed same-day
- Stale WORKING exit orders had no timeout path (found writing the incident regression):
  the >300s cancel + 900s zombie cap were unreachable while an order rested at the broker —
  a runaway market could wedge a position in CLOSING with a stale-priced exit. Now: one
  cancel request at >300s, tracking kept, terminal verdict books or CAS-reverts.
- `qualify_contracts_batch` returned a SHORTER list on partial failure (ib_insync 0.9.86
  DROPS failures; the wrapper comment claimed conId==0-in-place) → positional consumers
  could misalign conId-to-leg on a condor. Now N-for-N/None from the input objects.
- urllib3/requests silenced to INFO (a live FINNHUB_API_KEY sat in 11 log files via DEBUG
  URL logging — same class as the R6 ibapi flood; key rotation = user action).
- `constraints.txt` checked in (136 pins from the RUNTIME interpreter; R10 ordered this,
  never done). ib_insync==0.9.86 + Gateway 1044 documented as a load-bearing pinned PAIR
  (lib archived Mar 2024, author deceased; successor `ib_async` 2.1.0; IB is removing legacy
  wire methods on a forced-upgrade schedule — migration is a dedicated post-verdict change).

### Human-factors lens (rerun, completed later 07-14 — all executed evidence)
- **Config typos were silently swallowed** (pydantic `extra="ignore"` everywhere): executed
  proof — `max_contracts_per_trad` (one letter) silently traded **10× size**;
  `paper_trading_mod` silently re-enabled live-only learning overlays (the exact R5 class);
  zero warnings, and the deploy smoke only proved the yaml parsed. FIXED: `extra="forbid"`
  on every config model (SentimentConfig stays `allow` as the tombstone exception) + smoke
  sentinels (1-lot, paper_trading_mode, mode=paper). Real config.yaml verified loading clean.
- **The bot fights manual intervention**: after a manual TWS flatten, the monitor still
  demanded the exit and `_execute_exit` places the reverse combo with NO broker-position
  check → rebuilds the position inverted (07-13 incident class, operator-triggered). Also:
  flattening the LAST option position is never booked (zero-options mass-close guard), and
  a one-leg manual close would be re-opened reversed. SOON: broker-position check before
  reverse combos + guard escape hatch. NOW: RUNBOOK "manual intervention procedure" box.
  **U5 flow verified SAFE** (untracked positions have no exit path; close at broker →
  verify flat → delete HALT_UNTRACKED; deleting early just re-freezes).
- **Backup mirror was silently stale**: unclosed sqlite connections + no content verify —
  the off-box mirror was missing ALL of Monday's closes while the success log, digest, and
  the RUNBOOK's timestamp check all read green (copy2 back-dates mtimes). FIXED: explicit
  closes before the mirror copy + sha256 verify raising into the existing BACKUP FAILED
  alert. SOON: digest reports mirror CONTENT age.
- **Alert fatigue**: 07-10 produced 136 Telegram sends, 62 of them the identical unthrottled
  CIRCUIT BREAKER message (every 5-min cycle while tripped). FIXED: once per trip + hourly.
  RUNBOOK rows added for CIRCUIT BREAKER / LOOP IMPAIRED / component-down.
- **The entry freeze was unobservable**: `entries_halted` never logged in any file back to
  07-09 — including 90 min of frozen RTH today — because the check sits below the scan
  gates; the RUNBOOK drill grepped for exactly that event. FIXED: `entries_frozen` logs
  once per scan cycle whenever a HALT* file exists; drill updated.
- **Startup chain coherent but its failure is silent**: keeper never verifies a relaunch
  succeeded and never alerts (SOON) — and the three compensating layers are all STILL
  unarmed: U1 dead-man, U2 auto-logon, U3 w32time (the fundamental blockers here).
- RUNBOOK staleness fixed same-day: retired duckdb-pollution warning (fixed since R12),
  drill table filled (backup restore 07-11 PASS), "leave it and note it" untracked-option
  advice corrected (the file re-creates at the next reconcile — not an option the code
  supports).

---

## Round 14 (in progress) — the R13 SOON list, ordered by exposure
All evidence is already gathered (R13 lens reports). Ship in out-of-RTH windows while the
sample accumulates.

**Tier 1 — live exposure in a bad tape**
1. ✅ **Exit-price sanity bound.** *(shipped `dae9c64`)* The exit path had no ceiling:
   `ask + $0.10` with no cap, plus a MARKET fallback on a 4-leg BAG exactly when quotes
   evaporate. Executed proof: a $2-wide condor with a garbage 9.90 ask placed `BUY LMT 10.00` —
   5× the wing width, i.e. 5× the structural max loss. Shipped: credit buyback capped **at** the
   wing width (capping *below* it would be non-marketable on a deep-ITM condor, which is
   legitimately worth ~width, so the order would re-place forever while the ITM short carried
   assignment risk); **debit close capped at the crossing buffer** — the mirror hole the
   credit-only cap missed entirely, and the open IWM straddle's exact path: a sign-corrupted
   +9.90 ask made the bot place `BUY LMT 9.95`, *paying* to dispose of a position it owned;
   never MARKET a multi-leg BAG; and a computed limit of exactly `0.00` is a real marketable
   price for a debit close, not "no quote" (the `!= 0` truthiness test deferred a perfectly
   closeable position indefinitely).
2. ✅ **Broker-position check before reverse exit combos** (human-factors #2). *(shipped
   `dae9c64`)* After a manual TWS flatten the monitor still demanded the exit and `_execute_exit`
   reversed the legs with no broker check → REBUILT the position inverted. Same end-state as the
   07-13 incident, operator-triggered, needing no bug at all. Shipped as **three-state**
   `reconciler.position_liveness(trade)`, because each answer has its own catastrophe:
   - `gone` → refuse **and book the trade**. Refusing alone *strands* it: `reconcile()`'s
     zero-options guard ("refusing to mass-close") fires on exactly the state a flatten of the
     **last** position leaves, so its stale-local loop never books the row — the trade would sit
     FILLED forever, re-demanding an exit that is refused every pass. Booking logic is now shared
     with `reconcile()`, not duplicated.
   - `partial` → refuse and page. Reversing all legs of a half-flattened structure SELLS the
     already-flat ones, *opening* new inverted positions (a naked short in the worst case).
   - `unknown` → **proceed**. An empty position cache is indistinguishable from a wedged feed
     (ib_insync's startup `reqPositions` can time out with `connected` still True), so `gone` is
     only returned after an authoritative broker re-query (`get_positions_fresh`). Trusting the
     cache would disable stops exactly when the broker link is flaky.

   *48 tests; every guard is mutation-checked (reverting it fails its test). Findings 1's debit
   bound, the strand hole, and the partial/wedged states were all found by an adversarial review
   of the first draft of these two fixes — i.e. the fixes shipped with bugs of their own until
   that pass ran.*
3. ✅ **Staleness gate on exit inputs.** *(shipped `935e897`)* `Quote.timestamp` was write-only
   and `quality.validate_quote` had zero call sites; the touch stop fired on a single unvalidated
   spot read. Shipped: exits read the quote through `_spot_quote`, which returns a health verdict
   (`fresh`/`degraded`/`frozen`/`missing`). Fresh fires immediately (a touch rarely recovers);
   degraded/frozen require `touch_confirm_ticks` (default 2) agreeing looks, and frozen pages once
   per outage. The load-bearing call is **frozen vs. old**: the bot runs delayed data (type 4),
   which ticks *behind* but still *moves*, so an absolute-age rule would disable the touch stop
   outright — a feed that stops *advancing* is the real failure. A confirmed touch off a bad feed
   still FIRES (missed breach = wing width, unbounded on a strangle; false = the spread).
   `touch_confirm_ticks=1` restores pre-R14 behaviour. Also fixed a latent bug: a missing
   underlying quote used to `return None` and abandon the whole evaluation, killing the DTE safety
   exit; it now degrades only the two spot-dependent checks. *11 tests, all guards mutation-checked.*
3b. ✅ **Single-leg exit price bound** *(shipped `081be1b`)*. `_close_single_leg`: a SELL-to-close
   (long option) is bounded — worst fill is $0, premium already sunk — so it keeps a marketable-
   limit-then-market fallback. A BUY-to-close (short buyback) is the unbounded-fill catastrophe, so
   it is a LIMIT, never a market order — short puts capped at the strike (full intrinsic is the
   ceiling), short calls at the marketable ask (no structural cap), no-quote defers + pages. The
   retired `covered_call`/`cash_secured_put` short shapes are now safe to bring back.

**Sizing base (the "base = capital-at-risk" question)** ✅ *(shipped `081be1b`)*. The live sizer's
   contract COUNT was already correct (risk/manager SR-H3 sizes credit off `max_loss`, not the
   premium — the "~4× off" read was wrong). The real bug: the RETAINED per-trade risk
   (`capital_at_risk` and the `trade_maxloss_*` KV the aggregate guard sums) was stored
   **per-contract**, since every signal is built at quantity=1 but executes `adjusted_size`
   contracts. Dormant at 1-lot small-account sizing; understates the verdict denominator and the
   aggregate cap by the lot count the moment sizing scales past 1 — i.e. exactly at Phase-2. Now
   `per-contract max_loss × executed quantity` via `_position_capital_at_risk`. **This resolves the
   `capital_at_risk` backfill dependency in D2/Tier-2 item 5 for all trades entered from here on.**

**Tier 2 — correctness of the verdict itself**
4. **Book realized P&L from executions** (the D1 fix in code): first-closing-group price + real
   commissions, not the phantom-fill price + a flat estimate. Without this, D1's restatement
   just re-accrues the same error on the next close.
5. **DD-method pin** (D2 in code): max concurrent deployed risk as the denominator; backfill
   `capital_at_risk` where derivable. *Note: new trades now store `capital_at_risk` correctly
   (per-contract × quantity, `081be1b`) — this item is now just the HISTORICAL backfill for trades
   entered before that fix.*
6. **Commission attribution**: key on exec_id/orderId with an EOD sweep that re-stamps
   already-closed trades; keep phantom/unattributed costs in a separate incident bucket.
7. **`executions.exec_time` is +4h off** (already-UTC broker time double-converted as if ET).

**Tier 3 — resilience / ops** — *all shipped `081be1b`; 26 tests, mutation-checked.*
8. ✅ **MTM brake gap-blindness**. The SOD unrealized baseline was captured AFTER the open, so a
   −8% gap registered `mtm_day ≈ 0` and the brake never tripped. `_post_market` now pre-stamps the
   next trading day's baseline with the prior close's unrealized (`_prestamp_mtm_baseline`); the
   monitor's lazy write only fires on an empty key, so the pre-stamp wins.
9. ✅ **Exit-reject backoff + alert counter**. A rejected exit re-fired every 30s forever, paging
   nobody. A re-request IS the reject signal (a live exit sits in `closing_ids`), so each rapid
   strike widens an escalating backoff (30/60/120/180s) and pages at 3; the slow ~300s
   working-order re-price cadence resets the count so price-chasing isn't mistaken for a storm.
   *(Halted-contract awareness via a broker `errorEvent` hook is still net-new — deferred; the
   backoff + page covers the fixed-cadence/unbounded/silent parts.)*
10. ✅ **Zero-options mass-close escape hatch**. `reconcile()` now does the same authoritative
    re-query as `book_vanished_trade`: a position that vanishes with no exit pending (flattened
    after hours, bot restarted) is booked only when a FRESH query confirms zero options, and only
    the FILLED/PARTIAL trades — PENDING orphans still route to the stale-pending sweep, stocks are
    never auto-closed, and None/unconfirmed keeps the refusal.
11. ✅ **Keeper relaunch verification + digest content-age**. `start()` now verifies the child
    survives a grace window and pages on an instant death (was: Popen-and-assume, blind for 2 min).
    The digest reports the off-box mirror's CONTENT age (newest trade timestamp inside the mirror
    DB) instead of a local snapshot's `copy2`-back-dated mtime — the signal that read green while
    the mirror went a full run stale on 07-14 — and says MISSING loudly if the mirror is gone.

**Parked deliberately**
- **ib_async migration** (ib_insync is archived; author deceased). Pinned pair `ib_insync==0.9.86`
  + Gateway 1044 holds for now via `constraints.txt`. Migrate AFTER the paper verdict — a
  broker-library swap mid-sample would contaminate the very evidence it exists to protect.
- **Resting GTC take-profits** (R12 Tier B): worth 6-17% of each win and they survive bot
  death, but it is a new order path with double-fill risk — the same class as the bug that
  fired on 07-13. It gets its own change with its own verification, post-verdict.

### USER ACTIONS (new — see table at top: U7-U12)
Repo is PUBLIC; Gateway API binds 0.0.0.0 + IBC auto-accepts (one Wi-Fi-profile flip from a
LAN-exposed order API); MySQL reachable on Public Wi-Fi NOW; C:\IBC ACL world-readable;
Finnhub key rotation; Docker 2375.

### Right-sized OUT (decisions, not oversights — unchanged from planning)
Market-impact analysis at 1-lot; formal model governance (SR 11-7); HA/DR beyond current;
privacy.

---

## Round 12 (2026-07-13/14) — sophistication & maturity audit + full execution ("all of it")

Six lenses no prior round used: concurrency/state-machine, trade-lifecycle craft, chaos walks,
vol/pricing craft, over-engineering, test maturity. Decision sheet: `docs/AUDIT_R12.md`.
**All three tiers executed.**

### Tier A — hardening (shipped)
- **Trade status is a real state machine now.** Six writers did blind UPDATEs, and illegal
  edges were reachable: CLOSING→FILLED after a restart (duplicate close → **position
  REVERSAL**), an exit partial-fill-then-cancel treated as a clean cancel (**oversell → naked
  short**), PARTIAL→FILLED losing quantity, CLOSED resurrection. Fixed with a compare-and-swap
  `transition(from_statuses → to)` that every writer must use (illegal transitions are now
  logged no-ops), filled-qty-first exit logic with partial-exit booking, and startup
  re-adoption of broker-side working exit orders. 34/34 execution proofs.
- **clientId-fallback blindness**: a mid-session reconnect on a fallback id made all working
  orders invisible → mass false-CANCELLED while real orders worked at the broker. The fallback
  path now snapshots `reqAllOpenOrders` and refuses "cancelled" verdicts for foreign orderIds.
  An untracked option's FIRST sighting now reports an EOD break same-day.
- **Event-fill robustness**: the debounce flag could wedge permanently (silently reverting to
  30s polling), tasks were GC-cancellable, the busy-guard dropped events. All fixed.
- **The PENDING row now precedes placeOrder** (a DB failure used to leave a live untracked
  order at the broker).
- **Test/CI repair**: CI as shipped could never go green (it ran the ~1h GARCH suite inside a
  30-minute timeout, with `-x`). Now `slow`/`ibkr` markers, a fast lane by default, a nightly
  slow lane. The 8 event-loop flakes were root-caused (sync tests calling `get_event_loop`
  after pytest-asyncio closes it) and fixed; stale reds rewritten. New: `tests/fakes.py`
  (shape-faithful broker fakes — MagicMock is what hid the `shares=contracts` bug class),
  `tests/test_order_lifecycle.py` (drives the REAL executor through place→fill→exit→close,
  including partial-then-cancel and disconnect-blip), `tests/test_properties.py` (hypothesis:
  ladder monotonicity, P&L sign invariants by CREDIT_STRATEGIES membership, wing-width bounds).
- **Found and fixed mid-audit: pytest was clobbering LIVE model artifacts** — `range.pkl` and
  `ensemble.pkl` overwritten with synthetic-noise models (the R7/R11 fence covered walkforward
  saves, not test saves). A conftest fixture now isolates every test model write; artifacts
  were rebuilt from real data.

### Tier B — strategy (user-approved, evidence-backed)
- **Stop recalibrated on evidence.** 17-month walk-forward, 26 matched trades, identical
  entries across arms: **touch-close +$756 (PF 1.38) > no stop −$24 (0.99) > 1.25× flat −$134
  (0.96) > 2.0× flat −$1,067 (0.73)**. The flat stop fires THROUGH its trigger on gaps (worked
  example: a −1.25× trigger realized −2.01×). The backtest's average loss at short-strike touch
  was **0.475× credit** — independently confirming the audit's 0.49× derivation.
  **SHIPPED: the flat credit stop is DISABLED by default** (`AIT_CREDIT_LOSS_LIMIT=0`) and the
  **short-strike-touch close is the early loss exit** (`AIT_CREDIT_TOUCH_STOP=1`). It reads the
  UNDERLYING price, so it protects even during an option-marks outage. The wings cap the tail.
- **The designed trade is now enforced.** The 07-07 SPY condor filled with its short call at
  **0.49 delta** (intended 0.20) and passed every gate *because* of the error — credit floors
  monotonically reward closer-to-ATM strikes and no upper delta bound existed ("the gate stack's
  equilibrium, not an accident"). SHIPPED: delta band |Δ| ∈ [0.15, 0.30]; expected-move sanity
  gate (shorts within [0.6, 1.3]×EM); ATM IV + expected move extracted from every live chain
  (killing the ±50% flat-0.20-IV wing error); degraded-greeks rejection; one
  `condor_entry_quality` telemetry line per generated/rejected condor. *A replay of the 07-07
  incident chain is now rejected at strike selection.*
- **Cadence**: minimum entry DTE 14 (7-DTE entries were 2-day churn, best case ~60%
  cost-dominated); per-expiry cap of 3; **post-stop re-entry cooldown keyed on EXIT time** (the
  old cooldown keyed on entry time, so a stopped symbol could re-enter on the next scan into the
  same trend); VIX-tiered credit cap (6/4/2 by VIX band); IV-percentile cap at 85 (the variance
  risk premium in the top bucket measured −2.3 vol points, yet the floor said "trade").
- **Deferred deliberately**: resting GTC take-profits at the broker — worth 6-17% of each win
  and they survive bot death, but it is a NEW order path with double-fill risk, the same class
  as the CLOSING→FILLED bug just fixed. It gets its own change with its own verification.
- **Confirmed keep**: mechanical no-adjustment discipline (no rolls) — correct at 1-lot defined
  risk. The reviewing agent argued both sides and committed.

### Tier C — deletion slate (user-approved, zero live-behavior change)
**~6,650 source LOC + ~1,800 test LOC out of the live tree**, all retired-not-deleted with
history into `deprecated/src/` and `deprecated/research/`:
- **`hedging.py` — the dangerous one.** Dead since inception (greeks feed ~0) AND its
  auto-execute path placed SPY **stock market orders** straight through the broker client,
  bypassing every executor guardrail (rate limit, defined-risk gate, NBBO fat-finger check,
  market-hours guard). One `auto_hedge: true` flip from live.
- **Sentiment stack** (engine/news/finbert/fear_greed + ib_news + fundamentals_db, ~1,250 LOC):
  R7 traced its contribution to iron-condor decisions to exactly zero. Salvage, if ever wanted:
  a ~30-LOC VIX-backwardation gate in `risk/`.
- **torch + transformers → optional `sentiment` extra** (580MB serving a hard-disabled FinBERT;
  torch is forensically implicated in the historical c0000005 crash cluster). `arch` → `research`
  extra. Zero live importers, grep-verified. Install and CI time roughly halve.
- **Dead code**: the signal queue (unreachable post-R7, inside the most safety-critical file),
  `options_sim.py` (carried a 5×-premium tail-cap trap; its live pricing functions were extracted
  to `backtesting/pricing.py`), `directional.py`, `covered.py` (a naked-call generator),
  `options_flow.py`, `calendar.py`, `event_straddle.py`.
- **GARCH / MS-GARCH / OU → `deprecated/research/`** (~2,900 src + ~1,200 test LOC). It was
  already OFF live by deliberate decision, and R5-D2 showed its validation evidence untrustworthy
  in both directions — so the enable decision was blocked anyway — yet it consumed more repeated
  audit attention than any live module. Wake path unchanged: fix the validation math, produce a
  credible backtest, re-enable. **The live range gate (XGB+LGBM) was NOT touched** — its fate
  belongs to the pending ML ablation.
- **Simplified**: `capital_tiers` 250→144 LOC; `counterfactual` keeps `record_skip` but its
  "analysis" outputs (twice ruled misleading) are gone; `meta_label` default-off.

---

## Round 11 (2026-07-11) — R9/R10 adoption
Event loop UNBLOCKED (the risk monitor had been blind **48% of market hours** — synchronous I/O
inside async scans; proof: max ticker gap 1171ms → 116ms). Event-driven fill detection (33s →
~1s). Scan stage timing. Model-artifact separation + spec-mismatch guard (backtests had been
hijacking the live range gate: production ran 8.67%/21d instead of the designed ±5%/30d). The
first CI this repo ever had, plus `scripts/smoke_deploy.py` (12 runtime checks — it immediately
caught pytest polluting the production DuckDB). Scorecard TCA. **The slippage gate got a NUMBER
before data could bias it: median entry slippage ≤ 8% of credit over the trailing 20 fills, no
worsening trend.** Backup-restore drill REHEARSED (PASS).

## Round 10 (2026-07-11) — maturity vs external checklists (`docs/MATURITY_R10.md`)
The first audit to measure the project against *external* standards (trading-ops, SRE, MLOps,
SDLC) rather than its own intent. Verdict: engineering maturity far above $3k scale, but
**measurement existed with zero read-out**, and **last-resort layers had never been exercised**.
Drove the R11/R12 adoptions.

## Round 9 (2026-07-11) — first benchmarks ever (`docs/BENCHMARKS_R9.md`)
Latency: exit decision→order 170ms (good); fill detection 33s (poll-bound — since fixed); **the
risk monitor blind 48% of RTH**; scans at ~10.4 min against a 5-min design; a full retrain on
every boot (~110s unprotected). Accuracy: effectively never settled — the one settleable day
scored **0/12 direction hits at 0.92-0.98 confidence**; **risk caps, not ML, decided 164 of 165
entry decisions**. Statistics: **50 closes needs 42/50 wins just to confirm PF>1.0 at 90%
confidence** — a decisive PF>1.3 verdict realistically needs 100+.

## Round 8 (2026-07-11) — execution-based runtime audit
The lesson that stuck: **agents must EXECUTE code, not read it.** `StateManager._connect()` never
existed — all four R7 ledger methods raised AttributeError, swallowed at their call sites, so the
entire real-cost ledger was dead on arrival (executions empty, commissions $0). The DuckDB mirror
would also have silently stopped syncing at the next close. Both fixed, execute-verified,
regression-tested. Also: the watchdog→Telegram path was wired (dead since birth — 661
component-down events had paged nobody), error-streak escalation added, heartbeat gated on error
streaks.

## Round 7 (2026-07-09) — component gap audit (`docs/GAP_AUDIT_R7.md`)
13 agents, one question: "what's MISSING to make money" per component. Shipped: the earnings
parser (it had been returning DIVIDEND dates — AAPL read as a past date, AMD as 1995, so all
earnings guards were dead for the single names), VIX halt fail-CLOSED, the learning layer made
inert in paper mode, the real-cost ledger, NBBO-anchored mid-first entry ladder, cross-symbol
best-first selection, TRUE IV rank (2 years of implied vol backfilled), the go-live scorecard.
Plus budget-aware condor construction and a launch-size cap-coherence self-test — the $2.1k
account previously could not trade at all (the sizer's multiplier chain returned 0 contracts).

## Round 6 (2026-07-09) — strategy + ops batch
Credit-aware exits (killing the equity-style trailing-stop scratch machine: a condor peaked at
+32.3% of credit and was stopped out at +$5.80). Short strangles retired from entries (go-live is
defined-risk only). Vol-scaled condor wings and credit floors. The correlation cap actually
enforced (the book had reached 5 index-cluster positions against a cap of 2). The ibapi log flood
silenced, hang detection, the daily digest, dead-man ping plumbing (still unarmed — see U1).

## Round 5 (2026-07-08) — coverage-gap audit
8 agents over everything prior rounds had touched lightly. The biggest: `paper_trading_mode` was
NEVER actually set, so every "bypassed in paper" claim in the codebase was false. Also: combo
leg-shape validation, a paper-vs-live account assertion, the backtest entry-window UTC→ET fix
(**every past backtest had only ever entered between 13:30-15:30 UTC**), dashboards querying a
nonexistent table (and therefore rendering permanently green), and stale-doc warning banners.

## Rounds 1-4b (2026-07-07/08) — foundation
Rounds 1-3: the P&L was fiction (+$148k reported → −$12k real); marketable entries fixed (nothing
had been filling); BAG fill reconstruction was 100× too small; the PDT guard was inert; the
watchdog could never trip; credit sizing was 4-5× understated; expired-ITM credit was booked as a
full WIN. Round 3 was line-by-line across ~40k lines (8 agents, 100% coverage attested).
Rounds 4/4b (institutional lenses): daily DB backup + off-box mirror, the `data/HALT` kill switch,
untracked-position auto-freeze, an executor rate limiter, **defined-risk-only enforced AT the
executor**, the combo fat-finger check, model lineage on every trade, the EOD break report with
the book's −10% gap-stress number, the RUNBOOK, and a live-profile startup assertion.
Landmark finding: the paper book's −10% gap stress was **−$5,937** — −280% of the planned live
account. Empirical proof of the defined-risk-only gate, using our own positions.

---

## Go-live gates
*(decided 2026-07-07; amended by R9 statistics and R12 evidence — do NOT relitigate on a winning
streak)*

1. **Paper verdict**: ≥ 50 real closes for a PRELIMINARY read; **≥ 100 closes before funding**
   (R9: at n=50 you need 42/50 wins just to confirm PF>1.0 at 90% confidence — 50 cannot settle
   PF>1.3). Required: PF > 1.3; max drawdown < 8% **of deployed risk** (not of paper NLV);
   **median entry slippage ≤ 8% of credit over the trailing 20 fills, with no worsening trend**;
   zero unmanaged-position incidents. Criteria fixed BEFORE looking at results.
2. **Defined-risk only**: iron condors (and debit structures) ONLY. NO short strangles live —
   their max_loss is a model estimate, not a contract. Enforced at the executor.
3. **Tuition sizing**: $3,000 CAD (~$2.1k USD) → SMALL tier. Phase-switch recipe (run BEFORE
   funding): `.env AIT_SIMULATED_CAPITAL=2100` + `risk.max_position_risk_pct: 0.03 → 0.07`,
   restart, confirm the coherence self-test passes, run ≥ 4 weeks.
   **R12 caveat — the two-animals problem**: paper is currently earning the verdict on ~$22-wide
   condors while the funded account will trade ~$2-wide ones. **Phase 2 must therefore be the
   PRIMARY evidence, not a smoke test.**
   Economics honesty: ~$5-13 per iron-condor round trip against $20-45 targets. The $3k stage
   validates PROCESS with real money, not income.
4. **Backtest credibility**: core math fixed (sleeve capital, annualization, window overlap,
   price basis, R6-exit parity + a parameter-parity manifest). Old `reports/` artifacts are
   invalid — re-run before citing any number.
5. **The ML ablation is still open**: R9 showed risk caps (not models) decided 164/165 entries,
   and the range gate has never vetoed anything. Run ML-on vs ML-off through the parity engine
   before claiming the ML earns its place.
6. Expect live < paper (real fills). Scale only on 3+ months of live evidence.
   At go-live: remove `AIT_ALLOW_UNDEFINED_RISK`, tighten the liquidity gates (startup asserts
   this), set `AIT_EXPECT_LIVE_ACCOUNT=1`, and settle the adaptor-override wire-or-delete
   question.

## Dormant / degraded inventory (post-R12 — so nothing resurfaces as a surprise)
| Subsystem | State | Wakes when |
|---|---|---|
| Self-learning (analyzer / adaptor / Thompson) | DORMANT, and provably inert in paper mode (R7 made the adaptor a no-op) | ~30 real closes; decide wire-or-delete at the go-live cutover |
| Range gate + direction ensemble (XGB/LGBM) | LIVE — the range gate gates every condor | its value is unproven: **the ablation decides** |
| GARCH / MS-GARCH / OU | RETIRED to `deprecated/research/` (was already off live) | fix the R5-D2 validation math → credible backtest → re-enable |
| Sentiment, hedging, options_flow, directional, covered, calendar, event_straddle | RETIRED to `deprecated/src/` (all zero-influence; hedging was dangerous) | sentiment only as a ~30-LOC VIX-backwardation gate, if ever |
| Greeks feed / portfolio-delta gate | DEAD (no subscription). Substitutes in place: credit-position cap, VIX-tiered cap, correlation cap | a greeks subscription, or model-greeks fallback |
| Partial-exit ladder | DEAD at 1 contract | contract sizing > 2 |
| Drift detector | keys fixed (R7) but needs 20 samples it cannot reach at 3-4 closes/week; in-memory only | persist it post-verdict, or delete it |
| Meta-labeler | untrained, default-off | 100+ closes |
| IV / IV-rank | TRUE percentile live (2y of IBKR ATM IV backfilled). The feature matrix still lacks `implied_vol` | next feature-engine pass |

## Deferred / watchlist
- **Resting GTC take-profits** (R12 Tier B, deliberately deferred): worth 6-17% of each win and
  they survive bot death — but they need careful double-fill handling.
- Adverse-selection analysis (a real cost at any size, unlike market impact).
- The range gate models a fixed-% band, not the trade's actual profitable zone (measured
  gate-vs-P&L correlation: +0.18). The upgrade is a trade-matched P(no-touch of the actual short
  strikes).
- Skew-aware per-side delta targeting — post-verdict only; do not contaminate the sample.
- IBC read-only quirk: some Gateway restarts may still need a manual check→uncheck→Apply toggle
  (Telegram-alerted).
- Display-side `sqrt(252)` sites in research/export code; backtest fill realism (item 3.4).

## Operating rules (learned the hard way)
- Restarting the BOT ≠ restarting the GATEWAY. Entitlements and read-only state load at Gateway
  login.
- ONE live-data slot: the live account must stay logged out of mobile/web while the bot runs.
- Paper ≠ live fills. Paper is optimistic; marketable pricing narrows the gap.
- Secrets: the IBC password lives ONLY in `C:\IBC\config.ini`, outside the repo. Never in git,
  chat, or docs.
- **Deploy discipline** (bought with a lost trading session on 07-10): targeted tests → deploy
  outside RTH → `python scripts/smoke_deploy.py` must print SMOKE PASS → after the next open,
  confirm a clean cycle (fresh heartbeat, `scan_symbol_timing` events, no LOOP IMPAIRED alert).
  Green tests are NOT a deploy signal on their own.
- **Fix velocity mints new audit surface.** Half of R8's and R12's findings were in code written
  days earlier by these very audits. Audit the fixes, not just the original.
- **Fix CLASSES, not instances.** The artifact fence covered walkforward but not tests; the CAS
  transition guard covers every writer, present and future.

## Round 15 (2026-07-21) - main-vs-branch adversarial review + fixes
29-agent review of origin/main...HEAD: all 7 areas mission-ALIGNED, nothing of value lost from main, 8 defects confirmed (2 skeptics each) and ALL FIXED same day (cad16d5, 17 tests, suite 755 green): intrinsic sign-flip on single-leg debit expiry booking; PENDING rows booked stale while entry orders worked; MTM halt never un-tripping (reason-token mismatch); paper/live assertion validating config not session; foreign-order stash missing the fallback-to-base leg; naive-UTC vs naive-local quote stamps blinding the staleness gate; pytest clobbering live meta_label.pkl; realized_pnl embedding the flat commission estimate (now trued up from the ledger at booking - Tier-2 #4 closed). Plus: first-hour gate no longer suppresses exempt condors; vix=None no longer crashes debit sizing.
KNOWN RESIDUE: models/meta_label.pkl was clobbered by a test run Sat 07-18 11:04 and the daily retrain has NOT run since 07-17 (investigate scheduler); inert while paper_trading_mode disables the meta-label gate, but restore before it re-arms.

## 2026-07-22 - IC-ONLY pivot (user decision) + straddle resolved + R15b
Strategies pinned to [iron_condor], universe [SPY,QQQ,IWM]. Straddle manually flattened (299C@0.49/299P@3.63) after days stuck in the no-quote defer loop. R15b: zero-position IBKR rows now filtered in all liveness/matching paths (found by the hatch's first live use). Catch-up retrain shipped+fired (first since 07-17). Machine-side U-items done: U4 OneDrive mirror, U8 IBC reject, U10 logs clean, U11 ACL, U12 Docker. User-side: U1, U2, U3+U9 (elevated), U7 (browser), U6 (fund ~USD505 - THE blocker).

## 2026-07-28 - wing_k study (PRE-REGISTERED decision rule, written BEFORE results)
Sweep k in {0.6, 0.8, 1.0, 1.2}, iron_condor only, SPY/QQQ/IWM, all other params at live values (ratio floor 0.20, credit floor 0.70, delta band). RULE: require OOS trades >= 30 AND PF > 1.0; among qualifiers pick highest PF; ties within 0.1 PF -> higher trade count; if NONE qualifies, KEEP k=1.0 and wait for vol (no gate loosening). Chosen k gets locked in config + backtest before the first counted sample close.

## 2026-07-29 - wing_k study RESULT + DECISION: k=0.8 LOCKED
Dense sweep (train126/test63/step42/gap5, 1049 rows ~4.2y, SPY/QQQ/IWM, IC-only, ratio floor 0.20): k=0.6 -> 21w/39t PF1.06 win53.9 DD5.95; k=0.8 -> 21w/39t PF1.07 win56.4 DD5.13; k=1.0 -> 16w/25t PF0.84 ret NEGATIVE; k=1.2 -> zero trades (wings too wide to ever pass ratio). RULE APPLIED: qualifiers 0.6+0.8; PF tie within 0.1 -> trade count ALSO tied at 39 -> rule exhausted; resolved to k=0.8 by reverting to primary metric (higher PF) which also dominates win rate + DD. RESIDUAL JUDGMENT DISCLOSED. Caveats: window overlap ~1.5x (effective n ~26, below the raw-30 bar the registered rule counted); OOS PF barely above 1 - THE EDGE EVIDENCE IS THIN, which is why the live 50-close sample remains the verdict. k=0.8 locked in config.yaml (backtest) + AIT_IC_WING_K (live); narrower wings raise credit/width so condors can clear 0.20 at lower IV. Study: scripts/wingk_study.py, artifacts reports/wingk_study/.

## 2026-08-03 - ML ABLATION (PRE-REGISTERED rule, written BEFORE results)
Study: walk-forward SPY/QQQ/IWM iron_condor k=0.8, GATE-STACK-ONLY vs FULL-STACK (ML
confidence/range/meta gates active). RULE: ML "earns its place" ONLY if full-stack shows
BOTH higher PF AND lower max DD than gate-only on >=30 OOS trades each; anything less ->
REMOVE the ML gates from the entry path (simplicity wins ties). Harness note: engine ML
toggle to be wired next session; rule locked now. Also 2026-08-03: straddle FINAL
restatement to broker ledger truth -575.33 (record: 13 closes 6W-7L -$451.94); referee
0 BREAKS after fill attribution; corrupt meta_label.pkl quarantined (retrain does not
cover it - regenerate before meta gate ever re-arms).
ABLATION HARNESS (next session, ~30 min): engine already treats predictor=None/untrained as
the UNGATED path (walkforward.py:564,575 - gates apply only if is_trained). Wire: add
WalkForwardConfig.train_window_models: bool = True; skip the per-window train call when
False; clone wingk_study.py -> ablation_study.py running k=0.8 with flag False (arm A,
gate-stack-only) vs True (arm B, full-stack); apply the pre-registered rule above.

## 2026-08-03 - SHADOW STRATEGY PIPELINE (user-approved; PRE-REGISTERED before any results)
Live stays IC-ONLY until its verdict. Candidates compete in SHADOW (walk-forward + optional
live signal logging, zero orders): (1) iron_butterfly (low-IV complement), (2) vertical put
credit spread (2-leg cost profile). RULE per candidate: same harness as wing_k study, >=30
OOS trades, earns a live slot ONLY if PF > 1.2 AND max DD% <= IC's on the same windows;
else stays shadow. Implementation notes: backtest engine needs an iron_butterfly builder
(clone condor path, ATM shorts, same wing logic) and a 2-leg put-spread builder; run after
the ML ablation completes. No live config change without a passing pre-registered result.

## 2026-08-03 - ML ABLATION VERDICT (rule pre-registered above; CONFIRMED x2)
gate_only vs full_stack vs full_stack_v2 (raw-logged, ensemble predictions verified
flowing at 0.8+ conf): ALL IDENTICAL - n=58, win 58.62%, PF 0.86, DD 9.78%, ret -0.73%
over 25 windows. The ML gates veto NOTHING (confirms R9: caps decided 164/165 live).
RULE APPLIED: ML fails both criteria -> REMOVE from the entry path. Note: removal is
behaviorally a NO-OP (proven identical trades) - its value is simplification. EXECUTION
(next task, fresh session): config-level disable of ensemble-confidence/range/meta entry
gates, test-pinned, BEFORE the first counted IC close so the sample tests the gate stack
alone. Models keep training for future studies (meta-labeler at 50 closes still planned).
Then: shadow tournament builders (IB + put credit spread, rules already registered).
Also note: PF 0.86 on n=58 dense windows vs 1.07 on n=39 - simulated IC edge flickers
around breakeven; the LIVE sample remains the only verdict that counts.

## 2026-08-03 - SHADOW ROUND 2 (PRE-REGISTERED before results)
Adds: (a) call_credit_spread - promotable under the same rule (PF > 1.2 AND DD <= IC arm,
n>=30); (b) jade_lizard - BENCHMARK ONLY, permanently non-promotable at current capital
(naked put side ~ strike-to-zero risk; sized in-engine by strangle margin convention);
(c) UNCORRELATED UNIVERSE arms - all strategies re-run on XLE/GLD/TLT; a universe earns
live inclusion ONLY if its IC arm passes the same promotion bar on its own windows.
Runs chained AFTER shadow round 1 completes (CPU contention). Same harness, k=0.8.

## 2026-08-03 - SHADOW TOURNAMENT VERDICT (rules pre-registered above; raw logs = source)
R1 SPY/QQQ/IWM (~4y, 28 wf windows): IC n=89 PF 0.97 DD 18.4% | iron_butterfly n=73
PF 0.31 DD 29.2% | put_credit_spread n=7 (gate rarely clears) | short_strangle BENCH
n=53 PF 1.30 DD 1.2%. R1b: call_credit_spread n=7 PF inf (100% win - small-sample,
no verdict) | jade_lizard n=5 PF 0.79 (no verdict). R2 XLE/GLD/TLT: IC n=37 PF 0.58 |
IB n=58 PF 0.35 | strangle n=65 PF 0.77 | all spread/lizard arms n=0 (never traded -
premia too thin for gates).
RULES APPLIED: (1) NO candidate promotes - zero arms met PF>1.2 AND DD<=IC AND n>=30.
(2) iron_butterfly ELIMINATED (0.31/0.35 both universes - ATM shorts die by touch-stop).
(3) Vertical spreads + jade lizard: NO VERDICT as parameterized - they barely trade
(n<=7 R1, n=0 R2); re-parameterize (full-width wing / lower credit floor) before any
future round, else drop. (4) UNCORRELATED UNIVERSE REJECTED: XLE/GLD/TLT IC arm PF 0.58
fails its bar; even the wings-free strangle loses there (0.77). SPY/QQQ/IWM stays the
universe. (5) Wings-cost thesis: supported in R1 (strangle 1.30 vs IC 0.97 - wings cost
the whole edge) but does NOT transfer universes and lizard n too small to confirm.
Next test of the thesis: BROKEN-WING condor round (defined-risk, cheaper wings).
LIVE CONFIG UNCHANGED: iron_condor on SPY/QQQ/IWM remains the sole live arm; the
50-close live sample remains the decider (sim IC estimates now 1.07/0.86/0.97 ~ 1.0).

## 2026-08-03 - PRE-EVENT BLACKOUT 4 -> 1 DAY (user-approved; registered before effect)
Finding: the <=4-day credit blackout (orchestrator) + monthly NFP/CPI/PCE cadence blacked
out ~half of all trading days; 2026-08-03 QQQ passed every quality gate (IV rank 83,
credit/width ok) and was refused solely by d2e=4 (NFP 08-07). Inconsistency: every 14-30
DTE hold spans >=1 NFP and >=1 CPI anyway - the gate only refused ENTRY at peak pre-event
premium (the vol-crush trade a seller is paid for). Wings cap any event surprise; the
1-day window still skips event-eve gamma. CHANGE: RiskConfig.pre_event_blackout_days=1
(was hardcoded 4), live gate + backtest engine parity, test-pinned
(TestPreEventBlackoutRelaxed). Effect: first eligible entry day moves 08-13 -> 08-04.
RISK ACCEPTED: entries now sell into pre-event IV; an event-gap through a short strike
costs at most the fenced max loss (~$80-170/contract). Sample-attribution unchanged -
same strategy, same sizing, one gate parameter.
ADDENDUM 2026-08-04 (same decision, exit side): run_orchestrator.py sets
AIT_SKIP_MACRO_EVENTS=1, so the rule-3d flatten force-closed defined-risk credit at
d2e<=1 - with entry now allowed at d2e=2 that meant enter Wed, force-flatten Thu (1 day
theta, full round-trip costs, no crush harvest). iron_condor REMOVED from the flatten
list (portfolio.py + engine parity, test-pinned): condors hold THROUGH events; wings cap
the surprise. Undefined-risk unchanged (strangle 5d, CSP/CC 1d).

## 2026-08-04 - SHADOW ROUND 3: WINGS-COST INTERPOLANTS (PRE-REGISTERED before results)
Thesis under test (from R1: strangle PF 1.30 vs condor 0.97 same windows): the wings we
buy cost the whole edge; cheaper/farther insurance should recover part of it. Arms, both
PROMOTABLE under the standard bar (PF > 1.2 AND max DD <= the IC arm's on the SAME run,
n >= 30): (a) wide_wing_condor - both wings at 2x standard k*EM distance; (b)
broken_wing_condor - call wing standard, put wing 2x (skew-priced put insurance moved
out). IC baseline RERUN in the same launch so windows/data are identical. REGISTERED
DEVIATION: credit floor held ABSOLUTE at the standard-width equivalent (ratio floor
scaled by std/max width) - otherwise doubled width mechanically halves the ratio and the
arms never trade (the R1b spread lesson). PREDICTION registered: if the thesis is real,
PF ordering should be condor < broken_wing <= wide_wing < strangle(1.30).

## 2026-08-04 - SHADOW ROUND 3 VERDICT + PROMOTION (rule pre-registered above)
Same run, same 28 windows, SPY/QQQ/IWM: IC baseline n=90 PF 1.31 DD 8.5% | wide_wing
n=93 PF 1.51 DD 6.57% | broken_wing n=92 PF 1.48 DD 5.48%. Registered prediction
CONFIRMED exactly: condor 1.31 < broken 1.48 <= wide 1.51 (< strangle 1.30 no longer
holds - see baseline note). BASELINE NOTE: IC jumped 0.97 -> 1.31 (DD 18.4 -> 8.5) vs
round 1 with ONE change between runs - the entry blackout 4 -> 1 day (engine parity of
the live decision; flatten was already env-off in studies). The 4-day blackout alone
was costing ~0.34 PF and doubling DD in sim: it removed exactly the rich pre-event
entries. Sim IC estimates are now 1.07/0.86/0.97/1.31 - the last is the only one under
the CURRENT live ruleset.
RULE APPLIED: BOTH candidates pass (PF > 1.2, DD <= 8.5%, n >= 30). PROMOTED:
wide_wing_condor - higher PF and a config-only live change (AIT_IC_WING_K 0.8 -> 1.6,
AIT_IC_MIN_CREDIT_WIDTH 0.20 -> 0.10; absolute credit demand per structure unchanged).
broken_wing stays shadow (needs an asymmetric live builder; revisit if wide-wing live
closes diverge from sim). WING_K STUDY RECONCILIATION: the 07-29 k=0.8 lock compared k
under an UNSCALED 0.20 floor, so higher k silently shrank the entry population; round
3's absolute-credit design separates wing economics from population - the lock is
SUPERSEDED for the scaled-floor parameterization. EPOCH MARKER: live closes from
2026-08-04 are the wide-wing epoch (k=1.6, floor 0.10, gates-only, no ML veto, 1-day
blackout, hold-through-events). The open QQQ condor T-20260804-095630 (entered 09:56,
$6.44 credit, k=0.8 strikes) is the LAST k=0.8-epoch position.

## 2026-08-07 - R16 FULL AUDIT (16 dimensions, 118 agents, all findings 2-skeptic verified)
111 raw -> 70 survived verification (+40 unvetted lows). Full detail: session task output
wbkvg8gc1 + workflow journal wf_ebdc88ff-20d. CRITICAL: dead-man (U1) STILL unarmed while
the HOST was shut down (Start-menu, Event 1074) 5 of last 10 nights, returning 10:22-13:43
- market-hours blind with live positions unmanaged 08-05 (4.2h) and 08-06 (2.2h); zero
pages possible (all alert paths die with the box). TOP VERIFIED DEFECT CLUSTERS:
(1) DATA: options_chain picks first SMART chain-def -> adjusted class 2IWM/2SPY/2QQQ
garbage mini-chains; IWM dead since 07-17, SPY eval dead 08-05 pm (4 phantom strikes);
IV RANK FROZEN since 07-09 (all gates on stale snapshot); Yahoo degradation silent with
placeholder 1e-5 IVs. (2) EXITS (2 live positions!): VIX-spike thesis check STILL
flattens condors (contradicts hold-through decision - unowned branch orchestrator.py
~2722); BAG exit live-quote path dead code (qualification always fails); exit sanity
bounds rescaled 8-14x by wide wings; PDT guard can veto same-day stop exits; commission
true-up races partial commissionReports. (3) STATE: CLOSING-never-booked wedge when
broker flat; stale-pending sweep books never-filled while order works; orphaned-child
double-bot (guards match only run_orchestrator); CAS from-set includes FILLED.
(4) EVIDENCE INTEGRITY: all recent studies (ablation, wing_k, shadow R1-R3) evaluated
OOS windows with the LIVE fully-trained ensemble (look-ahead; relative arm-vs-arm
ordering likely survives - same contamination all arms - absolute PFs inflated);
AIT_CREDIT_LOSS_LIMIT live=0/disabled vs backtest=1.25/active parity break; referee
49.9% slippage BREAK is a ledger-semantics artifact. (5) OPS: repo PUBLIC (U7), Gateway
API 0.0.0.0:4002, MySQL LAN-reachable, no auto-logon (U2). daily_stats missing 07-22 row
(+$575.33 cumulative overstatement in daily view). COVERAGE GAPS (critic): broker-vs-DB
ground truth not compared; live exit path unexercised since 07-22; VIX-thesis branch
never executed by any auditor; end-to-end Telegram delivery unproven.
FIX TIERS: T0 USER = U1 dead-man file + power box on by 09:25 (or BIOS auto-on + U2) +
U7 private. T1 CODE (before next entry/exit): VIX-flatten defined-risk exemption; chain
tradingClass fix; IV-rank unfreeze; CLOSING wedge + sweep liveness; TELEGRAM_DEAD reader.
T2: credit-loss-limit parity, research/live model fence, exit sanity rescale, PDT stop
exemption, PLAN evidence annotations. T3: books/TZ/scheduler/dashboard/test hygiene.

## 2026-08-07 - R16 FIX-ALL EXECUTED (fc16531; 3 agents + core cluster; suite 850 green)
40+ verified defects fixed same day - full map in the commit message. Highlights: condors
now FULLY hold-through (VIX/thesis flatten exempted - the last flatten path); chain
tradingClass fix (IWM 2->249 strikes live-verified; IWM universe slot resurrects Monday);
iv_rank store SELF-HEALS from scan ATM IV + freshness gate (was frozen since 07-09 - all
recent iv_rank values suspect); exit limits mark-anchored; look-ahead fence (walkforward
OOS + Optuna can no longer score with the live future-trained ensemble; range-gate block
contract enforced + training status surfaced in summaries); credit-loss-limit parity
live=0; runtime env contract shared by all entry points (undefined-risk gate now CLOSED);
double-bot guards + singleton lock; TELEGRAM_DEAD reader; digest/report/backtest catch-
ups; 2027-H1 calendar. NOT DONE MACHINE-SIDE (classifier-blocked): daily_stats 07-22
backfill - OPERATOR: run `python scripts/restate_r16_dailystats.py --apply`; bot restart
- fixes go live on next boot (nightly shutdown covers it; market closed until Monday).
EVIDENCE CAVEAT NOW FIXABLE: the 08-03 ablation + 08-04 wide-wing promotion ran with the
look-ahead leak + dead range gate; RE-RUN both under the fence before treating either
verdict as evidenced (queued: rerun ablation + shadow R3 with allow_live_model_fallback
=False; relative orderings may survive, absolute PFs will drop). USER STANDING: U1
deadman file, box on by 09:25 (or BIOS auto-on + U2), U7 private repo, MySQL/Gateway
firewall scoping (U3/U9 elevated).

## 2026-08-07 - R16 STUDY RE-RUNS (PRE-REGISTERED before results; supersede 08-03/08-04)
WHY: R16 proved both prior verdicts rested on broken evidence — (a) per-window training
failed in 96/96 windows (only ~27 feature rows survived the warmup at train_days=126 vs
min_training_samples=100), and the engine then fell back to the LIVE future-trained
ensemble => look-ahead leak on every OOS window; (b) the "range model absent -> block
entries" contract never fired (engine skipped the gate when predictor was None), so the
ablation's "gates veto nothing" tested nothing.
RERUN A - shadow R3 (scripts/shadow_round3_rerun.py): iron_condor vs wide_wing_condor vs
broken_wing_condor, train_window_models=False => NO models at all, matching LIVE
(entry_gates_enabled=False). Look-ahead structurally impossible. Same windows/data/k as
the original so the numbers are directly comparable.
RERUN B - ablation (scripts/ablation_rerun.py): gate_only(False) vs full_stack(True) at
train_days=365 (~190 samples clear warmup) so window models ACTUALLY train — the first
honest test of whether ML gates veto anything.
DECISION RULES (registered now):
 A1. wide_wing keeps its live slot ONLY if it still beats the IC baseline on PF with DD
     <= baseline on the SAME rerun windows, n>=30. If it loses -> revert live to k=0.8/
     floor 0.20 (AIT_IC_WING_K/AIT_IC_MIN_CREDIT_WIDTH in ait/config/runtime_env.py).
 A2. If NO arm reaches n>=30 the rerun is INCONCLUSIVE — live config stays as-is and the
     wide-wing promotion is downgraded to "unevidenced, retained by default" in PLAN.
 B1. ML entry gates stay OFF unless full_stack beats gate_only on PF by >0.10 AND does
     not worsen DD, with >=20 windows where the model actually trained (status reported
     in the summary's RANGE MODEL TRAINING section). Any window count of trained models
     ==0 => rerun is void, report as such rather than re-deriving the old verdict.
 B2. Both arms identical again (same n, PF, DD) => gates provably inert -> the 08-03
     removal is CONFIRMED on valid evidence.
Absolute PFs from these reruns are expected LOWER than 08-03/08-04 (the leak inflated
them); the arm ORDERING is the load-bearing output.

## 2026-08-08 - RERUN A VERDICT: shadow R3 under the fence (rules pre-registered above)
Clean run (train_window_models=False => NO models, look-ahead impossible), same windows/
data/k as 08-04, 29 windows SPY/QQQ/IWM:
  iron_condor        n=97  win 82.5%  PF 1.85  DD 12.31%  ret +9.91%
  wide_wing_condor   n=100 win 84.0%  PF 1.91  DD  8.68%  ret +5.47%
  broken_wing_condor n=97  win 83.5%  PF 2.13  DD  5.52%  ret +5.18%
PREDICTION FALSIFIED: I registered "absolute PFs will drop"; every arm rose sharply
(IC 1.31->1.85). Mechanism: the leaked artifact was the DIRECTION ensemble, and its
future-trained calls were FILTERING OUT profitable neutral entries. The look-ahead model
made selection WORSE than no model — a result that only the fence could expose, and
independent corroboration of the ML-is-not-helping thesis (pending rerun B).
RULE A1 APPLIED: wide_wing beats the baseline on PF (1.91 > 1.85) with lower DD (8.68 <
12.31) at n=100 -> KEEPS its live slot. NO revert. Live config unchanged.
NEW, ON CLEAN EVIDENCE: the 08-04 wide-vs-broken ORDERING FLIPS. broken_wing now
DOMINATES wide_wing on both criteria (PF 2.13 vs 1.91; DD 5.52% vs 8.68%) and clears the
standard promotion bar (PF>1.2, DD<=IC arm, n>=30) by the widest margin of any arm ever
tested. It stays shadow ONLY because live has no asymmetric-wing builder
(strategies/iron_condor.py builds symmetric wings). PROMOTION BLOCKED ON CODE, NOT
EVIDENCE - user decision: build the asymmetric builder (put wing 2x call wing) or keep
wide-wing. Note the trade-off the numbers make explicit: IC has the highest total RETURN
(+9.91%) with the worst DD - narrower wings size more contracts per risk unit; PF/DD is
the pre-registered decision metric, not raw return.

## 2026-08-08 - RERUN B VERDICT: honest ML ablation REVERSES the 08-03 removal
First ablation where models ACTUALLY trained (72 window models + 79 range models trained;
only 9 insufficient_training_data + 2 no-accuracy, vs 96/96 FAILURES in the 08-03 run
that made that verdict vacuous). train_days=365, fence on, 25 windows:
  gate_only  (ML OFF) n=75 win 77.3% PF 1.04 DD 13.94% ret +0.43%  25 windows
  full_stack (ML ON)  n=20 win 70.0% PF 1.95 DD  1.72% ret +1.03%  13 windows
The gates VETO HARD: 75 -> 20 trades (73% rejected), and 12 of 25 windows produce no
trade at all. On PF and DD the gated arm wins by a mile (1.95 vs 1.04; 1.72% vs 13.94%).
This is the OPPOSITE of the 08-03 verdict ("gates veto nothing") - which we now know
tested nothing at all.
RULE B1 IS TECHNICALLY SATISFIED (PF +0.91 > 0.10, DD better, >=20 trained windows) =>
the rule says turn ML entry gates back ON. WITHHELD PENDING CONFIRMATION, disclosed:
B1 as I registered it OMITTED an n floor, and n=20 is below the n>=30 bar this project
applies to every other promotion. PF on 20 trades (14W/6L) cannot be separated from
noise, and flipping gates ON live would cut entries ~73%, gutting the 50-close sample
velocity that is the whole point of the live phase. Running ablation_confirm.py
(scripts/, ~11y history via days=4000) so the gated arm can clear n>=30 on
non-overlapping windows before any live change. If the confirm run holds (gated PF still
> ungated by >0.10, DD no worse, n>=30) -> set ml.entry_gates_enabled=True and record the
08-03 removal as REVERSED-ON-EVIDENCE; if it collapses toward parity -> gates stay OFF
and the 08-03 outcome stands for the right reason at last.
LESSON (process): a pre-registered rule with a missing floor is still a rule I must
report against honestly rather than quietly reinterpret - recorded here as a defect in
MY rule, not in the result.

## 2026-08-08 - RULE B1 APPLIED: ML ENTRY GATES BACK ON (08-03 removal REVERSED on evidence)
CONFIRM RUN (scripts/ablation_confirm.py, ~11y, 68 windows, 242 window models genuinely
trained - 14 insufficient + 9 no-accuracy only):
  gate_only  (ML OFF) n=274 win 77.4% PF 1.05 DD 34.57% ret +2.47%  68 windows
  full_stack (ML ON)  n= 92 win 64.1% PF 1.27 DD  9.97% ret +3.05%  40 windows
B1 SATISFIED on a proper sample (PF +0.22 > 0.10; DD 9.97 vs 34.57; n=92 >= 30; 40
trading windows >= 20). The 08-08 short-run PF gap (1.95 vs 1.04) narrowed to 1.27 vs
1.05 with 3.7x the trades - the small sample WAS inflated, exactly as the missing-n-floor
disclosure warned - but the direction and the DD result both held. APPLIED:
MLConfig.entry_gates_enabled True (settings.py), test pin flipped
(TestMlGatesRestored). Live effect at next boot.
THE REAL HEADLINE IS THE DRAWDOWN, NOT THE PF. Over 11 years spanning 2018/2020/2022 the
UNGATED condor - our live config until this commit - draws down 34.57%. On the planned
$3,000 live account that is a ~$1,040 hole; the gated arm's 9.97% is ~$300. Every prior
DD number we ever quoted (5-18%) came from 2-4 year windows that CONTAINED NO REAL VOL
EVENT. This is the first honest look at the strategy through a full cycle and it is the
most important number produced so far.
COST ACCEPTED + NEW PROBLEM: gates reject ~66% of candidates (274 -> 92 over 11y ~ 8
trades/yr across 3 symbols). The 50-close sample at that rate takes years. VELOCITY IS
NOW THE BINDING CONSTRAINT ON THE WHOLE PROGRAM - and note the ungated alternative is not
a fix, it is PF 1.05 (breakeven) with a 34.6% hole. Next studies must attack velocity
WITHOUT reopening that drawdown: (a) ml.range_min_confidence=0.65 is the binding live
floor and has NEVER been studied (current live QQQ p_in_range 0.587 sits just under it) -
sweep it; (b) universe breadth (the R2 uncorrelated-universe rejection was measured
UNGATED - retest gated); (c) DTE/entry-cadence.
COUPLING HANDLED: the 08-08 wing verdict (rerun A) was measured GATES-OFF and therefore
no longer matches live. scripts/shadow_round3_gated.py launched (same 3 arms, gates ON,
365-day train, 11y) - the wide-vs-broken wing decision is DEFERRED until it lands.

## 2026-08-10 - WING VERDICT UNDER GATES-ON: wide_wing CONFIRMED, broken_wing REJECTED
scripts/shadow_round3_gated.py (same 3 arms, gates ON = the regime live actually runs,
365-day train, fence on, 11y, 40 windows):
  iron_condor        n=92 win 64.1% PF 1.27 DD 9.97% ret +3.05%
  wide_wing_condor   n=95 win 71.6% PF 1.62 DD 4.91% ret +3.36%   <-- LIVE, CONFIRMED
  broken_wing_condor n=94 win 64.9% PF 1.27 DD 4.41% ret +1.44%
DEFERRED DECISION RESOLVED - NO LIVE CHANGE, NO ASYMMETRIC BUILDER NEEDED. wide_wing
wins PF (1.62 vs 1.27/1.27) and return, at a DD statistically tied with broken_wing
(4.91 vs 4.41) and half the baseline's. Broken_wing's apparent 08-08 dominance (PF 2.13)
was an ARTIFACT OF THE UNGATED REGIME: with gates on it collapses to 1.27, identical to
the plain condor. Deferring the builder was the correct call - building it on Saturday's
numbers would have shipped a DOWNGRADE.
METHOD NOTE (third reversal in a week, all from the same root): arm rankings are NOT
portable across regimes. wing_k, ML gates and wing SHAPE interact; every future arm study
must run under the live gate regime or its ranking means nothing. Registered as standing
process for all future shadow rounds.
LIVE (2026-08-10, box booted onto R16 + gates-ON): bot entered QQQ IC T-20260810-095326
@5.44 at 09:53 and SPY signals are generating at ratio 0.11 with p_in_range 0.959 - the
0.65 range floor is NOT freezing entries at current model outputs (the 08-08 velocity
worry was over-stated for the current regime; keep monitoring).

## 2026-08-10 - R16 ROUND 2: remaining 44 findings FIXED (suite 1014 green, referee 0 BREAKS)
Closed the medium/low tail across executor/broker, risk/books/state, engine/config/ops and
the orchestrator. Headlines:
 * EXIT QUOTE PATH WAS DEAD CODE: _close_multi_leg gated its NBBO fetch on
   qualify_contract(combo), which ALWAYS returns None for a BAG - so every multi-leg exit
   ever placed skipped live pricing and fell to the wing-width emergency branch (+ a
   CRITICAL page) on routine take-profits. Now quotes the raw Bag, polls up to 2.5s.
 * PRE-EVENT BLACKOUT FAILED OPEN behind a blanket except-pass, and counted CALENDAR days
   (a Friday entry with a Monday event saw "3 days" with ZERO sessions between). Now
   fails CLOSED and counts trading sessions.
 * CIRCUIT BREAKER now persists across restarts - the keeper's 90s relaunch used to reset
   the consecutive-loss count, erasing the protection during the crash-loop most likely
   to accompany a losing streak.
 * AGGREGATE RISK CAP could approve a new 3k-risk condor with "all checks passed" while
   two live condors reported max_loss 0 (missing KV). Structural fallback added; orphan
   KVs now swept post-market.
 * SLIPPAGE GATE was reading 65.1% because combo pricing context was written onto every
   LEG row and the BAG price kept the broker's sign. Writer fixed +
   scripts/restate_r16_executions.py applied (15 leg rows cleared, 4 BAG prices
   normalized) -> gate now reads 5.5%.
 * REFEREE FALSE POSITIVE FIXED: duplicate_closes counted any >=2-leg fill group, so an
   ENTRY group + its EXIT group read as a phantom re-close. It only surfaced now because
   the 08-04 QQQ condor is the FIRST trade whose entry legs were also swept into the
   ledger. Now requires the `closing` flag it already computed. Referee: 0 BREAKS.
 * META-LABELER BLOCKED, NOW EXPLICIT: trade_context.entry_signals is "{}" in the fixture
   AND in production, so 11 of the 20 META_FEATURES have NEVER been captured - the
   meta-labeler could only ever have trained on 9, which is exactly the degraded input
   that got meta_label.pkl quarantined 07-18. Coverage guard refuses to arm; the old test
   asserted the broken behavior and has been inverted (+ a positive control).
   PREREQUISITE for the "meta-labeler at 50 closes" plan: populate entry_signals first.
 * Also: single CREDIT_STRATEGIES authority (engine imported base's, was a 4-vs-11 fork);
   capital base single authority reading live NLV (was 196000 hardcoded in 3 files, ~65x
   wrong at the planned $3k scale); MTM brake failure now pages instead of DEBUG;
   notification tasks strongly referenced (an unreferenced create_task can be GC'd
   mid-flight - the shutdown page was the likeliest casualty); yfinance raw output no
   longer corrupts the JSON log stream; jade_lizard classified undefined-risk + flattened
   strangle-class; log rotation for keeper/dashboard; liquidity + parity manifests
   single-sourced. DELTA-BREACH RULE 3b confirmed STRUCTURALLY INERT (ib_insync
   PortfolioItem carries no greeks at all) - left inert but now LOUD once per session
   rather than silently dead; making it real needs a market-data decision after U6.
Tests +191 across 5 new files (r16_broker/risk/tail/round2 + core). Suite 1014 green.

## 2026-08-11 - RANGE-FLOOR STUDY (PRE-REGISTERED before results)
WHY: with ML gates back ON (rule B1), ml.range_min_confidence is now THE binding entry
constraint on every condor, and it has NEVER been studied - it was set to 0.65 once and
inherited. Velocity is the program's limiting factor (~8 trades/yr gated), and this is
the one knob that can buy trades back WITHOUT reopening the 34.6% ungated drawdown.
PARITY GAP THIS ALSO CLOSES: WalkForwardConfig.range_min_confidence defaults to 0.55
while live runs 0.65 - so the 08-08 confirm run measured a LOOSER gate than production
applies, and its n=92 OVERSTATES live velocity. Every arm here runs gates ON so the only
variable is the floor.
ARMS: 0.50 / 0.55 / 0.60 / 0.65(live) / 0.70, ~11y, 365-day train, fence on.
DECISION RULE (registered now): pick the LOWEST floor whose PF >= 0.95x the 0.65 arm's
PF AND whose max DD <= the 0.65 arm's DD, requiring n >= 30. That maximizes sample
velocity subject to not degrading quality. If NO arm beats 0.65 on trade count without
breaching those bounds, the floor STAYS 0.65 and velocity must be attacked elsewhere
(universe breadth, DTE cadence) - recorded as such rather than loosened by preference.
If the 0.50/0.55 arms show materially MORE trades at equal quality, that also means the
08-08 confirm run's absolute numbers were measured at a floor we do not run - annotate
that verdict accordingly.

## 2026-08-11 - TOUCH-STOP MIRROR ADDED TO THE ENGINE (methodology fix; PRE-REGISTERED study)
FINDING (user question, verified in code): every backtest this project has ever run steps
ONE BAR PER DAY and evaluated exits against the daily CLOSE. Live's PRIMARY loss exit is
the short-strike TOUCH, checked every 30s against spot. The engine never modelled it -
its own comment called this "a documented structural divergence". So an intraday pierce
that recovered by the bell was scored as an untouched winner, and EVERY verdict to date
(wing_k, shadow R1-R3, both ablations, the range-floor sweep now running) measured a
DIFFERENT EXIT POLICY than production runs - biased optimistic on exactly the days that
hurt. This is the most likely single explanation for sim win rates of 77-84% against a
live record of 7W/14.
FIX (engine.py _check_exit_credit rule 0): daily High/Low BRACKET the true intraday path,
so the touch is detected EXACTLY (no intraday bars needed) - if price reached 750 at any
point, Low <= 750. The exit is repriced AT the touched strike (where live transacts), not
at the close. Ordered BEFORE take-profit: live's 30s monitor sees the pierce first.
AIT_BT_TOUCH_STOP=1 default (live parity); =0 restores legacy close-only for A/B only.
Test-pinned (tests/test_r16_touch_stop.py, 10 tests incl. the recovered-pierce case).
INTRADAY DATA (asked): Yahoo caps 5m at 60 days (4,606 bars) and 1h at 730 days; only
daily reaches 11y. Local 5m store covers 2026-06-01+ only. So deep-history intraday is
NOT available from the current sources - but it is also NOT needed for touch DETECTION,
which High/Low gives exactly. Intraday would add fill-level realism and same-day
put-vs-call touch SEQUENCING (rare); IBKR reqHistoricalData could backfill 1-2y of 5m
with pacing if we later want that validation slice.
STUDY PRE-REGISTERED (scripts/touch_stop_impact.py): identical arm (k=1.6, floor 0.10,
gates ON, 11y), touch OFF vs ON. RULE: report the delta honestly whatever it shows; if
PF/DD degrade materially under the touch stop, EVERY prior verdict is restated as
measured-under-the-wrong-policy and the wing/gate decisions are RE-DERIVED from
touch-on numbers before any further live change. Queued behind the range-floor sweep
(CPU); NOTE the floor sweep is running the OLD close-only engine and its absolutes will
need the same treatment - its arm-vs-arm ORDERING is still valid (consistent policy).

## 2026-08-11 - SIM FIDELITY: three measurement corrections landed (suite 1025 green)
Prompted by the user's question "are we only testing against closing values?". All three
were REAL biases in every study this project has ever run, and they do NOT point the same
way - which is why no verdict should be re-derived until a study runs with all three.
 1. TOUCH STOP (methodology, biggest): engine now mirrors live's short-strike exit via
    daily High/Low (they bracket the intraday path, so detection is exact without
    intraday bars); exit repriced AT the touched strike. MEASURED IMPACT (identical arm,
    k=1.6/0.10, gates on, 11y, 34 windows):
        touch OFF  n=82 win 68.3% PF 1.38 DD 5.38% ret 1.86%
        touch ON   n=87 win 60.9% PF 1.30 DD 3.88% ret 1.45%
    -> sim was OPTIMISTIC by 7.4pp of win rate and 0.08 PF, but PESSIMISTIC on drawdown
    (-28%). Explains a large part of the 77-84% sim win rate vs 8W/15 live. Prior
    arm-vs-arm verdicts stand (the policy applies uniformly to all arms); the ABSOLUTE
    win rates we have quoted are restated ~7pp lower.
 2. PER-SYMBOL IMPLIED VOL: the engine applied VIX x1.10 to every underlying. Measured
    VXN/VIX (NDX vs SPX, implied) median 1.228 over 2,918 days; realized ratios 10y are
    QQQ/SPY 1.354, IWM/SPY 1.465. Now SPY x1.00, QQQ x1.228, IWM x1.33 (RVX is not
    served by yfinance, so IWM uses its realized ratio shrunk by QQQ's implied/realized
    factor). QQQ - our most-traded symbol - had its IV understated ~12% at the median and
    ~34% on 2026-08-11 (VIX 15.6 vs VXN 22.9), biasing credits low and wings narrow.
    Also unified the scalar-vs-DataFrame VIX paths (they used 1.05 vs 1.10 for the same
    reading - an inconsistency the old test had pinned).
 3. EMPIRICAL SPREADS: option_spread_samples/params existed, had NEVER been populated,
    and (once populated) were read by NOTHING. calibrate_option_spreads.py crashed on a
    cp1252 arrow char AFTER writing SPY but BEFORE QQQ/IWM - which is why it looked like
    it had never worked. Fixed + stdout hardened; captured 3,159 real quotes (7-45 DTE).
    Fitted base spread SPY 0.0123 / QQQ 0.0027 / IWM 0.0197 vs config 0.04 - we were
    charging 2-13x the real friction (a PESSIMISTIC bias). Engine now loads calibrated
    per-symbol params (AIT_BT_CALIBRATED_SPREADS=0 to A/B).
FREE OPTIONS DATA (asked): DoltHub post-no-preference/options is genuinely free with real
bid/ask/IV/greeks and is updated daily, BUT SPY has only ~4 expirations / 33 strikes per
date and QQQ/IWM are ABSENT - unusable as a backtest source. Alpha Vantage historical
options needs a (free) user key; CBOE/ORATS/OptionMetrics are paid. Conclusion: keep
building OUR OWN dataset via the now-working calibration capture.
NEXT: re-parameterized range-floor sweep is the FIRST study with all three corrections.

## 2026-08-12 - R17: independent external code review, 20 findings FIXED (branch r17-review-fixes)
WHY: a fresh Claude session (no access to this file or prior audit context) reviewed the
whole feature/event-straddle diff against main cold, as a code review rather than an
audit round. Findings recorded in docs/PR6_CODE_REVIEW.md. All 20 re-verified against
the tip TWICE as this branch kept moving mid-review/mid-fix (ee6abc8 -> b593583 via R16
ROUND 2, then b593583 -> a90e8d7 via SIM FIDELITY above, rebased onto cleanly - neither
touched the same lines except this file) - none were already fixed, none conflict with
that work. 7 commits, one per theme, each independently revertable. Tests +50 across 5
new files (r17_live/risk_data/ops/dormant_strategy/minor_fixes), suite green.
HEADLINES (live-affecting now - only iron_condor is enabled in config.yaml):
 * EARNINGS PRE-CLOSE EXIT WAS UNREACHABLE: rule 3b (delta breach) claimed the exit
   elif-chain slot on strategy-membership alone (no delta check gated ENTRY to the
   branch, only whether it acted once inside) - so once R16 ROUND 2 confirmed 3b is
   structurally inert (no greeks subscription), it still permanently blocked rule 3c
   (the 2-day pre-earnings close) for every iron_condor/short_strangle/etc position
   past DTE 5. Same bug class R13-CRIT-2 fixed for rule 2 vs the touch stop, missed
   here. Now 3b only claims the slot on a real breach (walrus-guarded condition).
 * ML AUTO-ROLLBACK WAS DEAD CODE: the "mean accuracy across all symbols trained"
   flattening always produced an empty list (results[symbol] is a dict of per-model
   scores, never a scalar), so cv_scores was never updated post-training and a
   materially worse retrain could never trigger rollback. Zero prior test coverage.
 * DASHBOARD P&L SILENTLY DROPPED NULL-LABELED TRADES: get_status() (CLI + web
   dashboard) was missing the COALESCE its own main() sibling query already had.
 * CALENDAR-EXHAUSTION ALARM NEVER REACHED A HUMAN: log.critical with no notify
   route - the exact "macro guards went blind, nobody noticed" failure R16 #10 above
   was built to catch, reproduced by its own follow-up alarm.
 * .ENV COULD NOT OVERRIDE PROTECTIVE DEFAULTS: apply_runtime_env_defaults() loaded
   .env AFTER the 7 setdefault() calls, so .env - the only documented user config
   path - could never touch them, including AIT_MARKET_DATA_TYPE (the delayed-vs-live
   switch the code comments say is meant to flip "after funding"). Reordered.
 * ORDERREF ADDED FOR EXIT DISAMBIGUATION: reconciler's combo-order matching fell
   back to symbol-only when leg-keys didn't disambiguate - a real misassignment risk
   between two same-symbol multi-leg positions both mid-close after a restart.
   orderRef (unused anywhere in this repo before now) is stamped with trade_id at
   every order-placement site; reconciler tries an exact match first.
 * Also (live but lower-severity): capital_base() go-live enforcement (was silent if
   it fell back to the hardcoded default in live mode); earnings.py used
   date.today() instead of the ET convention; a backfill script's MIDPOINT-vs-TRADES
   "fix" only relabeled, didn't prevent the overwrite; gate 6c (symbol concentration)
   summed credit collected instead of real max-loss; account-data staleness never
   escalated past a log line; winter EDT-hardcode false-failed the post-deploy
   liveness check every EST season; smoke_deploy's no-write guard had zero tolerance
   for the live bot's own concurrent writes; walk-forward window capital divided by
   configured symbol count instead of the count that actually survived the learner's
   gates; a NaN-fill safety net contradicted vix_level's own documented neutral
   value; Yahoo-fallback options data got IBKR's "OI=0 means unknown" leniency where
   a Yahoo zero is real.
DORMANT (only matters if short_strangle/long_straddle are re-enabled - fixed now per
this repo's own convention of not leaving known bugs dormant): capital-at-risk
accounting returned $0 for undefined-risk strategies with a real max_loss; the
long_straddle vol-magnitude ML override had no entry_gates_enabled check (unlike the
adjacent range-model override); long_straddle was missing from the direction-
inference-avoidance exclusion list.
HYGIENE: settings.py had a stray UTF-8 BOM + 23 mojibake sequences (cp1252/UTF-8
round-trip corruption) from a prior edit, repaired byte-exact, encoding-only commit.
counterfactual.py's skip-dedup silently froze after the first occurrence per combo
forever (the reset flag's setter was deleted in R12-C) - now resets daily. main.py
hard-cancelled orchestrator.run() on shutdown, bypassing R16 ROUND 2's own
notify-drain fix entirely - now stops gracefully with a bounded grace window first.
Two items documented rather than changed (no live bug, confirmed by grep): the
risk_budget shared-instance mutation (inert while scanning is sequential) and the
Sortino inf-vs-None inconsistency (PerformanceMetrics.sortino_ratio has no current
JSON consumer).
NOT IN SCOPE: a multi-contract debit-combo pricing bug found in passing
(executor.py's debit-cap divides by contracts twice) - only affects debit strategies
(bull_call_spread/bear_put_spread/calendar_spread/long_straddle), all currently
disabled; wasn't in the original 20 findings, left untouched to avoid scope creep.
PRE-EXISTING, UNRELATED FAILURES noted before rebase (not touched by this branch):
tests/test_iv_model.py::TestGetIV::test_vix_proxy_scalar_produces_plausible_iv (fails
identically on b593583; fixed incidentally by SIM FIDELITY's VIX-path unification
above, confirmed post-rebase) and tests/test_r16_ops_fixes.py::TestSingletonMutex (3
tests; the singleton-lock mechanism imports msvcrt, Windows-only - fails on macOS dev
boxes, unrelated to platform the bot actually runs on).
Rebased onto a90e8d7 and pushed directly to feature/event-straddle (user-approved) -
matches this branch's own convention (R12-R16 above all landed as direct commits to
an already-open PR #6, not stacked sub-PRs).

## 2026-08-14 - R18: the 08-11 outage was one bug hiding a CLASS. Class closed.
INCIDENT: `self._settings.trading.strategies` (strategies lives on OptionsConfig)
raised AttributeError on EVERY trading cycle from the 08-11 restart. 595 errors, 3
trading days with ZERO entry scans. Hotfix 7594e7f.
CORRECTION TO MY OWN FIRST REPORT: I said it "paged nobody". WRONG - the message text is
never logged (only a char count), so grepping for the alert text could never find it. The
LOOP IMPAIRED alert for this bug is EXACTLY 153 chars and the logs show 24 sends of 153
chars during the window. `_note_loop_error` worked as designed and paged Telegram 24
times. DETECTION WAS NOT THE GAP - the alerts were delivered and not acted on. Open
question for the operator: are Telegram pages being seen at all? Building more alerting
would be pointless if 24 went unnoticed; alert ROUTING/wording is the real lever (this
one says "Position protection may be OFF", not "the bot has stopped trading").
WHY IT SHIPPED: its test asserted `inspect.getsource(...)` contained strings - it passed
while the code raised on every execution. 34 such assertions exist; NOTHING in tests/ ever
called _trading_cycle / _scan_symbol / _try_execute / _monitor_positions_fast / __init__.
Exits have 15+ real invocations - which is exactly why "exits were unaffected, so it
looked healthy".
AMPLIFIER (worse than first reported): _trading_loop reset time_since_scan ONLY on
success, so a raising cycle re-took that branch forever and _monitor_positions_fast NEVER
RAN AGAIN. The outage therefore also silently killed the MTM daily-loss brake, the 8-K
material-event check and the read-only re-probe. Reset now happens in a `finally`.
THREE MORE LATENT BUGS OF THE SAME CLASS (all R16-introduced, all on paths no test ran):
 * _try_execute fail-closed returned "handled" on ANY blackout-check exception. Since
   iron_condor is the ONLY enabled strategy and IS a credit structure, a persistent fault
   blocked 100% of entries INDEFINITELY behind one log line. Now escalates + pages.
 * _sync_risk_manager_positions: float() on a raw KV raised out of the whole loop, so
   update_positions() never ran and every risk cap validated against a STALE book.
 * _close_multi_leg/_option_nbbo: the R16 poll loop tolerates a None bid, the consumers
   called math.isnan(None) -> TypeError -> exit fell back to pricing at the FULL WING
   WIDTH + a CRITICAL page, reverting the R16 fix.
FOURTH BUG, found by the new smoke harness on its FIRST RUN: _persist_daily_iv called
`c.atm_iv()` but atm_iv is a dataclass FIELD (float). TypeError into its own except every
time -> the R16 "self-healing IV store" NEVER WROTE A ROW (store still ends 2026-07-09 for
all 11 symbols) -> the freshness gate kept finding it stale -> EVERY iv_rank since has
silently used the realized-vol proxy, not implied. So the R16 IV fix never worked, and we
only know because a test finally EXECUTED the path.
FIXES SHIPPED: all four bugs + amplifier; three tests that COULD NOT FAIL repaired (two
inspected method names that have never existed -> empty source; one contained
`True if x or True else False`); highest-risk source-string tests converted to behavioral
(17/17 mutation checks caught); NEW tests/test_hot_path_smoke.py builds a REAL orchestrator
through the REAL __init__ with REAL settings and awaits the whole hot path, with a log spy
that fails on any SWALLOWED exception (a returning function proves nothing when every hot
path catches its own errors); smoke_deploy gains check_trading_cycle_dryrun (import-walk +
load-settings both passed green all through the outage); mypy type gate
(scripts/typecheck.py) blocking on attr-defined/name-defined/call-arg, PROVEN to fail on
the shipped bug and pass on HEAD. Suite 1100 green.
NOTE: the CI type gate does NOT protect main until PR #6 merges (GitHub runs workflows
from the default branch).
STANDING LESSON: a source-string assertion proves code was WRITTEN, never that it RUNS -
and it decays into a silent no-op on rename (two already had). Same "execute, don't read"
rule the audits kept proving; I broke it in my own fixes.

## 2026-08-18 - R19 EXTREME AUDIT (user-directed: line-by-line + "any hardcoded config?")
SCOPE: all 38,358 production lines / 66 files, 15 auditors each attesting per-file line
coverage, 95 agents, adversarial 2-skeptic verification. 48 findings survived; 248
lower-tier logged. THE HARDCODED-CONFIG LENS WAS THE HIGHEST-YIELD QUESTION ASKED SO FAR.
FINDING OF THE ROUND — the env contract was shadowed by 17 private reader fallbacks, FOUR
disagreeing with the contract they mirrored:
  AIT_IC_WING_K           contract 1.6  -> readers 1.0  (x2)
  AIT_IC_MIN_CREDIT_WIDTH contract 0.10 -> readers 0.20 (x4)
  AIT_SKIP_MACRO_EVENTS   contract "1"  -> readers "0"  (x3, FAIL-OPEN)
  AIT_CREDIT_LOSS_LIMIT   absent        -> split 0 vs 1.25
Live was safe ONLY because both entry points call apply_runtime_env_defaults() first; any
path skipping it ran pre-promotion economics with macro protection OFF. This is the wing_k
four-sources incident recurring one layer down. FIX: runtime_env.CONTRACT_DEFAULTS (one
table, 9 keys) + contract_float/flag/str accessors; 17 reader sites migrated; the applier
now drives off the same table; AIT_CREDIT_LOSS_LIMIT joined the contract at 0 (R6 evidence).
Verified: a bare process with NO applier now resolves 1.6 / 0.10 / protection-ON.
tests/test_r19_config_authority.py ENFORCES it — scans src/ and fails if any module
reintroduces a private fallback (executor's INST-5 read is the one documented exemption).
LIVE-MONEY BUGS FIXED: (1) an entry that FILLED during a disconnect could be booked
CANCELLED — ib_insync wipes the trade cache on disconnect and recovered orders carry
orderId 0, so orderId-only matching missed it, orphaning 4 live legs in a status nothing
re-adopts; now permId + ib.fills() + ledger evidence, and a cancel verdict is refused when
executions exist. (2) cancel CAS included PARTIAL (could orphan filled legs). (3) BAG price
reconstructed on ORDERED not FILLED quantity. (4) ledger fill attribution had the same
reconnect blind spot. (5) two dead cancel paths (30s partial remainder, >300s stale exit)
that could never transmit. (6) NaN realized P&L could reach the trades DB. (7) the R16
sweep could cancel ANOTHER trade's order (ignored R17's orderRef tagging). (8) _spot_quote
false-'frozen' on the 2nd same-symbol read per pass, suppressing real exits.
BOTH OF THE EXTERNAL R17 HEADLINE FIXES WERE DEAD CODE (found independently by 2 auditors):
ML auto-rollback assigned to a READ-ONLY property (AttributeError swallowed every time, so
a degraded retrain could never roll back) AND never persisted (bad ensemble.pkl survived to
the next boot); the stale-account escalation recorded ONE api failure where the breaker
needs 5, so the documented halt could never fire. Both now real + tested. LESSON: the
reviewer who caught MY unexecuted fixes shipped unexecuted fixes. The disease is universal;
only execution-based tests catch it.
MY OWN MIGRATION INTRODUCED ONE: the import-insertion guard checked whether the MODULE was
imported, not the NAMES — orchestrator.py already imported capital_base from runtime_env,
so contract_flag was never added -> NameError on every trading cycle. Caught by
tests/test_hot_path_smoke.py (R18) before it ever reached the bot. That harness has now
paid for itself twice.
TESTS PINNING THE OLD DIVERGENCE, corrected: engine wing_k default asserted 1.0 (the bug);
the R17 account-stale test used a bare MagicMock whose is_tripped is TRUTHY, so the
escalation short-circuited and the assertion passed against a state the real breaker can
never be in at outage start — replaced with a _FakeBreaker modelling real semantics.
STATE: suite 1,166 green; type gate, referee (0 BREAKS) and deploy smoke all pass.
DEFERRED (recorded, not fixed): the research-validity cluster — engine never forwards
market_context to _build_position (so VIX-proxy IV, incl. the R18 per-symbol VXN
calibration, may never have engaged), Optuna-searched params never reach trial Backtesters,
train_window_models=False still trains the meta-labeler, entry-window 09:30-vs-10:30 fork.
Arm ORDERINGS likely survive (shared defects); absolutes suspect. Re-run studies only after
this cluster is fixed. Also deferred: ~125 lower-tier hardcoded-config items (the register
is the spec), executor settings= wiring at orchestrator.py:149 (dormant: IC-only).

## 2026-08-18 - R19b: CONFIG NOW OUTRANKS CODE DEFAULTS (user question, answered in code)
User asked: "the config must have priority over the default, right?" It did NOT, twice over:
 (1) SECOND SHADOWING LAYER: pydantic Field defaults in settings.py diverged from
     config.yaml on 5+ knobs (wing_k 1.0-vs-1.6 the worst). Every module constructing a
     config model BARE (engine, walkforward, optimizer, trainer) ran code defaults, not the
     operator's yaml. FIX: wing_k default aligned to the live 1.6 (a divergent wing size is
     a DIFFERENT STRATEGY, not a safer fallback); the risk-side knobs stay deliberately
     stricter in code but load_settings() now emits config_default_divergence listing every
     yaml override (12 today) so the skew is never silent; default_divergences() is
     test-pinned.
 (2) PRECEDENCE HOLE: contract resolution went env -> hardcoded default, SKIPPING
     config.yaml - editing backtest.wing_k changed nothing on the live/default path. FIX:
     CONFIG_BACKED maps contract keys to their yaml homes; precedence is now explicit env >
     config.yaml > CONTRACT_DEFAULTS, enforced at the reader AND in the applier (which
     seeds env for live processes and would otherwise mask config forever). Proven with a
     5-scenario execution matrix + 4 pinned tests.
SECURITY CATCH DURING THE WORK: my first divergence report printed api_keys/ibkr VALUES
(Finnhub, Polygon, Telegram token, account number) - the exact R13 #12 leak class, caught
before it ever ran in the bot. Secrets are now excluded by section + field-name heuristic,
with a test asserting no secret can enter the report. NOTE: those keys passed through this
chat session - U10 (rotate Finnhub) now applies to ALL of them; rotate at the providers.
Suite 1,173 green; smoke + type gates pass.

## 2026-08-18 - R19c: CONFIG.YAML IS NOW THE OPERATING SOURCE FOR TRADING VALUES (user policy)
User decision: "config must be the ONLY place with values that manage trading." Executed:
every trading-economics contract key now has a config.yaml home visible to the operator -
backtest.ic_min_credit_width (0.10), backtest.ic_min_credit (0.70),
backtest.credit_loss_limit (0 = disabled per R6), risk.skip_macro_events (true) - joining
backtest.wing_k (1.6). Resolution everywhere: explicit env override > config.yaml >
CONTRACT_DEFAULTS (safety net only). Yaml booleans normalize to the contract's '1'/'0'.
Values unchanged - this is about WHERE they live, not WHAT they are (test-pinned).
DOCUMENTED EXCEPTIONS (each with a stated reason, enforced by test): KMP/OMP crash guards
(process-level, must precede imports); AIT_MARKET_DATA_TYPE (broker data entitlement -
deployment concern, flips at U6); AIT_ALLOW_UNDEFINED_RISK (INST-5 interlock, env-only ON
PURPOSE so re-enabling naked risk is a deliberate act, not a config edit).
ENFORCEMENT: test_every_trading_key_has_a_config_home fails if a future contract key lacks
a config home; test_config_homes_exist_in_yaml_and_settings fails if yaml and settings
drift; the R19 no-private-fallback scan still guards the reader side. Suite 1,177 green.
REMAINING CONFIG DEBT: ~125 lower-tier literals from the R19 register (VIX credit-cap
tiers, TP ladder copies in the engine, entry-window forks, param_spaces) - migrate
opportunistically using these same homes; the register is the spec.

## 2026-08-20 - R19d: VERDICT METRIC IS IRON-CONDOR-ONLY (user decision)
The go-live gates now score IC closes exclusively (status.py; all-strategy line kept
underneath for book honesty). Rationale: the mission is the IC edge question; the retired
experiments were drowning the signal - the mixed record read -$451 at its worst and
9W-7L/+$150.83 today, while the CONDOR record is 7 closes, 4W-3L, +$452.89, PF 3.91.
Sharper still: ALL 3 condor losses came from the since-abolished 07-13 macro-event flatten
and the lone early scratch-win from the since-removed trailing stop - under the CURRENT
ruleset (hold-through, wide wings, touch-stop) the condor is 3-for-3, +$602.77. Those 3
closes are the first entries of the verdict sample that actually test today's policy.
Gate math unchanged (50 closes, PF>1.3); the clock now counts what the mission asks.

## 2026-08-20 - R20: RESEARCH APPARATUS FIXED + ENTRY_SIGNALS CAPTURE + REGISTER TIER-2
THE HEADLINE, PROVEN BY A/B EXECUTION: the engine NEVER forwarded market_context into
position building - a run with VIX 30 supplied and a run without were IDENTICAL (10
trades each, entry IV = realized-vol x 1.15 synthetic). EVERY study this project ever ran
- wing_k, shadow R1-R3, both ablations, floor sweep, touch-stop impact - priced its
options WITHOUT VIX and WITHOUT the R18 per-symbol calibration. Post-fix: VIX 30 ->
QQQ prices at 0.3684 = 0.30 x 1.228 (per-symbol multiplier finally engaged); SPY control
at 0.30 (x1.00). Arm ORDERINGS remain the only trustworthy prior outputs (shared defect);
every ABSOLUTE number is restated as measured-on-synthetic-vol. Next study is the first
with an honest apparatus end to end.
ALSO FIXED (agent, all proven fail-pre-fix, 30 new tests): Optuna searched 2 params that
NEVER reached the trial engine (pure noise dimensions - the "best" values were arbitrary
and walkforward applied them OOS anyway); train_window_models=False still trained AND
applied the window meta-labeler (ablation contamination); phantom intraday knobs now
actually forwarded; NEW shared authority ait/execution/exit_policy.py - the engine
consumed a HAND-COPY of live's TP ladder/DTE-5/macro windows, now imports the shared
module + a test executes live portfolio._get_take_profit_targets vs the policy so any
future de-sync fails a test; entry-window fork resolved to config's 10:30 (engine/
walkforward drift said 09:30); range/vol-mag training floor literal 100 ->
ml.min_training_samples (mine - constructors take min_training_samples=None -> config).
ENTRY_SIGNALS CAPTURE (mine, live path, gated x3): _scan_symbol stashes the entry-time
feature row; _try_execute persists the 11 technical META_FEATURES + hour_of_day into
trade_context.entry_signals - the column that was "{}" on EVERY trade ever taken, which
is why the meta-labeler could never train (9/20 features). From the next entry, every
close builds the training set. Degrades to "{}" safely; never blocks an entry.
REGISTER TIER-2 (mine, pure relocations, semantics test-pinned identical): VIX-tiered
credit caps -> risk.credit_cap_vix_tiers ([[20,6],[25,4],[999,2]]); symbol concentration
0.20 -> risk.max_symbol_concentration_pct. Three test fixtures completed (mocked configs
missing the new fields - the _FakeBreaker lesson again).
Suite 1,210 green; type gate, smoke (13 checks), referee (0 BREAKS) all pass. Agent
flagged for a follow-up: walkforward wing_k=1.0 dataclass default (pre-registered-
comparison sensitive), optimizer baseline literals (stop 0.35, hurst, mf, max_hold 30).

## 2026-08-21 - R20b PRE-REGISTRATION: research defaults migrate to config resolution
User: "can't we fix them?" — yes. The deferral reason (old-study comparability) is void:
R20 proved every prior absolute was priced without volatility data, so protecting their
reproduction protects disavowed numbers. REGISTERED CHANGE, before implementation:
 1. WalkForwardConfig.wing_k 1.0 -> None => resolve from the contract (env>config>1.6);
    same for ic_min_credit_width and range_min_confidence (-> ml.range_min_confidence,
    closing the 0.55-vs-0.65 parity gap the floor sweep exposed).
 2. Engine constructor defaults that shadow config with DIFFERENT values (iv_floor 0.12
    vs 0.20, range_min_confidence 0.55 vs 0.65, min_confidence 0.55) -> None => resolve
    from load_settings(); initial_capital/max_concurrent stay explicit (test-harness
    knobs, documented).
 3. Optimizer baselines: hurst_regime_threshold/penalty read the EXISTING BacktestConfig
    fields; stop_loss_pct (0.35), profit_target_pct, max_hold_days (30),
    multifractal_width_threshold (0.50), iv_rank_rise_threshold (0.30),
    min_edge_over_baseline (0.05) get NEW BacktestConfig fields with today's operating
    values as defaults + config.yaml entries.
 4. Tier-4 reclassification: log strings/report formats are NOT config and will not be
    migrated; genuinely config-ish tier-4 items (dashboard ports, DB paths) stay on the
    register for an ops round.
CONSEQUENCE ACCEPTED: re-running an OLD study script without explicit params now measures
CURRENT config, not 2026-07 defaults. That is the point. Any historical reproduction must
pass explicit values (the scripts we care about already do).

## 2026-08-21 - R20b EXECUTED: research defaults now resolve from config (registration above)
All four registered items landed (agent, 14 new tests, 9/14 proven fail-pre-fix):
WalkForwardConfig wing_k/ic_min_credit_width/range_min_confidence -> None => contract/
config resolution (bare config now resolves 1.6/0.10/0.65 — the 0.55 floor-sweep parity
gap is closed at the source); engine iv_floor/range_min_confidence/min_confidence -> None
=> load_settings() (min_confidence homed to risk.min_confidence — it IS live's directional
gate, documented); optimizer trial baselines read config (5 NEW BacktestConfig fields:
stop_loss_pct 0.35, profit_target_pct 0.50, max_hold_days 30, iv_rank_rise_threshold
0.30, min_edge_over_baseline 0.05 + existing fractal fields wired; multifractal deviation:
reused existing multifractal_max_width rather than forking a duplicate). Two stale test
pins retired (they asserted the 1.0 literal this migration removes). Divergence report
unchanged (12 pre-existing entries; new fields drop out naturally, test-pinned).
Also closed this session: live TradeExecutor now receives settings (spread gate config-
bound). Suite 1,224 green; type gate, smoke, referee all pass.
RESIDUALS (named, bounded): run_backtest.py CLI argparse defaults re-freeze old values
(stale env fallbacks 1.0/0.20/0.55) — ops-round item; bare StrategyOptimizer() without
walkforward threading still defaults wing_k 1.0 (walk-forward path correct); dashboard
export display literals (cosmetic). Tier-4 log/report strings RECLASSIFIED as not-config.

## 2026-08-25 - R21: POST-STOP COOLDOWN NEVER MATCHED THE TOUCH STOP (found via live book)
DEFECT (live, executed proof): R12-B4's re-entry cooldown greps
exit_reason_detailed LIKE '%stop_loss%', but the short-strike touch stop —
live's PRIMARY loss exit since R12-B1 — writes 'short_strike_touch (spot ...)'.
08-24: QQQ touch-stopped 09:47:33 (−$272.29), bot re-entered QQQ 10:00:40 —
13 minutes later, straight back into the move the rule exists to avoid. The
rule silently never applied to the exit type it most needed to cover.
FIX: query extracted to TradingOrchestrator._post_stop_cooldown_until (now
testable), pattern broadened to stop_loss OR short_strike_touch. trailing_stop
/breakeven_stop stay excluded on purpose (fire at/above breakeven — not the
autocorrelated-loss sequence). Stale mirror comment in portfolio.py updated.
PROOF: old query 0 matches vs new 1 on the live reason string (executed);
tests/test_r21_post_stop_cooldown.py — 7 tests EXECUTE the real query against
a real sqlite file (touch blocks, confirmed-tick variant blocks, stop_loss
blocks, profit exits don't, 30h expiry, per-symbol scope, open rows ignored).
NOTE: the 08-24 re-entry itself stands (T-20260824-100040, $6.35 credit) — we
do not undo live positions retroactively; it is a normal position now.
Same day: merged origin/main (Ahmed's PR #6 merge = CI type gate + nightly
lane armed on main; his 2 IV commits add a PARALLEL daily_iv/intraday_iv OHLC
store — our daily_prices.implied_vol path unchanged; reconcile pending) and
pulled curious-talleb's review round (engine entry_dte decoupled from Optuna's
max_hold_days cap, UTC-midnight flaky-test fixes, vix-tier fallback, iv_floor
dup, GO_LIVE_VERDICT_STRATEGIES frozenset in base.py).

## 2026-08-25 - R21b: PR#7 REVIEW ROUND (Copilot, 4 findings) — ALL VERIFIED REAL, FIXED
All four are fallout from the 08-24 entry_dte/max_hold_days decoupling:
[1] param_spaces: 7 non-IC spaces searched max_hold_days from 14 up — debit
    expires at entry_dte (14), credit closes at DTE<=5 (~day 9), so the
    dimension was flat Optuna noise. Bounded: debit (5,14), credit (1,9).
[2] optimizer: stop_loss_pct/profit_target_pct/min_confidence had no ctor
    overrides — walkforward's explicit values reached the OOS Backtester but
    NOT trial baselines (train/OOS parity break). Threaded through, explicit
    wins, None still resolves from config.
[3] walkforward: range model trained + OOS-evaluated at horizon
    max_hold_days (30) while reachable IC holds cap at ~9 days — 30-day
    containment labels gating 9-day trades. New _range_label_horizon() =
    min(max_hold_days, dte_range[0] - EXPIRY_APPROACHING_DTE); all 3 range
    call sites switched (direction-model eval untouched — separate question).
[4] run_backtest parity manifest hardcoded dte_band [21,21] ("engine uses
    fixed DTE = max_hold_days" — retired convention) — now reports
    [dte_range[0]]x2, matched against the actual engine in a test.
PROOF: tests/test_r21b_pr7_review_round.py — 8 tests execute the real
resolution paths (space reachability vs CREDIT_STRATEGIES, ctor override
precedence + config defaults, horizon derivation + cap + degraded-settings
fallback, manifest == live Backtester._entry_dte).

## 2026-08-25 - R21c: PRE-MERGE ULTRA-VALIDATION OF PR#7 (12-agent adversarial pass)
Validated the three no-blocker merge claims. RESULTS: claim 2 (no Ahmed
conflict) CONFIRMED (merge-base = origin/main tip; his commits purely
additive; zero overlap, no schema collisions). Claims 1 and 3 FALSIFIED as
stated, both fixed same day:
[1] F33 half-fixed: manifest dte_band corrected earlier but the prose NOTE at
    run_backtest.py:207 still described the retired fixed-DTE=max_hold_days
    convention. Note rewritten; executed check: no 'max_hold_days=21' in notes.
[3] The ONE risk-loosening live change (orchestrator passing settings= into
    TradeExecutor, arming the executor's config spread ceiling 0.15->0.40 —
    deliberate R19 register change aligning executor with the scanner) had NO
    test on the WIRING hunk: reverting it kept the suite green. Pinned by
    test_hot_path_smoke.py::TestConstruction::test_executor_receives_settings_spread_ceiling.
Findings ledger: 33 distinct review findings across all rounds; 32 verified
fixed in-code (10 by named executing tests, 22 by ad-hoc real-code probes),
1 (F33) completed above. REGISTERED RESIDUALS from the critic: (a) 22 fixes
lack a permanent pinning pytest (regression-visible only via behavior);
(b) typecheck gate scope excludes backtesting/optimization/ml; (c) 5 config
homes are zero-delta today but create config-edit levers on live economics
(documented, intended — that IS the R19c policy); (d) meta_label.py changes
live-dormant behind meta_label.enabled=false.

## 2026-08-25 - R22: RELATIONSHIP-DEFECT HUNT (15-agent, 7 classes) — 46 STANDING, REGISTERED
Question: "any other defects like entry_dte-vs-horizon?" (facts in different
files that must agree, nothing enforcing it). 7 class-specialized finders +
adversarial verification (default-refute): 48 candidates -> 46 standing
(30 DIVERGENT TODAY, 16 enforcement gaps), 2 refuted. Full register with
proof/consequence/fix per finding: reports/relationship_defect_hunt_20260825.md.
Severity: live_money 8 | research_validity 23 | ops_reporting 11 | cosmetic 4.
TOP THREE (critic-concurred):
[1] units-scale-05: LIVE range gate asks P(+-5% over 30d)>=0.65 while
    post-R21b research validates a 9-day/adaptive-threshold question — live
    vetoes entries the validated evidence would take AND slows the 50-close
    verdict sample. (The mirror image of R21b: fixed research, didn't sweep
    the live consumer.)
[2] numeric-pairs-02: R21's cooldown kept the flat 30h window vs the R12-B4
    spec's 1 TRADING day — a Friday stop-out gets ZERO effective cooldown
    (window dies Saturday). The R21 fix itself carried a member of the class
    it fixed.
[3] Scorecard-integrity trio: meta_label trains on 6 phantom $0 never-filled
    rows (26% of its set, missing the shared not-real-close filter);
    reconciler $0 needs_review sentinels would count as real closes in
    PF/win-rate with ZERO consumers surfacing the review; status shows 5
    open positions when the book holds 2 (cancelled rows leak the filter).
FIX WAVES (pre-registered, NOT yet executed): wave 1 = live_money divergent
(units-scale-05, numeric-pairs-02) + scorecard trio; wave 2 = live-vs-backtest
mirror cluster BEFORE the next research run (else the apparatus still is not
honest: debit stop 0.35 vs 0.50, sizing, iv floors, delta_short triplication,
macro day-counting, intraday path skipping touch stop); wave 3 = ops
(dead dashboard keys incl. unconditional 'No errors logged', unreachable
breaker-coherence alarm, naive-clock timestamp sites). Critic blind spots
recorded in the register: concurrency contracts, multi-process sqlite,
consistent-but-wrong, deprecated/ + scripts/ under-swept.

## 2026-08-26 - R23: BLIND-SPOT + COMPOSITION HUNTS — 77 MORE STANDING; GRAND REGISTER 123
User directive: audit with COLLECTIVE repo context before fixing anything.
Round 2 (8 blind-spot classes, 17 agents): 50 standing (22 live_money).
Round 3 (4 vertical whole-flow audits with full flow maps, 8 agents): 27
standing (17 live_money). Register: reports/blindspot_composition_hunt_20260825.md.
Combined with R22's 46: 123 standing findings, none fixed yet.
HEADLINE CLUSTERS (critic-concurred):
[A] BREAKER BYPASS FAMILY: reconciler-booked closes (bot-down fills, manual
    flattens, expiries) hit trades.realized_pnl but NEVER the circuit
    breaker/daily stats/PDT/Thompson (trade-life-close-booking-two-owners,
    money-flow-01); fill-booking callback loss on crash loses the booking
    entirely (concurrency-2); breaker state silently resets on unreadable DB
    at relaunch (fail-direction-10). The 3-loss pause + daily-loss halt are
    defeatable from BOTH ends.
[B] FILL-VS-SWEEP RACE (concurrency-1): a stale entry that fills as the $0
    sweep books it leaves a REAL condor at the broker managed by NOTHING,
    with entry_price rewritten on a closed $0 row.
[C] FAIL-OPEN GATE CHAIN (common-cause): IV outage fabricates iv_rank=50
    passing both IV gates + quoteless strikes rank most-liquid + combo NBBO
    sanity fails open + duplicate-order guards fail open at debug — one
    degraded-feed morning can chain these.
[D] SCORECARD CONTRADICTION: Friday's automated GO-LIVE SCORECARD computes
    the RETIRED all-strategy metric (PF 0.89 FAIL) while status.py computes
    the pinned R19d IC metric (PF 1.42 PASS); master also authorizes at 25
    closes vs the pinned 50; maxDD criterion structurally unanswerable.
[E] RESEARCH COST-MODEL SIGN RISK: commissions ~$2.13/contract understated +
    gap-through-strike losses booked at touch price + IV skew ~10x flatter
    than measured — together the scale of the whole per-condor expectancy;
    can flip the researched edge's sign. Plus dead-surface-2: the 08-10
    --apply run permanently zeroed the only live fill-quality ground truth.
[F] PROTECTION-ARMING WINDOWS (bot-day): hang-detector kills slow startups
    in a loop (no reconcile/no touch stop all session); startup training
    blocks protection arming; exit-order-then-CLOSING crash window
    duplicates closes; scan duration stretches the 30s exit cadence.
NEVER-AUDITED (registered for a future round): IBC/TWS lifecycle (C:\IBC),
paper-vs-real fill realism of the verdict evidence, broker-statement P&L
parity, ML feature/label lookahead, backfill/calibration scripts,
deprecated/ subtree, fixture DDL drift, early-close sessions, config.yaml
multi-writer races, host substrate (backups, disk, Windows Update).
SEVERITY ELEVATIONS accepted: multiprocess-db-1, dead-surface-2,
log-contracts-5, policy-vs-impl-5 -> treat as live_money-adjacent;
R22's string-contracts-1/-4 + db-contracts-1 likewise.
FIX WAVES REVISED (supersedes R22 waves; still pre-registered, not executed):
W1 BOOKING INTEGRITY: one shared close-booking authority (outbox pattern) so
   EVERY close — executor, reconciler, sweep, vanished — reaches breaker/
   stats/PDT/Thompson exactly once, transactionally; CAS-gate the fill
   writes (concurrency-1); persist breaker state fail-closed.
W2 GATE FAIL-DIRECTION: protective gates fail CLOSED (or loud): iv_rank
   sentinel None not 50, liquidity refuses quoteless, NBBO sanity blocks,
   dup guards escalate, VIX outage pages (it already fails closed but
   silently), touch-stop except pages at CRITICAL.
W3 SCORECARD TRUTH: master scorecard computes the R19d IC metric with the
   50-close denominator + not-real-close authority (merges R22 trio);
   retire the contradicting Friday page.
W4 LIVE ECONOMICS COHERENCE: cooldown trading-day semantics, live range-gate
   horizon/threshold from shared authority, credit floors re-checked at
   fill, entry_vix LKG.
W5 RESEARCH HONESTY: commissions/skew/gap pricing in engine; then re-run
   studies (supersedes the earlier plan to re-run immediately post-merge).
W6 OPS: protection-arming windows, dead-man attests bot not supervisor,
   dashboard dead panels, log contracts.

## 2026-08-26 - W1 EXECUTED: BOOKING INTEGRITY (R23 breaker-bypass family closed)
[1] EXIT OUTBOX (concurrency-2, trade-life-close-booking-two-owners,
    money-flow-01): close_trade now enqueues a booking obligation in the
    SAME sqlite transaction as the close; every booking path CLAIMS the row
    first (single DELETE, sqlite-serialized => exactly-once). The executor
    callback claims before booking; a new orchestrator cycle drain
    (_drain_exit_outbox, 120s grace) books the orphans — reconciler/sweep/
    vanished closes and bookings lost to a crash — into daily stats, the
    circuit breaker, PDT, and Thompson, with an ORPHAN CLOSE BOOKED page.
    Prior-day orphans are claimed but NOT booked into today's daily
    quantities (logged stale_exit_booking_skipped).
[2] CAS-GATED FILLS (concurrency-1): _update_trade_filled/_update_trade_partial
    no longer write entry_price/quantity/open_positions when the status CAS
    is refused — the closed $0 row stays intact, and a CRITICAL page fires
    via the alert_fill_after_close flag (drained each cycle).
[3] BREAKER FAILS CLOSED (fail-direction-10): an unreadable/corrupt breaker
    store now trips conservatively for one pause window (auto-resume) instead
    of starting fresh and silently clearing an active 3-loss pause. R16's
    test_corrupt_blob_is_survivable updated to the new contract.
Smoke guard taught that a new EMPTY table is startup DDL, not a write.
PROOF: tests/test_w1_booking_integrity.py — 13 executing tests (real
StateManager/CircuitBreaker/TradeExecutor on a real sqlite file); stash
proof: 3 key tests FAIL against pre-W1 src. Suite 1254 green, type gate OK,
SMOKE PASS 13. NOTE: fixes take effect on next bot restart (restart already
pending for R21).

## 2026-08-31 - W2 + W4 + W5 EXECUTED (4 parallel agents, strict file ownership)
W2 GATE FAIL-DIRECTION — protective gates now fail CLOSED or LOUD:
[fail-direction-05] _estimate_iv_rank returned a FABRICATED 50.0 on any
  exception (passing BOTH IV gates during exactly the outages that break IV
  data). Now returns None; _scan_symbol skips NEW entries for that symbol and
  pages 1/hour. Exits unaffected (they never traverse _scan_symbol).
[fail-direction-02] With entry gates ON and the range model missing/untrained,
  IC/strangle signals silently kept DIRECTIONAL confidence (gate inverted by
  regime). Now dropped + paged. Gates-off keeps R16 observe-only behavior.
[fail-direction-01] VIX-outage full stop was fail-closed but SILENT -> pages.
[fail-direction-06 + external-contracts-02] Quote-less contracts scored spread
  0.0 = TIGHTEST in the chain and their stale last prints fed credit math.
  Spread is now ILLIQUID_SPREAD_PCT=inf (no threshold can admit it), is_liquid
  rejects no-two-sided-quote up front, crossed books (which scored NEGATIVE)
  rejected, filter_liquid logs the drop count.
[fail-direction-04] Touch-stop except swallowed at debug (silently disabling
  the ONLY credit loss exit) -> log.error every tick + one page per outage,
  latch cleared on recovery; bare except retained so one bad row cannot kill
  the loop over other positions.
[fail-direction-07] Combo-NBBO gate failed open quietly -> ENTRIES refused on
  missing/nan/crossed NBBO; EXITS still proceed but log.error (deliberate
  asymmetry: a missed exit costs more than a missed entry). 1.5s flat settle
  -> 15x0.1s poll (a book answering in 200ms no longer adds 1.3s of staleness
  to the staleness check itself).
[fail-direction-08] Duplicate-order guards swallowed their own errors at debug
  and placed anyway; a DISCONNECTED client returns [] indistinguishable from
  "verified empty". Three-way verdict clear/duplicate/unverified; unverified
  refuses the entry this pass.
[fail-direction-09] PDT guard fail-open on corrupt state/empty calendar ->
  blocks; window counting no longer returns 0 on degradation.
[fail-direction-11] Restricted list: utf-8-sig + UTF-16 retry (PowerShell
  writes), unreadable-but-present -> sentinel -> ENTRIES HALTED (an operator
  who dropped a ban file gets protection, not silence).
[fail-direction-12] Executor market-hours guard refuses on calendar exception.
W4 LIVE ECONOMICS COHERENCE:
[numeric-pairs-02] Cooldown is now ONE TRADING DAY (09:30 ET of the previous
  session) not a flat 30h — a Friday stop gave ZERO effective cooldown.
[trade-life-entry-vix-refetch] entry_vix reuses the validated value / LKG;
  writes NULL, never a false 0.
[trade-life-credit-econ-floors] Credit floors re-checked at REPRICE; ladder
  remainder refused on breach (the resting compliant order stays).
[units-scale-05] NEW src/ait/ml/range_spec.py = the ONE authority:
  live_range_spec() -> (0.05, dte_range[0] - EXPIRY_APPROACHING_DTE) = (0.05, 9)
  today. Orchestrator + trainer read it; a LOADED model keeping a different
  trained spec logs range_model_spec_mismatch (artifacts not deleted).
W5 RESEARCH HONESTY (engine cost model — measured from our OWN data):
[commission] Ledger probe, 78 leg fills: all-in 0.9168/contract-leg. Condor
  round trip $7.33 vs modeled $5.20 = $2.13/contract understated. Config homes
  commission_per_contract 0.65 + regulatory_fees_per_contract 0.2668.
[touch-gap] A day OPENING beyond a short strike was booked at the strike-touch
  price. Now reprices at the gapped open, books the WORSE of the two:
  $305/$681/$1,344 per contract previously hidden at 1%/2%/3.5% gaps.
[skew] Every strike priced at the SAME IV. Calibrated on 3,159 of our own IBKR
  quotes (put dIV/d|ln(K/S)| 1.141 SPY / 0.983 QQQ / 0.916 IWM, weighted 1.049)
  -> skew_slope_per_pct_otm 0.0736. An 8%-OTM 20-DTE put wing: $0.037 -> $0.740,
  inside the measured $0.71-0.92 real band. DELIBERATE: slope 0 reproduces
  PRE-W5 pricing (the engine was never literally flat); the CALL side is left
  alone because the same regression gave a sign-unstable slope (+0.064/+0.112/
  -0.089) and the model already OVER-prices short calls.
CONSEQUENCE (must not be soft-pedaled): every backtest/walk-forward/Optuna/
ablation/shadow-tournament ABSOLUTE produced before W5 is STALE — all three
corrections push the same pessimistic direction. Nothing in reports/ may be
quoted until a re-run. This is the wave that can flip the sign of the
researched edge; which way is not yet known.
PROOF: 4 new test files, 129 executing tests (38 orchestrator + 22 chain/
portfolio + 41 executor/risk + 28 engine). Each agent independently verified
its tests FAIL against pre-fix source (revert-and-restore probes). Suite 1383
green, type gate OK, SMOKE PASS 13.
INTEGRATION FIX (owner): tests/test_hot_path_smoke.py gained FakeRangePredictor
— the rig's MODEL_DIR is cwd-relative so no range.pkl exists in the sandbox;
that test passed ONLY because the range gate failed open (fail-direction-02
itself). Same boundary-swap pattern the rig already uses for DirectionPredictor.
REGISTERED FOLLOW-UPS (not done): master.py:830 builds RangePredictor at
horizon 30 inside the retrain subprocess -> will flap against the horizon-9
authority every retrain cycle (visible via range_train_spec_mismatch);
walkforward.py:57 still pins commission_per_contract 0.65 (fees DO resolve, so
windows get corrected friction); skew/commission are static measured constants,
not a live calibration path. W3 (scorecard truth) and W6 (ops) remain.

## 2026-08-31 - W3 + W6 EXECUTED — ALL SIX WAVES NOW COMPLETE
W3 SCORECARD TRUTH — one authority for the go-live verdict:
NEW src/ait/reporting/go_live.py: compute_go_live_verdict() + the not-real-close
  patterns + OPEN_TRADE_STATUSES (derived from TradeStatus) + gate constants.
  status.py AND master.py's Friday page now call the SAME functions; a test
  runs both against one DB and asserts every gate line is BYTE-IDENTICAL.
[policy-vs-impl-1/-3, money-flow-04] The Friday page graded the RETIRED
  all-strategy population; now IC-only via GO_LIVE_VERDICT_STRATEGIES, with
  both thresholds shown (50 prelim / 100 funding). CORRECTION to the R23
  register: master did NOT authorize at 25 closes — it already read /50. The
  real defects were the population and the missing funding threshold. No
  25-close constant exists in the repo.
[policy-vs-impl-2] A lifetime DOLLAR MEAN was rendered under a label promising
  a percent-of-credit MEDIAN -> now the pinned statistic (median |fill-mid| as
  % of credit, trailing 20 fills) plus the never-computed no-worsening clause.
[policy-vs-impl-5] The 2 unanswerable criteria printed as nothing; now
  UNAVAILABLE WITH THE REASON, and a failure prints GATE READOUT FAILED
  instead of silently deleting the whole readout.
[string-contracts-1] meta_label.build_training_data: 24 -> 18 rows on the live
  DB (the 6 never-filled $0 phantoms, 26% of the training set, are gone).
[string-contracts-4] Reconciler $0 sentinels named as constants and counted
  correctly (written strings deliberately UNCHANGED — other consumers read
  them); the scorecard now SURFACES excluded rows instead of hiding them.
[db-contracts-4] Open-position queries derive from TradeStatus (cancelled rows
  no longer read as open).
W6 OPS SURFACES — no surface may report green on an unwired channel:
NEW src/ait/monitoring/ops_health.py: ANSI-robust log parsing, bounded tail
  reads + incremental day counters, native-crash counting, bot-liveness
  evidence, and the bot_state health channel (plain sqlite, never the DuckDB
  RW lock, throttled 1/min, disabled under pytest).
[string-contracts-5, db-contracts-6] The 5 dead dashboard keys: watchdog now
  PUBLISHES component health + memory + an error ring; model_version re-wired
  to trade_context (a real source); meta_label_stats left honestly blank. The
  unconditional green st.success('No errors logged') is GONE — green now
  requires wired AND fresh AND empty; absent/stale say "NOT WIRED ... this is
  NOT an all-clear".
[log-contracts-1/-4/-6] web_logs: 76MB full re-read every 5s per tab -> bounded
  seek (0.01s); counters no longer derive from the display window and walk
  today's rotated backups; UTC->LOCAL date bucketing; ANSI-robust parsing;
  HTML-escaped output. Live: 63 orders / 39 ML predictions today where the old
  code reported 0.
[log-contracts-3] Crashes counted from logs/fatal.log (the real sink), with
  "nothing records crashes" distinguished from "no crashes".
[dead-surface-4] Capture efficiency was DOLLARS / PERCENT: -1653% -> median
  +70.7% (winners 85-96%), a usable exit-tuning number.
[dead-surface-3] Direction accuracy states plainly that state.close_trade never
  writes the column — "NOT RECORDED", not "no data yet".
[bot-day-02] Dead-man ping GATED on liveness evidence (heartbeat < 900s = 30
  missed beats, RTH-only, post-open warmup — the SAME threshold master.py uses
  for bot_hung_heartbeat_stale, so no new alert surface); not-alive sends
  /fail. keeper_ait.bat rewired; verify_deadman confirms "ping gated: True".
INTEGRATION (owner, cross-boundary items the agents flagged):
- All 6 remaining consumers rewired to the ONE not-real-close authority
  (dashboard, analytics, duckdb_analytics, learning/analyzer, shadow_referee,
  restate_d1) — each had omitted the reconciler sentinels.
- status.py: crash counting -> fatal.log; today-counts -> ops_health (was a
  fixed tail window + UTC-vs-local substring match).
- orchestrator: log.error("loop_impaired") now EMITTED as a structured event
  (it existed only inside the Telegram f-string, so every log consumer
  reported a healthy loop through an R8-class outage).
- keeper_ait.bat gated (above). tests/test_w6_ops_surfaces.py's
  test_the_real_keeper_is_detected_as_ungated INVERTED to
  test_the_real_keeper_is_gated — it pinned a defect that is now fixed.
PROOF: W3 32 tests (13 fail pre-fix), W6 63 tests (43 fail pre-fix). Suite
1478 green, type gate OK, SMOKE PASS 13.
STILL OPEN (registered, not done): master.py:830 builds RangePredictor at
horizon 30 inside the retrain subprocess (will flap against the horizon-9
authority every cycle — visible via range_train_spec_mismatch);
walkforward.py:57 pins commission_per_contract; state.close_trade never writes
direction_correct; scripts/check_first_rth_liveness.py has the old tail/regex
defects; docs/RUNBOOK.md:131 still says ">=50 closes" only; status_server's
8503 page still exposes no gate data. THE BIG ONE: every pre-W5 study absolute
is stale — the walk-forward re-run on the corrected cost model is the next
milestone and the real test of whether the IC edge survives.

## 2026-08-31 - RESTART + research-to-live-01 MATERIALIZED AND FIXED
BOT RESTARTED 15:50 ET (deliberately before the 15:58-16:13 duplicate-EOD
window, a registered unfixed finding). Clean boot: IBKR connected client_id 1
no fallback, startup_reconcile matched/promoted/stale 0, ZERO cycle or monitor
errors. R21 through W6 are now LIVE. range_model_spec_mismatch logged exactly
as designed (artifact trained at horizon 30 vs the horizon-9 authority; it
serves at its REAL spec and flags needs-retrain rather than pretending).
Dead-man verified: "ping gated on bot: True", liveness ALIVE, STATUS ARMED.
TRADING: two closes today, the book's best day — SPY +$280.36 (66.9%) and
QQQ +$329.36 (53.6%), both take_profit_short. IC verdict now 10 closes,
6W-4L, net +$790.32, PF 2.85 PASS, maxDD 3.3% of deployed risk PASS. Sample
10/50 is the only failing criterion. Book FLAT.
DEFECT FOUND VIA THE NIGHTLY REPORT (not a W5 regression): the 16:40 backtest
wrote exit_code 1 / "ERROR: No data fetched. Check internet connection." Root
cause is research-to-live-01 from the R23 register, materializing on almost
exactly the predicted date: load_daily_ohlcv judged the IB store sufficient
with a FIXED `len(df) < 60`, independent of the requested window. The intraday
store crossed 60 trading days ~2026-08-26, so every research entry point
asking for 730 days silently got 63 — and the failure was misattributed to the
network. FIX: sufficiency is measured against the REQUEST
(expected = days*252/365; need >= max(60, 80% of expected)), and a frame that
is still short AFTER the fallback logs daily_ohlcv_coverage_short instead of
looking healthy. Verified live: SPY 730d now returns 522 rows spanning
2024-08-01..2026-08-31 (was 63).
PROOF: tests/test_research_data_coverage.py, 5 executing tests against a real
sqlite store with the Yahoo boundary stubbed (offline/deterministic): the
2026-08-31 failure itself, store-preferred-when-sufficient, short-request via
the absolute floor, short-after-fallback logged, dead-Yahoo degradation.
Suite 1483 green, type gate OK, SMOKE PASS 13.
NOTE: this was BLOCKING the W5 re-run — the corrected-cost-model walk-forward
could not have run at all. That re-run is now unblocked and remains the next
milestone.

## 2026-09-01 - OBSERVE MODE PRE-REGISTRATION (user decision, written BEFORE results)
SITUATION (evidence, 2026-09-01): the overnight retrain rebuilt the range
model at the corrected horizon 9 (W2's spec-mismatch flag worked). Its per-
symbol skill is near coin-flip: mean CV AUC SPY 0.5453, QQQ 0.5446, IWM
0.5808 -> edge over baseline +0.045 / +0.045 / +0.081 against a
min_edge_over_baseline of 0.05. So predict() returns None for SPY and QQQ,
and W2's fail-CLOSED range gate blocks every IC entry on them. Today: 16
condor candidates logged all_gates_passed, ZERO orders placed, book flat.
Before W2 this gate failed OPEN, so those same condors traded on DIRECTIONAL
confidence — the inverted-gate defect. We have gone from "trades on an
inverted gate" to "cannot trade at all". The model is telling the truth: it
has no demonstrated skill on SPY/QQQ.
THE CONFLICT: the project exists to answer "does the iron condor have edge?"
and needs 50 IC closes; we have 10. A near-coin-flip FILTER must not be what
prevents the underlying question from ever being answered.
DECISION (user, option 1 of 3 offered): run the entry gates in OBSERVE MODE —
ml.entry_gates_enabled false. The ML models keep predicting and logging
(evidence accumulates) but never veto. Rejected: lowering
min_edge_over_baseline 0.05->0.04 (moving the goalposts to get the answer we
want) and leaving entries blocked indefinitely (the verdict stalls and we
learn nothing).
MANDATORY COMPANION FIX — without it this decision BACKFIRES. Registered
finding trade-life-gatesoff-reintroduces-neutral-autoreject: with gates off,
model_overridden is empty, so eff_conf falls back to the DIRECTIONAL
final_confidence (orchestrator.py:1833) and manager.py:351 rejects
confidence < risk.min_confidence (0.50). In the neutral regime a condor
WANTS, directional confidence is below 0.50 -> rejected; only trending-
aligned days clear it. Flipping the flag alone would swap "blocked
everywhere" for ADVERSE SELECTION (condors entering only in their worst
regime). Fix: in observe mode, direction-neutral structures are validated on
a config-homed neutral baseline instead of the directional number, because
the risk manager's min_confidence gate is a DIRECTIONAL gate and direction is
not what a market-neutral structure is paid for. All other gates (VIX,
liquidity, credit floors, concentration, correlation, PDT, cooldown, macro)
remain fully armed and unchanged.
WHAT WE EXPECT / HOW WE WILL JUDGE IT (pre-registered so it cannot be
rationalized later): entries resume on SPY/QQQ/IWM in neutral regimes; the
IC sample grows toward 50. The ML gate's value becomes MEASURABLE rather
than assumed — every logged range_prediction is scored against the realized
outcome, so at 50 closes we can ask whether the gate WOULD have helped
(compare closes where p_in_range was above vs below 0.65). If the observe-
mode record shows the gate would have improved PF, we re-arm it; if it shows
no discrimination, the gate is retired rather than carried as unexamined
faith. RISK ACCEPTED: entries that the (unskilled) gate would have blocked
may lose money; that is the price of measuring the filter instead of
believing it.

## 2026-09-01 - OBSERVE MODE EXECUTED (registration above)
config.yaml ml.entry_gates_enabled: false (declared EXPLICITLY — R19c: config
is the operating source; the flag was previously absent and ran the code
default). New config home ml.observe_mode_neutral_confidence 0.60.
COMPANION FIX SHIPPED (orchestrator eff_conf block): in observe mode a
direction-neutral structure (RANGE_GATED_STRATEGIES) is validated on the
neutral baseline instead of the DIRECTIONAL confidence. Without it the flag
flip alone would have produced ADVERSE SELECTION — condors entering only on
trending-aligned days, their worst regime. Gates-ON behaviour is untouched:
the payoff-matched model still owns the number.
PROOF: tests/test_observe_mode_neutral_confidence.py, 7 executing tests (real
risk-config contract, real orchestrator scan via the hot-path smoke rig in
collect mode). Executed pre-fix proof: reverting ONLY the eff_conf hunk makes
test_condor_carries_the_neutral_baseline_not_the_directional_number FAIL,
restored -> 7 pass. The premise is pinned too: a neutral-regime directional
confidence (0.42) is BELOW risk.min_confidence (0.50), so the fix is
load-bearing rather than decorative.
tests/test_w2_orchestrator_gates.py::TestRangeGateFailsClosed now ARMS the
gate explicitly (autouse fixture) instead of inheriting the shipped config —
the fail-closed behaviour stays proven and ready to re-arm.
Suite 1490 green, type gate OK, SMOKE PASS 13.

## 2026-09-06 - R24 FULL AUDIT: 109 STANDING (21 live_money) + TRIAGE SAYS 78 OF 123 STILL OPEN
User: "full audit again, every angle, every bug, every hardcoded, every parameter, every
logic and reason". 16 angles, 137 raw candidates, STRICT two-lens adversarial verification
(code-reading + execution probe; both must concede) -> 109 standing, 28 refuted. 80 agents,
0 errors. Register: reports/full_audit_r24_20260906.md.
CORRECTION TO THE RECORD: I described W1-W6 as "all six waves complete". The wave SCOPES
were complete; the REGISTER was not. Triage of the 123 prior findings: 28 FIXED, 19
PARTIAL, 78 OPEN. The waves closed the headline clusters, not the list.
THE TWO THAT DOMINATE EVERYTHING ELSE:
[1] logic-exit-risk-01/-02 — THE STOP-LOSS DOES NOT EXIST AS A MECHANISM. Every multi-leg
    exit since the R16 "quote the raw Bag" fix takes the NO-QUOTE fallback, which prices a
    credit close at max(2*mark, entry_credit + 0.25*width) = 6-10x the marked cost at the
    promoted $39-60 wings (SPY sent as BUY LMT 14.04 against a ~1.4 mark). 61 of 63 exit
    attempts were REJECTED by IBKR's price band — the broker's own guard is the only thing
    that prevented paying it. And a price-band reject is handled exactly like a cancel:
    revert CLOSING->FILLED, back off, re-place the SAME unexecutable limit forever; nothing
    reads Warning 202's acceptable price. On 08-31 the SPY take-profit looped 32 min and QQQ
    2h55m. On paper this corrupts the sample; at $3k it converts a defined-risk condor into
    a full-width loss. FIX THIS FIRST — it outranks every accounting finding.
[2] logic-entry-01 + params-trading-01 — THE FUNDING PLAN CANNOT TRADE. At the documented
    $3,000 CAD (~$2.1k USD), even after the PLAN gate-3 recipe (max_position_risk_pct
    0.03->0.07), the sizer's viability check (max_position_pct 0.05 x NLV = $105) refuses
    every $2-wide condor collecting the historical $0.67-0.72 credit (max_loss $130 > $105).
    Deterministically ZERO entries while the launch self-test reports problems=0. Phase 2 —
    the designated PRIMARY go-live evidence — opens no trades. Day-1 certainty, not a tail.
OTHER live_money CLUSTERS: W1 outbox is at-most-once not exactly-once (concurrency-01: a
kill between the claim DELETE and the booking write loses the obligation entirely);
claim-raises-then-drain double-books one loss into the breaker (regress-live-02,
concurrency-02); the drain books $0 not-real-close sentinels as LOSSES and RESETS the
consecutive-loss streak (regress-live-01, pnl-money-01); the fail-closed breaker overwrites
an intact persisted streak on auto-resume (regress-live-04); PDT history rewritten to a
single entry on a transient lock (regress-live-05, live at funding); credit floor enforced
at neither placement nor reprice (regress-live-06); the cooldown caller still swallows every
exception so a calendar/lock fault silently disables it (regress-live-07, time-calendar-02);
the pre-event blackout counts WEEKDAYS so holidays make it fail OPEN, opposite to its own
docstring (time-calendar-01); the cooldown is per-SYMBOL over three ETFs the bot's own
correlation module calls one 0.95 cluster (logic-exit-risk-03); trades_taken counts orders
PLACED and is never decremented, so phantom orders spend the intraday entry budget
(data-integrity-07); Gateway logoff + dead bot is an ABSORBING state the supervisor never
escapes — already materialized 2026-07-01 for 5h10m, 3h15m inside RTH (ops-substrate-01);
RUNBOOK "Full stop" never stops the keeper, which relaunches within 90s mid-procedure
(docs-policy-04).
THE PRE-REGISTERED OBSERVE-MODE VERDICT IS UNEXECUTABLE (8 findings, critic says merge as
ONE): the entry-time range probability is never persisted — it exists only in log lines that
rotate in ~9 trading days, was never produced for SPY/QQQ at all (predict() exits at the
min-edge gate), trades.ml_confidence silently changed meaning on 09-01, and
cleanup_old_logs will unlink the durable reports/ half on 2026-10-01. The only decision rule
we pre-registered cannot be scored. Fix before more observe-mode closes accumulate.
CRITIC — NEVER AUDITED BY ANY ROUND: src/ait/broker/* (844-line ibkr_client, combo leg
ratios/signs, orderRef/permId reuse), dashboard/app.py (2102 lines, 12+ raw trades queries),
the CAD->USD chain account.py -> capital_tiers -> position_sizer (a mismatch is ~35% sizing
error on day 1), Reg-T margin vs the %-of-NLV budget, correlation.py/selector.py/thompson.py,
ops_health's own protection logic, notifications/telegram.py (every alert rides it),
who writes executions.live_mid (go-live criterion [4] may be structurally unmeasurable),
schema-migration and sqlite->duckdb copy as data-loss surfaces. ALSO: go_live.py:163-167 —
the go-live bar itself (50/100 closes, PF 1.3, DD 8%, slip 8%) is an unexamined code literal
with no config home, i.e. the threshold authorizing real capital is outside the R19c policy.
NO FIXES APPLIED. Fix order proposed: (W7) exit pricing + reject handling; (W8) funding-size
viability; (W9) booking exactly-once + breaker integrity; (W10) observe-mode measurability;
(W11) fail-open gates + calendar; (W12) ops substrate.

## 2026-09-06 - W7 EXECUTED: THE STOP-LOSS NOW EXISTS AS A MECHANISM
Fixes R24's two dominant findings (logic-exit-risk-01/-02). Three linked
changes, because the exit was broken in three ways at once:
[1] PRICE CEILING ANCHORED TO THE MARK. The R16 bound was
    max(2*mark, entry_credit + 0.25*wing_width). The second term was meant as
    a floor for an unreliable mark but sits inside a max(), so at the promoted
    $39-60 wings it DOMINATED: SPY marked 1.40 priced its take-profit at
    4.29 + 0.25*39 = 14.04 (10x), QQQ 2.94 -> 18.60 (6.3x). New helper
    _exit_price_ceiling = min(width, max(mark + exit_cross, mult * mark)) with
    mult config-homed (exit.exit_mark_multiple 1.5). SPY 14.04 -> 2.10;
    QQQ 18.60 -> 4.41; the 08-24 touch stop 19.87 -> 13.14. The wing width
    stays the structural cap; mark+cross is the floor so the order remains
    marketable. Applied to BOTH the quoted anchor and the no-quote fallback.
[2] THE BROKER'S STATED PRICE IS NOW HONOURED. IBKR rejects with "Warning 202
    ... more aggressive than 1.745 ... current market price of 1.42" and the
    bot DISCARDED that text, re-placing the identical unexecutable limit every
    backoff (SPY looped 32 min, QQQ 2h55m on 08-31; a touch stop would loop
    while the loss grew). TradeExecutor.parse_price_band_reject mines the
    order log, the cancel path persists {band, market, at} to bot_state, and
    _exit_band_cap feeds it back as a hard cap on the next attempt (stale >1h
    ignored so an old band cannot pin a moved market).
[3] ROOT CAUSE — THE BAG NEVER TICKED. combo_exit_limit appears ZERO times in
    any retained log: all 63 exits took the no-quote branch and were priced by
    formula, while the ENTRY path's Bag quotes arrive fine. New
    _synthesize_combo_cost quotes the FOUR LEGS individually (buy-back legs at
    the ask, sold legs at the bid) and prices the exit from the real market.
    Returns None if ANY leg is unquotable — a partial sum would understate the
    cost and place an unfillable order. Subscriptions are cancelled 1:1 even
    on exception (the leaked-sub -> market-data-flood -> native-crash class).
PROOF: tests/test_w7_exit_pricing.py, 20 executing tests driving the real
functions with the real incident numbers. Pre-fix proof: restoring ONLY the
old bound makes 5 of them fail; restored -> 20 pass. A test caught a genuine
defect during development: _exit_band_cap used json.loads but orchestrator.py
has no module-level json import, so it raised NameError into its own except
and silently returned None — the exact silent-failure class this project keeps
finding. Fixed with a local import.
Suite 1510 green, type gate OK, SMOKE PASS 13.
NOT YET DEPLOYED — takes effect on the next bot restart.
REMAINING FROM R24 (unchanged): W8 funding-size viability (at $3k the sizer
refuses every viable condor while the self-test reports problems=0), W9
booking exactly-once, W10 observe-mode measurability, W11 fail-open gates +
holiday blackout, W12 ops substrate (Gateway absorbing state).

## 2026-09-07 - W10 EXECUTED: THE OBSERVE-MODE VERDICT IS NOW SCOREABLE
R24 found the 2026-09-01 pre-registration unexecutable: the rule "compare
closes where p_in_range was above vs below 0.65" had no data behind it.
PROOF FROM THE LIVE DB (why it was broken): trade_context.entry_confidence
holds 0.8977 for the 08-24 trade (a range-model probability, written because
the gates were armed) and exactly 0.60 for both 09-01 trades (the observe-mode
neutral baseline). The column silently changed MEANING at the mode switch, the
probability had no column of its own, and the only surviving copy was a log
line — logs rotate in ~9 trading days while the verdict is scored at 50-100
closes. For SPY/QQQ no probability was ever produced at all (predict() exits
at the min-edge gate), and "no prediction" is not a low probability.
FIX: six migrated columns on trade_context — range_probability,
range_threshold_pct, range_horizon_days, range_model_version,
entry_gates_mode, directional_confidence. _scan_symbol stashes the prediction
(INCLUDING its absence) per symbol; _entry_range_context feeds it into
save_trade_context at entry. NULL means "the model produced nothing", which is
the honest record for an edge-gate failure and must never be written as 0.0
(0.0 would read as "certain to leave the range" and land on the wrong side of
the 0.65 split). The spec is stored WITH the number because the live model was
retrained from horizon 30 to 9 — a probability without its horizon is not
comparable across that change.
PROOF: tests/test_w10_observe_mode_measurability.py, 11 executing tests
(real StateManager migrations, real orchestrator helper, and an end-to-end
test that runs the pre-registered above/below-0.65 split as SQL against
stored rows, asserting the three-way above/below/unscored partition).
Suite 1521 green, type gate OK, SMOKE PASS 13.
HONEST LIMIT: this fixes measurement from NOW ON. The 10 closes already in the
verdict sample, and the two open positions entered 09-01, have no stored
probability and never will — they are permanently unscoreable against the
pre-registered rule. The observe-mode evaluation therefore starts from the
next entry, not from the existing sample.
NOT YET DEPLOYED — takes effect on the next bot restart (as does W7).

## 2026-09-07 - W11: COMMISSIONS IN P&L (measured against our own IBKR ledger) + CI FIX
MEASURED FROM 94 REAL IBKR LEG FILLS (data/ait_state.db executions):
  per-leg  min $0.6195  median $1.0230  mean $0.9089  max $1.0586
  9 closed condor round trips: mean $7.62, median $8.23, range $6.64-$8.89
The spread is bimodal (a ~$0.62 cluster and a ~$1.02-1.06 cluster) and is NOT
explained by side (BOT $0.9081 vs SLD $0.9098), by entry-vs-exit ($0.9188 vs
$0.8724), by premium tier (the same $0.01 option drew both $0.6228 and
$1.0506), by symbol, or by month — it is exchange routing, which we do not
control. What we CAN do is stop pretending the estimate is the truth.
[1] hardcoded-live-03: TradeExecutor.commission_estimate was a flat $0.65/leg
    = $5.20 per 1-lot condor round trip — IBKR's COMMISSION LINE ONLY, with no
    exchange/regulatory/clearing pass-throughs. Against the measured $7.62 that
    understated EVERY close by ~$2.42. Now reads the W5-calibrated config home
    (commission_per_contract 0.65 + regulatory_fees_per_contract 0.2668 =
    $0.9168/leg -> $7.33/condor), which matches the measured $0.9089 mean to
    within a cent. Research and live now share one cost model.
[2] regress-live-09 / pnl-money-02: the ledger true-up ran ONLY on the executor
    callback path, so any close booked by the W1 outbox drain (crash between
    the close and the callback, or a callback slower than the 120s grace) kept
    the estimate forever and left trades.commission at 0.0 — a ONE-DIRECTIONAL
    overstatement of the go-live P&L. Extracted to _true_up_commission, now
    called on BOTH paths, and it also stamps the real figure so PF reads broker
    truth. The partial-ledger defer (R16) is preserved on both.
TEST DEBT REPAID: five test files re-implemented or pinned the $0.65 formula
(test_pnl_accounting, test_properties, test_r15_review_fixes,
test_order_lifecycle) — a second source of truth that broke the moment the
authority moved. All now DERIVE from commission_estimate;
test_properties._commission delegates instead of re-implementing.
CI FIX (the failure on 7d971ed): web_logs.api_logs called
orchestrator_status() bare, whose `path=ORCH_LOG` default binds at DEF time,
so the test's monkeypatch was ignored and the assertion was satisfied by the
DEVELOPER'S real logs/orchestrator.log. Green locally, "No supervisor log" in
CI. Now reads the module global at CALL time. Proven hermetic by moving the
real log aside and re-running (5 passed). This is the same late-binding class
the W6 work fixed in verify_deadman.py and missed here.
Suite 1524 green, type gate OK, SMOKE PASS 13.

## 2026-09-08 - W12: EXIT FILL QUALITY (slippage measurable on BOTH halves)
CORRECTION TO THE BROKER STUDY'S PREMISE: it read "109 of 116 executions have
nbbo_spread=0.0" as "slippage is uninstrumented". Measuring the ledger
directly is sharper — ALL 7 entry combo rows carry live_mid and nbbo_spread;
the leg rows are zeroed DELIBERATELY (R16 puts combo context on the BAG row
only). Entry slippage has been instrumented since R19. What was missing is the
EXIT half: PendingExitOrder had no quality fields and _order_ctx_map was
written only on the entry path (executor.py:437), so every exit fill resolved
through _ledger_context to (0,0,0).
WHY IT MATTERS MORE THAN W11's COMMISSIONS: on the 7 instrumented entries the
NBBO spread averaged $0.1429/leg — ~15x the $0.9168 all-in per-leg commission
we spent yesterday calibrating. We had instrumented the small cost to four
decimals and left the large one half-measured.
FIX: PendingExitOrder gains signal_price/live_mid/nbbo_spread;
register_exit_order accepts and stores them AND mirrors them into
_order_ctx_map (which survives the pending-dict cleanup — the R13
shadow-referee break the entry path already guards); _close_multi_leg stashes
the quote context on BOTH pricing paths (BAG NBBO -> mid and ask-bid; leg
synthesis -> the synthesized cost, with spread 0 because no two-sided combo
NBBO exists there) and the orchestrator threads it at registration, consuming
the stash so it cannot leak to the next exit.
ENTRY SLIPPAGE, MEASURED FROM THE 7 EXISTING ROWS (fill vs live_mid, credit
convention so positive = better than mid): +0.355, +0.07, -0.41, -0.045,
-0.14, -0.02, 0.00 -> mean -0.027/share = about -$2.70 per contract, i.e.
comparable to the entire commission. n=7 is not a sample; the point is that
the term is the same order of magnitude as the one we just fixed, and until
today we could only see half of it.
PROOF: tests/test_w12_exit_fill_quality.py, 11 executing tests against the
real executor (slots preserved, quality reaches the tracker AND the ctx map,
resolves after tracker cleanup — the pre-fix zero case, absent quality does
not pollute the map, backwards-compatible calls still register, and entry/exit
symmetry).
Suite 1535 green, type gate OK, SMOKE PASS 13.
NOT YET DEPLOYED — next restart. No exit has been priced under W7 yet either,
so the first close will be the first real test of both.

## 2026-09-08 - W13: GATEWAY AUTO-RECOVERY (ops-substrate-01, materialized TWICE)
IT HAPPENED AGAIN TODAY, AND I WATCHED IT. A Gateway JVM started 08:56 and
NEVER opened port 4002. master.py logged bot_restart_deferred_gateway_down
every 2 minutes until I intervened manually at 12:17 — 3h21m of a trading day
with two open condors and NO exit engine. The process EXISTED, so nothing
process-level saw a fault; only the PORT was dead, and the only thing watching
the port was the deferral that did nothing. Prior occurrence 2026-07-01: 5h10m,
3h15m inside RTH. This has now cost more real trading time than every other
finding in the register combined.
FIX: BotManager._maybe_recover_gateway. Deliberately conservative, because
restarting a broker connection is not free:
  * GRACE 10 min — IB's own nightly maintenance takes the Gateway down 30-60
    min and recovers unaided; we must not fight it. 10 min is well inside a
    session but ignores a blip.
  * KILL FIRST when a JVM exists — the 09-08 shape was live-but-wedged, and a
    relaunch cannot bind a port the old process still holds. (This is why a
    process-existence check was never enough.)
  * COOLDOWN 30 min between attempts — a Gateway that cannot come up (expired
    credentials, 2FA prompt) must not be thrashed every polling cycle.
  * BUDGET 3, then ESCALATE with "MANUAL ACTION NEEDED — check for a login/2FA
    prompt". Looping forever IS the original defect; automation must say when
    it is out of ideas.
  * Timer and budget RESET on recovery (gateway_recovered logs the outage
    length), so a later unrelated outage gets fresh attempts.
  * Start script config-overridable via AIT_GATEWAY_START_SCRIPT.
PROOF: tests/test_w13_gateway_recovery.py, 12 executing tests against the real
BotManager method: grace period protects nightly maintenance (and its exact
boundary), wedged JVM is killed before relaunch vs absent process simply
started, cooldown blocks a rapid second attempt then allows one after it
expires, exhausted budget escalates instead of looping, missing script and a
raising Popen are both survivable, and a replay of today's 201-minute outage.
Suite 1547 green, type gate OK, SMOKE PASS 13.
NOT YET DEPLOYED — next restart.

## 2026-09-08 - W14: BOOKING EXACTLY-ONCE (5 findings, one design problem)
The cluster all shared one root: the claim happened BEFORE the booking rather
than atomically with it, and nothing filtered WHAT got booked.
[concurrency-01] The claim was a committed DELETE performed before any booking
  write — a crash in that window destroyed the only record of the obligation
  while nothing had been booked, and the drain had nothing left to find. Now
  TWO-PHASE: claim_exit_booking UPDATEs claimed_at (still serialized by
  sqlite, exactly one winner), complete_exit_booking DELETEs only AFTER every
  stats/breaker/PDT/Thompson write lands, and pending_exit_bookings re-offers
  a claim left stale (>300s) by a dead booker. Both booking paths complete.
[concurrency-02 / regress-live-02] A claim that RAISED set _claimed=True and
  booked anyway — but the row survived, so the drain booked the SAME close
  again 2-5 min later: one -$150 stop read as -$300 to the breaker, and two
  such exits tripped the 3-loss pause on a two-loss day. Now a failed claim
  SKIPS booking and defers to the drain, which books it exactly once.
[regress-live-01 / pnl-money-01] The drain booked the reconciler's $0
  bookkeeping rows as LOSSES, and record_trade_result(0.0) RESETS an active
  streak — two real losses plus one swept never-filled entry meant the third
  real loss no longer tripped the pause. Now filtered through W3's
  is_real_close authority (which existed for exactly these rows and the drain
  never consulted) and retired without booking.
[regress-live-04] The fail-closed breaker auto-resumes after one pause window,
  and its first _persist() OVERWROTE the still-intact blob with zeros — the
  W1 fix for one fail-open created another. The breaker now refuses to
  persist while its own state is unreadable; a good read restores authority.
[concurrency-03] A CAS refusal was ALWAYS read as "terminal row", so a
  legitimate concurrent promotion (sweep/reconciler promoting the same PENDING
  row to FILLED) discarded the REAL fill price and quantity: entry_price kept
  the optimistic signal price (overstating realized_pnl by the entry slippage,
  a verdict input) and a partial fill promoted at full size could size the
  reverse-combo exit LARGER than the live position. Now reads the row's actual
  status: FILLED falls through and records the truth; only genuinely terminal
  rows refuse and page.
BUG FOUND WHILE FIXING: the claimed_at ALTER was placed BEFORE the
exit_outbox CREATE in the same ordered migration tuple, so it failed silently
(swallowed OperationalError) and the column was absent on fresh databases —
caught because the tests exercise a fresh DB.
PROOF: tests/test_w14_booking_exactly_once.py, 15 executing tests on real
StateManager/CircuitBreaker/TradeExecutor: claim marks rather than deletes,
completion retires, stale claims re-offered while fresh ones are not, an
end-to-end crash-between-claim-and-booking recovery, all three $0 sentinel
spellings leaving a 2-loss streak intact, a real loss still booking, the
breaker refusing to persist unreadable state (and an intact streak surviving a
locked relaunch), and both CAS branches.
Suite 1562 green, type gate OK, SMOKE PASS 13. NOT YET DEPLOYED.

## 2026-09-08 - W15: TWO GATES THAT FAILED OPEN
[time-calendar-01] The pre-event blackout converted calendar days to sessions
by counting WEEKDAYS, so every NYSE holiday counted as a trading session. The
consumer blocks when sessions <= blackout, so an INFLATED count means FEWER
blocks — the gate failed OPEN across every holiday, in exactly the direction
its own docstring called fail-safe ("erring toward MORE sessions is the
fail-safe direction" — backwards). Executed proof with the clock pinned to Fri
2026-09-04 (Labor Day 09-07 a holiday): an event on Wed 09-09 counted 2
sessions when only ONE real session (Tue) separates entry from the event, so
blackout=1 did not block. Now counts real trading days via the same calendar
the cooldown uses; the Fri->Wed case reads 1 and blocks. A calendar fault
degrades to weekdays without crashing, and the R16 weekend behaviour
(Fri entry, Mon event -> 0 sessions) is preserved.
[regress-live-07 / time-calendar-02] The post-stop cooldown's caller swallowed
every exception with a bare `pass`. W4 had just given that gate a NYSE-calendar
dependency and a fresh sqlite connection, so a calendar fault or a busy/locked
DB silently DISABLED the re-entry guard for every symbol on every scan — no
log line, no page — and the bot could re-enter the symbol it was just stopped
out of (the 08-24 QQQ shape). Now fails CLOSED for credit structures with a
throttled page, matching how W2 hardened the sibling calendar-dependent gates.
Debit shapes keep the permissive behaviour: they are not the
autocorrelated-loss case the rule exists for, and a paging failure must never
un-block the gate.
DESIGN NOTE: the decision was EXTRACTED to _cooldown_blocks_entry so its fail
direction is testable by execution. My first draft of the test asserted on
inspect.getsource — the exact pattern this project forbids (it decays into a
no-op on rename, and two such tests already had). Replaced with 7 executing
tests against the real helper.
PROOF: tests/test_w15_fail_open_gates.py, 16 executing tests (holiday matrix,
the gate now blocking that entry, weekend preserved, calendar-fault
degradation; locked DB and calendar fault both blocking, the operator paged, a
failing pager not un-blocking, debit shapes permissive, active and clear
cooldowns).
Also fixed: tests/test_r16_round2_fixes.py's blackout fixture lacked
_post_stop_cooldown_until, so the new fail-closed guard correctly blocked its
entry before the budget sentinel — the stub now models the real object.
Suite 1578 green, type gate OK, SMOKE PASS 13. NOT YET DEPLOYED.

## 2026-09-14 - FIRST MACOS DEPLOY: two findings from actually running the bot
First run of feature/event-straddle (4dbba32) off the Windows box, on a
separate paper account (TWS, not Gateway). Local ait_analytics.duckdb also
purged of pre-2026-07-11-class pytest fixture pollution (105 T-ML-TRAIN-*/
T-00N rows) carried in a stale local file predating that fix; not a code
change (backup kept: ait_analytics.duckdb.bak_pre_purge_20260914).
[1] _scan_foreign_bot_pids (R16 #10) was Windows-only (PowerShell/WMI) and
    fails open — silently, every 2-min health cycle — on any non-Windows
    host: FileNotFoundError('powershell') is swallowed and read as "no
    foreign bot found." Fixed via psutil.process_iter (already a pinned
    dependency, previously used only for self memory introspection) —
    cross-platform, no shell dependency. Existing test only stubbed the
    function; new test drives the real body against a spawned process.
[2] verify_can_trade()'s startup read-only probe (BUY 1 SPY @ $1, self-
    cancelled) never logged its own trading_verified confirmation on a
    successful run — its recognized-status tuple didn't include
    PendingSubmit/PendingCancel/ApiCancelled, the exact statuses this
    self-cancelling probe passes through. Not a correctness bug (still
    returned True, never blocked trading) — a pure observability gap: zero
    log evidence this self-test ever ran. Fixed by logging trading_verified
    whenever the probe completes and isn't flagged read-only, instead of
    matching against an enumerated status list.
DEFERRED (found, not fixed): three sibling Windows-only powershell
shell-outs in master.py (_gateway_process_alive, _clear_dashboard_port,
_maybe_recover_gateway's JVM-kill) — all tied to Windows IB Gateway/IBC
process management (hardcoded C:\IBC\StartGateway.bat), which doesn't apply
to this machine's TWS-based, manually-managed setup regardless of platform.
Separate, larger question; not in scope here.
PROOF: tests/test_r16_ops_fixes.py::TestScanForeignBotPidsRealBody (2 tests)
+ tests/test_r16_broker_fixes.py::TestVerifyCanTradeLogsConfirmation (2
tests) — all 4 proven to fail against pre-fix source (git stash of the two
source files only), pass restored. Suite 1581 green (was 1577), SMOKE PASS
13. DEPLOYED same day (bot restart 13:56:59): live log confirms zero
foreign_bot_scan_failed warnings and trading_verified status=PreSubmitted
now logging on the startup probe.

## 2026-09-14 - THIRD FINDING: dashboard (8501) crashed on every load, PLUS a flawed first fix caught by verification
After the restart above, opening http://localhost:8501 (the Streamlit
monitoring dashboard) hit `RuntimeError: There is no current event loop in
thread 'ScriptRunner.scriptThread'` — deterministic, every load, not a
glitch. TRACED (file:line): dashboard/app.py imports one constant
(NOT_REAL_CLOSE_PATTERNS) from reporting/go_live.py; go_live.py imported
GO_LIVE_VERDICT_STRATEGIES from strategies/base.py; base.py imports
data/options_chain.py -> data/market_data.py -> `from ib_insync import
Stock, util`; ib_insync's eventkit dependency calls
asyncio.get_event_loop() AT IMPORT TIME, which only auto-creates a loop in
the main thread (Python 3.10+) — Streamlit re-executes a page's entire
script, imports included, inside a dedicated non-main
`ScriptRunner.scriptThread` on every load. status_server.py (8503) and
web_logs.py (8502) hit the identical import chain but are unaffected: their
imports run once at main-thread process startup (Flask's threaded=True only
threads per-REQUEST handling, not the module imports already done before
app.run()) — this is Streamlit-specific, not a three-dashboards problem.
FIRST FIX ATTEMPT WAS ITSELF WRONG, CAUGHT BEFORE SHIPPING: moved
GO_LIVE_VERDICT_STRATEGIES into a new src/ait/strategies/constants.py,
re-exported from base.py. Verifying it with a direct import still pulled in
ib_insync — root cause was one layer deeper: ait/strategies/__init__.py
itself eagerly does `from ait.strategies.base import Strategy, Signal,
SignalDirection`, so importing ANY submodule under ait.strategies (even a
brand new one with zero imports of its own) forces that package __init__ to
run first, dragging in the same heavy chain regardless. Verified directly
(import ait.strategies.constants alone still loaded ib_insync/eventkit).
CORRECTED FIX: GO_LIVE_VERDICT_STRATEGIES now defined directly in
reporting/go_live.py (that package's own __init__.py docstring already
declares it "deliberately import-light... must never pull the broker ... in
at import time" — go_live.py itself has zero ties to ait.strategies now);
strategies/base.py re-exports it FROM go_live.py instead of the reverse
(mechanically identical for the one pinned consumer,
test_verdict_strategies_come_from_base, which only asserts object equality).
Considered and rejected: moving the three sibling constants
(LIVE_CREDIT_STRATEGIES/BACKTEST_CREDIT_STRATEGIES/UNDEFINED_RISK_STRATEGIES)
too — full-repo grep found every consumer is live-trading/backtest code
that already needs the broker stack for its actual job; would have been
pure churn for zero benefit. Also considered and rejected: patching
strategies/__init__.py's own eager import to be lazy — correct instance of
the same disease, much bigger blast radius (touches how the whole package
imports everywhere) for one bug.
PROOF: tests/test_w3_scorecard_truth.py (32 tests, incl. the single-
authority pin) unaffected; full suite 1581 green; smoke_deploy PASS 13;
python -c import confirmed zero ib_insync/eventkit modules loaded via
`import ait.reporting.go_live` or `import ait.dashboard.app` (still loaded,
correctly, via `import ait.strategies.base`); LIVE VERIFIED in a real
browser (Playwright/webapp-testing skill, installed for this) after
restarting just the dashboard subprocess (killed pid 13319, supervisor
health_check relaunched at pid 20670 within one 2-min cycle) — screenshot
shows the real dashboard (Portfolio Overview, 8 tabs, correct zero-position/
zero-trade state), no RuntimeError/ScriptRunner text anywhere in the DOM.
LESSON (process, not just code): I verified this fix with a plain `import`
statement first and declared it fixed — which was WRONG, because a bare
`import ait.strategies.constants` alone doesn't reveal that the package
`__init__.py` is what's actually heavy until you check what importing the
PARENT PACKAGE costs, not just the leaf module. Caught only because
verification kept going instead of stopping at the first green check — the
same "fixes ship with bugs of their own until an adversarial pass runs"
pattern R14 named for someone else's fix.

## 2026-09-14 - FOURTH FINDING: option-chain modelGreeks gap (snapshot -> streaming)
Live option-chain scans were taking minutes per symbol with a flood of
Error 10091 in the logs. Investigated with the user in two steps:
[1] RULED OUT market data type: AIT_MARKET_DATA_TYPE 1/3/4 all made zero
    difference to the errors -- never a data-type problem.
[2] CONFIRMED Error 10091 is loud but non-blocking for reqMktData, two
    independent ways: (a) a controlled live test (isolated clientId 95,
    real OPRA-subscribed account, connected outside the bot) got real
    bid/ask in all 4 combinations of snapshot/streaming x live/delayed
    despite the error firing every time; (b) the installed ib_insync
    wrapper.py's own error handler only takes blocking action
    (_endReq / order-cancel) for one-shot future requests or orders -- a
    live market-data ticker is neither, so the error is logged at ERROR
    severity (10091 isn't in ib_insync's own warningCodes set) and
    nothing else happens. Not chasing this further; it's cosmetic noise
    for this call path, confirmed from the library's own source, not
    assumed.
[3] FOUND THE REAL GAP: options_chain.py:687 was the ONLY reqMktData call
    in the entire src/ait/ tree using snapshot=True (grepped every site --
    reconciler, executor, orchestrator x3, market_data x2 -- all already
    streamed). Snapshot's ~11s server-side collection window for
    modelGreeks never fit inside the code's 2s per-batch wait, so
    delta/gamma/theta/vega/IV silently never arrived (the exact complaint
    an earlier R16 comment on this file already recorded and never
    actually landed a fix for -- the cancel-hygiene half got fixed, the
    snapshot-mode half didn't). Live test confirmed streaming delivers
    modelGreeks within ~4s where snapshot didn't even at 11s.
FIX: snapshot=True -> False (options_chain.py:687, aligning with every
other call site in the codebase); md_wait_s 2.0 -> 3.0
(options_chain.py:674, evidence-anchored to the live test, not a guess).
Cancellation already handled correctly for either mode by an existing R16
`finally` block -- confirmed by reading the full function, no change
needed there.
NOT IN SCOPE (found, not fixed): the ±20% price-band strike filter
(options_chain.py:610) is a separate, larger question -- likely the real
driver of multi-minute-per-symbol scan WALL-CLOCK time (~1,500 contracts
for a high-priced underlying like QQQ); this fix addresses data
CORRECTNESS (Greeks actually arriving), not scan speed.
SEPARATELY FOUND IN PASSING (pre-existing, not caused by this fix, not
fixed): tests/test_r19_config_authority.py::TestContractIsTheSingleAuthority
::test_applier_and_readers_share_one_table fails on this machine because
this session's earlier AIT_MARKET_DATA_TYPE=3 line in the (gitignored,
local-only) .env is a real, deliberate override -- but
apply_runtime_env_defaults() legitimately loads the real .env file on disk,
and the test isn't isolated from it, so its "env unset -> falls back to the
CONTRACT_DEFAULTS safety net" assumption breaks the moment a local .env
sets that key to anything other than the default. Confirmed by temporarily
removing the .env line (test passes) and restoring it (test fails again).
Local-only (.env is gitignored) -- does not affect CI or any other
machine's copy of this repo. A real test-isolation gap, not touched here
(out of scope for this fix).
PROOF: tests/test_r16_data_fixes.py::TestSnapshotCancelPairing, new test
`test_requests_are_streaming_not_snapshot` pinning snapshot=False, proven
to fail against pre-fix source (git stash of the source file only).
Suite 1581 passed / 1 pre-existing-unrelated-local-env failure (see above)
/ 43 skipped, smoke_deploy PASS 13.
LIVE VERIFIED post-restart (2026-09-14, ~15:39 UTC, SPY scan): chain_source
confirmed a real IBKR fetch (3 expiries); chain_quoteless_contracts_dropped
showed only 7 of 520 and 7 of 290 contracts lacked a two-sided market
(~98.6% got real bid/ask) -- ruling out a cache hit or early-exit before
trusting the fast (~217ms) scan_symbol_timing total; zero
ibkr_none_greeks_coerced events fired in that scan, where before the fix
this was the routine path for nearly every contract. No condor signal came
out of this particular pass (range_no_edge / vol_mag_no_edge -- an ML-gate
outcome, unrelated to this fix), but the chain+Greeks pipeline itself is
now confirmed working end-to-end on the live account, not just in tests.

## 2026-09-14 - FIFTH FINDING: implied_vol never actually gets backfilled (dead solver import)
Follow-up investigation (user asked "why do we look at tick data instead of
bar data for Greeks" while digging into the earlier partial-Greeks finding):
answered directly (IBKR's API has no historical-bar endpoint for
delta/gamma/theta/vega, only for IV, and that's used elsewhere for research
history, not live strike selection) -- and while re-reading _calculate_greeks
end to end to answer it, found _calculate_greeks imports
py_vollib.black_scholes.implied_volatility.implied_volatility (line 913) --
the tool that solves IV from a real observed price -- AND NEVER CALLS IT.
Dead import. The function's actual behavior: whenever implied_vol is
missing, chain-median IV is used as scratch sigma to backfill
delta/gamma/theta/vega via forward Black-Scholes, but the contract's own
implied_vol field is NEVER corrected -- confirmed by the existing test's own
pinned assertion (test_r16_data_fixes.py:260, "assert bad.implied_vol ==
0.0 # coerced to missing") asserting the gap as correct. Compounded by the
`if c.delta != 0: continue` gate found in the prior finding: when delta
arrives from IBKR but implied_vol doesn't, the whole block is skipped, so
implied_vol never even reaches the chain-median fallback.
WEB-VALIDATED (per request): IBKR's own docs
(interactivebrokers.github.io/tws-api/option_computations.html) confirm live
Greeks need subscriptions to BOTH option and underlying -- another
contributor to partial ticks on this account's incomplete entitlement. An
experienced-user ib_insync mailing-list thread (insync@groups.io, "Delta
Hedging") explicitly recommends never trusting IBKR's derived Greeks
directly -- calibrate your own vol model against IBKR's real prices instead
-- which is exactly what wiring up the existing (dead) implied_volatility
import does.
FIX: _calculate_greeks restructured so implied_vol, delta, gamma, theta,
vega are each checked and backfilled INDEPENDENTLY instead of gated on
delta alone. When implied_vol is missing/below-floor and a real two-sided
quote exists, solve the genuine contract-specific IV via
implied_volatility(c.mid, ...) (verified live: 784P's real mid price ->
16.95%, a sensible number; degenerate prices raise a specific
BelowIntrinsicException, caught same as the existing BS block). No real
quote -> falls back to the chain median as before, but now WRITTEN onto
c.implied_vol instead of only used as scratch sigma. delta/gamma/theta/vega
already populated by IBKR are left untouched (never redundantly
recomputed).
PROOF: tests/test_r16_data_fixes.py -- one existing assertion corrected
(was pinning the bug), two new tests (delta-present/IV-missing exact
reproduction; no-quote chain-median-now-written-back), all three proven to
fail against pre-fix source (git stash of the source file only). Suite
1583 green (was 1581, +2 net new tests), same single pre-existing
unrelated local-.env test failure as the prior finding (not caused by
this), smoke_deploy PASS 13. NOT YET DEPLOYED to the running bot (restart
is the user's call, per standing preference this session).

## 2026-09-14 - R25 FULL AUDIT: the STRUCTURE audit — 4 architectural findings, 19 verified, 6 refuted
User: "plan and conduct a code audit ... document your findings in PLAN.md". First draft of the
plan was rejected by the user with the right question — "did you look at the whole design
holistically and then audit individual components line by line?" — and the answer was no. The
first plan was LEAD-DRIVEN: chase whatever reconnaissance surfaced, no design pass, no coverage
guarantee. Rebuilt on two axes: the system as a whole, then components with the gaps made
countable. That correction produced the round's headline; a per-file sweep could not have.
Register: reports/full_audit_r25_20260914.md. REGISTERED, NOT FIXED (user-confirmed, as R24).

THE FOUR ARCHITECTURAL FINDINGS — why 24 rounds and ~465 instance-fixes have not stopped this:
[A1] THE BROKER BOUNDARY DOES NOT EXIST. IBKRClient publishes the raw ib_insync handle as a
    public .ib property; 7 modules reach past it across ~40 sites. There is NO reqMktData
    wrapper — instead SIX independent copies of the request/settle/cancel dance, and
    reqMarketDataType set from two places. The 09-14 snapshot=True defect was one of the six
    having drifted. That fix closed the instance; the generator is untouched.
[A2] THE STATE BOUNDARY DOES NOT EXIST. StateManager nominally owns the DB; 13 modules and 8
    scripts sqlite3.connect the same files directly (historical.py alone 20x). Schema knowledge
    is duplicated tree-wide with no owner — which is WHY there is no schema_version anywhere,
    why state.py:194-266's migration tuple can only be idempotent-by-swallowing
    (`except OperationalError: pass` cannot tell "column exists" from "no such table"), and why
    the DuckDB mirror (six CREATE TABLE IF NOT EXISTS, ZERO ALTER TABLE) can diverge forever
    with no detector.
[A3] THE LAYERING IS CYCLIC. bot<->execution, bot<->risk, bot<->reporting, plus ml->execution.
    The 09-14 Streamlit crash was a symptom; the ledger's own account shows the first fix was
    wrong and the second WORKED AROUND the cycle by moving a constant. The cycle is still there
    and strategies/__init__.py is still eagerly heavy.
[A4] THE ORCHESTRATOR IS A GOD OBJECT. 4,336 lines, 56 methods, 89 except-blocks (21% of the
    tree), owning market data, ML, selection, risk, placement, exits, booking, commissions, IV,
    notifications, calendar, cooldowns and reconciliation — reaching past risk/manager.py,
    execution/executor.py and strategies/selector.py, which exist for those jobs. FOUR waves
    (W11/W12/W14/W15) edited it in EIGHT DAYS, none re-reading the others. "Fix velocity mints
    new audit surface" is a description of this file.

COUNTS, HONESTLY: 74 raw candidates -> 19 VERIFIED (two-lens: code-reading AND execution probe,
both conceding) / 6 REFUTED / 49 REGISTERED-UNVERIFIED. Verified severity: live_money 4,
research_validity 3, ops_reporting 10, cosmetic 2. The 49 are LEADS, NOT FINDINGS — R24 refuted
28 of 137 at exactly this stage, and they must not be cited as defects until probed.

THE HOST-MOVE CLASS — every one of these PROVEN BY RUNNING THE CODE ON THIS MAC, not by reading:
[plat-01] W13's GATEWAY RECOVERY IS INERT AND ITS ESCALATION CAN NEVER FIRE. master.py:433-435
    returns on `os.path.exists(r"C:\IBC\StartGateway.bat")` == False — BEFORE
    `self._gateway_recoveries += 1` at :436 — so the counter never increments and the
    GATEWAY_MAX_RECOVERIES escalation ("MANUAL ACTION NEEDED") at :422-431 is unreachable. W13
    exists to end the absorbing state that materialized twice, once 3h25m INSIDE RTH with open
    positions and no exit engine. That state is restored here. And
    tests/test_w13_gateway_recovery.py patches os.path.exists->True at six sites: 177 lines of
    test pass green on macOS while the mechanism they test is inert on macOS.
[plat-02] _gateway_process_alive() returns False SILENTLY (no log at all — the only one of the
    four siblings with a fully silent except), and that unmeasured False is stamped into the
    gateway_recovery_attempt record as wedged_process=False. The log asserts a fact it never
    measured.
[plat-03] A FOURTH powershell sibling the 09-14 entry missed: status.py:46-58 _proc_running().
    Probed live while the bot was running: returns (False, 0) — reports the bot DOWN while it is
    UP. Same Get-CimInstance idiom as the function fixed on 09-14: the fix went to one call
    site, not to the idiom.
[plat-05] THE OPS SUBSTRATE DOES NOT EXIST ON THIS HOST. keeper_ait.bat is BOTH the 90-second
    supervisor relauncher AND the healthchecks.io dead-man ping gate (ops_health.py:565-566,
    972-973). `find . -name "*.sh" -o -name "*.plist"` returns ZERO. Nothing relaunches a dead
    supervisor here and the dead-man never fires. Failing loud is the right direction — but only
    if healthchecks.io is still configured, which nothing in-repo checks. scripts/verify_deadman.py
    is itself Windows-shaped and will now validate a file that is no longer the live mechanism.
[plat-06] THE SINGLETON LOCK FAILS OPEN. master.py:219-222 — `except Exception: return True`,
    and True means "you hold the lock, proceed." Any permission/disk/fcntl fault ADMITS A SECOND
    SUPERVISOR. This is the double-bot guard one layer ABOVE the one fixed on 09-14, and two
    bots is the class that produced the 07-13 position-reversal incident.
[tz-01] hour_of_day IS AN ML FEATURE STAMPED FROM THE HOST CLOCK. This box runs CEST = ET+6.
    META_FEATURES' hour_of_day has FOUR producers and THREE conventions: orchestrator.py:1579
    (live inference) and :3108 (persisted label context) use naive datetime.now() = CEST;
    meta_label.py:536 derives it from stored entry_time (ET for pre-move rows, CEST after);
    meta_label.py:435 and engine.py:634 hardcode 10. Same market hours map to {9..16} on the old
    ET box and {15..22} here. Every row written from this host is six hours off the historical
    corpus, so a model trained on the mix learns a split encoding WHICH MACHINE WROTE THE ROW.
    It reaches a live veto (orchestrator.py:1583->1601). SEVERITY, HONESTLY: the meta-labeler is
    untrained and default-off, so predict() returns None and the veto is inert — research_validity
    TODAY, live_money the moment it is armed at 100+ closes. Registering now is the point: arming
    it later on a contaminated corpus is the failure mode.

LINE-BY-LINE (data/quality.py, 185 LOC): [qual-01] the outlier anchor is LAST SEEN, not last
known good — :106-108 stores the price unconditionally, BEFORE is_valid is computed at :110, so
the anchor is poisoned by exactly the outlier it exists to catch; on the live exit path one bad
tick yields TWO consecutive `degraded` verdicts, and degraded withdraws the touch stop's
authority — a credit position's primary loss cap — for two cycles instead of one. The test file
exercises a single jump and cannot see it. [qual-02] `ts = timestamp or now` treats a falsy 0.0
epoch — a common "no time" sentinel — as "caller passed nothing", so staleness reads 0 and the
gate cannot fire on exactly the dead feed it exists for. [qual-03] validate_option_quote is
PRODUCTION-DEAD (27 lines of options validation, test-only call sites) — dead safety code that
reads as coverage, the same shape R14 found when DataQualityValidator was instantiated with zero
call sites. [qual-04] the live exit validator overrides only max_staleness_seconds from config;
max_spread_pct=0.15 and max_price_jump_pct=0.10 run on code defaults, outside R19c.

REFUTED — SIX, AND THREE OF THEM WERE MINE. A round that only confirms is not verifying.
(1) R24's "nothing writes executions.live_mid, so go-live criterion [4] is unmeasurable" is
    WRONG: two writers exist (executor.py:449-453 entry, :2778-2793 exit/W12, persisted
    state.py:890-921). The real gap is narrower — a monotonic-nonzero upsert that can never
    correct a bad first capture downward, and a conditional mirror at :2790 that silently shrinks
    the slippage sample. (2) MY scoping claim that the post-stop cooldown now resolves to 03:30
    ET and releases early: REFUTED BY PROBE — both an ET and a CEST host return BLOCKS=True,
    because inside the operating window (09:00-16:15 ET) the two calendars agree and every RTH
    exit maps above the 09:30 cutoff in both conventions. (3) MY claim that bot/scheduler.py is
    dead: it is imported at orchestrator.py:18. (4) MY claim that the APScheduler crons shift 6h:
    master.py:1710 pins timezone="US/Eastern". (5) utils/time.py and bot/scheduler.py are
    CORRECT — every helper defaults through now_et(), scheduler uses tzinfo=ET throughout.
    (6) R14's exit-input staleness gate DOES fire — portfolio.py:180 passes a real timestamp.
METHOD LIMITATION, recorded so the next round does not inherit a false map: the coverage
heuristic (in-code round markers x register mentions) UNDER-DETECTS modules whose audit trail
lives in a CONSUMER's comments or a dedicated test file. data/quality.py scored "never audited"
but was wired and tested by R14. It is a prioritiser, never proof.

COVERAGE — stated honestly, because an audit that overclaims its own reach is the thing this
ledger exists to prevent. Inventory 75 files / 43,757 LOC + 6 entry points / 1,434 LOC. DONE:
the holistic pass (all files), the host-move class in full, line-by-line on quality.py /
scheduler.py / utils/time.py, and six cross-cutting censuses (129 naive-datetime sites; 54
silent-swallow of 419 except-blocks; 40 raw .ib bypasses across 7 modules; 13 modules + 8 scripts
raw sqlite3.connect; 4 Windows shell-outs where the ledger had 3; hardcoded paths and ports).
NOT DONE: T1-b/T1-c line-by-line (broker/orders.py, account.py, capital_tiers, correlation,
selector, thompson, ops_health, telegram, gateway, pricing.py, result.py, pdt_guard,
position_sizer, diagnostics 876 LOC, dormant strategies 503 LOC); T2 — the four-wave collision
surface in orchestrator.py/executor.py; and the ENTIRE register triage (109 R24 + 78 prior; only
~15 named closed by W7-W15). R25 is a FOUNDATION: the durable artifact is the coverage matrix in
§6 of the register, so the next round inherits a map instead of a list.

FIX WAVES PROPOSED (ordered by whether the bot can survive its own failure modes on the host it
is actually running on, not by finding count): (W16) plat-01/02/05/06 — this host cannot today
recover a wedged Gateway, relaunch a dead supervisor, fire its dead-man, or refuse a second bot;
all four are live NOW. (W17) probe the top-10 leads, above all ops_health.py:608-614, where a
calendar-import failure routes to ok=True/not_rth and the dead-man ping would stay GREEN while
the bot is dead. (W18) tz-01 before the meta-labeler is ever armed, plus qual-01/02/04 — cheap,
bounded, and tz-01 is only cheap while the corpus is small. (W19) A1 then A3, one seam each.
(W20) register triage, then A2 and A4.
NOTE: the FIFTH FINDING entry above was written by the user at 21:24 DURING this audit; R25's
register carries three audit points on that change which its entry does not address — it alters
STRIKE SELECTION (strategies pick strikes by delta; delta now comes from a per-contract solved
sigma, not the chain median) under a data-correctness heading, it puts a py_vollib Newton solve
on the live path for every contract in a ~1,500-contract chain, and its
`except Exception: c.implied_vol = _chain_iv` masks BelowIntrinsicException — a genuinely
mispriced quote — as an ordinary fallback with no counter.

## 2026-09-14 - TAKE-PROFIT LADDER STUDY: NO LIVE CHANGE (engine cannot reproduce live entries)
Prompted by "could we have gotten more than 40%?" on the 09-11 closes.
HINDSIGHT (real IBKR mids, 1-min and 15-min bars): both condors kept decaying after the
40% close. SPY reached ~50% and QQQ 50.9% by the afternoon: ~+$120-134 gross on the two.
CORRECTION to my first read: execution was NOT half of SPY's gap. The 1-min mid at the
13:48 UTC fill was 3.19, and we filled at 3.21 (+0.02). I had compared against a 15-min
bar LABELLED 13:45, whose close is the 14:00 price. QQQ filled at mid +0.08 (natural +0.01).
The gap was the threshold, not the fill.
STUDY (scripts/take_profit_study.py, rule written into its docstring before any run):
8 variants, SPY/QQQ/IWM IC-only, 2022-11..2026-09, 22 non-overlapping windows, ungated
entries (live observe mode), W5 cost model. Research-only injection:
exit_policy ladder= arg + WalkForwardConfig.credit_tp_ladder / entry_dte (refused with
optimize_per_window, whose Optuna trials would score the defaults). Sanity arm
(explicit live ladder) reproduced the default run exactly in both runs.
| run | live ladder PF | flat_40 | flat_50 | flat_60 | ladder+10 | 14dte_50 | no TP | rule picks |
| entry DTE 14 (config) | 0.583 (n57) | 0.765 | 0.769 | 0.828 | 0.765 | 0.583 | 0.848 | no_take_profit |
| entry DTE 18 (live median) | 0.445 (n51) | 0.536 | 0.629 | 0.760 | 0.541 | 0.450 | 0.757 | flat_60 |
Direction is consistent: higher or no take-profit beats the ladder at both DTEs.
WHY NO CHANGE: (1) every variant LOSES money; the rule as written has no PF>1 floor
(wingk's did) so it nominates a losing arm. I did not edit the rule after seeing results.
(2) The engine does not reproduce live: from 2026-06 on it took 2 trades where live closed
~10; its entry rate is ~1/3 of live's; touch-stops 22/57 vs live's handful. (3) Live
enters at median 18 DTE (14-25 over 14 condors), not the engine's fixed 14. This is the
research re-run problem in concrete form: the engine's ENTRY model must match live before
any exit parameter can be chosen on it.
SLIPPAGE (plan step 3): only the two 09-11 closes carry W12 telemetry. There is no evidence
yet that exits near the open are worse. Revisit at ~10 instrumented closes.
DEFECT FOUND (live exit path, NOT fixed, awaiting approval): W7's price-band repricing never
engages on real IBKR rejects. ib_insync 0.9.86 classes error 202 as a WARNING, and warnings
are only logged: the "more aggressive than X / market price of Y" text is never appended to
trade.log, which is the only place parse_price_band_reject reads. On 09-11 IBKR rejected 7
SPY exits (band 3.61-3.69, market 3.09-3.17) and the bot logged band_price=None each time
and re-placed at 4.58-4.79 for 16 minutes. Harmless that day (flat market); in a fast market
it would repeat on the touch-stop path. W7's tests wrote the message into trade.log
directly, which the real library never does. Fix: capture errorEvent 201/202 per orderId in
IBKRClient and feed it to the parser; test by driving the real Wrapper.error.
ALSO: range.pkl alternates between 9-day and 30-day horizons (master.py retrain script
hardcodes horizon_days=30), reported 09-14 morning, not yet fixed. My 09-11 "horizon_days=None"
claim was wrong (wrong key name).
PROOF: tests/test_take_profit_study_injection.py, 28 executing tests (26 failed pre-change;
the other 2 are regression guards on already-correct behaviour). Suite 1606 green, type gate OK. Reports:
reports/take_profit_study/entry_dte_config/, entry_dte_18/.

## 2026-09-15 - W16 EXECUTED: the four platform gaps, one of which had already fired
R25's plat-01 MATERIALIZED before the fix landed, and the logs are the proof:
07:33:41 -> 16:09:41 CEST, `bot_restart_deferred_gateway_down` x133,
`gateway_recovery_unavailable` x134, `gateway_recovery_attempt` x0, pages x0.
TWS had been closed by accident at 07:33; the supervisor deferred for 8h36m
(~40 min of it inside RTH) and NOTHING told anyone. Book was FLAT (0 open
positions), so the cost was lost opportunity, not unmanaged exposure — but with
an open book this is the 2026-09-08 shape exactly: positions, no exit engine.
The accidental cause is what makes the fix matter MORE, not less: no automation
can un-close an app a human closed, so for this shape the PAGE IS the recovery.

[A1] `master.py:433` returned silently when the start script is absent, and that
    return sits BEFORE `self._gateway_recoveries += 1`, so the counter never
    moved and the "MANUAL ACTION NEEDED" escalation at :422-431 was unreachable
    on any host without C:\IBC\StartGateway.bat. Now pages on the unavailable
    path, sharing the existing 30-min `_last_gateway_alert` throttle so the
    2-min health loop cannot spam. The counter is deliberately NOT incremented
    (no attempt was made). EXECUTION PROOF on this host, nothing mocked: 3
    simulated cycles -> 2 pages (cycle 2 throttled, cycle 3 past 30 min),
    message naming port, duration and the action.
    W13's own suite passed green here throughout, because
    test_w13_gateway_recovery.py patches `os.path.exists -> True` at six sites.
    Its `test_missing_start_script_is_reported_not_crashed` asserted only what
    did NOT happen (no Popen, counter 0) and never asserted a page — it
    PERMITTED the silence. Both original assertions still hold; the missing one
    is now there, plus two throttle tests.
[A2] `_gateway_process_alive()` was `powershell Get-Process java` — the only one
    of the four Windows-only siblings with a FULLY silent except, and its
    unmeasured False is stamped into gateway_recovery_attempt as
    wedged_process=False: the log asserted a fact it never measured. Now psutil.
    Probed live: True (finds the TWS JVM); was False. Fail direction unchanged,
    silence removed.
[A3] `status.py:_proc_running()` — the FOURTH powershell sibling, the one the
    09-14 entry missed. Probed live against a healthy supervisor+bot: (False, 0)
    — reported the bot DOWN while it was UP; now (True, 2). The 09-14 fix went
    to one CALL SITE, not to the IDIOM, so the scan existed in four places. It
    now lives once, in `ait/utils/process.py`, shared by master.py and
    status.py. A failed scan returns None, never [] — "I could not look" and "I
    looked and found nothing" are different facts, and collapsing them is how
    the guard became a silent no-op in the first place.
[A4] `_acquire_singleton_lock` ended `except Exception: return True`, and True
    means "you hold the lock, proceed" — a permission/disk/fcntl fault ADMITTED
    A SECOND SUPERVISOR, one layer above the guard fixed on 09-14. Two bots is
    the 07-13 triple-fill/position-reversal class. Now fails CLOSED and pages,
    following W15's pattern (log _fail_closed, throttled page, inner try so a
    paging failure cannot flip the gate open). The old comment argued "lock
    plumbing must never keep the bot down"; W15 settled that trade — a stopped
    bot is visible and recoverable, a second bot corrupts the book.
[B1] THE OPS SUBSTRATE NOW EXISTS OFF WINDOWS. `find . -name "*.sh" -o -name
    "*.plist"` returned ZERO: keeper_ait.bat is BOTH the 90s relauncher AND the
    dead-man ping gate, and on macOS neither existed. The supervisor was found
    running out of a VS Code integrated terminal (zsh <- Code Helper <- Visual
    Studio Code) — quitting the editor would have killed the trading bot. Added
    ops/keeper_ait.sh + a launchd template + installer, ADDITIVE: keeper_ait.bat
    is byte-stable because test_r16_tail_fixes::TestKeeperBat and
    test_w6_ops_surfaces pin its exact text. Reuses what already exists rather
    than reimplementing it — `_acquire_singleton_lock` is the real double-launch
    guard (the keeper's check is advisory), and `_ROTATE_TARGETS` already caps
    keeper.log. Paths are derived, never baked: the .bat hardcodes another
    user's home, which is the mistake the template exists to avoid. NOT
    INSTALLED — starting a process that manages the bot is the user's call.
[B2] scripts/verify_deadman.py hardcoded keeper_ait.bat and asserted `curl.exe`,
    so on macOS it validated a file that never runs here — reassurance about the
    wrong mechanism. Now platform-selects the live keeper. It currently reports:
    keeper wired to ping TRUE, ping gated on bot TRUE, liveness ALIVE
    [heartbeat_fresh], deadman file MISSING -> STATUS: NOT ARMED.
TWO SELF-CORRECTIONS, recorded because the register is only worth what its
honesty is worth. (1) My first shared-helper design was WRONG: I claimed
whole-argv TOKEN matching was precise, and a test I wrote caught that it is not
— in `grep -r ait.main src/` the token IS its own element. Only `-m` adjacency
distinguishes a bot from a mention; the real bot is spawned at master.py:585 as
[sys.executable, "-m", "ait.main", "--paper"]. I had also left a docstring still
asserting the old claim — the same stale-comment defect R25 registered against
strategies/base.py. (2) That change failed
test_r16_ops_fixes::TestScanForeignBotPidsRealBody, whose fixture spawned
`python -c ... ait.main` — a STRAY token, i.e. the fixture enshrined the
false-positive shape. Amended to the real spawn shape and a negative test added
pinning that a mere mention is not a bot. An existing assertion was changed, so
it is said out loud here rather than buried in a diff.
PROOF: every new test proven to fail against pre-fix source (git stash of the
source files only) — 3 for A1, 10 for A2/A3/A4, 4 for B2 — then pass restored.
Suite 1612 passed / 43 skipped / 1 pre-existing local-.env failure
(test_r19_config_authority, unchanged and gitignored), was 1583. typecheck OK,
byte-compile OK, SMOKE PASS 13.
LESSON (process): my FIRST full-suite run reported 3 failures, two of which I
caused by editing master.py WHILE the suite ran — test_r16_tail_fixes uses
inspect.getsource(), which reads the file from disk at line offsets captured at
import time, so the offsets went stale mid-run. Not a defect, not pollution: a
full-suite result is only meaningful if the tree holds still underneath it. The
clean re-run is the one quoted above.
STILL OPEN, USER-SIDE: (U1) the dead-man has NEVER been armed —
data/deadman_url.txt does not exist, unarmed since 07-09, and the keeper is half
a safety net without it; (2) ops/install_launchd.sh is written but not run;
(3) three false `gateway_recovery_unavailable` lines at 20:47:29 in
logs/orchestrator.log were planted by my own A1 execution probe — `_log`
suppresses its file sink only under pytest (R16 #56), not for ad-hoc probes.
Removing them was correctly refused as audit-log tampering; the call is the
user's.

## 2026-09-15 - W16b CORRECTION: the POSIX keeper, launchd and the dead-man are RETIRED
The W16 entry above is now PARTLY WRONG and this corrects it rather than
rewriting it. W16's [A1]-[A4] and [B2] stand and are DEPLOYED (supervisor pid
41807 started 22:33:24, after the 21:40:59 commit — A1's page is live). [B1] —
ops/keeper_ait.sh, ops/install_launchd.sh, ops/com.ait.keeper.plist.template —
has been DELETED, and the healthchecks.io dead-man with it.

WHY, and the mistake underneath it. I built B1 on an unexamined premise. The
keeper exists on Windows for a reason its own header states plainly
(keeper_ait.bat:4-7): it "survives whatever is killing python on this machine
on a ~30-60 min cycle. This is a band-aid over that machine-level issue, not a
fix for it." That pathology is WINDOWS-BOX-SPECIFIC. Nothing establishes it
happens on the Mac — the supervisor ran 18h41m straight here without dying. I
ported the band-aid without checking whether this machine has the wound. R24's
own critic list warns about exactly this class (machinery carried forward
because it exists, not because it earns its place).

THE OPERATOR MODEL I HAD WRONG. W16 was scoped from "the bot must survive
unattended", which I read as "needs a keeper + a login-time daemon". The actual
model, stated 2026-09-15: "I maintain control on when I start the bot. Once
started, it should be kept running, until I shut it down." Start and stop are
DELIBERATE HUMAN ACTS. The keeper cannot tell "crashed" from "the operator
stopped it" — both are just an absent process — so with it installed, Ctrl-C
would have been undone within one ~90s loop and the bot could not have been
turned off at all. That is a direct functional conflict with the operating
model, not a tuning detail.

THE DEAD-MAN, AND WHY U1 IS NOW A DECISION NOT A GAP. healthchecks.io was armed
(Period 5m / Grace 15m, per verify_deadman.py's own docstring) and then found to
assume CONTINUOUS uptime: the keeper pings every ~90s including outside RTH
(bot_liveness returns ok=True [not_rth]), so any planned shutdown is
indistinguishable from a crash and alarms 20 min later. PLAN.md:33 already
records that this bot's machines are NOT always-on ("box on by 09:25 ET on
trading days (Start-menu shutdowns at ~22:00 are fine)"), so the configuration I
recommended would have false-alarmed every night — and a nightly alert you learn
to ignore is worse than no alert. The operator's call: delete the check, the
layer is premature. U1 ("dead-man UNARMED since 07-09") should therefore no
longer be read as an oversight to close; it is a DEFERRED DECISION. Revisit it
if and when the box becomes genuinely always-on or unattended for long stretches
— at which point the clean design (no keeper) is: bot pings on its own health
cycle, and master.py's existing SIGINT/SIGTERM graceful_shutdown pauses the check
as its last act, so Ctrl-C is indistinguishable from intent. That needs a
healthchecks.io management API key (there is no /pause ping endpoint).

ALSO FOUND AND NOT WORTH FIXING: the LaunchAgent could not run at all on this
host — `Operation not permitted` on getcwd and on the script itself (exit 126, 6
throttled retries before I unloaded it). Cause is macOS TCC: ~/Documents is a
protected folder and a launchd-spawned job inherits none of the grant the
Terminal/VS Code process tree already has, and a headless agent cannot show the
consent prompt. The fix would have been Full Disk Access for /bin/bash (coarse,
system-wide) or relocating the repo out of ~/Documents. Both were made moot by
the decision above — worth recording because any future attempt to daemonise
anything in this repo will hit the same wall.
WHAT SURVIVES AND IS LIVE: A1 (gateway escalation — the 8h36m fix), A2/A3
(psutil process scans, one shared idiom in ait/utils/process.py), A4 (singleton
lock fails closed), B2 (verify_deadman platform selection). keeper_ait.bat and
verify_deadman.py remain valid for the WINDOWS host, which is still primary
alongside the Mac; on macOS the verifier now correctly reports "not plumbed",
true by decision rather than oversight. tests/test_w16_platform_gaps.py keeps a
note where TestPosixKeeper used to be, so a later round does not "restore
missing coverage" by resurrecting retired machinery.
HOUSEKEEPING: the three false `gateway_recovery_unavailable` lines my own A1
execution probe planted in logs/orchestrator.log at 20:47:29 were REMOVED with
operator authorization (backup kept outside the repo). The underlying gap is
real and unfixed: `_log` suppresses its file sink only under pytest (R16 #56),
so any ad-hoc probe still writes to the production forensic log. Registered, not
fixed.

## 2026-09-15 - W17 BATCH 1: five R25 leads probed — 4 VERIFIED, 1 PARTIAL, 2 NEW (one of them mine)
R25 left 49 single-lens leads. W17 converts them to verdicts in consequence order, stopping after
each batch so the tail can be re-ranked. Register: reports/w17_probes_batch1_20260915.md; the five
§5 entries in the R25 register are stamped inline with their verdicts so it stops presenting
settled leads as open. Method unchanged: two-lens, and every probe drove the REAL function with
mocks only at the external edge. Probe hygiene held — PYTEST_CURRENT_TEST set before any
supervisor call, and 0 probe lines reached logs/orchestrator.log (the hazard that planted three
false lines during W16).

RE-RANKED BEFORE PROBING, because W16b changed the board: retiring the dead-man made Telegram the
ONLY alerting channel, which promoted the alert path itself from lead 7 to rank 1 — and A1, shipped
hours earlier, rides exactly that path.

[w17-01] `_alert()` IS SINGLE-SHOT AND NOTHING COUNTS ITS FAILURES — VERIFIED. Probe: 1 urlopen
    attempt (a retrying impl would be >1), returns False, and NO TELEGRAM_DEAD marker after 7
    consecutive failures. orchestrator.py:4278-4318 does 3 attempts with backoff and trips a
    dead-channel marker at 5; the supervisor's 21 call sites — including STATE DB BACKUP FAILED
    (:1177) and GO-LIVE SCORECARD FAILED (:1480) — have none of it.
[w17-01b] AND A1 PROPAGATED A THROTTLE DEFECT — VERIFIED, NEW, MINE. Both gateway-escalation sites
    set `_last_gateway_alert = now` BEFORE calling _alert() and discarded its bool, so a FAILED
    send still consumed the 30-minute throttle. Probe: after one dropped page, 0 further attempts
    in the next 20 minutes. One dropped packet bought 30 minutes of silence on the page that
    exists BECAUSE of an 8h36m silent outage. The pre-existing sibling at :422-431 had the
    identical ordering — I copied a defect faithfully rather than inventing one. FIXED AT BOTH
    SITES (fix the class, not the instance): the throttle is now consumed only on a delivered
    page. Re-probed: 3 attempts over 20 min when Telegram drops, 1 when it delivers.
[w17-02] A CALENDAR FAULT MAKES A DEAD BOT READ HEALTHY — VERIFIED, CONSEQUENCE RESTATED. Same
    6-hour-stale heartbeat: with a working calendar `ok=False state=heartbeat_stale`; with the
    calendar raising, `ok=True state=not_rth`. Driven end-to-end through the real dashboard panel:
    liveness_panel() returns 'stale' normally and 'ok' under the fault. R25 claimed the victim was
    the dead-man ping — that consumer no longer exists after W16b. The live victim is
    dashboard/app.py:475. The lead was right about the mechanism and wrong about the blast radius;
    both are recorded. Fix direction: an unavailable calendar is UNKNOWN, not CLOSED — return a
    third state and render NOT WIRED, the pattern read_channel_state already uses at :924-971.
[w17-03] combo() VALIDATES NOTHING, BUT THE EXIT PATH CANNOT EXPLOIT IT — PARTIAL, and R25
    OVERSTATED THIS. Probe: every leg shape the entry gate rejects is built silently — action
    'HOLD', conId 0, ratio 99, ratio None, lowercase 'buy'. That half is VERIFIED. But the exit
    path's legs are well-formed BY CONSTRUCTION: :3390-3392 aborts on a failed qualify (conId
    cannot be 0), the action is rebuilt as a literal, and ratio always resolves to the default.
    No reachable path produces a malformed leg. The real risk is that the invariant rests on three
    unrelated implementation details with nothing enforcing it, on a builder with zero tests.
[w17-03b] LEG `ratio` IS DROPPED ON PERSISTENCE — VERIFIED, NEW. executor.py:367-375 writes only
    strike/right/action/expiry, so every exit rebuilds at ratio=1 regardless of entry. Inert today
    (all strategies 1:1); becomes a silent structure change the moment a ratio spread is enabled —
    the exit would close a different shape than the entry opened.
[w17-04] THE DASHBOARD'S "WIN RATE" DIVIDES CLOSE-DAY WINS BY ENTRY-DAY ENTRIES — VERIFIED, with a
    real instance in this DB. trades_taken increments at ENTRY (:2620); trades_won/lost at CLOSE
    (:788-791); app.py:640-642 divides one by the other. Both real daily_stats rows —
    2026-04-23 and 04-24, taken=5 won=0 lost=0 — render as "Win Rate 0%" on days with FIVE entries
    and ZERO losses. Boundaries from the same expression: taken=0/won=2 shows "N/A" (a day of pure
    wins shows nothing), taken=1/won=3 shows "300%". Compounded by the real-close asymmetry: the
    drain path consults is_real_close (:688-691), the fill-callback path (:785-797) does not, and
    the dashboard cannot filter what daily_stats has already aggregated.
[w17-05] THE PARTIAL-EXIT ORDER CARRIES NO orderRef — VERIFIED. OrderBuilder.market() returns a
    bare MarketOrder; the tree has exactly five orderRef assignments (executor 570/798,
    orchestrator 3046/3087/3674) and :2849 is not among them. reconciler.py:604-609 documents the
    invariant "every entry/exit order has carried its trade_id in orderRef since R17" and falls
    back to the symbol/leg heuristic ONLY for pre-R17 orders — so this emits a post-R17 order that
    reads as pre-R17, triggering the matcher R19 replaced because it could cancel a different
    trade's working order, "including a live EXIT combo". LATENT: the partial-exit ladder is dead
    at 1 contract, so it wakes on funding.
COUNTS: 5 probed -> 4 VERIFIED, 1 PARTIAL, 0 REFUTED, 2 NEW findings surfaced while probing. Zero
refutations is a smaller sample, not a better hit rate — batch 1 was deliberately the end of the
list where claims had already survived the most scrutiny. R24 refuted 28 of 137 at this stage and
R25 refuted 6 of its own, three of them mine.
PROOF: w17-01b's fix has 3 tests, 2 proven to fail against pre-fix source (the third passes both
ways ON PURPOSE — it guards against over-correcting into no throttle at all). Suite 1609 passed /
43 skipped / 1 pre-existing local-.env failure. typecheck OK.
PROPOSED FIX ORDER: (now, done) w17-01b. (W18) w17-01 give supervisor alerts retry + dead-channel
accounting — Telegram is the only channel left; w17-02 render a calendar fault as NOT WIRED rather
than green. (W19) w17-04 compute the Overview metrics from `trades` with _REAL_CLOSE_SQL like every
other panel already does; then w17-03/03b/05, which are latent today and wake on funding or on a
strategy change.
BATCH 2 CANDIDATES: R25 §5 leads 5, 6, 8, 9, 10 and the 39 area leads. Ordering should be revisited
against what this batch changed — anything whose stated consequence depended on the dead-man needs
its blast radius restated the way w17-02 did.

## 2026-09-16 - W17 BATCH 2: six probes, all VERIFIED — one consequence corrected downward
Register: reports/w17_probes_batch2_20260916.md. Pre-probe findings and the batch-3 census were
committed BEFORE probing (a20f370, b33fa83) because planning artefacts do not survive a session —
the plan file lives outside the repo. 6 probed, 6 VERIFIED, 0 refuted. Zero refutations is a
PRE-SELECTED SAMPLE, not a hit rate: 1(b) was refuted and three register claims corrected during
planning, i.e. before this table existed. No fixes applied (register-don't-fix). No live state
touched: fixture DBs throughout, data/ait_state.db never written, bot pid 41807 undisturbed.

[w17-06] A SOLVED PER-CONTRACT SIGMA SELECTS A DIFFERENT STRIKE. Real QQQ chain from
    historical.db (117 puts, S~682, dte=33) through the REAL _find_strike_by_delta: 4 of 5 delta
    targets pick a different strike — 20, 15, 5 and 5 points — INCLUDING 0.20, the value
    iron_condor.py:169,177 actually uses. df704d7 changed which strikes the live bot sells; that
    is a strategy change shipped under a data-correctness heading. NOT PROVEN: the tolerance=0.07
    trade/no-trade flip did not fire in this slice and must not be cited.
[w17-07] A BELOW-INTRINSIC QUOTE IS LAUNDERED INTO A HEALTHY-LOOKING CONTRACT. K=60 call, S=100,
    intrinsic 40, mid 5.0 — py_vollib raises BelowIntrinsicException ("this price is impossible")
    and _calculate_greeks writes implied_vol=0.24 (chain median) and delta=0.9999 with ZERO log
    lines. iv_solved_from_price does not increment (it counts successes only). Honest scoping: a
    ~1.0 delta will not win a 0.20 argmin, so selection risk is low. The real damage is that
    degraded_greeks (iron_condor.py:161) CANNOT FIRE once backfill has made every delta non-zero —
    the gate designed to catch a degraded chain is disarmed by the degradation.
[w17-08] verify_can_trade FAILS OPEN — AND I CORRECTED MY OWN RE-SCOPING. Patched at
    qualifyContractsAsync (nothing reached a broker): ConnectionError/TimeoutError/RuntimeError all
    return True with placeOrder never called. But I had called this "a pre-trade safety probe that
    fails open", implying trading proceeds unguarded. It does not: orchestrator.py:2117 is the sole
    caller, and BOTH of ITS callers (:513, :900) DISCARD the return value. What can_trade=False
    actually does is send a Telegram alert — so the real consequence is that THE READ-ONLY ALARM
    NEVER FIRES, returning the operator to the silent failure the probe exists to prevent.
    Severity ops_reporting, NOT live_money. R25's framing and my correction of it were both wrong.
[w17-09] THE GO-LIVE SLIPPAGE GATE MEASURES THE WRONG POPULATION (3 sub-claims, all verified).
    Fixture: entry fill slipping 2%, EXIT fill slipping 30%, one row with live_mid=0. Result:
    3 rows written, slip_n=2 (the dropped row vanished uncounted), median 16.0% — the exit fill is
    in the pool and is divided by the ENTRY credit, and 16% FAILS the 8% gate on the strength of a
    fill the criterion claims not to measure. At n==1: available=True, passed=True, rendered
    "median 0.5% of credit over trailing 1 fills". A SINGLE FILL AUTHORISES THE GATE; UNAVAILABLE
    fires only at n==0.
[w17-10] THE RESTATE SCRIPT WOULD ZERO DATA THE CURRENT WRITER PRODUCES. Ran the REAL script
    main() dry against a fixture with one legitimate single-leg fill. It printed that row under
    "sample rows currently feeding the gate" and then "would clear context on 1 leg rows". Its
    predicate is con_id != BAG_CON_ID (true of EVERY single-leg order) while the writer branches on
    is_combo (executor.py:1530-1537) and stores live_mid deliberately on that shape. Compounding:
    state.py:913-918 never restores a zeroed value and go_live.py:327's WHERE live_mid > 0 then
    hides it while slip_n reports only survivors — destruction is invisible at the gate BY
    CONSTRUCTION. This already happened once: three single-leg fills zeroed 2026-08-10.
[w17-11] _capital_base() REVERTS TO A 3.7x WRONG DENOMINATOR, SILENTLY. With the authority import
    forced to fail, all three copies return 196,000.00 against a live 723,071.22 — 73% off — behind
    a bare unlogged except. Also confirmed the three copies are byte-IDENTICAL to each other, so
    R25's register (which never claimed otherwise) was right and my planning note was wrong. The
    divergence is copies-vs-authority: they lack the live-NLV tier. app.py:1004 raises this to the
    12th power for the annualised-return metric.
PROPOSED FIX ORDER: (W18) w17-09 + w17-10 together — they are ONE LOOP: the gate authorising real
capital measures the wrong population, and an untested script can silently destroy its inputs; plus
w17-11. (W19) w17-06 + w17-07 — first DECIDE whether per-contract solved IV is wanted for strike
selection (it may well be an improvement), then make the choice explicit and instrument the solver's
failures; plus w17-08's failure counting.
BATCH 3 IS PRE-STAGED: reports/w17_batch3_failopen_census_20260916.md. R25 registered "54
silent-swallow sites"; the real scale is 355 except clauses -> 177 candidates -> 120 traced
(38 decision-affecting / 51 telemetry / 31 cleanup), with a 12-site ranked shortlist. Four of the
twelve are CALENDAR FAULTS — a single NYSE-calendar failure would concurrently restart the live bot
mid-session, open the pre-event blackout, and disable both event-driven exits. Probe that as a
CLASS, not five instances.

## 2026-09-16 - W18 EXECUTED: Gateway port unblock + 4 verified findings fixed (and a W17 miss owned)
Operator moved TWS -> IB Gateway. That created no defect; it exposed a stale line. Register:
reports/w17_probes_batch2_20260916.md (the findings), this entry (the fixes).

[0] THE BLOCKER. Port 4002 was OPEN (Gateway paper), 7497 closed, and .env still said
    IBKR_PORT=7497 — so the supervisor (pid 41807) logged bot_restart_deferred_gateway_down every
    2 min for 9h47m. Both the supervisor probe and the bot's own connection read that variable,
    and master.py:603 spawns the child with env={**os.environ}, so the child inherits the
    SUPERVISOR's env: editing .env does nothing until the supervisor itself restarts. Fixed .env
    -> 4002 (gitignored, local). The bot had ALREADY been down since ~05:00 for an unrelated
    reason (TWS disconnect -> SIGTERM), so this outage was not caused by the switch.
[0b] AND MADE IT SELF-DIAGNOSING. New _port_hint() scans the other known IB ports (4001/4002/
    7496/7497 — the set .env's own comment documents) and splices a hint into the alert and the
    log: "nothing is on 7497, but 4002 (IB Gateway paper) IS open — check IBKR_PORT in .env".
    Verified against live machine state. The mismatch was measurable for ten hours and nothing
    looked.
[0c] A W17 MISS, OWNED. W17's w17-01b fixed the "consume the throttle only on a DELIVERED page"
    ordering and its commit claimed the CLASS. There are FOUR sites sharing _last_gateway_alert;
    it fixed two (both inside _maybe_recover_gateway) and MISSED two — :572 and :771 — and :771 is
    the one that had been firing every two minutes for ten hours. The one that actually mattered
    was the one I missed. Both now gated on delivery; a new test pins all four so the class cannot
    regress again. This is the third time this session one of my own claims was wrong and the
    two-lens discipline caught it.
[1] w17-08 THE READ-ONLY ALARM NO LONGER FAILS OPEN. verify_can_trade returned True after failing
    to place its probe. It now returns None for UNKNOWN — the sentinel this repo already uses
    (RESTRICTED_UNREADABLE, get_all_open_trades) — distinct from True (accepted) and False
    (read-only), and _check_trading_enabled pages on UNKNOWN with its own throttle key. The return
    value stays unused, correctly: both callers discard it, and the function's job is the ALERT.
    PROMOTED by the Gateway switch: READ-ONLY login is the failure Gateway actually has, and it
    recurs on Gateway's daily restart.
[2] w17-09 THE GO-LIVE SLIPPAGE GATE IS NOW HONEST. It is PINNED as median ENTRY slip, but the
    executions table has no lifecycle marker (`side` is BOT/SLD direction, not entry/exit), so the
    pinned statistic CANNOT be computed. Per this module's own promise — "any criterion whose data
    cannot support the PINNED definition is returned UNAVAILABLE with a reason rather than
    substituted by a differently-defined number" — criterion [4] is now UNAVAILABLE. The all-fills
    median is still rendered as explicitly-FYI so no visibility is lost, and the coverage hole is
    reported the way criteria [1] and [3] already do: "dropped: 1 with no matching trade, 1 with no
    live_mid of 4 fills". n==1 can no longer pass a gate.
    VISIBLE BEHAVIOUR CHANGE, NOT A REGRESSION: the scorecard will now show [4] as UNAVAILABLE
    rather than a number. That is the point — it was showing a differently-defined number.
    Chose UNAVAILABLE over adding a lifecycle column (operator decision): arch-02 found there is
    no migration engine, so a schema change carries its own risk.
[3] w17-10 scripts/restate_r16_executions.py DELETED. Its predicate (con_id != BAG_CON_ID) is true
    of EVERY single-leg order while the writer branches on is_combo, so it would zero fill-quality
    context the current writer legitimately produces — W17 batch 2 ran it for real and it printed a
    legitimate row under "sample rows currently feeding the gate" then declared it would clear it.
    One-off repair, applied on the Windows box 2026-08-10, never applied here, zero tests. Ledger
    records what it did; git retains the file. The tested sibling restate_r16_dailystats is kept.
[4] w17-11 NO MORE SILENT WRONG DENOMINATOR. The three local _capital_base() copies are gone,
    replaced by a direct import of runtime_env.capital_base (which already resolves env -> live NLV
    -> default, and imports only os/pathlib so it costs no import weight). Each copy had wrapped
    the authority in a bare unlogged except and fallen back to a SECOND hardcoded 196000 —
    measured at 196,000 against a live 723,071, then raised to the 12th power for the annualised
    return. If the authority is ever unreachable the module now fails LOUDLY at import instead of
    rendering a plausible lie. Also fixed the dead key: app.py queried bot_state['account_value'],
    NEVER WRITTEN anywhere in src/; it now reads 'last_net_liquidation', which orchestrator.py:1180
    actually writes.
TESTS AMENDED, SAID OUT LOUD: two tests in test_w3_scorecard_truth.py pinned the old slippage
rendering. W3's actual property — a PERCENT median, never a dollar mean — is unchanged and still
asserted; only `passed` moved, because a criterion that cannot be computed as pinned must not pass
a gate. A third test was added for the drop reporting. Writing the render also caught two bugs in
my own first attempt: render() already prepends "UNAVAILABLE — " (I double-prefixed it) and
DISCARDS the suffix for an unavailable criterion (so the trend clause had to be folded into the
text, or the operator would have silently lost a number).
PROOF: every fix has tests proven to fail against pre-fix source (6 for item 0, 6 for items 1+4),
then pass restored. NOT YET DEPLOYED — .env's new port needs the SUPERVISOR restarted, not just the
bot child, and that restart is the operator's call.

## 2026-09-16 - W19 DECISION: per-contract solved IV is KEPT for strike selection (chosen, not inherited)
THE GAP THIS CLOSES IS PROVENANCE, NOT BEHAVIOUR. df704d7 ("Fix implied_vol never getting a real
backfilled value") changed WHICH STRIKES THE BOT SELLS and nothing in this ledger said anyone chose
that. W17 batch 2 measured it against a real QQQ chain through the real _find_strike_by_delta: 4 of
5 delta targets select a DIFFERENT strike, by up to 20 points, including 0.20 — the value
iron_condor.py:169,177 actually uses. It shipped under a data-correctness heading. An unchosen
improvement is harder to notice than a defect.

DECISION: KEEP IT. The evidence, measured, not argued:
* The codebase already documents the mechanism. options_chain.py (finding A6): "when a contract
  lacks IV, use the chain's MEDIAN observed IV instead of a flat 30% — a wrong sigma shifts the
  computed delta, and strategies pick strikes BY delta." The trajectory was ALREADY
  flat-0.30 -> chain-median. Per-contract solved IV is the next step on that same path.
* How wrong is the median? Against real per-strike IV in data/historical.db
  (option_spread_samples), restricted to the strikes a condor actually sells (0.08-0.32 delta):
      QQQ  5d  median IV 0.293  |dIV| 0.0498  =  2.9 strike steps of delta error
      QQQ 12d  median IV 0.303  |dIV| 0.0747  =  6.4
      QQQ 19d  median IV 0.261  |dIV| 0.0512  =  5.6
      QQQ 33d  median IV 0.236  |dIV| 0.0446  =  6.6
      QQQ 37d  median IV 0.336  |dIV| 0.0804  =  9.7
  MEDIAN 6.4 STRIKE STEPS of systematic delta error, on every scan, at exactly the traded strikes.
  The solved sigma removes it.
LIMIT OF THAT EVIDENCE, STATED SO IT IS NOT OVERSOLD: backfill only fires when IBKR did NOT supply
a delta, so there is no ground-truth delta to check the result against. This measures the ERROR THE
MEDIAN INTRODUCES under the standard assumption that the market prices each strike's own vol. It is
NOT a backtest of outcomes and must not be cited as one. If that assumption is ever to be tested,
it needs a study, not this entry.

[w17-07 FIXED, part 1] SOLVER FAILURES ARE NOW COUNTABLE. iv_solved_from_price was incremented
INSIDE the try, after success — a success-only counter with no attempt counter, so failures could
not even be derived by subtraction, and the except was bare and silent. Now counts attempted /
solved / failed_arbitrage / failed_other, and logs a WARNING naming contracts whose quote sits
OUTSIDE THE NO-ARBITRAGE BAND (BelowIntrinsic / AboveMaximum). That is a BAD QUOTE — stale, crossed
or genuinely mispriced — and is a data-quality signal in its own right, not a solver hiccup.

[w17-07 FIXED, part 2] THE degraded_greeks GATE IS RE-ARMED. It rejected only when NO delta was
non-zero — and the 09-14 backfill gives EVERY contract a non-zero delta, so the gate became
unreachable: disarmed by the very degradation it exists to catch. Its own comment names the
incident it was built for ("the 07-07 SPY condor fill its short call at 0.49 DELTA (designed
0.20)"). The genuinely degraded case is now explicit: when NO contract carries a usable IV,
_chain_iv falls back to a hardcoded 0.30 and every backfilled delta is a guess. OptionsChain
.iv_is_guessed carries that, filter_by_delta and filter_liquid PROPAGATE it (a filtered chain that
dropped the flag would silently disarm the gate again — the same fragility atm_iv already has), and
iron_condor rejects on it BEFORE the legacy check, which is retained.
DELIBERATELY NARROW: not "any backfill happened". A few solved contracts per expiry is routine —
live logs show 2-55 — and rejecting on that would refuse normal chains. BEHAVIOUR CHANGE ON A
TRADING GATE: it can now reject entries that currently pass. That is the intent; those are exactly
the entries the 07-07 incident produced.

[REGISTERED, NOT FIXED — its own wave] THE PRICING CONSTANTS. r = 0.05 is hardcoded at THREE sites
(options_chain.py, backtesting/engine.py x2) with NO config home — a clean R19c violation. The
solve also uses py_vollib.black_scholes (no dividend yield) while IBKR's impliedVol uses a dividend
schedule, so _chain_iv takes a median over TWO DIFFERENT MODELS. Measured at a ~0.20-delta short
put: dividend yield shifts delta 0.0010 (QQQ) to 0.0027 (IWM); r at 3% vs the hardcoded 5% shifts
it 0.0044; one $1 strike step is 0.0048 — so roughly ONE strike step, against the 6.4 the decision
above addresses. Deferred by operator decision: two of the three sites are in the BACKTEST ENGINE,
and changing them changes backtest results, which breaks comparability with every pre-registered
study this repo has run. That needs its own wave WITH a re-run plan, not a side effect of this one.
PROOF: 7 tests, 5 proven to fail against pre-fix source; the 2 that pass both ways are deliberate
regression guards (the fallback still yields a usable IV; the legacy no-delta check is retained).

## 2026-09-16 - W19 DEPLOYED + WATCH LIST FOR THE OPEN
This entry closes a gap: the decision/fix record above was written BEFORE the restart, and
everything after — the deploy itself and what to watch — was only said in chat, not committed. The
ledger is the persistent record; chat is not. Same class of mistake as the plan-file durability gap
found earlier this session, now made against PLAN.md itself.

DEPLOY: graceful SIGTERM to supervisor pid 51749's predecessor; took ~45s this time (vs ~1s on the
prior W18 restart) because the bot child had a live IBKR connection to close — waited rather than
force-killing a trading process; log confirms shutdown_signal -> bot_stopping -> web_services_stopped
end to end. Relaunched detached (nohup, PPID reparents to launchd pid 1, survives this session).
Deployed commit 40e677a (matches `git log -1` on the running tree). Gates before deploy: suite 1630
passed / 43 skipped / 1 pre-existing local-.env failure, typecheck OK, SMOKE PASS 13.
VERIFIED POST-RESTART: trading_verified logged (the read-only probe passed on Gateway); 0
deferred/error/CRITICAL lines in the first 5 min; OptionsChain.iv_is_guessed confirmed present on
the running code (the W19 gate is live, not just committed).

MANUAL POSITIONS — CONFIRMED BY OPERATOR, NOT A BOT DEFECT. Shutdown logged updatePortfolio for two
QQQ option legs (strike 712, expiries 2026-09-18 and 2026-10-16) while open_positions read 0 rows.
Operator confirms: entered MANUALLY, not through the bot. open_positions still reads 0 and
data/HALT_UNTRACKED has not fired, so nothing is currently frozen — but this is exactly the
broker-vs-DB shape the untracked-position tripwire (reconciler.py:1078-1088) exists to catch, and
the reconciler runs every cycle against the LIVE broker state, so this is not a one-time check.
WATCH: if reconcile() ever decides these ARE untracked (vs. correctly recognizing them as
operator-owned and out of scope), it freezes ALL new entries via HALT_UNTRACKED until a human
deletes the file. Confirmed benign right now; not verified to STAY benign through the session.

WATCH LIST FOR THE 2026-09-16 OPEN (09:30 ET, first RTH session on Gateway with W18+W19 live):
1. Manual QQQ positions above — does data/HALT_UNTRACKED fire once the reconciler runs live?
2. degraded_greeks rejections (W19) — expected RARE; means a chain arrived with NO usable IV
   anywhere (every delta a flat-0.30 guess). This is the gate working, not a new fault — but if it
   fires on every scan rather than occasionally, that says something is wrong with the FEED, not
   the gate.
3. iv_solve_quote_outside_arbitrage_bounds / chain_iv_is_a_guess (W19) — new log events; first time
   they will appear under real market data rather than test fixtures.
4. Gateway read-only (W18) — first live cycle on Gateway; if it restarts read-only, confirm the new
   CANNOT VERIFY TRADING alert actually pages rather than assuming the fix works because it passed
   in tests.
5. Standard post-deploy check per this repo's own deploy discipline: fresh heartbeat,
   scan_symbol_timing events, no LOOP IMPAIRED alert.

## 2026-09-16 - BACKLOG CORRECTION: R25's §7 wave table is stale; consolidated here
User asked directly whether PLAN.md now reflects ALL future work the audit surfaced. Checked
rather than assumed the answer, and the honest answer was no — worse, the one place that claimed
to say what's next (reports/full_audit_r25_20260914.md §7) is now WRONG: it proposed W18=tz-01+
qual-01/02/04, W19=arch-01+arch-03, W20=triage+arch-02+arch-04. Real events consumed those wave
numbers for the Gateway-port emergency (W18) and the strike-selection decision (W19) instead.
Anyone reading §7 today would look for undone work under a label now occupied by something else.
This entry supersedes it — R25 §7 is left in place as history, not edited (this ledger is
append-only), but is no longer the source of truth for what remains.

VERIFIED UNTOUCHED SINCE R25 (2026-09-14), zero PLAN.md mentions since:
* tz-01 — hour_of_day (an ML feature) stamped from the host clock; CEST vs ET mismatch on this
  machine. Cheap only while the meta-labeler corpus stays small (still untrained/default-off).
* qual-01/02/04 (data/quality.py) — the outlier anchor is poisoned by the outlier it exists to
  catch (stores the bad price BEFORE checking validity); `ts = timestamp or now` reads a real
  epoch-0 sentinel as "no timestamp given"; the live exit validator's spread/jump thresholds are
  code defaults, not config, against the R19c policy.
* arch-01..04 — the four ARCHITECTURAL findings from R25's holistic pass: no broker boundary
  (.ib escape hatch, six independent copies of the reqMktData dance — the class behind BOTH the
  09-14 snapshot bug and the W19 options_chain trio); no state boundary (13 modules + 8 scripts
  open the DB directly, no schema_version anywhere); cyclic imports (bot<->execution/risk/
  reporting — the class behind the 09-14 Streamlit crash); orchestrator.py as a god object
  (4,336 lines, 89 except-blocks, four waves collided in it in eight days). These are STRUCTURAL
  and explain why instance-fixes keep recurring; none has been started.

VERIFIED UNTOUCHED FROM R25's "remaining leads" AREA LIST (39 leads, by area):
* Fail-open census (batch 3) — 355 except clauses -> 177 candidates -> 120 traced, 12-site ranked
  shortlist in reports/w17_batch3_failopen_census_20260916.md. CENSUS DONE, ZERO SITES PROBED.
  Four of the twelve share one dependency (a calendar fault) — flagged to probe as a class.
* Broker (ibkr_client.py) — FX cache has no TTL in a days-long process; connect()'s clientId
  fallback loop, place_order, cancel_order are untested; the local .env account mismatch (demoted
  to hygiene after a W17 probe showed the account IS enforced at connect, but the stale
  comment/fixture was never actually fixed).
* Orders/contracts dead surface — 5 of 9 OrderBuilder methods, 4 of 6 ContractBuilder methods,
  never called in production.
* Capital chain — capital_tiers.py:102-108 currency-agnostic thresholds (a CAD figure sits in the
  comment next to USD-compared numbers); :140 silently falls back to symbols[:4] if the tier's
  preferred list doesn't intersect the configured universe, bypassing the gate with no signal.
* Dashboard — five bare-`pass` handlers blank a whole analytics panel while its header still
  renders (app.py:819,1225,1239,1264,1278); `_safe_query` renders a failed query as an empty book
  indistinguishable from a real empty book; the real-close predicate has THREE independent
  implementations with different call conventions.
* Backtest display — dashboard/walkforward/export.py:687,820-821 return hardcoded 0.16/0.15
  volatility constants from an except clause, feeding pricing displays.
* Stale residue — strategies/base.py:58-61 still points readers at ait.strategies.constants, the
  module the 09-14 dashboard fix ABANDONED; it does not exist. A future reader will chase a ghost.
* Test isolation — tests/test_r19_config_authority.py::test_applier_and_readers_share_one_table
  has failed on THIS machine's local .env on every single suite run this entire session (W16
  through W19, unchanged). Local-only, gitignored, does not affect CI — but it is now a permanent
  fixture of every `pytest -q` on this box and nobody has fixed the test's isolation gap.
* Triage (angle E) — the 109 R24 findings + 78 carried-over, re-verification against TODAY's
  source. NEVER STARTED. Explicitly its own wave, not foldable into a lead-probing batch — it is
  re-verification work, not new-defect probing, and is the single largest item outstanding.

ALSO REGISTERED, NOT FIXED, BY EXPLICIT OPERATOR DECISION (not oversights — recorded here so they
are not mistaken for either): the r=0.05 hardcoded rate + missing dividend yield in options_chain
and the backtest engine (W19; needs its own wave with a re-run plan, since two of three sites are
in the backtest engine and changing them breaks comparability with every pre-registered study).

WHAT WAS ACTUALLY DONE, for contrast (so this entry reads as a gap list, not a verdict that
nothing happened): R25 (holistic + host-move class, full), W16 (4 platform fixes), W16b (keeper/
dead-man retired on reflection), W17 batch 1 (5 leads, 4 verified + 1 partial), W17 batch 2 (6
leads, all verified), W18 (Gateway-port unblock + 4 of the batch-2 findings fixed), W19 (strike-
selection decision recorded + 2 more batch-2-family findings fixed). Every item above is what
those rounds did NOT reach, not a claim that the rounds were unproductive.

NEXT WAVE NUMBERING, corrected: W20 is unclaimed. Candidates in rough priority — (a) probe the
batch-3 calendar-fault class (4 sites, one dependency, highest measured consequence of what
remains unprobed); (b) tz-01 + qual-01/02/04 (cheap, bounded, tz-01 only stays cheap while the ML
corpus is small); (c) the register triage (109+78) — largest, and the one thing every other wave
has deferred past twice now.

## 2026-09-16 - W20 EXECUTED: the naive-local-clock class — re-scoped twice, by its own probes
W20 was planned as "probe the batch-3 CALENDAR-FAULT class". It shipped as something else, twice
over, because probing contradicted me both times. Both corrections are the entry.

[0] FIRST RE-SCOPE: THE TRIGGER I NEVER TESTED. The batch-3 census claimed a single NYSE-calendar
    failure would concurrently restart the live bot mid-session, open the pre-event blackout and
    disable both event-driven exits. I ranked W20 on that consequence without probing how likely
    the trigger is. It isn't: pandas_market_calendars makes NO network call at schedule() time
    (verified in source), does NOT raise on out-of-range dates (1700 and 2100 both return
    normally), and there are ZERO calendar faults in any log or ledger entry, ever — the 13
    "calendar" hits in ait.log are all calendar_skip, the economic-event blackout working
    normally. Two fallback tests already existed. So "calendar fault" was the wrong organizing
    principle; what is actually broken is the FALLBACK BEHAVIOUR, for a reason with nothing to do
    with calendars — this host runs CEST (= ET+6) and the fallbacks read a naive local clock.
    That makes it the same root cause as tz-01, which the backlog ranked separately at (b). One
    wave, organized by root cause instead of by trigger.

[0b] SECOND RE-SCOPE, AND THE FINDING THIS WAVE ACTUALLY TURNS ON. After writing W20b I ran the
    post-fix probe and read its output properly: making _in_post_market_window timezone-correct
    DOES NOT CLOSE THE HOLE. It fixes the FUNCTION. "False, we are not post-market" is the RIGHT
    answer mid-RTH. The fail-open is one frame up, in the caller —

        except Exception: market_open = False     # the pre-W20 caller, now master.py:814

    — which turns "I don't know" into "safe to restart". Probed on this box with the calendar
    faulted and the .retrained marker present: 09:05, 10:05, 11:05, 13:05 and 15:05 ET ALL took
    the restart branch. That is a stop()/start() of the LIVE bot, mid-session, with positions open.
    WORSE, AND THIS IS THE PART WORTH RECORDING: W20b made one hour NEWLY exposed. On this CEST
    box the old naive fallback protected 10:00-10:15 ET BY ACCIDENT (10:05 ET = 16:05 local), and
    correcting the timezone removed that accident without closing the hole. W20b shipped alone
    would have been a net regression for that hour. The pre-fix test matrix proves it: with the
    fix stashed, does_not_restart_during_RTH[10] PASSES while [11] [13] [15] FAIL.

[1] W20d THE CALLER FAILS CLOSED NOW. New _assume_market_open_on_clock(): a calendar-FREE answer
    to "is the market open?" — weekday 09:30-16:00 ET, resolved through stdlib zoneinfo, NOT
    ait.utils.time (that module imports pandas_market_calendars at module scope, so in the very
    scenario that calls this it may be the broken thing). On a fault the marker site now DEFERS
    through RTH and the marker survives. Both halves of the contract are pinned: it still restarts
    overnight, at weekends and pre-open, so fresh models are never wedged by a fault — they land
    the same night. NOTE THE DIRECTION, because this looks like the heuristic W15 DELETED: W15
    removed weekday()<5 from the pre-event blackout because counting a holiday as a session made
    that gate fail OPEN. Here every caller asks "is it open?" in order to be MORE cautious when
    the answer is yes, so a holiday wrongly counted as a session only defers a restart. The
    docstring says so explicitly and forbids reuse where the true branch is the permissive one.
[1b] SAME CLASS, SECOND INSTANCE, FIXED WITH IT. The heartbeat HANG check (master.py:760-772) called
    is_market_open() inside the outer try, so a calendar fault escaped to the outer handler and
    silently DISARMED hang detection for the whole outage — a hung event loop mid-session going
    unnoticed is the exact failure class (R5-D1) that check exists for. Now falls back to the same
    proxy and stays armed. Fixing the class, not the instance — see W18 [0c] for why that rule is
    in force.

[2] W20a tz-01, THE LIVE ONE, AND A THIRD PRODUCER I DID NOT FIND BY READING. hour_of_day is an ML
    FEATURE. The historical corpus was written on an ET box; this one is CEST, so the same market
    hours that read {9..16} there read {15..22} here and a model trained on the mix learns a split
    that encodes WHICH MACHINE WROTE THE ROW. Unlike everything else in this wave it is gated on
    no fault at all — every row written on this host since 09-14 already carries it. I fixed the
    two producers I knew about (orchestrator.py:1597 live inference context, :3150 the PERSISTED
    entry_signals) and wrote a cheap class-guard test asserting no live code stamps an hour from a
    naive clock. THE GUARD IMMEDIATELY FOUND A THIRD: orchestrator.py:1221, and it is not
    telemetry — it is a LIVE ENTRY GATE. analyzer.py:461 derives the blocked hour from
    trades.entry_time and adaptor._block_hour (adaptor.py:382) accepts only 9<=hour<=16; both are ET-RTH
    conventions, while the gate read the naive CEST clock. Consequence: an ET-learned block (say
    10:00 ET) could only ever match at 10:00 CEST = 04:00 ET, outside RTH, so it NEVER fired; and
    CEST hours 17-22 (= 11:00-16:00 ET) fall outside _block_hour's guard, so most of the session
    was unblockable in the first place. LATENT, NOT FIRING: no blocked_hours state exists anywhere
    on disk, so nothing is blocked today — it fires the first time the learner blocks an hour.
    This is the second time in two waves that a class-guard test caught what my reading missed.
[2b] NOT FIXABLE HERE, REGISTERED. meta_label.py's training-side entry_hour stays contaminated:
    trades.entry_time is written naive-local at executor.py:389 with no offset, so pre-move rows
    are ET and post-move rows are CEST and the convention cannot be recovered on read. That is
    correct code operating on mixed-convention data. It clears only when entry_time becomes
    timezone-aware or the rows are restated. Cheap today only because the meta-labeler is
    untrained and default-off.
[2c] BACKTEST PARITY, CHECKED AND DELIBERATELY UNCHANGED. engine.py:634's hardcoded
    hour_of_day=10 is a CONSTANT, not a clock read, so it has no timezone bug — and it is an
    ET-plausible RTH hour. Fixing the live producers is precisely what RESTORES agreement with it
    (pre-fix, live rows on this host wrote 15..22 against the backtest's 10). Changing it would
    have been the error. Pinned by a test so a later reader does not "fix" it.

[3] W20b/c THE TWO REMAINING NAIVE FALLBACKS. _in_post_market_window's calendar fallback now
    resolves ET through stdlib zoneinfo (measured pre-fix: protected 10:00-10:15 ET by accident,
    allowed restarts at 09:05/11:05/13:05/15:05). _sessions_until anchored on
    datetime.now().date() — the HOST's date, which on CEST rolls over at 18:00 ET, so any evening
    run counted sessions from TOMORROW. And its calendar fallback was `day.weekday() < 5`, the
    EXACT heuristic W15 removed from the primary path directly above it: the consumer blocks when
    sessions <= blackout, so inflating the count makes the gate fail OPEN. W15 fixed the instance
    and left the fallback carrying the defect. Now `trading = False` — fewer sessions, more
    blocking, the safe direction when a holiday is unknowable.

[4] VERIFICATION. tests/test_w20_host_clock_and_calendar_faults.py, 47 tests. Every one PINS A
    REAL NON-ET HOST TZ (TZ + tzset) and drives the no-argument clock path, because that is
    exactly the gap that let test_calendar_outage_falls_back_to_the_clock_not_forever (R16) pass
    for months while the fallback was wrong on this machine — it passes EXPLICIT NAIVE DATETIMES,
    so it only ever exercised the arithmetic. Two tests test the HARNESS, so a silently
    ineffective TZ pin cannot make the other 45 vacuous.
    PROVEN AGAINST PRE-FIX SOURCE (source stashed, tests kept): 35 failed / 12 passed. The 12 are
    signal, not slack, and each is accounted for: the 2 harness tests; [US/Eastern] passing while
    [Europe/Berlin] fails, which LOCALIZES the defect to this host rather than the logic; the
    accidental 10:05 pass from [0b]; the 2 "still restarts overnight/at the weekend" tests, never
    broken and present so a future fix cannot over-correct into permanent deferral; the backtest
    parity pin from [2c], which asserts something deliberately UNCHANGED; and the 2 learning-gate
    tests, which pin the writer/reader invariant rather than the fix (the fix itself is pinned by
    the class-guard test, which does fail pre-fix).
    ONE OF MY OWN TESTS WAS TOO WEAK AND WAS REPLACED. test_anchors_on_the_ET_date_not_the_host_date
    first used a midweek evening, where BOTH anchors return 0 — it passed pre-fix and proved
    nothing, the same defect as the R16 test this wave exists to correct. Rewritten onto a FRIDAY
    evening, where the wrong anchor skips the weekend: it now fails pre-fix with 1 != 0, and that
    1 is the fail-open — the consumer blocks when sessions <= blackout, so a spurious session
    means no block.
    ONE EXISTING TEST NEEDED UPDATING: test_r16_round2_fixes.py's _FakeDT.now() took no arguments
    and _sessions_until now calls now(tz); the pinned date it asserts on is unchanged.
    FULL SUITE: 1677 passed / 43 skipped / 1 failed in 15:07 — the single failure is
    test_r19_config_authority::test_applier_and_readers_share_one_table, the local-.env isolation
    gap that has failed on THIS box on every suite run since W16, unchanged and unrelated. The
    baseline was 1630 passed, so +47 is exactly this wave's new tests and no regressions.
    scripts/typecheck.py clean. scripts/smoke_deploy.py -> SMOKE PASS (13 checks), including the
    real dry-run trading cycle. Probe hygiene held: PYTEST_CURRENT_TEST set before every
    supervisor call, 0 probe lines in logs/orchestrator.log (verified, not assumed).

[5] NOT DONE, DELIBERATELY. portfolio.py:779/:811 (the earnings and macro exits) are in the
    census's calendar-fault group but their fallback is a plain `except: pass` with no clock
    involved — a different defect in a different class (the fail-open census). Folding them in
    would recreate the organize-by-trigger error this wave exists to correct. The remaining 11
    batch-3 shortlist sites are still unprobed.

[5b] ONE LEAD RAISED AND REFUTED DURING THIS WAVE, plus my own bad probe. Sweeping master.py for
    other naive-clock reads surfaced _backtest_catchup_run_at, which tests `run_at.hour` against
    the ET session window (9,15)-(16,30) with the explicit intent "the 10-min full-core backtest
    has no business running beside the live bot". Read on a CEST host that looks catastrophic, and
    my FIRST probe said so: 6 of 9 boot times landing a full-core backtest inside RTH, three of
    them pushed to 10:40 ET by the very guard meant to prevent it. THAT PROBE WAS WRONG. It
    interpreted the function's naive return as local time, when the consumer passes it straight to
    APScheduler as run_date and the scheduler is pinned timezone="US/Eastern" (master.py:1884), so
    a naive run_date is read AS ET. The corrected probe swept all 24 boot hours: NONE lands inside
    RTH. The guard is robust by construction, not by luck — the same naive value is both the
    window-test input and the scheduled time, so the two agree whatever timezone it nominally
    represents. I also probed the counterfactual (what if a naive-clock sweep "fixed" this site to
    now_et() in isolation?): still none. REFUTED, and recorded here at the same weight as the
    confirmations, per the standing rule. Residue, ops-tier, NOT fixed: because the value is a
    local wall clock read as ET, the weekly catch-up fires about six hours later in real time than
    its docstring implies, and a boot after 18:00 ET stamps it with tomorrow's date. Never in RTH.
[5c] REGISTERED, NOT FIXED — two conventions for one marker. daily_digest writes
    data/digest_sent_<LOCAL date> (master.py:1512) while _digest_catchup_due reads
    digest_sent_<ET date> (:1534, correctly fed now_et()). They disagree only between 18:00 ET and
    midnight ET, and both digest crons fire at 09:35/16:05 ET — outside that window — so the
    marker is written with the right date in normal operation and nothing is currently broken. The
    same line's SQL date filter (:1458) has the same window. Latent; a digest sent inside that
    window would suppress the next day's. ops_reporting, out of this wave's live-path scope.
[5d] TWO STALE COMMENTS FIXED, because a false timezone claim in a codebase that just had a
    timezone bug class is worse than no comment. master.py:2006 read "naive local (=ET box),
    matches file mtimes" and _backtest_catchup_due's docstring said "`now` is naive local time
    (box runs ET)". The CODE is correct in both — the comparison is against file mtimes, which are
    also naive local. Only the parenthetical was false, and it is exactly the assumption this
    entire wave exists to retire.

[6] THE RECORD. This is the THIRD time this session that probing corrected one of my own priority
    calls, and the second time within this single wave. The ranking I wrote in the BACKLOG
    CORRECTION entry hours earlier — (a) calendar-fault class, (b) tz-01 — is superseded by its
    own evidence: (a) and (b) are one class, and the trigger that made (a) look urgent is
    implausible. The register is only worth keeping if it shows that as prominently as it shows
    the confirmations.

## 2026-09-16 - W20 DEPLOYED + WATCH LIST FOR THE OPEN
Operator authorised the restart. Deployed 12:15 local / 06:15 ET — PRE-MARKET, 3h15m before the
bell, with ZERO open trades. W20 touches restart logic and a live entry gate, so an in-session
deploy was not acceptable.

WHY THE SUPERVISOR, NOT JUST THE BOT. W20d lives in master.py — the SUPERVISOR's own code. A
bot-only restart would have deployed W20a/b/c and silently left the headline fix (the mid-RTH
restart fail-open) running on the old code. Full supervisor restart was required and performed.

SEQUENCE. SIGTERM -> 51749 (same handler as the operator's Ctrl-C: graceful_shutdown stops bot,
web services and scheduler). Clean trace: shutdown_signal 15 -> bot_stopping 51751 -> bot_stopped
(30s) -> all three web services stopped. Relaunched with nohup (NOT setsid — it does not exist on
macOS, a lesson this session already paid for once). New tree: supervisor 81012 (reparented to 1,
properly detached), bot 81032, dashboard 81069, weblog 81070, status 81071. Scheduler ready with
10 jobs, all crons ET-pinned.

THE STALE LOCK WAS CHECKED BEFORE RESTARTING, NOT AFTER. data/orchestrator.lock still held the
dead pid 51749. W16 A4 made that lock FAIL CLOSED, so "will it refuse to start?" had to be
answered first: it reclaims correctly. The guard is the fcntl flock, released by the OS on process
death; the pid in the file is informational only, and fail-closed applies to plumbing exceptions
(permissions, full disk, missing fcntl), not to a stale file.

VERIFIED RUNNING THE NEW CODE, positively — not inferred from "the files on disk have it":
_assume_market_open_on_clock present; health_check calls it; three _now_et().hour sites in
orchestrator; _sessions_until's EXECUTABLE lines are `trading = _itd(day)` / `trading = False`
with no weekday heuristic (the string survives only inside the W20c comment, which quotes the old
code — the first check I wrote was fooled by exactly that and was corrected). Working tree clean
at f6e3c36. 47 W20 tests re-run against the deployed source: all pass. Bot connected to Gateway
4002, placed and cancelled the $1 SPY probe order (W17-08 verify_can_trade), so the account is
trading-enabled and not read-only. No errors, warnings or criticals since restart.

WATCH LIST FOR THE 09:30 ET OPEN:
[W1] HEARTBEAT AT THE BELL — the highest-risk item and NOT caused by W20. data/bot_heartbeat is
     856 min stale right now, because the bot only touches it while the trading loop runs (last
     write was yesterday's close: 22:00 local = 16:00 ET, and 856 min lands exactly there). At
     09:30 ET is_market_open() flips True; if the loop has not refreshed the marker by then, the
     >15min rule restarts the bot and pages. Expected behaviour is a refresh as the loop starts.
     W20 touched this block, so confirm it is the ORDINARY path that runs, not the new fallback.
[W2] THE TWO NEW WARNINGS MUST NEVER APPEAR — fresh_models_market_check_failed and
     heartbeat_market_check_failed fire only on a genuine calendar fault. W20's whole re-scope
     rests on the claim that such faults are implausible (no network call, no date limits, zero
     ever recorded). If either line appears, that claim is WRONG and the wave's ranking needs
     revisiting. Their absence is the falsifiable prediction this wave makes.
[W3] hour_of_day IN THE FIRST LIVE ROW — the first entry today should write hour_of_day in 9..16
     (ET) into trade_context.entry_signals, not 15..22 (CEST). This is the only W20a change with
     an observable artifact; check it on the first fill rather than assuming.
[W4] NO RESTART LOOP — confirm the supervisor does not restart the bot repeatedly. W20d changed
     a branch that gates a restart; the failure mode of getting it wrong is visible within two
     health-check cycles (4 min).
[W5] PRE-EVENT BLACKOUT — _sessions_until now anchors on the ET date. Watch
     credit_entry_skipped_pre_event for appearing or disappearing unexpectedly; W20c made its
     calendar fallback strictly MORE blocking, which is the safe direction but is still a change
     in behaviour on a fault.
[W6] hour_blocked_by_learning should stay silent — the gate now reads ET, but no blocked_hours
     state exists on disk, so nothing should block. A block appearing at an odd hour would mean
     the learner wrote one and the convention is still mismatched somewhere.

## 2026-09-16 - W20 CORRECTIONS + W21 POINTER
Four corrections to the W20 entry above, and the "what next" pointer it failed to leave. Appended,
not edited, per the ledger rule.

[C1] THE CENSUS ALREADY NAMED W20d. MY PLAN DROPPED IT. The W20 entry's [0b] reads as though my
     post-fix probe discovered the caller's fail-open. It did not. The batch-3 census names it
     verbatim as shortlist site #11: "Calendar fault => market_open=False => _in_post_market_window()
     False at e.g. 11:00 => the post-retrain branch runs stop(); start() — restarting the live bot
     mid-RTH with positions open." That is W20d, written down a day before W20 existed. What
     happened is the reverse of the credit I gave myself: while re-scoping the wave from "trigger"
     to "root cause" I narrowed the target to _in_post_market_window's TIMEZONE and lost the
     caller the census had correctly identified; the probe then recovered it. The re-scope was
     still right — but re-scoping is where a correct finding got dropped, and that is the lesson
     worth keeping, not "probing found what reading missed".
[C2] THE TRIGGER SURFACE IS WIDER THAN "CALENDAR FAULT", so [0]'s implausibility argument is
     narrower than it sounds. The guard is a bare `except Exception` around BOTH the import and
     the call:  `from ait.utils.time import is_market_open; market_open = is_market_open()`.
     ANY failure of either — import error, a transient inside is_market_open, not just a broken
     pandas_market_calendars — set market_open=False and unlocked the restart. W20's evidence that
     calendar FAULTS are implausible stands; it does not establish that the except never fires.
     W20d is therefore justified on its own, independent of how likely calendar faults are.
[C3] COUNT CORRECTION: the W20 entry says "the remaining 11 batch-3 shortlist sites are still
     unprobed". It is 10. W20 probed and FIXED two of the twelve — #11 (master.py health_check,
     = W20d) and #12 (orchestrator _sessions_until, = W20b/c). Sites #4 and #5
     (portfolio.py:779/:811) remain deliberately unprobed and are a DIFFERENT class: plain
     `except: pass`, no clock.
[C4] THE STALE-POINTER MISTAKE, REPEATED ONE WAVE LATER. PLAN.md line 2967 still reads "NEXT WAVE
     NUMBERING, corrected: W20 is unclaimed" with candidates (a)/(b)/(c). That is now false —
     (a) and (b) were one class and W20 executed both. This is precisely the staleness the
     BACKLOG CORRECTION entry was appended to fix, recreated by the very wave that superseded it.
     A wave entry is not finished until it leaves a correct pointer to the next one.

W21 IS UNCLAIMED. RANKED, and this ranking is the one to follow:
 (a) THE REGISTER TRIAGE — 109 R24 findings + 78 carried over, re-verified against TODAY's source.
     The largest outstanding item, never started, now deferred past FOUR times (W17, W18, W19,
     W20). It is re-verification of existing claims, not new-defect probing, so it has never won
     a priority contest against live-path work — and that is exactly why it should stop competing
     and simply be scheduled.
 (b) THE 10 REMAINING BATCH-3 FAIL-OPEN SITES, consequence-ranked in
     reports/w17_batch3_failopen_census_20260916.md. Highest first: #2 reconciler.py:1083 (the
     HALT_UNTRACKED freeze file is never written), #3 orchestrator.py:701 (one unparseable
     exit_time silently drops realized P&L from the day), #6 state.py:705 (corrupt partial_exits
     JSON re-fires a completed exit level), #10 orchestrator.py:1074 (watchdog recovery action
     lost). Probe in batches, stop after each, re-rank the tail — the cadence the operator chose.
 (c) arch-01..04, qual-01/02/04, broker/orders/contracts dead & untested code, capital_tiers.py,
     5 silent dashboard panels + 3 conflicting real-close SQL implementations, hardcoded
     backtest-display constants.
 (d) SMALL AND VERIFIED-STILL-BROKEN TODAY, cheap enough to fold into any wave: strategies/base.py
     :58-61 points readers at ait.strategies.constants, which does NOT exist (confirmed 09-16);
     tests/test_r19_config_authority.py::test_applier_and_readers_share_one_table still fails on
     this box's local .env on every suite run (confirmed 09-16, unchanged since W16) — it is now a
     permanent fixture of every `pytest -q` here and nobody has fixed the isolation gap.
 (e) NEW FROM W20, registered not fixed: the two digest-marker date conventions (daily_digest
     writes a LOCAL-date marker, _digest_catchup_due reads an ET-date one; they disagree only
     18:00 ET -> midnight, and both crons fire outside that window, so nothing is broken today).
 DEFERRED BY EXPLICIT OPERATOR DECISION, not oversight: the r=0.05 hardcoded rate + missing
     dividend yield (needs its own wave with a re-run plan — two of three sites are in the
     backtest engine, so changing them breaks comparability with every pre-registered study).

STATE AT HANDOFF: HEAD b7be0be on branch feature/event-straddle, NOT PUSHED. Live bot runs W20
(supervisor 81012 / bot 81032) since 06:15 ET. The W20 DEPLOYED entry's 6-item watch list is
UNCHECKED — the market had not opened when this was written. Check W1 (heartbeat at the bell) and
W2 (the two new warnings must never appear) before starting any new wave.

## 2026-09-16 - W1 REFINED (pre-open code read) — no margin, and R8 makes staleness a real signal
Reading the heartbeat path before the open changed W1 from "watch it" to something specific. The
W20 DEPLOYED entry's [W1] is correct but incomplete; this supersedes it.

[a] THERE IS NO PRE-MARKET MARGIN. I first said the loop starts at the 09:00 ET PRE_MARKET phase,
    giving 30 minutes of slack before is_market_open() flips at 09:30. WRONG. _trading_loop is
    gated `while ... get_current_phase() == TradingPhase.MARKET_OPEN` (orchestrator.py:1027), and
    MARKET_OPEN begins at 09:30. At PRE_MARKET the bot runs _pre_market() and then
    wait_for_phase(MARKET_OPEN) (:519-521) — it does NOT write the heartbeat. So the marker starts
    refreshing at 09:30:00, the SAME INSTANT the supervisor's market gate opens. Zero designed
    margin; it is a race.
[b] THE BOT NORMALLY WINS THAT RACE, and the reason is worth knowing rather than assuming: the
    heartbeat write is the FIRST statement in the loop body (orchestrator.py:1042, right after the
    in-process watchdog touch), before any scan or IB call. It lands within milliseconds of the
    phase flip, while the supervisor only re-checks every 2 minutes.
[c] SO THE TWO REAL FAILURE MODES ARE NARROW AND NAMED:
    (i)  _pre_market() overruns past 09:30 — it is awaited BEFORE wait_for_phase, so a slow
         pre-market (chain fetch, IB latency, a retry) delays loop entry and therefore the first
         heartbeat, while the supervisor's gate is already open.
    (ii) R8 SUPPRESSION. The write is conditional: `if not any(n for n, _ in _err_streaks.values())`
         — R8 deliberately lets the heartbeat go stale during an active error streak, precisely so
         the supervisor's 15-min restart+alert DOES fire (the 07-10 error loop kept hang detection
         blind while stops were dead). CONSEQUENCE FOR WHOEVER WATCHES THIS: a stale heartbeat at
         the bell is a GENUINE SIGNAL, not a startup artifact. Do not dismiss it as "the loop just
         hasn't started yet" — check _err_streaks first.
[d] WHAT TO CHECK, CONCRETELY: at ~09:32 ET confirm data/bot_heartbeat mtime is within 1 min, and
    that logs/orchestrator.log contains no bot_hung_heartbeat_stale and no restart around 09:30.
    If it DID restart, [c](i) vs [c](ii) is distinguishable from the bot log: an error streak
    leaves error lines, a slow pre-market does not.

W2 STATUS AT 06:30 ET, partially closed already: 0 occurrences of heartbeat_market_check_failed
across 10 health-check cycles since the restart — that site calls is_market_open() every cycle, so
those are real clean samples and the calendar is working. The fresh_models_market_check_failed half
has NOT been exercised yet: it only runs when models/.retrained exists, which the 07:30 ET retrain
cron creates. That half closes at ~07:32 ET, still pre-open.

## 2026-09-16 - SESSION CLOSE: handoff state, and a CLOSING WINDOW worth acting on first
Final entry of the session. Supersedes the handoff block in "W20 CORRECTIONS + W21 POINTER".

[X1] CORRECTION — THE TRAINING CORPUS IS CLEAN. W20's [2] claims the tz-01 defect "is gated on no
     fault at all — every row written on this host since 09-14 already carries it" (line 3028).
     THAT IS FALSE, and measuring it was the only way to find out. `select ... where entry_time >=
     '2026-09-14'` returns ZERO rows. The ENTIRE trade corpus is 10 trades, all entered
     2026-04-23/24 — on the OLD ET host. So the corpus is uniformly ET and the contamination I
     described is hypothetical, not actual. W20a landed BEFORE a single bad row was written. The
     mechanism was real; the damage was zero. Both halves matter, and the entry above only said
     one of them.
[X2] THE WINDOW THAT IS CLOSING, AND WHY IT IS WORTH DOING BEFORE ALMOST ANYTHING ELSE.
     executor.py:389 still writes `entry_time=datetime.now().isoformat()` — naive, no offset — and
     both entry_time and exit_time are stored offset-free (verified against the live DB). The FIRST
     trade entered on this CEST box writes the first non-ET row into an otherwise-ET corpus. That
     can happen at the very next open. While the corpus is still uniformly ET, making these
     timestamps timezone-aware is a CONVENTION CHANGE with no data restatement. After the first
     live entry it needs a restatement with a pre/post boundary, which is the expensive version and
     the one W20's [2b] described as "clears only when entry_time becomes timezone-aware or the
     rows are restated". The cheap door is open right now and closes on the next fill.
[X3] BUT IT IS NOT A ONE-LINER — SCOPE IT BEFORE STARTING. 19 sites parse these two columns across
     9 modules (learning/analyzer ×5, dashboard/app ×3, monitoring/analytics ×3, bot/orchestrator
     ×4, ml/meta_label, execution/portfolio, execution/reconciler, dashboard/walkforward/export).
     At least one does ARITHMETIC against a naive clock — reconciler.py:788 computes
     `now - datetime.fromisoformat(t.entry_time)` — which raises TypeError the moment awareness is
     mixed. So the writer cannot change alone. Treat it as its own wave with the 19 readers
     enumerated, not as a quick fix folded into something else.
[X4] CONTEXT THAT CHANGES SEVERITY RANKINGS: the corpus is TEN TRADES, all from April. Every
     ML/meta-labelling finding in this ledger (tz-01, the meta-labeller contamination, the
     counterfactual/Thompson state) is research-validity work on a corpus too small to train on,
     and the meta-labeller is untrained and default-off. Do not rank those above live-path work on
     the strength of the word "contamination".
[X5] SECURITY POSTURE, RE-VERIFIED TODAY so it is not re-litigated: repo prithvipradip/trade_v2 is
     PRIVATE; `logs/`, `/data/` and `.env` are all gitignored; the account number appears in NO
     tracked file. It does appear in local logs/ait.log, which is correctly ignored. Nothing to do.
[X6] HANDOFF STATE, CURRENT: HEAD 47facfc on branch feature/event-straddle. FOUR COMMITS UNPUSHED
     (f6e3c36, b7be0be, 3d8871d, 47facfc) — fine for a next session on THIS machine, not fine
     against machine loss or a session elsewhere. Live bot runs W20: supervisor 81012 / bot 81032,
     restarted 06:15 ET. Suite baseline is now 1677 passed / 43 skipped / 1 known local-.env
     failure — a new agent seeing 1677 should NOT read the +47 over the old 1630 baseline as drift.
[X7] THE WATCH LIST IS STILL OPEN. W2 is half-closed (10 clean cycles; the fresh_models half needs
     the 07:30 ET retrain marker). W1 is unresolved and now has a specific 09:32 ET check with two
     named failure modes — see the "W1 REFINED" entry directly above; read it BEFORE interpreting
     a stale heartbeat, because R8 makes staleness a deliberate signal rather than an artifact.
     W3-W6 untouched. Check these before opening W21.

## 2026-09-16 - W21 EXECUTED: tz-aware trade timestamps — and three live bugs the suite found, not I
SESSION CLOSE [X2] said the cheap door closes at the next live fill. It was still open this
morning and this wave walked through it. Chosen over the ledger's own ranking (register triage
first) for one reason: the triage costs the same next week and this did not. The migration itself
was routine. What was NOT routine is that my method for enumerating the work was unsound, and
only the suite exposed it — see [4].

[0] WHAT WAS MEASURED FIRST, because the whole cost argument rests on it. `data/ait_state.db` at
    07:2x ET: 10 trades, ALL entered 2026-04-23/24, all `closed`; `open_positions` 0 rows,
    `exit_outbox` 0 rows, 0 rows with `entry_time >= '2026-09-14'`. [X1]'s correction holds — the
    corpus is uniformly ET and the contamination W20 [2] described never happened. So this was a
    CONVENTION CHANGE (stamp the offset each row already meant) and NOT a restatement. Nothing in
    the data moved: re-read after the backfill, the ET trading days are still 2026-04-23/24 and the
    entry hours are still 9/10/14 ET. That last number is the check that matters — had the naive
    values been misread as host-local CEST, the hours would have landed on 3/4/8, outside RTH.

[1] WHY THE WINDOW WAS STILL OPEN, and it was not luck. `data/HALT_UNTRACKED` has frozen ALL new
    entries since 09-15 20:00Z (orchestrator.py:2325 returns "symbol handled" for every signal
    while it exists). No fill could land. The operator wants that halt lifted within days — and
    LIFTING IT IS WHAT CLOSES THIS WINDOW. W21 before W22 is causal ordering, not preference.

[2] THE [X3] CORRECTION — THE FAILURE MODE IS SILENCE, NOT A CRASH. [X3] says reconciler.py:788
    "raises TypeError the moment awareness is mixed". It raises, but the raise is CAUGHT:
    `except (ValueError, TypeError): continue`. A half-migrated corpus therefore does not crash —
    the trade is dropped from the stale-PENDING sweep and the sweep degrades to a no-op with
    nothing in the log. That is why this wave added a `log.warning` on that path, pinned by an AST
    test so a future tidy-up cannot restore the silence.

[3] THE SCOPE [X3] MISSED — 5 COLUMNS, NOT 2.
    (a) `open_positions.entry_time` (state.py:490) — a fourth surface, 0 rows, migrated now rather
        than left as a naive island.
    (b) `exit_outbox.exit_time` cutoffs (state.py:851/:856) are compared as TEXT
        (`WHERE exit_time < ?`). A naive cutoff against an aware column mis-selects SILENTLY, and
        on a CEST box it mis-selects in the dangerous direction: a naive local string sorts ABOVE
        an ET cutoff, so every pending booking goes unclaimed.
    (c) `exit_outbox.claimed_at` (state.py:827/:832) — NOT in the plan. Found only because it is
        compared against a value (b) was changing.

[4] THE METHOD FAILURE, AND IT IS THE POINT OF THIS ENTRY. [X3] enumerated "19 sites" by grepping
    `fromisoformat` lines that NAME entry_time/exit_time. I inherited that method, and my first
    class-guard test encoded the same blind spot. IT IS UNSOUND: a reader can select the column in
    SQL and parse `_row[0]`, so the column name never appears on the parse line. TWO such readers
    existed, both in live paths, and BOTH were found by the test suite rather than by reading:
    (a) `orchestrator._post_stop_cooldown_until` (:2221). A LIVE ENTRY GATE. Its
        `_cooldown_now()` docstring stated the coupling outright — "kept naive-local on purpose —
        trades.exit_time is written with datetime.now() (state.py) and the host runs ET, so the DB
        rows and this clock share one convention". W21 broke the first half of that premise; the
        CEST host move had ALREADY broken the second, so its reference DATE was rolling over six
        hours early and selecting the wrong session — a W20-class defect W20 did not reach. Both
        halves fixed: `_cooldown_now()` returns `now_et()`, and the cutoff is aware ET.
    (b) `master._mirror_content_age_hours` (:1440). `MAX(entry_time)/MAX(exit_time)` minus a naive
        `datetime.now()` raised TypeError into the surrounding `except`, which returns None — and
        the caller reads None as a MISSING alarm. The backup-mirror freshness check would have
        alarmed permanently instead of measuring anything.
    THE GUARD NOW ASKS A QUESTION A GREP CANNOT: at FUNCTION scope, does this function touch a
    migrated column AND parse a timestamp by hand? That formulation catches both misses. Four
    allowlisted exceptions, each with a stated reason.

[5] A THIRD BUG, MINE, ALSO CAUGHT BY THE SUITE. orchestrator.py:2252 — I migrated the cooldown's
    `entry_time` parse but not the `cutoff = dt.now()` it is compared against. TypeError, swallowed
    by a bare `except`, verdict "unverified" — and that guard FAILS CLOSED, so it would have
    REFUSED EVERY ENTRY. Deploying without running the suite would have silently stopped the bot
    trading. Recorded plainly: three live-path defects in this wave, and reading found none of them.

[6] READERS: 18 BY NAME + 2 BY INDEX. `dashboard/walkforward/export.py:91` is NOT migrated and must
    not be — it slices `[:10]` for a DATE (no offset can perturb ten characters) and also handles
    backtest rows whose `exit_date` is a plain date. Migrating it would have been the error.

[7] VERIFIED, NOT ASSUMED: the DuckDB mirror is unaffected. Its `entry_time`/`exit_time` are
    `TIMESTAMP` (not `TIMESTAMPTZ`) and DuckDB DROPS the offset on ingest — probed 09-16, an aware
    string and a naive one land as the identical naive value. Both sides of its hold-time
    subtraction stay naive ET wall-clock, so `duckdb_analytics.py:416/417` is consistent and
    deliberately untouched. Had it converted to UTC instead, every duckdb hold time and date bucket
    would have shifted four hours.

[8] WHAT LANDED. `parse_trade_ts()` in ait/utils/time.py is the single reconciliation point: naive
    in -> localised ET, aware in -> CONVERTED to ET, so `.date()` and `.hour` are ET calendar
    semantics for every caller regardless of what offset a row carries. Writers emit
    `now_et().isoformat()`. Backfill is `scripts/restate_w21_tz_aware.py` (dry-run default,
    `--apply` snapshots first, idempotent, derives each offset from the ET zone so EST and EDT rows
    are each right, and ABORTS rather than relabelling any row written on/after the 09-14 host
    move). Snapshot taken: `data/backups/ait_state.pre_w21_tz.20260916_133646.db`.

[9] TESTS. `tests/test_w21_tz_aware_trade_timestamps.py` (15), following W20's rules: a real non-ET
    `TZ` pin driving the no-argument clock path, plus a harness test so an ineffective pin cannot
    make the rest vacuous. Ten EXISTING test files also had fixtures migrated — they seed rows the
    writer produces, so they must follow the writer. That migration was not bookkeeping: making
    `test_r21_post_stop_cooldown`'s fixture honest is what exposed [4](a).
    SUITE (final, post-fix): 1692 passed / 43 skipped / 1 known local-`.env` failure (153 deselected — slow/integration markers), up from the 1677/43/1 baseline in [X6]. Every new pass is a test this wave added or migrated; the one known failure is unchanged —
    `test_r19_config_authority.py::test_applier_and_readers_share_one_table`, the local-`.env`
    isolation gap open since W16.

[10] NOT DEPLOYED BY ME. The operator restarts `ait.main`. Ordering that matters: the backfill is
     ALREADY applied, so the DB is aware while the RUNNING bot (81032, up since 06:15 ET) still
     holds the old naive-writer code. Nothing can write a naive row right now — entries are frozen
     by HALT_UNTRACKED and there are no open positions — but that gap closes the moment the halt is
     lifted. RESTART BEFORE LIFTING THE HALT.

[11] CORRECTION OF A STALE POINTER, per [C4]. PLAN.md:2967 still reads "NEXT WAVE NUMBERING,
     corrected: W20 is unclaimed". False since W20 executed, flagged once by [C4], and false twice
     over now. Appended, not edited: W20 and W21 are both DONE; W22 is next.

[12] W22 IS NEXT, AND IT IS OPERATOR-BLOCKING. The halt is freezing entries on a bot the operator
     wants trading, and will keep doing so for days. The untracked legs are the operator's own TWS
     calendar spread (QQQ 712C 2026-10-16 +1 / 2026-09-18 -1), not a crash-window position.
     DESIGN CHOSEN BY THE OPERATOR: on first sighting the bot ASKS via Telegram, records the answer
     durably, suppresses future freezes for that contract, and confirms when the position
     disappears. Declared-manual legs are IGNORED FOR MANAGEMENT BUT STILL COUNTED IN EXPOSURE —
     portfolio.py:300/1013 reads `ib.portfolio()` directly and the bot's risk math must stay true
     to the account. Undeclared untracked options still freeze; INST-3 is not downgraded.
     FEASIBILITY, measured not assumed: ait/notifications/telegram.py is SEND-ONLY. `getUpdates`
     appears once, in a setup docstring, and nothing polls it. An inbound path (offset-tracked
     long-poll, pinned to the configured chat_id) has to be built, and THAT is the bulk of W22.
     THREE DEFECTS FOUND IN THAT PATH TODAY, registered not fixed:
     (a) Only the LAST untracked position reaches the halt file — reconciler.py:1080 does
         `write_text` per position inside the loop. There are TWO legs; the file names one.
         Resolving the named one and deleting the file leaves the other live.
     (b) The INST-3 debounce loses state on every bot restart. `_untracked_sightings` is
         per-instance; today's log re-logs "1st sighting, freeze deferred" at 06:56Z, 07:25Z and
         10:15Z — each exactly a bot restart. On a frequently-restarted box a genuine crash-window
         position can evade the freeze indefinitely.
     (c) The freeze that IS active came from a single two-sighting window inside one process
         lifetime (09-15 18:38Z -> 20:00Z). It is sticky only because the FILE persists, not
         because detection re-armed.
     NOTE ON THE BATCH-3 CENSUS: today's file proves the HAPPY PATH of reconciler.py:1080 writes.
     That does NOT refute census site #2, whose claim is about the `except Exception: pass` around
     the write — the freeze announced in the log while the file never appears. Still unprobed.

[13] W23 REGISTERED — THE SCHEDULER TIMEZONE CLASS, same family as W20, still live, found by doing
     the W2 watch check rather than by looking for it. master.py:1889 builds
     `BlockingScheduler(timezone="US/Eastern")` and then hands `add_job` NINE PRE-BUILT
     `CronTrigger` INSTANCES. APScheduler only injects the scheduler's timezone when IT builds the
     trigger from an alias string, so all nine silently fall back to `get_localzone()`.
     PROBED on this box (apscheduler 3.11.2): instance -> `Europe/Madrid`, alias -> `US/Eastern`.
     Every one of those jobs runs 6h early in ET terms. `daily_report` fires 10:30 ET instead of
     16:30, `digest_pm` 10:05, `db_backup` 11:00, `weekly_scorecard` Fri 10:10 — all MID-SESSION;
     `digest_am` fired today at 03:35 ET, ~6h BEFORE the open (`data/digest_sent_20260916`), and
     the retrain log `retrain_20260916_073000.log` is 01:30 ET.
     SEPARATE SECOND DEFECT, same file: the two catch-up jobs (master.py:1970, :2018) pass a NAIVE
     LOCAL `run_date` to an Eastern scheduler. Registered `date[2026-09-14 21:31:45 EDT]`, actually
     ran `[2026-09-15 03:31:45]` local — 6h late, to the second. W20 REFUTED the
     `_backtest_catchup_run_at` lead on this exact mechanism; the lead was right and W20's
     refutation was wrong. Correcting my own call, prominently, per the ledger rule.
     THIRD, minor, same family: the heartbeat marker's CONTENT is a naive local stamp
     (`data/bot_heartbeat` reads `2026-09-15T21:59:38`). Cosmetic only — the staleness check uses
     `st_mtime`, not the content — but it is the same class and belongs in the same sweep.
     NOT FOLDED INTO W21: different class, and folding it in is exactly the re-scope mistake [C1]
     recorded. Fix is mechanical (alias form or explicit `timezone=`, and an aware `run_date`).

[14] WATCH LIST. W2 IS CLOSED, BOTH HALVES: `heartbeat_market_check_failed` and
     `fresh_models_market_check_failed` have 0 occurrences across all of logs/. The fresh_models
     half is no longer untested — `bot_restart_for_fresh_models` fired today and the marker was
     consumed, no warning. "W1 REFINED" predicted that half would close ~07:32 ET; it closed at
     01:30 ET, and asking WHY is what produced [13].
     W1 IS STILL OPEN AND STILL UNCHECKED — everything above was done pre-open (last look 08:19 ET;
     supervisor 81012 / bot 81032 up since 06:15 ET, no restart, no `bot_hung_heartbeat_stale`).
     The 09:32 ET check in "W1 REFINED"[d] HAS NOT HAPPENED: confirm `data/bot_heartbeat` mtime is
     within 1 min and that orchestrator.log shows no `bot_hung_heartbeat_stale` and no restart
     around 09:30. Do not dismiss a stale heartbeat as a startup artifact — R8 makes it deliberate;
     check `_err_streaks` first. W3-W6 untouched.

## 2026-09-16 - W21 DEPLOYED, W1 FIRED FOR REAL, AND A NEW LIVE RACE CONFIRMED (not W20d's class)
Restarted, watched the open, and W1's check caught a genuine live defect while checking it —
not the two failure modes "W1 REFINED" named, a third one, fully root-caused with the bot's own
log line as the smoking gun.

[D1] DEPLOY. Full supervisor restart (required — master.py changed): SIGTERM 81012 at 08:36 ET,
     clean shutdown confirmed (bot/dashboard/weblog/status all stopped in sequence, ~36s), relaunched
     via nohup. New supervisor pid 7458. VERIFIED THE RUNNING CODE, not the files on disk: imported
     the loaded modules and asserted the specific changed lines are present in the executing source
     (now_et() in the cooldown guard, parse_trade_ts in the mirror-age check, the logging guard in
     reconciler, all seven checks OK). Trading verified via a self-placed/cancelled SPY probe order
     (same mechanism W20's deploy used) — Gateway connected, orders route. Zero errors/warnings in
     orchestrator.log across the whole startup sequence.

[D2] HALT_UNTRACKED: deleted 08:41 ET, and the self-reinstatement was PREDICTED before acting, then
     CONFIRMED. Reconciler had already logged a fresh "1st sighting" for both legs at bot startup
     (08:38 ET, new process = empty _untracked_sightings); the debounce's second sighting was
     guaranteed at the next reconcile, which is _pre_market() at 09:00:00 ET sharp — inside the
     [2min, 8h] window. It fired exactly on schedule: `pre_market_starting` -> both legs re-detected
     -> `reconcile_untracked_option_freeze` for BOTH -> file rewritten. Net effect of lifting it
     today: a 19-minute window that closed itself before the market even opened — not a fix, and
     not intended as one; W22 or the operator closing the position at IBKR are the real paths.
     LIVE CONFIRMATION OF A W22 FINDING: the freeze fired for BOTH legs but the file, again, names
     only the LAST one (2026-09-18) — reconciler.py:1080's per-position write_text inside the loop,
     now confirmed at a second, independent freeze event, not just the original 09-15 one.

[D3] W1 — THE CHECK CAUGHT A REAL DEFECT, LIVE, DURING THE CHECK ITSELF. At 09:32:00 ET the
     heartbeat was 63142s (17.5h) stale, `bot_hung_heartbeat_stale` had fired once, and the bot
     process was mid-restart. This is NOT "W1 REFINED"[c](i) slow pre-market (ait.log shows the bot
     already past _pre_market and inside wait_for_phase(MARKET_OPEN) at 09:30:00.071 ET) and NOT
     [c](ii) R8 suppression (zero error-level lines in the whole window). It is a THIRD failure mode,
     root-caused with the bot's own log line as proof, not inference:

       {"target": "market_open", "wait_seconds": 3600, "current_phase": "market_open",
        "event": "waiting_for_phase", "timestamp": "2026-09-16T13:30:00.071453Z"}

     scheduler.py's wait_for_phase(MARKET_OPEN): when `now.time() < MARKET_OPEN` is false (i.e. now
     is AT OR PAST 09:30:00.000, which it always eventually is), it falls to
     `next_open = next_market_open(); wait = (next_open - now).total_seconds()`.
     utils/time.py's next_market_open() makes the SAME strict `now.time() < MARKET_OPEN` check on
     its OWN fresh now_et() read, so at or past the boundary it ALSO takes the "today is done" path
     and returns TOMORROW's 09:30 ET — wait becomes ~86400s, clamped by
     `max(1, min(wait, 3600))` to exactly 3600. The log line proves it: current_phase had ALREADY
     ticked over to "market_open" by the time the log call re-read it microseconds later, but
     `wait_seconds=3600` had already been computed and committed — and the function sleeps
     UNCONDITIONALLY after logging, with no re-check before the sleep. The bug is a race between two
     independent fresh now_et() reads (one in wait_for_phase, one inside next_market_open()) and the
     wall clock crossing 09:30:00.000 between them — narrow, but guaranteed to be attempted at least
     once every trading day, at the one instant that matters most.
     BOUNDED, NOT BY DESIGN — BY LUCK. asyncio.sleep(3600) was entered at 09:30:00 ET. The ONLY
     recovery path is the supervisor's independent heartbeat-staleness check, and today's 2-minute
     health_check tick happened to land at 09:30:07 — 7 seconds after the open — and caught it via
     the heartbeat's age, not via any awareness of this bug. A tick landing at, say, 09:31:50 instead
     would have looked identical from the supervisor's side (heartbeat still stale, same alert), so
     the worst case is bounded by the health_check's own 2-minute cadence, not by anything in the
     buggy code path. RECOVERY ITSELF WAS ALSO NOT CLEAN: bot_stopping (09:30:07) -> bot_stopped
     (09:30:37) -> bot_already_running (09:30:37, a stop()/start() race — start() ran before the old
     process had actually released) -> bot_down exit_code=-9 (09:32:07, SIGKILL — it did not exit on
     SIGTERM alone) -> bot_restarting -> bot_started -> bot_launch_verified (09:32:15). Total outage:
     09:30:00 to 09:32:15, ~2m15s. A SIGKILL-required exit while nominally inside asyncio.sleep is
     itself worth a follow-up look — asyncio.sleep should cancel cleanly on SIGTERM.
     IMPACT TODAY: NONE on trading — entries were already frozen by [D2]'s self-reinstated halt
     before this fired. On a day WITHOUT that halt, this is ~2 minutes of zero entry/exit management
     at the open, bounded by the supervisor's independent safety net, not by any fix in this path.
     REGISTERED AS W24, NOT FIXED HERE — mid-check is not the place to make severity/urgency calls
     alone; flagged prominently to the operator instead. Two-lens satisfied: code read (both `<`
     comparisons, both fresh now_et() reads) AND the live log line, which is direct evidence, not an
     execution probe standing in for one.
     CURRENT STATE: stable since 09:32:15 ET, heartbeat fresh (checked 09:35:30 ET), no further
     restarts, bot pid 17922.

[D4] W2 — reconfirmed still clean straight through the open: zero `heartbeat_market_check_failed`
     / `fresh_models_market_check_failed` since the restart, including across the [D3] incident (the
     heartbeat CHECK itself worked correctly throughout; what failed was the trading loop's own entry
     into MARKET_OPEN, a different code path).

STATE AT HANDOFF: HEAD c309d31 on branch feature/event-straddle (plus this entry's commit).
Live bot: supervisor 7458 / bot 17922, restarted 08:37-09:32 ET (see [D3] for the mid-open restart).
Suite baseline unchanged at 1692 passed / 43 skipped / 1 known local-.env failure.
W22 (untracked-position mechanism) and W23 (scheduler timezone class) still registered, not started.
W24 (this entry's [D3]) is NEW and UNCLAIMED — live_path, fires at least once per trading day,
currently unprotected by anything except the supervisor's 2-minute heartbeat cadence.

## 2026-09-16 - W24 EXECUTED: the market-open race closed, deployed mid-day
Same-day turnaround from discovery to fix: the race W1's check caught live is now closed,
tested, and running.

[1] ROOT CAUSE, RESTATED PRECISELY. `wait_for_phase`, `get_current_phase` and
    `next_market_open` each independently called `now_et()` — up to five unsynchronized
    clock reads per loop iteration. Today's incident is the proof, not an inference: the
    bot's own log line showed `current_phase: "market_open"` (a late read) alongside
    `wait_seconds: 3600` (committed from an earlier read that still saw pre-market, whose
    branch treats "today's open already passed" as certain and hands off to
    `next_market_open()` for TOMORROW's open). No re-check before the `await`.

[2] FIX IS STRUCTURAL, NOT A PATCH OF THE ONE BRANCH THAT FIRED. `get_current_phase()` and
    `next_market_open()` both gained an optional `now: datetime | None = None` parameter —
    the SAME precedent already in this codebase (`orchestrator._post_stop_cooldown_cutoff`,
    W21). `wait_for_phase` now captures exactly ONE `now = now_et()` per loop iteration and
    threads it through every decision in that iteration: the phase check, both boundary
    comparisons, `next_market_open(now)`, and the log line's `current_phase`
    (`self.get_current_phase(now).value`, not a fresh call). An early return the instant that
    single read already satisfies the target means the contradictory branch can no longer
    execute in that case — not "less likely", structurally unreachable.

[3] BLAST RADIUS CONFIRMED NARROW BEFORE TOUCHING ANYTHING. Grepped `master.py` for all three
    function names: zero references. This lives entirely inside the bot's own process
    (`ait.main`), not the supervisor. All four existing no-arg call sites in orchestrator.py
    (`next_market_open()` x2, `get_current_phase()` x2) needed zero changes — the new
    parameters default to identical prior behavior. Two files touched:
    `ait/utils/time.py`, `ait/bot/scheduler.py`.

[4] TESTS — new file `tests/test_w24_market_open_race.py`, 16 tests, zero prior coverage of
    this module existed. The load-bearing one reproduces TODAY'S EXACT TWO OBSERVED INSTANTS
    (09:29:59.998, 09:30:00.071) via a queued fake clock and asserts the contradiction is now
    structurally unreachable: exactly one log line, `current_phase` and `wait_seconds` AGREE
    (pre_market / <=1s), converges in two single-read iterations with no hour-long sleep at
    any point. Plus: a boundary sweep (-100ms to the exact instant) proving no value near the
    boundary produces anything resembling 3600s; explicit-`now` unit tests for both helper
    functions including the exact-boundary instant (09:30:00.000000 -> MARKET_OPEN, not
    PRE_MARKET); a happy-path non-regression check (called from comfortably pre-open, still
    computes the genuine ~30 minute wait); a harness test proving the fake clock actually
    delivers staged values in order and errors on exhaustion rather than silently passing.
    ONE TEST BUG FOUND AND FIXED DURING WRITING, worth recording: a "no-arg still works"
    test for `next_market_open()` initially patched `now_et` in the WRONG module
    (`ait.bot.scheduler`, matching the `queued_clock` fixture's usual target) — but
    `next_market_open` is DEFINED in `ait.utils.time` and resolves `now_et` from ITS OWN
    namespace, so the patch had no effect and the test called the REAL wall clock. Caught by
    the test failing against real time rather than passing vacuously; fixed by patching
    `ait.utils.time.now_et` directly for that one case. Same class of lesson W21 kept paying
    for: know WHICH module's copy of an imported name you're actually patching.
    Suite: 1708 passed / 43 skipped / 1 known local-`.env` failure, up from 1692 — the 16 new
    tests, no regressions. Also re-ran the full orchestrator/scheduler/w20/w21/w24 cluster
    (124 passed, 1 skipped) before the full suite, for fast feedback.

[5] DEPLOY — mid-day restart, explicitly authorized by the operator for this fix. Same proven
    mechanism as this morning's W21 deploy: SIGTERM the supervisor (clean shutdown, ~27-30s,
    bot/dashboard/weblog/status all stopped in sequence), relaunch via `nohup`. New supervisor
    pid 24192, bot pid 24198. VERIFIED BEHAVIORALLY, not by grepping source: imported the
    actually-running modules fresh and (a) confirmed `get_current_phase`/`next_market_open`
    resolve the exact boundary correctly, (b) replayed TODAY'S EXACT incident sequence against
    the live-loaded code and confirmed no contradiction, no hour-long sleep — the same proof
    as the test suite, run again against what's actually deployed. File mtimes (15:50 host)
    confirmed well before the bot's start (16:08 host). Trading reverified (`trading_verified`,
    status Submitted). Zero errors/warnings since restart.
    A full supervisor restart was more than strictly required (only the bot subprocess's own
    modules changed; master.py never touches this code) but is the already-validated path from
    this morning and the cost difference is a few seconds.

[6] CONTEXT THAT DECIDED THE SEQUENCING, for the record. The operator asked directly whether
    W22 should still go first, since that was the standing ranking. Re-ranked W24 ahead of it
    for two reasons, not raw severity: (a) W24's "harmless today" status is fragile — it is
    covered only by `HALT_UNTRACKED`'s self-reinstatement, which ends whenever the operator
    closes the two manual TWS legs, on THEIR timeline, independent of any wave sequencing;
    (b) W24 was a small, fully root-caused, surgical fix (two files, additive-optional
    signatures) against W22's larger from-scratch surface (a new inbound Telegram channel that
    does not exist yet, new persistence, new UX flow). W22 remains next — nothing about doing
    W24 first argues against building it, only against building it before a smaller, currently
    live, trading-loop-entry defect with a diagnosed fix sitting there.

STATE AT HANDOFF: HEAD (this entry's commit) on branch feature/event-straddle, NOT PUSHED.
Live bot: supervisor 24192 / bot 24198, restarted 10:07-10:08 ET, stable, zero errors.
Suite: 1708 passed / 43 skipped / 1 known local-.env failure.
NEXT: W22 (untracked-position mechanism — inbound Telegram, ask-once, declared-manual
handling) is next in line. W23 (scheduler CronTrigger timezone class) and the register triage
remain registered, not started.

## 2026-09-16 - W22 EXECUTED + DEPLOYED: declared manual positions, proven live on the real legs
Built, tested, deployed mid-day (operator-authorized), and verified against the operator's
OWN two real untracked positions rather than a synthetic one — both prompts are sitting in
their Telegram right now with tappable buttons.

[1] THE MECHANISM. `ait/notifications/telegram.py` gains four HTTP primitives
    (send_with_buttons, edit_message, answer_callback, get_updates), same lazy-aiohttp-import
    style as the existing send(). New `ait/execution/manual_positions.py` owns the business
    logic — declare/reject/query against the existing bot_state KV table (no schema change,
    same convention as digest markers), the update dispatch, the long-poll loop, and the
    "confirms when it disappears" cleanup. Runs as its own asyncio task in main.py, decoupled
    from the trading loop's own shutdown semantics.

[2] REPLY UX, exactly as the operator chose: tappable inline-keyboard buttons, PLUS a
    plain-text reply-to-the-prompt-message as an explicit correction path at any time — their
    own follow-up question ("if I select the wrong choice, will I be able to correct it") is
    what shaped this. Both directions tested (button-then-text-fix, text-then-button-fix), and
    the edited message always shows whichever answer is CURRENT, not the first one.

[3] THREE DEFECTS REGISTERED LAST SESSION, CLOSED — and now LIVE-CONFIRMED, not just tested:
    (a) only the LAST untracked position reached HALT_UNTRACKED (per-iteration write_text
        clobbered itself). Today's real halt file, post-deploy:
          QQQ:712.0:C:2026-10-16
          QQQ:712.0:C:2026-09-18
        BOTH legs, in one file. The exact scenario that exposed this bug twice last session
        is now the exact scenario proving it fixed.
    (b)/(c) the sighting debounce (`_untracked_sightings`) was an in-memory dict, reset on
        every restart. Today's real bot_state: first sighting 11:44:11 ET (this deploy's
        first restart), second sighting 12:06:07 ET (the SECOND restart, a fresh process) —
        the ~22-minute gap was correctly measured ACROSS a real restart, which is precisely
        what the old code could never do.

[4] A FOURTH DEFECT, FOUND BY THIS WAVE'S OWN TEST SUITE, NOT REGISTERED BEFOREHAND — fixed
    before deploy, not after. The "write the halt file as a complete snapshot of this pass"
    design, combined with the PRE-EXISTING debounce's gap-since-the-LAST-call semantics
    (which W22 preserved as-is, not introduced), meant an ALREADY-CONFIRMED frozen position
    could transiently UN-FREEZE for a cycle if two reconcile() calls happened to land within
    the 2-minute debounce window — record_sighting resets last_seen on every call, deferred
    or not. Fixed with a reconciler-owned `is_confirmed`/`mark_confirmed` marker, deliberately
    independent of whether Telegram is configured or a send succeeds (INST-3 must not depend
    on that). A dedicated regression test proves two reconciles seconds apart after
    confirmation stay frozen. Caught by testing, before it could ever reach production.

[5] A FIFTH THING, ALSO CAUGHT BY THE SUITE: `record_sighting` crashed
    (`datetime.fromisoformat` on a non-string) against several PRE-EXISTING tests that
    construct `PositionReconciler` via `__new__` (bypassing `__init__`) with a bare
    `MagicMock()` state — a recurring, deliberate pattern in this codebase's own test suite
    (test_pnl_accounting.py, test_r15_review_fixes.py, others). Hardened to match
    `StateManager.get_state`'s actual contract (always returns str) rather than trusting a
    test double that doesn't honor it. 11 tests across those files were failing before this
    fix; all pass after.

[6] TESTS. 63 across three new files: the two interaction paths (buttons, text) as
    first-class SEPARATE cases per the operator's own request, the correction scenario in
    both directions, ambiguous/unambiguous bare replies, offset persistence across a
    simulated restart, the Telegram HTTP primitives against a faked aiohttp
    (sys.modules-level — telegram.py imports aiohttp lazily per-method, so there is no
    module-level attribute to monkeypatch the normal way), and the reconciler integration
    driven through the REAL reconcile() with the existing tests/fakes.py doubles (not a
    reimplementation of the logic under test).
    SUITE HAD A FALSE ALARM WORTH RECORDING: the first full run showed 2 failures in files
    never touched by this wave (test_r16_round2_fixes.py, test_r17_dormant_strategy_fixes.py).
    Isolated, they passed; combined with all three new W22 files, they passed; combined with
    each other's full files, they passed. A full rerun showed only the one known baseline
    failure. Flaky, not a regression — confirmed by reproduction attempts, not assumed.
    Final: 1757 passed / 43 skipped / 1 known local-.env failure (unchanged since W16).

[7] DEPLOY, mid-day, operator-authorized. Full supervisor restart (touches main.py's task
    wiring). Verified the RUNNING code, not disk: imported the live modules and confirmed the
    specific new functions/gates are present in the executing source. TWO restarts total —
    the first to deploy (produced the real first sighting on the operator's own two legs,
    11:44 ET), the second three minutes later at the operator's own choice specifically to
    prove the restart-survives-the-debounce fix live rather than waiting for a natural
    post-market reconcile (produced the real freeze + real Telegram prompts, message_ids 104
    and 105, both delivered successfully). This is the strongest verification this session
    has done: not a synthetic test position, not a mocked broker — the operator's actual
    account, actual Gateway connection, actual Telegram bot.

[8] WHAT'S LEFT, DELIBERATELY NOT DONE HERE: the operator has not yet tapped either button.
    That is their decision, not mine — I did not simulate a reply on their behalf. Once
    answered, `is_declared` will read it directly from bot_state; no further deploy needed.

STATE AT HANDOFF: HEAD (this entry's commit) on branch feature/event-straddle, NOT PUSHED.
Live bot: supervisor 44522 / bot (post-restart), stable, zero errors. Both real untracked
legs frozen and prompted; HALT_UNTRACKED correctly names both.
Suite: 1757 passed / 43 skipped / 1 known local-.env failure.
NEXT: W23 (scheduler CronTrigger timezone class — a DIFFERENT bug family from W24's race,
still unfixed: 9 host-local triggers, 2 naive catch-up run_dates) and the register triage
(109 R24 + 78 carried, deferred five waves now) remain the open backlog.

## 2026-09-16 - W22 FOLLOW-UP: two UX gaps found by actually using it, both closed same-day
Not new findings from review — the operator declared both real positions, then immediately
hit two rough edges that only using the feature (not reading the code) would surface.

[1] THE HALT DIDN'T LIFT ON ANSWERING. Both real positions were declared "yes" (12:17 and
    12:44 ET), but HALT_UNTRACKED only clears on the NEXT reconcile() pass — which under the
    original design meant waiting for post-market, tomorrow's pre-market, or another restart.
    `TradingOrchestrator.reconcile_now()` runs one pass on demand; the poller fires it (as a
    background task, never awaited inline) immediately after recording an answer. Kept
    deliberately decoupled: the poller still holds no broker reference of its own, so a slow
    or failed reconcile can never block it from processing the NEXT inbound update, and
    reconcile_now catches its own errors so a broken callback can't take the poller down.

[2] THE CONFIRMATION WASN'T VISIBLE. The operator tapped both buttons, then asked whether
    their reply was received — the actual answer (edited prompt text) was there, but it was
    genuinely easy to miss inside a message they weren't re-reading, and the Telegram
    answerCallback toast (required by the API regardless) is even easier to miss. Every
    recorded answer now ALSO sends a distinct new chat message — kept alongside the edit, not
    instead of it, since the edit is still useful as a self-documenting record at the prompt
    itself if you scroll back.

[3] A DISCREPANCY WORTH RECORDING HONESTLY, NOT PAPERING OVER: by the time this follow-up was
    ready to deploy, HALT_UNTRACKED had ALREADY cleared for both real positions — but the log
    shows no reconcile() ran between the second answer (12:44 ET) and the check (13:20 ET),
    and the bot process hadn't restarted (same pid). Neither this session's code nor a restart
    did it. Most likely a manual deletion of the file outside the bot, but that is inference,
    not confirmed — stated plainly rather than claimed as this fix's doing.

[4] TESTS: 17 new — the callback fires on both interaction paths (button, text-correction),
    is optional and never required (existing no-callback call sites unaffected), a failing
    callback doesn't break answer recording, the confirmation message is distinct from the
    edit and sent FRESH on every correction, not just the first answer. Full suite: 1764
    passed / 43 skipped / 1 known local-.env failure — confirmed across two full runs (one
    per fix layered on), no regressions either time.

[5] DEPLOY, same-day, same mechanism (full supervisor restart, mid-day, operator's standing
    authorization for today). Verified the running code behaviorally. Startup reconcile
    correctly read BOTH positions as declared_manual_position_seen — the is_declared() gate
    from the original W22 deploy, still working, now paired with reconcile_now for anything
    NEW. Nothing left to prove live against these two specific positions — both are already
    fully resolved; the new behavior applies to whatever the next untracked position is.

STATE AT HANDOFF: HEAD (this entry's commit) on branch feature/event-straddle, NOT PUSHED.
Live bot: supervisor 57775 / bot 57777, restarted 13:21 ET, stable, zero errors.
Suite: 1764 passed / 43 skipped / 1 known local-.env failure.
NEXT: unchanged — W23 (scheduler CronTrigger timezone class) and the register triage.

## 2026-09-16 - MERGE + W22 HARDENING: an unresolved anomaly, cross-session-investigated, closed as best it can be
Two unrelated things landed in the same stretch: reconciling this branch with independent work
pushed under the repo owner's own identity, and a real investigation into a live-trading safety
file that vanished with no explanation anyone could find — three sessions looked, including two
peer Claude sessions on this same machine (trade-v2-c6, trade-v2-1b), and it stayed unsolved.

[1] THE MERGE. `git push` was rejected — 4 commits existed on origin that weren't local
    (`57ba9aa`..`f09cc38`, a take-profit-ladder backtesting study, 403 files, mostly generated
    report data). Verified before merging, not assumed: every commit authored by
    `Prithvi Pradip <prithvi.pradip@gmail.com>` (the repo owner) — NOT either peer session running
    on this machine, an early guess that was wrong and corrected once checked. File-level diff
    showed exactly ONE overlapping path across all 403+31 changed files: `PLAN.md`, which is
    append-only by design for precisely this situation. `git merge` resolved clean — verified by
    diffing both parent tips against the merge result, confirming both sides' full content
    survived with correct chronological placement, no conflict markers anywhere. Full suite on the
    combined tree: 1793 passed / 43 skipped / 1 known failure (up from 1764, matching their new
    tests) — pushed only after that came back green, not before.

[2] THE ANOMALY. `data/HALT_UNTRACKED` (INST-3's live freeze flag) was confirmed present and
    actively blocking entries at 12:51:21 ET (`entries_frozen`, firing on a steady ~5-min cadence)
    and confirmed ABSENT at 13:20:11 ET. In that 29-minute window: zero `reconcile()` calls (the
    ONLY code path anywhere in the tree that touches this file — confirmed by full-repo grep, not
    assumed), zero bot restarts, and every one of the 150 log lines that fell inside the window
    checked individually — nothing touches it. Ruled out, each independently verified rather than
    assumed: every other code path in the repo (dashboard, status server, weblog, generic cleanup
    routines — `cleanup_old_logs` only ever touches `logs/`/`reports/`, never `data/`); Finder/GUI
    deletion (not in `~/.Trash`, which a programmatic `unlink()` never populates anyway); iCloud/
    Dropbox sync (no markers); this session's own test runs (different process, can't touch
    another process's files); the operator, directly, by their own account.
    TWO PEER SESSIONS EACH PROPOSED A MECHANISM. Both were checked against the precise timeline,
    not accepted on description alone, and both failed the check:
      - trade-v2-1b's first theory credited the restart at 13:21:32 ET with clearing the file —
        but the file was independently confirmed gone at 13:20:11 ET, a full 74 seconds BEFORE the
        SIGTERM that led to that restart was even sent. Wrong event, right mechanism description.
      - trade-v2-1b's second theory (on_answer's asyncio.create_task() being silently GC'd,
        meaning reconcile_now() never fired for any of four Telegram answers in the gap) was
        checked against git history directly: `_record_and_edit`'s signature had no `on_answer`
        parameter at all until commit `8d963c7`, timestamped 13:20:04 ET — after all four answers
        (12:17, 12:44, 13:08 x2). The wiring that could have failed this way did not exist yet.
        Retracted by its own author once shown the diff.
    One genuinely separate, now-fully-resolved side thread: a "no" then "yes" toggle at 13:08:25-27
    ET on the 2026-09-18 leg's prompt — confirmed by the operator as a deliberate test of the
    correction UX, unrelated to the file (that path only writes bot_state and edits a Telegram
    message; never touches HALT_UNTRACKED).
    STATUS: genuinely unresolved. Not "probably fine" — actually unexplained, stated plainly. What
    makes it tolerable rather than urgent: the STATE it left behind was correct (nothing needed
    freezing; both real positions were legitimately declared), and the mechanism enumeration above
    is exhaustive enough that a repeat is unlikely to hide the same way twice.

[3] TWO DEFENSIVE FIXES, neither explaining [2], both worth doing regardless:
    (a) `manual_positions.py`'s `asyncio.create_task(on_answer())` discarded its return value —
        a real, independent footgun (the event loop holds only a weak reference; an unreferenced
        task can be GC'd before running, per asyncio's own docs) that happened to not be the cause
        THIS time only because of when it shipped, not because it's safe. `_fire_and_forget()` now
        holds every background task in a module-level set with a done-callback.
    (b) `reconciler.py`'s halt-file write AND clear each get a dedicated log line now
        (`halt_untracked_written` / `halt_untracked_cleared`, both with the affected keys) instead
        of only the general `reconciliation_complete` summary — so a recurrence is a log lookup,
        not a repeat of today's multi-session investigation.
    4 new tests. Suite: 1797 passed / 43 skipped / 1 known failure, up from 1793.

[4] CROSS-SESSION COORDINATION, worth recording as its own data point. Three Claude sessions
    (this one, trade-v2-c6, trade-v2-1b) share ONE working directory on this machine, not separate
    clones — confirmed by trade-v2-1b directly. Messaged both peers this session; both responded
    with real, checkable evidence (not just assertions), both self-corrected when shown a timeline
    that contradicted their own theory, and both explicitly retracted rather than defended once
    checked. Two attribution mistakes were mine, not theirs — corrected in both directions.

STATE AT HANDOFF: HEAD (this entry's commit) on branch feature/event-straddle, PUSHED — origin and
local in sync as of this session (confirmed both directions, 0 commits ahead/behind either way).
Live bot: supervisor 70821, restarted 14:35 ET, stable, zero errors, both hardening fixes verified
running (not just on disk).
Suite: 1797 passed / 43 skipped / 1 known local-.env failure.
NEXT: W23 (scheduler CronTrigger timezone class) and the register triage — unchanged by any of
this. The HALT_UNTRACKED anomaly in [2] is not a wave; it's a registered open question with no
concrete next action beyond [3]'s logging, which is already done.

## 2026-09-16 - W23 EXECUTED + DEPLOYED: the scheduler's silent timezone fallback, closed
Confirmed live at 09-13-16 09:35 ET (digest_am firing at 03:35 ET), registered across three
sessions, finally probed to precision and fixed today.

[1] THREE BUGS, NOT ONE — each confirmed by direct probe against this box's real apscheduler
    3.11.2, not assumed from the API surface, and each needing a DIFFERENT fix:
    (a) Nine pre-built `CronTrigger(...)` objects never inherited the scheduler's own
        `timezone="US/Eastern"` — apscheduler only injects that when IT builds a trigger from an
        alias string ("cron"); a pre-built object keeps `get_localzone()` at construction time
        (probed: Europe/Madrid, this host). Fixed with explicit `timezone=ET` on all nine.
    (b) The two naive `run_date=` catch-up jobs are a DIFFERENT mechanism, easy to conflate with
        (a) but confirmed distinct: a naive datetime through the `"date"` alias is MISLABELED
        with the scheduler's offset, not converted — probed fresh today, naive `20:38:13` (real
        host wall clock) came back labeled `20:38:13-04:00`, an instant 6h in the future.
        Live-evidence match: `retrain_catchup` registered `21:31:45 EDT`, ran `03:31:45` local.
        An AWARE `run_date` passes through byte-for-byte unchanged — probed and now a permanent
        regression test, not just a plan-time claim.
    (c) `_backtest_catchup_run_at`'s own market-hours-avoidance check compared naive-host-local
        hours against ET thresholds (9:15/16:30) — the wrong clock, independent of what the
        scheduler later did with the result.

[2] DELIBERATELY UNCHANGED, and the reason is now load-bearing, not incidental:
    `_backtest_catchup_due`'s naive-local `now` stays naive — its own prior docstring already
    explained this is intentional (compared against equally-naive file mtimes; a naive
    Sunday-20:00 slot is immaterial for a coarse weekly gate). Fixing (b)/(c) means the SAME
    catch-up now passes TWO deliberately different clocks to two sibling functions — the
    due-check keeps naive `datetime.now()`, the run-at computation gets a fresh `now_et()`. A
    comment at the call site says so explicitly, and a dedicated test
    (`test_call_site_uses_two_different_clocks_not_one`) checks the SOURCE keeps them separate —
    the exact kind of "helpful simplification" that would silently reintroduce (c).

[3] THE RETRAIN CATCH-UP FIX REUSED AN ALREADY-AWARE CLOCK RATHER THAN READING A NEW ONE.
    `_now_c` (the retrain due-check's own clock) was ALREADY `now_et()` under a local alias —
    only the separate `_rt_at = datetime.now() + timedelta(minutes=2)` was naive. Fixed by
    reusing `_now_c` for the new `_retrain_catchup_run_at(_now_c)` call instead of reading the
    clock a second time — one `now` per decision, the same discipline W24 established for the
    market-open race.

[4] TESTS — new file, 14 tests, zero prior coverage of this exact class existed (though
    `_backtest_catchup_due`/`_backtest_catchup_run_at` had none dedicated either before today).
    An AST scan of `main()`'s source catches a future 10th `CronTrigger` site missing the fix —
    this codebase's own established rule (W20's `hour_of_day`, W21's 19-vs-18 reader count were
    BOTH found this way, not by re-checking known-fixed sites). Real execution against a real
    `BlockingScheduler`, not just the keyword's presence: `daily_report`'s next fire time,
    computed from a known instant, lands on 16:30 ET, not 10:30. NEGATIVE CONTROLS proving the
    underlying bugs are real, not just that the fix's shape looks plausible — the same
    `CronTrigger(...)` call WITHOUT `timezone=` resolves to something other than `US/Eastern`;
    the same naive `run_date` through the same alias comes back mislabeled, reproducing the
    exact 6h-late live incident as a permanent regression test. Both market-hours boundary
    instants (9:15 and 16:30 ET) exercised directly, not just interior points.
    Suite: 1811 passed / 43 skipped / 1 known local-.env failure — up from 1797, no regressions.

[5] DEPLOY. Full supervisor restart (master.py IS the supervisor). Clean shutdown (~31s), clean
    startup, zero errors. VERIFIED BEHAVIORALLY on the loaded module, not source-read: imported
    the live `ait.orchestration.master`, reconstructed the exact `CronTrigger(..., timezone=M.ET)`
    calls `main()` uses and confirmed `.timezone == US/Eastern`; called the live
    `_retrain_catchup_run_at` and confirmed it returns an aware datetime; confirmed the source of
    the loaded `main` function contains the aware `now_et()` call at the backtest catch-up site.
    File mtime confirmed older than the process start. The `job_registered` log line's trigger
    repr does not surface timezone (a logging-format limitation, not a verification gap — the
    behavioral check above covers what that log line can't show).

STATE AT HANDOFF: HEAD (this entry's commit) on branch feature/event-straddle.
Live bot: supervisor 77009, restarted 15:11 ET, stable, zero errors.
Suite: 1811 passed / 43 skipped / 1 known local-.env failure.
NEXT: the register triage (109 R24 + 78 carried) is now the only major item left unclaimed —
deferred through W17, W18, W19, W20, and effectively every wave since. The 10 remaining batch-3
fail-open sites are next after that, with one flag: #2 (reconciler.py, the HALT_UNTRACKED
freeze-file claim) needs RE-VERIFICATION before probing — W22 rewrote that entire code path
today, so the original finding may be stale, moot, or still real in a different form; not
re-checked yet.

---

## REGISTER TRIAGE BATCH 1 — the 20 live_money findings, re-verified against today's source

The last major unclaimed item, deferred through W17, W18, W19, W20 and effectively every wave
since. Pure triage: no source file was modified. The register lives in
`reports/register_triage_20260916.md` (a living document, appended per batch) rather than here,
because a 206-row table does not belong in a 4,000-line ledger.

[1] SCOPE, MEASURED BEFORE PLANNING rather than assumed. Universe = **206**: R24's own 109
    standing findings + 78 carried-over OPEN from the prior register + 19 R24-marked PARTIAL.
    HONEST COUNT DISCREPANCY, stated rather than absorbed: R24's header says it triaged "123
    prior findings" but its triage section holds **126** rows (78 OPEN + 19 PARTIAL + 28 FIXED
    + 1 OBSOLETE). Three rows unaccounted for. NOT resolved; carried forward.

[2] WHY BATCHED, and this is the number that shaped the whole approach. Only 22 of 109 R24
    findings are mentioned anywhere in PLAN.md after R24 was written; ZERO of the 78 carried-over
    ones are; only 17 of 109 carry a fix-marker comment in `src/` or `tests/`. So **165 of 187
    have no trace at all** across ten days and ~18 waves. There is no grep shortcut — the
    fix-marker convention is real but is a LOWER bound on what's closed, never an upper one.
    Closing 206 in one pass would either burn many sessions with nothing durable in between or
    silently degrade into skimming, which is worse than not doing it: a wrongly-closed finding
    is a live defect marked safe.

[3] THE METHOD ASYMMETRY, which is what makes the output trustworthy. Marking OPEN is cheap —
    re-read the cited `Where:` and see whether the claim still holds. Marking CLOSED must be
    expensive: the fix in code AND a test that pins it. A wrongly-OPEN finding costs a re-check;
    a wrongly-CLOSED one retires a real defect in silence.

[4] RESULTS — ALL 20 ASSESSED: 12 CLOSED, 2 CLOSED (untested), 6 OPEN.
    CLOSED (12): concurrency-01, concurrency-03, regress-live-01, regress-live-04, pnl-money-01
    (all W14, each with a dedicated test section); pnl-money-02 (W11 — the drain path now calls
    the same `_true_up_commission` at orchestrator.py:729/:813 that the finding said it lacked);
    time-calendar-01 (W15 primary path uses the real NYSE calendar; W20c replaced the fallback's
    `weekday() < 5` heuristic — the exact claimed defect — with `trading = False`, erring toward
    MORE blocking, so both directions of the claim are gone); time-calendar-02 and the overlapping
    regress-live-07 (the bare `except Exception: pass` is gone; the caller now logs
    `post_stop_cooldown_check_failed_fail_closed` at ERROR and fails CLOSED — one fix, and it
    closes both claimed failure surfaces: the calendar raise and the sqlite busy window);
    logic-exit-risk-01 and logic-exit-risk-02 (W7 — the `entry_credit + 0.25*wing_width` term that
    shipped SPY as BUY LMT 14.04 against a 1.40 mark is gone from BOTH the fallback and the quoted
    anchor, replaced by `_exit_price_ceiling` anchored to current worth; the broker band is now
    parsed and honoured rather than treated as a cancel. `tests/test_w7_exit_pricing.py` replays
    the real incident numbers and the VERBATIM IBKR warning text, and carries a
    `TestTheIncidentCannotRecur` class); ops-substrate-01 (W13, which has its own ledger entry at
    PLAN.md:1977 — the bot-down branch now calls `_maybe_recover_gateway` (master.py:882) under a
    comment naming the finding, so the absorbing state is gone; three test files pin it).
    CLOSED (untested) (2): concurrency-02 and its true duplicate regress-live-02. A NEW status added
    mid-batch because the binary didn't fit reality: `_claimed = True` no longer exists as
    executable code anywhere in `src/` (the sole remaining occurrence is inside a comment
    describing the old behavior), so the defect is structurally unreachable — a stronger
    guarantee than a test — but NO test exercises the claim-raises path, so an edit could
    reintroduce it unnoticed. Recorded as a known regression risk, not as a fix task.
    OPEN (2), both re-read and confirmed still true today:
      - data-integrity-07 — `stats.trades_taken += 1` still fires on every order PLACED
        (orchestrator.py:2695) with NO decrement anywhere; the symbol appears at only four sites
        and has ZERO references in reconciler.py, so the stale-pending sweep still cannot return
        the budget. The intraday entry budget remains consumable by phantom trades.
      - regress-live-06 — `_credit_floors_hold` (executor.py:1411) is invoked ONLY at reprice
        (:1390). A grep across the entire `_execute_multi_leg` initial-placement path returns no
        floor check at all, exactly as claimed.
      - params-trading-01 — THE SHARPEST FINDING IN THE BATCH, all three sub-claims intact, and
        one of them deserves to be read on its own: `manager.py:615` is still
        `final_contracts = max(1, int(final_contracts * haircut))`. When the sizer has already
        returned 0 contracts and `correlated_count > 0`, that evaluates to `max(1, 0) = 1` and the
        `if final_contracts <= 0` refusal on the very next line never fires. A FAIL-CLOSED
        0-viability verdict is silently converted into a live 1-contract position — and only when
        a correlated position is already open, i.e. exactly when it is least wanted. Also intact:
        the sizer's hard cap at position_sizer.py:166, and the fact that
        _launch_size_coherence_check never reads max_position_pct at all.
      - logic-entry-01 — the same self-test, failing the same operator. It still models
        `min_width = max(1.0, min_credit / 0.45)` (the $1.56 wing the real builder never
        constructs) and still never executes IronCondor.generate_signals, so it can still report
        COHERENT for budgets that deterministically emit zero signals.
      - logic-exit-risk-03 — primary claim stands: the cooldown query is still `WHERE symbol=?`
        (orchestrator.py:2244) with no cluster scoping, so a SPY touch stop still leaves QQQ and
        IWM free to enter the same move. PARTIAL CREDIT recorded precisely: its secondary half
        (the adjacent silent fail-open) is CLOSED by the same fix that closed time-calendar-02.
      - docs-policy-04 — OPEN IN A CHANGED FORM, the most interesting result of the batch. The
        RUNBOOK (:41-42) still tells the operator to "stop the bot processes" and still names no
        supervisor. `data/HALT` first genuinely fixes the ENTRY side, but the claimed hazard is
        the EXIT engine and startup reconcile returning mid-close, and HALT by the RUNBOOK's own
        wording does not stop those ("exits keep managing"). What MOVED is the mechanism: W16b
        retired the POSIX keeper and the host is macOS, so the cited `keeper_ait.bat` is inert —
        but master.py IS the supervisor now and `combined_health_check` runs every 2 MINUTES
        (master.py:1959) calling `bot.health_check()` -> `self.start()`. Same hazard, ≤2-minute
        window, different component than the finding names — and RUNBOOK:93 still offers
        `keeper_ait.bat` as a restart path, so the doc is now wrong in a second way too.

[5] VERIFICATION — spot-checked in the DANGEROUS direction, per the wave plan. For three CLOSED
    rows the bar was not "a test exists" but "the cited test would FAIL against the pre-fix
    behavior." All three turned out to be built as negative controls with the defect named in
    the assertion itself: `assert n == 1, "claim destroyed the obligation (the pre-W14 defect)"`;
    `assert ... _consecutive_losses == 2, "streak was reset"`; and
    `assert recovered._consecutive_losses == 2, "the streak was destroyed"` sitting directly
    below `crashed._persist()  # pre-W14: wrote zeros`. W14 file: 15 passed.
    Count reconciliation: 20 batch-1 rows + 186 untouched = 206 nominal IDs; within the batch,
    20 assessed = 12 + 2 + 6, none deferred. The EFFECTIVE universe is 205 distinct defects
    (concurrency-02/regress-live-02 being one defect under two IDs), and more such pairs are
    likely among the untouched 186.

[6] FINDING THAT CHANGES THE UNIVERSE: **the register contains duplicates** — and a SELF-CORRECTION
    made before this entry was committed, recorded because the error is instructive.
    I first wrote that this batch held TWO duplicate pairs. Re-reading both pairs in full against
    R24 showed only ONE is real: concurrency-02 / regress-live-02 share the same `Where:`
    (orchestrator.py:739-742, `_claimed = True`), the same claim, and the same probe shape — a true
    duplicate under two IDs. time-calendar-02 / regress-live-07 are NOT: they share the caller
    (the bare `except Exception: pass` at :2345-2352) and one root fix closes both, but they claim
    DIFFERENT failure surfaces — regress-live-07 the calendar-library raise plus the
    cache-only-on-success recurrence; time-calendar-02 the calendar raise AND the sqlite
    busy/locked window AND a malformed exit_time. R24's own interaction analysis treats them as
    distinct protections. Overlapping, not duplicate.
    The lesson generalizes and is worth more than the count: the method asymmetry in [3] was
    written to guard against over-calling CLOSED, but "duplicate" is the same hazard wearing
    different clothes — collapsing two findings into one retires a claim nobody re-checked. A
    dedup pass is still worth doing, and it must compare `Where:` AND `Claim:` in full, never a
    shared code location. The CLOSED status of all four IDs is unaffected: it rested on the code
    and the tests, not on the duplicate call.
    SECOND FINDING, with its own caveat: 11 of 13 assessed came back CLOSED. That rate does NOT
    generalize — this batch is the highest-severity slice, which is exactly where the fix waves
    concentrated. Expect the research_validity and cosmetic tiers to close far lower.

STATE AT HANDOFF: no source modified; `reports/register_triage_20260916.md` added.
Suite: 1811 passed / 43 skipped / 1 known local-.env failure — byte-for-byte the W23 baseline,
which is the point: this wave read code and wrote a document, and the unchanged suite proves it.
Cited-test spot-runs: test_w14_booking_exactly_once 15 passed; test_w7_exit_pricing +
test_w13_gateway_recovery 34 passed.
NEXT, chosen from the re-ranked tail rather than assumed, per [C4]. Batch 1 did its job: it
turned a 20-item unknown into SIX confirmed live-money defects, so the next move is a FIX wave,
not more triage.
  (a) W25 — THE GO-LIVE SIZING WAVE, and this outranks everything else because the operator is
      pre-funding. params-trading-01 + logic-entry-01 are ONE wave, not two: both fail at
      `_launch_size_coherence_check`, the self-test the go-live recipe explicitly tells the
      operator to trust before funding. Ship manager.py:615 `max(1,...)` -> `max(0,...)` (or skip
      the haircut when size.contracts == 0), teach the coherence check to read max_position_pct,
      and make it exercise the REAL builder instead of the `min_credit/0.45` model wing.
      Non-negotiable per this session's standing rule: a NEGATIVE CONTROL proving the pre-fix
      code really did turn a 0-contract verdict into 1, and an executing test at the documented
      $2.1k/7% recipe.
  (b) Then data-integrity-07 (bounded: give the budget back in the stale-pending sweep) and
      logic-exit-risk-03 (scope the cooldown to SECTOR_GROUPS, one authority shared with
      correlation.py). regress-live-06 is the most delicate of the six — it touches
      initial-placement pricing on the live path — so it wants its own wave with a live probe.
  (c) docs-policy-04 is a doc fix but NOT a trivial one: the RUNBOOK must name master.py's
      2-minute health_check as the thing to stop, and drop the stale keeper_ait.bat path. Cheap,
      and it protects a manual close.
  (d) Batch 2 of the triage = the next tier down (the remaining live_path findings), plus the
      dedup pass from [6]. Now genuinely lower priority than (a)-(c).
  Still open from before, unchanged: the 10 remaining batch-3 fail-open sites, with #2
  (reconciler.py / HALT_UNTRACKED) still needing re-verification because W22 rewrote that path.

---

## REGISTER TRIAGE BATCH 1 — CORRECTIONS + LIVE STATE (same session, appended not edited)

Three gaps found by re-reading the entry above as a COLD READER would — the test that matters,
since the project convention is that a new session starts from PLAN.md's last entries, not from
a summary. Appended rather than edited, per the append-only rule.

[C1] INTERNAL CONTRADICTION in the entry above, and the numbers to trust.
     PLAN.md:4040 says "ALL 20 ASSESSED: 12 CLOSED, 2 CLOSED (untested), 6 OPEN" — CORRECT.
     PLAN.md:4129 says "11 of 13 assessed came back CLOSED" — STALE. It survived from the point
     where the batch was planned to stop at 13 findings; the scope was then extended to all 20
     and that one sentence was not updated (the parallel sentence in
     `reports/register_triage_20260916.md` WAS). The correct figure is **14 of 20** came back
     CLOSED or CLOSED (untested).
     The caveat attached to it still stands and still matters: that rate does NOT generalize.
     This batch was the highest-severity slice, which is exactly where the fix waves
     concentrated; expect the research_validity and cosmetic tiers to close far lower.

[C2] LIVE BOT STATE — omitted above; prior entries carry it and a new session needs it.
     Verified 2026-09-16 16:29 ET, after the close:
       - supervisor PID 77009 (`run_orchestrator.py`) and bot PID 77028 (`python -m ait.main`)
         both alive — the same supervisor W23 started at 15:11 ET, so it has run a full session
         without a restart.
       - NO kill-switch files present: `data/HALT`, `data/HALT_UNTRACKED` and
         `data/RESTRICTED.txt` all absent. Entries are NOT frozen. (HALT_UNTRACKED stays lifted
         because the operator's manual TWS positions are declared — W22.)
       - `data/bot_heartbeat` last written 15:59:55 ET, i.e. 30 min stale at the time of
         writing. This is EXPECTED, not a fault: the market closed at 16:00 ET and the trading
         loop legitimately stops. `ops_health.bot_liveness` encodes exactly this, returning
         `ok=True, state="not_rth"` when the market is closed.

[C3] THE OPEN WATCH LIST, carried forward — the entry above dropped it, which would have lost it.
     - W1 (heartbeat liveness): **HEALTHY**, and checked in the dangerous direction rather than
       assumed. The heartbeat FILE stores a naive host-local ISO string
       (`orchestrator.py:1073` writes `datetime.now().isoformat()`), which is superficially the
       exact bug class W20/W21/W23 spent this session fixing — on this host naive local is ET+6,
       so a consumer parsing that text would be 6 hours wrong. It is NOT a bug: all three
       consumers read **mtime**, never the text. `ops_health.py:642` uses
       `now.timestamp() - hb.stat().st_mtime` (epoch, timezone-proof); `master.py:1505` and the
       dashboard both use `datetime.now() - datetime.fromtimestamp(st_mtime)` (naive minus naive,
       internally consistent). The string in the file is decorative. Recorded here so the next
       session does not re-open it, and does not "fix" the writer into a real inconsistency.
     - W2 (the two calendar-fault warnings must NEVER appear): **CLEAN**, zero occurrences in
       `logs/ait.log`.

STATE AT HANDOFF (supersedes the one above): HEAD = this entry's commit, branch
feature/event-straddle, pushed. No source modified by the triage. Live bot up, entries unfrozen.
Suite 1811 / 43 skipped / 1 known local-.env failure.
NEXT: unchanged from the entry above — (a) W25 the go-live sizing wave
(params-trading-01 + logic-entry-01, one wave, both failing at `_launch_size_coherence_check`),
then (b) data-integrity-07 and logic-exit-risk-03, regress-live-06 in its own wave,
(c) docs-policy-04's RUNBOOK fix, (d) triage batch 2 + the dedup pass.

## 2026-09-16 - W25 EXECUTED — the go-live sizing wave (params-trading-01 + logic-entry-01 + one new)

The first FIX wave off register triage batch 1, taken in the order the corrections entry left:
(a) W25. Both named findings fail inside `_launch_size_coherence_check`, the self-test the go-live
recipe tells the operator to run and trust before funding at ~$2.1k, so they were taken as ONE
wave. Register-don't-fix does not apply here. A third defect in the same function was found while
probing the second and is fixed in the same wave.

[1] params-trading-01 — CONFIRMED OPEN BY BOTH LENSES, and the register's mechanism is WRONG.
    The code lens had already conceded on 09-16 by reading `manager.py:615`. The execution lens
    now concedes too: a probe drove the REAL `validate_trade` with the REAL `PositionSizer`, REAL
    `CorrelationGuard` and REAL `config.yaml`, stubbing only the account seam. Negative-control
    pair, identical requests differing ONLY in whether a correlated position is open:
      - nothing open  -> approved=False, "position sizer returned 0 contracts"
      - SPY open      -> approved=True, position_size=1, **max_risk=0.0**
    Account-size independent (reproduced at $2.1k and at $10k).
    WHAT THE REGISTER GOT WRONG, stated rather than absorbed: it claims this fires at $2.1k via the
    `max_position_pct` route. It does not. Gate 6b (`manager.py:512`, risk.max_position_risk_pct
    3% = $63) rejects BEFORE sizing, and since 3% < the sizer's 5% that masking holds at EVERY
    account size whenever `max_loss` is set. Reaching :615 requires a FALSY `max_loss`, and
    `iron_condor.py:318-322` refuses to emit a condor with `max_loss <= 0`. So on today's IC-only
    live path the defect was **latent, not active** — a fail-open contained by two unrelated gates
    rather than by design. It is still worth closing: the containment is incidental, and the
    second-order half (an approved position reporting max_risk=0.0, invisible to the aggregate cap
    at :529) is real wherever it does fire.
    RANKING, since the ledger asked for disagreement: W25 still goes first, but NOT for the reason
    (a) gave. params-trading-01 is latent today; logic-entry-01 is ACTIVELY WRONG right now, at
    exactly $2.1k, inside the one self-test the funding decision rests on. That is what earns the
    ranking; params-trading-01 rides along in the same blast radius.

[2] THE LEDGER'S SUGGESTED FIX WAS WRONG, and the test caught it rather than the reasoning.
    (a) said ship `max(1,...) -> max(0,...)`. That was implemented, and the "haircut must still
    reduce a viable size" test failed immediately. Measured: with
    `positions.max_contracts_per_trade = 1` the sizer returns 1, `int(1 * 0.70) = 0`, so `max(0,...)`
    REFUSES every entry made while one correlated position is open — and the universe is
    SPY/QQQ/IWM, a single cluster, so the bot would have gone flat after its first fill. A silent
    trading-behavior change shipped as a fail-open fix. This is the wave's most useful result and
    the reason the negative control is non-negotiable.
    SHIPPED INSTEAD: guard the branch — `if correlated_count > 0 and final_contracts > 0:`. The
    rule a haircut must obey is that it may only REDUCE an already-viable size: never manufacture
    one from a refusal, never turn a viable size into a refusal. Live sizing behavior is
    byte-identical to pre-W25 for every viable size; only the 0-verdict path changes.

[3] logic-entry-01 — FIXED, and the check was wrong in a THIRD way nobody had registered.
    - It modelled the wing as `max(1.0, min_credit / 0.45)` = $1.56, a structure the real builder
      never constructs. Now calls the REAL authority, `IronCondor._vol_scaled_width`
      (`iron_condor.py:76`), which is fully duck-typed and runs at startup without a live chain.
    - It read only `risk.max_position_risk_pct`. The sizer independently returns 0 contracts on
      `positions.max_position_pct` (`position_sizer.py:166`), so the check could report COHERENT
      while the OTHER cap bricked every entry. Now binds on the tighter of the two and names which.
    - NEW, and this is what the $1.56 model was accidentally hiding: affordability is not
      viability. At $2.1k the real builder caps the wing at $1.00, which can yield ~$0.45 credit
      against a $0.70 floor, so `iron_condor.py:293` rejects EVERY signal — the bot emits ZERO
      trades. The old check compared dollars only and never asked this. The first version of this
      fix swapped the width source alone and reported COHERENT at $2.1k, which is the exact failure
      logic-entry-01 describes; caught by running it, not by reading it. The credit-floor and
      credit/width arms are now separate checks against the real builder's width.

[4] D3, NEW FINDING, found while probing [3]: the breaker arm was ARITHMETICALLY DEAD.
    `AIT_CREDIT_LOSS_LIMIT` contract-defaults to `"0"` = credit stop DISABLED (`runtime_env.py:85`,
    `portfolio.py:230`), so `typical_stop_loss = 0` and `breaker_usd < 1.5 * 0` was False for every
    account that has ever existed. The arm meant to catch "one stop-out halts the whole day" could
    never fire — at $2.1k, where the breaker is $42, that is precisely the regime it exists for.
    FIXED honestly rather than loudly: the first attempt reported "cannot be assessed" and alerted
    at EVERY NLV including the paper account's $197k, which is alert fatigue, not a fix. With the
    credit stop off, the honest bound on one trade's loss is the structure's DEFINED-RISK max loss
    — nothing else closes the position early — so the arm now assesses against that. It fires at
    $2.1k ($42 breaker vs $30) and $5k, and is silent at $10k and above.

[5] VERIFICATION. `tests/test_w25_sizing_coherence.py`, 28 tests, all green.
    - NEGATIVE CONTROL: feeds the REAL sizer's REAL 0-verdict into the pre-W25 expression and
      asserts it yields 1, with the defect named in the assertion. It asserts the PREMISE first
      (`verdict.contracts == 0`), so it fails loudly if the sizer ever stops returning 0 there
      instead of quietly testing nothing.
    - SECOND NEGATIVE CONTROL, added after [2]: pins why `max(0,...)` was rejected, so nobody
      "simplifies" the guard back into a trading halt.
    - CLASS GUARD, behavioral not syntactic: over the grid (sizer verdict 0/1/2/5 x correlated
      0/1/2), `approved => 1 <= position_size <= sizer_verdict` and `verdict == 0 => refused`.
      An AST guard was written FIRST and REJECTED — recorded because it is instructive: `qty =
      max(1, ...)` is also the shape of `portfolio.py:963` (flooring a partial CLOSE at one
      contract, correct) and `manager.py:172` (a divisor normalizer). The defect is about entry
      sizing, not syntax, so a syntactic guard produces false positives and would have been
      silenced or deleted by the next session.
    - Gates: full fast lane **1839 passed / 43 skipped / 1 failed** — 1811 baseline + 28 new, and
      the single failure is the known local-`.env` `test_r19_config_authority` case, unchanged
      since W16. `compileall` OK. `scripts/typecheck.py` OK. `scripts/smoke_deploy.py` SMOKE PASS
      (13 checks).

[6] REGISTERED, NOT FIXED (operator decision this session), in
    `reports/register_triage_20260916.md` under "W25 ADDENDUM": the cap-grid skew.
    `risk.max_position_risk_pct` (3%, gate 6b) and `positions.max_position_pct` (5%, the sizer)
    govern the same quantity through two unlinked knobs, and their ordering is the ONLY thing that
    kept params-trading-01 off the live path. Raise the first above the second and the fail-open
    would have become live on the normal IC path with no other code change. W25 removed the
    fail-open, so the hazard is closed today; the entry exists so a future tuning wave knows what
    it is standing on. Also registered there: `TradeValidation.max_risk` is sourced from a sizer
    result that may not describe the approved size.

STATE AT HANDOFF: branch feature/event-straddle. Live bot re-checked and UNCHANGED throughout —
supervisor PID 77009 / `ait.main` PID 77028 both alive, `ait.log` writing, no `data/HALT`,
`HALT_UNTRACKED` or `RESTRICTED.txt`; entries unfrozen. `data/bot_heartbeat` legitimately stale
outside RTH (`ops_health.bot_liveness()` run, not assumed: `ok=True, state='not_rth'`).
**NOT DEPLOYED — DELIBERATELY.** Per the operator's decision this session the wave is STAGED ONLY.
`manager.py` and `orchestrator.py` are both loaded by the running bot, so **W25 is not live until
the operator restarts `ait.main`**; the process still running is pre-W25 code. The operator owns
that restart.
WATCH LIST, carried forward and both CHECKED this session before any new work (per [C3]):
  - W1 (heartbeat liveness): HEALTHY, verified through the real consumer. The writer was NOT
    touched — the naive host-local string is decorative and every consumer reads mtime.
  - W2 (the two calendar-fault warnings must NEVER appear): CLEAN, zero occurrences across all of
    `logs/`. Both emit sites (`master.py:771`, `:813`) still exist, so the grep is not vacuous.
NEXT, unchanged in order from the corrections entry, with (a) now closed:
  (b) data-integrity-07 (give the entry budget back in the stale-pending sweep) and
      logic-exit-risk-03 (scope the cooldown to SECTOR_GROUPS, one authority shared with
      correlation.py). regress-live-06 wants its OWN wave with a live probe — it touches
      initial-placement pricing on the live path.
  (c) docs-policy-04's RUNBOOK fix: name master.py's 2-minute `combined_health_check` as the thing
      to stop, drop the stale `keeper_ait.bat` path.
  (d) Triage batch 2 (the remaining live_path tier) + the dedup pass from batch 1 [6], comparing
      `Where:` AND `Claim:` in full, never a shared code location.
  Still open from before, unchanged: the 10 remaining batch-3 fail-open sites, with #2
  (reconciler.py / HALT_UNTRACKED) still needing re-verification because W22 rewrote that path.

---

## 2026-09-17 - W25 CORRECTIONS + DEPLOYED — the $140 hint fix, the funding conclusion, the restart

Two gaps in the entry above, found answering the operator's own follow-up questions rather than by
a fresh audit pass. Appended per the append-only rule; the entry above is not edited.

[C1] THE ADVISED REMEDY IN [3]'S ALERT WAS WRONG BY 1.6x, and it is operator-facing text, not an
     internal log line — the same class of error the register itself would flag. The credit-floor
     problem message advised "Widen the budget to ~$86". $86 is `(min_width - min_credit) * 100`
     — the MAX LOSS of the needed structure. The operator does not set max loss; they set the
     per-trade budget, which reaches the wing only through `iron_condor.py:121`
     `affordable = risk_budget / (100 * (1 - min_ratio))`. Inverting that correctly gives **$140**.
     FIXED (commit cf2911c, same day, no separate ledger entry made at the time — that omission is
     itself part of what this correction records): the message now computes the budget by
     inverting the real formula, and a new test applies the advised number to the REAL
     `IronCondor._vol_scaled_width` and asserts the wing it produces actually clears the credit
     floor, so the message cannot drift back to a plausible-looking wrong quantity. 29 tests green
     (28 + 1), typecheck OK, compileall OK.

[C2] THE FUNDING CONCLUSION — asked for by the operator, derived after [3] shipped, and it belongs
     in the record, not only in chat: **at $2,100 there is no setting of `max_position_risk_pct`
     alone that lets an iron condor trade.**
       | NLV     | budget @ risk.max_position_risk_pct (3%) | budget @ positions.max_position_pct (5%, ceiling) | needed |
       |---------|-------------------------------------------|------------------------------------------------------|--------|
       | $2,100  | $63                                        | $105                                                   | $140   |
       | $2,800  | $84                                        | $140                                                   | $140   |
       | $4,667  | $140                                       | $233                                                   | $140   |
     The $140 requirement exceeds even the 5% sizer ceiling at $2.1k, so raising
     `risk.max_position_risk_pct` toward 5% is NOT sufficient by itself at that NLV — the sizer's
     OWN cap (positions.max_position_pct) also has to clear $140, which needs ~$2,800 NLV even at
     the full 5%. Options, none applied (operator's call, not scheduled):
       (i)   fund ~$4,700 at today's knobs (3%/5% unchanged);
       (ii)  fund ~$2,800 AND raise `risk.max_position_risk_pct` 3% -> 5% (this is exactly the
             skew [6]/the W25 ADDENDUM registered as a hazard — moving that knob un-masks
             params-trading-01's containment, so it cannot be done as a side effect of a funding
             decision; it needs its own look);
       (iii) lower `AIT_IC_MIN_CREDIT` below $0.70 — REJECTED as evaluated: the $0.70 floor is
             R6's 3x-round-trip-cost bar (8 legs x ~$0.65 + entry/exit crossing = $10-13); a $0.45
             credit is ~$45 gross on 1 contract, under 4x the low end of that cost estimate and a
             clear regression of the floor's own rationale.
     This is a PLANNING finding, not a code defect — recorded here because the operator asked, and
     because a future session should not have to re-derive it from chat scrollback that no longer
     exists.

[C3] DEPLOYED. Per the operator's explicit instruction this session ("go ahead and restart now"),
     outside RTH (19:08 ET on 2026-09-16 / 01:08 CEST on 2026-09-17 — market closed at 16:00 ET,
     matching the RUNBOOK's outside-RTH deploy window), following the documented stop/restart path
     (`docs/RUNBOOK.md:92-93`): `smoke_deploy.py` re-run green immediately before touching anything
     (13 checks, post-cf2911c); `kill -TERM` the supervisor (the lock-file PID, a real `flock()`,
     confirmed by reading `_acquire_singleton_lock` rather than assumed) rather than the bot
     directly, since `graceful_shutdown` -> `bot.stop()` -> `web.stop()` is the path the code
     itself defines; both processes confirmed exited (not just the parent) before relaunch;
     relaunched via the documented command (`python run_orchestrator.py`, nohup'd, the macOS path
     per the known docs/code discrepancy above — no `setsid`).
     New PIDs: supervisor 20950, bot 20967 (were 77009 / 77028). Verified past the
     `bot_launch_verified` grace window, scheduler re-registered all 10 jobs, dashboard/weblog/
     status all restarted clean. Confirmed the RUNNING process is W25, not asserted: imported
     `ait.risk.manager` from the live interpreter and read back
     `inspect.getsource(RiskManager.validate_trade)` — the `params-trading-01` provenance comment
     and the `correlated_count > 0 and final_contracts > 0` guard are both present. Zero
     errors/tracebacks in `bot_stdout.log` or `ait.log` timestamped after the new start; W2 still
     CLEAN (zero occurrences of either calendar-fault warning, checked across all of `logs/` again
     post-restart). No kill-switch files. Rollback point if needed: `git checkout cf2911c` is
     already HEAD, so rollback is `git checkout 74b636e` (last commit before this wave) + the same
     restart path.

STATE AT HANDOFF: branch feature/event-straddle, HEAD cf2911c, pushed to origin. W25 is LIVE —
supervisor PID 20950, bot PID 20967, entries unfrozen, no HALT/HALT_UNTRACKED/RESTRICTED.txt.
Market closed; first live observation of the new code is the RTH open on 2026-09-17. The [C2]
funding gap ($140 needed, no knob at $2.1k alone provides it) is UNRESOLVED and blocks funding —
next session should check the operator's decision before assuming the account has been funded.
NEXT: unchanged from the W25 entry above — (b) data-integrity-07 + logic-exit-risk-03,
regress-live-06 in its own wave with a live probe, (c) docs-policy-04's RUNBOOK fix, (d) triage
batch 2 + the dedup pass — with the funding decision in [C2] now the actual #1 blocker on whether
funding proceeds at all.

---

## 2026-09-17 - [C2] RESOLVED (operator decision): AIT_SIMULATED_CAPITAL=10000, not yet live

Closes the funding gap the corrections entry flagged as blocking the next session. No capital
moves — this is the paper account, and `AIT_SIMULATED_CAPITAL` (`broker/account.py:216`) only
scales every reported balance (NLV, buying power, available funds, etc.) down to a target, so the
risk stack sizes as if the account were that small while the real paper balance sits underneath.

[1] $5,000 (the operator's first number, rounded up from the earlier ~$4,700 estimate) does NOT
    clear the coherence check. That ~$4,700 figure was itself incomplete — it only solved the
    CREDIT-FLOOR arm of `_launch_size_coherence_check` (D2), not the BREAKER arm (D3, last wave's
    fix). Run against the real function: $5,000 clears the credit floor but still trips the
    breaker (`$100 < 1.5x $130`). Swept in $500 steps: the crossover is exactly **$9,750**
    (breaker `1.5 * $130 = $195` needs `nlv * 2% >= $195`). $10,000 clears both arms with margin,
    confirmed by running `_launch_size_coherence_check(10000.0)` directly: zero alerts.
[2] AIT_SIMULATED_CAPITAL=10000 added to `.env` (gitignored, not in this diff) with a comment
    citing this entry. Verified through the REAL startup path, not assumed: `apply_runtime_env_defaults()`
    loads it into `os.environ` correctly. `scripts/smoke_deploy.py` re-run, still SMOKE PASS
    (13 checks).
[3] NOT YET LIVE. The running bot (supervisor 20950 / bot 20967, per the W25 deploy entry) started
    before this env var was added; dotenv loads once at process startup, so it will not pick this
    up without a restart. Awaiting the operator's go-ahead for that restart, per standing policy.
[4] OPERATOR'S OWN CAVEAT, recorded because it matters for how much weight to put on this number:
    "we will revisit those numbers later anyway." $10,000 is a probe value to exercise the sizing
    path end-to-end on paper, not a considered final funding target — treat it as provisional in
    any future analysis that assumes it as a real constraint.

STATE AT HANDOFF: `.env` has the new var; process has not been restarted to pick it up.

---

## 2026-09-17 - AIT_SIMULATED_CAPITAL=10000 DEPLOYED

Per operator go-ahead. Same documented path as the W25 deploy: outside RTH (19:27 ET, book flat —
0 open positions), `smoke_deploy.py` re-run green immediately before, `SIGTERM` to the supervisor
lock-file PID (20950), both old PIDs confirmed exited, relaunched via `python run_orchestrator.py`.

New PIDs: supervisor 25177, bot 25224 (were 20950 / 25224... were 20950 / 20967). Verified past
`bot_launch_verified`, scheduler re-registered all 10 jobs. Confirmed the var is actually in the
NEW process's environment (`ps eww -p 25224`), not just in `.env` — direct proof rather than
waiting on a log line that only fires on the next account-snapshot refresh, which outside RTH may
not happen for a while. Zero errors/tracebacks since the new start; W2 still clean.

STATE AT HANDOFF: branch feature/event-straddle, HEAD 00aac6d. Supervisor 25177 / bot 25224 live,
entries unfrozen, book flat, AIT_SIMULATED_CAPITAL=10000 active. Per the operator's own caveat
recorded above, treat $10,000 as a provisional probe value, not a settled funding target.

---

## 2026-09-17 - TYPO CORRECTION to the entry above

The line "New PIDs: supervisor 25177, bot 25224 (were 20950 / 25224... were 20950 / 20967)" carries
a garbled mid-edit fragment. Correct reading, no facts changed: new PIDs are supervisor 25177 / bot
25224; old PIDs (from the prior restart) were supervisor 20950 / bot 20967. Appended per the
append-only rule rather than fixed in place.

---

## 2026-09-17 - W26 EXECUTED — the entry-budget give-back + the cluster-scoped cooldown

(b) from the W25 pointer, both findings from register-triage batch 1, both on the live entry path.
Register-don't-fix does not apply to the two named findings. STAGED, NOT DEPLOYED.

[1] data-integrity-07 — FIXED, and LATENT rather than materialized, measured before planning.
    The live DB has ZERO `stale_pending_never_filled` rows; 10 trades placed, 10 closed, and
    `daily_stats.trades_taken` reads 5 + 5 — matching exactly. So the counter has never actually
    been wrong in the live record. Worth fixing anyway, and the consequence is larger now than when
    it was found: the first-hour budget is 2, so ONE phantom halves first-hour capacity.
    The sweep (`reconciler._sweep_stale_pending`) now calls a new `_refund_entry_budget` when it
    books a never-filled close. Three correctness points, each of which would make this a WORSE bug
    than the one being fixed if got wrong:
      (a) GATED ON THE CAS. `close_trade` returns "True iff the close landed" and the sweep
          DISCARDED it, then ran `closed += 1` unconditionally — so a lost CAS was already being
          miscounted as a close. A refund on that path would credit a slot back for an order that
          actually FILLED. Now captured; on False it refunds nothing, counts nothing, and logs
          `sweep_close_lost_cas`.
      (b) THE TRADE'S OWN ENTRY DATE, never today. `reconcile()` runs on every startup, so a
          phantom placed Monday is typically swept Tuesday morning. Crediting TODAY would hand back
          a slot today never spent, and `taken = -1` makes `_get_trade_budget` return
          `max_trades + 1` — a FAIL-OPEN on the daily cap, produced by a data-integrity fix.
      (c) FLOORED AT 0. A negative counter inflates the budget the same way. W25 shipped the
          mirror image of this lesson (a floor of 1 manufacturing a position where 0 was correct):
          a floor is only safe in the direction that cannot create permission.
    REJECTED ALTERNATIVE, recorded because it is the obvious one: move the increment to
    fill-confirmation so the counter only counts real trades. The budget's purpose is pacing
    PLACEMENT (2 in the first hour, 4 by mid-day); counting only fills lets a whole scan cycle place
    orders before any fill, bypassing the tiers entirely. Same shape as W25's rejected `max(0,...)`
    — the tidier-looking fix changes trading behaviour.

[2] logic-exit-risk-03 — FIXED, with a DELIBERATE DEVIATION from the literal spec, stated here
    rather than left to read as a faithful implementation.
    One authority now: `correlation.symbol_clusters()` / `cluster_symbols()`. The fact lived in
    three shapes before — `CorrelationGuard._same_sector_cluster` (pairwise),
    `RiskManager._count_correlated_positions` (re-inlining the intersection off a direct
    SECTOR_GROUPS import), and NOT AT ALL in the cooldown, which is the defect. Both existing
    consumers delegate. The delegation is deliberately NOT `sym2 in cluster_symbols(sym1)`:
    cluster_symbols() always contains the symbol itself, which would make
    `_same_sector_cluster("GLD","GLD")` True where the original is False — unreachable through
    check_correlation today (the `existing == new_symbol` arm short-circuits) but an exact
    delegation cannot be broken by a future caller that does reach it. Pinned by a test.
    THE WINDOW IS GRADUATED, and this is the operator decision of the wave: the stopped symbol keeps
    the full R12-B4 one-trading-day window; cluster SIBLINGS get
    `risk.post_stop_cluster_cooldown_minutes` (default 120, `0` disables and restores the exact
    pre-W26 behaviour without a code change). Strict cluster-wide scoping — the finding as written —
    would darken the ENTIRE BOOK, because the universe is a single cluster, for the rest of the
    session AND all of the next (a Monday 14:00 stop clears only Wednesday: Tuesday's cutoff is
    Monday 09:30). With a touch stop as the primary loss exit and a volume-tuned config
    (max_daily_trades 12, min_confidence 0.50, 1 contract "= max # of trades") that is close to
    "the first stop ends the trading day". The hazard R12-B4 actually targets is IMMEDIATE re-entry
    into the same adverse move — the live 08-24 QQQ instance was 13 minutes. So: faithful for the
    stopped symbol, shorter for siblings. THIS DEVIATES from the spec's "one trading day" for
    siblings; a future reader should treat 120 as a tunable, not as R12-B4.
    The verdict return type changed from a bare exit_time to `CooldownBlock(symbol, exit_time)`
    because the blocking symbol is no longer necessarily the blocked one — an operator reading
    "QQQ entry blocked" otherwise has no way to see that a SPY stop caused it. The log now carries
    `stopped_symbol` and `via_cluster`.

[3] AN EXISTING TEST ASSERTED THE DEFECT, and was split rather than deleted.
    `test_r21_post_stop_cooldown.py::test_other_symbol_unaffected` asserted that a QQQ touch stop
    does NOT block SPY — which is precisely logic-exit-risk-03. The assertion was not wrong so much
    as under-specified: what it protected is that an UNRELATED symbol is not swept up, and SPY/QQQ
    were a poor example of "unrelated". Replaced by `test_unrelated_symbol_unaffected` using GLD
    (in no SECTOR_GROUPS set) with sibling scoping ON; the sibling case is pinned in the new file.
    That file's `_orch()` helper also had to start carrying Settings, and its comment claiming it
    "touches no orchestrator state besides the DB file" was corrected. It defaults to
    sibling_minutes=0 so every other test in it keeps measuring exactly what R12-B4/R22 wrote it to
    measure.

[4] MY OWN PLAN'S ITEM 3 WAS WRONG BY 16x, and the guard was NOT shipped.
    The plan said "three live `date.today()` sites" and proposed extending W20's naive-clock
    class-guard with a 3-site allowlist. That count came from grepping `src/ait/risk/` and
    `src/ait/bot/` only. An AST sweep of `src/ait` (call nodes, so docstrings and comments are
    excluded — the raw grep's ~50 was itself inflated by a docstring usage example in
    economic_calendar.py and R17's own fix-note comment in earnings.py) finds **49 REAL executable
    sites**: monitoring 16, data 15, dashboard 4, ml 4, learning 3, bot 2, execution 2,
    diagnostics 1, risk 1, strategies 1.
    An allowlist of 49 is a catalogue, not a guard, and scoping one to risk/+bot/ would have
    encoded my own bad grep into the suite while giving false comfort. So nothing shipped.
    THE LIVE-PATH SUBSET IS SHARPER THAN THE PLAN SAID, which is the reason this needs its own wave:
    `options_chain.py` x5 (:209, :301, :340, :550, :551) computes DTE, which drives STRIKE AND
    EXPIRY SELECTION; `iron_condor.py:104` feeds DTE into the wing-width formula
    (`sqrt(dte/365)`) — the very function W25's coherence check calls; `portfolio.py:467` is exit
    DTE and `:772` days-to-earnings; `orchestrator.py:2650` is the earnings blackout ENTRY gate;
    `economic_calendar.py` x6 resolves macro-blackout dates; `risk/manager.py:132` is the
    losing-days throttle. On this CEST box `date.today()` rolls at 18:00 ET, so every evening
    evaluation of these asks about the wrong day.
    ALSO REGISTERED: W20's class-guard covers `datetime.now()` and NOT `date.today()`, so the guard
    that was supposed to catch the next instance of this family cannot see any of these 49. And
    `earnings.py:88` carries a comment showing R17 already fixed ONE instance of exactly this class
    — instance-fixed, class left open, which is the arch-04 pattern R25 named.
    WANTS THE CENSUS TREATMENT W17 GAVE THE FAIL-OPEN CLASS (355 except clauses -> 177 candidates
    -> 120 traced -> 12-site ranked shortlist), as its own wave. Not scheduled.

[5] COLLATERAL FROM MY OWN EARLIER CHANGE, found here and fixed as a class.
    `AIT_SIMULATED_CAPITAL=10000` (added to `.env` earlier today) leaked into every test run and
    silently rewrote risk-gate arithmetic: the per-trade 3% cap became $300 instead of $3,000, which
    broke two PASSING `test_w2_orchestrator_gates::TestEntryVixUsesTheValidatedValue` tests that
    assert on a $380 max_loss and have nothing to do with simulated capital. I ran `smoke_deploy`
    after that `.env` edit and NOT the suite, which is how it was missed — so the real suite
    baseline was silently 1837/3, not 1840/1, between that change and this wave.
    Fixed in `tests/conftest.py` as an autouse fixture that unsets the var for the whole suite,
    rather than patching the two tests: no test should depend on an operator's local `.env`, and
    pinning NLV in those two would leave the next env var to do the same thing. Same isolation
    family as the known-failing `test_r19_config_authority::test_applier_and_readers_share_one_table`
    (failing on this box since W16 for exactly this reason). Verified no test sets or wants the var.

[6] VERIFICATION. `tests/test_w26_budget_and_cluster_cooldown.py`, 19 tests.
    - NEGATIVE CONTROL per finding. For data-integrity-07 the real sweep runs with the refund
      disabled — the literal pre-W26 path — and the slot is asserted still spent, with the PREMISE
      (a never-filled close really was booked) asserted first so it cannot pass by not sweeping.
    - A WIRING GUARD, not just a helper test: the refund helper can be perfect while the sweep
      stops calling it, so one test asserts refunds == bookings on the never-filled path.
    - BOTH HALVES of the graduated contract (sibling clears after its window while the stopped
      symbol does not), and the `0` escape hatch — an untested escape hatch is not an escape hatch.
    - Gates: full fast lane **1859 passed / 43 skipped / 1 failed** (the known local-`.env` case).
      Count reconciled rather than assumed: the pre-W26 baseline was **1840**, not the 1839 recorded
      at W25's lane, because cf2911c added one test afterwards and only that file was re-run.
      1840 + 19 = 1859. `compileall` OK, `scripts/typecheck.py` OK, `smoke_deploy.py` SMOKE PASS.

STATE AT HANDOFF: branch feature/event-straddle, HEAD 78d49c7. **NOT DEPLOYED** — `orchestrator.py`,
`reconciler.py`, `correlation.py`, `manager.py` and `settings.py` are all loaded by the running bot
(supervisor 25177 / bot 25224), so the process is still pre-W26 and W26 is not live until the
operator restarts. Book flat, entries unfrozen, no kill-switch files, AIT_SIMULATED_CAPITAL=10000
active in the live process.
NEXT, re-ranked with (b)'s first half now closed:
  (a) regress-live-06 — the credit floor is enforced at reprice but NOT at initial placement. Its
      own wave with a live probe, per the batch-1 ranking: it touches live entry pricing.
  (b) The 10 remaining batch-3 fail-open sites, consequence-ranked in
      reports/w17_batch3_failopen_census_20260916.md. Highest: #2 reconciler.py:1083
      (HALT_UNTRACKED never written) and #3 orchestrator.py:701 (one unparseable exit_time silently
      drops realized P&L from the day).
  (c) THE date.today() CENSUS from [4] — 49 sites, live-path subset touching strike/expiry
      selection, wing width, both blackout gates and exit DTE. New, and arguably outranks (b):
      unlike the fail-open sites it is UNCONDITIONAL, not fault-gated — every evening evaluation is
      already wrong, no trigger required.
  (d) docs-policy-04's RUNBOOK fix (name master.py's 2-minute combined_health_check as the thing to
      stop; drop the stale keeper_ait.bat path).
  (e) Triage batch 2 (~167 remaining R24 + carried-over findings) + the dedup pass from batch 1 [6].
  Unchanged: arch-01..04 and qual-01/02/04 remain untouched since R25.

---

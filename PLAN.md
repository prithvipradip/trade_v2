# AIT v2 — Plan & Audit Ledger

**Mission:** autonomous options-income bot (iron condors on liquid US ETFs/megacaps),
ML-gated entries, automated exits, hands-off ops. Paper account DUN603821.
**Goal:** a trustworthy real track record answering "does this have edge?" before funding
$3,000 CAD.

**Current state (2026-08-15):**
- **Track record: 16 closes, 9W-7L, +$150.83 (broker-true, referee 0 BREAKS).** POSITIVE for
  the first time — three consecutive auto take-profits (+$243 QQQ 08-07 through NFP, +$154 SPY
  08-11, +$206 QQQ 08-12) since the blackout relaxation + wide-wing promotion. Book FLAT.
  Sample: 16/50 closes. Sim caveat stands: the live sample is the only verdict that counts.
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

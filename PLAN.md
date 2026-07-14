# AIT v2 — Plan & Audit Ledger

**Mission:** autonomous options-income bot (iron condors on liquid US ETFs/megacaps),
ML-gated entries, automated exits, hands-off ops. Paper account DUN603821.
**Goal:** a trustworthy real track record answering "does this have edge?" before funding
$3,000 CAD.

**Current state (2026-07-14, end of day):**
- Booked track record since the 2026-07-06 reset: 9 closes, +$280.20. **R13's shadow referee
  says the true number is +$254.90 / 6W-3L / PF 2.64** (the booked scoreboard recorded phantom
  re-close prices + estimated commissions). **Restatement is a pending DECISION — see D1.**
- Open book: 1 IWM long straddle (−$233, CPI came in quiet — the long-vol bet didn't pay).
  Plus 4 UNTRACKED reverse condors from the 07-13 incident, not in the books (**U5**).
- **ENTRIES ARE FROZEN** (HALT_UNTRACKED since 09:00 07-14) and stay frozen until U5 is
  resolved. Zero sample-building while frozen — this is the top-priority blocker.
- **R12 + R13 both DEPLOYED and verified** — R12 at 09:09 (tag `deploy-20260714`), R13 at
  12:10 (tag `deploy-20260714b`, commits b154459 → 3901fab): 639 tests green, SMOKE PASS with
  the new config sentinels, `entries_frozen` observable, `trading_verified`, marks updating.
- Deferred verification: `scan_symbol_timing` couldn't be observed on 07-14 (the CPI-day
  `economic_event_skip` gate short-circuits before it). Scheduled task
  **`AIT-Deploy-Liveness-20260715` runs `scripts/check_first_rth_liveness.py` at 09:40 ET
  07-15** and Telegrams the verdict. That closes the deploy checklist.
- The machine works — real fills, real costs, an honest scoreboard, layered protection.
  **Edge is still unproven**; R9's statistics say even 50 closes is only a preliminary read.
- 13 audit rounds complete (~275 defects found).

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

**1. Unfreeze the machine (U5 → U6).** Nothing else matters until the bot is trading again.
**2. Take the two open decisions (D1, D2)** — both must be settled BEFORE the sample grows,
because both change how results are scored, and "criteria fixed before looking" is the rule.
**3. Close the security exposure (U7 → U8/U9 → U10/U11/U12).** Independent of trading; do in
an evening.
**4. Arm the silent-failure layers (U1, U2, U3, U4).** Long-standing, ~5 min each, and R13
re-confirmed all three are still unarmed — every recovery layer is downstream of them.
**5. Ship the R14 code queue** (below) in out-of-RTH windows while the sample accumulates.
**6. Then, and only then**: ML ablation → 50-close preliminary read → 100-close verdict →
Phase-2 sizing → go-live gates.

### OPEN — user actions (only YOU can do these; each verified on the box)

**Blocking the mission**
| # | Action | Why it matters |
|---|---|---|
| U5 | **Close the 4 untracked reverse iron condors at IBKR → verify flat in TWS → THEN delete `data/HALT_UNTRACKED`** (see INCIDENT 2026-07-13). They are NOT in the bot's books and must not be adopted (that would contaminate the sample). R13 verified this flow is SAFE — the bot has no exit path referencing them, so it will not fight your closes. Deleting the file BEFORE the broker is flat just re-freezes at the next reconcile. | **THE blocker.** Entries frozen = zero sample-building (R2: ~15-20 closes/month is the ceiling even when healthy). Plus ~$955 of unbooked premium bleeding theta to 07-24, with the CPI catalyst already passed. |
| U6 | **Restore the live market-data entitlement**: log the live account out of mobile/web (the ONE-data-slot rule), then restart the Gateway so entitlements reload at login. | Error 10089/354 on live quote requests since ≥07-09, low-rate but constant. Broker-stream marks still flow so exits work — but entry pricing quality depends on it the moment entries resume. Known class from the 06-30/07-02 saga. |

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
| D1 | **Scoreboard restatement.** The referee books exits from the FIRST closing fill group + real commissions; the system booked phantom re-close prices + a flat $0.65/leg estimate. Restating moves the record from +$280.20 / 7W / PF 3.03 to **+$254.90 / 6W-3L / PF 2.64** (QQQ's booked "win" is really a −$15.71 loss). | **Yes — do it once, with a DB backup, while the bot is stopped.** Honest numbers must precede the sample, not follow it. The go-live gates are meaningless if the inputs are wrong. |
| D2 | **Drawdown-gate method.** The scorecard's base is "sum of capital_at_risk of CURRENTLY OPEN trades, floored at $1,000" → after a flatten the base collapses to $1,000 and DD reads **13.8% (FAIL)**. The referee uses max CONCURRENT deployed risk over the window ($2,114) → **7.4% (PASS)**. The two methods disagree ACROSS the 8% gate line. | **Pin the concurrent-risk method now**, before more data accrues. It is the economically meaningful denominator (what was actually at risk), and picking it later — with results visible — would violate criteria-fixed-before-looking. Also backfill `capital_at_risk` where derivable (5 of 9 closes have 0). |
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

## Round 14 (queued) — the R13 SOON list, ordered by exposure
All evidence is already gathered (R13 lens reports). Ship in out-of-RTH windows while the
sample accumulates. **Status 2026-07-14 EOD: items 1-2 are WRITTEN but UNCOMMITTED in the
working tree** (`orchestrator.py`, `reconciler.py`) — finish, test, deploy as one batch.

**Tier 1 — live exposure in a bad tape**
1. **Exit-price sanity bound.** The exit path has no ceiling: `ask + $0.10` with no cap, plus a
   MARKET fallback on a 4-leg BAG exactly when quotes evaporate. Executed proof: a $2-wide
   condor with a garbage 9.90 ask placed `BUY LMT 10.00` — 5× the wing width, i.e. 5× the
   structural max loss. Fix: cap a credit buyback at the wing width (the max loss the wings
   already guarantee); never MARKET a multi-leg BAG — no quote ⇒ credit prices AT the wing
   width + alert, debit defers a cycle + alert. *(written)*
2. **Broker-position check before reverse exit combos** (human-factors #2). After a manual TWS
   flatten the monitor still demands the exit and `_execute_exit` reverses the legs with no
   broker check → REBUILDS the position inverted. Same end-state as the 07-13 incident, but
   operator-triggered and needing no bug at all. Fix: `reconciler.position_is_live(trade)`
   before any reverse order; `None` (broker view unavailable) must never read as "gone".
   *(written)*
3. **Staleness gate on exit inputs.** `Quote.timestamp` is write-only; the touch stop and DTE
   exits act on possibly-frozen marks. The gate ALREADY EXISTS (`quality.validate_quote`, 30s)
   and is instantiated — with **zero call sites**. Wire it into the touch/DTE path; reject or
   alert on quotes older than ~3 min in RTH; require two agreeing ticks before firing a touch
   on a degraded/fallback quote.

**Tier 2 — correctness of the verdict itself**
4. **Book realized P&L from executions** (the D1 fix in code): first-closing-group price + real
   commissions, not the phantom-fill price + a flat estimate. Without this, D1's restatement
   just re-accrues the same error on the next close.
5. **DD-method pin** (D2 in code): max concurrent deployed risk as the denominator; backfill
   `capital_at_risk` where derivable.
6. **Commission attribution**: key on exec_id/orderId with an EOD sweep that re-stamps
   already-closed trades; keep phantom/unattributed costs in a separate incident bucket.
7. **`executions.exec_time` is +4h off** (already-UTC broker time double-converted as if ET).

**Tier 3 — resilience / ops**
8. **MTM brake gap-blindness**: the SOD unrealized baseline is captured AFTER the gap, so a
   −8% gap-open registers `mtm_day = 0`. Persist the prior close's unrealized as the baseline.
9. **Exit-reject backoff + alert counter**: retries are fixed-cadence and unbounded, no
   halted-contract awareness, and exit rejects page nobody.
10. **Zero-options mass-close guard escape hatch** — a full manual flatten of the last position
    is never booked (the guard refuses, correctly, but there is no confirmed-empty path).
11. **Keeper relaunch verification + alert** (it never checks that a relaunch worked); digest
    reports the mirror's CONTENT age, not the local snapshot's mtime.

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

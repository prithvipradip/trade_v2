# AIT v2 — Operator Runbook

One page: what each alert means and what YOU do. The bot manages exits on its
own; your job is responding to these messages. (Governance audit GOV-5.)

## Kill switches (no `kill -9` needed)
| Action | How |
|---|---|
| Stop NEW entries (exits keep managing) | create empty file `data/HALT` |
| Resume | delete `data/HALT` |
| Ban one symbol | add ticker line to `data/RESTRICTED.txt` |
| Pause auto-relaunch (keeper_ait.bat) | create empty file `data/KEEPER_PAUSE` |
| Resume auto-relaunch | delete `data/KEEPER_PAUSE` |
| Full stop | `data/KEEPER_PAUSE` FIRST (or the keeper relaunches within ~90s and undoes the stop), then `data/HALT`, then stop processes if truly needed |

## Alert → action
| Telegram alert | Meaning | Do this |
|---|---|---|
| `READ-ONLY: orders rejected (Error 321)` | Gateway session lost trade rights | Gateway → Configure → API → Settings: CHECK "Read-Only API" → Apply → UNCHECK → Apply → OK. Bot recovers on next probe. |
| `IB Gateway unreachable (daily maintenance)` | Normal overnight restart | Nothing for 30–60 min. If persists into pre-market: start Gateway manually (IBC should auto-login). |
| `BOT DOWN — gave up after N restarts` | Supervisor exhausted restart budget | Check Gateway is logged in; then start manually: `python run_orchestrator.py`. Investigate logs/bot_stdout.log tail. |
| `CIRCUIT BREAKER: Trading halted` | Consecutive losses / API failures tripped the breaker | Entries frozen until the breaker resets (see reason in the message). Review what tripped it on :8503 + logs. Notification is throttled (once per trip + hourly); the per-cycle log line `trading_halted` shows it's still active. |
| `LOOP IMPAIRED` | The trading loop errored N consecutive cycles | Stops/TPs may not be running. Check `logs/bot_stdout.log` tail for the exception; if it repeats, `data/HALT` + restart per the deploy checklist. A restart does NOT fix a code bug — capture the traceback first. |
| `component down/degraded (watchdog)` | A subsystem (ibkr/market_data/...) stopped heartbeating | Usually self-recovers (watchdog restarts it). Two in a row for the same component: check Gateway login + data lines before restarting anything. |
| `DAILY MTM LOSS HALT` | Day's mark-to-market loss breached the cap | Entries auto-frozen; exits still run. Review the book on :8503. Halt clears at next ET day. If you decide to close a bleeder manually, FOLLOW THE MANUAL-INTERVENTION PROCEDURE below — the bot's exit engine is live on that position every 30s and will fight an unannounced manual close. |
| `RECONCILE NEEDS ATTENTION: ASSIGNMENT` | Short option assigned — stock appeared | In TWS/Gateway: liquidate the assigned stock, verify the remaining legs, then delete `data/HALT_UNTRACKED` if present. |
| `UNTRACKED OPTION POSITION ... ENTRIES FROZEN` | Order filled during a crash window; bot has no record | In TWS: identify the position and resolve it AT THE BROKER (usually: close it). Only after the broker book is flat of it, delete `data/HALT_UNTRACKED`. Deleting the file while the position still exists just re-freezes at the next reconcile — "leave it and note it" is not an option the code supports. |
| `PDT BLOCKED EXIT` | A stop wanted to fire but PDT rules block same-day close | Decide: accept overnight risk, or close manually (accepting the PDT strike). |
| `MARKS MISSING x10 ticks` | No option marks → stop/TP protection NOT running | Check: is the live account logged in elsewhere (mobile/web steals the ONE data slot)? Log it out. Check Gateway data lines. |
| `EOD RECON: N BREAK(S) ...` | Books vs broker mismatch | Compare :8503 positions vs TWS portfolio. Common cause: manual action at broker. Fix the DB via close/adopt or ask the assistant. |
| `EOD RECON: CLEAN | ... gap stress: $X` | Daily all-clear | Glance at the −10% gap-stress number. If it approaches your pain threshold, reduce short-premium count. |
| `model retrain FAILED/HUNG` | Models stale | Not urgent same-day. Re-run: `python run_orchestrator.py --retrain`. |
| `order_rate_limit_tripped` (log CRITICAL) | >8 orders/min — malfunction | Create `data/HALT`, inspect logs, contact assistant. Do NOT just restart. |
| `STATE DB BACKUP FAILED` | Track record unprotected | Check disk space; run a manual copy of `data/ait_state.db`. |

## Manual intervention procedure (closing/adjusting ANY tracked position yourself)
The exit engine evaluates every tracked position every 30 seconds and does not
know you are acting. R13 executed proof: after a manual flatten the monitor
still demanded the exit and `_execute_exit` places the reverse combo without a
broker-position check — the bot would rebuild what you just closed, inverted
(same end-state class as the 07-13 incident). So, in order:
1. `New-Item -ItemType File data\HALT` (freezes entries; exits keep managing).
2. In TWS, CANCEL the bot's working exit orders on that position — or, if
   you're closing the whole book, `New-Item -ItemType File data\KEEPER_PAUSE`
   FIRST (else keeper_ait.bat relaunches the whole stack within ~90s and
   undoes your stop — this raced the operator once, R13), then stop the bot
   processes.
3. Do your manual close(s). Verify fills in TWS.
4. Positions NOT in the bot's books (untracked): safe to close anytime; the
   bot has no exit path referencing them (verified). Then delete
   `data/HALT_UNTRACKED` once the broker is flat of them.
5. Restart/resume the bot; the next reconcile books your closes (note: if you
   flattened the LAST option position, the zero-options mass-close guard
   refuses to book it — expect an `EOD RECON` break and resolve via
   close/adopt with the assistant). If you created `data/KEEPER_PAUSE` at
   step 2, delete it too so the keeper resumes relaunching.

## Daily 2-minute routine
1. Glance at Telegram: any non-CLEAN EOD recon? any CRITICAL?
2. Open http://localhost:8503 — positions green/red, unrealized total sane?
3. Friday extra: confirm the off-box mirror has CURRENT CONTENT (timestamps
   lie — `copy2` back-dates them; the backup job now hash-verifies, so any
   `STATE DB BACKUP FAILED` alert this week means the mirror is suspect):
   `python -c "import sqlite3; print(sqlite3.connect(r'file:C:/Users/prith/Documents/ait_backups/ait_state.latest.db?mode=ro', uri=True).execute('SELECT MAX(exit_time) FROM trades').fetchone())"`
   — must show this week's latest close. (And that folder syncs to cloud.)
4. Reading the OTHER box's logs: the supervisor fetches `origin/shared-logs` itself every
   weekday at **16:50 ET** (5 min after this box's 16:45 publish; a fetch only, it never
   touches the checkout, and it does not need `AIT_LOG_OWNER`). Its log line
   `shared_logs_fetched` carries `latest_by_owner`: a stale date there means that box has
   stopped publishing. Read a day with
   `python scripts/read_shared_logs.py --owner <name> --date YYYY-MM-DD --no-fetch`
   (drop `--no-fetch` to force a fresh fetch). Publishing your OWN days additionally needs
   `AIT_LOG_OWNER=<name>` in this box's `.env` (the job is silent without it).

## Deploy checklist (EVERY deploy — the 2026-07-10 incident is why)
A green test suite is not a safe deploy: on 07-10 unit tests passed and the
deployed code still killed the trading loop for a full session (stops/TPs
unmanaged), found a day later. Do all five steps, in order:

1. **Tests green.** Locally (the fast lane — same selection as CI):
   `python -m pytest tests/ -q -m "not ibkr and not slow"`
   and the CI run for the pushed commit is green (`.github/workflows/ci.yml`
   also byte-compiles all of src on every push).
   (The old duckdb-pollution warning is retired: since R12, `_init_duckdb()`
   derives its path from the state-DB path, so `StateManager(tmp_path…)`
   isolates BOTH stores — verified by execution 2026-07-14.)
2. **Deploy outside RTH** (before 09:30 or after 16:00 ET) whenever possible.
   If it truly must happen during RTH: `New-Item -ItemType File data\HALT`
   first, deploy, smoke (step 3), then `Remove-Item data\HALT`.
3. **Run the smoke — no exceptions:** `python scripts/smoke_deploy.py`
   must print `SMOKE PASS (…)`. It executes the runtime paths unit tests
   don't (imports, real-config load, real-DB row mapping, digests, learning
   state, wing sizing) read-only, no IBKR needed — this is the 5-minute check
   that would have caught 07-10. Optional: `--telegram` sends the verdict.

   **One-off, on a box whose W21 timestamp migration is still outstanding**
   (R26 F5 — the Windows box as of 2026-09-20; the macOS box is already
   migrated). Stop the bot first; the script snapshots and writes:

   ```
   python scripts/restate_w21_tz_aware.py              # dry run — read the plan
   python scripts/restate_w21_tz_aware.py --apply      # snapshot + write
   python scripts/smoke_deploy.py --check-live-corpus  # 14 checks, not 13
   ```

   Do **not** treat the script's own exit code as proof. Its post-check could
   not fail before W38, and a migration should not be verified by the tool
   that performed it — `--check-live-corpus` re-reads the DB independently and
   is what closes F5. If it reports naive rows, the migration did not land;
   the snapshot path is printed by the `--apply` run.
   `--apply` aborts rather than relabelling any row written on or after
   `HOST_MOVE` (2026-09-14) — those are a restatement, not a convention
   change, and need handling by hand.
4. **Confirm liveness after the next RTH open** (~09:35–09:45 ET):
   - fresh heartbeat: `Get-Item data\bot_heartbeat | Select-Object LastWriteTime`
     — must be minutes old and refreshing;
   - one completed trading cycle in the logs:
     `Select-String -Path logs\bot_stdout.log -Pattern "scan_symbol_timing" | Select-Object -Last 3`
     (fires at the end of each symbol scan; timestamps must be post-open), and no fresh
     `Select-String -Path logs\bot_stdout.log -Pattern "LOOP IMPAIRED|trading_cycle_error" | Select-Object -Last 5`;
   - no `LOOP IMPAIRED` Telegram alert arrived;
   - marks updating: dashboard :8503 unrealized P&L moving, or
     `python -c "import sqlite3; print(sqlite3.connect('file:data/ait_state.db?mode=ro', uri=True).execute('SELECT symbol, mark_time, unrealized_pnl FROM open_positions').fetchall())"`
     — `mark_time` within the last few minutes.
5. **Rollback path** = `git checkout <last-good-tag>` + restart the bot
   (stop processes, then `python run_orchestrator.py` / `keeper_ait.bat`).
   For this to exist: TAG every verified deploy going forward —
   `git tag -a deploy-YYYYMMDD -m "smoke pass + first-RTH liveness OK"` then
   `git push origin --tags`. Only tag AFTER step 4 passes.

## Fixed limits (NOT configurable — visible on purpose)

These values are deliberately fixed in code, in ONE module: `src/ait/config/invariants.py`.
They are not in `config.yaml` or `.env` because each is either a **backstop that exists to
catch the system (or its configuration) being wrong**, a **rule imposed from outside**, or
the **line a config value is checked against**. Anything in `config.yaml` is tunable
regardless of a comment — the file is shared between machines through git and edited by
people and AI sessions — so putting these there would let one edit loosen both a gate and
its check. They are logged at every start (`fixed_limits_in_effect`). To change one: edit
the module, update this table (a test compares them), and record it in PLAN.md.

| Name | Value | Why it is fixed (and where enforced) |
|---|---|---|
| `ORDER_RATE_LIMIT_MAX` | 8 | Max orders in the window below; the next is refused and logged CRITICAL. A backstop against a logic bug spamming placements — if config could raise it, the bug it exists to stop would not be stopped. `executor.py` (entry + roll placement share one bucket). |
| `ORDER_RATE_LIMIT_WINDOW_S` | 60 | The rolling window for the limit above. |
| `NBBO_DISORDERLY_SPREAD_RATIO` | 1.0 | A combo entry whose (ask − bid) is wider than its own mid is rejected. Backstop against a bad quote (delayed data, crossed/locked market). Distinct from the configurable per-contract `options.max_bid_ask_spread_pct`. `executor.py`. |
| `NBBO_STALE_SIGNAL_DEVIATION` | 0.35 | A combo entry whose signal price differs from the live mid by more than 35% is rejected: the signal was priced on a market that has since moved. `executor.py`. |
| `NBBO_MID_FLOOR` | 0.05 | The two tests above apply only when \|mid\| exceeds this; below it ratios blow up and mean nothing. `executor.py`. |
| `LIVE_MAX_SPREAD_PCT` | 0.15 | Startup refuses a non-paper mode if `AIT_LIQ_MAX_SPREAD` or `options.max_bid_ask_spread_pct` exceeds it. It is the line config is checked AGAINST (paper is relaxed to 0.40), so it cannot live in the file it checks. `orchestrator.py` GOV-5. |
| `LIVE_MIN_VOLUME` | 10 | Startup refuses a non-paper mode if `AIT_LIQ_MIN_VOLUME` is below it. Same reason. Documented in the go-live checklist below. |
| `PDT_MAX_DAY_TRADES` | 3 | Day trades the bot allows itself per window. FINRA's pattern-day-trader rule flags margin accounts under $25,000 at 4 in 5 business days; this is the law, not a preference. The *switches* are config: `account.pdt_protection`, `account.pdt_account_under_25k`. `pdt_guard.py`. |
| `PDT_WINDOW_DAYS` | 5 | The rolling window for the rule above. |
| `BROKER_SNAPSHOT_TTL_SECONDS` | 5.0 | How long the all-clients open-order snapshot is re-used within and across fill checks. A cache lifetime that avoids repeated broker round trips — API-call pacing, not a trading preference. `executor.py`. |
| `IC_BUDGET_CAPPED_MIN_WING` | 1.0 | Narrowest wing a budget-capped iron condor may shrink to, in dollars: $1 is the narrowest the listed-strike grid offers. The *configurable* minimum wing is `backtest.wing_floor_dollars` (`AIT_IC_WING_FLOOR`, default $2), which binds only when the per-trade budget does not. `iron_condor.py`, `engine.py`. |

The 30 s fast-monitor tick is not repeated here: its single source is
`ait/utils/cadence.py::FAST_MONITOR_CADENCE_SECONDS`.

## Order lifecycle timeline (what happens to an order, and for how long)

Every age below is evaluated on the **30 s fast-monitor tick** (measured 2026-09: median
30.9 s between ticks, 98% under 35 s, worst 100 s), so a duration under 30 s takes effect at
the next tick. Tunables live in `config.yaml` (each key carries Floor / Ceiling / IMPACT next
to it); the startup log `order_timeline_in_effect` prints every effective value.

**Exit order (position is CLOSING)**

| Age | What happens | Key (default) | Floor and why |
|---|---|---|---|
| 0 – grace | Ignored by the fill checker so it can appear in IBKR's trade list; a brand-new order would otherwise read as cancelled. | `exit.order_visibility_grace_seconds` (10) | 5 s. A margin, not a measurement: the one acknowledgement we could time took < 1 s (an entry). Must be below the stop rest. |
| rest | Still unfilled: a cancel is requested but the position stays CLOSING so the monitor can re-price. Stops rest less than ordinary exits. | `exit.exit_stop_pending_rest_seconds` (120) / `exit.exit_pending_rest_seconds` (300) | 15 s / 30 s. The stop rest may not exceed the ordinary rest. |
| band TTL | A broker "price band" reject stops capping later attempts. | `exit.exit_band_ttl_seconds` (300) | 30 s. |
| foreign | The order belongs to ANOTHER clientId and cannot be cancelled from here: the operator is paged instead. | `exit.foreign_exit_escalation_seconds` (300) | 60 s, and not below the ordinary rest (paging earlier is noise). |
| zombie | Still unresolved: the broker is asked; only if it answers and does not list the order is CLOSING reverted to FILLED so a fresh exit can fire (otherwise held CLOSING and paged). | `exit.zombie_exit_revert_seconds` (900) | 120 s hard floor, and **>= 2 x the longer rest time** (600 s today): the cancel needs at least as long to resolve as the order was given to fill, or a fill that is merely not yet visible is re-armed as a duplicate close. Raising a rest therefore raises this and the foreign threshold too; startup names each. |

**Entry order**

| Age | What happens | Key (default) | Floor and why |
|---|---|---|---|
| every step | An unfilled entry combo's limit steps toward marketable (0.25 / 0.60 / 1.00 of the offset). | `positions.entry_ladder_step_seconds` (45) | 30 s = the tick; below it steps are skipped, not honored. Observed: the first reprice of a real order fired at +49 s because it is only seen on a tick. |
| partial | A PARTIALLY filled entry has its remainder cancelled so legs are not orphaned. | `positions.partial_fill_cancel_seconds` (30) | 5 s; below the entry timeout. |
| absent | An entry ABSENT from the trade list is treated as still pending while younger than this (a disconnect makes every working entry look cancelled for a moment). | `positions.absent_order_grace_seconds` (30) | 10 s; below the entry timeout. |
| timeout | An unfilled entry is cancelled. | `positions.entry_order_timeout_seconds` (240) | 30 s. |

**What the floors rest on, honestly.** One entry order (2026-09-28) was acknowledged by IBKR
in under 1 s; two more (QQQ and SPY, Prithvi's box) filled in 11.4 s and 14.4 s; no exit
order has been timed and no mid-session disconnect occurred in the retained logs. The floors
are engineering margins. The `order_ack_latency` log event (entry and exit, sub-second, from
ib_insync's own `Trade.log`) and `order_absent_judged` collect the real distribution and are
published to the `shared-logs` branch; revisit the floors once a few dozen paper orders,
including exits, exist.

## Short-DTE friction collection (W76; research only, read-only against the broker)

The research engine had no real bid/ask sample below 11 DTE, so every 0-6 DTE leg was charged an
extrapolated half-spread. IB serves historical BID/ASK bars for currently LISTED contracts only
(nothing for expired ones), so this data cannot be backfilled: it has to be collected forward,
one RTH session at a time, while the 0-6 DTE contracts are still listed.

**Each trading day** (paper Gateway/TWS is fine; it only requests historical data, never orders).
The port defaults to `IBKR_PORT` from `.env` (7497 for TWS paper, 4002 for Gateway paper), the
same value the bot uses; pass `--port` only to override. A "could not connect" error almost
always means TWS/Gateway is not running on that port:

```bash
# 1) see the plan first (no data requested, nothing stored)
python scripts/calibrate_option_spreads.py --source ib-bars --short-dte --symbols QQQ \
    --client-id 91 --dry-run

# 2) collect + fit (about 40-55 min per symbol: IB paces historical requests at 60 / 10 min)
python scripts/calibrate_option_spreads.py --source ib-bars --short-dte --symbols QQQ \
    --client-id 91
```

- **When:** after the close (16:10 ET or later) so today's 0-DTE contract has a full session of bars,
  while it is still listed. If a run shows `DTE 0: 0 samples`, IB had already dropped the expiring
  contract: run it instead ~15:45 ET on expiry days (the last hour's bars are then missing, which
  does not matter: the engine flattens at ~15:00).
- **Client id:** use one that is not the bot's (the bot uses 326 and its fallback ladder); the example
  uses 91, the repo's calibration convention. W66 verified this request type against the paper
  Gateway (historical BID and ASK bars of listed options, market closed). If a run
  returns no bars for every contract, that is an account/permission question for IB, not a bug here.
- **Coverage:** the run prints samples and distinct sessions per DTE 0..6 and, per band
  (`0-2`, `3-6`), `collecting: N/15 sessions (M more needed)` until `ACTIVE`.
- **Target:** at least 15 sessions per band (`backtest.friction_min_sessions_per_band`). Weekends
  and holidays do not count; missing a day only delays the date, nothing is lost.
- **Failure:** a pacing violation backs off 60 s and continues; an unqualifiable contract is skipped.
  Re-running the same day is safe (`INSERT OR IGNORE`).
- **Using it:** set `backtest.friction_short_dte: true` in `config_research.yaml`. A band below the
  minimum is charged the all-DTE model floored at `friction_extrapolated_floor_usd` ($0.01 per leg)
  and the run logs `friction_summary` (fitted / extrapolated counts). Off by default.
- **Storage:** `data/historical.db` tables `option_spread_bar_samples` (raw) and
  `option_leg_spread_band_params` (fits). Both are research tables; the live bot never reads them.

## Research-vs-reality parity replay (W72; research only, never touches live state)

Runs known real trades through the RESEARCH engine's pricing and exit code and reports where
research disagrees with what actually filled. Use it before trusting a DTE band in an Optuna
study, and after any change to `backtesting/pricing.py`, `friction.py` or the engine's exit path.

```bash
python scripts/parity_replay.py                 # reads data/replay/manual_trades_*.json
```

- **Inputs (all read-only):** the replay file (`data/replay/`, git-ignored; never commit it),
  `data/historical.db` research tables, and the bot's combo-fill context in `data/ait_state.db`.
- **Outputs:** `data/replay/out/<YYYYMMDD>/` (`summary.md` = statistics only and safe to quote;
  `trades.csv` / `legs.csv` carry per-trade prices and stay local). The CLI refuses any output
  directory inside the repo outside `data/replay/`.
- **Reading it:** error = model - actual, so positive means research is RICHER than reality.
  `engine` = what the engine does today (daily close + same-day IV); `clean` = what was knowable at
  the fill. A band with fewer trades than `parity.min_trades_per_band` is INSUFFICIENT, never PASS.
- **Tolerances:** `parity:` in `config_research.yaml` (operator-owned).
- **Never paste** per-trade prices, P&L or any account identifier into PLAN.md: it is shared via
  git with the other box. Report error statistics by DTE band only.

## Quarterly drills (scheduled — first week of Jan / Apr / Jul / Oct)
Run both, then record the date in the log table below. An untested kill
switch or backup is a hope, not a control.

**Kill-switch drill** (~5 min, during RTH on a quiet day):
1. `New-Item -ItemType File data\HALT`
2. Within one scan cycle (≤5 min) confirm the bot acknowledged it:
   `Select-String -Path logs\ait.log -Pattern "entries_frozen" | Select-Object -Last 1`
   (R13: the old `entries_halted` event only fired when a signal reached
   execution — on a gated day it NEVER logged and the drill read as a fail.
   `entries_frozen` now logs once per scan cycle whenever any HALT* file
   exists.)
3. Confirm exits are STILL managed while halted (open positions' marks keep
   updating on :8503 — HALT blocks entries only).
4. `Remove-Item data\HALT` and confirm normal scanning resumes.

**Backup-restore drill** (~10 min, outside RTH — proves the backups restore):
1. Latest snapshot:
   `Get-ChildItem data\backups\ait_state.*.db | Sort-Object Name | Select-Object -Last 1`
2. Restore to scratch and verify integrity + row parity:
   `Copy-Item data\backups\ait_state.<YYYYMMDD>.db $env:TEMP\restore_test.db`
   `python -c "import sqlite3; c=sqlite3.connect(r'$env:TEMP\restore_test.db'); print(c.execute('PRAGMA integrity_check').fetchone()); print('trades:', c.execute('SELECT COUNT(*) FROM trades').fetchone()[0]); print('open:', c.execute('SELECT COUNT(*) FROM open_positions').fetchone()[0])"`
   (integrity must be `('ok',)`; trades count within a day's worth of the live DB)
3. Confirm the off-box mirror is fresh:
   `Get-Item $HOME\Documents\ait_backups\ait_state.latest.db | Select-Object LastWriteTime`
4. `Remove-Item $env:TEMP\restore_test.db`

| Drill | Last run | Result |
|---|---|---|
| Kill switch | _(not yet run — next quiet RTH day)_ | |
| Backup restore | 2026-07-11 (R11) | PASS |

## Go-live day checklist (from PLAN.md gates — do not skip)
- [ ] Paper verdict passed (≥50 closes, PF > 1.3, criteria fixed beforehand)
- [ ] Remove `AIT_ALLOW_UNDEFINED_RISK` (strangles refuse at the executor)
- [ ] Revert paper-relaxed liquidity gates: `AIT_LIQ_MAX_SPREAD` ≤ 0.15,
      `AIT_LIQ_MIN_VOLUME` ≥ 10, config `max_bid_ask_spread_pct` ≤ 0.15
      (startup asserts this in live mode and will refuse to run otherwise)
- [ ] simulated capital OFF: `account.simulate_live_capital: false` in config.yaml (W63; was `AIT_SIMULATED_CAPITAL` in .env — also make sure no `AIT_SIMULATED_CAPITAL` override remains in .env); account funded $3k CAD
- [ ] Fresh DB backup taken; PLAN.md gates re-read

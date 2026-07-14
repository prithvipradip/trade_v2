# AIT v2 — Operator Runbook

One page: what each alert means and what YOU do. The bot manages exits on its
own; your job is responding to these messages. (Governance audit GOV-5.)

## Kill switches (no `kill -9` needed)
| Action | How |
|---|---|
| Stop NEW entries (exits keep managing) | create empty file `data/HALT` |
| Resume | delete `data/HALT` |
| Ban one symbol | add ticker line to `data/RESTRICTED.txt` |
| Full stop | `data/HALT` first, then stop processes if truly needed |

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
2. In TWS, CANCEL the bot's working exit orders on that position — or stop the
   bot processes if you're closing the whole book.
3. Do your manual close(s). Verify fills in TWS.
4. Positions NOT in the bot's books (untracked): safe to close anytime; the
   bot has no exit path referencing them (verified). Then delete
   `data/HALT_UNTRACKED` once the broker is flat of them.
5. Restart/resume the bot; the next reconcile books your closes (note: if you
   flattened the LAST option position, the zero-options mass-close guard
   refuses to book it — expect an `EOD RECON` break and resolve via
   close/adopt with the assistant).

## Daily 2-minute routine
1. Glance at Telegram: any non-CLEAN EOD recon? any CRITICAL?
2. Open http://localhost:8503 — positions green/red, unrealized total sane?
3. Friday extra: confirm the off-box mirror has CURRENT CONTENT (timestamps
   lie — `copy2` back-dates them; the backup job now hash-verifies, so any
   `STATE DB BACKUP FAILED` alert this week means the mirror is suspect):
   `python -c "import sqlite3; print(sqlite3.connect(r'file:C:/Users/prith/Documents/ait_backups/ait_state.latest.db?mode=ro', uri=True).execute('SELECT MAX(exit_time) FROM trades').fetchone())"`
   — must show this week's latest close. (And that folder syncs to cloud.)

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
- [ ] `AIT_SIMULATED_CAPITAL` removed; account funded $3k CAD
- [ ] Fresh DB backup taken; PLAN.md gates re-read

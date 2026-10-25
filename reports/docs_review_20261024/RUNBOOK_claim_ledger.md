# Claim ledger: docs/RUNBOOK.md (after fixes)

Format: claim | source (file:line as read in this session) | result. "FIXED" = the document was wrong and was corrected.

## Kill switches
- HALT blocks entries, exits keep running | orchestrator.py ~3380-3390 (halt flag check), entries_frozen log 1655 | OK
- RESTRICTED.txt one symbol per line | orchestrator.py:175,197 | OK
- KEEPER_PAUSE read by keeper_ait.bat | keeper_ait.bat:40-48 | OK
- HALT_UNTRACKED clears itself | reconciler.py:1672-1690 ("then this file clears itself") | FIXED (was "delete it")

## Alert rows (alert text matched against code; action checked against behaviour)
- BOT DOWN: supervisor keeps running, retries after >30 min; second supervisor refused | master.py:1268-1285, singleton lock master.py:338-354 | FIXED
- bot restarting (>=3 alert) | master.py:1280-1288 | OK
- BOT RELAUNCH FAILED / BOT HUNG / REFUSING (x2) / SINGLETON | master.py:1037,1137,946,959,493 (text matched) | OK
- cannot start bot (30 min throttle) | master.py:983-992 | FIXED (budget claim removed)
- IB Gateway unreachable ... No budget spent; recovery after grace | master.py:1239-1262, 777-790, constants 222-224 | FIXED (IBC guess removed)
- Gateway recovery rows | master.py:806,835,876,882 (text) | OK
- IB port fallback; ib_endpoint.json; status page row; digest line | master.py:395-427, data/ib_endpoint.py:17,47-53, status_server.py:56, master.py:2028 | OK
- sync_ibkr_port.py (--apply only IBKR_PORT) | scripts/sync_ibkr_port.py:5,12,58 | OK
- dashboard NOT started | master.py:1354 | OK
- lifecycle notices, Telegram RECOVERED hourly probe, not re-delivered | master.py:2409-2433 | OK
- READ-ONLY (Error 321) session-level; restart Gateway, log in as IB API | orchestrator.py:3163, ibkr_client.py:371-384 | FIXED
- CANNOT VERIFY TRADING | orchestrator.py:3151 | OK
- CIRCUIT BREAKER cadence, persistence, daily untrip; status page does not show it | orchestrator.py:1655-1679, circuit_breaker.py:305-319, status_server.py (no match) | FIXED (":8503" removed)
- DAILY MTM LOSS HALT clears on daily reset | circuit_breaker.py:403-411 | OK
- MTM BRAKE NOT EVALUATING / BASELINE DEGRADED | orchestrator.py:1411, 2691 (text) | OK
- LOOP IMPAIRED | orchestrator.py:3074 | OK
- WATCHDOG ALERT (3 components, 10 errors, ibkr recovery callback, hourly per component) | watchdog.py:157-179,281-303; orchestrator.py:473-516, 483-486 | FIXED
- ACCOUNT DATA STALE (HALTED / COULD NOT HALT) | broker/account.py:140-175 | OK
- ERROR: (waits 60 s) | orchestrator.py:748-756 | OK
- CAP-GRID INCOHERENT | orchestrator.py:3000-3024 | OK
- capital base warning | orchestrator.py:716 | OK
- VIX outage (IB then Yahoo, 45 min last-known-good) | orchestrator.py:3088-3120, market_data.py:449-490 | OK
- IV / EARNINGS / DIVIDEND / ENTRY GATE / CALENDAR / COOLDOWN | orchestrator.py:2039,2252,3566,3448,3187,4866 (text) | OK; calendar path FIXED to src/ait/data/economic_calendar.py
- RANGE GATE UNAVAILABLE only when ml.entry_gates_enabled | orchestrator.py:2470-2500 | OK
- MARKS MISSING / STALE EXIT FEED / TOUCH STOP / PDT | portfolio.py:549,318,354,1131 (text) | OK; 10197 reworded as "one common cause"
- EXIT REFUSED (not at broker): books trade itself | orchestrator.py:3840-3875, reconciler.py:577-606 | FIXED
- EXIT REFUSED (partial) / EXIT DEFERRED x3 / REPEATEDLY REJECTED / 8-K / TERMINAL | orchestrator.py:3888,4456,5373,5403,5431,6095,1469,917 (text) | OK
- order_rate_limit_tripped CRITICAL | executor.py:468 | OK
- RECONCILE NEEDS ATTENTION (assignment OR PARTIAL legs) | orchestrator.py:880-890, reconciler.py:1615,1835 | FIXED
- UNTRACKED OPTION POSITION (2 sightings, prompt, yes/no, clears itself) | reconciler.py:1515-1575, manual_positions.py:1-50,235-270 | FIXED
- Position discrepancies found (deferred/pre_market only) | orchestrator.py:766-811 | FIXED
- broker NOT connected | orchestrator.py:781 | OK
- EOD RECON counts discrepancies incl. booked ones | orchestrator.py:5520-5535, reconciler.py:1847 | FIXED
- model retrain exit 3 before any write | master.py:1546, retrain_ib.py:14 | OK
- STATE DB BACKUP FAILED; backups ait_state.YYYYMMDD.db; mirror ~/Documents/ait_backups | master.py:1807-1845,1862 | OK; mirror command FIXED (was Windows path)
- Short-DTE friction row | master.py:1792 | OK
- GO-LIVE SCORECARD readout / DIGEST (09:35, 16:05; Fri 16:10) | master.py:2476-2490, 2208 | OK
- status.py prints scorecard | status.py:23,260 | OK

## Books-versus-broker subsection (new)
- reconcile phases: startup, pre-market, post-market, deferred, on prompt answer | orchestrator.py:620,686,766-811,5483; main.py:175-190 | OK
- exit path fresh liveness; gone -> no order, book, alert | orchestrator.py:3840-3875 | OK
- zero-options guard + fresh re-query | reconciler.py:1737-1800 | OK
- option debounce / stock immediate | reconciler.py:1531-1568, 1578-1620 | OK
- declared position counted in risk, confirms when gone | manual_positions.py:1-50, 463-483 | OK
- expired -> intrinsic or needs_review | reconciler.py:500-545 | OK
- booking P&L broker-derived or $0 sentinel; sentinels excluded from verdict | reconciler.py:424-545, go_live.py:15-40 | OK

## Manual intervention / routine / deploy / data / troubleshooting / drills / go-live
- rationale rewritten to match exit-path liveness check | orchestrator.py:3840-3875 | FIXED
- shared-logs fetch Mon-Fri 16:50 ET, AIT_LOG_OWNER, read_shared_logs flags | master.py:2491-2500, shared_logs.py:15,45, read_shared_logs.py | OK
- smoke 14 vs 13 checks, --check-live-corpus, --telegram | smoke_deploy.py:707, 715-745 | OK
- restate_w21 --apply | restate_w21_tz_aware.py:24,64 | OK
- iv_refresh in retrain; summary line; backfill --mode vix/iv | master.py:1567, iv_refresh.py:140, backfill_historical_data.py:3-16,74 | OK
- unify_history_tables flags | scripts/unify_history_tables.py | OK
- entries_frozen log | orchestrator.py:1655-1657 | OK
- logs/ait.log, bot_stdout.log, orchestrator.log | utils/logging.py, master.py:105,1008 | OK
- go-live: paper or live is decided by config/launch_mode.py (config + .env TRADING_MODE + AIT_ALLOW_LIVE + live endpoint/account) in BotManager.start and run_bot; --paper is no longer hardcoded | master.py (BotManager.start), main.py (run_bot), launch_mode.py, tests/test_launch_mode.py | CHANGED after review (the hardcoded flag never selected the account; the IB port does)
- paper_trading_mode MUST be false | config.yaml:870-875 | OK
- live floors 0.15 / 10 | invariants.py:43-44 | OK
- 10 s abort window | main.py:128-131 | OK
- TRADING_MODE is now read (launch_mode.py); IBC mode still comes from IBC's own config | gateway.py:99-135 | CHANGED
- AIT_MARKET_DATA_TYPE .env only | runtime_env.py:22-56,132; ibkr_client.py:156 | OK
- go-live criteria thresholds | go_live.py:346-348 | FIXED (removed duplicate numbers, pointer to GUIDE section 12)

Unchecked / judgement (operator): whether each action is the best operational response; the quarterly-drill table dates (carried over from the old RUNBOOK, 2026-07-11 PASS).

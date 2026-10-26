# AIT v2 — Operator Runbook

One page: what each alert means and what YOU do. The bot manages exits on its own; your job is responding to these messages and keeping the box healthy.
Not for: how the system works ([GUIDE.md](GUIDE.md)), installing and first run ([GETTING_STARTED.md](GETTING_STARTED.md)), the models ([MODELS.md](MODELS.md)) or research ([RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)).
Limits, timers and ceilings these procedures refer to are generated tables in [GUIDE.md](GUIDE.md) sections 7 and 8; every setting is in [reference/SETTINGS_LIVE.md](reference/SETTINGS_LIVE.md). Terms: [GLOSSARY.md](GLOSSARY.md).

**Contents:** [Kill switches](#kill-switches-no-kill--9-needed) · [Alert → action](#alert--action) · [Books versus broker](#resolving-a-books-versus-broker-break) · [Manual intervention](#manual-intervention-procedure-closingadjusting-any-tracked-position-yourself) ·
[Daily routine](#daily-2-minute-routine) · [Deploy checklist](#deploy-checklist-every-deploy) · [Data catch-up](#data-catch-up-and-maintenance) · [Troubleshooting](#troubleshooting) ·
[Quarterly drills](#quarterly-drills-scheduled--first-week-of-jan--apr--jul--oct) · [Go-live checklist](#go-live-day-checklist-from-planmd-gates--do-not-skip)

## Kill switches (no `kill -9` needed)
| Action | How |
|---|---|
| Stop NEW entries (exits keep managing) | create empty file `data/HALT` |
| Resume | delete `data/HALT` |
| Ban one symbol | add ticker line to `data/RESTRICTED.txt` |
| Pause auto-relaunch (keeper_ait.bat) | create empty file `data/KEEPER_PAUSE` |
| Resume auto-relaunch | delete `data/KEEPER_PAUSE` |
| Full stop | `data/KEEPER_PAUSE` FIRST (or the keeper relaunches within ~90s and undoes the stop), then `data/HALT`, then stop processes if truly needed |

**What a clean bot stop looks like.** After SIGTERM (a supervisor restart, Ctrl-C, or the supervisor's own stop) the bot logs `shutdown_signal_received`, then `orchestrator_shutting_down`, sends the "Bot shutting down" page and logs `ait_shutdown_complete` within seconds. The supervisor logs `bot_stopped` with `escalated=false`. It waits up to 75 s (`SUPERVISOR_STOP_WAIT_SECONDS` in `utils/cadence.py`, which must exceed the bot's own 30 s grace + 15 s shutdown step + 15 s cleanup; a test holds that) and only then logs `bot_stop_escalated_to_kill` and kills the bot. If you see that line, or `graceful_shutdown_timed_out_cancelling`, `shutdown_step_timed_out` or `shutdown_cleanup_timed_out`, the stop was not clean: look at what the bot was doing (a scan in flight cannot be interrupted) and tell a developer session. Working exit orders are left at the broker and re-adopted at the next start.

`data/HALT_UNTRACKED` is set by the bot itself (see "UNTRACKED OPTION POSITION" below); do not create it by hand. It clears itself at the next reconcile once every untracked position is declared or gone ([Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break)).

## Alert → action

Alerts arrive on Telegram; most are also in `logs/`. The text before the first colon or dash is the key. `{...}` varies. A row says what it means and what to do; "nothing" means informational.
Every refusal below blocks **entries only**; open positions keep being managed unless the row says otherwise.

### The bot or the broker connection is down
| Alert | Meaning | Do this |
|---|---|---|
| `BOT DOWN — gave up after N restart attempts` | The supervisor used its restart budget and stopped restarting the bot. **The supervisor itself is still running** | Read the tail of `logs/bot_stdout.log` and fix the cause (Gateway not logged in, a config error, a traceback). The supervisor tries again by itself once more than 30 minutes have passed since its last restart attempt. To restart sooner, do a full stop and start (Kill switches, "Full stop"): `python run_orchestrator.py` is refused while the old supervisor holds the lock |
| `bot restarting (attempt n/N, exit_code=…)` | The bot exited and was restarted | Nothing for one or two; by attempt 3 something is wrong: read `logs/bot_stdout.log`. A restart does not fix a code bug: capture the traceback first |
| `BOT RELAUNCH FAILED: the process exited within Ns` | The new bot died at launch | `logs/bot_stdout.log`; the keeper keeps retrying, so fix the cause (config error, Gateway) |
| `BOT HUNG: trading-loop heartbeat N min stale` | The loop stopped answering; positions were unmanaged for that window; the supervisor restarted it | Check it came back (:8503). If it repeats, `data/HALT`, capture the traceback, then follow the deploy checklist |
| `REFUSING to start the trading bot: …` (two variants) | Another `ait.main` is already running, or the process table is unreadable; a second bot would double-trade | Verify the PID, kill the orphan by hand (never auto-killed), and the supervisor starts a fresh bot on the next health cycle; if "unreadable", check permissions on the host |
| `REFUSING to start: the mode is paper but IBKR_PORT … is a LIVE-account endpoint`, `REFUSING to start LIVE: …` (no `AIT_ALLOW_LIVE`, not a live-account endpoint, or a paper account id), `… could not decide paper or live` | The mode and the IB endpoint disagree, or live was requested without the interlock. Nothing was started. Repeats are throttled to one alert per 30 minutes | Make `config.yaml` `trading.mode`, `.env` `TRADING_MODE`, `AIT_ALLOW_LIVE`, `IBKR_PORT` and `IBKR_ACCOUNT` agree: paper needs a paper endpoint (4002 / 7497); live needs all of the go-live conditions. The supervisor retries on the next health cycle |
| `SINGLETON LOCK UNAVAILABLE` | Supervisor cannot take its lock (permissions or disk) and refused to start | Check `data/` permissions and free disk space, then start again |
| `cannot start bot — IB Gateway not available on port N` | Bot start deferred: Gateway/TWS is not answering | Is Gateway/TWS logged in? Start it. This alert repeats at most every 30 minutes |
| `IB Gateway unreachable on port N (likely daily maintenance)` | Normal overnight restart | Nothing for 30–60 min (IB restarts nightly); the alert says no restart budget is spent. If the port stays closed the supervisor tries to bring Gateway back itself after a grace period (next rows) |
| `IB Gateway was down N min — attempted automatic restart (k/K)` | The supervisor tried to bring Gateway back | Nothing; the bot resumes on its own once the port answers |
| `IB Gateway STILL down … MANUAL ACTION NEEDED`, `IB Gateway/TWS has not answered … no automated recovery`, `… automatic restart could not be launched` | Automatic recovery is exhausted or unavailable | Look for a login or 2FA prompt on the Gateway window; start Gateway/TWS by hand |
| `IB port fallback: .env IBKR_PORT=… but … is what is up` | `.env` names TWS but Gateway is up (or the reverse). The bot and supervisor use the other endpoint of the **same account type** (never across), so nothing is down. The status page ("IB endpoint"), the digest and `ib_port_fallback` keep showing it | Fix `.env` when convenient: `python scripts/sync_ibkr_port.py` (dry run), then `--apply` (changes only the `IBKR_PORT` value), then restart the supervisor. `data/ib_endpoint.json` is the supervisor's record; never edit it |
| `dashboard NOT started: port N is held by a process the supervisor won't kill` | A foreign process owns a dashboard port | See `dashboard_port_held_foreign` in `logs/orchestrator.log` for the holder; free the port or change `dashboard.*_port` |
| `supervisor started`, `BOT STARTED \| Mode: …`, `Bot shutting down`, `Telegram channel RECOVERED` | Lifecycle notices. After a Telegram outage, alerts sent while it was down were **not** re-delivered | Nothing; after "RECOVERED" check `notification_dropped_after_retries` in the logs |

### Can it trade?
| Alert | Meaning | Do this |
|---|---|---|
| `READ-ONLY: IB Gateway is rejecting all orders (Error 321)` | The bot's test order was rejected as read-only, so **nothing can trade or exit**. The code's own diagnosis is a Gateway login that came up read-only (a session state rather than the Read-Only API setting) | Restart Gateway and log in as IB API, as the alert says. Also keep the Read-Only API setting unchecked ([GETTING_STARTED.md](GETTING_STARTED.md) section 3). The bot keeps re-probing and sends `Trading ENABLED again` when orders are accepted |
| `CANNOT VERIFY TRADING: the read-only probe did not complete` | Unknown whether orders are accepted; a read-only session rejects silently | Check the Gateway window |
| `CIRCUIT BREAKER: Trading halted - {reason}` | Daily loss, consecutive losses or API failures tripped the breaker; **entries frozen** until it resets | Read the reason in the alert. It is sent once per trip and then hourly while the breaker stays tripped; `trading_halted` in `logs/` repeats it every scan cycle. The breaker persists across restarts and a daily-loss trip clears at the next ET day |
| `DAILY MTM LOSS HALT: day P&L … breached the daily-loss cap` | Entries auto-frozen for the day; exits still run | Review the book on :8503. Clears at the next ET day. If you decide to close a bleeder by hand, follow the manual-intervention procedure so your close and a working bot exit order cannot both fill |
| `MTM BRAKE NOT EVALUATING` | The daily-loss halt is inactive until fixed; entries are **not** blocked by it | Read the error in the message and `mtm_check_failed`; consider `data/HALT` until fixed |
| `MTM BASELINE DEGRADED` | Today's loss baseline was taken after the open; an overnight gap may be under-counted | Informational; the next clean pre-stamp fixes it |
| `LOOP IMPAIRED — {kind}: N consecutive failures` | The trading loop is erroring; **stops/take-profits may not be running** | Check the tail of `logs/bot_stdout.log` for the exception. If it repeats: `data/HALT` and restart per the deploy checklist (a restart does not fix a code bug: capture the traceback first) |
| `WATCHDOG ALERT: {component} is DOWN and recovery failed. Errors: N, Last: …` | The watchdog tracks three components: `trading_loop`, `ibkr` and `market_data`. One goes DOWN after 10 recorded errors, or after a missed heartbeat (`market_data` allows three scan intervals). `ibkr` is first retried by reconnecting; this alert means that failed. The other two have no automatic recovery | Read `Last:`. `ibkr`: check the Gateway login and connection. `market_data`: check the market-data feed and subscriptions. `trading_loop`: see `ERROR:` and `LOOP IMPAIRED` and `logs/bot_stdout.log`. At most one alert per component per hour |
| `ACCOUNT DATA STALE N min` | Risk uses a stale account snapshot; the breaker halts trading if it can ("COULD NOT HALT — halt manually" means create `data/HALT`) | Check the Gateway connection and that the account is the one you expect |
| `ERROR: {…}` | An unhandled error in a phase; the bot waits a minute and continues | Read the log around it; repeated errors: `data/HALT` and investigate |
| `CAP-GRID INCOHERENT at NLV $…` | The startup/size self-test found the per-trade budget, breaker or caps incoherent (entries may silently build nothing) | Fix the named knobs (`risk.max_position_risk_pct`, `positions.max_position_pct`, breaker); ranges and links: [GUIDE.md](GUIDE.md) section 7.4 |
| `WARNING: live-mode capital base is the HARDCODED DEFAULT` | Live mode with no capital base set; every risk % and drawdown reading is wrong | Set `AIT_CAPITAL_BASE` before trading live |

### Data and gate failures (all fail closed: entries refused, exits unaffected)
| Alert | Meaning | Do this |
|---|---|---|
| `VIX DATA OUTAGE` | No VIX print from either source for longer than the last-known-good window; credit entries refused | Check the market-data feed and Yahoo/IB VIX; clears itself |
| `IV DATA OUTAGE — {symbol}` | IV rank could not be measured for that symbol | Check IB option-IV data for the symbol; clears itself |
| `EARNINGS DATA UNAVAILABLE for {symbol}` / `DIVIDEND DATA UNAVAILABLE for {symbol}` | No confirmed calendar answer and no usable last-known-good | Check the data source; clears itself when a fetch succeeds |
| `RANGE GATE UNAVAILABLE` | In enforce mode, the range model produced no prediction | Check `models/range.pkl` exists and loaded; re-run the retrain (below). Irrelevant in observe mode |
| `ENTRY GATE BROKEN — the pre-event blackout check has failed N times` | The macro-calendar check is erroring; credit entries are refused | Read the error in the message; check the calendar tables in `src/ait/data/economic_calendar.py` |
| `ECONOMIC CALENDAR NEARLY EXHAUSTED` | The hard-coded FOMC/CPI/NFP/GDP/PCE tables are about to run out; the event guards would go blind | Extend the tables in `src/ait/data/economic_calendar.py` |
| `COOLDOWN CHECK FAILED ({symbol})` | The post-stop re-entry guard could not run; credit entries blocked | Read the exception in the log; clears when it can run |

### Positions and exits
| Alert | Meaning | Do this |
|---|---|---|
| `MARKS MISSING xN ticks` | No option marks: stop/take-profit protection is **not running** (only DTE safety exits) | One common cause: the live account is logged in elsewhere (mobile or web takes the single market-data slot, Error 10197): log it out. Otherwise check the Gateway data lines |
| `STALE EXIT FEED: {symbol}` | The underlying quote stopped advancing; the touch stop now needs corroborating ticks | Check the market-data feed |
| `TOUCH STOP NOT EVALUATING` | The short-strike touch check raised an error: the only loss exit on a credit structure is off for that position | Check the trade's legs data in :8503 and the log; consider closing by hand via the manual-intervention procedure |
| `PDT BLOCKED EXIT` | A stop wanted to fire but PDT rules block a same-day close; the position rides until tomorrow | Decide: accept the overnight risk, or close manually (accepting the PDT strike) |
| `EXIT REFUSED (position not at broker)` | The bot wanted to exit (stop, target or time) but a fresh broker query shows the legs are gone, usually because you closed them. It placed no order and booked the trade itself | Usually nothing: the alert ends "Booked closed from broker records." Compare the booked P&L with your TWS fill. If it ends "booking it failed — the row needs manual review", the trade is still open in the books: see [Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break) |
| `EXIT REFUSED (partial structure)` | Only some legs are at IBKR (manual close? assignment?); reversing all would open inverted positions | **Close the remainder at IBKR yourself** |
| `EXIT DEFERRED (no quotes)` / `(no quotes, no mark)` / `EXIT WITHOUT QUOTES` | No combo NBBO to price a close; the bot will not market-order a multi-leg or short blind. It retries each cycle, or prices from a fallback with a cap | Check the market-data feed; if it repeats, consider closing by hand |
| `EXIT REPEATEDLY REJECTED: {trade}` | Several fast rejects: the contract may be halted or too illiquid at the quoted price | Check it manually at the broker |
| `SEC 8-K filed for {symbol} — flattening positions` | A material filing triggered a flatten of that symbol | Watch the exit complete on :8503 |
| `CRITICAL: fill landed on a TERMINAL trade row` | A real position may be live at the broker with no managing row | Check TWS and the untracked-option freeze immediately |
| `order_rate_limit_tripped` (log, CRITICAL) | More orders in a minute than the fixed backstop allows: a malfunction | Create `data/HALT`, inspect logs, do **not** just restart |

### Books versus broker
| Alert | Meaning | Do this |
|---|---|---|
| `RECONCILE NEEDS ATTENTION: …` | The reconcile found either an assignment (`UNTRACKED STOCK POSITION (possible ASSIGNMENT)`: stock appeared) or an incomplete structure (`PARTIAL legs missing for <trade>: [...] — possible assignment`: some legs are gone at the broker, by a manual close or an assignment) | Assignment: in TWS liquidate the assigned stock and verify the remaining legs; the freeze clears itself at the next reconcile once the stock is gone or declared. Partial legs: the bot never closes half a structure, so close the remainder at the broker, then see [Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break) |
| `UNTRACKED OPTION POSITION … ENTRIES FROZEN` | An option position at the broker has no record in the bot's books (a trade you placed yourself, or an order that filled in a crash window). It was seen on two separate reconcile passes, so new entries are frozen (`data/HALT_UNTRACKED`) and a Telegram prompt asks whether it is yours | Reply **yes** (or tap "Yes, mine") if you placed it: the bot will not manage it but still counts it in its risk totals, and the freeze lifts at once. Otherwise close it at the broker; the file clears itself at the next reconcile. "No, investigate" keeps the freeze. See [Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break) |
| `exit_unbid_wing_left_out` (log, WARNING) / `expected_residual_leg_seen` (log, info) | A condor close left a long wing out of its closing order because nobody bids for it (IB quotes the bid as -1) and it is offered at `exit.exit_unbid_wing_max_ask` or less: an order that holds a leg with no bid never fills. The wing stays at the broker and expires worthless; the reconciler expects exactly that leg at exactly that quantity, so it does **not** freeze entries and sends no prompt, and it forgets the record after the leg is gone | Nothing. A different quantity, or a record older than one day past the expiry, is treated as an untracked option as usual. If the wing ever ends in the money, IB exercises it at expiry and the resulting stock position freezes entries like any assignment. `exit.exit_leave_out_unbid_wings: false` puts every leg back in the order |
| `Position discrepancies found` | A pre-market (or deferred) reconcile found differences between the books and the broker | Compare :8503 with the TWS portfolio; the usual cause is a manual action at the broker. See [Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break) |
| `{PHASE}: broker NOT connected — positions were NOT reconciled` | A reconcile ran against a disconnected broker; **this is not a clean result** | Fix the connection; the bot reconciles as soon as it returns |
| `EOD RECON: N BREAK(S) …` | The end-of-day reconcile listed N findings and shows the first three. A finding is not always an open problem: a trade you closed by hand appears as `Local position not in IBKR (likely closed)` and is booked in the same pass | Read the lines under it, compare :8503 with TWS, and follow [Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break) |
| `EOD RECON: CLEAN \| … gap stress: $X` | Daily all-clear | Glance at the −10% gap-stress number; if it nears your pain threshold, reduce the short-premium count |

### Models, backups and reporting
| Alert | Meaning | Do this |
|---|---|---|
| `model retrain FAILED (exit N)` / `model retrain HUNG` | Models are stale (not urgent same day) | Re-run `python run_orchestrator.py --retrain`. The retrain takes its daily bars from IB only (own session) and aborts with exit 3 **before any write** when TWS/Gateway is down, the client id is refused, or IB returns no daily bars (`retrain_*.log` says which): bring TWS/Gateway up first |
| `STATE DB BACKUP FAILED` | The track record is unprotected | Check disk space; make a manual copy of `data/ait_state.db` |
| `Short-DTE friction collection {status}` | The optional research collection failed; that day's samples may be lost for good | Only if you enabled it: see [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) section 10.1 |
| `GO-LIVE SCORECARD: GATE READOUT FAILED` | The Friday scorecard could not be computed | Run `python status.py` for the scorecard; read the error |
| `DIGEST: …` and `GO-LIVE SCORECARD (since …)` | The morning and evening summaries and the Friday scorecard. A **missing** morning digest is itself an alarm | Read them; investigate a missing one |

## Resolving a books-versus-broker break
The bot keeps its own books (`data/ait_state.db`); the broker holds the truth. Some trades are yours, placed at the broker, not the bot's. The **reconciler** compares the two and treats the broker as right. It runs at startup, pre-market, post-market (this produces `EOD RECON`), again if an earlier pass found the broker disconnected, and at once when you answer a Telegram prompt below. Separately, before the bot acts on any exit it checks that trade's legs against a fresh broker query.

| What happened | What the bot does by itself | What you do |
|---|---|---|
| You closed **all legs** of a bot trade by hand | If an exit then fires first (stop, target, time), the exit path confirms the legs are gone, places **no order**, books the trade and sends `EXIT REFUSED (position not at broker)`. Otherwise the next reconcile books it (`Local position not in IBKR`). If the broker shows no option positions at all, it first re-queries the broker and books only if that confirms the book is flat; if the broker does not answer, it books nothing and tries again at the next pass | Nothing to repair. Compare the booked P&L with your TWS fill (below) |
| You closed **only some legs** | Nothing is closed. The reconcile reports `PARTIAL legs missing for <trade>` (`RECONCILE NEEDS ATTENTION`) and an exit that fires is refused (`EXIT REFUSED (partial structure)`) | Close the remainder at the broker. The trade is then booked as in the row above |
| You **opened** an option or stock position the bot has no record of (also: a fill during a crash window, or an assignment) | An **option** is first reported as a break (`UNTRACKED OPTION POSITION (1st sighting, freeze deferred)`); if the next reconcile pass still sees it, entries freeze (`data/HALT_UNTRACKED`) and a Telegram prompt is sent. A **stock** position (usually an assignment) freezes at once (`UNTRACKED STOCK POSITION`). The bot never closes either | Reply **yes** if it is yours: the bot ignores it for management but still counts it in its risk totals (capital at risk, concentration, open positions), and the freeze lifts at once. Or close it at the broker. The file clears itself at the next reconcile once the position is declared or gone. When a declared position later disappears the bot tells you and forgets it |
| A **bot position expired** while the books still show it open | The reconcile books it at intrinsic value (`reconciler_expired_intrinsic`), or as `reconciler_unknown_exit_expired_needs_review` ($0) when the underlying price is unavailable | Check the booked row if it says needs review |
| The bot could **not book** it (`booking it failed — the row needs manual review`) | The trade stays open in the books and a later reconcile tries again | If it is still open after the next end-of-day reconcile, a developer session must close the row; see "What to give a developer session" |

**How a vanished trade is booked.** The bot takes the realized P&L the broker reports for exactly that trade's legs, when every leg's figure is a usable number (exit reason `reconciler_ibkr_realized`). Otherwise it books **$0** with exit reason `reconciler_unknown_exit` and flags it for review. Rows booked as `reconciler_unknown_exit*` do not count in the go-live verdict (`reporting/go_live.py`), so a $0 placeholder cannot flatter or damage the track record; your TWS fills remain the truth for the real P&L.

**What to give a developer session** when a row must be closed by hand: the trade id, the broker's position or the fill you saw in TWS (price, time, quantity), and the state of the bot (stopped, with a copy of `data/ait_state.db` taken first). Do not edit the live database by hand: it is the track record, and a trade that is already closed must never be closed twice.

## Manual intervention procedure (closing/adjusting ANY tracked position yourself)
The exit engine evaluates every tracked position every 30 seconds and does not know you are acting. Since it checks the broker before it exits, a manual close no longer makes it rebuild the position inverted: with all legs gone it places no order, books the trade and alerts, and with some legs gone it refuses and alerts ([Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break)). The risk that remains is a bot exit order that is *working* while you close: if both fill, the position is closed twice and an opposite position opens. So, in order:
1. `New-Item -ItemType File data\HALT` (freezes entries; exits keep managing).
2. In TWS, CANCEL the bot's working exit orders on that position. If you are closing the whole book, create `data\KEEPER_PAUSE` FIRST (else `keeper_ait.bat` relaunches the whole stack within ~90s and undoes your stop), then stop the bot processes.
3. Do your manual close(s). Verify fills in TWS.
4. A position that is **not** in the bot's books (untracked) can be closed any time; the bot has no exit path referencing it, and the freeze file clears itself at the next reconcile.
5. Restart/resume the bot. The books are corrected by the reconcile or the next exit, as in the table above; no manual repair is needed unless an alert says booking failed. If you created `data\KEEPER_PAUSE`, delete it too so the keeper resumes relaunching.

## Daily 2-minute routine
1. Glance at Telegram: any non-CLEAN EOD recon? any CRITICAL? Did the morning digest arrive?
2. Open http://localhost:8503: positions green/red, unrealized total sane?
3. Friday extra: confirm the off-box mirror has CURRENT CONTENT (timestamps lie; the backup job hash-verifies, so any `STATE DB BACKUP FAILED` alert this week means the mirror is suspect):
   `python -c "import os,sqlite3; p=os.path.expanduser('~/Documents/ait_backups/ait_state.latest.db'); print(sqlite3.connect('file:'+p+'?mode=ro', uri=True).execute('SELECT MAX(exit_time) FROM trades').fetchone())"`
   (the mirror is written under the home folder of the user running the supervisor). It must show this week's latest close.
4. Reading the OTHER box's logs: the supervisor fetches `origin/shared-logs` itself every weekday at 16:50 ET (a fetch only; it never touches the checkout and does not need `AIT_LOG_OWNER`). Its log line
   `shared_logs_fetched` carries `latest_by_owner`: a stale date there means that box has stopped publishing. Read a day with
   `python scripts/read_shared_logs.py --owner <name> --date YYYY-MM-DD --no-fetch` (drop `--no-fetch` to force a fresh fetch). Publishing your OWN days additionally needs `AIT_LOG_OWNER=<name>` in this box's `.env` (the job is silent without it).

## Deploy checklist (EVERY deploy)
A green test suite is not a safe deploy: unit tests once passed and the deployed code still killed the trading loop for a full session (stops/TPs unmanaged), found a day later. Do all five steps, in order:

1. **Tests green.** Locally (the fast lane, same selection as CI): `python -m pytest tests/ -q -m "not ibkr and not slow"`, and the CI run for the pushed commit is green (`.github/workflows/ci.yml` also byte-compiles all of `src` on every push).
2. **Deploy outside RTH** (before 09:30 or after 16:00 ET) whenever possible. If it truly must happen during RTH: `New-Item -ItemType File data\HALT` first, deploy, smoke (step 3), then `Remove-Item data\HALT`.
3. **Run the smoke, no exceptions:** `python scripts/smoke_deploy.py` must print `SMOKE PASS (…)`. It executes the runtime paths unit tests don't (imports, real-config load, real-DB row mapping, digests, learning state, wing sizing),
   read-only, no IBKR needed. Optional: `--telegram` sends the verdict.

   **One-off, on a box whose timestamp migration is still outstanding** (the Windows box at the last record; the macOS box is migrated). Stop the bot first; the script snapshots and writes:

   ```
   python scripts/restate_w21_tz_aware.py              # dry run: read the plan
   python scripts/restate_w21_tz_aware.py --apply      # snapshot + write
   python scripts/smoke_deploy.py --check-live-corpus  # 14 checks, not 13
   ```

   Do not treat the script's own exit code as proof: `--check-live-corpus` re-reads the DB independently and is what closes the migration. If it reports naive rows, the migration did not land; the snapshot path is printed by the `--apply` run.
   `--apply` aborts rather than relabelling any row written on or after the host move date; those need handling by hand.
4. **Confirm liveness after the next RTH open** (~09:35–09:45 ET). `python scripts/check_first_rth_liveness.py` makes the heartbeat, scan-event, LOOP IMPAIRED / cycle-error and marks checks below in one go, prints the verdict, sends it to Telegram (unless `--no-telegram`) and exits 1 on a failure. To check by hand:
   - fresh heartbeat: `Get-Item data\bot_heartbeat | Select-Object LastWriteTime` must be minutes old and refreshing;
   - one completed trading cycle in the logs: `Select-String -Path logs\bot_stdout.log -Pattern "scan_symbol_timing" | Select-Object -Last 3` (timestamps post-open), and no fresh
     `Select-String -Path logs\bot_stdout.log -Pattern "LOOP IMPAIRED|trading_cycle_error" | Select-Object -Last 5`;
   - no `LOOP IMPAIRED` Telegram alert arrived;
   - marks updating: dashboard :8503 unrealized P&L moving, or `python -c "import sqlite3; print(sqlite3.connect('file:data/ait_state.db?mode=ro', uri=True).execute('SELECT symbol, mark_time, unrealized_pnl FROM open_positions').fetchall())"` with `mark_time` within the last few minutes.
5. **Rollback path** = `git checkout <last-good-tag>` + restart the bot (stop processes, then `python run_orchestrator.py` / `keeper_ait.bat`). For this to exist: TAG every verified deploy: `git tag -a deploy-YYYYMMDD -m "smoke pass + first-RTH liveness OK"` then
   `git push origin --tags`. Only tag AFTER step 4 passes.

**A change to `orchestration/master.py`, or to the retrain it launches, needs a full supervisor restart**, not just a bot restart. The 07:30 ET retrain needs TWS/Gateway up (see its alert row); with it down the retrain aborts and the models stay as they were.

## Data catch-up and maintenance

- **The retrain is append-only** for stored daily bars: a stored completed session is never rewritten and today's partial bar is never stored. Backfill scripts overwrite only when you run them. The first successful IB retrain re-tags the last ~504
  `daily_prices` rows per symbol to `IB_TRADES_RTH`; check with `sqlite3 "file:data/historical.db?mode=ro" "SELECT source, COUNT(*) FROM daily_prices WHERE symbol='QQQ' GROUP BY source"`.
- **Implied vol stays fresh without a new job:** the 07:30 ET retrain, after its IB pre-flight, refreshes it for the universe and the VIX family (`iv_refresh QQQ:+2->… ` in `retrain_*.log`). A failure there is a warning, not a failed retrain; the next run fills the gap. Catch up
  by hand: `python scripts/backfill_historical_data.py --symbols QQQ SPY IWM --mode iv --years 1`, and the VIX family: `--mode vix --symbols VIX --years 8`. The table-by-table list of what writes what: [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) section 1.
- **The history-table merge tool** (`scripts/unify_history_tables.py`) merged the old `research_*` copies into the shared tables. Dry run by default; `--apply` backs up, merges (additive; research rows win on overlap, live-only rows kept) and verifies; `--verify`
  re-checks any time; `--archive` renames `research_X` to `archive_w91_research_X` (refused unless verification passes); `--drop-archives` drops them. It refuses to write the real database while the supervisor/bot runs (`--allow-running` overrides). Order:
  merge, merge the code and restart the supervisor, one clean retrain and a research run, THEN archive, THEN drop. Do not run it during the Sunday 20:00 weekly backtest or an option-history refresh.

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| "Competing live session" (Error 10197), or marks go missing | The live account is logged in elsewhere (mobile/web) and holds the single market-data slot | Log the other session out; paper data is delayed by design (`AIT_MARKET_DATA_TYPE=4`) |
| `Delayed market data` warnings (Error 10167) | The paper account has no live subscription for that exchange | Expected on paper. The bot works with delayed data; fills are modelled and may be less favourable |
| Entry orders do not fill | The combo limit starts near mid and steps toward marketable, then is cancelled after the entry timeout; an entry is refused outright if its combo quote is disorderly or stale (`combo_spread_disorderly_rejected`, `combo_signal_price_stale_rejected` in the log) | Look for those events and for `order_ack_latency`; the timeline and its timers are in [GUIDE.md](GUIDE.md) section 8.4 |
| Models look stale or missing at startup | The retrain is due or failed | The bot trains in the background and keeps managing positions meanwhile; force one with `python run_orchestrator.py --retrain` |
| Telegram alerts not arriving | Token or chat id wrong, or the channel is marked dead | Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`; test: `curl -X POST "https://api.telegram.org/bot<TOKEN>/sendMessage" -d "chat_id=<CHAT_ID>&text=test"`. The supervisor probes a dead channel hourly and announces `Telegram channel RECOVERED` |
| `No module named 'ait'` | The package is not installed in the active environment | Activate the environment and run `pip install -e . -c constraints.txt` from the project root ([GETTING_STARTED.md](GETTING_STARTED.md)) |
| `torch`, `transformers` or FinBERT import or install errors | They concern the retired sentiment extra, which nothing under `src/` imports | Ignore them; do not install the extra. The package installs without it ([GETTING_STARTED.md](GETTING_STARTED.md)) |
| The bot will not start | Gateway not listening, a stale process, or a lock | Check the port (`netstat -ano \| findstr 4002` on Windows); read `logs/orchestrator.log` and the alerts above; stop stale Python processes by PID, never blindly |

## Quarterly drills (scheduled — first week of Jan / Apr / Jul / Oct)
Run both, then record the date in the log table below. An untested kill switch or backup is a hope, not a control.

**Kill-switch drill** (~5 min, during RTH on a quiet day):
1. `New-Item -ItemType File data\HALT`
2. Within one scan cycle confirm the bot acknowledged it: `Select-String -Path logs\ait.log -Pattern "entries_frozen" | Select-Object -Last 1` (`entries_frozen` logs once per scan cycle whenever any `HALT*` file exists).
3. Confirm exits are STILL managed while halted (open positions' marks keep updating on :8503; HALT blocks entries only).
4. `Remove-Item data\HALT` and confirm normal scanning resumes.

**Backup-restore drill** (~10 min, outside RTH; proves the backups restore):
1. Latest snapshot: `Get-ChildItem data\backups\ait_state.*.db | Sort-Object Name | Select-Object -Last 1`
2. Restore to scratch and verify integrity and row parity:
   `Copy-Item data\backups\ait_state.<YYYYMMDD>.db $env:TEMP\restore_test.db`
   `python -c "import sqlite3; c=sqlite3.connect(r'$env:TEMP\restore_test.db'); print(c.execute('PRAGMA integrity_check').fetchone()); print('trades:', c.execute('SELECT COUNT(*) FROM trades').fetchone()[0]); print('open:', c.execute('SELECT COUNT(*) FROM open_positions').fetchone()[0])"`
   (integrity must be `('ok',)`; trades count within a day's worth of the live DB)
3. Confirm the off-box mirror is fresh: `Get-Item $HOME\Documents\ait_backups\ait_state.latest.db | Select-Object LastWriteTime`
4. `Remove-Item $env:TEMP\restore_test.db`

| Drill | Last run | Result |
|---|---|---|
| Kill switch | _(not yet run; next quiet RTH day)_ | |
| Backup restore | 2026-07-11 | PASS |

## Go-live day checklist (from PLAN.md gates — do not skip)
The criteria the scorecard grades are in [GUIDE.md](GUIDE.md) section 12; the decision to go live is the operator's, against `PLAN.md`.
- [ ] The paper verdict has passed against the criteria in [GUIDE.md](GUIDE.md) section 12 (fixed beforehand in `PLAN.md`)
- [ ] Remove `AIT_ALLOW_UNDEFINED_RISK` (strangles refuse at the executor)
- [ ] Revert paper-relaxed liquidity gates: `AIT_LIQ_MAX_SPREAD` ≤ 0.15, `AIT_LIQ_MIN_VOLUME` ≥ 10, config `max_bid_ask_spread_pct` ≤ 0.15 (startup asserts this in live mode and will refuse to run otherwise)
- [ ] Simulated capital OFF: `account.simulate_live_capital: false` in `config.yaml` (also make sure no `AIT_SIMULATED_CAPITAL` override remains in `.env`); account funded
- [ ] Fresh DB backup taken; PLAN.md gates re-read

**Configuration to confirm before going live** (the operator has reviewed these; each is read from the file named):
- [ ] Live needs all of these together, and the supervisor and the bot refuse to start live otherwise (`src/ait/config/launch_mode.py`): `trading.mode: live` in `config.yaml`, `TRADING_MODE=live` in `.env`, `AIT_ALLOW_LIVE=1` in `.env`, an IB endpoint of the live account (Gateway 4001 or TWS 7496) and, when `IBKR_ACCOUNT` is set, a live account id (paper ids start with `D`). If `config.yaml` and `.env` disagree the start is paper. Settings are read at start, so restart the supervisor after the change. Live mode then shows a 10-second abort window at start (`main.py`) that is only useful when the bot is started by hand
- [ ] `learning.paper_trading_mode: false` (the `config.yaml` comment says it MUST be false at go-live; it bypasses the adaptor, Thompson re-ranking and the meta-label gate)
- [ ] In `.env` only (never in `config.yaml` or these documents): `IBKR_PORT` and `IBKR_ACCOUNT` for the live account (Gateway live 4001, TWS live 7496), `TRADING_MODE`, `AIT_ALLOW_LIVE`; `AIT_MARKET_DATA_TYPE=1` once the live data entitlement is in place (the default is 4, delayed; this variable lives in `.env` on purpose)
- [ ] If IBC starts Gateway for you (the supervisor relaunches it through IBC after an outage): IBC's own `config.ini` decides paper or live (`TradingMode`, `OverrideTwsApiPort`); the `trading_mode` the code passes is only logged
- [ ] `account.pdt_account_under_25k: true` if the funded account is under $25,000
- [ ] The entry-gate mode (`ml.entry_gates_enabled`) is the operator's decision after a pre-registered test; do not change it as part of the cut-over by accident


## The bot or the broker connection is down
old `BOT DOWN — gave up after N restart attempts`
     -> Read the tail of `logs/bot_stdout.log` and fix the cause (Gateway not logged in, a config error, a traceback). The supervisor tries again by itself once more than 30 minutes have passed since its last restart attempt. To restart sooner, do a full stop and start (Kill switches, "Full stop"): `python run_orchestrator.py` is refuse
NEW `bot restarting (attempt n/N, exit_code=…)`
     -> Nothing for one or two; by attempt 3 something is wrong: read `logs/bot_stdout.log`. A restart does not fix a code bug: capture the traceback first
NEW `BOT RELAUNCH FAILED: the process exited within Ns`
     -> `logs/bot_stdout.log`; the keeper keeps retrying, so fix the cause (config error, Gateway)
NEW `BOT HUNG: trading-loop heartbeat N min stale`
     -> Check it came back (:8503). If it repeats, `data/HALT`, capture the traceback, then follow the deploy checklist
NEW `REFUSING to start the trading bot: …` (two variants)
     -> Verify the PID, kill the orphan by hand (never auto-killed), and the supervisor starts a fresh bot on the next health cycle; if "unreadable", check permissions on the host
NEW `SINGLETON LOCK UNAVAILABLE`
     -> Check `data/` permissions and free disk space, then start again
NEW `cannot start bot — IB Gateway not available on port N`
     -> Is Gateway/TWS logged in? Start it. This alert repeats at most every 30 minutes
old `IB Gateway unreachable on port N (likely daily maintenance)`
     -> Nothing for 30–60 min (IB restarts nightly); the alert says no restart budget is spent. If the port stays closed the supervisor tries to bring Gateway back itself after a grace period (next rows)
NEW `IB Gateway was down N min — attempted automatic restart (k/K)`
     -> Nothing; the bot resumes on its own once the port answers
NEW `IB Gateway STILL down … MANUAL ACTION NEEDED`, `IB Gateway/TWS has not answered … no automated recovery`, `… automatic restart could not be launched`
     -> Look for a login or 2FA prompt on the Gateway window; start Gateway/TWS by hand
old `IB port fallback: .env IBKR_PORT=… but … is what is up`
     -> Fix `.env` when convenient: `python scripts/sync_ibkr_port.py` (dry run), then `--apply` (changes only the `IBKR_PORT` value), then restart the supervisor. `data/ib_endpoint.json` is the supervisor's record; never edit it
NEW `dashboard NOT started: port N is held by a process the supervisor won't kill`
     -> See `dashboard_port_held_foreign` in `logs/orchestrator.log` for the holder; free the port or change `dashboard.*_port`
NEW `supervisor started`, `BOT STARTED \| Mode: …`, `Bot shutting down`, `Telegram channel RECOVERED`
     -> Nothing; after "RECOVERED" check `notification_dropped_after_retries` in the logs

## Can it trade?
NEW `READ-ONLY: IB Gateway is rejecting all orders (Error 321)`
     -> Restart Gateway and log in as IB API, as the alert says. Also keep the Read-Only API setting unchecked ([GETTING_STARTED.md](GETTING_STARTED.md) section 3). The bot keeps re-probing and sends `Trading ENABLED again` when orders are accepted
NEW `CANNOT VERIFY TRADING: the read-only probe did not complete`
     -> Check the Gateway window
old `CIRCUIT BREAKER: Trading halted - {reason}`
     -> Read the reason in the alert. It is sent once per trip and then hourly while the breaker stays tripped; `trading_halted` in `logs/` repeats it every scan cycle. The breaker persists across restarts and a daily-loss trip clears at the next ET day
old `DAILY MTM LOSS HALT: day P&L … breached the daily-loss cap`
     -> Review the book on :8503. Clears at the next ET day. If you decide to close a bleeder by hand, follow the manual-intervention procedure so your close and a working bot exit order cannot both fill
NEW `MTM BRAKE NOT EVALUATING`
     -> Read the error in the message and `mtm_check_failed`; consider `data/HALT` until fixed
NEW `MTM BASELINE DEGRADED`
     -> Informational; the next clean pre-stamp fixes it
NEW `LOOP IMPAIRED — {kind}: N consecutive failures`
     -> Check the tail of `logs/bot_stdout.log` for the exception. If it repeats: `data/HALT` and restart per the deploy checklist (a restart does not fix a code bug: capture the traceback first)
NEW `WATCHDOG ALERT: {component} is DOWN and recovery failed. Errors: N, Last: …`
     -> Read `Last:`. `ibkr`: check the Gateway login and connection. `market_data`: check the market-data feed and subscriptions. `trading_loop`: see `ERROR:` and `LOOP IMPAIRED` and `logs/bot_stdout.log`. At most one alert per component per hour
NEW `ACCOUNT DATA STALE N min`
     -> Check the Gateway connection and that the account is the one you expect
NEW `ERROR: {…}`
     -> Read the log around it; repeated errors: `data/HALT` and investigate
NEW `CAP-GRID INCOHERENT at NLV $…`
     -> Fix the named knobs (`risk.max_position_risk_pct`, `positions.max_position_pct`, breaker); ranges and links: [GUIDE.md](GUIDE.md) section 7.4
NEW `WARNING: live-mode capital base is the HARDCODED DEFAULT`
     -> Set `AIT_CAPITAL_BASE` before trading live

## Data and gate failures (all fail closed: entries refused, exits unaffected)
NEW `VIX DATA OUTAGE`
     -> Check the market-data feed and Yahoo/IB VIX; clears itself
NEW `IV DATA OUTAGE — {symbol}`
     -> Check IB option-IV data for the symbol; clears itself
NEW `EARNINGS DATA UNAVAILABLE for {symbol}` / `DIVIDEND DATA UNAVAILABLE for {symbol}`
     -> Check the data source; clears itself when a fetch succeeds
NEW `RANGE GATE UNAVAILABLE`
     -> Check `models/range.pkl` exists and loaded; re-run the retrain (below). Irrelevant in observe mode
NEW `ENTRY GATE BROKEN — the pre-event blackout check has failed N times`
     -> Read the error in the message; check the calendar tables in `src/ait/data/economic_calendar.py`
NEW `ECONOMIC CALENDAR NEARLY EXHAUSTED`
     -> Extend the tables in `src/ait/data/economic_calendar.py`
NEW `COOLDOWN CHECK FAILED ({symbol})`
     -> Read the exception in the log; clears when it can run

## Positions and exits
NEW `MARKS MISSING xN ticks`
     -> One common cause: the live account is logged in elsewhere (mobile or web takes the single market-data slot, Error 10197): log it out. Otherwise check the Gateway data lines
NEW `STALE EXIT FEED: {symbol}`
     -> Check the market-data feed
NEW `TOUCH STOP NOT EVALUATING`
     -> Check the trade's legs data in :8503 and the log; consider closing by hand via the manual-intervention procedure
old `PDT BLOCKED EXIT`
     -> Decide: accept the overnight risk, or close manually (accepting the PDT strike)
NEW `EXIT REFUSED (position not at broker)`
     -> Usually nothing: the alert ends "Booked closed from broker records." Compare the booked P&L with your TWS fill. If it ends "booking it failed — the row needs manual review", the trade is still open in the books: see [Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break)
NEW `EXIT REFUSED (partial structure)`
     -> **Close the remainder at IBKR yourself**
NEW `EXIT DEFERRED (no quotes)` / `(no quotes, no mark)` / `EXIT WITHOUT QUOTES`
     -> Check the market-data feed; if it repeats, consider closing by hand
NEW `EXIT REPEATEDLY REJECTED: {trade}`
     -> Check it manually at the broker
NEW `SEC 8-K filed for {symbol} — flattening positions`
     -> Watch the exit complete on :8503
NEW `CRITICAL: fill landed on a TERMINAL trade row`
     -> Check TWS and the untracked-option freeze immediately
old `order_rate_limit_tripped` (log, CRITICAL)
     -> Create `data/HALT`, inspect logs, do **not** just restart

## Books versus broker
old `RECONCILE NEEDS ATTENTION: …`
     -> Assignment: in TWS liquidate the assigned stock and verify the remaining legs; the freeze clears itself at the next reconcile once the stock is gone or declared. Partial legs: the bot never closes half a structure, so close the remainder at the broker, then see [Resolving a books-versus-broker break](#resolving-a-books-versus-br
old `UNTRACKED OPTION POSITION … ENTRIES FROZEN`
     -> Reply **yes** (or tap "Yes, mine") if you placed it: the bot will not manage it but still counts it in its risk totals, and the freeze lifts at once. Otherwise close it at the broker; the file clears itself at the next reconcile. "No, investigate" keeps the freeze. See [Resolving a books-versus-broker break](#resolving-a-books-v
NEW `Position discrepancies found`
     -> Compare :8503 with the TWS portfolio; the usual cause is a manual action at the broker. See [Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break)
NEW `{PHASE}: broker NOT connected — positions were NOT reconciled`
     -> Fix the connection; the bot reconciles as soon as it returns
old `EOD RECON: N BREAK(S) …`
     -> Read the lines under it, compare :8503 with TWS, and follow [Resolving a books-versus-broker break](#resolving-a-books-versus-broker-break)
old `EOD RECON: CLEAN \| … gap stress: $X`
     -> Glance at the −10% gap-stress number; if it nears your pain threshold, reduce the short-premium count

## Models, backups and reporting
old `model retrain FAILED (exit N)` / `model retrain HUNG`
     -> Re-run `python run_orchestrator.py --retrain`. The retrain takes its daily bars from IB only (own session) and aborts with exit 3 **before any write** when TWS/Gateway is down, the client id is refused, or IB returns no daily bars (`retrain_*.log` says which): bring TWS/Gateway up first
old `STATE DB BACKUP FAILED`
     -> Check disk space; make a manual copy of `data/ait_state.db`
old `Short-DTE friction collection {status}`
     -> Only if you enabled it: see [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) section 10.1
NEW `GO-LIVE SCORECARD: GATE READOUT FAILED`
     -> Run `python status.py` for the scorecard; read the error
NEW `DIGEST: …` and `GO-LIVE SCORECARD (since …)`
     -> Read them; investigate a missing one
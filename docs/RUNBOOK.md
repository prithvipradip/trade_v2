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
| `DAILY MTM LOSS HALT` | Day's mark-to-market loss breached the cap | Entries auto-frozen; exits still run. Review the book on :8503. Halt clears at next ET day. Manual close of bleeders is YOUR call. |
| `RECONCILE NEEDS ATTENTION: ASSIGNMENT` | Short option assigned — stock appeared | In TWS/Gateway: liquidate the assigned stock, verify the remaining legs, then delete `data/HALT_UNTRACKED` if present. |
| `UNTRACKED OPTION POSITION ... ENTRIES FROZEN` | Order filled during a crash window; bot has no record | In TWS: identify the position; either close it manually or leave it and note it. Then delete `data/HALT_UNTRACKED` to resume entries. |
| `PDT BLOCKED EXIT` | A stop wanted to fire but PDT rules block same-day close | Decide: accept overnight risk, or close manually (accepting the PDT strike). |
| `MARKS MISSING x10 ticks` | No option marks → stop/TP protection NOT running | Check: is the live account logged in elsewhere (mobile/web steals the ONE data slot)? Log it out. Check Gateway data lines. |
| `EOD RECON: N BREAK(S) ...` | Books vs broker mismatch | Compare :8503 positions vs TWS portfolio. Common cause: manual action at broker. Fix the DB via close/adopt or ask the assistant. |
| `EOD RECON: CLEAN | ... gap stress: $X` | Daily all-clear | Glance at the −10% gap-stress number. If it approaches your pain threshold, reduce short-premium count. |
| `model retrain FAILED/HUNG` | Models stale | Not urgent same-day. Re-run: `python run_orchestrator.py --retrain`. |
| `order_rate_limit_tripped` (log CRITICAL) | >8 orders/min — malfunction | Create `data/HALT`, inspect logs, contact assistant. Do NOT just restart. |
| `STATE DB BACKUP FAILED` | Track record unprotected | Check disk space; run a manual copy of `data/ait_state.db`. |

## Daily 2-minute routine
1. Glance at Telegram: any non-CLEAN EOD recon? any CRITICAL?
2. Open http://localhost:8503 — positions green/red, unrealized total sane?
3. Friday extra: confirm `Documents/ait_backups/ait_state.latest.db` timestamp is fresh
   (and that folder syncs to cloud).

## Go-live day checklist (from PLAN.md gates — do not skip)
- [ ] Paper verdict passed (≥50 closes, PF > 1.3, criteria fixed beforehand)
- [ ] Remove `AIT_ALLOW_UNDEFINED_RISK` (strangles refuse at the executor)
- [ ] Revert paper-relaxed liquidity gates: `AIT_LIQ_MAX_SPREAD` ≤ 0.15,
      `AIT_LIQ_MIN_VOLUME` ≥ 10, config `max_bid_ask_spread_pct` ≤ 0.15
      (startup asserts this in live mode and will refuse to run otherwise)
- [ ] `AIT_SIMULATED_CAPITAL` removed; account funded $3k CAD
- [ ] Fresh DB backup taken; PLAN.md gates re-read

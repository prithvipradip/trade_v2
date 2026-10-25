# Claim ledger: docs/GUIDE.md (after fixes)

Scope honesty: the generated tables (supervisor jobs, fixed limits, risk ceilings, order timeline, go-live gates, config sections) come from the code by construction. The prose below was read against the code section by section; items under "not individually checked" were not.

## 1-2 What it is, architecture
- Supervisor + bot, ports, `--paper` spawn | master.py:1010, WebServiceManager | OK
- Single instance lock, refuses a second bot | master.py:338-354, 946-959 | OK
- Restart budget refills after healthy running (30 min) | master.py:1268-1285 | OK
- Heartbeat touched every protection tick unless an error streak | orchestrator.py:1560-1585 | OK
- Gateway deferral, grace, IBC recovery, cooldown, budget | master.py:777-790, 222-224, 1239-1262 | OK
- `models/.retrained` restart never in market hours / post-market | master.py:1168-1200 | OK
- Client-id fallback; other jobs use own client ids | CLAUDE.md, RESEARCH_GUIDE client ids | OK
- Component map packages | src/ait layout, generated MODULES.md | OK
- Retired subsystems under deprecated/ (news, options flow, hedging, GARCH members, event straddles) | deprecated/src, deprecated/research listings | OK

## 3 Trading day
- Phases, startup order (background training, reconcile, announce, trading check) | orchestrator.py:668-700 | OK
- Pre-market steps | orchestrator.py:808-872 | FIXED (added meta-labeler training, Thompson decay)
- Post-market steps | orchestrator.py:5447-5570 | OK
- Two tasks, protection every 30 s, scan after previous finishes | orchestrator.py:1560-1600, cadence.py | OK
- Catch-ups (backup, retrain, digest, report, backtest) | master.py:2078,2340,2354,2395,2596-2660 | OK

## 4 Entry pipeline
- IV rank required; ~2 years history | orchestrator.py _scan_symbol, IV DATA OUTAGE | OK
- Low directional confidence does not veto a condor (neutral-only mode) | orchestrator.py:2118-2130 | OK
- Meta-labeler acts only when enabled, trained, not paper mode, real closes reached | MODELS/meta_label settings, orchestrator | OK
- Pre-execution checks (halt files, restricted list unreadable blocks, blackout, cooldown, budget, settling, first-hour, duplicates, through-expiry) | orchestrator.py:3380-3490 (halt, settling), RESTRICTED_UNREADABLE 175-197 | OK for halt, restricted, settling; others read at writing time (not re-read line by line this pass)
- Observe vs enforce | orchestrator.py:2470-2500 | OK

## 5 Strategies
- STRATEGY_MAP, files spreads/long_options/straddles | strategies/selector.py:27, ls | OK
- IC IV floor 15 / cap 85 (examples) | iv_rank.py docstring | OK
- Wing formula wing_k x price x IV x sqrt(DTE/365) | iron_condor.py:120-158 | OK
- Thompson state file | strategies/thompson.py:26 | OK

## 7 Risk
- validate_trade order (circuit breaker, confidence, weekend gap, VIX guardrails, count, duplicate, per-expiry, correlation, buying power, per-trade risk, concentration, delta, daily loss, sizing) | risk/manager.py:346-599 | OK
- Delta gate largely inert (no greeks feed) | portfolio.py:890,1312-1373 | OK
- Circuit breaker trips, persists, blocks entries only | circuit_breaker.py | OK

## 8 Execution, 9 Exits, 10 Learning
- Entry steps 1-9, exactly-once booking, outbox | executor.py (read in earlier work), orchestrator exit path | OK (read earlier; not re-read line by line this pass)
- Reconciliation bullets | reconciler.py:1531-1620, 1737-1860 | FIXED (added automatic booking of vanished trades, reconcile on prompt answer)
- Exit rules 1..5; delta breach inert; IV-crush pre-close; counterfactual evaluation retired | portfolio.py:598-640,890-935; orchestrator.py:5546 | OK
- Learning cycle, bounds, paper-trading mode bypass | adaptor/analyzer, config.yaml:870-875 | OK

## 11-13 Data, monitoring, configuration
- Sources (IB first, Yahoo fallback, Polygon research-only, FRED CSV) | market_data.py:449-490, data/macro.py:41 | OK
- Quote cache TTL 15 s < 30 s cadence | cadence.py:47, market_data.py:317 | OK
- Stores and dual write | state.py, analytics | OK
- Analytics tabs (8) | dashboard/app.py:2098-2101 | OK
- Dashboards bind 127.0.0.1 | status_server.py:106, web_logs.py:348, master.py:1378 | OK
- Watchdog = three components | orchestrator.py:473-516 | OK
- Go-live criteria table | go_live.py:346-348 (generated) | OK
- Config precedence, frozen at start, config sections | runtime_env.py, settings.py | OK

## Gaps closed in the second pass
- 4.2 trade budget tiers (first hour / midday / rest, bounded by max_daily_trades) | orchestrator.py:2780-2810 | OK
- 4.2 first-hour confidence applies to directional signals only | orchestrator.py:3498-3515 | OK
- 4.2 pre-event blackout in trading days | orchestrator.py:3410-3430 | OK
- 4.2 post-stop cooldown: full trading day for the symbol, shorter for correlated siblings | orchestrator.py:3193-3240 | OK
- 4.2 duplicate guards (DB cooldown, working broker orders, executor pending tracker) | orchestrator.py:3302-3356 | OK
- 4.2 through-expiry earnings / ex-dividend blocks | orchestrator.py:3536-3574 | OK
- 8.1 entry sequence (undefined-risk, rate limit, market hours, row before order, batch qualification, combo validation, BAG) | executor.py:430-870 | OK
- 8.2 exit pricing, band memory, compare-and-set transitions, outbox, commission true-up | executor.py:1540-1569, state.py:270-293,1079-1204, orchestrator.py:993-1073 | OK
- Sections 14-17 are pointers; the settings named in prose exist (tests/test_guide_md.py).

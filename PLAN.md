# AIT v2 — Improvement Plan (from full repo audit, 2026-07-07)

Mission: autonomous options-income bot (sell premium via iron condors / strangles on
liquid US ETFs/megacaps), ML-gated entries, automated exits, self-learning, hands-off ops.
Paper account DUN603821. Real track record as of audit: **$248.80 banked (5 real closes),
8 open filled positions** — machine works, edge unproven.

Strategic frame: with the learning layer dormant below ~30 real trades, the fastest path
to "is the edge real?" is **maximizing clean trade count while protecting track-record
integrity**. Priorities below rank by that, not code aesthetics.

Full audit reports: four-agent audit (pipeline core, risk/broker, ML/learning, ops/infra),
2026-07-07. Key file:line references inline.

---

## Batch 1 — safety/correctness bugs live right now

| # | Item | Where | Status |
|---|------|-------|--------|
| 1.1 | **Dead thesis-invalidation exit**: `_check_thesis_valid` raises NameError (`symbol` undefined) every call, swallowed by broad except → ML direction-flip exit never runs. Fix var + narrow except. | orchestrator.py:1797,1822 | done |
| 1.2 | **IC gated backwards**: iron condors require ≥0.65 *directional* confidence pre-check — blocks calm markets (ideal for ICs), admits trending ones. Backtest skips this gate for neutral strategies (engine.py:300); live must too. Gate ICs on range-model probability. | orchestrator.py:~730 | done |
| 1.3 | **bot_stdout.log unbounded (2.0 GB)**: raw append sink never rotated; mtime-based cleanup can never delete it. Size-capped rotation + size-based cleanup. | master.py:146,514 | done |
| 1.4 | **Per-position marks never persisted**: `open_positions.unrealized_pnl` stays 0 forever. Add `update_position_mark()` in state.py, call in `_evaluate_position` (marks-present guard), surface per-position P&L in status.py + 8503 dashboard (green/red table). | portfolio.py:240, state.py, status.py:122, status_server.py | done |
| 1.5 | **Reconnect wedge**: after 5 failed reconnects, `_reconnect` returns False forever (counter never resets). Reset counter on exhaustion + alert. | ibkr_client.py:166-189 | done |
| 1.6 | **FX silent overstatement**: failed USDCAD fetch → usd_to_base=1.0 → all CAD-base limits inflated ~39%. Cache last good rate; if none, mark account data unreliable (risk layer treats as no-data). | account.py / ibkr_client.py:401 area | done |
| 1.7 | **keeper_ait.bat untracked** — the crash-recovery script exists only on this machine. git add. | repo root | done |

## Batch 2 — track-record integrity + edge validation

| # | Item | Where | Status |
|---|------|-------|--------|
| 2.1 | **Strangle risk mis-modeled**: `max_loss = 3× credit` understates tail 10-30×; buying-power check inverted for credits (checks affordability of money received); IBKR margin never consulted. Use strike-distance-based tail estimate; check margin/available funds for credit trades. | straddles.py:155, manager.py:204 | done |
| 2.2 | **Partial exits book phantom P&L**: records estimated P&L + underlying price as fill before order fills → DB/IBKR drift. Register order, book on real fill. | orchestrator.py:1407-1447 | done |
| 2.3 | **Size multiplier bypasses risk gates**: learning multiplier applied AFTER validate_trade → gates check 1×, book can execute N×. Apply before validation. | orchestrator.py:1235 | done |
| 2.4 | **Range-model CV leakage**: 30-day overlapping labels with gap=5 → inflated AUROC feeding the trade/no-trade edge gate. gap ≥ horizon. | range_predictor.py:479-491 | done |
| 2.5 | **Thompson decay truncation**: `int(wins*0.995)` zeroes 1-win arms daily → bandit can never accumulate signal at low volume. Decay in float, round on read. | thompson.py:132-133 | done |

## Batch 3 — before trusting results / going live

| # | Item | Where | Status |
|---|------|-------|--------|
| 3.1 | **Retrain pipeline**: 7:30 daily retrain writes models the live bot doesn't load (7-day reload timer) — unify; pin OMP threads in subprocess; atomic pickle writes (tmp+rename). | master.py:396-436, trainer.py:67, ensemble.py:419 | done |
| 3.2 | **Tests for burned paths**: marketable-combo entry pricing + spread-reject gate (executor), BotManager restart budget / gateway-defer. Zero coverage today despite production failures. | tests/ | done |
| 3.3 | **Config consolidation**: hoist behavioral literals (EXIT_CROSS, entry offset, RANGE_MIN_CONFIDENCE, budget tiers, risk cap %s, correlation cap) into settings; delete dead gates (daily-limit on wrong object, concentration reading never-populated field); wire or remove phantom `max_portfolio_risk_pct`. | manager.py, orchestrator.py, executor.py, settings.py | done |
| 3.4 | **Backtest fill realism** (larger project, separate pass): historical option quotes where available, model combo non-fills. Until then treat +311% IC backtest as directional only. | engine.py | deferred |

## Deferred / watchlist
- ~~Live data~~ done 2026-07-02 (Network B/C + OPRA, ~$3/mo).
- IBC read-only automation: vmoptions `--add-opens` fixed the crash; box may still need manual
  check→uncheck→Apply toggle after some Gateway restarts. Guardrail alerts via Telegram.
- Options-chain OI=0 vs `min_open_interest` filter interplay; correlation guard gets no price
  data (sector-table fallback only); portfolio delta gate ~0 (no greeks subscription);
  ET-vs-local time in Friday gate + budget tiers; wmic deprecation in keeper.
- Learning layer wakes at ~30 real closes; expect ~1 month at current fill rate.

## Operating rules (learned the hard way)
- Restarting the BOT ≠ restarting the GATEWAY. Entitlements/read-only load at Gateway login.
- ONE live-data slot: live account must stay logged out of mobile/web while bot runs.
- Paper ≠ live fills: paper is optimistic; marketable pricing narrows the gap.
- Secrets: IBC password lives ONLY in C:\IBC\config.ini (outside repo). Never in git/chat/docs.

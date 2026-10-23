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
| `model retrain FAILED/HUNG` | Models stale | Not urgent same-day. Re-run: `python run_orchestrator.py --retrain`. W89: the retrain takes its daily bars from IB only (own session, `ml.retrain_ib_client_id`, never Yahoo) and aborts with exit 3 **before any model or `daily_prices` write** when TWS/Gateway is down, the client id is refused, or IB returns no daily bars for `^VIX`/SPY/the universe (`retrain_*.log` says which). Bring TWS/Gateway up first, then re-run. |
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

**W89 deploy note (daily retrain on IB).** `master.py`'s retrain script is part of the supervisor, so a
change to it needs a **full supervisor restart** (not just a bot restart). From then on the 07:30 ET retrain
needs TWS/Gateway up (see the retrain alert row above); with it down the retrain aborts and the models stay
as they were. The first successful IB retrain re-tags the last ~504 `daily_prices` rows per symbol back to
`IB_TRADES_RTH` (they had been overwritten with `YAHOO_UNADJ`); check with
`sqlite3 "file:data/historical.db?mode=ro" "SELECT source, COUNT(*) FROM daily_prices WHERE symbol='QQQ' GROUP BY source"`.
Mid-session, today's partial bar is never stored.

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

## Research fixed limits (W88a; NOT configurable — visible on purpose)

The research counterpart of the table above, in ONE module: `src/ait/config/invariants_research.py`. Each value is a correctness constraint of the research engine's own arithmetic, not a modelling choice, so it is not
in `config_research.yaml` or a study file (anything in those files is editable regardless of a comment). They are logged once per research process (`research_fixed_limits_in_effect`) and stored in every run record. To change
one: edit the module, update this table (a test compares them), and record it in PLAN.md; a change moves research numbers.

| Name | Value | Why it is fixed (and where enforced) |
|---|---|---|
| `RESEARCH_DELTA_MIN` | 0.05 | Lower end of the \|delta\| band a strike search may be asked for; below it a modelled quote is noise. Also the lower bound for `delta_short` / `delta_long` search-space overrides. `engine.py`, `param_spaces.HARD_BOUNDS`. |
| `RESEARCH_DELTA_MAX` | 0.45 | Upper end of the same band; above it a leg is no longer out of the money. `engine.py`, `param_spaces.HARD_BOUNDS`. |
| `IV_SCALE_MIN` | 0.5 | Lower clamp of the `iv_floor / iv` scaling a builder applies to its delta; one stale IV print cannot move a strike by more than a factor of two. `engine.py`. |
| `IV_SCALE_MAX` | 1.5 | Upper clamp of the same scaling. `engine.py`. |
| `SETTLEMENT_LOSS_TOLERANCE` | 0.000001 | Dollars by which a settled loss may exceed the defined max loss before it is counted as `over_max_loss`; a floating-point tolerance. `engine.py`. |
| `PROB_CLIP_EPS` | 0.0000001 | Probabilities are clipped to [eps, 1 - eps] before log-loss; numerical safety. `walkforward.py`. |

## Research tunables (W88b; research only)

`config_research.yaml` has three jobs (its header says so): the **diff** against `config.yaml` (shared keys appear only where research deliberately differs), the **overlay** applied by research entry points only, and, since W88b, the
**full list of research-only tunables**. A research-only key has no `config.yaml` counterpart, so it is listed in the overlay IN FULL, with its default and a comment; the schema default in `src/ait/config/settings.py` is only the
fallback for a bare construction (`Backtester()`, `BacktestLearner()` in a unit test). A test fails if a field of these blocks is missing from the file, and the file's values equal the schema defaults today on purpose
(the same convention as the `parity:` block). A study (`studies/<name>.yaml`) may override any of them for one run.

| Block (`backtest.`) | What it holds | Code that reads it |
|---|---|---|
| `learner` | Thresholds of the research `BacktestLearner`: trades needed to learn / disable, win-rate cutoffs, step sizes, multiplier and confidence bounds, protected strategies. Not the live learning layer. | `backtesting/learner.py` |
| `legacy_engine` | Constants of the dormant paths: fallback direction model, naked-margin fraction, bearish margin, thesis-invalidation confidence, vol-regime ratio. None is on the iron-condor path under `engine_model: realistic`. | `backtesting/engine.py` |
| `objectives` | Composite objective weights and the profit-factor cap. | `optimization/objectives.py` (the optimizer passes its own copy) |
| `walkforward_tuning` | Warm-start thresholds, the adaptive range threshold (multiplier, clips, lookback, fallback band), the validation-split ratio, the inert-parameter probe fallbacks. | `backtesting/walkforward.py`, `optimization/optimizer.py` |
| `walkforward_tuning` also holds (W90b / W90e) | The window lengths and Optuna trial counts, the training / test row floors, the context and fetch paddings, the out-of-sample diagnostic bins and minimums, the minimum number of windows, the window-score shrink. | `backtesting/walkforward.py`, `run_backtest.py`, `run_optimizer.py` |
| `engine_tuning` | The IV-skew coefficients, the realized-vol and IV-rank-rise look-backs, the minimum history, the fallback volatility (W90c / W90e). | `backtesting/engine.py`, `pricing.py` |
| `surface_tuning` | The acceptance rules of the research IV surface (volume, price, moneyness, points, span, IV bounds, clamp margin, stale days). The fit rules are in the surface cache key. | `backtesting/iv_surface.py`, `engine.py` (W90f) |
| `friction_fit` | Minimum samples, sessions, training rows and affine points to fit a leg-spread model. Offline calibration only. | `backtesting/friction.py` (W90f) |
| `spread_tuning` | Max cost as a share of capital (0.25) and the wing as a fraction of the expected move (0.5) in the non-condor builders; the iron-condor builder does not read them. | `backtesting/engine.py` (W90f) |

**A new research number goes in a setting, not in code (W88e).** `tests/test_w88_literal_census.py` compares the non-trivial numeric literals of the result-affecting research modules
(`scripts/literal_census.py` lists them: `backtesting/`, `optimization/`, the ML modules, `run_backtest.py`, `run_optimizer.py`) with `tests/data/literal_allowlist.json`. A function whose literals differ from its entry
fails the test: a NEW literal, a CHANGED one, a REMOVED one (converting a literal to a setting must also update the allowlist), or a stale entry. Either make it a schema field listed in `config_research.yaml` (or a fixed limit in
`config/invariants_research.py`), or run `python scripts/literal_census.py --update` and give the new entry a category: `display`, `epsilon`, `fixed_feature_group`, `dormant_path`, `structural`, or `registered_followup` (a note
saying what it is and where it should go). A new module in `backtesting/`, `optimization/` or `ml/` must be added to the census list (or excluded with a reason). `param_spaces.py` (the search-range registry) and `settings.py` (schema defaults) are the excluded registries.

`python scripts/show_research_config.py [--section backtest] [--differs]` prints every value with the layer that decides it (`study` > `overlay` > `config.yaml` > `default`); the same map is stored in every run record
(`settings_origin`). Values that are **not** tunable (clamps, numerical tolerances) are in "Research fixed limits" above.

Known, preserved behaviour worth knowing before tuning the learner: with the shipped `reduce_min_change` (0.03) and `strategy_multiplier_min` (0.30), a reduced strategy's multiplier stalls at 0.3277, so the "disable a
strategy" branch is unreachable by default; `boost_min_trades` and `reenable_min_trades` are inert at any value at or below `min_trades_to_learn` (15). The module's old `MIN_CONFIDENCE` constant (0.55) was read by nothing and was dropped. Each is documented on the field.

## Model hyperparameters (W88c; shared by live and research)

The tree hyperparameters of the gradient-boosted models are settings, not code literals: `ml.hyperparams` (direction ensemble, range model, vol-magnitude model) and `ml.meta_label_hyperparams` (the meta-labeler's own, shallower
profile), both `TreeHyperparamsConfig` (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `random_state`; `null` = unseeded). `config.yaml` lists them with the values the code used before, so listing them
changed nothing. **Shared keys are diff-only in `config_research.yaml`**: research differs only if you put a different value there (or in a study); the parity manifest then warns (`ml_hyperparams`).
Not settings, on purpose: `n_jobs` / `verbosity`, each task's objective and metric, LightGBM's `class_weight` / `deterministic`, and `ml/trainer.py::_objective` (a search-time proxy).

Every saved model (`ensemble.pkl`, `range.pkl`, `vol_magnitude.pkl`, the meta-label artifact) records the hyperparameters it was trained with (`spec`). Loading **never refuses**: a model trained under other values than the current
config is served and logs `model_spec_mismatch` with the differences; an artifact saved before W88c has no `spec` and loads silently. A change to the LIVE values takes effect for live at the next retrain (and bot restart), not by
editing `config.yaml` alone. The run record stores the resolved settings and the ML seed.

## Model labels, validation folds and retrain constants (W90d; shared by live and research)

The numbers that define what the live models learn and how they are validated are `ml.*` settings, listed in `config.yaml` with the values the code used before (so listing them changed nothing): the range band
`range_threshold_pct` (0.05) and its edge floor `range_min_edge_over_baseline`; the vol-magnitude move, horizon and edge floor (`vol_magnitude_threshold_pct` 0.07, `vol_magnitude_horizon_days` 30,
`vol_magnitude_min_edge_over_baseline` 0.10); the direction label (`direction_label_threshold_pct` 0.01 over `direction_label_horizon_days` 5); the walk-forward folds (`direction_cv_splits` / `direction_cv_gap_days` /
`cv_min_val_rows` / `cv_min_train_rows` for the direction model, `binary_cv_splits` / `binary_cv_min_gap_days` for range and vol-magnitude, whose gap is never shorter than the label horizon); the class-balance guards
(`label_balance_min` / `label_balance_max` warn, `min_class_samples` refuses); and the retrain (`retrain_optuna_trials`, `retrain_optuna_seed`, `rollback_margin`).
A change to a LIVE value takes effect at the next bot restart, and a changed label constant needs a retrain: a saved range or vol-magnitude model records the band and horizon it was trained at, and a mismatch is flagged
(`range_model_spec_mismatch`, CRITICAL) and the model retrained. An artifact written before the spec was recorded is compared as +/-5% / 30 days (range) or +/-7% / 30 days (vol-magnitude), whatever the settings say now. Research does not
read the band keys (it adapts its range band per window: `backtest.walkforward_tuning`); the other keys it shares with live.

## Feature windows (W88d; shared by live and research)

The indicator windows of the core feature groups are settings: `ml.features` (`momentum`, `volatility`, `volume`, `trend`, `multi_timeframe`, `vol_regime`), listed in `config.yaml` with the values the code used before, so listing them
changed nothing. **Shared keys are diff-only in `config_research.yaml`**: research differs only if you put a different window there (or in a study); the parity manifest then warns (`ml_features`).
In scope: RSI / MACD / ROC, ATR / Bollinger / realized vol, volume averages, SMAs and slopes, the simulated-weekly features, and the realized-vol regime ("IV proxy") features. Fixed (not settings): the fractal / wavelet / Hurst group, the 26 VLMC
intraday features, cross-asset, macro, seasonality and the market-structure group (including `above_sma200` / `distance_sma200`).

**What changes when you change a window.** (1) The feature VALUES. (2) For the thirteen model columns whose name embeds a window the NAME follows it: `rsi_long: 21` turns `rsi_14` into `rsi_21`; `sma_mid: 30` turns `price_vs_sma_20`,
`sma_20_slope` and `sma_10_20_cross` into `price_vs_sma_30`, `sma_30_slope`, `sma_10_30_cross`. The meta-labeler's inputs, the engine's and the live orchestrator's meta-context keys, and the entry-signals snapshot follow; the dashboard series keep
their stable keys (`rsi_14`, `sma_20`, ...). (3) A hash of the windows (`feature_spec_hash`) is recorded in every saved model and in every run record.

**A model trained under other windows is never served.** Its columns would not exist in the new frames, and `predict` zero-fills missing columns, so it would predict on silent zeros. Loading a model whose recorded hash differs logs
`model_feature_spec_mismatch` (CRITICAL) and returns False, so the normal needs-training path retrains under the new windows. A model with no recorded hash predates W88d and is treated as having the default windows, so today's artifacts keep loading
while the windows are at their defaults. The research engine also drops a predictor whose windows differ from its own feature frames. After a LIVE change: restart, expect a retrain of the direction, range, vol-magnitude and meta-label models;
old `entry_signals` snapshots keep the old names, so the meta-labeler's live training reads defaults for them until new trades accumulate.

**Constraints enforced at load** (a violation is a startup error): ordered pairs (`rsi_short < rsi_long`, `macd_fast < macd_slow`, `roc_short < roc_mid < roc_long`, `rv_short < rv_long`, `sma_fast < sma_mid < sma_slow`, `volume_fast_window < volume_avg_window`)
and `vol_regime.fast < rv_short < rv_long < vol_regime.slow`; derived names must be distinct and must not collide with a fixed column. Very short RSI periods make RSI undefined whenever a window has no down days, and those rows are dropped; `iv_rank_min_periods`
only matters once it exceeds the warm-up that other windows already drop.

## Risk-knob ceilings and the links between them (W86)

Unlike the fixed limits above, these are **tunable** (they live in `config.yaml`), but the **range** each may take is part of the schema in `src/ait/config/settings.py`, so a typo or a bad edit fails
at startup instead of trading. The ranges sit in code on purpose (a limit in the same file as the value lets one edit loosen both). To widen one: edit `settings.py`, update this table (a test compares it),
and record the decision in PLAN.md. A change takes effect at the next bot restart, which is the operator's call.

| Setting | Shipped value | Allowed range | What it bounds |
|---|---|---|---|
| `risk.max_position_risk_pct` | 0.05 | 0.005 - 0.25 | One structure's max loss, fraction of NLV (gate 6b), and the budget the condor builder is handed |
| `risk.max_daily_loss_pct` | 0.045 | 0.005 - 0.20 | The daily-loss breaker; blocks new ENTRIES only, exits keep running |
| `positions.max_position_pct` | 0.05 | 0.01 - 0.30 | The position sizer's hard per-position cap |
| `positions.max_portfolio_risk_pct` | 0.20 | 0.005 - 0.50 | Aggregate open capital-at-risk (gate 6b-2) |
| `risk.max_symbol_concentration_pct` | 0.20 | 0.05 - 0.50 | Exposure in one symbol (gate 6c) |
| `backtest.position_size_pct` | null (= `risk.max_position_risk_pct`) | 0.01 - 0.25 | Research per-trade size, never looser than live's per-trade range |

**Links enforced at startup** (a violation is a `ValueError`, because the runtime symptom is a silent total brick):
`risk.max_position_risk_pct` must be <= `positions.max_position_pct` (the sizer would refuse what gate 6b passes), <= `positions.max_portfolio_risk_pct` (gate 6b-2 would refuse every entry) and
<= `risk.max_symbol_concentration_pct` (gate 6c would refuse every entry). **Not** a startup error: the breaker below one full max loss. The launch coherence check (orchestrator A2) reports it by Telegram
only, because it is a measurement of the real condor builder, and the breaker never closes a position (it blocks entries after the loss). Raising the per-trade budget also needs the aggregate and
concentration caps raised with it, or the startup refuses.

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

- **When: about 15:30 ET, BEFORE the close.** Measured 2026-10-13: after the session ends IB no
  longer serves that day's expiring contracts, so an after-close run can never capture DTE 0 (its
  expiry list starts at the next day). At 15:30 the entry window's last bar (14:30-15:30) has ended,
  so one run holds the whole expiry day; the DTE-0 contracts are requested first. A run any later
  adds no bar (the 15:30 bar is outside the entry window). Bars whose hour had not ended are never
  stored (`in_progress_bars_skipped` in the summary).
- **Client id:** use one that is not the bot's (`IBKR_CLIENT_ID`, currently 1, and its fallback ladder
  base+100..+104). Convention: 90 = `backfill_historical_data.py`, 91 = this collection (script and
  supervisor job), 94 = the daily retrain (`ml.retrain_ib_client_id`, W89); ad-hoc runs pick another free
  id. Two sessions on one id make the second fail ("clientId already in use?"; seen 2026-10-21). The
  example uses 91, the repo's calibration convention. W66 verified this request type against the paper
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

### Lifecycle capture (W82): one run a day, each contract downloaded about twice in its life

The per-day collector above asks every contract in range for its last 7 days, every day, so a
contract is downloaded 5-11 times and `INSERT OR IGNORE` throws the overlap away. IB serves 30 days
of hourly BID/ASK bars for any LISTED option in one request, and what is perishable is the contract
at its EXPIRY, not each day. `--lifecycle` therefore captures a contract's life:

```bash
python scripts/calibrate_option_spreads.py --source ib-bars --short-dte --lifecycle \
    --symbols QQQ --client-id 91 --dry-run      # the plan: stage A / stage B and the minutes
python scripts/calibrate_option_spreads.py --source ib-bars --short-dte --lifecycle \
    --symbols QQQ --client-id 91
```

- **Stage A (perishable):** expiries today..today+`--expiring-days` (default 1, so one missed day is
  recoverable). DTE 0 is requested first; run at 15:30 ET like the per-day collector.
- **Stage B (coverage debt):** far-listed expiries at 15-21 DTE whose DTE 31-45 tail is not stored,
  nearest first, capped by `--max-far-contracts` (default 120, about 43 minutes); the rest is
  deferred to the next run. "Not stored" is read from the database (a logged stage-B capture, or
  at least 5 stored sessions at DTE >= 31), so a missed day or a manual catch-up needs no state file.
- **Strikes:** the union, over every `--thin-every`-th session of the underlying's path, of what a
  condor would have sold that day (that day's spot, stored IV and DTE), instead of the strikes near
  the CURRENT spot. `--thin-every 1` is every session (2-3x the contracts).
- **Every request asks 30 D**, so a run also recovers the preceding month of sessions for contracts
  still listed; a run any time within 30 days of a volatility spike retrieves the spike's mid/far-DTE
  sessions. Only contracts that EXPIRE during a spike need the daily run.
- **Bands:** 0-2, 3-6, 7-13, 14-29, 30-45 DTE are each fitted and gated on 15 sessions
  (`backtest.friction_min_sessions_per_band`). Only 0-6 DTE is extrapolated (floored at the tick)
  while unfitted; a 7+ band without a fit is answered by the all-DTE model as before.
- **TWS or IB Gateway:** the script and the supervisor job accept either endpoint of the configured
  account type (paper 7497 <-> 4002, live 7496 <-> 4001) and never probe across account types. The
  log prints `[port] 7497 ... is not listening; using 4002 (IB Gateway (paper))` when it falls back.
- **Capture log:** `option_capture_log` in `data/historical.db` (one row per requested expiry and
  run); the status file adds `stage_a`, `stage_b` and `deferred_b`.
- **Supervisor job:** set `data_collection.lifecycle: true` (with `short_dte_enabled: true`); the
  per-day behaviour stays the default.

### Automatic collection (supervisor job, W76)

Instead of running the command by hand every session, the supervisor (`run_orchestrator.py`) can
run it Mon-Fri after the close. It is OFF by default.

1. Prove it manually once: the `--dry-run`, then one real run (above).
2. In `config.yaml` uncomment the `data_collection:` block at the bottom and set
   `short_dte_enabled: true` (time defaults to 15:30 ET, symbols `[QQQ]`, client id 91).
3. **Restart the supervisor** (`run_orchestrator.py`): a process runs on the settings it started with.
   It logs `short_dte_collection_scheduled | time_et=15:30 ...` at startup if it took effect.
4. Keep the supervisor and TWS/Gateway running at that time. The box being off skips that day
   (nothing else is lost); the supervisor gives a job an hour of grace if it was only just started.

What it does at the scheduled time: skips non-trading days; refuses to run if the IB port
(`IBKR_PORT`) is not listening or the client id collides with the bot's (`IBKR_CLIENT_ID` and
+100..+104); runs the same script with an idle timeout (20 min of silence) and a hard cap
(150 min); never touches the bot or any order.

How to see progress, without opening logs:

- `data/short_dte_collection_status.json`: counters of the last run, stored samples per DTE and
  sessions, and per band `sessions` vs `needed`, `active` once the band has a fit and 15 sessions.
- `logs/short_dte_collect_YYYYMMDD.log`: the script's own output for that day.
- `logs/orchestrator.log`: `short_dte_collection_ok | stored=... bands={...}`.

You get a Telegram message (at most one per day) only when something is wrong: the IB port is
closed, the client id is refused, the run failed or timed out, or it stored zero samples. A
healthy day is silent.

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
- **Expiry settlement (W77):** `summary.md` also carries an "Expiry settlement parity" section for
  every structure that reached its expiry bell: how many leg outcomes (expired / exercised /
  assigned) `ait.backtesting.settlement` reproduced from the statement's close codes, whether the
  shares left over the gap match, and the P&L error. The model marks leftover shares at the NEXT
  session's open; a trader who waited shows up as error and the section says how long they waited.
  The assumptions (`backtest.pin_band_usd`, `assignment_fee_per_share`,
  `stock_commission_per_share`) are operator-owned. In the research engine the settlement path is
  only reached with `engine_model: realistic` AND a position the take-profit ladder / touch stop did
  not already close, so it matters most for studies that disable those exits.
- **Never paste** per-trade prices, P&L or any account identifier into PLAN.md: it is shared via
  git with the other box. Report error statistics by DTE band only.

## Per-horizon range models and the decision frame (W78; research only)

- **Range models per horizon.** The walk-forward trains the range gate's model for the horizon the
  study's entry DTE can actually reach (`ait.ml.range_spec.horizon_for`, shared with the live spec).
  The configured band's own horizon is the base model (unchanged: 9 bars at `dte_range [14, ...]`);
  any other DTE is rounded UP to a bucket in `RANGE_HORIZON_BUCKETS` and trained once per window, on
  demand. Non-base buckets save under `models/research/h<N>/`; the live `models/range.pkl` is never
  touched. `python scripts/range_horizon_study.py --symbol QQQ` measures out-of-sample skill, abstention
  and reliability per horizon (read-only; artifacts go to a temp directory).
- **Abstention is not a bug.** When no ensemble member has cross-validated edge over the base rate the
  model returns no probability; the engine's range gate then rejects (enforce) or would-reject
  (observe). The study reports how often that happens per horizon: read it before enforcing the gate.
- **Decision frame.** With `engine_model: realistic` and intraday entries, the entry gates and models
  see history through the PREVIOUS session (a 10:30 decision cannot know that day's close), and an
  entry counts positions that were still alive at its fill time toward `max_concurrent_positions`.
  Legacy and daily-close studies are unchanged.

## The IV-rank rise veto is research-only (W71)

- **What it is.** The research engine blocks a neutral credit entry when the FeatureEngine `iv_rank`
  (a REALIZED-vol range rank, 0-1) rose by more than `backtest.iv_rank_rise_threshold` (0.30) over
  10 rows. **Live has no such veto**: it gates on the LEVEL of the stored-IV percentile (condor floor
  15, credit cap 85; `ait.data.iv_rank`). `tests/test_w71_iv_rank_parity.py` pins that nothing outside
  the research modules reads the threshold.
- **Measure it:** `python scripts/iv_rank_parity_probe.py [--ab --workers 6]` (read-only; seconds for
  the series comparison, minutes for the engine A/B). It reports how the two ranks relate, how many of
  the veto's days live would have refused anyway and how many are days live WOULD trade.
- **It is OFF in research (W71c, decision D9):** `backtest.iv_rank_rise_gate: false` in
  `config_research.yaml`. The code, the 0.30 threshold and its 0.25-0.60 search range are kept; while
  the gate is off the rise is still recorded per entry (`iv_rank_rise_10d`) and the optimizer does not
  search the threshold. The parity manifest prints a RESEARCH-ONLY warning only while it is ON.
- **The bar for turning it back on** (also written next to the key): a MULTI-WINDOW walk-forward with
  the gate ON vs OFF, same windows / engine / pricing / friction (`iv_rank_parity_probe.py --ab` is
  the starting tool), showing an OUT-OF-SAMPLE gain beyond the window-to-window noise on days live's
  level gates would have TRADED. If it earns its place, port it to live as well (W80-class).
- The research IV series and live's stored series are the same IB historical IV except on the few days
  live's own scan wrote (a median across expiries), so the level gates are at parity.

## One-sided entry gates: what research blocks that live does not, and the reverse (W81)

- **Why this section exists.** Research and live can disagree about whether a condor may open
  without any parameter differing: a gate that exists on ONE side only. The 2026-10-13 audit found
  three: the research realized-vol cap (`backtest.max_entry_vol_annual`, 0.80; Optuna searches it),
  the research trending-down regime veto (`backtest.regime_veto_gate`), and live's VIX halt
  (`risk.credit_vix_halt`, 28, plus the VIX-tiered credit caps), which research does not simulate.
- **Where it is visible.** `python run_backtest.py` prints one warning per ACTIVE one-sided gate
  (`vol_gate`, `regime_veto`, `vix_halt`, plus the existing rise-veto line) and records
  `live_only_gates_not_simulated`. The registry is `ait.backtesting.gate_parity`: every reason the
  engine can reject on is classified there as a live twin, a research-only gate (with its switch) or
  mechanics, and `tests/test_w81_gate_parity.py` fails if the engine grows a reason without a row.
- **The switch.** `backtest.regime_veto_gate` (schema default true, so nothing moves; the overlay does
  not set it: decision D10). False skips the veto only; the day class is still recorded per entry
  (`regime_class`; a day the veto would have blocked is flagged `regime_veto_suppressed`). Its
  thresholds are `backtest.regime_down_threshold` (-0.05) and `regime_up_threshold` (0.02), which
  were bare literals at three engine sites.
- **Measure it:** `python scripts/gate_parity_ab.py --workers 3 [--entry-dte N]` (read-only; about 8
  minutes at three symbols and four arms; the VIX closes come from Yahoo). Arms: baseline, regime
  veto off, vol cap off, both off; then live's VIX halt is applied after the fact (no slot re-use).
  **Re-run it whenever the gate order, the IV-rank cap or the entry-DTE range changes: these gates
  only matter on the days that reach them.** First result (2026-10-13): at entry DTE 14 the veto
  reached 0 / 0 / 3 days (QQQ / SPY / IWM), the vol cap 0, and no trade entered on a day whose
  previous VIX close was >= 28 (18 such sessions); the same at DTE 1, 3 and 7 on QQQ and IWM.
- **Research `r`.** The engine now prices with `options.risk_free_rate` (the key live's Greeks read);
  it was a bare 0.05 at three sites. Same value today.

### The ex-dividend entry block (W84; research twin of live's W51 block)

- **What live does.** The bot refuses a new position that SELLS A CALL when the projected next
  ex-dividend date falls between today and the chosen expiry (log event `trade_blocked_ex_div`): a
  short call can be assigned early the day before an ex-date. The projection is
  `ait.data.dividends.project_next_ex_div` (the last 4 ex-dates from yfinance, mean gap, stepped to
  today). It blocks 14-17% of sessions at entry DTE 14.
- **The research twin.** `backtest.ex_div_entry_gate`: `off` | `live` (live's rule exactly, from the
  ex-dates known before the entry day) | `exact` (any real ex-date in [entry, expiry]) |
  `last_k_days` (a real ex-date within `backtest.ex_div_entry_k_days`, default 5, of expiry). The
  schema default is `off`; `config_research.yaml` sets `live`. The manifest warns whenever research
  differs from live (`ex_div_entry:`); the registry row is a live twin.
- **Data.** The gate reads `research_dividends` in `data/historical.db`; any mode but `off` raises
  without it. Refresh it (yfinance; prints per-year counts and off-cycle gaps; writes
  `reports/dividends_audit_<date>.json`; a non-empty prefix is required):
  `python scripts/backfill_dividends.py --symbols QQQ SPY IWM --table-prefix research_ [--dry-run]`.
  Re-run it before a study that ends after the last stored ex-date.
- **Measure it:** `python scripts/ex_div_gate_ab.py --workers 3 [--entry-dte 14 7] [--k 3 5 7]`
  (read-only; about 15 minutes for 36 runs). Real engine runs, so a blocked entry frees its slot.
  The `live` arm answers the parity question; `exact` and `last_k_*` are the evidence for changing
  live's rule (decisions F2 / F3, the operator's call). Research does NOT model early assignment, so
  the A/B shows what the block costs, not what it protects against.

## Searching the entry DTE (W79a; research only)

- **Switch:** `python run_backtest.py --optimize-per-window --search-dte [--dte-range 0 45]
  [--optimize-strict]`. Off by default; `backtest.dte_search` (schema defaults, overlay-overridable)
  holds the ranges. The existing `IRON_CONDOR_SPACE` is untouched, so every other study is unchanged.
- **What is searched:** entry DTE, the expiry guard (sampled only for entries it can exempt: at or
  below the top of its range) and the hold fraction (which REPLACES `max_hold_days`). The flatten is
  not searched: `scripts/dte_dimension_flatness.py` showed it moves nothing while the take-profit
  ladder is on. The credit floors (`ic_min_credit*`) are never searched; run them as sensitivities.
- **Always pass `--optimize-strict` for numbers you will quote.** Without it a window whose
  optimisation fails runs the BASE config and the summary only reports a count
  (`optimization_failures`); with it the study stops.
- **Reading the report:** a band is a *candidate* only if it holds a top-decile trial in at least two
  thirds of windows. Promotion also needs its parity band to PASS (`scripts/parity_replay.py`) and its
  friction band to be ACTIVE (`data/short_dte_collection_status.json`). The report prints a reminder;
  it does not check either.
- **History limit:** the option-bar history is about two years (free tier), so a surface-priced study
  has only a handful of out-of-sample quarters. Use shorter training windows and/or pool symbols, and
  say so next to any number.
- **Decisions D1 and D8** (realistic engine, short-DTE friction floor) are in `config_research.yaml`;
  the live bot never loads it. `run_optimizer.py --apply` still writes only four named params.

## Studies and run records (W87; research only)

> New to research mode? Start with [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md): what it contains, how a run works, the learner, studies and how to read results. This section is the operations reference.

A **study** is a committed file, `studies/<name>.yaml`, that says what one research run changes relative to `config.yaml` +
`config_research.yaml`. `config_research.yaml` stays the stable default for ordinary research runs; editing it per study would silently
change every later default run.

```yaml
name: w88_wing_k_wider
question: Does a wider wing_k range change the wings Optuna picks, or the out-of-sample PF on QQQ?
rule: exploratory            # or the text of a PLAN.md heading where the pass rule was written before the run
settings:                    # merged over config.yaml + config_research.yaml; same validation
  backtest:
    max_concurrent_positions: 1
search_space:                # per-dimension overrides; defaults stay in optimization/param_spaces.py
  iron_condor:
    wing_k: [1.0, 2.0]
```

* **Run it:** `python run_backtest.py --study-file studies/<name>.yaml ...` (also `run_optimizer.py`). `--study-file`, not `--study`: the optimizer
  already has the Optuna `--study-name`. Commit the file before the run and never edit it afterwards (a changed study is a new file). A study whose
  `rule:` is not `exploratory` and names no PLAN.md heading is refused at start.
* **Refused on purpose:** `ibkr` / `api_keys`; any setting the **env contract** reads (`backtest.wing_k`, `backtest.ic_*`, `risk.iv_rank_cap`, ...): the
  contract reads the raw `config.yaml` and ignores both overlays, so the value would be silently ignored (use the environment variable or the CLI
  flag). Walk-forward window workers (`optimize_n_jobs > 1`) receive the parent's resolved settings through a pool initializer (W88a), and the
  optimizer's `--n-jobs` uses threads, so both see the overlay and the study. `run_optimizer.py --apply` is refused with a study: params found under a study's settings are not promotable to the live config.
* **Search-space overrides** are validated (known dimension, same type, `low < high`, inside the hard limits in `param_spaces.HARD_BOUNDS`) and the run
  warns when a range excludes the live value (`wing_k` 1.6, `delta_short` 0.20). The optimizer stores a hash of the effective space on the Optuna study
  and refuses to resume a stored study created over a different space.
* **The run record** (`run_record.json` in the run folder `reports/runs/<backtest|optimize>_<symbols>_<study|adhoc>_<timestamp>/`, gitignored; `--no-run-record` skips it) holds the resolved settings
  (secrets excluded), the RAW `config.yaml` / overlay / study text with sha256, the effective `AIT_*` env-contract values, the git commit and a dirty flag,
  data fingerprints (rows and date span per research table and a hash of each symbol's close series, read through a read-only connection), seeds, the
  command line and the merged search space with its hash. **MLflow is optional:** when installed the record is also logged (experiment
  `research_<study name or adhoc>`, tracking URI `MLFLOW_TRACKING_URI` or `sqlite:///data/mlflow.db`); a failure never fails a run.
* **When does a study need a PLAN.md entry?** A **pre-registered** study (the result could change a gate, a live parameter or a go-live criterion) gets
  an AS PLANNED entry BEFORE the run, with the pass rule and a pointer to the study file, and a result entry after; its `rule:` names that heading. An
  **exploratory** study needs only the file and its record; its results may not be quoted as evidence, and a decision that leans on one cites it (file and
  commit) in its own PLAN entry. `tests/test_w87b_study_layer.py` fails when a study names a PLAN heading that does not exist.
* **Defaults for a bare run (W87):** `--symbols` is `backtest.default_symbols` (QQQ: the symbol with historical option prices, so an IV surface can be
  built) and `--strategies` is `options.strategies` (iron condor); explicit lists win. Before W87 a bare `python run_backtest.py` ran ten symbols and four
  strategies and crashed on the first name without dividend rows (AAPL). The supervisor job passes explicit arguments and is unaffected. The walk-forward
  `slippage_pct` also follows `backtest.slippage_pct` (0.01; debit structures only) instead of a 0.03 literal.

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

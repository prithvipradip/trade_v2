# Claim ledger: docs/reference/SCRIPTS.md (catalogue `scripts/script_catalogue.yaml`)

Format: claim | source (file:line as read) | result. FIXED = the catalogue was wrong when first drafted and was corrected after reading the code.
Method: every script was read; write sites were found by reading the code (not the docstring); flags and the IB setting are read from the code by the generator on every run; no script was executed (the exceptions below are static checks: `py_compile`, an AST walk, and one attribute access on the settings model).

## Findings that changed the reference

- 13 scripts read `backtest.initial_capital`, which `BacktestConfig` no longer has (`hasattr` is false; `bc.initial_capital` raises `AttributeError` when evaluated against the loaded settings): `post_event_study.py:192`, `take_profit_study.py:141`, `wing_budget_study.py:106`, `run_integration_test.py:617,692,1245`, `wingk_study.py:43`, `range_floor_study.py:58`, `touch_stop_impact.py:57`, `shadow_tournament.py:42`, `shadow_round1b.py:42`, `shadow_tournament_uncorr.py:42`, `shadow_round3.py:42`, `shadow_round3_gated.py:46`, `shadow_round3_rerun.py:46`. They were marked "Currently fails" in the first version of the rows. **Fixed afterwards at the operator's request**: each now passes `initial_capital=None` (the walk-forward config resolves it to `account.capital` of the loaded settings, `walkforward.py:375-378`, as the `ablation_*` scripts already did); `run_integration_test.py` records `resolve_account_capital(None, <--config settings>)` in its archive metadata. Checked without running any study: the `WalkForwardConfig(...)` keyword expressions of all 13 were evaluated against the real settings (no attribute error; capital resolves to the config.yaml `account.capital`), and an AST walk finds no remaining read of a missing `BacktestConfig` attribute. The studies themselves were still not run.
- In `run_integration_test.py` the failing line is outside the `try` of `_section_e_walkforward`, so the exception leaves the section. Sections A–D (backfill, data quality, features, IC) come before it. The script is the only writer of Walk-Forward Lab archives, so no new archive can be produced until it is repaired.
- Docstrings that would mislead an operator and were corrected (comment-only): `backfill_dividends.py`, `load_econ_calendar_csv.py`, `build_research_calendars.py` told the reader to pass `--table-prefix research_` (since the history tables were unified, a non-empty prefix would create and fill a separate set of tables). Not corrected (listed in the rows): the docstrings of `ablation_study/armB/confirm/rerun`, `range_floor_study`, `touch_stop_impact` describe the wing_k study; `shadow_tournament_uncorr` names the wrong output folder.
- `backfill_risk_free_rate.py --dry-run` still writes the `risk_free_rates` table (phase A always runs): `backfill_risk_free_rate.py` comment before phase A. FIXED in the row (the draft said dry-run writes nothing).
- `load_econ_calendar_csv.py`, `backfill_dividends.py`, `build_research_calendars.py` write their audit JSON even with `--dry-run` (audit is written before the dry-run return). FIXED.
- `calibrate_option_spreads.py --update-config` uses ruamel.yaml (keeps comments): `calibrate_option_spreads.py:792-807`. The draft said comments are lost. FIXED. `export_production_params.py:121` uses `yaml.dump` (comments are lost) — row correct.
- `verify_deadman.py` does not do what the draft said: it checks the keeper wiring and the bot's liveness, pings the URL or its `/fail` address, and exit codes 0/1/2/3 mean armed-alive / problem / not armed / armed-but-bot-not-alive (`verify_deadman.py:95-203`). It refers to `ops/keeper_ait.sh`, which is not tracked in this checkout (`git ls-files ops` is empty). FIXED.
- `backfill_mlflow.py` sets no tracking address (only `run_record.py:266` sets `data/mlflow.db`), so it uses MLflow's default unless `MLFLOW_TRACKING_URI` is set. FIXED (draft said `data/mlflow.db`).
- Engine-based scripts: `engine.py:2203,2234` construct `HistoricalDataStore`, whose constructor runs `CREATE TABLE IF NOT EXISTS` (`historical.py:132-437`); nothing in `engine.py`/`walkforward.py` writes store rows (only loads and counts). Rows say "writes no database rows; creates any missing empty table". FIXED (draft said "opened read-only" for several studies, taken from docstrings). `ResearchData` (`replay.py:230`) is the read-only path and is what `parity_replay`, `surface_validate`, `excursion_range_study` use.
- `smoke_deploy.py` allows new empty tables and a schema-version change (only notes them); it fails on a row-count change (`smoke_deploy.py:206-262`). FIXED.
- `unify_history_tables.py` regrouped to One-time migration: PLAN.md records the merge and archive as done; `--drop-archives` remains.
- `status.py`/`status_server.py` connect with `sqlite3.connect(DB)` (not `mode=ro`) and issue only SELECTs (`status.py:168`, `go_live.py:163,472,311,481-509`).
- `keeper_ait.bat` also pings the dead-man URL, gated on `ait.monitoring.ops_health liveness`, and rotates its log at 20 MB (read from the file).
- `run_fractal_diagnostics.py` reads daily bars from the history store and falls back to Yahoo (`diagnostics/report.py:58-104`), VIX from Yahoo; writes `fractal_report_<SYMBOL>.html` and `ic_summary.csv`. FIXED (draft said Yahoo only).
- `probe_w53_roll_atomicity.py` evidence folder is `reports/w53_probe/` (`:95`); non-`D` account refused (`:201`); `validate_w55_synthetic_delta.py` refuses a non-`D` account too (`:78`). FIXED/added.
- `post_event_study`, `take_profit_study`, `wing_budget_study` change directory to `--data-root` (`post_event_study.py:335`) and also write `results.json` and `results.md`. FIXED.
- `run_optimizer.py` also writes `reports/optimization_result.json` (`:226`); `--apply` changes four parameters (`results.py:81-122`: `min_confidence`, `stop_loss_pct`, `trailing_stop_pct`, `breakeven_trigger_pct`), refused with `--study-file` (`run_optimizer.py:153`).
- Cross-reference check: for 4 rows the document named in "Explained in" does not mention the script, so the link was removed (`publish_shared_logs`, `dte_dimension_flatness`, `excursion_range_study`) or the document was updated (`check_first_rth_liveness` is now named in RUNBOOK deploy step 4).
- Other documents corrected while doing this: RESEARCH_GUIDE (`gate_parity_ab` and `ex_div_gate_ab` said "read-only"; they write no rows but open the history store), RUNBOOK step 4 (names the liveness script), WALKFORWARD_DASHBOARD (points at the script row for whether it currently runs).

## Rows read and confirmed without change

- `run_orchestrator.py` flags and dispatch: the argument parser and `if/elif` chain at the end of the file. `status_server.py`/`web_logs.py` `--port` only. `start_bot.bat` duplicate guard matches `run_orchestrator|ait\.main`.
- `check_first_rth_liveness.py` checks: the `main()` body; exit 1 on failure; `--no-telegram` read from `sys.argv`.
- `sync_ibkr_port.py`: twin-pair refusal and atomic write in `main()` (`port_candidates` check, `os.replace`).
- `read_shared_logs.py`: fetch `origin/shared-logs` unless `--no-fetch`; `publish_shared_logs.py`: worktree beside the checkout (`shared_logs.py:275-276`), push `HEAD:shared-logs` (`:348`), 16:45 ET Mon–Fri (`master.py:2512-2514`), today + last 4 weekdays (`recent_trading_days`, n=5).
- `shadow_referee.py`: `mode=ro`, DuckDB `read_only=True`, exit 1 on any BREAK (`:97,253,274-280`).
- `backfill_historical_data.py`: modes `choices=[intraday, iv, iv-intraday, both, daily, research, vix]` (`:73-76`); client id 90, port 4002 (`:100-103`); dry-run returns before the store is built (`:664-686`); the validation report is written only for the daily/iv modes (`:752-754`).
- `backfill_intraday.py`: client id 90, port 4002 (`:55-60`), `--dry-run` flag (`:74`); `INSERT OR REPLACE` is the script's own docstring statement (`:14`), the write call itself was not traced.
- `backfill_option_history.py`: `POLYGON_API_KEY` required before the dry-run branch (`:470`), status file `data/backfill_option_history_status.json` (`:147`).
- `run_integration_test.py`: client id 88, port 4002 (`:76-82`), default DB/out (`:37-38`), purge of `window_*.json` (`:1203`), `_archive_to_data_branch` stash/checkout/commit/push (`:1125-1190`).
- `parity_replay.py`, `surface_validate.py`, `fill_vs_model_report.py`, `iv_rank_parity_probe.py`, `range_horizon_study.py`, `dte_dimension_flatness.py`: read-only handles as stated in the rows (`replay.py:230`, `iv_rank_parity_probe.py:48`, `dte_dimension_flatness.py:55`, `fill_vs_model_report.py:33`).
- `gate_parity_ab.py` has four arms (`:45-50`); `ex_div_gate_ab.py` arms off/live/exact/last_k_K (`:41-43`).
- `restate_d1.py` (`--apply` updates trades, daily_stats, bot_state and the DuckDB mirror: `:172-220`), `restate_r16_dailystats.py` (`:26-58`), `restate_w21_tz_aware.py` (snapshot `:112-115`, tables `:42-45`).
- `typecheck.py`, `literal_census.py` (`:31,180`), `gen_docs.py`.
- Historical group: output folders `OUT = ROOT / "reports" / ...` read in each file; training lengths, symbols and wing_k as stated; all 13 compile (`py_compile`).

## Not established

- Whether any script still runs end to end on this machine or on the trading box (none was executed). The Historical studies, `backfill_*`, `calibrate_*` and the IB probes were not run.
- Whether `restate_r16_dailystats.py` was applied (PLAN.md asks the operator to run it; no record of it was found).
- Whether the exit codes and Telegram texts in `check_first_rth_liveness.py` match the current alerts (the script reads the log for event names; not compared with the live emitters).
- Which MLflow store holds existing runs on the operator's machine.
- That every engine call path avoids store writes: the engine and walk-forward modules contain no store writes, but a call into another package from them was not traced.
- The remaining `docs` mentions of script behaviour were not re-audited beyond the cross-reference check above.

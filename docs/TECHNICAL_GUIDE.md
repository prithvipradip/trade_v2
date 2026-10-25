# Technical guide

For developers: how the code is organised, the conventions it follows, how to extend it, and how to test and ship a change safely. Not for: operating the bot ([RUNBOOK.md](RUNBOOK.md)), how the live system behaves ([GUIDE.md](GUIDE.md)), models ([MODELS.md](MODELS.md)), or running research ([RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)).
See also: [GETTING_STARTED.md](GETTING_STARTED.md) (install) · [GLOSSARY.md](GLOSSARY.md) · `PLAN.md` (the append-only decision and audit record).

**Contents:** [1 Layout and ownership](#1-layout-and-ownership) · [2 Conventions](#2-conventions) · [3 Changes that need the operator's confirmation](#3-changes-that-need-the-operators-confirmation) · [4 Extending the system](#4-extending-the-system) ·
[5 Testing](#5-testing) · [6 Quality gates](#6-quality-gates) · [7 Git conventions and the audit register](#7-git-conventions-and-the-audit-register) · [8 Maintaining the documentation](#8-maintaining-the-documentation) · [9 Known discrepancies](#9-known-discrepancies)

---

## 1. Layout and ownership

One owner per fact: each concern lives in one package, and other packages import it. The system's runtime structure (processes, tasks, data flow) is in [GUIDE.md](GUIDE.md) section 2; this section is about where the code is.

```
run_orchestrator.py      supervisor entry point (a thin wrapper; the supervisor is src/ait/orchestration/master.py)
run_backtest.py / run_optimizer.py   research entry points
status_server.py · status.py · web_logs.py   dashboards and status data
config.yaml · config_research.yaml · studies/   configuration layers
src/ait/              the package (src layout; installed with pip install -e .)
scripts/              operator and research tools (smoke_deploy.py, typecheck.py, gen_docs.py, backfills, study scripts); each is described in reference/SCRIPTS.md (hand-checked source: scripts/script_catalogue.yaml)
tests/                topic files plus per-change regression files; tests/data holds golden inputs and the literal allowlist
docs/                 the documentation set (README.md has the map)
deprecated/           retired code, kept for history, never imported
data/ models/ logs/ reports/   runtime state (git-ignored)
```

| Package under `src/ait/` | Owns |
|---|---|
| `broker/` | The only code that talks to Interactive Brokers (connect/reconnect, account, orders, contracts) |
| `data/` | Fetching and caching market data (including the economic calendar) |
| `ml/`, `strategies/`, `risk/`, `execution/` | Models and features; strategy shapes and selection; entry gates and sizing; executor, portfolio, exit policy, reconciler |
| `bot/` | The orchestrator loop, SQLite state, the market scheduler |
| `orchestration/` | The process supervisor (`master.py`) |
| `monitoring/`, `learning/`, `reporting/`, `notifications/` | Watchdog and health; adaptation from closed trades; reports and the go-live verdict; Telegram |
| `backtesting/`, `optimization/` | Research engine and Optuna searches |
| `dashboard/` | Streamlit analytics and the Walk-Forward Lab |
| `config/` | Settings models, the runtime environment contract, invariants, the feature catalogue |
| `utils/` | Time, logging, cadence |

Every package and module, with the first line of its docstring, is in the generated [reference/MODULES.md](reference/MODULES.md); every script (what it does, whether it needs IB, what it writes, when to run it) is in [reference/SCRIPTS.md](reference/SCRIPTS.md); the repository tree above is the short form.

Single-source-of-truth definitions, imported and never re-typed: `CREDIT_STRATEGIES` and `UNDEFINED_RISK_STRATEGIES` in `strategies/base.py`; the go-live verdict and real-close logic in `reporting/go_live.py`; the environment contract in
`config/runtime_env.py` (`CONTRACT_DEFAULTS`, `capital_base()`); the fractal gate in `ml/regime_gate.py`; IV rank in `data/iv_rank.py`; per-leg friction in `backtesting/friction.py`; the feature catalogue in `config/feature_names.py`.
Duplicating one of these has repeatedly caused silent divergence.

## 2. Conventions

- **Config is the only home for trading values.** Pydantic models in `config/settings.py`, all `extra="forbid"` (a mistyped key is a startup error, by design), with `Field(...)` bounds, validators and a `description=` that records why a default is what it is. Secrets come only from `.env`
  (`IBKREnvConfig`, `APIKeysConfig`) and are never logged (`_SECRET_SECTIONS`, `_is_secret`). Resolution precedence: explicit argument > environment/`.env` > `config.yaml` > `CONTRACT_DEFAULTS`; use `resolve_config_value()` and `contract_str|float|flag()` rather than a local fallback literal.
  A hardcoded strategy or risk number is a regression.
- **Research overlay.** Research-only values live in `config_research.yaml`, which holds only the keys that differ from `config.yaml` and is merged over it by research entry points through `ait.config.research.use_research_config()` (validate, then freeze). The live bot and supervisor never load it (a test fails if
  anything under `src/ait` references it). A key present in both files must differ (a test fails on a redundant copy). New research entry points call `use_research_config()` first. Details: [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) "Configuration layers".
- **Fixed limits are code, not config.** The few values that must not be editable are registered in `config/invariants.py` (live) and `invariants_research.py`, logged at start and listed in generated tables ([GUIDE.md](GUIDE.md) section 7.4). Tunables go in config; do not add a "limit" key that nothing reads (a test fails on a setting with no reader).
- **Every entry point** calls `apply_runtime_env_defaults()` before importing any trading subsystem (it also sets the KMP/OMP guards that must precede numpy and xgboost).
- **Logging:** `structlog` through `get_logger("package.module")`; a snake_case event name first, then key-value pairs (`log.info("order_placed", symbol=..., qty=...)`). Never log secrets.
- **Clocks are timezone-aware.** Use `ait.utils.time.now_et()`, `ET` and `is_market_open()`. A bare `datetime.now()` in scheduling, gating or trade timestamps is a known bug class. Read one `now` per decision and pass it down.
- **Style:** `from __future__ import annotations`, dataclasses for DTOs, `async`/`await` for IO, modern type hints, 100-column lines. ruff is configured (`E,F,I,N,W,UP`) but deliberately not a gate; do not add one as a side effect of other work.
- **Comments carry provenance.** Non-obvious code is annotated with the audit round or fix wave and the incident that produced it (see [section 7](#7-git-conventions-and-the-audit-register)). Preserve and extend those when you change defended behaviour; they are the rationale record.
- **Fail closed, say so, keep managing positions.** A gate that lacks information refuses the entry, logs a named event, alerts if it persists, and never blocks an exit. Decide each new gate's fail direction explicitly.

## 3. Changes that need the operator's confirmation

**Ask first before modifying any of the following, and before restarting the live bot process** (restarting the dashboard, log viewer or status server is fine; the bot and the supervisor are the operator's call).

- **IB connectivity**: `broker/ibkr_client.py` (connect/reconnect, the client-id fallback ladder and `foreign_open_order_ids`, `managedAccounts()` pinning, `reqMarketDataType`), `broker/account.py`, `broker/orders.py`, `broker/contracts.py`.
- **Order execution and routing**: `execution/executor.py` (the undefined-risk refusal gate `AIT_ALLOW_UNDEFINED_RISK`, the order-rate backstop, market-hours enforcement, row-written-before-`placeOrder` ordering, NBBO validation and the reprice ladder, exit price bounds), `execution/portfolio.py`, `execution/exit_policy.py`,
  `execution/reconciler.py` (exactly-once booking, `HALT_UNTRACKED`, assignment handling, the `*_unknown_exit` sentinels contracted with `go_live.py`).
- **Risk**: `risk/manager.py`, `circuit_breaker.py` (persists across restarts), `pdt_guard.py`, `position_sizer.py`, `correlation.py`.
- **Entry-gate mode**: `config.yaml` `ml.entry_gates_enabled` is `false` (observe) on purpose; flipping it is the operator's call after a pre-registered test ([MODELS.md](MODELS.md) "Gating: observe and enforce").
- **Paper/live switches**: `trading.mode`, `.env` `TRADING_MODE` and `AIT_ALLOW_LIVE`, the `--paper` argument, the decision in `config/launch_mode.py` (used by the supervisor and by `ait.main`), `.env` `IBKR_PORT`, `AIT_MARKET_DATA_TYPE`, `learning.paper_trading_mode`, `account.pdt_account_under_25k`.
- **Secrets**: `.env` only; never echo values into logs, reports, commits or `PLAN.md`.
- **Kill switches and runtime files**: `data/HALT`, `data/HALT_UNTRACKED`, `data/RESTRICTED.txt`, `data/orchestrator.lock`, `data/bot_heartbeat`, `data/KEEPER_PAUSE`. Never create or delete these casually; [RUNBOOK.md](RUNBOOK.md) defines the procedures.
- **Live state stores**: `data/ait_state.db`, `data/ait_analytics.duckdb` are the track record. Ad-hoc scripts open them read-only (`file:...?mode=ro`); `smoke_deploy.py` enforces no-write.
- **Deploy mechanics**: follow [RUNBOOK.md](RUNBOOK.md) "Deploy checklist". A change to `orchestration/master.py` needs a full supervisor restart, not a bot restart.

Comment-only edits in these files are still edits to a sensitive file: say so when you make them.

## 4. Extending the system

### Adding a feature
1. Compute it in the matching `_add_*` method of `ml/features.py`, from OHLCV and context only (stationary: ratios, slopes, normalised levels).
2. Add its name to `CATALOGUE` in `config/feature_names.py`. To ship it off, add it to `OFF_BY_DEFAULT` too.
3. Add its line to the group's `use:` block in `config.yaml` (set to `false` if it ships off). A test fails until the file lists everything the code produces.
4. Keep the shipped model recognisable: the feature-spec hash (`default_spec_hash()` in `ml/feature_spec.py`) covers the window settings, `FEATURE_SET_VERSION` and any non-default switch or window, and is how a saved model is matched to the spec it was built under ([MODELS.md](MODELS.md) "Saved models"). A feature that ships off, with no new window setting, should leave it unchanged; adding a window setting or bumping the version changes it on purpose. The golden-input tests under `tests/data/` pin feature output at the defaults.
5. Regenerate the documentation tables: `python scripts/gen_docs.py` ([section 8](#8-maintaining-the-documentation)).
6. Study it in research before turning it on: [MODELS.md](MODELS.md) "Changing a model setting".

### Adding a strategy
Subclass `Strategy` in `strategies/`, emit `Signal`s, register it in the selector, add its name to `CREDIT_STRATEGIES` or `UNDEFINED_RISK_STRATEGIES` if applicable (import these; do not copy them), give it a row in `position_sizing.strategy_risk`, decide which model gates it, and add an exit path in
`exit_policy.py` if it differs. Enable it in `options.strategies` only by a recorded decision. Undefined-risk shapes additionally hit the executor's interlock.

### Adding a gate
Put it in the stage where its information first exists (the entry pipeline for market conditions, the risk manager for account state). Decide its fail direction, log a named event, record the counterfactual on rejection, and make it observe-first if it depends on a model.

### Adding a configuration key
Add a field with bounds and a `description` to the matching model in `settings.py`; list it in `config.yaml` (research-only keys go in `config_research.yaml` with a provenance comment); read it through the loaded settings or `resolve_config_value()`; then regenerate the references. A test fails if a field is never read
and one fails if a key is listed in both config files with the same value.

### Adding a scheduled job
Register it in `master.py::_register_recurring_jobs`, give it a catch-up if a missed slot matters, make it safe to run twice, alert on failure, and add its row to the prose table in `scripts/gen_docs.py` (the generator refuses an undocumented job). A supervisor change needs a full supervisor restart.

### Adding an alert
Send it through the notifier with a stable leading phrase, then add a row to the matching table in [RUNBOOK.md](RUNBOOK.md): what it means and what the operator does. An alert with no operator action is listed as informational.

### Adding a script
Put it in `scripts/` and add an entry for it to `scripts/script_catalogue.yaml`: what it does, when to run it, what it writes (read from its write sites, not from its docstring) and whether it needs IB. Then run `python scripts/gen_docs.py`: the generator refuses a script with no entry, an entry with no script, and an IB setting that disagrees with the script's imports, and it writes [reference/SCRIPTS.md](reference/SCRIPTS.md). Take the entry's `stamp` with `python scripts/gen_docs.py --restamp scripts/<name>.py` once you have checked the entry against the code. Open the live databases read-only (section 3).

### Research numbers and the literal census
A number that decides a research result belongs in a setting the researcher can see, not in code. `tests/test_w88_literal_census.py` walks the AST of the result-affecting research modules and compares each function's numeric literals with the accepted list `tests/data/literal_allowlist.json`; a new, changed or
removed literal fails the test until a person has looked at it. Workflow: `python scripts/literal_census.py` prints the census and any difference; prefer to convert the literal into a setting (and read it through the research config); if it really is fixed, `python scripts/literal_census.py --update`
records it (keeping existing categories and classifying new entries; check the category it chose). The closed set of categories: `display` (formatting), `epsilon` (numerical tolerance), `fixed_feature_group` (a feature group left fixed on purpose), `dormant_path` (legacy structures off the iron-condor path), `structural` (a protocol or model constant, not a tunable), and
`registered_followup` (a research number still in code, registered in `PLAN.md`, with a note). Trivial values (0, 1, 2, -1), booleans, `round()` digits, logging arguments and docstrings are not counted.

## 5. Testing

```bash
python -m pytest tests/ -q -m "not ibkr and not slow"     # fast lane (CI and pre-deploy); `pytest` alone applies the same markers
python -m pytest tests/ -q -m "slow and not ibkr"         # nightly lane
python -m pytest tests/ -m ibkr                           # needs a live Gateway on 127.0.0.1:4002
RUN_INTEGRATION_TESTS=1 pytest tests/test_integration.py -v   # self-skips without the variable
```

House rules:
- **Execute the real code path.** Never assert on source strings.
- **Add a class-guard test** (often an AST scan) that catches the *next* instance of a bug class, not only the reported one.
- **Pin a non-ET `TZ`** when testing clock behaviour.
- **Never edit a source file while the suite is running.** A number of tests read source from disk with `inspect.getsource()` and produce phantom failures.
- Tests are topic files (`test_<area>.py`) plus per-change regression files named for the change that produced them. Markers `ibkr` and `slow` are excluded by default.
- Documentation is tested too ([section 8](#8-maintaining-the-documentation)).

## 6. Quality gates

| Gate | Command | Status |
|---|---|---|
| Fast lane | `python -m pytest tests/ -q -m "not ibkr and not slow"` | CI on every push and PR (Python 3.13) |
| Slow lane | `python -m pytest tests/ -q -m "slow and not ibkr"` | CI on a 06:00 UTC schedule |
| Syntax | `python -m compileall -q src` | CI on every push |
| Types | `python scripts/typecheck.py` | **Blocking** for attribute, name and call-argument errors in `config`, `bot`, `execution`, `risk`. `mypy src` as a whole reports many errors and is informational only; needs `pip install -e ".[dev]"` |
| Docs | `python scripts/gen_docs.py --check` | Fails when a generated block or reference file differs from the code |
| Post-deploy | `python scripts/smoke_deploy.py` | Must print `SMOKE PASS`; read-only, no broker; needs the real `config.yaml` and `data/ait_state.db` |

## 7. Git conventions and the audit register

Commit subjects are `W<n>: <what changed>` for fix waves, `R<n>: ...` for audit rounds, `PLAN.md: record ...` for ledger appends, and occasionally `fix(<area>): ...`. Verified deploys are tagged `deploy-YYYYMMDD` ([RUNBOOK.md](RUNBOOK.md) "Deploy checklist").

**What R and W mean.** `R<n>` is an audit round (a structured review that registers findings); `W<n>` is a fix wave (a batch of changes that remediates findings). You will see these IDs in code comments, test names and `PLAN.md`. They are the provenance record for why code looks the way it does.
They appear only in code, tests, `PLAN.md`, this guide and [STUDIES_FINDINGS.md](STUDIES_FINDINGS.md); the user-facing documents describe behaviour without them.

**`PLAN.md` is an append-only register** of audit rounds, findings and waves. Read its last entries before starting a new round. Never overwrite, edit, reorder or delete an existing entry; only append at the end, and keep it focused (a decision record, not a log of every small note).
The programme's conventions: a lead becomes a finding only when a code-reading lens *and* an execution probe both concede; register, do not fix, while auditing; probe in order of consequence; every wave entry ends with a correct "what is next" pointer.

## 8. Maintaining the documentation

Every document has one reader and one job; every fact has one home; other documents link, they do not copy. The map is in [README.md](../README.md). Rules:

- Each document opens with who it is for and what it is not for, plus "See also". Link to a heading; do not copy its content; link [GLOSSARY.md](GLOSSARY.md) on first use of a term.
- **Volatile numbers are generated**, not typed: `scripts/gen_docs.py` fills the regions between a pair of BEGIN GENERATED / END GENERATED comment markers in the documents and rewrites `docs/reference/SETTINGS_LIVE.md` and `SETTINGS_RESEARCH.md`. Run `python scripts/gen_docs.py` after a change to a setting, a
  limit, a job or a feature, and `--check` in CI-style checks. Prose that goes with a generated table is keyed by the code identifier, so a new job, limit or dimension forces someone to document it.
- **Scripts are catalogued by hand and fingerprinted.** Each entry in `scripts/script_catalogue.yaml` carries a fingerprint of its script's code (docstrings, blank lines and whole-line comments do not count). When the code changes, `tests/test_docs_generated.py` fails and names the script: re-read the entry against the changed code, correct it, then `python scripts/gen_docs.py --restamp scripts/<name>.py`. Restamping without reading the entry defeats the check, and it cannot tell whether the entry is still true; that is the reader's job.
- **Pointers from code and config** to a document use one form, `docs/<FILE>.md "<heading>"`, and a test resolves every one. If you rename a heading, the test lists the pointers to fix.
- **Doc impact of a change:** a new alert adds a RUNBOOK row; a new setting needs a `description` (and appears in the references); a new feature or model setting goes to MODELS; a new research setting or study to RESEARCH_GUIDE; a decision to `PLAN.md`; a result only to STUDIES_FINDINGS.
- **No study or experiment results anywhere except [STUDIES_FINDINGS.md](STUDIES_FINDINGS.md).** Other documents state behaviour and link to the finding.
- Documentation tests check links, anchors, named files, settings and identifiers, banned terms per document, and generated-block freshness.

## 9. Known discrepancies

- `CLAUDE.md` is git-ignored (local to each checkout), so committed documents never rely on it; the confirmation list in [section 3](#3-changes-that-need-the-operators-confirmation) is the committed copy.
- `src/ait/strategies/base.py` says `GO_LIVE_VERDICT_STRATEGIES` "lives in `ait.strategies.constants`"; that module does not exist. The constant is defined in `ait/reporting/go_live.py` and re-exported.
- `.github/workflows/ci.yml` comments call torch "a mandatory dep (transformers/FinBERT)" and pre-install it; torch is an unused optional extra and nothing under `src/` imports it (the `sentiment` extra and the sentiment subsystem are retired).
- `docs/RUNBOOK.md`, `start_bot.bat` and `keeper_ait.bat` assume the Windows trading box (PowerShell commands, hardcoded user paths); development and deploys also happen on macOS (relaunch with `nohup`; `setsid` does not exist there).
- The macOS development environment may run an older Python and numpy than `requires-python` (`>=3.13`) and `constraints.txt` (the trading box); `ruff` and `mypy` may be missing, so `scripts/typecheck.py` needs `pip install -e ".[dev]"` first.
- The documents in `docs/archive/` are point-in-time snapshots, not current state.

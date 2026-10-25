# AIT v2

An autonomous options-income trading bot. It connects to Interactive Brokers, scans a small list of liquid ETFs, and sells defined-risk premium (iron condors), with entries gated by machine-learning models and a portfolio risk manager,
exits managed on a fast monitor loop, and a research stack (walk-forward backtester, optimiser, studies) for testing changes before they are promoted. It currently runs on a **paper** account, building a track record against the go-live gates.

> **Disclaimer.** This is research software. Trading options involves substantial risk, including the loss of more than your initial investment in some structures. Use a paper account until you have read and understood the system. Nothing here is investment advice, and the authors accept no liability for losses.

## Where to go

Each document has one reader and one job; a fact lives in one place and the others link to it.

| I want to... | Read |
|---|---|
| Install it and run it for the first time | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Understand how the live system works, and what its limits are | [docs/GUIDE.md](docs/GUIDE.md) |
| Respond to an alert, stop the bot, deploy, run a drill | [docs/RUNBOOK.md](docs/RUNBOOK.md) |
| Understand a model, a feature or a gate, or change one safely | [docs/MODELS.md](docs/MODELS.md) |
| Run, trust and read a backtest, optimiser run or study | [docs/RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md) |
| See what past studies found, and whether I can cite them | [docs/STUDIES_FINDINGS.md](docs/STUDIES_FINDINGS.md) |
| Read the Walk-Forward Lab dashboard | [docs/WALKFORWARD_DASHBOARD.md](docs/WALKFORWARD_DASHBOARD.md) |
| Change the code, add a feature, strategy, gate or setting | [docs/TECHNICAL_GUIDE.md](docs/TECHNICAL_GUIDE.md) |
| Look up a term | [docs/GLOSSARY.md](docs/GLOSSARY.md) |
| Look up a module (what each file in `src/ait` is) | [docs/reference/MODULES.md](docs/reference/MODULES.md) (generated) |
| Find a script: what it does, whether it needs IB, what it writes, when to run it | [docs/reference/SCRIPTS.md](docs/reference/SCRIPTS.md) (generated from a hand-checked catalogue) |
| Look up a setting (name, range, default, meaning) | [docs/reference/SETTINGS_LIVE.md](docs/reference/SETTINGS_LIVE.md), [docs/reference/SETTINGS_RESEARCH.md](docs/reference/SETTINGS_RESEARCH.md) (generated) |
| Find why something was decided, or what is planned | [PLAN.md](PLAN.md) (the append-only decision and audit record) |
| Read old audits and retired-model papers | [docs/archive/](docs/archive/README.md) (point-in-time, not maintained) |

## At a glance

- **Live** (`python run_orchestrator.py`): a supervisor starts the trading bot (restarting it if it dies), three localhost dashboards and the scheduled jobs. Configuration is [`config.yaml`](config.yaml); secrets are in `.env`.
- **Research** (`python run_backtest.py`, `python run_optimizer.py`): never touches the live bot or its state. Research-only settings are in [`config_research.yaml`](config_research.yaml) and committed study files in [`studies/`](studies).
- **Dashboards** (localhost): status and go-live scorecard on 8503, analytics on 8501, logs on 8502, Walk-Forward Lab on 8080.
- **Requirements:** Python 3.13, an Interactive Brokers account with IB Gateway or TWS. Install from this checkout only (`pip install -e . -c constraints.txt`); the PyPI package named `ait` is unrelated.

## Repository

```
src/ait/        the package          config.yaml, config_research.yaml, studies/   configuration
scripts/        operator and research tools      tests/      the test suite
docs/           the documents above              PLAN.md     decision and audit record
deprecated/     retired code, never imported     data/ models/ logs/ reports/   runtime state (git-ignored)
```

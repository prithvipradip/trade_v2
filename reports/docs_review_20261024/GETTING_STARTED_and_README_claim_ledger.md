# Claim ledger: docs/GETTING_STARTED.md and README.md (after fixes)

Format: claim | source (file:line as read) | result. FIXED = the document was wrong and was corrected.

## GETTING_STARTED
- Python `>=3.13,<3.14`; pip refuses older interpreters | pyproject.toml requires-python | FIXED (said "installs but unsupported")
- ib-insync 0.9.86 paired with Gateway 1044 | constraints.txt:4 | OK
- Paper accounts run on delayed data, AIT_MARKET_DATA_TYPE=4 | runtime_env.py:77, ibkr_client.py:156 | OK
- Polygon key used only by the option-history backfill | scripts/backfill_option_history.py:11,470; grep src shows no other use | FIXED
- Finnhub key: setting exists, nothing calls Finnhub | settings.py APIKeysConfig; grep of src, scripts | FIXED (said supplementary data source)
- Install commands (`pip install -e . -c constraints.txt`, never `pip install ait`) | CLAUDE.md, pyproject | OK
- Gateway settings: socket clients on, Read-Only API off, ports 4002/4001/7497/7496 | standard IB ports; IBC keys from IBC itself, NOT from this repo | OK for ports; IBC keys external (unverified here)
- `.env` variables IBKR_HOST/PORT/CLIENT_ID/ACCOUNT, POLYGON, FINNHUB, TELEGRAM_*, MLFLOW_TRACKING_URI | settings.py IBKREnvConfig/APIKeysConfig, .env.example | OK
- AIT_MARKET_DATA_TYPE lives in .env only | runtime_env.py:22-56,132 | OK; TRADING_MODE and AIT_ALLOW_LIVE rows ADDED (read by config/launch_mode.py)
- config.yaml ships trading.mode paper; mistyped key is a startup error | config.yaml, extra="forbid" | OK
- Supervisor starts bot, three dashboards, scheduled jobs; bot starts as paper unless live is fully enabled | master.py BotManager.start, launch_mode.py | CHANGED
- Command table: --status/--retrain/--report/--backtest/--refresh-fundamentals | run_orchestrator.py:86-110 | OK
- `python -m ait.main --dashboard-only` runs only the Streamlit app | main.py:232-237 | FIXED
- status.py reads state DB and logs only, includes scorecard | status.py:1-8,23,260 | OK
- Research commands and flags | CLAUDE.md, flag-existence check over all docs | OK
- Dashboard ports 8503/8501/8502; Lab 8080 by hand | config.yaml:923-924, settings.py DashboardConfig, export docs | OK
- data/ait_state.db created by the bot; smoke needs it and refuses to create it | state.py:120-130, smoke_deploy.py:467-472 | OK
- smoke_deploy prints SMOKE PASS (13 checks) on this checkout | run during this work | OK
- Models trained in background at start | orchestrator.py:631,677 | OK
- Stopping: Ctrl-C handlers, KEEPER_PAUSE, bot decides every exit itself | master.py:2609-2610, keeper_ait.bat:40-48, orchestrator.py exit path | OK

## README
- Description (ETFs, iron condors, paper) | CLAUDE.md overview, config.yaml | OK
- Dashboard ports and Python 3.13 | as above | OK
- Document map lists every maintained document and the three reference files | tests/test_docs.py::test_readme_map_lists_every_maintained_document | OK (tested)

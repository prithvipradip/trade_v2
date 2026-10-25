# Getting started

Install AIT v2, connect it to Interactive Brokers, and run it on a **paper** account. Not for: what to do about an alert ([RUNBOOK.md](RUNBOOK.md)), how the system works ([GUIDE.md](GUIDE.md)), how to run research ([RESEARCH_GUIDE.md](RESEARCH_GUIDE.md)).
See also: [README.md](../README.md) (document map) · [GLOSSARY.md](GLOSSARY.md).

> Trading options involves substantial risk. Run on a paper account until the go-live gates in [GUIDE.md](GUIDE.md) section 12 pass. Nothing here is investment advice.

## 1. What you need

| Item | Needed? | Notes |
|---|---|---|
| Python **3.13** (`>=3.13,<3.14`) | Yes | Set by `pyproject.toml` (`requires-python`); `pip` refuses to install on an older interpreter |
| Git | Yes | To clone |
| An Interactive Brokers account with **IB Gateway** (or TWS) | Yes | A free paper account is enough. The code is paired with `ib-insync 0.9.86` and Gateway 1044 (see `constraints.txt`) |
| Options market data | Yes for live | Paper accounts run on delayed data by design (`AIT_MARKET_DATA_TYPE=4`) |
| Polygon API key | Optional | Used only by the research option-history backfill (`scripts/backfill_option_history.py`); free tier |
| Finnhub API key | Optional | The setting exists, but no current code calls Finnhub; leave it blank |
| Telegram bot | Strongly recommended | The bot reports through it; every alert in [RUNBOOK.md](RUNBOOK.md) arrives there |
| IBC (Gateway auto-login) | Optional | Needed only for unattended restarts |

## 2. Install

```bash
git clone <your-repository-url> trade_v2
cd trade_v2
python3.13 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -e . -c constraints.txt   # the exact runtime the trading box uses
# developers also want:  pip install -e ".[dev]"
```

The package is installed from this checkout only. The `ait` name on PyPI is unrelated third-party code: never `pip install ait`.
`constraints.txt` pins the versions the trading box runs; leave `-c constraints.txt` off only if you are deliberately experimenting.

## 3. Set up IB Gateway

1. Install IB Gateway and log in to the **paper** account.
2. Gateway → Configure → Settings → API → Settings:
   - check "Enable ActiveX and Socket Clients";
   - **uncheck "Read-Only API"** (with it on, every order is rejected; the bot reports `READ-ONLY: IB Gateway is rejecting all orders`);
   - socket port: **4002** (Gateway paper). The usual ports: Gateway paper 4002, Gateway live 4001, TWS paper 7497, TWS live 7496.
3. Optional: IBC keeps Gateway logged in unattended (`TradingMode=paper`, `OverrideTwsApiPort=4002`, `ReadOnlyApi=no` in its `config.ini`).

## 4. Configure

```bash
cp .env.example .env     # then edit; .env is git-ignored and holds every secret
```

| `.env` variable | Meaning |
|---|---|
| `IBKR_HOST`, `IBKR_PORT` | Where Gateway listens (usually `127.0.0.1`, `4002` for paper Gateway) |
| `IBKR_CLIENT_ID` | Session id for this bot; keep it unique per client |
| `IBKR_ACCOUNT` | Your account number (paper accounts start with `DU`) |
| `POLYGON_API_KEY`, `FINNHUB_API_KEY` | Optional keys (see the table above) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Alerts. Create a bot with @BotFather, message it once, then read your chat id from `https://api.telegram.org/bot<TOKEN>/getUpdates` |
| `TRADING_MODE` | `paper` (default) or `live`. Live also needs `trading.mode: live` in `config.yaml`, `AIT_ALLOW_LIVE=1` and a live IB endpoint |
| `AIT_ALLOW_LIVE` | Environment interlock for live trading: unset or anything but `1` means the bot will not start as live. Leave it unset on a paper box |
| `AIT_MARKET_DATA_TYPE` | Market-data entitlement: `4` delayed (paper, the default), `1` live. Lives in `.env` only, never in `config.yaml` |
| `MLFLOW_TRACKING_URI` | Optional experiment tracking for research runs |

Never paste these values into a log, a report or a commit. Trading values (universe, risk, exits, models) live in [`config.yaml`](../config.yaml), not in `.env`: the meaning of each is in
[reference/SETTINGS_LIVE.md](reference/SETTINGS_LIVE.md) and the rules that govern them in [GUIDE.md](GUIDE.md) section 13. `config.yaml` ships set to `trading.mode: paper`; leave it.
A mistyped key is a startup error by design.

## 5. Run

### The whole system (normal way)

```bash
python run_orchestrator.py
```

This starts the supervisor, which launches the trading bot (restarting it if it dies), the three dashboards, and the scheduled jobs (daily retrain, reports, backups, weekly backtest). Windows:
double-click `start_bot.bat` (the same command; `keeper_ait.bat` relaunches it if it is stopped). The supervisor's jobs and timings are in [GUIDE.md](GUIDE.md) section 3.
The bot starts as paper unless `config.yaml` and `.env` both say live and `AIT_ALLOW_LIVE=1` is set; the supervisor also refuses to start a paper bot against a live-account IB port. Going live is a deliberate change: [RUNBOOK.md](RUNBOOK.md) "Go-live day checklist".

### Individual commands

| Command | What it does |
|---|---|
| `python run_orchestrator.py --status` | Is the bot running? |
| `python run_orchestrator.py --retrain` | Retrain the models now (needs Gateway; see [MODELS.md](MODELS.md)) |
| `python run_orchestrator.py --report` | Write the daily report now |
| `python run_orchestrator.py --backtest` | Run the weekly backtest job now |
| `python run_orchestrator.py --refresh-fundamentals` | Refresh equity fundamentals |
| `python -m ait.main [--config config.yaml] [--paper]` | Run only the bot, without the supervisor (no auto-restart, no dashboards) |
| `python -m ait.main --dashboard-only` | Only the Streamlit analytics dashboard (port 8501), without trading |
| `python status.py` | Print a status snapshot (state DB and logs only, no broker connection), including the go-live scorecard |
| `python scripts/smoke_deploy.py` | Read-only health check; must print `SMOKE PASS` |

There are many more scripts than these (backfills, studies, one-time repairs, probes that place orders on a paper account). Before running any script you have not run before, read its row in [reference/SCRIPTS.md](reference/SCRIPTS.md): what it does, whether it needs IB, what it writes and when to run it.

### Research (does not touch the live bot)

| Command | What it does |
|---|---|
| `python run_backtest.py --symbols SPY QQQ --capital 50000` | Walk-forward backtest |
| `python run_backtest.py --study-file studies/_template.yaml` | The same as a recorded study |
| `python run_backtest.py --symbols SPY QQQ --optimize-per-window --optimize-n-trials 50` | Backtest with per-window optimisation |
| `python run_optimizer.py --strategies iron_condor --symbols SPY QQQ --n-trials 100` | Strategy-parameter search |

How to run and read these safely: [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md).

### Dashboards (localhost only; the supervisor starts them)

| URL | Shows | Started by |
|---|---|---|
| http://localhost:8503 | Live status: positions, unrealized P&L, go-live scorecard | `status_server.py` |
| http://localhost:8501 | Analytics: trade history, breakdowns | `streamlit run src/ait/dashboard/app.py` |
| http://localhost:8502 | Log viewer | `web_logs.py` |
| http://localhost:8080 | Walk-Forward Lab (research; started by hand) | [WALKFORWARD_DASHBOARD.md](WALKFORWARD_DASHBOARD.md) |

## 6. First checks

1. Gateway is logged in and listening on your `IBKR_PORT`.
2. After the first start the databases exist (`data/ait_state.db` is created by the bot). Then run `python scripts/smoke_deploy.py`: it must print `SMOKE PASS`. It needs the real `config.yaml` and `data/ait_state.db`, uses no broker connection, and writes nothing.
3. Telegram: you receive `supervisor started` and `BOT STARTED | Mode: paper`.
4. http://localhost:8503 loads.
5. Models: a fresh checkout has none under `models/`. The bot trains in the background at start and keeps managing positions meanwhile; you can force it with `python run_orchestrator.py --retrain` (Gateway must be up).

## 7. What to expect on a normal day

Pre-market checks and the model retrain, then scans through the session (new positions only when every gate passes; most days place none or few), a fast monitor that manages exits every 30 seconds, then the post-close report, backups and learning. The full timetable is in
[GUIDE.md](GUIDE.md) section 3. You should see a morning digest and an end-of-day reconciliation message on Telegram; a missing digest is itself an alarm ([RUNBOOK.md](RUNBOOK.md) "Daily 2-minute routine").

## 8. Stopping

- Stop new entries only: create an empty file `data/HALT` (exits keep running).
- Stop the system: stop the supervisor (Ctrl-C in its terminal). If you use `keeper_ait.bat`, create `data/KEEPER_PAUSE` first or it relaunches within about 90 seconds.
- Positions stay open at the broker when the bot stops, and the bot decides every exit itself (it places an exit order when a rule fires), so nothing manages them while it is stopped. Restart the bot, or close them yourself after following [RUNBOOK.md](RUNBOOK.md) "Manual intervention procedure".

## 9. Where next

| If you want to... | Read |
|---|---|
| Understand what the bot is doing and why | [GUIDE.md](GUIDE.md) |
| Respond to alerts, deploy, run drills | [RUNBOOK.md](RUNBOOK.md) |
| Understand or change a model | [MODELS.md](MODELS.md) |
| Run research | [RESEARCH_GUIDE.md](RESEARCH_GUIDE.md) |
| Find out what a script does and when to run it | [reference/SCRIPTS.md](reference/SCRIPTS.md) |
| Change the code | [TECHNICAL_GUIDE.md](TECHNICAL_GUIDE.md) |

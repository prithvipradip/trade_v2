#!/usr/bin/env python3
"""W58: historical options vol-surface backfill (via Polygon/Massive).

    python scripts/backfill_option_history.py --dry-run
    python scripts/backfill_option_history.py --max-contracts 20
    python scripts/backfill_option_history.py

Why this exists: W54's EV-comparison model needs historical per-strike,
per-expiry option data to ever be backtested -- nothing in this repo could
previously capture it (the only existing tool, calibrate_option_spreads.py,
snapshots only TODAY's chain). The existing .env POLYGON_API_KEY already
carries free-tier entitlement to 2 years of rolling historical options data
(confirmed live against the real API before this script was written).

What this does, and nothing else:
  1. Loads the underlying's own daily closes from HistoricalDataStore
     (already present -- no separate underlying-price fetch).
  2. Enumerates option contracts for the symbol via Polygon's contracts
     reference endpoint, paginated, rate-limited.
  3. Filters to contracts whose strike was within +/-max-moneyness of the
     underlying's close on ANY day of the requested window (not just on
     the contract's own expiry day -- a contract that started ATM and
     drifted out of band by expiry must still be included).
  4. For each surviving contract NOT already stored (resume-safe -- see
     HistoricalDataStore.option_ticker_has_data's own docstring for why a
     coarse per-contract check is sound), fetches its full daily OHLC
     history in ONE rate-limited call, solves IV + greeks per bar via the
     existing py_vollib pipeline (options_chain.py), and stores everything
     -- raw prices AND solved IV/greeks -- unfiltered, in a new table.

No quality filtering happens here on purpose (an operator decision, see
PLAN.md's W58 entry): raw data is stored as-is; a config-gated liquidity
filter for SURFACE CONSTRUCTION is separate, later work that reads this
table, not built here. The risk-free rate used for the IV solve is today's
static config value (options.risk_free_rate) -- a placeholder; it is stored
per-row (risk_free_rate column) so a later pass can cheaply re-solve IV
once a real historical rate series exists, with no re-fetch needed.

Strictly respects the free tier's 5-calls/minute limit via a uniform sleep
after every single outbound HTTP call (contracts pages and aggregate
fetches alike) -- not the polygon-api-client SDK's own pagination/retry
(neither has a throttle hook that fits this limit), raw HTTP via `requests`
instead, matching exactly what was manually verified against the live API
before this script was written.

KEEPING IT UP TO DATE (W85). Only contracts that have EXPIRED are downloaded --
one call returns a contract's whole life -- so a re-run with today's date adds
whatever expired since the last run, and already-stored contracts are skipped
locally. A date's longer-dated tenors therefore only appear once those contracts
expire: with live DTE 14-30, full coverage trails today by ~30 days and fills in
on each re-run (live, unexpired contracts are deliberately not fetched: that is
one call per live contract per run). Cadence: roughly 1,550 new QQQ contracts a
week, ~5.5h at the default pause.
  - First refresh: the plain command (full window) -- it also retries contracts
    that failed or returned no bars last time.
  - Later refreshes: `--since-last` lists only expiries from the latest stored one
    onward (fast); run the plain command now and then to catch retries.
  - The run is capped at the underlying's last stored COMPLETED close in daily_prices
    -- rows dated today (ET) are ignored, since the retrain can store a partial
    mid-session bar (W85b) -- (logged as underlying_closes_cap): contracts expiring after it are deferred to
    the next run rather than stored with missing tail bars. daily_prices is
    refreshed by the supervisor's daily retrain.

RUNNING THE REAL, MULTI-DAY JOB (~140,000 contracts for QQQ's full 2-year
window at the default +/-30% band -- see PLAN.md's W58 entry for the
measured math). This deliberately does NOT run via launchd: a launchd-
spawned process cannot access this repo at all (PLAN.md, 2026-09-15 "W16b
correction" -- `~/Documents` is macOS-TCC-protected, and a launchd child
inherits no "responsible app" grant, unlike a process launched from an
interactive Terminal/VS Code session). Start it yourself, in a normal
terminal.

ALWAYS RUN FROM THE REPO ROOT (the `trade_v2` directory). `--db-path`'s
default (`data/historical.db`) and the status file's path are both
relative to the CURRENT WORKING DIRECTORY at invocation, not to this
script's own location -- unlike `.env` loading, which resolves correctly
regardless of cwd. Running from elsewhere would silently create/use a
DIFFERENT, empty `data/historical.db` wherever you happened to be, with no
error. `cd` into the repo root first, every time.

    cd /path/to/trade_v2

Foreground (blocks the terminal, stops if you close it):

    python scripts/backfill_option_history.py

Background, survives closing the terminal (recommended for the real run):

    nohup python scripts/backfill_option_history.py \
        > logs/backfill_option_history.log 2>&1 &

Windows: the script itself is expected to run unmodified there (pure
Python, `pathlib` throughout, no POSIX-only calls) -- but this is UNTESTED
on Windows, and `nohup` is bash/POSIX-only. The Windows equivalent for a
terminal-survives-close background run is PowerShell's `Start-Process`,
e.g. `Start-Process python -ArgumentList "scripts\backfill_option_history.py" -RedirectStandardOutput logs\backfill_option_history.log -RedirectStandardError logs\backfill_option_history.err -WindowStyle Hidden` (also run from the repo root, same reasoning as above).

Three ways to check on it while it runs:
  - `cat data/backfill_option_history_status.json` -- structured snapshot
    (status/contracts_done/contracts_total/rows_written/current_ticker/
    last_error/last_updated). If `last_updated` is stale (more than a
    minute or so old) while the process is confirmed alive, something's
    stuck; if `status` reads "crashed", `last_error` says why.
  - `tail -f logs/backfill_option_history.log` -- live per-contract detail.
  - `ps aux | grep backfill_option_history` -- is the process itself alive.

If it's interrupted for any reason (sleep, a closed terminal without
`nohup`, a crash, a reboot) -- just re-run the exact same command. Already-
fetched contracts are skipped (option_ticker_has_data), nothing is lost,
nothing is re-downloaded. Network blips (e.g. WiFi reconnecting after
sleep) are retried internally with backoff before anything is treated as a
real failure -- see PolygonClient._get()'s own docstring.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ait.config.runtime_env import apply_runtime_env_defaults  # noqa: E402

apply_runtime_env_defaults()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from datetime import date, datetime, timedelta, timezone  # noqa: E402

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from ait.data.historical import HistoricalDataStore  # noqa: E402
from ait.data.options_chain import (  # noqa: E402
    _risk_free_rate,
    compute_greeks_from_sigma,
    solve_iv_from_price,
)
from ait.utils.logging import get_logger  # noqa: E402
from ait.utils.time import today_et  # noqa: E402

log = get_logger("scripts.backfill_option_history")

BASE_URL = "https://api.polygon.io"
AUTH_FAILURE_LIMIT = 3
STATUS_FILE = Path("data") / "backfill_option_history_status.json"
# Network-retry backoff (seconds); [0, ...] so the first attempt has no
# extra delay. Patient enough to ride out a post-sleep WiFi reconnection,
# which typically settles within seconds to a couple minutes -- this is
# about surviving a transient blip, not about the 5/min rate-limit pacing
# (that's `pause_secs`, applied only after a response is actually received).
NETWORK_RETRY_BACKOFFS = [0, 5, 10, 20, 40, 60]


# ---------------------------------------------------------------------------
# Pure selection logic (unit-tested in isolation)
# ---------------------------------------------------------------------------

def in_moneyness_band_over_life(
    strike: float,
    underlying_closes: "pd.Series",
    window_start: date,
    expiry_date: date,
    max_moneyness: float,
) -> bool:
    """True if `strike` was within +/-max_moneyness of the underlying's
    close on ANY trading day between window_start and expiry_date
    (inclusive) -- not just on the expiry date. A contract that started
    at-the-money and drifted out of band by expiry must still pass: the
    check is against the observed [min, max] close over the window, not a
    single date's close.
    """
    window = underlying_closes.loc[str(window_start):str(expiry_date)]
    if window.empty:
        return False
    lo = float(window.min()) * (1 - max_moneyness)
    hi = float(window.max()) * (1 + max_moneyness)
    return lo <= strike <= hi


def completed_sessions_only(closes: "pd.Series", today: date) -> "pd.Series":
    """W85b: the underlying's closes strictly BEFORE `today` (ET).

    The daily retrain (a subprocess with no IB client, so Yahoo) upserts daily_prices
    mid-session, so a row dated today can be a partial bar (e.g. 10:25 ET, 10.4M volume).
    Using it would bake a mid-day price into the solved IV of every bar on that date,
    permanently (the resume-skip never revisits a stored contract). Contracts expiring
    today are simply deferred to the next run. Residual risk: a row for an earlier date
    that was also written mid-session and never re-written; the retrain's next run fixes it.
    """
    return closes[closes.index < pd.Timestamp(today)]


# ---------------------------------------------------------------------------
# Rate-limited Polygon HTTP client (raw requests, not the SDK's own
# pagination/retry -- neither has a throttle hook for the 5/min tier)
# ---------------------------------------------------------------------------

class PolygonClient:
    def __init__(self, api_key: str, pause_secs: float, session=None) -> None:
        self._api_key = api_key
        self._pause = pause_secs
        self._session = session or requests.Session()
        self._auth_failures = 0

    def _get(self, url: str, params: dict | None = None) -> dict:
        """One rate-limited, network-retried GET. A `requests.exceptions.
        RequestException` (connection drop, timeout -- e.g. WiFi still
        reconnecting right after a sleep/wake) is retried per
        NETWORK_RETRY_BACKOFFS before giving up and raising RuntimeError;
        this is about surviving a TRANSIENT blip, not the 5/min rate-limit
        pacing (`self._pause`, applied only after a response is actually
        received). A 401 or a per-request NOT_AUTHORIZED body are handled
        separately below -- those are real API responses, not network
        failures, and need different (auth-breaker vs. skip-and-continue)
        handling."""
        params = dict(params or {})
        params["apiKey"] = self._api_key
        resp = None
        last_exc: Exception | None = None
        for attempt, backoff in enumerate(NETWORK_RETRY_BACKOFFS):
            if backoff:
                log.warning("polygon_request_retry", url=url, attempt=attempt,
                           backoff_secs=backoff)
                time.sleep(backoff)
            try:
                resp = self._session.get(url, params=params, timeout=30)
                last_exc = None
                break
            except requests.exceptions.RequestException as e:
                last_exc = e
        if last_exc is not None:
            raise RuntimeError(
                f"Polygon request failed after {len(NETWORK_RETRY_BACKOFFS)} "
                f"attempts: {url}"
            ) from last_exc
        time.sleep(self._pause)
        if resp.status_code == 401:
            self._auth_failures += 1
            log.warning("polygon_auth_failed", status=resp.status_code,
                       failures=self._auth_failures)
            if self._auth_failures >= AUTH_FAILURE_LIMIT:
                raise RuntimeError(
                    "Polygon API key rejected repeatedly (401) -- aborting "
                    "rather than retrying into a dead key."
                )
            return {}
        self._auth_failures = 0
        try:
            data = resp.json()
        except ValueError:
            resp.raise_for_status()
            return {}
        if data.get("status") == "NOT_AUTHORIZED":
            # Entitlement/timeframe issue for THIS specific request (e.g.
            # the rolling 2-year window shifted mid-run) -- log and skip,
            # not a dead-key abort.
            log.warning("polygon_not_authorized_for_request", url=url,
                       message=data.get("message"))
            return {}
        return data

    def list_contracts_pages(self, **params):
        """Yields each page of the contracts-reference response, following
        next_url until exhausted."""
        url = f"{BASE_URL}/v3/reference/options/contracts"
        data = self._get(url, params=params)
        if data:
            yield data
        next_url = data.get("next_url") if data else None
        while next_url:
            data = self._get(next_url)
            if data:
                yield data
            next_url = data.get("next_url") if data else None

    def get_daily_bars(self, option_ticker: str, from_date: date, to_date: date) -> list[dict]:
        url = f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/day/{from_date}/{to_date}"
        data = self._get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000})
        return data.get("results", []) if data else []


# ---------------------------------------------------------------------------
# Status file -- same "small file in data/, cheap to write, cheap to cat"
# spirit as data/bot_heartbeat, with the progress/outcome fields a bare
# timestamp can't carry. Lets an operator running this unattended for
# hours/days tell "still running" from "died silently" from "why it died"
# with one `cat`, no log-scrolling required.
# ---------------------------------------------------------------------------

def write_status(path: Path | None = None, **fields) -> None:
    # Resolved at CALL time, not as a bound default -- a plain
    # `path: Path = STATUS_FILE` default is evaluated once at def time and
    # would silently ignore any later monkeypatch/reassignment of the
    # module-level STATUS_FILE (a real bug caught by this file's own tests).
    path = path or STATUS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(fields)
    payload["last_updated"] = datetime.now().isoformat()
    try:
        path.write_text(json.dumps(payload, indent=2))
    except OSError as e:  # noqa: BLE001 -- a status-file write must never crash the job
        log.warning("status_file_write_failed", error=str(e))


# ---------------------------------------------------------------------------
# Contract enumeration + filtering
# ---------------------------------------------------------------------------

def enumerate_contracts(
    client: PolygonClient,
    symbol: str,
    start_date: date,
    end_date: date,
    max_moneyness: float,
    underlying_closes: "pd.Series",
    max_contracts: int | None = None,
    expired_from: date | None = None,
) -> list[dict]:
    """Return the filtered contract list: [{option_ticker, expiry, right,
    strike}, ...], in-band per in_moneyness_band_over_life.

    `start_date` is the window start for the moneyness band; `expired_from`
    (W85, default `start_date`) only narrows which EXPIRIES are listed."""
    selected: list[dict] = []
    params = {
        "underlying_ticker": symbol,
        "expired": "true",
        "limit": 1000,
        "expiration_date.gte": str(expired_from or start_date),
        "expiration_date.lte": str(end_date),
        "sort": "ticker",
        "order": "asc",
    }
    for page in client.list_contracts_pages(**params):
        for c in page.get("results", []):
            if max_contracts is not None and len(selected) >= max_contracts:
                return selected
            try:
                expiry = date.fromisoformat(c["expiration_date"])
                strike = float(c["strike_price"])
            except (KeyError, ValueError):
                continue
            if in_moneyness_band_over_life(
                strike, underlying_closes, start_date, expiry, max_moneyness
            ):
                selected.append({
                    "option_ticker": c["ticker"],
                    "expiry": c["expiration_date"],
                    "right": "C" if c.get("contract_type") == "call" else "P",
                    "strike": strike,
                })
    return selected


# ---------------------------------------------------------------------------
# Per-contract fetch + IV solve + store
# ---------------------------------------------------------------------------

def process_contract(
    client: PolygonClient,
    store: HistoricalDataStore,
    symbol: str,
    contract: dict,
    underlying_closes: "pd.Series",
    start_date: date,
    end_date: date,
    risk_free_rate: float,
) -> int:
    ticker = contract["option_ticker"]
    if store.option_ticker_has_data(ticker):
        return 0

    expiry = date.fromisoformat(contract["expiry"])
    fetch_to = min(expiry, end_date)
    bars = client.get_daily_bars(ticker, start_date, fetch_to)
    if not bars:
        return 0

    flag = "c" if contract["right"] == "C" else "p"
    rows = []
    for bar in bars:
        bar_date = datetime.fromtimestamp(bar["t"] / 1000, tz=timezone.utc).date()
        u_close = underlying_closes.get(pd.Timestamp(bar_date))
        if u_close is None or u_close != u_close:  # missing or NaN
            continue
        dte = (expiry - bar_date).days
        t = max(dte / 365.0, 0.001)
        iv, outcome = solve_iv_from_price(
            bar.get("c", 0.0), float(u_close), contract["strike"], t,
            risk_free_rate, flag)
        greeks = (
            compute_greeks_from_sigma(
                float(u_close), contract["strike"], t, risk_free_rate, iv, flag)
            if outcome == "solved" else None
        )
        rows.append({
            "option_ticker": ticker, "expiry": contract["expiry"],
            "right": contract["right"], "strike": contract["strike"],
            "date": str(bar_date), "dte": dte,
            "open": bar.get("o"), "high": bar.get("h"),
            "low": bar.get("l"), "close": bar.get("c"),
            "volume": bar.get("v"), "vwap": bar.get("vw"),
            "transactions": bar.get("n"), "underlying_close": float(u_close),
            "iv": iv, "iv_outcome": outcome,
            "delta": greeks[0] if greeks else None,
            "gamma": greeks[1] if greeks else None,
            "theta": greeks[2] if greeks else None,
            "vega": greeks[3] if greeks else None,
            "risk_free_rate": risk_free_rate,
        })
    if not rows:
        return 0
    return store.save_option_daily_bars(symbol, pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="backfill_option_history",
        description="Backfill historical per-contract EOD option bars + "
                    "solved IV/greeks from Polygon/Massive.",
    )
    parser.add_argument("--symbol", default="QQQ", metavar="SYMBOL")
    parser.add_argument("--start-date", default=None, metavar="YYYY-MM-DD",
                       help="Default: 730 days before --end-date.")
    parser.add_argument("--end-date", default=None, metavar="YYYY-MM-DD",
                       help="Default: today.")
    parser.add_argument("--max-moneyness", type=float, default=0.30, metavar="FRAC",
                       help="Max |strike/spot - 1| over the contract's life "
                            "(default: 0.30 = 30%%, matching calibrate_"
                            "option_spreads.py's own default).")
    parser.add_argument("--db-path", default="data/historical.db", metavar="PATH")
    parser.add_argument("--table-prefix", default="", metavar="PREFIX")
    parser.add_argument("--pause-secs", type=float, default=12.5, metavar="SECS",
                       help="Sleep after every outbound HTTP call "
                            "(default: 12.5s, under the 5/min free-tier limit).")
    parser.add_argument("--max-contracts", type=int, default=None, metavar="N",
                       help="Stop after N contracts pass the filter -- a "
                            "smoke-test escape hatch before the full run.")
    parser.add_argument("--since-last", action="store_true",
                       help="W85: list only expiries from the latest one already "
                            "stored (inclusive) onward, instead of the whole window. "
                            "Fast weekly refresh; does NOT retry older contracts that "
                            "failed or returned no bars -- run without it periodically.")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print the request plan without fetching or storing.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Top-level crash boundary: whatever goes wrong, log one clear line,
    mark the status file "crashed" with the reason, and exit non-zero --
    never a bare traceback with no record of why."""
    try:
        return _run(argv)
    except Exception as e:  # noqa: BLE001 -- see docstring
        log.error("backfill_crashed", error=str(e))
        write_status(status="crashed", last_error=str(e))
        return 1


def _run(argv: list[str] | None) -> int:
    args = _parse_args(argv)

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        log.error("no_polygon_api_key")
        return 1

    requested_end = (
        date.fromisoformat(args.end_date) if args.end_date else today_et()
    )
    start_date = (
        date.fromisoformat(args.start_date) if args.start_date
        else requested_end - timedelta(days=730)
    )

    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=args.table_prefix)
    underlying_df = store.load(args.symbol, start_date, requested_end)
    if underlying_df.empty:
        log.error("no_underlying_price_history", symbol=args.symbol,
                 note="daily_prices has no rows for this symbol/window; "
                      "the moneyness filter and IV solve both need it")
        return 1
    underlying_closes = completed_sessions_only(underlying_df["Close"], today_et())
    if underlying_closes.empty:
        log.error("no_completed_underlying_sessions", symbol=args.symbol)
        return 1

    # W85: cap the run at the underlying's own last stored close. A bar with no
    # underlying close is dropped (process_contract), and the coarse resume-skip
    # (option_ticker_has_data) then treats the contract as done forever -- so a
    # contract expiring past the last close would permanently lose its tail bars.
    # Such contracts are deferred to the next run instead, not truncated.
    underlying_last = underlying_closes.index.max().date()
    end_date = min(requested_end, underlying_last)
    if end_date < requested_end:
        log.warning("underlying_closes_cap", symbol=args.symbol,
                   requested_end=str(requested_end), effective_end=str(end_date),
                   note="daily_prices is stale; contracts expiring after "
                        "effective_end are deferred to the next run")

    expired_from = None
    if args.since_last:
        latest = store.get_latest_option_expiry(args.symbol)
        if latest is not None:
            expired_from = max(start_date, latest)

    client = PolygonClient(api_key, args.pause_secs)

    log.info("enumerating_contracts", symbol=args.symbol,
             start=str(start_date), end=str(end_date),
             expired_from=str(expired_from or start_date),
             max_moneyness=args.max_moneyness)
    try:
        contracts = enumerate_contracts(
            client, args.symbol, start_date, end_date, args.max_moneyness,
            underlying_closes, args.max_contracts, expired_from=expired_from)
    except Exception as e:  # noqa: BLE001 -- a multi-page enumeration failing
        # (e.g. a network blip right after sleep/wake) must not crash the
        # whole run with a raw traceback -- log clearly, mark status, stop.
        log.error("backfill_enumeration_failed", error=str(e))
        write_status(status="crashed", symbol=args.symbol,
                    last_error=f"enumeration failed: {e}")
        return 1
    log.info("contracts_selected", count=len(contracts))

    if args.dry_run:
        est_hours = len(contracts) * args.pause_secs / 3600
        print(
            f"[DRY RUN] {len(contracts)} contracts selected after the "
            f"+/-{args.max_moneyness:.0%} moneyness-over-life filter "
            f"(expiries {expired_from or start_date} .. {end_date}). "
            f"Estimated {len(contracts)} aggregate calls, "
            f"~{est_hours:.1f} hours at {args.pause_secs}s/call."
        )
        return 0

    started_at = datetime.now().isoformat()
    write_status(status="running", symbol=args.symbol, started_at=started_at,
                effective_end=str(end_date),
                contracts_total=len(contracts), contracts_done=0,
                rows_written=0, current_ticker=None, last_error=None)

    r = _risk_free_rate()
    total_rows = 0
    for i, contract in enumerate(contracts):
        try:
            rows = process_contract(
                client, store, args.symbol, contract, underlying_closes,
                start_date, end_date, r)
        except Exception as e:  # noqa: BLE001 -- one bad contract must not kill the run
            log.warning("contract_fetch_failed", ticker=contract["option_ticker"],
                       error=str(e))
            write_status(status="running", symbol=args.symbol, started_at=started_at,
                        contracts_total=len(contracts), contracts_done=i + 1,
                        rows_written=total_rows,
                        current_ticker=contract["option_ticker"], last_error=str(e))
            continue
        total_rows += rows
        log.info("contract_processed", ticker=contract["option_ticker"],
                 rows=rows, progress=f"{i + 1}/{len(contracts)}")
        write_status(status="running", symbol=args.symbol, started_at=started_at,
                    contracts_total=len(contracts), contracts_done=i + 1,
                    rows_written=total_rows,
                    current_ticker=contract["option_ticker"], last_error=None)

    log.info("backfill_complete", symbol=args.symbol,
             contracts=len(contracts), rows=total_rows)
    write_status(status="complete", symbol=args.symbol, started_at=started_at,
                effective_end=str(end_date),
                contracts_total=len(contracts), contracts_done=len(contracts),
                rows_written=total_rows, current_ticker=None, last_error=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

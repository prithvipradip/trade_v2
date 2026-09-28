#!/usr/bin/env python3
"""Historical risk-free-rate series -- closes W58's registered follow-up.

    python scripts/backfill_risk_free_rate.py --dry-run
    python scripts/backfill_risk_free_rate.py

Why this exists: every row backfill_option_history.py writes stores a
PLACEHOLDER risk-free rate (the static options.risk_free_rate config
value) -- a deliberate decision, registered at the time: use the
placeholder now, store it per-row, and re-solve later once a real
historical rate series exists. This script builds that series and does
the re-solve, using ONLY already-stored raw prices -- no re-fetch from
Polygon needed.

Uses SOFR (Secured Overnight Financing Rate), the actual modern industry-
standard risk-free reference for derivatives pricing (the post-LIBOR
standard) -- not Treasury CMT rates, which have no good short-end anchor
for 0-2 DTE contracts. Fetches a real 4-point term structure (overnight,
30/90/180-day compounded averages) from FRED's free, keyless CSV endpoint
via the existing MacroDataFetcher (src/ait/data/macro.py -- reused as-is,
no new HTTP/parsing code), and interpolates across it per-row based on
that row's own DTE (see interpolate_rate in src/ait/data/historical.py).

What this does, in two independent phases:
  Phase A -- fetch SOFR/SOFR30DAYAVG/SOFR90DAYAVG/SOFR180DAYAVG from FRED,
    store in the new risk_free_rates table (upserts -- this data SHOULD
    refresh on re-fetch, FRED occasionally revises recent prints, unlike
    the raw option bars which never change once printed).
  Phase B -- for each stored option_daily_bars row (optionally filtered by
    --symbol), look up the real historical rate for that row's own date +
    dte; if it's missing or already matches the stored rate closely
    enough, skip (no pointless rewrite); otherwise re-run the same
    solve_iv_from_price/compute_greeks_from_sigma pipeline the original
    backfill used, against the row's OWN already-stored close/
    underlying_close, and overwrite just the iv/greeks/risk_free_rate
    columns (never the raw price columns -- see
    HistoricalDataStore.update_option_bar_iv's own docstring).

Safe to run while backfill_option_history.py is actively running against
the same database -- SQLite serializes concurrent writers at the file
level; a transient "database is locked" is retried briefly (see
_UPDATE_RETRY_BACKOFFS) rather than treated as fatal. Always run from the
repo root (same reasoning as backfill_option_history.py's own docstring).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ait.config.runtime_env import apply_runtime_env_defaults  # noqa: E402

apply_runtime_env_defaults()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import sqlite3  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402

import pandas as pd  # noqa: E402

from ait.data.historical import HistoricalDataStore, SOFR_TENOR_DAYS  # noqa: E402
from ait.data.macro import MacroDataFetcher  # noqa: E402
from ait.data.options_chain import (  # noqa: E402
    compute_greeks_from_sigma,
    solve_iv_from_price,
)
from ait.utils.logging import get_logger  # noqa: E402

log = get_logger("scripts.backfill_risk_free_rate")

# SOFR history starts 2018-04 -- comfortably before any date the options
# backfill's own rolling 2-year window will ever reach. The fetch itself is
# cheap (4 HTTP calls total, no per-contract rate limit), so there's no
# reason to trim this to match the options backfill's own window exactly.
LOOKBACK_DAYS = 3000

# Local disk contention (a concurrent writer holding the SQLite file lock),
# not a network failure -- a much shorter retry budget than
# backfill_option_history.py's network-retry backoff is appropriate.
_UPDATE_RETRY_BACKOFFS = [0, 1, 2, 5]

_RATE_MATCH_EPSILON = 1e-6


async def fetch_rate_table() -> pd.DataFrame:
    """Phase A: fetch the SOFR term structure from FRED via the existing
    MacroDataFetcher, merged into one DataFrame keyed by date with columns
    matching HistoricalDataStore.save_risk_free_rates' expected shape."""
    fetcher = MacroDataFetcher()
    series_ids = list(SOFR_TENOR_DAYS.keys())  # {"sofr": 1, ...} -> column names
    fred_ids = {
        "sofr": "SOFR", "sofr_30d_avg": "SOFR30DAYAVG",
        "sofr_90d_avg": "SOFR90DAYAVG", "sofr_180d_avg": "SOFR180DAYAVG",
    }
    fetched: dict[str, pd.Series] = {}
    for col in series_ids:
        series = await fetcher.fetch_series(fred_ids[col], lookback_days=LOOKBACK_DAYS)
        if series is not None and not series.empty:
            # FRED publishes these as raw percentages (e.g. 4.12 meaning
            # 4.12%), but solve_iv_from_price's `r` and options.
            # risk_free_rate's own config units are a decimal fraction
            # (0.0412) -- convert HERE, once, so everything downstream
            # (the stored table, get_risk_free_rate, interpolate_rate) is
            # unit-consistent with the rest of this codebase. Caught by
            # this script's own tests before it ever reached real data.
            fetched[col] = series / 100.0
    if not fetched:
        return pd.DataFrame()
    df = pd.DataFrame(fetched)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def _update_with_retry(
    store: HistoricalDataStore, option_ticker: str, date_: str, iv, iv_outcome,
    delta, gamma, theta, vega, risk_free_rate: float,
) -> int:
    last_exc: Exception | None = None
    for backoff in _UPDATE_RETRY_BACKOFFS:
        if backoff:
            time.sleep(backoff)
        try:
            return store.update_option_bar_iv(
                option_ticker, date_, iv, iv_outcome, delta, gamma, theta,
                vega, risk_free_rate)
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower():
                raise
            last_exc = e
            log.warning("db_locked_retry", ticker=option_ticker, date=date_)
    raise RuntimeError(
        f"database stayed locked after {len(_UPDATE_RETRY_BACKOFFS)} attempts"
    ) from last_exc


def resolve_rows(store: HistoricalDataStore, symbol: str | None, dry_run: bool) -> dict:
    """Phase B. Returns a small summary dict for reporting."""
    query = (
        "SELECT underlying, option_ticker, right, strike, date, dte, "
        "close, underlying_close, risk_free_rate FROM option_daily_bars"
    )
    params: list = []
    if symbol:
        query += " WHERE underlying = ?"
        params.append(symbol)

    with sqlite3.connect(store._db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    checked = 0
    resolved = 0
    skipped_no_rate = 0
    skipped_unchanged = 0
    for (underlying, ticker, right, strike, bar_date, dte, close,
         u_close, stored_rate) in rows:
        checked += 1
        if close is None or u_close is None:
            skipped_no_rate += 1
            continue
        as_of = date.fromisoformat(bar_date)
        real_rate = store.get_risk_free_rate(as_of, int(dte))
        if real_rate is None:
            skipped_no_rate += 1
            continue
        if stored_rate is not None and abs(real_rate - stored_rate) < _RATE_MATCH_EPSILON:
            skipped_unchanged += 1
            continue

        if dry_run:
            resolved += 1
            continue

        flag = "c" if right == "C" else "p"
        t = max(int(dte) / 365.0, 0.001)
        iv, outcome = solve_iv_from_price(close, u_close, strike, t, real_rate, flag)
        greeks = (
            compute_greeks_from_sigma(u_close, strike, t, real_rate, iv, flag)
            if outcome == "solved" else None
        )
        _update_with_retry(
            store, ticker, bar_date, iv, outcome,
            greeks[0] if greeks else None, greeks[1] if greeks else None,
            greeks[2] if greeks else None, greeks[3] if greeks else None,
            real_rate)
        resolved += 1

    return {
        "checked": checked, "resolved": resolved,
        "skipped_no_rate": skipped_no_rate, "skipped_unchanged": skipped_unchanged,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="backfill_risk_free_rate",
        description="Fetch the historical SOFR term structure and "
                    "re-solve option_daily_bars' IV/greeks against it.",
    )
    parser.add_argument("--symbol", default=None, metavar="SYMBOL",
                       help="Only re-solve this symbol's rows (default: all).")
    parser.add_argument("--db-path", default="data/historical.db", metavar="PATH")
    parser.add_argument("--table-prefix", default="", metavar="PREFIX")
    parser.add_argument("--dry-run", action="store_true",
                       help="Report how many rows would be re-solved without "
                            "fetching FRED data or writing anything.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=args.table_prefix)

    # Phase A always runs, even under --dry-run: fetching + storing the rate
    # table is cheap and idempotent (an upsert, touches nothing in
    # option_daily_bars), and skipping it under --dry-run would make the
    # dry-run report meaningless -- get_risk_free_rate would always return
    # None with an empty risk_free_rates table, so Phase B could never show
    # what it WOULD resolve. --dry-run instead gates only the
    # (comparatively expensive/consequential) option_daily_bars rewrite.
    log.info("fetching_sofr_term_structure")
    rate_df = asyncio.run(fetch_rate_table())
    if rate_df.empty:
        log.error("no_rate_data_fetched")
        return 1
    saved = store.save_risk_free_rates(rate_df)
    log.info("risk_free_rates_stored", rows=saved)

    log.info("resolving_option_bars", symbol=args.symbol or "ALL")
    summary = resolve_rows(store, args.symbol, args.dry_run)
    log.info("resolve_complete", **summary)

    if args.dry_run:
        print(
            f"[DRY RUN] {summary['checked']} rows checked, "
            f"{summary['resolved']} would be re-solved, "
            f"{summary['skipped_unchanged']} already match, "
            f"{summary['skipped_no_rate']} have no rate data yet."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

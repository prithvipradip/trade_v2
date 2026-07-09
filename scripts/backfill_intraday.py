"""Bulk backfill of 5-min intraday bars from IBKR into SQLite.

Fetches up to 2 years of 5-min bars per symbol using paginated IBKR requests
(max 6 months per request for 5-min bars). Respects IBKR's pacing rule of
60 historical-data requests per 10 minutes per client ID.

Run once to seed the database before walk-forward or diagnostic analysis:

    python scripts/backfill_intraday.py \\
        --symbols SPY QQQ AAPL NVDA MSFT \\
        --years 2 \\
        --bar-size "5 mins"

Re-run at any time to fill gaps without duplicating rows (INSERT OR REPLACE).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="backfill_intraday",
        description="Backfill 5-min intraday bars from IBKR into SQLite (paginated).",
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True, metavar="SYMBOL",
        help="Tickers to backfill (e.g., SPY QQQ AAPL).",
    )
    parser.add_argument(
        "--years", type=float, default=2.0, metavar="N",
        help="How many years back to fetch (default: 2).",
    )
    parser.add_argument(
        "--bar-size", default="5 mins", metavar="BAR_SIZE",
        help='IBKR bar size string (default: "5 mins").',
    )
    parser.add_argument(
        "--db-path", default="data/historical.db", metavar="PATH",
        help="SQLite DB path (default: data/historical.db).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="IBKR TWS/Gateway host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port", type=int, default=4002, help="IBKR port (default: 4002).",
    )
    parser.add_argument(
        "--client-id", type=int, default=90, metavar="ID",
        help="IBKR client ID (default: 90). Must not conflict with the running bot.",
    )
    parser.add_argument(
        "--chunk-months", type=int, default=6, choices=[1, 2, 3, 6],
        help="Months per IBKR request chunk (default: 6, max for 5-min bars).",
    )
    parser.add_argument(
        "--pause-secs", type=float, default=1.0, metavar="SECS",
        help="Pause between requests to respect IBKR pacing (default: 1.0 s).",
    )
    parser.add_argument(
        "--table-prefix", default="", metavar="PREFIX",
        help='Table name prefix (default: ""). Use "test_" to write to test tables.',
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the request plan without executing any IBKR calls.",
    )
    return parser.parse_args(argv)


def _build_chunks(years: float, chunk_months: int) -> list[tuple[datetime, str]]:
    """Return (endDateTime, durationStr) pairs covering `years` back from now.

    Chunks are ordered most-recent first so progress is visible immediately.
    """
    now = datetime.now(tz=timezone.utc)
    total_months = int(years * 12)
    chunks: list[tuple[datetime, str]] = []

    # R5 audit C2: 31-day stride vs calendar-month durations left multi-day
    # holes at every boundary (worst at 6-month chunks: 2-5 days). Use real
    # calendar months.
    from dateutil.relativedelta import relativedelta
    end = now
    remaining = total_months
    while remaining > 0:
        this_chunk = min(chunk_months, remaining)
        duration = f"{this_chunk} M"
        chunks.append((end, duration))
        end = end - relativedelta(months=this_chunk)
        remaining -= this_chunk

    return chunks


async def _backfill_symbol(
    ib,
    symbol: str,
    chunks: list[tuple[datetime, str]],
    bar_size: str,
    store,
    pause_secs: float,
    dry_run: bool,
) -> int:
    """Fetch and store all chunks for one symbol. Returns total rows stored."""
    import pandas as pd
    from ib_insync import Stock, util

    contract = Stock(symbol, "SMART", "USD")
    try:
        qualified_list = await ib.qualifyContractsAsync(contract)
        if not qualified_list:
            print(f"  [{symbol}] ERROR: could not qualify contract")
            return 0
        qualified = qualified_list[0]
    except Exception as e:
        print(f"  [{symbol}] ERROR qualifying: {e}")
        return 0

    total_stored = 0
    for i, (end_dt, duration) in enumerate(chunks, start=1):
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S") + " UTC"
        print(f"  [{symbol}] chunk {i}/{len(chunks)}: {duration} ending {end_str}", end="", flush=True)

        if dry_run:
            print(" [DRY RUN — skipped]")
            continue

        try:
            bars = await ib.reqHistoricalDataAsync(
                qualified,
                endDateTime=end_str,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        if not bars:
            print(" 0 bars (empty response)")
            continue

        df = util.df(bars)
        df = df.rename(columns={
            "date": "Datetime", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df.set_index("Datetime", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        stored = store.save_intraday(symbol, df, interval="5m", source="TRADES")  # A9: tag bar semantics
        total_stored += stored
        print(f" → {len(df)} bars fetched, {stored} rows upserted")

        if i < len(chunks):
            time.sleep(pause_secs)

    return total_stored


async def _main(args: argparse.Namespace) -> int:
    from ib_insync import IB
    from ait.data.historical import HistoricalDataStore

    # Print request plan
    chunks = _build_chunks(args.years, args.chunk_months)
    table_name = f"{args.table_prefix}intraday_prices"
    print(f"Backfill plan: {len(args.symbols)} symbol(s) × {len(chunks)} chunk(s) = "
          f"{len(args.symbols) * len(chunks)} IBKR requests")
    print(f"  bar size: {args.bar_size}, years: {args.years}, "
          f"chunk size: {args.chunk_months} month(s), pause: {args.pause_secs}s, "
          f"table: {table_name}")
    if args.dry_run:
        print("  [DRY RUN — no IBKR calls will be made]\n")
        for sym in args.symbols:
            for i, (end_dt, duration) in enumerate(chunks, start=1):
                print(f"  [{sym}] chunk {i}/{len(chunks)}: {duration} ending "
                      f"{end_dt.strftime('%Y%m%d %H:%M:%S')} UTC")
        return 0

    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=args.table_prefix)

    ib = IB()
    print(f"\nConnecting to IBKR at {args.host}:{args.port} (clientId={args.client_id})...")
    try:
        await ib.connectAsync(args.host, args.port, clientId=args.client_id, timeout=10)
    except Exception as e:
        print(f"ERROR: could not connect to IBKR: {e}")
        return 1

    print(f"Connected. Starting backfill...\n")
    t_start = time.time()

    grand_total = 0
    for sym in args.symbols:
        print(f"\nBackfilling {sym}...")
        n = await _backfill_symbol(
            ib, sym, chunks, args.bar_size, store,
            pause_secs=args.pause_secs, dry_run=args.dry_run,
        )
        print(f"  [{sym}] total rows upserted: {n}")
        grand_total += n

    ib.disconnect()
    elapsed = time.time() - t_start
    print(f"\nDone. {grand_total} total rows upserted across {len(args.symbols)} "
          f"symbol(s) in {elapsed:.1f}s.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())

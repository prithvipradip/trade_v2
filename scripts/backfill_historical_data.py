"""Backfill historical 5-min bars AND daily implied-vol snapshots from IBKR.

Two modes:
  --mode intraday  (default) — paginated 5-min bar backfill into intraday_prices
  --mode iv        — daily OPTION_IMPLIED_VOLATILITY into daily_prices.implied_vol
  --mode both      — run intraday then iv

Usage:
    # Seed QQQ 5-min bars for the last 2 years:
    python scripts/backfill_historical_data.py --symbols QQQ --years 2

    # Backfill daily IV for QQQ:
    python scripts/backfill_historical_data.py --symbols QQQ --mode iv --years 2

    # Both in one pass:
    python scripts/backfill_historical_data.py --symbols QQQ SPY --mode both --years 2

    # Dry-run (no IBKR calls):
    python scripts/backfill_historical_data.py --symbols QQQ --mode both --dry-run

IBKR pacing: 60 historical-data requests per 10 minutes. The default --pause-secs 1.0
keeps well under the limit. Increase to 6+ for larger symbol lists.
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="backfill_historical_data",
        description="Backfill IBKR 5-min bars and/or daily IV into SQLite.",
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True, metavar="SYMBOL",
        help="Tickers to backfill (e.g., QQQ SPY AAPL).",
    )
    parser.add_argument(
        "--mode", default="intraday", choices=["intraday", "iv", "both"],
        help="What to backfill: 5-min bars, daily IV, or both (default: intraday).",
    )
    parser.add_argument(
        "--years", type=float, default=2.0, metavar="N",
        help="How many years back to fetch (default: 2).",
    )
    parser.add_argument(
        "--bar-size", default="5 mins", metavar="BAR_SIZE",
        help='IBKR bar size for intraday mode (default: "5 mins").',
    )
    parser.add_argument(
        "--db-path", default="data/historical.db", metavar="PATH",
        help="SQLite DB path (default: data/historical.db).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="IBKR TWS/Gateway host.",
    )
    parser.add_argument(
        "--port", type=int, default=4002, help="IBKR port (default: 4002).",
    )
    parser.add_argument(
        "--client-id", type=int, default=90, metavar="ID",
        help="IBKR client ID (default: 90 — must not conflict with bot).",
    )
    parser.add_argument(
        "--chunk-months", type=int, default=6, choices=[1, 2, 3, 6],
        help="Months per IBKR request chunk for intraday (default: 6).",
    )
    parser.add_argument(
        "--pause-secs", type=float, default=1.0, metavar="SECS",
        help="Pause between requests to respect IBKR pacing (default: 1.0 s).",
    )
    parser.add_argument(
        "--table-prefix", default="", metavar="PREFIX",
        help='Table name prefix (default: ""). Use "test_" for test tables.',
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print request plan without executing IBKR calls.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Chunk building
# ---------------------------------------------------------------------------

def _build_chunks(years: float, chunk_months: int) -> list[tuple[datetime, str]]:
    """Return (endDateTime, durationStr) pairs covering `years` back from now.

    Ordered most-recent first so partial runs show progress immediately.
    """
    now = datetime.now(tz=timezone.utc)
    total_months = int(years * 12)
    chunks: list[tuple[datetime, str]] = []

    end = now
    remaining = total_months
    while remaining > 0:
        this_chunk = min(chunk_months, remaining)
        duration = f"{this_chunk} M"
        chunks.append((end, duration))
        end = end - timedelta(days=this_chunk * 31)
        remaining -= this_chunk

    return chunks


# ---------------------------------------------------------------------------
# Intraday (5-min) backfill
# ---------------------------------------------------------------------------

async def _backfill_intraday_symbol(
    ib,
    symbol: str,
    chunks: list[tuple[datetime, str]],
    bar_size: str,
    store,
    pause_secs: float,
    dry_run: bool,
) -> int:
    """Fetch 5-min bars for one symbol. Returns total rows stored."""
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
        print(
            f"  [{symbol}] intraday chunk {i}/{len(chunks)}: {duration} "
            f"ending {end_str}",
            end="", flush=True,
        )

        if dry_run:
            print(" [DRY RUN]")
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
            print(" 0 bars")
            continue

        df = util.df(bars)
        df = df.rename(columns={
            "date": "Datetime", "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
        df.set_index("Datetime", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        stored = store.save_intraday(symbol, df, interval="5m")
        total_stored += stored
        print(f" → {len(df)} bars fetched, {stored} rows upserted")

        if i < len(chunks):
            time.sleep(pause_secs)

    return total_stored


# ---------------------------------------------------------------------------
# Daily IV backfill
# ---------------------------------------------------------------------------

async def _backfill_iv_symbol(
    ib,
    symbol: str,
    years: float,
    store,
    pause_secs: float,
    dry_run: bool,
) -> int:
    """Fetch daily OPTION_IMPLIED_VOLATILITY bars and store in daily_prices.implied_vol.

    IBKR returns daily IV bars that correspond to the ATM 30-day IV estimate.
    These are stored as update-only operations — only rows already present in
    daily_prices (from the OHLCV path) get an implied_vol value.

    Returns the number of rows updated.
    """
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

    # IBKR allows up to 1 Y per request for daily IV; chunk by year
    total_years = int(max(1, years))
    all_iv: dict[str, float] = {}

    for year_idx in range(total_years):
        end_dt = datetime.now(tz=timezone.utc) - timedelta(days=year_idx * 365)
        duration = "1 Y"
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S") + " UTC"

        print(
            f"  [{symbol}] iv chunk {year_idx + 1}/{total_years}: {duration} "
            f"ending {end_str}",
            end="", flush=True,
        )

        if dry_run:
            print(" [DRY RUN]")
            continue

        try:
            bars = await ib.reqHistoricalDataAsync(
                qualified,
                endDateTime=end_str,
                durationStr=duration,
                barSizeSetting="1 day",
                whatToShow="OPTION_IMPLIED_VOLATILITY",
                useRTH=True,
                formatDate=1,
            )
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        if not bars:
            print(" 0 bars")
            continue

        df = util.df(bars)
        # The "close" field for OPTION_IMPLIED_VOLATILITY bars is the IV value
        # (expressed as a decimal, e.g. 0.25 = 25%). The "date" field is the bar date.
        if "date" in df.columns and "close" in df.columns:
            for _, row in df.iterrows():
                try:
                    date_str = str(row["date"])[:10]
                    iv_val = float(row["close"])
                    if iv_val > 0:
                        all_iv[date_str] = iv_val
                except (ValueError, TypeError):
                    continue

        print(f" → {len(bars)} bars fetched")

        if year_idx < total_years - 1:
            time.sleep(pause_secs)

    if not all_iv or dry_run:
        return 0

    # Build Series and save
    iv_series = pd.Series(all_iv, name="implied_vol")
    iv_series.index = pd.to_datetime(iv_series.index)
    iv_series = iv_series.sort_index()

    updated = store.save_daily_iv(symbol, iv_series)
    print(f"  [{symbol}] iv: {len(all_iv)} values fetched, {updated} daily_prices rows updated")
    return updated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> int:
    from ib_insync import IB
    from ait.data.historical import HistoricalDataStore

    do_intraday = args.mode in ("intraday", "both")
    do_iv = args.mode in ("iv", "both")

    intraday_chunks = _build_chunks(args.years, args.chunk_months) if do_intraday else []
    total_requests = (
        len(args.symbols) * len(intraday_chunks) * int(do_intraday)
        + len(args.symbols) * int(max(1, args.years)) * int(do_iv)
    )

    print(f"Backfill plan: {len(args.symbols)} symbol(s), mode={args.mode}")
    if do_intraday:
        print(f"  intraday: {len(intraday_chunks)} chunk(s) × {len(args.symbols)} = "
              f"{len(intraday_chunks) * len(args.symbols)} IBKR requests")
    if do_iv:
        print(f"  daily iv: {int(max(1, args.years))} chunk(s) × {len(args.symbols)} = "
              f"{int(max(1, args.years)) * len(args.symbols)} IBKR requests")
    print(f"  total IBKR requests: {total_requests}, pause: {args.pause_secs}s\n")

    if args.dry_run:
        print("[DRY RUN — no IBKR calls will be made]")
        for sym in args.symbols:
            if do_intraday:
                for i, (end_dt, dur) in enumerate(intraday_chunks, 1):
                    print(f"  [{sym}] intraday chunk {i}/{len(intraday_chunks)}: {dur} "
                          f"ending {end_dt.strftime('%Y%m%d %H:%M:%S')} UTC")
            if do_iv:
                total_years = int(max(1, args.years))
                for y in range(total_years):
                    print(f"  [{sym}] iv chunk {y + 1}/{total_years}: 1 Y ending T-{y * 365}d")
        return 0

    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=args.table_prefix)

    ib = IB()
    print(f"Connecting to IBKR at {args.host}:{args.port} (clientId={args.client_id})...")
    try:
        await ib.connectAsync(args.host, args.port, clientId=args.client_id, timeout=10)
    except Exception as e:
        print(f"ERROR: could not connect to IBKR: {e}")
        return 1

    print("Connected. Starting backfill...\n")
    t_start = time.time()
    grand_total = 0

    for sym in args.symbols:
        print(f"\n=== {sym} ===")

        if do_intraday:
            print(f"  Backfilling 5-min bars for {sym}...")
            n = await _backfill_intraday_symbol(
                ib, sym, intraday_chunks, args.bar_size,
                store, pause_secs=args.pause_secs, dry_run=args.dry_run,
            )
            print(f"  [{sym}] intraday total rows upserted: {n}")
            grand_total += n

        if do_iv:
            print(f"  Backfilling daily IV for {sym}...")
            n = await _backfill_iv_symbol(
                ib, sym, args.years,
                store, pause_secs=args.pause_secs, dry_run=args.dry_run,
            )
            grand_total += n

    ib.disconnect()
    elapsed = time.time() - t_start
    print(f"\nDone. {grand_total} total rows written across {len(args.symbols)} "
          f"symbol(s) in {elapsed:.1f}s.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())

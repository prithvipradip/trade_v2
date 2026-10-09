"""Backfill historical 5-min bars AND implied-vol OHLC bars from IBKR.

Modes:
  --mode intraday     (default) — paginated 5-min bar backfill into intraday_prices
  --mode iv           — daily OPTION_IMPLIED_VOLATILITY: close into
                        daily_prices.implied_vol (feeds the live IV-rank path),
                        full OHLC into daily_iv
  --mode iv-intraday  — 5-min OPTION_IMPLIED_VOLATILITY OHLC into intraday_iv
  --mode both         — run intraday then iv
  --mode daily        — (W65) daily TRADES bars with REAL volume into {prefix}daily_prices,
                        source-tagged IB_TRADES_RTH (Yahoo YAHOO_UNADJ only as a whole-symbol
                        fallback). Requires --table-prefix (IB volume is ~0.64-0.71x the
                        stored Polygon/Yahoo volume, so it must never be spliced into the
                        live table; --allow-live-tables overrides explicitly)
  --mode research     — (W65) daily then iv, i.e. the research history in one pass

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
import json
import math
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ait.data.historical import (  # noqa: E402  (after the sys.path setup above)
    SOURCE_IB_OPTION_IV, SOURCE_IB_TRADES_RTH, SOURCE_YAHOO_UNADJ,
)

# W65: a gap larger than this many calendar days in a daily series is a data hole
# (long weekends/holidays reach 4-5 days).
MAX_GAP_DAYS = 7


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
        "--mode", default="intraday",
        choices=["intraday", "iv", "iv-intraday", "both", "daily", "research"],
        help="What to backfill: 5-min bars, daily IV, 5-min IV, both (intraday+iv), "
             "daily TRADES bars (W65), or research (daily+iv) (default: intraday).",
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
        "--what-to-show", default="MIDPOINT",
        choices=["TRADES", "MIDPOINT", "BID", "ASK"],
        help="IBKR whatToShow for intraday bars (default: MIDPOINT). "
             "TRADES requires a market data subscription; MIDPOINT works on paper accounts.",
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
        "--chunk-months", type=int, default=1, choices=[1, 2, 3, 6],
        help="Months per IBKR request chunk for intraday (default: 1). "
             "Smaller chunks are more reliable; 6 can time out on slow connections.",
    )
    parser.add_argument(
        "--pause-secs", type=float, default=10.0, metavar="SECS",
        help="Pause between requests to respect IBKR pacing (default: 10.0 s). "
             "IBKR allows 60 requests per 10 min; 10 s keeps well under the limit.",
    )
    parser.add_argument(
        "--table-prefix", default="", metavar="PREFIX",
        help='Table name prefix (default: ""). Use "test_" for test tables.',
    )
    parser.add_argument(
        "--allow-live-tables", action="store_true",
        help="W65: permit --mode daily/research with an EMPTY --table-prefix (writes IB "
             "daily bars into the LIVE daily_prices table). Off by default.",
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

    # R5 audit C2: stepping back a flat 31 days per month while requesting
    # IBKR calendar-month durations left 1-5 day holes at every chunk
    # boundary (months are shorter than 31 days). Step by real calendar
    # months so consecutive chunks abut exactly.
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


# ---------------------------------------------------------------------------
# Intraday (5-min) backfill
# ---------------------------------------------------------------------------

async def _backfill_intraday_symbol(
    ib,
    symbol: str,
    chunks: list[tuple[datetime, str]],
    bar_size: str,
    what_to_show: str,
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
                whatToShow=what_to_show,
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

        # R5 audit C1: source tag was omitted, so MIDPOINT bars (the default
        # --what-to-show) were stamped 'TRADES' and silently overwrote real
        # trade bars from backfill_intraday.py via INSERT OR REPLACE.
        stored = store.save_intraday(symbol, df, interval="5m",
                                     source=what_to_show)
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
    """Fetch daily OPTION_IMPLIED_VOLATILITY bars.

    IBKR returns these as real OHLC bars (start/high/low/last IV over the
    day), not a single value. The last/close reading is upserted into
    daily_prices.implied_vol (update-only — only rows already present from
    the OHLCV path get a value; this feeds the live IV-rank pipeline, which
    wants one reading per day same as it does for price Close). The full
    OHLC is additionally stored in the dedicated daily_iv table for anyone
    who needs the intraday-of-day IV range.

    Returns the number of daily_prices rows updated.
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
    all_ohlc: dict[str, tuple[float, float, float, float]] = {}

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
        # OPTION_IMPLIED_VOLATILITY bars are OHLC (start/high/low/last IV,
        # decimals e.g. 0.25 = 25%), same as a price bar — not a scalar.
        if {"date", "open", "high", "low", "close"}.issubset(df.columns):
            for _, row in df.iterrows():
                try:
                    date_str = str(row["date"])[:10]
                    o, h, l, c = (float(row["open"]), float(row["high"]),
                                  float(row["low"]), float(row["close"]))
                    if c > 0:
                        all_iv[date_str] = c
                        all_ohlc[date_str] = (o, h, l, c)
                except (ValueError, TypeError):
                    continue

        print(f" → {len(bars)} bars fetched")

        if year_idx < total_years - 1:
            time.sleep(pause_secs)

    if (not all_iv) or dry_run:
        return 0

    # Close-only series feeds the live daily_prices.implied_vol / IV-rank path.
    iv_series = pd.Series(all_iv, name="implied_vol")
    iv_series.index = pd.to_datetime(iv_series.index)
    iv_series = iv_series.sort_index()

    # Full OHLC feeds the dedicated daily_iv table.
    ohlc_df = pd.DataFrame.from_dict(
        all_ohlc, orient="index", columns=["Open", "High", "Low", "Close"],
    )
    ohlc_df.index = pd.to_datetime(ohlc_df.index)
    ohlc_df.sort_index(inplace=True)
    bars_stored = store.save_daily_iv_bars(symbol, ohlc_df, source=SOURCE_IB_OPTION_IV)
    print(f"  [{symbol}] iv: {bars_stored} daily_iv OHLC rows stored")

    updated = store.save_daily_iv(symbol, iv_series, source=SOURCE_IB_OPTION_IV)
    print(f"  [{symbol}] iv: {len(all_iv)} values fetched, {updated} daily_prices rows updated")
    return updated


# ---------------------------------------------------------------------------
# W65: daily TRADES bars (research history) + validation
# ---------------------------------------------------------------------------

def _drop_incomplete_today(df):
    """Drop today's bar unless the regular session has closed (16:05 ET). A bar fetched
    mid-session is partial (partial High/Low/Volume, mid-session Close) and would be
    stored as if it were a finished day."""
    import pandas as pd
    from ait.utils.time import now_et

    now = now_et()
    if df.empty or (now.hour, now.minute) >= (16, 5):
        return df
    return df[df.index < pd.Timestamp(now.date())]


def _ib_bars_to_daily_df(bars):
    """IB BarData list (daily, formatDate=1 -> datetime.date) -> OHLCV frame."""
    import pandas as pd

    rows = []
    for b in bars:
        try:
            rows.append((pd.Timestamp(b.date), float(b.open), float(b.high),
                         float(b.low), float(b.close), float(b.volume)))
        except (TypeError, ValueError):
            continue
    df = pd.DataFrame(rows, columns=["date", "Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return df
    df = df.drop_duplicates("date").set_index("date").sort_index()
    return df


async def _fetch_daily_ib(ib, qualified, years: float, pause_secs: float):
    """One request for the whole span (verified: 16 Y works for SPY/QQQ/IWM); on an
    error or an empty answer fall back to one-year chunks."""
    import pandas as pd

    duration = f"{int(math.ceil(max(1.0, years)))} Y"
    try:
        bars = await ib.reqHistoricalDataAsync(
            qualified, endDateTime="", durationStr=duration, barSizeSetting="1 day",
            whatToShow="TRADES", useRTH=True, formatDate=1)
        if bars:
            return _ib_bars_to_daily_df(bars)
    except Exception as e:  # noqa: BLE001
        print(f" single request failed ({e}); trying 1-year chunks", end="")
    frames = []
    for year_idx in range(int(math.ceil(max(1.0, years)))):
        end_dt = datetime.now(tz=timezone.utc) - timedelta(days=year_idx * 365)
        try:
            bars = await ib.reqHistoricalDataAsync(
                qualified, endDateTime=end_dt.strftime("%Y%m%d %H:%M:%S") + " UTC",
                durationStr="1 Y", barSizeSetting="1 day", whatToShow="TRADES",
                useRTH=True, formatDate=1)
        except Exception:  # noqa: BLE001
            bars = None
        if bars:
            frames.append(_ib_bars_to_daily_df(bars))
        time.sleep(pause_secs)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()[lambda d: ~d.index.duplicated()]


def _fetch_daily_yahoo(symbol: str, years: float):
    """Whole-symbol Yahoo fallback (A2): never spliced with IB rows inside a symbol."""
    import pandas as pd
    import yfinance as yf

    start = (datetime.now() - timedelta(days=int(years * 366) + 7)).date().isoformat()
    ydf = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=False)
    if ydf is None or ydf.empty:
        return pd.DataFrame()
    df = ydf[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


async def _backfill_daily_symbol(
    ib, symbol: str, years: float, store, pause_secs: float, dry_run: bool,
) -> tuple[int, str]:
    """Fetch daily TRADES bars for one symbol. Returns (rows_stored, source_tag).

    IB first; if IB cannot serve the symbol (qualification/request failure or an
    empty/short answer) the WHOLE symbol comes from Yahoo and is tagged
    YAHOO_UNADJ with a warning (its IV stays empty -> the engine's VIX-proxy path).
    """
    from ib_insync import Stock

    print(f"  [{symbol}] daily TRADES {int(math.ceil(max(1.0, years)))} Y", end="", flush=True)
    if dry_run:
        print(" [DRY RUN]")
        return 0, SOURCE_IB_TRADES_RTH

    df = None
    source = SOURCE_IB_TRADES_RTH
    try:
        qualified_list = await ib.qualifyContractsAsync(Stock(symbol, "SMART", "USD"))
        if qualified_list:
            df = await _fetch_daily_ib(ib, qualified_list[0], years, pause_secs)
    except Exception as e:  # noqa: BLE001
        print(f" IB error: {e}", end="")
    expected = int(years * 252 * 0.8)
    if df is None or df.empty or len(df) < expected:
        print(f"\n  [{symbol}] WARNING: IB daily short/unavailable "
              f"({0 if df is None else len(df)} < {expected}); using Yahoo for the whole symbol")
        df = _fetch_daily_yahoo(symbol, years)
        source = SOURCE_YAHOO_UNADJ
    if df is None or df.empty:
        print(f"  [{symbol}] ERROR: no daily bars from IB or Yahoo")
        return 0, source
    df = _drop_incomplete_today(df)
    n = store.save(symbol, df, source=source)
    print(f" -> {n} rows stored [{source}] {df.index[0].date()}..{df.index[-1].date()}")
    return n, source


def _validate_daily(store, symbol: str) -> dict:
    """W65 validation report for one symbol's research daily rows."""
    import pandas as pd

    df = store.load(symbol).dropna(subset=["Close"])
    rep: dict = {"symbol": symbol, "rows": int(len(df)),
                 "source_summary": {k: list(v) for k, v in store.source_summary(symbol).items()}}
    if df.empty:
        rep.update(first=None, last=None, violations=["no rows"])
        return rep
    gaps = df.index.to_series().diff().dt.days
    iv = store.load_daily_iv(symbol, days=int((pd.Timestamp.now() - df.index[0]).days) + 60)
    viol: list[str] = []
    hl = int((df.High < df.Low).sum())
    outside = int(((df.Close > df.High + 1e-9) | (df.Close < df.Low - 1e-9)).sum())
    zero_vol = int((df.Volume <= 0).sum())
    big_gaps = int((gaps > MAX_GAP_DAYS).sum())
    if hl:
        viol.append(f"{hl} rows with high < low")
    if outside:
        viol.append(f"{outside} rows with close outside [low, high]")
    if zero_vol:
        viol.append(f"{zero_vol} zero-volume days")
    if big_gaps:
        viol.append(f"{big_gaps} gaps > {MAX_GAP_DAYS} calendar days")
    rep.update(first=str(df.index[0].date()), last=str(df.index[-1].date()),
               zero_volume_days=zero_vol, high_lt_low=hl, close_outside_range=outside,
               gaps_over_limit=big_gaps, largest_gap_days=int(gaps.max()) if len(gaps) > 1 else 0,
               iv_rows=int(len(iv)),
               iv_min=float(iv.min()) if len(iv) else None,
               iv_max=float(iv.max()) if len(iv) else None,
               violations=viol)
    return rep


def _write_validation_report(reports: list[dict], table_prefix: str) -> Path:
    from ait.utils.time import now_et

    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"research_history_{now_et().strftime('%Y%m%d')}.json"
    path.write_text(json.dumps({"table_prefix": table_prefix, "symbols": reports}, indent=2))
    return path


# ---------------------------------------------------------------------------
# Intraday (5-min) IV backfill
# ---------------------------------------------------------------------------

async def _backfill_intraday_iv_symbol(
    ib,
    symbol: str,
    chunks: list[tuple[datetime, str]],
    store,
    pause_secs: float,
    dry_run: bool,
) -> int:
    """Fetch 5-min OPTION_IMPLIED_VOLATILITY bars for one symbol.

    Same chunking as the OHLCV intraday path (reuses _build_chunks). These
    are real OHLC bars (start/high/low/last IV per 5-min window), stored in
    full via save_intraday_iv_bars into the dedicated intraday_iv table.
    Returns total rows upserted.
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

    total_stored = 0
    for i, (end_dt, duration) in enumerate(chunks, start=1):
        end_str = end_dt.strftime("%Y%m%d %H:%M:%S") + " UTC"
        print(
            f"  [{symbol}] iv-intraday chunk {i}/{len(chunks)}: {duration} "
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
                barSizeSetting="5 mins",
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
        # OPTION_IMPLIED_VOLATILITY bars are OHLC, not a scalar close.
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")[["open", "high", "low", "close"]]
        df.columns = ["Open", "High", "Low", "Close"]
        df = df[df["Close"] > 0]

        stored = store.save_intraday_iv_bars(symbol, df, interval="5m")
        total_stored += stored
        print(f" → {len(bars)} bars fetched, {stored} rows upserted")

        if i < len(chunks):
            time.sleep(pause_secs)

    return total_stored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> int:
    from ib_insync import IB
    from ait.data.historical import HistoricalDataStore

    do_intraday = args.mode in ("intraday", "both")
    do_iv = args.mode in ("iv", "both", "research")
    do_iv_intraday = args.mode == "iv-intraday"
    do_daily = args.mode in ("daily", "research")

    # W65 guard: IB daily volume is ~0.64-0.71x the stored (Polygon/Yahoo) volume, so
    # daily bars must never be spliced into the live table by accident.
    if do_daily and not args.table_prefix and not args.allow_live_tables:
        print("ERROR: --mode daily/research needs a non-empty --table-prefix "
              "(e.g. research_); pass --allow-live-tables to write the LIVE table.")
        return 2

    intraday_chunks = (
        _build_chunks(args.years, args.chunk_months)
        if (do_intraday or do_iv_intraday) else []
    )
    total_requests = (
        len(args.symbols) * len(intraday_chunks) * int(do_intraday or do_iv_intraday)
        + len(args.symbols) * int(max(1, args.years)) * int(do_iv)
        + len(args.symbols) * int(do_daily)          # W65: one request per symbol
    )

    print(f"Backfill plan: {len(args.symbols)} symbol(s), mode={args.mode}")
    if do_intraday:
        print(f"  intraday: {len(intraday_chunks)} chunk(s) × {len(args.symbols)} = "
              f"{len(intraday_chunks) * len(args.symbols)} IBKR requests "
              f"[whatToShow={args.what_to_show}]")
    if do_iv_intraday:
        print(f"  iv-intraday: {len(intraday_chunks)} chunk(s) × {len(args.symbols)} = "
              f"{len(intraday_chunks) * len(args.symbols)} IBKR requests "
              f"[whatToShow=OPTION_IMPLIED_VOLATILITY, bar=5 mins]")
    if do_daily:
        print(f"  daily TRADES: 1 request × {len(args.symbols)} symbol(s) "
              f"({int(math.ceil(max(1.0, args.years)))} Y each) [source={SOURCE_IB_TRADES_RTH}]")
    if do_iv:
        print(f"  daily iv: {int(max(1, args.years))} chunk(s) × {len(args.symbols)} = "
              f"{int(max(1, args.years)) * len(args.symbols)} IBKR requests")
    print(f"  total IBKR requests: {total_requests}, pause: {args.pause_secs}s\n")

    if args.dry_run:
        print("[DRY RUN — no IBKR calls will be made]")
        for sym in args.symbols:
            if do_intraday or do_iv_intraday:
                for i, (end_dt, dur) in enumerate(intraday_chunks, 1):
                    print(f"  [{sym}] {'iv-intraday' if do_iv_intraday else 'intraday'} "
                          f"chunk {i}/{len(intraday_chunks)}: {dur} "
                          f"ending {end_dt.strftime('%Y%m%d %H:%M:%S')} UTC")
            if do_daily:
                print(f"  [{sym}] daily TRADES {int(math.ceil(max(1.0, args.years)))} Y "
                      f"-> {args.table_prefix}daily_prices [DRY RUN]")
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

    # Allow up to 5 minutes per historical data request (default is 60 s, which
    # is too short for a 6-month 5-min bar chunk — IBKR can take 90–180 s to
    # stream ~10,000 bars).
    ib.RequestTimeout = 300

    print("Connected. Starting backfill...\n")
    t_start = time.time()
    grand_total = 0

    for sym in args.symbols:
        print(f"\n=== {sym} ===")

        if do_intraday:
            print(f"  Backfilling 5-min bars for {sym}...")
            n = await _backfill_intraday_symbol(
                ib, sym, intraday_chunks, args.bar_size, args.what_to_show,
                store, pause_secs=args.pause_secs, dry_run=args.dry_run,
            )
            print(f"  [{sym}] intraday total rows upserted: {n}")
            grand_total += n

        if do_iv_intraday:
            print(f"  Backfilling 5-min IV for {sym}...")
            n = await _backfill_intraday_iv_symbol(
                ib, sym, intraday_chunks,
                store, pause_secs=args.pause_secs, dry_run=args.dry_run,
            )
            print(f"  [{sym}] iv-intraday total rows upserted: {n}")
            grand_total += n

        if do_daily:
            print(f"  Backfilling daily TRADES bars for {sym}...")
            n, _src_tag = await _backfill_daily_symbol(
                ib, sym, args.years, store, pause_secs=args.pause_secs, dry_run=args.dry_run,
            )
            grand_total += n
            if do_iv:
                time.sleep(args.pause_secs)

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

    if do_daily or do_iv:
        reports = [_validate_daily(store, sym) for sym in args.symbols]
        path = _write_validation_report(reports, args.table_prefix)
        bad = 0
        print("\nValidation:")
        for r in reports:
            v = r.get("violations") or []
            bad += len(v)
            print(f"  [{r['symbol']}] rows={r['rows']} {r.get('first')}..{r.get('last')} "
                  f"iv_rows={r.get('iv_rows')} sources={r.get('source_summary')} "
                  f"violations={v or 'none'}")
        print(f"  report: {path}")
        if bad:
            print(f"ERROR: {bad} validation violation(s)")
            return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())

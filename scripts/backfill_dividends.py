"""W84: store each symbol's past ex-dividend dates in `{prefix}dividends` (historical.db).

The research engine's ex-dividend entry gate (`backtest.ex_div_entry_gate`, the twin of
live's W51 block) and the W83a range study read this table; the engine never fetches over
the network. Source: yfinance `Ticker.dividends` (ex-date + cash amount per share), the same
source live's `ait.data.dividends` uses, tagged SOURCE_YAHOO_DIVIDENDS with the fetch time.

A non-empty --table-prefix is required (research data; the live tables are never written)
unless --allow-live-tables is passed.

Usage:
    python scripts/backfill_dividends.py --symbols QQQ SPY IWM --table-prefix research_ \\
        [--since 2009-01-01] [--db-path data/historical.db] [--dry-run]

Prints per-year counts and flags ex-date gaps outside the quarterly band (80-100 days) for
the operator to look at; writes reports/dividends_audit_<date>.json. A flagged gap is not an
error (funds do shift ex-dates); an empty fetch is (exit 1).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ait.data.historical import SOURCE_YAHOO_DIVIDENDS, HistoricalDataStore  # noqa: E402

QUARTERLY_GAP_DAYS = (80, 100)   # a gap outside this band is flagged for review, not rejected


def fetch_dividends(symbol: str, since: date) -> list[tuple[date, float]]:
    """(ex_date, amount) rows from yfinance on or after ``since``, ascending (network)."""
    import yfinance as yf

    hist = yf.Ticker(symbol).dividends
    if hist is None or hist.empty:
        return []
    rows = [(ix.date() if hasattr(ix, "date") else ix, float(v)) for ix, v in hist.items()]
    return sorted((d, a) for d, a in rows if d >= since)


def summarize(rows: list[tuple[date, float]]) -> dict:
    """Pure: per-year counts and the gaps outside QUARTERLY_GAP_DAYS."""
    dates = [d for d, _a in rows]
    lo, hi = QUARTERLY_GAP_DAYS
    odd_gaps = [(str(dates[i - 1]), str(dates[i]), (dates[i] - dates[i - 1]).days)
                for i in range(1, len(dates))
                if not lo <= (dates[i] - dates[i - 1]).days <= hi]
    return {"rows": len(rows), "first": str(dates[0]) if dates else None,
            "last": str(dates[-1]) if dates else None,
            "per_year": dict(sorted(Counter(d.year for d in dates).items())),
            "gaps_outside_band": odd_gaps}


def main(argv: list[str] | None = None, fetch=fetch_dividends) -> int:
    ap = argparse.ArgumentParser(description="Store past ex-dividend dates in {prefix}dividends.")
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--since", default="2009-01-01",
                    help="first ex-date kept (the 'live' projection needs the 4 before the study start)")
    ap.add_argument("--db-path", default="data/historical.db")
    ap.add_argument("--table-prefix", default="", metavar="PREFIX")
    ap.add_argument("--allow-live-tables", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-dir", default=str(Path(__file__).resolve().parent.parent / "reports"))
    args = ap.parse_args(argv)

    # W91: the guard that refused an empty --table-prefix is gone: the un-prefixed tables ARE the one shared
    # history (every row carries its source tag), so that is where this writes by default. --allow-live-tables
    # is kept as an accepted no-op so existing commands keep working.
    since = date.fromisoformat(args.since)
    from ait.utils.time import now_et
    fetched_at = now_et().isoformat(timespec="seconds")

    report: dict = {"table_prefix": args.table_prefix, "source": SOURCE_YAHOO_DIVIDENDS,
                    "fetched_at": fetched_at, "since": str(since), "symbols": {}}
    fetched: dict[str, list[tuple[date, float]]] = {}
    failed = []
    for sym in args.symbols:
        rows = fetch(sym, since)
        stats = summarize(rows)
        report["symbols"][sym] = stats
        print(f"  [{sym}] {stats['rows']} ex-dates {stats['first']} .. {stats['last']}; "
              f"per year {stats['per_year']}")
        for a, b, g in stats["gaps_outside_band"]:
            print(f"      gap {g} days ({a} -> {b}) outside {QUARTERLY_GAP_DAYS}: review")
        if not rows:
            failed.append(sym)
        fetched[sym] = rows

    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dividends_audit_{now_et().strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Audit written: {path}")
    if failed:
        print(f"ERROR: no dividends fetched for {failed}; nothing written")
        return 1
    if args.dry_run:
        print("[DRY RUN] nothing written to the database")
        return 0
    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=args.table_prefix)
    for sym, rows in fetched.items():
        n = store.save_dividends(sym, rows, source=SOURCE_YAHOO_DIVIDENDS, fetched_at=fetched_at)
        print(f"Stored {n} rows for {sym} in {args.table_prefix}dividends; "
              f"span now {store.dividends_span(sym)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

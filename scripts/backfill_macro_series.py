"""W116: store the four FRED macro series the models' macro features read, so research sees what live sees.

    python scripts/backfill_macro_series.py [--db-path data/historical.db] [--since 2009-01-01] [--dry-run]

Downloads FRED's public CSV for the 2-year and 10-year Treasury yields, the 10Y-2Y spread and the broad dollar index
(the same four series, from the same URL, as ``ait.data.macro.MacroDataFetcher`` fetches for the live bot and the daily
retrain; no API key) and upserts them into the ``macro_series`` table of the history database, tagged ``FRED_CSV``.
Research joins them with one session of lag. Run it once, then again now and then (the tail must reach the last few
days of any window a study ends in). ``--dry-run`` downloads and reports and writes nothing. The only network access
is ``fred.stlouisfed.org``.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from ait.data.macro import FRED_CSV_URL, FRED_SERIES  # noqa: E402
from ait.data.macro_store import parse_fred_csv, save_macro_series  # noqa: E402


def fetch(series_id: str, timeout: float = 30.0) -> str:
    request = urllib.request.Request(FRED_CSV_URL.format(series_id=series_id), headers={"User-Agent": "ait-bot/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:     # noqa: S310 - a fixed https FRED URL
        return response.read().decode("utf-8")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db-path", default="data/historical.db", help="the history database (default data/historical.db)")
    ap.add_argument("--since", default="2009-01-01", metavar="YYYY-MM-DD", help="first observation to keep")
    ap.add_argument("--table-prefix", default="", help="research table prefix (default: the shared tables)")
    ap.add_argument("--dry-run", action="store_true", help="download and report; write nothing")
    args = ap.parse_args(argv)

    stamp = datetime.now().isoformat(timespec="seconds")
    failures = 0
    for friendly, series_id in FRED_SERIES.items():
        try:
            series = parse_fred_csv(fetch(series_id))
        except Exception as exc:  # noqa: BLE001 - one series failing must not hide the others
            print(f"{series_id} ({friendly}): FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue
        series = series[series.index >= args.since].dropna()
        if series.empty:
            print(f"{series_id} ({friendly}): FAILED no observations since {args.since}", file=sys.stderr)
            failures += 1
            continue
        span = f"{series.index[0].date()}..{series.index[-1].date()}, {len(series)} observations"
        if args.dry_run:
            print(f"{series_id} ({friendly}): {span} (dry run, nothing written)")
        else:
            save_macro_series(args.db_path, series_id, series, stamp, args.table_prefix)
            print(f"{series_id} ({friendly}): {span} -> {args.db_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

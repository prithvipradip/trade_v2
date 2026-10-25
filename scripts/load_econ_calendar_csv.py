"""W65: load the operator's Investing.com economic-calendar CSV into `{prefix}econ_events`.

The CSV (UTC dates/times, UTF-8 with BOM) is normalized, de-duplicated, converted to
US/Eastern, tagged with its provenance (source INVESTING_CSV + file name + SHA-256)
and written through HistoricalDataStore. The five release types the engine's macro
gate uses (FOMC, CPI, NFP, PCE, GDP advance) are derived into `engine_event_type`
by `ait.data.economic_calendar.derive_engine_event_types` (the single definition).

An empty --table-prefix (the default) writes the one shared set of history tables; every row carries its
source tag. --allow-live-tables is an accepted no-op. (A non-empty prefix would create and fill a separate
set of tables: do not pass one unless you mean to.)

Usage:
    python scripts/load_econ_calendar_csv.py \\
        --csv data/investing_econ_calendar_2024-01-01_onward_UTC.csv \\
        [--db-path data/historical.db] [--dry-run]

Also writes reports/econ_calendar_audit_<date>.json: the CSV's five engine event
types compared with the live hardcoded calendar (read-only; fixing the hardcoded
list is the operator's call).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pandas as pd  # noqa: E402

from ait.data.economic_calendar import derive_engine_event_types  # noqa: E402
from ait.data.historical import SOURCE_INVESTING_CSV, HistoricalDataStore  # noqa: E402

REQUIRED_COLUMNS = ("date_utc", "time_utc", "type", "country", "event", "reference_period")


def normalize(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    """CSV -> normalized frame (ET dates, de-duplicated, engine types derived) + stats."""
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    stats: dict = {"rows_read": int(len(raw))}

    df = raw.copy()
    df["time_utc"] = df["time_utc"].fillna("")
    stamp = pd.to_datetime(df["date_utc"] + " " + df["time_utc"].replace("", "00:00"),
                           utc=True, errors="coerce")
    if stamp.isna().any():
        raise ValueError(f"{int(stamp.isna().sum())} rows have an unparseable date/time")
    et = stamp.dt.tz_convert("America/New_York")
    timed = df["time_utc"].ne("")
    # Untimed rows (holidays) keep the CSV date; timed rows take the ET calendar date.
    df["event_date"] = et.dt.strftime("%Y-%m-%d").where(timed, df["date_utc"])
    df["event_time_et"] = et.dt.strftime("%H:%M").where(timed, "")
    df["reference_period"] = df["reference_period"].fillna("")

    key = ["event_date", "event_time_et", "country", "event", "reference_period"]
    before = len(df)
    df = df.drop_duplicates(key).reset_index(drop=True)
    stats["duplicates_dropped"] = int(before - len(df))

    df["engine_event_type"] = derive_engine_event_types(df)
    keep = ["event_date", "event_time_et", "country", "currency", "importance", "event",
            "reference_period", "type", "category", "actual", "forecast", "previous", "unit",
            "engine_event_type"]
    df = df[[c for c in keep if c in df.columns]]
    stats["rows_loaded"] = int(len(df))
    stats["first_date"], stats["last_date"] = str(df.event_date.min()), str(df.event_date.max())
    eng = df[df.engine_event_type.notna()]
    stats["engine_events"] = {
        t: {str(y): int(n) for y, n in
            eng[eng.engine_event_type == t].groupby(eng.event_date.str[:4]).event_date.nunique().items()}
        for t in sorted(eng.engine_event_type.unique())}
    return df, stats


def audit_vs_hardcoded(df: pd.DataFrame) -> dict:
    """CSV engine events vs the live hardcoded calendar, per type, for the years both cover."""
    import ait.data.economic_calendar as ec

    hard = {"FOMC": ec._FOMC_2026 + ec._FOMC_2027, "CPI": ec._CPI_2026 + ec._CPI_2027,
            "NFP": ec._NFP_2026 + ec._NFP_2027, "PCE": ec._PCE_2026 + ec._PCE_2027,
            "GDP": ec._GDP_2026 + ec._GDP_2027}
    out: dict = {}
    for t, hdates in hard.items():
        csv_dates = sorted(set(df[df.engine_event_type == t].event_date))
        hset = {d.isoformat() for d in hdates}
        csv_last = csv_dates[-1] if csv_dates else None
        # compare only the window both sources claim to cover
        lo = min(hset) if hset else None
        hi = csv_last
        in_scope_csv = {d for d in csv_dates if lo and d >= lo}
        in_scope_hard = {d for d in hset if hi and d <= hi}
        out[t] = {
            "csv_dates_in_hardcoded_range": len(in_scope_csv),
            "hardcoded_dates_in_csv_range": len(in_scope_hard),
            "in_both": len(in_scope_csv & in_scope_hard),
            "only_csv": sorted(in_scope_csv - in_scope_hard),
            "only_hardcoded": sorted(in_scope_hard - in_scope_csv),
            "csv_last_date": csv_last,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Load the economic-calendar CSV into econ_events.")
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--db-path", default="data/historical.db")
    ap.add_argument("--table-prefix", default="", metavar="PREFIX")
    ap.add_argument("--allow-live-tables", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # W91: the guard that refused an empty --table-prefix is gone: the un-prefixed tables ARE the one shared
    # history (every row carries its source tag), so that is where this writes by default. --allow-live-tables
    # is kept as an accepted no-op so existing commands keep working.
    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found")
        return 2

    sha = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    df, stats = normalize(args.csv)
    print(f"Read {stats['rows_read']} rows; dropped {stats['duplicates_dropped']} duplicates; "
          f"{stats['rows_loaded']} to load ({stats['first_date']} .. {stats['last_date']}).")
    print("Engine event dates per type/year:")
    for t, per_year in stats["engine_events"].items():
        print(f"  {t}: {per_year}")
    audit = audit_vs_hardcoded(df)

    report = {"csv": str(args.csv), "sha256": sha, "table_prefix": args.table_prefix,
              "stats": stats, "audit_vs_hardcoded_calendar": audit}
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    from ait.utils.time import now_et
    path = out_dir / f"econ_calendar_audit_{now_et().strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Audit vs hardcoded calendar written: {path}")

    if args.dry_run:
        print("[DRY RUN] nothing written to the database")
        return 0
    store = HistoricalDataStore(db_path=Path(args.db_path), table_prefix=args.table_prefix)
    n = store.save_econ_events(df, source=SOURCE_INVESTING_CSV, source_file=args.csv.name,
                               source_sha256=sha)
    span = store.econ_events_span()
    print(f"Stored {n} rows in {args.table_prefix}econ_events; span now {span}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

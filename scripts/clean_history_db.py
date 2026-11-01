#!/usr/bin/env python3
"""W117: list and (reviewably) delete rows of data/historical.db. Nothing else in the code base deletes history.

    python scripts/clean_history_db.py list                                  # rows by table, symbol, source tag
    python scripts/clean_history_db.py delete --table daily_prices --symbol AAPL --source-null
    python scripts/clean_history_db.py delete --table daily_prices --symbol AAPL --source-null --apply
    python scripts/clean_history_db.py delete --table archive_w91_research_daily_prices --all-rows --apply

`list` shows, per table, symbol and `source` tag: rows and first/last date, and flags rows whose tag is missing
(NULL / empty, shown as UNRECORDED), a tag the code never writes (FOREIGN_SOURCE) and the `archive_w91_research_*`
tables. `delete` shows the exact row counts first and deletes only with `--apply`: the whole delete is ONE
transaction, taken after a sqlite `backup` copy of the database in data/backups/ (restore by copying it back).

Filters (all combine with AND):
  --symbol S         the symbol (or `underlying`) column equals S
  --source TAG       the `source` column equals TAG;  --source-null  selects NULL or empty tags
  --before D         rows dated strictly BEFORE D (D itself is kept)
  --after D          rows dated strictly AFTER D (D itself is kept)
Dates are compared as the day part (first 10 characters) of the table's date column, so a `--before 2026-10-06`
keeps every 5-minute bar of 2026-10-06. A delete with no filter at all is refused unless `--all-rows` is given.

Only the registered research-input tables (ait.data.research_inputs.registered_tables()) and the
archive_w91_research_* tables can be named; any other name is refused, and a name is used in SQL only after it
has been matched against sqlite_master.

`--apply` is refused (exit 3) while the bot may be running against the real data/historical.db, and there is no
override: the supervisor's flock on data/orchestrator.lock is held, data/bot_heartbeat is fresh, a bot or
supervisor process is found, or the process table could not be read. Temporary copies of the database are never
checked (`--db-path` and `--report-dir` go before the subcommand). `list` and the dry run open the file
read-only. Exit codes: 0 ok, 1 failure, 2 not found / bad
arguments, 3 refused. Nothing here prints a secret.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ait.data import historical as _historical  # noqa: E402
from ait.data.macro_store import SOURCE_FRED_CSV  # noqa: E402
from ait.data.research_inputs import registered_tables  # noqa: E402
from ait.utils.logging import get_logger  # noqa: E402

log = get_logger("scripts.clean_history_db")

ARCHIVE = "archive_w91_research_"
REAL_DB = ROOT / "data" / "historical.db"
LOCK_PATH = ROOT / "data" / "orchestrator.lock"
HEARTBEAT_PATH = ROOT / "data" / "bot_heartbeat"
# Same threshold as scripts/probe_w101_combo_close.py (the bot touches the heartbeat about every 30 s).
HEARTBEAT_MAX_AGE_S = 900.0
UNRECORDED = _historical.SOURCE_UNRECORDED

# Every tag the code writes; a stored tag outside this set is flagged FOREIGN_SOURCE in `list`.
KNOWN_SOURCES = ({v for k, v in vars(_historical).items() if k.startswith("SOURCE_") and isinstance(v, str)}
                 | {SOURCE_FRED_CSV}) - {UNRECORDED}
# The column that dates a row, by (unprefixed) table; a table not named here is dated by the first of
# _DATE_FALLBACK it has.
DATE_COLUMN = {
    "daily_prices": "date", "daily_iv": "date", "intraday_prices": "datetime", "dividends": "ex_date",
    "earnings_schedule": "filing_date", "econ_events": "event_date", "macro_series": "date",
    "option_daily_bars": "date", "option_leg_spread_params": "calibrated_on",
    "option_leg_spread_band_params": "calibrated_on",
}
_DATE_FALLBACK = ("date", "datetime", "event_date", "ex_date", "filing_date", "calibrated_on", "sample_date")
_SYMBOL_COLUMNS = ("symbol", "underlying")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Refused(Exception):
    """A request the tool will not carry out (exit 3)."""


class BadArgs(Exception):
    """A request that cannot be understood or matches nothing (exit 2)."""


# --------------------------------------------------------------------------------------------- the bot guard
def _scan(**kw) -> list[int] | None:
    from ait.utils.process import scan_pids
    return scan_pids(**kw)


def guard_bot_stopped(lock_path: Path | None = None, hb_path: Path | None = None,
                      now_ts: float | None = None) -> None:
    """Raise Refused unless nothing suggests the bot or supervisor is alive; any doubt refuses.

    The pattern of scripts/probe_w101_combo_close.py::guard_bot_stopped, duplicated here because that module
    imports the broker stack at import time. Differences, all stricter: the heartbeat counts at any hour (not only
    while the market is open), and a lock or heartbeat that cannot be read, a missing fcntl and a failed process
    scan all refuse instead of passing."""
    lock_path = LOCK_PATH if lock_path is None else lock_path
    hb_path = HEARTBEAT_PATH if hb_path is None else hb_path
    try:
        import fcntl
    except ImportError as e:
        raise Refused("fcntl is unavailable on this platform; cannot verify the bot is stopped") from e
    # The supervisor flocks the lock file for its whole life (the pid inside goes stale), so a non-blocking
    # shared flock answers "is a supervisor alive?" exactly.
    if lock_path.exists():
        try:
            fh = open(lock_path, "r")
        except OSError as e:
            raise Refused(f"cannot open {lock_path} to test the supervisor lock: {e}") from e
        try:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as e:
                raise Refused(f"the supervisor holds {lock_path} -- stop the supervisor (and the bot) first") from e
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
    if hb_path.exists():
        try:
            age = (time.time() if now_ts is None else now_ts) - hb_path.stat().st_mtime
        except OSError as e:
            raise Refused(f"cannot read {hb_path}: {e}") from e
        if age < HEARTBEAT_MAX_AGE_S:
            raise Refused(f"{hb_path} was touched {age:.0f}s ago (< {HEARTBEAT_MAX_AGE_S:.0f}s): a bot process "
                          "looks alive -- stop it first")
    for kw in ({"module": "ait.main"}, {"script": "run_orchestrator.py"}):
        pids = _scan(**kw)
        if pids is None:
            raise Refused("the process table could not be read, so a running bot cannot be ruled out")
        if pids:
            raise Refused(f"a bot or supervisor process is running (pids {sorted(pids)}) -- stop it first")


# --------------------------------------------------------------------------------------------- schema helpers
def _tables(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _quote(name: str) -> str:
    """Quote a table name read from sqlite_master. A name from the command line reaches SQL only through
    validate_table, which has already matched it to a plain identifier present in sqlite_master."""
    return '"' + name.replace('"', '""') + '"'


def _cols(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({_quote(table)})")]


def logical_name(table: str) -> str:
    return table[len(ARCHIVE):] if table.startswith(ARCHIVE) else table


def is_allowed(table: str) -> bool:
    return table in registered_tables() or (table.startswith(ARCHIVE) and len(table) > len(ARCHIVE))


def _layout(con: sqlite3.Connection, table: str) -> dict:
    cols = _cols(con, table)
    date_col = DATE_COLUMN.get(logical_name(table))
    if date_col not in cols:
        date_col = next((c for c in _DATE_FALLBACK if c in cols), None)
    return {"cols": cols, "symbol": next((c for c in _SYMBOL_COLUMNS if c in cols), None),
            "source": "source" if "source" in cols else None, "date": date_col}


def validate_table(con: sqlite3.Connection, name: str) -> str:
    """The table name, once it is a plain identifier, on the allow-list and present in this database."""
    if not isinstance(name, str) or not _IDENT.match(name):
        raise Refused(f"unsafe table name {name!r}")
    if not is_allowed(name):
        raise Refused(f"table {name!r} is not deletable here (allowed: {sorted(registered_tables())} and "
                      f"{ARCHIVE}*)")
    if name not in _tables(con):
        raise BadArgs(f"table {name!r} does not exist in this database")
    return name


# --------------------------------------------------------------------------------------------- list
def _flags(table: str, tag: str | None, has_source: bool) -> list[str]:
    out = []
    if table.startswith(ARCHIVE):
        out.append("ARCHIVE_TABLE")
    if has_source:
        if tag is None or tag == "":
            out.append("UNRECORDED")
        elif tag not in KNOWN_SOURCES:
            out.append("FOREIGN_SOURCE")
    return out


def list_tables(con: sqlite3.Connection) -> list[dict]:
    out = []
    for table in sorted(_tables(con)):
        lay = _layout(con, table)
        total = con.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
        entry ={"table": table, "rows": total, "deletable": is_allowed(table), "groups": [],
                 "flags": ["ARCHIVE_TABLE"] if table.startswith(ARCHIVE) else []}
        out.append(entry)
        if not entry["deletable"] or total == 0:
            continue
        sym, src, dcol = lay["symbol"], lay["source"], lay["date"]
        sel = [f'"{sym}"' if sym else "NULL", f'"{src}"' if src else "NULL", "COUNT(*)",
               f'MIN("{dcol}")' if dcol else "NULL", f'MAX("{dcol}")' if dcol else "NULL"]
        grp = [x for x in (f'"{sym}"' if sym else None, f'"{src}"' if src else None) if x]
        sql = f'SELECT {", ".join(sel)} FROM "{table}"' + (f' GROUP BY {", ".join(grp)}' if grp else "")
        for s, tag, n, first, last in con.execute(sql + " ORDER BY 1, 2"):
            flags = _flags(table, tag, bool(src))
            entry["groups"].append({"symbol": s, "source": (UNRECORDED if tag is None else tag) if src else None,
                                    "rows": n, "first": first, "last": last, "flags": flags})
            for f in flags:
                if f not in entry["flags"]:
                    entry["flags"].append(f)
    return out


def print_listing(listing: list[dict]) -> None:
    for t in listing:
        mark = "" if t["deletable"] else "  (not deletable here)"
        flag = f"  [{', '.join(t['flags'])}]" if t["flags"] else ""
        print(f"{t['table']}: {t['rows']} rows{mark}{flag}")
        for g in t["groups"]:
            tag = f"  source={g['source']}" if g["source"] is not None else ""
            sym = g["symbol"] if g["symbol"] is not None else "-"
            fl = f"  <{', '.join(g['flags'])}>" if g["flags"] else ""
            print(f"    {str(sym):12s} rows={g['rows']:>9}  {g['first']} .. {g['last']}{tag}{fl}")


# --------------------------------------------------------------------------------------------- delete
def build_where(lay: dict, args: argparse.Namespace) -> tuple[str, list]:
    """The WHERE clause (column names come from the table's own PRAGMA list) and its parameters."""
    where: list[str] = []
    params: list = []
    if args.symbol is not None:
        if not lay["symbol"]:
            raise BadArgs("this table has no symbol column; --symbol does not apply")
        where.append(f'"{lay["symbol"]}" = ?')
        params.append(args.symbol)
    if args.source is not None or args.source_null:
        if not lay["source"]:
            raise BadArgs("this table has no source column; --source / --source-null do not apply")
        if args.source_null:
            where.append('("source" IS NULL OR "source" = \'\')')
        else:
            where.append('"source" = ?')
            params.append(args.source)
    for flag, op in (("before", "<"), ("after", ">")):
        value = getattr(args, flag)
        if value is None:
            continue
        if not lay["date"]:
            raise BadArgs(f"this table has no date column; --{flag} does not apply")
        where.append(f'substr("{lay["date"]}", 1, 10) {op} ?')
        params.append(value)
    return " AND ".join(where), params


def plan_delete(con: sqlite3.Connection, table: str, lay: dict, where: str, params: list) -> dict:
    """Exact counts for the selection: matched rows, rows in the table, and the breakdown of what matches."""
    clause = f" WHERE {where}" if where else ""
    total = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    matched = con.execute(f'SELECT COUNT(*) FROM "{table}"{clause}', params).fetchone()[0]
    sym, src, dcol = lay["symbol"], lay["source"], lay["date"]
    breakdown = []
    if matched:
        sel = [f'"{sym}"' if sym else "NULL", f'"{src}"' if src else "NULL", "COUNT(*)",
               f'MIN("{dcol}")' if dcol else "NULL", f'MAX("{dcol}")' if dcol else "NULL"]
        grp = [x for x in (f'"{sym}"' if sym else None, f'"{src}"' if src else None) if x]
        sql = f'SELECT {", ".join(sel)} FROM "{table}"{clause}' + (f' GROUP BY {", ".join(grp)}' if grp else "")
        for s, tag, n, first, last in con.execute(sql + " ORDER BY 1, 2", params):
            breakdown.append({"symbol": s, "source": (UNRECORDED if tag is None else tag) if src else None,
                              "rows": n, "first": first, "last": last})
    return {"table_rows": total, "rows_matched": matched, "rows_kept": total - matched, "breakdown": breakdown}


def apply_delete(db: Path, table: str, where: str, params: list, expected: int, backup_dir: Path) -> dict:
    """Back up, then delete the selection in ONE transaction; roll back unless exactly `expected` rows go."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"historical_pre_clean_{datetime.now():%Y%m%d_%H%M%S}.db"
    con = sqlite3.connect(str(db), isolation_level=None)
    try:
        bcon = sqlite3.connect(str(dest))
        try:
            con.backup(bcon)
            kept = bcon.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        finally:
            bcon.close()
        before = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        if kept != before:
            raise RuntimeError(f"backup {dest} holds {kept} rows of {table}, the table has {before}: not deleting")
        clause = f" WHERE {where}" if where else ""
        con.execute("BEGIN IMMEDIATE")
        try:
            now_matched = con.execute(f'SELECT COUNT(*) FROM "{table}"{clause}', params).fetchone()[0]
            if now_matched != expected:
                raise RuntimeError(f"the selection changed since the plan ({now_matched} rows, planned "
                                   f"{expected}): nothing deleted")
            deleted = con.execute(f'DELETE FROM "{table}"{clause}', params).rowcount
            if deleted != expected:
                raise RuntimeError(f"deleted {deleted} rows, planned {expected}: rolled back")
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        after = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    finally:
        con.close()
    return {"backup": str(dest), "rows_deleted": deleted, "table_rows_after": after}


# --------------------------------------------------------------------------------------------- command line
def _iso_date(text: str) -> str:
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not a YYYY-MM-DD date") from e


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="List and delete rows of data/historical.db (W117).")
    ap.add_argument("--db-path", default=str(REAL_DB))
    ap.add_argument("--report-dir", default=str(ROOT / "reports"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="rows by table, symbol and source tag, with flags; read-only")
    d = sub.add_parser("delete", help="delete a selection of rows; a dry run unless --apply")
    d.add_argument("--table", required=True, help="a registered research-input table or archive_w91_research_*")
    d.add_argument("--symbol", help="symbol (or underlying) equals")
    d.add_argument("--source", help="source tag equals (use --source-null for rows with no tag)")
    d.add_argument("--source-null", action="store_true", help="rows whose source is NULL or empty")
    d.add_argument("--before", type=_iso_date, metavar="YYYY-MM-DD", help="rows dated strictly before (exclusive)")
    d.add_argument("--after", type=_iso_date, metavar="YYYY-MM-DD", help="rows dated strictly after (exclusive)")
    d.add_argument("--all-rows", action="store_true", help="required to delete a whole table (no other filter)")
    d.add_argument("--apply", action="store_true", help="back up, then delete in one transaction")
    d.add_argument("--dry-run", action="store_true", help="the default; accepted for explicitness")
    d.add_argument("--backup-dir", default=str(ROOT / "data" / "backups"))
    return ap


def _write_report(report_dir: str, report: dict) -> Path:
    rdir = Path(report_dir)
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / f"clean_history_db_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)


def _run(args: argparse.Namespace) -> int:
    db = Path(args.db_path)
    if not db.is_file():
        print(f"ERROR: {db} not found")
        return 2
    report: dict = {"db": str(db), "at": datetime.now().isoformat(timespec="seconds"), "command": args.cmd}
    if args.cmd == "list":
        con = _ro(db)
        try:
            report["tables"] = list_tables(con)
        finally:
            con.close()
        report["mode"] = "list"
        print_listing(report["tables"])
        print(f"report: {_write_report(args.report_dir, report)}")
        return 0

    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run contradict each other")
        return 2
    if args.source is not None and args.source_null:
        print("ERROR: give --source or --source-null, not both")
        return 2
    if args.source == UNRECORDED:
        print(f"ERROR: {UNRECORDED} is how `list` labels rows with no tag; select them with --source-null")
        return 2
    filters = [args.symbol is not None, args.source is not None, args.source_null,
               args.before is not None, args.after is not None]
    if args.all_rows and any(filters):
        print("ERROR: --all-rows deletes the whole table; do not combine it with another filter")
        return 2
    if not args.all_rows and not any(filters):
        print("REFUSED: an unscoped delete would empty the whole table; add a filter, or pass --all-rows "
              "if that is what you mean")
        return 3
    real = db.resolve() == REAL_DB.resolve()
    if args.apply and real:
        guard_bot_stopped()

    con = _ro(db)
    try:
        table = validate_table(con, args.table)
        lay = _layout(con, table)
        where, params = build_where(lay, args)
        plan = plan_delete(con, table, lay, where, params)
    finally:
        con.close()
    report.update({"mode": "apply" if args.apply else "dry-run", "table": table,
                   "filters": {"symbol": args.symbol, "source": args.source, "source_null": args.source_null,
                               "before": args.before, "after": args.after, "all_rows": args.all_rows},
                   "guard_checked": bool(args.apply and real), **plan})
    print(f"{table}: {plan['table_rows']} rows, {plan['rows_matched']} match, {plan['rows_kept']} kept")
    for g in plan["breakdown"]:
        tag = f"  source={g['source']}" if g["source"] is not None else ""
        print(f"    {str(g['symbol'] if g['symbol'] is not None else '-'):12s} rows={g['rows']:>9}  "
              f"{g['first']} .. {g['last']}{tag}")
    if args.apply and plan["rows_matched"]:
        report.update(apply_delete(db, table, where, params, plan["rows_matched"], Path(args.backup_dir)))
        print(f"backup: {report['backup']}")
        print(f"deleted: {report['rows_deleted']} rows from {table} ({report['table_rows_after']} remain)")
        log.warning("history_rows_deleted", table=table, rows=report["rows_deleted"], backup=report["backup"])
    elif args.apply:
        report["rows_deleted"] = 0
        print("nothing matches: nothing deleted, no backup taken")
    else:
        print("dry run: nothing deleted (pass --apply to back up and delete)")
    print(f"report: {_write_report(args.report_dir, report)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:                                  # argparse: 0 for --help, 2 for bad arguments
        return int(e.code or 0)
    try:
        return _run(args)
    except Refused as e:
        print(f"REFUSED: {e}")
        return 3
    except BadArgs as e:
        print(f"ERROR: {e}")
        return 2
    except (sqlite3.Error, OSError, RuntimeError) as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

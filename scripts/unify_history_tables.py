#!/usr/bin/env python3
"""W91: merge the `research_*` history tables into the one shared set, verify, archive.

    python scripts/unify_history_tables.py                       # dry run: counts + report, writes nothing
    python scripts/unify_history_tables.py --apply               # backup, then merge (additive)
    python scripts/unify_history_tables.py --verify              # read-only proof that nothing was lost
    python scripts/unify_history_tables.py --archive             # research_X -> archive_w91_research_X
    python scripts/unify_history_tables.py --drop-archives       # drop the archive_w91_* tables

Why: `data/historical.db` held the same history twice (un-prefixed "live" tables and `research_*` twins). The
W65 reason for the split (IB volume is ~0.7x the old stored volume) is gone: the live window is IB since W89.
One table per kind of data, every row tagged with its source, the live retrain only appends.

Per-table policy (POLICY below is the single definition):
  * daily_prices      research (IB) rows win on (symbol, date); live-only dates and other symbols are kept;
                      implied_vol / iv_source come from research where it has them, else live is kept.
  * daily_iv          union; research wins on overlap; live-only symbols (the VIX family) kept.
  * intraday_prices   research wins on overlap; live-only bars kept (the disagreement is counted first).
  * dividends, earnings_schedule, econ_events, econ_schedule, qqq_holdings
                      copied from research (the live twins are empty).
  * any other research_X with rows and a live twin: INSERT OR IGNORE (live wins); EMPTY research_X are only
    archived/dropped.

The merge is ADDITIVE: `research_*` stays untouched until --archive, so old code keeps working until the new code
is merged and the supervisor restarted (the Sunday weekly_backtest reads research history). --apply refuses while
the supervisor/bot runs against the real data/historical.db unless --allow-running. Nothing here prints a secret.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PREFIX = "research_"
ARCHIVE = "archive_w91_research_"

# table -> (key columns, policy). "research_wins": upsert, research values replace live on a key clash;
# "copy": same upsert (the live twin is empty); columns listed in KEEP_LIVE_IF_RESEARCH_NULL keep the live
# value when research has none. Everything else not named here is handled by the generic rule (live wins).
POLICY: dict[str, dict] = {
    "daily_prices": {"key": ["symbol", "date"], "mode": "research_wins",
                     "coalesce": ["implied_vol", "iv_source"],
                     "compare": ["open", "high", "low", "close", "volume", "source"]},
    "daily_iv": {"key": ["symbol", "date"], "mode": "research_wins", "coalesce": ["source"],
                 "compare": ["iv_open", "iv_high", "iv_low", "iv_close"]},
    "intraday_prices": {"key": ["symbol", "datetime", "interval"], "mode": "research_wins", "coalesce": [],
                        "compare": ["open", "high", "low", "close", "volume"]},
    "dividends": {"key": ["symbol", "ex_date"], "mode": "research_wins", "coalesce": [], "compare": None},
    "earnings_schedule": {"key": ["accession"], "mode": "research_wins", "coalesce": [], "compare": None},
    "econ_events": {"key": ["event_date", "event_time_et", "country", "event", "reference_period"],
                    "mode": "research_wins", "coalesce": [], "compare": None},
    "econ_schedule": {"key": ["event_date", "event"], "mode": "research_wins", "coalesce": [], "compare": None},
    "qqq_holdings": {"key": ["accession", "rank"], "mode": "research_wins", "coalesce": [], "compare": None},
}
# A source column outside these sets is a finding in the report (one recorded source per field).
ALLOWED_SOURCES: dict[str, set] = {
    "daily_prices": {None, "", "IB_TRADES_RTH", "YAHOO_UNADJ"},
    "daily_iv": {None, "", "IB_OPTION_IV", "IB_INDEX_TRADES"},
}


def _q(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe identifier {name!r}")
    return name


def _tables(con: sqlite3.Connection) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _cols(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f"PRAGMA table_info({_q(table)})")]


def _count(con: sqlite3.Connection, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {_q(table)}").fetchone()[0]


def _pairs(con: sqlite3.Connection) -> list[tuple[str, str]]:
    """(research table, shared table) for every research_X present."""
    have = _tables(con)
    return [(t, t[len(PREFIX):]) for t in sorted(have) if t.startswith(PREFIX)]


def _key_join(cols: list[str], a: str = "r", b: str = "l") -> str:
    return " AND ".join(f"{a}.{_q(c)} IS {b}.{_q(c)}" for c in cols)


def plan_table(con: sqlite3.Connection, research: str, shared: str) -> dict:
    """Counts for one pair: research rows, rows missing from the shared table, overlapping rows that differ."""
    out = {"research": research, "shared": shared, "research_rows": _count(con, research), "shared_rows": None,
           "to_insert": 0, "overlap": 0, "overlap_differs": 0, "policy": "generic (live wins)"}
    if shared not in _tables(con):
        out["note"] = "no shared twin: the research table is only archived"
        return out
    out["shared_rows"] = _count(con, shared)
    pol = POLICY.get(shared)
    if pol is None:
        out["to_insert"] = con.execute(
            f"SELECT COUNT(*) FROM {research} r WHERE NOT EXISTS (SELECT 1 FROM {shared} l WHERE "
            f"{_key_join(_pk(con, shared))})").fetchone()[0] if out["research_rows"] else 0
        return out
    out["policy"] = pol["mode"]
    key = pol["key"]
    on = _key_join(key)
    out["to_insert"] = con.execute(
        f"SELECT COUNT(*) FROM {research} r WHERE NOT EXISTS (SELECT 1 FROM {shared} l WHERE {on})").fetchone()[0]
    out["overlap"] = con.execute(
        f"SELECT COUNT(*) FROM {research} r JOIN {shared} l ON {on}").fetchone()[0]
    cmp_cols = pol["compare"] or [c for c in _cols(con, research) if c not in key]
    cond = " OR ".join(f"r.{_q(c)} IS NOT l.{_q(c)}" for c in cmp_cols)
    out["overlap_differs"] = con.execute(
        f"SELECT COUNT(*) FROM {research} r JOIN {shared} l ON {on} WHERE {cond}").fetchone()[0]
    return out


def _pk(con: sqlite3.Connection, table: str) -> list[str]:
    cols = [(r[5], r[1]) for r in con.execute(f"PRAGMA table_info({_q(table)})") if r[5]]
    return [c for _, c in sorted(cols)] or _cols(con, table)


def merge_table(con: sqlite3.Connection, research: str, shared: str) -> int:
    """Apply the policy for one pair in ONE transaction. Returns rows written/considered."""
    if shared not in _tables(con) or _count(con, research) == 0:
        return 0
    cols = [c for c in _cols(con, research) if c in set(_cols(con, shared))]
    clist = ", ".join(_q(c) for c in cols)
    pol = POLICY.get(shared)
    with con:
        if pol is None:
            cur = con.execute(f"INSERT OR IGNORE INTO {shared} ({clist}) SELECT {clist} FROM {research}")
            return cur.rowcount
        key = pol["key"]
        sets = []
        for c in cols:
            if c in key:
                continue
            sets.append(f"{_q(c)}=COALESCE(excluded.{_q(c)}, {shared}.{_q(c)})" if c in pol["coalesce"]
                        else f"{_q(c)}=excluded.{_q(c)}")
        sql = (f"INSERT INTO {shared} ({clist}) SELECT {clist} FROM {research} WHERE true "
               f"ON CONFLICT({', '.join(_q(k) for k in key)}) DO UPDATE SET {', '.join(sets)}")
        return con.execute(sql).rowcount


def closes_sha(con: sqlite3.Connection, table: str, symbol: str, lo: str | None = None,
               hi: str | None = None) -> tuple[int, str]:
    q = f"SELECT date, close FROM {_q(table)} WHERE symbol=?"
    args: list = [symbol]
    if lo is not None:
        q += " AND date >= ?"
        args.append(lo)
    if hi is not None:
        q += " AND date <= ?"
        args.append(hi)
    rows = con.execute(q + " ORDER BY date", args).fetchall()
    return len(rows), hashlib.sha256("\n".join(f"{d}|{c}" for d, c in rows).encode()).hexdigest()


def verify(con: sqlite3.Connection) -> dict:
    """Read-only proof: every research row is in the shared table (values equal except the policy columns),
    the close series hash matches per symbol, and source columns hold only allowed tags."""
    report: dict = {"tables": [], "closes": {}, "sources": {}, "ok": True, "problems": []}
    have = _tables(con)
    for research, shared in _pairs(con):
        if shared not in have or _count(con, research) == 0:
            continue
        pol = POLICY.get(shared)
        key = pol["key"] if pol else _pk(con, shared)
        on = _key_join(key)
        missing = con.execute(
            f"SELECT COUNT(*) FROM {research} r WHERE NOT EXISTS (SELECT 1 FROM {shared} l WHERE {on})").fetchone()[0]
        cmp_cols = (pol["compare"] if pol and pol["compare"] else [c for c in _cols(con, research) if c not in key])
        cmp_cols = [c for c in cmp_cols if c in set(_cols(con, shared))]
        cond = " OR ".join(f"r.{_q(c)} IS NOT l.{_q(c)}" for c in cmp_cols) or "0"
        differs = con.execute(
            f"SELECT COUNT(*) FROM {research} r JOIN {shared} l ON {on} WHERE {cond}").fetchone()[0]
        # research wins on a clash, so for the policy tables a remaining difference is a real problem
        expected = 0 if pol else None
        entry = {"table": shared, "research_rows": _count(con, research), "missing_in_shared": missing,
                 "differs": differs}
        report["tables"].append(entry)
        if missing:
            report["ok"] = False
            report["problems"].append(f"{shared}: {missing} research rows missing")
        if expected is not None and differs:
            report["ok"] = False
            report["problems"].append(f"{shared}: {differs} overlapping rows differ")
    if "research_daily_prices" in have and "daily_prices" in have:
        syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM research_daily_prices ORDER BY 1")]
        for s in syms:
            lo, hi = con.execute("SELECT MIN(date), MAX(date) FROM research_daily_prices WHERE symbol=?",
                                 (s,)).fetchone()
            a = closes_sha(con, "research_daily_prices", s)
            b = closes_sha(con, "daily_prices", s, lo, hi)
            same = a == b
            report["closes"][s] = {"rows": a[0], "span": [lo, hi], "sha256_equal": same}
            if not same:
                report["ok"] = False
                report["problems"].append(f"{s}: close series differs over {lo}..{hi} ({a[0]} vs {b[0]} rows)")
    for t, allowed in ALLOWED_SOURCES.items():
        if t in have:
            found = {r[0] for r in con.execute(f"SELECT DISTINCT source FROM {t}")}
            report["sources"][t] = sorted(map(str, found))
            bad = found - allowed
            if bad:
                report["problems"].append(f"{t}: unexpected source tag(s) {sorted(map(str, bad))}")
    return report


def archive(con: sqlite3.Connection) -> list[str]:
    done = []
    for research, _ in _pairs(con):
        with con:
            con.execute(f"ALTER TABLE {_q(research)} RENAME TO {_q(ARCHIVE + research[len(PREFIX):])}")
        done.append(research)
    return done


def drop_archives(con: sqlite3.Connection) -> list[str]:
    done = []
    for t in sorted(_tables(con)):
        if t.startswith(ARCHIVE):
            with con:
                con.execute(f"DROP TABLE {_q(t)}")
            done.append(t)
    return done


def _supervisor_running() -> bool:
    from ait.utils.process import scan_pids
    for kw in ({"module": "ait.main"}, {"script": "run_orchestrator.py"}):
        pids = scan_pids(**kw)
        if pids:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge research_* history tables into the shared tables (W91).")
    ap.add_argument("--db-path", default=str(ROOT / "data" / "historical.db"))
    ap.add_argument("--apply", action="store_true", help="back up, then merge (additive)")
    ap.add_argument("--verify", action="store_true", help="read-only verification, nothing written")
    ap.add_argument("--archive", action="store_true", help="rename research_X to archive_w91_research_X")
    ap.add_argument("--drop-archives", action="store_true", help="drop the archive_w91_* tables")
    ap.add_argument("--backup-dir", default=str(ROOT / "data" / "backups"))
    ap.add_argument("--report-dir", default=str(ROOT / "reports"))
    ap.add_argument("--allow-running", action="store_true",
                    help="skip the supervisor-running refusal (rehearsal copies are never checked)")
    args = ap.parse_args(argv)

    db = Path(args.db_path)
    if not db.is_file():
        print(f"ERROR: {db} not found")
        return 2
    writes = args.apply or args.archive or args.drop_archives
    real = db.resolve() == (ROOT / "data" / "historical.db").resolve()
    if writes and real and not args.allow_running and _supervisor_running():
        print("REFUSED: the supervisor / bot is running against the real data/historical.db. Stop it, or pass "
              "--allow-running if you are sure (the merge is additive and transactional per table).")
        return 3

    con = sqlite3.connect(str(db) if writes else f"file:{db}?mode=ro", uri=not writes)
    try:
        report: dict = {"db": str(db), "at": datetime.now().isoformat(timespec="seconds"), "mode": "dry-run"}
        if args.drop_archives:
            report["mode"] = "drop-archives"
            report["dropped"] = drop_archives(con)
            print(f"dropped: {report['dropped']}")
        elif args.archive:
            v = verify(con)
            if not v["ok"]:
                print("REFUSED: verification failed, not archiving: " + "; ".join(v["problems"]))
                return 4
            report["mode"] = "archive"
            report["archived"] = archive(con)
            print(f"archived: {report['archived']}")
        else:
            report["plan"] = [plan_table(con, r, s) for r, s in _pairs(con)]
            for p in report["plan"]:
                print(f"  {p['shared']:24s} research={p['research_rows']:>7} shared={p['shared_rows']} "
                      f"insert={p['to_insert']} overlap={p['overlap']} differs={p['overlap_differs']} "
                      f"[{p['policy']}]")
            if args.apply:
                backup = Path(args.backup_dir)
                backup.mkdir(parents=True, exist_ok=True)
                dest = backup / f"historical_pre_w91_{datetime.now():%Y%m%d_%H%M%S}.db"
                bcon = sqlite3.connect(str(dest))
                try:
                    con.backup(bcon)
                finally:
                    bcon.close()
                report["backup"] = str(dest)
                print(f"backup: {dest}")
                report["mode"] = "apply"
                report["written"] = {s: merge_table(con, r, s) for r, s in _pairs(con)}
            if args.apply or args.verify:
                report["verify"] = verify(con)
                v = report["verify"]
                print(f"verify: ok={v['ok']} problems={v['problems']}")
                print("closes sha256 equal: " + str({k: x["sha256_equal"] for k, x in v["closes"].items()}))
                print("sources: " + str(v["sources"]))
            else:
                print("dry run: nothing written (pass --apply; --verify checks an already-merged copy)")
        rdir = Path(args.report_dir)
        rdir.mkdir(parents=True, exist_ok=True)
        path = rdir / f"unify_history_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        print(f"report: {path}")
        if report.get("verify") and not report["verify"]["ok"] and args.apply:
            return 5
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())

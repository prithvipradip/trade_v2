"""Track D: build the official economic calendar (the table live and research both read) into historical.db.

    python scripts/build_official_calendar.py [--apply] [--accept-official] [--since 2024-01-01]
        [--db-path data/historical.db] [--table-prefix PREFIX] [--no-fred] [--offline]
        [--cache-dir data/research_cache/official_calendar] [--report-dir reports]

Writes `{prefix}official_econ_calendar`: one row per (date, type) for FOMC, CPI, NFP, GDP (the ADVANCE /
initial estimate only) and PCE (Personal Income and Outlays), each tagged with its official source, URL,
fetch time and the SHA-256 of the fetched page. Sources: the Fed FOMC calendar pages, the BLS release archive
and schedule pages, the BEA release schedule and past-release pages; with FRED_API_KEY set (environment or
.env, never printed or logged) the FRED release/dates API is fetched too and cross-checked date by date.
Without a key the agency pages alone are used.

DRY RUN BY DEFAULT: nothing is written to the database without --apply. A reconciliation report is written
to reports/ either way (research_calendars_audit-style JSON). It compares the new dates with (a) the stored
Investing.com `econ_events` rows (the validation baseline) and (b) the live hard-coded fallback lists in
economic_calendar.py, and with FRED when a key is set. Agreements load silently; every DISAGREEMENT is listed
and BLOCKS the apply (exit 2) until the operator has read the list and re-runs with --accept-official. A
structural failure (a type with no dates, an implausible yearly count, no future date) blocks always.

With --apply the rows from --since on are REPLACED in one transaction (a release the agency moved must not
leave its old date behind). Raw downloads are cached in --cache-dir (forward pages per day), so a re-run the
same day fetches nothing it already has. Network: federalreserve.gov, bls.gov, bea.gov and (with a key)
api.stlouisfed.org; the contact string from SEC_EDGAR_CONTACT is the User-Agent (BLS refuses anonymous ones).
Never opens the live databases for writing except the history store under --apply.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_research_calendars as bc  # noqa: E402 - the cached Fetcher and the contact lookup are reused
from ait.data import official_calendar_sources as src  # noqa: E402
from ait.data.historical import (  # noqa: E402
    SOURCE_BEA_SCHEDULE, SOURCE_BLS_SCHEDULE, SOURCE_FED_FOMC, SOURCE_FRED_RELEASE_DATES, HistoricalDataStore)
from ait.data.official_calendar import OFFICIAL_EVENT_TYPES  # noqa: E402

BLS_SCHEDULE = "https://www.bls.gov/schedule/news_release"
BLS_ARCHIVE = "https://www.bls.gov/bls/news-release"
BEA = "https://www.bea.gov"
BEA_SCHEDULE_URL = f"{BEA}/news/schedule/full"
FED_CAL = f"{bc.FED}/fomccalendars.htm"
LAST_FED_YEAR_PAGE = bc.LAST_FED_YEAR_PAGE
# yearly counts of a FULL year (the agencies' normal cadence); one cancelled / merged / delayed release a year
# is history (the 2025 shutdown merged the Oct and Nov 2025 PCE and NFP-era releases), two or more is a parse bug
EXPECTED_PER_YEAR = {"FOMC": 8, "CPI": 12, "NFP": 12, "PCE": 12, "GDP": 4}
TOLERANCE = 2


class OfficialFetcher(bc.Fetcher):
    """The W83a cached Fetcher plus: a 404 is an answer (cached as an empty file, never retried), and the
    fetch time of a cached document comes from its file (so the table records when the page was really read)."""

    def get_optional(self, url: str, name: str) -> bytes | None:
        p = self.cache / name
        if p.exists():
            return p.read_bytes() or None
        if self.offline:
            return None
        req = urllib.request.Request(url, headers={"User-Agent": self.ua})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                b = r.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            b = b""
        self.requests += 1
        import time
        time.sleep(self.spacing)
        p.write_bytes(b)
        return b or None

    def fetched_at(self, name: str) -> str:
        from ait.utils.time import ET
        return datetime.fromtimestamp((self.cache / name).stat().st_mtime, ET).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Cand:
    """One candidate official release before it is stored."""

    day: date
    time_et: str
    etype: str
    source: str
    url: str
    sha256: str
    fetched_at: str

    def row(self) -> tuple:
        return (str(self.day), self.time_et, self.etype, self.source, self.url, self.fetched_at, self.sha256)


def _slug(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", url.split("://")[-1]).strip("_")[:150]


class Pages:
    """Fetch + remember (name, bytes) so a row can carry the page's hash and fetch time."""

    def __init__(self, f, today: date) -> None:
        self.f, self.today = f, today

    def doc(self, url: str, forward: bool, optional: bool = False):
        """(bytes, sha256, fetched_at) of a page, or None when optional and absent (404)."""
        name = _slug(url) + (f"_{self.today}" if forward else "") + ".bin"
        raw = self.f.get_optional(url, name) if optional else self.f.get(url, name)
        if raw is None:
            return None
        return raw, hashlib.sha256(raw).hexdigest(), self.f.fetched_at(name)


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", "replace")


# ---------------------------------------------------------------------------------------------
# collection from the agency pages
# ---------------------------------------------------------------------------------------------

def collect_fomc(p: Pages, since: date) -> list[Cand]:
    out: list[Cand] = []
    for y in range(since.year, LAST_FED_YEAR_PAGE + 1):
        url = f"{bc.FED}/fomchistorical{y}.htm"
        raw, sha, at = p.doc(url, forward=False)
        out += [Cand(d, src.FOMC_DECISION_TIME_ET, "FOMC", SOURCE_FED_FOMC, url, sha, at)
                for d in src.parse_fomc_year_page(_text(raw), y)]
    raw, sha, at = p.doc(FED_CAL, forward=True)
    for y, ds in src.parse_fomc_calendars(_text(raw)).items():
        if y > LAST_FED_YEAR_PAGE:
            out += [Cand(d, src.FOMC_DECISION_TIME_ET, "FOMC", SOURCE_FED_FOMC, FED_CAL, sha, at) for d in ds]
    return [c for c in out if c.day >= since]


def collect_bls(p: Pages, since: date) -> list[Cand]:
    out: dict[tuple, Cand] = {}
    for etype, pre in (("CPI", "cpi"), ("NFP", "empsit")):
        url = f"{BLS_ARCHIVE}/{pre}.htm"
        raw, sha, at = p.doc(url, forward=True)
        for d in src.parse_bls_archive(_text(raw), pre):
            if d >= since:
                out[(d, etype)] = Cand(d, "", etype, SOURCE_BLS_SCHEDULE, url, sha, at)
        url = f"{BLS_SCHEDULE}/{pre}.htm"
        raw, sha, at = p.doc(url, forward=True)
        for d, hhmm in src.parse_bls_schedule(_text(raw)):
            if d >= since:      # the schedule row carries the release time, so it replaces the archive row
                out[(d, etype)] = Cand(d, hhmm, etype, SOURCE_BLS_SCHEDULE, url, sha, at)
    return sorted(out.values(), key=lambda c: (c.day, c.etype))


def collect_bea(p: Pages, since: date, today: date) -> tuple[list[Cand], dict]:
    """PCE and the GDP advance estimate: the current year from BEA's full schedule page, earlier years from
    the embargo line of each release page (the slug is not uniform, so every known form is tried)."""
    raw, sha, at = p.doc(BEA_SCHEDULE_URL, forward=True)
    page = _text(raw)
    year = src.bea_schedule_year(page) or today.year
    out: dict[tuple, Cand] = {}
    for rel in src.parse_bea_schedule(page, year):
        etype = src.classify_bea_title(rel.title)
        if etype and rel.day >= since:
            out[(rel.day, etype)] = Cand(rel.day, rel.time_et, etype, SOURCE_BEA_SCHEDULE, BEA_SCHEDULE_URL,
                                         sha, at)
    unresolved: list[str] = []

    def probe(paths: list[str], etype: str, label: str) -> None:
        for path in paths:
            doc = p.doc(BEA + path, forward=False, optional=True)
            if doc is None:
                continue
            rel = src.parse_bea_release_page(_text(doc[0]))
            if rel is None or src.classify_bea_title(rel.title) != etype:
                continue
            if rel.day >= since:
                out.setdefault((rel.day, etype),
                               Cand(rel.day, rel.time_et, etype, SOURCE_BEA_SCHEDULE, BEA + path, doc[1], doc[2]))
            return
        unresolved.append(label)

    # PCE: reference months whose release falls before the schedule page's year
    y, m = since.year - 1, 12
    while (y, m) < (year - 1, 12):
        probe(src.bea_pce_page_paths(y, m), "PCE", f"PCE {y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    # GDP advance: quarters whose advance release falls before the schedule page's year
    for qy in range(since.year - 1, year):
        for q in (1, 2, 3, 4):
            if (qy == since.year - 1 and q < 4) or (qy == year - 1 and q == 4):
                continue
            probe(src.bea_gdp_page_paths(qy, q), "GDP", f"GDP {qy}Q{q}")
    return sorted(out.values(), key=lambda c: (c.day, c.etype)), {"unresolved_pages": unresolved}


# ---------------------------------------------------------------------------------------------
# FRED (optional)
# ---------------------------------------------------------------------------------------------

def _fred_key() -> str | None:
    v = os.environ.get("FRED_API_KEY")
    if not v:
        try:
            from dotenv import dotenv_values
            v = dotenv_values(_root / ".env").get("FRED_API_KEY")
        except Exception:  # noqa: BLE001
            v = None
    return v or None


def collect_fred(f, today: date, key: str, since: date) -> dict[str, list[date]]:
    """{type: release dates >= since} from FRED. The key is only in the request URL; the cache file holds the
    answer, and any failure is re-raised WITHOUT the URL."""
    out = {}
    for etype, rid in src.FRED_RELEASE_IDS.items():
        try:
            name_url = f"{src.FRED_API_BASE}/release?release_id={rid}&file_type=json&api_key={key}"
            got = src.parse_fred_release_name(f.get(name_url, f"fred_release_{rid}_{today}.json"))
            if got != src.FRED_RELEASE_NAMES[etype]:
                raise SystemExit(f"ERROR: FRED release id {rid} is {got!r}, expected "
                                 f"{src.FRED_RELEASE_NAMES[etype]!r} for {etype}: fix FRED_RELEASE_IDS")
            url = (f"{src.FRED_API_BASE}/release/dates?"
                   + src.FRED_RELEASE_DATES_QUERY.format(release_id=rid) + f"&api_key={key}")
            out[etype] = [d for d in src.parse_fred_release_dates(f.get(url, f"fred_dates_{rid}_{today}.json"))
                          if d >= since]
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - never let the key into a traceback
            raise SystemExit(f"ERROR: FRED request for {etype} failed ({type(exc).__name__}); "
                             f"re-run with --no-fred to use the agency pages alone") from None
    return out


# ---------------------------------------------------------------------------------------------
# validation and reconciliation
# ---------------------------------------------------------------------------------------------

def validate(cands: list[Cand], since: date, today: date) -> list[str]:
    """Structural failures: these block every run, --accept-official does not override them."""
    fail = []
    by: dict[str, list[Cand]] = {t: [] for t in OFFICIAL_EVENT_TYPES}
    for c in cands:
        by[c.etype].append(c)
    for t, cs in by.items():
        if not cs:
            fail.append(f"{t}: no dates at all")
            continue
        if max(c.day for c in cs) < today and t != "GDP":
            fail.append(f"{t}: the newest date {max(c.day for c in cs)} is already past (no future release)")
        per_year = Counter(c.day.year for c in cs)
        for y in range(max(since.year, 2010), today.year):         # full years only (the current one is partial)
            if y == since.year and since > date(y, 1, 31):
                continue
            n = per_year.get(y, 0)
            if abs(n - EXPECTED_PER_YEAR[t]) > TOLERANCE:
                fail.append(f"{t} {y}: {n} dates (expected about {EXPECTED_PER_YEAR[t]})")
    return fail


def _investing(db: Path, prefix: str, since: date) -> "dict[str, set[date]] | None":
    """{type: dates} of the stored Investing.com rows (read-only), or None when there is no baseline."""
    if not db.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute(f"SELECT event_date, engine_event_type, event FROM {prefix}econ_events "
                                f"WHERE event_date >= ? AND country='United States'", (str(since),)).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    out: dict[str, set[date]] = {t: set() for t in OFFICIAL_EVENT_TYPES}
    for d, et, ev in rows:
        if et in out:
            out[et].add(date.fromisoformat(d))
        elif str(ev).startswith("U.S. Consumer Price Index (CPI)"):   # W106: a CPI release with no MoM headline
            out["CPI"].add(date.fromisoformat(d))
    return out


def _compare(official: set[date], other: set[date]) -> dict:
    lo = max(min(official), min(other)) if official and other else None
    hi = min(max(official), max(other)) if official and other else None
    if lo is None:
        return {"window": None, "matched": 0, "only_official": [], "only_other": sorted(map(str, other))}
    o = {d for d in official if lo <= d <= hi}
    x = {d for d in other if lo <= d <= hi}
    return {"window": [str(lo), str(hi)], "matched": len(o & x), "only_official": sorted(map(str, o - x)),
            "only_other": sorted(map(str, x - o))}


def reconcile(cands: list[Cand], investing, fred, since: date) -> dict:
    """The disagreement lists. Every entry of an `only_*` list is a disagreement that blocks the apply."""
    from ait.data import economic_calendar as ec
    off: dict[str, set[date]] = {t: set() for t in OFFICIAL_EVENT_TYPES}
    for c in cands:
        off[c.etype].add(c.day)
    live_lists = {"FOMC": ec._FOMC_2026 + ec._FOMC_2027, "CPI": ec._CPI_2026 + ec._CPI_2027,
                  "NFP": ec._NFP_2026 + ec._NFP_2027, "GDP": ec._GDP_2026 + ec._GDP_2027,
                  "PCE": ec._PCE_2026 + ec._PCE_2027}
    out: dict = {"investing": None if investing is None else {}, "live_lists": {}, "fred": None}
    if investing is not None:
        for t in OFFICIAL_EVENT_TYPES:
            out["investing"][t] = _compare(off[t], investing[t])
    for t in OFFICIAL_EVENT_TYPES:
        live = {d for d in live_lists[t] if d >= since}
        horizon = max(off[t]) if off[t] else date.min
        beyond = sorted(str(d) for d in live if d > horizon)               # past the official horizon: unverified
        inside = {d for d in live if d <= horizon}
        lo = min(live) if live else None
        # the fallback list only claims dates up to its own last one: later official dates are not a mismatch
        o = {d for d in off[t] if lo is not None and lo <= d <= max(live)}
        out["live_lists"][t] = {"matched": len(o & inside), "only_official": sorted(map(str, o - inside)),
                                "only_live": sorted(map(str, inside - o)), "beyond_official_horizon": beyond}
    if fred is not None:
        out["fred"] = {}
        for t in (t for t in OFFICIAL_EVENT_TYPES if t in fred):     # FRED has no FOMC release
            fset = set(fred.get(t, []))
            cmp = _compare(off[t], fset)
            if t == "GDP":       # FRED lists every GDP release: it can only confirm the advance date is one of them
                cmp = {"window": cmp["window"], "matched": len(off[t] & fset),
                       "only_official": sorted(str(d) for d in off[t] - fset), "only_other": []}
            # A FRED date the pages lack is INFORMATIONAL, not a disagreement: FRED also lists dates of data
            # revisions (a benchmark or seasonal-factor update, 2025-12-23 Personal Income and Outlays), so a
            # missed release shows up through the Investing.com comparison instead. A page date FRED lacks
            # blocks: a parse or schedule error.
            cmp["fred_only_informational"] = cmp.pop("only_other")
            out["fred"][t] = cmp
    return out


def disagreements(rec: dict) -> list[str]:
    lines = []
    for t, r in (rec["investing"] or {}).items():
        lines += [f"{t}: official {d} is not in Investing.com" for d in r["only_official"]]
        lines += [f"{t}: Investing.com {d} is not in the official calendar" for d in r["only_other"]]
    for t, r in rec["live_lists"].items():
        lines += [f"{t}: official {d} is not in the live fallback list" for d in r["only_official"]]
        lines += [f"{t}: live fallback {d} is not an official date" for d in r["only_live"]]
    for t, r in (rec["fred"] or {}).items():
        lines += [f"{t}: official {d} is not a FRED release date" for d in r["only_official"]]
    return lines


# ---------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None, fetcher=None) -> int:
    ap = argparse.ArgumentParser(description="Build the official economic calendar into historical.db.")
    ap.add_argument("--apply", action="store_true", help="write the table (default: dry run)")
    ap.add_argument("--accept-official", action="store_true",
                    help="apply even though the reconciliation lists disagreements (read them first)")
    ap.add_argument("--since", default="2024-01-01", help="first release date to store (default: the "
                    "research history start; every type starts at the same date)")
    ap.add_argument("--db-path", default=str(_root / "data" / "historical.db"))
    ap.add_argument("--table-prefix", default="", metavar="PREFIX")
    ap.add_argument("--no-fred", action="store_true", help="ignore FRED_API_KEY (agency pages only)")
    ap.add_argument("--offline", action="store_true", help="use only the cache (no network)")
    ap.add_argument("--cache-dir", default=str(_root / "data" / "research_cache" / "official_calendar"))
    ap.add_argument("--report-dir", default=str(_root / "reports"))
    args = ap.parse_args(argv)

    from ait.utils.time import now_et, today_et
    today, since, db = today_et(), date.fromisoformat(args.since), Path(args.db_path)
    f = fetcher or OfficialFetcher(Path(args.cache_dir), "" if args.offline else bc._contact(),
                                   offline=args.offline)
    p = Pages(f, today)
    cands = collect_fomc(p, since) + collect_bls(p, since)
    bea, bea_audit = collect_bea(p, since, today)
    cands += bea
    key = None if args.no_fred else _fred_key()
    fred = collect_fred(f, today, key, since) if key else None
    cands = sorted({(c.day, c.etype): c for c in cands}.values(), key=lambda c: (c.day, c.etype))
    if fred is not None:
        # FRED fills only a type the agency pages gave nothing for (never a date beside page dates)
        for t in ("CPI", "NFP", "PCE"):
            if not any(c.etype == t for c in cands):
                cands += [Cand(d, "", t, SOURCE_FRED_RELEASE_DATES, f"{src.FRED_API_BASE}/release/dates "
                               f"(release_id={src.FRED_RELEASE_IDS[t]})", "", now_et().isoformat(timespec="seconds"))
                          for d in fred[t]]

    failures = validate(cands, since, today)
    rec = reconcile(cands, _investing(db, args.table_prefix, since), fred, since)
    diffs = disagreements(rec)
    per_type = {t: sum(c.etype == t for c in cands) for t in OFFICIAL_EVENT_TYPES}
    report = {"generated": now_et().isoformat(timespec="seconds"), "since": str(since),
              "fred": "used" if fred is not None else ("disabled (--no-fred)" if args.no_fred else
                                                         "not used: FRED_API_KEY is not set"),
              "rows": per_type, "sources": dict(Counter(c.source for c in cands)),
              "structural_failures": failures, "disagreements": diffs, "reconciliation": rec,
              "bea": bea_audit, "requests": f.requests}
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"official_calendar_reconciliation_{now_et().strftime('%Y%m%d')}.json"
    path.write_text(json.dumps(report, indent=2, default=str))

    print(f"official calendar: {sum(per_type.values())} rows since {since} {per_type}; FRED: {report['fred']}")
    for line in bea_audit["unresolved_pages"]:
        print(f"   note: no BEA release page found for {line}")
    for line in failures:
        print(f"   FAIL {line}")
    for line in diffs:
        print(f"   DISAGREE {line}")
    print(f"Report written: {path} ({f.requests} network requests)")
    if failures:
        print("ERROR: structural validation failed; nothing written")
        return 1
    if diffs and not args.accept_official:
        print(f"BLOCKED: {len(diffs)} disagreement(s) above. Read them; if the OFFICIAL date is right, re-run "
              f"with --accept-official" + ("" if args.apply else " --apply"))
        return 2
    if not args.apply:
        print("[DRY RUN] nothing written to the database (use --apply)")
        return 0
    store = HistoricalDataStore(db_path=db, table_prefix=args.table_prefix)
    n = store.save_official_econ_calendar([c.row() for c in cands], replace_from=since)
    print(f"Stored {n} rows in {args.table_prefix}official_econ_calendar"
          + (" (disagreements accepted)" if diffs else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

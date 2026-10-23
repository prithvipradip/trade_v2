"""W83a-1: build the event calendars of the strike-aware range study into historical.db.

    python scripts/build_research_calendars.py --what all --table-prefix research_ [--dry-run]
        [--what econ|holdings|earnings] [--cache-dir data/research_cache/w83a] [--since 2009-01-01]

Three research tables (PLAN.md 2026-10-17, W83a AS PLANNED + amendment 4):

  {prefix}econ_schedule      scheduled FOMC decision days (Fed pages), CPI and NFP release dates
                             (BLS archives), 2009 onward
  {prefix}qqq_holdings       QQQ's top-10 holdings and weights per report (N-30B-2 annual, NPORT-P
                             quarterly), with the FILING date the weights become known
  {prefix}earnings_schedule  every 8-K Item 2.02 release of the companies that were ever in a QQQ
                             top 10, with the ET acceptance time and the first session that can react

Nothing is written unless every validation passes (and never with --dry-run); the audit JSON is
written to reports/ either way. Network: federalreserve.gov, bls.gov and SEC EDGAR (User-Agent from
SEC_EDGAR_CONTACT in the environment / .env; at most ~8 requests a second). Raw downloads are cached
in --cache-dir, so a re-run fetches nothing it already has.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from ait.data import research_calendars as rc  # noqa: E402
from ait.data.historical import (  # noqa: E402
    SOURCE_BLS_ARCHIVE, SOURCE_EDGAR_8K_202, SOURCE_EDGAR_N30B2, SOURCE_EDGAR_NPORT, SOURCE_FED_FOMC,
    HistoricalDataStore)

FED = "https://www.federalreserve.gov/monetarypolicy"
BLS = "https://www.bls.gov/bls/news-release"
SEC_DATA = "https://data.sec.gov/submissions"
SEC_ARCH = "https://www.sec.gov/Archives/edgar/data"
QQQ_CIK = 1067839
LAST_FED_YEAR_PAGE = 2020           # 2021 onward live on fomccalendars.htm (the 2021 page is a 404)
FULL_YEARS = range(2010, 2026)      # years whose counts are checked (2026 is partial)


class Fetcher:
    """GET with a cache, a fair-access User-Agent and a request spacing."""

    def __init__(self, cache: Path, user_agent: str, spacing: float = 0.13, offline: bool = False):
        self.cache, self.ua, self.spacing, self.offline = cache, user_agent, spacing, offline
        self.cache.mkdir(parents=True, exist_ok=True)
        self.requests = 0

    def get(self, url: str, name: str) -> bytes:
        p = self.cache / name
        if p.exists():
            return p.read_bytes()
        if self.offline:
            raise FileNotFoundError(f"{name} not cached and --offline is set")
        req = urllib.request.Request(url, headers={"User-Agent": self.ua})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=40) as r:
                    b = r.read()
                break
            except Exception:  # noqa: BLE001 - retried, then raised
                if attempt == 2:
                    raise
                time.sleep(2.0 * (attempt + 1))
        self.requests += 1
        time.sleep(self.spacing)
        p.write_bytes(b)
        return b


def _contact() -> str:
    v = os.environ.get("SEC_EDGAR_CONTACT")
    if not v:
        try:
            from dotenv import dotenv_values
            v = dotenv_values(_root / ".env").get("SEC_EDGAR_CONTACT")
        except Exception:  # noqa: BLE001
            v = None
    if not v or "@" not in v:
        raise SystemExit("ERROR: SEC_EDGAR_CONTACT (a real contact, 'Name email') is not set in the "
                         "environment or .env; the SEC fair-access policy requires it.")
    return v


def _sessions(start: date, end: date) -> list[date]:
    import pandas_market_calendars as mcal
    return [d.date() for d in mcal.get_calendar("NYSE").valid_days(str(start), str(end))]


# ---------------------------------------------------------------------------------------------
# builders: each returns (rows to write, audit dict, list of validation failures)
# ---------------------------------------------------------------------------------------------

def build_econ(f: Fetcher, db: Path, prefix: str, since: date):
    rows, fail = [], []
    fomc: dict[int, list[date]] = {}
    for y in range(since.year, LAST_FED_YEAR_PAGE + 1):
        url = f"{FED}/fomchistorical{y}.htm"
        fomc[y] = rc.parse_fomc_year_page(f.get(url, f"fomc_{y}.htm").decode("utf-8", "replace"), y)
        rows += [(d, "FOMC", url) for d in fomc[y]]
    url = f"{FED}/fomccalendars.htm"
    for y, ds in rc.parse_fomc_calendars(f.get(url, "fomc_calendars.htm").decode("utf-8", "replace")).items():
        if y > LAST_FED_YEAR_PAGE:
            fomc[y] = ds
            rows += [(d, "FOMC", url) for d in ds]
    bls = {}
    for event, pre in (("CPI", "cpi"), ("NFP", "empsit")):
        url = f"{BLS}/{pre}.htm"
        bls[event] = [d for d in rc.parse_bls_archive(f.get(url, f"bls_{pre}.htm").decode("utf-8", "replace"), pre)
                      if d >= since]
        rows += [(d, event, url) for d in bls[event]]
    per_year = {"FOMC": Counter(d.year for d in sum(fomc.values(), []))}
    per_year.update({e: Counter(d.year for d in ds) for e, ds in bls.items()})
    expected = {"FOMC": 8, "CPI": 12, "NFP": 12}
    exceptions = {e: {y: per_year[e].get(y, 0) for y in FULL_YEARS if per_year[e].get(y, 0) != n}
                  for e, n in expected.items()}
    for e, ex in exceptions.items():
        for y, n in ex.items():
            if abs(n - expected[e]) > 1:   # one cancelled / delayed release a year is history, more is a parse bug
                fail.append(f"econ: {e} {y} has {n} dates (expected {expected[e]})")
    # overlap with the operator's Investing.com calendar (research_econ_events, 2024 onward). Matched
    # on the engine tag OR the event name: a W65-untagged row of the same release counts (the
    # 2025-12-18 CPI release has rows but no "CPI MoM" headline row, so W65 left it untagged).
    ref = HistoricalDataStore(db_path=db, table_prefix=prefix).load_econ_events()
    overlap = {}
    names = {"FOMC": r"^Fed Interest Rate Decision$", "CPI": r"^U\.S\. Consumer Price Index \(CPI\)",
             "NFP": r"^U\.S\. Nonfarm Payrolls$"}
    if not ref.empty:
        typed = ref[ref["engine_event_type"].isin(list(names))]
        lo = typed["event_date"].min()
        hi = min(typed["event_date"].max(), str(date.today()))
        for e, pat in names.items():
            mine = {str(d) for d, ev, _u in rows if ev == e and lo <= str(d) <= hi}
            tagged = set(typed.loc[typed["engine_event_type"] == e, "event_date"].astype(str))
            by_name = set(ref.loc[ref["event"].astype(str).str.contains(pat, regex=True), "event_date"].astype(str))
            theirs = {d for d in tagged | by_name if lo <= d <= hi}
            overlap[e] = {"window": [lo, hi], "matched": len(mine & theirs), "matched_on_tag": len(mine & tagged),
                          "only_here": sorted(mine - theirs), "only_investing": sorted(theirs - mine)}
            if mine != theirs:
                fail.append(f"econ: {e} differs from research_econ_events in {lo}..{hi}: "
                            f"only here {sorted(mine - theirs)}, only Investing {sorted(theirs - mine)}")
    else:
        fail.append("econ: research_econ_events is empty: the overlap check could not run")
    audit = {"rows": len(rows), "per_year": {e: dict(sorted(c.items())) for e, c in per_year.items()},
             "exceptions": exceptions, "overlap_vs_research_econ_events": overlap}
    return rows, audit, fail


def _qqq_reports(f: Fetcher, since: date):
    sub = json.loads(f.get(f"{SEC_DATA}/CIK{QQQ_CIK:010d}.json", "qqq_submissions.json"))
    r = sub["filings"]["recent"]
    out = []
    for i, form in enumerate(r["form"]):
        if form not in ("N-30B-2", "NPORT-P"):
            continue
        report = date.fromisoformat(r["reportDate"][i])
        if report < since:
            continue
        acc = r["accessionNumber"][i]
        doc = r["primaryDocument"][i] if form == "N-30B-2" else "primary_doc.xml"
        url = f"{SEC_ARCH}/{QQQ_CIK}/{acc.replace('-', '')}/{doc.split('/')[-1] if form == 'NPORT-P' else doc}"
        raw = f.get(url, f"qqq_{form}_{acc}.{'xml' if form == 'NPORT-P' else 'htm'}")
        out.append((form, report, date.fromisoformat(r["filingDate"][i]), acc, raw))
    return out


def build_holdings(f: Fetcher, since: date):
    rows, fail, reports = [], [], []
    foreign: dict[str, float] = {}
    for form, report, filed, acc, raw in _qqq_reports(f, since):
        if form == "N-30B-2":
            hs = rc.parse_n30b2_schedule(raw.decode("latin-1"))
        else:
            _rp, hs = rc.parse_nport_holdings(raw.decode("utf-8", "replace"))
        top = rc.top_weights(hs)
        total_w = sum(w for *_x, w in top)
        reports.append({"form": form, "report": str(report), "filed": str(filed), "holdings": len(hs),
                        "top10_weight": round(total_w, 4),
                        "top3": [(k or n, round(w, 4)) for _r, k, n, _v, w in top[:3]]})
        if len(hs) < 50:
            fail.append(f"holdings: {form} {report} parsed only {len(hs)} holdings")
        if not 0.25 <= total_w <= 0.75:
            fail.append(f"holdings: {form} {report} top-10 weight {total_w:.3f} outside 25-75%")
        for rank, key, name, value, w in top:
            if key is None:
                fail.append(f"holdings: {form} {report} top-10 name not mapped to a company: {name!r}")
            if key in {k for k, _p in rc.FOREIGN_ISSUERS}:
                foreign[key] = max(foreign.get(key, 0.0), w)
            rows.append((report, filed, form, acc, rank, key, name, value, w))
    annual_years = sorted(date.fromisoformat(r["report"]).year for r in reports if r["form"] == "N-30B-2")
    missing = [y for y in range(since.year, 2026) if y not in annual_years]
    if missing:
        fail.append(f"holdings: no annual report for fiscal years {missing}")
    n_nport = sum(r["form"] == "NPORT-P" for r in reports)
    if n_nport < 28:
        fail.append(f"holdings: only {n_nport} NPORT-P reports (expected 28 as of 2026-10)")
    audit = {"reports": reports, "annual_fiscal_years": annual_years, "nport_reports": n_nport,
             "foreign_issuers_in_top10_max_weight": foreign,
             "company_keys": sorted({r[5] for r in rows if r[5]})}
    return rows, audit, fail


def _submission_pages(f: Fetcher, cik: int) -> list[dict]:
    main = json.loads(f.get(f"{SEC_DATA}/CIK{cik:010d}.json", f"sub_{cik}.json"))
    pages = [main]
    for extra in main.get("filings", {}).get("files", []):
        pages.append(json.loads(f.get(f"{SEC_DATA}/{extra['name']}", f"sub_{extra['name']}")))
    return pages


def _index_accepted(f: Fetcher, cik: int, acc: str):
    page = f.get(f"{SEC_ARCH}/{cik}/{acc.replace('-', '')}/{acc}-index.htm", f"idx_{acc}.htm").decode(
        "utf-8", "replace")
    return rc.parse_index_accepted(page)


def build_earnings(f: Fetcher, holdings_rows: list, since: date):
    rows, fail = [], []
    foreign = {k for k, _p in rc.FOREIGN_ISSUERS}
    keys_years: dict[str, set[int]] = defaultdict(set)
    for report, filed, _form, _acc, _rank, key, *_rest in holdings_rows:
        if key and key not in foreign:
            keys_years[key] |= {filed.year, filed.year + 1}
    sessions = _sessions(since, date(date.today().year + 1, 12, 31))
    per_company, merged_all = {}, {}
    for key in sorted(keys_years):
        filings = []
        for k, cik, _pat, frag in rc.COMPANIES:
            if k != key:
                continue
            pages = _submission_pages(f, cik)
            entity = (pages[0].get("name") or "").lower()
            if frag not in entity:
                fail.append(f"earnings: CIK {cik} is {entity!r}, expected a name containing {frag!r} ({key})")
                continue
            filings += [(cik, x) for x in rc.item_202_filings(pages) if x.filing_date >= since]
        kept, merged = rc.release_events([x for _c, x in filings])
        cik_of = {x.accession: c for c, x in filings}
        merged_all[key] = [(str(x.filing_date), x.accession) for x in merged]
        by_year = Counter(x.filing_date.year for x in kept)
        per_company[key] = {"releases": len(kept), "per_year": dict(sorted(by_year.items())),
                            "top10_years": sorted(keys_years[key])}
        for y in sorted(keys_years[key]):
            if y in FULL_YEARS and not 3 <= by_year.get(y, 0) <= 8:
                fail.append(f"earnings: {key} has {by_year.get(y, 0)} Item 2.02 events in {y}, a year it "
                            f"was in a QQQ top 10 (expected 4, up to 8 with delivery-type reports)")
        for x in kept:
            acc_et = _index_accepted(f, cik_of[x.accession], x.accession)
            if acc_et is None:
                fail.append(f"earnings: no 'Accepted' time on the index page of {x.accession}")
                continue
            # EDGAR dates a filing accepted after 17:30 ET on the next business day: anything else
            # means the acceptance time or the filing date is wrong
            if not (acc_et.date() == x.filing_date or
                    (x.filing_date > acc_et.date() and acc_et.hour * 60 + acc_et.minute >= 17 * 60 + 30)
                    or x.filing_date > acc_et.date() and acc_et.weekday() >= 5):
                fail.append(f"earnings: {x.accession} accepted {acc_et} but filing date {x.filing_date}")
            sess = rc.event_session(acc_et, sessions)
            if sess is not None:
                rows.append((x.accession, key, cik_of[x.accession], x.filing_date, acc_et.isoformat(), sess))
    audit = {"rows": len(rows), "companies": per_company, "merged_within_3_days": merged_all,
             "after_close_share": round(sum(r[4][11:16] >= "16:00" for r in rows) / max(1, len(rows)), 3)}
    return rows, audit, fail


def main(argv: list[str] | None = None, fetcher: Fetcher | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the W83a event calendars into historical.db.")
    ap.add_argument("--what", choices=("econ", "holdings", "earnings", "all"), default="all")
    ap.add_argument("--db-path", default=str(_root / "data" / "historical.db"))
    ap.add_argument("--table-prefix", default="", metavar="PREFIX")
    ap.add_argument("--allow-live-tables", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--since", default="2009-01-01")
    ap.add_argument("--cache-dir", default=str(_root / "data" / "research_cache" / "w83a"))
    ap.add_argument("--report-dir", default=str(_root / "reports"))
    ap.add_argument("--offline", action="store_true", help="use only the cache (no network)")
    args = ap.parse_args(argv)

    # W91: the guard that refused an empty --table-prefix is gone: the un-prefixed tables ARE the one shared
    # history (every row carries its source tag), so that is where this writes by default. --allow-live-tables
    # is kept as an accepted no-op so existing commands keep working.
    since = date.fromisoformat(args.since)
    db = Path(args.db_path)
    f = fetcher or Fetcher(Path(args.cache_dir), "" if args.offline else _contact(), offline=args.offline)
    want = {"econ", "holdings", "earnings"} if args.what == "all" else {args.what}
    if "earnings" in want:
        want.add("holdings")          # the earnings company set comes from the holdings

    results, failures = {}, []
    if "econ" in want:
        results["econ"] = build_econ(f, db, args.table_prefix, since)
    if "holdings" in want:
        results["holdings"] = build_holdings(f, since)
    if "earnings" in want:
        results["earnings"] = build_earnings(f, results["holdings"][0], since)
    for name, (rows, audit, fail) in results.items():
        failures += fail
        print(f"[{name}] {len(rows)} rows; {len(fail)} validation failure(s)")
        for line in fail:
            print(f"   FAIL {line}")

    from ait.utils.time import now_et
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"research_calendars_audit_{now_et().strftime('%Y%m%d')}.json"
    path.write_text(json.dumps({k: v[1] for k, v in results.items()} | {"failures": failures,
                                                                         "requests": f.requests},
                               indent=2, default=str))
    print(f"Audit written: {path} ({f.requests} network requests)")
    if failures:
        print("ERROR: validation failed; nothing written")
        return 1
    if args.dry_run:
        print("[DRY RUN] nothing written to the database")
        return 0
    store = HistoricalDataStore(db_path=db, table_prefix=args.table_prefix)
    if "econ" in results:
        rows = results["econ"][0]
        n = store.save_econ_schedule([r for r in rows if r[1] == "FOMC"], SOURCE_FED_FOMC)
        n += store.save_econ_schedule([r for r in rows if r[1] != "FOMC"], SOURCE_BLS_ARCHIVE)
        print(f"Stored {n} rows in {args.table_prefix}econ_schedule")
    if "holdings" in results and args.what in ("holdings", "all", "earnings"):
        rows = results["holdings"][0]
        n = store.save_qqq_holdings([r for r in rows if r[2] == "N-30B-2"], SOURCE_EDGAR_N30B2)
        n += store.save_qqq_holdings([r for r in rows if r[2] == "NPORT-P"], SOURCE_EDGAR_NPORT)
        print(f"Stored {n} rows in {args.table_prefix}qqq_holdings")
    if "earnings" in results:
        print(f"Stored {store.save_earnings_schedule(results['earnings'][0], SOURCE_EDGAR_8K_202)} rows in "
              f"{args.table_prefix}earnings_schedule")
    return 0


if __name__ == "__main__":
    sys.exit(main())

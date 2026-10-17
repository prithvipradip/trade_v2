"""W83a-1: parsers for the event calendars of the strike-aware range study (research only).

Pure functions over source documents already downloaded (no network, no IO): the build script
(``scripts/build_research_calendars.py``) fetches, these parse, the tests run them on trimmed real
fixtures. The live bot never imports this module (AST-guarded).

Sources (PLAN.md 2026-10-17, W83a AS PLANNED + amendment 4):

* FOMC: the Fed's per-year pages ``fomchistoricalYYYY.htm`` (2009-2020) and ``fomccalendars.htm``
  (2021 onward). SCHEDULED meetings only; the decision day is the meeting's last day. Unscheduled
  meetings, cancelled meetings, conference calls and notation votes are excluded (their dates were not
  public in advance).
* CPI / Employment Situation: the BLS archives, which link every past release as
  ``/news.release/archives/<prefix>_MMDDYYYY.htm``: the release date is in the link.
* QQQ holdings: the trust's N-30B-2 annual Schedule of Investments and its NPORT-P quarterly XML.
* Earnings releases: EDGAR 8-K filings carrying Item 2.02, with the acceptance time.
"""
from __future__ import annotations

import html as _html
import re
import xml.etree.ElementTree as ET
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_MONTHS = {m: i for i, m in enumerate(
    ("january february march april may june july august september october november december").split(), 1)}
_MONTH_ABBR = {k[:3]: v for k, v in _MONTHS.items()}

# ---------------------------------------------------------------------------------------------
# FOMC
# ---------------------------------------------------------------------------------------------

def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", s)).replace("\xa0", " ")).strip()


def _decision_day(text: str, year: int) -> date | None:
    """The LAST day of a meeting, from every heading form the Fed uses: 'January 26-27',
    'March 16', 'July 31-August 1', 'April/May 30-1', 'Jan/Feb 31-1'."""
    t = text.strip()
    m = re.fullmatch(r"([A-Za-z]+)/([A-Za-z]+)\s+(\d{1,2})\s*-\s*(\d{1,2})", t)
    if m:
        month_s, day = m.group(2), int(m.group(4))
    else:
        m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})(?:\s*-\s*(?:([A-Za-z]+)\s+)?(\d{1,2}))?", t)
        if not m:
            return None
        month_s, day = (m.group(3) or m.group(1)), int(m.group(4) or m.group(2))
    month = _MONTH_ABBR.get(month_s[:3].lower())
    return date(year, month, day) if month else None


def parse_fomc_year_page(page: str, year: int) -> list[date]:
    """Scheduled FOMC decision days from a ``fomchistoricalYYYY.htm`` page."""
    heads = re.findall(r'<div class="panel-heading">\s*<h\d[^>]*>(.*?)</h\d>|<h5[^>]*>(.*?)</h5>',
                       page, flags=re.S)
    out = []
    for a, b in heads:
        h = _flat(a or b)
        # Exclusion is STRUCTURAL: only '<date> Meeting - YYYY' headings whose date part parses are
        # kept. 'March 15 (unscheduled) Meeting', 'March 17-18 (cancelled) Meeting', 'March 19
        # (notation vote)' and '... Conference Call' never parse (W83a mutation check: a separate
        # exclusion pattern was dead code and was removed).
        m = re.fullmatch(rf"(.+?) Meeting - {year}", h)
        if not m:
            continue
        d = _decision_day(m.group(1), year)
        if d is not None:
            out.append(d)
    return sorted(set(out))


def parse_fomc_calendars(page: str) -> dict[int, list[date]]:
    """{year: scheduled decision days} from ``fomccalendars.htm`` (the recent years)."""
    out: dict[int, list[date]] = {}
    parts = re.split(r"(\d{4}) FOMC Meetings", page)
    for i in range(1, len(parts) - 1, 2):
        year, block = int(parts[i]), parts[i + 1]
        rows = re.findall(r'fomc-meeting__month[^>]*><strong>([^<]+)</strong></div>\s*'
                          r'<div class="fomc-meeting__date[^>]*>([^<]+)</div>', block)
        days = []
        for month_s, date_s in rows:
            ds = date_s.strip()
            if not re.fullmatch(r"\d{1,2}(?:-\d{1,2})?\*?", ds):     # '22 (notation vote)' never parses
                continue
            last_month = month_s.strip().split("/")[-1][:3].lower()
            month = _MONTH_ABBR.get(last_month)
            if month:
                days.append(date(year, month, int(ds.rstrip("*").split("-")[-1])))
        out[year] = sorted(set(days))
    return out


# ---------------------------------------------------------------------------------------------
# BLS
# ---------------------------------------------------------------------------------------------

def parse_bls_archive(page: str, prefix: str) -> list[date]:
    """Release dates from the BLS archive links ``<prefix>_MMDDYYYY.htm|pdf`` (cpi, empsit)."""
    raw = set(re.findall(rf"/news\.release/archives/{re.escape(prefix)}_(\d{{8}})\.(?:htm|pdf)", page))
    return sorted(date(int(s[4:]), int(s[:2]), int(s[2:4])) for s in raw)


# ---------------------------------------------------------------------------------------------
# QQQ holdings
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Holding:
    name: str
    value_usd: float


_NUM = re.compile(r"^\$?\s*([\d]{1,3}(?:,\d{3})+|\d+)$")
_STOP = re.compile(r"Total\s+(?:Common\s+Stocks|Investments)|Statement\s+of\s+Assets", re.I)


def _cells(tr: str) -> list[str]:
    out = []
    for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, flags=re.S | re.I):
        s = _flat(c)
        if s and s not in ("$", "%"):
            out.append(s)
    return out


def _schedule_rows(seg: str) -> list[Holding]:
    rows: list[Holding] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", seg, flags=re.S | re.I):
        c = _cells(tr)
        if _STOP.search(" ".join(c)):
            if rows:
                break
            continue
        nums = [x for x in c if _NUM.match(x)]
        words = [x for x in c if not _NUM.match(x)]
        # one text cell (the name) and two numbers (shares, value): the value is the LAST number
        # in every layout seen 2006-2025 (names first, or shares first as in FY2016-2018)
        if (len(nums) == 2 and len(words) == 1 and re.search(r"[A-Za-z]{2}", words[0])
                and not re.search(r"\d{4}|%", words[0])):
            name = re.sub(r"\s*\([a-d]\)|\*", "", words[0]).strip()
            rows.append(Holding(name, float(int(_NUM.match(nums[1]).group(1).replace(",", "")))))
    return rows


def parse_n30b2_schedule(doc: str, min_rows: int = 50) -> list[Holding]:
    """Holdings of an N-30B-2 annual report: the first 'Schedule of Investments' that is a real
    holdings table (the table of contents and later prose mentions yield no rows)."""
    for m in re.finditer(r"Schedule\s+of\s+Investments", doc, flags=re.I):
        rows = _schedule_rows(doc[m.start():])
        if len(rows) >= min_rows:
            return rows
    return []


def parse_nport_holdings(xml_text: str) -> tuple[date | None, list[Holding]]:
    """(reporting period end, holdings) of an NPORT-P primary document."""
    root = ET.fromstring(xml_text)
    ns = {"n": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
    q = (lambda t: f"n:{t}") if ns else (lambda t: t)
    rp = root.findtext(f".//{q('repPdDate')}", namespaces=ns)
    out = []
    for s in root.findall(f".//{q('invstOrSec')}", namespaces=ns):
        name = (s.findtext(q("name"), namespaces=ns) or "").strip()
        val = s.findtext(q("valUSD"), namespaces=ns)
        if name and val:
            out.append(Holding(name, float(val)))
    return (date.fromisoformat(rp) if rp else None), out


# Companies that were in a QQQ top 10 at some report 2009-2026, as (key, CIK, name pattern,
# expected EDGAR entity-name fragment). A KEY is one economic exposure and may span several
# filers: Google Inc. reported until the 2015 Alphabet restructuring, Broadcom Ltd. until the 2018
# redomicile; their releases count for the same key. The build script fails if a top-10 name maps
# to nothing, and verifies each CIK's EDGAR entity name against the fragment, so a wrong CIK cannot
# pass.
COMPANIES: tuple[tuple[str, int, str, str], ...] = (
    ("AAPL", 320193, r"^apple", "apple"),
    ("MSFT", 789019, r"^microsoft", "microsoft"),
    ("AMZN", 1018724, r"^amazon", "amazon"),
    ("GOOGL", 1652044, r"^alphabet", "alphabet"),
    ("GOOGL", 1288776, r"^google", "google"),
    ("META", 1326801, r"^(facebook|meta platforms)", "meta platforms"),
    ("NVDA", 1045810, r"^nvidia", "nvidia"),
    ("TSLA", 1318605, r"^tesla", "tesla"),
    ("AVGO", 1730168, r"^broadcom", "broadcom"),
    ("AVGO", 1649338, r"^broadcom ltd", "broadcom"),
    ("COST", 909832, r"^costco", "costco"),
    ("NFLX", 1065280, r"^netflix", "netflix"),
    ("PEP", 77476, r"^pepsico", "pepsico"),
    ("CSCO", 858877, r"^cisco", "cisco"),
    ("INTC", 50863, r"^intel", "intel"),
    ("CMCSA", 1166691, r"^comcast", "comcast"),
    ("AMGN", 318154, r"^amgen", "amgen"),
    ("GILD", 882095, r"^gilead", "gilead"),
    ("QCOM", 804328, r"^qualcomm", "qualcomm"),
    ("ORCL", 1341439, r"^oracle", "oracle"),
    ("ADBE", 796343, r"^adobe", "adobe"),
    ("AMD", 2488, r"^advanced micro devices", "advanced micro devices"),
    ("PYPL", 1633917, r"^paypal", "paypal"),
    ("TMUS", 1283699, r"^t-mobile", "t-mobile"),
    ("CELG", 816284, r"^celgene", "celgene"),
    ("PLTR", 1321655, r"^palantir", "palantir"),
    ("MU", 723125, r"^micron", "micron"),
    ("WMT", 104169, r"^walmart", "walmart"),
    ("SBUX", 829224, r"^starbucks", "starbucks"),
    ("EBAY", 1065088, r"^ebay", "ebay"),
    ("KHC", 1637459, r"^kraft heinz", "kraft heinz"),
    ("KRFT", 1545158, r"^kraft foods", "kraft foods"),
    ("WBA", 1618921, r"^walgreens", "walgreens"),
)
# Foreign private issuers file 6-K, not 8-K Item 2.02: their releases are not in the earnings
# calendar. Recognised so the build reports them (by top-10 weight) instead of failing to map them.
FOREIGN_ISSUERS: tuple[tuple[str, str], ...] = (
    ("BIDU", r"^baidu"), ("RIMM", r"^research in motion"), ("TEVA", r"^teva"),
)


def company_key(name: str) -> str | None:
    n = name.strip().lower()
    for key, _cik, pat, _frag in COMPANIES:
        if re.search(pat, n):
            return key
    for key, pat in FOREIGN_ISSUERS:
        if re.search(pat, n):
            return key
    return None


def top_weights(holdings: Iterable[Holding], n: int = 10) -> list[tuple[int, str | None, str, float, float]]:
    """[(rank, company key, name, value, weight)] of the top ``n`` companies by value.

    Weight = value / total value of all holdings. Holdings of one company (Alphabet's or Google's
    share classes) are merged under its key; an unmapped name stays on its own (key None)."""
    hs = list(holdings)
    total = sum(h.value_usd for h in hs)
    if total <= 0:
        return []
    merged: dict[str, list] = {}
    for h in hs:
        k = company_key(h.name)
        slot = k or f"?{h.name}"
        if slot not in merged:
            merged[slot] = [k, h.name, 0.0]
        merged[slot][2] += h.value_usd
    ranked = sorted(merged.values(), key=lambda r: -r[2])[:n]
    return [(i + 1, k, name, v, v / total) for i, (k, name, v) in enumerate(ranked)]


def weights_as_of(reports: Sequence[tuple[date, dict[str, float]]], d: date) -> dict[str, float]:
    """The {company key: weight} of the latest report FILED strictly before ``d`` (no look-ahead).
    ``reports`` = [(filed_date, weights)] in any order. Empty before the first filing."""
    known = sorted((f, w) for f, w in reports if f < d)
    return dict(known[-1][1]) if known else {}


# ---------------------------------------------------------------------------------------------
# Earnings releases (EDGAR 8-K Item 2.02)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Filing:
    accession: str
    filing_date: date


def item_202_filings(pages: Iterable[dict]) -> list[Filing]:
    """Original 8-K filings carrying Item 2.02 from EDGAR submissions pages (the 'recent' block of
    the main JSON and every older 'files' page share the same columnar shape). 8-K/A amendments are
    excluded: they restate a release that already happened.

    The JSON's ``acceptanceDateTime`` is NOT used: MEASURED 2026-10-17 it is inconsistent (for some
    filings it is the true UTC time, for others the ET time plus twice the UTC offset, both labelled
    Z: Apple 2024-08-01 16:30:26 ET -> "2024-08-02T00:30:26Z", Amgen 2020-07-28 16:05:21 ET ->
    "2020-07-28T20:05:21Z"). The acceptance time comes from the filing index page instead
    (``parse_index_accepted``)."""
    out = {}
    for p in pages:
        cols = p.get("filings", {}).get("recent", p)
        for i, form in enumerate(cols.get("form", [])):
            items = (cols.get("items") or [""] * len(cols["form"]))[i] or ""
            if form == "8-K" and "2.02" in items.split(","):
                acc = cols["accessionNumber"][i]
                out[acc] = Filing(acc, date.fromisoformat(cols["filingDate"][i]))
    return sorted(out.values(), key=lambda f: (f.filing_date, f.accession))


def parse_index_accepted(page: str) -> datetime | None:
    """The 'Accepted' timestamp of an EDGAR filing index page (``<acc>-index.htm``), which EDGAR
    prints in Eastern time, as an aware ET datetime."""
    m = re.search(r'Accepted</div>\s*<div class="info">\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*</div>',
                  page)
    return datetime.fromisoformat(m.group(1)).replace(tzinfo=_ET) if m else None


def event_session(accepted_et: datetime, sessions: Sequence[date]) -> date | None:
    """The first session whose price can react: accepted at or after 16:00 ET -> the next session;
    otherwise that day if it is a session, else the next one (weekend / holiday filings)."""
    d = accepted_et.date()
    if accepted_et.timetz().replace(tzinfo=None) >= time(16, 0):
        i = bisect_right(sessions, d)
    else:
        i = bisect_left(sessions, d)
    return sessions[i] if i < len(sessions) else None


def release_events(filings: Sequence[Filing], merge_days: int = 3
                   ) -> tuple[list[Filing], list[Filing]]:
    """(events, merged): one event per release, keeping the FIRST filing and merging any Item 2.02
    filing within ``merge_days`` of the previous kept one (a second filing about the same release).

    Every original Item 2.02 8-K is an event: besides the quarterly results this keeps scheduled
    company news filed under the same item (Tesla's quarterly delivery reports, about three weeks
    before its results). W83a amendment 5 (2026-10-17, before any result): the pre-registered "one
    release per fiscal quarter, the first after quarter end" would have kept Tesla's DELIVERY report
    and dropped its results."""
    kept: list[Filing] = []
    merged: list[Filing] = []
    for f in sorted(filings, key=lambda f: (f.filing_date, f.accession)):
        if kept and (f.filing_date - kept[-1].filing_date).days <= merge_days:
            merged.append(f)
        else:
            kept.append(f)
    return kept, merged

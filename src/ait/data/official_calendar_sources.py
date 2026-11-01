"""Parsers for the official economic-release calendars: Fed, BLS, BEA and FRED (pure; no network, no IO).

One shared home for the pages the economic calendar is built from, so the live ``EconomicCalendar`` and the
research gate read ONE stored table (``official_econ_calendar``) that ``scripts/build_official_calendar.py``
fills from these parsers. The build script fetches; these parse; the tests run them on saved fixtures of the
real pages (``tests/data/official_calendar/``). Nothing here imports a network, file or database module
(``tests/test_w117_official_calendar.py`` pins that).

Sources (all official; Investing.com is no longer a gate source, it is the reconciliation baseline):

* FOMC: the Fed's per-year pages ``fomchistoricalYYYY.htm`` (2009-2020) and ``fomccalendars.htm`` (2021 on,
  the forward schedule too). SCHEDULED meetings only; the decision day is the meeting's last day.
  Unscheduled meetings, cancelled meetings, conference calls and notation votes are excluded (their dates
  were not public in advance).
* CPI / Employment Situation (NFP): the BLS archive links (past releases, the date is in the link) and the
  BLS schedule pages ``schedule/news_release/cpi.htm|empsit.htm`` (forward releases, with the release time).
* PCE (Personal Income and Outlays) and GDP (ADVANCE / initial estimate only): BEA's full-year release
  schedule ``bea.gov/news/schedule/full`` (past and future releases of the current year) and the embargo line
  of each past release page ``bea.gov/news/<year>/<slug>`` for earlier years.
* FRED ``release/dates`` (St. Louis Fed; free key): every release date of a release, history and scheduled
  future; the independent cross-check of the pages above.
"""
from __future__ import annotations

import html as _html
import json
import re
from dataclasses import dataclass
from datetime import date

_MONTHS = {m: i for i, m in enumerate(
    ("january february march april may june july august september october november december").split(), 1)}
_MONTH_ABBR = {k[:3]: v for k, v in _MONTHS.items()}

# The FOMC statement is released at 2:00 PM ET on the decision day (the Fed's own statement time).
FOMC_DECISION_TIME_ET = "14:00"

# FRED release ids. Names confirmed against the public release pages (fred.stlouisfed.org/release?rid=<id>)
# on 2026-10-31, when no API key was available: CONFIRM WITH THE KEY. The build script re-confirms every id
# against the key's own /fred/release answer before it trusts a date and refuses on a mismatch. NOTE the GDP
# id lists EVERY GDP release (advance, second, third, revisions): FRED cannot tell the advance estimate
# apart, so BEA's page is the authority for GDP and FRED only confirms the date is a GDP release date.
FRED_RELEASE_IDS: dict[str, int] = {"CPI": 10, "NFP": 50, "GDP": 53, "PCE": 54}
FRED_RELEASE_NAMES: dict[str, str] = {
    "CPI": "Consumer Price Index", "NFP": "Employment Situation", "GDP": "Gross Domestic Product",
    "PCE": "Personal Income and Outlays"}
FRED_API_BASE = "https://api.stlouisfed.org/fred"
# include_release_dates_with_no_data=true is what returns the SCHEDULED future dates.
FRED_RELEASE_DATES_QUERY = ("release_id={release_id}&include_release_dates_with_no_data=true&file_type=json"
                            "&sort_order=asc&limit=10000")

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
    """{year: scheduled decision days} from ``fomccalendars.htm`` (the recent years and the forward schedule)."""
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


def parse_bls_schedule(page: str) -> list[tuple[date, str]]:
    """(release date, 'HH:MM' ET) rows of a BLS ``schedule/news_release/<release>.htm`` page: the forward
    schedule, one row per reference month (``<td>Nov. 10, 2026</td><td>08:30 AM</td>``)."""
    out = []
    for cells in re.findall(r"<tr[^>]*>(.*?)</tr>", page, flags=re.S | re.I):
        tds = [_flat(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", cells, flags=re.S | re.I)]
        if len(tds) < 3:
            continue
        d = _parse_us_date(tds[1])
        if d is not None:
            out.append((d, _hhmm(tds[2])))
    return sorted(set(out))


def _parse_us_date(text: str) -> date | None:
    """'Nov. 10, 2026' / 'May 12, 2027' / 'September 3, 2026' -> date."""
    m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2}),\s*(\d{4})", text.strip())
    month = _MONTH_ABBR.get(m.group(1)[:3].lower()) if m else None
    return date(int(m.group(3)), month, int(m.group(2))) if m and month else None


def _hhmm(text: str) -> str:
    """'08:30 AM' / '8:30 a.m.' / '10:00 AM' -> '08:30' (24-hour ET); '' when it does not read as a time."""
    m = re.search(r"(\d{1,2}):(\d{2})\s*([ap])\.?m", text.strip().lower())
    if not m:
        return ""
    hour = int(m.group(1)) % 12 + (12 if m.group(3) == "p" else 0)
    return f"{hour:02d}:{m.group(2)}"


# ---------------------------------------------------------------------------------------------
# BEA
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class BeaRelease:
    """One row of BEA's release schedule (or one past-release page)."""

    day: date
    time_et: str          # 'HH:MM' ET ('' when not stated)
    title: str
    path: str             # '/news/2026/personal-income-and-outlays-march-2026' ('' when the row has no page)


def classify_bea_title(title: str) -> str | None:
    """'PCE' for a Personal Income and Outlays release, 'GDP' for the ADVANCE (or, after the 2025 shutdown,
    INITIAL) estimate of GDP, else None. The second / third / updated GDP estimates, GDP by industry /
    county / state and every other BEA release are not engine events (the engine's GDP list has always been
    "advance estimates only": the later estimates rarely move markets)."""
    t = _flat(title)
    if t.startswith("Personal Income and Outlays"):
        return "PCE"
    if re.match(r"(GDP|Gross Domestic Product)\b.*?\((Advance|Initial) Estimate\)", t):
        return "GDP"
    return None


def bea_schedule_year(page: str) -> int | None:
    """The year a BEA full-schedule page covers: the most common ``/news/<year>/`` in its release links."""
    years = re.findall(r'href="/news/(\d{4})/', page)
    return int(max(set(years), key=years.count)) if years else None


def parse_bea_schedule(page: str, year: int | None = None) -> list[BeaRelease]:
    """Every row of bea.gov/news/schedule/full: the release schedule of ONE calendar year, past and future.
    The dates carry no year; it is ``year`` or, when None, the year of the rows' own page links."""
    rows = re.findall(r'<tr class="scheduled-releases-type[^"]*">(.*?)</tr>', page, flags=re.S)
    year = year or bea_schedule_year(page)
    if year is None:
        raise ValueError("BEA schedule page: no year given and none found in its links")
    out = []
    for row in rows:
        d = re.search(r'class="release-date">([^<]+)<', row)
        title = re.search(r"release-title[^>]*>(.*?)</td>", row, flags=re.S)
        if not d or not title:
            continue
        m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})", _flat(d.group(1)))
        month = _MONTH_ABBR.get(m.group(1)[:3].lower()) if m else None
        if not m or not month:
            continue
        t = re.search(r'text-muted">([^<]+)<', row)
        link = re.search(r'href="(/news/\d{4}/[^"]+)"', row)
        out.append(BeaRelease(date(year, month, int(m.group(2))), _hhmm(t.group(1)) if t else "",
                              _flat(title.group(1)), link.group(1) if link else ""))
    return sorted(out, key=lambda r: (r.day, r.title))


def parse_bea_release_page(page: str) -> BeaRelease | None:
    """A past BEA release page: its title and release date from the embargo line ('EMBARGOED UNTIL RELEASE
    AT 8:30 a.m. EST, Friday, January 26, 2024'). None when the page carries no embargo line."""
    # BEA writes the time as 'a.m.' in most pages and 'A.M.' in some (Aug-2024 PCE): case-insensitive
    m = re.search(r"EMBARGOED UNTIL RELEASE AT\s+(\d{1,2}:\d{2}\s*[ap]\.?m\.?)[^,]*,\s*[A-Za-z]+,\s*"
                  r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", _flat(page), flags=re.I)
    month = _MONTH_ABBR.get(m.group(2)[:3].lower()) if m else None
    if not m or not month:
        return None
    # the page's first <h1> is the site banner ("News Release"); the release title is the next one
    titles = [t for t in (_flat(h) for h in re.findall(r"<h1[^>]*>(.*?)</h1>", page, flags=re.S))
              if t and t != "News Release"]
    return BeaRelease(date(int(m.group(4)), month, int(m.group(3))), _hhmm(m.group(1)),
                      titles[0] if titles else "", "")


_ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
_WORD = {1: "first", 2: "second", 3: "third", 4: "fourth"}


def bea_pce_page_paths(ref_year: int, ref_month: int) -> list[str]:
    """Candidate URL paths of the Personal Income and Outlays release for one reference month. The release
    year is the reference year or the next one (December data comes out in January; a delayed release can
    cross a year end), so both are candidates."""
    slug = f"personal-income-and-outlays-{list(_MONTHS)[ref_month - 1]}-{ref_year}"
    first, second = (ref_year + 1, ref_year) if ref_month == 12 else (ref_year, ref_year + 1)
    return [f"/news/{first}/{slug}", f"/news/{second}/{slug}"]


def bea_gdp_page_paths(ref_year: int, quarter: int) -> list[str]:
    """Candidate URL paths of the ADVANCE (initial) GDP estimate of one quarter. BEA's slugs were not
    consistent (ordinal or word quarter; 'gross-domestic-product-' or 'gdp-' first; the Q4 release is
    'fourth-quarter-and-year'; the shutdown-delayed Q3 2025 release is 'initial-estimate-and-corporate-
    profits'), so every known form is a candidate and the build script takes the first page that exists."""
    stems = ([f"{_ORD[quarter]}-quarter-{ref_year}", f"{_WORD[quarter]}-quarter-{ref_year}"] if quarter < 4 else
             [f"{_ORD[4]}-quarter-and-year-{ref_year}", f"{_WORD[4]}-quarter-and-year-{ref_year}"])
    forms = []
    for stem in stems:
        for suffix in ("advance-estimate", "initial-estimate", "initial-estimate-and-corporate-profits"):
            forms.append(f"gross-domestic-product-{stem}-{suffix}")
        forms.append(f"gdp-advance-estimate-{stem}")
        forms.append(f"gdp-initial-estimate-{stem}")
    release_year = ref_year + 1 if quarter == 4 else ref_year
    # a shutdown or holiday delay can move a release into the next year: that year is tried last
    return [f"/news/{y}/{f}" for y in (release_year, release_year + 1) for f in forms]


# ---------------------------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------------------------

def parse_fred_release_dates(payload: "str | bytes | dict") -> list[date]:
    """The dates of a FRED ``/fred/release/dates`` answer (JSON), ascending. Raises ValueError when the answer
    is FRED's own error object (a bad key or id) instead of returning an empty calendar."""
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    if "error_code" in data:
        raise ValueError(f"FRED error {data.get('error_code')}: {str(data.get('error_message', ''))[:120]}")
    if "release_dates" not in data:
        raise ValueError("FRED answer carries no release_dates")
    return sorted({date.fromisoformat(r["date"]) for r in data["release_dates"]})


def parse_fred_release_name(payload: "str | bytes | dict") -> str:
    """The release name of a FRED ``/fred/release`` answer (to confirm an id is the release it should be)."""
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    if "error_code" in data:
        raise ValueError(f"FRED error {data.get('error_code')}: {str(data.get('error_message', ''))[:120]}")
    return str(data["releases"][0]["name"])

"""Track D: ONE stored, official-source economic calendar read by both research and the live bot.

Covers, each against the real code path (no network anywhere; the parsers run on saved fixtures of the real
pages in ``tests/fixtures/official_calendar``):

* the parsers (Fed / BLS / BEA / FRED), including the shutdown-era forms the real pages carry;
* the store (``official_econ_calendar``): round trip, idempotent creation, replace-from semantics;
* the build script: dry run, reconciliation against Investing.com / the live lists / FRED, blocking
  disagreements, ``--accept-official``, structural failures, no FRED key needed;
* the research gate: the same trades when the table equals the old dates, a different gate when a date shifts;
* the live ``EconomicCalendar()``: loads the table read-only, falls back and alerts when it is missing, stale,
  incomplete or mis-tagged, extends a type past the table's horizon with the declared fallback tail, never
  writes, never raises;
* the fallback lists themselves (drift against the verified snapshot) and the class guards.
"""
from __future__ import annotations

import ast
import json
import re
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.data import economic_calendar as ec
from ait.data import official_calendar as oc
from ait.data import official_calendar_sources as src
from ait.data.economic_calendar import EconomicCalendar, EconomicEvent, EventType
from ait.data.historical import (
    OFFICIAL_CALENDAR_SOURCES,
    SOURCE_BEA_SCHEDULE,
    SOURCE_BLS_SCHEDULE,
    SOURCE_FED_FOMC,
    SOURCE_FRED_RELEASE_DATES,
    SOURCE_INVESTING_CSV,
    HistoricalDataStore,
)

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "tests" / "fixtures" / "official_calendar"
FETCHED = "2026-10-31T18:00:00-04:00"
_ORIGINAL_DEFAULT_PATH = ec.default_live_db_path        # captured before the suite's isolation fixture replaces it


def _fx(name: str) -> str:
    return (FX / name).read_text(encoding="utf-8")


def _row(d: str, etype: str, source: str = SOURCE_BLS_SCHEDULE, fetched: str = FETCHED, time: str = ""):
    return (d, time, etype, source, "https://example.test/page", fetched, "ab" * 32)


def _snapshot() -> dict[str, list[date]]:
    raw = json.loads(_fx("official_dates_2026_2027.json"))["dates"]
    return {t: [date.fromisoformat(d) for d in ds] for t, ds in raw.items()}


def _full_rows(fetched: str = FETCHED) -> list[tuple]:
    """A complete, correctly tagged table: the verified snapshot (all five types)."""
    tag = {"FOMC": SOURCE_FED_FOMC, "CPI": SOURCE_BLS_SCHEDULE, "NFP": SOURCE_BLS_SCHEDULE,
           "GDP": SOURCE_BEA_SCHEDULE, "PCE": SOURCE_BEA_SCHEDULE}
    return [_row(str(d), t, tag[t], fetched) for t, ds in _snapshot().items() for d in ds]


# ---------------------------------------------------------------------------------------------
# parsers (saved fixtures of the real pages)
# ---------------------------------------------------------------------------------------------

class TestParsers:
    def test_the_fed_calendar_page_gives_2027_including_the_june_move(self):
        cal = src.parse_fomc_calendars(_fx("fomc_calendars_2026_2027.htm"))
        assert cal[2026][-1] == date(2026, 12, 9) and len(cal[2026]) == 8
        assert date(2027, 6, 9) in cal[2027] and date(2027, 6, 16) not in cal[2027]      # the live list said 06-16
        assert len(cal[2027]) == 8

    def test_the_bls_schedule_page_has_dates_and_times(self):
        cpi = src.parse_bls_schedule(_fx("bls_cpi_schedule.htm"))
        assert (date(2026, 11, 10), "08:30") in cpi and (date(2027, 2, 11), "08:30") in cpi
        assert date(2027, 2, 10) not in [d for d, _ in cpi]                               # the live list said 02-10
        nfp = src.parse_bls_schedule(_fx("bls_empsit_schedule.htm"))
        assert (date(2027, 1, 8), "08:30") in nfp and (date(2026, 12, 4), "08:30") in nfp

    def test_the_bls_archive_has_the_shutdown_reschedules(self):
        """The Jan-2026 CPI came out 02-13 and the Jan-2026 jobs report 02-11 (the live list had 02-11 / 02-06)."""
        cpi = src.parse_bls_archive(_fx("bls_cpi_archive.htm"), "cpi")
        nfp = src.parse_bls_archive(_fx("bls_empsit_archive.htm"), "empsit")
        assert date(2026, 2, 13) in cpi and date(2026, 2, 11) not in cpi
        assert date(2026, 2, 11) in nfp and date(2026, 2, 6) not in nfp

    def test_the_bea_schedule_classifies_pce_and_only_the_advance_gdp(self):
        rel = src.parse_bea_schedule(_fx("bea_schedule_2026.htm"))
        pce = [r.day for r in rel if src.classify_bea_title(r.title) == "PCE"]
        gdp = [r.day for r in rel if src.classify_bea_title(r.title) == "GDP"]
        assert pce == [date(2026, 1, 22), date(2026, 2, 20), date(2026, 3, 13), date(2026, 4, 9),
                       date(2026, 4, 30), date(2026, 5, 28), date(2026, 6, 25), date(2026, 7, 30),
                       date(2026, 8, 26), date(2026, 9, 30), date(2026, 10, 29), date(2026, 11, 25),
                       date(2026, 12, 23)]
        assert gdp == [date(2026, 2, 20), date(2026, 4, 30), date(2026, 7, 30), date(2026, 10, 29)]
        titles = {r.title for r in rel}
        assert any("(Second Estimate)" in t for t in titles) and any("(Third Estimate)" in t for t in titles)
        assert any("(Updated Estimate)" in t for t in titles)              # present on the page, never an event
        # the 10:00 AM release of the combined Oct+Nov 2025 data keeps its time
        assert [r.time_et for r in rel if r.day == date(2026, 1, 22) and "Income" in r.title] == ["10:00"]

    @pytest.mark.parametrize("title,kind", [
        ("Personal Income and Outlays, March 2026", "PCE"),
        ("Personal Income and Outlays, October and November 2025", "PCE"),
        ("GDP (Advance Estimate), 1st Quarter 2026", "GDP"),
        ("Gross Domestic Product, 3rd Quarter 2025 (Initial Estimate) and Corporate Profits (Preliminary)", "GDP"),
        ("Gross Domestic Product (Advance Estimate), 4th Quarter and Year 2023", "GDP"),
        ("GDP (Second Estimate) and Corporate Profits, 1st Quarter 2026", None),
        ("GDP (Third Estimate), Industries, Corporate Profits, State GDP, and State Personal Income, 4th Quarter and "
         "Year 2025", None),
        ("Gross Domestic Product, 3rd Quarter 2025 (Updated Estimate), GDP by Industry, and Corporate Profits "
         "(Revised)", None),
        ("GDP by County and Personal Income by County, 2025", None),
        ("U.S. International Trade in Goods and Services, October 2025", None),
    ])
    def test_title_classification(self, title, kind):
        assert src.classify_bea_title(title) == kind

    def test_a_past_release_page_gives_its_date_from_the_embargo_line(self):
        aug24 = src.parse_bea_release_page(_fx("bea_pio_august_2024.htm"))          # written 'A.M.' in capitals
        assert (aug24.day, aug24.time_et) == (date(2024, 9, 27), "08:30")
        assert aug24.title.startswith("Personal Income and Outlays, August 2024")   # not the 'News Release' banner
        q3 = src.parse_bea_release_page(_fx("bea_gdp_q3_2025_initial.htm"))
        assert q3.day == date(2025, 12, 23) and src.classify_bea_title(q3.title) == "GDP"
        assert src.parse_bea_release_page(_fx("bea_gdp_q1_2024_advance.htm")).day == date(2024, 4, 25)
        assert src.parse_bea_release_page("<html>no embargo line</html>") is None

    def test_candidate_page_paths_cover_the_known_slug_forms(self):
        assert src.bea_pce_page_paths(2025, 3) == ["/news/2025/personal-income-and-outlays-march-2025",
                                                   "/news/2026/personal-income-and-outlays-march-2025"]
        assert src.bea_pce_page_paths(2023, 12)[0] == "/news/2024/personal-income-and-outlays-december-2023"
        q3 = src.bea_gdp_page_paths(2025, 3)
        assert "/news/2025/gross-domestic-product-3rd-quarter-2025-initial-estimate-and-corporate-profits" in q3
        assert "/news/2024/gross-domestic-product-first-quarter-2024-advance-estimate" in src.bea_gdp_page_paths(2024, 1)
        assert "/news/2024/gross-domestic-product-fourth-quarter-and-year-2023-advance-estimate" in \
            src.bea_gdp_page_paths(2023, 4)
        assert "/news/2026/gdp-advance-estimate-3rd-quarter-2026" in src.bea_gdp_page_paths(2026, 3)

    def test_fred_answers(self):
        cpi = src.parse_fred_release_dates(_fx("fred_release_dates_cpi.json"))
        assert date(2025, 12, 18) in cpi and cpi == sorted(cpi)
        assert src.parse_fred_release_name(_fx("fred_release_cpi.json")) == src.FRED_RELEASE_NAMES["CPI"]
        assert src.parse_fred_release_name(_fx("fred_release_gdp.json")) == src.FRED_RELEASE_NAMES["GDP"]
        with pytest.raises(ValueError, match="FRED error 400"):
            src.parse_fred_release_dates('{"error_code": 400, "error_message": "Bad Request. api_key is not set"}')
        with pytest.raises(ValueError, match="no release_dates"):
            src.parse_fred_release_dates('{"count": 0}')

    def test_fred_ids_are_the_four_releases_and_the_query_asks_for_the_scheduled_future(self):
        assert set(src.FRED_RELEASE_IDS) == {"CPI", "NFP", "GDP", "PCE"}
        assert "include_release_dates_with_no_data=true" in src.FRED_RELEASE_DATES_QUERY
        assert "api_key" not in src.FRED_RELEASE_DATES_QUERY          # the key is appended by the script only

    def test_the_shared_parsers_do_no_io(self):
        """The module is imported by the live tree's data package: it takes documents, it never fetches."""
        tree = ast.parse((ROOT / "src" / "ait" / "data" / "official_calendar_sources.py").read_text())
        mods = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not mods & {"urllib", "requests", "aiohttp", "socket", "sqlite3", "os", "pathlib", "http"}

    def test_the_old_import_path_still_works(self):
        from ait.data import research_calendars as rc
        assert rc.parse_fomc_calendars is src.parse_fomc_calendars and rc.parse_bls_archive is src.parse_bls_archive


# ---------------------------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------------------------

class TestStore:
    def test_the_table_is_created_by_the_store_init_and_init_is_idempotent(self, tmp_path):
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db)
        HistoricalDataStore(db_path=db)              # the live bot runs the same init again
        with sqlite3.connect(db) as c:
            cols = [r[1] for r in c.execute("PRAGMA table_info(official_econ_calendar)")]
        assert cols == ["event_date", "event_time_et", "event_type", "source", "source_url", "fetched_at",
                        "sha256"]

    def test_round_trip_and_span(self, tmp_path):
        st = HistoricalDataStore(db_path=tmp_path / "h.db")
        assert st.official_econ_calendar_span() is None
        n = st.save_official_econ_calendar([_row("2026-02-13", "CPI", time="08:30"), _row("2026-01-28", "FOMC",
                                            SOURCE_FED_FOMC, time="14:00")])
        assert n == 2
        df = st.load_official_econ_calendar()
        assert list(df.event_date) == ["2026-01-28", "2026-02-13"]
        assert df.iloc[1].to_dict() == {"event_date": "2026-02-13", "event_time_et": "08:30", "event_type": "CPI",
                                        "source": SOURCE_BLS_SCHEDULE, "source_url": "https://example.test/page",
                                        "fetched_at": FETCHED, "sha256": "ab" * 32}
        assert list(st.load_official_econ_calendar(event_types=["FOMC"]).event_type) == ["FOMC"]
        assert st.official_econ_calendar_span() == ("2026-01-28", "2026-02-13", 2)

    def test_one_row_per_date_and_type(self, tmp_path):
        st = HistoricalDataStore(db_path=tmp_path / "h.db")
        st.save_official_econ_calendar([_row("2026-04-30", "PCE", SOURCE_BEA_SCHEDULE),
                                        _row("2026-04-30", "GDP", SOURCE_BEA_SCHEDULE),
                                        _row("2026-04-30", "PCE", SOURCE_FRED_RELEASE_DATES)])
        df = st.load_official_econ_calendar()
        assert sorted(df.event_type) == ["GDP", "PCE"]
        assert df[df.event_type == "PCE"].iloc[0].source == SOURCE_FRED_RELEASE_DATES      # replaced, not doubled

    def test_replace_from_removes_a_moved_release_in_the_same_transaction(self, tmp_path):
        """The BLS moves a release: the old date must not remain as a second event."""
        st = HistoricalDataStore(db_path=tmp_path / "h.db")
        st.save_official_econ_calendar([_row("2025-12-10", "CPI"), _row("2026-02-11", "CPI")])
        st.save_official_econ_calendar([_row("2026-02-13", "CPI")], replace_from=date(2026, 1, 1))
        assert list(st.load_official_econ_calendar().event_date) == ["2025-12-10", "2026-02-13"]

    def test_the_source_tags_are_the_four_official_ones(self):
        assert OFFICIAL_CALENDAR_SOURCES == ("FED_FOMC_PAGES", "BLS_SCHEDULE", "BEA_SCHEDULE", "FRED_RELEASE_DATES")
        assert SOURCE_INVESTING_CSV not in OFFICIAL_CALENDAR_SOURCES


# ---------------------------------------------------------------------------------------------
# the build script (no network: a fetcher serving the saved fixtures)
# ---------------------------------------------------------------------------------------------

sys.path.insert(0, str(ROOT / "scripts"))
import build_official_calendar as bo  # noqa: E402


class FixtureFetcher:
    """Serves the saved page fixtures by URL; anything else is a 404 (``get_optional`` -> None, ``get`` raises)."""

    def __init__(self, extra: dict[str, bytes] | None = None, override: dict[str, bytes] | None = None):
        b = lambda n: (FX / n).read_bytes()  # noqa: E731
        self.pages = {
            f"{bo.BLS_ARCHIVE}/cpi.htm": b("bls_cpi_archive.htm"),
            f"{bo.BLS_ARCHIVE}/empsit.htm": b("bls_empsit_archive.htm"),
            f"{bo.BLS_SCHEDULE}/cpi.htm": b("bls_cpi_schedule.htm"),
            f"{bo.BLS_SCHEDULE}/empsit.htm": b("bls_empsit_schedule.htm"),
            bo.FED_CAL: b("fomc_calendars_2026_2027.htm"),
            bo.BEA_SCHEDULE_URL: b("bea_schedule_2026.htm"),
        }
        self.pages.update(extra or {})
        self.pages.update(override or {})
        self.requests = 0
        self.asked: list[str] = []

    def get(self, url, name):
        self.asked.append(url)
        if url not in self.pages:
            raise FileNotFoundError(url)
        return self.pages[url]

    def get_optional(self, url, name):
        self.asked.append(url)
        return self.pages.get(url)

    def fetched_at(self, name):
        return FETCHED


def _build(tmp_path, *args, fetcher=None, db=None, monkeypatch=None):
    db = db or tmp_path / "h.db"
    argv = ["--since", "2026-01-01", "--db-path", str(db), "--report-dir", str(tmp_path / "reports"),
            "--no-fred", *args]
    rc = bo.main(argv, fetcher=fetcher or FixtureFetcher())
    return rc, db


def _seed_investing(db: Path, rows: list[tuple[str, str]]):
    """Investing.com baseline rows (date, engine type) in the same table the real loader fills."""
    st = HistoricalDataStore(db_path=db)
    df = pd.DataFrame([{"event_date": d, "event_time_et": "08:30", "country": "United States", "event": f"{t} x",
                        "reference_period": "", "engine_event_type": t} for d, t in rows])
    st.save_econ_events(df, SOURCE_INVESTING_CSV, "cal.csv", "ab")


class TestBuildScript:
    def test_the_fixture_pages_reproduce_the_verified_dates(self, tmp_path):
        """The parsers' own output, not a web summary, is what equals the official schedule."""
        p = bo.Pages(FixtureFetcher(), date(2026, 10, 31))
        cands = bo.collect_fomc(p, date(2026, 1, 1)) + bo.collect_bls(p, date(2026, 1, 1))
        bea, _ = bo.collect_bea(p, date(2026, 1, 1), date(2026, 10, 31))
        got: dict[str, list[date]] = {}
        for c in sorted(cands + bea, key=lambda c: c.day):
            got.setdefault(c.etype, []).append(c.day)
        snap = _snapshot()
        # the CPI / NFP archive fixture holds 2025-2026 links only up to the Oct-2026 release; the schedule
        # fixture carries the rest. FOMC / GDP / PCE come whole.
        assert got["FOMC"] == snap["FOMC"]
        assert got["CPI"] == [d for d in snap["CPI"] if d <= date(2027, 12, 31)][:len(got["CPI"])]
        assert got["NFP"][:3] == snap["NFP"][:3]
        assert got["GDP"] == [d for d in snap["GDP"] if d < date(2027, 1, 1)]
        assert got["PCE"] == [d for d in snap["PCE"] if d < date(2027, 1, 1)]

    def test_a_dry_run_reports_and_writes_nothing(self, tmp_path):
        rc, db = _build(tmp_path)
        assert rc == 0
        assert not db.exists()                               # not even created: the dry run never opens the store
        rep = json.loads(next((tmp_path / "reports").glob("official_calendar_reconciliation_*.json")).read_text())
        assert rep["rows"]["PCE"] == 13 and rep["rows"]["GDP"] == 4 and rep["rows"]["FOMC"] == 16
        assert rep["fred"].startswith("disabled") and rep["disagreements"] == []
        assert rep["reconciliation"]["investing"] is None            # no baseline file -> stated, not invented

    def test_apply_writes_the_table_with_provenance(self, tmp_path):
        rc, db = _build(tmp_path, "--apply")
        assert rc == 0
        df = HistoricalDataStore(db_path=db).load_official_econ_calendar()
        assert set(df.event_type) == set(oc.OFFICIAL_EVENT_TYPES)
        assert set(df.source) == {SOURCE_FED_FOMC, SOURCE_BLS_SCHEDULE, SOURCE_BEA_SCHEDULE}
        assert (df.fetched_at == FETCHED).all() and df.sha256.str.len().eq(64).all()
        feb = df[(df.event_date == "2026-02-13") & (df.event_type == "CPI")].iloc[0]
        assert feb.source_url.endswith("/cpi.htm")
        row = df[(df.event_date == "2026-01-22") & (df.event_type == "PCE")].iloc[0]
        assert row.event_time_et == "10:00" and row.source == SOURCE_BEA_SCHEDULE

    def test_apply_is_idempotent(self, tmp_path):
        _, db = _build(tmp_path, "--apply")
        n1 = len(HistoricalDataStore(db_path=db).load_official_econ_calendar())
        _build(tmp_path, "--apply", db=db)
        assert len(HistoricalDataStore(db_path=db).load_official_econ_calendar()) == n1

    def test_agreement_with_investing_loads_silently(self, tmp_path):
        db = tmp_path / "h.db"
        _seed_investing(db, [("2026-02-13", "CPI"), ("2026-02-11", "NFP"), ("2026-03-18", "FOMC"),
                             ("2026-04-09", "PCE"), ("2026-04-30", "GDP")])
        rc, _ = _build(tmp_path, "--apply", db=db)
        assert rc == 0
        rep = json.loads(next((tmp_path / "reports").glob("official_calendar_reconciliation_*.json")).read_text())
        assert rep["reconciliation"]["investing"]["CPI"]["matched"] >= 1
        assert rep["disagreements"] == []

    def test_an_investing_disagreement_blocks_the_apply_until_accepted(self, tmp_path):
        """Investing.com has the OLD Jan-CPI date: the official date wins only after the operator sees the list."""
        db = tmp_path / "h.db"
        _seed_investing(db, [("2026-02-11", "CPI"), ("2026-03-11", "CPI")])
        rc, _ = _build(tmp_path, "--apply", db=db)
        assert rc == 2
        assert HistoricalDataStore(db_path=db).load_official_econ_calendar().empty            # nothing written
        rep = json.loads(next((tmp_path / "reports").glob("official_calendar_reconciliation_*.json")).read_text())
        assert "CPI: Investing.com 2026-02-11 is not in the official calendar" in rep["disagreements"]
        assert "CPI: official 2026-02-13 is not in Investing.com" in rep["disagreements"]
        rc, _ = _build(tmp_path, "--apply", "--accept-official", db=db)
        assert rc == 0
        assert "2026-02-13" in set(HistoricalDataStore(db_path=db).load_official_econ_calendar().event_date)

    def test_a_dry_run_also_exits_2_so_the_operator_sees_the_block(self, tmp_path):
        db = tmp_path / "h.db"
        _seed_investing(db, [("2026-02-11", "CPI"), ("2026-03-11", "CPI")])
        rc, _ = _build(tmp_path, db=db)
        assert rc == 2

    def test_a_live_list_disagreement_blocks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ec, "_CPI_2026", [d if d != date(2026, 2, 13) else date(2026, 2, 11)
                                              for d in ec._CPI_2026])        # the pre-fix list
        rc, db = _build(tmp_path, "--apply")
        assert rc == 2 and not db.exists()
        rep = json.loads(next((tmp_path / "reports").glob("official_calendar_reconciliation_*.json")).read_text())
        assert "CPI: live fallback 2026-02-11 is not an official date" in rep["disagreements"]
        assert "CPI: official 2026-02-13 is not in the live fallback list" in rep["disagreements"]

    def test_the_corrected_lists_agree_with_the_pages_so_a_clean_build_is_not_blocked(self, tmp_path):
        rep_rc, _ = _build(tmp_path)
        assert rep_rc == 0

    def test_a_structural_failure_blocks_even_with_accept_official(self, tmp_path):
        fx = FixtureFetcher(override={bo.BEA_SCHEDULE_URL: (FX / "bea_schedule_2026.htm").read_bytes().replace(
            b"Personal Income and Outlays", b"Something else")})
        rc, db = _build(tmp_path, "--apply", "--accept-official", fetcher=fx)
        assert rc == 1 and not db.exists()
        rep = json.loads(next((tmp_path / "reports").glob("official_calendar_reconciliation_*.json")).read_text())
        assert "PCE: no dates at all" in rep["structural_failures"]

    def test_gdp_without_a_future_date_is_a_horizon_note_not_a_failure(self, tmp_path):
        """BEA publishes one year at a time: in late December there is no later GDP advance date yet."""
        fails = bo.validate([bo.Cand(date(2026, 4, 30), "", "GDP", SOURCE_BEA_SCHEDULE, "u", "x", FETCHED)]
                            + [bo.Cand(date(2026, 12, 9), "", t, SOURCE_BLS_SCHEDULE, "u", "x", FETCHED)
                               for t in ("FOMC", "CPI", "NFP", "PCE")], date(2026, 6, 1), date(2026, 12, 20))
        assert not any(f.startswith("GDP") for f in fails)

    def test_a_missing_future_for_another_type_is_a_failure(self):
        fails = bo.validate([bo.Cand(date(2026, 4, 10), "", "CPI", SOURCE_BLS_SCHEDULE, "u", "x", FETCHED)],
                            date(2026, 1, 1), date(2026, 10, 31))
        assert any("CPI: the newest date" in f for f in fails)

    def test_unresolved_past_release_pages_are_listed_not_fatal(self, tmp_path):
        """A past BEA page that cannot be found (a combined release) is reported; the cross-checks catch a real hole."""
        fx = FixtureFetcher()
        p = bo.Pages(fx, date(2026, 10, 31))
        _, audit = bo.collect_bea(p, date(2024, 1, 1), date(2026, 10, 31))
        assert "PCE 2024-01" in audit["unresolved_pages"] and "GDP 2025Q3" in audit["unresolved_pages"]

    def test_a_past_release_page_is_used_for_a_year_the_schedule_page_does_not_cover(self):
        extra = {bo.BEA + "/news/2024/personal-income-and-outlays-august-2024": (FX / "bea_pio_august_2024.htm").read_bytes(),
                 bo.BEA + "/news/2025/gross-domestic-product-3rd-quarter-2025-initial-estimate-and-corporate-profits":
                     (FX / "bea_gdp_q3_2025_initial.htm").read_bytes()}
        bea, audit = bo.collect_bea(bo.Pages(FixtureFetcher(extra), date(2026, 10, 31)), date(2024, 1, 1),
                                    date(2026, 10, 31))
        days = {(c.etype, c.day) for c in bea}
        assert ("PCE", date(2024, 9, 27)) in days and ("GDP", date(2025, 12, 23)) in days
        assert "PCE 2024-08" not in audit["unresolved_pages"] and "GDP 2025Q3" not in audit["unresolved_pages"]

    def test_a_page_of_the_wrong_kind_is_not_taken_for_the_release(self):
        """A path that resolves to a page whose title is not the wanted release is skipped, not trusted."""
        extra = {bo.BEA + "/news/2024/gross-domestic-product-first-quarter-2024-advance-estimate":
                 (FX / "bea_pio_august_2024.htm").read_bytes()}          # a PCE page where a GDP page was expected
        bea, audit = bo.collect_bea(bo.Pages(FixtureFetcher(extra), date(2026, 10, 31)), date(2024, 1, 1),
                                    date(2026, 10, 31))
        assert "GDP 2024Q1" in audit["unresolved_pages"]
        assert not any(c.etype == "GDP" and c.day.year == 2024 for c in bea)

    def test_fred_cross_check_confirms_and_a_missing_page_date_blocks(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "test-key-not-real")
        cpi = json.loads((FX / "fred_release_dates_cpi.json").read_text())
        gdp = json.loads((FX / "fred_release_dates_gdp.json").read_text())
        names = {rid: (FX / f"fred_release_{nm}.json").read_text() for rid, nm in ((10, "cpi"), (53, "gdp"))}

        class Fred(FixtureFetcher):
            def get(self, url, name):
                if url.startswith(src.FRED_API_BASE):
                    rid = int(url.split("release_id=")[1].split("&")[0])
                    if "/release/dates" in url:
                        body = {10: cpi, 53: gdp}.get(rid, cpi)       # NFP / PCE are served CPI's dates
                        return json.dumps(body).encode()
                    return (names.get(rid) or json.dumps({"releases": [{"name": src.FRED_RELEASE_NAMES[
                        {v: k for k, v in src.FRED_RELEASE_IDS.items()}[rid]]}]})).encode()
                return super().get(url, name)

        argv = ["--since", "2026-01-01", "--db-path", str(tmp_path / "h.db"), "--report-dir", str(tmp_path / "r")]
        rc = bo.main(argv, fetcher=Fred())
        rep = json.loads(next((tmp_path / "r").glob("official_calendar_reconciliation_*.json")).read_text())
        assert rep["fred"] == "used"
        # the (real) fixture answers hold the CPI and GDP release dates; NFP and PCE were served CPI's dates
        assert rep["reconciliation"]["fred"]["CPI"]["only_official"] == []
        assert rep["reconciliation"]["fred"]["GDP"]["only_official"] == []
        assert rc == 2 and any(line.startswith("NFP: official") and "FRED" in line for line in rep["disagreements"])
        assert "FOMC" not in rep["reconciliation"]["fred"]                     # FRED has no FOMC release

    def test_a_fred_id_that_is_not_the_expected_release_refuses(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FRED_API_KEY", "test-key-not-real")

        class Wrong(FixtureFetcher):
            def get(self, url, name):
                if url.startswith(src.FRED_API_BASE):
                    return json.dumps({"releases": [{"name": "Something Else"}]}).encode()
                return super().get(url, name)

        with pytest.raises(SystemExit, match="FRED release id 10 is 'Something Else'"):
            bo.main(["--since", "2026-01-01", "--db-path", str(tmp_path / "h.db"),
                     "--report-dir", str(tmp_path / "r")], fetcher=Wrong())

    def test_the_fred_key_never_reaches_the_report_or_the_output(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("FRED_API_KEY", "SECRET-KEY-VALUE-123")

        class Boom(FixtureFetcher):
            def get(self, url, name):
                if url.startswith(src.FRED_API_BASE):
                    raise RuntimeError(f"connection failed for {url}")
                return super().get(url, name)

        with pytest.raises(SystemExit) as e:
            bo.main(["--since", "2026-01-01", "--db-path", str(tmp_path / "h.db"),
                     "--report-dir", str(tmp_path / "r")], fetcher=Boom())
        assert "SECRET-KEY-VALUE-123" not in str(e.value)
        assert "SECRET-KEY-VALUE-123" not in capsys.readouterr().out
        assert e.value.__cause__ is None and e.value.__suppress_context__

    def test_no_key_falls_back_to_the_agency_pages(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        monkeypatch.setattr(bo, "_root", tmp_path)                    # so no .env of the checkout is read
        argv = ["--since", "2026-01-01", "--db-path", str(tmp_path / "h.db"), "--report-dir", str(tmp_path / "r")]
        assert bo.main(argv, fetcher=FixtureFetcher()) == 0
        rep = json.loads(next((tmp_path / "r").glob("official_calendar_reconciliation_*.json")).read_text())
        assert rep["fred"] == "not used: FRED_API_KEY is not set"

    def test_the_script_never_prints_a_key_variable(self):
        text = (ROOT / "scripts" / "build_official_calendar.py").read_text()
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "print":
                assert "key" not in ast.unparse(node).lower().replace("keyerror", "")


# ---------------------------------------------------------------------------------------------
# research: the gate reads the table
# ---------------------------------------------------------------------------------------------

class TestResearchGate:
    @pytest.fixture(scope="class")
    def harness(self):
        from ait.ml.features import FeatureEngine
        idx = pd.bdate_range("2024-01-02", periods=160)
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(idx))))
        df = pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close,
                           "Volume": 1e6}, index=idx)
        return df, FeatureEngine().compute(df), [d.date() for d in idx]

    @staticmethod
    def _run(harness, calendar):
        from ait.backtesting.engine import Backtester
        df, feats, _ = harness
        res = Backtester(
            data=df, strategies=["jade_lizard"], range_min_confidence=0.55, initial_capital=100_000,
            macro_event_gate=calendar is not None, economic_calendar=calendar, allow_live_model_fallback=False,
            hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0, iv_rank_rise_threshold=10.0,
            ic_min_credit=0.0, ic_min_credit_width=0.0, iv_floor=0.01, features_cache=feats).run()
        return [(t["entry_date"], t["exit_date"], round(t["pnl"], 6)) for t in res.trades]

    @staticmethod
    def _table(tmp_path, name, dates_by_type):
        db = tmp_path / f"{name}.db"
        HistoricalDataStore(db_path=db).save_official_econ_calendar(
            [_row(str(d), t, SOURCE_BEA_SCHEDULE) for t, ds in dates_by_type.items() for d in ds])
        return db

    def test_the_same_dates_give_the_same_trades_and_a_shifted_date_changes_the_gate(self, tmp_path, harness):
        days = harness[2]
        ungated = self._run(harness, None)
        assert len(ungated) > 5, "the synthetic run must trade for this test to mean anything"
        entries = sorted({date.fromisoformat(e) for e, _x, _p in ungated})
        hit = entries[2]                                        # an event ON a day the engine wants to enter
        quiet = next(d for d in days[40:] if all(abs((d - e).days) > 6 for e in entries))
        old = {"CPI": [hit], "FOMC": [days[-1]]}
        # the SAME dates in another table, split over other event types: the engine gates on dates
        split = {"NFP": [hit], "PCE": [days[-1]]}
        shifted = {"CPI": [quiet], "FOMC": [days[-1]]}        # the hit date moved to a quiet day

        def cal(db, types):
            return ec.build_research_calendar(db, "", types)

        base = self._run(harness, cal(self._table(tmp_path, "a", old), ["CPI", "FOMC"]))
        same = self._run(harness, cal(self._table(tmp_path, "b", split), ["NFP", "PCE"]))
        moved = self._run(harness, cal(self._table(tmp_path, "c", shifted), ["CPI", "FOMC"]))
        assert base == same                                  # identical trades when the table equals the old dates
        assert base != ungated                               # the gate bites on that date
        assert moved != base                                 # and moves with a shifted date
        assert hit.isoformat() not in {e for e, _x, _p in base}

    def test_the_research_calendar_does_not_read_the_investing_rows(self, tmp_path):
        db = tmp_path / "h.db"
        _seed_investing(db, [("2026-02-11", "CPI")])
        assert ec.build_research_calendar(db, "", ["CPI"]) is None

    def test_only_the_asked_types_are_gated(self, tmp_path):
        db = self._table(tmp_path, "t", {"CPI": [date(2026, 2, 13)], "FOMC": [date(2026, 1, 28)]})
        cal = ec.build_research_calendar(db, "", ["FOMC"])
        assert [e.event_type for e in cal._events] == [EventType.FOMC] and cal.source == "explicit"

    def test_the_walk_forward_reads_the_official_table_and_trims_to_its_span(self, tmp_path):
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        db = self._table(tmp_path, "w", {"FOMC": [date(2024, 3, 20), date(2024, 5, 1)]})
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], db_path=db, table_prefix="", strict_inputs=False,
                                   config=WalkForwardConfig(macro_event_gate=True, macro_event_types=("FOMC",),
                                                            friction_model="legacy"))
        idx = pd.bdate_range("2024-01-02", periods=120)
        frame = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1.0}, index=idx)
        out = wf._apply_research_calendar({"SPY": frame})
        assert out["SPY"].index[0] >= pd.Timestamp("2024-03-20")
        assert [e.date for e in wf._research_calendar._events] == [date(2024, 3, 20), date(2024, 5, 1)]


# ---------------------------------------------------------------------------------------------
# live: EconomicCalendar() reads the table, falls back and alerts
# ---------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _fresh_exhaustion_flag(monkeypatch):
    """A research calendar near its end sets the process-wide exhaustion flag as a side effect (pre-existing); start
    every test of this module from a clean flag so none depends on test order."""
    monkeypatch.setattr(EconomicCalendar, "_exhausted_warned", False)


@pytest.fixture
def live_alerts(monkeypatch):
    """Un-silence the once-per-process fallback alert for the test (conftest marks it raised) and capture logs."""
    from structlog.testing import capture_logs
    monkeypatch.setattr(EconomicCalendar, "_fallback_alerted", False)
    monkeypatch.setattr(EconomicCalendar, "_exhausted_warned", False)
    with capture_logs() as logs:
        yield logs


def _live_db(tmp_path, rows=None, name="live.db") -> Path:
    db = tmp_path / name
    HistoricalDataStore(db_path=db).save_official_econ_calendar(_full_rows() if rows is None else rows)
    return db


def _now(monkeypatch, day: date):
    monkeypatch.setattr(ec, "today_et", lambda: day)


class TestLiveCalendar:
    def test_loads_the_table_and_the_table_wins_over_the_lists(self, tmp_path, live_alerts, monkeypatch):
        _now(monkeypatch, date(2026, 10, 31))
        cal = EconomicCalendar(db_path=_live_db(tmp_path))
        assert cal.source == "table" and not cal.exhausted_warned
        assert [e.event_type for e in cal.get_events_on(date(2026, 2, 13))] == [EventType.CPI]    # CPI moved to 02-13
        assert [e.event_type for e in cal.get_events_on(date(2026, 2, 11))] == [EventType.NFP]    # ...jobs report to 02-11
        assert cal.is_event_day(date(2027, 6, 9)) and not cal.is_event_day(date(2027, 6, 16))
        assert [e.event_type.value for e in cal.get_events_on(date(2026, 10, 29))] == ["GDP", "PCE"]
        assert not [e for e in live_alerts if e["event"] == "economic_calendar_table_unusable"]
        assert any(e["event"] == "economic_calendar_loaded" for e in live_alerts)

    def test_a_table_date_with_no_fallback_twin_is_used_and_the_old_list_date_is_gone(self, tmp_path, live_alerts,
                                                                                     monkeypatch):
        _now(monkeypatch, date(2026, 10, 31))
        rows = [r for r in _full_rows() if not (r[0] == "2026-02-13" and r[2] == "CPI")] + [_row("2026-02-12", "CPI")]
        cal = EconomicCalendar(db_path=_live_db(tmp_path, rows))
        assert cal.is_event_day(date(2026, 2, 12)) and not cal.is_event_day(date(2026, 2, 13))

    def test_the_default_path_is_the_one_the_bot_uses_and_is_resolved_inside_the_module(self, monkeypatch):
        from ait.data.historical import DB_PATH
        assert _ORIGINAL_DEFAULT_PATH() == DB_PATH       # the function as shipped (the suite's fixture replaces it)

    def test_the_default_path_is_read_when_no_argument_is_given(self, tmp_path, live_alerts, monkeypatch):
        _now(monkeypatch, date(2026, 10, 31))
        db = _live_db(tmp_path)
        monkeypatch.setattr(ec, "default_live_db_path", lambda: db)
        assert EconomicCalendar().source == "table"

    def test_it_reads_read_only_and_never_creates_a_file_or_table(self, tmp_path, live_alerts):
        missing = tmp_path / "absent.db"
        EconomicCalendar(db_path=missing)
        assert not missing.exists()
        empty = tmp_path / "empty.db"
        sqlite3.connect(empty).close()
        before = empty.read_bytes()
        EconomicCalendar(db_path=empty)
        assert empty.read_bytes() == before                         # no table created in someone else's database
        with sqlite3.connect(empty) as c:
            assert not c.execute("SELECT name FROM sqlite_master").fetchall()

    def test_the_table_is_not_modified_by_loading_it(self, tmp_path, live_alerts, monkeypatch):
        _now(monkeypatch, date(2026, 10, 31))
        db = _live_db(tmp_path)
        before = db.read_bytes()
        cal = EconomicCalendar(db_path=db)
        cal.is_event_day(date(2026, 2, 13))
        assert db.read_bytes() == before

    @pytest.mark.parametrize("make,reason", [
        (lambda t: t / "absent.db", "unreadable"),
        (lambda t: _live_db(t, [], "norows.db"), "unreadable"),
        (lambda t: _live_db(t, [r for r in _full_rows() if r[2] != "PCE"], "nope.db"), "incomplete: no rows for PCE"),
        (lambda t: _live_db(t, [r[:3] + (SOURCE_INVESTING_CSV,) + r[4:] for r in _full_rows()], "tag.db"),
         "non-official source"),
        (lambda t: _live_db(t, _full_rows("2026-05-01T10:00:00-04:00"), "old.db"), "stale: last fetched 2026-05-01"),
    ])
    def test_every_unusable_table_falls_back_to_the_lists_and_alerts(self, tmp_path, live_alerts, monkeypatch,
                                                                    make, reason):
        _now(monkeypatch, date(2026, 10, 31))
        cal = EconomicCalendar(db_path=make(tmp_path))
        assert cal.source == "fallback" and cal._events is ec._ALL_EVENTS
        assert cal.exhausted_warned                                # the orchestrator's Telegram poll sees it
        crit = [e for e in live_alerts if e["event"] == "economic_calendar_table_unusable"]
        assert len(crit) == 1 and crit[0]["log_level"] == "critical" and reason in crit[0]["reason"]
        assert crit[0]["fallback"] == "hardcoded_lists"

    def test_the_fallback_still_guards_exactly_as_before(self, tmp_path, live_alerts):
        cal = EconomicCalendar(db_path=tmp_path / "absent.db")
        assert cal.is_event_day(date(2026, 2, 13)) and cal.should_skip_trading(date(2026, 2, 12))
        assert cal.is_safe_to_hold_through_expiry(date(2026, 8, 17), date(2026, 8, 21)) is True
        assert cal.is_safe_to_hold_through_expiry(date(2026, 8, 10), date(2026, 8, 14)) is False      # CPI 08-12

    def test_the_fallback_alert_is_once_per_process(self, tmp_path, live_alerts):
        EconomicCalendar(db_path=tmp_path / "a.db")
        EconomicCalendar(db_path=tmp_path / "b.db")
        assert len([e for e in live_alerts if e["event"] == "economic_calendar_table_unusable"]) == 1

    def test_a_corrupt_file_never_raises_into_the_entry_loop(self, tmp_path, live_alerts):
        bad = tmp_path / "bad.db"
        bad.write_bytes(b"this is not a sqlite database" * 50)
        cal = EconomicCalendar(db_path=bad)
        assert cal.source == "fallback"

    def test_a_stale_table_does_not_hide_the_exhaustion_alarm(self, tmp_path, live_alerts, monkeypatch):
        _now(monkeypatch, date(2026, 10, 31))
        cal = EconomicCalendar(db_path=_live_db(tmp_path))
        cal.is_event_day(cal._last_event_date - timedelta(days=5))
        assert EconomicCalendar._exhausted_warned and any(
            e["event"] == "economic_calendar_nearly_exhausted" and e["source"] == "table" for e in live_alerts)

    def test_a_type_past_the_tables_horizon_keeps_the_declared_fallback_tail(self, tmp_path, live_alerts, monkeypatch):
        """BEA publishes a year at a time: 2027 GDP / PCE are not in the table, the unverified list still guards."""
        _now(monkeypatch, date(2026, 10, 31))
        rows = [r for r in _full_rows() if not (r[2] in ("GDP", "PCE") and r[0] > "2026-12-31")]
        cal = EconomicCalendar(db_path=_live_db(tmp_path, rows))
        assert cal.source == "table" and cal.is_event_day(date(2027, 1, 28)) and cal.is_event_day(date(2027, 1, 29))
        loaded = [e for e in live_alerts if e["event"] == "economic_calendar_loaded"][0]
        assert loaded["fallback_tail"] == {"GDP": 2, "PCE": 6}

    def test_an_exit_is_never_blocked_by_a_calendar_problem(self, tmp_path, live_alerts):
        """The calendar only answers questions; with an unusable table it answers from the fallback instead of raising."""
        cal = EconomicCalendar(db_path=tmp_path / "absent.db")
        assert cal.days_until_next_event(date(2026, 8, 1)) is not None
        assert cal.get_upcoming_events(7) is not None

    def test_public_api_and_event_semantics_are_unchanged(self):
        import inspect
        sig = inspect.signature(EconomicCalendar.__init__)
        assert list(sig.parameters) == ["self", "events", "db_path"] and sig.parameters["db_path"].kind.name == \
            "KEYWORD_ONLY"
        assert {e.name for e in EventType} == {"FOMC", "CPI", "NFP", "GDP", "PCE"}
        for m in ("is_event_day", "is_pre_event_day", "should_skip_trading", "get_events_on", "get_upcoming_events",
                  "days_until_next_event", "is_safe_to_hold_through_expiry", "get_events_between"):
            assert callable(getattr(EconomicCalendar, m))

    def test_explicit_events_still_bypass_the_table(self, tmp_path):
        cal = EconomicCalendar(events=[EconomicEvent(date(2024, 1, 11), EventType.CPI, "x")])
        assert cal.source == "explicit" and not cal.exhausted_warned


# ---------------------------------------------------------------------------------------------
# the fallback lists
# ---------------------------------------------------------------------------------------------

class TestFallbackLists:
    LISTS = {"FOMC": ("_FOMC_2026", "_FOMC_2027"), "CPI": ("_CPI_2026", "_CPI_2027"),
             "NFP": ("_NFP_2026", "_NFP_2027"), "GDP": ("_GDP_2026", "_GDP_2027"),
             "PCE": ("_PCE_2026", "_PCE_2027")}

    def test_every_verified_fallback_date_equals_the_official_snapshot(self):
        """Drift guard: the declared fallback is never wrong where an official page confirms it, and misses nothing
        the pages list inside its span. Lists named in UNVERIFIED_FALLBACK are exempt only while they lie beyond
        what the official pages cover."""
        snap = _snapshot()
        for etype, names in self.LISTS.items():
            official = set(snap[etype])
            verified = {d for n in names if n not in ec.UNVERIFIED_FALLBACK for d in getattr(ec, n)}
            assert verified == {d for d in official if d <= max(verified)}, (etype, sorted(verified ^ official))
            for n in names:
                if n in ec.UNVERIFIED_FALLBACK:
                    assert not [d for d in getattr(ec, n) if d <= max(official)], f"{n} is covered by the pages now"

    def test_the_known_corrections_are_in(self):
        assert date(2026, 2, 13) in ec._CPI_2026 and date(2026, 2, 11) not in ec._CPI_2026
        assert date(2026, 2, 11) in ec._NFP_2026 and date(2026, 2, 6) not in ec._NFP_2026
        assert date(2027, 6, 9) in ec._FOMC_2027 and date(2027, 6, 16) not in ec._FOMC_2027
        assert date(2027, 2, 11) in ec._CPI_2027 and date(2027, 2, 10) not in ec._CPI_2027
        assert date(2026, 2, 20) in ec._GDP_2026
        assert date(2026, 10, 29) in ec._PCE_2026 and date(2026, 10, 30) not in ec._PCE_2026
        assert {date(2026, 3, 13), date(2026, 4, 9), date(2026, 5, 28), date(2026, 6, 25), date(2026, 8, 26),
                date(2026, 9, 30)} <= set(ec._PCE_2026)

    def test_what_could_not_be_verified_is_named(self):
        assert ec.UNVERIFIED_FALLBACK == ("_GDP_2027", "_PCE_2027")
        snap = _snapshot()
        assert max(snap["GDP"]) < date(2027, 1, 1) and max(snap["PCE"]) < date(2027, 1, 1)   # BEA: not published


# ---------------------------------------------------------------------------------------------
# class guards
# ---------------------------------------------------------------------------------------------

class TestClassGuards:
    def test_only_the_store_and_the_shared_reader_name_the_table_in_src(self):
        """No other module under src/ reads (or writes) official_econ_calendar with its own SQL: the store owns the
        DDL and the writer, ait.data.official_calendar owns every read."""
        allowed = {"data/historical.py", "data/official_calendar.py"}
        hits = []
        for p in (ROOT / "src" / "ait").rglob("*.py"):
            rel = p.relative_to(ROOT / "src" / "ait").as_posix()
            if rel in allowed:
                continue
            tree = ast.parse(p.read_text())
            docs = {id(n.body[0].value) for n in ast.walk(tree)
                    if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs \
                        and "official_econ_calendar" in n.value \
                        and re.search(r"\b(SELECT|INSERT|DELETE|CREATE|UPDATE)\b", n.value):
                    hits.append(f"{rel}:{n.lineno}")
        assert hits == []

    def test_the_live_tree_reads_the_table_only_through_the_shared_reader(self):
        text = (ROOT / "src" / "ait" / "data" / "economic_calendar.py").read_text()
        assert "read_official_rows" in text and "sqlite3" not in text

    def test_the_reader_never_opens_the_database_for_writing(self):
        tree = ast.parse((ROOT / "src" / "ait" / "data" / "official_calendar.py").read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "sqlite3.connect":
                assert "mode=ro" in ast.unparse(n.args[0]) or "uri" in ast.unparse(n.args[0])
        assert "mode=ro" in (ROOT / "src" / "ait" / "data" / "official_calendar.py").read_text()

    def test_no_research_module_falls_back_to_the_hard_coded_lists_on_a_missing_table(self):
        """Strict research refuses (tests/test_w116_strict_inputs.py): no module under backtesting/ or optimization/
        imports the hard-coded lists."""
        for rel in ("backtesting", "optimization"):
            for p in (ROOT / "src" / "ait" / rel).rglob("*.py"):
                for n in ast.walk(ast.parse(p.read_text())):
                    names = [a.name for a in n.names] if isinstance(n, ast.ImportFrom) else []
                    assert not [x for x in names if x.startswith(("_FOMC_", "_CPI_", "_NFP_", "_GDP_", "_PCE_",
                                                                    "_ALL_EVENTS"))], p

    def test_orchestrator_needs_no_edit_for_the_new_constructor(self):
        text = (ROOT / "src" / "ait" / "bot" / "orchestrator.py").read_text()
        assert "self._economic_cal = EconomicCalendar()" in text

    def test_the_isolation_fixture_points_the_default_at_a_nonexistent_store(self):
        assert not ec.default_live_db_path().exists()

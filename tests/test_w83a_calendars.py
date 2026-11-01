"""W83a-1: the event calendars of the strike-aware range study.

  * parsers on trimmed REAL source documents (tests/fixtures/w83a/): Fed FOMC pages (every heading
    form, unscheduled / cancelled / notation votes excluded), the BLS archive, QQQ's N-30B-2 annual
    schedules (both table layouts), NPORT-P XML, EDGAR submissions and filing index pages
  * top-10 weights (Alphabet's classes merged, unmapped names kept visible), filing-date
    applicability (no look-ahead), the event session of a release, release merging (amendment 5)
  * the store round trip, the build script's guards, the live tree never imports the module
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from ait.data import research_calendars as rc
from ait.data.historical import (SOURCE_BLS_ARCHIVE, SOURCE_EDGAR_8K_202, SOURCE_EDGAR_N30B2,
                                 SOURCE_FED_FOMC, HistoricalDataStore)

ROOT = Path(__file__).resolve().parents[1]
FX = ROOT / "tests" / "fixtures" / "w83a"
ET = ZoneInfo("America/New_York")
sys.path.insert(0, str(ROOT / "scripts"))


def _fx(name: str, enc: str = "utf-8") -> str:
    return (FX / name).read_text(encoding=enc)


# --------------------------------------------------------------------------
# FOMC / BLS
# --------------------------------------------------------------------------

class TestFomc:
    def test_a_normal_year_and_a_cross_month_meeting(self):
        d = rc.parse_fomc_year_page(_fx("fomc_2012.htm"), 2012)
        assert len(d) == 8
        assert date(2012, 8, 1) in d                       # 'July 31-August 1 Meeting'
        assert date(2012, 1, 25) in d                      # 'January 24-25': the LAST day

    def test_slash_form_and_unscheduled_excluded(self):
        d = rc.parse_fomc_year_page(_fx("fomc_2013.htm"), 2013)
        assert len(d) == 8
        assert date(2013, 5, 1) in d                       # 'April/May 30-1 Meeting'
        assert date(2013, 10, 16) not in d                 # 'October 16 (unscheduled)'

    def test_2020_keeps_only_scheduled_meetings(self):
        d = rc.parse_fomc_year_page(_fx("fomc_2020.htm"), 2020)
        assert d == [date(2020, 1, 29), date(2020, 4, 29), date(2020, 6, 10), date(2020, 7, 29),
                     date(2020, 9, 16), date(2020, 11, 5), date(2020, 12, 16)]
        # the cancelled March 17-18 meeting, the unscheduled March 3 / 15 meetings and the
        # notation votes are not scheduled decision days

    def test_the_calendar_page(self):
        cal = rc.parse_fomc_calendars(_fx("fomc_calendars.htm"))
        assert cal[2025] == [date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
                             date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10)]
        assert len(cal[2023]) == 8 and date(2023, 2, 1) in cal[2023]     # 'Jan/Feb 31-1'

    @pytest.mark.parametrize("text, year, expected", [
        ("January 26-27", 2010, date(2010, 1, 27)), ("March 16", 2010, date(2010, 3, 16)),
        ("July 31-August 1", 2012, date(2012, 8, 1)), ("April/May 30-1", 2013, date(2013, 5, 1)),
        ("Jan/Feb 31-1", 2017, date(2017, 2, 1)), ("Smarch 3", 2010, None), ("soon", 2010, None)])
    def test_decision_day_forms(self, text, year, expected):
        assert rc._decision_day(text, year) == expected


class TestBls:
    def test_dates_come_from_the_links(self):
        d = rc.parse_bls_archive(_fx("bls_cpi.htm"), "cpi")
        assert d[:3] == [date(2024, 1, 11), date(2024, 2, 13), date(2024, 3, 12)]
        assert all(x.year in (2024, 2025) for x in d)            # 'cpi_2027.htm' is not a date
        assert len(d) == len(set(d))                             # .htm and .pdf of one release -> one date

    def test_another_prefix_finds_nothing(self):
        assert rc.parse_bls_archive(_fx("bls_cpi.htm"), "empsit") == []


# --------------------------------------------------------------------------
# QQQ holdings
# --------------------------------------------------------------------------

class TestHoldings:
    @pytest.mark.parametrize("fixture", ["n30b2_fy2017_shares_first.htm", "n30b2_fy2023_name_first.htm"])
    def test_both_layouts_parse_and_stop_at_the_total(self, fixture):
        hs = rc.parse_n30b2_schedule(_fx(fixture, "latin-1"), min_rows=5)
        assert len(hs) == 10
        assert all(h.value_usd > 1e6 for h in hs)
        names = {h.name for h in hs}
        assert "Zeta Corp." not in names                         # after the Total line
        tesla = {"n30b2_fy2017_shares_first.htm": 434_396_308.0, "n30b2_fy2023_name_first.htm": 6_308_906_206.0}
        assert [h.value_usd for h in hs if h.name == "Tesla, Inc."] == [tesla[fixture]]   # the VALUE, not shares
        assert not any("%" in n or "Contents" in n for n in names)

    def test_the_table_of_contents_is_not_the_schedule(self):
        doc = _fx("n30b2_fy2023_name_first.htm", "latin-1")
        assert rc.parse_n30b2_schedule(doc, min_rows=50) == []     # the trimmed table is short
        assert rc.parse_n30b2_schedule(doc, min_rows=5)

    def test_nport(self):
        rp, hs = rc.parse_nport_holdings(_fx("nport_2019q3.xml"))
        assert rp == date(2019, 9, 30)
        assert [h.name for h in hs] == ["Apple, Inc.", "Alphabet, Inc.", "Microsoft Corp.", "Alphabet, Inc."]
        assert hs[0].value_usd == pytest.approx(8_220_421_303.25)        # valUSD, not pctVal

    def test_top_weights_merge_share_classes_and_keep_unmapped_visible(self):
        hs = [rc.Holding("Alphabet, Inc., Class A", 30.0), rc.Holding("Alphabet, Inc., Class C", 25.0),
              rc.Holding("Apple, Inc.", 50.0), rc.Holding("Zeta Corp.", 15.0), rc.Holding("Baidu, Inc. ADR", 5.0)]
        top = rc.top_weights(hs, n=3)
        assert [(r, k) for r, k, _n, _v, _w in top] == [(1, "GOOGL"), (2, "AAPL"), (3, None)]
        assert top[0][4] == pytest.approx(55 / 125)
        assert rc.top_weights(hs, n=5)[-1][1] == "BIDU"           # a foreign issuer is recognised

    def test_google_and_alphabet_are_one_exposure(self):
        assert rc.company_key("Google, Inc., Class C") == rc.company_key("Alphabet Inc.") == "GOOGL"
        assert rc.company_key("Facebook, Inc., Class A") == rc.company_key("Meta Platforms, Inc.") == "META"
        assert rc.company_key("Broadcom Ltd.") == rc.company_key("Broadcom Inc.") == "AVGO"
        assert {c for k, c, *_ in rc.COMPANIES if k == "GOOGL"} == {1652044, 1288776}

    def test_weights_apply_only_after_the_filing_date(self):
        reports = [(date(2023, 12, 28), {"AAPL": 0.11}), (date(2024, 2, 29), {"AAPL": 0.09})]
        assert rc.weights_as_of(reports, date(2023, 12, 28)) == {}            # filed that day: not yet
        assert rc.weights_as_of(reports, date(2023, 12, 29)) == {"AAPL": 0.11}
        assert rc.weights_as_of(reports, date(2024, 3, 1)) == {"AAPL": 0.09}


# --------------------------------------------------------------------------
# earnings releases
# --------------------------------------------------------------------------

def _sub(rows):
    keys = ("accessionNumber", "filingDate", "form", "items", "acceptanceDateTime")
    return {"filings": {"recent": {k: [r[i] for r in rows] for i, k in enumerate(keys)}}}


class TestEarnings:
    def test_item_202_originals_only(self):
        page = _sub([("a1", "2024-08-01", "8-K", "2.02,9.01", "x"), ("a2", "2024-08-05", "8-K/A", "2.02", "x"),
                     ("a3", "2024-08-06", "8-K", "5.02", "x"), ("a4", "2024-08-07", "10-Q", "", "x"),
                     ("a5", "2024-10-31", "8-K", "2.02", "x"), ("a1", "2024-08-01", "8-K", "2.02", "x")])
        got = rc.item_202_filings([page])
        assert [f.accession for f in got] == ["a1", "a5"]

    def test_an_older_files_page_has_the_same_shape(self):
        older = {k: v for k, v in _sub([("b1", "2012-01-24", "8-K", "2.02", "x")])["filings"]["recent"].items()}
        assert [f.accession for f in rc.item_202_filings([older])] == ["b1"]

    def test_real_apple_submissions_and_index_page(self):
        """Trimmed real EDGAR documents: Apple's 2024-H2 filings (8-K, 10-Q, Form 4, ...) contain
        exactly the two results releases, and the index page gives the ET acceptance time that the
        JSON's acceptanceDateTime ('2024-08-02T00:30:26.000Z') does not."""
        sub = json.loads(_fx("sub_aapl_2024h2.json"))
        got = rc.item_202_filings([sub])
        assert [(f.accession, f.filing_date) for f in got] == [
            ("0000320193-24-000080", date(2024, 8, 1)), ("0000320193-24-000120", date(2024, 10, 31))]
        i = sub["filings"]["recent"]["accessionNumber"].index("0000320193-24-000080")
        assert sub["filings"]["recent"]["acceptanceDateTime"][i] == "2024-08-02T00:30:26.000Z"
        assert rc.parse_index_accepted(_fx("idx_aapl_20240801.htm")) == datetime(
            2024, 8, 1, 16, 30, 26, tzinfo=ET)

    def test_the_index_page_gives_eastern_time(self):
        page = ('<div class="infoHead">Accepted</div>\n<div class="info">2024-08-01 16:30:26</div>')
        got = rc.parse_index_accepted(page)
        assert got == datetime(2024, 8, 1, 16, 30, 26, tzinfo=ET)
        assert rc.parse_index_accepted("<html>no time here</html>") is None

    @pytest.mark.parametrize("accepted, expected", [
        (datetime(2024, 8, 1, 16, 30, tzinfo=ET), date(2024, 8, 2)),    # after the close -> next session
        (datetime(2024, 8, 1, 16, 0, tzinfo=ET), date(2024, 8, 2)),     # 16:00 counts as after
        (datetime(2024, 8, 1, 15, 59, tzinfo=ET), date(2024, 8, 1)),    # during the session
        (datetime(2024, 10, 8, 6, 0, tzinfo=ET), date(2024, 10, 8)),    # before the open
        (datetime(2024, 10, 7, 17, 49, tzinfo=ET), date(2024, 10, 8)),  # evening before (PepsiCo)
        (datetime(2024, 8, 3, 10, 0, tzinfo=ET), date(2024, 8, 5)),     # a Saturday -> Monday
        (datetime(2024, 8, 2, 18, 0, tzinfo=ET), date(2024, 8, 5)),     # Friday evening -> Monday
    ])
    def test_event_session(self, accepted, expected):
        sessions = [date(2024, 8, 1), date(2024, 8, 2), date(2024, 8, 5), date(2024, 10, 7), date(2024, 10, 8)]
        assert rc.event_session(accepted, sessions) == expected

    def test_past_the_calendar_is_none(self):
        assert rc.event_session(datetime(2030, 1, 2, 9, 0, tzinfo=ET), [date(2024, 1, 2)]) is None

    def test_release_merging_keeps_deliveries_and_results(self):
        """Tesla 2020: delivery reports in the first days of a quarter AND results ~3 weeks later are
        both Item 2.02 events; a same-release second filing within 3 days is merged."""
        fs = [rc.Filing("d1", date(2020, 1, 3)), rc.Filing("r1", date(2020, 1, 29)),
              rc.Filing("r1b", date(2020, 1, 31)), rc.Filing("d2", date(2020, 4, 2)),
              rc.Filing("r2", date(2020, 4, 29)), rc.Filing("x", date(2020, 4, 30))]
        kept, merged = rc.release_events(fs)
        assert [f.accession for f in kept] == ["d1", "r1", "d2", "r2"]
        assert [f.accession for f in merged] == ["r1b", "x"]

    def test_the_boundary_of_the_merge(self):
        kept, merged = rc.release_events([rc.Filing("a", date(2024, 1, 1)), rc.Filing("b", date(2024, 1, 4)),
                                          rc.Filing("c", date(2024, 1, 8))])
        assert [f.accession for f in kept] == ["a", "c"] and [f.accession for f in merged] == ["b"]


# --------------------------------------------------------------------------
# store + script
# --------------------------------------------------------------------------

class TestStore:
    def test_round_trips(self, tmp_path):
        s = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        s.save_econ_schedule([(date(2024, 1, 31), "FOMC", "u1")], SOURCE_FED_FOMC)
        s.save_econ_schedule([(date(2024, 1, 11), "CPI", "u2"), (date(2024, 2, 2), "NFP", "u3")], SOURCE_BLS_ARCHIVE)
        assert s.load_econ_schedule() == [(date(2024, 1, 11), "CPI"), (date(2024, 1, 31), "FOMC"),
                                          (date(2024, 2, 2), "NFP")]
        assert s.load_econ_schedule("FOMC") == [(date(2024, 1, 31), "FOMC")]
        s.save_qqq_holdings([(date(2023, 9, 30), date(2023, 12, 28), "N-30B-2", "acc", 1, "AAPL",
                              "Apple, Inc.", 21.0, 0.108)], SOURCE_EDGAR_N30B2)
        h = s.load_qqq_holdings()
        assert h.iloc[0][["filed_date", "company_key", "weight", "source"]].tolist() == [
            "2023-12-28", "AAPL", 0.108, SOURCE_EDGAR_N30B2]
        s.save_earnings_schedule([("acc1", "AAPL", 320193, date(2024, 8, 1),
                                   "2024-08-01T16:30:26-04:00", date(2024, 8, 2))], SOURCE_EDGAR_8K_202)
        e = s.load_earnings_schedule()
        assert e.iloc[0][["company_key", "event_session", "source"]].tolist() == [
            "AAPL", "2024-08-02", SOURCE_EDGAR_8K_202]


class TestScript:
    def test_no_prefix_is_needed_any_more(self, tmp_path):
        """W91: the empty-prefix guard is gone (the un-prefixed tables are the one shared set), so a run without
        --table-prefix gets as far as the cache lookup instead of exiting 2."""
        import build_research_calendars as b
        with pytest.raises(FileNotFoundError):
            b.main(["--what", "econ", "--db-path", str(tmp_path / "h.db"), "--offline",
                    "--cache-dir", str(tmp_path / "empty"), "--report-dir", str(tmp_path)])

    def test_a_missing_contact_refuses_to_fetch(self, tmp_path, monkeypatch):
        import build_research_calendars as b
        monkeypatch.delenv("SEC_EDGAR_CONTACT", raising=False)
        monkeypatch.setattr(b, "_root", tmp_path)           # no .env there
        with pytest.raises(SystemExit, match="SEC_EDGAR_CONTACT"):
            b._contact()

    def test_offline_without_cache_fails_before_writing(self, tmp_path):
        import build_research_calendars as b
        db = tmp_path / "h.db"
        with pytest.raises(FileNotFoundError):
            b.main(["--what", "econ", "--table-prefix", "research_", "--db-path", str(db), "--offline",
                    "--cache-dir", str(tmp_path / "empty"), "--report-dir", str(tmp_path)])
        assert not HistoricalDataStore(db_path=db, table_prefix="research_").load_econ_schedule()

    def test_failed_validation_writes_nothing(self, tmp_path, monkeypatch):
        import build_research_calendars as b
        db = tmp_path / "h.db"
        monkeypatch.setattr(b, "build_econ", lambda f, d, p, s: ([(date(2024, 1, 31), "FOMC", "u")], {},
                                                                  ["econ: forced failure"]))
        rc_ = b.main(["--what", "econ", "--table-prefix", "research_", "--db-path", str(db), "--offline",
                      "--cache-dir", str(tmp_path / "c"), "--report-dir", str(tmp_path)])
        assert rc_ == 1
        assert HistoricalDataStore(db_path=db, table_prefix="research_").load_econ_schedule() == []

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        import build_research_calendars as b
        db = tmp_path / "h.db"
        monkeypatch.setattr(b, "build_econ", lambda f, d, p, s: ([(date(2024, 1, 31), "FOMC", "u")], {}, []))
        assert b.main(["--what", "econ", "--table-prefix", "research_", "--db-path", str(db), "--offline",
                       "--cache-dir", str(tmp_path / "c"), "--report-dir", str(tmp_path), "--dry-run"]) == 0
        assert HistoricalDataStore(db_path=db, table_prefix="research_").load_econ_schedule() == []
        assert b.main(["--what", "econ", "--table-prefix", "research_", "--db-path", str(db), "--offline",
                       "--cache-dir", str(tmp_path / "c"), "--report-dir", str(tmp_path)]) == 0
        assert HistoricalDataStore(db_path=db, table_prefix="research_").load_econ_schedule() == [
            (date(2024, 1, 31), "FOMC")]
        audit = json.loads(next(tmp_path.glob("research_calendars_audit_*.json")).read_text())
        assert audit["failures"] == []


def test_the_live_tree_never_imports_the_calendars():
    """AST: only research code (backtesting/, ml/, data/research_calendars.py itself) may import it.

    Track D (deliberate update of this guard, not a bypass): the Fed and BLS parsers moved out of
    research_calendars into ait.data.official_calendar_sources, the IO-free module the live economic calendar's
    table shares with research (pinned by tests/test_w117_official_calendar.py). What stays here, and stays
    research-only, is the strike-aware study's QQQ-holdings and earnings parsing; the import restriction below is
    unchanged for it."""
    hits = []
    for p in (ROOT / "src" / "ait").rglob("*.py"):
        rel = p.relative_to(ROOT / "src" / "ait").as_posix()
        if rel.startswith(("backtesting/", "ml/")) or rel == "data/research_calendars.py":
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""] if isinstance(n, ast.ImportFrom) else [])
            if any("research_calendars" in m for m in mods) or (
                    isinstance(n, ast.ImportFrom) and any(a.name == "research_calendars" for a in n.names)):
                hits.append(f"{rel}:{n.lineno}")
    assert hits == []


def test_the_module_does_no_io():
    """The parsers take documents, never fetch: no network or file module is imported."""
    tree = ast.parse((ROOT / "src" / "ait" / "data" / "research_calendars.py").read_text())
    mods = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not mods & {"urllib", "requests", "aiohttp", "socket", "sqlite3", "os", "pathlib", "http"}


def test_the_fomc_and_bls_parsers_are_the_shared_ones():
    """Track D: one definition of the Fed / BLS parsers; this module only re-exports them for the W83a script."""
    from ait.data import official_calendar_sources as shared
    assert rc.parse_fomc_year_page is shared.parse_fomc_year_page
    assert rc.parse_fomc_calendars is shared.parse_fomc_calendars
    assert rc.parse_bls_archive is shared.parse_bls_archive

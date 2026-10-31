"""W116 WP1: the preflight checks every input a study needs against the stored history, read-only.

Each input has a green case and a red case on a temporary history database seeded like the real one (IB-tagged
bars, stored IV, VIX in ``daily_iv``). A red case must name the input, the symbol, what was needed, what was found and
the command that closes the gap.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import pytest

from ait.data import research_inputs as ri
from ait.data.historical import (
    SOURCE_IB_TRADES_RTH,
    SOURCE_INVESTING_CSV,
    SOURCE_YAHOO_DIVIDENDS,
    SOURCE_YAHOO_UNADJ,
    HistoricalDataStore,
)
from tests.data import _w116_store_golden_inputs as gi

LAST = gi.LAST_SESSION                         # the fixture's last stored session
TODAY = LAST + timedelta(days=1)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "historical.db"
    gi.seed(path)
    return path


def _req(db, **kw):
    last = LAST
    base = dict(window=ri.Window(date(2022, 6, 1), last), symbols=("SPY", "QQQ"), db_path=db,
                max_null_iv_run=3, max_dividend_gap_days=100, max_event_gap_days=120)
    base.update(kw)
    return ri.ResearchRequest(**base)


def _sql(db, statement, params=()):
    with sqlite3.connect(db) as conn:
        conn.execute(statement, params)


def _gaps(report, name):
    return [g for g in report.gaps if g.input == name]


class TestPrices:
    def test_complete_bars_pass(self, db):
        r = ri.check_research_inputs(_req(db))
        assert r.ok and "daily_prices" in r.checked and "implied_vol" in r.checked

    def test_a_missing_session_is_named(self, db):
        _sql(db, "DELETE FROM daily_prices WHERE symbol='QQQ' AND date='2023-03-15'")
        (g,) = _gaps(ri.check_research_inputs(_req(db)), "daily_prices")
        assert g.symbol == "QQQ" and "1 session(s) missing, first 2023-03-15" in g.found
        assert "backfill_historical_data.py --symbols QQQ --mode daily" in g.remedy

    def test_a_stale_tail_is_a_gap(self, db):
        later = ri.required_window(today=LAST + timedelta(days=14), start_date=date(2022, 6, 1))
        (g, *_) = _gaps(ri.check_research_inputs(_req(db, window=later)), "daily_prices")
        assert "session(s) missing" in g.found and "--mode daily" in g.remedy

    def test_a_foreign_source_tag_is_a_gap(self, db):
        _sql(db, "UPDATE daily_prices SET source=? WHERE symbol='SPY' AND date='2023-01-03'", (SOURCE_YAHOO_UNADJ,))
        gaps = _gaps(ri.check_research_inputs(_req(db)), "daily_prices")
        assert any("YAHOO_UNADJ" in g.found and g.symbol == "SPY" for g in gaps)

    def test_an_untagged_row_is_a_gap(self, db):
        _sql(db, "UPDATE daily_prices SET source=NULL WHERE symbol='SPY' AND date='2023-01-03'")
        assert any("UNRECORDED" in g.found for g in _gaps(ri.check_research_inputs(_req(db)), "daily_prices"))

    def test_context_symbols_are_checked_over_the_context_window(self, db):
        wide = ri.Window(date(2021, 1, 4), ri.last_completed_session(TODAY))           # before the stored history
        r = ri.check_research_inputs(_req(db, context_symbols=("SPY",), context_window=wide, symbols=("QQQ",)))
        assert any(g.symbol == "SPY" for g in _gaps(r, "daily_prices"))


class TestImpliedVol:
    def test_an_isolated_missing_iv_is_a_counted_no_entry_day(self, db):
        _sql(db, "UPDATE daily_prices SET implied_vol=NULL WHERE symbol='SPY' AND date='2023-02-01'")
        r = ri.check_research_inputs(_req(db))
        assert r.ok and r.notes["iv_no_entry_sessions"] == {"SPY": ["2023-02-01"]}

    def test_a_long_run_without_iv_is_a_gap(self, db):
        _sql(db, "UPDATE daily_prices SET implied_vol=NULL WHERE symbol='SPY' AND date BETWEEN '2023-02-01' AND '2023-02-06'")
        (g,) = _gaps(ri.check_research_inputs(_req(db)), "implied_vol")
        assert g.symbol == "SPY" and "run of 4 consecutive" in g.found and "--mode iv" in g.remedy

    def test_a_symbol_with_no_iv_at_all_is_a_gap(self, db):
        _sql(db, "UPDATE daily_prices SET implied_vol=NULL WHERE symbol='QQQ'")
        (g,) = _gaps(ri.check_research_inputs(_req(db)), "implied_vol")
        assert g.symbol == "QQQ" and "no stored implied vol" in g.found

    def test_the_check_can_be_switched_off(self, db):
        _sql(db, "UPDATE daily_prices SET implied_vol=NULL")
        assert "implied_vol" not in ri.check_research_inputs(_req(db, needs_iv=False)).checked


class TestVix:
    def test_a_complete_vix_series_passes(self, db):
        r = ri.check_research_inputs(_req(db, vix_symbols=("VIX",)))
        assert r.ok and "vix_family" in r.checked

    def test_a_short_vix_series_is_a_gap_with_the_vix_command(self, db):
        wide = ri.Window(date(2010, 1, 4), ri.last_completed_session(TODAY))
        (g,) = _gaps(ri.check_research_inputs(_req(db, vix_symbols=("VIX",), context_window=wide)), "vix_family")
        assert g.symbol == "VIX" and "--mode vix --symbols VIX" in g.remedy

    def test_a_vix_member_that_was_never_stored_is_a_gap(self, db):
        (g,) = _gaps(ri.check_research_inputs(_req(db, vix_symbols=("VIX9D",))), "vix_family")
        assert g.symbol == "VIX9D"

    def test_a_foreign_vix_tag_is_a_gap(self, db):
        _sql(db, "UPDATE daily_iv SET source='YAHOO' WHERE symbol='VIX' AND date='2023-01-03'")
        assert any("YAHOO" in g.found for g in _gaps(ri.check_research_inputs(_req(db, vix_symbols=("VIX",))),
                                                     "vix_family"))


class TestDividends:
    def _seed(self, db, sym="SPY", last=None, source=SOURCE_YAHOO_DIVIDENDS):
        last = last or str(LAST - timedelta(days=40))
        HistoricalDataStore(db_path=db).save_dividends(sym, [(date(2022, 3, 18), 1.5), (date.fromisoformat(last), 1.6)],
                                                      source, "2024-01-01")

    def test_current_dividends_pass(self, db):
        self._seed(db, "SPY")
        self._seed(db, "QQQ")
        assert ri.check_research_inputs(_req(db, ex_div_gate=True)).ok

    def test_no_rows_is_a_gap(self, db):
        gaps = _gaps(ri.check_research_inputs(_req(db, ex_div_gate=True)), "dividends")
        assert {g.symbol for g in gaps} == {"SPY", "QQQ"} and all("backfill_dividends.py" in g.remedy for g in gaps)

    def test_a_stale_table_is_a_gap(self, db):
        self._seed(db, "SPY", last="2023-06-16")
        self._seed(db, "QQQ")
        (g,) = _gaps(ri.check_research_inputs(_req(db, ex_div_gate=True)), "dividends")
        assert g.symbol == "SPY" and "2023-06-16" in g.found

    def test_a_foreign_tag_is_a_gap(self, db):
        self._seed(db, "SPY", source="MANUAL")
        self._seed(db, "QQQ")
        assert any("MANUAL" in g.found for g in _gaps(ri.check_research_inputs(_req(db, ex_div_gate=True)), "dividends"))

    def test_not_checked_when_the_gate_is_off(self, db):
        assert "dividends" not in ri.check_research_inputs(_req(db)).checked


class TestEconEvents:
    def _seed(self, db, types=("FOMC", "CPI"), first="2021-12-01", last="2026-12-31", source=SOURCE_INVESTING_CSV,
              extra_rows=()):
        HistoricalDataStore(db_path=db)
        rows = [(d, t) for t in types for d in (first, last)] + list(extra_rows)
        for d, t in rows:
            _sql(db, "INSERT INTO econ_events (event_date, event, country, engine_event_type, source) "
                     "VALUES (?,?,?,?,?)", (d, f"{t} {d}", "United States", t, source))

    def test_events_inside_the_window_and_an_export_that_covers_it_pass(self, db):
        self._seed(db, extra_rows=[("2023-06-14", "FOMC"), ("2023-06-13", "CPI")])
        assert ri.check_research_inputs(_req(db, macro_gate=True, macro_event_types=("FOMC", "CPI"))).ok

    def test_a_type_that_first_occurs_after_the_window_start_is_not_a_gap(self, db):
        """The first FOMC of a year is weeks into it: the export's own start, not the type's, is the bound."""
        self._seed(db, types=("FOMC",), first="2021-12-01", last="2026-12-31",
                   extra_rows=[("2022-07-27", "FOMC"), ("2023-09-20", "FOMC")])
        assert ri.check_research_inputs(_req(db, macro_gate=True, macro_event_types=("FOMC",),
                                             window=ri.Window(date(2022, 6, 1), LAST))).ok

    def test_a_type_with_no_rows_is_a_gap(self, db):
        self._seed(db, types=("FOMC",))
        gaps = _gaps(ri.check_research_inputs(_req(db, macro_gate=True, macro_event_types=("FOMC", "NFP"))),
                     "econ_events")
        assert any(g.symbol == "NFP" and "load_econ_calendar_csv.py" in g.remedy for g in gaps)

    def test_an_export_that_starts_after_the_window_start_is_a_gap(self, db):
        self._seed(db, first="2023-01-03", extra_rows=[("2023-06-14", "FOMC")])
        gaps = _gaps(ri.check_research_inputs(_req(db, macro_gate=True, macro_event_types=("FOMC",))), "econ_events")
        assert any(g.symbol is None and "starts on or before" in g.required and "2023-01-03" in g.found for g in gaps)

    def test_a_type_whose_last_event_is_long_before_the_window_ends_is_a_gap(self, db):
        self._seed(db, types=("FOMC",), first="2021-12-01", last="2023-01-25", extra_rows=[("2025-01-01", "CPI")])
        gaps = _gaps(ri.check_research_inputs(_req(db, macro_gate=True, macro_event_types=("FOMC",))), "econ_events")
        assert any(g.symbol == "FOMC" and "2023-01-25" in g.found for g in gaps)

    def test_a_foreign_tag_is_a_gap(self, db):
        self._seed(db, source="HARDCODED", extra_rows=[("2023-06-14", "FOMC"), ("2023-06-13", "CPI")])
        assert any("HARDCODED" in g.found for g in _gaps(
            ri.check_research_inputs(_req(db, macro_gate=True, macro_event_types=("FOMC", "CPI"))), "econ_events"))


class TestEarnings:
    def test_exempt_symbols_need_no_rows_and_are_recorded(self, db):
        r = ri.check_research_inputs(_req(db, earnings_gate=True, earnings_exempt=("SPY", "QQQ")))
        assert r.ok and r.notes["earnings_exempt"] == ["SPY", "QQQ"]

    def test_a_symbol_neither_exempt_nor_scheduled_is_a_gap(self, db):
        (g,) = _gaps(ri.check_research_inputs(_req(db, earnings_gate=True, earnings_exempt=("SPY",))), "earnings")
        assert g.symbol == "QQQ" and "earnings_exempt_symbols" in g.remedy

    def test_a_scheduled_symbol_covering_the_window_passes(self, db):
        HistoricalDataStore(db_path=db)
        for d in ("2022-05-01", str(LAST + timedelta(days=5))):
            _sql(db, "INSERT INTO earnings_schedule (accession, company_key, cik, filing_date, accepted_et, "
                     "event_session, source) VALUES (?,?,?,?,?,?,?)", (f"a{d}", "QQQ", 1, d, d, d, "EDGAR"))
        assert ri.check_research_inputs(_req(db, earnings_gate=True, earnings_exempt=("SPY",))).ok


class TestFriction:
    def _params(self, db, sym, table="option_leg_spread_params"):
        HistoricalDataStore(db_path=db)
        cols = "(symbol, form, a, b, n, n_sessions, calibrated_on" + (", dte_band" if "band" in table else "") + ")"
        vals = "(?,?,?,?,?,?,?" + (",?" if "band" in table else "") + ")"
        extra = ("0-5",) if "band" in table else ()
        _sql(db, f"INSERT INTO {table} {cols} VALUES {vals}", (sym, "linear", 0.1, 0.0, 10, 20, "2024-01-01", *extra))

    def test_params_present_pass(self, db):
        self._params(db, "SPY")
        self._params(db, "QQQ")
        assert ri.check_research_inputs(_req(db, friction_per_leg=True)).ok

    def test_missing_params_are_a_gap(self, db):
        self._params(db, "SPY")
        (g,) = _gaps(ri.check_research_inputs(_req(db, friction_per_leg=True)), "friction_params")
        assert g.symbol == "QQQ" and "calibrate_option_spreads.py" in g.remedy

    def test_missing_bands_are_a_declared_extrapolation_not_a_gap(self, db):
        self._params(db, "SPY")
        self._params(db, "QQQ")
        self._params(db, "QQQ", table="option_leg_spread_band_params")
        r = ri.check_research_inputs(_req(db, friction_per_leg=True, friction_bands=True))
        assert r.ok and r.notes["friction_bands_extrapolated"] == ["SPY"]


class TestIntradayAndOptionBars:
    def _bars(self, db, sym, days):
        HistoricalDataStore(db_path=db)
        for d in days:
            _sql(db, "INSERT INTO intraday_prices (symbol, datetime, interval, open, high, low, close, volume, source) "
                     "VALUES (?,?,?,?,?,?,?,?,?)", (sym, f"{d} 10:00:00", "5m", 1, 1, 1, 1, 1, "TRADES"))

    def test_full_intraday_coverage_passes_and_partial_is_a_gap(self, db):
        sessions = ri.sessions_between(date(2024, 1, 2), date(2024, 5, 17))
        self._bars(db, "SPY", [str(d) for d in sessions])
        self._bars(db, "QQQ", [str(d) for d in sessions[:10]])
        win = ri.Window(date(2024, 1, 2), date(2024, 5, 17))
        gaps = _gaps(ri.check_research_inputs(_req(db, window=win, intraday=True, intraday_min_coverage=0.95)),
                     "intraday_prices")
        assert [g.symbol for g in gaps] == ["QQQ"] and "intraday_features" in gaps[0].remedy

    def test_option_bars_required_when_the_surface_is_on(self, db):
        HistoricalDataStore(db_path=db)
        (g, *_) = _gaps(ri.check_research_inputs(_req(db, surface=True)), "option_daily_bars")
        assert "backfill_option_history.py" in g.remedy


class TestWholeReport:
    def test_every_gap_is_listed_at_once(self, db):
        _sql(db, "DELETE FROM daily_prices WHERE symbol='SPY' AND date='2023-03-15'")
        r = ri.check_research_inputs(_req(db, ex_div_gate=True, vix_symbols=("VIX9D",)))
        assert {g.input for g in r.gaps} >= {"daily_prices", "dividends", "vix_family"}
        with pytest.raises(ri.ResearchDataUnavailable) as e:
            ri.preflight_or_raise(_req(db, ex_div_gate=True, vix_symbols=("VIX9D",)), "unit test")
        assert len(e.value.gaps) == len(r.gaps) and "unit test" in str(e.value)

    def test_a_missing_database_is_one_clear_gap(self, tmp_path):
        r = ri.check_research_inputs(_req(tmp_path / "nope.db"))
        assert [g.input for g in r.gaps] == ["history_database"]

    def test_an_empty_database_reports_missing_tables_and_creates_nothing(self, tmp_path):
        path = tmp_path / "empty.db"
        sqlite3.connect(path).close()
        before = path.read_bytes()
        r = ri.check_research_inputs(_req(path, vix_symbols=("VIX",), ex_div_gate=True))
        assert {g.input for g in r.gaps} >= {"daily_prices", "vix_family", "dividends"}
        assert all("does not exist" in g.found for g in r.gaps)
        assert path.read_bytes() == before

    def test_the_report_is_json_ready(self, db):
        import json
        r = ri.check_research_inputs(_req(db))
        assert json.loads(json.dumps(r.as_dict()))["ok"] is True

    def test_a_complete_store_leaves_the_file_untouched(self, db):
        before = db.read_bytes()
        ri.check_research_inputs(_req(db, vix_symbols=("VIX",)))
        assert db.read_bytes() == before


class TestRegistry:
    def test_every_registered_table_is_one_the_store_creates(self, tmp_path):
        path = tmp_path / "h.db"
        HistoricalDataStore(db_path=path)
        with sqlite3.connect(path) as conn:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert ri.registered_tables() <= names

    def test_input_names_are_unique(self):
        names = [e.name for e in ri.REGISTRY]
        assert len(names) == len(set(names))


# --------------------------------------------------------------------------- class guard: nothing is read unregistered

# every history-store reader the research modules call -> the (unprefixed) table it reads
READERS = {
    "load": "daily_prices",
    "load_daily_iv": "daily_prices",
    "load_daily_iv_bars": "daily_iv",
    "load_dividends": "dividends",
    "load_earnings_sessions": "earnings_schedule",
    "econ_events_span": "econ_events",
    "load_econ_events": "econ_events",
    "load_intraday": "intraday_prices",
    "load_intraday_range": "intraday_prices",
    "intraday_day_count": "intraday_prices",
    "resample_to_daily": "intraday_prices",
    "load_leg_spread_model": "option_leg_spread_params",
    "load_leg_spread_band_models": "option_leg_spread_band_params",
    "load_spread_params": "option_spread_params",
}
# tables a research module reads that are DECLARED not to be a registered input, with the reason
NOT_INPUT = {
    "option_spread_params": "the legacy calibrated-spread model; only used when the per-leg friction model is off, "
                            "and a missing table there is the legacy config formula, declared in the policy page",
}
RESEARCH_MODULES = ("src/ait/backtesting", "src/ait/optimization", "src/ait/data/market_data.py",
                    "src/ait/data/iv_surface.py", "src/ait/ml")


def _research_files():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for rel in RESEARCH_MODULES:
        p = root / rel
        yield from ([p] if p.is_file() else sorted(p.rglob("*.py")))


def _store_calls():
    import ast
    found = {}
    for path in _research_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                recv = ast.unparse(node.func.value)
                if "store" in recv.lower() and node.func.attr in READERS:
                    found.setdefault(node.func.attr, set()).add(path.name)
    return found


def _unmapped_store_methods():
    """Public methods of HistoricalDataStore that look like readers but are not mapped in READERS."""
    import inspect
    names = {n for n, f in inspect.getmembers(HistoricalDataStore, inspect.isfunction)
             if not n.startswith("_") and (n.startswith("load") or n.endswith("_span") or n.startswith("intraday_")
                                           or n.startswith("get_") or n == "resample_to_daily")}
    return names - set(READERS)


class TestEveryResearchReadIsRegistered:
    def test_each_table_a_research_module_reads_is_a_registered_input_or_declared_not_one(self):
        read_tables = {READERS[m] for m in _store_calls()}
        missing = {t for t in read_tables if t not in ri.registered_tables() and t not in NOT_INPUT}
        assert not missing, f"research code reads {sorted(missing)} but no registry entry checks it"

    def test_the_scan_sees_the_known_readers(self):
        calls = _store_calls()
        assert {"load", "load_daily_iv", "load_dividends", "load_leg_spread_model", "intraday_day_count"} <= set(calls)

    def test_a_new_store_reader_must_be_mapped_here_so_its_table_is_classified(self):
        # new reader methods on the store that nobody has classified: the list below is the known, deliberate
        # exceptions (live-only or ingest-only readers, never called from research modules)
        called = set(_store_calls())
        live_or_ingest_only = {n for n in _unmapped_store_methods()}
        import ast
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        offenders = set()
        for path in _research_files():
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and "store" in ast.unparse(node.func.value).lower() and node.func.attr in live_or_ingest_only):
                    offenders.add((path.name, node.func.attr))
        assert not offenders, f"research modules call unclassified store readers: {sorted(offenders)}"
        assert called

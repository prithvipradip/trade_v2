"""W116 WP3: with ``strict_inputs`` the research engine and the walk-forward refuse where they used to substitute.

Every test here runs under the production policy (the module is marked ``strict_inputs``; the rest of the suite runs
with it off because it builds engines on synthetic frames). Each substitution has a red case (refusal naming the gap
and the fix) and, where the synthetic-inputs mode keeps the old behaviour, a control proving it still does.
"""
from __future__ import annotations

import ast
import asyncio
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ait.backtesting.walkforward as wf_mod
from ait.backtesting import strict
from ait.backtesting.engine import Backtester
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.data.historical import HistoricalDataStore
from ait.data.research_inputs import ResearchDataUnavailable
from ait.strategies.iron_condor import IronCondor
from tests.data import _w116_store_golden_inputs as gi

pytestmark = pytest.mark.strict_inputs

ROOT = Path(__file__).resolve().parent.parent


def _bars(n=80, iv=0.2, with_iv=True):
    idx = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000.0}, index=idx)
    if with_iv:
        df["implied_vol"] = iv
    return df


def _partial_engine(strict_on: bool):
    """A partial engine (the suite's idiom): just enough for the methods under test."""
    eng = Backtester.__new__(Backtester)
    eng._strict = strict_on
    eng._symbol = "SPY"
    eng._VOL_MULT_DEFAULT = 1.0
    return eng


class TestTheDefault:
    def test_the_production_default_is_strict(self):
        assert strict.STRICT_INPUTS_DEFAULT is True
        assert strict.resolve_strict(None) is True and strict.resolve_strict(False) is False

    def test_an_engine_and_a_walk_forward_built_without_the_flag_are_strict(self):
        assert WalkForwardBacktester(["SPY"], ["iron_condor"], db_path=Path("x.db"))._strict is True


class TestImpliedVolIsStoredOrAbsent:
    def test_the_last_sessions_stored_iv_is_used(self):
        assert _partial_engine(True)._get_iv(_bars(), market_context={"vix": 30}) == pytest.approx(0.2)

    def test_a_session_without_stored_iv_has_none_not_a_proxy(self):
        hist = _bars()
        hist.iloc[-1, hist.columns.get_loc("implied_vol")] = np.nan
        # the old tiers: the stale earlier value (0.2) -> VIX proxy -> realized vol
        assert _partial_engine(True)._get_iv(hist, market_context={"vix": 30}) is None

    def test_no_iv_column_is_none_even_with_a_vix_context(self):
        assert _partial_engine(True)._get_iv(_bars(with_iv=False), market_context={"vix": 30}) is None

    def test_the_synthetic_inputs_mode_keeps_the_old_tiers(self):
        hist = _bars()
        hist.iloc[-1, hist.columns.get_loc("implied_vol")] = np.nan
        assert _partial_engine(False)._get_iv(hist, market_context={"vix": 30}) == pytest.approx(0.2)   # stale carry
        assert _partial_engine(False)._get_iv(_bars(with_iv=False), market_context={"vix": 30}) is not None


@pytest.fixture
def seeded(tmp_path):
    db = tmp_path / "historical.db"
    last = gi.seed(db)
    return db, last


class TestEndToEndOnAStoredHistory:
    def test_isolated_sessions_without_iv_become_counted_no_entry_days(self, seeded):
        db, last = seeded
        sessions = pd.bdate_range(gi.START, periods=gi.SESSIONS)[250:520:3]       # single-session holes, never a run
        with sqlite3.connect(db) as conn:
            conn.executemany("UPDATE daily_prices SET implied_vol=NULL WHERE symbol='QQQ' AND date=?",
                             [(str(d.date()),) for d in sessions])
        result = gi.run(db, reseed=False)
        funnel = result.entry_funnel
        assert funnel["QQQ"].get("builder:iv_missing", 0) > 0
        assert funnel["SPY"].get("builder:iv_missing", 0) == 0

    def test_a_long_hole_in_the_iv_stops_the_study_before_it_starts(self, seeded):
        db, last = seeded
        hole = pd.bdate_range(gi.START, periods=gi.SESSIONS)[300:340]
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE daily_prices SET implied_vol=NULL WHERE symbol='QQQ' AND date>=? AND date<=?",
                         (str(hole[0].date()), str(hole[-1].date())))
        with pytest.raises(ResearchDataUnavailable) as e:
            gi.run(db, reseed=False)
        gap = e.value.gaps[0]
        assert gap.input == "implied_vol" and gap.symbol == "QQQ" and "--mode iv" in gap.remedy


class TestCalendarIsStoredOrRefused:
    def test_a_macro_gate_without_a_calendar_refuses(self):
        with pytest.raises(ResearchDataUnavailable) as e:
            Backtester(data=_bars(), strategies=[IronCondor()], symbol="SPY", macro_event_gate=True,
                       friction_model="legacy", ex_div_entry_gate="off")
        assert e.value.gaps[0].input == "econ_events"

    def test_the_gate_off_explicitly_is_a_stated_decision(self):
        Backtester(data=_bars(), strategies=[IronCondor()], symbol="SPY", macro_event_gate=False,
                   friction_model="legacy", ex_div_entry_gate="off")

    def test_a_passed_calendar_is_used(self):
        from ait.data.economic_calendar import EconomicCalendar
        bt = Backtester(data=_bars(), strategies=[IronCondor()], symbol="SPY", macro_event_gate=True,
                        economic_calendar=EconomicCalendar(events=[]), friction_model="legacy", ex_div_entry_gate="off")
        assert bt._economic_cal is not None

    def test_the_synthetic_inputs_mode_still_builds_the_hard_coded_calendar(self):
        bt = Backtester(data=_bars(), strategies=[IronCondor()], symbol="SPY", macro_event_gate=True,
                        friction_model="legacy", ex_div_entry_gate="off", strict_inputs=False)
        assert bt._economic_cal is not None


class TestFrictionIsFittedOrRefused:
    def test_a_missing_per_leg_model_refuses(self, tmp_path):
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db)
        with pytest.raises(ResearchDataUnavailable) as e:
            Backtester(data=_bars(), strategies=[IronCondor()], symbol="SPY", db_path=db, macro_event_gate=False,
                       friction_model="per_leg_usd", ex_div_entry_gate="off")
        assert e.value.gaps[0].input == "friction_params" and "calibrate_option_spreads.py" in e.value.gaps[0].remedy

    def test_the_synthetic_inputs_mode_falls_to_the_legacy_model_with_a_warning(self, tmp_path):
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db)
        bt = Backtester(data=_bars(), strategies=[IronCondor()], symbol="SPY", db_path=db, macro_event_gate=False,
                        friction_model="per_leg_usd", ex_div_entry_gate="off", strict_inputs=False)
        assert bt._leg_spread_model is None


class TestSettingsAreLoadedOrRefused:
    def test_an_engine_that_cannot_load_the_settings_refuses(self, monkeypatch):
        import ait.config.settings as settings_mod

        def _missing(*a, **k):
            raise FileNotFoundError("Config file not found: config.yaml")
        monkeypatch.setattr(settings_mod, "load_settings", _missing)
        with pytest.raises(RuntimeError, match="settings could not be loaded"):
            Backtester(data=_bars(), strategies=[IronCondor()], symbol="SPY", macro_event_gate=False,
                       friction_model="legacy", ex_div_entry_gate="off")

    def test_the_synthetic_inputs_mode_runs_on_the_model_defaults(self, monkeypatch):
        import ait.config.settings as settings_mod
        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("x")))
        Backtester(data=_bars(), strategies=[IronCondor()], symbol="SPY", macro_event_gate=False,
                   friction_model="legacy", ex_div_entry_gate="off", strict_inputs=False)


class TestSurfaceIsBuiltOrRefused:
    def test_a_surface_that_cannot_be_built_refuses(self, tmp_path):
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db)
        with pytest.raises(ResearchDataUnavailable) as e:
            Backtester(data=_bars(), strategies=[IronCondor()], symbol="SPY", db_path=db, macro_event_gate=False,
                       friction_model="legacy", ex_div_entry_gate="off", pricing_source="surface")
        assert e.value.gaps[0].input == "option_daily_bars" and "bs_legacy" in e.value.gaps[0].remedy


def _wf(tmp_path, **cfg):
    kw = dict(train_days=200, test_days=63, step_days=63, gap_days=5, macro_event_gate=False, friction_model="legacy",
              intraday_features=False)
    kw.update(cfg)
    return WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(**kw), db_path=tmp_path / "h.db")


class TestTheWalkForwardRefuses:
    def test_a_context_that_cannot_be_loaded_stops_the_run(self, tmp_path, monkeypatch):
        import ait.data.market_data as md

        def _boom(symbol, **kw):
            raise ResearchDataUnavailable([], context="context")
        monkeypatch.setattr(md, "load_daily_ohlcv", _boom)
        with pytest.raises(ResearchDataUnavailable):
            asyncio.run(_wf(tmp_path).run(data={"SPY": _bars(300)}))

    def test_the_synthetic_inputs_mode_runs_on_empty_context_frames(self, tmp_path, monkeypatch):
        import ait.data.market_data as md
        monkeypatch.setattr(md, "load_daily_ohlcv", lambda s, **kw: (_ for _ in ()).throw(RuntimeError("no store")))
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], db_path=tmp_path / "h.db", strict_inputs=False,
                                   config=WalkForwardConfig(macro_event_gate=False, friction_model="legacy"))
        wf._generate_windows = lambda d: []                                    # type: ignore[method-assign]
        asyncio.run(wf.run(data={"SPY": _bars(300)}))

    def test_a_symbol_that_cannot_be_fetched_stops_the_study(self, tmp_path, monkeypatch):
        import ait.data.market_data as md
        monkeypatch.setattr(md, "load_daily_ohlcv",
                            lambda s, **kw: (_ for _ in ()).throw(ResearchDataUnavailable([], context=s)))
        with pytest.raises(ResearchDataUnavailable):
            asyncio.run(_wf(tmp_path)._fetch_data())

    def test_a_symbol_with_too_few_rows_is_a_gap_not_a_silent_drop(self, tmp_path, monkeypatch):
        import ait.data.market_data as md
        monkeypatch.setattr(md, "load_daily_ohlcv", lambda s, **kw: _bars(30))
        with pytest.raises(ResearchDataUnavailable) as e:
            asyncio.run(_wf(tmp_path)._fetch_data())
        assert e.value.gaps[0].symbol == "SPY"
        # the synthetic-inputs mode drops it, as before
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], db_path=tmp_path / "h.db", strict_inputs=False)
        assert asyncio.run(wf._fetch_data()) == {}

    def test_a_macro_gate_with_no_event_types_refuses_instead_of_using_the_hard_coded_list(self, tmp_path):
        wf = _wf(tmp_path, macro_event_gate=True, macro_event_types=None)
        with pytest.raises(ResearchDataUnavailable) as e:
            wf._apply_research_calendar({"SPY": _bars()})
        assert "hard-coded" in e.value.gaps[0].found

    def test_an_empty_econ_events_table_refuses(self, tmp_path):
        HistoricalDataStore(db_path=tmp_path / "h.db")
        wf = _wf(tmp_path, macro_event_gate=True, macro_event_types=("FOMC",))
        with pytest.raises(ResearchDataUnavailable) as e:
            wf._apply_research_calendar({"SPY": _bars()})
        assert "load_econ_calendar_csv.py" in e.value.gaps[0].remedy

    def test_data_that_starts_before_the_calendar_refuses_instead_of_being_trimmed(self, tmp_path):
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db)
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT INTO econ_events (event_date, event, country, engine_event_type, source) "
                         "VALUES ('2024-03-20', 'FOMC', 'United States', 'FOMC', 'INVESTING_CSV')")
        wf = _wf(tmp_path, macro_event_gate=True, macro_event_types=("FOMC",))
        with pytest.raises(ResearchDataUnavailable) as e:
            wf._apply_research_calendar({"SPY": _bars()})
        assert "events start 2024-03-20" in e.value.gaps[0].found

    def test_short_intraday_coverage_refuses_unless_the_setting_says_daily_only(self, tmp_path):
        HistoricalDataStore(db_path=tmp_path / "h.db")
        wf = _wf(tmp_path, intraday_features=True)
        with pytest.raises(ResearchDataUnavailable) as e:
            wf._check_vlmc_coverage({"SPY": _bars()})
        assert "intraday_features to false" in e.value.gaps[0].remedy
        daily_only = _wf(tmp_path, intraday_features=False)
        daily_only._check_vlmc_coverage({"SPY": _bars()})
        assert daily_only._vlmc_enabled is False

    def test_the_synthetic_inputs_mode_auto_disables_intraday_features_as_before(self, tmp_path):
        HistoricalDataStore(db_path=tmp_path / "h.db")
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], db_path=tmp_path / "h.db", strict_inputs=False,
                                   config=WalkForwardConfig(macro_event_gate=False, friction_model="legacy"))
        wf._check_vlmc_coverage({"SPY": _bars()})
        assert wf._vlmc_enabled is False

    def test_a_walk_forward_without_a_database_reads_the_default_history_database(self):
        from ait.data.historical import DB_PATH
        assert WalkForwardBacktester(["SPY"], ["iron_condor"])._db_path == DB_PATH
        assert WalkForwardBacktester(["SPY"], ["iron_condor"], strict_inputs=False)._db_path is None


# --------------------------------------------------------------------------- class guards

def _src_files():
    for base in ("src", "scripts"):
        yield from sorted((ROOT / base).rglob("*.py"))
    yield from sorted(ROOT.glob("*.py"))


class TestNothingTurnsTheSwitchOff:
    def test_no_production_file_passes_strict_inputs_false(self):
        offenders = []
        for p in _src_files():
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(n, ast.keyword) and n.arg == "strict_inputs" \
                        and isinstance(n.value, ast.Constant) and n.value.value is False:
                    offenders.append(f"{p.relative_to(ROOT)}:{n.value.lineno}")
        assert not offenders, offenders

    def test_the_module_default_is_assigned_nowhere_in_production(self):
        offenders = []
        for p in _src_files():
            if p.name == "strict.py":
                continue
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                targets = []
                if isinstance(n, ast.Assign):
                    targets = n.targets
                elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
                    targets = [n.target]
                names = {getattr(t, "id", getattr(t, "attr", None)) for t in targets}
                setattr_call = (isinstance(n, ast.Call) and getattr(n.func, "id", None) == "setattr"
                                and any(isinstance(a, ast.Constant) and a.value == "STRICT_INPUTS_DEFAULT"
                                        for a in n.args))
                if "STRICT_INPUTS_DEFAULT" in names or setattr_call:
                    offenders.append(f"{p.relative_to(ROOT)}:{n.lineno}")
        assert not offenders, offenders

    def test_no_environment_variable_or_setting_controls_it(self):
        assert "STRICT_INPUTS" not in (ROOT / "config.yaml").read_text(encoding="utf-8")
        assert "strict_inputs" not in (ROOT / "src/ait/config/settings.py").read_text(encoding="utf-8")
        tree = ast.parse((ROOT / "src/ait/backtesting/strict.py").read_text(encoding="utf-8"))
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "os" not in imported and not ({"environ", "getenv"} & attrs)

"""W65 — enough clean research history to run an optimized walk-forward.

Root cause being pinned: `load_daily_ohlcv` resampled stored 5-min bars (QQQ/IWM had one
month) and ignored `daily_prices`; the walk-forward then fetched train+test+100 = 528 days
and built windows from the COMMON range of all symbols => 2 windows, 0 trades.

W65 adds an IB-sourced research history (research_* tables), provenance columns, a source
policy (IB first, Yahoo fallback, Polygon never for underlying bars), the operator's
economic-calendar CSV, and a VLMC-coverage guard. These tests drive the real code paths.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from ait.data.historical import (
    SOURCE_IB_OPTION_IV, SOURCE_IB_TRADES_RTH, SOURCE_INVESTING_CSV, SOURCE_UNRECORDED,
    SOURCE_YAHOO_UNADJ, HistoricalDataStore,
)
from ait.utils.time import today_et

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "ait"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"w65_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"w65_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _ohlcv(start: str, n: int, volume: int = 1_000, base: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = base + np.arange(n) * 0.1
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": volume}, index=idx)


# =========================================================================
# H. Provenance columns
# =========================================================================
class TestProvenanceColumns:
    def _pre_w65_db(self, path: Path) -> None:
        with sqlite3.connect(path) as c:
            c.execute("CREATE TABLE daily_prices (symbol TEXT NOT NULL, date TEXT NOT NULL, "
                      "open REAL, high REAL, low REAL, close REAL, volume INTEGER, "
                      "implied_vol REAL, PRIMARY KEY (symbol, date))")
            c.execute("INSERT INTO daily_prices VALUES ('SPY','2024-01-02',1,2,0.5,1.5,100,0.2)")
            c.execute("CREATE TABLE daily_iv (symbol TEXT NOT NULL, date TEXT NOT NULL, "
                      "iv_open REAL, iv_high REAL, iv_low REAL, iv_close REAL, "
                      "PRIMARY KEY (symbol, date))")

    def test_migration_adds_columns_without_touching_existing_rows(self, tmp_path):
        db = tmp_path / "h.db"
        self._pre_w65_db(db)
        HistoricalDataStore(db_path=db)
        with sqlite3.connect(db) as c:
            cols = [r[1] for r in c.execute("PRAGMA table_info(daily_prices)")]
            iv_cols = [r[1] for r in c.execute("PRAGMA table_info(daily_iv)")]
            row = c.execute("SELECT * FROM daily_prices").fetchall()
        assert "source" in cols and "iv_source" in cols and "source" in iv_cols
        assert row == [("SPY", "2024-01-02", 1.0, 2.0, 0.5, 1.5, 100, 0.2, None, None)]

    def test_untagged_save_never_erases_an_existing_tag(self, tmp_path):
        st = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        df = _ohlcv("2024-01-02", 1)
        st.save("SPY", df, source=SOURCE_IB_TRADES_RTH)
        st.save("SPY", df)                       # an older, untagged caller
        assert st.source_summary("SPY") == {SOURCE_IB_TRADES_RTH: (1, "2024-01-02", "2024-01-02")}

    def test_iv_writers_tag_both_iv_tables(self, tmp_path):
        st = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        idx = pd.to_datetime(["2024-01-02"])
        st.save("SPY", _ohlcv("2024-01-02", 1), source=SOURCE_IB_TRADES_RTH)
        st.save_daily_iv("SPY", pd.Series([0.25], index=idx), source=SOURCE_IB_OPTION_IV)
        st.save_daily_iv_bars("SPY", pd.DataFrame(
            {"Open": [0.2], "High": [0.3], "Low": [0.1], "Close": [0.25]}, index=idx),
            source=SOURCE_IB_OPTION_IV)
        with sqlite3.connect(tmp_path / "h.db") as c:
            assert c.execute("SELECT iv_source FROM research_daily_prices").fetchone() == (SOURCE_IB_OPTION_IV,)
            assert c.execute("SELECT source FROM research_daily_iv").fetchone() == (SOURCE_IB_OPTION_IV,)

    def test_legacy_rows_report_as_unrecorded_not_guessed(self, tmp_path):
        db = tmp_path / "h.db"
        self._pre_w65_db(db)
        st = HistoricalDataStore(db_path=db)
        assert st.source_summary("SPY") == {SOURCE_UNRECORDED: (1, "2024-01-02", "2024-01-02")}

    def test_intraday_day_count(self, tmp_path):
        st = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        rows = []
        for d in ("2024-03-04", "2024-03-05", "2024-03-07"):
            idx = pd.date_range(f"{d} 14:30", periods=6, freq="5min", tz="UTC")
            rows.append(pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                                      "Volume": 10}, index=idx))
        st.save_intraday("SPY", pd.concat(rows), source="TRADES")
        assert st.intraday_day_count("SPY", date(2024, 3, 1), date(2024, 3, 31)) == 3
        assert st.intraday_day_count("SPY", date(2024, 3, 5), date(2024, 3, 6)) == 1


# =========================================================================
# B. Loader: research tables first, legacy path unchanged
# =========================================================================
class TestResearchLoader:
    """W116: the loader reads the stored history of the research tables or refuses (no legacy path to fall to)."""

    def _seed(self, db: Path, n: int = 400, symbol: str = "SPY", prefix: str = "research_") -> pd.DatetimeIndex:
        from ait.data.research_inputs import last_completed_session, sessions_between
        st = HistoricalDataStore(db_path=db, table_prefix=prefix)
        end = last_completed_session(today_et())
        idx = pd.DatetimeIndex(pd.to_datetime(sessions_between(end - timedelta(days=int(n * 1.6)), end)[-n:]))
        df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5,
                           "Volume": 5_000}, index=idx)
        st.save(symbol, df, source=SOURCE_IB_TRADES_RTH)
        st.save_daily_iv(symbol, pd.Series(0.2, index=idx), source=SOURCE_IB_OPTION_IV)
        # IV-only skeleton row (no price) and TODAY's partial session
        st.save_daily_iv(symbol, pd.Series([0.3], index=[pd.Timestamp(end) + pd.Timedelta(days=1)]))
        st.save(symbol, df.tail(1).set_axis([pd.Timestamp(today_et())]),
                source=SOURCE_IB_TRADES_RTH)
        return idx

    def test_sufficient_research_history_is_used_with_real_iv(self, tmp_path):
        from ait.data.market_data import load_daily_ohlcv
        db = tmp_path / "h.db"
        idx = self._seed(db)
        with capture_logs() as logs:
            df = load_daily_ohlcv("SPY", db_path=db, table_prefix="research_", start_date=idx[0].date())
        assert len(df) == 400
        assert df["Close"].notna().all()                       # skeleton row dropped
        assert df.index.max() < pd.Timestamp(today_et())       # today's partial dropped
        assert df["implied_vol"].notna().all() and df["implied_vol"].iloc[0] == pytest.approx(0.2)
        assert any(e["event"] == "daily_ohlcv_from_research_store" for e in logs)

    def test_start_date_trims_and_sets_the_requested_span(self, tmp_path):
        from ait.data.market_data import load_daily_ohlcv
        db = tmp_path / "h.db"
        self._seed(db)
        start = today_et() - timedelta(days=200)
        df = load_daily_ohlcv("SPY", days=999, db_path=db, table_prefix="research_",
                              start_date=start)
        assert df.index.min() >= pd.Timestamp(start)
        assert 100 < len(df) < 160

    def test_short_research_history_refuses(self, tmp_path, monkeypatch):
        import ait.data.market_data as md
        from ait.data.research_inputs import ResearchDataUnavailable
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db, table_prefix="research_").save(
            "SPY", _ohlcv("2025-01-02", 30), source=SOURCE_IB_TRADES_RTH)
        asked = {"n": 0}
        monkeypatch.setattr(md, "yf", SimpleNamespace(Ticker=lambda s: asked.update(n=asked["n"] + 1)))
        with pytest.raises(ResearchDataUnavailable) as e:
            md.load_daily_ohlcv("SPY", days=400, db_path=db, table_prefix="research_")
        assert asked["n"] == 0 and e.value.gaps[0].input == "daily_prices"

    def test_the_prefix_selects_the_tables_and_the_other_set_is_never_read(self, tmp_path):
        import ait.data.market_data as md
        from ait.data.research_inputs import ResearchDataUnavailable
        db = tmp_path / "h.db"
        idx = self._seed(db, prefix="")                         # data only in the shared (unprefixed) tables
        assert len(md.load_daily_ohlcv("SPY", db_path=db, start_date=idx[0].date())) == 400
        with pytest.raises(ResearchDataUnavailable):
            md.load_daily_ohlcv("SPY", db_path=db, table_prefix="research_", start_date=idx[0].date())


# =========================================================================
# C. Walk-forward plumbing
# =========================================================================
def _wf(tmp_path=None, **cfg_kw):
    from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
    cfg = WalkForwardConfig(**cfg_kw)
    kw = {}
    if tmp_path is not None:
        kw = dict(db_path=tmp_path / "h.db", table_prefix="research_")
    return WalkForwardBacktester(["SPY", "QQQ", "IWM"], ["iron_condor"], config=cfg, **kw)


class TestWalkForwardHistory:
    def test_context_days(self):
        start = today_et() - timedelta(days=1000)
        assert _wf(history_start_date=start)._context_days() == 1000 + 400
        assert _wf(history_days=1826)._context_days() == 1826 + 400
        assert _wf()._context_days() == 365 + 900                    # legacy

    async def test_fetch_passes_prefix_start_and_a_span_that_covers_it(self, tmp_path, monkeypatch):
        import ait.data.market_data as md
        calls = []

        def _fake(symbol, days=None, db_path=None, table_prefix="", start_date=None,
                  end_date=None):
            calls.append(dict(symbol=symbol, days=days, table_prefix=table_prefix,
                              start_date=start_date, end_date=end_date))
            return _ohlcv("2023-06-01", 700)

        monkeypatch.setattr(md, "load_daily_ohlcv", _fake)
        start = date(2024, 1, 1)
        bt = _wf(tmp_path, history_start_date=start)
        data = await bt._fetch_data()
        assert {c["table_prefix"] for c in calls} == {"research_"}
        assert {c["end_date"] for c in calls} == {None}                              # unpinned: no end passed
        assert all(c["start_date"] == start for c in calls)
        assert all(c["days"] >= (today_et() - start).days + 30 for c in calls)
        assert all(df.index[0] >= pd.Timestamp(start) for df in data.values())  # trimmed

    def test_window_counts_pin_the_original_failure(self):
        bt = _wf()
        long_span = {s: _ohlcv("2024-01-02", 694) for s in ("SPY", "QQQ", "IWM")}
        short_span = {s: _ohlcv("2025-01-02", 353) for s in ("SPY", "QQQ", "IWM")}  # 528-day fetch
        assert len(bt._generate_windows(long_span)) >= 9
        assert len(bt._generate_windows(short_span)) <= 2

    def test_legacy_walkforward_config_defaults_are_untouched(self):
        from ait.backtesting.walkforward import WalkForwardConfig
        c = WalkForwardConfig()
        assert c.history_start_date is None and c.history_days is None
        assert c.macro_event_types == ('FOMC', 'CPI', 'NFP', 'PCE', 'GDP') and c.vlmc_min_coverage == pytest.approx(0.95)   # W116: the stored research calendar by default


class TestVlmcConsistencyGuard:
    def _seed_intraday(self, db: Path, days: list[pd.Timestamp]) -> None:
        st = HistoricalDataStore(db_path=db, table_prefix="research_")
        frames = []
        for d in days:
            idx = pd.date_range(f"{d.date()} 14:30", periods=6, freq="5min", tz="UTC")
            frames.append(pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0,
                                        "Volume": 10}, index=idx))
        st.save_intraday("SPY", pd.concat(frames), source="TRADES")

    def test_low_coverage_disables_vlmc_everywhere_with_a_warning(self, tmp_path):
        df = _ohlcv("2025-01-02", 100)
        self._seed_intraday(tmp_path / "h.db", list(df.index[:50]))          # 50%
        bt = _wf(tmp_path)
        assert bt._make_intraday_store() is not None                         # legacy: db_path => store
        with capture_logs() as logs:
            bt._check_vlmc_coverage({"SPY": df})
        assert bt._vlmc_enabled is False and bt._make_intraday_store() is None
        assert any(e["event"] == "vlmc_disabled_low_intraday_coverage" for e in logs)

    def test_full_coverage_keeps_vlmc_on(self, tmp_path):
        df = _ohlcv("2025-01-02", 100)
        self._seed_intraday(tmp_path / "h.db", list(df.index))
        bt = _wf(tmp_path)
        bt._check_vlmc_coverage({"SPY": df})
        assert bt._vlmc_enabled is True and bt._make_intraday_store() is not None

    def test_no_db_path_means_no_store(self):
        assert _wf()._make_intraday_store() is None

    def test_worker_tuple_carries_prefix_and_flag(self):
        src = (SRC / "backtesting" / "walkforward.py").read_text()
        tree = ast.parse(src)
        worker = next(n for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) and n.name == "_window_task_mp")
        names = {a.id for a in ast.walk(worker) if isinstance(a, ast.Name)}
        assert {"table_prefix", "vlmc_enabled", "research_calendar"} <= names


# =========================================================================
# Config
# =========================================================================
class TestConfigKeys:
    def test_real_config_values(self):
        # W68: the research history keys live in the research overlay (config_research.yaml),
        # not in the live config.yaml.
        from ait.config.research import load_research_settings
        b = load_research_settings(str(ROOT / "config.yaml")).backtest
        assert b.history_start_date == date(2024, 1, 1)
        assert b.research_table_prefix == ""          # W91: the one shared set of tables
        assert b.research_history is True
        assert b.macro_event_types == ["FOMC", "CPI", "NFP", "PCE", "GDP"]
        assert b.vlmc_min_coverage == pytest.approx(0.95)

    @pytest.mark.parametrize("bad", [
        {"macro_event_types": ["XYZ"]}, {"macro_event_types": []},
        {"research_table_prefix": "Bad-Prefix"}, {"vlmc_min_coverage": 0.2},
        {"history_days": 10}, {"bogus_key": 1},
    ])
    def test_invalid_values_are_startup_errors(self, bad):
        from pydantic import ValidationError
        from ait.config.settings import BacktestConfig
        with pytest.raises(ValidationError):
            BacktestConfig(**bad)

    def test_empty_prefix_means_live_tables(self):
        from ait.config.settings import BacktestConfig
        assert BacktestConfig(research_table_prefix="").research_table_prefix == ""


# =========================================================================
# A. Backfill script (--mode daily / research)
# =========================================================================
class _FakeIB:
    def __init__(self, bars):
        self._bars = bars
        self.requests = []

    async def qualifyContractsAsync(self, c):
        return [c]

    async def reqHistoricalDataAsync(self, *a, **k):
        self.requests.append(k)
        return list(self._bars)


def _bars(n: int, volume: float = 1_000_000.0):
    d0 = date(2025, 1, 2)
    out, d = [], d0
    while len(out) < n:
        if d.weekday() < 5:
            out.append(SimpleNamespace(date=d, open=100.0, high=101.0, low=99.0,
                                       close=100.5, volume=volume))
        d += timedelta(days=1)
    return out


class TestBackfillDailyMode:
    def test_modes_exist(self):
        mod = _load_script("backfill_historical_data")
        a = mod._parse_args(["--symbols", "SPY", "--mode", "research", "--table-prefix", "research_"])
        assert a.mode == "research" and a.allow_live_tables is False

    def test_daily_modes_no_longer_refuse_the_shared_table(self, tmp_path):
        """W91: the W65 guard is gone (an empty prefix IS the one shared set of tables); the run now gets as far
        as needing IB, so a --dry-run returns 0 instead of the old exit code 2."""
        mod = _load_script("backfill_historical_data")
        for mode in ("daily", "research"):
            a = mod._parse_args(["--symbols", "SPY", "--mode", mode, "--db-path",
                                 str(tmp_path / "h.db"), "--dry-run"])
            assert asyncio.run(mod._main(a)) == 0

    async def test_ib_bars_land_only_in_the_prefixed_table_with_a_source_tag(self, tmp_path, monkeypatch):
        mod = _load_script("backfill_historical_data")
        monkeypatch.setattr(mod, "_drop_incomplete_today", lambda df: df)
        st = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        live = HistoricalDataStore(db_path=tmp_path / "h.db")
        n, tag = await mod._backfill_daily_symbol(_FakeIB(_bars(260)), "SPY", 1.0, st,
                                                  pause_secs=0, dry_run=False)
        assert n == 260 and tag == SOURCE_IB_TRADES_RTH
        assert st.source_summary("SPY")[SOURCE_IB_TRADES_RTH][0] == 260
        assert live.source_summary("SPY") == {}                    # live table untouched

    async def test_short_ib_answer_uses_yahoo_for_the_whole_symbol(self, tmp_path, monkeypatch):
        mod = _load_script("backfill_historical_data")
        monkeypatch.setattr(mod, "_drop_incomplete_today", lambda df: df)
        monkeypatch.setattr(mod, "_fetch_daily_yahoo", lambda sym, years: _ohlcv("2025-01-02", 260))
        st = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        n, tag = await mod._backfill_daily_symbol(_FakeIB(_bars(10)), "SPY", 1.0, st,
                                                  pause_secs=0, dry_run=False)
        assert tag == SOURCE_YAHOO_UNADJ and n == 260
        assert set(st.source_summary("SPY")) == {SOURCE_YAHOO_UNADJ}   # never spliced with IB

    def test_todays_partial_bar_is_dropped_until_the_close(self, monkeypatch):
        import ait.utils.time as t
        mod = _load_script("backfill_historical_data")
        df = _ohlcv("2026-10-05", 4)
        today = df.index[-1].date()
        et = t.ET
        monkeypatch.setattr(t, "now_et", lambda: datetime(today.year, today.month, today.day, 11, 0, tzinfo=et))
        assert len(mod._drop_incomplete_today(df)) == 3
        monkeypatch.setattr(t, "now_et", lambda: datetime(today.year, today.month, today.day, 17, 0, tzinfo=et))
        assert len(mod._drop_incomplete_today(df)) == 4

    def test_validation_flags_bad_data(self, tmp_path):
        mod = _load_script("backfill_historical_data")
        st = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        good = _ohlcv("2025-01-02", 40)
        bad = good.copy()
        bad.iloc[5, bad.columns.get_loc("Volume")] = 0                       # zero volume
        bad.iloc[6, bad.columns.get_loc("High")] = bad.iloc[6]["Low"] - 5     # high < low
        bad = pd.concat([bad.iloc[:20], bad.iloc[30:]])                       # a 14-day gap
        st.save("SPY", bad, source=SOURCE_IB_TRADES_RTH)
        v = " | ".join(mod._validate_daily(st, "SPY")["violations"])
        assert "zero-volume" in v and "high < low" in v and "gaps" in v
        st2 = HistoricalDataStore(db_path=tmp_path / "h2.db", table_prefix="research_")
        st2.save("QQQ", good, source=SOURCE_IB_TRADES_RTH)
        assert mod._validate_daily(st2, "QQQ")["violations"] == []

    def test_script_never_places_orders(self):
        tree = ast.parse((ROOT / "scripts" / "backfill_historical_data.py").read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "placeOrder" not in names and "placeOrderAsync" not in names


# =========================================================================
# K. Economic calendar (operator CSV)
# =========================================================================
_CSV = """﻿date_utc,time_utc,type,country,currency,importance,event,reference_period,actual,forecast,previous,unit,preliminary,category,event_id,occurrence_id
2024-01-01,,holiday,United States,USD,,New Year's Day - NYSE,,,,,,,holiday,,
2024-01-05,13:30,event,United States,USD,high,U.S. Nonfarm Payrolls,Dec,216,170,173,K,False,employment,1,1
2024-01-11,13:30,event,United States,USD,high,U.S. Consumer Price Index (CPI) MoM,Dec,0.3,0.2,0.1,%,False,inflation,2,2
2024-01-25,13:30,event,United States,USD,high,U.S. Gross Domestic Product (GDP) QoQ,Q4,3.3,2.0,4.9,%,True,gdp,3,3
2024-01-31,19:00,event,United States,USD,high,Fed Interest Rate Decision,,5.5,5.5,5.5,%,False,rates,4,4
2024-01-31,19:00,event,United States,USD,high,Fed Interest Rate Decision,,5.5,5.5,5.5,%,False,rates,4,4
2024-02-01,00:30,event,Japan,JPY,low,Japan Foo Index,Jan,1,1,1,,False,other,5,5
2024-02-28,13:30,event,United States,USD,medium,U.S. Gross Domestic Product (GDP) QoQ,Q4,3.2,3.3,3.3,%,True,gdp,6,6
2024-03-28,12:30,event,United States,USD,medium,U.S. Gross Domestic Product (GDP) QoQ,Q4,3.4,3.2,3.2,%,False,gdp,7,7
2024-04-25,12:30,event,United States,USD,high,U.S. Gross Domestic Product (GDP) QoQ,Q1,1.6,2.5,3.4,%,True,gdp,8,8
2024-03-29,12:30,event,United States,USD,high,U.S. Core PCE Price Index MoM,Feb,0.3,0.3,0.4,%,False,inflation,9,9
2025-12-16,13:30,event,United States,USD,high,U.S. Nonfarm Payrolls,Oct,-105,,108,K,False,employment,10,10
2025-12-16,13:30,event,United States,USD,high,U.S. Nonfarm Payrolls,Nov,64,51,-105,K,False,employment,10,11
"""


class TestEconomicCalendar:
    @pytest.fixture
    def csv_path(self, tmp_path):
        p = tmp_path / "cal.csv"
        p.write_bytes(_CSV.encode("utf-8"))
        return p

    def test_normalize_dedupes_converts_to_et_and_derives_types(self, csv_path):
        mod = _load_script("load_econ_calendar_csv")
        df, stats = mod.normalize(csv_path)
        assert stats["rows_read"] == 13 and stats["duplicates_dropped"] == 1
        fomc = df[df.engine_event_type == "FOMC"].iloc[0]
        assert (fomc.event_date, fomc.event_time_et) == ("2024-01-31", "14:00")    # 19:00 UTC
        nfp = df[(df.engine_event_type == "NFP") & (df.event_date == "2024-01-05")].iloc[0]
        assert nfp.event_time_et == "08:30"
        jp = df[df.country == "Japan"].iloc[0]
        assert jp.event_date == "2024-01-31"                                       # UTC 00:30 -> prior ET day
        assert df[df.type == "holiday"].iloc[0].event_date == "2024-01-01"         # untimed keeps its date

    def test_gdp_keeps_only_the_advance_estimate_of_each_quarter(self, csv_path):
        mod = _load_script("load_econ_calendar_csv")
        df, _ = mod.normalize(csv_path)
        gdp = sorted(df[df.engine_event_type == "GDP"].event_date)
        assert gdp == ["2024-01-25", "2024-04-25"]      # Q4 second/third estimates excluded

    def test_combined_release_is_one_event_date_in_the_calendar(self, csv_path, tmp_path):
        from ait.data.economic_calendar import EventType, build_research_calendar
        mod = _load_script("load_econ_calendar_csv")
        df, _ = mod.normalize(csv_path)
        st = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        assert st.save_econ_events(df, SOURCE_INVESTING_CSV, "cal.csv", "abc") == len(df)
        cal = build_research_calendar(tmp_path / "h.db", "research_", ["NFP"])
        assert [e.date for e in cal._events] == [date(2024, 1, 5), date(2025, 12, 16)]
        assert all(e.event_type is EventType.NFP for e in cal._events)

    def test_rows_carry_provenance_and_span(self, csv_path, tmp_path):
        mod = _load_script("load_econ_calendar_csv")
        df, _ = mod.normalize(csv_path)
        st = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        st.save_econ_events(df, SOURCE_INVESTING_CSV, "cal.csv", "abc123")
        with sqlite3.connect(tmp_path / "h.db") as c:
            assert c.execute("SELECT DISTINCT source, source_file, source_sha256 FROM "
                             "research_econ_events").fetchall() == [
                (SOURCE_INVESTING_CSV, "cal.csv", "abc123")]
        first, last, rows = st.econ_events_span()
        assert first == "2024-01-01" and last == "2025-12-16" and rows == len(df)

    def test_event_set_is_filterable(self, csv_path, tmp_path):
        from ait.data.economic_calendar import build_research_calendar
        mod = _load_script("load_econ_calendar_csv")
        df, _ = mod.normalize(csv_path)
        HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_").save_econ_events(
            df, SOURCE_INVESTING_CSV, "cal.csv", "abc")
        full = build_research_calendar(tmp_path / "h.db", "research_", ["FOMC", "CPI", "NFP", "PCE", "GDP"])
        only = build_research_calendar(tmp_path / "h.db", "research_", ["CPI"])
        assert len(only._events) < len(full._events) and len(only._events) == 1

    def test_calendar_with_injected_events_gates_on_them(self):
        from ait.data.economic_calendar import EconomicCalendar, EconomicEvent, EventType
        cal = EconomicCalendar(events=[EconomicEvent(date(2024, 1, 11), EventType.CPI, "x")])
        assert cal.is_event_day(date(2024, 1, 11)) and not cal.is_event_day(date(2024, 1, 12))
        assert cal.days_until_next_event(date(2024, 1, 8)) == 3
        assert cal.days_until_next_event(date(2024, 1, 12)) is None

    def test_default_calendar_is_still_the_hardcoded_live_list(self):
        from ait.data import economic_calendar as ec
        assert ec.EconomicCalendar()._events is ec._ALL_EVENTS and len(ec._ALL_EVENTS) == 71

    def test_engine_uses_an_injected_calendar_and_defaults_to_the_live_one(self):
        from ait.backtesting.engine import Backtester
        from ait.data.economic_calendar import EconomicCalendar
        cal = EconomicCalendar(events=[])
        df = _ohlcv("2025-01-02", 120)
        assert Backtester(data=df, strategies=["iron_condor"], economic_calendar=cal)._economic_cal is cal
        assert isinstance(Backtester(data=df, strategies=["iron_condor"])._economic_cal, EconomicCalendar)

    def test_ingest_writes_the_shared_table_without_a_flag(self, csv_path, tmp_path):
        """W91: the empty-prefix guard is gone; the un-prefixed econ_events is the one shared table."""
        mod = _load_script("load_econ_calendar_csv")
        assert mod.main(["--csv", str(csv_path), "--db-path", str(tmp_path / "h.db")]) == 0
        with sqlite3.connect(tmp_path / "h.db") as c:
            assert c.execute("SELECT COUNT(*) FROM econ_events").fetchone()[0] > 0

    def test_walkforward_trims_data_to_the_calendar_span_and_builds_it(self, csv_path, tmp_path):
        mod = _load_script("load_econ_calendar_csv")
        df, _ = mod.normalize(csv_path)
        HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_").save_econ_events(
            df, SOURCE_INVESTING_CSV, "cal.csv", "abc")
        bt = _wf(tmp_path, macro_event_types=("FOMC", "CPI", "NFP", "PCE", "GDP"))
        data = {"SPY": _ohlcv("2023-06-01", 400)}
        with capture_logs() as logs:
            out = bt._apply_research_calendar(data)
        assert out["SPY"].index[0] >= pd.Timestamp("2023-12-31")
        assert bt._research_calendar is not None
        assert any(e["event"] == "macro_calendar_not_covering_span" for e in logs)

    def test_missing_calendar_falls_back_loudly_and_none_means_legacy(self, tmp_path):
        data = {"SPY": _ohlcv("2024-01-02", 300)}
        bt = _wf(tmp_path, macro_event_types=("CPI",))
        with capture_logs() as logs:
            assert bt._apply_research_calendar(data) is data
        assert bt._research_calendar is None
        assert any(e["event"] == "research_calendar_empty" for e in logs)
        assert _wf(tmp_path)._apply_research_calendar(data) is data     # macro_event_types None


# =========================================================================
# G. Live underlying-bar source: IB first, Yahoo fallback, no Polygon
# =========================================================================
class _FakeIBKR:
    connected = True

    def __init__(self, bars=None, raise_exc=None):
        self.requests, self.contracts = [], []
        self._bars, self._exc = bars, raise_exc
        self.ib = SimpleNamespace(reqHistoricalDataAsync=self._req)

    async def qualify_contract(self, contract):
        self.contracts.append(contract)
        return contract

    async def _req(self, contract, **k):
        self.requests.append(k)
        if self._exc:
            raise self._exc
        return self._bars


def _svc(ibkr):
    from ait.data.market_data import MarketDataService
    return MarketDataService(ibkr)


class TestLiveSourceIsIbFirst:
    @pytest.mark.parametrize("days,expected", [(30, "45 D"), (120, "180 D"), (200, "300 D"),
                                               (504, "3 Y"), (252, "2 Y")])
    def test_duration_mapping_expresses_long_spans_in_years(self, days, expected):
        from ait.data.market_data import MarketDataService
        assert MarketDataService._days_to_ibkr_daily_duration(days) == expected

    async def test_ib_is_used_first_tagged_and_shaped_like_before(self, monkeypatch):
        svc = _svc(_FakeIBKR(bars=_bars(300)))

        async def _no_yahoo(*a, **k):
            raise AssertionError("Yahoo must not be called when IB answers")

        monkeypatch.setattr(svc, "_get_yahoo_historical", _no_yahoo)
        df = await svc.get_historical("SPY", days=252)
        assert df.attrs["source"] == SOURCE_IB_TRADES_RTH
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert df.index.tz is None and df.index.is_monotonic_increasing
        assert len(df) == 252                                           # trimmed to `days` rows

    async def test_no_ib_client_falls_to_yahoo_with_a_warning(self, monkeypatch):
        svc = _svc(None)

        async def _yahoo(symbol, days, interval):
            return _ohlcv("2025-01-02", 100)

        monkeypatch.setattr(svc, "_get_yahoo_historical", _yahoo)
        with capture_logs() as logs:
            df = await svc.get_historical("SPY", days=100)
        assert df.attrs["source"] == SOURCE_YAHOO_UNADJ
        assert any(e["event"] == "daily_history_ib_miss" and e["log_level"] == "warning" for e in logs)

    @pytest.mark.parametrize("ibkr", [
        _FakeIBKR(raise_exc=RuntimeError("pacing")),          # error
        _FakeIBKR(bars=_bars(5)),                             # short answer
        _FakeIBKR(bars=[]),                                   # empty
    ])
    async def test_error_short_or_empty_ib_answer_falls_to_yahoo(self, ibkr, monkeypatch):
        svc = _svc(ibkr)

        async def _yahoo(symbol, days, interval):
            return _ohlcv("2025-01-02", 100)

        monkeypatch.setattr(svc, "_get_yahoo_historical", _yahoo)
        df = await svc.get_historical("SPY", days=100)
        assert df.attrs["source"] == SOURCE_YAHOO_UNADJ

    async def test_vix_uses_the_cboe_index_and_other_indices_go_to_yahoo(self, monkeypatch):
        ib = _FakeIBKR(bars=_bars(300))
        svc = _svc(ib)
        df = await svc.get_historical("^VIX", days=200)
        assert df.attrs["source"] == SOURCE_IB_TRADES_RTH
        c = ib.contracts[0]
        assert (type(c).__name__, c.symbol, c.exchange) == ("Index", "VIX", "CBOE")

        async def _yahoo(symbol, days, interval):
            return _ohlcv("2025-01-02", 100)

        ib2 = _FakeIBKR(bars=_bars(300))
        svc2 = _svc(ib2)
        monkeypatch.setattr(svc2, "_get_yahoo_historical", _yahoo)
        await svc2.get_historical("^GSPC", days=100)
        assert ib2.requests == []                                       # never asked IB

    async def test_non_daily_intervals_do_not_use_ib(self, monkeypatch):
        ib = _FakeIBKR(bars=_bars(300))
        svc = _svc(ib)

        async def _yahoo(symbol, days, interval):
            return _ohlcv("2025-01-02", 20)

        monkeypatch.setattr(svc, "_get_yahoo_historical", _yahoo)
        await svc.get_historical("SPY", days=20, interval="1h")
        assert ib.requests == []

    def test_class_guard_no_polygon_stock_path_remains(self):
        src = (SRC / "data" / "market_data.py").read_text()
        tree = ast.parse(src)
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "list_aggs" not in attrs and "_get_polygon_historical" not in {
            n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        init = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                    and n.name == "MarketDataService")
        ctor = next(n for n in init.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        assert "polygon_api_key" not in [a.arg for a in ctor.args.args + ctor.args.kwonlyargs]

    @pytest.mark.parametrize("path", ["bot/orchestrator.py", "orchestration/master.py"])
    def test_callers_no_longer_pass_a_polygon_key(self, path):
        tree = ast.parse((SRC / path).read_text())
        kws = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords}
        assert "polygon_api_key" not in kws


class TestTrainerPersistsTheProvider:
    async def test_trained_rows_are_saved_with_the_frames_source(self):
        from ait.ml.trainer import ModelTrainer
        saved = []

        class _Store:
            def save(self, symbol, df, source=None, overwrite=True):
                saved.append((symbol, source))

        class _MD:
            async def get_historical(self, symbol, days=252, interval="1d"):
                df = _ohlcv("2025-01-02", 60)
                df.attrs["source"] = SOURCE_IB_TRADES_RTH
                return df

        t = object.__new__(ModelTrainer)
        t._config = SimpleNamespace(lookback_days=504)
        t._predictor = SimpleNamespace(model_version="v0", cv_scores={},
                                       train=lambda *a, **k: {"xgb": 0.5})
        t._market_data, t._store = _MD(), _Store()
        t._range_predictor = None
        t._vol_mag_predictor = None
        t._drift_detector = SimpleNamespace(acknowledge_retrain=lambda: None)
        t.check_range_spec = lambda: None

        async def _ctx():
            return {}

        t._fetch_market_context = _ctx
        res = await t.train_all_symbols(["SPY"])
        assert res == {"SPY": {"xgb": 0.5}}
        assert saved == [("SPY", SOURCE_IB_TRADES_RTH)]


# =========================================================================
# Optimizer: inert wing_k, research fetch
# =========================================================================
class TestOptimizerResearchSupport:
    def _opt(self, capital, strategies=("iron_condor",), **kw):
        from ait.optimization.optimizer import StrategyOptimizer
        # W67: these tests are about wing_k. With the default penalty_on_range_prob=0 the
        # (condor-only) fractal dimensions are ALSO dropped as inert; a non-zero value
        # keeps them in the search so the assertions below isolate the wing_k rule.
        kw.setdefault("penalty_on_range_prob", 0.05)
        o = StrategyOptimizer(symbols=["SPY"], strategies=list(strategies), n_trials=1,
                              initial_capital=capital, **kw)
        df = _ohlcv("2025-01-02", 300, base=700.0)
        df["implied_vol"] = 0.15
        o._data = {"SPY": df}
        return o

    def test_wing_k_is_dropped_when_the_budget_cap_always_binds(self):
        o = self._opt(5_000, position_size_pct=0.05)          # budget $250 -> ~$3 wing (W100: pinned, shipped is 0.20)
        o._maybe_drop_inert_params()
        assert o._dropped_params == {"iron_condor__wing_k"}

    def test_wing_k_is_searched_when_capital_is_large_enough(self):
        o = self._opt(1_000_000)                              # cap never binds
        o._maybe_drop_inert_params()
        assert o._dropped_params == set()

    def test_not_dropped_without_the_budget_cap_or_without_iron_condor(self):
        o = self._opt(5_000, budget_aware_wings=False)
        o._maybe_drop_inert_params()
        assert o._dropped_params == set()
        o2 = self._opt(5_000, strategies=("bull_call_spread",))
        o2._maybe_drop_inert_params()
        assert o2._dropped_params == set()

    def test_dropped_param_is_not_suggested_but_the_rest_are(self):
        import optuna
        o = self._opt(5_000, position_size_pct=0.05)
        o._maybe_drop_inert_params()
        trial = optuna.create_study().ask()
        params = o._suggest_params(trial)
        assert "iron_condor__wing_k" not in params and "iron_condor__delta_short" in params
        big = self._opt(1_000_000)
        big._maybe_drop_inert_params()
        assert "iron_condor__wing_k" in big._suggest_params(optuna.create_study().ask())

    def test_standalone_fetch_uses_the_research_prefix(self, monkeypatch, tmp_path):
        import ait.data.market_data as md
        seen = []
        monkeypatch.setattr(md, "load_daily_ohlcv", lambda s, days=730, db_path=None,
                            table_prefix="", start_date=None, end_date=None, research=None:
                            seen.append((table_prefix, start_date)) or _ohlcv("2024-01-02", 300))
        o = self._opt(5_000, table_prefix="research_", db_path=tmp_path / "h.db",
                      history_start_date=date(2024, 1, 1))
        o._data = {}
        o._fetch_data()
        assert seen == [("research_", date(2024, 1, 1))]


# =========================================================================
# Fixes found while running the real research backtest
# =========================================================================
class TestVlmcAlignsByDateRange:
    def _seed(self, tmp_path, days):
        st = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        frames = []
        for d in days:
            idx = pd.date_range(f"{d.date()} 14:30", periods=40, freq="5min", tz="UTC")
            close = 100 + np.arange(40) * 0.05
            frames.append(pd.DataFrame({"Open": close, "High": close + 0.1, "Low": close - 0.1,
                                        "Close": close, "Volume": 1_000}, index=idx))
        st.save_intraday("SPY", pd.concat(frames), source="TRADES")
        return st

    def test_historical_frames_get_real_vlmc_values_not_all_nan(self, tmp_path):
        """Pre-fix: the merge loaded the last len(features)+10 days counted back from
        TODAY, so a 2024 frame matched nothing (matched_days=0) and every VLMC column
        was NaN."""
        from ait.ml.features import FeatureEngine
        days = list(pd.bdate_range("2024-03-04", periods=5))
        st = self._seed(tmp_path, days)
        feats = pd.DataFrame({"Close": np.arange(5, dtype=float)}, index=pd.DatetimeIndex(days))
        out = FeatureEngine()._merge_intraday_features(feats, st, "SPY")
        vlmc_cols = [c for c in out.columns if c != "Close"]
        assert len(vlmc_cols) >= 13
        assert out[vlmc_cols].notna().any(axis=1).all()        # every historical day matched

    def test_store_without_range_loader_keeps_the_legacy_path(self):
        from ait.ml.features import FeatureEngine

        class _Legacy:
            def load_intraday(self, symbol, days=7, interval="5m"):
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        feats = pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex(["2024-03-04"]))
        assert FeatureEngine()._merge_intraday_features(feats, _Legacy(), "SPY").equals(feats)


class TestPoolGateAgreesWithTheEngine:
    KW = dict(position_size_pct=0.05, max_portfolio_risk_pct=0.20, max_open_positions=10)

    def _trade(self, **over):
        t = {"symbol": "SPY", "entry_date": "2025-03-03", "exit_date": "2025-03-10",
             "contracts": 1, "pnl": 40.0, "max_loss": 260.0, "max_loss_per_share": 2.6,
             "credit": 80.0, "entry_commission": 7.0}
        t.update(over)
        return t

    def test_trade_the_engine_took_on_mid_credit_is_not_re_rejected(self):
        from ait.backtesting.pool import allocate_pool
        # friction-inclusive max loss $260 > the $250 budget, but the budget gate the
        # engine used (mid credit) was $240: live takes it, so must the pool.
        r = allocate_pool([self._trade(budget_gate_risk_per_contract=240.0)], 5000, **self.KW)
        assert len(r.accepted) == 1 and not r.rejections

    def test_negative_control_without_the_key_it_is_unaffordable(self):
        from ait.backtesting.pool import allocate_pool
        r = allocate_pool([self._trade()], 5000, **self.KW)
        assert r.accepted == [] and r.rejections["unaffordable"] == 1

    def test_engine_records_the_gate_risk_on_budget_aware_condors(self):
        from datetime import date as _d
        from ait.backtesting.engine import Backtester
        df = _ohlcv("2025-01-02", 400, base=277.0)
        bt = Backtester(data=df, strategies=["iron_condor"], initial_capital=5000,
                        position_size_pct=0.05, budget_aware_wings=True,
                        max_contracts_per_trade=1, wing_floor_dollars=2.0)
        pos = bt._build_credit_position("iron_condor", 277.0, 0.16, 21 / 365, 0.05, 21,
                                        _d(2026, 3, 2), 5000.0)
        assert pos is not None
        assert pos["budget_gate_risk_per_contract"] <= pos["max_loss_per_share"] * 100 + 1e-6


class TestMaxWindows:
    def test_keeps_only_the_most_recent_windows(self):
        bt = _wf(max_windows=3)
        windows = list(range(10))
        assert bt._limit_windows(windows) == [7, 8, 9]
        assert _wf()._limit_windows(windows) == windows              # default: all
        assert _wf(max_windows=50)._limit_windows(windows) == windows


class TestWeeklyHealthBacktestStaysWithinItsTimeout:
    def test_health_run_is_limited_to_recent_windows(self):
        """run_backtest now defaults to the 10-window research history (~12 min); the
        supervisor's weekly health job has a 600 s timeout, so it pins --max-windows."""
        tree = ast.parse((SRC / "orchestration" / "master.py").read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "run_backtest_job" or
                  isinstance(n, ast.FunctionDef) and "backtest" in n.name and
                  any(isinstance(c, ast.Constant) and c.value == "--strategies"
                      for c in ast.walk(n)))
        consts = [c.value for c in ast.walk(fn) if isinstance(c, ast.Constant)]
        assert "--max-windows" in consts
        assert consts[consts.index("--max-windows") + 1] == "4"

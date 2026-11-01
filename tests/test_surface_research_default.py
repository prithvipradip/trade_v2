"""pricing_source: surface is the research default (operator decision D2); the schema default stays bs_legacy.

Real code paths on a temporary history database. Layers:

  * the loaded research view (overlay over config.yaml) and the engine reading it
  * the preflight for the 2024-10-01 window: bars that cover it pass; bars that start later, and a symbol with no bars
    (SPY, IWM), refuse naming the first missing session and BOTH remedies
  * the fixed per-leg-miss limit: stamped in the data contract, flagged by run_backtest, never a refusal
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from ait.backtesting.result import surface_pricing_stamp
from ait.config import invariants_research as ri_limits
from ait.data import research_inputs as ri
from ait.data.historical import HistoricalDataStore

ROOT = Path(__file__).resolve().parent.parent
WINDOW = ri.Window(date(2024, 10, 1), date(2024, 12, 31))


def _bars(sessions: list[date]) -> pd.DataFrame:
    """One option bar per session (the shape ``save_option_daily_bars`` takes)."""
    rows = []
    for d in sessions:
        rows.append({
            "option_ticker": "O:QQQ250117C00480000", "expiry": "2025-01-17", "right": "C", "strike": 480.0,
            "date": str(d), "dte": (date(2025, 1, 17) - d).days,
            "open": 5.0, "high": 5.5, "low": 4.5, "close": 5.2, "volume": 1000, "vwap": 5.1, "transactions": 100,
            "underlying_close": 480.0, "iv": 0.2, "iv_outcome": "solved",
            "delta": 0.5, "gamma": 0.02, "theta": -0.05, "vega": 0.10, "risk_free_rate": 0.05})
    return pd.DataFrame(rows)


def _store(tmp_path, first_session: date | None, symbol: str = "QQQ") -> Path:
    db = tmp_path / "historical.db"
    store = HistoricalDataStore(db_path=db)
    if first_session is not None:
        store.save_option_daily_bars(symbol, _bars(ri.sessions_between(first_session, WINDOW.end)))
    return db


def _gaps(db, symbols=("QQQ",)):
    report = ri.check_research_inputs(ri.ResearchRequest(
        window=WINDOW, symbols=symbols, db_path=db, surface=True,
        max_null_iv_run=3, max_dividend_gap_days=100, max_event_gap_days=120))
    return [g for g in report.gaps if g.input == "option_daily_bars"]


class TestLoadedResearchView:
    def test_the_overlay_sets_surface_and_the_window_start(self):
        from ait.config.research import load_research_settings
        b = load_research_settings(str(ROOT / "config.yaml")).backtest
        assert b.pricing_source == "surface" and b.history_start_date == date(2024, 10, 1)

    def test_the_schema_default_and_the_live_config_keep_bs_legacy(self):
        import yaml

        from ait.config.settings import BacktestConfig, load_settings
        assert BacktestConfig().pricing_source == "bs_legacy"
        assert load_settings(str(ROOT / "config.yaml")).backtest.pricing_source == "bs_legacy"
        assert "pricing_source" not in (yaml.safe_load((ROOT / "config.yaml").read_text()).get("backtest") or {})

    def test_the_engine_reads_surface_from_the_loaded_overlay(self, tmp_path):
        from ait.backtesting.engine import Backtester
        from ait.config.research import use_research_config
        use_research_config()
        idx = pd.bdate_range("2024-10-01", periods=30)
        data = pd.DataFrame({"Open": 480.0, "High": 481.0, "Low": 479.0, "Close": 480.0, "Volume": 5e6,
                             "implied_vol": 0.2}, index=idx)
        bt = Backtester(data=data, strategies=["iron_condor"], symbol="QQQ", db_path=_store(tmp_path, None),
                        strict_inputs=False, macro_event_gate=False, friction_model="legacy",
                        ex_div_entry_gate="off")
        assert bt._pricing_source == "surface"         # no bars here: it degrades loudly, which strict mode refuses
        assert bt._surface is None

    def test_a_registered_study_keeps_its_explicit_bs_legacy(self):
        from ait.backtesting.study_harness import study_engine_kwargs
        # SPY has no option bars: a registered study names its pricing instead of inheriting the overlay's surface
        kw = study_engine_kwargs(data=pd.DataFrame(), symbol="SPY", db_path=Path("x"), intraday_store=None, entry_dte=3)
        assert kw["pricing_source"] == "bs_legacy"


class TestPreflightForTheNewWindow:
    def test_bars_covering_the_window_pass(self, tmp_path):
        assert _gaps(_store(tmp_path, date(2024, 9, 30))) == []        # the first bar day is the session BEFORE the window

    def test_bars_that_start_later_name_the_first_missing_session(self, tmp_path):
        (g,) = _gaps(_store(tmp_path, date(2024, 10, 15)))
        assert g.symbol == "QQQ" and "first missing 2024-10-01" in g.found

    @pytest.mark.parametrize("symbol", ["SPY", "IWM"])
    def test_a_symbol_with_no_bars_refuses_with_both_remedies(self, tmp_path, symbol):
        db = _store(tmp_path, date(2024, 9, 30))                       # QQQ bars exist, the named symbol has none
        (g,) = _gaps(db, symbols=(symbol,))
        assert g.symbol == symbol and g.found.startswith("0 sessions") and "first missing 2024-10-01" in g.found
        assert f"python scripts/backfill_option_history.py --symbol {symbol}" in g.remedy
        assert "backtest.pricing_source to bs_legacy" in g.remedy
        with pytest.raises(ri.ResearchDataUnavailable) as e:
            ri.preflight_or_raise(ri.ResearchRequest(
                window=WINDOW, symbols=(symbol,), db_path=db, surface=True, max_null_iv_run=3,
                max_dividend_gap_days=100, max_event_gap_days=120), "unit test")
        assert f"option_daily_bars[{symbol}]" in str(e.value)

    def test_the_same_symbol_passes_the_option_bar_check_on_bs_legacy(self, tmp_path):
        report = ri.check_research_inputs(ri.ResearchRequest(
            window=WINDOW, symbols=("SPY",), db_path=_store(tmp_path, None), surface=False,
            max_null_iv_run=3, max_dividend_gap_days=100, max_event_gap_days=120))
        assert "option_daily_bars" not in report.checked


class TestSurfaceMissLimit:
    def test_the_limit_is_a_registered_fixed_research_limit(self):
        by_name = {i.name: i for i in ri_limits.RESEARCH_INVARIANTS}
        inv = by_name["RESEARCH_MAX_SURFACE_LEG_MISS_FRACTION"]
        assert inv.value == ri_limits.RESEARCH_MAX_SURFACE_LEG_MISS_FRACTION == 0.10
        assert ri_limits.research_invariant_values()["RESEARCH_MAX_SURFACE_LEG_MISS_FRACTION"] == 0.10

    def test_no_surface_legs_means_no_stamp(self):
        assert surface_pricing_stamp({}) is None
        assert surface_pricing_stamp({"iv_blend_remarks": 4}) is None

    @pytest.mark.parametrize("hit,miss,exceeds", [(90, 10, False), (89, 11, True), (100, 0, False), (0, 5, True)])
    def test_the_stamp_flags_a_miss_share_above_the_limit(self, hit, miss, exceeds):
        s = surface_pricing_stamp({"surface_legs_hit": hit, "surface_legs_miss": miss})
        assert s["legs_priced"] == hit + miss and s["from_surface"] == hit and s["fell_back"] == miss
        assert s["exceeds_limit"] is exceeds and s["limit"] == 0.10
        assert s["miss_fraction"] == pytest.approx(miss / (hit + miss), abs=1e-4)

    def test_the_walk_forward_data_contract_carries_the_stamp_summed_over_windows(self, tmp_path):
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        wf = WalkForwardBacktester(["QQQ"], ["iron_condor"], db_path=tmp_path / "h.db", strict_inputs=False,
                                   config=WalkForwardConfig(macro_event_gate=False, friction_model="legacy"))
        wf._declared_by_window = {0: {"QQQ": {"surface_legs_hit": 40, "surface_legs_miss": 10, "iv_blend_remarks": 3}},
                                  1: {"QQQ": {"surface_legs_hit": 40, "surface_legs_miss": 10}}}
        dc = wf.data_contract()
        assert dc["declared_models"]["surface_legs_miss"] == 20
        assert dc["surface_pricing"]["legs_priced"] == 100 and dc["surface_pricing"]["exceeds_limit"] is True
        wf._declared_by_window = {0: {"QQQ": {"iv_blend_remarks": 3}}}
        assert wf.data_contract()["surface_pricing"] is None             # a bs_legacy run stamps nothing

    def test_run_backtest_flags_only_a_run_above_the_limit_and_never_refuses(self):
        import run_backtest as rb
        assert rb.surface_pricing_notice(None) is None
        assert rb.surface_pricing_notice({"surface_pricing": None}) is None
        assert rb.surface_pricing_notice({"surface_pricing": surface_pricing_stamp(
            {"surface_legs_hit": 95, "surface_legs_miss": 5})}) is None
        note = rb.surface_pricing_notice({"surface_pricing": surface_pricing_stamp(
            {"surface_legs_hit": 70, "surface_legs_miss": 30})})
        assert "30 of 100 legs" in note and "RESEARCH_MAX_SURFACE_LEG_MISS_FRACTION" in note

    def test_the_manifest_warns_on_bs_legacy_only(self, monkeypatch):
        import run_backtest as rb
        from ait.config.settings import load_settings
        s = load_settings()

        def manifest(source):
            patched = s.model_copy(update={"backtest": s.backtest.model_copy(update={"pricing_source": source})})
            monkeypatch.setattr(rb, "_research_settings", lambda: patched)
            old = sys.argv
            sys.argv = ["run_backtest.py"]
            try:
                return rb.build_parity_manifest(rb.parse_args())
            finally:
                sys.argv = old
        assert [w for w in manifest("bs_legacy")["parity_warnings"] if w.startswith("pricing_source")]
        assert not [w for w in manifest("surface")["parity_warnings"] if w.startswith("pricing_source")]

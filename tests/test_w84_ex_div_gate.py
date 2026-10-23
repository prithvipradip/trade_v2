"""W84: the research twin of live's ex-dividend ENTRY block (W51), with modes.

  * the projection live uses is now ONE pure function (dividends.project_next_ex_div); live's
    _fetch_dividend still returns exactly what the pre-W84 body returned
  * ex_div_blocks: off / live (one projected date, from ex-dates known before the entry day) /
    exact (any real ex-date in [entry, expiry]) / last_k_days (within K days of expiry)
  * the engine refuses a call-selling entry on the BUILT position's expiry; off is inert; a gate
    that is on without data raises
  * walk-forward, the window optimizer, the shadow engine and the trial engines all receive it
  * the store, the backfill script, the registry row, the manifest warning, the research overlay
"""
from __future__ import annotations

import random
import sqlite3
import sys
import types
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import optuna  # noqa: F401  (the optimizer module imports it)
import pandas as pd
import pytest
import yaml

import ait.backtesting.engine as engine_mod
import ait.backtesting.gate_parity as gp
import ait.backtesting.walkforward as wf_mod
import ait.data.dividends as dividends_mod
from ait.backtesting.engine import Backtester
from ait.backtesting.ex_div import (MODES, ex_dates_in_window, ex_div_blocks,
                                    sells_a_call)
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.config.settings import BacktestConfig
from ait.data.dividends import DividendCalendar, project_next_ex_div
from ait.data.historical import SOURCE_YAHOO_DIVIDENDS, HistoricalDataStore
from ait.optimization.optimizer import StrategyOptimizer

from tests.test_r20_research_validity import (  # noqa: F401 - shared fixtures
    _engine_kwargs, _recorder_factory, _run_window, synth_df, synth_features)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# --------------------------------------------------------------------------
# the projection: one pure function, live unchanged
# --------------------------------------------------------------------------

def _pre_w84_projection(dates: list[date], today: date) -> date:
    """The pre-W84 body of DividendCalendar._fetch_dividend (projection part), verbatim in
    behaviour: the reference the refactor is proved against."""
    dates = dates[-4:]
    gaps = [(dates[i] - dates[i - 1]).days for i in range(1, len(dates))]
    avg_interval = (sum(gaps) / len(gaps)) if gaps else 91
    projected = dates[-1] + timedelta(days=round(avg_interval))
    steps = 0
    while projected < today and steps < 8:
        projected = projected + timedelta(days=round(avg_interval))
        steps += 1
    return projected


def _series(dates: list[date], amounts: list[float] | None = None) -> pd.Series:
    amounts = amounts or [0.5 + 0.01 * i for i in range(len(dates))]
    return pd.Series(amounts, index=pd.DatetimeIndex([pd.Timestamp(d) for d in dates]))


class TestProjection:
    def test_matches_the_pre_w84_body(self):
        """2,000 seeded histories (1-9 gaps of 1-200 days; as_of from 60 days before the last
        ex-date to ~7 years after, so the step cap is exercised too)."""
        rng = random.Random(84)
        for _ in range(2000):
            dates = [date(2005, 1, 1) + timedelta(days=rng.randint(0, 7300))]
            for _g in range(rng.randint(1, 9)):
                dates.append(dates[-1] + timedelta(days=rng.randint(1, 200)))
            as_of = dates[-1] + timedelta(days=rng.randint(-60, 2500))
            assert project_next_ex_div(dates, as_of) == _pre_w84_projection(dates, as_of), (
                dates, as_of)

    @pytest.mark.parametrize("today", [date(2024, 1, 2), date(2025, 6, 30), date(2031, 1, 1)])
    def test_live_fetch_returns_what_it_always_returned(self, monkeypatch, today):
        # irregular gaps (80, 100, 95, 84): the mean of the last 3 gaps (93) differs from the
        # mean of the last 2 (89.5 -> 90), so a projection over the wrong window is visible
        dates = [date(2023, 3, 20), date(2023, 6, 8), date(2023, 9, 16), date(2023, 12, 20),
                 date(2024, 3, 13)]
        assert _pre_w84_projection(dates, today) != _pre_w84_projection(dates[1:][-3:], today) \
            or today > date(2030, 1, 1)
        fake = types.ModuleType("yfinance")
        fake.Ticker = lambda sym: SimpleNamespace(dividends=_series(dates))
        monkeypatch.setitem(sys.modules, "yfinance", fake)
        monkeypatch.setattr(dividends_mod, "now_et",
                            lambda: pd.Timestamp(today).to_pydatetime())
        info = DividendCalendar()._fetch_dividend("QQQ")
        assert info.data_available is True
        assert info.next_ex_div_date == _pre_w84_projection(dates, today)
        assert info.expected_amount == pytest.approx(0.54)          # the last payment

    def test_too_little_history_is_no_projection(self, monkeypatch):
        assert project_next_ex_div([], date(2024, 1, 1)) is None
        assert project_next_ex_div([date(2024, 1, 1)], date(2024, 1, 1)) is None
        fake = types.ModuleType("yfinance")
        fake.Ticker = lambda sym: SimpleNamespace(dividends=_series([date(2024, 1, 2)]))
        monkeypatch.setitem(sys.modules, "yfinance", fake)
        info = DividendCalendar()._fetch_dividend("NEW")
        assert (info.next_ex_div_date, info.data_available) == (None, True)

    def test_bankers_rounding_is_kept(self):
        d = [date(2024, 1, 1), date(2024, 3, 31), date(2024, 6, 30)]      # gaps 90, 91 -> 90.5
        assert project_next_ex_div(d, date(2024, 7, 1)) == date(2024, 6, 30) + timedelta(days=90)

    def test_the_step_cap_can_still_answer_a_past_date(self):
        d = [date(2000, 1, 1), date(2000, 4, 1)]                       # 91-day gap
        got = project_next_ex_div(d, date(2024, 1, 1))
        assert got == date(2000, 4, 1) + timedelta(days=91 * 9) and got < date(2024, 1, 1)


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------

QUARTERLY = [date(2023, 3, 17) + timedelta(days=91 * i) for i in range(12)]


class TestRule:
    def test_off_never_blocks(self):
        assert ex_div_blocks("off", QUARTERLY, date(2023, 6, 10), date(2023, 6, 30)) == (False, None)

    @pytest.mark.parametrize("entry, expiry, blocked", [
        (date(2023, 6, 16), date(2023, 6, 30), True),     # ex-date ON the entry day
        (date(2023, 6, 2), date(2023, 6, 16), True),      # ex-date ON the expiry
        (date(2023, 6, 17), date(2023, 7, 1), False),     # the day after
        (date(2023, 5, 1), date(2023, 6, 15), False),     # expiry the day before
    ])
    def test_exact_window_is_inclusive_at_both_ends(self, entry, expiry, blocked):
        assert QUARTERLY[1] == date(2023, 6, 16)
        assert ex_div_blocks("exact", QUARTERLY, entry, expiry)[0] is blocked

    def test_live_uses_only_dates_known_before_the_entry_day(self):
        # history regular every 91 days, then the fund moves the next ex-date 20 days later
        hist = QUARTERLY[:5]
        moved = hist[-1] + timedelta(days=111)
        dates = hist + [moved]
        projected = hist[-1] + timedelta(days=91)
        entry, expiry = projected - timedelta(days=3), projected + timedelta(days=11)
        assert ex_div_blocks("live", dates, entry, expiry) == (True, projected)
        assert ex_div_blocks("exact", dates, entry, expiry) == (False, None)
        # and the reverse: the real date is in the window, the projection is not
        e2, x2 = moved - timedelta(days=2), moved + timedelta(days=12)
        assert ex_div_blocks("live", dates, e2, x2)[0] is False
        assert ex_div_blocks("exact", dates, e2, x2) == (True, moved)

    def test_an_ex_date_on_the_entry_day_is_not_yet_known_to_live(self):
        d = QUARTERLY[:6]
        on = d[-1]
        blocked, used = ex_div_blocks("live", d, on, on + timedelta(days=14))
        assert used == _pre_w84_projection(d[:-1], on)                 # projected from the 5 before

    @pytest.mark.parametrize("k, blocked", [(13, False), (14, True), (30, True)])
    def test_last_k_days_boundary(self, k, blocked):
        ex = QUARTERLY[1]                                              # 2023-06-16
        assert ex_div_blocks("last_k_days", QUARTERLY, ex - timedelta(days=1),
                             ex + timedelta(days=14), k)[0] is blocked

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="unknown ex_div_entry_gate mode"):
            ex_div_blocks("sometimes", QUARTERLY, date(2023, 6, 1), date(2023, 6, 20))

    def test_window_helper(self):
        assert ex_dates_in_window(QUARTERLY, date(2023, 6, 16), date(2023, 9, 15)) == [
            date(2023, 6, 16), date(2023, 9, 15)]
        assert ex_dates_in_window(QUARTERLY, date(2023, 6, 17), date(2023, 9, 14)) == []

    @pytest.mark.parametrize("pos, sells", [
        ({"strategy": "iron_condor", "short_call_strike": 110, "short_put_strike": 90}, True),
        ({"strategy": "short_strangle", "short_call_strike": 110}, True),
        ({"strategy": "call_credit_spread", "short_call_strike": 110}, True),
        ({"strategy": "put_credit_spread", "short_put_strike": 90}, False),
        ({"strategy": "bull_call_spread", "short_strike": 105, "option_type": "call"}, True),
        ({"strategy": "bear_put_spread", "short_strike": 95, "option_type": "put"}, False),
        ({"strategy": "long_call", "strike": 100, "option_type": "call"}, False),
    ])
    def test_sells_a_call_matches_lives_definition(self, pos, sells):
        assert sells_a_call(pos) is sells


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------

@pytest.fixture
def div_db(tmp_path):
    db = tmp_path / "historical.db"
    HistoricalDataStore(db_path=db).save_dividends(      # W91: the one shared (un-prefixed) dividends table
        "SPY", [(d, 0.5) for d in QUARTERLY], source=SOURCE_YAHOO_DIVIDENDS,
        fetched_at="2026-10-16T00:00:00-04:00")
    return db


def _engine(synth_df, synth_features, db, mode, **over):
    kw = _engine_kwargs(synth_df, synth_features, symbol="SPY", db_path=db,
                        earnings_buffer_days_before=0, earnings_buffer_days_after=0,
                        ex_div_entry_gate=mode, **over)
    return Backtester(**kw)


def _spans(mode, trades, k=5):
    return [t for t in trades
            if ex_div_blocks(mode, QUARTERLY, date.fromisoformat(str(t["entry_date"])[:10]),
                             date.fromisoformat(str(t["expiry_date"])[:10]), k)[0]]


class TestEngine:
    def test_off_is_inert_and_never_reads_dividends(self, synth_df, synth_features, div_db,
                                                    monkeypatch):
        def _boom(*_a, **_k):
            raise AssertionError("mode off must not read the dividend table")
        monkeypatch.setattr(HistoricalDataStore, "load_dividends", _boom)
        bt = _engine(synth_df, synth_features, div_db, "off")
        assert bt._ex_dates is None
        res = bt.run()
        assert bt._funnel.get("ex_div_entry", 0) == 0
        assert _spans("exact", res.trades), "the scenario must contain entries that span an ex-date"

    @pytest.mark.parametrize("mode", ["live", "exact"])
    def test_on_refuses_exactly_the_spanning_entries(self, synth_df, synth_features, div_db, mode):
        off = _engine(synth_df, synth_features, div_db, "off").run().trades
        bt = _engine(synth_df, synth_features, div_db, mode)
        res = bt.run()
        assert bt._ex_dates == QUARTERLY
        assert bt._funnel.get("ex_div_entry", 0) > 0
        assert _spans(mode, res.trades) == [], "a traded entry spans an ex-date the gate knows"
        assert len(_spans(mode, off)) > 0

    def test_only_a_position_that_sells_a_call_is_blocked(self, synth_df, synth_features, div_db,
                                                         monkeypatch):
        """Live blocks only signals that SELL A CALL. The engine asks sells_a_call about the
        BUILT position (its real legs); a position that does not sell one trades through
        ex-dates. (The synthetic tape never selects a put-only structure, so the answer is
        forced here; sells_a_call itself is covered by TestRule.)"""
        asked: list[dict] = []

        def _no_call(pos):
            asked.append(pos)
            return False
        monkeypatch.setattr(engine_mod, "sells_a_call", _no_call)
        bt = _engine(synth_df, synth_features, div_db, "exact")
        res = bt.run()
        assert asked and all("expiry_date" in p and "short_call_strike" in p for p in asked)
        assert bt._funnel.get("ex_div_entry", 0) == 0
        assert _spans("exact", res.trades), "entries spanning ex-dates must have traded"

    def test_last_k_blocks_only_ex_dates_near_expiry(self, synth_df, synth_features, tmp_path):
        """Ex-dates placed against the days this tape attempts entries (14-day expiries):
        2024-02-12 sits 3 days before the expiry of the 02-01 attempt (inside K=7), 2024-04-06
        sits 12 days before the expiry of the 04-04 attempt (outside K=7, inside exact)."""
        dates = [date(2024, 2, 12), date(2024, 4, 6)]
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db).save_dividends(      # W91: the one shared (un-prefixed) table
            "SPY", [(d, 0.5) for d in dates], SOURCE_YAHOO_DIVIDENDS, "t")
        ex = _engine(synth_df, synth_features, db, "exact")
        lk = _engine(synth_df, synth_features, db, "last_k_days", ex_div_entry_k_days=7)
        ex.run()
        res = lk.run()
        assert lk._ex_div_k == 7
        assert 0 < lk._funnel.get("ex_div_entry", 0) < ex._funnel.get("ex_div_entry", 0), \
            "K=7 must block some entries and strictly fewer than exact"
        traded = [(date.fromisoformat(str(t["entry_date"])[:10]),
                   date.fromisoformat(str(t["expiry_date"])[:10])) for t in res.trades]
        assert not [e for e, x in traded for d in dates if e <= d <= x and (x - d).days <= 7]
        assert [e for e, x in traded for d in dates if e <= d <= x and (x - d).days > 7], \
            "an entry spanning an ex-date far from expiry must trade under last_k_days"

    def test_on_without_data_raises_with_the_fix(self, synth_df, synth_features, tmp_path):
        with pytest.raises(ValueError, match="backfill_dividends.py"):
            _engine(synth_df, synth_features, tmp_path / "empty.db", "live")

    def test_on_without_a_symbol_raises(self, synth_df, synth_features, div_db):
        with pytest.raises(ValueError, match="needs a symbol"):
            Backtester(**_engine_kwargs(synth_df, synth_features, db_path=div_db,
                                        ex_div_entry_gate="exact"))

    def test_an_unknown_mode_is_refused(self, synth_df, synth_features, div_db):
        with pytest.raises(ValueError, match="ex_div_entry_gate"):
            _engine(synth_df, synth_features, div_db, "sometimes")

    def test_unset_resolves_from_settings_and_bare_engines_are_off(self, synth_df, synth_features):
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        b = BacktestConfig()
        assert (bt._ex_div_mode, bt._ex_div_k) == (b.ex_div_entry_gate, b.ex_div_entry_k_days)
        assert b.ex_div_entry_gate == "off"
        bare = Backtester.__new__(Backtester)
        assert (bare._ex_div_mode, bare._ex_dates) == ("off", None)

    def test_schema_bounds(self):
        assert set(MODES) == {"off", "live", "exact", "last_k_days"}
        with pytest.raises(ValueError):
            BacktestConfig(ex_div_entry_gate="sometimes")
        with pytest.raises(ValueError):
            BacktestConfig(ex_div_entry_k_days=0)


# --------------------------------------------------------------------------
# walk-forward + optimizer forward it
# --------------------------------------------------------------------------

def _wfc(**kw):
    return WalkForwardConfig(train_window_models=False, **kw)


class TestForwarding:
    def test_resolution(self):
        c = _wfc()
        assert (c.ex_div_entry_gate, c.ex_div_entry_k_days) == ("off", 5)   # no overlay loaded
        c = _wfc(ex_div_entry_gate="exact", ex_div_entry_k_days=9)
        assert (c.ex_div_entry_gate, c.ex_div_entry_k_days) == ("exact", 9)

    def test_the_engines_receive_it(self, synth_df, monkeypatch):
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(
            ex_div_entry_gate="exact", ex_div_entry_k_days=9))
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink and (sink[-1]["ex_div_entry_gate"], sink[-1]["ex_div_entry_k_days"]) == ("exact", 9)
        sink.clear()
        wf._train_window_meta_labeler(train_df=synth_df, symbol="SPY", window_id=1,
                                      predictor=None, window_cfg=wf._config)
        assert sink and (sink[-1]["ex_div_entry_gate"], sink[-1]["ex_div_entry_k_days"]) == ("exact", 9)

    def test_the_window_optimizer_receives_it(self, synth_df, monkeypatch):
        import ait.optimization.optimizer as om
        seen = {}

        class Spy:
            def __init__(self, **kw):
                seen.update(kw)

            def run(self, **kw):
                raise RuntimeError("stop here")
        monkeypatch.setattr(om, "StrategyOptimizer", Spy)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=_wfc(
            ex_div_entry_gate="last_k_days", ex_div_entry_k_days=3, optimize_per_window=True))
        wf._optimize_window_params(synth_df, "SPY", 1, ["iron_condor"])
        assert (seen["ex_div_entry_gate"], seen["ex_div_entry_k_days"]) == ("last_k_days", 3)

    def test_trial_engines_carry_it(self, monkeypatch, synth_df):
        sink: list = []
        monkeypatch.setattr(engine_mod, "Backtester", _recorder_factory(sink))
        opt = StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1,
                                ex_div_entry_gate="exact", ex_div_entry_k_days=7)
        opt._data = {"SPY": synth_df}
        opt._run_backtest({})
        assert sink and (sink[0]["ex_div_entry_gate"], sink[0]["ex_div_entry_k_days"]) == ("exact", 7)


# --------------------------------------------------------------------------
# store + backfill script
# --------------------------------------------------------------------------

class TestStoreAndBackfill:
    def test_round_trip(self, tmp_path):
        s = HistoricalDataStore(db_path=tmp_path / "h.db", table_prefix="research_")
        assert s.dividends_span("QQQ") is None and s.load_dividends("QQQ") == []
        rows = [(date(2024, 6, 24), 0.76), (date(2024, 3, 18), 0.57), (date(2024, 9, 23), None)]
        assert s.save_dividends("QQQ", rows, SOURCE_YAHOO_DIVIDENDS, "t") == 3
        assert s.load_dividends("QQQ") == sorted(rows)
        assert s.dividends_span("QQQ") == ("2024-03-18", "2024-09-23", 3)
        assert s.load_dividends("SPY") == []
        s.save_dividends("QQQ", [(date(2024, 3, 18), 0.58)], SOURCE_YAHOO_DIVIDENDS, "t2")
        assert s.load_dividends("QQQ")[0] == (date(2024, 3, 18), 0.58)   # upsert

    @staticmethod
    def _fake_fetch(sym, since):
        return [(d, 0.5) for d in QUARTERLY if d >= since] if sym != "NONE" else []

    def _main(self, tmp_path, *extra):
        import backfill_dividends as bd
        db = tmp_path / "h.db"
        rc = bd.main(["--symbols", "QQQ", "--db-path", str(db), "--report-dir",
                      str(tmp_path / "rep"), *extra], fetch=self._fake_fetch)
        return rc, db

    def test_no_prefix_writes_the_shared_table(self, tmp_path):
        """W91: the empty-prefix guard is gone; the un-prefixed dividends is the one shared table."""
        rc, db = self._main(tmp_path)
        assert rc == 0
        with sqlite3.connect(db) as c:
            assert c.execute("SELECT COUNT(*) FROM dividends").fetchone()[0] > 0

    def test_dry_run_writes_nothing(self, tmp_path):
        rc, db = self._main(tmp_path, "--table-prefix", "research_", "--dry-run")
        assert rc == 0 and not db.exists()
        assert list((tmp_path / "rep").glob("dividends_audit_*.json"))

    def test_a_real_run_stores_with_provenance(self, tmp_path):
        rc, db = self._main(tmp_path, "--table-prefix", "research_", "--since", "2023-06-01")
        assert rc == 0
        s = HistoricalDataStore(db_path=db, table_prefix="research_")
        assert [d for d, _ in s.load_dividends("QQQ")] == [d for d in QUARTERLY if d >= date(2023, 6, 1)]
        import sqlite3
        with sqlite3.connect(db) as c:
            assert {r[0] for r in c.execute("SELECT source FROM research_dividends")} == {
                SOURCE_YAHOO_DIVIDENDS}

    def test_an_empty_fetch_fails_and_writes_nothing(self, tmp_path):
        import backfill_dividends as bd
        db = tmp_path / "h.db"
        rc = bd.main(["--symbols", "QQQ", "NONE", "--table-prefix", "research_", "--db-path",
                      str(db), "--report-dir", str(tmp_path / "rep")], fetch=self._fake_fetch)
        assert rc == 1 and not db.exists()

    def test_summary_flags_off_cycle_gaps(self):
        import backfill_dividends as bd
        rows = [(date(2024, 1, 1), 1.0), (date(2024, 4, 1), 1.0), (date(2024, 4, 10), 0.2)]
        s = bd.summarize(rows)
        assert s["per_year"] == {2024: 3}
        assert s["gaps_outside_band"] == [("2024-04-01", "2024-04-10", 9)]


# --------------------------------------------------------------------------
# registry, manifest, overlay
# --------------------------------------------------------------------------

class TestRegistryManifestOverlay:
    def test_the_reason_is_a_registered_live_twin(self):
        row = gp.classify("ex_div_entry")
        assert row is not None and row.kind == gp.TWIN
        reasons, dynamic = gp.engine_reject_reasons(
            (ROOT / "src" / "ait" / "backtesting" / "engine.py").read_text())
        assert "ex_div_entry" in reasons and dynamic == []

    @pytest.mark.parametrize("mode, warned", [("off", True), ("live", False), ("exact", True),
                                              ("last_k_days", True), (None, False)])
    def test_the_warning_follows_the_mode(self, mode, warned):
        cfg = SimpleNamespace(ex_div_entry_gate=mode, ex_div_entry_k_days=5)
        starts = [w.split(":")[0] for w in gp.one_sided_gate_warnings(cfg, None)]
        assert ("ex_div_entry" in starts) is warned

    def test_the_manifest_records_the_mode(self, monkeypatch):
        import run_backtest as rb
        from ait.config.settings import load_settings
        s = load_settings()
        s = s.model_copy(update={"backtest": s.backtest.model_copy(
            update={"ex_div_entry_gate": "exact"})})
        monkeypatch.setattr(rb, "_research_settings", lambda: s)
        old = sys.argv
        sys.argv = ["run_backtest.py"]
        try:
            m = rb.build_parity_manifest(rb.parse_args())
        finally:
            sys.argv = old
        assert m["backtest"]["ex_div_entry_gate"] == "exact"
        assert any(w.startswith("ex_div_entry:") for w in m["parity_warnings"])

    def test_the_overlay_sets_live_and_the_live_config_does_not(self):
        research = yaml.safe_load((ROOT / "config_research.yaml").read_text())["backtest"]
        live = yaml.safe_load((ROOT / "config.yaml").read_text()).get("backtest") or {}
        assert research["ex_div_entry_gate"] == "live"
        assert "ex_div_entry_gate" not in live and "ex_div_entry_k_days" not in live

    def test_the_loaded_research_view(self):
        from ait.config.research import use_research_config
        from ait.config.settings import load_settings
        use_research_config()
        assert load_settings().backtest.ex_div_entry_gate == "live"

    def test_the_live_bot_never_reads_the_switch(self):
        """AST: outside backtesting/, optimization/ and config/, nothing in src/ait names it."""
        import ast
        names = {"ex_div_entry_gate", "ex_div_entry_k_days"}
        hits = []
        for p in (ROOT / "src" / "ait").rglob("*.py"):
            rel = p.relative_to(ROOT / "src" / "ait").parts
            if rel[0] in ("backtesting", "optimization", "config"):
                continue
            for n in ast.walk(ast.parse(p.read_text())):
                v = (n.attr if isinstance(n, ast.Attribute) else n.arg if isinstance(n, ast.keyword)
                     else n.value if isinstance(n, ast.Constant) else None)
                if v in names:
                    hits.append(f"{p.name}:{n.lineno}")
        assert hits == []

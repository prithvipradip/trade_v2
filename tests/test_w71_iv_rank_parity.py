"""W71: the research IV-rank rise veto vs live's IV-rank gates.

Layers (real code paths, synthetic data only):

  * ``iv_rank_compare``  the pure comparisons, on series with known answers
  * the engine's rise definition is the one the comparator uses
  * the probe script end to end on a tiny temporary sqlite store
  * AST class guard: nothing outside the research modules reads ``iv_rank_rise_threshold``
  * the parity-manifest line
  * W71b: the three live host-date sites now use the ET date (fake host clock one day ahead)
"""
from __future__ import annotations

import ast
import importlib.util
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ait.backtesting import iv_rank_compare as cmp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ait"


def _idx(n, start="2024-01-02"):
    return pd.bdate_range(start, periods=n)


# --------------------------------------------------------------------------
# comparator
# --------------------------------------------------------------------------

class TestComparator:
    def test_rise_is_last_minus_the_value_ten_rows_back(self):
        s = pd.Series(np.arange(30) * 0.05, index=_idx(30))
        r = cmp.rise(s)
        assert r.iloc[:10].isna().all()
        assert r.iloc[10] == pytest.approx(0.5) and r.iloc[-1] == pytest.approx(0.5)

    def test_engines_window_is_the_same_definition(self):
        """engine.py: iv_rank_series = features_df['iv_rank'].iloc[-11:]; rise = last - first."""
        s = pd.Series(np.random.default_rng(3).random(60), index=_idx(60))
        for end in (20, 35, 59):
            window = s.iloc[: end + 1].iloc[-11:]
            assert float(window.iloc[-1]) - float(window.iloc[0]) == pytest.approx(cmp.rise(s).iloc[end])

    def test_veto_days_threshold_is_strict_and_undefined_is_false(self):
        s = pd.Series([0.0] * 10 + [0.30, 0.31, 0.50], index=_idx(13))
        v = cmp.veto_days(s, 0.30)
        assert not v.iloc[:10].any()
        assert list(v.iloc[10:]) == [False, True, True]

    def test_level_gates_match_live_and_unknown_fails_closed(self):
        r = pd.Series([10.0, 15.0, 50.0, 85.0, 90.0, np.nan], index=_idx(6))
        assert list(cmp.live_level_blocks(r)) == [True, False, False, False, True, True]
        assert list(cmp.side(r).fillna("n/a")) == ["below", "in", "in", "in", "above", "n/a"]

    def test_spearman_and_pairs(self):
        a = pd.Series(np.arange(20.0), index=_idx(20))
        assert cmp.spearman(a, a * 3) == pytest.approx(1.0)
        assert cmp.spearman(a, -a) == pytest.approx(-1.0)
        assert cmp.spearman(a.iloc[:5], a.iloc[:5]) is None

    def test_spearman_is_rank_based_not_linear(self):
        a = pd.Series(np.arange(1.0, 21.0), index=_idx(20))
        assert cmp.spearman(a, np.exp(a)) == pytest.approx(1.0)       # monotone, wildly non-linear
        assert float(a.corr(np.exp(a))) < 0.95                        # pearson would NOT say 1.0

    def test_agreement_counts(self):
        a = pd.Series([True, True, False, False, True], index=_idx(5))
        b = pd.Series([True, False, True, False, np.nan], index=_idx(5))
        assert cmp.agreement(a, b) == {"n": 4, "both": 1, "only_a": 1, "only_b": 1, "neither": 1}

    def test_veto_vs_live_separates_blocked_from_extra(self):
        n = 40
        idx = _idx(n)
        f = pd.Series(0.2, index=idx)
        f.iloc[20:] = 0.8                     # a +0.6 jump at row 20: the veto fires for rows 20..29
        s = pd.Series(50.0, index=idx)        # live level: always tradeable...
        s.iloc[20:23] = 95.0                  # ...except rows 20-22, above the cap
        r = cmp.veto_vs_live(f, s, 0.30)
        assert r["n_days"] == n and r["research_veto_days"] == 10
        assert r["already_blocked_by_live"] == 3          # rows 20-22 live refuses anyway
        assert r["extra_restriction_days"] == 7           # rows 23-29: live would trade, research does not
        assert r["live_level_block_days"] == 3
        assert r["extra_share_of_tradeable_days"] == pytest.approx(7 / 37)

    def test_rise_on_the_stored_iv_percentile_is_in_the_same_units(self):
        idx = _idx(40)
        f = pd.Series(0.5, index=idx)                      # feature rank: flat, mid
        s = pd.Series(50.0, index=idx)
        s.iloc[30:] = 56.0                                 # +6 percentile points = 0.06 in 0-1 units
        r = cmp.veto_vs_live(f, s, 0.30)
        assert r["rise_on_stored_iv_days"] == 0            # 6 points is NOT a 0.30 rise
        s.iloc[30:] = 90.0                                 # +40 points = 0.40
        r = cmp.veto_vs_live(f, s, 0.30)
        assert r["rise_on_stored_iv_days"] == 10           # rows 30..39 each see a >0.30 rise
        assert r["veto_agreement"]["only_b"] == 10         # feature rank never fired

    def test_side_disagreement_compares_the_ranks_in_the_same_units(self):
        idx = _idx(30)
        r = cmp.veto_vs_live(pd.Series(0.50, index=idx), pd.Series(50.0, index=idx), 0.30)
        assert r["side_disagreement_days"] == 0            # 0.50 x 100 = 50: both 'in'
        r = cmp.veto_vs_live(pd.Series(0.05, index=idx), pd.Series(50.0, index=idx), 0.30)
        assert r["side_disagreement_days"] == 30           # 5 is below the floor, 50 is not

    def test_nothing_fires_on_a_quiet_series(self):
        idx = _idx(80)
        r = cmp.veto_vs_live(pd.Series(0.5, index=idx), pd.Series(50.0, index=idx), 0.30)
        assert r["research_veto_days"] == 0 and r["extra_restriction_days"] == 0

    def test_module_is_import_light(self):
        tree = ast.parse((SRC / "backtesting" / "iv_rank_compare.py").read_text())
        top = {a.name.split(".")[0] for n in tree.body if isinstance(n, ast.Import) for a in n.names}
        top |= {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.module}
        assert top <= {"__future__", "numpy", "pandas"}


# --------------------------------------------------------------------------
# the probe, end to end on a temporary store
# --------------------------------------------------------------------------

def _make_db(path: Path, n: int = 420) -> None:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    iv = 0.18 + 0.04 * np.sin(np.arange(n) / 25.0) + rng.normal(0, 0.005, n)
    con = sqlite3.connect(path)
    for table in ("research_daily_prices", "daily_prices"):
        con.execute(f"CREATE TABLE {table} (symbol TEXT, date TEXT, open REAL, high REAL, low REAL, "
                    "close REAL, volume REAL, implied_vol REAL, PRIMARY KEY (symbol, date))")
    for k, d in enumerate(idx):
        row = ("TST", d.strftime("%Y-%m-%d"), close[k], close[k] * 1.004, close[k] * 0.996, close[k], 1e6)
        con.execute("INSERT INTO research_daily_prices VALUES (?,?,?,?,?,?,?,?)", (*row, float(iv[k])))
        if k >= 120:          # the live store starts later; it differs from research only from k=250
            con.execute("INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?,?)",
                        (*row, float(iv[k] + (0.01 if k >= 250 else 0.0))))
    con.commit()
    con.close()


def _load_probe():
    spec = importlib.util.spec_from_file_location("iv_rank_parity_probe", ROOT / "scripts" / "iv_rank_parity_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestProbeScript:
    def test_series_and_report_on_a_synthetic_store(self, tmp_path, capsys):
        db = tmp_path / "h.db"
        _make_db(db)
        probe = _load_probe()
        frame, riv, liv, f, s_res, s_live = probe._series("TST", db)
        assert len(riv.dropna()) == 420 and len(liv.dropna()) == 300
        assert f.between(0, 1).all()                         # the feature rank is a 0-1 quantity
        assert s_res.dropna().between(0, 100).all()          # the live rank is a 0-100 percentile
        assert s_live.notna().sum() > 0 and s_live.first_valid_index() > s_res.first_valid_index()
        probe.compare(["TST"], db, 0.30, "2023-09-01")
        out = capsys.readouterr().out
        for section in ("## 1.", "## 2.", "## 3.", "| TST | full history |", "| TST | 2023-09-01+ |"):
            assert section in out
        # section 3: live == research up to row 250, +1 vol point after; count what the table reports
        idx = pd.bdate_range("2023-01-02", periods=420)
        overlap = [d for k, d in enumerate(idx) if k >= 120 and d >= pd.Timestamp("2023-09-01")]
        differing = [d for k, d in enumerate(idx) if k >= 250 and d >= pd.Timestamp("2023-09-01")]
        row = next(l for l in out.splitlines() if l.startswith("| TST | 300 |"))
        cols = [c.strip() for c in row.strip("|").split("|")]
        assert int(cols[2]) == len(overlap)
        assert int(cols[5]) == len(differing) and 0 < len(differing) < len(overlap)
        assert float(cols[4]) == pytest.approx(1.0, abs=0.01)          # p95 diff = 1 vol point

    def test_the_narrow_feature_rank_is_the_engines_series(self, tmp_path):
        """The probe calls FeatureEngine's IV-proxy block directly (speed); it must be exactly
        the iv_rank column compute() hands the engine's veto."""
        from ait.ml.features import FeatureEngine
        db = tmp_path / "h.db"
        _make_db(db)
        probe = _load_probe()
        frame, _riv, _liv, f, _a, _b = probe._series("TST", db)
        full = FeatureEngine().compute(frame[probe.OHLCV])["iv_rank"].astype(float)
        common = full.index.intersection(f.index)
        assert len(common) > 100
        pd.testing.assert_series_equal(f.loc[common], full.loc[common], check_names=False)

    def test_store_is_opened_read_only(self, tmp_path):
        db = tmp_path / "h.db"
        _make_db(db)
        before = db.read_bytes()
        _load_probe()._series("TST", db)
        assert db.read_bytes() == before


class TestReadOnlyIntradayAdapter:
    """ResearchData.load_intraday_range: the engine's intraday_store contract, read-only."""

    def test_returns_utc_indexed_bars_for_the_symbol_and_range_only(self, tmp_path):
        db = tmp_path / "h.db"
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE research_daily_prices (symbol TEXT, date TEXT, open REAL, high REAL, "
                    "low REAL, close REAL, volume REAL, implied_vol REAL)")
        con.execute("CREATE TABLE research_intraday_prices (symbol TEXT, datetime TEXT, interval TEXT, "
                    "open REAL, high REAL, low REAL, close REAL, volume REAL)")
        rows = [("QQQ", "2026-03-10T14:30:00+00:00", "5m", 1, 2, 0.5, 1.5, 100),
                ("QQQ", "2026-03-10T14:35:00+00:00", "5m", 1.5, 2, 1, 1.8, 90),
                ("QQQ", "2026-03-11T14:30:00+00:00", "5m", 9, 9, 9, 9, 9),      # next day
                ("QQQ", "2026-03-10T14:30:00+00:00", "1m", 7, 7, 7, 7, 7),      # wrong interval
                ("SPY", "2026-03-10T14:30:00+00:00", "5m", 5, 5, 5, 5, 5)]      # other symbol
        con.executemany("INSERT INTO research_intraday_prices VALUES (?,?,?,?,?,?,?,?)", rows)
        con.commit()
        con.close()
        before = db.read_bytes()
        from ait.backtesting.replay import ResearchData
        rd = ResearchData(str(db), "QQQ")
        df = rd.load_intraday_range("QQQ", date(2026, 3, 10), date(2026, 3, 10))
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert len(df) == 2 and str(df.index.tz) == "UTC"
        assert df["Close"].tolist() == [1.5, 1.8]
        assert rd.load_intraday_range("QQQ", date(2026, 3, 10), date(2026, 3, 10)) is df     # cached
        spy = rd.load_intraday_range("SPY", date(2026, 3, 10), date(2026, 3, 10))
        assert spy["Close"].tolist() == [5.0]                                                 # symbol filter
        assert db.read_bytes() == before                                                      # read-only


# --------------------------------------------------------------------------
# class guard: live code never reads the research-only threshold
# --------------------------------------------------------------------------

ALLOWED_DIRS = ("backtesting/", "optimization/", "config/")


def _readers(source: str) -> bool:
    tree = ast.parse(source)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "iv_rank_rise_threshold":
            return True
        if isinstance(n, ast.keyword) and n.arg == "iv_rank_rise_threshold":
            return True
        if isinstance(n, ast.Name) and n.id == "iv_rank_rise_threshold":
            return True
        if isinstance(n, ast.Constant) and n.value == "iv_rank_rise_threshold":
            return True
    return False


class TestLiveHasNoRiseVeto:
    def test_nothing_outside_the_research_modules_reads_it(self):
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            if rel.startswith(ALLOWED_DIRS):
                continue
            try:
                if _readers(path.read_text(encoding="utf-8", errors="ignore")):
                    offenders.append(rel)
            except SyntaxError:
                continue
        assert offenders == [], (
            f"{offenders} read iv_rank_rise_threshold. W71 established that live has NO IV-rank rise "
            "veto (it gates on the stored-IV percentile level, floor 15 / cap 85). If the gate was "
            "ported to live on purpose, update this guard and the W71 PLAN entry together.")

    def test_the_detector_sees_every_shape(self):
        for src in ("x = s.iv_rank_rise_threshold", "f(iv_rank_rise_threshold=0.3)",
                    "iv_rank_rise_threshold = 0.3", "d['iv_rank_rise_threshold']"):
            assert _readers(src), src
        assert not _readers("x = s.iv_rank_cap")

    def test_the_research_modules_do_read_it(self):
        """So the guard is not vacuous: the gate really lives in the allowed modules."""
        assert _readers((SRC / "backtesting" / "engine.py").read_text())
        assert _readers((SRC / "optimization" / "optimizer.py").read_text())


# --------------------------------------------------------------------------
# manifest line
# --------------------------------------------------------------------------

class TestManifestLine:
    @staticmethod
    def _manifest(monkeypatch, threshold):
        import run_backtest as rb
        from ait.config.settings import load_settings
        s = load_settings()
        s = s.model_copy(update={"backtest": s.backtest.model_copy(
            update={"iv_rank_rise_threshold": threshold})})
        monkeypatch.setattr(rb, "_research_settings", lambda: s)
        old = sys.argv
        sys.argv = ["run_backtest.py"]
        try:
            return rb.build_parity_manifest(rb.parse_args())
        finally:
            sys.argv = old

    def test_active_veto_is_declared_a_research_only_gate(self, monkeypatch):
        m = self._manifest(monkeypatch, 0.30)
        hits = [w for w in m["parity_warnings"] if w.startswith("iv_rank_rise_threshold")]
        assert len(hits) == 1 and "RESEARCH-ONLY" in hits[0] and "no such veto" in hits[0]

    def test_disabled_veto_is_silent(self, monkeypatch):
        m = self._manifest(monkeypatch, 10.0)
        assert not [w for w in m["parity_warnings"] if w.startswith("iv_rank_rise_threshold")]


# --------------------------------------------------------------------------
# W71b: the three live host-date sites use the ET date
# --------------------------------------------------------------------------

HOST_TOMORROW = datetime(2026, 3, 11, 0, 30)      # a CEST host after midnight: one day ahead of ET
ET_TODAY = date(2026, 3, 10)


class _FakeDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return HOST_TOMORROW if tz is None else HOST_TOMORROW.astimezone(tz)


@pytest.fixture
def orch_mod(monkeypatch):
    import ait.bot.orchestrator as om
    monkeypatch.setattr(om, "datetime", _FakeDatetime)
    monkeypatch.setattr(om, "today_et", lambda: ET_TODAY)
    return om


def _orch(om):
    return om.TradingOrchestrator.__new__(om.TradingOrchestrator)


class TestLiveSitesUseTheEtDate:
    def test_daily_iv_marker_and_row_carry_the_et_date(self, orch_mod):
        o = _orch(orch_mod)
        marks, rows = {}, []
        o._state = SimpleNamespace(get_state=lambda k, d="": marks.get(k, d),
                                   set_state=lambda k, v: marks.__setitem__(k, v))
        o._historical = SimpleNamespace(save_daily_iv=lambda sym, s: (rows.append((sym, s)), 1)[1])
        o._persist_daily_iv("QQQ", [SimpleNamespace(atm_iv=0.20), SimpleNamespace(atm_iv=0.22),
                                     SimpleNamespace(atm_iv=0.30)])
        assert marks == {"iv_saved_QQQ": "2026-03-10"}
        (sym, series), = rows
        assert sym == "QQQ" and series.index[0] == pd.Timestamp("2026-03-10")
        assert series.iloc[0] == pytest.approx(0.22)             # the median across expiries (unchanged)

    def test_daily_iv_is_not_rewritten_when_the_et_day_already_saved(self, orch_mod):
        o = _orch(orch_mod)
        o._state = SimpleNamespace(get_state=lambda k, d="": "2026-03-10", set_state=lambda *a: None)
        o._historical = SimpleNamespace(save_daily_iv=lambda *a: pytest.fail("rewritten"))
        o._persist_daily_iv("QQQ", [SimpleNamespace(atm_iv=0.2)])

    def test_staleness_age_is_measured_against_the_et_date(self, orch_mod, monkeypatch):
        # newest stored IV is 2026-03-05: age 5 at ET 03-10 (fresh, not > 5) but 6 against the
        # host's 03-11 (stale). The ET date must win.
        pairs = [((date(2026, 3, 5) - pd.Timedelta(days=k)).isoformat(), 0.15 + 0.0005 * k)
                 for k in range(100)]

        class _Con:
            def execute(self, *a, **k):
                return SimpleNamespace(fetchall=lambda: pairs)

            def close(self):
                pass
        import sqlite3 as _sq
        monkeypatch.setattr(_sq, "connect", lambda *a, **k: _Con())
        o = _orch(orch_mod)
        o._market_data = SimpleNamespace(get_historical=pytest.fail)    # the proxy must not be reached
        import asyncio
        rank = asyncio.run(o._estimate_iv_rank("QQQ"))
        assert rank is not None and 0.0 <= rank <= 100.0

    def test_commission_trueup_marker_carries_the_et_date(self, orch_mod):
        o = _orch(orch_mod)
        marks = {}
        o._state = SimpleNamespace(
            total_commission=lambda tid: 0.0,
            count_executions_with_commission=lambda tid, exclude_con_id=None: 0,
            set_state=lambda k, v: marks.__setitem__(k, v))
        o._find_trade_by_id = lambda tid: object()
        o._legs_count = lambda t: 4
        assert o._true_up_commission("T1", 12.5) == 12.5
        assert marks == {"trueup_pending_T1": "2026-03-10"}

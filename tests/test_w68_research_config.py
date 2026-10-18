"""W68: the research config overlay (config_research.yaml) and IV-rank parity with live.

Oracle for the IV rank is a verbatim copy of the pre-W68 inline code in
`Orchestrator._estimate_iv_rank`, so the shared function is proven against what it replaced.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ait.backtesting.engine import Backtester
from ait.data.iv_rank import iv_rank_percentile, iv_rank_series

ROOT = Path(__file__).resolve().parent.parent


# ============================================================================ overlay
class TestDeepMergeAndLoader:
    def test_merge_semantics(self):
        from ait.config.settings import _deep_merge
        base = {"a": {"x": 1, "y": [1, 2]}, "b": 5, "c": {"k": 1}}
        out = _deep_merge(base, {"a": {"y": [9], "z": None}, "b": None, "d": 1})
        assert out == {"a": {"x": 1, "y": [9], "z": None}, "b": None, "c": {"k": 1}, "d": 1}
        assert base["a"]["y"] == [1, 2]                     # inputs are not mutated

    def _write(self, tmp_path, base, overlay):
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(base))
        (tmp_path / "config_research.yaml").write_text(yaml.safe_dump(overlay))
        return tmp_path / "config.yaml"

    def test_overlay_wins_per_key_and_the_base_is_untouched(self, tmp_path):
        from ait.config.research import load_research_settings
        from ait.config.settings import load_settings
        cfg = self._write(tmp_path, {"backtest": {"gates_mode": "observe", "iv_floor": 0.2}},
                          {"backtest": {"gates_mode": "enforce"}})
        r = load_research_settings(cfg)
        assert r.backtest.gates_mode == "enforce" and r.backtest.iv_floor == 0.2   # merged
        assert load_settings(cfg).backtest.gates_mode == "observe"                 # live view

    def test_unknown_overlay_key_is_a_startup_error(self, tmp_path):
        from pydantic import ValidationError
        from ait.config.research import load_research_settings
        cfg = self._write(tmp_path, {}, {"backtest": {"gates_modee": "enforce"}})
        with pytest.raises(ValidationError):
            load_research_settings(cfg)

    def test_overlay_may_not_carry_secrets(self, tmp_path):
        from ait.config.research import load_research_settings
        cfg = self._write(tmp_path, {}, {"api_keys": {"finnhub": "x"}})
        with pytest.raises(ValueError, match="secrets"):
            load_research_settings(cfg)

    def test_missing_overlay_refuses_to_run_research_on_the_live_config(self, tmp_path):
        from ait.config.research import load_research_settings
        (tmp_path / "config.yaml").write_text("{}")
        with pytest.raises(FileNotFoundError, match="Research overlay not found"):
            load_research_settings(tmp_path / "config.yaml")

    def test_ignore_frozen_reads_the_files_even_under_a_snapshot(self, tmp_path, monkeypatch):
        import ait.config.settings as st
        cfg = self._write(tmp_path, {"backtest": {"gates_mode": "observe"}}, {})
        frozen = st.load_settings(cfg).model_copy(deep=True)
        frozen.backtest.gates_mode = "enforce"
        monkeypatch.setattr(st, "_FROZEN", frozen)
        assert st.load_settings(cfg).backtest.gates_mode == "enforce"             # snapshot
        assert st.load_settings(cfg, ignore_frozen=True).backtest.gates_mode == "observe"

    def test_use_research_config_freezes_the_merged_settings(self, tmp_path, monkeypatch):
        import ait.config.settings as st
        from ait.config.research import use_research_config
        cfg = self._write(tmp_path, {"backtest": {"gates_mode": "observe"}},
                          {"backtest": {"gates_mode": "enforce"}})
        monkeypatch.setattr(st, "_FROZEN", None)
        use_research_config(cfg)
        try:
            assert st.load_settings().backtest.gates_mode == "enforce"
        finally:
            monkeypatch.setattr(st, "_FROZEN", None)


class TestTheShippedFiles:
    def _yaml(self, name):
        return yaml.safe_load((ROOT / name).read_text())

    def test_live_and_research_views_differ_exactly_where_intended(self):
        from ait.config.research import load_research_settings
        from ait.config.settings import load_settings
        live, res = load_settings(ignore_frozen=True), load_research_settings()
        assert live.ml.entry_gates_enabled is False                 # live stays observe
        assert live.backtest.gates_mode == "observe" and res.backtest.gates_mode == "enforce"
        assert live.backtest.iv_rank_gate is False and res.backtest.iv_rank_gate is True
        assert res.backtest.iv_floor == 0.05 and live.backtest.iv_floor == 0.20
        assert res.backtest.position_size_pct is None
        assert res.backtest.condor_any_direction is False           # 2026-10-10 decision
        assert res.backtest.friction_model == "per_leg_usd"
        # the five live retune values are in config.yaml, so BOTH views carry them
        for s in (live, res):
            assert (s.account.capital, s.risk.max_position_risk_pct,
                    s.risk.max_daily_loss_pct, s.risk.breaker_to_max_loss_ratio) == (
                5000.0, 0.05, 0.045, 1.0)

    def test_overlay_holds_only_real_differences(self):
        """No redundant keys: a key present in both files must hold a different value (else
        the overlay is a stale copy that will silently shadow a future live change)."""
        base, over = self._yaml("config.yaml"), self._yaml("config_research.yaml")
        redundant = []
        for section, kv in over.items():
            for k, v in kv.items():
                if k in (base.get(section) or {}) and base[section][k] == v:
                    redundant.append(f"{section}.{k}")
        assert not redundant, f"overlay keys equal to config.yaml: {redundant}"

    def test_config_yaml_has_no_research_only_backtest_keys(self):
        base = self._yaml("config.yaml")["backtest"]
        research_only = {"history_start_date", "history_days", "research_table_prefix",
                         "vlmc_min_coverage", "macro_event_types", "condor_any_direction",
                         "friction_model", "friction_entry_capture", "friction_exit_capture",
                         "friction_stress_multiplier", "gates_mode", "fractal_veto_spread",
                         "penalty_on_range_prob", "iv_rank_gate", "optimize_hard_min_trades",
                         "optimize_below_floor_gradient", "budget_aware_wings"}
        assert not research_only & set(base), research_only & set(base)
        assert "initial_capital" not in base            # field removed by W63 (extra="forbid")


class TestOverlayNeverReachesLive:
    def test_no_module_under_src_imports_the_research_loader(self):
        offenders = []
        for py in (ROOT / "src" / "ait").rglob("*.py"):
            if py.name == "research.py" and py.parent.name == "config":
                continue
            txt = py.read_text()
            if "ait.config.research" in txt or "use_research_config" in txt \
                    or "config_research" in txt:
                offenders.append(str(py.relative_to(ROOT)))
        assert not offenders, f"live-side code references the research overlay: {offenders}"

    @pytest.mark.parametrize("rel", ["run_backtest.py", "run_optimizer.py",
                                      "scripts/friction_sensitivity.py",
                                      "scripts/gate_study.py",
                                      "scripts/excursion_range_study.py"])   # W83a-2
    def test_every_research_entry_point_activates_it(self, rel):
        tree = ast.parse((ROOT / rel).read_text())
        called = {getattr(n.func, "id", getattr(n.func, "attr", ""))
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert "use_research_config" in called, f"{rel} would run research on the live config"


# ============================================================================ IV rank
def _old_inline(rows_newest_first):
    """Verbatim pre-W68 Orchestrator._estimate_iv_rank percentile block."""
    rows = rows_newest_first
    if len(rows) >= 60:
        cur = rows[0]
        below = sum(1 for v in rows if v < cur)
        rank = below / len(rows) * 100
        return max(0.0, min(100.0, rank))
    return None


class TestSharedIvRank:
    @pytest.mark.parametrize("n", [0, 10, 59, 60, 61, 120, 252, 300])
    def test_equals_the_old_inline_rule(self, n):
        rng = np.random.default_rng(n)
        rows = list(0.1 + rng.random(n) * 0.3)
        want = _old_inline(rows[:252])
        got = iv_rank_percentile(rows)
        assert (got is None and want is None) or got == pytest.approx(want)

    def test_ties_are_not_below_and_bounds(self):
        assert iv_rank_percentile([0.2] * 80) == 0.0                 # nothing strictly below
        assert iv_rank_percentile([1.0] + [0.1] * 79) == pytest.approx(79 / 80 * 100)

    def test_window_is_the_newest_252(self):
        rows = [0.5] + [0.1] * 251 + [9.0] * 50                      # old, huge values ignored
        assert iv_rank_percentile(rows) == pytest.approx(251 / 252 * 100)

    def test_series_matches_the_scalar_for_every_date_and_never_looks_ahead(self):
        rng = np.random.default_rng(1)
        iv = pd.Series(0.15 + rng.random(400) * 0.2, index=pd.bdate_range("2023-01-02", periods=400))
        iv.iloc[[5, 120, 250]] = np.nan                              # missing IV days
        iv.iloc[[7]] = 0.0                                           # non-positive is dropped
        s = iv_rank_series(iv)
        for t in iv.index[::17]:
            hist = iv[:t].where(lambda x: x > 0).dropna()
            if pd.isna(iv[t]) or iv[t] <= 0:
                assert np.isnan(s[t])
                continue
            want = iv_rank_percentile(list(hist.iloc[::-1].values))
            assert (np.isnan(s[t]) and want is None) or s[t] == pytest.approx(want)
        mutated = iv.copy()
        mutated.iloc[300:] = 9.9                                     # change only the FUTURE
        s2 = iv_rank_series(mutated)
        pd.testing.assert_series_equal(s.iloc[:300], s2.iloc[:300])

    def test_live_orchestrator_calls_the_shared_function(self):
        src = (ROOT / "src/ait/bot/orchestrator.py").read_text()
        called = {getattr(n.func, "id", "") for n in ast.walk(ast.parse(src))
                  if isinstance(n, ast.Call)}
        assert "iv_rank_percentile" in called
        assert "below = sum(1 for v in rows if v < cur)" not in src   # no second copy


# ============================================================================ engine gate
@pytest.fixture(scope="module")
def synth_df() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=160)
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(idx))))
    return pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996,
                         "Close": close, "Volume": 1e6}, index=idx)


@pytest.fixture(scope="module")
def synth_features(synth_df) -> pd.DataFrame:
    from ait.ml.features import FeatureEngine
    return FeatureEngine().compute(synth_df)


def _kw(df, feats, **over) -> dict:
    kw = dict(data=df, strategies=["iron_condor"], initial_capital=100_000,
              macro_event_gate=False, allow_live_model_fallback=False,
              hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0,
              iv_rank_rise_threshold=10.0, ic_min_credit=0.0, ic_min_credit_width=0.0,
              iv_floor=0.01, features_cache=feats)
    kw.update(over)
    return kw


def _rank(df, v):
    return pd.Series(v, index=df.index, dtype=float)


def _top(f):
    return sum(x for k, x in f.items() if k not in ("days_evaluated", "entries_taken")
               and not k.startswith(("builder:", "would_reject:")))


class TestEngineIvRankGate:
    def test_gate_off_is_the_historical_behaviour(self, synth_df, synth_features):
        res = Backtester(**_kw(synth_df, synth_features, iv_rank_series=_rank(synth_df, 1.0))).run()
        assert res.total_trades > 0
        assert not any(k.startswith("iv_rank_") for k in res.entry_funnel)

    def test_below_the_floor_is_refused_and_recorded(self, synth_df, synth_features):
        res = Backtester(**_kw(synth_df, synth_features, iv_rank_gate=True,
                               iv_rank_series=_rank(synth_df, 5.0))).run()   # 5 < 15
        assert res.total_trades == 0 and res.entry_funnel["iv_rank_floor"] > 0

    def test_above_the_cap_is_refused(self, synth_df, synth_features):
        res = Backtester(**_kw(synth_df, synth_features, iv_rank_gate=True,
                               iv_rank_series=_rank(synth_df, 95.0))).run()  # 95 > 85
        assert res.total_trades == 0 and res.entry_funnel["iv_rank_cap"] > 0

    def test_inside_the_band_trades_and_records_the_rank(self, synth_df, synth_features):
        res = Backtester(**_kw(synth_df, synth_features, iv_rank_gate=True,
                               iv_rank_series=_rank(synth_df, 50.0))).run()
        assert res.total_trades > 0
        assert all(t["decision"]["iv_rank"] == 50.0 for t in res.trades)

    def test_boundaries_match_live_strictness(self, synth_df, synth_features):
        """Live: refuse rank < 15 and rank > 85 -- exactly 15 and exactly 85 trade."""
        for v in (15.0, 85.0):
            res = Backtester(**_kw(synth_df, synth_features, iv_rank_gate=True,
                                   iv_rank_series=_rank(synth_df, v))).run()
            assert res.total_trades > 0, v

    def test_unmeasurable_rank_fails_closed(self, synth_df, synth_features):
        res = Backtester(**_kw(synth_df, synth_features, iv_rank_gate=True,
                               iv_rank_series=_rank(synth_df, np.nan))).run()
        assert res.total_trades == 0 and res.entry_funnel["iv_rank_unknown"] > 0

    def test_no_iv_column_and_no_series_fails_closed_not_open(self, synth_df, synth_features):
        res = Backtester(**_kw(synth_df, synth_features, iv_rank_gate=True)).run()
        assert res.total_trades == 0 and res.entry_funnel["iv_rank_unknown"] > 0

    def test_series_is_derived_from_the_engines_own_iv_column(self, synth_df, synth_features):
        df = synth_df.copy()
        rng = np.random.default_rng(3)
        df["implied_vol"] = 0.15 + rng.random(len(df)) * 0.1
        from ait.ml.features import FeatureEngine
        bt = Backtester(**_kw(df, FeatureEngine().compute(df), iv_rank_gate=True))
        want = iv_rank_series(df["implied_vol"]).to_numpy()
        np.testing.assert_allclose(bt._iv_rank_arr, want, equal_nan=True)

    def test_conservation_holds_with_the_new_reasons(self, synth_df, synth_features):
        ser = _rank(synth_df, 50.0)
        ser.iloc[40:80] = 5.0
        ser.iloc[80:100] = 95.0
        ser.iloc[100:110] = np.nan
        f = Backtester(**_kw(synth_df, synth_features, iv_rank_gate=True,
                             iv_rank_series=ser)).run().entry_funnel
        assert sum(f.get(k, 0) for k in ("iv_rank_floor", "iv_rank_cap", "iv_rank_unknown")) > 0
        assert f["days_evaluated"] == f.get("entries_taken", 0) + _top(f)

    def test_rank_uses_the_precomputed_series_not_the_window_slice(self, synth_df, synth_features):
        """A short test window must see live's 252-obs lookback: the series handed in (computed
        on the FULL history) wins over anything derivable from the sliced frame."""
        ser = _rank(synth_df, 50.0)
        bt = Backtester(**_kw(synth_df, synth_features, iv_rank_gate=True, iv_rank_series=ser))
        assert np.all(bt._iv_rank_arr == 50.0)


# ============================================================================ plumbing guards
class TestIvRankPlumbing:
    def test_every_walkforward_backtester_passes_the_rank_kwargs(self):
        src = inspect.getsource(importlib.import_module("ait.backtesting.walkforward"))
        calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) == "Backtester"]
        need = {"iv_rank_gate", "iv_rank_series"}
        bad = [(c.lineno, sorted(need - {k.arg for k in c.keywords})) for c in calls
               if not need <= {k.arg for k in c.keywords}]
        assert calls and not bad, bad

    def test_the_optimizer_trial_backtester_carries_them(self):
        src = inspect.getsource(importlib.import_module("ait.optimization.optimizer"))
        tree = ast.parse(src)
        keys = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert "iv_rank_gate" in keys                                  # bt_kwargs literal
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) == "Backtester"]
        assert all("iv_rank_series" in {k.arg for k in c.keywords} for c in calls) and calls

    def test_walkforward_builds_the_series_once_over_the_full_history(self):
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        wf = WalkForwardBacktester.__new__(WalkForwardBacktester)
        wf._config = WalkForwardConfig(iv_rank_gate=True)
        wf._iv_rank_series = {}
        idx = pd.bdate_range("2020-01-01", periods=400)
        df = pd.DataFrame({"implied_vol": np.linspace(0.1, 0.4, 400)}, index=idx)
        wf._ensure_iv_rank_series({"SPY": df, "NOIV": pd.DataFrame({"Close": [1.0]})})
        assert set(wf._iv_rank_series) == {"SPY"}
        first = wf._iv_rank_series["SPY"]
        wf._ensure_iv_rank_series({"SPY": df})
        assert wf._iv_rank_series["SPY"] is first                        # not rebuilt
        assert first.iloc[-1] > 99.0                                      # rising IV -> top rank
        assert len(first) == 400                                          # full history, not a slice

    def test_gate_off_builds_nothing(self):
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        wf = WalkForwardBacktester.__new__(WalkForwardBacktester)
        wf._config = WalkForwardConfig(iv_rank_gate=False)
        wf._iv_rank_series = {}
        wf._ensure_iv_rank_series({"SPY": pd.DataFrame({"implied_vol": [0.2] * 80})})
        assert wf._iv_rank_series == {}

    def test_config_default_is_off_and_bounds(self):
        from ait.config.settings import BacktestConfig
        assert BacktestConfig().iv_rank_gate is False

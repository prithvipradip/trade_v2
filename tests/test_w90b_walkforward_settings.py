"""W90b: the walk-forward geometry and evaluation constants are ``backtest.walkforward_tuning`` settings.

Window lengths (365 / 63 / 63 / 5), the Optuna trial counts (50 / 100), the row floors (50 / 5), the context and fetch
paddings (900 / 400 / 100 / 30), the out-of-sample diagnostics (2% / 8 bins / 9 bins / +20 rows) and the VLMC coverage
were literals in ``WalkForwardConfig`` and the helper methods. Each now resolves from the loaded (research) settings;
the defaults equal the old literals, so a default run is unchanged, and every field is shown to MOVE behaviour.
"""
from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ait.backtesting.walkforward as wf_mod
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.config.settings import WalkForwardTuningConfig, load_settings
from tests.test_r20_research_validity import _recorder_factory, _run_window, synth_df  # noqa: F401  (fixture)

OLD_LITERALS = {
    "train_days": 365, "test_days": 63, "step_days": 63, "gap_days": 5, "optimize_n_trials": 50, "optimizer_n_trials": 100,
    "min_train_rows": 50, "min_test_rows": 5, "context_padding_days": 900, "context_span_padding_days": 400,
    "fetch_padding_days": 100, "history_span_padding_days": 30, "oos_direction_return_threshold": 0.02,
    "oos_calibration_bins": 8, "prob_score_bins": 9, "oos_min_extra_rows": 20,
}


def _freeze_tuning(monkeypatch, **tuning):
    """Freeze the loaded settings with ``walkforward_tuning`` fields moved."""
    import ait.config.settings as st
    s = load_settings()
    moved = s.backtest.walkforward_tuning.model_copy(update=tuning)
    monkeypatch.setattr(st, "_FROZEN", s.model_copy(update={"backtest": s.backtest.model_copy(update={"walkforward_tuning": moved})}))
    return st._FROZEN


def test_the_defaults_equal_the_old_literals_everywhere():
    schema = WalkForwardTuningConfig()
    loaded = load_settings().backtest.walkforward_tuning
    for name, old in OLD_LITERALS.items():
        assert getattr(schema, name) == old, name
        assert getattr(loaded, name) == old, f"config_research.yaml / config.yaml moved {name} (a deliberate change needs a PLAN entry)"


class TestWalkForwardConfigSentinels:
    def test_a_bare_config_is_the_old_defaults(self):
        c = WalkForwardConfig(train_window_models=False)
        assert (c.train_days, c.test_days, c.step_days, c.gap_days, c.optimize_n_trials) == (365, 63, 63, 5, 50)
        assert c.vlmc_min_coverage == load_settings().backtest.vlmc_min_coverage == 0.95

    def test_each_field_follows_the_loaded_settings_and_explicit_wins(self, monkeypatch):
        _freeze_tuning(monkeypatch, train_days=300, test_days=40, step_days=20, gap_days=2, optimize_n_trials=7)
        c = WalkForwardConfig(train_window_models=False)
        assert (c.train_days, c.test_days, c.step_days, c.gap_days, c.optimize_n_trials) == (300, 40, 20, 2, 7)
        e = WalkForwardConfig(train_window_models=False, train_days=111, test_days=22, step_days=33, gap_days=4, optimize_n_trials=9)
        assert (e.train_days, e.test_days, e.step_days, e.gap_days, e.optimize_n_trials) == (111, 22, 33, 4, 9)

    def test_vlmc_coverage_is_the_backtest_setting(self, monkeypatch):
        import ait.config.settings as st
        s = load_settings()
        monkeypatch.setattr(st, "_FROZEN", s.model_copy(update={"backtest": s.backtest.model_copy(update={"vlmc_min_coverage": 0.8})}))
        assert WalkForwardConfig(train_window_models=False).vlmc_min_coverage == 0.8
        assert WalkForwardConfig(train_window_models=False, vlmc_min_coverage=0.99).vlmc_min_coverage == 0.99

    def test_the_dead_range_threshold_field_is_gone(self):
        """Nothing read ``range_threshold_pct``; a setting that does nothing is a trap, the live band is range_threshold_fallback."""
        assert "range_threshold_pct" not in WalkForwardConfig.__dataclass_fields__

    def test_windows_are_generated_from_the_resolved_lengths(self, monkeypatch):
        _freeze_tuning(monkeypatch, train_days=100, test_days=20, step_days=20, gap_days=0)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))
        idx = pd.date_range("2024-01-01", "2025-06-30", freq="D")
        wins = wf._generate_windows({"SPY": pd.DataFrame({"Close": 1.0}, index=idx)})
        (tr_s, tr_e, te_s, te_e), second = wins[0], wins[1]
        assert (tr_e - tr_s).days == 100 and (te_e - te_s).days == 20 and (te_s - tr_e).days == 0
        assert (second[0] - tr_s).days == 20                                  # step_days


class TestWindowRowFloors:
    def _run(self, monkeypatch, synth_df, **tuning):
        _freeze_tuning(monkeypatch, **tuning)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        return sink

    def test_the_default_floors_let_the_window_run(self, monkeypatch, synth_df):
        assert self._run(monkeypatch, synth_df), "the default 50 / 5 row floors skipped a ~100-bar window"

    def test_min_train_rows_skips_the_symbol(self, monkeypatch, synth_df):
        assert not self._run(monkeypatch, synth_df, min_train_rows=2000)

    def test_min_test_rows_skips_the_symbol(self, monkeypatch, synth_df):
        assert not self._run(monkeypatch, synth_df, min_test_rows=2000)

    def test_pretrain_uses_min_train_rows(self, monkeypatch, synth_df):
        windows = [(date(2024, 1, 1), date(2024, 6, 1), date(2024, 6, 10), date(2024, 9, 1))]
        calls = []

        def _spy(self, *a, **k):
            calls.append(1)
            return (None, "spy", 0.05)

        monkeypatch.setattr(WalkForwardBacktester, "_train_window_range_model_inprocess", _spy)
        _freeze_tuning(monkeypatch, min_train_rows=2000)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=True))
        out = wf._pretrain_range_models(windows, {"SPY": synth_df}, pd.DataFrame())
        assert out[0]["SPY"][1] == "insufficient_train_data" and not calls
        _freeze_tuning(monkeypatch, min_train_rows=50)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=True))
        out = wf._pretrain_range_models(windows, {"SPY": synth_df}, pd.DataFrame())
        assert out[0]["SPY"][1] != "insufficient_train_data"


class TestPaddings:
    def test_context_days(self, monkeypatch):
        _freeze_tuning(monkeypatch, context_padding_days=111, context_span_padding_days=222)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(
            train_window_models=False, train_days=365, history_start_date=None, history_days=None))
        assert wf._context_days() == 365 + 111               # no span configured: the padding on the train window
        wf._config.history_days = 4000
        assert wf._context_days() == 4000 + 222              # a span configured: the span plus its own padding

    def test_fetch_days(self, monkeypatch):
        import ait.data.market_data as md
        seen = []
        monkeypatch.setattr(md, "load_daily_ohlcv", lambda s, days=0, db_path=None, **kw: seen.append(days) or pd.DataFrame())
        _freeze_tuning(monkeypatch, fetch_padding_days=7, history_span_padding_days=3)
        cfg = WalkForwardConfig(train_window_models=False, train_days=100, test_days=20)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        asyncio.run(wf._fetch_data())
        assert seen[-1] == 100 + 20 + 7
        cfg.history_days = None
        cfg.history_start_date = date(2000, 1, 3)
        asyncio.run(wf._fetch_data())
        from ait.utils.time import today_et
        assert seen[-1] == (today_et() - date(2000, 1, 3)).days + 3


class _Scaler:
    def transform(self, x):
        return np.asarray(x, dtype=float)


class _Model:
    """A 3-class model whose confidence runs 0.34 -> 0.99 over the rows (so the calibration bins fill)."""

    def predict_proba(self, X):
        n = len(X)
        conf = np.linspace(0.34, 0.99, n)
        rest = (1 - conf) / 2
        out = np.column_stack([rest, conf, rest])
        return out


def _direction_inputs(n=120):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.where(np.arange(n) % 2 == 0, 0.015, -0.01))
    df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6}, index=idx)
    feats = pd.DataFrame({"f1": np.arange(n, dtype=float)}, index=idx)
    pred = SimpleNamespace(_symbol_models={"SPY": {"models": {"xgb": _Model()}, "scaler": _Scaler(), "feature_names": ["f1"]}},
                           _fitted_weights={"xgb": 1.0})
    return pred, df, feats


class TestOosDiagnostics:
    def _wf(self, monkeypatch, **tuning):
        _freeze_tuning(monkeypatch, **tuning)
        return WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))

    def test_direction_threshold_bins_and_rows(self, monkeypatch):
        pred, df, feats = _direction_inputs()
        wf = self._wf(monkeypatch)
        base = wf._evaluate_direction_model_oos(pred, df, "SPY", horizon_days=5, precomputed_features=feats)
        assert base["return_threshold"] == 0.02
        assert max(b["p"] for b in base["ensemble"]["calibration_bins"]) == pytest.approx(0.9375, abs=0.001)   # top of 8 bins
        assert len(base["ensemble"]["calibration_bins"]) <= 8
        wf = self._wf(monkeypatch, oos_direction_return_threshold=0.0123, oos_calibration_bins=4)
        moved = wf._evaluate_direction_model_oos(pred, df, "SPY", horizon_days=5, precomputed_features=feats)
        assert moved["return_threshold"] == 0.0123
        assert {b["p"] for b in moved["ensemble"]["calibration_bins"]} <= {0.125, 0.375, 0.625, 0.875}
        assert len(moved["ensemble"]["calibration_bins"]) <= 4
        # an explicit argument still wins over the setting
        assert wf._evaluate_direction_model_oos(pred, df, "SPY", horizon_days=5, return_threshold_pct=0.05,
                                                precomputed_features=feats)["return_threshold"] == 0.05

    def test_direction_needs_min_extra_rows(self, monkeypatch):
        pred, df, feats = _direction_inputs(n=40)
        wf = self._wf(monkeypatch)                                   # 40 rows >= 5 + 20: evaluated
        assert wf._evaluate_direction_model_oos(pred, df, "SPY", horizon_days=5, precomputed_features=feats)
        wf = self._wf(monkeypatch, oos_min_extra_rows=45)            # 40 rows < 5 + 45: not enough
        assert wf._evaluate_direction_model_oos(pred, df, "SPY", horizon_days=5, precomputed_features=feats) == {}

    def test_range_threshold_defaults_to_the_fallback_band(self, monkeypatch):
        pred, df, feats = _direction_inputs(n=200)
        rng = np.random.default_rng(3)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(df))))   # a 1.2%-daily-vol walk: 5-day swings of 1-6%
        df = df.assign(Close=close, Open=close, High=close, Low=close)
        wf = self._wf(monkeypatch)
        default = wf._evaluate_range_model_oos(pred, df, "SPY", horizon_days=5, precomputed_features=feats)
        explicit = wf._evaluate_range_model_oos(pred, df, "SPY", threshold_pct=0.05, horizon_days=5, precomputed_features=feats)
        assert default and default["base_rate"] == explicit["base_rate"]       # None -> the 0.05 fallback band
        wf = self._wf(monkeypatch, range_threshold_fallback=0.02)
        moved = wf._evaluate_range_model_oos(pred, df, "SPY", horizon_days=5, precomputed_features=feats)
        assert moved and moved["base_rate"] < default["base_rate"]             # a tighter band contains fewer paths

    def test_prob_score_bins(self, monkeypatch):
        y = np.array([0, 1] * 30)
        p = np.linspace(0.05, 0.95, 60)
        _freeze_tuning(monkeypatch, prob_score_bins=4)
        out = WalkForwardBacktester._prob_score_metrics(y, p, float(y.mean()))
        assert len(out["calibration_bins"]) <= 4 and {b["p"] for b in out["calibration_bins"]} <= {0.125, 0.375, 0.625, 0.875}
        assert len(WalkForwardBacktester._prob_score_metrics(y, p, 0.5, n_bins=9)["calibration_bins"]) > 4


def test_optuna_window_data_trial_count_follows_the_setting(monkeypatch):
    class _NoStudy:
        study = None
    _freeze_tuning(monkeypatch, optimize_n_trials=13)
    out = wf_mod._build_optuna_window_data([_NoStudy(), _NoStudy()], ["a", "b"], 1, "SPY")
    assert out["meta"]["n_trials_requested"] == 26 and out["meta"]["sampler"] == "TPESampler(seed=42)"
    explicit = wf_mod._build_optuna_window_data([_NoStudy()], ["a"], 1, "SPY", n_trials_requested=5, seed=9)
    assert explicit["meta"]["n_trials_requested"] == 5 and explicit["meta"]["sampler"] == "TPESampler(seed=9)"
    _freeze_tuning(monkeypatch, optimize_seed=11)
    assert wf_mod._build_optuna_window_data([], [], 1, "SPY")["meta"]["sampler"] == "TPESampler(seed=11)"


class TestRangeBankFallback:
    def test_failed_horizons_get_the_configured_band(self, monkeypatch):
        from ait.backtesting.range_bank import RangeModelBank
        _freeze_tuning(monkeypatch, range_threshold_fallback=0.031)

        def boom(h):
            raise RuntimeError("no model")
        assert RangeModelBank(boom, lambda *a, **k: 5).get(5)[2] == 0.031
        assert RangeModelBank(lambda h: None, lambda *a, **k: 5).get(5)[2] == 0.031


class TestReportingAndFallbacks:
    def test_range_oos_scoring_uses_the_instances_bin_count(self, monkeypatch):
        """The instance's tuning (not just the frozen settings) sets the range reliability-chart bins."""
        pred, df, feats = _direction_inputs(n=200)
        rng = np.random.default_rng(3)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(df))))
        df = df.assign(Close=close, Open=close, High=close, Low=close)
        _freeze_tuning(monkeypatch)                                  # frozen: 9 bins
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))
        wf._tuning = WalkForwardTuningConfig(prob_score_bins=3)      # the instance differs from the frozen view
        out = wf._evaluate_range_model_oos(pred, df, "SPY", threshold_pct=0.03, horizon_days=5, precomputed_features=feats)
        bins = out["ml"]["xgb"]["calibration_bins"]
        assert bins and len(bins) <= 3 and len(out["ensemble"]["calibration_bins"]) <= 3

    def test_the_optuna_report_carries_the_configs_seed_and_trial_count(self, monkeypatch, synth_df):
        import ait.optimization.optimizer as optimizer_mod

        class _Opt:
            def __init__(self, **kw):
                pass

            def run(self, data, prior_params=None):
                return SimpleNamespace(best_params={}, best_value=0.0, study=None, usable=True, best_n_trades=0)

        seen = {}
        real = wf_mod._build_optuna_window_data

        def _spy(*a, **k):
            seen.update(k)
            return real(*a, **k)
        monkeypatch.setattr(optimizer_mod, "StrategyOptimizer", _Opt)
        monkeypatch.setattr(wf_mod, "_build_optuna_window_data", _spy)
        cfg = WalkForwardConfig(train_window_models=False, optimize_per_window=True, optimize_seed=31, optimize_n_trials=6)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        wf._optimize_window_params(synth_df, "SPY", 1, ["iron_condor"])
        assert (seen["seed"], seen["n_trials_requested"]) == (31, 6)

    def test_the_range_bank_fallback_without_a_config_is_the_schema_default(self, monkeypatch):
        import ait.config.settings as st
        from ait.backtesting.range_bank import RangeModelBank
        monkeypatch.setattr(st, "load_settings", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no config")))
        real = st.WalkForwardTuningConfig
        monkeypatch.setattr(st, "WalkForwardTuningConfig", lambda: real(range_threshold_fallback=0.077))
        assert RangeModelBank(lambda h: None, lambda *a, **k: 5).get(5)[2] == 0.077


class TestOptimizerAndClis:
    def test_optimizer_trials_and_history_window(self, monkeypatch):
        from ait.optimization.optimizer import StrategyOptimizer
        s = _freeze_tuning(monkeypatch, optimizer_n_trials=17, train_days=222)
        o = StrategyOptimizer(symbols=["QQQ"], strategies=["iron_condor"])
        assert (o._n_trials, o._train_days) == (17, 222)
        e = StrategyOptimizer(symbols=["QQQ"], strategies=["iron_condor"], n_trials=3, train_days=44, settings=s)
        assert (e._n_trials, e._train_days) == (3, 44)

    def test_run_backtest_cli_defaults_are_sentinels_resolved_from_the_research_view(self, monkeypatch):
        import sys

        import run_backtest as rb
        monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
        a = rb.parse_args()
        assert (a.train_days, a.test_days, a.step_days, a.gap_days, a.optimize_n_trials) == (None,) * 5
        assert not hasattr(rb, "_WF_DEFAULTS")

    def test_run_backtest_resolves_the_unset_flags_from_the_tuning(self):
        import run_backtest as rb
        tuning = WalkForwardTuningConfig(train_days=300, test_days=40, step_days=20, gap_days=2, optimize_n_trials=7)
        unset = SimpleNamespace(train_days=None, test_days=None, step_days=None, gap_days=None, optimize_n_trials=None)
        rb.resolve_walkforward_args(unset, tuning)
        assert (unset.train_days, unset.test_days, unset.step_days, unset.gap_days, unset.optimize_n_trials) == (300, 40, 20, 2, 7)
        given = SimpleNamespace(train_days=1, test_days=2, step_days=3, gap_days=4, optimize_n_trials=5)
        rb.resolve_walkforward_args(given, tuning)
        assert (given.train_days, given.test_days, given.step_days, given.gap_days, given.optimize_n_trials) == (1, 2, 3, 4, 5)

    def test_run_optimizer_resolves_the_unset_flags_from_the_tuning(self):
        import run_optimizer as ro
        tuning = WalkForwardTuningConfig(optimizer_n_trials=17, train_days=222)
        unset = SimpleNamespace(n_trials=None, train_days=None)
        ro.resolve_optimizer_args(unset, tuning)
        assert (unset.n_trials, unset.train_days) == (17, 222)
        given = SimpleNamespace(n_trials=3, train_days=44)
        ro.resolve_optimizer_args(given, tuning)
        assert (given.n_trials, given.train_days) == (3, 44)

    def test_run_optimizer_cli_defaults_are_sentinels(self, monkeypatch):
        import sys

        import run_optimizer as ro
        monkeypatch.setattr(sys, "argv", ["run_optimizer.py"])
        a = ro.parse_args()
        assert (a.n_trials, a.train_days) == (None, None)

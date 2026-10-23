"""W90e: stale copies of existing settings now read them (group A), the fallback volatility is a setting (B), the meta-label decision
record reports the real cut (C), and the walk-forward guards are settings (E).

The census, once its ignore list was narrowed, showed small numbers that repeated a setting (StrategyOptimizer's ``min_trades=10``, the
meta-labeler's ``min_probability=0.50``, the ex-dividend ``k_days=5``, three ``30``-day horizons, the DTE band edges), a 0.25 fallback
volatility in five places, a meta-label threshold written as 0.5 whatever the labeler's cut was, and the walk-forward's window minimum,
score shrink and out-of-sample minimums. Defaults equal the old literals.
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ait.backtesting.walkforward as wf_mod
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.config.settings import (
    BacktestConfig,
    EngineTuningConfig,
    MetaLabelConfig,
    WalkForwardTuningConfig,
    load_settings,
)
from tests.test_w74_engine_realism import _bt, _frame, _pos, _session


def _moved(dotted: str, **vals):
    """Loaded settings with ``section[.sub]`` fields moved."""
    s = load_settings()
    top, _, sub = dotted.partition(".")
    cur = getattr(s, top)
    new = cur.model_copy(update={sub: getattr(cur, sub).model_copy(update=vals)}) if sub else cur.model_copy(update=vals)
    return s.model_copy(update={top: new})


def _freeze(monkeypatch, settings):
    import ait.config.settings as st
    monkeypatch.setattr(st, "_FROZEN", settings)


def test_the_new_defaults_equal_the_old_literals():
    e, w = EngineTuningConfig(), WalkForwardTuningConfig()
    assert e.fallback_iv == 0.25 == load_settings().backtest.engine_tuning.fallback_iv
    got = (w.oos_min_labelled_rows, w.oos_range_min_extra_rows, w.min_windows, w.window_score_full_credit_trades)
    assert got == (10, 5, 3, 5) == tuple(getattr(load_settings().backtest.walkforward_tuning, n) for n in (
        "oos_min_labelled_rows", "oos_range_min_extra_rows", "min_windows", "window_score_full_credit_trades"))


class TestGroupA:
    def test_optimizer_min_trades_follows_the_setting_and_explicit_wins(self):
        from ait.optimization.optimizer import StrategyOptimizer
        kw = dict(symbols=["QQQ"], strategies=["iron_condor"], n_trials=1)
        assert StrategyOptimizer(**kw)._min_trades == load_settings().backtest.optimize_min_trades == 10
        assert StrategyOptimizer(**kw, settings=_moved("backtest", optimize_min_trades=17))._min_trades == 17
        assert StrategyOptimizer(**kw, settings=_moved("backtest", optimize_min_trades=17), min_trades=4)._min_trades == 4

    def test_meta_labeler_min_probability_follows_the_setting_and_explicit_wins(self, monkeypatch):
        from ait.ml.meta_label import MetaLabeler
        assert MetaLabeler().min_probability == MetaLabelConfig().min_probability == 0.50
        _freeze(monkeypatch, _moved("meta_label", min_probability=0.62))
        assert MetaLabeler().min_probability == 0.62
        assert MetaLabeler(min_probability=0.55).min_probability == 0.55

    def test_ex_div_k_days_follows_the_setting_and_explicit_wins(self, monkeypatch):
        """last_k_days blocks when a real ex-date falls in the K days before expiry: here it is 14 days before."""
        from ait.backtesting.ex_div import ex_div_blocks
        quarterly = [date(2023, 3, 17), date(2023, 6, 16), date(2023, 9, 15)]
        ex = quarterly[1]
        args = ("last_k_days", quarterly, ex - timedelta(days=1), ex + timedelta(days=14))
        assert ex_div_blocks(*args)[0] is False                      # the default window (5 days) does not reach it
        _freeze(monkeypatch, _moved("backtest", ex_div_entry_k_days=20))
        assert ex_div_blocks(*args)[0] is True                       # a wider configured window does
        assert ex_div_blocks(*args, 3)[0] is False                   # explicit wins over the setting

    def test_without_a_config_the_fallbacks_are_the_schema_defaults(self, monkeypatch):
        import ait.config.settings as st
        from ait.backtesting import dte_report as dr
        from ait.backtesting.ex_div import ex_div_blocks
        monkeypatch.setattr(st, "load_settings", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no config")))
        assert dr.band_order() == ["0-2", "3-7", "8-14", "15-45", "46+"]
        ex = date(2023, 6, 16)
        args = ("last_k_days", [date(2023, 3, 17), ex, date(2023, 9, 15)], ex - timedelta(days=1), ex + timedelta(days=6))
        assert ex_div_blocks(*args)[0] is False                      # 6 days before expiry is outside the schema's 5-day window
        real = st.BacktestConfig.model_fields["ex_div_entry_k_days"]
        monkeypatch.setattr(real, "default", 7)
        assert ex_div_blocks(*args)[0] is True                       # ... and inside a 7-day one: the schema default is what is read

    def test_the_partial_engine_default_is_the_schema_default(self):
        from ait.backtesting.engine import Backtester
        assert Backtester._ex_div_k == BacktestConfig().ex_div_entry_k_days

    def test_the_walk_forward_horizon_defaults(self, monkeypatch):
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False, max_hold_days=21))
        seen = {}
        monkeypatch.setattr(wf, "_adaptive_range_threshold", lambda df, horizon_days: seen.setdefault("h", horizon_days) or 0.05)
        monkeypatch.setattr(wf, "_train_window_range_model_inprocess", lambda *a, **k: (None, "x", 0.05))
        monkeypatch.setattr(wf, "_range_label_horizon", lambda: 6)
        wf._train_window_range_model(_frame(120), "SPY", 1)
        assert seen["h"] == 6                                          # None -> the reachable range-label horizon (was 30)
        seen.clear()
        wf._train_window_range_model(_frame(120), "SPY", 1, max_hold_days=11)
        assert seen["h"] == 11

    def test_the_oos_evaluation_horizon_defaults(self, monkeypatch):
        from tests.test_w90b_walkforward_settings import _direction_inputs
        pred, df, feats = _direction_inputs(120)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False, max_hold_days=7))
        out = wf._evaluate_direction_model_oos(pred, df, "SPY", precomputed_features=feats)
        assert out["horizon_days"] == 7                                # None -> backtest.max_hold_days (was 30)
        monkeypatch.setattr(wf, "_range_label_horizon", lambda: 4)
        rng = np.random.default_rng(3)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(df))))
        r = wf._evaluate_range_model_oos(pred, df.assign(Close=close), "SPY", threshold_pct=0.03, precomputed_features=feats)
        assert r["n_samples"] == len(df) - 4                           # None -> the range-label horizon (was 30)

    def test_the_dte_report_bands_follow_the_parity_setting(self, monkeypatch):
        from ait.backtesting import dte_report as dr
        assert dr.band_order() == ["0-2", "3-7", "8-14", "15-45", "46+"]
        assert dr.band_label(10) == "8-14"
        _freeze(monkeypatch, _moved("parity", dte_band_upper_edges=[3, 10]))
        assert dr.band_order() == ["0-3", "4-10", "11+"] and dr.band_label(10) == "4-10"
        assert dr.band_label(10, (30,)) == "0-30"                      # explicit wins


class TestFallbackVolatility:
    def test_realized_vol_with_too_few_prices(self, monkeypatch):
        from ait.backtesting.pricing import realized_vol
        assert realized_vol(np.array([100.0, 101.0])) == 0.25
        _freeze(monkeypatch, _moved("backtest.engine_tuning", fallback_iv=0.33))
        assert realized_vol(np.array([100.0, 101.0])) == 0.33
        assert realized_vol(np.array([100.0, 101.0]), fallback=0.4) == 0.4
        closes = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 60)))
        assert realized_vol(closes) != 0.33                                        # enough prices: measured

    def _spy_bt(self, **kw):
        bt = _bt(**kw)
        seen = []
        real = bt._options_half_spread
        bt._options_half_spread = lambda iv, dte: (seen.append(iv), real(iv, dte))[1]
        return bt, seen

    @pytest.mark.parametrize("model", ["legacy", "realistic"])
    def test_a_position_without_entry_iv_uses_the_fallback_at_every_exit_site(self, model):
        d = date(2024, 3, 4)
        row = pd.Series({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0})
        for site in ("check_exit", "force_close", "intraday"):
            bt, seen = self._spy_bt(engine_model=model, settings=_moved("backtest.engine_tuning", fallback_iv=0.41))
            pos = _pos()
            del pos["entry_iv"]
            if site == "check_exit":
                bt._check_exit(dict(pos), row, d, _frame().iloc[:40])
            elif site == "force_close":
                bt._force_close(dict(pos), row, d, _frame().iloc[:40])
            else:
                bt._check_intraday_exit(dict(pos), _session([100.0]), d)
            assert seen and set(seen) == {0.41}, (model, site, seen)
            bt2, seen2 = self._spy_bt(engine_model=model)
            bt2._force_close(dict(pos), row, d, _frame().iloc[:40])
            assert set(seen2) == {0.25}                                              # the default is unchanged
            bt3, seen3 = self._spy_bt(engine_model=model)
            bt3._force_close(dict(_pos()), row, d, _frame().iloc[:40])
            assert set(seen3) == {0.20}                                              # a position's own entry_iv still wins


class TestMetaLabelDecisionRecord:
    def test_the_record_reports_the_labelers_cut(self):
        from ait.backtesting.engine import Backtester
        from tests.test_w90c_engine_and_meta_label import _CapturingMeta, synth_df, synth_features  # noqa: F401
        bt = _bt()
        assert bt._meta_threshold() == 0.50                                         # no labeler: meta_label.min_probability
        bt2 = _bt(settings=_moved("meta_label", min_probability=0.64))
        assert bt2._meta_threshold() == 0.64
        bt2._meta_labeler = SimpleNamespace(min_probability=0.71)
        assert bt2._meta_threshold() == 0.71                                         # the labeler's own cut wins
        assert Backtester._meta_min_probability == MetaLabelConfig().min_probability

    def test_a_whole_run_records_it_in_the_entry_decision(self, monkeypatch):
        from ait.ml.meta_label import MetaLabeler
        from tests.test_r20_research_validity import _engine_kwargs, synth_df, synth_features  # noqa: F401
        from ait.backtesting.engine import Backtester

        class Gate(MetaLabeler):
            def __init__(self):
                super().__init__(min_probability=0.66)
                self._trained = True

            def predict(self, ctx):
                return SimpleNamespace(take_trade=True, probability=0.9, features_used=1)

        df = _frame()
        from ait.ml.features import FeatureEngine
        res = Backtester(**_engine_kwargs(df, FeatureEngine().compute(df)), meta_labeler=Gate()).run()
        decisions = [t.get("decision", {}).get("meta_label") for t in res.trades if t.get("decision")]
        assert decisions and all(m["threshold"] == 0.66 for m in decisions), decisions[:2]


class TestWalkForwardGuards:
    def _wf(self, monkeypatch, **tuning):
        _freeze(monkeypatch, _moved("backtest.walkforward_tuning", **tuning))
        return WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))

    def test_too_few_windows_follows_min_windows(self, monkeypatch):
        from structlog.testing import capture_logs
        data = {"SPY": _frame(50)}
        windows = [(date(2024, 1, 2), date(2024, 2, 1), date(2024, 2, 6), date(2024, 3, 1))] * 2
        wf = self._wf(monkeypatch)
        with capture_logs() as ev:
            assert wf._warn_if_too_few_windows(windows, data) is True               # 2 < the default minimum of 3
        assert any(e["event"] == "walk_forward_too_few_windows" and e["minimum"] == 3 for e in ev)
        wf4 = self._wf(monkeypatch, min_windows=4)
        with capture_logs() as ev4:
            assert wf4._warn_if_too_few_windows(windows + windows[:1], data) is True        # 3 windows < 4
        assert any(e["event"] == "walk_forward_too_few_windows" and e["minimum"] == 4 for e in ev4)
        wf2 = self._wf(monkeypatch, min_windows=2)
        with capture_logs() as ev2:
            assert wf2._warn_if_too_few_windows(windows, data) is False
        assert not ev2

    def test_run_backtest_flags_the_same_minimum(self):
        import run_backtest as rb
        t = WalkForwardTuningConfig(min_windows=5)
        assert rb.too_few_windows(4, t) is True and rb.too_few_windows(5, t) is False
        assert rb.too_few_windows(2, WalkForwardTuningConfig()) is True and rb.too_few_windows(3, WalkForwardTuningConfig()) is False

    def test_the_window_score_shrinks_by_the_full_credit_trade_count(self, monkeypatch):
        br = SimpleNamespace(win_rate=0.8, total_trades=2)
        assert self._wf(monkeypatch)._window_score(br) == pytest.approx(0.8 * (2 / 5) ** 0.5)
        assert self._wf(monkeypatch, window_score_full_credit_trades=10)._window_score(br) == pytest.approx(0.8 * (2 / 10) ** 0.5)
        assert self._wf(monkeypatch)._window_score(SimpleNamespace(win_rate=0.8, total_trades=50)) == pytest.approx(0.8)

    def test_the_oos_minimums(self, monkeypatch):
        from tests.test_w90b_walkforward_settings import _direction_inputs
        pred, df, feats = _direction_inputs(40)
        base = self._wf(monkeypatch)
        # 40 rows, horizon 5: 35 labelled rows >= 10 -> evaluated; a 36-row floor refuses it
        assert base._evaluate_direction_model_oos(pred, df, "SPY", horizon_days=5, precomputed_features=feats)
        assert self._wf(monkeypatch, oos_min_labelled_rows=36, oos_min_extra_rows=0)._evaluate_direction_model_oos(
            pred, df, "SPY", horizon_days=5, precomputed_features=feats) == {}
        rng = np.random.default_rng(3)
        d2 = df.assign(Close=100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(df)))))
        assert base._evaluate_range_model_oos(pred, d2, "SPY", threshold_pct=0.03, horizon_days=5, precomputed_features=feats)
        assert self._wf(monkeypatch, oos_range_min_extra_rows=40)._evaluate_range_model_oos(
            pred, d2, "SPY", threshold_pct=0.03, horizon_days=5, precomputed_features=feats) == {}

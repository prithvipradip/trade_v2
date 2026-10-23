"""W95: eight history / row-count numbers that the census's "unit-like" ignore list hid are settings now.

Research (``config_research.yaml``): ``backtest.engine_tuning.{drawdown_high_lookback_rows, replay_lookback_days}`` (60 / 60) and
``backtest.walkforward_tuning.{oos_feature_context_bars, meta_label_min_window_rows, min_fetched_rows, min_backtest_rows}``
(252 / 60 / 100 / 60). Live (``config.yaml`` ``regime:``): ``min_history_rows`` / ``iv_rank_min_rows`` / ``iv_rank_window_rows``
(60 / 60 / 252). Defaults equal the old literals; every setting is shown to MOVE the behaviour it names, at the real code path.
The ninth number, the Sharpe annualization cap, stays a named constant.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import math
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import ait.backtesting.walkforward as wf_mod
from ait.backtesting.engine import Backtester
from ait.backtesting.walkforward import (
    WalkForwardBacktester,
    WalkForwardConfig,
    check_oos_context_covers_features,
    fetched_rows_sufficient,
    meta_label_window_long_enough,
    oos_context_bars,
)
from ait.config.settings import (
    EngineTuningConfig,
    RegimeConfig,
    WalkForwardTuningConfig,
    load_settings,
    regime_trend_history_need,
)
from ait.ml.regime import MarketRegime, RegimeDetector
from tests.test_r20_research_validity import (  # noqa: F401  (fixtures)
    _engine_kwargs,
    _recorder_factory,
    _run_window,
    synth_df,
    synth_features,
)

ROOT = Path(__file__).resolve().parents[1]
ENGINE_KEYS = {"drawdown_high_lookback_rows": 60, "replay_lookback_days": 60}
WALKFORWARD_KEYS = {"oos_feature_context_bars": 252, "meta_label_min_window_rows": 60, "min_fetched_rows": 100,
                    "min_backtest_rows": 60}
REGIME_KEYS = {"min_history_rows": 60, "iv_rank_min_rows": 60, "iv_rank_window_rows": 252}


def _freeze(monkeypatch, *, eng=None, wt=None, regime=None, lookback=None):
    """Freeze the loaded settings with ``engine_tuning`` / ``walkforward_tuning`` / ``regime`` fields moved and,
    optionally, the feature spec's IV-rank look-back (``ml.features.vol_regime.iv_rank_lookback``)."""
    import ait.config.settings as st
    s = load_settings()
    bt = s.backtest.model_copy(update={
        "engine_tuning": s.backtest.engine_tuning.model_copy(update=eng or {}),
        "walkforward_tuning": s.backtest.walkforward_tuning.model_copy(update=wt or {})})
    upd = {"backtest": bt, "regime": s.regime.model_copy(update=regime or {})}
    if lookback is not None:
        vr = s.ml.features.vol_regime.model_copy(update={"iv_rank_lookback": lookback})
        upd["ml"] = s.ml.model_copy(update={"features": s.ml.features.model_copy(update={"vol_regime": vr})})
        # W97: the context check reads FeatureSpec.longest_lookback() (the longest of ~10 windows, 252 at the defaults), so
        # a small test context needs that to be small too; the real definition is tested in test_w97_feature_windows.py
        from ait.ml.feature_spec import FeatureSpec
        monkeypatch.setattr(FeatureSpec, "longest_lookback", lambda self: lookback)
    moved = s.model_copy(update=upd)
    monkeypatch.setattr(st, "_FROZEN", moved)
    return moved


def _closes(n, seed=7, vol_end=None):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0004, 0.008, n)
    if vol_end:
        r[-vol_end:] *= 4.0                       # the last bars are much more volatile: a high percentile
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2023-01-02", periods=n))


def _ohlcv(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close,
                         "Volume": 5e6}, index=close.index)


# --------------------------------------------------------------------------------------------------------------
# schema and files
# --------------------------------------------------------------------------------------------------------------

class TestSchemaAndFiles:
    def test_defaults_equal_the_old_literals_in_schema_and_files(self):
        s = load_settings()
        for name, old in ENGINE_KEYS.items():
            assert getattr(EngineTuningConfig(), name) == old == getattr(s.backtest.engine_tuning, name), name
        for name, old in WALKFORWARD_KEYS.items():
            assert getattr(WalkForwardTuningConfig(), name) == old == getattr(s.backtest.walkforward_tuning, name), name
        for name, old in REGIME_KEYS.items():
            assert getattr(RegimeConfig(), name) == old == getattr(s.regime, name), name

    def test_research_keys_are_in_the_overlay_and_regime_keys_in_config_yaml_only(self):
        live = yaml.safe_load((ROOT / "config.yaml").read_text())
        research = yaml.safe_load((ROOT / "config_research.yaml").read_text())
        for block, keys in (("engine_tuning", ENGINE_KEYS), ("walkforward_tuning", WALKFORWARD_KEYS)):
            assert set(keys) <= set(research["backtest"][block]), block
            assert not set(keys) & set((live["backtest"].get(block) or {})), f"{block}: research-only keys leaked into config.yaml"
        assert set(REGIME_KEYS) <= set(live["regime"])
        assert not set(REGIME_KEYS) & set((research.get("regime") or {})), "a shared key must not be copied into the overlay"

    @pytest.mark.parametrize("cls,field,bad", [
        (EngineTuningConfig, "drawdown_high_lookback_rows", 1), (EngineTuningConfig, "replay_lookback_days", -1),
        (WalkForwardTuningConfig, "oos_feature_context_bars", 19), (WalkForwardTuningConfig, "meta_label_min_window_rows", 0),
        (WalkForwardTuningConfig, "min_fetched_rows", -1), (WalkForwardTuningConfig, "min_backtest_rows", 3),
        (RegimeConfig, "min_history_rows", 20), (RegimeConfig, "iv_rank_min_rows", 1), (RegimeConfig, "iv_rank_window_rows", 1),
    ])
    def test_each_field_is_bounded(self, cls, field, bad):
        with pytest.raises(Exception):
            cls(**{field: bad})

    def test_every_new_field_is_explained_in_the_schema(self):
        for cls, keys in ((EngineTuningConfig, ENGINE_KEYS), (WalkForwardTuningConfig, WALKFORWARD_KEYS), (RegimeConfig, REGIME_KEYS)):
            for name in keys:
                assert "W95" in (cls.model_fields[name].description or ""), name


# --------------------------------------------------------------------------------------------------------------
# the live regime classifier
# --------------------------------------------------------------------------------------------------------------

class TestRegime:
    def test_the_trend_history_need_is_55_at_the_defaults(self):
        assert regime_trend_history_need(RegimeConfig()) == 55

    def test_min_history_below_the_trend_need_is_refused(self):
        with pytest.raises(ValueError, match="trend measure needs"):
            RegimeConfig(min_history_rows=54)
        assert RegimeConfig(min_history_rows=55).min_history_rows == 55
        with pytest.raises(ValueError):                                    # a longer momentum raises the need to 61
            RegimeConfig(trend_momentum_days=60)
        assert RegimeConfig(trend_momentum_days=60, min_history_rows=61).trend_momentum_days == 60
        assert RegimeConfig(trend_sma_periods=[10, 20], trend_deviation_periods=[20], min_history_rows=30).min_history_rows == 30
        with pytest.raises(ValueError):                                    # a 100-day deviation average needs 100 bars
            RegimeConfig(trend_sma_periods=[10], trend_deviation_periods=[100])
        assert regime_trend_history_need(RegimeConfig(trend_sma_periods=[10], trend_deviation_periods=[100], min_history_rows=100)) == 100

    def test_the_minimum_history_is_the_setting(self):
        close = _closes(80)
        df = _ohlcv(close)
        det = RegimeDetector(RegimeConfig())
        assert det.analyze(df.iloc[:59]).confidence == 0.0 and det.analyze(df.iloc[:59]).regime == MarketRegime.RANGE_BOUND
        assert det.analyze(df.iloc[:60]).confidence > 0.0                   # exactly 60 rows: it makes a call
        late = RegimeDetector(RegimeConfig(min_history_rows=70))
        assert late.analyze(df.iloc[:69]).confidence == 0.0 and late.analyze(df.iloc[:70]).confidence > 0.0

    def test_the_iv_rank_minimum_counts_vol_values_so_it_needs_80_bars(self):
        det = RegimeDetector(RegimeConfig())
        full = _closes(120, vol_end=10)
        # 79 closes -> 78 returns -> 59 values of the 20-day vol: below the 60 minimum -> neutral 0.5
        assert det._measure_iv_rank(full.iloc[:79]) == 0.5
        assert det._measure_iv_rank(full.iloc[:80]) != 0.5                  # 60 values: computed
        eager = RegimeDetector(RegimeConfig(iv_rank_min_rows=20))
        assert eager._measure_iv_rank(full.iloc[:79]) != 0.5

    def test_the_iv_rank_window_is_the_setting(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0.0004, 0.008, 400)
        r[100:160] *= 5.0                         # a volatility spike long ago: inside the 252 window, outside the 60 one
        close = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2023-01-02", periods=400))
        wide = RegimeDetector(RegimeConfig())._measure_iv_rank(close)
        short = RegimeDetector(RegimeConfig(iv_rank_window_rows=60))._measure_iv_rank(close)
        assert wide != short
        # the definition, written out: percentile of the last vol value inside the last `window` values
        lr = np.log(close / close.shift(1)).dropna()
        vol = (lr.rolling(20).std() * np.sqrt(252)).dropna()
        for window, got in ((252, wide), (60, short)):
            tail = vol.tail(window)
            assert got == pytest.approx((vol.iloc[-1] - tail.min()) / (tail.max() - tail.min()), abs=1e-12)

    def test_the_detector_reads_the_config_it_was_given(self):
        src = inspect.getsource(RegimeDetector)
        for name in ("min_history_rows", "iv_rank_min_rows", "iv_rank_window_rows"):
            assert f"cfg.{name}" in src or f"c.{name}" in src, name


# --------------------------------------------------------------------------------------------------------------
# engine: the drawdown gate's rolling high
# --------------------------------------------------------------------------------------------------------------

class TestDrawdownLookback:
    def _vetoes(self, synth_df, synth_features, settings=None):
        extra = {"settings": settings} if settings is not None else {}
        kw = _engine_kwargs(synth_df, synth_features, pct_from_60d_high_threshold=-0.01)
        return Backtester(**kw, **extra).run().entry_funnel.get("pct_from_high", 0)

    def test_the_gate_vetoes_with_the_default_window(self, synth_df, synth_features):
        assert self._vetoes(synth_df, synth_features) > 0

    def test_an_explicit_60_is_the_default_and_a_short_window_vetoes_less(self, synth_df, synth_features, monkeypatch):
        base = self._vetoes(synth_df, synth_features)
        same = self._vetoes(synth_df, synth_features, _freeze(monkeypatch, eng={"drawdown_high_lookback_rows": 60}))
        short = self._vetoes(synth_df, synth_features, _freeze(monkeypatch, eng={"drawdown_high_lookback_rows": 3}))
        assert same == base
        assert short < base                                             # a 3-row high is never far above the last close

    def test_the_window_is_capped_by_the_history_available(self, synth_df, synth_features, monkeypatch):
        huge = self._vetoes(synth_df, synth_features, _freeze(monkeypatch, eng={"drawdown_high_lookback_rows": 1000}))
        assert huge > 0


# --------------------------------------------------------------------------------------------------------------
# replay: the history pad
# --------------------------------------------------------------------------------------------------------------

class _Stop(Exception):
    pass


class TestReplayLookback:
    def _window_rows(self, monkeypatch, days):
        import ait.backtesting.engine as engine_mod
        from ait.backtesting.replay import engine_exit_scan
        _freeze(monkeypatch, eng={"replay_lookback_days": days})
        idx = pd.bdate_range("2024-01-02", periods=200)
        frame = _ohlcv(pd.Series(100.0, index=idx))
        entry = idx[150]
        seen = {}

        class _Rec:
            def __init__(self, **kw):
                seen["rows"] = len(kw["data"])
                raise _Stop

        monkeypatch.setattr(engine_mod, "Backtester", _Rec)
        trade = SimpleNamespace(entry_date=entry.date(), front_expiry=idx[160].date(), entry_dte=10,
                                condor_roles=lambda: {"short_call": 1}, quantity=1, actual_credit=1.0)
        with pytest.raises(_Stop):
            engine_exit_scan(trade, SimpleNamespace(daily_frame=lambda: frame), {"engine": {"iv": 0.2}})
        return seen["rows"]

    def test_the_pad_is_the_setting(self, monkeypatch):
        idx = pd.bdate_range("2024-01-02", periods=200)
        entry, expiry = idx[150], idx[160]

        def expected(days):
            return int(((idx >= entry - pd.Timedelta(days=days)) & (idx <= expiry)).sum())

        for days in (5, 60, 120):
            assert self._window_rows(monkeypatch, days) == expected(days), days
        assert expected(120) > expected(60) > expected(5)


# --------------------------------------------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------------------------------------------

class TestWalkForwardHelpers:
    def test_the_context_is_capped_at_the_training_length(self):
        assert oos_context_bars(252, 100) == 100 and oos_context_bars(252, 400) == 252 and oos_context_bars(300, 400) == 300

    def test_the_meta_label_guard_is_strictly_more_than(self):
        assert not meta_label_window_long_enough(60, 60) and meta_label_window_long_enough(61, 60)
        assert not meta_label_window_long_enough(10, 60)

    def test_the_fetched_rows_guard_is_strictly_more_than(self):
        assert not fetched_rows_sufficient(100, 100) and fetched_rows_sufficient(101, 100)

    def test_a_context_below_the_feature_lookback_is_refused(self):
        check_oos_context_covers_features(252, 252)
        check_oos_context_covers_features(300, 252)
        with pytest.raises(ValueError, match="iv_rank_lookback"):
            check_oos_context_covers_features(251, 252)


class TestWalkForwardWiring:
    def test_the_constructor_refuses_a_short_context(self, monkeypatch):
        _freeze(monkeypatch, wt={"oos_feature_context_bars": 100})
        with pytest.raises(ValueError, match="oos_feature_context_bars"):
            WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))
        _freeze(monkeypatch, wt={"oos_feature_context_bars": 100}, lookback=100)        # the spec's look-back moved with it
        WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))

    def _oos_rows(self, monkeypatch, synth_df, context):
        _freeze(monkeypatch, wt={"oos_feature_context_bars": context}, lookback=30)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink
        return len(sink[0]["data"])

    def test_the_test_window_gets_exactly_the_configured_context(self, monkeypatch, synth_df):
        a, b = self._oos_rows(monkeypatch, synth_df, 40), self._oos_rows(monkeypatch, synth_df, 70)
        assert b - a == 30
        assert self._oos_rows(monkeypatch, synth_df, 5000) - a > 30            # capped by the training window, not by 5000

    def test_the_meta_labeler_minimum_is_the_setting(self, monkeypatch, synth_df):
        train = synth_df.iloc[:100]
        reached = []

        class _Bt:
            def __init__(self, **kw):
                reached.append(1)
                raise _Stop

        monkeypatch.setattr(wf_mod, "Backtester", _Bt)
        for minimum, expect_reached in ((100, False), (99, True)):
            reached.clear()
            _freeze(monkeypatch, wt={"meta_label_min_window_rows": minimum})
            wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))
            try:
                out = wf._train_window_meta_labeler(train_df=train, symbol="SPY", window_id=1, predictor=None,
                                                    window_cfg=wf._config, artifact_dir=None, vix_ctx=None)
            except _Stop:
                out = "constructed"
            assert bool(reached) is expect_reached, minimum
            if not expect_reached:
                assert out is None

    def test_the_fetched_rows_minimum_is_the_setting(self, monkeypatch, synth_df):
        import ait.data.market_data as md
        df = synth_df.iloc[:120]
        monkeypatch.setattr(md, "load_daily_ohlcv", lambda s, days=0, db_path=None, **kw: df)
        for minimum, kept in ((119, True), (120, False)):
            _freeze(monkeypatch, wt={"min_fetched_rows": minimum})
            wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(train_window_models=False))
            assert ("SPY" in asyncio.run(wf._fetch_data())) is kept, minimum


# --------------------------------------------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------------------------------------------

class TestOptimizerMinimum:
    def _opt(self, monkeypatch, n, **wt):
        from ait.optimization.optimizer import StrategyOptimizer
        s = _freeze(monkeypatch, wt=wt)
        opt = StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1, settings=s, val_split=True)
        opt._data = {"SPY": _ohlcv(_closes(n))}
        return opt

    def _eval_start(self, opt):
        seen = {}
        orig = Backtester.__init__

        def spy(self, *a, **k):
            seen["eval_start_date"] = k.get("eval_start_date")
            raise _Stop

        Backtester.__init__ = spy
        try:
            try:
                opt._run_backtest({})
            except _Stop:
                pass
        finally:
            Backtester.__init__ = orig
        return seen.get("eval_start_date", "not constructed")

    def test_fewer_rows_than_the_minimum_raise_insufficient_data(self, monkeypatch):
        with pytest.raises(ValueError, match="Insufficient data"):
            self._opt(monkeypatch, 79, min_backtest_rows=80)._run_backtest({})
        assert self._eval_start(self._opt(monkeypatch, 80, min_backtest_rows=80)) != "not constructed"

    def test_the_validation_split_needs_twice_the_minimum(self, monkeypatch):
        # ratio 0.4 keeps a long validation slice, so only the "twice the minimum" rule can switch the split off
        assert self._eval_start(self._opt(monkeypatch, 150, min_backtest_rows=75, optimize_val_split_ratio=0.4)) is not None   # 150 >= 2 x 75
        assert self._eval_start(self._opt(monkeypatch, 150, min_backtest_rows=76, optimize_val_split_ratio=0.4)) is None        # 150 < 2 x 76

    def test_the_training_part_is_never_shorter_than_the_minimum(self, monkeypatch):
        idx = pd.bdate_range("2023-01-02", periods=200)
        # ratio 0.7 would put the cut at row 60; the minimum moves it to row 80
        assert self._eval_start(self._opt(monkeypatch, 200, min_backtest_rows=80, optimize_val_split_ratio=0.7)) == idx[80].date()
        assert self._eval_start(self._opt(monkeypatch, 200, min_backtest_rows=50, optimize_val_split_ratio=0.7)) == idx[60].date()

    def test_a_validation_slice_shorter_than_half_the_minimum_is_dropped(self, monkeypatch):
        # 150 bars, ratio 0.2: cut at 120, a 30-bar slice. Kept while 30 >= minimum // 2, dropped beyond
        assert self._eval_start(self._opt(monkeypatch, 150, min_backtest_rows=61)) is not None      # 61 // 2 = 30
        assert self._eval_start(self._opt(monkeypatch, 150, min_backtest_rows=62)) is None          # 62 // 2 = 31

    def test_the_fetch_filter_uses_the_setting(self, monkeypatch):
        big_df = _ohlcv(_closes(300))
        monkeypatch.setattr("ait.data.market_data.load_daily_ohlcv", lambda symbol, days, db_path=None, **kw: big_df)
        for minimum, kept in ((120, True), (121, False)):
            opt = self._opt(monkeypatch, 300, min_backtest_rows=minimum)
            opt._train_days = 120
            assert ("SPY" in opt._fetch_data()) is kept, minimum

    def test_the_module_constant_is_gone(self):
        import ait.optimization.optimizer as om
        assert not hasattr(om, "_MIN_BACKTEST_ROWS")


class TestBelowFloorScore:
    """The -100 base of the optimizer's graded below-floor ramp is ``backtest.optimize_below_floor_score``."""

    def _score(self, opt, n, monkeypatch):
        from tests.test_w67_regime_gate import _FakeResult, _FakeTrial
        monkeypatch.setattr(opt, "_suggest_params", lambda trial: {})
        monkeypatch.setattr(opt, "_run_backtest", lambda params: _FakeResult(n))
        return opt._objective_fn(_FakeTrial())

    def test_default_and_files(self):
        from ait.config.settings import BacktestConfig
        assert BacktestConfig().optimize_below_floor_score == -100.0 == load_settings().backtest.optimize_below_floor_score
        research = yaml.safe_load((ROOT / "config_research.yaml").read_text())
        live = yaml.safe_load((ROOT / "config.yaml").read_text())
        assert research["backtest"]["optimize_below_floor_score"] == -100.0
        assert "optimize_below_floor_score" not in live["backtest"]

    def test_bounds_and_the_ramp_must_stay_negative(self):
        from ait.config.settings import BacktestConfig
        for bad in (0.0, 5.0, -1001.0):
            with pytest.raises(Exception):
                BacktestConfig(optimize_below_floor_score=bad)
        with pytest.raises(ValueError, match="must stay below 0"):
            BacktestConfig(optimize_below_floor_score=-10.0, optimize_below_floor_gradient=20.0)
        with pytest.raises(ValueError, match="must stay below 0"):                  # exactly 0 is refused too
            BacktestConfig(optimize_below_floor_score=-20.0, optimize_below_floor_gradient=20.0)
        assert BacktestConfig(optimize_below_floor_score=-30.0, optimize_below_floor_gradient=20.0)

    def test_the_optimizer_scores_a_starved_trial_from_the_setting(self, monkeypatch):
        from tests.test_w67_regime_gate import _optimizer
        assert self._score(_optimizer(objective="sharpe_ratio"), 0, monkeypatch) == -100.0
        moved = _optimizer(objective="sharpe_ratio", below_floor_score=-250.0)
        assert self._score(moved, 0, monkeypatch) == -250.0
        assert self._score(moved, 2, monkeypatch) == pytest.approx(-250.0 + 20.0 * 2 / 3)

    def test_the_setting_reaches_the_optimizer_from_the_loaded_config(self, monkeypatch):
        from tests.test_w67_regime_gate import _optimizer
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(update={"optimize_below_floor_score": -300.0})})
        assert _optimizer(objective="sharpe_ratio", settings=moved)._below_floor_score == -300.0

    def test_the_walk_forward_hands_it_to_the_optimizer(self, monkeypatch, synth_df):
        import ait.optimization.optimizer as optimizer_mod
        captured: dict = {}

        class _RecordingOptimizer:
            def __init__(self, **kwargs) -> None:
                captured.update(kwargs)

            def run(self, data, prior_params=None):
                return SimpleNamespace(best_params={}, best_value=0.0, study=None)

        monkeypatch.setattr(optimizer_mod, "StrategyOptimizer", _RecordingOptimizer)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(
            train_window_models=False, optimize_below_floor_score=-300.0))
        wf._optimize_window_params(synth_df, "SPY", 1, ["iron_condor"])
        assert captured["below_floor_score"] == -300.0
        assert captured["below_floor_gradient"] == 20.0

    def test_the_walk_forward_passes_it_down(self, monkeypatch):
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(update={"optimize_below_floor_score": -300.0})})
        import ait.config.settings as st
        monkeypatch.setattr(st, "_FROZEN", moved)
        assert WalkForwardConfig(train_window_models=False).optimize_below_floor_score == -300.0
        assert WalkForwardConfig(train_window_models=False, optimize_below_floor_score=-50.0).optimize_below_floor_score == -50.0


# --------------------------------------------------------------------------------------------------------------
# the ninth number: a named constant, and no bare converted number is left behind
# --------------------------------------------------------------------------------------------------------------

class TestSharpeCapAndGuards:
    def test_the_sharpe_cap_is_a_named_constant(self):
        from ait.backtesting import result
        assert result.SHARPE_ANNUALIZATION_CAP == 252.0
        daily = [{"entry_date": f"2024-01-{d:02d}"} for d in range(1, 29)]          # 28 trades in 27 days: capped at daily
        assert result.annualization_factor(daily) == pytest.approx(math.sqrt(252.0))
        sparse = [{"entry_date": "2024-01-01"}, {"entry_date": "2025-01-01"}]
        assert result.annualization_factor(sparse) == pytest.approx(math.sqrt(2 * 365.0 / 366.0), rel=1e-6)

    @pytest.mark.parametrize("path,func,forbidden", [
        ("src/ait/ml/regime.py", "RegimeDetector.analyze", {60}),
        ("src/ait/ml/regime.py", "RegimeDetector._measure_iv_rank", {60}),
        ("src/ait/optimization/optimizer.py", "StrategyOptimizer._run_backtest", {60}),
        ("src/ait/backtesting/walkforward.py", "WalkForwardBacktester._train_window_meta_labeler", {60}),
        ("src/ait/backtesting/walkforward.py", "WalkForwardBacktester._fetch_data", {100}),
    ])
    def test_no_bare_converted_number_is_left_at_the_site(self, path, func, forbidden):
        tree = ast.parse((ROOT / path).read_text())
        cls_name, _, fn_name = func.partition(".")
        node = next(f for c in ast.walk(tree) if isinstance(c, ast.ClassDef) and c.name == cls_name
                    for f in c.body if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == fn_name)
        found = {n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
                 and not isinstance(n.value, bool)}
        assert not (found & forbidden), f"{func} still carries {sorted(found & forbidden)}"

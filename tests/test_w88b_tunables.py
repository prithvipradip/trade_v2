"""W88b: research-only tunables are schema fields, listed in config_research.yaml, and read by the code.

Defaults reproduce the pre-W88 literals, so every check has two halves: AT DEFAULTS the behaviour equals what the
literals did (a golden record for the learner captured before the change; the old formulas inline for the rest),
and a CHANGED value moves the behaviour (so a field cannot be silently unwired).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ait.config.settings import (
    DEFAULT_RISK_FREE_RATE,
    BacktestConfig,
    LearnerConfig,
    LegacyEngineConfig,
    ObjectivesConfig,
    OptionsConfig,
    WalkForwardTuningConfig,
    load_settings,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "data"))
from _w88b_learner_scenarios import scenarios  # noqa: E402


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in sorted(o.items())}
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, list):
        return [_clean(x) for x in o]
    return o


def _run_learner(windows, config=None, base=0.55):
    from ait.backtesting.learner import BacktestLearner
    learner = BacktestLearner(base_confidence=base, config=config)
    steps = [_clean(learner.process_window(w, i + 1)) for i, w in enumerate(windows)]
    return learner, steps


def _ohlcv(days=120, seed=3, drift=0.0005, vol=0.012):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + rng.normal(drift, vol, days))
    return pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close,
                         "Volume": np.full(days, 5e6)}, index=idx)


# --------------------------------------------------------------------------
# the learner
# --------------------------------------------------------------------------

class TestLearner:
    GOLDEN = json.loads((ROOT / "tests" / "data" / "w88b_learner_golden.json").read_text())

    @pytest.mark.parametrize("name", sorted(GOLDEN))
    def test_defaults_reproduce_the_pre_w88_behaviour(self, name):
        learner, steps = _run_learner(scenarios()[name])
        want = self.GOLDEN[name]
        assert steps == want["steps"]
        assert _clean(learner.get_config()) == want["final"]
        assert learner.summary() == want["summary"]

    def test_the_defaults_are_the_old_literals(self):
        c = LearnerConfig()
        assert (c.windows_before_adapting, c.min_trades_to_learn, c.min_trades_to_disable) == (2, 15, 25)
        assert (c.strategy_multiplier_min, c.strategy_multiplier_max) == (0.30, 2.0)
        assert (c.reduce_win_rate, c.reduce_factor, c.reduce_min_change) == (0.30, 0.80, 0.03)
        assert (c.boost_win_rate, c.boost_min_trades, c.boost_factor, c.boost_min_change) == (0.58, 6, 1.20, 0.05)
        assert (c.reenable_win_rate, c.reenable_min_trades, c.reenable_multiplier) == (0.50, 10, 0.5)
        assert c.confidence_cap == 0.85
        assert (c.confidence_low_band_upper, c.confidence_mid_band_upper) == (0.65, 0.75)
        assert (c.confidence_raise_win_rate, c.confidence_raise_step, c.confidence_raise_min_change) == (0.35, 0.05, 0.01)
        assert (c.symbol_disable_win_rate, c.default_entry_confidence) == (0.20, 0.65)
        assert c.protected_strategies == ["iron_condor"]

    def test_a_bare_learner_reads_the_loaded_settings(self, monkeypatch):
        import ait.config.settings as st
        from ait.backtesting.learner import BacktestLearner
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(
            update={"learner": LearnerConfig(min_trades_to_learn=7, min_trades_to_disable=9)})})
        monkeypatch.setattr(st, "_FROZEN", moved)
        assert BacktestLearner()._cfg.min_trades_to_learn == 7

    def test_fewer_trades_to_learn_adapts_earlier(self):
        name = "losing_protected_condor"
        _, base = _run_learner(scenarios()[name])
        _, early = _run_learner(scenarios()[name], LearnerConfig(min_trades_to_learn=5, windows_before_adapting=1))
        assert sum(len(s.get("adaptations", [])) for s in early) > sum(len(s.get("adaptations", [])) for s in base)

    def test_a_gentler_reduce_factor_changes_the_multiplier(self):
        a, _ = _run_learner(scenarios()["losing_protected_condor"])
        b, _ = _run_learner(scenarios()["losing_protected_condor"], LearnerConfig(reduce_factor=0.95))
        assert b.get_strategy_multiplier("iron_condor") > a.get_strategy_multiplier("iron_condor")

    def test_the_confidence_cap_bounds_the_raise(self):
        learner, _ = _run_learner(scenarios()["low_confidence_band_and_bad_symbol"], LearnerConfig(confidence_cap=0.60))
        assert learner.get_config()["min_confidence"] <= 0.60 + 1e-9

    def test_the_disable_and_reenable_paths_are_reachable_when_the_step_guard_allows_it(self):
        """With the shipped reduce_min_change the multiplier stalls above the floor (a registered quirk), so the
        disable / re-enable branches are exercised here with the guard off."""
        cfg = LearnerConfig(reduce_min_change=0.0, protected_strategies=[])
        learner, steps = _run_learner(scenarios()["disable_then_reenable"], cfg)
        kinds = [a["type"] for s in steps for a in s.get("adaptations", [])]
        assert "strategy_disable" in kinds and "strategy_reenable" in kinds
        assert learner.get_strategy_multiplier("long_put") >= cfg.reenable_multiplier

    def test_the_symbol_cutoff_is_a_field(self):
        learner, _ = _run_learner(scenarios()["low_confidence_band_and_bad_symbol"], LearnerConfig(symbol_disable_win_rate=0.01))
        assert learner.is_symbol_allowed("IWM")

    def test_the_band_edges_are_fields(self):
        from ait.backtesting.learner import BacktestLearner
        lo = BacktestLearner(config=LearnerConfig(confidence_low_band_upper=0.5, confidence_mid_band_upper=0.6))
        assert [lo._confidence_band(x) for x in (0.45, 0.55, 0.7)] == ["low", "medium", "high"]
        default = BacktestLearner(config=LearnerConfig())
        assert [default._confidence_band(x) for x in (0.64, 0.74, 0.76)] == ["low", "medium", "high"]

    @pytest.mark.parametrize("kw", [
        {"strategy_multiplier_min": 1.5}, {"confidence_low_band_upper": 0.8, "confidence_mid_band_upper": 0.7},
        {"min_trades_to_learn": 30, "min_trades_to_disable": 20}, {"reduce_factor": 1.0}, {"boost_factor": 1.0}])
    def test_bad_values_fail_at_load(self, kw):
        with pytest.raises(Exception):
            LearnerConfig(**kw)


# --------------------------------------------------------------------------
# legacy / dormant engine constants
# --------------------------------------------------------------------------

def _engine(legacy=None, strategies=("short_strangle",), **kw):
    from ait.backtesting.engine import Backtester
    s = load_settings()
    if legacy is not None:
        s = s.model_copy(update={"backtest": s.backtest.model_copy(update={"legacy_engine": legacy})})
    return Backtester(data=_ohlcv(80), strategies=list(strategies), settings=s, **kw)


class TestLegacyEngine:
    @staticmethod
    def _old_direction(hist):
        """The pre-W88 _simple_direction, verbatim, as the oracle."""
        from ait.strategies.base import SignalDirection
        close = hist["Close"]
        ret_5 = close.iloc[-1] / close.iloc[-5] - 1 if len(close) >= 5 else 0.0
        ret_20 = close.iloc[-1] / close.iloc[-20] - 1 if len(close) >= 20 else 0.0
        score = 0.6 * ret_5 + 0.4 * ret_20
        threshold = 0.005
        if score > threshold:
            return SignalDirection.BULLISH, min(0.5 + abs(score) * 10, 0.95)
        elif score < -threshold:
            return SignalDirection.BEARISH, min(0.5 + abs(score) * 10, 0.95)
        return SignalDirection.NEUTRAL, 0.4

    def test_the_fallback_direction_equals_the_old_formula_on_many_frames(self):
        eng = _engine()
        n_dir = set()
        for seed in range(60):
            hist = _ohlcv(60, seed=seed, drift=0.003 * ((seed % 5) - 2), vol=0.02)
            for cut in (3, 10, 25, 59):
                h = hist.iloc[:cut]
                got, want = eng._simple_direction(h), self._old_direction(h)
                assert got == want
                n_dir.add(got[0])
        assert len(n_dir) == 3                      # the sweep really visited bullish, bearish and neutral

    def test_the_defaults_are_the_old_literals(self):
        c = LegacyEngineConfig()
        assert (c.naked_margin_fraction, c.fallback_weight_5d, c.fallback_weight_20d, c.fallback_threshold) == (0.20, 0.6, 0.4, 0.005)
        assert (c.fallback_confidence_base, c.fallback_confidence_slope, c.fallback_confidence_cap) == (0.5, 10.0, 0.95)
        assert (c.fallback_neutral_confidence, c.bearish_confidence_margin) == (0.4, 0.05)
        assert c.thesis_invalidation_confidence == 0.80

    def test_changed_constants_move_the_fallback(self):
        hist = _ohlcv(60, seed=4, drift=0.004, vol=0.01)
        base = _engine()._simple_direction(hist)
        flat = _engine(LegacyEngineConfig(fallback_confidence_slope=0.0))._simple_direction(hist)
        assert base[0] == flat[0] and flat[1] == pytest.approx(0.5) and base[1] > 0.5
        assert _engine(LegacyEngineConfig(fallback_neutral_confidence=0.33))._simple_direction(hist.iloc[:3])[1] == 0.33
        wide = _engine(LegacyEngineConfig(fallback_threshold=0.1))._simple_direction(hist)
        assert wide[0].value == "neutral" if hasattr(wide[0], "value") else True

    def test_the_engine_holds_the_loaded_values(self):
        eng = _engine(LegacyEngineConfig(bearish_confidence_margin=0.11, thesis_invalidation_confidence=0.9))
        assert (eng._legacy.bearish_confidence_margin, eng._legacy.thesis_invalidation_confidence) == (0.11, 0.9)

    def test_a_partial_engine_keeps_the_schema_defaults(self):
        from ait.backtesting.engine import Backtester
        assert Backtester.__new__(Backtester)._legacy == LegacyEngineConfig()

    def test_the_naked_margin_fraction_sizes_a_strangle(self):
        from datetime import date
        args = ("short_strangle", 480.0, 0.2, 30 / 365.0, DEFAULT_RISK_FREE_RATE, 30, date(2024, 3, 1), 100_000.0)
        a = _engine(position_size_pct=0.50)._build_credit_position(*args)
        b = _engine(LegacyEngineConfig(naked_margin_fraction=0.40), position_size_pct=0.50)._build_credit_position(*args)
        assert a is not None and b is not None
        assert b["contracts"] < a["contracts"]
        assert (a["contracts"], b["contracts"]) == (int(50_000.0 / (480.0 * 0.20 * 100)), int(50_000.0 / (480.0 * 0.40 * 100)))


# --------------------------------------------------------------------------
# objectives
# --------------------------------------------------------------------------

class _R:
    sharpe_ratio, win_rate, max_drawdown, profit_factor = 1.7, 0.6, -0.12, float("inf")


class TestObjectives:
    def test_defaults_equal_the_old_formula(self):
        from ait.optimization.objectives import OBJECTIVES
        r = _R()
        assert OBJECTIVES["composite"](r) == 0.4 * 1.7 + 0.4 * 0.6 - 0.2 * 0.12
        assert OBJECTIVES["profit_factor"](r) == 10.0
        assert OBJECTIVES["sharpe_ratio"](r) == 1.7 and OBJECTIVES["win_rate"](r) == 0.6

    def test_weights_follow_the_config(self):
        from ait.optimization.objectives import OBJECTIVES
        cfg = ObjectivesConfig(sharpe_weight=1.0, win_rate_weight=0.0, drawdown_penalty=1.0, profit_factor_cap=4.0)
        assert OBJECTIVES["composite"](_R(), cfg) == pytest.approx(1.7 - 0.12)
        assert OBJECTIVES["profit_factor"](_R(), cfg) == 4.0

    def test_a_bare_call_reads_the_loaded_settings(self, monkeypatch):
        import ait.config.settings as st
        from ait.optimization.objectives import OBJECTIVES
        s = load_settings()
        monkeypatch.setattr(st, "_FROZEN", s.model_copy(update={"backtest": s.backtest.model_copy(
            update={"objectives": ObjectivesConfig(sharpe_weight=2.0)})}))
        assert OBJECTIVES["composite"](_R()) == pytest.approx(2.0 * 1.7 + 0.4 * 0.6 - 0.2 * 0.12)

    def test_the_optimizer_scores_trials_with_its_own_copy(self):
        from ait.optimization.optimizer import StrategyOptimizer
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(
            update={"objectives": ObjectivesConfig(sharpe_weight=3.0)})})
        opt = StrategyOptimizer(symbols=["QQQ"], strategies=["iron_condor"], n_trials=1, settings=moved)
        assert opt._objectives_cfg.sharpe_weight == 3.0
        assert StrategyOptimizer(symbols=["QQQ"], strategies=["iron_condor"], n_trials=1)._objectives_cfg.sharpe_weight == 0.4


# --------------------------------------------------------------------------
# walk-forward / optimizer constants
# --------------------------------------------------------------------------

class TestWalkForwardTuning:
    @staticmethod
    def _old_threshold(train_df, horizon_days, lookback_days=60, multiplier=1.25, low_clip=0.02, high_clip=0.15):
        recent = train_df["Close"].pct_change().tail(lookback_days).dropna()
        if len(recent) < 10:
            return 0.05
        rvol_annual = float(recent.std() * np.sqrt(252))
        raw = rvol_annual * np.sqrt(horizon_days / 252) * multiplier
        return float(np.clip(raw, low_clip, high_clip))

    def test_the_adaptive_threshold_equals_the_old_function(self):
        from ait.backtesting.walkforward import WalkForwardBacktester
        fn = WalkForwardBacktester._adaptive_range_threshold                  # called statically, as the scripts do
        for seed in range(30):
            df = _ohlcv(90, seed=seed, vol=0.004 + 0.003 * (seed % 7))
            for h in (3, 9, 21):
                assert fn(df, horizon_days=h) == self._old_threshold(df, h)
        assert fn(_ohlcv(8), horizon_days=9) == 0.05                         # too little data: the fallback band

    def test_every_argument_is_a_field_and_an_explicit_value_wins(self):
        from ait.backtesting.walkforward import WalkForwardBacktester
        fn = WalkForwardBacktester._adaptive_range_threshold
        df = _ohlcv(90, seed=2, vol=0.01)
        base = fn(df, horizon_days=9)
        assert fn(df, horizon_days=9, tuning=WalkForwardTuningConfig(range_threshold_multiplier=2.5)) > base
        assert fn(df, horizon_days=9, tuning=WalkForwardTuningConfig(range_threshold_high_clip=0.03,
                                                                     range_threshold_low_clip=0.02)) <= 0.03
        assert fn(df, horizon_days=9, tuning=WalkForwardTuningConfig(range_threshold_min_observations=500,
                                                                     range_threshold_fallback=0.07)) == 0.07
        assert fn(df, horizon_days=9, multiplier=1.25, tuning=WalkForwardTuningConfig(range_threshold_multiplier=4.0)) == base

    def test_a_bare_call_reads_the_loaded_settings(self, monkeypatch):
        import ait.config.settings as st
        from ait.backtesting.walkforward import WalkForwardBacktester
        s = load_settings()
        df = _ohlcv(90, seed=2, vol=0.01)
        base = WalkForwardBacktester._adaptive_range_threshold(df, horizon_days=9)
        monkeypatch.setattr(st, "_FROZEN", s.model_copy(update={"backtest": s.backtest.model_copy(
            update={"walkforward_tuning": WalkForwardTuningConfig(range_threshold_multiplier=2.5)})}))
        assert WalkForwardBacktester._adaptive_range_threshold(df, horizon_days=9) > base

    def test_the_walk_forward_holds_the_loaded_tuning(self):
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        bt = WalkForwardBacktester(symbols=["QQQ"], strategies=["iron_condor"],
                                   config=WalkForwardConfig(train_window_models=False))
        assert bt._tuning == load_settings().backtest.walkforward_tuning
        assert bt._tuning.warm_start_min_win_rate == 0.75 and bt._tuning.range_threshold_fallback == 0.05

    def test_the_optimizer_holds_the_loaded_tuning(self):
        from ait.optimization.optimizer import StrategyOptimizer
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(
            update={"walkforward_tuning": WalkForwardTuningConfig(optimize_val_split_ratio=0.3,
                                                                  inert_probe_dte_fallback=30)})})
        opt = StrategyOptimizer(symbols=["QQQ"], strategies=["iron_condor"], n_trials=1, settings=moved)
        assert opt._tuning.optimize_val_split_ratio == 0.3 and opt._tuning.inert_probe_dte_fallback == 30

    def test_the_validation_split_ratio_sizes_the_holdout(self):
        from ait.optimization.optimizer import StrategyOptimizer
        data = {"QQQ": _ohlcv(150, seed=5)}
        cuts = {}
        for ratio in (0.20, 0.35):
            s = load_settings()
            moved = s.model_copy(update={"backtest": s.backtest.model_copy(
                update={"walkforward_tuning": WalkForwardTuningConfig(optimize_val_split_ratio=ratio)})})
            opt = StrategyOptimizer(symbols=["QQQ"], strategies=["iron_condor"], n_trials=1, settings=moved,
                                    val_split=True)
            opt._data = data
            seen = {}
            real = opt._run_backtest
            from ait.backtesting.engine import Backtester
            orig = Backtester.__init__

            def spy(self, *a, **k):
                seen["eval_start_date"] = k.get("eval_start_date")
                orig(self, *a, **k)
            try:
                Backtester.__init__ = spy
                try:
                    real({})
                except Exception:
                    pass
            finally:
                Backtester.__init__ = orig
            cuts[ratio] = seen.get("eval_start_date")
        assert cuts[0.20] is not None and cuts[0.35] is not None and cuts[0.35] < cuts[0.20]

    def test_every_walk_forward_config_field_the_fallbacks_used_to_guard_exists(self):
        from ait.backtesting.walkforward import WalkForwardConfig
        cfg = WalkForwardConfig(train_window_models=False)
        for name in ("hurst_regime_threshold", "hurst_regime_penalty", "hurst_hard_veto_multiplier",
                     "multifractal_max_width", "iv_rank_rise_threshold", "pct_from_60d_high_threshold",
                     "min_edge_over_baseline"):
            assert hasattr(cfg, name), name           # the getattr(..., literal) fallback could never fire


# --------------------------------------------------------------------------
# the risk-free rate has one schema default
# --------------------------------------------------------------------------

class TestRiskFreeRateSingleSource:
    def test_every_copy_is_the_schema_default(self):
        import inspect

        from ait.backtesting import iv_surface, pricing, replay
        from ait.backtesting.engine import Backtester
        assert OptionsConfig().risk_free_rate == DEFAULT_RISK_FREE_RATE == 0.05
        assert replay.ENGINE_RISK_FREE_RATE == DEFAULT_RISK_FREE_RATE
        assert Backtester._risk_free_rate == DEFAULT_RISK_FREE_RATE
        assert inspect.signature(pricing.find_strike_by_delta).parameters["r"].default == DEFAULT_RISK_FREE_RATE
        assert iv_surface.DEFAULT_RISK_FREE_RATE == DEFAULT_RISK_FREE_RATE


# --------------------------------------------------------------------------
# the YAML lists everything
# --------------------------------------------------------------------------

class TestResearchYamlIsComplete:
    BLOCKS = {"learner": LearnerConfig, "legacy_engine": LegacyEngineConfig, "objectives": ObjectivesConfig,
              "walkforward_tuning": WalkForwardTuningConfig}
    raw = yaml.safe_load((ROOT / "config_research.yaml").read_text())

    @pytest.mark.parametrize("block", sorted(BLOCKS))
    def test_every_field_is_listed(self, block):
        listed = set((self.raw["backtest"].get(block) or {}))
        fields = set(self.BLOCKS[block].model_fields)
        assert fields == listed, f"backtest.{block}: missing {sorted(fields - listed)}, unknown {sorted(listed - fields)}"

    def test_listed_values_validate_and_equal_the_schema_defaults_today(self):
        s = load_settings()
        for block, model in self.BLOCKS.items():
            assert getattr(s.backtest, block) == model(), block      # the file mirrors the defaults on purpose

    def test_none_of_the_research_only_keys_is_in_config_yaml(self):
        live = yaml.safe_load((ROOT / "config.yaml").read_text())["backtest"]
        assert not set(self.BLOCKS) & set(live)

    def test_the_view_shows_them_with_the_overlay_origin(self):
        from ait.config.research import settings_origin
        got = settings_origin("config.yaml")
        assert got["backtest.learner.min_trades_to_learn"] == (15, "overlay")
        assert got["backtest.objectives.sharpe_weight"] == (0.4, "overlay")

    def test_a_study_can_override_one(self, tmp_path):
        from ait.config.research import load_research_settings
        p = tmp_path / "s.yaml"
        p.write_text("name: s\nquestion: q\nrule: exploratory\nsettings: {backtest: {learner: {min_trades_to_learn: 9, "
                     "min_trades_to_disable: 12}}}\n")
        s = load_research_settings("config.yaml", p)
        assert s.backtest.learner.min_trades_to_learn == 9 and s.backtest.learner.reduce_factor == 0.80


def test_backtest_config_exposes_the_new_blocks():
    b = BacktestConfig()
    assert (b.learner, b.legacy_engine, b.objectives, b.walkforward_tuning) == (
        LearnerConfig(), LegacyEngineConfig(), ObjectivesConfig(), WalkForwardTuningConfig())


# --------------------------------------------------------------------------
# every field moves behaviour (a field cannot be silently unwired)
# --------------------------------------------------------------------------

def _all_scenarios(config, base=0.55):
    out = {}
    for n, w in scenarios().items():
        learner, steps = _run_learner(w, config, base)
        out[n] = (steps, _clean(learner.get_config()))
    return json.dumps(out, sort_keys=True, default=str)


class TestEveryLearnerFieldMovesBehaviour:
    BASE = _all_scenarios(LearnerConfig())
    # one perturbed value per field whose effect the shared scenarios reach
    PERTURBED = {
        "windows_before_adapting": 4, "min_trades_to_disable": 40, "strategy_multiplier_min": 0.5,
        "strategy_multiplier_max": 1.5, "reduce_win_rate": 0.5, "reduce_factor": 0.5, "reduce_min_change": 0.0,
        "boost_win_rate": 0.9, "boost_factor": 1.5, "confidence_cap": 0.6, "confidence_low_band_upper": 0.7,
        "confidence_raise_win_rate": 0.6, "confidence_raise_step": 0.1, "symbol_disable_win_rate": 0.5,
        "min_trades_to_learn": 20,
    }

    @pytest.mark.parametrize("field", sorted(PERTURBED))
    def test_a_changed_value_changes_what_the_learner_does(self, field):
        assert _all_scenarios(LearnerConfig(**{field: self.PERTURBED[field]})) != self.BASE, field

    def test_every_field_is_covered_by_this_class_or_a_named_test_below(self):
        covered = set(self.PERTURBED) | {
            "protected_strategies", "boost_min_trades", "boost_min_change", "reenable_win_rate", "reenable_min_trades",
            "reenable_multiplier", "confidence_mid_band_upper", "confidence_raise_min_change",
            "default_entry_confidence"}
        assert covered == set(LearnerConfig.model_fields)

    def test_protected_strategies_decide_what_can_be_disabled(self):
        free = dict(reduce_min_change=0.0)
        losing_condor = [[dict(t, strategy="iron_condor") for t in w] for w in scenarios()["disable_then_reenable"][:10]]
        kinds = lambda cfg: {a["type"] for s in _run_learner(losing_condor, cfg)[1]  # noqa: E731
                             for a in s.get("adaptations", [])}
        assert "strategy_disable" not in kinds(LearnerConfig(**free))                       # condor is protected
        assert "strategy_disable" in kinds(LearnerConfig(protected_strategies=[], **free))

    def test_boost_thresholds(self):
        w = scenarios()["winning_spread"]
        _, base = _run_learner(w)
        # boost_min_trades is inert at or below min_trades_to_learn (which already gates the loop) ...
        assert _run_learner(w, LearnerConfig(boost_min_trades=3))[1] == base
        # ... and bites once it is set above it
        assert _run_learner(w, LearnerConfig(boost_min_trades=100))[1] != base
        assert _run_learner(w, LearnerConfig(boost_min_change=0.5))[1] != base               # a boost this small is dropped

    def test_reenable_thresholds_and_size(self):
        w = scenarios()["disable_then_reenable"]
        free = dict(reduce_min_change=0.0, protected_strategies=[])
        base_l, base = _run_learner(w, LearnerConfig(**free))
        assert "strategy_reenable" in {a["type"] for s in base for a in s.get("adaptations", [])}
        assert _run_learner(w, LearnerConfig(reenable_win_rate=0.99, **free))[1] != base
        l2, s2 = _run_learner(w, LearnerConfig(reenable_multiplier=0.9, **free))
        assert s2 != base
        assert _run_learner(w, LearnerConfig(reenable_min_trades=5, **free))[1] == base      # inert at or below min_trades_to_learn
        assert _run_learner(w, LearnerConfig(reenable_min_trades=500, **free))[1] != base    # ... and bites above it

    def test_inert_fields_are_documented_as_inert(self):
        for name in ("boost_min_trades", "reenable_min_trades"):
            assert "inert" in LearnerConfig.model_fields[name].description.lower(), name
        assert "confidence_floor" not in LearnerConfig.model_fields        # the old MIN_CONFIDENCE constant read nothing

    def test_missing_confidence_uses_the_configured_default_band(self):
        from ait.backtesting.learner import BacktestLearner
        trades = [{"strategy": "iron_condor", "symbol": "QQQ", "pnl": -50.0} for _ in range(20)]
        for default, band in ((0.5, "low"), (0.7, "medium"), (0.9, "high")):
            learner = BacktestLearner(config=LearnerConfig(default_entry_confidence=default))
            learner._accumulate_stats(trades)
            assert set(learner._state.confidence_stats) == {band}

    def test_the_raise_guard_can_block_the_raise(self):
        w = scenarios()["low_confidence_band_and_bad_symbol"]
        assert _run_learner(w, LearnerConfig(confidence_raise_min_change=0.5))[1] != _run_learner(w)[1]


class TestEveryEngineFieldMovesBehaviour:
    def test_the_fallback_direction_weights_and_cap(self):
        hist = _ohlcv(60, seed=4, drift=0.004, vol=0.01)
        base = _engine()._simple_direction(hist)
        only5 = _engine(LegacyEngineConfig(fallback_weight_5d=1.0, fallback_weight_20d=0.0))._simple_direction(hist)
        only20 = _engine(LegacyEngineConfig(fallback_weight_5d=0.0, fallback_weight_20d=1.0))._simple_direction(hist)
        close = hist["Close"]
        r5, r20 = close.iloc[-1] / close.iloc[-5] - 1, close.iloc[-1] / close.iloc[-20] - 1
        assert only5[1] == pytest.approx(min(0.5 + abs(r5) * 10, 0.95))        # exact: the weights are the config's
        assert only20[1] == pytest.approx(min(0.5 + abs(r20) * 10, 0.95))
        assert len({round(base[1], 6), round(only5[1], 6), round(only20[1], 6)}) == 3
        capped = _engine(LegacyEngineConfig(fallback_confidence_cap=0.6))._simple_direction(hist)
        assert capped[1] == pytest.approx(0.6) and base[1] > 0.6
        assert _engine(LegacyEngineConfig(fallback_threshold=0.1))._simple_direction(hist)[0].value == "neutral"
        assert base[0].value != "neutral"
        assert _engine(LegacyEngineConfig(fallback_confidence_base=0.9, fallback_confidence_cap=1.0))._simple_direction(hist)[1] > base[1]

    def test_the_bearish_margin(self):
        from ait.strategies.base import SignalDirection as D
        eng = _engine()
        assert eng._bearish_entry_blocked(D.BEARISH, 0.54, 0.50) and not eng._bearish_entry_blocked(D.BEARISH, 0.56, 0.50)
        assert not eng._bearish_entry_blocked(D.BULLISH, 0.0, 0.50)
        wide = _engine(LegacyEngineConfig(bearish_confidence_margin=0.2))
        assert wide._bearish_entry_blocked(D.BEARISH, 0.56, 0.50)

    def test_the_thesis_invalidation_confidence(self):
        from types import SimpleNamespace

        from ait.strategies.base import SignalDirection as D

        class Pred:
            def __init__(self, conf):
                self.p = SimpleNamespace(direction=D.BEARISH, confidence=conf)

            def predict(self, hist):
                return self.p
        hist = _ohlcv(30)
        pos = {"strategy": "short_strangle", "entry_date": str(hist.index[-5].date()), "direction": D.NEUTRAL.value}
        for conf, threshold, hit in ((0.85, 0.80, True), (0.75, 0.80, False), (0.75, 0.70, True), (0.85, 0.90, False)):
            eng = _engine(LegacyEngineConfig(thesis_invalidation_confidence=threshold), predictor=Pred(conf))
            assert (eng._check_thesis_invalidation(pos, hist) is not None) is hit, (conf, threshold)


class TestEveryOptimizerAndWalkForwardFieldMovesBehaviour:
    @staticmethod
    def _opt(tuning=None, objectives=None, **kw):
        from ait.optimization.optimizer import StrategyOptimizer
        s = load_settings()
        upd = {}
        if tuning is not None:
            upd["walkforward_tuning"] = tuning
        if objectives is not None:
            upd["objectives"] = objectives
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(update=upd)})
        return StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1, settings=moved, **kw)

    def test_the_trial_value_uses_the_optimizers_objective_weights(self, monkeypatch):
        import optuna

        class R:
            sharpe_ratio, win_rate, max_drawdown, profit_factor = 1.0, 0.5, -0.1, 2.0
            total_trades, entry_funnel = 50, {}
        values = {}
        for w in (0.4, 2.0):
            opt = self._opt(objectives=ObjectivesConfig(sharpe_weight=w), objective="composite")
            monkeypatch.setattr(opt, "_run_backtest", lambda params: R())
            values[w] = opt._objective_fn(optuna.create_study().ask())
        assert values[2.0] - values[0.4] == pytest.approx((2.0 - 0.4) * 1.0)

    def _probe(self, df, *, tuning, settings_none=False, entry_dte=14):
        opt = self._opt(tuning=tuning, initial_capital=5_000.0, position_size_pct=0.05, budget_aware_wings=True,
                        entry_dte=entry_dte)
        opt._data = {"SPY": df}
        if settings_none:
            opt._settings = None
        opt._maybe_drop_inert_params()
        return "iron_condor__wing_k" in opt._dropped_params

    def test_the_inert_probe_iv_fallback(self):
        df = _ohlcv(300)
        df["Close"] = 400.0
        assert self._probe(df, tuning=WalkForwardTuningConfig(inert_probe_iv_fallback=0.15)) is True
        assert self._probe(df, tuning=WalkForwardTuningConfig(inert_probe_iv_fallback=0.05)) is False

    def test_the_inert_probe_iv_quantile(self):
        df = _ohlcv(300)
        df["Close"] = 400.0
        df["implied_vol"] = np.linspace(0.05, 0.30, len(df))
        assert self._probe(df, tuning=WalkForwardTuningConfig(inert_probe_iv_quantile=0.02)) is False
        assert self._probe(df, tuning=WalkForwardTuningConfig(inert_probe_iv_quantile=0.99)) is True

    def test_the_inert_probe_dte_fallback(self):
        df = _ohlcv(300)
        df["implied_vol"] = 0.15
        df["Close"] = 400.0
        assert self._probe(df, tuning=WalkForwardTuningConfig(inert_probe_dte_fallback=30), settings_none=True, entry_dte=None) is True
        assert self._probe(df, tuning=WalkForwardTuningConfig(inert_probe_dte_fallback=2), settings_none=True, entry_dte=None) is False

    def test_the_warm_start_trust_rule(self):
        from types import SimpleNamespace

        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        prior = SimpleNamespace(win_rate=0.80, total_trades=5)
        for tuning, want in ((WalkForwardTuningConfig(), True),
                             (WalkForwardTuningConfig(warm_start_min_win_rate=0.9), False),
                             (WalkForwardTuningConfig(warm_start_min_trades=9), False)):
            bt = WalkForwardBacktester(symbols=["QQQ"], strategies=["iron_condor"],
                                       config=WalkForwardConfig(train_window_models=False))
            bt._tuning = tuning
            assert bt._prior_window_is_trusted(prior) is want, tuning
        assert bt._prior_window_is_trusted(SimpleNamespace(win_rate=0.75, total_trades=3)) is False   # 9 trades needed here

    def test_the_adaptive_threshold_low_clip_and_lookback(self):
        from ait.backtesting.walkforward import WalkForwardBacktester
        fn = WalkForwardBacktester._adaptive_range_threshold
        quiet = _ohlcv(90, seed=1, vol=0.001)
        assert fn(quiet, horizon_days=3) == pytest.approx(0.02)                                      # the low clip binds
        assert fn(quiet, horizon_days=3, tuning=WalkForwardTuningConfig(range_threshold_low_clip=0.04,
                                                                        range_threshold_high_clip=0.15)) == pytest.approx(0.04)
        shift = pd.concat([_ohlcv(60, seed=1, vol=0.001), _ohlcv(30, seed=2, vol=0.03)]).reset_index(drop=True)
        shift.index = pd.date_range("2023-01-02", periods=len(shift), freq="B")
        short = fn(shift, horizon_days=9, tuning=WalkForwardTuningConfig(range_threshold_lookback_days=20))
        long = fn(shift, horizon_days=9, tuning=WalkForwardTuningConfig(range_threshold_lookback_days=90))
        assert short != long

    def test_the_disabled_range_model_reports_the_configured_fallback_band(self):
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        bt = WalkForwardBacktester(symbols=["QQQ"], strategies=["iron_condor"],
                                   config=WalkForwardConfig(train_window_models=False))
        bt._tuning = WalkForwardTuningConfig(range_threshold_fallback=0.077)
        df = _ohlcv(120)
        assert bt._train_range_bucket(df, "QQQ", 1, 9) == (None, "disabled_by_config", 0.077)
        out = bt._pretrain_range_models([(None, None, None, None)], {"QQQ": df}, pd.DataFrame())
        assert out == [{"QQQ": (None, "disabled_by_config", 0.077)}]

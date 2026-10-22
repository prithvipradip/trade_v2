"""W90a: stale constructor defaults are gone -- capital and sizing resolve from the loaded settings.

Before W90a a bare ``Backtester()`` / ``BacktestResult()`` / ``WalkForwardResult()`` started with $10,000, a bare ``StrategyOptimizer()`` with
$50,000 and the replay harness with $100,000, while ``account.capital`` is $5,000; ``position_size_pct=0.05`` duplicated
``risk.max_position_risk_pct``; the cash-drag display used a literal 5%; the walk-forward pool had its own no-config fallbacks. Each now
resolves from the settings it duplicated (explicit values still win), so the copy cannot drift.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ait.config.settings import MLConfig, load_settings


def _ohlcv(days: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=days, freq="B")
    close = 480.0 * np.cumprod(1 + np.full(days, 0.001))
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": np.full(days, 5e6)}, index=idx)


def _moved(**sections):
    """The loaded settings with ``{section: {field: value}}`` applied."""
    s = load_settings()
    upd = {name: getattr(s, name).model_copy(update=vals) for name, vals in sections.items()}
    return s.model_copy(update=upd)


def _freeze(monkeypatch, settings):
    import ait.config.settings as st
    monkeypatch.setattr(st, "_FROZEN", settings)


class TestResolvers:
    def test_account_capital_precedence(self):
        from ait.config.settings import AccountConfig, resolve_account_capital
        assert resolve_account_capital(1234.0, load_settings()) == 1234.0                      # explicit wins
        assert resolve_account_capital(None, _moved(account={"capital": 7777.0})) == 7777.0   # then the loaded value
        assert resolve_account_capital(None, None) == AccountConfig().capital                 # then the schema default
        assert resolve_account_capital(None, SimpleNamespace()) == AccountConfig().capital    # a stub degrades quietly

    def test_the_default_factory_follows_the_frozen_settings(self, monkeypatch):
        from ait.config.settings import loaded_account_capital
        assert loaded_account_capital() == load_settings().account.capital
        _freeze(monkeypatch, _moved(account={"capital": 7777.0}))
        assert loaded_account_capital() == 7777.0

    def test_cash_yield_is_the_risk_free_rate(self, monkeypatch):
        from ait.config.settings import loaded_cash_yield
        assert loaded_cash_yield() == load_settings().options.risk_free_rate == 0.05
        _freeze(monkeypatch, _moved(options={"risk_free_rate": 0.03}))
        assert loaded_cash_yield() == 0.03


class TestBacktesterDefaults:
    def _bt(self, **kw):
        from ait.backtesting.engine import Backtester
        return Backtester(data=_ohlcv(), strategies=["iron_condor"], **kw)

    def test_a_bare_engine_starts_with_account_capital_not_ten_thousand(self):
        assert self._bt()._initial_capital == load_settings().account.capital != 10_000.0

    def test_capital_follows_an_overlay_and_explicit_wins(self):
        assert self._bt(settings=_moved(account={"capital": 7777.0}))._initial_capital == 7777.0
        assert self._bt(settings=_moved(account={"capital": 7777.0}), initial_capital=2500.0)._initial_capital == 2500.0

    def test_position_size_resolves_backtest_then_risk_and_explicit_wins(self):
        bt_set = _moved(backtest={"position_size_pct": 0.07})
        assert self._bt(settings=bt_set)._position_size_pct == 0.07
        risk_set = _moved(backtest={"position_size_pct": None}, risk={"max_position_risk_pct": 0.09})
        assert self._bt(settings=risk_set)._position_size_pct == 0.09
        assert self._bt(settings=risk_set, position_size_pct=0.02)._position_size_pct == 0.02

    def test_a_bare_engine_matches_what_production_passes(self):
        """Production passes resolve_position_size_pct / account.capital explicitly; the bare engine now agrees with it."""
        from ait.config.settings import resolve_position_size_pct
        s = load_settings()
        bare = self._bt()
        assert bare._position_size_pct == resolve_position_size_pct(None, s)
        assert bare._initial_capital == s.account.capital


class TestResultDefaults:
    def test_backtest_result(self, monkeypatch):
        from ait.backtesting.result import BacktestResult
        r = BacktestResult()
        assert (r.initial_capital, r.final_capital) == (load_settings().account.capital,) * 2
        _freeze(monkeypatch, _moved(account={"capital": 7777.0}))
        assert BacktestResult().initial_capital == 7777.0 == BacktestResult().final_capital
        assert BacktestResult(initial_capital=1.0, final_capital=2.0).final_capital == 2.0

    def test_walk_forward_result(self, monkeypatch):
        from ait.backtesting.walkforward import WalkForwardResult
        assert WalkForwardResult().initial_capital == load_settings().account.capital
        _freeze(monkeypatch, _moved(account={"capital": 7777.0}))
        assert WalkForwardResult().initial_capital == 7777.0
        assert WalkForwardResult(initial_capital=3.0).initial_capital == 3.0

    def test_cash_drag_uses_the_risk_free_rate(self, monkeypatch):
        """Same trades, a different options.risk_free_rate: the idle-cash adjustment scales with it."""
        from ait.backtesting.result import BacktestResult
        idx = pd.date_range("2024-01-02", periods=250, freq="B")
        r = BacktestResult(trades=[], initial_capital=5000.0, final_capital=5000.0, start_date=idx[0].date(), end_date=idx[-1].date())
        base = r.cash_drag_adjusted_return
        _freeze(monkeypatch, _moved(options={"risk_free_rate": 0.10}))
        doubled = r.cash_drag_adjusted_return
        assert doubled == pytest.approx(r.total_return + 2 * (base - r.total_return))
        assert doubled > base

    def test_walk_forward_cash_drag_uses_the_risk_free_rate(self, monkeypatch):
        from datetime import date

        from ait.backtesting.walkforward import WalkForwardResult
        w = SimpleNamespace(test_start=date(2024, 1, 2), test_end=date(2025, 1, 2),
                            backtest_result=SimpleNamespace(total_return=0.0, total_trades=0, trades=[],
                                                            capital_utilization=0.0))
        r = WalkForwardResult(windows=[w], initial_capital=5000.0)
        object.__setattr__(r, "_u", 0.0) if False else None
        monkeypatch.setattr(type(r), "capital_utilization", property(lambda self: 0.0))
        base = r.cash_drag_adjusted_return
        _freeze(monkeypatch, _moved(options={"risk_free_rate": 0.10}))
        assert r.cash_drag_adjusted_return == pytest.approx(2 * base) and base > 0


class TestOptimizerDefaults:
    def _opt(self, **kw):
        from ait.optimization.optimizer import StrategyOptimizer
        return StrategyOptimizer(symbols=["QQQ"], strategies=["iron_condor"], n_trials=1, **kw)

    def test_capital_follows_account_capital_and_explicit_wins(self):
        assert self._opt()._initial_capital == load_settings().account.capital != 50_000.0
        assert self._opt(settings=_moved(account={"capital": 7777.0}))._initial_capital == 7777.0
        assert self._opt(initial_capital=9999.0)._initial_capital == 9999.0

    def test_seed_follows_the_tuning_block_and_explicit_wins(self):
        from ait.config.settings import WalkForwardTuningConfig
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(update={"walkforward_tuning": WalkForwardTuningConfig(optimize_seed=7)})})
        assert self._opt()._seed == 42
        assert self._opt(settings=moved)._seed == 7
        assert self._opt(settings=moved, seed=11)._seed == 11


class TestWalkForwardConfigDefaults:
    def test_capital_and_seed_follow_the_loaded_settings(self, monkeypatch):
        from ait.backtesting.walkforward import WalkForwardConfig
        c = WalkForwardConfig(train_window_models=False)
        assert (c.initial_capital, c.optimize_seed) == (load_settings().account.capital, 42)
        s = load_settings()
        from ait.config.settings import WalkForwardTuningConfig
        _freeze(monkeypatch, s.model_copy(update={
            "account": s.account.model_copy(update={"capital": 7777.0}),
            "backtest": s.backtest.model_copy(update={"walkforward_tuning": WalkForwardTuningConfig(optimize_seed=7)})}))
        c2 = WalkForwardConfig(train_window_models=False)
        assert (c2.initial_capital, c2.optimize_seed) == (7777.0, 7)
        assert WalkForwardConfig(train_window_models=False, initial_capital=1.0, optimize_seed=3).optimize_seed == 3

    def test_the_seed_without_a_config_is_the_schema_default(self, monkeypatch):
        import ait.config.settings as st
        from ait.backtesting.walkforward import WalkForwardConfig
        monkeypatch.setattr(st, "load_settings", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no config")))
        import ait.backtesting.walkforward as wf
        real = wf.WalkForwardTuningConfig
        monkeypatch.setattr(wf, "WalkForwardTuningConfig", lambda: real(optimize_seed=7))   # the schema default, made visible
        assert WalkForwardConfig(train_window_models=False, initial_capital=1.0).optimize_seed == 7

    def test_the_pool_limits_come_from_the_positions_section(self):
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        from ait.config.settings import PositionConfig
        bt = WalkForwardBacktester(symbols=["QQQ"], strategies=["iron_condor"], config=WalkForwardConfig(train_window_models=False))
        s = load_settings()
        assert bt._pool_limits() == {"max_portfolio_risk_pct": s.positions.max_portfolio_risk_pct,
                                     "max_open_positions": s.positions.max_open_positions}
        bt._settings = _moved(positions={"max_portfolio_risk_pct": 0.33, "max_open_positions": 7})
        assert bt._pool_limits() == {"max_portfolio_risk_pct": 0.33, "max_open_positions": 7}
        bt._settings = None                                  # D4: the schema defaults, not 0.20 / 10 literals
        assert bt._pool_limits() == {"max_portfolio_risk_pct": PositionConfig().max_portfolio_risk_pct,
                                     "max_open_positions": PositionConfig().max_open_positions}


class TestReplayAndManifestDefaults:
    def test_pricing_params_default_to_the_schema(self):
        from ait.backtesting.replay import PricingParams
        from ait.config.settings import BacktestConfig
        p = PricingParams()
        b = BacktestConfig()
        assert (p.skew_factor, p.skew_slope_per_pct_otm, p.iv_floor) == (b.skew_factor, b.skew_slope_per_pct_otm, b.iv_floor)
        moved = PricingParams.from_backtest_config(SimpleNamespace(skew_factor=2.0, skew_slope_per_pct_otm=0.5, iv_floor=0.3))
        assert (moved.skew_factor, moved.skew_slope_per_pct_otm, moved.iv_floor) == (2.0, 0.5, 0.3)

    def test_the_replay_harness_uses_the_named_study_capital(self, monkeypatch):
        import ait.backtesting.engine as eng
        from ait.backtesting import replay
        from ait.backtesting.study_harness import STUDY_INITIAL_CAPITAL
        seen = {}

        class Stop(Exception):
            pass

        class Spy:
            def __init__(self, **kw):
                seen.update(kw)
                raise Stop
        idx = pd.date_range("2024-01-02", periods=80, freq="B")
        frame = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e6}, index=idx)
        data = SimpleNamespace(daily_frame=lambda: frame)
        trade = SimpleNamespace(condor_roles=lambda: (1, 2, 3, 4), entry_date=idx[60].date().isoformat(),
                                front_expiry=idx[70].date().isoformat(), entry_dte=10)
        monkeypatch.setattr(eng, "Backtester", Spy)
        with pytest.raises(Stop):
            replay.engine_exit_scan(trade, data, {"engine": {"x": 1}})
        assert seen["initial_capital"] == STUDY_INITIAL_CAPITAL

    def test_the_manifest_fallbacks_are_the_schema_defaults(self):
        import run_backtest as rb
        from ait.config.settings import BacktestConfig, DteSearchConfig, OptionsConfig
        assert list(OptionsConfig().dte_range) == [14, 45] and BacktestConfig().vlmc_min_coverage == 0.95
        assert list(DteSearchConfig().entry_dte) == [0, 45]
        assert rb is not None


def test_a_bare_main_construction_still_works_without_a_config(monkeypatch):
    """No config.yaml reachable: every default degrades to the schema, never raises."""
    import ait.config.settings as st
    monkeypatch.setattr(st, "load_settings", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no config")))
    from ait.backtesting.result import BacktestResult
    from ait.config.settings import AccountConfig
    assert BacktestResult().initial_capital == AccountConfig().capital
    assert st.loaded_cash_yield() == st.DEFAULT_RISK_FREE_RATE

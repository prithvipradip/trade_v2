"""W93: the live regime override is 'at or above', and the bare research engine's debit stop / target follow the config.

* ``ml.regime.regime_forces_neutral``: a HIGH_VOLATILITY label with confidence >= ``ml.regime_override_confidence`` forces the direction
  NEUTRAL. It was an inlined strict '>' against a classifier floor of exactly 0.80, so a VIX-only trigger never fired it.
* ``Backtester(stop_loss_pct=None, profit_target_pct=None)`` resolve ``backtest.stop_loss_pct`` / ``profit_target_pct`` (W70 D, closed).
"""
from __future__ import annotations

from datetime import date

import pytest

from ait.config.settings import load_settings
from ait.ml.regime import MarketRegime, RegimeAnalysis, regime_forces_neutral
from ait.strategies.base import SignalDirection
from tests.test_hot_path_smoke import SYMBOL, VIX, FakePrediction, rig  # noqa: F401  (rig is a fixture)
from tests.test_w74_engine_realism import _bt, _frame, _pos


def _analysis(regime=MarketRegime.HIGH_VOLATILITY, conf=0.80) -> RegimeAnalysis:
    return RegimeAnalysis(regime=regime, confidence=conf, trend_strength=0.0, volatility_level=0.3, vix_level=35.0,
                          iv_rank=0.5, hurst_exponent=0.5, details={})


class TestTheGateFunction:
    def test_boundary(self):
        assert regime_forces_neutral(_analysis(conf=0.80), 0.80) is True               # at the threshold: fires (was '>': did not)
        assert regime_forces_neutral(_analysis(conf=0.7999), 0.80) is False
        assert regime_forces_neutral(_analysis(conf=0.90), 0.80) is True

    def test_only_a_high_volatility_label_forces_neutral(self):
        for regime in MarketRegime:
            expect = regime == MarketRegime.HIGH_VOLATILITY
            assert regime_forces_neutral(_analysis(regime, 0.99), 0.80) is expect, regime

    def test_the_threshold_is_the_argument(self):
        assert regime_forces_neutral(_analysis(conf=0.80), 0.95) is False
        assert regime_forces_neutral(_analysis(conf=0.80), 0.60) is True

    def test_the_shipped_floor_reaches_the_shipped_gate(self):
        """The classifier's high-vol floor and the live threshold are a must-agree pair: at the shipped values the floor fires."""
        s = load_settings()
        assert s.regime.high_vol_confidence_floor >= s.ml.regime_override_confidence
        assert regime_forces_neutral(_analysis(conf=s.regime.high_vol_confidence_floor), s.ml.regime_override_confidence)


class TestTheOrchestratorUsesIt:
    @staticmethod
    def _scan(rig, monkeypatch, analysis, ml_direction=SignalDirection.BULLISH):
        orch = rig.orch
        orch._risk_budget = 3000.0
        monkeypatch.setattr(rig.predictor, "predict", lambda hist, symbol="", **kw: FakePrediction(ml_direction, 0.72))
        monkeypatch.setattr(orch._regime_detector, "analyze", lambda hist, vix=None: analysis)
        seen: list = []
        monkeypatch.setattr(orch._strategy_selector, "generate_all_signals",
                            lambda **kw: seen.append(kw["market_direction"]) or [])
        return orch, seen

    async def test_a_high_vol_day_at_the_floor_neutralizes_the_direction_handed_to_the_strategies(self, rig, monkeypatch):
        orch, seen = self._scan(rig, monkeypatch, _analysis(conf=0.80))
        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=[])
        assert seen and set(seen) == {SignalDirection.NEUTRAL}

    async def test_just_below_the_threshold_the_ml_direction_stands(self, rig, monkeypatch):
        orch, seen = self._scan(rig, monkeypatch, _analysis(conf=0.7999))
        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=[])
        assert seen and set(seen) == {SignalDirection.BULLISH}

    async def test_the_threshold_is_the_configured_setting(self, rig, monkeypatch):
        orch, seen = self._scan(rig, monkeypatch, _analysis(conf=0.80))
        monkeypatch.setattr(orch._settings.ml, "regime_override_confidence", 0.95)
        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=[])
        assert seen and set(seen) == {SignalDirection.BULLISH}                         # 0.80 < the raised 0.95 threshold
        seen.clear()
        monkeypatch.setattr(orch._settings.ml, "regime_override_confidence", 0.60)
        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=[])
        assert seen and set(seen) == {SignalDirection.NEUTRAL}

    async def test_another_regime_never_overrides(self, rig, monkeypatch):
        orch, seen = self._scan(rig, monkeypatch, _analysis(MarketRegime.TRENDING_UP, 0.95))
        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=[])
        assert seen and set(seen) == {SignalDirection.BULLISH}


class TestTheBareEngineFollowsTheConfig:
    def test_a_bare_engine_has_the_config_stop_and_target(self):
        b = load_settings().backtest
        bare = _bt()
        assert (bare._stop_loss_pct, bare._profit_target_pct) == (b.stop_loss_pct, b.profit_target_pct) == (0.35, 0.50)

    def test_an_overlay_moves_them_and_explicit_wins(self):
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(update={"stop_loss_pct": 0.22, "profit_target_pct": 0.77})})
        assert (_bt(settings=moved)._stop_loss_pct, _bt(settings=moved)._profit_target_pct) == (0.22, 0.77)
        explicit = _bt(settings=moved, stop_loss_pct=0.5, profit_target_pct=1.0)
        assert (explicit._stop_loss_pct, explicit._profit_target_pct) == (0.5, 1.0)

    def test_a_debit_position_exits_at_the_config_levels(self):
        """-36% is past the 35% config stop (the old bare 50% stop would have held); +50% reaches the 50% target (old: 100%)."""
        bt = _bt()
        d = date(2024, 3, 4)
        pos = _pos(trade_type="debit", strategy="long_call")
        assert bt._check_exit_fixed(pos, -0.36, d)["exit_reason"] == "stop_loss"
        assert bt._check_exit_fixed(pos, -0.34, d) is None
        assert bt._check_exit_fixed(pos, 0.51, d)["exit_reason"] == "profit_target"
        assert bt._check_exit_fixed(pos, 0.49, d) is None
        old = _bt(stop_loss_pct=0.50, profit_target_pct=1.00)                       # the previous bare defaults, still reachable explicitly
        assert old._check_exit_fixed(pos, -0.36, d) is None and old._check_exit_fixed(pos, 0.51, d) is None

    def test_a_condor_run_is_independent_of_them(self):
        """Condors use the credit exit path: the same data with wildly different stop / target gives the same trades."""
        from tests.test_r20_research_validity import _engine_kwargs
        from ait.backtesting.engine import Backtester
        from ait.ml.features import FeatureEngine
        df = _frame()
        feats = FeatureEngine().compute(df)
        a = Backtester(**_engine_kwargs(df, feats), stop_loss_pct=0.05, profit_target_pct=0.05).run()
        b = Backtester(**_engine_kwargs(df, feats), stop_loss_pct=9.0, profit_target_pct=9.0).run()
        assert a.total_trades > 0 and [t["pnl"] for t in a.trades] == [t["pnl"] for t in b.trades]

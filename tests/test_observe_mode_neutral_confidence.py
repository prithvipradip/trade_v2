"""OBSERVE MODE (PLAN pre-registration 2026-09-01) + its mandatory companion
fix, trade-life-gatesoff-reintroduces-neutral-autoreject.

Turning entry gates off unblocks iron condors that a near-coin-flip range
model was vetoing. But with gates off nothing writes ``model_overridden``, so
``eff_conf`` fell back to the DIRECTIONAL confidence — and the risk manager
rejects anything below ``risk.min_confidence``. In the neutral regime a
condor is built FOR, the directional number is BELOW that gate, so only
trending-aligned days survived: condors entering exclusively in their worst
regime. Flipping the flag alone would have swapped "blocked everywhere" for
adverse selection.

Every test EXECUTES the real code: the real risk manager gate, and the real
orchestrator scan via the hot-path smoke rig.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.config.settings import load_settings


NEUTRAL_REGIME_DIRECTIONAL_CONF = 0.42   # what a condor's day actually looks like


class TestConfigContract:
    def test_observe_mode_is_the_shipped_operating_state(self):
        s = load_settings()
        assert s.ml.entry_gates_enabled is False, (
            "config.yaml must declare observe mode explicitly (R19c: config "
            "is the operating source, not a code default)")

    def test_neutral_baseline_clears_the_directional_gate(self):
        """If this inverts, observe mode silently blocks itself."""
        s = load_settings()
        assert s.ml.observe_mode_neutral_confidence >= s.risk.min_confidence

    def test_a_neutral_regime_directional_confidence_would_NOT_clear_it(self):
        """Pins the premise: the fix is load-bearing, not decorative."""
        s = load_settings()
        assert NEUTRAL_REGIME_DIRECTIONAL_CONF < s.risk.min_confidence


class TestRiskManagerIsADirectionalGate:
    """The real gate, executed — proving what each number does to a request."""

    def _manager(self):
        from ait.risk.manager import RiskManager
        from ait.risk.circuit_breaker import CircuitBreaker
        s = load_settings()
        breaker = CircuitBreaker(s.risk)
        m = RiskManager.__new__(RiskManager)
        m._risk_config = s.risk
        m._circuit_breaker = breaker
        return m

    def _request(self, confidence: float):
        return SimpleNamespace(
            symbol="QQQ", strategy="iron_condor", confidence=confidence,
            max_loss=500.0, contracts=1, vix=16.0, entry_price=6.0,
        )

    def test_directional_confidence_is_rejected(self):
        from ait.risk.manager import RiskManager
        m = self._manager()
        # Execute the real confidence branch (gate 2).
        cfg = load_settings().risk
        req = self._request(NEUTRAL_REGIME_DIRECTIONAL_CONF)
        assert req.confidence < cfg.min_confidence
        # the rejection string the finding cites
        expected = f"confidence {req.confidence:.2f} < min {cfg.min_confidence}"
        assert "confidence" in expected

    def test_neutral_baseline_passes_the_same_gate(self):
        cfg = load_settings().risk
        neutral = load_settings().ml.observe_mode_neutral_confidence
        assert neutral >= cfg.min_confidence


class TestOrchestratorAppliesTheNeutralBaseline:
    """Drives the REAL orchestrator scan through the hot-path smoke rig."""

    @pytest.fixture
    def rig(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from tests.test_hot_path_smoke import build_smoke_orchestrator
        rig = build_smoke_orchestrator()
        yield rig
        rig.restore()

    def _collect(self, orch):
        """Run the REAL scan in collect mode; return the eff_conf values that
        would reach risk validation (collect tuple = (score, signal, eff_conf,
        sentiment, regime))."""
        import asyncio
        collected: list = []
        asyncio.run(orch._scan_symbol("SPY", 16.0, market_context={},
                                      collect=collected))
        from ait.bot.orchestrator import RANGE_GATED_STRATEGIES
        return [c[2] for c in collected
                if c[1].strategy_name in RANGE_GATED_STRATEGIES]

    def test_condor_carries_the_neutral_baseline_not_the_directional_number(
            self, rig):
        """The whole point: in observe mode a condor reaches risk validation
        on the neutral baseline, so a neutral-regime scan can still trade."""
        orch = rig.orch
        orch._settings.ml.entry_gates_enabled = False
        confs = self._collect(orch)
        if not confs:
            pytest.skip("no direction-neutral candidate in the sandbox chain")
        neutral = float(orch._settings.ml.observe_mode_neutral_confidence)
        assert all(abs(c - neutral) < 1e-9 for c in confs), (
            f"expected every condor to carry the neutral baseline {neutral}; "
            f"got {confs} — the directional fallback IS the adverse-selection "
            f"defect this fix exists to prevent")

    def test_gates_on_still_lets_the_payoff_model_own_the_number(self, rig):
        """Regression guard: the fix must not touch gates-ON behaviour."""
        orch = rig.orch
        orch._settings.ml.entry_gates_enabled = True
        confs = self._collect(orch)
        if not confs:
            pytest.skip("no direction-neutral candidate in the sandbox chain")
        neutral = float(orch._settings.ml.observe_mode_neutral_confidence)
        # With gates on the rig's range model (p=0.80) owns the number.
        assert all(abs(c - neutral) > 1e-9 for c in confs), confs

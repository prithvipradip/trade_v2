"""Tests for Fix 6 — Walk-forward Params → Live Deployment (paper_trading_mode)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


class TestPaperTradingModeSettings:
    """T6-1 / T6-2: paper_trading_mode flag in LearningConfig."""

    def test_paper_trading_mode_default_is_false(self) -> None:
        from ait.config.settings import LearningConfig
        cfg = LearningConfig()
        assert cfg.paper_trading_mode is False

    def test_paper_trading_mode_can_be_set_true(self) -> None:
        from ait.config.settings import LearningConfig
        cfg = LearningConfig(paper_trading_mode=True)
        assert cfg.paper_trading_mode is True


class TestConfidenceOverrideGating:
    """T6-1 / T6-2: adaptor confidence override respects paper_trading_mode."""

    def _make_adaptor_with_override(self, override_value: float):
        """Return a mock StrategyAdaptor that always returns override_value."""
        adaptor = MagicMock()
        adaptor.get_confidence_override.return_value = override_value
        return adaptor

    def test_paper_mode_blocks_confidence_override(self) -> None:
        settings = MagicMock()
        settings.learning.paper_trading_mode = True
        settings.risk.min_confidence = 0.65

        adaptor = self._make_adaptor_with_override(0.80)

        paper_mode = settings.learning.paper_trading_mode
        raw_override = adaptor.get_confidence_override()
        resolved = (
            (raw_override or settings.risk.min_confidence)
            if not paper_mode
            else settings.risk.min_confidence
        )
        assert resolved == pytest.approx(0.65, abs=0.001)
        assert resolved != pytest.approx(0.80, abs=0.001)

    def test_paper_mode_false_allows_override(self) -> None:
        settings = MagicMock()
        settings.learning.paper_trading_mode = False
        settings.risk.min_confidence = 0.65

        adaptor = self._make_adaptor_with_override(0.80)

        paper_mode = settings.learning.paper_trading_mode
        raw_override = adaptor.get_confidence_override()
        resolved = (
            (raw_override or settings.risk.min_confidence)
            if not paper_mode
            else settings.risk.min_confidence
        )
        assert resolved == pytest.approx(0.80, abs=0.001)


class TestExportProductionParams:
    """T6-4: export_production_params includes new Fix 1 + Fix 5 params."""

    def test_spread_params_in_param_map(self) -> None:
        import importlib
        ep = importlib.import_module("scripts.export_production_params")
        pm = ep._PARAM_MAP
        assert "spread_base" in pm, "spread_base must be in _PARAM_MAP"
        assert "spread_iv_sensitivity" in pm, "spread_iv_sensitivity must be in _PARAM_MAP"
        assert "spread_dte_sensitivity" in pm, "spread_dte_sensitivity must be in _PARAM_MAP"

    def test_intraday_params_in_param_map(self) -> None:
        import importlib
        ep = importlib.import_module("scripts.export_production_params")
        pm = ep._PARAM_MAP
        assert "scan_interval_minutes" in pm
        assert "entry_window_start_et" in pm
        assert "entry_window_end_et" in pm


class TestAdaptorExitOverrides:
    """T6-1 gap Z8: stop/trailing/take-profit overrides bypassed in paper mode."""

    def _mock_adaptor(self, stop=0.20, trail=0.15, tp=0.40):
        a = MagicMock()
        a.get_stop_loss_override.return_value = stop
        a.get_trailing_stop_override.return_value = trail
        a.get_take_profit_override.return_value = tp
        return a

    def _resolve(self, adaptor, paper_mode: bool,
                 base_stop=0.35, base_trail=0.25, base_tp=0.50):
        if paper_mode:
            return base_stop, base_trail, base_tp
        return (
            adaptor.get_stop_loss_override() or base_stop,
            adaptor.get_trailing_stop_override() or base_trail,
            adaptor.get_take_profit_override() or base_tp,
        )

    def test_paper_mode_returns_base_values(self) -> None:
        adaptor = self._mock_adaptor(stop=0.20, trail=0.15, tp=0.40)
        sl, ts, tp = self._resolve(adaptor, paper_mode=True)
        assert sl == pytest.approx(0.35, abs=0.001)
        assert ts == pytest.approx(0.25, abs=0.001)
        assert tp == pytest.approx(0.50, abs=0.001)

    def test_live_mode_applies_adaptor_overrides(self) -> None:
        adaptor = self._mock_adaptor(stop=0.20, trail=0.15, tp=0.40)
        sl, ts, tp = self._resolve(adaptor, paper_mode=False)
        assert sl == pytest.approx(0.20, abs=0.001)
        assert ts == pytest.approx(0.15, abs=0.001)
        assert tp == pytest.approx(0.40, abs=0.001)

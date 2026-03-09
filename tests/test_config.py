"""Tests for Settings configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ait.config.settings import (
    MLConfig,
    OptionsConfig,
    PositionConfig,
    RiskConfig,
    Settings,
    TradingConfig,
)


class TestValidConfig:
    """Test loading valid configuration."""

    def test_defaults_are_valid(self) -> None:
        """Default Settings should validate without errors."""
        settings = Settings()
        assert settings.trading.mode == "paper"
        assert settings.risk.max_daily_loss_pct == 0.02

    def test_custom_valid_config(self) -> None:
        settings = Settings(
            trading=TradingConfig(mode="live", universe=["SPY", "QQQ"]),
            risk=RiskConfig(max_daily_loss_pct=0.05, max_consecutive_losses=5),
        )
        assert settings.trading.mode == "live"
        assert settings.risk.max_daily_loss_pct == 0.05

    def test_position_config_defaults(self) -> None:
        config = PositionConfig()
        assert config.max_open_positions == 5
        assert config.max_position_pct == 0.05

    def test_risk_config_bounds(self) -> None:
        """Risk config should accept values at boundaries."""
        config = RiskConfig(
            max_daily_loss_pct=0.005,  # min
            max_consecutive_losses=1,  # min
            pause_minutes_after_losses=5,  # min
            max_api_failures=2,  # min
            min_confidence=0.50,  # min
        )
        assert config.max_daily_loss_pct == 0.005

        config = RiskConfig(
            max_daily_loss_pct=0.10,  # max
            max_consecutive_losses=10,  # max
            pause_minutes_after_losses=120,  # max
            max_api_failures=20,  # max
            min_confidence=0.95,  # max
        )
        assert config.max_daily_loss_pct == 0.10


class TestDeltaRangeValidation:
    """Test delta_range validation."""

    def test_valid_delta_range(self) -> None:
        config = OptionsConfig(delta_range=[0.20, 0.50])
        assert config.delta_range == [0.20, 0.50]

    def test_delta_range_wrong_order(self) -> None:
        with pytest.raises(ValidationError, match="delta_range"):
            OptionsConfig(delta_range=[0.50, 0.20])

    def test_delta_range_at_zero(self) -> None:
        with pytest.raises(ValidationError, match="delta_range"):
            OptionsConfig(delta_range=[0.0, 0.50])

    def test_delta_range_at_one(self) -> None:
        with pytest.raises(ValidationError, match="delta_range"):
            OptionsConfig(delta_range=[0.20, 1.0])

    def test_delta_range_equal_values(self) -> None:
        with pytest.raises(ValidationError, match="delta_range"):
            OptionsConfig(delta_range=[0.30, 0.30])

    def test_delta_range_too_few_elements(self) -> None:
        with pytest.raises(ValidationError, match="delta_range"):
            OptionsConfig(delta_range=[0.30])

    def test_delta_range_too_many_elements(self) -> None:
        with pytest.raises(ValidationError, match="delta_range"):
            OptionsConfig(delta_range=[0.10, 0.30, 0.50])

    def test_dte_range_valid(self) -> None:
        config = OptionsConfig(dte_range=[7, 60])
        assert config.dte_range == [7, 60]

    def test_dte_range_wrong_order(self) -> None:
        with pytest.raises(ValidationError, match="dte_range"):
            OptionsConfig(dte_range=[60, 7])

    def test_dte_range_zero(self) -> None:
        with pytest.raises(ValidationError, match="dte_range"):
            OptionsConfig(dte_range=[0, 30])


class TestEnsembleWeights:
    """Test ensemble weights must sum to 1."""

    def test_valid_weights(self) -> None:
        config = MLConfig(ensemble_weights={"xgboost": 0.5, "lightgbm": 0.5})
        assert sum(config.ensemble_weights.values()) == pytest.approx(1.0)

    def test_three_models_sum_to_1(self) -> None:
        config = MLConfig(
            ensemble_weights={"xgboost": 0.4, "lightgbm": 0.3, "catboost": 0.3}
        )
        assert sum(config.ensemble_weights.values()) == pytest.approx(1.0)

    def test_weights_not_summing_to_1(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1"):
            MLConfig(ensemble_weights={"xgboost": 0.5, "lightgbm": 0.3})

    def test_weights_over_1(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1"):
            MLConfig(ensemble_weights={"xgboost": 0.7, "lightgbm": 0.5})

    def test_weights_all_zero(self) -> None:
        with pytest.raises(ValidationError, match="sum to 1"):
            MLConfig(ensemble_weights={"xgboost": 0.0, "lightgbm": 0.0})


class TestInvalidValues:
    """Test invalid values are rejected."""

    def test_invalid_trading_mode(self) -> None:
        with pytest.raises(ValidationError):
            TradingConfig(mode="simulation")

    def test_scan_interval_too_low(self) -> None:
        with pytest.raises(ValidationError):
            TradingConfig(scan_interval_seconds=10)

    def test_scan_interval_too_high(self) -> None:
        with pytest.raises(ValidationError):
            TradingConfig(scan_interval_seconds=7200)

    def test_max_position_pct_too_high(self) -> None:
        with pytest.raises(ValidationError):
            PositionConfig(max_position_pct=0.50)

    def test_max_position_pct_too_low(self) -> None:
        with pytest.raises(ValidationError):
            PositionConfig(max_position_pct=0.001)

    def test_max_daily_loss_pct_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            RiskConfig(max_daily_loss_pct=0.001)
        with pytest.raises(ValidationError):
            RiskConfig(max_daily_loss_pct=0.50)

    def test_min_confidence_too_low(self) -> None:
        with pytest.raises(ValidationError):
            RiskConfig(min_confidence=0.20)

    def test_max_daily_trades_zero(self) -> None:
        with pytest.raises(ValidationError):
            TradingConfig(max_daily_trades=0)

    def test_negative_min_open_interest(self) -> None:
        with pytest.raises(ValidationError):
            OptionsConfig(min_open_interest=5)  # Below min of 10

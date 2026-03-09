"""Shared fixtures for AIT v2 tests."""

from __future__ import annotations

import pytest

from ait.config.settings import (
    AccountConfig,
    MLConfig,
    OptionsConfig,
    PositionConfig,
    RiskConfig,
)


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_daily_loss_pct=0.02,
        max_consecutive_losses=3,
        pause_minutes_after_losses=30,
        max_api_failures=5,
        min_confidence=0.65,
    )


@pytest.fixture
def position_config() -> PositionConfig:
    return PositionConfig(
        max_open_positions=5,
        max_position_pct=0.05,
        max_portfolio_delta=0.30,
        max_portfolio_risk_pct=0.02,
    )


@pytest.fixture
def account_config() -> AccountConfig:
    return AccountConfig(pdt_protection=True, pdt_account_under_25k=True)


@pytest.fixture
def account_config_over_25k() -> AccountConfig:
    return AccountConfig(pdt_protection=True, pdt_account_under_25k=False)

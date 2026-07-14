"""Shared fixtures for AIT v2 tests."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """R12: marker registration. CANONICAL registration lives in pyproject
    [tool.pytest.ini_options] markers; this mirror only keeps local runs
    warning-clean if pyproject and tests land out of order (duplicate
    registration is harmless)."""
    config.addinivalue_line(
        "markers",
        "ibkr: live tests requiring IB Gateway on 127.0.0.1:4002 (run with -m ibkr)",
    )
    config.addinivalue_line(
        "markers",
        "slow: long-running suites (walkforward, optimizer, training); "
        "excluded from the default selection, run nightly with -m slow",
    )

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


@pytest.fixture(autouse=True)
def _isolate_model_artifacts(tmp_path, monkeypatch):
    """R12 (vol-craft audit): tests constructed predictors with the DEFAULT
    model dir — the LIVE models/ — and pytest runs overwrote models/range.pkl
    with a synthetic-noise QQQ model (caught 2026-07-13, file mtime 13:12,
    while the live gate depends on that artifact). Redirect every model save
    in every test to tmp_path. The live spec-mismatch guard remains the
    second net; this fixture is the fence."""
    import ait.ml.range_predictor as _rp
    import ait.ml.vol_magnitude_predictor as _vp
    import ait.ml.ensemble as _en
    for _mod in (_rp, _vp, _en):
        if hasattr(_mod, "MODEL_DIR"):
            monkeypatch.setattr(_mod, "MODEL_DIR", tmp_path, raising=False)

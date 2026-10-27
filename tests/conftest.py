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

@pytest.fixture(autouse=True)
def _neutralise_simulated_capital(monkeypatch):
    """W26: the suite must not inherit the operator's AIT_SIMULATED_CAPITAL.

    That env var (broker/account.py) SCALES every account balance so the risk
    stack sizes as if the account were smaller. It is a live operator setting
    read from the gitignored .env, and it silently rewrote risk-gate arithmetic
    underneath the tests: setting it to 10000 on 2026-09-17 turned the per-trade
    3% cap from $3,000 into $300 and broke two passing
    test_w2_orchestrator_gates tests that assert on a $380 max_loss — tests with
    nothing to do with simulated capital.

    Fixing the CLASS rather than those two tests: no test should depend on an
    operator's local .env value, and the alternative (every account-sensitive
    test pinning its own NLV) is the same bug waiting for the next env var. This
    is the same isolation family as the known-failing
    test_r19_config_authority::test_applier_and_readers_share_one_table, which
    has failed on this box since W16 for exactly this reason.

    No test sets or depends on this var (verified), so unsetting it is safe. A
    test that ever WANTS it can monkeypatch it back inside its own body.
    """
    # W-capital: simulation is now ALSO driven by config.yaml account.capital
    # (simulate_live_capital: true), so merely UNSETTING the var would let the
    # yaml turn it back on and re-scale every balance. The env override's "0"
    # means OFF in runtime_env.simulated_capital() -- same isolation, closed
    # against the new source too.
    monkeypatch.setenv("AIT_SIMULATED_CAPITAL", "0")


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
    # R15 #7: meta_label was MISSING from this fence — its own CWD-relative
    # MODEL_DIR meant every pytest run on the trading box overwrote the LIVE
    # models/meta_label.pkl with a test-fixture model. Enumerate by package
    # scan would be better still; at minimum every module with a MODEL_DIR
    # must appear here.
    import ait.ml.meta_label as _ml
    for _mod in (_rp, _vp, _en, _ml):
        if hasattr(_mod, "MODEL_DIR"):
            monkeypatch.setattr(_mod, "MODEL_DIR", tmp_path, raising=False)


@pytest.fixture(autouse=True)
def _fence_supervisor_page_stamps(tmp_path, monkeypatch):
    """W34: the supervisor's durable page throttle is a stamp FILE (the process
    that pages then exits, so a module dict cannot remember). The first W34 run
    left logs/.ait_page_singleton_lock_unavailable.stamp on the live box — for
    the next 30 minutes a REAL singleton-lock failure would not have paged.
    Same class as the model-artifact fence above: this suite runs on the
    trading host, so every test gets its stamps in tmp_path. (The other W34
    hazard, a test reaching the real process table and killing the live
    Gateway, is fenced in production code: master._kill_pid refuses under
    pytest.)"""
    import ait.orchestration.master as _m
    monkeypatch.setattr(_m, "_page_stamp_dirs", lambda: [tmp_path], raising=False)
    # W34b: the recovery launcher now keeps its output under LOGS_DIR. Older
    # supervisor tests drive _maybe_recover_gateway with a mocked Popen and the
    # REAL LOGS_DIR - each run would leave gateway_launch_*.log files in the
    # live box's logs/. Same fence, same reason.
    monkeypatch.setattr(_m, "LOGS_DIR", tmp_path, raising=False)


@pytest.fixture(autouse=True)
def _never_leak_frozen_settings():
    """W35: freeze_settings() makes one configuration permanent for a PROCESS —
    correct for the bot and the supervisor, poison for a test session, where
    the next test would silently read the previous test's config. Any test
    that reaches an entry point in-process is cleaned up after."""
    import ait.config.settings as _s
    yield
    _s._FROZEN = None

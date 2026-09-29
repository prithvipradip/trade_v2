"""Tests for data quality validation."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from ait.config.settings import ExitConfig
from ait.data.quality import DataQualityValidator
from ait.execution.portfolio import PortfolioManager


@pytest.fixture
def validator():
    return DataQualityValidator(
        max_staleness_seconds=30.0,
        max_spread_pct=0.15,
        max_price_jump_pct=0.10,
    )


class TestQuoteValidation:
    """Test market quote validation."""

    def test_valid_quote(self, validator):
        q = validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00)
        assert q.is_valid
        assert q.issues == []

    def test_zero_price_invalid(self, validator):
        q = validator.validate_quote("AAPL", bid=0, ask=0, last=0)
        assert not q.is_valid
        assert any("zero" in i for i in q.issues)

    def test_crossed_market(self, validator):
        q = validator.validate_quote("AAPL", bid=151.00, ask=149.00, last=150.00)
        assert not q.is_valid
        assert any("crossed" in i for i in q.issues)

    def test_wide_spread(self, validator):
        q = validator.validate_quote("AAPL", bid=130.00, ask=170.00, last=150.00)
        assert not q.is_valid
        assert any("wide spread" in i for i in q.issues)

    def test_stale_quote(self, validator):
        old_ts = time.time() - 60  # 60 seconds ago
        q = validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00, timestamp=old_ts)
        assert not q.is_valid
        assert any("stale" in i for i in q.issues)

    def test_price_jump_detection(self, validator):
        # First quote establishes baseline
        validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00)

        # Second quote with 20% jump
        q = validator.validate_quote("AAPL", bid=179.90, ask=180.10, last=180.00)
        assert not q.is_valid
        assert any("price jump" in i for i in q.issues)

    def test_small_price_move_ok(self, validator):
        validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00)
        q = validator.validate_quote("AAPL", bid=150.90, ask=151.10, last=151.00)
        assert q.is_valid

    def test_negative_bid(self, validator):
        q = validator.validate_quote("AAPL", bid=-1.0, ask=150.10, last=150.00)
        assert not q.is_valid


class TestOptionQuoteValidation:
    """Test options-specific quote validation."""

    def test_valid_option_quote(self, validator):
        q = validator.validate_option_quote(
            "AAPL_C150", bid=2.00, ask=2.20, last=2.10,
            open_interest=500, volume=100, delta=0.45,
        )
        assert q.is_valid

    def test_low_open_interest(self, validator):
        q = validator.validate_option_quote(
            "AAPL_C150", bid=2.00, ask=2.20, last=2.10,
            open_interest=5, volume=100, delta=0.45,
        )
        assert not q.is_valid

    def test_invalid_delta(self, validator):
        q = validator.validate_option_quote(
            "AAPL_C150", bid=2.00, ask=2.20, last=2.10,
            open_interest=500, volume=100, delta=1.5,
        )
        assert not q.is_valid

    def test_zero_bid_no_market(self, validator):
        q = validator.validate_option_quote(
            "AAPL_C150", bid=0.0, ask=2.20, last=2.10,
            open_interest=500, volume=100,
        )
        assert not q.is_valid


class TestHistoricalValidation:
    """Test historical data validation."""

    def test_valid_history(self, validator):
        prices = [150.0 + i * 0.5 for i in range(30)]
        assert validator.validate_historical("AAPL", prices, min_points=20)

    def test_insufficient_points(self, validator):
        prices = [150.0, 151.0, 152.0]
        assert not validator.validate_historical("AAPL", prices, min_points=20)

    def test_too_many_zeros(self, validator):
        prices = [0.0] * 25 + [150.0] * 5
        assert not validator.validate_historical("AAPL", prices, min_points=20)

    def test_flat_prices(self, validator):
        prices = [150.0] * 30
        assert not validator.validate_historical("AAPL", prices, min_points=20)


class TestReset:
    """Test tracking reset."""

    def test_reset_symbol(self, validator):
        validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00)
        validator.reset_tracking("AAPL")
        # After reset, a big jump shouldn't be flagged
        q = validator.validate_quote("AAPL", bid=179.90, ask=180.10, last=180.00)
        assert q.is_valid

    def test_reset_all(self, validator):
        validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00)
        validator.validate_quote("MSFT", bid=399.90, ask=400.10, last=400.00)
        validator.reset_tracking()
        q = validator.validate_quote("AAPL", bid=200.00, ask=200.20, last=200.10)
        assert q.is_valid

    def test_reset_clears_pending_too(self, validator):
        """reset_tracking must also drop a not-yet-promoted pending price --
        otherwise a stale pending value from before the reset could agree
        with (and wrongly promote) an unrelated later jump."""
        validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00)
        validator.validate_quote("AAPL", bid=179.90, ask=180.10, last=180.00)  # jump -> pending=180
        assert validator._pending_prices.get("AAPL") == 180.00
        validator.reset_tracking("AAPL")
        assert "AAPL" not in validator._pending_prices
        assert "AAPL" not in validator._last_prices


class TestOutlierAnchorNotPoisoned:
    """qual-01 (R25, 2026-09-14): the anchor used to update unconditionally,
    even on a rejected print, so a single bad tick poisoned the very
    baseline the next comparison used. Real call site:
    execution/portfolio.py's _spot_quote, which feeds the touch stop's
    spot_health classification."""

    def test_single_bad_print_does_not_poison_the_anchor(self, validator):
        """A lone bad print must not corrupt the baseline for the NEXT,
        genuinely-recovered quote."""
        validator.validate_quote("AAPL", bid=99.90, ask=100.10, last=100.00)  # anchor=100
        bad = validator.validate_quote("AAPL", bid=87.90, ask=88.10, last=88.00)  # single bad print
        assert not bad.is_valid
        assert any("price jump" in i for i in bad.issues)

        # Real recovery back to the true anchor. At HEAD (pre-fix) this would
        # compare against the POISONED anchor (88), reading a spurious
        # ~13.6% jump and staying degraded -- exactly the bug.
        recovered = validator.validate_quote("AAPL", bid=99.90, ask=100.10, last=100.00)
        assert recovered.is_valid, (
            f"anchor was poisoned by the rejected print: {recovered.issues}"
        )

    def test_two_agreeing_jumps_reanchor_a_genuine_move(self, validator):
        """A genuine, sustained move must still be accepted as the new
        baseline -- otherwise "never update on an invalid quote" would leave
        the symbol permanently degraded after any real >10% move."""
        validator.validate_quote("AAPL", bid=99.90, ask=100.10, last=100.00)  # anchor=100

        first = validator.validate_quote("AAPL", bid=84.90, ask=85.10, last=85.00)  # -15%
        assert not first.is_valid  # jump vs the still-100 anchor

        second = validator.validate_quote("AAPL", bid=84.40, ask=84.60, last=84.50)  # agrees with 85
        assert not second.is_valid  # still a jump vs the OLD (100) anchor this call
        assert validator._last_prices["AAPL"] == pytest.approx(84.50), (
            "two agreeing prints should have promoted the new baseline"
        )

        third = validator.validate_quote("AAPL", bid=84.65, ask=84.85, last=84.75)  # near new anchor
        assert third.is_valid, "the re-anchored baseline should now read fresh"

    def test_disagreeing_second_print_does_not_falsely_promote(self, validator):
        """Two DIFFERENT bad prints in a row must not agree with each other
        by accident and promote a still-wrong anchor."""
        validator.validate_quote("AAPL", bid=99.90, ask=100.10, last=100.00)  # anchor=100
        validator.validate_quote("AAPL", bid=87.90, ask=88.10, last=88.00)  # pending=88
        validator.validate_quote("AAPL", bid=69.90, ask=70.10, last=70.00)  # unrelated jump, pending=70

        # Anchor must still be the ORIGINAL 100 -- neither bad print agreed
        # with the other, so nothing should have promoted.
        assert validator._last_prices["AAPL"] == pytest.approx(100.00)
        recovered = validator.validate_quote("AAPL", bid=99.90, ask=100.10, last=100.00)
        assert recovered.is_valid


class TestTimestampSentinelHandling:
    """qual-02 (R25, 2026-09-14): `timestamp or now` used truthiness, so an
    explicit epoch-0 (a real "no data" sentinel) silently became `now` and
    defeated the staleness check it exists to run."""

    def test_epoch_zero_timestamp_is_treated_as_stale(self, validator):
        q = validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00, timestamp=0)
        assert not q.is_valid
        assert any("stale" in i for i in q.issues)

    def test_epoch_zero_float_timestamp_is_treated_as_stale(self, validator):
        q = validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00, timestamp=0.0)
        assert not q.is_valid
        assert any("stale" in i for i in q.issues)

    def test_none_timestamp_still_means_fresh_now(self, validator):
        """The one legitimate default case (caller passed nothing) must be
        unchanged -- this is not the bug."""
        q = validator.validate_quote("AAPL", bid=149.90, ask=150.10, last=150.00, timestamp=None)
        assert q.is_valid

    def test_nan_timestamp_is_treated_as_stale_not_silently_passed(self, validator):
        q = validator.validate_quote(
            "AAPL", bid=149.90, ask=150.10, last=150.00, timestamp=float("nan"))
        assert not q.is_valid
        assert any("stale" in i for i in q.issues)


class TestSpreadAndJumpThresholdsAreConfigBacked:
    """qual-04 (R25, 2026-09-14): max_spread_pct/max_price_jump_pct were
    DataQualityValidator's own class defaults, never overridden by either
    live construction site in execution/portfolio.py. Drives the REAL
    PortfolioManager.__init__ (not __new__ + manual attrs) so this is an
    honest proof of wiring, not of the validator's own defaults."""

    def _build(self, exit_config: ExitConfig) -> PortfolioManager:
        return PortfolioManager(
            ibkr_client=MagicMock(),
            market_data=MagicMock(),
            state=MagicMock(),
            circuit_breaker=MagicMock(),
            pdt_guard=MagicMock(),
            exit_config=exit_config,
        )

    def test_init_wires_the_configured_thresholds(self):
        # Teeth check: both configured values must differ from
        # DataQualityValidator's own class defaults (quality.py:40-41,
        # 0.15 / 0.10), or this test cannot tell wired from unwired.
        cfg = ExitConfig(max_quote_spread_pct=0.22, max_quote_jump_pct=0.07)
        assert cfg.max_quote_spread_pct != 0.15
        assert cfg.max_quote_jump_pct != 0.10
        mgr = self._build(cfg)
        assert mgr._quality._max_spread_pct == pytest.approx(0.22)
        assert mgr._quality._max_price_jump == pytest.approx(0.07)

    def test_lazy_fallback_also_wires_the_configured_thresholds(self):
        """The lazy _quality_validator() path (used by every test that
        builds PortfolioManager via __new__, skipping __init__) must wire
        the same way as __init__ -- both construction sites had the bug."""
        cfg = ExitConfig(max_quote_spread_pct=0.31, max_quote_jump_pct=0.06)
        mgr = PortfolioManager.__new__(PortfolioManager)
        mgr._exit_config = cfg
        validator = mgr._quality_validator()
        assert validator._max_spread_pct == pytest.approx(0.31)
        assert validator._max_price_jump == pytest.approx(0.06)

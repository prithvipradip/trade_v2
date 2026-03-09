"""Tests for data quality validation."""

from __future__ import annotations

import time

import pytest

from ait.data.quality import DataQualityValidator


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

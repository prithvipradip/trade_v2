"""Tests for CircuitBreaker risk management component."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from ait.config.settings import RiskConfig
from ait.risk.circuit_breaker import CircuitBreaker, CircuitBreakerStatus


@pytest.fixture
def cb(risk_config: RiskConfig) -> CircuitBreaker:
    return CircuitBreaker(risk_config)


class TestDailyLossLimit:
    """Test daily loss limit triggers halt."""

    def test_daily_loss_trips_breaker(self, cb: CircuitBreaker) -> None:
        """Recording losses that exceed max_daily_loss_pct should trip."""
        # Account = $100k, max_daily_loss_pct = 0.02 => $2000 limit
        cb.record_trade_result(-2100.0)
        can_continue = cb.check_daily_loss(account_value=100_000.0)

        assert can_continue is False
        assert cb.is_tripped is True

    def test_daily_loss_at_threshold(self, cb: CircuitBreaker) -> None:
        """Loss exactly at threshold should trip."""
        cb.record_trade_result(-2000.0)
        can_continue = cb.check_daily_loss(account_value=100_000.0)

        assert can_continue is False
        assert cb.is_tripped is True

    def test_below_threshold_does_not_trip(self, cb: CircuitBreaker) -> None:
        """Loss below threshold should NOT trip."""
        cb.record_trade_result(-1000.0)
        can_continue = cb.check_daily_loss(account_value=100_000.0)

        assert can_continue is True
        assert cb.is_tripped is False

    def test_daily_loss_has_no_auto_resume(self, cb: CircuitBreaker) -> None:
        """Daily loss trip should not auto-resume (resume_time == 0)."""
        cb.record_trade_result(-3000.0)
        cb.check_daily_loss(account_value=100_000.0)

        status = cb.get_status()
        assert status.tripped is True
        assert status.resume_time == 0.0
        assert "daily_loss" in status.reason

    def test_winning_trades_offset_losses(self, cb: CircuitBreaker) -> None:
        """Net P&L considers wins and losses together."""
        cb.record_trade_result(-3000.0)
        cb.record_trade_result(2500.0)
        # Net = -500, which is 0.5% of 100k => below 2% threshold
        can_continue = cb.check_daily_loss(account_value=100_000.0)

        assert can_continue is True
        assert cb.is_tripped is False

    def test_zero_account_value_returns_true(self, cb: CircuitBreaker) -> None:
        """Zero or negative account value should not trip (safety guard)."""
        cb.record_trade_result(-5000.0)
        assert cb.check_daily_loss(account_value=0.0) is True
        assert cb.check_daily_loss(account_value=-100.0) is True


class TestConsecutiveLosses:
    """Test consecutive losses trigger pause (with auto-resume)."""

    def test_consecutive_losses_trip(self, cb: CircuitBreaker) -> None:
        """3 consecutive losses (the configured max) should trip."""
        cb.record_trade_result(-100.0)
        cb.record_trade_result(-200.0)
        cb.record_trade_result(-150.0)

        assert cb.is_tripped is True

    def test_win_resets_streak(self, cb: CircuitBreaker) -> None:
        """A winning trade should reset the consecutive loss counter."""
        cb.record_trade_result(-100.0)
        cb.record_trade_result(-200.0)
        cb.record_trade_result(50.0)  # Win resets streak
        cb.record_trade_result(-100.0)

        assert cb.is_tripped is False

    def test_consecutive_losses_auto_resume(self, cb: CircuitBreaker) -> None:
        """Consecutive loss trip should auto-resume after pause_minutes."""
        with patch("ait.risk.circuit_breaker.time") as mock_time:
            now = 1000000.0
            mock_time.time.return_value = now

            cb.record_trade_result(-100.0)
            cb.record_trade_result(-200.0)
            cb.record_trade_result(-150.0)

            assert cb.is_tripped is True

            # Advance time past the pause (30 min = 1800s)
            mock_time.time.return_value = now + 1801.0
            assert cb.is_tripped is False

    def test_consecutive_losses_still_tripped_before_resume(
        self, cb: CircuitBreaker
    ) -> None:
        """Should remain tripped before the pause elapses."""
        with patch("ait.risk.circuit_breaker.time") as mock_time:
            now = 1000000.0
            mock_time.time.return_value = now

            cb.record_trade_result(-100.0)
            cb.record_trade_result(-200.0)
            cb.record_trade_result(-150.0)

            # Only 10 minutes later
            mock_time.time.return_value = now + 600.0
            assert cb.is_tripped is True

    def test_status_shows_consecutive_losses(self, cb: CircuitBreaker) -> None:
        """Status should reflect the count of consecutive losses."""
        cb.record_trade_result(-100.0)
        cb.record_trade_result(-50.0)

        status = cb.get_status()
        assert status.consecutive_losses == 2
        assert status.tripped is False


class TestAPIFailures:
    """Test API failure counting in 10-min window."""

    def test_api_failures_trip_at_threshold(self, cb: CircuitBreaker) -> None:
        """5 API failures (configured max) in 10 min should trip."""
        with patch("ait.risk.circuit_breaker.time") as mock_time:
            now = 1000000.0
            mock_time.time.return_value = now

            for _ in range(5):
                cb.record_api_failure()

            assert cb.is_tripped is True

    def test_api_failures_below_threshold(self, cb: CircuitBreaker) -> None:
        """4 failures (below max of 5) should NOT trip."""
        with patch("ait.risk.circuit_breaker.time") as mock_time:
            mock_time.time.return_value = 1000000.0
            for _ in range(4):
                cb.record_api_failure()

            assert cb.is_tripped is False

    def test_old_api_failures_expire(self, cb: CircuitBreaker) -> None:
        """Failures older than 10 minutes should not count."""
        with patch("ait.risk.circuit_breaker.time") as mock_time:
            # Record 3 failures at t=0
            mock_time.time.return_value = 1000000.0
            for _ in range(3):
                cb.record_api_failure()

            # Advance past 10-min window
            mock_time.time.return_value = 1000000.0 + 601.0

            # Record 2 more failures (total recent = 2, not 5)
            for _ in range(2):
                cb.record_api_failure()

            assert cb.is_tripped is False

    def test_api_failure_trip_has_10min_resume(self, cb: CircuitBreaker) -> None:
        """API failure trip should auto-resume after 10 minutes."""
        with patch("ait.risk.circuit_breaker.time") as mock_time:
            now = 1000000.0
            mock_time.time.return_value = now

            for _ in range(5):
                cb.record_api_failure()

            assert cb.is_tripped is True

            # After 10 minutes, should auto-resume
            mock_time.time.return_value = now + 601.0
            assert cb.is_tripped is False

    def test_api_success_cleans_old_failures(self, cb: CircuitBreaker) -> None:
        """record_api_success should prune old failure timestamps."""
        with patch("ait.risk.circuit_breaker.time") as mock_time:
            mock_time.time.return_value = 1000000.0
            for _ in range(3):
                cb.record_api_failure()

            # Advance past window and call success
            mock_time.time.return_value = 1000000.0 + 700.0
            cb.record_api_success()

            # Internal list should be empty now
            assert len(cb._api_failures) == 0

    def test_status_shows_api_failure_count(self, cb: CircuitBreaker) -> None:
        """get_status should report recent API failure count."""
        with patch("ait.risk.circuit_breaker.time") as mock_time:
            mock_time.time.return_value = 1000000.0
            cb.record_api_failure()
            cb.record_api_failure()

            status = cb.get_status()
            assert status.api_failures == 2


class TestDailyReset:
    """Test daily reset clears counters."""

    def test_daily_reset_clears_pnl_and_losses(self, cb: CircuitBreaker) -> None:
        """New day should reset daily P&L and consecutive losses."""
        cb.record_trade_result(-500.0)
        cb.record_trade_result(-200.0)

        from datetime import date, timedelta

        tomorrow = date.today() + timedelta(days=1)
        with patch("ait.risk.circuit_breaker.date") as mock_date:
            mock_date.today.return_value = tomorrow
            cb.check_daily_reset()

        status = cb.get_status()
        assert status.daily_pnl == 0.0
        assert status.consecutive_losses == 0

    def test_daily_reset_untrips_daily_loss_breaker(
        self, cb: CircuitBreaker
    ) -> None:
        """Daily loss trip should auto-clear on new day."""
        cb.record_trade_result(-3000.0)
        cb.check_daily_loss(account_value=100_000.0)
        assert cb.is_tripped is True

        from datetime import date, timedelta

        tomorrow = date.today() + timedelta(days=1)
        with patch("ait.risk.circuit_breaker.date") as mock_date:
            mock_date.today.return_value = tomorrow
            cb.check_daily_reset()

        assert cb.is_tripped is False

    def test_daily_reset_does_not_clear_non_daily_trip(
        self, cb: CircuitBreaker
    ) -> None:
        """Non-daily-loss trips should NOT be cleared by daily reset."""
        # Trip via consecutive losses
        cb.record_trade_result(-100.0)
        cb.record_trade_result(-100.0)
        cb.record_trade_result(-100.0)
        assert cb.is_tripped is True

        from datetime import date, timedelta

        tomorrow = date.today() + timedelta(days=1)
        with patch("ait.risk.circuit_breaker.time") as mock_time:
            # Keep it from auto-resuming
            mock_time.time.return_value = time.time()
            with patch("ait.risk.circuit_breaker.date") as mock_date:
                mock_date.today.return_value = tomorrow
                cb.check_daily_reset()

            # Still tripped because consecutive_losses reason != daily_loss
            assert cb.is_tripped is True

    def test_no_reset_same_day(self, cb: CircuitBreaker) -> None:
        """check_daily_reset should do nothing if called on the same day."""
        cb.record_trade_result(-500.0)
        cb.check_daily_reset()

        status = cb.get_status()
        assert status.daily_pnl == -500.0


class TestManualReset:
    """Test manual reset."""

    def test_manual_reset_clears_trip(self, cb: CircuitBreaker) -> None:
        """manual_reset should clear a tripped breaker."""
        cb.record_trade_result(-3000.0)
        cb.check_daily_loss(account_value=100_000.0)
        assert cb.is_tripped is True

        cb.manual_reset()

        assert cb.is_tripped is False
        status = cb.get_status()
        assert status.reason == ""
        assert status.consecutive_losses == 0

    def test_manual_reset_when_not_tripped(self, cb: CircuitBreaker) -> None:
        """manual_reset on a non-tripped breaker should be harmless."""
        cb.manual_reset()
        assert cb.is_tripped is False

    def test_manual_reset_preserves_daily_pnl(self, cb: CircuitBreaker) -> None:
        """manual_reset should NOT clear daily P&L (only daily_reset does)."""
        cb.record_trade_result(-1000.0)
        cb.manual_reset()

        status = cb.get_status()
        assert status.daily_pnl == -1000.0

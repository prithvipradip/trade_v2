"""Circuit breaker — automatic trading halt on adverse conditions.

Prevents catastrophic losses by stopping all trading when:
- Daily loss exceeds configured maximum
- Too many consecutive losing trades
- Too many API/system failures
- Portfolio risk exceeds limits
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date

from ait.config.settings import RiskConfig
from ait.utils.logging import get_logger

log = get_logger("risk.circuit_breaker")


@dataclass
class CircuitBreakerStatus:
    """Current circuit breaker state."""

    tripped: bool
    reason: str
    resume_time: float  # Unix timestamp when trading can resume (0 = manual reset)
    daily_pnl: float
    consecutive_losses: int
    api_failures: int


class CircuitBreaker:
    """Monitors trading conditions and halts on adverse events."""

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._tripped = False
        self._trip_reason = ""
        self._resume_time = 0.0

        # Tracking
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._api_failures: list[float] = []  # Timestamps of recent failures
        self._last_reset_date = date.today()

    @property
    def is_tripped(self) -> bool:
        """Check if circuit breaker is currently active."""
        if not self._tripped:
            return False

        # Check if pause period has elapsed
        if self._resume_time > 0 and time.time() >= self._resume_time:
            log.info("circuit_breaker_auto_resumed", reason=self._trip_reason)
            self._tripped = False
            self._trip_reason = ""
            self._resume_time = 0.0
            return False

        return True

    def get_status(self) -> CircuitBreakerStatus:
        return CircuitBreakerStatus(
            tripped=self.is_tripped,
            reason=self._trip_reason,
            resume_time=self._resume_time,
            daily_pnl=self._daily_pnl,
            consecutive_losses=self._consecutive_losses,
            api_failures=len(self._recent_api_failures()),
        )

    def check_daily_reset(self) -> None:
        """Reset daily counters if it's a new day."""
        today = date.today()
        if today != self._last_reset_date:
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
            self._api_failures.clear()
            self._last_reset_date = today

            # Auto-reset daily loss circuit breaker
            if self._tripped and "daily_loss" in self._trip_reason:
                self._tripped = False
                self._trip_reason = ""
                log.info("circuit_breaker_daily_reset")

    def record_trade_result(self, pnl: float) -> None:
        """Record a trade's P&L and check for circuit breaker triggers."""
        self._daily_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Check consecutive losses
        if self._consecutive_losses >= self._config.max_consecutive_losses:
            pause_seconds = self._config.pause_minutes_after_losses * 60
            self._trip(
                f"consecutive_losses ({self._consecutive_losses})",
                pause_seconds=pause_seconds,
            )

    def check_daily_loss(self, account_value: float) -> bool:
        """Check if daily loss limit has been exceeded.

        Returns True if trading can continue, False if halted.
        """
        if account_value <= 0:
            return True

        loss_pct = abs(self._daily_pnl) / account_value
        if self._daily_pnl < 0 and loss_pct >= self._config.max_daily_loss_pct:
            self._trip(
                f"daily_loss ({loss_pct:.1%} of ${account_value:.0f})",
                pause_seconds=0,  # No auto-resume for daily loss
            )
            return False
        return True

    def record_api_failure(self) -> None:
        """Record an API failure and check if threshold exceeded."""
        self._api_failures.append(time.time())

        recent = self._recent_api_failures()
        if len(recent) >= self._config.max_api_failures:
            self._trip(
                f"api_failures ({len(recent)} in 10 min)",
                pause_seconds=600,  # 10-minute pause
            )

    def record_api_success(self) -> None:
        """Record a successful API call — clears failure count."""
        # Keep only recent failures for windowed tracking
        self._api_failures = self._recent_api_failures()

    def manual_reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._tripped = False
        self._trip_reason = ""
        self._resume_time = 0.0
        self._consecutive_losses = 0
        log.info("circuit_breaker_manual_reset")

    # --- Private ---

    def _trip(self, reason: str, pause_seconds: int = 0) -> None:
        """Trip the circuit breaker."""
        self._tripped = True
        self._trip_reason = reason
        self._resume_time = (time.time() + pause_seconds) if pause_seconds > 0 else 0.0

        log.critical(
            "circuit_breaker_tripped",
            reason=reason,
            daily_pnl=self._daily_pnl,
            consecutive_losses=self._consecutive_losses,
            auto_resume="never" if pause_seconds == 0 else f"{pause_seconds}s",
        )

    def _recent_api_failures(self) -> list[float]:
        """Get API failures in the last 10 minutes."""
        cutoff = time.time() - 600
        return [t for t in self._api_failures if t > cutoff]

"""Circuit breaker — automatic trading halt on adverse conditions.

Prevents catastrophic losses by stopping all trading when:
- Daily loss exceeds configured maximum
- Too many consecutive losing trades
- Too many API/system failures
- Portfolio risk exceeds limits
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from ait.utils.time import now_et

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

    # R16: bot_state KV holding everything that must survive a process death
    # (see _load_persisted). One blob, one write — the breaker changes rarely.
    STATE_KEY = "circuit_breaker_state"

    # R16: the token every MTM daily-loss trip reason carries. Two rules read
    # it: check_daily_reset's untrip matcher ("daily_loss") and
    # check_daily_loss_mtm's "is the ACTIVE trip mine?" test.
    MTM_TRIP_TOKEN = "daily_loss (MTM)"

    def __init__(self, config: RiskConfig, state=None) -> None:
        self._config = config
        self._tripped = False
        self._trip_reason = ""
        self._resume_time = 0.0

        # Tracking
        self._daily_pnl = 0.0
        self._consecutive_losses = 0
        self._api_failures: list[float] = []  # Timestamps of recent failures
        self._last_reset_date = now_et().date()

        # R16: optional StateManager for restart persistence. Optional so every
        # existing construction site (orchestrator.py:85, tests) keeps working;
        # attach_state() is the seam the RiskManager uses for the live wiring.
        self._state = None
        if state is not None:
            self.attach_state(state)

    # --- Restart persistence (R16) ---

    def attach_state(self, state) -> None:
        """Wire DB-backed persistence and reload what survived the restart.

        R16: _daily_pnl, _consecutive_losses and the tripped flag were
        process-memory ONLY, and the keeper relaunches this bot after a crash
        (~30-60 min python-kill cycle documented in keeper_ait.bat). Every
        relaunch zeroed the loss streak and silently cleared an active
        consecutive-losses/api-failure pause: three stop-outs, crash, relaunch
        -> counter 0, next entry allowed immediately. The daily-loss halt was
        incidentally restart-resilient (the MTM variant recomputes from
        DB-backed daily_stats within 30s), the 3-loss pause was not.

        Separate from __init__ because the orchestrator builds the breaker
        before it hands a StateManager to anything (orchestrator.py:85 passes
        settings.risk alone); RiskManager.__init__ holds both objects and
        performs the attach.
        """
        self._state = state
        self._load_persisted()

    def _load_persisted(self) -> None:
        """Restore the breaker from bot_state. Never raises: a breaker that
        cannot read its history must still run, just without the history."""
        state = getattr(self, "_state", None)
        if state is None:
            return
        try:
            raw = state.get_state(self.STATE_KEY, "")
            blob = json.loads(raw) if raw else None
        except Exception as e:  # noqa: BLE001
            # W1 (R23 fail-direction-10): an UNREADABLE store used to fall
            # through to a fresh zero state — a relaunch during a brief DB
            # lock silently cleared an active 3-loss pause and the loss
            # streak, re-entering immediately into the same losing
            # conditions. Fail CLOSED: pause for one configured window so a
            # possibly-active halt is honored; a healthy read next restart
            # (or the resume timer) lifts it.
            log.error("circuit_breaker_state_unreadable_fail_closed",
                      error=str(e))
            self._tripped = True
            self._trip_reason = "state_unreadable_fail_closed"
            self._resume_time = time.time() + max(
                60.0, float(getattr(self._config,
                                    "pause_minutes_after_losses", 30)) * 60.0)
            return
        if not isinstance(blob, dict):
            return  # empty/first-run store: genuinely fresh, no fail-close
        try:
            self._consecutive_losses = int(blob.get("consecutive_losses", 0) or 0)
            self._daily_pnl = float(blob.get("daily_pnl", 0.0) or 0.0)
            self._tripped = bool(blob.get("tripped", False))
            self._trip_reason = str(blob.get("reason", "") or "")
            self._resume_time = float(blob.get("resume_time", 0.0) or 0.0)
            stored_date = str(blob.get("date", "") or "")
            if stored_date:
                self._last_reset_date = date.fromisoformat(stored_date)
        except (TypeError, ValueError) as e:
            log.warning("circuit_breaker_state_corrupt", error=str(e))
            return
        # A restored blob from an earlier session may predate today: run the
        # SAME rollover the live loop runs rather than duplicating its rules.
        self.check_daily_reset()
        if self._tripped or self._consecutive_losses:
            log.warning(
                "circuit_breaker_state_restored",
                tripped=self._tripped,
                reason=self._trip_reason,
                consecutive_losses=self._consecutive_losses,
                daily_pnl=round(self._daily_pnl, 2),
                resume_in_s=(max(0, int(self._resume_time - time.time()))
                             if self._resume_time > 0 else 0),
            )

    def _persist(self) -> None:
        """Write the survivable state. Best-effort: a DB hiccup must never
        propagate into the trading loop from a protection component."""
        state = getattr(self, "_state", None)
        if state is None:
            return
        try:
            state.set_state(self.STATE_KEY, json.dumps({
                "date": self._last_reset_date.isoformat(),
                "consecutive_losses": self._consecutive_losses,
                "daily_pnl": round(float(self._daily_pnl), 4),
                "tripped": bool(self._tripped),
                "reason": self._trip_reason,
                "resume_time": float(self._resume_time),
            }))
        except Exception as e:  # noqa: BLE001
            log.warning("circuit_breaker_state_persist_failed", error=str(e))

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
            # Deep-audit SR-L10: without this, the counter stayed at the
            # threshold and a single loss after auto-resume instantly
            # re-tripped — inconsistent with manual/daily resets.
            self._consecutive_losses = 0
            self._persist()  # R16: the resume must survive a restart too
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
        today = now_et().date()  # ET-pinned (deep-audit SR-M5)
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
            self._persist()  # R16

    def record_partial_pnl(self, pnl: float) -> None:
        """Fold partial-exit P&L into the daily total WITHOUT touching the
        consecutive-loss counter (a scale-out is not a completed trade).
        Deep-audit BC-H3: partial P&L previously never reached the breaker,
        so the daily-loss halt undercounted realized losses."""
        self._daily_pnl += pnl
        self._persist()  # R16

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
        else:
            # R16: _trip persists; a non-tripping result still moved the
            # streak and the daily total, so it must be written too.
            self._persist()

    def check_daily_loss_mtm(self, mtm_day_pnl: float, account_value: float) -> bool:
        """Mark-to-market daily-loss halt (A1, deep-audit BC-M4/R2.8).

        The realized-only check below is also ENTRY-GATED (evaluated only
        inside validate_trade), so on a gap day with no new entries the 2%
        halt was never even looked at while open positions bled unrealized.
        This variant is called from the 30s monitor with
        mtm_day_pnl = realized_today + (unrealized_now - unrealized_at_SOD)
        and trips the same breaker (blocks NEW entries; exits keep working).

        The return value answers ONE question — "is an MTM daily-loss halt in
        force?" — because that is the only question its caller asks. The
        orchestrator's fast monitor (orchestrator.py:599-607) turns True into
        "DAILY MTM LOSS HALT: day P&L $X ... breached the daily-loss cap".
        """
        # R16: this used to `return self._tripped` for ANY active trip, so a
        # consecutive-losses or api-failures pause made the monitor page
        # "DAILY MTM LOSS HALT ... breached the daily-loss cap" with the day's
        # P&L POSITIVE. Worse, the caller's _mtm_halt_alerted latch is set
        # once and never cleared, so that one false page permanently silenced
        # the notification for a LATER genuine MTM halt. Answer only for this
        # rule's own trip; halting itself is unaffected (is_tripped still
        # blocks entries, and the hourly "trading_halted" notice still fires).
        # A genuine MTM breach that arrives DURING an unrelated pause is not
        # lost: entries are already blocked, and the check re-evaluates and
        # trips for real as soon as that pause auto-resumes.
        if self._tripped:
            return self.MTM_TRIP_TOKEN in self._trip_reason
        if account_value <= 0:
            return False
        if mtm_day_pnl < 0 and abs(mtm_day_pnl) / account_value >= self._config.max_daily_loss_pct:
            self._trip(
                # R15 #3: MUST contain 'daily_loss' — check_daily_reset's
                # matcher untrips only reasons containing that token, and
                # "daily MTM loss" didn't match: one MTM trip blocked entries
                # FOREVER (pause_seconds=0 means no time-based resume either).
                # R16: built from MTM_TRIP_TOKEN so the reason string and the
                # "is this trip mine?" test above can never drift apart.
                f"{self.MTM_TRIP_TOKEN} {abs(mtm_day_pnl):.0f} >= "
                f"{self._config.max_daily_loss_pct:.0%} of {account_value:.0f}",
                pause_seconds=0,  # clears on daily reset, like the realized halt
            )
            return True
        return False

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
        self._persist()  # R16: a manual clear must not be undone by a restart
        log.info("circuit_breaker_manual_reset")

    # --- Private ---

    def _trip(self, reason: str, pause_seconds: int = 0) -> None:
        """Trip the circuit breaker."""
        self._tripped = True
        self._trip_reason = reason
        self._resume_time = (time.time() + pause_seconds) if pause_seconds > 0 else 0.0
        # R16: persist BEFORE logging — a trip that a keeper-kill erases is
        # the exact failure this fix exists for.
        self._persist()

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

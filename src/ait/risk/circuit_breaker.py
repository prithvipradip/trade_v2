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
    # W42 (R26 fail-direction-10 residual): False means the last persist
    # attempt failed — this process's state, possibly including an active
    # trip, is not on disk. No existing consumer pages on this yet; it is
    # exposed here so one can without CircuitBreaker growing its own notify
    # wiring.
    store_write_healthy: bool = True


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
        self._state_unreadable = False  # W14: set when a load fails
        # W42 (R26 fail-direction-10 residual): set when a WRITE fails — the
        # sibling of _state_unreadable, for the other direction. Distinct
        # because a write can fail while the store is still perfectly
        # readable (a momentary lock on this process's own attempt), so
        # _state_unreadable's re-read machinery (_maybe_reload) never
        # observes it and never clears it.
        self._persist_write_failed = False
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
            # W14 (R24 regress-live-04): the fail-closed trip auto-resumes
            # after one pause window, and the first _persist() after that
            # OVERWROTE the still-intact blob with tripped=false / streak 0 /
            # daily 0. A real 2-loss streak became 0 permanently and the
            # realized daily-loss gate read $0 for the rest of the day — the
            # fix for one fail-open created another. Remember that our
            # in-memory state is NOT authoritative until a successful read.
            self._state_unreadable = True
            self._tripped = True
            self._trip_reason = "state_unreadable_fail_closed"
            self._resume_time = time.time() + max(
                60.0, float(getattr(self._config,
                                    "pause_minutes_after_losses", 30)) * 60.0)
            return
        self._state_unreadable = False  # W14: a good read makes us authoritative
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
        if getattr(self, "_state_unreadable", False):
            # W14 (regress-live-04): we could not READ the persisted state, so
            # our in-memory counters are a fail-closed placeholder, not truth.
            # Writing them would destroy a real streak/daily-loss the next
            # process would otherwise recover. Stay read-only until a
            # successful load makes us authoritative again.
            log.warning("circuit_breaker_persist_skipped_unreadable_state")
            return
        try:
            state.set_state(self.STATE_KEY, self._blob())
            # W42 (R26 fail-direction-10 residual): clear on the NEXT success,
            # not immediately — _persist() runs again on its own the next time
            # any state-changing method calls it (record_trade_result, trip,
            # reset), so there is no separate retry queue to build; this just
            # stops the failure from being invisible in between.
            self._persist_write_failed = False
        except Exception as e:  # noqa: BLE001
            # W42: was log.warning with nothing else — a write failure here
            # means this process's OWN in-memory state (which may include an
            # active trip) is not on disk, so a crash before the next
            # successful persist loses it exactly like regress-live-04 did
            # before W14. Bumped to log.error and made queryable via
            # get_status().store_write_healthy, since nothing pages on it yet
            # (see PLAN.md — the paging half is a registered residual, not
            # claimed fixed here: this component has no notify_cb, unlike
            # PortfolioManager/AccountManager, and _persist() is sync).
            self._persist_write_failed = True
            log.error("circuit_breaker_state_persist_failed", error=str(e))

    def retry_persist(self) -> bool:
        """W104: write the state again and report whether the store is healthy.

        ``_persist`` only runs when a state-changing method calls it, so after a
        failed write the flag stayed up (and a trip stayed off the disk) until
        the next trade result. The orchestrator calls this each cycle while the
        store is unhealthy, so a transient lock heals itself and a persistent
        failure keeps paging.
        """
        self._persist()
        return not getattr(self, "_persist_write_failed", False)

    def _blob(self) -> str:
        return json.dumps({
            "date": self._last_reset_date.isoformat(),
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": round(float(self._daily_pnl), 4),
            "tripped": bool(self._tripped),
            "reason": self._trip_reason,
            "resume_time": float(self._resume_time),
        })

    # --- W32 (R26 two-phase-claim-double-books): transactional booking hooks.
    # The orchestrator applies a close to this breaker in memory, then writes
    # the blob in the SAME sqlite transaction as daily_stats and the outbox
    # marker. If that transaction fails the process lives on, so the in-memory
    # change must be undone or the retry counts the loss twice.
    _SNAP_FIELDS = ("_daily_pnl", "_consecutive_losses", "_tripped",
                    "_trip_reason", "_resume_time")

    def snapshot(self) -> dict:
        snap = {f: getattr(self, f) for f in self._SNAP_FIELDS}
        snap["_pending_results"] = list(getattr(self, "_pending_results", []))
        return snap

    def restore(self, snap: dict) -> None:
        for f, v in snap.items():
            setattr(self, f, list(v) if isinstance(v, list) else v)

    def persistable_blob(self) -> str | None:
        """The blob to persist, or None while our state is a fail-closed
        placeholder (W14: never write what we could not read)."""
        if getattr(self, "_state", None) is None:
            return None
        return None if getattr(self, "_state_unreadable", False) else self._blob()

    #: W32: how often a breaker that could not read its store tries again.
    _RELOAD_MIN_INTERVAL_S = 30.0

    def _maybe_reload(self) -> None:
        """W32 (R26 breaker-unreadable-never-reloads): keep trying to become
        authoritative again.

        W14 stopped the placeholder from OVERWRITING an intact blob, but
        nothing ever re-read it: after one transient lock at startup the
        breaker ran the rest of the process on zeros, is_tripped lifted the
        fail-closed pause after one window without looking, and persistence
        stayed off until the next restart. A daily-loss halt sitting intact
        on disk was simply not enforced. Results recorded while blind are
        queued and replayed onto the store's truth once a read succeeds.
        """
        if not getattr(self, "_state_unreadable", False):
            return
        now = time.time()
        if now - getattr(self, "_last_reload_attempt", 0.0) < self._RELOAD_MIN_INTERVAL_S:
            return
        self._last_reload_attempt = now
        pending = list(getattr(self, "_pending_results", []))
        self._load_persisted()          # a failure re-arms the fail-closed window
        if self._state_unreadable:
            return
        self._pending_results = []
        log.warning("circuit_breaker_state_reloaded", replayed=len(pending),
                    tripped=self._tripped, reason=self._trip_reason,
                    consecutive_losses=self._consecutive_losses,
                    daily_pnl=round(self._daily_pnl, 2))
        for pnl in pending:
            self.record_trade_result(pnl)

    @property
    def is_tripped(self) -> bool:
        """Check if circuit breaker is currently active."""
        if getattr(self, "_state_unreadable", False):
            self._maybe_reload()
            if self._state_unreadable:
                # W32: the pause window ending proves nothing about what the
                # store holds. Stay closed until a read succeeds.
                return True
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
            store_write_healthy=not getattr(self, "_persist_write_failed", False),
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

    def record_trade_result(self, pnl: float, persist: bool = True,
                            outcome_pnl: float | None = None) -> None:
        """Record a trade's P&L and check for circuit breaker triggers.

        persist=False (W32) leaves the write to the caller, which lands the
        blob in the same transaction as daily_stats — see persistable_blob().

        W50 (money-flow-05): `pnl` is the amount ADDED to `_daily_pnl` — for
        a trade with partial-exit history this is the remainder only, since
        the slices already moved `_daily_pnl` via record_partial_pnl when
        each one closed. `outcome_pnl` (defaults to `pnl`) is the trade's
        FULL round-trip P&L and decides the consecutive-loss counter — a
        trade that nets positive overall must reset the streak even when its
        remainder slice alone was a loss.
        """
        _outcome = pnl if outcome_pnl is None else outcome_pnl
        if getattr(self, "_state_unreadable", False):
            self._maybe_reload()
            if self._state_unreadable:
                # Still blind: count it in memory (3 fresh losses must still
                # pause) and remember it for replay onto the store's truth.
                self._pending_results = list(
                    getattr(self, "_pending_results", [])) + [pnl]
        self._daily_pnl += pnl

        if _outcome < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Check consecutive losses
        if self._consecutive_losses >= self._config.max_consecutive_losses:
            pause_seconds = self._config.pause_minutes_after_losses * 60
            self._trip(
                f"consecutive_losses ({self._consecutive_losses})",
                pause_seconds=pause_seconds, persist=persist,
            )
        elif persist:
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

    def _trip(self, reason: str, pause_seconds: int = 0, persist: bool = True) -> None:
        """Trip the circuit breaker."""
        self._tripped = True
        self._trip_reason = reason
        self._resume_time = (time.time() + pause_seconds) if pause_seconds > 0 else 0.0
        # R16: persist BEFORE logging — a trip that a keeper-kill erases is
        # the exact failure this fix exists for.
        if persist:
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

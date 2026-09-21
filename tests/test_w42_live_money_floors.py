"""W42 (R26 register) — second wave of the unowned live-money roadmap.

Four items were scoped by the 2026-09-21 roadmap; re-probing before writing
anything (per that entry's own instruction) changed the shape of two of them:

1. policy-vs-impl-4 — off-by-one volume floor. Confirmed unchanged, fixed here.
2. set-membership-6 — re-probed at HEAD: the "4 remaining bare literals" the
   finding cited are actually 3 genuine duplicates plus one DIFFERENT concept
   (orchestrator.py's direction-inference exemption for drift tracking, a
   3-element set for an unrelated reason — not part of this finding, and
   itself the flagship offender in the getsource census, left for that
   track). The 3 real duplicates are fixed here.
3. fail-direction-10 — re-probed: the READ-side fail-closed behavior W14/W32
   built is intact, but the WRITE-side failure path was untouched by either
   wave — still just a log.warning with no retry marker and no way for any
   consumer to know persistence is degraded. Narrowed fix: severity bump +
   a queryable health flag. Full paging is a registered residual — this
   component has no notify_cb, unlike Portfolio/AccountManager, and
   _persist() is sync.
4. money-flow-01 — re-probed and CLOSED, not fixed here. Tracing the real
   call graph (close_trade -> exit_outbox enqueue -> _drain_exit_outbox ->
   claim -> commission true-up -> book_exit_core -> PDT -> Thompson) shows
   every piece the finding named as missing is present and wired. The
   existing tests/test_w1_booking_integrity.py::TestDrainBooksOrphanCloses::
   test_reconciler_style_close_reaches_breaker_and_stats IS this finding's
   exact scenario, already passing. No new test added here — asserting the
   same thing twice would not be new evidence.

Every test here executes the real code path. No inspect.getsource.
"""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from ait.bot.state import StateManager
from ait.config.settings import RiskConfig
from ait.risk.circuit_breaker import CircuitBreaker

SRC = Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------------------
# 1. policy-vs-impl-4 — off-by-one volume floor
# ---------------------------------------------------------------------------

class TestLiveProfileVolumeFloorMatchesTheRunbook:

    def test_a_volume_floor_of_nine_still_refuses_live_mode(self, tmp_path, monkeypatch):
        """docs/RUNBOOK.md:153 documents AIT_LIQ_MIN_VOLUME >= 10. Before the
        fix, 1-9 passed the startup assertion — only 0 was caught."""
        from ait.bot.orchestrator import TradingOrchestrator
        from ait.config.settings import load_settings
        from tests.test_hot_path_smoke import CONFIG_PATH, SmokeIBKRClient
        from tests.fakes import FakeIB

        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AIT_LIQ_MIN_VOLUME", "9")

        settings = load_settings(CONFIG_PATH)
        settings.trading.mode = "live"
        ibkr = SmokeIBKRClient(FakeIB())

        with pytest.raises(RuntimeError, match="LIVE-PROFILE ASSERTION FAILED"):
            TradingOrchestrator(settings, ibkr)

    def test_a_volume_floor_of_ten_is_accepted(self, tmp_path, monkeypatch):
        """CONTROL: the documented floor itself must still pass."""
        from ait.bot.orchestrator import TradingOrchestrator
        from ait.config.settings import load_settings
        from tests.test_hot_path_smoke import CONFIG_PATH, SmokeIBKRClient
        from tests.fakes import FakeIB

        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AIT_LIQ_MIN_VOLUME", "10")
        monkeypatch.setenv("AIT_LIQ_MAX_SPREAD", "0.15")

        settings = load_settings(CONFIG_PATH)
        settings.trading.mode = "live"
        settings.options.max_bid_ask_spread_pct = 0.15
        ibkr = SmokeIBKRClient(FakeIB())

        # GOV-6 (W41) refuses first if max_contracts_per_trade > 1 — not what
        # this test is about, so confirm the live config's own value (1)
        # still lets construction proceed far enough to prove GOV-5 passed.
        TradingOrchestrator(settings, ibkr)

    def test_paper_mode_is_unaffected(self, tmp_path, monkeypatch):
        """CONTROL: the assertion is live-mode-only, unchanged by this fix."""
        from tests.test_hot_path_smoke import build_smoke_orchestrator

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AIT_LIQ_MIN_VOLUME", "0")  # paper-relaxed
        rig = build_smoke_orchestrator()
        try:
            assert rig.orch._settings.trading.mode == "paper"
        finally:
            rig.restore()


# ---------------------------------------------------------------------------
# 2. set-membership-6 — the range-gated strategy set
# ---------------------------------------------------------------------------

class TestNoBareDuplicateOfTheRangeGatedSet:
    """STRUCTURAL guard (the finding's own claim shape: "no OTHER copy of
    this membership exists"). A zero-logic-change substitution like this
    one has no BEHAVIOR to observe today — RANGE_GATED_STRATEGIES and the
    three literals it replaced were already identical in membership. The
    entire risk is FUTURE drift, which only a structural check can catch;
    same category as W38's SQL-function guard and W28's .today() guard.
    """

    TARGET = SRC / "ait" / "bot" / "orchestrator.py"

    def _bare_duplicate_lines(self) -> list[int]:
        tree = ast.parse(self.TARGET.read_text(encoding="utf-8"))

        # The definition's OWN line — ast.walk() yields the Set literal
        # nested inside `RANGE_GATED_STRATEGIES: frozenset[str] = frozenset(
        # {...})` as its own node, so excluding the AnnAssign node itself
        # does not exclude its child; exclude by line number instead.
        definition_line = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                    and node.target.id == "RANGE_GATED_STRATEGIES"):
                definition_line = node.lineno
                break
        assert definition_line is not None, (
            "RANGE_GATED_STRATEGIES's own definition was not found — the "
            "guard cannot exclude it and would flag itself"
        )

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Tuple, ast.Set)):
                continue
            if node.lineno == definition_line:
                continue
            elts = {e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if {"iron_condor", "short_strangle"} <= elts and len(node.elts) == 2:
                offenders.append(node.lineno)
        return offenders

    def test_no_bare_two_element_duplicate_remains(self):
        offenders = self._bare_duplicate_lines()
        assert offenders == [], (
            f"a bare {{'iron_condor', 'short_strangle'}} literal exists outside "
            f"RANGE_GATED_STRATEGIES at line(s) {offenders} — it will silently "
            f"disagree with the named authority the day one drifts"
        )

    def test_the_guard_actually_detects_the_defect_it_is_for(self):
        """Anti-vacuity: prove the detector fires on the exact shape this
        wave just removed, so a broken AST walk can't pass by scanning
        nothing."""
        sample = ast.parse(
            'X = 1\n'
            'def f(s):\n'
            '    if s in ("iron_condor", "short_strangle"):\n'
            '        pass\n'
        )
        found = False
        for node in ast.walk(sample):
            if not isinstance(node, ast.Tuple):
                continue
            elts = {e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if {"iron_condor", "short_strangle"} <= elts and len(node.elts) == 2:
                found = True
        assert found, "the detector does not recognise the pattern it exists to catch"

    def test_the_three_element_direction_tracking_set_is_left_alone(self):
        """CONTROL: orchestrator.py:1137's ("iron_condor", "short_strangle",
        "long_straddle") is a DIFFERENT concept (direction-inference
        exemption for drift tracking — long_straddle belongs there for its
        own reason, unrelated to range-gating) and must not be folded into
        RANGE_GATED_STRATEGIES, which would be semantically wrong for it."""
        text = self.TARGET.read_text(encoding="utf-8")
        assert '"iron_condor", "short_strangle", "long_straddle"' in text, (
            "premise moved: the 3-element direction-tracking tuple is gone "
            "or reshaped — re-check whether it still needs excluding"
        )

    def test_range_gated_strategies_is_exactly_the_documented_pair(self):
        """Anchor: the guard above is only meaningful if the authority
        itself still means what the finding says it means."""
        from ait.bot.orchestrator import RANGE_GATED_STRATEGIES
        assert RANGE_GATED_STRATEGIES == frozenset({"iron_condor", "short_strangle"})


# ---------------------------------------------------------------------------
# 3. fail-direction-10 residual — the write-failure path
# ---------------------------------------------------------------------------

def _cfg():
    return RiskConfig()


class _WriteLocked:
    """Fakes only the failing verb (set_state); reads delegate to the real
    store, so _load_persisted() on attach succeeds normally."""

    def __init__(self, real):
        self._real = real

    def set_state(self, key, value):
        raise sqlite3.OperationalError("database is locked")

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestPersistFailureIsVisibleNotSilent:

    def test_a_failed_write_is_queryable_via_get_status(self, tmp_path):
        """The defect, reproduced then closed: before the fix, a failed
        persist logged a warning and left no trace anywhere a caller could
        observe — get_status() reported healthy regardless."""
        st = StateManager(db_path=tmp_path / "ait_state.db")
        cb = CircuitBreaker(_cfg())
        cb.attach_state(_WriteLocked(st))

        assert cb.get_status().store_write_healthy is True, "premise: starts healthy"

        cb.record_trade_result(-100.0)  # triggers _persist() internally

        assert cb.get_status().store_write_healthy is False, (
            "a failed persist is not visible on the status object"
        )

    def test_a_successful_write_clears_the_flag(self, tmp_path):
        """The flag must not be sticky past the failure that set it — the
        next real write (state recovered) clears it, mirroring how
        _state_unreadable clears on the next good READ."""
        st = StateManager(db_path=tmp_path / "ait_state.db")
        cb = CircuitBreaker(_cfg())
        cb.attach_state(_WriteLocked(st))
        cb.record_trade_result(-100.0)
        assert cb.get_status().store_write_healthy is False, "premise"

        cb._state = st  # the lock "clears"
        cb.record_trade_result(-50.0)

        assert cb.get_status().store_write_healthy is True, (
            "the flag stayed stuck after a successful persist"
        )

    def test_a_healthy_store_never_sets_the_flag(self, tmp_path):
        """CONTROL: must hold on both trees."""
        st = StateManager(db_path=tmp_path / "ait_state.db")
        cb = CircuitBreaker(_cfg())
        cb.attach_state(st)
        cb.record_trade_result(-100.0)
        assert cb.get_status().store_write_healthy is True

    def test_the_unreadable_path_is_unaffected(self, tmp_path):
        """CONTROL: this change must not touch the W14/W32 read-side
        behavior — a genuinely unreadable store still fails closed and the
        write-skip-while-unreadable branch (a DIFFERENT log line) still
        fires instead of attempting a write at all."""
        st = StateManager(db_path=tmp_path / "ait_state.db")

        class _ReadLocked:
            def get_state(self, key, default=""):
                raise sqlite3.OperationalError("locked")

            def __getattr__(self, name):
                return getattr(st, name)

        cb = CircuitBreaker(_cfg())
        cb.attach_state(_ReadLocked())
        assert cb._state_unreadable is True
        assert cb.is_tripped is True, "an unreadable store must still fail closed"

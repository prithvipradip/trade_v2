"""W22: the reconciler's untracked-OPT block, rewritten — driven through the
REAL reconcile(), not a unit test of the extracted logic in isolation.

Covers the three defects registered against this exact code path (W21
session) that this rewrite fixes:
  (a) only the LAST untracked position reached HALT_UNTRACKED (per-iteration
      write_text clobbered itself) — here, the halt file must list ALL
      currently-frozen keys.
  (b)/(c) the sighting debounce was an in-memory dict, reset on every
      restart — here, a FRESH PositionReconciler instance against the same
      StateManager must still recognize a prior sighting.
Plus the core W22 behavior: a declared position never reaches the freeze
machinery at all, and an explicit "no" is treated with the same urgency as
an unknown position.
"""
from __future__ import annotations

import asyncio

import pytest

from ait.bot.state import StateManager
from ait.execution.reconciler import PositionReconciler
from tests.fakes import FakeIB, FakeIBKRClient, _option_position
from tests.test_w22_manual_positions import FakeTelegram

KEY_SHORT = "QQQ:712.0:C:2026-09-18"   # _option_position("QQQ", 712.0, "C", "20260918", ...)
KEY_LONG = "QQQ:712.0:C:2026-10-16"    # _option_position("QQQ", 712.0, "C", "20261016", ...)


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _rec(state, ib=None, telegram=None) -> PositionReconciler:
    r = PositionReconciler(FakeIBKRClient(ib or FakeIB()), state)
    if telegram is not None:
        r.set_telegram(telegram)
    return r


def _backdate_last_seen(state, key, minutes_ago):
    from datetime import timedelta
    from ait.utils.time import now_et
    state.set_state(f"manual_position_last_seen_{key}",
                    (now_et() - timedelta(minutes=minutes_ago)).isoformat())


def _halt_text(tmp_path) -> str | None:
    p = tmp_path / "data" / "HALT_UNTRACKED"
    return p.read_text() if p.exists() else None


@pytest.fixture(autouse=True)
def _cwd_tmp(tmp_path, monkeypatch):
    """The reconciler writes data/HALT_UNTRACKED relative to cwd — run each
    test from its own tmp_path so tests never touch the real repo's file."""
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


class TestDeclaredPositionNeverFreezes:
    def test_declared_yes_position_never_reaches_the_halt_file(self, tmp_path):
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        # Even with backdated sightings that WOULD otherwise trigger a
        # freeze, a declared key must short-circuit before the debounce.
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)

        rec = _rec(st, ib)
        result = asyncio.run(rec.reconcile())

        assert _halt_text(tmp_path) is None
        assert not any(KEY_SHORT in d for d in result.discrepancies)

    def test_declared_position_still_counts_as_new_from_ibkr(self, tmp_path):
        """It's still a real broker position for accounting purposes —
        only the FREEZE/discrepancy machinery is skipped."""
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        result = asyncio.run(_rec(st, ib).reconcile())
        assert result.new_from_ibkr == 1


class TestExplicitNoStillFreezes:
    def test_rejected_position_freezes_same_as_unknown(self, tmp_path):
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "no")
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]

        asyncio.run(_rec(st, ib).reconcile())

        halt = _halt_text(tmp_path)
        assert halt is not None
        assert KEY_SHORT in halt


class TestUndeclaredRequiresTwoSightings:
    def test_first_sighting_defers_no_halt_file(self, tmp_path):
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        result = asyncio.run(_rec(st, ib).reconcile())

        assert _halt_text(tmp_path) is None
        assert any("1st sighting" in d for d in result.discrepancies)

    def test_second_sighting_past_the_debounce_window_freezes(self, tmp_path):
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]

        asyncio.run(_rec(st, ib).reconcile())  # 1st sighting
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)  # simulate elapsed time
        asyncio.run(_rec(st, ib).reconcile())  # 2nd sighting, past 2min

        halt = _halt_text(tmp_path)
        assert halt is not None
        assert KEY_SHORT in halt

    def test_sighting_too_soon_still_defers(self, tmp_path):
        """Gap < 2 minutes must NOT freeze — the debounce's whole reason for
        existing (a phantom Gateway row disproved minutes later)."""
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        asyncio.run(_rec(st, ib).reconcile())
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=0.5)  # 30s ago, < 2min
        asyncio.run(_rec(st, ib).reconcile())
        assert _halt_text(tmp_path) is None


class TestHaltFileListsAllFrozenKeys:
    """Defect (a): the old per-iteration write_text clobbered itself,
    leaving the file naming only the LAST untracked position."""

    def test_two_simultaneously_frozen_positions_both_appear(self, tmp_path):
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [
            _option_position("QQQ", 712.0, "C", "20260918", -1.0),
            _option_position("QQQ", 712.0, "C", "20261016", 1.0),
        ]
        rec = _rec(st, ib)
        asyncio.run(rec.reconcile())  # both 1st-sighting, deferred
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        _backdate_last_seen(st, KEY_LONG, minutes_ago=10)
        asyncio.run(rec.reconcile())  # both past debounce -> both freeze

        halt = _halt_text(tmp_path)
        assert halt is not None
        assert KEY_SHORT in halt
        assert KEY_LONG in halt

    def test_a_pass_with_only_one_frozen_key_does_not_carry_over_the_other(
            self, tmp_path):
        """A key that WAS frozen (declared since, or genuinely closed) must
        drop out of the file on the next pass — the file is a snapshot of
        THIS pass, not an accumulating log."""
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [
            _option_position("QQQ", 712.0, "C", "20260918", -1.0),
            _option_position("QQQ", 712.0, "C", "20261016", 1.0),
        ]
        rec = _rec(st, ib)
        asyncio.run(rec.reconcile())
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        _backdate_last_seen(st, KEY_LONG, minutes_ago=10)
        asyncio.run(rec.reconcile())
        assert KEY_LONG in _halt_text(tmp_path)

        # KEY_LONG gets declared; only KEY_SHORT remains at the broker+frozen.
        # No backdating here on purpose — KEY_SHORT is ALREADY confirmed
        # (was_prompted), so it must stay frozen on the very next pass with
        # no gap at all. See TestAlreadyConfirmedNeverTransientlyUnfreezes.
        st.set_state(f"manual_position_answer_{KEY_LONG}", "yes")
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        asyncio.run(rec.reconcile())

        halt = _halt_text(tmp_path)
        assert KEY_SHORT in halt
        assert KEY_LONG not in halt


class TestDebounceSurvivesARestart:
    """Defects (b)/(c): the old in-memory _untracked_sightings dict reset on
    every restart. This box restarts often — the debounce must not."""

    def test_a_fresh_reconciler_instance_still_recognizes_the_prior_sighting(
            self, tmp_path):
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]

        rec1 = _rec(st, ib)
        asyncio.run(rec1.reconcile())  # 1st sighting, on instance #1
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)

        # Simulates a restart: a BRAND NEW PositionReconciler, same DB.
        rec2 = _rec(st, ib)
        asyncio.run(rec2.reconcile())

        halt = _halt_text(tmp_path)
        assert halt is not None, (
            "a fresh instance failed to recognize the prior sighting — "
            "the debounce reset on 'restart' exactly like the old bug")
        assert KEY_SHORT in halt


class TestAlreadyConfirmedNeverTransientlyUnfreezes:
    """The regression this session's own testing caught: record_sighting
    resets last_seen on EVERY call (the pre-existing debounce algorithm's
    gap-since-the-LAST-call semantics, unchanged by W22). Combined with
    "write the halt file as a complete snapshot of this pass", two
    reconcile() calls landing close together could otherwise silently drop
    an ALREADY-CONFIRMED freeze for a cycle — was_prompted() closes this."""

    def test_two_reconciles_seconds_apart_after_confirmation_stay_frozen(
            self, tmp_path):
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        rec = _rec(st, ib)

        asyncio.run(rec.reconcile())  # 1st sighting, deferred
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        asyncio.run(rec.reconcile())  # 2nd sighting -> CONFIRMED, frozen
        assert KEY_SHORT in _halt_text(tmp_path)

        asyncio.run(rec.reconcile())  # immediately again, zero gap
        halt = _halt_text(tmp_path)
        assert halt is not None and KEY_SHORT in halt, (
            "an already-confirmed position transiently un-froze because "
            "two reconcile() calls landed within the debounce window")


class TestHaltFileLifecycleIsLogged:
    """Diagnostic hardening from the 2026-09-16 investigation: both halves
    of the file's lifecycle (written / cleared) now get a dedicated log
    line, not just the general reconciliation_complete summary — so a
    future occurrence of that day's unexplained disappearance is a log
    lookup instead of a multi-session investigation."""

    def test_writing_the_halt_file_is_logged_with_the_keys(self, tmp_path, caplog):
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        rec = _rec(st, ib)
        asyncio.run(rec.reconcile())
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)

        logged = []
        import ait.execution.reconciler as recmod
        orig = recmod.log.info
        recmod.log.info = lambda event, **kw: (logged.append((event, kw)), orig(event, **kw))[0]
        try:
            asyncio.run(rec.reconcile())
        finally:
            recmod.log.info = orig

        matches = [kw for ev, kw in logged if ev == "halt_untracked_written"]
        assert matches, "halt_untracked_written never logged"
        assert KEY_SHORT in matches[-1]["keys"]

    def test_clearing_the_halt_file_is_logged(self, tmp_path):
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        rec = _rec(st, ib)
        asyncio.run(rec.reconcile())
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        asyncio.run(rec.reconcile())
        assert _halt_text(tmp_path) is not None  # frozen, file present

        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")

        logged = []
        import ait.execution.reconciler as recmod
        orig = recmod.log.info
        recmod.log.info = lambda event, **kw: (logged.append((event, kw)), orig(event, **kw))[0]
        try:
            asyncio.run(rec.reconcile())
        finally:
            recmod.log.info = orig

        assert _halt_text(tmp_path) is None  # actually cleared
        assert any(ev == "halt_untracked_cleared" for ev, _ in logged), (
            "file was cleared but no diagnostic log line fired")


class TestPromptIsSentOnFreeze:
    def test_prompt_sent_when_telegram_is_wired(self, tmp_path):
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        tg = FakeTelegram()
        rec = _rec(st, ib, telegram=tg)
        asyncio.run(rec.reconcile())
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        asyncio.run(rec.reconcile())

        assert len(tg.buttoned) == 1
        assert KEY_SHORT in tg.buttoned[0]["text"]

    def test_no_telegram_wired_does_not_crash_reconcile(self, tmp_path):
        """set_telegram() is never called — reconcile() must still freeze
        correctly, just without a prompt (INST-3 must not depend on
        Telegram being configured)."""
        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_option_position("QQQ", 712.0, "C", "20260918", -1.0)]
        rec = _rec(st, ib)  # no telegram
        asyncio.run(rec.reconcile())
        _backdate_last_seen(st, KEY_SHORT, minutes_ago=10)
        asyncio.run(rec.reconcile())
        assert KEY_SHORT in _halt_text(tmp_path)


class TestDeclaredPositionDisappearanceIsConfirmed:
    def test_closed_declared_position_triggers_confirmation_during_reconcile(
            self, tmp_path):
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_SHORT}", "yes")
        tg = FakeTelegram()
        ib = FakeIB()  # no positions at all now — it's been closed
        rec = _rec(st, ib, telegram=tg)

        asyncio.run(rec.reconcile())

        assert any(KEY_SHORT in s for s in tg.sent)
        assert st.get_state(f"manual_position_answer_{KEY_SHORT}") == ""

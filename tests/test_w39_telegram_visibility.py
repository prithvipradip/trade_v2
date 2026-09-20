"""W39 (R26 F6 rows 7-8) — the W22 Telegram rows, and the dead config switch.

FINDING A, `w22-unrecognised-telegram-updates-are-dropped-with-no-trace`, is
real and fixed; its behavioural tests live beside the rest of the poller's in
`tests/test_w22_manual_positions.py`. This file covers what that one cannot:
the config switch, and the part of FINDING B that survives.

FINDING B, `w22-leaked-msgid-keeps-a-dead-position-pending-forever`, IS CLOSED
— by W30, not by this wave, and the ledger says so. Re-derived at HEAD before
anything was claimed:

    prompted, unanswered      -> pending_keys ['QQQ:712.0:C:2026-09-18']
    after authoritative sweep -> pending_keys []            msgid deleted

A `manual_position_prompt_msgid_` row can only exist downstream of
`record_sighting()`, so a `manual_position_last_seen_` row always accompanies
it, and W30's second cleanup loop clears both. BOTH consumers the finding names
— `pending_keys()` and the digest's `awaiting_reply` — read that same msgid
row, so both self-heal on the next pass that can confirm the broker set.

What survives is narrower than the finding, and R26's suggested fix ("filter to
keys still present at the broker") cannot be implemented as written: the ONLY
state in which a stale prompt lingers is the one where the sweep is skipped
because the broker set is NOT authoritative — and the digest runs in the
supervisor, which has no IB connection at all. Inferring presence from stamp
freshness there would be exactly the guessing W30 removed (and a naive TTL
would mark a live position stale overnight, when no reconcile passes run).

So the residual is treated as what it is: a correct fail-closed skip that was
INVISIBLE. It now says so, and the self-healing is pinned below so a future
change to the sweep cannot silently reopen the finding.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path

import pytest

from ait.bot.state import StateManager
from ait.execution import manual_positions as mp
from ait.notifications.telegram import TelegramNotifier

KEY = "QQQ:712.0:C:2026-09-18"


class _TG:
    """Duck-typed notifier double."""

    enabled = True

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, message: str) -> int:
        self.sent.append(message)
        return 1

    async def send_with_buttons(self, message: str, buttons) -> int:
        self.sent.append(message)
        return 555

    async def edit_message(self, *a, **k) -> None:
        pass

    async def answer_callback(self, *a, **k) -> None:
        pass


@pytest.fixture
def state(tmp_path):
    return StateManager(db_path=tmp_path / "ait_state.db")


def _digest_awaiting(db_path) -> set[str]:
    """The EXACT computation master.py's daily digest runs, so this test moves
    when that does."""
    con = sqlite3.connect(str(db_path))
    try:
        prompted = {r[0][len("manual_position_prompt_msgid_"):] for r in con.execute(
            "SELECT key FROM bot_state WHERE key LIKE "
            "'manual_position_prompt_msgid_%'")}
        answered = {r[0][len("manual_position_answer_"):] for r in con.execute(
            "SELECT key FROM bot_state WHERE key LIKE 'manual_position_answer_%'")}
    finally:
        con.close()
    return prompted - answered


class TestW30AlreadyClosedThePendingForeverClaim:
    """REGRESSION PIN, not a fix. If a future change to the sweep reopens this,
    these go red and the finding is live again."""

    def test_an_authoritative_pass_clears_a_prompt_for_a_closed_position(self, state):
        tg = _TG()
        mp.record_sighting(state, KEY)          # the real call order: last_seen first
        asyncio.run(mp.prompt_if_needed(state, tg, KEY, 1.0))
        assert mp.pending_keys(state) == [KEY], "premise: the key must start pending"

        # The broker no longer reports it, and the caller could confirm that.
        asyncio.run(mp.reconcile_declared_positions(state, tg, set()))

        assert mp.pending_keys(state) == [], (
            "a prompt for a position that is gone is still pending — the W30 "
            "sighting-state sweep no longer clears the msgid row"
        )

    def test_the_digest_self_heals_through_the_same_row(self, state):
        """The finding names the digest separately, but it reads the same key."""
        tg = _TG()
        mp.record_sighting(state, KEY)
        asyncio.run(mp.prompt_if_needed(state, tg, KEY, 1.0))
        assert _digest_awaiting(state._db_path) == {KEY}, "premise"

        asyncio.run(mp.reconcile_declared_positions(state, tg, set()))

        assert _digest_awaiting(state._db_path) == set(), (
            "the daily digest still reports a manual position awaiting a reply "
            "after the position closed and a pass confirmed it"
        )

    def test_a_live_position_is_left_alone(self, state):
        """CONTROL: the sweep must only clear keys the broker no longer has."""
        tg = _TG()
        mp.record_sighting(state, KEY)
        asyncio.run(mp.prompt_if_needed(state, tg, KEY, 1.0))

        asyncio.run(mp.reconcile_declared_positions(state, tg, {KEY}))

        assert mp.pending_keys(state) == [KEY], (
            "a position still at the broker had its prompt cleared"
        )


class TestTheConfigSwitchNowSwitchesSomething:
    """`notifications.telegram.enabled` was read by nothing in src/."""

    def test_disabling_it_disables_the_notifier(self):
        assert TelegramNotifier("tok", "chat", enabled=False).enabled is False, (
            "notifications.telegram.enabled still does nothing"
        )

    def test_enabling_it_with_credentials_stays_enabled(self):
        """CONTROL: must hold on both trees."""
        assert TelegramNotifier("tok", "chat", enabled=True).enabled is True

    def test_the_default_is_unchanged_for_existing_callers(self):
        """Every existing construction site omits the keyword."""
        assert TelegramNotifier("tok", "chat").enabled is True

    def test_credentials_still_win_when_absent(self):
        """Config cannot enable a notifier that has nowhere to send."""
        assert TelegramNotifier("", "", enabled=True).enabled is False

    def test_a_disabled_notifier_stops_the_poller(self, state):
        """The consequence the startup warning names, asserted.

        The poller is how a HALT_UNTRACKED prompt gets answered, so this is
        the operator-visible cost of the switch and must not be a surprise.
        """
        tg = TelegramNotifier("tok", "chat", enabled=False)
        asyncio.run(mp.run_poller(state, tg, "123456"))  # returns immediately

    def test_the_dead_send_flags_are_gone(self):
        """They were never read; leaving them would keep implying they work."""
        from ait.config.settings import TelegramConfig

        fields = set(TelegramConfig.model_fields)
        assert fields == {"enabled"}, (
            f"TelegramConfig still declares unread switches: {fields - {'enabled'}}"
        )

    def test_the_yaml_and_the_model_agree(self):
        """`extra=forbid` means a stale yaml key is a startup crash, so the
        two must land together."""
        from ait.config.settings import load_settings

        assert load_settings("config.yaml").notifications.telegram.enabled is True

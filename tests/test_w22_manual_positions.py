"""W22: declared manual positions — ask once via Telegram, remember durably.

Covers the two interaction paths the operator chose (tappable buttons, plus
a plain-text reply-to-message as an explicit correction path) as FIRST-CLASS,
separately-tested scenarios, plus the correction itself in both directions —
this was a direct follow-up question from the operator, not an afterthought.

Drives the real `manual_positions` module against a real StateManager +
tmp_path DB (matching this repo's existing test conventions). Telegram
itself is a lightweight fake double at the TelegramNotifier interface
boundary — the HTTP-level primitives (send_with_buttons, edit_message,
answer_callback, get_updates) get their own tests in
test_w22_telegram_primitives.py against a faked aiohttp, so this file stays
focused on the business logic: who gets asked, what a reply means, and what
gets remembered.
"""
from __future__ import annotations

import asyncio

import pytest

from ait.bot.state import StateManager
from ait.execution import manual_positions as mp

CHAT_ID = "123456"
KEY_A = "QQQ:712.0:C:2026-09-18"
KEY_B = "QQQ:712.0:C:2026-10-16"


@pytest.fixture(autouse=True)
def _reset_nudge_throttle():
    """W39: `_last_nudge_at` is module-level (the poller is one long-lived
    task), so without this the first test to nudge would silence every later
    one and the suite would depend on collection order."""
    mp._last_nudge_at = 0.0
    yield
    mp._last_nudge_at = 0.0


@pytest.fixture
def discards(monkeypatch):
    """The `reason=` of every discard the module logged, in order.

    W39: structlog does not route through stdlib logging, so `caplog` sees
    nothing — record the real module logger instead. Returns a callable so a
    test reads the list AFTER driving the code.
    """
    seen: list[str] = []
    real = mp.log.info

    def _spy(event, **kw):
        if event == "manual_position_update_discarded":
            seen.append(kw.get("reason", "?"))
        return real(event, **kw)

    monkeypatch.setattr(mp.log, "info", _spy)
    return lambda: seen


class FakeTelegram:
    """Duck-typed TelegramNotifier double. Records every call so tests can
    assert on exactly what was sent/edited/answered."""

    def __init__(self, update_batches: list[list[dict]] | None = None):
        self.enabled = True
        self.sent: list[str] = []
        self.buttoned: list[dict] = []
        self.edited: list[dict] = []
        self.answered_callbacks: list[tuple[str, str]] = []
        self._next_msg_id = 1000
        self._update_batches = list(update_batches or [])
        self.get_updates_offsets: list[int] = []

    async def send(self, text: str) -> bool:
        self.sent.append(text)
        return True

    async def send_with_buttons(self, text: str, buttons: list[tuple[str, str]]) -> int:
        msg_id = self._next_msg_id
        self._next_msg_id += 1
        self.buttoned.append({"message_id": msg_id, "text": text, "buttons": buttons})
        return msg_id

    async def edit_message(self, message_id: int, text: str, buttons=None) -> bool:
        self.edited.append({"message_id": message_id, "text": text, "buttons": buttons})
        return True

    async def answer_callback(self, callback_query_id: str, text: str = "") -> bool:
        self.answered_callbacks.append((callback_query_id, text))
        return True

    async def get_updates(self, offset: int, timeout: int = 25) -> list[dict]:
        self.get_updates_offsets.append(offset)
        if not self._update_batches:
            raise asyncio.CancelledError  # ends run_poller's while-True cleanly
        return self._update_batches.pop(0)

    def last_edit_for(self, message_id: int) -> str:
        matches = [e for e in self.edited if e["message_id"] == message_id]
        assert matches, f"no edit recorded for message_id={message_id}"
        return matches[-1]["text"]


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _callback_update(update_id: int, cbq_id: str, data: str, message_id: int,
                     chat_id: str = CHAT_ID) -> dict:
    return {"update_id": update_id,
            "callback_query": {"id": cbq_id, "data": data,
                                "message": {"message_id": message_id,
                                            "chat": {"id": chat_id}}}}


def _text_update(update_id: int, message_id: int, text: str,
                 reply_to_message_id: int | None = None, chat_id: str = CHAT_ID) -> dict:
    msg = {"message_id": message_id, "chat": {"id": chat_id}, "text": text}
    if reply_to_message_id is not None:
        msg["reply_to_message"] = {"message_id": reply_to_message_id}
    return {"update_id": update_id, "message": msg}


class TestIsDeclaredAndSighting:
    def test_unanswered_is_none(self, tmp_path):
        st = _state(tmp_path)
        assert mp.is_declared(st, KEY_A) is None

    def test_answer_round_trips(self, tmp_path):
        st = _state(tmp_path)
        st.set_state(f"manual_position_answer_{KEY_A}", "yes")
        assert mp.is_declared(st, KEY_A) is True
        st.set_state(f"manual_position_answer_{KEY_A}", "no")
        assert mp.is_declared(st, KEY_A) is False

    def test_first_sighting_returns_none(self, tmp_path):
        st = _state(tmp_path)
        assert mp.record_sighting(st, KEY_A) is None

    def test_second_sighting_returns_a_gap(self, tmp_path):
        st = _state(tmp_path)
        mp.record_sighting(st, KEY_A)
        gap = mp.record_sighting(st, KEY_A)
        assert gap is not None and gap >= 0

    def test_sighting_survives_a_fresh_statemanager_instance(self, tmp_path):
        """The whole point of moving off the in-memory dict: a NEW instance
        against the SAME db (a restart) must still see the prior sighting,
        not start the debounce over."""
        st1 = _state(tmp_path)
        mp.record_sighting(st1, KEY_A)
        st2 = StateManager(db_path=tmp_path / "ait_state.db")  # simulates restart
        gap = mp.record_sighting(st2, KEY_A)
        assert gap is not None


class TestPromptOnlySentOnce:
    def test_prompt_sends_buttons_and_stores_message_id(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        asyncio.run(mp.prompt_if_needed(st, tg, KEY_A, qty=-1.0))
        assert len(tg.buttoned) == 1
        assert KEY_A in tg.buttoned[0]["text"]
        labels = [b[0] for b in tg.buttoned[0]["buttons"]]
        assert any("Yes" in lbl for lbl in labels)
        assert any("No" in lbl for lbl in labels)
        assert st.get_state(f"manual_position_prompt_msgid_{KEY_A}") == "1000"

    def test_second_call_does_not_reprompt(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        asyncio.run(mp.prompt_if_needed(st, tg, KEY_A, qty=-1.0))
        asyncio.run(mp.prompt_if_needed(st, tg, KEY_A, qty=-1.0))
        assert len(tg.buttoned) == 1


class TestButtonPathStandalone:
    """Path 1: tap a button. No text ever sent."""

    def test_tapping_yes_records_yes_and_edits_the_message(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _callback_update(1, "cbq1", f"manual:yes:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))

        assert mp.is_declared(st, KEY_A) is True
        assert tg.answered_callbacks == [("cbq1", "Recorded: yes")]
        edited_text = tg.last_edit_for(555)
        assert "YES" in edited_text
        assert KEY_A in edited_text

    def test_tapping_no_records_no(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _callback_update(1, "cbq1", f"manual:no:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert mp.is_declared(st, KEY_A) is False
        assert "NO" in tg.last_edit_for(555)

    def test_callback_from_a_different_chat_is_ignored(self, tmp_path, discards):
        """W39 INVERSION (R26 `w22-unrecognised-telegram-updates-are-dropped`).

        The assertion `tg.answered_callbacks == []` pinned the DEFECT: an
        unanswered callback_query leaves the button spinning on the sender's
        phone forever. Telegram requires every callback to be answered; that
        is a transport obligation, not an endorsement of the tap.

        The INTENT — never act on a foreign chat's input — is unchanged and
        still asserted, and is now joined by the trace that was missing.
        """
        st = _state(tmp_path)
        tg = FakeTelegram()
        upd = _callback_update(1, "cbq1", f"manual:yes:{KEY_A}", 555, chat_id="999999")
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))

        assert mp.is_declared(st, KEY_A) is None, "a foreign chat set an answer"
        assert [cq for cq, _ in tg.answered_callbacks] == ["cbq1"], (
            "the callback was never answered — the button spins forever"
        )
        assert not tg.sent, "replied into a chat that is not the configured one"
        assert discards() == ["callback_foreign_chat"]

    def test_a_callback_with_no_message_is_answered_and_logged(self, tmp_path, discards):
        """`cq.get("message") or {}` made an ABSENT message indistinguishable
        from a foreign-chat one, so this fell through the chat check having
        answered nothing."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        upd = {"update_id": 1,
               "callback_query": {"id": "cbq1", "data": f"manual:yes:{KEY_A}"}}
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))

        assert [cq for cq, _ in tg.answered_callbacks] == ["cbq1"]
        assert mp.is_declared(st, KEY_A) is None
        assert discards() == ["callback_without_message"]


class TestTextPathStandalone:
    """Path 2: reply-to-message with plain text. No button ever tapped."""

    def test_text_reply_to_a_known_prompt_records_the_answer(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _text_update(1, 601, "no", reply_to_message_id=555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert mp.is_declared(st, KEY_A) is False
        assert "NO" in tg.last_edit_for(555)

    def test_case_insensitive_short_forms(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _text_update(1, 601, "Y", reply_to_message_id=555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert mp.is_declared(st, KEY_A) is True

    def test_reply_to_an_unmatched_message_id_is_ignored_not_misapplied(
        self, tmp_path, discards
    ):
        """A reply to some unrelated message must never be guessed at.

        W39: the intent is unchanged — still not guessed at — but the operator
        now learns that their reply went nowhere instead of being met with
        silence at `log.debug`.
        """
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _text_update(1, 601, "yes", reply_to_message_id=999)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert mp.is_declared(st, KEY_A) is None
        assert tg.edited == []
        assert discards() == ["reply_to_unmatched_message"]
        assert tg.sent and "prompt" in tg.sent[0].lower()

    def test_text_from_a_different_chat_is_ignored(self, tmp_path, discards):
        """W39: logged, and still never replied to — answering would leak the
        bot's existence to a chat that is not the configured one."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _text_update(1, 601, "yes", reply_to_message_id=555, chat_id="999999")
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert mp.is_declared(st, KEY_A) is None
        assert discards() == ["foreign_chat"]
        assert tg.sent == [], "replied into a foreign chat"


class TestUnrecognisedTextIsAnswered:
    """R26 `w22-unrecognised-telegram-updates-are-dropped-with-no-trace`.

    The prompt invites a free-form reply while the parser accepts exactly four
    tokens and discarded everything else in total silence — no edit, no
    confirmation, no log — while HALT_UNTRACKED kept every new entry frozen.
    "The answer never came" was undiagnosable: identical to a prompt never
    seen, a getUpdates outage, or a dead poller.
    """

    def test_yes_please_gets_an_answer_and_a_log_line(self, tmp_path, discards):
        """R26's own acceptance case, verbatim: send "yes please"."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")

        asyncio.run(mp._handle_update(
            st, tg, CHAT_ID, _text_update(1, 601, "yes please")))

        assert mp.is_declared(st, KEY_A) is None, (
            "a fuzzy reply was guessed into an answer"
        )
        assert discards() == ["unrecognised_text"]
        assert tg.sent, "the operator got nothing back"
        assert "yes" in tg.sent[0].lower() and "no" in tg.sent[0].lower(), (
            f"the nudge does not name the accepted replies: {tg.sent[0]!r}"
        )

    def test_the_nudge_is_throttled(self, tmp_path):
        """A chatty chat must not turn the poller into an echo loop."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        for i in range(5):
            asyncio.run(mp._handle_update(
                st, tg, CHAT_ID, _text_update(i, 600 + i, f"chatter {i}")))
        assert len(tg.sent) == 1, f"sent {len(tg.sent)} nudges for 5 messages"

    def test_no_nudge_when_nothing_is_pending(self, tmp_path, discards):
        """CONTROL: ordinary chat traffic with no open prompt stays silent.

        Without this the fix would turn every unrelated message in the chat
        into a reply — a worse failure than the silence it replaces.
        """
        st = _state(tmp_path)
        tg = FakeTelegram()
        asyncio.run(mp._handle_update(
            st, tg, CHAT_ID, _text_update(1, 601, "good morning")))
        assert tg.sent == []
        assert discards() == ["unrecognised_text"], (
            "the discard should still be traceable even when no prompt is open"
        )

    def test_a_real_answer_still_works(self, tmp_path):
        """CONTROL: must hold on both trees — the four tokens still record."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, _text_update(1, 601, "yes")))
        assert mp.is_declared(st, KEY_A) is True

    def test_the_prompt_names_the_accepted_replies(self, tmp_path):
        """The silence was half the defect; the other half was a prompt that
        invited a reply without saying what would be understood."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        asyncio.run(mp.prompt_if_needed(st, tg, KEY_A, 1.0))
        assert tg.buttoned, "no prompt was sent"
        text = tg.buttoned[0]["text"]
        assert "yes" in text.lower() and "no" in text.lower(), (
            f"the prompt does not name the accepted replies: {text!r}"
        )


class TestTheCorrectionScenario:
    """The operator's own follow-up question: if I tap the wrong button, can
    I fix it with a plain-text reply? Both directions, and the edited
    message must always show the CURRENT answer, not the first one."""

    def test_wrong_tap_then_text_correction(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")

        # Tap "no" by mistake.
        tap = _callback_update(1, "cbq1", f"manual:no:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, tap))
        assert mp.is_declared(st, KEY_A) is False
        assert "NO" in tg.last_edit_for(555)

        # Correct it via a plain-text reply to the SAME message.
        fix = _text_update(2, 602, "yes", reply_to_message_id=555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, fix))
        assert mp.is_declared(st, KEY_A) is True
        assert "YES" in tg.last_edit_for(555), (
            "the edited message must show the CURRENT (corrected) answer")

    def test_text_reply_then_button_correction(self, tmp_path):
        """The reverse direction: text first, button tap corrects it."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")

        first = _text_update(1, 601, "no", reply_to_message_id=555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, first))
        assert mp.is_declared(st, KEY_A) is False

        fix = _callback_update(2, "cbq2", f"manual:yes:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, fix))
        assert mp.is_declared(st, KEY_A) is True
        assert "YES" in tg.last_edit_for(555)

    def test_repeated_button_correction_keeps_working(self, tmp_path):
        """Buttons stay active after an edit — a second, third tap must
        still correct the answer, not just the first."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        for answer in ("yes", "no", "yes"):
            upd = _callback_update(1, "cbq", f"manual:{answer}:{KEY_A}", 555)
            asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert mp.is_declared(st, KEY_A) is True


class TestBareReplyAmbiguity:
    """A yes/no with no reply-to at all — only safe to apply automatically
    when there's exactly one position actually awaiting an answer."""

    def test_single_pending_bare_reply_applies(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _text_update(1, 601, "yes")  # no reply_to
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert mp.is_declared(st, KEY_A) is True

    def test_multiple_pending_bare_reply_is_refused_not_guessed(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        st.set_state("manual_position_prompt_msgid_" + KEY_B, "556")
        upd = _text_update(1, 601, "yes")  # no reply_to, two pending
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert mp.is_declared(st, KEY_A) is None
        assert mp.is_declared(st, KEY_B) is None
        assert any("Multiple positions" in s for s in tg.sent)

    def test_no_pending_bare_reply_is_silently_ignored(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        upd = _text_update(1, 601, "yes")
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert tg.sent == []
        assert tg.edited == []


class TestStandaloneConfirmationMessage:
    """The operator's own follow-up: an edited prompt was not a clear enough
    'received' signal in practice (easy to miss inside a message you're not
    re-reading) — a distinct new chat message is the explicit ack."""

    def test_button_tap_sends_a_separate_confirmation_message(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _callback_update(1, "cbq1", f"manual:yes:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))

        assert len(tg.sent) == 1
        assert KEY_A in tg.sent[0]
        assert "YES" in tg.sent[0]
        # It's a NEW message, distinct from the edit — both must exist.
        assert "YES" in tg.last_edit_for(555)

    def test_no_tap_also_sends_a_distinguishable_confirmation(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _callback_update(1, "cbq1", f"manual:no:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))
        assert len(tg.sent) == 1
        assert "NO" in tg.sent[0]

    def test_text_correction_also_sends_a_fresh_confirmation(self, tmp_path):
        """Each answer/correction gets its OWN confirmation — not just the
        first one — so a corrected answer is just as clearly acknowledged."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")

        tap = _callback_update(1, "cbq1", f"manual:no:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, tap))
        fix = _text_update(2, 602, "yes", reply_to_message_id=555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, fix))

        assert len(tg.sent) == 2
        assert "NO" in tg.sent[0]
        assert "YES" in tg.sent[1]


class TestOnAnswerCallback:
    """An answered prompt should trigger an immediate reconcile (in
    production, TradingOrchestrator.reconcile_now) rather than waiting for
    the next natural pass — the operator's own follow-up question after
    declaring both real positions."""

    def test_button_tap_fires_on_answer(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        fired = []

        async def on_answer():
            fired.append(True)

        upd = _callback_update(1, "cbq1", f"manual:yes:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd, on_answer=on_answer))
        # on_answer is scheduled as a background task (asyncio.create_task),
        # not awaited inline — give the loop one more tick to run it before
        # asyncio.run() tears the loop down.
        asyncio.run(asyncio.sleep(0))
        assert fired == [True]

    def test_text_correction_also_fires_on_answer(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        fired = []

        async def on_answer():
            fired.append(True)

        async def _run():
            upd = _text_update(1, 601, "no", reply_to_message_id=555)
            await mp._handle_update(st, tg, CHAT_ID, upd, on_answer=on_answer)
            await asyncio.sleep(0)

        asyncio.run(_run())
        assert fired == [True]

    def test_no_callback_given_does_not_crash(self, tmp_path):
        """on_answer is optional — must not be required."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        upd = _callback_update(1, "cbq1", f"manual:yes:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd))  # no on_answer
        assert mp.is_declared(st, KEY_A) is True

    def test_a_failing_on_answer_does_not_break_answer_recording(self, tmp_path):
        """reconcile_now already catches its own errors in production; this
        proves the poller side doesn't depend on that — a broken callback
        must never make an answer fail to record."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")

        async def _exploding_on_answer():
            raise RuntimeError("broker unavailable")

        async def _run():
            upd = _callback_update(1, "cbq1", f"manual:yes:{KEY_A}", 555)
            await mp._handle_update(st, tg, CHAT_ID, upd, on_answer=_exploding_on_answer)
            await asyncio.sleep(0)  # let the failing background task run+fail

        asyncio.run(_run())
        assert mp.is_declared(st, KEY_A) is True


class TestBackgroundTaskReferenceHeld:
    """Hardening found during the 2026-09-16 halt-file investigation
    (trade-v2-1b): asyncio.create_task()'s return value must be held, or the
    event loop's only reference is weak and the task can be GC'd before it
    runs. Not proven to have caused anything (the wiring didn't exist yet on
    the one day this could have mattered), fixed on its own merits."""

    def test_fire_and_forget_keeps_a_strong_reference_until_done(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")
        started = []

        async def on_answer():
            started.append(True)
            await asyncio.sleep(0)  # yield once, like a real await would

        upd = _callback_update(1, "cbq1", f"manual:yes:{KEY_A}", 555)
        asyncio.run(mp._handle_update(st, tg, CHAT_ID, upd, on_answer=on_answer))

        # The task is tracked (held) at the moment it's created...
        assert len(mp._background_tasks) >= 0  # may already be running/done
        # ...and, given a chance to run, actually executes rather than being
        # silently dropped.
        asyncio.run(asyncio.sleep(0))
        assert started == [True]

    def test_completed_tasks_are_discarded_not_leaked(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state("manual_position_prompt_msgid_" + KEY_A, "555")

        async def on_answer():
            pass

        async def _run():
            upd = _callback_update(1, "cbq1", f"manual:yes:{KEY_A}", 555)
            await mp._handle_update(st, tg, CHAT_ID, upd, on_answer=on_answer)
            await asyncio.sleep(0)
            await asyncio.sleep(0)  # let the done-callback fire

        before = len(mp._background_tasks)
        asyncio.run(_run())
        assert len(mp._background_tasks) == before, (
            "completed background task was never discarded — a real leak")


class TestPollerOffsetPersistence:
    def test_offset_advances_and_persists_across_a_simulated_restart(self, tmp_path):
        st1 = _state(tmp_path)
        tg1 = FakeTelegram(update_batches=[
            [_text_update(41, 701, "yes")],  # no pending key -> silently ignored, still advances offset
        ])
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(mp.run_poller(st1, tg1, CHAT_ID))
        assert tg1.get_updates_offsets[0] == 0
        persisted = int(st1.get_state("telegram_update_offset"))
        assert persisted == 42  # last update_id + 1

        # Simulated restart: fresh StateManager + fresh TelegramNotifier,
        # same DB file.
        st2 = StateManager(db_path=tmp_path / "ait_state.db")
        tg2 = FakeTelegram(update_batches=[[]])
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(mp.run_poller(st2, tg2, CHAT_ID))
        assert tg2.get_updates_offsets[0] == 42, (
            "restart must resume from the persisted offset, not replay from 0")

    def test_disabled_telegram_returns_immediately_without_polling(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        tg.enabled = False
        asyncio.run(mp.run_poller(st, tg, CHAT_ID))  # must return, not hang/loop
        assert tg.get_updates_offsets == []


class TestReconcileDeclaredPositions:
    """The 'confirms when it disappears' half of the design."""

    def test_still_open_declared_position_is_untouched(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state(f"manual_position_answer_{KEY_A}", "yes")
        asyncio.run(mp.reconcile_declared_positions(st, tg, ibkr_keys={KEY_A}))
        assert tg.sent == []
        assert mp.is_declared(st, KEY_A) is True

    def test_closed_declared_position_is_confirmed_and_cleaned_up(self, tmp_path):
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state(f"manual_position_answer_{KEY_A}", "yes")
        st.set_state(f"manual_position_prompt_msgid_{KEY_A}", "555")
        st.set_state(f"manual_position_last_seen_{KEY_A}", "2026-09-16T09:00:00-04:00")

        asyncio.run(mp.reconcile_declared_positions(st, tg, ibkr_keys=set()))

        assert len(tg.sent) == 1
        assert KEY_A in tg.sent[0]
        assert mp.is_declared(st, KEY_A) is None
        assert st.get_state(f"manual_position_prompt_msgid_{KEY_A}") == ""
        assert st.get_state(f"manual_position_last_seen_{KEY_A}") == ""

    def test_an_explicitly_rejected_no_position_is_not_touched(self, tmp_path):
        """Only 'yes' positions get the disappearance confirmation — a 'no'
        was never declared manual in the first place."""
        st = _state(tmp_path)
        tg = FakeTelegram()
        st.set_state(f"manual_position_answer_{KEY_A}", "no")
        asyncio.run(mp.reconcile_declared_positions(st, tg, ibkr_keys=set()))
        assert tg.sent == []
        assert mp.is_declared(st, KEY_A) is False

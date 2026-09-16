"""W22: declared manual positions — ask once via Telegram, remember durably.

`HALT_UNTRACKED` (reconciler.py) freezes all new entries whenever a live
option position at the broker has no local trade record (INST-3 — the
mid-placement-crash protection). An operator's own manual TWS trade trips
the exact same guard, for as long as the position stays open. This module
lets the operator declare "this one is mine, ignore it for management" once,
durably, without weakening INST-3 for a genuinely unrecognized position.

DESIGN, per the operator's own choices:
  - Tappable inline-keyboard buttons on the prompt (not just plain text).
  - A plain-text reply-to-the-prompt-message ALSO works, at any time,
    overriding whatever was tapped — the explicit correction path. Both
    paths funnel through _record_and_edit so a wrong tap is always visibly
    fixable, not just technically reversible: the prompt message is edited
    to show whichever answer is CURRENT.
  - A declared position is ignored for management but still counted in
    exposure — that needs no code here: portfolio.py's exposure/delta/mark
    functions already iterate ib.portfolio() directly, never filtered by
    "is this in our local trades table".

All durable state lives in the existing bot_state key/value table (no schema
change), the same convention already used for digest markers and
trueup_pending_<id>:
  manual_position_answer_<key>          "yes" | "no"      (absent = unanswered)
  manual_position_prompt_msgid_<key>    Telegram message_id, as a string
  manual_position_last_seen_<key>       ISO ET timestamp — replaces the old
                                         in-memory _untracked_sightings dict,
                                         so the sighting debounce survives a
                                         restart (this box restarts often).
  telegram_update_offset                getUpdates cursor
"""
from __future__ import annotations

import asyncio

from ait.bot.state import StateManager
from ait.notifications.telegram import TelegramNotifier
from ait.utils.logging import get_logger
from ait.utils.time import now_et

log = get_logger("execution.manual_positions")

_ANSWER_PREFIX = "manual_position_answer_"
_MSGID_PREFIX = "manual_position_prompt_msgid_"
_LAST_SEEN_PREFIX = "manual_position_last_seen_"
_CONFIRMED_PREFIX = "manual_position_confirmed_"
_OFFSET_KEY = "telegram_update_offset"

# W22 hardening (trade-v2-1b, 2026-09-16): asyncio.create_task()'s return
# value must be held somewhere, or the event loop's only reference to the
# task is a weak one — nothing stops it being garbage-collected before it
# ever runs, per asyncio's own documented guidance. Not proven to have
# happened here (verified against git history: the wiring this protects
# didn't exist yet on the one occasion it could have explained something),
# but the risk is real and unrelated to that, so fixed on its own merits
# rather than left as a latent footgun.
_background_tasks: set = set()


def _fire_and_forget(coro) -> None:
    task = asyncio.ensure_future(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def is_declared(state: StateManager, key: str) -> bool | None:
    """True/False/None (unanswered) for whether `key` is a declared-manual
    position."""
    v = state.get_state(_ANSWER_PREFIX + key)
    if v == "yes":
        return True
    if v == "no":
        return False
    return None


def record_sighting(state: StateManager, key: str) -> float | None:
    """Durable replacement for the old in-memory `_untracked_sightings`
    dict — same semantics (overwrite the last-seen stamp every call, gap =
    now minus the PRIOR call's stamp), now surviving a restart.

    Returns the gap in minutes since the previous sighting, or None if this
    is the first sighting ever recorded for this key.
    """
    now = now_et()
    prev_iso = state.get_state(_LAST_SEEN_PREFIX + key)
    state.set_state(_LAST_SEEN_PREFIX + key, now.isoformat())
    # StateManager.get_state always returns str (the stored value, or its
    # "" default) — this isinstance check only ever matters against a test
    # double (several existing reconciler tests pass a bare MagicMock() as
    # _state) that doesn't honor that contract; treat anything else as "no
    # prior value" rather than crashing fromisoformat on a non-string.
    if not isinstance(prev_iso, str) or not prev_iso:
        return None
    from datetime import datetime
    prev = datetime.fromisoformat(prev_iso)
    return (now - prev).total_seconds() / 60


def is_confirmed(state: StateManager, key: str) -> bool:
    """Has this key already passed the sighting debounce at least once and
    not yet been resolved (declared or disappeared)? A RECONCILER-OWNED
    fact, deliberately independent of whether a Telegram prompt was ever
    sent or succeeded (Telegram may be unconfigured, or a send may fail —
    INST-3's freeze must not depend on that).

    Used to skip RE-running the sighting debounce on every later pass once a
    position is already confirmed — the debounce's job is proving
    PERSISTENCE, and once proven it must not need re-proving every reconcile
    call. Without this, a key already confirmed could transiently drop out
    of HALT_UNTRACKED if two reconcile() calls happen to land within the
    2-minute debounce window (record_sighting resets the last-seen stamp on
    EVERY call, deferred or not — the pre-existing debounce algorithm's own
    gap-since-the-LAST-call semantics, unchanged by W22), silently lifting a
    live INST-3 freeze for a cycle. Caught by this wave's own test suite.
    """
    return state.get_state(_CONFIRMED_PREFIX + key) == "1"


def mark_confirmed(state: StateManager, key: str) -> None:
    state.set_state(_CONFIRMED_PREFIX + key, "1")


def pending_keys(state: StateManager) -> list[str]:
    """Keys that have been prompted but not yet answered — used to decide
    whether a bare (no reply-to) yes/no reply is unambiguous."""
    out = []
    for k in state.state_keys_like(_MSGID_PREFIX + "%"):
        key = k[len(_MSGID_PREFIX):]
        if is_declared(state, key) is None:
            out.append(key)
    return out


def _buttons(key: str) -> list[tuple[str, str]]:
    # "manual:yes:<key>" / "manual:no:<key>" — well under Telegram's 64-byte
    # callback_data limit for any realistic contract key.
    return [("✅ Yes, mine", f"manual:yes:{key}"),
            ("❌ No, investigate", f"manual:no:{key}")]


async def prompt_if_needed(
        state: StateManager, telegram: TelegramNotifier, key: str, qty: float) -> None:
    """Send the declare/reject prompt exactly once per key — re-entrant safe
    across reconcile cycles, since it checks for an existing message_id
    first."""
    if state.get_state(_MSGID_PREFIX + key):
        return  # already asked
    text = (f"🔔 Unmanaged option position detected:\n{key}  qty={qty}\n\n"
            f"Is this a manual/intentional trade of yours? New entries stay "
            f"frozen until you answer.")
    msg_id = await telegram.send_with_buttons(text, _buttons(key))
    if msg_id is not None:
        state.set_state(_MSGID_PREFIX + key, str(msg_id))
        log.info("manual_position_prompt_sent", key=key, message_id=msg_id)
    else:
        log.warning("manual_position_prompt_failed", key=key)


def _key_for_message_id(state: StateManager, message_id: int) -> str | None:
    """Reverse lookup: which position was this prompt message about? Scans
    the (always small — a handful at most) set of sent prompts rather than
    keeping a second, easy-to-desync key namespace."""
    target = str(message_id)
    for k in state.state_keys_like(_MSGID_PREFIX + "%"):
        if state.get_state(k) == target:
            return k[len(_MSGID_PREFIX):]
    return None


async def _record_and_edit(
        state: StateManager, telegram: TelegramNotifier, key: str, answer: str,
        message_id: int, on_answer=None) -> None:
    """The one place an answer is written — used by every path (button tap,
    reply-to correction, bare unambiguous reply), so the edited message
    always reflects whichever answer is CURRENT.

    `on_answer`, if given, is an async callable (TradingOrchestrator.
    reconcile_now, in production) fired as a background task so HALT_UNTRACKED
    reflects the new answer immediately rather than waiting for the next
    natural reconcile pass — without making the poller wait on a broker call
    before it can process the NEXT inbound update.

    Sends TWO things, deliberately, not one: the original prompt is edited
    in place (self-documenting if you scroll back to that specific
    question), AND a separate, standalone confirmation message is sent as
    its own chat bubble. The edit alone was not a clear enough "received"
    signal in practice — easy to miss inside a longer message you're not
    re-reading — so a distinct new message is the explicit acknowledgment.
    """
    state.set_state(_ANSWER_PREFIX + key, answer)
    label = ("✅ YES — declared manual, ignored for management"
              if answer == "yes" else
              "❌ NO — still frozen for review")
    text = (f"🔔 Unmanaged option position:\n{key}\n\n"
            f"You said: {label}\n\n"
            f"(tap a button or reply to THIS message to change your answer)")
    await telegram.edit_message(message_id, text, _buttons(key))
    confirm = ("✅ Got it — recorded as YES"
               if answer == "yes" else
               "📝 Got it — recorded as NO")
    await telegram.send(f"{confirm}: {key}")
    log.info("manual_position_answer_recorded", key=key, answer=answer,
              message_id=message_id)
    if on_answer is not None:
        _fire_and_forget(on_answer())


async def _handle_update(
        state: StateManager, telegram: TelegramNotifier, chat_id: str, upd: dict,
        on_answer=None) -> None:
    cq = upd.get("callback_query")
    if cq is not None:
        msg = cq.get("message") or {}
        if str((msg.get("chat") or {}).get("id", "")) != str(chat_id):
            return
        data = cq.get("data", "")
        parts = data.split(":", 2)
        if len(parts) == 3 and parts[0] == "manual" and parts[1] in ("yes", "no"):
            _, answer, key = parts
            await _record_and_edit(state, telegram, key, answer, msg["message_id"],
                                   on_answer=on_answer)
            await telegram.answer_callback(cq["id"], f"Recorded: {answer}")
        else:
            await telegram.answer_callback(cq["id"])
        return

    msg = upd.get("message")
    if msg is None:
        return
    if str((msg.get("chat") or {}).get("id", "")) != str(chat_id):
        return
    text = (msg.get("text") or "").strip().lower()
    if text not in ("yes", "y", "no", "n"):
        return
    answer = "yes" if text in ("yes", "y") else "no"

    reply_to = msg.get("reply_to_message")
    if reply_to is not None:
        # Explicit correction path: replying to a SPECIFIC prompt always
        # resolves to that position, whatever the current answer is. A
        # reply to some other, unrelated message is ignored rather than
        # guessed at.
        key = _key_for_message_id(state, reply_to["message_id"])
        if key is None:
            log.debug("manual_position_reply_unmatched",
                       message_id=reply_to["message_id"])
            return
        await _record_and_edit(state, telegram, key, answer, reply_to["message_id"],
                               on_answer=on_answer)
        return

    pending = pending_keys(state)
    if len(pending) == 1:
        key = pending[0]
        msgid = int(state.get_state(_MSGID_PREFIX + key))
        await _record_and_edit(state, telegram, key, answer, msgid, on_answer=on_answer)
    elif len(pending) > 1:
        await telegram.send(
            "Multiple positions are awaiting a reply — please reply "
            "directly to the specific prompt message so I know which one "
            "you mean.")
    # len(pending) == 0: nothing awaiting an answer; ignore silently.


async def run_poller(state: StateManager, telegram: TelegramNotifier, chat_id: str,
                     on_answer=None) -> None:
    """Long-poll Telegram for inbound updates (button taps + replies) and
    apply them. Runs as its own task alongside the trading loop (main.py),
    cancelled the same way on shutdown. A single bad poll or a single
    malformed update never kills the task — both are caught and logged so
    the poller keeps running.

    `on_answer`, if given, is an async callable fired (as a background task,
    never awaited inline) each time an answer is recorded — in production,
    TradingOrchestrator.reconcile_now, so HALT_UNTRACKED reflects a new
    answer immediately instead of waiting for the next natural reconcile
    pass. Optional and decoupled on purpose: the poller must keep working
    (recording answers, editing messages) even with no orchestrator wired,
    e.g. in tests, or if reconcile_now itself is slow or fails — it never
    blocks processing the NEXT inbound update.
    """
    if not telegram.enabled:
        log.info("manual_position_poller_skipped_telegram_disabled")
        return
    offset = int(state.get_state(_OFFSET_KEY, "0") or "0")
    log.info("manual_position_poller_starting", offset=offset)
    while True:
        try:
            updates = await telegram.get_updates(offset, timeout=25)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("manual_position_poll_failed", error=str(e))
            await asyncio.sleep(5)
            continue

        for upd in updates:
            offset = max(offset, upd.get("update_id", offset - 1) + 1)
            try:
                await _handle_update(state, telegram, chat_id, upd, on_answer=on_answer)
            except Exception as e:  # noqa: BLE001
                log.warning("manual_position_update_handling_failed",
                            error=str(e), update_id=upd.get("update_id"))
        if updates:
            state.set_state(_OFFSET_KEY, str(offset))


async def reconcile_declared_positions(
        state: StateManager, telegram: TelegramNotifier, ibkr_keys: set[str]) -> None:
    """Confirm and clean up declared positions that have disappeared from
    the broker (the operator closed them) — the "confirms when the position
    disappears" half of the design. Called once per reconcile() pass with
    the CURRENT set of broker position keys.
    """
    for k in state.state_keys_like(_ANSWER_PREFIX + "%"):
        key = k[len(_ANSWER_PREFIX):]
        if state.get_state(k) != "yes":
            continue
        if key in ibkr_keys:
            continue  # still open, nothing to do
        await telegram.send(
            f"✅ {key} is no longer at the broker — no longer tracking it "
            f"as a declared manual position.")
        state.delete_state(_ANSWER_PREFIX + key)
        state.delete_state(_MSGID_PREFIX + key)
        state.delete_state(_LAST_SEEN_PREFIX + key)
        state.delete_state(_CONFIRMED_PREFIX + key)
        log.info("manual_position_closed_confirmed", key=key)

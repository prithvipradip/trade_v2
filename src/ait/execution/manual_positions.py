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
  - A declared position is ignored for management but stays counted in
    exposure for portfolio.py's exposure/delta/mark functions, which already
    iterate ib.portfolio() directly, never filtered by "is this in our local
    trades table". RiskManager._open_positions is NOT one of those — it is
    built only from local trades (orchestrator._sync_risk_manager_positions),
    so a declared naked position was invisible to the aggregate capital-at-
    risk cap, the symbol-concentration cap, correlation, and the open-
    position count (W30, R26 F3 finding #3). Fixed via
    PositionReconciler.declared_position_risk_rows(), which synthesizes a
    risk row per declared key from the broker position.

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
# W30 (R26 F3, finding #5): the quantity in effect at the moment a key was
# DECLARED. is_declared() alone matches on contract identity only (symbol:
# strike:right:expiry) — see declared_quantity_matches().
_QTY_PREFIX = "manual_position_declared_qty_"

#: The only replies that record an answer. Declared once so the guidance the
#: operator is shown cannot drift from what `_handle_update` actually accepts —
#: W39 found the prompt inviting a free-form reply while the parser took four
#: tokens and discarded everything else in silence.
_ACCEPTED_REPLIES = ("yes", "y", "no", "n")
_ACCEPTED_HELP = "reply **yes** or **no** (y / n work too), or tap a button"

#: W39: one nudge per this many seconds, so an unrecognised-text reply in a
#: busy chat cannot turn the poller into an echo loop. Module-level because the
#: poller is a single long-lived task.
_NUDGE_INTERVAL_S = 300.0
_last_nudge_at: float = 0.0
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


async def drain_background_tasks(timeout: float = 5.0) -> None:
    """W30 (R26 F3, finding #6): on_answer (TradingOrchestrator.reconcile_now)
    runs detached via _fire_and_forget — nothing in the shutdown path ever
    waited for one to finish, so a prompt answered right as the bot shuts
    down could have its reconcile_now() call abandoned mid-write when
    main.py disconnects IBKR and exits. Call this AFTER the poller task is
    cancelled (so no new answer can be recorded) and BEFORE disconnecting
    IBKR. Bounded so a wedged task cannot hang shutdown forever.
    """
    pending = [t for t in _background_tasks if not t.done()]
    if not pending:
        return
    log.info("manual_position_draining_background_tasks", count=len(pending))
    _done, still_pending = await asyncio.wait(pending, timeout=timeout)
    if still_pending:
        log.warning("manual_position_background_drain_timed_out",
                    count=len(still_pending))


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


def _clear_key_state(state: StateManager, key: str) -> None:
    """Every durable state key this module writes for `key` — one place, so
    a cleanup path can never miss one and leave a stale row behind (W30,
    R26 F3 findings #2/#4)."""
    state.delete_state(_ANSWER_PREFIX + key)
    state.delete_state(_MSGID_PREFIX + key)
    state.delete_state(_LAST_SEEN_PREFIX + key)
    state.delete_state(_CONFIRMED_PREFIX + key)
    state.delete_state(_QTY_PREFIX + key)


def declared_quantity_matches(state: StateManager, key: str, qty: float) -> bool:
    """W30 (R26 F3, finding #5): is_declared() matches on CONTRACT IDENTITY
    alone — an operator who declared qty=-1 as theirs had a LATER, undeclared
    qty=-3 at the identical contract silently waved through too, invisible to
    every risk gate. Returns True only when the position is declared YES
    *and* the broker's current quantity still matches what was declared at
    prompt time; otherwise clears the stale answer/prompt (but deliberately
    NOT the sighting/confirmed state) so the size change re-enters the
    freeze/re-prompt path on THIS pass — immediately, not after a fresh
    2-sighting debounce, because a size change on an already-approved
    contract is MORE suspect than a brand-new untracked one, not less.

    A key declared before this quantity tracking existed has no recorded
    qty (empty string): treated as a match (nothing to compare against yet)
    rather than an automatic freeze of every pre-existing declaration.
    """
    if is_declared(state, key) is not True:
        return False
    declared_qty = state.get_state(_QTY_PREFIX + key)
    if not declared_qty:
        return True
    try:
        if float(declared_qty) == float(qty):
            return True
    except (TypeError, ValueError):
        return True  # unparseable stored value — a data glitch, not a size change
    log.warning("manual_position_quantity_changed", key=key,
                declared_qty=declared_qty, current_qty=qty)
    state.delete_state(_ANSWER_PREFIX + key)
    state.delete_state(_MSGID_PREFIX + key)
    state.delete_state(_QTY_PREFIX + key)
    return False


def _nudge_is_due(_now: float | None = None) -> bool:
    """True at most once per `_NUDGE_INTERVAL_S`, so a chatty chat cannot turn
    an unrecognised-reply nudge into an echo loop. Stamps on success."""
    global _last_nudge_at
    import time as _t
    now = _t.monotonic() if _now is None else _now
    if now - _last_nudge_at < _NUDGE_INTERVAL_S:
        return False
    _last_nudge_at = now
    return True


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
    # W30 (finding #5): the qty the operator is being SHOWN, so a later size
    # change on the same contract can be detected against this baseline.
    state.set_state(_QTY_PREFIX + key, str(qty))
    text = (f"🔔 Unmanaged option position detected:\n{key}  qty={qty}\n\n"
            f"Is this a manual/intentional trade of yours? New entries stay "
            f"frozen until you answer.\n"
            f"({_ACCEPTED_HELP}.)")
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
            f"(to change your answer, tap a button or reply to THIS message "
            f"with {' / '.join(_ACCEPTED_REPLIES)})")
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
        # W39 (R26 `w22-unrecognised-telegram-updates-are-dropped-with-no-trace`):
        # `cq.get("message") or {}` made an ABSENT message behave exactly like a
        # foreign-chat one, so both fell out of the next line having answered
        # nothing — and an unanswered callback_query leaves the button spinning
        # on the operator's phone indefinitely. Answer first, then decide.
        msg = cq.get("message")
        if msg is None:
            await telegram.answer_callback(cq["id"])
            log.info("manual_position_update_discarded",
                     reason="callback_without_message", update_id=upd.get("update_id"))
            return
        if str((msg.get("chat") or {}).get("id", "")) != str(chat_id):
            await telegram.answer_callback(cq["id"])
            log.info("manual_position_update_discarded",
                     reason="callback_foreign_chat",
                     from_chat=str((msg.get("chat") or {}).get("id", "")),
                     update_id=upd.get("update_id"))
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
        log.info("manual_position_update_discarded", reason="not_a_message",
                 update_id=upd.get("update_id"))
        return
    if str((msg.get("chat") or {}).get("id", "")) != str(chat_id):
        # Logged but NEVER answered: replying into a chat that is not the
        # configured one would leak the bot's existence to it.
        log.info("manual_position_update_discarded", reason="foreign_chat",
                 from_chat=str((msg.get("chat") or {}).get("id", "")),
                 update_id=upd.get("update_id"))
        return
    text = (msg.get("text") or "").strip().lower()
    if text not in _ACCEPTED_REPLIES:
        # W39: this was the sharpest of the four silent drops. The prompt
        # invites a free-form reply, so an operator answering "Yes it's mine"
        # got NOTHING back — no edit, no confirmation, no log — while
        # HALT_UNTRACKED kept every new entry frozen. Indistinguishable from a
        # dead poller, a missed prompt or a getUpdates outage, and therefore
        # undiagnosable from the logs.
        log.info("manual_position_update_discarded", reason="unrecognised_text",
                 text=text[:80], update_id=upd.get("update_id"))
        if pending_keys(state) and _nudge_is_due():
            await telegram.send(
                f"I didn't understand that — {_ACCEPTED_HELP}.")
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
            # W39: was log.debug, i.e. invisible at the live INFO level. The
            # operator replied to SOMETHING and got silence.
            log.info("manual_position_update_discarded",
                     reason="reply_to_unmatched_message",
                     message_id=reply_to["message_id"])
            if _nudge_is_due():
                await telegram.send(
                    "That reply wasn't attached to one of my position prompts, "
                    "so I've left it alone. Reply to the prompt message itself, "
                    "or tap its buttons.")
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
    """Confirm and clean up state for positions that have disappeared from
    the broker (the operator closed them) — the "confirms when the position
    disappears" half of the design. Called once per reconcile() pass with
    the CURRENT set of broker position keys — the caller (reconciler.py)
    must already have resolved `ibkr_keys` AUTHORITATIVELY (W30, R26 F3):
    absence from this set is taken as proof the position is gone, so an
    ambiguous/stale cache must never reach here as if it were confirmed.
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
        _clear_key_state(state, key)
        log.info("manual_position_closed_confirmed", key=key)

    # W30 (R26 F3, finding #4): a key that was SIGHTED/CONFIRMED (frozen,
    # maybe prompted) but never got an explicit yes/no answer leaked its
    # sighting/prompt state forever once the position closed — the loop
    # above only ever looks at ANSWERED keys. A stale
    # manual_position_confirmed_<key> then let a LATER, genuinely NEW
    # sighting of the identical contract skip the R5 debounce entirely
    # (is_confirmed() treats it as already-proven), and a stale
    # manual_position_prompt_msgid_<key> silently suppressed the new prompt
    # (prompt_if_needed treats any stored msgid as "already asked").
    for k in state.state_keys_like(_LAST_SEEN_PREFIX + "%"):
        key = k[len(_LAST_SEEN_PREFIX):]
        if is_declared(state, key) is not None:
            continue  # answered yes (handled above) or no (kept, by design)
        if key in ibkr_keys:
            continue  # still open
        _clear_key_state(state, key)
        log.info("manual_position_sighting_state_cleared", key=key)

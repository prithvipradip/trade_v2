"""The ET timezone and the stored-timestamp convention — STANDARD LIBRARY ONLY.

THE CONTRACT, which a guard test enforces: nothing in this module may import
anything outside the standard library, directly or transitively.

It exists because two callers cannot afford the alternative. `ait.utils.time`
owns the market calendar and therefore imports `pandas_market_calendars` ->
pandas -> numpy AT MODULE SCOPE, so importing it costs the whole scientific
stack:

  - `orchestration/master.py` is the SUPERVISOR. W23 imported `ait.utils.time`
    there and a broken scientific stack — the classic ABI break after an
    upgrade — killed the supervisor at import: no bot restart, no Gateway
    recovery, no page, while the bot may be holding condors. W34 removed it.
  - `reporting/go_live.py` is required to stay import-light for reporting-only
    consumers (see the note at `strategies/base.py:17`); the dashboard and the
    status server import it.

Before W38 both resolved ET with their own inline copies, and W38 was about to
add a third. This project's own rule is the reason not to: duplicating a
single-source-of-truth constant "has repeatedly caused silent divergence —
import instead". So the rule lives here once, `ait.utils.time` re-exports it so
every existing caller is unchanged, and the two constrained modules import it
directly at a cost of nothing but stdlib.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")


def now_et() -> datetime:
    """Current time as an aware ET datetime."""
    return datetime.now(ET)


def parse_trade_ts(value: str) -> datetime:
    """Parse a stored trade timestamp and return it as an AWARE ET datetime.

    W21: `trades.entry_time` / `trades.exit_time` (and the `open_positions` /
    `exit_outbox` mirrors) were written offset-free by `datetime.now()`, so the
    whole corpus is ET only by accident of the host. Writers now emit
    `now_et().isoformat()`, but a naive legacy row must keep parsing correctly
    and must keep meaning ET — otherwise a half-migrated DB silently shifts
    every trading-day and hour-of-day bucket downstream.

    Naive input is localised to ET; aware input is CONVERTED to ET, so `.date()`
    and `.hour` are ET calendar semantics for every caller regardless of which
    offset the row happens to carry. Raises ValueError/TypeError exactly as
    `datetime.fromisoformat` does, so existing caller guards still apply.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ET)
    return parsed.astimezone(ET)


def assume_market_open_on_clock(now: datetime | None = None) -> bool:
    """Calendar-FREE conservative answer to "is the market open?".

    Used only when the NYSE calendar is unavailable. A weekday 09:30-16:00 ET
    is treated as "may be open". Resolves ET through stdlib ``zoneinfo``, not
    ``ait.utils.time`` — the latter imports ``pandas_market_calendars`` at
    module scope, so in the very scenario that calls this it may be the thing
    that is broken.

    W20d — on the DIRECTION of this heuristic, because it looks like the one
    W15 deleted: W15 removed ``weekday() < 5`` from the pre-event blackout
    because counting a holiday as a session made that gate fail OPEN. Here the
    same heuristic errs the other way. Every caller asks "is it open?" in order
    to be MORE cautious when the answer is yes, so a holiday wrongly counted as
    a session only defers a restart or keeps hang-detection armed. Never
    reuse it anywhere the true branch is the permissive one.
    """
    from zoneinfo import ZoneInfo as _ZI
    _et = _ZI("US/Eastern")
    n = now or datetime.now(_et)
    n = n.replace(tzinfo=_et) if n.tzinfo is None else n.astimezone(_et)
    if n.weekday() >= 5:
        return False
    mins = n.hour * 60 + n.minute
    return (9 * 60 + 30) <= mins < (16 * 60)

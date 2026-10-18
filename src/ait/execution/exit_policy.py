"""Shared credit-exit policy — ONE authority for live and research.

R20 #5a: the backtesting engine hand-copied live portfolio.py's credit
take-profit ladder (0.50/0.40/0.30/0.20 with DTE breakpoints), the DTE<=5
expiry-approaching close, and the macro-flatten windows. The copies were
verified equal at the time — but a live policy tuning would have silently
de-synced every parity backtest (the same divergence class that motivated
importing CREDIT_STRATEGIES from strategies/base.py). The values now live
here, in a pure import-light module consumed by the engine.

R20 follow-up: src/ait/execution/portfolio.py now imports these directly
(take_profit_targets, EXPIRY_APPROACHING_DTE, macro_flatten_window_days)
instead of carrying its own inline copies — this module is the only
implementation, live and research both read it.
tests/test_r20_research_validity.py::TestSharedExitPolicy still EXECUTES the
live _get_take_profit_targets and compares it against this module as a
regression guard against a future hand-copy creeping back in.

This module must stay pure and import-light (no pandas/numpy/broker imports):
it is imported by both the live exit path and the research engine.
"""

from __future__ import annotations

# Flat (non-scaled) targets: +100% long, +50% of credit short. Live uses these
# whenever exit.time_decay_scaling is false or DTE is unknown
# (portfolio.py _get_take_profit_targets).
DEFAULT_TAKE_PROFIT: tuple[float, float] = (1.0, 0.50)

# DTE-laddered (long_target, short_target) — mirrors live portfolio.py
# _get_take_profit_targets: as DTE shrinks, take profit earlier before theta
# accelerates. Rows are (dte_floor_exclusive, long, short): the first row
# whose floor is strictly below the DTE applies.
Ladder = tuple[tuple[int, float, float], ...]

# Floor of the catch-all last row: every DTE is strictly above it.
TERMINAL_DTE_FLOOR: int = -(10 ** 9)

CREDIT_TP_LADDER: Ladder = (
    (20, 1.00, 0.50),   # DTE > 20
    (10, 0.75, 0.40),   # DTE 11-20
    (5,  0.50, 0.30),   # DTE 6-10
    (TERMINAL_DTE_FLOOR, 0.25, 0.20),  # DTE <= 5 — very aggressive, grab what you can
)

# A short target no credit position can reach (P&L tops out at 100% of
# credit): research uses it for the "hold to the DTE exit" variant.
NO_TAKE_PROFIT: float = float("inf")

# Live rule 3a (portfolio.py): close ANY position at DTE <= this many days.
# Paired with the config-side 14-DTE entry floor so every position gets >= 9
# calendar days of theta runway (R12-B cadence guard) — do not change one
# without revisiting the other.
EXPIRY_APPROACHING_DTE: int = 5

# Live rule 3d (portfolio.py) macro-event flatten windows, in days-to-event.
# PLAN 2026-08-04: defined-risk (condor family) EXEMPT — wings cap the
# surprise and the post-event vol crush is the payoff — so those strategies
# have no entry here (macro_flatten_window_days returns None). Undefined /
# assignment-risk strategies keep the early exit: strangle-class 5 days
# (avoid naked weekend gap risk pre-event), CSP/CC 1 day.
# jade_lizard is a BACKTEST-ONLY arm absent from live's list (live never
# trades it); it carries a naked short put, so it takes the strangle-class
# window (R16: it previously held through events, inflating its arm's PF on
# exactly the highest-variance days).
MACRO_FLATTEN_WINDOW_DAYS: dict[str, int] = {
    "short_strangle": 5,
    "jade_lizard": 5,
    "cash_secured_put": 1,
    "covered_call": 1,
}


def expiry_guard_applies(entry_dte: int, guard_dte: int | None) -> bool:
    """W74: do the DTE-based closes (live rules 3 and 3a: ``DTE <= EXPIRY_APPROACHING_DTE``
    and credit ``DTE <= 1``) apply to a position ENTERED with ``entry_dte`` days to expiry?

    They exist so a 14+ DTE entry leaves before gamma week. For an entry that
    starts inside that week they would close the position on the next tick, so
    short-DTE entries (``entry_dte <= guard_dte``) are governed by the hold
    fraction and the expiry-day flatten instead. ``guard_dte=None`` is today's
    behaviour: the rules apply to every position.
    """
    return guard_dte is None or entry_dte > guard_dte


def hold_limit_reached(entry_ts, now_ts, expiry_close_ts, fraction: float | None) -> bool:
    """W74: has the position been held for ``fraction`` of its life (entry -> expiry close)?

    ``fraction`` in (0, 1]; ``None`` disables the rule (the legacy day-count cap
    applies instead). Timestamps are tz-aware datetimes. A non-positive span
    (entered at/after the bell) is reported as reached so it can never linger.
    """
    if fraction is None:
        return False
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"max hold fraction must be in (0, 1], got {fraction}")
    span = (expiry_close_ts - entry_ts).total_seconds()
    if span <= 0:
        return True
    return (now_ts - entry_ts).total_seconds() >= float(fraction) * span


def expiry_day_flatten_due(now_et, expiry, minutes_before_close: int | None, close_et=None) -> bool:
    """W74: is it time to flatten a position because its expiry session is ending?

    True from ``minutes_before_close`` before that session's close, and for any
    time after the expiry date. ``None`` disables it. ``close_et`` is the
    session close as a ``datetime.time`` (13:00 on early-close days; default
    16:00). Hard backstop: it exists so no position reaches the settlement and
    assignment mechanics (pin risk, an unhedged gap after a one-sided expiry).
    """
    if minutes_before_close is None:
        return False
    from datetime import datetime as _dt, time as _time, timedelta as _td
    d = now_et.date()
    if d > expiry:
        return True
    if d < expiry:
        return False
    close = close_et if close_et is not None else _time(16, 0)
    cutoff = _dt.combine(expiry, close, tzinfo=now_et.tzinfo) - _td(minutes=int(minutes_before_close))
    return now_et >= cutoff


def validate_ladder(ladder) -> Ladder:
    """Normalise a research ladder to tuples and refuse malformed ones.

    Accepts lists (a ladder read back from JSON). Rows are (dte_floor, long,
    short) with floors strictly descending and a terminal TERMINAL_DTE_FLOOR
    row, so every DTE resolves to a row instead of silently falling through
    to DEFAULT_TAKE_PROFIT.
    """
    rows = tuple((int(f), float(lt), float(st)) for f, lt, st in ladder)
    if not rows:
        raise ValueError("take-profit ladder is empty")
    floors = [f for f, _, _ in rows]
    if any(a <= b for a, b in zip(floors, floors[1:])):
        raise ValueError(f"take-profit ladder floors must strictly descend: {floors}")
    if floors[-1] > TERMINAL_DTE_FLOOR:
        raise ValueError(
            f"take-profit ladder needs a terminal row with floor {TERMINAL_DTE_FLOOR}; "
            f"DTE <= {floors[-1]} would fall through")
    if any(lt <= 0 or st <= 0 for _, lt, st in rows):
        raise ValueError("take-profit targets must be positive")
    return rows


def take_profit_targets(
    dte: int | None,
    time_decay_scaling: bool = True,
    ladder: Ladder | None = None,
) -> tuple[float, float]:
    """(long_target, short_target) — mirrors live _get_take_profit_targets.

    time_decay_scaling=False or unknown DTE collapses to the flat defaults,
    exactly as live does (ExitConfig.time_decay_scaling).

    ``ladder`` is a RESEARCH override (scripts/take_profit_study.py via
    WalkForwardConfig.credit_tp_ladder); live never passes it, so None always
    means CREDIT_TP_LADDER. Callers are expected to pass a validate_ladder()
    result.
    """
    if not time_decay_scaling or dte is None:
        return DEFAULT_TAKE_PROFIT
    for floor, long_t, short_t in (CREDIT_TP_LADDER if ladder is None else ladder):
        if dte > floor:
            return long_t, short_t
    return DEFAULT_TAKE_PROFIT  # unreachable — terminal floor is -1e9


def credit_take_profit_pct(
    dte: int | None,
    time_decay_scaling: bool = True,
    ladder: Ladder | None = None,
) -> float:
    """SHORT-side take-profit target as a fraction of credit received."""
    return take_profit_targets(dte, time_decay_scaling, ladder)[1]


def macro_flatten_window_days(strategy: str | None) -> int | None:
    """Days-to-event inside which the strategy is flattened pre macro event.

    None = no macro flatten for this strategy (defined-risk holds through).
    """
    if not strategy:
        return None
    return MACRO_FLATTEN_WINDOW_DAYS.get(strategy)

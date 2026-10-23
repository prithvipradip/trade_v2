"""W84: the research twin of live's ex-dividend ENTRY block (W51).

Live (``bot/orchestrator.py``, ``trade_blocked_ex_div``) refuses a new position that SELLS
A CALL when the projected next ex-dividend date falls between today and the chosen expiry
(``DividendCalendar.would_hold_through_ex_div``): a short call can be assigned early the
day before an ex-date. The research engine had no twin, so every research reference
number included trades live refuses (PLAN.md, 2026-10-16 ex-dividend finding: 14-17% of
sessions, 22-39% of the reference P&L at entry DTE 14).

Modes (``backtest.ex_div_entry_gate``):

* ``off``          no block.
* ``live``         live's rule exactly: ONE projected ex-date from the ex-dates known before
                   the entry day (``dividends.project_next_ex_div``, the function live calls).
* ``exact``        any real ex-date in [entry, expiry].
* ``last_k_days``  a real ex-date in [entry, expiry] that is at most K days before expiry
                   (the end of a short call's life, where its extrinsic can fall below the
                   dividend: the F2 candidate).

Pure functions, no IO: the engine, the A/B script and the tests share one definition.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from datetime import date

from ait.data.dividends import project_next_ex_div

MODE_OFF = "off"
MODE_LIVE = "live"
MODE_EXACT = "exact"
MODE_LAST_K = "last_k_days"
MODES = (MODE_OFF, MODE_LIVE, MODE_EXACT, MODE_LAST_K)


def sells_a_call(pos: dict) -> bool:
    """Does a BUILT engine position sell a call? Live's ``_signal_sells_a_call`` on the
    engine's position shape: every multi-leg structure records its short call as
    ``short_call_strike``; a vertical spread (``_finalize_spread_position``) records
    ``short_strike`` with ``option_type`` 'call' (a bull call spread sells the upper call)."""
    if pos.get("short_call_strike") is not None:
        return True
    return pos.get("short_strike") is not None and pos.get("option_type") == "call"


def ex_dates_in_window(ex_dates: Sequence[date], start: date, end: date) -> list[date]:
    """The real ex-dates in [start, end] (both inclusive, as live's check). Ascending input."""
    i = bisect_left(ex_dates, start)
    out = []
    while i < len(ex_dates) and ex_dates[i] <= end:
        out.append(ex_dates[i])
        i += 1
    return out


def ex_div_blocks(mode: str, ex_dates: Sequence[date], entry: date, expiry: date,
                  k_days: int | None = None) -> tuple[bool, date | None]:
    """(blocked, the ex-date that decides it) for an entry on ``entry`` expiring ``expiry``.

    ``ex_dates`` are ALL stored ex-dates for the symbol, ascending. ``live`` uses only the
    ones before ``entry`` (what live's fetch could have returned on that day) and checks the
    single projected date, as live does; the other modes use the real dates. The returned
    date is the projected one in ``live`` mode (blocked or not), else the first real ex-date
    in the window (or the K-window), or None.
    """
    if k_days is None:      # W90e: backtest.ex_div_entry_k_days of the loaded settings (was a 5 literal repeating it)
        try:
            from ait.config.settings import load_settings
            k_days = int(load_settings().backtest.ex_div_entry_k_days)
        except Exception:  # noqa: BLE001 - no config.yaml -> the schema default
            from ait.config.settings import BacktestConfig
            k_days = int(BacktestConfig.model_fields["ex_div_entry_k_days"].default)
    if mode not in MODES:
        raise ValueError(f"unknown ex_div_entry_gate mode {mode!r}; expected one of {MODES}")
    if mode == MODE_OFF:
        return False, None
    if mode == MODE_LIVE:
        known = ex_dates[:bisect_left(ex_dates, entry)]
        projected = project_next_ex_div(known, entry)
        if projected is None:
            return False, None
        return entry <= projected <= expiry, projected
    in_window = ex_dates_in_window(ex_dates, entry, expiry)
    if mode == MODE_EXACT:
        return (True, in_window[0]) if in_window else (False, None)
    late = [d for d in in_window if (expiry - d).days <= int(k_days)]
    return (True, late[0]) if late else (False, None)

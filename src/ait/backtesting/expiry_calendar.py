"""W74: which expiry does an option "N days out" actually have?

The engine used ``expiry = entry + N calendar days``, which can be a Saturday
or a holiday: no listed option expires there, and the position's remaining life
(and every DTE-keyed rule) was off by up to three days. Listed expiries are
session dates. QQQ/SPY/IWM list daily expiries, so the contract that is "N days
out" is the first session on or after ``entry + N``.

``listed`` lets a caller pass the REAL set of listed expiry dates (the option
bars carry them, W75); without it the NYSE-session rule above applies. Pure and
read-only.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

from ait.utils.time import next_session_on_or_after


def listed_expiry(entry: date, target_dte: int, listed: Iterable[date] | None = None) -> date:
    """First listed expiry on or after ``entry + target_dte`` days.

    With ``listed`` the answer is the earliest listed date >= the target; if the
    set has none that far out it falls back to the session rule (so a short
    bar history never crashes a study). Without ``listed``: the next NYSE
    session on or after the target date.
    """
    if target_dte < 0:
        raise ValueError(f"target_dte must be >= 0, got {target_dte}")
    target = entry + timedelta(days=int(target_dte))
    if listed is not None:
        later = sorted(d for d in listed if d >= target)
        if later:
            return later[0]
    return next_session_on_or_after(target)


def actual_dte(entry: date, target_dte: int, listed: Iterable[date] | None = None) -> int:
    """Calendar days from ``entry`` to the expiry ``listed_expiry`` selects."""
    return (listed_expiry(entry, target_dte, listed) - entry).days

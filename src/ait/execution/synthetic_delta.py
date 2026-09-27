"""W55 (synthetic per-leg delta for the monitor loop): compute a real,
per-leg delta (and the rest of the Greeks) for an OPEN position from the ONE
live price source the exit/monitor loop actually has --
`PortfolioManager._get_option_marks()`'s `marketPrice`, no bid/ask, no IV.

Nothing calls into this module yet. Rule 3b (delta_breach, portfolio.py)
stays dead -- it needs `ib.ticker(contract).modelGreeks`, a live market-data
subscription the exit loop never holds, and this module does not revive it.
This is the operator-chosen alternative: adapt the same py_vollib
IV-solve-then-Greeks pattern `data/options_chain.py` already runs at
entry-time chain scanning (`solve_iv_from_price`/`compute_greeks_from_sigma`,
extracted there as shared functions this wave) to the monitor loop's single
live price instead of a chain snapshot's bid/ask mid. A future wave (W56, or
a revived rule 3b) is the first real caller.

Unlike the chain path, there is no chain-level median IV to fall back on for
a single leg outside a chain -- ANY failure (missing mark, non-positive
price, solve failure, arbitrage exception) returns None, never a guessed
value. Matches `_get_position_delta`'s own existing None-on-failure
convention (portfolio.py:1117-1166) exactly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

from ait.bot.state import TradeRecord
from ait.data.options_chain import (
    _risk_free_rate,
    compute_greeks_from_sigma,
    solve_iv_from_price,
)
from ait.utils.logging import get_logger
from ait.utils.time import today_et

log = get_logger("execution.synthetic_delta")


@dataclass
class ContractGreeks:
    """A synthetic Greeks solve for one leg, from its live monitor-loop mark."""

    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float


def _expiry_key(expiry: str) -> str:
    """Same normalization `_get_option_marks()`/`_option_position_unrealized`
    use (portfolio.py:322,370) -- YYYYMMDD. A caller that doesn't replicate
    this exactly silently misses every lookup; kept as its own function so it
    can only be wrong in one place."""
    return expiry.replace("-", "")[:8]


def synthetic_leg_greeks(
    option_marks: dict[tuple, float], symbol: str, expiry: str, strike: float,
    right: str, spot: float,
) -> ContractGreeks | None:
    """Solve one leg's IV and Greeks from its live monitor-loop mark.

    `expiry` is the leg's own ISO date string (`trade.legs`' convention, per
    W52/W53 -- NOT necessarily `trade.expiry`, which is roll-unaware). Never
    raises; returns None on a missing mark, a non-positive price (the one
    guard `_get_option_marks()` itself doesn't apply -- it only filters
    None/NaN), a malformed expiry, or any solve failure.
    """
    key = (symbol, _expiry_key(expiry), float(strike), right)
    price = option_marks.get(key)
    if price is None or price <= 0:
        return None
    try:
        expiry_date = date.fromisoformat(expiry)
    except ValueError:
        log.warning("synthetic_delta_bad_expiry", symbol=symbol, expiry=expiry)
        return None

    # Same whole-calendar-day convention OptionsChain.dte/OptionContract.dte
    # use (options_chain.py:302-303,433-434) -- consistency with the audited
    # pattern being adapted, not a new time-resolution scheme. Finer
    # intraday precision near DTE<=1 (where this matters most) is a real
    # idea, registered as a follow-up, not built here.
    dte = (expiry_date - today_et()).days
    t = max(dte / 365.0, 0.001)
    flag = "c" if right.upper().startswith("C") else "p"

    iv, outcome = solve_iv_from_price(price, spot, strike, t, _risk_free_rate(), flag)
    if outcome != "solved" or iv is None:
        return None
    greeks = compute_greeks_from_sigma(spot, strike, t, _risk_free_rate(), iv, flag)
    if greeks is None:
        return None
    d, g, th, v = greeks
    return ContractGreeks(iv=iv, delta=d, gamma=g, theta=th, vega=v)


def synthetic_vertical_deltas(
    trade: TradeRecord, side: str, option_marks: dict[tuple, float], spot: float,
) -> dict[str, ContractGreeks | None]:
    """Synthetic Greeks for BOTH legs of `trade`'s at-risk vertical on `side`
    ("P" or "C") -- the short and its paired long, same legs-JSON idiom W52's
    `_short_strikes_from_legs` and W53's `_at_risk_vertical`
    (roll_selector.py:41-61) use. Returns `{"short": ..., "long": ...}`,
    either value None if that leg's own solve failed or the vertical isn't a
    clean single short+long pair. Never raises.
    """
    result: dict[str, ContractGreeks | None] = {"short": None, "long": None}
    try:
        legs = json.loads(trade.legs) if trade.legs else []
    except (ValueError, TypeError):
        return result

    def _side_legs(action: str) -> list[dict]:
        return [
            leg for leg in legs
            if str(leg.get("action", "")).upper() == action
            and str(leg.get("right", leg.get("type", ""))).upper().startswith(side)
        ]

    shorts, longs = _side_legs("SELL"), _side_legs("BUY")
    if len(shorts) != 1 or len(longs) != 1:
        return result   # not a clean single short+long pair -- same
                         # invariant _at_risk_vertical enforces

    short_leg, long_leg = shorts[0], longs[0]
    result["short"] = synthetic_leg_greeks(
        option_marks, trade.symbol, str(short_leg.get("expiry", trade.expiry)),
        float(short_leg["strike"]), side, spot)
    result["long"] = synthetic_leg_greeks(
        option_marks, trade.symbol, str(long_leg.get("expiry", trade.expiry)),
        float(long_leg["strike"]), side, spot)
    return result

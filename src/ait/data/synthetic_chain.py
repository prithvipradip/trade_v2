"""W31: a deterministic Black-Scholes option chain for the sizing self-test.

NOT A MARKET-DATA SOURCE. Nothing here fetches, caches or approximates a real
chain, and no entry, exit or backtest path may read it. It exists for exactly
one caller — the launch cap-grid self-test in `bot/orchestrator.py` — which
needs a chain it can hand to the REAL `IronCondor.generate_signals`.

Why it exists at all (R26 F4, `logic-entry-01` residual): the self-test used to
call the builder's private `_vol_scaled_width` with a $1 underlying price and
reason about the number it returned. That probe drives the vol term under the
builder's own `max(2.0, ...)` floor, so it returned the NARROWEST structure the
builder can ever make — `builder_width` was bit-identical at $9,750 and
$201,475 — and nothing downstream of width selection ran at all. Strike-grid
snapping was unmodelled and `max_loss_over_budget` had no arm, so on a $5 grid
the check reported `problems=0` while the builder emitted zero signals.

A model of the builder cannot catch the builder rejecting things. So: build a
chain, run the real thing, and report what it actually did.

Kept in `data/` because an options chain is a data fact (one owner per fact) —
not a method on the orchestrator, which could not then be tested without a
broker, and not on `IronCondor`, which would tie chain construction to one
strategy. `backtesting.pricing` is a pure numpy/scipy leaf with no `ait`
imports, so reusing its Black-Scholes primitives here adds no cycle and avoids
a second copy of the pricing math.
"""

from __future__ import annotations

import math
from datetime import timedelta

from ait.backtesting.pricing import OptionType, black_scholes_price, bs_delta
from ait.data.options_chain import OptionsChain, OptionContract
from ait.utils.time import today_et

#: Marks every contract and chain this module builds, so a probe can never be
#: mistaken for a real feed in telemetry.
PROBE_SOURCE = "synthetic_probe"

#: Floor for a probe contract's mid price. Deep-OTM wings priced at fractions
#: of a cent have no meaningful two-sided market, and a contract that fails
#: `is_liquid` is dropped before the builder can select it — which would make
#: the self-test measure this module's pricing rather than the cap grid. The
#: floor RAISES what a bought wing costs, so it understates net credit: the
#: conservative direction for a check that asks "can this account trade?".
MIN_PROBE_PRICE = 0.05


def build_probe_chain(
    *,
    underlying_price: float,
    implied_vol: float,
    dte: int,
    strike_grid: float,
    risk_free_rate: float,
    symbol: str = "PROBE",
) -> OptionsChain:
    """A liquid, fully-priced chain on a `strike_grid`-spaced ladder.

    Deterministic and IO-free: the same arguments always give the same chain,
    which is what lets the self-test's verdict be a fact about the cap grid
    rather than about the day's market.

    `strike_grid` is the point of the whole exercise. The builder snaps its
    target wing to real strikes, so a $5 ladder can only express a structure
    $5 wider than the one the width formula asked for — which is how a budget
    that looks sufficient produces `max_loss_over_budget` and zero trades.
    """
    if underlying_price <= 0 or implied_vol <= 0 or dte <= 0 or strike_grid <= 0:
        raise ValueError(
            f"probe chain needs positive inputs, got price={underlying_price} "
            f"iv={implied_vol} dte={dte} grid={strike_grid}"
        )

    expiry = today_et() + timedelta(days=dte)
    t = dte / 365.0
    sigma_move = underlying_price * implied_vol * math.sqrt(t)

    # Span far enough that both 0.20-delta shorts AND their wings exist on the
    # ladder. 3 sigma covers the shorts (~0.85 sigma) with room for wings; the
    # 10-grid term keeps a coarse ladder from collapsing to a few strikes.
    span = max(3.0 * sigma_move, 10.0 * strike_grid)
    atm = round(underlying_price / strike_grid) * strike_grid
    steps = int(span / strike_grid)

    calls: list[OptionContract] = []
    puts: list[OptionContract] = []
    for i in range(-steps, steps + 1):
        strike = round(atm + i * strike_grid, 2)
        if strike <= 0:
            continue
        for right, opt_type, bucket in (
            ("C", OptionType.CALL, calls),
            ("P", OptionType.PUT, puts),
        ):
            price = max(
                MIN_PROBE_PRICE,
                black_scholes_price(
                    underlying_price, strike, t, risk_free_rate, implied_vol, opt_type
                ),
            )
            # A tight PROPORTIONAL spread: an absolute one would read as
            # illiquid on cheap wings (spread_pct = spread / mid) and silently
            # shrink the ladder the builder gets to choose from.
            bid = round(price * 0.99, 2)
            ask = round(price * 1.01, 2)
            if ask <= bid:  # rounding collapse at the price floor
                ask = round(bid + 0.01, 2)
            bucket.append(
                OptionContract(
                    symbol=symbol,
                    expiry=expiry,
                    strike=strike,
                    right=right,
                    bid=bid,
                    ask=ask,
                    last=round(price, 2),
                    # Comfortably clear of options.min_volume / min_open_interest
                    # so the probe never measures the liquidity gate by accident.
                    volume=1_000,
                    open_interest=1_000,
                    implied_vol=implied_vol,
                    delta=bs_delta(
                        underlying_price, strike, t, risk_free_rate, implied_vol, opt_type
                    ),
                    con_id=0,
                    source=PROBE_SOURCE,
                )
            )

    # atm_iv / expected_move are derived by OptionsChain.__post_init__ from the
    # contracts above, exactly as they are for a real chain. iv_is_guessed stays
    # False — every contract carries a real IV, so the builder's degraded-greeks
    # gate must not fire on a probe.
    return OptionsChain(
        symbol=symbol,
        underlying_price=underlying_price,
        expiry=expiry,
        calls=calls,
        puts=puts,
        source=PROBE_SOURCE,
    )

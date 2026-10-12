"""W77: what happens to an option structure that reaches its expiry bell.

Research only (the live reconciler books expiry from IB's own statement; see
``PositionReconciler._structure_intrinsic``, which this module is parity-tested against).
Pure functions, no engine or broker imports, so the rules are exercised directly.

The mechanics modelled (assumption A5 of the short-DTE programme, measured against three
real manual-trade expiries):

* OCC exercise-by-exception: an option ITM by ``EXERCISE_THRESHOLD_USD`` or more at the
  close is exercised / assigned automatically. A long leg that qualifies is EXERCISED
  (the holder is us); a short leg that qualifies is ASSIGNED.
* A short leg within ``pin_band`` of its strike is a PIN: whether it is assigned is not
  knowable at the bell (after-hours moves, the holder's choice), so both outcomes are
  scored and the WORSE one for us is booked.
* An option that is neither exercised nor assigned EXPIRES: no trade, no fee.
* Exercise and assignment deliver stock. A structure whose delivered shares do not net to
  zero (short ITM, its protective long OTM) is left holding shares overnight; they are
  marked at the NEXT session's open. A structure that nets to zero (both wings ITM)
  settles at intrinsic with no gap.

Sign convention: "cost" is what we pay to settle the structure per option-share, the same
convention as the engine's credit ``current_value`` (P&L = (credit - cost) x 100 x
contracts), so a settled position drops straight into ``_calc_pnl``'s credit formula.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

# OCC exercise-by-exception threshold: an exchange rule, not a tunable, hence a constant.
EXERCISE_THRESHOLD_USD = 0.01

EXPIRED = "expired"
EXERCISED = "exercised"
ASSIGNED = "assigned"
PIN = "pin"


@dataclass(frozen=True)
class SettleLeg:
    strike: float
    right: str    # "C" | "P"
    side: str     # "long" | "short"

    def itm_amount(self, S: float) -> float:
        """How far in the money the leg is at underlying ``S`` (negative when OTM)."""
        return (S - self.strike) if self.right == "C" else (self.strike - S)


@dataclass(frozen=True)
class StockLeg:
    """One block of delivered stock, per option-share: ``q`` = +1 bought / -1 sold at ``K``."""
    q: int
    K: float
    assigned: bool   # True = we were assigned (fee applies); False = we exercised


@dataclass(frozen=True)
class Resolution:
    cost: float            # per option-share, intrinsic / stock mark-to-open
    fees: float            # per option-share: assignment fee + stock close-out commission
    net_shares: int        # per contract-unit, signed (+ long stock)
    statuses: tuple[str, ...]   # one per leg, pins resolved to the chosen scenario
    stock: tuple[StockLeg, ...]

    @property
    def total(self) -> float:
        return self.cost + self.fees


def settle_status(leg: SettleLeg, S: float, pin_band: float) -> str:
    """``expired | exercised | assigned | pin`` for one leg at settlement price ``S``."""
    itm = leg.itm_amount(S)
    if leg.side == "long":
        return EXERCISED if itm >= EXERCISE_THRESHOLD_USD else EXPIRED
    if abs(itm) <= pin_band:
        return PIN
    return ASSIGNED if itm >= EXERCISE_THRESHOLD_USD else EXPIRED


def settle_legs(legs: list[SettleLeg], S: float, pin_band: float) -> list[str]:
    return [settle_status(l, S, pin_band) for l in legs]


def scenarios(statuses: list[str]) -> list[tuple[str, ...]]:
    """Every resolution of the ``pin`` legs to assigned / expired (one tuple per scenario;
    a structure with no pins has exactly one)."""
    pins = [i for i, s in enumerate(statuses) if s == PIN]
    out: list[tuple[str, ...]] = []
    for combo in product((ASSIGNED, EXPIRED), repeat=len(pins)):
        cur = list(statuses)
        for i, c in zip(pins, combo):
            cur[i] = c
        out.append(tuple(cur))
    return out


def stock_legs(legs: list[SettleLeg], statuses: tuple[str, ...] | list[str]) -> tuple[StockLeg, ...]:
    """Shares delivered by a resolved scenario.

    short call assigned -> sold stock at K (q=-1); short put assigned -> bought (q=+1);
    long call exercised -> bought (q=+1); long put exercised -> sold (q=-1)."""
    out: list[StockLeg] = []
    for leg, st in zip(legs, statuses):
        if st == ASSIGNED:
            out.append(StockLeg(-1 if leg.right == "C" else +1, leg.strike, True))
        elif st == EXERCISED:
            out.append(StockLeg(+1 if leg.right == "C" else -1, leg.strike, False))
    return tuple(out)


def net_shares(stock: tuple[StockLeg, ...]) -> int:
    return sum(s.q for s in stock)


def settlement_cost(stock: tuple[StockLeg, ...], O: float) -> float:
    """Per option-share cost to settle: the delivered stock marked at ``O``.

    With net shares 0 the O terms cancel, leaving the pure intrinsic cost (the same number
    ``PositionReconciler._structure_intrinsic`` returns for the structure)."""
    return -sum(s.q * (O - s.K) for s in stock)


def unit_fees(stock: tuple[StockLeg, ...], assignment_fee: float, stock_commission: float) -> float:
    """Per option-share: the fee on shares we were ASSIGNED (measured: exercised delivery was
    free on the operator's statement) plus the commission to close any net stock."""
    assigned_shares = sum(1 for s in stock if s.assigned)
    return assignment_fee * assigned_shares + stock_commission * abs(net_shares(stock))


def needs_overnight(legs: list[SettleLeg], S: float, pin_band: float) -> bool:
    """True when ANY scenario leaves net shares, i.e. the result depends on the next open."""
    sts = settle_legs(legs, S, pin_band)
    return any(net_shares(stock_legs(legs, sc)) != 0 for sc in scenarios(sts))


def worst_case(legs: list[SettleLeg], S: float, pin_band: float, O: float,
               assignment_fee: float, stock_commission: float) -> Resolution:
    """The scenario that costs us most, marking delivered stock at ``O``.

    ``S`` decides who is exercised / assigned (the settlement price); ``O`` is where the
    resulting shares are closed (the next open; ``O == S`` when nothing is left over)."""
    best: Resolution | None = None
    for sc in scenarios(settle_legs(legs, S, pin_band)):
        stock = stock_legs(legs, sc)
        res = Resolution(cost=settlement_cost(stock, O),
                         fees=unit_fees(stock, assignment_fee, stock_commission),
                         net_shares=net_shares(stock), statuses=sc, stock=stock)
        if best is None or res.total > best.total:
            best = res
    assert best is not None   # scenarios() always yields at least one
    return best

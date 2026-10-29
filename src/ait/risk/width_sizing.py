"""The ONE budget-to-width rule for defined-risk verticals, shared by the live builders and the research engine.

W108 phase 4a. Only the iron condor fitted its width to the per-trade budget (``IronCondor.affordable_width``, W31/W64):
the live ``bull_call_spread`` / ``bear_put_spread`` builders never read ``risk_budget`` and research sized the credit-spread
wing as ``wing_k x expected move x 0.5`` with no cap, so the lot rule (``ait.risk.lot_sizing``) simply refused what the
builder made. This module is the only place the budget-to-width arithmetic lives; like ``lot_sizing`` it imports nothing
from the broker, the strategies or the engine, so ``strategies/`` and ``backtesting/`` can both depend on it.

The "affordable width" depends on what a lot's max loss IS, so there are two rules:

* **Credit structures** (iron condor; put / call credit spread). Max loss per lot = (width - credit) x 100. Before pricing,
  the only bound on the credit is the one the builder's own gate enforces, ``credit >= min_credit_ratio x width``, so the
  worst case is ``width x (1 - ratio) x 100 <= budget``, i.e. ``width <= budget / (100 x (1 - ratio))``. The condor passes
  its credit/width floor (``AIT_IC_MIN_CREDIT_WIDTH``: budget / 90); a builder with no such floor (research's credit spreads)
  passes 0 (budget / 100). This is W31's formula, unchanged.
* **Debit structures** (bull call / bear put spread). Max loss per lot = the DEBIT x 100, not the width. The only bound
  before pricing is no-arbitrage (a vertical's debit never exceeds its width), which is loose by 2-3x for these spreads
  (the debit is ~35-50% of the width): a width cap built on it would narrow spreads whose real max loss already fits. So
  the debit rule is priced: the builder hands over the debit of each listed width and gets back the widest that fits.

The budget is ``risk.max_position_risk_pct x equity`` on both sides (live: ``Strategy.risk_budget``, research: capital x
``position_size_pct``). A width only ever NARROWS: a structure the budget already carries is returned unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

from ait.config.invariants import IC_BUDGET_CAPPED_MIN_WING  # W70: the narrowest budget-capped wing ($1)

SHARES_PER_CONTRACT = 100.0
# W31 (iron_condor.py): the worst-case loss share never drops under one half, so a mis-set credit/width floor (say 0.9)
# cannot more than double the width a budget buys.
MIN_LOSS_SHARE = 0.5
# Float noise in budget arithmetic (e.g. 5000 x 0.05 = 250.00000000000003): a listed width this close to the cap fits.
_WIDTH_TOLERANCE = 1e-9


def _no_budget(risk_budget: float | None) -> bool:
    """True when there is no constraint (``Strategy.risk_budget`` None = the paper big-account convention)."""
    if risk_budget is None:
        return True
    try:
        b = float(risk_budget)
    except (TypeError, ValueError):
        return True
    return math.isnan(b) or b <= 0


def _credit_loss_share(min_credit_ratio: float) -> float:
    return max(MIN_LOSS_SHARE, 1.0 - float(min_credit_ratio))


def affordable_width(risk_budget: float, *, min_credit_ratio: float = 0.0) -> float:
    """Widest CREDIT-structure width whose worst-case max loss fits ``risk_budget`` (before any snapping).

    ``min_credit_ratio`` is the credit/width floor the builder ENFORCES before it accepts a structure (0 when it has
    none). Pass the same value on the live and the research side, or the two build different widths.
    """
    return float(risk_budget) / (SHARES_PER_CONTRACT * _credit_loss_share(min_credit_ratio))


def budget_for_width(width: float, *, min_credit_ratio: float = 0.0) -> float:
    """The per-trade budget that carries a CREDIT-structure ``width``: the exact inverse of ``affordable_width``."""
    return float(width) * SHARES_PER_CONTRACT * _credit_loss_share(min_credit_ratio)


def whole_dollar_widths(natural_width: float) -> list[float]:
    """The research engine's strike grid as widths: $1 steps (``find_strike_by_delta`` rounds strikes to whole dollars)
    from ``IC_BUDGET_CAPPED_MIN_WING`` up to the natural width."""
    if natural_width is None or math.isnan(float(natural_width)):
        return []
    top = int(math.floor(float(natural_width) + _WIDTH_TOLERANCE))
    lo = int(IC_BUDGET_CAPPED_MIN_WING)
    return [float(w) for w in range(lo, top + 1)]


def fitted_credit_width(
    natural_width: float,
    risk_budget: float | None,
    listed_widths: Iterable[float],
    *,
    min_credit_ratio: float = 0.0,
) -> float:
    """The width a CREDIT spread is built at.

    ``natural_width`` is what the builder uses with no budget (research: ``wing_k x expected move x fraction``, floored at
    the wing floor). No budget, or a natural width the budget carries: unchanged. Otherwise the widest listed width at or
    below ``affordable_width`` (so the worst-case guarantee survives the snap); when none is listed, the narrowest listed
    width, and the builder's lot rule refuses it if its real max loss does not fit (as the condor's own re-check does).
    """
    if _no_budget(risk_budget):
        return natural_width
    cap = affordable_width(float(risk_budget), min_credit_ratio=min_credit_ratio)
    if natural_width <= cap + _WIDTH_TOLERANCE:
        return natural_width
    listed = sorted(float(w) for w in listed_widths if 0 < float(w) < natural_width)
    if not listed:
        return natural_width
    fitting = [w for w in listed if w <= cap + _WIDTH_TOLERANCE]
    return fitting[-1] if fitting else listed[0]


def fitted_debit_width(
    natural_width: float,
    risk_budget: float | None,
    debit_by_width: Mapping[float, float],
) -> float | None:
    """The width a DEBIT spread is built at, or None when no listed width fits the budget (the builder refuses).

    ``debit_by_width`` maps each listed width (long strike to short strike) to its priced debit per share, as the builder's
    own budget gate will see it (live: the chain mid; research: the model price with its slippage). The natural width
    (the delta-placed short strike) is kept when there is no budget or when its own debit x 100 fits; otherwise the widest
    narrower width whose positive debit x 100 fits.
    """
    if _no_budget(risk_budget):
        return natural_width
    budget = float(risk_budget)
    natural_debit = debit_by_width.get(natural_width)
    if natural_debit is not None and natural_debit * SHARES_PER_CONTRACT <= budget + _WIDTH_TOLERANCE:
        return natural_width
    fitting = [float(w) for w, d in debit_by_width.items()
               if 0 < float(w) < natural_width and d is not None and not math.isnan(float(d))
               and 0 < float(d) and float(d) * SHARES_PER_CONTRACT <= budget + _WIDTH_TOLERANCE]
    return max(fitting) if fitting else None

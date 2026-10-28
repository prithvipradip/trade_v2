"""The ONE lot-count rule, shared by the live position sizer and every research sizing site.

W108. Live (``risk/position_sizer.py``) and research (``backtesting/engine.py`` builders and
``backtesting/pool.py``) used to count lots with different code: live floored to 1 and scaled
the budget by six multipliers, research divided the budget by the risk of one lot and, for
spreads, opened one lot anyway when it cost up to 25% of capital. A research result about a
lot count therefore described something live could not reproduce. This module is the only
place the arithmetic lives; it imports nothing from the broker, the strategies or the engine,
so ``risk/`` and ``backtesting/`` can both depend on it.

The rule (operator decision 2026-10-27, "option A"):

* **Viability is decided by the UNSCALED budget.** ``budget = equity * budget_pct``
  (``risk.max_position_risk_pct``, the same number the condor builder is funded with and gate 6b
  enforces). One lot whose max loss exceeds it is refused: zero lots, the caller rejects.
* **The count is the budget divided by the risk of one lot**, never below 1 once viable.
* **A scaler (confidence, volatility, ...) only throttles the count.** It multiplies the budget
  in the count, never in the viability test, so it cannot turn a viable structure into a refusal.
  It is clamped into [0, 1]: a product above 1 (the IV-rank and long-straddle factors reach 1.2)
  never raises the count above floor(budget / risk), which is also all the per-trade gate would
  allow live. ``scaler`` is 1.0 for every strategy not listed in ``position_sizing.scaled_strategies``
  (the list is empty today); the factors live in ``ait.risk.scalers``.
* ``max_lots`` (``positions.max_contracts_per_trade``) caps the result.
"""

from __future__ import annotations

import math


def lots_for_budget(
    risk_per_lot: float | None,
    equity: float,
    budget_pct: float,
    *,
    max_lots: int | None = None,
    scaler: float = 1.0,
) -> int:
    """Whole lots a per-trade budget carries, or 0 when one lot does not fit it.

    ``risk_per_lot`` is the dollars one lot can lose (a credit structure's width minus its
    credit, times 100; a debit structure's debit, times 100; a naked structure's margin
    proxy). ``None``, zero, negative or NaN means the risk is unknown: nothing is sized.
    """
    if risk_per_lot is None:
        return 0
    try:
        risk = float(risk_per_lot)
        budget = float(equity) * float(budget_pct)
    except (TypeError, ValueError):
        return 0
    if math.isnan(risk) or math.isnan(budget) or risk <= 0 or budget <= 0:
        return 0
    if risk > budget:
        return 0
    lots = max(1, int(math.floor(budget * min(1.0, max(0.0, float(scaler))) / risk)))
    if max_lots is not None:
        lots = min(lots, int(max_lots))
    return max(0, lots)

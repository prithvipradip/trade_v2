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
CREDIT_TP_LADDER: tuple[tuple[int, float, float], ...] = (
    (20, 1.00, 0.50),   # DTE > 20
    (10, 0.75, 0.40),   # DTE 11-20
    (5,  0.50, 0.30),   # DTE 6-10
    (-(10 ** 9), 0.25, 0.20),  # DTE <= 5 — very aggressive, grab what you can
)

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


def take_profit_targets(
    dte: int | None, time_decay_scaling: bool = True
) -> tuple[float, float]:
    """(long_target, short_target) — mirrors live _get_take_profit_targets.

    time_decay_scaling=False or unknown DTE collapses to the flat defaults,
    exactly as live does (ExitConfig.time_decay_scaling).
    """
    if not time_decay_scaling or dte is None:
        return DEFAULT_TAKE_PROFIT
    for floor, long_t, short_t in CREDIT_TP_LADDER:
        if dte > floor:
            return long_t, short_t
    return DEFAULT_TAKE_PROFIT  # unreachable — terminal floor is -1e9


def credit_take_profit_pct(dte: int | None, time_decay_scaling: bool = True) -> float:
    """SHORT-side take-profit target as a fraction of credit received."""
    return take_profit_targets(dte, time_decay_scaling)[1]


def macro_flatten_window_days(strategy: str | None) -> int | None:
    """Days-to-event inside which the strategy is flattened pre macro event.

    None = no macro flatten for this strategy (defined-risk holds through).
    """
    if not strategy:
        return None
    return MACRO_FLATTEN_WINDOW_DAYS.get(strategy)

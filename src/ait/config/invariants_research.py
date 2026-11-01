"""Fixed RESEARCH limits (W88a): values the research engine deliberately does not let a study configure.

``config/invariants.py`` is the live counterpart; the rule is the same (``feedback-visible-vs-tunable``): *visible*
is not *tunable*. Everything a researcher may vary lives in the research overlay file (or a study file) with bounds and
a stated consequence. The values below are different in kind: each is a CORRECTNESS constraint of the engine's own
arithmetic (a clamp the pricing code needs to stay inside the strike grid, a numerical tolerance), not a modelling
choice. A key in a YAML file is editable regardless of any comment, and the research overlay and study files are
edited by people and AI sessions, so putting these there would let one edit loosen both a limit and its check.

They are code, in ONE place, named, documented, logged once per research process (``research_fixed_limits_in_effect``),
stored in every run record, and listed in docs/RESEARCH_GUIDE.md "Are some values not configurable?" (generated; scripts/gen_docs.py --check keeps them in step).

HOW TO CHANGE ONE: edit this file, run scripts/gen_docs.py, record it in PLAN.md. A change moves research numbers.

THIS MODULE MUST STAY A LEAF: stdlib plus ``ait.config.invariants`` (itself a leaf). It does not name the research
overlay or any loader, so it may be imported from ``backtesting/`` and ``optimization/``.
"""
from __future__ import annotations

from ait.config.invariants import Invariant

# --- Strike-selection clamps (the dormant strangle / debit builders; condors use the wing builder) -------------
# The delta a builder asks the strike search for is IV-scaled and then clamped to this band: below the floor the
# strike is so far OTM the quote is meaningless, above the ceiling the leg is no longer "out of the money".
RESEARCH_DELTA_MIN = 0.05
RESEARCH_DELTA_MAX = 0.45
# The IV scaling factor ``iv_floor / iv`` is clamped to this band so one stale IV print cannot move a strike by
# more than half again.
IV_SCALE_MIN = 0.5
IV_SCALE_MAX = 1.5

# --- Numerical tolerances ----------------------------------------------------------------------------------------
# A settled position whose gross loss exceeds its DEFINED max loss by more than this many dollars is counted as
# ``over_max_loss`` (a sanity counter, not a gate).
SETTLEMENT_LOSS_TOLERANCE = 1e-6
# Probabilities are clipped to [eps, 1 - eps] before log-loss / calibration so log(0) cannot occur.
PROB_CLIP_EPS = 1e-7

# --- Research data contract (W116) -----------------------------------------------------------------------------
# The longest run of consecutive sessions with no stored implied vol that a study may contain. An isolated missing
# session becomes a counted no-entry day; a longer run (or a symbol with no stored IV at all) refuses the study.
RESEARCH_MAX_NULL_IV_RUN_SESSIONS = 3
# The longest gap between a symbol's last stored ex-dividend date and the end of a window before the stored dividends
# count as stale (the quarterly payers in the universe never exceed it).
RESEARCH_MAX_DIVIDEND_GAP_DAYS = 100
# The longest stretch after a macro-event type's last stored event for which the stored calendar still counts as current
# at the end of a window (the quarterly GDP release is the sparsest type in use).
RESEARCH_MAX_EVENT_GAP_DAYS = 120
# The longest stretch of calendar days inside a window (and at its end) without a stored observation of a FRED macro
# series (FRED skips weekends and bank holidays; the longest regular gap is four days over a long weekend).
RESEARCH_MAX_MACRO_GAP_DAYS = 5
# How far before the end of a window the newest stored macro observation may be: the broad dollar index is published
# about a week late, so the tail tolerance is wider than the gap tolerance inside a window.
RESEARCH_MAX_MACRO_TAIL_DAYS = 10
# The largest share of a run's surface-priced option legs that may fall back to the legacy single-IV model (a leg with
# no fitted smile for its expiry, or a strike outside the fitted range). The fallback is a declared model, not a
# refusal; above this share the run record flags the result as only partly surface-priced.
RESEARCH_MAX_SURFACE_LEG_MISS_FRACTION = 0.10

RESEARCH_INVARIANTS: tuple[Invariant, ...] = (
    Invariant(
        "RESEARCH_DELTA_MIN", RESEARCH_DELTA_MIN, "|delta|",
        "Lower end of the delta band a strike search may be asked for.",
        "Below it the strike is far enough OTM that a modelled quote is noise. Also the lower bound the optimizer "
        "accepts for delta_short / delta_long overrides (a wider range would search values the engine clamps).",
        "backtesting/engine.py (_build_debit_position, _build_credit_position); optimization/param_spaces.py HARD_BOUNDS"),
    Invariant(
        "RESEARCH_DELTA_MAX", RESEARCH_DELTA_MAX, "|delta|",
        "Upper end of the delta band a strike search may be asked for.",
        "Above it a leg is no longer out of the money; the structures built from it stop meaning what their names say.",
        "backtesting/engine.py; optimization/param_spaces.py HARD_BOUNDS"),
    Invariant(
        "IV_SCALE_MIN", IV_SCALE_MIN, "ratio",
        "Lower clamp of the iv_floor / iv scaling applied to a builder's delta.",
        "Keeps one stale or extreme IV print from moving a strike by more than a factor of two.",
        "backtesting/engine.py (_build_debit_position, _build_credit_position)"),
    Invariant(
        "RESEARCH_MAX_NULL_IV_RUN_SESSIONS", RESEARCH_MAX_NULL_IV_RUN_SESSIONS, "sessions",
        "Longest run of consecutive sessions with no stored implied vol that a study may contain; an isolated "
        "gap is a counted no-entry day, a longer run (or a symbol with no stored IV) refuses the study.",
        "It is the line the stored history is checked AGAINST: if a study file could raise it, a study could be "
        "told to run on weeks of invented IV, which is the substitution the data contract exists to stop.",
        "data/research_inputs.py (the implied-vol check); backtesting/preflight.py"),
    Invariant(
        "RESEARCH_MAX_DIVIDEND_GAP_DAYS", RESEARCH_MAX_DIVIDEND_GAP_DAYS, "days",
        "Longest gap after a symbol's last stored ex-dividend date for which the stored dividends still count as "
        "current for a window.",
        "Same reason: a stale dividend table silently turns the ex-dividend gate off at the tail of the window.",
        "data/research_inputs.py (the dividends check); backtesting/preflight.py"),
    Invariant(
        "RESEARCH_MAX_EVENT_GAP_DAYS", RESEARCH_MAX_EVENT_GAP_DAYS, "days",
        "Longest stretch after a macro-event type's last stored event for which the stored calendar still counts as "
        "current at the end of a window.",
        "Same reason: a calendar that ends early silently turns the macro gate off for the rest of the window.",
        "data/research_inputs.py (the official_econ_calendar check); backtesting/preflight.py"),
    Invariant(
        "RESEARCH_MAX_MACRO_GAP_DAYS", RESEARCH_MAX_MACRO_GAP_DAYS, "days",
        "Longest stretch of calendar days inside a window, or at its end, without a stored observation of a macro series.",
        "FRED publishes on business days; a longer hole means missing data, which would otherwise be carried forward "
        "as if it were the market.",
        "data/research_inputs.py (the macro_series check); data/macro_store.py"),
    Invariant(
        "RESEARCH_MAX_MACRO_TAIL_DAYS", RESEARCH_MAX_MACRO_TAIL_DAYS, "days",
        "How far before the end of a window the newest stored observation of a macro series may be.",
        "The broad dollar index is published about a week late, so a window ending today legitimately lags it; a "
        "longer lag means the backfill is stale.",
        "data/research_inputs.py (the macro_series check); data/macro_store.py"),
    Invariant(
        "RESEARCH_MAX_SURFACE_LEG_MISS_FRACTION", RESEARCH_MAX_SURFACE_LEG_MISS_FRACTION, "fraction of legs",
        "Largest share of a run's surface-priced legs that may fall back to the legacy single-IV model before the "
        "run record flags the result as only partly surface-priced (the run is flagged and stamped, not refused).",
        "The per-leg fallback is a declared model so one thin strike does not stop a study, but a study whose "
        "legs mostly miss the surface is a legacy-priced study with a surface label; if a study file could raise "
        "the line, it could hide that.",
        "backtesting/result.py (surface_pricing_stamp); backtesting/walkforward.py (data_contract); run_backtest.py"),
    Invariant(
        "IV_SCALE_MAX", IV_SCALE_MAX, "ratio",
        "Upper clamp of the same scaling.",
        "Same reason, other direction.",
        "backtesting/engine.py"),
    Invariant(
        "SETTLEMENT_LOSS_TOLERANCE", SETTLEMENT_LOSS_TOLERANCE, "dollars",
        "How far a settled gross loss may exceed the defined max loss before it is counted as over_max_loss.",
        "A floating-point tolerance, not a modelling choice; loosening it would hide a pricing bug.",
        "backtesting/engine.py (_finish_settlement)"),
    Invariant(
        "PROB_CLIP_EPS", PROB_CLIP_EPS, "probability",
        "Probabilities are clipped to [eps, 1 - eps] before log-loss.",
        "Numerical safety (log 0); not a modelling choice.",
        "backtesting/walkforward.py (_prob_score_metrics)"),
)


def research_invariant_values() -> dict[str, float]:
    """name -> value for every fixed research limit (the log and run-record payload)."""
    return {i.name: i.value for i in RESEARCH_INVARIANTS}


def log_research_invariants() -> None:
    """Log the limits once per process; like its live sibling it can never block a run."""
    try:
        from ait.utils.logging import get_logger  # lazy: keep this module a leaf
        limits = research_invariant_values()
        get_logger("config.invariants_research").info(
            "research_fixed_limits_in_effect", count=len(limits), limits=limits,
            note="fixed in code on purpose (ait/config/invariants_research.py); not set by config.yaml, "
                 "the research overlay or a study. Why each is fixed: "
                 "docs/RESEARCH_GUIDE.md \"Are some values not configurable?\".")
    except Exception:  # noqa: BLE001 - diagnostics must never block a run
        pass

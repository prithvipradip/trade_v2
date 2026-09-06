"""ONE authority for the LIVE range-model spec (threshold, horizon).

units-scale-05 (R22 relationship-defect register, 2026-08-25). The live
RangePredictor was constructed from bare literals in orchestrator.py
(``threshold_pct=0.05, horizon_days=30``) while the research half of the same
must-agree pair had already moved: ``walkforward._range_label_horizon`` derives
the label/eval horizon from the REACHABLE trade horizon —
``options.dte_range[0] - EXPIRY_APPROACHING_DTE`` — because positions are
opened at the low end of the DTE band and credit structures are closed at
``EXPIRY_APPROACHING_DTE``, so a real hold caps at ~9 calendar days.

Why that divergence was expensive: the live 0.65 confidence floor
(``ml.range_min_confidence``) was justified by a parameter sweep of the
RESEARCH pipeline. P(stay within +/-5% over 30 days) is strictly lower than
P(stay within +/-5% over 9 days), so applying a floor calibrated on the 9-day
question to a 30-day probability systematically vetoed live iron-condor /
short-strangle entries that the validated research gate would have taken —
suppressing exactly the IC closes the go-live verdict sample needs.

MIRROR NOTE (must-agree, enforce on edit): the horizon returned here and
``ait.backtesting.walkforward.WalkForwardEngine._range_label_horizon()`` MUST
be equal. Research additionally caps its horizon at ``config.max_hold_days``
(``min(max_hold_days, reachable)``); on every band this repo has shipped
(dte_range [14,30] / [14,45] -> reachable 9) the cap is inert and the two
agree. If a future DTE band pushes ``reachable`` above ``max_hold_days``, both
sides must adopt the cap together — do not change one alone.

Threshold: research uses ``_adaptive_range_threshold`` (clipped 0.02-0.15);
live deliberately stays at the fixed +/-5% for now (the wing width the live
condor actually sells), but it is routed through this function so there is a
single place to change it and a single place a test can read it.

Import-light on purpose (mirrors exit_policy.py): no pandas/numpy/broker
imports, so both the live orchestrator and research code can import it.
"""

from __future__ import annotations

# Fixed live containment band: +/-5%. Research adapts its threshold per window;
# live keeps the wing-matched constant until that is deliberately changed HERE.
LIVE_RANGE_THRESHOLD_PCT: float = 0.05


def live_range_spec(settings=None) -> tuple[float, int]:
    """Return ``(threshold_pct, horizon_days)`` for the LIVE range model.

    ``settings`` is an optional loaded ``Settings``; when omitted the config is
    loaded (and, if that fails, the ``OptionsConfig`` field defaults are used)
    so a caller without a settings handle still gets the same answer.

    horizon = ``max(1, options.dte_range[0] - EXPIRY_APPROACHING_DTE)`` — the
    horizon a live trade can actually reach. See the module MIRROR NOTE: this
    must equal ``walkforward._range_label_horizon()``.
    """
    from ait.execution.exit_policy import EXPIRY_APPROACHING_DTE

    opts = getattr(settings, "options", None) if settings is not None else None
    if opts is None:
        from ait.config.settings import OptionsConfig, load_settings
        try:
            opts = load_settings().options
        except Exception:  # noqa: BLE001 — never block construction on config I/O
            opts = OptionsConfig()

    entry_dte = int(opts.dte_range[0])
    horizon = max(1, entry_dte - int(EXPIRY_APPROACHING_DTE))
    return LIVE_RANGE_THRESHOLD_PCT, horizon

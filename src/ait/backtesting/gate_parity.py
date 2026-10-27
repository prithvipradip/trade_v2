"""W81: which research entry gates have a live twin, and which exist on one side only.

The 2026-10-13 session-end audit found three gates that make research and live disagree
about whether a condor may open, none of them visible in the parity manifest (P1 the
research-only realized-vol cap, P2 the research-only trending-down regime veto, P3 the
live-only VIX halt). This module is the ONE registry of the engine's rejection reasons so
the next one-sided gate cannot appear silently:

* ``ENGINE_REJECT_REASONS`` classifies every reason the research engine can record through
  ``Backtester._reject`` / ``_would_reject``;
* ``LIVE_ONLY_GATES`` lists what live applies and research does not simulate;
* ``one_sided_gate_warnings`` turns the active one-sided gates into manifest warnings;
* ``engine_reject_reasons`` extracts the reasons from the engine SOURCE (AST), so a guard
  test fails when a reason is added without a row here.

Pure: imports only the standard library (an AST guard pins that).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

TWIN = "live_twin"                 # the same decision (shared code, or an equivalent live gate)
RESEARCH_ONLY = "research_only"    # a research gate live does not have: a declared deviation
MECHANICS = "mechanics"            # not a gate: why a candidate could not be built or filled


@dataclass(frozen=True)
class GateRow:
    reason: str
    kind: str
    live: str                      # what live does instead, in one line
    switch: str | None = None      # backtest.* key that turns a research-only gate off/on


def _g(reason: str, kind: str, live: str, switch: str | None = None) -> tuple[str, GateRow]:
    return reason, GateRow(reason, kind, live, switch)


ENGINE_REJECT_REASONS: dict[str, GateRow] = dict([
    # ---- gates with a live twin ---------------------------------------------------------
    _g("hurst_hard_veto", TWIN, "fractal gate: ait.ml.regime_gate (observe while "
       "ml.entry_gates_enabled is false; research gates_mode decides)"),
    _g("range_gate_no_predictor", TWIN, "range model gate (observe live; gates_mode decides)"),
    _g("range_gate_below_threshold", TWIN, "range model gate (observe live; gates_mode decides)"),
    _g("meta_label", TWIN, "meta-label gate (orchestrator)"),
    _g("iv_rank_unknown", TWIN, "fails closed on an unmeasurable rank"),
    _g("iv_rank_floor", TWIN, "condor builder floor AIT_IRON_CONDOR_IV_FLOOR"),
    _g("iv_rank_cap", TWIN, "risk manager VRP cap AIT_IV_RANK_CAP"),
    _g("macro_event", TWIN, "orchestrator pre-event blackout (risk.pre_event_blackout_days)"),
    _g("earnings", TWIN, "earnings proximity guard (W70 parity)"),
    _g("ex_div_entry", TWIN, "W51 orchestrator trade_blocked_ex_div: a call-selling entry whose "
       "[entry, expiry] spans the projected ex-date (W84 twin, backtest.ex_div_entry_gate)"),
    _g("max_concurrent", TWIN, "risk manager max open positions / credit cap"),
    _g("max_concurrent_intraday", TWIN, "risk manager max open positions / credit cap"),
    _g("direction_confidence", TWIN, "directional strategies only; condors are exempt (L5)"),
    _g("direction_confidence_bearish", TWIN, "directional strategies only; condors are exempt"),
    # ---- gates research has and live does not -------------------------------------------
    _g("vol_gate", RESEARCH_ONLY, "no realized-vol entry cap (live gates on VIX, see "
       "LIVE_ONLY_GATES)", "max_entry_vol_annual"),
    _g("regime_veto", RESEARCH_ONLY, "no regime entry veto (regime feeds the direction "
       "override, meta-label features and roll defense only)", "regime_veto_gate"),
    _g("iv_rank_rise", RESEARCH_ONLY, "no rise veto (live gates on the rank LEVEL only)",
       "iv_rank_rise_gate"),
    _g("pct_from_high", RESEARCH_ONLY, "no drawdown-from-60d-high gate (experimental, "
       "disabled at -1.0)", "pct_from_60d_high_threshold"),
    _g("post_event", RESEARCH_ONLY, "no post-event blackout (inert unless a study sets it)"),
    # ---- mechanics: not gates -----------------------------------------------------------
    _g("no_strategy", MECHANICS, "selector returned nothing"),
    _g("before_eval_start", MECHANICS, "warm-up rows before the evaluation start"),
    _g("builder_none", MECHANICS, "the structure could not be built (details: builder:*)"),
    _g("no_bars_in_entry_window", MECHANICS, "no intraday bar inside the entry window"),
    _g("limit_not_filled", MECHANICS, "the entry limit did not fill in the timeout"),
])

# `builder:<detail>` reasons record WHY a build failed (sub-reasons of builder_none).
BUILDER_PREFIX = "builder:"

# What live applies at entry and the research engine does not simulate.
LIVE_ONLY_GATES: tuple[GateRow, ...] = (
    GateRow("vix_halt", RESEARCH_ONLY,
            "risk manager: no NEW credit entries at VIX >= risk.credit_vix_halt (fails closed "
            "when VIX is unknown); research has no twin", "risk.credit_vix_halt"),
    GateRow("vix_tier_credit_cap", RESEARCH_ONLY,
            "risk manager: the number of open credit positions is capped by VIX tier "
            "(risk.credit_cap_vix_tiers); research caps by max_concurrent_positions only",
            "risk.credit_cap_vix_tiers"),
    GateRow("account_risk", RESEARCH_ONLY,
            "circuit breaker, PDT guard and live's 60-day price-correlation branch are live-only; "
            "research caps contracts at max_contracts_per_trade and does not simulate account "
            "state (the lot count, the per-expiry cap, the SECTOR_GROUPS cluster cap and the "
            "per-symbol cap ARE shared, see POOL_REJECT_REASONS)", None),
)

# W108: the rejections the walk-forward POOL (backtesting/pool.py allocate_pool) records, each with
# the live gate it mirrors. The engine's own reasons are ENGINE_REJECT_REASONS above; these are
# the book-level gates applied after every symbol's engine run. A guard test (tests/
# test_lot_sizing_parity.py) fails when pool.py records a reason that has no row here.
POOL_REJECT_REASONS: dict[str, GateRow] = dict([
    _g("no_contracts", MECHANICS, "the engine sized the candidate at zero lots"),
    _g("max_open_positions", TWIN, "risk manager gate 3b: positions.max_open_positions"),
    _g("per_expiry_cap", TWIN, "risk manager gate 4b: risk.max_positions_per_expiry (the "
       "AIT_MAX_PER_EXPIRY env override is not modelled)"),
    _g("correlation_cluster", TWIN, "risk manager gate 5: risk.max_correlated_positions in the same "
       "SECTOR_GROUPS cluster or symbol (live's 60-day price-correlation branch is not modelled)"),
    _g("unaffordable", TWIN, "risk manager gate 6b / PositionSizer: one lot above "
       "risk.max_position_risk_pct x NLV (the shared rule, ait.risk.lot_sizing)"),
    _g("portfolio_risk", TWIN, "risk manager gate 6b-2: positions.max_portfolio_risk_pct"),
    _g("symbol_concentration", TWIN, "risk manager gate 6c: risk.max_symbol_concentration_pct"),
])


def classify(reason: str) -> GateRow | None:
    """The registry row for an engine rejection reason (None = unregistered)."""
    if reason.startswith(BUILDER_PREFIX):
        return ENGINE_REJECT_REASONS["builder_none"]
    return ENGINE_REJECT_REASONS.get(reason)


def engine_reject_reasons(source: str) -> tuple[set[str], list[int]]:
    """Every string literal passed to ``<x>._reject(...)`` / ``<x>._would_reject(...)``.

    Returns ``(reasons, dynamic_lines)``: the second element lists the source lines of calls
    whose first argument is NOT a plain string literal (those cannot be classified, so the
    guard test fails on them).
    """
    tree = ast.parse(source)
    reasons: set[str] = set()
    dynamic: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("_reject", "_would_reject")):
            continue
        arg = node.args[0] if node.args else None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            reasons.add(arg.value)
        else:
            dynamic.append(node.lineno)
    return reasons, dynamic


def one_sided_gate_warnings(bt_cfg: Any, live_settings: Any) -> list[str]:
    """Manifest warnings for every gate that is ACTIVE on one side only.

    ``bt_cfg`` is the research ``BacktestConfig`` (may be None); ``live_settings`` the loaded
    config.yaml settings (may be None). A missing input yields no warning for that gate rather
    than a guess.
    """
    out: list[str] = []
    vol_cap = getattr(bt_cfg, "max_entry_vol_annual", None)
    if vol_cap is not None and float(vol_cap) < 5.0:
        out.append(
            f"vol_gate: RESEARCH-ONLY realized-vol cap - a neutral credit entry is skipped when "
            f"10-day realized vol exceeds backtest.max_entry_vol_annual={float(vol_cap):g} "
            "(Optuna searches it). Live has no such gate; its volatility brake is the VIX halt, "
            "which research does not simulate (vix_halt below). See the W81 PLAN entry (P1).")
    if bool(getattr(bt_cfg, "regime_veto_gate", False)):
        down = getattr(bt_cfg, "regime_down_threshold", None)
        out.append(
            "regime_veto: RESEARCH-ONLY trending-down veto is ON - a neutral credit entry is "
            "skipped when 5-day realized vol exceeds 20-day and price is more than "
            f"{abs(float(down)):.0%} below its 20-day SMA. Live has no such entry veto (regime "
            "feeds the direction override, meta-label features and roll defense only). Set "
            "backtest.regime_veto_gate: false in the research overlay to turn it off (operator "
            "decision D10); the class is still recorded per entry.")
    pct = getattr(bt_cfg, "pct_from_60d_high_threshold", None)
    if pct is not None and float(pct) > -0.99:
        out.append(
            f"pct_from_60d_high_threshold: {float(pct):g} - RESEARCH-ONLY drawdown gate is "
            "enabled; live has none.")
    halt = getattr(getattr(live_settings, "risk", None), "credit_vix_halt", None)
    if halt is not None:
        out.append(
            f"vix_halt: LIVE-ONLY - the bot refuses new credit entries at VIX >= {float(halt):g} "
            "(risk.credit_vix_halt) and caps open credit positions by VIX tier; the research "
            "engine has no twin, so it may trade days live cannot. Quantify with "
            "scripts/gate_parity_ab.py; see the W81 PLAN entry (P3 / decision D11).")
    ex_div = getattr(bt_cfg, "ex_div_entry_gate", None)
    if ex_div == "off":
        out.append(
            "ex_div_entry: OFF in research - live refuses a call-selling entry whose [entry, "
            "expiry] spans the projected ex-dividend date (W51; 14-17% of sessions at entry DTE "
            "14), so research trades days live cannot. Set backtest.ex_div_entry_gate: live for "
            "parity; quantify with scripts/ex_div_gate_ab.py (PLAN.md W84).")
    elif ex_div in ("exact", "last_k_days"):
        k = getattr(bt_cfg, "ex_div_entry_k_days", None)
        rule = ("any real ex-date in [entry, expiry]" if ex_div == "exact"
                else f"a real ex-date within {k} days of expiry")
        out.append(
            f"ex_div_entry: research blocks on {rule} (mode {ex_div}); live blocks on ONE "
            "PROJECTED ex-date in [entry, expiry] (mode live). A deliberate deviation for the "
            "F2 / F3 study (PLAN.md W84), not parity.")
    return out

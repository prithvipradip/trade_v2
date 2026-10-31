"""The research preflight for a walk-forward study: the run's resolved settings in, a ``ResearchRequest`` out.

W116. ``ait.data.research_inputs`` knows how to check a request against the stored history; it reads no config. This
module is the one place that turns what a walk-forward is configured to do (symbols, window, gates, friction model,
pricing source, intraday features) into that request, so the walk-forward, the optimizer's study scripts and the
standalone ``scripts/research_preflight.py`` all ask the same question of the store. The windows come from
``required_window``, the same function the loaders use, so the preflight cannot check a different span than the run
loads.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from ait.config.invariants_research import (
    RESEARCH_MAX_DIVIDEND_GAP_DAYS, RESEARCH_MAX_EVENT_GAP_DAYS, RESEARCH_MAX_NULL_IV_RUN_SESSIONS)
from ait.data.research_inputs import ResearchRequest, required_window


def build_request(*, symbols, db_path: "Path | str", table_prefix: str, today: date,
                  history_start_date: "date | None", history_end_date: "date | None",
                  fetch_days: int, context_days: int, settings, macro_event_gate: bool,
                  macro_event_types, friction_model: str, intraday_features: bool, vlmc_min_coverage: float,
                  ex_div_entry_gate: str) -> ResearchRequest:
    """What a study with these settings needs from the history database.

    ``settings`` is the RESOLVED settings object the run uses (the earnings buffers, the exempt symbols, the pricing
    source and the short-DTE friction switch live there); the rest are the run's own configured values."""
    bt = settings.backtest
    window = required_window(today=today, days=fetch_days, start_date=history_start_date, end_date=history_end_date)
    spy_window = required_window(today=today, days=context_days, start_date=history_start_date,
                                 end_date=history_end_date)
    vix_window = required_window(today=today, days=context_days, end_date=history_end_date)
    symbols = tuple(symbols)
    return ResearchRequest(
        window=window, symbols=symbols, db_path=db_path, table_prefix=table_prefix,
        max_null_iv_run=RESEARCH_MAX_NULL_IV_RUN_SESSIONS, max_dividend_gap_days=RESEARCH_MAX_DIVIDEND_GAP_DAYS,
        max_event_gap_days=RESEARCH_MAX_EVENT_GAP_DAYS,
        context_window=spy_window, vix_window=vix_window,
        context_symbols=(() if "SPY" in symbols else ("SPY",)), vix_symbols=("VIX",),
        needs_iv=True, ex_div_gate=ex_div_entry_gate != "off",
        macro_gate=bool(macro_event_gate and macro_event_types), macro_event_types=tuple(macro_event_types or ()),
        earnings_gate=(settings.earnings.buffer_days_before > 0 or settings.earnings.buffer_days_after > 0),
        earnings_exempt=tuple(str(s).upper() for s in bt.earnings_exempt_symbols),
        friction_per_leg=friction_model == "per_leg_usd", friction_bands=bool(bt.friction_short_dte),
        intraday=bool(intraday_features), intraday_min_coverage=float(vlmc_min_coverage),
        surface=bt.pricing_source == "surface")

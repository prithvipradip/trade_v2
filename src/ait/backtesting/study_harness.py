"""W87a: the research "study harness" engine setup, defined once.

``scripts/gate_parity_ab.py``, ``ex_div_gate_ab.py``, ``dte_dimension_flatness.py`` and
``iv_rank_parity_probe.py`` each pasted the same ~25-keyword ``Backtester(...)`` block (W81 / W84 / W79a).
That block is a deliberate STUDY configuration, not live's, and it is looser than live in three ways that
used to be buried in copies:

* ``ic_min_credit`` 0.30 (live: 0.70) and ``ic_min_credit_width`` 0.05 (live: 0.10), so condors that live's
  credit floor would refuse are still built and the gates under study are actually reached;
* ``iv_floor`` 0.05 (config: 0.20), so the IV floor does not mask the gate being measured;
* the fractal veto off (``hurst_*`` 0.0), 1 contract, one position at a time (explicit, so a change to the
  research DEFAULT of ``max_concurrent_positions`` cannot move these studies), Black-Scholes pricing,
  per-leg friction with the short-DTE bands, a 10:30-15:30 ET entry window scanned hourly, $100k capital.

Every reference number in PLAN.md W81 / W84 was produced with these. They are named below so the difference
from live is visible, and ``tests/test_w87a_hygiene.py`` pins each script's kwargs to what it used before.
Research only; the live bot never imports this module.
"""
from __future__ import annotations

from pathlib import Path

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
WARMUP_START = "2023-01-01"     # features start a year before the first window, as the walk-forward does

STUDY_INITIAL_CAPITAL = 100_000.0
STUDY_IC_MIN_CREDIT = 0.30          # live: AIT_IC_MIN_CREDIT 0.70
STUDY_IC_MIN_CREDIT_WIDTH = 0.05    # live: AIT_IC_MIN_CREDIT_WIDTH 0.10
STUDY_IV_FLOOR = 0.05               # config: backtest.iv_floor 0.20
STUDY_ENTRY_WINDOW_ET = ("10:30", "15:30")
STUDY_SCAN_INTERVAL_MINUTES = 60


def study_engine_kwargs(*, data, symbol: str, db_path: Path, intraday_store, entry_dte: int,
                        features_cache=None, iv_rank_series=None, iv_rank_rise_gate: bool = False,
                        **extra) -> dict:
    """The keyword arguments of the W81 study ``Backtester``.

    ``iv_rank_series`` (when given) also switches ``iv_rank_gate`` on, as three of the four scripts do;
    ``dte_dimension_flatness`` passes neither and leaves the engine default. ``extra`` is applied last
    (per-script switches: ``regime_veto_gate``, ``ex_div_entry_gate``, the sweep values, ...).
    """
    kw = dict(
        data=data, strategies=["iron_condor"], initial_capital=STUDY_INITIAL_CAPITAL,
        macro_event_gate=False, allow_live_model_fallback=False, features_cache=features_cache,
        condor_any_direction=True, intraday_store=intraday_store, symbol=symbol,
        friction_model="per_leg_usd", db_path=db_path, gates_mode="observe", max_contracts_per_trade=1,
        max_concurrent_positions=1, hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0,
        iv_rank_rise_gate=iv_rank_rise_gate, iv_floor=STUDY_IV_FLOOR, ic_min_credit=STUDY_IC_MIN_CREDIT,
        ic_min_credit_width=STUDY_IC_MIN_CREDIT_WIDTH,
        entry_window_start_et=STUDY_ENTRY_WINDOW_ET[0], entry_window_end_et=STUDY_ENTRY_WINDOW_ET[1],
        scan_interval_minutes=STUDY_SCAN_INTERVAL_MINUTES, engine_model="realistic",
        friction_short_dte=True, pricing_source="bs_legacy", entry_dte=entry_dte,
    )
    if iv_rank_series is not None:
        kw["iv_rank_gate"] = True
        kw["iv_rank_series"] = iv_rank_series
    kw.update(extra)
    return kw


def study_inputs(db_path: Path, symbol: str, start: str, end: str):
    """(ResearchData, data slice, features, IV-rank series) for one symbol and window: the data-prep trio
    three of the scripts repeated. Imported lazily so ``apply_runtime_env_defaults()`` can run first."""
    from datetime import date

    from ait.backtesting.replay import ResearchData
    from ait.config.invariants_research import (
        RESEARCH_MAX_DIVIDEND_GAP_DAYS, RESEARCH_MAX_EVENT_GAP_DAYS, RESEARCH_MAX_NULL_IV_RUN_SESSIONS)
    from ait.data.iv_rank import iv_rank_series
    from ait.data.research_inputs import ResearchRequest, Window, preflight_or_raise
    from ait.ml.features import FeatureEngine

    # W116: the window this study reads must be stored (bars, IV, the per-leg friction model and its bands) or the
    # script does not start; the same check the walk-forward runs
    preflight_or_raise(ResearchRequest(
        window=Window(date.fromisoformat(start), date.fromisoformat(end)), symbols=(symbol,), db_path=db_path,
        max_null_iv_run=RESEARCH_MAX_NULL_IV_RUN_SESSIONS, max_dividend_gap_days=RESEARCH_MAX_DIVIDEND_GAP_DAYS,
        max_event_gap_days=RESEARCH_MAX_EVENT_GAP_DAYS, friction_per_leg=True, friction_bands=True),
        context=f"study inputs {symbol}")

    rd = ResearchData(str(db_path), symbol)
    frame = rd.daily_frame()
    data = frame.loc[start:end].copy()
    features = FeatureEngine().compute(frame[OHLCV].loc[WARMUP_START:])
    return rd, data, features, iv_rank_series(frame["implied_vol"])


def assert_friction_loaded(bt) -> None:
    """A study that silently priced with no friction model would read as a result; fail loudly instead."""
    assert bt._leg_spread_model is not None, "per-leg friction model did NOT load"
    assert bt._friction_bands is not None, "short-DTE friction bands did NOT load"

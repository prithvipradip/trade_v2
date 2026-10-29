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
    from ait.backtesting.replay import ResearchData
    from ait.data.iv_rank import iv_rank_series
    from ait.ml.features import FeatureEngine

    rd = ResearchData(str(db_path), symbol)
    frame = rd.daily_frame()
    data = frame.loc[start:end].copy()
    features = FeatureEngine().compute(frame[OHLCV].loc[WARMUP_START:])
    return rd, data, features, iv_rank_series(frame["implied_vol"])


from ait.backtesting.run_record import REPO_ROOT  # noqa: E402  (stdlib-light; one definition of the repo root)
RUNS_DIR = REPO_ROOT / "reports" / "runs"


def add_study_args(ap) -> None:
    """W106: the four study scripts take a study file and write a run record, as ``run_backtest.py`` does."""
    ap.add_argument("--study-file", default=None, metavar="PATH",
                    help="a studies/<name>.yaml merged over config.yaml + config_research.yaml for this run, in "
                         "every worker. Without it the workers read config.yaml only, as before. A pre-registered "
                         "study must name a PLAN.md heading in its rule:.")
    ap.add_argument("--no-run-record", action="store_true", default=False,
                    help="do not write reports/runs/<run_id>/run_record.json (or log to MLflow)")


def load_study_config(study_file):
    """Validate the study (and its PLAN.md registration) and make it this process's configuration.
    Returns the Study, or None without a study file (configuration untouched)."""
    if not study_file:
        return None
    from ait.config.research import assert_registered, load_study, use_research_config
    study = load_study(study_file)
    assert_registered(study, REPO_ROOT / "PLAN.md")
    use_research_config(REPO_ROOT / "config.yaml", study)
    return study


def _init_study_worker(study_file) -> None:
    if study_file:
        from ait.config.research import use_research_config
        use_research_config(REPO_ROOT / "config.yaml", study_file)


def study_pool(workers: int, study_file=None):
    """A process pool whose workers run under the same study configuration as the parent."""
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init_study_worker,
                               initargs=(str(study_file) if study_file else None,))


def write_study_record(kind: str, args, study, results: list, *, symbols: list[str],
                       db_path: Path | None = None) -> None:
    """One run record per study-script run (W87b layout). Never fails the run."""
    if getattr(args, "no_run_record", False):
        return
    try:
        from ait.backtesting.run_record import finish_run
        from ait.config.research import research_overlay_path, settings_origin
        from ait.config.settings import load_settings
        cfg = REPO_ROOT / "config.yaml"
        settings = load_settings(str(cfg)) if study is None else load_settings()
        run_dir, mlflow_id = finish_run(
            runs_dir=RUNS_DIR, kind="study", name=f"{kind}_{study.name if study else 'adhoc'}",
            settings=settings, config_path=cfg,
            overlay_path=research_overlay_path(cfg) if study else None,
            study_path=study.path if study else None,
            study=({"name": study.name, "question": study.question, "rule": study.rule,
                    "settings": study.settings} if study else None),
            settings_origin=settings_origin(cfg, study) if study else None,
            cli_args=dict(vars(args)), symbols=symbols, db_path=db_path,
            metrics={"runs": len(results)}, extra={"results": results})
        print(f"\nRun record: {run_dir}" + (f"  (MLflow run {mlflow_id})" if mlflow_id else ""))
    except Exception as exc:  # noqa: BLE001 - a record failure must never fail the study
        print(f"\n! run record not written ({type(exc).__name__}: {exc})")


def assert_friction_loaded(bt) -> None:
    """A study that silently priced with no friction model would read as a result; fail loudly instead."""
    assert bt._leg_spread_model is not None, "per-leg friction model did NOT load"
    assert bt._friction_bands is not None, "short-DTE friction bands did NOT load"

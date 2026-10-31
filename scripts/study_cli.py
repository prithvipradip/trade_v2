"""W106 item 8: the study-file and run-record command-line layer of the four study scripts.

`gate_parity_ab`, `ex_div_gate_ab`, `dte_dimension_flatness` and `iv_rank_parity_probe` take
`--study-file` and write a run record, as `run_backtest.py` does. It lives here, not in
`ait.backtesting.study_harness`, because only `ait.config.research` and the research entry points may
touch the research overlay (W68: nothing under `src/ait` the live bot could import references it).
Imported by those scripts; not run directly.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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

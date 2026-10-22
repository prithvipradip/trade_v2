#!/usr/bin/env python
"""AIT v2 — Strategy Parameter Optimizer

Runs Bayesian hyperparameter search (Optuna TPE) over strategy parameter
spaces and/or ML hyperparameter spaces, then reports the best parameters
and optionally writes them back to config.yaml.

Usage examples:
    # Quick 20-trial search on iron_condor with SPY data
    python run_optimizer.py --strategies iron_condor --symbols SPY --n-trials 20

    # No --symbols / --strategies: backtest.default_symbols (QQQ) and options.strategies (W87a)
    python run_optimizer.py --n-trials 20

    # Multi-strategy composite objective, 4 parallel jobs
    python run_optimizer.py \\
        --strategies iron_condor bull_call_spread \\
        --symbols SPY QQQ \\
        --n-trials 200 \\
        --objective composite \\
        --n-jobs 4 \\
        --train-days 365 \\
        --capital 50000

    # Include ML hyperparams in the search
    python run_optimizer.py --strategies iron_condor --optimize-ml --n-trials 100

    # Persist study to SQLite for resumable runs
    python run_optimizer.py --strategies iron_condor \\
        --storage sqlite:///data/optuna.db \\
        --study-name my_study \\
        --n-trials 50

    # Apply best params to config.yaml
    python run_optimizer.py --strategies iron_condor --n-trials 20 --apply
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AIT v2 Strategy Parameter Optimizer (Optuna)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategy names to optimize. Default: config options.strategies (W87a).",
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols for backtesting during optimization. Default: config backtest.default_symbols (W87a).",
    )
    p.add_argument("--n-trials", type=int, default=None,
                   help="Number of Optuna trials. Default: backtest.walkforward_tuning.optimizer_n_trials")
    p.add_argument(
        "--objective",
        choices=["sharpe_ratio", "composite", "profit_factor", "win_rate"],
        default="sharpe_ratio",
        help="Objective to maximise",
    )
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel Optuna workers")
    p.add_argument("--train-days", type=int, default=None,
                   help="Days of historical data. Default: backtest.walkforward_tuning.train_days")
    p.add_argument("--capital", type=float, default=None,
                   help="Initial capital per backtest. Default: config.yaml "
                        "account.capital (W-capital: the single home).")
    p.add_argument("--optimize-ml", action="store_true", help="Also search ML hyperparameter spaces")
    p.add_argument("--table-prefix", default=None,
                   help="Research table prefix (default: config backtest.research_table_prefix). "
                        "Pass '' for the legacy loader / live tables.")
    p.add_argument("--db-path", default="data/historical.db",
                   help="SQLite store holding the research tables (default: data/historical.db).")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///data/optuna.db")
    p.add_argument("--study-name", type=str, default=None, help="Optuna study name (for resuming)")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write best params back to config.yaml after optimization",
    )
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p.add_argument("--study-file", default=None, metavar="PATH",
                   help="W87b: a studies/<name>.yaml merged over config.yaml + config_research.yaml for THIS run "
                        "(settings: and search_space: overrides). Cannot be combined with "
                        "--apply (params found under study settings must not be written to the live config).")
    p.add_argument("--no-run-record", action="store_true", default=False,
                   help="W87b: do not write reports/runs/<run_id>/run_record.json (or log to MLflow).")
    p.add_argument(
        "--output",
        type=str,
        default="reports/optimization_result.json",
        help="Path to save results JSON",
    )
    return p.parse_args()


def resolve_optimizer_args(args: argparse.Namespace, tuning) -> None:
    """W90b: fill the trial count and history window the operator did not pass from ``backtest.walkforward_tuning``
    of the research view (config_research.yaml / a study); an explicit flag wins."""
    if args.n_trials is None:
        args.n_trials = int(tuning.optimizer_n_trials)
    if args.train_days is None:
        args.train_days = int(tuning.train_days)


def _write_run_record(args, study, settings, optimizer, result, prefix) -> None:
    """W87b: what produced this optimisation -- one JSON in reports/runs/<run_id>/ (+ MLflow when installed),
    with the merged search space and its hash. Never fails the run."""
    try:
        from ait.backtesting.run_record import finish_run
        from ait.config.research import research_overlay_path, settings_origin
        metrics = {"best_value": result.best_value, "n_trials": args.n_trials,
                   "usable": 1.0 if result.usable else 0.0, "best_n_trades": result.best_n_trades}
        run_dir, mlflow_id = finish_run(
            kind="optimize", name=f"{'-'.join(args.symbols).lower()}_{study.name if study else 'adhoc'}",
            settings=settings, config_path=args.config, overlay_path=research_overlay_path(args.config),
            study_path=study.path if study else None,
            study=({"name": study.name, "question": study.question, "rule": study.rule,
                    "settings": study.settings, "search_space": study.search_space} if study else None),
            settings_origin=settings_origin(args.config, study),
            cli_args=dict(vars(args)), symbols=args.symbols, db_path=args.db_path,
            search_space=optimizer.effective_search_space(), search_space_hash=optimizer.search_space_hash(),
            seeds={"optuna_seed": getattr(optimizer, "_seed", None)},
            metrics={k: v for k, v in metrics.items() if v is not None},
            extra={"best_params": dict(result.best_params)})
        print(f"Run record: {run_dir}" + (f"  (MLflow run {mlflow_id})" if mlflow_id else ""))
    except Exception as exc:  # noqa: BLE001 - a record failure must never fail the research run
        print(f"! run record not written ({type(exc).__name__}: {exc})")


def main() -> None:
    args = parse_args()

    from ait.config.research import (
        assert_registered, load_study, resolve_universe, use_research_config)
    _study = load_study(args.study_file) if args.study_file else None     # W87b: the third config layer
    if _study is not None:
        assert_registered(_study)
        if args.apply:
            sys.exit("REFUSING --apply with --study-file: params found under a study's settings / search space "
                     "are not promotable to the live config; promote through a registered wave.")
    _research_cfg = use_research_config(study=_study)   # W68: config.yaml + config_research.yaml (research only)
    args.symbols, args.strategies = resolve_universe(args.symbols, args.strategies, _research_cfg)
    from ait.optimization.optimizer import StrategyOptimizer
    if args.capital is None:
        # W90a: account.capital of the research view (overlay / study aware); was a raw read of config.yaml
        args.capital = float(_research_cfg.account.capital)
    resolve_optimizer_args(args, _research_cfg.backtest.walkforward_tuning)   # W90b

    print("=" * 70)
    print("  AIT v2 — STRATEGY PARAMETER OPTIMIZER")
    print("=" * 70)
    print(f"  Strategies:  {', '.join(args.strategies)}")
    print(f"  Symbols:     {', '.join(args.symbols)}")
    print(f"  Trials:      {args.n_trials}")
    print(f"  Objective:   {args.objective}")
    print(f"  ML search:   {'yes' if args.optimize_ml else 'no'}")
    print(f"  Storage:     {args.storage or 'in-memory'}")
    print("=" * 70)

    # W65: research history (IB tables + the operator's economic calendar), config-driven.
    from pathlib import Path
    try:
        from ait.config.settings import load_settings
        _bcfg = load_settings().backtest
    except Exception:  # noqa: BLE001
        _bcfg = None
    _prefix = (args.table_prefix if args.table_prefix is not None
               else (_bcfg.research_table_prefix if _bcfg else ""))
    _calendar = None
    if _prefix and _bcfg is not None:
        from ait.data.economic_calendar import build_research_calendar
        try:
            _calendar = build_research_calendar(Path(args.db_path), _prefix, _bcfg.macro_event_types)
        except Exception as _exc:  # noqa: BLE001
            print(f"  ! research calendar unavailable ({_exc}); hardcoded calendar used")
    print(f"  History:     {'research tables ' + _prefix + '* in ' + args.db_path if _prefix else 'legacy loader'}"
          f"{' | research macro calendar' if _calendar else ''}")

    optimizer = StrategyOptimizer(
        table_prefix=_prefix,
        db_path=(Path(args.db_path) if _prefix else None),
        history_start_date=(_bcfg.history_start_date if (_bcfg and _prefix) else None),
        economic_calendar=_calendar,
        symbols=args.symbols,
        strategies=args.strategies,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        objective=args.objective,
        study_name=args.study_name,
        storage=args.storage,
        optimize_ml=args.optimize_ml,
        initial_capital=args.capital,
        train_days=args.train_days,
        search_space_overrides=(_study.search_space or None) if _study is not None else None,   # W87b
    )

    t0 = time.time()
    print("\nRunning optimization...")
    try:
        result = optimizer.run()
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s\n")
    print(result.summary())

    # Save JSON report
    result.save(args.output)
    print(f"\nResults saved to: {args.output}")

    if not args.no_run_record:
        _write_run_record(args, _study, _research_cfg, optimizer, result, _prefix)

    # Optionally apply to config.yaml
    if args.apply:
        # Deep-audit BT-H2: --apply wrote single-window, no-holdout params
        # straight into the LIVE config.yaml. Require an explicit env opt-in
        # so overfit params can't be promoted by muscle memory.
        if os.environ.get("AIT_ALLOW_APPLY") != "1":
            print("REFUSING --apply: set AIT_ALLOW_APPLY=1 to write optimizer "
                  "params into the live config (they are in-sample only).")
        else:
            result.apply_to_config(args.config)
            # R5 audit: this print sat OUTSIDE the else — the tool claimed
            # "Best params written" even when the apply was refused.
            print(f"Best params written to: {args.config}")


if __name__ == "__main__":
    main()

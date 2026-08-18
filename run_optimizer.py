#!/usr/bin/env python
"""AIT v2 — Strategy Parameter Optimizer

Runs Bayesian hyperparameter search (Optuna TPE) over strategy parameter
spaces and/or ML hyperparameter spaces, then reports the best parameters
and optionally writes them back to config.yaml.

Usage examples:
    # Quick 20-trial search on iron_condor with SPY data
    python run_optimizer.py --strategies iron_condor --symbols SPY --n-trials 20

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
        default=["iron_condor"],
        help="Strategy names to optimize",
    )
    p.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ"],
        help="Symbols for backtesting during optimization",
    )
    p.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials")
    p.add_argument(
        "--objective",
        choices=["sharpe_ratio", "composite", "profit_factor", "win_rate"],
        default="sharpe_ratio",
        help="Objective to maximise",
    )
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel Optuna workers")
    p.add_argument("--train-days", type=int, default=365, help="Days of historical data")
    p.add_argument("--capital", type=float, default=50_000.0, help="Initial capital per backtest")
    p.add_argument("--optimize-ml", action="store_true", help="Also search ML hyperparameter spaces")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///data/optuna.db")
    p.add_argument("--study-name", type=str, default=None, help="Optuna study name (for resuming)")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write best params back to config.yaml after optimization",
    )
    p.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--output",
        type=str,
        default="reports/optimization_result.json",
        help="Path to save results JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from ait.optimization.optimizer import StrategyOptimizer

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

    optimizer = StrategyOptimizer(
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

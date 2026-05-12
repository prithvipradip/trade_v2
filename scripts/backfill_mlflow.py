"""Import existing reports/runs/ archives into MLflow (idempotent).

Each run directory is logged as an MLflow run under experiment
``walkforward_{symbol}``.  Re-running is safe — already-imported runs are
skipped based on the ``run_id`` tag.

Usage:
    python scripts/backfill_mlflow.py                  # all symbols
    python scripts/backfill_mlflow.py --symbol QQQ     # one symbol
    python scripts/backfill_mlflow.py --run-dir reports/runs/QQQ_2Y_...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


_RUNS_DIR = Path(__file__).resolve().parent.parent / "reports" / "runs"


def _read_initial_capital_from_snapshot(run_dir: Path) -> float | None:
    """Fallback: read initial_capital from config_snapshot.yaml if not in metadata."""
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        return None
    try:
        cfg = yaml.safe_load(snapshot.read_text()) or {}
        return cfg.get("backtest", {}).get("initial_capital")
    except Exception:
        return None


def import_run(run_dir: Path, client, experiment_id: str) -> str:
    """Import one run directory into MLflow. Returns 'imported' or 'skipped'."""
    meta_file = Path(run_dir) / "run_metadata.json"
    if not meta_file.exists():
        return "skipped"

    meta = json.loads(meta_file.read_text())
    run_id_tag = meta.get("run_id", run_dir.name)

    existing = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.run_id = '{run_id_tag}'",
    )
    if existing:
        return "skipped"

    mlflow_run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_id_tag,
        tags={
            "run_id":       run_id_tag,
            "symbol":       meta.get("symbol", ""),
            "strategy":     meta.get("strategy", ""),
            "git_commit":   meta.get("git_commit", ""),
            "git_branch":   meta.get("git_branch", ""),
            "optimization": meta.get("optimization", ""),
        },
    )
    mlflow_run_id = mlflow_run.info.run_id

    # Resolve initial_capital — may be missing in older archives
    initial_capital = meta.get("initial_capital")
    if initial_capital is None:
        initial_capital = _read_initial_capital_from_snapshot(Path(run_dir))

    params = {
        "symbol":             str(meta.get("symbol", "")),
        "strategy":           str(meta.get("strategy", "")),
        "train_days":         str(meta.get("train_days", "")),
        "test_days":          str(meta.get("test_days", "")),
        "step_days":          str(meta.get("step_days", "")),
        "gap_days":           str(meta.get("gap_days", "")),
        "wf_trials":          str(meta.get("wf_trials", "")),
        "initial_capital":    str(float(initial_capital)) if initial_capital is not None else "",
        "position_size_pct":  str(meta.get("position_size_pct", "")),
        "optimization":       str(meta.get("optimization", "")),
    }
    for key, val in params.items():
        if val:
            client.log_param(mlflow_run_id, key=key, value=val)

    summary = meta.get("summary", {})
    summary_metrics = {
        "total_pnl":       summary.get("total_pnl", 0.0),
        "total_return_pct": summary.get("total_return_pct", 0.0),
        "win_rate":        summary.get("win_rate", 0.0),
        "sharpe_ratio":    summary.get("sharpe_ratio", 0.0),
        "max_drawdown_pct": summary.get("max_drawdown_pct", 0.0),
        "profit_factor":   summary.get("profit_factor", 0.0),
        "total_trades":    float(summary.get("total_trades", 0)),
    }
    for key, val in summary_metrics.items():
        client.log_metric(mlflow_run_id, key=key, value=val)

    for window in meta.get("windows", []):
        step = window.get("window", 0)
        client.log_metric(mlflow_run_id, key="w_pnl",     value=window.get("pnl", 0.0),      step=step)
        client.log_metric(mlflow_run_id, key="w_trades",  value=float(window.get("trades", 0)), step=step)
        client.log_metric(mlflow_run_id, key="w_win_rate", value=window.get("win_rate", 0.0), step=step)
        client.log_metric(mlflow_run_id, key="w_sharpe",  value=window.get("sharpe", 0.0),    step=step)

    client.log_artifacts(mlflow_run_id, local_dir=str(run_dir))
    client.set_terminated(mlflow_run_id)
    return "imported"


def backfill(runs_dir: Path, symbol: str | None = None) -> None:
    """Import all matching run archives into MLflow."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        print(f"Runs directory not found: {runs_dir}")
        return

    for meta_file in sorted(runs_dir.glob("*/run_metadata.json")):
        run_dir = meta_file.parent
        try:
            meta = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            print(f"  Skipping {run_dir.name}: unreadable metadata")
            continue

        sym = meta.get("symbol", "")
        if symbol and sym.upper() != symbol.upper():
            continue

        experiment_name = f"walkforward_{sym}"
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = exp.experiment_id

        status = import_run(run_dir, client, experiment_id)
        run_id_tag = meta.get("run_id", run_dir.name)
        print(f"  {status.upper()}: {run_id_tag} → experiment {experiment_name!r}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="backfill_mlflow",
        description="Import existing run archives into MLflow (idempotent).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--symbol", metavar="SYMBOL", help="Filter by symbol (e.g. QQQ).")
    group.add_argument("--run-dir", metavar="DIR",
                       help="Import a single specific run directory.")
    parser.add_argument(
        "--runs-dir", default=str(_RUNS_DIR), metavar="DIR",
        help=f"Directory containing run archives (default: {_RUNS_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print("Backfilling MLflow from run archives…")

    if args.run_dir:
        import mlflow
        from mlflow.tracking import MlflowClient
        run_dir = Path(args.run_dir)
        meta = json.loads((run_dir / "run_metadata.json").read_text())
        sym = meta.get("symbol", "unknown")
        experiment_name = f"walkforward_{sym}"
        exp = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = exp.experiment_id if exp else mlflow.create_experiment(experiment_name)
        client = MlflowClient()
        status = import_run(run_dir, client, experiment_id)
        print(f"  {status.upper()}: {run_dir.name} → experiment {experiment_name!r}")
    else:
        backfill(Path(args.runs_dir), symbol=args.symbol)

    print("Done. Run `mlflow ui` to browse experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

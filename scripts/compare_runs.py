"""Compare walk-forward run archives side-by-side.

Usage:
    python scripts/compare_runs.py --symbol QQQ
    python scripts/compare_runs.py --symbol QQQ --sort win_rate
    python scripts/compare_runs.py --runs-dir path/to/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_RUNS_DIR = Path(__file__).resolve().parent.parent / "reports" / "runs"


def _resolve_initial_capital(data: dict, run_dir: Path) -> float | None:
    """Return initial_capital from metadata, falling back to config_snapshot.yaml."""
    if "initial_capital" in data:
        return data["initial_capital"]
    snapshot = run_dir / "config_snapshot.yaml"
    if snapshot.exists():
        try:
            import yaml
            cfg = yaml.safe_load(snapshot.read_text()) or {}
            return cfg.get("backtest", {}).get("initial_capital")
        except Exception:
            pass
    return None


def load_runs(runs_dir: Path, symbol: str | None = None) -> list[dict]:
    """Load all run_metadata.json files under runs_dir, optionally filtered by symbol."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    results = []
    for meta_file in sorted(runs_dir.glob("*/run_metadata.json")):
        try:
            data = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if symbol and data.get("symbol", "").upper() != symbol.upper():
            continue
        # Resolve initial_capital from config_snapshot if missing in older archives
        if "initial_capital" not in data:
            cap = _resolve_initial_capital(data, meta_file.parent)
            if cap is not None:
                data["initial_capital"] = cap
        results.append(data)
    return results


def format_table(runs: list[dict], sort_by: str = "sharpe_ratio") -> str:
    """Return a formatted comparison table string, sorted by sort_by metric."""
    if not runs:
        return "(no runs found)"

    def _sort_key(r: dict) -> float:
        return r.get("summary", {}).get(sort_by, float("-inf"))

    sorted_runs = sorted(runs, key=_sort_key, reverse=True)

    header = (
        f"{'Run ID':<52} {'train':>5} {'test':>4} {'step':>4} "
        f"{'capital':>9} {'trials':>6} {'trades':>6} {'win%':>5} "
        f"{'sharpe':>7} {'dd%':>6} {'PnL':>10}"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in sorted_runs:
        s = r.get("summary", {})
        capital = r.get("initial_capital", 0)
        capital_str = f"${capital:,.0f}" if capital else "?"
        lines.append(
            f"{r.get('run_id', '?'):<52} "
            f"{r.get('train_days', '?'):>5} "
            f"{r.get('test_days', '?'):>4} "
            f"{r.get('step_days', '?'):>4} "
            f"{capital_str:>9} "
            f"{r.get('wf_trials', '?'):>6} "
            f"{s.get('total_trades', 0):>6} "
            f"{100*s.get('win_rate', 0):>5.1f} "
            f"{s.get('sharpe_ratio', 0):>7.3f} "
            f"{100*s.get('max_drawdown_pct', 0):>6.1f} "
            f"{s.get('total_pnl', 0):>+10,.0f}"
        )

    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="compare_runs",
        description="Compare walk-forward run archives side-by-side.",
    )
    parser.add_argument("--symbol", metavar="SYMBOL", help="Filter by symbol (e.g. QQQ).")
    parser.add_argument(
        "--runs-dir", default=str(_RUNS_DIR), metavar="DIR",
        help=f"Directory containing run archives (default: {_RUNS_DIR}).",
    )
    parser.add_argument(
        "--sort", default="sharpe_ratio",
        choices=["sharpe_ratio", "total_pnl", "win_rate", "max_drawdown_pct", "total_trades"],
        help="Metric to sort by, descending (default: sharpe_ratio).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs = load_runs(Path(args.runs_dir), symbol=args.symbol)
    symbol_label = args.symbol.upper() if args.symbol else "ALL"
    print(f"\n{symbol_label} Walk-Forward Run Comparison  (sorted by {args.sort})")
    print("=" * 100)
    print(format_table(runs, sort_by=args.sort))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

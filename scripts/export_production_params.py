"""Export the most recently trained window's best_params as a production config YAML.

Usage:
    # Auto-find latest run for QQQ:
    python scripts/export_production_params.py --symbol QQQ

    # Explicit run directory:
    python scripts/export_production_params.py \\
        --run-dir reports/runs/QQQ_2Y_iron_condor_per_strategy_20260512 \\
        --base-config config_QQQ_test.yaml \\
        --output config_QQQ_production.yaml

    # Preview only (no file written):
    python scripts/export_production_params.py --symbol QQQ --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


_RUNS_DIR = Path(__file__).resolve().parent.parent / "reports" / "runs"

# Maps bare param name → (config_section, config_field).
# Params not in this table fall back to the "backtest" section.
_PARAM_MAP: dict[str, tuple[str, str]] = {
    "min_confidence":         ("risk",    "min_confidence"),
    "stop_loss_pct":          ("exit",    "initial_stop_loss_pct"),
    "trailing_stop_pct":      ("exit",    "trailing_stop_pct"),
    "breakeven_trigger_pct":  ("exit",    "breakeven_trigger_pct"),
    "profit_target_pct":      ("backtest", "profit_target_pct"),
    "delta_short":            ("backtest", "delta_short"),
    "max_hold_days":          ("backtest", "max_hold_days"),
    "iv_floor":               ("backtest", "iv_floor"),
    "wing_k":                 ("backtest", "wing_k"),
    "max_entry_vol_annual":   ("backtest", "max_entry_vol_annual"),
    "hurst_regime_threshold": ("backtest", "hurst_regime_threshold"),
    "hurst_regime_penalty":   ("backtest", "hurst_regime_penalty"),
    "multifractal_max_width": ("backtest", "multifractal_max_width"),
    "delta_long":             ("backtest", "delta_long"),
    "delta_iv_scale":         ("backtest", "delta_iv_scale"),
}


def _resolve_initial_capital(meta: dict, run_dir: Path) -> float | None:
    """Return initial_capital from metadata, falling back to config_snapshot.yaml."""
    if "initial_capital" in meta:
        return meta["initial_capital"]
    snapshot = run_dir / "config_snapshot.yaml"
    if snapshot.exists():
        try:
            cfg = yaml.safe_load(snapshot.read_text()) or {}
            return cfg.get("backtest", {}).get("initial_capital")
        except Exception:
            pass
    return None


def find_latest_run(runs_dir: Path, symbol: str) -> Path:
    """Return the most recently modified run directory matching symbol."""
    runs_dir = Path(runs_dir)
    candidates = []
    for meta_file in runs_dir.glob("*/run_metadata.json"):
        try:
            data = json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("symbol", "").upper() == symbol.upper():
            candidates.append((meta_file.stat().st_mtime, meta_file.parent))
    if not candidates:
        raise FileNotFoundError(f"No run found for symbol {symbol!r} under {runs_dir}")
    return sorted(candidates, reverse=True)[0][1]


def extract_deployment_params(run_dir: Path) -> dict:
    """Return best_params from the last window with trades > 0."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    windows = meta.get("windows", [])
    active = [w for w in windows if w.get("trades", 0) > 0 and w.get("best_params")]
    if not active:
        raise ValueError(f"no active windows with trades in {run_dir}")
    last = active[-1]
    return last["best_params"]


def apply_params_to_config(
    params: dict,
    base_config_path: str,
    output_path: str,
    dry_run: bool = False,
) -> dict[str, tuple]:
    """Apply best_params to base config and write output. Returns mapping of changes."""
    base = Path(base_config_path)
    data: dict = {}
    if base.exists():
        data = yaml.safe_load(base.read_text()) or {}

    changes: dict[str, tuple] = {}
    for key, val in params.items():
        _, _, param_name = key.partition("__")
        section, field = _PARAM_MAP.get(param_name, ("backtest", param_name))
        old = data.get(section, {}).get(field)
        data.setdefault(section, {})[field] = val
        changes[key] = (old, val, section, field)

    if not dry_run:
        Path(output_path).write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False)
        )
    return changes


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_production_params",
        description="Export walk-forward best params as a production config YAML.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--symbol", metavar="SYMBOL",
                       help="Auto-find latest run for this symbol.")
    group.add_argument("--run-dir", metavar="DIR",
                       help="Explicit run archive directory.")
    parser.add_argument(
        "--base-config", default="config.yaml", metavar="PATH",
        help="Base config YAML to merge params into (default: config.yaml).",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output YAML path (default: config_{symbol}_production.yaml).",
    )
    parser.add_argument(
        "--runs-dir", default=str(_RUNS_DIR), metavar="DIR",
        help=f"Directory containing run archives (default: {_RUNS_DIR}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print changes without writing the output file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.symbol:
        run_dir = find_latest_run(Path(args.runs_dir), args.symbol)
    else:
        print("Error: specify --symbol or --run-dir")
        return 1

    meta = json.loads((run_dir / "run_metadata.json").read_text())
    windows = meta.get("windows", [])
    active = [w for w in windows if w.get("trades", 0) > 0 and w.get("best_params")]

    params = extract_deployment_params(run_dir)
    last_active = active[-1]
    symbol = meta.get("symbol", "?")
    initial_capital = _resolve_initial_capital(meta, run_dir)
    capital_str = f"${initial_capital:,.0f}" if isinstance(initial_capital, (int, float)) else "?"

    output_path = args.output or f"config_{symbol}_production.yaml"

    print(f"\nRun:             {meta.get('run_id', run_dir.name)}")
    print(f"Commit:          {meta.get('git_commit', '?')[:8]} ({meta.get('git_branch', '?')})")
    print(f"Symbol:          {symbol}  |  Strategy: {meta.get('strategy', '?')}")
    print(f"Initial capital: {capital_str}")
    print(
        f"Source window:   W{last_active['window']} "
        f"({last_active.get('test_start', '?')} → {last_active.get('test_end', '?')}, "
        f"{last_active['trades']} trades, "
        f"win_rate={100*last_active.get('win_rate', 0):.0f}%)"
    )
    print(f"\nParameters to apply → {output_path}:")

    changes = apply_params_to_config(params, args.base_config, output_path, dry_run=args.dry_run)
    for key, (old, new, section, field) in sorted(changes.items()):
        old_str = f"{old:.4f}" if isinstance(old, float) else str(old) if old is not None else "(absent)"
        new_str = f"{new:.4f}" if isinstance(new, float) else str(new)
        print(f"  {key:<40}  {old_str:>10}  →  {new_str}")

    if args.dry_run:
        print("\n(dry-run: no file written)")
    else:
        print(f"\nWritten: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

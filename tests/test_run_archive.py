import json
import re
import argparse
from pathlib import Path

from scripts.run_integration_test import _create_run_archive


def test_archive_fixes(tmp_path, monkeypatch):
    # Arrange — fake integration_test output dir with 3 window files
    out = tmp_path / "integration_test"
    out.mkdir()
    for i in range(1, 4):
        w = {
            "window": i,
            "pnl": 100.0 * i,
            "trades": i,
            "win_rate": 1.0,
            "sharpe": 5.0,
            "test_start": f"2025-0{i}-01",
            "test_end": f"2025-0{i}-28",
        }
        (out / f"window_{i:03d}.json").write_text(json.dumps(w))

    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports" / "runs").mkdir(parents=True)
    (tmp_path / "config_QQQ_test.yaml").write_text(
        "backtest:\n  initial_capital: 100000\n"
    )

    args = argparse.Namespace(
        symbols=["QQQ"],
        strategies=["iron_condor"],
        config="config_QQQ_test.yaml",
        output_dir=str(out),
        skip_backfill=True,
        years=1,
        port=7497,
        train_days=365,
        test_days=42,
        step_days=14,
        gap_days=5,
        optuna_seed=42,
        wf_trials=50,
        wf_patience=None,
    )
    wf = {
        "total_trades": 6,
        "total_return": 0.06,
        "sharpe_ratio": 5.0,
        "max_drawdown": 0.01,
        "win_rate": 1.0,
        "consistency": 1.0,
        "n_windows": 3,
        "profit_factor": 12.5,
    }
    wf_cfg_fields = {
        "train_days": 365,
        "initial_capital": 100_000,
        "position_size_pct": 0.05,
        "optuna_seed": 42,
        "optimize_n_trials": 50,
    }

    # Act
    archive_path = _create_run_archive(args, out, wf, wf_cfg_fields)

    # Assert
    assert archive_path is not None
    meta = json.loads((archive_path / "run_metadata.json").read_text())

    # Bug 1: n_windows must not be 0
    assert meta["n_windows"] == 3, f"n_windows was {meta['n_windows']}, expected 3"
    assert meta["active_windows"] == 3

    # Bug 2: run_id must include HHmm time component (underscore + 4 digits)
    assert re.search(r"_\d{8}_\d{4}$", meta["run_id"]), (
        f"run_id '{meta['run_id']}' missing _YYYYMMDD_HHMM suffix"
    )

    # Bonus: profit_factor must be populated from wf dict
    assert meta["summary"]["profit_factor"] == 12.5

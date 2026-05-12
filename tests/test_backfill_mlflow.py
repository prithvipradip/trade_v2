"""Tests for scripts/backfill_mlflow.py"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from scripts.backfill_mlflow import import_run


def _make_run_dir(
    tmp_path: Path,
    run_id: str = "test_run",
    symbol: str = "QQQ",
    initial_capital: float = 100_000.0,
) -> Path:
    d = tmp_path / run_id
    d.mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "symbol": symbol,
        "strategy": "iron_condor",
        "train_days": 365,
        "test_days": 63,
        "step_days": 21,
        "wf_trials": 50,
        "initial_capital": initial_capital,
        "position_size_pct": 0.05,
        "optimization": "per_strategy",
        "git_commit": "abc1234",
        "git_branch": "features-request-2",
        "summary": {
            "total_trades": 50,
            "total_pnl": 28038.0,
            "total_return_pct": 28.04,
            "win_rate": 0.56,
            "sharpe_ratio": 0.82,
            "max_drawdown_pct": 0.148,
            "profit_factor": 1.4,
        },
        "windows": [
            {
                "window": 1,
                "pnl": 1000.0,
                "trades": 5,
                "win_rate": 0.6,
                "sharpe": 0.7,
                "best_params": {"iron_condor__wing_k": 0.93},
            },
        ],
    }
    (d / "run_metadata.json").write_text(json.dumps(meta))
    return d


class TestImportRun:
    def _make_client(self) -> MagicMock:
        client = MagicMock()
        mock_run = MagicMock()
        mock_run.info.run_id = "mlflow_run_123"
        client.create_run.return_value = mock_run
        client.search_runs.return_value = []  # default: no existing runs
        return client

    def test_logs_params_including_initial_capital(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path, initial_capital=100_000.0)
        client = self._make_client()

        import_run(run_dir, client, experiment_id="1")

        # log_param is called as: client.log_param(run_id, key=k, value=v)
        logged_params = {
            c.kwargs.get("key"): c.kwargs.get("value")
            for c in client.log_param.call_args_list
        }
        assert "train_days" in logged_params
        assert logged_params["train_days"] == "365"
        assert "initial_capital" in logged_params
        assert logged_params["initial_capital"] == "100000.0"

    def test_logs_summary_metrics(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path)
        client = self._make_client()

        import_run(run_dir, client, experiment_id="1")

        # log_metric is called as: client.log_metric(run_id, key=k, value=v)
        logged_metric_keys = {
            c.kwargs.get("key")
            for c in client.log_metric.call_args_list
        }
        assert "total_pnl" in logged_metric_keys
        assert "sharpe_ratio" in logged_metric_keys
        assert "win_rate" in logged_metric_keys

    def test_logs_per_window_metrics(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path)
        client = self._make_client()

        import_run(run_dir, client, experiment_id="1")

        metric_calls = client.log_metric.call_args_list
        stepped_keys = set()
        for c in metric_calls:
            args = c.args
            kwargs = c.kwargs
            key = args[1] if len(args) > 1 else kwargs.get("key")
            step = args[3] if len(args) > 3 else kwargs.get("step")
            if step == 1:
                stepped_keys.add(key)
        assert "w_pnl" in stepped_keys or "w_trades" in stepped_keys

    def test_is_idempotent_skips_existing_run(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path, run_id="existing_run")
        client = self._make_client()
        client.search_runs.return_value = [MagicMock()]  # existing run found

        result = import_run(run_dir, client, experiment_id="1")
        assert result == "skipped"
        client.create_run.assert_not_called()

    def test_returns_imported_on_success(self, tmp_path: Path):
        run_dir = _make_run_dir(tmp_path)
        client = self._make_client()

        result = import_run(run_dir, client, experiment_id="1")
        assert result == "imported"

    def test_returns_skipped_when_no_metadata(self, tmp_path: Path):
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        client = self._make_client()

        result = import_run(empty_dir, client, experiment_id="1")
        assert result == "skipped"

    def test_fallback_reads_initial_capital_from_config_snapshot(self, tmp_path: Path):
        """When initial_capital is missing from metadata, fall back to config_snapshot.yaml."""
        import yaml as _yaml

        run_dir = tmp_path / "old_run"
        run_dir.mkdir()
        meta_without_capital = {
            "run_id": "old_run",
            "symbol": "QQQ",
            "strategy": "iron_condor",
            "train_days": 365,
            "test_days": 63,
            "step_days": 21,
            "wf_trials": 50,
            "git_commit": "abc",
            "summary": {"total_trades": 10, "total_pnl": 1000.0, "win_rate": 0.6,
                        "sharpe_ratio": 0.8, "max_drawdown_pct": 0.1, "profit_factor": 1.2},
            "windows": [],
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(meta_without_capital))
        (run_dir / "config_snapshot.yaml").write_text(
            _yaml.dump({"backtest": {"initial_capital": 75_000.0}})
        )

        client = self._make_client()
        import_run(run_dir, client, experiment_id="1")

        logged_params = {
            c.kwargs.get("key"): c.kwargs.get("value")
            for c in client.log_param.call_args_list
        }
        assert logged_params.get("initial_capital") == "75000.0"

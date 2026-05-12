"""Tests for scripts/export_production_params.py"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.export_production_params import (
    apply_params_to_config,
    extract_deployment_params,
    find_latest_run,
)


def _make_run_dir(
    tmp_path: Path,
    run_id: str = "test_run",
    symbol: str = "QQQ",
    windows: list | None = None,
    initial_capital: float = 100_000.0,
) -> Path:
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    if windows is None:
        windows = [
            {
                "window": 1,
                "trades": 5,
                "win_rate": 0.6,
                "test_start": "2025-01-01",
                "test_end": "2025-03-31",
                "best_params": {
                    "iron_condor__min_confidence": 0.70,
                    "iron_condor__delta_short": 0.16,
                    "iron_condor__wing_k": 0.93,
                },
            },
            {
                "window": 2,
                "trades": 0,
                "win_rate": 0.0,
                "best_params": {},
                "note": "no_trades",
            },
        ]
    meta = {
        "run_id": run_id,
        "symbol": symbol,
        "strategy": "iron_condor",
        "initial_capital": initial_capital,
        "position_size_pct": 0.05,
        "git_commit": "abc1234",
        "git_branch": "features-request-2",
        "windows": windows,
    }
    (d / "run_metadata.json").write_text(json.dumps(meta))
    return d


class TestFindLatestRun:
    def test_picks_most_recently_modified(self, tmp_path: Path):
        d1 = _make_run_dir(tmp_path, "run_20260510")
        import time
        time.sleep(0.02)
        d2 = _make_run_dir(tmp_path, "run_20260512")
        result = find_latest_run(tmp_path / "runs", symbol="QQQ")
        assert result == d2

    def test_raises_when_no_match(self, tmp_path: Path):
        _make_run_dir(tmp_path, "spy_run", symbol="SPY")
        with pytest.raises(FileNotFoundError):
            find_latest_run(tmp_path / "runs", symbol="QQQ")

    def test_case_insensitive(self, tmp_path: Path):
        _make_run_dir(tmp_path, "qqq_run", symbol="QQQ")
        result = find_latest_run(tmp_path / "runs", symbol="qqq")
        assert result.name == "qqq_run"


class TestExtractDeploymentParams:
    def test_skips_zero_trade_windows(self, tmp_path: Path):
        d = _make_run_dir(tmp_path)
        params = extract_deployment_params(d)
        assert params["iron_condor__min_confidence"] == pytest.approx(0.70)

    def test_uses_last_active_window(self, tmp_path: Path):
        windows = [
            {"window": 1, "trades": 3, "best_params": {"iron_condor__wing_k": 0.80}},
            {"window": 2, "trades": 7, "best_params": {"iron_condor__wing_k": 0.95}},
            {"window": 3, "trades": 0, "best_params": {}, "note": "no_trades"},
        ]
        d = _make_run_dir(tmp_path, windows=windows)
        params = extract_deployment_params(d)
        assert params["iron_condor__wing_k"] == pytest.approx(0.95)

    def test_raises_when_all_windows_empty(self, tmp_path: Path):
        windows = [{"window": 1, "trades": 0, "best_params": {}, "note": "no_trades"}]
        d = _make_run_dir(tmp_path, windows=windows)
        with pytest.raises(ValueError, match="no active windows"):
            extract_deployment_params(d)

    def test_single_active_window(self, tmp_path: Path):
        windows = [{"window": 1, "trades": 5, "best_params": {"iron_condor__iv_floor": 0.22}}]
        d = _make_run_dir(tmp_path, windows=windows)
        params = extract_deployment_params(d)
        assert params["iron_condor__iv_floor"] == pytest.approx(0.22)


class TestApplyParamsToConfig:
    def test_writes_yaml_with_overrides(self, tmp_path: Path):
        base_cfg = {
            "backtest": {
                "initial_capital": 100_000.0,
                "min_confidence": 0.55,
                "wing_k": 1.0,
            }
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(base_cfg))
        out_path = tmp_path / "config_production.yaml"

        params = {
            "iron_condor__min_confidence": 0.70,
            "iron_condor__wing_k": 0.93,
        }
        apply_params_to_config(params, str(cfg_path), str(out_path))

        result = yaml.safe_load(out_path.read_text())
        assert result["risk"]["min_confidence"] == pytest.approx(0.70)
        assert result["backtest"]["wing_k"] == pytest.approx(0.93)

    def test_preserves_initial_capital(self, tmp_path: Path):
        base_cfg = {
            "backtest": {"initial_capital": 100_000.0, "min_confidence": 0.55},
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(base_cfg))
        out_path = tmp_path / "config_production.yaml"

        apply_params_to_config({"iron_condor__wing_k": 0.93}, str(cfg_path), str(out_path))
        result = yaml.safe_load(out_path.read_text())
        assert result["backtest"]["initial_capital"] == pytest.approx(100_000.0)

    def test_dry_run_does_not_write(self, tmp_path: Path):
        base_cfg = {"backtest": {"min_confidence": 0.55}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(base_cfg))
        out_path = tmp_path / "config_production.yaml"

        apply_params_to_config({}, str(cfg_path), str(out_path), dry_run=True)
        assert not out_path.exists()

    def test_unmapped_param_falls_back_to_backtest_section(self, tmp_path: Path):
        base_cfg = {"backtest": {}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(base_cfg))
        out_path = tmp_path / "config_production.yaml"

        apply_params_to_config(
            {"iron_condor__some_unknown_param": 0.42}, str(cfg_path), str(out_path)
        )
        result = yaml.safe_load(out_path.read_text())
        assert result["backtest"]["some_unknown_param"] == pytest.approx(0.42)

    def test_returns_changes_dict(self, tmp_path: Path):
        base_cfg = {"backtest": {"wing_k": 1.0}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(base_cfg))
        out_path = tmp_path / "config_production.yaml"

        changes = apply_params_to_config(
            {"iron_condor__wing_k": 0.85}, str(cfg_path), str(out_path)
        )
        assert "iron_condor__wing_k" in changes
        old, new, section, field = changes["iron_condor__wing_k"]
        assert new == pytest.approx(0.85)
        assert field == "wing_k"

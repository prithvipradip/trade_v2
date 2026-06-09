"""Tests for scripts/compare_runs.py"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_runs import load_runs, format_table


def _make_run(
    tmp_path: Path,
    run_id: str,
    symbol: str = "QQQ",
    train_days: int = 365,
    test_days: int = 63,
    total_pnl: float = 28000.0,
    win_rate: float = 0.56,
    sharpe: float = 0.82,
    drawdown: float = 0.148,
    trades: int = 50,
    initial_capital: float = 100_000.0,
) -> Path:
    d = tmp_path / "runs" / run_id
    d.mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "symbol": symbol,
        "strategy": "iron_condor",
        "train_days": train_days,
        "test_days": test_days,
        "step_days": 21,
        "wf_trials": 50,
        "initial_capital": initial_capital,
        "position_size_pct": 0.05,
        "summary": {
            "total_trades": trades,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": drawdown,
            "profit_factor": 1.4,
        },
        "windows": [],
    }
    (d / "run_metadata.json").write_text(json.dumps(meta))
    return d


class TestLoadRuns:
    def test_finds_all_metadata(self, tmp_path: Path):
        _make_run(tmp_path, "run_A")
        _make_run(tmp_path, "run_B")
        runs = load_runs(tmp_path / "runs", symbol="QQQ")
        assert len(runs) == 2

    def test_filters_by_symbol(self, tmp_path: Path):
        _make_run(tmp_path, "qqq_run", symbol="QQQ")
        spy_dir = tmp_path / "runs" / "spy_run"
        spy_dir.mkdir(parents=True)
        (spy_dir / "run_metadata.json").write_text(
            json.dumps({"run_id": "spy_run", "symbol": "SPY", "summary": {}, "windows": []})
        )
        runs = load_runs(tmp_path / "runs", symbol="QQQ")
        assert len(runs) == 1
        assert runs[0]["run_id"] == "qqq_run"

    def test_case_insensitive_symbol_filter(self, tmp_path: Path):
        _make_run(tmp_path, "qqq_run", symbol="QQQ")
        runs = load_runs(tmp_path / "runs", symbol="qqq")
        assert len(runs) == 1

    def test_no_filter_returns_all(self, tmp_path: Path):
        _make_run(tmp_path, "run_qqq", symbol="QQQ")
        _make_run(tmp_path, "run_spy", symbol="SPY")
        runs = load_runs(tmp_path / "runs", symbol=None)
        assert len(runs) == 2

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        runs = load_runs(tmp_path / "nonexistent", symbol="QQQ")
        assert runs == []

    def test_skips_unreadable_metadata(self, tmp_path: Path):
        bad_dir = tmp_path / "runs" / "bad_run"
        bad_dir.mkdir(parents=True)
        (bad_dir / "run_metadata.json").write_text("not valid json{{")
        runs = load_runs(tmp_path / "runs", symbol=None)
        assert runs == []


class TestFormatTable:
    def test_sorted_by_sharpe_descending(self, tmp_path: Path):
        _make_run(tmp_path, "low_sharpe", sharpe=0.5)
        _make_run(tmp_path, "high_sharpe", sharpe=1.2)
        runs = load_runs(tmp_path / "runs", symbol="QQQ")
        table = format_table(runs, sort_by="sharpe_ratio")
        assert table.index("high_sharpe") < table.index("low_sharpe")

    def test_contains_train_test_days(self, tmp_path: Path):
        _make_run(tmp_path, "run_A", train_days=365, test_days=63)
        runs = load_runs(tmp_path / "runs", symbol="QQQ")
        table = format_table(runs)
        assert "365" in table
        assert "63" in table

    def test_contains_pnl(self, tmp_path: Path):
        _make_run(tmp_path, "run_A", total_pnl=28038.0)
        runs = load_runs(tmp_path / "runs", symbol="QQQ")
        table = format_table(runs)
        assert "28038" in table or "28,038" in table

    def test_contains_initial_capital(self, tmp_path: Path):
        _make_run(tmp_path, "run_A", initial_capital=100_000.0)
        runs = load_runs(tmp_path / "runs", symbol="QQQ")
        table = format_table(runs)
        assert "100,000" in table or "100000" in table

    def test_empty_input_returns_no_runs_message(self):
        table = format_table([])
        assert "no runs" in table.lower() or len(table) < 30

    def test_no_crash_on_missing_summary_fields(self, tmp_path: Path):
        d = tmp_path / "runs" / "sparse_run"
        d.mkdir(parents=True)
        (d / "run_metadata.json").write_text(
            json.dumps({"run_id": "sparse_run", "symbol": "QQQ", "summary": {}, "windows": []})
        )
        runs = load_runs(tmp_path / "runs", symbol="QQQ")
        table = format_table(runs)
        assert "sparse_run" in table

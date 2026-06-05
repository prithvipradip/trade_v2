"""Tests for the Walk-Forward Analysis Dashboard exporter (Layer 3).

These tests use only standard-library / minimal dependencies so they run
without a live database or heavy ML packages.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ait.dashboard.walkforward.export import (
    _build_optuna_stubs,
    _build_trades,
    _days_between,
    _parse_backtest_period,
    _round,
    build_ait,
    write_wf_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trade(
    entry_date: str = "2025-06-01",
    exit_date: str = "2025-06-10",
    pnl: float = 200.0,
    exit_reason: str = "profit_target",
    entry_confidence: float = 0.72,
    entry_regime: str = "range_bound",
) -> dict:
    return {
        "symbol": "QQQ",
        "strategy": "iron_condor",
        "entry_date": entry_date,
        "entry_time": entry_date,
        "exit_date": exit_date,
        "exit_time": exit_date,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "entry_confidence": entry_confidence,
        "entry_regime": entry_regime,
    }


def _make_window(window_id: int = 1, pnl: float = 300.0, trades: int = 2) -> dict:
    return {
        "window": window_id,
        "test_start": "2025-06-01",
        "test_end": "2025-06-30",
        "pnl": pnl,
        "return_pct": pnl / 1000.0,
        "trades": trades,
        "win_rate": 0.6,
        "sharpe": 1.5,
        "max_drawdown": 0.05,
        "strategies": {"iron_condor": trades},
        "best_params": {"iron_condor__stop_loss_pct": 0.4},
        "trades_detail": [_make_trade() for _ in range(trades)],
    }


def _minimal_metadata(tmp_path: Path, windows: list[dict] | None = None) -> Path:
    """Write a minimal run_metadata.json and equity_curve.csv to tmp_path."""
    if windows is None:
        windows = [_make_window(1, 300.0, 2), _make_window(2, -100.0, 1)]

    meta = {
        "run_id": "TEST_EXP_001",
        "run_date": "2026-01-01",
        "symbol": "QQQ",
        "strategy": "iron_condor",
        "optimization": "per_strategy",
        "n_windows": len(windows),
        "active_windows": sum(1 for w in windows if w["trades"] > 0),
        "train_days": 365,
        "test_days": 30,
        "step_days": 30,
        "gap_days": 5,
        "wf_trials": 50,
        "optuna_seed": 42,
        "initial_capital": 100_000.0,
        "position_size_pct": 0.05,
        "backtest_period": "2025-06-01 to 2025-12-31",
        "cli_command": "test",
        "git_branch": "main",
        "git_commit": "abc1234",
        "summary": {
            "total_trades": sum(w["trades"] for w in windows),
            "total_pnl": sum(w["pnl"] for w in windows),
            "total_return_pct": sum(w["pnl"] for w in windows) / 1000.0,
            "win_rate": 0.6,
            "sharpe_ratio": 1.5,
            "max_drawdown_pct": 0.05,
            "profit_factor": 1.8,
        },
        "windows": windows,
    }
    (tmp_path / "run_metadata.json").write_text(json.dumps(meta))

    # equity_curve.csv
    csv_lines = ["date,equity,pnl,strategy,symbol,window"]
    eq = 100_000.0
    for w in windows:
        for t in w["trades_detail"]:
            eq += t["pnl"]
            csv_lines.append(f"{t['exit_date']},{eq},{t['pnl']},iron_condor,QQQ,{w['window']}")
    (tmp_path / "equity_curve.csv").write_text("\n".join(csv_lines))

    return tmp_path


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_round_none(self):
        assert _round(None) is None

    def test_round_nan(self):
        import math
        assert _round(math.nan) is None

    def test_round_normal(self):
        assert _round(3.14159, 2) == 3.14

    def test_days_between(self):
        assert _days_between("2025-06-01", "2025-06-11") == 10

    def test_days_between_invalid(self):
        assert _days_between("", "") == 0

    def test_parse_backtest_period(self):
        start, end = _parse_backtest_period("2025-05-14 to 2026-05-13")
        assert start == "2025-05-14"
        assert end == "2026-05-13"

    def test_parse_backtest_period_empty(self):
        assert _parse_backtest_period("") == ("", "")


# ---------------------------------------------------------------------------
# _build_trades
# ---------------------------------------------------------------------------

class TestBuildTrades:
    def test_ids_sequential(self):
        windows = [_make_window(1, 100.0, 2), _make_window(2, 200.0, 1)]
        trades = _build_trades(windows, "QQQ")
        assert [t["id"] for t in trades] == ["T001", "T002", "T003"]

    def test_window_id_assigned(self):
        windows = [_make_window(1, 100.0, 2), _make_window(5, 200.0, 1)]
        trades = _build_trades(windows, "QQQ")
        assert trades[0]["window_id"] == 1
        assert trades[2]["window_id"] == 5

    def test_hold_days_computed(self):
        w = _make_window(1, 100.0, 1)
        w["trades_detail"] = [_make_trade("2025-06-01", "2025-06-11")]
        trades = _build_trades([w], "QQQ")
        assert trades[0]["hold_days"] == 10

    def test_pnl_preserved(self):
        w = _make_window(1, 100.0, 1)
        w["trades_detail"] = [_make_trade(pnl=555.55)]
        trades = _build_trades([w], "QQQ")
        assert trades[0]["pnl"] == 555.55

    def test_return_pct_is_none(self):
        # return_pct requires max_loss — should be None until Layer 2a
        trades = _build_trades([_make_window()], "QQQ")
        assert trades[0]["return_pct"] is None

    def test_n_legs_is_four(self):
        trades = _build_trades([_make_window()], "QQQ")
        assert trades[0]["n_legs"] == 4

    def test_direction_is_neutral(self):
        trades = _build_trades([_make_window()], "QQQ")
        assert trades[0]["direction"] == "neutral"

    def test_legs_empty_without_layer2(self):
        # legs not in trades_detail yet → empty list
        trades = _build_trades([_make_window()], "QQQ")
        assert trades[0]["legs"] == []

    def test_decision_empty_without_layer2(self):
        trades = _build_trades([_make_window()], "QQQ")
        assert trades[0]["decision"] == {}

    def test_features_at_entry_empty_without_layer2(self):
        trades = _build_trades([_make_window()], "QQQ")
        assert trades[0]["features_at_entry"] == {}

    def test_entry_confidence_preserved(self):
        w = _make_window(1, 100.0, 1)
        w["trades_detail"] = [_make_trade(entry_confidence=0.85)]
        trades = _build_trades([w], "QQQ")
        assert trades[0]["entry_confidence"] == 0.85

    def test_empty_windows(self):
        assert _build_trades([], "QQQ") == []


# ---------------------------------------------------------------------------
# _build_optuna_stubs
# ---------------------------------------------------------------------------

class TestBuildOptunaStubs:
    def _meta(self):
        return {"symbol": "QQQ", "strategy": "iron_condor", "wf_trials": 50, "optimize_patience": 20}

    def test_keys_are_window_ids(self):
        windows = [_make_window(3), _make_window(7)]
        stubs = _build_optuna_stubs(windows, self._meta())
        assert set(stubs.keys()) == {3, 7}

    def test_best_params_preserved(self):
        w = _make_window(1)
        w["best_params"] = {"iron_condor__stop_loss_pct": 0.4, "iron_condor__delta_short": 0.25}
        stubs = _build_optuna_stubs([w], self._meta())
        assert stubs[1]["best_params"]["iron_condor__stop_loss_pct"] == 0.4

    def test_no_trial_history_flag(self):
        stubs = _build_optuna_stubs([_make_window()], self._meta())
        assert stubs[1]["_has_trial_history"] is False
        assert stubs[1]["trials"] == []

    def test_with_trial_history(self):
        w = _make_window(1)
        w["optuna_trials"] = [
            {"number": 0, "state": "COMPLETE", "value": 0.3, "params": {"iron_condor__stop_loss_pct": 0.4}, "n_trades": 5},
            {"number": 1, "state": "PRUNED", "value": None, "params": {}, "n_trades": 2},
        ]
        w["optuna_meta"] = {"n_trials_run": 2, "status": "completed", "stop_reason": "Completed."}
        stubs = _build_optuna_stubs([w], self._meta())
        s = stubs[1]
        assert s["_has_trial_history"] is True
        assert len(s["trials"]) == 2
        assert s["best_value"] == 0.3
        assert s["best_trial"] == 0
        assert s["n_complete"] == 1
        assert s["n_pruned"] == 1

    def test_study_name_format(self):
        stubs = _build_optuna_stubs([_make_window(4)], self._meta())
        assert stubs[4]["study_name"] == "wf_w4_QQQ_iron_condor"


# ---------------------------------------------------------------------------
# build_ait (integration — no OHLCV, no FeatureEngine)
# ---------------------------------------------------------------------------

class TestBuildAit:
    def test_raises_on_missing_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_ait(tmp_path / "nonexistent")

    def test_required_ait_keys(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        required = {"experiment", "bars", "features", "predictions", "windows",
                    "trades", "equityCurve", "optuna_studies", "experiments", "FEATURE_LIBRARY"}
        assert required.issubset(ait.keys())

    def test_experiment_id_from_run_id(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert ait["experiment"]["id"] == "TEST_EXP_001"

    def test_total_return_is_decimal(self, tmp_path):
        # real data stores total_return_pct as percent (e.g., 0.2); exporter must divide by 100
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        r = ait["experiment"]["results"]
        # total_return_pct = (300 - 100) / 1000 = 0.2 → stored as 0.002
        assert r["total_return"] == pytest.approx(0.002, abs=1e-4)

    def test_window_return_pct_is_decimal(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        # window 1: return_pct stored as 300/1000 = 0.3 in metadata → exporter divides by 100 → 0.003
        assert ait["windows"][0]["return_pct"] == pytest.approx(0.003, abs=1e-5)

    def test_windows_count(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert len(ait["windows"]) == 2

    def test_trades_from_all_windows(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        # window 1 has 2 trades, window 2 has 1 → total 3
        assert len(ait["trades"]) == 3

    def test_equity_curve_populated(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert len(ait["equityCurve"]) == 3
        assert ait["equityCurve"][0]["equity"] > 100_000.0

    def test_consistency_metric(self, tmp_path):
        # 1 profitable window (pnl=300), 1 losing (pnl=-100) → 1/2
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert ait["experiment"]["results"]["consistency"] == pytest.approx(0.5)

    def test_optuna_stubs_keyed_by_window_id(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert "1" in ait["optuna_studies"] or 1 in ait["optuna_studies"]

    def test_has_predictions_false(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert ait["_has_predictions"] is False
        assert ait["predictions"] == []

    def test_has_trial_history_false(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert ait["_has_trial_history"] is False

    def test_feature_library_present(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        keys = {f["key"] for f in ait["FEATURE_LIBRARY"]}
        assert "rsi_14" in keys
        assert "hurst_wavelet" in keys

    def test_experiments_list(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert len(ait["experiments"]) == 1
        assert ait["experiments"][0]["id"] == "TEST_EXP_001"

    def test_git_sha_truncated(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert ait["experiment"]["git_sha"] == "abc1234"

    def test_none_fields_for_missing_metrics(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        r = ait["experiment"]["results"]
        assert r["sortino_ratio"] is None
        assert r["capital_utilization"] is None
        assert r["raroc"] is None

    def test_trade_return_pct_is_none(self, tmp_path):
        _minimal_metadata(tmp_path)
        ait = build_ait(tmp_path)
        assert ait["trades"][0]["return_pct"] is None


# ---------------------------------------------------------------------------
# write_wf_data
# ---------------------------------------------------------------------------

class TestWriteWfData:
    def test_output_starts_with_assignment(self, tmp_path):
        out = tmp_path / "wf_data.js"
        write_wf_data({"experiment": {}, "bars": []}, out)
        content = out.read_text()
        assert content.startswith("window.AIT=")

    def test_output_is_valid_json(self, tmp_path):
        out = tmp_path / "wf_data.js"
        payload = {"experiment": {"id": "X"}, "bars": [], "windows": []}
        write_wf_data(payload, out)
        content = out.read_text()
        json_str = content.removeprefix("window.AIT=").removesuffix(";\n")
        parsed = json.loads(json_str)
        assert parsed["experiment"]["id"] == "X"

    def test_output_ends_with_semicolon_newline(self, tmp_path):
        out = tmp_path / "wf_data.js"
        write_wf_data({}, out)
        assert out.read_text().endswith(";\n")

    def test_full_round_trip(self, tmp_path):
        report_dir = tmp_path / "report"
        report_dir.mkdir()
        _minimal_metadata(report_dir)
        out = tmp_path / "wf_data.js"
        _minimal_metadata(report_dir)
        ait = build_ait(report_dir)
        write_wf_data(ait, out)
        content = out.read_text()
        json_str = content.removeprefix("window.AIT=").removesuffix(";\n")
        parsed = json.loads(json_str)
        assert parsed["experiment"]["id"] == "TEST_EXP_001"
        assert len(parsed["windows"]) == 2
        assert len(parsed["trades"]) == 3

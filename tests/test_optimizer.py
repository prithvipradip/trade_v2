"""Tests for Feature 3: Optuna optimization module."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.optimization.objectives import OBJECTIVES
from ait.optimization.optimizer import StrategyOptimizer
from ait.optimization.param_spaces import IRON_CONDOR_SPACE, STRATEGY_SPACES, ML_SPACES
from ait.optimization.results import OptimizationResult
from ait.backtesting.result import BacktestResult
from ait.backtesting.walkforward import WalkForwardConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(pnl: float = 100.0, win_rate: float = 0.6, drawdown: float = 0.05) -> BacktestResult:
    trades = []
    n_wins = int(10 * win_rate)
    for _ in range(n_wins):
        trades.append({"pnl": pnl / 10, "strategy": "iron_condor"})
    for _ in range(10 - n_wins):
        trades.append({"pnl": -10.0, "strategy": "iron_condor"})
    initial = 10_000.0
    return BacktestResult(
        trades=trades,
        initial_capital=initial,
        final_capital=initial + pnl,
    )


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Minimal OHLCV DataFrame for testing."""
    rng = np.random.default_rng(42)
    closes = 400 + np.cumsum(rng.normal(0, 1.5, n))
    dates = pd.date_range(end=date.today(), periods=n, freq="B")
    return pd.DataFrame({
        "Open":   closes * 0.999,
        "High":   closes * 1.005,
        "Low":    closes * 0.995,
        "Close":  closes,
        "Volume": rng.integers(1_000_000, 50_000_000, n),
    }, index=dates)


# ---------------------------------------------------------------------------
# Param spaces
# ---------------------------------------------------------------------------

class TestParamSpaces:
    def test_iron_condor_space_has_required_keys(self):
        # The space only contains params that Backtester actually accepts.
        required = {"min_confidence", "stop_loss_pct", "profit_target_pct"}
        assert required.issubset(IRON_CONDOR_SPACE.keys())

    def test_all_strategy_spaces_have_min_confidence(self):
        for name, space in STRATEGY_SPACES.items():
            assert "min_confidence" in space, f"{name} missing min_confidence"

    def test_strategy_spaces_dict_covers_major_strategies(self):
        expected = {"iron_condor", "long_call", "bull_call_spread"}
        assert expected.issubset(STRATEGY_SPACES.keys())

    def test_ml_spaces_cover_xgboost_and_lightgbm(self):
        assert "xgboost" in ML_SPACES
        assert "lightgbm" in ML_SPACES

    def test_param_spec_has_valid_types(self):
        for space in list(STRATEGY_SPACES.values()) + list(ML_SPACES.values()):
            for name, spec in space.items():
                assert spec[0] in ("int", "float", "categorical"), (
                    f"Invalid type '{spec[0]}' in param '{name}'"
                )
                if spec[0] in ("int", "float"):
                    low, high = spec[1], spec[2]
                    assert low < high, f"{name}: low={low} must be < high={high}"


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

class TestObjectives:
    def test_sharpe_ratio_objective(self):
        result = _make_result(pnl=1000.0, win_rate=0.7)
        val = OBJECTIVES["sharpe_ratio"](result)
        assert isinstance(val, float)

    def test_composite_objective(self):
        result = _make_result(pnl=500.0, win_rate=0.6)
        val = OBJECTIVES["composite"](result)
        assert isinstance(val, float)

    def test_profit_factor_capped_at_10(self):
        # All wins → profit_factor = inf → should be capped at 10
        all_wins = BacktestResult(
            trades=[{"pnl": 100.0}] * 5,
            initial_capital=1000.0,
            final_capital=1500.0,
        )
        val = OBJECTIVES["profit_factor"](all_wins)
        assert val == pytest.approx(10.0)

    def test_win_rate_objective(self):
        # _make_result uses int(10 * win_rate) wins → 0.60 gives exactly 6/10 = 0.60
        result = _make_result(win_rate=0.60)
        val = OBJECTIVES["win_rate"](result)
        assert val == pytest.approx(0.60)

    def test_objectives_dict_has_all_expected_keys(self):
        expected = {"sharpe_ratio", "composite", "profit_factor", "win_rate"}
        assert expected == set(OBJECTIVES.keys())


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------

class TestOptimizationResult:
    def _make_study(self, tmp_path: Path):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")

        def obj(trial):
            x = trial.suggest_float("x", -5, 5)
            y = trial.suggest_int("y", 1, 10)
            return -(x ** 2) + y

        study.optimize(obj, n_trials=5)
        return study

    def test_best_params_accessible(self, tmp_path: Path):
        study = self._make_study(tmp_path)
        result = OptimizationResult(study)
        assert "x" in result.best_params
        assert "y" in result.best_params

    def test_best_value_is_float(self, tmp_path: Path):
        study = self._make_study(tmp_path)
        result = OptimizationResult(study)
        assert isinstance(result.best_value, float)

    def test_summary_returns_string(self, tmp_path: Path):
        study = self._make_study(tmp_path)
        result = OptimizationResult(study)
        s = result.summary(top_n=3)
        assert "OPTUNA OPTIMIZATION RESULTS" in s
        assert "Best value" in s

    def test_save_creates_json(self, tmp_path: Path):
        study = self._make_study(tmp_path)
        result = OptimizationResult(study)
        out = str(tmp_path / "result.json")
        result.save(out)
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert "best_params" in data
        assert "best_value" in data
        assert "n_trials" in data

    def test_apply_to_config_writes_overrides(self, tmp_path: Path):
        """apply_to_config should write recognised params into real config sections."""
        import optuna
        import yaml

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")

        def obj(trial):
            # Use real param names so _PARAM_MAP picks them up
            trial.suggest_float("iron_condor__min_confidence", 0.55, 0.80)
            trial.suggest_float("iron_condor__stop_loss_pct", 0.30, 0.70)
            trial.suggest_float("iron_condor__trailing_stop_pct", 0.15, 0.40)
            return 1.0

        study.optimize(obj, n_trials=2)
        result = OptimizationResult(study)
        cfg_path = str(tmp_path / "config.yaml")
        result.apply_to_config(cfg_path)

        data = yaml.safe_load(Path(cfg_path).read_text())
        # Recognised params are written into real config sections, not strategy_overrides
        assert "risk" in data
        assert "min_confidence" in data["risk"]
        assert "exit" in data
        assert "initial_stop_loss_pct" in data["exit"] or "trailing_stop_pct" in data["exit"]


# ---------------------------------------------------------------------------
# WalkForwardConfig — new fields
# ---------------------------------------------------------------------------

class TestWalkForwardConfigOptimizer:
    def test_default_optimize_per_window_false(self):
        cfg = WalkForwardConfig()
        assert cfg.optimize_per_window is False

    def test_default_optimize_n_trials_is_50(self):
        cfg = WalkForwardConfig()
        assert cfg.optimize_n_trials == 50

    def test_custom_n_trials(self):
        cfg = WalkForwardConfig(optimize_per_window=True, optimize_n_trials=100)
        assert cfg.optimize_n_trials == 100

    def test_optimize_n_trials_is_user_configurable(self):
        for n in [10, 25, 200]:
            cfg = WalkForwardConfig(optimize_n_trials=n)
            assert cfg.optimize_n_trials == n


# ---------------------------------------------------------------------------
# StrategyOptimizer (fast integration test with injected data)
# ---------------------------------------------------------------------------

class TestStrategyOptimizer:
    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError, match="Unknown objective"):
            StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], objective="invalid")

    def test_run_produces_result(self):
        """End-to-end run with injected data and 3 trials — verifies the pipeline."""
        opt = StrategyOptimizer(
            symbols=["SPY"],
            strategies=["iron_condor"],
            n_trials=3,
            objective="sharpe_ratio",
        )
        result = opt.run(data={"SPY": _make_ohlcv(300)})
        assert result is not None
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_value, float)
        assert len(result.study.trials) == 3

    def test_composite_objective_run(self):
        opt = StrategyOptimizer(
            symbols=["SPY"],
            strategies=["iron_condor"],
            n_trials=2,
            objective="composite",
        )
        result = opt.run(data={"SPY": _make_ohlcv(300)})
        assert result is not None

    def test_study_name_uses_strategies_and_objective(self):
        opt = StrategyOptimizer(
            symbols=["SPY"],
            strategies=["iron_condor", "long_call"],
            objective="win_rate",
        )
        assert "iron_condor" in opt._study_name
        assert "win_rate" in opt._study_name

    def test_custom_study_name_preserved(self):
        opt = StrategyOptimizer(
            symbols=["SPY"],
            strategies=["iron_condor"],
            study_name="my_custom_study",
        )
        assert opt._study_name == "my_custom_study"

    def test_resumable_study_with_storage(self, tmp_path: Path):
        """A study with file-based storage can be resumed (trial count accumulates)."""
        storage = f"sqlite:///{tmp_path}/optuna.db"
        study_name = "resume_test"

        for expected_trials in [2, 4]:
            opt = StrategyOptimizer(
                symbols=["SPY"],
                strategies=["iron_condor"],
                n_trials=2,
                objective="sharpe_ratio",
                study_name=study_name,
                storage=storage,
            )
            result = opt.run(data={"SPY": _make_ohlcv(300)})

        # After two runs of 2 trials, total should be 4
        assert len(result.study.trials) == 4

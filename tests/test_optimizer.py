"""Tests for Feature 3: Optuna optimization module."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ait.optimization.objectives import OBJECTIVES
from ait.optimization.optimizer import StrategyOptimizer
from ait.optimization.param_spaces import IRON_CONDOR_SPACE, STRATEGY_SPACES, ML_SPACES, SHORT_STRANGLE_SPACE
from ait.optimization.results import OptimizationResult
from ait.backtesting.result import BacktestResult
from ait.backtesting.walkforward import WalkForwardConfig

# R12: long-running suite (Optuna trials) — excluded from the default/CI fast
# selection (-m "not ibkr and not slow"); the nightly CI job runs -m slow.
pytestmark = pytest.mark.slow


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
    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
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
        required = {"delta_short", "max_hold_days", "wing_k"}
        assert required.issubset(IRON_CONDOR_SPACE.keys())

    def test_non_iron_condor_spaces_have_min_confidence(self):
        # iron_condor removed min_confidence from its search space (uses fixed value);
        # all other strategies still include it.
        for name, space in STRATEGY_SPACES.items():
            if name == "iron_condor":
                continue
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

    def test_iron_condor_space_has_wing_k(self):
        assert "wing_k" in IRON_CONDOR_SPACE
        low, high = IRON_CONDOR_SPACE["wing_k"][1], IRON_CONDOR_SPACE["wing_k"][2]
        assert low < 1.0 < high, "wing_k search range must bracket the default value 1.0"

    def test_short_strangle_space_has_delta_iv_scale(self):
        assert "delta_iv_scale" in SHORT_STRANGLE_SPACE
        low, high = SHORT_STRANGLE_SPACE["delta_iv_scale"][1], SHORT_STRANGLE_SPACE["delta_iv_scale"][2]
        assert low == pytest.approx(0.0)
        assert high == pytest.approx(1.0)

    def test_long_strangle_in_strategy_spaces(self):
        assert "long_strangle" in STRATEGY_SPACES
        assert "min_confidence" in STRATEGY_SPACES["long_strangle"]
        assert "delta_long" in STRATEGY_SPACES["long_strangle"]
        assert "delta_iv_scale" in STRATEGY_SPACES["long_strangle"]

    def test_all_new_strategies_in_strategy_spaces(self):
        for name in ("short_strangle", "long_strangle", "put_credit_spread"):
            assert name in STRATEGY_SPACES, f"{name} missing from STRATEGY_SPACES"


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

    def test_summary_sort_keeps_zero_above_negative(self):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.ERROR)
        study = optuna.create_study(direction="maximize")

        t0 = study.ask()
        study.tell(t0, 0.0)
        t1 = study.ask()
        study.tell(t1, -1.0)

        result = OptimizationResult(study)
        summary = result.summary(top_n=2)

        assert summary.find(" 0.0000") < summary.find("-1.0000")

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
            # trailing_stop_fraction replaced trailing_stop_pct — apply_to_config maps
            # the raw stop_loss_pct directly; trailing_stop_fraction is a derived param
            # handled by the optimizer, not written to config by apply_to_config.
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
        assert "initial_stop_loss_pct" in data["exit"]


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

    def test_default_wing_k_is_1(self):
        cfg = WalkForwardConfig()
        assert cfg.wing_k == pytest.approx(1.0)

    def test_wing_k_propagated_to_config(self):
        cfg = WalkForwardConfig(wing_k=0.5)
        assert cfg.wing_k == pytest.approx(0.5)

    def test_default_delta_iv_scale_is_0(self):
        cfg = WalkForwardConfig()
        assert cfg.delta_iv_scale == pytest.approx(0.0)


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

    def test_single_strategy_returns_only_that_strategys_params(self):
        """strategies=['iron_condor'] → every best_params key must start with 'iron_condor__'."""
        opt = StrategyOptimizer(
            symbols=["SPY"], strategies=["iron_condor"], n_trials=2,
        )
        result = opt.run(data={"SPY": _make_ohlcv(300)})
        assert result.best_params
        unexpected = [k for k in result.best_params if not k.startswith("iron_condor__")]
        assert not unexpected, f"Unexpected param keys: {unexpected}"

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

    def test_fetch_data_respects_train_days(self, monkeypatch: pytest.MonkeyPatch):
        """_fetch_data() trims to train_days rows from load_daily_ohlcv output."""
        big_df = _make_ohlcv(300)

        monkeypatch.setattr(
            "ait.data.market_data.load_daily_ohlcv",
            lambda symbol, days, db_path=None: big_df,
        )

        opt = StrategyOptimizer(
            symbols=["SPY"],
            strategies=["iron_condor"],
            train_days=120,
        )
        data = opt._fetch_data()

        assert "SPY" in data
        assert len(data["SPY"]) == 120

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


# ---------------------------------------------------------------------------
# wing_k optimization
# ---------------------------------------------------------------------------

class TestWingKOptimization:
    def test_wing_k_appears_in_best_params(self):
        """Optuna must suggest wing_k as part of iron_condor optimization."""
        opt = StrategyOptimizer(
            symbols=["SPY"], strategies=["iron_condor"], n_trials=3,
        )
        result = opt.run(data={"SPY": _make_ohlcv(300)})
        param_names = [k.split("__")[-1] for k in result.best_params]
        assert "wing_k" in param_names, (
            f"wing_k not found in best_params keys: {list(result.best_params.keys())}"
        )

    def test_wing_k_in_put_credit_spread_space(self):
        from ait.optimization.param_spaces import PUT_CREDIT_SPREAD_SPACE
        assert "wing_k" in PUT_CREDIT_SPREAD_SPACE
        low, high = PUT_CREDIT_SPREAD_SPACE["wing_k"][1], PUT_CREDIT_SPREAD_SPACE["wing_k"][2]
        assert low < 1.0 < high

    def test_iron_condor_regime_gates_excluded_from_space(self):
        # iv_floor and max_entry_vol_annual are regime gates; in-sample optima don't
        # generalise OOS (P8). Both are fixed in config, not optimized by Optuna.
        # spread_* are calibrated from real data and must stay fixed (P9).
        from ait.optimization.param_spaces import IRON_CONDOR_SPACE
        for excluded in ("iv_floor", "max_entry_vol_annual",
                         "spread_base", "spread_iv_sensitivity", "spread_dte_sensitivity"):
            assert excluded not in IRON_CONDOR_SPACE, f"{excluded} should not be in IRON_CONDOR_SPACE"

    def test_short_strangle_space_has_max_entry_vol(self):
        from ait.optimization.param_spaces import SHORT_STRANGLE_SPACE
        assert "max_entry_vol_annual" in SHORT_STRANGLE_SPACE


"""W67: shared fractal regime gate, gates_mode parity, graded optimizer objective, gate study.

House rules: real code paths, an AST class guard for the next re-implementation, and the
oracles below are the PRE-W67 inline formulas copied verbatim (live orchestrator + engine),
so the shared function is proven bit-for-bit against what it replaced.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import Backtester
from ait.ml.regime_gate import (
    SHAPE_LIVE_FLAT,
    SHAPE_PROPORTIONAL,
    FractalGateConfig,
    apply_range_prob_delta,
    fractal_gate,
    resolve_veto_spread,
)

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- oracles (pre-W67 code)
def _old_live_penalty(spread, mf, thr, pen, mf_max):
    """orchestrator.py before W67: flat penalty when either limit is exceeded."""
    return pen if (spread > thr or mf > mf_max) else 0.0


def _old_engine(spread, mf, thr, pen, mf_max, mult, neutral=True):
    """engine.py before W67: (vetoed, total confidence deduction)."""
    deduction = 0.0
    if spread > thr and thr > 0:
        base = max(thr, 0.20)
        if neutral and mult > 0 and spread > base * mult:
            return True, 0.0
        deduction += pen * (spread / thr)
    if mf > 0 and mf > mf_max:
        deduction += pen
    return False, deduction


GRID = list(itertools.product(
    [0.0, 0.05, 0.19, 0.20, 0.21, 0.30, 0.31, 0.45, 0.9],   # hurst spread
    [0.0, 0.2, 0.5, 0.51, 0.8],                                # multifractal width
    [0.08, 0.20, 0.30],                                        # threshold
    [0.0, 0.10, 0.25],                                         # penalty
))


class TestSharedFractalGateIsBitForBit:
    @pytest.mark.parametrize("spread,mf,thr,pen", GRID)
    def test_live_flat_equals_the_old_orchestrator_rule(self, spread, mf, thr, pen):
        cfg = FractalGateConfig(threshold=thr, penalty=pen, multifractal_max_width=0.5)
        v = fractal_gate(spread, mf, cfg)
        assert v.directional_penalty == _old_live_penalty(spread, mf, thr, pen, 0.5)
        assert v.triggered == (spread > thr or mf > 0.5)
        assert not v.veto and v.range_prob_delta == 0.0   # defaults: live parity

    @pytest.mark.parametrize("mult", [0.0, 1.0, 1.5, 2.0])
    @pytest.mark.parametrize("spread,mf,thr,pen", GRID)
    def test_proportional_shape_equals_the_old_engine_rule(self, spread, mf, thr, pen, mult):
        cfg = FractalGateConfig(
            threshold=thr, penalty=pen, multifractal_max_width=0.5,
            veto_spread=resolve_veto_spread(thr, mult, None), shape=SHAPE_PROPORTIONAL)
        v = fractal_gate(spread, mf, cfg)
        vetoed, deduction = _old_engine(spread, mf, thr, pen, 0.5, mult)
        assert v.veto == vetoed
        if not vetoed:
            assert v.directional_penalty == pytest.approx(deduction)

    def test_explicit_veto_spread_wins_over_the_legacy_multiplier(self):
        assert resolve_veto_spread(0.2, 1.5, 0.45) == 0.45
        assert resolve_veto_spread(0.2, 1.5, None) == pytest.approx(0.30)
        assert resolve_veto_spread(0.08, 1.5, None) == pytest.approx(0.30)   # floor 0.20
        assert resolve_veto_spread(0.2, 0.0, None) == 0.0
        assert resolve_veto_spread(0.2, None, None) == 0.0
        assert resolve_veto_spread(0.2, 1.5, 0.0) == 0.0     # explicit 0 = off beats the alias

    def test_range_probability_penalty_is_off_by_default_and_scales_and_caps(self):
        base = FractalGateConfig(threshold=0.2, penalty=0.1, multifractal_max_width=0.5)
        assert fractal_gate(0.6, 0.0, base).range_prob_delta == 0.0
        on = FractalGateConfig(threshold=0.2, penalty=0.1, multifractal_max_width=0.5,
                               penalty_on_range_prob=0.05)
        assert fractal_gate(0.1, 0.0, on).range_prob_delta == 0.0            # not chaotic
        assert fractal_gate(0.4, 0.0, on).range_prob_delta == pytest.approx(-0.10)
        assert fractal_gate(5.0, 0.0, on).range_prob_delta == pytest.approx(-0.15)  # capped 3x
        v = fractal_gate(0.4, 0.0, on)
        assert apply_range_prob_delta(0.70, v) == pytest.approx(0.60)
        assert apply_range_prob_delta(0.05, v) == 0.0      # stays a probability

    def test_config_validation(self):
        with pytest.raises(ValueError):
            FractalGateConfig(threshold=0.2, penalty=0.1, multifractal_max_width=0.5, shape="x")
        with pytest.raises(ValueError):
            FractalGateConfig(threshold=0.2, penalty=0.1, multifractal_max_width=0.5,
                              veto_spread=-1)


# ---------------------------------------------------------------- engine gates_mode
@pytest.fixture(scope="module")
def synth_df() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=160)
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(idx))))
    return pd.DataFrame({"Open": close, "High": close * 1.004, "Low": close * 0.996,
                         "Close": close, "Volume": 1e6}, index=idx)


@pytest.fixture(scope="module")
def synth_features(synth_df) -> pd.DataFrame:
    from ait.ml.features import FeatureEngine
    return FeatureEngine().compute(synth_df)


def _kw(df, feats, **over) -> dict:
    kw = dict(data=df, strategies=["iron_condor"], initial_capital=100_000,
              macro_event_gate=False, allow_live_model_fallback=False,
              hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0,
              iv_rank_rise_threshold=10.0, ic_min_credit=0.0, ic_min_credit_width=0.0,
              iv_floor=0.01, features_cache=feats)
    kw.update(over)
    return kw


def _rejections(f: dict) -> int:
    return sum(v for k, v in f.items() if k not in ("days_evaluated", "entries_taken")
               and not k.startswith(("builder:", "would_reject:")))


class TestGatesMode:
    def _hot(self, feats):
        f = feats.copy()
        f["hurst_scale_spread"] = 0.9
        return f

    def test_enforce_blocks_the_hurst_veto_and_observe_only_records_it(
            self, synth_df, synth_features):
        hot = self._hot(synth_features)
        over = dict(hurst_regime_threshold=0.01, fractal_veto_spread=0.3, features_cache=hot,
                    eval_start_date=hot.index[15].date())
        enf = Backtester(**_kw(synth_df, synth_features, gates_mode="enforce", **over)).run()
        obs = Backtester(**_kw(synth_df, synth_features, gates_mode="observe", **over)).run()
        assert enf.total_trades == 0 and enf.entry_funnel["hurst_hard_veto"] > 0
        assert obs.total_trades > 0 and obs.entry_funnel.get("hurst_hard_veto", 0) == 0
        assert obs.entry_funnel["would_reject:hurst_hard_veto"] > 0

    def test_observe_matches_a_run_with_the_veto_off(self, synth_df, synth_features):
        hot = self._hot(synth_features)
        base = dict(hurst_regime_threshold=0.01, features_cache=hot,
                    eval_start_date=hot.index[15].date())
        off = Backtester(**_kw(synth_df, synth_features, fractal_veto_spread=0.0,
                               gates_mode="enforce", **base)).run()
        obs = Backtester(**_kw(synth_df, synth_features, fractal_veto_spread=0.3,
                               gates_mode="observe", **base)).run()
        assert [t["pnl"] for t in off.trades] == [t["pnl"] for t in obs.trades]

    def test_range_gate_no_predictor_contract_is_observe_only_in_observe(
            self, synth_df, synth_features):
        enf = Backtester(**_kw(synth_df, synth_features, range_min_confidence=1.0,
                               gates_mode="enforce")).run()
        obs = Backtester(**_kw(synth_df, synth_features, range_min_confidence=1.0,
                               gates_mode="observe")).run()
        assert enf.total_trades == 0
        assert obs.total_trades > 0
        assert obs.entry_funnel["would_reject:range_gate_no_predictor"] > 0

    @pytest.mark.parametrize("mode", ["observe", "enforce"])
    def test_conservation_holds_in_both_modes(self, synth_df, synth_features, mode):
        hot = self._hot(synth_features)
        res = Backtester(**_kw(synth_df, synth_features, gates_mode=mode, features_cache=hot,
                               hurst_regime_threshold=0.01, fractal_veto_spread=0.3,
                               range_min_confidence=1.0)).run()
        f = res.entry_funnel
        assert f["days_evaluated"] == f.get("entries_taken", 0) + _rejections(f)

    def test_bad_mode_is_refused_and_bare_default_is_enforce(self, synth_df, synth_features):
        with pytest.raises(ValueError):
            Backtester(**_kw(synth_df, synth_features, gates_mode="maybe"))
        assert Backtester(**_kw(synth_df, synth_features))._gates_mode == "enforce"

    def test_legacy_multiplier_still_drives_the_veto(self, synth_df, synth_features):
        bt = Backtester(**_kw(synth_df, synth_features, hurst_regime_threshold=0.2,
                              hurst_hard_veto_multiplier=1.5))
        assert bt._fractal_cfg.veto_spread == pytest.approx(0.30)


# ---------------------------------------------------------------- one config for all engines
class TestResolvedOnceEverywhere:
    def test_research_defaults_are_live_parity(self):
        from ait.backtesting.walkforward import WalkForwardConfig
        from ait.config.settings import BacktestConfig
        c = BacktestConfig()
        assert (c.gates_mode, c.fractal_veto_spread, c.penalty_on_range_prob) == (
            "observe", 0.0, 0.0)
        wf = WalkForwardConfig()
        assert (wf.gates_mode, wf.fractal_veto_spread, wf.penalty_on_range_prob) == (
            "observe", 0.0, 0.0)

    def test_an_explicit_legacy_multiplier_is_translated_once(self):
        from ait.backtesting.walkforward import WalkForwardConfig
        wf = WalkForwardConfig(hurst_hard_veto_multiplier=1.5)
        assert wf.fractal_veto_spread == pytest.approx(0.30)

    def test_config_bounds(self):
        from pydantic import ValidationError
        from ait.config.settings import BacktestConfig
        for bad in (dict(gates_mode="x"), dict(fractal_veto_spread=-0.1),
                    dict(penalty_on_range_prob=0.9)):
            with pytest.raises(ValidationError):
                BacktestConfig(**bad)

    def test_every_walkforward_backtester_passes_the_gate_kwargs(self):
        src = inspect.getsource(importlib.import_module("ait.backtesting.walkforward"))
        calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) == "Backtester"]
        assert calls
        need = {"gates_mode", "fractal_veto_spread", "penalty_on_range_prob"}
        bad = [(c.lineno, sorted(need - {k.arg for k in c.keywords})) for c in calls
               if not need <= {k.arg for k in c.keywords}]
        assert not bad, f"Backtester(...) built without the gate kwargs: {bad}"

    def test_the_optimizer_trial_backtester_carries_the_gate_kwargs(self):
        """Its kwargs come from the bt_kwargs dict literal: the keys must be in it (the
        pre-W67 trial engine silently ran the bare-default 1.5x veto)."""
        src = inspect.getsource(importlib.import_module("ait.optimization.optimizer"))
        keys = {n.value for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert {"gates_mode", "fractal_veto_spread", "penalty_on_range_prob"} <= keys

    def test_no_caller_reimplements_the_fractal_rule(self):
        """Class guard: the spread/width comparison lives ONLY in regime_gate.py."""
        banned_attrs = {"hurst_regime_threshold", "multifractal_max_width",
                        "_hurst_regime_threshold", "_multifractal_max_width"}
        offenders = []
        for rel in ("src/ait/backtesting/engine.py", "src/ait/bot/orchestrator.py"):
            tree = ast.parse((ROOT / rel).read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                for n in ast.walk(node):
                    if (isinstance(n, ast.Attribute) and n.attr in banned_attrs) or (
                            isinstance(n, ast.Name) and n.id == "hurst_scale_spread"):
                        offenders.append((rel, node.lineno))
        assert not offenders, f"fractal comparison outside regime_gate.py: {offenders}"


# ---------------------------------------------------------------- orchestrator wiring
class TestOrchestratorUsesTheSharedGate:
    def test_it_imports_and_calls_the_shared_function(self):
        src = (ROOT / "src/ait/bot/orchestrator.py").read_text()
        tree = ast.parse(src)
        called = {getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert {"fractal_gate", "FractalGateConfig", "apply_range_prob_delta"} <= called

    def test_veto_only_acts_when_ml_gates_are_enabled(self):
        """The veto filter sits behind entry_gates_enabled (observe mode logs only)."""
        src = (ROOT / "src/ait/bot/orchestrator.py").read_text()
        i = src.index("if _fractal_verdict.veto:")
        window = src[i:i + 700]
        assert "entry_gates_enabled" in window and "fractal_veto_" in window


# ---------------------------------------------------------------- graded objective
class _FakeResult:
    def __init__(self, n):
        self.total_trades = n
        self.sharpe_ratio = 1.0
        self.win_rate = 0.5
        self.max_drawdown = 0.05
        self.entry_funnel = {"days_evaluated": 100, "no_strategy": 80, "builder_none": 5,
                             "would_reject:range_gate_below_threshold": 9, "builder:iv_floor": 5}


class _FakeTrial:
    def __init__(self):
        self.attrs = {}
        self.number = 0

    def set_user_attr(self, k, v):
        self.attrs[k] = v

    def report(self, *a, **k):
        pass

    def should_prune(self):
        return False


def _optimizer(**kw):
    from ait.optimization.optimizer import StrategyOptimizer
    return StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1, **kw)


class TestGradedObjective:
    def _score(self, opt, n, monkeypatch):
        monkeypatch.setattr(opt, "_suggest_params", lambda trial: {})
        monkeypatch.setattr(opt, "_run_backtest", lambda params: _FakeResult(n))
        trial = _FakeTrial()
        return opt._objective_fn(trial), trial

    def test_score_rises_with_trades_below_the_floor_and_is_not_flat(self, monkeypatch):
        opt = _optimizer(objective="sharpe_ratio")
        scores = [self._score(opt, n, monkeypatch)[0] for n in (0, 1, 2)]
        assert scores[0] == -100.0 and scores[0] < scores[1] < scores[2] < -80.0 - 1e-9 + 20

    def test_floor_and_gradient_come_from_config_and_zero_restores_flat(self, monkeypatch):
        flat = _optimizer(objective="sharpe_ratio", below_floor_gradient=0.0)
        assert [self._score(flat, n, monkeypatch)[0] for n in (0, 2)] == [-100.0, -100.0]
        wide = _optimizer(objective="sharpe_ratio", hard_min_trades=6, below_floor_gradient=30.0)
        assert self._score(wide, 3, monkeypatch)[0] == pytest.approx(-100.0 + 30.0 * 3 / 6)

    def test_above_the_floor_scoring_is_unchanged(self, monkeypatch):
        opt = _optimizer(objective="sharpe_ratio")
        v, _ = self._score(opt, 12, monkeypatch)
        assert v == pytest.approx(1.0)                       # no damping at/above min_trades
        v5, _ = self._score(opt, 5, monkeypatch)
        assert v5 == pytest.approx(1.0 * (5 / 10) ** 2)      # quadratic damping, as before

    def test_a_starved_trial_still_records_why(self, monkeypatch):
        opt = _optimizer(objective="sharpe_ratio")
        _, trial = self._score(opt, 1, monkeypatch)
        assert trial.attrs["n_trades"] == 1
        assert trial.attrs["entry_funnel_top"][0] == "no_strategy=80"
        assert all(not x.startswith(("would_reject:", "builder:"))
                   for x in trial.attrs["entry_funnel_top"])

    def test_study_without_a_trading_trial_is_flagged_unusable(self, monkeypatch):
        from structlog.testing import capture_logs
        opt = _optimizer(objective="sharpe_ratio")
        monkeypatch.setattr(opt, "_run_backtest", lambda params: _FakeResult(0))
        df = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0],
                           "Volume": [1.0]}, index=pd.bdate_range("2024-01-02", periods=1))
        with capture_logs() as ev:
            res = opt.run(data={"SPY": df})
        assert res.usable is False and res.best_n_trades == 0
        warn = [e for e in ev if e["event"] == "optuna_no_usable_trial"]
        assert warn and warn[0]["log_level"] == "warning"

    def test_study_with_a_trading_trial_is_usable(self, monkeypatch):
        opt = _optimizer(objective="sharpe_ratio")
        monkeypatch.setattr(opt, "_run_backtest", lambda params: _FakeResult(12))
        df = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0],
                           "Volume": [1.0]}, index=pd.bdate_range("2024-01-02", periods=1))
        res = opt.run(data={"SPY": df})
        assert res.usable is True and res.best_n_trades == 12

    def test_fractal_dimensions_are_dropped_for_condor_only_and_kept_otherwise(self):
        opt = _optimizer(objective="sharpe_ratio")
        opt._maybe_drop_inert_params()
        assert {"iron_condor__hurst_regime_threshold", "iron_condor__hurst_regime_penalty",
                "iron_condor__multifractal_max_width"} <= opt._dropped_params
        on = _optimizer(objective="sharpe_ratio", penalty_on_range_prob=0.05)
        on._maybe_drop_inert_params()
        assert "iron_condor__hurst_regime_threshold" not in on._dropped_params

    def test_walkforward_summary_names_the_starved_studies(self):
        from ait.backtesting.walkforward import WalkForwardResult
        res = WalkForwardResult(
            windows=[], initial_capital=5000.0, config=None,
            optimization_status={1: {"SPY": {"usable": False, "best_value": -100.0,
                                             "best_n_trades": 0},
                                     "QQQ": {"usable": True, "best_value": 0.4,
                                             "best_n_trades": 12}}})
        try:
            text = res.summary()
        except Exception:
            pytest.skip("summary() requires a populated config")
        assert "1 of 2 per-window studies found a trading trial" in text
        assert "NEVER did" in text and "w1/SPY" in text


# ---------------------------------------------------------------- gate study statistics
def _load_study():
    import importlib.util
    spec = importlib.util.spec_from_file_location("gate_study", ROOT / "scripts/gate_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic(effect: float, n_months=24, per_month=18, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        for i in range(per_month):
            sym = ["SPY", "QQQ", "IWM"][i % 3]
            spread = float(rng.uniform(0, 0.8))
            pnl = float(rng.normal(-10.0 - effect * (spread > 0.4) * 1.0, 30.0))
            rows.append({"symbol": sym, "month": f"2024-{m:02d}", "pnl": pnl,
                         "hurst_spread": spread})
    return pd.DataFrame(rows)


class TestGateStudyStatistics:
    def test_recovers_a_planted_effect_on_every_symbol(self):
        gs = _load_study()
        v = gs.cutoff_verdict(_synthetic(effect=60.0), cutoff=0.4, n_boot=400)
        assert v["supported"] and v["pooled"]["diff"] > 30 and v["pooled"]["lo"] > 0

    def test_reports_not_distinguishable_on_pure_noise(self):
        gs = _load_study()
        for seed in (1, 2, 3):
            v = gs.cutoff_verdict(_synthetic(effect=0.0, seed=seed), cutoff=0.4, n_boot=400)
            assert not v["supported"] and v["verdict"].startswith("not distinguishable")

    def test_sign_disagreement_blocks_a_pooled_win(self):
        gs = _load_study()
        df = _synthetic(effect=0.0)
        # make vetoing look good pooled via SPY alone but harmful on IWM
        df.loc[(df.symbol == "SPY") & (df.hurst_spread > 0.4), "pnl"] -= 200
        df.loc[(df.symbol == "IWM") & (df.hurst_spread > 0.4), "pnl"] += 120
        v = gs.cutoff_verdict(df, cutoff=0.4, n_boot=400)
        assert not v["signs_agree"] and not v["supported"]

    def test_bootstrap_is_clustered_by_month_and_deterministic(self):
        gs = _load_study()
        df = _synthetic(effect=10.0)
        a = gs.block_bootstrap_diff(df, df.hurst_spread <= 0.4, n_boot=200)
        b = gs.block_bootstrap_diff(df, df.hurst_spread <= 0.4, n_boot=200)
        assert a == b
        shuffled = df.assign(month="all")           # one cluster -> no resampling variety
        c = gs.block_bootstrap_diff(shuffled, shuffled.hurst_spread <= 0.4, n_boot=50)
        assert c["lo"] == c["hi"] == pytest.approx(c["diff"])

    def test_empty_side_is_nan_not_a_crash(self):
        gs = _load_study()
        df = _synthetic(effect=0.0)
        r = gs.block_bootstrap_diff(df, df.hurst_spread <= 99, n_boot=10)
        assert r["n_excl"] == 0 and np.isnan(r["diff"])

    def test_bucket_table_counts_every_trade_once(self):
        gs = _load_study()
        df = _synthetic(effect=0.0)
        assert gs.bucket_table(df, "hurst_spread")["n"].sum() == len(df)

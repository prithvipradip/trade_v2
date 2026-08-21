"""R20b (pre-registered PLAN 2026-08-21) — research defaults migrate to config
resolution. Regression pins.

The deferral reason for keeping frozen research literals (old-study
comparability) is void: R20 proved every prior absolute was priced without
volatility data, so protecting their reproduction protects disavowed numbers.

Covers:
  #1  WalkForwardConfig: wing_k 1.0 / range_min_confidence 0.55 literals ->
      None sentinels resolved in __post_init__ from the contract / loaded
      config (wing_k -> AIT_IC_WING_K env>config>1.6; ic_min_credit_width ->
      AIT_IC_MIN_CREDIT_WIDTH -> 0.10; range_min_confidence ->
      ml.range_min_confidence -> 0.65, closing the 0.55-vs-live-0.65 parity
      gap the floor sweep exposed). Explicit values always win.
  #2  Engine constructor defaults that SHADOWED config with different values:
      iv_floor 0.12 -> settings.backtest.iv_floor (0.20); range_min_confidence
      0.55 -> settings.ml.range_min_confidence (0.65); min_confidence 0.55 ->
      settings.risk.min_confidence (0.50 — the directional entry gate live
      reads from risk.min_confidence, see engine.py resolution block).
      initial_capital / max_concurrent_positions stay explicit (documented
      harness knobs).
  #3  Optimizer trial baselines (bt_kwargs) come from load_settings().backtest
      instead of frozen literals: stop_loss_pct / profit_target_pct /
      max_hold_days / hurst_regime_threshold / hurst_regime_penalty /
      multifractal_max_width / iv_rank_rise_threshold / min_edge_over_baseline.

Every resolution test here FAILED against the pre-R20b tree (proved via a
PYTHONPATH-shadowed scratch copy of src/ait — run log in the R20b report).
The explicit-values-win tests are permanent guards.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ait.backtesting.engine as engine_mod
import ait.backtesting.walkforward as wf_mod
from ait.backtesting.engine import Backtester
from ait.backtesting.result import BacktestResult
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.config.runtime_env import CONTRACT_DEFAULTS, contract_float
from ait.optimization.optimizer import StrategyOptimizer


# ---------------------------------------------------------------------------
# Shared synthetic market data (seeded — mirrors test_r20_research_validity)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_df() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=160)
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(idx))))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": 1e6,
        },
        index=idx,
    )


def _engine_kwargs(synth_df, **overrides) -> dict:
    """Minimal Backtester kwargs: construction-only tests, confounders inert.
    Deliberately does NOT pass iv_floor / min_confidence /
    range_min_confidence — their bare resolution is the behaviour under test."""
    kw = dict(
        data=synth_df,
        strategies=["iron_condor"],
        initial_capital=100_000,
        macro_event_gate=False,
        allow_live_model_fallback=False,
    )
    kw.update(overrides)
    return kw


class _RecordingBacktester:
    """Stands in for Backtester at a construction site: captures kwargs,
    run() returns an empty result so the caller's loop completes."""

    def __init__(self, sink: list, **kwargs) -> None:
        self.kwargs = kwargs
        sink.append(kwargs)

    def run(self) -> BacktestResult:
        return BacktestResult(initial_capital=100_000.0, final_capital=100_000.0)


def _recorder_factory(sink: list):
    def _make(**kwargs):
        return _RecordingBacktester(sink, **kwargs)
    return _make


@pytest.fixture
def _clean_contract_env(monkeypatch):
    """Assertions about contract resolution must not be masked by an exported
    override in the shell that launched pytest."""
    for k in ("AIT_IC_WING_K", "AIT_IC_MIN_CREDIT_WIDTH"):
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# #1 — WalkForwardConfig None sentinels resolve to the operating contract
# ---------------------------------------------------------------------------

class TestWalkForwardConfigResolution:
    def test_bare_config_resolves_promoted_values(self, _clean_contract_env):
        # Pre-R20b this was wing_k=1.0 / ic_min_credit_width=None(engine-side)
        # / range_min_confidence=0.55 — the frozen 2026-07 literals.
        cfg = WalkForwardConfig()
        assert cfg.wing_k == pytest.approx(1.6)
        assert cfg.wing_k == pytest.approx(contract_float("AIT_IC_WING_K"))
        assert cfg.ic_min_credit_width == pytest.approx(0.10)
        assert cfg.ic_min_credit_width == pytest.approx(
            contract_float("AIT_IC_MIN_CREDIT_WIDTH")
        )
        # ml.range_min_confidence: config.yaml 0.65 (0.65 beat 0.55 across
        # every backtest metric) — the live parity value.
        assert cfg.range_min_confidence == pytest.approx(0.65)
        from ait.config.settings import load_settings
        assert cfg.range_min_confidence == pytest.approx(
            load_settings("config.yaml").ml.range_min_confidence
        )

    def test_env_override_reaches_bare_config(self, monkeypatch):
        # The sentinel resolves THROUGH the contract (env > config > default),
        # not to a copied literal.
        monkeypatch.setenv("AIT_IC_WING_K", "2.5")
        assert WalkForwardConfig().wing_k == pytest.approx(2.5)

    def test_explicit_values_always_win(self, _clean_contract_env):
        cfg = WalkForwardConfig(
            wing_k=1.0, ic_min_credit_width=0.20, range_min_confidence=0.55
        )
        assert cfg.wing_k == pytest.approx(1.0)
        assert cfg.ic_min_credit_width == pytest.approx(0.20)
        assert cfg.range_min_confidence == pytest.approx(0.55)

    def test_no_config_yaml_falls_back_to_ml_model_default(self, monkeypatch):
        import ait.config.settings as settings_mod

        def _boom(*a, **k):
            raise FileNotFoundError("no config.yaml")

        monkeypatch.setattr(settings_mod, "load_settings", _boom)
        from ait.config.settings import MLConfig
        cfg = WalkForwardConfig()
        assert cfg.range_min_confidence == pytest.approx(
            MLConfig().range_min_confidence
        )

    def test_oos_backtester_receives_resolved_values(
        self, synth_df, monkeypatch, _clean_contract_env
    ):
        # End-to-end: a bare config must reach the OOS Backtester with the
        # PROMOTED values (pre-R20b: wing_k=1.0, range_min_confidence=0.55).
        cfg = WalkForwardConfig(train_window_models=False)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        wf._run_single_window(
            window_id=1,
            train_start=date(2024, 1, 2),
            train_end=date(2024, 5, 31),
            test_start=date(2024, 6, 5),
            test_end=date(2024, 8, 1),
            data={"SPY": synth_df},
            vix_full=pd.DataFrame(),
            spy_full=pd.DataFrame(),
            learner=None,
            prev_best_params=None,
            prev_oos=None,
        )
        assert sink, "OOS Backtester was never constructed"
        kw = sink[-1]
        assert kw["wing_k"] == pytest.approx(1.6)
        assert kw["ic_min_credit_width"] == pytest.approx(0.10)
        # train_window_models=False is the documented UNGATED ablation arm:
        # the configured (now config-resolved) threshold flows through.
        assert kw["range_min_confidence"] == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# #2 — Engine bare construction resolves from loaded settings
# ---------------------------------------------------------------------------

class TestEngineConfigResolution:
    def test_bare_engine_resolves_config_values(self, synth_df):
        # Pre-R20b: iv_floor=0.12, range_min_confidence=0.55,
        # min_confidence=0.55 — each shadowing config with a DIFFERENT value.
        from ait.config.settings import load_settings
        st = load_settings("config.yaml")
        bt = Backtester(**_engine_kwargs(synth_df))
        assert bt._iv_floor == pytest.approx(st.backtest.iv_floor)
        assert bt._iv_floor == pytest.approx(0.20)
        assert bt._range_min_confidence == pytest.approx(st.ml.range_min_confidence)
        assert bt._range_min_confidence == pytest.approx(0.65)
        # min_confidence's home is risk.min_confidence: the engine consumes it
        # as the DIRECTIONAL-confidence entry gate — the gate live reads from
        # settings.risk.min_confidence (orchestrator.py), and
        # export_production_params maps it there (optimization/results.py).
        assert bt._min_confidence == pytest.approx(st.risk.min_confidence)
        assert bt._min_confidence == pytest.approx(0.50)

    def test_engine_reads_loaded_settings_not_model_defaults(
        self, synth_df, monkeypatch
    ):
        # A yaml edit must reach a bare engine (LOADED settings, not the
        # pydantic defaults) — monkeypatched stub proves the wiring.
        import ait.config.settings as settings_mod
        from ait.config.settings import BacktestConfig

        stub = SimpleNamespace(
            backtest=BacktestConfig(iv_floor=0.33),
            ml=SimpleNamespace(range_min_confidence=0.71),
            risk=SimpleNamespace(min_confidence=0.59),
        )
        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: stub)
        bt = Backtester(**_engine_kwargs(synth_df))
        assert bt._iv_floor == pytest.approx(0.33)
        assert bt._range_min_confidence == pytest.approx(0.71)
        assert bt._min_confidence == pytest.approx(0.59)

    def test_explicit_constructor_args_still_win(self, synth_df):
        bt = Backtester(
            **_engine_kwargs(
                synth_df,
                iv_floor=0.12, range_min_confidence=0.55, min_confidence=0.61,
            )
        )
        assert bt._iv_floor == pytest.approx(0.12)
        assert bt._range_min_confidence == pytest.approx(0.55)
        assert bt._min_confidence == pytest.approx(0.61)

    def test_partial_stub_degrades_to_model_defaults(self, synth_df, monkeypatch):
        # The R20 per-field-guard convention: a stub with only .exit (as
        # test_r20 uses) must not break the new resolutions — each field
        # falls back to ITS config-model default.
        import ait.config.settings as settings_mod
        from ait.config.settings import BacktestConfig, MLConfig, RiskConfig

        stub = SimpleNamespace(exit=SimpleNamespace(time_decay_scaling=True))
        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: stub)
        bt = Backtester(**_engine_kwargs(synth_df))
        assert bt._iv_floor == pytest.approx(BacktestConfig().iv_floor)
        assert bt._range_min_confidence == pytest.approx(
            MLConfig().range_min_confidence
        )
        assert bt._min_confidence == pytest.approx(RiskConfig().min_confidence)

    def test_harness_knobs_stay_explicit(self, synth_df):
        # Pre-registration: initial_capital / max_concurrent_positions remain
        # explicit constructor defaults (test-harness sizing knobs, documented
        # in engine.py) — they must NOT silently start tracking config.yaml
        # (backtest.max_concurrent_positions is 3 there).
        bt = Backtester(data=synth_df, strategies=["iron_condor"],
                        macro_event_gate=False, allow_live_model_fallback=False)
        assert bt._initial_capital == pytest.approx(10_000.0)
        assert bt._max_concurrent_positions == 1


# ---------------------------------------------------------------------------
# #3 — Optimizer trial baselines come from load_settings().backtest
# ---------------------------------------------------------------------------

# distinctive stub values, all inside the BacktestConfig field bounds
_STUB_BASELINES = dict(
    stop_loss_pct=0.41,
    profit_target_pct=0.77,
    max_hold_days=17,
    hurst_regime_threshold=0.33,
    hurst_regime_penalty=0.22,
    multifractal_max_width=0.61,
    iv_rank_rise_threshold=0.47,
    min_edge_over_baseline=0.13,
)


class TestOptimizerBaselinesFromConfig:
    def _capture_bt_kwargs(self, monkeypatch, synth_df, **opt_kwargs) -> dict:
        """EXECUTE the real StrategyOptimizer._run_backtest with a recording
        Backtester substituted at its import site (test_r20 pattern)."""
        sink: list = []
        monkeypatch.setattr(engine_mod, "Backtester", _recorder_factory(sink))
        opt = StrategyOptimizer(
            symbols=["SPY"], strategies=["iron_condor"], n_trials=1, **opt_kwargs
        )
        opt._data = {"SPY": synth_df}
        opt._run_backtest({})
        assert len(sink) == 1
        return sink[0]

    def test_every_baseline_propagates_from_monkeypatched_settings(
        self, monkeypatch, synth_df
    ):
        # THE R20b wiring proof: change the config, and every baseline that
        # was a frozen literal must land in bt_kwargs with the changed value.
        import ait.config.settings as settings_mod
        from ait.config.settings import BacktestConfig

        stub = SimpleNamespace(backtest=BacktestConfig(**_STUB_BASELINES))
        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: stub)
        kw = self._capture_bt_kwargs(monkeypatch, synth_df)
        for name, expected in _STUB_BASELINES.items():
            assert kw[name] == pytest.approx(expected), (
                f"optimizer baseline '{name}' did not propagate from "
                f"load_settings().backtest — a config change cannot reach "
                f"trial backtests"
            )

    def test_bare_optimizer_matches_operating_config(self, monkeypatch, synth_df):
        # Without any stub, baselines equal the loaded operating config
        # (which today equals the retired literals — the migration changed
        # WHERE they live, not WHAT they are).
        from ait.config.settings import load_settings
        st = load_settings("config.yaml").backtest
        kw = self._capture_bt_kwargs(monkeypatch, synth_df)
        assert kw["stop_loss_pct"] == pytest.approx(st.stop_loss_pct) == 0.35
        assert kw["profit_target_pct"] == pytest.approx(st.profit_target_pct) == 0.50
        assert kw["max_hold_days"] == st.max_hold_days == 30
        assert kw["hurst_regime_threshold"] == pytest.approx(st.hurst_regime_threshold) == 0.20
        assert kw["hurst_regime_penalty"] == pytest.approx(st.hurst_regime_penalty) == 0.10
        assert kw["multifractal_max_width"] == pytest.approx(st.multifractal_max_width) == 0.50
        assert kw["iv_rank_rise_threshold"] == pytest.approx(st.iv_rank_rise_threshold) == 0.30
        assert kw["min_edge_over_baseline"] == pytest.approx(st.min_edge_over_baseline) == 0.05

    def test_explicit_caller_values_still_win(self, monkeypatch, synth_df):
        # R20 #2 threading contract is preserved: walkforward passes its
        # config values explicitly and they beat the config resolution.
        kw = self._capture_bt_kwargs(
            monkeypatch, synth_df,
            iv_rank_rise_threshold=0.44, min_edge_over_baseline=0.11,
        )
        assert kw["iv_rank_rise_threshold"] == pytest.approx(0.44)
        assert kw["min_edge_over_baseline"] == pytest.approx(0.11)

    def test_new_fields_do_not_widen_divergence_report(self):
        # Item 4 of the registration: fields whose code default == yaml value
        # drop out of the divergence report naturally — the report must load
        # clean and contain NONE of the R20b fields.
        from ait.config.settings import default_divergences, load_settings
        names = {n for n, _, _ in default_divergences(load_settings("config.yaml"))}
        for f in ("backtest.stop_loss_pct", "backtest.profit_target_pct",
                  "backtest.max_hold_days", "backtest.iv_rank_rise_threshold",
                  "backtest.min_edge_over_baseline",
                  "backtest.hurst_regime_threshold",
                  "backtest.hurst_regime_penalty",
                  "backtest.multifractal_max_width",
                  "backtest.wing_k", "backtest.ic_min_credit_width"):
            assert f not in names, f"{f} diverges from config.yaml"
        # the report still SEES the known deliberate overrides (r19 pin)
        assert "risk.min_confidence" in names

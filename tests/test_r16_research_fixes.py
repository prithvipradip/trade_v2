"""R16 research-integrity fixes — regression pins.

Covers the ten R16 findings fixed in:
  - src/ait/backtesting/engine.py
  - src/ait/backtesting/walkforward.py
  - src/ait/ml/ensemble.py
  - src/ait/data/economic_calendar.py

  #1  Look-ahead leak: engine falling back to the LIVE ensemble.pkl for
      walk-forward OOS windows (allow_live_model_fallback fence).
  #2  Live ensemble.pkl clobber: DirectionPredictor model_dir fence +
      persist_artifacts=False for window training.
  #3  Range-gate contract: predictor None + range_min_confidence >= 1.0 must
      BLOCK entries (engine) and the summary must surface training status.
  #4  train_window_models=False must disable range training too.
  #5  walkforward internal fetch tz normalization.
  #6  AIT_CREDIT_LOSS_LIMIT default parity (0 = disabled, matching live).
  #7  Engine blackout window from loaded settings, not RiskConfig() default.
  #8  AIT_IC_WING_K env resolution in the engine.
  #9  jade_lizard / call_credit_spread in the neutral gate dispatch.
  #10 Economic calendar 2027-H1 extension + early staleness alarm.
"""

from __future__ import annotations

import asyncio
import inspect
import pickle
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import (
    NEUTRAL_CREDIT_GATED,
    Backtester,
)
from ait.backtesting.walkforward import (
    WalkForwardBacktester,
    WalkForwardConfig,
    WalkForwardResult,
    WindowResult,
)
from ait.config.settings import MLConfig


# ---------------------------------------------------------------------------
# Shared synthetic market data (seeded — trade counts are deterministic)
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


@pytest.fixture(scope="module")
def synth_features(synth_df) -> pd.DataFrame:
    from ait.ml.features import FeatureEngine

    feats = FeatureEngine().compute(synth_df)
    assert not feats.empty
    return feats


def _engine_kwargs(synth_df, synth_features, **overrides) -> dict:
    """Backtester kwargs that isolate the gate under test: macro calendar off,
    fractal/iv-rank confounders disabled, credit floors zeroed."""
    kw = dict(
        data=synth_df,
        strategies=["iron_condor"],
        initial_capital=100_000,
        macro_event_gate=False,
        allow_live_model_fallback=False,
        hurst_hard_veto_multiplier=0.0,
        hurst_regime_threshold=0.0,
        iv_rank_rise_threshold=10.0,
        ic_min_credit=0.0,
        ic_min_credit_width=0.0,
        iv_floor=0.01,
        features_cache=synth_features,
    )
    kw.update(overrides)
    return kw


# ---------------------------------------------------------------------------
# #1 — Look-ahead fence
# ---------------------------------------------------------------------------

class TestLookAheadFence:
    def _write_fake_live_artifact(self) -> None:
        """Drop a loadable ensemble.pkl in the (conftest-fenced) MODEL_DIR."""
        import ait.ml.ensemble as ens

        ens.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(ens.MODEL_DIR / "ensemble.pkl", "wb") as f:
            pickle.dump(
                {"models": {}, "scaler": None, "feature_names": [], "version": "v-fake"},
                f,
            )

    def test_fallback_disabled_never_loads_live_artifact(self, synth_df, synth_features):
        self._write_fake_live_artifact()
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._predictor is None, (
            "allow_live_model_fallback=False must leave predictor=None even "
            "when a live ensemble.pkl exists (walk-forward OOS look-ahead fence)"
        )

    def test_fallback_enabled_loads_artifact(self, synth_df, synth_features):
        self._write_fake_live_artifact()
        bt = Backtester(
            **_engine_kwargs(synth_df, synth_features, allow_live_model_fallback=True)
        )
        assert bt._predictor is not None, (
            "standalone runs (fallback=True) should still load the artifact"
        )

    def test_explicit_predictor_always_wins(self, synth_df, synth_features):
        sentinel = object()
        bt = Backtester(
            **_engine_kwargs(synth_df, synth_features, predictor=sentinel)
        )
        assert bt._predictor is sentinel

    def test_walkforward_wires_fallback_off_for_oos_and_shadow(self):
        src_oos = inspect.getsource(WalkForwardBacktester._run_single_window)
        src_meta = inspect.getsource(WalkForwardBacktester._train_window_meta_labeler)
        assert "allow_live_model_fallback=False" in src_oos
        assert "allow_live_model_fallback=False" in src_meta


# ---------------------------------------------------------------------------
# #2 — DirectionPredictor artifact fence
# ---------------------------------------------------------------------------

class _DummyModel:
    def fit(self, X, y):
        return self


class TestDirectionPredictorFence:
    def test_module_model_dir_is_repo_anchored_absolute(self):
        import ait.ml.ensemble as ens

        src = inspect.getsource(ens)
        assert 'Path(__file__).resolve().parents[3] / "models"' in src, (
            "MODEL_DIR must be repo-anchored, never CWD-relative"
        )
        # parents[3] of src/ait/ml/ensemble.py is the repo root (trade_v2)
        from pathlib import Path

        expected = Path(ens.__file__).resolve().parents[3] / "models"
        assert expected.is_absolute()
        assert expected.parts[-1] == "models"
        assert (expected.parent / "src" / "ait" / "ml").is_dir()

    def test_model_dir_param_fences_saves(self, tmp_path):
        from ait.ml.ensemble import DirectionPredictor

        research_dir = tmp_path / "research_fence"
        pred = DirectionPredictor(MLConfig(), model_dir=research_dir)
        assert pred.model_dir == research_dir
        pred._model_version = "v-test"
        pred._save_models()
        assert (research_dir / "ensemble.pkl").exists()
        # The default (conftest-fenced) dir must NOT receive the artifact
        assert not (tmp_path / "ensemble.pkl").exists()

    def _train_with_dummies(self, monkeypatch, persist: bool):
        from ait.ml.ensemble import DirectionPredictor

        def fake_xgb(self, X, y, tscv):
            self._models["xgboost"] = _DummyModel()
            return 0.6

        def fake_lgbm(self, X, y, tscv):
            self._models["lightgbm"] = _DummyModel()
            return 0.6

        monkeypatch.setattr(DirectionPredictor, "_train_xgboost", fake_xgb)
        monkeypatch.setattr(DirectionPredictor, "_train_lightgbm", fake_lgbm)

        idx = pd.bdate_range("2022-01-03", periods=400)
        rng = np.random.default_rng(3)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(idx))))
        df = pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99,
             "Close": close, "Volume": 1e6},
            index=idx,
        )
        pred = DirectionPredictor(MLConfig(), persist_artifacts=persist)
        calls: list[int] = []
        pred._save_models = lambda: calls.append(1)  # type: ignore[method-assign]
        accs = pred.train(df)
        return accs, calls

    def test_persist_artifacts_false_never_saves(self, monkeypatch):
        accs, calls = self._train_with_dummies(monkeypatch, persist=False)
        assert accs, "training should still succeed"
        assert calls == [], "persist_artifacts=False must NEVER write artifacts"

    def test_persist_artifacts_default_saves(self, monkeypatch):
        accs, calls = self._train_with_dummies(monkeypatch, persist=True)
        assert accs
        assert calls == [1], "default (live) training keeps persisting"

    def test_walkforward_window_training_never_persists(self):
        src = inspect.getsource(WalkForwardBacktester._train_window_model)
        assert "persist_artifacts=False" in src, (
            "window models must never be persisted (live ensemble.pkl clobber)"
        )


# ---------------------------------------------------------------------------
# #3 — Range-gate contract
# ---------------------------------------------------------------------------

class TestRangeGateContract:
    def test_engine_blocks_entries_when_contract_active(self, synth_df, synth_features):
        # Control: predictor None + reachable threshold -> gate skipped,
        # entries flow (the engine's designed ungated fallback).
        open_res = Backtester(
            **_engine_kwargs(synth_df, synth_features, range_min_confidence=0.55)
        ).run()
        assert open_res.total_trades > 0, "control run must generate entries"

        # Contract: predictor None + threshold >= 1.0 (walkforward's
        # "range model failed -> block") -> ZERO entries.
        blocked_res = Backtester(
            **_engine_kwargs(synth_df, synth_features, range_min_confidence=1.0)
        ).run()
        assert blocked_res.total_trades == 0, (
            "range_min_confidence=1.0 with no range predictor must BLOCK all "
            "neutral-credit entries (the contract every 2026-08 study broke)"
        )

    def test_threshold_resolver(self):
        f = WalkForwardBacktester._resolve_oos_range_min_conf
        assert f(None, "training_returned_no_accuracy", 0.55) == 1.0
        assert f(None, "exception: boom", 0.55) == 1.0
        assert f(None, "skipped", 0.55) == 1.0
        # Ablation arm: disabled-by-config runs UNGATED at the configured threshold
        assert f(None, "disabled_by_config", 0.55) == 0.55
        assert f(object(), "ok", 0.55) == 0.55

    def test_summary_surfaces_range_training_status(self):
        res = WalkForwardResult(
            range_training_status={
                1: {"SPY": "training_returned_no_accuracy"},
                2: {"SPY": "ok"},
            }
        )
        text = res.summary()
        assert "RANGE MODEL TRAINING" in text
        assert "Trained ok:        1/2" in text
        assert "training_returned_no_accuracy" in text
        assert "BLOCKED" in text

    def test_summary_screams_when_no_window_ever_trained(self):
        res = WalkForwardResult(
            range_training_status={i: {"SPY": "training_returned_no_accuracy"} for i in range(1, 33)}
        )
        assert "says NOTHING about the range gate" in res.summary()

    def test_window_result_carries_status(self):
        wr = WindowResult(
            window_id=1,
            train_start=date(2024, 1, 1),
            train_end=date(2024, 6, 1),
            test_start=date(2024, 6, 10),
            test_end=date(2024, 9, 1),
            backtest_result=None,  # type: ignore[arg-type]
            range_model_status={"SPY": "ok"},
        )
        assert wr.range_model_status == {"SPY": "ok"}

    def test_run_single_window_uses_resolver(self):
        src = inspect.getsource(WalkForwardBacktester._run_single_window)
        assert "self._resolve_oos_range_min_conf(" in src


# ---------------------------------------------------------------------------
# #4 — train_window_models=False disables range training too
# ---------------------------------------------------------------------------

class TestTrainWindowModelsFlag:
    def test_pretrain_skipped_when_disabled(self, synth_df, monkeypatch):
        cfg = WalkForwardConfig(train_window_models=False)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)

        def _explode(*a, **k):  # pragma: no cover - failure path
            raise AssertionError("range training ran despite train_window_models=False")

        monkeypatch.setattr(
            WalkForwardBacktester, "_train_window_range_model_inprocess", _explode
        )
        windows = [(date(2024, 1, 1), date(2024, 6, 1), date(2024, 6, 10), date(2024, 9, 1))]
        out = wf._pretrain_range_models(windows, {"SPY": synth_df}, pd.DataFrame())
        assert out == [{"SPY": (None, "disabled_by_config", 0.05)}]

    def test_run_single_window_gates_range_training(self):
        src = inspect.getsource(WalkForwardBacktester._run_single_window)
        assert "if not self._config.train_window_models:" in src
        assert '"disabled_by_config"' in src


# ---------------------------------------------------------------------------
# #5 — internal fetch tz normalization
# ---------------------------------------------------------------------------

def _aware_df(rows: int = 150) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=rows, tz="America/New_York")
    close = np.linspace(100.0, 110.0, rows)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
        index=idx,
    )


class TestWalkforwardTz:
    def test_fetch_data_normalizes_tz(self, monkeypatch):
        import ait.data.market_data as md

        monkeypatch.setattr(
            md, "load_daily_ohlcv", lambda s, days=0, db_path=None: _aware_df()
        )
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"])
        data = asyncio.run(wf._fetch_data())
        assert "SPY" in data
        assert data["SPY"].index.tz is None, (
            "_fetch_data must return tz-naive frames — the aware/naive mix "
            "crashed run(data=None) on every window"
        )

    def test_run_normalizes_provided_data_defensively(self, monkeypatch):
        import ait.data.market_data as md

        monkeypatch.setattr(
            md, "load_daily_ohlcv", lambda s, days=0, db_path=None: pd.DataFrame()
        )
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"])
        captured: dict = {}

        def fake_windows(d):
            captured.update(d)
            return []

        wf._generate_windows = fake_windows  # type: ignore[method-assign]
        aware = _aware_df()
        asyncio.run(wf.run(data={"SPY": aware}))
        assert captured["SPY"].index.tz is None
        # caller's frame must be untouched (shallow copy)
        assert aware.index.tz is not None


# ---------------------------------------------------------------------------
# #6 — AIT_CREDIT_LOSS_LIMIT parity (default 0 = disabled)
# ---------------------------------------------------------------------------

class TestCreditLossParity:
    def test_default_matches_live_disabled(self, synth_df, synth_features, monkeypatch):
        monkeypatch.delenv("AIT_CREDIT_LOSS_LIMIT", raising=False)
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._credit_loss_limit_mult == 0.0, (
            "engine default must match live portfolio.py (0 = flat stop "
            "DISABLED; R12-B1 evidence + R16 #6 parity)"
        )

    def test_env_override_still_works(self, synth_df, synth_features, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_LOSS_LIMIT", "1.5")
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._credit_loss_limit_mult == 1.5

    def test_zero_mult_never_fires_flat_stop(self, synth_df, synth_features, monkeypatch):
        monkeypatch.delenv("AIT_CREDIT_LOSS_LIMIT", raising=False)
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        pos = {"expiry_date": "2099-01-01", "entry_date": "2024-06-01", "strategy": "iron_condor"}
        # deep loss, far expiry: with the stop disabled nothing may fire
        assert bt._check_exit_credit(dict(pos), -2.0, date(2024, 6, 3)) is None

    def test_positive_mult_still_fires(self, synth_df, synth_features):
        bt = Backtester(
            **_engine_kwargs(synth_df, synth_features, credit_loss_limit_mult=1.25)
        )
        pos = {"expiry_date": "2099-01-01", "entry_date": "2024-06-01", "strategy": "iron_condor"}
        out = bt._check_exit_credit(dict(pos), -1.3, date(2024, 6, 3))
        assert out is not None and out["exit_reason"] == "credit_loss_limit"


# ---------------------------------------------------------------------------
# #7 — blackout window from loaded settings
# ---------------------------------------------------------------------------

class TestBlackoutWiring:
    def test_constructor_param_wins(self, synth_df, synth_features):
        bt = Backtester(
            **_engine_kwargs(synth_df, synth_features, pre_event_blackout_days=3)
        )
        assert bt._pre_event_blackout_days == 3

    def test_default_resolves_from_loaded_settings(
        self, synth_df, synth_features, monkeypatch
    ):
        import ait.config.settings as settings_mod

        class _Risk:
            pre_event_blackout_days = 5

        class _Settings:
            risk = _Risk()

        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: _Settings())
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._pre_event_blackout_days == 5, (
            "engine must read the LOADED settings (live parity), not the "
            "RiskConfig() default"
        )

    def test_falls_back_to_riskconfig_default(
        self, synth_df, synth_features, monkeypatch
    ):
        import ait.config.settings as settings_mod
        from ait.config.settings import RiskConfig

        def _boom(*a, **k):
            raise FileNotFoundError("no config.yaml")

        monkeypatch.setattr(settings_mod, "load_settings", _boom)
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._pre_event_blackout_days == RiskConfig().pre_event_blackout_days

    def test_walkforward_passes_field_through(self):
        assert WalkForwardConfig().pre_event_blackout_days is None
        src = inspect.getsource(WalkForwardBacktester._run_single_window)
        assert "pre_event_blackout_days=" in src


# ---------------------------------------------------------------------------
# #8 — AIT_IC_WING_K env resolution
# ---------------------------------------------------------------------------

class TestWingKEnv:
    def test_env_resolution_when_unset(self, synth_df, synth_features, monkeypatch):
        monkeypatch.setenv("AIT_IC_WING_K", "1.6")
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._wing_k == 1.6, "engine must mirror live AIT_IC_WING_K resolution"

    def test_default_without_env(self, synth_df, synth_features, monkeypatch):
        # R19: this asserted 1.0 — the DIVERGENT reader default that was the
        # bug. The env contract (runtime_env.CONTRACT_DEFAULTS) has declared
        # 1.6 since the 2026-08-04 wide-wing promotion, so an engine built in
        # a process that never applied the contract used to research 1.0-wing
        # structures while live traded 1.6. The whole point of the single
        # authority is that the unset case now resolves the PROMOTED value.
        monkeypatch.delenv("AIT_IC_WING_K", raising=False)
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        from ait.config.runtime_env import CONTRACT_DEFAULTS
        assert bt._wing_k == float(CONTRACT_DEFAULTS["AIT_IC_WING_K"]) == 1.6

    def test_explicit_param_beats_env(self, synth_df, synth_features, monkeypatch):
        monkeypatch.setenv("AIT_IC_WING_K", "1.6")
        bt = Backtester(**_engine_kwargs(synth_df, synth_features, wing_k=0.8))
        assert bt._wing_k == 0.8


# ---------------------------------------------------------------------------
# #9 — neutral gate dispatch includes jade_lizard + call_credit_spread
# ---------------------------------------------------------------------------

class TestNeutralGateDispatch:
    def test_tuple_membership(self):
        assert "jade_lizard" in NEUTRAL_CREDIT_GATED
        assert "call_credit_spread" in NEUTRAL_CREDIT_GATED
        for s in ("iron_condor", "iron_butterfly", "wide_wing_condor",
                  "broken_wing_condor", "short_strangle"):
            assert s in NEUTRAL_CREDIT_GATED

    def test_all_six_dispatch_sites_use_shared_tuple(self):
        src = inspect.getsource(Backtester.run)
        assert src.count("NEUTRAL_CREDIT_GATED") >= 6, (
            "hurst veto, direction-gate bypass, regime veto, range gate, "
            "vol gate, and iv-rank-rise veto must all dispatch on the shared set"
        )
        # No stale hand-rolled tuple may remain in the entry loop
        assert '"broken_wing_condor", "short_strangle")' not in src

    def test_shadow_arms_now_trade(self, synth_df, synth_features):
        # Pre-fix these arms produced ZERO trades: NEUTRAL@0.4 from
        # _simple_direction could never clear min_confidence=0.55.
        for arm in ("jade_lizard", "call_credit_spread"):
            res = Backtester(
                **_engine_kwargs(
                    synth_df, synth_features,
                    strategies=[arm], range_min_confidence=0.55,
                )
            ).run()
            assert res.total_trades > 0, f"{arm} arm must generate trades post-fix"

    def test_blocked_contract_applies_to_new_arms_too(self, synth_df, synth_features):
        res = Backtester(
            **_engine_kwargs(
                synth_df, synth_features,
                strategies=["jade_lizard"], range_min_confidence=1.0,
            )
        ).run()
        assert res.total_trades == 0


# ---------------------------------------------------------------------------
# #10 — economic calendar 2027-H1 + early staleness alarm
# ---------------------------------------------------------------------------

@pytest.fixture
def _reset_calendar_alarm():
    from ait.data.economic_calendar import EconomicCalendar

    prev = EconomicCalendar._exhausted_warned
    EconomicCalendar._exhausted_warned = False
    yield
    EconomicCalendar._exhausted_warned = prev


class TestCalendar2027:
    def test_no_blind_window_after_2026_12_23(self, _reset_calendar_alarm):
        from ait.data.economic_calendar import EconomicCalendar

        cal = EconomicCalendar()
        d2e = cal.days_until_next_event(date(2026, 12, 24))
        assert d2e is not None, "guards went blind on 2026-12-24 pre-fix"
        assert d2e <= 30

    def test_hold_through_2027_events_is_unsafe(self, _reset_calendar_alarm):
        from ait.data.economic_calendar import EconomicCalendar

        cal = EconomicCalendar()
        assert cal.is_safe_to_hold_through_expiry(
            date(2026, 12, 28), date(2027, 2, 19)
        ) is False, "a late-Dec condor spans Jan-2027 NFP/CPI/FOMC"

    def test_2027_h1_tables_present(self):
        from ait.data.economic_calendar import (
            _CPI_2027,
            _FOMC_2027,
            _NFP_2027,
            _PCE_2027,
            EconomicCalendar,
        )

        cal = EconomicCalendar()
        ev27 = cal.get_events_between(date(2027, 1, 1), date(2027, 6, 30))
        assert len(ev27) >= 20
        # NFP: first Friday (Jan shifted past the holiday Friday Jan 1)
        assert date(2027, 1, 8) in _NFP_2027
        for d in _NFP_2027:
            assert d.weekday() == 4  # Friday
        for d in _PCE_2027:
            assert d.weekday() == 4  # last business Friday
        assert len(_CPI_2027) == 6 and all(10 <= d.day <= 13 for d in _CPI_2027)
        assert len(_FOMC_2027) == 4

    def test_staleness_alarm_fires_early(self, _reset_calendar_alarm):
        from ait.data.economic_calendar import EconomicCalendar

        cal = EconomicCalendar()
        # Plenty of runway today: no alarm
        cal.is_event_day(date(2026, 8, 7))
        assert EconomicCalendar._exhausted_warned is False
        # < 30 days of future events left: alarm fires BEFORE the blind window
        cal.is_event_day(cal._last_event_date - timedelta(days=10))
        assert EconomicCalendar._exhausted_warned is True

    def test_alarm_also_wired_into_d2e_and_hold_check(self, _reset_calendar_alarm):
        from ait.data.economic_calendar import EconomicCalendar

        cal = EconomicCalendar()
        cal.days_until_next_event(cal._last_event_date - timedelta(days=5))
        assert EconomicCalendar._exhausted_warned is True
        EconomicCalendar._exhausted_warned = False
        cal.is_safe_to_hold_through_expiry(
            cal._last_event_date - timedelta(days=5),
            cal._last_event_date + timedelta(days=40),
        )
        assert EconomicCalendar._exhausted_warned is True

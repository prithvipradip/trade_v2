"""R20 research-apparatus validity fixes — regression pins.

Covers the R19-verified findings fixed in:
  - src/ait/backtesting/engine.py
  - src/ait/backtesting/walkforward.py
  - src/ait/optimization/optimizer.py
  - src/ait/execution/exit_policy.py (new shared authority)

  #1  market_context never forwarded to _build_position: the VIX-proxy IV
      branch and the R16 per-symbol vol calibration were dead at entry-pricing
      time — every study priced entries off realized_vol*1.15 synthetic.
  #2  Optuna-searched iv_rank_rise_threshold / min_edge_over_baseline were
      dropped before Backtester construction — the objective was flat noise
      along those dimensions, yet the arbitrary "best" values were applied OOS.
  #3  train_window_models=False still trained AND applied the window
      MetaLabeler (R16 #4 gated direction + range only) — "ML-free" ablation
      arms could run WITH a trained XGB entry gate.
  #4  WalkForwardConfig intraday knobs (scan_interval_minutes,
      entry_window_start_et/end_et, limit_order_timeout_bars) were phantom —
      declared, never forwarded to Backtester.  Now config-backed
      (settings.backtest) end to end, which also resolves the
      09:30-vs-10:30 three-source entry-window fork (#5b).
  #5a Credit TP ladder / DTE<=5 close / macro-flatten windows were hand-copied
      from live portfolio.py — now imported from the shared
      ait.execution.exit_policy module, with a parity test that fails on any
      future live-side de-sync instead of silently diverging.

Every test here FAILED against the pre-R20 tree (run log in the R20 report).
"""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

import ait.backtesting.engine as engine_mod
import ait.backtesting.walkforward as wf_mod
from ait.backtesting.engine import Backtester
from ait.backtesting.result import BacktestResult
from ait.backtesting.walkforward import (
    WalkForwardBacktester,
    WalkForwardConfig,
    WalkForwardResult,
)
from ait.optimization.optimizer import StrategyOptimizer
from ait.optimization.param_spaces import STRATEGY_SPACES


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
    """Backtester kwargs that isolate the behaviour under test (mirrors the
    R16 helper): macro calendar off, fractal/iv-rank confounders disabled,
    credit floors zeroed."""
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


def _run_window(wf: WalkForwardBacktester, df: pd.DataFrame):
    """Drive the REAL _run_single_window on a small synthetic window."""
    return wf._run_single_window(
        window_id=1,
        train_start=date(2024, 1, 2),
        train_end=date(2024, 5, 31),
        test_start=date(2024, 6, 5),
        test_end=date(2024, 8, 1),
        data={"SPY": df},
        vix_full=pd.DataFrame(),
        spy_full=pd.DataFrame(),
        learner=None,
        prev_best_params=None,
        prev_oos=None,
    )


# ---------------------------------------------------------------------------
# #1 — market_context threaded to entry pricing
# ---------------------------------------------------------------------------

class TestMarketContextThreading:
    def test_vix_context_changes_entry_pricing(self, synth_df, synth_features):
        # A/B: identical runs except for market_context. Pre-fix both priced
        # every entry off the priority-3 synthetic fallback (rv*1.15) because
        # run() never forwarded the context to _build_position.
        res_ctx = Backtester(
            **_engine_kwargs(
                synth_df, synth_features, symbol="QQQ",
                market_context={"vix": 30.0},
            )
        ).run()
        res_noctx = Backtester(
            **_engine_kwargs(synth_df, synth_features, symbol="QQQ")
        ).run()
        assert res_ctx.total_trades > 0 and res_noctx.total_trades > 0

        ivs_ctx = {t["entry_iv"] for t in res_ctx.trades}
        ivs_noctx = {t["entry_iv"] for t in res_noctx.trades}
        # VIX branch: 30.0/100 * QQQ multiplier 1.228 (R16 calibration)
        assert all(iv == pytest.approx(0.3684, abs=1e-4) for iv in ivs_ctx), (
            "with vix context every entry must be priced off the VIX-proxy "
            "branch with the per-symbol multiplier applied"
        )
        assert ivs_ctx != ivs_noctx, (
            "supplying market_context must change priced entry IV — pre-R20 "
            "both runs used the synthetic realized_vol*1.15 fallback"
        )

    def test_per_symbol_multiplier_reaches_entry_pricing(self, synth_df, synth_features):
        # SPY anchors at 1.00 (SPY IV ~ VIX); QQQ carries the measured
        # VXN/VIX 1.228. Pre-fix neither multiplier ever reached a trade.
        res_spy = Backtester(
            **_engine_kwargs(
                synth_df, synth_features, symbol="SPY",
                market_context={"vix": 30.0},
            )
        ).run()
        assert res_spy.total_trades > 0
        assert all(
            t["entry_iv"] == pytest.approx(0.30, abs=1e-4) for t in res_spy.trades
        )

    # CENSUS-5 (2026-09-27): test_build_position_call_site_passes_context
    # deleted here -- regexed run()'s SOURCE TEXT for "market_context" inside
    # the _build_position(...) call instead of proving it actually reaches
    # pricing. Strictly subsumed by test_vix_context_changes_entry_pricing and
    # test_per_symbol_multiplier_reaches_entry_pricing above, both of which
    # run() the real engine with different market_context and show real
    # trades' entry_iv actually differs / lands on the calibrated value.

    def test_no_context_free_get_iv_call_sites(self):
        # The one other _get_iv call site (_select_strategy) was a dead,
        # context-free call — removed so no site can silently price without
        # the context run() received.
        src = inspect.getsource(Backtester._select_strategy)
        assert "_get_iv(" not in src, (
            "_select_strategy carried an unused, context-free _get_iv call"
        )


# ---------------------------------------------------------------------------
# #2 — every Optuna-searched param reaches the trial Backtester
# ---------------------------------------------------------------------------

class TestOptunaParamsReachEngine:
    def _capture_bt_kwargs(self, monkeypatch, synth_df, strategy: str,
                           params: dict, **opt_kwargs) -> dict:
        """EXECUTE the real StrategyOptimizer._run_backtest with a recording
        Backtester substituted at its import site."""
        sink: list = []
        monkeypatch.setattr(engine_mod, "Backtester", _recorder_factory(sink))
        opt = StrategyOptimizer(
            symbols=["SPY"], strategies=[strategy], n_trials=1, **opt_kwargs
        )
        opt._data = {"SPY": synth_df}
        opt._run_backtest(params)
        assert len(sink) == 1
        return sink[0]

    @pytest.mark.parametrize("strategy", sorted(STRATEGY_SPACES))
    def test_every_searched_param_lands(self, monkeypatch, synth_df, strategy):
        # Build one trial suggestion per searched param with sentinel values,
        # then assert each one arrives at the Backtester constructor. Pre-R20
        # iron_condor dropped iv_rank_rise_threshold and
        # min_edge_over_baseline — a search over parameters that cannot move
        # the objective is pure noise, later applied OOS.
        space = STRATEGY_SPACES[strategy]
        params = {}
        for name, spec in space.items():
            sentinel = 17 if spec[0] == "int" else 0.31415
            params[f"{strategy}__{name}"] = sentinel

        kw = self._capture_bt_kwargs(monkeypatch, synth_df, strategy, params)

        for name, spec in space.items():
            sentinel = 17 if spec[0] == "int" else 0.31415
            if name == "trailing_stop_fraction":
                # Derived param: lands as trailing_stop_pct relative to target
                assert kw["trailing_stop_pct"] == pytest.approx(
                    sentinel * kw["profit_target_pct"]
                )
                continue
            assert name in kw, (
                f"searched param '{name}' ({strategy}) dropped before the "
                f"trial Backtester — the search over it is noise"
            )
            assert kw[name] == pytest.approx(sentinel)

    def test_baselines_threaded_from_constructor(self, monkeypatch, synth_df):
        # When the trial does NOT search a param, the baseline must come from
        # the caller (walkforward config), not a literal.
        kw = self._capture_bt_kwargs(
            monkeypatch, synth_df, "iron_condor", params={},
            iv_rank_rise_threshold=0.44, min_edge_over_baseline=0.11,
        )
        assert kw["iv_rank_rise_threshold"] == pytest.approx(0.44)
        assert kw["min_edge_over_baseline"] == pytest.approx(0.11)

    def test_attributes_land_on_real_engine(self, synth_df, synth_features):
        bt = Backtester(
            **_engine_kwargs(
                synth_df, synth_features,
                iv_rank_rise_threshold=0.42, min_edge_over_baseline=0.09,
            )
        )
        assert bt._iv_rank_rise_threshold == pytest.approx(0.42)
        assert bt._min_edge_over_baseline == pytest.approx(0.09)

    def test_walkforward_threads_baselines_to_optimizer(self, synth_df, monkeypatch):
        """CENSUS-1 (2026-09-27): test_baselines_threaded_from_constructor
        above already proves StrategyOptimizer->Backtester wiring; this test's
        own job is the missing link -- that _optimize_window_params actually
        constructs StrategyOptimizer with the walk-forward config's baselines,
        not a StrategyOptimizer built some other way. Stubs StrategyOptimizer
        itself (imported locally inside the method) to record its own
        constructor kwargs, instead of checking the literal source text."""
        import ait.optimization.optimizer as optimizer_mod

        captured: dict = {}

        class _RecordingOptimizer:
            def __init__(self, **kwargs) -> None:
                captured.update(kwargs)

            def run(self, data, prior_params=None):
                return SimpleNamespace(best_params={}, best_value=0.0, study=None)

        monkeypatch.setattr(optimizer_mod, "StrategyOptimizer", _RecordingOptimizer)
        cfg = WalkForwardConfig(
            iv_rank_rise_threshold=0.44, min_edge_over_baseline=0.11)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        wf._optimize_window_params(synth_df, "SPY", 1, ["iron_condor"])

        assert captured, "StrategyOptimizer was never constructed"
        assert captured["iv_rank_rise_threshold"] == pytest.approx(0.44)
        assert captured["min_edge_over_baseline"] == pytest.approx(0.11)

    def test_max_hold_days_changes_iron_condor_trial(self, synth_df, synth_features):
        # End-to-end guard: changing max_hold_days must change IC exits.
        short_hold = Backtester(
            **_engine_kwargs(
                synth_df,
                synth_features,
                max_hold_days=1,
                symbol="SPY",
            )
        )
        long_hold = Backtester(
            **_engine_kwargs(
                synth_df,
                synth_features,
                max_hold_days=30,
                symbol="SPY",
            )
        )
        short_hold._touch_stop_enabled = False
        long_hold._touch_stop_enabled = False
        short_hold._credit_take_profit_pct = lambda *_a, **_k: 9_999.0
        long_hold._credit_take_profit_pct = lambda *_a, **_k: 9_999.0

        res_short = short_hold.run()
        res_long = long_hold.run()
        assert res_short.total_trades > 0 and res_long.total_trades > 0

        short_reasons = {t.get("exit_reason") for t in res_short.trades}
        long_reasons = {t.get("exit_reason") for t in res_long.trades}
        assert "max_hold_reached" in short_reasons
        assert "max_hold_reached" not in long_reasons


# ---------------------------------------------------------------------------
# #3 — train_window_models=False disables the MetaLabeler too
# ---------------------------------------------------------------------------

class TestMetaLabelerAblationGate:
    def test_disabled_flag_skips_meta_training(self, synth_df, monkeypatch):
        # Same proof pattern as R16 #4: training under a disabled flag explodes.
        cfg = WalkForwardConfig(train_window_models=False)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)

        def _explode(*a, **k):  # pragma: no cover - failure path
            raise AssertionError(
                "meta-labeler training ran despite train_window_models=False"
            )

        monkeypatch.setattr(
            WalkForwardBacktester, "_train_window_meta_labeler", _explode
        )
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)  # must not raise
        assert sink, "OOS Backtester was never constructed"
        assert sink[-1]["meta_labeler"] is None, (
            "disabled arm must run with NO meta-label gate"
        )

    def test_disabled_flag_blocks_meta_gating(self, synth_df, monkeypatch):
        # Even if a trained labeler existed, the disabled arm must not gate
        # with it: the OOS Backtester receives meta_labeler=None.
        cfg = WalkForwardConfig(train_window_models=False)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sentinel = object()
        monkeypatch.setattr(
            WalkForwardBacktester, "_train_window_meta_labeler",
            lambda *a, **k: sentinel,
        )
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink[-1]["meta_labeler"] is None

    def test_enabled_flag_still_trains_and_gates(self, synth_df, monkeypatch):
        # Control: default arm keeps the Gap Z1 behaviour.
        cfg = WalkForwardConfig(train_window_models=True)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sentinel = object()
        monkeypatch.setattr(
            WalkForwardBacktester, "_train_window_meta_labeler",
            lambda *a, **k: sentinel,
        )
        # Direction/range training must not run against the network/db in this
        # unit test — return the documented "no model" results.
        monkeypatch.setattr(
            WalkForwardBacktester, "_train_window_model", lambda *a, **k: None
        )
        monkeypatch.setattr(
            WalkForwardBacktester, "_train_window_range_model",
            lambda *a, **k: (None, "disabled_by_config", 0.05),
        )
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink[-1]["meta_labeler"] is sentinel

    def test_summary_surfaces_meta_status(self):
        # Mirrors range_model_status surfacing (R16 #3): a study whose meta
        # gate silently armed (or silently didn't) must say so in the summary.
        res = WalkForwardResult(
            meta_training_status={
                1: {"SPY": "disabled_by_config"},
                2: {"SPY": "ok"},
                3: {"SPY": "not_trained"},
            }
        )
        text = res.summary()
        assert "META-LABEL GATE" in text
        assert "disabled_by_config" in text
        assert "not_trained" in text

    def test_run_wires_meta_status_into_result(self, synth_df, monkeypatch):
        """CENSUS-1 (2026-09-27): the three tests above already prove
        _run_single_window's own meta-labeler gating; this test's own job is
        that the top-level run() orchestration actually SURFACES that
        per-window status on its returned WalkForwardResult -- driven end to
        end with real windows generated from synth_df, not a source-text
        check. train_window_models=False keeps this fast (no real ML)."""
        cfg = WalkForwardConfig(
            train_window_models=False, train_days=120, test_days=30,
            step_days=30, gap_days=5)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory([]))

        result = asyncio.run(wf.run(data={"SPY": synth_df}))

        assert result.meta_training_status, (
            "run() must populate meta_training_status from real windows"
        )
        assert all(
            "SPY" in per_symbol
            for per_symbol in result.meta_training_status.values()
        )
        assert all(
            status == "disabled_by_config"
            for per_symbol in result.meta_training_status.values()
            for status in per_symbol.values()
        )


# ---------------------------------------------------------------------------
# #4 / #5b — intraday knobs wired end-to-end, config-backed (ONE source)
# ---------------------------------------------------------------------------

class TestIntradayKnobsWired:
    def test_walkforward_forwards_all_four_knobs(self, synth_df, monkeypatch):
        cfg = WalkForwardConfig(
            train_window_models=False,
            scan_interval_minutes=7,
            entry_window_start_et="11:11",
            entry_window_end_et="14:44",
            limit_order_timeout_bars=9,
        )
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=cfg)
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        kw = sink[-1]
        assert kw.get("scan_interval_minutes") == 7
        assert kw.get("entry_window_start_et") == "11:11"
        assert kw.get("entry_window_end_et") == "14:44"
        assert kw.get("limit_order_timeout_bars") == 9

    def test_walkforward_defaults_defer_to_config(self):
        # None = "engine resolves from settings.backtest" — the dataclass no
        # longer re-declares its own '09:30' fork of the entry window.
        cfg = WalkForwardConfig()
        assert cfg.scan_interval_minutes is None
        assert cfg.entry_window_start_et is None
        assert cfg.entry_window_end_et is None
        assert cfg.limit_order_timeout_bars is None

    def test_engine_resolves_window_from_loaded_settings(
        self, synth_df, synth_features, monkeypatch
    ):
        import ait.config.settings as settings_mod
        from ait.config.settings import BacktestConfig

        stub = SimpleNamespace(
            backtest=BacktestConfig(
                entry_window_start_et="12:34",
                entry_window_end_et="14:56",
                scan_interval_minutes=15,
                limit_order_timeout_bars=7,
            )
        )
        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: stub)
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._entry_window_start_et == "12:34"
        assert bt._entry_window_end_et == "14:56"
        assert bt._scan_interval_minutes == 15
        assert bt._limit_order_timeout_bars == 7

    def test_engine_default_matches_backtest_config(self, synth_df, synth_features):
        # #5b: settings.backtest (config.yaml-backed) is THE authority. The
        # engine's old hardcoded '09:30' contradicted BacktestConfig's '10:30'
        # (the documented Fix 1/Gap H parity value that
        # scripts/export_production_params.py reports as "production parity").
        from ait.config.settings import BacktestConfig, load_settings

        try:
            expected = load_settings().backtest
        except Exception:
            expected = BacktestConfig()
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._entry_window_start_et == expected.entry_window_start_et
        assert bt._entry_window_end_et == expected.entry_window_end_et
        assert bt._scan_interval_minutes == expected.scan_interval_minutes
        assert bt._limit_order_timeout_bars == expected.limit_order_timeout_bars
        # And the parity value itself: 10:30, not the engine's old 09:30 fork.
        assert BacktestConfig().entry_window_start_et == "10:30"

    def test_explicit_knobs_still_win(self, synth_df, synth_features):
        bt = Backtester(
            **_engine_kwargs(
                synth_df, synth_features,
                entry_window_start_et="09:45", scan_interval_minutes=30,
            )
        )
        assert bt._entry_window_start_et == "09:45"
        assert bt._scan_interval_minutes == 30


# ---------------------------------------------------------------------------
# #5a — ONE exit-policy authority shared by engine and live portfolio
# ---------------------------------------------------------------------------

class TestSharedExitPolicy:
    def test_engine_ladder_delegates_to_shared_module(self):
        # CENSUS-5 (2026-09-27): the getsource/string-containment prelude that
        # used to sit here ("exit_policy" in Backtester._credit_take_profit_pct's
        # source) is deleted -- the loop below already proves delegation far
        # more strongly, by calling BOTH implementations at 5 DTEs and showing
        # they agree; a source-text match doesn't prove the two stay in sync.
        from ait.execution import exit_policy

        for dte, expect in ((25, 0.50), (15, 0.40), (8, 0.30), (3, 0.20), (None, 0.50)):
            assert Backtester._credit_take_profit_pct(dte) == pytest.approx(expect)
            assert exit_policy.credit_take_profit_pct(dte) == pytest.approx(expect)

    def test_policy_matches_live_portfolio_today(self):
        # TODO(R20): portfolio.py should IMPORT ait.execution.exit_policy
        # instead of carrying its own copy of the ladder/windows (it is under
        # concurrent edit and not owned by this change). Until it does, THIS
        # test is the de-sync tripwire: it EXECUTES the live
        # _get_take_profit_targets and compares it against the shared policy,
        # so a live-side tuning silently de-syncing research fails loudly here.
        from ait.execution import exit_policy
        from ait.execution.portfolio import PortfolioManager

        pm = PortfolioManager.__new__(PortfolioManager)
        pm._exit_config = SimpleNamespace(time_decay_scaling=True)
        for dte in (None, 40, 25, 20, 15, 11, 10, 8, 6, 5, 3, 0):
            assert pm._get_take_profit_targets(dte) == exit_policy.take_profit_targets(dte), (
                f"live portfolio TP ladder de-synced from exit_policy at dte={dte}"
            )
        # time_decay_scaling=False collapses to the flat default in BOTH
        pm._exit_config = SimpleNamespace(time_decay_scaling=False)
        assert pm._get_take_profit_targets(25) == exit_policy.take_profit_targets(
            25, time_decay_scaling=False
        ) == (1.0, 0.50)

    # --- shared live-side harness, adapted from test_r15_review_fixes.py's
    # TestPreEventBlackoutRelaxed._live_pm/_live_trade (kept local -- this
    # file's own convention -- rather than imported across test modules).

    def _live_pm(self, *, d2e):
        """PortfolioManager wired far enough to run _evaluate_position."""
        from ait.config.settings import ExitConfig
        from ait.execution.portfolio import PortfolioManager
        pm = PortfolioManager.__new__(PortfolioManager)
        pm._last_quote_ts = {}
        pm._frozen_alerted = set()
        pm._touch_confirm = {}
        pm._marks_missing_streak = {}
        pm._pdt_alerted = set()
        pm._notify_cb = None
        pm._earnings = None
        pm._exit_config = ExitConfig()
        pm._quality = MagicMock()
        pm._quality.validate_quote.return_value = SimpleNamespace(
            is_valid=True, issues=[], staleness_seconds=1.0)
        pm._ibkr = MagicMock()
        pm._ibkr.connected = False
        pm._market_data = MagicMock()
        pm._market_data.get_quote = AsyncMock(return_value=SimpleNamespace(
            mid=700.0, bid=699.99, ask=700.01, last=700.0, volume=1000,
            timestamp=datetime(2026, 8, 10, 14, 0, 0)))
        pm._option_position_unrealized = lambda *a, **k: 12.0  # small winner
        pm._state = MagicMock()
        pm._state.get_high_water_mark.return_value = 0.05
        pm._pdt_guard = MagicMock()
        pm._pdt_guard.would_be_day_trade.return_value = False
        cal = MagicMock()
        cal.days_until_next_event.return_value = d2e
        pm._economic_cal = cal
        return pm

    def _live_trade(self, strategy, expiry=None):
        legs = [{"strike": 690.0, "right": "P", "action": "SELL"},
                {"strike": 720.0, "right": "C", "action": "SELL"}]
        if strategy == "iron_condor":
            legs += [{"strike": 660.0, "right": "P", "action": "BUY"},
                     {"strike": 750.0, "right": "C", "action": "BUY"}]
        return SimpleNamespace(
            trade_id=f"T-{strategy}", symbol="SPY", strategy=strategy,
            contract_type=strategy, quantity=1, entry_price=4.24,
            direction=SimpleNamespace(value="neutral"),
            expiry=expiry, strike=None, entry_time="2026-08-04T10:51:37",
            legs=json.dumps(legs))

    async def test_macro_flatten_windows_match_across_live_and_engine(
            self, synth_df, synth_features, monkeypatch):
        """CENSUS-2 (2026-09-27): drives BOTH the real engine
        (Backtester._check_exit_credit) and the real live path
        (PortfolioManager._evaluate_position) across every strategy and a
        spread of days-to-event, instead of grepping each one's source for
        the shared table's name. exit_policy.macro_flatten_window_days
        (imported via `from ... import exit_policy`, a MODULE reference, not
        a value import) is what both engines actually call at run time --
        confirmed by reading engine.py:2609 and portfolio.py directly."""
        from ait.execution import exit_policy

        assert exit_policy.macro_flatten_window_days("short_strangle") == 5
        assert exit_policy.macro_flatten_window_days("jade_lizard") == 5
        assert exit_policy.macro_flatten_window_days("cash_secured_put") == 1
        assert exit_policy.macro_flatten_window_days("covered_call") == 1
        # Defined-risk condors hold through events (PLAN 2026-08-04)
        assert exit_policy.macro_flatten_window_days("iron_condor") is None

        monkeypatch.setenv("AIT_SKIP_MACRO_EVENTS", "1")

        class _Cal:
            def __init__(self, d2e):
                self._d2e = d2e

            def days_until_next_event(self, d):
                return self._d2e

        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        today = date(2026, 6, 3)
        pos_common = {"expiry_date": "2026-07-03",  # 30 DTE, clear of the
                      "entry_date": "2026-06-03"}    # DTE<=5 close below

        # Hardcoded, not re-derived from macro_flatten_window_days itself --
        # deriving `window` from the function under test would make
        # expect_flatten self-referential and unable to catch a mutation of
        # the table (confirmed: an earlier draft that called
        # macro_flatten_window_days() here passed even with
        # cash_secured_put's window mutated to 4).
        expected_windows = {"short_strangle": 5, "jade_lizard": 5,
                            "cash_secured_put": 1, "covered_call": 1,
                            "iron_condor": None}
        for strategy in ("short_strangle", "jade_lizard", "cash_secured_put",
                         "covered_call", "iron_condor"):
            window = expected_windows[strategy]
            for d2e in (0, 1, 2, 5, 6):
                expect_flatten = window is not None and d2e <= window

                bt._economic_cal = _Cal(d2e)
                out = bt._check_exit_credit(
                    dict(pos_common, strategy=strategy), pnl_pct=0.05,
                    current_date=today)
                if expect_flatten:
                    assert out is not None and out["exit_reason"] == "macro_event_flatten", (
                        f"engine: {strategy} at d2e={d2e} (window={window}) "
                        f"should flatten, got {out}"
                    )
                else:
                    assert out is None, (
                        f"engine: {strategy} at d2e={d2e} (window={window}) "
                        f"should hold, got {out}"
                    )

                pm = self._live_pm(d2e=d2e)
                st = await pm._evaluate_position(self._live_trade(strategy))
                if expect_flatten:
                    assert st.should_exit is True and "macro_event_flatten" in (st.exit_reason or ""), (
                        f"live: {strategy} at d2e={d2e} (window={window}) "
                        f"should flatten, got {st}"
                    )
                else:
                    assert "macro_event_flatten" not in (st.exit_reason or ""), (
                        f"live: {strategy} at d2e={d2e} (window={window}) "
                        f"should not macro-flatten, got {st}"
                    )

        # Delegation, not a hand-copy: patching the SHARED table must move
        # BOTH engines together.
        monkeypatch.setitem(exit_policy.MACRO_FLATTEN_WINDOW_DAYS, "cash_secured_put", 3)
        bt._economic_cal = _Cal(3)
        out = bt._check_exit_credit(
            dict(pos_common, strategy="cash_secured_put"), pnl_pct=0.05,
            current_date=today)
        assert out is not None and out["exit_reason"] == "macro_event_flatten"
        pm = self._live_pm(d2e=3)
        st = await pm._evaluate_position(self._live_trade("cash_secured_put"))
        assert st.should_exit is True and "macro_event_flatten" in (st.exit_reason or "")

    async def test_expiry_close_dte_matches_across_live_and_engine(
            self, synth_df, synth_features, monkeypatch):
        """CENSUS-2 (2026-09-27): both engine.py:2597 and portfolio.py read
        EXPIRY_APPROACHING_DTE, but NOT the same way -- engine.py does
        `exit_policy.EXPIRY_APPROACHING_DTE` (a module-attribute lookup at
        call time), while portfolio.py does `from ...exit_policy import
        (EXPIRY_APPROACHING_DTE, ...)` (a VALUE import, bound once at import
        time) -- confirmed directly: patching exit_policy's attribute moves
        the engine but NOT live's already-bound name. So "delegates to the
        same constant" can only be proven live-side by a BOUNDARY-VALUE
        match, not by a patch-and-compare -- the engine gets both proofs."""
        from ait.execution import exit_policy
        from ait.utils.time import today_et

        assert exit_policy.EXPIRY_APPROACHING_DTE == 5
        monkeypatch.setenv("AIT_SKIP_MACRO_EVENTS", "0")  # keep macro out of the way

        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        today = date(2026, 6, 3)
        # Live's own DTE math reads today_et() (portfolio.py:481), not the
        # engine's `current_date` -- expiry must be built off the REAL
        # wall-clock today for the live half of this comparison to be honest.
        live_today = today_et()

        for dte, expect_close in ((5, True), (6, False)):
            expiry = today + pd.Timedelta(days=dte)
            pos = {"expiry_date": str(expiry), "entry_date": str(today),
                  "strategy": "iron_condor"}
            out = bt._check_exit_credit(dict(pos), pnl_pct=0.05, current_date=today)
            if expect_close:
                assert out is not None and out["exit_reason"] in (
                    "expiry", "expiry_approaching"), (
                    f"engine: dte={dte} should close at the shared DTE "
                    f"boundary, got {out}"
                )
            else:
                assert out is None, f"engine: dte={dte} should hold, got {out}"

            live_expiry = live_today + pd.Timedelta(days=dte)
            trade = self._live_trade("iron_condor", expiry=str(live_expiry))
            st = await self._live_pm(d2e=99)._evaluate_position(trade)
            if expect_close:
                assert st.should_exit is True and "expiry_approaching" in (st.exit_reason or ""), (
                    f"live: dte={dte} should close at the shared DTE "
                    f"boundary, got {st}"
                )
            else:
                assert "expiry_approaching" not in (st.exit_reason or ""), (
                    f"live: dte={dte} should hold, got {st}"
                )

        # Engine-only: the boundary itself must come from the shared
        # constant, not a hand-copied 5 (live can't be proven this way --
        # see the docstring's value-import note).
        monkeypatch.setattr(exit_policy, "EXPIRY_APPROACHING_DTE", 7)
        expiry7 = today + pd.Timedelta(days=7)
        pos7 = {"expiry_date": str(expiry7), "entry_date": str(today),
               "strategy": "iron_condor"}
        out7 = bt._check_exit_credit(dict(pos7), pnl_pct=0.05, current_date=today)
        assert out7 is not None and out7["exit_reason"] in ("expiry", "expiry_approaching")

    def test_engine_honors_time_decay_scaling_config(
        self, synth_df, synth_features, monkeypatch
    ):
        # Live gates the ladder on exit.time_decay_scaling (portfolio.py); the
        # engine used to run the ladder unconditionally, so setting the flag
        # false moved live to flat 0.50 targets while research kept the ladder.
        import ait.config.settings as settings_mod

        stub = SimpleNamespace(exit=SimpleNamespace(time_decay_scaling=False))
        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: stub)
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._exit_time_decay_scaling is False
        # dte=8, pnl at +35% of credit: laddered target 0.30 would fire;
        # flat 0.50 (scaling off) must NOT.
        pos = {
            "expiry_date": str(date(2024, 6, 11)),
            "entry_date": str(date(2024, 6, 1)),
            "strategy": "iron_condor",
        }
        out = bt._check_exit_credit(dict(pos), 0.35, date(2024, 6, 3))
        assert out is None, "scaling off must run live's flat 0.50 target"

        stub_on = SimpleNamespace(exit=SimpleNamespace(time_decay_scaling=True))
        monkeypatch.setattr(settings_mod, "load_settings", lambda *a, **k: stub_on)
        bt_on = Backtester(**_engine_kwargs(synth_df, synth_features))
        out_on = bt_on._check_exit_credit(dict(pos), 0.35, date(2024, 6, 3))
        assert out_on is not None and out_on["exit_reason"] == "take_profit_short"

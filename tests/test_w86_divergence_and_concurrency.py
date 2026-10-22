"""W86b: research follows the config's concurrency; the default-divergence report has a class guard.

  * `backtest.max_concurrent_positions` (config 3, research used a hard-coded 1): WalkForwardConfig and
    StrategyOptimizer now resolve it from the loaded settings; the bare Backtester keeps its explicit
    harness default (R20b); every engine they build receives it
  * the per-window trainer reads the LOADED `ml` section (it built a bare MLConfig())
  * `default_divergences` also sees list-valued / None-default fields and no longer hides
    `account.pdt_account_under_25k`; every divergence is classified in DIVERGENCE_CLASSES, and each
    `research_reads_loaded` row is proven on the real code path
  * the parity manifest shows the three research-read knobs
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import optuna  # noqa: F401  (the optimizer module imports it)
import pytest

import ait.backtesting.engine as engine_mod
import ait.backtesting.walkforward as wf_mod
import ait.config.settings as st
from ait.backtesting.engine import Backtester
from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
from ait.config.settings import (
    DIVERGENCE_CLASSES, DIVERGENCE_KINDS, BacktestConfig, MLConfig, Settings, classify_divergences,
    default_divergences, load_settings, resolve_max_contracts_per_trade, resolve_position_size_pct,
)
from ait.optimization.optimizer import StrategyOptimizer

from tests.test_r20_research_validity import (  # noqa: F401 - shared fixtures
    _engine_kwargs, _recorder_factory, _run_window, synth_df, synth_features)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real():
    return load_settings(str(ROOT / "config.yaml"), ignore_frozen=True)


def _with(real, **sections) -> Settings:
    """A copy of the real settings with section fields replaced: _with(real, backtest={"x": 2})."""
    d = real.model_dump(exclude={"ibkr", "api_keys"})
    for sec, fields in sections.items():
        d[sec].update(fields)
    return Settings(**d)


def _patched_load(monkeypatch, settings):
    monkeypatch.setattr(st, "load_settings", lambda *a, **k: settings)


# --------------------------------------------------------------------------
# max_concurrent_positions
# --------------------------------------------------------------------------

class TestConcurrencyWiring:
    def test_the_shipped_config_says_three_and_research_now_follows_it(self, real):
        assert real.backtest.max_concurrent_positions == 3
        assert WalkForwardConfig(train_window_models=False).max_concurrent_positions == 3

    def test_an_explicit_value_wins(self):
        assert WalkForwardConfig(train_window_models=False, max_concurrent_positions=1
                                 ).max_concurrent_positions == 1
        assert WalkForwardConfig(train_window_models=False, max_concurrent_positions=2
                                 ).max_concurrent_positions == 2

    def test_the_loaded_value_is_what_is_read(self, real, monkeypatch):
        _patched_load(monkeypatch, _with(real, backtest={"max_concurrent_positions": 2}))
        assert WalkForwardConfig(train_window_models=False).max_concurrent_positions == 2

    def test_no_config_falls_back_to_the_schema_default(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("no config")
        monkeypatch.setattr(st, "load_settings", boom)
        assert WalkForwardConfig(train_window_models=False).max_concurrent_positions == \
            BacktestConfig().max_concurrent_positions == 1

    def test_the_bare_engine_keeps_its_explicit_harness_default(self, synth_df, synth_features):
        """R20b decision D: initial_capital / max_concurrent_positions stay explicit on the bare engine."""
        bt = Backtester(**_engine_kwargs(synth_df, synth_features))
        assert bt._max_concurrent_positions == 1

    def test_the_oos_and_shadow_engines_receive_it(self, synth_df, monkeypatch):
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"],
                                   config=WalkForwardConfig(train_window_models=False))
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink and sink[-1]["max_concurrent_positions"] == 3
        sink.clear()
        wf._train_window_meta_labeler(train_df=synth_df, symbol="SPY", window_id=1,
                                      predictor=None, window_cfg=wf._config)
        assert sink and sink[-1]["max_concurrent_positions"] == 3

    def test_an_explicit_one_still_reaches_the_engines(self, synth_df, monkeypatch):
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(
            train_window_models=False, max_concurrent_positions=1))
        sink: list = []
        monkeypatch.setattr(wf_mod, "Backtester", _recorder_factory(sink))
        _run_window(wf, synth_df)
        assert sink and sink[-1]["max_concurrent_positions"] == 1

    def test_the_window_optimizer_receives_it(self, synth_df, monkeypatch):
        import ait.optimization.optimizer as om
        seen = {}

        class Spy:
            def __init__(self, **kw):
                seen.update(kw)

            def run(self, **kw):
                raise RuntimeError("stop here")
        monkeypatch.setattr(om, "StrategyOptimizer", Spy)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"], config=WalkForwardConfig(
            train_window_models=False, optimize_per_window=True))
        wf._optimize_window_params(synth_df, "SPY", 1, ["iron_condor"])
        assert seen["max_concurrent_positions"] == 3

    def test_the_optimizer_resolves_it_and_its_trials_carry_it(self, monkeypatch, synth_df):
        sink: list = []
        monkeypatch.setattr(engine_mod, "Backtester", _recorder_factory(sink))
        opt = StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1)
        assert opt._max_concurrent_positions == 3
        opt._data = {"SPY": synth_df}
        opt._run_backtest({})
        assert sink and sink[0]["max_concurrent_positions"] == 3
        sink.clear()
        one = StrategyOptimizer(symbols=["SPY"], strategies=["iron_condor"], n_trials=1,
                                max_concurrent_positions=1)
        one._data = {"SPY": synth_df}
        one._run_backtest({})
        assert sink and sink[0]["max_concurrent_positions"] == 1

    def test_more_concurrency_trades_at_least_as_much(self, synth_df, synth_features):
        one = Backtester(**_engine_kwargs(synth_df, synth_features, max_concurrent_positions=1)).run()
        three = Backtester(**_engine_kwargs(synth_df, synth_features, max_concurrent_positions=3)).run()
        assert len(three.trades) >= len(one.trades)


class TestTrainerReadsTheLoadedMlSection:
    def test_the_window_trainer_gets_the_loaded_ml_config(self, real, synth_df, monkeypatch):
        seen = {}

        class Spy:
            def __init__(self, cfg, persist_artifacts=True):
                seen["cfg"] = cfg

            def train(self, *a, **k):
                return {}
        monkeypatch.setattr(wf_mod, "DirectionPredictor", Spy)
        loaded = _with(real, ml={"lookback_days": 321})
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"],
                                   config=WalkForwardConfig(train_window_models=False))
        monkeypatch.setattr(wf, "_settings", loaded)
        wf._train_window_model(train_df=synth_df, symbol="SPY", window_id=1)
        assert seen["cfg"].lookback_days == 321                    # a bare MLConfig() would say 504

    def test_without_a_config_the_schema_default_applies(self, synth_df, monkeypatch):
        seen = {}

        class Spy:
            def __init__(self, cfg, persist_artifacts=True):
                seen["cfg"] = cfg

            def train(self, *a, **k):
                return {}
        monkeypatch.setattr(wf_mod, "DirectionPredictor", Spy)
        wf = WalkForwardBacktester(["SPY"], ["iron_condor"],
                                   config=WalkForwardConfig(train_window_models=False))
        monkeypatch.setattr(wf, "_settings", None)
        wf._train_window_model(train_df=synth_df, symbol="SPY", window_id=1)
        assert seen["cfg"].lookback_days == MLConfig().lookback_days


# --------------------------------------------------------------------------
# the divergence report and its class guard
# --------------------------------------------------------------------------

class TestDivergenceReport:
    def test_every_current_divergence_is_classified(self, real):
        got = classify_divergences(default_divergences(real))
        assert got["unclassified"] == [], (
            "a config.yaml value differs from its code default and nobody has classified it: add it to "
            "DIVERGENCE_CLASSES (live_only | research_reads_loaded | deliberate_research_difference | dead)")

    def test_the_registry_rows_are_real_fields_with_a_known_kind_and_a_reason(self):
        for dotted, (kind, reason) in DIVERGENCE_CLASSES.items():
            section, field = dotted.split(".")
            assert field in type(getattr(Settings(), section)).model_fields, f"{dotted} is not a setting"
            assert kind in DIVERGENCE_KINDS and len(reason) > 20, dotted

    def test_a_new_unclassified_divergence_is_reported_as_such(self, real):
        edited = _with(real, risk={"max_api_failures": 7})
        by = classify_divergences(default_divergences(edited))
        assert by["unclassified"] == ["risk.max_api_failures"]

    def test_list_and_none_default_fields_are_now_seen(self, real):
        names = {d for d, _c, _a in default_divergences(real)}
        assert {"trading.universe", "options.dte_range", "options.strategies",
                "backtest.position_size_pct"} <= names

    def test_the_pdt_flag_is_no_longer_hidden_by_the_secret_filter(self, real):
        assert "account.pdt_account_under_25k" in {d for d, _c, _a in default_divergences(real)}

    def test_secrets_stay_hidden(self):
        s = Settings(ibkr={"account": "DU123", "host": "10.9.9.9"}, api_keys={"finnhub_api_key": "k"})
        shown = {d for d, _c, _a in default_divergences(s)}
        assert not any(d.startswith(("ibkr.", "api_keys.")) for d in shown)
        assert st._is_secret("ibkr", "host") and st._is_secret("risk", "some_token")
        assert not st._is_secret("account", "pdt_account_under_25k")

    def test_the_known_classes(self, real):
        by = classify_divergences(default_divergences(real))
        assert "ml.entry_gates_enabled" in by["deliberate_research_difference"]
        assert "sentiment.enabled" in by["dead"]
        assert "backtest.max_concurrent_positions" in by["research_reads_loaded"]
        assert "trading.max_daily_trades" in by["live_only"]

    def test_the_startup_report_carries_the_classes(self, real, monkeypatch):
        events = []

        class Log:
            def info(self, event, **kw):
                events.append((event, kw))
        import ait.utils.logging as ulog
        monkeypatch.setattr(ulog, "get_logger", lambda name: Log())
        st._report_default_divergence(real)
        ev = dict(events)["config_default_divergence"]
        assert ev["by_class"]["unclassified"] == [] and ev["count"] == len(default_divergences(real))
        assert "BARE" in ev["note"] and "deliberate_research_difference" in ev["note"]


class TestResearchReadsLoadedRows:
    """Each `research_reads_loaded` row, proven on the real code path: change the LOADED value and the
    research-side resolver follows."""

    def test_the_registry_has_exactly_the_rows_proven_here(self):
        rows = {d for d, (k, _r) in DIVERGENCE_CLASSES.items() if k == "research_reads_loaded"}
        assert rows == {"positions.max_open_positions", "positions.max_contracts_per_trade",
                        "risk.min_confidence", "options.dte_range", "backtest.position_size_pct",
                        "backtest.max_concurrent_positions", "options.strategies"}   # W87a added the last

    def test_options_strategies(self, real):
        """W87a: a bare research run trades the LOADED live selector list (explicit lists still win)."""
        from ait.config.research import resolve_universe
        assert resolve_universe(None, None, _with(real, options={"strategies": ["iron_condor", "long_call"]}))[1] \
            == ["iron_condor", "long_call"]
        assert resolve_universe(None, ["bull_call_spread"], real)[1] == ["bull_call_spread"]

    def test_max_contracts_per_trade(self, real):
        assert resolve_max_contracts_per_trade(None, _with(real, positions={"max_contracts_per_trade": 4})) == 4

    def test_position_size_pct(self, real):
        assert resolve_position_size_pct(None, _with(real, backtest={"position_size_pct": 0.07})) == 0.07
        assert resolve_position_size_pct(None, _with(real, backtest={"position_size_pct": None})) == \
            real.risk.max_position_risk_pct

    def test_min_confidence(self, real, monkeypatch):
        _patched_load(monkeypatch, _with(real, risk={"min_confidence": 0.61}))
        assert WalkForwardConfig(train_window_models=False).min_confidence == 0.61

    def test_the_engines_entry_dte_is_dte_range_zero(self, real, synth_df, synth_features, monkeypatch):
        _patched_load(monkeypatch, _with(real, options={"dte_range": [18, 30]}))
        assert Backtester(**_engine_kwargs(synth_df, synth_features))._entry_dte == 18

    def test_max_open_positions_feeds_the_shared_pool(self):
        """AST: the walk-forward hands allocate_pool the LOADED positions.max_open_positions."""
        tree = ast.parse((ROOT / "src" / "ait" / "backtesting" / "walkforward.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) == "allocate_pool"]
        assert calls, "allocate_pool is no longer called by the walk-forward"
        kw = {k.arg: ast.dump(k.value) for k in calls[0].keywords}
        assert "max_open_positions" in kw and "max_open_positions" in kw["max_open_positions"]
        assert "_pos" in kw["max_open_positions"] or "_settings" in kw["max_open_positions"]


# --------------------------------------------------------------------------
# the parity manifest
# --------------------------------------------------------------------------

class TestManifest:
    @staticmethod
    def _manifest(monkeypatch, research_settings):
        import run_backtest as rb
        monkeypatch.setattr(rb, "_research_settings", lambda: research_settings)
        old = sys.argv
        sys.argv = ["run_backtest.py"]
        try:
            return rb.build_parity_manifest(rb.parse_args())
        finally:
            sys.argv = old

    def test_the_three_rows_are_shown_and_agree_today(self, real, monkeypatch):
        m = self._manifest(monkeypatch, real)
        for key, value in (("min_confidence", 0.5), ("max_open_positions", 10), ("max_concurrent_positions", 3)):
            assert m["live"][key] == value and m["backtest"][key] == value
        starts = [w.split(":")[0] for w in m["parity_warnings"]]
        assert not {"min_confidence", "max_open_positions", "max_concurrent_positions"} & set(starts)

    def test_an_overlay_that_differs_becomes_a_warning(self, real, monkeypatch):
        other = _with(real, backtest={"max_concurrent_positions": 1}, risk={"min_confidence": 0.7})
        m = self._manifest(monkeypatch, other)
        warned = {w.split(":")[0] for w in m["parity_warnings"]}
        assert {"max_concurrent_positions", "min_confidence"} <= warned

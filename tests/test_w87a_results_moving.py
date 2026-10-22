"""W87a (results-moving part): the research entry points' default targets and the walk-forward slippage.

Two deliberate changes that move numbers for a BARE run, separated from the no-behaviour-change hygiene
(tests/test_w87a_hygiene.py) and measured before/after in PLAN.md:

* ``run_backtest.py`` / ``run_optimizer.py`` default ``--symbols`` to ``backtest.default_symbols`` (QQQ) and
  ``--strategies`` to ``options.strategies``. Before: a ten-name list (the bare run crashed on AAPL: no dividend
  rows) and four strategies, two of them not in the live selector.
* ``WalkForwardConfig.slippage_pct`` is a ``None`` sentinel resolved from ``backtest.slippage_pct`` (0.01) instead
  of a 0.03 literal. Debit structures only; iron condors never read it.
"""
from __future__ import annotations

import asyncio
import sys

import pytest
from pydantic import ValidationError

from ait.config.research import resolve_universe
from ait.config.settings import DIVERGENCE_CLASSES, BacktestConfig, load_settings


@pytest.fixture(scope="module")
def real():
    return load_settings()


def _with(settings, **sections):
    update = {name: getattr(settings, name).model_copy(update=vals) for name, vals in sections.items()}
    return settings.model_copy(update=update)


class TestDefaultSymbolsField:
    def test_the_schema_default_is_qqq_and_the_shipped_config_agrees(self, real):
        assert BacktestConfig().default_symbols == ["QQQ"]
        assert real.backtest.default_symbols == ["QQQ"]

    def test_values_are_normalised(self):
        assert BacktestConfig(default_symbols=["qqq", " spy ", "QQQ"]).default_symbols == ["QQQ", "SPY"]

    @pytest.mark.parametrize("bad", [[], ["QQQ", ""], ["not a ticker"], ["Q" * 11], ["1ABC"],
                                     [f"S{i}" for i in range(21)]])
    def test_bad_lists_fail_at_load(self, bad):
        with pytest.raises(ValidationError):
            BacktestConfig(default_symbols=bad)

    def test_the_universe_row_still_says_live_only_and_names_the_new_home(self):
        kind, reason = DIVERGENCE_CLASSES["trading.universe"]
        assert kind == "live_only" and "default_symbols" in reason


class TestResolveUniverse:
    def test_bare_runs_follow_the_config(self, real):
        assert resolve_universe(None, None, real) == (["QQQ"], list(real.options.strategies))
        moved = _with(real, backtest={"default_symbols": ["SPY", "IWM"]}, options={"strategies": ["long_call"]})
        assert resolve_universe(None, None, moved) == (["SPY", "IWM"], ["long_call"])

    def test_explicit_lists_win_and_are_copied(self, real):
        syms, strats = ["AAPL"], ["iron_condor"]
        out = resolve_universe(syms, strats, real)
        assert out == (["AAPL"], ["iron_condor"]) and out[0] is not syms

    def test_an_empty_explicit_list_falls_back_to_the_config(self, real):
        assert resolve_universe([], [], real)[0] == ["QQQ"]


class TestEntryPointsDefaultToTheConfig:
    def test_the_cli_defaults_are_unset(self, monkeypatch):
        import run_backtest as rb
        import run_optimizer as ro
        for mod, name in ((rb, "run_backtest.py"), (ro, "run_optimizer.py")):
            monkeypatch.setattr(sys, "argv", [name])
            a = mod.parse_args()
            assert a.symbols is None and a.strategies is None, name

    def test_run_backtest_resolves_them_before_anything_reads_them(self, monkeypatch, real):
        import ait.config.research as research
        import run_backtest as rb

        class Stop(Exception):
            pass

        seen = {}

        def spy(args):
            seen.update(symbols=args.symbols, strategies=args.strategies)
            raise Stop

        monkeypatch.setattr(research, "use_research_config", lambda *a, **k: real)
        monkeypatch.setattr(rb, "build_parity_manifest", spy)
        monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
        with pytest.raises(Stop):
            asyncio.run(rb.run_backtest(rb.parse_args()))
        assert seen == {"symbols": ["QQQ"], "strategies": list(real.options.strategies)}

    def test_an_explicit_flag_still_wins(self, monkeypatch, real):
        import ait.config.research as research
        import run_backtest as rb

        class Stop(Exception):
            pass

        seen = {}
        monkeypatch.setattr(research, "use_research_config", lambda *a, **k: real)
        monkeypatch.setattr(rb, "build_parity_manifest",
                            lambda args: (seen.update(symbols=args.symbols), (_ for _ in ()).throw(Stop))[1])
        monkeypatch.setattr(sys, "argv", ["run_backtest.py", "--symbols", "SPY", "IWM"])
        with pytest.raises(Stop):
            asyncio.run(rb.run_backtest(rb.parse_args()))
        assert seen["symbols"] == ["SPY", "IWM"]

    def test_run_optimizer_resolves_them_too(self, monkeypatch, real):
        import ait.config.research as research
        import ait.optimization.optimizer as opt_mod
        import run_optimizer as ro

        captured = {}

        class Stop(Exception):
            pass

        class Spy:
            def __init__(self, **kw):
                captured.update(symbols=kw["symbols"], strategies=kw["strategies"])
                raise Stop

        monkeypatch.setattr(research, "use_research_config", lambda *a, **k: real)
        monkeypatch.setattr(opt_mod, "StrategyOptimizer", Spy)
        monkeypatch.setattr(sys, "argv", ["run_optimizer.py", "--n-trials", "1"])
        with pytest.raises(Stop):
            ro.main()
        assert captured == {"symbols": ["QQQ"], "strategies": list(real.options.strategies)}


class TestSlippageFollowsTheConfig:
    def test_the_walk_forward_default_is_the_sentinel(self):
        from ait.backtesting.walkforward import WalkForwardConfig
        assert WalkForwardConfig().slippage_pct is None
        assert WalkForwardConfig(slippage_pct=0.03).slippage_pct == 0.03      # the old behaviour, on request

    def test_the_engine_a_window_builds_resolves_the_config_value(self, real):
        import numpy as np
        import pandas as pd

        from ait.backtesting.engine import Backtester
        from ait.backtesting.walkforward import WalkForwardConfig
        idx = pd.date_range("2024-01-02", periods=60, freq="B")
        close = 480.0 * np.cumprod(1 + np.full(60, 0.001))
        data = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                             "Volume": np.full(60, 5e6)}, index=idx)
        wf = WalkForwardConfig()
        assert Backtester(data=data, strategies=["iron_condor"], settings=real,
                          slippage_pct=wf.slippage_pct)._slippage_pct == real.backtest.slippage_pct == 0.01
        moved = _with(real, backtest={"slippage_pct": 0.02})
        assert Backtester(data=data, strategies=["iron_condor"], settings=moved,
                          slippage_pct=wf.slippage_pct)._slippage_pct == 0.02
        assert Backtester(data=data, strategies=["iron_condor"], settings=real,
                          slippage_pct=0.03)._slippage_pct == 0.03

"""W108 S1: carried equity in the walk-forward (an opt-in estimate) and ONE equity path for the reported figures.

Before W108 S1 every walk-forward window restarted at ``account.capital`` (the per-symbol OOS engines, the shared pool and
the window's result), and the aggregate figures disagreed: ``total_return`` multiplied the window returns together while
``max_drawdown`` and ``equity_curve()`` added every trade's dollars to one starting capital.

  1. ``carry_equity`` off (the default): every trade, window and metric equals the record captured at 11be785 BEFORE the
     change (``tests/data/w108s1_walkforward_golden.json``), except ``total_return`` and the two figures derived from it.
  2. One method: ``total_return`` is the end of the same dollar path ``max_drawdown`` and ``equity_curve()`` walk.
  3. ``carry_equity`` on: the reset run is unchanged (it stays the basis for parameter selection) and a carried estimate
     is attached, whose windows chain (each starts at the previous one's ending equity) and whose engines and pool are
     seeded with that equity.
  4. Parallel window workers with ``carry_equity`` are refused, not silently used.

Every test runs the real walk-forward code (the context loader is the only stub: it would read a store or the network).
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import ait.backtesting.walkforward as wf_mod
from ait.backtesting.result import BacktestResult
from ait.backtesting.walkforward import (
    WalkForwardBacktester,
    WalkForwardConfig,
    WalkForwardResult,
    WindowResult,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "data"))
import _w108s1_golden_inputs as gi  # noqa: E402

GOLDEN = json.loads(gi.GOLDEN_PATH.read_text(encoding="utf-8"))
# The only figures a carry-off run is allowed to move: total_return (compounded window returns -> the dollar path) and
# the two that are computed from it.
MOVED = {"total_return", "cash_drag_adjusted_return", "raroc"}


def _same(got, want, path="") -> None:
    """Recursive equality: exact for strings / ints / structure, 1e-9 relative for floats (the golden-record convention)."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and sorted(got) == sorted(want), path
        for k in want:
            _same(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want)):
            _same(g, w, f"{path}[{i}]")
    elif isinstance(want, float) and not isinstance(want, bool):
        if math.isinf(want) or math.isnan(want):
            assert str(got) == str(want), path
        else:
            assert got == pytest.approx(want, rel=1e-9, abs=1e-9), path
    else:
        assert got == want, path


def _all_pnl(result) -> float:
    return sum(t.get("pnl", 0) for w in result.windows for t in w.backtest_result.trades)


@pytest.fixture(scope="module")
def off_run():
    return gi.run(gi.make_config())


@pytest.fixture(scope="module")
def on_run():
    return gi.run(gi.make_config(carry_equity=True))


# ---------------------------------------------------------------------------------------------------------------------
# 1. carry_equity off: the golden record
# ---------------------------------------------------------------------------------------------------------------------
class TestDefaultRunIsUnchanged:
    def test_the_switch_defaults_off(self):
        assert WalkForwardConfig(train_window_models=False).carry_equity is False

    def test_every_trade_and_window_equals_the_record_captured_before_the_change(self, off_run):
        snap = gi.snapshot(off_run)
        _same(snap["windows"], GOLDEN["windows"], "windows")
        _same(snap["symbol_results"], GOLDEN["symbol_results"], "symbol_results")
        _same(snap["strategy_results"], GOLDEN["strategy_results"], "strategy_results")
        _same(snap["entry_funnel"], GOLDEN["entry_funnel"], "entry_funnel")
        _same(snap["equity_curve"], GOLDEN["equity_curve"], "equity_curve")

    def test_every_metric_but_total_return_and_its_two_derivatives_equals_the_record(self, off_run):
        snap = gi.snapshot(off_run)
        kept = {k: v for k, v in GOLDEN["metrics"].items() if k not in MOVED}
        _same({k: snap["metrics"][k] for k in kept}, kept, "metrics")

    def test_the_fixture_has_trades_in_several_windows(self, off_run):
        """The record is only evidence if it exercises the multi-window aggregation."""
        assert sum(1 for w in GOLDEN["windows"] if w["trades"]) >= 3
        assert off_run.carried_estimate is None and off_run.equity_basis == "reset"


# ---------------------------------------------------------------------------------------------------------------------
# 2. One method: every aggregate figure reads one dollar path
# ---------------------------------------------------------------------------------------------------------------------
def _window(wid: int, pnls: list[float], capital: float) -> WindowResult:
    trades = [{"pnl": p, "symbol": "SPY", "strategy": "iron_condor",
               "entry_date": f"2024-{wid:02d}-{10 + i:02d}", "exit_date": f"2024-{wid:02d}-{12 + i:02d}"}
              for i, p in enumerate(pnls)]
    return WindowResult(window_id=wid, train_start=date(2023, 1, 2), train_end=date(2023, 12, 29),
                        test_start=date(2024, wid, 1), test_end=date(2024, wid, 28),
                        backtest_result=BacktestResult(trades=trades, initial_capital=capital,
                                                       final_capital=capital + sum(pnls)))


class TestOneEquityPath:
    def test_total_return_is_the_end_of_the_path_drawdown_and_curve_walk(self):
        # Two windows, each restarted at $5,000: +$400 then +$300. Compounding said 1.08 x 1.06 - 1 = 14.48%; the
        # dollars on one $5,000 account are +$700 = 14.00%, which is where equity_curve() ends.
        r = WalkForwardResult(windows=[_window(1, [500.0, -100.0], 5000.0), _window(2, [300.0], 5000.0)],
                              initial_capital=5000.0)
        assert r.total_return == pytest.approx(700.0 / 5000.0)
        assert r.total_return == pytest.approx(r.equity_curve()["equity"].iloc[-1] / 5000.0 - 1.0)
        assert r.total_return != pytest.approx(1.08 * 1.06 - 1.0)

    def test_drawdown_is_measured_on_that_same_path(self):
        r = WalkForwardResult(windows=[_window(1, [500.0, -100.0], 5000.0), _window(2, [300.0], 5000.0)],
                              initial_capital=5000.0)
        assert r.max_drawdown == pytest.approx(100.0 / 5500.0)

    def test_on_chained_windows_the_compounded_and_dollar_figures_agree(self):
        """A carried run's windows chain, so the product of the window returns is the same number."""
        r = WalkForwardResult(windows=[_window(1, [500.0, -100.0], 5000.0), _window(2, [300.0], 5400.0)],
                              initial_capital=5000.0)
        compounded = (1 + 400.0 / 5000.0) * (1 + 300.0 / 5400.0) - 1
        assert r.total_return == pytest.approx(compounded) == pytest.approx(700.0 / 5000.0)

    def test_the_default_runs_total_return_is_its_dollar_path(self, off_run):
        assert off_run.total_return == pytest.approx(_all_pnl(off_run) / gi.CAPITAL)
        assert off_run.total_return == pytest.approx(off_run.equity_curve()["equity"].iloc[-1] / gi.CAPITAL - 1.0)
        assert off_run.final_equity == pytest.approx(gi.CAPITAL + _all_pnl(off_run))

    def test_the_record_shows_the_old_figure_was_the_compounded_one(self):
        """The golden total_return (pre-change) is the product of the window returns, not the dollar path."""
        compounded = math.prod(1 + w["total_return"] for w in GOLDEN["windows"]) - 1
        dollars = sum(t["pnl"] for w in GOLDEN["windows"] for t in w["trades"]) / gi.CAPITAL
        assert GOLDEN["metrics"]["total_return"] == pytest.approx(compounded)
        assert abs(compounded - dollars) > 1e-3


# ---------------------------------------------------------------------------------------------------------------------
# 3. carry_equity on: the reset run unchanged, a carried estimate attached
# ---------------------------------------------------------------------------------------------------------------------
class TestCarriedEstimate:
    def test_the_reset_result_is_identical_to_the_carry_off_run(self, off_run, on_run):
        """Parameter selection (learner, warm starts, global best) reads the reset windows only."""
        _same(gi.snapshot(on_run), gi.snapshot(off_run), "reset")
        assert on_run.equity_basis == "reset"

    def test_a_carried_estimate_is_attached_and_labelled(self, on_run):
        est = on_run.carried_estimate
        assert isinstance(est, WalkForwardResult)
        assert est.equity_basis == "carried_estimate" and est.carried_estimate is None
        assert est.initial_capital == gi.CAPITAL and est.windows

    def test_each_carried_window_starts_with_only_the_pnl_closed_before_its_first_test_day(self, on_run):
        """Review follow-up: not the previous window's ending equity (consecutive windows can share a boundary
        day, and overlapping windows would size entries with P&L realised weeks later)."""
        est = on_run.carried_estimate
        earlier: list[dict] = []
        for w in est.windows:
            br = w.backtest_result
            closed_before = sum(t["pnl"] for t in earlier
                                if date.fromisoformat(str(t["exit_date"])[:10]) < w.test_start)
            assert br.initial_capital == pytest.approx(gi.CAPITAL + closed_before), w.window_id
            assert br.final_capital == pytest.approx(br.initial_capital + sum(t["pnl"] for t in br.trades))
            earlier.extend(br.trades)

    def test_the_fixture_has_a_trade_closing_on_a_later_windows_first_test_day(self, on_run):
        """So the test above is not vacuous: the old chain would have sized that morning with that close."""
        est = on_run.carried_estimate
        firsts = {w.test_start for w in est.windows}
        assert any(date.fromisoformat(str(t["exit_date"])[:10]) in firsts
                   for w in est.windows for t in w.backtest_result.trades)

    def test_the_carried_total_return_is_the_dollar_path(self, on_run):
        est = on_run.carried_estimate
        assert est.total_return == pytest.approx(_all_pnl(est) / gi.CAPITAL, rel=1e-9)
        assert est.total_return == pytest.approx(est.equity_curve()["equity"].iloc[-1] / gi.CAPITAL - 1, rel=1e-9)

    def test_the_carried_path_differs_from_the_reset_path_on_the_fixture(self, on_run):
        """After the losing first window the carried pool is smaller, so the later windows are not the reset ones."""
        reset = [(w.window_id, w.backtest_result.initial_capital) for w in on_run.windows]
        carried = [(w.window_id, w.backtest_result.initial_capital) for w in on_run.carried_estimate.windows]
        assert any(c != pytest.approx(gi.CAPITAL) for _wid, c in carried[1:])
        assert all(c == pytest.approx(gi.CAPITAL) for _wid, c in reset)
        assert on_run.carried_estimate.total_return != pytest.approx(on_run.total_return)

    def test_the_summary_reports_the_carried_path_as_an_estimate(self, off_run, on_run):
        assert "CARRIED-EQUITY ESTIMATE" in on_run.summary()
        assert "CARRIED-EQUITY ESTIMATE" not in off_run.summary()
        head = on_run.carried_estimate.summary()
        assert "CARRIED-EQUITY ESTIMATE" in head and "estimate" in head.lower()


class TestCarriedEquityBefore:
    """The rule itself, on hand-built trades."""

    T = [{"exit_date": "2024-03-01", "pnl": -100.0}, {"exit_date": "2024-03-04", "pnl": 40.0},
         {"exit_date": "2024-03-25", "pnl": 7.0}]

    def test_a_close_on_the_first_test_day_does_not_size_that_day(self):
        assert wf_mod.carried_equity_before(5000.0, self.T, date(2024, 3, 4)) == pytest.approx(4900.0)
        assert wf_mod.carried_equity_before(5000.0, self.T, date(2024, 3, 5)) == pytest.approx(4940.0)

    def test_overlapping_windows_see_only_what_closed_before_they_start(self):
        assert wf_mod.carried_equity_before(5000.0, self.T, "2024-03-02") == pytest.approx(4900.0)
        assert wf_mod.carried_equity_before(5000.0, self.T, "2024-04-01") == pytest.approx(4947.0)

    def test_no_trades_is_the_pool(self):
        assert wf_mod.carried_equity_before(5000.0, [], date(2024, 1, 2)) == 5000.0


class _RecordingEngine(wf_mod.Backtester):
    """The real engine class with its construction capital recorded and run() replaced by one $200-risk condor."""
    capitals: list[float] = []

    def __init__(self, *a, **kw):
        type(self).capitals.append(float(kw["initial_capital"]))
        self._sym = kw.get("symbol")

    def run(self) -> BacktestResult:
        return BacktestResult(trades=[{
            "strategy": "iron_condor", "symbol": self._sym, "entry_date": "2023-11-06", "exit_date": "2023-11-20",
            "contracts": 1, "max_loss_per_share": 2.0, "max_loss": 200.0, "pnl": 30.0, "credit": 100.0,
            "expiry_date": "2023-11-24"}])


class TestCarriedReplayUsesTheCarriedEquity:
    def _window(self, monkeypatch, carry_capital):
        from tests.test_w63_capital_unification import _ohlcv
        _RecordingEngine.capitals = []
        monkeypatch.setattr(wf_mod, "Backtester", _RecordingEngine)
        cfg = WalkForwardConfig(train_days=300, test_days=50, gap_days=5, initial_capital=5000.0,
                                train_window_models=False)
        bt = WalkForwardBacktester(["SPY", "QQQ"], ["iron_condor"], config=cfg)
        wr, _, _ = bt._run_single_window(
            window_id=1, train_start=date(2023, 1, 1), train_end=date(2023, 10, 28),
            test_start=date(2023, 11, 2), test_end=date(2023, 12, 22),
            data={"SPY": _ohlcv(500), "QQQ": _ohlcv(500)}, vix_full=pd.DataFrame(), learner=None,
            carry_capital=carry_capital)
        return bt, wr

    def test_every_symbols_engine_and_the_pool_get_the_carried_equity(self, monkeypatch):
        # $3,000 carried: 5% = $150 < the $200 condor, so the carried pool refuses both; the reset pool ($5,000 -> $250)
        # takes the first and the cluster cap (SPY / QQQ, one position) the second.
        bt, wr = self._window(monkeypatch, 3000.0)
        assert _RecordingEngine.capitals == [5000.0, 3000.0, 5000.0, 3000.0]   # reset then carried, per symbol
        assert wr.backtest_result.initial_capital == 5000.0 and len(wr.backtest_result.trades) >= 1
        carried = bt._carried_by_window[1]
        assert carried.backtest_result.initial_capital == 3000.0
        assert carried.backtest_result.trades == [] and carried.pool_rejections == {"unaffordable": 2}
        assert carried.backtest_result.final_capital == 3000.0

    def test_no_carried_replay_without_a_carry_capital(self, monkeypatch):
        bt, wr = self._window(monkeypatch, None)
        assert _RecordingEngine.capitals == [5000.0, 5000.0]
        assert bt._carried_by_window == {}


# ---------------------------------------------------------------------------------------------------------------------
# 4. Parallel workers are refused, not silently used
# ---------------------------------------------------------------------------------------------------------------------
class TestCarryNeedsSequentialWindows:
    def test_carry_with_parallel_window_workers_is_refused(self):
        with pytest.raises(ValueError, match=r"carry_equity.*optimize_n_jobs"):
            WalkForwardConfig(train_window_models=False, carry_equity=True, optimize_n_jobs=2)

    def test_carry_with_one_worker_is_accepted(self):
        assert WalkForwardConfig(train_window_models=False, carry_equity=True, optimize_n_jobs=1).carry_equity

    def test_a_config_changed_after_construction_is_refused_by_run(self):
        cfg = gi.make_config()
        cfg.carry_equity, cfg.optimize_n_jobs = True, 2
        with pytest.raises(ValueError, match=r"carry_equity.*optimize_n_jobs"):
            gi.run(cfg)

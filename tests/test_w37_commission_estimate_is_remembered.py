"""W37 (R26 commission-estimate-live-config-read, part c).

A close is booked in two steps. At fill time the executor subtracts a
commission ESTIMATE (config: commission + regulatory fees per contract-leg).
Later the true-up replaces it with the broker's real figure:

    trued_pnl = booked_pnl + (estimate - real)

That only works if `estimate` is the number that was actually subtracted. It
was not remembered anywhere — the true-up RE-DERIVED it from whatever config
the process held at that moment. W35 froze settings per process, which removes
the drift inside one process; it cannot remove it ACROSS a restart, and the
deferred true-up (partial ledger -> settled post-market, or after the next
start) routinely crosses one. R26 measured the result: a fill booked under a
$0.65/leg config and trued up under $0.9168/leg leaves realized P&L off by
$2.13 on a one-lot condor, permanently — in the go-live verdict's own input.

Real StateManager, real executions ledger, real booking and true-up code.
"""
from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.execution.executor import TradeExecutor
from ait.utils.time import now_et

TID = "T-W37"
GROSS = 200.0                     # (2.50 credit - 0.50 buyback) * 100
REAL = 7.3344                     # what the broker charged, 8 leg fills


def _cost_model(monkeypatch, per_leg):
    monkeypatch.setattr(
        "ait.config.settings.load_settings",
        lambda *a, **k: SimpleNamespace(backtest=SimpleNamespace(
            commission_per_contract=per_leg, regulatory_fees_per_contract=0.0)))


def _state(tmp_path) -> StateManager:
    st = StateManager(db_path=tmp_path / "ait_state.db")
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, status, "
            "entry_time, entry_price, quantity, contract_type, legs) "
            "VALUES (?,?,?,?,?,?,?,1,'iron_condor','[1,2,3,4]')",
            (TID, "SPY", "iron_condor", "short", "filled",
             now_et().isoformat(), 2.50))
    return st


def _ledger(st, n_fills):
    with sqlite3.connect(st._db_path) as conn:
        for i in range(n_fills):
            conn.execute(
                "INSERT INTO executions (exec_id, order_id, trade_id, symbol, "
                "commission) VALUES (?,?,?,?,?)",
                (f"E{i}", 1, TID, "SPY", REAL / 8))


def _executor(st):
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._state = st
    return ex


def _orch(st):
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._state = st
    o._find_trade_by_id = lambda tid: st.get_trade_by_id(tid)
    return o


def _book_at_fill(st, monkeypatch, per_leg):
    _cost_model(monkeypatch, per_leg)
    trade = st.get_trade_by_id(TID)
    return _executor(st)._calculate_realized_pnl(trade, exit_price=0.50)


class TestTheTrueUpReversesWhatWasActuallySubtracted:
    def test_a_config_change_between_fill_and_true_up_does_not_drift_pnl(
            self, tmp_path, monkeypatch):
        st = _state(tmp_path)
        booked = _book_at_fill(st, monkeypatch, per_leg=0.65)        # est 5.20
        assert booked == pytest.approx(GROSS - 5.20)

        _ledger(st, 8)
        _cost_model(monkeypatch, 0.9168)       # a restart onto a recalibrated config
        trued = _orch(st)._true_up_commission(TID, booked)

        assert trued == pytest.approx(GROSS - REAL, abs=0.01), (
            f"realized P&L is off by {trued - (GROSS - REAL):+.2f}: the true-up "
            f"reversed today's estimate, not the one the fill subtracted")

    def test_the_deferred_post_market_settle_uses_it_too(self, tmp_path, monkeypatch):
        """Partial ledger at booking -> deferred -> settled by _post_market's
        own loop, which carried a second copy of the same re-derivation."""
        st = _state(tmp_path)
        booked = _book_at_fill(st, monkeypatch, per_leg=0.65)
        st.close_trade(TID, exit_price=0.50, realized_pnl=booked)
        _ledger(st, 3)                                   # commissionReports still arriving
        o = _orch(st)
        assert o._true_up_commission(TID, booked) == pytest.approx(booked)   # deferred
        assert TID in st.pending_trueup_trade_ids()

        _ledger_rest = 8 - 3
        with sqlite3.connect(st._db_path) as conn:
            for i in range(_ledger_rest):
                conn.execute(
                    "INSERT INTO executions (exec_id, order_id, trade_id, symbol, "
                    "commission) VALUES (?,?,?,?,?)",
                    (f"L{i}", 1, TID, "SPY", REAL / 8))
        _cost_model(monkeypatch, 0.9168)
        o._settle_deferred_trueups()

        assert st.get_trade_by_id(TID).realized_pnl == pytest.approx(
            GROSS - REAL, abs=0.01)
        assert TID not in st.pending_trueup_trade_ids()

    def test_the_memory_is_cleaned_up_once_used(self, tmp_path, monkeypatch):
        st = _state(tmp_path)
        booked = _book_at_fill(st, monkeypatch, per_leg=0.65)
        assert st.get_state(f"commission_est_{TID}", "") != ""
        _ledger(st, 8)
        _orch(st)._true_up_commission(TID, booked)
        assert st.get_state(f"commission_est_{TID}", "") == ""

    def test_a_close_with_no_remembered_estimate_still_trues_up(
            self, tmp_path, monkeypatch):
        """Control / upgrade path: closes booked before W37 have no key."""
        st = _state(tmp_path)
        _cost_model(monkeypatch, 0.9168)
        _ledger(st, 8)
        booked = GROSS - 0.9168 * 8
        assert _orch(st)._true_up_commission(TID, booked) == pytest.approx(
            GROSS - REAL, abs=0.01)

    def test_the_estimate_is_still_subtracted_without_a_state_store(self, monkeypatch):
        """The minimal host shim other suites drive this method through."""
        _cost_model(monkeypatch, 0.65)
        shim = SimpleNamespace(CREDIT_STRATEGIES=TradeExecutor.CREDIT_STRATEGIES)
        trade = SimpleNamespace(trade_id=TID, entry_price=2.50, quantity=1,
                                contract_type="iron_condor", strategy="iron_condor")
        assert TradeExecutor._calculate_realized_pnl(
            shim, trade, 0.50) == pytest.approx(GROSS - 5.20)

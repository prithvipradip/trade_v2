"""W48 (R24/R26 register, `trade-life-commission-trueup-gate-measures-wrong-
quantity`) — the commission true-up gate now measures commission PRESENCE,
not row presence.

`_true_up_commission` used to gate on `count_executions() >= 2*legs_n` — raw
row presence. `_sweep_executions` wrote every fill with `commission=0.0` the
instant it was seen, real or not, so a BAG-summary-inflated row count could
satisfy the gate while real leg commissions were still outstanding.
`commission` now defaults to NULL ("not yet reported") until a real
commissionReport lands; `count_executions_with_commission` counts only rows
where it has.

Found only by reading `tests/test_r16_broker_fixes.py`'s own production
fixture (a real 2026-08-04 SPY condor): the combo's BAG summary row gets its
OWN commissionReport too — reported as a genuine 0.0, not a missing one — so
excluding it from the completeness count (not just requiring
`commission IS NOT NULL`) is load-bearing, not decorative. Without the
exclusion, a couple of quickly-reported BAG rows can substitute for
still-pending LEG rows and reproduce this wave's own defect from a different
angle.

`_settle_deferred_trueups` (the post-market path) previously had NO
completeness check at all and deleted its `trueup_pending_*` marker
unconditionally, win or lose. It now shares the same gate and only clears
the marker once the ledger is actually complete.

Every test drives the real `StateManager`/`TradingOrchestrator` methods
against real sqlite, not source-string inspection. No inspect.getsource.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.execution.executor import BAG_CON_ID
from ait.utils.time import now_et


def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _seed_trade(
    st: StateManager, trade_id: str, legs: int = 4, realized_pnl: float = -100.0,
) -> None:
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, "
            "status, entry_time, entry_price, quantity, contract_type, legs, "
            "realized_pnl) VALUES (?,?,?,?,?,?,?,1,'iron_condor',?,?)",
            (trade_id, "QQQ", "iron_condor", "short", "closed",
             now_et().isoformat(), 1.90, json.dumps([{}] * legs), realized_pnl))


def _insert_execution(
    st: StateManager, exec_id: str, trade_id: str, con_id: int, commission: float,
) -> None:
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO executions (exec_id, trade_id, con_id, commission) "
            "VALUES (?,?,?,?)", (exec_id, trade_id, con_id, commission))


def _orch(st: StateManager) -> TradingOrchestrator:
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._state = st
    o._find_trade_by_id = lambda tid: st.get_trade_by_id(tid)
    return o


class TestBagRowExcludedFromCompleteness:
    """This wave's own reproduction: a real production shape where the BAG
    summary row reports its own -- genuinely real -- 0.0 commission before
    the leg rows do."""

    def test_bag_rows_do_not_substitute_for_missing_leg_rows(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-BAG", legs=4)
        for i in range(6):  # 6 of the 8 required leg rows
            _insert_execution(st, f"L{i}", "T-BAG", con_id=111, commission=1.0)
        # Both BAG rows (entry + exit) already reported -- real, not missing.
        _insert_execution(st, "BAG-1", "T-BAG", con_id=BAG_CON_ID, commission=0.0)
        _insert_execution(st, "BAG-2", "T-BAG", con_id=BAG_CON_ID, commission=0.0)

        n = st.count_executions_with_commission("T-BAG", exclude_con_id=BAG_CON_ID)

        assert n == 6, "BAG rows must not count toward the 8 required leg rows"

    def test_all_leg_rows_reported_satisfies_it_regardless_of_bag_rows(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-FULL", legs=4)
        for i in range(8):
            _insert_execution(st, f"L{i}", "T-FULL", con_id=111, commission=1.0)
        _insert_execution(st, "BAG-1", "T-FULL", con_id=BAG_CON_ID, commission=0.0)

        n = st.count_executions_with_commission("T-FULL", exclude_con_id=BAG_CON_ID)

        assert n == 8

    def test_true_up_commission_defers_while_only_bag_rows_are_known(self, tmp_path):
        """End to end through _true_up_commission itself, not just the count
        helper -- proves the gate is actually wired to exclude BAG rows."""
        st = _state(tmp_path)
        _seed_trade(st, "T-DEFER", legs=4)
        for i in range(6):
            _insert_execution(st, f"L{i}", "T-DEFER", con_id=111, commission=1.0)
        _insert_execution(st, "BAG-1", "T-DEFER", con_id=BAG_CON_ID, commission=0.0)
        _insert_execution(st, "BAG-2", "T-DEFER", con_id=BAG_CON_ID, commission=0.0)
        o = _orch(st)

        result = o._true_up_commission("T-DEFER", -100.0)

        assert result == -100.0, "must defer, not true up against an incomplete leg ledger"
        assert "T-DEFER" in st.pending_trueup_trade_ids()


class TestUpsertNeverRegressesAKnownCommission:
    def test_a_later_upsert_with_no_report_does_not_erase_a_known_commission(self, tmp_path):
        st = _state(tmp_path)
        st.record_execution(exec_id="E1", order_id=1, perm_id=1, trade_id="T-X",
                            symbol="QQQ", con_id=111, side="SLD", shares=1,
                            price=1.0, exec_time=now_et().isoformat(),
                            commission=1.05)
        assert st.total_commission("T-X") == pytest.approx(1.05)

        # A later sweep pass re-sees the SAME fill (same exec_id) without a
        # fresh commissionReport attached -- must not regress the already-
        # known commission back to "not yet reported".
        st.record_execution(exec_id="E1", order_id=1, perm_id=1, trade_id="T-X",
                            symbol="QQQ", con_id=111, side="SLD", shares=1,
                            price=1.0, exec_time=now_et().isoformat(),
                            commission=None)

        assert st.total_commission("T-X") == pytest.approx(1.05), (
            "a later upsert with commission=None regressed a known value")
        assert st.count_executions_with_commission("T-X") == 1

    def test_record_execution_default_means_not_yet_reported(self, tmp_path):
        st = _state(tmp_path)
        st.record_execution(exec_id="E2", order_id=1, perm_id=1, trade_id="T-Y",
                            symbol="QQQ", con_id=111, side="SLD", shares=1,
                            price=1.0, exec_time=now_et().isoformat())

        assert st.count_executions("T-Y") == 1, "the row itself must still be recorded"
        assert st.count_executions_with_commission("T-Y") == 0, (
            "an unspecified commission must read as NOT YET REPORTED, not a real zero")


class TestSettleDeferredTrueupsKeepsPendingIfIncomplete:
    """_settle_deferred_trueups (Path B) previously had no completeness
    check at all and deleted its marker unconditionally."""

    def test_marker_stays_pending_when_still_incomplete_at_post_market(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-PARTIAL", legs=4, realized_pnl=-100.0)
        st.set_state("trueup_pending_T-PARTIAL", date.today().isoformat())
        for i in range(4):  # entry side only
            _insert_execution(st, f"L{i}", "T-PARTIAL", con_id=111, commission=1.0)
        o = _orch(st)

        o._settle_deferred_trueups()

        assert "T-PARTIAL" in st.pending_trueup_trade_ids(), (
            "an incomplete ledger must not clear the pending marker -- the "
            "old unconditional delete lost the retry chance permanently")
        with sqlite3.connect(st._db_path) as conn:
            pnl = conn.execute(
                "SELECT realized_pnl FROM trades WHERE trade_id='T-PARTIAL'"
            ).fetchone()[0]
        assert pnl == pytest.approx(-100.0), "must not settle against a partial ledger"

    def test_settles_and_stamps_commission_once_the_ledger_is_complete(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-COMPLETE", legs=4, realized_pnl=-100.0)
        st.set_state("trueup_pending_T-COMPLETE", date.today().isoformat())
        st.set_state("commission_est_T-COMPLETE", "5.20")
        for i in range(8):
            _insert_execution(st, f"L{i}", "T-COMPLETE", con_id=111, commission=1.02)
        o = _orch(st)

        o._settle_deferred_trueups()

        assert "T-COMPLETE" not in st.pending_trueup_trade_ids()
        with sqlite3.connect(st._db_path) as conn:
            pnl, comm = conn.execute(
                "SELECT realized_pnl, commission FROM trades "
                "WHERE trade_id='T-COMPLETE'").fetchone()
        # real = 8 * 1.02 = 8.16; estimate was 5.20; delta = 5.20 - 8.16 = -2.96
        assert pnl == pytest.approx(-102.96, abs=0.01)
        assert comm == pytest.approx(8.16, abs=0.01), (
            "W48: the post-market path must stamp trades.commission too, "
            "not just the inline path")

    def test_orphaned_marker_with_no_trade_row_is_cleared_not_retried_forever(self, tmp_path):
        st = _state(tmp_path)
        st.set_state("trueup_pending_T-GHOST", date.today().isoformat())
        o = _orch(st)

        o._settle_deferred_trueups()

        assert "T-GHOST" not in st.pending_trueup_trade_ids()

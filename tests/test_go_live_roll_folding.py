"""go_live.py's rolled-chain folding: fold_roll_chains (pure) and its two
real callers, compute_go_live_verdict/real_close_count, driven end to end
against a REAL temp-DB StateManager (same convention test_w3_scorecard_
truth.py/test_roll_booking.py use — direct SQL INSERT into `trades` for
fixture rows, real code for everything under test).

A rolled chain (A closed via execution/roll_booking.py, exit_reason_detailed
containing "rolled_to_<B>"; B carries rolled_from_trade_id=A) must count as
ONE outcome in the verdict sample, credited at the chain's terminal (real)
close -- never dropped, never double-counted.

No inspect.getsource.
"""
from __future__ import annotations

import sqlite3

import pytest

from ait.bot.state import StateManager
from ait.reporting.go_live import fold_roll_chains, compute_go_live_verdict, real_close_count


def _row(trade_id, strategy="iron_condor", entry_time="2026-08-01T10:00:00",
         exit_time="2026-08-03T15:00:00", realized_pnl=100.0, car=1000.0,
         reason="", rolled_from_trade_id=None):
    return {
        "trade_id": trade_id, "strategy": strategy, "entry_time": entry_time,
        "exit_time": exit_time, "realized_pnl": realized_pnl, "car": car,
        "reason": reason, "rolled_from_trade_id": rolled_from_trade_id,
    }


# ---------------------------------------------------------------------------
# fold_roll_chains -- pure
# ---------------------------------------------------------------------------

class TestFoldRollChains:
    def test_non_rolled_row_passes_through_unchanged(self):
        row = _row("T-1", realized_pnl=250.0)
        folded, removed = fold_roll_chains([row])
        assert removed == 0
        assert len(folded) == 1
        assert folded[0]["realized_pnl"] == 250.0
        assert folded[0]["trade_id"] == "T-1"

    def test_simple_chain_folds_to_one_outcome(self):
        a = _row("T-A", entry_time="2026-08-01T10:00:00", exit_time="2026-08-03T15:00:00",
                 realized_pnl=-30.0, reason="rolled_to_T-B")
        b = _row("T-B", entry_time="2026-08-03T15:05:00", exit_time="2026-08-10T15:00:00",
                 realized_pnl=180.0, reason="take_profit_short (P&L: 40.0%)",
                 rolled_from_trade_id="T-A")
        folded, removed = fold_roll_chains([a, b])
        assert removed == 1  # A, the non-terminal leg
        assert len(folded) == 1
        merged = folded[0]
        assert merged["trade_id"] == "T-B"  # terminal's own identity
        assert merged["realized_pnl"] == pytest.approx(-30.0 + 180.0)
        assert merged["entry_time"] == "2026-08-01T10:00:00"  # A's (origin)
        assert merged["exit_time"] == "2026-08-10T15:00:00"  # B's (terminal)

    def test_three_hop_chain_folds_correctly(self):
        a = _row("T-A", entry_time="2026-08-01T10:00:00", realized_pnl=-10.0,
                 reason="rolled_to_T-B")
        b = _row("T-B", entry_time="2026-08-03T10:00:00", realized_pnl=-20.0,
                 reason="rolled_to_T-C", rolled_from_trade_id="T-A")
        c = _row("T-C", entry_time="2026-08-06T10:00:00", exit_time="2026-08-15T15:00:00",
                 realized_pnl=300.0, reason="take_profit_short", rolled_from_trade_id="T-B")
        folded, removed = fold_roll_chains([a, b, c])
        assert removed == 2  # A and B, both non-terminal
        assert len(folded) == 1
        merged = folded[0]
        assert merged["trade_id"] == "T-C"
        assert merged["realized_pnl"] == pytest.approx(-10.0 - 20.0 + 300.0)
        assert merged["entry_time"] == "2026-08-01T10:00:00"  # A's (chain origin)
        assert merged["exit_time"] == "2026-08-15T15:00:00"  # C's (terminal)

    def test_incomplete_chain_is_excluded_entirely(self):
        """A points to B, but B isn't in this batch (still open, or beyond
        the query window) -- the whole chain contributes zero rows, not a
        partial credit for A alone."""
        a = _row("T-A", realized_pnl=-30.0, reason="rolled_to_T-B")
        folded, removed = fold_roll_chains([a])
        assert folded == []
        assert removed == 1

    def test_missing_back_link_still_folds_correctly_no_double_count(self):
        """The direct proof this design is robust to the roll_booking.py
        gap it was built alongside: B's rolled_from_trade_id is None (as if
        the back-link write never landed), but A's forward link still names
        B -- the chain must still fold to ONE outcome, not count B a second
        time as its own standalone origin."""
        a = _row("T-A", entry_time="2026-08-01T10:00:00", realized_pnl=-30.0,
                 reason="rolled_to_T-B")
        b = _row("T-B", exit_time="2026-08-10T15:00:00", realized_pnl=180.0,
                 reason="take_profit_short", rolled_from_trade_id=None)  # missing back-link
        folded, removed = fold_roll_chains([a, b])
        assert removed == 1
        assert len(folded) == 1
        assert folded[0]["trade_id"] == "T-B"
        assert folded[0]["realized_pnl"] == pytest.approx(-30.0 + 180.0)

    def test_two_independent_chains_do_not_cross_contaminate(self):
        a1 = _row("T-A1", entry_time="2026-08-01T10:00:00", realized_pnl=-10.0,
                  reason="rolled_to_T-B1")
        b1 = _row("T-B1", exit_time="2026-08-05T15:00:00", realized_pnl=50.0,
                  reason="take_profit_short", rolled_from_trade_id="T-A1")
        a2 = _row("T-A2", entry_time="2026-08-02T10:00:00", realized_pnl=-20.0,
                  reason="rolled_to_T-B2")
        b2 = _row("T-B2", exit_time="2026-08-06T15:00:00", realized_pnl=90.0,
                  reason="take_profit_short", rolled_from_trade_id="T-A2")
        folded, removed = fold_roll_chains([a1, b1, a2, b2])
        assert removed == 2
        by_id = {r["trade_id"]: r for r in folded}
        assert by_id["T-B1"]["realized_pnl"] == pytest.approx(-10.0 + 50.0)
        assert by_id["T-B2"]["realized_pnl"] == pytest.approx(-20.0 + 90.0)


# ---------------------------------------------------------------------------
# compute_go_live_verdict / real_close_count -- end to end, real DB
# ---------------------------------------------------------------------------

def _insert_trade(con, trade_id, strategy, status, entry_time, exit_time,
                   realized_pnl, reason, car, entry_price, rolled_from_trade_id=None):
    con.execute(
        "INSERT INTO trades (trade_id, symbol, strategy, direction, "
        "status, entry_time, entry_price, quantity, contract_type, "
        "exit_time, realized_pnl, exit_reason_detailed, capital_at_risk, "
        "rolled_from_trade_id) VALUES (?,?,?,?,?,?,?,1,'iron_condor',?,?,?,?,?)",
        (trade_id, "SPY", strategy, "neutral", status, entry_time, entry_price,
         exit_time, realized_pnl, reason, car, rolled_from_trade_id),
    )


class TestVerdictAndCountAgreeOnFoldedRolls:
    def test_rolled_chain_counts_once_with_summed_pnl(self, tmp_path):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        with sqlite3.connect(st._db_path) as con:
            _insert_trade(con, "T-A", "iron_condor", "closed",
                          "2026-08-01T10:00:00", "2026-08-03T15:00:00",
                          -30.0, "rolled_to_T-B", 1000.0, 5.00)
            _insert_trade(con, "T-B", "iron_condor", "closed",
                          "2026-08-03T15:05:00", "2026-08-10T15:00:00",
                          180.0, "take_profit_short (P&L: 40.0%)", 1000.0, 4.50,
                          rolled_from_trade_id="T-A")
            # A normal, non-rolled close alongside it.
            _insert_trade(con, "T-C", "iron_condor", "closed",
                          "2026-08-04T10:00:00", "2026-08-06T15:00:00",
                          75.0, "take_profit_short (P&L: 35.0%)", 800.0, 3.00)

        verdict = compute_go_live_verdict(st._db_path)
        assert verdict.book.n == 2  # the chain once + T-C, never 3
        assert verdict.book.net == pytest.approx(-30.0 + 180.0 + 75.0)
        assert verdict.excluded.get("rolled_intermediate") == 1

        count = real_close_count(st._db_path)
        assert count == verdict.book.n == 2  # single-source-of-truth proof

    def test_in_flight_chain_contributes_nothing_yet(self, tmp_path):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        with sqlite3.connect(st._db_path) as con:
            _insert_trade(con, "T-A", "iron_condor", "closed",
                          "2026-08-01T10:00:00", "2026-08-03T15:00:00",
                          -30.0, "rolled_to_T-B", 1000.0, 5.00)
            # B deliberately NOT inserted -- still genuinely open.
            _insert_trade(con, "T-C", "iron_condor", "closed",
                          "2026-08-04T10:00:00", "2026-08-06T15:00:00",
                          75.0, "take_profit_short (P&L: 35.0%)", 800.0, 3.00)

        verdict = compute_go_live_verdict(st._db_path)
        assert verdict.book.n == 1  # only T-C; A's chain isn't decided yet
        assert verdict.book.net == pytest.approx(75.0)

        count = real_close_count(st._db_path)
        assert count == verdict.book.n == 1

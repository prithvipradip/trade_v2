"""W53 (roll-mechanics groundwork): the rolled_from_trade_id migration.

Same pattern as test_w32_booking_breaker_integrity.py's
test_an_existing_outbox_gains_the_new_columns -- build a pre-migration
table by hand, run StateManager, assert the column landed via
PRAGMA table_info. No inspect.getsource.
"""
from __future__ import annotations

import sqlite3

from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus


def _trade_columns(db_path) -> set[str]:
    with sqlite3.connect(db_path) as conn:
        return {r[1] for r in conn.execute("PRAGMA table_info(trades)")}


def test_an_existing_trades_table_gains_the_column(tmp_path):
    """A live DB created before this column existed -- the realistic case,
    since every current paper trade predates this wave."""
    db = tmp_path / "ait_state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE trades (
            trade_id TEXT PRIMARY KEY, symbol TEXT, strategy TEXT,
            direction TEXT, status TEXT, entry_time TEXT, entry_price REAL,
            quantity INTEGER, contract_type TEXT, strike REAL, expiry TEXT,
            exit_time TEXT, exit_price REAL, realized_pnl REAL,
            commission REAL, ml_confidence REAL, sentiment_score REAL,
            market_regime TEXT, notes TEXT, legs TEXT,
            exit_reason_detailed TEXT, peak_pnl_pct REAL,
            time_to_peak_hours REAL, direction_correct INTEGER
        )""")
    StateManager(db_path=db)
    assert "rolled_from_trade_id" in _trade_columns(db)


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "ait_state.db"
    StateManager(db_path=db)
    StateManager(db_path=db)  # second startup must not raise
    assert "rolled_from_trade_id" in _trade_columns(db)


def test_a_fresh_database_also_gets_the_column(tmp_path):
    """No pre-existing table at all -- the CREATE TABLE body doesn't carry
    this column (house convention: new columns are added by ALTER only,
    never the CREATE body), so a fresh DB must still pick it up from the
    migration tuple."""
    db = tmp_path / "ait_state.db"
    StateManager(db_path=db)
    assert "rolled_from_trade_id" in _trade_columns(db)


def test_trade_record_round_trips_the_new_field(tmp_path):
    """A row read back after the migration carries the field (None when
    unset) rather than being dropped as an unknown column."""
    db = tmp_path / "ait_state.db"
    state = StateManager(db_path=db)
    state.record_trade(TradeRecord(
        trade_id="T-W53", symbol="SPY", strategy="iron_condor",
        direction=TradeDirection.LONG, status=TradeStatus.PENDING,
        entry_time="2026-09-23T10:00:00", entry_price=1.0, quantity=1,
        contract_type="iron_condor",
    ))
    trade = state.get_trade_by_id("T-W53")
    assert trade is not None
    assert trade.rolled_from_trade_id is None

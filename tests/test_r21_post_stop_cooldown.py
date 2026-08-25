"""R12-B4 follow-up (2026-08-25): the post-stop re-entry cooldown query
matched only '%stop_loss%', but the short-strike touch stop — live's primary
loss exit since R12-B1 — writes 'short_strike_touch (spot ...)'. The live book
is the pre-fix failing case: QQQ touch-stopped 2026-08-24 09:47:33 (−$272.29)
and the bot re-entered QQQ at 10:00:40, 13 minutes later.

Every test EXECUTES the real query (extracted into
TradingOrchestrator._post_stop_cooldown_until) against a real sqlite file laid
out exactly like data/ait_state.db — no source-string assertions.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from ait.bot.orchestrator import TradingOrchestrator

LIVE_TOUCH_REASON = "short_strike_touch (spot 703.86 <= put 704.00)"


def _mk_db(tmp_path, reason: str, hours_ago: float = 1.0, symbol: str = "QQQ",
           status: str = "closed") -> None:
    (tmp_path / "data").mkdir(exist_ok=True)
    con = sqlite3.connect(tmp_path / "data" / "ait_state.db")
    con.execute("CREATE TABLE IF NOT EXISTS trades (symbol TEXT, status TEXT, "
                "exit_reason_detailed TEXT, exit_time TEXT)")
    con.execute(
        "INSERT INTO trades VALUES (?,?,?,?)",
        (symbol, status, reason,
         (datetime.now() - timedelta(hours=hours_ago)).isoformat()))
    con.commit()
    con.close()


def _orch() -> TradingOrchestrator:
    # The helper touches no orchestrator state besides the DB file.
    return TradingOrchestrator.__new__(TradingOrchestrator)


def test_touch_stop_blocks_reentry(tmp_path, monkeypatch):
    # THE 08-24 miss: pre-fix this returned None and QQQ re-entered in 13 min.
    _mk_db(tmp_path, LIVE_TOUCH_REASON)
    monkeypatch.chdir(tmp_path)
    assert _orch()._post_stop_cooldown_until("QQQ") is not None


def test_confirmed_tick_variant_blocks(tmp_path, monkeypatch):
    # Degraded-feed touches append ' [frozen quote, 2 agreeing ticks]'.
    _mk_db(tmp_path, "short_strike_touch (spot 700.10 >= call 700.00) "
                     "[frozen quote, 2 agreeing ticks]")
    monkeypatch.chdir(tmp_path)
    assert _orch()._post_stop_cooldown_until("QQQ") is not None


def test_stop_loss_still_blocks(tmp_path, monkeypatch):
    _mk_db(tmp_path, "stop_loss (P&L: -38.0%)")
    monkeypatch.chdir(tmp_path)
    assert _orch()._post_stop_cooldown_until("QQQ") is not None


def test_profit_exits_do_not_block(tmp_path, monkeypatch):
    _mk_db(tmp_path, "take_profit_short (P&L: 51.0%)")
    _mk_db(tmp_path, "trailing_stop (P&L: 12.0%, peak: 30.0%, stop: 10.0%)")
    _mk_db(tmp_path, "breakeven_stop (P&L: 0.5%, peak: 22.0%)")
    monkeypatch.chdir(tmp_path)
    assert _orch()._post_stop_cooldown_until("QQQ") is None


def test_window_expires_after_30h(tmp_path, monkeypatch):
    _mk_db(tmp_path, LIVE_TOUCH_REASON, hours_ago=31)
    monkeypatch.chdir(tmp_path)
    assert _orch()._post_stop_cooldown_until("QQQ") is None


def test_other_symbol_unaffected(tmp_path, monkeypatch):
    _mk_db(tmp_path, LIVE_TOUCH_REASON)
    monkeypatch.chdir(tmp_path)
    assert _orch()._post_stop_cooldown_until("SPY") is None


def test_open_trade_rows_ignored(tmp_path, monkeypatch):
    # A filled (open) row must not trigger the cooldown even with a stale
    # reason string left in the column.
    _mk_db(tmp_path, LIVE_TOUCH_REASON, status="filled")
    monkeypatch.chdir(tmp_path)
    assert _orch()._post_stop_cooldown_until("QQQ") is None

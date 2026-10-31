"""W106 review follow-up: an 8-K the bot already acted on is not acted on again after a restart.

Item 7 measures the 8-K window on the acceptance time over a rolling 24 hours, so a filing
accepted yesterday is still "recent" this morning. The monitor's record of filings already
handled lived only in memory: after an overnight restart the same 8-K flattened again,
closing a position opened after the event was handled. The handled accessions now persist
in the bot's state database and seed the monitor at start-up.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import ait.data.edgar_filings as ef
from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from tests.test_w106_edgar_window import _Session


def _sec_returns_one_8k(monkeypatch, hours_ago: float = 20.0):
    t = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    recent = {"form": ["8-K"], "filingDate": [t.strftime("%Y-%m-%d")],
              "acceptanceDateTime": [t.strftime("%Y-%m-%dT%H:%M:%S.000Z")],
              "accessionNumber": ["0000884394-26-000001"], "primaryDocument": ["d.htm"]}
    monkeypatch.setattr(ef.aiohttp, "ClientSession", lambda *a, **k: _Session({"filings": {"recent": recent}}))


def _sec_returns_nothing(monkeypatch):
    empty = {"form": [], "filingDate": [], "acceptanceDateTime": [], "accessionNumber": [], "primaryDocument": []}
    monkeypatch.setattr(ef.aiohttp, "ClientSession", lambda *a, **k: _Session({"filings": {"recent": empty}}))


def _orch(state: StateManager) -> TradingOrchestrator:
    orch = TradingOrchestrator.__new__(TradingOrchestrator)
    orch._state = state
    orch._edgar = orch._build_edgar_monitor(["SPY"])
    orch._send_notification = AsyncMock()
    orch._execute_exit = AsyncMock()
    orch._portfolio = MagicMock()
    orch._portfolio.check_positions = AsyncMock(return_value=[SimpleNamespace(symbol="SPY")])
    state.get_open_trades = MagicMock(return_value=[SimpleNamespace(symbol="SPY")])
    return orch


def test_a_restart_does_not_flatten_again_for_a_handled_8k(tmp_path, monkeypatch):
    _sec_returns_one_8k(monkeypatch)
    state = StateManager(tmp_path / "s.db")

    first = _orch(state)
    asyncio.run(first._check_material_events())
    assert first._execute_exit.await_count == 1                     # acted on once

    after_restart = _orch(state)                                    # a new process, same state database
    asyncio.run(after_restart._check_material_events())
    assert after_restart._execute_exit.await_count == 0


def test_a_new_8k_after_the_restart_still_flattens(tmp_path, monkeypatch):
    """Negative control: the persisted record holds only what was handled."""
    _sec_returns_nothing(monkeypatch)
    state = StateManager(tmp_path / "s.db")
    asyncio.run(_orch(state)._check_material_events())              # no filing yet: nothing handled
    _sec_returns_one_8k(monkeypatch)
    after_restart = _orch(state)
    asyncio.run(after_restart._check_material_events())
    assert after_restart._execute_exit.await_count == 1


def test_the_persisted_record_is_bounded(tmp_path):
    state = StateManager(tmp_path / "s.db")
    orch = _orch(state)
    for i in range(TradingOrchestrator.EDGAR_SEEN_KEEP + 50):
        orch._edgar.mark_seen(f"acc-{i}")
    orch._persist_edgar_seen()
    reloaded = _orch(state)._edgar.seen_accessions()
    assert len(reloaded) == TradingOrchestrator.EDGAR_SEEN_KEEP
    assert reloaded[-1] == f"acc-{TradingOrchestrator.EDGAR_SEEN_KEEP + 49}"   # the newest are kept

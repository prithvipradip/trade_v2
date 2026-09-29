"""Closes bot-day-09 (W47 registered, not fixed): orchestrator._shutdown()
now snapshots every open position's mark one last time before the process
exits, instead of relying solely on whatever the last regular ~30s
FAST_MONITOR_CADENCE_SECONDS pass happened to persist.

Deliberately reuses check_positions() -- the SAME call _monitor_positions_
fast already makes every cycle -- under the SAME _get_position_lock, rather
than inventing new persistence mechanics. Real StateManager where DB state
matters (never mocked). No inspect.getsource.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager, TradeDirection, TradeRecord, TradeStatus
from ait.config.settings import ExitConfig
from ait.data.market_data import Quote
from ait.execution.portfolio import PortfolioManager

CONDOR_LEGS = json.dumps([
    {"strike": 745.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 735.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 775.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 785.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])


def _orch(portfolio=None) -> TradingOrchestrator:
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._portfolio = portfolio or MagicMock()
    o._watchdog = MagicMock()
    o._watchdog.get_summary.return_value = "all green"
    sent: list[str] = []

    async def _notify(msg):
        sent.append(msg)
        return True
    o._notify = _notify
    o._sent = sent  # test-only convenience handle
    return o


class TestSnapshotRunsBeforeNotification:
    async def test_check_positions_awaited_before_the_shutdown_message(self):
        order: list[str] = []
        portfolio = MagicMock()

        async def _check_positions():
            order.append("snapshot")
            return []
        portfolio.check_positions = _check_positions
        o = _orch(portfolio)

        async def _notify(msg):
            order.append("notify")
            return True
        o._notify = _notify

        await o._shutdown()

        assert order == ["snapshot", "notify"]

    async def test_missing_portfolio_attribute_does_not_block_shutdown(self):
        """The established minimal-fixture shape (test_r16_round2_fixes.py's
        own test_shutdown_drains) never sets _portfolio at all -- the
        snapshot step's own try/except must swallow that AttributeError,
        not propagate it."""
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._watchdog = MagicMock()
        o._watchdog.get_summary.return_value = "all green"
        sent = []

        async def _notify(msg):
            sent.append(msg)
            return True
        o._notify = _notify

        await o._shutdown()

        assert sent and "shutting down" in sent[0]


class TestSnapshotFailureNeverBlocksShutdown:
    async def test_check_positions_raising_still_sends_notification(self):
        portfolio = MagicMock()

        async def _boom():
            raise RuntimeError("broker hiccup")
        portfolio.check_positions = _boom
        o = _orch(portfolio)

        await o._shutdown()

        assert o._sent and "shutting down" in o._sent[0]

    async def test_check_positions_hanging_past_timeout_still_shuts_down(self, monkeypatch):
        portfolio = MagicMock()

        async def _hang():
            await asyncio.sleep(3600)
        portfolio.check_positions = _hang
        o = _orch(portfolio)
        # Bound the wait so this test doesn't actually take the real 10s.
        monkeypatch.setattr(
            "ait.bot.orchestrator.SHUTDOWN_MARKS_SNAPSHOT_TIMEOUT_SECONDS", 0.05)

        await o._shutdown()

        assert o._sent and "shutting down" in o._sent[0]


class TestSnapshotNeverPlacesOrders:
    async def test_a_should_exit_true_verdict_triggers_no_exit_or_roll(self):
        """_snapshot_marks_on_shutdown only calls check_positions() itself --
        it must never dispatch on the result the way _trading_cycle does."""
        portfolio = MagicMock()

        async def _check_positions():
            return [SimpleNamespace(trade_id="T-1", should_exit=True,
                                    partial_exit_quantity=0, roll_side=None)]
        portfolio.check_positions = _check_positions
        o = _orch(portfolio)
        o._execute_exit = MagicMock(side_effect=AssertionError("must not be called"))
        o._execute_roll = MagicMock(side_effect=AssertionError("must not be called"))

        await o._shutdown()  # must not raise

        o._execute_exit.assert_not_called()
        o._execute_roll.assert_not_called()


# ---------------------------------------------------------------------------
# End to end: marks are actually persisted, real StateManager + PortfolioManager
# ---------------------------------------------------------------------------

def _real_manager(state: StateManager, spot: float) -> PortfolioManager:
    """Same proven __new__ harness test_w52_multileg_itm_diagnosis.py's own
    _manager uses, with a REAL StateManager swapped in so persistence is
    actually exercised end to end."""
    mgr = PortfolioManager.__new__(PortfolioManager)
    mgr._ibkr = MagicMock()
    mgr._ibkr.ib.portfolio.return_value = []
    mgr._state = state
    mgr._market_data = MagicMock()

    async def _price(symbol):
        return spot
    mgr._market_data.get_current_price = _price

    async def _quote(symbol):
        return Quote(symbol=symbol, bid=round(spot - 0.01, 2), ask=round(spot + 0.01, 2),
                    last=spot, volume=1_000_000, timestamp=datetime.now())
    mgr._market_data.get_quote = _quote

    mgr._exit_config = ExitConfig()
    mgr._earnings = None
    mgr._economic_cal = None

    async def _vol_mult(symbol):
        return 1.0
    mgr._get_volatility_stop_multiplier = _vol_mult
    mgr._pdt_guard = MagicMock()
    mgr._pdt_guard.would_be_day_trade.return_value = False
    mgr._option_position_unrealized = lambda trade, marks: 12.5
    mgr._get_position_delta = lambda trade: None
    return mgr


class TestMarksActuallyPersistedEndToEnd:
    async def test_shutdown_persists_a_fresh_mark(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")
        st = StateManager(db_path=tmp_path / "ait_state.db")
        trade = TradeRecord(
            trade_id="T-SHUTDOWN", symbol="SPY", strategy="iron_condor",
            direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
            entry_time="2026-09-27T10:00:00", entry_price=1.50, quantity=1,
            contract_type="iron_condor", legs=CONDOR_LEGS,
            expiry=(datetime.now() + timedelta(days=20)).date().isoformat(),
        )
        st.record_trade(trade)
        st.insert_open_position(
            trade_id=trade.trade_id, symbol="SPY", contract_type="iron_condor",
            quantity=1, entry_price=1.50, legs=CONDOR_LEGS)

        assert st.get_position_mark(trade.trade_id)["unrealized_pnl"] == 0.0  # insert-time default

        mgr = _real_manager(st, spot=760.0)
        o = _orch(mgr)

        await o._shutdown()

        mark = st.get_position_mark(trade.trade_id)
        assert mark is not None
        assert mark["unrealized_pnl"] == pytest.approx(12.5)

"""R13 credit-exit chain tests — the hole both R13 criticals shipped through.

No test had ever pushed a CREDIT position through the real
``_evaluate_position``. Two structural defects lived there undetected while
pytest was green:

- R13-CRIT-1: a local ``import os`` (macro-flatten block) made ``os``
  function-local, so the touch-stop env read raised UnboundLocalError on
  every credit tick — killing the whole exit monitor once any condor opened.
- R13-CRIT-2: the R12 touch-stop block was inserted as an ``if`` in the
  middle of the exit elif-chain, so for credit positions (touch enabled by
  default) take-profit / assignment / DTE / delta / earnings exits were
  structurally unreachable.

These tests drive the REAL ``_evaluate_position`` with an iron condor across
{touch on/off} x {TP, touch, DTE, healthy, marks-missing} and assert the
exit_reason per cell. Pre-fix, the touch-on cells ERROR (CRIT-1) and the
TP/DTE cells return no-exit (CRIT-2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.config.settings import ExitConfig
from ait.execution.portfolio import PortfolioManager

CONDOR_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])


@dataclass
class FakeTrade:
    entry_price: float = 1.00           # $1.00 credit collected
    quantity: int = 1
    contract_type: str = "spread"
    strategy: str = "iron_condor"
    symbol: str = "SPY"
    trade_id: str = "T-CREDIT"
    direction: object = None
    legs: str = CONDOR_LEGS
    expiry: str | None = None
    strike: float | None = None
    entry_time: str = "2026-07-01T10:00:00"


def _manager(spot: float, unrealized: float | None) -> PortfolioManager:
    """Real PortfolioManager via __new__ with only collaborators mocked —
    the same proven harness as TestMarksMissingSafetyExits."""
    mgr = PortfolioManager.__new__(PortfolioManager)
    mgr._ibkr = MagicMock()
    mgr._ibkr.ib.portfolio.return_value = []
    mgr._state = MagicMock()
    mgr._state.get_high_water_mark.return_value = 0.0
    mgr._market_data = MagicMock()

    async def _price(symbol):
        return spot
    mgr._market_data.get_current_price = _price
    mgr._exit_config = ExitConfig()
    mgr._earnings = None
    mgr._economic_cal = None

    async def _vol_mult(symbol):
        return 1.0
    mgr._get_volatility_stop_multiplier = _vol_mult
    mgr._pdt_guard = MagicMock()
    mgr._pdt_guard.would_be_day_trade.return_value = False
    # Steer P&L directly: None models marks-missing, a float is dollars.
    mgr._option_position_unrealized = lambda trade, marks: unrealized
    mgr._get_position_delta = lambda trade: None
    return mgr


def _condor(dte: int) -> FakeTrade:
    t = FakeTrade(expiry=(date.today() + timedelta(days=dte)).isoformat())
    t.direction = SimpleNamespace(value="neutral")
    return t


# --------------------------------------------------------------------------
# Touch stop ON (the live default) — the configuration both criticals hit
# --------------------------------------------------------------------------

async def test_take_profit_fires_with_touch_enabled(monkeypatch):
    """CRIT-1 (crashes pre-fix) + CRIT-2 (no-exit pre-fix): a condor at +55%
    of credit, spot inside the shorts, must take profit."""
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
    monkeypatch.delenv("AIT_SKIP_MACRO_EVENTS", raising=False)
    mgr = _manager(spot=100.0, unrealized=55.0)  # +55% of $100 cost basis
    status = await mgr._evaluate_position(_condor(dte=20))
    assert status is not None
    assert status.should_exit
    assert "take_profit_short" in status.exit_reason


async def test_short_strike_touch_fires(monkeypatch):
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
    mgr = _manager(spot=97.5, unrealized=-20.0)  # spot through the 98 put
    status = await mgr._evaluate_position(_condor(dte=20))
    assert status.should_exit
    assert "short_strike_touch" in status.exit_reason


async def test_dte_safety_exit_fires_with_touch_enabled(monkeypatch):
    """CRIT-2: DTE<=5 forced close was unreachable for condors."""
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
    mgr = _manager(spot=100.0, unrealized=5.0)
    status = await mgr._evaluate_position(_condor(dte=3))
    assert status.should_exit
    assert "expiry_approaching" in status.exit_reason


async def test_healthy_condor_does_not_exit(monkeypatch):
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
    mgr = _manager(spot=100.0, unrealized=10.0)  # +10%, inside strikes
    status = await mgr._evaluate_position(_condor(dte=20))
    assert status is not None
    assert not status.should_exit


async def test_marks_missing_no_false_pnl_exit_but_dte_alive(monkeypatch):
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
    # Far DTE + no marks: nothing should fire (no false trips off pnl=0).
    mgr = _manager(spot=100.0, unrealized=None)
    status = await mgr._evaluate_position(_condor(dte=20))
    assert not status.should_exit
    # Near DTE + no marks: the safety exit must still fire.
    mgr = _manager(spot=100.0, unrealized=None)
    status = await mgr._evaluate_position(_condor(dte=3))
    assert status.should_exit
    assert "expiry_approaching" in status.exit_reason


async def test_touch_exit_wins_over_later_rules(monkeypatch):
    """When touch fires, the chain must not overwrite its exit_reason."""
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
    mgr = _manager(spot=97.5, unrealized=60.0)  # touch AND nominal TP level
    status = await mgr._evaluate_position(_condor(dte=3))  # AND DTE<=5
    assert status.should_exit
    assert "short_strike_touch" in status.exit_reason


# --------------------------------------------------------------------------
# Touch stop OFF — the chain must behave identically to pre-R12
# --------------------------------------------------------------------------

async def test_take_profit_fires_with_touch_disabled(monkeypatch):
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")
    mgr = _manager(spot=100.0, unrealized=55.0)
    status = await mgr._evaluate_position(_condor(dte=20))
    assert status.should_exit
    assert "take_profit_short" in status.exit_reason


async def test_touch_disabled_does_not_touch_close(monkeypatch):
    monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")
    mgr = _manager(spot=97.5, unrealized=-20.0)
    status = await mgr._evaluate_position(_condor(dte=20))
    assert not status.should_exit

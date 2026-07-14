"""Regression tests for EventStraddle signal generation (2026-06-18).

The strategy had NEVER fired in production: the orchestrator fetches chains
in config dte_range (14-45), but event_straddle demanded an expiry with
dte <= 14, so it found no chain and bailed silently — through its entire
FOMC window. It now takes the nearest available expiry and logs every bail.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest

from ait.data.options_chain import OptionContract, OptionsChain
from ait.strategies.base import SignalDirection
from ait.strategies.event_straddle import EventStraddle


def _contract(strike, right, dte, bid=2.0, ask=2.2):
    return OptionContract(
        symbol="SPY", expiry=date.today() + timedelta(days=dte), strike=strike,
        right=right, bid=bid, ask=ask, last=(bid + ask) / 2, volume=500,
        open_interest=1000, implied_vol=0.2,
    )


def _chain(dte, strikes=(595, 600, 605), price=600.0):
    exp = date.today() + timedelta(days=dte)
    return OptionsChain(
        symbol="SPY", underlying_price=price, expiry=exp,
        calls=[_contract(s, "C", dte) for s in strikes],
        puts=[_contract(s, "P", dte) for s in strikes],
    )


def _cal(days):
    c = MagicMock()
    c.days_until_next_event.return_value = days
    return c


class TestEventStraddleFires:
    def setup_method(self):
        self.es = EventStraddle()

    def _gen(self, chains, days, iv_rank=40.0, symbol="SPY"):
        return self.es.generate_signals(
            symbol=symbol, chains=chains, market_direction=SignalDirection.NEUTRAL,
            confidence=0.65, iv_rank=iv_rank, economic_cal=_cal(days),
        )

    def test_fires_on_21dte_chain_within_orchestrator_window(self):
        # The exact production scenario that silently failed before.
        sigs = self._gen([_chain(21)], days=0)
        assert len(sigs) == 1
        assert sigs[0].strategy_name == "event_straddle"

    def test_fires_one_day_before_event(self):
        assert len(self._gen([_chain(21)], days=1)) == 1

    def test_picks_nearest_expiry_when_multiple(self):
        # 14, 21, 35 DTE available -> should use the 14 (nearest)
        sigs = self._gen([_chain(35), _chain(14), _chain(21)], days=0)
        assert len(sigs) == 1
        dte = (sigs[0].legs[0]["expiry"] if isinstance(sigs[0].legs[0], dict)
               else None)
        # nearest expiry = 14 days out
        assert sigs[0].contract.expiry == date.today() + timedelta(days=14)

    def test_no_fire_outside_event_window(self):
        assert self._gen([_chain(21)], days=5) == []

    def test_no_fire_on_non_preferred_symbol(self):
        assert self._gen([_chain(21)], days=0, symbol="AMD") == []

    def test_skips_when_iv_already_high(self):
        assert self._gen([_chain(21)], days=0, iv_rank=80.0) == []

    def test_no_calendar_no_fire(self):
        sigs = self.es.generate_signals(
            symbol="SPY", chains=[_chain(21)], market_direction=SignalDirection.NEUTRAL,
            confidence=0.65, iv_rank=40.0, economic_cal=None,
        )
        assert sigs == []

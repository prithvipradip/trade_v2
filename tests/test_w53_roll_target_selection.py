"""W53 (roll-mechanics groundwork): select_roll_target / _at_risk_vertical
(roll_selector.py).

Real OptionContract/OptionsChain objects with a fake chain service standing
in for OptionsChainService -- same convention test_hot_path_smoke.py's
FakeOptionsChainService uses (real domain objects, not MagicMock).

Every test drives the real select_roll_target / _at_risk_vertical.
No inspect.getsource.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json

import pytest

from ait.config.settings import AssignmentResponseConfig
from ait.data.options_chain import OptionContract, OptionsChain
from ait.execution.roll_selector import _at_risk_vertical, select_roll_target

SYMBOL = "SPY"
SPOT = 100.0

# The at-risk vertical these tests roll FROM: short put 98 / long put 95
# (width 3), short call 102 / long call 105 (width 3).
OLD_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-09-24"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-09-24"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-09-24"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-09-24"},
])


@dataclass
class _FakeTrade:
    trade_id: str = "T-W53"
    symbol: str = SYMBOL
    legs: str = OLD_LEGS


def _contract(right: str, strike: float, expiry: date, delta: float,
              *, volume=2_500, open_interest=9_000, bid=1.45, ask=1.55) -> OptionContract:
    return OptionContract(
        symbol=SYMBOL, expiry=expiry, strike=strike, right=right,
        bid=bid, ask=ask, last=round((bid + ask) / 2, 2),
        volume=volume, open_interest=open_interest, implied_vol=0.18,
        delta=delta, con_id=int(strike * 10) + (1 if right == "C" else 2),
        source="yahoo_delayed",
    )


def _target_chain(expiry: date) -> OptionsChain:
    """A later-expiry chain: one strike per side sits at the 0.20 delta
    roll_selector targets; wings sit further out on both sides."""
    puts = [
        _contract("P", 90.0, expiry, delta=-0.05),
        _contract("P", 95.0, expiry, delta=-0.20),   # target new short
        _contract("P", 100.0, expiry, delta=-0.50),
    ]
    calls = [
        _contract("C", 100.0, expiry, delta=0.50),
        _contract("C", 105.0, expiry, delta=0.20),   # target new short
        _contract("C", 110.0, expiry, delta=0.05),
    ]
    return OptionsChain(symbol=SYMBOL, underlying_price=SPOT, expiry=expiry,
                        calls=calls, puts=puts, source="yahoo_delayed")


class _FakeChainService:
    def __init__(self, chains):
        self._chains = chains
        self.calls: list[tuple] = []

    async def get_chain(self, symbol, min_dte=None, max_dte=None):
        self.calls.append((symbol, min_dte, max_dte))
        return self._chains


CONFIG = AssignmentResponseConfig()


class TestSelectRollTarget:
    async def test_picks_the_furthest_expiry_in_the_window(self):
        near = _target_chain(date.today() + timedelta(days=16))
        far = _target_chain(date.today() + timedelta(days=28))
        svc = _FakeChainService([near, far])
        result = await select_roll_target(_FakeTrade(), "P", svc, CONFIG)
        assert result is not None
        assert result.expiry == str(far.expiry)

    async def test_new_short_lands_at_the_target_delta(self):
        chain = _target_chain(date.today() + timedelta(days=21))
        svc = _FakeChainService([chain])
        result = await select_roll_target(_FakeTrade(), "P", svc, CONFIG)
        assert result is not None
        assert result.new_short.strike == 95.0
        assert result.new_short.delta == pytest.approx(-0.20)

    async def test_new_long_preserves_the_original_width(self):
        # Old vertical width = |98 - 95| = 3. New short lands at 95 (put
        # side), so the target long is 95 - 3 = 92 -- closest available is 90.
        chain = _target_chain(date.today() + timedelta(days=21))
        svc = _FakeChainService([chain])
        result = await select_roll_target(_FakeTrade(), "P", svc, CONFIG)
        assert result is not None
        assert result.new_long.strike == 90.0

    async def test_call_side_works_symmetrically(self):
        chain = _target_chain(date.today() + timedelta(days=21))
        svc = _FakeChainService([chain])
        result = await select_roll_target(_FakeTrade(), "C", svc, CONFIG)
        assert result is not None
        assert result.new_short.strike == 105.0
        assert result.new_long.strike == 110.0

    async def test_empty_chain_list_returns_none_not_raise(self):
        svc = _FakeChainService([])
        result = await select_roll_target(_FakeTrade(), "P", svc, CONFIG)
        assert result is None

    async def test_chain_fetch_exception_returns_none_not_raise(self):
        class _Boom:
            async def get_chain(self, *a, **kw):
                raise ConnectionError("gateway down")
        result = await select_roll_target(_FakeTrade(), "P", _Boom(), CONFIG)
        assert result is None

    async def test_no_liquid_contracts_on_the_target_side_returns_none(self):
        chain = _target_chain(date.today() + timedelta(days=21))
        illiquid_puts = [
            OptionContract(symbol=SYMBOL, expiry=c.expiry, strike=c.strike, right=c.right,
                          bid=0.0, ask=0.0, last=0.0, volume=0, open_interest=0,
                          implied_vol=0.18, delta=c.delta, con_id=c.con_id)
            for c in chain.puts
        ]
        chain = OptionsChain(symbol=SYMBOL, underlying_price=SPOT, expiry=chain.expiry,
                             calls=chain.calls, puts=illiquid_puts, source="yahoo_delayed")
        svc = _FakeChainService([chain])
        result = await select_roll_target(_FakeTrade(), "P", svc, CONFIG)
        assert result is None

    async def test_strike_outside_delta_tolerance_returns_none(self):
        expiry = date.today() + timedelta(days=21)
        puts = [_contract("P", 90.0, expiry, delta=-0.50)]
        calls = [_contract("C", 110.0, expiry, delta=0.50)]
        chain = OptionsChain(symbol=SYMBOL, underlying_price=SPOT, expiry=expiry,
                             calls=calls, puts=puts, source="yahoo_delayed")
        svc = _FakeChainService([chain])
        result = await select_roll_target(_FakeTrade(), "P", svc, CONFIG)
        assert result is None

    async def test_no_clean_at_risk_vertical_returns_none(self):
        trade = _FakeTrade(legs="not json")
        chain = _target_chain(date.today() + timedelta(days=21))
        svc = _FakeChainService([chain])
        result = await select_roll_target(trade, "P", svc, CONFIG)
        assert result is None


class TestAtRiskVertical:
    def test_reads_the_short_and_long_strike_for_a_side(self):
        assert _at_risk_vertical(_FakeTrade(), "P") == (98.0, 95.0)
        assert _at_risk_vertical(_FakeTrade(), "C") == (102.0, 105.0)

    def test_malformed_legs_returns_none(self):
        trade = _FakeTrade(legs="not json")
        assert _at_risk_vertical(trade, "P") is None

    def test_two_shorts_no_long_is_not_a_clean_vertical(self):
        legs = json.dumps([
            {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-09-24"},
            {"strike": 96.0, "right": "P", "action": "SELL", "expiry": "2026-09-24"},
        ])
        trade = _FakeTrade(legs=legs)
        assert _at_risk_vertical(trade, "P") is None

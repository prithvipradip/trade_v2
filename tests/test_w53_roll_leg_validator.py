"""W53 (roll-mechanics groundwork): _validate_roll_legs (executor.py).

Unlike _validate_combo_legs (a count-only check against a per-strategy-name
table -- tested nowhere today, per this wave's own research), a roll must be
checked against the trade's REAL open legs: the same reversed-leg hazard the
R5 audit built _validate_combo_legs to catch (a single reversed leg turns
defined-risk into undefined-risk) applies just as directly here. Pure
function, no broker/IBKR involved -- runs before qualification.

Every test drives the real _validate_roll_legs. No inspect.getsource.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ait.execution.executor import _validate_roll_legs

CONDOR_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-10-17"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-10-17"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-10-17"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-10-17"},
])


@dataclass
class _FakeTrade:
    trade_id: str = "T-W53"
    legs: str = CONDOR_LEGS


def _leg(strike: float, right: str, action: str, expiry: str = "2026-11-21") -> dict:
    return {"strike": strike, "right": right, "action": action, "expiry": expiry}


# A valid put-vertical roll: buy back the 98/95 put spread, sell a new 90/85
# put spread further out in time.
VALID_CLOSING = [_leg(98.0, "P", "BUY", "2026-10-17"), _leg(95.0, "P", "SELL", "2026-10-17")]
VALID_OPENING = [_leg(90.0, "P", "SELL"), _leg(85.0, "P", "BUY")]


class TestValidRoll:
    def test_a_clean_put_vertical_roll_is_accepted(self):
        assert _validate_roll_legs(_FakeTrade(), VALID_CLOSING, VALID_OPENING) is None

    def test_a_clean_call_vertical_roll_is_accepted(self):
        closing = [_leg(102.0, "C", "BUY", "2026-10-17"),
                   _leg(105.0, "C", "SELL", "2026-10-17")]
        opening = [_leg(108.0, "C", "SELL"), _leg(112.0, "C", "BUY")]
        assert _validate_roll_legs(_FakeTrade(), closing, opening) is None


class TestLegCount:
    def test_wrong_closing_count_is_rejected(self):
        err = _validate_roll_legs(_FakeTrade(), VALID_CLOSING[:1], VALID_OPENING)
        assert err is not None and "2 closing" in err

    def test_wrong_opening_count_is_rejected(self):
        err = _validate_roll_legs(_FakeTrade(), VALID_CLOSING, VALID_OPENING + [_leg(80.0, "P", "BUY")])
        assert err is not None and "opening" in err


class TestClosingMustMatchTheRealPosition:
    def test_closing_a_strike_not_actually_open_is_rejected(self):
        closing = [_leg(99.0, "P", "BUY", "2026-10-17"), _leg(95.0, "P", "SELL", "2026-10-17")]
        err = _validate_roll_legs(_FakeTrade(), closing, VALID_OPENING)
        assert err is not None and "does not match an open leg" in err

    def test_the_reversed_leg_hazard_is_rejected(self):
        """The exact R5 failure mode: instead of BUYING back the short put
        (closing it), this SELLS more of it -- turning a defined-risk close
        into an undefined-risk add. Action must be reversed from the open."""
        closing = [_leg(98.0, "P", "SELL", "2026-10-17"),  # should be BUY
                   _leg(95.0, "P", "SELL", "2026-10-17")]
        err = _validate_roll_legs(_FakeTrade(), closing, VALID_OPENING)
        assert err is not None and "does not match an open leg" in err


class TestOpeningMustBeGenuinelyNew:
    def test_opening_a_strike_already_in_the_position_is_rejected(self):
        opening = [_leg(95.0, "P", "SELL"), _leg(85.0, "P", "BUY")]  # 95 is the old long
        err = _validate_roll_legs(_FakeTrade(), VALID_CLOSING, opening)
        assert err is not None and "duplicates an already-open leg" in err


class TestSideConsistency:
    def test_closing_put_and_opening_call_is_rejected(self):
        # Strikes chosen clear of CONDOR_LEGS's own 102/105 call legs, so
        # this isolates the side-mismatch check from the duplicate-leg one.
        opening = [_leg(120.0, "C", "SELL"), _leg(125.0, "C", "BUY")]
        err = _validate_roll_legs(_FakeTrade(), VALID_CLOSING, opening)
        assert err is not None and "must open a" in err

    def test_closing_legs_that_mix_put_and_call_is_rejected(self):
        closing = [_leg(98.0, "P", "BUY", "2026-10-17"), _leg(102.0, "C", "BUY", "2026-10-17")]
        err = _validate_roll_legs(_FakeTrade(), closing, VALID_OPENING)
        assert err is not None and "mix put and call" in err


class TestOpeningPairMustStayDefinedRisk:
    def test_opening_two_shorts_instead_of_a_short_plus_long_is_rejected(self):
        opening = [_leg(90.0, "P", "SELL"), _leg(85.0, "P", "SELL")]
        err = _validate_roll_legs(_FakeTrade(), VALID_CLOSING, opening)
        assert err is not None and "one SELL" in err

    def test_opening_two_longs_is_rejected(self):
        opening = [_leg(90.0, "P", "BUY"), _leg(85.0, "P", "BUY")]
        err = _validate_roll_legs(_FakeTrade(), VALID_CLOSING, opening)
        assert err is not None and "one SELL" in err


class TestMalformedTradeLegs:
    def test_unparseable_trade_legs_is_rejected_not_raised(self):
        trade = _FakeTrade(legs="not json")
        err = _validate_roll_legs(trade, VALID_CLOSING, VALID_OPENING)
        assert err is not None and "not valid JSON" in err

    def test_empty_trade_legs_is_rejected(self):
        trade = _FakeTrade(legs="[]")
        err = _validate_roll_legs(trade, VALID_CLOSING, VALID_OPENING)
        assert err is not None and "does not match an open leg" in err

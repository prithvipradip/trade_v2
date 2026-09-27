"""W55 (synthetic per-leg delta for the monitor loop):
execution/synthetic_delta.py.

Real TradeRecord + legs-JSON fixtures (same convention
test_w53_roll_target_selection.py uses), a plain dict for option_marks
(never MagicMock -- CLAUDE.md's own rule; matches how test_w51_ex_div_
assignment.py already feeds marks). Every test drives the real
synthetic_leg_greeks/synthetic_vertical_deltas/_expiry_key.

No inspect.getsource.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

pytest.importorskip("py_vollib")
from py_vollib.black_scholes import black_scholes  # noqa: E402

from ait.bot.state import TradeDirection, TradeRecord, TradeStatus  # noqa: E402
from ait.execution.synthetic_delta import (  # noqa: E402
    ContractGreeks,
    _expiry_key,
    synthetic_leg_greeks,
    synthetic_vertical_deltas,
)
from ait.utils.time import today_et  # noqa: E402

SYMBOL = "SPY"
SPOT = 500.0
SIGMA = 0.22
EXPIRY = today_et() + timedelta(days=10)
EXPIRY_STR = str(EXPIRY)
DTE_YEARS = 10 / 365.0

# The at-risk vertical these tests read: short put 480 / long put 470
# (width 10), short call 520 / long call 530.
CONDOR_LEGS = json.dumps([
    {"strike": 470.0, "right": "P", "action": "BUY", "expiry": EXPIRY_STR},
    {"strike": 480.0, "right": "P", "action": "SELL", "expiry": EXPIRY_STR},
    {"strike": 520.0, "right": "C", "action": "SELL", "expiry": EXPIRY_STR},
    {"strike": 530.0, "right": "C", "action": "BUY", "expiry": EXPIRY_STR},
])


def _trade(legs: str = CONDOR_LEGS) -> TradeRecord:
    return TradeRecord(
        trade_id="T-W55", symbol=SYMBOL, strategy="iron_condor",
        direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
        entry_time="2026-09-27T10:00:00", entry_price=1.50, quantity=1,
        contract_type="iron_condor", legs=legs)


def _price(strike: float, right: str, sigma: float = SIGMA) -> float:
    flag = "c" if right == "C" else "p"
    return round(black_scholes(flag, SPOT, strike, DTE_YEARS, 0.05, sigma), 4)


def _marks(*entries: tuple[float, str, float]) -> dict[tuple, float]:
    """entries: (strike, right, price). Builds the same key shape
    _get_option_marks() does."""
    key = _expiry_key(EXPIRY_STR)
    return {(SYMBOL, key, strike, right): price for strike, right, price in entries}


class TestExpiryKeyMatchesGetOptionMarks:
    """Class guard: portfolio.py's _get_option_marks()/_option_position_
    unrealized() both normalize an expiry with
    `str(x).replace("-", "")[:8]` (portfolio.py:322,370) -- a new caller
    that doesn't replicate this exactly silently misses every lookup."""

    @pytest.mark.parametrize("raw,expected", [
        ("2026-10-02", "20261002"),
        ("20261002", "20261002"),
        ("2026-01-05", "20260105"),
    ])
    def test_matches_the_real_inline_normalization(self, raw, expected):
        real = str(raw).replace("-", "")[:8]
        assert _expiry_key(raw) == real == expected


class TestSyntheticLegGreeks:
    def test_healthy_mark_solves_a_real_delta(self):
        marks = _marks((480.0, "P", _price(480.0, "P")))
        result = synthetic_leg_greeks(marks, SYMBOL, EXPIRY_STR, 480.0, "P", SPOT)
        assert isinstance(result, ContractGreeks)
        assert result.iv == pytest.approx(SIGMA, abs=0.01)
        assert -1.0 <= result.delta <= 0.0   # put delta

    def test_call_delta_is_bounded_zero_to_one(self):
        marks = _marks((520.0, "C", _price(520.0, "C")))
        result = synthetic_leg_greeks(marks, SYMBOL, EXPIRY_STR, 520.0, "C", SPOT)
        assert result is not None
        assert 0.0 <= result.delta <= 1.0

    def test_missing_mark_returns_none(self):
        marks: dict = {}
        assert synthetic_leg_greeks(marks, SYMBOL, EXPIRY_STR, 480.0, "P", SPOT) is None

    def test_zero_mark_returns_none(self):
        """_get_option_marks() itself does NOT filter zero/negative prices --
        this is the one guard it lacks that the new code must apply."""
        marks = _marks((480.0, "P", 0.0))
        assert synthetic_leg_greeks(marks, SYMBOL, EXPIRY_STR, 480.0, "P", SPOT) is None

    def test_negative_mark_returns_none(self):
        marks = _marks((480.0, "P", -1.0))
        assert synthetic_leg_greeks(marks, SYMBOL, EXPIRY_STR, 480.0, "P", SPOT) is None

    def test_malformed_expiry_returns_none_not_raise(self):
        marks = {(SYMBOL, "notadate", 480.0, "P"): 5.0}
        result = synthetic_leg_greeks(marks, SYMBOL, "not-a-real-date", 480.0, "P", SPOT)
        assert result is None

    def test_a_below_intrinsic_mark_returns_none_not_a_wrong_delta(self):
        """Same arbitrage-failure protection solve_iv_from_price already
        proves -- a single leg has no chain median to fall back to, so this
        must be None, never a guessed/wrong delta."""
        marks = _marks((400.0, "C", 0.01))   # deep ITM call priced at noise
        assert synthetic_leg_greeks(marks, SYMBOL, EXPIRY_STR, 400.0, "C", SPOT) is None


class TestSyntheticVerticalDeltas:
    def test_solves_both_short_and_long_on_the_put_side(self):
        marks = _marks(
            (480.0, "P", _price(480.0, "P")),
            (470.0, "P", _price(470.0, "P")),
        )
        result = synthetic_vertical_deltas(_trade(), "P", marks, SPOT)
        assert result["short"] is not None and result["long"] is not None
        # the short (closer to spot) has a larger-magnitude delta than the
        # long (further away) for a put credit vertical
        assert abs(result["short"].delta) > abs(result["long"].delta)

    def test_solves_the_call_side_independently(self):
        marks = _marks(
            (520.0, "C", _price(520.0, "C")),
            (530.0, "C", _price(530.0, "C")),
        )
        result = synthetic_vertical_deltas(_trade(), "C", marks, SPOT)
        assert result["short"] is not None and result["long"] is not None
        assert result["short"].delta > result["long"].delta

    def test_only_the_requested_side_is_solved_condor_has_both(self):
        """A 4-leg condor trade, marks present for ALL four legs -- asking
        for the put side must not touch/require the call legs' marks."""
        marks = _marks((480.0, "P", _price(480.0, "P")),
                       (470.0, "P", _price(470.0, "P")))
        result = synthetic_vertical_deltas(_trade(), "P", marks, SPOT)
        assert result["short"] is not None and result["long"] is not None

    def test_missing_marks_for_the_side_give_none_not_raise(self):
        result = synthetic_vertical_deltas(_trade(), "P", {}, SPOT)
        assert result == {"short": None, "long": None}

    def test_malformed_legs_json_returns_none_pair_not_raise(self):
        trade = _trade(legs="not json")
        result = synthetic_vertical_deltas(trade, "P", {}, SPOT)
        assert result == {"short": None, "long": None}

    def test_a_single_leg_trade_not_a_clean_vertical_returns_none_pair(self):
        legs = json.dumps([{"strike": 480.0, "right": "P", "action": "SELL",
                            "expiry": EXPIRY_STR}])
        trade = _trade(legs=legs)
        marks = _marks((480.0, "P", _price(480.0, "P")))
        result = synthetic_vertical_deltas(trade, "P", marks, SPOT)
        assert result == {"short": None, "long": None}

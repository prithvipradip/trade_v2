"""Roll-combo construction mechanics -- execution/roll_executor.py. Nothing
in src/ait calls into this module; these tests drive it directly.

Mirrors test_w53_probe_script.py's TestLegNetPrice cases exactly (leg_net_
price is lifted near-verbatim) and reuses the same real-TradeRecord/legs-
JSON fixture convention test_w53_roll_target_selection.py and test_w56_
condor_defense.py use. A fake IB stands in for the broker boundary --
never MagicMock for option marks/quotes, per this codebase's own house
rule.

No inspect.getsource.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from ait.bot.state import TradeDirection, TradeRecord, TradeStatus
from ait.data.options_chain import OptionContract
from ait.execution.roll_executor import (
    PreparedRollCombo,
    leg_net_price,
    legs_for_roll,
    prepare_roll_combo,
    qualify_roll_legs,
)
from ait.execution.roll_selector import RollTarget

SYMBOL = "SPY"


@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    """Keep poll_leg_quotes' rounds*delay pacing out of test wall-time --
    same pattern test_r16_data_fixes.py uses for options_chain.py's own
    reqMktData polling: replace the MODULE-LOCAL asyncio name binding (not
    the real global asyncio module, which other concurrently-running async
    machinery may still depend on)."""
    import asyncio as _asyncio
    from types import SimpleNamespace

    import ait.execution.roll_executor as re_mod

    real_sleep = _asyncio.sleep

    async def _instant(_secs):
        await real_sleep(0)

    monkeypatch.setattr(
        re_mod, "asyncio",
        SimpleNamespace(sleep=_instant, get_running_loop=_asyncio.get_running_loop),
    )

CONDOR_LEGS = json.dumps([
    {"strike": 470.0, "right": "P", "action": "BUY", "expiry": "2026-10-05"},
    {"strike": 480.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"},
    {"strike": 520.0, "right": "C", "action": "SELL", "expiry": "2026-10-05"},
    {"strike": 530.0, "right": "C", "action": "BUY", "expiry": "2026-10-05"},
])


def _trade(legs: str = CONDOR_LEGS) -> TradeRecord:
    return TradeRecord(
        trade_id="T-ROLL", symbol=SYMBOL, strategy="iron_condor",
        direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
        entry_time="2026-09-27T10:00:00", entry_price=1.50, quantity=1,
        contract_type="iron_condor", legs=legs)


def _roll_target(new_short=460.0, new_long=450.0, expiry="2026-10-23") -> RollTarget:
    return RollTarget(
        new_short=OptionContract(symbol=SYMBOL, expiry=expiry, strike=new_short, right="P",
                                 bid=0, ask=0, last=0, volume=0, open_interest=0,
                                 implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        new_long=OptionContract(symbol=SYMBOL, expiry=expiry, strike=new_long, right="P",
                                bid=0, ask=0, last=0, volume=0, open_interest=0,
                                implied_vol=0, delta=0, gamma=0, theta=0, vega=0),
        expiry=expiry,
    )


# ---------------------------------------------------------------------------
# legs_for_roll
# ---------------------------------------------------------------------------

class TestLegsForRoll:
    def test_reverses_the_closing_side_and_builds_the_opening_side(self):
        result = legs_for_roll(_trade(), "P", _roll_target())
        assert result is not None
        closing, opening, _carried = result
        assert {(leg["strike"], leg["action"]) for leg in closing} == {
            (480.0, "BUY"), (470.0, "SELL"),
        }
        assert {(leg["strike"], leg["action"]) for leg in opening} == {
            (460.0, "SELL"), (450.0, "BUY"),
        }
        assert all(leg["expiry"] == "2026-10-23" for leg in opening)

    def test_call_side_untouched_when_rolling_the_put_side(self):
        closing, _opening, _carried = legs_for_roll(_trade(), "P", _roll_target())
        assert all(leg["right"] == "P" for leg in closing)

    def test_carried_legs_are_the_other_side_with_actions_unchanged(self):
        """W57: rolling the put side of a 4-leg IC must hand back the call
        vertical's own two legs, actions UNCHANGED (still the real open
        position) -- these are what a caller attaches to the new trade so
        the call vertical isn't orphaned at the broker."""
        _closing, _opening, carried = legs_for_roll(_trade(), "P", _roll_target())
        assert {(leg["strike"], leg["right"], leg["action"]) for leg in carried} == {
            (520.0, "C", "SELL"), (530.0, "C", "BUY"),
        }

    def test_carried_legs_empty_for_a_plain_vertical(self):
        legs = json.dumps([
            {"strike": 470.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"},
            {"strike": 460.0, "right": "P", "action": "BUY", "expiry": "2026-10-05"},
        ])
        _closing, _opening, carried = legs_for_roll(_trade(legs), "P", _roll_target())
        assert carried == []

    def test_not_a_clean_vertical_returns_none(self):
        legs = json.dumps([
            {"strike": 470.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"},
            {"strike": 460.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"},
        ])
        assert legs_for_roll(_trade(legs), "P", _roll_target()) is None

    def test_malformed_legs_json_returns_none(self):
        assert legs_for_roll(_trade(legs="not json"), "P", _roll_target()) is None


# ---------------------------------------------------------------------------
# leg_net_price -- mirrors test_w53_probe_script.py::TestLegNetPrice exactly
# ---------------------------------------------------------------------------

class TestLegNetPrice:
    def test_credit_combo_is_signed_negative(self):
        legs = [{"conId": 1, "action": "SELL"}, {"conId": 2, "action": "BUY"}]
        quotes = {1: (1.00, 1.10), 2: (0.20, 0.30)}
        mid = leg_net_price(legs, quotes, "mid")
        assert mid is not None and mid < 0
        assert mid == pytest.approx(-1.05 + 0.25)

    def test_debit_combo_is_signed_positive(self):
        legs = [{"conId": 1, "action": "BUY"}, {"conId": 2, "action": "SELL"}]
        quotes = {1: (1.00, 1.10), 2: (0.20, 0.30)}
        mid = leg_net_price(legs, quotes, "mid")
        assert mid is not None and mid > 0

    def test_natural_side_is_buy_at_ask_sell_at_bid(self):
        legs = [{"conId": 1, "action": "SELL"}, {"conId": 2, "action": "BUY"}]
        quotes = {1: (1.00, 1.10), 2: (0.20, 0.30)}
        natural = leg_net_price(legs, quotes, "natural")
        assert natural == pytest.approx(-1.00 + 0.30)

    def test_missing_quote_returns_none(self):
        legs = [{"conId": 1, "action": "SELL"}, {"conId": 2, "action": "BUY"}]
        quotes = {1: (1.00, 1.10), 2: (None, None)}
        assert leg_net_price(legs, quotes, "mid") is None
        assert leg_net_price(legs, quotes, "natural") is None


# ---------------------------------------------------------------------------
# qualify_roll_legs / prepare_roll_combo -- fake IB boundary
# ---------------------------------------------------------------------------

@dataclass
class _FakeQualified:
    """Stands in for a qualified ib_insync Option: conId set, strike/right
    carried through for the fake ticker lookup below."""
    conId: int
    strike: float
    right: str


@dataclass
class _FakeTicker:
    bid: float | None
    ask: float | None


class _FakeIB:
    """qualify_contracts_batch matches IBKRClient's own real contract
    (list in, list out, None per failed leg, never a shorter list -- see
    ibkr_client.py:539-566's own R13 note on why). fail_strikes marks
    which strikes should fail qualification; quotes_by_conid feeds ticker()."""

    def __init__(self, fail_strikes: set[float] | None = None,
                 quotes_by_strike: dict[float, tuple[float | None, float | None]] | None = None):
        self._next_conid = 1000
        self._fail_strikes = fail_strikes or set()
        self._quotes_by_strike = quotes_by_strike or {}
        self._conid_to_strike: dict[int, float] = {}
        self.qualify_calls: list[list] = []

    async def qualify_contracts_batch(self, contracts):
        self.qualify_calls.append(list(contracts))
        out = []
        for c in contracts:
            if c.strike in self._fail_strikes:
                out.append(None)
                continue
            self._next_conid += 1
            conid = self._next_conid
            self._conid_to_strike[conid] = c.strike
            out.append(_FakeQualified(conId=conid, strike=c.strike, right=c.right))
        return out

    @property
    def ib(self):
        return self

    def reqMktData(self, *args):
        pass

    def cancelMktData(self, *args):
        pass

    def ticker(self, contract):
        strike = getattr(contract, "strike", None)
        bid, ask = self._quotes_by_strike.get(strike, (None, None))
        return _FakeTicker(bid=bid, ask=ask)


DEFAULT_QUOTES = {
    480.0: (1.00, 1.10), 470.0: (0.20, 0.30),   # closing put vertical
    460.0: (0.90, 1.00), 450.0: (0.15, 0.25),   # opening put vertical
    520.0: (0.85, 0.95), 530.0: (0.10, 0.20),   # carried call vertical (untouched)
}


class TestQualifyRollLegs:
    async def test_happy_path_enriches_every_leg(self):
        fake = _FakeIB()
        legs = [{"strike": 480.0, "right": "P", "action": "BUY", "expiry": "2026-10-05"},
                {"strike": 470.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"}]
        out = await qualify_roll_legs(fake, SYMBOL, legs)
        assert out is not None
        assert all("conId" in leg and "_contract" in leg for leg in out)

    async def test_one_leg_failing_fails_the_whole_batch(self):
        fake = _FakeIB(fail_strikes={470.0})
        legs = [{"strike": 480.0, "right": "P", "action": "BUY", "expiry": "2026-10-05"},
                {"strike": 470.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"}]
        assert await qualify_roll_legs(fake, SYMBOL, legs) is None


class TestPrepareRollCombo:
    async def test_happy_path_returns_a_prepared_combo_never_placed(self):
        fake = _FakeIB(quotes_by_strike=DEFAULT_QUOTES)
        result = await prepare_roll_combo(
            fake, _trade(), "P", _roll_target(), SYMBOL, quantity=1, order_ref="TEST-ROLL",
        )
        assert isinstance(result, PreparedRollCombo)
        assert len(result.closing_legs) == 2
        assert len(result.opening_legs) == 2
        assert result.order.orderRef == "TEST-ROLL"
        assert result.mid_net == pytest.approx(
            leg_net_price(result.closing_legs + result.opening_legs,
                          {leg["conId"]: DEFAULT_QUOTES[leg["strike"]]
                           for leg in result.closing_legs + result.opening_legs},
                          "mid"))

    async def test_place_order_is_never_even_attempted(self):
        """_FakeIB never implements place_order at all -- an accidental
        call inside prepare_roll_combo would raise AttributeError, not
        silently no-op. Its absence alone is the regression guard: this
        test passing at all proves prepare_roll_combo never calls it."""
        fake = _FakeIB(quotes_by_strike=DEFAULT_QUOTES)
        assert not hasattr(fake, "place_order")
        result = await prepare_roll_combo(
            fake, _trade(), "P", _roll_target(), SYMBOL, quantity=1, order_ref="TEST-ROLL",
        )
        assert isinstance(result, PreparedRollCombo)

    async def test_not_a_clean_vertical_refused_before_any_qualification(self):
        """legs_for_roll's own guard (not a clean short+long pair) refuses
        before spending an API call."""
        legs = json.dumps([
            {"strike": 470.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"},
            {"strike": 460.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"},
        ])
        fake = _FakeIB(quotes_by_strike=DEFAULT_QUOTES)
        result = await prepare_roll_combo(
            fake, _trade(legs), "P", _roll_target(), SYMBOL, quantity=1, order_ref="TEST-ROLL",
        )
        assert result is None
        assert fake.qualify_calls == []

    async def test_validate_roll_legs_refusal_is_respected_before_qualification(self, monkeypatch):
        """legs_for_roll is correct-by-construction (it faithfully reverses
        the trade's own real legs), so it can never itself hand
        _validate_roll_legs a malformed pair in practice -- isolate that
        SEPARATE gate directly by stubbing legs_for_roll's output, proving
        prepare_roll_combo genuinely calls _validate_roll_legs and refuses
        on ITS verdict too, not just legs_for_roll's own guard above."""
        import ait.execution.roll_executor as re_mod

        # A non-vertical opening pair (both SELL, no BUY) -- exactly what
        # _validate_roll_legs's own "one SELL, one BUY" check refuses.
        bad_opening = [
            {"strike": 460.0, "right": "P", "action": "SELL", "expiry": "2026-10-23"},
            {"strike": 450.0, "right": "P", "action": "SELL", "expiry": "2026-10-23"},
        ]
        good_closing = [
            {"strike": 480.0, "right": "P", "action": "BUY", "expiry": "2026-10-05"},
            {"strike": 470.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"},
        ]
        monkeypatch.setattr(re_mod, "legs_for_roll",
                            lambda trade, side, target: (good_closing, bad_opening, []))

        fake = _FakeIB(quotes_by_strike=DEFAULT_QUOTES)
        result = await prepare_roll_combo(
            fake, _trade(), "P", _roll_target(), SYMBOL, quantity=1, order_ref="TEST-ROLL",
        )
        assert result is None
        assert fake.qualify_calls == []

    async def test_missing_quote_refuses(self):
        fake = _FakeIB(quotes_by_strike={480.0: (1.00, 1.10)})  # missing the other 3
        result = await prepare_roll_combo(
            fake, _trade(), "P", _roll_target(), SYMBOL, quantity=1, order_ref="TEST-ROLL",
        )
        assert result is None

    async def test_over_debit_cap_refuses(self):
        fake = _FakeIB(quotes_by_strike=DEFAULT_QUOTES)
        result = await prepare_roll_combo(
            fake, _trade(), "P", _roll_target(), SYMBOL, quantity=1, order_ref="TEST-ROLL",
            debit_cap=0.01,
        )
        assert result is None

    async def test_carried_vertical_is_qualified_and_marked_but_never_in_the_bag(self):
        """W57: a 4-leg IC rolling its put side must qualify+quote the
        untouched call vertical too, but the BAG order must still trade
        only the 4 rolled-vertical legs (2 closing + 2 opening), never the
        carried ones -- the carried side stays a separate, untouched
        position at the broker."""
        fake = _FakeIB(quotes_by_strike=DEFAULT_QUOTES)
        result = await prepare_roll_combo(
            fake, _trade(), "P", _roll_target(), SYMBOL, quantity=1, order_ref="TEST-ROLL",
        )
        assert isinstance(result, PreparedRollCombo)
        assert {leg["strike"] for leg in result.carried_legs} == {520.0, 530.0}
        assert all("conId" in leg for leg in result.carried_legs)
        bag_con_ids = {leg["conId"] for leg in result.closing_legs + result.opening_legs}
        carried_con_ids = {leg["conId"] for leg in result.carried_legs}
        assert bag_con_ids.isdisjoint(carried_con_ids)
        # position-sign mid: short 520C SELL -0.90, long 530C BUY +0.15
        assert result.carried_value == pytest.approx(-0.90 + 0.15)

    async def test_carried_vertical_missing_quote_refuses(self):
        quotes = dict(DEFAULT_QUOTES)
        del quotes[520.0]
        fake = _FakeIB(quotes_by_strike=quotes)
        result = await prepare_roll_combo(
            fake, _trade(), "P", _roll_target(), SYMBOL, quantity=1, order_ref="TEST-ROLL",
        )
        assert result is None

    async def test_no_carried_legs_for_a_plain_vertical_roll(self):
        legs = json.dumps([
            {"strike": 480.0, "right": "P", "action": "SELL", "expiry": "2026-10-05"},
            {"strike": 470.0, "right": "P", "action": "BUY", "expiry": "2026-10-05"},
        ])
        fake = _FakeIB(quotes_by_strike=DEFAULT_QUOTES)
        result = await prepare_roll_combo(
            fake, _trade(legs), "P", _roll_target(), SYMBOL, quantity=1, order_ref="TEST-ROLL",
        )
        assert isinstance(result, PreparedRollCombo)
        assert result.carried_legs == []
        assert result.carried_value == 0.0

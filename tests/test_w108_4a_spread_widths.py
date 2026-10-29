"""W108 phase 4a: budget-fitted widths for the spreads, ONE rule for live and research.

Before this wave only the iron condor fitted its width to the per-trade budget (``IronCondor.affordable_width``):
the live ``bull_call_spread`` / ``bear_put_spread`` builders never read ``risk_budget`` (a 0.45/0.25-delta spread on a
$500 underlying is ~13 wide with a ~$5 debit, twice a $250 budget, so gate 6b would refuse every one) and research
built the credit-spread wing at ``wing_k x expected move x 0.5`` (~$19 here) with no cap, so the lot rule refused it.

``ait.risk.width_sizing`` now holds the arithmetic for both kinds of max loss:

* credit (condor, credit spreads): max loss = (width - credit) x 100; width <= budget / (100 x max(0.5, 1 - ratio)),
  ratio = the credit/width floor the builder enforces (the condor's 0.10; 0 for research's credit spreads);
* debit (bull call / bear put): max loss = the DEBIT x 100, so the fit is priced: the widest width whose debit fits.

Every test here executes the builders or the shared functions (no source-text assertions). The parity class prices a
live chain with the research engine's own model and checks both sides pick the same width.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from ait.backtesting.pricing import OptionType, black_scholes_price, bs_delta
from ait.config import runtime_env
from ait.config.invariants import IC_BUDGET_CAPPED_MIN_WING
from ait.data.options_chain import OptionContract, OptionsChain
from ait.risk.width_sizing import (
    affordable_width,
    budget_for_width,
    fitted_credit_width,
    fitted_debit_width,
    whole_dollar_widths,
)
from ait.strategies.base import SignalDirection
from ait.strategies.iron_condor import IronCondor
from ait.strategies.spreads import BearPutSpread, BullCallSpread
from tests.test_w74_engine_realism import _bt

S, IV, DTE, R = 500.0, 0.20, 21, 0.05
T = DTE / 365.0
TODAY = date(2024, 3, 4)
BUDGET = 250.0               # the shipped 5% of the $5,000 account
CAPITAL, PCT = 5_000.0, 0.05


# --------------------------------------------------------------------------- #
# The shared rule
# --------------------------------------------------------------------------- #

class TestTheSharedRule:
    @pytest.mark.parametrize("budget, ratio, width", [
        (250.0, 0.10, 250.0 / 90.0),     # the condor at the shipped budget: 2.78
        (250.0, 0.0, 2.5),               # a credit spread with no credit floor: budget / 100
        (1_000.0, 0.9, 20.0),            # a mis-set floor never more than doubles the width (share >= 0.5)
        (90.0, 0.10, 1.0),
    ])
    def test_affordable_width_and_its_inverse(self, budget, ratio, width):
        assert affordable_width(budget, min_credit_ratio=ratio) == pytest.approx(width)
        assert budget_for_width(width, min_credit_ratio=ratio) == pytest.approx(budget)

    def test_the_research_grid_is_whole_dollars_from_the_budget_capped_minimum(self):
        assert whole_dollar_widths(7.3) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert whole_dollar_widths(3.0) == [1.0, 2.0, 3.0]
        assert whole_dollar_widths(0.6) == []
        assert IC_BUDGET_CAPPED_MIN_WING == 1.0

    @pytest.mark.parametrize("natural, budget, expected", [
        (19.2, None, 19.2),                  # no budget: unchanged (the paper big-account convention)
        (19.2, 0.0, 19.2),
        (19.2, 5_000.0, 19.2),               # the budget carries it: unchanged (a cap only narrows)
        (19.2, 250.0, 2.0),                  # cap 2.5 -> the widest listed width AT OR BELOW it
        (19.2, 5_000.0 * 0.05, 2.0),         # 250.00000000000003: float noise does not move it
        (19.2, 300.0, 3.0),                  # cap exactly 3.0 -> 3, not 2
        (19.2, 60.0, 1.0),                   # cap 0.6 < every listed width -> the narrowest ($1); the lot rule decides
        (2.4, 250.0, 2.4),                   # natural under the cap: unchanged, not snapped
    ])
    def test_fitted_credit_width(self, natural, budget, expected):
        assert fitted_credit_width(natural, budget, whole_dollar_widths(natural)) == pytest.approx(expected)

    def test_fitted_credit_width_takes_the_builders_own_floor(self):
        # cap 250 / 90 = 2.78: the floor-0.10 builder may use 2 on a $1 grid, a $0.5 grid gives 2.5
        assert fitted_credit_width(10.0, 250.0, [0.5 * k for k in range(1, 21)], min_credit_ratio=0.10) == 2.5

    DEBITS = {1.0: 0.48, 2.0: 0.94, 3.0: 1.38, 4.0: 1.80, 5.0: 2.20, 6.0: 2.58, 7.0: 2.93}

    @pytest.mark.parametrize("natural, budget, expected", [
        (7.0, None, 7.0),        # no budget: unchanged
        (7.0, 300.0, 7.0),       # the natural debit ($293) fits: unchanged
        (7.0, 250.0, 5.0),       # $258 at 6-wide does not fit, $220 at 5-wide does
        (7.0, 220.0, 5.0),       # exactly the budget fits
        (7.0, 100.0, 2.0),
        (7.0, 40.0, None),       # not even the 1-wide ($48) fits: refused
    ])
    def test_fitted_debit_width(self, natural, budget, expected):
        assert fitted_debit_width(natural, budget, self.DEBITS) == expected

    def test_a_non_positive_narrower_debit_is_never_chosen(self):
        assert fitted_debit_width(3.0, 100.0, {1.0: 0.0, 2.0: -0.1, 3.0: 1.5}) is None

    def test_a_fitting_natural_spread_is_returned_even_with_a_bad_debit(self):
        """The existing `net_debit <= 0` gate refuses it, as before; the fit does not hunt for another."""
        assert fitted_debit_width(3.0, 100.0, {1.0: 0.4, 3.0: 0.0}) == 3.0


class TestTheCondorIsANoOp:
    """IronCondor.affordable_width / budget_for_width delegate to the shared rule with the condor's own floor."""

    @pytest.mark.parametrize("ratio", ["0.10", "0.20", "0.6"])
    @pytest.mark.parametrize("budget", [50.0, 150.0, 250.0, 300.0, 1_000.0, 4_500.0])
    def test_same_numbers_as_the_w31_formula(self, monkeypatch, ratio, budget):
        monkeypatch.setenv("AIT_IC_MIN_CREDIT_WIDTH", ratio)
        r = float(ratio)
        w31 = budget / (100.0 * max(0.5, 1.0 - r))        # the pre-4a body of IronCondor.affordable_width
        assert IronCondor.affordable_width(budget) == w31
        assert IronCondor.affordable_width(budget) == affordable_width(budget, min_credit_ratio=r)
        assert IronCondor.budget_for_width(w31) == pytest.approx(budget)
        assert IronCondor.budget_for_width(3.0) == 3.0 * 100.0 * max(0.5, 1.0 - r)

    def test_the_live_condor_wing_at_the_shipped_budget(self, monkeypatch):
        monkeypatch.delenv("AIT_IC_MIN_CREDIT_WIDTH", raising=False)
        monkeypatch.setattr(runtime_env, "_config_yaml", lambda: {})       # the contract default (0.10)
        expiry = date.today() + timedelta(days=30)
        from types import SimpleNamespace
        leg = SimpleNamespace(implied_vol=0.20)
        ic = IronCondor()
        ic.risk_budget = BUDGET
        assert ic._vol_scaled_width(S, leg, leg, SimpleNamespace(expiry=expiry.isoformat(), atm_iv=0.2)) == \
            pytest.approx(250.0 / 90.0)


# --------------------------------------------------------------------------- #
# A $1-grid chain priced by the research engine's own model
# --------------------------------------------------------------------------- #

def _engine(**kw):
    base = dict(budget_aware_wings=True, position_size_pct=PCT, initial_capital=CAPITAL,
                delta_long=0.45, delta_short=0.25, slippage_pct=0.0)
    base.update(kw)
    return _bt(**base)


def _contract(bt, right: str, k: float, expiry: date) -> OptionContract:
    opt = OptionType.CALL if right == "C" else OptionType.PUT
    mid = black_scholes_price(S, k, T, R, bt._get_leg_iv(IV, k, S, opt), opt)
    return OptionContract(symbol="SIM", expiry=expiry, strike=k, right=right,
                          bid=mid * 0.995, ask=mid * 1.005, last=mid, volume=1_000, open_interest=5_000,
                          implied_vol=IV, delta=bs_delta(S, k, T, R, IV, opt), source="ibkr")


def _chain(bt) -> OptionsChain:
    expiry = date.today() + timedelta(days=DTE)
    strikes = [float(k) for k in range(int(S) - 30, int(S) + 31)]
    return OptionsChain(symbol="SIM", underlying_price=S, expiry=expiry,
                        calls=[_contract(bt, "C", k, expiry) for k in strikes],
                        puts=[_contract(bt, "P", k, expiry) for k in strikes], source="ibkr")


def _live(cls, chain, budget):
    builder = cls()
    builder.risk_budget = budget
    direction = SignalDirection.BULLISH if cls is BullCallSpread else SignalDirection.BEARISH
    return builder.generate_signals("SIM", chain, direction, 0.7, 40.0)


def _width(signal) -> float:
    long_leg, short_leg = signal.legs[0]["contract"], signal.legs[1]["contract"]
    assert signal.legs[0]["action"] == "BUY" and signal.legs[1]["action"] == "SELL"
    return abs(short_leg.strike - long_leg.strike)


SPREADS = [(BullCallSpread, "bull_call_spread", SignalDirection.BULLISH),
           (BearPutSpread, "bear_put_spread", SignalDirection.BEARISH)]


@pytest.fixture(scope="module")
def priced():
    bt = _engine()
    return bt, _chain(bt)


class TestLiveDebitSpreads:
    @pytest.mark.parametrize("cls, _name, _dir", SPREADS)
    def test_no_budget_keeps_the_delta_placed_spread(self, priced, cls, _name, _dir):
        _bt_, chain = priced
        sig = _live(cls, chain, None)[0]
        assert _width(sig) >= 10                               # ~13 wide, debit ~$5: what live built before 4a
        assert sig.max_loss > BUDGET

    @pytest.mark.parametrize("cls, _name, _dir", SPREADS)
    def test_the_budget_narrows_it_to_the_widest_width_whose_debit_fits(self, priced, cls, _name, _dir):
        _bt_, chain = priced
        natural = _width(_live(cls, chain, None)[0])
        sig = _live(cls, chain, BUDGET)[0]
        w = _width(sig)
        assert 1 <= w < natural
        assert sig.max_loss <= BUDGET and sig.max_loss == pytest.approx(sig.entry_price * 100)
        # the next listed strike out (one dollar wider) would NOT fit: it is the widest
        long_leg = sig.legs[0]["contract"]
        out = 1.0 if cls is BullCallSpread else -1.0
        side = chain.calls if cls is BullCallSpread else chain.puts
        wider = next(c for c in side if c.strike == long_leg.strike + out * (w + 1))
        assert (long_leg.mid - wider.mid) * 100 > BUDGET
        # the long leg (the directional exposure) did not move
        assert long_leg.strike == _live(cls, chain, None)[0].legs[0]["contract"].strike

    @pytest.mark.parametrize("cls, _name, _dir", SPREADS)
    def test_a_spread_the_budget_already_carries_is_unchanged(self, priced, cls, _name, _dir):
        _bt_, chain = priced
        free = _live(cls, chain, None)[0]
        big = _live(cls, chain, 10_000.0)[0]
        assert _width(big) == _width(free) and big.entry_price == free.entry_price

    @pytest.mark.parametrize("cls, _name, _dir", SPREADS)
    def test_no_width_fits_no_signal(self, priced, cls, _name, _dir):
        _bt_, chain = priced
        assert _live(cls, chain, 20.0) == []                   # even the 1-wide costs ~$50


# --------------------------------------------------------------------------- #
# Research engine
# --------------------------------------------------------------------------- #

def _engine_debit(bt, strategy, direction):
    return bt._build_debit_position(strategy, direction, S, IV, T, R, DTE, TODAY, CAPITAL)


def _engine_credit(bt, strategy):
    return bt._build_credit_position(strategy, S, IV, T, R, DTE, TODAY, CAPITAL)


CREDIT_SPREADS = [("put_credit_spread", "short_put_strike", "long_put_strike"),
                  ("call_credit_spread", "short_call_strike", "long_call_strike")]


class TestResearchDebitSpreads:
    @pytest.mark.parametrize("_cls, name, direction", SPREADS)
    def test_budget_aware_narrows_to_the_widest_width_whose_debit_fits(self, _cls, name, direction):
        pos = _engine_debit(_engine(), name, direction)
        assert pos is not None, "the fitted spread must be built (before 4a the lot rule refused the 13-wide)"
        w = abs(pos["short_strike"] - pos["long_strike"])
        assert 1 <= w < 10
        assert pos["entry_price"] * 100 <= BUDGET and pos["contracts"] == 1

    @pytest.mark.parametrize("_cls, name, direction", SPREADS)
    def test_off_is_the_pre_4a_engine(self, _cls, name, direction):
        """budget_aware_wings off (a bare Backtester): the natural spread, refused by the lot rule at 5%."""
        assert _engine_debit(_engine(budget_aware_wings=False), name, direction) is None
        wide = _engine_debit(_engine(budget_aware_wings=False, position_size_pct=0.5), name, direction)
        assert abs(wide["short_strike"] - wide["long_strike"]) >= 10

    @pytest.mark.parametrize("_cls, name, direction", SPREADS)
    def test_the_fit_includes_research_slippage(self, _cls, name, direction):
        """Research fits on the debit _lots checks (x (1 + slippage_pct)), so a fitted spread always passes it."""
        plain = _engine_debit(_engine(), name, direction)
        slipped = _engine_debit(_engine(slippage_pct=0.25), name, direction)
        assert slipped is not None and slipped["entry_price"] * 100 <= BUDGET
        assert abs(slipped["short_strike"] - slipped["long_strike"]) < abs(plain["short_strike"] - plain["long_strike"])

    def test_no_width_fits_is_a_named_rejection(self):
        bt = _engine(position_size_pct=20.0 / CAPITAL)
        assert _engine_debit(bt, "bull_call_spread", SignalDirection.BULLISH) is None
        assert bt._funnel["builder:no_debit_width_fits_budget"] == 1


class TestResearchCreditSpreads:
    @pytest.mark.parametrize("name, short_key, long_key", CREDIT_SPREADS)
    def test_budget_aware_caps_the_wing_at_the_widest_dollar_under_budget_over_100(self, name, short_key, long_key):
        pos = _engine_credit(_engine(), name)
        assert pos is not None, "the fitted spread must be built (before 4a the ~19-wide was refused)"
        assert abs(pos[short_key] - pos[long_key]) == 2.0                 # floor(250 / 100)
        assert pos["max_loss_per_share"] * 100 <= BUDGET and pos["contracts"] == 1

    @pytest.mark.parametrize("name, short_key, long_key", CREDIT_SPREADS)
    def test_off_is_the_pre_4a_engine(self, name, short_key, long_key):
        assert _engine_credit(_engine(budget_aware_wings=False), name) is None   # refused by the lot rule
        wide = _engine_credit(_engine(budget_aware_wings=False, position_size_pct=0.5), name)
        assert abs(wide[short_key] - wide[long_key]) >= 18

    @pytest.mark.parametrize("name, short_key, long_key", CREDIT_SPREADS)
    def test_a_wing_the_budget_already_carries_is_unchanged(self, name, short_key, long_key):
        on = _engine_credit(_engine(position_size_pct=0.5), name)            # cap 25 > the ~19 wing
        off = _engine_credit(_engine(budget_aware_wings=False, position_size_pct=0.5), name)
        assert on == off


# --------------------------------------------------------------------------- #
# Parity: the live builder and the research engine pick the same width
# --------------------------------------------------------------------------- #

class TestLiveResearchParity:
    @pytest.mark.parametrize("budget_pct", [0.01, 0.03, 0.05, 0.08])
    @pytest.mark.parametrize("cls, name, direction", SPREADS)
    def test_same_chain_same_budget_same_width(self, cls, name, direction, budget_pct):
        bt = _engine(position_size_pct=budget_pct)
        chain = _chain(bt)
        budget = CAPITAL * budget_pct
        live = _live(cls, chain, budget)
        research = _engine_debit(bt, name, direction)
        assert (live == []) == (research is None)
        if research is None:
            return
        sig = live[0]
        # precondition: both sides placed the long leg on the same strike
        assert sig.legs[0]["contract"].strike == research["long_strike"]
        assert _width(sig) == abs(research["short_strike"] - research["long_strike"])
        assert sig.entry_price == pytest.approx(research["entry_price"], abs=1e-4)   # the engine stores 4 decimals

    @pytest.mark.parametrize("cls, name, direction", SPREADS)
    def test_the_budget_binds_in_this_scenario(self, cls, name, direction):
        """Guard on the parity test above: at 5% both sides really narrowed (not a no-op match)."""
        bt = _engine()
        free = _width(_live(cls, _chain(bt), None)[0])
        research = _engine_debit(bt, name, direction)
        assert abs(research["short_strike"] - research["long_strike"]) < free

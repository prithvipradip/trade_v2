"""W108 phase 4b: live builders for ``put_credit_spread`` (bullish) and ``call_credit_spread`` (bearish).

Until this wave the two verticals existed only in the research engine (``Backtester._build_credit_position``).
``strategies/credit_spreads.py`` builds them live with research's structure (the condor's short delta, a wing
of ``wing_k x expected move x backtest.spread_tuning.wing_expected_move_fraction``) and the live condor's
economics (the wing fitted to the per-trade budget through ``IronCondor.affordable_width``; the condor's two
credit floors, which the executor enforces on every credit entry anyway).

They are registered in the selector and NOT enabled: ``options.strategies`` ships ``[iron_condor]``, and the
selector tests below prove the shipped configuration never returns them. The exit tests push a spread the
real builder made through the real executor (fake broker, real state DB) and then through the real exit
monitor: no new exit machinery was built, so these confirm the existing paths handle a two-leg credit
structure (touch stop on its short strike, the shared take-profit ladder, the DTE close).

House rules: every test executes the real code path; no source-text assertions.
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

import ait.config.settings as settings_mod
from ait.backtesting.pricing import OptionType, black_scholes_price, bs_delta
from ait.config.invariants import IC_BUDGET_CAPPED_MIN_WING
from ait.config.runtime_env import contract_float
from ait.config.settings import DEFAULT_RISK_FREE_RATE, OptionsConfig, load_settings
from ait.data.options_chain import OptionContract, OptionsChain
from ait.execution import portfolio as portfolio_mod
from ait.execution.exit_policy import (
    EXPIRY_APPROACHING_DTE,
    macro_flatten_window_days,
    take_profit_targets,
)
from ait.risk.scalers import RANGE_GATED_STRATEGIES, sizing_confidence
from ait.strategies.base import (
    CREDIT_STRATEGIES,
    LIVE_CREDIT_STRATEGIES,
    UNDEFINED_RISK_STRATEGIES,
    SignalDirection,
)
from ait.strategies.credit_spreads import CallCreditSpread, PutCreditSpread
from ait.strategies.iron_condor import SHORT_DELTA_TARGET, IronCondor
from ait.strategies.selector import STRATEGY_MAP, StrategySelector
from ait.utils.time import today_et

from tests.test_w2_executor_risk_gates import _nbbo_for, env  # noqa: F401 — env is a fixture

R = DEFAULT_RISK_FREE_RATE
NAMES = ("put_credit_spread", "call_credit_spread")
SIDES = [(PutCreditSpread, SignalDirection.BULLISH, "P"), (CallCreditSpread, SignalDirection.BEARISH, "C")]


@pytest.fixture(autouse=True)
def pinned_contract(monkeypatch):
    """Pin every contract key the builders read, so these tests do not drift with config.yaml. The values are
    the shipped ones (backtest.wing_k / wing_floor_dollars / ic_* in config.yaml)."""
    for key, value in {"AIT_IC_WING_K": "1.6", "AIT_IC_WING_FLOOR": "2.0", "AIT_IC_MIN_CREDIT": "0.70",
                       "AIT_IC_MIN_CREDIT_WIDTH": "0.10", "AIT_IC_DELTA_MIN": "0.15",
                       "AIT_IC_DELTA_MAX": "0.30"}.items():
        monkeypatch.setenv(key, value)


def _bs_chain(S: float, iv: float, dte: int, *, step: float = 1.0, symbol: str = "SPY") -> OptionsChain:
    """A $-grid chain priced and delta'd by Black-Scholes at one flat IV (the engine's own pricer), so the
    live builder and the research builder see the same market. Two-sided 2-cent quotes, liquid."""
    expiry = today_et() + timedelta(days=dte)
    t = dte / 365.0
    calls, puts = [], []
    k = math.floor(S * 0.75 / step) * step
    while k <= S * 1.25:
        for right, typ, out in (("C", OptionType.CALL, calls), ("P", OptionType.PUT, puts)):
            px = black_scholes_price(S, k, t, R, iv, typ)
            out.append(OptionContract(
                symbol=symbol, expiry=expiry, strike=float(k), right=right,
                bid=round(px - 0.01, 4), ask=round(px + 0.01, 4), last=px, volume=1000,
                open_interest=1000, implied_vol=iv, delta=bs_delta(S, k, t, R, iv, typ)))
        k += step
    return OptionsChain(symbol=symbol, underlying_price=S, expiry=expiry, calls=calls, puts=puts)


def _build(cls, chain, direction, budget=None, **kw):
    strategy = cls(**kw)
    strategy.risk_budget = budget
    return strategy, strategy.generate_signals("SPY", chain, direction, 0.6, 40.0)


def _legs(sig):
    short = next(l["contract"] for l in sig.legs if l["action"] == "SELL")
    long_ = next(l["contract"] for l in sig.legs if l["action"] == "BUY")
    return short, long_


# --------------------------------------------------------------------------
# the builder
# --------------------------------------------------------------------------

class TestTheBuilder:
    CHAIN = (600.0, 0.18, 21)

    @pytest.mark.parametrize("cls, direction, right", SIDES)
    def test_builds_only_on_its_own_direction(self, cls, direction, right):
        chain = _bs_chain(*self.CHAIN)
        _, sigs = _build(cls, chain, direction, budget=600.0)
        assert [s.strategy_name for s in sigs] == [cls().name]
        for other in SignalDirection:
            if other is direction:
                continue
            strategy, none = _build(cls, chain, other, budget=600.0)
            assert none == [] and strategy.last_entry_reason == "direction_mismatch"

    @pytest.mark.parametrize("cls, direction, right", SIDES)
    @pytest.mark.parametrize("budget", [None, 600.0, 1000.0])
    def test_the_structure_is_a_defined_risk_credit_vertical(self, cls, direction, right, budget):
        chain = _bs_chain(*self.CHAIN)
        strategy, sigs = _build(cls, chain, direction, budget=budget)
        assert len(sigs) == 1 and strategy.last_entry_reason is None
        sig = sigs[0]
        short, long_ = _legs(sig)
        assert len(sig.legs) == 2 and short.right == long_.right == right
        S = chain.underlying_price
        if right == "P":
            assert long_.strike < short.strike < S
        else:
            assert S < short.strike < long_.strike
        # the short is the liquid contract nearest the condor's short delta
        side = chain.puts if right == "P" else chain.calls
        nearest = min(side, key=lambda c: abs(abs(c.delta) - SHORT_DELTA_TARGET))
        assert short.strike == nearest.strike
        width = abs(short.strike - long_.strike)
        credit = short.mid - long_.mid
        assert sig.entry_price == pytest.approx(credit)
        assert sig.max_loss == pytest.approx((width - credit) * 100)
        assert sig.max_profit == pytest.approx(credit * 100)
        assert sig.direction is direction and sig.contract is short and sig.expiry == chain.expiry
        assert sig.is_defined_risk
        assert sig.strategy_name in CREDIT_STRATEGIES and sig.strategy_name not in UNDEFINED_RISK_STRATEGIES

    @pytest.mark.parametrize("cls, direction, right", SIDES)
    @pytest.mark.parametrize("budget", [600.0, 800.0, 1000.0, 1500.0, 3000.0])
    def test_a_built_spread_never_exceeds_the_budget(self, cls, direction, right, budget):
        for case in [(600.0, 0.18, 21), (500.0, 0.22, 25), (220.0, 0.24, 21)]:
            _, sigs = _build(cls, _bs_chain(*case), direction, budget=budget)
            assert all(s.max_loss <= budget for s in sigs), (case, budget)

    def test_the_wing_is_the_widest_listed_one_the_budget_carries(self):
        """$600 budget: the fitted target is 6.67 wide and the nearest listed wing (7 wide) would cost $614.
        The builder takes the widest listed wing at or below the cap (6 wide, as research's fitted_credit_width
        does) instead of refusing a structure the budget can carry."""
        chain = _bs_chain(220.0, 0.24, 21)
        strategy, sigs = _build(CallCreditSpread, chain, SignalDirection.BEARISH, budget=600.0)
        assert len(sigs) == 1 and strategy.last_entry_reason is None
        short, long_ = _legs(sigs[0])
        assert abs(long_.strike - short.strike) == pytest.approx(6.0)
        assert sigs[0].max_loss <= 600.0

    def test_when_no_listed_wing_fits_the_budget_it_is_refused_like_the_condor(self):
        """A $5 strike grid and a $300 budget (cap 3.33): the narrowest listed wing is 5 wide and its max loss is
        above the budget. The condor's R16 guard refuses it; so does the vertical."""
        chain = _bs_chain(600.0, 0.18, 21, step=5.0)
        strategy, sigs = _build(CallCreditSpread, chain, SignalDirection.BEARISH, budget=300.0)
        assert sigs == [] and strategy.last_entry_reason == "max_loss_over_budget"

    @pytest.mark.parametrize("cls, direction, right", SIDES)
    def test_each_refusal_names_its_reason(self, cls, direction, right, monkeypatch):
        def reason(chain=None, budget=600.0):
            strategy, sigs = _build(cls, chain or _bs_chain(*self.CHAIN), direction, budget=budget)
            assert sigs == []
            return strategy.last_entry_reason

        guessed = _bs_chain(*self.CHAIN)
        guessed.iv_is_guessed = True
        assert reason(guessed) == "degraded_greeks"
        no_em = _bs_chain(*self.CHAIN)
        no_em.expected_move = 0.0
        assert reason(no_em) == "em_unavailable"
        # $250 (the shipped 5% of $5k): a 3-wide whose credit is under the condor's $0.70 floor
        assert reason(budget=250.0) == "credit_floor"
        monkeypatch.setenv("AIT_IC_MIN_CREDIT_WIDTH", "0.5")
        assert reason() == "credit_width_ratio"
        monkeypatch.setenv("AIT_IC_MIN_CREDIT_WIDTH", "0.10")
        monkeypatch.setenv("AIT_IC_DELTA_MIN", "0.25")       # the 0.20 short now sits below the band
        assert reason() == "delta_band"

    @pytest.mark.parametrize("cls, direction, right", SIDES)
    def test_the_credit_floors_are_the_condors(self, cls, direction, right, monkeypatch):
        """The same contract keys the condor and the executor's placement clamp read: moving them moves this."""
        chain = _bs_chain(*self.CHAIN)
        _, sigs = _build(cls, chain, direction, budget=600.0)
        credit = sigs[0].entry_price
        monkeypatch.setenv("AIT_IC_MIN_CREDIT", f"{credit + 0.01:.2f}")
        strategy, none = _build(cls, chain, direction, budget=600.0)
        assert none == [] and strategy.last_entry_reason == "credit_floor"
        monkeypatch.setenv("AIT_IC_MIN_CREDIT", f"{credit - 0.01:.2f}")
        _, again = _build(cls, chain, direction, budget=600.0)
        assert len(again) == 1


# --------------------------------------------------------------------------
# the wing: research's rule, the condor's budget fit
# --------------------------------------------------------------------------

class TestTheWing:
    @pytest.mark.parametrize("cls, direction, right", SIDES)
    def test_unconstrained_wing_is_researchs_rule(self, cls, direction, right):
        chain = _bs_chain(600.0, 0.18, 21)
        strategy = cls()
        em = chain.expected_move
        assert em == pytest.approx(600.0 * 0.18 * math.sqrt(21 / 365))      # the engine's expected move
        target = max(contract_float("AIT_IC_WING_FLOOR"),
                     contract_float("AIT_IC_WING_K") * em * strategy.wing_fraction())
        assert strategy.wing_width(em) == pytest.approx(target)
        _, sigs = _build(cls, chain, direction)
        short, long_ = _legs(sigs[0])
        # snapped to the listed strike nearest the target beyond the short
        assert abs(abs(short.strike - long_.strike) - target) <= 0.5

    def test_the_fraction_is_the_research_setting(self, monkeypatch):
        s = load_settings()
        assert PutCreditSpread().wing_fraction() == s.backtest.spread_tuning.wing_expected_move_fraction
        frozen = s.model_copy(update={"backtest": s.backtest.model_copy(update={
            "spread_tuning": s.backtest.spread_tuning.model_copy(update={"wing_expected_move_fraction": 1.0})})})
        monkeypatch.setattr(settings_mod, "_FROZEN", frozen)
        wide = CallCreditSpread()
        assert wide.wing_fraction() == 1.0
        assert wide.wing_width(20.0) == pytest.approx(1.6 * 20.0 * 1.0)
        assert PutCreditSpread(wing_fraction=0.25).wing_width(20.0) == pytest.approx(max(2.0, 1.6 * 20 * 0.25))

    @pytest.mark.parametrize("budget", [50.0, 150.0, 600.0, 1000.0])
    def test_a_binding_budget_fits_the_wing_exactly_as_the_condor(self, budget):
        """Same budget -> the same wing ceiling as the live condor's _vol_scaled_width (both run through
        IronCondor.affordable_width and the IC_BUDGET_CAPPED_MIN_WING floor)."""
        chain = _bs_chain(600.0, 0.18, 21)
        put = next(p for p in chain.puts if p.strike == 580.0)
        call = next(c for c in chain.calls if c.strike == 625.0)
        ic = IronCondor()
        ic.risk_budget = budget
        condor = ic._vol_scaled_width(600.0, put, call, chain)
        assert condor < 1.6 * chain.expected_move            # the budget binds the condor
        for cls in (PutCreditSpread, CallCreditSpread):
            spread = cls()
            spread.risk_budget = budget
            assert spread.wing_width(chain.expected_move) == pytest.approx(condor)
        assert PutCreditSpread().wing_width(chain.expected_move) > condor   # unconstrained is wider

    def test_the_budget_algebra_is_imported_not_copied(self, monkeypatch):
        """Change the condor's authority and the vertical follows."""
        monkeypatch.setattr(IronCondor, "affordable_width", staticmethod(lambda budget: 3.25))
        spread = PutCreditSpread()
        spread.risk_budget = 500.0
        assert spread.wing_width(25.0) == 3.25
        monkeypatch.setattr(IronCondor, "affordable_width", staticmethod(lambda budget: 0.4))
        assert spread.wing_width(25.0) == IC_BUDGET_CAPPED_MIN_WING


# --------------------------------------------------------------------------
# structure parity with the research engine
# --------------------------------------------------------------------------

class TestResearchParity:
    """The same (spot, IV, DTE) through the engine's builder and through the live builder on a Black-Scholes
    chain of that market: the same short strike, the same long strike, the same direction, two legs.

    The credit floors are set to zero here: research's spreads apply none (a registered difference, see the
    module docstring of credit_spreads.py), and this test is about structure. Cases avoid a delta root within
    a few cents of a half-dollar: there research rounds the STRIKE and live picks the nearest DELTA, which can
    be one strike apart."""

    CASES = [(600.0, 0.18, 21, "P"), (600.0, 0.18, 21, "C"), (500.0, 0.22, 25, "P"),
             (220.0, 0.24, 21, "P"), (220.0, 0.24, 21, "C"), (450.0, 0.20, 30, "P"), (450.0, 0.20, 30, "C")]

    @pytest.mark.parametrize("S, iv, dte, right", CASES)
    def test_same_structure_as_research(self, S, iv, dte, right, monkeypatch):
        from tests.test_w74_engine_realism import _bt
        monkeypatch.setenv("AIT_IC_MIN_CREDIT", "0")
        monkeypatch.setenv("AIT_IC_MIN_CREDIT_WIDTH", "0")
        name = "put_credit_spread" if right == "P" else "call_credit_spread"
        bt = _bt(position_size_pct=0.5, initial_capital=1_000_000.0)
        # the engine's knobs are the live builder's (one key each)
        assert bt._delta_short == SHORT_DELTA_TARGET
        assert bt._wing_k == contract_float("AIT_IC_WING_K")
        assert bt._wing_floor_dollars == contract_float("AIT_IC_WING_FLOOR")
        cls = STRATEGY_MAP[name]
        assert bt._spr.wing_expected_move_fraction == cls().wing_fraction()

        pos = bt._build_credit_position(name, S, iv, dte / 365.0, R, dte, date(2024, 3, 4), 1_000_000.0)
        assert pos is not None
        _, sigs = _build(cls, _bs_chain(S, iv, dte), cls.DIRECTION)
        assert len(sigs) == 1
        short, long_ = _legs(sigs[0])
        key = "put" if right == "P" else "call"
        assert (short.strike, long_.strike) == (pos[f"short_{key}_strike"], pos[f"long_{key}_strike"])
        assert sigs[0].direction.value == pos["direction"]
        assert len(sigs[0].legs) == pos["n_legs"] == 2
        assert pos["trade_type"] == "credit" and sigs[0].strategy_name in CREDIT_STRATEGIES


# --------------------------------------------------------------------------
# the selector: registered, not enabled
# --------------------------------------------------------------------------

class TestTheSelectorShipsWithoutThem:
    def test_the_shipped_config_and_the_schema_default_do_not_enable_them(self):
        assert not set(NAMES) & set(load_settings().options.strategies)
        assert not set(NAMES) & set(OptionsConfig().strategies)

    @pytest.mark.parametrize("direction", [SignalDirection.BULLISH, SignalDirection.BEARISH,
                                           SignalDirection.NEUTRAL])
    def test_the_shipped_selector_never_returns_them(self, direction):
        chain = _bs_chain(600.0, 0.18, 21)
        for options in (load_settings().options, OptionsConfig()):
            selector = StrategySelector(options)
            assert not set(NAMES) & {s.name for s in selector._strategies}
            signals = selector.generate_all_signals("SPY", chain, direction, 0.6, 40.0, risk_budget=600.0)
            assert not set(NAMES) & {s.strategy_name for s in signals}

    def test_positive_control_a_selector_that_lists_them_returns_them(self):
        """The shipped-config test above would pass vacuously if the builders emitted nothing on this chain."""
        chain = _bs_chain(600.0, 0.18, 21)
        selector = StrategySelector(OptionsConfig(strategies=list(NAMES)))
        assert {s.name for s in selector._strategies} == set(NAMES)
        bull = selector.generate_all_signals("SPY", chain, SignalDirection.BULLISH, 0.6, 40.0, risk_budget=600.0)
        bear = selector.generate_all_signals("SPY", chain, SignalDirection.BEARISH, 0.6, 40.0, risk_budget=600.0)
        assert [s.strategy_name for s in bull] == ["put_credit_spread"]
        assert [s.strategy_name for s in bear] == ["call_credit_spread"]

    def test_every_registered_name_is_its_class_name(self):
        assert all(cls().name == key for key, cls in STRATEGY_MAP.items())

    def test_every_live_credit_builder_is_classified_live(self):
        live_credit = {n for n in STRATEGY_MAP if n in CREDIT_STRATEGIES}
        assert set(NAMES) <= live_credit <= LIVE_CREDIT_STRATEGIES

    def test_they_rank_as_premium_sellers(self):
        chain = _bs_chain(600.0, 0.18, 21)
        selector = StrategySelector(OptionsConfig(strategies=["put_credit_spread"]))
        sig = selector.generate_all_signals("SPY", chain, SignalDirection.BULLISH, 0.6, 40.0, risk_budget=600.0)[0]
        hi, lo = (sig.__class__(**{**sig.__dict__, "iv_rank": r}) for r in (80.0, 20.0))
        assert selector._rank_signals([lo, hi])[0] is hi


# --------------------------------------------------------------------------
# which model gates them (TECHNICAL_GUIDE section 4, step 5)
# --------------------------------------------------------------------------

class TestModelGating:
    @pytest.mark.parametrize("name", NAMES)
    def test_gated_by_the_direction_model_not_the_range_model(self, name):
        assert name not in RANGE_GATED_STRATEGIES
        for gates_on in (False, True):
            assert sizing_confidence(name, gates_on=gates_on, range_prob=0.9, directional_conf=0.55,
                                     neutral_conf=0.60) == 0.55


# --------------------------------------------------------------------------
# the existing exit paths handle a two-leg credit structure
# --------------------------------------------------------------------------

async def _booked(env, cls, *, quantity=1, dte=25):
    """A spread the real builder made, placed through the real executor; returns (signal, the trade row)."""
    chain = _bs_chain(600.0, 0.18, dte)
    _, sigs = _build(cls, chain, cls.DIRECTION, budget=600.0)
    sig = sigs[0]
    env.ib.quote = _nbbo_for(sig)
    trade_id = await env.executor.execute_signal(sig, quantity)
    assert trade_id is not None
    return sig, env.state.get_trade_by_id(trade_id)


def _monitor(spot):
    """The proven exit-monitor harness (tests/test_credit_exits.py) with the REAL leg-mark P&L."""
    from tests.test_credit_exits import _manager
    mgr = _manager(spot=spot, unrealized=None)
    del mgr._option_position_unrealized          # back to PortfolioManager's own method
    return mgr


def _marks(row, pnl_pct):
    """IB-style per-leg marks that put the position at ``pnl_pct`` of its credit."""
    legs = json.loads(row.legs)
    short = next(l for l in legs if l["action"] == "SELL")
    long_ = next(l for l in legs if l["action"] == "BUY")
    cost_to_close = row.entry_price * (1.0 - pnl_pct)
    key = lambda l: (row.symbol, l["expiry"].replace("-", "")[:8], float(l["strike"]), l["right"])  # noqa: E731
    return {key(long_): 0.20, key(short): 0.20 + cost_to_close}


def _shift_today(monkeypatch, days):
    today = today_et()
    monkeypatch.setattr(portfolio_mod, "today_et", lambda: today + timedelta(days=days))


class TestTheExistingExitPathsHandleTheVertical:
    @pytest.mark.parametrize("cls", [PutCreditSpread, CallCreditSpread])
    async def test_the_row_the_executor_writes(self, env, cls):
        sig, row = await _booked(env, cls)
        short, long_ = _legs(sig)
        assert row.strategy == cls().name and row.contract_type == "spread"
        assert row.entry_price == pytest.approx(sig.entry_price) and row.strike == short.strike
        legs = {(l["strike"], l["right"], l["action"]) for l in json.loads(row.legs)}
        assert legs == {(short.strike, short.right, "SELL"), (long_.strike, long_.right, "BUY")}

    @pytest.mark.parametrize("cls", [PutCreditSpread, CallCreditSpread])
    async def test_a_touch_of_the_short_strike_closes_it(self, env, cls):
        sig, row = await _booked(env, cls)
        short, _ = _legs(sig)
        status = await _monitor(spot=short.strike)._evaluate_position(row, option_marks=_marks(row, -0.30))
        side = "put" if short.right == "P" else "call"
        assert status.should_exit and status.exit_reason.startswith("short_strike_touch")
        assert f"{side} {short.strike:.2f}" in status.exit_reason
        # one dollar on the safe side of the short: no touch, no exit
        safe = short.strike + 1.0 if short.right == "P" else short.strike - 1.0
        held = await _monitor(spot=safe)._evaluate_position(row, option_marks=_marks(row, -0.30))
        assert not held.should_exit

    @pytest.mark.parametrize("cls", [PutCreditSpread, CallCreditSpread])
    async def test_take_profit_follows_the_shared_ladder(self, env, cls, monkeypatch):
        """45% of credit: below the DTE>20 rung (50%), above the DTE 11-20 rung (40%). The same position exits
        only once ten days have passed, with the ladder's own target in the reason."""
        _, row = await _booked(env, cls, dte=25)
        spot = 600.0
        early = await _monitor(spot)._evaluate_position(row, option_marks=_marks(row, 0.45))
        assert not early.should_exit and early.dte == 25
        _shift_today(monkeypatch, 10)
        late = await _monitor(spot)._evaluate_position(row, option_marks=_marks(row, 0.45))
        target = take_profit_targets(15, True)[1]
        assert late.dte == 15 and target < 0.45 < take_profit_targets(25, True)[1]
        assert late.should_exit and late.exit_reason.startswith("take_profit_short")
        assert f"target: {target:.1%}" in late.exit_reason

    @pytest.mark.parametrize("cls", [PutCreditSpread, CallCreditSpread])
    async def test_the_dte_close(self, env, cls, monkeypatch):
        _, row = await _booked(env, cls, dte=25)
        _shift_today(monkeypatch, 25 - (EXPIRY_APPROACHING_DTE - 1))
        status = await _monitor(600.0)._evaluate_position(row, option_marks=_marks(row, 0.10))
        assert status.dte == EXPIRY_APPROACHING_DTE - 1
        assert status.should_exit and status.exit_reason == f"expiry_approaching (DTE: {EXPIRY_APPROACHING_DTE - 1})"

    @pytest.mark.parametrize("cls", [PutCreditSpread, CallCreditSpread])
    async def test_a_healthy_position_holds(self, env, cls):
        _, row = await _booked(env, cls, dte=25)
        status = await _monitor(600.0)._evaluate_position(row, option_marks=_marks(row, 0.10))
        assert not status.should_exit and status.action.value == "hold"

    @pytest.mark.parametrize("cls", [PutCreditSpread, CallCreditSpread])
    async def test_a_multi_lot_vertical_is_managed_whole(self, env, cls):
        """W112: no scale-out on a multi-leg position. A milestone the P&L has passed proposes nothing here."""
        from ait.config.settings import ExitConfig
        _, row = await _booked(env, cls, quantity=2, dte=25)
        assert row.quantity == 2
        mgr = _monitor(600.0)
        mgr._exit_config = ExitConfig(partial_exit_levels=[{"pnl_pct": 0.25, "close_pct": 0.5}])
        status = await mgr._evaluate_position(row, option_marks=_marks(row, 0.30))
        assert status.partial_exit_quantity == 0 and not status.should_exit

    @pytest.mark.parametrize("name", NAMES)
    def test_held_through_macro_events_like_the_condor(self, name):
        assert macro_flatten_window_days(name) is None
        assert macro_flatten_window_days("iron_condor") is None

    @pytest.mark.parametrize("cls", [PutCreditSpread, CallCreditSpread])
    async def test_close_pricing_settlement_and_booking_see_a_vertical(self, env, cls):
        """The exit width bound, the settlement value and the realized P&L sign on the executor's own row."""
        from ait.bot.orchestrator import TradingOrchestrator
        from ait.execution.executor import TradeExecutor
        from ait.execution.reconciler import PositionReconciler
        sig, row = await _booked(env, cls, quantity=2)
        short, long_ = _legs(sig)
        width = abs(short.strike - long_.strike)
        assert TradingOrchestrator._defined_risk_width(json.loads(row.legs)) == width
        # cost to settle: 0 out of the money, the width beyond the long, short-to-spot between
        otm, deep = (short.strike + 5, long_.strike - 5) if short.right == "P" else (short.strike - 5, long_.strike + 5)
        between = (short.strike + long_.strike) / 2
        assert PositionReconciler._structure_intrinsic(row, otm) == 0.0
        assert PositionReconciler._structure_intrinsic(row, deep) == pytest.approx(width)
        assert PositionReconciler._structure_intrinsic(row, between) == pytest.approx(width / 2)
        # credit P&L: collected entry, paid exit; 2 legs a side x 2 sides x lots in the commission estimate
        bt = load_settings().backtest
        per_leg = bt.commission_per_contract + bt.regulatory_fees_per_contract
        assert TradeExecutor.commission_estimate(row) == pytest.approx(per_leg * 2 * 2 * row.quantity)
        exit_price = round(row.entry_price / 2, 2)
        pnl = env.executor._calculate_realized_pnl(row, exit_price)
        expected = (row.entry_price - exit_price) * 100 * 2 - TradeExecutor.commission_estimate(row)
        assert expected > 0 and pnl == pytest.approx(expected, abs=0.005)    # booked to the cent

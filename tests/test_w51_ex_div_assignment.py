"""W51 (model-vs-reality-ex-div-assignment-void, deep half): the daily
pre-ex-div close and the entry-blocking window. W41 already fixed piece 1
(the reconciler's STK branch freezes new entries on an untracked stock
position, the signature of an assignment) — re-verified still in place at
`reconciler.py:1503` before this wave started, and its own comment
explicitly left two follow-ups "for W51's deeper pass": whether STK should
get a debounce (decided: no — assignment is a discrete broker event, not
the position-stream flicker the OPT debounce exists for) and whether it
should route through the Telegram declare-yes/no flow (decided: yes, for
consistency and so a declared-manual stock position stops re-freezing every
pass).

No ex-dividend date or amount existed anywhere in this codebase before this
wave (verified: `ticker.calendar`, earnings.py's own primary source, is
EMPTY for SPY/TLT — checked directly against live yfinance output, not
assumed — a 404 "No fundamentals data found"). `DividendCalendar`
(ait/data/dividends.py) is EarningsCalendar's structural twin, sourcing
`ticker.dividends` (real historical per-payment dates+amounts) instead, and
projecting the next payment forward from the average interval between
recent ones.

Design correction made during implementation, not just planning: the
close rule (3e, portfolio.py) runs on check_positions' own ~30s cadence —
same as its closest sibling, rule 3c (pre_earnings_iv_crush), which already
includes iron_condor — NOT as a separate daily job. The "daily" framing in
the finding's own text is about the dividend LOOKUP being cheap to cache
(DividendCalendar's 6h TTL), not the comparison needing to run once a day.

Every test executes the real code path. No inspect.getsource.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.config.settings import ExitConfig
from ait.data.dividends import DividendCalendar, DividendInfo
from ait.data.market_data import Quote
from ait.execution.portfolio import PortfolioManager
from ait.utils.time import today_et

CONDOR_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])


# ---------------------------------------------------------------------------
# DividendCalendar — mirrors earnings.py's own test shape
# ---------------------------------------------------------------------------

class TestDividendCalendar:
    def test_a_fetch_failure_falls_back_to_last_known_good(self, monkeypatch):
        cal = DividendCalendar()
        good = DividendInfo(symbol="SPY", next_ex_div_date=date(2026, 12, 18),
                            expected_amount=1.89)
        cal._remember("SPY", good)  # a prior successful fetch

        monkeypatch.setattr(cal, "_fetch_dividend",
                            lambda s: DividendInfo(symbol=s, next_ex_div_date=None,
                                                   expected_amount=None,
                                                   data_available=False))
        cal._cache._store.clear()  # force past the TTL-cache hit

        info = cal.get_next_ex_div("SPY")
        assert info.data_available is True, "must serve LKG, not report unknown"
        assert info.next_ex_div_date == date(2026, 12, 18)
        assert info.expected_amount == 1.89

    def test_a_fetch_failure_with_no_lkg_reports_unavailable(self, monkeypatch):
        cal = DividendCalendar()
        monkeypatch.setattr(cal, "_fetch_dividend",
                            lambda s: DividendInfo(symbol=s, next_ex_div_date=None,
                                                   expected_amount=None,
                                                   data_available=False))
        info = cal.get_next_ex_div("NEWETF")
        assert info.data_available is False

    def test_a_fetch_failure_is_never_cached_as_no_dividend(self, monkeypatch):
        """CONTROL: the same W44 lesson earnings.py already learned — a
        transient failure must not poison the 6h TTL cache as a confirmed
        answer."""
        cal = DividendCalendar()
        monkeypatch.setattr(cal, "_fetch_dividend",
                            lambda s: DividendInfo(symbol=s, next_ex_div_date=None,
                                                   expected_amount=None,
                                                   data_available=False))
        cal.get_next_ex_div("SPY")
        assert cal._cache.get("div_SPY") is None

    def test_would_hold_through_ex_div_compares_against_the_et_calendar_day(self):
        """Same call shape as EarningsCalendar.would_hold_through_earnings —
        drives the real method against a stubbed get_next_ex_div."""
        cal = DividendCalendar()
        today = date(2026, 9, 22)
        cal.get_next_ex_div = lambda symbol: DividendInfo(
            symbol=symbol, next_ex_div_date=today + timedelta(days=10),
            expected_amount=1.89)
        expiry = today + timedelta(days=21)
        assert cal.would_hold_through_ex_div("SPY", today, expiry) is True

    def test_ex_div_outside_the_window_does_not_block(self):
        """CONTROL: must hold on both trees."""
        cal = DividendCalendar()
        today = date(2026, 9, 22)
        cal.get_next_ex_div = lambda symbol: DividendInfo(
            symbol=symbol, next_ex_div_date=today + timedelta(days=90),
            expected_amount=1.89)
        assert cal.would_hold_through_ex_div(
            "SPY", today, today + timedelta(days=21)) is False

    def test_data_unavailable_refuses_entries_by_default(self):
        """CONTROL: the fail-closed-for-entries policy, config-driven, same
        default as EarningsConfig."""
        cal = DividendCalendar(refuse_entries_when_data_unavailable=True)
        cal.get_next_ex_div = lambda symbol: DividendInfo(
            symbol=symbol, next_ex_div_date=None, expected_amount=None,
            data_available=False)
        assert cal.would_hold_through_ex_div(
            "SPY", date(2026, 9, 22), date(2026, 10, 13)) is True


# ---------------------------------------------------------------------------
# Rule 3e (portfolio.py check_positions) — real code path, real fixture
# rig (same shape as test_r17_live_fixes.py's rule-3c rig)
# ---------------------------------------------------------------------------

@dataclass
class _FakeTrade:
    entry_price: float = 1.00
    quantity: int = 1
    contract_type: str = "spread"
    strategy: str = "iron_condor"
    symbol: str = "SPY"
    trade_id: str = "T-EXDIV"
    direction: object = None
    legs: str = CONDOR_LEGS
    expiry: str | None = "2026-12-18"
    strike: float | None = None
    entry_time: str = "2026-07-01T10:00:00"


def _manager(spot: float, next_ex_div_days: int | None, dividend_amount: float | None,
            data_available: bool = True):
    mgr = PortfolioManager.__new__(PortfolioManager)
    mgr._ibkr = MagicMock()
    mgr._ibkr.ib.portfolio.return_value = []
    mgr._state = MagicMock()
    mgr._state.get_high_water_mark.return_value = 0.0
    mgr._market_data = MagicMock()

    tick = {"n": 0}

    async def _quote(symbol):
        tick["n"] += 1
        return Quote(
            symbol=symbol, bid=round(spot - 0.01, 2), ask=round(spot + 0.01, 2),
            last=spot, volume=1_000_000,
            timestamp=datetime.now() + timedelta(seconds=tick["n"]),
        )
    mgr._market_data.get_quote = _quote

    async def _price(symbol):
        return spot
    mgr._market_data.get_current_price = _price
    mgr._exit_config = ExitConfig()

    # No earnings/macro noise — isolate rule 3e.
    earnings = MagicMock()
    earnings.get_next_earnings.return_value = SimpleNamespace(next_earnings_date=None)
    mgr._earnings = earnings
    mgr._economic_cal = None

    if next_ex_div_days is None:
        _next_date = None
    else:
        # CENSUS-6 (2026-09-28): days_to_ex_div is computed against
        # today_et() (portfolio.py:901), not local wall-clock -- date.today()
        # here diverges from it whenever local time has crossed midnight but
        # ET hasn't yet, producing a spurious off-by-one that made this
        # fixture flaky near local midnight.
        _next_date = today_et() + timedelta(days=next_ex_div_days)
    dividends = MagicMock()
    dividends.get_next_ex_div.return_value = DividendInfo(
        symbol="SPY", next_ex_div_date=_next_date,
        expected_amount=dividend_amount, data_available=data_available)
    mgr._dividends = dividends

    async def _vol_mult(symbol):
        return 1.0
    mgr._get_volatility_stop_multiplier = _vol_mult
    mgr._pdt_guard = MagicMock()
    mgr._pdt_guard.would_be_day_trade.return_value = False
    mgr._get_position_delta = lambda trade: None
    return mgr


def _condor(dte: int, entry_price: float = 1.00) -> _FakeTrade:
    t = _FakeTrade(expiry=(today_et() + timedelta(days=dte)).isoformat(),
                  entry_price=entry_price)
    t.direction = SimpleNamespace(value="neutral")
    return t


# Short call leg is strike 102.0 (CONDOR_LEGS). Spot 100.0 -> OTM -> intrinsic
# 0, so extrinsic == the mark directly.
_SHORT_CALL_KEY = ("SPY", "20261218", 102.0, "C")


class TestExDivAssignmentRiskRule:
    async def test_low_extrinsic_the_day_before_ex_div_closes_regardless_of_dte(
            self, monkeypatch):
        """The defect, reproduced then closed: before this wave, NOTHING in
        the exit rules checked ex-dividend risk at any DTE (only rule 3's
        dte<=1 ITM check exists, and it's keyed off trade.strike — meaningless
        for a multi-leg condor). dte=20 here specifically proves the
        regardless-of-DTE claim."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")  # isolate rule 3e
        mgr = _manager(spot=100.0, next_ex_div_days=1, dividend_amount=1.80)
        marks = {_SHORT_CALL_KEY: 0.30}  # extrinsic 0.30 < dividend 1.80

        status = await mgr._evaluate_position(_condor(dte=20), marks)

        assert status.should_exit is True
        assert status.exit_reason.startswith("pre_ex_div_assignment_risk"), status.exit_reason

    async def test_extrinsic_above_the_dividend_does_not_close(self, monkeypatch):
        """CONTROL: must hold on both trees — plenty of time value left, no
        early-assignment incentive."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")
        mgr = _manager(spot=100.0, next_ex_div_days=1, dividend_amount=1.80)
        marks = {_SHORT_CALL_KEY: 5.00}  # extrinsic 5.00 > dividend 1.80

        status = await mgr._evaluate_position(_condor(dte=20), marks)

        assert status.should_exit is False

    async def test_two_days_before_ex_div_does_not_close(self, monkeypatch):
        """CONTROL: the finding's own timing is the day before, not a
        multi-day window — early assignment risk is a last-evening decision."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")
        mgr = _manager(spot=100.0, next_ex_div_days=2, dividend_amount=1.80)
        marks = {_SHORT_CALL_KEY: 0.30}

        status = await mgr._evaluate_position(_condor(dte=20), marks)

        assert status.should_exit is False

    async def test_no_dividend_data_does_not_force_an_exit(self, monkeypatch):
        """CONTROL: this is the EXIT side — fails OPEN on missing/unknown
        calendar data, matching rule 3c's own documented rationale. A missed
        early close is strictly better than a forced exit on bad data."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")
        mgr = _manager(spot=100.0, next_ex_div_days=None, dividend_amount=None,
                      data_available=False)
        marks = {_SHORT_CALL_KEY: 0.30}

        status = await mgr._evaluate_position(_condor(dte=20), marks)

        assert status.should_exit is False

    async def test_missing_mark_does_not_force_an_exit(self, monkeypatch):
        """CONTROL: no usable IBKR mark for the short call leg — skip, don't
        guess (same principle as the marks_missing handling elsewhere in
        this function)."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "0")
        mgr = _manager(spot=100.0, next_ex_div_days=1, dividend_amount=1.80)

        status = await mgr._evaluate_position(_condor(dte=20), {})  # no marks at all

        assert status.should_exit is False


# ---------------------------------------------------------------------------
# Entry block: _signal_sells_a_call (orchestrator.py, genuinely NEW logic —
# unlike would_hold_through_ex_div, which is EarningsCalendar's direct
# mirror, this detection has no earnings-gate equivalent to copy, so it
# gets its own direct tests rather than only indirect coverage) + the
# calendar predicate itself, mirroring test_w41_live_money_floors.py's
# earnings-gate test pattern.
# ---------------------------------------------------------------------------

def _contract(right: str, strike: float = 100.0) -> SimpleNamespace:
    """Minimal OptionContract-shaped stand-in — _signal_sells_a_call only
    reads .right, so a full OptionContract (bid/ask/volume/... required
    fields) would be pure noise here."""
    return SimpleNamespace(right=right, strike=strike)


def _signal(*, action: str = "BUY", contract=None, legs=None) -> SimpleNamespace:
    return SimpleNamespace(action=action, contract=contract, legs=legs or [])


class TestSignalSellsACall:
    """Direct tests of the new detection helper — orchestrator.py's
    _signal_sells_a_call, extracted so it's testable without fighting
    through _try_execute's full gate gauntlet (duplicate guard, HALT,
    macro blackout, cooldown, budget, ...) just to reach this check."""

    def test_a_condor_shaped_multi_leg_signal_sells_a_call(self):
        from ait.bot.orchestrator import TradingOrchestrator
        legs = [
            {"action": "BUY", "contract": _contract("P", 95.0)},
            {"action": "SELL", "contract": _contract("P", 98.0)},
            {"action": "SELL", "contract": _contract("C", 102.0)},
            {"action": "BUY", "contract": _contract("C", 105.0)},
        ]
        assert TradingOrchestrator._signal_sells_a_call(_signal(legs=legs)) is True

    def test_a_put_only_multi_leg_signal_does_not(self):
        """CONTROL: a put credit spread (no call leg at all) must not block
        on ex-dividend risk — that risk is specifically a short CALL's."""
        from ait.bot.orchestrator import TradingOrchestrator
        legs = [
            {"action": "BUY", "contract": _contract("P", 90.0)},
            {"action": "SELL", "contract": _contract("P", 95.0)},
        ]
        assert TradingOrchestrator._signal_sells_a_call(_signal(legs=legs)) is False

    def test_a_bull_call_spread_has_a_short_call_leg_too(self):
        """A bull call spread (BUY the near strike, SELL the far one) still
        has a SELL/call leg — assignment risk is real on its short leg even
        though the STRUCTURE is a net debit. Confirms the check is genuinely
        per-leg, not a strategy-level credit/debit classification."""
        from ait.bot.orchestrator import TradingOrchestrator
        legs = [
            {"action": "BUY", "contract": _contract("C", 100.0)},
            {"action": "SELL", "contract": _contract("C", 105.0)},
        ]
        assert TradingOrchestrator._signal_sells_a_call(_signal(legs=legs)) is True

    def test_a_pure_debit_multi_leg_signal_does_not(self):
        """CONTROL: both legs BUY (e.g. a long call calendar) — no SELL
        leg at all, let alone a short call."""
        from ait.bot.orchestrator import TradingOrchestrator
        legs = [
            {"action": "BUY", "contract": _contract("C", 100.0)},
            {"action": "BUY", "contract": _contract("C", 105.0)},
        ]
        assert TradingOrchestrator._signal_sells_a_call(_signal(legs=legs)) is False

    def test_a_single_leg_short_call_sells_a_call(self):
        """covered_call / naked short call — no .legs at all, the primary
        action/contract fields ARE the position."""
        from ait.bot.orchestrator import TradingOrchestrator
        sig = _signal(action="SELL", contract=_contract("C", 100.0), legs=[])
        assert TradingOrchestrator._signal_sells_a_call(sig) is True

    def test_a_single_leg_short_put_does_not(self):
        """CONTROL: cash_secured_put — sells a PUT, not a call."""
        from ait.bot.orchestrator import TradingOrchestrator
        sig = _signal(action="SELL", contract=_contract("P", 100.0), legs=[])
        assert TradingOrchestrator._signal_sells_a_call(sig) is False

    def test_a_single_leg_long_call_does_not(self):
        """CONTROL: a bare long call — BUY, not SELL."""
        from ait.bot.orchestrator import TradingOrchestrator
        sig = _signal(action="BUY", contract=_contract("C", 100.0), legs=[])
        assert TradingOrchestrator._signal_sells_a_call(sig) is False

    def test_no_contract_and_no_legs_does_not_raise(self):
        """CONTROL: a malformed/empty signal must not crash the gate."""
        from ait.bot.orchestrator import TradingOrchestrator
        assert TradingOrchestrator._signal_sells_a_call(
            _signal(action="SELL", contract=None, legs=[])) is False


class TestExDivEntryBlock:
    def test_a_short_call_signal_straddling_ex_div_is_blocked(self):
        cal = DividendCalendar()
        today = date(2026, 9, 22)
        expiry = today + timedelta(days=21)
        cal.get_next_ex_div = lambda symbol: DividendInfo(
            symbol=symbol, next_ex_div_date=today + timedelta(days=10),
            expected_amount=1.80)

        assert cal.would_hold_through_ex_div("SPY", today, expiry) is True

    def test_ex_div_outside_the_hold_window_does_not_block(self):
        """CONTROL: must hold on both trees."""
        cal = DividendCalendar()
        today = date(2026, 9, 22)
        cal.get_next_ex_div = lambda symbol: DividendInfo(
            symbol=symbol, next_ex_div_date=today + timedelta(days=90),
            expected_amount=1.80)
        assert cal.would_hold_through_ex_div(
            "SPY", today, today + timedelta(days=21)) is False


# ---------------------------------------------------------------------------
# Reconciler STK branch — Telegram declare-flow (W41's own flagged follow-up)
# ---------------------------------------------------------------------------

class FakeTelegram:
    """Same duck-typed double as test_w22_manual_positions.py."""

    def __init__(self):
        self.enabled = True
        self.buttoned: list[dict] = []
        self._next_msg_id = 1000

    async def send_with_buttons(self, text: str, buttons) -> int:
        msg_id = self._next_msg_id
        self._next_msg_id += 1
        self.buttoned.append({"message_id": msg_id, "text": text, "buttons": buttons})
        return msg_id


def _state(tmp_path):
    from ait.bot.state import StateManager
    return StateManager(db_path=tmp_path / "ait_state.db")


def _rec(state, ib, telegram=None):
    from ait.execution.reconciler import PositionReconciler
    from tests.fakes import FakeIBKRClient
    r = PositionReconciler(FakeIBKRClient(ib), state)
    if telegram is not None:
        r._telegram = telegram
    return r


@pytest.fixture(autouse=True)
def _cwd_tmp(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


class TestStkBranchTelegramFlow:
    def test_an_assignment_now_sends_a_stock_labeled_telegram_prompt(self, tmp_path):
        from tests.fakes import FakeIB, _stock_position

        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_stock_position("SPY", 100.0, avg_cost=660.0)]
        tg = FakeTelegram()

        asyncio.run(_rec(st, ib, telegram=tg).reconcile())

        assert len(tg.buttoned) == 1
        assert "stock position" in tg.buttoned[0]["text"], (
            "must say STOCK, not the OPT branch's hardcoded 'option position' text"
        )
        assert "SPY:STK" in tg.buttoned[0]["text"]

    def test_a_declared_stock_position_stops_re_freezing(self, tmp_path):
        """The actual point of routing through the flow: once the operator
        answers 'yes, mine' (declared_quantity_matches), the SAME position
        must not re-freeze / re-prompt on the next reconcile pass. Same
        state-key convention test_w22_manual_positions.py itself uses
        (documented in manual_positions.py's own module docstring)."""
        from tests.fakes import FakeIB, _stock_position

        st = _state(tmp_path)
        key = "SPY:STK"
        st.set_state(f"manual_position_answer_{key}", "yes")
        st.set_state(f"manual_position_declared_qty_{key}", "100.0")

        ib = FakeIB()
        ib.positions_list = [_stock_position("SPY", 100.0, avg_cost=660.0)]
        tg = FakeTelegram()

        result = asyncio.run(_rec(st, ib, telegram=tg).reconcile())

        assert tg.buttoned == [], "a declared position must not be re-prompted"
        assert not any("ASSIGNMENT" in d for d in result.discrepancies), (
            "a declared position must not re-freeze either"
        )

    def test_no_telegram_client_does_not_break_the_freeze(self, tmp_path):
        """CONTROL: reconcile() must still freeze correctly when _telegram
        doesn't exist (several existing tests construct the reconciler this
        way) — the getattr guard must degrade to a no-op, not raise."""
        from tests.fakes import FakeIB, _stock_position

        st = _state(tmp_path)
        ib = FakeIB()
        ib.positions_list = [_stock_position("SPY", 100.0, avg_cost=660.0)]

        result = asyncio.run(_rec(st, ib, telegram=None).reconcile())

        assert any("ASSIGNMENT" in d for d in result.discrepancies)
        halt = tmp_path / "data" / "HALT_UNTRACKED"
        assert halt.exists()

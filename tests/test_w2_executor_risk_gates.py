"""W2: entry-gate FAIL DIRECTION defects from
reports/blindspot_composition_hunt_20260825.md.

Every test EXECUTES the real method under test (real TradeExecutor /
TradingOrchestrator / PDTGuard, real StateManager on a tmp SQLite file, only
the broker boundary faked via tests/fakes.py). inspect.getsource is NOT used
anywhere in this file: a structural assertion cannot tell a gate that blocks
from a gate that logs and proceeds, which is precisely the defect class here.

Findings covered:
  fail-direction-07  combo NBBO sanity gate failed open quietly — a quote
                     outage or nan NBBO sent a marketable limit anchored to a
                     stale scan-time mid, at log.debug/silence. Entries now
                     fail CLOSED; exits stay fail-OPEN but LOUD.
  fail-direction-08  the three duplicate-order layers swallowed their own
                     errors at debug and proceeded to place.
  fail-direction-09  PDT guard: a corrupt counter or a broken market calendar
                     silently became "always allow".
  fail-direction-11  the restricted/ban list treated an unreadable or
                     BOM/UTF-16 encoded file as "no bans", silently.
  fail-direction-12  the executor's market-hours guard swallowed calendar
                     exceptions at a bare `pass`.
  trade-life-credit-econ-floors-signal-time-only
                     the credit-economics floors were validated at SIGNAL
                     time only; the reprice ladder could fill below both.

Pre-fix behaviour (what these tests would have caught): every "refused"
assertion below was a PLACED order, and every PDT/ban-list block was an
allow.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from ait.bot.orchestrator import (
    RESTRICTED_UNREADABLE,
    TradingOrchestrator,
    read_restricted_symbols,
)
from ait.bot.state import StateManager, TradeStatus
from ait.config.settings import AccountConfig
from ait.execution.executor import PendingOrder, TradeExecutor
from ait.risk.circuit_breaker import CircuitBreaker
from ait.risk.pdt_guard import PDTStatus, PDTGuard
from ait.strategies.base import CREDIT_STRATEGIES, Signal, SignalDirection

from tests.fakes import FakeIB, FakeIBKRClient

EXPIRY_ISO = "2026-08-21"


# ---------------------------------------------------------------------------
# Shared fixtures / builders
# ---------------------------------------------------------------------------

@pytest.fixture
def events():
    """Capture structlog events regardless of level (setup_logging is never
    called under pytest, so caplog sees nothing)."""
    from structlog.testing import capture_logs
    with capture_logs() as ev:
        yield ev


def _named(events, name: str) -> list[dict]:
    return [e for e in events if e.get("event") == name]


def _leg(strike: float, right: str, action: str) -> dict:
    return {
        "contract": SimpleNamespace(strike=strike, right=right, expiry=EXPIRY_ISO),
        "action": action,
        "ratio": 1,
    }


def _ic_signal(credit: float = 1.00, width: float = 3.0,
               symbol: str = "SPY") -> Signal:
    """Defined-risk iron condor collecting ``credit`` (2 SELL / 2 BUY)."""
    return Signal(
        symbol=symbol,
        strategy_name="iron_condor",
        direction=SignalDirection.NEUTRAL,
        confidence=0.70,
        legs=[
            _leg(100.0 - width, "P", "BUY"),
            _leg(100.0, "P", "SELL"),
            _leg(110.0, "C", "SELL"),
            _leg(110.0 + width, "C", "BUY"),
        ],
        entry_price=credit,
        max_loss=(width - credit) * 100,
        max_profit=credit * 100,
        expiry=date(2026, 8, 21),
    )


def _nbbo_for(signal: Signal, spread: float = 0.04) -> tuple[float, float]:
    """A healthy combo NBBO centred on the signal price. Always-BUY
    convention: credit combos quote NEGATIVE."""
    px = abs(signal.entry_price)
    half = spread / 2
    if signal.strategy_name in CREDIT_STRATEGIES:
        return (-(px + half), -(px - half))
    return (px - half, px + half)


@pytest.fixture
def env(tmp_path, monkeypatch, risk_config):
    """Real executor + real state on a tmp DB; broker boundary faked."""
    monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
    monkeypatch.setenv("AIT_ENTRY_LADDER_START", "0.25")
    state = StateManager(tmp_path / "state.db")
    ib = FakeIB()
    client = FakeIBKRClient(ib)
    executor = TradeExecutor(
        ibkr_client=client,
        state=state,
        circuit_breaker=CircuitBreaker(risk_config),
        order_timeout=240,
    )
    return SimpleNamespace(state=state, ib=ib, client=client,
                           executor=executor, db=tmp_path / "state.db")


def _only_trade_row(env) -> tuple[str, str]:
    with sqlite3.connect(env.db) as conn:
        rows = conn.execute("SELECT trade_id, status FROM trades").fetchall()
    assert len(rows) == 1, f"expected exactly one trade row, got {rows}"
    return rows[0]


# ===========================================================================
# fail-direction-07 — combo NBBO sanity gate
# ===========================================================================
class TestComboNbboGate:
    """ENTRY fails CLOSED without a live NBBO; EXIT never blocks (loud)."""

    async def test_entry_refused_on_quote_outage(self, env, events):
        # FakeIB.reqMktData raises until .quote is set — the reqMktData
        # failure mode of the finding. Pre-fix: log.debug and a marketable
        # limit anchored to the (possibly minutes-old) signal mid.
        trade_id = await env.executor.execute_signal(_ic_signal(), 1)

        assert trade_id is None
        assert env.ib.placed == [], "no order may reach the broker"
        assert env.executor._pending_orders == {}
        refused = _named(events, "combo_nbbo_unavailable_entry_refused")
        assert len(refused) == 1
        assert refused[0]["log_level"] == "warning"
        _tid, status = _only_trade_row(env)
        assert status == TradeStatus.CANCELLED.value

    async def test_entry_refused_on_nan_nbbo(self, env, events):
        # The nan/empty-NBBO mode: the ticker answers, both sides are nan.
        env.ib.quote = (float("nan"), float("nan"))
        trade_id = await env.executor.execute_signal(_ic_signal(), 1)

        assert trade_id is None
        assert env.ib.placed == []
        refused = _named(events, "combo_nbbo_unavailable_entry_refused")
        assert len(refused) == 1
        assert refused[0]["reason"] == "nan_or_empty_nbbo"

    async def test_entry_refused_on_crossed_book(self, env, events):
        # bid >= ask is not a usable sanity input either (pre-fix: fell
        # through the `if` with no else and no log).
        env.ib.quote = (-0.98, -1.02)
        trade_id = await env.executor.execute_signal(_ic_signal(), 1)

        assert trade_id is None
        assert _named(events, "combo_nbbo_unavailable_entry_refused")[0][
            "reason"] == "crossed_or_locked_nbbo"

    async def test_entry_proceeds_on_healthy_nbbo(self, env, events):
        sig = _ic_signal(credit=1.00)
        env.ib.quote = _nbbo_for(sig)
        trade_id = await env.executor.execute_signal(sig, 1)

        assert trade_id is not None
        assert len(env.ib.placed) == 1
        assert _named(events, "combo_nbbo_unavailable_entry_refused") == []

    async def test_exit_never_blocked_but_screams(self, env, events):
        # The asymmetry is the point: a missed exit is worse than a missed
        # entry, so the SAME unavailable NBBO must still place an exit — at
        # log.error, never silently.
        sig = _ic_signal(credit=1.00)
        trade = await env.executor._execute_multi_leg(
            sig, 1, "T-W2-EXIT", is_exit=True)

        assert trade is not None, "an exit must never be blocked by this gate"
        assert len(env.ib.placed) == 1
        loud = _named(events, "combo_nbbo_unavailable_exit_proceeding")
        assert len(loud) == 1
        assert loud[0]["log_level"] == "error"
        assert _named(events, "combo_nbbo_unavailable_entry_refused") == []


# ===========================================================================
# fail-direction-08 — duplicate-order guards
# ===========================================================================
def _seed_trade(state: StateManager, symbol="SPY", strategy="iron_condor",
                minutes_ago: float = 5.0, trade_id="T-DUP") -> None:
    entry = (datetime.now() - timedelta(minutes=minutes_ago)).isoformat()
    with sqlite3.connect(state._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, "
            "status, entry_time, entry_price, quantity, contract_type) "
            "VALUES (?,?,?,?,?,?,?,1,'iron_condor')",
            (trade_id, symbol, strategy, "short", "filled", entry, 1.0))


class TestDuplicateGuardVerdict:
    """A layer that ERRORS is not a layer that found nothing."""

    def _orch(self, state, client, pending=None):
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._state = state
        o._ibkr = client
        o._executor = SimpleNamespace(_pending_orders=dict(pending or {}))
        return o

    def test_clean_pass_is_clear(self, tmp_path):
        state = StateManager(tmp_path / "state.db")
        orch = self._orch(state, FakeIBKRClient(FakeIB()))
        assert orch._duplicate_guard_verdict(_ic_signal()) == "clear"

    def test_locked_db_refuses_instead_of_placing(self, tmp_path, events):
        def _locked(**_kw):
            raise sqlite3.OperationalError("database is locked")

        state = SimpleNamespace(get_recent_trades=_locked)
        orch = self._orch(state, FakeIBKRClient(FakeIB()))

        # Pre-fix: log.debug("cooldown_check_failed") and on to placement.
        assert orch._duplicate_guard_verdict(_ic_signal()) == "unverified"
        refused = _named(events, "cooldown_check_failed_entry_refused")
        assert len(refused) == 1 and refused[0]["log_level"] == "warning"

    def test_disconnected_broker_read_refuses(self, tmp_path, events):
        state = StateManager(tmp_path / "state.db")
        # get_open_orders() returns [] while disconnected — indistinguishable
        # from a verified-empty book, which is how a reconnect blip used to
        # read as "no working orders".
        client = FakeIBKRClient(FakeIB(), connected=False)
        orch = self._orch(state, client)

        assert orch._duplicate_guard_verdict(_ic_signal()) == "unverified"
        assert _named(events, "working_order_check_disconnected_entry_refused")

    def test_broker_read_error_refuses(self, tmp_path, events):
        def _boom():
            raise RuntimeError("API not responding")

        state = StateManager(tmp_path / "state.db")
        client = SimpleNamespace(connected=True, get_open_orders=_boom)
        orch = self._orch(state, client)

        assert orch._duplicate_guard_verdict(_ic_signal()) == "unverified"
        assert _named(events, "working_order_check_failed_entry_refused")

    def test_recent_same_symbol_strategy_is_duplicate(self, tmp_path):
        state = StateManager(tmp_path / "state.db")
        _seed_trade(state, minutes_ago=5)
        orch = self._orch(state, FakeIBKRClient(FakeIB()))
        assert orch._duplicate_guard_verdict(_ic_signal()) == "duplicate"

    def test_old_trade_outside_cooldown_is_clear(self, tmp_path):
        state = StateManager(tmp_path / "state.db")
        _seed_trade(state, minutes_ago=200)  # cooldown is 120 min
        orch = self._orch(state, FakeIBKRClient(FakeIB()))
        assert orch._duplicate_guard_verdict(_ic_signal()) == "clear"

    def test_working_broker_order_is_duplicate(self, tmp_path):
        state = StateManager(tmp_path / "state.db")
        ib = FakeIB()
        ib.placeOrder(SimpleNamespace(secType="BAG", symbol="SPY"),
                      SimpleNamespace(orderId=77, lmtPrice=-1.0, totalQuantity=1))
        orch = self._orch(state, FakeIBKRClient(ib))
        assert orch._duplicate_guard_verdict(_ic_signal()) == "duplicate"

    def test_in_memory_pending_is_duplicate(self, tmp_path):
        state = StateManager(tmp_path / "state.db")
        pending = {1: SimpleNamespace(signal=SimpleNamespace(symbol="SPY"))}
        orch = self._orch(state, FakeIBKRClient(FakeIB()), pending=pending)
        assert orch._duplicate_guard_verdict(_ic_signal()) == "duplicate"


# ===========================================================================
# fail-direction-09 — PDT guard
# ===========================================================================
def _trading_days(n: int = 5) -> list[date]:
    out: list[date] = []
    d = date.today()
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    out.reverse()
    return out


class TestPdtGuardFailsClosed:
    """A guard whose own inputs are broken must BLOCK: a PDT violation
    freezes the account for 90 days and cannot be unwound."""

    def _guard(self, state, monkeypatch, days=None, boom=False,
               config: AccountConfig | None = None) -> PDTGuard:
        def _cal(_n):
            if boom:
                raise RuntimeError("market calendar unavailable")
            return list(days if days is not None else _trading_days())

        monkeypatch.setattr("ait.risk.pdt_guard.get_recent_trading_days", _cal)
        cfg = config or AccountConfig(pdt_protection=True,
                                      pdt_account_under_25k=True)
        return PDTGuard(cfg, state)

    def test_healthy_zero_count_still_allows(self, tmp_path, monkeypatch):
        state = StateManager(tmp_path / "state.db")
        guard = self._guard(state, monkeypatch)
        assert guard.can_day_trade() is True
        assert guard.get_status().day_trades_used == 0

    def test_corrupt_counter_blocks(self, tmp_path, monkeypatch, events):
        state = StateManager(tmp_path / "state.db")
        state.set_state("pdt_day_trades", "{corrupt json")
        guard = self._guard(state, monkeypatch)

        # Pre-fix: silent reset to deque() -> 0 used -> can_day_trade True.
        assert guard.can_day_trade() is False
        assert _named(events, "pdt_state_unreadable_fail_closed")
        status = guard.get_status()
        assert isinstance(status, PDTStatus)
        assert status.can_day_trade is False

    def test_unreadable_state_store_blocks(self, tmp_path, monkeypatch):
        def _locked(_key, _default=""):
            raise sqlite3.OperationalError("database is locked")

        state = SimpleNamespace(get_state=_locked, set_state=lambda *a: None)
        guard = self._guard(state, monkeypatch)
        assert guard.can_day_trade() is False

    def test_empty_calendar_blocks_even_with_no_recorded_trades(
            self, tmp_path, monkeypatch, events):
        state = StateManager(tmp_path / "state.db")
        guard = self._guard(state, monkeypatch, days=[])

        # Pre-fix: _count_in_window returned 0 for an empty calendar, so the
        # guard read "0 of 3 used" and allowed every day trade.
        assert guard.can_day_trade() is False
        assert _named(events, "pdt_calendar_empty_fail_closed")

    def test_empty_calendar_counts_every_recorded_trade(self, tmp_path,
                                                        monkeypatch):
        state = StateManager(tmp_path / "state.db")
        state.set_state("pdt_day_trades", json.dumps(
            [{"date": d.isoformat(), "symbol": s}
             for d, s in zip(_trading_days(3), ["SPY", "QQQ", "IWM"])]))
        guard = self._guard(state, monkeypatch, days=[])

        assert guard.can_day_trade() is False
        # conservative reading: all 3 are assumed in-window, not zero
        assert guard.get_status().day_trades_used == 3

    def test_calendar_exception_blocks(self, tmp_path, monkeypatch, events):
        state = StateManager(tmp_path / "state.db")
        guard = self._guard(state, monkeypatch, boom=True)
        assert guard.can_day_trade() is False
        assert _named(events, "pdt_calendar_unavailable_fail_closed")

    def test_calendar_recovery_restores_allow(self, tmp_path, monkeypatch):
        # The calendar flag is recomputed per check (only the corrupt-state
        # flag is sticky), so a transient calendar fault does not wedge the
        # guard shut for the session.
        state = StateManager(tmp_path / "state.db")
        guard = self._guard(state, monkeypatch, days=[])
        assert guard.can_day_trade() is False

        monkeypatch.setattr("ait.risk.pdt_guard.get_recent_trading_days",
                            lambda _n: _trading_days())
        assert guard.can_day_trade() is True

    def test_disabled_guard_unaffected(self, tmp_path, monkeypatch):
        # Today's live config (pdt_account_under_25k=false, $198k account):
        # the guard is OFF and a broken calendar must not start blocking.
        state = StateManager(tmp_path / "state.db")
        state.set_state("pdt_day_trades", "{corrupt json")
        guard = self._guard(
            state, monkeypatch, boom=True,
            config=AccountConfig(pdt_protection=True,
                                 pdt_account_under_25k=False))
        assert guard.can_day_trade() is True
        assert guard.get_status().enabled is False

    def test_three_day_trades_still_block(self, tmp_path, monkeypatch):
        # the ordinary rule still holds on healthy inputs
        state = StateManager(tmp_path / "state.db")
        days = _trading_days()
        state.set_state("pdt_day_trades", json.dumps(
            [{"date": days[-1].isoformat(), "symbol": s}
             for s in ("SPY", "QQQ", "IWM")]))
        guard = self._guard(state, monkeypatch, days=days)
        assert guard.can_day_trade() is False
        assert guard.get_status().day_trades_remaining == 0


# ===========================================================================
# fail-direction-11 — restricted / ban list
# ===========================================================================
class TestRestrictedListReader:
    def _write(self, tmp_path, data: bytes) -> str:
        p = tmp_path / "RESTRICTED.txt"
        p.write_bytes(data)
        return str(p)

    def test_absent_file_means_no_restrictions(self, tmp_path):
        assert read_restricted_symbols(str(tmp_path / "nope.txt")) == set()

    def test_plain_utf8_parses(self, tmp_path):
        path = self._write(tmp_path, b"SPY\nqqq\n\n")
        assert read_restricted_symbols(path) == {"SPY", "QQQ"}

    def test_bom_file_parses_as_the_symbol(self, tmp_path):
        # Pre-fix (plain read_text on cp1252/utf-8): {'﻿SPY'} — 'SPY'
        # was NOT in the banned set and nothing said so.
        path = self._write(tmp_path, "SPY\n".encode("utf-8-sig"))
        assert read_restricted_symbols(path) == {"SPY"}

    def test_powershell_utf16_file_parses(self, tmp_path):
        # `echo SPY > data\RESTRICTED.txt` in Windows PowerShell 5.1.
        path = self._write(tmp_path, "SPY\n".encode("utf-16"))
        assert read_restricted_symbols(path) == {"SPY"}

    def test_undecodable_file_returns_the_unreadable_sentinel(self, tmp_path,
                                                              events):
        path = self._write(tmp_path, b"\xff\xfe\x00\x81SPY\x81\xff")
        assert read_restricted_symbols(path) is RESTRICTED_UNREADABLE
        assert _named(events, "restricted_list_unreadable_fail_closed")

    def test_unreadable_path_returns_the_sentinel(self, tmp_path, events):
        # A directory where the ban file should be: exists, cannot be read.
        d = tmp_path / "RESTRICTED.txt"
        d.mkdir()
        assert read_restricted_symbols(str(d)) is RESTRICTED_UNREADABLE
        assert _named(events, "restricted_list_unreadable_fail_closed")


class TestRestrictedListCaller:
    """The caller must refuse NEW entries while the list cannot be read."""

    class _Reached(Exception):
        """Raised past the ban gate to prove the entry was NOT blocked."""

    def _orch(self):
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._learning = SimpleNamespace(adaptor=None)
        o._economic_cal = None          # skips the pre-event blackout block
        o._post_stop_cooldown_until = lambda _sym: None
        o._executor = SimpleNamespace(_pending_orders={})

        def _budget():
            raise TestRestrictedListCaller._Reached()

        o._get_trade_budget = _budget
        return o

    def _cwd(self, tmp_path, monkeypatch, data: bytes | None):
        (tmp_path / "data").mkdir()
        if data is not None:
            (tmp_path / "data" / "RESTRICTED.txt").write_bytes(data)
        monkeypatch.chdir(tmp_path)

    async def test_unreadable_list_blocks_new_entries(self, tmp_path,
                                                      monkeypatch, events):
        self._cwd(tmp_path, monkeypatch, b"\xff\xfe\x00\x81SPY\x81\xff")
        handled = await self._orch()._try_execute(_ic_signal(), 0.72, None, None)

        assert handled is True, "an unreadable ban file must block entries"
        assert _named(events, "entries_halted_restricted_list_unreadable")

    async def test_listed_symbol_blocked(self, tmp_path, monkeypatch, events):
        self._cwd(tmp_path, monkeypatch, "SPY\n".encode("utf-8-sig"))
        handled = await self._orch()._try_execute(_ic_signal(), 0.72, None, None)

        assert handled is True
        assert _named(events, "symbol_restricted")

    async def test_clean_list_without_the_symbol_lets_the_entry_through(
            self, tmp_path, monkeypatch):
        self._cwd(tmp_path, monkeypatch, b"IWM\n")
        with pytest.raises(TestRestrictedListCaller._Reached):
            await self._orch()._try_execute(_ic_signal(), 0.72, None, None)

    async def test_absent_list_lets_the_entry_through(self, tmp_path,
                                                      monkeypatch):
        self._cwd(tmp_path, monkeypatch, None)
        with pytest.raises(TestRestrictedListCaller._Reached):
            await self._orch()._try_execute(_ic_signal(), 0.72, None, None)


# ===========================================================================
# fail-direction-12 — executor market-hours guard
# ===========================================================================
class TestMarketHoursGuard:
    async def test_calendar_exception_refuses_the_order(self, env, monkeypatch,
                                                        events):
        monkeypatch.delenv("AIT_ALLOW_AFTER_HOURS", raising=False)

        def _boom():
            raise RuntimeError("pandas_market_calendars: schedule() failed")

        monkeypatch.setattr("ait.utils.time.is_market_open", _boom)
        sig = _ic_signal()
        env.ib.quote = _nbbo_for(sig)

        # Pre-fix: bare `except: pass` — a DAY combo went to the wire with no
        # record the guard had been bypassed.
        assert await env.executor.execute_signal(sig, 1) is None
        assert env.ib.placed == []
        failed = _named(events, "market_hours_guard_failed")
        assert len(failed) == 1 and failed[0]["log_level"] == "error"

    async def test_closed_market_still_refuses(self, env, monkeypatch, events):
        monkeypatch.delenv("AIT_ALLOW_AFTER_HOURS", raising=False)
        monkeypatch.setattr("ait.utils.time.is_market_open", lambda: False)
        assert await env.executor.execute_signal(_ic_signal(), 1) is None
        assert _named(events, "order_refused_market_closed")

    async def test_open_market_places(self, env, monkeypatch, events):
        monkeypatch.delenv("AIT_ALLOW_AFTER_HOURS", raising=False)
        monkeypatch.setattr("ait.utils.time.is_market_open", lambda: True)
        sig = _ic_signal()
        env.ib.quote = _nbbo_for(sig)

        assert await env.executor.execute_signal(sig, 1) is not None
        assert len(env.ib.placed) == 1
        assert _named(events, "market_hours_guard_failed") == []


# ===========================================================================
# trade-life-credit-econ-floors-signal-time-only — floors at the reprice
# ===========================================================================
@pytest.fixture
def floors(monkeypatch):
    """Pin the contract floors so the test does not drift with config.yaml."""
    monkeypatch.setenv("AIT_IC_MIN_CREDIT", "0.70")
    monkeypatch.setenv("AIT_IC_MIN_CREDIT_WIDTH", "0.10")


class TestCreditFloorsAtReprice:
    def _executor(self, state=None):
        ex = TradeExecutor.__new__(TradeExecutor)
        ex._pending_orders = {}
        ex._ibkr = SimpleNamespace(ib=FakeIB())
        ex._state = state
        return ex

    def _arm(self, ex, signal, *, base: float, offset: float, age: float,
             is_credit: bool = True, limit: float = 0.0,
             trade_id: str = "T-W2-LADDER") -> tuple[int, object]:
        order = SimpleNamespace(orderId=4242, lmtPrice=limit, totalQuantity=1)
        ex._ibkr.ib.placeOrder(
            SimpleNamespace(secType="BAG", symbol=signal.symbol), order)
        pending = PendingOrder(trade_id=trade_id, signal=signal, contracts=1,
                               base_price=base, full_offset=offset,
                               is_credit=is_credit)
        pending.submitted_at = time.time() - age
        ex._pending_orders[order.orderId] = pending
        return order, pending

    async def test_min_credit_floor_refuses_the_remaining_ladder(
            self, floors, events):
        # Signal cleared 0.75 on $3 wings; step 1 of the ladder would concede
        # to 0.63 — below the AIT_IC_MIN_CREDIT floor that justified entry.
        ex = self._executor()
        sig = _ic_signal(credit=0.75, width=3.0)
        order, pending = self._arm(ex, sig, base=0.75, offset=0.20, age=50,
                                   limit=-0.75)
        await ex._reprice_pending_entries()

        assert order.lmtPrice == -0.75, "the resting price must not move"
        assert len(ex._ibkr.ib.placed) == 1, "no modification may be sent"
        breach = _named(events, "credit_floor_breached_at_reprice")
        assert len(breach) == 1 and breach[0]["log_level"] == "warning"
        assert breach[0]["credit"] == pytest.approx(0.63)
        # the REMAINDER of the ladder is refused, not just this step
        assert pending.step == len(TradeExecutor._LADDER_STEPS) - 1

    async def test_ratio_floor_refuses_even_when_credit_clears(self, floors,
                                                              events):
        # The finding's worked example: 1.39 credit on $14 wings = 0.099 of
        # width, under the 0.10 ratio floor, while the absolute credit floor
        # (0.70) is comfortably cleared.
        ex = self._executor()
        sig = _ic_signal(credit=2.20, width=14.0)
        order, pending = self._arm(ex, sig, base=1.63, offset=0.24, age=100,
                                   limit=-1.55)
        await ex._reprice_pending_entries()

        assert order.lmtPrice == -1.55
        breach = _named(events, "credit_floor_breached_at_reprice")
        assert len(breach) == 1
        assert breach[0]["credit"] == pytest.approx(1.39)
        assert breach[0]["width"] == pytest.approx(14.0)
        assert breach[0]["ratio"] == pytest.approx(0.0993, abs=1e-3)

    async def test_compliant_credit_still_escalates(self, floors, events):
        ex = self._executor()
        sig = _ic_signal(credit=1.20, width=3.0)
        order, _p = self._arm(ex, sig, base=1.20, offset=0.18, age=50,
                              limit=-1.20)
        await ex._reprice_pending_entries()

        assert order.lmtPrice == pytest.approx(-1.09)
        assert len(ex._ibkr.ib.placed) == 2, "the modification was sent"
        assert _named(events, "entry_ladder_reprice")
        assert _named(events, "credit_floor_breached_at_reprice") == []

    async def test_debit_ladder_is_untouched(self, floors, events):
        # ENTRY credit orders only — the debit side keeps its R8 max_loss cap
        # and never consults the credit floors.
        ex = self._executor()
        sig = _ic_signal(credit=2.00, width=5.0)
        sig.strategy_name = "bull_call_spread"
        order, _p = self._arm(ex, sig, base=2.00, offset=0.30, age=50,
                              is_credit=False, limit=2.00)
        await ex._reprice_pending_entries()

        assert order.lmtPrice == pytest.approx(2.18)
        assert _named(events, "credit_floor_breached_at_reprice") == []

    def test_width_from_signal_legs(self, floors):
        ex = self._executor()
        assert ex._pending_wing_width(
            SimpleNamespace(trade_id="T", signal=_ic_signal(width=14.0))
        ) == pytest.approx(14.0)

    def test_width_falls_back_to_the_pending_legs_json(self, tmp_path, floors):
        # The in-memory signal can carry no legs (e.g. a rebuilt tracker);
        # the PENDING row's legs JSON is the same source signal time used.
        state = StateManager(tmp_path / "state.db")
        legs = json.dumps([
            {"strike": 86.0, "right": "P", "action": "BUY", "expiry": EXPIRY_ISO},
            {"strike": 100.0, "right": "P", "action": "SELL", "expiry": EXPIRY_ISO},
            {"strike": 110.0, "right": "C", "action": "SELL", "expiry": EXPIRY_ISO},
            {"strike": 124.0, "right": "C", "action": "BUY", "expiry": EXPIRY_ISO},
        ])
        with sqlite3.connect(state._db_path) as conn:
            conn.execute(
                "INSERT INTO trades (trade_id, symbol, strategy, direction, "
                "status, entry_time, entry_price, quantity, contract_type, "
                "legs) VALUES (?,?,?,?,?,?,?,1,'iron_condor',?)",
                ("T-LEGS", "SPY", "iron_condor", "short", "pending",
                 datetime.now().isoformat(), 1.39, legs))

        ex = self._executor(state=state)
        pending = SimpleNamespace(trade_id="T-LEGS",
                                  signal=SimpleNamespace(legs=[]))
        assert ex._pending_wing_width(pending) == pytest.approx(14.0)
        assert ex._credit_floors_hold(pending, -1.39) is False
        assert ex._credit_floors_hold(pending, -1.45) is True

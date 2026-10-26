"""W101-C: an expired condor is booked at its settlement, to its own session.

QQQ condor T-20260928-195224-adfe01 (BUY 704P / SELL 707P / SELL 769C / BUY 772C,
credit 0.92, expiry 2026-10-23) expired with QQQ closing 704.40: the short put
2.60 in the money, the long put out of it. At the next start the reconciler used
to book it from a LIVE snapshot of QQQ (whatever it traded at then), dated now,
so the loss landed on that day's circuit breaker, and the commission true-up
waited forever for exit fills that cannot exist.

Every test drives the real ``_book_stale_trade`` / ``close_trade`` /
``_drain_exit_outbox`` / ``_true_up_commission``.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.execution.reconciler import PositionReconciler
from ait.utils.time import now_et
from tests.test_w73_live_dte_and_clock_fixes import ET, _freeze

EXPIRY = "2026-10-23"
MONDAY = datetime(2026, 10, 26, 9, 0, tzinfo=ET)
LEGS = [
    {"strike": 704.0, "right": "P", "action": "BUY", "expiry": EXPIRY},
    {"strike": 707.0, "right": "P", "action": "SELL", "expiry": EXPIRY},
    {"strike": 769.0, "right": "C", "action": "SELL", "expiry": EXPIRY},
    {"strike": 772.0, "right": "C", "action": "BUY", "expiry": EXPIRY},
]


def _trade(expiry: str = EXPIRY):
    return SimpleNamespace(
        trade_id="T-ADFE01", symbol="QQQ", strategy="iron_condor",
        contract_type="iron_condor", entry_price=0.92, quantity=1, expiry=expiry,
        direction=None, strike=None, legs=json.dumps(LEGS))


def _rec(*, settlement=704.4, live=705.0, stock=None, portfolio=()):
    rec = PositionReconciler.__new__(PositionReconciler)
    rec._ibkr = MagicMock()
    rec._ibkr.get_portfolio.return_value = list(portfolio)
    rec._state = MagicMock()
    rec._underlying_last = AsyncMock(return_value=live)
    calls = []

    def _source(symbol, day):
        calls.append((symbol, day))
        if isinstance(settlement, Exception):
            raise settlement
        return settlement

    if settlement is not False:
        rec.set_settlement_source(_source)
    rec._source_calls = calls
    rec._live_ibkr_leg_keys = lambda: ({"QQQ:STK": stock} if stock is not None else {})
    return rec


def _book(rec, trade=None):
    asyncio.run(rec._book_stale_trade(trade or _trade(), list(rec._ibkr.get_portfolio())))
    return rec._state.close_trade.call_args.kwargs


def _state_keys(rec):
    return {c.args[0]: c.args[1] for c in rec._state.set_state.call_args_list}


@pytest.fixture(autouse=True)
def _monday(monkeypatch):
    _freeze(monkeypatch, MONDAY)


# --------------------------------------------------------------------------
# the valuation
# --------------------------------------------------------------------------
class TestSettlementValuation:
    def test_the_incident_books_at_the_settlement_close_not_a_live_snapshot(self):
        rec = _rec(settlement=704.4, live=731.0)           # QQQ moved a lot since
        kw = _book(rec)
        assert kw["exit_reason_detailed"] == "reconciler_expired_intrinsic"
        assert kw["exit_price"] == pytest.approx(2.60)
        assert kw["realized_pnl"] == pytest.approx(-168.00)
        rec._underlying_last.assert_not_called()
        assert rec._source_calls == [("QQQ", date(2026, 10, 23))]

    @pytest.mark.parametrize("settle,pnl", [
        (703.0, -208.0),        # through both puts: the structure's max loss
        (720.0, 92.0),          # between the short strikes: the full credit
        (771.0, -108.0),        # call side: the 769 short owes 2.00, the 772 long is out of the money
    ])
    def test_the_other_settlements(self, settle, pnl):
        kw = _book(_rec(settlement=settle))
        assert kw["realized_pnl"] == pytest.approx(pnl)

    def test_the_brokers_per_leg_pnl_is_used_first_and_a_difference_is_logged(self, monkeypatch):
        """Operator decision: a usable broker figure keeps precedence. After an
        assignment it can be a partial number, so the settlement value is logged
        next to it when they differ."""
        import ait.execution.reconciler as rec_mod
        log = MagicMock()
        monkeypatch.setattr(rec_mod, "log", log)
        keys = [("P", 704.0), ("P", 707.0), ("C", 769.0), ("C", 772.0)]
        items = [SimpleNamespace(
            contract=SimpleNamespace(secType="OPT", symbol="QQQ", strike=k, right=r,
                                     lastTradeDateOrContractMonth="20261023"),
            position=0, realizedPNL=25.0) for r, k in keys]
        kw = _book(_rec(settlement=704.4, portfolio=items))
        assert kw["exit_reason_detailed"] == "reconciler_ibkr_realized"
        assert kw["realized_pnl"] == pytest.approx(100.0)
        assert kw["exit_time"] == "2026-10-23T16:00:00-04:00"        # still its own session
        events = [c.args[0] for c in log.warning.call_args_list]
        assert "reconcile_expired_broker_pnl_differs_from_settlement" in events
        differs = next(c for c in log.warning.call_args_list
                       if c.args[0] == "reconcile_expired_broker_pnl_differs_from_settlement")
        assert differs.kwargs["settlement_value"] == pytest.approx(-168.0)

    def test_a_broker_figure_that_agrees_with_the_settlement_is_not_flagged(self, monkeypatch):
        import ait.execution.reconciler as rec_mod
        log = MagicMock()
        monkeypatch.setattr(rec_mod, "log", log)
        keys = [("P", 704.0), ("P", 707.0), ("C", 769.0), ("C", 772.0)]
        items = [SimpleNamespace(
            contract=SimpleNamespace(secType="OPT", symbol="QQQ", strike=k, right=r,
                                     lastTradeDateOrContractMonth="20261023"),
            position=0, realizedPNL=(-168.0 if i == 0 else 0.0)) for i, (r, k) in enumerate(keys)]
        kw = _book(_rec(settlement=704.4, portfolio=items))
        assert kw["realized_pnl"] == pytest.approx(-168.0)
        assert "reconcile_expired_broker_pnl_differs_from_settlement" not in [
            c.args[0] for c in log.warning.call_args_list]


class TestWhenTheSettlementIsNotThere:
    def test_a_missing_row_falls_back_to_the_live_snapshot_and_says_so(self):
        rec = _rec(settlement=None, live=705.0)
        kw = _book(rec)
        assert kw["exit_reason_detailed"] == "reconciler_expired_intrinsic"
        assert kw["exit_price"] == pytest.approx(2.0)      # the live snapshot, 705.0
        assert "expiry_settle_unverified_T-ADFE01" in _state_keys(rec)
        assert kw["exit_time"].startswith("2026-10-23T16:00:00")   # still dated to its session

    def test_a_source_that_raises_is_a_missing_row(self):
        rec = _rec(settlement=RuntimeError("db locked"), live=705.0)
        kw = _book(rec)
        assert kw["exit_price"] == pytest.approx(2.0)
        assert "expiry_settle_unverified_T-ADFE01" in _state_keys(rec)

    @pytest.mark.parametrize("bad", [float("nan"), 0.0, -3.0, float("inf"), "x"])
    def test_a_value_that_is_not_a_price_is_ignored(self, bad):
        rec = _rec(settlement=bad, live=705.0)
        kw = _book(rec)
        assert kw["exit_price"] == pytest.approx(2.0)
        assert "expiry_settle_unverified_T-ADFE01" in _state_keys(rec)

    def test_no_source_wired_is_the_old_behaviour_plus_the_flag(self):
        rec = _rec(settlement=False, live=705.0)
        kw = _book(rec)
        assert kw["exit_price"] == pytest.approx(2.0)
        assert "expiry_settle_unverified_T-ADFE01" in _state_keys(rec)

    def test_no_price_at_all_is_not_invented(self):
        rec = _rec(settlement=None, live=None)
        kw = _book(rec)
        assert kw["realized_pnl"] == 0.0
        assert "needs_review" in kw["exit_reason_detailed"] or "unknown" in kw["exit_reason_detailed"]


class TestTheExpiryDay:
    def test_before_the_close_the_settlement_is_not_due(self, monkeypatch):
        """On the expiry day itself, before 16:15 ET, the day's bar is partial:
        the source is not asked, the live snapshot is the right input, and the
        booking is not flagged."""
        _freeze(monkeypatch, datetime(2026, 10, 23, 14, 0, tzinfo=ET))
        rec = _rec(settlement=704.4, live=705.0)
        kw = _book(rec)
        assert rec._source_calls == []
        assert kw["exit_price"] == pytest.approx(2.0)
        assert "expiry_settle_unverified_T-ADFE01" not in _state_keys(rec)
        assert "exit_time" not in kw                          # today's close, dated now

    def test_after_the_close_the_settlement_is_used_and_the_booking_is_today(self, monkeypatch):
        _freeze(monkeypatch, datetime(2026, 10, 23, 16, 20, tzinfo=ET))
        rec = _rec(settlement=704.4, live=731.0)
        kw = _book(rec)
        assert rec._source_calls == [("QQQ", date(2026, 10, 23))]
        assert kw["exit_price"] == pytest.approx(2.60)
        assert "exit_time" not in kw


class TestTheBookingCarriesItsSession:
    def test_the_close_is_dated_to_the_expiry_close(self):
        kw = _book(_rec())
        assert kw["exit_time"] == "2026-10-23T16:00:00-04:00"

    def test_a_trade_that_vanished_before_expiry_keeps_the_old_dating(self, monkeypatch):
        _freeze(monkeypatch, datetime(2026, 10, 20, 11, 0, tzinfo=ET))
        kw = _book(_rec(settlement=None))
        assert "exit_time" not in kw

    def test_the_commission_marker_says_none_was_subtracted(self):
        rec = _rec()
        _book(rec)
        assert _state_keys(rec).get("commission_est_T-ADFE01") == "0.0"


class TestTheAssignmentIsLinkedNotAccounted:
    def test_100_shares_for_a_short_put_that_expired_in_the_money(self):
        rec = _rec(stock=100)
        _book(rec)
        link = json.loads(_state_keys(rec)["assignment_link_T-ADFE01"])
        assert link["stock_key"] == "QQQ:STK" and link["shares"] == 100
        assert link["short_legs_itm"] == [{"strike": 707.0, "right": "P"}]

    @pytest.mark.parametrize("stock", [None, 50, 200])
    def test_no_link_unless_the_shares_match(self, stock):
        rec = _rec(stock=stock)
        _book(rec)
        assert "assignment_link_T-ADFE01" not in _state_keys(rec)

    def test_no_link_when_nothing_expired_in_the_money(self):
        rec = _rec(settlement=720.0, stock=100)
        _book(rec)
        assert "assignment_link_T-ADFE01" not in _state_keys(rec)


# --------------------------------------------------------------------------
# the state layer
# --------------------------------------------------------------------------
def _seed(st, tid="T-ADFE01"):
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, status, "
            "entry_time, entry_price, quantity, contract_type, expiry, legs) "
            "VALUES (?,?,?,?,?,?,?,1,'iron_condor',?,?)",
            (tid, "QQQ", "iron_condor", "short", "filled",
             "2026-09-28T13:54:16-04:00", 0.92, EXPIRY, json.dumps(LEGS)))


class TestCloseTradeExitTime:
    def test_exit_time_reaches_the_trade_and_the_outbox(self, tmp_path):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        _seed(st)
        assert st.close_trade("T-ADFE01", 2.6, -168.0,
                              exit_reason_detailed="reconciler_expired_intrinsic",
                              exit_time="2026-10-23T16:00:00-04:00") is True
        with sqlite3.connect(st._db_path) as conn:
            t = conn.execute("SELECT exit_time FROM trades WHERE trade_id='T-ADFE01'").fetchone()[0]
            o = conn.execute("SELECT exit_time FROM exit_outbox WHERE trade_id='T-ADFE01'").fetchone()[0]
        assert t == o == "2026-10-23T16:00:00-04:00"

    def test_the_default_is_still_now(self, tmp_path):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        _seed(st)
        st.close_trade("T-ADFE01", 2.6, -168.0)
        with sqlite3.connect(st._db_path) as conn:
            t = conn.execute("SELECT exit_time FROM trades WHERE trade_id='T-ADFE01'").fetchone()[0]
        assert t.startswith(now_et().date().isoformat())


# --------------------------------------------------------------------------
# the settlement reader
# --------------------------------------------------------------------------
class TestTheSettlementReader:
    def _db(self, tmp_path, rows):
        db = tmp_path / "historical.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE daily_prices (symbol TEXT, date TEXT, close REAL)")
            conn.executemany("INSERT INTO daily_prices VALUES (?,?,?)", rows)
        return db

    def test_the_row_for_exactly_that_date(self, tmp_path, monkeypatch):
        import ait.data.historical as h
        monkeypatch.setattr(h, "DB_PATH", self._db(tmp_path, [
            ("QQQ", "2026-10-22", 706.0), ("QQQ", "2026-10-23", 704.4), ("SPY", "2026-10-23", 760.0)]))
        read = TradingOrchestrator._read_settlement_close
        assert read("QQQ", date(2026, 10, 23)) == pytest.approx(704.4)
        assert read("QQQ", date(2026, 10, 24)) is None
        assert read("IWM", date(2026, 10, 23)) is None

    @pytest.mark.parametrize("close", [None, 0.0, -1.0, float("nan")])
    def test_a_close_that_is_not_a_price_is_none(self, tmp_path, monkeypatch, close):
        import ait.data.historical as h
        monkeypatch.setattr(h, "DB_PATH", self._db(tmp_path, [("QQQ", "2026-10-23", close)]))
        assert TradingOrchestrator._read_settlement_close("QQQ", date(2026, 10, 23)) is None

    def test_it_never_creates_the_database(self, tmp_path, monkeypatch):
        import ait.data.historical as h
        missing = tmp_path / "nope" / "historical.db"
        monkeypatch.setattr(h, "DB_PATH", missing)
        assert TradingOrchestrator._read_settlement_close("QQQ", date(2026, 10, 23)) is None
        assert not missing.exists() and not missing.parent.exists()


# --------------------------------------------------------------------------
# end to end: reconcile-book, drain, true-up
# --------------------------------------------------------------------------
class TestEndToEnd:
    def _world(self, tmp_path, commissions=(0.62, 0.63, 0.62, 0.63)):
        st = StateManager(db_path=tmp_path / "ait_state.db")
        _seed(st)
        for i, c in enumerate(commissions):
            st.record_execution(f"E{i}", 467611, 1, "T-ADFE01", "QQQ", 1000 + i, "BOT", 1.0,
                                1.0, "2026-09-28T15:54:15+00:00", commission=c)
        st.record_execution("EB", 467611, 1, "T-ADFE01", "QQQ", 28812380, "BOT", 1.0,
                            0.92, "2026-09-28T15:54:15+00:00", commission=0.0)   # the BAG summary row
        rec = PositionReconciler.__new__(PositionReconciler)
        rec._ibkr = MagicMock()
        rec._ibkr.get_portfolio.return_value = []
        rec._state = st
        rec._underlying_last = AsyncMock(return_value=731.0)
        rec.set_settlement_source(lambda sym, d: 704.4)
        rec._live_ibkr_leg_keys = lambda: {"QQQ:STK": 100}
        from ait.risk.circuit_breaker import CircuitBreaker
        o = TradingOrchestrator.__new__(TradingOrchestrator)
        o._state = st
        o._circuit_breaker = CircuitBreaker(SimpleNamespace(
            max_daily_loss_pct=0.2, max_consecutive_losses=3,
            pause_minutes_after_losses=30, max_api_failures=5))
        o._pdt_guard = MagicMock()
        o._thompson = MagicMock()
        o._trainer = MagicMock()
        o._send_notification = AsyncMock()
        return st, rec, o

    def test_the_loss_lands_on_the_expiry_session_with_the_ledgers_commission(self, tmp_path):
        st, rec, o = self._world(tmp_path)
        asyncio.run(rec._book_stale_trade(st.get_trade_by_id("T-ADFE01"), []))
        asyncio.run(o._drain_exit_outbox())

        t = st.get_trade_by_id("T-ADFE01")
        assert t.status.value == "closed"
        assert t.exit_reason_detailed == "reconciler_expired_intrinsic"
        assert t.exit_time.startswith("2026-10-23T16:00:00")
        assert t.realized_pnl == pytest.approx(-170.50)       # -168.00 less the 2.50 entry commission
        assert t.commission == pytest.approx(2.50)
        assert st.get_daily_stats(date(2026, 10, 23)).total_pnl == pytest.approx(-170.50)
        assert st.get_daily_stats(MONDAY.date()).total_pnl == pytest.approx(0.0)
        assert st.get_state("trueup_pending_T-ADFE01", "") == ""
        assert st.get_state("assignment_link_T-ADFE01", "") != ""
        o._thompson.record_outcome.assert_called_once()

    def test_negative_control_a_trade_closed_by_its_own_exit_still_needs_the_exit_rows(self, tmp_path):
        """Entry rows alone are not a complete ledger for an ordinary close."""
        st, rec, o = self._world(tmp_path)
        st.close_trade("T-ADFE01", 1.5, 58.0, exit_reason_detailed="take_profit_short (P&L: 41%)")
        o._booked_commission_estimate = lambda trade: 5.0
        pnl = o._true_up_commission("T-ADFE01", 58.0)
        assert pnl == pytest.approx(58.0)                      # deferred, not corrected from half a ledger
        assert st.get_state("trueup_pending_T-ADFE01", "") != ""

    def test_an_expired_booking_without_the_marker_keeps_the_strict_gate(self, tmp_path):
        st, rec, o = self._world(tmp_path)
        st.close_trade("T-ADFE01", 2.6, -168.0, exit_reason_detailed="reconciler_expired_intrinsic")
        t = st.get_trade_by_id("T-ADFE01")
        assert o._expected_commission_rows(t) == 8
        st.set_state("commission_est_T-ADFE01", "0.0")
        assert o._expected_commission_rows(t) == 4


# --------------------------------------------------------------------------
# the real orchestrator wires the source
# --------------------------------------------------------------------------
class TestTheOrchestratorWiresTheSettlementSource:
    def test_the_real_init_hands_the_reconciler_the_reader(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)                         # __init__ writes data/*.db
        from tests.test_hot_path_smoke import build_smoke_orchestrator
        rig = build_smoke_orchestrator()
        try:
            assert rig.orch._reconciler._settlement_close is TradingOrchestrator._read_settlement_close
            # the sandbox store has no such row: the reader answers None, never raises
            assert rig.orch._reconciler._settlement_price("QQQ", date(2026, 10, 23)) is None
        finally:
            rig.restore()

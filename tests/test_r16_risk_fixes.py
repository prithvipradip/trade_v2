"""R16 risk/execution-cluster fixes — each test fails against the pre-fix code.

Covers the seven audit findings owned by the risk/execution cluster:
  [1] check_daily_loss_mtm returned True for ANY active trip (false
      "DAILY MTM LOSS HALT" page whose one-shot latch then silenced the real one)
  [2] CircuitBreaker consecutive-loss streak / active trip died with the process
  [3] aggregate 20% cap counted $0 for any position missing its trade_maxloss_ KV
  [4] reconcile/sweep promotions never inserted the open_positions row
  [5] CLOSING-kept-unknown appended one discrepancy PER LEG (4x for a condor)
  [6] the persist=False report path mutated touch-stop protection state
  [7] delta-breach exit rule 3b is inert and must say so
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.state import TradeStatus
from ait.execution.portfolio import PortfolioManager
from ait.execution.reconciler import PositionReconciler
from ait.risk.circuit_breaker import CircuitBreaker
from ait.risk.manager import RiskManager


def _risk_cfg(**over):
    cfg = SimpleNamespace(
        max_daily_loss_pct=0.02,
        max_consecutive_losses=3,
        pause_minutes_after_losses=30,
        max_api_failures=5,
        min_confidence=0.65,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class _FakeState:
    """In-memory stand-in for StateManager's bot_state KV (never touches DB)."""

    def __init__(self, seed: dict | None = None) -> None:
        self.kv = dict(seed or {})
        self.writes = 0

    def get_state(self, key: str, default: str = "") -> str:
        return self.kv.get(key, default)

    def set_state(self, key: str, value: str) -> None:
        self.kv[key] = value
        self.writes += 1


# --- [1] MTM halt must answer only for its OWN trip -------------------------
class TestMtmVerdictScopedToItsOwnTrip:
    def _breaker(self):
        return CircuitBreaker(_risk_cfg())

    def test_unrelated_trip_does_not_report_mtm_halt(self):
        b = self._breaker()
        b._trip("consecutive_losses (3)", pause_seconds=1800)
        # Day P&L is POSITIVE and yet the pre-fix code returned True here,
        # paging "day P&L ... breached the daily-loss cap".
        assert b.check_daily_loss_mtm(+5000.0, 198_000.0) is False
        # ...and a genuine-looking loss must still not be attributed to MTM
        # while the ACTIVE trip belongs to another rule.
        assert b.check_daily_loss_mtm(-9900.0, 198_000.0) is False
        assert b._trip_reason == "consecutive_losses (3)"

    def test_api_failure_trip_does_not_report_mtm_halt(self):
        b = self._breaker()
        b._trip("api_failures (5 in 10 min)", pause_seconds=600)
        assert b.check_daily_loss_mtm(-9900.0, 198_000.0) is False

    def test_real_mtm_breach_still_trips_and_reports(self):
        b = self._breaker()
        assert b.check_daily_loss_mtm(-9900.0, 198_000.0) is True
        assert CircuitBreaker.MTM_TRIP_TOKEN in b._trip_reason
        # Re-asked while its own halt is in force: still True (the halt is real).
        assert b.check_daily_loss_mtm(-9900.0, 198_000.0) is True

    def test_mtm_reason_still_clears_on_daily_reset(self):
        # R15 #3 regression guard: the token must keep matching
        # check_daily_reset's 'daily_loss' untrip matcher.
        b = self._breaker()
        assert b.check_daily_loss_mtm(-9900.0, 198_000.0) is True
        b._last_reset_date = date.today() - timedelta(days=1)
        b.check_daily_reset()
        assert b._tripped is False

    def test_no_trip_and_bad_account_value_is_false(self):
        b = self._breaker()
        assert b.check_daily_loss_mtm(-9900.0, 0.0) is False


# --- [2] the breaker must survive a process restart -------------------------
class TestCircuitBreakerRestartPersistence:
    def test_consecutive_losses_survive_restart(self):
        st = _FakeState()
        b1 = CircuitBreaker(_risk_cfg(), state=st)
        b1.record_trade_result(-100.0)
        b1.record_trade_result(-100.0)
        assert b1._consecutive_losses == 2 and not b1._tripped

        # keeper kills python; a fresh process builds a fresh breaker
        b2 = CircuitBreaker(_risk_cfg(), state=st)
        assert b2._consecutive_losses == 2  # pre-fix: 0
        # the third loss must trip, exactly as it would have without the crash
        b2.record_trade_result(-100.0)
        assert b2.is_tripped is True
        assert "consecutive_losses (3)" in b2._trip_reason

    def test_active_trip_survives_restart_with_its_resume_time(self):
        st = _FakeState()
        b1 = CircuitBreaker(_risk_cfg(), state=st)
        for _ in range(3):
            b1.record_trade_result(-100.0)
        assert b1.is_tripped
        resume = b1._resume_time

        b2 = CircuitBreaker(_risk_cfg(), state=st)
        assert b2.is_tripped is True  # pre-fix: pause silently cleared
        assert b2._resume_time == pytest.approx(resume)
        assert b2.get_status().reason == b1.get_status().reason

    def test_elapsed_pause_auto_resumes_after_restart(self):
        st = _FakeState()
        b1 = CircuitBreaker(_risk_cfg(), state=st)
        for _ in range(3):
            b1.record_trade_result(-100.0)
        b1._resume_time = 1.0  # long past
        b1._persist()
        b2 = CircuitBreaker(_risk_cfg(), state=st)
        assert b2.is_tripped is False
        assert b2._consecutive_losses == 0

    def test_stale_day_rolls_over_on_load(self):
        st = _FakeState()
        b1 = CircuitBreaker(_risk_cfg(), state=st)
        for _ in range(3):
            b1.record_trade_result(-100.0)
        # rewrite the blob as if it were written yesterday
        blob = json.loads(st.kv[CircuitBreaker.STATE_KEY])
        blob["date"] = (date.today() - timedelta(days=1)).isoformat()
        st.kv[CircuitBreaker.STATE_KEY] = json.dumps(blob)

        b2 = CircuitBreaker(_risk_cfg(), state=st)
        assert b2._consecutive_losses == 0  # new day: streak and P&L reset
        assert b2._daily_pnl == 0.0

    def test_manual_reset_survives_restart(self):
        st = _FakeState()
        b1 = CircuitBreaker(_risk_cfg(), state=st)
        for _ in range(3):
            b1.record_trade_result(-100.0)
        b1.manual_reset()
        b2 = CircuitBreaker(_risk_cfg(), state=st)
        assert b2.is_tripped is False and b2._consecutive_losses == 0

    def test_no_state_is_pure_memory_and_never_raises(self):
        b = CircuitBreaker(_risk_cfg())
        b.record_trade_result(-100.0)
        b._persist()  # no-op
        assert b._consecutive_losses == 1

    def test_corrupt_blob_is_survivable(self):
        st = _FakeState({CircuitBreaker.STATE_KEY: "{not json"})
        b = CircuitBreaker(_risk_cfg(), state=st)
        assert b._consecutive_losses == 0 and b.is_tripped is False

    def test_risk_manager_wires_persistence(self):
        # The orchestrator builds the breaker without a state handle
        # (orchestrator.py:85); RiskManager is the seam that attaches it.
        st = _FakeState()
        breaker = CircuitBreaker(_risk_cfg())
        assert getattr(breaker, "_state", None) is None
        RiskManager(
            SimpleNamespace(max_open_positions=5, max_portfolio_risk_pct=0.20,
                            max_portfolio_delta=0.30),
            _risk_cfg(), MagicMock(), breaker, MagicMock(), MagicMock(),
            state=st,
        )
        assert breaker._state is st
        breaker.record_trade_result(-50.0)
        assert CircuitBreaker.STATE_KEY in st.kv


# --- [3] aggregate cap must never count a live position as $0 ---------------
class TestAggregateRiskBackfill:
    def _rm(self, positions, trades=()):
        rm = RiskManager.__new__(RiskManager)
        rm._open_positions = list(positions)
        rm._risk_backfill_warned = set()
        state = MagicMock()
        state.get_open_trades.return_value = list(trades)
        rm._state = state
        return rm

    def _condor_legs(self):
        return json.dumps([
            {"strike": 655.0, "right": "P", "action": "BUY"},
            {"strike": 685.0, "right": "P", "action": "SELL"},
            {"strike": 741.0, "right": "C", "action": "SELL"},
            {"strike": 770.0, "right": "C", "action": "BUY"},
        ])

    def test_kv_present_is_authoritative(self):
        rm = self._rm([{"symbol": "SPY", "strategy": "iron_condor",
                        "max_loss": 3076.0, "quantity": 1}])
        assert rm._aggregate_open_risk() == pytest.approx(3076.0)
        rm._state.get_open_trades.assert_not_called()  # no DB read needed

    def test_missing_kv_falls_back_to_trade_row_capital_at_risk(self):
        trade = SimpleNamespace(symbol="QQQ", strategy="iron_condor",
                                expiry="2026-08-21", quantity=1,
                                capital_at_risk=2383.5, legs=self._condor_legs())
        rm = self._rm([{"symbol": "QQQ", "strategy": "iron_condor",
                        "expiry": "2026-08-21", "max_loss": 0.0, "quantity": 1,
                        "market_value": 616.5}], [trade])
        assert rm._aggregate_open_risk() == pytest.approx(2383.5)  # pre-fix: 0.0

    def test_missing_kv_and_row_falls_back_to_structure_width(self):
        trade = SimpleNamespace(symbol="QQQ", strategy="iron_condor",
                                expiry="2026-08-21", quantity=2,
                                capital_at_risk=0.0, legs=self._condor_legs())
        rm = self._rm([{"symbol": "QQQ", "strategy": "iron_condor",
                        "expiry": "2026-08-21", "max_loss": 0.0, "quantity": 2,
                        "market_value": 1233.0}], [trade])
        # widest wing = 685-655 = 30 -> 30 * 100 * 2 contracts
        assert rm._aggregate_open_risk() == pytest.approx(6000.0)

    def test_no_trade_row_still_never_zero(self):
        rm = self._rm([{"symbol": "IWM", "strategy": "iron_condor",
                        "max_loss": 0.0, "quantity": 1, "market_value": 421.0}])
        assert rm._aggregate_open_risk() == pytest.approx(421.0)  # pre-fix: 0.0

    def test_structural_max_loss_shapes(self):
        f = RiskManager._structural_max_loss
        # vertical put credit spread, 5 wide, 3 contracts
        vertical = json.dumps([
            {"strike": 400.0, "right": "P", "action": "SELL"},
            {"strike": 395.0, "right": "P", "action": "BUY"},
        ])
        assert f(vertical, 3) == pytest.approx(1500.0)
        # naked short: no long on that right -> 20%-of-notional margin proxy
        naked = json.dumps([{"strike": 400.0, "right": "P", "action": "SELL"}])
        assert f(naked, 1) == pytest.approx(8000.0)
        # unusable input degrades to 0 (caller falls through to the notional)
        assert f("", 1) == 0.0
        assert f("[]", 1) == 0.0
        assert f("{bad json", 1) == 0.0

    async def test_gate_6b2_rejects_when_hidden_risk_is_counted(self):
        # The finding's executed scenario: the live two-condor book with both
        # KVs missing approved a new 3k-risk condor. With the backfill the
        # same book now blocks it.
        trades = [
            SimpleNamespace(symbol="SPY", strategy="iron_condor",
                            expiry="2026-08-21", quantity=1,
                            capital_at_risk=3076.0, legs=self._condor_legs()),
            SimpleNamespace(symbol="QQQ", strategy="iron_condor",
                            expiry="2026-08-21", quantity=1,
                            capital_at_risk=2383.5, legs=self._condor_legs()),
        ]
        positions = [
            {"symbol": "SPY", "strategy": "iron_condor", "expiry": "2026-08-21",
             "max_loss": 0.0, "quantity": 1, "market_value": 431.0},
            {"symbol": "QQQ", "strategy": "iron_condor", "expiry": "2026-08-21",
             "max_loss": 0.0, "quantity": 1, "market_value": 644.0},
        ]
        rm = self._rm(positions, trades)
        rm._pos_config = SimpleNamespace(max_open_positions=5,
                                         max_portfolio_risk_pct=0.03,
                                         max_portfolio_delta=0.30)
        rm._risk_config = _risk_cfg(max_position_risk_pct=0.03,
                                    credit_vix_halt=28.0, max_credit_positions=6,
                                    # R20: config-backed since the register migration
                                    credit_cap_vix_tiers=[[20.0, 6], [25.0, 4], [999.0, 2]],
                                    max_symbol_concentration_pct=0.20)
        cb = MagicMock()
        cb.is_tripped = False
        cb.check_daily_loss.return_value = True
        rm._circuit_breaker = cb
        rm._correlation = MagicMock()
        rm._correlation.check_correlation.return_value = (True, "")
        acct = MagicMock()
        acct.get_net_liquidation = AsyncMock(return_value=198_000.0)
        acct.can_afford = AsyncMock(return_value=True)
        rm._account = acct
        rm._sizer = MagicMock()
        rm._portfolio_greeks = MagicMock(delta=0.0)

        from ait.risk.manager import TradeRequest
        req = TradeRequest(symbol="IWM", strategy="iron_condor",
                           direction="short", contracts=1, entry_price=1.5,
                           confidence=0.80, max_loss=3000.0, vix=15.0)
        res = await rm.validate_trade(req)
        # cap = 3% of 198k = $5,940; open (backfilled) 5,459.5 + 3,000 > cap
        assert res.approved is False
        assert "aggregate risk" in res.reason


# --- [4] promotions must create the open_positions row ----------------------
def _trade(trade_id="T-R16", status=TradeStatus.PENDING, symbol="SPY"):
    return SimpleNamespace(
        trade_id=trade_id, symbol=symbol, strategy="iron_condor",
        contract_type="iron_condor", quantity=1, entry_price=4.24,
        expiry="2026-08-21", strike=None, status=status,
        entry_time="2026-08-04T10:51:37",
        legs=json.dumps([
            {"strike": 700.0, "right": "P", "action": "SELL",
             "expiry": "2026-08-21"},
        ]),
    )


class TestPromotionInsertsOpenPositionRow:
    def _rec(self, state):
        rec = PositionReconciler.__new__(PositionReconciler)
        rec._state = state
        rec._ibkr = MagicMock()
        return rec

    def test_sweep_promotion_inserts_row(self):
        state = MagicMock()
        t = _trade("T-SWEEP")
        # 40 minutes old -> past STALE_PENDING_MINUTES
        t.entry_time = (__import__("datetime").datetime.now()
                        - timedelta(minutes=40)).isoformat()
        state.get_open_trades.return_value = [t]
        state.transition.return_value = True
        rec = self._rec(state)
        rec._live_ibkr_leg_keys = lambda: {"SPY:700.0:P:2026-08-21"}
        assert rec._sweep_stale_pending() == 0  # promoted, not closed
        state.insert_open_position.assert_called_once()
        assert state.insert_open_position.call_args.kwargs["trade_id"] == "T-SWEEP"

    def test_sweep_promotion_skips_insert_when_cas_refused(self):
        state = MagicMock()
        t = _trade("T-CAS")
        t.entry_time = (__import__("datetime").datetime.now()
                        - timedelta(minutes=40)).isoformat()
        state.get_open_trades.return_value = [t]
        state.transition.return_value = False  # someone else moved the row
        rec = self._rec(state)
        rec._live_ibkr_leg_keys = lambda: {"SPY:700.0:P:2026-08-21"}
        rec._sweep_stale_pending()
        state.insert_open_position.assert_not_called()

    async def test_reconcile_pending_promotion_inserts_row(self):
        state = MagicMock()
        t = _trade("T-PROM", TradeStatus.PENDING)
        state.get_open_trades.return_value = [t]
        state.update_trade_status.return_value = True
        rec = self._rec(state)
        live = SimpleNamespace(contract=SimpleNamespace(
            secType="OPT", symbol="SPY", strike=700.0, right="P",
            lastTradeDateOrContractMonth="20260821"), position=-1, avgCost=1.0)
        rec._ibkr.get_positions.return_value = [live]
        rec._ibkr.get_portfolio.return_value = []
        rec._sweep_stale_pending = lambda: 0
        result = await rec.reconcile()
        assert result.promoted == 1
        state.insert_open_position.assert_called_once()
        assert state.insert_open_position.call_args.kwargs["trade_id"] == "T-PROM"

    async def test_reconcile_closing_promotion_inserts_row(self):
        state = MagicMock()
        t = _trade("T-CLOSE", TradeStatus.CLOSING)
        state.get_open_trades.return_value = [t]
        state.transition.return_value = True
        rec = self._rec(state)
        live = SimpleNamespace(contract=SimpleNamespace(
            secType="OPT", symbol="SPY", strike=700.0, right="P",
            lastTradeDateOrContractMonth="20260821"), position=-1, avgCost=1.0)
        rec._ibkr.get_positions.return_value = [live]
        rec._ibkr.get_portfolio.return_value = []
        rec._sweep_stale_pending = lambda: 0
        rec._fetch_working_orders = AsyncMock(return_value=[])  # view OK, none
        result = await rec.reconcile()
        assert result.promoted == 1
        state.insert_open_position.assert_called_once()

    def test_insert_failure_never_escapes(self):
        state = MagicMock()
        state.insert_open_position.side_effect = RuntimeError("db locked")
        rec = self._rec(state)
        rec._ensure_open_position_row(_trade())  # must not raise


# --- [5] one kept-CLOSING discrepancy per TRADE, not per leg ----------------
class TestKeptUnknownDedup:
    async def test_condor_reports_one_break_not_four(self):
        state = MagicMock()
        legs = [{"strike": s, "right": r, "action": a, "expiry": "2026-08-21"}
                for s, r, a in ((655.0, "P", "BUY"), (685.0, "P", "SELL"),
                                (741.0, "C", "SELL"), (770.0, "C", "BUY"))]
        t = _trade("T4C", TradeStatus.CLOSING, symbol="QQQ")
        t.legs = json.dumps(legs)
        state.get_open_trades.return_value = [t]
        rec = PositionReconciler.__new__(PositionReconciler)
        rec._state = state
        rec._ibkr = MagicMock()
        rec._ibkr.get_positions.return_value = [
            SimpleNamespace(contract=SimpleNamespace(
                secType="OPT", symbol="QQQ", strike=lg["strike"],
                right=lg["right"], lastTradeDateOrContractMonth="20260821"),
                position=(-1 if lg["action"] == "SELL" else 1), avgCost=1.0)
            for lg in legs
        ]
        rec._ibkr.get_portfolio.return_value = []
        rec._sweep_stale_pending = lambda: 0
        rec._fetch_working_orders = AsyncMock(return_value=None)  # view down

        result = await rec.reconcile()
        kept = [d for d in result.discrepancies if "broker open-order view" in d]
        assert len(kept) == 1  # pre-fix: 4
        assert result.promoted == 0  # still kept CLOSING — no duplicate close


# --- [6] the report path must not touch protection state --------------------
class TestReportPathIsReadOnly:
    def _pm(self):
        pm = PortfolioManager.__new__(PortfolioManager)
        pm._last_quote_ts = {}
        pm._frozen_alerted = set()
        pm._touch_confirm = {}
        pm._quality = MagicMock()
        pm._quality.validate_quote.return_value = SimpleNamespace(
            is_valid=True, issues=[], staleness_seconds=1.0)
        return pm

    def _quote(self, ts, mid=100.0):
        return SimpleNamespace(mid=mid, bid=mid - 0.01, ask=mid + 0.01,
                               last=mid, volume=1000, timestamp=ts)

    async def test_summary_read_does_not_consume_the_tick_advance(self):
        from datetime import datetime as _dt
        pm = self._pm()
        t0, t1 = _dt(2026, 8, 10, 14, 0, 0), _dt(2026, 8, 10, 14, 0, 20)
        md = MagicMock()
        pm._market_data = md

        md.get_quote = AsyncMock(return_value=self._quote(t0))
        assert (await pm._spot_quote("SPY"))[1] == "fresh"     # monitor pass
        md.get_quote = AsyncMock(return_value=self._quote(t1))
        assert (await pm._spot_quote("SPY", persist=False))[1] == "fresh"  # report
        # The monitor's next pass reads the SAME 15s-cached quote. Pre-fix the
        # report had already stored t1, so this came back "frozen" and the
        # touch stop dropped to 2-tick confirmation.
        md.get_quote = AsyncMock(return_value=self._quote(t1))
        assert (await pm._spot_quote("SPY"))[1] == "fresh"

    async def test_report_does_not_rearm_the_frozen_page_latch(self):
        from datetime import datetime as _dt
        pm = self._pm()
        pm._frozen_alerted.add("SPY")
        pm._market_data = MagicMock()
        pm._market_data.get_quote = AsyncMock(
            return_value=self._quote(_dt(2026, 8, 10, 14, 0, 0)))
        await pm._spot_quote("SPY", persist=False)
        assert "SPY" in pm._frozen_alerted   # pre-fix: discarded by a report
        await pm._spot_quote("SPY")
        assert "SPY" not in pm._frozen_alerted

    def _evaluable(self, spot):
        """A PortfolioManager wired far enough to run _evaluate_position."""
        from datetime import datetime as _dt
        from ait.config.settings import ExitConfig
        pm = self._pm()
        pm._exit_config = ExitConfig()
        pm._earnings = None
        pm._economic_cal = None
        pm._notify_cb = None
        pm._marks_missing_streak = {}
        pm._pdt_alerted = set()
        pm._ibkr = MagicMock()
        pm._ibkr.connected = False
        pm._market_data = MagicMock()
        pm._market_data.get_quote = AsyncMock(
            return_value=self._quote(_dt(2026, 8, 10, 14, 0, 0), mid=spot))
        pm._option_position_unrealized = lambda *a, **k: 12.0
        st = MagicMock()
        st.get_high_water_mark.return_value = 0.05
        pm._state = st
        pdt = MagicMock()
        pdt.would_be_day_trade.return_value = False
        pm._pdt_guard = pdt
        return pm

    def _condor(self):
        return SimpleNamespace(
            trade_id="T-LIVE", symbol="SPY", strategy="iron_condor",
            # status is what get_portfolio_summary filters on
            status=TradeStatus.FILLED,
            contract_type="iron_condor", quantity=1, entry_price=4.24,
            direction=SimpleNamespace(value="neutral"),
            expiry=None, strike=None, entry_time="2026-08-04T10:51:37",
            legs=json.dumps([{"strike": 690.0, "right": "P", "action": "SELL"},
                             {"strike": 660.0, "right": "P", "action": "BUY"},
                             {"strike": 720.0, "right": "C", "action": "SELL"},
                             {"strike": 750.0, "right": "C", "action": "BUY"}]))

    async def test_report_cannot_clear_a_pending_touch_streak(self):
        # Spot back INSIDE the strikes: the rule-1b branch that popped the
        # streak regardless of `persist`. Driven through the real
        # _evaluate_position, exactly as get_portfolio_summary calls it.
        trade = self._condor()
        pm = self._evaluable(spot=700.0)  # between 690 put and 720 call
        pm._touch_confirm["T-LIVE"] = 1
        await pm._evaluate_position(trade, persist=False)
        assert pm._touch_confirm.get("T-LIVE") == 1  # pre-fix: silently wiped
        # the 30s monitor (persist=True) legitimately kills a dead streak
        await pm._evaluate_position(trade, persist=True)
        assert "T-LIVE" not in pm._touch_confirm

    async def test_report_cannot_advance_a_touch_streak_either(self):
        # Touched + non-fresh quote: the increment was already persist-guarded;
        # pin it so the guard cannot regress while its sibling is fixed.
        trade = self._condor()
        pm = self._evaluable(spot=685.0)  # below the 690 short put
        pm._quality.validate_quote.return_value = SimpleNamespace(
            is_valid=False, issues=["stale"], staleness_seconds=90.0)
        st = await pm._evaluate_position(trade, persist=False)
        assert pm._touch_confirm.get("T-LIVE") is None
        assert st.should_exit is False  # 1 tick on a degraded quote != exit

    async def test_summary_call_does_not_consume_a_touch_streak(self):
        """The EOD/report caller, end to end.

        R17-bis: this was `assert "persist=False" in
        inspect.getsource(get_portfolio_summary)` — satisfied by the string
        appearing anywhere (a dead branch, a comment, a different call), and
        by nothing at all if the loop is reshaped. get_portfolio_summary was
        otherwise never CALLED by any test in the suite, so the report path
        that mutates live protection state had zero executable coverage.
        Drive it for real: same rule-1b scenario as the test above, reached
        through the public method instead of _evaluate_position directly.
        """
        trade = self._condor()
        pm = self._evaluable(spot=700.0)    # spot back INSIDE the short strikes
        pm._state.get_open_trades.return_value = [trade]
        pm._state.get_daily_stats.return_value = SimpleNamespace(
            total_pnl=12.5, trades_taken=1)
        pm._touch_confirm["T-LIVE"] = 1

        summary = await pm.get_portfolio_summary()

        # the position was actually priced (not skipped by the status filter),
        # so the touch-streak assertion below is about a real evaluation
        assert summary["open_positions"] == 1
        assert summary["today_realized_pnl"] == 12.5
        assert pm._touch_confirm.get("T-LIVE") == 1  # persist=True: wiped


# --- [7] the inert delta rule must announce itself --------------------------
class TestDeltaBreachInertIsLoud:
    def _pm(self, portfolio_items):
        pm = PortfolioManager.__new__(PortfolioManager)
        ib = MagicMock()
        ib.ib.portfolio.return_value = portfolio_items
        ib.ib.ticker.return_value = None  # no subscription -> no ticker
        ib.connected = True
        pm._ibkr = ib
        return pm

    def _item(self, symbol="SPY"):
        return SimpleNamespace(
            contract=SimpleNamespace(symbol=symbol), position=-1)

    def test_no_greeks_returns_none_and_warns_once(self, monkeypatch):
        calls = []
        monkeypatch.setattr("ait.execution.portfolio.log.warning",
                            lambda ev, **kw: calls.append(ev))
        pm = self._pm([self._item()])
        trade = SimpleNamespace(trade_id="T-D", symbol="SPY",
                                strategy="iron_condor")
        assert pm._get_position_delta(trade) is None
        assert calls.count("delta_breach_rule_inert") == 1
        # once per SESSION, not per 30s tick
        for _ in range(5):
            pm._get_position_delta(trade)
        assert calls.count("delta_breach_rule_inert") == 1

    def test_real_greeks_still_work_and_stay_silent(self, monkeypatch):
        calls = []
        monkeypatch.setattr("ait.execution.portfolio.log.warning",
                            lambda ev, **kw: calls.append(ev))
        pm = self._pm([self._item()])
        pm._ibkr.ib.ticker.return_value = SimpleNamespace(
            modelGreeks=SimpleNamespace(delta=0.40))
        trade = SimpleNamespace(trade_id="T-D2", symbol="SPY",
                                strategy="iron_condor")
        assert pm._get_position_delta(trade) == pytest.approx(-0.40)
        assert "delta_breach_rule_inert" not in calls

    def test_inertness_is_documented_at_both_sites(self):
        import inspect
        doc = PortfolioManager._get_position_delta.__doc__ or ""
        assert "INERT" in doc.upper()
        src = inspect.getsource(PortfolioManager._evaluate_position)
        assert "INERT" in src.upper()

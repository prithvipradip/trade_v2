"""R15 — fixes for the 8 defects confirmed by the main-vs-branch adversarial
review (plus 2 sample-velocity minors). Each test fails against the pre-fix
code by construction (asserts the corrected behavior the review proved wrong).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import TradeStatus
from ait.broker.ibkr_client import IBKRClient
from ait.execution.executor import TradeExecutor
from ait.execution.reconciler import PositionReconciler
from ait.risk.circuit_breaker import CircuitBreaker
from ait.risk.position_sizer import PositionSizer


# [1] intrinsic expiry booking: single-leg DEBIT must synthesize a BUY leg
class TestIntrinsicFallbackSign:
    def _trade(self, strategy, strike=300.0):
        return SimpleNamespace(legs="[]", strike=strike, strategy=strategy)

    def test_long_put_itm_values_positive(self):
        # bought a 300P, settle 295 -> intrinsic +5 to US (pre-fix: -5)
        v = PositionReconciler._structure_intrinsic(self._trade("long_put"), 295.0)
        assert v == pytest.approx(5.0)

    def test_long_call_itm_values_positive(self):
        v = PositionReconciler._structure_intrinsic(self._trade("long_call"), 305.0)
        assert v == pytest.approx(5.0)

    def test_cash_secured_put_still_sell(self):
        # credit single-leg: cost-to-settle convention must be unchanged
        v = PositionReconciler._structure_intrinsic(
            self._trade("cash_secured_put"), 295.0)
        assert v == pytest.approx(5.0)  # -value of (-5) per credit convention


# [2] startup reconcile must never book a PENDING row as a stale close
class TestPendingNotBookedStale:
    async def test_pending_with_working_entry_skipped(self):
        rec = PositionReconciler.__new__(PositionReconciler)
        ibkr = MagicMock()
        ibkr.connected = True
        # one unrelated live option so the zero-options guard does not trip
        live = SimpleNamespace(contract=SimpleNamespace(
            secType="OPT", symbol="SPY", strike=700.0, right="P",
            lastTradeDateOrContractMonth="20260821"), position=1, avgCost=1.0)
        ibkr.get_positions.return_value = [live]
        ibkr.get_portfolio.return_value = []
        ibkr.get_positions_fresh = AsyncMock(return_value=[live])
        rec._ibkr = ibkr
        state = MagicMock()
        pend = SimpleNamespace(
            trade_id="T-PEND", symbol="IWM", strategy="iron_condor",
            contract_type="iron_condor", quantity=1, entry_price=1.0,
            expiry="2026-08-21", strike=None, status=TradeStatus.PENDING,
            entry_time="2026-07-21T10:00:00",
            legs='[{"strike":290.0,"right":"P","action":"SELL","expiry":"2026-08-21"}]',
        )
        state.get_open_trades.return_value = [pend]
        rec._state = state
        rec._sweep_stale_pending = lambda: 0  # the SWEEP owns pendings, not this loop
        await rec.reconcile()
        state.close_trade.assert_not_called()


# [3] MTM daily-loss halt must clear on the daily reset
class TestMtmHaltClears:
    def _breaker(self):
        cfg = SimpleNamespace(max_daily_loss_pct=0.02, max_consecutive_losses=99,
                              max_api_failures=99, pause_duration_minutes=1)
        try:
            return CircuitBreaker(cfg)
        except TypeError:
            b = CircuitBreaker.__new__(CircuitBreaker)
            b._config = cfg
            b._tripped, b._trip_reason = False, ""
            b._daily_pnl, b._consecutive_losses = 0.0, 0
            b._api_failures, b._resume_time = {}, None
            b._last_reset_date = date.today()
            return b

    def test_mtm_trip_untrips_next_day(self):
        b = self._breaker()
        assert b.check_daily_loss_mtm(-2000.0, 60000.0) is True
        assert b._tripped
        b._last_reset_date = date.today() - timedelta(days=1)
        b.check_daily_reset()
        assert not b._tripped  # pre-fix: reason lacked 'daily_loss' -> stuck forever


# [4] paper/live assertion resolves the SESSION account, config only fallback
class TestSessionAccountResolution:
    def test_session_wins_over_config(self):
        assert IBKRClient._resolve_session_account(
            ["U21959335"], "DUN603821") == "U21959335"

    def test_config_only_when_session_empty(self):
        assert IBKRClient._resolve_session_account([], "DUN603821") == "DUN603821"

    def test_unknown_when_nothing(self):
        assert IBKRClient._resolve_session_account([], None) == "unknown"


# [5] foreign-order stash covers the reconnect-back-to-base sequence
class TestForeignStash:
    def _client(self, ever, last=None):
        c = IBKRClient.__new__(IBKRClient)
        c._ever_connected = ever
        if last is not None:
            c._last_client_id = last
        return c

    def test_first_connect_never_stashes(self):
        assert self._client(False)._should_stash_foreign(1, 1) is False

    def test_base_to_fallback_stashes(self):
        assert self._client(True, last=1)._should_stash_foreign(105, 1) is True

    def test_fallback_back_to_base_stashes(self):
        # THE missed leg: orders live under 105, reconnect lands on base 1
        assert self._client(True, last=105)._should_stash_foreign(1, 1) is True

    def test_same_id_reconnect_no_stash(self):
        assert self._client(True, last=1)._should_stash_foreign(1, 1) is False


# [6] IBKR quote timestamps must be naive-LOCAL (match the Yahoo fallback)
class TestQuoteTimestampNormalization:
    async def test_ibkr_stamp_is_local_wall_clock(self):
        from ait.data.market_data import MarketDataService
        svc = MarketDataService.__new__(MarketDataService)
        ibkr = MagicMock()
        ibkr.connected = True
        ibkr.qualify_contract = AsyncMock(return_value=SimpleNamespace(conId=1))
        ibkr.ib.reqMarketDataType = MagicMock()
        ibkr.ib.reqMktData = MagicMock()
        ibkr.ib.cancelMktData = MagicMock()
        ibkr.ib.ticker = MagicMock(return_value=SimpleNamespace(
            bid=1.0, ask=1.1, last=1.05, volume=100,
            time=datetime.now(timezone.utc)))  # aware-UTC, as ib_insync gives
        svc._ibkr = ibkr
        q = await svc._get_ibkr_quote("SPY")
        assert q is not None
        # pre-fix this was the UTC wall-clock: ~4h in the future vs local
        assert abs((q.timestamp - datetime.now()).total_seconds()) < 120


# [7] the model-artifact fence covers meta_label
class TestMetaLabelFence:
    def test_meta_label_model_dir_redirected(self):
        import ait.ml.meta_label as ml
        assert ml.MODEL_DIR.name != "models"  # autouse fixture must repoint it


# [8] realized_pnl commission true-up before any booking
class TestCommissionTrueUp:
    async def test_estimate_swapped_for_ledger_truth(self):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        state = MagicMock()
        stats = SimpleNamespace(total_pnl=0.0, trades_won=0, trades_lost=0,
                                trades_taken=0)
        state.get_daily_stats.return_value = stats
        state.total_commission.return_value = 8.89       # ledger truth
        state.count_executions.return_value = 8          # R16: complete ledger (2x4 legs)
        orch._state = state
        trade = SimpleNamespace(trade_id="T-X", symbol="NVDA",
                                strategy="iron_condor", contract_type="iron_condor",
                                quantity=1, entry_time=None,
                                legs='[{},{},{},{}]')
        orch._find_trade_by_id = lambda tid: trade
        orch._circuit_breaker = MagicMock()
        orch._thompson = MagicMock()
        orch._pdt_guard = MagicMock()
        orch._trainer = MagicMock()
        orch._exit_attempts = {}
        async def _notify(m):
            pass
        orch._send_notification = _notify
        ex = {"trade_id": "T-X", "realized_pnl": -10.00, "exit_reason": "stop",
              "exit_price": 0.74}
        await orch._process_completed_exits([ex])
        # est (IC, qty1) = 0.65*4*2 = 5.20; delta = 5.20 - 8.89 = -3.69
        state.update_trade_realized_pnl.assert_called_once_with("T-X", -13.69)
        assert stats.total_pnl == pytest.approx(-13.69)

    async def test_partial_ledger_defers_trueup(self):
        # R16: commissionReports arrive async — a partial ledger (entry legs
        # only) must DEFER, not true up against half the commissions.
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        state = MagicMock()
        stats = SimpleNamespace(total_pnl=0.0, trades_won=0, trades_lost=0,
                                trades_taken=0)
        state.get_daily_stats.return_value = stats
        state.total_commission.return_value = 2.60       # entry side only
        state.count_executions.return_value = 4          # < 2x4 legs
        orch._state = state
        trade = SimpleNamespace(trade_id="T-Y", symbol="SPY",
                                strategy="iron_condor", contract_type="iron_condor",
                                quantity=1, entry_time=None,
                                legs='[{},{},{},{}]')
        orch._find_trade_by_id = lambda tid: trade
        orch._circuit_breaker = MagicMock()
        orch._thompson = MagicMock()
        orch._pdt_guard = MagicMock()
        orch._trainer = MagicMock()
        orch._exit_attempts = {}
        async def _notify(m):
            pass
        orch._send_notification = _notify
        ex = {"trade_id": "T-Y", "realized_pnl": -10.00, "exit_reason": "stop",
              "exit_price": 0.74}
        await orch._process_completed_exits([ex])
        state.update_trade_realized_pnl.assert_not_called()
        state.set_state.assert_any_call("trueup_pending_T-Y",
                                        date.today().isoformat())

    def test_estimate_formula_unchanged(self):
        t = SimpleNamespace(contract_type="iron_condor", strategy="iron_condor",
                            quantity=1)
        assert TradeExecutor.commission_estimate(t) == pytest.approx(5.20)
        t2 = SimpleNamespace(contract_type="spread", strategy="long_straddle",
                             quantity=2)
        assert TradeExecutor.commission_estimate(t2) == pytest.approx(5.20)


# [minors] first-hour gate + VIX-None sizer
class TestMinors:
    def test_sizer_survives_vix_none(self):
        sizer = PositionSizer.__new__(PositionSizer)
        sizer._pos_config = SimpleNamespace(max_position_pct=0.10,
                                            max_contracts_per_trade=1,
                                            max_portfolio_risk_pct=0.20)
        sizer._risk_config = SimpleNamespace(min_confidence=0.5)
        size = sizer.calculate(
            account_value=100_000, option_price=2.0, confidence=0.7,
            implied_vol=0.3, strategy="long_straddle", underlying_price=300,
            vix=None)  # pre-fix: TypeError on `vix >= 30`
        assert size.contracts >= 1


# PLAN 2026-08-03 (user-approved): pre-event credit blackout 4 -> 1 day.
# The <=4 window blacked out ~half of trading days (NFP+CPI+PCE lead-outs)
# and refused the richest pre-event premium; holds span events regardless.
class TestPreEventBlackoutRelaxed:
    def test_default_is_one_day(self):
        from ait.config.settings import RiskConfig
        assert RiskConfig().pre_event_blackout_days == 1

    def test_engine_parity_uses_config_default(self):
        # the backtest gate must read the SAME default (no drifting constant)
        import inspect
        from ait.backtesting import engine
        src = inspect.getsource(engine)
        assert "pre_event_blackout_days" in src
        assert "_d2e <= 4" not in src

    def test_two_days_out_no_longer_blocked(self):
        from ait.config.settings import RiskConfig
        blackout = RiskConfig().pre_event_blackout_days
        for d2e, expect_block in [(0, True), (1, True), (2, False), (4, False)]:
            assert (d2e <= blackout) is expect_block

    def test_defined_risk_exempt_from_macro_flatten(self):
        # the paired exit rule: condors HOLD through events (wings cap the
        # surprise) — else entries at d2e=2 get force-closed at d2e=1 for one
        # day of theta and full round-trip costs. Undefined-risk keeps it.
        import inspect
        from ait.execution import portfolio
        from ait.backtesting import engine
        for mod in (portfolio, engine):
            src = inspect.getsource(mod)
            i = src.find("AIT_SKIP_MACRO_EVENTS")
            while i != -1:
                clause = src[i:i + 400]
                if "short_strangle" in clause:
                    assert '"iron_condor"' not in clause
                i = src.find("AIT_SKIP_MACRO_EVENTS", i + 1)


# ABLATION VERDICT 2026-08-03: ML entry gates removed (default OFF)
class TestMlGatesRemoved:
    def test_entry_gates_disabled_by_default(self):
        from ait.config.settings import MLConfig
        assert MLConfig().entry_gates_enabled is False  # 3 identical runs: gates veto nothing

    def test_flag_can_reenable_for_studies(self):
        from ait.config.settings import MLConfig
        assert MLConfig(entry_gates_enabled=True).entry_gates_enabled is True

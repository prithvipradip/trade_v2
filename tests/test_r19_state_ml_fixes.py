"""R19 state/ML-cluster fixes — every test below FAILS against the pre-fix code.

Six audit findings owned by this cluster, driven through the REAL functions:

  [1] ensemble.DirectionPredictor.cv_scores was a getter-only property, so the
      R17 auto-rollback fix (`predictor.cv_scores = {"all_symbol_mean": ...}`)
      raised AttributeError on every retrain into its own `except: pass`
      — and rollback() only reloaded in memory, leaving the DEGRADED
      ensemble.pkl on disk for the live bot to load. Both halves.
  [2] account._handle_stale recorded ONE api failure per outage while the
      breaker trips at five — the stale-account halt could never engage.
  [3] reconciler._book_stale_trade booked NaN realized P&L (NaN is truthy).
  [4] reconciler._cancel_working_entry_order could cancel a DIFFERENT trade's
      working order (symbol+BAG / strike-substring matching, orderRef ignored).
  [5] portfolio._spot_quote called the second same-symbol read of a pass
      "frozen" (per-symbol timestamp dict vs per-pass multiplicity).
  [6] reconciler._sweep_stale_pending believed a possibly wedged-EMPTY cached
      position list that every other 'gone' path re-queries authoritatively.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.state import TradeStatus
from ait.broker.account import AccountManager
from ait.config.settings import ExitConfig, MLConfig, RiskConfig
from ait.data.market_data import Quote
from ait.execution.portfolio import PortfolioManager
from ait.execution.reconciler import PositionReconciler
from ait.ml.ensemble import DirectionPredictor
from ait.ml.trainer import ModelTrainer
from ait.risk.circuit_breaker import CircuitBreaker


# ===========================================================================
# [1] ML auto-rollback — the assignment must land AND the rollback must stick
# ===========================================================================

def _predictor(tmp_path, version: str, tag: str, acc: float, persist=True):
    """A real DirectionPredictor holding picklable placeholder 'models'."""
    p = DirectionPredictor(MLConfig(), model_dir=tmp_path, persist_artifacts=persist)
    p._models = {"xgboost": f"MODEL-{tag}"}
    p._scaler = f"SCALER-{tag}"
    p._feature_names = ["rsi", "atr"]
    p._model_version = version
    p._cv_scores = {"xgboost": acc}
    p._symbol_models = {}
    p._trained = True
    return p


class TestCvScoresSetter:
    def test_property_has_a_setter(self):
        # The R17 fix statement is `self._predictor.cv_scores = {...}`;
        # pre-fix `fset` was None, so it could only ever raise.
        assert DirectionPredictor.cv_scores.fset is not None

    def test_assignment_is_real_not_swallowed(self, tmp_path):
        p = _predictor(tmp_path, "v-1", "A", 0.61)
        # EXACTLY what trainer.train_all_symbols does at the rollback gate,
        # including the blind except that hid the AttributeError.
        try:
            p.cv_scores = {"all_symbol_mean": 0.55}
        except Exception:  # noqa: BLE001 — the pre-fix swallow, reproduced
            pass
        # Pre-fix: still {"xgboost": 0.61} (the LAST symbol's scores).
        assert p.cv_scores == {"all_symbol_mean": 0.55}
        assert p._cv_scores == {"all_symbol_mean": 0.55}

    def test_setter_rejects_non_dict(self, tmp_path):
        p = _predictor(tmp_path, "v-1", "A", 0.61)
        with pytest.raises(TypeError):
            p.cv_scores = 0.55

    def test_all_symbol_mean_actually_changes_the_rollback_verdict(self, tmp_path):
        """The consequence the R17 comment claimed and never delivered.

        (ModelTrainer._should_rollback is another owner's file — driven, not
        modified.)
        """
        prev = {"xgboost": 0.62, "lightgbm": 0.60}
        # Two symbols retrained: one fine, one collapsed -> mean 0.475.
        results = {"SPY": {"xgboost": 0.61, "lightgbm": 0.59},
                   "QQQ": {"xgboost": 0.35, "lightgbm": 0.35}}
        vals = [a for s in results.values() for a in s.values()]
        p = _predictor(tmp_path, "v-new", "BAD", 0.60)
        p._cv_scores = dict(results["QQQ"])  # what train() leaves: LAST symbol
        # Pre-fix the assignment threw, so the verdict was taken on the last
        # symbol alone; here it happens to be the collapsed one, but with the
        # symbols in the other order the degradation is invisible.
        assert ModelTrainer._should_rollback(prev, p.cv_scores) is True
        p.cv_scores = {"all_symbol_mean": sum(vals) / len(vals)}
        assert ModelTrainer._should_rollback(prev, p.cv_scores) is True
        # ...and a healthy retrain must NOT roll back off one weak symbol.
        ok = {"SPY": {"xgboost": 0.61}, "QQQ": {"xgboost": 0.59}}
        ok_vals = [a for s in ok.values() for a in s.values()]
        p.cv_scores = {"all_symbol_mean": sum(ok_vals) / len(ok_vals)}
        assert ModelTrainer._should_rollback(prev, p.cv_scores) is False


class TestRollbackPersists:
    def _good_then_degraded(self, tmp_path):
        good = _predictor(tmp_path, "v-good", "GOOD", 0.62)
        good._save_models()
        bad = _predictor(tmp_path, "v-bad", "DEGRADED", 0.40)
        bad._save_models()  # ensemble.pkl now holds the degraded model
        return bad

    def test_rollback_rewrites_ensemble_pkl(self, tmp_path):
        bad = self._good_then_degraded(tmp_path)
        assert bad.rollback("v-good") is True

        # The live bot's next start reads ensemble.pkl from a FRESH process.
        # Pre-fix it got MODEL-DEGRADED — the model the rollback rejected.
        nxt = DirectionPredictor(MLConfig(), model_dir=tmp_path)
        assert nxt.load_models() is True
        assert nxt.model_version == "v-good"
        assert nxt._models == {"xgboost": "MODEL-GOOD"}
        assert nxt.cv_scores == {"xgboost": 0.62}

        on_disk = pickle.loads((tmp_path / "ensemble.pkl").read_bytes())
        assert on_disk["version"] == "v-good"

    def test_rollback_target_missing_never_claims_success(self, tmp_path):
        bad = self._good_then_degraded(tmp_path)
        (tmp_path / "ensemble_v-good.pkl").unlink()
        # Pre-fix: loaded the latest (= the degraded model) and returned True,
        # reporting a rollback that had not happened.
        assert bad.rollback("v-good") is False
        # ...and the degraded artifact must not be re-persisted as if fixed.
        on_disk = pickle.loads((tmp_path / "ensemble.pkl").read_bytes())
        assert on_disk["version"] == "v-bad"

    def test_rollback_honours_the_r16_artifact_fence(self, tmp_path):
        good = _predictor(tmp_path, "v-good", "GOOD", 0.62)
        good._save_models()
        bad = _predictor(tmp_path, "v-bad", "DEGRADED", 0.40)
        bad._save_models()
        # A research/window predictor may never write the live artifact.
        window = _predictor(tmp_path, "v-bad", "DEGRADED", 0.40, persist=False)
        assert window.rollback("v-good") is True
        assert window._models == {"xgboost": "MODEL-GOOD"}  # memory only
        on_disk = pickle.loads((tmp_path / "ensemble.pkl").read_bytes())
        assert on_disk["version"] == "v-bad"

    def test_save_models_reports_success(self, tmp_path):
        p = _predictor(tmp_path, "v-1", "A", 0.61)
        assert p._save_models() is True


# ===========================================================================
# [2] stale account snapshot must actually HALT trading
# ===========================================================================

def _account(cb=None, notify=None):
    mgr = AccountManager(MagicMock())
    mgr._circuit_breaker = cb
    mgr._notify_cb = notify
    return mgr


class TestStaleAccountEscalation:
    def _breaker(self):
        return CircuitBreaker(RiskConfig(max_api_failures=5))

    async def test_stale_past_threshold_trips_the_breaker(self):
        cb = self._breaker()
        mgr = _account(cb)
        await mgr._handle_stale(3600)  # 1 hour of a dead account feed
        # Pre-fix: exactly ONE recorded failure = 1/5 of a trip; never halted.
        assert cb.is_tripped is True

    async def test_full_outage_through_get_snapshot_halts(self):
        """The real production path: get_account_values() returns {} while a
        prior snapshot exists (the documented FX-unavailable hard stop)."""
        cb = self._breaker()
        mgr = _account(cb)
        mgr._client.get_account_values = AsyncMock(return_value={})
        mgr._snapshot.net_liquidation = 198_000.0
        mgr._last_fetch = time.time() - 3600
        await mgr.get_snapshot(force_refresh=True)
        assert cb.is_tripped is True

    async def test_below_threshold_does_not_halt(self):
        cb = self._breaker()
        mgr = _account(cb)
        await mgr._handle_stale(600)  # 10 min: loud, but not a halt condition
        assert cb.is_tripped is False

    async def test_reescalates_after_the_10_minute_auto_resume(self):
        """The api-failure trip auto-resumes after 600s. An outage that
        outlives the pause must re-halt — pre-fix the one-shot latch made the
        second escalation a no-op (on top of never tripping at all)."""
        cb = self._breaker()
        mgr = _account(cb)
        await mgr._handle_stale(3600)
        assert cb.is_tripped is True
        cb._resume_time = time.time() - 1  # pause elapsed
        assert cb.is_tripped is False
        await mgr._handle_stale(4200)  # still stale on the next read
        assert cb.is_tripped is True

    async def test_page_stays_once_per_outage_and_names_the_halt(self):
        cb = self._breaker()
        sent = []

        async def _notify(msg):
            sent.append(msg)

        mgr = _account(cb, _notify)
        await mgr._handle_stale(3600)
        await mgr._handle_stale(3700)
        assert len(sent) == 1
        assert "HALTED" in sent[0]

    async def test_no_breaker_wired_is_loud_not_silent(self, monkeypatch):
        """If the halt lever is structurally unreachable, say so — once."""
        crits = []
        import ait.broker.account as acct_mod
        monkeypatch.setattr(acct_mod.log, "critical",
                            lambda ev, **kw: crits.append(ev))
        mgr = _account(None)
        assert mgr._escalate_to_breaker(3600) is False
        assert mgr._escalate_to_breaker(3700) is False
        assert crits.count("account_stale_halt_unavailable") == 1

    async def test_breaker_that_refuses_to_trip_is_paged(self, monkeypatch):
        crits = []
        import ait.broker.account as acct_mod
        monkeypatch.setattr(acct_mod.log, "critical",
                            lambda ev, **kw: crits.append(ev))
        dud = SimpleNamespace(is_tripped=False, record_api_failure=lambda: None)
        mgr = _account(dud)
        assert mgr._escalate_to_breaker(3600) is False
        assert "account_stale_halt_failed" in crits

    async def test_successful_fetch_rearms_the_latch(self):
        cb = self._breaker()
        mgr = _account(cb)
        mgr._client.get_account_values = AsyncMock(return_value={
            "NetLiquidation": "198000", "BuyingPower": "50000",
        })
        mgr._last_fetch = time.time() - 3600
        await mgr.get_snapshot(force_refresh=True)
        assert mgr._stale_escalated is False


# ===========================================================================
# reconciler fixtures
# ===========================================================================

CONDOR_LEGS = [
    {"strike": 745.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 735.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
]


def _rec_trade(trade_id="T-MINE", status=TradeStatus.PENDING, symbol="SPY",
               legs=None, age_min=40):
    return SimpleNamespace(
        trade_id=trade_id, symbol=symbol, strategy="iron_condor",
        contract_type="iron_condor", quantity=1, entry_price=4.24,
        expiry="2026-12-18", strike=None, status=status,
        entry_time=(datetime.now() - timedelta(minutes=age_min)).isoformat(),
        legs=json.dumps(legs if legs is not None else CONDOR_LEGS),
    )


def _reconciler(state=None):
    rec = PositionReconciler.__new__(PositionReconciler)
    rec._state = state if state is not None else MagicMock()
    rec._ibkr = MagicMock()
    return rec


def _pf_item(strike, pnl, symbol="SPY", right="P", expiry="20261218"):
    return SimpleNamespace(
        contract=SimpleNamespace(secType="OPT", symbol=symbol, strike=strike,
                                 right=right,
                                 lastTradeDateOrContractMonth=expiry),
        position=0, realizedPNL=pnl,
    )


# ===========================================================================
# [3] NaN realized P&L must never reach the trades DB
# ===========================================================================

class TestNanRealizedPnlNeverBooked:
    async def test_nan_leg_is_not_booked_as_broker_realized(self):
        state = MagicMock()
        rec = _reconciler(state)
        t = _rec_trade("T-NAN", TradeStatus.FILLED)
        await rec._book_stale_trade(t, [_pf_item(745.0, float("nan"))])
        kw = state.close_trade.call_args.kwargs
        # Pre-fix: realized_pnl=nan, exit_reason_detailed='reconciler_ibkr_realized'
        assert math.isfinite(kw["realized_pnl"])
        assert kw["realized_pnl"] == 0.0
        assert kw["exit_reason_detailed"] != "reconciler_ibkr_realized"

    async def test_one_nan_leg_poisons_nothing_when_others_are_good(self):
        state = MagicMock()
        rec = _reconciler(state)
        t = _rec_trade("T-MIX", TradeStatus.FILLED)
        await rec._book_stale_trade(t, [
            _pf_item(745.0, float("nan")),
            _pf_item(735.0, -120.0),
        ])
        kw = state.close_trade.call_args.kwargs
        # Pre-fix: nan + -120 = nan, booked as broker-derived.
        assert math.isfinite(kw["realized_pnl"])
        # FAIL-SAFE: one unknown leg makes the STRUCTURE's P&L unknown —
        # a partial sum must not be labelled broker-derived.
        assert kw["exit_reason_detailed"] != "reconciler_ibkr_realized"

    async def test_infinite_leg_is_refused_too(self):
        state = MagicMock()
        rec = _reconciler(state)
        t = _rec_trade("T-INF", TradeStatus.FILLED)
        await rec._book_stale_trade(t, [_pf_item(745.0, float("inf"))])
        kw = state.close_trade.call_args.kwargs
        assert math.isfinite(kw["realized_pnl"])

    async def test_finite_broker_pnl_still_books_normally(self):
        """Regression guard: the guard must not disarm the good path."""
        state = MagicMock()
        rec = _reconciler(state)
        t = _rec_trade("T-OK", TradeStatus.FILLED)
        await rec._book_stale_trade(t, [
            _pf_item(745.0, -80.0), _pf_item(735.0, 30.0),
        ])
        kw = state.close_trade.call_args.kwargs
        assert kw["realized_pnl"] == -50.0
        assert kw["exit_reason_detailed"] == "reconciler_ibkr_realized"


# ===========================================================================
# [4] the sweep's cancel must not kill another trade's order
# ===========================================================================

def _broker_order(symbol="SPY", sec="BAG", ref="", status="Submitted",
                  strike=0.0, right="P", expiry="20261218", order_id=11):
    return SimpleNamespace(
        contract=SimpleNamespace(symbol=symbol, secType=sec, strike=strike,
                                 right=right,
                                 lastTradeDateOrContractMonth=expiry),
        order=SimpleNamespace(orderId=order_id, orderRef=ref),
        orderStatus=SimpleNamespace(status=status),
    )


class TestCancelUsesOrderRef:
    def _rec_with_orders(self, orders):
        rec = _reconciler()
        ib = MagicMock()
        ib.openTrades.return_value = orders
        ib.cancelOrder = MagicMock()
        rec._ibkr.ib = ib
        return rec, ib

    def test_never_cancels_an_order_tagged_for_another_trade(self):
        """A live EXIT combo for a DIFFERENT trade, same symbol."""
        other = _broker_order(ref="T-OTHER-EXIT-42", sec="BAG")
        rec, ib = self._rec_with_orders([other])
        # Pre-fix: any active same-symbol BAG matched -> True + cancelled.
        assert rec._cancel_working_entry_order(_rec_trade("T-MINE")) is False
        ib.cancelOrder.assert_not_called()

    def test_cancels_this_trades_own_tagged_order(self):
        mine = _broker_order(ref="T-MINE", sec="BAG")
        rec, ib = self._rec_with_orders([_broker_order(ref="T-OTHER"), mine])
        assert rec._cancel_working_entry_order(_rec_trade("T-MINE")) is True
        ib.cancelOrder.assert_called_once_with(mine.order)

    def test_untagged_legacy_bag_still_matches_by_symbol(self):
        """Pre-R17 orders carry no orderRef — the old heuristic must survive
        for exactly those, or the R16 fix regresses."""
        legacy = _broker_order(ref="", sec="BAG")
        rec, ib = self._rec_with_orders([legacy])
        assert rec._cancel_working_entry_order(_rec_trade("T-MINE")) is True
        ib.cancelOrder.assert_called_once_with(legacy.order)

    def test_strike_substring_false_positive_is_gone(self):
        """'45.0' is a substring of 'SPY:745.0:P:2026-12-18'."""
        stray = _broker_order(ref="", sec="OPT", strike=45.0)
        rec, ib = self._rec_with_orders([stray])
        assert rec._cancel_working_entry_order(_rec_trade("T-MINE")) is False
        ib.cancelOrder.assert_not_called()

    def test_untagged_exact_leg_match_still_cancels(self):
        legacy = _broker_order(ref="", sec="OPT", strike=745.0)
        rec, ib = self._rec_with_orders([legacy])
        assert rec._cancel_working_entry_order(_rec_trade("T-MINE")) is True
        ib.cancelOrder.assert_called_once_with(legacy.order)

    def test_terminal_orders_are_ignored(self):
        filled = _broker_order(ref="T-MINE", status="Filled")
        rec, ib = self._rec_with_orders([filled])
        assert rec._cancel_working_entry_order(_rec_trade("T-MINE")) is False
        ib.cancelOrder.assert_not_called()

    def test_wrong_expiry_untagged_option_is_not_cancelled(self):
        stray = _broker_order(ref="", sec="OPT", strike=745.0, expiry="20270115")
        rec, ib = self._rec_with_orders([stray])
        assert rec._cancel_working_entry_order(_rec_trade("T-MINE")) is False
        ib.cancelOrder.assert_not_called()


# ===========================================================================
# [6] the sweep must not book off a wedged-empty position cache
# ===========================================================================

class TestSweepRequiresAuthoritativeFlat:
    def _rec_with_pending(self):
        state = MagicMock()
        t = _rec_trade("T-STALE", TradeStatus.PENDING)
        state.get_open_trades.return_value = [t]
        rec = _reconciler(state)
        rec._cancel_working_entry_order = lambda tr: False  # no working order
        return rec, state, t

    def test_empty_cache_defers_instead_of_booking_fiction(self):
        rec, state, _ = self._rec_with_pending()
        rec._live_ibkr_leg_keys = lambda: set()  # connected but wedged-empty
        # Pre-fix: booked $0 'stale_pending_never_filled' on an empty cache.
        assert rec._sweep_stale_pending() == 0
        state.close_trade.assert_not_called()

    def test_unreadable_cache_still_defers(self):
        rec, state, _ = self._rec_with_pending()
        rec._live_ibkr_leg_keys = lambda: None
        assert rec._sweep_stale_pending() == 0
        state.close_trade.assert_not_called()

    def test_populated_cache_still_books_without_a_fresh_query(self):
        """Only the EMPTY cache is ambiguous — a cache holding other legs is
        positive evidence the stream is alive."""
        rec, state, _ = self._rec_with_pending()
        rec._live_ibkr_leg_keys = lambda: {"QQQ:500.0:C:2026-12-18"}
        assert rec._sweep_stale_pending() == 1
        assert state.close_trade.call_args.kwargs["realized_pnl"] == 0.0

    def test_client_without_a_fresh_query_surface_books_but_says_so(self):
        """A client with no authoritative-query surface (legacy client / the
        test fakes) has nothing to wait for — keep the pre-R19 booking rather
        than wedge every orphan open forever, but never silently."""
        rec, state, _ = self._rec_with_pending()
        rec._ibkr = SimpleNamespace(ib=None)  # no get_positions_fresh at all
        rec._live_ibkr_leg_keys = lambda: set()
        assert rec._sweep_stale_pending() == 1
        state.close_trade.assert_called_once()

    async def test_fresh_query_confirming_flat_allows_the_booking(self):
        rec, state, _ = self._rec_with_pending()
        rec._live_ibkr_leg_keys = lambda: set()
        rec._ibkr.get_positions_fresh = AsyncMock(return_value=[])
        assert await rec.sweep_stale_pending() == 1
        assert state.close_trade.call_args.kwargs[
            "exit_reason_detailed"] == "stale_pending_never_filled"

    async def test_fresh_query_finding_the_legs_promotes_instead(self):
        """The R2/C3 fiction this gate exists to prevent: the order DID fill,
        only the in-memory tracker died."""
        rec, state, t = self._rec_with_pending()
        state.transition.return_value = True
        rec._live_ibkr_leg_keys = lambda: set()  # wedged cache says 'nothing'
        rec._ibkr.get_positions_fresh = AsyncMock(return_value=[
            SimpleNamespace(position=-1, contract=SimpleNamespace(
                secType="OPT", symbol="SPY", strike=745.0, right="P",
                lastTradeDateOrContractMonth="20261218")),
        ])
        assert await rec.sweep_stale_pending() == 0
        state.close_trade.assert_not_called()
        state.transition.assert_called_once()

    async def test_broker_silent_on_the_fresh_query_books_nothing(self):
        rec, state, _ = self._rec_with_pending()
        rec._live_ibkr_leg_keys = lambda: set()
        rec._ibkr.get_positions_fresh = AsyncMock(return_value=None)
        assert await rec.sweep_stale_pending() == 0
        state.close_trade.assert_not_called()

    async def test_fresh_confirmation_is_consumed_exactly_once(self):
        """A confirmation must never authorise a LATER cycle's booking."""
        rec, state, _ = self._rec_with_pending()
        rec._live_ibkr_leg_keys = lambda: set()
        rec._ibkr.get_positions_fresh = AsyncMock(return_value=[])
        await rec.sweep_stale_pending()
        assert getattr(rec, "_fresh_keys_for_sweep", None) is None
        state.close_trade.reset_mock()
        assert rec._sweep_stale_pending() == 0  # sync cycle: no confirmation
        state.close_trade.assert_not_called()


# ===========================================================================
# [5] one spot classification per symbol per monitor pass
# ===========================================================================

PF_LEGS = json.dumps([
    {"strike": 95.0, "right": "P", "action": "BUY", "expiry": "2026-12-18"},
    {"strike": 98.0, "right": "P", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 102.0, "right": "C", "action": "SELL", "expiry": "2026-12-18"},
    {"strike": 105.0, "right": "C", "action": "BUY", "expiry": "2026-12-18"},
])


@dataclass
class PfTrade:
    trade_id: str = "T-A"
    entry_price: float = 1.00
    quantity: int = 1
    contract_type: str = "spread"
    strategy: str = "iron_condor"
    symbol: str = "SPY"
    direction: object = None
    legs: str = PF_LEGS
    expiry: str | None = None
    strike: float | None = None
    entry_time: str = "2026-12-01T10:00:00"
    status: object = TradeStatus.FILLED


def _pf_manager(spot: float, trades: list, clock: dict):
    """Real PortfolioManager; market_data mimics the 15s quote cache — the
    SAME Quote object (same tick timestamp) for every read within a pass."""
    mgr = PortfolioManager.__new__(PortfolioManager)
    mgr._ibkr = MagicMock()
    mgr._ibkr.ib.portfolio.return_value = []
    mgr._state = MagicMock()
    mgr._state.get_open_trades.return_value = trades
    mgr._state.get_high_water_mark.return_value = 0.0
    mgr._market_data = MagicMock()

    async def _quote(symbol):
        return Quote(symbol=symbol, bid=round(spot - 0.01, 2),
                     ask=round(spot + 0.01, 2), last=spot, volume=1_000_000,
                     timestamp=clock["ts"])
    mgr._market_data.get_quote = _quote

    async def _price(symbol):
        return spot
    mgr._market_data.get_current_price = _price
    mgr._exit_config = ExitConfig()
    mgr._earnings = None
    mgr._economic_cal = None

    async def _vol_mult(symbol):
        return 1.0
    mgr._get_volatility_stop_multiplier = _vol_mult
    mgr._pdt_guard = MagicMock()
    mgr._pdt_guard.would_be_day_trade.return_value = False
    mgr._option_position_unrealized = lambda trade, marks=None: -20.0
    mgr._get_position_delta = lambda trade: None
    return mgr


def _pf_condor(trade_id: str, dte: int = 20) -> PfTrade:
    t = PfTrade(trade_id=trade_id,
                expiry=(date.today() + timedelta(days=dte)).isoformat())
    t.direction = SimpleNamespace(value="neutral")
    return t


class TestSpotQuotePerPassClassification:
    async def test_second_position_on_one_symbol_is_not_called_frozen(
            self, monkeypatch):
        """Two condors on SPY, spot through the 98 put. Both touches are real
        and the feed is healthy — pre-fix the second position read the same
        cached quote, was classified 'frozen', and its exit was downgraded to
        a 2-tick confirmation (+30-60s on the loss cap)."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        monkeypatch.setenv("AIT_SKIP_MACRO_EVENTS", "1")
        clock = {"ts": datetime.now()}
        trades = [_pf_condor("T-A"), _pf_condor("T-B")]
        mgr = _pf_manager(spot=97.5, trades=trades, clock=clock)

        statuses = await mgr.check_positions()
        assert len(statuses) == 2
        for st in statuses:
            assert st.should_exit, f"{st.trade_id} did not exit"
            # Pre-fix T-B: no exit at all (1 of 2 agreeing ticks needed).
            assert "frozen" not in st.exit_reason
            assert st.exit_reason.startswith("short_strike_touch")
        assert "SPY" not in getattr(mgr, "_frozen_alerted", set())

    async def test_a_genuinely_frozen_feed_is_still_detected(self, monkeypatch):
        """Regression guard for the fix itself: the memo is pass-scoped, so a
        feed that stops advancing BETWEEN passes must still read 'frozen'."""
        monkeypatch.setenv("AIT_CREDIT_TOUCH_STOP", "1")
        monkeypatch.setenv("AIT_SKIP_MACRO_EVENTS", "1")
        clock = {"ts": datetime.now()}
        trades = [_pf_condor("T-A"), _pf_condor("T-B")]
        mgr = _pf_manager(spot=97.5, trades=trades, clock=clock)

        await mgr.check_positions()                 # pass 1: fresh
        statuses = await mgr.check_positions()      # pass 2: same tick time
        for st in statuses:
            assert not st.should_exit               # holding for corroboration
        assert "SPY" in mgr._frozen_alerted
        # ...and a resumed feed clears it again.
        clock["ts"] = clock["ts"] + timedelta(seconds=30)
        statuses = await mgr.check_positions()
        for st in statuses:
            assert st.should_exit
            assert st.exit_reason.startswith("short_strike_touch")

    async def test_memo_does_not_outlive_the_pass(self):
        clock = {"ts": datetime.now()}
        mgr = _pf_manager(spot=100.0, trades=[], clock=clock)
        await mgr.check_positions()
        assert getattr(mgr, "_pass_quote_verdicts", None) in (None, {})

    async def test_report_path_neither_fills_nor_reads_the_memo(self):
        """A4 invariant: persist=False stays a pure read."""
        clock = {"ts": datetime.now()}
        mgr = _pf_manager(spot=100.0, trades=[], clock=clock)
        mgr._pass_quote_verdicts = {}
        await mgr._spot_quote("SPY", persist=False)
        assert mgr._pass_quote_verdicts == {}

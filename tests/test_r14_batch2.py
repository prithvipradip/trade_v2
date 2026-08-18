"""R14 batch 2 — the sizing-base correction plus the Tier-3 resilience items.

Covers six fixes:
  1. capital-at-risk scaling: retained risk = per-contract max_loss x executed
     contracts (was stored per-contract, understating the verdict denominator
     and the aggregate risk guard the moment sizing scales past 1 lot).
  2. single-leg exit price bound: a short buyback is a LIMIT (never a market
     order — unbounded fill), short puts capped at strike, no-quote defers; a
     long-option SELL keeps a bounded market fallback.
  3. MTM brake gap-blindness: the next day's baseline is pre-stamped with the
     prior close's unrealized so a gap shows as the full move from it.
  4. exit-reject backoff + escalating page.
  5. reconcile() fresh-query hatch: a genuinely-flat book is booked only when
     an authoritative re-query confirms zero options.
  6. keeper launch verification + digest mirror content-age.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import TradeStatus


# ==========================================================================
# 1. capital-at-risk scaling  (_position_capital_at_risk)
# ==========================================================================

def _sig(max_loss: float, defined: bool = True):
    return SimpleNamespace(max_loss=max_loss, is_defined_risk=defined)


class TestCapitalAtRiskScaling:
    def test_one_contract_equals_per_contract(self):
        assert TradingOrchestrator._position_capital_at_risk(_sig(400.0), 1) == 400.0

    def test_scales_with_executed_quantity(self):
        # THE bug: a $400/contract condor executed at 3 lots risks $1200, not
        # $400. Per-contract storage understated the verdict denominator and
        # the aggregate guard by the lot count.
        assert TradingOrchestrator._position_capital_at_risk(_sig(400.0), 3) == 1200.0

    def test_undefined_risk_is_zero(self):
        assert TradingOrchestrator._position_capital_at_risk(_sig(0.0, defined=False), 5) == 0.0

    def test_quantity_floored_at_one(self):
        assert TradingOrchestrator._position_capital_at_risk(_sig(250.0), 0) == 250.0

    def test_none_max_loss_safe(self):
        assert TradingOrchestrator._position_capital_at_risk(_sig(None), 2) == 0.0


# ==========================================================================
# 2. single-leg exit price bound  (_close_single_leg)
# ==========================================================================

@pytest.fixture(autouse=True)
def _fast_sleep(monkeypatch):
    import asyncio
    real = asyncio.sleep

    async def _instant(_s):
        await real(0)

    monkeypatch.setattr("ait.bot.orchestrator.asyncio.sleep", _instant)


def _single_leg_orch(bid, ask, cross=0.05):
    orch = TradingOrchestrator.__new__(TradingOrchestrator)
    notes: list[str] = []
    ibkr = MagicMock()
    ibkr.place_order = AsyncMock(
        return_value=SimpleNamespace(order=SimpleNamespace(orderId=7)))
    ibkr.ib.reqMktData = MagicMock()
    ibkr.ib.cancelMktData = MagicMock()
    ibkr.ib.ticker = MagicMock(return_value=SimpleNamespace(bid=bid, ask=ask))
    orch._ibkr = ibkr
    orch._settings = SimpleNamespace(exit=SimpleNamespace(exit_cross_amount=cross))

    async def _notify(m):
        notes.append(m)

    orch._send_notification = _notify
    orch._alert_gate = lambda *a, **k: True
    return orch, ibkr.place_order, notes


def _trade(strategy, contract_type, strike=None, qty=1):
    return SimpleNamespace(
        trade_id="T-SL", symbol="SPY", strategy=strategy,
        contract_type=contract_type, strike=strike, quantity=qty, expiry="2026-08-21")


class TestSingleLegExitBound:
    async def test_long_sell_uses_limit_off_bid(self):
        orch, place, _ = _single_leg_orch(bid=2.00, ask=2.10)
        t = _trade("long_call", "call")
        await orch._close_single_leg(t, MagicMock(), "SELL")
        o = place.call_args.args[1]
        assert o.orderType == "LMT"
        assert o.lmtPrice == pytest.approx(1.95)  # bid - cross

    async def test_long_sell_no_quote_falls_back_to_market(self):
        # Long option: worst market fill is $0, premium already sunk — bounded.
        orch, place, _ = _single_leg_orch(bid=float("nan"), ask=float("nan"))
        t = _trade("long_put", "put")
        await orch._close_single_leg(t, MagicMock(), "SELL")
        o = place.call_args.args[1]
        assert o.orderType == "MKT"

    async def test_short_buyback_is_limit_never_market(self):
        orch, place, _ = _single_leg_orch(bid=0.40, ask=0.50)
        t = _trade("cash_secured_put", "put", strike=100.0)
        await orch._close_single_leg(t, MagicMock(), "BUY")
        o = place.call_args.args[1]
        assert o.orderType == "LMT"
        assert o.lmtPrice == pytest.approx(0.55)  # ask + cross

    async def test_short_put_buyback_capped_at_strike(self):
        # Garbage ask on a short put: the most it can ever cost to close is the
        # strike (full intrinsic as underlying -> 0). Cap there.
        orch, place, _ = _single_leg_orch(bid=90.0, ask=150.0)
        t = _trade("cash_secured_put", "put", strike=100.0)
        await orch._close_single_leg(t, MagicMock(), "BUY")
        o = place.call_args.args[1]
        assert o.lmtPrice == pytest.approx(100.0)

    async def test_short_call_buyback_no_structural_cap(self):
        # A short call has no strike ceiling; the LIMIT still prevents a lone
        # garbage print from filling beyond it, but it is not capped at a strike.
        orch, place, _ = _single_leg_orch(bid=3.00, ask=3.20)
        t = _trade("covered_call", "call", strike=100.0)
        await orch._close_single_leg(t, MagicMock(), "BUY")
        o = place.call_args.args[1]
        assert o.lmtPrice == pytest.approx(3.25)

    async def test_short_buyback_no_quote_defers_never_markets(self):
        orch, place, notes = _single_leg_orch(bid=float("nan"), ask=float("nan"))
        t = _trade("cash_secured_put", "put", strike=100.0)
        result = await orch._close_single_leg(t, MagicMock(), "BUY")
        assert result is None
        place.assert_not_called()
        assert any("EXIT DEFERRED" in n for n in notes)


# ==========================================================================
# 3. MTM baseline pre-stamp  (_prestamp_mtm_baseline)
# ==========================================================================

class TestMtmBaselinePrestamp:
    def test_stamps_next_trading_day_with_close_unrealized(self, monkeypatch):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._state = MagicMock()
        monkeypatch.setattr(
            "ait.bot.orchestrator.next_market_open",
            lambda: datetime(2026, 7, 16, 9, 30),
        )
        orch._prestamp_mtm_baseline(-1234.5)
        orch._state.set_state.assert_called_once_with("mtm_sod_2026-07-16", "-1234.5")

    def test_prestamp_failure_is_swallowed(self, monkeypatch):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._state = MagicMock()
        orch._state.set_state.side_effect = RuntimeError("db locked")
        monkeypatch.setattr(
            "ait.bot.orchestrator.next_market_open",
            lambda: datetime(2026, 7, 16, 9, 30),
        )
        orch._prestamp_mtm_baseline(0.0)  # must not raise


# ==========================================================================
# 4. exit-reject backoff  (_exit_retry_ready)
# ==========================================================================

def _backoff_orch(monkeypatch):
    orch = TradingOrchestrator.__new__(TradingOrchestrator)
    clock = {"t": 1000.0}
    monkeypatch.setattr("ait.bot.orchestrator.time.monotonic", lambda: clock["t"])
    notes: list[str] = []

    async def _notify(m):
        notes.append(m)

    orch._send_notification = _notify
    orch._alert_gate_last = {}
    return orch, clock, notes


class TestExitRejectBackoff:
    def test_first_attempt_always_allowed(self, monkeypatch):
        orch, clock, _ = _backoff_orch(monkeypatch)
        assert orch._exit_retry_ready("T1") is True

    def test_rapid_reretry_is_blocked_then_allowed_after_backoff(self, monkeypatch):
        orch, clock, _ = _backoff_orch(monkeypatch)
        assert orch._exit_retry_ready("T1") is True      # strike 1, no backoff
        clock["t"] += 5                                   # fast reject re-request
        assert orch._exit_retry_ready("T1") is True      # strike 2 -> 30s backoff
        clock["t"] += 5
        assert orch._exit_retry_ready("T1") is False     # inside the 30s window
        clock["t"] += 30
        assert orch._exit_retry_ready("T1") is True      # window elapsed

    def test_slow_reprice_cadence_does_not_escalate(self, monkeypatch):
        orch, clock, _ = _backoff_orch(monkeypatch)
        assert orch._exit_retry_ready("T1") is True
        clock["t"] += 300                                 # normal working-order re-price
        assert orch._exit_retry_ready("T1") is True       # allowed, strikes reset
        assert orch._exit_attempts["T1"]["strikes"] == 1

    def test_pages_after_threshold_rapid_rejects(self, monkeypatch):
        import asyncio
        orch, clock, notes = _backoff_orch(monkeypatch)

        async def drive():
            orch._exit_retry_ready("T1")            # strike 1
            for _ in range(6):
                clock["t"] += 1
                if not orch._exit_retry_ready("T1"):
                    # advance past whatever backoff is pending
                    clock["t"] += 200
                    orch._exit_retry_ready("T1")
            await asyncio.sleep(0)  # let the fire-and-forget page task run

        asyncio.run(drive())
        assert any("REPEATEDLY REJECTED" in n for n in notes)

    def test_distinct_trades_independent(self, monkeypatch):
        orch, clock, _ = _backoff_orch(monkeypatch)
        assert orch._exit_retry_ready("A") is True     # A strike 1
        assert orch._exit_retry_ready("B") is True     # B strike 1
        clock["t"] += 5
        assert orch._exit_retry_ready("A") is True     # A strike 2 -> sets 30s backoff
        clock["t"] += 1
        # A is now inside its backoff window; B, untouched, is still free to act.
        assert orch._exit_retry_ready("A") is False    # A backing off
        assert orch._exit_retry_ready("B") is True     # B independent


# ==========================================================================
# 5. reconcile() fresh-query hatch  (get_positions_fresh confirmation)
# ==========================================================================
# Covered structurally by test_exit_order_safety.py::TestPositionLiveness for
# position_liveness; here we assert the reconcile-level guard only mass-books
# when a FRESH re-query confirms zero options.

from ait.execution.reconciler import PositionReconciler  # noqa: E402


def _recon_for_hatch(fresh):
    rec = PositionReconciler.__new__(PositionReconciler)
    ibkr = MagicMock()
    ibkr.connected = True
    ibkr.get_positions.return_value = []            # cached: zero options
    ibkr.get_portfolio.return_value = []
    ibkr.get_positions_fresh = AsyncMock(return_value=fresh)
    rec._ibkr = ibkr
    state = MagicMock()
    trade = SimpleNamespace(
        trade_id="T-GONE", symbol="SPY", strategy="iron_condor",
        contract_type="iron_condor", quantity=1, entry_price=1.0,
        expiry="2026-08-21", strike=None,
        legs='[{"strike":713.0,"right":"P","action":"BUY","expiry":"2026-08-21"},'
             '{"strike":715.0,"right":"P","action":"SELL","expiry":"2026-08-21"}]',
        status=TradeStatus.FILLED,
    )
    state.get_open_trades.return_value = [trade]
    rec._state = state
    return rec, state


def _opt_pos(symbol, strike, right, expiry):
    return SimpleNamespace(position=1.0, contract=SimpleNamespace(
        secType="OPT", symbol=symbol, strike=strike, right=right,
        lastTradeDateOrContractMonth=expiry))


class TestReconcileFreshHatch:
    async def test_fresh_confirms_flat_books_the_trade(self):
        rec, state = _recon_for_hatch(fresh=[])   # fresh re-query: genuinely flat
        await rec.reconcile()
        assert state.close_trade.called

    async def test_fresh_none_keeps_refusing(self):
        rec, state = _recon_for_hatch(fresh=None)  # broker won't confirm
        await rec.reconcile()
        assert not state.close_trade.called

    async def test_fresh_shows_positions_keeps_refusing(self):
        # Cache was stale — the options actually exist. Must not mass-close.
        rec, state = _recon_for_hatch(
            fresh=[_opt_pos("SPY", 713.0, "P", "20260821")])
        await rec.reconcile()
        assert not state.close_trade.called


# ==========================================================================
# 6a. digest mirror content-age  (_mirror_content_age_hours)
# ==========================================================================

from ait.orchestration import master as _master  # noqa: E402


def _write_mirror(path, newest_iso):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE trades (entry_time TEXT, exit_time TEXT)")
    if newest_iso is not None:
        con.execute("INSERT INTO trades VALUES (?, ?)", ("2026-07-01T10:00:00", newest_iso))
    con.commit()
    con.close()


class TestMirrorContentAge:
    def test_age_from_newest_trade_not_mtime(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / "Documents" / "ait_backups").mkdir(parents=True)
        mirror = home / "Documents" / "ait_backups" / "ait_state.latest.db"
        newest = (datetime.now() - timedelta(hours=5)).isoformat()
        _write_mirror(mirror, newest)
        monkeypatch.setattr(_master.Path, "home", staticmethod(lambda: home))
        age = _master._mirror_content_age_hours()
        assert age is not None
        assert 4.5 < age < 5.5

    def test_missing_mirror_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_master.Path, "home", staticmethod(lambda: tmp_path))
        assert _master._mirror_content_age_hours() is None

    def test_empty_mirror_returns_none(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / "Documents" / "ait_backups").mkdir(parents=True)
        _write_mirror(home / "Documents" / "ait_backups" / "ait_state.latest.db", None)
        monkeypatch.setattr(_master.Path, "home", staticmethod(lambda: home))
        assert _master._mirror_content_age_hours() is None


# ==========================================================================
# 6b. keeper launch verification  (_verify_launch)
# ==========================================================================

class TestVerifyLaunch:
    def test_process_alive_through_grace_verifies(self, monkeypatch):
        from ait.orchestration.master import BotManager
        mgr = BotManager.__new__(BotManager)
        mgr._proc = SimpleNamespace(poll=lambda: None, pid=123, returncode=None)
        mgr._LAUNCH_GRACE_S = 0.02
        alerts = []
        monkeypatch.setattr("ait.orchestration.master._alert", lambda m: alerts.append(m))
        monkeypatch.setattr("ait.orchestration.master.time.sleep", lambda s: None)
        assert mgr._verify_launch() is True
        assert not alerts

    def test_immediate_death_alerts_and_fails(self, monkeypatch):
        from ait.orchestration.master import BotManager
        mgr = BotManager.__new__(BotManager)
        mgr._proc = SimpleNamespace(poll=lambda: 1, pid=123, returncode=1)
        mgr._LAUNCH_GRACE_S = 0.05
        alerts = []
        monkeypatch.setattr("ait.orchestration.master._alert", lambda m: alerts.append(m))
        monkeypatch.setattr("ait.orchestration.master.time.sleep", lambda s: None)
        assert mgr._verify_launch() is False
        assert any("RELAUNCH FAILED" in a for a in alerts)

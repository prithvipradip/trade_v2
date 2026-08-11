"""R17 data-integrity + risk-math fixes — external review findings 8-11,
each test fails against the pre-fix code.

  [8]  save_intraday's INSERT OR REPLACE let MIDPOINT bars silently
       overwrite real TRADES bars (source isn't part of the primary key)
  [9]  risk/manager.py gate 6c (same-symbol concentration cap) summed
       market_value (credit collected) instead of real max-loss risk
  [10] reconciler.py exit-order matching could misassign between two
       same-symbol multi-leg positions both mid-close after a restart
  [11] broker/account.py never escalated beyond a log line when account
       data went stale
"""
from __future__ import annotations

import datetime as dt
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from ait.data.historical import HistoricalDataStore


def _bars(n: int, start: dt.datetime, price: float = 100.0) -> pd.DataFrame:
    times = [start + dt.timedelta(minutes=5 * i) for i in range(n)]
    return pd.DataFrame(
        {"Open": [price] * n, "High": [price + 0.1] * n,
         "Low": [price - 0.1] * n, "Close": [price] * n,
         "Volume": [1000] * n},
        index=pd.DatetimeIndex(times),
    )


class TestBackfillCannotDowngradeTradesData:
    def test_midpoint_backfill_does_not_overwrite_trades_bar(self, tmp_path):
        store = HistoricalDataStore(db_path=tmp_path / "hist.db")
        ts = dt.datetime(2026, 3, 2, 9, 30)
        df = _bars(1, ts, price=101.23)

        store.save_intraday("SPY", df, interval="5m", source="TRADES")
        # Pre-fix: INSERT OR REPLACE clobbers the row wholesale, including
        # the real traded price, with the MIDPOINT re-backfill.
        store.save_intraday("SPY", _bars(1, ts, price=999.0),
                             interval="5m", source="MIDPOINT")

        con = sqlite3.connect(tmp_path / "hist.db")
        row = con.execute(
            "SELECT close, source FROM intraday_prices WHERE symbol='SPY'"
        ).fetchone()
        con.close()
        assert row[0] == pytest.approx(101.23)
        assert row[1] == "TRADES"

    def test_trades_backfill_still_overwrites_midpoint(self, tmp_path):
        """A genuine re-backfill with real fill data must still win."""
        store = HistoricalDataStore(db_path=tmp_path / "hist.db")
        ts = dt.datetime(2026, 3, 2, 9, 30)

        store.save_intraday("SPY", _bars(1, ts, price=999.0),
                             interval="5m", source="MIDPOINT")
        store.save_intraday("SPY", _bars(1, ts, price=101.23),
                             interval="5m", source="TRADES")

        con = sqlite3.connect(tmp_path / "hist.db")
        row = con.execute(
            "SELECT close, source FROM intraday_prices WHERE symbol='SPY'"
        ).fetchone()
        con.close()
        assert row[0] == pytest.approx(101.23)
        assert row[1] == "TRADES"

    def test_resave_with_same_source_is_idempotent(self, tmp_path):
        store = HistoricalDataStore(db_path=tmp_path / "hist.db")
        ts = dt.datetime(2026, 3, 2, 9, 30)
        df = _bars(3, ts)
        assert store.save_intraday("SPY", df, interval="5m", source="TRADES") == 3
        assert store.save_intraday("SPY", df, interval="5m", source="TRADES") == 3

        con = sqlite3.connect(tmp_path / "hist.db")
        n = con.execute(
            "SELECT COUNT(*) FROM intraday_prices WHERE symbol='SPY'"
        ).fetchone()[0]
        con.close()
        assert n == 3


class TestSymbolConcentrationUsesRealRisk:
    def _manager(self, positions):
        from ait.risk.manager import RiskManager

        mgr = RiskManager.__new__(RiskManager)
        mgr._open_positions = positions
        mgr._state = None
        return mgr

    def test_credit_strategy_uses_max_loss_not_credit_collected(self):
        # Two SPY iron condors: $300 credit each, but $2,000 max_loss each.
        mgr = self._manager([
            {"symbol": "SPY", "strategy": "iron_condor", "market_value": 300.0,
             "max_loss": 2000.0, "quantity": 1},
            {"symbol": "SPY", "strategy": "iron_condor", "market_value": 300.0,
             "max_loss": 2000.0, "quantity": 1},
        ])
        # Pre-fix: symbol_exposure would be 600.0 (credit collected).
        assert mgr._symbol_capital_at_risk("SPY") == pytest.approx(4000.0)

    def test_other_symbols_excluded(self):
        mgr = self._manager([
            {"symbol": "SPY", "strategy": "iron_condor", "market_value": 300.0,
             "max_loss": 2000.0, "quantity": 1},
            {"symbol": "QQQ", "strategy": "iron_condor", "market_value": 300.0,
             "max_loss": 1500.0, "quantity": 1},
        ])
        assert mgr._symbol_capital_at_risk("SPY") == pytest.approx(2000.0)

    def test_falls_back_to_backfill_when_max_loss_missing(self):
        mgr = self._manager([
            {"symbol": "SPY", "strategy": "iron_condor", "market_value": 300.0,
             "max_loss": 0.0, "quantity": 1},
        ])
        mgr._backfilled_position_risk = lambda pos, idx: (750.0, "structure_width")
        mgr._open_trade_index = lambda: {}
        assert mgr._symbol_capital_at_risk("SPY") == pytest.approx(750.0)


class TestAccountStaleDataEscalates:
    def _account(self):
        from ait.broker.account import AccountManager

        acct = AccountManager.__new__(AccountManager)
        acct._client = MagicMock()
        acct._cache_ttl = 30
        acct._notify_cb = AsyncMock()
        acct._circuit_breaker = MagicMock()
        acct._stale_escalated = False
        return acct

    async def test_past_threshold_notifies_and_trips_breaker_once(self):
        acct = self._account()
        await acct._handle_stale(1000.0)  # past ESCALATE_STALE_SECONDS=900
        acct._notify_cb.assert_awaited_once()
        acct._circuit_breaker.record_api_failure.assert_called_once()

        # A second call while still stale must not double-fire.
        await acct._handle_stale(1100.0)
        acct._notify_cb.assert_awaited_once()
        acct._circuit_breaker.record_api_failure.assert_called_once()

    async def test_below_threshold_does_not_escalate(self):
        acct = self._account()
        await acct._handle_stale(400.0)  # >300s (logs) but <900s (no escalation)
        acct._notify_cb.assert_not_awaited()
        acct._circuit_breaker.record_api_failure.assert_not_called()

    async def test_fresh_fetch_rearms_the_latch(self):
        acct = self._account()
        acct._stale_escalated = True
        acct._last_fetch = 0.0
        acct._client.get_account_values = AsyncMock(return_value={
            "NetLiquidation": "100000", "BuyingPower": "50000",
            "AvailableFunds": "50000", "ExcessLiquidity": "50000",
            "MaintMarginReq": "0", "InitMarginReq": "0",
            "UnrealizedPnL": "0", "RealizedPnL": "0", "CashBalance": "0",
        })
        await acct.get_snapshot(force_refresh=True)
        assert acct._stale_escalated is False


class TestReconcilerOrderRefDisambiguation:
    def _reconciler(self):
        from ait.execution.reconciler import PositionReconciler
        return PositionReconciler.__new__(PositionReconciler)

    def _trade(self, trade_id, symbol="SPY"):
        return SimpleNamespace(
            trade_id=trade_id, symbol=symbol, strategy="iron_condor",
            legs="[]", strike=None, expiry=None,
        )

    def test_exact_order_ref_match_disambiguates_same_symbol_combos(self):
        recon = self._reconciler()
        trade_a = self._trade("T-A")
        trade_b = self._trade("T-B")
        working = [
            (111, {"SPY:BAG"}, "SPY", "T-A"),
            (222, {"SPY:BAG"}, "SPY", "T-B"),
        ]
        assert recon._working_exit_order_for(trade_a, working) == 111
        assert recon._working_exit_order_for(trade_b, working) == 222

    def test_tagged_order_never_claimed_by_a_different_trade(self):
        recon = self._reconciler()
        trade_c = self._trade("T-C")  # no working order of its own
        working = [(111, {"SPY:BAG"}, "SPY", "T-A")]
        # Pre-fix: symbol-only match would have claimed order 111 for T-C too.
        assert recon._working_exit_order_for(trade_c, working) is None

    def test_falls_back_to_symbol_heuristic_when_order_ref_absent(self):
        """Orders placed before this change carry no order_ref."""
        recon = self._reconciler()
        trade = self._trade("T-LEGACY")
        working = [(111, {"SPY:BAG"}, "SPY", "")]
        assert recon._working_exit_order_for(trade, working) == 111

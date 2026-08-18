"""R16: the backtest engine's short-strike touch stop.

Live closes a credit structure the moment spot REACHES a short strike (30s
monitor). The engine only ever read the daily CLOSE, so an intraday pierce
that recovered by the bell scored as an untouched winner — every study to
date measured a different exit policy than production runs.

Daily High/Low bracket the true intraday path, so they detect the touch
without intraday bars.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from ait.backtesting.engine import Backtester


def _pos(short_put=750.0, short_call=784.0, entry="2026-08-04"):
    return {
        "strategy": "iron_condor", "trade_type": "credit",
        "entry_date": entry, "entry_price": 4.31, "contracts": 1,
        "short_put_strike": short_put, "short_call_strike": short_call,
        "long_put_strike": short_put - 35, "long_call_strike": short_call + 31,
        "expiry_date": str(date.fromisoformat(entry) + timedelta(days=32)),
        "entry_iv": 0.13, "underlying_at_entry": 767.0,
        "max_loss_per_share": 30.69, "high_water_mark": 0.0,
        "n_legs": 4, "option_type": "iron_condor",
    }


def _bt(**kw):
    bt = Backtester.__new__(Backtester)
    bt._credit_loss_limit_mult = 0.0
    bt._touch_stop_enabled = kw.get("touch", True)
    bt._economic_cal = None
    return bt


def _row(low, high, close=None):
    return pd.Series({"Low": low, "High": high,
                      "Close": close if close is not None else (low + high) / 2})


class TestTouchDetection:
    def test_put_side_pierce_triggers(self):
        bt = _bt()
        # dipped to 749 intraday, recovered to 765 by the close: the close-only
        # engine saw a healthy winner; live would have exited at 750.
        out = bt._check_exit_credit(_pos(), pnl_pct=0.20,
                                    current_date=date(2026, 8, 12),
                                    row=_row(low=749.0, high=766.0, close=765.0))
        assert out is not None
        assert out["exit_reason"] == "touch_stop"
        assert out["touch_underlying"] == pytest.approx(750.0)

    def test_call_side_pierce_triggers(self):
        bt = _bt()
        out = bt._check_exit_credit(_pos(), pnl_pct=0.20,
                                    current_date=date(2026, 8, 12),
                                    row=_row(low=770.0, high=785.0, close=772.0))
        assert out is not None and out["exit_reason"] == "touch_stop"
        assert out["touch_underlying"] == pytest.approx(784.0)

    def test_untouched_day_does_not_trigger(self):
        bt = _bt()
        out = bt._check_exit_credit(_pos(), pnl_pct=0.20,
                                    current_date=date(2026, 8, 12),
                                    row=_row(low=752.0, high=780.0))
        # inside both shorts -> no touch; 20% is below the 50% target
        assert out is None

    def test_exact_touch_counts(self):
        # live triggers on REACHING the strike, not exceeding it
        bt = _bt()
        out = bt._check_exit_credit(_pos(), pnl_pct=0.10,
                                    current_date=date(2026, 8, 12),
                                    row=_row(low=750.0, high=770.0))
        assert out is not None and out["exit_reason"] == "touch_stop"


class TestPrecedenceAndFlag:
    def test_touch_precedes_take_profit(self):
        # a day that both hits the profit target AND pierces a short strike
        # must book the TOUCH — live's 30s monitor sees the pierce first and
        # the position is no longer the one the target was measured on.
        bt = _bt()
        out = bt._check_exit_credit(_pos(), pnl_pct=0.99,
                                    current_date=date(2026, 8, 12),
                                    row=_row(low=749.0, high=766.0))
        assert out["exit_reason"] == "touch_stop"

    def test_flag_off_restores_close_only(self):
        bt = _bt(touch=False)
        out = bt._check_exit_credit(_pos(), pnl_pct=0.20,
                                    current_date=date(2026, 8, 12),
                                    row=_row(low=749.0, high=766.0))
        assert out is None  # the pre-R16 behaviour, for comparison runs only

    def test_missing_high_low_is_safe(self):
        # older frames / synthetic rows without OHLC must not crash
        bt = _bt()
        out = bt._check_exit_credit(_pos(), pnl_pct=0.20,
                                    current_date=date(2026, 8, 12),
                                    row=pd.Series({"Close": 765.0}))
        assert out is None

    def test_no_row_is_safe(self):
        bt = _bt()
        assert bt._check_exit_credit(_pos(), pnl_pct=0.20,
                                     current_date=date(2026, 8, 12),
                                     row=None) is None


class TestWiring:
    def test_dispatch_threads_row(self):
        import inspect
        src = inspect.getsource(Backtester._dispatch_exit_check)
        assert "row" in inspect.signature(
            Backtester._dispatch_exit_check).parameters
        assert "_check_exit_credit(pos, pnl_pct, current_date, row)" in src

    def test_exit_repriced_at_touched_strike(self):
        import inspect
        src = inspect.getsource(Backtester._check_exit)
        assert 'result.get("exit_reason") == "touch_stop"' in src
        assert "_reprice_position(pos, _tu" in src

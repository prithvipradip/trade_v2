"""W77: expiry settlement, assignment and the overnight gap (research engine).

Layers, each exercising the real code path (no source-string assertions except the AST class
guards, which parse the tree rather than grep it):

  * ``ait.backtesting.settlement``  pure rules: every status, the boundaries, worse-of, fees
  * parity with ``PositionReconciler._structure_intrinsic`` (the live booking) over random condors
  * the engine's bell handling (daily + intraday paths) and a full ``run()`` with an assignment
  * legacy stays legacy; partial engines (``Backtester.__new__``) stay legacy
  * the replay: ``settlement_replay`` / ``research_reasons_for`` on synthetic fixtures only
    (never reads ``data/replay``, which CI does not have)
"""
from __future__ import annotations

import ast
import inspect
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from ait.backtesting import replay as rp
from ait.backtesting import settlement as st
from ait.backtesting.engine import Backtester
from ait.config.settings import BacktestConfig
from ait.utils.time import ET

import tests.test_w74_engine_realism as W74  # fixtures: _bt, _frame, _pos, _Store, _no_take_profit


# --------------------------------------------------------------------------
# pure rules
# --------------------------------------------------------------------------

def _condor(sc=104.0, lc=107.0, sp=96.0, lp=93.0):
    return [st.SettleLeg(sc, "C", "short"), st.SettleLeg(lc, "C", "long"),
            st.SettleLeg(sp, "P", "short"), st.SettleLeg(lp, "P", "long")]


class TestStatuses:
    def test_otm_everything_expires(self):
        assert st.settle_legs(_condor(), 100.0, 0.10) == ["expired"] * 4

    def test_long_exercise_threshold_is_one_cent(self):
        long_call = st.SettleLeg(100.0, "C", "long")
        assert st.settle_status(long_call, 100.009, 0.10) == "expired"
        assert st.settle_status(long_call, 100.02, 0.10) == "exercised"
        long_put = st.SettleLeg(100.0, "P", "long")
        assert st.settle_status(long_put, 99.991, 0.10) == "expired"
        assert st.settle_status(long_put, 99.98, 0.10) == "exercised"

    def test_short_pin_band_boundaries_both_sides_of_the_strike(self):
        sc = st.SettleLeg(100.0, "C", "short")
        assert st.settle_status(sc, 100.10, 0.10) == "pin"        # |itm| == band
        assert st.settle_status(sc, 100.11, 0.10) == "assigned"   # ITM beyond the band
        assert st.settle_status(sc, 99.90, 0.10) == "pin"
        assert st.settle_status(sc, 99.89, 0.10) == "expired"     # OTM beyond the band
        sp = st.SettleLeg(100.0, "P", "short")
        assert st.settle_status(sp, 99.90, 0.10) == "pin"
        assert st.settle_status(sp, 99.89, 0.10) == "assigned"
        assert st.settle_status(sp, 100.11, 0.10) == "expired"

    def test_exact_boundaries_are_inclusive(self):
        # strike 0.0 makes S - K exact in binary floating point
        assert st.settle_status(st.SettleLeg(0.0, "C", "long"), 0.01, 0.10) == "exercised"
        assert st.settle_status(st.SettleLeg(0.0, "C", "short"), 0.10, 0.10) == "pin"
        assert st.settle_status(st.SettleLeg(0.0, "C", "short"), -0.10, 0.10) == "pin"
        assert st.settle_status(st.SettleLeg(0.0, "C", "short"), 0.0, 0.0) == "pin"

    def test_zero_band_still_needs_the_exercise_threshold_to_assign(self):
        sc = st.SettleLeg(100.0, "C", "short")
        assert st.settle_status(sc, 100.005, 0.0) == "expired"    # ITM by < $0.01: not exercised
        assert st.settle_status(sc, 100.02, 0.0) == "assigned"

    def test_scenarios_enumerate_pins_only(self):
        assert len(st.scenarios(["expired", "assigned"])) == 1
        two = st.scenarios(["pin", "expired", "pin", "exercised"])
        assert len(two) == 4
        assert {s[0] for s in two} == {"assigned", "expired"}
        assert all(s[1] == "expired" and s[3] == "exercised" for s in two)


class TestStockAndCost:
    def test_delivery_signs(self):
        legs = [st.SettleLeg(10, "C", "short"), st.SettleLeg(20, "C", "long"),
                st.SettleLeg(30, "P", "short"), st.SettleLeg(40, "P", "long")]
        got = st.stock_legs(legs, ["assigned", "exercised", "assigned", "exercised"])
        assert [(s.q, s.K, s.assigned) for s in got] == [
            (-1, 10, True), (+1, 20, False), (+1, 30, True), (-1, 40, False)]

    def test_both_calls_itm_net_zero_cost_is_the_width_and_independent_of_open(self):
        legs = _condor()
        a = st.worst_case(legs, 110.0, 0.10, 110.0, 0.015, 0.01)
        b = st.worst_case(legs, 110.0, 0.10, 1.0, 0.015, 0.01)       # wildly different "open"
        assert a.net_shares == 0 and a.cost == pytest.approx(3.0) and b.cost == pytest.approx(3.0)
        assert a.statuses == ("assigned", "exercised", "expired", "expired")
        assert not st.needs_overnight(legs, 110.0, 0.10)
        # fee only on the ASSIGNED share; nothing to close out (net zero)
        assert a.fees == pytest.approx(0.015)

    def test_one_sided_short_call_carries_shares_marked_at_the_open(self):
        legs = _condor()
        assert st.needs_overnight(legs, 105.0, 0.10)
        res = st.worst_case(legs, 105.0, 0.10, 106.5, 0.015, 0.01)
        assert res.net_shares == -1
        assert res.cost == pytest.approx(2.5)             # short stock at 104, bought back at 106.5
        assert res.fees == pytest.approx(0.015 + 0.01)    # assignment + closing the net share
        gap_down = st.worst_case(legs, 105.0, 0.10, 102.0, 0.015, 0.01)
        assert gap_down.cost == pytest.approx(-2.0)       # the carried short stock gains

    def test_one_sided_short_put(self):
        res = st.worst_case(_condor(), 94.5, 0.10, 93.0, 0.015, 0.01)
        assert res.net_shares == +1 and res.cost == pytest.approx(3.0)   # long stock at 96, sold 93

    def test_worse_of_picks_the_costlier_pin_scenario_at_the_open(self):
        legs = _condor()
        up = st.worst_case(legs, 103.95, 0.10, 106.0, 0.015, 0.01)       # pin, gaps up
        assert up.statuses[0] == "assigned" and up.cost == pytest.approx(2.0)
        down = st.worst_case(legs, 103.95, 0.10, 100.0, 0.015, 0.01)     # pin, gaps down
        assert down.statuses[0] == "expired"                              # not assigned is worse
        assert down.cost == 0.0 and down.net_shares == 0

    def test_exercised_delivery_is_free_but_assigned_delivery_is_not(self):
        got = st.unit_fees(st.stock_legs(_condor(), ["assigned", "exercised", "expired", "expired"]),
                           0.015, 0.01)
        assert got == pytest.approx(0.015)


class TestParityWithTheLiveBooking:
    """The settlement cost at O == S equals what the reconciler books for the same expiry."""

    def test_random_condors_match_structure_intrinsic(self):
        from ait.execution.reconciler import PositionReconciler

        rng = np.random.default_rng(77)
        for _ in range(600):
            sp = int(rng.integers(60, 140))
            lp = sp - int(rng.integers(1, 8))
            sc = sp + int(rng.integers(2, 12))
            lc = sc + int(rng.integers(1, 8))
            S = float(rng.integers(40, 170)) + float(rng.choice([0.37, 0.61, 0.83]))  # never in a pin band
            legs = [{"strike": sc, "right": "C", "action": "SELL"},
                    {"strike": lc, "right": "C", "action": "BUY"},
                    {"strike": sp, "right": "P", "action": "SELL"},
                    {"strike": lp, "right": "P", "action": "BUY"}]
            trade = SimpleNamespace(legs=json.dumps(legs), strategy="iron_condor", strike=None)
            live = PositionReconciler._structure_intrinsic(trade, S)
            res = st.worst_case(_condor(sc, lc, sp, lp), S, 0.10, S, 0.0, 0.0)
            assert res.cost == pytest.approx(live, abs=1e-9), (sc, lc, sp, lp, S)


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

EXPIRY = date(2024, 3, 4)          # a Monday: the next session is Tuesday
ROW = lambda close: pd.Series({"Open": close, "High": close + 0.01, "Low": close - 0.01, "Close": close})


def _eng(monkeypatch, **kw):
    monkeypatch.setenv("AIT_BT_TOUCH_STOP", "0")          # reach the bell ITM (no stop on touch)
    base = dict(engine_model="realistic", entry_dte=0, expiry_guard_dte=0,
                credit_tp_ladder=W74._no_take_profit(), credit_loss_limit_mult=100.0)
    base.update(kw)
    return W74._bt(**base)


def _position(entry_price=0.30, **over):
    return W74._pos(entry="2024-03-04", expiry="2024-03-04", entry_price=entry_price,
                    entry_commission=3.0, **over)


HIST = W74._frame().iloc[:40]


class TestEngineBell:
    def test_otm_at_the_bell_settles_at_full_credit_with_no_exit_cost(self, monkeypatch):
        bt = _eng(monkeypatch)
        got = bt._check_exit(_position(), ROW(100.0), EXPIRY, HIST)
        assert got["exit_reason"] == "expiry_settled"
        assert got["exit_price"] == 0.0 and got["net_shares"] == 0
        assert got["settlement_legs"] == ["expired"] * 4
        assert got["pnl"] == pytest.approx(0.30 * 100 - 3.0)      # credit - ENTRY commission only
        assert "_settle" not in got

    def test_both_calls_itm_closes_now_at_the_width(self, monkeypatch):
        bt = _eng(monkeypatch)
        got = bt._check_exit(_position(), ROW(110.0), EXPIRY, HIST)
        assert got["exit_reason"] == "expiry_settled" and got["net_shares"] == 0
        assert got["exit_price"] == pytest.approx(3.0)
        assert got["pnl"] == pytest.approx((0.30 - 3.0) * 100 - 0.015 * 100 - 3.0)
        # the defined maximum is not exceeded by a net-zero settlement
        assert bt._settlement_stats["over_max_loss"] == 0

    def test_one_sided_is_left_pending_then_closed_at_the_next_open(self, monkeypatch):
        bt = _eng(monkeypatch)
        pos = _position(max_loss=270.0)
        assert bt._check_exit(pos, ROW(105.0), EXPIRY, HIST) is None
        sett = pos["settlement"]
        assert sett["underlying"] == pytest.approx(105.0) and sett["expiry"] == str(EXPIRY)
        got = bt._close_pending_settlement(pos, 112.0, date(2024, 3, 5))
        assert "settlement" not in pos
        assert got["exit_reason"] == "assigned_gap" and got["net_shares"] == -1
        assert got["exit_price"] == pytest.approx(8.0)                       # 112 - 104
        assert got["pnl"] == pytest.approx((0.30 - 8.0) * 100 - 0.025 * 100 - 3.0)
        assert bt._settlement_stats["over_max_loss"] == 1                    # beyond the defined max
        assert got["exit_time"].startswith("2024-03-05T09:30")

    def test_pending_is_never_settled_twice(self, monkeypatch):
        bt = _eng(monkeypatch)
        pos = _position()
        assert bt._check_exit(pos, ROW(105.0), EXPIRY, HIST) is None
        before = dict(pos["settlement"])
        assert bt._check_exit(pos, ROW(105.0), EXPIRY, HIST) is None            # daily after intraday
        sess = pd.DataFrame({"Open": [105.0], "High": [105.0], "Low": [105.0], "Close": [105.0]},
                            index=pd.DatetimeIndex([datetime(2024, 3, 4, 19, 55)], tz="UTC"))
        assert bt._check_intraday_exit(pos, sess, EXPIRY) is None
        assert pos["settlement"] == before

    def test_intraday_bar_ending_at_the_bell_settles_at_its_close(self, monkeypatch):
        bt = _eng(monkeypatch)
        # 15:55-16:00 ET on 2024-03-04 is 20:55Z (EST) -- early March is before DST
        idx = pd.DatetimeIndex([datetime(2024, 3, 4, 20, 55)], tz="UTC")
        sess = pd.DataFrame({"Open": [100.0], "High": [100.01], "Low": [99.99], "Close": [100.0]}, index=idx)
        got = bt._check_intraday_exit(_position(), sess, EXPIRY)
        assert got["exit_reason"] == "expiry_settled"
        assert got["exit_time"].startswith("2024-03-04T20:55")

    def test_put_side_one_sided_and_both_puts_itm(self, monkeypatch):
        bt = _eng(monkeypatch)
        pos = _position()
        assert bt._check_exit(pos, ROW(94.5), EXPIRY, HIST) is None            # short put 96 ITM, wing 93 OTM
        got = bt._close_pending_settlement(pos, 91.0, date(2024, 3, 5))
        assert got["net_shares"] == +1 and got["settlement_legs"][2] == "assigned"
        assert got["exit_price"] == pytest.approx(5.0)                         # long stock at 96, sold at 91
        both = bt._check_exit(_position(), ROW(90.0), EXPIRY, HIST)             # both puts ITM: width 3
        assert both["exit_reason"] == "expiry_settled" and both["net_shares"] == 0
        assert both["exit_price"] == pytest.approx(3.0)
        assert both["settlement_legs"] == ["expired", "expired", "assigned", "exercised"]

    def test_pin_is_pending_and_resolved_by_the_open(self, monkeypatch):
        bt = _eng(monkeypatch)
        pos = _position()
        assert bt._check_exit(pos, ROW(103.95), EXPIRY, HIST) is None            # 5c under the short call
        got = bt._close_pending_settlement(pos, 100.0, date(2024, 3, 5))
        assert got["settlement_legs"][0] == "expired"                           # worse than assigned-at-100
        assert got["net_shares"] == 0 and got["exit_price"] == 0.0

    def test_flatten_still_wins_before_the_bell(self, monkeypatch):
        bt = _eng(monkeypatch, flatten_minutes_before_close=60)
        got = bt._check_exit(_position(), ROW(100.0), EXPIRY, HIST)
        assert got["exit_reason"] == "expiry_day_flatten"

    def test_guard_applied_entries_never_reach_settlement(self, monkeypatch):
        bt = _eng(monkeypatch, expiry_guard_dte=None)        # guard applies to every entry
        got = bt._check_exit(_position(), ROW(100.0), EXPIRY, HIST)
        assert got["exit_reason"] in ("expiry", "expiry_approaching")
        assert "settlement_legs" not in got

    def test_unknown_legs_keep_the_remark_close(self, monkeypatch):
        bt = _eng(monkeypatch)
        bt._settle_legs_of = lambda pos: None
        got = bt._check_exit(_position(), ROW(100.0), EXPIRY, HIST)
        assert got["exit_reason"] == "expiry" and "settlement_legs" not in got

    def test_legacy_engine_never_settles(self, monkeypatch):
        monkeypatch.setenv("AIT_BT_TOUCH_STOP", "0")
        bt = W74._bt(entry_dte=0, expiry_guard_dte=0, credit_tp_ladder=W74._no_take_profit(),
                     credit_loss_limit_mult=100.0)            # engine_model defaults to legacy
        got = bt._check_exit(_position(), ROW(105.0), EXPIRY, HIST)
        assert got is not None and got["exit_reason"] == "expiry"
        assert "settlement_legs" not in got and "settlement" not in _position()

    def test_leg_map_rejects_missing_keys_and_unknown_strategies(self, monkeypatch):
        bt = _eng(monkeypatch)
        pos = _position()
        assert len(bt._settle_legs_of(pos)) == 4
        broken = {k: v for k, v in pos.items() if k != "long_put_strike"}
        assert bt._settle_legs_of(broken) is None
        assert bt._settle_legs_of({**pos, "strategy": "long_call"}) is None


class TestRunLevel:
    """A full realistic run on the W74 'close above the open' market, ramp chosen so the
    short call finishes ITM and its protective wing OTM (the operator's pin episodes)."""

    @staticmethod
    def _market(ramp=1.6):
        idx = pd.bdate_range("2024-04-01", periods=100)
        base = 100 * np.exp(np.cumsum(np.random.default_rng(11).normal(0, 0.007, len(idx))))
        iv = 0.15 + 0.002 * (np.arange(100) % 17)
        return pd.DataFrame({"Open": base, "Close": base + ramp, "High": base + ramp + 0.4,
                             "Low": base - 0.4, "Volume": 1e6, "implied_vol": iv}, index=idx)

    def _run(self, monkeypatch, df, **kw):
        monkeypatch.setenv("AIT_BT_TOUCH_STOP", "0")
        bt = W74._bt(df, intraday_store=W74._Store(df), symbol="QQQ", condor_any_direction=True,
                     entry_window_start_et="10:30", entry_window_end_et="15:30",
                     scan_interval_minutes=60, engine_model="realistic", entry_dte=0,
                     expiry_guard_dte=0, credit_tp_ladder=W74._no_take_profit(),
                     credit_loss_limit_mult=100.0, max_contracts_per_trade=1, **kw)
        return bt, bt.run()

    def test_one_sided_expiry_is_closed_at_the_next_sessions_open(self, monkeypatch):
        df = self._market()
        bt, res = self._run(monkeypatch, df)
        gaps = [t for t in res.trades if t["exit_reason"] == "assigned_gap"]
        assert gaps, "scenario must produce at least one carried-shares exit"
        for t in gaps:
            exp = pd.Timestamp(t["expiry_date"])
            nxt = df.index[df.index > exp][0]
            assert t["exit_date"] == str(nxt.date())
            assert t["exit_price"] == pytest.approx(
                round(float(df.loc[nxt, "Open"]) - t["short_call_strike"], 4), abs=1e-4)
            assert t["net_shares"] == -1 and t["settlement_legs"][0] == "assigned"
            assert t["pnl"] == pytest.approx(
                (t["entry_price"] - t["exit_price"]) * 100 - 0.025 * 100 - t["entry_commission"], abs=0.02)
            assert "settlement" not in t and "_settle" not in t
        assert bt._settlement_stats["assigned_gap"] == len(gaps)

    def test_net_zero_runs_settle_the_same_session(self, monkeypatch):
        bt, res = self._run(monkeypatch, self._market(ramp=3.0))
        settled = [t for t in res.trades if t["exit_reason"] == "expiry_settled"]
        assert settled and all(t["exit_date"] == t["expiry_date"] for t in settled)
        assert not any(t["exit_reason"] == "assigned_gap" for t in res.trades)

    def test_no_pending_settlement_survives_the_run(self, monkeypatch):
        _, res = self._run(monkeypatch, self._market())
        assert all("settlement" not in t and "_settle" not in t for t in res.trades)

    def test_pending_at_the_end_of_the_data_is_marked_at_the_last_close(self, monkeypatch):
        monkeypatch.setenv("AIT_BT_TOUCH_STOP", "0")
        bt = _eng(monkeypatch)
        pos = _position()
        assert bt._check_exit(pos, ROW(105.0), EXPIRY, HIST) is None
        last = pd.Series({"Open": 104.0, "High": 109.0, "Low": 103.0, "Close": 108.0})
        got = bt._force_close(pos, last, EXPIRY, HIST)
        assert got["exit_reason"] == "backtest_end" and got["net_shares"] == -1
        assert got["exit_price"] == pytest.approx(4.0)


class TestKnobsAndPartialEngines:
    def test_defaults_come_from_config_and_kwargs_win(self):
        bt = W74._bt()
        cfg = BacktestConfig()
        assert (bt._pin_band, bt._assignment_fee, bt._stock_commission) == (
            cfg.pin_band_usd, cfg.assignment_fee_per_share, cfg.stock_commission_per_share)
        bt2 = W74._bt(pin_band_usd=0.25, assignment_fee_per_share=0.0, stock_commission_per_share=0.0)
        assert (bt2._pin_band, bt2._assignment_fee, bt2._stock_commission) == (0.25, 0.0, 0.0)

    def test_documented_defaults(self):
        """Measured on the 2026-09/10 statement (see the field descriptions); moving one is a decision."""
        cfg = BacktestConfig()
        assert (cfg.pin_band_usd, cfg.assignment_fee_per_share, cfg.stock_commission_per_share) == (
            0.10, 0.015, 0.01)

    def test_config_bounds(self):
        for field, bad in (("pin_band_usd", -0.01), ("assignment_fee_per_share", -1.0),
                           ("stock_commission_per_share", 5.0)):
            with pytest.raises(ValidationError):
                BacktestConfig(**{field: bad})

    def test_partial_engines_stay_legacy(self):
        bare = Backtester.__new__(Backtester)
        assert bare._realistic is False and bare._settlement_stats is None
        bare._bump_settle("settled")          # creates the dict on first use, never raises
        assert bare._settlement_stats["settled"] == 1


class TestClassGuards:
    """The next hand-written exit path that forgets the bell fails here, not in a study."""

    @staticmethod
    def _fn(name):
        import ait.backtesting.engine as eng
        tree = ast.parse(inspect.getsource(eng))
        return next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)

    def test_the_bell_rule_marks_realistic_exits_for_settlement(self):
        fn = self._fn("_check_exit_credit")
        bell_ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "_asof >= _bell"]
        assert len(bell_ifs) == 1
        marks = [n for n in ast.walk(bell_ifs[0]) if isinstance(n, ast.Constant) and n.value == "_settle"]
        assert marks, "the expiry-bell rule must set _settle for the realistic engine"

    @pytest.mark.parametrize("name", ["_check_exit", "_check_intraday_exit_realistic"])
    def test_every_consumer_of_the_credit_dispatch_handles_the_marker(self, name):
        fn = self._fn(name)
        pops = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
                and ast.unparse(n.func).endswith(".pop")
                and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "_settle"]
        assert pops, f"{name} must pop and route the _settle marker"

    @pytest.mark.parametrize("name", ["_check_exit", "_check_intraday_exit_realistic"])
    def test_exit_checks_skip_a_position_already_holding_shares(self, name):
        src = ast.unparse(self._fn(name))
        assert "pos.get('settlement') is not None" in src or 'pos.get("settlement") is not None' in src

    def test_settlement_module_is_pure(self):
        import ait.backtesting.settlement as mod
        tree = ast.parse(inspect.getsource(mod))
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree)
                     if isinstance(n, ast.ImportFrom) and n.module}
        assert imported <= {"__future__", "dataclasses", "itertools"}


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

class _Data:
    """Minimal stand-in for ResearchData: a settle bar, daily rows, nothing on disk."""

    def __init__(self, settle_close, next_open, expiry=date(2026, 9, 25)):
        self.expiry, self._c, self._o = expiry, settle_close, next_open
        nxt = expiry + timedelta(days=3)                       # Friday -> Monday
        self._frame = pd.DataFrame(
            {"Open": [(self._c or 0.0) - 1.0, next_open or 0.0], "High": [0, 0], "Low": [0, 0],
             "Close": [self._c or 0.0, next_open or 0.0]}, index=pd.DatetimeIndex([expiry, nxt]))

    def bar_containing(self, ts):
        return None if self._c is None else {"close": self._c}

    def session_row(self, d):
        ts = pd.Timestamp(d)
        return self._frame.loc[ts] if ts in self._frame.index else None

    def daily_frame(self):
        return self._frame


def _trade(legs, stock=(), pnl=None, credit_usd=272.0, comm=-3.15, expiry=date(2026, 9, 25)):
    mk = lambda side, right, k, code: rp.ReplayLeg(
        side=side, right=right, strike=k, expiry=expiry, open_price=1.0,
        close_time=datetime(2026, 9, 25, 16, 20, tzinfo=ET) if code else None,
        close_price=0.0, close_code=code, realized_usd=0.0)
    return rp.ReplayTrade(
        id="T", origin="manual", bot_trade_id=None,
        entry_time=datetime(2026, 9, 18, 14, 20, tzinfo=ET), quantity=1, structure="iron_condor",
        legs=tuple(mk(*l) for l in legs), still_open=False, net_credit_usd=credit_usd,
        commissions_usd=comm, total_pnl_usd=pnl, linked_stock=tuple(stock))


KW = dict(pin_band=0.10, assignment_fee=0.015, stock_commission=0.01)


class TestReplay:
    def test_both_calls_itm_reproduces_statuses_net_zero_and_pnl(self):
        legs = [("short", "C", 723.0, "A;C"), ("long", "C", 731.0, "C;Ex"),
                ("long", "P", 694.0, "C;Ep"), ("short", "P", 702.0, "C;Ep")]
        stock = [rp.ReplayStock(datetime(2026, 9, 25, 16, 20, tzinfo=ET), 100, 731.0, 0.0, "Ex;O"),
                 rp.ReplayStock(datetime(2026, 9, 25, 16, 20, tzinfo=ET), -100, 723.0, -1.5, "A;C")]
        r = rp.settlement_replay(_trade(legs, stock, pnl=-532.65), _Data(744.0, 740.0), **KW)
        assert r["status"] == "ok" and not r["overnight"]
        assert r["model_statuses"] == ["assigned", "exercised", "expired", "expired"]
        assert (r["legs_scored"], r["legs_match"]) == (4, 4)
        assert r["net_shares_model"] == r["net_shares_actual"] == 0
        # credit 2.72 - width 8.00, minus 1.5 assignment fee, plus the statement's commissions
        assert r["pnl_model"] == pytest.approx((2.72 - 8.0) * 100 - 1.5 - 3.15)
        assert r["pnl_error"] == pytest.approx(r["pnl_model"] - (-532.65))

    def test_one_sided_uses_the_next_open_and_reports_the_late_close(self):
        legs = [("short", "C", 749.0, "A;C"), ("long", "C", 754.0, "C;Ep"),
                ("long", "P", 721.0, "C;Ep"), ("short", "P", 726.0, "C;Ep")]
        stock = [rp.ReplayStock(datetime(2026, 9, 25, 16, 20, tzinfo=ET), -100, 749.0, -1.54, "A;O"),
                 rp.ReplayStock(datetime(2026, 9, 28, 14, 38, tzinfo=ET), 100, 755.47, -1.0, "C;P")]
        r = rp.settlement_replay(_trade(legs, stock, pnl=-482.68, credit_usd=170.0),
                                 _Data(749.55, 749.35), **KW)
        assert r["overnight"] and r["net_shares_model"] == -1 and r["net_shares_actual"] == -1
        assert r["O"] == pytest.approx(749.35)
        assert r["close_after_open_min"] == pytest.approx(308.0)
        assert r["pnl_model"] == pytest.approx((1.70 - 0.35) * 100 - 2.5 - 3.15)

    def test_trader_closed_and_open_structures_are_not_replayed(self):
        closed = _trade([("short", "C", 100.0, "C"), ("long", "C", 105.0, "C")])
        assert rp.settlement_replay(closed, _Data(100.0, 100.0), **KW) is None

    def test_missing_prices_are_reported_not_invented(self):
        legs = [("short", "C", 100.0, "A;C"), ("long", "C", 105.0, "C;Ep")]
        d = _Data(None, None)
        d._frame = d._frame.iloc[:0]
        d.session_row = lambda _d: None
        r = rp.settlement_replay(_trade(legs), d, **KW)
        assert r["status"] == "no_data"

    def test_loader_keeps_linked_stock(self, tmp_path):
        doc = {"schema": rp.SCHEMA, "structures": [{
            "id": "X", "underlying": "QQQ", "origin": "manual", "bot_trade_id": None,
            "entry_time": "2026-09-18 14:20:00", "entry_tz": "America/New_York", "quantity": 1,
            "still_open": False, "structure": "iron_condor", "net_credit_usd": 100.0,
            "commissions_usd": -3.0, "total_pnl_usd": 10.0,
            "legs": [{"side": "short", "right": "C", "strike": 100.0, "expiry": "2026-09-25",
                      "open_price": 1.0, "close_time": None, "close_price": None,
                      "close_code": "A;C", "realized_usd": 0.0}],
            "linked_stock": [{"time": "2026-09-25 16:20:00", "shares": -100, "price": 100.0,
                              "commission": -1.5, "code": "A;O"}]}]}
        p = Path(tmp_path) / "t.json"
        p.write_text(json.dumps(doc))
        (t,) = rp.load_trades(p)
        assert t.linked_stock[0].shares == -100 and t.linked_stock[0].code == "A;O"


class TestReasonMap:
    def test_longest_prefix_wins(self):
        assert rp.research_reasons_for("take_profit_short (P&L 40%)") == ("take_profit_short",)
        assert rp.research_reasons_for("take_profit (P&L 51%)") == ("profit_target",)
        assert rp.research_reasons_for("partial_take_profit 1/2") == ()

    def test_longest_prefix_wins_whatever_the_table_order(self, monkeypatch):
        monkeypatch.setattr(rp, "LIVE_REASON_TO_RESEARCH",
                            (("take_profit", ("short_one",)), ("take_profit_short", ("long_one",))))
        assert rp.research_reasons_for("take_profit_short x") == ("long_one",)

    def test_known_and_unknown(self):
        assert "touch_stop" in rp.research_reasons_for("short_strike_touch call")
        assert "assigned_gap" in rp.research_reasons_for("reconciler_expired_intrinsic")
        assert rp.research_reasons_for("something_new") is None
        assert rp.research_reasons_for(None) is None

    def test_every_research_twin_is_a_reason_the_engine_can_emit(self):
        import ait.backtesting.engine as eng
        src = inspect.getsource(eng)
        for _prefix, twins in rp.LIVE_REASON_TO_RESEARCH:
            for reason in twins:
                assert f'"{reason}"' in src, f"{reason} is not an engine exit reason"

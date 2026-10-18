"""W74: the research engine at live fidelity, A/B against the legacy model.

Every W72 lead (L2-L6) is a test that BOTH proves the legacy defect still exists
under ``engine_model="legacy"`` (so old studies stay reproducible) AND that
``"realistic"`` fixes it. Plus the new exit knobs and the pure helpers they use.

House rules: real code paths; synthetic fixtures only.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import Backtester
from ait.backtesting.expiry_calendar import actual_dte, listed_expiry
from ait.backtesting.pricing import expected_move
from ait.execution import exit_policy as ep
from ait.strategies.base import SignalDirection
from ait.utils.time import ET, next_session_on_or_after, time_to_expiry_years


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def _frame(n=160, seed=42, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.008, len(idx))))
    return pd.DataFrame({"Open": c, "High": c * 1.004, "Low": c * 0.996, "Close": c, "Volume": 1e6}, index=idx)


def _bt(df=None, **kw):
    base = dict(data=_frame() if df is None else df, strategies=["iron_condor"], initial_capital=100_000,
                macro_event_gate=False, allow_live_model_fallback=False, hurst_hard_veto_multiplier=0.0,
                hurst_regime_threshold=0.0, iv_rank_rise_threshold=10.0, ic_min_credit=0.0,
                ic_min_credit_width=0.0, iv_floor=0.01, features_cache=None, friction_model="legacy")
    base.update(kw)
    return Backtester(**base)


def _pos(entry="2024-03-01", expiry="2024-03-31", entry_price=1.0, **over):
    p = {"strategy": "iron_condor", "trade_type": "credit", "entry_date": entry, "entry_price": entry_price,
         "short_call_strike": 104.0, "short_put_strike": 96.0, "long_call_strike": 107.0, "long_put_strike": 93.0,
         "entry_iv": 0.20, "expiry_date": expiry, "contracts": 1, "n_legs": 4, "high_water_mark": 0.0,
         "direction": SignalDirection.NEUTRAL.value}
    p.update(over)
    return p


class _Store:
    """Each session sits at that day's OPEN until 15:00 ET, then ramps to the daily Close."""

    def __init__(self, df):
        self.df = df

    def load_intraday_range(self, symbol, start_date, end_date):
        ts = pd.Timestamp(start_date)
        if ts not in self.df.index:
            return pd.DataFrame()
        o, c = float(self.df.loc[ts, "Open"]), float(self.df.loc[ts, "Close"])
        d = start_date
        start = datetime(d.year, d.month, d.day, 13, 30, tzinfo=timezone.utc)   # EDT sessions only
        n = 78
        idx = [start + timedelta(minutes=5 * k) for k in range(n)]
        px = np.array([o] * 60 + list(np.linspace(o, c, n - 60)))
        return pd.DataFrame({"Open": px, "High": px + 0.02, "Low": px - 0.02, "Close": px, "Volume": 1e5},
                            index=pd.DatetimeIndex(idx))


def _gap_market(n=100, seed=11):
    """Spring/summer sessions (EDT); close sits $4 above the open every day."""
    global _MARKET
    if _MARKET is not None and (n, seed) == (100, 11):
        return _MARKET
    idx = pd.bdate_range("2024-04-01", periods=n)
    base = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.007, len(idx))))
    iv = 0.15 + 0.002 * (np.arange(n) % 17)           # varies day to day (the leak is visible)
    df = pd.DataFrame({"Open": base, "Close": base + 4.0, "High": base + 4.4, "Low": base - 0.4,
                       "Volume": 1e6, "implied_vol": iv}, index=idx)
    if (n, seed) == (100, 11):
        _MARKET = df
    return df


_RUNS: dict = {}
_MARKET = None
_CALM = None


def _calm_market(n=100, seed=5):
    """No ramp: the close equals the open and the day barely moves, so a 0-DTE
    condor is neither touched nor profitable-by-noise before the flatten."""
    global _CALM
    if _CALM is None:
        idx = pd.bdate_range("2024-04-01", periods=n)
        base = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.003, len(idx))))
        _CALM = pd.DataFrame({"Open": base, "Close": base, "High": base * 1.0005,
                              "Low": base * 0.9995, "Volume": 1e6,
                              "implied_vol": 0.17 + 0.001 * (np.arange(n) % 5)}, index=idx)
    return _CALM


def _no_take_profit():
    return ep.validate_ladder([(ep.TERMINAL_DTE_FLOOR, ep.NO_TAKE_PROFIT, ep.NO_TAKE_PROFIT)])


def _run(df, **kw):
    """Full run, memoised (deterministic; several tests share a configuration)."""
    memo = df is _MARKET or df is _CALM
    key = (id(df), tuple(sorted((k, repr(v)) for k, v in kw.items())))
    if memo and key in _RUNS:
        return _RUNS[key]
    res = _bt(df, intraday_store=_Store(df), symbol="QQQ", condor_any_direction=True,
              entry_window_start_et="10:30", entry_window_end_et="15:30", scan_interval_minutes=60,
              **kw).run()
    if memo:
        _RUNS[key] = res
    return res


# --------------------------------------------------------------------------
# the switch itself
# --------------------------------------------------------------------------

class TestSwitch:
    def test_bare_engine_is_legacy_and_exit_knobs_are_off(self):
        bt = _bt()
        assert bt._engine_model == "legacy" and bt._realistic is False
        assert bt._expiry_guard_dte is None and bt._max_hold_fraction is None and bt._flatten_minutes is None

    def test_unknown_model_and_bad_fraction_are_refused(self):
        with pytest.raises(ValueError):
            _bt(engine_model="nope")
        with pytest.raises(ValueError):
            _bt(max_hold_fraction=1.5)

    def test_none_resolves_from_the_loaded_settings(self):
        from ait.config.settings import BacktestConfig
        stub = SimpleNamespace(backtest=BacktestConfig(
            engine_model="realistic", expiry_guard_dte=3, max_hold_fraction=0.5,
            flatten_minutes_before_close=45))
        bt = _bt(settings=stub)
        assert bt._realistic and bt._expiry_guard_dte == 3
        assert bt._max_hold_fraction == 0.5 and bt._flatten_minutes == 45
        # an explicit argument still wins over the config
        assert _bt(settings=stub, engine_model="legacy")._realistic is False


class TestPartialEnginesStayLegacy:
    """Harnesses build engines with ``Backtester.__new__`` and set only what they need
    (the R15/R16 touch-stop tests do). Those instances must run LEGACY semantics, so
    the class-level defaults are a contract, pinned against what __init__ resolves."""

    ATTRS = ("_engine_model", "_realistic", "_expiry_guard_dte", "_max_hold_fraction", "_flatten_minutes")

    def test_a_partially_built_engine_is_legacy(self):
        bare = Backtester.__new__(Backtester)
        assert (bare._engine_model, bare._realistic) == ("legacy", False)
        assert bare._expiry_guard_dte is None
        assert bare._max_hold_fraction is None and bare._flatten_minutes is None

    def test_class_defaults_equal_what_init_resolves_for_a_default_engine(self):
        built = _bt()
        for a in self.ATTRS:
            assert getattr(built, a) == getattr(Backtester, a), a

    def test_a_partial_engine_runs_the_credit_rules_without_the_new_attributes_set(self):
        bare = Backtester.__new__(Backtester)
        for k, v in dict(_touch_stop_enabled=False, _credit_loss_limit_mult=0.0, _max_hold_days=30,
                         _exit_time_decay_scaling=True, _credit_tp_ladder=None, _economic_cal=None,
                         _entry_dte=14).items():
            setattr(bare, k, v)
        got = bare._check_exit_credit(_pos(entry="2024-03-01", expiry="2024-03-04", entry_price=0.05),
                                      pnl_pct=-1.0, current_date=date(2024, 3, 2))
        assert got is not None and got["exit_reason"] == "expiry_approaching"


# --------------------------------------------------------------------------
# L2 / listed expiries / 0-DTE: the entry
# --------------------------------------------------------------------------

class TestEntryAtTheFill:
    def test_legacy_prices_a_morning_fill_at_the_daily_close(self):
        df = _gap_market()
        t = _run(df, engine_model="legacy", entry_dte=14).trades[0]
        day = pd.Timestamp(t["entry_date"])
        assert t["underlying_at_entry"] == pytest.approx(round(float(df.loc[day, "Close"]), 2), abs=0.01)

    def test_realistic_prices_the_fill_at_the_limit_price_not_the_close(self):
        df = _gap_market()
        t = _run(df, engine_model="realistic", entry_dte=14).trades[0]
        day = pd.Timestamp(t["entry_date"])
        assert t["underlying_at_entry"] == pytest.approx(float(df.loc[day, "Open"]), abs=0.5)
        assert t["underlying_at_entry"] != pytest.approx(float(df.loc[day, "Close"]), abs=1.0)
        assert t["engine_model"] == "realistic"

    def test_realistic_uses_the_previous_sessions_iv(self):
        df = _gap_market()
        leg = _run(df, engine_model="legacy", entry_dte=14).trades[0]
        real = _run(df, engine_model="realistic", entry_dte=14).trades[0]
        d_l = pd.Timestamp(leg["entry_date"]); d_r = pd.Timestamp(real["entry_date"])
        assert leg["entry_iv"] == pytest.approx(round(float(df.loc[d_l, "implied_vol"]), 4), abs=1e-4)
        prev = df.index[df.index.get_loc(d_r) - 1]
        assert real["entry_iv"] == pytest.approx(round(float(df.loc[prev, "implied_vol"]), 4), abs=1e-4)

    def test_expiry_is_a_listed_session_not_a_weekend(self):
        # a Friday entry with 1 target day lands on a Saturday under the legacy rule
        df = _gap_market()
        fridays = [d for d in df.index if d.weekday() == 4]
        assert fridays
        leg = _bt(df)._build_position  # noqa: F841 (constructor smoke)
        res_leg = _run(df, engine_model="legacy", entry_dte=1)
        res_real = _run(df, engine_model="realistic", entry_dte=1)
        for t in res_leg.trades:
            exp = date.fromisoformat(t["expiry_date"])
            if date.fromisoformat(t["entry_date"]).weekday() == 4:
                assert exp.weekday() == 5          # legacy: Saturday
                break
        else:
            pytest.skip("no Friday entry in the sample")
        for t in res_real.trades:
            exp = date.fromisoformat(t["expiry_date"])
            assert exp.weekday() < 5               # realistic: always a session
            assert t["entry_dte_actual"] == (exp - date.fromisoformat(t["entry_date"])).days

    def test_zero_dte_is_buildable_only_in_realistic_mode(self):
        df = _gap_market()
        leg = _run(df, engine_model="legacy", entry_dte=0)
        assert leg.trades == [] and leg.entry_funnel.get("builder:em_unavailable", 0) > 0
        real = _run(df, engine_model="realistic", entry_dte=0)
        assert real.trades, f"0-DTE condor never built: {real.entry_funnel}"
        t = real.trades[0]
        assert t["entry_dte_actual"] == 0 and t["expiry_date"] == t["entry_date"]


# --------------------------------------------------------------------------
# L6: strikes are priced as stored
# --------------------------------------------------------------------------

class TestPerPositionExpiry:
    def test_remark_uses_the_positions_own_dates_not_the_engine_wide_dte(self):
        from ait.backtesting.pricing import ic_leg_marks
        bt = _bt(entry_dte=14)                                    # engine-wide default 14
        pos = _pos(entry="2024-03-01", expiry="2024-03-06", entry_iv=0.20)   # THIS one is 5 DTE
        got = bt._reprice_position(pos, 100.0, 0, None)
        sc, sp, lc, lp = ic_leg_marks(100.0, 104.0, 96.0, 107.0, 93.0, 5 / 365.0, 0.05, 0.20,
                                      bt._skew_factor, bt._skew_slope_per_pct_otm, bt._iv_floor)
        assert got == pytest.approx((sc + sp) - (lc + lp), abs=1e-12)
        wrong = ic_leg_marks(100.0, 104.0, 96.0, 107.0, 93.0, 14 / 365.0, 0.05, 0.20,
                             bt._skew_factor, bt._skew_slope_per_pct_otm, bt._iv_floor)
        assert got != pytest.approx((wrong[0] + wrong[1]) - (wrong[2] + wrong[3]), abs=1e-6)

    def test_a_position_without_dates_falls_back_to_the_engine_dte(self):
        bt = _bt(entry_dte=21)
        assert bt._position_dte({"strategy": "long_call"}) == 21


class TestRemarkClock:
    """The defect found by sanity-checking the reference study: realistic ENTRY used
    fractional time but the RE-MARK still used whole days floored at ~53 min, so a
    0-DTE condor priced with 5.5 h at its fill was marked with 0.9 h one bar later
    and looked instantly profitable (a 97.5% win rate that was pure artifact)."""

    def _fresh_zero_dte(self, model):
        bt = _bt(entry_dte=0, engine_model=model)
        fill = datetime(2024, 4, 3, 10, 30, tzinfo=ET)
        t = time_to_expiry_years(fill, date(2024, 4, 3))
        # QQQ-scale spot: at $100 a whole-dollar strike on a 5 h option rounds to
        # at-the-money and the EM band (correctly) refuses it.
        pos = bt._build_credit_position("iron_condor", 700.0, 0.20, t, 0.05, 0,
                                        date(2024, 4, 3), 1_000_000.0, expiry=date(2024, 4, 3))
        assert pos is not None, "probe condor not built"
        return bt, pos, fill

    def test_realistic_remark_one_bar_later_is_continuous_with_the_entry(self):
        bt, pos, fill = self._fresh_zero_dte("realistic")
        later = fill + timedelta(minutes=5)
        v = bt._reprice_position(pos, 700.0, 0, None, as_of=later)
        # five minutes of theta on a 5.5 h option: a sliver, never a collapse
        assert v == pytest.approx(pos["mid_credit"], rel=0.06)
        assert v <= pos["mid_credit"] + 1e-6              # decay only ever helps the short

    def test_the_whole_day_clock_collapses_the_value(self):
        """Documents what the whole-day clock does to the same position (no decision
        instant given): the floor t=0.0001 y (0.9 h) makes it look far cheaper at
        once. (A legacy engine cannot even BUILD a 0-DTE condor: EM=0.)"""
        bt, pos, fill = self._fresh_zero_dte("realistic")
        collapsed = bt._reprice_position(pos, 700.0, 0, None, as_of=None)
        assert collapsed < 0.6 * pos["mid_credit"]

    def test_the_remark_reaches_intrinsic_at_the_bell(self):
        bt, pos, fill = self._fresh_zero_dte("realistic")
        at_bell = datetime(2024, 4, 3, 16, 0, tzinfo=ET)
        v = bt._reprice_position(pos, 700.0, 0, None, as_of=at_bell)
        assert v == pytest.approx(0.0, abs=1e-9)          # spot inside both short strikes

    def test_exit_friction_is_charged_on_the_legs_marked_at_the_same_instant(self):
        """The fitted per-leg friction scales with each leg's mid, so it must be
        evaluated on the legs re-marked at the DECISION instant, not on whole days."""
        from ait.backtesting.friction import LegSpreadModel
        bt, pos, fill = self._fresh_zero_dte("realistic")
        bt._leg_spread_model = LegSpreadModel("pct", b=0.20)
        early = bt._apply_exit_friction(pos, 0.0, 700.0, 0, None, 0.0, "credit", as_of=fill)
        late = bt._apply_exit_friction(pos, 0.0, 700.0, 0, None, 0.0, "credit",
                                       as_of=datetime(2024, 4, 3, 15, 55, tzinfo=ET))
        assert early > late > 0.0            # more life left => richer mids => wider $ half-spread

    def test_daily_path_marks_at_the_session_close(self):
        bt = _bt(entry_dte=5, engine_model="realistic")
        pos = _pos(entry="2024-04-01", expiry="2024-04-08", entry_price=0.5)
        close = datetime(2024, 4, 3, 16, 0, tzinfo=ET)
        t = bt._remark_t(pos, 2, close)
        assert t == pytest.approx(5 / 365.0)               # Apr 3 close -> Apr 8 close


class TestRoundedStrikes:
    def _fresh(self, **kw):
        bt = _bt(entry_dte=21, wing_k=1.37, **kw)
        pos = bt._build_credit_position("iron_condor", 100.0, 0.22, 21 / 365.0, 0.05, 21,
                                        date(2024, 3, 1), 100_000.0)
        return bt, pos

    def test_legacy_remarks_away_from_the_entry_credit_on_day_zero(self):
        bt, pos = self._fresh(engine_model="legacy")
        assert abs(bt._reprice_position(pos, 100.0, 0, None) - pos["mid_credit"]) > 5e-3

    def test_realistic_day_zero_remark_reproduces_the_entry_credit(self):
        bt, pos = self._fresh(engine_model="realistic")
        # mid_credit is stored to 4 dp; the legacy defect is ~1e-2, so 1e-4 separates them cleanly
        assert bt._reprice_position(pos, 100.0, 0, None) == pytest.approx(pos["mid_credit"], abs=1e-4)
        for k in ("long_call_strike", "long_put_strike", "short_call_strike", "short_put_strike"):
            assert float(pos[k]) == round(float(pos[k]))        # whole dollars, as stored


# --------------------------------------------------------------------------
# L3 / L4 / L5: the exits
# --------------------------------------------------------------------------

def _session(prices, day=date(2024, 3, 4)):
    start = datetime(day.year, day.month, day.day, 14, 30, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex([start + timedelta(minutes=5 * k) for k in range(len(prices))])
    return pd.DataFrame({"Open": prices, "High": [p + 0.05 for p in prices],
                         "Low": [p - 0.05 for p in prices], "Close": prices, "Volume": 1}, index=idx)


class TestExitDecisions:
    def test_l3_take_profit_decision_is_mid_based_in_both_paths_when_realistic(self):
        results = {}
        for model in ("legacy", "realistic"):
            bt = _bt(spread_base=0.20, spread_cap=0.40, spread_iv_sensitivity=0.0,
                     spread_dte_sensitivity=0.0, entry_dte=30, engine_model=model)
            pos = _pos()
            d = date(2024, 3, 4)
            value = bt._reprice_position(pos, 100.0, 3, None)
            pos["entry_price"] = value / (1 - 0.52)            # mid pnl_pct 0.52 >= 0.50 target
            sess = _session([100.0])
            row = pd.Series({"Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0})
            daily = bt._check_exit(dict(pos), row, d, _frame().iloc[:40])
            intra = bt._check_intraday_exit(dict(pos), sess, d)
            results[model] = (daily, intra)
        assert results["legacy"][0]["exit_reason"] == "take_profit_short"
        assert results["legacy"][1] is None                         # the defect, preserved
        assert results["realistic"][0]["exit_reason"] == "take_profit_short"
        assert results["realistic"][1]["exit_reason"] == "take_profit_short"
        # the spread is still PAID: booked value above the mid mark
        bt = _bt(spread_base=0.20, spread_cap=0.40, entry_dte=30, engine_model="realistic")
        pos = _pos(); mid = bt._reprice_position(pos, 100.0, 3, None)
        pos["entry_price"] = mid / (1 - 0.52)
        got = bt._check_intraday_exit(dict(pos), _session([100.0]), date(2024, 3, 4))
        assert got["exit_price"] > mid

    def _touch_then_take_profit_day(self, model):
        bt = _bt(entry_dte=30, engine_model=model)
        pos = _pos()
        v = bt._reprice_position(pos, 100.0, 3, None)
        pos["entry_price"] = v / (1 - 0.55)
        px = [103.4] * 20 + [100.0] * 60
        px[10] = 104.5                                              # pierces the 104 short call
        return bt, pos, _session(px)

    def test_l4_legacy_scan_cannot_see_a_touch_so_a_later_take_profit_wins(self):
        bt, pos, sess = self._touch_then_take_profit_day("legacy")
        got = bt._check_intraday_exit(dict(pos), sess, date(2024, 3, 4))
        assert got["exit_reason"] == "take_profit_short"

    def test_l4_realistic_scan_stops_out_at_the_touch_bar(self):
        bt, pos, sess = self._touch_then_take_profit_day("realistic")
        got = bt._check_intraday_exit(dict(pos), sess, date(2024, 3, 4))
        assert got["exit_reason"] == "touch_stop"
        assert got["exit_time"].startswith("2024-03-04T14:30:00") is False   # not the first bar...
        assert got["exit_time"][11:16] == "15:20"                           # ...bar 10 = 14:30Z + 50m
        # priced AT the touched strike (and never booked as the later profit)
        assert got["exit_underlying"] >= 104.0
        # ...with the structure re-marked at the touch bar's END (fractional time)
        # and the spread paid on top
        bar_end = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc).astimezone(ET) + timedelta(minutes=55)
        mid = bt._reprice_position(pos, got["exit_underlying"], 3, None, bar_end)
        hs = bt._options_half_spread(0.20, (date(2024, 3, 31) - date(2024, 3, 4)).days)
        want = bt._apply_exit_friction(pos, mid, got["exit_underlying"], 3, None, hs, "credit", bar_end)
        assert got["exit_price"] == pytest.approx(round(want, 4), abs=1e-4)

    def test_an_intrabar_pierce_is_booked_at_the_strike_remarked_at_the_bar_end(self):
        """No gap: the bar OPENS inside the short strike and its High reaches it, so
        the exit is priced AT the 104 strike (live polls spot; it transacts there),
        with the structure re-marked at the bar's end and the spread paid."""
        bt = _bt(entry_dte=30, engine_model="realistic")
        pos = _pos()
        v = bt._reprice_position(pos, 100.0, 3, None)
        pos["entry_price"] = v / (1 - 0.30)
        start = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)
        idx = pd.DatetimeIndex([start])
        bar = pd.DataFrame({"Open": [103.0], "High": [104.2], "Low": [102.9], "Close": [103.0],
                            "Volume": [1]}, index=idx)
        got = bt._check_intraday_exit(dict(pos), bar, date(2024, 3, 4))
        assert got["exit_reason"] == "touch_stop"
        assert got["exit_underlying"] == pytest.approx(104.0)        # the strike, not the close
        bar_end = (start + timedelta(minutes=5)).astimezone(ET)
        mid = bt._reprice_position(pos, 104.0, 3, None, bar_end)
        hs = bt._options_half_spread(0.20, (date(2024, 3, 31) - date(2024, 3, 4)).days)
        want = bt._apply_exit_friction(pos, mid, 104.0, 3, None, hs, "credit", bar_end)
        assert got["exit_price"] == pytest.approx(round(want, 4), abs=1e-4)
        # and it is NOT the whole-day mark
        whole_day = bt._apply_exit_friction(pos, bt._reprice_position(pos, 104.0, 3, None), 104.0, 3,
                                            None, hs, "credit")
        assert got["exit_price"] != pytest.approx(round(whole_day, 4), abs=1e-6)

    def test_a_bar_that_opens_beyond_the_short_strike_is_booked_at_the_gapped_open(self):
        """W5 gap rule, now reached by the intraday scan: a bar that OPENS through
        the short call could never be closed AT the strike, so the exit is priced
        at the worse of the strike and the gapped open (higher buy-back cost)."""
        bt = _bt(entry_dte=30, engine_model="realistic")
        pos = _pos()
        v = bt._reprice_position(pos, 100.0, 3, None)
        pos["entry_price"] = v / (1 - 0.30)          # not near the take-profit
        start = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)
        idx = pd.DatetimeIndex([start])
        gap = pd.DataFrame({"Open": [105.5], "High": [106.0], "Low": [105.0], "Close": [105.5],
                            "Volume": [1]}, index=idx)
        got = bt._check_intraday_exit(dict(pos), gap, date(2024, 3, 4))
        assert got["exit_reason"] == "touch_stop"
        assert got["exit_underlying"] == pytest.approx(105.5)       # the gapped open, not 104
        bar_end = datetime(2024, 3, 4, 14, 35, tzinfo=timezone.utc).astimezone(ET)
        at_strike = bt._reprice_position(pos, 104.0, 3, None, bar_end)
        at_gap = bt._reprice_position(pos, 105.5, 3, None, bar_end)
        assert at_gap > at_strike                                  # and it really is the costlier one
        # booked value = the gapped mark at the BAR END (fractional time), plus the spread
        hs = bt._options_half_spread(0.20, (date(2024, 3, 31) - date(2024, 3, 4)).days)
        want = bt._apply_exit_friction(pos, at_gap, 105.5, 3, None, hs, "credit", bar_end)
        assert got["exit_price"] == pytest.approx(round(want, 4), abs=1e-4)
        assert got["exit_price"] != pytest.approx(round(bt._apply_exit_friction(
            pos, bt._reprice_position(pos, 105.5, 3, None), 105.5, 3, None, hs, "credit"), 4), abs=1e-6)

    def test_l5_condors_are_not_invalidated_on_direction_when_realistic(self):
        class _Pred:
            def predict(self, hist):
                return SimpleNamespace(confidence=0.9, direction=SignalDirection.BULLISH)
        hist = _frame().iloc[:60].copy()
        hist.index = pd.bdate_range("2024-02-01", periods=60)
        legacy = _bt(entry_dte=30, predictor=_Pred(), engine_model="legacy")
        real = _bt(entry_dte=30, predictor=_Pred(), engine_model="realistic")
        assert legacy._check_thesis_invalidation(_pos(), hist)["exit_reason"].startswith("thesis_invalidated")
        assert real._check_thesis_invalidation(_pos(), hist) is None
        # only the live-exempt structures: a short strangle is still invalidated
        assert real._check_thesis_invalidation(_pos(strategy="short_strangle"), hist) is not None


# --------------------------------------------------------------------------
# the three exit knobs
# --------------------------------------------------------------------------

class TestExitKnobs:
    D_ENTRY, D_EXP = "2024-03-01", "2024-03-04"      # entered 3 days out

    def _daily(self, **kw):
        bt = _bt(entry_dte=3, credit_tp_ladder=_no_take_profit(), **kw)
        pos = _pos(entry=self.D_ENTRY, expiry=self.D_EXP, entry_price=0.05)   # mark >> credit: no take-profit
        row = pd.Series({"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0})
        return bt, pos, row

    def test_default_closes_a_short_dte_entry_on_the_first_check(self):
        bt, pos, row = self._daily()
        got = bt._check_exit(pos, row, date(2024, 3, 2), _frame().iloc[:40])
        assert got["exit_reason"] == "expiry_approaching"

    def test_guard_exempts_entries_at_or_below_the_threshold(self):
        bt, pos, row = self._daily(expiry_guard_dte=5)
        assert bt._check_exit(pos, row, date(2024, 3, 2), _frame().iloc[:40]) is None
        # ...but the rule still governs a 20-DTE entry
        bt2 = _bt(entry_dte=20, expiry_guard_dte=5, credit_tp_ladder=_no_take_profit())
        long = _pos(entry="2024-02-12", expiry="2024-03-03", entry_price=0.05)
        assert bt2._check_exit(long, row, date(2024, 3, 1), _frame().iloc[:40])["exit_reason"] == "expiry_approaching"

    def test_nothing_outlives_its_expiry_even_when_guard_exempt(self):
        bt, pos, row = self._daily(expiry_guard_dte=5)
        got = bt._check_exit(pos, row, date(2024, 3, 4), _frame().iloc[:40])
        assert got is not None and got["exit_reason"] == "expiry"

    def test_hold_fraction_replaces_the_day_cap(self):
        # a 10-day position; day-cap legacy would hold 30 days, the fraction closes at 50%
        kw = dict(entry_dte=10, expiry_guard_dte=10, max_hold_days=30, credit_tp_ladder=_no_take_profit())
        row = pd.Series({"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0})
        pos = _pos(entry="2024-03-01", expiry="2024-03-11", entry_price=0.05)
        off = _bt(**kw)
        frac = _bt(max_hold_fraction=0.5, **kw)
        # EOD decisions: day 4 is 40% of the life, day 5 is 50%
        assert frac._check_exit(dict(pos), row, date(2024, 3, 5), _frame().iloc[:40]) is None
        got = frac._check_exit(dict(pos), row, date(2024, 3, 6), _frame().iloc[:40])
        assert got is not None and got["exit_reason"] == "max_hold_reached"
        assert off._check_exit(dict(pos), row, date(2024, 3, 6), _frame().iloc[:40]) is None

    def test_flatten_fires_on_the_expiry_date_only(self):
        bt = _bt(entry_dte=3, expiry_guard_dte=5, flatten_minutes_before_close=60,
                 credit_tp_ladder=_no_take_profit())
        row = pd.Series({"Open": 100.0, "High": 100.5, "Low": 99.5, "Close": 100.0})
        pos = _pos(entry=self.D_ENTRY, expiry=self.D_EXP, entry_price=0.05)
        assert bt._check_exit(dict(pos), row, date(2024, 3, 3), _frame().iloc[:40]) is None
        got = bt._check_exit(dict(pos), row, date(2024, 3, 4), _frame().iloc[:40])
        assert got["exit_reason"] == "expiry_day_flatten"

    def test_intraday_flatten_fires_on_the_bar_that_ends_at_the_cutoff(self):
        bt = _bt(entry_dte=0, expiry_guard_dte=0, flatten_minutes_before_close=60, engine_model="realistic",
                 credit_tp_ladder=_no_take_profit())
        pos = _pos(entry="2024-03-04", expiry="2024-03-04", entry_price=0.05)
        prices = [100.0] * 78
        sess = _session(prices)                       # 14:30Z (10:30 ET) onward
        # session starts 10:30 ET, so bar k ends 10:35 + 5k; cutoff 15:00 ET = bar 54 ends 15:05? find it
        got = bt._check_intraday_exit(pos, sess, date(2024, 3, 4))
        assert got["exit_reason"] == "expiry_day_flatten"
        end_et = (pd.Timestamp(got["exit_time"]) + pd.Timedelta(minutes=5)).tz_convert(ET)
        assert end_et.time() >= time(15, 0) and (end_et - pd.Timedelta(minutes=5)).time() < time(15, 0)


class TestFullRunSameSession:
    def test_zero_dte_round_trip_closes_the_same_day_at_the_flatten(self):
        df = _calm_market()
        res = _run(df, engine_model="realistic", entry_dte=0, expiry_guard_dte=0,
                   flatten_minutes_before_close=60, credit_tp_ladder=_no_take_profit())
        assert res.trades
        t = res.trades[0]
        assert t["exit_reason"] == "expiry_day_flatten"
        assert t["exit_date"] == t["entry_date"] == t["expiry_date"]

    def test_without_the_guard_a_zero_dte_entry_is_closed_on_the_next_bar(self):
        df = _calm_market()
        res = _run(df, engine_model="realistic", entry_dte=0, credit_tp_ladder=_no_take_profit())
        t = res.trades[0]
        assert t["exit_date"] == t["entry_date"]
        assert t["exit_reason"] in ("expiry", "expiry_approaching")


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------

class TestPureHelpers:
    def test_guard(self):
        assert ep.expiry_guard_applies(3, 5) is False
        assert ep.expiry_guard_applies(5, 5) is False
        assert ep.expiry_guard_applies(6, 5) is True
        assert ep.expiry_guard_applies(0, None) is True

    def test_hold_limit(self):
        e = datetime(2026, 10, 5, 10, 0, tzinfo=ET)
        bell = datetime(2026, 10, 9, 16, 0, tzinfo=ET)
        half = e + (bell - e) / 2
        assert ep.hold_limit_reached(e, half - timedelta(seconds=1), bell, 0.5) is False
        assert ep.hold_limit_reached(e, half, bell, 0.5) is True
        assert ep.hold_limit_reached(e, bell, bell, None) is False
        assert ep.hold_limit_reached(bell, bell, bell, 0.5) is True       # entered at the bell
        with pytest.raises(ValueError):
            ep.hold_limit_reached(e, half, bell, 0.0)

    def test_flatten_due(self):
        exp = date(2026, 10, 9)
        f = ep.expiry_day_flatten_due
        assert f(datetime(2026, 10, 9, 14, 59, tzinfo=ET), exp, 60) is False
        assert f(datetime(2026, 10, 9, 15, 0, tzinfo=ET), exp, 60) is True
        assert f(datetime(2026, 10, 8, 15, 30, tzinfo=ET), exp, 60) is False   # day before
        assert f(datetime(2026, 10, 12, 9, 30, tzinfo=ET), exp, 60) is True    # after expiry
        assert f(datetime(2026, 10, 9, 15, 30, tzinfo=ET), exp, None) is False
        # early close: 13:00 bell, 60 min -> 12:00
        assert f(datetime(2026, 11, 27, 11, 59, tzinfo=ET), date(2026, 11, 27), 60, time(13, 0)) is False
        assert f(datetime(2026, 11, 27, 12, 0, tzinfo=ET), date(2026, 11, 27), 60, time(13, 0)) is True

    def test_time_to_expiry_is_fractional_and_session_aware(self):
        yr = 365.0 * 86400.0
        assert time_to_expiry_years(datetime(2026, 10, 9, 10, 30, tzinfo=ET), date(2026, 10, 9)) == \
            pytest.approx(5.5 * 3600 / yr)
        assert time_to_expiry_years(datetime(2026, 10, 9, 16, 30, tzinfo=ET), date(2026, 10, 9)) == 0.0
        # early close: Friday after Thanksgiving ends at 13:00
        assert time_to_expiry_years(datetime(2026, 11, 27, 10, 30, tzinfo=ET), date(2026, 11, 27)) == \
            pytest.approx(2.5 * 3600 / yr)
        with pytest.raises(ValueError):
            time_to_expiry_years(datetime(2026, 10, 9, 10, 30), date(2026, 10, 9))

    def test_expiry_calendar(self):
        assert listed_expiry(date(2026, 10, 9), 1) == date(2026, 10, 12)       # Sat -> Mon
        assert listed_expiry(date(2026, 10, 5), 4) == date(2026, 10, 9)        # a session stays
        assert listed_expiry(date(2026, 11, 25), 1) == date(2026, 11, 27)      # Thanksgiving -> Fri
        assert actual_dte(date(2026, 10, 9), 1) == 3
        assert listed_expiry(date(2026, 10, 5), 4, listed=[date(2026, 10, 16)]) == date(2026, 10, 16)
        # several candidates: the EARLIEST on/after the target, whatever the input order
        assert listed_expiry(date(2026, 10, 5), 4, listed=[
            date(2026, 10, 23), date(2026, 10, 12), date(2026, 10, 8), date(2026, 10, 16)]) == date(2026, 10, 12)
        # a listed set with nothing that far out falls back to the session rule
        assert listed_expiry(date(2026, 10, 5), 4, listed=[date(2026, 10, 6)]) == date(2026, 10, 9)
        assert next_session_on_or_after(date(2026, 10, 10)) == date(2026, 10, 12)
        with pytest.raises(ValueError):
            listed_expiry(date(2026, 10, 5), -1)

    def test_expected_move_never_collapses_to_zero_with_hours_left(self):
        assert expected_move(100.0, 0.2, 5.5 * 3600 / (365 * 86400)) > 0.3
        assert expected_move(100.0, 0.2, 0.0) == 0.0
        assert expected_move(0.0, 0.2, 0.1) == 0.0
        assert expected_move(100.0, 0.25, 30 / 365) == pytest.approx(100 * 0.25 * np.sqrt(30 / 365))

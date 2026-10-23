"""W72: parity replay + the single-owner condor pricing it rests on.

House rules: every test drives the real code path (no assertions on source
strings); an AST guard catches the NEXT hand-copy of the condor pricing; the
fixtures are SYNTHETIC — CI never reads data/replay (git-ignored, private).
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.backtesting import replay as rp
from ait.backtesting.friction import LegSpreadModel
from ait.backtesting.pricing import (
    OptionType,
    black_scholes_price,
    ic_leg_marks,
    skewed_leg_iv,
)
from ait.utils.time import ET

P = rp.PricingParams(r=0.05, skew_factor=1.0, skew_slope_per_pct_otm=0.0736, iv_floor=0.05)


# --------------------------------------------------------------------------
# Single-owner pricing: the pure functions equal the engine's historical maths
# --------------------------------------------------------------------------

def _legacy_leg_iv(base_iv, strike, S, opt, skew_factor, slope, floor):
    """The engine's pre-W72 inline formula, kept here as the reference."""
    log_m = np.log(strike / S)
    if opt == OptionType.PUT:
        otm = max(0.0, -log_m)
        adj = skew_factor * otm * 0.10 + base_iv * slope * (otm * 100.0)
    else:
        adj = skew_factor * max(0.0, log_m) * 0.02
    return max(base_iv + adj, floor)


class TestSharedPricing:
    @pytest.mark.parametrize("seed", range(25))
    def test_skewed_leg_iv_equals_the_historical_formula(self, seed):
        rng = np.random.default_rng(seed)
        S = float(rng.uniform(50, 800)); K = S * float(rng.uniform(0.85, 1.15))
        iv = float(rng.uniform(0.08, 0.6)); sf = float(rng.uniform(0, 2))
        slope = float(rng.uniform(0, 0.15)); floor = float(rng.uniform(0.01, 0.25))
        for opt in (OptionType.CALL, OptionType.PUT):
            assert skewed_leg_iv(iv, K, S, opt, sf, slope, floor, 0.10, 0.02) == _legacy_leg_iv(
                iv, K, S, opt, sf, slope, floor)

    @pytest.mark.parametrize("seed", range(15))
    def test_ic_leg_marks_equals_four_independent_black_scholes_calls(self, seed):
        rng = np.random.default_rng(100 + seed)
        S = float(rng.uniform(80, 700)); w = float(rng.uniform(1, 15))
        sc = S * 1.02; sp = S * 0.98
        t = float(rng.uniform(0.001, 0.2)); iv = float(rng.uniform(0.1, 0.5))
        got = ic_leg_marks(S, sc, sp, sc + w, sp - w, t, 0.05, iv, 1.0, 0.0736, 0.05, 0.10, 0.02)
        ref = (
            black_scholes_price(S, sc, t, 0.05, _legacy_leg_iv(iv, sc, S, OptionType.CALL, 1.0, 0.0736, 0.05), OptionType.CALL),
            black_scholes_price(S, sp, t, 0.05, _legacy_leg_iv(iv, sp, S, OptionType.PUT, 1.0, 0.0736, 0.05), OptionType.PUT),
            black_scholes_price(S, sc + w, t, 0.05, _legacy_leg_iv(iv, sc + w, S, OptionType.CALL, 1.0, 0.0736, 0.05), OptionType.CALL),
            black_scholes_price(S, sp - w, t, 0.05, _legacy_leg_iv(iv, sp - w, S, OptionType.PUT, 1.0, 0.0736, 0.05), OptionType.PUT),
        )
        assert got == ref  # bit-identical, not approximately equal

    def test_engine_methods_delegate_to_the_shared_functions(self):
        from ait.backtesting.engine import Backtester
        idx = pd.bdate_range("2024-01-02", periods=40)
        c = 100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0, 0.005, len(idx))))
        df = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6}, index=idx)
        bt = Backtester(data=df, strategies=["iron_condor"], entry_dte=14, features_cache=None,
                        macro_event_gate=False, allow_live_model_fallback=False,
                        skew_factor=1.3, skew_slope_per_pct_otm=0.09, iv_floor=0.07)
        assert bt._get_leg_iv(0.21, 97.0, 100.0, OptionType.PUT) == skewed_leg_iv(
            0.21, 97.0, 100.0, OptionType.PUT, 1.3, 0.09, 0.07, 0.10, 0.02)
        pos = {"short_call_strike": 104.0, "short_put_strike": 96.0,
               "long_call_strike": 107.0, "long_put_strike": 93.0, "entry_iv": 0.2}
        got = bt._reprice_ic_legs(pos, 100.0, 3, None)
        t = max(max(14 - 3, 0) / 365.0, 0.0001)
        assert got == ic_leg_marks(100.0, 104.0, 96.0, 107.0, 93.0, t, 0.05, 0.2, 1.3, 0.09, 0.07, 0.10, 0.02)

    def test_ast_guard_condor_pricing_has_one_owner(self):
        """No function in engine.py may price the four condor legs by hand again:
        the iron-condor entry build and re-mark must go through ic_leg_marks."""
        from ait.backtesting import engine
        tree = ast.parse(inspect.getsource(engine))

        def calls(fn: ast.AST) -> set[str]:
            return {n.func.id for n in ast.walk(fn)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "ic_leg_marks" in calls(funcs["_reprice_ic_legs"])
        assert "black_scholes_price" not in calls(funcs["_reprice_ic_legs"])
        # the iron_condor branch of the credit builder
        build = funcs["_build_credit_position"]
        assert "ic_leg_marks" in calls(build)
        # _get_leg_iv is a thin delegate: no log/exp arithmetic of its own
        assert "skewed_leg_iv" in calls(funcs["_get_leg_iv"])


# --------------------------------------------------------------------------
# Synthetic research store + trades
# --------------------------------------------------------------------------

def _make_db(path: Path, *, with_leg_model: bool = True) -> Path:
    c = sqlite3.connect(path)
    c.executescript("""
      CREATE TABLE research_daily_prices (symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
          close REAL, volume REAL, implied_vol REAL, source TEXT, iv_source TEXT);
      CREATE TABLE research_intraday_prices (symbol TEXT, datetime TEXT, interval TEXT, open REAL,
          high REAL, low REAL, close REAL, volume REAL, source TEXT);
      CREATE TABLE option_leg_spread_params (symbol TEXT, form TEXT, a REAL, b REAL, n INT,
          n_sessions INT, cv_mae REAL, cv_mae_json TEXT, source TEXT, calibrated_on TEXT);
    """)
    days = pd.bdate_range("2026-08-03", "2026-09-30")
    px = 100.0
    rng = np.random.default_rng(7)
    for i, d in enumerate(days):
        px *= float(np.exp(rng.normal(0, 0.004)))
        iv = 0.18 + 0.001 * (i % 5)
        c.execute("INSERT INTO research_daily_prices VALUES (?,?,?,?,?,?,?,?,?,?)",
                  ("QQQ", d.strftime("%Y-%m-%d"), px, px * 1.004, px * 0.996, px, 1e6, iv, "t", "t"))
    # one 09:30-16:00 ET session of 5m bars on 2026-08-17 (EDT = UTC-4)
    start = datetime(2026, 8, 17, 13, 30, tzinfo=timezone.utc)
    for k in range(78):
        ts = start + timedelta(minutes=5 * k)
        o = 100.0 + 0.01 * k
        c.execute("INSERT INTO research_intraday_prices VALUES (?,?,?,?,?,?,?,?,?)",
                  ("QQQ", ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"), "5m", o, o + 0.05, o - 0.05, o + 0.02, 1e5, "t"))
    if with_leg_model:
        c.execute("INSERT INTO option_leg_spread_params VALUES (?,?,?,?,?,?,?,?,?,?)",
                  ("QQQ", "const_usd", 0.02, 0.0, 100, 10, 0.003, "{}", "t", "2026-10-09"))
    c.commit(); c.close()
    return path


def _leg(side, right, strike, expiry, px, close_time=None, close_px=None, code=None):
    return rp.ReplayLeg(side, right, float(strike), expiry, px, close_time, close_px, code, 0.0)


def _condor(entry: datetime, expiry: date, *, qty=1, legs_px=(1.0, 1.0, 0.3, 0.3), strikes=(104, 96, 107, 93),
            still_open=False, closed_at=None, code="C", close_px=(0.2, 0.2, 0.0, 0.0)) -> rp.ReplayTrade:
    sc, sp, lc, lp = strikes
    ct = closed_at
    legs = (
        _leg("short", "C", sc, expiry, legs_px[0], ct, close_px[0] if ct else None, code if ct else None),
        _leg("short", "P", sp, expiry, legs_px[1], ct, close_px[1] if ct else None, code if ct else None),
        _leg("long", "C", lc, expiry, legs_px[2], ct, close_px[2] if ct else None, code if ct else None),
        _leg("long", "P", lp, expiry, legs_px[3], ct, close_px[3] if ct else None, code if ct else None),
    )
    credit = (legs_px[0] + legs_px[1] - legs_px[2] - legs_px[3])
    return rp.ReplayTrade("T1", "manual", None, entry, qty, "iron_condor", legs, still_open,
                          credit * 100 * qty, 2.5, None)


ENTRY = datetime(2026, 8, 17, 10, 7, 30, tzinfo=ET)   # inside the 14:05Z bar
EXPIRY = date(2026, 8, 28)


class TestLoader:
    def _doc(self, **over):
        s = {"id": "QQQ-X", "origin": "manual", "bot_trade_id": None,
             "entry_time": "2026-08-17T10:07:30", "quantity": 2, "still_open": False,
             "structure": "iron_condor", "net_credit_usd": 300.0, "commissions_usd": -4.0,
             "total_pnl_usd": 280.0,
             "legs": [
                 {"side": "short", "right": "C", "strike": 104.0, "expiry": "2026-08-28", "open_price": 1.0,
                  "close_time": "2026-08-28 16:20:00", "close_price": 0.0, "close_code": "C;Ep", "realized_usd": 100.0},
                 {"side": "long", "right": "C", "strike": 107.0, "expiry": "2026-08-28", "open_price": 0.2,
                  "close_time": None, "close_price": None, "close_code": None, "realized_usd": 0.0}]}
        s.update(over)
        return {"schema": rp.SCHEMA, "structures": [s]}

    def test_roundtrip_and_derived_fields(self, tmp_path):
        p = tmp_path / "t.json"; p.write_text(json.dumps(self._doc()))
        (t,) = rp.load_trades(p)
        assert t.entry_time.tzinfo is not None and t.entry_time.utcoffset() == timedelta(hours=-4)
        assert t.entry_date == date(2026, 8, 17) and t.front_expiry == date(2026, 8, 28)
        assert t.entry_dte == 11
        assert t.actual_credit == pytest.approx(300.0 / (100 * 2))
        assert t.condor_roles() is None          # two legs only: not a condor
        assert t.parity_eligible is True         # but the structure label is a buildable one

    def test_wrong_schema_is_refused(self, tmp_path):
        p = tmp_path / "t.json"; d = self._doc(); d["schema"] = "other/9"; p.write_text(json.dumps(d))
        with pytest.raises(ValueError):
            rp.load_trades(p)

    def test_condor_roles_and_close_instant(self):
        t = _condor(ENTRY, EXPIRY, closed_at=datetime(2026, 8, 20, 11, 0, tzinfo=ET))
        assert set(t.condor_roles()) == {"short_call", "short_put", "long_call", "long_put"}
        assert t.legs_closed_together() == datetime(2026, 8, 20, 11, 0, tzinfo=ET)
        # expiry/assignment closes are NOT a trader close instant
        t2 = _condor(ENTRY, EXPIRY, closed_at=datetime(2026, 8, 28, 16, 20, tzinfo=ET), code="C;Ep")
        assert t2.legs_closed_together() is None
        assert _condor(ENTRY, EXPIRY, still_open=True).legs_closed_together() is None


class TestResearchData:
    def test_reads_are_correct_and_never_modify_the_file(self, tmp_path):
        db = _make_db(tmp_path / "h.db")
        before = hashlib.sha256(db.read_bytes()).hexdigest()
        d = rp.ResearchData(db, "qqq", prefix="research_")
        row = d.session_row(date(2026, 8, 17))
        assert row is not None and row["Close"] > 0
        iv_same = d.stored_iv(date(2026, 8, 17))
        iv_prev = d.stored_iv(date(2026, 8, 17), strictly_before=True)
        assert iv_same is not None and iv_prev is not None
        m = d.leg_spread_model()
        assert m is not None and m.form == "const_usd" and m.half_spread_usd(0.5) == pytest.approx(0.02)
        assert len(d.closes_through(date(2026, 8, 17))) == 11
        assert hashlib.sha256(db.read_bytes()).hexdigest() == before

    def test_missing_db_is_an_error_not_a_new_empty_file(self, tmp_path):
        ghost = tmp_path / "nope.db"
        with pytest.raises(sqlite3.OperationalError):
            rp.ResearchData(ghost, "QQQ", prefix="research_").daily_frame()
        assert not ghost.exists()

    def test_bar_containment_boundaries(self, tmp_path):
        d = rp.ResearchData(_make_db(tmp_path / "h.db"), "QQQ", prefix="research_")
        inside = d.bar_containing(datetime(2026, 8, 17, 9, 32, 59, tzinfo=ET))      # 13:32:59Z -> 13:30 bar
        assert inside is not None and inside["open"] == pytest.approx(100.0)
        nxt = d.bar_containing(datetime(2026, 8, 17, 9, 35, 0, tzinfo=ET))           # exactly the next bar
        assert nxt is not None and nxt["open"] == pytest.approx(100.01)
        assert d.bar_containing(datetime(2026, 8, 17, 8, 0, tzinfo=ET)) is None      # pre-market: no bar
        assert d.bar_containing(datetime(2026, 8, 17, 17, 0, tzinfo=ET)) is None     # after the session

    def test_a_gap_in_the_bars_is_not_filled_by_the_previous_bar(self, tmp_path):
        db = _make_db(tmp_path / "h.db")
        c = sqlite3.connect(db)
        c.execute("DELETE FROM research_intraday_prices WHERE datetime = '2026-08-17T13:35:00+00:00'")
        c.commit(); c.close()
        d = rp.ResearchData(db, "QQQ", prefix="research_")
        # 09:35:00 ET is exactly 5:00 after the 13:30Z bar started: that bar's
        # interval is half-open, so the instant is NOT inside it (bar missing).
        assert d.bar_containing(datetime(2026, 8, 17, 9, 35, 0, tzinfo=ET)) is None
        assert d.bar_containing(datetime(2026, 8, 17, 9, 34, 59, tzinfo=ET)) is not None

    def test_wrong_table_prefix_fails_loudly_instead_of_silently_degrading(self, tmp_path):
        d = rp.ResearchData(_make_db(tmp_path / "h.db"), "QQQ", prefix="typo_")
        loud = (sqlite3.Error, pd.errors.DatabaseError)   # pandas wraps the sqlite error
        with pytest.raises(loud):
            d.daily_frame()
        with pytest.raises(loud):
            d.bar_containing(ENTRY)

    def test_no_leg_model_returns_none(self, tmp_path):
        d = rp.ResearchData(_make_db(tmp_path / "h.db", with_leg_model=False), "QQQ", prefix="research_")
        assert d.leg_spread_model() is None


class TestClock:
    def test_time_to_expiry_runs_to_the_1600_et_close(self):
        yr = 365.0 * 86400.0
        exp = date(2026, 8, 28)
        assert rp.t_clean(exp, datetime(2026, 8, 28, 15, 0, tzinfo=ET)) == pytest.approx(3600 / yr)
        assert rp.t_clean(exp, datetime(2026, 8, 28, 16, 0, tzinfo=ET)) == 0.0
        assert rp.t_clean(exp, datetime(2026, 8, 28, 16, 20, tzinfo=ET)) == 0.0   # floored, never negative
        # a full session earlier: 1 day + 6.5h to the close
        assert rp.t_clean(exp, datetime(2026, 8, 27, 9, 30, tzinfo=ET)) == pytest.approx(
            (24 + 6.5) * 3600 / yr)

    def test_engine_time_is_whole_calendar_days_and_never_negative(self):
        assert rp.t_engine(date(2026, 8, 28), date(2026, 8, 17)) == pytest.approx(11 / 365.0)
        assert rp.t_engine(date(2026, 8, 28), date(2026, 8, 28)) == 0.0
        assert rp.t_engine(date(2026, 8, 28), date(2026, 8, 31)) == 0.0


class TestImpliedVol:
    @pytest.mark.parametrize("opt", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("K", [92.0, 100.0, 108.0])
    def test_roundtrip(self, opt, K):
        S, t, r, s = 100.0, 0.05, 0.05, 0.23
        px = black_scholes_price(S, K, t, r, s, opt)
        assert rp.implied_vol_from_price(px, S, K, t, r, opt) == pytest.approx(s, abs=1e-4)

    def test_unreachable_prices_are_none_not_garbage(self):
        assert rp.implied_vol_from_price(0.0, 100, 110, 0.05, 0.05, OptionType.CALL) is None
        assert rp.implied_vol_from_price(-1.0, 100, 110, 0.05, 0.05, OptionType.CALL) is None
        assert rp.implied_vol_from_price(0.5, 100, 110, 0.0, 0.05, OptionType.CALL) is None
        # a call can never be worth more than the stock
        assert rp.implied_vol_from_price(150.0, 100, 110, 0.05, 0.05, OptionType.CALL) is None


class TestEntryReplay:
    def _setup(self, tmp_path, friction=None):
        db = _make_db(tmp_path / "h.db")
        data = rp.ResearchData(db, "QQQ", prefix="research_")
        # make the 'actual' fills equal to what the clean model predicts, so error == 0
        probe = _condor(ENTRY, EXPIRY)
        e = rp.entry_replay(probe, data, P, friction)["clean"]
        px = tuple(round(x, 6) for x in e["leg_fill"])
        t = _condor(ENTRY, EXPIRY, legs_px=px)
        return data, t

    def test_zero_error_when_actual_equals_model(self, tmp_path):
        data, t = self._setup(tmp_path)
        res = rp.entry_replay(t, data, P, None)
        assert res["clean"]["err"] == pytest.approx(0.0, abs=1e-5)
        assert res["engine"] is not None  # daily close + same-day IV exist for 08-17

    def test_sign_convention_positive_means_model_richer(self, tmp_path):
        data, t = self._setup(tmp_path)
        cheaper = _condor(ENTRY, EXPIRY, legs_px=tuple(l.open_price for l in t.legs))
        # actual credit 0.10 LOWER than the model's -> model is richer -> positive error
        px = list(l.open_price for l in t.legs); px[0] -= 0.10
        lower = _condor(ENTRY, EXPIRY, legs_px=tuple(px))
        got = rp.entry_replay(lower, data, P, None)["clean"]["err"]
        assert got == pytest.approx(+0.10, abs=1e-5)
        assert rp.entry_replay(cheaper, data, P, None)["clean"]["err"] == pytest.approx(0.0, abs=1e-5)

    def test_friction_is_charged_on_all_four_legs_at_the_right_side(self, tmp_path):
        data, t = self._setup(tmp_path)
        fr = LegSpreadModel("const_usd", a=0.02)
        res = rp.entry_replay(t, data, P, fr)["clean"]
        assert res["friction_total"] == pytest.approx(0.08)
        # opening: shorts sell at the bid (mid - hs), longs buy the ask (mid + hs)
        assert res["credit_fill"] == pytest.approx(res["credit_mid"] - 0.08)
        for leg, mid, fill in zip(t.legs, res["leg_mid"], res["leg_fill"]):
            assert fill == pytest.approx(mid - 0.02 if leg.side == "short" else mid + 0.02)

    def test_engine_variant_uses_same_day_iv_and_clean_the_previous_session(self, tmp_path):
        data, t = self._setup(tmp_path)
        res = rp.entry_replay(t, data, P, None)
        assert res["engine"]["iv"] == data.stored_iv(t.entry_date)
        assert res["clean"]["iv"] == data.stored_iv(t.entry_date, strictly_before=True)
        assert res["engine"]["spot_src"] == "daily_close" and res["clean"]["spot_src"] == "bar_mid"

    def test_missing_intraday_bar_degrades_clean_only(self, tmp_path):
        db = _make_db(tmp_path / "h.db")
        data = rp.ResearchData(db, "QQQ", prefix="research_")
        t = _condor(datetime(2026, 8, 18, 10, 0, tzinfo=ET), EXPIRY)   # no bars that day
        res = rp.entry_replay(t, data, P, None)
        assert res["clean"] is None and res["engine"] is not None

    def test_leg_implied_iv_matches_the_model_when_actual_is_the_model(self, tmp_path):
        data, t = self._setup(tmp_path, friction=LegSpreadModel("const_usd", a=0.02))
        res = rp.entry_replay(t, data, P, LegSpreadModel("const_usd", a=0.02))["clean"]
        for market, model in zip(res["leg_actual_iv"], res["leg_model_iv"]):
            assert market == pytest.approx(model, abs=2e-4)


class TestExitReplay:
    def test_value_error_sign_and_calendar_engine_variant_suppressed(self, tmp_path):
        db = _make_db(tmp_path / "h.db")
        data = rp.ResearchData(db, "QQQ", prefix="research_")
        when = datetime(2026, 8, 17, 14, 0, tzinfo=ET)
        t = _condor(ENTRY, EXPIRY, closed_at=when, close_px=(0.5, 0.4, 0.05, 0.05))
        entry = rp.entry_replay(t, data, P, None)
        ex = rp.exit_replay(t, data, P, None, entry)
        assert ex is not None
        # buy back the shorts, sell the longs: 0.5 + 0.4 - 0.05 - 0.05
        assert ex["actual_exit_value"] == pytest.approx(0.80)
        assert ex["clean"]["err"] == pytest.approx(ex["clean"]["value_fill"] - 0.80)
        # a calendar (two expiries) has no meaningful engine-variant re-mark
        cal_legs = (
            _leg("short", "C", 104, EXPIRY, 1.0, when, 0.5, "C"),
            _leg("long", "C", 104, date(2026, 9, 4), 1.4, when, 1.2, "C"))
        cal = rp.ReplayTrade("C1", "manual", None, ENTRY, 1, "calendar", cal_legs, False, -40.0, 1.0, None)
        cex = rp.exit_replay(cal, data, P, None, rp.entry_replay(cal, data, P, None))
        assert cex["engine"] is None

    def test_not_applicable_when_legs_closed_at_expiry_or_open(self, tmp_path):
        data = rp.ResearchData(_make_db(tmp_path / "h.db"), "QQQ", prefix="research_")
        t = _condor(ENTRY, EXPIRY, still_open=True)
        assert rp.exit_replay(t, data, P, None, rp.entry_replay(t, data, P, None)) is None


class TestEngineExitScan:
    def _data_with_path(self, tmp_path, closes, lows=None, highs=None):
        db = tmp_path / "h.db"
        c = sqlite3.connect(db)
        c.executescript("""
          CREATE TABLE research_daily_prices (symbol TEXT, date TEXT, open REAL, high REAL, low REAL,
              close REAL, volume REAL, implied_vol REAL, source TEXT, iv_source TEXT);
          CREATE TABLE research_intraday_prices (symbol TEXT, datetime TEXT, interval TEXT, open REAL,
              high REAL, low REAL, close REAL, volume REAL, source TEXT);""")
        days = pd.bdate_range("2026-06-01", periods=len(closes))
        for i, d in enumerate(days):
            lo = lows[i] if lows else closes[i] * 0.998
            hi = highs[i] if highs else closes[i] * 1.002
            c.execute("INSERT INTO research_daily_prices VALUES (?,?,?,?,?,?,?,?,?,?)",
                      ("QQQ", d.strftime("%Y-%m-%d"), closes[i], hi, lo, closes[i], 1e6, 0.2, "t", "t"))
        c.commit(); c.close()
        return db, days

    def test_touch_stop_fires_on_the_day_low_reaches_the_short_put(self, tmp_path):
        n = 60
        closes = [100.0] * n
        lows = [99.8] * n
        db, days = self._data_with_path(tmp_path, closes, lows=lows)
        # day 45 pierces the short put at 96
        c = sqlite3.connect(db)
        c.execute("UPDATE research_daily_prices SET low = 95.5 WHERE date = ?", (days[45].strftime("%Y-%m-%d"),))
        c.commit(); c.close()
        entry = datetime.combine(days[40].date(), datetime.min.time(), tzinfo=ET).replace(hour=10)
        expiry = days[40].date() + timedelta(days=30)
        t = _condor(entry, expiry, strikes=(104, 96, 107, 93))
        data = rp.ResearchData(db, "QQQ", prefix="research_")
        e = rp.entry_replay(t, data, P, None)
        before = hashlib.sha256(db.read_bytes()).hexdigest()
        scan = rp.engine_exit_scan(t, data, e)
        assert scan["engine_exit_reason"] == "touch_stop"
        assert scan["engine_exit_date"] == days[45].strftime("%Y-%m-%d")
        assert hashlib.sha256(db.read_bytes()).hexdigest() == before   # no DB writes

    def test_a_position_is_first_checked_the_session_after_entry(self, tmp_path):
        """run() exits positions BEFORE entering, so an entry-day pierce is never an exit."""
        closes = [100.0] * 70
        db, days = self._data_with_path(tmp_path, closes)
        c = sqlite3.connect(db)
        c.execute("UPDATE research_daily_prices SET low = 95.0 WHERE date = ?",
                  (days[40].strftime("%Y-%m-%d"),))     # the ENTRY day pierces the 96 put
        c.commit(); c.close()
        entry = datetime.combine(days[40].date(), datetime.min.time(), tzinfo=ET).replace(hour=10)
        t = _condor(entry, days[40].date() + timedelta(days=30), strikes=(104, 96, 107, 93))
        data = rp.ResearchData(db, "QQQ", prefix="research_")
        scan = rp.engine_exit_scan(t, data, rp.entry_replay(t, data, P, None))
        assert scan["engine_exit_date"] != days[40].strftime("%Y-%m-%d")
        assert scan["engine_exit_reason"] != "touch_stop"

    def test_quiet_market_holds_until_the_expiry_approaching_rule(self, tmp_path):
        closes = [100.0] * 80
        db, days = self._data_with_path(tmp_path, closes)
        entry = datetime.combine(days[40].date(), datetime.min.time(), tzinfo=ET).replace(hour=10)
        expiry = days[40].date() + timedelta(days=14)
        # actual credit so rich that 50% take-profit is unreachable at flat prices
        t = _condor(entry, expiry, legs_px=(4.0, 4.0, 0.1, 0.1), strikes=(104, 96, 107, 93))
        data = rp.ResearchData(db, "QQQ", prefix="research_")
        scan = rp.engine_exit_scan(t, data, rp.entry_replay(t, data, P, None))
        # DTE <= EXPIRY_APPROACHING_DTE (5) on the first session with <= 5 calendar days left
        assert scan["engine_exit_reason"] in ("expiry_approaching", "take_profit_short")
        if scan["engine_exit_reason"] == "expiry_approaching":
            got = date.fromisoformat(scan["engine_exit_date"])
            assert (expiry - got).days <= 5

    def test_env_flag_is_restored_after_the_scan(self, tmp_path, monkeypatch):
        import os
        monkeypatch.delenv("AIT_BT_CALIBRATED_SPREADS", raising=False)
        db, days = self._data_with_path(tmp_path, [100.0] * 70)
        entry = datetime.combine(days[40].date(), datetime.min.time(), tzinfo=ET).replace(hour=10)
        t = _condor(entry, days[40].date() + timedelta(days=21))
        data = rp.ResearchData(db, "QQQ", prefix="research_")
        rp.engine_exit_scan(t, data, rp.entry_replay(t, data, P, None))
        assert "AIT_BT_CALIBRATED_SPREADS" not in os.environ
        monkeypatch.setenv("AIT_BT_CALIBRATED_SPREADS", "1")
        rp.engine_exit_scan(t, data, rp.entry_replay(t, data, P, None))
        assert os.environ["AIT_BT_CALIBRATED_SPREADS"] == "1"


class TestVerdicts:
    KW = dict(error_max=0.15, bias_max=0.10, min_trades=3)

    def test_pass_fail_insufficient(self):
        assert rp.verdict_for([0.02, -0.03, 0.04], **self.KW)[0] == "PASS"
        assert rp.verdict_for([0.30, -0.25, 0.28], **self.KW)[0] == "FAIL"
        assert rp.verdict_for([0.01, 0.02], **self.KW)[0] == "INSUFFICIENT"   # never PASS on a thin band

    def test_a_consistent_bias_fails_even_when_spread_passes(self):
        verdict, mabs, msgn = rp.verdict_for([0.12, 0.12, 0.12, 0.12], **self.KW)
        assert mabs == pytest.approx(0.12) and msgn == pytest.approx(0.12)
        assert verdict == "FAIL"   # |median| 0.12 > 0.10 although median |err| 0.12 <= 0.15

    def test_nan_errors_do_not_count_toward_the_minimum(self):
        assert rp.verdict_for([0.01, float("nan"), 0.02], **self.KW)[0] == "INSUFFICIENT"

    def test_band_labels_and_order(self):
        edges = [2, 7, 14, 45]
        assert [rp.band_label(d, edges) for d in (0, 2, 3, 7, 8, 14, 15, 45, 46, 200)] == [
            "0-2", "0-2", "3-7", "3-7", "8-14", "8-14", "15-45", "15-45", "46+", "46+"]
        assert rp.band_order(edges) == ["0-2", "3-7", "8-14", "15-45", "46+"]

    def test_summarise_excludes_ineligible_structures(self):
        rows = [{"band": "0-2", "variant": "clean", "err_pct": 0.5, "eligible": False}] * 5
        rows += [{"band": "0-2", "variant": "clean", "err_pct": 0.01, "eligible": True}] * 3
        out = {(v.band, v.variant): v for v in rp.summarise_bands(rows, [2, 7], **{
            "error_max": 0.15, "bias_max": 0.10, "min_trades": 3})}
        assert out[("0-2", "clean")].n == 3 and out[("0-2", "clean")].verdict == "PASS"


class TestParityConfig:
    def test_defaults_and_validation(self):
        from ait.config.settings import ParityConfig
        c = ParityConfig()
        assert (c.credit_error_pct_max, c.credit_bias_pct_max, c.min_trades_per_band) == (0.15, 0.10, 3)
        assert c.dte_band_upper_edges == [2, 7, 14, 45]
        for bad in ([], [7, 2], [2, 2, 9], [-1, 5]):
            with pytest.raises(ValueError):
                ParityConfig(dte_band_upper_edges=bad)
        with pytest.raises(ValueError):
            ParityConfig(credit_error_pct_max=0.0)
        with pytest.raises(ValueError):
            ParityConfig(not_a_key=1)   # extra="forbid": a typo is a startup error

    def test_research_overlay_carries_the_block_and_live_does_not_need_it(self):
        from ait.config.research import use_research_config  # noqa: F401  (import must work)
        import yaml
        root = Path(__file__).resolve().parents[1]
        research = yaml.safe_load((root / "config_research.yaml").read_text())
        live = yaml.safe_load((root / "config.yaml").read_text())
        assert "parity" in research and "parity" not in live
        from ait.config.settings import ParityConfig
        assert ParityConfig(**research["parity"]).dte_band_upper_edges == [2, 7, 14, 45]


class TestPrivacyGuard:
    def _root(self):
        return Path(__file__).resolve().parents[1]

    def test_replay_data_directory_is_git_ignored(self):
        import subprocess
        r = subprocess.run(["git", "check-ignore", "-q", "data/replay/out/x/trades.csv"],
                           cwd=self._root())
        assert r.returncode == 0   # ignored

    def test_cli_refuses_to_write_per_trade_outputs_into_tracked_paths(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "parity_replay_cli", self._root() / "scripts" / "parity_replay.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        probe = self._root() / "reports" / "_w72_guard_probe"
        try:
            rc = mod.main(["--out-dir", str(probe), "--trades", str(tmp_path / "none.json")])
            assert rc == 2
            assert not probe.exists()          # refused BEFORE creating anything
        finally:
            if probe.exists():
                probe.rmdir()

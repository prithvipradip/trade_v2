"""W75: the real-bar IV surface and its engine switch.

Synthetic bars with a KNOWN smile and term structure, so every claim is checkable
against the generating function. CI never needs the private research DB.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.backtesting import iv_surface as ivs
from ait.backtesting.iv_surface import IVSurface, fit_smile, usable_rows
from ait.backtesting.pricing import OptionType, black_scholes_price, ic_leg_marks_from_ivs

S0 = 500.0


def true_iv(dte: int, m: float) -> float:
    """The generating smile: level rises with DTE, put skew, mild convexity."""
    return 0.14 + 0.004 * dte - 0.35 * m + 1.2 * m * m


def make_bars(dates, *, max_dte=10, spot=S0, vol=100, price=0.50, outcome="solved", iv_shift=0.0):
    rows = []
    for d in dates:
        for k in range(1, max_dte + 1):
            e = d + timedelta(days=k)
            for strike in range(int(spot * 0.92), int(spot * 1.08) + 1):
                m = math.log(strike / spot)
                rows.append(dict(
                    date=d, expiry=e, right="P" if strike < spot else "C", strike=float(strike),
                    dte=k, close=price, volume=vol, underlying_close=spot,
                    iv=true_iv(k, m) + iv_shift, iv_outcome=outcome, risk_free_rate=0.05))
    return pd.DataFrame(rows)


DATES = [date(2025, 3, 3) + timedelta(days=i) for i in range(6)]


@pytest.fixture(scope="module")
def bars():
    return make_bars(DATES)


@pytest.fixture(scope="module")
def surface(bars):
    return IVSurface.from_bars(bars)


# --------------------------------------------------------------------------
# fit + row filter
# --------------------------------------------------------------------------

class TestFit:
    def test_recovers_a_known_quadratic(self):
        m = np.linspace(-0.08, 0.08, 25)
        iv = 0.2 - 0.3 * m + 1.5 * m * m
        a, b, c, rmse = fit_smile(m, iv, np.ones_like(m))
        assert (a, b, c) == pytest.approx((0.2, -0.3, 1.5), abs=1e-9)
        assert rmse == pytest.approx(0.0, abs=1e-9)

    def test_refuses_what_it_cannot_support(self):
        m = np.linspace(-0.08, 0.08, 4)                       # too few points
        assert fit_smile(m, np.full(4, 0.2), np.ones(4)) is None
        m2 = np.linspace(-0.002, 0.002, 12)                   # span below 1%
        assert fit_smile(m2, np.full(12, 0.2), np.ones(12)) is None

    def test_weights_pull_the_fit_toward_the_heavy_points(self):
        m = np.linspace(-0.05, 0.05, 11)
        iv = np.full(11, 0.20)
        iv[5] = 0.40                                          # one outlier at the money
        light = fit_smile(m, iv, np.ones(11))[0]
        heavy = fit_smile(m, iv, np.where(np.arange(11) == 5, 100.0, 1.0))[0]
        assert heavy > light


class TestUsableRows:
    def _rows(self, **over):
        base = dict(date=date(2025, 3, 3), expiry=date(2025, 3, 5), right="P", strike=495.0, dte=2,
                    close=0.5, volume=100, underlying_close=500.0, iv=0.2, iv_outcome="solved",
                    risk_free_rate=0.05)
        base.update(over)
        return pd.DataFrame([base])

    def test_a_clean_otm_row_is_kept(self):
        assert len(usable_rows(self._rows())) == 1

    @pytest.mark.parametrize("over", [
        dict(dte=0), dict(iv_outcome="failed_arbitrage"), dict(volume=3), dict(close=0.01),
        dict(right="C", strike=495.0),           # ITM call
        dict(right="P", strike=505.0),           # ITM put
        dict(strike=400.0),                      # |m| beyond the cap
        dict(iv=0.001), dict(iv=9.0), dict(underlying_close=0.0),
    ])
    def test_each_exclusion(self, over):
        assert len(usable_rows(self._rows(**over))) == 0

    def test_zero_dte_rows_are_excluded_because_their_close_iv_is_meaningless(self):
        df = pd.concat([self._rows(dte=0), self._rows(dte=1)], ignore_index=True)
        assert list(usable_rows(df)["dte"]) == [1]


# --------------------------------------------------------------------------
# surface lookup
# --------------------------------------------------------------------------

class TestSurface:
    def test_own_expiry_smile_reproduces_the_generating_function(self, surface):
        d = DATES[2]
        for dte, strike in ((1, 497.0), (3, 510.0), (7, 480.0)):
            got = surface.iv(d, d + timedelta(days=dte), strike, S0)
            assert got == pytest.approx(true_iv(dte, math.log(strike / S0)), abs=1e-6)

    def test_a_thin_noisy_print_is_outvoted_by_the_liquid_strikes(self):
        """Volume weighting: one 12-lot print at a silly IV must not drag the smile off
        the liquid quotes around it."""
        d = DATES[0]
        clean = make_bars([d], max_dte=1, vol=5000)
        e = d + timedelta(days=1)
        noisy = clean.copy()
        mask = (noisy["strike"] == 500.0) | (noisy["strike"] == 501.0)
        noisy.loc[mask, "iv"] = 0.95
        noisy.loc[mask, "volume"] = 12
        got = IVSurface.from_bars(noisy).iv(d, e, 500.0, S0)
        want = true_iv(1, 0.0)
        assert abs(got - want) < 0.02, f"the thin print dragged the ATM IV to {got:.4f} (want ~{want:.4f})"
        flat = noisy.copy()
        flat["volume"] = 5000                                    # same prints, equal weight
        assert abs(IVSurface.from_bars(flat).iv(d, e, 500.0, S0) - want) > abs(got - want)

    def test_missing_expiry_interpolates_in_total_variance(self, bars):
        # drop the 4-DTE expiry from one date: it must be rebuilt from 3 and 5
        d = DATES[1]
        e_missing = d + timedelta(days=4)
        thinned = bars[~((bars["date"] == d) & (bars["expiry"] == e_missing))]
        surf = IVSurface.from_bars(thinned)
        assert surf.smile(d, e_missing) is None
        m, strike = 0.0, S0
        got = surf.iv(d, e_missing, strike, S0)
        w3 = true_iv(3, m) ** 2 * 3 / 365.0
        w5 = true_iv(5, m) ** 2 * 5 / 365.0
        want = math.sqrt(((w3 + w5) / 2.0) / (4 / 365.0))
        assert got == pytest.approx(want, rel=1e-6)

    def test_beyond_the_listed_range_is_flat_in_iv(self, surface):
        d = DATES[0]
        last = d + timedelta(days=10)
        far = surface.iv(d, last + timedelta(days=20), S0, S0)
        assert far == pytest.approx(surface.smile(d, last).atm_iv, abs=1e-9)

    def test_an_expiry_before_the_first_listed_is_flat_in_iv(self, bars):
        d = DATES[0]
        no_short = bars[~((bars["date"] == d) & (bars["dte"] <= 2))]
        surf = IVSurface.from_bars(no_short)
        first = d + timedelta(days=3)
        got = surf.iv(d, d + timedelta(days=1), S0, S0)
        assert got == pytest.approx(surf.smile(d, first).atm_iv, abs=1e-9)

    def test_staleness_is_bounded(self, surface):
        last = DATES[-1]
        assert surface.latest_on_or_before(last + timedelta(days=5)) == last
        assert surface.latest_on_or_before(last + timedelta(days=6)) is None
        assert surface.latest_on_or_before(DATES[0] - timedelta(days=1)) is None
        assert surface.iv(last + timedelta(days=30), last + timedelta(days=31), S0, S0) is None

    def test_clamped_outside_the_fitted_moneyness_range(self, surface):
        d = DATES[0]
        e = d + timedelta(days=2)
        f = surface.smile(d, e)
        far = surface.iv(d, e, S0 * 1.5, S0)            # +40% log-moneyness, far off the fit
        assert far == pytest.approx(f.iv_at(f.m_hi + ivs.CLAMP_MARGIN), abs=1e-12)
        assert far < 1.0                                # never an extrapolated blow-up

    def test_a_zero_dte_contract_is_priced_from_yesterdays_one_dte_smile(self, surface):
        """The 0-DTE contract of day D is the 1-DTE contract of D-1 at its close: the
        D-1 smile of that expiry is exactly what was knowable on D's morning."""
        prev = DATES[1]
        expiry_today = prev + timedelta(days=1)         # 1 DTE at `prev`, 0 DTE on `today`
        got = surface.iv(prev, expiry_today, 501.0, S0)
        assert got == pytest.approx(true_iv(1, math.log(501.0 / S0)), abs=1e-6)

    def test_no_lookahead_appending_later_bars_changes_nothing_before_them(self, bars):
        cut = DATES[3]
        early = IVSurface.from_bars(bars[bars["date"] <= cut])
        full = IVSurface.from_bars(bars)
        for dte, strike in ((1, 497.0), (4, 505.0), (9, 490.0)):
            e = cut + timedelta(days=dte)
            assert early.iv(cut, e, strike, S0) == pytest.approx(full.iv(cut, e, strike, S0), abs=1e-12)

    def test_rejects_nonsense_inputs(self, surface):
        assert surface.iv(DATES[0], DATES[0] + timedelta(days=2), 0.0, S0) is None
        assert surface.iv(DATES[0], DATES[0] + timedelta(days=2), 500.0, -1.0) is None


# --------------------------------------------------------------------------
# persistence + read-only loading
# --------------------------------------------------------------------------

def _write_db(path: Path, bars: pd.DataFrame, underlying="QQQ") -> Path:
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE option_daily_bars (underlying TEXT, option_ticker TEXT, expiry TEXT,
        right TEXT, strike REAL, date TEXT, dte INT, open REAL, high REAL, low REAL, close REAL,
        volume INT, vwap REAL, transactions INT, underlying_close REAL, iv REAL, iv_outcome TEXT,
        delta REAL, gamma REAL, theta REAL, vega REAL, risk_free_rate REAL)""")
    for r in bars.itertuples(index=False):
        c.execute("INSERT INTO option_daily_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (underlying, f"O:{underlying}{r.expiry:%y%m%d}{r.right}{int(r.strike)}", r.expiry.isoformat(),
                   r.right, r.strike, r.date.isoformat(), r.dte, None, None, None, r.close, r.volume,
                   None, None, r.underlying_close, r.iv, r.iv_outcome, None, None, None, None,
                   r.risk_free_rate))
    c.commit()
    c.close()
    return path


class TestPersistence:
    def test_load_bars_never_modifies_the_database(self, tmp_path, bars):
        db = _write_db(tmp_path / "h.db", bars)
        before = hashlib.sha256(db.read_bytes()).hexdigest()
        df = ivs.load_bars(db, "QQQ")
        assert len(df) == len(bars) and isinstance(df["date"].iloc[0], date)
        assert hashlib.sha256(db.read_bytes()).hexdigest() == before

    def test_missing_database_is_an_error_not_a_new_file(self, tmp_path):
        ghost = tmp_path / "nope.db"
        with pytest.raises(Exception):
            ivs.load_bars(ghost, "QQQ")
        assert not ghost.exists()

    def test_cache_is_used_until_the_bars_change(self, tmp_path, bars, monkeypatch):
        db = _write_db(tmp_path / "h.db", bars)
        cache = tmp_path / "cache"
        first = ivs.build_or_load(db, "QQQ", cache)
        assert len(first) > 0 and (cache / "iv_surface_QQQ.parquet").exists()

        # second call must NOT rebuild
        def _boom(*a, **k):
            raise AssertionError("rebuilt despite a current cache")
        monkeypatch.setattr(IVSurface, "from_bars", classmethod(lambda cls, *a, **k: _boom()))
        again = ivs.build_or_load(db, "QQQ", cache)
        d = DATES[0]
        e = d + timedelta(days=3)
        assert again.iv(d, e, 505.0, S0) == pytest.approx(first.iv(d, e, 505.0, S0), abs=1e-12)

        # adding a row (the backfill appending) invalidates it
        c = sqlite3.connect(db)
        c.execute("INSERT INTO option_daily_bars (underlying, expiry, right, strike, date, dte, close,"
                  " volume, underlying_close, iv, iv_outcome, risk_free_rate) VALUES "
                  "('QQQ','2025-03-20','P',490,'2025-03-10',10,0.5,100,500,0.2,'solved',0.05)")
        c.commit()
        c.close()
        with pytest.raises(AssertionError, match="rebuilt"):
            ivs.build_or_load(db, "QQQ", cache)

    def test_a_corrupt_cache_is_rebuilt_not_trusted(self, tmp_path, bars):
        db = _write_db(tmp_path / "h.db", bars)
        cache = tmp_path / "cache"
        ivs.build_or_load(db, "QQQ", cache)
        (cache / "iv_surface_QQQ.parquet").write_bytes(b"not a parquet file")
        again = ivs.build_or_load(db, "QQQ", cache)
        assert len(again) > 0


# --------------------------------------------------------------------------
# the engine switch
# --------------------------------------------------------------------------

def _engine(db, **kw):
    from ait.backtesting.engine import Backtester
    idx = pd.bdate_range("2025-02-03", periods=40)
    c = S0 + np.zeros(len(idx))
    df = pd.DataFrame({"Open": c, "High": c * 1.002, "Low": c * 0.998, "Close": c, "Volume": 1e6,
                       "implied_vol": 0.17}, index=idx)
    base = dict(data=df, strategies=["iron_condor"], initial_capital=1_000_000, features_cache=None,
                macro_event_gate=False, allow_live_model_fallback=False, ic_min_credit=0.0,
                ic_min_credit_width=0.0, iv_floor=0.01, friction_model="legacy", db_path=str(db),
                symbol="QQQ", entry_dte=3)
    base.update(kw)
    return Backtester(**base)


class TestEngineSwitch:
    def test_default_is_bs_legacy_and_loads_no_surface(self, tmp_path, bars):
        bt = _engine(_write_db(tmp_path / "h.db", bars))
        assert bt._pricing_source == "bs_legacy" and bt._surface is None

    def test_unknown_source_is_refused(self, tmp_path, bars):
        with pytest.raises(ValueError):
            _engine(_write_db(tmp_path / "h.db", bars), pricing_source="magic")

    def test_surface_mode_prices_the_legs_from_the_expiry_smile(self, tmp_path, bars):
        db = _write_db(tmp_path / "h.db", bars)
        bt = _engine(db, pricing_source="surface")
        assert bt._surface is not None and bt._pricing_source == "surface"
        d = DATES[2]
        expiry = d + timedelta(days=3)
        pos = {"strategy": "iron_condor", "short_call_strike": 510.0, "short_put_strike": 490.0,
               "long_call_strike": 515.0, "long_put_strike": 485.0, "entry_iv": 0.17,
               "entry_date": (d - timedelta(days=0)).isoformat(), "expiry_date": expiry.isoformat()}
        # decision day = d + 1, so the surface date is d
        got = bt._reprice_ic_legs(pos, S0, 1, None)
        t = max(bt._position_dte(pos) - 1, 0) / 365.0
        t = max(t, 0.0001)
        ivs_ = [true_iv(3, math.log(k / S0)) for k in (510.0, 490.0, 515.0, 485.0)]
        want = ic_leg_marks_from_ivs(S0, 510.0, 490.0, 515.0, 485.0, t, 0.05, *ivs_)
        assert got == pytest.approx(want, abs=1e-6)
        legacy = _engine(db, pricing_source="bs_legacy")._reprice_ic_legs(pos, S0, 1, None)
        assert got != pytest.approx(legacy, abs=1e-4)           # and it is NOT the legacy mark

    def test_entry_credit_is_built_from_the_surface_legs(self, tmp_path, bars):
        db = _write_db(tmp_path / "h.db", bars)
        # realistic model: strikes are priced as STORED (legacy prices the unrounded long
        # strikes, W72 lead L6, so a stored-strike recomputation would differ by cents)
        bt = _engine(db, pricing_source="surface", entry_dte=3, delta_short=0.20,
                     engine_model="realistic")
        today = DATES[3]
        expiry = today + timedelta(days=3)
        t = 3 / 365.0
        pos = bt._build_credit_position("iron_condor", S0, 0.17, t, 0.05, 3, today, 1_000_000.0,
                                        expiry=expiry)
        assert pos is not None, "probe condor not built"
        strikes = (pos["short_call_strike"], pos["short_put_strike"],
                   pos["long_call_strike"], pos["long_put_strike"])
        leg_ivs = [true_iv(4, math.log(k / S0)) for k in strikes]       # surface date = today - 1
        marks = ic_leg_marks_from_ivs(S0, *strikes, t, 0.05, *leg_ivs)
        want = (marks[0] + marks[1]) - (marks[2] + marks[3])
        assert pos["mid_credit"] == pytest.approx(round(want, 4), abs=2e-4)
        legacy = _engine(db, pricing_source="bs_legacy", entry_dte=3,
                         engine_model="realistic")._build_credit_position(
            "iron_condor", S0, 0.17, t, 0.05, 3, today, 1_000_000.0, expiry=expiry)
        assert pos["mid_credit"] != pytest.approx(legacy["mid_credit"], abs=1e-3)

    def test_the_iv_floor_applies_to_surface_ivs(self, tmp_path, bars):
        bt = _engine(_write_db(tmp_path / "h.db", bars), pricing_source="surface", iv_floor=0.60)
        d = DATES[2]
        out = bt._surface_leg_ivs(d + timedelta(days=3), (510.0, 490.0, 515.0, 485.0), S0,
                                  d + timedelta(days=1), 0.17)
        assert all(v == pytest.approx(0.60) for v in out)      # the smile is ~0.15-0.25

    def test_a_leg_without_a_smile_falls_back_and_is_counted(self, tmp_path, bars):
        bt = _engine(_write_db(tmp_path / "h.db", bars), pricing_source="surface")
        before = dict(bt._surface_stats)
        far_date = DATES[-1] + timedelta(days=60)               # beyond the staleness bound
        out = bt._surface_leg_ivs(far_date + timedelta(days=2), (510.0, 490.0, 515.0, 485.0),
                                  S0, far_date, 0.17)
        assert bt._surface_stats["miss"] == before["miss"] + 4
        assert out[0] == pytest.approx(bt._get_leg_iv(0.17, 510.0, S0, OptionType.CALL))

    def test_unknown_symbol_degrades_to_legacy_loudly(self, tmp_path, bars, caplog):
        bt = _engine(_write_db(tmp_path / "h.db", bars), pricing_source="surface", symbol="ZZZZ")
        assert bt._surface is None                              # no bars for ZZZZ

    def test_surface_atm_iv_replaces_the_30_day_iv_for_strike_selection(self, tmp_path, bars):
        db = _write_db(tmp_path / "h.db", bars)
        bt = _engine(db, pricing_source="surface")
        d = DATES[3]
        row = pd.Series({"Close": S0, "Open": S0, "High": S0, "Low": S0})
        hist = bt._data.iloc[:5].copy()
        hist.index = pd.DatetimeIndex([pd.Timestamp(x) for x in DATES[:5]])
        bt._get_iv = lambda *a, **k: 0.17                       # the stored 30-day IV
        seen = {}
        orig = bt._build_credit_position

        def spy(strategy, S, iv, t, r, dte, today_date, capital, expiry=None):
            seen["iv"] = iv
            return None
        bt._build_credit_position = spy
        bt._build_position("iron_condor", None, row, hist, d, 1_000_000.0)
        # decision day D, surface date D-1: the entry_dte=3 expiry is 4 days out THAT day
        assert seen["iv"] == pytest.approx(true_iv(4, 0.0), abs=1e-6)
        assert seen["iv"] != pytest.approx(0.17)
        _ = orig


class TestConfig:
    def test_field_default_and_validation(self):
        from ait.config.settings import BacktestConfig
        assert BacktestConfig().pricing_source == "bs_legacy"
        assert BacktestConfig(pricing_source="surface").pricing_source == "surface"
        with pytest.raises(Exception):
            BacktestConfig(pricing_source="vibes")


# --------------------------------------------------------------------------
# the parity replay's 'surface' variant
# --------------------------------------------------------------------------

class TestReplaySurfaceVariant:
    def _setup(self, tmp_path):
        from ait.backtesting import replay as rp
        from tests.test_w72_parity_replay import EXPIRY, ENTRY, P, _condor, _make_db
        db = _make_db(tmp_path / "h.db")
        # the entry is Mon 2026-08-17; the previous SESSION is Fri 08-14 (Sunday has no bars)
        bars = make_bars([date(2026, 8, 12), date(2026, 8, 13), date(2026, 8, 14)],
                         spot=100.0, max_dte=20)
        surf = IVSurface.from_bars(bars)
        return rp, rp.ResearchData(db, "QQQ", prefix="research_"), P, _condor(ENTRY, EXPIRY), surf

    def test_each_leg_is_priced_from_the_previous_sessions_smile(self, tmp_path):
        rp, data, P, trade, surf = self._setup(tmp_path)
        res = rp.entry_replay(trade, data, P, None, surf)
        s = res["surface"]
        assert s is not None and s["iv_src"] == "surface_prev_session"
        for leg, got in zip(trade.legs, s["leg_model_iv"]):
            want = surf.iv(date(2026, 8, 16), leg.expiry, leg.strike, s["S"])
            assert got == pytest.approx(want, abs=1e-12)
        # and the mid credit is Black-Scholes at exactly those IVs, fractional time
        want = 0.0
        for leg, iv, t in zip(trade.legs, s["leg_model_iv"], [s["t_front_years"]] * 4):
            want += leg.sign * black_scholes_price(
                s["S"], leg.strike, t, P.r, iv,
                OptionType.CALL if leg.right == "C" else OptionType.PUT)
        assert s["credit_mid"] == pytest.approx(want, abs=1e-9)

    def test_a_bar_on_the_decision_day_is_never_used(self, tmp_path):
        """No look-ahead: a (richer) smile that exists ON the entry date must be ignored."""
        rp, data, P, trade, surf = self._setup(tmp_path)
        base = rp.entry_replay(trade, data, P, None, surf)["surface"]
        same_day = make_bars([date(2026, 8, 17)], spot=100.0, max_dte=20, iv_shift=0.05)
        with_leak = IVSurface.from_bars(pd.concat(
            [make_bars([date(2026, 8, 12), date(2026, 8, 13), date(2026, 8, 14)], spot=100.0, max_dte=20),
             same_day], ignore_index=True))
        again = rp.entry_replay(trade, data, P, None, with_leak)["surface"]
        assert again["leg_model_iv"] == pytest.approx(base["leg_model_iv"], abs=1e-12)

    def test_it_differs_from_the_legacy_inputs(self, tmp_path):
        rp, data, P, trade, surf = self._setup(tmp_path)
        res = rp.entry_replay(trade, data, P, None, surf)
        assert res["surface"]["credit_mid"] != pytest.approx(res["clean"]["credit_mid"], abs=1e-4)

    def test_no_surface_or_no_coverage_means_no_variant_not_a_guess(self, tmp_path):
        rp, data, P, trade, surf = self._setup(tmp_path)
        assert rp.entry_replay(trade, data, P, None, None)["surface"] is None
        stale = IVSurface.from_bars(make_bars([date(2026, 5, 1)], spot=100.0, max_dte=20))
        assert rp.entry_replay(trade, data, P, None, stale)["surface"] is None

    def test_the_variant_is_part_of_the_banded_verdicts(self, tmp_path):
        rp, *_ = self._setup(tmp_path)
        assert "surface" in rp.ENTRY_VARIANTS and "surface" not in rp.VARIANTS
        rows = [{"band": "0-2", "variant": "surface", "err_pct": 0.01, "eligible": True}] * 3
        out = rp.summarise_bands(rows, [2, 7], error_max=0.15, bias_max=0.10, min_trades=3)
        assert any(v.variant == "surface" and v.band == "0-2" and v.verdict == "PASS" for v in out)

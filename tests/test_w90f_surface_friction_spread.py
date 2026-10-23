"""W90f: the IV surface's acceptance rules, the friction-fit minimums and the spread builders' sizing numbers are settings.

``backtest.surface_tuning`` (min volume 10, min price 0.05, |moneyness| 0.10, 5 points, 1% span, IV bounds 0.03-2.5, clamp margin
0.01, 5 stale days), ``backtest.friction_fit`` (5 samples, 3 sessions, 3 training rows, 3 affine points) and
``backtest.spread_tuning`` (25% of capital, wing = 0.5 expected move) were literals. Research-only; defaults equal the old
literals. The surface is cached on disk, so the fit rules must be part of the cache key: a changed rule rebuilds it.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ait.backtesting import friction as fr
from ait.backtesting import iv_surface as ivs
from ait.backtesting.iv_surface import IVSurface, fit_params, lookup_params, usable_rows
from ait.config.settings import FrictionFitConfig, SpreadTuningConfig, SurfaceTuningConfig, load_settings
from tests.test_w74_engine_realism import _bt
from tests.test_w75_iv_surface import DATES, S0, _write_db, make_bars


def _moved(name: str, **vals):
    s = load_settings()
    return s.model_copy(update={"backtest": s.backtest.model_copy(update={name: getattr(s.backtest, name).model_copy(update=vals)})})


def _freeze(monkeypatch, settings):
    import ait.config.settings as st
    monkeypatch.setattr(st, "_FROZEN", settings)


def test_defaults_equal_the_old_literals_in_schema_and_yaml():
    old = {"surface_tuning": dict(min_volume=10, min_price=0.05, max_abs_moneyness=0.10, min_points=5, min_span=0.01,
                                  iv_lower_bound=0.03, iv_upper_bound=2.5, clamp_margin=0.01, max_stale_days=5),
           "friction_fit": dict(min_samples=5, min_sessions_for_cv=3, min_train_rows=3, min_affine_points=3),
           "spread_tuning": dict(max_cost_fraction_of_capital=0.25, wing_expected_move_fraction=0.5)}
    b = load_settings().backtest
    for block, fields in old.items():
        for name, value in fields.items():
            assert getattr(getattr(b, block), name) == value, (block, name)
    assert {SurfaceTuningConfig().min_volume, FrictionFitConfig().min_samples, SpreadTuningConfig().max_cost_fraction_of_capital} == {10, 5, 0.25}
    assert (ivs.MIN_VOLUME, ivs.MIN_PRICE, ivs.MAX_ABS_M, ivs.MIN_POINTS, ivs.MIN_SPAN, ivs.IV_BOUNDS, ivs.CLAMP_MARGIN,
            ivs.MAX_STALE_DAYS) == (10, 0.05, 0.10, 5, 0.01, (0.03, 2.5), 0.01, 5)       # the importable aliases are the schema defaults


def test_the_iv_bounds_must_be_ordered():
    with pytest.raises(ValueError, match="iv_lower_bound"):
        SurfaceTuningConfig(iv_lower_bound=1.0, iv_upper_bound=0.5)


def _row_kw(params: dict) -> dict:
    """The subset of fit_params that usable_rows takes (the rest, points and span, belong to the smile fit)."""
    return {k: params[k] for k in ("min_volume", "min_price", "max_abs_m", "iv_bounds")}


class TestSurface:
    def test_each_fit_rule_changes_which_bars_are_used(self):
        bars = make_bars(DATES[:1], max_dte=3)
        base = len(usable_rows(bars, **_row_kw(fit_params())))
        assert base > 0
        assert len(usable_rows(bars, **_row_kw(fit_params(SurfaceTuningConfig(min_volume=1000))))) == 0
        assert len(usable_rows(bars, **_row_kw(fit_params(SurfaceTuningConfig(min_price=5.0))))) == 0
        assert 0 < len(usable_rows(bars, **_row_kw(fit_params(SurfaceTuningConfig(max_abs_moneyness=0.02))))) < base
        assert len(usable_rows(bars, **_row_kw(fit_params(SurfaceTuningConfig(iv_lower_bound=0.9, iv_upper_bound=1.0))))) == 0
        assert 0 < len(usable_rows(bars, **_row_kw(fit_params(SurfaceTuningConfig(iv_upper_bound=0.16))))) < base      # the synthetic smile runs 0.14-0.2
        assert 0 < len(usable_rows(bars, **_row_kw(fit_params(SurfaceTuningConfig(iv_lower_bound=0.16))))) < base

    def test_the_point_and_span_minimums(self):
        bars = make_bars(DATES[:1], max_dte=2)
        assert len(IVSurface.from_bars(bars, **fit_params())) > 0
        assert len(IVSurface.from_bars(bars, **fit_params(SurfaceTuningConfig(min_points=500)))) == 0
        assert len(IVSurface.from_bars(bars, **fit_params(SurfaceTuningConfig(min_span=0.9)))) == 0

    def test_clamp_margin_and_staleness_are_lookup_rules(self):
        bars = make_bars(DATES[:1], max_dte=3)
        d, e = DATES[0], DATES[0] + timedelta(days=2)
        tight = IVSurface.from_bars(bars, **fit_params(), **lookup_params(SurfaceTuningConfig(clamp_margin=0.0)))
        wide = IVSurface.from_bars(bars, **fit_params(), **lookup_params(SurfaceTuningConfig(clamp_margin=0.5)))
        far = S0 * 1.6                                                    # well outside the fitted +/-8%
        assert tight.iv(d, e, far, S0) != wide.iv(d, e, far, S0)
        short = IVSurface.from_bars(bars, **fit_params(), **lookup_params(SurfaceTuningConfig(max_stale_days=1)))
        long = IVSurface.from_bars(bars, **fit_params(), **lookup_params(SurfaceTuningConfig(max_stale_days=30)))
        later = d + timedelta(days=10)
        assert short.latest_on_or_before(later) is None and long.latest_on_or_before(later) == d
        assert short.latest_on_or_before(later, 30) == d                  # an explicit argument still wins

    def test_the_clamp_margin_reaches_interpolation_and_edge_extrapolation(self):
        """Query an expiry BETWEEN two fitted ones, and one beyond the last, at a strike far outside the fitted range."""
        d = DATES[0]
        bars = make_bars([d], max_dte=3)
        bars = bars[bars["dte"].isin([1, 3])]                      # fitted expiries d+1 and d+3 only
        far = S0 * 1.5
        between, beyond = d + timedelta(days=2), d + timedelta(days=9)

        def surf(margin):
            return IVSurface.from_bars(bars, **fit_params(), **lookup_params(SurfaceTuningConfig(clamp_margin=margin)))
        tight, wide = surf(0.0), surf(0.4)
        assert tight.iv(d, between, far, S0) != wide.iv(d, between, far, S0)       # linear-in-variance interpolation
        assert tight.iv(d, beyond, far, S0) != wide.iv(d, beyond, far, S0)         # flat extrapolation from the edge smile
        # the interpolated value is exactly the total-variance blend of the two smiles evaluated WITH the margin
        import math
        lo, hi = wide.smile(d, d + timedelta(days=1)), wide.smile(d, d + timedelta(days=3))
        m = math.log(far / S0)
        t_q, t_lo, t_hi = 2 / 365.0, 1 / 365.0, 3 / 365.0
        w_lo, w_hi = lo.iv_at(m, 0.4) ** 2 * t_lo, hi.iv_at(m, 0.4) ** 2 * t_hi
        want = math.sqrt(max((w_lo + 0.5 * (w_hi - w_lo)) / t_q, 1e-8))
        assert wide.iv(d, between, far, S0) == pytest.approx(want, rel=1e-12)

    def test_the_lookup_rules_survive_an_empty_fit_and_a_cache_hit(self, tmp_path):
        d = DATES[0]
        empty = IVSurface.from_bars(make_bars([d], max_dte=2), **fit_params(SurfaceTuningConfig(min_volume=100000)),
                                    **lookup_params(SurfaceTuningConfig(max_stale_days=17)))
        assert len(empty) == 0 and empty._max_stale_days == 17
        db = _write_db(tmp_path / "h.db", make_bars(DATES))
        ivs.build_or_load(db, "QQQ", tmp_path / "c")
        hit = ivs.build_or_load(db, "QQQ", tmp_path / "c", max_stale_days=9, clamp_margin=0.2)      # served from the cache
        assert (hit._max_stale_days, hit._clamp_margin) == (9, 0.2)

    def test_a_changed_fit_rule_rebuilds_the_cached_surface_but_a_lookup_rule_does_not(self, tmp_path, monkeypatch):
        db = _write_db(tmp_path / "h.db", make_bars(DATES))
        cache = tmp_path / "cache"
        first = ivs.build_or_load(db, "QQQ", cache)
        built = []
        real = IVSurface.from_bars.__func__

        def spy(cls, bars, **kw):
            built.append(kw)
            return real(cls, bars, **kw)
        monkeypatch.setattr(IVSurface, "from_bars", classmethod(spy))
        ivs.build_or_load(db, "QQQ", cache)
        assert built == []                                               # same rules: served from the cache
        ivs.build_or_load(db, "QQQ", cache, clamp_margin=0.3, max_stale_days=9)
        assert built == []                                               # lookup rules do not touch the fit
        ivs.build_or_load(db, "QQQ", cache, min_volume=500)
        assert len(built) == 1 and built[0]["min_volume"] == 500          # a fit rule rebuilt it
        assert len(first) > 0

    def test_build_or_load_defaults_to_the_loaded_settings(self, tmp_path, monkeypatch):
        db = _write_db(tmp_path / "h.db", make_bars(DATES))
        _freeze(monkeypatch, _moved("surface_tuning", min_volume=100000))
        assert len(ivs.build_or_load(db, "QQQ", tmp_path / "c1")) == 0    # nothing trades 100000 lots: the setting reached the fit
        _freeze(monkeypatch, load_settings(ignore_frozen=True))
        assert len(ivs.build_or_load(db, "QQQ", tmp_path / "c2")) > 0

    def test_the_engine_passes_its_own_surface_settings(self, tmp_path, monkeypatch):
        import ait.backtesting.iv_surface as mod
        seen = {}
        monkeypatch.setattr(mod, "build_or_load", lambda db, sym, cache, **kw: seen.update(kw) or IVSurface([]))
        bt = _bt(symbol="QQQ", settings=_moved("surface_tuning", min_volume=77, clamp_margin=0.2, max_stale_days=9))
        bt._load_surface()
        assert (seen["min_volume"], seen["clamp_margin"], seen["max_stale_days"]) == (77, 0.2, 9)
        assert seen["iv_bounds"] == [0.03, 2.5]


class TestFrictionFit:
    def _data(self, n=40, sessions=5, seed=1):
        rng = np.random.default_rng(seed)
        mid = rng.uniform(0.2, 4.0, n)
        hs = 0.02 + 0.01 * mid + rng.normal(0, 0.002, n)
        return mid, np.abs(hs), np.repeat(np.arange(sessions), n // sessions)

    def test_minimum_samples(self):
        mid, hs, sess = self._data(8, 2)
        assert fr.fit_leg_spread_models(mid, hs, sess)["n"] == 8
        with pytest.raises(ValueError, match=">= 20 usable"):
            fr.fit_leg_spread_models(mid, hs, sess, FrictionFitConfig(min_samples=20))
        with pytest.raises(ValueError, match=">= 5 usable"):
            fr.fit_leg_spread_models(mid[:4], hs[:4], sess[:4])

    def test_sessions_needed_for_grouped_cv(self):
        mid, hs, sess = self._data(40, 5)
        assert fr.fit_leg_spread_models(mid, hs, sess)["cv"] == "grouped"
        assert fr.fit_leg_spread_models(mid, hs, sess, FrictionFitConfig(min_sessions_for_cv=6))["cv"] == "insample"

    def test_training_rows_per_fold(self):
        mid, hs, sess = self._data(40, 5)                        # 8 rows per session: 32 training rows per fold
        ok = fr.fit_leg_spread_models(mid, hs, sess)
        assert ok["cv"] == "grouped"
        with pytest.raises(Exception):                           # every fold skipped -> nothing to average
            fr.fit_leg_spread_models(mid, hs, sess, FrictionFitConfig(min_train_rows=100))

    def test_the_affine_floor_reaches_every_fit_site(self, monkeypatch):
        rng = np.random.default_rng(5)
        mid = rng.uniform(0.2, 4.0, 12)
        hs = 0.02 + 0.02 * mid
        sess = np.repeat(np.arange(3), 4)                      # 3 sessions of 4 rows: 8 training rows per fold, 12 in sample
        grouped_default = fr.fit_leg_spread_models(mid, hs, sess)["cv_mae"]
        assert grouped_default["affine_usd"] < grouped_default["const_usd"]          # a real slope: affine wins when it may fit
        cfg = FrictionFitConfig(min_affine_points=9)              # 8-row folds can no longer fit the affine form
        grouped = fr.fit_leg_spread_models(mid, hs, sess, cfg)["cv_mae"]
        assert grouped["affine_usd"] == pytest.approx(grouped["const_usd"])           # ... so it degrades to the constant in every fold
        insample = fr.fit_leg_spread_models(mid, hs, sess, FrictionFitConfig(min_affine_points=13, min_sessions_for_cv=4))
        assert insample["cv"] == "insample" and insample["cv_mae"]["affine_usd"] == pytest.approx(insample["cv_mae"]["const_usd"])
        # the default floor of _fit_one is the LOADED setting
        _freeze(monkeypatch, _moved("friction_fit", min_affine_points=5))
        assert fr._fit_one("affine_usd", np.array([0.5, 1.0, 2.0, 3.0]), np.array([0.025, 0.03, 0.04, 0.05])).b == 0.0

    def test_affine_points(self):
        mid = np.array([0.5, 1.0, 2.0, 3.0])
        hs = 0.02 + 0.01 * mid
        assert fr._fit_one("affine_usd", mid, hs).b > 0                                    # 4 points >= the default 3
        assert fr._fit_one("affine_usd", mid, hs, 5).b == 0.0                              # below a floor of 5: a constant
        assert fr.loaded_friction_fit().min_affine_points == 3


class TestSpreadBuilders:
    S, IV, DTE = 100.0, 0.20, 21

    def _build(self, strategy="put_credit_spread", **spread):
        s = load_settings()
        settings = s.model_copy(update={"backtest": s.backtest.model_copy(update={
            "spread_tuning": s.backtest.spread_tuning.model_copy(update=spread)})})
        # a 1% budget buys no contract of a credit spread (the cap path); the jade lizard is sized on naked margin, so it needs more
        bt = _bt(settings=settings, position_size_pct=0.5 if strategy == "jade_lizard" else 0.01, initial_capital=5000.0)
        t = self.DTE / 365.0
        return bt._build_credit_position(strategy, self.S, self.IV, t, 0.05, self.DTE, date(2024, 3, 4), 5000.0)

    @pytest.mark.parametrize("strategy", ["put_credit_spread", "call_credit_spread"])
    def test_the_capital_cap_decides_whether_a_too_big_spread_is_opened_at_one_contract(self, strategy):
        pos = self._build(strategy)
        assert pos is not None and pos["contracts"] == 1           # the 1% budget buys 0 contracts; the 25% cap lets 1 in
        assert self._build(strategy, max_cost_fraction_of_capital=0.01) is None

    @pytest.mark.parametrize("strategy,key", [("put_credit_spread", "long_put_strike"), ("call_credit_spread", "long_call_strike"),
                                              ("jade_lizard", "long_call_strike")])
    def test_the_wing_fraction_sets_the_wing_width(self, strategy, key):
        base = self._build(strategy, max_cost_fraction_of_capital=1.0)
        wide = self._build(strategy, max_cost_fraction_of_capital=1.0, wing_expected_move_fraction=1.5)
        assert base is not None and wide is not None
        shorts = {"long_put_strike": "short_put_strike", "long_call_strike": "short_call_strike"}
        assert abs(wide[key] - wide[shorts[key]]) > abs(base[key] - base[shorts[key]])

    def test_the_debit_structures_use_the_same_cap(self):
        from ait.strategies.base import SignalDirection
        s = load_settings()

        def build(strategy, **spread):
            settings = s.model_copy(update={"backtest": s.backtest.model_copy(update={
                "spread_tuning": s.backtest.spread_tuning.model_copy(update=spread)})})
            bt = _bt(settings=settings, position_size_pct=0.001, initial_capital=5000.0)
            return bt._build_debit_position(strategy, SignalDirection.BULLISH, self.S, self.IV, self.DTE / 365.0, 0.05,
                                            self.DTE, date(2024, 3, 4), 5000.0)
        for strategy in ("long_strangle", "bull_call_spread"):
            assert build(strategy) is not None, strategy
            assert build(strategy, max_cost_fraction_of_capital=0.001) is None, strategy

"""W83a-2: the math of the strike-aware range study (ait.ml.excursion) and the study's features.

Every function the registered rule depends on, on hand-built inputs with known answers: the
excursion target, the barrier baseline (against a Monte Carlo), HAR (recovering a known process),
the quantile CDF, Brier / BSS / CRPS / calibration, the stationary bootstrap, the folds' purge, the
verdict and the outcome; plus the study's look-ahead guard (data from D onward never moves D's
features) and the guards that keep this research module out of the live tree.
"""
from __future__ import annotations

import ast
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.ml import excursion as ex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _bars(closes, highs=None, lows=None, opens=None, start="2024-01-02"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = np.asarray(closes, float)
    return pd.DataFrame({"Open": opens if opens is not None else c, "High": highs if highs is not None else c,
                         "Low": lows if lows is not None else c, "Close": c}, index=idx)


class TestTarget:
    def test_excursion_from_the_previous_close_with_high_and_low(self):
        df = _bars([100, 101, 99, 100, 100], highs=[100, 103, 99.5, 108, 100], lows=[100, 100.5, 96, 90, 100])
        e = ex.excursions(df, h=2)
        # row 1 (D = 2nd bar): S = 100; window bars 1-2: max high 103, min low 96
        assert e["up"].iloc[1] == pytest.approx(0.03) and e["down"].iloc[1] == pytest.approx(0.04)
        assert e["m_frac"].iloc[1] == pytest.approx(0.04)
        assert e["window_end"].iloc[1] == df.index[2]          # the 108 / 90 bar after the window is not in it

    def test_an_overnight_gap_counts(self):
        """The D-1 close is the reference, so a gap at D's open is part of the excursion."""
        df = _bars([100, 95, 95], highs=[100, 95, 95], lows=[100, 94, 95], opens=[100, 95, 95])
        assert ex.excursions(df, h=1)["down"].iloc[1] == pytest.approx(0.06)

    def test_incomplete_windows_and_the_first_row_are_missing(self):
        e = ex.excursions(_bars([100, 101, 102, 103]), h=3)
        assert e["m_frac"].isna().tolist() == [True, False, True, True]
        assert pd.isna(e["window_end"].iloc[0]) and pd.isna(e["window_end"].iloc[3])

    def test_implied_move_and_z(self):
        assert ex.implied_move(np.array([0.20]), 9)[0] == pytest.approx(0.20 * math.sqrt(9 / 252))
        assert ex.z_of_delta(0.20) == pytest.approx(0.8416, abs=1e-4)
        assert ex.z_of_delta(0.10) > ex.z_of_delta(0.20) > ex.z_of_delta(0.30)


class TestBarrier:
    def test_matches_a_monte_carlo(self):
        rng = np.random.default_rng(83)
        steps, n = 2000, 20000
        paths = np.cumsum(rng.normal(0, 1 / math.sqrt(steps), (n, steps)), axis=1)
        mx = np.abs(paths).max(axis=1)
        for z in (0.8, 1.0, 1.5, 2.0):
            mc = float(np.mean(mx < z))
            # discrete monitoring misses some crossings: the MC sits slightly ABOVE the continuous value
            assert ex.barrier_no_touch(z)[0] == pytest.approx(mc, abs=0.025)
            assert ex.barrier_no_touch(z)[0] <= mc + 0.005

    def test_limits_and_monotonicity(self):
        p = ex.barrier_no_touch(np.array([0.3, 0.8, 1.2, 2.0, 4.0]))
        assert np.all(np.diff(p) > 0) and p[0] < 0.01 and p[-1] > 0.999


class TestHar:
    def test_recovers_a_known_process(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(4000, 3))
        beta = np.array([-6.0, 0.3, 0.4, 0.2])
        y = beta[0] + X @ beta[1:] + rng.normal(0, 0.1, 4000)
        b, smear = ex.har_fit(y, X)
        assert b == pytest.approx(beta, abs=0.02)
        assert smear == pytest.approx(math.exp(0.1 ** 2 / 2), abs=0.002)      # E[exp(e)] for N(0, 0.1)
        assert ex.har_predict_sigma(b, smear, X[:1])[0] == pytest.approx(
            math.sqrt(math.exp(b[0] + X[0] @ b[1:]) * smear))

    def test_garman_klass_and_the_lags_use_only_the_past(self):
        df = _bars([100, 101, 102, 103, 104, 105], highs=[101] * 6, lows=[99] * 6, opens=[100] * 6)
        gk = ex.garman_klass(df)
        assert gk.iloc[0] == pytest.approx(0.5 * math.log(101 / 99) ** 2)
        bar = ex.garman_klass(_bars([103.0], highs=[104.0], lows=[99.0], opens=[100.0]))
        assert bar.iloc[0] == pytest.approx(0.5 * math.log(104 / 99) ** 2 - (2 * math.log(2) - 1) * math.log(103 / 100) ** 2)
        lags = ex.har_lags(gk)
        assert math.isnan(lags["lag1"].iloc[0]) and lags["lag1"].iloc[1] == pytest.approx(math.log(gk.iloc[0]))
        t = ex.har_target(gk, 2)
        assert t.iloc[0] == pytest.approx(math.log(gk.iloc[0] + gk.iloc[1])) and math.isnan(t.iloc[-1])


class TestProbabilities:
    def test_empirical_cdf(self):
        f = ex.empirical_cdf(np.array([1.0, 2.0, 3.0, 4.0, np.nan]))
        assert f(np.array([0.5, 2.0, 2.5, 9.0])).tolist() == [0.0, 0.5, 0.5, 1.0]

    def test_cdf_from_quantiles_inverts_a_known_distribution(self):
        taus = np.array(ex.TAUS)
        q = norm_ppf_row = np.log(1.0) + 0.5 * np.array([__import__("scipy").stats.norm.ppf(t) for t in taus])
        z = math.exp(0.5 * 0.5244)           # log z = 0.5 * Phi^-1(0.70)
        assert ex.cdf_from_quantiles(norm_ppf_row[None, :], z)[0] == pytest.approx(0.70, abs=0.01)
        assert q.shape == (19,)

    def test_monotone_sort_and_the_clip(self):
        q = np.log(np.array([[1.0, 0.9, 1.1] + [1.2] * 16]))     # crossed quantiles are sorted
        assert ex.cdf_from_quantiles(q, 0.1)[0] == ex.CDF_CLIP[0]
        assert ex.cdf_from_quantiles(q, 50.0)[0] == ex.CDF_CLIP[1]
        assert 0.025 < ex.cdf_from_quantiles(q, 1.0)[0] < 0.975
        crossed = np.log(np.array([[0.5, 0.7, 0.6, 0.9, 0.8] + list(np.linspace(1.0, 2.0, 14))]))
        for z in (0.55, 0.65, 0.75, 0.85, 1.5):
            assert ex.cdf_from_quantiles(crossed, z)[0] == ex.cdf_from_quantiles(np.sort(crossed, axis=1), z)[0]


class TestScores:
    def test_brier_and_bss(self):
        y = np.array([1, 0, 1, 1])
        assert ex.brier(np.array([1, 0, 1, 1]), y) == 0.0
        assert ex.brier(np.full(4, 0.5), y) == pytest.approx(0.25)
        assert ex.bss(np.full(4, 0.75), np.full(4, 0.5), y) == pytest.approx(1 - ex.brier(np.full(4, 0.75), y) / 0.25)

    def test_crps_pinball(self):
        q = np.zeros((1, 19))
        assert ex.crps_pinball(q, np.array([0.0])) == 0.0
        # y above every quantile by 1: pinball = tau for each, mean tau = 0.5 -> CRPS = 1.0
        assert ex.crps_pinball(q, np.array([1.0])) == pytest.approx(1.0)

    def test_calibration_slope(self):
        rng = np.random.default_rng(7)
        p = rng.uniform(0.05, 0.95, 40000)
        y = (rng.random(40000) < p).astype(float)
        assert ex.calibration_slope(p, y) == pytest.approx(1.0, abs=0.05)
        logit = np.log(p / (1 - p))
        over = 1 / (1 + np.exp(-2 * logit))                      # twice as confident as the truth
        assert ex.calibration_slope(over, y) == pytest.approx(0.5, abs=0.05)

    def test_stationary_bootstrap(self):
        idx = ex.stationary_bootstrap_indices(5000, mean_block=20, n_boot=40, seed=83)
        assert idx.shape == (40, 5000) and idx.min() >= 0 and idx.max() < 5000
        runs = (np.diff(idx, axis=1) != 1).sum(axis=1) + 1        # blocks per resample
        assert 5000 / runs.mean() == pytest.approx(20, rel=0.15)
        again = ex.stationary_bootstrap_indices(5000, mean_block=20, n_boot=40, seed=83)
        assert np.array_equal(idx, again)                       # the same draws for every predictor


class TestFolds:
    def test_purge_and_test_rows(self):
        sessions = list(pd.bdate_range("2018-11-01", "2020-03-31").date)
        h = 9
        rows = sessions[:-h]
        ends = [sessions[i + h - 1] for i in range(len(rows))]
        folds = ex.yearly_folds(rows, ends, sessions, [2019, 2020], purge=h)
        y, train, test = folds[0]
        first = sessions.index(next(s for s in sessions if s.year == 2019))
        assert y == 2019 and test.sum() == sum(1 for d in rows if d.year == 2019)
        assert max(e for e, t in zip(ends, train) if t) < sessions[first - h]
        assert not any(t and d.year >= 2019 for d, t in zip(rows, train))

    def test_incomplete_windows_never_test(self):
        sessions = list(pd.bdate_range("2019-12-02", "2020-01-31").date)
        ends = [s if i < 30 else None for i, s in enumerate(sessions)]
        _y, _tr, test = ex.yearly_folds(sessions, ends, sessions, [2020], purge=5)[0]
        assert not any(t and e is None for t, e in zip(test, ends))


class TestVerdict:
    @staticmethod
    def _m(c=(0.03, 0.01, 1.0), cl=(0.0, -0.01, 1.0)):
        return {"C": dict(zip(("bss", "lo", "slope"), c)), "C-lin": dict(zip(("bss", "lo", "slope"), cl))}

    def test_pass(self):
        v = ex.verdict(self._m())
        assert v["forwarded"] == "C" and v["pass"]

    @pytest.mark.parametrize("c, why", [((0.019, 0.005, 1.0), "bss>=0.02"), ((0.03, 0.0, 1.0), "lo>0"),
                                        ((0.03, 0.01, 0.79), "slope_in_band"), ((0.03, 0.01, 1.21), "slope_in_band")])
    def test_each_condition_fails_alone(self, c, why):
        v = ex.verdict(self._m(c=c))
        assert not v["pass"] and not v["checks"][why]

    def test_the_clin_tie_break(self):
        v = ex.verdict(self._m(c=(0.03, 0.01, 1.0), cl=(0.021, 0.004, 1.0)))
        assert v["forwarded"] == "C-lin" and v["pass"]          # within 0.01 -> the simpler model
        v = ex.verdict(self._m(c=(0.04, 0.01, 1.0), cl=(0.029, 0.004, 1.0)))
        assert v["forwarded"] == "C"

    def test_outcome_is_qqq_primary(self):
        p, f = {"pass": True}, {"pass": False}
        assert ex.outcome({"QQQ": p, "SPY": f, "IWM": f}).startswith("PASS: W83b may wire QQQ only")
        assert ex.outcome({"QQQ": p, "SPY": p, "IWM": f}) == "PASS: W83b may wire QQQ and SPY"
        assert ex.outcome({"QQQ": f, "SPY": p, "IWM": f}).startswith("MIXED")
        assert ex.outcome({"QQQ": f, "SPY": f, "IWM": f}).startswith("FAIL")


# --------------------------------------------------------------------------
# the study's features: no look-ahead
# --------------------------------------------------------------------------

def _synthetic_inputs(n=700, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = c * (1 + rng.normal(0, 0.002, n))
    f = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.004, "Low": np.minimum(o, c) * 0.996,
                      "Close": c, "Volume": 1e6, "implied_vol": 0.2 + 0.05 * np.sin(np.arange(n) / 30)}, index=idx)
    vix = pd.DataFrame({"VIX": 0.2, "VIX3M": 0.21, "VIX9D": 0.19}, index=idx)
    econ = pd.DataFrame({"event_date": [str(d.date()) for d in idx[::21]], "event": ["CPI"] * len(idx[::21])})
    econ = pd.concat([econ, pd.DataFrame({"event_date": [str(d.date()) for d in idx[5::42]], "event": "FOMC"}),
                      pd.DataFrame({"event_date": [str(d.date()) for d in idx[3::21]], "event": "NFP"})])
    holdings = pd.DataFrame({"filed_date": [str(idx[100].date())] * 3 + [str(idx[400].date())] * 3,
                             "company_key": ["AAPL", "MSFT", "AMZN"] * 2, "weight": [0.12, 0.08, 0.05, 0.10, 0.09, 0.06]})
    earnings = pd.DataFrame({"company_key": ["AAPL", "MSFT", "AMZN"] * 10,
                             "event_session": [str(d.date()) for d in idx[30::63][:10] for _ in range(3)]})
    divs = pd.DataFrame({"ex_date": [str(d.date()) for d in idx[10::63]],
                         "amount": [0.40 + 0.01 * k for k in range(len(idx[10::63]))]})
    return f, {"vix": vix, "econ": econ, "holdings": holdings, "earnings": earnings, "dividends": {"QQQ": divs}}


class TestStudyFeatures:
    def test_changing_data_from_d_onward_never_moves_the_features_of_d(self):
        import excursion_range_study as st
        f, inputs = _synthetic_inputs()
        fr = pd.DataFrame({"mf_width": 0.3, "hurst_spread": 0.1}, index=f.index)   # fractal: own test below
        base = st.build_rows("QQQ", f, inputs, h=9, fractal=fr)
        cut = f.index[500]
        g = f.copy()
        g.loc[cut:, ["Open", "High", "Low", "Close"]] *= 1.37
        g.loc[cut:, "implied_vol"] = 0.9
        moved = st.build_rows("QQQ", g, inputs, h=9, fractal=fr)
        feats = st.FEATURES
        a, b = base.loc[:cut, feats], moved.loc[:cut, feats]          # rows up to and including D = cut
        pd.testing.assert_frame_equal(a, b)
        # ...while the TARGET of rows whose window reaches the change does move (the guard is not vacuous)
        assert not np.allclose(base.loc[f.index[495]:cut, "m_frac"], moved.loc[f.index[495]:cut, "m_frac"])

    def test_the_fractal_features_use_only_the_past(self):
        import excursion_range_study as st
        f, _ = _synthetic_inputs(n=600)
        a = st.fractal_columns(f, "X", cache=None)
        g = f.copy()
        g.loc[f.index[560]:, "Close"] *= 1.5
        b = st.fractal_columns(g, "X", cache=None)
        pd.testing.assert_frame_equal(a.iloc[:560], b.iloc[:560])
        assert a["mf_width"].iloc[:504].isna().all() and a["mf_width"].iloc[504:].notna().all()

    def test_the_feature_list_is_the_registered_one(self):
        import excursion_range_study as st
        assert len(st.FEATURES) == 24 and len(set(st.FEATURES)) == 24
        assert st.LGB_PARAMS["n_estimators"] == 300 and st.LGB_PARAMS["random_state"] == 83
        assert st.TEST_YEARS == tuple(range(2019, 2027))

    def test_window_features_and_weights_filed_before_d(self):
        import excursion_range_study as st
        f, inputs = _synthetic_inputs()
        fr = pd.DataFrame({"mf_width": 0.3, "hurst_spread": 0.1}, index=f.index)
        r = st.build_rows("QQQ", f, inputs, h=9, fractal=fr)
        cpi_days = set(pd.to_datetime(inputs["econ"].query("event == 'CPI'")["event_date"]))
        d = f.index[42]                                   # a CPI day: in its own window
        assert d in cpi_days and r.loc[d, "cpi_in"] == 1.0
        assert r.loc[f.index[43], "cpi_in"] == 0.0         # next CPI is 21 sessions later (> h-1)
        assert r.loc[f.index[55], "cpi_in"] == 1.0 and r.loc[f.index[55], "to_next_macro"] >= 0
        assert r.loc[f.index[54], "cpi_in"] == 0.0     # the CPI at session 63 is D+9 = D+h: outside
        # weights become known only AFTER their filing date: on the first report's filing day the
        # earnings features are missing, the next session they exist
        first = f.index[100]
        nxt = f.index[101]
        assert np.isnan(r.loc[first, "earn_w_in"]) and not np.isnan(r.loc[nxt, "earn_w_in"])
        e_days = sorted(set(pd.to_datetime(inputs["earnings"]["event_session"])))
        d_e = e_days[2]
        i = f.index.get_loc(d_e)
        assert r.loc[f.index[i - 8], "earn_w_in"] > 0 and r.loc[f.index[i - 9], "earn_w_in"] == 0
        # the dividend yield uses the payment BEFORE D: on an ex-date it is still the previous one
        ex_d = pd.to_datetime(inputs["dividends"]["QQQ"]["ex_date"]); amt = inputs["dividends"]["QQQ"]["amount"]
        d2 = ex_d.iloc[3]
        assert r.loc[d2, "prev_div_yield"] == pytest.approx(amt.iloc[2] / f["Close"].shift(1).loc[d2])
        # the target in implied-move units
        k = f.index[300]
        assert r.loc[k, "M"] == pytest.approx(r.loc[k, "m_frac"] / (f["implied_vol"].shift(1).loc[k] * math.sqrt(9 / 252)))


class TestStudyScoring:
    def test_live_eligible_follows_the_w84_live_rule(self):
        import excursion_range_study as st
        from ait.backtesting.ex_div import ex_div_blocks
        from ait.backtesting.expiry_calendar import listed_expiry
        f, inputs = _synthetic_inputs()
        fr = pd.DataFrame({"mf_width": 0.3, "hurst_spread": 0.1}, index=f.index)
        r = st.build_rows("QQQ", f, inputs, h=9, fractal=fr)
        ex_dates = sorted(pd.to_datetime(inputs["dividends"]["QQQ"]["ex_date"]).dt.date)
        expect = [not ex_div_blocks("live", ex_dates, d, listed_expiry(d, 14))[0] for d in r.index.date]
        assert r["live_ok"].tolist() == expect
        assert 0 < (~r["live_ok"]).sum() < len(r)                 # the scenario has blocked days

    def test_score_wires_the_registered_metrics(self):
        import excursion_range_study as st
        rng = np.random.default_rng(3)
        n = 600
        y = (rng.random((n, 5)) < 0.8).astype(int)
        ref = np.full((n, 5), 0.8)
        good = np.clip(0.8 + 0.15 * (y - 0.8) + rng.normal(0, 0.02, (n, 5)), 0.01, 0.99)
        rows = {"date": [str(d.date()) for d in pd.bdate_range("2019-01-02", periods=n)],
                "live_ok": [True] * n, "y": y.tolist(), "logM": rng.normal(size=n).tolist(),
                "A": ref.tolist(), "A'": ref.tolist(), "B": ref.tolist(), "ref": ref.tolist(),
                "C": good.tolist(), "C-lin": ref.tolist(), "C|own": good.tolist(), "C|pooled": good.tolist(),
                "C-lin|own": ref.tolist(), "C-lin|pooled": ref.tolist(),
                "C|q": np.zeros((n, 19)).tolist(), "C-lin|q": np.zeros((n, 19)).tolist()}
        m = st.score(rows, n_boot=300)
        per = [1 - ex.brier(good[:, k], y[:, k]) / ex.brier(ref[:, k], y[:, k]) for k in range(5)]
        assert m["C"]["bss"] == pytest.approx(np.mean(per))
        idx = ex.stationary_bootstrap_indices(n, 20.0, 300, st.SEED)
        boot = [np.mean(1 - ((good[b] - y[b]) ** 2).mean(0) / ((ref[b] - y[b]) ** 2).mean(0)) for b in idx]
        assert m["C"]["lo"] == pytest.approx(np.percentile(boot, 2.5))
        assert m["C-lin"]["bss"] == pytest.approx(0.0) and m["verdict"]["forwarded"] == "C"
        assert m["verdict"] == ex.verdict({"C": m["C"], "C-lin": m["C-lin"]})
        assert m["C"]["slope"] == pytest.approx(ex.calibration_slope(good[:, 2], y[:, 2]))


def test_the_module_stays_research_only_and_pure():
    src = (ROOT / "src" / "ait" / "ml" / "excursion.py").read_text()
    tree = ast.parse(src)
    mods = {(n.module or "").split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert mods <= {"__future__", "math", "collections", "datetime", "numpy", "pandas", "scipy"}
    hits = []
    for p in (ROOT / "src" / "ait").rglob("*.py"):
        rel = p.relative_to(ROOT / "src" / "ait").as_posix()
        if rel.startswith(("backtesting/", "ml/excursion")):
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("excursion") or (
                    isinstance(n, ast.ImportFrom) and any(a.name == "excursion" for a in n.names)):
                hits.append(f"{rel}:{n.lineno}")
    assert hits == []

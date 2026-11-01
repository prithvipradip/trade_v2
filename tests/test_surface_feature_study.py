"""W118: the surface-feature study's pure logic (labels, folds, bootstrap, Holm, criteria, guards).

No data store: panels, bars and surfaces are fabricated, so every claim is checkable against the generating
function. The rule is pre-registered in PLAN.md ("W118 AS PLANNED" and "W118 ADDENDUM 1"); these tests pin that the
script implements that rule, so the verdict of the real run cannot be bent after seeing it.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("surface_feature_study", ROOT / "scripts" / "surface_feature_study.py")
sf = importlib.util.module_from_spec(_spec)
sys.modules["surface_feature_study"] = sf
_spec.loader.exec_module(sf)

from ait.backtesting.friction import BandedFriction, LegSpreadModel  # noqa: E402


def business_days(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


NYSE = business_days(date(2024, 9, 1), date(2027, 3, 31))


# --------------------------------------------------------------------------- the registered constants

def test_registered_constants_match_plan_md():
    assert sf.TARGET_DTES == (1, 2, 5, 7, 10, 14, 18, 25, 30)                 # ADDENDUM 1: 45 dropped
    assert sf.BUCKETS == {"A": (1, 2), "B": (3, 10), "C": (14, 30)}
    assert sf.SHORT_DELTA == 0.20 and sf.WING_EM_FRACTION == 0.35 and sf.RR_DELTA == 0.25
    assert sf.GAP_SESSIONS == 5 and sf.MIN_N_EFF == 30 and sf.ALPHA == 0.10
    assert sf.BOOT_N == 2000 and sf.BOOT_SEED == 20261101 and sf.STRESS == 2.0
    assert sf.WINDOW_START == date(2024, 10, 1) and sf.WINDOW_END == date(2026, 10, 20)
    assert len(sf.FOLDS) == 6 and sf.FOLDS[0][0] == date(2025, 4, 1) and sf.FOLDS[-1][1] == sf.WINDOW_END
    assert sf.PRIMARY == ("vrp_h", "term_slope", "rr25_norm") and sf.EXPECTED_SIGN == {"vrp_h": +1}
    assert sf.STRUCTURAL_REASONS == ("no_expiry", "collapsed_duplicate_expiry")
    assert sf.CANARY_SHUFFLES == 20 and sf.CANARY_SHUFFLE_MAX_HITS == 6 and sf.CANARY_LEAK_P == 0.01


def test_folds_are_contiguous_and_cover_the_test_period():
    for (lo, hi), (nlo, _) in zip(sf.FOLDS, sf.FOLDS[1:]):
        assert hi < nlo and (nlo - hi).days <= 3
    assert sf.fold_of(date(2025, 3, 31)) is None                                 # warm-up is never a test fold
    assert sf.fold_of(date(2025, 4, 1)) == 0 and sf.fold_of(date(2026, 10, 20)) == 5


def test_every_target_has_a_bucket_and_the_buckets_do_not_overlap():
    assert {sf.bucket_of(t) for t in sf.TARGET_DTES} == {"A", "B", "C"}
    assert [sf.bucket_of(t) for t in (1, 2, 5, 7, 10, 14, 18, 25, 30)] == list("AABBBCCCC")
    with pytest.raises(ValueError):
        sf.bucket_of(45)


# --------------------------------------------------------------------------- n_eff counts sessions, not rows

def test_n_eff_counts_distinct_sessions_not_rows():
    # three rows per session (three targets), 100 sessions, mean hold 10 sessions: 100 / 10, not 300 / 10
    d0 = date(2025, 4, 1)
    rows = pd.DataFrame({"date": [d0 + timedelta(days=i // 3) for i in range(300)], "hold_sessions": [10] * 300})
    assert sf.n_eff(rows) == pytest.approx(10.0)
    assert sf.n_eff(rows.iloc[0:0]) == 0.0


# --------------------------------------------------------------------------- label arithmetic

class _Frame:
    """A friction model charging a flat half-spread per leg (so the arithmetic below is exact)."""

    def __init__(self, half=0.01):
        self.default = LegSpreadModel("const_usd", a=half)
        self.bands = {"x": self.default}
        self._m = BandedFriction(self.default, {}, 0.01)

    def half_spread_usd(self, mid, dte):
        return self._m.half_spread_usd(mid, dte)


def _inputs(spot_by_expiry: dict, closes: dict | None = None):
    frame = pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.to_datetime(["2025-05-01", "2025-05-02"]))
    return SimpleNamespace(
        spot=spot_by_expiry, frame=frame, friction=_Frame(0.01),
        bt=SimpleNamespace(commission_per_contract=0.65, friction_exit_capture=1.0))


def _panel(S_T_date=date(2025, 5, 2)):
    # short put 95 / long put 90, short call 105 / long call 110: width 5, credit 1.00, risk 4.00
    return pd.DataFrame([dict(
        date=date(2025, 5, 1), expiry=S_T_date, bucket="A", p_short=95.0, p_long=90.0, c_short=105.0, c_long=110.0,
        credit=1.00, width=5.0, risk=4.00, entry_spread=0.04, entry_comm=0.026)])


@pytest.mark.parametrize("S_T,settle,exit_legs", [
    (100.0, 0.0, 0),        # expires between the shorts: every leg worthless, no exit cost
    (93.0, 2.0, 1),         # short put 2 in the money: only that leg pays an exit cost
    (80.0, 5.0, 2),         # through the long put: both put legs in the money, loss capped at the width
    (107.0, 2.0, 1),        # short call 2 in the money
    (120.0, 5.0, 2),
])
def test_label_arithmetic(S_T, settle, exit_legs):
    inp = _inputs({date(2025, 5, 2): S_T})
    out, lost = sf.attach_outcomes(_panel(), inp)
    r = out.iloc[0]
    exit_cost = exit_legs * (0.01 + 0.65 / 100.0)
    assert sum(lost.values()) == 0
    assert r["settle"] == pytest.approx(settle)
    assert r["pnl_1"] == pytest.approx((1.00 - (0.04 + 0.026 + exit_cost) - settle) / 4.00)
    assert r["pnl_2"] == pytest.approx((1.00 - 2 * (0.04 + 0.026 + exit_cost) - settle) / 4.00)
    assert bool(r["max_loss_hit"]) == (settle >= 5.0 - 1e-9)
    assert bool(r["inside_shorts"]) == (95.0 <= S_T <= 105.0)


def test_max_loss_is_bounded_by_the_width_and_a_missing_expiry_close_is_counted():
    out, _ = sf.attach_outcomes(_panel(), _inputs({date(2025, 5, 2): 1.0}))       # far through every strike
    assert out.iloc[0]["settle"] == pytest.approx(5.0)
    assert out.iloc[0]["pnl_1"] >= -1.5                                           # (1 - 5 - costs) / 4: bounded
    out, lost = sf.attach_outcomes(_panel(), _inputs({}))
    assert out.empty and lost["A"] == 1


def test_leaked_canary_is_realised_vol_from_entry_to_expiry():
    inp = _inputs({date(2025, 5, 2): 100.0})
    out, _ = sf.attach_outcomes(_panel(), inp)
    assert out.iloc[0]["leak_rv"] == pytest.approx(abs(math.log(101.0 / 100.0)) * math.sqrt(252.0))


# --------------------------------------------------------------------------- folds: purge and training-only fits

def test_training_cutoff_is_five_sessions_before_the_fold_and_purges_by_expiry():
    for fold, (start, _) in enumerate(sf.FOLDS):
        cutoff = sf.fold_train_cutoff(fold, NYSE)
        i = NYSE.index(cutoff)
        assert NYSE.index(next(d for d in NYSE if d >= start)) - i == sf.GAP_SESSIONS
        assert cutoff < start


def _panel_for_folds(n_per_day=1, signal=0.0, seed=1, hold=5, bucket="B"):
    rng = np.random.default_rng(seed)
    rows = []
    for d in [x for x in NYSE if sf.WINDOW_START <= x <= sf.WINDOW_END]:
        for _ in range(n_per_day):
            x = rng.normal()
            exp = NYSE[NYSE.index(d) + hold]
            rows.append(dict(date=d, expiry=exp, bucket=bucket, fold=sf.fold_of(d), hold_sessions=hold,
                             vrp_h=x, term_slope=rng.normal(), rr25_norm=rng.normal(),
                             iv_rank=rng.uniform(0, 100), atm_iv=rng.uniform(0.1, 0.3),
                             pnl_1=signal * x + rng.normal(), pricing_resid=rng.normal()))
    df = pd.DataFrame(rows)
    df["pnl_2"] = df["pnl_1"] - 0.05
    return df


def test_no_training_row_expires_on_or_after_the_fold_cutoff_and_cut_points_use_training_only():
    df = _panel_for_folds()
    # make the TEST fold's feature distribution wildly different from every earlier period: if cut points or
    # percentile transforms leaked test information they would adapt to it
    shift = df["fold"] == 2
    df.loc[shift, "vrp_h"] = df.loc[shift, "vrp_h"] + 100.0
    frames = dict(sf.fold_frames(df, "vrp_h", NYSE))
    assert set(frames) == set(range(6))
    t2 = frames[2]
    assert (t2["tercile"] == 1).all()                      # all above the TRAINING upper cut point, not re-cut on test
    cutoff = sf.fold_train_cutoff(2, NYSE)
    assert (df[(df["expiry"] <= cutoff)]["date"] < sf.FOLDS[2][0]).all()


def test_ecdf_transform_ranks_against_the_given_training_values_only():
    train = np.array([1.0, 2.0, 3.0, 4.0])
    assert list(sf.ecdf_transform(train, np.array([0.0, 2.5, 99.0]))) == [0.0, 0.5, 1.0]


# --------------------------------------------------------------------------- statistics

def test_spearman_pearson_and_degenerate_inputs():
    x = np.arange(20.0)
    assert sf.spearman(x, x ** 3) == pytest.approx(1.0)
    assert sf.spearman(x, -x) == pytest.approx(-1.0)
    assert math.isnan(sf.spearman(np.ones(20), x)) and math.isnan(sf.spearman([1, 2], [1, 2]))
    assert sf.pearson(x, 2 * x + 1) == pytest.approx(1.0) and math.isnan(sf.pearson(np.ones(5), np.arange(5.0)))


def test_holm_is_monotone_capped_and_orders_by_p():
    adj = sf.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj["a"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.09) and adj["b"] == pytest.approx(0.09)
    assert adj["d"] == pytest.approx(0.5)
    assert sf.holm_adjust({"x": 0.9, "y": 0.8})["x"] == 1.0
    assert sf.holm_adjust({}) == {}


def test_boot_p_one_sided_and_two_sided():
    pos = np.full(999, 0.3)
    assert sf.boot_p(pos, True) == pytest.approx(1 / 1000)
    assert sf.boot_p(pos, False) == pytest.approx(2 / 1000)
    neg = np.full(999, -0.3)
    assert sf.boot_p(neg, True) == pytest.approx(1.0)                  # wrong side for a one-sided positive test
    assert math.isnan(sf.boot_p(np.array([np.nan]), True))


def test_block_bootstrap_is_deterministic_and_keeps_sessions_whole():
    sess = np.repeat(np.arange(40), 3)                                 # 40 sessions x 3 rows
    seen = []

    def stat(idx):
        seen.append(np.bincount(sess[idx], minlength=40))
        return np.array([float(np.sum(sess[idx]))])

    a = sf.block_bootstrap(stat, sess, 5, 50, 7)
    b = sf.block_bootstrap(stat, sess, 5, 50, 7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, sf.block_bootstrap(stat, sess, 5, 50, 8))
    assert all((c % 3 == 0).all() for c in seen)                        # a session is never split: its 3 rows come together
    assert a.shape == (50, 1)


def test_block_bootstrap_draws_consecutive_sessions_in_blocks():
    sess = np.arange(30)                                                # one row per session, in time order
    runs = []

    def stat(idx):
        runs.append(np.array(idx))
        return np.array([0.0])

    sf.block_bootstrap(stat, sess, 10, 20, 3)
    for idx in runs[1:]:                                                # [0] is the full-sample call
        for start in range(0, len(idx), 10):
            blk = idx[start:start + 10]
            assert (np.diff(blk) == 1).all() or len(blk) < 10           # inside a block the sessions are consecutive
    # one block as long as the sample reproduces the sample every time
    full = sf.block_bootstrap(lambda i: np.array([float(np.sum(i))]), sess, 30, 5, 1)
    assert (full == float(np.sum(np.arange(30)))).all()


# --------------------------------------------------------------------------- the four criteria

def _good(**kw):
    r = dict(tested=True, partial_same_sign=True, p_partial=0.01, tercile_spread_1x=0.2, tercile_spread_2x=0.1,
             folds_same_sign=5, folds={})
    r.update(kw)
    return r


def test_a_feature_passes_only_when_every_criterion_holds():
    assert sf.passes(_good(), 0.05, None)[0] is True
    ok, fails = sf.passes(_good(), 0.20, None)
    assert not ok and any("criterion 1" in f for f in fails)
    assert not sf.passes(_good(partial_same_sign=False), 0.05, None)[0]
    assert not sf.passes(_good(p_partial=0.30), 0.05, None)[0]
    assert not sf.passes(_good(tercile_spread_1x=-0.01), 0.05, None)[0]
    assert not sf.passes(_good(tercile_spread_2x=0.0), 0.05, None)[0]            # a tie is not a pass
    assert not sf.passes(_good(folds_same_sign=3), 0.05, None)[0]
    assert sf.passes(_good(folds_same_sign=4), 0.05, None)[0]
    assert not sf.passes(_good(), 0.05, "vrp_h")[0]                                # blocked by a correlated earlier feature
    assert not sf.passes({"tested": False}, 0.0, None)[0]


# --------------------------------------------------------------------------- end to end on a fabricated panel

def _boot(n):                                                          # fewer resamples keep the tests quick
    return n


def test_planted_signal_is_found_and_noise_is_not():
    sig = sf.analyse_feature(_panel_for_folds(signal=0.5, n_per_day=2), "vrp_h", NYSE, block_len=5, seed=3, n_boot=300)
    assert sig["tested"] and sig["rho"] > 0.2 and sig["p_raw"] < 0.01 and sig["partial_rho"] > 0.2
    assert sig["tercile_spread_1x"] > 0 and sig["folds_same_sign"] >= 5
    noise = sf.analyse_feature(_panel_for_folds(signal=0.0, n_per_day=2, seed=9), "vrp_h", NYSE, block_len=5, seed=3, n_boot=300)
    assert abs(noise["rho"]) < 0.15 and noise["p_raw"] > 0.02


def test_a_negative_effect_does_not_pass_a_one_sided_positive_test_but_a_two_sided_feature_takes_its_own_sign():
    df = _panel_for_folds(signal=-0.5, n_per_day=2)
    r = sf.analyse_feature(df, "vrp_h", NYSE, block_len=5, seed=3, n_boot=300)
    assert r["rho"] < 0 and r["p_raw"] > 0.9                                      # wrong side of the pre-stated direction
    df["term_slope"] = df["vrp_h"]                                                # same data under a two-sided feature
    r2 = sf.analyse_feature(df, "term_slope", NYSE, block_len=5, seed=3, n_boot=300)
    assert r2["rho"] < 0 and r2["p_raw"] < 0.01 and r2["direction"] == -1 and r2["tercile_spread_1x"] > 0


def test_partial_correlation_removes_what_the_controls_explain():
    df = _panel_for_folds(signal=0.0, n_per_day=2, seed=5)
    # outcome and feature both driven by IV rank: the raw correlation is real, the partial one is not
    df["vrp_h"] = df["iv_rank"] / 50.0 + np.random.default_rng(1).normal(scale=0.3, size=len(df))
    df["pnl_1"] = df["iv_rank"] / 50.0 + np.random.default_rng(2).normal(scale=0.3, size=len(df))
    df["pnl_2"] = df["pnl_1"]
    r = sf.analyse_feature(df, "vrp_h", NYSE, block_len=5, seed=3, n_boot=300)
    assert r["rho"] > 0.5 and abs(r["partial_rho"]) < 0.2


def test_leaked_canary_fires_and_shuffled_canary_stays_quiet():
    df = _panel_for_folds(signal=0.0, n_per_day=2, seed=11)
    df["leak_rv"] = -df["pnl_1"] + np.random.default_rng(4).normal(scale=0.2, size=len(df))   # the unknowable future
    sf_boot = sf.BOOT_N
    try:
        sf.BOOT_N = 300
        out = sf.canaries(df, NYSE, "B", block_len=5, sufficient=True)
    finally:
        sf.BOOT_N = sf_boot
    assert out["ran"] and out["leak_ok"] is True and out["leak_p"] < 0.01
    assert out["shuffle_ok"] is True and out["shuffle_hits"] < 6


def test_analyse_marks_an_insufficient_bucket_descriptive_and_never_useful():
    # one bucket whose n_eff is far below 30 (few sessions, long hold): descriptive statistics only, no verdict
    df = _panel_for_folds(signal=0.8, n_per_day=1, hold=60, bucket="C")
    df = df[df["date"].between(date(2025, 4, 1), date(2025, 7, 15)) | df["fold"].isna()]
    df["atm_iv"], df["leak_rv"] = 0.2, -df["pnl_1"]
    for c in ("bf25", "fwd_var"):
        df[c] = 0.0
    df["entry_spread"] = df["entry_comm"] = 0.0
    df["max_loss_hit"] = df["inside_shorts"] = df["extrapolated"] = False
    inp = SimpleNamespace(friction_report={})
    inp.nyse = NYSE
    res = sf.analyse(df, inp, Counter(), Counter())
    c = res["buckets"]["C"]
    assert c["insufficient"] and c["verdict"] == "INSUFFICIENT" and c["passed"] == []
    assert all(f.get("descriptive_only") for f in c["features"].values() if f.get("tested"))
    assert res["buckets"]["A"]["rows"] == 0                                        # empty buckets do not crash


def test_markdown_renders_statistics_only():
    df = _panel_for_folds(signal=0.5, n_per_day=2, hold=5, bucket="B")
    df["leak_rv"] = -df["pnl_1"]
    df["max_loss_hit"] = df["inside_shorts"] = df["extrapolated"] = False
    inp = SimpleNamespace(friction_report={}, nyse=NYSE)
    old = sf.BOOT_N
    try:
        sf.BOOT_N = 200
        res = sf.analyse(df, inp, Counter(), Counter())
    finally:
        sf.BOOT_N = old
    md = sf.render_markdown(res, {"ok": True, "checked": 8, "changed": []}, smoke=True)
    assert "SMOKE RUN" in md and "Bucket B" in md and "vrp_h" in md
    assert "pnl_1" not in md and "credit" not in md                                # no per-row prices or P&L


# --------------------------------------------------------------------------- the as-of mutation guard can fail

def _synthetic_inputs():
    """A small real IVSurface (known smile, daily expiries), a long daily history for the HAR fit and stored IV."""
    from tests.test_w75_iv_surface import DATES, S0, make_bars
    from ait.backtesting.iv_surface import IVSurface
    from ait.data.iv_rank import iv_rank_series

    rng = np.random.default_rng(0)
    days = pd.to_datetime([d for d in business_days(date(2020, 1, 1), date(2025, 3, 31))])
    close = 400 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    frame = pd.DataFrame({"Open": close * (1 + rng.normal(0, 0.002, len(days))), "High": close * 1.006,
                          "Low": close * 0.994, "Close": close, "Volume": 1e6,
                          "implied_vol": 0.2 + 0.02 * rng.normal(size=len(days))}, index=days)
    surf = IVSurface.from_bars(make_bars(DATES))
    inp = sf.Inputs(surface=surf, sessions=list(DATES), spot={d: S0 for d in DATES}, legs={}, friction=None,
                    friction_report={}, bt=None, tuning=None, frame=frame,
                    iv_rank=iv_rank_series(frame["implied_vol"]), nyse=NYSE)
    D = DATES[3]
    panel = pd.DataFrame([dict(date=D, expiry=D + timedelta(days=5), hold_sessions=3)])
    return inp, panel


def test_asof_mutation_check_passes_for_the_real_feature_code_and_catches_a_leak(monkeypatch):
    inp, panel = _synthetic_inputs()
    ok = sf.asof_mutation_check(inp, panel, n_samples=1)
    assert ok["ok"] and ok["checked"] == 1 and ok["changed"] == []

    real = sf.compute_features

    def leaky(i, har, cache, D, Dm, E, smile, atm, hold):
        f = real(i, har, cache, D, Dm, E, smile, atm, hold)
        f["iv_rank"] = i.spot[D]                                       # reads the entry day's own spot: not knowable at D-1
        return f

    monkeypatch.setattr(sf, "compute_features", leaky)
    bad = sf.asof_mutation_check(inp, panel, n_samples=1)
    assert not bad["ok"] and bad["changed"][0]["features_changed"] == ["iv_rank"]

    def leaky_smile(i, har, cache, D, Dm, E, smile, atm, hold):
        f = real(i, har, cache, D, Dm, E, smile, atm, hold)
        later = i.surface.smile(D, E)                                  # the smile fitted AFTER the as-of date
        f["bf25"] = later.a if later is not None else float("nan")
        return f

    monkeypatch.setattr(sf, "compute_features", leaky_smile)
    assert not sf.asof_mutation_check(inp, panel, n_samples=1)["ok"]


# --------------------------------------------------------------------------- shared-code fixes found by the first run

def test_surface_cache_stamp_describes_what_was_loaded_when_rows_land_between_the_count_and_the_load(tmp_path, monkeypatch):
    """A backfill that appends rows after ``build_or_load`` counted but before it loads must not leave a cache stamped
    older than its content: the stamp is taken from the loaded bars, and the very next call is a cache hit."""
    import json
    import sqlite3
    from ait.backtesting import iv_surface as ivs
    from tests.test_w75_iv_surface import DATES, _write_db, make_bars

    bars = make_bars(DATES)
    db, cache = _write_db(tmp_path / "h.db", bars), tmp_path / "cache"
    real_load = ivs.load_bars

    def a_backfill_lands_first(*a, **k):
        c = sqlite3.connect(db)
        c.execute("INSERT INTO option_daily_bars (underlying, expiry, right, strike, date, dte, close, volume,"
                  " underlying_close, iv, iv_outcome, risk_free_rate) VALUES "
                  "('QQQ','2025-03-20','P',490,'2025-03-10',10,0.5,100,500,0.2,'solved',0.05)")
        c.commit()
        c.close()
        return real_load(*a, **k)

    monkeypatch.setattr(ivs, "load_bars", a_backfill_lands_first)
    ivs.build_or_load(db, "QQQ", cache)
    stamp = json.loads((cache / "iv_surface_QQQ.json").read_text())
    with sqlite3.connect(db) as c:
        n, last = c.execute("SELECT COUNT(*), MAX(date) FROM option_daily_bars WHERE underlying='QQQ'").fetchone()
    assert stamp["rows"] == n == len(bars) + 1 and stamp["max_date"] == last

    def must_not_reload(*a, **k):
        raise AssertionError("rebuilt although the cache matches the table")
    monkeypatch.setattr(ivs, "load_bars", must_not_reload)
    ivs.build_or_load(db, "QQQ", cache)


def test_run_record_fingerprints_the_option_bars_and_a_landing_backfill_changes_it(tmp_path):
    import sqlite3
    from ait.backtesting import run_record as rr

    db = tmp_path / "h.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE option_daily_bars (underlying TEXT, option_ticker TEXT, date TEXT)")
    rows = [("QQQ", "a", "2026-10-19"), ("QQQ", "b", "2026-10-19"), ("QQQ", "c", "2026-10-20"),
            ("QQQ", "d", "2026-10-21"), ("SPY", "e", "2026-10-19")]
    con.executemany("INSERT INTO option_daily_bars VALUES (?,?,?)", rows)
    con.commit()
    con.close()
    before = db.stat().st_mtime_ns
    fp = rr.data_fingerprint(db, "", ["QQQ", "SPY"], as_of="2026-10-20")
    assert db.stat().st_mtime_ns == before                                          # read-only
    assert fp["tables"]["option_daily_bars"] == {"rows": 5, "first": "2026-10-19", "last": "2026-10-21"}
    assert fp["option_bars"]["QQQ"] == {"rows": 4, "last": "2026-10-21", "rows_to_as_of": 3}
    assert fp["option_bars"]["SPY"] == {"rows": 1, "last": "2026-10-19", "rows_to_as_of": 1}
    con = sqlite3.connect(db)                                                       # a backfill adds a late expiry's bars
    con.execute("INSERT INTO option_daily_bars VALUES ('QQQ','f','2026-10-20')")    # dated inside the window
    con.commit()
    con.close()
    after = rr.data_fingerprint(db, "", ["QQQ", "SPY"], as_of="2026-10-20")
    assert after["option_bars"]["QQQ"]["rows_to_as_of"] == 4 and after != fp
    assert "option_bars" not in rr.data_fingerprint(tmp_path / "missing.db", "", ["QQQ"])      # no store: nothing to report
    other = tmp_path / "o.db"
    sqlite3.connect(other).close()
    assert "option_bars" not in rr.data_fingerprint(other, "", ["QQQ"])                        # no option table: key absent


# --------------------------------------------------------------------------- opt-in surface cut and the exploratory exit

def test_cut_expiries_keeps_only_smiles_up_to_the_end_and_changes_nothing_inside():
    from ait.backtesting.iv_surface import IVSurface
    from tests.test_w75_iv_surface import DATES, S0, make_bars

    surf = IVSurface.from_bars(make_bars(DATES))
    d, last = DATES[0], DATES[0] + timedelta(days=5)
    cut = surf.cut_expiries(last)
    assert cut.expiries(d) == [e for e in surf.expiries(d) if e <= last] and len(cut) < len(surf)
    inside = d + timedelta(days=3)
    assert cut.iv(d, inside, 505.0, S0) == surf.iv(d, inside, 505.0, S0)
    late = d + timedelta(days=9)                                         # past the cut: no own smile, flat from the last kept expiry
    assert cut.iv(d, late, 505.0, S0) != surf.iv(d, late, 505.0, S0)
    everything = surf.cut_expiries(d + timedelta(days=400))
    assert len(everything) == len(surf) and everything.iv(d, late, 505.0, S0) == surf.iv(d, late, 505.0, S0)


def _tp_inputs(marks):
    return SimpleNamespace(
        sessions=[date(2025, 5, 1), date(2025, 5, 2), date(2025, 5, 5), date(2025, 5, 6), date(2025, 5, 7)],
        marks=marks, friction=_Frame(0.01),
        bt=SimpleNamespace(commission_per_contract=0.65, friction_exit_capture=1.0))


def _tp_marks(by_day):
    """{day: (short put, long put, short call, long call)} -> the dict the exit reads (put strikes 95/90, call 105/110)."""
    out = {}
    for day, q in by_day.items():
        if q[0] is not None:
            out[(day, date(2025, 5, 7), "P")] = (np.array([90.0, 95.0]), np.array([q[1], q[0]]))
            out[(day, date(2025, 5, 7), "C")] = (np.array([105.0, 110.0]), np.array([q[2], q[3]]))
    return out


def _tp_panel(pnl_1=0.123):
    return pd.DataFrame([dict(date=date(2025, 5, 1), expiry=date(2025, 5, 7), p_short=95.0, p_long=90.0, c_short=105.0,
                              c_long=110.0, credit=1.00, risk=4.00, entry_spread=0.04, entry_comm=0.026,
                              pnl_1=pnl_1, pnl_2=pnl_1 - 0.05)])


D2, D5, D6 = date(2025, 5, 2), date(2025, 5, 5), date(2025, 5, 6)
EXIT_COSTS = 0.04 + 0.026                                                 # four legs at 0.01, four commissions of 0.0065


def test_take_profit_exits_on_the_first_priced_session_at_or_below_half_the_credit():
    marks = _tp_marks({D2: (0.60, 0.10, 0.55, 0.10),                      # mark 0.95: not yet
                       D5: (0.35, 0.05, 0.30, 0.05),                      # mark 0.55: not yet ... then
                       D6: (0.20, 0.05, 0.15, 0.05)})                     # mark 0.25: first at or below 0.50
    out = sf.attach_tp_exit(_tp_panel(), _tp_inputs(marks)).iloc[0]
    assert out["tp_hit"] and out["tp_exit"] == D6 and out["tp_days_evaluable"] == 3
    assert out["pnl_tp_1"] == pytest.approx((1.00 - (0.066 + EXIT_COSTS) - 0.25) / 4.00)
    assert out["pnl_tp_2"] == pytest.approx((1.00 - 2 * (0.066 + EXIT_COSTS) - 0.25) / 4.00)


def test_take_profit_boundary_is_inclusive_and_later_sessions_are_ignored_once_hit():
    marks = _tp_marks({D2: (0.75, 0.50, 0.75, 0.50),                      # mark exactly 0.50 = half the credit: a hit
                       D5: (0.05, 0.01, 0.05, 0.01)})                     # even better, but after the exit
    out = sf.attach_tp_exit(_tp_panel(), _tp_inputs(marks)).iloc[0]
    assert out["tp_hit"] and out["tp_exit"] == D2 and out["tp_days_evaluable"] == 1
    assert out["pnl_tp_1"] == pytest.approx((1.00 - (0.066 + EXIT_COSTS) - 0.50) / 4.00)


def test_take_profit_skips_sessions_with_a_leg_missing_and_counts_only_the_priced_ones():
    marks = _tp_marks({D2: (0.60, 0.10, 0.55, 0.10), D5: (0.20, 0.05, 0.15, 0.05), D6: (0.20, 0.05, 0.15, 0.05)})
    del marks[(D5, date(2025, 5, 7), "C")]                                # the 5th: no call bars traded: unpriced, skipped
    out = sf.attach_tp_exit(_tp_panel(), _tp_inputs(marks)).iloc[0]
    assert out["tp_hit"] and out["tp_exit"] == D6 and out["tp_days_evaluable"] == 2          # D2 and D6 were priced


def test_a_row_that_never_reaches_the_target_keeps_its_hold_to_expiry_result():
    marks = _tp_marks({D2: (0.60, 0.10, 0.55, 0.10), D5: (0.55, 0.10, 0.50, 0.10)})        # marks 0.95 and 0.85
    out = sf.attach_tp_exit(_tp_panel(0.123), _tp_inputs(marks)).iloc[0]
    assert not out["tp_hit"] and out["tp_exit"] is None and out["tp_days_evaluable"] == 2
    assert out["pnl_tp_1"] == 0.123 and out["pnl_tp_2"] == pytest.approx(0.073)
    none = sf.attach_tp_exit(_tp_panel(0.5), _tp_inputs({})).iloc[0]                         # no priced session at all
    assert not none["tp_hit"] and none["tp_days_evaluable"] == 0 and none["pnl_tp_1"] == 0.5


def test_a_different_label_changes_the_statistics_but_the_default_is_the_registered_one():
    df = _panel_for_folds(signal=0.5, n_per_day=2)
    kw = dict(block_len=5, seed=3, n_boot=150)
    a = sf.analyse_feature(df, "vrp_h", NYSE, **kw)
    b = sf.analyse_feature(df, "vrp_h", NYSE, label="pnl_1", label_2x="pnl_2", **kw)
    for k in ("rho", "rho_ci90", "p_raw", "partial_rho", "tercile_spread_1x", "tercile_spread_2x", "folds_same_sign"):
        assert a[k] == b[k], k
    df["other"] = -df["pnl_1"]
    c = sf.analyse_feature(df, "vrp_h", NYSE, label="other", label_2x="other", **kw)
    assert c["rho"] == pytest.approx(-a["rho"])


def test_exploratory_report_is_labelled_and_statistics_only():
    df = _panel_for_folds(signal=0.5, n_per_day=1, hold=5, bucket="B")
    df["pnl_tp_1"], df["pnl_tp_2"] = df["pnl_1"] + 0.1, df["pnl_2"] + 0.1
    df["tp_hit"], df["tp_days_evaluable"] = df["pnl_1"] > 0, 3
    old = sf.BOOT_N
    try:
        sf.BOOT_N = 150
        rep = sf.tp_exploratory(df, SimpleNamespace(nyse=NYSE))
    finally:
        sf.BOOT_N = old
    assert rep["buckets"]["A"]["test_rows"] == 0 and rep["buckets"]["B"]["test_rows"] > 0
    assert 0 < rep["buckets"]["B"]["tp_hit_rate"] < 1 and set(rep["buckets"]["B"]["features"]) == set(sf.PRIMARY)
    md = sf.render_tp_markdown(rep)
    assert "EXPLORATORY" in md and "not part of the registered rule" in md and "credit" not in md.replace("of the credit", "")

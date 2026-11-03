"""W119 S1: the nine arm study files match the arm table registered in PLAN.md "W119 S1 AS PLANNED".

The table lives in scripts/fractal_ablation_study.py (ARMS); the study files are what a run actually reads, so a file
that drifts from the table would silently change what an arm measures.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "fractal_ablation_study.py"
    spec = importlib.util.spec_from_file_location("fractal_ablation_study", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fractal_ablation_study"] = mod
    spec.loader.exec_module(mod)
    return mod


def _switched_off(study: dict) -> dict[str, set[str]]:
    feats = (study.get("settings") or {}).get("ml", {}).get("features", {})
    return {g: {c for c, on in (v.get("use") or {}).items() if on is False} for g, v in feats.items()}


def test_every_arm_has_a_study_file_with_exactly_its_switches():
    m = _load_script()
    assert set(m.ARMS) == {"a0", "a1", "a2", "a3", "a4", "a5", "a6", "b0", "b1", "b2"}
    for arm, (_stage, groups) in m.ARMS.items():
        study = yaml.safe_load(m.study_path(arm).read_text())
        assert study["name"] == f"fractal_ablation_{arm}"
        assert study["rule"] == "W119 S1 AS PLANNED"
        assert _switched_off(study) == {g: set(cols) for g, cols in groups.items()}, arm
        # nothing but ml.features switches is set by an arm
        assert set((study.get("settings") or {}).keys()) <= {"ml"}


def test_arm_columns_are_real_fractal_columns_and_blocks_do_not_overlap_wrongly():
    m = _load_script()
    assert len(m.DAILY_FRACTAL) == 10 and len(m.INTRADAY_FRACTAL) == 7
    daily = set(m.DAILY_FRACTAL)
    for arm in ("a2", "a3", "a4", "a5"):
        assert set(m.ARMS[arm][1]["fractal"]) <= daily
    # the three blocks of the daily group partition it, apart from the canary and gate arms
    blocks = [set(m.ARMS[a][1]["fractal"]) for a in ("a2", "a3", "a4", "a5")]
    assert set().union(*blocks) == daily and sum(len(b) for b in blocks) == 10


def test_windows_have_non_overlapping_test_periods_and_stay_inside_the_frame():
    m = _load_script()
    wins = m.windows(1000, 252, 63, 63)
    assert wins and all(d <= 1000 for _a, _b, _c, d in wins)
    for (a1, b1, c1, d1), (a2, b2, c2, d2) in zip(wins, wins[1:]):
        assert d1 <= c2          # the next test starts where the previous one ended
        assert b1 == c1 and b2 == c2 and a2 - a1 == 63
    assert m.windows(300, 252, 63, 63) == []
    assert len(m.windows(252 + 63 + 63 * 3, 252, 63, 63)) == 4
    idx = pd.bdate_range("2020-01-01", periods=1000)
    assert idx[wins[0][2]] > idx[wins[0][1] - 1]


# ---- the statistics of the scoring path (pure functions) -------------------------------------------------------------
def test_labels_match_the_engine_range_label():
    import numpy as np
    m = _load_script()
    close = np.array([100.0, 101.0, 99.0, 104.0, 100.0, 100.5])
    y = m.labels(close, 2, 0.029)
    # t=0: max |ret| over closes 1..2 = 1% < 2.9% -> 1 ; t=1: 99/101-1 = -1.98% and 104/101-1 = +2.97% (not below 2.9%) -> 0
    assert y[0] == 1.0 and y[1] == 0.0 and np.isnan(y[-1]) and np.isnan(y[-2])


def test_holm_is_monotone_capped_and_matches_the_textbook_values():
    m = _load_script()
    adj = m.holm([0.01, 0.04, 0.03])
    assert adj == [0.03, 0.06, 0.06]            # sorted p .01 x3 = .03 ; .03 x2 = .06 ; .04 x1 = .04 -> running max .06
    assert m.holm([0.6, 0.7]) == [1.0, 1.0] or max(m.holm([0.6, 0.7])) <= 1.0


def test_block_bootstrap_is_reproducible_and_centred_on_the_sample_mean():
    import numpy as np
    m = _load_script()
    x = np.linspace(-0.02, 0.0, 30)
    b1, b2 = m.block_bootstrap_means(x), m.block_bootstrap_means(x)
    assert np.array_equal(b1, b2) and len(b1) == m.BOOT_B
    assert abs(b1.mean() - x.mean()) < 0.002


def test_one_sided_p_separates_a_clear_effect_from_noise_and_is_never_zero():
    import numpy as np
    m = _load_script()
    clear = np.full(30, -0.05)                       # removal clearly lowers skill by far more than 0.01
    assert m.one_sided_p(clear, null=-m.THRESHOLD, direction=-1) < 0.01
    assert m.one_sided_p(clear, null=-m.THRESHOLD, direction=-1) > 0.0
    noise = np.tile([-0.03, 0.03], 15)               # mean 0
    assert m.one_sided_p(noise, null=-m.THRESHOLD, direction=-1) > 0.10


def test_classify_applies_the_registered_rule():
    m = _load_script()
    assert m.classify(-0.02, 0.05, 30) == "KEEP"
    assert m.classify(+0.02, 0.05, 30) == "REMOVAL OK"
    assert m.classify(-0.02, 0.20, 30) == "NOT SHOWN"          # not significant
    assert m.classify(-0.005, 0.01, 30) == "NOT SHOWN"         # smaller than the 0.01 threshold
    assert m.classify(-0.05, 0.001, m.MIN_WINDOWS - 1) == "INSUFFICIENT"


def test_market_context_slices_like_the_walk_forward_and_forward_fills():
    import pandas as pd
    m = _load_script()
    idx = pd.bdate_range("2024-01-01", periods=10)
    vix = pd.DataFrame({"Close": range(5)}, index=idx[::2])        # every other day
    ctx = m.market_context(vix, None, {"x": 1}, idx)
    assert list(ctx) == ["vix", "macros"] and len(ctx["vix"]) == 10
    assert ctx["vix"]["Close"].iloc[1] == ctx["vix"]["Close"].iloc[0]       # ffill
    assert m.market_context(None, None, None, idx) is None


def test_deep_merge_keeps_siblings_and_overrides_leaves():
    m = _load_script()
    out = m._deep_merge({"ml": {"features": {"fractal": {"use": {"a": False}}}}}, {"ml": {"hyperparams": {"random_state": 7}}})
    assert out["ml"]["features"]["fractal"]["use"] == {"a": False} and out["ml"]["hyperparams"]["random_state"] == 7

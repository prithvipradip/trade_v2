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

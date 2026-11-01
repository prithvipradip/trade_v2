"""W106 item 3: the bulky, regenerated files under reports/ are git-ignored.

2,238 per-window `window_N.json` files and 39 `timeseries_bars.json` make up most of
the 2,504 tracked report files. New ones are ignored; a study's results and
summaries stay committable. Nothing already tracked is removed (operators' call).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

IGNORED = [
    "reports/some_study/variant/window_081.json",
    "reports/some_study/variant/timeseries_bars.json",
    "reports/some_study/entry_dte_18/run.log",
    "reports/daily_20261028_163000.json",          # the supervisor's daily report
    "reports/backtest_20261025_200000.json",       # the supervisor's weekly backtest
    "reports/econ_calendar_audit_20261028.json",   # written by load_econ_calendar_csv (and its tests)
    "reports/official_calendar_reconciliation_20261031.json",   # written by build_official_calendar
]
KEPT = [
    "reports/some_study/entry_dte_18/results.md",
    "reports/some_study/entry_dte_18/results.json",
    "reports/some_study/variant_summary.txt",
    "reports/some_study/variant.json",
]


def _ignored(path: str) -> bool:
    r = subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT)
    return r.returncode == 0


@pytest.mark.parametrize("path", IGNORED)
def test_regenerated_artifacts_are_ignored(path):
    assert _ignored(path), path


@pytest.mark.parametrize("path", KEPT)
def test_results_stay_committable(path):
    assert not _ignored(path), path

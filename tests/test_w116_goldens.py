"""W116 golden: complete stored inputs give the SAME research result before and after the loaders turn strict.

``tests/data/w116_store_walkforward_golden.json`` was captured from the code BEFORE any W116 source change, by a
walk-forward that loads everything (IB-tagged daily bars, stored IV, the VIX series) through the real research
loaders from a seeded history database. A strict loader may only refuse a study whose inputs are missing; for a
study whose inputs exist it must reproduce this record exactly. (The offline, frames-in version of the same check
is ``tests/test_w108s1_carried_equity.py``.)

House rules: real code paths, no source-text assertions.
"""
from __future__ import annotations

import json
import math

import pytest

from tests.data import _w116_store_golden_inputs as gi
from tests.data._w108s1_golden_inputs import snapshot

GOLDEN = json.loads(gi.GOLDEN_PATH.read_text(encoding="utf-8"))


def _same(got, want, path="") -> None:
    """Recursive equality: exact for strings / ints / structure, 1e-9 relative for floats (the golden-record convention)."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and sorted(got) == sorted(want), path
        for k in want:
            _same(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), path
        for i, (g, w) in enumerate(zip(got, want)):
            _same(g, w, f"{path}[{i}]")
    elif isinstance(want, float) and not isinstance(want, bool):
        if math.isinf(want) or math.isnan(want):
            assert str(got) == str(want), path
        else:
            assert got == pytest.approx(want, rel=1e-9, abs=1e-9), path
    else:
        assert got == want, path


@pytest.fixture(scope="module")
def store_run(tmp_path_factory):
    """One store-backed run; Yahoo is counted so a silent fallback cannot hide behind identical numbers."""
    import yfinance

    calls = {"n": 0}

    class _Ticker:
        def __init__(self, *a, **k):
            calls["n"] += 1
            raise AssertionError("Yahoo was asked although the stored history is complete")

    mp = pytest.MonkeyPatch()
    mp.setattr(yfinance, "Ticker", _Ticker)
    try:
        snap = snapshot(gi.run(tmp_path_factory.mktemp("w116") / "historical.db"))
    finally:
        mp.undo()
    return snap, calls


class TestStoreBackedWalkForward:
    def test_no_yahoo_request_is_made(self, store_run):
        assert store_run[1]["n"] == 0

    def test_every_trade_window_and_metric_equals_the_record_captured_before_the_change(self, store_run):
        snap = store_run[0]
        for key in ("windows", "metrics", "symbol_results", "strategy_results", "equity_curve", "entry_funnel"):
            _same(snap[key], GOLDEN[key], key)

    def test_the_fixture_trades_in_several_windows(self):
        assert sum(1 for w in GOLDEN["windows"] if w["trades"]) >= 3

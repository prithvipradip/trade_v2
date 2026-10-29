"""W106 item 8: the four study scripts take `--study-file` and write a run record.

W87b gave `run_backtest.py` / `run_optimizer.py` a study file and a run record; the
four study scripts (`gate_parity_ab`, `ex_div_gate_ab`, `dte_dimension_flatness`,
`iv_rank_parity_probe`) took neither (W87 FELL SHORT 2). They run their arms in a
process pool, so the study has to reach every WORKER, not only the parent.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import ait.backtesting.study_harness as sh
from tests.test_w71_iv_rank_parity import _make_db
from tests.test_w87a_hygiene import _load_script

STUDY = "name: w106_probe\nquestion: does the study reach the workers\nrule: exploratory\n" \
        "settings:\n  backtest:\n    slippage_pct: 0.02\n"


def _study(tmp_path):
    p = tmp_path / "w106_probe.yaml"
    p.write_text(STUDY)
    return p


def _worker_slippage():
    from ait.config.settings import load_settings
    return load_settings().backtest.slippage_pct


def test_a_real_worker_runs_under_the_study(tmp_path):
    with sh.study_pool(1, _study(tmp_path)) as ex:
        assert ex.submit(_worker_slippage).result() == pytest.approx(0.02)


def test_without_a_study_the_worker_reads_config_yaml(tmp_path):
    with sh.study_pool(1, None) as ex:
        assert ex.submit(_worker_slippage).result() == pytest.approx(0.01)


class _FakePool:
    """The process pool at its boundary: records the study it was given, returns canned rows."""

    def __init__(self, rows, seen):
        self._rows, self._seen = rows, seen

    def __call__(self, workers, study_file=None):
        self._seen.append(study_file)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, tasks):
        return [dict(self._rows) for _ in tasks]


ROWS = {
    "gate_parity_ab": {"symbol": "QQQ", "arm": "live", "trades": [], "regime_veto_days": 0,
                       "vol_gate_days": 0},
    "ex_div_gate_ab": {"symbol": "QQQ", "entry_dte": 14, "arm": "off", "n": 0, "win": 0.0, "total": 0.0,
                       "worst": 0.0, "blocked": 0, "ex_dates": 0},
    "dte_dimension_flatness": {"dim": "x", "dte": 3, "value": 1, "fp": "f", "n": 0, "total": 0.0},
    "iv_rank_parity_probe": {"symbol": "TST", "threshold": 0.3, "gate": True, "n": 0, "win": 0.0, "total": 0.0,
                             "rise_days": 0, "rank_floor_days": 0, "rank_cap_days": 0},
}


@pytest.mark.parametrize("name", list(ROWS))
def test_each_script_hands_the_study_to_its_pool_and_records_the_run(name, tmp_path, monkeypatch):
    seen: list = []
    monkeypatch.setattr(sh, "study_pool", _FakePool(ROWS[name], seen))
    monkeypatch.setattr(sh, "RUNS_DIR", tmp_path / "runs")
    m = _load_script(name)
    study = _study(tmp_path)
    argv = ["--study-file", str(study), "--symbols", "QQQ", "--workers", "2"]
    if name == "gate_parity_ab":
        monkeypatch.setattr(m, "vix_prev_close", lambda *a: pd.Series(dtype=float))
    if name == "dte_dimension_flatness":
        argv = ["--study-file", str(study), "--workers", "2", "--dtes", "3"]
    if name == "iv_rank_parity_probe":
        db = tmp_path / "h.db"
        _make_db(db)
        argv = ["--study-file", str(study), "--symbols", "TST", "--db", str(db),
                "--research-prefix", "research_", "--since", "2023-09-01", "--ab", "--workers", "2"]
    assert m.main(argv) == 0
    assert seen == [str(study)]
    records = list((tmp_path / "runs").glob("*/run_record.json"))
    assert len(records) == 1
    rec = json.loads(records[0].read_text())
    assert rec["study"]["name"] == "w106_probe" and rec["kind"] == "study"
    assert rec["settings"]["backtest"]["slippage_pct"] == pytest.approx(0.02)


def test_no_run_record_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(sh, "study_pool", _FakePool(ROWS["ex_div_gate_ab"], []))
    monkeypatch.setattr(sh, "RUNS_DIR", tmp_path / "runs")
    assert _load_script("ex_div_gate_ab").main(["--symbols", "QQQ", "--no-run-record"]) == 0
    assert not (tmp_path / "runs").exists()

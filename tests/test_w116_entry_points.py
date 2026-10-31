"""W116 WP4: the entry points run the preflight, refuse with exit status 2, and stamp what the study's inputs were."""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from ait.data.research_inputs import DataGap, ResearchDataUnavailable
from tests.data import _w116_store_golden_inputs as gi

pytestmark = pytest.mark.strict_inputs        # the production policy for every test here (module scope: fixtures too)

ROOT = Path(__file__).resolve().parent.parent


def _load_script(relpath: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestTheWalkForwardStampsItsInputs:
    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        return gi.run(tmp_path_factory.mktemp("w116ep") / "historical.db")

    def test_the_preflight_report_is_on_the_result(self, result):
        pre = result.data_contract["preflight"]
        assert pre["ok"] is True and pre["gaps"] == []
        assert {"daily_prices", "implied_vol", "vix_family"} <= set(pre["checked"])
        assert pre["window"][0] == str(gi.START)

    def test_the_declared_models_are_counted(self, result):
        declared = result.data_contract["declared_models"]
        assert declared.get("iv_blend_remarks", 0) > 0              # open positions are re-marked from the blend
        assert all(isinstance(v, int) for v in declared.values())

    def test_the_contract_is_json_serialisable_for_the_run_record(self, result):
        assert json.loads(json.dumps(result.data_contract))["preflight"]["ok"] is True


class TestARefusalHappensBeforeAnythingIsLoaded:
    def test_a_study_on_an_empty_store_lists_every_gap_at_once(self, tmp_path):
        import asyncio
        from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
        from ait.data.historical import HistoricalDataStore
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db)
        cfg = WalkForwardConfig(history_start_date=gi.START, macro_event_gate=False, friction_model="legacy",
                                intraday_features=False)
        wf = WalkForwardBacktester(["SPY", "QQQ"], ["iron_condor"], config=cfg, db_path=db)
        with pytest.raises(ResearchDataUnavailable) as e:
            asyncio.run(wf.run())
        inputs = {g.input for g in e.value.gaps}
        assert {"daily_prices", "vix_family"} <= inputs               # both symbols and the VIX, in ONE message
        assert {g.symbol for g in e.value.gaps if g.input == "daily_prices"} == {"SPY", "QQQ"}


class TestTheScripts:
    def test_run_backtest_exits_2_on_a_refusal(self, monkeypatch, capsys):
        rb = _load_script("run_backtest.py", "_w116_run_backtest")

        async def _refuse(args):
            raise ResearchDataUnavailable([DataGap("daily_prices", "QQQ", "x", "y", "python fix.py")], "run_backtest")
        monkeypatch.setattr(rb, "run_backtest", _refuse)
        monkeypatch.setattr(sys, "argv", ["run_backtest.py"])
        with pytest.raises(SystemExit) as e:
            rb.main()
        assert e.value.code == 2
        assert "REFUSED" in capsys.readouterr().err

    def test_run_backtest_has_no_legacy_history_switch(self, monkeypatch):
        rb = _load_script("run_backtest.py", "_w116_run_backtest2")
        monkeypatch.setattr(sys, "argv", ["run_backtest.py", "--legacy-history"])
        with pytest.raises(SystemExit):                              # argparse: unrecognised argument
            rb.parse_args()

    def test_the_preflight_cli_exits_2_with_the_gaps_and_0_when_none(self, tmp_path, capsys):
        cli = _load_script("scripts/research_preflight.py", "_w116_cli")
        assert cli.main(["--db-path", str(tmp_path / "missing.db")]) == 2
        out = capsys.readouterr().out
        assert "REFUSED" in out and "history_database" in out

    def test_the_preflight_cli_json_is_the_report(self, tmp_path, capsys):
        cli = _load_script("scripts/research_preflight.py", "_w116_cli2")
        db = tmp_path / "h.db"
        gi.seed(db)
        code = cli.main(["--db-path", str(db), "--symbols", "QQQ", "--history-start", "2022-06-01",
                         "--history-end", str(gi.LAST_SESSION), "--json"])
        out = capsys.readouterr().out
        report = json.loads(out[out.index('{\n  "window"'):])
        assert code == (0 if report["ok"] else 2) and report["window"] == ["2022-06-01", str(gi.LAST_SESSION)]
        assert "daily_prices" in report["checked"]


class TestTheStudyScriptsPreflightTheirWindow:
    def test_study_inputs_refuse_a_window_the_store_does_not_hold(self, tmp_path):
        from ait.backtesting.study_harness import study_inputs
        from ait.data.historical import HistoricalDataStore
        db = tmp_path / "h.db"
        HistoricalDataStore(db_path=db)
        with pytest.raises(ResearchDataUnavailable) as e:
            study_inputs(db, "QQQ", "2024-01-02", "2024-06-28")
        assert {"daily_prices", "implied_vol", "friction_params"} <= {g.input for g in e.value.gaps}

    def test_study_inputs_pass_on_a_complete_store(self, tmp_path):
        import sqlite3
        from ait.backtesting.study_harness import study_inputs
        db = tmp_path / "h.db"
        last = gi.seed(db)
        with sqlite3.connect(db) as conn:
            for table, extra in (("option_leg_spread_params", ""), ("option_leg_spread_band_params", ", '0-5'")):
                cols = "(symbol, form, a, b, n, n_sessions, calibrated_on" + (", dte_band" if extra else "") + ")"
                conn.execute(f"INSERT INTO {table} {cols} VALUES ('QQQ','linear',0.1,0.0,10,20,'2024-01-01'{extra})")
        rd, data, features, ivr = study_inputs(db, "QQQ", "2022-06-01", str(last))
        assert len(data) > 400 and "implied_vol" in data.columns

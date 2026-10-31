"""W88a: fixed research limits, the origin of every setting, and the run record that carries both.

  * ``config/invariants_research.py`` is the one home of the research engine's correctness constants; the engine,
    the optimizer's hard bounds and the log / run record read it, and a RESEARCH_GUIDE drift test keeps the table honest
  * ``research.settings_origin`` says which layer decides each value (study > overlay > config.yaml > default)
  * ``scripts/show_research_config.py`` prints that view
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ait.backtesting import run_record as rr
from ait.config import research as rs
from ait.config import invariants_research as ri
from ait.config.settings import load_settings

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_GUIDE = ROOT / "docs" / "RESEARCH_GUIDE.md"          # the research fixed-limits table lives there (generated)


def _engine(**kw):
    from ait.backtesting.engine import Backtester
    idx = pd.date_range("2024-01-02", periods=60, freq="B")
    close = 480.0 * np.cumprod(1 + np.full(60, 0.001))
    data = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                         "Volume": np.full(60, 5e6)}, index=idx)
    return Backtester(data=data, strategies=["short_strangle"], **kw)


class TestResearchInvariants:
    def test_the_registry_is_complete_and_unique(self):
        names = [i.name for i in ri.RESEARCH_INVARIANTS]
        assert len(names) == len(set(names)) == 11
        assert ri.research_invariant_values() == {
            "RESEARCH_DELTA_MIN": 0.05, "RESEARCH_DELTA_MAX": 0.45, "IV_SCALE_MIN": 0.5, "IV_SCALE_MAX": 1.5,
            "SETTLEMENT_LOSS_TOLERANCE": 1e-6, "PROB_CLIP_EPS": 1e-7,
            "RESEARCH_MAX_NULL_IV_RUN_SESSIONS": 3, "RESEARCH_MAX_DIVIDEND_GAP_DAYS": 100,
            "RESEARCH_MAX_EVENT_GAP_DAYS": 120,
            "RESEARCH_MAX_MACRO_GAP_DAYS": 5, "RESEARCH_MAX_MACRO_TAIL_DAYS": 10}   # W116: the data contract
        for i in ri.RESEARCH_INVARIANTS:
            assert i.what and i.why_fixed and i.enforced_in and len(i.why_fixed) > 20, i.name

    def test_the_guide_table_is_the_registry(self):
        m = re.search(r"<!-- BEGIN GENERATED: research_fixed_limits -->\n(.*?)<!-- END GENERATED: research_fixed_limits -->",
                      RESEARCH_GUIDE.read_text(), re.S)
        assert m, "docs/RESEARCH_GUIDE.md has no generated block 'research_fixed_limits'"
        sec = m.group(1)
        rows = re.findall(r"^\|\s*`([A-Z0-9_]+)`\s*\|\s*([0-9.]+)\s*\|", sec, re.M)
        by_name = {i.name: i.value for i in ri.RESEARCH_INVARIANTS}
        assert {n for n, _ in rows} == set(by_name)
        for name, shown in rows:
            assert float(shown) == float(by_name[name]), name

    def test_the_module_is_a_leaf_that_names_no_overlay(self):
        text = (ROOT / "src" / "ait" / "config" / "invariants_research.py").read_text()
        for banned in ("config_research", "use_research_config", "ait.config.research "):
            assert banned not in text
        mods = set()
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, ast.ImportFrom):
                mods.add(n.module)
        assert mods <= {"__future__", "ait.config.invariants", "ait.utils.logging"}

    def test_the_hard_bounds_follow_the_delta_band(self):
        from ait.optimization.param_spaces import HARD_BOUNDS
        assert HARD_BOUNDS["delta_short"] == HARD_BOUNDS["delta_long"] == (ri.RESEARCH_DELTA_MIN, ri.RESEARCH_DELTA_MAX)

    def _strikes(self, monkeypatch, **patch):
        """Run the strangle builder and return the |delta| the strike search was asked for."""
        import ait.backtesting.engine as eng
        from datetime import date
        asked = []
        real = eng.find_strike_by_delta
        with monkeypatch.context() as m:                       # undone per call, so cases do not stack
            for k, v in patch.items():
                m.setattr(eng, k, v)
            m.setattr(eng, "find_strike_by_delta",
                      lambda S, t, iv, d, *a, **k: (asked.append(abs(d)), real(S, t, iv, d, *a, **k))[1])
            bt = _engine(delta_short=0.30, delta_iv_scale=1.0, iv_floor=0.20)
            bt._build_credit_position("short_strangle", 480.0, 0.01, 30 / 365.0, 0.05, 30, date(2024, 3, 1), 100_000.0)
        return asked

    def test_the_engine_clamps_the_iv_scaled_delta_with_the_module_constants(self, monkeypatch):
        # iv 0.01 vs floor 0.20 -> scale clamps at IV_SCALE_MAX -> delta 0.30 * (1 + 1 * (scale - 1))
        assert max(self._strikes(monkeypatch)) == pytest.approx(0.45)                         # 0.30 * 1.5, at the band's top
        assert max(self._strikes(monkeypatch, IV_SCALE_MAX=1.2)) == pytest.approx(0.36)       # the scale clamp is the module's
        assert max(self._strikes(monkeypatch, RESEARCH_DELTA_MAX=0.40)) == pytest.approx(0.40)  # so is the delta ceiling

    def test_the_log_line_carries_the_limits_once_per_research_process(self, monkeypatch):
        events = []

        class Log:
            def info(self, event, **kw):
                events.append((event, kw))
        import ait.utils.logging as ulog
        monkeypatch.setattr(ulog, "get_logger", lambda name: Log())
        rs.use_research_config()
        ev = [kw for e, kw in events if e == "research_fixed_limits_in_effect"]
        assert len(ev) == 1 and ev[0]["limits"] == ri.research_invariant_values() and ev[0]["count"] == len(ri.RESEARCH_INVARIANTS)

    def test_the_run_record_carries_them(self):
        rec = rr.build_run_record(kind="backtest", name="x", settings=None, config_path=ROOT / "config.yaml",
                                  overlay_path=None)
        assert rec["research_invariants"] == ri.research_invariant_values()


@pytest.fixture
def cfg_dir(tmp_path):
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    shutil.copy(ROOT / "config_research.yaml", tmp_path / "config_research.yaml")
    return tmp_path


def _study(tmp_path, body="settings: {backtest: {max_concurrent_positions: 2}}\n"):
    p = tmp_path / "s.yaml"
    p.write_text("name: s\nquestion: q\nrule: exploratory\n" + body)
    return p


class TestSettingsOrigin:
    def test_each_layer_is_named(self, cfg_dir):
        got = rs.settings_origin(cfg_dir / "config.yaml", _study(cfg_dir))
        assert got["backtest.max_concurrent_positions"] == (2, "study")        # study beats config.yaml's 3
        assert got["backtest.gates_mode"] == ("enforce", "overlay")             # overlay beats the schema default
        assert got["trading.mode"][1] == "config.yaml"
        assert got["backtest.spread_base"][1] == "default"                       # nobody sets it

    def test_without_a_study_the_config_value_shows(self, cfg_dir):
        got = rs.settings_origin(cfg_dir / "config.yaml")
        assert got["backtest.max_concurrent_positions"] == (3, "config.yaml")

    def test_a_key_in_two_files_is_attributed_to_the_stronger_one(self, cfg_dir):
        (cfg_dir / "config_research.yaml").write_text(
            (cfg_dir / "config_research.yaml").read_text() + "\nrisk:\n  min_confidence: 0.61\n")
        assert rs.settings_origin(cfg_dir / "config.yaml")["risk.min_confidence"] == (0.61, "overlay")

    def test_secrets_never_appear(self, cfg_dir):
        got = rs.settings_origin(cfg_dir / "config.yaml")
        assert not any(k.startswith(("ibkr.", "api_keys.")) for k in got)
        assert not any("token" in k or "secret" in k for k in got)

    def test_the_view_matches_the_resolved_settings(self, cfg_dir):
        got = rs.settings_origin(cfg_dir / "config.yaml")
        s = rs.load_research_settings(cfg_dir / "config.yaml")
        assert got["backtest.friction_model"][0] == s.backtest.friction_model
        assert got["options.dte_range"][0] == list(s.options.dte_range)

    def test_the_run_record_stores_it(self, cfg_dir):
        origin = rs.settings_origin(cfg_dir / "config.yaml", _study(cfg_dir))
        rec = rr.build_run_record(kind="backtest", name="x", settings=None, config_path=cfg_dir / "config.yaml",
                                  overlay_path=None, settings_origin=origin)
        assert rec["settings_origin"]["backtest.max_concurrent_positions"] == [2, "study"]
        json.dumps(rec, default=str)                                           # round-trips


class TestShowResearchConfig:
    @staticmethod
    def _script():
        spec = importlib.util.spec_from_file_location("w88a_show", ROOT / "scripts" / "show_research_config.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_it_prints_values_origins_and_the_fixed_limits(self, capsys):
        assert self._script().main([]) == 0
        out = capsys.readouterr().out
        assert re.search(r"^backtest\.gates_mode\s+overlay\s+'enforce'", out, re.M)
        assert "RESEARCH_DELTA_MAX" in out and "setting(s) shown" in out

    def test_differs_hides_values_at_their_schema_default(self, capsys):
        self._script().main(["--differs", "--section", "backtest"])
        out = capsys.readouterr().out
        assert "backtest.gates_mode" in out and "backtest.spread_base" not in out

    def test_a_study_file_changes_the_view(self, tmp_path, capsys):
        self._script().main(["--study-file", str(_study(tmp_path)), "--section", "backtest"])
        assert re.search(r"^backtest\.max_concurrent_positions\s+study\s+2$", capsys.readouterr().out, re.M)

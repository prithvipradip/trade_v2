"""W108b: no script or module reads a setting W108 retired.

W108 removed ``backtest.position_size_pct``, ``positions.max_position_pct`` and
``backtest.spread_tuning.max_cost_fraction_of_capital`` (research and live now size from the one
budget ``risk.max_position_risk_pct``). Seventeen study scripts still read
``bc.position_size_pct`` and would have failed with an AttributeError the next time anyone ran them:
the test suite never imports a script, so nothing noticed. This AST scan is the missing net.

Legal and NOT flagged: the dataclass fields ``WalkForwardConfig.position_size_pct`` and the
``position_size_pct`` constructor parameters of ``Backtester`` / ``StrategyOptimizer`` (explicit study
overrides), reached through names such as ``self._config`` or ``window_cfg``.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RETIRED_EVERYWHERE = {"max_position_pct", "max_cost_fraction_of_capital"}
# names a BacktestConfig (the retired key's old home) is conventionally bound to in the scripts
BACKTEST_CONFIG_NAMES = {"bc", "_bc", "bcfg", "bt_cfg", "backtest_cfg"}


def _files() -> list[Path]:
    out = sorted((ROOT / "scripts").glob("*.py")) + sorted(ROOT.glob("run_*.py"))
    out += sorted((ROOT / "src").rglob("*.py"))
    return out


def violations(source: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr in RETIRED_EVERYWHERE:
            found.append((node.lineno, f".{node.attr}"))
        elif node.attr == "position_size_pct":
            base = node.value
            if (isinstance(base, ast.Name) and base.id in BACKTEST_CONFIG_NAMES) or (
                    isinstance(base, ast.Attribute) and base.attr == "backtest"):
                found.append((node.lineno, "backtest.position_size_pct"))
    return found


class TestTheScan:
    def test_it_flags_the_retired_read_that_broke_the_scripts(self):
        assert violations("x = WalkForwardConfig(position_size_pct=bc.position_size_pct)")
        assert violations("y = settings.backtest.position_size_pct")
        assert violations("z = s.positions.max_position_pct")
        assert violations("w = s.backtest.spread_tuning.max_cost_fraction_of_capital")

    def test_it_leaves_the_legal_uses_alone(self):
        assert not violations("self.position_size_pct = resolve_position_size_pct(self.position_size_pct, s)")
        assert not violations("a = window_cfg.position_size_pct * 2")
        assert not violations("b = self._config.position_size_pct")
        assert not violations("c = WalkForwardConfig(position_size_pct=None)")


class TestTheTree:
    def test_no_script_or_module_reads_a_retired_setting(self):
        bad = {}
        for path in _files():
            hits = violations(path.read_text(encoding="utf-8"))
            if hits:
                bad[str(path.relative_to(ROOT))] = hits
        assert not bad, f"these read a setting W108 retired: {bad}"

    def test_the_scan_covers_the_scripts_it_was_written_for(self):
        names = {p.name for p in _files()}
        assert {"wing_budget_study.py", "post_event_study.py", "take_profit_study.py",
                "run_integration_test.py", "settings.py", "engine.py"} <= names

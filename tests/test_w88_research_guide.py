"""docs/RESEARCH_GUIDE.md stays true: its links resolve, the files and settings it names exist, its learner table equals the
schema defaults, and its study example loads. A guide that drifts from the code is worse than none."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from ait.config.research import load_study
from ait.config.settings import LearnerConfig, ObjectivesConfig, WalkForwardTuningConfig

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "RESEARCH_GUIDE.md"
TEXT = GUIDE.read_text()


def _slug(heading: str) -> str:
    s = re.sub(r"[^\w\s-]", "", heading.lower())
    return re.sub(r"\s", "-", s.strip())


class TestLinks:
    def test_anchors_resolve_to_headings(self):
        slugs = {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+)$", TEXT, re.M)}
        anchors = re.findall(r"\]\(#([^)]+)\)", TEXT)
        assert anchors
        assert [a for a in anchors if a not in slugs] == []

    def test_relative_links_exist(self):
        links = [l for l in re.findall(r"\]\(([^)#]+?)(?:#[^)]*)?\)", TEXT) if not l.startswith("http")]
        assert [l for l in links if not (GUIDE.parent / l).exists()] == []

    def test_named_files_exist(self):
        names = set(re.findall(r"`((?:scripts|studies|src/ait)/[\w/.\-<>]+?\.(?:py|yaml))`", TEXT))
        names |= set(re.findall(r"`(run_(?:backtest|optimizer)\.py)`", TEXT))
        real = {n for n in names if "<" not in n}
        assert real
        assert [n for n in real if not (ROOT / n).exists() and not any(ROOT.glob(f"src/ait/**/{Path(n).name}"))] == []

    def test_named_runbook_sections_exist(self):
        runbook = (ROOT / "docs" / "RUNBOOK.md").read_text()
        for title in ("Studies and run records", "Research tunables", "Research fixed limits", "Model hyperparameters",
                      "One-sided entry gates", "Research-vs-reality parity replay", "Searching the entry DTE"):
            assert re.search(rf"^## {re.escape(title)}", runbook, re.M), title


class TestContent:
    def test_the_learner_table_is_the_schema(self):
        rows = re.findall(r"^\| `(\w+)` \| ([0-9.]+) \|", TEXT.split("**Learner settings**")[1].split("\n---\n")[0], re.M)
        assert len(rows) >= 14
        defaults = LearnerConfig()
        for name, shown in rows:
            assert float(shown) == pytest.approx(float(getattr(defaults, name))), name

    def test_the_quoted_defaults_match(self):
        w, o = WalkForwardTuningConfig(), ObjectivesConfig()
        assert (w.warm_start_min_win_rate, w.warm_start_min_trades) == (0.75, 3) and "0.75" in TEXT
        assert (o.sharpe_weight, o.win_rate_weight, o.drawdown_penalty) == (0.4, 0.4, 0.2)
        assert "0.4 × Sharpe + 0.4 × win rate − 0.2" in TEXT

    def test_every_research_only_block_is_mentioned(self):
        for block in ("learner", "legacy_engine", "objectives", "walkforward_tuning"):
            assert f"`{block}`" in TEXT, block

    def test_the_walk_forward_defaults_quoted_are_the_real_ones(self):
        from ait.backtesting.walkforward import WalkForwardConfig
        c = WalkForwardConfig(train_window_models=False)
        assert (c.train_days, c.gap_days, c.test_days, c.step_days) == (365, 5, 63, 63)
        assert "train 365 days, a 5-day purge gap, test 63 days, step 63 days" in TEXT

    def test_the_study_example_loads(self, tmp_path):
        block = re.search(r"```yaml\n(name: .*?)```", TEXT, re.S).group(1)
        p = tmp_path / "ex.yaml"
        p.write_text(block)
        s = load_study(p)
        assert s.exploratory and s.search_space == {"iron_condor": {"wing_k": [1.0, 2.0]}}

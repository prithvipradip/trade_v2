"""docs/GUIDE.md stays true: its links and anchors resolve, the files, settings and identifiers it names exist, the supervisor
jobs it describes are the jobs that are registered, retired subsystems and experiment results stay out of it, and it keeps telling the
reader that its values are examples. A guide that drifts from the code is worse than none.

It does NOT assert any setting's value: the guide shows values only as examples, and the config files are the authority."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "GUIDE.md"
TEXT = GUIDE.read_text()
SRC_TEXT = "\n".join(p.read_text(errors="ignore") for p in (ROOT / "src" / "ait").rglob("*.py")) + "\n" + "\n".join(
    p.read_text(errors="ignore") for p in ROOT.glob("*.py"))


def _slug(heading: str) -> str:
    s = re.sub(r"[^\w\s-]", "", heading.lower())
    return re.sub(r"\s", "-", s.strip())


def _slugs(text: str) -> set[str]:
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+)$", text, re.M)}


# Runtime artefacts: named in the guide but created while the system runs, so they need not exist in a checkout.
RUNTIME = ("data/", "models/", "logs/", "reports/", ".retrained", "RESTRICTED.txt", "HALT", "bot_heartbeat", "orchestrator.lock",
           "ib_endpoint.json", "thompson_state.json", "counterfactual_log.json", "KEEPER_PAUSE", "ensemble_v-", "ait_state.db",
           "ait_analytics.duckdb", "historical.db", "ensemble.pkl", "range.pkl", "vol_magnitude.pkl", "meta_label.pkl")


class TestLinks:
    def test_anchors_resolve_to_headings(self):
        anchors = re.findall(r"\]\(#([^)]+)\)", TEXT)
        assert anchors
        assert [a for a in anchors if a not in _slugs(TEXT)] == []

    def test_relative_links_exist(self):
        links = [l for l in re.findall(r"\]\(([^)#\s]+?)(?:#[^)]*)?\)", TEXT) if not l.startswith("http")]
        assert links
        assert [l for l in links if not (GUIDE.parent / l).exists()] == []

    def test_anchors_into_other_documents_resolve(self):
        pairs = re.findall(r"\]\(([\w./]+\.md)#([^)]+)\)", TEXT)
        for doc, anchor in pairs:
            assert anchor in _slugs((GUIDE.parent / doc).read_text()), (doc, anchor)

    def test_named_files_exist(self):
        names = set(re.findall(r"`([\w\-]+(?:/[\w.\-]+)+\.(?:py|yaml|md|toml|bat|yml))`", TEXT))       # paths with a directory
        bare = set(re.findall(r"`([\w\-]+\.(?:py|yaml|md|toml|bat))`", TEXT))                          # bare file names
        bare -= {"CLAUDE.md"}                      # git-ignored, local to a checkout: the guide may name it, a clean clone may lack it
        assert names and bare
        for n in names:
            assert (ROOT / n).exists() or (ROOT / "src" / "ait" / n).exists(), n
        for n in bare:
            assert (ROOT / n).exists() or any(ROOT.glob(f"**/{n}")), n

    def test_named_runbook_sections_exist(self):
        runbook = (ROOT / "docs" / "RUNBOOK.md").read_text()
        titles = ("Kill switches", "Alert → action", "Manual intervention procedure", "Daily 2-minute routine", "Deploy checklist")
        for title in titles:
            assert re.search(rf"^##+ {re.escape(title)}", runbook, re.M), title
            assert title in TEXT, f"the guide no longer names RUNBOOK section {title!r}; update this list"

    def test_named_research_guide_sections_exist(self):
        rg = (ROOT / "docs" / "RESEARCH_GUIDE.md").read_text()
        for title in ("Where each field comes from",):
            assert re.search(rf"^##+ {re.escape(title)}", rg, re.M), title
            assert title in TEXT, title

    def test_the_walkforward_document_links_here(self):
        wf = (ROOT / "docs" / "WALKFORWARD_DASHBOARD.md").read_text()
        assert "](GUIDE.md)" in wf and "](../GUIDE.md)" not in wf


class TestNamedThingsExist:
    def test_every_setting_it_names_is_in_the_schema(self):
        from ait.config.settings import Settings
        keys = set(re.findall(
            r"`((?:trading|account|positions|position_sizing|risk|options|backtest|ml|regime|meta_label|earnings|dividends|"
            r"assignment_response|condor_defense|exit|learning|notifications|dashboard|logging)\.[a-z_]+(?:\.[a-z_]+)*)`", TEXT))
        keys = {k for k in keys if not k.endswith((".pkl", ".db", ".py", ".md", ".yaml"))}      # model / store file names, not settings
        assert len(keys) >= 15

        def known(parts: list[str]) -> bool:
            model = Settings
            for i, part in enumerate(parts):
                fields = getattr(model, "model_fields", None)
                if not fields or part not in fields:
                    return False
                model = fields[part].annotation
            return True
        assert sorted(k for k in keys if not known(k.split("."))) == []

    def test_every_constant_it_names_exists_in_the_code(self):
        consts = set(re.findall(r"`([A-Z][A-Z0-9_]{4,})`", TEXT)) - {"HALT_UNTRACKED", "KEEPER_PAUSE"}    # control FILES, created at run time
        assert len(consts) >= 8
        assert sorted(c for c in consts if not re.search(rf"\b{c}\b", SRC_TEXT)) == []

    def test_every_class_and_function_it_names_exists_in_the_code(self):
        """Backticked CamelCase names and snake_case identifiers (after splitting `Class.method`) must occur in the source or the config."""
        corpus = SRC_TEXT + (ROOT / "config.yaml").read_text() + (ROOT / "config_research.yaml").read_text()
        idents = set()
        for tok in re.findall(r"`([A-Za-z_][\w.]*)(?:\(\))?`", TEXT):
            if tok.endswith((".py", ".md", ".yaml", ".pkl", ".db", ".bat", ".toml")):
                continue
            for seg in tok.split("."):
                if re.fullmatch(r"[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+", seg) or re.fullmatch(r"_?[a-z][a-z0-9]*(?:_[a-z0-9]+)+", seg):
                    idents.add(seg)
        assert len(idents) >= 15
        assert sorted(i for i in idents if not re.search(rf"\b{re.escape(i)}\b", corpus)) == []

    def test_the_selector_registry_it_describes(self):
        from ait.strategies.selector import STRATEGY_MAP
        assert {"iron_condor", "bull_call_spread", "bear_put_spread", "long_call", "long_put", "long_straddle", "short_strangle"} <= set(STRATEGY_MAP)

    def test_the_feature_catalogue_it_points_to(self):
        from ait.config import feature_names as fn
        assert fn.CATALOGUE and fn.OFF_BY_DEFAULT


class TestSupervisorJobs:
    """Section 3 documents the supervisor's scheduled jobs. If a job is added or removed, update the table and this set."""

    DOCUMENTED = {"health_check", "daily_retrain", "db_backup", "weekly_scorecard", "digest_am", "digest_pm", "daily_report",
                  "shared_logs_publish", "shared_logs_fetch", "weekly_backtest", "monthly_cleanup", "daily_log_rotation"}

    def test_the_registered_jobs_are_the_documented_ones(self):
        from ait.orchestration import master

        class Sched:
            def __init__(self):
                self.ids = []

            def add_job(self, fn, trigger=None, **kw):
                self.ids.append(kw.get("id"))

        s = Sched()
        master._register_recurring_jobs(s, bot=object(), web=object())
        assert set(s.ids) - {"short_dte_collection"} == self.DOCUMENTED, "update GUIDE.md section 3 and DOCUMENTED"

    def test_the_guide_lists_each_documented_job(self):
        for job_id in self.DOCUMENTED:
            assert f"`{job_id}`" in TEXT, job_id


class TestContent:
    def test_it_says_its_values_are_examples_and_where_the_real_ones_are(self):
        head = TEXT[:2500]
        assert "examples" in head and "config.yaml" in head and "config_research.yaml" in head

    def test_retired_subsystems_and_experiment_results_stay_out(self):
        banned = ("FinBERT", "finbert", "sentiment engine", "Sentiment Engine", "Sharpe", "ablation", "MLflow", "Fear & Greed",
                  "+183", "OOS return", "win rate of")
        assert [b for b in banned if b in TEXT] == []
        assert re.findall(r"\bExp(?:eriment)? ?\d+", TEXT) == []

    def test_no_audit_round_or_fix_wave_ids(self):
        assert re.findall(r"\b(?:R\d{1,2}|W\d{2,3})\b", TEXT) == []

    def test_no_secrets_or_account_ids(self):
        assert not re.search(r"\b(?:DU|U)\d{6,}\b", TEXT)
        assert not re.search(r"(?i)(?:api[_-]?key|token|password)\s*[:=]\s*\S{8,}", TEXT)

    def test_the_old_guide_is_untouched_history(self):
        old = ROOT / "deprecated" / "docs" / "GUIDE.md"
        assert old.exists() and "STALE" in old.read_text()[:600]

"""The documentation set stays true: links and anchors resolve, pointers from code and config land on a real heading,
the named files exist, user-facing documents carry no audit-round / fix-wave IDs, and every document is on the README map.

Generated blocks and the two settings references are checked by ``tests/test_docs_generated.py``.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Documents that are maintained and read by users. docs/PR6_CODE_REVIEW.md is a loose review note, docs/archive/ is
# point-in-time and docs/reference/ is generated.
USER_DOCS = ["GETTING_STARTED", "GUIDE", "RUNBOOK", "MODELS", "RESEARCH_GUIDE", "WALKFORWARD_DASHBOARD", "GLOSSARY"]
# Only these may mention audit rounds (R<n>) and fix waves (W<n>).
ID_FREE = USER_DOCS + ["README"]
ALL_DOCS = USER_DOCS + ["STUDIES_FINDINGS", "TECHNICAL_GUIDE"]


def _path(name: str) -> Path:
    return ROOT / "README.md" if name == "README" else DOCS / f"{name}.md"


def _strip_fences(text: str) -> str:
    return re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    h = re.sub(r"<[^>]+>", "", heading).strip().lower()
    h = h.replace("`", "")
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def _headings(path: Path) -> list[str]:
    return [m.group(2).strip() for m in re.finditer(r"^(#{1,6})\s+(.*?)\s*$", _strip_fences(path.read_text()), re.M)]


def _anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    for h in _headings(path):
        s = _slug(h)
        n = seen.get(s, 0)
        seen[s] = n + 1
        out.add(s if n == 0 else f"{s}-{n}")
    return out


def _bold_leads(path: Path) -> list[str]:
    """Phrases that open a paragraph in bold: ``**How is Sharpe annualized?**`` or ``**Lifecycle capture (...): ...**``."""
    return [m.group(1).strip() for m in re.finditer(r"^\*\*(.+?)\*\*", _strip_fences(path.read_text()), re.M)]


def _norm(text: str) -> str:
    return re.sub(r"[`*_]", "", " ".join(text.split())).strip().lower()


def _link_targets(path: Path):
    text = _strip_fences(path.read_text())
    text = re.sub(r"`[^`\n]*`", "", text)               # a link shown as code is not a link
    for m in re.finditer(r"\]\(([^)\s]+)\)", text):
        yield m.group(1)


def _all_md() -> list[Path]:
    return ([_path(n) for n in ALL_DOCS + ["README"]] + sorted((DOCS / "archive").glob("*.md"))
            + sorted((DOCS / "reference").glob("*.md")))


@pytest.mark.parametrize("name", ALL_DOCS + ["README"])
def test_document_exists(name):
    assert _path(name).exists(), f"{name} is missing"


@pytest.mark.parametrize("path", _all_md(), ids=lambda p: p.name)
def test_relative_links_and_anchors_resolve(path):
    bad = []
    for target in _link_targets(path):
        if re.match(r"^(https?:|mailto:)", target):
            continue
        file_part, _, anchor = target.partition("#")
        dest = path if not file_part else (path.parent / file_part).resolve()
        if "archive" in path.parts and dest.suffix != ".md":
            continue                      # the archive is point-in-time: its links to code that has since moved are history
        if not dest.exists():
            bad.append(f"{target}: file not found")
            continue
        if anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
            bad.append(f"{target}: no heading with that anchor in {dest.name}")
    assert not bad, f"{path.name}: " + "; ".join(bad)


# Code and config point at a document section as:   docs/<FILE>.md "<Heading>"
POINTER = re.compile(r"docs/([A-Za-z_]+)\.md\s*(?:[,:(])?\s*(?:\\?[\"'])([^\"'\\\n]+?)\\?[\"']")
POINTER_DIRS = ["src", "scripts", "tests", "studies", ".github"]
POINTER_FILES = ["config.yaml", "config_research.yaml", "run_orchestrator.py", "run_backtest.py", "run_optimizer.py",
                 "status.py", "status_server.py", "web_logs.py"]


def _pointer_sources():
    for d in POINTER_DIRS:
        for p in (ROOT / d).rglob("*"):
            if p.suffix in {".py", ".yaml", ".yml"} and "egg-info" not in str(p) and "__pycache__" not in str(p):
                yield p
    for f in POINTER_FILES:
        if (ROOT / f).exists():
            yield ROOT / f


def _resolves(doc: Path, phrase: str) -> bool:
    want = _norm(phrase)
    cands = [_norm(h) for h in _headings(doc)] + [_norm(b) for b in _bold_leads(doc)]
    # the phrase may be a heading, or the start of a heading / bold lead ("Lifecycle capture (--lifecycle): ...")
    return any(c == want or c.startswith(want) or re.sub(r"^\d+(\.\d+)*\.?\s+", "", c).startswith(want) for c in cands)


def test_pointers_from_code_and_config_land_on_a_heading():
    bad = []
    for src in _pointer_sources():
        if src.name == "test_docs.py":
            continue
        text = src.read_text(errors="ignore")
        for m in POINTER.finditer(text):
            name, phrase = m.group(1), m.group(2)
            doc = DOCS / f"{name}.md"
            if not doc.exists():
                bad.append(f"{src.relative_to(ROOT)}: docs/{name}.md does not exist")
            elif not _resolves(doc, phrase):
                bad.append(f'{src.relative_to(ROOT)}: docs/{name}.md has no heading "{phrase}"')
    assert not bad, "\n".join(bad)


def test_code_does_not_point_at_the_git_ignored_claude_md():
    """CLAUDE.md is git-ignored (local to a checkout), so nothing committed outside the tests may rely on it."""
    bad = []
    for src in _pointer_sources():
        if "tests" in src.relative_to(ROOT).parts:
            continue
        for i, line in enumerate(src.read_text(errors="ignore").splitlines(), 1):
            if re.search(r"\bCLAUDE\.md\b", line):
                bad.append(f"{src.relative_to(ROOT)}:{i}")
    assert not bad, "pointers to the git-ignored CLAUDE.md: " + ", ".join(bad)


ID = re.compile(r"(?<![\w/.\-])[WR]\d{1,3}[a-z]?(?![\w\-])")


@pytest.mark.parametrize("path", [_path(n) for n in ID_FREE] + sorted((DOCS / "reference").glob("*.md")), ids=lambda p: p.name)
def test_user_documents_carry_no_round_or_wave_ids(path):
    name = path.name
    text = re.sub(r"`[^`\n]*`", "", _strip_fences(path.read_text()))
    hits = sorted(set(ID.findall(text)))
    assert not hits, f"{name}: audit-round / fix-wave IDs belong in code, PLAN.md, TECHNICAL_GUIDE and STUDIES_FINDINGS, not here: {hits}"


@functools.lru_cache(maxsize=1)
def _code_text() -> str:
    roots = [ROOT / "src" / "ait", ROOT / "scripts"]
    files = [p for r in roots for p in r.rglob("*.py")] + list(ROOT.glob("*.py"))
    return "\n".join(p.read_text(errors="ignore") for p in files)


NAMED_PATH = re.compile(r"`((?:src/ait|scripts|tests|docs|studies|config|ml|data|risk|execution|broker|bot|strategies|backtesting)/[\w./\-]+\.(?:py|yaml|md|json|js|jsx))`")


@pytest.mark.parametrize("name", ALL_DOCS)
def test_files_named_in_documents_exist(name):
    text = _path(name).read_text()
    # `config/settings.py` is written relative to the package (src/ait); everything else relative to the repository
    # data/ is git-ignored runtime state: a file the system writes does not exist in a fresh checkout (or in CI), so for data/
    # the check is that the code names the file, not that this machine happens to have it
    exists = lambda p: ((ROOT / p).exists() or (ROOT / "src" / "ait" / p).exists()  # noqa: E731
                        or (p.startswith("data/") and Path(p).name in _code_text()))
    missing = sorted({p for p in NAMED_PATH.findall(text) if not exists(p) and "<" not in p and "*" not in p})
    assert not missing, f"{name} names files that do not exist: {missing}"


@pytest.mark.parametrize("name", USER_DOCS + ["TECHNICAL_GUIDE", "STUDIES_FINDINGS"])
def test_each_document_says_who_it_is_for(name):
    head = "\n".join(_path(name).read_text().splitlines()[:25])
    assert re.search(r"Not for|See also|Terms:", head), f"{name}: the opening should say who it is for / not for and link its neighbours"


def test_readme_map_lists_every_maintained_document():
    readme = (ROOT / "README.md").read_text()
    missing = [n for n in ALL_DOCS if f"docs/{n}.md" not in readme]
    assert not missing, f"README.md's document map is missing: {missing}"
    assert all(f"docs/reference/{n}.md" in readme for n in ("SETTINGS_LIVE", "SETTINGS_RESEARCH", "MODULES", "SCRIPTS"))


def test_study_results_live_only_in_studies_findings():
    """The rule is behavioural; the cheap tripwire is that the migrated-results register is the only document with a findings
    index, and the other documents link to it rather than carrying a result table of their own."""
    for name in ["RUNBOOK", "GUIDE", "GETTING_STARTED"]:
        text = _path(name).read_text()
        assert not re.search(r"^#+\s.*(findings|results register)", text, re.I | re.M), f"{name} has a results section"


PROVENANCE = re.compile(r"audit\s+(item|20\d\d|round)|Finding [a-z]+-[a-z]|\bMEASURED\b|\b20\d\d-\d\d-\d\d\b|\bphantom\b|\bincident\b")
# Dates that are data, not history: the drill log, and the research calendar's first day.
PROVENANCE_OK = ("Backup restore", "history_start_date", "research calendar starts")


@pytest.mark.parametrize("path", [_path(n) for n in ID_FREE] + sorted((DOCS / "reference").glob("*.md")), ids=lambda p: p.name)
def test_user_documents_and_references_carry_no_audit_history_or_dated_measurements(path):
    """Descriptions copied from the schema used to leak 'audit 2026-07-07 item 3.3', 'Finding <slug>:' prefixes and dated
    measurements into the generated references; behaviour is described without its history, results live in STUDIES_FINDINGS."""
    hits = [f"{i}: {m.group(0)}" for i, line in enumerate(path.read_text().splitlines(), 1)
            if (m := PROVENANCE.search(line)) and not any(ok in line for ok in PROVENANCE_OK)]
    assert not hits, f"{path.name}: {hits}"


def test_scripts_reference_explained_in_links_resolve_and_name_the_script():
    """Each row of docs/reference/SCRIPTS.md may point at the document that explains the script in context. The document must exist,
    the heading must exist, and the document must actually mention the script (otherwise the pointer sends the reader nowhere)."""
    import yaml
    cat = yaml.safe_load((ROOT / "scripts" / "script_catalogue.yaml").read_text())
    bad = []
    for rel, e in cat.items():
        m = re.match(r"\[[^\]]+\]\(\.\./([A-Z_]+\.md)(?:#([^)]+))?\)$", e["explained"])
        if not m:
            assert e["explained"] == "-" or e["explained"].startswith("PLAN.md"), f"{rel}: unrecognised pointer {e['explained']!r}"
            continue
        doc = DOCS / m.group(1)
        if not doc.exists():
            bad.append(f"{rel}: {m.group(1)} does not exist")
            continue
        if m.group(2) and m.group(2) not in _anchors(doc):
            bad.append(f"{rel}: no heading #{m.group(2)} in {m.group(1)}")
        name = Path(rel).name
        if name not in doc.read_text() and name.rsplit(".", 1)[0] not in doc.read_text():
            bad.append(f"{rel}: {m.group(1)} never mentions it")
    assert not bad, bad


def test_the_technical_guide_states_the_script_catalogue_rules():
    text = _path("TECHNICAL_GUIDE").read_text()
    for needle in ("scripts/script_catalogue.yaml", "--restamp", "reference/SCRIPTS.md"):
        assert needle in text, needle


def test_guide_section_0_points_at_the_scripts_reference():
    text = _path("GUIDE").read_text()
    section = text.split("## 0.", 1)[1].split("\n## 1.", 1)[0]
    assert "reference/SCRIPTS.md" in section

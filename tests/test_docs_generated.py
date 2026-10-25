"""The generated parts of docs/ equal the code (scripts/gen_docs.py): fixed limits, risk ceilings, the order timeline, the supervisor
schedule, the go-live criteria, and the reference files. If this fails, run ``python scripts/gen_docs.py`` and review the diff."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gen():
    spec = importlib.util.spec_from_file_location("gen_docs", ROOT / "scripts" / "gen_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generated_blocks_and_files_are_up_to_date():
    gen = _gen()
    targets = {**gen.render_documents(), **gen.generated_files()}
    stale = [str(p.relative_to(ROOT)) for p, new in targets.items() if not p.exists() or p.read_text() != new]
    assert stale == [], f"out of date: {stale}; run python scripts/gen_docs.py"


def test_every_block_marker_is_known_and_paired():
    gen = _gen()
    for path in (ROOT / "docs").glob("*.md"):
        text = path.read_text()
        begins = re.findall(r"<!-- BEGIN GENERATED: (\w+) -->", text)
        ends = re.findall(r"<!-- END GENERATED: (\w+) -->", text)
        assert begins == ends, path.name
        assert set(begins) <= set(gen.GENERATORS), (path.name, set(begins) - set(gen.GENERATORS))


def test_a_generated_block_is_not_hand_edited_into_a_second_document():
    """One home per generated table: each block name appears in at most one document."""
    seen: dict[str, str] = {}
    for path in sorted((ROOT / "docs").glob("*.md")):
        for name in re.findall(r"<!-- BEGIN GENERATED: (\w+) -->", path.read_text()):
            assert name not in seen, f"{name} is in both {seen[name]} and {path.name}"
            seen[name] = path.name


def test_the_guide_has_the_live_blocks():
    names = set(re.findall(r"<!-- BEGIN GENERATED: (\w+) -->", (ROOT / "docs" / "GUIDE.md").read_text()))
    assert {"fixed_limits", "risk_ceilings", "order_timeline", "supervisor_jobs", "go_live_gates"} <= names


def test_every_setting_has_a_description_in_the_schema():
    """The generated references take each setting's meaning from its schema description. A setting with none shows a dash there,
    so a new setting must say what it does (a MEANING entry in scripts/gen_docs.py is only for text the schema cannot carry)."""
    gen = _gen()
    missing = [k for k in gen._all_leaves() if not (gen._field(k).description or k in gen.MEANING)]
    assert not missing, f"settings without a description: {missing}"


# ---- scripts reference (docs/reference/SCRIPTS.md) ------------------------------------------------------------------------

def test_every_script_has_a_catalogue_entry_and_none_is_stale():
    """scripts/script_catalogue.yaml is the hand-checked half of docs/reference/SCRIPTS.md: a script without an entry (or an entry
    without a script) must fail here, so a new script cannot be added undocumented."""
    gen = _gen()
    gen._check_catalogue(gen.load_catalogue())          # raises SystemExit naming the missing / extra entry
    files = set(gen.script_files())
    on_disk = {f"scripts/{p.name}" for p in (ROOT / "scripts").glob("*.py") if p.name != "__init__.py"}
    assert on_disk <= files and set(gen.ROOT_ENTRY_POINTS) <= files
    for rel in gen.ROOT_ENTRY_POINTS:
        assert (ROOT / rel).exists(), rel


def test_a_missing_or_extra_catalogue_entry_is_refused():
    import pytest
    gen = _gen()
    cat = gen.load_catalogue()
    missing = {k: v for k, v in cat.items() if k != "scripts/smoke_deploy.py"}
    with pytest.raises(SystemExit, match="smoke_deploy"):
        gen._check_catalogue(missing)
    with pytest.raises(SystemExit, match="no_such_script"):
        gen._check_catalogue({**cat, "scripts/no_such_script.py": cat["scripts/smoke_deploy.py"]})


def test_the_ib_setting_must_agree_with_the_scripts_imports():
    """A script that imports an IB client cannot be catalogued as 'no'; one that does not cannot be 'yes'."""
    import pytest
    gen = _gen()
    cat = gen.load_catalogue()
    assert gen._script_facts("scripts/backfill_intraday.py")[1] is True
    assert gen._script_facts("scripts/smoke_deploy.py")[1] is False
    with pytest.raises(SystemExit, match="imports an IB client"):
        gen._check_catalogue({**cat, "scripts/backfill_intraday.py": {**cat["scripts/backfill_intraday.py"], "needs_ib": "no"}})
    with pytest.raises(SystemExit, match="imports no IB client"):
        gen._check_catalogue({**cat, "scripts/smoke_deploy.py": {**cat["scripts/smoke_deploy.py"], "needs_ib": "yes"}})


def test_catalogue_groups_and_fields_are_closed_sets():
    import pytest
    gen = _gen()
    cat = gen.load_catalogue()
    with pytest.raises(SystemExit, match="unknown group"):
        gen._check_catalogue({**cat, "scripts/smoke_deploy.py": {**cat["scripts/smoke_deploy.py"], "group": "Misc"}})
    with pytest.raises(SystemExit, match="fields must be exactly"):
        gen._check_catalogue({**cat, "scripts/smoke_deploy.py": {**cat["scripts/smoke_deploy.py"], "notes": "x"}})
    assert {e["group"] for e in cat.values()} <= set(gen.SCRIPT_GROUPS)


def test_script_code_matches_the_stamp_taken_when_its_catalogue_entry_was_checked():
    """The catalogue wording is hand-checked against the code. When a script's code changes this fails, naming the script: re-read its
    entry in scripts/script_catalogue.yaml against the change, correct it, then
    ``python scripts/gen_docs.py --restamp scripts/<name>.py``. Restamping without reading the entry defeats the check."""
    gen = _gen()
    stale = gen.stale_stamps()
    assert stale == [], (f"code changed since the catalogue entry was checked: {stale}. Re-read each entry against the code, correct it, "
                         f"then: python scripts/gen_docs.py --restamp {' '.join(stale)}")


def test_the_stamp_ignores_docstrings_and_comments_but_not_code(tmp_path, monkeypatch):
    gen = _gen()
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    f = tmp_path / "s.py"
    f.write_text('"""Doc."""\n# a comment\ndef main():\n    """Inner doc."""\n    return 1\n')
    base = gen.code_stamp("s.py")
    f.write_text('"""A different, longer docstring.\n\nWith more lines."""\n\n# another comment\n# and one more\ndef main():\n    """Changed."""\n    return 1\n')
    assert gen.code_stamp("s.py") == base
    f.write_text('"""Doc."""\ndef main():\n    return 2\n')
    assert gen.code_stamp("s.py") != base
    b = tmp_path / "k.bat"
    b.write_text("@echo off\nREM note\n:: other\npython run.py\n")
    bat = gen.code_stamp("k.bat")
    b.write_text("@echo off\nrem different note\n\npython run.py\n")
    assert gen.code_stamp("k.bat") == bat
    b.write_text("@echo off\npython other.py\n")
    assert gen.code_stamp("k.bat") != bat


def test_restamp_rewrites_only_the_named_entry_and_keeps_the_files_comments(tmp_path, monkeypatch):
    gen = _gen()
    cat = tmp_path / "cat.yaml"
    cat.write_text("# header comment\n\na.py:\n  group: Developer\n  stamp: \"\"\n\nb.py:\n  group: Developer\n  stamp: \"\"\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.py").write_text("y = 2\n")
    monkeypatch.setattr(gen, "CATALOGUE", cat)
    monkeypatch.setattr(gen, "ROOT", tmp_path)
    monkeypatch.setattr(gen, "script_files", lambda: ["a.py", "b.py"])
    gen.restamp(["a.py"])
    text = cat.read_text()
    assert text.startswith("# header comment") and f'stamp: "{gen.code_stamp("a.py")}"' in text
    assert text.count('stamp: ""') == 1                  # b.py untouched


def test_the_scripts_reference_lists_every_script_with_a_group_and_no_raw_angle_brackets():
    gen = _gen()
    text = (ROOT / "docs" / "reference" / "SCRIPTS.md").read_text()
    for rel in gen.script_files():
        assert f"`{rel}`" in text, rel
    for line in text.splitlines():
        if line.startswith("| `"):
            cells = line.split(" | ")
            # outside code spans a placeholder such as <run_id> must be escaped or the page drops it as an HTML tag
            for i, c in enumerate(cells[1:4] + cells[4:5]):
                outside = "".join(part for k, part in enumerate(c.split("`")) if k % 2 == 0)
                assert not re.search(r"<(?!br>)[A-Za-z_/]", outside), line[:80]

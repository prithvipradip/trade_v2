"""W88e: the literal-census class guard.

Numeric literals that decide a research result must be in a setting a researcher can see (``config_research.yaml`` / the schema), or in
``config/invariants_research.py``, or be a deliberate, classified exception in ``tests/data/literal_allowlist.json``. This test compares
the allowlist with the code (``scripts/literal_census.py``): a new literal, a changed literal, a removed literal and a stale entry all fail,
so every change to the research code's inline numbers is seen and classified by a person. It is the guard for the *next* instance of the
class W88 b-d fixed by hand.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import literal_census as lc  # noqa: E402


@pytest.fixture(scope="module")
def census():
    return lc.census()


@pytest.fixture(scope="module")
def allow():
    return lc.load_allowlist()


class TestTheAllowlistIsTheCode:
    def test_code_and_allowlist_agree(self, census, allow):
        problems = lc.differences(census, allow)
        assert problems == [], (
            "numeric literals in the research code changed. Convert a research number to a setting "
            "(schema + config_research.yaml) or to config/invariants_research.py; if it is a deliberate exception run "
            "`python scripts/literal_census.py --update`, classify the new entry and add a note.\n  " + "\n  ".join(problems))

    def test_every_entry_has_a_valid_category_and_follow_ups_have_a_note(self, allow):
        bad = []
        for key, e in allow["entries"].items():
            if e.get("category") not in lc.CATEGORIES:
                bad.append(f"{key}: category {e.get('category')!r}")
            elif e["category"] == "registered_followup" and len(e.get("note", "")) < 15:
                bad.append(f"{key}: a registered follow-up needs a note saying what it is and where it should go")
            elif e["category"] == "registered_followup" and e["note"].startswith("W88 census baseline"):
                bad.append(f"{key}: the generated placeholder note must be replaced by a real one")
        assert bad == []

    def test_every_entry_lists_its_literals_sorted_and_non_trivial(self, allow):
        for key, e in allow["entries"].items():
            assert e["values"] == sorted(e["values"]), key
            assert not [v for v in e["values"] if lc._is_trivial(v)], key


class TestEveryModuleIsCounted:
    def test_listed_modules_exist(self):
        assert [m for m in lc.MODULES if not (ROOT / m).is_file()] == []

    def test_a_new_module_in_a_watched_directory_must_be_listed(self):
        listed = set(lc.MODULES) | set(lc.EXCLUDED)
        unlisted = [p.relative_to(ROOT).as_posix() for d in lc.WATCHED_DIRS for p in sorted((ROOT / d).glob("*.py"))
                    if p.relative_to(ROOT).as_posix() not in listed]
        assert unlisted == [], f"add to scripts/literal_census.py MODULES (or EXCLUDED with a reason): {unlisted}"

    def test_exclusions_have_a_reason(self):
        assert all(len(r) > 10 for r in lc.EXCLUDED.values())

    def test_the_registries_are_the_excluded_ones(self):
        assert "src/ait/optimization/param_spaces.py" in lc.EXCLUDED


class TestTheScannerSeesWhatItShould:
    def _scan(self, tmp_path, body):
        f = tmp_path / "m.py"
        f.write_text(body)
        return lc.scan(f)

    def test_counts_a_new_literal_in_a_function_and_at_module_level(self, tmp_path):
        got = self._scan(tmp_path, "X = 0.37\n\ndef f(a):\n    return a * 1.23 + 7\n")
        assert got == {"<module>": [0.37], "f": [1.23]}

    def test_counts_methods_by_qualified_name(self, tmp_path):
        got = self._scan(tmp_path, "class C:\n    def m(self):\n        return 0.71\n")
        assert got == {"C.m": [0.71]}

    def test_ignores_trivial_values_logging_round_and_string_repeat(self, tmp_path):
        got = self._scan(tmp_path, 'def f(x):\n    log.info("e", v=0.81)\n    print(0.82)\n    y = round(x, 4)\n'
                                   '    s = "=" * 70\n    return x + 1 + 2 + 10 + 252\n')
        assert got == {}

    def test_a_changed_literal_is_a_difference(self, tmp_path):
        before = {"m.py": self._scan(tmp_path, "def f():\n    return 0.37\n")}
        allow = {"entries": {"m.py::f": {"values": [0.37], "category": "structural"}}}
        assert lc.differences(before, allow) == []
        after = {"m.py": self._scan(tmp_path, "def f():\n    return 0.38\n")}
        assert any(p.startswith("CHANGED") for p in lc.differences(after, allow))

    def test_a_new_function_with_literals_and_a_stale_entry_are_differences(self):
        allow = {"entries": {"m.py::old": {"values": [0.5], "category": "structural"}}}
        probs = lc.differences({"m.py": {"new": [0.9]}}, allow)
        assert any(p.startswith("NEW") for p in probs) and any(p.startswith("STALE") for p in probs)

    def test_a_removed_literal_is_a_difference(self):
        allow = {"entries": {"m.py::f": {"values": [0.37, 0.41], "category": "structural"}}}
        assert any("removed [0.41]" in p for p in lc.differences({"m.py": {"f": [0.37]}}, allow))


class TestItBitesOnTheRealCode:
    def test_a_literal_added_to_a_real_function_fails_the_guard(self, tmp_path, monkeypatch):
        """Copy the repo's census modules, inject a literal into the engine, and check the comparison fails."""
        engine = ROOT / "src" / "ait" / "backtesting" / "engine.py"
        mutated = tmp_path / "engine.py"
        mutated.write_text(engine.read_text().replace("def _remark_t(", "def _remark_t(", 1).replace(
            "    def _simple_direction(self, hist", "    _hidden = 0.4321\n\n    def _simple_direction(self, hist", 1))
        cur = lc.census()
        cur["src/ait/backtesting/engine.py"] = lc.scan(mutated)
        probs = lc.differences(cur, lc.load_allowlist())
        assert any("Backtester" in p and "0.4321" in p for p in probs)

    def test_converting_a_literal_to_a_setting_also_has_to_touch_the_allowlist(self, tmp_path):
        engine = ROOT / "src" / "ait" / "backtesting" / "engine.py"
        mutated = tmp_path / "engine.py"
        mutated.write_text(engine.read_text().replace("    _risk_free_rate: float = DEFAULT_RISK_FREE_RATE\n", "", 1)
                           .replace("lru_cache(maxsize=4096)", "lru_cache(maxsize=8)", 1))
        cur = lc.census()
        cur["src/ait/backtesting/engine.py"] = lc.scan(mutated)
        assert any("_session_close" in p and p.startswith("CHANGED") for p in lc.differences(cur, lc.load_allowlist()))

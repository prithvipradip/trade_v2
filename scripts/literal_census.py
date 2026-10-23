#!/usr/bin/env python
"""W88e: census of NUMERIC LITERALS in the result-affecting research code, and the allowlist that keeps it from growing.

The question this answers: "what numbers decide a research result and are written inside the code instead of in a
setting the researcher can see?" The census walks the AST of the modules in ``MODULES`` and records, per function
(``Class.method`` / ``<module>``), the sorted multiset of its non-trivial numeric literals. ``tests/data/literal_allowlist.json``
holds the accepted census with a CATEGORY for each function; ``tests/test_w88_literal_census.py`` fails when the code and the
allowlist disagree, so a new literal (or a converted one) has to be seen and classified by a person.

    python scripts/literal_census.py              # print the census summary and any difference from the allowlist
    python scripts/literal_census.py --update     # rewrite the allowlist (keeps existing categories, classifies new entries)

What is NOT counted: trivial values (0, 1, 2, -1: see ``TRIVIAL``; W90e and W95 narrowed it), booleans, ``round()`` digits, arguments of
logging / ``print`` calls, string-repeat counts (``"=" * 70``), and anything in a docstring or comment.

Categories (closed set): ``display`` (formatting), ``epsilon`` (numerical tolerance), ``fixed_feature_group`` (a feature group W88d
left fixed on purpose), ``dormant_path`` (legacy / dormant structures, off the iron-condor path), ``structural`` (a protocol or
model constant that is not a tunable), ``registered_followup`` (a research number still in code; registered in PLAN.md, needs a note).
Research-only module: nothing under ``src/ait`` imports it.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "tests" / "data" / "literal_allowlist.json"

# W95: the ignore list is now ONLY 0 / 1 / 2 / -1. W90e had narrowed it to unit-like values (100, 252, 365, 12, 24, 60, 1000), but
# those hid nine real windows and minimums (a 60-day high, a 252-bar context, a 100-row minimum, ...), which became settings in
# W95; so the unit conversions (x 100 shares per contract, / 365 days, sqrt(252)) are listed and classified like everything else.
# Before W90e it also hid 3, 4, 5, 7, 10, 30, 0.5 and 0.25 (a 5-day lookback, a 3-window minimum, a 25% capital cap, a 0.5 baseline).
TRIVIAL = {0, 1, -1, 2, 0.0, 1.0, -1.0, 2.0}
CATEGORIES = ("display", "epsilon", "fixed_feature_group", "dormant_path", "structural", "registered_followup")

# The result-affecting research modules. A new module under these directories must be added here or to EXCLUDED.
MODULES = [
    *(f"src/ait/backtesting/{m}.py" for m in (
        "engine", "walkforward", "learner", "replay", "pricing", "iv_surface", "iv_model", "settlement", "friction",
        "study_harness", "result", "pool", "range_bank", "run_record", "ex_div", "dte_report", "gate_parity",
        "expiry_calendar", "iv_rank_compare")),
    *(f"src/ait/optimization/{m}.py" for m in ("optimizer", "objectives", "results")),
    *(f"src/ait/ml/{m}.py" for m in (
        "trainer", "meta_label", "hyperparams", "features", "ensemble", "range_predictor", "vol_magnitude_predictor",
        "range_spec", "excursion", "feature_spec", "regime_gate", "drift", "regime")),
    "run_backtest.py", "run_optimizer.py",
]
# Directories whose every .py must be in MODULES or EXCLUDED, so a new module cannot slip past the census.
WATCHED_DIRS = ["src/ait/backtesting", "src/ait/optimization", "src/ait/ml"]
EXCLUDED = {
    "src/ait/optimization/param_spaces.py": "the Optuna search-range registry (overridable per study, validated against HARD_BOUNDS)",
    "src/ait/optimization/__init__.py": "package init",
    "src/ait/backtesting/__init__.py": "package init",
    "src/ait/ml/__init__.py": "package init",
}


def _is_trivial(v) -> bool:
    return v in TRIVIAL


def _skip_call(node: ast.Call) -> bool:
    f = node.func
    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
    base = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else ""
    return name in {"print", "round"} or base in {"log", "logger"} or name in {"info", "debug", "warning", "error", "critical"}


def scan(path: Path) -> dict[str, list]:
    """{function name: sorted list of non-trivial numeric literals} for one file."""
    tree = ast.parse(path.read_text())
    parents: dict = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p

    def owner(n) -> str:
        names = []
        while n in parents:
            n = parents[n]
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(n.name)
        return ".".join(reversed(names)) or "<module>"

    out: dict[str, list] = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)):
            continue
        if _is_trivial(n.value):
            continue
        # skip: arguments of print / round / logging calls, string-repeat counts, f-string internals
        anc, skip = n, False
        while anc in parents:
            anc = parents[anc]
            if isinstance(anc, ast.Call) and _skip_call(anc):
                skip = True
                break
            if isinstance(anc, ast.BinOp) and isinstance(anc.op, ast.Mult) and any(
                    isinstance(s, ast.Constant) and isinstance(s.value, str) for s in (anc.left, anc.right)):
                skip = True
                break
            if isinstance(anc, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                break
        if skip:
            continue
        out.setdefault(owner(n), []).append(n.value)
    return {k: sorted(v) for k, v in out.items()}


def census() -> dict[str, dict[str, list]]:
    return {m: scan(ROOT / m) for m in MODULES}


def _epsilon(values: list) -> bool:
    return all(isinstance(v, float) and 0 < abs(v) < 1e-4 for v in values)


def classify(module: str, function: str, values: list) -> tuple[str, str]:
    """Initial category for a NEW allowlist entry (a person reviews the generated file; existing entries are kept)."""
    if _epsilon(values):
        return "epsilon", ""
    leaf = function.split(".")[-1]
    if module.endswith(("results.py", "result.py", "dte_report.py")) and any(
            k in leaf for k in ("summary", "format", "report", "to_", "_dt", "print")):
        return "display", ""
    if any(k in leaf for k in ("summary", "format_", "_print", "__repr__", "render")):
        return "display", ""
    if module.endswith("ml/features.py"):
        return "fixed_feature_group", ""
    if module.endswith(("ml/regime.py", "ml/regime_gate.py", "ml/drift.py", "ml/excursion.py")):
        return "structural", ""
    return "registered_followup", "W88 census baseline: a research number still in code (see PLAN.md W88e); convert it or classify it"


def load_allowlist() -> dict:
    return json.loads(ALLOWLIST.read_text()) if ALLOWLIST.exists() else {"entries": {}}


def differences(current: dict, allow: dict) -> list[str]:
    """Human-readable differences between the code and the allowlist; empty when they agree."""
    entries = allow.get("entries", {})
    problems = []
    for module, funcs in current.items():
        for func, values in funcs.items():
            key = f"{module}::{func}"
            want = entries.get(key)
            if want is None:
                problems.append(f"NEW  {key}: literals {values} are not in the allowlist")
            elif sorted(want["values"]) != sorted(values):
                added = list((Counter(values) - Counter(want["values"])).elements())
                removed = list((Counter(want["values"]) - Counter(values)).elements())
                problems.append(f"CHANGED {key}: added {added} removed {removed}")
    for key in entries:
        module, func = key.split("::")
        if func not in current.get(module, {}):
            problems.append(f"STALE {key}: the allowlist lists literals that are no longer in the code")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--update", action="store_true", help="rewrite the allowlist from the current code")
    args = ap.parse_args(argv)
    cur, allow = census(), load_allowlist()
    if args.update:
        old = allow.get("entries", {})
        entries = {}
        for module, funcs in cur.items():
            for func, values in funcs.items():
                key = f"{module}::{func}"
                if key in old and old[key].get("category") in CATEGORIES:
                    cat, note = old[key]["category"], old[key].get("note", "")
                else:
                    cat, note = classify(module, func, values)
                entries[key] = {"values": values, "category": cat, **({"note": note} if note else {})}
        ALLOWLIST.write_text(json.dumps({"categories": list(CATEGORIES), "entries": dict(sorted(entries.items()))}, indent=1) + "\n")
        print(f"wrote {len(entries)} entries to {ALLOWLIST.relative_to(ROOT)}")
        allow = load_allowlist()
    total = sum(len(v) for f in cur.values() for v in f.values())
    cats = Counter(e["category"] for e in allow.get("entries", {}).values())
    print(f"{sum(len(f) for f in cur.values())} functions / {total} literals in {len(cur)} modules; by category: {dict(cats)}")
    probs = differences(cur, allow)
    for p in probs:
        print(" ", p)
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())

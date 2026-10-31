"""W116 class guards: research never reaches Yahoo or a resampled stand-in, and every loader call names its window.

These scan the source (AST, never strings in comments) so the NEXT module that imports yfinance, or the next script
that asks for daily bars without a window, fails here instead of quietly mixing a source into a study.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Modules allowed to import yfinance, with the reason. Everything else under src/ and scripts/ must not.
YFINANCE_ALLOWED = {
    "src/ait/data/market_data.py": "the LIVE MarketDataService (IBKR -> Yahoo); the research functions are guarded below",
    "src/ait/data/dividends.py": "live dividend calendar",
    "src/ait/data/earnings.py": "live earnings calendar",
    "src/ait/data/equity_stats.py": "live fundamentals refresh",
    "src/ait/data/options_chain.py": "live options chain fallback",
    "src/ait/utils/logging.py": "silences yfinance's logger only",
    "scripts/backfill_dividends.py": "ingest tool: the one place dividends enter the store",
    "scripts/backfill_historical_data.py": "ingest tool",
    "scripts/calibrate_option_spreads.py": "ingest tool",
    # removed by later W116 packages (each deletes its entry from this list):
    "scripts/gate_parity_ab.py": "WP5: its VIX closes",
}
# callers of HistoricalDataStore.resample_to_daily (the 5-minute -> daily stand-in); research code may not use it
RESAMPLE_ALLOWED = {
    "src/ait/data/historical.py": "the definition",
    "scripts/run_integration_test.py": "WP5: the legacy integration harness",
}
RESEARCH_LOADER_FUNCTIONS = {"load_daily_ohlcv", "_load_research_daily", "_load_stored_index_daily"}


def _py_files():
    for base in ("src", "scripts"):
        yield from sorted((ROOT / base).rglob("*.py"))
    yield from sorted(ROOT.glob("*.py"))


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _imports_yfinance(tree: ast.AST) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name.split(".")[0] == "yfinance" for a in n.names):
            return True
        if isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == "yfinance":
            return True
    return False


def test_only_the_listed_modules_import_yfinance():
    found = {_rel(p) for p in _py_files() if _imports_yfinance(_tree(p))}
    assert found <= set(YFINANCE_ALLOWED), f"new yfinance importers: {sorted(found - set(YFINANCE_ALLOWED))}"


def test_every_allowlist_entry_still_imports_yfinance():
    """An entry whose module no longer imports it must be deleted, so the list only ever shrinks."""
    stale = [rel for rel in YFINANCE_ALLOWED if not _imports_yfinance(_tree(ROOT / rel))]
    assert not stale, f"remove from YFINANCE_ALLOWED: {stale}"


def test_the_research_loader_functions_never_touch_yahoo():
    tree = _tree(ROOT / "src/ait/data/market_data.py")
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in RESEARCH_LOADER_FUNCTIONS]
    assert {f.name for f in fns} == RESEARCH_LOADER_FUNCTIONS
    for fn in fns:
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        assert not ({"yf", "yfinance"} & names), fn.name
        assert "resample_to_daily" not in attrs, fn.name


def test_only_the_listed_modules_call_resample_to_daily():
    found = set()
    for p in _py_files():
        for n in ast.walk(_tree(p)):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "resample_to_daily") \
                    or (isinstance(n, ast.FunctionDef) and n.name == "resample_to_daily"):
                found.add(_rel(p))
    assert found <= set(RESAMPLE_ALLOWED), f"unexpected resample_to_daily users: {sorted(found - set(RESAMPLE_ALLOWED))}"
    stale = [rel for rel in RESAMPLE_ALLOWED if rel not in found]
    assert not stale, f"remove from RESAMPLE_ALLOWED: {stale}"


def test_every_loader_call_names_its_window():
    """A daily-bars request without ``days`` or ``start_date`` has no window to verify."""
    offenders = []
    for p in _py_files():
        for n in ast.walk(_tree(p)):
            if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "load_daily_ohlcv":
                kws = {k.arg for k in n.keywords}
                if not ({"days", "start_date"} & kws) and len(n.args) < 2:
                    offenders.append(f"{_rel(p)}:{n.lineno}")
    assert not offenders, offenders

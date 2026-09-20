"""W38 (R26 F5) — the go-live pace window, and the stdlib-only ET leaf.

`w21-go-live-pace-window-compares-a-naive-cutoff-to-aware-rows`.

`compute_go_live_verdict` built its 28-day cutoff from the NAIVE host clock and
compared it as TEXT against rows W21 made aware ET. Nothing raised — both sides
are strings and ISO ordering survives an offset suffix — so the mismatch was
invisible, and the error equalled the host's offset from ET. This box runs at
ET+6, so the window was shifted six hours every time the Friday scorecard ran.
Pace is a go-live gate criterion, so a close near the boundary was counted in
or out by where the operator happened to be sitting.

The second half of this file guards `ait.utils.tz`, the stdlib-only leaf W38
extracted so the supervisor and this module can share one ET parser. Its whole
value is that importing it is free; a single heavy import inside it would
silently re-arm the supervisor-death incident W34 fixed.

Everything here drives the real `compute_go_live_verdict`.
"""

from __future__ import annotations

import ast
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from ait.bot.state import StateManager
from ait.reporting.go_live import PACE_WINDOW_DAYS, compute_go_live_verdict
from ait.utils.tz import now_et

SRC = Path(__file__).resolve().parents[1] / "src"


def _db(tmp_path: Path, exit_times: list[str]) -> Path:
    """A REAL state DB through StateManager, then closed IC rows inserted
    directly — the same idiom as tests/test_w3_scorecard_truth.py."""
    st = StateManager(db_path=tmp_path / "ait_state.db")
    with sqlite3.connect(st._db_path) as con:
        for i, ts in enumerate(exit_times):
            con.execute(
                "INSERT INTO trades (trade_id, symbol, strategy, direction, "
                "status, entry_time, entry_price, quantity, contract_type, "
                "exit_time, realized_pnl, exit_reason_detailed, "
                "capital_at_risk) VALUES (?,?,?,?,?,?,?,1,'iron_condor',?,?,?,?)",
                (f"T{i}", "SPY", "iron_condor", "neutral", "closed", ts, 1.0,
                 ts, 25.0, "take_profit", 200.0),
            )
    return Path(st._db_path)


class TestThePaceWindowUsesTheETClock:

    def test_a_close_just_inside_the_window_is_counted(self, tmp_path):
        """27d23h back must be inside a 28-day window on any host.

        Pre-fix on a host east of ET the naive cutoff sat up to six hours in
        the future relative to the aware rows, so a close in that band fell
        out of the window it belongs in.
        """
        edge = (now_et() - timedelta(days=PACE_WINDOW_DAYS, hours=-1)).isoformat()
        v = compute_go_live_verdict(_db(tmp_path, [edge]))
        assert v.pace_per_week > 0, (
            f"a close at {edge} is {PACE_WINDOW_DAYS}d-1h old and was excluded "
            f"from the {PACE_WINDOW_DAYS}-day pace window"
        )

    def test_a_close_well_outside_the_window_is_not_counted(self, tmp_path):
        """CONTROL: the window must still exclude something."""
        old = (now_et() - timedelta(days=PACE_WINDOW_DAYS + 10)).isoformat()
        v = compute_go_live_verdict(_db(tmp_path, [old]))
        assert v.pace_per_week == 0, "the window no longer excludes old closes"

    def test_the_verdict_does_not_move_with_the_host_timezone(
        self, tmp_path, monkeypatch
    ):
        """The measurement is about the market's clock, not the operator's.

        Pins a real non-ET zone, which is the house rule for clock tests.
        """
        import time as _time

        edge = (now_et() - timedelta(days=PACE_WINDOW_DAYS, hours=-1)).isoformat()
        db = _db(tmp_path, [edge])

        seen = []
        for tz in ("UTC", "Europe/Berlin", "Australia/Sydney"):
            monkeypatch.setenv("TZ", tz)
            if hasattr(_time, "tzset"):
                _time.tzset()
            seen.append(compute_go_live_verdict(db).pace_per_week)

        monkeypatch.delenv("TZ", raising=False)
        if hasattr(_time, "tzset"):
            _time.tzset()

        assert len(set(seen)) == 1, (
            f"pace changed with the host timezone: {dict(zip(('UTC','Berlin','Sydney'), seen))}"
        )

    def test_an_explicit_naive_now_still_works(self, tmp_path):
        """CONTROL: callers passing a naive `now` must not start crashing."""
        naive = now_et().replace(tzinfo=None)
        recent = (now_et() - timedelta(days=1)).isoformat()
        v = compute_go_live_verdict(_db(tmp_path, [recent]), now=naive)
        assert v.pace_per_week > 0

    def test_an_unparseable_row_does_not_break_the_scorecard(self, tmp_path):
        """One malformed row must not take the whole operator readout down."""
        recent = (now_et() - timedelta(days=1)).isoformat()
        v = compute_go_live_verdict(_db(tmp_path, [recent, "not-a-timestamp"]))
        assert v.pace_per_week > 0


class TestTheETLeafStaysImportLight:
    """CLASS GUARD — the contract that makes `ait.utils.tz` worth having.

    `ait.utils.time` builds the NYSE calendar at module scope, so importing it
    costs pandas + numpy. W23 did that in the supervisor and a broken
    scientific stack killed it at import — no bot restart, no Gateway
    recovery, no page. W34 removed it; W38 extracted this leaf so the rule can
    still be shared. One heavy import here silently re-arms that incident.
    """

    ALLOWED_STDLIB = {"datetime", "zoneinfo", "__future__"}

    def test_the_leaf_imports_only_the_standard_library(self):
        tree = ast.parse((SRC / "ait" / "utils" / "tz.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported <= self.ALLOWED_STDLIB, (
            f"ait.utils.tz imports {imported - self.ALLOWED_STDLIB} — the "
            f"supervisor and go_live depend on this module being free to import"
        )

    def test_importing_it_pulls_in_nothing_heavy(self):
        """Executed, not read: transitive cost is what actually matters."""
        heavy = {"pandas", "numpy", "pandas_market_calendars", "structlog",
                 "pydantic", "sqlalchemy"}
        for mod in list(sys.modules):
            if mod.split(".")[0] in heavy | {"ait"}:
                pass  # already imported by the suite; measure by subprocess
        import subprocess
        code = (
            "import sys; import ait.utils.tz; "
            f"print(sorted({{m.split('.')[0] for m in sys.modules}} & {heavy!r}))"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, cwd=SRC.parent)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]", (
            f"importing ait.utils.tz pulled in {out.stdout.strip()}"
        )

    def test_the_constrained_modules_do_not_import_the_heavy_one(self):
        """master.py and go_live.py must not reach `ait.utils.time` at module
        scope — the whole point of the leaf."""
        for rel in ("ait/orchestration/master.py", "ait/reporting/go_live.py"):
            tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
            for node in tree.body:  # module scope only; lazy imports are fine
                if isinstance(node, ast.ImportFrom) and node.module == "ait.utils.time":
                    pytest.fail(
                        f"{rel}:{node.lineno} imports ait.utils.time at module "
                        f"scope — that rebuilds the NYSE calendar at import"
                    )

    def test_the_guard_reads_a_real_tree(self):
        """Anti-vacuity."""
        assert (SRC / "ait" / "utils" / "tz.py").exists()
        assert len(list(SRC.rglob("*.py"))) > 50

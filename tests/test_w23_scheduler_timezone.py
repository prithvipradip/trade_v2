"""W23: the supervisor scheduler's silent timezone fallback — three distinct
bugs, confirmed by direct probe against this box's real apscheduler install,
not assumed from the API surface.

  1. Nine pre-built CronTrigger(...) objects passed to scheduler.add_job()
     never inherited the scheduler's own timezone="US/Eastern" — apscheduler
     only injects that when IT builds a trigger from an alias string
     ("cron"); a pre-built object keeps whatever get_localzone() returned at
     construction (this host: Europe/Madrid). Confirmed live: digest_am fired
     at 03:35 ET instead of 09:35 (data/digest_sent_20260916).

  2. The two naive run_date= catch-up jobs (retrain_catchup, backtest_catchup)
     are a DIFFERENT mechanism, easy to conflate with #1 but confirmed
     distinct: passing a naive datetime through the "date" alias does NOT
     correctly localize it — it mislabels the naive wall-clock number with
     the scheduler's offset rather than converting it. Confirmed live:
     retrain_catchup registered "2026-09-14 21:31:45 EDT", actually ran
     2026-09-15 03:31:45 local — 6h late, to the second. An AWARE run_date
     passes through unchanged (probed below) — the fix is to stop
     constructing the value from naive datetime.now(), not to touch the
     scheduler or the "date" call itself.

  3. _backtest_catchup_run_at's own market-hours-avoidance check compared
     naive-host-local hours against ET thresholds (9:15/16:30) — wrong
     clock entirely, independent of what the scheduler does with the result.

Deliberately UNCHANGED: _backtest_catchup_due's own naive-local `now` — its
docstring already explains this is intentional (compared against equally
naive file mtimes; a naive Sunday-20:00 slot is immaterial for a coarse
weekly gate). Fixing #2/#3 means the SAME catch-up now passes two
deliberately different clocks to two sibling functions — tested here so a
future "simplification" doesn't collapse them back into a bug.
"""
from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ait.orchestration import master as M

ET = ZoneInfo("US/Eastern")
SRC = Path(master_file := M.__file__)


def _et(y, mo, d, h, mi, s=0) -> datetime:
    return datetime(y, mo, d, h, mi, s, tzinfo=ET)


class TestEveryCronTriggerHasATimezone:
    """The class guard — this session's own established rule: an AST scan
    catches the NEXT missed instance, not just the ones already found (W20's
    hour_of_day, W21's 19-vs-18 reader count were both found this way, not
    by re-reading the ones already fixed)."""

    def test_ast_scan_of_main_finds_no_bare_crontrigger(self):
        tree = ast.parse(SRC.read_text())
        main_fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "main")
        offenders = []
        for node in ast.walk(main_fn):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "CronTrigger"):
                has_tz = any(kw.arg == "timezone" for kw in node.keywords)
                if not has_tz:
                    offenders.append(f"line {node.lineno}")
        assert not offenders, (
            "CronTrigger(...) missing timezone=ET at: " + ", ".join(offenders))

    def test_exactly_nine_crontrigger_calls_are_covered(self):
        """A sanity floor — if this count changes, the scan above is still
        running, but it's worth knowing the shape changed."""
        tree = ast.parse(SRC.read_text())
        main_fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "main")
        count = sum(
            1 for node in ast.walk(main_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "CronTrigger")
        assert count == 9, f"expected 9 CronTrigger(...) sites, found {count}"


class TestCronJobsActuallyResolveToET:
    """Not just the keyword's presence — real execution, real jobs, proving
    the fix works rather than that the pattern matches."""

    def test_daily_retrain_resolves_to_us_eastern(self):
        sch = BackgroundScheduler(timezone="US/Eastern")
        job = sch.add_job(
            lambda: None,
            CronTrigger(day_of_week="mon-fri", hour=7, minute=30, timezone=M.ET),
            id="daily_retrain")
        assert str(job.trigger.timezone) == "US/Eastern"

    def test_daily_report_resolves_to_us_eastern_not_host_local(self):
        sch = BackgroundScheduler(timezone="US/Eastern")
        job = sch.add_job(
            lambda: None,
            CronTrigger(day_of_week="mon-fri", hour=16, minute=30, timezone=M.ET),
            id="daily_report")
        assert str(job.trigger.timezone) == "US/Eastern"
        # The actual live-observed failure mode before this fix: the job
        # fired at 10:30 ET instead of 16:30 because it silently carried
        # the host's own zone. Confirm the trigger's next fire time, computed
        # from a known instant, lands on 16:30 ET — not 10:30, not any
        # host-local reinterpretation of "16:30".
        start = _et(2026, 9, 17, 0, 0)  # Thursday, well before either time
        nxt = job.trigger.get_next_fire_time(None, start)
        assert (nxt.hour, nxt.minute) == (16, 30), nxt

    def test_without_the_fix_it_would_have_used_the_host_zone(self):
        """Negative control: prove the OLD (pre-fix) shape really is broken,
        so this test file would have failed loudly against the original
        code — not just that the new shape happens to look right."""
        sch = BackgroundScheduler(timezone="US/Eastern")
        job = sch.add_job(
            lambda: None,
            CronTrigger(day_of_week="mon-fri", hour=16, minute=30),  # no timezone=
            id="unfixed")
        assert str(job.trigger.timezone) != "US/Eastern"


class TestCatchupRunAtFunctions:
    def test_retrain_catchup_run_at_is_two_minutes_out_and_stays_aware(self):
        now = _et(2026, 9, 16, 8, 45)
        run_at = M._retrain_catchup_run_at(now)
        assert run_at == _et(2026, 9, 16, 8, 47)
        assert run_at.tzinfo is not None

    def test_backtest_catchup_run_at_outside_market_hours_is_unmoved(self):
        now = _et(2026, 9, 14, 20, 0)  # Sunday 20:00 ET boot
        run_at = M._backtest_catchup_run_at(now)
        assert run_at == _et(2026, 9, 14, 20, 10)

    def test_backtest_catchup_run_at_inside_market_hours_pushed_to_1640(self):
        now = _et(2026, 9, 15, 10, 0)  # Monday 10:00 ET — +10min lands at 10:10, in-session
        run_at = M._backtest_catchup_run_at(now)
        assert (run_at.hour, run_at.minute) == (16, 40)
        assert run_at.tzinfo is not None

    def test_backtest_catchup_boundary_915_et_is_correctly_evaluated(self):
        """The exact defect W23 fixed: this boundary check must read ET
        hours. A naive host-local `now` (this host: ET+6) would never
        correctly land in the (9,15)-(16,30) window at all."""
        now = _et(2026, 9, 15, 9, 5)  # +10min = 9:15 ET exactly — in-session
        run_at = M._backtest_catchup_run_at(now)
        assert (run_at.hour, run_at.minute) == (16, 40)

    def test_backtest_catchup_boundary_1630_et_is_exclusive(self):
        now = _et(2026, 9, 15, 16, 20)  # +10min = 16:30 ET exactly — window is exclusive
        run_at = M._backtest_catchup_run_at(now)
        assert (run_at.hour, run_at.minute) == (16, 30), (
            "16:30 ET is the report slot itself; the window must be exclusive of it")


class TestAwareRunDatePassesThroughUnchanged:
    """The mechanism that makes the #2 fix actually work — reproduced as a
    real test, not just asserted in the plan. Would fail loudly if the
    underlying apscheduler behavior ever changes."""

    def test_aware_run_date_is_not_mislabeled(self):
        sch = BackgroundScheduler(timezone="US/Eastern")
        aware = _et(2026, 9, 16, 8, 47)
        job = sch.add_job(lambda: None, "date", run_date=aware, id="aware_case")
        assert job.trigger.run_date == aware

    def test_naive_run_date_IS_mislabeled_not_converted(self):
        """Negative control proving the bug this fix avoids: a naive
        datetime through the SAME alias gets the scheduler's offset stapled
        onto its unchanged wall-clock number, not a real timezone
        conversion — the exact live incident (registered ...21:31:45 EDT,
        ran ...03:31:45 local, 6h off)."""
        sch = BackgroundScheduler(timezone="US/Eastern")
        naive = datetime(2026, 9, 16, 8, 47)  # no tzinfo
        job = sch.add_job(lambda: None, "date", run_date=naive, id="naive_case")
        result = job.trigger.run_date
        # Mislabeled: same wall-clock numbers, now claiming to be ET.
        assert (result.hour, result.minute) == (8, 47)
        assert str(result.tzinfo) in ("US/Eastern",)
        # Which is a DIFFERENT real-world instant than the naive value meant
        # on any host not already sitting at UTC-4/-5.
        assert result.utcoffset() == timedelta(hours=-4)


class TestBacktestDueCheckStaysNaiveOnPurpose:
    """Guards the deliberate split recorded in master.py's own comment at
    the call site: _backtest_catchup_due keeps a naive `now` (compared
    against equally-naive file mtimes) while _backtest_catchup_run_at now
    needs an aware one. A future edit collapsing these back to one shared
    clock would silently reintroduce bug #3."""

    def test_due_check_signature_takes_naive_datetimes_without_erroring(self, tmp_path):
        naive_now = datetime(2026, 9, 14, 20, 0)  # no tzinfo — must not raise
        assert M._backtest_catchup_due(naive_now, reports_dir=tmp_path) is True

    def test_call_site_uses_two_different_clocks_not_one(self):
        """Source-level guard: the call site must construct now_et() (or
        an equivalent aware value) for the run-at computation, separately
        from the naive datetime.now() feeding the due-check — not reuse a
        single variable for both."""
        src = inspect.getsource(M.main)
        # The naive due-check clock and the aware run-at clock must be
        # visibly distinct expressions at the call site.
        assert "_now_bt = datetime.now()" in src
        assert "_backtest_catchup_run_at(now_et())" in src
        assert "_backtest_catchup_run_at(_now_bt)" not in src, (
            "run-at computation reverted to the naive due-check clock")

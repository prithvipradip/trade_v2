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
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
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
    by re-reading the ones already fixed).

    W43: scoped to the WHOLE MODULE, not just `main`. It was function-scoped
    to `main` because that is where all nine calls lived when this class was
    written; W43 moved them into `_register_recurring_jobs` (extracted so the
    jobs could be registered on a real scheduler outside `main`'s other side
    effects), and a function-scoped scan of `main` then silently found ZERO
    CronTrigger calls — the guard below still technically passed (vacuously:
    no offenders because nothing was scanned), and only the sanity-floor
    test caught the drift, exactly the anti-vacuity role it exists for. A
    module-wide scan is not merely a fix for this one refactor: it is robust
    to the NEXT one too, since it no longer assumes any particular function
    holds these calls.
    """

    def _crontrigger_calls(self):
        tree = ast.parse(SRC.read_text())
        return [node for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "CronTrigger"]

    def test_ast_scan_of_master_finds_no_bare_crontrigger(self):
        offenders = [
            f"line {node.lineno}" for node in self._crontrigger_calls()
            if not any(kw.arg == "timezone" for kw in node.keywords)
        ]
        assert not offenders, (
            "CronTrigger(...) missing timezone=ET at: " + ", ".join(offenders))

    def test_exactly_nine_crontrigger_calls_are_covered(self):
        """A sanity floor — if this count changes, the scan above is still
        running, but it's worth knowing the shape changed. Also the
        anti-vacuity companion: it is what actually caught the W43 refactor
        moving every call out of `main`, when the guard above stayed
        (vacuously) green on its own."""
        # 10 since 2026-09-29: + shared_logs_publish (16:45 ET, shared_logs.py).
        count = len(self._crontrigger_calls())
        assert count == 10, f"expected 10 CronTrigger(...) sites, found {count}"

    def test_the_scan_does_not_depend_on_which_function_holds_the_calls(self):
        """Anti-vacuity for the anti-vacuity fix itself. Not a source-string
        check (that would be the exact anti-pattern this wave removes
        elsewhere) — an AST property: the nine calls' enclosing function is
        NOT `main` (proving the scan's count does not coincidentally still
        depend on that one function), and they are not scattered one-per-
        function either (proving a future split into many tiny functions
        would not silently starve the count back toward vacuous)."""
        tree = ast.parse(SRC.read_text())
        enclosing = {}
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "CronTrigger"):
                    enclosing.setdefault(node.lineno, []).append(fn.name)

        calls = self._crontrigger_calls()
        assert len(enclosing) == len(calls), (
            "premise: every CronTrigger call must resolve to exactly one "
            "enclosing function for this check to mean anything"
        )
        owners = {min(names, key=len) for names in enclosing.values()}
        assert owners == {"_register_recurring_jobs"}, (
            f"expected all nine calls inside one function, found: {owners} — "
            f"if that function is 'main' again, the module-wide scan is no "
            f"longer proven to matter over the old function-scoped one"
        )


class TestCronJobsActuallyResolveToET:
    """Not just the keyword's presence — real execution, real jobs, proving
    the fix works rather than that the pattern matches.

    W43 (R26 `w23-et-cron-tests-build-their-own-crontrigger`): the previous
    form of these two tests built their OWN `BackgroundScheduler` and their
    OWN `CronTrigger(..., timezone=M.ET)` — the only `master` reference was
    `M.ET` itself, so a revert of the real registration in
    `_register_recurring_jobs` would not have moved either assertion.
    Verified directly: tracing which `master.py` lines execute during the
    old form showed only `M.ET`'s own module-scope line (line 40) and the
    two class definitions — none of the nine real `add_job` sites.
    These now call the REAL `_register_recurring_jobs` (extracted from
    `main()` in this same wave) against a real, unstarted
    `BlockingScheduler` — `get_jobs()`/`get_job(id)` work without `.start()`
    ever being called, confirmed against the installed apscheduler 3.11.2.
    """

    def _real_scheduler(self):
        sch = M.BlockingScheduler(timezone="US/Eastern")
        M._register_recurring_jobs(sch, bot=MagicMock(), web=MagicMock())
        return sch

    def test_daily_retrain_resolves_to_us_eastern(self):
        sch = self._real_scheduler()
        job = sch.get_job("daily_retrain")
        assert job is not None, "the real registration did not create this job"
        assert str(job.trigger.timezone) == "US/Eastern", (
            # str(CronTrigger) does not surface the timezone at all — confirmed
            # by reading a live job_registered log line — so the assertion
            # must be on .trigger.timezone specifically, never the repr.
            f"job.trigger.timezone={job.trigger.timezone!r}"
        )

    def test_daily_report_resolves_to_us_eastern_not_host_local(self):
        sch = self._real_scheduler()
        job = sch.get_job("daily_report")
        assert job is not None
        assert str(job.trigger.timezone) == "US/Eastern"
        # The actual live-observed failure mode before this fix: the job
        # fired at 10:30 ET instead of 16:30 because it silently carried
        # the host's own zone. Confirm the trigger's next fire time, computed
        # from a known instant, lands on 16:30 ET — not 10:30, not any
        # host-local reinterpretation of "16:30". `job.next_run_time` does
        # not exist on a pending (never-started) job — confirmed directly —
        # so the fire time has to come from the trigger itself.
        start = _et(2026, 9, 17, 0, 0)  # Thursday, well before either time
        nxt = job.trigger.get_next_fire_time(None, start)
        assert (nxt.hour, nxt.minute) == (16, 30), nxt

    def test_all_nine_recurring_jobs_are_registered(self):
        """CONTROL: the extraction must not have dropped or renamed a job —
        every id main() used to register directly must still exist."""
        sch = self._real_scheduler()
        ids = {job.id for job in sch.get_jobs()}
        assert ids == {
            "health_check", "daily_retrain", "db_backup", "weekly_scorecard",
            "digest_am", "digest_pm", "daily_report", "weekly_backtest",
            "monthly_cleanup", "daily_log_rotation",
            "shared_logs_publish",   # 2026-09-29
        }

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

    def test_call_site_uses_the_et_clock_for_run_date_not_the_due_check_clock(
        self, tmp_path, monkeypatch
    ):
        """W43 (R26 `w23-et-cron-tests-build-their-own-crontrigger`).

        Replaces an `inspect.getsource(M.main)` substring pin
        (`assert "_backtest_catchup_run_at(now_et())" in src`) with real
        execution. Drives the extracted `_maybe_schedule_backtest_catchup`
        with the due-check forced True (empty `REPORTS_DIR`) and the ET
        clock patched to a value DELIBERATELY far from the injected naive
        `now_bt` — if the call site regressed to computing `run_date` from
        `now_bt` instead of `now_et()` (bug #2/#3 this whole file exists to
        guard), the registered job would land on the wrong date entirely,
        not just the wrong offset.
        """
        monkeypatch.setattr(M, "REPORTS_DIR", tmp_path)  # forces the due-check True
        naive_now_bt = datetime(2026, 9, 14, 20, 0)       # Sunday, host-local, no tzinfo
        fake_et_now = _et(2026, 9, 20, 3, 0)              # deliberately a different DAY
        monkeypatch.setattr(M, "now_et", lambda: fake_et_now)

        assert M._backtest_catchup_due(naive_now_bt) is True, (
            "premise: an empty REPORTS_DIR must make the due-check fire"
        )

        sch = M.BlockingScheduler(timezone="US/Eastern")
        M._maybe_schedule_backtest_catchup(sch, now_bt=naive_now_bt)

        job = sch.get_job("backtest_catchup")
        assert job is not None, "the due-check fired but no job was registered"
        assert job.trigger.run_date.tzinfo is not None, (
            "run_date is naive — apscheduler will MISLABEL it with the "
            "scheduler's offset rather than converting it (bug #2)"
        )
        expected = M._backtest_catchup_run_at(fake_et_now)
        assert job.trigger.run_date == expected, (
            f"run_date={job.trigger.run_date} does not match "
            f"_backtest_catchup_run_at(now_et())={expected} — the call site "
            f"may have reverted to the naive due-check clock"
        )
        assert job.trigger.run_date.date() != naive_now_bt.date(), (
            "premise: the two clocks must land on different dates for this "
            "test to be able to tell them apart"
        )

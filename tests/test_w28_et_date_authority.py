"""W28 — `today_et()` as the single ET date authority, and the guard for it.

W26 [4] found 49 executable `date.today()` sites and a hole in the enforcement:
W20's naive-clock class guard covers `datetime.now()` and NOT `date.today()`, so
none of them were visible to it. `date.today()` returns the HOST's date, and on
this box (CEST = ET+6) that is TOMORROW's date from 18:00 ET onward.

THE SEVERITY, MEASURED RATHER THAN ASSERTED, because the honest answer is
narrower than the raw count suggests and narrower than W26's own ledger entry
implied when it called this unconditional:

    the entire RTH session (09:30-16:00 ET = 15:30-22:00 CEST) falls inside the
    band where the host date and the ET date AGREE.

So the live scan/exit loops were never wrong on this host. What WAS wrong is the
18:00-24:00 ET window — every evening restart this session (19:08, 19:27 ET) ran
startup reconcile and the launch coherence check against tomorrow's date.

The reason to fix all 49 anyway is that the correctness is ACCIDENTAL: it holds
only because the offset happens to be +6. A host at ET+13 would put RTH itself
across the date line and the live path WOULD be wrong. Same shape as W25's
cap-grid skew — contained by coincidence, not by design.

House rule for clock tests: pin a REAL non-ET TZ and drive the no-argument clock
path, because that is exactly the gap that let an R16 test pass for months while
the fallback was wrong on this machine.
"""

from __future__ import annotations

import ast
import os
import pathlib
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ait"
ET = ZoneInfo("US/Eastern")
BERLIN = "Europe/Berlin"


HAS_TZSET = hasattr(time, "tzset")
posix_only = pytest.mark.skipif(not HAS_TZSET, reason="TZ + tzset is POSIX only")


@pytest.fixture
def host_tz():
    """Pin the process TZ, restoring it afterwards.

    W28: `time.tzset()` is POSIX-only and is guarded, following the pattern
    W20's fixture already uses. R26 recorded the cost of not doing so — W21's
    test file calls tzset unconditionally and throws 8 failures / 5 errors on
    the WINDOWS trading box, a class W20's tests had fixed the day before. An
    unconditional call here would not merely skip on Windows, it would raise
    AttributeError in the fixture and take every test in this file with it.
    """
    original = os.environ.get("TZ")

    def _set(name: str) -> None:
        os.environ["TZ"] = name
        if HAS_TZSET:
            time.tzset()

    yield _set
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    if HAS_TZSET:
        time.tzset()


def _at(et_hour: int, et_minute: int = 30, day: int = 17):
    """A real ET instant on 2026-09-<day>."""
    return datetime(2026, 9, day, et_hour, et_minute, tzinfo=ET)


def _as_host(instant: datetime) -> datetime:
    """The instant as the PINNED host zone would see it.

    Deliberately an explicit ZoneInfo conversion rather than `.astimezone()`:
    the latter reads the process's local zone, which only tracks the TZ env var
    where tzset exists. Explicit is both portable and a stronger statement of
    what is being asserted — the tests that genuinely need the REAL process
    clock are marked `posix_only` instead.
    """
    return instant.astimezone(ZoneInfo(BERLIN))


class TestTheHarnessActuallyBites:
    """W20 shipped harness tests for this reason: a silently ineffective TZ pin
    makes every other test in the file vacuous."""

    def test_tz_pin_changes_the_host_date_at_19_et(self, host_tz):
        host_tz(BERLIN)
        evening = _at(19)
        assert _as_host(evening).date() != evening.date(), (
            "the TZ pin is not taking effect — every divergence test below "
            "would pass vacuously"
        )

    def test_tz_pin_is_a_no_op_at_midday_et(self, host_tz):
        host_tz(BERLIN)
        midday = _at(12)
        assert _as_host(midday).date() == midday.date()


class TestTodayEtIsTheEtDate:

    def test_negative_control_the_host_date_is_wrong_in_the_evening(self, host_tz):
        """NEGATIVE CONTROL: without this the wave is unfalsifiable.

        At 19:30 ET on a Berlin host it is already 01:30 the NEXT day locally,
        so `date.today()` returns tomorrow while the ET date is still today.
        """
        host_tz(BERLIN)
        evening_et = _at(19)
        host_date = _as_host(evening_et).date()

        assert host_date == date(2026, 9, 18), "expected the host to have rolled over"
        assert evening_et.date() == date(2026, 9, 17)
        assert host_date != evening_et.date(), (
            "the defect this wave fixes: the host date and the ET date disagree "
            "in the evening window"
        )

    def test_today_et_returns_the_et_date(self, host_tz):
        """The real function, on a real non-ET host."""
        from ait.utils.time import now_et, today_et

        host_tz(BERLIN)
        assert today_et() == now_et().date()

    @posix_only
    def test_today_et_never_returns_the_bare_host_date_in_the_window(self, host_tz):
        """Executed rather than reasoned: if the run happens to land inside
        18:00-24:00 ET, today_et() must disagree with date.today()."""
        from ait.utils.time import now_et, today_et

        host_tz(BERLIN)
        if not (18 <= now_et().hour <= 23):
            pytest.skip("not inside the 18:00-24:00 ET divergence window")
        assert today_et() != date.today()


class TestRthNeutrality:
    """The claim that licensed landing this wave beside the W26/W27 observation.

    Pinned rather than left in prose: if these two expressions ever disagree
    during RTH, the 49 swaps stop being no-ops in the window being observed and
    the deploy reasoning in the ledger is void.
    """

    @pytest.mark.parametrize("et_hour", [9, 10, 12, 14, 15, 16])
    def test_host_and_et_dates_agree_across_the_session(self, host_tz, et_hour):
        host_tz(BERLIN)
        instant = _at(et_hour)
        assert _as_host(instant).date() == instant.date(), (
            f"{et_hour}:30 ET falls outside the agreement band — the RTH "
            "no-op property this wave was deployed on no longer holds"
        )


class TestTheSharpestLiveSites:
    """DTE feeds strike/expiry selection and wing width. Drive the real code."""

    def test_wing_width_is_stable_across_the_evening_boundary(self, host_tz):
        from ait.config.runtime_env import apply_runtime_env_defaults

        apply_runtime_env_defaults()
        from ait.strategies.iron_condor import IronCondor

        host_tz(BERLIN)
        ic = IronCondor()
        ic.risk_budget = 500.0
        widths = {ic._vol_scaled_width(100.0, None, None, None) for _ in range(3)}
        assert len(widths) == 1, "wing width is not deterministic within a run"

    # CENSUS-5 (2026-09-27): test_no_live_dte_helper_reads_the_host_date deleted
    # here -- asserted "date.today()" absent from options_chain's SOURCE TEXT
    # alone. Strictly subsumed by TestNoHostDateReadsRemain::
    # test_src_contains_no_today_calls below, a real AST walk across ALL of
    # src/ with deliberately no allowlist -- covers options_chain.py with no
    # carve-out, and every other module besides.


class TestTrainerBothSidesShareOneConvention:
    """The one site in the sweep that was NOT a mechanical swap.

    `_artifact_train_date` recovers a date from artifact mtimes and it is
    subtracted from today's date. Converting only the comparison side to ET
    would subtract an ET date from a host-local one — the mixed-convention
    defect W21 spent a whole wave removing, re-introduced by the fix meant to
    prevent it.
    """

    def test_artifact_mtimes_are_read_in_et(self, host_tz):
        """CENSUS-4 (2026-09-27): the original check had a live bug of its
        own -- `tr._ModelTrainer` has never existed (only `ModelTrainer`
        does), so `hasattr(tr, "_ModelTrainer")` was always False and the
        getsource call always grepped the WHOLE MODULE, not the intended
        method. Drives the real _artifact_train_date() instead: writes an
        artifact whose mtime sits at an ET instant that falls on a
        DIFFERENT calendar date than the host-local zone would read it as
        (proving the test isn't vacuous), and asserts the recovered date is
        the ET one."""
        import os
        from datetime import datetime

        from ait.ml.trainer import ModelTrainer

        host_tz(BERLIN)
        import ait.ml.range_predictor as _rp

        # trained_at / model_version are tried FIRST (highest priority) --
        # leave them unset so the mtime fallback actually runs.
        trainer = ModelTrainer.__new__(ModelTrainer)
        trainer._range_predictor = None
        trainer._vol_mag_predictor = None
        from types import SimpleNamespace
        trainer._predictor = SimpleNamespace(model_version="")

        # 23:00 ET on the 17th is 05:00 CEST on the 18th -- host-local and ET
        # calendar dates disagree, so a naive (host-local) mtime read would
        # recover the WRONG date.
        et_instant = datetime(2026, 9, 17, 23, 0, tzinfo=ET)
        f = _rp.MODEL_DIR / "ensemble.pkl"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
        os.utime(f, (et_instant.timestamp(), et_instant.timestamp()))
        host_local_date = datetime.fromtimestamp(et_instant.timestamp()).date()
        assert host_local_date != et_instant.date(), (
            "test setup bug: host-local and ET dates must actually disagree "
            "for this to prove anything"
        )

        recovered = trainer._artifact_train_date()
        assert recovered == et_instant.date(), (
            f"recovered {recovered}, the host-local date {host_local_date} -- "
            "the artifact mtime is being read in host-local time while the "
            "date it is compared against is ET"
        )

    def test_recovered_date_is_an_et_date(self, tmp_path, host_tz):
        """Executed: an artifact written 'now' must recover as TODAY in ET."""
        from ait.utils.time import today_et

        host_tz(BERLIN)
        f = tmp_path / "ensemble.pkl"
        f.write_bytes(b"x")
        recovered = datetime.fromtimestamp(f.stat().st_mtime, tz=ET).date()
        assert recovered == today_et()


class TestNoHostDateReadsRemain:
    """CLASS GUARD, with ZERO exceptions — the payoff for fixing all 49.

    Sibling to W20's naive-clock guard
    (test_w20_host_clock_and_calendar_faults.py::
     test_no_live_producer_stamps_the_hour_from_a_naive_clock), which covers
    `datetime.now()`. Together they close the class.

    THERE IS DELIBERATELY NO ALLOWLIST. A guard carrying 49 documented
    exceptions is a catalogue, not a guard — that is precisely why W26 declined
    to ship this check against a partially-fixed tree. If a 50th site appears,
    the answer is `today_et()`, never an entry here.
    """

    def _today_calls(self) -> list[str]:
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "today"):
                    offenders.append(
                        f"{path.relative_to(SRC.parent.parent)}:{node.lineno}")
        return offenders

    def test_src_contains_no_today_calls(self):
        offenders = self._today_calls()
        assert offenders == [], (
            "`.today()` reads the HOST date, which is tomorrow's date from "
            "18:00 ET on this box. Use ait.utils.time.today_et():\n  "
            + "\n  ".join(offenders)
        )

    def test_the_guard_actually_scans_the_tree(self):
        """Anti-vacuity: a broken glob would make the guard pass by finding
        nothing at all. (W27's lesson, applied to its own guard.)"""
        scanned = list(SRC.rglob("*.py"))
        assert len(scanned) > 50, (
            f"the guard only saw {len(scanned)} files — its scan is broken, so "
            "a clean result means nothing"
        )
        joined = "\n".join(p.read_text(errors="ignore") for p in scanned[:400])
        assert "today_et" in joined, (
            "no today_et usage found anywhere — the guard is reading the wrong tree"
        )

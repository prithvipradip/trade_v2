"""W24: the market-open race in MarketScheduler.wait_for_phase.

Confirmed live 2026-09-16, 09:30:00 ET, from the bot's own log line — the
smoking gun, not an inference:

    {"target": "market_open", "wait_seconds": 3600, "current_phase":
     "market_open", "event": "waiting_for_phase"}

wait_for_phase, get_current_phase and next_market_open each independently
called now_et() — up to five unsynchronized wall-clock reads in one loop
iteration. When two of them landed on opposite sides of the 09:30:00.000 ET
boundary, the function could log current_phase=market_open (a late read)
while having already committed to wait_seconds=3600 (an earlier read that
still saw pre-market, whose branch treats "today's open already passed" as
certain and hands off to next_market_open() for TOMORROW's open) — and
sleeps, with no re-check before the await. Recovered only because the
supervisor's independent heartbeat-staleness check happened to tick 7
seconds later; not a safeguard this function knows about or should rely on.

THE FIX is structural, not a patch of the one branch that fired: exactly one
`now` per loop iteration, threaded through every decision in that iteration
(the phase check, the boundary comparisons, next_market_open, the log line).
So the tests here do not mock away the race and assert the "nice" answer —
they drive the REAL now_et()-reading code with a fake clock that can supply
DIFFERENT values across separate now_et() calls, and prove there is now only
ONE such call per iteration by construction: feeding the exact sequence of
instants observed live must no longer be able to produce a contradiction.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from ait.bot import scheduler as sched_mod
from ait.bot.scheduler import MarketScheduler, TradingPhase
from ait.utils.time import next_market_open

ET = ZoneInfo("US/Eastern")

# A normal Wednesday with a normal Thursday after it — no calendar-holiday
# special-casing needed for these boundary tests.
DAY = "2026-09-16"
NEXT_DAY = "2026-09-17"


def _et(hms: str, day: str = DAY) -> datetime:
    return datetime.fromisoformat(f"{day}T{hms}").replace(tzinfo=ET)


@pytest.fixture
def queued_clock(monkeypatch):
    """Fake now_et() that pops a queued instant per call, in scheduler's own
    namespace (scheduler.py did `from ait.utils.time import now_et`, so the
    name to patch lives in ait.bot.scheduler, not ait.utils.time).

    Exhaustion raises rather than silently returning a stale/default value —
    a test that runs more iterations than it staged should fail loudly, not
    pass on an accident.
    """
    def _install(instants: list[datetime]):
        queue = list(instants)
        calls: list[datetime] = []

        def _next() -> datetime:
            if not queue:
                raise AssertionError(
                    f"now_et() called more times than staged; consumed "
                    f"{len(calls)}: {calls}")
            v = queue.pop(0)
            calls.append(v)
            return v

        monkeypatch.setattr(sched_mod, "now_et", _next)
        return calls
    return _install


@pytest.fixture
def fast_sleep(monkeypatch):
    """Replace asyncio.sleep with a no-op that RECORDS requested durations,
    so tests never block on real time and can assert on what was requested."""
    durations: list[float] = []

    async def _sleep(seconds):
        durations.append(seconds)

    monkeypatch.setattr(sched_mod.asyncio, "sleep", _sleep)
    return durations


class TestTheHarnessItself:
    """If the fake clock doesn't deliver a distinct value per call, in
    order, every test below is vacuous."""

    def test_queue_delivers_values_in_order_and_errors_on_exhaustion(
            self, queued_clock):
        calls = queued_clock([_et("09:00:00"), _et("09:00:01")])
        assert sched_mod.now_et() == _et("09:00:00")
        assert sched_mod.now_et() == _et("09:00:01")
        with pytest.raises(AssertionError, match="more times than staged"):
            sched_mod.now_et()
        assert calls == [_et("09:00:00"), _et("09:00:01")]


class TestGetCurrentPhaseBoundaries:
    """Direct, synchronous, explicit `now` — the whole point of the optional
    parameter is that these need no clock faking at all."""

    ms = MarketScheduler()

    def test_exactly_market_open_is_market_open_not_pre_market(self):
        assert self.ms.get_current_phase(_et("09:30:00.000000")) == \
            TradingPhase.MARKET_OPEN

    def test_one_microsecond_before_open_is_still_pre_market(self):
        assert self.ms.get_current_phase(_et("09:29:59.999999")) == \
            TradingPhase.PRE_MARKET

    def test_exactly_pre_market_start_is_pre_market_not_off_hours(self):
        assert self.ms.get_current_phase(_et("09:00:00.000000")) == \
            TradingPhase.PRE_MARKET

    def test_exactly_close_is_post_market_not_market_open(self):
        assert self.ms.get_current_phase(_et("16:00:00.000000")) == \
            TradingPhase.POST_MARKET

    def test_no_arg_still_works_matching_prior_behaviour(self, queued_clock):
        queued_clock([_et("10:00:00")])
        assert self.ms.get_current_phase() == TradingPhase.MARKET_OPEN


class TestNextMarketOpenExplicitNow:
    """The bug was never in what this returns — only in wait_for_phase
    treating a racy read of 'has today's open passed' as certain."""

    def test_just_before_open_returns_today(self):
        got = next_market_open(_et("09:29:59.999999"))
        assert got == _et("09:30:00.000000")

    def test_at_or_after_open_returns_tomorrow(self):
        got = next_market_open(_et("09:30:00.000000"))
        assert got == _et("09:30:00.000000", day=NEXT_DAY)

    def test_no_arg_still_works_matching_prior_behaviour(self, monkeypatch):
        # next_market_open is DEFINED in ait.utils.time and resolves now_et()
        # from that module's own namespace — a different target than
        # scheduler's imported copy, so queued_clock (which patches
        # ait.bot.scheduler.now_et) has no effect here.
        import ait.utils.time as time_mod
        monkeypatch.setattr(time_mod, "now_et", lambda: _et("03:00:00"))
        assert next_market_open() == _et("09:30:00.000000")


class TestWaitForPhaseNoLongerRaces:
    """Drive the REAL wait_for_phase with a fake clock that can answer
    DIFFERENTLY across calls, and prove exactly one call happens per
    iteration — which is what makes the production contradiction
    impossible now, not merely unlikely."""

    def test_todays_exact_incident_no_longer_produces_a_stale_hour_sleep(
            self, queued_clock, fast_sleep, monkeypatch):
        """The precise reproduction: today's own two observed instants.

        Under the OLD code, current_phase for the log line could read
        "market_open" while wait_seconds=3600 had already been committed
        from an earlier read. Feed those same two instants here — one just
        before the boundary, one just after (09:30:00.071, today's actual
        value) — and prove the function converges in exactly two single-read
        iterations with NO contradiction and no hour-long wait at any point.
        """
        instants = [_et("09:29:59.998000"), _et("09:30:00.071000")]
        queued_clock(instants)
        logged = []
        monkeypatch.setattr(
            sched_mod.log, "info",
            lambda event, **kw: logged.append((event, kw)))

        ms = MarketScheduler()
        asyncio.run(ms.wait_for_phase(TradingPhase.MARKET_OPEN))

        # Exactly one now_et() call per iteration: two instants staged, both
        # consumed, nothing left over (the harness would have raised
        # otherwise) — this IS the fix: one read, not several independent
        # ones that can disagree.
        assert len(logged) == 1, "expected exactly one wait-and-retry cycle"
        event, kw = logged[0]
        assert event == "waiting_for_phase"
        # The contradiction from production — current_phase=market_open
        # alongside wait_seconds=3600 — must be structurally unreachable:
        # this log line's current_phase and wait_seconds are derived from
        # the SAME now (09:29:59.998, still pre-market), so they agree.
        assert kw["current_phase"] == "pre_market"
        assert kw["wait_seconds"] <= 1, (
            f"expected a sub-second catch-up wait, got {kw['wait_seconds']}s "
            "— the stale-hour-sleep bug is back")
        assert fast_sleep == [pytest.approx(0.002, abs=0.01)] or \
            all(d <= 1 for d in fast_sleep)

    def test_already_open_on_first_read_returns_with_no_sleep_at_all(
            self, queued_clock, fast_sleep):
        """The common case: the caller's own check already agrees with a
        fresh read, so the branch logic never runs at all."""
        queued_clock([_et("10:15:00")])
        ms = MarketScheduler()
        asyncio.run(ms.wait_for_phase(TradingPhase.MARKET_OPEN))
        assert fast_sleep == []

    @pytest.mark.parametrize("offset_us", [-100_000, -50_000, -1, 0])
    def test_boundary_sweep_never_produces_an_hour_long_wait(
            self, offset_us, queued_clock, fast_sleep, monkeypatch):
        """Sweep `now` from 100ms before the open through the exact instant.

        For every value on the "not yet open" side, the computed wait must be
        the short remainder to 09:30:00, never anything resembling the
        old bug's ~3600s fallback. For the exact boundary itself, the
        function must return immediately.
        """
        now = _et("09:30:00.000000") + timedelta(microseconds=offset_us)
        logged = []
        monkeypatch.setattr(
            sched_mod.log, "info",
            lambda event, **kw: logged.append((event, kw)))

        if offset_us == 0:
            queued_clock([now])
            ms = MarketScheduler()
            asyncio.run(ms.wait_for_phase(TradingPhase.MARKET_OPEN))
            assert fast_sleep == []
            return

        # One "not yet" read, then an "already open" read so the loop
        # terminates instead of needing a real sleep in the test.
        queued_clock([now, _et("09:30:00.500000")])
        ms = MarketScheduler()
        asyncio.run(ms.wait_for_phase(TradingPhase.MARKET_OPEN))
        assert len(logged) == 1
        wait_seconds = logged[0][1]["wait_seconds"]
        assert wait_seconds <= 1, (
            f"now={now.isoformat()} produced wait_seconds={wait_seconds}; "
            "expected the sub-second remainder to the open, not the stale "
            "'today already passed' fallback")

    def test_happy_path_well_before_open_computes_a_sane_wait(
            self, queued_clock, fast_sleep, monkeypatch):
        """Non-regression: called comfortably before the open (e.g. from
        _pre_market()), the wait should be the genuine ~30 minutes, and the
        function should still eventually return once phase flips."""
        logged = []
        monkeypatch.setattr(
            sched_mod.log, "info",
            lambda event, **kw: logged.append((event, kw)))
        queued_clock([_et("09:00:00"), _et("09:30:00.100000")])
        ms = MarketScheduler()
        asyncio.run(ms.wait_for_phase(TradingPhase.MARKET_OPEN))
        assert len(logged) == 1
        wait_seconds = logged[0][1]["wait_seconds"]
        assert 1799 <= wait_seconds <= 1800, wait_seconds
        assert logged[0][1]["current_phase"] == "pre_market"

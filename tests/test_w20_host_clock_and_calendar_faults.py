"""W20: the naive-local-clock class, and the fail-open it was hiding.

WHY THIS FILE PINS A HOST TIMEZONE. `test_r16_tail_fixes.py:711`
(`test_calendar_outage_falls_back_to_the_clock_not_forever`) already covers
`_in_post_market_window`'s calendar fallback — and it passed for months while
that fallback was wrong on this machine. It passes EXPLICIT NAIVE DATETIMES,
so it only ever exercised the arithmetic, never the `datetime.now()` path
where the host timezone enters. The box moved to CEST (= ET+6) and nothing
noticed. Every test here therefore pins a real, non-ET `TZ` and drives the
no-argument path.

WHAT W20b ALONE DID NOT FIX. Making `_in_post_market_window` timezone-correct
fixes the FUNCTION and leaves the hole, because "False, we are not
post-market" is the *right* answer mid-RTH. The fail-open is in the CALLER:
`except: market_open = False` turned "I don't know" into "safe to restart".
Probed 2026-09-16 with the calendar faulted: 09:05, 10:05, 11:05, 13:05 and
15:05 ET all took the restart branch — on a live bot, mid-session, with
positions open. Worse, W20b made 10:05 ET *newly* reachable: on the CEST box
the old naive fallback protected 10:00-10:15 ET by accident, and correcting it
removed that accident without closing the hole. Hence W20d, pinned below.
"""
from __future__ import annotations

import os
import time as _time
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

import ait.orchestration.master as M

ET = ZoneInfo("US/Eastern")

# The pinned host zone, read by _naive_local below. TZ + tzset moves the REAL
# process clock, which is the strongest form of this pin — but tzset is POSIX
# only, and Windows ignores TZ, so on the Windows box (the one that runs live)
# the whole file errored in setup AND teardown. Holding the zone explicitly
# makes every test below work on either host; on POSIX the real clock is still
# moved as well, so nothing is weakened there.
_HOST: dict = {"zone": None}


def _naive_local(instant):
    """`instant` as NAIVE LOCAL time on the pinned host — the quantity every
    bug in this file turns on."""
    zone = _HOST["zone"]
    return (instant.astimezone(zone) if zone
            else instant.astimezone()).replace(tzinfo=None)


@pytest.fixture
def host_tz(monkeypatch):
    """Pin the process's LOCAL timezone, the way the box's own clock is set.

    Yields a setter. Restores via monkeypatch (+ an explicit tzset where the
    platform has one) so no later test in the session inherits a foreign clock.
    """
    def _set(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        if hasattr(_time, "tzset"):
            _time.tzset()
        _HOST["zone"] = ZoneInfo(name)
    yield _set
    _HOST["zone"] = None
    monkeypatch.undo()
    if hasattr(_time, "tzset"):
        _time.tzset()


@pytest.fixture
def frozen(monkeypatch):
    """Pin the real instant, leaving timezone conversion genuine.

    `datetime.now(tz)` returns the instant expressed in `tz`; `datetime.now()`
    with no argument returns it as NAIVE LOCAL time, which is exactly the
    quantity every bug in this file turns on.
    """
    def _at(instant_et: datetime, *mods):
        assert instant_et.tzinfo is not None

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is None:
                    return _naive_local(instant_et)
                return instant_et.astimezone(tz)

        for mod in mods or (M,):
            monkeypatch.setattr(mod, "datetime", _Frozen)
        return instant_et
    return _at


def _fault_the_calendar(monkeypatch):
    """Make every NYSE calendar entry point raise, as a broken package would."""
    import ait.utils.time as _t

    def _boom(*a, **kw):
        raise RuntimeError("calendar unavailable")

    for fn in ("is_trading_day", "is_market_open", "get_market_close"):
        monkeypatch.setattr(_t, fn, _boom)


class TestTheHarnessItself:
    """If the TZ pin silently no-ops, every test below becomes vacuous."""

    def test_tz_pin_actually_moves_the_naive_local_clock(self, host_tz):
        instant = datetime.now(ET)
        host_tz("Europe/Berlin")
        berlin = _naive_local(instant)
        host_tz("US/Eastern")
        eastern = _naive_local(instant)
        drift_h = round((berlin - eastern).total_seconds() / 3600)
        assert drift_h in (5, 6), (
            f"TZ pin did not take effect (drift {drift_h}h); the CEST/ET gap "
            "is 6h in summer, 5h in winter"
        )

    @pytest.mark.skipif(not hasattr(_time, "tzset"), reason="POSIX only")
    def test_tz_pin_also_moves_the_real_process_clock(self, host_tz):
        """Where the platform allows it, the pin is the real thing and not
        just the seam — this is what proves the seam models the host."""
        host_tz("Europe/Berlin")
        berlin = datetime.now()
        host_tz("US/Eastern")
        eastern = datetime.now()
        assert round((berlin - eastern).total_seconds() / 3600) in (5, 6)

    def test_frozen_clock_separates_local_from_et(self, host_tz, frozen):
        host_tz("Europe/Berlin")
        frozen(datetime(2026, 9, 16, 15, 5, tzinfo=ET))
        assert M.datetime.now(ET).hour == 15, "ET hour must survive"
        assert M.datetime.now().hour == 21, "naive local must read CEST"


class TestPostMarketWindowIsHostIndependent:
    """W20b: the calendar fallback must protect 16:00-16:15 ET on ANY host."""

    @pytest.mark.parametrize("tz", ["Europe/Berlin", "UTC", "US/Eastern",
                                    "Asia/Tokyo"])
    def test_protects_just_after_the_ET_close_everywhere(
            self, host_tz, frozen, monkeypatch, tz):
        host_tz(tz)
        _fault_the_calendar(monkeypatch)
        frozen(datetime(2026, 9, 16, 16, 5, tzinfo=ET))
        assert M._in_post_market_window() is True, (
            f"16:05 ET must be inside the post-market window on a {tz} host"
        )

    @pytest.mark.parametrize("tz", ["Europe/Berlin", "UTC", "US/Eastern",
                                    "Asia/Tokyo"])
    def test_does_not_protect_mid_morning_anywhere(
            self, host_tz, frozen, monkeypatch, tz):
        """The accidental CEST window. 10:05 ET is 16:05 *local* in Berlin —
        the pre-fix code read that naive hour and called it post-market."""
        host_tz(tz)
        _fault_the_calendar(monkeypatch)
        frozen(datetime(2026, 9, 16, 10, 5, tzinfo=ET))
        assert M._in_post_market_window() is False, (
            f"10:05 ET is mid-RTH, not post-market, on a {tz} host"
        )

    def test_fallback_does_not_import_the_broken_package(self, monkeypatch,
                                                         host_tz, frozen):
        """The fallback must not resolve ET through `ait.utils.time`, which
        imports pandas_market_calendars at module scope — the very package
        whose failure lands us here."""
        host_tz("Europe/Berlin")
        import ait.utils.time as _t
        _fault_the_calendar(monkeypatch)
        monkeypatch.delattr(_t, "ET", raising=False)
        monkeypatch.delattr(_t, "now_et", raising=False)
        frozen(datetime(2026, 9, 16, 16, 5, tzinfo=ET))
        assert M._in_post_market_window() is True


class TestClockProxy:
    """W20d: the calendar-free "is it open?" fallback."""

    @pytest.mark.parametrize("tz", ["Europe/Berlin", "UTC", "Asia/Tokyo"])
    @pytest.mark.parametrize("hh,mm,expected", [
        (9, 29, False),   # one minute before the bell
        (9, 30, True),    # the open
        (12, 0, True),
        (15, 59, True),   # last minute of RTH
        (16, 0, False),   # the close
        (20, 0, False),
        (3, 0, False),
    ])
    def test_reads_ET_not_the_host_clock(self, host_tz, frozen, tz,
                                         hh, mm, expected):
        host_tz(tz)
        frozen(datetime(2026, 9, 16, hh, mm, tzinfo=ET))   # a Wednesday
        assert M._assume_market_open_on_clock() is expected

    def test_weekends_are_never_assumed_open(self, host_tz, frozen):
        host_tz("Europe/Berlin")
        frozen(datetime(2026, 9, 19, 12, 0, tzinfo=ET))    # Saturday
        assert M._assume_market_open_on_clock() is False


class TestFreshModelsRestartNeverFiresMidRTH:
    """W20d, the finding this wave actually turns on.

    A calendar fault must not restart a LIVE bot mid-session. Driven through
    the real `health_check`, with stop/start recorded rather than performed.
    """

    def _manager(self, monkeypatch, tmp_path, marker=True):
        mgr = M.BotManager()
        calls: list[str] = []
        monkeypatch.setattr(type(mgr), "is_running",
                            property(lambda self: True))
        mgr._proc = type("P", (), {"pid": 424242})()
        monkeypatch.setattr(mgr, "stop", lambda: calls.append("stop"))
        monkeypatch.setattr(mgr, "start", lambda: calls.append("start"))
        models = tmp_path / "models"
        models.mkdir(parents=True, exist_ok=True)
        if marker:
            (models / ".retrained").write_text("")
        monkeypatch.setattr(M, "ROOT", tmp_path)
        monkeypatch.setattr(M, "DATA_DIR", tmp_path / "data")
        (tmp_path / "data").mkdir(exist_ok=True)
        return mgr, calls

    @pytest.mark.parametrize("hh", [10, 11, 13, 15])
    def test_calendar_fault_does_not_restart_during_RTH(
            self, monkeypatch, tmp_path, host_tz, frozen, hh):
        host_tz("Europe/Berlin")
        _fault_the_calendar(monkeypatch)
        frozen(datetime(2026, 9, 16, hh, 5, tzinfo=ET))
        mgr, calls = self._manager(monkeypatch, tmp_path)
        mgr.health_check()
        assert calls == [], (
            f"a calendar fault restarted the live bot at {hh}:05 ET, "
            "mid-session, with positions open"
        )
        assert (tmp_path / "models" / ".retrained").exists(), (
            "the marker must survive the deferral so the restart still happens"
        )

    def test_calendar_fault_still_restarts_overnight(
            self, monkeypatch, tmp_path, host_tz, frozen):
        """The other half of the contract: never WEDGED by a fault."""
        host_tz("Europe/Berlin")
        _fault_the_calendar(monkeypatch)
        frozen(datetime(2026, 9, 16, 20, 5, tzinfo=ET))
        mgr, calls = self._manager(monkeypatch, tmp_path)
        mgr.health_check()
        assert calls == ["stop", "start"], (
            "fresh models must still load once the session is safely over"
        )
        assert not (tmp_path / "models" / ".retrained").exists()

    def test_calendar_fault_still_restarts_at_the_weekend(
            self, monkeypatch, tmp_path, host_tz, frozen):
        host_tz("Europe/Berlin")
        _fault_the_calendar(monkeypatch)
        frozen(datetime(2026, 9, 19, 12, 0, tzinfo=ET))    # Saturday noon
        mgr, calls = self._manager(monkeypatch, tmp_path)
        mgr.health_check()
        assert calls == ["stop", "start"]

    def test_hang_detection_stays_armed_through_a_calendar_fault(
            self, monkeypatch, tmp_path, host_tz, frozen):
        """W20d, second instance of the same class: the heartbeat check's
        market gate used to escape to the OUTER handler on a calendar fault,
        silently disarming hang detection for the whole outage."""
        host_tz("Europe/Berlin")
        _fault_the_calendar(monkeypatch)
        instant = datetime(2026, 9, 16, 13, 5, tzinfo=ET)
        frozen(instant)
        mgr, calls = self._manager(monkeypatch, tmp_path, marker=False)
        hb = tmp_path / "data" / "bot_heartbeat"
        hb.write_text("")
        stale = _naive_local(instant).timestamp() - 40 * 60
        os.utime(hb, (stale, stale))
        monkeypatch.setattr(M, "_alert", lambda *a, **k: None)
        mgr.health_check()
        assert calls == ["stop", "start"], (
            "a hung event loop mid-session must still be caught when the "
            "calendar is unavailable"
        )


class TestHourOfDayIsStampedInET:
    """W20a (R25 tz-01). `hour_of_day` is an ML FEATURE. The historical corpus
    was written on an ET box; this one is CEST (= ET+6). A model trained on
    the mix learns a split that encodes WHICH MACHINE WROTE THE ROW.

    Unlike the calendar findings, this one is not gated on any fault — every
    row written on this host since 2026-09-14 already carried it.
    """

    def test_persisted_entry_signals_carry_the_ET_hour(self, host_tz, frozen,
                                                       monkeypatch):
        import pandas as pd
        import ait.utils.time as _t
        from ait.bot.orchestrator import TradingOrchestrator

        host_tz("Europe/Berlin")
        frozen(datetime(2026, 9, 16, 11, 5, tzinfo=ET), _t)

        orch = object.__new__(TradingOrchestrator)     # no __init__: no IBKR
        orch._entry_feature_snap = {"SPY": pd.Series({"vix": 14.0})}

        import json
        out = json.loads(orch._entry_signals_json("SPY"))
        assert "hour_of_day" in out, "feature missing — check the test setup"
        assert out["hour_of_day"] == 11.0, (
            f"persisted hour was {out['hour_of_day']}, the CEST wall clock; "
            "the historical corpus is ET"
        )

    def test_no_live_producer_stamps_the_hour_from_a_naive_clock(self):
        """Pins the CLASS, not the two instances. W17-01b fixed two of four
        sites sharing a field and claimed the class; this is the cheap guard
        that would have caught it."""
        import inspect
        import re
        import ait.bot.orchestrator as _o

        src = inspect.getsource(_o)
        naive = [ln.strip() for ln in src.splitlines()
                 if re.search(r"\.now\(\)\.hour", ln)
                 and not ln.strip().startswith("#")]
        assert naive == [], (
            f"live code still stamps an hour from a naive clock: {naive}"
        )
        assert src.count("_now_et().hour") == 3, (
            "expected three live hour producers, all on now_et(): the inference "
            "context, the persisted entry_signals, and the learning hour gate"
        )

    def test_backtest_constant_is_parity_with_the_live_producers(self):
        """Parity check required by the W20 plan. engine.py's hardcoded 10 is
        a CONSTANT, not a clock read, so it has no timezone bug and must NOT
        change: it is an ET-plausible RTH hour, and fixing the live producers
        is precisely what restores agreement with it. Pre-fix, live rows on
        this host wrote 15..22 against the backtest's 10."""
        import inspect
        import ait.backtesting.engine as _e

        src = inspect.getsource(_e)
        assert 'meta_ctx.setdefault("hour_of_day", 10)' in src
        assert 9 <= 10 < 16, "the backtest constant must be inside ET RTH"


class TestSessionsUntilFallback:
    """W20b/c: the pre-event blackout's session counter."""

    def _orch(self):
        from ait.bot.orchestrator import TradingOrchestrator
        return TradingOrchestrator

    def test_anchors_on_the_ET_date_not_the_host_date(self, host_tz, frozen,
                                                      monkeypatch):
        """On a CEST box the local date rolls over at 18:00 ET, so an evening
        run counted sessions from TOMORROW."""
        import ait.bot.orchestrator as _o

        host_tz("Europe/Berlin")
        # FRIDAY 19:30 ET = 01:30 CEST Saturday. The anchor must be Friday.
        # Chosen because a weekend follows: anchoring on Saturday instead
        # SKIPS PAST it and finds real sessions, which is what makes the two
        # anchors give different answers. (An ordinary midweek evening does
        # not discriminate — both anchors return 0 — so this test would have
        # passed pre-fix and proved nothing.)
        frozen(datetime(2026, 9, 18, 19, 30, tzinfo=ET), _o)
        assert _o.datetime.now().date().day == 19, (
            "harness: the host's local date must have rolled over, else this "
            "test cannot distinguish the two anchors"
        )
        # Fri anchor: Sat, Sun, Mon -> 1 trading day -> 0 sessions between.
        # Sat anchor: Sun, Mon, Tue -> 2 trading days -> 1 session. The
        # consumer blocks when `sessions <= blackout`, so with a blackout of 0
        # the wrong anchor reports 1 and the pre-event gate FAILS OPEN.
        assert self._orch()._sessions_until(3) == 0

    def test_calendar_fault_errs_toward_blocking_not_entering(
            self, host_tz, frozen, monkeypatch):
        """W20c. The consumer blocks when `sessions <= blackout`, so a fallback
        that INFLATES the count admits an entry into the event the gate exists
        to avoid. The old `day.weekday() < 5` did exactly that — the same
        heuristic W15 removed from the primary path, left behind in the
        fallback. A holiday is unknowable without the calendar, so the only
        safe fallback is the one that blocks."""
        import ait.utils.time as _t

        host_tz("Europe/Berlin")
        monkeypatch.setattr(_t, "is_trading_day",
                            lambda *a, **k: (_ for _ in ()).throw(
                                RuntimeError("calendar unavailable")))
        sessions = self._orch()._sessions_until(5)
        assert sessions == 0, (
            f"calendar fault produced {sessions} sessions; anything above a "
            "blackout of 0 would let a credit entry through"
        )


class TestLearningHourGateConvention:
    """W20a, third instance — the one the class guard found.

    Two halves of one gate must agree on a timezone: `analyzer.py` derives the
    blocked hour from `trades.entry_time`, `adaptor._block_hour` accepts only
    `9 <= hour <= 16`, and the orchestrator compares the live clock against the
    result. The first two are ET-RTH conventions; the live read was naive
    local. This pins the agreement rather than any one side of it.
    """

    def test_block_hour_only_accepts_ET_regular_hours(self):
        from ait.learning.adaptor import StrategyAdaptor

        adaptor = object.__new__(StrategyAdaptor)
        adaptor._blocked_hours = set()
        insight = type("I", (), {"insight": "x", "confidence": 0.5})()

        assert adaptor._block_hour(8, insight) is None, "08:00 ET is pre-open"
        assert adaptor._block_hour(10, insight) is not None
        adaptor._blocked_hours.clear()
        for cest_hour in (17, 18, 19, 20, 21, 22):
            assert adaptor._block_hour(cest_hour, insight) is None, (
                f"{cest_hour} is what 11:00-16:00 ET reads on a CEST host; "
                "the writer rejects it, so most of the session was unblockable"
            )

    def test_the_live_read_uses_the_same_convention_as_the_writer(
            self, host_tz, frozen):
        """The whole defect in one assertion: mid-session, the hour the gate
        reads must be one the gate's writer could actually have blocked."""
        import ait.utils.time as _t

        host_tz("Europe/Berlin")
        frozen(datetime(2026, 9, 16, 14, 30, tzinfo=ET), _t)   # mid-RTH
        live_hour = _t.now_et().hour
        assert live_hour == 14
        assert 9 <= live_hour <= 16, (
            f"live gate read {live_hour}; _block_hour only ever stores 9..16, "
            "so a naive-local read can never match a learned block"
        )

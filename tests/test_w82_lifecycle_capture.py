"""W82: lifecycle capture of option bid/ask bars, 7-45 DTE friction bands, TWS / IB Gateway ports.

  * ``ait.data.ib_ports``: a closed configured port falls back to the TWIN endpoint of the SAME
    account type (paper 7497 <-> 4002, live 7496 <-> 4001), never across account types
  * friction bands: edges (2, 6, 13, 29, 45); only 0-6 DTE is floored / extrapolated, so a 7+
    band without a fit behaves exactly as the all-DTE model did before
  * lifecycle planning: stage A (expiring today..+N, perishable) and stage B (far-listed
    expiries at 15-21 DTE whose DTE 31-45 tail is not stored, capped), strikes as the union of
    the per-day rule over the underlying's path
  * the collector against a fake broker: 30 D requests, a capture log, idempotent re-runs
  * the CLI / supervisor job hand the collector the port that answered
No real broker is used anywhere.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from ait.backtesting.friction import (
    FLOOR_MAX_DTE,
    SHORT_BAND_EDGES,
    BandedFriction,
    LegSpreadModel,
    band_dte_range,
    friction_band,
)
from ait.data import ib_ports as ports
from ait.data import short_dte_collector as sc
from ait.data.option_spread_sampler import (
    SHORT_DTE_DELTA_BUCKETS,
    SHORT_DTE_MAX,
    SHORT_DTE_WINGS,
    plan_short_dte_strikes,
    plan_strikes,
)
from ait.utils.time import ET

from tests.test_w76_short_dte_friction import (  # noqa: F401 - shared fake broker + fixtures
    SPOT, TODAY, WINDOW, FakeIB, store)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ait"

EXP_A = TODAY                              # Fri 2026-10-09, expires today (0 DTE)
EXP_MID = date(2026, 10, 12)               # Mon, 3 DTE: neither stage A (N=0) nor in the far window
EXP_B = date(2026, 10, 26)                 # Mon, 17 DTE: inside the 15-21 far window
EXP_B2 = date(2026, 10, 27)                # 18 DTE
EXP_FAR = date(2026, 11, 20)               # 42 DTE: not yet in the window


# --------------------------------------------------------------------------
# ports: TWS and IB Gateway
# --------------------------------------------------------------------------

class TestPorts:
    def test_twin_endpoints_never_cross_account_types(self):
        assert ports.TWIN_PORTS == {7497: 4002, 4002: 7497, 7496: 4001, 4001: 7496}
        for paper, live in ((7497, 7496), (4002, 4001)):
            assert ports.TWIN_PORTS[paper] not in ports.LIVE_PORTS
            assert ports.TWIN_PORTS[live] in ports.LIVE_PORTS

    def test_candidates_are_the_configured_port_then_its_twin(self):
        assert ports.port_candidates(7497) == [7497, 4002]
        assert ports.port_candidates(4002) == [4002, 7497]
        assert ports.port_candidates(7496) == [7496, 4001]
        assert ports.port_candidates(5555) == [5555]            # unknown port: no guess

    def test_pick_prefers_the_configured_port_then_the_twin_then_none(self):
        assert ports.pick_listening_port(7497, lambda p: p in (7497, 4002)) == 7497
        assert ports.pick_listening_port(7497, lambda p: p == 4002) == 4002
        assert ports.pick_listening_port(4002, lambda p: p == 7497) == 7497
        assert ports.pick_listening_port(7497, lambda p: False) is None

    def test_a_paper_setup_never_probes_a_live_port(self):
        probed: list[int] = []

        def probe(p):
            probed.append(p)
            return False
        ports.pick_listening_port(7497, probe)
        assert set(probed) == {7497, 4002} and not set(probed) & ports.LIVE_PORTS

    def test_labels(self):
        assert ports.endpoint_label(7497) == "TWS (paper)"
        assert ports.endpoint_label(4002) == "IB Gateway (paper)"
        assert ports.endpoint_label(4001) == "IB Gateway (live)"
        assert ports.endpoint_label(9999) == "port 9999"

    def test_module_is_pure_stdlib(self):
        tree = ast.parse((SRC / "data" / "ib_ports.py").read_text())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.level == 0:
                mods.add((n.module or "").split(".")[0])
        assert mods <= {"__future__", "collections", "socket"}, mods


def _script():
    spec = importlib.util.spec_from_file_location("calib_w82", ROOT / "scripts" /
                                                  "calibrate_option_spreads.py")
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = ["calibrate_option_spreads.py"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    return mod


class TestCli:
    def test_port_falls_back_to_the_gateway_when_tws_is_closed(self, capsys):
        mod = _script()
        assert mod._resolve_port(7497, "127.0.0.1", probe=lambda p: p == 4002) == 4002
        assert "7497" in capsys.readouterr().out

    def test_the_configured_port_is_kept_when_it_answers_or_nothing_does(self):
        mod = _script()
        assert mod._resolve_port(7497, "h", probe=lambda p: True) == 7497
        assert mod._resolve_port(7497, "h", probe=lambda p: False) == 7497   # connect will report

    def test_lifecycle_defaults_to_30_day_history_and_an_explicit_flag_wins(self):
        mod = _script()
        assert mod._parse_args(["--source", "ib-bars", "--short-dte"]).short_dte_duration == "7 D"
        assert mod._parse_args(["--source", "ib-bars", "--short-dte", "--lifecycle"
                                ]).short_dte_duration == "30 D"
        a = mod._parse_args(["--source", "ib-bars", "--short-dte", "--lifecycle",
                             "--short-dte-duration", "14 D"])
        assert a.short_dte_duration == "14 D"
        assert (a.expiring_days, a.thin_every, a.max_far_contracts) == (1, 3, 120)


# --------------------------------------------------------------------------
# friction bands
# --------------------------------------------------------------------------

class TestBands:
    def test_edges_and_collected_bands(self):
        assert SHORT_BAND_EDGES == (2, 6, 13, 29, 45) and FLOOR_MAX_DTE == 6
        assert sc.SHORT_BANDS == ("0-2", "3-6", "7-13", "14-29", "30-45")
        for b in sc.SHORT_BANDS:
            lo, hi = band_dte_range(b)
            assert friction_band(lo) == b and friction_band(hi) == b

    @staticmethod
    def _bf(bands=None):
        return BandedFriction(LegSpreadModel("const_usd", a=0.006), bands or {}, 0.01)

    @pytest.mark.parametrize("dte, floored", [(0, True), (6, True), (7, False), (13, False),
                                              (14, False), (29, False), (30, False), (45, False),
                                              (46, False), (200, False)])
    def test_only_0_to_6_dte_is_floored_and_extrapolated(self, dte, floored):
        v, ex = self._bf().half_spread_usd(0.30, dte)
        assert ex is floored
        assert v == pytest.approx(0.01 if floored else 0.006)

    def test_a_fitted_long_band_is_used_without_touching_its_neighbours(self):
        bf = self._bf({"14-29": LegSpreadModel("const_usd", a=0.02)})
        assert bf.half_spread_usd(0.30, 20) == (pytest.approx(0.02), False)
        assert bf.half_spread_usd(0.30, 10) == (pytest.approx(0.006), False)
        assert bf.half_spread_usd(0.30, 3) == (pytest.approx(0.01), True)

    def test_the_old_edges_behave_exactly_as_before(self):
        old = BandedFriction(LegSpreadModel("const_usd", a=0.006), {}, 0.01, edges=(2, 6))
        assert old.half_spread_usd(0.30, 1) == (pytest.approx(0.01), True)
        assert old.half_spread_usd(0.30, 4) == (pytest.approx(0.01), True)
        assert old.half_spread_usd(0.30, 7) == (pytest.approx(0.006), False)
        assert old.half_spread_usd(0.30, 40) == (pytest.approx(0.006), False)


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

LISTED = [float(k) for k in range(680, 801)]


def _path(n: int = 5, spot: float = SPOT, iv: float = 0.17, drift: float = 0.0):
    days, d = [], TODAY
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return [(day, spot + drift * i, iv) for i, day in enumerate(sorted(days))]


class TestHelpers:
    def test_duration_days(self):
        assert [sc.duration_days(x) for x in ("30 D", "7 D", "2 W", "1 M")] == [30, 7, 14, 30]

    def test_session_path_takes_the_last_close_of_each_session_and_the_latest_iv(self):
        up = {"2026-10-08T10:00:00-04:00": 700.0, "2026-10-08T15:00:00-04:00": 705.0,
              "2026-10-09T10:00:00-04:00": 710.0}
        ivs = pd.Series([0.20, 0.30], index=pd.to_datetime(["2026-10-07", "2026-10-09"]))
        got = sc.session_path(up, ivs, iv_default=0.18)
        assert got == [(date(2026, 10, 8), 705.0, 0.20), (date(2026, 10, 9), 710.0, 0.30)]

    def test_an_implausible_or_missing_iv_falls_back_to_the_default(self):
        up = {"2026-10-08T10:00:00-04:00": 700.0}
        assert sc.session_path(up, None, 0.19)[0][2] == 0.19
        bad = pd.Series([9.0], index=pd.to_datetime(["2026-10-08"]))
        assert sc.session_path(up, bad, 0.19)[0][2] == 0.19


class TestLifecycleStrikes:
    def test_union_of_the_per_day_rule_with_thinning(self):
        path = _path(5)
        got = dict(sc.lifecycle_strikes(EXP_A, "C", path, LISTED, today=TODAY, thin_every=1))
        want: dict[float, str] = {}
        for d, S, iv in path:
            near = [k for k in LISTED if abs(k / S - 1.0) <= sc.STRIKE_WINDOW]
            for k, role in plan_short_dte_strikes(S, iv, (EXP_A - d).days, near, "C"):
                want[k] = "short" if (role == "short" or want.get(k) == "short") else role
        assert got == want
        thin = dict(sc.lifecycle_strikes(EXP_A, "C", path, LISTED, today=TODAY, thin_every=3))
        assert set(thin) <= set(got)

    def test_a_drifting_underlying_widens_the_union(self):
        flat = sc.lifecycle_strikes(EXP_B, "P", _path(5, drift=0.0), LISTED, today=TODAY, thin_every=1)
        moved = sc.lifecycle_strikes(EXP_B, "P", _path(5, drift=-6.0), LISTED, today=TODAY, thin_every=1)
        assert len(moved) > len(flat)

    def test_above_6_dte_the_compact_delta_plan_is_used(self):
        path = _path(1)
        d, S, iv = path[0]
        want = plan_strikes(S, iv, (EXP_B - d).days, LISTED, "C",
                            buckets=SHORT_DTE_DELTA_BUCKETS, wings=SHORT_DTE_WINGS)
        got = sc.lifecycle_strikes(EXP_B, "C", path, LISTED, today=TODAY, thin_every=1)
        assert {k for k, _ in got} == {k for k, _ in want}
        assert (EXP_B - d).days > SHORT_DTE_MAX

    def test_a_strike_that_is_short_on_any_day_stays_short(self):
        # the same strike is a wing on one day (far from spot) and a short bucket on another
        path = [(date(2026, 10, 5), 700.0, 0.17), (date(2026, 10, 9), 740.0, 0.17)]
        out = dict(sc.lifecycle_strikes(EXP_B, "C", path, LISTED, today=TODAY, thin_every=1))
        assert "short" in out.values() and set(out.values()) <= {"short", "wing"}

    def test_only_listed_strikes_within_twelve_percent_of_that_days_spot(self):
        out = sc.lifecycle_strikes(EXP_A, "C", _path(3), [300.0, 500.0, 740.0, 741.0, 1200.0],
                                   today=TODAY, thin_every=1)
        assert out and all(abs(k / SPOT - 1.0) <= sc.STRIKE_WINDOW for k, _ in out)

    def test_thinning_keeps_the_most_recent_session(self):
        """The newest sessions are the closest to now (and to the contract's expiry): with
        thin_every=3 over 5 sessions the planned days are the last and the 4th from last."""
        path = _path(5)
        path[-1] = (path[-1][0], 780.0, 0.17)          # only the newest session is far away
        got = {k for k, _ in sc.lifecycle_strikes(EXP_A, "C", path, LISTED, today=TODAY,
                                                  thin_every=3)}
        assert any(k >= 775.0 for k in got), "the most recent session was not planned"

    def test_a_strike_short_on_one_day_and_wing_on_another_is_short_in_either_order(self):
        """Find, from the real per-day rule (with each day's own DTE), a strike that is a wing
        on one day and a short bucket on the other, in BOTH chronological orders, and check the
        union keeps 'short' (a wing label would bypass the delta test on its ITM rows)."""
        d1, d2 = _path(2)[0][0], _path(2)[1][0]

        def roles(spot, day):
            near = [k for k in LISTED if abs(k / spot - 1.0) <= sc.STRIKE_WINDOW]
            return dict(plan_strikes(spot, 0.17, (EXP_B - day).days, near, "C",
                                     buckets=SHORT_DTE_DELTA_BUCKETS, wings=SHORT_DTE_WINGS))
        spots = [float(x) for x in range(700, 781, 2)]
        for first, second in (("wing", "short"), ("short", "wing")):
            hit = next(((a, b, k) for a in spots for b in spots
                        for k, r in roles(a, d1).items()
                        if r == first and roles(b, d2).get(k) == second), None)
            assert hit is not None, f"no {first}->{second} overlap in the fixture"
            a, b, k = hit
            out = dict(sc.lifecycle_strikes(EXP_B, "C", [(d1, a, 0.17), (d2, b, 0.17)], LISTED,
                                            today=TODAY, thin_every=1))
            assert out[k] == "short", (first, second)

    def test_strikes_beyond_twelve_percent_are_not_requested_even_if_nothing_else_is_listed(self):
        """nearest_listed would pick a 19%-away strike if the window let it through."""
        assert sc.lifecycle_strikes(EXP_A, "P", _path(3), [600.0, 650.0], today=TODAY,
                                    thin_every=1) == []

    def test_sessions_after_today_or_after_the_expiry_are_ignored(self):
        path = _path(3) + [(TODAY + timedelta(days=3), 900.0, 0.17)]
        out = sc.lifecycle_strikes(EXP_A, "C", path, LISTED, today=TODAY, thin_every=1)
        assert out and all(k < 780 for k, _ in out)    # the 900 session would add 792+


def _samples(expiry: date, sessions: int, dte: int = 35) -> pd.DataFrame:
    return pd.DataFrame([{"expiry": expiry.isoformat(), "dte": dte,
                          "bar_time": f"2026-09-{10 + i:02d}T10:00:00-04:00"}
                         for i in range(sessions)])


class TestPlanLifecycle:
    @staticmethod
    def _exps(*ds):
        return [d.strftime("%Y%m%d") for d in ds]

    def _plan(self, exps, **kw):
        return sc.plan_lifecycle(_path(5), TODAY, self._exps(*exps), LISTED, **kw)

    def test_stage_a_is_today_through_n_days_dte_zero_first(self):
        lp = self._plan([EXP_A, EXP_MID, EXP_B], expiring_days=3)
        assert lp.stage_a == [EXP_A, EXP_MID] and lp.stage_b == [EXP_B]
        a = [c for c in lp.contracts if c[4] == sc.STAGE_A]
        assert a[0][0] == EXP_A                       # perishable first
        assert lp.contracts.index(a[-1]) < lp.contracts.index(
            next(c for c in lp.contracts if c[4] == sc.STAGE_B))

    def test_zero_expiring_days_is_today_only(self):
        assert self._plan([EXP_A, EXP_MID], expiring_days=0).stage_a == [EXP_A]

    def test_stage_b_is_only_the_15_to_21_dte_window(self):
        lp = self._plan([EXP_A, EXP_MID, EXP_B, EXP_FAR, date(2026, 10, 23)], expiring_days=0)
        assert lp.stage_b == [EXP_B]                  # 14 DTE (10-23) and 42 DTE (11-20) are out

    @pytest.mark.parametrize("dte, owed", [(14, False), (15, True), (21, True), (22, False)])
    def test_window_edges(self, dte, owed):
        e = TODAY + timedelta(days=dte)
        assert (self._plan([e], expiring_days=0).stage_b == [e]) is owed

    def test_a_logged_stage_b_capture_means_nothing_is_owed(self):
        log = pd.DataFrame([{"expiry": EXP_B.isoformat(), "stage": "B"}])
        assert self._plan([EXP_B], log=log).stage_b == []

    def test_a_stage_a_log_row_does_not_satisfy_stage_b(self):
        log = pd.DataFrame([{"expiry": EXP_B.isoformat(), "stage": "A"}])
        assert self._plan([EXP_B], log=log).stage_b == [EXP_B]

    def test_stored_far_tail_sessions_mean_nothing_is_owed(self):
        """A manual 30 D catch-up run stores the tail without writing a log row."""
        assert self._plan([EXP_B], samples=_samples(EXP_B, 5)).stage_b == []
        assert self._plan([EXP_B], samples=_samples(EXP_B, 4)).stage_b == [EXP_B]
        assert self._plan([EXP_B], samples=_samples(EXP_B, 9, dte=20)).stage_b == [EXP_B]

    def test_the_cap_defers_whole_expiries_but_always_takes_the_first(self):
        lp = self._plan([EXP_B, EXP_B2], expiring_days=0, max_far_contracts=1)
        assert lp.stage_b == [EXP_B] and lp.deferred_b == [EXP_B2]
        one = len([c for c in lp.contracts if c[0] == EXP_B])
        lp2 = self._plan([EXP_B, EXP_B2], expiring_days=0, max_far_contracts=one)
        assert lp2.stage_b == [EXP_B] and lp2.deferred_b == [EXP_B2]
        lp3 = self._plan([EXP_B, EXP_B2], expiring_days=0, max_far_contracts=10 * one)
        assert lp3.stage_b == [EXP_B, EXP_B2] and lp3.deferred_b == []

    def test_a_contract_is_requested_once_and_stage_a_wins(self):
        lp = self._plan([EXP_A, EXP_B], expiring_days=30, far_window=(15, 21))
        keys = [(c[0], c[1], c[2]) for c in lp.contracts]
        assert len(keys) == len(set(keys))
        assert all(c[4] == sc.STAGE_A for c in lp.contracts if c[0] == EXP_B)

    def test_unparseable_expiry_strings_are_ignored(self):
        lp = sc.plan_lifecycle(_path(5), TODAY, ["garbage", EXP_A.strftime("%Y%m%d")], LISTED)
        assert lp.stage_a == [EXP_A]


# --------------------------------------------------------------------------
# the collector against a fake broker
# --------------------------------------------------------------------------

class RecordingIB(FakeIB):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.durations: list[str] = []

    async def reqHistoricalDataAsync(self, contract, endDateTime, durationStr, barSizeSetting,
                                     whatToShow, useRTH, formatDate):
        self.durations.append(durationStr)
        return await super().reqHistoricalDataAsync(
            contract, endDateTime, durationStr, barSizeSetting, whatToShow, useRTH, formatDate)


async def _lifecycle(ib, store, **kw):
    kw.setdefault("now", datetime.combine(TODAY, time(23, 59), tzinfo=ET))
    sleep = AsyncMock()
    quiet: list[str] = []
    res = await sc.collect_lifecycle(ib, "QQQ", store, WINDOW, today=TODAY, sleep=sleep,
                                     out=quiet.append, expiring_days=kw.pop("expiring_days", 0),
                                     **kw)
    return res, sleep, quiet


def _ib(*exps):
    return RecordingIB(expirations=list(exps))


class TestCollectLifecycle:
    @pytest.mark.asyncio
    async def test_stages_are_requested_stored_and_logged(self, store):
        ib = _ib(EXP_A, EXP_MID, EXP_B)
        res, sleep, _ = await _lifecycle(ib, store)
        assert (res.stage_a, res.stage_b) == (1, 1) and res.errors == 0 and res.stored > 0
        df = store.load_spread_bar_samples("QQQ")
        assert {EXP_A.isoformat(), EXP_B.isoformat()} <= set(df["expiry"])
        assert EXP_MID.isoformat() not in set(df["expiry"])        # neither stage owns it
        log = store.load_capture_log("QQQ").set_index("expiry")
        assert log.loc[EXP_A.isoformat(), "stage"] == "A"
        assert log.loc[EXP_B.isoformat(), "stage"] == "B"
        assert int(log.loc[EXP_B.isoformat(), "dte_at_capture"]) == 17
        assert (log["lookback_days"] == 30).all() and (log["contracts"] > 0).all()
        assert len(sleep.call_args_list) == res.requested         # paced after every request

    @pytest.mark.asyncio
    async def test_every_request_asks_for_thirty_days(self, store):
        ib = _ib(EXP_A, EXP_B)
        await _lifecycle(ib, store)
        assert ib.durations and set(ib.durations) == {"30 D"}

    @pytest.mark.asyncio
    async def test_a_second_run_does_not_owe_stage_b_again(self, store):
        ib = _ib(EXP_A, EXP_B)
        first, _, _ = await _lifecycle(ib, store)
        n1 = len(store.load_spread_bar_samples("QQQ"))
        second, _, _ = await _lifecycle(_ib(EXP_A, EXP_B), store)
        assert first.stage_b == 1 and second.stage_b == 0 and second.stage_a == 1
        assert second.planned < first.planned
        assert len(store.load_spread_bar_samples("QQQ")) == n1     # idempotent insert

    @pytest.mark.asyncio
    async def test_dry_run_requests_nothing_and_writes_nothing(self, store):
        ib = _ib(EXP_A, EXP_B)
        res, sleep, quiet = await _lifecycle(ib, store, dry_run=True)
        assert res.dry_run and res.planned > 0 and res.stored == 0 and res.requested == 0
        assert store.load_spread_bar_samples("QQQ").empty and store.load_capture_log("QQQ").empty
        assert [c for c in ib.calls if c[2] in ("BID", "ASK")] == []
        assert any("stage A" in m and "stage B" in m for m in quiet)
        sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_cap_defers_and_reports_the_rest(self, store):
        res, _, _ = await _lifecycle(_ib(EXP_B, EXP_B2), store, max_far_contracts=1)
        assert (res.stage_b, res.deferred_b) == (1, [EXP_B2.isoformat()])
        log = store.load_capture_log("QQQ")
        assert set(log["expiry"]) == {EXP_B.isoformat()}

    @pytest.mark.asyncio
    async def test_a_failing_contract_backs_off_and_the_run_continues(self, store):
        probe = sc.plan_lifecycle(_path(5, iv=0.20), TODAY, [EXP_A.strftime("%Y%m%d")], FakeIB().strikes)
        _, right, k, *_ = probe.contracts[len(probe.contracts) // 2]
        ib = RecordingIB(expirations=[EXP_A], fail_on=(right, k))
        res, sleep, quiet = await _lifecycle(ib, store)
        assert res.errors >= 1 and res.stored > 0
        assert any(c.args[0] == 60.0 for c in sleep.call_args_list)
        assert any("pacing violation" in m for m in quiet)

    @pytest.mark.asyncio
    async def test_no_underlying_bars_is_reported_not_raised(self, store):
        class Empty(FakeIB):
            async def reqHistoricalDataAsync(self, contract, **kw):
                return []
        res, _, quiet = await _lifecycle(Empty(expirations=[EXP_A]), store)
        assert res.stored == 0 and any("no underlying" in m for m in quiet)

    @pytest.mark.asyncio
    async def test_the_status_block_carries_the_stage_counters(self, store):
        res, _, _ = await _lifecycle(_ib(EXP_A, EXP_B, EXP_B2), store, max_far_contracts=1)
        blk = sc.symbol_status(res, {}, {}, {}, 15)
        assert (blk["stage_a"], blk["stage_b"], blk["deferred_b"]) == (1, 1, [EXP_B2.isoformat()])

    @pytest.mark.asyncio
    async def test_the_original_collector_still_works_through_the_shared_path(self, store):
        res = await sc.collect_short_dte(FakeIB(), "QQQ", store, WINDOW, today=TODAY,
                                         sleep=AsyncMock(), out=lambda m: None,
                                         now=datetime.combine(TODAY, time(23, 59), tzinfo=ET))
        assert res.stored > 0 and res.stage_a == res.stage_b == 0
        assert store.load_capture_log("QQQ").empty              # only lifecycle runs log captures


class TestCaptureLogStorage:
    def test_roundtrip_and_replace_on_the_same_key(self, store):
        row = {"expiry": "2026-10-26", "stage": "B", "captured_on": "2026-10-09",
               "dte_at_capture": 17, "lookback_days": 30, "contracts": 40}
        assert store.record_captures("QQQ", [row]) == 1
        store.record_captures("QQQ", [{**row, "contracts": 44}])
        got = store.load_capture_log("QQQ")
        assert len(got) == 1 and int(got.iloc[0]["contracts"]) == 44

    def test_empty_inputs(self, store):
        assert store.record_captures("QQQ", []) == 0
        assert store.load_capture_log("QQQ").empty


# --------------------------------------------------------------------------
# the supervisor job
# --------------------------------------------------------------------------

class TestJob:
    @staticmethod
    def _mod():
        from ait.orchestration import short_dte_job as J
        return J

    def test_the_config_defaults_keep_the_original_behaviour(self):
        from ait.config.settings import DataCollectionConfig
        c = DataCollectionConfig()
        assert c.lifecycle is False and c.lifecycle_duration == "30 D"
        assert (c.lifecycle_expiring_days, c.lifecycle_thin_every,
                c.lifecycle_max_far_contracts) == (1, 3, 120)

    @pytest.mark.parametrize("bad", [dict(lifecycle_expiring_days=9), dict(lifecycle_thin_every=0),
                                     dict(lifecycle_max_far_contracts=999),
                                     dict(lifecycle_duration="month")])
    def test_bad_lifecycle_values_are_refused(self, bad):
        from ait.config.settings import DataCollectionConfig
        with pytest.raises(Exception):
            DataCollectionConfig(**bad)

    def test_the_command_carries_the_lifecycle_flags_and_the_port(self):
        from ait.config.settings import DataCollectionConfig
        J = self._mod()
        cfg = DataCollectionConfig(lifecycle=True, lifecycle_expiring_days=2,
                                   lifecycle_thin_every=4, lifecycle_max_far_contracts=60)
        cmd = J.build_command(cfg, python="PY", script=Path("s.py"), status_path=Path("st.json"),
                              min_sessions=15, port=4002)
        flat = " ".join(cmd)
        assert "--lifecycle" in cmd and "--expiring-days 2" in flat and "--thin-every 4" in flat
        assert "--max-far-contracts 60" in flat and "--short-dte-duration 30 D" in flat
        assert cmd[cmd.index("--port") + 1] == "4002"

    def test_without_lifecycle_the_command_is_unchanged_apart_from_the_port(self):
        from ait.config.settings import DataCollectionConfig
        J = self._mod()
        cmd = J.build_command(DataCollectionConfig(), python="PY", script=Path("s.py"),
                              status_path=Path("st.json"), min_sessions=15)
        assert "--lifecycle" not in cmd and "--port" not in cmd
        assert cmd[cmd.index("--short-dte-duration") + 1] == "7 D"

    @pytest.mark.parametrize("listening, expect_port", [({7497}, "7497"), ({4002}, "4002"),
                                                        ({7497, 4002}, "7497")])
    def test_the_job_runs_when_either_tws_or_the_gateway_answers(self, tmp_path, listening,
                                                                 expect_port):
        from tests import test_w76_scheduled_collection as T
        clock, T.FakeProc.instances = T.FakeClock(), []
        (tmp_path / "data").mkdir()
        J = self._mod()
        out = J.run_collection(
            T._cfg(), root=tmp_path, data_dir=tmp_path / "data", logs_dir=tmp_path / "logs",
            min_sessions=15, ibkr_port=7497, base_client_id=1, is_trading_day=lambda d: True,
            gateway_listening=lambda p: p in listening, today=T.MON,
            popen=lambda cmd, **kw: T.FakeProc(cmd, script=T.Script(status=T._status()), **kw),
            python="PY", clock=clock, sleep=clock.sleep)
        assert out.status == "ok"
        cmd = T.FakeProc.instances[0].cmd
        assert cmd[cmd.index("--port") + 1] == expect_port

    def test_both_endpoints_closed_is_skipped_and_names_both(self, tmp_path):
        from tests import test_w76_scheduled_collection as T
        J = self._mod()
        out = J.run_collection(
            T._cfg(), root=tmp_path, data_dir=tmp_path / "data", logs_dir=tmp_path / "logs",
            min_sessions=15, ibkr_port=7497, base_client_id=1, is_trading_day=lambda d: True,
            gateway_listening=lambda p: False, today=T.MON)
        assert out.status == "skipped_gateway_down" and out.alert_worthy
        assert "7497" in out.detail and "4002" in out.detail

    def test_a_paper_job_never_falls_back_to_a_live_port(self, tmp_path):
        from tests import test_w76_scheduled_collection as T
        J = self._mod()
        out = J.run_collection(
            T._cfg(), root=tmp_path, data_dir=tmp_path / "data", logs_dir=tmp_path / "logs",
            min_sessions=15, ibkr_port=7497, base_client_id=1, is_trading_day=lambda d: True,
            gateway_listening=lambda p: p in (7496, 4001), today=T.MON)
        assert out.status == "skipped_gateway_down"

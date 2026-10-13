"""W76: 0-6 DTE friction: strike plan, collector (fake broker), band fit, banded model,
engine switch. No real broker is used anywhere; the collector takes the IB handle as an
argument and is exercised end to end against a fake.
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.friction import (
    BandedFriction,
    LegSpreadModel,
    banded_condor_leg_cost_usd,
    band_dte_range,
    friction_band,
)
from ait.data import short_dte_collector as sc
from ait.data.option_spread_sampler import (
    SHORT_DTE_MONEYNESS,
    delta_from_mid,
    parse_window,
    plan_short_dte_strikes,
)
from ait.utils.time import ET

TODAY = date(2026, 10, 9)           # a Friday: 0-DTE expiry today, weekend inside the 0-6 range
SPOT = 740.0
WINDOW = parse_window("10:00-15:30")


# --------------------------------------------------------------------------
# strike plan + 0-DTE delta
# --------------------------------------------------------------------------

class TestPlan:
    STRIKES = [float(k) for k in range(650, 781)]

    def test_listed_strikes_only_and_both_rights(self):
        for right in ("C", "P"):
            plan = plan_short_dte_strikes(SPOT, 0.17, 1, self.STRIKES, right)
            assert plan and all(k in self.STRIKES for k, _ in plan)
            assert len({k for k, _ in plan}) == len(plan)            # de-duplicated

    def test_moneyness_anchors_are_on_the_right_side_of_spot(self):
        calls = {k for k, _ in plan_short_dte_strikes(SPOT, 0.17, 0, self.STRIKES, "C")}
        puts = {k for k, _ in plan_short_dte_strikes(SPOT, 0.17, 0, self.STRIKES, "P")}
        for frac in SHORT_DTE_MONEYNESS:
            assert min(calls, key=lambda k: abs(k - SPOT * (1 + frac))) in calls
            assert min(puts, key=lambda k: abs(k - SPOT * (1 - frac))) in puts
        assert max(calls) > SPOT and min(puts) < SPOT

    def test_no_listed_strikes_or_bad_spot_yields_nothing(self):
        assert plan_short_dte_strikes(SPOT, 0.17, 1, [], "C") == []
        assert plan_short_dte_strikes(0.0, 0.17, 1, self.STRIKES, "C") == []

    def test_a_zero_dte_delta_can_be_inverted_only_with_fractional_time(self):
        assert delta_from_mid(0.45, 740.0, 741.0, 0, "C") == (None, None)       # the old form
        d, iv = delta_from_mid(0.45, 740.0, 741.0, 0, "C", t_years=5.5 / (24 * 365))
        assert d is not None and 0.2 < d < 0.5 and iv is not None


# --------------------------------------------------------------------------
# the banded model
# --------------------------------------------------------------------------

class TestBanded:
    def _bf(self, bands=None, floor=0.01):
        return BandedFriction(LegSpreadModel("const_usd", a=0.006), bands or {}, floor)

    def test_band_labels_round_trip(self):
        for dte, want in ((0, "0-2"), (2, "0-2"), (3, "3-6"), (6, "3-6"), (7, "7+"), (45, "7+")):
            assert friction_band(dte) == want
        assert band_dte_range("0-2") == (0, 2) and band_dte_range("3-6") == (3, 6)
        assert band_dte_range("7+")[0] == 7
        with pytest.raises(ValueError):
            band_dte_range("9-12")

    def test_unfitted_short_band_is_extrapolated_and_floored(self):
        v, ex = self._bf().half_spread_usd(0.30, 1)
        assert ex is True and v == pytest.approx(0.01)              # floor beats the 0.006 model

    def test_floor_never_lowers_a_larger_model_value(self):
        bf = BandedFriction(LegSpreadModel("const_usd", a=0.03), {}, 0.01)
        assert bf.half_spread_usd(0.30, 1) == (pytest.approx(0.03), True)

    def test_a_fitted_band_replaces_the_extrapolation(self):
        bf = self._bf({"0-2": LegSpreadModel("const_usd", a=0.02)})
        assert bf.half_spread_usd(0.30, 1) == (pytest.approx(0.02), False)
        assert bf.half_spread_usd(0.30, 4)[1] is True               # 3-6 still unfitted

    def test_the_long_dte_home_is_never_extrapolated_or_floored(self):
        v, ex = self._bf().half_spread_usd(0.30, 20)
        assert ex is False and v == pytest.approx(0.006)

    def test_condor_cost_sums_the_four_legs_and_flags_extrapolation(self):
        cost, ex = banded_condor_leg_cost_usd(self._bf(), (0.5, 0.4, 0.1, 0.1), 1, capture=1.0, stress=2.0)
        assert ex is True and cost == pytest.approx(2.0 * 4 * 0.01)
        cost, ex = banded_condor_leg_cost_usd(self._bf(), (0.5, 0.4, 0.1, 0.1), 30)
        assert ex is False and cost == pytest.approx(4 * 0.006)


# --------------------------------------------------------------------------
# a fake broker
# --------------------------------------------------------------------------

def _session_days(n_back: int = 6):
    d, out = TODAY, []
    while len(out) < 5 and (TODAY - d).days <= n_back:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


class FakeIB:
    """Just enough of ib_insync's async API; half-spread is a constant `half_usd`."""

    def __init__(self, *, half_usd=0.01, fail_on=None, unqualified=None, expirations=None):
        self.half_usd = half_usd
        self.fail_on = fail_on          # (right, strike) -> raise on historical request
        self.unqualified = unqualified  # (right, strike) -> qualify returns []
        self.calls: list[tuple] = []
        exps = expirations or [(TODAY + timedelta(days=i)) for i in range(0, 9)
                               if (TODAY + timedelta(days=i)).weekday() < 5]
        self.expirations = [e.strftime("%Y%m%d") for e in exps] + ["20261120"]
        self.strikes = [float(k) for k in range(680, 801)]

    async def qualifyContractsAsync(self, *contracts):
        out = []
        for c in contracts:
            key = (getattr(c, "right", None), getattr(c, "strike", None))
            if self.unqualified and key == self.unqualified:
                continue
            out.append(c)
        return out

    async def reqSecDefOptParamsAsync(self, symbol, exch, sec_type, con_id):
        return [SimpleNamespace(exchange="SMART", expirations=self.expirations, strikes=self.strikes)]

    def _bars_for_days(self, value_fn):
        out = []
        for d in _session_days():
            for h in (10, 11, 12, 13, 14, 15):
                ts = datetime.combine(d, time(h, 0), tzinfo=ET)
                out.append(SimpleNamespace(date=ts, close=float(value_fn(ts))))
        return out

    async def reqHistoricalDataAsync(self, contract, endDateTime, durationStr, barSizeSetting,
                                     whatToShow, useRTH, formatDate):
        self.calls.append((getattr(contract, "right", "STK"), getattr(contract, "strike", 0.0),
                           whatToShow))
        if whatToShow == "TRADES":
            return self._bars_for_days(lambda ts: SPOT)
        key = (contract.right, contract.strike)
        if self.fail_on and key == self.fail_on:
            raise RuntimeError("pacing violation")
        exp = datetime.strptime(contract.lastTradeDateOrContractMonth, "%Y%m%d").date()

        def val(ts):
            dte = max((exp - ts.date()).days, 0)
            mid = 0.05 + 3.0 * math.exp(-abs(contract.strike - SPOT) / 3.0) / (1 + dte)
            return mid - self.half_usd if whatToShow == "BID" else mid + self.half_usd

        # a contract only exists up to its expiry date
        return [b for b in self._bars_for_days(val) if b.date.date() <= exp]


@pytest.fixture
def store(tmp_path):
    from ait.data.historical import HistoricalDataStore
    return HistoricalDataStore(db_path=tmp_path / "h.db")


async def _run(ib, store, **kw):
    kw.setdefault("now", datetime.combine(TODAY, time(23, 59), tzinfo=ET))   # every fake bar has ended
    sleep = AsyncMock()
    quiet = []
    res = await sc.collect_short_dte(ib, "QQQ", store, WINDOW, today=TODAY, sleep=sleep,
                                     out=quiet.append, **kw)
    return res, sleep, quiet


# --------------------------------------------------------------------------
# the collector
# --------------------------------------------------------------------------

class TestInProgressBars:
    """W76b: a bar whose hour is still running holds a partial quote and is never stored."""

    @staticmethod
    async def _stored_times(store, now):
        ib = FakeIB()
        res, _sleep, _q = await _run(ib, store, now=now)
        return res, store.load_spread_bar_samples("QQQ")

    @pytest.mark.asyncio
    async def test_a_bar_that_has_not_ended_is_skipped(self, store):
        mid_hour = datetime.combine(TODAY, time(12, 45), tzinfo=ET)       # the 12:00 bar is running
        res, df = await self._stored_times(store, mid_hour)
        assert res.skipped_partial > 0 and res.stored > 0
        got = {t for t in df["bar_time"]}
        today_bars = {t for t in got if t.startswith(TODAY.isoformat())}
        assert any("T11:00" in t for t in today_bars)                     # ended at 12:00 <= 12:45
        assert not any("T12:00" in t or "T13:00" in t or "T14:00" in t or "T15:00" in t
                       for t in today_bars)
        # earlier sessions are complete: stored in full
        assert any(not t.startswith(TODAY.isoformat()) for t in got)

    @pytest.mark.asyncio
    async def test_the_last_bar_ends_at_the_bell_not_an_hour_later(self, store):
        # 15:00 bar: its end is min(16:00, close) = 16:00
        before_bell = datetime.combine(TODAY, time(15, 59), tzinfo=ET)
        res, df = await self._stored_times(store, before_bell)
        assert not any(t.startswith(TODAY.isoformat()) and "T15:00" in t for t in df["bar_time"])

    @pytest.mark.asyncio
    async def test_after_the_bell_the_last_bar_is_stored(self, store):
        after = datetime.combine(TODAY, time(16, 1), tzinfo=ET)
        res, df = await self._stored_times(store, after)
        assert res.skipped_partial == 0
        assert any(t.startswith(TODAY.isoformat()) and "T15:00" in t for t in df["bar_time"])

    @pytest.mark.asyncio
    async def test_a_later_run_completes_what_an_earlier_one_skipped(self, store):
        ib = FakeIB()
        early, _s, _q = await _run(ib, store, now=datetime.combine(TODAY, time(12, 45), tzinfo=ET))
        late, _s, _q = await _run(ib, store, now=datetime.combine(TODAY, time(23, 59), tzinfo=ET))
        assert early.skipped_partial > 0 and late.skipped_partial == 0
        assert late.stored > 0          # the bars that were skipped are now stored (idempotent insert)


class TestCollector:
    @pytest.mark.asyncio
    async def test_plan_covers_expiries_zero_to_six_days_out_and_only_those(self):
        ib = FakeIB()
        plan = sc.plan_contracts(SPOT, 0.17, TODAY, ib.expirations, ib.strikes, 6)
        days = sorted({(e - TODAY).days for e, *_ in plan})
        assert days[0] == 0 and days[-1] <= 6
        assert (date(2026, 11, 20) - TODAY).days not in days
        assert all(abs(k / SPOT - 1) <= 0.12 for _, _, k, _ in plan)
        assert {r for _, r, *_ in plan} == {"C", "P"}

    def test_strikes_outside_twelve_percent_of_spot_are_never_planned(self):
        far = [float(k) for k in (300, 500, 640, 650, 655, 740, 825, 830, 900, 1200)]
        plan = sc.plan_contracts(SPOT, 0.17, TODAY, [TODAY.strftime("%Y%m%d")], far, 6)
        got = {k for _, _, k, _ in plan}
        assert got and all(abs(k / SPOT - 1.0) <= 0.12 for k in got)
        assert not got & {300.0, 500.0, 900.0, 1200.0}

    def test_a_sparse_chain_far_from_spot_yields_no_plan_not_far_strikes(self):
        """`nearest_listed` would happily pick a strike 20% away if that is all there is;
        the +-12% window must stop it from being requested."""
        plan = sc.plan_contracts(SPOT, 0.17, TODAY, [TODAY.strftime("%Y%m%d")], [600.0, 900.0], 6)
        assert plan == []

    def test_expiries_beyond_max_dte_and_in_the_past_are_not_planned(self):
        exps = [(TODAY + timedelta(days=i)).strftime("%Y%m%d") for i in (-2, -1, 0, 6, 7, 30)]
        plan = sc.plan_contracts(SPOT, 0.17, TODAY, exps, [740.0, 741.0, 742.0], 6)
        assert {(e - TODAY).days for e, *_ in plan} == {0, 6}

    @pytest.mark.asyncio
    async def test_dry_run_requests_no_bars_and_stores_nothing(self, store):
        ib = FakeIB()
        res, sleep, _ = await _run(ib, store, dry_run=True)
        assert res.dry_run and res.planned > 0 and res.stored == 0 and res.requested == 0
        assert [c for c in ib.calls if c[2] in ("BID", "ASK")] == []
        assert store.load_spread_bar_samples("QQQ").empty
        sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_run_stores_samples_for_every_short_dte_including_zero(self, store):
        res, sleep, _ = await _run(FakeIB(), store)
        assert res.stored > 0 and res.errors == 0
        assert set(res.by_dte) >= {0, 1, 2, 3, 4, 5, 6}, res.by_dte
        df = store.load_spread_bar_samples("QQQ")
        # longer-dated bars of the same contracts are STORED (they enrich the all-DTE fit)
        # but only DTE 0..max_dte is reported as short-DTE coverage
        assert (df["dte"] > 6).any()
        assert max(res.by_dte) <= 6
        assert (df["half_spread_usd"].round(6) == 0.01).all()      # bid/ask built that way
        zero = df[df["dte"] == 0]
        hour = zero["bar_time"].str[11:13].astype(int)
        # every bar that ends BEFORE the bell has time left, so its delta is defined; the last
        # bar of the expiry day (15:00-16:00) ends at the bell: zero time, no delta, and it is
        # dropped from the fit by the relevance mask unless it is a wing.
        assert len(zero) > 0 and zero.loc[hour <= 14, "delta"].notna().all(), \
            "0-DTE rows lost their delta"
        assert zero.loc[hour == 15, "delta"].isna().all()
        sleep.assert_called()                                       # paced between requests

    @pytest.mark.asyncio
    async def test_pacing_pause_is_applied_after_every_bar_request(self, store):
        ib = FakeIB()
        res, sleep, _ = await _run(ib, store, pause_secs=7.5)
        pauses = [c.args[0] for c in sleep.call_args_list]
        assert pauses and set(pauses) == {7.5}
        assert len(pauses) == res.requested

    @pytest.mark.asyncio
    async def test_rerun_is_idempotent(self, store):
        await _run(FakeIB(), store)
        n1 = len(store.load_spread_bar_samples("QQQ"))
        await _run(FakeIB(), store)
        assert len(store.load_spread_bar_samples("QQQ")) == n1

    @pytest.mark.asyncio
    async def test_a_failing_contract_backs_off_and_the_run_continues(self, store):
        plan = sc.plan_contracts(SPOT, 0.17, TODAY, FakeIB().expirations, FakeIB().strikes, 6)
        _, right, strike, _ = plan[len(plan) // 2]
        ib = FakeIB(fail_on=(right, strike))
        res, sleep, quiet = await _run(ib, store)
        assert res.errors >= 1 and res.stored > 0
        assert any(c.args[0] == 60.0 for c in sleep.call_args_list)   # the 60 s back-off
        assert any("pacing violation" in m for m in quiet)

    @pytest.mark.asyncio
    async def test_an_unqualifiable_contract_is_skipped_not_fatal(self, store):
        plan = sc.plan_contracts(SPOT, 0.17, TODAY, FakeIB().expirations, FakeIB().strikes, 6)
        _, right, strike, _ = plan[3]
        res, _, _ = await _run(FakeIB(unqualified=(right, strike)), store)
        assert res.unqualified >= 1 and res.stored > 0

    @pytest.mark.asyncio
    async def test_no_underlying_bars_is_reported_not_raised(self, store):
        class Empty(FakeIB):
            async def reqHistoricalDataAsync(self, contract, **kw):
                return []
        res, _, quiet = await _run(Empty(), store)
        assert res.stored == 0 and any("no underlying" in m for m in quiet)


# --------------------------------------------------------------------------
# coverage + band fit
# --------------------------------------------------------------------------

class TestFit:
    @pytest.mark.asyncio
    async def test_coverage_counts_samples_and_sessions_per_dte(self, store):
        await _run(FakeIB(), store)
        cov = sc.coverage(store.load_spread_bar_samples("QQQ"), 6)
        assert set(cov) == set(range(7))
        assert cov[0]["sessions"] >= 1 and cov[0]["samples"] > 0
        assert all(c["sessions"] <= 5 for c in cov.values())        # 5 sessions generated

    @pytest.mark.asyncio
    async def test_fit_recovers_the_constant_half_spread_and_reports_progress(self, store):
        await _run(FakeIB(half_usd=0.012), store)
        out = []
        saved = sc.fit_short_dte_bands("QQQ", store, min_sessions=15, out=out.append)
        assert set(saved) == {"0-2", "3-6"}
        for band, row in saved.items():
            assert row["a"] == pytest.approx(0.012, abs=1e-6) or row["b"] > 0   # const or pct fit
        text = "\n".join(out)
        assert "collecting:" in text and "more needed" in text      # below 15 sessions
        rows = store.load_leg_spread_band_models("QQQ")
        assert set(rows) == {"0-2", "3-6"} and all(r["n_sessions"] < 15 for r in rows.values())

    @pytest.mark.asyncio
    async def test_a_band_with_enough_sessions_is_reported_active(self, store):
        await _run(FakeIB(), store)
        out = []
        sc.fit_short_dte_bands("QQQ", store, min_sessions=2, out=out.append)
        assert any("ACTIVE" in m for m in out)

    def test_nothing_stored_is_a_no_op(self, store):
        out = []
        assert sc.fit_short_dte_bands("QQQ", store, min_sessions=15, out=out.append) == {}
        assert any("no bar samples" in m for m in out)

    def test_band_sessions_counts_distinct_relevant_days(self):
        rows = []
        for day in ("2026-10-05", "2026-10-06", "2026-10-06"):
            rows.append({"bar_time": f"{day}T10:00:00-04:00", "dte": 1, "mid": 0.3, "delta": 0.2,
                         "half_spread_usd": 0.01, "role": "short"})
        rows.append({"bar_time": "2026-10-07T10:00:00-04:00", "dte": 1, "mid": 0.003, "delta": 0.2,
                     "half_spread_usd": 0.01, "role": "short"})     # below the 0.01 mid floor
        assert sc.band_sessions(pd.DataFrame(rows), "0-2") == 2


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

class TestStorage:
    def test_band_table_roundtrip_and_replace(self, store):
        row = {"form": "const_usd", "a": 0.011, "b": 0.0, "n": 120, "n_sessions": 7,
               "cv_mae": 0.002, "cv_mae_by_form": {"const_usd": 0.002}, "source": "t",
               "calibrated_on": "2026-10-12"}
        store.save_leg_spread_band_model("QQQ", "0-2", row)
        got = store.load_leg_spread_band_models("QQQ")
        assert got["0-2"]["a"] == pytest.approx(0.011) and got["0-2"]["n_sessions"] == 7
        store.save_leg_spread_band_model("QQQ", "0-2", {**row, "a": 0.02, "n_sessions": 9})
        again = store.load_leg_spread_band_models("QQQ")
        assert len(again) == 1 and again["0-2"]["a"] == pytest.approx(0.02)
        assert store.load_leg_spread_band_models("SPY") == {}

    def test_the_existing_all_dte_table_is_untouched(self, store):
        assert store.load_leg_spread_model("QQQ") is None
        store.save_leg_spread_band_model("QQQ", "0-2", {
            "form": "pct", "a": 0.0, "b": 0.01, "n": 10, "n_sessions": 3, "cv_mae": None,
            "source": None, "calibrated_on": "2026-10-12"})
        assert store.load_leg_spread_model("QQQ") is None


# --------------------------------------------------------------------------
# the engine switch
# --------------------------------------------------------------------------

def _engine(db, **kw):
    from ait.backtesting.engine import Backtester
    idx = pd.bdate_range("2025-02-03", periods=40)
    c = SPOT + np.zeros(len(idx))
    df = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1e6,
                       "implied_vol": 0.17}, index=idx)
    base = dict(data=df, strategies=["iron_condor"], initial_capital=1_000_000, features_cache=None,
                macro_event_gate=False, allow_live_model_fallback=False, ic_min_credit=0.0,
                ic_min_credit_width=0.0, iv_floor=0.01, friction_model="per_leg_usd",
                db_path=db, symbol="QQQ", entry_dte=3)      # HistoricalDataStore wants a Path
    base.update(kw)
    return Backtester(**base)


@pytest.fixture
def db_with_models(tmp_path):
    from ait.data.historical import HistoricalDataStore
    db = tmp_path / "h.db"
    st = HistoricalDataStore(db_path=db)
    st.save_leg_spread_model("QQQ", {"form": "const_usd", "a": 0.006, "b": 0.0, "n": 500,
                                     "n_sessions": 18, "cv_mae": 0.001, "source": "t",
                                     "calibrated_on": "2026-10-09"})
    return db, st


def _band_row(a, sessions):
    return {"form": "const_usd", "a": a, "b": 0.0, "n": 200, "n_sessions": sessions,
            "cv_mae": 0.001, "cv_mae_by_form": {}, "source": "t", "calibrated_on": "2026-10-12"}


MIDS = (0.5, 0.4, 0.1, 0.1)


class TestEngineSwitch:
    def test_default_is_off_and_charges_the_single_model(self, db_with_models):
        db, _ = db_with_models
        bt = _engine(db)
        assert bt._friction_short_dte is False and bt._friction_bands is None
        assert bt._leg_friction_usd(1.0, MIDS, dte=1) == pytest.approx(4 * 0.006)

    def test_unfitted_short_band_is_floored_and_counted_when_on(self, db_with_models):
        db, _ = db_with_models
        bt = _engine(db, friction_short_dte=True)
        assert bt._leg_friction_usd(1.0, MIDS, dte=1) == pytest.approx(4 * 0.01)
        assert bt._friction_stats == {"fitted": 0, "default": 0, "extrapolated": 1}
        assert bt._leg_friction_usd(1.0, MIDS, dte=20) == pytest.approx(4 * 0.006)  # 7+: untouched
        assert bt._friction_stats["fitted"] == 1

    def test_a_band_with_enough_sessions_is_used(self, db_with_models):
        db, st = db_with_models
        st.save_leg_spread_band_model("QQQ", "0-2", _band_row(0.02, 20))
        bt = _engine(db, friction_short_dte=True)
        assert "0-2" in bt._friction_bands.bands
        assert bt._leg_friction_usd(1.0, MIDS, dte=1) == pytest.approx(4 * 0.02)
        assert bt._friction_stats["fitted"] == 1 and bt._friction_stats["extrapolated"] == 0

    def test_a_band_below_the_session_minimum_is_ignored(self, db_with_models):
        db, st = db_with_models
        st.save_leg_spread_band_model("QQQ", "0-2", _band_row(0.02, 5))
        bt = _engine(db, friction_short_dte=True)
        assert "0-2" not in bt._friction_bands.bands
        assert bt._leg_friction_usd(1.0, MIDS, dte=1) == pytest.approx(4 * 0.01)    # floor, not 0.02
        bt2 = _engine(db, friction_short_dte=True, friction_min_sessions_per_band=5)
        assert "0-2" in bt2._friction_bands.bands

    def test_the_floor_is_configurable_and_stress_multiplies_it(self, db_with_models):
        db, _ = db_with_models
        bt = _engine(db, friction_short_dte=True, friction_extrapolated_floor_usd=0.02,
                     friction_stress_multiplier=1.5)
        assert bt._leg_friction_usd(1.0, MIDS, dte=0) == pytest.approx(1.5 * 4 * 0.02)

    def test_entry_credit_is_charged_the_band_friction_at_the_positions_dte(self, db_with_models):
        db, _ = db_with_models
        off = _engine(db, entry_dte=1)
        on = _engine(db, entry_dte=1, friction_short_dte=True)
        a = off._build_credit_position("iron_condor", SPOT, 0.17, 1 / 365.0, 0.05, 1,
                                       date(2025, 3, 3), 1_000_000.0)
        b = on._build_credit_position("iron_condor", SPOT, 0.17, 1 / 365.0, 0.05, 1,
                                      date(2025, 3, 3), 1_000_000.0)
        assert a is not None and b is not None
        assert b["entry_friction_usd"] > a["entry_friction_usd"]     # 0.04 floor vs 4 x 0.006

    def test_exit_friction_uses_the_remaining_dte_not_the_entry_dte(self, db_with_models):
        db, _ = db_with_models
        bt = _engine(db, friction_short_dte=True, engine_model="realistic")
        pos = {"strategy": "iron_condor", "short_call_strike": 745.0, "short_put_strike": 735.0,
               "long_call_strike": 750.0, "long_put_strike": 730.0, "entry_iv": 0.17,
               "entry_date": "2025-03-03", "expiry_date": "2025-03-24"}      # 21 DTE at entry
        far = bt._apply_exit_friction(pos, 0.0, SPOT, 1, None, 0.0, "credit",
                                      as_of=datetime(2025, 3, 4, 12, 0, tzinfo=ET))
        near = bt._apply_exit_friction(pos, 0.0, SPOT, 20, None, 0.0, "credit",
                                       as_of=datetime(2025, 3, 23, 12, 0, tzinfo=ET))
        assert near > far          # 1 day left: floored tick on every leg; 20 days: all-DTE model
        assert bt._friction_stats["extrapolated"] == 1

    def test_partial_engines_default_to_off(self):
        from ait.backtesting.engine import Backtester
        bare = Backtester.__new__(Backtester)
        assert bare._friction_short_dte is False and bare._friction_bands is None


class TestConfig:
    def test_fields_defaults_and_bounds(self):
        from ait.config.settings import BacktestConfig
        c = BacktestConfig()
        assert c.friction_short_dte is False
        assert (c.friction_min_sessions_per_band, c.friction_extrapolated_floor_usd) == (15, 0.01)
        for bad in (dict(friction_min_sessions_per_band=0), dict(friction_extrapolated_floor_usd=0.001),
                    dict(friction_extrapolated_floor_usd=1.0)):
            with pytest.raises(Exception):
                BacktestConfig(**bad)


class TestSensitivityFlags:
    def _mod(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "friction_sensitivity_cli", Path(__file__).resolve().parents[1] / "scripts" / "friction_sensitivity.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _args(self, **over):
        base = dict(capital=100_000.0, budget_pct=0.05, any_direction=True, iv_rank_gate=False,
                    db_path="data/historical.db", entry_dte=None, engine_model=None,
                    pricing_source=None, friction_short_dte=None, expiry_guard_dte=None,
                    flatten_minutes=None)
        base.update(over)
        return SimpleNamespace(**base)

    def test_unset_flags_leave_the_w66_kwargs_untouched(self):
        kw = self._mod()._engine_kwargs("QQQ", None, None, None, self._args())
        for k in ("entry_dte", "engine_model", "pricing_source", "friction_short_dte",
                  "expiry_guard_dte", "flatten_minutes_before_close"):
            assert k not in kw

    def test_set_flags_reach_the_engine_under_the_engines_own_names(self):
        kw = self._mod()._engine_kwargs("QQQ", None, None, None, self._args(
            entry_dte=1, engine_model="realistic", pricing_source="surface",
            friction_short_dte=True, expiry_guard_dte=5, flatten_minutes=60))
        assert kw["entry_dte"] == 1 and kw["engine_model"] == "realistic"
        assert kw["pricing_source"] == "surface" and kw["friction_short_dte"] is True
        assert kw["expiry_guard_dte"] == 5 and kw["flatten_minutes_before_close"] == 60

    def test_a_flag_set_to_false_is_still_passed(self):
        kw = self._mod()._engine_kwargs("QQQ", None, None, None, self._args(friction_short_dte=False))
        assert kw["friction_short_dte"] is False


class TestDbPathTypes:
    """The W74/W75 reference studies passed db_path as a str; HistoricalDataStore needs a Path
    (it does `.parent`), the loaders failed with only a WARNING, and the engine silently
    priced with the LEGACY % friction model while the report said 'fitted per-leg'."""

    def test_a_str_db_path_loads_the_per_leg_model(self, db_with_models):
        db, _ = db_with_models
        bt = _engine(str(db))                        # str, not Path
        assert bt._leg_spread_model is not None
        assert bt._leg_friction_usd(1.0, MIDS) == pytest.approx(4 * 0.006)

    def test_a_str_and_a_path_give_the_same_engine_state(self, db_with_models):
        db, _ = db_with_models
        a, b = _engine(str(db), friction_short_dte=True), _engine(db, friction_short_dte=True)
        assert a._leg_spread_model == b._leg_spread_model
        assert a._friction_bands.floor_usd == b._friction_bands.floor_usd

    def test_the_engine_stores_a_path(self, db_with_models):
        from pathlib import Path
        db, _ = db_with_models
        assert isinstance(_engine(str(db))._db_path, Path)


class TestCliPort:
    """The collector's CLI used to hard-code port 4002; on a TWS-paper box (7497) every run was
    refused. It must follow IBKR_PORT like the bot does, and an explicit --port must still win."""

    def _mod(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "calibrate_cli_port", Path(__file__).resolve().parents[1] / "scripts"
            / "calibrate_option_spreads.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_default_port_follows_ibkr_port(self, monkeypatch):
        monkeypatch.setenv("IBKR_PORT", "7497")
        assert self._mod()._parse_args(["--source", "ib-bars", "--short-dte"]).port == 7497
        monkeypatch.setenv("IBKR_PORT", "4002")
        assert self._mod()._parse_args(["--source", "ib-bars", "--short-dte"]).port == 4002

    def test_an_explicit_port_wins(self, monkeypatch):
        monkeypatch.setenv("IBKR_PORT", "7497")
        args = self._mod()._parse_args(["--source", "ib-bars", "--short-dte", "--port", "4001"])
        assert args.port == 4001

    def test_a_config_failure_falls_back_to_4002_not_a_crash(self, monkeypatch):
        mod = self._mod()
        import ait.config.settings as st
        monkeypatch.setattr(st, "IBKREnvConfig", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert mod._default_ibkr_port() == 4002

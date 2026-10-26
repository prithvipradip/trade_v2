"""W101-D: a worthless long wing nobody bids for is left out of the closing combo.

Paper probe 2026-10-26: a closing BAG that holds a leg to SELL with no bid (IB
quotes bid -1) has no market -- IB accepts it at any limit and it never fills --
and the same combo without that wing fills in seconds. The QQQ 772C wing was
quoted -1/0.01 on all 176 closes of 2026-10-22/23.

Every test drives the real ``_close_multi_leg`` on the W101 rig (real settings,
stubbed broker), and the reconciler tests drive the real ``reconcile()`` on a
real StateManager. The leftover wing is recorded by the orchestrator and read
back by the reconciler, so the key format is exercised end to end.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager
from ait.execution import reconciler as rec_mod
from ait.execution.reconciler import PositionReconciler
from ait.utils.time import now_et, today_et
from tests.fakes import FakeIB, FakeIBKRClient, _option_position
from tests.test_w33_exit_pricing_corrective import NAN
from tests.test_w101_exit_floor import (
    INCIDENT, LEGS as _FIXED_LEGS, TOUCH, WIDTH, _go, _incident, _instant_polls,  # noqa: F401
    _rig,
)

ROOT = Path(__file__).resolve().parent.parent

_EXP = (today_et() + timedelta(days=4)).strftime("%Y%m%d")       # a live expiry
LEGS = [{**l, "expiry": _EXP} for l in _FIXED_LEGS]
# conIds by leg order: 1 = 704P (-> SELL), 2 = 707P (-> BUY), 3 = 769C (-> BUY), 4 = 772C (-> SELL)
WING_KEY = f"SPY:772.0:C:{_EXP[:4]}-{_EXP[4:6]}-{_EXP[6:]}"


def _placed_legs(ibkr) -> list[tuple[int, str]]:
    combo = ibkr.place_order.call_args.args[0]
    return [(l.conId, l.action) for l in combo.comboLegs]


def _with(con_id, quote):
    return {**INCIDENT, con_id: quote}


def _close(quotes=None, *, legs=LEGS, **kw):
    o, ibkr, trade = _incident(leg_quotes=dict(quotes or INCIDENT), **kw)
    px = _go(o, ibkr, trade, legs=legs)
    return o, ibkr, trade, px


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------
class TestTheWingIsLeftOut:
    def test_the_incident_closes_three_legs_with_both_shorts_in(self):
        o, ibkr, trade, px = _close()
        assert px is not None
        assert sorted(_placed_legs(ibkr)) == [(1, "SELL"), (2, "BUY"), (3, "BUY")]

    def test_only_the_legs_in_the_combo_are_priced(self):
        o, ibkr, trade, _ = _close()
        assert len(o._last_leg_poll) == 3

    def test_the_limit_is_what_it_was_with_the_wing_in(self):
        """The wing priced at 0 either way, so the old expectation still holds."""
        assert _close()[3] == pytest.approx(2.48)

    def test_the_wing_is_logged(self, monkeypatch):
        seen = []
        import ait.bot.orchestrator as om
        monkeypatch.setattr(om.log, "warning",
                            lambda ev, **kw: seen.append((ev, kw)))
        _close()
        events = [kw for ev, kw in seen if ev == "exit_unbid_wing_left_out"]
        assert len(events) == 1 and events[0]["con_id"] == 4 and events[0]["ask"] == 0.01

    def test_the_wing_poll_is_not_a_placement(self):
        o, ibkr, trade, _ = _close()
        assert getattr(o, "_exit_unfilled_attempts", {}).get("T-1", {}).get("n", 0) in (0, 1)
        before = dict(getattr(o, "_exit_unfilled_attempts", {}))
        asyncio.run(o._unbid_wings_to_leave_out(trade, [
            {"conId": 1, "action": "SELL", "_contract": SimpleNamespace(conId=1)}]))
        assert dict(getattr(o, "_exit_unfilled_attempts", {})) == before


class TestTheWingStaysIn:
    def test_a_wing_with_a_bid_stays(self):
        q = _with(4, (0.01, 0.02))
        assert len(_placed_legs(_close(q)[1])) == 4

    def test_an_ask_above_the_ceiling_stays(self):
        q = _with(4, (-1.0, 0.06))
        assert len(_placed_legs(_close(q)[1])) == 4

    def test_an_ask_exactly_at_the_ceiling_goes(self):
        q = _with(4, (-1.0, 0.05))
        assert len(_placed_legs(_close(q)[1])) == 3

    def test_a_zero_bid_counts_as_unbid(self):
        q = _with(4, (0.0, 0.02))
        assert len(_placed_legs(_close(q)[1])) == 3

    def test_an_unknown_quote_is_not_an_unbid_wing(self):
        for unknown in ((NAN, NAN), (NAN, 0.01), (-1.0, NAN), None):
            q = _with(4, unknown)
            o, ibkr, trade, _ = _close(q)
            if ibkr.place_order.called:
                assert (4, "SELL") in _placed_legs(ibkr), unknown

    def test_an_unbid_short_is_never_left_out(self):
        """769C (a short, bought back) quoted -1/.01 is in the incident itself."""
        assert (3, "BUY") in _placed_legs(_close()[1])

    def test_the_setting_off_puts_every_leg_in(self):
        o, ibkr, trade = _incident(leg_quotes=dict(INCIDENT))
        o._settings.exit.exit_leave_out_unbid_wings = False
        _go(o, ibkr, trade, legs=LEGS)
        assert len(_placed_legs(ibkr)) == 4

    def test_a_settings_stub_without_the_key_changes_nothing(self):
        o, ibkr, trade = _incident(leg_quotes=dict(INCIDENT))
        o._settings = SimpleNamespace(exit=SimpleNamespace(
            **{k: v for k, v in vars(o._settings.exit).items()
               if not k.startswith("exit_leave_out") and not k.startswith("exit_unbid")}))
        _go(o, ibkr, trade, legs=LEGS)
        assert len(_placed_legs(ibkr)) == 4

    def test_a_two_leg_vertical_keeps_both_legs(self):
        vert = [LEGS[0], LEGS[1]]                                  # 704P long, 707P short
        q = {1: (-1.0, 0.01), 2: (2.82, 2.87)}
        o, ibkr, trade = _incident(leg_quotes=q)
        _go(o, ibkr, trade, legs=vert)
        assert len(_placed_legs(ibkr)) == 2

    def test_a_debit_structure_is_untouched(self):
        o, ibkr, trade = _incident(leg_quotes=dict(INCIDENT))
        trade.strategy = "long_straddle"
        sells = [{"conId": 4, "action": "SELL",
                  "_contract": SimpleNamespace(conId=4)}] * 2
        assert asyncio.run(o._unbid_wings_to_leave_out(trade, sells)) == set()


class TestNeverWithoutABuyBackLeg:
    def test_nothing_is_left_out_when_no_buy_back_would_remain(self):
        o, ibkr, trade = _incident(leg_quotes={1: (-1.0, 0.01), 2: (0.5, 0.6), 3: (0.5, 0.6)})
        legs = [{"conId": i, "action": "SELL", "_contract": SimpleNamespace(conId=i)}
                for i in (1, 2, 3)]
        assert asyncio.run(o._unbid_wings_to_leave_out(trade, legs)) == set()


class TestBothWings:
    def test_two_worthless_wings_leave_two_legs_and_both_shorts(self):
        q = _with(1, (-1.0, 0.01))
        o, ibkr, trade, _ = _close(q)
        assert sorted(_placed_legs(ibkr)) == [(2, "BUY"), (3, "BUY")]


class TestWidthAndFloorKeepTheFullStructure:
    def test_the_floor_and_the_width_cap_still_bind(self):
        """Near expiry on delayed data the incident closes at the wing width (3.00). With
        the reduced three-leg list the width would be None and this would not hold."""
        o, ibkr, trade = _incident(leg_quotes=dict(INCIDENT))
        px = _go(o, ibkr, trade, spot=704.4, dte=0, legs=LEGS)
        assert len(_placed_legs(ibkr)) == 3
        assert px == pytest.approx(3.00)

    def test_the_point_floor_still_applies_before_the_window(self):
        o, ibkr, trade = _incident(leg_quotes=dict(INCIDENT))
        assert _go(o, ibkr, trade, spot=704.4, dte=3, legs=LEGS) == pytest.approx(2.70)
        assert len(_placed_legs(ibkr)) == 3


# --------------------------------------------------------------------------
# what is recorded, and when
# --------------------------------------------------------------------------
class TestThePlacementRecord:
    def test_the_count_and_the_residual_are_written(self):
        o, ibkr, trade, _ = _close()
        assert o._state.kv["exit_excluded_legs_T-1"] == "1"
        rec = json.loads(o._state.kv[f"residual_leg_{WING_KEY}"])
        assert rec == {"trade_id": "T-1", "qty": 1.0, "expiry": _EXP}

    def test_a_later_four_leg_attempt_overwrites_the_count(self):
        o, ibkr, trade, _ = _close()
        ibkr.place_order.reset_mock()
        o._ibkr.ib.ticker = lambda c: SimpleNamespace(
            **{"bid": 0.01, "ask": 0.02}) if getattr(c, "conId", None) == 4 else \
            SimpleNamespace(bid=1.0, ask=1.1)
        asyncio.run(o._close_multi_leg(trade, LEGS, reason=TOUCH))
        assert len(_placed_legs(ibkr)) == 4
        assert o._state.kv["exit_excluded_legs_T-1"] == "0"

    def test_the_record_exists_before_the_order_goes(self):
        o, ibkr, trade = _incident(leg_quotes=dict(INCIDENT))
        seen = {}

        async def _place(combo, order):
            seen.update(o._state.kv)
            return SimpleNamespace(order=SimpleNamespace(orderId=42))

        ibkr.place_order.side_effect = _place
        _go(o, ibkr, trade, legs=LEGS)
        assert "exit_excluded_legs_T-1" in seen and f"residual_leg_{WING_KEY}" in seen

    def test_nothing_is_written_when_the_close_does_not_start(self):
        o, ibkr, trade = _incident(leg_quotes=dict(INCIDENT))
        o._state.update_trade_status = lambda *a, **k: False       # the CAS to CLOSING is lost
        _go(o, ibkr, trade, legs=LEGS)
        assert not ibkr.place_order.called
        assert not any(k.startswith(("residual_leg_", "exit_excluded_legs_"))
                       for k in o._state.kv)


# --------------------------------------------------------------------------
# the reconciler
# --------------------------------------------------------------------------
def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _halt_text(tmp_path) -> str | None:
    p = tmp_path / "data" / "HALT_UNTRACKED"
    return p.read_text() if p.exists() else None


def _backdate_last_seen(state, key, minutes_ago=10):
    state.set_state(f"manual_position_last_seen_{key}",
                    (now_et() - timedelta(minutes=minutes_ago)).isoformat())


@pytest.fixture
def cwd_tmp(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)


def _wing_at_broker(qty=1.0, expiry=_EXP):
    ib = FakeIB()
    ib.positions_list = [_option_position("SPY", 772.0, "C", expiry, qty)]
    return ib


def _reconcile(st, ib):
    return asyncio.run(PositionReconciler(FakeIBKRClient(ib), st).reconcile())


@pytest.mark.usefixtures("cwd_tmp")
class TestTheReconcilerExpectsTheWing:
    def _record(self, st, qty=1.0, expiry=_EXP):
        rec_mod.note_residual_leg(st, WING_KEY, "T-1", qty, expiry)

    def test_negative_control_without_a_record_the_wing_freezes_entries(self, tmp_path):
        st = _state(tmp_path)
        _backdate_last_seen(st, WING_KEY)
        _reconcile(st, _wing_at_broker())
        assert WING_KEY in (_halt_text(tmp_path) or "")

    def test_with_a_record_nothing_freezes_and_nothing_is_reported(self, tmp_path):
        st = _state(tmp_path)
        self._record(st)
        _backdate_last_seen(st, WING_KEY)
        result = _reconcile(st, _wing_at_broker())
        assert _halt_text(tmp_path) is None
        assert not any(WING_KEY in d for d in result.discrepancies)
        # no sighting was recorded: the debounce state is exactly as it was
        seen = st.get_state(f"manual_position_last_seen_{WING_KEY}")
        assert (now_et() - datetime.fromisoformat(seen)).total_seconds() > 9 * 60

    def test_end_to_end_what_the_orchestrator_wrote_is_what_the_reconciler_reads(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(ROOT)                    # the rig loads config.yaml
        o, ibkr, trade, _ = _close()
        monkeypatch.chdir(tmp_path)
        st = _state(tmp_path)
        for k, v in o._state.kv.items():
            st.set_state(k, v)
        _backdate_last_seen(st, WING_KEY)
        _reconcile(st, _wing_at_broker())
        assert _halt_text(tmp_path) is None

    def test_another_quantity_still_freezes(self, tmp_path):
        st = _state(tmp_path)
        self._record(st, qty=1.0)
        _backdate_last_seen(st, WING_KEY)
        _reconcile(st, _wing_at_broker(qty=2.0))
        assert WING_KEY in (_halt_text(tmp_path) or "")

    def test_a_short_at_the_same_contract_still_freezes(self, tmp_path):
        st = _state(tmp_path)
        self._record(st, qty=1.0)
        _backdate_last_seen(st, WING_KEY)
        _reconcile(st, _wing_at_broker(qty=-1.0))
        assert WING_KEY in (_halt_text(tmp_path) or "")

    def test_a_record_long_past_its_expiry_no_longer_protects(self, tmp_path):
        old = (today_et() - timedelta(days=3)).strftime("%Y%m%d")
        key = f"SPY:772.0:C:{old[:4]}-{old[4:6]}-{old[6:]}"
        st = _state(tmp_path)
        rec_mod.note_residual_leg(st, key, "T-1", 1.0, old)
        _backdate_last_seen(st, key)
        _reconcile(st, _wing_at_broker(expiry=old))
        assert key in (_halt_text(tmp_path) or "")
        assert st.state_keys_like("residual_leg_%") == []     # purged although still shown

    def test_the_day_after_expiry_is_still_covered(self, tmp_path):
        """IB drops an expired contract overnight; a 09:00 run may still see it."""
        d = (today_et() - timedelta(days=1)).strftime("%Y%m%d")
        key = f"SPY:772.0:C:{d[:4]}-{d[4:6]}-{d[6:]}"
        st = _state(tmp_path)
        rec_mod.note_residual_leg(st, key, "T-1", 1.0, d)
        _backdate_last_seen(st, key)
        _reconcile(st, _wing_at_broker(expiry=d))
        assert _halt_text(tmp_path) is None

    def test_the_record_is_forgotten_once_the_leg_is_gone(self, tmp_path):
        st = _state(tmp_path)
        self._record(st)
        _reconcile(st, _wing_at_broker())
        assert st.state_keys_like("residual_leg_%")                   # still there: leg held
        ib = FakeIB()
        ib.positions_list = [_option_position("SPY", 500.0, "P", _EXP, 1.0)]    # another leg
        _reconcile(st, ib)
        assert st.state_keys_like("residual_leg_%") == []

    def test_a_book_that_cannot_be_confirmed_flat_keeps_the_record(self, tmp_path):
        st = _state(tmp_path)
        self._record(st)

        class _Unconfirmed(FakeIBKRClient):
            async def get_positions_fresh(self):
                return None

        asyncio.run(PositionReconciler(_Unconfirmed(FakeIB()), st).reconcile())
        assert st.state_keys_like("residual_leg_%")

    def test_the_helper_survives_a_broken_record(self):
        st = SimpleNamespace(get_state=lambda k, d="": "{not json")
        assert rec_mod.is_expected_residual(st, WING_KEY, 1.0, today_et()) is False


# --------------------------------------------------------------------------
# the commission true-up
# --------------------------------------------------------------------------
class _KV:
    def __init__(self, **kv):
        self.kv = dict(kv)

    def get_state(self, key, default=""):
        return self.kv.get(key, default)

    def delete_state(self, key):
        self.kv.pop(key, None)


def _orch(state):
    o = TradingOrchestrator.__new__(TradingOrchestrator)
    o._state = state
    return o


def _trade(**kw):
    return SimpleNamespace(trade_id="T-1", legs=json.dumps(LEGS), exit_reason_detailed="",
                           **kw)


class TestTheCommissionLedgerExpectsOneRowFewer:
    def test_a_full_close_expects_eight_rows(self):
        assert _orch(_KV())._expected_commission_rows(_trade()) == 8

    def test_a_three_leg_close_expects_seven(self):
        assert _orch(_KV(**{"exit_excluded_legs_T-1": "1"}))._expected_commission_rows(_trade()) == 7

    def test_a_count_of_zero_is_a_full_close(self):
        assert _orch(_KV(**{"exit_excluded_legs_T-1": "0"}))._expected_commission_rows(_trade()) == 8

    def test_an_expiry_booking_ignores_the_count(self):
        kv = _KV(**{"exit_excluded_legs_T-1": "1", "commission_est_T-1": "0.0"})
        t = SimpleNamespace(trade_id="T-1", legs=json.dumps(LEGS),
                            exit_reason_detailed="reconciler_expired_settlement")
        assert _orch(kv)._expected_commission_rows(t) == 4

    def test_a_mocked_store_changes_nothing(self):
        from unittest.mock import MagicMock
        assert _orch(MagicMock())._expected_commission_rows(_trade()) == 8

    def test_forgetting_the_estimate_forgets_the_count(self):
        kv = _KV(**{"exit_excluded_legs_T-1": "1", "commission_est_T-1": "1.3"})
        _orch(kv)._forget_commission_estimate("T-1")
        assert kv.kv == {}


# --------------------------------------------------------------------------
# parity with the probe that found it
# --------------------------------------------------------------------------
def test_the_bot_leaves_out_what_the_probe_excludes():
    spec = importlib.util.spec_from_file_location(
        "probe_w101_parity", ROOT / "scripts" / "probe_w101_combo_close.py")
    probe = importlib.util.module_from_spec(spec)
    sys.modules["probe_w101_parity"] = probe
    spec.loader.exec_module(probe)
    close_legs = [{"conId": i + 1, "action": "SELL" if l["action"] == "BUY" else "BUY"}
                  for i, l in enumerate(LEGS)]
    unbid = probe.sell_legs_without_bid(close_legs, INCIDENT)
    worthless = [c for c in unbid if INCIDENT[c][1] <= probe.WORTHLESS_ASK]
    o, ibkr, trade, _ = _close()
    kept = {c for c, _a in _placed_legs(ibkr)}
    assert set(worthless) == {1, 2, 3, 4} - kept == {4}
    assert probe.WORTHLESS_ASK == o._settings.exit.exit_unbid_wing_max_ask

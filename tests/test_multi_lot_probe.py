"""W112: the W101 probe at more than one lot (`--quantity 2|3` on open / close, a staged close).

The probe places real (paper) orders, so the broker is faked at the network edge as in
``test_w101_probe_script``; every function the probe defines runs for real, and so does the bot's
``PositionReconciler.position_liveness`` that the probe's evidence calls.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.test_w101_probe_script import (  # noqa: F401  (_sandbox is autouse and must be in scope)
    FakeIBKR, TestCmdOpen, World, _Event, _args, _quote_legs, _sandbox, probe,
)


class MultiWorld(World):
    """Fills the order's whole quantity: positions move by +-N, executions report N shares."""

    def _fill(self, tr, px):
        n = float(tr.order.totalQuantity)
        st = tr.orderStatus
        st.status, st.filled, st.remaining, st.avgFillPrice = "Filled", n, 0.0, px
        legs = []
        if getattr(tr.contract, "secType", "") == "BAG":
            for cl in tr.contract.comboLegs:
                sign = 1.0 if cl.action == "BUY" else -1.0
                self.positions[cl.conId] = self.positions.get(cl.conId, 0.0) + sign * n
                legs.append((cl.conId, cl.action))
            rows = [("BAG", 0, "BOT", px)] + [("OPT", c, "BOT" if a == "BUY" else "SLD", px / 4) for c, a in legs]
        else:
            sign = 1.0 if tr.order.action == "BUY" else -1.0
            self.positions[tr.contract.conId] = self.positions.get(tr.contract.conId, 0.0) + sign * n
            rows = [("OPT", tr.contract.conId, "BOT" if sign > 0 else "SLD", px)]
        for i, (sec, cid, side, price) in enumerate(rows):
            row = SimpleNamespace(
                execution=SimpleNamespace(execId=f"E{tr.order.orderId}-{i}", permId=st.permId,
                                          orderId=tr.order.orderId, side=side, price=price, shares=n),
                contract=SimpleNamespace(secType=sec, conId=cid), time=self.t,
                commissionReport=SimpleNamespace(commission=0.62 if sec == "OPT" else 0.0))
            tr.fills.append(row)
            self.executions.append(row)


class MultiIBKR(FakeIBKR):
    connected = True

    def get_positions(self):
        return self.world.positions_list()


@pytest.fixture
def mworld(monkeypatch):
    w = MultiWorld()
    real_sleep = asyncio.sleep

    async def fake_sleep(s):
        steps = max(1, int(round(s)))
        for _ in range(steps):
            w.t += s / steps
            w.step()
            await real_sleep(0)

    monkeypatch.setattr(probe, "_sleep", fake_sleep)
    monkeypatch.setattr(probe, "_clock", lambda: w.t)

    async def no_sampler_loop(self):
        return None

    monkeypatch.setattr(probe.QuoteSampler, "run", no_sampler_loop)
    yield w
    for tr in w.placed:
        assert str(getattr(tr.order, "orderRef", "") or "").startswith("W101PROBE")
        assert 1 <= tr.order.totalQuantity <= probe.MAX_QUANTITY
    assert w.max_live <= 1, "two probe orders were live at once"
    probe.use_quantity(1)


@pytest.fixture
def cmds(mworld, monkeypatch):
    ibkr = MultiIBKR(mworld)

    async def _connect(data_type=3):
        return ibkr

    monkeypatch.setattr(probe, "connect_probe", _connect)
    monkeypatch.setattr(probe, "guard_window", lambda *a, **k: None)
    monkeypatch.setattr(probe, "guard_bot_stopped", lambda *a, **k: None)
    mworld.ibkr = ibkr
    return mworld


def _trial(world, lots, *, remaining=None, strikes=(600.0, 603.0, 612.0, 615.0), expiry="20261030"):
    """A filled trial in the state file, the broker holding `lots` lots of it."""
    specs = [(strikes[0], "P", "BUY", "SELL"), (strikes[1], "P", "SELL", "BUY"),
             (strikes[2], "C", "SELL", "BUY"), (strikes[3], "C", "BUY", "SELL")]
    held = lots if remaining is None else remaining
    legs = []
    for strike, right, oa, ca in specs:
        cid = world._con_ids.setdefault((strike, right, expiry), 9000 + len(world._con_ids))
        world.contracts[cid] = SimpleNamespace(conId=cid, symbol="SPY", secType="OPT", strike=strike, right=right,
                                               lastTradeDateOrContractMonth=expiry)
        world.positions[cid] = held if oa == "BUY" else -held
        legs.append({"conId": cid, "strike": strike, "right": right, "expiry": expiry,
                     "open_action": oa, "close_action": ca})
    tstate = {"name": "T", "kind": "far", "symbol": "SPY", "created": "x", "expiry": expiry, "dte": 5,
              "width": 3.0, "data_type": 3, "legs": legs, "open": {"status": "filled"},
              "quantity": lots, "remaining": lots if remaining is None else remaining}
    state = {"schema": 1, "trials": {"T": tstate}}
    probe.save_state(state)
    return state, tstate


def _bag_fills_at_any_limit():
    return lambda tr, age: tr.order.lmtPrice if getattr(tr.contract, "secType", "") == "BAG" else None


def _events():
    return [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]


# --------------------------------------------------------------------------
# the lot count and its guard
# --------------------------------------------------------------------------
class TestTheLotCount:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_one_to_three_lots_are_accepted(self, n):
        probe.use_quantity(n)
        assert probe.QUANTITY == n
        probe.use_quantity(1)

    @pytest.mark.parametrize("bad", [0, 4, -1, True, 2.0, "2", None])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(probe.ProbeGuardError, match="1 to 3 lots"):
            probe.use_quantity(bad)
        assert probe.QUANTITY == 1

    @pytest.mark.parametrize("q", [2, 3])
    def test_guard_caps_allows_a_multi_lot_condor(self, q, monkeypatch):
        monkeypatch.setattr(probe, "QUANTITY", q)
        legs = [{"strike": 600.0, "right": "P", "action": "BUY", "expiry": "20261030"},
                {"strike": 603.0, "right": "P", "action": "SELL", "expiry": "20261030"},
                {"strike": 612.0, "right": "C", "action": "SELL", "expiry": "20261030"},
                {"strike": 615.0, "right": "C", "action": "BUY", "expiry": "20261030"}]
        probe.guard_caps(legs)

    @pytest.mark.parametrize("q", [0, 4])
    def test_guard_caps_still_refuses_the_rest(self, q, monkeypatch):
        monkeypatch.setattr(probe, "QUANTITY", q)
        with pytest.raises(probe.ProbeGuardError, match="1 to 3 lots"):
            probe.guard_caps([])

    def test_the_parser_takes_quantity_on_open_close_and_preflight(self):
        p = probe.build_parser()
        assert p.parse_args(["open", "--trial", "M", "--kind", "far", "--quantity", "2"]).quantity == 2
        assert p.parse_args(["close", "--trial", "M", "--quantity", "1"]).quantity == 1
        assert p.parse_args(["close", "--trial", "M"]).quantity is None
        assert p.parse_args(["preflight", "--quantity", "3"]).quantity == 3
        with pytest.raises(SystemExit):
            p.parse_args(["open", "--trial", "M", "--kind", "far", "--quantity", "4"])


# --------------------------------------------------------------------------
# the trial's bookkeeping
# --------------------------------------------------------------------------
class TestTrialBookkeeping:
    def test_a_new_trial_records_its_lots(self):
        probe.use_quantity(3)
        t = probe.new_trial("X", "far", {"expiry": "e", "dte": 5, "width": 3.0, "legs": []}, 3)
        probe.use_quantity(1)
        assert t["quantity"] == 3 and t["remaining"] == 3

    def test_closed_means_flat(self):
        assert probe.trial_closed({"remaining": 0}) is True
        assert probe.trial_closed({"remaining": 1, "close": {"status": "filled"}}) is False
        assert probe.trial_closed({"close": {"status": "filled"}}) is True      # an older trial
        assert probe.trial_closed({"open": {"status": "filled"}}) is False

    def test_a_partly_closed_trial_still_owns_its_legs(self, mworld):
        state, t = _trial(mworld, 3, remaining=2)
        assert probe.owned_con_ids(state) == {int(l["conId"]) for l in t["legs"]}
        t["remaining"] = 0
        assert probe.owned_con_ids(state) == set()

    def test_positions_must_match_the_lots_still_held(self, mworld):
        state, t = _trial(mworld, 3, remaining=2)
        ok, _ = probe._positions_match_open(t, mworld.positions_list())
        assert ok
        t["remaining"] = 3
        ok, _ = probe._positions_match_open(t, mworld.positions_list())
        assert not ok


# --------------------------------------------------------------------------
# the staged close
# --------------------------------------------------------------------------
class TestAStagedClose:
    def test_one_lot_of_two_leaves_one_and_the_bot_reads_it_correctly(self, cmds):
        state, t = _trial(cmds, 2)
        _quote_legs(cmds, t)
        cmds.matcher = _bag_fills_at_any_limit()
        out = asyncio.run(probe.cmd_close(_args(quantity=1)))
        assert out["outcome"] == "filled" and out["remaining"] == 1
        assert [tr.order.totalQuantity for tr in cmds.placed] == [1]
        assert {cmds.positions[int(l["conId"])] for l in t["legs"]} == {1.0, -1.0}
        saved = probe.load_state()["trials"]["T"]
        assert saved["remaining"] == 1 and not probe.trial_closed(saved)
        check = next(e for e in _events() if e["event"] == "quantity_check" and e["step"] == "close")
        assert check["lots_closed"] == 1 and check["remaining"] == 1
        # the bot's exit path: a row at the lots still held is whole; at the old size it is short of it
        assert check["bot_liveness_at_remaining"] == "live"
        assert check["bot_liveness_at_old_size"] == "partial"

    def test_the_second_close_takes_what_is_left_and_the_trial_is_then_flat(self, cmds):
        state, t = _trial(cmds, 2, remaining=1)
        _quote_legs(cmds, t)
        cmds.matcher = _bag_fills_at_any_limit()
        out = asyncio.run(probe.cmd_close(_args()))
        assert out["remaining"] == 0
        assert [tr.order.totalQuantity for tr in cmds.placed] == [1]
        assert all(cmds.positions[int(l["conId"])] == 0 for l in t["legs"])
        assert probe.trial_closed(probe.load_state()["trials"]["T"])
        with pytest.raises(probe.ProbeGuardError, match="already closed"):
            asyncio.run(probe.cmd_close(_args()))

    def test_closing_with_no_quantity_closes_every_lot(self, cmds):
        state, t = _trial(cmds, 3)
        _quote_legs(cmds, t)
        cmds.matcher = _bag_fills_at_any_limit()
        out = asyncio.run(probe.cmd_close(_args()))
        assert out["remaining"] == 0
        assert [tr.order.totalQuantity for tr in cmds.placed] == [3]
        assert all(cmds.positions[int(l["conId"])] == 0 for l in t["legs"])

    def test_more_lots_than_are_held_is_refused_before_anything_is_placed(self, cmds):
        state, t = _trial(cmds, 2)
        _quote_legs(cmds, t)
        with pytest.raises(probe.ProbeGuardError, match="holds 2 lot"):
            asyncio.run(probe.cmd_close(_args(quantity=3)))
        assert cmds.placed == []

    def test_a_close_that_does_not_fill_leaves_the_lots_untouched(self, cmds):
        state, t = _trial(cmds, 2)
        _quote_legs(cmds, t)
        cmds.matcher = lambda tr, age: None
        out = asyncio.run(probe.cmd_close(_args(quantity=1)))
        assert out["outcome"] != "filled"
        assert probe.load_state()["trials"]["T"]["remaining"] == 2

    def test_the_positions_must_match_the_lots_the_trial_says_it_holds(self, cmds):
        state, t = _trial(cmds, 2, remaining=2)
        _quote_legs(cmds, t)
        cmds.positions[t["legs"][0]["conId"]] = 1.0          # the broker holds one lot of that leg
        with pytest.raises(probe.ProbeGuardError, match="do not match"):
            asyncio.run(probe.cmd_close(_args(quantity=1)))
        assert cmds.placed == []


class TestAMultiLotOpen:
    @pytest.mark.parametrize("lots", [2, 3])
    def test_the_combo_is_sent_at_the_lot_count_and_the_trial_records_it(self, cmds, monkeypatch, lots):
        TestCmdOpen()._setup(cmds, monkeypatch)
        cmds.matcher = lambda tr, age: tr.order.lmtPrice if tr.order.lmtPrice >= -0.72 else None
        out = asyncio.run(probe.cmd_open(_args(quantity=lots)))
        assert out["outcome"] == "filled"
        bags = [tr for tr in cmds.placed if getattr(tr.contract, "secType", "") == "BAG"]
        assert bags and all(tr.order.totalQuantity == lots for tr in bags)
        saved = probe.load_state()["trials"]["T"]
        assert saved["quantity"] == lots and saved["remaining"] == lots
        assert sorted(abs(cmds.positions[int(l["conId"])]) for l in saved["legs"]) == [lots] * 4
        check = next(e for e in _events() if e["event"] == "quantity_check" and e["step"] == "open")
        assert check["lots"] == lots
        assert check["bot_liveness"] == "live"                      # the bot would send a reverse combo
        assert check["bot_liveness_one_lot_more"] == "partial"      # and would refuse one a lot too large

    def test_the_default_is_one_lot(self, cmds, monkeypatch):
        TestCmdOpen()._setup(cmds, monkeypatch)
        cmds.matcher = lambda tr, age: tr.order.lmtPrice if tr.order.lmtPrice >= -0.72 else None
        asyncio.run(probe.cmd_open(_args()))
        assert {tr.order.totalQuantity for tr in cmds.placed} == {1}
        assert probe.load_state()["trials"]["T"]["remaining"] == 1

    def test_four_lots_never_reach_the_broker(self, cmds, monkeypatch):
        TestCmdOpen()._setup(cmds, monkeypatch)
        with pytest.raises(probe.ProbeGuardError, match="1 to 3 lots"):
            asyncio.run(probe.cmd_open(_args(quantity=4)))
        assert cmds.placed == []


class TestFlatteningAMultiLotTrial:
    def test_the_combo_flatten_closes_every_lot_still_held(self, cmds):
        state, t = _trial(cmds, 3, remaining=2)
        _quote_legs(cmds, t)
        cmds.matcher = _bag_fills_at_any_limit()
        out = asyncio.run(probe.cmd_flatten(_args()))
        assert out["status"] == "flattened_by_combo"
        assert [tr.order.totalQuantity for tr in cmds.placed] == [2]
        assert probe.trial_closed(probe.load_state()["trials"]["T"])

    def test_a_leg_out_sizes_each_order_from_the_leg_position(self, cmds):
        state, t = _trial(cmds, 2)
        cmds.set_spot(607.0)
        for l in t["legs"]:
            cmds.set_leg(l["conId"], 0.05, 0.10)
        cmds.matcher = lambda tr, age: (tr.order.lmtPrice if getattr(tr.contract, "secType", "") == "OPT" else None)
        out = asyncio.run(probe.cmd_flatten(_args(legout_only=True)))
        assert out["legout"]["status"] == "filled"
        assert sorted(tr.order.totalQuantity for tr in cmds.placed) == [2, 2, 2, 2]
        assert probe.trial_closed(probe.load_state()["trials"]["T"])

"""W101-P: scripts/probe_w101_combo_close.py.

The probe places real (paper) IBKR orders, so nothing here touches a real IBKR connection. IBKR
is faked at the network edge only; every business-logic function the probe defines (the pricing
maths, the guards, the ladder, the leg-out, the report) runs for real. No inspect.getsource.
"""
from __future__ import annotations

import asyncio
import fcntl
import importlib.util
import json
import os
import sqlite3
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ait.utils.time import ET  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "probe_w101_combo_close", ROOT / "scripts" / "probe_w101_combo_close.py")
probe = importlib.util.module_from_spec(_SPEC)
sys.modules["probe_w101_combo_close"] = probe
_SPEC.loader.exec_module(probe)  # type: ignore[union-attr]

_REAL_SAMPLER_RUN = probe.QuoteSampler.run                 # the autouse `world` fixture stubs the loop

# The incident (QQQ condor 704P/707P/769C/772C, 2026-10-23 15:58 ET): delayed leg quotes.
QUOTES = {1: (0.50, 0.52), 2: (2.82, 2.87), 3: (-1.0, 0.01), 4: (-1.0, 0.01)}
OPEN_LEGS = [
    {"conId": 1, "strike": 704.0, "right": "P", "action": "BUY", "expiry": "20261023"},
    {"conId": 2, "strike": 707.0, "right": "P", "action": "SELL", "expiry": "20261023"},
    {"conId": 3, "strike": 769.0, "right": "C", "action": "SELL", "expiry": "20261023"},
    {"conId": 4, "strike": 772.0, "right": "C", "action": "BUY", "expiry": "20261023"},
]
CLOSE_LEGS = [{**l, "action": "SELL" if l["action"] == "BUY" else "BUY"} for l in OPEN_LEGS]
WIDTH = 3.0


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Evidence, state and every file the guards read live under tmp_path."""
    monkeypatch.setattr(probe, "PROBE_DIR", tmp_path / "w101_probe")
    monkeypatch.setattr(probe, "STATE_PATH", tmp_path / "w101_probe" / "state.json")
    monkeypatch.setattr(probe, "LOCK_PATH", tmp_path / "orchestrator.lock")
    monkeypatch.setattr(probe, "HEARTBEAT_PATH", tmp_path / "bot_heartbeat")
    monkeypatch.setattr(probe, "DB_PATH", tmp_path / "ait_state.db")
    monkeypatch.setattr(probe, "HALT_UNTRACKED_PATH", tmp_path / "HALT_UNTRACKED")


# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
class TestIdentity:
    def test_the_client_id_is_not_the_bots_nor_an_existing_script(self):
        assert probe.PROBE_CLIENT_ID == 96
        assert probe.PROBE_CLIENT_ID not in (1, 83, 88, 90, 91, 95, 97, 98) + tuple(range(101, 106))

    def test_pinned_to_one_spy_lot(self):
        assert probe.SYMBOL == "SPY" and probe.QUANTITY == 1 and probe.MAX_WIDTH <= 3.0

    def test_order_ref_format(self):
        assert probe.make_order_ref("F1", "close", 3, epoch=1790707000) == "W101PROBE-F1-close-r3-1790707000"


# --------------------------------------------------------------------------
# the maths
# --------------------------------------------------------------------------
class TestNaturalCost:
    def test_the_incident_quotes(self):
        # BUY 707P at 2.87 + BUY 769C at 0.01 - SELL 704P at 0.50 - SELL 772C at bid -1 (= 0)
        assert probe.natural_cost(CLOSE_LEGS, QUOTES) == pytest.approx(2.38)

    def test_a_zero_bid_is_a_bid(self):
        q = {**QUOTES, 4: (0.0, 0.02)}
        assert probe.natural_cost(CLOSE_LEGS, q) == pytest.approx(2.38)

    @pytest.mark.parametrize("conid,quote", [
        (2, (2.82, None)), (2, (2.82, float("nan"))), (2, (2.82, 0.0)), (2, (2.82, -1.0)),   # BUY leg needs an ask
        (1, (None, 0.52)), (1, (float("nan"), 0.52)),                                         # SELL leg: no data
    ])
    def test_missing_data_on_a_leg_is_none_never_a_guess(self, conid, quote):
        assert probe.natural_cost(CLOSE_LEGS, {**QUOTES, conid: quote}) is None

    def test_minus_one_on_a_buy_leg_is_not_a_bid_it_is_no_ask(self):
        assert probe.natural_cost(CLOSE_LEGS, {**QUOTES, 3: (-1.0, -1.0)}) is None

    def test_mid_cost_needs_both_sides(self):
        assert probe.mid_cost(CLOSE_LEGS, QUOTES) is None            # the bidless wings
        full = {1: (0.50, 0.52), 2: (2.82, 2.87), 3: (0.0, 0.02), 4: (0.0, 0.02)}
        # BUY 707P mid 2.845, BUY 769C mid 0.01, SELL 704P mid 0.51, SELL 772C mid 0.01
        assert probe.mid_cost(CLOSE_LEGS, full) == pytest.approx(2.845 + 0.01 - 0.51 - 0.01)


class TestIntrinsic:
    def test_the_incident_settles_at_2_60(self):
        assert probe.structure_intrinsic(OPEN_LEGS, 704.4) == pytest.approx(2.60)

    def test_the_sign_pitfall_with_reversed_legs(self):
        """The valuation needs the OPENING legs: the reversed (closing) legs flip the sign."""
        assert probe.structure_intrinsic(CLOSE_LEGS, 704.4) == pytest.approx(-2.60)

    def test_floor_is_point_intrinsic_plus_cross_far_from_expiry(self):
        assert probe.intrinsic_floor(OPEN_LEGS, 704.4, 3, WIDTH, 3) == pytest.approx(2.70)

    def test_floor_allows_for_a_delayed_spot_only_near_expiry_on_delayed_data(self):
        assert probe.intrinsic_floor(OPEN_LEGS, 704.4, 0, WIDTH, 3) == pytest.approx(3.0)
        assert probe.intrinsic_floor(OPEN_LEGS, 704.4, 1, WIDTH, 4) == pytest.approx(3.0)
        assert probe.intrinsic_floor(OPEN_LEGS, 704.4, 0, WIDTH, 1) == pytest.approx(2.70)       # live data
        assert probe.intrinsic_floor(OPEN_LEGS, 704.4, 2, WIDTH, 3) == pytest.approx(2.70)       # DTE 2
        assert probe.intrinsic_floor(OPEN_LEGS, 704.4, 0, WIDTH, 3, must_get_out=False) == pytest.approx(2.70)

    def test_out_of_the_money_has_no_floor(self):
        assert probe.intrinsic_floor(OPEN_LEGS, 740.0, 0, WIDTH, 3) is None

    @pytest.mark.parametrize("bad", [None, 0.0, -1.0, float("nan"), "x"])
    def test_an_unusable_spot_has_no_floor(self, bad):
        assert probe.intrinsic_floor(OPEN_LEGS, bad, 0, WIDTH, 3) is None

    def test_never_above_the_width(self):
        assert probe.intrinsic_floor(OPEN_LEGS, 650.0, 0, WIDTH, 3) == pytest.approx(WIDTH)


class TestRungs:
    def test_rungs_0_to_3_are_the_bots_rule(self):
        n = 2.38
        want = [probe.bot_rung_limit(k, n) for k in range(4)]
        assert [round(w, 2) for w in want] == [2.48, 2.60, 2.73, 2.85]
        prev = l3 = None
        got = []
        for k in range(4):
            lim, _ = probe.rung_limit(k, n, prev, l3, WIDTH, None)
            got.append(lim)
            prev = lim
        assert got == [2.48, 2.60, 2.73, 2.85]

    def test_rungs_4_to_6_close_the_gap_to_the_width(self):
        ladder = probe.planned_ladder(2.38, WIDTH)
        assert ladder[:4] == [2.48, 2.60, 2.73, 2.85]
        l3 = 2.85
        assert ladder[4] == round(l3 + 0.25 * (WIDTH - l3), 2)
        assert ladder[5] == round(l3 + 0.50 * (WIDTH - l3), 2)
        assert ladder[-1] == WIDTH

    def test_a_ladder_that_hits_the_width_early_has_no_duplicates(self):
        ladder = probe.planned_ladder(2.90, WIDTH)
        assert ladder[-1] == WIDTH and len(set(ladder)) == len(ladder)

    def test_floor_lifts_a_rung_and_is_recorded(self):
        lim, info = probe.rung_limit(0, 2.38, None, None, WIDTH, 2.70)
        assert lim == 2.70 and info["floor_bound"] is True

    def test_cap_binds_flags_a_natural_above_the_width(self):
        lim, info = probe.rung_limit(0, 3.20, None, None, WIDTH, None)
        assert lim == WIDTH and info["cap_binds"] is True and info["cap_bound"] is True

    def test_never_steps_down_when_quotes_fall(self):
        lim, _ = probe.rung_limit(1, 1.50, 2.60, None, WIDTH, None)
        assert lim is None or lim >= 2.60

    def test_a_duplicate_is_skipped(self):
        lim, info = probe.rung_limit(5, 2.0, WIDTH, 2.9, WIDTH, None)
        assert lim is None and info["skipped_duplicate"] is True

    def test_property_loop(self):
        import random
        rng = random.Random(101)
        for _ in range(200):
            n = round(rng.uniform(0.2, 3.4), 2)
            floor = rng.choice([None, round(rng.uniform(0.1, 3.0), 2)])
            ladder = probe.planned_ladder(n, WIDTH, floor)
            assert ladder, (n, floor)
            assert all(b > a for a, b in zip(ladder, ladder[1:])), ladder
            assert all(x <= WIDTH + 1e-9 for x in ladder), ladder
            if floor is not None:
                assert all(x >= floor - 1e-9 for x in ladder), (ladder, floor)
            if n + probe.EXIT_CROSS <= WIDTH:
                assert ladder[-1] == WIDTH, (n, ladder)           # the last rung reaches the wing width

    def test_entry_ladder_walks_from_mid_through_natural(self):
        assert [probe.entry_rung_limit(k, 1.00, 1.10) for k in range(5)] == [1.0, 1.05, 1.1, 1.15, 1.2]


class TestParityWithTheBot:
    """The probe is only evidence about the bot if it prices like the bot. Drive the real
    `TradingOrchestrator._close_multi_leg` and compare."""

    def test_rungs_0_to_3_equal_the_bots_placements(self, monkeypatch):
        import ait.bot.orchestrator as orch_mod
        from tests.test_w101_exit_floor import INCIDENT, _go, _rig
        real = asyncio.sleep

        async def _no_wait(_s):
            await real(0)

        monkeypatch.setattr(orch_mod.asyncio, "sleep", _no_wait)
        o, ibkr, trade = _rig(leg_quotes=dict(INCIDENT), entry=0.92)      # no mark, no spot: path B only
        bot = [_go(o, ibkr, trade) for _ in range(4)]
        n = probe.natural_cost(CLOSE_LEGS, QUOTES)
        mine, prev = [], None
        for k in range(4):
            lim, _ = probe.rung_limit(k, n, prev, None, WIDTH, None)
            mine.append(lim)
            prev = lim
        assert mine == pytest.approx(bot)

    def test_the_floor_equals_the_bots_floor(self, monkeypatch):
        from tests.test_w101_exit_floor import INCIDENT, LEGS, _rig, TOUCH
        monkeypatch.setenv("AIT_MARKET_DATA_TYPE", "3")
        o, _ibkr, trade = _rig(leg_quotes=dict(INCIDENT), entry=0.92)
        for spot, dte in ((704.4, 0), (704.4, 1), (704.4, 3), (706.17, 0), (740.0, 0)):
            bot = o._exit_intrinsic_floor(trade, LEGS, spot, dte, WIDTH, TOUCH)
            mine = probe.intrinsic_floor(OPEN_LEGS, spot, dte, WIDTH, 3)
            assert (mine is None) == (bot is None), (spot, dte)
            if bot is not None:
                assert mine == pytest.approx(bot), (spot, dte)


# --------------------------------------------------------------------------
# evidence and state
# --------------------------------------------------------------------------
class TestEvidenceAndState:
    def test_evidence_lines_carry_the_envelope(self):
        ev = probe.Evidence("close", "F1", run_id="abc")
        ev.emit("rung_plan", k=0, limit=2.48)
        ev.emit("trial_end", outcome="filled")
        files = list((probe.PROBE_DIR).glob("*/events.jsonl"))
        assert len(files) == 1
        lines = [json.loads(l) for l in files[0].read_text().splitlines()]
        assert [l["event"] for l in lines] == ["rung_plan", "trial_end"]
        assert all(l["schema"] == 1 and l["run_id"] == "abc" and l["cmd"] == "close"
                   and l["trial"] == "F1" and "ts" in l and "mono" in l for l in lines)
        assert lines[0]["limit"] == 2.48

    def test_state_round_trip_and_atomic_write(self):
        assert probe.load_state() == {"schema": 1, "trials": {}}
        probe.save_state({"schema": 1, "trials": {"F1": {"open": {"status": "filled"}}}})
        assert probe.load_state()["trials"]["F1"]["open"]["status"] == "filled"
        assert not list(probe.PROBE_DIR.glob("*.tmp"))

    def test_state_survives_a_date_change(self, monkeypatch):
        probe.save_state({"schema": 1, "trials": {"P1": {"open": {"status": "filled"}}}})
        monkeypatch.setattr(probe, "now_et", lambda: datetime(2026, 10, 28, 10, 0, tzinfo=ET))
        assert "P1" in probe.load_state()["trials"]

    def test_a_corrupt_state_file_is_refused_with_a_message(self):
        probe.PROBE_DIR.mkdir(parents=True)
        probe.STATE_PATH.write_text("{not json")
        with pytest.raises(probe.ProbeGuardError, match="corrupt"):
            probe.load_state()

    def test_an_unknown_schema_is_refused(self):
        probe.PROBE_DIR.mkdir(parents=True)
        probe.STATE_PATH.write_text(json.dumps({"schema": 99, "trials": {}}))
        with pytest.raises(probe.ProbeGuardError, match="schema"):
            probe.load_state()


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------
def _at(hour, minute, monkeypatch, *, open_=True):
    monkeypatch.setattr(probe, "now_et", lambda: datetime(2026, 10, 27, hour, minute, tzinfo=ET))
    monkeypatch.setattr(probe, "is_market_open", lambda: open_)
    monkeypatch.setattr(probe, "get_market_close", lambda d: dtime(16, 0))


class TestGuardAccountAndClient:
    def test_paper_passes_live_refused(self):
        probe.guard_account(SimpleNamespace(session_account="DU0000001"))
        for bad in ("U123456", "", None):
            with pytest.raises(probe.ProbeGuardError):
                probe.guard_account(SimpleNamespace(session_account=bad))

    def test_only_client_96_passes(self):
        probe.guard_client_id(SimpleNamespace(client_id=96))
        for bad in (1, 97, 196, None):
            with pytest.raises(probe.ProbeGuardError, match="client id"):
                probe.guard_client_id(SimpleNamespace(client_id=bad))


class TestGuardBotStopped:
    def test_a_held_supervisor_lock_is_refused(self, tmp_path):
        lock = tmp_path / "orch.lock"
        lock.write_text("12599\n")
        holder = open(lock, "a+")
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(probe.ProbeGuardError, match="supervisor"):
                probe.guard_bot_stopped(lock, tmp_path / "hb", market_open=False)
        finally:
            fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
            holder.close()
        probe.guard_bot_stopped(lock, tmp_path / "hb", market_open=False)       # released: passes

    def test_a_stale_pid_in_an_unlocked_file_passes(self, tmp_path):
        lock = tmp_path / "orch.lock"
        lock.write_text("12599\n")
        probe.guard_bot_stopped(lock, tmp_path / "hb", market_open=False)

    def test_no_lock_file_passes(self, tmp_path):
        probe.guard_bot_stopped(tmp_path / "nope", tmp_path / "hb", market_open=False)

    def test_a_fresh_heartbeat_in_rth_is_refused_a_stale_or_off_hours_one_passes(self, tmp_path):
        hb = tmp_path / "hb"
        hb.write_text("x")
        now = hb.stat().st_mtime + 60
        with pytest.raises(probe.ProbeGuardError, match="alive"):
            probe.guard_bot_stopped(tmp_path / "nope", hb, now_ts=now, market_open=True)
        probe.guard_bot_stopped(tmp_path / "nope", hb, now_ts=now, market_open=False)
        probe.guard_bot_stopped(tmp_path / "nope", hb, now_ts=hb.stat().st_mtime + 3600, market_open=True)


class TestGuardWindow:
    @pytest.mark.parametrize("hm,ok", [((9, 59), False), ((10, 0), True), ((15, 15), True), ((15, 16), False)])
    def test_open_window(self, monkeypatch, hm, ok):
        _at(*hm, monkeypatch)
        if ok:
            probe.guard_window("open", "far")
        else:
            with pytest.raises(probe.ProbeGuardError):
                probe.guard_window("open", "far")

    @pytest.mark.parametrize("hm,ok", [((10, 0), True), ((15, 30), True), ((15, 31), False)])
    def test_close_window(self, monkeypatch, hm, ok):
        _at(*hm, monkeypatch)
        if ok:
            probe.guard_window("close", "far")
        else:
            with pytest.raises(probe.ProbeGuardError):
                probe.guard_window("close", "far")

    @pytest.mark.parametrize("hm,ok", [((9, 34), False), ((9, 35), True), ((15, 55), True), ((15, 56), False)])
    def test_flatten_and_cancel_keep_only_five_minutes_clear(self, monkeypatch, hm, ok):
        _at(*hm, monkeypatch)
        for cmd in ("flatten", "cancel"):
            if ok:
                probe.guard_window(cmd)
            else:
                with pytest.raises(probe.ProbeGuardError):
                    probe.guard_window(cmd)

    def test_a_0dte_trial_has_its_own_latest_start(self, monkeypatch):
        _at(14, 31, monkeypatch)
        with pytest.raises(probe.ProbeGuardError, match="14:30"):
            probe.guard_window("open", "pin")
        probe.guard_window("open", "far")
        _at(15, 16, monkeypatch)
        with pytest.raises(probe.ProbeGuardError, match="15:15"):
            probe.guard_window("close", "pin")
        probe.guard_window("close", "far")
        probe.guard_window("flatten", "pin")                      # an emergency close is never blocked by this

    def test_a_closed_market_is_refused(self, monkeypatch):
        _at(11, 0, monkeypatch, open_=False)
        with pytest.raises(probe.ProbeGuardError, match="not open"):
            probe.guard_window("open", "far")


class TestGuardCaps:
    LEGS = [
        {"strike": 600.0, "right": "P", "action": "BUY", "expiry": "20261027"},
        {"strike": 603.0, "right": "P", "action": "SELL", "expiry": "20261027"},
        {"strike": 612.0, "right": "C", "action": "SELL", "expiry": "20261027"},
        {"strike": 615.0, "right": "C", "action": "BUY", "expiry": "20261027"},
    ]

    def test_a_proper_condor_passes(self):
        probe.guard_caps(self.LEGS)

    def test_wide_wing_refused(self):
        bad = [dict(l) for l in self.LEGS]
        bad[0]["strike"] = 597.0
        with pytest.raises(probe.ProbeGuardError, match="width"):
            probe.guard_caps(bad)

    def test_three_legs_refused(self):
        with pytest.raises(probe.ProbeGuardError, match="4 legs"):
            probe.guard_caps(self.LEGS[:3])

    def test_two_sells_on_one_side_refused(self):
        bad = [dict(l) for l in self.LEGS]
        bad[0]["action"] = "SELL"
        with pytest.raises(probe.ProbeGuardError):
            probe.guard_caps(bad)

    def test_mixed_expiries_refused(self):
        bad = [dict(l) for l in self.LEGS]
        bad[3]["expiry"] = "20261030"
        with pytest.raises(probe.ProbeGuardError, match="expiry"):
            probe.guard_caps(bad)


class TestGuardBooks:
    @staticmethod
    def _pos(con_id, position=1, symbol="SPY", sec="OPT"):
        return SimpleNamespace(position=position, contract=SimpleNamespace(
            conId=con_id, symbol=symbol, secType=sec, strike=600.0, right="P",
            lastTradeDateOrContractMonth="20261027"))

    @staticmethod
    def _trade(ref, symbol="SPY"):
        return SimpleNamespace(contract=SimpleNamespace(symbol=symbol),
                               order=SimpleNamespace(orderId=7, orderRef=ref))

    def test_unknown_positions_refuse(self):
        with pytest.raises(probe.ProbeGuardError, match="unknown"):
            probe.guard_spy_book(None, [], set())

    def test_a_position_on_a_contract_the_trial_would_trade_refuses_but_its_own_passes(self):
        with pytest.raises(probe.ProbeGuardError, match="does not own"):
            probe.guard_spy_book([self._pos(5)], [], {1, 2, 3, 4}, trade_con_ids={5, 6, 7, 8})
        probe.guard_spy_book([self._pos(1), self._pos(2, -1)], [], {1, 2, 3, 4}, trade_con_ids={1, 2, 3, 4})

    def test_the_operators_positions_on_other_contracts_are_left_alone(self):
        probe.guard_spy_book([self._pos(790, symbol="QQQ"), self._pos(5)], [], {1, 2, 3, 4},
                             trade_con_ids={11, 12, 13, 14})
        probe.guard_spy_book([self._pos(5)], [], set())                 # no trial contracts known yet

    def test_other_symbols_and_stock_and_flat_are_ignored(self):
        probe.guard_spy_book([self._pos(9, symbol="IWM"), self._pos(9, sec="STK"), self._pos(9, position=0)], [], set())

    def test_qqq_contracts_are_protected_like_spy_ones(self):
        with pytest.raises(probe.ProbeGuardError, match="QQQ option position"):
            probe.guard_spy_book([self._pos(9, symbol="QQQ")], [], set(), trade_con_ids={9})

    def test_a_foreign_working_spy_order_refuses_a_probe_order_does_not(self):
        with pytest.raises(probe.ProbeGuardError, match="not a probe order"):
            probe.guard_spy_book([], [self._trade("T-20261027-1")], set())
        with pytest.raises(probe.ProbeGuardError):
            probe.guard_spy_book([], [self._trade("")], set())
        probe.guard_spy_book([], [self._trade("W101PROBE-F1-close-r0-1")], set())
        probe.guard_spy_book([], [self._trade("T-1", symbol="IWM")], set())
        with pytest.raises(probe.ProbeGuardError, match="not a probe order"):
            probe.guard_spy_book([], [self._trade("T-1", symbol="QQQ")], set())

    @pytest.mark.parametrize("status", ["working", "placing"])
    def test_a_working_step_in_state_refuses(self, status):
        with pytest.raises(probe.ProbeGuardError, match="status|marked"):
            probe.guard_no_working({"trials": {"F1": {"close": {"status": status}}}})
        probe.guard_no_working({"trials": {"F1": {"close": {"status": "filled"}}}})


class TestGuardCollision:
    LEG = {"strike": 600.0, "right": "P", "expiry": "20261027"}

    def test_clear(self):
        probe.guard_no_collision([self.LEG])

    def test_halt_untracked_key(self):
        probe.HALT_UNTRACKED_PATH.write_text("SPY:600.0:P:20261027\n")
        with pytest.raises(probe.ProbeGuardError, match="collides"):
            probe.guard_no_collision([self.LEG])

    def test_open_db_trade_leg_but_not_a_closed_one(self):
        con = sqlite3.connect(probe.DB_PATH)
        con.execute("CREATE TABLE trades (symbol TEXT, legs TEXT, status TEXT)")
        con.execute("INSERT INTO trades VALUES ('SPY', ?, 'filled')",
                    (json.dumps([{"strike": 600.0, "right": "P", "expiry": "20261027"}]),))
        con.commit()
        con.close()
        with pytest.raises(probe.ProbeGuardError, match="collides"):
            probe.guard_no_collision([self.LEG])
        con = sqlite3.connect(probe.DB_PATH)
        con.execute("UPDATE trades SET status='closed'")
        con.commit()
        con.close()
        probe.guard_no_collision([self.LEG])


class TestGuardAccountValues:
    OK = {"NetLiquidation": "198000", "ExcessLiquidity": "150000"}

    def test_ok(self):
        probe.guard_account_values(self.OK, 300.0)
        probe.guard_account_values(self.OK, None)

    def test_pdt_floor(self):
        with pytest.raises(probe.ProbeGuardError, match="PDT"):
            probe.guard_account_values({**self.OK, "NetLiquidation": "24000"}, 300.0)

    def test_margin_cover(self):
        with pytest.raises(probe.ProbeGuardError, match="ExcessLiquidity"):
            probe.guard_account_values({**self.OK, "ExcessLiquidity": "800"}, 300.0)

    def test_missing_values_refuse(self):
        with pytest.raises(probe.ProbeGuardError, match="unavailable"):
            probe.guard_account_values({}, 300.0)


# ==========================================================================
# The fake broker: IBKR at the network edge only, a virtual clock, programmable matching.
# ==========================================================================
class _Event:
    def __init__(self):
        self.handlers = []

    def __iadd__(self, f):
        self.handlers.append(f)
        return self

    def __isub__(self, f):
        self.handlers.remove(f)
        return self

    def emit(self, *a):
        for h in list(self.handlers):
            h(*a)


class _Trade:
    def __init__(self, contract, order, born):
        self.contract, self.order, self._born = contract, order, born
        self.orderStatus = SimpleNamespace(status="PendingSubmit", filled=0.0, remaining=float(order.totalQuantity),
                                           avgFillPrice=0.0, permId=0, whyHeld="", mktCapPrice=None)
        self.fills, self.log, self.advancedError = [], [], ""
        self._cancel_at = None

    def isDone(self):
        return self.orderStatus.status in ("Filled", "Cancelled", "ApiCancelled")


class World:
    """A virtual clock plus a fake IB. `matcher(trade, age_s) -> fill price | None` decides fills."""

    def __init__(self):
        self.t = 1000.0
        self.trades, self.executions, self.placed = [], [], []
        self.positions: dict[int, float] = {}
        self.contracts: dict[int, SimpleNamespace] = {}
        self.matcher = lambda tr, age: None
        self.cancel_mode = "normal"        # normal | race_fill | pending_forever | local_cancel_then_exec
        self.inactive = False
        self.reject_code = None
        self.max_live = 0
        self.tickers: dict = {}
        self.errorEvent = _Event()
        self.extra_open_trades: list = []
        self.raise_on_place = False
        self.place_log: list = []                  # (sec_type, action, conId, positions at that moment)
        self.leg_defaults: dict = {}               # (strike, right) -> (bid, ask) for tickers created on demand
        self.allow_many = self.allow_foreign = False
        self.band = None                             # IB's price band: limits at or above it are refused
        self._con_ids: dict = {}
        self._next_oid = 100

    # --- the IB surface the probe touches
    def reqMarketDataType(self, n):
        self.data_type = n

    def _key(self, c):
        if getattr(c, "secType", "") == "BAG":
            return "BAG"
        if getattr(c, "secType", "") == "STK":
            return "SPOT"
        return int(c.conId)

    def reqMktData(self, c, *a):
        key = self._key(c)
        if key not in self.tickers:
            bid = ask = float("nan")
            if isinstance(key, int) and key in self.contracts:
                k = self.contracts[key]
                bid, ask = self.leg_defaults.get((k.strike, k.right), (bid, ask))
            self.tickers[key] = SimpleNamespace(bid=bid, ask=ask, last=float("nan"), close=float("nan"),
                                                marketDataType=3)
        return self.tickers[key]

    def ticker(self, c):
        return self.tickers.get(self._key(c))

    def cancelMktData(self, c):
        pass

    def set_leg(self, con_id, bid, ask):
        t = self.tickers.setdefault(con_id, SimpleNamespace(bid=bid, ask=ask, last=float("nan"),
                                                            close=float("nan"), marketDataType=3))
        t.bid, t.ask = bid, ask

    def set_spot(self, last):
        t = self.tickers.setdefault("SPOT", SimpleNamespace(bid=float("nan"), ask=float("nan"), last=last,
                                                            close=float("nan"), marketDataType=3))
        t.last = last

    def set_bag(self, bid, ask):
        t = self.tickers.setdefault("BAG", SimpleNamespace(bid=bid, ask=ask, last=float("nan"),
                                                           close=float("nan"), marketDataType=3))
        t.bid, t.ask = bid, ask

    def placeOrder(self, contract, order):
        if self.raise_on_place:
            raise RuntimeError("boom")
        order.orderId = self._next_oid
        self._next_oid += 1
        tr = _Trade(contract, order, self.t)
        self.trades.append(tr)
        self.placed.append(tr)
        self.place_log.append((getattr(contract, "secType", ""), order.action, getattr(contract, "conId", None),
                               dict(self.positions)))
        live = [x for x in self.trades if not x.isDone()]
        self.max_live = max(self.max_live, len(live))
        return tr

    def _fill(self, tr, px):
        st = tr.orderStatus
        st.status, st.filled, st.remaining, st.avgFillPrice = "Filled", 1.0, 0.0, px
        legs = []
        if getattr(tr.contract, "secType", "") == "BAG":
            for cl in tr.contract.comboLegs:
                sign = 1.0 if cl.action == "BUY" else -1.0
                self.positions[cl.conId] = self.positions.get(cl.conId, 0.0) + sign
                legs.append((cl.conId, cl.action))
            rows = [("BAG", 0, "BOT", px)] + [("OPT", c, "BOT" if a == "BUY" else "SLD", px / 4) for c, a in legs]
        else:
            sign = 1.0 if tr.order.action == "BUY" else -1.0
            self.positions[tr.contract.conId] = self.positions.get(tr.contract.conId, 0.0) + sign
            rows = [("OPT", tr.contract.conId, "BOT" if sign > 0 else "SLD", px)]
        for i, (sec, cid, side, price) in enumerate(rows):
            row = SimpleNamespace(
                execution=SimpleNamespace(execId=f"E{tr.order.orderId}-{i}", permId=st.permId,
                                          orderId=tr.order.orderId, side=side, price=price, shares=1.0),
                contract=SimpleNamespace(secType=sec, conId=cid), time=self.t,
                commissionReport=SimpleNamespace(commission=0.62 if sec == "OPT" else 0.0))
            tr.fills.append(row)
            self.executions.append(row)

    def cancelOrder(self, order):
        tr = next(t for t in self.trades if t.order.orderId == order.orderId)
        st = tr.orderStatus
        if st.status == "Filled":
            self.errorEvent.emit(order.orderId, 161, "Cancel attempted when order is not in a cancellable state", None)
            return
        if self.cancel_mode == "race_fill":
            self._fill(tr, order.lmtPrice)
        elif self.cancel_mode == "local_cancel_then_exec":
            self._fill(tr, order.lmtPrice)
            st.status, st.filled, st.remaining, tr.fills = "Cancelled", 0.0, 1.0, []    # the local flip
            self.errorEvent.emit(order.orderId, 10148, "OrderId that needs to be cancelled can not be cancelled", None)
        else:
            st.status = "PendingCancel"
            tr._cancel_at = self.t + 2

    def step(self):
        for tr in self.trades:
            st = tr.orderStatus
            if st.status == "PendingSubmit" and self.t >= tr._born + 1:
                st.status, st.permId = "Submitted", 5000 + tr.order.orderId
                if self.band is not None and tr.order.lmtPrice >= self.band:
                    st.status = "Cancelled"
                    self.errorEvent.emit(tr.order.orderId, 202, "Order Canceled - reason:We cannot accept an order at a "
                                         f"limit price at or more aggressive than {self.band}. Please submit your order "
                                         "using a <br>limit price that is closer to the current market price of 1.", None)
                elif self.reject_code:
                    st.status = "Cancelled"
                    self.errorEvent.emit(tr.order.orderId, self.reject_code, "rejected", None)
                elif self.inactive:
                    st.status = "Inactive"
            if st.status == "Submitted":
                px = self.matcher(tr, self.t - tr._born)
                if px is not None:
                    self._fill(tr, px)
            if st.status == "PendingCancel" and self.cancel_mode != "pending_forever" and self.t >= tr._cancel_at:
                st.status = "Cancelled"

    async def reqExecutionsAsync(self, flt):
        return list(self.executions)

    async def reqPositionsAsync(self):
        return self.positions_list()

    def positions_list(self):
        return [SimpleNamespace(position=p, contract=self.contracts[c]) for c, p in self.positions.items() if c in self.contracts]

    async def reqAllOpenOrdersAsync(self):
        return [t for t in self.trades if not t.isDone()] + self.extra_open_trades

    async def whatIfOrderAsync(self, bag, order):
        return SimpleNamespace(initMarginChange="300", maintMarginChange="300", commission="2.5", warningText="")

    async def reqSecDefOptParamsAsync(self, sym, exch, typ, con_id):
        return self.chain_result


class FakeIBKR:
    def __init__(self, world, client_id=96, account="DU0000001"):
        self.ib, self.world = world, world
        self.client_id, self.session_account = client_id, account
        self.cancelled = []

    async def place_order(self, contract, order):
        try:
            return self.world.placeOrder(contract, order)
        except RuntimeError:
            raise

    async def cancel_order(self, trade):
        self.cancelled.append(trade.order.orderId)
        self.world.cancelOrder(trade.order)
        return True

    async def get_positions_fresh(self, timeout=8.0):
        return self.world.positions_list()

    async def get_all_open_trades(self):
        return await self.world.reqAllOpenOrdersAsync()

    async def qualify_contract(self, c):
        if getattr(c, "secType", "") == "STK":
            c.conId = 756733
            return c
        return (await self.qualify_contracts_batch([c]))[0]

    async def qualify_contracts_batch(self, contracts):
        out = []
        for c in contracts:
            key = (float(c.strike), c.right, str(c.lastTradeDateOrContractMonth))
            c.conId = self.world._con_ids.setdefault(key, 9000 + len(self.world._con_ids))
            self.world.contracts[c.conId] = SimpleNamespace(
                conId=c.conId, symbol=c.symbol, secType="OPT", strike=float(c.strike), right=c.right,
                lastTradeDateOrContractMonth=str(c.lastTradeDateOrContractMonth))
            out.append(c)
        return out

    async def get_account_values(self):
        return {"NetLiquidation": "198000", "ExcessLiquidity": "150000", "BuyingPower": "400000"}

    async def disconnect(self):
        pass


def _legs_state(world, ibkr, *, spot_strikes=(600.0, 603.0, 612.0, 615.0), expiry="20261030"):
    """A filled trial in the state file with positions at the fake broker; returns (state, tstate)."""
    specs = [(spot_strikes[0], "P", "BUY", "SELL"), (spot_strikes[1], "P", "SELL", "BUY"),
             (spot_strikes[2], "C", "SELL", "BUY"), (spot_strikes[3], "C", "BUY", "SELL")]
    legs = []
    for strike, right, oa, ca in specs:
        cid = world._con_ids.setdefault((strike, right, expiry), 9000 + len(world._con_ids))
        world.contracts[cid] = SimpleNamespace(conId=cid, symbol="SPY", secType="OPT", strike=strike, right=right,
                                               lastTradeDateOrContractMonth=expiry)
        world.positions[cid] = 1.0 if oa == "BUY" else -1.0
        legs.append({"conId": cid, "strike": strike, "right": right, "expiry": expiry,
                     "open_action": oa, "close_action": ca})
    tstate = {"name": "T", "kind": "far", "created": "x", "expiry": expiry, "dte": 5, "width": 3.0,
              "data_type": 3, "legs": legs, "open": {"status": "filled"}}
    state = {"schema": 1, "trials": {"T": tstate}}
    probe.save_state(state)
    return state, tstate


def _quote_legs(world, tstate, *, natural=2.38):
    """Quotes whose closing natural is `natural`: the BUY-back short put carries it, the rest near zero."""
    for l in tstate["legs"]:
        if l["close_action"] == "BUY" and l["right"] == "P":
            world.set_leg(l["conId"], natural - 0.05, natural + 0.01 - 0.01 + 0.01)   # ask = natural + 0.01 - ...
        else:
            world.set_leg(l["conId"], 0.0, 0.01)
    # recompute so the natural is exact: BUY puts at ask a, BUY call at 0.01, SELL legs at bid 0
    short_put = next(l for l in tstate["legs"] if l["close_action"] == "BUY" and l["right"] == "P")
    short_call = next(l for l in tstate["legs"] if l["close_action"] == "BUY" and l["right"] == "C")
    world.set_leg(short_put["conId"], natural - 0.06, round(natural - 0.01, 2))
    world.set_leg(short_call["conId"], 0.0, 0.01)
    world.set_spot(607.0)                       # between the short strikes (603 / 612): out of the money, no floor


@pytest.fixture
def world(monkeypatch):
    w = World()
    real_sleep = asyncio.sleep

    async def fake_sleep(s):
        steps = max(1, int(round(s)))
        for _ in range(steps):
            w.t += s / steps
            w.step()
            await real_sleep(0)

    monkeypatch.setattr(probe, "_sleep", fake_sleep)
    monkeypatch.setattr(probe, "_clock", lambda: w.t)

    async def no_sampler_loop(self):                # the sampler loop would double-count virtual time
        return None

    monkeypatch.setattr(probe.QuoteSampler, "run", no_sampler_loop)
    yield w
    for tr in w.placed:                              # the invariants of EVERY probe order
        ref = str(getattr(tr.order, "orderRef", "") or "")
        if not w.allow_foreign:
            assert ref.startswith("W101PROBE"), ref
        assert tr.order.totalQuantity == 1
        if (getattr(tr.contract, "secType", "") == "BAG" and tr.order.action == "BUY"
                and ("-close-" in ref or "-flatten-" in ref)):
            assert tr.order.lmtPrice <= 3.0 + 1e-9, (ref, tr.order.lmtPrice)
    if not w.allow_many:
        assert w.max_live <= 1, "two probe orders were live at once"


def _ev(cmd="close"):
    return probe.Evidence(cmd, "T")


async def _close(world, ibkr, state, tstate, *, rest=120.0, tail=False, params=None, step="close"):
    ev = _ev(step)
    errors = probe.ErrorRecorder(world, ev)
    try:
        return await probe._run_close_like(ibkr, state, tstate, "T", step, ev, errors,
                                           params or probe.load_params(), rest_s=rest, tail=tail)
    finally:
        errors.detach()


def _fill_when_limit_at_least(x):
    return lambda tr, age: (tr.order.lmtPrice if getattr(tr.contract, "secType", "") == "BAG"
                            and tr.order.lmtPrice >= x else None)


# ==========================================================================
# the ladder
# ==========================================================================
class TestCloseLadder:
    def test_fills_at_rung_0(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.matcher = _fill_when_limit_at_least(2.0)
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["outcome"] == "filled" and res["filled_rung"] == 0
        assert len(world.placed) == 1
        assert state["trials"]["T"]["close"]["status"] == "filled"
        assert state["trials"]["T"]["close"]["rungs"][0]["status"] == "filled"

    def test_fills_at_rung_2_with_a_new_order_per_rung_each_cancelled_first(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.matcher = _fill_when_limit_at_least(2.70)
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["filled_rung"] == 2 and len(world.placed) == 3
        ids = [tr.order.orderId for tr in world.placed]
        assert len(set(ids)) == 3
        assert [tr.orderStatus.status for tr in world.placed[:2]] == ["Cancelled", "Cancelled"]
        refs = [tr.order.orderRef for tr in world.placed]
        assert all(r.startswith("W101PROBE-T-close-r") for r in refs) and len(set(refs)) == 3

    def test_never_fills_walks_all_seven_rungs_and_ends_at_the_width(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["outcome"] == "unfilled" and len(world.placed) == 7
        limits = [tr.order.lmtPrice for tr in world.placed]
        assert all(b > a for a, b in zip(limits, limits[1:])) and limits[-1] == 3.0
        assert all(tr.orderStatus.status == "Cancelled" for tr in world.placed)
        assert state["trials"]["T"]["close"]["status"] == "aborted"

    def test_each_rung_rests_the_configured_time(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        t0 = world.t
        asyncio.run(_close(world, ibkr, state, t, rest=120.0))
        births = [tr._born for tr in world.placed]
        gaps = [b - a for a, b in zip(births, births[1:])]
        assert all(120 <= g <= 135 for g in gaps), gaps          # rest + the confirmed cancel
        assert world.t - t0 >= 6 * 120

    def test_the_floor_lifts_rung_0_when_the_structure_is_in_the_money(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t, natural=1.0)
        world.set_spot(602.0)                                       # the short put is ITM by 1
        world.matcher = _fill_when_limit_at_least(0.0)
        asyncio.run(_close(world, ibkr, state, t))
        assert world.placed[0].order.lmtPrice == pytest.approx(1.10)    # intrinsic 1.0 + cross, above natural + cross

    def test_a_cancel_that_loses_to_a_fill_ends_the_ladder(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.cancel_mode = "race_fill"
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["outcome"] == "filled" and len(world.placed) == 1

    def test_a_local_cancelled_with_an_execution_is_classified_filled(self, world):
        """ib_insync flips the order to Cancelled on error 10148, but the fill is real."""
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.cancel_mode = "local_cancel_then_exec"
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["outcome"] == "cancel_race_filled" and len(world.placed) == 1

    def test_a_cancel_that_never_confirms_stops_everything(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.cancel_mode = "pending_forever"
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["outcome"] == "cancel_unconfirmed" and len(world.placed) == 1

    def test_inactive_stops_the_ladder(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.inactive = True
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["outcome"] == "inactive" and len(world.placed) == 1

    def test_a_rejected_order_is_recorded_with_its_code(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.reject_code = 201
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["outcome"] == "rejected" and len(world.placed) == 1
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        assert any(e["event"] == "order_error" and e["code"] == 201 for e in events)

    def test_a_ticking_bag_is_recorded_but_pricing_stays_leg_based(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.set_bag(2.30, 2.50)
        world.matcher = _fill_when_limit_at_least(0.0)
        asyncio.run(_close(world, ibkr, state, t))
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        assert [e["ticked"] for e in events if e["event"] == "bag_probe"] == [True]
        assert world.placed[0].order.lmtPrice == pytest.approx(2.37 + 0.10, abs=0.011)     # natural 2.37 + cross

    def test_a_buy_leg_without_an_ask_places_nothing(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        sp = next(l for l in t["legs"] if l["close_action"] == "BUY" and l["right"] == "P")
        world.set_leg(sp["conId"], 2.0, float("nan"))
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["outcome"] == "aborted_no_quotes" and world.placed == []

    def test_the_state_is_written_before_the_order_is_placed(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.raise_on_place = True
        with pytest.raises(RuntimeError):
            asyncio.run(_close(world, ibkr, state, t))
        saved = probe.load_state()["trials"]["T"]["close"]
        assert saved["status"] == "working" and saved["rungs"][0]["status"] == "placing"

    def test_no_limit_above_the_wing_width_is_ever_placed(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t, natural=3.4)                          # natural above the width: cap binds
        res = asyncio.run(_close(world, ibkr, state, t))
        assert res["outcome"] == "unfilled"
        assert all(tr.order.lmtPrice <= 3.0 + 1e-9 for tr in world.placed)
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        assert any(e["event"] == "rung_plan" and e.get("cap_binds") for e in events)

    def test_the_hard_cap_is_enforced_by_the_ladder_itself(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)

        async def too_high(k, ctx):
            return 3.5, {"k": k}

        async def go():
            ev = _ev()
            return await probe.run_combo_ladder(
                ibkr, state=state, tstate=t, trial="T", step="close", bag=probe.bag_for([]),
                next_limit=too_high, n_rungs=2, rest_s=1, ev=ev, errors=probe.ErrorRecorder(world, ev),
                hard_cap=3.0)

        with pytest.raises(probe.ProbeGuardError, match="exceeds the wing width"):
            asyncio.run(go())
        assert world.placed == []


# ==========================================================================
# structure selection and the opening maths (pure)
# ==========================================================================
def _grid(lo=560, hi=640, step=1.0):
    n = int((hi - lo) / step) + 1
    return [lo + i * step for i in range(n)]


class TestSelectStructure:
    TODAY = datetime(2026, 10, 27).date()
    EXPS = ["20261027", "20261028", "20261030", "20261102", "20261120"]

    def test_far_is_the_nearest_expiry_three_to_seven_days_out_with_shorts_1_5_percent_out(self):
        s = probe.select_structure("far", 600.0, self.EXPS, _grid(), self.TODAY)
        assert (s["expiry"], s["dte"], s["width"]) == ("20261030", 3, 3.0)
        by = {(l["right"], l["open_action"]): l["strike"] for l in s["legs"]}
        assert by == {("P", "SELL"): 591.0, ("P", "BUY"): 588.0, ("C", "SELL"): 609.0, ("C", "BUY"): 612.0}
        probe.guard_caps([{**l, "action": l["open_action"]} for l in s["legs"]])

    def test_pin_is_zero_dte_with_a_short_put_at_or_just_above_the_money(self):
        s = probe.select_structure("pin", 600.4, self.EXPS, _grid(), self.TODAY)
        by = {(l["right"], l["open_action"]): l["strike"] for l in s["legs"]}
        assert s["dte"] == 0 and s["expiry"] == "20261027"
        assert by == {("P", "SELL"): 601.0, ("P", "BUY"): 598.0, ("C", "SELL"): 604.0, ("C", "BUY"): 607.0}
        assert 601.0 - 600.4 <= 1.0          # ITM by < $1: the natural stays below the width

    def test_pin_needs_an_expiry_today(self):
        with pytest.raises(probe.ProbeGuardError, match="0DTE"):
            probe.select_structure("pin", 600.4, self.EXPS[1:], _grid(), self.TODAY)

    def test_far_needs_an_expiry_in_the_window(self):
        with pytest.raises(probe.ProbeGuardError, match="3-7"):
            probe.select_structure("far", 600.0, ["20261027", "20261028", "20261120"], _grid(), self.TODAY)

    def test_an_unlisted_wing_is_refused(self):
        strikes = [s for s in _grid() if s != 588.0]
        with pytest.raises(probe.ProbeGuardError, match="not listed"):
            probe.select_structure("far", 600.0, self.EXPS, strikes, self.TODAY)

    def test_unknown_kind(self):
        with pytest.raises(probe.ProbeGuardError, match="unknown kind"):
            probe.select_structure("wide", 600.0, self.EXPS, _grid(), self.TODAY)


class TestOpeningMaths:
    LEGS = [{"conId": 1, "action": "BUY"}, {"conId": 2, "action": "SELL"},
            {"conId": 3, "action": "SELL"}, {"conId": 4, "action": "BUY"}]
    Q = {1: (0.30, 0.33), 2: (0.70, 0.73), 3: (0.60, 0.63), 4: (0.25, 0.28)}

    def test_natural_and_mid_are_signed_credits(self):
        assert probe.open_net(self.LEGS, self.Q, "natural") == pytest.approx(0.33 + 0.28 - 0.70 - 0.60)
        assert probe.open_net(self.LEGS, self.Q, "mid") == pytest.approx(0.315 + 0.265 - 0.715 - 0.615)

    def test_a_bought_leg_with_no_bid_has_its_offer_as_its_mid(self):
        q = {**self.Q, 4: (-1.0, 0.28)}
        assert probe.open_net(self.LEGS, q, "mid") == pytest.approx(0.315 + 0.28 - 0.715 - 0.615)
        assert probe.open_net(self.LEGS, q, "natural") == pytest.approx(0.33 + 0.28 - 0.70 - 0.60)

    def test_a_missing_side_is_none(self):
        assert probe.open_net(self.LEGS, {**self.Q, 1: (0.3, None)}, "natural") is None
        assert probe.open_net(self.LEGS, {**self.Q, 2: (None, 0.73)}, "natural") is None
        assert probe.open_net(self.LEGS, {**self.Q, 2: (None, 0.73)}, "mid") is None

    def test_two_sided(self):
        assert probe.two_sided(self.LEGS, self.Q)
        assert not probe.two_sided(self.LEGS, {**self.Q, 4: (-1.0, 0.28)})


# ==========================================================================
# the broker-edge helpers
# ==========================================================================
class TestErrorRecorder:
    def test_records_codes_the_client_ignores_and_skips_farm_chatter(self, world):
        ev = _ev()
        rec = probe.ErrorRecorder(world, ev)
        world.errorEvent.emit(101, 10148, "cannot cancel", None)
        world.errorEvent.emit(-1, 2104, "farm ok", None)
        world.errorEvent.emit(102, 161, "x", SimpleNamespace(conId=77))
        assert [i["code"] for i in rec.items] == [10148, 161]
        assert rec.for_order(102)[0]["con_id"] == 77
        rec.detach()
        world.errorEvent.emit(103, 201, "after detach", None)
        assert len(rec.items) == 2


class TestQuoteSampler:
    def test_subscribes_reads_and_always_cancels(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.set_spot(607.0)
        contracts = {l["conId"]: SimpleNamespace(conId=l["conId"], secType="OPT") for l in t["legs"]}
        cancelled = []
        world.cancelMktData = lambda c: cancelled.append(world._key(c))
        s = probe.QuoteSampler(ibkr, _ev(), contracts, SimpleNamespace(conId=756733, secType="STK"),
                               SimpleNamespace(secType="BAG"))
        s.start()
        assert s.spot() == 607.0 and s.bag_ticked() is False
        world.set_bag(1.0, 1.2)
        assert s.bag_ticked() is True
        assert set(s.quotes()) == set(contracts)
        s.stop()
        assert len(cancelled) == 6 and s._tickers == {}

    def test_spot_falls_back_to_the_mid(self, world):
        ibkr = FakeIBKR(world)
        s = probe.QuoteSampler(ibkr, _ev(), {}, SimpleNamespace(conId=1, secType="STK"), None)
        s.start()
        tk = world.tickers["SPOT"]
        tk.bid, tk.ask = 606.0, 608.0
        assert s.spot() == 607.0
        tk.bid = tk.ask = float("nan")
        assert s.spot() is None


# ==========================================================================
# commands (the broker is faked; every guard that is not under test is stubbed)
# ==========================================================================
@pytest.fixture
def cmds(world, monkeypatch):
    ibkr = FakeIBKR(world)

    async def _connect(data_type=3):
        return ibkr

    monkeypatch.setattr(probe, "connect_probe", _connect)
    monkeypatch.setattr(probe, "guard_window", lambda *a, **k: None)
    monkeypatch.setattr(probe, "guard_bot_stopped", lambda *a, **k: None)
    world.ibkr = ibkr
    return world


def _args(**kw):
    base = dict(trial="T", kind="far", data_type=3, rest=None, no_truth_tail=True, legout_only=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _chain(world, spot=600.4, exps=("20261027", "20261030"), strikes=None):
    stock = SimpleNamespace(conId=756733, secType="STK", symbol="SPY")
    chain = SimpleNamespace(tradingClass="SPY", exchange="SMART", expirations=list(exps),
                            strikes=strikes or _grid())
    world.chain_result = [chain]
    return stock, spot


class TestCmdOpen:
    OPEN_Q = {588.0: (0.30, 0.33), 591.0: (0.70, 0.73), 609.0: (0.60, 0.63), 612.0: (0.25, 0.28)}

    def _setup(self, cmds, monkeypatch, *, today=datetime(2026, 10, 27, 11, 0, tzinfo=ET)):
        cmds.leg_defaults = {(588.0, "P"): self.OPEN_Q[588.0], (591.0, "P"): self.OPEN_Q[591.0],
                             (609.0, "C"): self.OPEN_Q[609.0], (612.0, "C"): self.OPEN_Q[612.0]}
        monkeypatch.setattr(probe, "now_et", lambda: today)
        stock, spot = _chain(cmds, 600.0)
        async def _fc(ibkr):
            return stock, spot, sorted(cmds.chain_result[0].expirations), cmds.chain_result[0].strikes
        monkeypatch.setattr(probe, "fetch_chain", _fc)

    def test_a_far_condor_opens_and_the_trial_is_recorded(self, cmds, monkeypatch):
        self._setup(cmds, monkeypatch)
        cmds.matcher = lambda tr, age: tr.order.lmtPrice if tr.order.lmtPrice >= -0.72 else None
        out = asyncio.run(probe.cmd_open(_args()))
        assert out["outcome"] == "filled" and out["filled_rung"] is not None
        st = probe.load_state()["trials"]["T"]
        assert st["open"]["status"] == "filled" and st["kind"] == "far" and st["dte"] == 3
        assert {l["strike"] for l in st["legs"]} == {588.0, 591.0, 609.0, 612.0}
        assert all(l["conId"] for l in st["legs"])
        # a credit: every entry limit is negative, walking from the mid toward and past the natural
        limits = [tr.order.lmtPrice for tr in cmds.placed]
        assert all(l < 0 for l in limits) and limits == sorted(limits)
        assert sum(cmds.positions.values()) == 0                    # +1 +1 -1 -1
        assert sorted(cmds.positions.values()) == [-1.0, -1.0, 1.0, 1.0]

    def test_the_entry_ladder_never_fills_and_the_open_is_reported_failed(self, cmds, monkeypatch):
        self._setup(cmds, monkeypatch)
        with pytest.raises(probe.ProbeGuardError, match="open ended 'unfilled'"):
            asyncio.run(probe.cmd_open(_args()))
        assert len(cmds.placed) == 5 and not cmds.positions
        assert probe.load_state()["trials"]["T"]["open"]["status"] == "aborted"

    def test_a_credit_below_the_floor_places_nothing(self, cmds, monkeypatch):
        self._setup(cmds, monkeypatch)
        cmds.leg_defaults[(591.0, "P")] = (0.31, 0.34)
        cmds.leg_defaults[(609.0, "C")] = (0.27, 0.30)
        with pytest.raises(probe.ProbeGuardError, match="credit_too_small"):
            asyncio.run(probe.cmd_open(_args()))
        assert cmds.placed == []

    def test_a_foreign_position_on_a_contract_the_trial_would_trade_refuses_before_any_order(self, cmds, monkeypatch):
        self._setup(cmds, monkeypatch)
        cmds._con_ids[(588.0, "P", "20261030")] = 5            # the long put the trial will qualify gets conId 5
        cmds.contracts[5] = SimpleNamespace(conId=5, symbol="SPY", secType="OPT", strike=588.0, right="P",
                                            lastTradeDateOrContractMonth="20261030")
        cmds.positions[5] = 1.0
        with pytest.raises(probe.ProbeGuardError, match="does not own"):
            asyncio.run(probe.cmd_open(_args()))
        assert cmds.placed == []

    def test_the_operators_own_positions_elsewhere_do_not_block_an_open(self, cmds, monkeypatch):
        self._setup(cmds, monkeypatch)
        cmds.contracts[790] = SimpleNamespace(conId=790, symbol="QQQ", secType="OPT", strike=790.0, right="C",
                                              lastTradeDateOrContractMonth="20261030")
        cmds.positions[790] = 1.0
        cmds.matcher = lambda tr, age: tr.order.lmtPrice if tr.order.lmtPrice >= -0.72 else None
        out = asyncio.run(probe.cmd_open(_args()))
        assert out["outcome"] == "filled" and cmds.positions[790] == 1.0

    def test_an_existing_trial_name_is_refused(self, cmds, monkeypatch):
        self._setup(cmds, monkeypatch)
        probe.save_state({"schema": 1, "trials": {"T": {}}})
        with pytest.raises(probe.ProbeGuardError, match="already exists"):
            asyncio.run(probe.cmd_open(_args()))

    def test_legs_that_do_not_quote_two_sided_are_not_opened(self, cmds, monkeypatch):
        self._setup(cmds, monkeypatch)
        cmds.leg_defaults[(588.0, "P")] = (float("nan"), float("nan"))
        with pytest.raises(probe.ProbeGuardError, match="did not all quote"):
            asyncio.run(probe.cmd_open(_args()))
        assert cmds.placed == []

    def test_a_far_wing_we_buy_may_have_no_bid(self, cmds, monkeypatch):
        """The nobid trial: the long wing has an offer but nobody bids. Opening needs only its ask."""
        self._setup(cmds, monkeypatch)
        cmds.leg_defaults[(612.0, "C")] = (-1.0, 0.28)
        cmds.matcher = lambda tr, age: tr.order.lmtPrice if tr.order.lmtPrice >= -0.72 else None
        out = asyncio.run(probe.cmd_open(_args()))
        assert out["outcome"] == "filled"

    def test_a_far_wing_we_sell_still_needs_a_bid(self, cmds, monkeypatch):
        self._setup(cmds, monkeypatch)
        cmds.leg_defaults[(609.0, "C")] = (-1.0, 0.63)
        with pytest.raises(probe.ProbeGuardError, match="did not all quote"):
            asyncio.run(probe.cmd_open(_args()))
        assert cmds.placed == []

    def test_a_small_account_is_refused(self, cmds, monkeypatch):
        self._setup(cmds, monkeypatch)

        async def small(self_):
            return {"NetLiquidation": "9000", "ExcessLiquidity": "9000"}

        monkeypatch.setattr(FakeIBKR, "get_account_values", small)
        with pytest.raises(probe.ProbeGuardError, match="PDT"):
            asyncio.run(probe.cmd_open(_args()))
        assert cmds.placed == []


class TestCmdClose:
    def test_runs_the_experiment_and_records_the_positions_after(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        cmds.matcher = _fill_when_limit_at_least(2.60)
        out = asyncio.run(probe.cmd_close(_args()))
        assert out["outcome"] == "filled" and out["filled_rung"] == 1
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        ends = [e for e in events if e["event"] == "trial_end" and e.get("step") == "close_positions"]
        assert ends and all(v == 0 for v in ends[0]["positions"].values())

    def test_refuses_when_the_brokers_positions_do_not_match_the_trial(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        cmds.positions[t["legs"][0]["conId"]] = 0.0
        with pytest.raises(probe.ProbeGuardError, match="do not match"):
            asyncio.run(probe.cmd_close(_args()))
        assert cmds.placed == []

    def test_refuses_an_unopened_or_already_closed_trial(self, cmds):
        probe.save_state({"schema": 1, "trials": {"T": {"open": {"status": "aborted"}}}})
        with pytest.raises(probe.ProbeGuardError, match="no filled 'open'"):
            asyncio.run(probe.cmd_close(_args()))
        state, t = _legs_state(cmds, cmds.ibkr)
        t["close"] = {"status": "filled"}
        probe.save_state(state)
        with pytest.raises(probe.ProbeGuardError, match="already closed"):
            asyncio.run(probe.cmd_close(_args()))

    def test_the_truth_tail_keeps_sampling_after_the_ladder(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        cmds.matcher = _fill_when_limit_at_least(0.0)
        t0 = cmds.t
        asyncio.run(probe.cmd_close(_args(no_truth_tail=False)))
        assert cmds.t - t0 >= probe.TRUTH_TAIL_S


def _opt_matcher(world, *, fill_bag_at=None, fill_opt=True, never_conids=()):
    """BAG fills when its limit >= fill_bag_at; single-option orders fill when marketable."""
    def m(tr, age):
        if getattr(tr.contract, "secType", "") == "BAG":
            return tr.order.lmtPrice if fill_bag_at is not None and tr.order.lmtPrice >= fill_bag_at else None
        if not fill_opt or tr.contract.conId in never_conids:
            return None
        t = world.ticker(tr.contract)
        if tr.order.action == "BUY" and tr.order.lmtPrice >= t.ask:
            return tr.order.lmtPrice
        if tr.order.action == "SELL" and tr.order.lmtPrice <= max(t.bid, 0.0):
            return tr.order.lmtPrice
        return None
    return m


def _quote_every_leg(world, tstate):
    _quote_legs(world, tstate)
    for l in tstate["legs"]:
        if l["close_action"] == "SELL":
            world.set_leg(l["conId"], 0.20, 0.25)
    for l in tstate["legs"]:
        world.tickers[l["conId"]].marketDataType = 3


class TestFlatten:
    def test_the_combo_ladder_is_tried_first_and_a_fill_ends_it(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        cmds.matcher = _opt_matcher(cmds, fill_bag_at=2.0)
        out = asyncio.run(probe.cmd_flatten(_args()))
        assert out["status"] == "flattened_by_combo"
        assert [p[0] for p in cmds.place_log] == ["BAG"] and all(v == 0 for v in cmds.positions.values())
        assert probe.load_state()["trials"]["T"]["flatten"]["status"] == "filled"

    def test_leg_out_buys_the_shorts_back_first_and_sells_a_long_only_when_both_shorts_are_flat(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_every_leg(cmds, t)
        cmds.matcher = _opt_matcher(cmds, fill_bag_at=None)
        out = asyncio.run(probe.cmd_flatten(_args()))
        assert out["status"] == "flattened_by_legout"
        opt = [(a, c, pos) for (sec, a, c, pos) in cmds.place_log if sec == "OPT"]
        assert [a for a, _c, _p in opt] == ["BUY", "BUY", "SELL", "SELL"]
        shorts = {l["conId"] for l in t["legs"] if l["open_action"] == "SELL"}
        for action, con_id, positions_then in opt:
            if action == "SELL":
                assert all(positions_then.get(s, 0) == 0 for s in shorts), "a long was sold with a short still open"
            else:
                assert positions_then[con_id] < 0
        assert all(v == 0 for v in cmds.positions.values())
        assert probe.load_state()["trials"]["T"]["legout"]["status"] == "filled"

    def test_the_riskier_short_goes_first(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_every_leg(cmds, t)
        cmds.set_spot(603.5)                                         # the short put (603) is the tested side
        cmds.matcher = _opt_matcher(cmds, fill_bag_at=None)
        asyncio.run(probe.cmd_flatten(_args(legout_only=True)))
        first = [c for (sec, a, c, _p) in cmds.place_log if sec == "OPT"][0]
        put = next(l["conId"] for l in t["legs"] if l["right"] == "P" and l["open_action"] == "SELL")
        assert first == put

    def test_a_short_that_will_not_come_back_means_nothing_is_sold(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_every_leg(cmds, t)
        stuck = next(l["conId"] for l in t["legs"] if l["right"] == "C" and l["open_action"] == "SELL")
        cmds.matcher = _opt_matcher(cmds, fill_bag_at=None, never_conids=(stuck,))
        with pytest.raises(probe.ProbeGuardError, match="SHORT STILL OPEN"):
            asyncio.run(probe.cmd_flatten(_args(legout_only=True)))
        assert not any(a == "SELL" for (sec, a, c, p) in cmds.place_log if sec == "OPT")
        assert probe.load_state()["trials"]["T"]["legout"]["status"] == "short_still_open"

    def test_refused_while_a_probe_order_is_working(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_every_leg(cmds, t)
        o = SimpleNamespace(orderId=0, totalQuantity=1, lmtPrice=1.0, action="BUY", orderRef="W101PROBE-X-close-r0-1")
        cmds.placeOrder(SimpleNamespace(secType="BAG", conId=0), o)
        cmds.allow_many = True
        with pytest.raises(probe.ProbeGuardError, match="still working"):
            asyncio.run(probe.cmd_flatten(_args()))

    def test_a_flat_book_is_a_noop(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        for c in list(cmds.positions):
            cmds.positions[c] = 0.0
        out = asyncio.run(probe.cmd_flatten(_args()))
        assert out["status"] == "already_flat" and cmds.placed == []
        assert asyncio.run(probe.cmd_flatten(_args())) == {"status": "already_flat"}

    def test_nothing_to_flatten_without_a_filled_open(self, cmds):
        probe.save_state({"schema": 1, "trials": {"T": {"open": {"status": "aborted"}}}})
        assert asyncio.run(probe.cmd_flatten(_args())) == {"status": "nothing_to_flatten"}

    def test_positions_come_from_the_broker_not_the_state(self, cmds):
        """The state says four legs are open; the broker says one short is already gone."""
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_every_leg(cmds, t)
        gone = next(l["conId"] for l in t["legs"] if l["right"] == "C" and l["open_action"] == "SELL")
        cmds.positions[gone] = 0.0
        cmds.matcher = _opt_matcher(cmds, fill_bag_at=None)
        asyncio.run(probe.cmd_flatten(_args(legout_only=True)))
        assert gone not in [c for (sec, a, c, p) in cmds.place_log if sec == "OPT"]

    def test_reduce_only(self):
        probe.assert_reduce_only("BUY", -1.0)
        probe.assert_reduce_only("SELL", 1.0)
        for action, pos in (("BUY", 0.0), ("BUY", 1.0), ("SELL", 0.0), ("SELL", -1.0)):
            with pytest.raises(probe.ProbeGuardError):
                probe.assert_reduce_only(action, pos)


class TestCancelAndStatus:
    def test_cancel_touches_only_tagged_orders(self, cmds):
        cmds.allow_many = cmds.allow_foreign = True
        mine = SimpleNamespace(orderId=0, totalQuantity=1, lmtPrice=1.0, action="BUY", orderRef="W101PROBE-F1-close-r0-1")
        bots = SimpleNamespace(orderId=0, totalQuantity=1, lmtPrice=1.0, action="BUY", orderRef="T-20261027-123")
        none = SimpleNamespace(orderId=0, totalQuantity=1, lmtPrice=1.0, action="BUY", orderRef="")
        trades = [cmds.placeOrder(SimpleNamespace(secType="BAG", conId=0), o) for o in (mine, bots, none)]
        out = asyncio.run(probe.cmd_cancel(_args()))
        assert list(out["cancelled"]) == [str(trades[0].order.orderId)]
        assert trades[0].orderStatus.status == "Cancelled"
        assert trades[1].orderStatus.status != "Cancelled" and trades[2].orderStatus.status != "Cancelled"

    def test_status_reports_ready_only_when_everything_is_clean(self, cmds):
        out = asyncio.run(probe.cmd_status(_args()))
        assert out["ready_to_restart_the_bot"] is True
        state, t = _legs_state(cmds, cmds.ibkr)
        out = asyncio.run(probe.cmd_status(_args()))
        c = out["checklist"]
        assert out["ready_to_restart_the_bot"] is False
        assert c["no_spy_option_positions"] is False and c["no_trial_open_or_working"] is False
        assert out["unfinished_trials"] == ["T"]

    def test_status_sees_spy_keys_in_halt_untracked(self, cmds):
        probe.HALT_UNTRACKED_PATH.write_text("SPY:600.0:P:20261030\nQQQ:727.0:C:20261023\n")
        out = asyncio.run(probe.cmd_status(_args()))
        assert out["halt_untracked_probe_keys"] == ["QQQ:727.0:C:20261023", "SPY:600.0:P:20261030"]
        assert out["checklist"]["no_spy_keys_in_halt_untracked"] is False


class TestPreflight:
    def test_places_nothing_and_reports_the_structures(self, cmds, monkeypatch):
        cmds.leg_defaults = {(588.0, "P"): (0.30, 0.33), (591.0, "P"): (0.70, 0.73),
                             (609.0, "C"): (0.60, 0.63), (612.0, "C"): (0.25, 0.28)}
        monkeypatch.setattr(probe, "now_et", lambda: datetime(2026, 10, 27, 11, 0, tzinfo=ET))
        stock, spot = _chain(cmds, 600.0)

        async def _fc(ibkr):
            return stock, spot, sorted(cmds.chain_result[0].expirations), cmds.chain_result[0].strikes

        monkeypatch.setattr(probe, "fetch_chain", _fc)
        out = asyncio.run(probe.cmd_preflight(_args()))
        assert cmds.placed == []
        far = out["structures"]["far"]
        assert far["selected"] and far["bag_ticks"] is False and far["two_sided"] is True
        assert far["close_natural"] == pytest.approx(0.73 + 0.63 - 0.30 - 0.25)
        assert far["planned_close_ladder"][-1] == 3.0
        assert out["structures"]["pin"]["selected"] is False or out["structures"]["pin"]["selected"] is True


# ==========================================================================
# the report: verdicts from synthetic evidence (never connects)
# ==========================================================================
from datetime import timedelta  # noqa: E402

LEGS_EV = [{"conId": 1, "close_action": "SELL"}, {"conId": 2, "close_action": "BUY"},
           {"conId": 3, "close_action": "BUY"}, {"conId": 4, "close_action": "SELL"}]
BASE = datetime(2026, 10, 27, 10, 0, tzinfo=ET)
LADDER = [2.48, 2.60, 2.73, 2.85, 2.89, 2.93, 3.0]


def _t(sec):
    return (BASE + timedelta(seconds=sec)).isoformat()


def _write(events):
    d = probe.PROBE_DIR / "2026-10-27"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "events.jsonl").open("a") as fh:
        for e in events:
            fh.write(json.dumps({"schema": 1, "mono": 0, **e}) + "\n")


def _open_event(trial="T"):
    return {"ts": _t(0), "run_id": "o", "cmd": "open", "trial": trial, "event": "legs_selected",
            "legs": LEGS_EV, "width": 3.0}


def _run(trial="T", run_id="r1", cmd="close", *, limits=LADDER, natural=2.38, fill_k=None, bag_ticked=False,
         true_nat=lambda k: 2.40, outcomes=None, cap_binds=False, width=3.0, samples=True, start=100):
    """One close run: a plan/placed/end per rung, and a quote sample 900 s after each placement."""
    ev = []
    base = {"run_id": run_id, "cmd": cmd, "trial": trial}
    ev.append({**base, "ts": _t(start - 5), "event": "bag_probe", "ticked": bag_ticked, "bid": None, "ask": None})
    for k, lim in enumerate(limits):
        placed = start + k * 130
        ev.append({**base, "ts": _t(placed - 1), "event": "rung_plan", "k": k, "limit": lim, "natural": natural,
                   "mid": natural - 0.05, "floor": None, "cap_binds": cap_binds, "floor_bound": False, "width": width})
        ev.append({**base, "ts": _t(placed), "event": "order_placed", "k": k, "order_id": 100 + k, "limit": lim})
        out = (outcomes or {}).get(k) or ("filled" if k == fill_k else "rested_out")
        ev.append({**base, "ts": _t(placed + (40 if out == "filled" else 120)), "event": "rung_end", "k": k,
                   "outcome": out, "t_to_fill_s": 40 if out == "filled" else None,
                   "avg_fill_price": round(lim - 0.02, 2) if out == "filled" else None})
        if samples:
            tn = true_nat(k)
            ev.append({**base, "ts": _t(placed + probe.DELAY_S), "event": "quote_sample", "phase": "ladder",
                       "legs": {"1": {"bid": 0.0, "ask": 0.02}, "2": {"bid": tn - 0.07, "ask": round(tn - 0.01, 2)},
                                "3": {"bid": 0.0, "ask": 0.01}, "4": {"bid": 0.0, "ask": 0.02}},
                       "bag": {"bid": None, "ask": None}})
        if out == "filled":
            break
    return ev


def _report():
    return asyncio.run(probe.cmd_report(_args()))


class TestReport:
    def test_filled_by_rung_3_is_a(self):
        _write([_open_event()] + _run(fill_k=2))
        r = _report()
        assert r["verdict"] == "BOT_LADDER_SUFFICIENT" and r["runs"][0]["class"] == "A"
        run = r["runs"][0]
        assert run["filled_rung"] == 2 and run["fill_price"] == pytest.approx(2.71)
        assert run["concession_vs_first_natural"] == pytest.approx(2.71 - 2.38)
        assert run["concession_vs_first_mid"] == pytest.approx(2.71 - 2.33)
        assert run["t_from_start_s"] == pytest.approx(2 * 130 + 40, abs=1)

    def test_filled_at_rung_5_is_b(self):
        _write([_open_event()] + _run(fill_k=5))
        assert _report()["verdict"] == "NEEDS_DEEPER_LADDER"

    def test_two_unfilled_marketable_runs_with_a_silent_bag_are_c(self):
        _write([_open_event()] + _run(run_id="r1") + _run(run_id="r2", start=20000))
        r = _report()
        assert r["verdict"] == "NO_MATCH_WITHOUT_COMBO_QUOTE" and r["counts"]["C"] == 2
        assert all(run["any_truly_marketable"] is True for run in r["runs"])

    def test_one_such_run_is_not_enough(self):
        _write([_open_event()] + _run())
        r = _report()
        assert r["runs"][0]["class"] == "C" and r["verdict"] == "INCONCLUSIVE"

    def test_limits_that_were_never_marketable_are_c_prime(self):
        _write([_open_event()] + _run(true_nat=lambda k: 3.5))
        r = _report()
        assert r["runs"][0]["class"] == "C'" and r["verdict"] == "LIMITS_NEVER_MARKETABLE"

    def test_a_ticking_bag_does_not_demote_a_run_it_only_renames_the_verdict(self):
        """The BAG is recorded but is not the question: an unfilled, marketable run is C either way;
        if the combo was quoted the verdict says so."""
        _write([_open_event()] + _run(bag_ticked=True) + _run(run_id="r2", bag_ticked=True, start=20000))
        r = _report()
        assert [x["class"] for x in r["runs"]] == ["C", "C"]
        assert r["verdict"] == "NO_MATCH_AT_MARKETABLE_PRICE" and r["runs"][0]["bag_ticked"] is True

    @pytest.mark.parametrize("kw", [dict(cap_binds=True), dict(outcomes={0: "inactive"}), dict(samples=False),
                                    dict(limits=LADDER[:5])])
    def test_runs_that_cannot_be_read_are_inconclusive(self, kw):
        _write([_open_event()] + _run(**kw))
        assert _report()["runs"][0]["class"] == "D"

    def test_an_unpriceable_rung_is_inconclusive(self):
        ev = _run()
        for e in ev:
            if e["event"] == "rung_plan" and e["k"] == 3:
                e["natural"] = None
        _write([_open_event()] + ev)
        assert _report()["runs"][0]["class"] == "D"

    def test_two_cs_outvote_a_fill_elsewhere_but_the_mix_is_flagged(self):
        _write([_open_event()] + _run(run_id="a") + _run(run_id="b", start=20000) + _run(run_id="c", fill_k=1, start=40000))
        r = _report()
        assert r["verdict"] == "NO_MATCH_WITHOUT_COMBO_QUOTE" and r["mixed"] is True

    def test_truth_tail_marketability_is_per_rung(self):
        _write([_open_event()] + _run(true_nat=lambda k: 2.9))         # only rungs >= 2.9 are marketable
        run = _report()["runs"][0]
        flags = [r["truly_marketable"] for r in run["rungs"]]
        assert flags == [False, False, False, False, False, True, True]

    def test_leg_out_outcomes(self):
        _write([_open_event()] + _run() + [{"ts": _t(9000), "run_id": "f", "cmd": "flatten", "trial": "T",
                                            "event": "trial_end", "step": "legout", "outcome": "filled"}])
        assert _report()["leg_out"] == "LEG_OUT_VIABLE"
        _write([{"ts": _t(9100), "run_id": "g", "cmd": "flatten", "trial": "U", "event": "trial_end",
                 "step": "legout", "outcome": "short_still_open"}])
        assert probe.leg_out_verdict(probe.load_events()) == "LEG_OUT_VIABLE"        # one viable run is enough
        assert probe.leg_out_verdict([{"event": "trial_end", "step": "legout", "outcome": "short_still_open"}]) \
            == "LEG_OUT_FAILED"
        assert probe.leg_out_verdict([{"event": "trial_end", "step": "legout", "outcome": "longs_left"}]) \
            == "LEG_OUT_VIABLE"
        assert probe.leg_out_verdict([]) is None

    def test_no_evidence_is_inconclusive(self):
        assert _report()["verdict"] == "INCONCLUSIVE"

    def test_a_schema_mismatch_is_refused(self):
        d = probe.PROBE_DIR / "2026-10-27"
        d.mkdir(parents=True)
        (d / "events.jsonl").write_text(json.dumps({"schema": 2, "event": "x"}) + "\n")
        with pytest.raises(probe.ProbeGuardError, match="schema"):
            _report()

    def test_the_report_never_connects(self, monkeypatch):
        async def boom(*a, **k):
            raise AssertionError("report must not connect")

        monkeypatch.setattr(probe, "connect_probe", boom)
        _write([_open_event()] + _run(fill_k=0))
        assert _report()["verdict"] == "BOT_LADDER_SUFFICIENT"

    def test_the_flatten_runs_are_reported_but_do_not_vote(self):
        _write([_open_event()] + _run(cmd="flatten", fill_k=0))
        r = _report()
        assert r["runs"][0]["cmd"] == "flatten" and r["verdict"] == "INCONCLUSIVE"


class TestClassifyAndVerdictRules:
    BASE = dict(filled_rung=None, flags=[], cap_binds=False, priceable=True, bag_ticked=False,
                reached_width=True, any_truly_marketable=True)

    @pytest.mark.parametrize("change,want", [
        ({}, "C"), ({"any_truly_marketable": False}, "C'"), ({"any_truly_marketable": None}, "D"),
        ({"reached_width": False}, "D"), ({"bag_ticked": True}, "C"), ({"priceable": False}, "D"),
        ({"cap_binds": True}, "D"), ({"flags": ["rejected"]}, "D"), ({"filled_rung": 0}, "A"),
        ({"filled_rung": 3}, "A"), ({"filled_rung": 4}, "B"), ({"filled_rung": 6}, "B")])
    def test_classes(self, change, want):
        assert probe.classify_run({**self.BASE, **change}) == want

    @pytest.mark.parametrize("classes,want", [
        ([], "D"), (["A"], "A"), (["A", "A"], "A"), (["A", "B"], "B"), (["C"], "D"), (["C", "C"], "C"),
        (["C", "A"], "A"), (["C'"], "C'"), (["C", "C'"], "D"), (["D", "D"], "D"), (["C", "C", "A"], "C")])
    def test_overall(self, classes, want):
        assert probe.verdict(classes)["code"] == want

    def test_a_lone_c_beside_a_fill_is_reported_but_flagged_as_mixed(self):
        v = probe.verdict(["C", "A"])
        assert v["code"] == "A" and v["mixed"] is True


# ==========================================================================
# the sampler loop, the parser and main()
# ==========================================================================
class TestSamplerLoop:
    def test_emits_a_sample_per_interval_until_stopped(self, world, monkeypatch):
        ibkr = FakeIBKR(world)
        s = probe.QuoteSampler(ibkr, _ev("close"), {}, SimpleNamespace(conId=1, secType="STK"), None)
        s.start()
        world.set_spot(607.0)
        slept = []

        async def fake_sleep(sec):
            slept.append(sec)
            if len(slept) == 3:
                s.phase = "tail"
            if len(slept) >= 4:
                s._stop = True

        monkeypatch.setattr(probe, "_sleep", fake_sleep)
        asyncio.run(_REAL_SAMPLER_RUN(s))
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        samples = [e for e in events if e["event"] == "quote_sample"]
        assert [x["phase"] for x in samples] == ["ladder", "ladder", "ladder", "tail"]
        assert slept == [probe.SAMPLE_EVERY_S] * 4
        assert samples[0]["spot"]["last"] == 607.0


class TestStandalone:
    def test_the_script_runs_on_its_own(self, tmp_path):
        import subprocess
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "probe_w101_combo_close.py"), "--help"],
                           capture_output=True, text=True, cwd=tmp_path, timeout=120)
        assert r.returncode == 0 and "W101-P" in r.stdout and "preflight" in r.stdout


class TestMain:
    def test_help_and_parser(self):
        p = probe.build_parser()
        a = p.parse_args(["open", "--trial", "F1", "--kind", "pin", "--data-type", "4"])
        assert (a.command, a.trial, a.kind, a.data_type) == ("open", "F1", "pin", 4)
        a = p.parse_args(["close", "--trial", "F1", "--rest", "60", "--no-truth-tail"])
        assert a.rest == 60 and a.no_truth_tail is True
        a = p.parse_args(["flatten", "--trial", "F1", "--legout-only"])
        assert a.legout_only is True
        with pytest.raises(SystemExit):
            p.parse_args(["open", "--trial", "F1", "--kind", "wide"])
        with pytest.raises(SystemExit):
            p.parse_args(["open", "--kind", "far"])

    def _handlers(self, monkeypatch, raise_exc):
        cancelled = []

        async def failing(args):
            raise raise_exc

        async def cancel(args):
            cancelled.append(args.command)
            return {"cancelled": {}}

        for name in ("open", "close", "flatten", "status", "preflight"):
            monkeypatch.setitem(probe._HANDLERS, name, failing)
        monkeypatch.setattr(probe, "cmd_cancel", cancel)
        return cancelled

    def test_ctrl_c_during_close_cancels_orders_and_exits_130(self, monkeypatch):
        cancelled = self._handlers(monkeypatch, KeyboardInterrupt())
        assert probe.main(["close", "--trial", "F1"]) == 130
        assert cancelled == ["close"]

    def test_an_error_exits_1_and_cancels_but_never_flattens(self, monkeypatch):
        cancelled = self._handlers(monkeypatch, RuntimeError("boom"))
        monkeypatch.setattr(probe, "cmd_flatten", lambda *a, **k: (_ for _ in ()).throw(AssertionError("flattened")))
        assert probe.main(["open", "--trial", "F1", "--kind", "far"]) == 1
        assert cancelled == ["open"]

    def test_a_guard_refusal_is_reported_and_exits_1(self, monkeypatch, capsys):
        self._handlers(monkeypatch, probe.ProbeGuardError("no"))
        assert probe.main(["flatten", "--trial", "F1"]) == 1
        assert "GUARD REFUSED: no" in capsys.readouterr().err

    @pytest.mark.parametrize("cmd", ["status", "preflight"])
    def test_read_only_commands_do_not_cancel_anything(self, monkeypatch, cmd):
        cancelled = self._handlers(monkeypatch, RuntimeError("boom"))
        assert probe.main([cmd]) == 1 and cancelled == []

    def test_a_failing_cancel_is_reported_not_raised(self, monkeypatch, capsys):
        self._handlers(monkeypatch, RuntimeError("boom"))

        async def bad(args):
            raise RuntimeError("gateway gone")

        monkeypatch.setattr(probe, "cmd_cancel", bad)
        assert probe.main(["close", "--trial", "F1"]) == 1
        assert "cancel FAILED" in capsys.readouterr().err

    def test_report_runs_through_main(self, monkeypatch, capsys):
        assert probe.main(["report"]) == 0
        assert json.loads(capsys.readouterr().out)["verdict"] == "INCONCLUSIVE"


# ==========================================================================
# delay emulation: price from the quotes of `lag` seconds ago, match against the live market
# ==========================================================================
class TestSamplerHistory:
    def _sampler(self, world, **quotes):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        contracts = {l["conId"]: SimpleNamespace(conId=l["conId"], secType="OPT") for l in t["legs"]}
        s = probe.QuoteSampler(ibkr, _ev(), contracts, None, None)
        s.start()
        return s, t

    def test_records_spans_and_returns_the_newest_quote_that_is_old_enough(self, world):
        s, t = self._sampler(world)
        sp = next(l["conId"] for l in t["legs"] if l["close_action"] == "BUY" and l["right"] == "P")
        for i in range(10):                                   # 10 samples, 100 s apart, ask rising
            world.set_leg(sp, 1.0 + i * 0.1, 1.05 + i * 0.1)
            s.record()
            world.t += 100
        assert s.span() == pytest.approx(900)
        quotes, age = s.lagged_quotes(300)
        assert age >= 300 and quotes[sp][1] == pytest.approx(1.05 + 7 * 0.1)       # the sample 300 s back
        assert s.lagged_quotes(5000) is None

    def test_no_history_means_no_lagged_quotes(self, world):
        s, _t = self._sampler(world)
        assert s.span() == 0.0 and s.lagged_quotes(1) is None


class TestLaggedPricing:
    def test_a_rung_is_priced_from_old_quotes_but_records_the_live_natural(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t, natural=1.00)
        contracts = {l["conId"]: SimpleNamespace(conId=l["conId"], secType="OPT") for l in t["legs"]}
        s = probe.QuoteSampler(ibkr, _ev(), contracts, None, None)
        s.start()
        s.record()
        world.t += 900
        _quote_legs(world, t, natural=1.50)                    # the market moved up 0.50 meanwhile
        s.record()
        open_legs = probe.trial_legs(t, "open")
        close_legs = probe.trial_legs(t, "close")
        fn = probe.close_limit_fn(s, close_legs, open_legs, 3.0, 5, 3, _ev(), probe.load_params(), lag_s=900)
        limit, info = asyncio.run(fn(0, {"prev": None, "l3": None}))
        assert info["natural"] == pytest.approx(1.00) and info["natural_live"] == pytest.approx(1.50)
        assert limit == pytest.approx(1.10) and info["lag_s"] == 900 and info["lag_age_s"] >= 900
        # the stale limit is below the live natural: it is NOT truly marketable
        assert limit < info["natural_live"]

    def test_no_lag_prices_from_the_live_quotes(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t, natural=1.00)
        contracts = {l["conId"]: SimpleNamespace(conId=l["conId"], secType="OPT") for l in t["legs"]}
        s = probe.QuoteSampler(ibkr, _ev(), contracts, None, None)
        s.start()
        fn = probe.close_limit_fn(s, probe.trial_legs(t, "close"), probe.trial_legs(t, "open"),
                                  3.0, 5, 3, _ev(), probe.load_params())
        limit, info = asyncio.run(fn(0, {"prev": None, "l3": None}))
        assert info["natural"] == info["natural_live"] and info["lag_s"] == 0 and limit == pytest.approx(1.10)

    def test_too_little_history_aborts_the_rung(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        contracts = {l["conId"]: SimpleNamespace(conId=l["conId"], secType="OPT") for l in t["legs"]}
        s = probe.QuoteSampler(ibkr, _ev(), contracts, None, None)
        s.start()
        s.record()
        fn = probe.close_limit_fn(s, probe.trial_legs(t, "close"), probe.trial_legs(t, "open"),
                                  3.0, 5, 3, _ev(), probe.load_params(), lag_s=900)
        limit, info = asyncio.run(fn(0, {"prev": None, "l3": None}))
        assert limit is None and info["abort"] == "no_lagged_quotes"

    def test_cap_binds_considers_the_live_natural_too(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t, natural=1.0)
        contracts = {l["conId"]: SimpleNamespace(conId=l["conId"], secType="OPT") for l in t["legs"]}
        s = probe.QuoteSampler(ibkr, _ev(), contracts, None, None)
        s.start()
        s.record()
        world.t += 900
        _quote_legs(world, t, natural=3.4)
        s.record()
        fn = probe.close_limit_fn(s, probe.trial_legs(t, "close"), probe.trial_legs(t, "open"),
                                  3.0, 5, 3, _ev(), probe.load_params(), lag_s=900)
        _limit, info = asyncio.run(fn(0, {"prev": None, "l3": None}))
        assert info["cap_binds"] is True

    def test_the_last_known_spot_is_used_when_the_stock_feed_drops_out(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t, natural=1.0)
        world.set_spot(602.0)                                    # the short put (603) is in the money
        contracts = {l["conId"]: SimpleNamespace(conId=l["conId"], secType="OPT") for l in t["legs"]}
        s = probe.QuoteSampler(ibkr, _ev(), contracts, SimpleNamespace(secType="STK", conId=1), None)
        s.start()
        fn = probe.close_limit_fn(s, probe.trial_legs(t, "close"), probe.trial_legs(t, "open"),
                                  3.0, 5, 3, _ev(), probe.load_params(), spot0=None)
        _l, info = asyncio.run(fn(0, {"prev": None, "l3": None}))
        assert info["spot"] == 602.0
        world.tickers["SPOT"].last = float("nan")                # the feed drops out (competing session)
        _l, info = asyncio.run(fn(1, {"prev": 1.1, "l3": None}))
        assert info["spot"] == 602.0 and info["floor"] is not None

    def test_the_seed_spot_is_used_before_any_tick(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t, natural=1.0)
        world.tickers.pop("SPOT", None)
        contracts = {l["conId"]: SimpleNamespace(conId=l["conId"], secType="OPT") for l in t["legs"]}
        s = probe.QuoteSampler(ibkr, _ev(), contracts, None, None)
        s.start()
        fn = probe.close_limit_fn(s, probe.trial_legs(t, "close"), probe.trial_legs(t, "open"),
                                  3.0, 5, 3, _ev(), probe.load_params(), spot0=602.0)
        _l, info = asyncio.run(fn(0, {"prev": None, "l3": None}))
        assert info["spot"] == 602.0


class TestEmulatedClose:
    def test_waits_for_the_history_then_prices_from_it_and_matches_live(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t, natural=1.00)
        cmds.matcher = _fill_when_limit_at_least(0.0)
        t0 = cmds.t
        out = asyncio.run(probe.cmd_close(_args(no_truth_tail=True, emulate_delay=900.0)))
        assert out["outcome"] == "filled" and cmds.t - t0 >= 900
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        assert any(e["event"] == "emulate_delay_wait" and e["seconds"] == 900 for e in events)
        plan = next(e for e in events if e["event"] == "rung_plan" and e["cmd"] == "close")
        assert plan["lag_s"] == 900 and plan["lag_age_s"] >= 900

    def test_the_stale_limit_does_not_fill_where_the_live_natural_has_moved_up(self, cmds):
        """The incident in miniature: the market rises while the delayed quotes stay behind."""
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t, natural=1.00)
        sp = next(l["conId"] for l in t["legs"] if l["close_action"] == "BUY" and l["right"] == "P")
        real_sleep = cmds.step

        def drift():                                           # the live ask climbs 0.001/s
            cmds.tickers[sp].ask = round(cmds.tickers[sp].ask + 0.001, 4)
            real_sleep()

        cmds.step = drift
        cmds.matcher = lambda tr, age: (tr.order.lmtPrice if getattr(tr.contract, "secType", "") == "BAG"
                                        and tr.order.lmtPrice >= cmds.tickers[sp].ask else None)
        out = asyncio.run(probe.cmd_close(_args(no_truth_tail=True, emulate_delay=900.0)))
        assert out["filled_rung"] is not None and out["filled_rung"] >= 1      # rung 0 was priced 900 s behind

    def test_history_that_never_builds_aborts(self, cmds, monkeypatch):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        monkeypatch.setattr(probe.QuoteSampler, "record", lambda self: None)
        out = asyncio.run(probe.cmd_close(_args(no_truth_tail=True, emulate_delay=900.0)))
        assert out["outcome"] == "aborted_history_too_short" and cmds.placed == []

    def test_the_flag_reaches_the_parser(self):
        a = probe.build_parser().parse_args(["close", "--trial", "F1", "--emulate-delay", "900"])
        assert a.emulate_delay == 900.0
        assert probe.build_parser().parse_args(["close", "--trial", "F1"]).emulate_delay == 0.0


class TestReportFeeds:
    def _live(self, ev, natural_live):
        for e in ev:
            if e["event"] == "rung_plan":
                e["natural_live"] = natural_live(e["k"])
                e["mdt"] = {"1": 1, "2": 1, "3": 1, "4": 1}
        return ev

    def test_with_a_live_feed_the_limit_is_judged_against_the_live_natural_not_a_later_sample(self):
        ev = self._live(_run(true_nat=lambda k: 9.9), lambda k: 2.40)         # the +900 s samples would say "never marketable"
        _write([_open_event()] + ev)
        run = _report()["runs"][0]
        assert run["feed"] == "live" and run["any_truly_marketable"] is True
        assert [r["truly_marketable"] for r in run["rungs"]] == [True] * 7

    def test_an_emulated_run_is_judged_against_the_live_natural_at_each_placement(self):
        ev = _run(true_nat=lambda k: 1.0)
        for e in ev:
            if e["event"] == "rung_plan":
                e["lag_s"], e["natural_live"] = 900, 2.65
        _write([_open_event()] + ev)
        run = _report()["runs"][0]
        assert run["feed"] == "emulated" and run["emulated_delay_s"] == 900
        assert [r["truly_marketable"] for r in run["rungs"]] == [False, False, True, True, True, True, True]

    def test_a_delayed_feed_still_uses_the_later_sample(self):
        ev = _run(true_nat=lambda k: 2.9)
        for e in ev:
            if e["event"] == "rung_plan":
                e["mdt"] = {"1": 3, "2": 3, "3": 3, "4": 3}
        _write([_open_event()] + ev)
        run = _report()["runs"][0]
        assert run["feed"] == "delayed"
        assert [r["truly_marketable"] for r in run["rungs"]] == [False] * 5 + [True, True]

    def test_the_concession_against_the_live_natural(self):
        ev = self._live(_run(fill_k=1), lambda k: 2.45)
        _write([_open_event()] + ev)
        run = _report()["runs"][0]
        assert run["concession_vs_live_natural"] == pytest.approx(2.58 - 2.45)


class TestVerdictNames:
    def test_c_is_named_for_whether_the_combo_was_quoted(self):
        assert probe.verdict(["C", "C"])["verdict"] == "NO_MATCH_WITHOUT_COMBO_QUOTE"
        assert probe.verdict(["C", "C"], c_bag_ticked=True)["verdict"] == "NO_MATCH_AT_MARKETABLE_PRICE"
        assert probe.verdict(["A", "A"], c_bag_ticked=True)["verdict"] == "BOT_LADDER_SUFFICIENT"


class TestWhatIfSentinel:
    def test_ibs_unset_value_is_none_not_a_number(self, world):
        async def go():
            async def wi(bag, order):
                return SimpleNamespace(initMarginChange="462.4", maintMarginChange="431.4",
                                       commission=1.7976931348623157e308, warningText="")
            world.whatIfOrderAsync = wi
            return await probe.whatif_margin(FakeIBKR(world), SimpleNamespace(), SimpleNamespace())
        out = asyncio.run(go())
        assert out["init_margin"] == pytest.approx(462.4) and out["commission"] is None


# ==========================================================================
# the underlying from the options when the stock feed is silent
# ==========================================================================
class TestParitySpot:
    def test_k_plus_call_minus_put(self):
        assert probe.parity_spot(600.0, (6.0, 6.04), (5.0, 5.04)) == pytest.approx(601.0)

    @pytest.mark.parametrize("c,p", [
        ((6.0, None), (5.0, 5.04)), ((float("nan"), 6.04), (5.0, 5.04)), ((6.0, 6.04), (5.0, 0.0)),
        ((6.0, 7.5), (5.0, 5.04)), ((6.0, 6.04), (-1.0, 5.04))])
    def test_unusable_or_wide_quotes_give_nothing(self, c, p):
        assert probe.parity_spot(600.0, c, p) is None


class TestSpotFromOptions:
    def test_a_good_pair_at_the_guess_is_enough(self, world):
        world.leg_defaults = {(600.0, "C"): (6.0, 6.04), (600.0, "P"): (5.0, 5.04)}
        out = asyncio.run(probe.estimate_spot_from_options(
            FakeIBKR(world), ["20261027"], _grid(), 600.0, datetime(2026, 10, 27).date()))
        assert out == pytest.approx(601.0)

    def test_a_bad_guess_is_refined_once(self, world):
        world.leg_defaults = {(600.0, "C"): (9.0, 9.04), (600.0, "P"): (3.0, 3.04),     # says S = 606
                              (606.0, "C"): (3.0, 3.04), (606.0, "P"): (3.0, 3.04)}
        out = asyncio.run(probe.estimate_spot_from_options(
            FakeIBKR(world), ["20261027"], _grid(), 600.0, datetime(2026, 10, 27).date()))
        assert out == pytest.approx(606.0)

    def test_no_quotes_means_none_and_expired_chains_are_ignored(self, world):
        assert asyncio.run(probe.estimate_spot_from_options(
            FakeIBKR(world), ["20261027"], _grid(), 600.0, datetime(2026, 10, 27).date())) is None
        assert asyncio.run(probe.estimate_spot_from_options(
            FakeIBKR(world), ["20261020"], _grid(), 600.0, datetime(2026, 10, 27).date())) is None

    def test_with_no_guess_it_starts_from_the_middle_strike(self, world):
        world.leg_defaults = {(600.0, "C"): (6.0, 6.04), (600.0, "P"): (5.0, 5.04)}
        out = asyncio.run(probe.estimate_spot_from_options(
            FakeIBKR(world), ["20261027"], _grid(), None, datetime(2026, 10, 27).date()))
        assert out == pytest.approx(601.0)

    def test_the_last_recorded_spot_is_the_guess(self):
        assert probe._last_known_spot() is None
        _write([{"ts": _t(0), "run_id": "p", "cmd": "preflight", "trial": None, "event": "legs_selected", "spot": 762.95}])
        _write([{"ts": _t(1), "run_id": "p", "cmd": "preflight", "trial": None, "event": "legs_selected", "spot": 763.35}])
        assert probe._last_known_spot() == 763.35


class TestFetchChainFallback:
    def _go(self, world, monkeypatch, *, stock_price):
        monkeypatch.setattr(probe, "now_et", lambda: datetime(2026, 10, 27, 11, 0, tzinfo=ET))
        _chain(world, 600.0, exps=("20261027", "20261030"))

        async def fake_poll(ibkr, stock, *a, **k):
            return stock_price

        monkeypatch.setattr(probe, "poll_spot", fake_poll)
        return asyncio.run(probe.fetch_chain(FakeIBKR(world)))

    def test_the_stock_price_is_preferred(self, world, monkeypatch):
        world.leg_defaults = {(600.0, "C"): (9.0, 9.04), (600.0, "P"): (3.0, 3.04)}
        _stock, spot, exps, strikes = self._go(world, monkeypatch, stock_price=611.0)
        assert spot == 611.0 and exps[0] == "20261027" and 600.0 in strikes

    def test_a_silent_stock_falls_back_to_the_options(self, world, monkeypatch):
        world.leg_defaults = {(600.0, "C"): (6.0, 6.04), (600.0, "P"): (5.0, 5.04)}
        _stock, spot, _e, _s = self._go(world, monkeypatch, stock_price=None)
        assert spot == pytest.approx(601.0)

    def test_nothing_at_all_is_refused(self, world, monkeypatch):
        with pytest.raises(probe.ProbeGuardError, match="no price for the underlying"):
            self._go(world, monkeypatch, stock_price=None)


class TestSamplerSpotFallback:
    def _sampler(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        put = next(l for l in t["legs"] if l["right"] == "P" and l["open_action"] == "SELL")
        call_c = SimpleNamespace(conId=77777, secType="OPT")
        world.contracts[77777] = SimpleNamespace(conId=77777, symbol="SPY", secType="OPT", strike=put["strike"],
                                                 right="C", lastTradeDateOrContractMonth=put["expiry"])
        world.leg_defaults[(put["strike"], "C")] = (6.0, 6.04)
        contracts = {l["conId"]: SimpleNamespace(conId=l["conId"], secType="OPT") for l in t["legs"]}
        world.set_leg(put["conId"], 3.0, 3.04)
        s = probe.QuoteSampler(ibkr, _ev(), contracts, SimpleNamespace(conId=1, secType="STK"), None,
                               parity=(call_c, put["conId"], float(put["strike"])))
        s.start()
        return s, put

    def test_uses_the_pair_when_the_stock_is_silent_and_the_stock_when_it_is_not(self, world):
        s, put = self._sampler(world)
        assert s.spot() == pytest.approx(put["strike"] + 6.02 - 3.02)
        world.set_spot(611.5)
        assert s.spot() == 611.5

    def test_the_extra_subscription_is_cancelled_on_stop(self, world):
        s, _put = self._sampler(world)
        cancelled = []
        world.cancelMktData = lambda c: cancelled.append(getattr(c, "conId", c))
        s.stop()
        assert 77777 in cancelled


# ==========================================================================
# the controlled trials: a QQQ run, a long wing nobody bids for
# ==========================================================================
class TestSymbolOption:
    def test_use_symbol_accepts_the_probe_underlyings_only(self, monkeypatch):
        monkeypatch.setattr(probe, "SYMBOL", "SPY")
        probe.use_symbol("QQQ")
        assert probe.SYMBOL == "QQQ"
        with pytest.raises(probe.ProbeGuardError, match="only"):
            probe.use_symbol("IWM")
        assert probe.SYMBOL == "QQQ"

    def test_caps_accept_qqq_and_refuse_anything_else(self, monkeypatch):
        legs = TestGuardCaps.LEGS
        monkeypatch.setattr(probe, "SYMBOL", "QQQ")
        probe.guard_caps(legs)
        monkeypatch.setattr(probe, "SYMBOL", "IWM")
        with pytest.raises(probe.ProbeGuardError, match="pinned"):
            probe.guard_caps(legs)

    def test_main_applies_the_flag_before_the_command_runs(self, monkeypatch):
        monkeypatch.setattr(probe, "SYMBOL", "SPY")
        seen = []

        async def handler(args):
            seen.append(probe.SYMBOL)
            return {}

        monkeypatch.setitem(probe._HANDLERS, "preflight", handler)
        assert probe.main(["preflight", "--symbol", "QQQ"]) == 0
        assert seen == ["QQQ"]
        assert probe.build_parser().parse_args(["open", "--trial", "Q1", "--kind", "pin"]).symbol == "SPY"
        with pytest.raises(SystemExit):
            probe.build_parser().parse_args(["open", "--trial", "Q1", "--kind", "pin", "--symbol", "IWM"])

    def test_the_trial_remembers_its_symbol_and_later_commands_use_it(self, cmds, monkeypatch):
        state, t = _legs_state(cmds, cmds.ibkr)
        t["symbol"] = "QQQ"
        probe.save_state(state)
        _quote_legs(cmds, t)
        cmds.matcher = _fill_when_limit_at_least(0.0)
        monkeypatch.setattr(probe, "SYMBOL", "SPY")
        asyncio.run(probe.cmd_close(_args(no_truth_tail=True)))
        assert probe.SYMBOL == "QQQ"
        assert probe.new_trial("X", "pin", {"expiry": "e", "dte": 0, "width": 3.0, "legs": []}, 3)["symbol"] == "QQQ"

    def test_the_window_rule_for_0dte_applies_to_nobid_too(self, monkeypatch):
        _at(14, 31, monkeypatch)
        with pytest.raises(probe.ProbeGuardError, match="14:30"):
            probe.guard_window("open", "nobid")
        _at(15, 16, monkeypatch)
        with pytest.raises(probe.ProbeGuardError, match="15:15"):
            probe.guard_window("close", "nobid")


class TestNobidKind:
    TODAY = datetime(2026, 10, 27).date()
    EXPS = ["20261027", "20261030"]

    def test_the_call_wing_sits_about_one_and_a_half_percent_out(self):
        s = probe.select_structure("nobid", 765.0, self.EXPS, _grid(700, 840), self.TODAY)
        by = {(l["right"], l["open_action"]): l["strike"] for l in s["legs"]}
        assert s["dte"] == 0
        assert by[("P", "SELL")] == 765.0 and by[("P", "BUY")] == 762.0           # the put side is the pin shape
        assert by[("C", "SELL")] == 777.0 and by[("C", "BUY")] == 780.0           # 777 = first strike >= 765 * 1.015
        probe.guard_caps([{**l, "action": l["open_action"]} for l in s["legs"]])

    def test_an_explicit_short_call_is_used(self):
        s = probe.select_structure("nobid", 765.0, self.EXPS, _grid(700, 840), self.TODAY, short_call=774.0)
        by = {(l["right"], l["open_action"]): l["strike"] for l in s["legs"]}
        assert by[("C", "SELL")] == 774.0 and by[("C", "BUY")] == 777.0

    @pytest.mark.parametrize("kind,sc,why", [("pin", 774.0, "only for the nobid"), ("nobid", 774.5, "not a listed"),
                                              ("nobid", 760.0, "not a listed strike at or above")])
    def test_a_bad_short_call_is_refused(self, kind, sc, why):
        with pytest.raises(probe.ProbeGuardError, match=why):
            probe.select_structure(kind, 765.0, self.EXPS, _grid(700, 840), self.TODAY, short_call=sc)

    def test_the_flag_reaches_the_parser(self):
        a = probe.build_parser().parse_args(["open", "--trial", "N1", "--kind", "nobid", "--short-call", "774"])
        assert a.short_call == 774.0

    def test_pin_is_unchanged(self):
        s = probe.select_structure("pin", 765.0, self.EXPS, _grid(700, 840), self.TODAY)
        by = {(l["right"], l["open_action"]): l["strike"] for l in s["legs"]}
        assert by[("C", "SELL")] == 768.0

    def test_it_needs_a_zero_dte_expiry(self):
        with pytest.raises(probe.ProbeGuardError, match="0DTE"):
            probe.select_structure("nobid", 765.0, ["20261030"], _grid(700, 840), self.TODAY)


class TestSellLegsWithoutBid:
    LEGS = [{"conId": 1, "action": "SELL"}, {"conId": 2, "action": "BUY"},
            {"conId": 3, "action": "BUY"}, {"conId": 4, "action": "SELL"}]

    @pytest.mark.parametrize("bid,expected", [(-1.0, [4]), (0.0, [4]), (None, [4]), (float("nan"), [4]), (0.01, [])])
    def test_a_sold_leg_without_a_usable_bid_is_reported(self, bid, expected):
        q = {1: (0.5, 0.52), 2: (2.8, 2.9), 3: (0.0, 0.01), 4: (bid, 0.01)}
        assert probe.sell_legs_without_bid(self.LEGS, q) == expected

    def test_a_bought_leg_without_a_bid_is_not_a_problem(self):
        q = {1: (0.5, 0.52), 2: (2.8, 2.9), 3: (-1.0, 0.01), 4: (0.02, 0.03)}
        assert probe.sell_legs_without_bid(self.LEGS, q) == []

    def test_the_rung_plan_records_it(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        for l in t["legs"]:
            if l["close_action"] == "SELL":
                world.set_leg(l["conId"], 0.20, 0.25)             # every sold leg bid ...
        wing = next(l["conId"] for l in t["legs"] if l["close_action"] == "SELL" and l["right"] == "C")
        world.set_leg(wing, -1.0, 0.01)                           # ... except the long call wing
        world.matcher = _fill_when_limit_at_least(0.0)
        asyncio.run(_close(world, ibkr, state, t))
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        plan = next(e for e in events if e["event"] == "rung_plan")
        assert plan["no_bid_sell_legs"] == [wing]


class TestRequireNobid:
    def test_refuses_to_start_when_every_sold_leg_has_a_bid(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        for l in t["legs"]:                                    # the helper quotes the SELL legs at 0.0: give them bids
            if l["close_action"] == "SELL":
                cmds.set_leg(l["conId"], 0.20, 0.25)
        out = asyncio.run(probe.cmd_close(_args(no_truth_tail=True, require_nobid=True)))
        assert out["outcome"] == "aborted_nobid_condition_absent" and cmds.placed == []
        assert probe.load_state()["trials"]["T"]["close"]["status"] == "aborted"

    def test_proceeds_when_a_sold_leg_has_no_bid(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        wing = next(l["conId"] for l in t["legs"] if l["close_action"] == "SELL" and l["right"] == "C")
        cmds.set_leg(wing, -1.0, 0.01)
        cmds.matcher = _fill_when_limit_at_least(0.0)
        out = asyncio.run(probe.cmd_close(_args(no_truth_tail=True, require_nobid=True)))
        assert out["outcome"] == "filled" and len(cmds.placed) == 1

    def test_without_the_flag_the_condition_is_only_recorded(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        cmds.matcher = _fill_when_limit_at_least(0.0)
        asyncio.run(probe.cmd_close(_args(no_truth_tail=True)))
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        assert any(e["event"] == "nobid_check" and e["required"] is False for e in events)

    def test_the_flag_reaches_the_parser(self):
        assert probe.build_parser().parse_args(["close", "--trial", "P2", "--require-nobid"]).require_nobid is True


class TestQuotesCommand:
    def test_reports_each_leg_and_the_sold_legs_without_a_bid(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t, natural=1.0)
        wing = next(l["conId"] for l in t["legs"] if l["close_action"] == "SELL" and l["right"] == "C")
        cmds.set_leg(wing, -1.0, 0.01)
        out = asyncio.run(probe.cmd_quotes(_args()))
        assert out["trial"] == "T" and len(out["legs"]) == 4
        assert out["close_sell_legs_without_bid"] >= 1
        row = next(r for r in out["legs"] if r["bid"] == -1.0)
        assert row["close_action"] == "SELL" and row["right"] == "C"
        assert out["natural_cost_to_close"] is not None

    def test_an_unknown_trial_is_refused(self, cmds):
        probe.save_state({"schema": 1, "trials": {}})
        with pytest.raises(probe.ProbeGuardError, match="no trial"):
            asyncio.run(probe.cmd_quotes(_args(trial="Z")))

    def test_places_nothing(self, cmds):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        asyncio.run(probe.cmd_quotes(_args()))
        assert cmds.placed == []


class TestExcludeUnbid:
    def _setup(self, cmds, *, wing_ask=0.01, wing_bid=-1.0):
        state, t = _legs_state(cmds, cmds.ibkr)
        _quote_legs(cmds, t)
        for l in t["legs"]:
            if l["close_action"] == "SELL":
                cmds.set_leg(l["conId"], 0.20, 0.25)
        wing = next(l["conId"] for l in t["legs"] if l["close_action"] == "SELL" and l["right"] == "C")
        cmds.set_leg(wing, wing_bid, wing_ask)
        cmds.matcher = _fill_when_limit_at_least(0.0)
        return t, wing

    def test_the_closing_combo_is_built_without_the_worthless_unbid_wing(self, cmds):
        t, wing = self._setup(cmds)
        out = asyncio.run(probe.cmd_close(_args(no_truth_tail=True, exclude_unbid=True)))
        assert out["outcome"] == "filled"
        legs = cmds.placed[0].contract.comboLegs
        assert len(legs) == 3 and wing not in [cl.conId for cl in legs]
        assert all(cl.action in ("BUY", "SELL") for cl in legs)
        assert sum(1 for cl in legs if cl.action == "BUY") == 2                 # both shorts are bought back
        assert cmds.positions[wing] == 1.0                                       # the wing is left to expire
        assert probe.load_state()["trials"]["T"]["close"]["left_open"] == [wing]

    def test_the_natural_excludes_the_wing(self, cmds):
        t, wing = self._setup(cmds)
        asyncio.run(probe.cmd_close(_args(no_truth_tail=True, exclude_unbid=True)))
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        assert any(e["event"] == "excluded_legs" and e["legs"] == [wing] for e in events)

    def test_a_wing_that_still_has_value_is_never_left_behind(self, cmds):
        t, wing = self._setup(cmds, wing_ask=0.40)
        out = asyncio.run(probe.cmd_close(_args(no_truth_tail=True, exclude_unbid=True)))
        assert out["outcome"] == "aborted_unbid_leg_has_value" and cmds.placed == []

    def test_without_the_flag_the_combo_keeps_all_four_legs(self, cmds):
        t, wing = self._setup(cmds)
        asyncio.run(probe.cmd_close(_args(no_truth_tail=True)))
        assert len(cmds.placed[0].contract.comboLegs) == 4

    def test_nothing_is_excluded_when_every_sold_leg_has_a_bid(self, cmds):
        t, wing = self._setup(cmds, wing_bid=0.05, wing_ask=0.06)
        asyncio.run(probe.cmd_close(_args(no_truth_tail=True, exclude_unbid=True)))
        assert len(cmds.placed[0].contract.comboLegs) == 4

    def test_the_flag_reaches_the_parser(self):
        assert probe.build_parser().parse_args(["close", "--trial", "N2", "--exclude-unbid"]).exclude_unbid is True


class TestPriceBand:
    def test_the_band_text_is_parsed(self):
        e = [{"text": "Order Canceled - reason:We cannot accept an order at a limit price at or more aggressive "
                      "than 1.325. Please submit your order using a <br>limit price that is closer to the current"}]
        assert probe.parse_band_reject(e) == 1.325
        assert probe.parse_band_reject([{"text": "other"}, {"text": "at or more aggressive than 2"}]) == 2.0
        assert probe.parse_band_reject([]) is None and probe.parse_band_reject(None) is None

    def test_a_band_reject_is_not_the_end_the_next_rung_prices_under_it(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t, natural=1.0)
        world.band = 1.05
        world.matcher = _fill_when_limit_at_least(0.0)
        res = asyncio.run(_close(world, ibkr, state, t))
        limits = [tr.order.lmtPrice for tr in world.placed]
        assert res["outcome"] == "filled" and len(limits) == 2
        assert limits[0] >= 1.05 and limits[1] == 1.04                  # clamped one tick under the band
        events = [json.loads(l) for f in probe.PROBE_DIR.glob("*/events.jsonl") for l in f.read_text().splitlines()]
        assert any(e["event"] == "band_reject" and e["band"] == 1.05 for e in events)
        assert any(e["event"] == "rung_plan" and e.get("band_cap") == 1.04 for e in events)

    def test_a_non_band_reject_still_stops_the_ladder(self, world):
        ibkr = FakeIBKR(world)
        state, t = _legs_state(world, ibkr)
        _quote_legs(world, t)
        world.reject_code = 201
        assert asyncio.run(_close(world, ibkr, state, t))["outcome"] == "rejected"
        assert len(world.placed) == 1


class TestUnbidLegFinding:
    def _run_with(self, run_id, start, *, unbid, fill_k=None, exclude=False):
        ev = _run(run_id=run_id, start=start, fill_k=fill_k)
        for e in ev:
            if e["event"] == "rung_plan":
                e["no_bid_sell_legs"] = [4] if unbid else []
        if exclude:
            ev.append({"run_id": run_id, "cmd": "close", "trial": "T", "ts": _t(start - 1), "event": "excluded_legs",
                       "legs": [4], "remaining": [1, 2, 3]})
        return ev

    def test_blocked_then_fixed_by_leaving_the_leg_out(self):
        _write([_open_event()] + self._run_with("a", 100, unbid=True)
               + self._run_with("b", 20000, unbid=True, fill_k=0, exclude=True))
        f = _report()["unbid_leg"]
        assert f["verdict"] == "UNBID_LEG_BLOCKS_THE_COMBO"
        assert f["combo_kept_unbid_leg"] == {"runs": 1, "filled": 0} and f["combo_left_it_out"] == {"runs": 1, "filled": 1}

    def test_blocked_without_the_exclusion_check_is_labelled_unconfirmed(self):
        _write([_open_event()] + self._run_with("a", 100, unbid=True))
        assert _report()["unbid_leg"]["verdict"] == "UNBID_LEG_BLOCKED_THE_COMBO_NOT_YET_CONFIRMED_BY_EXCLUSION"

    def test_an_unbid_leg_that_did_not_stop_a_fill_is_reported_as_such(self):
        _write([_open_event()] + self._run_with("a", 100, unbid=True, fill_k=1))
        assert _report()["unbid_leg"]["verdict"] == "UNBID_LEG_DID_NOT_PREVENT_FILLS"

    def test_no_unbid_legs_no_finding(self):
        _write([_open_event()] + self._run_with("a", 100, unbid=False, fill_k=0))
        assert _report()["unbid_leg"]["verdict"] is None

    def test_band_rejects_are_counted(self):
        ev = self._run_with("a", 100, unbid=False, fill_k=0)
        ev.append({"run_id": "a", "cmd": "close", "trial": "T", "ts": _t(105), "event": "band_reject", "k": 0, "band": 1.1})
        _write([_open_event()] + ev)
        assert _report()["runs"][0]["band_rejects"] == 1


class TestReportHousekeeping:
    def test_one_fill_among_the_exclusion_runs_is_enough_to_confirm(self):
        ev = TestUnbidLegFinding()._run_with("a", 100, unbid=True)
        ev += TestUnbidLegFinding()._run_with("b", 20000, unbid=True, exclude=True, fill_k=None, )
        ev += TestUnbidLegFinding()._run_with("c", 40000, unbid=True, exclude=True, fill_k=1)
        _write([_open_event()] + ev)
        assert _report()["unbid_leg"]["verdict"] == "UNBID_LEG_BLOCKS_THE_COMBO"

    def test_a_leg_out_run_is_not_a_combo_run(self):
        _write([_open_event(), {"ts": _t(5), "run_id": "lo", "cmd": "flatten", "trial": "T", "event": "rung_plan",
                                "step": "legout", "k": 0, "leg": 2, "action": "BUY", "bid": 1.0, "ask": 1.1, "limit": 1.1}])
        assert _report()["runs"] == []


class TestHeadline:
    def test_the_headline_states_the_finding_when_it_is_confirmed(self):
        t = TestUnbidLegFinding()
        _write([_open_event()] + t._run_with("a", 100, unbid=True) + t._run_with("b", 20000, unbid=True, fill_k=0, exclude=True))
        assert "nobody bids for does not fill" in _report()["headline"]

    def test_otherwise_it_is_the_verdict(self):
        _write([_open_event()] + _run(fill_k=0))
        assert _report()["headline"] == "BOT_LADDER_SUFFICIENT"

"""W53 follow-up: scripts/probe_w53_roll_atomicity.py.

The probe places real IBKR orders, so nothing here touches a real IBKR
connection. Following test_w53_roll_target_selection.py's convention (real
OptionContract/OptionsChain objects behind a fake chain service, not
MagicMock), IBKR itself is mocked at the network edge (connect_probe /
IBKRClient) while every business-logic function the probe defines --
guards, pricing, contract selection, the reprice ladder, the open/roll/
flatten orchestration -- runs for real. `_FakeIB.placeOrder` auto-fills by
default so most tests observe one 1-second status poll; the ladder/timeout
test controls that explicitly.

No inspect.getsource -- every test drives the real functions.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import time as _time_mod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ait.config.settings import AssignmentResponseConfig  # noqa: E402
from ait.data.options_chain import OptionContract, OptionsChain  # noqa: E402
from ait.utils.time import ET  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "probe_w53_roll_atomicity", ROOT / "scripts" / "probe_w53_roll_atomicity.py")
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)  # type: ignore[union-attr]

SYMBOL = probe.SYMBOL


# --------------------------------------------------------------------------
# Fakes -- the IBKR/network edge only.
# --------------------------------------------------------------------------

class _FakeOrderStatus:
    def __init__(self, status: str = "PreSubmitted") -> None:
        self.status = status
        self.filled = 0
        self.remaining = 1
        self.avgFillPrice = 0.0


class _FakeTrade:
    def __init__(self, contract, order, status: str) -> None:
        self.contract = contract
        self.order = order
        self.orderStatus = _FakeOrderStatus(status)
        self.fills: list = []
        self.log: list = []


class _FakeIB:
    """Stands in for ib_insync.IB at the boundary IBKRClient wraps."""

    def __init__(self, auto_fill: bool = True) -> None:
        self._next_order_id = 1000
        self._quotes: dict[int, tuple[float | None, float | None]] = {}
        # Orchestration-level tests (cmd_open/cmd_roll/cmd_flatten) qualify
        # legs with dynamically-assigned conIds they don't know in advance --
        # any leg without an explicitly seeded quote gets this default so the
        # net-price synthesis has something to work with. Tests that care
        # about a SPECIFIC price call set_leg_quote() to override it.
        self.default_leg_quote: tuple[float | None, float | None] = (1.00, 1.10)
        self._bag_quote: tuple[float | None, float | None] = (None, None)
        self._trades: dict[int, _FakeTrade] = {}
        self.placed: list[_FakeTrade] = []
        self.auto_fill = auto_fill
        self._positions: list = []
        self._open_trades: list = []
        self.whatif_calls: list = []

    def set_leg_quote(self, con_id: int, bid: float | None, ask: float | None) -> None:
        self._quotes[con_id] = (bid, ask)

    def set_bag_quote(self, bid: float | None, ask: float | None) -> None:
        self._bag_quote = (bid, ask)

    def reqMktData(self, contract, *a, **k):
        return None

    def cancelMktData(self, contract):
        return None

    def ticker(self, contract):
        if getattr(contract, "comboLegs", None):
            bid, ask = self._bag_quote
        else:
            con_id = getattr(contract, "conId", 0)
            bid, ask = self._quotes.get(con_id, self.default_leg_quote)
        return SimpleNamespace(bid=bid, ask=ask)

    def placeOrder(self, contract, order):
        order_id = getattr(order, "orderId", 0)
        if not order_id or order_id not in self._trades:
            order.orderId = self._next_order_id
            self._next_order_id += 1
            status = "Filled" if self.auto_fill else "Submitted"
            trade = _FakeTrade(contract, order, status)
            if self.auto_fill:
                trade.orderStatus.filled = order.totalQuantity
                trade.orderStatus.avgFillPrice = order.lmtPrice
            self._trades[order.orderId] = trade
            self.placed.append(trade)
            return trade
        trade = self._trades[order_id]
        trade.order.lmtPrice = order.lmtPrice
        return trade

    def cancelOrder(self, order):
        trade = self._trades.get(getattr(order, "orderId", 0))
        if trade:
            trade.orderStatus.status = "Cancelled"

    def positions(self):
        return self._positions

    def openTrades(self):
        return self._open_trades

    async def whatIfOrderAsync(self, contract, order):
        self.whatif_calls.append((contract, order))
        return SimpleNamespace(initMarginChange="100.0", maintMarginChange="100.0",
                                equityWithLoanChange="-100.0", commission="1.30",
                                warningText="")


class _FakeIBKRClient:
    def __init__(self, session_account: str = "DU123456", client_id: int = probe.PROBE_CLIENT_ID,
                 auto_fill: bool = True) -> None:
        self.ib = _FakeIB(auto_fill=auto_fill)
        self.session_account = session_account
        self.client_id = client_id
        self._qualify_map: dict[tuple, int] = {}
        self._next_con_id = 5000
        self._order_errors: dict[int, list] = {}

    async def qualify_contracts_batch(self, contracts):
        out = []
        for c in contracts:
            key = (c.symbol, c.lastTradeDateOrContractMonth, c.strike, c.right)
            con_id = self._qualify_map.get(key)
            if con_id is None:
                con_id = self._next_con_id
                self._next_con_id += 1
                self._qualify_map[key] = con_id
            c.conId = con_id
            out.append(c)
        return out

    async def place_order(self, contract, order):
        return self.ib.placeOrder(contract, order)

    async def cancel_order(self, trade_or_id):
        if hasattr(trade_or_id, "order"):
            self.ib.cancelOrder(trade_or_id.order)
        return True

    def order_error_codes(self, order_id):
        return self._order_errors.get(order_id, [])

    async def disconnect(self):
        return None


class _FakeChainService:
    def __init__(self, chains: list[OptionsChain]) -> None:
        self._chains = chains

    async def get_chain(self, symbol, min_dte=None, max_dte=None):
        return self._chains


def _contract(right: str, strike: float, expiry: date, delta: float,
              *, volume=2_500, open_interest=9_000, bid=1.45, ask=1.55,
              con_id: int | None = None) -> OptionContract:
    return OptionContract(
        symbol=SYMBOL, expiry=expiry, strike=strike, right=right,
        bid=bid, ask=ask, last=round((bid + ask) / 2, 2),
        volume=volume, open_interest=open_interest, implied_vol=0.18,
        delta=delta, con_id=con_id or (int(strike * 10) + (1 if right == "C" else 2)),
        source="yahoo_delayed",
    )


def _near_chain(expiry: date) -> OptionsChain:
    """Vertical A candidates: short put lands at -0.20 delta (95), long
    wing 3 wide (92)."""
    puts = [
        _contract("P", 92.0, expiry, delta=-0.08),
        _contract("P", 95.0, expiry, delta=-0.20),
        _contract("P", 100.0, expiry, delta=-0.50),
    ]
    calls = [_contract("C", 105.0, expiry, delta=0.20)]
    return OptionsChain(symbol=SYMBOL, underlying_price=100.0, expiry=expiry,
                         calls=calls, puts=puts, source="yahoo_delayed")


def _far_chain(expiry: date) -> OptionsChain:
    """Roll target B: short put at -0.20 delta (90), long wing 3 wide (87)."""
    puts = [
        _contract("P", 87.0, expiry, delta=-0.08),
        _contract("P", 90.0, expiry, delta=-0.20),
        _contract("P", 96.0, expiry, delta=-0.50),
    ]
    calls = [_contract("C", 110.0, expiry, delta=0.20)]
    return OptionsChain(symbol=SYMBOL, underlying_price=100.0, expiry=expiry,
                         calls=calls, puts=puts, source="yahoo_delayed")


def _wide_only_chain(expiry: date) -> OptionsChain:
    """The only strike that beats the short's OWN distance to the target long
    (target = short - OPEN_WIDTH = 95-3 = 92, so the short itself sits 3.0
    from target) is still wider than MAX_WIDTH: 89.4 is 2.6 from target
    (closer than the short's 3.0, so pick_vertical doesn't self-collide) but
    5.6 wide from the short (over MAX_WIDTH=5) -- pick_vertical must refuse
    rather than build an oversized wing."""
    puts = [
        _contract("P", 95.0, expiry, delta=-0.20),
        _contract("P", 89.4, expiry, delta=-0.05),
    ]
    return OptionsChain(symbol=SYMBOL, underlying_price=100.0, expiry=expiry,
                         calls=[], puts=puts, source="yahoo_delayed")


@dataclass
class _FakeSettings:
    assignment_response: AssignmentResponseConfig = field(default_factory=AssignmentResponseConfig)


def _wire_fakes(monkeypatch, ibkr: _FakeIBKRClient, chains: list[OptionsChain]) -> None:
    async def _fake_connect_probe():
        return ibkr

    def _fake_build_services(_ibkr):
        return None, _FakeChainService(chains), _FakeSettings()

    monkeypatch.setattr(probe, "connect_probe", _fake_connect_probe)
    monkeypatch.setattr(probe, "build_services", _fake_build_services)


def _wire_rth_open(monkeypatch, minutes_after_open: int = 60) -> None:
    """A fixed ET instant safely inside RTH, `minutes_after_open` past 09:30."""
    fixed = datetime(2026, 9, 28, 9, 30, tzinfo=ET) + timedelta(minutes=minutes_after_open)
    monkeypatch.setattr(probe, "now_et", lambda: fixed)
    monkeypatch.setattr(probe, "is_market_open", lambda: True)
    monkeypatch.setattr(probe, "get_market_close", lambda d: fixed.time().replace(
        hour=16, minute=0))


def _quote_all_legs(ibkr: _FakeIBKRClient, legs_conids: list[int],
                     bid: float = 1.00, ask: float = 1.10) -> None:
    for cid in legs_conids:
        ibkr.ib.set_leg_quote(cid, bid, ask)


def _seed_state(tmp_path: Path, state: dict) -> None:
    """Write a state.json under the SAME ET-date folder _today_dir() will
    resolve to -- keyed off probe.now_et() (which _wire_rth_open may have
    pinned to a fixed instant), never the host's date.today()."""
    day_dir = tmp_path / probe.now_et().date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / "state.json").write_text(json.dumps(state))


@pytest.fixture(autouse=True)
def _isolate_repo_data_files(tmp_path, monkeypatch):
    """Every test is isolated from THIS checkout's real data/HALT_UNTRACKED
    and data/ait_state.db by default (both exist on this box -- HALT_UNTRACKED
    currently lists real untracked QQQ legs). Collision-guard tests override
    these explicitly with their own seeded content."""
    monkeypatch.setattr(probe, "HALT_UNTRACKED_PATH", tmp_path / "HALT_UNTRACKED")
    monkeypatch.setattr(probe, "DB_PATH", tmp_path / "ait_state.db")


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

class TestClientIdChoice:
    def test_probe_client_id_is_outside_the_bot_ladder(self):
        bot_ladder = {1} | {1 + 100 + i for i in range(5)}
        assert probe.PROBE_CLIENT_ID not in bot_ladder

    def test_quantity_is_pinned_to_one_contract(self):
        assert probe.QUANTITY == 1


class TestGuardAccount:
    def test_paper_account_passes(self):
        probe.guard_account(_FakeIBKRClient(session_account="DU123456"))  # no raise

    def test_live_looking_account_refused(self):
        with pytest.raises(probe.ProbeGuardError):
            probe.guard_account(_FakeIBKRClient(session_account="U1234567"))

    def test_empty_account_refused(self):
        with pytest.raises(probe.ProbeGuardError):
            probe.guard_account(_FakeIBKRClient(session_account=""))


class TestGuardRthWindow:
    def test_inside_rth_well_clear_of_edges_passes(self, monkeypatch):
        _wire_rth_open(monkeypatch, minutes_after_open=90)
        probe.guard_rth_window()  # no raise

    def test_market_closed_refused(self, monkeypatch):
        _wire_rth_open(monkeypatch, minutes_after_open=90)
        monkeypatch.setattr(probe, "is_market_open", lambda: False)
        with pytest.raises(probe.ProbeGuardError, match="not open"):
            probe.guard_rth_window()

    def test_within_buffer_of_open_refused(self, monkeypatch):
        _wire_rth_open(monkeypatch, minutes_after_open=10)
        with pytest.raises(probe.ProbeGuardError, match="open"):
            probe.guard_rth_window()

    def test_within_buffer_of_close_refused(self, monkeypatch):
        fixed = datetime(2026, 9, 28, 15, 45, tzinfo=ET)
        monkeypatch.setattr(probe, "now_et", lambda: fixed)
        monkeypatch.setattr(probe, "is_market_open", lambda: True)
        monkeypatch.setattr(probe, "get_market_close", lambda d: fixed.time().replace(
            hour=16, minute=0))
        with pytest.raises(probe.ProbeGuardError, match="close"):
            probe.guard_rth_window()


class TestGuardNoWorkingStep:
    def test_no_prior_state_passes(self):
        probe.guard_no_working_step({}, "open")  # no raise

    def test_working_step_refused(self):
        with pytest.raises(probe.ProbeGuardError):
            probe.guard_no_working_step({"open": {"status": "working"}}, "open")

    def test_filled_step_does_not_trip_this_guard(self):
        # a distinct guard (in cmd_open/cmd_roll) refuses re-running a
        # filled step -- this one only refuses an in-flight one.
        probe.guard_no_working_step({"open": {"status": "filled"}}, "open")


class TestGuardNoCollision:
    def _seed_halt_untracked(self, tmp_path: Path, monkeypatch, keys: list[str]) -> None:
        p = tmp_path / "HALT_UNTRACKED"
        p.write_text("untracked positions:\n" + "\n".join(f"  {k}" for k in keys))
        monkeypatch.setattr(probe, "HALT_UNTRACKED_PATH", p)

    def _seed_db(self, tmp_path: Path, monkeypatch, rows: list[tuple[str, str, list[dict]]]) -> None:
        db = tmp_path / "ait_state.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE trades (trade_id TEXT, symbol TEXT, status TEXT, legs TEXT)")
            for trade_id, status, legs in rows:
                conn.execute("INSERT INTO trades VALUES (?, ?, ?, ?)",
                             (trade_id, SYMBOL, status, json.dumps(legs)))
        monkeypatch.setattr(probe, "DB_PATH", db)

    def test_no_files_no_collision(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "HALT_UNTRACKED_PATH", tmp_path / "HALT_UNTRACKED")
        monkeypatch.setattr(probe, "DB_PATH", tmp_path / "ait_state.db")
        probe.guard_no_collision(
            [{"strike": 95.0, "right": "P", "expiry": "2026-10-02"}])  # no raise

    def test_halt_untracked_key_collision_refused(self, tmp_path, monkeypatch):
        self._seed_halt_untracked(tmp_path, monkeypatch, ["SPY:95.0:P:2026-10-02"])
        monkeypatch.setattr(probe, "DB_PATH", tmp_path / "ait_state.db")
        with pytest.raises(probe.ProbeGuardError, match="HALT_UNTRACKED"):
            probe.guard_no_collision(
                [{"strike": 95.0, "right": "P", "expiry": "2026-10-02"}])

    def test_open_db_trade_leg_collision_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "HALT_UNTRACKED_PATH", tmp_path / "HALT_UNTRACKED")
        self._seed_db(tmp_path, monkeypatch, [
            ("T1", "filled", [{"strike": 95.0, "right": "P", "expiry": "2026-10-02",
                                "action": "SELL"}]),
        ])
        with pytest.raises(probe.ProbeGuardError, match="collides"):
            probe.guard_no_collision(
                [{"strike": 95.0, "right": "P", "expiry": "2026-10-02"}])

    def test_closed_db_trade_legs_do_not_block(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "HALT_UNTRACKED_PATH", tmp_path / "HALT_UNTRACKED")
        self._seed_db(tmp_path, monkeypatch, [
            ("T1", "closed", [{"strike": 95.0, "right": "P", "expiry": "2026-10-02",
                                "action": "SELL"}]),
        ])
        probe.guard_no_collision(
            [{"strike": 95.0, "right": "P", "expiry": "2026-10-02"}])  # no raise

    def test_different_strike_does_not_collide(self, tmp_path, monkeypatch):
        self._seed_halt_untracked(tmp_path, monkeypatch, ["SPY:95.0:P:2026-10-02"])
        monkeypatch.setattr(probe, "DB_PATH", tmp_path / "ait_state.db")
        probe.guard_no_collision(
            [{"strike": 96.0, "right": "P", "expiry": "2026-10-02"}])  # no raise


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

class TestLegNetPrice:
    def test_credit_combo_is_signed_negative(self):
        # SELL the short (bid 1.00/ask 1.10), BUY the long (bid 0.20/ask 0.30)
        # -- a real credit vertical, mid net must be negative.
        legs = [{"conId": 1, "action": "SELL"}, {"conId": 2, "action": "BUY"}]
        quotes = {1: (1.00, 1.10), 2: (0.20, 0.30)}
        mid = probe.leg_net_price(legs, quotes, "mid")
        assert mid is not None and mid < 0
        assert mid == pytest.approx(-1.05 + 0.25)

    def test_debit_combo_is_signed_positive(self):
        legs = [{"conId": 1, "action": "BUY"}, {"conId": 2, "action": "SELL"}]
        quotes = {1: (1.00, 1.10), 2: (0.20, 0.30)}
        mid = probe.leg_net_price(legs, quotes, "mid")
        assert mid is not None and mid > 0

    def test_natural_side_is_buy_at_ask_sell_at_bid(self):
        legs = [{"conId": 1, "action": "SELL"}, {"conId": 2, "action": "BUY"}]
        quotes = {1: (1.00, 1.10), 2: (0.20, 0.30)}
        natural = probe.leg_net_price(legs, quotes, "natural")
        # SELL at bid (1.00, contributes -1.00), BUY at ask (0.30, contributes +0.30)
        assert natural == pytest.approx(-1.00 + 0.30)

    def test_missing_quote_returns_none(self):
        legs = [{"conId": 1, "action": "SELL"}, {"conId": 2, "action": "BUY"}]
        quotes = {1: (1.00, 1.10), 2: (None, None)}
        assert probe.leg_net_price(legs, quotes, "mid") is None
        assert probe.leg_net_price(legs, quotes, "natural") is None


# --------------------------------------------------------------------------
# The reprice ladder
# --------------------------------------------------------------------------

class TestMonitorAndLadder:
    async def test_immediate_fill_takes_one_poll_no_reprice(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        ibkr = _FakeIBKRClient(auto_fill=True)
        contract = SimpleNamespace(conId=1)
        order = SimpleNamespace(orderId=1, lmtPrice=-1.0, totalQuantity=1)
        trade = ibkr.ib.placeOrder(contract, order)
        result = await probe._monitor_and_ladder(ibkr, trade, -1.0, -0.8, "test_fill")
        assert result["status"] == "Filled"

    async def test_unfilled_order_ladders_monotonically_then_cancels(self, tmp_path, monkeypatch):
        """mid=-1.00 (credit), natural=-0.60 (less credit, marketable).

        2026-09-29 same-day W53 retry: LADDER_STEPS was extended past natural
        (0.0, 0.5, 1.0, 1.5, 2.0) to see whether a more aggressive price gets
        the roll combo to actually fill instead of cancelling unfilled (it
        did, on the 3rd trial). So the ladder now moves monotonically from
        mid toward AND PAST natural, ending at the final extrapolated rung,
        then cancels once that rung has rested past CANCEL_AFTER_SECONDS."""
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        monkeypatch.setattr(probe, "LADDER_STEP_SECONDS", 10)
        monkeypatch.setattr(probe, "CANCEL_AFTER_SECONDS", 10)
        ibkr = _FakeIBKRClient(auto_fill=False)
        contract = SimpleNamespace(conId=1)
        order = SimpleNamespace(orderId=1, lmtPrice=-1.0, totalQuantity=1)
        trade = ibkr.ib.placeOrder(contract, order)

        fake_clock = {"t": 0.0}

        def _fake_time():
            fake_clock["t"] += 12.0  # each check crosses one LADDER_STEP_SECONDS bucket
            return fake_clock["t"]

        async def _fast_sleep(_seconds):
            return None

        monkeypatch.setattr(probe.time, "time", _fake_time)
        monkeypatch.setattr(probe.asyncio, "sleep", _fast_sleep)

        result = await probe._monitor_and_ladder(ibkr, trade, -1.00, -0.60, "test_ladder")

        evidence_path = probe._today_dir() / "test_ladder.jsonl"
        reprice_log = [json.loads(line) for line in evidence_path.read_text().splitlines()]
        reprices = [e["limit"] for e in reprice_log if e["event"] == "reprice"]
        assert reprices, "expected at least one reprice"
        # monotonic toward and past natural: increasing (less negative ==
        # less credit conceded each step), bounded within [mid, the final
        # extrapolated rung at frac=2.0], ending there.
        natural_overshoot = -1.00 + 2.0 * (-0.60 - -1.00)  # frac=2.0 rung == -0.20
        assert reprices == sorted(reprices)
        assert all(-1.00 - 1e-9 <= r <= natural_overshoot + 1e-9 for r in reprices)
        assert reprices[-1] == pytest.approx(natural_overshoot)
        assert result["status"] == "Cancelled"


# --------------------------------------------------------------------------
# pick_vertical
# --------------------------------------------------------------------------

class TestPickVertical:
    async def test_picks_short_at_target_delta_and_a_bounded_wing(self):
        chain = _near_chain(date.today() + timedelta(days=7))
        svc = _FakeChainService([chain])
        short, long_ = await probe.pick_vertical(
            svc, "P", probe.OPEN_DTE_MIN, probe.OPEN_DTE_MAX,
            probe.OPEN_SHORT_DELTA_TARGET, probe.OPEN_SHORT_DELTA_TOLERANCE, probe.OPEN_WIDTH)
        assert short.strike == 95.0
        assert long_.strike == 92.0
        assert abs(short.strike - long_.strike) <= probe.MAX_WIDTH

    async def test_refuses_when_only_an_oversized_wing_is_available(self):
        chain = _wide_only_chain(date.today() + timedelta(days=7))
        svc = _FakeChainService([chain])
        with pytest.raises(probe.ProbeGuardError, match="MAX_WIDTH"):
            await probe.pick_vertical(
                svc, "P", probe.OPEN_DTE_MIN, probe.OPEN_DTE_MAX,
                probe.OPEN_SHORT_DELTA_TARGET, probe.OPEN_SHORT_DELTA_TOLERANCE, probe.OPEN_WIDTH)

    async def test_empty_chain_refuses(self):
        svc = _FakeChainService([])
        with pytest.raises(probe.ProbeGuardError):
            await probe.pick_vertical(
                svc, "P", probe.OPEN_DTE_MIN, probe.OPEN_DTE_MAX,
                probe.OPEN_SHORT_DELTA_TARGET, probe.OPEN_SHORT_DELTA_TOLERANCE, probe.OPEN_WIDTH)


# --------------------------------------------------------------------------
# Orchestration: cmd_open / cmd_roll / cmd_flatten
# --------------------------------------------------------------------------

class TestCmdOpen:
    async def test_places_a_two_leg_vertical_tagged_with_orderref(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        _wire_rth_open(monkeypatch)
        ibkr = _FakeIBKRClient(auto_fill=True)
        chain = _near_chain(date.today() + timedelta(days=7))
        _wire_fakes(monkeypatch, ibkr, [chain])

        result = await probe.cmd_open(SimpleNamespace())

        assert result["status"] == "filled"
        assert len(ibkr.ib.placed) == 1
        placed_order = ibkr.ib.placed[0].order
        assert placed_order.orderRef.startswith("W53PROBE-open-")
        bag = ibkr.ib.placed[0].contract
        assert len(bag.comboLegs) == 2
        actions = sorted(leg.action for leg in bag.comboLegs)
        assert actions == ["BUY", "SELL"]

    async def test_refuses_outside_rth(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        monkeypatch.setattr(probe, "is_market_open", lambda: False)
        ibkr = _FakeIBKRClient()
        _wire_fakes(monkeypatch, ibkr, [])
        with pytest.raises(probe.ProbeGuardError):
            await probe.cmd_open(SimpleNamespace())
        assert ibkr.ib.placed == []

    async def test_refuses_a_collision_with_halt_untracked(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        _wire_rth_open(monkeypatch)
        halt_path = tmp_path / "HALT_UNTRACKED"
        halt_path.write_text("untracked positions:\n  SPY:95.0:P:"
                              + str(date.today() + timedelta(days=7)))
        monkeypatch.setattr(probe, "HALT_UNTRACKED_PATH", halt_path)
        monkeypatch.setattr(probe, "DB_PATH", tmp_path / "ait_state.db")
        ibkr = _FakeIBKRClient()
        chain = _near_chain(date.today() + timedelta(days=7))
        _wire_fakes(monkeypatch, ibkr, [chain])
        with pytest.raises(probe.ProbeGuardError, match="collides"):
            await probe.cmd_open(SimpleNamespace())
        assert ibkr.ib.placed == []


class TestCmdRoll:
    async def test_roll_bag_has_exactly_four_legs_reversed_correctly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        _wire_rth_open(monkeypatch)
        monkeypatch.setattr(probe, "HALT_UNTRACKED_PATH", tmp_path / "HALT_UNTRACKED")
        monkeypatch.setattr(probe, "DB_PATH", tmp_path / "ait_state.db")

        open_expiry = date.today() + timedelta(days=7)
        roll_expiry = date.today() + timedelta(days=21)
        near = _near_chain(open_expiry)
        far = _far_chain(roll_expiry)

        # Seed state as if 'open' already filled vertical A (95 short / 92 long).
        state = {"open": {"status": "filled",
                           "legs": [
                               {"strike": 95.0, "right": "P", "action": "SELL",
                                "expiry": str(open_expiry)},
                               {"strike": 92.0, "right": "P", "action": "BUY",
                                "expiry": str(open_expiry)},
                           ]}}
        _seed_state(tmp_path, state)

        ibkr = _FakeIBKRClient(auto_fill=True)
        _wire_fakes(monkeypatch, ibkr, [far])
        # select_roll_target uses config.roll_target_dte_min/max, not the
        # probe's own DTE window -- default AssignmentResponseConfig (14-30)
        # covers roll_expiry (21 days out).

        result = await probe.cmd_roll(SimpleNamespace())

        assert result["status"] == "filled"
        bag = ibkr.ib.placed[-1].contract
        assert len(bag.comboLegs) == 4
        actions = sorted(leg.action for leg in bag.comboLegs)
        assert actions == ["BUY", "BUY", "SELL", "SELL"]
        # closing legs (reversed vs. the open vertical) + opening legs (new short/long)
        assert result["closing_legs"][0]["action"] == "BUY"   # was SELL (short) at open
        assert result["closing_legs"][1]["action"] == "SELL"  # was BUY (long) at open
        assert result["opening_legs"][0]["strike"] == 90.0
        assert result["opening_legs"][1]["strike"] == 87.0

    async def test_refuses_without_a_filled_open_step(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        _wire_rth_open(monkeypatch)
        ibkr = _FakeIBKRClient()
        _wire_fakes(monkeypatch, ibkr, [])
        with pytest.raises(probe.ProbeGuardError, match="'open'"):
            await probe.cmd_roll(SimpleNamespace())
        assert ibkr.ib.placed == []


class TestCmdFlatten:
    async def test_nothing_to_flatten_when_no_step_filled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        result = await probe.cmd_flatten(SimpleNamespace())
        assert result["status"] == "nothing_to_flatten"

    async def test_flattens_the_open_vertical_with_reversed_legs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        _wire_rth_open(monkeypatch)
        open_expiry = date.today() + timedelta(days=7)
        state = {"open": {"status": "filled",
                           "legs": [
                               {"strike": 95.0, "right": "P", "action": "SELL",
                                "expiry": str(open_expiry)},
                               {"strike": 92.0, "right": "P", "action": "BUY",
                                "expiry": str(open_expiry)},
                           ]}}
        _seed_state(tmp_path, state)

        ibkr = _FakeIBKRClient(auto_fill=True)
        _wire_fakes(monkeypatch, ibkr, [])

        result = await probe.cmd_flatten(SimpleNamespace())

        assert result["status"] == "filled"
        assert result["source"] == "open"
        actions = {leg["strike"]: leg["action"] for leg in result["legs"]}
        assert actions[95.0] == "BUY"   # was SELL
        assert actions[92.0] == "SELL"  # was BUY

    async def test_already_flat_is_a_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        state = {"open": {"status": "filled", "legs": []},
                 "flatten": {"status": "filled"}}
        _seed_state(tmp_path, state)
        result = await probe.cmd_flatten(SimpleNamespace())
        assert result["status"] == "already_flat"


# --------------------------------------------------------------------------
# place_combo_ladder guards
# --------------------------------------------------------------------------

class TestPlaceComboLadderGuards:
    async def test_missing_leg_quote_refuses_before_placing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        ibkr = _FakeIBKRClient()
        ibkr.ib.default_leg_quote = (None, None)  # simulate no quote reachable
        legs = [
            {"conId": 1, "action": "SELL", "_contract": SimpleNamespace(conId=1)},
            {"conId": 2, "action": "BUY", "_contract": SimpleNamespace(conId=2)},
        ]
        with pytest.raises(probe.ProbeGuardError, match="per-leg net price"):
            await probe.place_combo_ladder(ibkr, legs, "test", "W53PROBE-test-1")
        assert ibkr.ib.placed == []

    async def test_over_cap_roll_debit_refuses_before_placing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        ibkr = _FakeIBKRClient()
        legs = [
            {"conId": 1, "action": "BUY", "_contract": SimpleNamespace(conId=1)},
            {"conId": 2, "action": "BUY", "_contract": SimpleNamespace(conId=2)},
        ]
        # both legs cost money to buy -> net debit well above MAX_ROLL_NET_DEBIT
        ibkr.ib.set_leg_quote(1, 3.00, 3.20)
        ibkr.ib.set_leg_quote(2, 2.50, 2.70)
        with pytest.raises(probe.ProbeGuardError, match="debit cap"):
            await probe.place_combo_ladder(
                ibkr, legs, "test", "W53PROBE-test-1", debit_cap=probe.MAX_ROLL_NET_DEBIT)
        assert ibkr.ib.placed == []

    async def test_within_cap_places_normally(self, tmp_path, monkeypatch):
        monkeypatch.setattr(probe, "STATE_DIR", tmp_path)
        ibkr = _FakeIBKRClient(auto_fill=True)
        legs = [
            {"conId": 1, "action": "SELL", "_contract": SimpleNamespace(conId=1)},
            {"conId": 2, "action": "BUY", "_contract": SimpleNamespace(conId=2)},
        ]
        ibkr.ib.set_leg_quote(1, 1.00, 1.10)
        ibkr.ib.set_leg_quote(2, 0.20, 0.30)
        result = await probe.place_combo_ladder(
            ibkr, legs, "test", "W53PROBE-test-1", debit_cap=probe.MAX_ROLL_NET_DEBIT)
        assert result["status"] == "Filled"
        assert len(ibkr.ib.placed) == 1


# --------------------------------------------------------------------------
# Contract-key helper
# --------------------------------------------------------------------------

class TestContractKey:
    def test_matches_reconcilers_key_format(self):
        # data/HALT_UNTRACKED's real format, verified against reconciler.py:114-120
        assert probe._contract_key("QQQ", 790.0, "C", "2026-10-30") == "QQQ:790.0:C:2026-10-30"

    def test_right_is_normalized_to_one_letter_uppercase(self):
        assert probe._contract_key("SPY", 95.0, "put", "2026-10-02") == "SPY:95.0:P:2026-10-02"

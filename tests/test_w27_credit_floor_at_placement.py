"""W27 — regress-live-06: the credit floor at INITIAL PLACEMENT.

The credit-economics floors (absolute AIT_IC_MIN_CREDIT, plus
AIT_IC_MIN_CREDIT_WIDTH as a fraction of the wider wing) were enforced at SIGNAL
time (iron_condor.py) and at REPRICE (_credit_floors_hold), but never on the
order actually placed. The reprice ladder's steps were policed; its FIRST and
largest concession was not.

Two things the register did not state, both measured before this wave:

  * A signal sitting EXACTLY on the floor always breached, with a perfectly
    fresh quote and no adverse move: combo_entry_limit's offset has a 10c
    minimum, so step 0 concedes 2.5c unconditionally and $0.70 was placed at
    $0.67. Not an edge case — it is every marginal signal, i.e. precisely the
    trades the floor exists to filter.
  * The system REFUSED AT REPRICE EXACTLY WHAT IT ACCEPTED AT PLACEMENT.
    _credit_floors_hold(pending, -0.67) returns False for the very price the
    placement path put on the wire unchecked.

Worst case admitted by the staleness gate is ~30% below the floor, because the
pricing anchor is the live mid rather than the signal's validated credit.

The fix is two-part, mirroring guards that already existed on sibling paths:
REFUSE when the live mid itself is under the floor (the market cannot pay the
economics that justified the signal — same shape as the disorderly-spread and
stale-signal refusals), otherwise CLAMP so step 0 never concedes below it (same
shape as the debit side's _debit_cap).

House rules: every test drives the real executor through the real path with only
the broker boundary faked; no source-string assertions.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from ait.bot.state import StateManager, TradeStatus
from ait.execution.executor import (
    PendingOrder,
    TradeExecutor,
    _ladder_limit,
    combo_entry_limit,
)
from ait.risk.circuit_breaker import CircuitBreaker

from tests.fakes import FakeIB, FakeIBKRClient
from tests.test_w2_executor_risk_gates import (  # reuse the established builders
    _ic_signal,
    _named,
    _nbbo_for,
    events,  # noqa: F401 — fixture
)

MIN_CREDIT = 0.70
MIN_RATIO = 0.10


@pytest.fixture
def floors(monkeypatch):
    """Pin the contract floors so these tests do not drift with config.yaml."""
    monkeypatch.setenv("AIT_IC_MIN_CREDIT", str(MIN_CREDIT))
    monkeypatch.setenv("AIT_IC_MIN_CREDIT_WIDTH", str(MIN_RATIO))


@pytest.fixture
def env(tmp_path, monkeypatch, risk_config, floors):
    """Real executor + real state on a tmp DB; broker boundary faked."""
    monkeypatch.setenv("AIT_ALLOW_AFTER_HOURS", "1")
    monkeypatch.setenv("AIT_ENTRY_LADDER_START", "0.25")
    state = StateManager(tmp_path / "state.db")
    ib = FakeIB()
    client = FakeIBKRClient(ib)
    executor = TradeExecutor(
        ibkr_client=client,
        state=state,
        circuit_breaker=CircuitBreaker(risk_config),
        order_timeout=240,
    )
    return SimpleNamespace(state=state, ib=ib, client=client,
                           executor=executor, db=tmp_path / "state.db")


def _placed_limits(env) -> list[float]:
    return [o.lmtPrice for _c, o in env.ib.placed]


def _trade_status(env, trade_id: str) -> str | None:
    with sqlite3.connect(env.db) as conn:
        row = conn.execute("SELECT status FROM trades WHERE trade_id=?",
                           (trade_id,)).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# The asymmetry itself
# ---------------------------------------------------------------------------

class TestTheGuardWasNotWiredToPlacement:

    def test_negative_control_the_placed_price_was_one_the_guard_refuses(self, floors):
        """NEGATIVE CONTROL, and the whole finding in one test.

        Reproduces the PRE-W27 step-0 arithmetic for a signal sitting exactly on
        the floor, then asks the reprice guard — unchanged by this wave — what it
        thinks of that price. It says no. The guard was correct all along and
        simply never saw the first concession.
        """
        ex = TradeExecutor.__new__(TradeExecutor)
        signal = _ic_signal(credit=MIN_CREDIT, width=5.0)

        _, offset = combo_entry_limit(MIN_CREDIT, True)
        pre_fix_limit = _ladder_limit(MIN_CREDIT, offset, 0.25, is_credit=True)

        assert abs(pre_fix_limit) == pytest.approx(0.67), (
            "premise moved: step 0 no longer concedes 2.5c on a $0.70 signal, "
            f"got {pre_fix_limit}"
        )

        pending = PendingOrder(trade_id="T-NC", signal=signal, contracts=1,
                               base_price=MIN_CREDIT, full_offset=offset,
                               is_credit=True)
        assert ex._credit_floors_hold(pending, pre_fix_limit) is False, (
            "the reprice guard accepts the pre-W27 placement price, so this "
            "control no longer demonstrates the asymmetry"
        )

    async def test_a_floor_hugging_signal_is_no_longer_placed_below_the_floor(
        self, env, events
    ):
        """The unconditional case: fresh quote, no adverse move, at the floor."""
        sig = _ic_signal(credit=MIN_CREDIT, width=5.0)
        env.ib.quote = _nbbo_for(sig)

        trade_id = await env.executor.execute_signal(sig, 1)

        assert trade_id is not None, "a floor-compliant signal must still trade"
        assert len(env.ib.placed) == 1
        placed = abs(_placed_limits(env)[0])
        assert placed >= MIN_CREDIT - 1e-9, (
            f"placed at ${placed:.2f} against a ${MIN_CREDIT:.2f} floor — "
            "regress-live-06 has regressed"
        )
        assert _named(events, "credit_floor_clamped_at_placement"), (
            "the clamp must be visible in the log, not silent"
        )

    async def test_the_market_being_under_the_floor_refuses_the_entry(
        self, env, events
    ):
        """The mid itself cannot pay the economics -> the signal's premise is void.

        Within the 35% staleness tolerance, so the stale-signal gate does NOT
        fire first: this is the credit floor refusing on its own.
        """
        sig = _ic_signal(credit=MIN_CREDIT, width=5.0)
        env.ib.quote = _nbbo_for(SimpleNamespace(
            entry_price=0.56, strategy_name=sig.strategy_name))

        trade_id = await env.executor.execute_signal(sig, 1)

        assert trade_id is None
        assert env.ib.placed == [], "no order may reach the broker"
        refused = _named(events, "credit_floor_breached_at_placement_refused")
        assert len(refused) == 1
        assert refused[0]["log_level"] == "warning"

    async def test_a_healthy_signal_is_priced_exactly_as_before(self, env, events):
        """The clamp must not perturb the normal path."""
        sig = _ic_signal(credit=2.00, width=5.0)
        env.ib.quote = _nbbo_for(sig)

        trade_id = await env.executor.execute_signal(sig, 1)

        assert trade_id is not None
        _, offset = combo_entry_limit(2.00, True)
        expected = _ladder_limit(2.00, offset, 0.25, is_credit=True)
        assert _placed_limits(env)[0] == pytest.approx(round(expected, 2), abs=0.02)
        assert _named(events, "credit_floor_clamped_at_placement") == []
        assert _named(events, "credit_floor_breached_at_placement_refused") == []


class TestExitsAreUntouched:
    """The catastrophic version of this change, pinned.

    _execute_multi_leg serves exits too. The floors describe the economics that
    justify OPENING a position; applying them to a close would refuse or reprice
    the exit of a losing trade — the opposite of protection. A missed exit is
    the worst outcome this system has.
    """

    async def test_an_exit_below_the_floor_still_places(self, env, events):
        sig = _ic_signal(credit=0.20, width=5.0)   # far below the floor

        trade = await env.executor._execute_multi_leg(
            sig, 1, "T-W27-EXIT", is_exit=True)

        assert trade is not None, "an exit must never be blocked by the credit floor"
        assert len(env.ib.placed) == 1
        assert _named(events, "credit_floor_breached_at_placement_refused") == []
        assert _named(events, "credit_floor_clamped_at_placement") == []

    async def test_an_exit_is_not_repriced_by_the_floor(self, env):
        """The exit's limit must be whatever the exit pricing decided."""
        sig = _ic_signal(credit=0.20, width=5.0)

        await env.executor._execute_multi_leg(sig, 1, "T-W27-EXIT2", is_exit=True)

        assert abs(_placed_limits(env)[0]) < MIN_CREDIT, (
            "the exit price was raised to the entry credit floor"
        )


class TestFloorMinimumIsOneAuthority:

    def test_the_ratio_floor_binds_on_wide_wings(self, floors):
        ex = TradeExecutor.__new__(TradeExecutor)
        # 14-wide wings: 10% of width = $1.40 > the $0.70 absolute floor.
        assert ex._credit_floor_minimum(14.0) == pytest.approx(1.40)

    def test_the_absolute_floor_binds_on_narrow_wings(self, floors):
        ex = TradeExecutor.__new__(TradeExecutor)
        assert ex._credit_floor_minimum(5.0) == pytest.approx(MIN_CREDIT)

    def test_unknown_width_keeps_the_absolute_floor_only(self, floors):
        """Pre-W27 semantics of _credit_floors_hold, preserved exactly."""
        ex = TradeExecutor.__new__(TradeExecutor)
        assert ex._credit_floor_minimum(0.0) == pytest.approx(MIN_CREDIT)

    def test_width_from_legs_matches_the_pending_derivation(self, floors):
        """The split helper and its original caller must agree."""
        ex = TradeExecutor.__new__(TradeExecutor)
        ex._state = None
        sig = _ic_signal(credit=1.0, width=7.0)
        pending = PendingOrder(trade_id="T-W", signal=sig, contracts=1,
                               base_price=1.0, full_offset=0.1, is_credit=True)

        assert ex._wing_width_from_legs(sig.legs) == pytest.approx(7.0)
        assert ex._pending_wing_width(pending) == pytest.approx(7.0)


class TestPlacementAndRepriceCannotDrift:
    """CLASS GUARD: the wire price must always satisfy the guard that polices
    repricing.

    This is the general form of regress-live-06 — an entry priced without
    consulting the floor — rather than the single line that caused it. Any
    future path that prices an entry and skips the floor trips this.
    """

    @pytest.mark.parametrize("credit", [0.70, 0.71, 0.75, 0.90, 1.50, 3.00])
    @pytest.mark.parametrize("width", [5.0, 14.0])
    async def test_every_placed_entry_limit_satisfies_the_reprice_guard(
        self, env, credit, width
    ):
        sig = _ic_signal(credit=credit, width=width)
        env.ib.quote = _nbbo_for(sig)

        trade_id = await env.executor.execute_signal(sig, 1)

        if trade_id is None:
            assert env.ib.placed == []
            return

        placed = _placed_limits(env)[0]
        pending = PendingOrder(trade_id=trade_id, signal=sig, contracts=1,
                               base_price=abs(credit), full_offset=0.1,
                               is_credit=True)
        assert env.executor._credit_floors_hold(pending, placed), (
            f"placed {placed} for a ${credit:.2f}/${width:.0f}-wide condor, a "
            "price the reprice guard would refuse — placement and reprice have "
            "drifted apart again"
        )

    async def test_the_grid_above_is_not_all_refusals(self, env):
        """Anti-vacuity guard on the guard.

        The parametrized test passes trivially for any case it refuses. If a
        future change made the floor refuse everything, that suite would go green
        while placing nothing — so pin that the grid really does exercise the
        placement path.
        """
        placed = 0
        for width in (5.0, 14.0):
            for credit in (0.70, 0.71, 0.75, 0.90, 1.50, 3.00):
                env.ib.placed.clear()
                sig = _ic_signal(credit=credit, width=width)
                env.ib.quote = _nbbo_for(sig)
                if await env.executor.execute_signal(sig, 1) is not None:
                    placed += 1

        assert placed >= 6, (
            f"only {placed}/12 grid cases placed an order — the drift guard is "
            "close to vacuous"
        )

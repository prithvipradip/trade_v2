"""W48 (R24/R26 register, `trade-life-credit-econ-floors-signal-time-only`,
restate half) — capital-at-risk now reflects the ACTUAL fill, not the
signal's pre-fill guess.

`_position_capital_at_risk` (orchestrator.py) stamps `trade_maxloss_<id>`
(bot_state) and `trades.capital_at_risk` once, right after `execute_signal`
returns — order PLACEMENT time, before the real fill price is known (a
credit order can ladder for 45s+). Neither was ever restated once the fill
landed, even though `_update_trade_filled`/`_update_trade_partial` already
correct `trades.entry_price` to the real fill.

Every defined-risk credit strategy prices `max_loss = (max_width -
total_credit) * 100` (iron_condor.py:362-365); wing width is fixed between
signal and fill, only credit can move, so the true post-fill max_loss is
`signal_max_loss + (signal_entry_price - actual_fill_price) * 100` — no new
field needed. This does NOT hold for undefined-risk credit strategies
(short_strangle etc.), whose max_loss is a stress estimate, so the pure
function `restated_capital_at_risk` (strategies/base.py) is scoped to
defined-risk credit strategies only and returns None otherwise.

Group A tests the pure function directly (the algebra and the scope guard).
Group B drives the real `_update_trade_filled`/`_update_trade_partial`
(real StateManager) and confirms both write targets land the restated
figure. Group C proves the "no extra wiring needed" claim end to end,
through the build_smoke_orchestrator rig
(test_hot_path_smoke.py): `_sync_risk_manager_positions` re-reads the KV
fresh, so risk/manager.py's aggregate cap sees the restated number on the
very next sync with nothing else changed.

No inspect.getsource.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from ait.bot.state import StateManager
from ait.execution.executor import PendingOrder, TradeExecutor
from ait.strategies.base import Signal, SignalDirection, restated_capital_at_risk
from ait.utils.time import now_et
from tests.test_hot_path_smoke import build_smoke_orchestrator


# ---------------------------------------------------------------------------
# Group A -- the pure function
# ---------------------------------------------------------------------------

class TestRestatedCapitalAtRiskAlgebra:
    def test_matches_the_real_iron_condor_worked_example(self):
        """iron_condor.py:362-365: max_width = max(put_width, call_width);
        max_loss = (max_width - total_credit) * 100. A $10-wide condor
        signalled at 2.20 credit -> max_loss = (10 - 2.20) * 100 = 780."""
        max_width, signal_credit = 10.0, 2.20
        signal_max_loss = (max_width - signal_credit) * 100  # 780.0

        # Restating AT the signal's own price must reproduce max_loss
        # exactly -- the identity's own self-consistency check.
        identity = restated_capital_at_risk(
            "iron_condor", signal_max_loss, signal_credit, signal_credit, 1)
        assert identity == pytest.approx(signal_max_loss)

        # The finding's own worked extreme: a laddered fill at 1.39 (vs.
        # signal 2.20) -- the true max loss is worse than what was stamped.
        restated = restated_capital_at_risk(
            "iron_condor", signal_max_loss, signal_credit, 1.39, 1)
        assert restated == pytest.approx((max_width - 1.39) * 100)  # 861.0
        assert restated > signal_max_loss

    def test_quantity_multiplies_the_per_contract_figure(self):
        per_contract = restated_capital_at_risk("iron_condor", 780.0, 2.20, 2.20, 1)
        total = restated_capital_at_risk("iron_condor", 780.0, 2.20, 2.20, 3)
        assert total == pytest.approx(per_contract * 3)

    def test_floored_at_zero_rather_than_negative(self):
        """A fill capturing MORE credit than the wing width allows would
        make the raw identity negative -- the credit-floor clamp elsewhere
        should prevent this, but this function does not assume that."""
        restated = restated_capital_at_risk("iron_condor", 100.0, 2.0, 50.0, 1)
        assert restated == 0.0

    @pytest.mark.parametrize("strategy", ["short_strangle", "covered_call",
                                          "cash_secured_put", "short_straddle",
                                          "jade_lizard"])
    def test_undefined_risk_strategies_are_never_restated(self, strategy):
        """Their max_loss is a stress ESTIMATE, not width-minus-credit --
        this identity would be flatly wrong for them."""
        assert restated_capital_at_risk(strategy, 500.0, 2.0, 1.0, 1) is None

    def test_non_credit_strategies_are_never_restated(self):
        assert restated_capital_at_risk("long_call", 500.0, 2.0, 1.0, 1) is None

    def test_no_genuine_signal_time_value_is_never_restated(self):
        assert restated_capital_at_risk("iron_condor", 0.0, 2.0, 1.39, 1) is None
        assert restated_capital_at_risk("iron_condor", -1.0, 2.0, 1.39, 1) is None


# ---------------------------------------------------------------------------
# Group B -- wired into the real fill-booking functions
# ---------------------------------------------------------------------------

def _state(tmp_path) -> StateManager:
    return StateManager(db_path=tmp_path / "ait_state.db")


def _seed_trade(st: StateManager, trade_id: str, strategy: str = "iron_condor") -> None:
    with sqlite3.connect(st._db_path) as conn:
        conn.execute(
            "INSERT INTO trades (trade_id, symbol, strategy, direction, "
            "status, entry_time, entry_price, quantity, contract_type, legs) "
            "VALUES (?,?,?,?,?,?,?,1,'iron_condor',?)",
            (trade_id, "QQQ", strategy, "short", "pending",
             now_et().isoformat(), 2.20, json.dumps([{}] * 4)))
    st.insert_open_position(trade_id, "QQQ", "iron_condor", 1, 2.20)
    st.set_state(f"trade_maxloss_{trade_id}", "780.0")
    st.set_trade_capital_at_risk(trade_id, 780.0)


def _executor(st: StateManager) -> TradeExecutor:
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._state = st
    return ex


def _condor_legs() -> list[dict]:
    return [
        {"strike": 480.0, "right": "P", "action": "BUY", "expiry": "2026-10-16"},
        {"strike": 490.0, "right": "P", "action": "SELL", "expiry": "2026-10-16"},
        {"strike": 510.0, "right": "C", "action": "SELL", "expiry": "2026-10-16"},
        {"strike": 520.0, "right": "C", "action": "BUY", "expiry": "2026-10-16"},
    ]


def _pending(trade_id: str, strategy: str, max_loss: float, entry_price: float,
             contracts: int = 1) -> PendingOrder:
    signal = Signal(symbol="QQQ", strategy_name=strategy,
                    direction=SignalDirection.NEUTRAL, confidence=0.7,
                    entry_price=entry_price, max_loss=max_loss, quantity=1,
                    legs=_condor_legs())
    return PendingOrder(trade_id=trade_id, signal=signal, contracts=contracts)


class TestUpdateTradeFilledRestates:
    def test_a_worse_fill_raises_both_the_kv_and_the_column(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-FILL")
        pending = _pending("T-FILL", "iron_condor", max_loss=780.0, entry_price=2.20)
        ex = _executor(st)

        ex._update_trade_filled(pending, actual_price=1.39)

        expected = (10.0 - 1.39) * 100  # 861.0, matching the worked example
        assert float(st.get_state("trade_maxloss_T-FILL", "0")) == pytest.approx(expected)
        with sqlite3.connect(st._db_path) as conn:
            car = conn.execute(
                "SELECT capital_at_risk FROM trades WHERE trade_id='T-FILL'"
            ).fetchone()[0]
        assert car == pytest.approx(expected)

    def test_undefined_risk_strategy_is_left_exactly_as_it_was(self, tmp_path):
        st = _state(tmp_path)
        _seed_trade(st, "T-STRANGLE", strategy="short_strangle")
        pending = _pending("T-STRANGLE", "short_strangle", max_loss=500.0, entry_price=2.20)
        ex = _executor(st)

        ex._update_trade_filled(pending, actual_price=1.39)

        # Untouched -- still whatever the signal-time stamp wrote.
        assert st.get_state("trade_maxloss_T-STRANGLE", "") == "780.0"
        with sqlite3.connect(st._db_path) as conn:
            car = conn.execute(
                "SELECT capital_at_risk FROM trades WHERE trade_id='T-STRANGLE'"
            ).fetchone()[0]
        assert car == pytest.approx(780.0)

    def test_partial_fill_restates_from_the_actual_partial_quantity(self, tmp_path):
        """The placement-time stamp assumed the full REQUESTED size; a
        partial fill's real quantity is smaller (and can be priced
        differently) -- exactly the case the signal-time number cannot
        have accounted for."""
        st = _state(tmp_path)
        _seed_trade(st, "T-PARTIAL")
        # Requested 3 contracts, only 1 actually filled.
        pending = _pending("T-PARTIAL", "iron_condor", max_loss=780.0,
                           entry_price=2.20, contracts=3)
        ex = _executor(st)

        ex._update_trade_partial(pending, filled_qty=1, actual_price=1.39)

        expected = (10.0 - 1.39) * 100 * 1  # quantity=1, not 3
        assert float(st.get_state("trade_maxloss_T-PARTIAL", "0")) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Group C -- end to end: the aggregate-risk-cap consumer needs no new wiring
# ---------------------------------------------------------------------------

class TestSyncRiskManagerPicksUpTheRestatementForFree:
    @pytest.fixture
    def rig(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        smoke = build_smoke_orchestrator()
        try:
            yield smoke
        finally:
            smoke.restore()

    async def test_aggregate_cap_reads_the_restated_kv_on_the_next_sync(self, rig):
        orch = rig.orch
        _seed_trade(orch._state, "T-SYNC")
        pending = _pending("T-SYNC", "iron_condor", max_loss=780.0, entry_price=2.20)

        orch._executor._update_trade_filled(pending, actual_price=1.39)
        await orch._sync_risk_manager_positions()

        positions = orch._risk_manager._open_positions
        matched = [p for p in positions if p.get("symbol") == "QQQ"]
        assert matched, "the seeded trade must be visible to the risk manager"
        expected = (10.0 - 1.39) * 100
        assert matched[0]["max_loss"] == pytest.approx(expected), (
            "risk/manager.py must see the RESTATED figure with no extra "
            "wiring -- _sync_risk_manager_positions already re-reads the "
            "trade_maxloss_* KV fresh every cycle")

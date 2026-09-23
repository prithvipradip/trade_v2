"""W52a: a STOP exit rests for a shorter, configurable time before it is
cancelled for a re-price.

2026-09-21, QQQ: the short-strike touch stop took 25 minutes to fill
(8.67 unfilled, 9.76 unfilled, 10.29 filled at 9.94). Each unfilled
placement rested the hard-coded 300s of ``check_fills`` before the cancel
that lets the monitor re-trigger (and W33b escalate) — a rest sized for a
take-profit on a quiet book, applied to an exit whose whole point is to
get out. Every quote here is ~15 minutes old (delayed data), so a
marketable limit that has not filled in two minutes is not going to; the
next attempt, escalated, is the one that fills.

Two knobs on ``exit``: ``exit_pending_rest_seconds`` (300, the old
literal, for everything else) and ``exit_stop_pending_rest_seconds``
(120) for stop_loss / short_strike_touch / trailing_stop / breakeven_stop
exits. Both sites in ``check_fills`` — the working order and the
invisible-pending order — read them; the 900s zombie cap and the foreign-
order escalations are untouched.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ait.bot.state import TradeStatus
from ait.execution.executor import PendingExitOrder, TradeExecutor
from tests.test_order_lifecycle import (  # noqa: F401 — shared harness
    _ic_signal,
    _place,
    _trade_row,
    env,
)

pytestmark = pytest.mark.asyncio


def _settings(rest: float = 300.0, stop_rest: float = 120.0):
    return SimpleNamespace(exit=SimpleNamespace(
        exit_pending_rest_seconds=rest,
        exit_stop_pending_rest_seconds=stop_rest))


class TestTheRestIsChosenByExitReason:
    @pytest.mark.parametrize("reason", [
        "stop_loss (P&L: -45.0%)",
        "short_strike_touch (spot 598.10 vs short put 600.00)",
        "trailing_stop (P&L: 12.0%, peak: 30.0%, stop: 15.0%)",
        "breakeven_stop (P&L: 0.5%, peak: 22.0%)",
    ])
    def test_stop_exits_use_the_stop_rest(self, env, reason):
        env.executor._settings = _settings(rest=300.0, stop_rest=120.0)
        pe = PendingExitOrder("T1", reason, estimated_pnl=-100.0)
        assert env.executor._exit_rest_seconds(pe) == 120.0

    @pytest.mark.parametrize("reason", [
        "take_profit_short (P&L: 41.0%, target: 40.0%)",
        "macro_event_flatten",
        "dte_exit",
        "",
    ])
    def test_other_exits_use_the_general_rest(self, env, reason):
        env.executor._settings = _settings(rest=300.0, stop_rest=120.0)
        pe = PendingExitOrder("T1", reason, estimated_pnl=40.0)
        assert env.executor._exit_rest_seconds(pe) == 300.0

    def test_defaults_without_settings_are_the_shipped_values(self, env):
        env.executor._settings = None
        assert env.executor._exit_rest_seconds(
            PendingExitOrder("T1", "stop_loss (P&L: -40.0%)", -1.0)) == 120.0
        assert env.executor._exit_rest_seconds(
            PendingExitOrder("T1", "take_profit_short (P&L: 41.0%)", 1.0)) == 300.0

    def test_the_config_model_carries_both_knobs(self):
        from ait.config.settings import ExitConfig
        cfg = ExitConfig()
        assert cfg.exit_pending_rest_seconds == 300.0
        assert cfg.exit_stop_pending_rest_seconds == 120.0
        assert ExitConfig(exit_stop_pending_rest_seconds=45).exit_stop_pending_rest_seconds == 45.0
        with pytest.raises(ValueError):
            ExitConfig(exit_stop_pending_rest_seconds=5)  # below the floor

    def test_config_yaml_declares_them(self):
        from pathlib import Path

        import yaml
        live = Path(__file__).resolve().parents[1] / "config.yaml"
        data = yaml.safe_load(live.read_text(encoding="utf-8"))
        assert data["exit"]["exit_pending_rest_seconds"] == 300
        assert data["exit"]["exit_stop_pending_rest_seconds"] == 120


async def _working_exit(env, reason: str) -> tuple[str, int]:
    """Entry filled, one exit order working at the broker, registered."""
    trade_id, order_id = await _place(env, _ic_signal())
    env.ib.resolve(order_id, "Filled", avg_price=-0.95, filled=1)
    await env.executor.check_fills()
    assert _trade_row(env, trade_id).status == TradeStatus.FILLED
    assert env.state.update_trade_status(
        trade_id, TradeStatus.CLOSING,
        from_statuses=[TradeStatus.FILLED, TradeStatus.PARTIAL])
    exit_order = SimpleNamespace(orderId=0, totalQuantity=1, lmtPrice=1.55,
                                 action="BUY", orderType="LMT")
    exit_trade = env.ib.placeOrder(SimpleNamespace(secType="BAG", symbol="SPY"), exit_order)
    exit_oid = exit_trade.order.orderId
    env.executor.register_exit_order(exit_oid, trade_id, reason, estimated_pnl=-60.0)
    return trade_id, exit_oid


class TestCheckFillsHonoursTheRest:
    async def test_a_working_stop_exit_is_cancelled_after_the_stop_rest(self, env):
        env.executor._settings = _settings(rest=300.0, stop_rest=120.0)
        trade_id, exit_oid = await _working_exit(
            env, "short_strike_touch (spot 598.10 vs short put 600.00)")

        env.executor._pending_exit_orders[exit_oid].submitted_at -= 121
        await env.executor.check_fills()
        assert exit_oid in env.ib.cancelled_order_ids, (
            "a 121s-old stop exit was left working; pre-fix it rested 300s")
        assert exit_oid in env.executor._pending_exit_orders  # still tracked
        assert _trade_row(env, trade_id).status == TradeStatus.CLOSING

    async def test_a_working_take_profit_still_rests_the_general_time(self, env):
        env.executor._settings = _settings(rest=300.0, stop_rest=120.0)
        _trade_id, exit_oid = await _working_exit(
            env, "take_profit_short (P&L: 41.0%, target: 40.0%)")

        env.executor._pending_exit_orders[exit_oid].submitted_at -= 121
        await env.executor.check_fills()
        assert exit_oid not in env.ib.cancelled_order_ids, (
            "a take-profit was cancelled at 121s; only stops rest shorter")

        env.executor._pending_exit_orders[exit_oid].submitted_at -= 180
        await env.executor.check_fills()
        assert exit_oid in env.ib.cancelled_order_ids

    async def test_a_stop_exit_under_the_stop_rest_is_left_working(self, env):
        env.executor._settings = _settings(rest=300.0, stop_rest=120.0)
        _trade_id, exit_oid = await _working_exit(env, "stop_loss (P&L: -45.0%)")

        env.executor._pending_exit_orders[exit_oid].submitted_at -= 100
        await env.executor.check_fills()
        assert exit_oid not in env.ib.cancelled_order_ids

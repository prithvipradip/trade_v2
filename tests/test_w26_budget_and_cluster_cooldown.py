"""W26 — the entry-budget give-back and the cluster-scoped post-stop cooldown.

Two findings from register-triage batch 1, both on the live entry path:

  data-integrity-07  `stats.trades_taken += 1` fires when an order is PLACED and
      had NO decrement anywhere, so a placed-but-never-filled order consumed an
      intraday entry slot permanently. _get_trade_budget returns
      `min(budget, max_trades) - taken` and the first-hour budget is 2, so one
      phantom halved first-hour capacity for the rest of the session.
  logic-exit-risk-03  the post-stop cooldown asked `WHERE symbol=?`, so a SPY
      touch stop left QQQ and IWM free to re-enter the same move — while the
      bot's own SECTOR_GROUPS calls all three one ~0.95 cluster.

The cooldown fix is GRADUATED, not cluster-wide-for-a-full-day: the stopped
symbol keeps the full R12-B4 one-trading-day window, cluster siblings get
risk.post_stop_cluster_cooldown_minutes. The strict reading would darken the
whole book (the universe is one cluster) for the rest of the session AND all of
the next. Both halves of that contract are pinned below, as is the 0-disables
escape hatch — an untested escape hatch is not an escape hatch.

House rules: every test drives the real code path, no assertions on source
strings, a negative control per finding, and class guards that catch the NEXT
instance rather than only this one.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()

import ait.bot.orchestrator as om  # noqa: E402
from ait.bot.state import DailyStats, StateManager, TradeStatus  # noqa: E402
from ait.config.settings import load_settings  # noqa: E402
from ait.execution.reconciler import (  # noqa: E402
    NEVER_FILLED_EXIT_REASON,
    PositionReconciler,
)
from ait.utils.time import now_et  # noqa: E402

TOUCH_STOP = "short_strike_touch (spot 123.45)"
STOP_LOSS = "stop_loss (-250.00)"

CONDOR_LEGS = [
    {"right": "P", "strike": 600.0, "action": "SELL", "expiry": "2026-12-18"},
    {"right": "P", "strike": 595.0, "action": "BUY", "expiry": "2026-12-18"},
    {"right": "C", "strike": 640.0, "action": "SELL", "expiry": "2026-12-18"},
    {"right": "C", "strike": 645.0, "action": "BUY", "expiry": "2026-12-18"},
]


# ---------------------------------------------------------------------------
# data-integrity-07 — the entry-budget give-back
# ---------------------------------------------------------------------------

def _pending_trade(trade_id="T-PHANTOM", symbol="SPY", age_min=40, entry_day_offset=0):
    """A PENDING trade old enough for the sweep. entry_day_offset shifts the DAY."""
    stamp = now_et() - timedelta(minutes=age_min) - timedelta(days=entry_day_offset)
    return SimpleNamespace(
        trade_id=trade_id, symbol=symbol, strategy="iron_condor",
        contract_type="iron_condor", quantity=1, entry_price=4.24,
        expiry="2026-12-18", strike=None, status=TradeStatus.PENDING,
        entry_time=stamp.isoformat(), legs=json.dumps(CONDOR_LEGS),
    )


def _reconciler(state):
    rec = PositionReconciler.__new__(PositionReconciler)
    rec._state = state
    rec._ibkr = MagicMock()
    # Authoritative flat book: no legs live, no working order -> the sweep books.
    rec._live_ibkr_leg_keys = lambda: {("XXX", "P", 1.0, "20261218")}
    rec._cancel_working_entry_order = lambda tr: False
    return rec


def _state_with(trade, taken_by_date: dict[date, int], close_returns=True):
    """MagicMock state whose daily_stats behave like real per-date rows."""
    rows: dict[date, DailyStats] = {}
    for d, n in taken_by_date.items():
        s = DailyStats(date=d)
        s.trades_taken = n
        rows[d] = s

    state = MagicMock()
    state.get_open_trades.return_value = [trade]
    state.close_trade.return_value = close_returns

    def _get(d=None):
        d = d or now_et().date()
        if d not in rows:
            s = DailyStats(date=d)
            s.trades_taken = 0
            rows[d] = s
        return rows[d]

    state.get_daily_stats.side_effect = _get
    state.update_daily_stats.side_effect = lambda s: rows.__setitem__(s.date, s)
    state._rows = rows
    return state


class TestPhantomOrderReturnsItsBudgetSlot:

    def test_negative_control_the_slot_was_permanently_consumed_pre_fix(self):
        """NEGATIVE CONTROL: without the refund the counter stays spent.

        Runs the real sweep with the refund disabled — exactly the pre-W26 code
        path — and asserts the phantom's slot is still gone afterwards. The
        premise (the sweep really booked a never-filled close) is asserted too,
        so this cannot pass by not sweeping at all.
        """
        today = now_et().date()
        trade = _pending_trade()
        state = _state_with(trade, {today: 1})
        rec = _reconciler(state)
        rec._refund_entry_budget = lambda tr: None  # pre-W26 behaviour

        assert rec._sweep_stale_pending() == 1

        booked = state.close_trade.call_args.kwargs
        assert booked["exit_reason_detailed"] == NEVER_FILLED_EXIT_REASON, (
            "premise broken: the sweep did not book a never-filled close, so "
            "this control tests nothing"
        )
        assert state._rows[today].trades_taken == 1, (
            "the pre-W26 defect: a never-filled order kept its intraday entry "
            "slot forever"
        )

    def test_the_slot_comes_back(self):
        """The fix, through the real sweep."""
        today = now_et().date()
        state = _state_with(_pending_trade(), {today: 1})

        assert _reconciler(state)._sweep_stale_pending() == 1

        assert state._rows[today].trades_taken == 0

    def test_it_refunds_the_trades_own_day_not_today(self):
        """The fail-open this fix must not create.

        reconcile() runs on every startup, so a phantom placed yesterday is
        typically swept today. Crediting TODAY would hand back a slot today never
        spent — and `taken = -1` makes _get_trade_budget return max_trades + 1,
        i.e. MORE entries than the daily cap allows.
        """
        today = now_et().date()
        yesterday = today - timedelta(days=1)
        trade = _pending_trade(entry_day_offset=1)
        state = _state_with(trade, {yesterday: 1, today: 2})

        assert _reconciler(state)._sweep_stale_pending() == 1

        assert state._rows[yesterday].trades_taken == 0, "yesterday should be refunded"
        assert state._rows[today].trades_taken == 2, (
            "today's budget was altered by a phantom that belongs to yesterday — "
            "this is a fail-open on the daily cap"
        )

    def test_a_lost_close_cas_refunds_nothing(self):
        """close_trade returning False means the row was promoted to FILLED.

        The order actually filled, so refunding its slot would be worse than the
        bug being fixed. Pre-W26 the return value was discarded entirely.
        """
        today = now_et().date()
        state = _state_with(_pending_trade(), {today: 1}, close_returns=False)

        assert _reconciler(state)._sweep_stale_pending() == 0, (
            "a refused CAS must not be counted as a close"
        )
        assert state._rows[today].trades_taken == 1, (
            "refunded a slot for an order that FILLED"
        )

    def test_the_counter_never_goes_negative(self):
        """A floor is only safe in the direction that cannot create permission."""
        today = now_et().date()
        state = _state_with(_pending_trade(), {today: 0})

        assert _reconciler(state)._sweep_stale_pending() == 1

        assert state._rows[today].trades_taken == 0
        state.update_daily_stats.assert_not_called()

    def test_an_unparseable_entry_time_refunds_nothing_and_still_books(self):
        """W21's lesson: a bad row must not abort the sweep, nor go unlogged."""
        today = now_et().date()
        trade = _pending_trade()
        state = _state_with(trade, {today: 1})
        rec = _reconciler(state)
        # Booked fine (the sweep parsed it for age), then corrupted before refund.
        original = rec._refund_entry_budget

        def _corrupt(tr):
            tr.entry_time = "not-a-timestamp"
            return original(tr)

        rec._refund_entry_budget = _corrupt

        assert rec._sweep_stale_pending() == 1, "the close must still be booked"
        assert state._rows[today].trades_taken == 1


class TestBudgetRoundTripThroughTheRealMethod:

    def test_get_trade_budget_recovers_the_slot(self, tmp_path, monkeypatch):
        """End-to-end on a REAL StateManager and the real _get_trade_budget."""
        monkeypatch.chdir(tmp_path)
        state = StateManager(tmp_path / "state.db")
        today = now_et().date()

        stats = state.get_daily_stats(today)
        stats.trades_taken = 2
        state.update_daily_stats(stats)

        orch = om.TradingOrchestrator.__new__(om.TradingOrchestrator)
        orch._settings = load_settings(str(CONFIG_PATH))
        orch._state = state
        spent = orch._get_trade_budget()

        rec = PositionReconciler.__new__(PositionReconciler)
        rec._state = state
        rec._ibkr = MagicMock()
        rec._refund_entry_budget(_pending_trade())

        assert orch._get_trade_budget() == spent + 1, (
            "the refunded slot did not reach the real budget calculation"
        )
        assert state.get_daily_stats(today).trades_taken == 1


class TestRefundCannotSilentlyStopBeingCalled:
    """Class guard: pin the WIRING, not just the helper.

    The helper can be perfect and the bug still live if the sweep stops calling
    it. This fails if the never-filled booking path ever books without refunding.
    """

    def test_every_never_filled_booking_refunds(self):
        today = now_et().date()
        calls = []
        state = _state_with(_pending_trade(), {today: 3})
        rec = _reconciler(state)
        real = rec._refund_entry_budget
        rec._refund_entry_budget = lambda tr: (calls.append(tr.trade_id), real(tr))[1]

        booked = rec._sweep_stale_pending()

        assert booked == 1
        assert len(calls) == booked, (
            f"{booked} never-filled close(s) booked but {len(calls)} refund(s) — "
            "a booking path skips the budget give-back"
        )


# ---------------------------------------------------------------------------
# logic-exit-risk-03 — the cluster-scoped, graduated cooldown
# ---------------------------------------------------------------------------

CONFIG_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "config.yaml"


def _cooldown_db(tmp_path, rows):
    """rows = [(symbol, exit_reason, minutes_ago)] in a real trades table."""
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(tmp_path / "data" / "ait_state.db")
    con.execute("CREATE TABLE IF NOT EXISTS trades (symbol TEXT, status TEXT, "
                "exit_reason_detailed TEXT, exit_time TEXT)")
    for sym, reason, mins in rows:
        con.execute(
            "INSERT INTO trades VALUES (?,?,?,?)",
            (sym, "closed", reason, (now_et() - timedelta(minutes=mins)).isoformat()),
        )
    con.commit()
    con.close()


def _orch(sibling_minutes: int):
    o = om.TradingOrchestrator.__new__(om.TradingOrchestrator)
    settings = load_settings(str(CONFIG_PATH))
    settings.risk.post_stop_cluster_cooldown_minutes = sibling_minutes
    o._settings = settings
    return o


@pytest.fixture(autouse=True)
def _clear_cutoff_cache():
    om._COOLDOWN_CUTOFF_CACHE.clear()
    yield
    om._COOLDOWN_CUTOFF_CACHE.clear()


class TestClusterSiblingCooldown:

    def test_a_spy_stop_now_blocks_qqq(self, tmp_path, monkeypatch):
        """The defect: pre-W26 this returned None and QQQ re-entered the move."""
        _cooldown_db(tmp_path, [("SPY", TOUCH_STOP, 30)])
        monkeypatch.chdir(tmp_path)

        block = _orch(120)._post_stop_cooldown_until("QQQ")

        assert block is not None, (
            "a SPY touch stop 30 minutes ago left QQQ free to enter the same "
            "move — logic-exit-risk-03 has regressed"
        )
        assert block.symbol == "SPY", (
            "the verdict must name the BLOCKING symbol so the operator can tell "
            "why QQQ was refused"
        )

    def test_stop_loss_spelling_also_blocks_the_cluster(self, tmp_path, monkeypatch):
        _cooldown_db(tmp_path, [("IWM", STOP_LOSS, 15)])
        monkeypatch.chdir(tmp_path)
        assert _orch(120)._post_stop_cooldown_until("SPY") is not None

    def test_the_graduated_contract_both_halves(self, tmp_path, monkeypatch):
        """Sibling clears after its window; the stopped symbol does NOT.

        This is the whole point of the graduated design — pinning only the block
        would let a later 'simplification' apply the full trading-day window
        cluster-wide and darken the book.
        """
        _cooldown_db(tmp_path, [("SPY", TOUCH_STOP, 300)])  # 5h ago
        monkeypatch.chdir(tmp_path)
        orch = _orch(120)

        assert orch._post_stop_cooldown_until("QQQ") is None, (
            "sibling still blocked 5h after a stop with a 120-minute window"
        )
        self_block = orch._post_stop_cooldown_until("SPY")
        assert self_block is not None and self_block.symbol == "SPY", (
            "the stopped symbol must keep its full one-trading-day window"
        )

    def test_a_different_cluster_is_unaffected(self, tmp_path, monkeypatch):
        """Preserves the real intent of the retired test_other_symbol_unaffected.

        That test asserted a QQQ stop does not block SPY, which is the defect —
        they are cluster siblings. What it was actually protecting is this: an
        UNRELATED symbol must not be swept up. GLD is in no SECTOR_GROUPS set.
        """
        _cooldown_db(tmp_path, [("GLD", TOUCH_STOP, 10)])
        monkeypatch.chdir(tmp_path)
        assert _orch(120)._post_stop_cooldown_until("QQQ") is None

    def test_profit_exits_never_block(self, tmp_path, monkeypatch):
        _cooldown_db(tmp_path, [("SPY", "take_profit (P&L: 41%)", 10)])
        monkeypatch.chdir(tmp_path)
        assert _orch(120)._post_stop_cooldown_until("QQQ") is None

    def test_zero_disables_sibling_scoping_exactly(self, tmp_path, monkeypatch):
        """The escape hatch, proven rather than assumed.

        0 must reproduce the pre-W26 per-symbol behaviour precisely: the sibling
        goes free, the stopped symbol stays blocked.
        """
        _cooldown_db(tmp_path, [("SPY", TOUCH_STOP, 30)])
        monkeypatch.chdir(tmp_path)
        orch = _orch(0)

        assert orch._post_stop_cooldown_until("QQQ") is None
        assert orch._post_stop_cooldown_until("SPY") is not None

    def test_the_most_recent_blocking_stop_is_reported(self, tmp_path, monkeypatch):
        """With several cluster stops open, name the freshest cause."""
        _cooldown_db(tmp_path, [("SPY", TOUCH_STOP, 90), ("IWM", TOUCH_STOP, 10)])
        monkeypatch.chdir(tmp_path)

        block = _orch(120)._post_stop_cooldown_until("QQQ")

        assert block is not None and block.symbol == "IWM"

    def test_open_rows_are_still_ignored(self, tmp_path, monkeypatch):
        """A filled/open row with a stale reason string must not block."""
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(tmp_path / "data" / "ait_state.db")
        con.execute("CREATE TABLE trades (symbol TEXT, status TEXT, "
                    "exit_reason_detailed TEXT, exit_time TEXT)")
        con.execute("INSERT INTO trades VALUES (?,?,?,?)",
                    ("SPY", "filled", TOUCH_STOP, now_et().isoformat()))
        con.commit()
        con.close()
        monkeypatch.chdir(tmp_path)
        assert _orch(120)._post_stop_cooldown_until("QQQ") is None


class TestClusterAuthorityHasOneOwner:
    """logic-exit-risk-03's structural half: "same macro bet" must mean one thing."""

    def test_the_risk_manager_and_the_cooldown_agree(self):
        """Both gates resolve the cluster through the same function.

        They disagreed by construction before: the cooldown had no notion of a
        cluster at all, and the risk manager re-implemented the SECTOR_GROUPS
        intersection locally.
        """
        from ait.risk.correlation import cluster_symbols
        from ait.risk.manager import RiskManager

        rm = RiskManager.__new__(RiskManager)
        cluster = cluster_symbols("SPY")
        siblings = sorted(cluster - {"SPY"})

        assert rm._count_correlated_positions("SPY", siblings) == len(siblings)
        assert rm._count_correlated_positions("SPY", ["GLD", "TLT"]) == 0

    def test_same_sector_cluster_semantics_are_exactly_preserved(self):
        """The delegation must not have widened _same_sector_cluster.

        cluster_symbols() always contains the symbol itself (callers want a
        usable set), so a naive `sym2 in cluster_symbols(sym1)` would make
        ("GLD", "GLD") True where the original is False for a symbol in no group.
        """
        from ait.risk.correlation import CorrelationGuard as CG

        assert CG._same_sector_cluster("SPY", "IWM") is True
        assert CG._same_sector_cluster("SPY", "GLD") is False
        assert CG._same_sector_cluster("SPY", "SPY") is True
        assert CG._same_sector_cluster("GLD", "GLD") is False

    def test_a_symbol_in_no_group_is_its_own_cluster(self):
        from ait.risk.correlation import cluster_symbols

        assert cluster_symbols("GLD") == {"GLD"}
        assert cluster_symbols("SPY") >= {"SPY", "QQQ", "IWM"}

"""R16 broker/executor fixes — the audit findings whose file is
executor.py or ibkr_client.py.

Every test asserts the POST-fix contract and fails against the pre-fix code
by construction:

[A] executions ledger stamped the COMBO mid onto every per-leg row, so the
    go-live slippage gate compared |per-leg price - combo mid| and read 49.9%
    (gate <= 8%) on fills that were actually FAVOURABLE by 1.6%.
[B] a foreign-clientId exit order is a frozen 'Submitted' snapshot in
    ib_insync's cache — present-forever pinned the trade in CLOSING with an
    uncancellable order, absent let the 900s zombie cap revert CLOSING->FILLED
    while the original close was still working (duplicate close = reversal).
[C] a reconnect whose open-orders sync silently timed out left the caches
    empty, which the entry path read as "every working order was cancelled".
[D] foreign_open_order_ids was add-only, so a dead stashed order forced
    "pending" forever and re-attempted an impossible cancel every pass.
[E] multi-account sessions: the paper/live gate validated managedAccounts()[0]
    only, account values blended across accounts, and no order pinned
    order.account.
[F] get_account_values threw away an already-parsed ExchangeRate USD rate.
[G] _reconnect had no concurrency guard and no post-backoff liveness check,
    so a second waiter tore down the session the first had just restored.

Pattern notes: see tests/fakes.py — real executor + real state where the DB
matters, SimpleNamespace broker shapes elsewhere (never MagicMock for broker
objects: it invents attributes and hides shape drift).
"""

from __future__ import annotations

import asyncio
import statistics
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.state import TradeStatus
from ait.broker.ibkr_client import IBKRClient
from ait.execution import executor as executor_mod
from ait.execution.executor import TradeExecutor

from tests.fakes import FakeIB, FakeIBKRClient

# The referee's go-live gate (scripts/shadow_referee.py [7]).
GATE_SLIP_PCT = 8.0
# IBKR's generic combo (BAG) contract id — an IBKR protocol constant, spelled
# out here (not imported) so this file still COLLECTS against the pre-fix
# executor and every test reports its own failure.
BAG_CON_ID = 28812380


def test_executor_agrees_on_the_bag_contract_id():
    assert getattr(executor_mod, "BAG_CON_ID", None) == BAG_CON_ID


# ---------------------------------------------------------------------------
# Broker-aware client: FakeIBKRClient + the R16 authoritative probe
# ---------------------------------------------------------------------------

class BrokerAwareClient(FakeIBKRClient):
    """FakeIBKRClient plus the all-clients open-order probe the R16 fixes
    depend on.

    ``broker_open_ids`` is the BROKER's own answer, independent of the
    session cache: a set means "these ids are working under some clientId";
    None means "the broker could not be asked" — the UNKNOWN every caller
    must defer on rather than act.
    """

    def __init__(self, ib: FakeIB | None = None, connected: bool = True,
                 client_id: int = 105) -> None:
        super().__init__(ib, connected)
        self.client_id = client_id
        self.broker_open_ids: set[int] | None = set()
        self.probe_calls = 0

    async def get_all_open_order_ids(self, timeout: float = 8.0) -> set[int] | None:
        self.probe_calls += 1
        if self.broker_open_ids is None:
            return None
        return set(self.broker_open_ids)

    def prune_foreign_open_orders(self, broker_open_ids):
        # exercise the REAL implementation against this duck-typed client
        return IBKRClient.prune_foreign_open_orders(self, broker_open_ids)


def _frozen_foreign_trade(order_id: int, client_id: int = 1) -> SimpleNamespace:
    """An order owned by ANOTHER clientId as ib_insync caches it: a snapshot
    fed in by reqAllOpenOrders, pinned at 'Submitted', never updated again."""
    return SimpleNamespace(
        order=SimpleNamespace(orderId=order_id, permId=order_id * 10 + 7,
                              clientId=client_id, totalQuantity=1,
                              lmtPrice=0.55, action="BUY", orderType="LMT"),
        orderStatus=SimpleNamespace(status="Submitted", filled=0, remaining=1,
                                    avgFillPrice=0.0),
        contract=SimpleNamespace(secType="BAG", symbol="SPY"),
        fills=[],
        log=[],
    )


def _executor(client, state, order_timeout: int = 240) -> TradeExecutor:
    return TradeExecutor(ibkr_client=client, state=state,
                         circuit_breaker=MagicMock(), order_timeout=order_timeout)


def _closing_state(trade_id: str = "T-CLOSING") -> MagicMock:
    """StateManager mock whose trade row is CLOSING (the status the exit
    tracker exists for)."""
    state = MagicMock()
    state.get_trade_by_id.return_value = SimpleNamespace(
        trade_id=trade_id, status=TradeStatus.CLOSING, quantity=1,
        entry_price=4.31, contract_type="iron_condor", strategy="iron_condor")
    return state


def _register_exit(executor: TradeExecutor, order_id: int, trade_id: str,
                   age_seconds: float) -> None:
    executor.register_exit_order(order_id, trade_id, "take_profit_50pct",
                                 estimated_pnl=0.0)
    executor._pending_exit_orders[order_id].submitted_at -= age_seconds


# ===========================================================================
# [A] ledger slippage semantics (_sweep_executions)
# ===========================================================================

def _fill(exec_id: str, con_id: int, side: str, price: float,
          shares: float = 1.0, commission: float = 0.65,
          sec_type: str = "OPT") -> SimpleNamespace:
    return SimpleNamespace(
        execution=SimpleNamespace(execId=exec_id, side=side, shares=shares,
                                  price=price,
                                  time="2026-08-04T14:51:37+00:00"),
        commissionReport=SimpleNamespace(commission=commission, realizedPNL=0.0),
        contract=SimpleNamespace(conId=con_id, secType=sec_type),
    )


# The real 2026-08-04 SPY condor (T-20260804-105137-298875): live combo mid
# 4.24, legs netting a 4.31 credit, IBKR's BAG summary row at -4.31.
SPY_LEGS = [
    _fill("0001.01", 792205417, "SLD", 2.09),
    _fill("0001.02", 792205363, "SLD", 3.01),
    _fill("0001.03", 792205400, "BOT", 0.19),
    _fill("0001.04", 792205377, "BOT", 0.60),
]
SPY_BAG = _fill("0001.05", BAG_CON_ID, "BOT", -4.31, commission=0.0,
                sec_type="BAG")
SPY_MID = 4.24
SPY_FILLED_CREDIT = 4.31   # trades.entry_price after the fill true-up


def _sweep(fills: list, sec_type: str = "BAG") -> list[dict]:
    """Run the real _sweep_executions over one broker trade; return the rows
    it wrote to the executions ledger."""
    rows: list[dict] = []
    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=555, permId=99123),
        contract=SimpleNamespace(symbol="SPY", secType=sec_type),
        fills=fills,
    )
    ex = TradeExecutor.__new__(TradeExecutor)
    ex._state = SimpleNamespace(record_execution=lambda **kw: rows.append(kw))
    ex._ibkr = SimpleNamespace(ib=SimpleNamespace(trades=lambda: [trade]))
    ex._order_trade_map = {555: "T-SPY"}
    ex._pending_orders = {}
    ex._pending_exit_orders = {}
    ex._order_ctx_map = {555: (SPY_MID, SPY_MID, 0.10)}
    ex._sweep_executions()
    return rows


def _referee_slippage_pct(rows: list[dict], entry_price: float) -> list[float]:
    """scripts/shadow_referee.py check [7], verbatim: rows with live_mid > 0,
    abs(price - live_mid) / trades.entry_price * 100."""
    return [abs(r["price"] - r["live_mid"]) / entry_price * 100
            for r in rows if r["live_mid"] > 0]


class TestLedgerSlippageSemantics:
    def test_sweep_wrote_every_execution(self):
        rows = _sweep(SPY_LEGS + [SPY_BAG])
        assert len(rows) == 5, "the sweep must still persist all 5 executions"

    def test_leg_rows_carry_no_combo_context(self):
        # The combo mid describes the WHOLE structure; stamping it on a leg
        # row invented 49.9% / 94.0% / 198.4% "slippage" out of nothing.
        rows = _sweep(SPY_LEGS + [SPY_BAG])
        legs = [r for r in rows if r["con_id"] != BAG_CON_ID]
        assert len(legs) == 4
        for r in legs:
            assert r["live_mid"] == 0.0
            assert r["signal_price"] == 0.0
            assert r["nbbo_spread"] == 0.0

    def test_leg_rows_keep_raw_signed_broker_prices(self):
        # P&L reconstruction (referee/restate fill_groups) reads these; they
        # must remain the untouched broker values.
        rows = _sweep(SPY_LEGS + [SPY_BAG])
        by_id = {r["exec_id"]: r for r in rows}
        assert by_id["0001.01"]["price"] == pytest.approx(2.09)
        assert by_id["0001.03"]["price"] == pytest.approx(0.19)
        assert by_id["0001.01"]["side"] == "SLD"

    def test_bag_row_is_the_only_slippage_row_and_shares_entry_price_units(self):
        rows = _sweep(SPY_LEGS + [SPY_BAG])
        bag = [r for r in rows if r["con_id"] == BAG_CON_ID]
        assert len(bag) == 1
        bag = bag[0]
        assert bag["live_mid"] == pytest.approx(SPY_MID)
        assert bag["signal_price"] == pytest.approx(SPY_MID)
        # magnitude, matching trades.entry_price's unsigned convention — the
        # raw as-defined -4.31 made ABS(price - live_mid) read 198%.
        assert bag["price"] == pytest.approx(SPY_FILLED_CREDIT)

    def test_referee_gate_now_measures_true_slippage(self):
        rows = _sweep(SPY_LEGS + [SPY_BAG])
        vals = _referee_slippage_pct(rows, SPY_FILLED_CREDIT)
        assert len(vals) == 1, "exactly one comparable row per combo order"
        # true slip: filled 4.31 vs 4.24 mid = 0.07 = 1.62% of credit
        assert vals[0] == pytest.approx(1.62, abs=0.05)
        assert statistics.median(vals) <= GATE_SLIP_PCT

    def test_pre_fix_semantics_would_break_the_gate(self):
        # The discriminator, stated explicitly: stamping the combo mid on
        # every row (and keeping the signed BAG price) is what produced the
        # 49.9% median BREAK on fills that were actually favourable.
        pre_fix = [{"price": f.execution.price, "live_mid": SPY_MID}
                   for f in SPY_LEGS + [SPY_BAG]]
        vals = _referee_slippage_pct(pre_fix, SPY_FILLED_CREDIT)
        assert statistics.median(vals) > GATE_SLIP_PCT
        assert max(vals) > 190  # the BAG row alone: |-4.31 - 4.24| / 4.31

    def test_single_leg_orders_keep_their_context(self):
        # Not a combo: price and mid are the same unit, nothing to separate.
        single = _fill("0002.01", 792205417, "BOT", 3.30, sec_type="OPT")
        rows = _sweep([single], sec_type="OPT")
        assert len(rows) == 1
        assert rows[0]["live_mid"] == pytest.approx(SPY_MID)
        assert rows[0]["price"] == pytest.approx(3.30)


# ===========================================================================
# [B] foreign / frozen exit orders
# ===========================================================================

class TestForeignExitOrderResolution:
    async def test_frozen_working_foreign_order_is_never_cancelled_or_reverted(self):
        """A frozen 'Submitted' snapshot pinned still_open True forever: the
        >300s branch fired a cross-client cancel IBKR always rejects, and the
        trade sat in CLOSING with an order nobody could reprice."""
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=105)
        ib.open_trades.append(_frozen_foreign_trade(4242, client_id=1))
        ib.all_trades.append(ib.open_trades[0])
        client.broker_open_ids = {4242}  # broker: still working under id 1
        state = _closing_state()
        ex = _executor(client, state)
        _register_exit(ex, 4242, "T-CLOSING", age_seconds=400)

        filled, exits = await ex.check_fills()

        assert (filled, exits) == ([], [])
        assert ib.cancelled_order_ids == [], "cross-client cancel must not be attempted"
        assert 4242 in ex._pending_exit_orders, "tracking must continue"
        state.transition.assert_not_called()
        assert (4242, "exit_order_foreign_uncancellable") in ex._stuck_order_pages

    async def test_operator_is_paged_once_not_every_pass(self):
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=105)
        ib.open_trades.append(_frozen_foreign_trade(4242, client_id=1))
        client.broker_open_ids = {4242}
        ex = _executor(client, _closing_state())
        _register_exit(ex, 4242, "T-CLOSING", age_seconds=400)

        for _ in range(3):
            ex._broker_open_ids_cache = None  # force a fresh probe each pass
            await ex.check_fills()
        pages = [p for p in ex._stuck_order_pages if p[0] == 4242]
        assert len(pages) == 1

    async def test_foreign_order_gone_from_broker_holds_closing(self):
        """Filled-vs-cancelled is unobservable for a foreign order, and both
        guesses are money bugs (phantom close / re-armed duplicate close):
        hold CLOSING and hand it to reconcile."""
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=105)
        ib.open_trades.append(_frozen_foreign_trade(4242, client_id=1))
        client.broker_open_ids = set()  # broker: not working anywhere
        state = _closing_state()
        ex = _executor(client, state)
        _register_exit(ex, 4242, "T-CLOSING", age_seconds=1000)

        filled, exits = await ex.check_fills()

        assert (filled, exits) == ([], [])
        state.transition.assert_not_called()
        assert 4242 in ex._pending_exit_orders
        assert (4242, "exit_order_foreign_gone_unbookable") in ex._stuck_order_pages

    async def test_unknown_broker_state_defers_everything(self):
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=105)
        ib.open_trades.append(_frozen_foreign_trade(4242, client_id=1))
        client.broker_open_ids = None  # request failed / disconnected
        state = _closing_state()
        ex = _executor(client, state)
        _register_exit(ex, 4242, "T-CLOSING", age_seconds=1000)

        await ex.check_fills()

        state.transition.assert_not_called()
        assert ib.cancelled_order_ids == []
        assert 4242 in ex._pending_exit_orders
        assert (4242, "exit_order_foreign_state_unknown") in ex._stuck_order_pages

    async def test_tracker_released_once_reconcile_settles_the_trade(self):
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=105)
        ib.open_trades.append(_frozen_foreign_trade(4242, client_id=1))
        client.broker_open_ids = set()
        state = _closing_state()
        state.get_trade_by_id.return_value = SimpleNamespace(
            trade_id="T-CLOSING", status=TradeStatus.CLOSED, quantity=1,
            entry_price=4.31, contract_type="iron_condor", strategy="iron_condor")
        ex = _executor(client, state)
        _register_exit(ex, 4242, "T-CLOSING", age_seconds=1000)

        await ex.check_fills()

        assert 4242 not in ex._pending_exit_orders
        state.transition.assert_not_called()

    def test_adopted_previous_session_order_is_classified_at_adoption(self):
        """The R12 stash cannot cover adoption: _should_stash_foreign is
        skipped on the first connect of a restarted process, which is exactly
        when the reconciler adopts the dead process's exit order."""
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=105)
        ib.all_trades.append(_frozen_foreign_trade(777, client_id=1))
        ex = _executor(client, _closing_state())

        ex.adopt_exit_order(777, "T-CLOSING", "reconcile_working_exit")

        assert 777 in ex._foreign_exit_orders
        assert 777 in client.foreign_open_order_ids
        assert 777 in ex._pending_exit_orders

    async def test_foreign_classification_survives_stash_pruning(self):
        """Pruning the stash (finding D) must not silently downgrade an
        unobservable exit order back onto the 900s revert path."""
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=105)
        client.foreign_open_order_ids.add(4242)   # stashed at reconnect
        client.broker_open_ids = set()            # broker: gone -> prune
        state = _closing_state()
        ex = _executor(client, state)
        _register_exit(ex, 4242, "T-CLOSING", age_seconds=1000)

        await ex.check_fills()

        assert 4242 not in client.foreign_open_order_ids, "stash pruned"
        assert 4242 in ex._foreign_exit_orders, "classification is sticky"
        state.transition.assert_not_called()
        assert 4242 in ex._pending_exit_orders


# ===========================================================================
# [B'] 900s zombie revert must consult the broker
# ===========================================================================

class TestZombieRevertGuard:
    def _env(self, *, broker_open_ids, stash=(), client_id=1):
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=client_id)
        client.broker_open_ids = broker_open_ids
        for oid in stash:
            client.foreign_open_order_ids.add(oid)
        state = _closing_state()
        ex = _executor(client, state)
        _register_exit(ex, 901, "T-CLOSING", age_seconds=901)
        return ib, client, state, ex

    async def test_no_revert_while_the_close_still_works_at_the_broker(self):
        """The exact finding: the stashed foreign close is working under the
        old clientId, invisible here. Reverting CLOSING->FILLED re-arms the
        monitor and a SECOND close goes out beside the first."""
        _ib, _client, state, ex = self._env(broker_open_ids={901}, stash=[901])
        await ex.check_fills()
        state.transition.assert_not_called()
        assert 901 in ex._pending_exit_orders

    async def test_no_revert_when_the_cache_was_wiped_but_the_order_lives(self):
        # Not foreign, simply invisible: a disconnect wiped ib_insync's cache.
        _ib, _client, state, ex = self._env(broker_open_ids={901})
        await ex.check_fills()
        state.transition.assert_not_called()
        assert 901 in ex._pending_exit_orders
        assert (901, "exit_zombie_revert_deferred") in ex._stuck_order_pages

    async def test_no_revert_when_the_broker_cannot_be_asked(self):
        _ib, _client, state, ex = self._env(broker_open_ids=None)
        await ex.check_fills()
        state.transition.assert_not_called()
        assert 901 in ex._pending_exit_orders

    async def test_revert_still_fires_once_the_broker_confirms_it_is_gone(self):
        # Positive control: the zombie cap must not become a no-op.
        _ib, _client, state, ex = self._env(broker_open_ids=set())
        await ex.check_fills()
        state.transition.assert_called_once_with(
            "T-CLOSING", (TradeStatus.CLOSING,), TradeStatus.FILLED)
        assert 901 not in ex._pending_exit_orders


# ===========================================================================
# [C] entry cancel verdicts need broker confirmation
# ===========================================================================

class TestEntryCancelVerdictConfirmation:
    def _pending_entry(self, client, state, order_id: int = 3001,
                       age_seconds: float = 60.0):
        ex = _executor(client, state)
        pending = SimpleNamespace(
            trade_id="T-ENTRY",
            signal=SimpleNamespace(symbol="SPY", entry_price=1.0,
                                   strategy_name="iron_condor", legs=[]),
            contracts=1, age_seconds=age_seconds, base_price=0.0,
            full_offset=0.0, is_credit=True, step=0, debit_cap=0.0,
            live_mid=0.0, nbbo_spread=0.0,
        )
        ex._pending_orders[order_id] = pending
        return ex

    async def test_reconnect_amnesia_does_not_cancel_a_working_entry(self):
        """ib_insync only LOGS an open-orders sync timeout and still reports a
        successful connect; trades()/openTrades() stay EMPTY. Every pending
        entry older than 30s was then flipped to CANCELLED while live at the
        broker — and its later fill created no Trade object at all."""
        client = BrokerAwareClient(FakeIB(), client_id=1)
        client.broker_open_ids = {3001}   # broker: still working
        state = MagicMock()
        ex = self._pending_entry(client, state)

        filled, exits = await ex.check_fills()

        assert (filled, exits) == ([], [])
        state.transition.assert_not_called()
        assert 3001 in ex._pending_orders
        assert 3001 in client.foreign_open_order_ids, (
            "an order working at the broker but invisible here is recorded so "
            "later passes refuse the verdict cheaply")

    async def test_unreachable_broker_defers_the_cancel_verdict(self):
        client = BrokerAwareClient(FakeIB(), client_id=1)
        client.broker_open_ids = None
        state = MagicMock()
        ex = self._pending_entry(client, state)

        await ex.check_fills()

        state.transition.assert_not_called()
        assert 3001 in ex._pending_orders

    async def test_empty_session_snapshot_defers_even_when_broker_answers(self):
        # The broker says "not working", but this session can see NO orders at
        # all — a fill would be equally invisible, so booking CANCELLED here
        # is a guess. Reconcile resolves it from positions instead.
        client = BrokerAwareClient(FakeIB(), client_id=1)
        client.broker_open_ids = set()
        state = MagicMock()
        ex = self._pending_entry(client, state)

        await ex.check_fills()

        state.transition.assert_not_called()
        assert 3001 in ex._pending_orders

    async def test_confirmed_dead_order_still_books_cancelled(self):
        """Positive control: with a healthy session view and a broker that
        answers, the cancel verdict must still be issued."""
        ib = FakeIB()
        # a healthy session sees SOME order — the cache is not amnesiac
        ib.all_trades.append(_frozen_foreign_trade(9999, client_id=1))
        client = BrokerAwareClient(ib, client_id=1)
        client.broker_open_ids = set()
        state = MagicMock()
        ex = self._pending_entry(client, state)

        await ex.check_fills()

        # R19: the CAS from-set narrowed to PENDING only — a PARTIAL row has
        # contracts LIVE at the broker and must never be flipped to CANCELLED
        # (see tests/test_r19_executor_fixes.py). The R16 contract this test
        # guards is unchanged: a confirmed-dead PENDING order still books
        # CANCELLED, exactly once.
        state.transition.assert_called_once_with(
            "T-ENTRY", (TradeStatus.PENDING,), TradeStatus.CANCELLED)
        assert 3001 not in ex._pending_orders


# ===========================================================================
# [D] foreign stash pruning
# ===========================================================================

class TestForeignStashPruning:
    async def test_dead_stashed_order_is_released(self):
        client = BrokerAwareClient(FakeIB(), client_id=1)
        client.foreign_open_order_ids.update({7001, 7002})
        client.broker_open_ids = {7002}
        ex = _executor(client, MagicMock())

        await ex.check_fills()

        assert client.foreign_open_order_ids == {7002}

    async def test_stash_is_never_pruned_on_an_unanswered_probe(self):
        client = BrokerAwareClient(FakeIB(), client_id=1)
        client.foreign_open_order_ids.update({7001, 7002})
        client.broker_open_ids = None
        ex = _executor(client, MagicMock())

        await ex.check_fills()

        assert client.foreign_open_order_ids == {7001, 7002}

    async def test_stale_foreign_entry_is_not_re_cancelled_every_pass(self):
        """cancel_order() can only fail for an order with no Trade object in
        this session; it logged an error on every 30s pass forever."""
        client = BrokerAwareClient(FakeIB(), client_id=1)
        client.foreign_open_order_ids.add(3001)
        client.broker_open_ids = {3001}
        ex = TestEntryCancelVerdictConfirmation()._pending_entry(
            client, MagicMock(), age_seconds=900.0)

        for _ in range(3):
            ex._broker_open_ids_cache = None
            await ex.check_fills()

        assert client.cancelled == []
        pages = [p for p in ex._stuck_order_pages if p[0] == 3001]
        assert len(pages) == 1

    def test_prune_helper_requires_an_authoritative_answer(self):
        c = IBKRClient.__new__(IBKRClient)
        c.foreign_open_order_ids = {1, 2, 3}
        assert c.prune_foreign_open_orders(None) == {1, 2, 3}
        assert c.foreign_open_order_ids == {1, 2, 3}
        assert c.prune_foreign_open_orders({2}) == {2}
        assert c.foreign_open_order_ids == {2}

    def test_first_connect_on_a_fallback_id_arms_the_stash(self):
        """A crash-restart lands on a fallback id while the Gateway still
        holds the base one — every order the dead process left working is
        foreign, but the first connect was exempt from stashing."""
        c = IBKRClient.__new__(IBKRClient)
        c._ever_connected = False
        assert c._should_stash_foreign(101, 1) is True
        assert c._should_stash_foreign(1, 1) is False  # R15 contract intact


# ===========================================================================
# [E] multi-account session
# ===========================================================================

class TestMultiAccountSession:
    def test_single_account_and_config_fallback_unchanged(self):
        assert IBKRClient._resolve_session_account(["U21959335"], "DUN603821") \
            == "U21959335"
        assert IBKRClient._resolve_session_account([], "DUN603821") == "DUN603821"
        assert IBKRClient._resolve_session_account([], None) == "unknown"

    def test_multi_account_without_a_selector_is_ambiguous(self):
        # pre-fix: returned "DUN603821" and the paper gate PASSED while a live
        # account was tradeable in the same session (reversed order blocked —
        # pure list-ordering luck).
        assert IBKRClient._resolve_session_account(
            ["DUN603821", "U1234567"], "") == IBKRClient.AMBIGUOUS_ACCOUNT
        assert IBKRClient._resolve_session_account(
            ["U1234567", "DUN603821"], None) == IBKRClient.AMBIGUOUS_ACCOUNT

    def test_multi_account_with_a_matching_selector_resolves(self):
        assert IBKRClient._resolve_session_account(
            ["DUN603821", "U1234567"], "U1234567") == "U1234567"

    def test_selector_naming_a_foreign_account_is_still_ambiguous(self):
        assert IBKRClient._resolve_session_account(
            ["DUN603821", "U1234567"], "U9999999") == IBKRClient.AMBIGUOUS_ACCOUNT

    async def test_connect_refuses_an_ambiguous_session(self, monkeypatch):
        monkeypatch.delenv("AIT_ACCOUNT", raising=False)
        c = IBKRClient.__new__(IBKRClient)
        c._config = SimpleNamespace(ibkr_host="127.0.0.1", ibkr_port=4002,
                                    ibkr_client_id=1, ibkr_account="")
        disconnects = []
        c._ib = SimpleNamespace(
            connectAsync=AsyncMock(return_value=None),
            reqMarketDataType=lambda t: None,
            managedAccounts=lambda: ["DUN603821", "U1234567"],
            disconnect=lambda: disconnects.append(1),
            isConnected=lambda: False,
        )
        c._connected = False
        c._reconnect_attempts = 0
        c._ever_connected = False
        c.foreign_open_order_ids = set()

        assert await c.connect() is False
        assert disconnects, "an ambiguous session must be torn down, not traded"
        assert c._connected is False

    async def test_account_values_filtered_to_the_selected_account(self):
        def _av(tag, value, currency, account):
            return SimpleNamespace(tag=tag, value=str(value), currency=currency,
                                   account=account)
        rows = [
            _av("ExchangeRate", "1.00", "BASE", "DU1"),
            _av("ExchangeRate", "1.00", "USD", "DU1"),
            _av("NetLiquidation", "100000", "USD", "DU1"),
            _av("ExchangeRate", "1.00", "BASE", "DU2"),
            _av("ExchangeRate", "1.00", "USD", "DU2"),
            # pre-fix: the BASE-row assignment made the LAST account win, so
            # sizing could run off an account the gate never validated.
            _av("NetLiquidation", "5000", "BASE", "DU2"),
        ]
        c = IBKRClient.__new__(IBKRClient)
        c.ensure_connected = AsyncMock(return_value=True)
        c._ib = SimpleNamespace(accountValues=lambda: rows)
        c._session_account = "DU1"

        values = await c.get_account_values()
        assert float(values["NetLiquidation"]) == pytest.approx(100000.0)

    async def test_place_order_pins_the_session_account(self):
        c = IBKRClient.__new__(IBKRClient)
        c._session_account = "DUN603821"
        c.ensure_connected = AsyncMock(return_value=True)
        order = SimpleNamespace(orderId=1, action="BUY", totalQuantity=1,
                                orderType="LMT", account="")
        contract = SimpleNamespace(symbol="SPY", secType="BAG")
        c._ib = SimpleNamespace(
            placeOrder=lambda ct, o: SimpleNamespace(order=o, contract=ct))

        await c.place_order(contract, order)
        assert order.account == "DUN603821"

    async def test_place_order_never_pins_an_unresolved_account(self):
        c = IBKRClient.__new__(IBKRClient)
        c._session_account = None  # config fallback only — the Gateway would reject it
        c.ensure_connected = AsyncMock(return_value=True)
        order = SimpleNamespace(orderId=1, action="BUY", totalQuantity=1,
                                orderType="LMT", account="")
        c._ib = SimpleNamespace(
            placeOrder=lambda ct, o: SimpleNamespace(order=o, contract=ct))

        await c.place_order(SimpleNamespace(symbol="SPY", secType="BAG"), order)
        assert order.account == ""


# ===========================================================================
# [F] get_account_values FX fallback
# ===========================================================================

class TestFxFallbackUsesParsedRate:
    def _client(self, rows, fx_result):
        c = IBKRClient.__new__(IBKRClient)
        c.ensure_connected = AsyncMock(return_value=True)
        c._ib = SimpleNamespace(accountValues=lambda: rows)
        c._fx_usd_to = AsyncMock(return_value=fx_result)
        return c

    def _rows(self):
        def _av(tag, value, currency):
            return SimpleNamespace(tag=tag, value=str(value), currency=currency)
        # USD ExchangeRate present, but NO rate-1.0 row to name the base ccy
        return [
            _av("ExchangeRate", "1.4063", "USD"),
            _av("NetLiquidation", "280000.00", "CAD"),
            _av("BuyingPower", "900000.00", "CAD"),
        ]

    async def test_parsed_exchange_rate_used_when_fx_fetch_fails(self):
        # pre-fix: returned {} despite holding a valid 1.4063, so
        # AccountManager kept a stale snapshot (or a currency-blind estimate).
        values = await self._client(self._rows(), None).get_account_values()
        assert values, "a rate was already in hand; the snapshot is usable"
        assert float(values["NetLiquidation"]) == pytest.approx(
            280000.00 / 1.4063, abs=0.5)

    async def test_live_fx_still_wins_when_available(self):
        values = await self._client(self._rows(), 1.3500).get_account_values()
        assert float(values["NetLiquidation"]) == pytest.approx(
            280000.00 / 1.35, abs=0.5)

    async def test_hard_stop_still_applies_with_no_rate_at_all(self):
        def _av(tag, value, currency):
            return SimpleNamespace(tag=tag, value=str(value), currency=currency)
        rows = [_av("NetLiquidation", "280000.00", "CAD")]
        assert await self._client(rows, None).get_account_values() == {}


# ===========================================================================
# [G] _reconnect concurrency
# ===========================================================================

class _Socket:
    def __init__(self, live: bool = False) -> None:
        self.live = live
        self.disconnects = 0

    def isConnected(self) -> bool:  # noqa: N802
        return self.live

    def disconnect(self) -> None:
        self.disconnects += 1
        self.live = False


class TestReconnectConcurrency:
    def _client(self, socket):
        c = IBKRClient.__new__(IBKRClient)
        c._ib = socket
        c._connected = socket.live
        c._reconnect_attempts = 0
        c._max_reconnect_attempts = 5
        c._reconnect_delay = 0
        c._reconnect_lock = None  # lazily created, as on a __new__ instance
        return c

    async def test_concurrent_waiters_produce_one_reconnect(self):
        """An event-driven fill task can reach ensure_connected while the main
        loop is already reconnecting. Unserialized, the second waiter woke
        from backoff and disconnected the session the first had restored."""
        sock = _Socket(live=False)
        c = self._client(sock)
        calls = []

        async def _connect():
            calls.append(1)
            await asyncio.sleep(0.01)
            sock.live = True
            c._connected = True
            return True

        c.connect = _connect
        results = await asyncio.gather(c._reconnect(), c._reconnect())

        assert results == [True, True]
        assert len(calls) == 1, "the second waiter must not reconnect again"
        assert sock.disconnects == 0, "a healthy session was torn down"

    async def test_live_session_is_not_torn_down_to_reconnect_it(self):
        sock = _Socket(live=True)
        c = self._client(sock)
        c._connected = True
        c.connect = AsyncMock(return_value=True)

        assert await c._reconnect() is True
        c.connect.assert_not_called()
        assert sock.disconnects == 0

    async def test_backoff_wakes_into_a_liveness_check(self):
        """The session can come back on its own during the backoff; the retry
        must notice instead of disconnecting a live socket."""
        sock = _Socket(live=False)
        c = self._client(sock)
        c.connect = AsyncMock(return_value=True)

        async def _revive(_delay):
            sock.live = True
            c._connected = True

        c_sleep = asyncio.sleep
        try:
            asyncio.sleep = _revive  # type: ignore[assignment]
            assert await c._reconnect() is True
        finally:
            asyncio.sleep = c_sleep  # type: ignore[assignment]

        c.connect.assert_not_called()
        assert sock.disconnects == 0
        assert c._reconnect_attempts == 0

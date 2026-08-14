"""R16 round-2 (long-tail) fixes for the orchestrator/config cluster.

Covers: the dead BAG quote path, the fail-open pre-event blackout, calendar-
vs-trading-day counting, the single capital-base authority, orphaned risk-key
sweeping, honest capital-tier logging, and notification task retention.

R17-bis (2026-08-14): three tests in this file COULD NOT FAIL.
  * TestMtmBrakeLoud inspected `_monitor_positions`, a method that has never
    existed (the real name is `_monitor_positions_fast`); the
    `... if hasattr(...) else ""` fallback made half the test inert.
  * TestBlackoutFailsClosed drove `_should_skip_entry`, another method that
    has never existed, through an in-test re-implementation whose return was
    `True if <cond> or True else False` — a tautology.
  * TestHonestTierLogging asserted a string in the source of the line that was
    raising AttributeError on EVERY cycle — that bug shipped and stopped the
    entry scan for three days behind a green test.
A source string proves the code was WRITTEN, never that it RUNS. Every live
path below now executes production code.
"""
from __future__ import annotations

import asyncio
import gc
import inspect
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.bot.orchestrator import TradingOrchestrator
from ait.bot.state import StateManager


@contextmanager
def structlog_events():
    """Capture structlog events regardless of level.

    setup_logging() is never called under pytest, so these loggers do NOT go
    through stdlib logging and caplog sees nothing; capture at the structlog
    layer instead. Yields a list that fills as events are emitted.
    """
    from structlog.testing import capture_logs
    with capture_logs() as events:
        yield events


def _events(events, name: str) -> list[dict]:
    return [e for e in events if e.get("event") == name]


class TestBagQuotePathAlive:
    def test_exit_quote_not_gated_on_bag_qualification(self):
        # A BAG cannot be qualified (reqContractDetails returns None), so the
        # old `if qualified_combo:` gate made the entire live-quote path dead
        # and every condor exit priced at the full wing width.
        # Structural check only — the executable coverage of this exit path
        # lives in tests/test_exit_order_safety.py, which drives the real
        # _close_multi_leg; this pins that the dead gate stays deleted.
        src = inspect.getsource(TradingOrchestrator._close_multi_leg)
        code = "\n".join(ln for ln in src.splitlines()
                         if not ln.strip().startswith("#"))
        assert "qualified_combo" not in code       # the dead gate is gone
        assert "reqMktData(quote_contract" in code
        assert "cancelMktData(quote_contract" in code  # pairing preserved


class _BudgetReached(Exception):
    """Sentinel raised from _get_trade_budget — the FIRST gate after the
    pre-event blackout block. Reaching it is proof the blackout let the signal
    through; never reaching it is proof the blackout stopped it."""


class TestBlackoutFailsClosed:
    """R16: the blackout's handler was a blanket `except: pass`, so a calendar
    outage made the gate FAIL OPEN and sell premium straight into the event it
    exists to avoid.

    R17-bis: the gate is inline in `_try_execute` (there is no
    `_should_skip_entry`), so drive `_try_execute` itself.
    """

    def _orch(self, tmp_path, monkeypatch, *, calendar_error=True, d2e=None):
        # CWD -> tmp: data/HALT, data/HALT_UNTRACKED, data/RESTRICTED.txt and
        # the post-stop-cooldown sqlite read are all CWD-relative. Pointing at
        # tmp keeps this test off the LIVE state DB entirely (the cooldown's
        # connect() then fails and is swallowed, as designed).
        monkeypatch.chdir(tmp_path)
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._learning = MagicMock()
        cal = MagicMock()
        if calendar_error:
            cal.days_until_next_event.side_effect = RuntimeError("calendar down")
        else:
            cal.days_until_next_event.return_value = d2e
        orch._economic_cal = cal
        orch._settings = SimpleNamespace(
            risk=SimpleNamespace(pre_event_blackout_days=1))
        orch._get_trade_budget = MagicMock(side_effect=_BudgetReached)
        return orch

    def _sig(self, strategy: str):
        return SimpleNamespace(symbol="SPY", strategy_name=strategy, expiry=None)

    async def test_calendar_failure_blocks_credit_entry(self, tmp_path, monkeypatch):
        orch = self._orch(tmp_path, monkeypatch)
        with structlog_events() as events:
            handled = await orch._try_execute(self._sig("iron_condor"),
                                              0.80, None, None)
        # True == "symbol handled", i.e. the caller stops; combined with the
        # untouched budget gate that means NO entry was attempted.
        assert handled is True
        orch._get_trade_budget.assert_not_called()
        assert _events(events, "pre_event_blackout_check_failed_failing_closed")

    async def test_calendar_failure_does_not_block_a_debit_entry(
            self, tmp_path, monkeypatch):
        # Fail-CLOSED is deliberately scoped to CREDIT structures: a debit
        # signal is long premium and has no short-vol exposure to the event,
        # so blanket-blocking it would be a second bug. It must fall THROUGH.
        orch = self._orch(tmp_path, monkeypatch)
        with pytest.raises(_BudgetReached):
            await orch._try_execute(self._sig("long_call"), 0.80, None, None)

    async def test_credit_blocked_when_event_is_inside_the_window(
            self, tmp_path, monkeypatch):
        # Working calendar, event today/tomorrow -> per-signal reject (False),
        # so the caller can still fall through to a debit strategy.
        orch = self._orch(tmp_path, monkeypatch, calendar_error=False, d2e=0)
        with structlog_events() as events:
            handled = await orch._try_execute(self._sig("iron_condor"),
                                              0.80, None, None)
        assert handled is False
        orch._get_trade_budget.assert_not_called()
        assert _events(events, "credit_entry_skipped_pre_event")

    async def test_credit_passes_when_event_is_far_out(self, tmp_path, monkeypatch):
        # 10 calendar days is >=5 sessions in every weekday alignment, so this
        # is the negative control: the gate must not block everything.
        orch = self._orch(tmp_path, monkeypatch, calendar_error=False, d2e=10)
        with pytest.raises(_BudgetReached):
            await orch._try_execute(self._sig("iron_condor"), 0.80, None, None)


class TestTradingDayBlackout:
    def test_friday_to_monday_event_has_zero_sessions_between(self, monkeypatch):
        import ait.bot.orchestrator as mod

        class _FakeDT:
            @staticmethod
            def now():
                return SimpleNamespace(date=lambda: date(2026, 8, 7))  # Friday
        monkeypatch.setattr(mod, "datetime", _FakeDT)
        # Monday event = 3 calendar days out, but NO session in between
        assert TradingOrchestrator._sessions_until(3) == 0

    def test_none_passthrough(self):
        assert TradingOrchestrator._sessions_until(None) is None


class TestCapitalBaseAuthority:
    def test_single_authority_exists(self):
        from ait.config.runtime_env import capital_base
        assert capital_base() > 0

    def test_env_override_wins(self, monkeypatch):
        from ait.config.runtime_env import capital_base
        monkeypatch.setenv("AIT_CAPITAL_BASE", "3000")
        assert capital_base() == pytest.approx(3000.0)

    def test_consumers_delegate(self):
        # Documentation-grade only: pins that the analytics modules IMPORT the
        # single authority rather than redefining a constant. The value-level
        # behaviour is covered by the two executable tests above.
        import ait.monitoring.analytics as an
        import ait.monitoring.duckdb_analytics as dan
        for mod in (an, dan):
            src = inspect.getsource(mod)
            assert "from ait.config.runtime_env import capital_base" in src


class _StopAfterSweep(Exception):
    """Sentinel raised from the first collaborator call AFTER the orphan-key
    sweep. _post_market then runs learning, reporting and Telegram work that
    is out of scope here (and must not run against a live book), so stop the
    coroutine the instant the swept state is observable."""


class _SweepState:
    """Real StateManager for the state-key API, with two seams: a controlled
    open-trade set, and a stop sentinel immediately after the sweep block."""

    def __init__(self, real: StateManager, open_ids: set[str]):
        self._real = real
        self._open_ids = open_ids

    def get_open_trades(self):
        return [SimpleNamespace(trade_id=i) for i in sorted(self._open_ids)]

    def pending_trueup_trade_ids(self):
        raise _StopAfterSweep

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestOrphanRiskKeySweep:
    def test_state_keys_like_present(self, tmp_path):
        st = StateManager(tmp_path / "s.db")
        st.set_state("trade_maxloss_T-1", "100")
        st.set_state("unrelated", "x")
        assert st.state_keys_like("trade_maxloss_%") == ["trade_maxloss_T-1"]

    async def test_post_market_sweeps_orphans_and_spares_live_keys(self, tmp_path):
        # R16: delete_state only runs on the _process_completed_exits path, so
        # CANCELLED / manually-flattened / restated trades leave trade_maxloss_*
        # behind (28 such keys in the 07-07 forensics) and each one permanently
        # inflates the capital-at-risk denominator.
        # R17-bis: was `assert "state_keys_like" in getsource(_post_market)` —
        # which would still pass if the loop deleted the WRONG keys. Run it.
        st = StateManager(tmp_path / "s.db")
        st.set_state("trade_maxloss_T-OPEN", "250")
        st.set_state("trade_maxloss_T-GONE", "400")
        st.set_state("unrelated_key", "keep-me")

        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._state = _SweepState(st, {"T-OPEN"})

        with structlog_events() as events:
            with pytest.raises(_StopAfterSweep):
                await orch._post_market()

        assert st.get_state("trade_maxloss_T-GONE", "") == ""    # orphan swept
        assert st.get_state("trade_maxloss_T-OPEN", "") == "250"  # still open
        assert st.get_state("unrelated_key", "") == "keep-me"    # not our prefix
        rec = _events(events, "orphan_maxloss_keys_swept")
        assert rec and rec[0]["count"] == 1

    async def test_sweep_failure_is_contained_and_logged(self, tmp_path):
        # The sweep runs at the very top of the post-market phase; a state
        # error there must not abort reconciliation/booking behind it.
        st = MagicMock()
        st.get_open_trades.side_effect = RuntimeError("state db locked")
        st.pending_trueup_trade_ids.side_effect = _StopAfterSweep

        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._state = st
        with structlog_events() as events:
            with pytest.raises(_StopAfterSweep):
                await orch._post_market()
        assert _events(events, "maxloss_key_sweep_failed")


class TestHonestTierLogging:
    async def test_logs_intersection_not_menu(self):
        """R16: log what is ACTUALLY tradeable, not the tier's menu.

        R17-bis REGRESSION: this test used to be
        `assert "tier_menu=" in inspect.getsource(_trading_cycle)`. It stayed
        green for three days while that very line raised AttributeError on
        EVERY cycle ('TradingConfig' object has no attribute 'strategies' —
        `strategies` lives on OptionsConfig): the entry scan never ran, exits
        kept working so it looked healthy, and the repeated failures tripped
        the breaker on api_failures. Drive the real cycle with the REAL
        Settings and the REAL tier table — the log line cannot be emitted
        unless every attribute in it resolves.
        """
        from ait.config.settings import load_settings
        from ait.risk.capital_tiers import CapitalTierManager

        settings = load_settings("config.yaml")
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._settings = settings
        orch._capital_tiers = CapitalTierManager()      # real tier menu
        orch._circuit_breaker = MagicMock(is_tripped=False)
        orch._sync_risk_manager_positions = AsyncMock()
        orch._portfolio = MagicMock()
        orch._portfolio.check_positions = AsyncMock(return_value=[])
        orch._scheduler = MagicMock()
        orch._scheduler.should_avoid_new_trades.return_value = False
        orch._executor = MagicMock()
        orch._executor.check_fills_safe = AsyncMock(return_value=([], []))
        orch._process_completed_exits = AsyncMock()
        adaptor = MagicMock()
        adaptor.is_symbol_allowed.return_value = True
        # False -> the cycle returns on the very next branch after the tier
        # log, so nothing downstream (scan/execute) can run from a unit test.
        adaptor.is_hour_allowed.return_value = False
        orch._learning = MagicMock(adaptor=adaptor)
        orch._account = MagicMock()
        orch._account.get_snapshot = AsyncMock(
            return_value=SimpleNamespace(net_liquidation=200_000.0))
        orch._state = MagicMock()

        with structlog_events() as events:
            await orch._trading_cycle()

        rec = _events(events, "capital_tier_active")
        assert rec, "tier line never emitted — the cycle died before reaching it"
        e = rec[0]
        enabled = set(settings.options.strategies)
        # `strategies` is the honest intersection; both inputs are shown too
        assert e["strategies"] == [s for s in e["tier_menu"] if s in enabled]
        assert e["config_enabled"] == list(settings.options.strategies)
        # and at $200k (LARGE tier) the menu is strictly wider than what config
        # enables — the exact discrepancy the R16 line exists to expose.
        assert set(e["tier_menu"]) - set(e["strategies"])

    def test_strategies_attribute_actually_exists(self):
        """R17-bis REGRESSION: the source-string test above passed while the
        code raised AttributeError on EVERY cycle.

        The tier-logging fix read self._settings.trading.strategies, but
        `strategies` lives on OptionsConfig, not TradingConfig. Result:
        'TradingConfig' object has no attribute 'strategies' 300 times, the
        entry scan never ran for 3 days (exits were unaffected, which is why
        it looked healthy), and the repeated failures tripped the circuit
        breaker on api_failures. A source-string assertion can never catch a
        runtime attribute error — resolve the real config object instead.
        """
        from ait.config.settings import load_settings, TradingConfig
        st = load_settings("config.yaml")
        assert "strategies" not in TradingConfig.model_fields
        enabled = set(st.options.strategies)          # must not raise
        assert enabled and "iron_condor" in enabled
        # and the exact expression the cycle evaluates
        tier_menu = ["iron_condor", "short_strangle", "long_straddle"]
        assert [s for s in tier_menu if s in enabled] == ["iron_condor"]


class TestNotificationTaskRetention:
    async def test_task_is_strongly_referenced(self):
        """R16: an unreferenced create_task can be garbage-collected in flight
        (CPython gives the loop only a weak ref), so an alert could vanish
        before its first await.

        R17-bis: was a getsource string check, which cannot observe whether
        the reference actually survives a collection.
        """
        sent: list[str] = []

        async def _notify(msg):
            await asyncio.sleep(0)     # a real send always suspends at least once
            sent.append(msg)
            return True

        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._notify = _notify

        await orch._send_notification("page me")
        assert not sent, "task ran inline — this test would prove nothing"
        assert len(orch._notify_tasks) == 1
        gc.collect()                   # the ONLY strong ref is _notify_tasks
        assert len(orch._notify_tasks) == 1

        await asyncio.wait(set(orch._notify_tasks), timeout=5)
        assert sent == ["page me"]
        await asyncio.sleep(0)
        assert not orch._notify_tasks  # done-callback discards, so no leak

    async def test_shutdown_drains(self):
        """R16: the shutdown page is fire-and-forget like every other alert, so
        the loop used to close out from under it — the one message that says
        'the bot is gone' was the message most likely to be lost."""
        delivered: list[str] = []

        async def _notify(msg):
            await asyncio.sleep(0.05)  # a Telegram POST is not instantaneous
            delivered.append(msg)
            return True

        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._notify = _notify
        orch._watchdog = MagicMock()
        orch._watchdog.get_summary.return_value = "all green"

        await orch._shutdown()
        # Pre-fix: _shutdown returned while the task sat unstarted and the
        # loop tore down -> delivered == [].
        assert delivered and "shutting down" in delivered[0]


class TestMtmBrakeLoud:
    """A1: the mark-to-market daily-loss brake is the only control that stops
    a bleeding gap day (the realized-only breaker is entry-gated and blind to
    unrealized bleeding).

    R17-bis: the old test inspected `TradingOrchestrator._monitor_positions`,
    which does not exist — `... if hasattr(...) else ""` silently produced an
    empty string, so half the assertions were inert and the other half read a
    string out of the module text. Drive the real fast monitor.
    """

    def _orch(self):
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._check_trading_enabled = AsyncMock(return_value=True)
        orch._check_economic_calendar_exhaustion = AsyncMock()
        orch._portfolio = MagicMock()
        orch._portfolio.check_positions = AsyncMock(return_value=[])
        orch._watchdog = MagicMock()
        orch._state = MagicMock()
        orch._state.get_state.return_value = ""        # first tick of the day
        orch._circuit_breaker = MagicMock()
        orch._circuit_breaker.check_daily_loss_mtm.return_value = False
        orch._account = MagicMock()
        orch._account.get_net_liquidation = AsyncMock(return_value=25_000.0)
        orch._executor = MagicMock()
        orch._executor._pending_exit_orders = {}
        orch._executor.check_fills_safe = AsyncMock(return_value=([], []))
        orch._process_completed_exits = AsyncMock()
        orch._send_notification = AsyncMock()
        orch._edgar_check_count = 0
        return orch

    async def test_failure_is_warning_and_pages(self):
        orch = self._orch()
        orch._state.get_daily_stats.side_effect = RuntimeError("state db locked")

        with structlog_events() as events:
            await orch._monitor_positions_fast()

        rec = _events(events, "mtm_check_failed")
        assert rec, "the brake failed silently"
        # R16: this was log.debug — at the configured level the one control
        # that stops a bleeding day could fail every tick and look healthy.
        assert rec[0]["log_level"] == "warning"
        page = orch._send_notification.await_args.args[0]
        assert "MTM BRAKE NOT EVALUATING" in page
        # and the failure must stay CONTAINED: exits/fills keep running.
        assert not _events(events, "fast_monitor_error")
        orch._executor.check_fills_safe.assert_awaited_once()

    async def test_healthy_tick_is_quiet_and_can_still_halt(self):
        # Negative control: without it, a test that always warns would pass.
        orch = self._orch()
        orch._state.get_daily_stats.return_value = SimpleNamespace(total_pnl=-900.0)
        orch._circuit_breaker.check_daily_loss_mtm.return_value = True

        with structlog_events() as events:
            await orch._monitor_positions_fast()

        assert not _events(events, "mtm_check_failed")
        page = orch._send_notification.await_args.args[0]
        assert "DAILY MTM LOSS HALT" in page

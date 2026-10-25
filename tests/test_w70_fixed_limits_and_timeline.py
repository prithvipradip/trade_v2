"""W70 — fixed limits made visible; order-timeline durations made configurable.

Two halves, matching the operator's rule "separate VISIBLE from TUNABLE":

  * the values deliberately fixed in code live in ONE module
    (ait.config.invariants), are logged at startup and documented in the GUIDE;
  * the six order-lifecycle durations that were literals in execution/executor.py
    are config (positions.* / exit.*), with floors and cross-field rules.

Every default equals the literal it replaced (the parity class is the "zero
behavior change" contract). Behavioral tests drive the REAL executor paths; the
class guards are AST walks that catch the next re-hardcoding.
"""
from __future__ import annotations

import ast
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError

from ait.bot.state import TradeStatus
from ait.config import invariants as inv
from ait.config.settings import (
    ExitConfig,
    PositionConfig,
    Settings,
    load_settings,
)
from ait.execution.executor import TradeExecutor
from tests.fakes import FakeIB
from tests.test_r16_broker_fixes import BrokerAwareClient, _closing_state, _register_exit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ait"
CONFIG_PATH = ROOT / "config.yaml"
GUIDE = ROOT / "docs" / "GUIDE.md"          # the fixed-limits and timeline tables live there (generated: scripts/gen_docs.py)

# --------------------------------------------------------------------------- #
# 1. The invariants module
# --------------------------------------------------------------------------- #

# name -> the literal that lived in code before W70 (the parity contract)
FORMER_LITERALS = {
    "ORDER_RATE_LIMIT_MAX": 8,
    "ORDER_RATE_LIMIT_WINDOW_S": 60,
    "NBBO_DISORDERLY_SPREAD_RATIO": 1.0,
    "NBBO_STALE_SIGNAL_DEVIATION": 0.35,
    "NBBO_MID_FLOOR": 0.05,
    "LIVE_MAX_SPREAD_PCT": 0.15,
    "LIVE_MIN_VOLUME": 10,
    "PDT_MAX_DAY_TRADES": 3,
    "PDT_WINDOW_DAYS": 5,
    "BROKER_SNAPSHOT_TTL_SECONDS": 5.0,
    "IC_BUDGET_CAPPED_MIN_WING": 1.0,
}


class TestInvariantsModule:
    def test_values_are_the_former_literals(self):
        assert {k: getattr(inv, k) for k in FORMER_LITERALS} == FORMER_LITERALS

    def test_registry_covers_every_constant_and_nothing_else(self):
        names = {i.name for i in inv.INVARIANTS}
        assert names == set(FORMER_LITERALS)
        assert len(names) == len(inv.INVARIANTS), "duplicate registry row"
        for row in inv.INVARIANTS:
            assert getattr(inv, row.name) == row.value, row.name
            assert row.what and row.why_fixed and row.enforced_in, (
                f"{row.name}: what / why_fixed / enforced_in must all be written")

    def test_module_is_a_stdlib_only_leaf(self):
        """Settings, executor, PDT guard and orchestrator all import it, so it
        must import none of them back (circular-import and layering guard)."""
        tree = ast.parse((SRC / "config" / "invariants.py").read_text())
        imported = []
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                imported.append(n.module or "")
        assert not [m for m in imported if m.split(".")[0] == "ait"], imported

    def test_invariant_values_is_the_log_payload(self):
        assert inv.invariant_values() == FORMER_LITERALS


class TestUseSitesImportTheInvariants:
    """The values at the use sites ARE the invariants (not second copies)."""

    def test_pdt_guard_aliases(self):
        from ait.risk import pdt_guard
        assert pdt_guard.MAX_DAY_TRADES == inv.PDT_MAX_DAY_TRADES
        assert pdt_guard.WINDOW_DAYS == inv.PDT_WINDOW_DAYS

    def test_orchestrator_live_floors_alias(self):
        from ait.bot import orchestrator
        assert orchestrator._LIVE_MAX_SPREAD_PCT == inv.LIVE_MAX_SPREAD_PCT
        assert orchestrator._LIVE_MIN_VOLUME == inv.LIVE_MIN_VOLUME

    def test_executor_snapshot_ttl_alias(self):
        assert TradeExecutor._BROKER_SNAPSHOT_TTL == inv.BROKER_SNAPSHOT_TTL_SECONDS

    @pytest.mark.parametrize("rel,names", [
        ("risk/pdt_guard.py", ("MAX_DAY_TRADES", "WINDOW_DAYS")),
        ("bot/orchestrator.py", ("_LIVE_MAX_SPREAD_PCT", "_LIVE_MIN_VOLUME")),
    ])
    def test_module_constants_are_not_bare_literals(self, rel, names):
        """Class guard: re-typing `MAX_DAY_TRADES = 3` would silently fork the value."""
        tree = ast.parse((SRC / rel).read_text())
        offenders = []
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in names for t in n.targets):
                if isinstance(n.value, ast.Constant):
                    offenders.append(n.targets[0].id)
        assert not offenders, f"{rel}: literal re-introduced for {offenders}"

    def test_executor_has_no_literal_for_the_rate_limit_or_nbbo_gates(self):
        """No comparison in executor.py may test against 8 / 60 / 0.35 / 0.05, and the
        1.0 spread ratio may not appear as a comparison operand either."""
        tree = ast.parse((SRC / "execution" / "executor.py").read_text())
        suspects = {8, 60, 0.35, 0.05}
        hits = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for c in [node.left, *node.comparators]:
                if isinstance(c, ast.Constant) and not isinstance(c.value, bool) \
                        and c.value in suspects:
                    hits.append(f"line {node.lineno}: {ast.unparse(node)[:80]}")
        assert not hits, "fixed limit re-hardcoded in executor.py:\n  " + "\n  ".join(hits)


# --------------------------------------------------------------------------- #
# 2. Startup logs
# --------------------------------------------------------------------------- #

@pytest.fixture
def recorded_logs(monkeypatch):
    """Capture every structlog event emitted through ait.utils.logging.get_logger."""
    events: list[tuple[str, dict]] = []

    class _Rec:
        def __getattr__(self, level):
            def _emit(event, **kw):
                events.append((event, kw))
            return _emit

    import ait.utils.logging as ul
    monkeypatch.setattr(ul, "get_logger", lambda *_a, **_k: _Rec())
    return events


class TestStartupLogs:
    def test_fixed_limits_event_lists_every_invariant(self, recorded_logs):
        load_settings(str(CONFIG_PATH))
        ev = [kw for e, kw in recorded_logs if e == "fixed_limits_in_effect"]
        assert len(ev) == 1
        assert ev[0]["limits"] == inv.invariant_values()
        assert ev[0]["count"] == len(inv.INVARIANTS)

    def test_timeline_event_lists_every_duration(self, recorded_logs):
        s = load_settings(str(CONFIG_PATH))
        ev = [kw for e, kw in recorded_logs if e == "order_timeline_in_effect"]
        assert len(ev) == 1
        e = ev[0]
        assert e["fast_monitor_tick_seconds"] == 30.0   # ait.utils.cadence
        assert set(e["exit_order"]) == {
            "order_visibility_grace_seconds", "exit_stop_pending_rest_seconds",
            "exit_pending_rest_seconds", "exit_band_ttl_seconds",
            "foreign_exit_escalation_seconds", "zombie_exit_revert_seconds"}
        assert set(e["entry_order"]) == {
            "entry_ladder_step_seconds", "partial_fill_cancel_seconds",
            "absent_order_grace_seconds", "entry_order_timeout_seconds"}
        assert e["exit_order"]["zombie_exit_revert_seconds"] == s.exit.zombie_exit_revert_seconds
        assert e["fixed"] == {"broker_snapshot_ttl_seconds": inv.BROKER_SNAPSHOT_TTL_SECONDS}


# --------------------------------------------------------------------------- #
# 3. Config: parity, floors, cross-field rules
# --------------------------------------------------------------------------- #

NEW_KEYS = {
    "exit.order_visibility_grace_seconds": 10,
    "exit.foreign_exit_escalation_seconds": 300,
    "exit.zombie_exit_revert_seconds": 900,
    "positions.partial_fill_cancel_seconds": 30,
    "positions.absent_order_grace_seconds": 30,
    "positions.entry_ladder_step_seconds": 45,
}


def _dig(obj, dotted):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


class TestTimelineConfigParity:
    def test_defaults_are_the_former_literals(self):
        s = Settings()
        assert {k: _dig(s, k) for k in NEW_KEYS} == NEW_KEYS

    def test_shipped_yaml_carries_them_and_validates(self):
        s = load_settings(str(CONFIG_PATH))
        assert {k: _dig(s, k) for k in NEW_KEYS} == NEW_KEYS
        raw = yaml.safe_load(CONFIG_PATH.read_text())
        for dotted in NEW_KEYS:
            section, key = dotted.split(".")
            assert key in raw[section], f"{dotted} missing from config.yaml"

    def test_executor_class_defaults_match_the_models(self):
        """Instances built via __new__ (or with settings=None) use these."""
        assert TradeExecutor._partial_fill_cancel_s == 30
        assert TradeExecutor._absent_order_grace_s == 30
        assert TradeExecutor._ladder_step_s == 45
        assert TradeExecutor._exit_visibility_grace_s == 10
        assert TradeExecutor._foreign_exit_escalation_s == 300
        assert TradeExecutor._zombie_revert_s == 900


class TestTimelineValidators:
    @pytest.mark.parametrize("kwargs", [
        dict(order_visibility_grace_seconds=4),      # floor 5
        dict(foreign_exit_escalation_seconds=59),    # floor 60
        dict(zombie_exit_revert_seconds=119),        # hard floor 120
        dict(order_visibility_grace_seconds=61),     # ceiling 60
    ])
    def test_exit_field_bounds(self, kwargs):
        with pytest.raises(ValidationError):
            ExitConfig(**kwargs)

    @pytest.mark.parametrize("kwargs", [
        dict(partial_fill_cancel_seconds=4),
        dict(absent_order_grace_seconds=9),
        dict(entry_ladder_step_seconds=29),          # floor = the 30s tick
        dict(entry_ladder_step_seconds=301),
    ])
    def test_position_field_bounds(self, kwargs):
        with pytest.raises(ValidationError):
            PositionConfig(**kwargs)

    def test_zombie_cap_needs_twice_the_longer_rest(self):
        # default rests 300 / 120 -> floor 600
        ExitConfig(zombie_exit_revert_seconds=600)          # exactly the floor: ok
        with pytest.raises(ValidationError, match="2 x the longer rest"):
            ExitConfig(zombie_exit_revert_seconds=599)

    def test_raising_a_rest_cascades_to_the_page_threshold_and_the_cap(self):
        """Raising exit_pending_rest_seconds is a three-key edit by design: the page
        threshold must follow it, and the zombie cap must be >= 2 x it. Each refusal
        names the key to change, so the operator is walked through it."""
        with pytest.raises(ValidationError, match="foreign_exit_escalation_seconds"):
            ExitConfig(exit_pending_rest_seconds=600)       # foreign (300) < rest (600)
        with pytest.raises(ValidationError) as ei:
            ExitConfig(exit_pending_rest_seconds=600, foreign_exit_escalation_seconds=600)
        assert "zombie_exit_revert_seconds" in str(ei.value)    # cap 900 < 1200
        assert "Raise it, or lower exit_pending_rest_seconds" in str(ei.value)
        ExitConfig(exit_pending_rest_seconds=600, foreign_exit_escalation_seconds=600,
                   zombie_exit_revert_seconds=1200)

    def test_stop_rest_may_not_exceed_the_general_rest(self):
        with pytest.raises(ValidationError, match="stop exit must be re-priced"):
            ExitConfig(exit_stop_pending_rest_seconds=400)  # > 300

    def test_visibility_grace_must_be_below_the_stop_rest(self):
        with pytest.raises(ValidationError, match="swallow the rest"):
            ExitConfig(order_visibility_grace_seconds=30, exit_stop_pending_rest_seconds=30)

    def test_foreign_escalation_may_not_precede_the_rest(self):
        with pytest.raises(ValidationError, match="just noise"):
            ExitConfig(foreign_exit_escalation_seconds=120)  # < 300

    @pytest.mark.parametrize("field", [
        "partial_fill_cancel_seconds", "absent_order_grace_seconds",
        "entry_ladder_step_seconds"])
    def test_entry_durations_must_fit_inside_the_entry_timeout(self, field):
        with pytest.raises(ValidationError, match="must be below"):
            PositionConfig(entry_order_timeout_seconds=60, **{field: 60})


class TestYamlFloorsMatchTheModel:
    """Drift guard: the Floor/Ceiling numbers a human reads in config.yaml must be the
    bounds the model enforces (the model is the enforcer; the comment is documentation)."""

    CASES = {  # key -> (model class, field, whether the yaml states a numeric floor)
        "order_visibility_grace_seconds": (ExitConfig, "order_visibility_grace_seconds"),
        "foreign_exit_escalation_seconds": (ExitConfig, "foreign_exit_escalation_seconds"),
        "entry_ladder_step_seconds": (PositionConfig, "entry_ladder_step_seconds"),
        "partial_fill_cancel_seconds": (PositionConfig, "partial_fill_cancel_seconds"),
        "absent_order_grace_seconds": (PositionConfig, "absent_order_grace_seconds"),
    }

    def _comment_block(self, key: str) -> str:
        lines = CONFIG_PATH.read_text().splitlines()
        idx = next(i for i, l in enumerate(lines) if re.match(rf"\s+{key}:\s", l))
        out = []
        j = idx - 1
        while j >= 0 and lines[j].lstrip().startswith("#"):
            out.append(lines[j])
            j -= 1
        return "\n".join(reversed(out))

    @pytest.mark.parametrize("key", list(CASES))
    def test_floor_and_ceiling_in_the_comment_are_the_model_bounds(self, key):
        cls, field = self.CASES[key]
        meta = {type(m).__name__: m for m in cls.model_fields[field].metadata}
        floor, ceiling = meta["Ge"].ge, meta["Le"].le
        block = self._comment_block(key)
        m = re.search(r"Floor:\s*([\d.]+)s.*?Ceiling:\s*([\d.]+)s", block, re.S)
        assert m, f"{key}: no 'Floor: Ns ... Ceiling: Ns' in its yaml comment"
        assert float(m.group(1)) == floor and float(m.group(2)) == ceiling, (
            f"{key}: yaml says {m.group(1)}/{m.group(2)}, model enforces {floor}/{ceiling}")
        assert "IMPACT:" in block, f"{key}: yaml comment must state the consequence"

    def test_zombie_cap_comment_states_the_rule_and_ceiling(self):
        block = self._comment_block("zombie_exit_revert_seconds")
        assert "2 x the longer rest time" in block
        assert re.search(r"Ceiling:\s*7200s", block)
        assert "IMPACT:" in block


# --------------------------------------------------------------------------- #
# 4. The executor reads the durations from config (real paths)
# --------------------------------------------------------------------------- #

def _executor(client=None, state=None, *, exit_cfg=None, pos_cfg=None):
    client = client or BrokerAwareClient(FakeIB(), client_id=105)
    settings = Settings(exit=exit_cfg or ExitConfig(), positions=pos_cfg or PositionConfig())
    return TradeExecutor(ibkr_client=client, state=state or MagicMock(),
                         circuit_breaker=MagicMock(), settings=settings)


class TestExecutorReadsConfig:
    def test_attributes_follow_settings(self):
        ex = _executor(
            exit_cfg=ExitConfig(order_visibility_grace_seconds=20,
                                foreign_exit_escalation_seconds=500,
                                zombie_exit_revert_seconds=1300),
            pos_cfg=PositionConfig(partial_fill_cancel_seconds=40,
                                   absent_order_grace_seconds=50,
                                   entry_ladder_step_seconds=60))
        assert (ex._exit_visibility_grace_s, ex._foreign_exit_escalation_s,
                ex._zombie_revert_s) == (20, 500, 1300)
        assert (ex._partial_fill_cancel_s, ex._absent_order_grace_s,
                ex._ladder_step_s) == (40, 50, 60)

    @pytest.mark.parametrize("settings", [None, MagicMock(), SimpleNamespace()])
    def test_no_settings_a_mock_or_a_stub_keep_the_former_values(self, settings):
        """float(MagicMock()) is 1.0 — without the isinstance guard every age
        threshold would silently become one second."""
        ex = TradeExecutor(ibkr_client=MagicMock(), state=MagicMock(),
                           circuit_breaker=MagicMock(), settings=settings)
        assert (ex._exit_visibility_grace_s, ex._foreign_exit_escalation_s,
                ex._zombie_revert_s) == (10, 300, 900)
        assert (ex._partial_fill_cancel_s, ex._absent_order_grace_s,
                ex._ladder_step_s) == (30, 30, 45)

    def test_instance_built_without_init_still_has_defaults(self):
        ex = TradeExecutor.__new__(TradeExecutor)
        assert ex._zombie_revert_s == 900 and ex._ladder_step_s == 45


def _absent_pending(age):
    return SimpleNamespace(age_seconds=age, trade_id="T-ENTRY")


class TestAbsentOrderGraceIsConfig:
    def _verdict(self, ex, age):
        ex._cancel_verdict = lambda oid, p: "JUDGED"
        return ex._determine_fill_status(3001, [], _absent_pending(age))

    def test_default_boundary_is_30s(self):
        ex = _executor()
        assert self._verdict(ex, 29.9) == "pending"
        assert self._verdict(ex, 30.0) == "JUDGED"

    def test_boundary_moves_with_config(self):
        ex = _executor(pos_cfg=PositionConfig(absent_order_grace_seconds=60))
        assert self._verdict(ex, 45) == "pending"
        assert self._verdict(ex, 60) == "JUDGED"
        ex2 = _executor(pos_cfg=PositionConfig(absent_order_grace_seconds=10))
        assert self._verdict(ex2, 15) == "JUDGED"


class TestZombieCapIsConfig:
    def _env(self, cfg):
        client = BrokerAwareClient(FakeIB(), client_id=1)
        client.broker_open_ids = set()                 # broker answers: not working
        state = _closing_state()
        ex = _executor(client, state, exit_cfg=cfg)
        return state, ex

    async def test_default_cap_reverts_at_901s(self):
        state, ex = self._env(ExitConfig())
        _register_exit(ex, 901, "T-CLOSING", age_seconds=901)
        await ex.check_fills()
        state.transition.assert_called_once_with(
            "T-CLOSING", (TradeStatus.CLOSING,), TradeStatus.FILLED)

    async def test_a_longer_configured_cap_holds_at_901s_then_reverts(self):
        state, ex = self._env(ExitConfig(zombie_exit_revert_seconds=1500))
        _register_exit(ex, 901, "T-CLOSING", age_seconds=901)
        await ex.check_fills()
        state.transition.assert_not_called()           # 901 < 1500: still waiting
        assert 901 in ex._pending_exit_orders
        ex._pending_exit_orders[901].submitted_at -= 700   # now ~1601s old
        await ex.check_fills()
        state.transition.assert_called_once_with(
            "T-CLOSING", (TradeStatus.CLOSING,), TradeStatus.FILLED)


class TestForeignEscalationIsConfig:
    def _env(self, cfg):
        from tests.test_r16_broker_fixes import _frozen_foreign_trade
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=105)
        ib.all_trades.append(_frozen_foreign_trade(777, client_id=1))
        client.broker_open_ids = {777}                 # still working under the old clientId
        ex = _executor(client, _closing_state(), exit_cfg=cfg)
        ex.adopt_exit_order(777, "T-CLOSING", "reconcile_working_exit")
        return ex

    async def test_pages_after_300s_by_default(self):
        ex = self._env(ExitConfig())
        ex._pending_exit_orders[777].submitted_at -= 301
        await ex.check_fills()
        assert (777, "exit_order_foreign_uncancellable") in ex._stuck_order_pages

    async def test_configured_threshold_delays_the_page(self):
        ex = self._env(ExitConfig(foreign_exit_escalation_seconds=600))
        ex._pending_exit_orders[777].submitted_at -= 301
        await ex.check_fills()
        assert (777, "exit_order_foreign_uncancellable") not in ex._stuck_order_pages
        ex._pending_exit_orders[777].submitted_at -= 300     # ~601s
        await ex.check_fills()
        assert (777, "exit_order_foreign_uncancellable") in ex._stuck_order_pages


class TestVisibilityGraceIsConfig:
    async def _statuses_checked(self, cfg, age):
        client = BrokerAwareClient(FakeIB(), client_id=105)
        state = _closing_state()
        ex = _executor(client, state, exit_cfg=cfg)
        calls = []
        ex._determine_exit_fill_status = lambda oid, trades: calls.append(oid) or "pending"
        _register_exit(ex, 5001, "T-CLOSING", age_seconds=age)
        await ex.check_fills()
        return calls

    async def test_default_grace_is_10s(self):
        assert await self._statuses_checked(ExitConfig(), 9) == []
        assert await self._statuses_checked(ExitConfig(), 11) == [5001]

    async def test_configured_grace_moves_the_boundary(self):
        cfg = ExitConfig(order_visibility_grace_seconds=40)
        assert await self._statuses_checked(cfg, 25) == []
        assert await self._statuses_checked(cfg, 45) == [5001]


class TestLadderStepIsConfig:
    def _pending(self, age):
        return SimpleNamespace(
            trade_id="T-LADDER", base_price=1.0, full_offset=0.4, is_credit=False,
            debit_cap=0.0, step=0, age_seconds=age)

    async def _step_after(self, step_s, age):
        client = BrokerAwareClient(FakeIB(), client_id=105)
        ex = _executor(client, pos_cfg=PositionConfig(entry_ladder_step_seconds=step_s))
        p = self._pending(age)
        ex._pending_orders[9001] = p
        await ex._reprice_pending_entries()
        return p.step

    async def test_default_45s_step(self):
        assert await self._step_after(45, 44) == 0
        assert await self._step_after(45, 50) == 1
        assert await self._step_after(45, 95) == 2

    async def test_60s_step_delays_the_first_escalation(self):
        assert await self._step_after(60, 50) == 0     # would be step 1 at 45s
        assert await self._step_after(60, 65) == 1


class TestPartialFillCancelIsConfig:
    def _partial_trade(self, order_id):
        return SimpleNamespace(
            order=SimpleNamespace(orderId=order_id, permId=order_id * 10 + 7, clientId=105,
                                  totalQuantity=2, lmtPrice=0.55, action="BUY",
                                  orderType="LMT"),
            orderStatus=SimpleNamespace(status="PartiallyFilled", filled=1, remaining=1,
                                        avgFillPrice=0.55),
            contract=SimpleNamespace(secType="BAG", symbol="SPY"),
            fills=[], log=[])

    async def _cancel_called(self, cfg, age):
        ib = FakeIB()
        client = BrokerAwareClient(ib, client_id=105)
        ib.all_trades.append(self._partial_trade(4001))
        ex = _executor(client, MagicMock(), pos_cfg=cfg)
        cancels = []
        ex._blind_cancel_once = lambda *a, **k: cancels.append(a[1])
        ex._pending_orders[4001] = SimpleNamespace(
            trade_id="T-PARTIAL", signal=SimpleNamespace(
                symbol="SPY", entry_price=1.0, strategy_name="iron_condor", legs=[],
                contract=None, strategy="iron_condor", direction=None),
            contracts=2, age_seconds=age, base_price=0.0, full_offset=0.0,
            is_credit=True, step=0, debit_cap=0.0, live_mid=0.0, nbbo_spread=0.0)
        await ex.check_fills()
        return cancels

    async def test_default_30s(self):
        assert await self._cancel_called(PositionConfig(), 20) == []
        assert await self._cancel_called(PositionConfig(), 40) == ["partial_remainder_cancel_unsent"]

    async def test_configured_value_moves_the_boundary(self):
        cfg = PositionConfig(partial_fill_cancel_seconds=120)
        assert await self._cancel_called(cfg, 40) == []
        assert await self._cancel_called(cfg, 130) == ["partial_remainder_cancel_unsent"]


# --------------------------------------------------------------------------- #
# 5. Telemetry: order_ack_latency
# --------------------------------------------------------------------------- #

def _log_entry(t, status):
    return SimpleNamespace(time=t, status=status, message="")


def _trade_with_log(order_id, statuses_and_offsets):
    t0 = datetime(2026, 10, 9, 14, 0, 0, tzinfo=timezone.utc)
    entries = [_log_entry(t0 + timedelta(seconds=off), st) for st, off in statuses_and_offsets]
    return SimpleNamespace(order=SimpleNamespace(orderId=order_id), log=entries)


class TestOrderAckLatencyTelemetry:
    def _ex(self, monkeypatch):
        import ait.execution.executor as em
        events = []

        class _Rec:
            def __getattr__(self, level):
                return lambda event, **kw: events.append((event, kw))
        monkeypatch.setattr(em, "log", _Rec())
        ex = _executor()
        ex._pending_orders[11] = SimpleNamespace(trade_id="T-E", age_seconds=12.5)
        ex._pending_exit_orders[22] = SimpleNamespace(trade_id="T-X", age_seconds=31.0)
        return ex, events

    def test_emits_once_per_order_with_subsecond_ack(self, monkeypatch):
        ex, events = self._ex(monkeypatch)
        trades = [_trade_with_log(11, [("PendingSubmit", 0.0), ("PreSubmitted", 0.4)]),
                  _trade_with_log(22, [("PendingSubmit", 0.0), ("ApiPending", 0.1),
                                       ("Submitted", 0.9)])]
        ex._record_order_acks(trades)
        ex._record_order_acks(trades)             # second pass must not duplicate
        acks = {kw["order_id"]: kw for e, kw in events if e == "order_ack_latency"}
        assert set(acks) == {11, 22} and len(events) == 2
        assert acks[11]["kind"] == "entry" and acks[11]["ack_seconds"] == 0.4
        assert acks[11]["first_status"] == "PreSubmitted"
        assert acks[22]["kind"] == "exit" and acks[22]["ack_seconds"] == 0.9
        assert acks[22]["first_seen_age_seconds"] == 31.0

    def test_waits_until_an_ack_status_exists(self, monkeypatch):
        ex, events = self._ex(monkeypatch)
        ex._record_order_acks([_trade_with_log(11, [("PendingSubmit", 0.0)])])
        assert events == []
        ex._record_order_acks([_trade_with_log(11, [("PendingSubmit", 0.0),
                                                    ("Submitted", 1.2)])])
        assert [e for e, _ in events] == ["order_ack_latency"]

    @pytest.mark.parametrize("junk", [
        [None], [MagicMock()], [SimpleNamespace(order=None, log=None)],
        [SimpleNamespace(order=SimpleNamespace(orderId=11), log=[object(), object()])],
    ])
    def test_never_raises_on_malformed_trades(self, monkeypatch, junk):
        ex, _events = self._ex(monkeypatch)
        ex._record_order_acks(junk)               # must not raise

    async def test_check_fills_calls_it_and_stays_behavior_neutral(self, monkeypatch):
        ex, events = self._ex(monkeypatch)
        ex._pending_exit_orders.clear()
        ex._pending_orders.clear()
        called = []
        monkeypatch.setattr(ex, "_record_order_acks", lambda trades: called.append(trades))
        await ex.check_fills()
        assert len(called) == 1


# --------------------------------------------------------------------------- #
# 6. Documentation drift: the GUIDE tables and the config.yaml pointer block
# --------------------------------------------------------------------------- #

def _guide_block(name: str) -> str:
    """The generated block ``name`` of docs/GUIDE.md (between its BEGIN/END GENERATED markers)."""
    m = re.search(rf"<!-- BEGIN GENERATED: {name} -->\n(.*?)<!-- END GENERATED: {name} -->", GUIDE.read_text(), re.S)
    assert m, f"docs/GUIDE.md has no generated block {name!r}"
    return m.group(1)


class TestFixedLimitsDocumentation:
    def test_guide_table_is_the_registry(self):
        sec = _guide_block("fixed_limits")
        rows = re.findall(r"^\|\s*`([A-Z0-9_]+)`\s*\|\s*([0-9.]+)\s*\|", sec, re.M)
        assert {n for n, _ in rows} == {i.name for i in inv.INVARIANTS}, (
            "GUIDE 'fixed_limits' rows must be exactly the invariants registry")
        by_name = {i.name: i.value for i in inv.INVARIANTS}
        for name, shown in rows:
            assert float(shown) == float(by_name[name]), f"{name}: GUIDE says {shown}"

    def test_yaml_pointer_block_is_the_registry(self):
        rows = re.findall(r"^#\s{3}([A-Z0-9_]+) = ([0-9.]+)\s*$",
                          CONFIG_PATH.read_text(), re.M)
        assert {n: float(v) for n, v in rows} == {
            i.name: float(i.value) for i in inv.INVARIANTS}

    def test_yaml_block_is_comment_only(self):
        """The pointer block must never become keys (a key would be editable surface)."""
        raw = yaml.safe_load(CONFIG_PATH.read_text())
        assert not any(k.isupper() for k in raw), "an invariant leaked into config.yaml keys"

    def test_timeline_section_documents_every_duration_with_its_default(self):
        sec = _guide_block("order_timeline")
        s = Settings()
        shown: dict[str, float] = {}
        for line in sec.splitlines():
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) == 5 and cols[2].startswith("`"):
                keys = re.findall(r"`([a-z_.]+)`", cols[2])
                defaults = [d.strip() for d in cols[3].split("/")]
                assert len(keys) == len(defaults), line
                shown.update({k: float(d) for k, d in zip(keys, defaults)})
        for dotted in list(NEW_KEYS) + [
                "exit.exit_stop_pending_rest_seconds", "exit.exit_pending_rest_seconds",
                "exit.exit_band_ttl_seconds", "positions.entry_order_timeout_seconds"]:
            assert dotted in shown, f"GUIDE timeline does not document {dotted} with its default"
            assert shown[dotted] == float(_dig(s, dotted)), (
                f"{dotted}: GUIDE says {shown[dotted]}, model default {_dig(s, dotted)}")
        assert "30-second fast-monitor tick" in GUIDE.read_text()

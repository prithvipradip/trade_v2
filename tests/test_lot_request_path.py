"""W111 (Phase 2a): the live lot-request path.

``TradeRequest.scale_to_budget`` lets the risk manager decide the number of lots (the sizer's count after the
correlation haircut, then the largest count whose dollar gates all pass). ``positions.max_contracts_per_trade``
is the switch: at 1 nothing changes. These tests pin both halves:

* at a cap of 1, and for every explicit-count request, ``validate_trade`` reproduces the golden table that was
  generated from the code BEFORE this wave (165 scenarios, every dollar gate);
* with a cap above 1 the count is what the W108 rule says, the dollar gates judge the TOTAL risk of the lots
  placed, a failing gate trims the count instead of vetoing a trade that fits at fewer lots, and a rejection at
  one lot reads exactly as it always did;
* the orchestrator hands the manager a one-lot signal with ``scale_to_budget`` and executes what the manager
  approved, with capital-at-risk recorded for that many lots.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import ait
from ait.risk.lot_sizing import lots_for_budget
from tests.lot_request_rig import (
    BOOKS, STRUCTURES, build_manager, make_request, run_table, settings_for, shipped_settings,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "data" / "lot_request_golden.json").read_text())


def cap(settings, n: int):
    s = settings.model_copy(deep=True)
    s.positions.max_contracts_per_trade = n
    return s


def settings_cap(n: int):
    return lambda variant: cap(settings_for(variant), n)


class TestTheseTestsExerciseThisCheckout:
    def test_the_package_under_test_is_the_one_next_to_the_tests(self):
        """The editable install points at the live checkout; a worktree run must import its own src."""
        assert Path(ait.__file__).resolve().is_relative_to(ROOT / "src")


@pytest.mark.asyncio
class TestACapOfOneChangesNothing:
    async def test_explicit_count_requests_reproduce_the_golden_table(self):
        assert await run_table() == GOLDEN

    async def test_scaling_requests_at_the_shipped_cap_reproduce_it_too(self):
        assert await run_table(scale_to_budget=True) == GOLDEN

    async def test_a_cap_of_one_is_what_config_ships(self):
        assert shipped_settings().positions.max_contracts_per_trade == 1

    async def test_an_explicit_count_ignores_a_higher_cap(self):
        """Without scale_to_budget a request is judged for the count it names: min(request, sizer) = 1.
        (``max_risk`` is excluded: for an explicit request it has always been the SIZER's recommendation,
        which a higher cap makes larger than the one lot placed; nothing in src reads it.)"""
        got = await run_table(settings_for=settings_cap(3))
        keep = ("approved", "reason", "position_size")
        assert {k: {f: v[f] for f in keep} for k, v in got.items()} == {
            k: {f: v[f] for f in keep} for k, v in GOLDEN.items()}


async def validate(structure="condor_3w", *, nlv=25_000.0, book="empty", n_cap=3, settings=None, vix=16.0,
                   confidence=0.70, symbol="QQQ", can_afford=None, **extra):
    s = cap(settings or shipped_settings(), n_cap)
    rm = build_manager(s, nlv, BOOKS[book] if isinstance(book, str) else book, can_afford=can_afford)
    req = make_request(structure, vix=vix, confidence=confidence, symbol=symbol, scale_to_budget=True, **extra)
    return await rm.validate_trade(req), req


@pytest.mark.asyncio
class TestScalingWithAHigherCap:
    async def test_the_count_is_what_the_budget_rule_says(self):
        v, req = await validate()                       # $25k, 5% = $1,250, a $210 lot -> 5 by budget, cap 3
        assert lots_for_budget(210.0, 25_000.0, 0.05, max_lots=3) == 3
        assert v.approved and v.position_size == 3 and v.max_risk == pytest.approx(630.0)

    @pytest.mark.parametrize("cap_n, expected", [(1, 1), (2, 2), (3, 3), (10, 5)])
    async def test_the_cap_binds(self, cap_n, expected):
        v, _ = await validate(n_cap=cap_n)
        assert v.approved and v.position_size == expected

    async def test_a_small_account_stays_at_one_lot_whatever_the_cap(self):
        v, _ = await validate(nlv=5_000.0, n_cap=3)     # $250 budget, a $210 lot: one
        assert v.approved and v.position_size == 1 and v.max_risk == pytest.approx(210.0)

    async def test_the_aggregate_cap_sees_the_total_and_trims_the_count(self):
        # portfolio cap 20% of $25k = $5,000; open risk $4,500 leaves room for two $210 lots, not three
        book = [{"symbol": "IWM", "strategy": "iron_condor", "max_loss": 4_500.0, "expiry": "20261120"}]
        v, _ = await validate(book=book)
        assert v.approved and v.position_size == 2 and v.max_risk == pytest.approx(420.0)

    async def test_a_rejection_at_one_lot_reads_exactly_as_it_always_did(self):
        book = [{"symbol": "IWM", "strategy": "iron_condor", "max_loss": 4_800.0, "expiry": "20261120"}]
        v, req = await validate(book=book)
        assert not v.approved and v.reason.startswith("aggregate risk $5010 (open $4800 + new $210)")
        # the explicit-count request for the same structure is rejected with the identical text
        rm = build_manager(cap(shipped_settings(), 3), 25_000.0, book)
        explicit = await rm.validate_trade(make_request("condor_3w", vix=16.0, confidence=0.70))
        assert explicit.reason == v.reason

    async def test_the_per_symbol_cap_sees_the_total_and_trims_the_count(self):
        # 10% per symbol of $25k = $2,500; $1,900 already on QQQ leaves room for two $210 lots
        s = shipped_settings(0.05, portfolio=0.50, symbol=0.10)
        book = [{"symbol": "QQQ", "strategy": "long_call", "max_loss": 1_900.0, "expiry": "20261120"}]
        v, _ = await validate(book=book, settings=s)
        assert v.approved and v.position_size == 2

    async def test_buying_power_sees_the_total_and_trims_the_count(self):
        v, _ = await validate(can_afford=lambda cost: cost <= 450.0)
        assert v.approved and v.position_size == 2      # 2 x $210 = $420 fits, 3 x $210 = $630 does not

    async def test_when_even_one_lot_cannot_be_afforded_it_is_rejected_as_before(self):
        v, _ = await validate(can_afford=lambda cost: cost <= 100.0)
        assert not v.approved and v.reason == "insufficient buying power"

    async def test_the_correlation_haircut_applies_to_the_count(self):
        v, _ = await validate(book="cluster_open")      # SPY open: 0.7 x 3 lots -> 2
        assert v.approved and v.position_size == 2

    async def test_the_portfolio_delta_gate_takes_part_in_the_fit(self):
        from ait.data.options_chain import OptionContract
        from datetime import date
        s = cap(shipped_settings(), 3)
        rm = build_manager(s, 25_000.0, [])
        # delta limit 30% of $25k = $7,500; a lot adds 0.5 x 100 = 50: book delta 7,420 + 3 lots (7,570) and
        # 2 lots (7,520) break it, 1 lot (7,470) does not
        rm._portfolio_greeks.delta = 7_420.0
        opt = OptionContract(symbol="QQQ", expiry=date(2026, 11, 20), strike=700.0, right="P", bid=1.0, ask=1.1,
                             last=1.05, volume=100, open_interest=100, implied_vol=0.2, delta=0.5, con_id=0)
        req = make_request("condor_3w", vix=16.0, confidence=0.70, scale_to_budget=True)
        req.option = opt
        v = await rm.validate_trade(req)
        assert v.approved and v.position_size == 1

    async def test_a_request_without_a_defined_risk_never_scales(self):
        v, _ = await validate("no_max_loss")
        assert v.position_size <= 1

    async def test_a_request_that_names_a_count_is_judged_for_that_count(self):
        s = cap(shipped_settings(), 3)
        rm = build_manager(s, 25_000.0, [])
        req = make_request("condor_3w", vix=16.0, confidence=0.70, scale_to_budget=True)
        req.contracts = 2
        req.max_loss = 420.0                              # the explicit TOTAL for two lots
        v = await rm.validate_trade(req)
        assert v.approved and v.position_size <= 2

    async def test_a_debit_structure_scales_on_the_debit(self):
        v, _ = await validate("debit_spread")           # $140 a lot: 8 by budget, cap 3; cost = 1.40 x 3 x 100
        assert v.approved and v.position_size == 3 and v.max_risk == pytest.approx(420.0)

    async def test_buying_power_trims_a_debit_structure_on_the_debit_of_all_the_lots(self):
        # a $140 debit per lot: two lots cost $280, three $420; only $300 can be afforded
        v, _ = await validate("debit_spread", can_afford=lambda cost: cost <= 300.0)
        assert v.approved and v.position_size == 2 and v.max_risk == pytest.approx(280.0)

    async def test_the_aggregate_cap_uses_the_defined_risk_not_the_debit(self):
        """A debit whose defined risk (300) exceeds its cost (140): 6b-2 must count the risk."""
        from ait.risk.manager import TradeRequest
        s = cap(shipped_settings(0.10), 1)            # budget $500 at $5k, portfolio cap 20% = $1,000
        book = [{"symbol": "IWM", "strategy": "iron_condor", "max_loss": 800.0, "expiry": "20261120"}]
        for scale in (False, True):
            rm = build_manager(s, 5_000.0, book)
            req = TradeRequest(symbol="QQQ", strategy="bull_call_spread", direction="long", contracts=1,
                               entry_price=1.40, confidence=0.70, implied_vol=0.22, max_loss=300.0, vix=16.0,
                               iv_rank=55.0, scale_to_budget=scale)
            v = await rm.validate_trade(req)
            assert not v.approved and v.reason.startswith("aggregate risk $1100 (open $800 + new $300)"), scale

    async def test_a_zero_lot_sizer_verdict_is_still_a_refusal(self):
        """The sizer prices a debit on its premium; if that exceeds the budget while the (inconsistent) defined
        risk fits, every gate passes at one lot and the sizer's zero must still refuse the trade."""
        from ait.risk.manager import TradeRequest
        for scale in (False, True):
            rm = build_manager(cap(shipped_settings(), 3), 5_000.0, [])
            req = TradeRequest(symbol="QQQ", strategy="bull_call_spread", direction="long", contracts=1,
                               entry_price=5.00, confidence=0.70, implied_vol=0.22, max_loss=100.0, vix=16.0,
                               iv_rank=55.0, scale_to_budget=scale)
            v = await rm.validate_trade(req)
            assert not v.approved and v.reason == "position sizer returned 0 contracts", scale

    async def test_a_scaled_trade_never_exceeds_the_per_trade_budget(self):
        for nlv in (5_000.0, 10_000.0, 25_000.0, 200_000.0):
            v, _ = await validate(nlv=nlv, n_cap=10)
            if v.approved:
                assert v.max_risk <= nlv * 0.05 + 1e-6

    async def test_the_approved_size_is_stable_between_calls(self):
        a, _ = await validate()
        b, _ = await validate()
        assert (a.position_size, a.max_risk) == (b.position_size, b.max_risk)


class TestTheOrchestrator:
    pytestmark = pytest.mark.asyncio

    @pytest.fixture
    def rig(self, tmp_path, monkeypatch):
        from tests.test_hot_path_smoke import build_smoke_orchestrator
        monkeypatch.chdir(tmp_path)
        smoke = build_smoke_orchestrator()
        try:
            yield smoke
        finally:
            smoke.restore()

    @staticmethod
    def _validator(lots):
        seen = []

        async def _validate(request):
            from ait.risk.manager import TradeValidation
            seen.append(request)
            if len(seen) == 1:
                return TradeValidation(approved=True, reason="stubbed", position_size=lots, max_risk=lots * 380.0)
            return TradeValidation(approved=True, reason="revalidated", position_size=request.contracts,
                                   max_risk=request.max_loss)
        return _validate, seen

    async def _run(self, rig, validate, multiplier=1.0):
        orch = rig.orch
        orch._risk_manager.validate_trade = validate
        orch._learning.adaptor.get_strategy_multiplier = lambda strategy: multiplier

        async def _vix():
            return 16.0
        orch._market_data.get_vix = _vix
        from tests.test_hot_path_smoke import iron_condor_signal
        sig = iron_condor_signal()
        sig.quantity = 1
        await orch._try_execute(sig, 0.72, None, None)
        await rig.drain_notifications()
        return sig

    async def test_the_request_is_a_one_lot_description_with_scaling_on(self, rig):
        validate, seen = self._validator(1)
        await self._run(rig, validate)
        r = seen[0]
        assert r.scale_to_budget is True and r.contracts == 1 and r.max_loss == 380.0

    async def test_the_executor_gets_what_the_manager_approved(self, rig):
        validate, seen = self._validator(3)
        await self._run(rig, validate)
        assert rig.executed and rig.executed[0][1] == 3

    async def test_no_second_validation_runs_when_the_multiplier_is_one(self, rig):
        validate, seen = self._validator(3)
        await self._run(rig, validate)
        assert len(seen) == 1

    async def test_capital_at_risk_is_recorded_for_the_lots_placed(self, rig):
        validate, _ = self._validator(3)
        await self._run(rig, validate)
        tid = "T-SMOKE-1"
        assert float(rig.orch._state.get_state(f"trade_maxloss_{tid}")) == pytest.approx(3 * 380.0)

    async def test_a_multiplier_above_one_revalidates_an_explicit_total(self, rig):
        validate, seen = self._validator(2)
        await self._run(rig, validate, multiplier=3.0)      # approved 2 x 3.0 = 6 > 2 -> re-validate
        assert len(seen) == 2
        second = seen[1]
        assert second.contracts == 6 and second.max_loss == pytest.approx(6 * 380.0) and not second.scale_to_budget
        assert rig.executed[0][1] == 6

    async def test_the_old_one_lot_multiplier_shape_is_unchanged(self, rig):
        validate, seen = self._validator(1)
        await self._run(rig, validate, multiplier=3.0)
        assert [r.contracts for r in seen] == [1, 3]

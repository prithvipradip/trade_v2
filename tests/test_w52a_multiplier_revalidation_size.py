"""W52a: the learning multiplier's re-validation must honour the size the
risk manager actually approved.

Registered (unfixed) in Ahmed's W50 entry: after the sizing multiplier
scales a trade up, ``_try_execute`` re-validates at the larger size and,
on rejection, falls back to the gate-approved base size. But
``RiskManager.validate_trade`` can APPROVE at a smaller size than asked
(``final_contracts = min(request.contracts, size.contracts)``, then the
correlation haircut) — and that approved-but-smaller ``position_size`` was
discarded: the code read only ``revalidation.approved`` and executed the
multiplied size the gate had just cut down. The multiplier bypassed the
sizer exactly where the sizer said "less".

Live today this is inert twice over (paper_trading_mode pins the multiplier
at 1.0 and the sizer caps at 1 contract); it becomes live the day either
changes, which is the funding decision.
"""
from __future__ import annotations

import pytest

from ait.risk.manager import TradeValidation

pytestmark = pytest.mark.asyncio


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from tests.test_hot_path_smoke import build_smoke_orchestrator
    monkeypatch.chdir(tmp_path)
    smoke = build_smoke_orchestrator()
    try:
        yield smoke
    finally:
        smoke.restore()


def _validator(base: int, on_revalidation: TradeValidation):
    """First call approves ``base`` contracts; the second (the multiplier's
    re-validation) returns ``on_revalidation``. Records every request."""
    seen: list[int] = []

    async def _validate(request):
        seen.append(request.contracts)
        if len(seen) == 1:
            return TradeValidation(approved=True, reason="stubbed",
                                   position_size=base, max_risk=380.0)
        return on_revalidation
    return _validate, seen


async def _run(rig, validate):
    orch = rig.orch
    orch._risk_manager.validate_trade = validate
    orch._learning.adaptor.get_strategy_multiplier = lambda strategy: 3.0

    async def _vix():
        return 16.0
    orch._market_data.get_vix = _vix

    from tests.test_hot_path_smoke import iron_condor_signal
    sig = iron_condor_signal()
    sig.quantity = 1
    await orch._try_execute(sig, 0.72, None, None)
    await rig.drain_notifications()


class TestApprovedButSmallerIsHonoured:
    async def test_the_revalidated_size_is_the_size_executed(self, rig):
        """Multiplier wants 3; the sizer approves 2. Execute 2, not 3."""
        validate, seen = _validator(
            base=1, on_revalidation=TradeValidation(
                approved=True, reason="sizer capped", position_size=2, max_risk=760.0))
        await _run(rig, validate)

        assert seen == [1, 3], "re-validation did not run at the multiplied size"
        assert rig.executed, "the trade never executed — precondition failed"
        _signal, contracts = rig.executed[0]
        assert contracts == 2, (
            f"executed {contracts} contracts; the risk manager approved 2 of the 3 asked")
        assert "multiplier_size_capped_by_revalidation" in rig.log.names()

    async def test_a_full_approval_still_executes_the_multiplied_size(self, rig):
        validate, seen = _validator(
            base=1, on_revalidation=TradeValidation(
                approved=True, reason="all checks passed", position_size=3, max_risk=1140.0))
        await _run(rig, validate)
        assert seen == [1, 3]
        assert rig.executed[0][1] == 3
        assert "multiplier_size_capped_by_revalidation" not in rig.log.names()

    async def test_a_rejection_still_falls_back_to_the_base_size(self, rig):
        validate, seen = _validator(
            base=1, on_revalidation=TradeValidation(
                approved=False, reason="max_position_risk_pct", position_size=0))
        await _run(rig, validate)
        assert seen == [1, 3]
        assert rig.executed[0][1] == 1
        assert "multiplier_size_rejected_using_base" in rig.log.names()

    async def test_an_approval_with_no_size_keeps_the_asked_size(self, rig):
        """A stubbed/legacy approval carrying position_size=0 is not a cap —
        the pre-fix behaviour for that shape is preserved."""
        validate, seen = _validator(
            base=1, on_revalidation=TradeValidation(
                approved=True, reason="stubbed", position_size=0))
        await _run(rig, validate)
        assert rig.executed[0][1] == 3

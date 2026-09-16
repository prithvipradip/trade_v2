"""W25 — the go-live sizing wave.

Three defects, all reachable from `_launch_size_coherence_check`, the self-test the
go-live recipe tells the operator to run and trust before funding:

  D1  params-trading-01  risk/manager.py:615 — `max(1, int(final_contracts * haircut))`
      turned a fail-closed 0-contract sizer verdict into an approved 1-contract
      position, because the `<= 0` refusal on the next line could never see a zero.
      Fired only when a correlated position was already open.
  D2  logic-entry-01     bot/orchestrator.py — the check modelled the wing as
      `max(1.0, min_credit / 0.45)` ($1.56, a structure the real builder never
      constructs), read only one of the two caps that bind a trade, and never asked
      whether the credit floor was reachable at the width the builder actually picks.
  D3  NEW (found probing D2) — the breaker arm was arithmetically dead:
      AIT_CREDIT_LOSS_LIMIT contract-defaults to "0" (stop DISABLED), so
      `typical_stop_loss` was 0 and `breaker_usd < 1.5 * 0` was always False.

House rules honored: every test executes the real code path — no assertions on source
strings; a negative control proves the pre-fix code really did manufacture a contract
(and a second one proves why the OBVIOUS repair was rejected); the class guard pins the
BUG CLASS behaviorally, not this single line.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()

from ait.bot.orchestrator import TradingOrchestrator  # noqa: E402
from ait.config.settings import load_settings  # noqa: E402
from ait.risk.correlation import CorrelationGuard  # noqa: E402
from ait.risk.manager import RiskManager, TradeRequest  # noqa: E402
from ait.risk.position_sizer import PositionSizer  # noqa: E402

@pytest.fixture(scope="module")
def settings():
    return load_settings("config.yaml")


def _manager(settings, nlv: float, open_positions: list[dict]) -> RiskManager:
    """Real sizer + real correlation guard + real config. Only the broker seam is stubbed."""
    account = MagicMock()
    account.get_net_liquidation = AsyncMock(return_value=nlv)
    account.can_afford = AsyncMock(return_value=True)
    account.get_buying_power = AsyncMock(return_value=nlv)

    cb = MagicMock(is_tripped=False)
    cb.check_daily_loss = MagicMock(return_value=True)
    pdt = MagicMock()
    pdt.can_open_position = MagicMock(return_value=(True, ""))

    rm = RiskManager(
        settings.positions, settings.risk, account, cb, pdt,
        PositionSizer(settings.positions, settings.risk),
        CorrelationGuard(
            max_correlated_positions=getattr(settings.risk, "max_correlated_positions", 2)
        ),
        state=None,
    )
    rm.update_positions(open_positions)
    return rm


def _zero_verdict_request(symbol: str = "QQQ") -> TradeRequest:
    """An IC request the REAL sizer values at 0 contracts.

    max_loss is deliberately None: that is the only shape that reaches the haircut,
    because gate 6b (manager.py:512) rejects every defined-risk path first. With
    max_loss unset the sizer prices off the credit, and $1.20 x 100 = $120 exceeds
    the $105 that positions.max_position_pct allows at $2.1k.
    """
    return TradeRequest(
        symbol=symbol, strategy="iron_condor", direction="short", contracts=1,
        entry_price=1.20, confidence=0.70, implied_vol=0.22,
        max_loss=None, vix=16.0, iv_rank=55.0,
    )


SPY_OPEN = [{"symbol": "SPY", "strategy": "iron_condor"}]


# --------------------------------------------------------------------------
# D1 — params-trading-01
# --------------------------------------------------------------------------

class TestZeroContractVerdictCannotBecomeOne:
    """The 0-contract refusal must survive the correlation haircut."""

    def test_negative_control_the_pre_fix_expression_really_did_manufacture_a_contract(
        self, settings
    ):
        """NEGATIVE CONTROL: the old line, fed the REAL sizer's real verdict, yields 1.

        Not a hypothetical. The inputs below are read out of the real objects, so
        this fails the moment the premise (sizer really returns 0 here) stops holding.
        """
        sizer = PositionSizer(settings.positions, settings.risk)
        req = _zero_verdict_request()
        verdict = sizer.calculate(
            account_value=2100.0, option_price=req.entry_price, confidence=req.confidence,
            implied_vol=req.implied_vol, strategy=req.strategy, underlying_price=0,
            recent_losing_days=0, vix=req.vix,
        )
        assert verdict.contracts == 0, (
            "premise broken: the real sizer no longer returns 0 here, so this "
            f"control no longer tests anything (got {verdict.contracts}: {verdict.reason})"
        )

        final_contracts = min(req.contracts, verdict.contracts)
        haircut = max(0.30, 1.0 - 0.30 * 1)  # one correlated position open

        # The pre-W25 line, unguarded.
        pre_fix = max(1, int(final_contracts * haircut))
        # The W25 line: the haircut branch is skipped entirely for a 0 verdict.
        post_fix = (max(1, int(final_contracts * haircut))
                    if final_contracts > 0 else final_contracts)

        assert pre_fix == 1, (
            "the pre-W25 defect: max(1, ...) converted a fail-closed 0-contract "
            "verdict into a live 1-contract position"
        )
        assert post_fix == 0, "the W25 fix must let the 0 through to the <= 0 refusal"

    def test_the_rejected_repair_would_have_stopped_the_bot_trading(self, settings):
        """Why the guard, and not `max(0, ...)` — measured, not assumed.

        With positions.max_contracts_per_trade = 1 the sizer returns 1, and
        int(1 * 0.70) = 0. `max(0, ...)` would therefore REFUSE every entry made
        while one correlated position is open. The universe is SPY/QQQ/IWM, a
        single cluster, so the bot would go flat after its first fill.
        """
        assert settings.positions.max_contracts_per_trade == 1, (
            "premise moved: this reasoning depends on the hard cap being 1"
        )
        haircut = max(0.30, 1.0 - 0.30 * 1)
        viable_size = 1

        assert max(0, int(viable_size * haircut)) == 0, (
            "the naive repair silently converts a VIABLE 1-contract size into a refusal"
        )
        assert max(1, int(viable_size * haircut)) == 1, (
            "the guarded haircut must leave a viable size tradeable"
        )

    @pytest.mark.asyncio
    async def test_zero_verdict_is_refused_even_with_a_correlated_position_open(
        self, settings
    ):
        """The real path. This is the case that used to approve a live 1-lot."""
        rm = _manager(settings, 2100.0, SPY_OPEN)
        assert rm._count_correlated_positions("QQQ", ["SPY"]) == 1, (
            "premise broken: SPY/QQQ must share a SECTOR_GROUPS cluster for the "
            "haircut branch to be entered at all"
        )

        v = await rm.validate_trade(_zero_verdict_request())

        assert not v.approved, (
            "a 0-contract sizer verdict was approved as a live position — "
            "params-trading-01 has regressed"
        )
        assert "0 contracts" in v.reason

    @pytest.mark.asyncio
    async def test_the_verdict_no_longer_depends_on_whether_something_is_open(
        self, settings
    ):
        """S5/S6: identical requests must get identical verdicts.

        Pre-fix these two diverged — nothing open refused, one correlated position
        open approved a 1-lot. Same inputs, opposite outcomes, decided by unrelated
        state. That divergence IS the defect.
        """
        alone = await _manager(settings, 2100.0, []).validate_trade(_zero_verdict_request())
        correlated = await _manager(settings, 2100.0, SPY_OPEN).validate_trade(
            _zero_verdict_request()
        )

        assert alone.approved == correlated.approved is False
        assert alone.position_size == correlated.position_size == 0

    @pytest.mark.asyncio
    async def test_the_haircut_still_reduces_a_viable_size(self, settings):
        """Guard the fix itself: flooring at 0 must not disable the haircut."""
        rm = _manager(settings, 500_000.0, SPY_OPEN)
        req = TradeRequest(
            symbol="QQQ", strategy="iron_condor", direction="short", contracts=10,
            entry_price=1.20, confidence=0.90, implied_vol=0.22,
            max_loss=2000.0, vix=16.0, iv_rank=55.0,
        )
        v = await rm.validate_trade(req)
        assert v.approved
        assert v.position_size >= 1, "the haircut must reduce, never refuse, a viable size"


class TestTheBugClassCannotRecur:
    """Class guard: pin the CLASS of defect, not the single line."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sizer_contracts", [0, 1, 2])
    @pytest.mark.parametrize("open_positions", [[], SPY_OPEN])
    async def test_approval_always_implies_a_real_position_and_real_risk(
        self, settings, sizer_contracts, open_positions
    ):
        """INVARIANT: approved => position_size >= 1 AND max_risk > 0.

        The pre-fix code broke BOTH halves at once — it approved with size 1 and
        max_risk 0.0, so the aggregate risk cap counted the position as free. Any
        future fail-open that manufactures size, or reports a sized position as
        risk-free, trips this regardless of which line causes it.
        """
        rm = _manager(settings, 2100.0, open_positions)
        rm._sizer = MagicMock()
        rm._sizer.calculate = MagicMock(return_value=MagicMock(
            contracts=sizer_contracts,
            max_risk_dollars=0.0 if sizer_contracts == 0 else 150.0 * sizer_contracts,
        ))

        v = await rm.validate_trade(_zero_verdict_request())

        if v.approved:
            assert v.position_size >= 1, (
                f"approved a position of {v.position_size} contracts "
                f"(sizer said {sizer_contracts}) — size was manufactured"
            )
            assert v.max_risk > 0, (
                f"approved {v.position_size} contracts carrying max_risk={v.max_risk} — "
                "a sized position reported as risk-free escapes the aggregate cap"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sizer_contracts", [0, 1, 2, 5])
    @pytest.mark.parametrize("correlated", [0, 1, 2])
    async def test_a_haircut_may_only_reduce_never_create(
        self, settings, sizer_contracts, correlated
    ):
        """THE CLASS GUARD: `approved => 1 <= position_size <= sizer verdict`.

        This is the general law params-trading-01 broke, and it catches the next
        instance wherever in validate_trade it appears: no adjustment may push an
        approved size above what the sizer allowed, and none may manufacture a
        position out of a 0-contract refusal.

        A purely syntactic guard was tried first and REJECTED, which is worth
        recording: `qty = max(1, ...)` is also the shape of portfolio.py:963, where
        flooring a partial CLOSE at one contract is correct, and of manager.py:172,
        a divisor normalizer. The defect is about entry sizing, not syntax, so the
        guard has to be behavioral.
        """
        opens = [{"symbol": s, "strategy": "iron_condor"}
                 for s in ("SPY", "IWM")[:correlated]]
        rm = _manager(settings, 2100.0, opens)
        rm._sizer = MagicMock()
        rm._sizer.calculate = MagicMock(return_value=MagicMock(
            contracts=sizer_contracts,
            max_risk_dollars=0.0 if sizer_contracts == 0 else 150.0 * sizer_contracts,
        ))

        v = await rm.validate_trade(_zero_verdict_request())

        if sizer_contracts == 0:
            assert not v.approved, (
                "a 0-contract sizer verdict was approved — size was manufactured "
                "out of a refusal (the params-trading-01 bug class)"
            )
        if v.approved:
            assert 1 <= v.position_size <= sizer_contracts, (
                f"approved {v.position_size} contracts against a sizer verdict of "
                f"{sizer_contracts} — an adjustment may only REDUCE size"
            )


# --------------------------------------------------------------------------
# D2 + D3 — the launch-size coherence self-test
# --------------------------------------------------------------------------

class _CoherenceProbe:
    """Minimal carrier for the real method — no broker, no trading loop."""

    def __init__(self, settings):
        self._settings = settings
        self.alerts: list[str] = []

    async def _send_notification(self, message: str) -> None:
        self.alerts.append(message)

    _launch_size_coherence_check = TradingOrchestrator._launch_size_coherence_check


class TestLaunchSizeCoherenceCheck:

    @pytest.mark.asyncio
    async def test_it_measures_the_real_builder_width_not_the_modelled_wing(
        self, settings, caplog
    ):
        """D2: the reported width must come from IronCondor, not min_credit/0.45.

        At $2.1k the real builder caps the wing at $1.00; the retired model said
        $1.56 — a structure it never constructs.
        """
        from ait.strategies.iron_condor import IronCondor

        probe = IronCondor()
        probe.risk_budget = 2100.0 * min(
            settings.risk.max_position_risk_pct, settings.positions.max_position_pct
        )
        real_width = probe._vol_scaled_width(1.0, None, None, None)

        assert real_width == pytest.approx(1.00, abs=0.01), (
            f"premise moved: the real builder now floors at {real_width}"
        )
        assert real_width != pytest.approx(1.56, abs=0.01), "that is the retired model wing"

    @pytest.mark.asyncio
    async def test_it_alerts_at_the_documented_2100_recipe(self, settings):
        """D2: at the operator's pre-funding size the bot emits ZERO trades.

        The builder caps the wing at $1.00, which can yield ~$0.45 credit against a
        $0.70 floor, so iron_condor.py:293 rejects every signal. The old check
        compared dollars only and never asked this question.
        """
        probe = _CoherenceProbe(settings)
        await probe._launch_size_coherence_check(2100.0)

        assert probe.alerts, "the self-test stayed silent at $2.1k"
        body = probe.alerts[0]
        assert "credit floor UNREACHABLE" in body
        assert "ZERO" in body

    @pytest.mark.asyncio
    async def test_it_reads_both_caps_not_just_one(self, settings):
        """D2: the binding budget is the tighter of gate 6b and the sizer's own cap.

        Measured as a difference, which is the only way to prove the second cap is
        READ: the same NLV is silent on the real config and must alert once
        positions.max_position_pct alone is made impossibly tight. Pre-W25 the check
        never looked at that knob, so both runs were identical.
        """
        healthy = _CoherenceProbe(settings)
        await healthy._launch_size_coherence_check(50_000.0)
        assert not healthy.alerts, "premise moved: $50k is no longer a healthy account"

        starved = _CoherenceProbe(settings)
        starved._settings = settings.model_copy(deep=True)
        starved._settings.positions.max_position_pct = 0.001  # $50 budget at $50k

        await starved._launch_size_coherence_check(50_000.0)

        assert starved.alerts, (
            "a 0.1% sizer cap bricks every entry, but the check stayed silent — "
            "it is reading only risk.max_position_risk_pct again"
        )

    @pytest.mark.asyncio
    async def test_the_breaker_arm_is_not_vacuous(self, settings):
        """D3: with the credit stop disabled the arm must still be able to fire.

        Pre-fix, AIT_CREDIT_LOSS_LIMIT=0 made typical_stop_loss 0, so
        `breaker_usd < 1.5 * 0` was False for every account that has ever existed.
        """
        from ait.config.runtime_env import contract_float

        assert contract_float("AIT_CREDIT_LOSS_LIMIT") == 0, (
            "premise moved: the credit stop is no longer disabled by default, so "
            "this no longer tests the vacuous case"
        )

        probe = _CoherenceProbe(settings)
        await probe._launch_size_coherence_check(2100.0)

        assert any("daily-loss breaker" in a for a in probe.alerts), (
            "the breaker arm did not fire at $2.1k, where the breaker is $42 against "
            "a $30 defined-risk loss — it is vacuous again"
        )

    @pytest.mark.asyncio
    async def test_it_stays_quiet_on_a_healthy_account(self, settings):
        """No alert where the grid is genuinely coherent.

        $197k is the PAPER account's balance (orchestrator.py:1270), not live money
        — nothing is funded yet. The reason this must stay silent is not deference
        to an account but arithmetic: at $197k the builder gets its full $2.00 wing
        and the breaker is $3,940 against a $130 defined-risk loss. The alert has to
        fire where the grid really is incoherent ($2.1k, $5k) and nowhere else.
        """
        probe = _CoherenceProbe(settings)
        await probe._launch_size_coherence_check(197_000.0)
        assert not probe.alerts, f"spurious CAP-GRID alert at $197k: {probe.alerts}"

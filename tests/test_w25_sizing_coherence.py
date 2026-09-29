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

    @pytest.mark.asyncio
    async def test_negative_control_the_pre_fix_expression_really_did_manufacture_a_contract(
        self, settings, monkeypatch
    ):
        """NEGATIVE CONTROL: the pre-W25 line, run through the REAL validate_trade.

        W31 REBUILD (R26 F7 w25-negative-controls-reimplement-the-line-under-test).
        This used to compute `max(1, int(final_contracts * haircut))` in the test
        body and assert the answer was 1 — arithmetic on test-local values that
        never entered risk/manager.py. R26 proved it: with validate_trade replaced
        by an always-approve stub (the maximal form of the very defect it claims
        to control) this test still PASSED. A control that re-implements the line
        it controls cannot detect that line changing.

        Now it substitutes the old expression at the real seam and drives the real
        validate_trade, so it fails if the defect is reintroduced ANYWHERE on that
        path — and the shipped code is exercised in the same breath.
        """
        rm = _manager(settings, 2100.0, SPY_OPEN)
        req = _zero_verdict_request()

        assert rm._count_correlated_positions("QQQ", ["SPY"]) == 1, (
            "premise broken: SPY/QQQ must share a cluster or the haircut branch "
            "is never entered"
        )
        verdict = rm._sizer.calculate(
            account_value=2100.0, option_price=req.entry_price,
            confidence=req.confidence, implied_vol=req.implied_vol,
            strategy=req.strategy, underlying_price=0, recent_losing_days=0,
            vix=req.vix,
        )
        assert verdict.contracts == 0, (
            "premise broken: the real sizer no longer returns 0 here, so this "
            f"control no longer tests anything (got {verdict.contracts}: {verdict.reason})"
        )

        # The pre-W25 line: max(1, ...) with NO `final_contracts > 0` guard.
        monkeypatch.setattr(
            RiskManager, "_haircut_contracts",
            lambda self, n, c: (max(1, int(n * max(0.30, 1.0 - 0.30 * c)))
                                if c > 0 else n),
        )
        defect = await rm.validate_trade(req)

        assert defect.approved and defect.position_size == 1, (
            "premise broken: the pre-W25 expression no longer manufactures a "
            f"contract from a 0 verdict (got approved={defect.approved} "
            f"size={defect.position_size}) — this control is no longer negative"
        )
        assert defect.max_risk == 0.0, (
            "the full defect: the manufactured position was also invisible to the "
            f"aggregate capital-at-risk cap (max_risk={defect.max_risk})"
        )

        # And the shipped line, on the same inputs, refuses.
        monkeypatch.undo()
        shipped = await _manager(settings, 2100.0, SPY_OPEN).validate_trade(req)
        assert not shipped.approved, (
            f"a 0-contract sizer verdict was approved: {shipped.reason}"
        )

    @pytest.mark.asyncio
    async def test_the_rejected_repair_would_have_stopped_the_bot_trading(
        self, settings, monkeypatch
    ):
        """Why the guard, and not `max(0, ...)` — measured through the real path.

        W31 REBUILD, same finding as the control above: this asserted
        `max(0, int(1 * 0.70)) == 0` over two test-local literals and never
        entered risk/manager.py.

        The reasoning it records is still right and still worth pinning. With
        positions.max_contracts_per_trade = 1 the sizer returns 1 and
        int(1 * 0.70) = 0, so `max(0, ...)` REFUSES every entry made while one
        correlated position is open. The universe is SPY/QQQ/IWM — a single
        cluster — so the bot goes flat after its first fill. That is a trading
        behaviour change masquerading as a fail-open fix, which is why W25
        rejected it. Now the claim is made against the real validate_trade.
        """
        assert settings.positions.max_contracts_per_trade == 1, (
            "premise moved: this reasoning depends on the hard cap being 1"
        )
        # A genuinely VIABLE request — big account, small structure.
        req = TradeRequest(
            symbol="QQQ", strategy="iron_condor", direction="short", contracts=1,
            entry_price=1.20, confidence=0.70, implied_vol=0.22,
            max_loss=None, vix=16.0, iv_rank=55.0,
        )
        shipped = await _manager(settings, 500_000.0, SPY_OPEN).validate_trade(req)
        assert shipped.approved and shipped.position_size == 1, (
            "premise broken: this request is no longer viable with a correlated "
            f"position open, so the comparison below is meaningless ({shipped.reason})"
        )

        monkeypatch.setattr(
            RiskManager, "_haircut_contracts",
            lambda self, n, c: (max(0, int(n * max(0.30, 1.0 - 0.30 * c)))
                                if c > 0 else n),
        )
        repaired = await _manager(settings, 500_000.0, SPY_OPEN).validate_trade(req)

        assert not repaired.approved, (
            "the rejected repair no longer refuses a viable 1-lot — the reason "
            "W25 chose the guard over max(0, ...) has stopped applying, and the "
            "choice should be revisited rather than silently kept"
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
        self.loop_errors: list[tuple[str, str]] = []

    async def _send_notification(self, message: str) -> None:
        self.alerts.append(message)

    async def _note_loop_error(self, kind: str, error: str, *, threshold: int = 5) -> None:
        self.loop_errors.append((kind, error))

    _launch_size_coherence_check = TradingOrchestrator._launch_size_coherence_check
    # W31: the check now runs the REAL builder, so the carrier needs the
    # helpers that do it. `_probe_grid_of` must be re-wrapped: copying a
    # staticmethod off the class onto another class turns it back into an
    # instance method, and it would silently receive `self` as its chain.
    _probe_cap_grid = TradingOrchestrator._probe_cap_grid
    _build_probe_chains = TradingOrchestrator._build_probe_chains
    _run_probe_builder = TradingOrchestrator._run_probe_builder
    _probe_grid_of = staticmethod(TradingOrchestrator._probe_grid_of)


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
    async def test_the_remedy_it_advises_actually_works(self, settings):
        """The advised budget must be the one that FIXES it — checked by applying it.

        The first cut of this message advised ~$86, the max LOSS of the needed
        structure. The operator sets the BUDGET, which reaches the wing only through
        `affordable = risk_budget / (100 * (1 - min_ratio))`, so the real answer is
        $140. Advice that does not work is worse than no advice: it sends the
        operator to move a knob that leaves the account still unable to trade.
        """
        import re

        from ait.strategies.iron_condor import IronCondor

        probe = _CoherenceProbe(settings)
        await probe._launch_size_coherence_check(2100.0)
        advised = re.search(r"per-trade budget of ~\$(\d+)", probe.alerts[0])
        assert advised, f"no budget advice in: {probe.alerts[0]}"

        # W31: apply exactly what it advised, to the REAL builder on the REAL
        # probe chains — the same question the check itself asked. This used to
        # re-derive `0.45 * _vol_scaled_width(1.0, ...)`, i.e. it checked the
        # advice against the retired $1-probe model rather than against the
        # builder, so it could bless advice the builder would refuse.
        probes = probe._probe_cap_grid(float(advised.group(1)))["probes"]
        assert any(p["max_loss"] is not None for p in probes), (
            f"the check advised a ${advised.group(1)} per-trade budget, but at "
            f"that budget the real builder STILL emits nothing: "
            f"{[(p['grid'], p['reason']) for p in probes]} — the advice does not work"
        )

    @pytest.mark.asyncio
    async def test_it_reads_both_caps_not_just_one(self, settings):
        """D2: positions.max_position_pct must reach the verdict, not just gate 6b.

        W31 REWRITE. This used to assert the $50k run was SILENT and the starved
        run alerted. Post-W31 both runs alert — $50k trips the breaker arm (see
        TestTheBreakerArmSeesTheLargestStructure) — so "silent vs alerting" would
        pass while measuring nothing, the vacuity this file exists to prevent.
        The claim was never about silence: it is that the sizer cap is READ. So
        compare the PROBLEM SETS, and require a problem that names that knob and
        appears only when it binds.
        """
        healthy = _CoherenceProbe(settings)
        await healthy._launch_size_coherence_check(50_000.0)
        assert not any("positions.max_position_pct" in a for a in healthy.alerts), (
            "premise moved: the sizer cap already binds at $50k on the real "
            f"config, so this can no longer isolate it: {healthy.alerts}"
        )

        starved = _CoherenceProbe(settings)
        starved._settings = settings.model_copy(deep=True)
        starved._settings.positions.max_position_pct = 0.001  # $50 cap at $50k

        await starved._launch_size_coherence_check(50_000.0)

        assert any("positions.max_position_pct" in a for a in starved.alerts), (
            "a 0.1% sizer cap caps a position at $50 while the builder is funded "
            "$1,500 and emits a structure risking far more — the check never "
            "named that knob, so it is reading only risk.max_position_risk_pct "
            f"again: {starved.alerts}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("nlv,expected", [
        (2_100.0, "emits ZERO"),          # nothing is built: A1, not the breaker
        (10_000.0, "daily-loss breaker"),
        (201_475.0, "daily-loss breaker"),
    ])
    async def test_the_breaker_arm_is_not_vacuous(self, settings, nlv, expected):
        """D3: with the credit stop disabled the arm must still be able to fire.

        Pre-W25, AIT_CREDIT_LOSS_LIMIT=0 made typical_stop_loss 0, so
        `breaker_usd < 1.5 * 0` was False for every account that has ever existed.

        W31 REWRITE. W25's fix replaced that with the cost of a $1-probe wing,
        which is a CONSTANT $130 above ~$6k NLV — so the arm fired only at $2.1k,
        the one size this test happened to check, and was fail-open everywhere the
        operator might actually fund. Its old premise ("$42 breaker vs a $30
        defined-risk loss") was an artefact of that probe. The arm is now sized
        from the largest structure the real builder emits, so the vacuity test
        has to span the range: below the trading floor nothing is built at all,
        and at both real funding levels the breaker cannot absorb one max loss.
        """
        from ait.config.runtime_env import contract_float

        assert contract_float("AIT_CREDIT_LOSS_LIMIT") == 0, (
            "premise moved: the credit stop is no longer disabled by default, so "
            "this no longer tests the vacuous case"
        )

        # PLAN 2026-09-29: the live config's max_position_risk_pct was lowered
        # to 1.33% (W52a item 3, the two-knob unification), so the two
        # "daily-loss breaker" cases below no longer occur for real. Inject
        # R26's original 3% cap to keep proving the arm fires on an
        # incoherent config; the "emits ZERO" case (below the trading floor)
        # is unaffected by this knob and keeps using the real settings.
        s = settings
        if expected == "daily-loss breaker":
            s = settings.model_copy(deep=True)
            s.risk.max_position_risk_pct = 0.03
        probe = _CoherenceProbe(s)
        await probe._launch_size_coherence_check(nlv)

        assert any(expected in a for a in probe.alerts), (
            f"at NLV ${nlv:,.0f} the check did not report {expected!r} — "
            f"got: {probe.alerts}"
        )

    @pytest.mark.asyncio
    async def test_it_speaks_at_197k_because_one_loss_outruns_the_breaker(self, settings):
        """W31 INVERSION of test_it_stays_quiet_on_a_healthy_account.

        That test asserted SILENCE at $197k on this reasoning: "at $197k the
        builder gets its full $2.00 wing and the breaker is $3,940 against a $130
        defined-risk loss". Both numbers came from the $1 probe. The real builder
        at this budget emits a ~$50 wing risking ~$4,400, and R26 measured the
        live book's own largest condor at $5,356 against a $4,030 breaker: three
        of eleven real condors carried a single max loss exceeding the ENTIRE
        daily breaker while this check reported problems=0.

        So the silence WAS the defect, and pinning it was pinning the bug. What
        must hold instead is that the account is told.

        PLAN 2026-09-29: injects R26's own 3% cap since the live config is now
        coherent by design (W52a item 3) and would log problems=0.
        """
        s = settings.model_copy(deep=True)
        s.risk.max_position_risk_pct = 0.03
        probe = _CoherenceProbe(s)
        await probe._launch_size_coherence_check(197_000.0)

        assert any("daily-loss breaker" in a for a in probe.alerts), (
            "at $197k one max-loss condor exceeds the whole daily breaker and the "
            f"check said nothing: {probe.alerts}"
        )

    @pytest.mark.asyncio
    async def test_a_genuinely_coherent_grid_is_silent(self, settings):
        """CONTROL: the check must still be capable of saying nothing.

        Replaces the negative half of the inverted test above, which is the part
        that was worth keeping: an alert on every input is as useless as an alert
        on none. Widen the breaker until it can absorb the structure the builder
        builds, and the check must go quiet — proving the arms measure the grid
        rather than firing unconditionally.
        """
        s = settings.model_copy(deep=True)
        s.risk.max_daily_loss_pct = 0.10  # 10% of NLV absorbs any 3%-funded trade
        probe = _CoherenceProbe(s)

        await probe._launch_size_coherence_check(197_000.0)

        assert not probe.alerts, (
            f"the check alerts even on a coherent grid: {probe.alerts}"
        )

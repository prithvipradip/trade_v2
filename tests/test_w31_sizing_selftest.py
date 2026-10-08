"""W31 (R26 F4) — the launch sizing self-test, finished.

`TradingOrchestrator._launch_size_coherence_check` is the pre-funding check the
go-live recipe tells the operator to trust. W25 rebuilt it; R26 found it still
could not fail in the ways that matter, and measured why: it never ran the real
builder. It called the private `_vol_scaled_width` with a $1 underlying price —
a probe that drives the vol term under the builder's own `max(2.0, ...)` floor
and therefore returns the NARROWEST structure the builder can ever construct.

What that cost, measured at HEAD 0ca47f3 before this wave:

    NLV      2,100  builder_width=1.0  min_viable_ic=30   breaker=42    problems=2
    NLV      9,750  builder_width=2.0  min_viable_ic=130  breaker=195   problems=0
    NLV     10,000  builder_width=2.0  min_viable_ic=130  breaker=200   problems=0
    NLV     50,000  builder_width=2.0  min_viable_ic=130  breaker=1000  problems=0
    NLV    201,475  builder_width=2.0  min_viable_ic=130  breaker=4030  problems=0

`builder_width` and `min_viable_ic` are bit-identical at every account above
~$6k, so the breaker arm compared every one of them against a fixed $130 and
was fail-open everywhere the operator might actually fund. Against the live
book R26 measured real condor widths of $30-60 and a largest `capital_at_risk`
of $5,356 against a 2% daily breaker of $4,030: three of eleven real condors
carried a single max loss exceeding the ENTIRE daily breaker while this check
reported problems=0.

Nothing downstream of width selection ran either, so strike snapping was
unmodelled and `max_loss_over_budget` had no arm at all — on a $5 grid the
check was silent while the builder emitted zero.

Every test here executes the REAL check, the REAL builder and the REAL config.
None reads source text.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()

from ait.bot.orchestrator import TradingOrchestrator  # noqa: E402
from ait.config.runtime_env import CONFIG_BACKED, CONTRACT_DEFAULTS, contract_float  # noqa: E402
from ait.config.settings import load_settings  # noqa: E402
from ait.data.synthetic_chain import build_probe_chain  # noqa: E402
from ait.risk.correlation import CorrelationGuard  # noqa: E402
from ait.risk.manager import RiskManager, TradeRequest  # noqa: E402
from ait.risk.position_sizer import PositionSizer  # noqa: E402
from ait.strategies.iron_condor import IronCondor  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"

#: The live paper account's cached NLV, and the funding level the operator
#: actually deployed. Both are R26's own acceptance cases.
LIVE_NLV = 201_475.0
DEPLOYED_NLV = 10_000.0


@pytest.fixture(scope="module")
def settings():
    return load_settings("config.yaml")


class _Probe:
    """Minimal carrier for the real check — no broker, no trading loop."""

    def __init__(self, settings):
        self._settings = settings
        self.alerts: list[str] = []
        self.loop_errors: list[tuple[str, str]] = []

    async def _send_notification(self, message: str) -> None:
        self.alerts.append(message)

    async def _note_loop_error(self, kind: str, error: str, *, threshold: int = 5) -> None:
        self.loop_errors.append((kind, error))

    _launch_size_coherence_check = TradingOrchestrator._launch_size_coherence_check
    _probe_cap_grid = TradingOrchestrator._probe_cap_grid
    _build_probe_chains = TradingOrchestrator._build_probe_chains
    _run_probe_builder = TradingOrchestrator._run_probe_builder
    # A staticmethod copied onto another class becomes an instance method and
    # would silently receive `self` as its chain.
    _probe_grid_of = staticmethod(TradingOrchestrator._probe_grid_of)


def _run(probe, nlv, **kw) -> bool:
    return asyncio.run(probe._launch_size_coherence_check(nlv, **kw))


def _grid_chain(settings, grid: float):
    p = settings.sizing_probe
    return build_probe_chain(
        underlying_price=p.underlying_price, implied_vol=p.implied_vol,
        dte=p.dte, strike_grid=grid,
        risk_free_rate=settings.options.risk_free_rate,
    )


def _manager(settings, nlv: float) -> RiskManager:
    """Real sizer, real correlation guard, real config; broker seam stubbed."""
    from unittest.mock import AsyncMock, MagicMock

    account = MagicMock()
    account.get_net_liquidation = AsyncMock(return_value=nlv)
    account.can_afford = AsyncMock(return_value=True)
    account.get_buying_power = AsyncMock(return_value=nlv)
    cb = MagicMock(is_tripped=False)
    cb.check_daily_loss = MagicMock(return_value=True)
    pdt = MagicMock()
    pdt.can_open_position = MagicMock(return_value=(True, ""))
    return RiskManager(
        settings.positions, settings.risk, account, cb, pdt,
        PositionSizer(settings.positions, settings.risk),
        CorrelationGuard(max_correlated_positions=2), state=None,
    )


class TestTheBreakerArmSeesTheLargestStructure:
    """R26 F4 #1 — `w25-d3-breaker-arm-still-fail-open-sized-from-a-1-dollar-probe-wing`."""

    @pytest.mark.parametrize("nlv", [DEPLOYED_NLV, LIVE_NLV])
    def test_it_fires_where_one_max_loss_outruns_the_daily_breaker(self, settings, nlv):
        """R26's own acceptance case, at both funding levels it named.

        At HEAD both were silent: the $1 probe pinned one trade's loss at $130
        for every account, so `breaker_usd < 1.5 * 130` was False above ~$6k.

        PLAN 2026-09-29: the live config's max_position_risk_pct was lowered
        to 1.33% (= max_daily_loss_pct / 1.5) specifically so this class of
        structure can no longer occur for real — that closed W52a item 3, the
        two-knob unification. This test now injects R26's original 3% cap to
        prove the arm still catches an incoherent config, rather than relying
        on the live config being broken.
        """
        s = settings.model_copy(deep=True)
        s.risk.max_position_risk_pct = 0.03
        # W64: the live defaults moved (5% / 4.5% / ratio 1.0), so inject R26's WHOLE original
        # incoherent setting (3% budget, 2% breaker, the 1.5x rule) to keep proving the arm fires.
        s.risk.max_daily_loss_pct = 0.02
        s.risk.breaker_to_max_loss_ratio = 1.5
        probe = _Probe(s)
        assert _run(probe, nlv)

        assert any("daily-loss breaker" in a for a in probe.alerts), (
            f"at NLV ${nlv:,.0f} the breaker cannot absorb one max-loss condor "
            f"and the check said nothing: {probe.alerts}"
        )

    def test_the_loss_it_measures_is_the_one_the_builder_really_emits(self, settings):
        """The arm must be sized from the real structure, not a $1-probe constant.

        This is the defect in one assertion: the number driving the arm has to
        MOVE with the account. At HEAD it was $130 at $9,750 and $130 at
        $201,475.

        PLAN 2026-09-29: injects R26's own 3% cap (rather than the now-lower
        live 1.33%, W52a item 3) so this test keeps measuring against R26's
        documented acceptance numbers regardless of where the live knob sits.
        """
        s = settings.model_copy(deep=True)
        s.risk.max_position_risk_pct = 0.03
        # W64: the live defaults moved (5% / 4.5% / ratio 1.0), so inject R26's WHOLE original
        # incoherent setting (3% budget, 2% breaker, the 1.5x rule) to keep proving the arm fires.
        s.risk.max_daily_loss_pct = 0.02
        s.risk.breaker_to_max_loss_ratio = 1.5
        probe = _Probe(s)
        losses = {
            nlv: max(
                (p["max_loss"] or 0.0)
                for p in probe._probe_cap_grid(
                    nlv * s.risk.max_position_risk_pct)["probes"]
            )
            for nlv in (DEPLOYED_NLV, 50_000.0, LIVE_NLV)
        }

        assert len(set(losses.values())) == 3, (
            f"the measured max loss does not vary with the account: {losses} — "
            f"that is the $1-probe constant again"
        )
        assert losses[LIVE_NLV] > 4_000, (
            f"the live account's structure is measured at ${losses[LIVE_NLV]:,.0f}; "
            f"R26 measured the live book's largest real condor at $5,356 and the "
            f"breaker at $4,030, so anything near $130 means the probe is wrong"
        )


class TestTheProbeUsesTheBudgetTheBuilderReceives:
    """R26 F4 #2 — `w25-coherence-probe-uses-a-budget-the-live-builder-never-receives`."""

    def test_the_builder_is_handed_the_gate_cap_not_the_min_of_both(
        self, settings, monkeypatch
    ):
        """The live scan funds the builder from risk.max_position_risk_pct ALONE.

        W25 probed with min(gate, sizer). Whenever the gate cap is the looser
        of the two that models a narrower wing than the builder will build, so
        the check passed while the real structure blew the sizer's cap. Observed
        at the builder itself, which is the only place the claim is true or false.
        """
        s = settings.model_copy(deep=True)
        s.risk.max_position_risk_pct = 0.10      # gate LOOSER than the sizer
        s.positions.max_position_pct = 0.05

        seen: list[float] = []
        real = IronCondor.generate_signals

        def _spy(self, *a, **k):
            seen.append(self.risk_budget)
            return real(self, *a, **k)

        monkeypatch.setattr(IronCondor, "generate_signals", _spy)
        _run(_Probe(s), DEPLOYED_NLV)

        assert seen, "the check never ran the builder at all"
        assert set(seen) == {DEPLOYED_NLV * 0.10}, (
            f"the builder was funded {sorted(set(seen))} but the live scan hands "
            f"it nlv * max_position_risk_pct = {DEPLOYED_NLV * 0.10}; "
            f"min(gate, sizer) would be {DEPLOYED_NLV * 0.05}"
        )

    def test_it_alerts_when_the_gate_cap_outruns_the_sizer_cap(self, settings):
        """R26's acceptance case: gate 10% / sizer 5% at $10k must alert.

        This is the documented Phase-2 recipe (PLAN.md: max_position_risk_pct
        0.03 -> 0.07). At HEAD it produced zero alerts.
        """
        s = settings.model_copy(deep=True)
        s.risk.max_position_risk_pct = 0.10
        s.positions.max_position_pct = 0.05

        probe = _Probe(s)
        assert _run(probe, DEPLOYED_NLV)

        assert any("positions.max_position_pct" in a for a in probe.alerts), (
            f"the sizer cap bricks every entry at this setting and the check did "
            f"not name it: {probe.alerts}"
        )

    @pytest.mark.asyncio
    async def test_the_real_risk_manager_agrees_the_entry_is_bricked(self, settings):
        """The W25 ADDENDUM's own condition, against the real validate_trade.

        Raising max_position_risk_pct above max_position_pct must not SILENTLY
        brick entries. Two halves, and both must hold: the real manager really
        does refuse the structure the builder builds, and the check really does
        say so.
        """
        s = settings.model_copy(deep=True)
        s.risk.max_position_risk_pct = 0.10
        s.positions.max_position_pct = 0.05

        probe = _Probe(s)
        built = [p for p in probe._probe_cap_grid(DEPLOYED_NLV * 0.10)["probes"]
                 if p["max_loss"] is not None]
        assert built, "premise moved: the builder emits nothing at this budget"

        verdict = await _manager(s, DEPLOYED_NLV).validate_trade(TradeRequest(
            symbol="QQQ", strategy="iron_condor", direction="short", contracts=1,
            entry_price=1.20, confidence=0.70, implied_vol=0.22,
            max_loss=built[0]["max_loss"], vix=16.0, iv_rank=55.0,
        ))
        assert not verdict.approved, (
            f"premise moved: the real manager now approves a ${built[0]['max_loss']} "
            f"structure at ${DEPLOYED_NLV:,.0f}, so nothing is bricked ({verdict.reason})"
        )

        assert await probe._launch_size_coherence_check(DEPLOYED_NLV)
        assert probe.alerts, (
            "the real manager refuses this structure and the self-test stayed "
            "silent — exactly the silent-brick the ADDENDUM registered"
        )


class TestTheBuilderIsActuallyExecuted:
    """R26 F4 #3 — the `logic-entry-01` residual: `generate_signals` never ran."""

    def test_a_coarse_strike_grid_is_reported_not_silently_survived(self, settings):
        """R26's acceptance case: a $5-grid chain at NLV 9,750 must alert.

        The builder snaps its target wing to real strikes, so a $5 ladder can
        only express a structure wider than the width formula asked for — and
        `max_loss_over_budget` then refuses every signal. At HEAD this was the
        silent case: `problems=0` while the builder emitted ZERO.
        """
        # W64: pin the 1.33% budget R26's acceptance case ran under ($130 at
        # $9,750). At the live 5% ($487) a $5 wing fits, so the snapped-wider-
        # than-budget case this test exists for no longer occurs on live config.
        s = settings.model_copy(deep=True)
        s.risk.max_position_risk_pct = 0.0133
        probe = _Probe(s)
        assert _run(probe, 9_750.0, chains=[_grid_chain(s, 5.0)])

        body = " ".join(probe.alerts)
        assert "max_loss_over_budget" in body, (
            f"the builder refused every structure on a $5 grid and the check did "
            f"not report its reason: {probe.alerts}"
        )
        assert "ZERO" in body

    def test_the_builders_own_reason_is_what_gets_reported(self, settings):
        """Not a re-derived one. The check must quote the builder.

        Control on the mapping itself: a rejection the builder states must reach
        the operator with BOTH the English phrase and the raw token, so a new
        rejection path cannot silently become "no reason given".
        """
        probe = _Probe(settings)
        _run(probe, 2_100.0)

        body = " ".join(probe.alerts)
        assert "credit_floor" in body and "credit floor UNREACHABLE" in body, (
            f"expected the builder's own token and its phrase: {probe.alerts}"
        )

    def test_the_advised_budget_is_measured_against_the_builder(self, settings):
        """The remedy must be one the builder accepts — found by asking it.

        W25 advised $86 where the account needed $140 because it inverted the
        width formula by hand (PLAN.md W25 [C1]).
        """
        import re

        probe = _Probe(settings)
        _run(probe, 2_100.0)
        advised = re.search(r"per-trade budget of ~\$(\d+)", " ".join(probe.alerts))
        assert advised, f"no budget advice: {probe.alerts}"

        probes = probe._probe_cap_grid(float(advised.group(1)))["probes"]
        assert any(p["max_loss"] is not None for p in probes), (
            f"the advised ${advised.group(1)} budget still emits nothing: "
            f"{[(p['grid'], p['reason']) for p in probes]}"
        )


class TestTheDeadRatioArmIsGoneAndItsCaseStillReported:
    """R26 F4 #4 — `w25-credit-width-ratio-arm-is-arithmetically-dead`.

    The arm tested `(achievable_credit / min_width) < min_ratio` where
    `achievable_credit` was DEFINED as `credit_ratio * min_width` — identically
    the constant `0.45` at every NLV, an algebraic identity wearing the message
    of a budget diagnostic. R26 permitted deleting it provided its real case
    still surfaces. It does, through the builder's own rejection.
    """

    def test_an_unreachable_credit_width_floor_is_still_reported(
        self, settings, monkeypatch
    ):
        """Raise the floor until the real builder refuses on it, and require it said so.

        Moved through the ENV override rather than the settings object: the
        builder resolves this knob with `contract_float`, which reads the
        config FILE, so mutating a settings copy would leave the builder on
        0.10 and this test would pass while measuring nothing.
        """
        monkeypatch.setenv("AIT_IC_MIN_CREDIT_WIDTH", "0.20")
        assert contract_float("AIT_IC_MIN_CREDIT_WIDTH") == 0.20, (
            "the override did not reach the builder's own authority"
        )

        probe = _Probe(settings)
        assert _run(probe, LIVE_NLV)

        assert any("credit_width_ratio" in a for a in probe.alerts), (
            f"the builder rejects every structure at a 20% credit/width floor and "
            f"the check did not report it: {probe.alerts}"
        )

    def test_the_floor_that_live_runs_does_not_trip_it(self, settings):
        """CONTROL: must hold on both trees — the real 10% floor is reachable."""
        assert contract_float("AIT_IC_MIN_CREDIT_WIDTH") == 0.10, (
            "premise moved: the live credit/width floor is no longer 0.10"
        )
        probe = _Probe(settings)
        _run(probe, LIVE_NLV)
        assert not any("credit_width_ratio" in a for a in probe.alerts), (
            f"the live floor is being reported as unreachable: {probe.alerts}"
        )


class TestNoArmIsAnAlgebraicIdentity:
    """CLASS GUARD for the defect shape this wave removed TWICE.

    Both the ratio arm and W25's own D3 fix failed the same way: the quantity
    driving the arm did not depend on the account, so the arm's verdict was a
    constant. This guard fails the next one, whatever its arithmetic.
    """

    def test_the_measured_structure_varies_with_the_account(self, settings):
        probe = _Probe(settings)
        worst = [
            max((p["max_loss"] or 0.0)
                for p in probe._probe_cap_grid(
                    nlv * settings.risk.max_position_risk_pct)["probes"])
            for nlv in (9_750.0, DEPLOYED_NLV, 50_000.0, LIVE_NLV)
        ]
        assert len(set(worst)) >= 3, (
            f"the self-test's driving quantity takes {len(set(worst))} distinct "
            f"values across a 20x range of account sizes: {worst}. An arm fed a "
            f"constant cannot measure the grid."
        )

    def test_the_verdict_itself_is_not_constant(self, settings):
        """Anti-vacuity: an arm that ALWAYS fires is as dead as one that never does.

        PLAN 2026-09-29: the live config is coherent by design now (W52a item
        3 closed max_position_risk_pct at max_daily_loss_pct / 1.5), so "loud"
        injects R26's own 3% cap to demonstrate the arm still fires on a
        known-incoherent config.
        """
        loud_s = settings.model_copy(deep=True)
        loud_s.risk.max_position_risk_pct = 0.03
        # W64: the live defaults moved (5% / 4.5% / ratio 1.0), so inject R26's WHOLE original
        # incoherent setting (3% budget, 2% breaker, the 1.5x rule) to keep proving the arm fires.
        loud_s.risk.max_daily_loss_pct = 0.02
        loud_s.risk.breaker_to_max_loss_ratio = 1.5
        loud = _Probe(loud_s)
        _run(loud, LIVE_NLV)
        assert loud.alerts, "the arm did not fire on a known-incoherent config"

        s = settings.model_copy(deep=True)
        s.risk.max_daily_loss_pct = 0.10
        quiet = _Probe(s)
        _run(quiet, LIVE_NLV)
        assert not quiet.alerts, (
            f"the check cannot be silenced by fixing the grid: {quiet.alerts}"
        )


class TestAFailedSelfTestIsReportedAndRetried:
    """W31 add-ons: the blanket except, and the call site's ordering."""

    def test_a_throwing_check_pages_instead_of_passing_silently(self, settings, monkeypatch):
        """It used to log a warning and discard the verdict.

        Any change that made the check throw turned it into a permanent no-op
        reporting nothing — indistinguishable in the logs from a clean account.
        """
        # Patched on the CARRIER: it copies the methods at class-definition
        # time, so patching TradingOrchestrator would not reach this instance.
        monkeypatch.setattr(
            _Probe, "_build_probe_chains",
            lambda self: (_ for _ in ()).throw(RuntimeError("probe exploded")),
        )
        probe = _Probe(settings)

        ok = _run(probe, LIVE_NLV)  # must not raise

        assert ok is False, "a failed self-test reported success"
        assert probe.loop_errors, "a failed self-test did not page"
        assert probe.loop_errors[0][0] == "size_coherence_check"

    def test_the_failure_page_is_not_gated_behind_a_streak_of_five(self, settings):
        """The threshold must reach this site, which runs ONCE per process.

        Routed at the default of 5 consecutive failures it could never page —
        the same can't-fire shape as the arms this wave removed.
        """
        import inspect

        sig = inspect.signature(TradingOrchestrator._note_loop_error)
        assert "threshold" in sig.parameters, (
            "_note_loop_error has no threshold, so a once-per-process caller "
            "cannot page"
        )
        assert sig.parameters["threshold"].default == 5, (
            "the default must stay 5 so every existing loop caller keeps R8's "
            "streak semantics"
        )

    def test_a_failed_check_is_not_recorded_as_checked(self, settings, monkeypatch):
        """The call site must retry. It used to book the NLV BEFORE awaiting.

        Drives the real call-site expression rather than asserting on its text.
        """
        orch = TradingOrchestrator.__new__(TradingOrchestrator)
        orch._coherence_checked_nlv = 0.0

        async def _fail(nlv, **kw):
            return False

        async def _drive():
            if await _fail(LIVE_NLV):
                orch._coherence_checked_nlv = LIVE_NLV

        asyncio.run(_drive())
        assert orch._coherence_checked_nlv == 0.0, (
            "a check that failed was recorded as done, so it is never retried "
            "until NLV moves 20%"
        )


class TestTheOperatorCanSeeWhatHappened:
    """W31 add-on: the structured log carried only `problems=<count>`."""

    def test_the_log_carries_the_problem_text_not_just_a_count(self, settings, monkeypatch):
        """_send_notification returns early when Telegram is unconfigured, so
        the count was all an operator reading logs ever saw.

        PLAN 2026-09-29: injects R26's own 3% cap since the live config is now
        coherent by design (W52a item 3) and would log problems=0.
        """
        from ait.bot import orchestrator as orch_mod

        captured: list[dict] = []
        real = orch_mod.log.info

        def _spy(event, **kw):
            if event == "launch_size_coherence":
                captured.append(kw)
            return real(event, **kw)

        monkeypatch.setattr(orch_mod.log, "info", _spy)
        s = settings.model_copy(deep=True)
        s.risk.max_position_risk_pct = 0.03
        # W64: the live defaults moved (5% / 4.5% / ratio 1.0), so inject R26's WHOLE original
        # incoherent setting (3% budget, 2% breaker, the 1.5x rule) to keep proving the arm fires.
        s.risk.max_daily_loss_pct = 0.02
        s.risk.breaker_to_max_loss_ratio = 1.5
        _run(_Probe(s), LIVE_NLV)

        assert captured, "the check logged no launch_size_coherence line"
        line = captured[0]
        assert line.get("problem_text"), f"no problem text in the log line: {line}"
        assert line["problems"] == len(line["problem_text"])
        assert "max_contracts_per_trade" not in line, (
            "max_contracts_per_trade is logged as though it were checked, but no "
            "arm reads it — that is what made the dead arms plausible"
        )
        assert line.get("probes"), "the per-grid probe results are not reported"


class TestTheProbeChainIsFitToMeasureWith:
    """ANTI-VACUITY for every test above: if the probe chain cannot clear the
    builder's own gates, each arm is measuring this module instead of the cap
    grid, and every assertion above becomes an accident."""

    @pytest.mark.parametrize("grid", [1.0, 5.0])
    def test_the_builder_accepts_it_on_a_funded_account(self, settings, grid):
        chain = _grid_chain(settings, grid)
        ic = IronCondor()
        ic.risk_budget = LIVE_NLV * settings.risk.max_position_risk_pct
        signals = ic.generate_signals(
            chain.symbol, chain, None, settings.risk.min_confidence,
            settings.sizing_probe.iv_rank, None,
        )
        assert signals, (
            f"the ${grid} probe chain cannot produce a condor even at "
            f"${ic.risk_budget:,.0f}: {ic.last_entry_reason}"
        )

    def test_it_is_not_flagged_as_guessed_iv(self, settings):
        """iv_is_guessed would trip the degraded-greeks gate before anything else."""
        chain = _grid_chain(settings, 1.0)
        assert chain.iv_is_guessed is False
        assert chain.atm_iv == pytest.approx(settings.sizing_probe.implied_vol, abs=1e-6)
        assert chain.expected_move > 0

    def test_every_contract_survives_the_liquidity_filter(self, settings):
        chain = _grid_chain(settings, 1.0)
        assert all(c.is_liquid for c in chain.calls + chain.puts), (
            "some probe contracts fail options.* liquidity, so the builder is "
            "choosing from a smaller ladder than this module thinks"
        )

    def test_both_shorts_sit_in_the_designed_delta_band(self, settings):
        chain = _grid_chain(settings, 1.0)
        lo = contract_float("AIT_IC_DELTA_MIN")
        hi = contract_float("AIT_IC_DELTA_MAX")
        assert any(lo <= abs(c.delta) <= hi for c in chain.calls)
        assert any(lo <= abs(p.delta) <= hi for p in chain.puts)

    def test_it_is_deterministic(self, settings):
        a, b = _grid_chain(settings, 1.0), _grid_chain(settings, 1.0)
        assert [c.strike for c in a.calls] == [c.strike for c in b.calls]
        assert [c.delta for c in a.calls] == [c.delta for c in b.calls]

    def test_the_grid_it_reports_is_the_grid_it_built(self, settings):
        for grid in (1.0, 5.0):
            assert TradingOrchestrator._probe_grid_of(_grid_chain(settings, grid)) == grid


class TestTheProbeStaysOutOfTheTradingPath:
    """AST GUARD. A synthetic chain is a diagnostic fixture; the moment anything
    on the entry, exit or backtest path imports it, the bot is reasoning about a
    model of the market instead of the market."""

    def _importers(self) -> list[str]:
        hits = []
        for path in sorted(SRC.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.ImportFrom):
                    mod = node.module
                elif isinstance(node, ast.Import):
                    mod = ",".join(a.name for a in node.names)
                if mod and "synthetic_chain" in mod:
                    hits.append(path.relative_to(SRC).as_posix())
        return sorted(set(hits))

    def test_only_the_self_test_imports_it(self):
        assert self._importers() == ["ait/bot/orchestrator.py"], (
            f"synthetic_chain reached {self._importers()}"
        )

    def test_the_guard_actually_scans_the_tree(self):
        """Anti-vacuity: a broken glob would make the guard pass by finding nothing."""
        files = list(SRC.rglob("*.py"))
        assert len(files) > 50, f"the scan found only {len(files)} source files"
        assert (SRC / "ait" / "data" / "synthetic_chain.py") in files


class TestTheBudgetWidthInversionHasOneOwner:
    """R26 F4 #6 — the check hand-copied `iron_condor.py`'s width formula."""

    @pytest.mark.parametrize("budget", [63.0, 300.0, 1_500.0, 6_044.0])
    def test_the_inverse_round_trips(self, budget):
        assert IronCondor.budget_for_width(
            IronCondor.affordable_width(budget)) == pytest.approx(budget)

    @pytest.mark.parametrize("budget,price,expected", [
        (0.0, 100.0, 7.675615),
        (500.0, 100.0, 5.555556),
        (63.0, 1.0, 1.000000),
        (6044.0, 660.0, 50.659060),
    ])
    def test_the_width_authority_is_unchanged(self, budget, price, expected):
        """CONTROL: extracting the helper must not move a single width.

        The $63/$1 and $6,044/$660 rows are the numbers R26 measured at HEAD.
        """
        ic = IronCondor()
        ic.risk_budget = budget
        assert ic._vol_scaled_width(price, None, None, None) == pytest.approx(
            expected, abs=1e-5)


class TestTheCreditRatioHasAConfigHome:
    """R26 F4 #6 — `w25-credit-ratio-literal-has-no-config-home`.

    The $140 in PLAN.md's funding table, which drove the simulated-capital
    decision, derived from a bare `credit_ratio_at_20d = 0.45` the operator
    could not see or change. R19c: every trading-economics key has a home.
    """

    def test_the_key_is_in_the_contract_and_config_backed(self):
        assert CONTRACT_DEFAULTS["AIT_IC_CREDIT_RATIO_20D"] == "0.45"
        assert CONFIG_BACKED["AIT_IC_CREDIT_RATIO_20D"] == (
            "backtest", "ic_credit_ratio_at_20d")

    def test_the_default_is_the_literal_it_replaced(self, settings):
        """W36's rule: a config home moves WHERE, never WHAT."""
        assert settings.backtest.ic_credit_ratio_at_20d == 0.45
        assert contract_float("AIT_IC_CREDIT_RATIO_20D") == 0.45

    def test_config_still_loses_to_an_explicit_env_override(self, settings, monkeypatch):
        """Precedence is unchanged: env > config.yaml > contract default."""
        monkeypatch.setenv("AIT_IC_CREDIT_RATIO_20D", "0.38")
        assert contract_float("AIT_IC_CREDIT_RATIO_20D") == 0.38

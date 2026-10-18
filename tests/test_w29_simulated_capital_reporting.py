"""W29 — AIT_SIMULATED_CAPITAL must not re-score the track record (R26 F5 #6).

`AIT_SIMULATED_CAPITAL` exists to shrink the RISK STACK's view of the account so
the live go-live configuration can be exercised on a big paper account. It also
silently shrank the PERFORMANCE DENOMINATOR: `account.py` scales
`NetLiquidation` in place before the snapshot is built, `orchestrator.py`
persists that scaled figure as `bot_state.last_net_liquidation`, and
`capital_base()` resolves that row ahead of its default.

Consequence measured by R26 against the real closed-trade series: the SAME $863
drawdown reads 0.43% against a $201k base and 8.29% against $10k — across this
project's own <8% go-live gate. `reporting/go_live.py` is immune (it divides by
`max_concurrent_car`), so the Friday scorecard and the dashboard would disagree
by ~72x about one book, and a later session reading the dashboard would conclude
the strategy failed the drawdown gate.

Introduced by this session's own 2026-09-17 deploy, which is why it is fixed
first.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ait.broker.account import AccountManager, AccountSnapshot
from ait.bot.orchestrator import TradingOrchestrator

REAL_NLV = 201475.35
SIM = "10000"

BROKER_VALUES = {
    "NetLiquidation": str(REAL_NLV), "BuyingPower": "402950.70",
    "AvailableFunds": "180000", "ExcessLiquidity": "175000",
    "MaintMarginReq": "12000", "InitMarginReq": "14000",
    "UnrealizedPnL": "-3400", "RealizedPnL": "120", "CashBalance": "50000",
}


async def _snapshot(monkeypatch, sim: str | None) -> AccountSnapshot:
    """A snapshot from the REAL AccountManager refresh path."""
    if sim is None:
        # W-capital: "no simulation" is now an explicit "0" -- merely unsetting
        # the var lets config.yaml account.capital switch the simulation on.
        monkeypatch.setenv("AIT_SIMULATED_CAPITAL", "0")
    else:
        monkeypatch.setenv("AIT_SIMULATED_CAPITAL", sim)
    client = MagicMock()
    client.get_account_values = AsyncMock(return_value=dict(BROKER_VALUES))
    return await AccountManager(client).get_snapshot()


class TestTheRiskStackStillSeesTheSmallAccount:
    """Guard against fixing this the wrong way round.

    The scaling is the feature. A 'fix' that un-scaled net_liquidation would
    silently restore full-size position sizing on a probe account — far worse
    than the reporting bug.
    """

    async def test_net_liquidation_is_still_scaled(self, monkeypatch):
        snap = await _snapshot(monkeypatch, SIM)
        assert snap.net_liquidation == pytest.approx(10_000.0)

    async def test_buying_power_is_still_scaled(self, monkeypatch):
        snap = await _snapshot(monkeypatch, SIM)
        assert snap.buying_power < float(BROKER_VALUES["BuyingPower"])


class TestTheReportingDenominatorIsTheRealNlv:

    async def test_negative_control_the_scaled_value_is_what_leaked(self, monkeypatch):
        """NEGATIVE CONTROL: the pre-W29 source of the persisted value.

        Pre-W29 the orchestrator persisted `snapshot.net_liquidation`. Assert
        that value really is the scaled one, so this wave is demonstrably
        fixing something rather than restating a no-op.
        """
        snap = await _snapshot(monkeypatch, SIM)
        assert snap.net_liquidation == pytest.approx(10_000.0)
        assert snap.net_liquidation != pytest.approx(REAL_NLV)

    async def test_the_snapshot_carries_the_unscaled_nlv(self, monkeypatch):
        snap = await _snapshot(monkeypatch, SIM)
        assert snap.real_net_liquidation == pytest.approx(REAL_NLV)

    async def test_the_orchestrator_persists_the_unscaled_nlv(self, monkeypatch):
        """The real decision, through the real method — not a re-implementation.

        R26 F7 flags 'negative controls reimplement the line under test'; this
        drives `_reporting_nlv`, which is the code the persist site calls.
        """
        snap = await _snapshot(monkeypatch, SIM)
        assert TradingOrchestrator._reporting_nlv(snap) == pytest.approx(REAL_NLV)

    async def test_unset_simulated_capital_is_a_no_op(self, monkeypatch):
        """Guards against over-correction: normal operation must not change."""
        snap = await _snapshot(monkeypatch, None)
        assert snap.net_liquidation == pytest.approx(REAL_NLV)
        assert TradingOrchestrator._reporting_nlv(snap) == pytest.approx(REAL_NLV)

    def test_a_pre_w29_snapshot_still_resolves(self):
        """Back-compat: a snapshot built before the field existed must not
        persist 0.0 and blank the denominator."""
        legacy = AccountSnapshot(net_liquidation=REAL_NLV)
        assert legacy.real_net_liquidation == 0.0
        assert TradingOrchestrator._reporting_nlv(legacy) == pytest.approx(REAL_NLV)

    def test_an_empty_snapshot_does_not_raise(self):
        assert TradingOrchestrator._reporting_nlv(AccountSnapshot()) == 0.0


class TestTheGoLiveGateConsequence:
    """Pin the reason this mattered, not just the mechanism.

    The numbers are R26's, recomputed here against the same denominators so a
    later reader can see why a reporting-only bug was treated as urgent.
    """

    @pytest.mark.parametrize(
        "base, expected_pct, crosses_gate",
        [(201_475.0, 0.43, False), (10_000.0, 8.63, True)],
    )
    def test_the_same_drawdown_crosses_the_gate_on_a_scaled_base(
        self, base, expected_pct, crosses_gate
    ):
        max_dd_dollars = 863.0
        pct = max_dd_dollars / base * 100
        assert pct == pytest.approx(expected_pct, abs=0.35)
        assert (pct > 8.0) is crosses_gate

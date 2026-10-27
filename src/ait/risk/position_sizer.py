"""Volatility-adjusted position sizing with Kelly criterion.

Determines how many contracts to trade based on:
- Account size and the per-trade budget (risk.max_position_risk_pct)
- Current volatility of the underlying
- ML model confidence
- Strategy type (spreads risk less than naked options)
"""

from __future__ import annotations

from dataclasses import dataclass

from ait.config.settings import PositionConfig, PositionSizingConfig, RiskConfig
from ait.risk.lot_sizing import lots_for_budget
from ait.utils.logging import get_logger

log = get_logger("risk.sizer")


@dataclass
class PositionSize:
    """Recommended position size with reasoning."""

    contracts: int
    max_risk_dollars: float
    confidence_adjustment: float
    volatility_adjustment: float
    reason: str


class PositionSizer:
    """Calculate position sizes based on risk parameters."""

    # Strategy risk multipliers — how much of the max position to use.
    # W69: the table's home is now config (position_sizing.strategy_risk); this
    # class attribute is the DEFAULT view of it (tests and callers that ask the
    # class, not an instance). An instance sizes from its own config.
    STRATEGY_RISK = dict(PositionSizingConfig.model_fields["strategy_risk"].default)

    # W69: class-level default so an instance built without __init__ (tests
    # use PositionSizer.__new__ + hand-set attributes) still sizes from the
    # model defaults rather than raising AttributeError on a live path.
    _sizing: PositionSizingConfig = PositionSizingConfig()

    def __init__(
        self,
        position_config: PositionConfig,
        risk_config: RiskConfig,
        sizing_config: PositionSizingConfig | None = None,
    ) -> None:
        self._pos_config = position_config
        self._risk_config = risk_config
        # W69: None -> the model defaults (== the pre-W69 literals), so every
        # existing two-argument construction sizes exactly as before.
        self._sizing = sizing_config or PositionSizingConfig()

    @staticmethod
    def _tier_factor(value: float, tiers: list[list[float]]) -> float:
        """1.0 below the first row; otherwise the factor of the highest row whose
        threshold `value` has reached (rows are validated strictly ascending)."""
        factor = 1.0
        for threshold, row_factor in tiers:
            if value >= threshold:
                factor = row_factor
        return factor

    # Strategies that benefit from high IV (selling premium)
    _SELLING_STRATEGIES = frozenset({
        "covered_call", "cash_secured_put", "iron_condor", "short_strangle",
    })

    def calculate(
        self,
        account_value: float,
        option_price: float,
        confidence: float,
        implied_vol: float,
        strategy: str,
        underlying_price: float,
        iv_rank: float = 50.0,
        recent_losing_days: int = 0,
        vix: float = 0.0,
    ) -> PositionSize:
        """Calculate recommended position size.

        Args:
            account_value: Total account value (net liquidation)
            option_price: Price per contract (mid price)
            confidence: ML model confidence (0.0 to 1.0)
            implied_vol: Implied volatility of the option
            strategy: Strategy name (e.g., "bull_call_spread")
            underlying_price: Current price of underlying
            iv_rank: IV rank percentile 0-100 (used for strategy-aware sizing)

        Returns:
            PositionSize with recommended contracts and reasoning
        """
        if account_value <= 0 or option_price <= 0:
            return PositionSize(0, 0, 0, 0, "invalid inputs")

        # R15 minor: a VIX outage passes vix=None (credit strategies fail
        # closed upstream by design, but DEBIT strategies reach this compare
        # and died with TypeError — silently blocking ALL debit entries).
        # None = regime unknown: treat as calm (no VIX shrink) rather than
        # crash; the upstream credit fail-closed gate is unaffected.
        vix = vix or 0.0

        # W108: the per-trade budget is ONE number, risk.max_position_risk_pct, the same one gate 6b
        # enforces and the condor builder is funded with (the sizer's former second cap,
        # positions.max_position_pct, is retired). It is applied inside lots_for_budget below.
        cost_per_contract = option_price * 100  # Options are 100 shares

        # Confidence adjustment: scale down for low confidence
        # At min_confidence, use 50% of max. At 1.0, use 100%.
        min_conf = self._risk_config.min_confidence
        conf_adj = 0.5 + 0.5 * ((confidence - min_conf) / (1.0 - min_conf))
        conf_adj = max(self._sizing.confidence_size_floor, min(1.0, conf_adj))

        # Volatility adjustment: smooth linear scale-down in high-vol environments
        # IV <= iv_full_size_max (20%) -> full size (1.0), IV >= iv_floor_size_min
        # (60%) -> iv_floor_size_factor (0.5). round(.., 12) keeps 0.60-0.20 the
        # exact 0.40 literal this formula was written with (0.6-0.2 is 0.39999..).
        sz = self._sizing
        iv_span = round(sz.iv_floor_size_min - sz.iv_full_size_max, 12)
        if implied_vol <= sz.iv_full_size_max:
            vol_adj = 1.0
        elif implied_vol >= sz.iv_floor_size_min:
            vol_adj = sz.iv_floor_size_factor
        else:
            # Linear interpolation: 1.0 at full_size_max -> factor at floor_min
            vol_adj = 1.0 - (1.0 - sz.iv_floor_size_factor) * (
                (implied_vol - sz.iv_full_size_max) / iv_span)

        # IV rank alignment: boost size when IV conditions favor the strategy
        # Selling strategies benefit from high IV rank (more premium to collect)
        # Buying strategies benefit from low IV rank (cheaper options)
        iv_rank_span = round(sz.iv_rank_adj_max - sz.iv_rank_adj_min, 12)
        iv_rank_adj = 1.0
        if strategy in self._SELLING_STRATEGIES:
            # High IV rank -> up to iv_rank_adj_max; low IV rank -> iv_rank_adj_min
            iv_rank_adj = sz.iv_rank_adj_min + iv_rank_span * (iv_rank / 100.0)
        else:
            # Low IV rank -> up to iv_rank_adj_max; high IV rank -> iv_rank_adj_min
            iv_rank_adj = sz.iv_rank_adj_max - iv_rank_span * (iv_rank / 100.0)
        iv_rank_adj = max(sz.iv_rank_adj_min, min(sz.iv_rank_adj_max, iv_rank_adj))

        # Strategy adjustment
        strategy_adj = sz.strategy_risk.get(strategy, 1.0)

        # Drawdown throttle (defensive during rough patches): the default tiers
        # halve size after 3+ consecutive losing days, quarter it after 5+.
        drawdown_adj = self._tier_factor(recent_losing_days, sz.drawdown_tiers)

        # VIX regime adjustment: shrink positions in high-vol regimes where
        # gamma risk is amplified. Default: VIX>=30 crisis (half), VIX>=25
        # elevated (75%).
        vix_adj = self._tier_factor(vix, sz.vix_size_tiers)

        # W108: the six multipliers are computed (they are reported on PositionSize) but they
        # ACT only for a strategy in position_sizing.scaled_strategies (empty today); every other
        # strategy is sized by the budget alone. Either way the count comes from the ONE shared
        # rule (ait.risk.lot_sizing), the same one the research engine and pool call:
        #   - viability is decided by the UNSCALED budget (Deep-audit SR-M4 + R7-SOON: at small
        #     NLV the soft multipliers shrank the budget below ONE contract and bricked every
        #     defined-risk entry: 0.07 x $2.1k = $147 x 0.4 IC strategy_adj x ... = $17-42 vs a
        #     $130 minimum condor -> 0 contracts, account unusable). A structure one lot of which
        #     exceeds the raw budget is 0 (rejected);
        #   - the multipliers throttle the COUNT above the 1-contract floor, never below it;
        #   - positions.max_contracts_per_trade caps it (keeps cost-per-trade small so the book
        #     can hold many concurrent positions).
        scaler = 1.0
        if strategy in sz.scaled_strategies:
            scaler = (conf_adj * vol_adj * strategy_adj * iv_rank_adj
                      * drawdown_adj * vix_adj)
        contracts = lots_for_budget(
            cost_per_contract, account_value, self._risk_config.max_position_risk_pct,
            max_lots=self._pos_config.max_contracts_per_trade, scaler=scaler)
        if contracts < 1:
            return PositionSize(0, 0.0, conf_adj, vol_adj,
                                "single contract exceeds the per-trade budget "
                                "(risk.max_position_risk_pct)")

        # Max risk: for spreads, risk is the net debit (option_price already is net debit).
        # For naked options, risk is the full premium paid.
        max_risk = contracts * cost_per_contract

        reason = (
            f"account=${account_value:.0f}, "
            f"budget={self._risk_config.max_position_risk_pct:.0%}, "
            f"scaled={'yes' if strategy in sz.scaled_strategies else 'no'}, "
            f"conf_adj={conf_adj:.2f}, vol_adj={vol_adj:.2f}, "
            f"iv_rank_adj={iv_rank_adj:.2f}, strat_adj={strategy_adj:.1f}"
        )

        log.debug(
            "position_sized",
            contracts=contracts,
            max_risk=max_risk,
            confidence=confidence,
            implied_vol=implied_vol,
            strategy=strategy,
        )

        return PositionSize(
            contracts=contracts,
            max_risk_dollars=max_risk,
            confidence_adjustment=conf_adj,
            volatility_adjustment=vol_adj,
            reason=reason,
        )

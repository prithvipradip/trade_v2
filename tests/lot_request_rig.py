"""W111: the scenario rig behind tests/test_lot_request_path.py and the golden table.

A REAL ``RiskManager`` with the REAL ``PositionSizer`` and ``CorrelationGuard`` over the shipped configuration;
only the broker seam is faked (net liquidation, buying power, circuit breaker, PDT guard). ``scenarios()`` is a
fixed table; ``run_table`` runs every row through ``validate_trade`` and records (approved, reason,
position_size, max_risk). ``tests/data/lot_request_golden.json`` was generated from the code as it stood
BEFORE the lot-request path (W111) and pins that nothing changes at a cap of one lot.
"""
from __future__ import annotations

import itertools
from unittest.mock import AsyncMock, MagicMock

from ait.config.settings import load_settings
from ait.risk.correlation import CorrelationGuard
from ait.risk.manager import RiskManager, TradeRequest
from ait.risk.position_sizer import PositionSizer

STRUCTURES = {
    # name: (strategy, direction, entry_price, max_loss)
    "condor_3w": ("iron_condor", "short", 0.90, 210.0),
    "condor_11w": ("iron_condor", "short", 2.40, 860.0),
    "debit_spread": ("bull_call_spread", "long", 1.40, 140.0),
    "no_max_loss": ("iron_condor", "short", 1.20, None),
}
NLVS = (5_000.0, 25_000.0, 200_000.0)
BOOKS = {
    "empty": [],
    "aggregate_near_cap": [{"symbol": "IWM", "strategy": "iron_condor", "max_loss": 700.0, "expiry": "20261120"},
                           {"symbol": "XOM", "strategy": "iron_condor", "max_loss": 250.0, "expiry": "20261127"}],
    "same_symbol": [{"symbol": "QQQ", "strategy": "bull_call_spread", "max_loss": 150.0, "expiry": "20261120"}],
    "cluster_open": [{"symbol": "SPY", "strategy": "iron_condor", "max_loss": 200.0, "expiry": "20261120"}],
    "duplicate": [{"symbol": "QQQ", "strategy": "iron_condor", "max_loss": 200.0, "expiry": "20261120"}],
    "same_symbol_heavy": [{"symbol": "QQQ", "strategy": "long_call", "max_loss": 400.0, "expiry": "20261120"}],
}
VIXES = (16.0, 27.0)
CONFIDENCES = (0.70,)


def shipped_settings(budget: float = 0.05, *, portfolio: float | None = None, symbol: float | None = None):
    s = load_settings("config.yaml").model_copy(deep=True)
    s.risk.max_position_risk_pct = budget
    if portfolio is not None:
        s.positions.max_portfolio_risk_pct = portfolio
    if symbol is not None:
        s.risk.max_symbol_concentration_pct = symbol
    return s


def build_manager(settings, nlv: float, open_positions: list[dict], *, can_afford=None) -> RiskManager:
    account = MagicMock()
    account.get_net_liquidation = AsyncMock(return_value=nlv)
    if can_afford is None:
        account.can_afford = AsyncMock(side_effect=lambda cost: cost <= nlv * 0.9)
    else:
        account.can_afford = AsyncMock(side_effect=can_afford)
    account.get_buying_power = AsyncMock(return_value=nlv)
    cb = MagicMock(is_tripped=False)
    cb.check_daily_loss = MagicMock(return_value=True)
    pdt = MagicMock()
    pdt.can_open_position = MagicMock(return_value=(True, ""))
    rm = RiskManager(
        settings.positions, settings.risk, account, cb, pdt,
        PositionSizer(settings.positions, settings.risk, settings.position_sizing),
        CorrelationGuard(max_correlated_positions=getattr(settings.risk, "max_correlated_positions", 2)),
        state=None,
    )
    rm.update_positions([dict(p) for p in open_positions])
    return rm


def make_request(structure: str, *, vix: float, confidence: float, symbol: str = "QQQ", **extra) -> TradeRequest:
    strategy, direction, entry, max_loss = STRUCTURES[structure]
    return TradeRequest(symbol=symbol, strategy=strategy, direction=direction, contracts=1, entry_price=entry,
                        confidence=confidence, implied_vol=0.22, max_loss=max_loss, vix=vix, iv_rank=55.0, **extra)


def scenarios():
    """(key, variant, structure, nlv, book, vix, confidence). The ``shipped`` variant is the full grid; ``wide_caps``
    (10% per trade, a 50% aggregate cap and a 10% per-symbol cap) reaches gate 6c, which the shipped caps (all 20%)
    never let trip; ``tight_bp`` makes buying power the binding gate."""
    for structure, nlv, book, vix, conf in itertools.product(STRUCTURES, NLVS, BOOKS, VIXES, CONFIDENCES):
        yield f"shipped|{structure}|{int(nlv)}|{book}|vix{int(vix)}|conf{conf}", "shipped", structure, nlv, book, vix, conf
    for structure, nlv, book in itertools.product(("condor_3w", "debit_spread"), (5_000.0, 25_000.0),
                                                  ("empty", "same_symbol", "same_symbol_heavy", "aggregate_near_cap")):
        yield f"wide_caps|{structure}|{int(nlv)}|{book}", "wide_caps", structure, nlv, book, 16.0, 0.70
    yield "shipped|condor_3w|5000|empty|vix16|conf0.45", "shipped", "condor_3w", 5_000.0, "empty", 16.0, 0.45
    for structure, nlv in itertools.product(("condor_3w", "debit_spread"), (5_000.0, 25_000.0)):
        yield f"tight_bp|{structure}|{int(nlv)}", "tight_bp", structure, nlv, "empty", 16.0, 0.70


def settings_for(variant: str):
    if variant == "wide_caps":
        return shipped_settings(0.10, portfolio=0.50, symbol=0.10)
    return shipped_settings()


async def run_table(settings_for=settings_for, **request_extra) -> dict[str, dict]:
    """Every scenario through the real ``validate_trade``; ``request_extra`` is passed to ``TradeRequest``."""
    out: dict[str, dict] = {}
    for key, variant, structure, nlv, book, vix, conf in scenarios():
        afford = (lambda cost: cost <= 150.0) if variant == "tight_bp" else None
        rm = build_manager(settings_for(variant), nlv, BOOKS[book], can_afford=afford)
        v = await rm.validate_trade(make_request(structure, vix=vix, confidence=conf, **request_extra))
        out[key] = {"approved": bool(v.approved), "reason": v.reason, "position_size": int(v.position_size),
                    "max_risk": round(float(v.max_risk), 2)}
    return out

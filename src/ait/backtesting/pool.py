"""W-capital: shared-capital allocator for walk-forward.

Walk-forward simulates one symbol at a time (the engine is single-symbol), and
BT-H3 gave every symbol its OWN full-capital sleeve. That made three symbols
trade as if the account held 3x its money -- and made the configured capital
(`account.capital`) meaningless as a total. Live has ONE account: every trade is
sized off the account's current equity and the symbols compete for it.

`allocate_pool` replays the per-symbol candidate trades chronologically against
ONE pool, applying the same three gates live applies at entry:

  * per-trade budget   contracts = floor(equity * size_pct / max_loss
                       per contract), never more than the engine itself chose.
                       0 contracts -> rejected "unaffordable" (gate 6b parity).
  * aggregate risk     open max-loss + this trade <= equity * max_portfolio_risk_pct
                       (positions.max_portfolio_risk_pct) -> "portfolio_risk".
  * open-position cap  positions.max_open_positions -> "max_open_positions".

A trade may carry a private `_pool_size_pct` (the effective fraction its engine
run used, learner multipliers included); it is honoured, then stripped.
Pure and IO-free so tests drive the real decision. Equity is POOL + REALISED
P&L of accepted, closed trades (unrealised marks are not modelled, like the
engine's own capital).

Ordering contract: events sort by (date, entries-before-exits, input order).
A same-day exit therefore frees capital only AFTER that day's entries -- the
conservative reading when exit timestamps are date-only. Dollar fields scale
linearly in contracts in the engine (`_calc_pnl`), so a resized trade scales
pnl / credit / max_loss / entry_commission by the contract ratio.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

# Total-dollar trade fields that are linear in the contract count.
_LINEAR_FIELDS = ("pnl", "credit", "max_loss", "entry_commission")


@dataclass
class PoolResult:
    accepted: list[dict] = field(default_factory=list)
    rejections: Counter = field(default_factory=Counter)
    final_equity: float = 0.0
    peak_open_risk: float = 0.0

    @property
    def rejected_total(self) -> int:
        return sum(self.rejections.values())


def _day(value) -> str:
    return str(value or "")[:10]


def _per_contract_risk(trade: dict) -> float | None:
    """Max loss of ONE contract in dollars, or None when the trade carries none."""
    contracts = int(trade.get("contracts") or 0)
    per_share = trade.get("max_loss_per_share")
    if per_share is not None:
        try:
            v = float(per_share) * 100.0
            if v > 0 and not math.isnan(v):
                return v
        except (TypeError, ValueError):
            pass
    total = trade.get("max_loss")
    if total is not None and contracts > 0:
        try:
            v = float(total) / contracts
            if v > 0 and not math.isnan(v):
                return v
        except (TypeError, ValueError):
            pass
    return None


def allocate_pool(
    candidate_trades: list[dict],
    pool_capital: float,
    position_size_pct: float,
    max_portfolio_risk_pct: float,
    max_open_positions: int,
) -> PoolResult:
    """Replay `candidate_trades` (all symbols) against one shared pool."""
    result = PoolResult(final_equity=float(pool_capital))
    # (day, phase, order, trade_index): phase 0 = entry, 1 = exit.
    events: list[tuple[str, int, int, int]] = []
    for i, t in enumerate(candidate_trades):
        events.append((_day(t.get("entry_date") or t.get("entry_time")), 0, i, i))
        events.append((_day(t.get("exit_date") or t.get("exit_time")
                            or t.get("entry_date")), 1, i, i))
    events.sort()

    equity = float(pool_capital)
    open_risk = 0.0
    open_book: dict[int, tuple[float, dict]] = {}  # idx -> (risk, resized trade)

    for _day_key, phase, _order, idx in events:
        if phase == 0:
            t = candidate_trades[idx]
            engine_contracts = int(t.get("contracts") or 0)
            per_contract = _per_contract_risk(t)
            if engine_contracts < 1:
                result.rejections["no_contracts"] += 1
                continue
            if len(open_book) >= int(max_open_positions):
                result.rejections["max_open_positions"] += 1
                continue
            if per_contract is None:
                # Cannot size or risk-count it: pass through untouched (the
                # live selector is IC-only, which always carries max loss).
                open_book[idx] = (
                    0.0, {k: v for k, v in t.items() if k != "_pool_size_pct"})
                continue
            pct = float(t.get("_pool_size_pct") or position_size_pct)
            afford = int(math.floor(equity * pct / per_contract))
            contracts = min(engine_contracts, afford)
            if contracts < 1:
                result.rejections["unaffordable"] += 1
                continue
            risk = per_contract * contracts
            if open_risk + risk > equity * max_portfolio_risk_pct:
                result.rejections["portfolio_risk"] += 1
                continue
            resized = {k: v for k, v in t.items() if k != "_pool_size_pct"}
            if contracts != engine_contracts:
                ratio = contracts / engine_contracts
                for key in _LINEAR_FIELDS:
                    if resized.get(key) is not None:
                        resized[key] = round(float(resized[key]) * ratio, 2)
                resized["contracts"] = contracts
                resized["pool_contracts_engine"] = engine_contracts
            open_book[idx] = (risk, resized)
            open_risk += risk
            result.peak_open_risk = max(result.peak_open_risk, open_risk)
        else:
            booked = open_book.pop(idx, None)
            if booked is None:
                continue  # was rejected at entry
            risk, resized = booked
            open_risk -= risk
            equity += float(resized.get("pnl") or 0.0)
            result.accepted.append(resized)

    result.final_equity = equity
    result.accepted.sort(
        key=lambda t: str(t.get("entry_date") or t.get("entry_time") or ""))
    return result

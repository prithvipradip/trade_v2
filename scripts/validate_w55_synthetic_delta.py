#!/usr/bin/env python3
"""W55 follow-up: validate the synthetic per-leg delta against real live prices.

    python scripts/validate_w55_synthetic_delta.py

Why this exists: W55's own scope ("adapt options_chain.py's py_vollib pattern
to _get_option_marks()'s live price... validate the solve against real
monitor-loop prices before anything depends on it") explicitly requires this
before the new code (`execution/synthetic_delta.py`, `data/options_chain.py`'s
`solve_iv_from_price`/`compute_greeks_from_sigma`) is trusted — offline
tests prove the MATH against known-sigma synthetic prices, but not that it
behaves sanely against IBKR's actual live tick stream.

What this does, and nothing else:
  1. Connects to TWS paper on its own clientId (98 — outside the bot's own
     ladder 1/101-105, and the W53 probe's 97).
  2. Fetches a real live options chain for SPY (via the SAME audited
     OptionsChainService/`_calculate_greeks` path) — this gives a
     cross-check IV/delta for a handful of near-ATM contracts.
  3. For each of those same contracts, simulates what the MONITOR LOOP would
     see (a single `marketPrice`, no bid/ask — the mid of the live
     bid/ask just fetched, standing in for what `_get_option_marks()` would
     report) and runs it through `synthetic_leg_greeks` — the NEW path.
  4. Reports both side by side: does the synthetic path's solved IV/delta
     land close to the chain path's own (they're not expected to be
     identical — the chain path solves from a two-sided mid, the synthetic
     path from a single mid-substitute — but they should be close, and
     neither should be nonsense).
  5. If any REAL open positions exist on the account by the time this runs,
     ALSO pulls `ib.portfolio()` marks directly (genuine `_get_option_marks()`
     shape) and runs synthetic_leg_greeks against those too — the actual
     production data path, not a stand-in.

Places NO orders, calls NO whatIfOrder, requests NO Greeks subscription.
Read-only against the broker; makes no changes to data/ait_state.db.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ait.config.runtime_env import apply_runtime_env_defaults  # noqa: E402

apply_runtime_env_defaults()

import asyncio  # noqa: E402

from ait.broker.ibkr_client import IBKRClient  # noqa: E402
from ait.config.settings import IBKREnvConfig, load_settings  # noqa: E402
from ait.data.market_data import MarketDataService  # noqa: E402
from ait.data.options_chain import OptionsChainService  # noqa: E402
from ait.execution.synthetic_delta import synthetic_leg_greeks  # noqa: E402
from ait.utils.logging import get_logger  # noqa: E402

log = get_logger("scripts.validate_w55")

VALIDATE_CLIENT_ID = 98
SYMBOL = "SPY"
VALIDATE_MARKET_DATA_TYPE = 4  # delayed-frozen; see the W53 probe's own
                               # finding about a cold session getting no
                               # quotes at type 3 outside RTH


class ValidationError(RuntimeError):
    pass


async def connect() -> IBKRClient:
    cfg = IBKREnvConfig(ibkr_client_id=VALIDATE_CLIENT_ID)
    ibkr = IBKRClient(cfg)
    if not await ibkr.connect():
        raise ValidationError("IBKRClient.connect() failed or refused")
    acct = ibkr.session_account or ""
    if not acct.upper().startswith("D"):
        raise ValidationError(f"session account {acct!r} is not paper — refusing")
    ibkr.ib.reqMarketDataType(VALIDATE_MARKET_DATA_TYPE)
    return ibkr


async def main() -> int:
    ibkr = await connect()
    try:
        settings = load_settings(str(ROOT / "config.yaml"))
        market_data = MarketDataService(ibkr, polygon_api_key=settings.api_keys.polygon_api_key)
        chain_service = OptionsChainService(ibkr, market_data, settings.options)

        chains = await chain_service.get_chain(SYMBOL, min_dte=14, max_dte=30)
        if not chains:
            print("NO CHAIN RETURNED — cannot validate (see the W53 probe's "
                  "cold-session finding if this is outside RTH).")
            return 1
        chain = chains[0]
        liquid = chain.filter_liquid()
        if chain.iv_is_guessed:
            print(f"WARNING: chain.iv_is_guessed=True for {SYMBOL} {chain.expiry} — "
                  "the chain path itself has no usable IV right now; a "
                  "cross-check against it would be comparing two guesses.")

        spot = chain.underlying_price
        candidates = (liquid.puts + liquid.calls)[:6] or (chain.puts + chain.calls)[:6]
        if not candidates:
            print("NO CONTRACTS TO VALIDATE AGAINST.")
            return 1

        rows = []
        for c in candidates:
            chain_iv, chain_delta = c.implied_vol, c.delta
            monitor_mark = c.mid if c.has_two_sided_quote else c.last
            if not monitor_mark or monitor_mark <= 0:
                continue
            synthetic = synthetic_leg_greeks(
                {(SYMBOL, str(c.expiry).replace("-", "")[:8], float(c.strike),
                  c.right): monitor_mark},
                SYMBOL, str(c.expiry), c.strike, c.right, spot)
            rows.append({
                "strike": c.strike, "right": c.right, "mark": monitor_mark,
                "chain_iv": round(chain_iv, 4), "chain_delta": round(chain_delta, 4),
                "synthetic_iv": round(synthetic.iv, 4) if synthetic else None,
                "synthetic_delta": round(synthetic.delta, 4) if synthetic else None,
                "delta_diff": (round(abs(synthetic.delta - chain_delta), 4)
                               if synthetic else None),
            })

        real_open_positions = [
            p for p in ibkr.ib.positions()
            if p.contract.symbol == SYMBOL and getattr(p.contract, "secType", "") == "OPT"
            and p.position != 0
        ]
        portfolio_rows = []
        if real_open_positions:
            marks = {}
            for item in ibkr.ib.portfolio():
                c = item.contract
                if getattr(c, "secType", "") != "OPT":
                    continue
                price = item.marketPrice
                if price is None or (isinstance(price, float) and math.isnan(price)):
                    continue
                key = (c.symbol, c.lastTradeDateOrContractMonth.replace("-", "")[:8],
                       float(c.strike), c.right)
                marks[key] = float(price)
            for p in real_open_positions:
                c = p.contract
                expiry_iso = (f"{c.lastTradeDateOrContractMonth[:4]}-"
                              f"{c.lastTradeDateOrContractMonth[4:6]}-"
                              f"{c.lastTradeDateOrContractMonth[6:8]}")
                synthetic = synthetic_leg_greeks(marks, c.symbol, expiry_iso,
                                                 c.strike, c.right, spot)
                portfolio_rows.append({
                    "strike": c.strike, "right": c.right,
                    "synthetic_iv": round(synthetic.iv, 4) if synthetic else None,
                    "synthetic_delta": round(synthetic.delta, 4) if synthetic else None,
                    "source": "real ib.portfolio() marks — the actual production path",
                })

        sane = all(
            r["synthetic_delta"] is not None
            and (-1.0 <= r["synthetic_delta"] <= 1.0)
            and r["synthetic_iv"] is not None and r["synthetic_iv"] > 0
            for r in rows
        )
        close_enough = all(
            r["delta_diff"] is None or r["delta_diff"] < 0.10 for r in rows
        )

        result = {
            "symbol": SYMBOL, "spot": spot, "expiry": str(chain.expiry),
            "chain_iv_is_guessed": chain.iv_is_guessed,
            "chain_vs_synthetic": rows,
            "real_open_position_check": portfolio_rows,
            "verdict": {
                "all_synthetic_outputs_in_range": sane,
                "chain_vs_synthetic_delta_within_0_10": close_enough,
            },
        }
        print(json.dumps(result, indent=2, default=str))
        return 0 if (sane and close_enough) else 1
    finally:
        await ibkr.disconnect()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

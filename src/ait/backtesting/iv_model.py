"""W87a: the research engine's implied-vol PROXY constants, defined once.

When no stored or VIX-derived IV exists the engine estimates IV from realized vol, and it re-marks an open
position's IV as a blend of the entry IV and that estimate. The constants (x1.15, 70% / 30%, the 21-close
guard) were copied between ``Backtester`` (``_get_iv``, ``_get_current_iv``) and the parity replay
(``replay.engine_current_iv``), with no test that the two stayed equal. Both now call the functions below;
``tests/test_w87a_hygiene.py`` runs the engine method and the replay mirror on a grid and asserts equality.

The per-symbol VIX multipliers (R16 calibration) and the legacy %-spread reference points live here too.
They are CALIBRATED MODEL CONSTANTS, not tunables: changing one changes what a research number means, so
it is a code change with a PLAN.md entry, not a config edit. Pure functions, no IO.
"""
from __future__ import annotations

import numpy as np

from ait.backtesting.pricing import realized_vol

# Synthetic fallback: IV ~ 1.15 x 20-day realized vol (implied vol runs above realized on average).
SYNTHETIC_IV_PREMIUM: float = 1.15
RV_WINDOW: int = 20

# Re-mark of an open position: 70% entry IV + 30% current (premium-adjusted) realized vol; with fewer than
# REMARK_MIN_CLOSES closes there is not enough history and the entry IV is kept.
REMARK_ENTRY_WEIGHT: float = 0.70
REMARK_RV_WEIGHT: float = 0.30
REMARK_MIN_CLOSES: int = RV_WINDOW + 1

# R16: VIX -> per-symbol implied-vol multipliers. Unknown symbols keep the historical 1.10 so nothing
# silently changes for underlyings this calibration was never measured on.
VOL_MULT_VS_VIX: dict[str, float] = {"SPY": 1.00, "QQQ": 1.228, "IWM": 1.33, "DIA": 0.98}
VOL_MULT_DEFAULT: float = 1.10

# The legacy %-spread model's reference point: spread widens above this IV and below this DTE.
HALF_SPREAD_REF_IV: float = 0.20
HALF_SPREAD_REF_DTE: int = 21


def synthetic_iv(closes) -> float:
    """Priority-3 IV estimate: ``realized_vol(closes, 20) * 1.15`` (no floor; the caller floors)."""
    return realized_vol(np.asarray(closes), window=RV_WINDOW) * SYNTHETIC_IV_PREMIUM


def remark_iv(entry_iv: float, closes, iv_floor: float) -> float:
    """The re-mark IV: the entry IV when fewer than 21 closes exist, else
    ``max(0.70 * entry_iv + 0.30 * synthetic_iv, iv_floor)``."""
    if len(closes) < REMARK_MIN_CLOSES:
        return entry_iv
    rv = realized_vol(np.asarray(closes), window=RV_WINDOW) * SYNTHETIC_IV_PREMIUM
    return max(REMARK_ENTRY_WEIGHT * entry_iv + REMARK_RV_WEIGHT * rv, iv_floor)

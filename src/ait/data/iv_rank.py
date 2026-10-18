"""W68: the ONE definition of IV rank, shared by the live orchestrator and the research engine.

Live gates credit entries on it twice -- `iron_condor.py` refuses `iv_rank < AIT_IRON_CONDOR_IV_FLOOR`
(15) and the risk manager refuses `iv_rank > AIT_IV_RANK_CAP` (85) -- but the research engine only
had an absolute-level floor (`iv < 0.20`), so a day could pass one and fail the other (the W67
CLARIFICATION: a parity gap, not a units mix-up). This module is the definition both sides now call.

Definition (the orchestrator's R7 "true percentile", unchanged): over the last `window` (252)
stored daily implied-vol observations INCLUDING today's, rank = 100 x (number strictly below
today's IV) / (number of observations); None when fewer than `min_obs` (60) exist (unknown is
never a fabricated 50: callers fail closed, R22 fail-direction-05).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

IV_RANK_WINDOW = 252
IV_RANK_MIN_OBS = 60


def iv_rank_percentile(values_newest_first: Sequence[float], window: int = IV_RANK_WINDOW,
                       min_obs: int = IV_RANK_MIN_OBS) -> float | None:
    """IV rank (0-100) of the first element against the first `window` elements, else None.

    Input is the stored IV series NEWEST FIRST (the live query is `ORDER BY date DESC LIMIT 252`);
    non-positive / NaN observations are the caller's to drop, as the live query does.
    """
    rows = list(values_newest_first)[:window]
    if len(rows) < min_obs:
        return None
    cur = rows[0]
    below = sum(1 for v in rows if v < cur)
    return max(0.0, min(100.0, below / len(rows) * 100))


def iv_rank_series(iv: pd.Series, window: int = IV_RANK_WINDOW,
                   min_obs: int = IV_RANK_MIN_OBS) -> pd.Series:
    """`iv_rank_percentile` for every date of a stored daily IV series (oldest first).

    Row t uses only observations at or before t (no look-ahead). Non-positive / missing IVs are
    dropped first, exactly as the live query's `implied_vol IS NOT NULL AND implied_vol > 0`;
    a date whose own IV is missing gets NaN (live would have no fresh IV that day).
    """
    valid = iv.where(iv > 0).dropna()
    vals = valid.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        w = vals[lo:i + 1]
        if len(w) >= min_obs:
            out[i] = max(0.0, min(100.0, float((w < vals[i]).sum()) / len(w) * 100))
    return pd.Series(out, index=valid.index).reindex(iv.index)

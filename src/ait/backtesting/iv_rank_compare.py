"""W71: pure comparisons between the research engine's IV-rank veto and live's IV-rank gates.

Two different quantities carry the name "IV rank" in this repo, and only one of them is what
the live bot gates on:

* **live rank** (``ait.data.iv_rank``): the percentile (0-100) of today's STORED implied vol
  among the last 252 stored daily IVs. Live refuses a condor below ``floor`` (15) and any
  credit above ``cap`` (85); an unmeasurable rank fails closed.
* **feature rank** (``FeatureEngine`` ``iv_rank``): a REALIZED-vol range rank in 0-1,
  ``(vol_20 - min252) / (max252 - min252)``. Only the research engine's
  ``iv_rank_rise_threshold`` veto reads it (it blocks an entry when the feature rank ROSE by
  more than the threshold over the last 10 rows). Live has no rise veto.

These functions answer, on any pair of aligned daily series: how often does each fire, how often
do they agree, and how many of the research veto's days would live have refused anyway. No
pandas-heavy engine imports: tests drive them on synthetic series with known answers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# W90c: the three numbers this module used to hold as constants are single-sourced (config/imports are done lazily so the
# module stays import-light): the veto's row count is ``backtest.engine_tuning.iv_rank_rise_lookback_rows - 1`` (the
# engine's ``features_df["iv_rank"].iloc[-rows:]`` -> last minus first), and live's floor / cap are the env contract
# AIT_IRON_CONDOR_IV_FLOOR / AIT_IV_RANK_CAP (config ``backtest.ic_iv_rank_floor`` / ``risk.iv_rank_cap``).


def rise_rows() -> int:
    """Rows the engine's rise veto spans (its look-back window less one)."""
    from ait.config.settings import loaded_engine_tuning
    return int(loaded_engine_tuning().iv_rank_rise_lookback_rows) - 1


def live_floor() -> float:
    """Live refuses a condor below this IV rank (percentile, 0-100)."""
    from ait.config.runtime_env import contract_float
    return float(contract_float("AIT_IRON_CONDOR_IV_FLOOR"))


def live_cap() -> float:
    """Live refuses any credit above this IV rank (percentile, 0-100)."""
    from ait.config.runtime_env import contract_float
    return float(contract_float("AIT_IV_RANK_CAP"))


def rise(series: pd.Series, n: int | None = None) -> pd.Series:
    """Change over ``n`` rows (the engine's ``iloc[-1] - iloc[-(n+1)]``), NaN until defined. ``n`` defaults to
    ``rise_rows()``."""
    return series - series.shift(rise_rows() if n is None else n)


def veto_days(series: pd.Series, threshold: float, n: int | None = None) -> pd.Series:
    """Boolean per day: the rise exceeds ``threshold`` (False where the rise is undefined)."""
    r = rise(series, n)
    return (r > threshold).fillna(False)


def live_level_blocks(rank_0_100: pd.Series, floor: float | None = None,
                      cap: float | None = None) -> pd.Series:
    """Boolean per day: live would refuse the entry on the rank LEVEL (below the condor floor
    or above the credit cap). An unknown rank (NaN) fails closed, as live does, so it is
    reported as blocked."""
    floor, cap = live_floor() if floor is None else floor, live_cap() if cap is None else cap
    r = rank_0_100.astype(float)
    return ((r < floor) | (r > cap) | r.isna())


def side(rank_0_100: pd.Series, floor: float | None = None, cap: float | None = None) -> pd.Series:
    """'below' | 'in' | 'above' relative to the live floor / cap (NaN stays NaN)."""
    floor, cap = live_floor() if floor is None else floor, live_cap() if cap is None else cap
    r = rank_0_100.astype(float)
    out = pd.Series(np.where(r < floor, "below", np.where(r > cap, "above", "in")), index=r.index)
    return out.where(r.notna())


def spearman(a: pd.Series, b: pd.Series) -> float | None:
    """Rank correlation on the dates both are defined (None below 10 pairs)."""
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(df) < 10:
        return None
    return float(df["a"].corr(df["b"], method="spearman"))


def agreement(a: pd.Series, b: pd.Series) -> dict[str, int]:
    """Counts of the two boolean day-masks over the dates both are defined."""
    df = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna().astype(bool)
    return {
        "n": int(len(df)),
        "both": int((df["a"] & df["b"]).sum()),
        "only_a": int((df["a"] & ~df["b"]).sum()),
        "only_b": int((~df["a"] & df["b"]).sum()),
        "neither": int((~df["a"] & ~df["b"]).sum()),
    }


def veto_vs_live(feature_rank_0_1: pd.Series, live_rank_0_100: pd.Series,
                 threshold: float | None = None, floor: float | None = None, cap: float | None = None) -> dict:
    """The W71 headline numbers on the dates where BOTH ranks are measurable.

    ``veto``: days the research rise veto fires. Of those, ``already_blocked_by_live`` are days
    live's level gates refuse anyway; ``extra`` are days live WOULD trade, i.e. the restriction
    that exists only in research.

    ``threshold`` defaults to ``backtest.iv_rank_rise_threshold`` (was a 0.30 literal); ``floor`` / ``cap`` to live's."""
    if threshold is None:
        from ait.config.settings import load_settings
        threshold = float(load_settings().backtest.iv_rank_rise_threshold)
    floor, cap = live_floor() if floor is None else floor, live_cap() if cap is None else cap
    df = pd.concat([feature_rank_0_1, live_rank_0_100], axis=1, keys=["f", "s"]).dropna()
    f_veto = veto_days(feature_rank_0_1, threshold).reindex(df.index).fillna(False)
    s_veto = veto_days(live_rank_0_100 / 100.0, threshold).reindex(df.index).fillna(False)
    live_block = live_level_blocks(df["s"], floor, cap)
    n = int(len(df))
    veto = int(f_veto.sum())
    already = int((f_veto & live_block).sum())
    return {
        "n_days": n,
        "live_level_block_days": int(live_block.sum()),
        "research_veto_days": veto,
        "research_veto_share": (veto / n) if n else float("nan"),
        "already_blocked_by_live": already,
        "extra_restriction_days": veto - already,
        "extra_share_of_tradeable_days": (
            (veto - already) / int((~live_block).sum()) if int((~live_block).sum()) else float("nan")),
        "rise_on_stored_iv_days": int(s_veto.sum()),
        "veto_agreement": agreement(f_veto, s_veto),
        "side_disagreement_days": int((side(df["f"] * 100.0, floor, cap) != side(df["s"], floor, cap)).sum()),
        "spearman_feature_vs_stored": spearman(df["f"] * 100.0, df["s"]),
    }

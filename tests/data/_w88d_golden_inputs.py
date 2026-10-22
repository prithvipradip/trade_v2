"""Inputs and capture functions shared by the W88d golden-behaviour test and the script that recorded the golden file.

The golden file (``w88d_feature_golden.npz`` + ``w88d_feature_golden.json``) was recorded with the code AS IT WAS before
W88d, so a test that recomputes everything here and compares proves the feature engine, the feature names and the
meta-label context keys are unchanged at the default windows. All inputs are seeded: identical on every run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAIL = 120          # rows of each feature frame kept in the golden file (enough to cover every rolling window)


def _frame(days: int, seed: int, drift: float, vol: float, zero_volume: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=days, freq="B")
    close = 400.0 * np.cumprod(1 + rng.normal(drift, vol, days))
    open_ = close * (1 + rng.normal(0, 0.002, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, days)))
    volume = np.zeros(days) if zero_volume else rng.integers(2_000_000, 9_000_000, days).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def frames() -> dict[str, pd.DataFrame]:
    return {
        "calm_700": _frame(700, 1, 0.0004, 0.011),
        "wild_320": _frame(320, 2, -0.0002, 0.024),
        "zero_volume_450": _frame(450, 3, 0.0003, 0.015, zero_volume=True),
    }


def market_context() -> dict[str, pd.DataFrame]:
    base = _frame(700, 1, 0.0004, 0.011)
    vix = _frame(700, 11, 0.0, 0.04)
    vix[["Open", "High", "Low", "Close"]] = 18.0 + (vix[["Open", "High", "Low", "Close"]] - 400.0) / 40.0
    spy = _frame(700, 12, 0.0004, 0.010)
    return {"vix": vix, "spy": spy, "_base": base}


def compute_all(engine=None) -> dict[str, pd.DataFrame]:
    """name -> FeatureEngine().compute(...) for every input (with and without cross-asset context)."""
    from ait.ml.features import FeatureEngine
    fe = engine if engine is not None else FeatureEngine()
    out = {name: fe.compute(df) for name, df in frames().items()}
    ctx = market_context()
    out["calm_700_with_context"] = fe.compute(ctx["_base"], market_context={"vix": ctx["vix"], "spy": ctx["spy"]})
    return out


def name_lists(engine=None) -> dict[str, list[str]]:
    from ait.ml.features import FeatureEngine
    fe = engine if engine is not None else FeatureEngine()
    return {"base": list(fe.get_feature_names()), "with_vlmc": list(fe.get_feature_names(include_vlmc=True))}


def fixed_trades(features: pd.DataFrame) -> list[dict]:
    """Closed-trade dicts at fixed dates inside the feature frame, for the meta-label training-data builder."""
    dates = [d for d in features.index[200::37]][:8]
    regimes = ["range_bound", "trending_up", "trending_down", "high_volatility"]
    return [{"entry_date": str(d.date()), "pnl": (60.0 if i % 3 else -90.0), "entry_confidence": 0.55 + 0.03 * i,
             "entry_regime": regimes[i % 4], "entry_iv_rank": 0.2 + 0.05 * i, "entry_vix_level": 0.4 + 0.02 * i}
            for i, d in enumerate(dates)]


def meta_training_frame(features: pd.DataFrame) -> pd.DataFrame:
    from ait.ml.meta_label import MetaLabeler
    return MetaLabeler().build_training_data_from_backtest(fixed_trades(features), features)


def engine_meta_contexts(max_contexts: int = 6) -> dict:
    """Run the (legacy) research engine over a seeded frame with a recording meta-labeler and return the contexts it
    builds plus the ``features_at_entry`` snapshot keys of the trades it opens."""
    from ait.backtesting.engine import Backtester

    seen: list[dict] = []

    class Recorder:
        is_trained = True

        def predict(self, ctx):
            seen.append(dict(ctx))
            return None

    from ait.ml.features import FeatureEngine
    data = _frame(700, 5, 0.0004, 0.012)
    cache = FeatureEngine().compute(data)                 # as the walk-forward hands it to the engine
    bt = Backtester(data=data, strategies=["iron_condor"], meta_labeler=Recorder(), allow_live_model_fallback=False,
                    features_cache=cache, condor_any_direction=True, iv_floor=0.05, ic_min_credit=0.0, ic_min_credit_width=0.0,
                    min_confidence=0.0, initial_capital=1_000_000.0, hurst_hard_veto_multiplier=0.0,
                    hurst_regime_threshold=0.0, regime_veto_gate=False, iv_rank_rise_gate=False)
    res = bt.run()
    snaps = [t.get("features_at_entry") for t in res.trades if t.get("features_at_entry")]
    snap = snaps[-1] if snaps else {}
    return {"contexts": [{k: (round(float(v), 8) if isinstance(v, (int, float)) else v) for k, v in c.items()}
                         for c in seen[-max_contexts:]],
            "n_contexts": len(seen),
            "snapshot": snap}

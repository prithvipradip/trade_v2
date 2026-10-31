"""W116: the inputs of the STORE-BACKED walk-forward golden record (``tests/data/w116_store_walkforward_golden.json``).

``_w108s1_golden_inputs`` hands the walk-forward its frames directly, so it never touches a loader. This one seeds a
temporary history database (IB-tagged daily bars, stored IV, a VIX series in ``daily_iv``) and lets the REAL
``WalkForwardBacktester.run`` load everything through ``load_daily_ohlcv`` on the research path, with a pinned
``history_start_date`` and the clock FROZEN on the day after the last stored session (so the store is "current",
exactly the shape of a default study run the morning after a refresh). Complete data in, so a strict loader must
give a result BIT-IDENTICAL to the one recorded here: that is the proof that making the loaders strict changes
nothing for a study whose inputs exist.

Why a frozen clock. The loaders measure their windows from today (``today_et``) and ``load_daily_iv`` from
``datetime.now``; a fixture dated in 2022-24 is "old" on any real day and the legacy code then loses its IV and
falls to Yahoo for the VIX. That is the pinned-past-end defect W116 fixes (see ``test_w116_goldens`` for the
dedicated test); this record is about the case that already works.

The record was captured at commit ca0439a (before any W116 source change) by running this file as a script::

    python tests/data/_w116_store_golden_inputs.py

Only the earnings lookup (a yfinance request; ETFs have none) is stubbed. If a later wave moves a research default
this record moves with it: regenerate it and say so in PLAN.md.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tests.data._w108s1_golden_inputs import _no_earnings, snapshot

GOLDEN_PATH = Path(__file__).with_name("w116_store_walkforward_golden.json")
SYMBOLS = ["SPY", "QQQ"]
CAPITAL = 5000.0
START = date(2022, 1, 3)
SESSIONS = 900
VIX_SESSIONS = 1500        # the VIX context is asked for a longer span than the prices (train + 900 days)


def make_frames() -> dict[str, pd.DataFrame]:
    """Seeded daily bars with a stored IV column (a smooth seeded path around 0.2)."""
    out = {}
    idx = pd.bdate_range(START, periods=SESSIONS)
    for k, (sym, start) in enumerate((("SPY", 450.0), ("QQQ", 380.0))):
        rng = np.random.default_rng(116 + k)
        close = start * np.cumprod(1 + rng.normal(0.0002, 0.012, len(idx)))
        iv = np.clip(0.20 + np.cumsum(rng.normal(0, 0.004, len(idx))) * 0.2, 0.10, 0.45)
        out[sym] = pd.DataFrame({
            "Open": close * (1 + rng.normal(0, 0.002, len(idx))),
            "High": close * (1 + np.abs(rng.normal(0, 0.006, len(idx)))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.006, len(idx)))),
            "Close": close, "Volume": 2_000_000.0, "implied_vol": iv,
        }, index=idx)
    return out


def seed(db_path: Path) -> date:
    """Write the fixture into a fresh history database; returns the last stored session."""
    from ait.data.historical import SOURCE_IB_OPTION_IV, SOURCE_IB_TRADES_RTH, SOURCE_IB_VIX_INDEX, HistoricalDataStore
    store = HistoricalDataStore(db_path=db_path)
    frames = make_frames()
    for sym, df in frames.items():
        store.save(sym, df[["Open", "High", "Low", "Close", "Volume"]], source=SOURCE_IB_TRADES_RTH)
        store.save_daily_iv(sym, df["implied_vol"], source=SOURCE_IB_OPTION_IV)
    idx = pd.bdate_range(end=frames["SPY"].index[-1], periods=VIX_SESSIONS)
    rng = np.random.default_rng(7)
    vix = pd.Series(np.clip(0.17 + np.cumsum(rng.normal(0, 0.003, len(idx))) * 0.3, 0.10, 0.40), index=idx)
    store.save_daily_iv_bars("VIX", pd.DataFrame({"Open": vix, "High": vix * 1.02, "Low": vix * 0.98, "Close": vix}),
                             source=SOURCE_IB_VIX_INDEX)
    return frames["SPY"].index[-1].date()


def make_config(**overrides):
    from ait.backtesting.walkforward import WalkForwardConfig
    kw = dict(train_days=200, test_days=63, step_days=63, gap_days=5, initial_capital=CAPITAL,
              train_window_models=False, macro_event_gate=False, history_start_date=START)
    kw.update(overrides)
    return WalkForwardConfig(**kw)


@contextmanager
def frozen_clock(day: date):
    """Every ``today_et`` / ``datetime.now`` the research loaders read says ``day``."""
    import ait.backtesting.walkforward as wf
    import ait.data.historical as hist
    import ait.data.market_data as md

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            moment = datetime(day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc)
            return moment if tz is not None else moment.replace(tzinfo=None)

    saved = (md.today_et, wf.today_et, hist.datetime)
    md.today_et = wf.today_et = lambda: day
    hist.datetime = _FrozenDatetime
    try:
        yield
    finally:
        md.today_et, wf.today_et, hist.datetime = saved


def run(db_path: Path, config=None, today: date | None = None):
    """Run the REAL walk-forward on the seeded store, loading through the real research loaders.

    ``today`` defaults to the day after the last stored session."""
    from ait.backtesting.engine import Backtester
    from ait.backtesting.walkforward import WalkForwardBacktester
    last_day = seed(db_path)
    saved = Backtester.__dict__["_load_earnings_dates"]
    Backtester._load_earnings_dates = staticmethod(_no_earnings)
    try:
        with frozen_clock(today or last_day + timedelta(days=1)):
            bt = WalkForwardBacktester(list(SYMBOLS), ["iron_condor"], config=config or make_config(),
                                       db_path=db_path, table_prefix="")
            return asyncio.run(bt.run())
    finally:
        Backtester._load_earnings_dates = saved


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        snap = snapshot(run(Path(d) / "historical.db"))
    GOLDEN_PATH.write_text(json.dumps(snap, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {GOLDEN_PATH}: {len(snap['windows'])} windows, {snap['metrics']['total_trades']} trades")

"""W91: the one IB implied-vol refresh (script backfill and the daily retrain share it).

`daily_iv` (OHLC of the daily OPTION_IMPLIED_VOLATILITY bar) and `daily_prices.implied_vol` (its close,
the live IV-rank input) had no scheduled writer: they ended 2026-09-11, and the VIX family
(VIX, VIX3M, VIX6M, VIX9D) ended 2026-08-25. This module fetches from IB only and writes:

  * stocks/ETFs: OPTION_IMPLIED_VOLATILITY daily bars -> `daily_iv` (source IB_OPTION_IV) and the close ->
    `daily_prices.implied_vol`;
  * the CBOE VIX-family indices: daily TRADES, divided by 100 (the stored scale, 0.1837 = VIX3M 18.37)
    -> `daily_iv` only (source IB_INDEX_TRADES).

`append_only=True` (the daily retrain) fills only what is missing; today's mid-session bar is never stored
(`drop_incomplete_session`). Every symbol is isolated: one failure is reported and the rest still run.
`ib` is the ib_insync `IB` object (`IBKRClient.ib`); the store is a `HistoricalDataStore`.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from ait.data.historical import SOURCE_IB_OPTION_IV, SOURCE_IB_VIX_INDEX
from ait.utils.logging import get_logger
from ait.utils.time import drop_incomplete_session, now_et

log = get_logger("data.iv_refresh")

VIX_FAMILY = ("VIX", "VIX3M", "VIX6M", "VIX9D")
_DEFAULT_DURATION = "1 M"


def parse_ohlc_bars(bars: Any, scale: float = 1.0) -> pd.DataFrame:
    """IB bars (attributes date/open/high/low/close) -> DataFrame[Open,High,Low,Close] on a DatetimeIndex.

    Rows with a non-positive close or unparsable numbers are skipped; values are multiplied by `scale`."""
    rows: dict[str, tuple[float, float, float, float]] = {}
    for b in bars or []:
        try:
            c = float(b.close) * scale
            if c <= 0:
                continue
            rows[str(b.date)[:10]] = (float(b.open) * scale, float(b.high) * scale,
                                      float(b.low) * scale, c)
        except (TypeError, ValueError, AttributeError):
            continue
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


async def fetch_option_iv(ib: Any, symbol: str, duration: str = _DEFAULT_DURATION,
                          end: str = "") -> pd.DataFrame:
    """Daily OPTION_IMPLIED_VOLATILITY OHLC (decimals, 0.25 = 25%) of a stock/ETF."""
    from ib_insync import Stock

    qualified = await ib.qualifyContractsAsync(Stock(symbol, "SMART", "USD"))
    if not qualified:
        raise RuntimeError(f"could not qualify {symbol}")
    bars = await ib.reqHistoricalDataAsync(
        qualified[0], endDateTime=end, durationStr=duration, barSizeSetting="1 day",
        whatToShow="OPTION_IMPLIED_VOLATILITY", useRTH=True, formatDate=1)
    return parse_ohlc_bars(bars)


async def fetch_index_levels(ib: Any, symbol: str, duration: str = _DEFAULT_DURATION,
                             end: str = "") -> pd.DataFrame:
    """Daily TRADES OHLC of a CBOE index, divided by 100 (the stored scale)."""
    from ib_insync import Index

    qualified = await ib.qualifyContractsAsync(Index(symbol, "CBOE", "USD"))
    if not qualified:
        raise RuntimeError(f"could not qualify index {symbol}")
    bars = await ib.reqHistoricalDataAsync(
        qualified[0], endDateTime=end, durationStr=duration, barSizeSetting="1 day",
        whatToShow="TRADES", useRTH=True, formatDate=1)
    return parse_ohlc_bars(bars, scale=0.01)


def store_option_iv(store: Any, symbol: str, ohlc: pd.DataFrame, *, append_only: bool,
                    now: datetime | None = None) -> tuple[int, int]:
    """Write a stock's IV frame. Returns (daily_iv rows written, daily_prices.implied_vol rows set)."""
    ohlc = drop_incomplete_session(ohlc, now)
    if ohlc is None or ohlc.empty:
        return 0, 0
    bars_df, vals_df = ohlc, ohlc
    if append_only:
        have_bars, have_vals = store.iv_dates_present(symbol, ohlc.index.min().date())
        bars_df = ohlc[~ohlc.index.strftime("%Y-%m-%d").isin(have_bars)]
        vals_df = ohlc[~ohlc.index.strftime("%Y-%m-%d").isin(have_vals)]
    n_bars = store.save_daily_iv_bars(symbol, bars_df, source=SOURCE_IB_OPTION_IV) if len(bars_df) else 0
    n_vals = store.save_daily_iv(symbol, vals_df["Close"].rename("implied_vol"),
                                 source=SOURCE_IB_OPTION_IV) if len(vals_df) else 0
    return n_bars, n_vals


def store_index_levels(store: Any, symbol: str, ohlc: pd.DataFrame, *, append_only: bool,
                       now: datetime | None = None) -> int:
    """Write a VIX-family frame into daily_iv. Returns rows written."""
    ohlc = drop_incomplete_session(ohlc, now)
    if ohlc is None or ohlc.empty:
        return 0
    if append_only:
        have_bars, _ = store.iv_dates_present(symbol, ohlc.index.min().date())
        ohlc = ohlc[~ohlc.index.strftime("%Y-%m-%d").isin(have_bars)]
    return store.save_daily_iv_bars(symbol, ohlc, source=SOURCE_IB_VIX_INDEX) if len(ohlc) else 0


async def refresh_daily_iv(ib: Any, store: Any, symbols: list[str], *, vix_family: bool = True,
                           duration: str = _DEFAULT_DURATION, append_only: bool = True,
                           now: datetime | None = None, pause_secs: float = 1.0) -> dict[str, dict]:
    """Refresh IV for `symbols` (and the VIX family). Never raises; per-symbol results:
    {"rows": n, "latest": "YYYY-MM-DD" | None, "error": str | None}."""
    now = now or now_et()
    out: dict[str, dict] = {}
    jobs = [(s, "option") for s in symbols] + ([(v, "index") for v in VIX_FAMILY] if vix_family else [])
    for sym, kind in jobs:
        res: dict = {"rows": 0, "latest": None, "error": None}
        try:
            if kind == "option":
                frame = await fetch_option_iv(ib, sym, duration)
                n_bars, n_vals = store_option_iv(store, sym, frame, append_only=append_only, now=now)
                res["rows"] = n_bars + n_vals
            else:
                frame = await fetch_index_levels(ib, sym, duration)
                res["rows"] = store_index_levels(store, sym, frame, append_only=append_only, now=now)
            kept = drop_incomplete_session(frame, now)
            if kept is not None and not kept.empty:
                res["latest"] = kept.index.max().date().isoformat()
        except Exception as e:  # noqa: BLE001 - one symbol must not stop the rest
            res["error"] = f"{type(e).__name__}: {e}"
            log.warning("iv_refresh_symbol_failed", symbol=sym, error=res["error"])
        out[sym] = res
        if pause_secs and not res["error"]:       # IB pacing between successful requests only
            await asyncio.sleep(pause_secs)
    return out


def summary_line(results: dict[str, dict]) -> str:
    """One log/print line: `iv_refresh QQQ:+2->2026-10-22 VIX3M:+0->2026-10-22 SPY:ERR`."""
    parts = []
    for sym, r in results.items():
        parts.append(f"{sym}:ERR" if r.get("error") else f"{sym}:+{r['rows']}->{r.get('latest')}")
    return "iv_refresh " + " ".join(parts)

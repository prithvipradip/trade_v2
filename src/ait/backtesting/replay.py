"""W72: research-vs-reality PARITY REPLAY (read-only).

Why this exists. ``run_backtest.build_parity_manifest`` compares PARAMETERS
(live vs research config). Nothing measured whether the research engine would
have PRICED, FILLED or EXITED a real trade the way reality did. This module runs
known real trades (the operator's manual trades + the bot's own fills) through
the research engine's pricing code and compares:

  * entry credit      modelled vs the actual fill (per share, signed)
  * per-leg price     modelled mid +/- fitted half-spread vs the actual leg fill
  * exit value        modelled re-mark vs the actual buy-back (same-instant closes)
  * exit decision     which rule the ENGINE's own ``_check_exit`` would have fired
                      and when, vs what actually happened
  * expiry settlement W77: the legs ``ait.backtesting.settlement`` says expired / were
                      exercised / assigned, the shares carried over the gap and the
                      resulting P&L, vs the statement's close codes and stock lines

Single-owner rule. Leg pricing is ``ait.backtesting.pricing.ic_leg_marks`` /
``skewed_leg_iv`` — the very functions the engine's entry build and daily re-mark
call — so the replay cannot drift from the engine. The exit scan calls the
engine's real ``Backtester._check_exit`` on a position built from the ACTUAL
strikes; nothing is re-implemented here.

Two model variants are always reported, so the error decomposes:

  engine   what the engine does today: the DAILY close as spot, the same-day
           stored IV (so day D's end-of-day information prices a position
           'entered' earlier in D: the W72 look-ahead lead L2), integer days
           to expiry.
  clean    information available at the fill: the 5-minute bar's mid as spot,
           the PREVIOUS session's IV (assumption A3), fractional time to the
           16:00 ET expiry close (assumption A9).

Sign convention everywhere: error = model - actual. A positive entry-credit
error means the research model is RICHER than reality (research too optimistic).

Pure functions + a read-only sqlite adapter; no live-state writes. Tests drive
the real functions on synthetic fixtures (tests/test_w72_parity_replay.py).
"""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from ait.backtesting import iv_model  # W87a: the IV proxy shared with the engine
from ait.backtesting.friction import LegSpreadModel
from ait.backtesting.pricing import (
    OptionType,
    black_scholes_price,
    ic_leg_marks,
    skewed_leg_iv,
)
from ait.backtesting.study_harness import STUDY_INITIAL_CAPITAL
from ait.config.settings import (
    DEFAULT_RISK_FREE_RATE,
    BacktestConfig,
    EngineTuningConfig,
    loaded_engine_tuning,
)
from ait.utils.time import ET

# engine.py used a bare r = 0.05 at entry and in the daily re-mark (W72 finding); since W81 it reads
# options.risk_free_rate and PricingParams.from_backtest_config takes that value from the caller.
# This constant is only the default when no setting is passed (it equals the config default).
ENGINE_RISK_FREE_RATE = DEFAULT_RISK_FREE_RATE   # W88b: was a literal; still importable
EXPIRY_CLOSE_ET = time(16, 0)          # PM-settled ETF options stop trading 16:00 ET
BAR_MINUTES = 5                        # research_intraday_prices interval
SCHEMA = "ait.replay.manual_trades/1"
VARIANTS = ("engine", "clean")
# W75: entry parity also has a 'surface' variant (the 'clean' inputs with each leg priced
# from the real-bar smile of its own expiry, fit on the previous session). It has no exit
# counterpart, so exit_replay keeps iterating VARIANTS.
ENTRY_VARIANTS = VARIANTS + ("surface",)
PARITY_STRUCTURES = ("iron_condor", "vertical_spread")   # what the engine can build


# --------------------------------------------------------------------------
# Trades
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayLeg:
    side: str                      # "short" | "long"
    right: str                     # "C" | "P"
    strike: float
    expiry: date
    open_price: float
    close_time: datetime | None
    close_price: float | None
    close_code: str | None
    realized_usd: float

    @property
    def option_type(self) -> OptionType:
        return OptionType.CALL if self.right == "C" else OptionType.PUT

    @property
    def sign(self) -> int:
        """+1 for a short leg (premium received), -1 for a long leg (paid)."""
        return 1 if self.side == "short" else -1


@dataclass(frozen=True)
class ReplayStock:
    """One stock line delivered by exercise/assignment (or closing it), from the statement."""
    time: datetime
    shares: int                    # signed: +bought, -sold
    price: float
    commission: float
    code: str


@dataclass(frozen=True)
class ReplayTrade:
    id: str
    origin: str                    # "manual" | "bot"
    bot_trade_id: str | None
    entry_time: datetime           # tz-aware, America/New_York
    quantity: int
    structure: str
    legs: tuple[ReplayLeg, ...]
    still_open: bool
    net_credit_usd: float          # whole structure, all contracts, USD
    commissions_usd: float
    total_pnl_usd: float | None
    linked_stock: tuple["ReplayStock", ...] = ()   # W77: stock delivered by exercise/assignment

    @property
    def entry_date(self) -> date:
        return self.entry_time.date()

    @property
    def front_expiry(self) -> date:
        return min(l.expiry for l in self.legs)

    @property
    def entry_dte(self) -> int:
        """Calendar days from entry to the FRONT expiry (the band key)."""
        return (self.front_expiry - self.entry_date).days

    @property
    def actual_credit(self) -> float:
        """Net credit per share (negative = a debit structure)."""
        return self.net_credit_usd / (100.0 * self.quantity)

    def condor_roles(self) -> dict[str, ReplayLeg] | None:
        """{short_call, short_put, long_call, long_put} for a single-expiry
        four-leg condor, else None."""
        if len(self.legs) != 4 or len({l.expiry for l in self.legs}) != 1:
            return None
        roles: dict[str, ReplayLeg] = {}
        for l in self.legs:
            key = f"{l.side}_{'call' if l.right == 'C' else 'put'}"
            if key in roles:
                return None
            roles[key] = l
        return roles if len(roles) == 4 else None

    @property
    def parity_eligible(self) -> bool:
        """Structures the research engine can build, so a verdict is meaningful."""
        return self.structure in PARITY_STRUCTURES

    def legs_closed_together(self) -> datetime | None:
        """The shared close instant if every leg closed at the same timestamp
        by a plain trader close (no expiry/assignment/exercise), else None."""
        if self.still_open or any(l.close_time is None for l in self.legs):
            return None
        if any(l.close_code != "C" for l in self.legs):
            return None
        stamps = {l.close_time for l in self.legs}
        return next(iter(stamps)) if len(stamps) == 1 else None


def _et(ts: str) -> datetime:
    d = datetime.fromisoformat(ts)
    return d.replace(tzinfo=ET) if d.tzinfo is None else d.astimezone(ET)


def load_trades(path: str | Path) -> list[ReplayTrade]:
    """Load the replay file written under data/replay (git-ignored)."""
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"unexpected replay schema {doc.get('schema')!r}; want {SCHEMA!r}")
    out: list[ReplayTrade] = []
    for s in doc["structures"]:
        legs = tuple(ReplayLeg(
            side=l["side"], right=l["right"], strike=float(l["strike"]),
            expiry=date.fromisoformat(l["expiry"]), open_price=float(l["open_price"]),
            close_time=_et(l["close_time"]) if l.get("close_time") else None,
            close_price=(None if l.get("close_price") is None else float(l["close_price"])),
            close_code=l.get("close_code"), realized_usd=float(l.get("realized_usd") or 0.0),
        ) for l in s["legs"])
        out.append(ReplayTrade(
            id=s["id"], origin=s["origin"], bot_trade_id=s.get("bot_trade_id"),
            entry_time=_et(s["entry_time"]), quantity=int(s["quantity"]),
            structure=s["structure"], legs=legs, still_open=bool(s["still_open"]),
            net_credit_usd=float(s["net_credit_usd"]),
            commissions_usd=float(s.get("commissions_usd") or 0.0),
            total_pnl_usd=(None if s.get("total_pnl_usd") is None else float(s["total_pnl_usd"])),
            linked_stock=tuple(ReplayStock(
                time=_et(x["time"]), shares=int(x["shares"]), price=float(x["price"]),
                commission=float(x.get("commission") or 0.0), code=str(x.get("code") or ""),
            ) for x in (s.get("linked_stock") or [])),
        ))
    return out


# --------------------------------------------------------------------------
# Research market data (read-only sqlite; never writes, never migrates)
# --------------------------------------------------------------------------

class ResearchData:
    """Read-only view of the stored history tables in data/historical.db.

    W91: the tables are shared (prefix "" by default). `end_date` pins the last day the daily frame and the
    intraday bars return (backtest.history_end_date), so a study's inputs stay fixed as the tables grow."""

    def __init__(self, db_path: str | Path, symbol: str, prefix: str = "",
                 end_date: "date | None" = None) -> None:
        self._uri = f"file:{db_path}?mode=ro"
        self.symbol = symbol.upper()
        self._end = end_date
        self._daily = f"{prefix}daily_prices"
        self._intra = f"{prefix}intraday_prices"
        self._frame: pd.DataFrame | None = None

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._uri, uri=True)

    def daily_frame(self) -> pd.DataFrame:
        """Open/High/Low/Close/Volume/implied_vol indexed by date (cached)."""
        if self._frame is None:
            with self._conn() as c:
                df = pd.read_sql_query(
                    f"SELECT date, open AS Open, high AS High, low AS Low, close AS Close, "
                    f"volume AS Volume, implied_vol FROM {self._daily} "
                    f"WHERE symbol = ?" + (" AND date <= ?" if self._end else "") + " ORDER BY date",
                    c, params=((self.symbol, str(self._end)) if self._end else (self.symbol,)))
            df.index = pd.to_datetime(df.pop("date"))
            self._frame = df
        return self._frame

    def session_row(self, d: date) -> pd.Series | None:
        f = self.daily_frame()
        ts = pd.Timestamp(d)
        return f.loc[ts] if ts in f.index else None

    def stored_iv(self, d: date, *, strictly_before: bool = False) -> float | None:
        """Last non-null stored IV on/before (or strictly before) ``d``."""
        f = self.daily_frame()
        s = f["implied_vol"].dropna()
        s = s[s.index < pd.Timestamp(d)] if strictly_before else s[s.index <= pd.Timestamp(d)]
        s = s[s > 0]
        return float(s.iloc[-1]) if len(s) else None

    def closes_through(self, d: date) -> list[float]:
        f = self.daily_frame()
        return [float(x) for x in f.loc[f.index <= pd.Timestamp(d), "Close"].values]

    def bar_containing(self, ts_et: datetime) -> dict[str, float] | None:
        """The 5-minute bar whose [start, start+5m) contains ``ts_et`` (or None
        when the instant is outside a session / the bar is missing)."""
        ts_utc = ts_et.astimezone(timezone.utc)
        iso = ts_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")   # the store's stored form
        with self._conn() as c:
            row = c.execute(
                f"SELECT datetime, open, high, low, close FROM {self._intra} "
                f"WHERE symbol = ? AND interval = '5m' AND datetime <= ? "
                f"ORDER BY datetime DESC LIMIT 1", (self.symbol, iso)).fetchone()
        if row is None:
            return None
        start = datetime.fromisoformat(row[0])
        if not (timedelta(0) <= ts_utc - start < timedelta(minutes=BAR_MINUTES)):
            return None
        return {"open": float(row[1]), "high": float(row[2]),
                "low": float(row[3]), "close": float(row[4])}

    def load_intraday_range(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Read-only 5-minute bars for ``[start_date, end_date]`` (UTC index, OHLCV).

        The shape the engine's ``intraday_store`` contract expects, so research studies can hand a
        ``ResearchData`` straight to ``Backtester(intraday_store=...)`` without opening the
        research tables through ``HistoricalDataStore`` (whose constructor may create tables)."""
        cache = self.__dict__.setdefault("_intra_cache", {})
        key = (symbol.upper(), start_date, end_date)
        if key in cache:
            return cache[key]
        lo = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
        if self._end is not None and end_date > self._end:      # W91: never past the pinned last day
            end_date = self._end
        hi = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)
        fmt = "%Y-%m-%dT%H:%M:%S+00:00"
        with self._conn() as c:
            df = pd.read_sql_query(
                f"SELECT datetime, open AS Open, high AS High, low AS Low, close AS Close, "
                f"volume AS Volume FROM {self._intra} WHERE symbol = ? AND interval = '5m' "
                f"AND datetime >= ? AND datetime <= ? ORDER BY datetime",
                c, params=(symbol.upper(), lo.strftime(fmt), hi.strftime(fmt)))
        if not df.empty:
            df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("datetime"), utc=True))
        cache[key] = df
        return df

    def leg_spread_model(self) -> LegSpreadModel | None:
        """Fitted per-leg half-spread model (un-prefixed table, as the engine reads it)."""
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT form, a, b FROM option_leg_spread_params WHERE symbol = ?",
                    (self.symbol,)).fetchone()
        except sqlite3.Error:
            return None
        return LegSpreadModel.from_row({"form": row[0], "a": row[1], "b": row[2]}) if row else None


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PricingParams:
    """The engine's pricing knobs (resolve from the research-overlaid settings)."""
    # W90a: a bare PricingParams() carries the BacktestConfig SCHEMA defaults (single home), not copies of them
    r: float = ENGINE_RISK_FREE_RATE
    skew_factor: float = BacktestConfig.model_fields["skew_factor"].default
    skew_slope_per_pct_otm: float = BacktestConfig.model_fields["skew_slope_per_pct_otm"].default
    iv_floor: float = BacktestConfig.model_fields["iv_floor"].default
    # W90c: the leg-IV model's skew coefficients (backtest.engine_tuning), also schema defaults when bare
    put_skew_coefficient: float = EngineTuningConfig.model_fields["put_skew_coefficient"].default
    call_skew_coefficient: float = EngineTuningConfig.model_fields["call_skew_coefficient"].default

    @classmethod
    def from_backtest_config(cls, bc: Any, risk_free_rate: float | None = None) -> PricingParams:
        # W81: the engine now reads options.risk_free_rate; the replay must price with the same r
        # (callers pass settings.options.risk_free_rate; None keeps the 0.05 default).
        eng = getattr(bc, "engine_tuning", None) or EngineTuningConfig()     # a stub config degrades to the schema defaults
        return cls(r=float(risk_free_rate) if risk_free_rate is not None else ENGINE_RISK_FREE_RATE,
                   skew_factor=float(bc.skew_factor),
                   skew_slope_per_pct_otm=float(bc.skew_slope_per_pct_otm),
                   iv_floor=float(bc.iv_floor),
                   put_skew_coefficient=float(eng.put_skew_coefficient),
                   call_skew_coefficient=float(eng.call_skew_coefficient))


def t_engine(expiry: date, as_of: date) -> float:
    """The engine's time to expiry: whole calendar days / 365 (entry build)."""
    return max((expiry - as_of).days, 0) / 365.0


def t_clean(expiry: date, as_of: datetime) -> float:
    """Fractional calendar time from ``as_of`` to the 16:00 ET expiry close."""
    close = datetime.combine(expiry, EXPIRY_CLOSE_ET, tzinfo=ET)
    return max((close - as_of).total_seconds(), 0.0) / (365.0 * 86400.0)


def model_leg_mids(trade: ReplayTrade, S: float, base_iv: float,
                   t_by_leg: Sequence[float], p: PricingParams,
                   leg_ivs: Sequence[float] | None = None) -> list[float]:
    """Modelled mid per leg, in ``trade.legs`` order.

    A single-expiry condor goes through ``ic_leg_marks`` — the engine's call —
    everything else (verticals, calendars) through the same per-leg primitives.
    ``leg_ivs`` (W75) prices each leg at an explicit IV instead (the surface).
    """
    if leg_ivs is not None:
        return [black_scholes_price(S, l.strike, t, p.r, float(iv), l.option_type)
                for l, t, iv in zip(trade.legs, t_by_leg, leg_ivs)]
    roles = trade.condor_roles()
    if roles is not None:
        t = t_by_leg[0]
        sc, sp, lc, lp = ic_leg_marks(
            S, roles["short_call"].strike, roles["short_put"].strike,
            roles["long_call"].strike, roles["long_put"].strike,
            t, p.r, base_iv, p.skew_factor, p.skew_slope_per_pct_otm, p.iv_floor,
            p.put_skew_coefficient, p.call_skew_coefficient)
        by_role = {"short_call": sc, "short_put": sp, "long_call": lc, "long_put": lp}
        return [by_role[f"{l.side}_{'call' if l.right == 'C' else 'put'}"] for l in trade.legs]
    out = []
    for l, t in zip(trade.legs, t_by_leg):
        iv = skewed_leg_iv(base_iv, l.strike, S, l.option_type, p.skew_factor,
                           p.skew_slope_per_pct_otm, p.iv_floor,
                           p.put_skew_coefficient, p.call_skew_coefficient)
        out.append(black_scholes_price(S, l.strike, t, p.r, iv, l.option_type))
    return out


def implied_vol_from_price(price: float, S: float, K: float, t: float, r: float,
                           option_type: OptionType) -> float | None:
    """Black-Scholes implied vol by bracketing; None when the price is outside
    the model's reachable range (below intrinsic, or no root in [1%, 300%])."""
    if price <= 0.0 or t <= 0.0 or S <= 0.0 or K <= 0.0:
        return None
    lo, hi = 0.01, 3.0
    f_lo = black_scholes_price(S, K, t, r, lo, option_type) - price
    f_hi = black_scholes_price(S, K, t, r, hi, option_type) - price
    if f_lo > 0.0 or f_hi < 0.0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if black_scholes_price(S, K, t, r, mid, option_type) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def structure_credit(trade: ReplayTrade, per_leg: Sequence[float]) -> float:
    """Net credit per share for per-leg prices in ``trade.legs`` order."""
    return sum(l.sign * px for l, px in zip(trade.legs, per_leg))


def _fill_price(leg: ReplayLeg, mid: float, hs: float) -> float:
    """Where an OPENING order fills: short sells at the bid, long buys the ask."""
    return mid - hs if leg.side == "short" else mid + hs


def entry_replay(trade: ReplayTrade, data: ResearchData, p: PricingParams,
                 friction: LegSpreadModel | None, surface: Any = None) -> dict[str, Any]:
    """Model the entry under every variant. Returns {variant: result | None}.

    ``surface`` (an ``ait.backtesting.iv_surface.IVSurface``) adds the W75
    'surface' variant; without it that key is None.
    """
    result: dict[str, Any] = {}
    D = trade.entry_date
    actual = trade.actual_credit

    row = data.session_row(D)
    iv_same = data.stored_iv(D)
    if row is not None and iv_same is not None:
        S = float(row["Close"])
        ts = [t_engine(l.expiry, D) for l in trade.legs]
        result["engine"] = _entry_variant(trade, S, iv_same, ts, p, friction, actual,
                                          spot_src="daily_close", iv_src="same_day")
    else:
        result["engine"] = None

    bar = data.bar_containing(trade.entry_time)
    iv_prev = data.stored_iv(D, strictly_before=True)
    if bar is not None and iv_prev is not None:
        S = 0.5 * (bar["open"] + bar["close"])
        ts = [t_clean(l.expiry, trade.entry_time) for l in trade.legs]
        res = _entry_variant(trade, S, iv_prev, ts, p, friction, actual,
                             spot_src="bar_mid", iv_src="prev_session")
        res["bar_range"] = bar["high"] - bar["low"]
        result["clean"] = res
    else:
        result["clean"] = None

    result["surface"] = None
    if surface is not None and bar is not None:
        S = 0.5 * (bar["open"] + bar["close"])
        prev_day = D - timedelta(days=1)
        leg_ivs = [surface.iv(prev_day, l.expiry, l.strike, S) for l in trade.legs]
        if all(v is not None for v in leg_ivs):
            ts = [t_clean(l.expiry, trade.entry_time) for l in trade.legs]
            # base_iv is only a label here; every leg carries its own surface IV
            res = _entry_variant(trade, S, float(statistics.mean(float(v) for v in leg_ivs)), ts, p, friction, actual,
                                 spot_src="bar_mid", iv_src="surface_prev_session",
                                 leg_ivs=[float(v) for v in leg_ivs])
            res["bar_range"] = bar["high"] - bar["low"]
            result["surface"] = res
    return result


def _entry_variant(trade, S, iv, t_by_leg, p, friction, actual, *, spot_src, iv_src,
                   leg_ivs: Sequence[float] | None = None) -> dict[str, Any]:
    mids = model_leg_mids(trade, S, iv, t_by_leg, p, leg_ivs)
    hs = [friction.half_spread_usd(m) if friction is not None else 0.0 for m in mids]
    fills = [_fill_price(l, m, h) for l, m, h in zip(trade.legs, mids, hs)]
    credit_mid = structure_credit(trade, mids)
    credit_fill = structure_credit(trade, fills)
    denom = abs(actual) if abs(actual) > 1e-9 else float("nan")
    # The IV the MARKET used for each leg: invert BS on the actual fill after
    # removing the (modelled) half-spread the order crossed. Compared with the
    # engine's single stored IV this is the term-structure/skew error, directly.
    model_leg_iv = (list(leg_ivs) if leg_ivs is not None else
                    [skewed_leg_iv(iv, l.strike, S, l.option_type, p.skew_factor,
                                   p.skew_slope_per_pct_otm, p.iv_floor,
                                   p.put_skew_coefficient, p.call_skew_coefficient) for l in trade.legs])
    actual_leg_iv = []
    for l, h, t in zip(trade.legs, hs, t_by_leg):
        mid_est = l.open_price + h if l.side == "short" else l.open_price - h
        actual_leg_iv.append(implied_vol_from_price(
            mid_est, S, l.strike, t, p.r, l.option_type))
    return {
        "S": S, "iv": iv, "t_front_years": min(t_by_leg), "spot_src": spot_src, "iv_src": iv_src,
        "friction_modelled": friction is not None,
        "leg_mid": mids, "leg_half_spread": hs, "leg_fill": fills,
        "credit_mid": credit_mid, "friction_total": sum(hs), "credit_fill": credit_fill,
        "actual_credit": actual,
        "err": credit_fill - actual,                 # primary: model fill vs actual fill
        "err_pct": (credit_fill - actual) / denom,
        "err_mid": credit_mid - actual,              # mid vs fill (includes real friction)
        "leg_err": [f - l.open_price for f, l in zip(fills, trade.legs)],
        "leg_actual_iv": actual_leg_iv,
        "leg_model_iv": model_leg_iv,    # base IV + the engine's skew, per leg
    }


# --------------------------------------------------------------------------
# Exit value and exit decision
# --------------------------------------------------------------------------

def engine_current_iv(entry_iv: float, closes: Sequence[float], iv_floor: float) -> float:
    """The engine's re-mark IV: 70% entry IV + 30% (1.15 x rv20); entry IV when
    fewer than 21 closes exist. W87a: the SAME function ``Backtester._get_current_iv`` calls
    (``backtesting.iv_model.remark_iv``), so the two cannot drift."""
    return iv_model.remark_iv(entry_iv, pd.Series(closes).values, iv_floor)


def exit_replay(trade: ReplayTrade, data: ResearchData, p: PricingParams,
                friction: LegSpreadModel | None, entry: dict[str, Any]) -> dict[str, Any] | None:
    """Modelled buy-back value vs the actual one, for structures whose legs were
    all closed by the trader at one instant. None when not applicable."""
    when = trade.legs_closed_together()
    if when is None:
        return None
    actual = sum(l.sign * (l.close_price or 0.0) for l in trade.legs)  # cost to close per share
    D = when.date()
    closes = data.closes_through(D)
    out: dict[str, Any] = {"actual_exit_value": actual, "close_time": when.isoformat()}
    bar = data.bar_containing(when)
    multi_expiry = len({l.expiry for l in trade.legs}) > 1
    for variant in VARIANTS:
        e = entry.get(variant)
        # The engine prices every leg at ONE expiry (``_entry_dte``), so its
        # re-mark of a calendar is meaningless; only the clean variant applies.
        if e is None or (variant == "clean" and bar is None) or (variant == "engine" and multi_expiry):
            out[variant] = None
            continue
        iv = engine_current_iv(e["iv"], closes, p.iv_floor)
        if variant == "engine":
            row = data.session_row(D)
            if row is None:
                out[variant] = None
                continue
            S = float(row["Close"])
            days_held = (D - trade.entry_date).days
            t = max(max(trade.entry_dte - days_held, 0) / 365.0, 0.0001)
            ts = [t] * len(trade.legs)
        else:
            S = 0.5 * (bar["open"] + bar["close"])
            ts = [max(t_clean(l.expiry, when), 0.0001 / 365.0) for l in trade.legs]
        mids = model_leg_mids(trade, S, iv, ts, p)
        hs = [friction.half_spread_usd(m) if friction is not None else 0.0 for m in mids]
        # closing: buy back shorts at the ask, sell longs at the bid
        value_mid = sum(l.sign * m for l, m in zip(trade.legs, mids))
        value_fill = value_mid + sum(hs)
        out[variant] = {"S": S, "iv": iv, "value_mid": value_mid, "value_fill": value_fill,
                        "err": value_fill - actual}
    return out


def engine_exit_scan(trade: ReplayTrade, data: ResearchData, entry: dict[str, Any],
                     settings_overrides: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Which exit the ENGINE'S OWN ``_check_exit`` would fire, and when.

    Builds a ``Backtester`` on the real daily frame and a position from the
    ACTUAL strikes and the ACTUAL credit (so the scan isolates exit-RULE parity
    from entry-pricing error), then walks the sessions after entry, exactly as
    ``run()`` does (a position is first exit-checked the session AFTER entry).
    Condors only. Never touches a database: legacy friction, calibrated spreads
    off.
    """
    roles = trade.condor_roles()
    e = entry.get("engine")
    if roles is None or e is None:
        return None
    import os

    from ait.backtesting.engine import Backtester

    frame = data.daily_frame()
    start = pd.Timestamp(trade.entry_date)
    end = pd.Timestamp(trade.front_expiry)
    # W95: the 60 calendar days are backtest.engine_tuning.replay_lookback_days
    window = frame.loc[(frame.index >= start - pd.Timedelta(days=loaded_engine_tuning().replay_lookback_days))
                       & (frame.index <= end)]
    if window.empty or pd.Timestamp(trade.entry_date) not in window.index:
        return None

    prev = os.environ.get("AIT_BT_CALIBRATED_SPREADS")
    os.environ["AIT_BT_CALIBRATED_SPREADS"] = "0"
    try:
        # W90a: the replay needs capital never to bind (it re-prices ONE recorded trade), so it uses the named
        # study-harness capital, single-sourced with the gate studies, not account.capital
        kw = dict(data=window, strategies=["iron_condor"], initial_capital=STUDY_INITIAL_CAPITAL,
                  entry_dte=max(trade.entry_dte, 1), features_cache=None,
                  friction_model="legacy", macro_event_gate=False,
                  allow_live_model_fallback=False)
        kw.update(settings_overrides or {})
        bt = Backtester(**kw)
    finally:
        if prev is None:
            os.environ.pop("AIT_BT_CALIBRATED_SPREADS", None)
        else:
            os.environ["AIT_BT_CALIBRATED_SPREADS"] = prev

    pos = {
        "strategy": "iron_condor", "trade_type": "credit",
        "entry_date": str(trade.entry_date), "entry_price": float(trade.actual_credit),
        "short_call_strike": roles["short_call"].strike, "short_put_strike": roles["short_put"].strike,
        "long_call_strike": roles["long_call"].strike, "long_put_strike": roles["long_put"].strike,
        "entry_iv": float(e["iv"]), "expiry_date": str(trade.front_expiry),
        "contracts": trade.quantity, "n_legs": 4, "high_water_mark": 0.0,
    }
    sessions = window.index[window.index > start]
    last_seen = None
    for ts in sessions:
        d = ts.date()
        last_seen = d
        res = bt._check_exit(dict(pos), window.loc[ts], d, window.loc[:ts])
        if res is not None:
            return {"engine_exit_date": str(d), "engine_exit_reason": res.get("exit_reason"),
                    "data_through": str(last_seen)}
    return {"engine_exit_date": None, "engine_exit_reason": None,
            "data_through": None if last_seen is None else str(last_seen)}


# --------------------------------------------------------------------------
# W77: expiry settlement replay and the live -> research exit-reason map
# --------------------------------------------------------------------------

_CODE_STATUS = {"Ep": "expired", "Ex": "exercised", "A": "assigned"}   # IB statement codes
SETTLE_BAR_START = time(15, 55)   # the last 5-minute bar of a regular session; its close is the settle price


def actual_leg_status(leg: ReplayLeg) -> str | None:
    """What the statement says happened to a leg at expiry (None: a plain trader close)."""
    for tok in str(leg.close_code or "").split(";"):
        if tok in _CODE_STATUS:
            return _CODE_STATUS[tok]
    return None


def settlement_replay(trade: ReplayTrade, data: ResearchData, *, pin_band: float,
                      assignment_fee: float, stock_commission: float) -> dict[str, Any] | None:
    """Run ``ait.backtesting.settlement`` on a real structure that reached its expiry.

    Returns None when the structure was closed by the trader before the bell, is still open,
    or has legs on several expiries. ``status='no_data'`` when the underlying's settle price
    or next open is missing. Otherwise a dict comparing, per leg, the model's outcome with the
    statement's code, the shares left over the gap, and the P&L (statement total vs model).

    The next-open mark is the model's convention (assumption A5); ``close_after_open_min``
    reports how long after that open the real shares were actually closed, so a trader who
    waited is visible instead of being booked as model error.
    """
    from ait.backtesting import settlement as st

    if trade.still_open or len({l.expiry for l in trade.legs}) != 1:
        return None
    actual = [actual_leg_status(l) for l in trade.legs]
    if not any(a is not None for a in actual):
        return None
    expiry = trade.front_expiry
    legs = [st.SettleLeg(l.strike, l.right, l.side) for l in trade.legs]

    bar = data.bar_containing(datetime.combine(expiry, SETTLE_BAR_START, tzinfo=ET))
    row = data.session_row(expiry)
    S = bar["close"] if bar else (float(row["Close"]) if row is not None else None)
    frame = data.daily_frame()
    later = frame.loc[frame.index > pd.Timestamp(expiry)]
    O = float(later.iloc[0]["Open"]) if len(later) else None
    out: dict[str, Any] = {"id": trade.id, "expiry": expiry.isoformat(),
                           "actual_statuses": actual}
    if S is None or (O is None and st.needs_overnight(legs, S, pin_band)):
        out["status"] = "no_data"
        return out

    overnight = st.needs_overnight(legs, S, pin_band)
    res = st.worst_case(legs, S, pin_band, O if overnight else S, assignment_fee, stock_commission)
    scored = [(m, a) for m, a in zip(res.statuses, actual) if a is not None]
    expiry_stock = [x for x in trade.linked_stock if x.time.date() == expiry]
    net_actual = sum(x.shares for x in expiry_stock) // 100 // max(trade.quantity, 1)
    pnl_model = ((trade.actual_credit - res.cost) * 100.0 * trade.quantity
                 - res.fees * 100.0 * trade.quantity + trade.commissions_usd)
    close_gap = None
    if overnight and trade.linked_stock and len(later):
        nxt_open = datetime.combine(later.index[0].date(), time(9, 30), tzinfo=ET)
        closes = [x.time for x in trade.linked_stock if x.time.date() > expiry]
        if closes:
            close_gap = round((max(closes) - nxt_open).total_seconds() / 60.0, 1)
    out.update({
        "status": "ok", "S": S, "O": O, "overnight": overnight,
        "model_statuses": list(res.statuses), "legs_scored": len(scored),
        "legs_match": sum(1 for m, a in scored if m == a),
        "net_shares_model": res.net_shares, "net_shares_actual": net_actual,
        "pnl_model": round(pnl_model, 2),
        "pnl_actual": None if trade.total_pnl_usd is None else round(trade.total_pnl_usd, 2),
        "pnl_error": (None if trade.total_pnl_usd is None
                      else round(pnl_model - trade.total_pnl_usd, 2)),
        "close_after_open_min": close_gap,
    })
    return out


# Live exit reasons (execution/portfolio.py + the reconciler) -> the research engine's
# exit_reason(s) that are their counterpart. Longest prefix wins. An empty tuple means
# "no research twin" (a registered gap, not an error).
LIVE_REASON_TO_RESEARCH: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("take_profit_short", ("take_profit_short",)),
    ("partial_take_profit", ()),
    ("take_profit", ("profit_target",)),
    ("short_strike_touch", ("touch_stop",)),
    ("stop_loss", ("credit_loss_limit", "stop_loss")),
    ("trailing_stop", ("trailing_stop",)),
    ("breakeven_stop", ("breakeven_stop",)),
    ("expiry_approaching", ("expiry_approaching", "expiry")),
    ("macro_event_flatten", ("macro_event_flatten",)),
    ("delta_breach", ()),
    ("pre_earnings_iv_crush", ()),
    ("exit_blocked_by_pdt", ()),
    ("reconciler_expired_intrinsic", ("expiry_settled", "assigned_gap", "expiry")),
)


def research_reasons_for(live_reason: str | None) -> tuple[str, ...] | None:
    """The research exit reasons matching a live one; None if the reason is unknown to the map."""
    if not live_reason:
        return None
    text = str(live_reason)
    best: tuple[str, tuple[str, ...]] | None = None
    for prefix, twins in LIVE_REASON_TO_RESEARCH:
        if text.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, twins)
    return None if best is None else best[1]


# --------------------------------------------------------------------------
# Bot execution context (read-only)
# --------------------------------------------------------------------------

def load_bot_exec_context(state_db: str | Path, bot_trade_id: str) -> dict[str, float] | None:
    """signal_price / live_mid / nbbo_spread / fill for a bot trade's combo row."""
    try:
        c = sqlite3.connect(f"file:{state_db}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        row = c.execute(
            "SELECT price, signal_price, live_mid, nbbo_spread FROM executions "
            "WHERE trade_id = ? AND (signal_price IS NOT NULL OR live_mid IS NOT NULL) "
            "ORDER BY exec_time LIMIT 1", (bot_trade_id,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        c.close()
    if row is None:
        return None
    return {"fill": row[0], "signal_price": row[1], "live_mid": row[2], "nbbo_spread": row[3]}


# --------------------------------------------------------------------------
# Banding and verdicts
# --------------------------------------------------------------------------

def band_label(dte: int, edges: Sequence[int]) -> str:
    lo = 0
    for e in edges:
        if dte <= e:
            return f"{lo}-{e}"
        lo = e + 1
    return f"{lo}+"


def band_order(edges: Sequence[int]) -> list[str]:
    out, lo = [], 0
    for e in edges:
        out.append(f"{lo}-{e}")
        lo = e + 1
    out.append(f"{lo}+")
    return out


@dataclass(frozen=True)
class BandVerdict:
    band: str
    variant: str
    n: int
    median_abs_pct: float | None
    median_signed_pct: float | None
    verdict: str            # PASS | FAIL | INSUFFICIENT


def verdict_for(errs_pct: Sequence[float], *, error_max: float, bias_max: float,
                min_trades: int) -> tuple[str, float | None, float | None]:
    """PASS iff median |err| <= error_max AND |median err| <= bias_max, with at
    least ``min_trades`` samples; never PASS on a thin band."""
    vals = [x for x in errs_pct if x is not None and not math.isnan(x)]
    if len(vals) < min_trades:
        return "INSUFFICIENT", None, None
    mabs = statistics.median(abs(x) for x in vals)
    msgn = statistics.median(vals)
    ok = mabs <= error_max and abs(msgn) <= bias_max
    return ("PASS" if ok else "FAIL"), mabs, msgn


def summarise_bands(rows: Iterable[dict[str, Any]], edges: Sequence[int], *,
                    error_max: float, bias_max: float, min_trades: int) -> list[BandVerdict]:
    """``rows``: dicts with band, variant, err_pct, eligible. Calendars and other
    structures the engine cannot build are excluded from verdicts."""
    rows = [r for r in rows if r.get("eligible")]
    out: list[BandVerdict] = []
    for band in band_order(edges):
        for variant in ENTRY_VARIANTS:
            vals = [r["err_pct"] for r in rows if r["band"] == band and r["variant"] == variant]
            v, mabs, msgn = verdict_for(vals, error_max=error_max, bias_max=bias_max,
                                        min_trades=min_trades)
            out.append(BandVerdict(band, variant, len(vals), mabs, msgn, v))
    return out

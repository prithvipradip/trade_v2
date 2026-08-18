"""Options chain service — fetch, filter, and price options with Greeks.

Fetches options chains from IBKR (primary) with Yahoo Finance fallback.
Calculates Greeks using py_vollib for proper Black-Scholes pricing.
"""

from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from ait.broker.ibkr_client import IBKRClient
from ait.config.settings import OptionsConfig
from ait.data.cache import TTLCache
from ait.data.market_data import MarketDataService
from ait.utils.logging import get_logger

log = get_logger("data.options")

# R16: Yahoo serves placeholder impliedVolatility values (~1e-5) off-hours /
# pre-open, and they passed the old `> 0` filters — atm_iv=1e-5 produced a
# ~50x error in expected_move and garbage-sigma deltas. Any IV below this
# floor is non-physical for the traded universe and is treated as MISSING.
MIN_PHYSICAL_IV = 0.005


def _parse_env_override(raw: str | None, cast, default):
    if raw is None:
        return default
    try:
        return cast(raw.split("#", 1)[0].strip())
    except (ValueError, TypeError):
        return default


# R16: contract liquidity used to be filtered from TWO independent sources
# with different defaults — OptionContract.is_liquid read env AIT_LIQ_*
# (defaults 50/100/0.15) while OptionsChain.filter_liquid(config) read
# config.yaml options.min_volume/min_open_interest/max_bid_ask_spread_pct
# (0/10/0.40). They agreed only by coincidence, so editing one knob changed
# only half the filtering, and the env-side defaults (volume >= 50) are
# entry-killing on this delayed feed if .env is ever not loaded.
#
# ONE authority: config.yaml options.* is the base; AIT_LIQ_* are EXPLICIT
# per-key overrides on top of it; the hardcoded triple below survives only as
# the last resort when settings cannot be loaded at all. The resolved values
# and which source won each key are logged once per process.
_LIQ_LAST_RESORT = (50, 100, 0.15)
_LIQ_ENV_KEYS = ("AIT_LIQ_MIN_VOLUME", "AIT_LIQ_MIN_OI", "AIT_LIQ_MAX_SPREAD")
_liq_logged: tuple | None = None


def _resolve_liquidity(config: "OptionsConfig | None") -> tuple[int, int, float]:
    """Resolve (min_volume, min_open_interest, max_spread_pct) + log sources."""
    sources = ["code_default"] * 3
    base = list(_LIQ_LAST_RESORT)
    cfg = config
    if cfg is None:
        try:
            from ait.config.settings import load_settings
            cfg = load_settings().options
        except Exception:  # noqa: BLE001 — never let config break chain filtering
            cfg = None
    if cfg is not None:
        try:
            base = [int(cfg.min_volume), int(cfg.min_open_interest),
                    float(cfg.max_bid_ask_spread_pct)]
            sources = ["config.yaml"] * 3
        except Exception:  # noqa: BLE001
            pass

    casts = (int, int, float)
    for i, (key, cast) in enumerate(zip(_LIQ_ENV_KEYS, casts)):
        raw = os.environ.get(key)
        if raw is None:
            continue  # unset -> config.yaml (or last resort) stands
        base[i] = _parse_env_override(raw, cast, base[i])
        sources[i] = f"env:{key}"

    resolved = (int(base[0]), int(base[1]), float(base[2]))
    # Log the winning source per key, but only when it CHANGES — filter_liquid
    # runs per chain per scan and this is a config fact, not an event.
    global _liq_logged
    fingerprint = (resolved, tuple(sources))
    if fingerprint != _liq_logged:
        _liq_logged = fingerprint
        log.info(
            "liquidity_thresholds_resolved",
            min_volume=resolved[0], min_volume_source=sources[0],
            min_open_interest=resolved[1], min_open_interest_source=sources[1],
            max_spread_pct=resolved[2], max_spread_source=sources[2],
        )
    return resolved


@lru_cache(maxsize=1)
def _liquidity_thresholds() -> tuple[int, int, float]:
    """Process-wide resolved thresholds (see _resolve_liquidity).

    Cached because is_liquid runs per contract per scan. Consequence, kept
    deliberately: editing .env or config.yaml requires a process restart —
    call _liquidity_thresholds.cache_clear() to re-resolve (tests do).
    """
    return _resolve_liquidity(None)


@dataclass
class OptionContract:
    """A single option contract with market data and Greeks."""

    symbol: str
    expiry: date
    strike: float
    right: str  # "C" or "P"
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_vol: float
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    # Metadata
    con_id: int = 0  # IBKR contract ID (needed for combo orders)
    # R17: which feed this contract came from. "ibkr" reports open_interest=0
    # to mean UNKNOWN (no real-time OI tick) rather than a real zero; Yahoo
    # ("yahoo_delayed") reports a genuine value, including a genuine zero.
    source: str = "ibkr"

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_pct(self) -> float:
        if self.mid <= 0:
            return 1.0  # Treat as illiquid
        return (self.ask - self.bid) / self.mid

    @property
    def dte(self) -> int:
        return (self.expiry - date.today()).days

    @property
    def is_liquid(self) -> bool:
        # Env-tunable so we can loosen on paper / delayed-data accounts where
        # `volume` is often 0 because it's a live tick stream we don't get.
        min_vol, min_oi, max_spread = _liquidity_thresholds()
        # open_interest==0 means UNKNOWN on the IBKR realtime path (no OI
        # tick) — treating unknown as illiquid rejected every IBKR contract
        # (audit R2/C2). Enforce the OI floor only when OI is reported.
        # R17: that leniency is only valid for IBKR — the Yahoo fallback
        # reports a genuine open_interest, including a genuine zero.
        oi_ok = (self.open_interest <= 0 and self.source == "ibkr") or self.open_interest >= min_oi
        return self.volume >= min_vol and oi_ok and self.spread_pct < max_spread


def atm_iv(chain: "OptionsChain") -> float:
    """ATM implied vol interpolated at the two strikes bracketing spot.

    R12-B (vol agent): the flat-0.20 IV fallback in wing sizing produced
    +/-50% wing-width errors vs the chain's real ATM vol — every live chain
    already carries the data to price the expected move, so extract it here
    once and let strategies read it. Falls back to the median observed IV
    (with a logged tag) when spot isn't bracketed by quoted-IV strikes.
    Returns 0.0 only when the chain has no usable IV at all.
    """
    spot = chain.underlying_price
    # Per-strike IV: average call+put IV when both quote at a strike.
    # R16: `c.implied_vol and ...` is also the None guard (ib_insync maps
    # IB's "not yet computed" sentinels to None); the MIN_PHYSICAL_IV floor
    # rejects Yahoo's ~1e-5 placeholders so they read as missing, not real.
    by_strike: dict[float, list[float]] = {}
    dropped_placeholder = 0
    for grp in (chain.calls, chain.puts):
        for c in grp:
            if not c.implied_vol:
                continue
            if c.implied_vol < MIN_PHYSICAL_IV:
                dropped_placeholder += 1
                continue
            by_strike.setdefault(float(c.strike), []).append(float(c.implied_vol))
    if dropped_placeholder:
        log.debug(
            "atm_iv_placeholder_iv_dropped",
            symbol=chain.symbol,
            contracts=dropped_placeholder,
            floor=MIN_PHYSICAL_IV,
        )
    if not by_strike:
        log.warning("atm_iv_unavailable", symbol=chain.symbol, tag="no_iv_data")
        return 0.0

    ivs = {k: sum(v) / len(v) for k, v in by_strike.items()}
    lower = max((s for s in ivs if s <= spot), default=None)
    upper = min((s for s in ivs if s >= spot), default=None)

    if lower is not None and upper is not None:
        if upper == lower:
            return ivs[lower]
        w = (spot - lower) / (upper - lower)
        return ivs[lower] * (1.0 - w) + ivs[upper] * w

    # Spot outside the quoted-IV strike ladder — median fallback, tagged so
    # entry-quality telemetry can distinguish interpolated from degraded IV.
    med_vals = sorted(ivs.values())
    med = med_vals[len(med_vals) // 2]
    log.warning(
        "atm_iv_fallback", symbol=chain.symbol, tag="median_iv_no_bracket",
        value=round(med, 4), spot=spot,
    )
    return med


def expected_move(chain: "OptionsChain", spot: float | None = None) -> float:
    """1-sigma expected move to the chain expiry: spot * atm_iv * sqrt(dte/365).

    R12-B: the sanity yardstick for short-strike placement (the 07-07 SPY
    condor filled its short call at 0.49 delta — ~0 expected moves from spot
    — and no gate noticed). Uses the chain's own expiry for DTE.
    """
    import math

    if spot is None:
        spot = chain.underlying_price
    iv = chain.atm_iv if getattr(chain, "atm_iv", 0.0) > 0 else atm_iv(chain)
    if iv <= 0 or spot <= 0:
        return 0.0
    dte = max((chain.expiry - date.today()).days, 0)
    return spot * iv * math.sqrt(dte / 365.0)


@dataclass
class OptionsChain:
    """Full options chain for a symbol at a specific expiry."""

    symbol: str
    underlying_price: float
    expiry: date
    calls: list[OptionContract]
    puts: list[OptionContract]
    # R12-B: computed once on build (see __post_init__) so strategies read
    # them off the chain without refetching or re-deriving vol.
    atm_iv: float = 0.0
    expected_move: float = 0.0
    # R16: which feed actually produced this chain ("ibkr" | "yahoo_delayed").
    # The IBKR->Yahoo degradation was completely silent, so entry telemetry
    # attributed delayed-snapshot deltas/IVs to the realtime feed.
    source: str = ""

    def __post_init__(self) -> None:
        # R12-B: attach ATM IV + expected move to every chain at build time.
        # Guarded so explicitly-passed values (e.g. filter_* copies carrying
        # the FULL chain's values) are preserved.
        if not self.atm_iv:
            self.atm_iv = atm_iv(self)
        if not self.expected_move:
            self.expected_move = expected_move(self, self.underlying_price)

    @property
    def dte(self) -> int:
        return (self.expiry - date.today()).days

    def get_atm_strike(self) -> float:
        """Get the at-the-money strike (closest to underlying price)."""
        all_strikes = set()
        for c in self.calls:
            all_strikes.add(c.strike)
        for p in self.puts:
            all_strikes.add(p.strike)
        if not all_strikes:
            return self.underlying_price
        return min(all_strikes, key=lambda s: abs(s - self.underlying_price))

    def filter_by_delta(self, min_delta: float, max_delta: float) -> OptionsChain:
        """Filter contracts by absolute delta range."""
        # R16: ib_insync maps IB's "not yet computed" greek sentinels to
        # None; abs(None) raised TypeError here and aborted the whole
        # symbol's scan. Skip None-delta contracts instead (debug-logged).
        dropped_none = sum(
            1 for grp in (self.calls, self.puts) for c in grp if c.delta is None
        )
        if dropped_none:
            log.debug(
                "filter_by_delta_none_greeks_dropped",
                symbol=self.symbol,
                contracts=dropped_none,
            )
        return OptionsChain(
            symbol=self.symbol,
            underlying_price=self.underlying_price,
            expiry=self.expiry,
            calls=[
                c for c in self.calls
                if c.delta is not None and min_delta <= abs(c.delta) <= max_delta
            ],
            puts=[
                p for p in self.puts
                if p.delta is not None and min_delta <= abs(p.delta) <= max_delta
            ],
            # R12-B: carry the FULL chain's vol stats — recomputing from a
            # delta-filtered (OTM-only) subset would bias ATM IV.
            atm_iv=self.atm_iv,
            expected_move=self.expected_move,
            source=self.source,
        )

    def filter_liquid(self, config: OptionsConfig | None = None) -> OptionsChain:
        """Filter to only liquid contracts.

        R16: this used to read the passed OptionsConfig directly while
        OptionContract.is_liquid read env AIT_LIQ_* with different defaults —
        two filters, two sources. Both now resolve through
        _resolve_liquidity(): config.yaml is the base, AIT_LIQ_* override it.
        Passing config keeps the caller's base explicit (the orchestrator
        passes settings.options); None resolves the loaded settings.
        """
        min_vol, min_oi, max_spread = _resolve_liquidity(config)
        return OptionsChain(
            symbol=self.symbol,
            underlying_price=self.underlying_price,
            expiry=self.expiry,
            # R17: the OI<=0-means-unknown leniency only holds for IBKR;
            # each contract's own `source` decides, not the chain-level one.
            calls=[
                c
                for c in self.calls
                if c.volume >= min_vol
                and ((c.open_interest <= 0 and c.source == "ibkr") or c.open_interest >= min_oi)
                and c.spread_pct <= max_spread
            ],
            puts=[
                p
                for p in self.puts
                if p.volume >= min_vol
                and ((p.open_interest <= 0 and p.source == "ibkr") or p.open_interest >= min_oi)
                and p.spread_pct <= max_spread
            ],
            # R12-B: carry the full chain's vol stats (see filter_by_delta).
            atm_iv=self.atm_iv,
            expected_move=self.expected_move,
            source=self.source,
        )


class OptionsChainService:
    """Fetches and processes options chains from IBKR and Yahoo Finance."""

    def __init__(
        self,
        ibkr_client: IBKRClient,
        market_data: MarketDataService,
        config: OptionsConfig,
    ) -> None:
        self._ibkr = ibkr_client
        self._market_data = market_data
        self._config = config
        self._cache = TTLCache(default_ttl=120)  # 2 min cache for chains

    async def get_chain(
        self,
        symbol: str,
        min_dte: int | None = None,
        max_dte: int | None = None,
    ) -> list[OptionsChain]:
        """Get options chains for a symbol within DTE range.

        Returns one OptionsChain per expiry date that falls within range.
        """
        min_dte = min_dte or self._config.dte_range[0]
        max_dte = max_dte or self._config.dte_range[1]

        cache_key = f"chain:{symbol}:{min_dte}:{max_dte}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Get underlying price
        price = await self._market_data.get_current_price(symbol)
        if price is None:
            log.warning("cannot_get_chain", symbol=symbol, reason="no underlying price")
            return []

        # Try IBKR first
        chains = await self._get_ibkr_chain(symbol, price, min_dte, max_dte)
        source = "ibkr"

        # Fallback to Yahoo
        if not chains:
            chains = await self._get_yahoo_chain(symbol, price, min_dte, max_dte)
            source = "yahoo_delayed"

        # R16: the degradation to Yahoo was COMPLETELY SILENT — no log line
        # anywhere recorded which feed a scan's strikes, deltas and IVs came
        # from, so a Gateway/market-data outage looked identical to a healthy
        # scan in every downstream telemetry (condor_entry_quality, atm_iv,
        # expected_move). Emit the source actually used on every resolution
        # (cache miss, i.e. at most once per symbol/DTE-band per 120s cache
        # window ~ once per scan), WARNING when degraded.
        self._log_chain_source(symbol, source, chains)

        if chains:
            # Calculate Greeks for all contracts
            for chain in chains:
                chain.source = source
                self._calculate_greeks(chain)

            self._cache.set(cache_key, chains, ttl=120)

        return chains

    def _log_chain_source(self, symbol: str, source: str,
                          chains: list[OptionsChain]) -> None:
        """R16: one line per symbol per scan naming the feed that served it."""
        if not chains:
            log.warning(
                "chain_source_unavailable",
                symbol=symbol,
                source="none",
                reason="IBKR returned nothing and the Yahoo fallback was empty",
            )
            return
        if source == "ibkr":
            log.info("chain_source", symbol=symbol, source=source,
                     expiries=len(chains))
        else:
            log.warning(
                "chain_source_degraded",
                symbol=symbol,
                source=source,
                expiries=len(chains),
                note="IBKR chain empty/unavailable — strikes, IV and greeks "
                     "for this scan come from DELAYED Yahoo snapshots",
            )

    async def get_chain_for_expiry(
        self, symbol: str, expiry: date
    ) -> OptionsChain | None:
        """Get a specific expiry's options chain."""
        chains = await self.get_chain(
            symbol,
            min_dte=(expiry - date.today()).days - 1,
            max_dte=(expiry - date.today()).days + 1,
        )
        for chain in chains:
            if chain.expiry == expiry:
                return chain
        return chains[0] if chains else None

    # --- Private methods ---

    async def _get_ibkr_chain(
        self, symbol: str, price: float, min_dte: int, max_dte: int
    ) -> list[OptionsChain]:
        """Fetch options chain from IBKR."""
        if not self._ibkr.connected:
            return []

        try:
            from ib_insync import Stock

            contract = Stock(symbol, "SMART", "USD")
            qualified = await self._ibkr.qualify_contract(contract)
            if not qualified:
                return []

            # Get available option chains (use async version to avoid blocking event loop)
            chains_data = await self._ibkr.ib.reqSecDefOptParamsAsync(
                qualified.symbol, "", qualified.secType, qualified.conId
            )

            if not chains_data:
                return []

            # Find SMART exchange chain for the STANDARD trading class.
            # R16: reqSecDefOptParams lists the post-split adjusted class
            # ('2IWM'/'2SPY'/'2QQQ') FIRST on SMART; taking the first SMART
            # def served a 1-3 strike garbage mini-chain (IWM had no usable
            # chain for ~3 weeks, SPY scans requested non-existent strikes).
            # Prefer exchange=='SMART' AND tradingClass==symbol; fall back to
            # any SMART def only when no exact class match exists.
            chain_def = None
            for cd in chains_data:
                if cd.exchange == "SMART" and getattr(cd, "tradingClass", "") == symbol:
                    chain_def = cd
                    break
            if chain_def is None:
                for cd in chains_data:
                    if cd.exchange == "SMART":
                        chain_def = cd
                        break
            if chain_def is None:
                chain_def = chains_data[0]
            if getattr(chain_def, "tradingClass", symbol) != symbol:
                log.warning(
                    "chain_def_nonstandard_class",
                    symbol=symbol,
                    trading_class=getattr(chain_def, "tradingClass", ""),
                    reason="no SMART def with tradingClass==symbol",
                )

            # Filter expiries by DTE range
            today = date.today()
            target_expiries = []
            for exp_str in sorted(chain_def.expirations):
                exp_date = datetime.strptime(exp_str, "%Y%m%d").date()
                dte = (exp_date - today).days
                if min_dte <= dte <= max_dte:
                    target_expiries.append(exp_date)

            if not target_expiries:
                return []

            # Filter strikes around ATM (±20% of price)
            # Skip non-standard (adjusted) strikes — e.g. .5 increments from stock splits
            strike_range = price * 0.20
            target_strikes = [
                s for s in chain_def.strikes
                if abs(s - price) <= strike_range and (s * 2) == int(s * 2)  # allow $0.50 strikes; whole-dollar-only deleted ATM strikes on sub-$25 names (deep-audit DATA-M7)
            ]

            # Fetch option contracts for each expiry
            chains = []
            for exp_date in target_expiries[:3]:  # Limit to 3 closest expiries
                chain = await self._fetch_ibkr_expiry(
                    symbol, qualified, exp_date, target_strikes, price
                )
                if chain:
                    chains.append(chain)

            return chains

        except Exception as e:
            log.warning("ibkr_chain_failed", symbol=symbol, error=str(e))
            return []

    async def _fetch_ibkr_expiry(
        self,
        symbol: str,
        underlying,
        expiry: date,
        strikes: list[float],
        price: float,
    ) -> OptionsChain | None:
        """Fetch a single expiry's chain from IBKR."""
        from ib_insync import Option

        calls = []
        puts = []
        exp_str = expiry.strftime("%Y%m%d")

        # Build option contracts for qualification.
        # R16: pin tradingClass=symbol — strikes that exist in BOTH the
        # standard and the adjusted class ('2IWM' etc.) came back 'Ambiguous
        # contract' from qualifyContractsAsync and were silently dropped.
        option_contracts = []
        for strike in strikes:
            for right in ("C", "P"):
                opt = Option(symbol, exp_str, strike, right, "SMART", tradingClass=symbol)
                option_contracts.append(opt)

        def _num(val) -> float:
            # R16: ib_insync maps IB's -1/-2 "not yet computed" sentinels to
            # None in modelGreeks — storing raw None crashed downstream math
            # (TypeError in _calculate_greeks / filter_by_delta), aborting
            # the whole symbol's scan. Coerce None to 0.0 = "missing".
            return float(val) if val is not None else 0.0

        none_greek_contracts = 0

        # Qualify in batches to avoid timeout
        batch_size = 50
        # R16: the old snapshot pattern (50 concurrent reqMktData, 0.5s wait,
        # no cancel) meant delayed-snapshot modelGreeks essentially never
        # arrived (chain-wide IV/delta = 0) and un-cancelled subs stacked
        # toward the market-data line cap while their ~11s snapshotEnd
        # windows overlapped (10091/10197 error floods). Request fewer
        # concurrently, wait longer, and always cancel in a finally.
        md_batch_size = 25
        md_wait_s = 2.0
        for i in range(0, len(option_contracts), batch_size):
            batch = option_contracts[i : i + batch_size]
            try:
                qualified = await self._ibkr.ib.qualifyContractsAsync(*batch)
                valid = [q for q in qualified if q.conId != 0]

                for j in range(0, len(valid), md_batch_size):
                    md_batch = valid[j : j + md_batch_size]
                    requested: list = []
                    try:
                        for q in md_batch:
                            # Request market data (snapshot)
                            self._ibkr.ib.reqMktData(q, "", True, False)
                            requested.append(q)

                        await asyncio.sleep(md_wait_s)

                        for q in md_batch:
                            ticker = self._ibkr.ib.ticker(q)
                            if not ticker:
                                continue

                            mg = ticker.modelGreeks
                            if mg and any(
                                v is None
                                for v in (mg.impliedVol, mg.delta, mg.gamma, mg.theta, mg.vega)
                            ):
                                none_greek_contracts += 1

                            contract = OptionContract(
                                symbol=symbol,
                                expiry=expiry,
                                strike=q.strike,
                                right=q.right,
                                bid=ticker.bid if ticker.bid and ticker.bid > 0 else 0.0,
                                ask=ticker.ask if ticker.ask and ticker.ask > 0 else 0.0,
                                last=ticker.last if ticker.last and ticker.last > 0 else 0.0,
                                volume=(int(ticker.volume) if (ticker.volume and ticker.volume == ticker.volume and ticker.volume > 0) else 0),  # NaN is truthy; int(NaN) killed the whole 50-contract batch (deep-audit DATA-M6)
                                open_interest=0,  # IBKR doesn't provide OI in real-time
                                implied_vol=_num(mg.impliedVol) if mg else 0.0,
                                delta=_num(mg.delta) if mg else 0.0,
                                gamma=_num(mg.gamma) if mg else 0.0,
                                theta=_num(mg.theta) if mg else 0.0,
                                vega=_num(mg.vega) if mg else 0.0,
                                con_id=q.conId,
                                source="ibkr",  # open_interest=0 here means UNKNOWN, not zero
                            )

                            if q.right == "C":
                                calls.append(contract)
                            else:
                                puts.append(contract)
                    finally:
                        # R16: explicit cancel for EVERY request. Snapshot
                        # subs auto-cancel only at snapshotEnd (~11s); until
                        # then each holds a market-data line and error-floods
                        # on unsubscribable contracts.
                        for q in requested:
                            try:
                                self._ibkr.ib.cancelMktData(q)
                            except Exception:
                                pass

            except Exception as e:
                log.debug("ibkr_batch_failed", symbol=symbol, batch=i, error=str(e))

        if none_greek_contracts:
            log.debug(
                "ibkr_none_greeks_coerced",
                symbol=symbol,
                expiry=exp_str,
                contracts=none_greek_contracts,
            )

        if not calls and not puts:
            return None

        return OptionsChain(
            symbol=symbol,
            underlying_price=price,
            expiry=expiry,
            calls=sorted(calls, key=lambda c: c.strike),
            puts=sorted(puts, key=lambda p: p.strike),
        )

    def _fetch_yahoo_chain_sync(
        self, symbol: str, price: float, min_dte: int, max_dte: int
    ) -> list[OptionsChain]:
        """Synchronous Yahoo Finance chain fetch (runs in executor)."""
        import math

        import yfinance as yf

        ticker = yf.Ticker(symbol)
        expiry_strings = ticker.options  # List of expiry date strings

        if not expiry_strings:
            return []

        today = date.today()
        chains = []

        def _safe_int(val, default=0):
            try:
                v = float(val) if val is not None else default
                return default if math.isnan(v) else int(v)
            except (TypeError, ValueError):
                return default

        def _safe_float(val, default=0.0):
            try:
                v = float(val) if val is not None else default
                return default if math.isnan(v) else v
            except (TypeError, ValueError):
                return default

        # R16: Yahoo stamps placeholder impliedVolatility (~1e-5) on rows
        # without a computed IV (routine off-hours/pre-open). They passed the
        # old `> 0` filters and polluted ATM-IV / median-IV calcs (50x-small
        # expected_move). Treat sub-floor IVs as missing and count the drops.
        placeholder_ivs = 0

        def _physical_iv(val) -> float:
            nonlocal placeholder_ivs
            iv = _safe_float(val)
            if 0 < iv < MIN_PHYSICAL_IV:
                placeholder_ivs += 1
                return 0.0
            return iv

        for exp_str in expiry_strings:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if not (min_dte <= dte <= max_dte):
                continue

            try:
                opt_chain = ticker.option_chain(exp_str)
            except Exception:
                continue

            calls = []
            for _, row in opt_chain.calls.iterrows():
                strike = _safe_float(row.get("strike"))
                if not strike or abs(strike - price) > price * 0.20:
                    continue
                calls.append(
                    OptionContract(
                        symbol=symbol,
                        expiry=exp_date,
                        strike=strike,
                        right="C",
                        bid=_safe_float(row.get("bid")),
                        ask=_safe_float(row.get("ask")),
                        last=_safe_float(row.get("lastPrice")),
                        volume=_safe_int(row.get("volume")),
                        open_interest=_safe_int(row.get("openInterest")),
                        implied_vol=_physical_iv(row.get("impliedVolatility")),
                        source="yahoo_delayed",
                    )
                )

            puts = []
            for _, row in opt_chain.puts.iterrows():
                strike = _safe_float(row.get("strike"))
                if not strike or abs(strike - price) > price * 0.20:
                    continue
                puts.append(
                    OptionContract(
                        symbol=symbol,
                        expiry=exp_date,
                        strike=strike,
                        right="P",
                        bid=_safe_float(row.get("bid")),
                        ask=_safe_float(row.get("ask")),
                        last=_safe_float(row.get("lastPrice")),
                        volume=_safe_int(row.get("volume")),
                        open_interest=_safe_int(row.get("openInterest")),
                        implied_vol=_physical_iv(row.get("impliedVolatility")),
                        source="yahoo_delayed",
                    )
                )

            if calls or puts:
                chains.append(
                    OptionsChain(
                        symbol=symbol,
                        underlying_price=price,
                        expiry=exp_date,
                        calls=sorted(calls, key=lambda c: c.strike),
                        puts=sorted(puts, key=lambda p: p.strike),
                    )
                )

            if len(chains) >= 3:
                break

        if placeholder_ivs:
            log.debug(
                "yahoo_placeholder_iv_dropped",
                symbol=symbol,
                contracts=placeholder_ivs,
                floor=MIN_PHYSICAL_IV,
            )

        return chains

    async def _get_yahoo_chain(
        self, symbol: str, price: float, min_dte: int, max_dte: int
    ) -> list[OptionsChain]:
        """Fetch options chain from Yahoo Finance as fallback."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._fetch_yahoo_chain_sync, symbol, price, min_dte, max_dte
            )
        except Exception as e:
            log.warning("yahoo_chain_failed", symbol=symbol, error=str(e))
            return []

    def _calculate_greeks(self, chain: OptionsChain) -> None:
        """Calculate Greeks using Black-Scholes for contracts missing them."""
        try:
            from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
            from py_vollib.black_scholes.implied_volatility import implied_volatility
        except ImportError:
            log.debug("py_vollib_not_available", action="skipping greeks calculation")
            return

        S = chain.underlying_price
        r = 0.05  # Risk-free rate approximation
        t = max(chain.dte / 365.0, 0.001)  # Time to expiry in years

        # A6 (deep-audit DATA-M6 residual): when a contract lacks IV, use the
        # chain's MEDIAN observed IV instead of a flat 30% — a wrong sigma
        # shifts the computed delta, and strategies pick strikes BY delta.
        # R16: `c.implied_vol and ...` is None-safe, and the MIN_PHYSICAL_IV
        # floor keeps Yahoo's ~1e-5 placeholders out of the median.
        _ivs = sorted(c.implied_vol for grp in (chain.calls, chain.puts)
                      for c in grp
                      if c.implied_vol and c.implied_vol >= MIN_PHYSICAL_IV)
        _chain_iv = _ivs[len(_ivs) // 2] if _ivs else 0.30

        none_greeks = 0
        for contracts in (chain.calls, chain.puts):
            for c in contracts:
                # R16: ib_insync maps IB's "not yet computed" sentinels to
                # None — a raw None here raised TypeError (the sigma compare
                # sat OUTSIDE the try below) and aborted the whole symbol's
                # scan. Coerce to 0.0 = missing so the BS backfill runs.
                if any(v is None for v in (c.implied_vol, c.delta, c.gamma, c.theta, c.vega)):
                    none_greeks += 1
                    c.implied_vol = c.implied_vol or 0.0
                    c.delta = c.delta or 0.0
                    c.gamma = c.gamma or 0.0
                    c.theta = c.theta or 0.0
                    c.vega = c.vega or 0.0
                if c.delta != 0:
                    continue  # Already has Greeks (from IBKR)

                flag = "c" if c.right == "C" else "p"
                # R16: placeholder floor — a degenerate ~1e-5 sigma must never
                # feed the BS backfill (it pins deltas to 0/±1).
                sigma = (
                    c.implied_vol
                    if c.implied_vol and c.implied_vol >= MIN_PHYSICAL_IV
                    else _chain_iv
                )

                try:
                    c.delta = delta(flag, S, c.strike, t, r, sigma)
                    c.gamma = gamma(flag, S, c.strike, t, r, sigma)
                    c.theta = theta(flag, S, c.strike, t, r, sigma) / 365.0  # Daily theta
                    c.vega = vega(flag, S, c.strike, t, r, sigma) / 100.0  # Per 1% vol move
                except Exception:
                    pass  # Leave as 0 if calculation fails

        if none_greeks:
            log.debug(
                "greeks_none_coerced", symbol=chain.symbol, contracts=none_greeks
            )

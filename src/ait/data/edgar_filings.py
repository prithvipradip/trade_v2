"""SEC EDGAR 8-K filing monitor.

8-K filings disclose material events: M&A, executive changes, earnings,
restructuring, regulatory issues. They MOVE markets. We poll EDGAR's
free RSS feed for our universe and trigger position-flatten when a
filing arrives during market hours.

EDGAR API docs: https://www.sec.gov/edgar/sec-api-documentation
Rate limit: 10 req/sec. We poll every 60s during market hours, well within limits.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

import aiohttp

from ait.data.cache import TTLCache
from ait.utils.logging import get_logger

log = get_logger("data.edgar")

# IBKR ticker → SEC CIK (Central Index Key) mapping for our universe.
# Pre-resolved to avoid lookup overhead. CIKs from sec.gov/cgi-bin/browse-edgar.
TICKER_TO_CIK = {
    "SPY":   "0000884394",  # SPDR S&P 500 ETF
    "QQQ":   "0001067839",  # Invesco QQQ
    "IWM":   "0001100663",  # iShares Russell 2000
    "DIA":   "0001100966",  # SPDR Dow
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "NVDA":  "0001045810",
    "TSLA":  "0001318605",
    "AMD":   "0000002488",
    "AMZN":  "0001018724",
    "META":  "0001326801",
    "GOOGL": "0001652044",
    "SOFI":  "0001818874",
    "PLTR":  "0001321655",
}

# Friendly user-agent required by EDGAR
# R5 audit: was a placeholder example.com address — SEC blocks non-compliant
# UAs with persistent 403s, silently disabling the 8-K safety flatten.
USER_AGENT = "AIT v2 Trading Bot prithvi.pradip@cluedigital.com"


def _filing_time(filing_date: str, acceptance: str) -> datetime:
    """When the filing became public, UTC: acceptanceDateTime when present, else
    the END of filingDate (a bare date must not fall out of the window early).

    KNOWN QUIRK (measured 2026-10-17, research_calendars.item_202_filings): for some filings
    EDGAR's acceptanceDateTime is the ET time plus twice the UTC offset, still labelled Z, i.e.
    4-5 hours LATER than the true acceptance. It only ever lengthens the look-back window (a
    filing stays "recent" 4-5 hours longer), never drops one; a filing already acted on is not
    acted on again because the handled accessions persist (TradingOrchestrator._build_edgar_monitor)."""
    if acceptance:
        return datetime.fromisoformat(acceptance.replace("Z", "+00:00")).astimezone(timezone.utc)
    d = datetime.strptime(filing_date, "%Y-%m-%d").date()
    return datetime.combine(d, time.max, tzinfo=timezone.utc)


@dataclass
class Filing:
    """A single SEC filing."""
    symbol: str
    form_type: str          # "8-K", "10-Q", "10-K", "4", etc.
    filing_date: datetime
    accession_number: str
    primary_document: str   # URL to filing
    description: str = ""

    @property
    def is_material_event(self) -> bool:
        """8-K filings are material events that typically move markets."""
        return self.form_type == "8-K"


class EDGARMonitor:
    """Polls SEC EDGAR for new filings on tracked symbols."""

    def __init__(self, tracked_symbols: list[str] | None = None, seen: "list[str] | None" = None) -> None:
        self._tracked = tracked_symbols or list(TICKER_TO_CIK.keys())
        # Accessions already emitted (W106: seeded from the state database, so a restart inside the
        # look-back window does not emit, and flatten on, the same 8-K again). A dict keeps the order
        # handled, so the persisted record can keep the newest.
        self._seen_filings: dict[str, None] = dict.fromkeys(seen or [])
        self._filings_cache = TTLCache(default_ttl=300, max_size=1000)
        self._last_check: datetime | None = None

    def mark_seen(self, accession: str) -> None:
        self._seen_filings[accession] = None

    def seen_accessions(self) -> list[str]:
        """Accessions already emitted, oldest first."""
        return list(self._seen_filings)

    async def fetch_recent_filings(
        self, symbol: str, max_age_hours: int = 24
    ) -> list[Filing]:
        """Fetch recent filings for a symbol from EDGAR.

        Returns filings newer than max_age_hours. Empty list on error.
        """
        cik = TICKER_TO_CIK.get(symbol)
        if not cik:
            log.debug("edgar_no_cik", symbol=symbol)
            return []

        cached = self._filings_cache.get(symbol)
        if cached is not None:
            return cached

        # EDGAR submissions API — JSON endpoint, free, no key needed
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers={"User-Agent": USER_AGENT},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        log.warning("edgar_fetch_failed", symbol=symbol, status=resp.status)
                        return []
                    data = await resp.json()
        except Exception as e:
            log.warning("edgar_fetch_error", symbol=symbol, error=str(e))
            return []

        # Parse the recent filings array
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accepted = recent.get("acceptanceDateTime", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        # W106: the window is measured on the acceptance time (UTC). filingDate is a
        # bare date: read as midnight against a host-naive now - 24h it dropped a
        # filing accepted late yesterday before the next morning's check.
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        filings = []
        for i in range(min(len(forms), 20)):  # last 20 filings only
            try:
                filing_date = _filing_time(dates[i], accepted[i] if i < len(accepted) else "")
                if filing_date < cutoff:
                    break  # filings are sorted desc; older = stop
                filings.append(Filing(
                    symbol=symbol,
                    form_type=forms[i],
                    filing_date=filing_date,
                    accession_number=accessions[i],
                    primary_document=primary_docs[i],
                ))
            except (ValueError, IndexError):
                continue

        self._filings_cache.set(symbol, filings)
        return filings

    async def check_for_material_events(
        self, symbols: list[str] | None = None
    ) -> list[Filing]:
        """Check for new 8-K filings on tracked symbols.

        Returns only NEW (unseen) 8-K filings. Subsequent calls won't
        re-emit the same filing.
        """
        symbols = symbols or self._tracked
        new_material = []

        # Fetch in parallel
        results = await asyncio.gather(
            *[self.fetch_recent_filings(s) for s in symbols],
            return_exceptions=True,
        )

        for symbol, filings in zip(symbols, results):
            if isinstance(filings, Exception):
                continue
            for filing in filings:
                if not filing.is_material_event:
                    continue
                if filing.accession_number in self._seen_filings:
                    continue
                self.mark_seen(filing.accession_number)
                new_material.append(filing)
                log.info(
                    "edgar_8k_detected",
                    symbol=filing.symbol,
                    filing_date=filing.filing_date.isoformat(),
                    accession=filing.accession_number,
                )

        self._last_check = datetime.now()
        return new_material

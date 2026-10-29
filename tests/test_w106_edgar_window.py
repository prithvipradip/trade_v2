"""W106 item 7: the 8-K look-back window is measured on the filing's acceptance time.

`filingDate` is a bare date. Parsed as midnight and compared with a host-naive
`now - 24h`, a filing accepted late yesterday (after the close) was dropped by
the 24-hour window the next morning, so the 8-K flatten never saw it.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import ait.data.edgar_filings as ef


class _Resp:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    def __init__(self, payload):
        self._payload = payload

    def get(self, *a, **k):
        return _Resp(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _fetch(monkeypatch, filings):
    recent = {"form": [f[0] for f in filings], "filingDate": [f[1] for f in filings],
              "acceptanceDateTime": [f[2] for f in filings],
              "accessionNumber": [f"acc-{i}" for i in range(len(filings))],
              "primaryDocument": ["d.htm"] * len(filings)}
    monkeypatch.setattr(ef.aiohttp, "ClientSession",
                        lambda *a, **k: _Session({"filings": {"recent": recent}}))
    return asyncio.run(ef.EDGARMonitor(["SPY"]).fetch_recent_filings("SPY", max_age_hours=24))


def _row(age_hours, form="8-K", with_acceptance=True):
    t = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    acc = t.strftime("%Y-%m-%dT%H:%M:%S.000Z") if with_acceptance else ""
    return (form, t.strftime("%Y-%m-%d"), acc)


def test_a_filing_accepted_23_hours_ago_is_inside_a_24_hour_window(monkeypatch):
    got = _fetch(monkeypatch, [_row(23.5)])
    assert [f.accession_number for f in got] == ["acc-0"]
    assert got[0].filing_date.tzinfo is not None


def test_a_filing_accepted_30_hours_ago_is_outside(monkeypatch):
    assert _fetch(monkeypatch, [_row(1), _row(30)]) == _fetch(monkeypatch, [_row(1)])
    assert len(_fetch(monkeypatch, [_row(1), _row(30)])) == 1


def test_without_acceptance_time_the_filing_date_is_used(monkeypatch):
    """Fallback: a bare filingDate counts from the end of that day (UTC), so a filing
    dated today is always inside and one dated three days ago is outside."""
    assert len(_fetch(monkeypatch, [_row(0, with_acceptance=False)])) == 1
    assert _fetch(monkeypatch, [_row(72, with_acceptance=False)]) == []

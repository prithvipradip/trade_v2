"""W106 item 9: a contract that came back empty is not fetched again on every run.

A contract with no bars (never traded) or no storable bar (no underlying close)
stored nothing, so `option_ticker_has_data` stayed False and every plain run spent
another rate-limited call on it (~2,000 SPY contracts, ~7 h per restart at 12.5 s).
A fetch made after the contract expired is now recorded and is final; one made
before expiry is not (more bars can still come); `--retry-empty` overrides.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from tests.test_backfill_option_history import _FakeClient, _closes, _contract, _temp_store, m

_WINDOW = (date(2024, 9, 1), date(2024, 10, 18))


def _run(store, client, contract=None, **kw):
    return m.process_contract(client, store, "QQQ", contract or _contract(),
                              _closes({"2024-09-01": 480.0}), *_WINDOW, 0.05, **kw)


def test_an_expired_empty_contract_is_fetched_once():
    store = _temp_store()
    first = _FakeClient(bars=[])
    assert _run(store, first) == 0 and first.fetch_calls == 1
    again = _FakeClient(raise_on_fetch=True)            # a second network call fails the test
    assert _run(store, again) == 0 and again.fetch_calls == 0


def test_a_bar_without_an_underlying_close_counts_as_empty_too():
    store = _temp_store()
    t_ms = int(pd.Timestamp(date(2024, 9, 18), tz="UTC").timestamp() * 1000)
    bars = [{"o": 5.0, "h": 5.5, "l": 4.8, "c": 5.2, "v": 1000, "vw": 5.1, "n": 200, "t": t_ms}]
    assert _run(store, _FakeClient(bars=bars)) == 0
    assert _run(store, _FakeClient(raise_on_fetch=True)) == 0


def test_retry_empty_fetches_again():
    store = _temp_store()
    _run(store, _FakeClient(bars=[]))
    again = _FakeClient(bars=[])
    _run(store, again, retry_empty=True)
    assert again.fetch_calls == 1


def test_an_empty_fetch_before_expiry_is_not_final():
    """A contract still alive when fetched can trade later: never settled."""
    store = _temp_store()
    alive = _contract(ticker="O:QQQ991217C00480000", expiry="9999-12-17")
    _run(store, _FakeClient(bars=[]), contract=alive)
    again = _FakeClient(bars=[])
    _run(store, again, contract=alive)
    assert again.fetch_calls == 1

"""W106 item 9: a contract that came back empty is not fetched again on every run.

A contract with no bars (never traded) stored nothing, so `option_ticker_has_data` stayed False
and every plain run spent another rate-limited call on it (~2,000 SPY contracts, ~7 h per restart
at 12.5 s). A fetch made after the contract expired that returned no bars is now recorded and is
final; one made before expiry is not (more bars can still come); a fetch whose bars were all
dropped for want of an underlying close is not final either (it is asked again once the closes
exist); `--retry-empty` overrides.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tests.test_backfill_option_history import (_FakeClient, _FakeResponse, _FlakySession, _closes, _contract,
                                                _temp_store, m)

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


def test_a_bar_without_an_underlying_close_is_asked_again():
    """Bars came back but none could be stored (no underlying close that day): not final, so the
    contract is fetched again once daily_prices has the close."""
    store = _temp_store()
    t_ms = int(pd.Timestamp(date(2024, 9, 18), tz="UTC").timestamp() * 1000)
    bars = [{"o": 5.0, "h": 5.5, "l": 4.8, "c": 5.2, "v": 1000, "vw": 5.1, "n": 200, "t": t_ms}]
    assert _run(store, _FakeClient(bars=bars)) == 0
    again = _FakeClient(bars=bars)
    _run(store, again)
    assert again.fetch_calls == 1


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


# --------------------------------------------------------------------------- review follow-ups (2026-10-29)

class _ErrorReplyClient(_FakeClient):
    """A provider error reply (401 / NOT_AUTHORIZED / unparseable): no bars, and not a real answer."""

    def get_daily_bars(self, option_ticker, from_date, to_date):
        self.last_answer_ok = False
        return super().get_daily_bars(option_ticker, from_date, to_date)


def test_an_error_reply_is_not_recorded_and_is_asked_again():
    store = _temp_store()
    _run(store, _ErrorReplyClient(bars=[]))
    again = _FakeClient(bars=[])
    _run(store, again)
    assert again.fetch_calls == 1


@pytest.mark.parametrize("payload,ok", [
    ({"status": "OK", "resultsCount": 0}, True),
    ({"status": "DELAYED", "results": []}, True),
    ({"status": "NOT_AUTHORIZED", "message": "x"}, False),
    ({"status": "ERROR"}, False),
])
def test_the_real_client_flags_only_real_answers(monkeypatch, payload, ok):
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    client = m.PolygonClient("fake-key", pause_secs=0,
                             session=_FlakySession(fail_times=0, response=_FakeResponse(json_data=payload)))
    client.get_daily_bars("O:QQQ241018C00480000", date(2024, 9, 1), date(2024, 10, 18))
    assert client.last_answer_ok is ok


def test_a_fetch_over_part_of_the_life_is_not_final():
    store = _temp_store()
    late_start = (date(2024, 10, 1), date(2024, 10, 18))          # a later --start-date
    m.process_contract(_FakeClient(bars=[]), store, "QQQ", _contract(), _closes({"2024-09-01": 480.0}),
                       *late_start, 0.05)
    again = _FakeClient(bars=[])
    _run(store, again)                                            # the full window starts earlier
    assert again.fetch_calls == 1

    store = _temp_store()
    short_end = (date(2024, 9, 1), date(2024, 10, 10))            # --end-date before the expiry
    m.process_contract(_FakeClient(bars=[]), store, "QQQ", _contract(), _closes({"2024-09-01": 480.0}),
                       *short_end, 0.05)
    again = _FakeClient(bars=[])
    _run(store, again)
    assert again.fetch_calls == 1

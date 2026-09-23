"""W52a: the earnings and dividend calendars' last-known-good survives a
restart.

W44 and W51 gave both calendars a last-known-good (LKG) answer so a Yahoo
outage falls back to the last real fetch instead of refusing entries. The
LKG lived in a dict on the instance, so every restart began with none:
the first ticks after a cold start fetched in a background thread and,
until it landed, every symbol was "unknown" -> refuse -> page. 09-22 09:35
was exactly that page, 0.5 s after the fetch was kicked, on a symbol whose
answer had been "confirmed no earnings" for weeks.

Both calendars now take an optional ``persist`` — anything with the
StateManager's ``get_state(key, default)`` / ``set_state(key, value)`` —
and write each real answer as ``earnings_lkg_<SYMBOL>`` /
``dividend_lkg_<SYMBOL>`` JSON with its fetch time. A cold instance
hydrates from it on the first miss, under the same ``lkg_max_age_hours``
freshness rule (a persisted answer past its age is not served). Nothing
about the TTL cache, the fetch, or the fail-direction policy changes.
"""
from __future__ import annotations

import json
import time
from datetime import date, timedelta

import pytest

from ait.bot.state import StateManager
from ait.data.dividends import DividendCalendar, DividendInfo
from ait.data.earnings import EarningsCalendar, EarningsInfo
from ait.utils.time import today_et


@pytest.fixture
def state(tmp_path):
    return StateManager(tmp_path / "state.db")


def _fail_earnings(symbol):
    return EarningsInfo(symbol=symbol, next_earnings_date=None, data_available=False)


def _fail_dividend(symbol):
    return DividendInfo(symbol=symbol, next_ex_div_date=None,
                        expected_amount=None, data_available=False)


class TestEarningsLkgIsPersisted:
    def test_a_real_answer_is_written_to_bot_state(self, state, monkeypatch):
        cal = EarningsCalendar(persist=state)
        when = today_et() + timedelta(days=12)
        monkeypatch.setattr(cal, "_fetch_earnings", lambda s: EarningsInfo(
            symbol=s, next_earnings_date=when, is_confirmed=True))
        cal.get_next_earnings("NVDA")

        raw = state.get_state("earnings_lkg_NVDA", "")
        assert raw, "no persisted LKG row"
        row = json.loads(raw)
        assert row["next_earnings_date"] == when.isoformat()
        assert row["is_confirmed"] is True
        assert abs(row["fetched_at"] - time.time()) < 5

    def test_a_failed_fetch_is_never_persisted(self, state, monkeypatch):
        cal = EarningsCalendar(persist=state)
        monkeypatch.setattr(cal, "_fetch_earnings", _fail_earnings)
        cal.get_next_earnings("SPY")
        assert state.get_state("earnings_lkg_SPY", "") == ""

    def test_a_cold_instance_serves_the_persisted_answer_on_a_failure(self, state, monkeypatch):
        """The 09-22 09:35 shape: the previous process knew SPY had no
        earnings; the new process's first fetch fails. Serve the LKG."""
        warm = EarningsCalendar(persist=state)
        monkeypatch.setattr(warm, "_fetch_earnings", lambda s: EarningsInfo(
            symbol=s, next_earnings_date=None))
        assert warm.get_next_earnings("SPY").data_available is True

        cold = EarningsCalendar(persist=state)
        monkeypatch.setattr(cold, "_fetch_earnings", _fail_earnings)
        info = cold.get_next_earnings("SPY")
        assert info.data_available is True, "the persisted LKG was not served"
        assert info.next_earnings_date is None
        assert cold.is_near_earnings("SPY") is False

    def test_a_persisted_date_still_blocks_after_a_restart(self, state, monkeypatch):
        when = today_et() + timedelta(days=1)
        warm = EarningsCalendar(persist=state)
        monkeypatch.setattr(warm, "_fetch_earnings", lambda s: EarningsInfo(
            symbol=s, next_earnings_date=when, is_confirmed=True))
        warm.get_next_earnings("NVDA")

        cold = EarningsCalendar(persist=state)
        monkeypatch.setattr(cold, "_fetch_earnings", _fail_earnings)
        assert cold.is_near_earnings("NVDA") is True

    def test_a_persisted_answer_past_max_age_is_not_served(self, state, monkeypatch):
        state.set_state("earnings_lkg_SPY", json.dumps({
            "symbol": "SPY", "next_earnings_date": None, "is_confirmed": False,
            "fetched_at": time.time() - 3700}))
        cold = EarningsCalendar(persist=state, lkg_max_age_hours=1)
        monkeypatch.setattr(cold, "_fetch_earnings", _fail_earnings)
        assert cold.get_next_earnings("SPY").data_available is False
        assert cold.is_near_earnings("SPY") is True  # refuse, as W44 says

    def test_a_corrupt_row_is_ignored(self, state, monkeypatch):
        state.set_state("earnings_lkg_SPY", "{not json")
        cold = EarningsCalendar(persist=state)
        monkeypatch.setattr(cold, "_fetch_earnings", _fail_earnings)
        assert cold.get_next_earnings("SPY").data_available is False

    def test_no_persist_is_the_w44_behaviour(self, monkeypatch):
        cal = EarningsCalendar()
        monkeypatch.setattr(cal, "_fetch_earnings", _fail_earnings)
        assert cal.get_next_earnings("SPY").data_available is False


class TestDividendLkgIsPersisted:
    def test_a_real_answer_is_written_and_read_back_cold(self, state, monkeypatch):
        ex = today_et() + timedelta(days=9)
        warm = DividendCalendar(persist=state)
        monkeypatch.setattr(warm, "_fetch_dividend", lambda s: DividendInfo(
            symbol=s, next_ex_div_date=ex, expected_amount=1.75))
        warm.get_next_ex_div("SPY")
        row = json.loads(state.get_state("dividend_lkg_SPY", ""))
        assert row["next_ex_div_date"] == ex.isoformat()
        assert row["expected_amount"] == pytest.approx(1.75)

        cold = DividendCalendar(persist=state)
        monkeypatch.setattr(cold, "_fetch_dividend", _fail_dividend)
        info = cold.get_next_ex_div("SPY")
        assert info.data_available is True
        assert info.next_ex_div_date == ex
        assert info.expected_amount == pytest.approx(1.75)

    def test_a_failed_fetch_is_never_persisted(self, state, monkeypatch):
        cal = DividendCalendar(persist=state)
        monkeypatch.setattr(cal, "_fetch_dividend", _fail_dividend)
        cal.get_next_ex_div("QQQ")
        assert state.get_state("dividend_lkg_QQQ", "") == ""

    def test_a_persisted_answer_past_max_age_is_not_served(self, state, monkeypatch):
        state.set_state("dividend_lkg_SPY", json.dumps({
            "symbol": "SPY", "next_ex_div_date": None, "expected_amount": None,
            "fetched_at": time.time() - 3700}))
        cold = DividendCalendar(persist=state, lkg_max_age_hours=1)
        monkeypatch.setattr(cold, "_fetch_dividend", _fail_dividend)
        assert cold.get_next_ex_div("SPY").data_available is False


class TestTheOrchestratorWiresIt:
    def test_both_calendars_persist_through_the_bot_state(self, tmp_path, monkeypatch):
        from tests.test_hot_path_smoke import build_smoke_orchestrator
        monkeypatch.chdir(tmp_path)
        rig = build_smoke_orchestrator()
        try:
            orch = rig.orch
            assert orch._earnings._persist is orch._state
            assert orch._dividends._persist is orch._state
        finally:
            rig.restore()

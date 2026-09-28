"""W58: historical options vol-surface backfill script.

Covers: the moneyness-over-life selection filter (pure function, no
network), and process_contract's resume-skip + real IV-solve integration
against a FAKE PolygonClient (real domain data, no MagicMock, no network).
No inspect.getsource.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# The script applies apply_runtime_env_defaults() and does real imports at
# module load time -- importing it here (rather than execing the file) is
# safe because it has no top-level side effects beyond that shared,
# idempotent env setup every other script in this repo already relies on.
import backfill_option_history as m  # noqa: E402

from ait.data.historical import HistoricalDataStore  # noqa: E402


def _temp_store() -> HistoricalDataStore:
    tmpdir = tempfile.mkdtemp()
    return HistoricalDataStore(db_path=Path(tmpdir) / "test.db")


def _closes(prices: dict[str, float]) -> pd.Series:
    idx = pd.to_datetime(list(prices.keys()))
    return pd.Series(list(prices.values()), index=idx)


# ---------------------------------------------------------------------------
# in_moneyness_band_over_life
# ---------------------------------------------------------------------------

class TestInMoneynessBandOverLife:
    def test_atm_at_issuance_never_leaves_band(self):
        closes = _closes({"2024-09-01": 480.0, "2024-09-15": 482.0, "2024-09-30": 479.0})
        assert m.in_moneyness_band_over_life(
            480.0, closes, date(2024, 9, 1), date(2024, 9, 30), 0.30) is True

    def test_starts_atm_drifts_out_of_band_by_expiry_still_included(self):
        """The exact case the 'whole life, not expiry-day' fix targets: a
        strike that was ATM on day 1 but is 46% OTM by expiry must still
        pass, because it WAS in-band at some point in its life."""
        closes = _closes({"2024-09-01": 480.0, "2024-10-18": 700.0})
        assert m.in_moneyness_band_over_life(
            480.0, closes, date(2024, 9, 1), date(2024, 10, 18), 0.30) is True

    def test_always_outside_band_excluded(self):
        closes = _closes({"2024-09-01": 470.0, "2024-09-15": 480.0, "2024-09-30": 490.0})
        assert m.in_moneyness_band_over_life(
            1000.0, closes, date(2024, 9, 1), date(2024, 9, 30), 0.30) is False

    def test_empty_window_excluded(self):
        closes = _closes({"2024-09-01": 480.0})
        assert m.in_moneyness_band_over_life(
            480.0, closes, date(2025, 1, 1), date(2025, 1, 31), 0.30) is False


# ---------------------------------------------------------------------------
# process_contract -- resume-skip and real IV-solve, against a fake client
# ---------------------------------------------------------------------------

class _FakeClient:
    """Stands in for PolygonClient at the network boundary. Raises if
    get_daily_bars is called when the test expects a resume-skip, so a
    regression that fetches anyway fails loudly rather than silently."""

    def __init__(self, bars=None, raise_on_fetch=False):
        self._bars = bars if bars is not None else []
        self._raise_on_fetch = raise_on_fetch
        self.fetch_calls = 0

    def get_daily_bars(self, option_ticker, from_date, to_date):
        self.fetch_calls += 1
        if self._raise_on_fetch:
            raise AssertionError(
                "get_daily_bars called on an already-fetched contract -- "
                "resume-skip should have prevented this network call"
            )
        return self._bars


def _contract(ticker="O:QQQ241018C00480000", expiry="2024-10-18", right="C", strike=480.0):
    return {"option_ticker": ticker, "expiry": expiry, "right": right, "strike": strike}


class TestProcessContract:
    def test_skips_already_fetched_contract_without_a_network_call(self):
        store = _temp_store()
        contract = _contract()
        # Seed the DB as if this contract were already fetched.
        store.save_option_daily_bars("QQQ", pd.DataFrame([{
            "option_ticker": contract["option_ticker"], "expiry": contract["expiry"],
            "right": "C", "strike": 480.0, "date": "2024-09-01", "dte": 47,
            "open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 100,
            "vwap": 5.0, "transactions": 10, "underlying_close": 480.0,
            "iv": 0.20, "iv_outcome": "solved", "delta": 0.5, "gamma": 0.02,
            "theta": -0.05, "vega": 0.10, "risk_free_rate": 0.05,
        }]))
        client = _FakeClient(raise_on_fetch=True)
        closes = _closes({"2024-09-01": 480.0})

        rows = m.process_contract(
            client, store, "QQQ", contract, closes,
            date(2024, 9, 1), date(2024, 10, 18), 0.05)

        assert rows == 0
        assert client.fetch_calls == 0

    def test_real_iv_solve_for_a_new_contract(self):
        """Drives the REAL solve_iv_from_price/compute_greeks_from_sigma
        (not mocked) against one fake daily bar, confirms the stored row
        has a genuinely solved IV and greeks, not placeholders."""
        store = _temp_store()
        contract = _contract(strike=480.0)
        bar_date = date(2024, 9, 18)
        t_ms = int(pd.Timestamp(bar_date, tz="UTC").timestamp() * 1000)
        bars = [{"o": 5.0, "h": 5.5, "l": 4.8, "c": 5.2, "v": 1000,
                "vw": 5.1, "n": 200, "t": t_ms}]
        client = _FakeClient(bars=bars)
        closes = _closes({str(bar_date): 480.0})

        rows = m.process_contract(
            client, store, "QQQ", contract, closes,
            date(2024, 9, 1), date(2024, 10, 18), 0.05)

        assert rows == 1
        assert client.fetch_calls == 1
        assert store.option_ticker_has_data(contract["option_ticker"]) is True

        import sqlite3
        with sqlite3.connect(store._db_path) as conn:
            row = conn.execute(
                "SELECT iv, iv_outcome, delta, risk_free_rate, dte FROM "
                "option_daily_bars WHERE option_ticker = ?",
                (contract["option_ticker"],),
            ).fetchone()
        iv, outcome, delta, rfr, dte = row
        assert outcome == "solved"
        assert iv is not None and iv > 0
        assert delta is not None
        assert rfr == pytest.approx(0.05)
        assert dte == (date(2024, 10, 18) - bar_date).days

    def test_missing_underlying_close_skips_that_bar(self):
        """A bar on a date with no underlying close (e.g. a data gap) must
        be dropped, not fed a garbage spot into the IV solve."""
        store = _temp_store()
        contract = _contract()
        bar_date = date(2024, 9, 18)
        t_ms = int(pd.Timestamp(bar_date, tz="UTC").timestamp() * 1000)
        bars = [{"o": 5.0, "h": 5.5, "l": 4.8, "c": 5.2, "v": 1000,
                "vw": 5.1, "n": 200, "t": t_ms}]
        client = _FakeClient(bars=bars)
        closes = _closes({"2024-09-01": 480.0})  # no entry for bar_date

        rows = m.process_contract(
            client, store, "QQQ", contract, closes,
            date(2024, 9, 1), date(2024, 10, 18), 0.05)

        assert rows == 0

    def test_no_bars_returned_stores_nothing(self):
        store = _temp_store()
        contract = _contract()
        client = _FakeClient(bars=[])
        closes = _closes({"2024-09-01": 480.0})

        rows = m.process_contract(
            client, store, "QQQ", contract, closes,
            date(2024, 9, 1), date(2024, 10, 18), 0.05)

        assert rows == 0
        assert store.option_ticker_has_data(contract["option_ticker"]) is False


# ---------------------------------------------------------------------------
# W58 follow-up: network-retry resilience (PolygonClient._get)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}

    def json(self):
        return self._json_data


class _FlakySession:
    """Raises a real requests ConnectionError `fail_times` times, then
    returns a real (fake-content) response -- proves the retry loop is
    driven by genuine requests.exceptions.RequestException handling, not a
    stand-in exception type."""

    def __init__(self, fail_times: int, response):
        self._fail_times = fail_times
        self._response = response
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise m.requests.exceptions.ConnectionError("simulated network blip")
        return self._response


class _AlwaysFailsSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        raise m.requests.exceptions.ConnectionError("permanent simulated failure")


class TestPolygonClientGetRetry:
    def test_retries_transient_network_errors_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(m.time, "sleep", lambda s: None)  # no real waiting in tests
        response = _FakeResponse(json_data={"status": "OK", "results": []})
        session = _FlakySession(fail_times=3, response=response)
        client = m.PolygonClient("fake-key", pause_secs=0, session=session)

        data = client._get("https://api.polygon.io/v3/reference/options/contracts")

        assert data == {"status": "OK", "results": []}
        assert session.calls == 4, "3 failures + 1 success"

    def test_exhausting_retries_raises_runtime_error(self, monkeypatch):
        monkeypatch.setattr(m.time, "sleep", lambda s: None)
        session = _AlwaysFailsSession()
        client = m.PolygonClient("fake-key", pause_secs=0, session=session)

        with pytest.raises(RuntimeError, match="Polygon request failed"):
            client._get("https://api.polygon.io/v3/reference/options/contracts")

        assert session.calls == len(m.NETWORK_RETRY_BACKOFFS)


# ---------------------------------------------------------------------------
# W58 follow-up: main()/_run()'s enumeration-failure and status-file paths
# ---------------------------------------------------------------------------

def _seed_underlying(store: HistoricalDataStore, symbol: str = "QQQ") -> None:
    df = pd.DataFrame({
        "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.0],
        "Volume": [1000],
    }, index=pd.to_datetime(["2026-06-01"]))
    store.save(symbol, df)


class TestRunEnumerationFailure:
    def test_enumeration_failure_writes_crashed_status_and_returns_nonzero(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(m, "STATUS_FILE", tmp_path / "status.json")
        monkeypatch.setenv("POLYGON_API_KEY", "fake-key-for-test")
        store = _temp_store()
        _seed_underlying(store)
        monkeypatch.setattr(m, "HistoricalDataStore", lambda **kw: store)

        def _boom(*a, **kw):
            raise m.requests.exceptions.ConnectionError("simulated enumeration failure")

        monkeypatch.setattr(m, "enumerate_contracts", _boom)

        rc = m.main(["--symbol", "QQQ", "--start-date", "2026-06-01",
                    "--end-date", "2026-06-02"])

        assert rc == 1
        status = json.loads((tmp_path / "status.json").read_text())
        assert status["status"] == "crashed"
        assert "enumeration failed" in status["last_error"]


class TestRunStatusFile:
    def test_status_file_after_a_successful_small_run(self, monkeypatch, tmp_path):
        monkeypatch.setattr(m, "STATUS_FILE", tmp_path / "status.json")
        monkeypatch.setenv("POLYGON_API_KEY", "fake-key-for-test")
        store = _temp_store()
        _seed_underlying(store)
        monkeypatch.setattr(m, "HistoricalDataStore", lambda **kw: store)

        fake_contracts = [{
            "option_ticker": "O:QQQ260601C00100000", "expiry": "2026-06-01",
            "right": "C", "strike": 100.0,
        }]
        monkeypatch.setattr(m, "enumerate_contracts", lambda *a, **kw: fake_contracts)
        monkeypatch.setattr(m, "process_contract", lambda *a, **kw: 3)

        rc = m.main(["--symbol", "QQQ", "--start-date", "2026-06-01",
                    "--end-date", "2026-06-02"])

        assert rc == 0
        status = json.loads((tmp_path / "status.json").read_text())
        assert status["status"] == "complete"
        assert status["symbol"] == "QQQ"
        assert status["contracts_total"] == 1
        assert status["contracts_done"] == 1
        assert status["rows_written"] == 3
        assert status["current_ticker"] is None
        assert status["last_error"] is None
        assert "started_at" in status and "last_updated" in status

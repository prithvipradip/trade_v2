"""Tests for PDTGuard (Pattern Day Trader protection)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ait.config.settings import AccountConfig
from ait.risk.pdt_guard import MAX_DAY_TRADES, PDTGuard, PDTStatus


def _make_trading_days(n: int, end_date: date | None = None) -> list[date]:
    """Build a list of n consecutive weekday dates ending at end_date."""
    end = end_date or date.today()
    days: list[date] = []
    d = end
    while len(days) < n:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    return days


@pytest.fixture
def mock_state():
    """A mock StateManager that stores state in a dict."""
    state = MagicMock()
    _store: dict[str, str] = {}

    def _get(key, default=""):
        return _store.get(key, default)

    def _set(key, value):
        _store[key] = value

    state.get_state = MagicMock(side_effect=_get)
    state.set_state = MagicMock(side_effect=_set)
    state._store = _store  # Expose for test introspection
    return state


@pytest.fixture
def trading_days():
    """5 recent trading days for the rolling window."""
    return _make_trading_days(10)


@pytest.fixture
def pdt_guard(account_config: AccountConfig, mock_state, trading_days):
    """PDTGuard with mocked state and time utilities."""
    with patch(
        "ait.risk.pdt_guard.get_recent_trading_days", return_value=trading_days[-5:]
    ):
        guard = PDTGuard(account_config, mock_state)
    return guard


class TestCountDayTrades:
    """Test counting day trades in rolling 5-day window."""

    def test_no_trades_initially(self, pdt_guard, trading_days) -> None:
        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days",
            return_value=trading_days[-5:],
        ):
            status = pdt_guard.get_status()
        assert status.day_trades_used == 0
        assert status.day_trades_remaining == MAX_DAY_TRADES

    def test_counts_recorded_trades(self, pdt_guard, trading_days) -> None:
        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days",
            return_value=trading_days[-5:],
        ):
            with patch("ait.risk.pdt_guard.date") as mock_date:
                mock_date.today.return_value = trading_days[-1]
                pdt_guard.record_day_trade("AAPL")
                pdt_guard.record_day_trade("MSFT")

            status = pdt_guard.get_status()
        assert status.day_trades_used == 2
        assert status.day_trades_remaining == 1

    def test_can_day_trade_with_remaining(self, pdt_guard, trading_days) -> None:
        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days",
            return_value=trading_days[-5:],
        ):
            with patch("ait.risk.pdt_guard.date") as mock_date:
                mock_date.today.return_value = trading_days[-1]
                pdt_guard.record_day_trade("SPY")

            assert pdt_guard.can_day_trade() is True


class TestBlockingAt3:
    """Test blocking when 3 day trades used."""

    def test_blocked_at_3_day_trades(self, pdt_guard, trading_days) -> None:
        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days",
            return_value=trading_days[-5:],
        ):
            with patch("ait.risk.pdt_guard.date") as mock_date:
                mock_date.today.return_value = trading_days[-1]
                pdt_guard.record_day_trade("AAPL")
                pdt_guard.record_day_trade("MSFT")
                pdt_guard.record_day_trade("GOOG")

            assert pdt_guard.can_day_trade() is False

    def test_status_shows_zero_remaining(self, pdt_guard, trading_days) -> None:
        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days",
            return_value=trading_days[-5:],
        ):
            with patch("ait.risk.pdt_guard.date") as mock_date:
                mock_date.today.return_value = trading_days[-1]
                for sym in ["A", "B", "C"]:
                    pdt_guard.record_day_trade(sym)

            status = pdt_guard.get_status()
        assert status.can_day_trade is False
        assert status.day_trades_remaining == 0


class TestPurgeOldTrades:
    """Test purging old trades outside the 5-day window."""

    def test_old_trades_purged(self, account_config, mock_state) -> None:
        """Trades older than the 5-day window should be removed."""
        today = date(2026, 3, 9)
        # Build a window where the cutoff is 5 trading days back
        window_days = _make_trading_days(5, end_date=today)
        old_date = (window_days[0] - timedelta(days=3)).isoformat()

        # Pre-populate state with an old trade and a recent one
        trades = [
            {"date": old_date, "symbol": "OLD"},
            {"date": today.isoformat(), "symbol": "NEW"},
        ]
        mock_state._store["pdt_day_trades"] = json.dumps(trades)

        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days",
            return_value=window_days,
        ):
            guard = PDTGuard(account_config, mock_state)
            status = guard.get_status()

        assert status.day_trades_used == 1  # Only NEW remains

    def test_all_old_trades_purged(self, account_config, mock_state) -> None:
        """If all trades are old, count should be zero."""
        today = date(2026, 3, 9)
        window_days = _make_trading_days(5, end_date=today)
        old_date = (window_days[0] - timedelta(days=5)).isoformat()

        trades = [{"date": old_date, "symbol": "OLD"}]
        mock_state._store["pdt_day_trades"] = json.dumps(trades)

        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days",
            return_value=window_days,
        ):
            guard = PDTGuard(account_config, mock_state)
            assert guard.can_day_trade() is True
            status = guard.get_status()
            assert status.day_trades_used == 0


class TestDisabledMode:
    """Test disabled mode (account >= $25k)."""

    def test_disabled_always_allows(
        self, account_config_over_25k: AccountConfig, mock_state
    ) -> None:
        """When account is over $25k, PDT guard is disabled."""
        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days", return_value=[]
        ):
            guard = PDTGuard(account_config_over_25k, mock_state)

        assert guard.can_day_trade() is True

    def test_disabled_does_not_record(
        self, account_config_over_25k: AccountConfig, mock_state
    ) -> None:
        """When disabled, record_day_trade should be a no-op."""
        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days", return_value=[]
        ):
            guard = PDTGuard(account_config_over_25k, mock_state)

        guard.record_day_trade("AAPL")
        # _save_state should not have been called
        mock_state.set_state.assert_not_called()

    def test_disabled_status(
        self, account_config_over_25k: AccountConfig, mock_state
    ) -> None:
        """Status should show enabled=False when over $25k."""
        with patch(
            "ait.risk.pdt_guard.get_recent_trading_days", return_value=[]
        ):
            guard = PDTGuard(account_config_over_25k, mock_state)
            status = guard.get_status()

        assert status.enabled is False
        assert status.can_day_trade is True


class TestWouldBeDayTrade:
    """Test would_be_day_trade helper."""

    def test_same_day_is_day_trade(self, pdt_guard) -> None:
        today = date.today()
        assert pdt_guard.would_be_day_trade("SPY", today) is True

    def test_different_day_is_not(self, pdt_guard) -> None:
        yesterday = date.today() - timedelta(days=1)
        assert pdt_guard.would_be_day_trade("SPY", yesterday) is False

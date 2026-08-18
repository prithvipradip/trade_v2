"""Tests for Phase 9: IBKR as primary intraday data source.

Live tests (marked ibkr) require IB Gateway running on 127.0.0.1:4002.
Mock tests run in all environments.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ibkr_bars(n: int = 78) -> list:
    """Build minimal ib_insync BarData-like objects for mocking."""
    from ib_insync import BarData
    start = datetime(2026, 4, 28, 9, 30, tzinfo=timezone.utc)
    bars = []
    price = 500.0
    for i in range(n):
        dt = start + timedelta(minutes=5 * i)
        price += np.random.default_rng(i).normal(0, 0.5)
        bar = BarData(
            date=dt, open=price - 0.1, high=price + 0.3,
            low=price - 0.3, close=price, volume=10000,
            average=price, barCount=100,
        )
        bars.append(bar)
    return bars


def _make_intraday_df(n_sessions: int = 7) -> pd.DataFrame:
    """Synthetic intraday DataFrame with UTC DatetimeIndex."""
    n = 78 * n_sessions
    start = datetime(2026, 4, 29, 13, 30, tzinfo=timezone.utc)
    idx = pd.DatetimeIndex(
        [start + timedelta(minutes=5 * i) for i in range(n)], tz="UTC"
    )
    price = 500.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.001, n)))
    return pd.DataFrame({
        "Open": price * 0.999, "High": price * 1.001,
        "Low": price * 0.999, "Close": price,
        "Volume": np.random.default_rng(1).integers(5000, 50000, n),
    }, index=idx)


def _make_market_data_svc(ibkr_connected: bool = False) -> "MarketDataService":
    from ait.data.market_data import MarketDataService
    mock_ibkr = MagicMock()
    mock_ibkr.connected = ibkr_connected
    return MarketDataService(ibkr_client=mock_ibkr)


# ---------------------------------------------------------------------------
# Unit tests — mock IBKR
# ---------------------------------------------------------------------------

class TestDaysToIbkrDuration:

    def test_short_window_uses_days(self) -> None:
        from ait.data.market_data import MarketDataService
        assert MarketDataService._days_to_ibkr_duration(7) == "7 D"
        assert MarketDataService._days_to_ibkr_duration(14) == "14 D"

    def test_medium_window_uses_months(self) -> None:
        from ait.data.market_data import MarketDataService
        assert MarketDataService._days_to_ibkr_duration(30) == "1 M"
        assert MarketDataService._days_to_ibkr_duration(90) == "3 M"
        assert MarketDataService._days_to_ibkr_duration(180) == "6 M"

    def test_long_window_uses_years(self) -> None:
        from ait.data.market_data import MarketDataService
        assert MarketDataService._days_to_ibkr_duration(365) == "1 Y"
        assert MarketDataService._days_to_ibkr_duration(730) == "1 Y"


class TestGetIntradayFallback:

    def test_returns_none_when_ibkr_disconnected_and_yahoo_empty(self) -> None:
        svc = _make_market_data_svc(ibkr_connected=False)

        async def run():
            with patch.object(svc, "_get_yahoo_intraday", new=AsyncMock(return_value=None)):
                return await svc.get_intraday("SPY", interval="5m", days=7)

        result = asyncio.run(run())
        assert result is None

    def test_falls_back_to_yahoo_when_ibkr_disconnected(self) -> None:
        svc = _make_market_data_svc(ibkr_connected=False)
        yahoo_df = _make_intraday_df(3)

        async def run():
            with patch.object(svc, "_get_yahoo_intraday", new=AsyncMock(return_value=yahoo_df)):
                return await svc.get_intraday("SPY", interval="5m", days=7)

        result = asyncio.run(run())
        assert result is not None
        assert len(result) == len(yahoo_df)

    def test_ibkr_result_preferred_over_yahoo(self) -> None:
        svc = _make_market_data_svc(ibkr_connected=True)
        ibkr_df = _make_intraday_df(7)
        yahoo_df = _make_intraday_df(3)

        async def run():
            with patch.object(svc, "_get_ibkr_intraday", new=AsyncMock(return_value=ibkr_df)):
                with patch.object(svc, "_get_yahoo_intraday", new=AsyncMock(return_value=yahoo_df)):
                    return await svc.get_intraday("SPY", interval="5m", days=7)

        result = asyncio.run(run())
        assert len(result) == len(ibkr_df), "IBKR result should take precedence over Yahoo"

    def test_output_has_ohlcv_columns(self) -> None:
        svc = _make_market_data_svc(ibkr_connected=True)
        ibkr_df = _make_intraday_df(3)

        async def run():
            with patch.object(svc, "_get_ibkr_intraday", new=AsyncMock(return_value=ibkr_df)):
                return await svc.get_intraday("SPY", interval="5m", days=7)

        result = asyncio.run(run())
        assert result is not None
        assert set(result.columns) >= {"Open", "High", "Low", "Close", "Volume"}

    def test_output_has_utc_index(self) -> None:
        svc = _make_market_data_svc(ibkr_connected=True)
        ibkr_df = _make_intraday_df(3)

        async def run():
            with patch.object(svc, "_get_ibkr_intraday", new=AsyncMock(return_value=ibkr_df)):
                return await svc.get_intraday("SPY", interval="5m", days=7)

        result = asyncio.run(run())
        assert result is not None
        assert result.index.tzinfo is not None, "Index must be timezone-aware (UTC)"


class TestGetIntradaySince:

    def test_returns_only_bars_after_cutoff(self) -> None:
        svc = _make_market_data_svc(ibkr_connected=True)
        full_df = _make_intraday_df(5)
        cutoff = full_df.index[100]

        async def run():
            with patch.object(svc, "_get_ibkr_intraday", new=AsyncMock(return_value=full_df)):
                return await svc.get_intraday_since("SPY", since=cutoff)

        result = asyncio.run(run())
        assert result is not None and not result.empty
        assert result.index.min() > cutoff, "All returned bars must be strictly after cutoff"

    def test_returns_none_when_no_new_bars(self) -> None:
        svc = _make_market_data_svc(ibkr_connected=True)
        full_df = _make_intraday_df(1)
        cutoff = full_df.index[-1]  # cutoff is at the last bar

        async def run():
            with patch.object(svc, "_get_ibkr_intraday", new=AsyncMock(return_value=full_df)):
                return await svc.get_intraday_since("SPY", since=cutoff)

        result = asyncio.run(run())
        assert result is None or (result is not None and result.empty), (
            "No bars should be returned when cutoff >= last bar"
        )

    def test_falls_back_to_yahoo_when_ibkr_unavailable(self) -> None:
        svc = _make_market_data_svc(ibkr_connected=False)
        yahoo_df = _make_intraday_df(3)
        cutoff = yahoo_df.index[50]

        async def run():
            with patch.object(svc, "_get_ibkr_intraday", new=AsyncMock(return_value=None)):
                with patch.object(svc, "_get_yahoo_intraday", new=AsyncMock(return_value=yahoo_df)):
                    return await svc.get_intraday_since("SPY", since=cutoff)

        result = asyncio.run(run())
        if result is not None and not result.empty:
            assert result.index.min() > cutoff


# ---------------------------------------------------------------------------
# Backfill script tests (CLI)
# ---------------------------------------------------------------------------

class TestBackfillScript:

    def test_script_exists(self) -> None:
        assert Path("scripts/backfill_intraday.py").exists(), (
            "scripts/backfill_intraday.py must exist (Phase 9)"
        )

    def test_help_exits_zero(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/backfill_intraday.py", "--help"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0

    def test_requires_symbols_argument(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/backfill_intraday.py"],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode != 0

    def test_dry_run_exits_zero(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable, "scripts/backfill_intraday.py",
                "--symbols", "SPY",
                "--years", "0.5",
                "--dry-run",
                "--db-path", str(tmp_path / "test.db"),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"dry-run failed:\n{result.stderr}"

    def test_dry_run_prints_plan(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable, "scripts/backfill_intraday.py",
                "--symbols", "SPY", "QQQ",
                "--years", "0.5",
                "--dry-run",
                "--db-path", str(tmp_path / "test.db"),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert "SPY" in result.stdout
        assert "QQQ" in result.stdout
        assert "DRY RUN" in result.stdout


# ---------------------------------------------------------------------------
# Live integration tests — require IB Gateway on 127.0.0.1:4002
# ---------------------------------------------------------------------------

@pytest.mark.ibkr
class TestIBKRIntradayLive:
    """Live tests against IB Gateway. Run with: pytest -m ibkr"""

    @pytest.fixture(scope="class")
    def ibkr_svc(self):
        """Build a MarketDataService backed by a real IBKR connection."""
        from ib_insync import IB, util
        from ait.data.market_data import MarketDataService

        ib = IB()

        async def _connect():
            await ib.connectAsync("127.0.0.1", 4002, clientId=96, timeout=10)

        asyncio.run(_connect())

        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.ib = ib

        async def _qualify(contract):
            q = await ib.qualifyContractsAsync(contract)
            return q[0] if q else None

        mock_client.qualify_contract = _qualify

        svc = MarketDataService(ibkr_client=mock_client)
        yield svc
        ib.disconnect()

    def test_get_intraday_returns_dataframe(self, ibkr_svc) -> None:
        async def run():
            return await ibkr_svc.get_intraday("SPY", interval="5m", days=7)
        df = asyncio.run(run())
        assert df is not None and not df.empty
        assert set(df.columns) >= {"Open", "High", "Low", "Close", "Volume"}
        assert df.index.tzinfo is not None
        # ~7 trading days × 78 bars = ~546 bars (≥300 to account for partial weeks)
        assert len(df) >= 300, f"Expected ≥300 bars for 7 days, got {len(df)}"

    def test_get_intraday_prices_are_positive(self, ibkr_svc) -> None:
        async def run():
            return await ibkr_svc.get_intraday("SPY", interval="5m", days=5)
        df = asyncio.run(run())
        assert df is not None
        assert (df["Close"] > 0).all(), "All close prices must be positive"
        assert (df["High"] >= df["Low"]).all(), "High must be >= Low for all bars"

    def test_get_intraday_since_returns_incremental_bars(self, ibkr_svc) -> None:
        async def run():
            full = await ibkr_svc.get_intraday("SPY", interval="5m", days=5)
            cutoff = full.index[-50]
            incremental = await ibkr_svc.get_intraday_since("SPY", since=cutoff)
            return full, cutoff, incremental

        full, cutoff, incremental = asyncio.run(run())
        assert incremental is not None and not incremental.empty
        assert incremental.index.min() > cutoff

    def test_backfill_stores_data_in_sqlite(self, tmp_path: Path) -> None:
        result = subprocess.run(
            [
                sys.executable, "scripts/backfill_intraday.py",
                "--symbols", "SPY",
                "--years", "0.1",      # ~1 month — fast
                "--client-id", "95",
                "--db-path", str(tmp_path / "backfill_test.db"),
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"backfill failed:\n{result.stderr}\n{result.stdout}"
        assert "upserted" in result.stdout

        from ait.data.historical import HistoricalDataStore
        store = HistoricalDataStore(db_path=tmp_path / "backfill_test.db")
        df = store.load_intraday("SPY", days=60)
        assert len(df) >= 100, f"Expected bars in DB after backfill, got {len(df)}"

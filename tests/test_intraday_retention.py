"""W113: the live bot never deletes 5-minute bars.

The table is shared with research (one set of tables), whose windows need the bars from 2024 on. The
orchestrator used to run ``cleanup_old_intraday(keep_days=730)`` after every save of new bars, which
would have removed everything older than two years the first time a scan saved a bar. The 730 days the
bot reads for its features is a LOAD window; nothing deletes rows.
"""
from __future__ import annotations

import pathlib
import re
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ait.data.historical import HistoricalDataStore
from tests.test_hot_path_smoke import SYMBOL, VIX, build_smoke_orchestrator

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "ait"


def _bars(start: datetime, n: int, px: float = 600.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex([start + timedelta(minutes=5 * i) for i in range(n)])
    return pd.DataFrame({"Open": px, "High": px + 1, "Low": px - 1, "Close": px, "Volume": 1000}, index=idx)


def _count(store: HistoricalDataStore, before: datetime | None = None) -> int:
    with sqlite3.connect(store._db_path) as conn:
        if before is None:
            return conn.execute("SELECT COUNT(*) FROM intraday_prices").fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM intraday_prices WHERE datetime < ?",
                            (before.isoformat(),)).fetchone()[0]


class TestAScanThatSavesBarsDeletesNothing:
    pytestmark = pytest.mark.asyncio

    @pytest.fixture
    def rig(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        smoke = build_smoke_orchestrator()
        try:
            yield smoke
        finally:
            smoke.restore()

    async def test_old_bars_survive_the_orchestrators_save_path(self, rig):
        orch = rig.orch
        store = orch._historical
        assert isinstance(store, HistoricalDataStore)
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        old_start = now - timedelta(days=1000)                     # well past the 730-day load window
        borderline = now - timedelta(days=731)
        store.save_intraday(SYMBOL, _bars(old_start, 40))
        store.save_intraday(SYMBOL, _bars(borderline, 12))
        old_rows = _count(store, now - timedelta(days=730))
        assert old_rows == 52

        fresh = _bars(now - timedelta(hours=2), 12)

        async def _since(symbol, since=None):
            return fresh
        orch._market_data.get_intraday_since = _since

        await orch._scan_symbol(SYMBOL, VIX, market_context={}, collect=[])

        assert _count(store, now - timedelta(days=730)) == old_rows         # nothing was deleted
        assert _count(store) == old_rows + 12                              # and the scan DID save the new bars

    async def test_the_load_window_is_still_two_years_and_does_not_delete(self, tmp_path):
        store = HistoricalDataStore(tmp_path / "h.db")
        now = datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
        store.save_intraday("SPY", _bars(now - timedelta(days=900), 20))
        store.save_intraday("SPY", _bars(now - timedelta(days=10), 20))
        loaded = store.load_intraday("SPY", days=730)
        assert len(loaded) == 20                                          # the load is windowed ...
        assert _count(store) == 40                                        # ... the table is not


class TestNothingCanDeleteMarketBarsAgain:
    def test_the_cleanup_method_is_gone(self):
        assert not hasattr(HistoricalDataStore, "cleanup_old_intraday")

    def test_no_source_file_references_a_cleanup_or_deletes_intraday_rows(self):
        offenders = []
        for path in SRC.rglob("*.py"):
            text = path.read_text()
            if "cleanup_old_intraday" in text:
                offenders.append(f"{path.relative_to(SRC)}: cleanup_old_intraday")
            for m in re.finditer(r"DELETE\s+FROM\s+\{?[^\s]*intraday[^\s]*", text, re.I):
                offenders.append(f"{path.relative_to(SRC)}: {m.group(0)}")
        assert offenders == []

    def test_re_saving_an_overlapping_range_leaves_existing_rows_alone(self, tmp_path):
        store = HistoricalDataStore(tmp_path / "h.db")
        start = datetime(2024, 3, 4, 14, 30, tzinfo=timezone.utc)
        store.save_intraday("SPY", _bars(start, 30), source="TRADES")
        store.save_intraday("SPY", _bars(start + timedelta(minutes=100), 30), source="TRADES")   # overlaps
        store.save_intraday("SPY", _bars(start, 30, px=1.0), source="MIDPOINT")                  # must not downgrade
        assert _count(store) == 30 + 30 - 10
        with sqlite3.connect(store._db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM intraday_prices WHERE source = 'TRADES' AND close = 600.0"
                                ).fetchone()[0] == 50

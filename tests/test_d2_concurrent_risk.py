"""D2 (decided 2026-07-16): the drawdown denominator is the MAX CONCURRENT
deployed risk over the window. Pins the event-sweep helper the scorecard uses.
"""
from ait.orchestration.master import _max_concurrent_car


def _r(entry, exit_, car):
    return {"entry_time": entry, "exit_time": exit_, "car": car}


class TestMaxConcurrentCar:
    def test_overlapping_windows_sum(self):
        rows = [_r("2026-07-06", "2026-07-10", 1000.0),
                _r("2026-07-07", "2026-07-09", 500.0)]
        assert _max_concurrent_car(rows) == 1500.0

    def test_sequential_windows_do_not_sum(self):
        rows = [_r("2026-07-06", "2026-07-07", 1000.0),
                _r("2026-07-08", "2026-07-09", 800.0)]
        assert _max_concurrent_car(rows) == 1000.0

    def test_open_trade_never_expires(self):
        # No exit_time -> still deployed; overlaps everything after entry.
        rows = [_r("2026-07-06", None, 900.0),
                _r("2026-07-12", "2026-07-13", 600.0)]
        assert _max_concurrent_car(rows) == 1500.0

    def test_zero_car_rows_contribute_nothing(self):
        rows = [_r("2026-07-06", "2026-07-10", 0.0),
                _r("2026-07-06", "2026-07-10", 700.0)]
        assert _max_concurrent_car(rows) == 700.0

    def test_exit_before_next_entry_releases_risk(self):
        # A flatten releasing risk must NOT drop the historical peak — the
        # exact artifact D2 kills (old method: open-book car -> $1k floor).
        rows = [_r("2026-07-06", "2026-07-08", 2000.0),
                _r("2026-07-09", "2026-07-10", 300.0)]
        assert _max_concurrent_car(rows) == 2000.0

    def test_empty(self):
        assert _max_concurrent_car([]) == 0.0

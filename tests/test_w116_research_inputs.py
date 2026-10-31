"""W116 WP0: the research data contract's types and the one window rule."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from ait.data.research_inputs import (
    DataGap,
    ResearchDataUnavailable,
    Window,
    last_completed_session,
    last_session_on_or_before,
    required_window,
    sessions_between,
    window_start,
)


class TestSessions:
    def test_a_saturday_run_needs_friday(self):
        assert last_completed_session(date(2026, 10, 31)) == date(2026, 10, 30)

    def test_a_monday_run_needs_the_friday_before(self):
        assert last_completed_session(date(2026, 11, 2)) == date(2026, 10, 30)

    def test_todays_partial_session_is_never_required(self):
        assert last_completed_session(date(2026, 10, 28)) == date(2026, 10, 27)

    def test_the_day_after_a_holiday_skips_it(self):
        # 2026-11-26 is Thanksgiving: the run on Friday 11-27 needs Wednesday 11-25
        assert last_completed_session(date(2026, 11, 27)) == date(2026, 11, 25)

    def test_sessions_between_has_no_weekend_or_holiday(self):
        days = sessions_between(date(2026, 11, 23), date(2026, 11, 30))
        assert days == [date(2026, 11, 23), date(2026, 11, 24), date(2026, 11, 25),
                        date(2026, 11, 27), date(2026, 11, 30)]

    def test_on_or_before_snaps_a_weekend_back(self):
        assert last_session_on_or_before(date(2026, 10, 31)) == date(2026, 10, 30)
        assert last_session_on_or_before(date(2026, 10, 30)) == date(2026, 10, 30)


class TestRequiredWindow:
    TODAY = date(2026, 10, 31)

    def test_an_unpinned_run_ends_at_the_last_completed_session(self):
        w = required_window(today=self.TODAY, start_date=date(2024, 1, 2))
        assert w == Window(date(2024, 1, 2), date(2026, 10, 30))

    def test_a_pinned_run_ends_on_its_pin_not_today(self):
        w = required_window(today=self.TODAY, start_date=date(2024, 1, 2), end_date=date(2025, 12, 31))
        assert w.end == date(2025, 12, 31)

    def test_a_pin_on_a_weekend_snaps_back_to_a_session(self):
        assert required_window(today=self.TODAY, start_date=date(2024, 1, 2), end_date=date(2025, 12, 28)).end \
            == date(2025, 12, 26)

    def test_a_pin_in_the_future_is_clipped_to_the_last_completed_session(self):
        assert required_window(today=self.TODAY, start_date=date(2024, 1, 2), end_date=date(2027, 1, 4)).end \
            == date(2026, 10, 30)

    def test_days_count_back_from_the_end_with_the_buffer(self):
        w = required_window(today=self.TODAY, days=730, end_date=date(2025, 12, 31))
        assert w.start == window_start(730, date(2025, 12, 31)) == date(2023, 12, 2)

    def test_a_days_only_request_counts_back_from_today_when_unpinned(self):
        w = required_window(today=self.TODAY, days=730)
        assert w.start == self.TODAY - timedelta(days=760) == window_start(730, self.TODAY)
        assert w.end == date(2026, 10, 30)

    def test_an_explicit_start_wins_over_days(self):
        w = required_window(today=self.TODAY, days=730, start_date=date(2024, 6, 3))
        assert w.start == date(2024, 6, 3)

    def test_neither_start_nor_days_is_an_error(self):
        with pytest.raises(ValueError):
            required_window(today=self.TODAY)

    def test_a_backwards_window_is_an_error(self):
        with pytest.raises(ValueError):
            Window(date(2026, 1, 2), date(2025, 1, 2))


class TestRefusal:
    def test_the_message_lists_every_gap_with_its_fix(self):
        gaps = [DataGap("daily_prices", "QQQ", "every NYSE session 2024-01-02..2026-10-30",
                        "2024-01-02..2026-10-27 (3 sessions missing)",
                        "python scripts/backfill_historical_data.py --symbols QQQ --mode daily"),
                DataGap("daily_iv", "VIX", "VIX bars 2022-09-01..2026-10-30", "no rows",
                        "python scripts/backfill_historical_data.py --mode vix --symbols VIX")]
        err = ResearchDataUnavailable(gaps, context="walk_forward")
        text = str(err)
        assert "2 gap(s)" in text and "walk_forward" in text
        assert "daily_prices[QQQ]" in text and "backfill_historical_data.py --symbols QQQ --mode daily" in text
        assert "daily_iv[VIX]" in text and err.gaps == tuple(gaps)

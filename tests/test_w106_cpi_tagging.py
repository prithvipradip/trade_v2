"""W106 item 6: a CPI release with no "CPI MoM" headline row is still a CPI date.

The 2025-12-18 release (after the 2025 shutdown) has `U.S. Consumer Price Index
(CPI)` rows in the Investing.com calendar but no MoM headline, so the exact-name
match left the date untagged and research's macro blackout did not see a CPI print.
"""
from __future__ import annotations

import pandas as pd

from ait.data.economic_calendar import derive_engine_event_types


def _frame(rows):
    return pd.DataFrame([{"country": "United States", "type": "event", "reference_period": "",
                          **r} for r in rows])


def test_cpi_release_without_mom_headline_is_tagged_once():
    df = _frame([
        {"event_date": "2025-12-18", "event": "U.S. Consumer Price Index (CPI) YoY"},
        {"event_date": "2025-12-18", "event": "U.S. Consumer Price Index (CPI) n.s.a"},
        {"event_date": "2025-12-18", "event": "U.S. Core Consumer Price Index (CPI) YoY"},
    ])
    tags = derive_engine_event_types(df)
    assert list(tags) == ["CPI", None, None]


def test_a_date_with_the_headline_keeps_the_headline_tag_only():
    df = _frame([
        {"event_date": "2025-11-13", "event": "U.S. Consumer Price Index (CPI) YoY"},
        {"event_date": "2025-11-13", "event": "U.S. Consumer Price Index (CPI) MoM"},
    ])
    assert list(derive_engine_event_types(df)) == [None, "CPI"]


def test_core_only_or_non_us_rows_are_not_a_cpi_release():
    df = pd.concat([
        _frame([{"event_date": "2025-12-19", "event": "U.S. Core Consumer Price Index (CPI) YoY"}]),
        pd.DataFrame([{"country": "Canada", "type": "event", "reference_period": "",
                       "event_date": "2025-12-18", "event": "U.S. Consumer Price Index (CPI) YoY"}]),
    ], ignore_index=True)
    assert list(derive_engine_event_types(df)) == [None, None]

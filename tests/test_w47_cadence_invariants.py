"""W47 numeric-pairs-07 (R24/R26 register) — the fast-monitor cadence, the
quote cache TTL, the marks-missing alert window and the touch-stop staleness
budget were four numbers that had to agree, enforced only by hand-typed
comments at each of four sites (orchestrator.py, market_data.py,
portfolio.py, settings.py). R16/R19 already lived through the consequence of
this coupling drifting apart once (execution/portfolio.py:122-134): a
cache/cadence interaction misclassified a live feed as frozen and delayed a
real short-strike touch exit.

All four sites now derive from ait.utils.cadence. These tests assert the
relationships against the REAL imported constants and a REAL Settings/
ExitConfig load and a REAL MarketDataService cache fetch — not source-string
matching — so a future edit to any one number fails here instead of
silently re-opening the incident.

No inspect.getsource.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from ait.utils.cadence import (
    FAST_MONITOR_CADENCE_SECONDS,
    MARKS_MISSING_ALERT_SECONDS,
    QUOTE_CACHE_TTL_SECONDS,
    QUOTE_STALENESS_BUDGET_SECONDS,
)


class TestCadenceConstantsAgreeWithThemselves:
    """The invariants numeric-pairs-07 named, checked against the numbers
    every live call site actually imports (see the grep-audit in
    TestEveryCallSiteUsesTheSharedConstant below for proof they DO import
    them, not just that the constants are internally consistent)."""

    def test_quote_ttl_is_strictly_below_the_monitor_cadence(self):
        # Otherwise two consecutive monitor passes read the same cached
        # print and touch_confirm_ticks=2 confirms a touch against itself.
        assert QUOTE_CACHE_TTL_SECONDS < FAST_MONITOR_CADENCE_SECONDS

    def test_staleness_budget_is_at_least_six_times_the_cadence(self):
        assert QUOTE_STALENESS_BUDGET_SECONDS >= 6 * FAST_MONITOR_CADENCE_SECONDS

    def test_marks_missing_alert_ticks_times_cadence_is_five_minutes(self):
        from ait.execution.portfolio import PortfolioManager
        ticks = PortfolioManager.MARKS_MISSING_ALERT_TICKS
        assert ticks * FAST_MONITOR_CADENCE_SECONDS == MARKS_MISSING_ALERT_SECONDS
        assert ticks == round(MARKS_MISSING_ALERT_SECONDS / FAST_MONITOR_CADENCE_SECONDS)

    def test_touch_confirmation_window_fits_inside_the_staleness_budget(self):
        # The window a degraded quote needs to confirm a touch (ticks x
        # cadence) must not itself exceed the budget that triggered
        # "degraded" in the first place, or confirmation could never finish
        # before the input is considered unusable for a different reason.
        from ait.config.settings import ExitConfig
        exit_cfg = ExitConfig()
        confirm_window = exit_cfg.touch_confirm_ticks * FAST_MONITOR_CADENCE_SECONDS
        assert confirm_window < QUOTE_STALENESS_BUDGET_SECONDS

    def test_settings_default_staleness_is_the_shared_constant(self):
        from ait.config.settings import ExitConfig
        assert ExitConfig().max_quote_staleness_seconds == QUOTE_STALENESS_BUDGET_SECONDS


class TestEveryCallSiteUsesTheSharedConstant:
    """Proof the four sites import the authority rather than merely agreeing
    with it by coincidence — the exact failure mode numeric-pairs-07 named."""

    def test_orchestrator_monitor_interval_is_the_shared_constant(self):
        # Identity (`is`), not equality: proves orchestrator.py IMPORTED the
        # authority rather than retyping an equal-today literal. A real
        # end-to-end drive of `_trading_loop`'s own monitor_interval local
        # would need live IBKR wiring; this is the honest proxy for it, and
        # test_w47_protection_cadence.py drives the loop itself for real.
        from ait.bot import orchestrator as orch_mod
        assert orch_mod.FAST_MONITOR_CADENCE_SECONDS is FAST_MONITOR_CADENCE_SECONDS

    def test_portfolio_imports_the_shared_constants(self):
        from ait.execution import portfolio as portfolio_mod
        assert portfolio_mod.FAST_MONITOR_CADENCE_SECONDS is FAST_MONITOR_CADENCE_SECONDS
        assert portfolio_mod.MARKS_MISSING_ALERT_SECONDS is MARKS_MISSING_ALERT_SECONDS

    def test_market_data_imports_the_shared_constant(self):
        from ait.data import market_data as market_data_mod
        assert market_data_mod.QUOTE_CACHE_TTL_SECONDS is QUOTE_CACHE_TTL_SECONDS

    def test_settings_imports_the_shared_constant(self):
        from ait.config import settings as settings_mod
        assert settings_mod.QUOTE_STALENESS_BUDGET_SECONDS is QUOTE_STALENESS_BUDGET_SECONDS


class TestQuoteCacheAppliesTheSharedTTL:
    """Drives the real MarketDataService.get_quote() cache path end to end —
    proof the TTL actually APPLIED by TTLCache is the shared constant, not
    just that the constant exists."""

    async def test_cached_quote_expires_at_exactly_the_shared_ttl(self, monkeypatch):
        from ait.data.market_data import MarketDataService, Quote

        mock_ibkr = MagicMock()
        mock_ibkr.connected = False
        svc = MarketDataService(ibkr_client=mock_ibkr)

        calls = {"n": 0}

        async def _fake_ibkr_quote(symbol: str):
            calls["n"] += 1
            return Quote(symbol=symbol, bid=100.0, ask=100.10, last=100.05,
                         volume=1000, timestamp=datetime.now(timezone.utc))

        monkeypatch.setattr(svc, "_get_ibkr_quote", _fake_ibkr_quote)

        # ait.data.cache does `import time` (module import, not `from time
        # import time`), so patching the stdlib time module's own `.time`
        # affects every module that imported it, including the cache.
        fake_now = {"t": 1_700_000_000.0}
        monkeypatch.setattr(time, "time", lambda: fake_now["t"])

        first = await svc.get_quote("SPY")
        assert calls["n"] == 1
        assert first is not None

        # One second inside the TTL: still cached, no re-fetch.
        fake_now["t"] += QUOTE_CACHE_TTL_SECONDS - 1
        second = await svc.get_quote("SPY")
        assert calls["n"] == 1, "quote was re-fetched before the shared TTL elapsed"
        assert second is first

        # One second past the TTL: cache miss, real re-fetch.
        fake_now["t"] += 2
        third = await svc.get_quote("SPY")
        assert calls["n"] == 2, "quote was NOT re-fetched once the shared TTL elapsed"
        assert third is not None

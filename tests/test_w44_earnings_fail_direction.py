"""W44 (R26 register) — fail-direction-03 + concurrency-4, one root cause.

EarningsCalendar built an identically-shaped EarningsInfo(next_earnings_date=
None) whether a symbol genuinely has no upcoming earnings (every ETF in the
current universe) or the Yahoo fetch broke -- and cached either one for the
full 6-hour TTL. Both entry gates then read "no date" as "allow". concurrency-4
is the mechanism that makes the fail-open reachable at all: the TTLCache the
background fetch thread writes into had no lock, racing the main thread's own
reads/writes of the same instance.

Fixed with a last-known-good, verified against HEAD before implementing:
- data_available distinguishes a real answer (confirmed date, or CONFIRMED
  absence) from "unknown" (never fetched, or the fetch just failed).
- A failure is never cached; it falls back to the last successful fetch if
  one is still fresh (earnings.lkg_max_age_hours), else "unknown".
- Both gates refuse on "unknown" (earnings.refuse_entries_when_data_unavailable,
  default True) -- but an ETF's last-known-good already says "no earnings", so
  a later outage costs the ETF-only live universe nothing. Verified live
  against yfinance during planning: SPY/QQQ/IWM return an empty calendar dict
  (no exception), so they reach the confirmed-absence path, not the failure
  path, today.
- TTLCache gained an RLock (reentrant: set() calls _evict_expired() while
  already holding it).

Every test executes the real code path. No inspect.getsource.
"""

from __future__ import annotations

import ast
import threading
import time
from datetime import timedelta
from pathlib import Path

from ait.data.cache import TTLCache
from ait.data.earnings import EarningsCalendar, EarningsInfo
from ait.utils.time import today_et

SRC = Path(__file__).resolve().parents[1] / "src"


# ---------------------------------------------------------------------------
# 1. fail-direction-03 — a fetch failure must never poison the 6h cache
# ---------------------------------------------------------------------------

class TestFailedFetchNeverCached:
    """The finding's actual reproduction: pre-fix, a failure built the same
    EarningsInfo shape as confirmed-absence and both got cached for 6h.
    _fetch_earnings is stubbed at its own documented return contract
    (data_available=False on failure) -- that try/except is pre-existing and
    unchanged by this wave; what's new is get_next_earnings/_remember/
    _unavailable, which this exercises for real."""

    def test_failed_fetch_is_not_written_to_the_ttl_cache(self, monkeypatch):
        cal = EarningsCalendar()
        monkeypatch.setattr(
            cal, "_fetch_earnings",
            lambda symbol: EarningsInfo(symbol=symbol, next_earnings_date=None,
                                         data_available=False))

        info = cal.get_next_earnings("XYZ")

        assert info.data_available is False
        assert cal._cache.get("earnings_XYZ") is None, (
            "a failed fetch was written to the 6-hour TTL cache — the "
            "original defect this wave closes"
        )


# ---------------------------------------------------------------------------
# 2. fail-direction-03 — the three states, and the LKG-aware gate policy
# ---------------------------------------------------------------------------

class TestLkgAwareGatePolicy:
    def test_confirmed_no_earnings_allows_both_gates(self, monkeypatch):
        """The ETF shape: fetch succeeds, no date. This is the safety
        property the whole LKG-aware policy is built on — it must be
        explicit, not incidental."""
        cal = EarningsCalendar()
        monkeypatch.setattr(
            cal, "_fetch_earnings",
            lambda symbol: EarningsInfo(symbol=symbol, next_earnings_date=None))

        assert cal.is_near_earnings("SPY") is False
        assert cal.would_hold_through_earnings(
            "SPY", today_et(), today_et() + timedelta(days=21)) is False

    def test_lkg_with_a_real_date_still_blocks_after_a_later_failure(self, monkeypatch):
        cal = EarningsCalendar()
        earnings_date = today_et() + timedelta(days=1)
        monkeypatch.setattr(
            cal, "_fetch_earnings",
            lambda symbol: EarningsInfo(symbol=symbol, next_earnings_date=earnings_date))
        first = cal.get_next_earnings("AAPL")
        assert first.next_earnings_date == earnings_date

        # Simulate the 6h TTL cache expiring while the LKG persists — that
        # gap is the entire reason the two are kept separate.
        cal._cache.invalidate("earnings_AAPL")
        monkeypatch.setattr(
            cal, "_fetch_earnings",
            lambda symbol: EarningsInfo(symbol=symbol, next_earnings_date=None,
                                         data_available=False))

        assert cal.is_near_earnings("AAPL") is True
        assert cal.would_hold_through_earnings(
            "AAPL", today_et(), today_et() + timedelta(days=21)) is True

    def test_never_fetched_symbol_refuses_on_failure(self, monkeypatch):
        cal = EarningsCalendar()
        monkeypatch.setattr(
            cal, "_fetch_earnings",
            lambda symbol: EarningsInfo(symbol=symbol, next_earnings_date=None,
                                         data_available=False))

        assert cal.is_near_earnings("NEWSYM") is True
        assert cal.would_hold_through_earnings(
            "NEWSYM", today_et(), today_et() + timedelta(days=21)) is True

    def test_etf_lkg_survives_a_later_outage(self, monkeypatch):
        """The policy's whole point: an ETF's confirmed "no earnings"
        answer keeps it trading straight through a Yahoo outage."""
        cal = EarningsCalendar()
        monkeypatch.setattr(
            cal, "_fetch_earnings",
            lambda symbol: EarningsInfo(symbol=symbol, next_earnings_date=None))
        first = cal.get_next_earnings("SPY")
        assert first.data_available is True
        assert first.next_earnings_date is None

        cal._cache.invalidate("earnings_SPY")
        monkeypatch.setattr(
            cal, "_fetch_earnings",
            lambda symbol: EarningsInfo(symbol=symbol, next_earnings_date=None,
                                         data_available=False))

        assert cal.is_near_earnings("SPY") is False
        assert cal.would_hold_through_earnings(
            "SPY", today_et(), today_et() + timedelta(days=21)) is False

    def test_stale_lkg_beyond_max_age_falls_back_to_refuse(self, monkeypatch):
        cal = EarningsCalendar(lkg_max_age_hours=1)
        stale = EarningsInfo(symbol="SPY", next_earnings_date=None)
        cal._last_good["SPY"] = (stale, time.time() - 3700)  # just over 1h old
        monkeypatch.setattr(
            cal, "_fetch_earnings",
            lambda symbol: EarningsInfo(symbol=symbol, next_earnings_date=None,
                                         data_available=False))

        assert cal.is_near_earnings("SPY") is True, (
            "a stale last-known-good was served past its own max age"
        )

    def test_escape_hatch_restores_pre_w44_always_allow(self, monkeypatch):
        """The operator override: refuse_entries_when_data_unavailable=False
        must restore the exact pre-W44 behavior with no code deploy."""
        cal = EarningsCalendar(refuse_entries_when_data_unavailable=False)
        monkeypatch.setattr(
            cal, "_fetch_earnings",
            lambda symbol: EarningsInfo(symbol=symbol, next_earnings_date=None,
                                         data_available=False))

        assert cal.is_near_earnings("NEWSYM") is False
        assert cal.would_hold_through_earnings(
            "NEWSYM", today_et(), today_et() + timedelta(days=21)) is False


# ---------------------------------------------------------------------------
# 3. concurrency-4 — the deterministic race reproduction
# ---------------------------------------------------------------------------

class TestConcurrency4CacheRace:
    """Pre-fix (no lock), N threads concurrently calling set()/get() on the
    SAME TTLCache instance -- exactly earnings.py's background-fetch-thread
    pattern racing the main thread's own reads/writes -- can trip CPython's
    "dictionary changed size during iteration" inside _evict_expired's
    eviction sort, or silently corrupt _store. A tiny max_size forces every
    set() through the eviction path; the switch interval is lowered so an
    unlocked race is exposed reliably across the run rather than depending
    on one lucky interleaving (a single precisely-timed interleaving was
    tried first and did not reproduce reliably -- the interloper thread
    simply raced to completion inside the pause window instead of landing
    mid-iteration).

    Mutation-proof: remove TTLCache._lock (or any of its `with self._lock:`
    uses) and this test fails intermittently but reliably across repeated
    runs; with the lock, contending threads simply serialize and it cannot
    fail.
    """

    def test_concurrent_access_under_contention_does_not_raise_or_corrupt(self):
        import sys

        cache = TTLCache(default_ttl=300, max_size=8)
        errors: list[BaseException] = []
        stop = threading.Event()

        def _hammer(prefix: str) -> None:
            i = 0
            try:
                while not stop.is_set():
                    cache.set(f"{prefix}{i % 20}", i)
                    cache.get(f"{prefix}{i % 20}")
                    i += 1
            except BaseException as e:  # noqa: BLE001 — must capture, not raise, off-thread
                errors.append(e)

        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # maximize GIL handoff frequency during the race window
        try:
            threads = [threading.Thread(target=_hammer, args=(f"t{n}_",)) for n in range(6)]
            for t in threads:
                t.start()
            time.sleep(0.3)
            stop.set()
            for t in threads:
                t.join(timeout=2.0)
        finally:
            sys.setswitchinterval(old_interval)

        assert all(not t.is_alive() for t in threads), "a hammer thread never finished"
        assert errors == [], f"concurrent cache access raised: {errors!r}"


# ---------------------------------------------------------------------------
# 4. concurrency-4 — structural guard: every TTLCache method holds the lock
# ---------------------------------------------------------------------------

class TestConcurrency4CacheLockGuard:
    """STRUCTURAL guard, same category as W42's RANGE_GATED_STRATEGIES guard
    and W28's .today() guard: today's behavior is already covered by the
    race reproduction above, so this exists purely to catch FUTURE drift —
    a new or edited TTLCache method landing without the lock."""

    TARGET = SRC / "ait" / "data" / "cache.py"
    EXPECTED_METHODS = {"get", "set", "invalidate", "clear", "_evict_expired", "size"}

    @staticmethod
    def _has_lock_with(func_node: ast.AST) -> bool:
        return any(
            isinstance(n, ast.With)
            and any(
                isinstance(item.context_expr, ast.Attribute)
                and item.context_expr.attr == "_lock"
                for item in n.items
            )
            for n in ast.walk(func_node)
        )

    def _methods_missing_the_lock(self) -> tuple[list[str], int]:
        tree = ast.parse(self.TARGET.read_text(encoding="utf-8"))
        class_node = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == "TTLCache"
        )
        inspected = 0
        missing = []
        for node in class_node.body:
            if not isinstance(node, ast.FunctionDef) or node.name not in self.EXPECTED_METHODS:
                continue
            inspected += 1
            if not self._has_lock_with(node):
                missing.append(node.name)
        return missing, inspected

    def test_every_public_method_holds_the_lock(self):
        missing, inspected = self._methods_missing_the_lock()
        assert inspected == len(self.EXPECTED_METHODS), (
            f"expected to inspect {len(self.EXPECTED_METHODS)} methods "
            f"{sorted(self.EXPECTED_METHODS)}, only found {inspected} — "
            f"TTLCache's shape changed; update EXPECTED_METHODS"
        )
        assert missing == [], (
            f"{missing} do not acquire self._lock — a future edit to "
            f"TTLCache can silently reintroduce concurrency-4"
        )

    def test_the_guard_actually_detects_a_missing_lock(self):
        """Anti-vacuity: prove the detector fires on the exact shape it
        exists to catch, so a broken AST walk can't pass by finding nothing."""
        sample = ast.parse(
            "class TTLCache:\n"
            "    def get(self, key):\n"
            "        return self._store.get(key)\n"
        )
        class_node = next(
            n for n in ast.walk(sample)
            if isinstance(n, ast.ClassDef) and n.name == "TTLCache"
        )
        assert not self._has_lock_with(class_node.body[0]), (
            "the detector does not recognise an unlocked method"
        )

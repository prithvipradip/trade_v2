"""In-memory TTL cache.

Lightweight alternative to Redis — suitable for a single-process bot
running on a shared Mac. No external dependencies required.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class TTLCache:
    """Simple time-to-live cache with bounded size."""

    def __init__(self, default_ttl: int = 300, max_size: int = 1000) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        # W44 (concurrency-4): earnings.py sets on this cache from a
        # background thread (cold-cache miss inside a running event loop)
        # while the main thread concurrently get()s/set()s the SAME
        # instance. set()'s eviction sort and _evict_expired()'s
        # del-while-iterating are real race targets without a lock.
        # RLock, not Lock: set() calls _evict_expired() while already
        # holding it, from the same thread.
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get a cached value, or None if expired/missing."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expiry, value = entry
            if time.time() > expiry:
                self._store.pop(key, None)  # pop: expiry race must not KeyError
                return None
        # A11/L9: hand back a COPY of pandas objects — cached DataFrames were
        # returned by reference, so any consumer mutating one (adding a
        # column, sorting) silently poisoned every other consumer's view.
        # Copied outside the lock: .copy() can be slow on a large frame and
        # must not hold up other threads touching the cache.
        if type(value).__module__.startswith("pandas"):
            try:
                return value.copy()
            except Exception:  # noqa: BLE001
                return value
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a cached value with optional custom TTL."""
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._store) >= self._max_size:
                self._evict_expired()
                if len(self._store) >= self._max_size:
                    # Remove oldest 10%
                    entries = sorted(self._store.items(), key=lambda x: x[1][0])
                    for k, _ in entries[: max(1, self._max_size // 10)]:
                        del self._store[k]

            ttl = ttl if ttl is not None else self._default_ttl
            self._store[key] = (time.time() + ttl, value)

    def invalidate(self, key: str) -> None:
        """Remove a specific key from cache."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._store.clear()

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        with self._lock:
            now = time.time()
            expired = [k for k, (expiry, _) in self._store.items() if now > expiry]
            for k in expired:
                del self._store[k]

    @property
    def size(self) -> int:
        """Number of entries currently in cache (including expired)."""
        with self._lock:
            return len(self._store)

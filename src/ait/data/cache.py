"""In-memory TTL cache.

Lightweight alternative to Redis — suitable for a single-process bot
running on a shared Mac. No external dependencies required.
"""

from __future__ import annotations

import time
from typing import Any


class TTLCache:
    """Simple time-to-live cache with bounded size."""

    def __init__(self, default_ttl: int = 300, max_size: int = 1000) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size

    def get(self, key: str) -> Any | None:
        """Get a cached value, or None if expired/missing."""
        entry = self._store.get(key)
        if entry is None:
            return None
        expiry, value = entry
        if time.time() > expiry:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a cached value with optional custom TTL."""
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
        self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()

    def _evict_expired(self) -> None:
        """Remove all expired entries."""
        now = time.time()
        expired = [k for k, (expiry, _) in self._store.items() if now > expiry]
        for k in expired:
            del self._store[k]

    @property
    def size(self) -> int:
        """Number of entries currently in cache (including expired)."""
        return len(self._store)

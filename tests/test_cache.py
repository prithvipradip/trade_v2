"""Tests for TTLCache data component."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ait.data.cache import TTLCache


@pytest.fixture
def cache() -> TTLCache:
    return TTLCache(default_ttl=60, max_size=10)


class TestGetSet:
    """Test get/set operations."""

    def test_set_and_get(self, cache: TTLCache) -> None:
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self, cache: TTLCache) -> None:
        assert cache.get("nonexistent") is None

    def test_overwrite_value(self, cache: TTLCache) -> None:
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_various_types(self, cache: TTLCache) -> None:
        cache.set("int", 42)
        cache.set("list", [1, 2, 3])
        cache.set("dict", {"a": 1})
        cache.set("none", None)

        assert cache.get("int") == 42
        assert cache.get("list") == [1, 2, 3]
        assert cache.get("dict") == {"a": 1}
        # None values are stored but get() returns None for missing too
        # The stored tuple is (expiry, None), so it should return None
        assert cache.get("none") is None

    def test_size_tracks_entries(self, cache: TTLCache) -> None:
        assert cache.size == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size == 2

    def test_custom_ttl_per_key(self, cache: TTLCache) -> None:
        """Custom TTL on set should override default."""
        with patch("ait.data.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            cache.set("short", "val", ttl=5)
            cache.set("long", "val", ttl=300)

            # After 10 seconds: short expired, long still valid
            mock_time.time.return_value = 1011.0
            assert cache.get("short") is None
            assert cache.get("long") == "val"


class TestExpiry:
    """Test TTL expiry behavior."""

    def test_expired_returns_none(self, cache: TTLCache) -> None:
        with patch("ait.data.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            cache.set("key", "value")

            # Advance past TTL (60s default)
            mock_time.time.return_value = 1061.0
            assert cache.get("key") is None

    def test_not_expired_returns_value(self, cache: TTLCache) -> None:
        with patch("ait.data.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            cache.set("key", "value")

            # Still within TTL
            mock_time.time.return_value = 1050.0
            assert cache.get("key") == "value"

    def test_expired_entry_removed_from_store(self, cache: TTLCache) -> None:
        """Getting an expired key should remove it from internal store."""
        with patch("ait.data.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            cache.set("key", "value")

            mock_time.time.return_value = 1061.0
            cache.get("key")  # Triggers removal
            assert cache.size == 0


class TestEviction:
    """Test eviction when cache is full."""

    def test_evicts_when_full(self) -> None:
        """When at max_size, oldest entries should be evicted."""
        small_cache = TTLCache(default_ttl=300, max_size=5)

        with patch("ait.data.cache.time") as mock_time:
            # Fill cache with 5 entries, each with increasing expiry
            for i in range(5):
                mock_time.time.return_value = 1000.0 + i
                small_cache.set(f"key{i}", f"val{i}")

            assert small_cache.size == 5

            # Add one more — should evict oldest 10% (1 entry = key0)
            mock_time.time.return_value = 1010.0
            small_cache.set("new", "newval")

            assert small_cache.get("new") == "newval"
            # key0 had earliest expiry, should be evicted
            assert small_cache.get("key0") is None

    def test_evicts_expired_first(self) -> None:
        """Expired entries should be cleaned before evicting by age."""
        small_cache = TTLCache(default_ttl=10, max_size=5)

        with patch("ait.data.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            for i in range(5):
                small_cache.set(f"key{i}", f"val{i}")

            # Advance so all are expired
            mock_time.time.return_value = 1020.0
            small_cache.set("fresh", "freshval")

            # All expired entries should be gone, only fresh remains
            assert small_cache.size == 1
            assert small_cache.get("fresh") == "freshval"


class TestInvalidate:
    """Test invalidate operation."""

    def test_invalidate_removes_key(self, cache: TTLCache) -> None:
        cache.set("key", "value")
        cache.invalidate("key")
        assert cache.get("key") is None

    def test_invalidate_missing_key_no_error(self, cache: TTLCache) -> None:
        """Invalidating a non-existent key should not raise."""
        cache.invalidate("nonexistent")  # Should not raise

    def test_invalidate_reduces_size(self, cache: TTLCache) -> None:
        cache.set("a", 1)
        cache.set("b", 2)
        cache.invalidate("a")
        assert cache.size == 1

    def test_clear_removes_all(self, cache: TTLCache) -> None:
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

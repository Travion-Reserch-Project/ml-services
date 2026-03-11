"""
Multi-Tier Caching Layer for the Travion AI Engine.

Implements a thread-safe, TTL-based caching system with separate tiers
for different data freshness requirements:

    ┌───────────────────────────────────────────────────┐
    │                  TieredCache                      │
    │                                                   │
    │  Tier: weather_alerts    TTL = 60 min             │
    │  Tier: news_alerts       TTL = 60 min             │
    │  Tier: mcp_search        TTL = 4 hours            │
    │  Tier: event_sentinel    TTL = 24 hours           │
    │  Tier: golden_hour       TTL = 12 hours           │
    │                                                   │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
    │  │  L1 Memory │  │  L1 Memory │  │  L1 Memory │  │
    │  │  (dict)    │  │  (dict)    │  │  (dict)    │  │
    │  └────────────┘  └────────────┘  └────────────┘  │
    └───────────────────────────────────────────────────┘

Design Decisions:
    • Pure in-memory (dict-based) — no Redis dependency for dev/small
      deployments.  The ``settings.SESSION_BACKEND`` flag can later
      be used to swap in a Redis adapter.
    • Each tier has an independent TTL and max-size.
    • LRU eviction when a tier exceeds its max-size.
    • Thread-safe via ``threading.Lock`` (async-safe because GIL +
      dict operations are atomic for single puts/gets).

Performance:
    Cache hit returns in < 0.01 ms — well within the 3-second budget.
"""

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier Configuration
# ---------------------------------------------------------------------------
@dataclass
class CacheTierConfig:
    """Configuration for a single cache tier."""

    name: str
    ttl_seconds: float          # Time-to-live for entries
    max_size: int = 256         # Maximum number of entries (LRU eviction)


# Default tier configurations
DEFAULT_TIERS: Dict[str, CacheTierConfig] = {
    "weather_alerts": CacheTierConfig(
        name="weather_alerts",
        ttl_seconds=60 * 60,        # 60 minutes
        max_size=128,
    ),
    "news_alerts": CacheTierConfig(
        name="news_alerts",
        ttl_seconds=60 * 60,        # 60 minutes
        max_size=128,
    ),
    "mcp_search": CacheTierConfig(
        name="mcp_search",
        ttl_seconds=4 * 60 * 60,    # 4 hours
        max_size=512,
    ),
    "event_sentinel": CacheTierConfig(
        name="event_sentinel",
        ttl_seconds=24 * 60 * 60,   # 24 hours
        max_size=256,
    ),
    "golden_hour": CacheTierConfig(
        name="golden_hour",
        ttl_seconds=12 * 60 * 60,   # 12 hours
        max_size=256,
    ),
}


# ---------------------------------------------------------------------------
# Single Tier (LRU + TTL)
# ---------------------------------------------------------------------------
@dataclass
class _CacheEntry:
    """Internal cache entry with value and expiry timestamp."""

    value: Any
    expires_at: float   # time.time() epoch


class _CacheTier:
    """
    Single LRU + TTL cache tier backed by ``OrderedDict``.

    On every ``get`` and ``put`` the entry's timestamp is checked.
    Expired entries are lazily evicted on read miss.  When the tier
    exceeds ``max_size``, the least-recently-used entry is dropped.
    """

    def __init__(self, config: CacheTierConfig) -> None:
        self._config = config
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ---- public -----
    def get(self, key: str) -> Optional[Any]:
        """Return cached value or ``None`` if miss / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.time() > entry.expires_at:
                # Expired — evict
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, key: str, value: Any) -> None:
        """Insert or update an entry."""
        with self._lock:
            expires_at = time.time() + self._config.ttl_seconds
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = _CacheEntry(value=value, expires_at=expires_at)
            else:
                self._store[key] = _CacheEntry(value=value, expires_at=expires_at)
                # Evict LRU if over capacity
                while len(self._store) > self._config.max_size:
                    self._store.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> int:
        """Flush all entries. Returns count of evicted entries."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def prune_expired(self) -> int:
        """Actively remove all expired entries. Returns count."""
        now = time.time()
        pruned = 0
        with self._lock:
            expired_keys = [
                k for k, e in self._store.items() if now > e.expires_at
            ]
            for k in expired_keys:
                del self._store[k]
                pruned += 1
        return pruned

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "tier": self._config.name,
                "size": len(self._store),
                "max_size": self._config.max_size,
                "ttl_seconds": self._config.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": (
                    self._hits / (self._hits + self._misses)
                    if (self._hits + self._misses) > 0
                    else 0.0
                ),
            }


# ---------------------------------------------------------------------------
# Top-Level Tiered Cache
# ---------------------------------------------------------------------------
class TieredCache:
    """
    Multi-tier, thread-safe, in-memory cache.

    Each tier is independently configurable for TTL and max-size.
    Callers select the tier by name to apply the appropriate freshness
    policy.

    Usage::

        cache = TieredCache()

        # Store weather data (60-min TTL)
        cache.put("weather_alerts", "colombo", weather_json)

        # Store MCP search results (4-hour TTL)
        cache.put("mcp_search", "sha256_key", results_list)

        # Retrieve
        hit = cache.get("mcp_search", "sha256_key")
    """

    def __init__(
        self,
        tier_configs: Optional[Dict[str, CacheTierConfig]] = None,
    ) -> None:
        configs = tier_configs or DEFAULT_TIERS
        self._tiers: Dict[str, _CacheTier] = {
            name: _CacheTier(cfg) for name, cfg in configs.items()
        }

    def _tier(self, tier_name: str) -> _CacheTier:
        """Return the tier, creating a default one on the fly if needed."""
        if tier_name not in self._tiers:
            logger.warning(
                f"Cache tier '{tier_name}' not pre-configured — "
                f"creating with 30-min TTL default"
            )
            cfg = CacheTierConfig(name=tier_name, ttl_seconds=1800)
            self._tiers[tier_name] = _CacheTier(cfg)
        return self._tiers[tier_name]

    # ---- public API -----
    def get(self, tier_name: str, key: str) -> Optional[Any]:
        """Retrieve a cached value from the specified tier."""
        return self._tier(tier_name).get(key)

    def put(self, tier_name: str, key: str, value: Any) -> None:
        """Store a value in the specified tier."""
        self._tier(tier_name).put(key, value)

    def invalidate(self, tier_name: str, key: str) -> bool:
        """Remove a specific entry from a tier."""
        return self._tier(tier_name).invalidate(key)

    def clear_tier(self, tier_name: str) -> int:
        """Flush all entries in a tier."""
        return self._tier(tier_name).clear()

    def clear_all(self) -> int:
        """Flush every tier. Returns total entries evicted."""
        total = 0
        for tier in self._tiers.values():
            total += tier.clear()
        return total

    def prune_all(self) -> int:
        """Remove expired entries across all tiers."""
        total = 0
        for tier in self._tiers.values():
            total += tier.prune_expired()
        return total

    @property
    def stats(self) -> Dict[str, Any]:
        """Aggregate statistics for observability dashboards."""
        return {
            name: tier.stats for name, tier in self._tiers.items()
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_cache_instance: Optional[TieredCache] = None
_cache_lock = threading.Lock()


def get_tiered_cache() -> TieredCache:
    """Return the module-level singleton ``TieredCache``."""
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = TieredCache()
                logger.info(
                    "TieredCache initialised with tiers: "
                    f"{list(DEFAULT_TIERS.keys())}"
                )
    return _cache_instance

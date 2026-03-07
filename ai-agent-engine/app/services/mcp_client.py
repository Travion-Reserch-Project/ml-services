"""
Model Context Protocol (MCP) Client for Structured Tourism Data Retrieval.

This module implements a high-performance MCP client that connects to
specialised tourism data servers (Google Maps, Yelp, custom tourism APIs)
to retrieve **structured JSON** for Hotels, Restaurants, Events, and Parties.

Architecture:
    ┌───────────────────────────────────┐
    │         MCPTourismClient          │
    │  ┌─────────┐   ┌──────────────┐  │
    │  │PoolMgr  │   │ TieredCache  │  │
    │  └────┬────┘   └──────┬───────┘  │
    │       │               │          │
    │  ┌────▼────────────────▼───────┐  │
    │  │  MCP Server Adapters        │  │
    │  │  • Google Maps / Places     │  │
    │  │  • Yelp Fusion              │  │
    │  │  • Custom Tourism Data API  │  │
    │  └─────────────────────────────┘  │
    └───────────────────────────────────┘

Response Contract (per candidate):
    {
        "name": str,
        "exact_latitude": float,
        "exact_longitude": float,
        "price_range": "$" | "$$" | "$$$",
        "real_time_rating": float (0-5),
        "operational_hours": str,
        "description": str,
        "url": str | None,
        "source": str
    }

Research Pattern:
    Model Context Protocol (MCP, Anthropic 2024) — standardised interface
    between LLM applications and external data sources.  The client sends
    typed `tool_call` messages; the server responds with structured JSON.
    Connection pooling keeps latency under the 3-second budget.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Data-Classes
# ---------------------------------------------------------------------------
@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server endpoint."""

    name: str                        # Human-readable server name
    base_url: str                    # Server root URL
    api_key: Optional[str] = None    # Bearer / API-key header value
    timeout_seconds: float = 5.0     # Per-request timeout
    max_connections: int = 10        # Pool size for this server
    priority: int = 1                # Lower = higher priority
    capabilities: List[str] = field(
        default_factory=lambda: ["hotel", "restaurant", "bar", "event"]
    )


@dataclass
class MCPToolCall:
    """Typed tool-call message sent to an MCP server."""

    tool_name: str                   # e.g. "search_places"
    arguments: Dict[str, Any]        # Query arguments
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class MCPToolResult:
    """Typed tool-result message received from an MCP server."""

    call_id: str
    server_name: str
    results: List[Dict[str, Any]]    # Structured candidate dicts
    latency_ms: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Connection Pool Manager
# ---------------------------------------------------------------------------
class MCPConnectionPool:
    """
    Async connection-pool manager for MCP server requests.

    Uses ``aiohttp.TCPConnector`` with per-server connection limits to
    keep all external calls within the sub-3-second response budget.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._lock = asyncio.Lock()

    async def get_session(self, config: MCPServerConfig) -> aiohttp.ClientSession:
        """Return (or create) a pooled ``ClientSession`` for *config*."""
        async with self._lock:
            if config.name not in self._sessions:
                connector = aiohttp.TCPConnector(
                    limit=config.max_connections,
                    ttl_dns_cache=300,       # 5-min DNS cache
                    enable_cleanup_closed=True,
                )
                timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
                headers: Dict[str, str] = {"Content-Type": "application/json"}
                if config.api_key:
                    headers["Authorization"] = f"Bearer {config.api_key}"

                self._sessions[config.name] = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers=headers,
                )
            return self._sessions[config.name]

    async def close_all(self) -> None:
        """Gracefully close every pooled session."""
        async with self._lock:
            for session in self._sessions.values():
                await session.close()
            self._sessions.clear()


# ---------------------------------------------------------------------------
# MCP Tourism Client
# ---------------------------------------------------------------------------
class MCPTourismClient:
    """
    High-level MCP client for retrieving structured tourism data.

    The client fans out requests to multiple MCP-compatible servers in
    parallel, merges and de-duplicates results, then returns a uniform
    list of structured candidate dicts.

    Usage::

        client = MCPTourismClient(servers=[google_cfg, yelp_cfg])
        candidates = await client.search(
            search_type="restaurant",
            location="Galle, Sri Lanka",
            vibe="Local seafood",
            max_results=6,
        )
    """

    def __init__(
        self,
        servers: Optional[List[MCPServerConfig]] = None,
        cache: Optional[Any] = None,           # TieredCache instance
    ) -> None:
        self._servers = sorted(
            servers or self._default_servers(),
            key=lambda s: s.priority,
        )
        self._pool = MCPConnectionPool()
        self._cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def search(
        self,
        search_type: str,
        location: str,
        vibe: str = "",
        max_results: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Search all registered MCP servers for candidates.

        Args:
            search_type: "hotel", "restaurant", "bar", or "event"
            location: Textual location (e.g. "Unawatuna, Sri Lanka")
            vibe: Optional user vibe/preference string
            max_results: Maximum number of candidates to return

        Returns:
            List of structured candidate dicts conforming to the
            MCP response contract.
        """
        # ----- Check cache first -----
        cache_key = self._cache_key(search_type, location, vibe)
        if self._cache:
            cached = self._cache.get("mcp_search", cache_key)
            if cached is not None:
                logger.info(f"MCP cache hit for {cache_key}")
                return cached[:max_results]

        # ----- Fan-out to all capable servers -----
        eligible = [
            s for s in self._servers if search_type in s.capabilities
        ]
        if not eligible:
            logger.warning(f"No MCP servers configured for '{search_type}'")
            return []

        tool_call = MCPToolCall(
            tool_name="search_places",
            arguments={
                "type": search_type,
                "location": location,
                "query": vibe,
                "country": "Sri Lanka",
                "max_results": max_results,
                "fields": [
                    "name",
                    "exact_latitude",
                    "exact_longitude",
                    "price_range",
                    "real_time_rating",
                    "operational_hours",
                    "description",
                    "url",
                    "photos",
                ],
            },
        )

        tasks = [
            self._call_server(server, tool_call) for server in eligible
        ]
        results: List[MCPToolResult] = await asyncio.gather(
            *tasks, return_exceptions=False
        )

        # ----- Merge, de-duplicate, rank -----
        merged = self._merge_results(results, max_results)

        # ----- Populate cache -----
        if self._cache and merged:
            self._cache.put("mcp_search", cache_key, merged)

        return merged

    async def get_place_details(
        self,
        place_name: str,
        location: str,
        search_type: str = "hotel",
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed structured data for a single named place.

        Used during the Data Grounding step (Step C) to fill in any
        missing fields that the broad search did not return.
        """
        candidates = await self.search(
            search_type=search_type,
            location=location,
            vibe=place_name,
            max_results=1,
        )
        return candidates[0] if candidates else None

    async def close(self) -> None:
        """Release all connection-pool resources."""
        await self._pool.close_all()

    # ------------------------------------------------------------------
    # Private — Server Communication
    # ------------------------------------------------------------------
    async def _call_server(
        self,
        server: MCPServerConfig,
        tool_call: MCPToolCall,
    ) -> MCPToolResult:
        """
        Send an MCP ``tool_call`` to a single server and return the result.

        The wire protocol follows the MCP specification:
            POST /mcp/v1/tool
            {
                "tool_name": "search_places",
                "call_id": "...",
                "arguments": { ... }
            }
        """
        t0 = time.time()
        try:
            session = await self._pool.get_session(server)
            payload = {
                "tool_name": tool_call.tool_name,
                "call_id": tool_call.call_id,
                "arguments": tool_call.arguments,
            }

            async with session.post(
                f"{server.base_url}/mcp/v1/tool", json=payload
            ) as resp:
                latency_ms = (time.time() - t0) * 1000
                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("results", data.get("candidates", []))
                    # Normalise each result to the contract schema
                    normalised = [
                        self._normalise_candidate(r, server.name)
                        for r in results
                    ]
                    return MCPToolResult(
                        call_id=tool_call.call_id,
                        server_name=server.name,
                        results=normalised,
                        latency_ms=latency_ms,
                    )
                else:
                    body = await resp.text()
                    logger.warning(
                        f"MCP server {server.name} returned {resp.status}: "
                        f"{body[:200]}"
                    )
                    return MCPToolResult(
                        call_id=tool_call.call_id,
                        server_name=server.name,
                        results=[],
                        latency_ms=latency_ms,
                        error=f"HTTP {resp.status}",
                    )

        except asyncio.TimeoutError:
            latency_ms = (time.time() - t0) * 1000
            logger.warning(
                f"MCP server {server.name} timed out after {latency_ms:.0f}ms"
            )
            return MCPToolResult(
                call_id=tool_call.call_id,
                server_name=server.name,
                results=[],
                latency_ms=latency_ms,
                error="timeout",
            )
        except Exception as exc:
            latency_ms = (time.time() - t0) * 1000
            logger.error(
                f"MCP server {server.name} call failed: {exc}"
            )
            return MCPToolResult(
                call_id=tool_call.call_id,
                server_name=server.name,
                results=[],
                latency_ms=latency_ms,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Private — Normalisation & Merging
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_candidate(
        raw: Dict[str, Any], source: str
    ) -> Dict[str, Any]:
        """
        Normalise a raw server response into the MCP response contract.

        Handles variations in field names across different MCP servers
        (e.g. ``lat`` vs ``latitude`` vs ``exact_latitude``).
        """
        def _pick_float(*keys: str) -> Optional[float]:
            for k in keys:
                v = raw.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        continue
            return None

        def _pick_str(*keys: str) -> Optional[str]:
            for k in keys:
                v = raw.get(k)
                if v:
                    return str(v)
            return None

        return {
            "name": _pick_str("name", "title", "display_name") or "Unknown",
            "exact_latitude": _pick_float(
                "exact_latitude", "latitude", "lat", "geo_lat"
            ),
            "exact_longitude": _pick_float(
                "exact_longitude", "longitude", "lng", "lon", "geo_lng"
            ),
            "price_range": _pick_str(
                "price_range", "price_level", "price"
            ),
            "real_time_rating": _pick_float(
                "real_time_rating", "rating", "stars", "score"
            ),
            "operational_hours": _pick_str(
                "operational_hours", "opening_hours", "hours",
                "business_hours",
            ),
            "description": _pick_str(
                "description", "snippet", "content", "summary"
            ) or "",
            "url": _pick_str("url", "link", "web_url", "website"),
            "source": source,
            "place_id": _pick_str("place_id", "id", "yelp_id"),
            "phone": _pick_str("phone", "phone_number"),
            "image_url": _pick_str("image_url", "photo", "thumbnail"),
            "photo_urls": (
                raw.get("photo_urls")
                or raw.get("photos")
                or raw.get("images")
                or ([raw["image_url"]] if raw.get("image_url") else [])
            ) or [],
        }

    @staticmethod
    def _merge_results(
        results: List[MCPToolResult], max_results: int
    ) -> List[Dict[str, Any]]:
        """
        Merge results from multiple MCP servers, de-duplicate by name,
        and return the top *max_results* candidates ranked by rating.
        """
        seen_names: set = set()
        merged: List[Dict[str, Any]] = []

        # Flatten and de-duplicate
        for tr in results:
            if tr.error:
                continue
            for candidate in tr.results:
                norm_name = candidate["name"].strip().lower()
                if norm_name in seen_names:
                    continue
                seen_names.add(norm_name)
                merged.append(candidate)

        # Sort by rating (descending), then by source priority
        merged.sort(
            key=lambda c: (
                -(c.get("real_time_rating") or 0.0),
                c.get("source", ""),
            )
        )
        return merged[:max_results]

    # ------------------------------------------------------------------
    # Private — Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_key(search_type: str, location: str, vibe: str) -> str:
        raw = f"{search_type}|{location.lower().strip()}|{vibe.lower().strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    @staticmethod
    def _default_servers() -> List[MCPServerConfig]:
        """
        Return default MCP server configs read from environment / settings.

        In production these URLs come from ``settings.MCP_*`` env vars.
        The defaults below point to a localhost dev instance.
        """
        from ..config import settings

        servers: List[MCPServerConfig] = []

        # Google Maps MCP adapter
        if getattr(settings, "MCP_GOOGLE_MAPS_URL", None):
            servers.append(MCPServerConfig(
                name="google_maps",
                base_url=settings.MCP_GOOGLE_MAPS_URL,
                api_key=getattr(settings, "MCP_GOOGLE_MAPS_KEY", None),
                timeout_seconds=4.0,
                max_connections=10,
                priority=1,
                capabilities=["hotel", "restaurant", "bar", "event"],
            ))

        # Yelp Fusion MCP adapter
        if getattr(settings, "MCP_YELP_URL", None):
            servers.append(MCPServerConfig(
                name="yelp_fusion",
                base_url=settings.MCP_YELP_URL,
                api_key=getattr(settings, "MCP_YELP_KEY", None),
                timeout_seconds=4.0,
                max_connections=8,
                priority=2,
                capabilities=["restaurant", "bar", "hotel"],
            ))

        # Custom Tourism Data Server
        if getattr(settings, "MCP_TOURISM_URL", None):
            servers.append(MCPServerConfig(
                name="tourism_data",
                base_url=settings.MCP_TOURISM_URL,
                api_key=getattr(settings, "MCP_TOURISM_KEY", None),
                timeout_seconds=3.0,
                max_connections=12,
                priority=1,
                capabilities=["hotel", "restaurant", "bar", "event"],
            ))

        # Fallback: local dev MCP server
        if not servers:
            servers.append(MCPServerConfig(
                name="local_dev",
                base_url="http://localhost:8010",
                timeout_seconds=3.0,
                max_connections=5,
                priority=99,
                capabilities=["hotel", "restaurant", "bar", "event"],
            ))

        return servers


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------
_client_instance: Optional[MCPTourismClient] = None
_client_lock = asyncio.Lock()


async def get_mcp_client(
    cache: Optional[Any] = None,
) -> MCPTourismClient:
    """
    Return the module-level singleton ``MCPTourismClient``.

    The singleton is lazily initialised on first call.  Pass a
    ``TieredCache`` instance to enable the 4-hour MCP result cache.
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = MCPTourismClient(cache=cache)
    return _client_instance

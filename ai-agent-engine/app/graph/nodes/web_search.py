"""
Web Search Node: Real-Time Information Fallback.

This node is triggered when the Grader determines that local knowledge
is insufficient. It uses Tavily to search for real-time information.

Research Pattern:
    Retrieval-Augmented Generation with Fallback - When the primary
    knowledge source (ChromaDB) lacks information, the system
    transparently falls back to web search.

Use Cases:
    - Current weather conditions
    - Recent price changes
    - New attractions or closures
    - Events not in the calendar database
"""

import logging
import re
import time as _time_mod
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import quote_plus

# Tracing
try:
    from ...utils.tracing import trace_node
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    def trace_node(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from ..state import GraphState, ShadowMonitorLog
from ...tools.web_search import get_web_search_tool
from ...config import settings

logger = logging.getLogger(__name__)

# Map query patterns — route these to Google Maps MCP
MAP_QUERY_PATTERNS = [
    r"\b(show|find|open|view|see).*(on map|on google map|on maps)\b",
    r"\b(map of|map for)\b",
    r"\bdirections? to\b",
    r"\bhow (do i|to) (get|reach|go|travel) (to|from)\b",
    r"\broute (to|from|between)\b",
    r"\bnavigate (to|from)\b",
    r"\bwhere (is|are) .* (located|on the map)\b",
    r"\b(coordinates?|lat(itude)?|lon(gitude)?|gps) (of|for)\b",
    r"\bfind (me |us )?(on map|the location|the place)\b",
    r"\bnearby (places?|attractions?|restaurants?|hotels?)\b",
]


def is_map_query(query: str) -> bool:
    """Return True if the user is asking for map/location/directions data."""
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in MAP_QUERY_PATTERNS)


async def search_via_google_maps_mcp(
    query: str,
    target_location: Optional[str],
) -> list:
    """
    Call the Google Maps MCP server for place/location data.

    Returns a list of web_context-compatible dicts with title, content, url.
    Falls back to an empty list if the MCP server is unavailable.
    """
    try:
        import httpx
        from ...config import settings as cfg

        mcp_url = getattr(cfg, "MCP_GOOGLE_MAPS_URL", None)
        if not mcp_url:
            logger.warning("MCP_GOOGLE_MAPS_URL not configured — skipping MCP lookup")
            return []

        location_label = target_location or "Sri Lanka"
        payload = {
            "tool_name": "search_places",
            "arguments": {
                "type": "attraction",
                "location": location_label,
                "query": query,
                "country": "Sri Lanka",
                "max_results": 5,
                "fields": [
                    "name", "exact_latitude", "exact_longitude",
                    "formatted_address", "real_time_rating",
                    "operational_hours", "description", "url",
                ],
            },
        }

        api_key = getattr(cfg, "MCP_GOOGLE_MAPS_KEY", None)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key

        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(f"{mcp_url.rstrip('/')}/mcp/v1/tool", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for place in (data.get("results") or [])[:5]:
            name = place.get("name", "Unknown Place")
            address = place.get("formatted_address", location_label + ", Sri Lanka")
            lat = place.get("exact_latitude")
            lng = place.get("exact_longitude")
            desc = place.get("description", "")
            rating = place.get("real_time_rating")
            hours = place.get("operational_hours", "")

            # Build a Google Maps URL for this place
            maps_url = (
                f"https://www.google.com/maps/search/{quote_plus(name + ' ' + address)}"
                if not (lat and lng)
                else f"https://www.google.com/maps?q={lat},{lng}"
            )

            content_parts = [f"📍 {name}", f"📌 {address}"]
            if desc:
                content_parts.append(desc[:250])
            if rating:
                content_parts.append(f"⭐ Rating: {rating}/5")
            if hours:
                content_parts.append(f"🕐 Hours: {hours}")
            content_parts.append(f"🗺️ View on Google Maps: {maps_url}")

            results.append({
                "title": f"{name} — Google Maps",
                "content": "\n".join(content_parts),
                "url": maps_url,
                "source": "google_maps_mcp",
            })

        logger.info(f"Google Maps MCP returned {len(results)} places for: {query[:50]}")
        return results

    except Exception as exc:
        logger.warning(f"Google Maps MCP lookup failed: {exc}")
        return []


@trace_node("web_search", run_type="tool")
async def web_search_node(state: GraphState) -> GraphState:
    """
    Web Search Node: Fetch real-time information from the web.

    This node is triggered by the grader when local knowledge is insufficient.
    It performs a targeted web search and adds results to the context.

    Args:
        state: Current graph state

    Returns:
        Updated GraphState with web search results

    Research Note:
        The web search is transparent to the user - the agent acknowledges
        when it needs to look up external information, maintaining trust.
    """
    query = state["user_query"]
    target_location = state.get("target_location")

    _start = _time_mod.time()

    # --- Map query: route to Google Maps MCP instead of Tavily ---
    if is_map_query(query):
        logger.info(f"Map query detected — routing to Google Maps MCP: {query[:50]}")
        web_context = await search_via_google_maps_mcp(query, target_location)
        source_label = "Google Maps MCP"
    else:
        logger.info(f"Web Search (Tavily) triggered for: {query[:50]}...")

        web_search = get_web_search_tool(api_key=settings.TAVILY_API_KEY)

        # Build search query with location context
        search_query = query
        if target_location:
            search_query = f"{target_location} Sri Lanka {query}"

        results = web_search.search_tourism(search_query)
        web_context = []
        if results.get("results"):
            for r in results["results"][:3]:
                web_context.append({
                    "title": r.get("title", ""),
                    "content": r.get("content", "")[:500],
                    "url": r.get("url", ""),
                    "source": "web"
                })
        source_label = "Tavily"

    # Log the search
    log_entry = ShadowMonitorLog(
        timestamp=datetime.now().isoformat(),
        check_type="web_search",
        input_context={
            "query": query,
            "target_location": target_location,
            "source": source_label,
        },
        result="success" if web_context else "fallback",
        details=f"Found {len(web_context)} results | Source: {source_label}",
        action_taken="augment_context"
    )

    _duration_ms = (_time_mod.time() - _start) * 1000
    return {
        **state,
        "web_search_results": web_context,
        "step_results": [{
            "node": "web_search",
            "status": "success" if web_context else "warning",
            "summary": f"Found {len(web_context)} results | Source: {source_label} | Query: {query[:50]}",
            "duration_ms": round(_duration_ms, 2),
        }],
        "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry]
    }

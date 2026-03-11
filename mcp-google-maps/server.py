"""
Google Maps MCP Server — Model Context Protocol adapter for Google Places API.

Exposes a single MCP-compatible endpoint (POST /mcp/v1/tool) that the
MCPTourismClient in the AI Engine can call.  The server translates
MCP tool_call messages into Google Places API (New) HTTP Text Search
requests and returns normalised structured JSON.

Requires:
    GOOGLE_MAPS_API_KEY — a key with **Places API (New)** enabled in
    Google Cloud Console.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8010
"""

import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-google-maps")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"

# ── Simple in-memory cache (TTL = 4 hours) ──
_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = 4 * 60 * 60  # 4 hours


def _cache_key(query: str, location: str) -> str:
    raw = f"{query.lower().strip()}|{location.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _cache_get(key: str) -> Optional[List[Dict[str, Any]]]:
    entry = _CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["data"]
    if entry:
        del _CACHE[key]
    return None


def _cache_put(key: str, data: List[Dict[str, Any]]) -> None:
    # Evict oldest if cache grows too large (keep max 500 entries)
    if len(_CACHE) >= 500:
        oldest_key = min(_CACHE, key=lambda k: _CACHE[k]["ts"])
        del _CACHE[oldest_key]
    _CACHE[key] = {"data": data, "ts": time.time()}


# ── Pydantic models ──

class MCPToolCallRequest(BaseModel):
    tool_name: str
    call_id: str = ""
    arguments: Dict[str, Any] = {}


class MCPToolCallResponse(BaseModel):
    call_id: str
    results: List[Dict[str, Any]]
    server: str = "google_maps_mcp"
    latency_ms: float = 0.0
    error: Optional[str] = None


# ── FastAPI App ──

app = FastAPI(
    title="Google Maps MCP Server",
    description="MCP adapter for Google Places API (New)",
    version="1.0.0",
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "server": "google_maps_mcp",
        "has_api_key": bool(GOOGLE_MAPS_API_KEY),
    }


@app.post("/mcp/v1/tool", response_model=MCPToolCallResponse)
async def mcp_tool_call(request: MCPToolCallRequest):
    """
    MCP-compatible tool endpoint.

    Supports tool_name: "search_places"
    Arguments:
        type: "restaurant" | "hotel" | "bar" | "event"
        location: e.g. "Galle, Sri Lanka"
        query: optional vibe/keyword
        max_results: int (default 6)
    """
    t0 = time.time()

    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_MAPS_API_KEY not configured on the MCP server",
        )

    if request.tool_name != "search_places":
        return MCPToolCallResponse(
            call_id=request.call_id,
            results=[],
            error=f"Unknown tool: {request.tool_name}",
        )

    args = request.arguments
    search_type = args.get("type", "restaurant")
    location = args.get("location", "Sri Lanka")
    query_vibe = args.get("query", "")
    country = args.get("country", "Sri Lanka")
    max_results = min(args.get("max_results", 6), 20)

    # Build text query
    text_query = f"{search_type}s near {location}"
    if query_vibe:
        text_query = f"{query_vibe} {text_query}"
    if country and country.lower() not in location.lower():
        text_query += f", {country}"

    # ── Check cache ──
    ck = _cache_key(text_query, location)
    cached = _cache_get(ck)
    if cached is not None:
        logger.info(f"Cache HIT for '{text_query}' ({len(cached)} results)")
        latency_ms = (time.time() - t0) * 1000
        return MCPToolCallResponse(
            call_id=request.call_id,
            results=cached[:max_results],
            latency_ms=latency_ms,
        )

    # ── Call Google Places API (New) Text Search ──
    # https://developers.google.com/maps/documentation/places/web-service/text-search
    requested_fields = [
        "places.id",
        "places.displayName",
        "places.formattedAddress",
        "places.location",
        "places.rating",
        "places.userRatingCount",
        "places.priceLevel",
        "places.websiteUri",
        "places.regularOpeningHours",
        "places.photos",
        "places.editorialSummary",
        "places.primaryType",
    ]

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": ",".join(requested_fields),
    }

    body = {
        "textQuery": text_query,
        "maxResultCount": max_results,
        "languageCode": "en",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(PLACES_TEXT_SEARCH_URL, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        latency_ms = (time.time() - t0) * 1000
        logger.error(f"Google Places API error: {exc.response.status_code} — {exc.response.text[:300]}")
        return MCPToolCallResponse(
            call_id=request.call_id,
            results=[],
            latency_ms=latency_ms,
            error=f"Google API HTTP {exc.response.status_code}",
        )
    except Exception as exc:
        latency_ms = (time.time() - t0) * 1000
        logger.error(f"Google Places API call failed: {exc}")
        return MCPToolCallResponse(
            call_id=request.call_id,
            results=[],
            latency_ms=latency_ms,
            error=str(exc),
        )

    # ── Normalise results to MCP contract ──
    places = data.get("places", [])
    results: List[Dict[str, Any]] = []

    for place in places:
        loc = place.get("location", {})
        display_name = place.get("displayName", {})
        editorial = place.get("editorialSummary", {})
        hours = place.get("regularOpeningHours", {})

        # Build photo URL (first photo)
        photo_url = None
        photo_urls = []
        photos = place.get("photos", [])
        for p in photos[:3]:
            photo_name = p.get("name", "")
            if photo_name:
                # Google Places (New) photo reference URL
                url = (
                    f"https://places.googleapis.com/v1/{photo_name}/media"
                    f"?maxWidthPx=400&key={GOOGLE_MAPS_API_KEY}"
                )
                photo_urls.append(url)
                if not photo_url:
                    photo_url = url

        # Price level mapping
        price_level_map = {
            "PRICE_LEVEL_FREE": "$",
            "PRICE_LEVEL_INEXPENSIVE": "$",
            "PRICE_LEVEL_MODERATE": "$$",
            "PRICE_LEVEL_EXPENSIVE": "$$$",
            "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
        }
        raw_price = place.get("priceLevel", "")
        price_range = price_level_map.get(raw_price, "$$")

        results.append({
            "name": display_name.get("text", "Unknown"),
            "place_id": place.get("id"),
            "exact_latitude": loc.get("latitude"),
            "exact_longitude": loc.get("longitude"),
            "price_range": price_range,
            "real_time_rating": place.get("rating"),
            "user_rating_count": place.get("userRatingCount"),
            "operational_hours": (
                "; ".join(hours.get("weekdayDescriptions", [])[:3])
                if hours.get("weekdayDescriptions")
                else None
            ),
            "description": editorial.get("text", ""),
            "url": place.get("websiteUri"),
            "formatted_address": place.get("formattedAddress"),
            "primary_type": place.get("primaryType"),
            "image_url": photo_url,
            "photo_urls": photo_urls,
        })

    # ── Cache results ──
    if results:
        _cache_put(ck, results)

    latency_ms = (time.time() - t0) * 1000
    logger.info(
        f"Google Places returned {len(results)} results for '{text_query}' "
        f"in {latency_ms:.0f}ms"
    )

    return MCPToolCallResponse(
        call_id=request.call_id,
        results=results,
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)

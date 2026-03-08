"""
MCP-Powered Advanced Multi-Step Search & Selection Node.

This node implements a three-stage, MCP-driven search pipeline that
retrieves **structured JSON** for Hotels, Restaurants, Events, and
Parties via the Model Context Protocol.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │  Step A — MCP Structured Retrieval                      │
    │  Fan-out to Google Maps / Yelp / Tourism MCP servers    │
    │  → 6-8 candidates with exact coordinates & metadata     │
    │                                                         │
    │  Step B — LLM Semantic Ranking                          │
    │  Score each candidate against user vibe/preferences     │
    │  → Top 4 ranked candidates                              │
    │                                                         │
    │  Step C — Data Enrichment & Grounding                   │
    │  Fill missing fields via secondary MCP detail calls     │
    │  → Fully grounded SearchCandidate objects               │
    └─────────────────────────────────────────────────────────┘

After completion the node sets ``pending_user_selection = True`` and
returns a ``SELECTION_REQUIRED`` state with a list of **Selection Cards**
(JSON) ready for the React Native frontend.  The LangGraph
``interrupt_before`` on ``selection_handler`` causes the graph to pause
until the user chooses a candidate.

Research Pattern:
    Model Context Protocol (MCP, Anthropic 2024) replaces the previous
    Tavily-based web scraping with structured, schema-conformant data
    retrieval.  Multi-server fan-out + 4-hour tiered cache keeps
    latency under the 3-second budget while minimising redundant API
    calls and cost.
"""

import logging
import json
import re
import time
import uuid
from typing import Dict, List, Optional, Any

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

from ..state import GraphState, SearchCandidate, StepResult
from ...config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search-type keywords
# ---------------------------------------------------------------------------
SEARCH_KEYWORDS = {
    "hotel": [
        "hotel", "hotels", "accommodation", "stay", "lodge", "hostel",
        "resort", "guesthouse", "airbnb", "room", "where to stay",
    ],
    "restaurant": [
        "restaurant", "restaurants", "dining", "eat", "food", "cafe",
        "cafes", "cuisine", "vegan", "vegetarian", "lunch", "dinner",
        "breakfast", "brunch", "where to eat",
    ],
    "bar": [
        "bar", "bars", "nightlife", "party", "club", "pub", "cocktail",
        "drinks", "lounge",
    ],
    "event": [
        "event", "events", "festival", "show", "concert", "performance",
        "exhibition", "market", "fair",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def detect_search_type(query: str) -> str:
    """Classify the search type from user query text."""
    q = query.lower()
    for stype, keywords in SEARCH_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return stype
    return "hotel"


def should_trigger_advanced_search(query: str) -> bool:
    """Return True if the query contains any search-trigger keyword."""
    q = query.lower()
    return any(kw in q for kw in sum(SEARCH_KEYWORDS.values(), []))


def _extract_location(state: GraphState) -> str:
    """Extract the best-available location from state."""
    loc = state.get("target_location")
    if loc:
        return loc
    ctx = state.get("tour_plan_context")
    if ctx:
        locs = ctx.get("selected_locations", [])
        if locs:
            return locs[0].get("name", "Sri Lanka")
    return "Sri Lanka"


def _extract_vibe(state: GraphState) -> str:
    """Derive a vibe string from user preferences or query keywords."""
    prefs = state.get("user_preferences") or {}
    parts: List[str] = []
    if prefs.get("relaxation", 0) >= 0.6:
        parts.append("Luxury")
    if prefs.get("adventure", 0) >= 0.6:
        parts.append("Adventure")
    if prefs.get("nature", 0) >= 0.6:
        parts.append("Eco-friendly")
    if prefs.get("history", 0) >= 0.6:
        parts.append("Cultural")

    q = state.get("user_query", "").lower()
    if "vegan" in q:
        parts.append("Vegan")
    if "local" in q or "authentic" in q:
        parts.append("Local")
    if "nightlife" in q or "party" in q:
        parts.append("Nightlife")
    if "budget" in q or "cheap" in q:
        parts.append("Budget")
    if "luxury" in q or "premium" in q:
        parts.append("Luxury")

    return ", ".join(parts) if parts else "General"


# ---------------------------------------------------------------------------
# Step A — MCP Structured Retrieval
# ---------------------------------------------------------------------------
async def _step_a_mcp_search(
    search_type: str,
    location: str,
    vibe: str,
) -> List[Dict[str, Any]]:
    """
    Step A: Structured MCP Search — fan-out to registered MCP servers
    (Google Maps, Yelp, custom tourism data) and collect 6-8 candidates
    with exact coordinates, pricing, ratings, and operational hours.

    Falls back to Tavily web search if no MCP servers respond.
    """
    from ...services.mcp_client import get_mcp_client
    from ...services.tiered_cache import get_tiered_cache

    cache = get_tiered_cache()

    try:
        client = await get_mcp_client(cache=cache)
        results = await client.search(
            search_type=search_type,
            location=location,
            vibe=vibe,
            max_results=8,
        )
        if results:
            logger.info(
                f"MCP Step A: {len(results)} structured results "
                f"from MCP servers for {search_type} near {location}"
            )
            return results
    except Exception as exc:
        logger.warning(f"MCP search failed, falling back to Tavily: {exc}")

    # ------- Fallback: Tavily web search -------
    return await _step_a_tavily_fallback(search_type, location)


async def _step_a_tavily_fallback(
    search_type: str,
    location: str,
) -> List[Dict[str, Any]]:
    """Fallback to Tavily when MCP servers are unavailable."""
    try:
        from ...tools.web_search import get_web_search_tool
    except ImportError:
        logger.error("Tavily web_search tool not available")
        return []

    web = get_web_search_tool(settings.TAVILY_API_KEY)
    if not web.enabled:
        return []

    query_map = {
        "hotel": f"best hotels resorts guesthouses near {location} Sri Lanka 2026 reviews price",
        "restaurant": f"best restaurants cafes dining near {location} Sri Lanka cuisine reviews",
        "bar": f"best bars nightlife clubs near {location} Sri Lanka reviews",
        "event": f"upcoming events festivals shows near {location} Sri Lanka 2026",
    }
    search_query = query_map.get(search_type, query_map["hotel"])

    SEARCH_DOMAINS: Dict[str, List[str]] = {
        "hotel": ["booking.com", "tripadvisor.com", "agoda.com", "hotels.com"],
        "restaurant": ["tripadvisor.com", "yelp.com", "lonelyplanet.com"],
        "bar": ["tripadvisor.com", "lonelyplanet.com", "timeout.com"],
        "event": ["tripadvisor.com", "lonelyplanet.com", "eventbrite.com"],
    }
    domains = SEARCH_DOMAINS.get(search_type, SEARCH_DOMAINS["hotel"])

    try:
        original_max = web.max_results
        web.max_results = 8
        raw = web.search(search_query, include_domains=domains)
        web.max_results = original_max

        # Convert Tavily results into the MCP contract shape
        results: List[Dict[str, Any]] = []
        for r in raw.get("results", []):
            results.append({
                "name": r.get("title", "Unknown"),
                "exact_latitude": None,
                "exact_longitude": None,
                "price_range": None,
                "real_time_rating": None,
                "operational_hours": None,
                "description": r.get("content", "")[:300],
                "url": r.get("url"),
                "source": "tavily_fallback",
            })
        return results
    except Exception as exc:
        logger.error(f"Tavily fallback search failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# Step B — LLM Semantic Ranking
# ---------------------------------------------------------------------------
async def _step_b_semantic_ranking(
    candidates: List[Dict[str, Any]],
    search_type: str,
    vibe: str,
    llm,
) -> List[Dict[str, Any]]:
    """
    Step B: Semantic Ranking — Use LLM to score and rank candidates
    against the user's vibe/preferences.

    Returns the top 4 candidates with an added ``vibe_match_score``.
    """
    if not llm or not candidates:
        return candidates[:4]

    summaries = []
    for i, c in enumerate(candidates):
        rating_str = f"★{c.get('real_time_rating')}/5" if c.get("real_time_rating") else "N/A"
        price_str = c.get("price_range") or "N/A"
        summaries.append(
            f"{i + 1}. {c['name']} ({price_str}, {rating_str}) — "
            f"{(c.get('description') or '')[:180]}"
        )
    listing = "\n".join(summaries)

    prompt = f"""You are a Sri Lankan tourism expert. A traveler wants the best {search_type} options.
Their vibe / preference: {vibe}

Here are the candidates:
{listing}

TASK: Return a JSON array of objects, one per candidate, with:
  - "index": 1-based index from the list above
  - "score": float 0-1 indicating vibe match (1 = perfect match)

Order the array from BEST match to LEAST match.
Only include the TOP 4. Output ONLY the JSON array, nothing else.
Example: [{{"index":3,"score":0.95}},{{"index":1,"score":0.82}},{{"index":5,"score":0.71}},{{"index":2,"score":0.60}}]"""

    try:
        from langchain_core.messages import HumanMessage
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        text = resp.content.strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            scored = json.loads(match.group())
            ranked: List[Dict[str, Any]] = []
            for entry in scored[:4]:
                idx = entry.get("index", 0)
                score = entry.get("score", 0.5)
                if 1 <= idx <= len(candidates):
                    cand = {**candidates[idx - 1], "vibe_match_score": score}
                    ranked.append(cand)
            if ranked:
                return ranked
    except Exception as exc:
        logger.warning(f"Step B LLM ranking failed, using fallback: {exc}")

    return candidates[:4]


# ---------------------------------------------------------------------------
# Step C — Data Enrichment & Grounding
# ---------------------------------------------------------------------------
async def _step_c_data_grounding(
    ranked: List[Dict[str, Any]],
    search_type: str,
    location: str,
    llm,
) -> List[SearchCandidate]:
    """
    Step C: Data Enrichment — For candidates with missing coordinates
    or metadata, make targeted MCP detail calls to fill gaps.

    Returns a list of fully-grounded ``SearchCandidate`` objects.
    """
    from ...services.mcp_client import get_mcp_client
    from ...services.tiered_cache import get_tiered_cache

    cache = get_tiered_cache()
    candidates: List[SearchCandidate] = []

    for item in ranked[:4]:
        name = item.get("name", "Unknown Place")
        lat = item.get("exact_latitude")
        lng = item.get("exact_longitude")
        rating = item.get("real_time_rating")
        price = item.get("price_range")
        hours = item.get("operational_hours")

        # If critical fields are missing, attempt a focused detail lookup
        if lat is None or lng is None or rating is None:
            try:
                client = await get_mcp_client(cache=cache)
                detail = await client.get_place_details(
                    place_name=name,
                    location=location,
                    search_type=search_type,
                )
                if detail:
                    lat = lat or detail.get("exact_latitude")
                    lng = lng or detail.get("exact_longitude")
                    rating = rating or detail.get("real_time_rating")
                    price = price or detail.get("price_range")
                    hours = hours or detail.get("operational_hours")
            except Exception as exc:
                logger.warning(f"MCP detail lookup for '{name}' failed: {exc}")

        # If still missing lat/lng, try LLM extraction from description
        if (lat is None or lng is None) and llm:
            extracted = await _extract_coords_with_llm(
                name, location, item.get("description", ""), llm
            )
            lat = lat or extracted.get("lat")
            lng = lng or extracted.get("lng")

        # Heuristic fallbacks
        if not price:
            price = _heuristic_price(item.get("description", ""))
        if not rating:
            rating = _heuristic_rating(item.get("description", ""))

        cid = f"{search_type}_{uuid.uuid4().hex[:8]}"

        # Resolve photo URLs from MCP data
        photo_urls = item.get("photo_urls") or []
        if not photo_urls and item.get("image_url"):
            photo_urls = [item["image_url"]]

        candidates.append(SearchCandidate(
            id=cid,
            name=name,
            type=search_type,
            description=(item.get("description") or "")[:300].strip(),
            price_range=price,
            rating=_safe_float(rating),
            opening_hours=hours,
            lat=_safe_float(lat),
            lng=_safe_float(lng),
            url=item.get("url"),
            location_name=location,
            vibe_match_score=item.get("vibe_match_score"),
            photo_urls=photo_urls,
        ))

    return candidates


async def _extract_coords_with_llm(
    name: str, location: str, text: str, llm
) -> Dict[str, Optional[float]]:
    """Last-resort LLM extraction for coordinates."""
    prompt = f"""What are the approximate GPS coordinates (latitude, longitude) of "{name}" near {location}, Sri Lanka?
Return ONLY valid JSON: {{"lat": <float>, "lng": <float>}}
If unknown, return: {{"lat": null, "lng": null}}"""
    try:
        from langchain_core.messages import HumanMessage
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()
        match = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {"lat": _safe_float(data.get("lat")), "lng": _safe_float(data.get("lng"))}
    except Exception as exc:
        logger.warning(f"LLM coord extraction failed for {name}: {exc}")
    return {"lat": None, "lng": None}


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _heuristic_price(text: str) -> Optional[str]:
    t = text.lower()
    if any(w in t for w in ["luxury", "premium", "$$$", "high-end", "5-star", "five star"]):
        return "$$$"
    if any(w in t for w in ["mid-range", "moderate", "$$", "3-star", "4-star"]):
        return "$$"
    if any(w in t for w in ["budget", "cheap", "affordable", "$", "hostel", "backpacker"]):
        return "$"
    return "$$"


def _heuristic_rating(text: str) -> Optional[float]:
    match = re.search(r'(\d\.?\d?)\s*/?\s*5', text)
    if match:
        try:
            val = float(match.group(1))
            if 0 <= val <= 5:
                return val
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Selection Card Builder
# ---------------------------------------------------------------------------
def _build_selection_cards(
    candidates: List[SearchCandidate],
    search_type: str,
    location: str,
    vibe: str,
) -> List[Dict[str, Any]]:
    """
    Build structured Selection Card JSON objects for the React Native
    frontend.  Each card contains all information needed for the user
    to make an informed choice.

    Schema per card::

        {
            "card_id": str,
            "name": str,
            "type": str,
            "price_range": str | None,
            "real_time_rating": float | None,
            "operational_hours": str | None,
            "exact_latitude": float | None,
            "exact_longitude": float | None,
            "description": str,
            "url": str | None,
            "vibe_match_score": float | None,
            "location_name": str,
            "badge": str | None        # e.g. "Best Match", "Budget Pick"
        }
    """
    cards: List[Dict[str, Any]] = []

    for i, c in enumerate(candidates):
        badge = None
        if i == 0:
            badge = "Best Match"
        elif c.get("price_range") == "$":
            badge = "Budget Pick"
        elif (c.get("rating") or 0) >= 4.5:
            badge = "Top Rated"

        cards.append({
            "card_id": c["id"],
            "name": c["name"],
            "type": c["type"],
            "price_range": c.get("price_range"),
            "real_time_rating": c.get("rating"),
            "operational_hours": c.get("opening_hours"),
            "exact_latitude": c.get("lat"),
            "exact_longitude": c.get("lng"),
            "description": c.get("description", ""),
            "url": c.get("url"),
            "vibe_match_score": c.get("vibe_match_score"),
            "location_name": c.get("location_name", location),
            "badge": badge,
            "photo_urls": c.get("photo_urls", []),
            "image_url": (c.get("photo_urls") or [None])[0],
        })

    return cards


# ---------------------------------------------------------------------------
# Main Node Entry Point
# ---------------------------------------------------------------------------
@trace_node("advanced_search", run_type="tool")
async def advanced_search_node(state: GraphState, llm=None) -> GraphState:
    """
    MCP-Powered Advanced Multi-Step Search Node.

    Orchestrates a three-stage pipeline:
        Step A → MCP structured retrieval (6-8 candidates with coordinates)
        Step B → LLM semantic ranking (top 4 with vibe_match_score)
        Step C → Data enrichment & grounding (fill missing fields)

    On completion sets:
        - ``search_candidates``:  list of fully-grounded SearchCandidate dicts
        - ``selection_cards``:    list of Selection Card JSON (for React Native)
        - ``pending_user_selection = True``
        - ``mcp_search_metadata``: latency & source diagnostics

    The graph's ``interrupt_before`` on ``selection_handler`` causes
    execution to pause here, returning a ``SELECTION_REQUIRED`` state
    to the mobile app.

    Args:
        state: Current GraphState
        llm: LangChain LLM instance

    Returns:
        Updated GraphState with SELECTION_REQUIRED payload
    """
    start = time.time()
    query = state.get("user_query", "")
    location = _extract_location(state)
    search_type = detect_search_type(query)
    vibe = _extract_vibe(state)

    logger.info(
        f"Advanced MCP Search — type={search_type}, "
        f"location={location}, vibe={vibe}"
    )

    step_results: List[StepResult] = []

    # ---- Step A: MCP Structured Retrieval ----
    t0 = time.time()
    raw_candidates = await _step_a_mcp_search(search_type, location, vibe)
    mcp_source = (
        "mcp"
        if raw_candidates and raw_candidates[0].get("source") != "tavily_fallback"
        else "tavily_fallback"
    )
    step_results.append({
        "node": "advanced_search_step_a",
        "status": "success" if raw_candidates else "warning",
        "summary": (
            f"MCP retrieval returned {len(raw_candidates)} structured "
            f"candidates via {mcp_source} for {search_type}s near {location}"
        ),
        "duration_ms": (time.time() - t0) * 1000,
    })
    logger.info(f"Step A: {len(raw_candidates)} raw candidates (source: {mcp_source})")

    if not raw_candidates:
        duration_ms = (time.time() - start) * 1000
        return {
            **state,
            "search_candidates": [],
            "selection_cards": [],
            "pending_user_selection": False,
            "mcp_search_metadata": {
                "source": mcp_source,
                "total_latency_ms": duration_ms,
                "candidates_found": 0,
            },
            "generated_response": (
                f"I couldn't find any {search_type} options near {location}. "
                f"Try a more specific location."
            ),
            "final_response": (
                f"I couldn't find any {search_type} options near {location}. "
                f"Try a more specific location."
            ),
            "step_results": step_results + [{
                "node": "advanced_search",
                "status": "warning",
                "summary": "No results found from MCP or fallback",
                "duration_ms": duration_ms,
            }],
        }

    # ---- Step B: LLM Semantic Ranking ----
    t1 = time.time()
    ranked = await _step_b_semantic_ranking(
        raw_candidates, search_type, vibe, llm
    )
    step_results.append({
        "node": "advanced_search_step_b",
        "status": "success",
        "summary": (
            f"LLM ranked {len(ranked)} candidates by vibe: {vibe}"
        ),
        "duration_ms": (time.time() - t1) * 1000,
    })
    logger.info(f"Step B: {len(ranked)} ranked candidates")

    # ---- Step C: Data Enrichment & Grounding ----
    t2 = time.time()
    candidates = await _step_c_data_grounding(
        ranked, search_type, location, llm
    )
    grounded_count = sum(
        1 for c in candidates if c.get("lat") and c.get("lng")
    )
    step_results.append({
        "node": "advanced_search_step_c",
        "status": "success",
        "summary": (
            f"Grounded {len(candidates)} candidates "
            f"({grounded_count} with GPS coordinates)"
        ),
        "duration_ms": (time.time() - t2) * 1000,
    })
    logger.info(f"Step C: {len(candidates)} grounded ({grounded_count} with coords)")

    # ---- Build Selection Cards for React Native ----
    selection_cards = _build_selection_cards(
        candidates, search_type, location, vibe
    )

    duration_ms = (time.time() - start) * 1000

    # ---- Build human-readable response ----
    if candidates:
        type_label = {
            "hotel": "accommodation",
            "restaurant": "dining",
            "bar": "nightlife",
            "event": "event",
        }.get(search_type, search_type)

        parts = [
            f"I found {len(candidates)} curated {type_label} options near "
            f"**{location}** matching your vibe (*{vibe}*).\n"
            f"Please select one to add to your itinerary:\n"
        ]
        for i, c in enumerate(candidates, 1):
            line = f"**{i}. {c['name']}**"
            if c.get("price_range"):
                line += f"  ({c['price_range']})"
            if c.get("rating"):
                line += f"  ★ {c['rating']}/5"
            if c.get("vibe_match_score"):
                pct = int(c["vibe_match_score"] * 100)
                line += f"  [{pct}% match]"
            line += f"\n   {c['description']}"
            if c.get("opening_hours"):
                line += f"\n   Hours: {c['opening_hours']}"
            if c.get("lat") and c.get("lng"):
                line += f"\n   📍 {c['lat']:.4f}, {c['lng']:.4f}"
            if c.get("url"):
                line += f"\n   [Details]({c['url']})"
            parts.append(line)

        response_text = "\n\n".join(parts)
    else:
        response_text = (
            f"I couldn't find well-matched {search_type} options near "
            f"{location}. Try broadening your search."
        )

    # Convert SearchCandidate TypedDicts to plain dicts for JSON safety
    candidate_dicts = [dict(c) for c in candidates]

    # ---- Build MCP metadata for diagnostics ----
    mcp_metadata = {
        "source": mcp_source,
        "total_latency_ms": round(duration_ms, 1),
        "candidates_found": len(candidates),
        "grounded_with_coords": grounded_count,
        "search_type": search_type,
        "location": location,
        "vibe": vibe,
    }

    step_results.append({
        "node": "advanced_search",
        "status": "success",
        "summary": (
            f"MCP search complete: {len(candidates)} grounded candidates "
            f"for {search_type}s near {location} in {duration_ms:.0f}ms "
            f"[SELECTION_REQUIRED]"
        ),
        "duration_ms": duration_ms,
    })

    return {
        **state,
        "search_candidates": candidate_dicts,
        "selection_cards": selection_cards,
        "pending_user_selection": bool(candidates),
        "selected_search_candidate_id": None,
        "mcp_search_metadata": mcp_metadata,
        "generated_response": response_text,
        "final_response": response_text,
        "step_results": step_results,
    }

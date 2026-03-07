"""
Hotel/Restaurant/Activity Search Node: Real-Time External Search via Tavily.

This node activates when the user asks for hotels, restaurants, dining,
or nightlife near a specific location. It uses the existing Tavily/Web
Search integration to retrieve 3-5 structured options with details
(price, rating, availability) that can be added to the itinerary.

Research Pattern:
    Agentic Tool Use — The agent autonomously decides when to invoke
    external search based on user intent, enriching the itinerary with
    real-time accommodation and dining options.
"""

import logging
import re
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

from ..state import GraphState, HotelSearchResult

logger = logging.getLogger(__name__)

# Keywords that trigger hotel/restaurant search
HOTEL_SEARCH_KEYWORDS = [
    "hotel", "hotels", "accommodation", "stay", "lodge", "hostel", "resort",
    "restaurant", "restaurants", "dining", "eat", "food", "cafe", "cafes",
    "bar", "bars", "nightlife", "party", "club", "pub",
    "booking", "book a room", "where to stay", "where to eat",
]


def should_trigger_hotel_search(query: str) -> bool:
    """Check if the user query should trigger a hotel/restaurant search."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in HOTEL_SEARCH_KEYWORDS)


def _extract_location_from_query(query: str, state: GraphState) -> str:
    """Extract the target location from the query or state."""
    # Try to find location mentions in the query
    # Fall back to state's target_location or first tour plan location
    target_loc = state.get("target_location")
    if target_loc:
        return target_loc

    tour_context = state.get("tour_plan_context")
    if tour_context:
        locations = tour_context.get("selected_locations", [])
        if locations:
            return locations[0].get("name", "Sri Lanka")

    return "Sri Lanka"


def _determine_search_type(query: str) -> str:
    """Determine if the user is looking for hotels, restaurants, or activities."""
    query_lower = query.lower()
    if any(w in query_lower for w in ["hotel", "accommodation", "stay", "lodge", "hostel", "resort", "room"]):
        return "hotel"
    if any(w in query_lower for w in ["restaurant", "dining", "eat", "food", "cafe"]):
        return "restaurant"
    if any(w in query_lower for w in ["bar", "nightlife", "party", "club", "pub"]):
        return "bar"
    return "hotel"


def _parse_search_results(
    results: Dict[str, Any],
    search_type: str,
    location_name: str,
) -> List[HotelSearchResult]:
    """Parse Tavily search results into structured HotelSearchResult entries."""
    parsed: List[HotelSearchResult] = []

    for result in results.get("results", [])[:5]:
        title = result.get("title", "")
        content = result.get("content", "")
        url = result.get("url", "")

        # Extract price range from content (heuristic)
        price_range = None
        if any(w in content.lower() for w in ["luxury", "premium", "$$$", "high-end"]):
            price_range = "$$$"
        elif any(w in content.lower() for w in ["mid-range", "moderate", "$$"]):
            price_range = "$$"
        elif any(w in content.lower() for w in ["budget", "cheap", "affordable", "$"]):
            price_range = "$"
        else:
            price_range = "$$"

        # Extract rating (heuristic from content)
        rating = None
        rating_match = re.search(r'(\d\.?\d?)\s*/?\s*5', content)
        if rating_match:
            try:
                rating = float(rating_match.group(1))
            except ValueError:
                pass

        # Build description (first 200 chars of content)
        description = content[:200].strip()
        if len(content) > 200:
            description += "..."

        parsed.append(HotelSearchResult(
            name=title,
            type=search_type,
            price_range=price_range,
            rating=rating,
            url=url,
            description=description,
            distance_from_location=None,
            location_name=location_name,
        ))

    return parsed


@trace_node("hotel_search", run_type="tool")
async def hotel_search_node(state: GraphState, llm=None) -> GraphState:
    """
    Hotel/Restaurant Search Node: Search for accommodation and dining options.

    This node:
    1. Determines search type (hotel/restaurant/bar)
    2. Extracts the target location
    3. Uses Tavily to search for 3-5 options
    4. Returns structured results with price, rating, and links

    Args:
        state: Current graph state
        llm: LangChain LLM (not used directly, but needed for wrapper compatibility)

    Returns:
        Updated GraphState with hotel_search_results
    """
    import time
    start_time = time.time()
    logger.info("Hotel Search node processing...")

    query = state.get("user_query", "")
    location = _extract_location_from_query(query, state)
    search_type = _determine_search_type(query)

    logger.info(f"Searching for {search_type}s near {location}")

    try:
        from ...tools.web_search import get_web_search_tool
        from ...config import settings

        web_search = get_web_search_tool(settings.TAVILY_API_KEY)

        # Build search query
        search_query = f"best {search_type}s near {location} Sri Lanka"
        if search_type == "hotel":
            search_query += " booking reviews price"
        elif search_type == "restaurant":
            search_query += " reviews cuisine price"
        elif search_type == "bar":
            search_query += " nightlife reviews"

        # Use tourism-focused domains for better results
        domains = {
            "hotel": ["booking.com", "tripadvisor.com", "agoda.com", "hotels.com", "lonelyplanet.com"],
            "restaurant": ["tripadvisor.com", "yelp.com", "lonelyplanet.com", "timeout.com"],
            "bar": ["tripadvisor.com", "lonelyplanet.com", "timeout.com"],
        }

        results = web_search.search(
            search_query,
            include_domains=domains.get(search_type, domains["hotel"]),
        )

        # Parse results into structured format
        hotel_results = _parse_search_results(results, search_type, location)

        duration_ms = (time.time() - start_time) * 1000

        # Build a response summary
        if hotel_results:
            type_label = {"hotel": "accommodation", "restaurant": "dining", "bar": "nightlife"}.get(search_type, search_type)
            response_parts = [f"I found {len(hotel_results)} {type_label} options near {location}:\n"]
            for i, hr in enumerate(hotel_results, 1):
                line = f"{i}. **{hr['name']}**"
                if hr.get("price_range"):
                    line += f" ({hr['price_range']})"
                if hr.get("rating"):
                    line += f" — {hr['rating']}/5"
                line += f"\n   {hr['description']}"
                if hr.get("url"):
                    line += f"\n   [View details]({hr['url']})"
                response_parts.append(line)
            response_text = "\n\n".join(response_parts)
        else:
            response_text = f"I couldn't find specific {search_type} options near {location}. Try searching for a more specific location."

        return {
            **state,
            "hotel_search_results": hotel_results,
            "generated_response": response_text,
            "final_response": response_text,
            "step_results": [{
                "node": "hotel_search",
                "status": "success",
                "summary": f"Found {len(hotel_results)} {search_type} options near {location}",
                "duration_ms": duration_ms,
            }],
        }

    except Exception as e:
        logger.error(f"Hotel search failed: {e}")
        duration_ms = (time.time() - start_time) * 1000
        return {
            **state,
            "hotel_search_results": [],
            "generated_response": f"I had trouble searching for {search_type}s near {location}. Please try again.",
            "final_response": f"I had trouble searching for {search_type}s near {location}. Please try again.",
            "step_results": [{
                "node": "hotel_search",
                "status": "error",
                "summary": f"Hotel search failed: {str(e)[:100]}",
                "duration_ms": duration_ms,
            }],
        }


def route_to_hotel_search(state: GraphState) -> bool:
    """Check if the request should be routed to hotel search."""
    query = state.get("user_query", "")
    return should_trigger_hotel_search(query)

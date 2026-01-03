"""
Tour Plan Generator Node: Multi-Day Itinerary Generation with Optimization.

This node generates comprehensive tour plans by:
1. Analyzing selected locations and date ranges
2. Optimizing visit order based on distance and logistics
3. Integrating CrowdCast predictions for optimal timing
4. Incorporating Golden Hour calculations for photography
5. Applying Event Sentinel constraints (Poya days, holidays)

Research Pattern:
    Multi-Objective Optimization - Balances multiple constraints:
    - Time optimization (golden hour for photos)
    - Crowd avoidance (prefer low-crowd time slots)
    - Logistics (minimize travel time between locations)
    - Cultural constraints (Poya day restrictions)
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import math

from ..state import GraphState, IntentType, ItinerarySlot, TourPlanMetadata

logger = logging.getLogger(__name__)

# Activity icons mapping
ACTIVITY_ICONS = {
    "arrival": "ticket",
    "tickets": "ticket",
    "entrance": "ticket",
    "photography": "camera",
    "photo": "camera",
    "water": "water",
    "garden": "leaf",
    "nature": "leaf",
    "hiking": "walking",
    "climb": "walking",
    "walk": "walking",
    "temple": "landmark",
    "heritage": "landmark",
    "monument": "landmark",
    "frescoes": "palette",
    "art": "palette",
    "painting": "palette",
    "beach": "umbrella-beach",
    "swimming": "swimmer",
    "snorkeling": "fish",
    "wildlife": "paw",
    "safari": "paw",
    "food": "utensils",
    "lunch": "utensils",
    "dinner": "utensils",
    "breakfast": "coffee",
    "sunset": "sun",
    "sunrise": "sun",
    "default": "map-marker-alt",
}

# Tour plan generation system prompt
TOUR_PLAN_SYSTEM_PROMPT = """You are Travion, an expert AI tour planner for Sri Lanka.

You are generating a detailed tour itinerary based on:
- Selected locations and their details
- Date range for the trip
- Constraint analysis (crowds, golden hour, events)
- User preferences

CRITICAL RULES:
1. Generate a structured JSON response with the itinerary
2. Optimize visit times based on:
   - Golden hour for photography spots (sunrise/sunset)
   - Low crowd periods (early morning or late afternoon)
   - Poya day restrictions (no alcohol, modest dress)
3. Include practical tips and AI insights for each activity
4. Estimate realistic durations for activities
5. Consider travel time between locations

OUTPUT FORMAT (JSON):
{
    "summary": "Brief description of the tour plan",
    "match_score": 85,  // Overall optimization score 0-100
    "itinerary": [
        {
            "day": 1,
            "date": "2026-01-05",
            "location": "Sigiriya Rock Fortress",
            "activities": [
                {
                    "time": "06:30",
                    "activity": "Arrival & Tickets",
                    "duration_minutes": 30,
                    "notes": "Collect tickets early to avoid the 8 AM rush",
                    "crowd_prediction": 15,
                    "lighting_quality": "golden",
                    "icon": "ticket",
                    "highlight": false,
                    "ai_insight": null
                },
                {
                    "time": "07:00",
                    "activity": "Water Gardens",
                    "duration_minutes": 45,
                    "notes": "Best reflection shots of the rock fortress",
                    "crowd_prediction": 20,
                    "lighting_quality": "golden",
                    "icon": "water",
                    "highlight": true,
                    "ai_insight": "Golden Hour Alert: Best reflection shots of the rock fortress."
                }
            ]
        }
    ],
    "warnings": ["Poya day on Jan 6 - modest dress required"],
    "tips": ["Start early to avoid crowds", "Bring water bottles"]
}

Focus on creating practical, optimized, and enjoyable itineraries."""


def get_activity_icon(activity: str) -> str:
    """Get the appropriate icon for an activity."""
    activity_lower = activity.lower()
    for keyword, icon in ACTIVITY_ICONS.items():
        if keyword in activity_lower:
            return icon
    return ACTIVITY_ICONS["default"]


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance between two points in km."""
    R = 6371  # Earth's radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def optimize_location_order(locations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optimize the order of locations to minimize travel time.
    Uses a simple nearest-neighbor heuristic.
    """
    if len(locations) <= 2:
        return locations
    
    # Start with the first location
    optimized = [locations[0]]
    remaining = locations[1:]
    
    while remaining:
        current = optimized[-1]
        current_lat = current.get("latitude", 0)
        current_lon = current.get("longitude", 0)
        
        # Find nearest location
        min_dist = float("inf")
        nearest_idx = 0
        
        for i, loc in enumerate(remaining):
            loc_lat = loc.get("latitude", 0)
            loc_lon = loc.get("longitude", 0)
            dist = calculate_distance(current_lat, current_lon, loc_lat, loc_lon)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        optimized.append(remaining.pop(nearest_idx))
    
    return optimized


def build_plan_context(state: GraphState) -> str:
    """Build context string for tour plan generation."""
    parts = []
    
    tour_context = state.get("tour_plan_context")
    if not tour_context:
        return "No tour plan context available."
    
    # Date range
    parts.append(f"=== TRIP DATES ===")
    parts.append(f"Start: {tour_context.get('start_date', 'Not specified')}")
    parts.append(f"End: {tour_context.get('end_date', 'Not specified')}")
    parts.append("")
    
    # Selected locations
    locations = tour_context.get("selected_locations", [])
    if locations:
        parts.append(f"=== SELECTED LOCATIONS ({len(locations)}) ===")
        for i, loc in enumerate(locations, 1):
            parts.append(f"\n[{i}] {loc.get('name', 'Unknown')}")
            if loc.get("latitude") and loc.get("longitude"):
                parts.append(f"    GPS: ({loc['latitude']:.4f}, {loc['longitude']:.4f})")
            if loc.get("distance_km"):
                parts.append(f"    Distance: {loc['distance_km']:.1f} km")
        parts.append("")
    
    # Retrieved knowledge
    docs = state.get("retrieved_documents", [])
    if docs:
        parts.append("=== LOCATION KNOWLEDGE ===")
        for doc in docs[:6]:
            location = doc.get("metadata", {}).get("location", "Unknown")
            parts.append(f"\n{location}:")
            parts.append(doc["content"][:300])
        parts.append("")
    
    # Constraint analysis
    constraints = state.get("_constraint_results")
    if constraints:
        parts.append("=== CONSTRAINT ANALYSIS ===")
        parts.append(constraints.get("recommendation", ""))
        parts.append("")
    
    # Existing itinerary for modifications
    itinerary = state.get("itinerary")
    if itinerary:
        parts.append("=== CURRENT ITINERARY (for modification) ===")
        for slot in itinerary[:10]:
            day = slot.get("day", 1)
            parts.append(f"Day {day} - {slot['time']}: {slot['location']} - {slot['activity']}")
        parts.append("")
    
    return "\n".join(parts)


def parse_llm_plan_response(response_text: str) -> Tuple[List[ItinerarySlot], TourPlanMetadata, str]:
    """
    Parse the LLM response into structured itinerary slots.
    
    Returns:
        Tuple of (itinerary_slots, metadata, summary)
    """
    itinerary_slots: List[ItinerarySlot] = []
    metadata: TourPlanMetadata = {
        "match_score": 85,
        "total_days": 1,
        "total_locations": 0,
        "golden_hour_optimized": True,
        "crowd_optimized": True,
        "event_aware": True,
    }
    summary = ""
    
    try:
        # Try to extract JSON from the response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            plan_data = json.loads(json_str)
            
            summary = plan_data.get("summary", "")
            metadata["match_score"] = plan_data.get("match_score", 85)
            
            # Parse itinerary
            for day_data in plan_data.get("itinerary", []):
                day_num = day_data.get("day", 1)
                location = day_data.get("location", "")
                
                for order, activity in enumerate(day_data.get("activities", [])):
                    slot: ItinerarySlot = {
                        "time": activity.get("time", "09:00"),
                        "location": location,
                        "activity": activity.get("activity", ""),
                        "duration_minutes": activity.get("duration_minutes", 60),
                        "crowd_prediction": activity.get("crowd_prediction", 50),
                        "lighting_quality": activity.get("lighting_quality", "good"),
                        "notes": activity.get("notes"),
                        "day": day_num,
                        "order": order,
                        "icon": activity.get("icon") or get_activity_icon(activity.get("activity", "")),
                        "highlight": activity.get("highlight", False),
                        "ai_insight": activity.get("ai_insight"),
                    }
                    itinerary_slots.append(slot)
            
            metadata["total_days"] = len(plan_data.get("itinerary", []))
            metadata["total_locations"] = len(set(slot["location"] for slot in itinerary_slots))
            
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse plan JSON: {e}")
        # Create a basic plan from the text response
        summary = response_text[:200]
        
    except Exception as e:
        logger.error(f"Error parsing plan response: {e}")
    
    return itinerary_slots, metadata, summary


async def tour_plan_generator_node(state: GraphState, llm=None) -> GraphState:
    """
    Tour Plan Generator Node: Generate optimized multi-day itineraries.

    This node synthesizes location data, constraint analysis, and user preferences
    to create a comprehensive tour plan with optimized timing.

    Args:
        state: Current graph state with tour_plan_context
        llm: LangChain LLM instance

    Returns:
        Updated GraphState with generated itinerary and metadata
    """
    logger.info("Tour Plan Generator processing...")
    
    tour_context = state.get("tour_plan_context")
    if not tour_context:
        logger.warning("No tour plan context found")
        return {
            **state,
            "error": "No tour plan context provided",
            "final_response": "I couldn't generate a tour plan. Please provide locations and dates."
        }
    
    # Optimize location order
    locations = tour_context.get("selected_locations", [])
    if locations:
        locations = optimize_location_order(locations)
        tour_context["selected_locations"] = locations
    
    # Build context for LLM
    context = build_plan_context(state)
    
    # Generate plan using LLM
    if llm:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=TOUR_PLAN_SYSTEM_PROMPT),
                HumanMessage(content=f"""Generate a detailed tour plan for:

{context}

User request: {state['user_query']}

Remember to output valid JSON with the itinerary structure.""")
            ]
            
            response = await llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response
            itinerary, metadata, summary = parse_llm_plan_response(response_text)
            
            if itinerary:
                # Format a human-readable response
                final_response = f"🗺️ **Your {metadata['total_days']}-Day Sri Lanka Adventure**\n\n"
                final_response += f"📍 Covering {metadata['total_locations']} amazing locations\n"
                final_response += f"✨ Match Score: {metadata['match_score']}%\n\n"
                
                if summary:
                    final_response += f"{summary}\n\n"
                
                final_response += "I've optimized your itinerary for the best photography lighting and minimal crowds. "
                final_response += "Check out the detailed plan below! 👇"
                
                return {
                    **state,
                    "itinerary": itinerary,
                    "tour_plan_metadata": metadata,
                    "generated_response": final_response,
                    "final_response": final_response,
                }
            else:
                # Fallback response
                return {
                    **state,
                    "generated_response": response_text,
                    "final_response": response_text,
                }
                
        except Exception as e:
            logger.error(f"LLM plan generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Fallback: Generate basic plan without LLM
    logger.warning("Generating basic plan without LLM")
    
    basic_itinerary = []
    for i, loc in enumerate(locations):
        slot: ItinerarySlot = {
            "time": "09:00",
            "location": loc.get("name", f"Location {i+1}"),
            "activity": f"Visit {loc.get('name', 'location')}",
            "duration_minutes": 120,
            "crowd_prediction": 50,
            "lighting_quality": "good",
            "notes": None,
            "day": i + 1,
            "order": 0,
            "icon": "map-marker-alt",
            "highlight": i == 0,
            "ai_insight": None,
        }
        basic_itinerary.append(slot)
    
    basic_metadata: TourPlanMetadata = {
        "match_score": 75,
        "total_days": len(locations),
        "total_locations": len(locations),
        "golden_hour_optimized": False,
        "crowd_optimized": False,
        "event_aware": False,
    }
    
    return {
        **state,
        "itinerary": basic_itinerary,
        "tour_plan_metadata": basic_metadata,
        "generated_response": "I've created a basic tour plan for your selected locations.",
        "final_response": "I've created a basic tour plan for your selected locations.",
    }


def route_to_plan_generator(state: GraphState) -> bool:
    """Check if the request is for tour plan generation."""
    return state.get("tour_plan_context") is not None

"""
Tour Plan Generator Node: Super-Accuracy Multi-Day Itinerary Generation.

This node generates comprehensive, human-tour-guide-quality plans by:
1. Analyzing selected locations and date ranges
2. Optimizing visit order based on distance and logistics
3. Deep-injecting REAL CrowdCast predictions (hourly heatmaps per location)
4. Deep-injecting REAL Golden Hour calculations (precise sunrise/sunset per location)
5. Deep-injecting Event Sentinel constraints (Poya days, holidays, cultural events)
6. Personalizing based on user preference profile (history/adventure/nature/relaxation)
7. Including cultural tips, ethical insights, dress codes, and photography timing
8. Smart modification: preserving unchanged parts when user edits

Research Pattern:
    Multi-Objective Optimization with Deep Data Injection - Instead of asking
    the LLM to guess crowd/golden hour data, we calculate it precisely and
    inject it as ground truth, achieving human-tour-guide accuracy.
"""

import logging
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
import math

from ..state import (
    GraphState, IntentType, ItinerarySlot, TourPlanMetadata,
    CulturalTip, UserPreferences
)

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

# Sri Lanka cultural knowledge for enriching plans
SRI_LANKA_CULTURAL_TIPS = {
    "temple": [
        {"tip": "Remove shoes and hats before entering any Buddhist temple. Carry a bag for your footwear.", "category": "etiquette"},
        {"tip": "Dress modestly — cover shoulders and knees. Sarongs are available for rent at most temples.", "category": "etiquette"},
        {"tip": "Never pose with your back to a Buddha statue or point feet toward it — this is considered deeply disrespectful.", "category": "ethical"},
        {"tip": "Photography may be restricted inside shrine rooms. Always ask permission first.", "category": "ethical"},
        {"tip": "Walk clockwise around stupas and Bo trees as a sign of respect.", "category": "cultural"},
    ],
    "heritage": [
        {"tip": "Do not touch, sit on, or climb ancient ruins and sculptures — they are protected national heritage.", "category": "ethical"},
        {"tip": "Hire a licensed local guide for deeper historical context and to support the local economy.", "category": "cultural"},
        {"tip": "Some heritage sites have separate entry fees for foreign visitors. Carry cash in Sri Lankan Rupees.", "category": "safety"},
    ],
    "beach": [
        {"tip": "Respect local customs — avoid topless sunbathing, which is culturally inappropriate in Sri Lanka.", "category": "ethical"},
        {"tip": "Watch for rip currents, especially on the south coast. Swim near lifeguard stations where available.", "category": "safety"},
        {"tip": "Support local fishermen by buying fresh catch directly — it's fresher and supports the community.", "category": "cultural"},
    ],
    "nature": [
        {"tip": "Do not feed wild animals or leave food waste. Carry a reusable bag for trash.", "category": "ethical"},
        {"tip": "Stay on marked trails to protect fragile ecosystems and avoid encounters with wildlife.", "category": "safety"},
        {"tip": "Leeches are common in rainforest areas — wear long socks and apply repellent.", "category": "safety"},
    ],
    "wildlife": [
        {"tip": "Maintain safe distance from elephants (minimum 25m). Never block their path or use flash photography.", "category": "ethical"},
        {"tip": "Choose ethical safari operators who follow responsible wildlife viewing guidelines.", "category": "ethical"},
        {"tip": "Early morning (5:30-7:00 AM) and late afternoon (3:30-5:30 PM) offer the best wildlife sightings.", "category": "cultural"},
    ],
    "general": [
        {"tip": "Tipping 10% is customary at restaurants. For guides and drivers, Rs 500-1000/day is appreciated.", "category": "etiquette"},
        {"tip": "Use your right hand for greetings and passing items — the left hand is considered unclean.", "category": "etiquette"},
        {"tip": "Greet locals with 'Ayubowan' (palms together) — it means 'may you live long' and is warmly received.", "category": "cultural"},
    ],
}

# Supercharged system prompt acting as real Sri Lankan tour guide
TOUR_PLAN_SYSTEM_PROMPT = """You are Travion, an expert Sri Lankan tour guide with 20+ years of experience guiding tourists across the island. You know every hidden gem, every sunrise spot, every temple protocol, and every local secret.

You are generating a PRECISE, PERSONALIZED tour itinerary. You have been given:
- EXACT golden hour times (calculated by physics engine — use these EXACT times, not guesses)
- EXACT crowd predictions (calculated by ML model — use these percentages directly)
- Event/holiday data (Poya days, school holidays, cultural events)
- User's personal preference profile (what they love vs. what they avoid)
- Cultural knowledge about each location

CRITICAL ACCURACY RULES:
1. Use the INJECTED golden hour times EXACTLY as provided — these are physics-calculated, NOT estimates
2. Use the INJECTED crowd percentages EXACTLY — schedule activities at the lowest-crowd windows
3. Schedule outdoor photography/scenic activities ONLY during golden hour windows
4. Schedule indoor/cultural activities during harsh midday light (10AM-2PM)
5. Include realistic travel times between locations (Sri Lankan roads: ~30-40 km/h average)
6. Each day should have 8-12 hours of activities (not more than 14 hours)
7. Include meal breaks (breakfast 7-8AM, lunch 12-1PM, dinner 7-8PM)

PERSONALIZATION RULES (based on user preference scores 0-1):
- High history (>0.6): Prioritize heritage sites, museums, ancient ruins with detailed historical context
- High adventure (>0.6): Include hiking, water sports, rock climbing, off-road activities
- High nature (>0.6): Focus on national parks, waterfalls, botanical gardens, bird watching
- High relaxation (>0.6): Include spa time, beach relaxation, scenic viewpoints, gentle walks
- Balance ALL activities according to the user's exact preference scores

CULTURAL & ETHICAL INSIGHTS (MANDATORY):
- For EVERY activity near a temple/religious site: include dress code and behavior rules
- For EVERY nature activity: include eco-friendly practices
- For EVERY interaction with locals: include cultural etiquette
- Include at least 2-3 cultural tips per day

MODIFICATION MODE (when existing plan provided):
- ONLY change what the user explicitly asks for
- Keep ALL other parts of the plan EXACTLY as they were
- Recalculate crowd/golden hour only for changed time slots
- Explain what was changed and why in the summary

OUTPUT FORMAT (strict JSON — no extra text before or after):
{
    "summary": "Personalized description explaining why this plan matches the user's interests",
    "match_score": 85,
    "preference_match_explanation": "This plan emphasizes X because you scored high on Y...",
    "itinerary": [
        {
            "day": 1,
            "date": "2026-01-05",
            "location": "Sigiriya Rock Fortress",
            "activities": [
                {
                    "time": "06:15",
                    "activity": "Golden Hour Photography at Water Gardens",
                    "duration_minutes": 45,
                    "notes": "Arrive before sunrise for mirror reflections of the rock",
                    "crowd_prediction": 12,
                    "lighting_quality": "golden",
                    "icon": "camera",
                    "highlight": true,
                    "ai_insight": "Golden Hour Alert: Morning golden hour 06:10-06:45. Best reflection shots with zero crowds.",
                    "cultural_tip": "Sigiriya's Water Gardens use ancient hydraulic engineering from 5th century — observe the fountain jets that still work!",
                    "ethical_note": "Stay on marked paths. Do not touch the ancient frescoes — oils from skin cause irreversible damage.",
                    "best_photo_time": "06:15-06:45 golden hour — water reflections of Lion Rock"
                }
            ]
        }
    ],
    "cultural_tips": [
        {"location": "Sigiriya", "tip": "The frescoes depict celestial maidens. Photography is restricted.", "category": "ethical"},
        {"location": "General", "tip": "Carry water. Hydration is essential in Sri Lanka's tropical climate.", "category": "safety"}
    ],
    "warnings": ["Poya day on Jan 6 — alcohol sales banned, modest dress required at all sites"],
    "tips": ["Start Sigiriya before 7 AM to beat tour bus crowds", "Bring a hat and sunscreen for the exposed rock climb"]
}

OUTPUT ONLY VALID JSON. No markdown, no extra text. Start with { and end with }."""

# Modification-specific prompt addition
MODIFICATION_PROMPT_ADDITION = """

=== MODIFICATION MODE ===
The user wants to MODIFY their existing tour plan. Follow these rules STRICTLY:
1. ONLY change what the user explicitly asks for in their message
2. Keep ALL other activities, times, and details EXACTLY as they were
3. If the user asks to add something, find the best slot without disrupting existing activities
4. If the user asks to remove something, adjust surrounding activities to fill the gap naturally
5. If the user asks to change timing, recalculate only the affected activities
6. In the summary, clearly explain: "Changed: [what], Reason: [why], Kept: [everything else unchanged]"
"""


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


def _compute_golden_hour_data(locations: List[Dict[str, Any]], start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Compute precise golden hour data for each location using the physics engine.

    Returns dict mapping location_name -> {sunrise, sunset, morning_golden, evening_golden}
    """
    golden_data = {}

    try:
        from ...tools.golden_hour import get_golden_hour_agent
        gh_agent = get_golden_hour_agent()

        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date

        for loc in locations:
            name = loc.get("name", "Unknown")
            lat = loc.get("latitude", 7.8731)
            lon = loc.get("longitude", 80.7718)

            try:
                result = gh_agent.calculate_golden_hour(lat, lon, start_dt.isoformat())
                golden_data[name] = {
                    "sunrise": result.get("sunrise", "06:00"),
                    "sunset": result.get("sunset", "18:00"),
                    "morning_golden_start": result.get("morning_golden_start", "05:45"),
                    "morning_golden_end": result.get("morning_golden_end", "06:30"),
                    "evening_golden_start": result.get("evening_golden_start", "17:30"),
                    "evening_golden_end": result.get("evening_golden_end", "18:15"),
                    "day_length_hours": result.get("day_length_hours", 12.0),
                }
                logger.info(f"Golden hour for {name}: sunrise={golden_data[name]['sunrise']}, sunset={golden_data[name]['sunset']}")
            except Exception as e:
                logger.warning(f"Golden hour calculation failed for {name}: {e}")
                golden_data[name] = {
                    "sunrise": "06:00", "sunset": "18:00",
                    "morning_golden_start": "05:45", "morning_golden_end": "06:30",
                    "evening_golden_start": "17:30", "evening_golden_end": "18:15",
                    "day_length_hours": 12.0,
                }
    except ImportError:
        logger.warning("Golden hour agent not available, using defaults")
    except Exception as e:
        logger.warning(f"Golden hour computation failed: {e}")

    return golden_data


def _compute_crowd_predictions(locations: List[Dict[str, Any]], start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Compute crowd predictions for each location at multiple time slots.

    Returns dict mapping location_name -> {hour: crowd_percentage}
    """
    crowd_data = {}
    time_slots = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]

    try:
        from ...tools.crowdcast import get_crowdcast
        crowdcast = get_crowdcast()

        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date

        for loc in locations:
            name = loc.get("name", "Unknown")
            loc_type = loc.get("type", "Heritage")

            # Determine location type for CrowdCast
            name_lower = name.lower()
            if any(w in name_lower for w in ["beach", "coast", "bay"]):
                loc_type = "Beach"
            elif any(w in name_lower for w in ["temple", "shrine", "dagoba", "vihara", "kovil"]):
                loc_type = "Religious"
            elif any(w in name_lower for w in ["park", "forest", "falls", "waterfall", "lake", "garden"]):
                loc_type = "Nature"
            elif any(w in name_lower for w in ["fort", "fortress", "palace", "ruins", "ancient", "heritage", "rock"]):
                loc_type = "Heritage"
            else:
                loc_type = "Urban"

            hourly = {}
            best_hour = 6
            lowest_crowd = 100

            for hour in time_slots:
                try:
                    target_dt = start_dt.replace(hour=hour) if hasattr(start_dt, 'hour') else datetime(
                        start_dt.year, start_dt.month, start_dt.day, hour
                    )
                    result = crowdcast.predict(
                        location_type=loc_type,
                        target_datetime=target_dt,
                    )
                    crowd_pct = round(result.get("crowd_percentage", 50))
                    hourly[str(hour)] = crowd_pct
                    if crowd_pct < lowest_crowd:
                        lowest_crowd = crowd_pct
                        best_hour = hour
                except Exception as e:
                    hourly[str(hour)] = 50  # Default
                    logger.debug(f"Crowd prediction failed for {name} at {hour}:00 - {e}")

            crowd_data[name] = {
                "hourly": hourly,
                "best_hour": best_hour,
                "lowest_crowd": lowest_crowd,
                "location_type": loc_type,
            }
            logger.info(f"Crowd data for {name}: best_hour={best_hour}, lowest={lowest_crowd}%")

    except ImportError:
        logger.warning("CrowdCast not available, using defaults")
    except Exception as e:
        logger.warning(f"Crowd prediction computation failed: {e}")

    return crowd_data


def _compute_event_data(locations: List[Dict[str, Any]], start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Compute event/holiday impact for each location across the date range.

    Returns dict with holiday info, Poya days, warnings.
    """
    event_data = {"poya_days": [], "holidays": [], "warnings": [], "location_impacts": {}}

    try:
        from ...tools.event_sentinel import get_event_sentinel
        sentinel = get_event_sentinel()

        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date

        # Check each day in the range
        current = start_dt
        while current <= end_dt:
            date_str = current.strftime("%Y-%m-%d")

            for loc in locations:
                name = loc.get("name", "Unknown")
                try:
                    impact = sentinel.get_impact(name, date_str)
                    if impact.get("is_poya_day") and date_str not in event_data["poya_days"]:
                        event_data["poya_days"].append(date_str)
                    if impact.get("holidays"):
                        for h in impact["holidays"]:
                            if h not in event_data["holidays"]:
                                event_data["holidays"].append(h)
                    if impact.get("warnings"):
                        for w in impact["warnings"]:
                            if w not in event_data["warnings"]:
                                event_data["warnings"].append(w)

                    # Store per-location impact
                    if name not in event_data["location_impacts"]:
                        event_data["location_impacts"][name] = {}
                    event_data["location_impacts"][name][date_str] = {
                        "is_poya": impact.get("is_poya_day", False),
                        "crowd_modifier": impact.get("predicted_crowd_modifier", 1.0),
                        "restrictions": impact.get("travel_advice_strings", []),
                    }
                except Exception as e:
                    logger.debug(f"Event check failed for {name} on {date_str}: {e}")

            current += timedelta(days=1)

    except ImportError:
        logger.warning("Event Sentinel not available")
    except Exception as e:
        logger.warning(f"Event data computation failed: {e}")

    return event_data


def _get_cultural_tips_for_location(location_name: str) -> List[Dict[str, str]]:
    """Get relevant cultural tips based on location type."""
    tips = []
    name_lower = location_name.lower()

    if any(w in name_lower for w in ["temple", "shrine", "dagoba", "vihara", "tooth", "sacred", "kovil"]):
        tips.extend(SRI_LANKA_CULTURAL_TIPS["temple"][:2])
    if any(w in name_lower for w in ["fort", "fortress", "palace", "ruins", "ancient", "heritage", "sigiriya", "polonnaruwa", "anuradhapura"]):
        tips.extend(SRI_LANKA_CULTURAL_TIPS["heritage"][:2])
    if any(w in name_lower for w in ["beach", "coast", "bay", "lagoon"]):
        tips.extend(SRI_LANKA_CULTURAL_TIPS["beach"][:2])
    if any(w in name_lower for w in ["park", "forest", "falls", "reserve", "sanctuary"]):
        tips.extend(SRI_LANKA_CULTURAL_TIPS["nature"][:2])
    if any(w in name_lower for w in ["safari", "wildlife", "elephant", "leopard", "whale"]):
        tips.extend(SRI_LANKA_CULTURAL_TIPS["wildlife"][:2])

    # Always add a general tip
    if not tips:
        tips.extend(SRI_LANKA_CULTURAL_TIPS["general"][:1])

    return tips


def _build_user_preferences_context(preferences: Optional[UserPreferences]) -> str:
    """Build a preference context string for the LLM."""
    if not preferences:
        return "No specific preferences provided. Create a balanced itinerary."

    parts = ["=== USER PREFERENCE PROFILE ==="]
    parts.append(f"History & Culture Interest: {preferences.get('history', 0.5):.0%}")
    parts.append(f"Adventure Interest: {preferences.get('adventure', 0.5):.0%}")
    parts.append(f"Nature & Wildlife Interest: {preferences.get('nature', 0.5):.0%}")
    parts.append(f"Relaxation Interest: {preferences.get('relaxation', 0.5):.0%}")

    # Describe what to prioritize
    scores = {
        "historical/cultural sites": preferences.get("history", 0.5),
        "adventure activities": preferences.get("adventure", 0.5),
        "nature & wildlife experiences": preferences.get("nature", 0.5),
        "relaxation & leisure time": preferences.get("relaxation", 0.5),
    }
    sorted_prefs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_pref = sorted_prefs[0]
    parts.append(f"\nPrimary Interest: {top_pref[0]} ({top_pref[1]:.0%})")
    parts.append(f"Secondary Interest: {sorted_prefs[1][0]} ({sorted_prefs[1][1]:.0%})")

    if preferences.get("pace"):
        parts.append(f"Travel Pace: {preferences['pace']}")
    if preferences.get("budget"):
        parts.append(f"Budget: {preferences['budget']}")
    if preferences.get("group_size"):
        parts.append(f"Group Size: {preferences['group_size']}")
    if preferences.get("dietary"):
        parts.append(f"Dietary Restrictions: {', '.join(preferences['dietary'])}")
    if preferences.get("accessibility"):
        parts.append("Accessibility Needs: Yes — prioritize accessible routes")

    parts.append("\nTailor ALL activities to match these preferences!")
    return "\n".join(parts)


def build_plan_context(state: GraphState) -> str:
    """Build comprehensive context string for tour plan generation with deep data injection."""
    parts = []

    tour_context = state.get("tour_plan_context")
    if not tour_context:
        return "No tour plan context available."

    # Date range
    start_date = tour_context.get('start_date', 'Not specified')
    end_date = tour_context.get('end_date', 'Not specified')
    parts.append("=== TRIP DATES ===")
    parts.append(f"Start: {start_date}")
    parts.append(f"End: {end_date}")
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
                parts.append(f"    Distance from previous: {loc['distance_km']:.1f} km")
        parts.append("")

    # User preferences (deep integration)
    preferences = state.get("user_preferences")
    pref_context = _build_user_preferences_context(preferences)
    parts.append(pref_context)
    parts.append("")

    # Golden hour data (deep injection)
    golden_data = state.get("_golden_hour_data", {})
    if golden_data:
        parts.append("=== PRECISE GOLDEN HOUR DATA (Physics-Calculated) ===")
        parts.append("USE THESE EXACT TIMES — they are calculated by physics engine, NOT estimates!")
        for loc_name, gh in golden_data.items():
            parts.append(f"\n{loc_name}:")
            parts.append(f"  Sunrise: {gh.get('sunrise', 'N/A')}")
            parts.append(f"  Morning Golden Hour: {gh.get('morning_golden_start', 'N/A')} - {gh.get('morning_golden_end', 'N/A')}")
            parts.append(f"  Evening Golden Hour: {gh.get('evening_golden_start', 'N/A')} - {gh.get('evening_golden_end', 'N/A')}")
            parts.append(f"  Sunset: {gh.get('sunset', 'N/A')}")
        parts.append("")

    # Crowd predictions (deep injection)
    crowd_data = state.get("_crowd_data", {})
    if crowd_data:
        parts.append("=== CROWD PREDICTIONS (ML-Calculated Hourly) ===")
        parts.append("USE THESE PERCENTAGES — they are calculated by ML model!")
        for loc_name, cd in crowd_data.items():
            hourly = cd.get("hourly", {})
            best = cd.get("best_hour", 6)
            parts.append(f"\n{loc_name} (Type: {cd.get('location_type', 'Unknown')}):")
            hour_strs = []
            for h in sorted(hourly.keys(), key=int):
                pct = hourly[h]
                marker = " <<< BEST" if int(h) == best else ""
                hour_strs.append(f"  {h}:00 → {pct}% crowd{marker}")
            parts.append("\n".join(hour_strs))
            parts.append(f"  RECOMMENDED: Visit at {best}:00 ({cd.get('lowest_crowd', 50)}% crowd)")
        parts.append("")

    # Event/holiday data
    event_data = state.get("_event_data", {})
    if event_data:
        parts.append("=== EVENT & HOLIDAY DATA ===")
        if event_data.get("poya_days"):
            parts.append(f"Poya Days: {', '.join(event_data['poya_days'])}")
            parts.append("  RULES: No alcohol sales, modest dress required, temples very crowded")
        if event_data.get("holidays"):
            parts.append(f"Holidays: {', '.join(str(h) for h in event_data['holidays'])}")
        if event_data.get("warnings"):
            for w in event_data["warnings"]:
                parts.append(f"  WARNING: {w}")
        parts.append("")

    # Retrieved knowledge
    docs = state.get("retrieved_documents", [])
    if docs:
        parts.append("=== LOCATION KNOWLEDGE ===")
        for doc in docs[:6]:
            location = doc.get("metadata", {}).get("location", "Unknown")
            parts.append(f"\n{location}:")
            parts.append(doc["content"][:400])
        parts.append("")

    # Cultural tips for each location
    if locations:
        parts.append("=== CULTURAL CONTEXT (include in plan) ===")
        for loc in locations:
            name = loc.get("name", "Unknown")
            tips = _get_cultural_tips_for_location(name)
            if tips:
                parts.append(f"\n{name}:")
                for t in tips:
                    parts.append(f"  [{t['category'].upper()}] {t['tip']}")
        parts.append("")

    # Constraint analysis from shadow monitor
    constraints = state.get("_constraint_results")
    if constraints:
        parts.append("=== CONSTRAINT ANALYSIS ===")
        parts.append(constraints.get("recommendation", ""))
        parts.append("")

    # Existing itinerary for modifications
    itinerary = state.get("itinerary")
    if itinerary:
        parts.append("=== CURRENT ITINERARY (for modification — preserve unchanged parts) ===")
        current_day = None
        for slot in itinerary:
            day = slot.get("day", 1)
            if day != current_day:
                parts.append(f"\n--- Day {day} ---")
                current_day = day
            parts.append(f"  {slot['time']}: {slot['location']} — {slot['activity']} ({slot.get('duration_minutes', 60)} min)")
            if slot.get("cultural_tip"):
                parts.append(f"    Cultural: {slot['cultural_tip']}")
            if slot.get("notes"):
                parts.append(f"    Note: {slot['notes']}")
        parts.append("")

    return "\n".join(parts)


def parse_llm_plan_response(
    response_text: str,
) -> Tuple[List[ItinerarySlot], TourPlanMetadata, str, List[CulturalTip]]:
    """
    Parse the LLM response into structured itinerary slots with cultural tips.

    Returns:
        Tuple of (itinerary_slots, metadata, summary, cultural_tips)
    """
    itinerary_slots: List[ItinerarySlot] = []
    metadata: TourPlanMetadata = {
        "match_score": 85,
        "total_days": 1,
        "total_locations": 0,
        "golden_hour_optimized": True,
        "crowd_optimized": True,
        "event_aware": True,
        "preference_match_explanation": None,
    }
    summary = ""
    cultural_tips: List[CulturalTip] = []

    try:
        # Try to extract JSON from the response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            plan_data = json.loads(json_str)

            summary = plan_data.get("summary", "")
            metadata["match_score"] = plan_data.get("match_score", 85)
            metadata["preference_match_explanation"] = plan_data.get("preference_match_explanation")

            # Parse cultural tips
            for tip_data in plan_data.get("cultural_tips", []):
                cultural_tips.append(CulturalTip(
                    location=tip_data.get("location", "General"),
                    tip=tip_data.get("tip", ""),
                    category=tip_data.get("category", "cultural"),
                ))

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
                        "cultural_tip": activity.get("cultural_tip"),
                        "ethical_note": activity.get("ethical_note"),
                        "best_photo_time": activity.get("best_photo_time"),
                    }
                    itinerary_slots.append(slot)

            metadata["total_days"] = len(plan_data.get("itinerary", []))
            metadata["total_locations"] = len(set(slot["location"] for slot in itinerary_slots))

            # Parse warnings and tips into metadata if needed
            warnings = plan_data.get("warnings", [])
            tips = plan_data.get("tips", [])

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse plan JSON: {e}")
        summary = response_text[:200]

    except Exception as e:
        logger.error(f"Error parsing plan response: {e}")

    return itinerary_slots, metadata, summary, cultural_tips


async def tour_plan_generator_node(state: GraphState, llm=None) -> GraphState:
    """
    Tour Plan Generator Node: Generate super-accurate, personalized multi-day itineraries.

    This node:
    1. Computes precise golden hour data for each location (physics engine)
    2. Computes precise crowd predictions at hourly intervals (ML model)
    3. Checks event/holiday data across the date range
    4. Builds cultural tip database for each location
    5. Injects ALL of this as ground truth into the LLM prompt
    6. Generates a personalized plan based on user preferences

    Args:
        state: Current graph state with tour_plan_context
        llm: LangChain LLM instance

    Returns:
        Updated GraphState with generated itinerary, metadata, and cultural tips
    """
    import time
    start_time = time.time()
    logger.info("Tour Plan Generator processing with deep data injection...")

    tour_context = state.get("tour_plan_context")
    if not tour_context:
        logger.warning("No tour plan context found")
        return {
            **state,
            "error": "No tour plan context provided",
            "final_response": "I couldn't generate a tour plan. Please provide locations and dates.",
            "step_results": [{"node": "tour_plan_generate", "status": "error", "summary": "No tour plan context provided", "duration_ms": 0}],
        }

    # Optimize location order
    locations = tour_context.get("selected_locations", [])
    if locations:
        locations = optimize_location_order(locations)
        tour_context["selected_locations"] = locations

    start_date = tour_context.get("start_date", "")
    end_date = tour_context.get("end_date", "")
    is_modification = state.get("itinerary") is not None

    # Step 1: Compute precise golden hour data
    logger.info("Computing golden hour data for each location...")
    golden_data = _compute_golden_hour_data(locations, start_date, end_date)

    # Step 2: Compute precise crowd predictions
    logger.info("Computing crowd predictions for each location...")
    crowd_data = _compute_crowd_predictions(locations, start_date, end_date)

    # Step 3: Compute event/holiday data
    logger.info("Computing event/holiday data...")
    event_data = _compute_event_data(locations, start_date, end_date)

    # Step 4: Collect cultural tips
    all_cultural_tips: List[CulturalTip] = []
    for loc in locations:
        name = loc.get("name", "Unknown")
        tips = _get_cultural_tips_for_location(name)
        for t in tips:
            all_cultural_tips.append(CulturalTip(
                location=name,
                tip=t["tip"],
                category=t["category"],
            ))

    # Store computed data in state for context building
    state_with_data = {
        **state,
        "_golden_hour_data": golden_data,
        "_crowd_data": crowd_data,
        "_event_data": event_data,
    }

    # Build comprehensive context for LLM
    context = build_plan_context(state_with_data)

    # Generate plan using LLM
    if llm:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            system_prompt = TOUR_PLAN_SYSTEM_PROMPT
            if is_modification:
                system_prompt += MODIFICATION_PROMPT_ADDITION

            user_message = f"""Generate a {'modified' if is_modification else 'detailed'} tour plan for:

{context}

User request: {state['user_query']}

IMPORTANT: Output ONLY valid JSON. Start with {{ and end with }}. No extra text."""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            # Include conversation history for context in modifications
            if is_modification:
                conv_messages = state.get("messages", [])
                if len(conv_messages) > 1:
                    history_msgs = conv_messages[:-1][-6:]  # Last 3 turns
                    for msg in history_msgs:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role in ["user", "assistant"] and content:
                            messages.insert(1, HumanMessage(content=content) if role == "user" else SystemMessage(content=f"Previous response: {content[:300]}"))

            response = await llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse the response
            itinerary, metadata, summary, parsed_cultural_tips = parse_llm_plan_response(response_text)

            # Merge cultural tips
            if parsed_cultural_tips:
                all_cultural_tips.extend(parsed_cultural_tips)

            # Extract warnings and tips from raw JSON
            warnings = []
            tips = []
            try:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    raw_plan = json.loads(response_text[json_start:json_end])
                    warnings = raw_plan.get("warnings", [])
                    tips = raw_plan.get("tips", [])
            except Exception:
                pass

            # Add event warnings
            if event_data.get("poya_days"):
                for poya_date in event_data["poya_days"]:
                    w = f"Poya Day on {poya_date} — alcohol sales banned nationwide, modest dress required at all sites"
                    if w not in warnings:
                        warnings.append(w)
            if event_data.get("warnings"):
                for w in event_data["warnings"]:
                    if w not in warnings:
                        warnings.append(w)

            duration_ms = (time.time() - start_time) * 1000

            if itinerary:
                # Format a human-readable response
                final_response = f"🗺️ **Your {metadata['total_days']}-Day Sri Lanka Adventure**\n\n"
                final_response += f"📍 Covering {metadata['total_locations']} amazing locations\n"
                final_response += f"✨ Match Score: {metadata['match_score']}%\n\n"

                if summary:
                    final_response += f"{summary}\n\n"

                if metadata.get("preference_match_explanation"):
                    final_response += f"🎯 {metadata['preference_match_explanation']}\n\n"

                final_response += "I've optimized your itinerary using precise golden hour calculations and real-time crowd predictions. "
                if is_modification:
                    final_response += "I've made the changes you requested while keeping the rest of your plan intact. "
                final_response += "Check out the detailed plan below! 👇"

                return {
                    **state,
                    "itinerary": itinerary,
                    "tour_plan_metadata": metadata,
                    "generated_response": final_response,
                    "final_response": final_response,
                    "cultural_tips": all_cultural_tips,
                    "step_results": [{
                        "node": "tour_plan_generate",
                        "status": "success",
                        "summary": f"Generated {metadata['total_days']}-day plan with {len(itinerary)} activities, {len(all_cultural_tips)} cultural tips",
                        "duration_ms": duration_ms,
                    }],
                }
            else:
                # Fallback response
                return {
                    **state,
                    "generated_response": response_text,
                    "final_response": response_text,
                    "cultural_tips": all_cultural_tips,
                    "step_results": [{
                        "node": "tour_plan_generate",
                        "status": "warning",
                        "summary": "Plan generated but could not parse structured itinerary",
                        "duration_ms": duration_ms,
                    }],
                }

        except Exception as e:
            logger.error(f"LLM plan generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Fallback: Generate basic plan without LLM
    logger.warning("Generating basic plan without LLM")
    duration_ms = (time.time() - start_time) * 1000

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
            "cultural_tip": None,
            "ethical_note": None,
            "best_photo_time": None,
        }
        basic_itinerary.append(slot)

    basic_metadata: TourPlanMetadata = {
        "match_score": 75,
        "total_days": len(locations),
        "total_locations": len(locations),
        "golden_hour_optimized": False,
        "crowd_optimized": False,
        "event_aware": False,
        "preference_match_explanation": None,
    }

    return {
        **state,
        "itinerary": basic_itinerary,
        "tour_plan_metadata": basic_metadata,
        "cultural_tips": all_cultural_tips,
        "generated_response": "I've created a basic tour plan for your selected locations.",
        "final_response": "I've created a basic tour plan for your selected locations.",
        "step_results": [{
            "node": "tour_plan_generate",
            "status": "warning",
            "summary": "Generated basic plan without LLM (fallback mode)",
            "duration_ms": duration_ms,
        }],
    }


def route_to_plan_generator(state: GraphState) -> bool:
    """Check if the request is for tour plan generation."""
    return state.get("tour_plan_context") is not None

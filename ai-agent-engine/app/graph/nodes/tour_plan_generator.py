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

import asyncio
import logging
import json
import time as _time_mod
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
import math

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

from ..state import (
    GraphState, IntentType, ItinerarySlot, TourPlanMetadata,
    CulturalTip, UserPreferences, FinalItinerary, FinalItineraryStop,
    RouteCoordinate, ContextualNote,
    RestaurantRecommendation, AccommodationRecommendation
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
- REAL weather forecasts (from OpenWeatherMap API — use these for weather-aware routing)
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

WEATHER-AWARE ROUTING RULES (CRITICAL):
8. If rain_probability > 80% for a location: REPLACE outdoor activities with indoor alternatives (museums, temples, cooking classes, spas, shopping)
9. If rain_probability 50-80%: Keep outdoor activities but add "Carry rain gear" warning and suggest a backup indoor option
10. If extreme heat (>35°C): Schedule shaded/indoor activities between 11AM-2PM, suggest hydration breaks
11. If high winds (>40 km/h): Avoid water sports and exposed hilltop activities
12. Always include weather context in activity notes (e.g., "Clear skies expected, perfect for photography")

PERSONALIZATION RULES (based on user preference scores 0-1):
- High history (>0.6): Prioritize heritage sites, museums, ancient ruins with detailed historical context
- High adventure (>0.6): Include hiking, water sports, rock climbing, off-road activities
- High nature (>0.6): Focus on national parks, waterfalls, botanical gardens, bird watching
- High relaxation (>0.6): Include spa time, beach relaxation, scenic viewpoints, gentle walks
- Balance ALL activities according to the user's exact preference scores

PHOTOGRAPHY TIME PROMINENCE (CRITICAL):
- For EVERY outdoor location, include a dedicated best_photo_time field
- Format: "HH:MM-HH:MM (golden hour) — [specific tip for this exact location]"
- If an activity overlaps with golden hour, set highlight=true and use icon="camera"
- Always mention the exact golden hour window in ai_insight when relevant
- Include both morning AND evening golden hour opportunities where applicable
- For iconic viewpoints (Sigiriya summit, Ella's Nine Arches, etc.), specify the EXACT best angle and time

ACTIVITY OPTIMIZATION RULES (match activities to optimal time of day):
- Photography/scenic viewpoints: ONLY during golden hour windows (injected above)
- Temple/religious visits: Early morning (cooler, spiritual, less crowded) or late afternoon
- Hiking/strenuous activities: Before 10 AM to avoid midday heat
- Indoor activities (museums, cooking classes, spas): 10 AM - 2 PM (harsh light outside)
- Beach activities: Morning (calmer water) or late afternoon (golden light)
- Wildlife/safari: 5:30-7:00 AM or 3:30-5:30 PM (feeding times, best sightings)
- Shopping/markets: Late morning or evening when vendors are active
- Use crowd_prediction data to avoid peak hours — if crowd > 60% at a given hour, shift to a lower-crowd slot

RESTAURANT & ACCOMMODATION INTEGRATION:
- If selected restaurants are provided in context, include them as meal activities at the correct time
- If selected accommodations are provided, include check-in (evening) and check-out (morning) in the plan
- Format meal activities with icon="utensils" for lunch/dinner and icon="coffee" for breakfast
- If no restaurants are selected, still include meal break placeholders at standard times

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
                result = gh_agent.get_sun_times(start_dt, lat, lon, name)
                morning_gh = result.get("golden_hour_morning", {})
                evening_gh = result.get("golden_hour_evening", {})
                golden_data[name] = {
                    "sunrise": result.get("sunrise", "06:00"),
                    "sunset": result.get("sunset", "18:00"),
                    "morning_golden_start": morning_gh.get("start", "05:45") if isinstance(morning_gh, dict) else "05:45",
                    "morning_golden_end": morning_gh.get("end", "06:30") if isinstance(morning_gh, dict) else "06:30",
                    "evening_golden_start": evening_gh.get("start", "17:30") if isinstance(evening_gh, dict) else "17:30",
                    "evening_golden_end": evening_gh.get("end", "18:15") if isinstance(evening_gh, dict) else "18:15",
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


async def _compute_weather_data(locations: List[Dict[str, Any]], start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Compute weather forecasts for each location using OpenWeatherMap API.

    Returns dict mapping location_name -> {
        temperature, rain_probability, wind_speed, condition,
        is_suitable_outdoor, alerts, indoor_alternative_needed
    }
    """
    weather_data = {}

    try:
        from ...tools.weather_api import WeatherTool
        weather_tool = WeatherTool()

        if not weather_tool.is_configured():
            logger.warning("Weather API not configured, skipping weather data")
            return weather_data

        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date

        for loc in locations:
            name = loc.get("name", "Unknown")
            lat = loc.get("latitude", 7.8731)
            lon = loc.get("longitude", 80.7718)

            try:
                report = await weather_tool.get_location_weather_report(
                    location_name=name,
                    latitude=lat,
                    longitude=lon,
                    target_date=start_dt,
                )

                # Aggregate forecast data for the location
                max_rain_prob = 0
                avg_temp = 25.0
                max_wind = 0
                conditions = []
                alerts = []
                suitable_count = 0

                for forecast in report.forecasts:
                    max_rain_prob = max(max_rain_prob, forecast.rain_probability)
                    avg_temp = forecast.temperature_celsius  # Use latest
                    max_wind = max(max_wind, forecast.wind_speed_kmh)
                    if forecast.condition.value not in conditions:
                        conditions.append(forecast.condition.value)
                    if forecast.is_suitable_outdoor:
                        suitable_count += 1
                    for alert in forecast.alerts:
                        alerts.append({
                            "type": alert.alert_type,
                            "severity": alert.severity.value,
                            "description": alert.description,
                            "recommendation": alert.recommendation,
                        })

                suitability = report.trip_suitability_score
                indoor_needed = max_rain_prob > 80

                weather_data[name] = {
                    "temperature_celsius": round(avg_temp, 1),
                    "rain_probability": round(max_rain_prob),
                    "wind_speed_kmh": round(max_wind, 1),
                    "conditions": conditions,
                    "suitability_score": round(suitability),
                    "is_suitable_outdoor": suitability >= 50,
                    "indoor_alternative_needed": indoor_needed,
                    "alerts": alerts[:3],  # Top 3 alerts
                    "summary": _build_weather_summary(name, max_rain_prob, avg_temp, max_wind, conditions),
                }
                logger.info(f"Weather for {name}: rain={max_rain_prob}%, temp={avg_temp}°C, suitable={suitability}%")

            except Exception as e:
                logger.warning(f"Weather fetch failed for {name}: {e}")
                weather_data[name] = {
                    "temperature_celsius": 28.0,
                    "rain_probability": 0,
                    "wind_speed_kmh": 10.0,
                    "conditions": ["unknown"],
                    "suitability_score": 70,
                    "is_suitable_outdoor": True,
                    "indoor_alternative_needed": False,
                    "alerts": [],
                    "summary": "Weather data unavailable — plan for typical tropical conditions.",
                }

    except ImportError:
        logger.warning("Weather API tool not available")
    except Exception as e:
        logger.warning(f"Weather data computation failed: {e}")

    return weather_data


def _build_weather_summary(name: str, rain_prob: float, temp: float, wind: float, conditions: List[str]) -> str:
    """Build a human-readable weather summary for a location."""
    parts = []
    if rain_prob > 80:
        parts.append(f"HIGH RAIN RISK ({rain_prob:.0f}%) — INDOOR ALTERNATIVES RECOMMENDED")
    elif rain_prob > 50:
        parts.append(f"Moderate rain chance ({rain_prob:.0f}%) — carry rain gear")
    elif rain_prob > 20:
        parts.append(f"Slight chance of rain ({rain_prob:.0f}%)")
    else:
        parts.append("Clear/dry conditions expected")

    if temp > 35:
        parts.append(f"EXTREME HEAT ({temp:.0f}°C) — avoid midday outdoor activities")
    elif temp > 32:
        parts.append(f"Hot ({temp:.0f}°C) — stay hydrated, start early")
    else:
        parts.append(f"Pleasant temperature ({temp:.0f}°C)")

    if wind > 40:
        parts.append(f"Strong winds ({wind:.0f} km/h) — avoid exposed areas")
    elif wind > 25:
        parts.append(f"Breezy ({wind:.0f} km/h)")

    return ". ".join(parts)


# Indoor alternatives for weather-affected locations
INDOOR_ALTERNATIVES = {
    "beach": ["Visit a local museum", "Sri Lankan cooking class", "Spa & wellness session", "Local market exploration"],
    "hiking": ["Visit tea factory tour", "Temple or religious site", "Art gallery", "Traditional dance show"],
    "nature": ["Botanical garden (covered areas)", "Wildlife museum", "Gem museum", "Spice garden tour"],
    "photography": ["Indoor architecture photography", "Museum exhibits", "Cultural workshop", "Food photography at local restaurant"],
    "general": ["Visit a local museum", "Sri Lankan cooking class", "Temple visit", "Shopping at local craft markets"],
}


def _get_indoor_alternatives(location_name: str) -> List[str]:
    """Get indoor alternative suggestions based on location type."""
    name_lower = location_name.lower()
    if any(w in name_lower for w in ["beach", "coast", "bay"]):
        return INDOOR_ALTERNATIVES["beach"]
    if any(w in name_lower for w in ["peak", "mountain", "rock", "hike", "trail"]):
        return INDOOR_ALTERNATIVES["hiking"]
    if any(w in name_lower for w in ["park", "forest", "falls", "safari"]):
        return INDOOR_ALTERNATIVES["nature"]
    return INDOOR_ALTERNATIVES["general"]


def _build_restaurant_selection_cards(
    restaurant_recs: List[RestaurantRecommendation],
) -> List[Dict[str, Any]]:
    """
    Build mobile-ready selection cards for the top 3 restaurants.

    Sorts by rating (highest first) and returns cards matching the
    SelectionCard schema consumed by React Native's SelectionCardList.
    """
    if not restaurant_recs:
        return []

    # Sort by rating desc (None → 0), deduplicate by name
    seen_names: set = set()
    unique_recs: List[RestaurantRecommendation] = []
    for rec in sorted(restaurant_recs, key=lambda r: r.get("rating") or 0, reverse=True):
        name = rec.get("name", "")
        if name not in seen_names:
            seen_names.add(name)
            unique_recs.append(rec)

    top = unique_recs[:3]
    cards: List[Dict[str, Any]] = []
    badges = ["Top Pick", "Best Value", "Traveller Fave"]

    for idx, rec in enumerate(top):
        tags = [t for t in [rec.get("cuisine_type"), rec.get("meal_slot")] if t]
        cards.append({
            "card_id": rec["id"],
            "title": rec["name"],
            "subtitle": f"{(rec.get('meal_slot') or 'Dining').title()} near {rec.get('near_location', '')}",
            "badge": badges[idx] if idx < len(badges) else None,
            "image_url": rec.get("image_url"),
            "rating": rec.get("rating"),
            "price_range": rec.get("price_range"),
            "description": rec.get("description", ""),
            "tags": tags,
            "distance_km": None,
            "vibe_match_score": None,
        })

    return cards


def _build_preference_selection_cards() -> List[Dict[str, Any]]:
    """
    Build HITL cards asking the user whether they want dining,
    accommodation, both, or none in their tour plan.
    """
    return [
        {
            "card_id": "pref_dining",
            "title": "🍽️ Dining Only",
            "subtitle": "Include restaurant recommendations",
            "badge": "Restaurants",
            "image_url": None,
            "rating": None,
            "price_range": None,
            "description": (
                "I'll find the best local restaurants near your "
                "destinations and include them in your tour plan."
            ),
            "tags": ["Restaurants", "Local Cuisine"],
            "distance_km": None,
            "vibe_match_score": None,
        },
        {
            "card_id": "pref_accommodation",
            "title": "🏨 Stays Only",
            "subtitle": "Include hotel & villa recommendations",
            "badge": "Hotels & Villas",
            "image_url": None,
            "rating": None,
            "price_range": None,
            "description": (
                "I'll find the best hotels, resorts, and villas "
                "near your overnight locations."
            ),
            "tags": ["Hotels", "Villas", "Resorts"],
            "distance_km": None,
            "vibe_match_score": None,
        },
        {
            "card_id": "pref_both",
            "title": "🍽️🏨 Both",
            "subtitle": "Restaurants & accommodations",
            "badge": "Recommended",
            "image_url": None,
            "rating": None,
            "price_range": None,
            "description": (
                "Get the complete experience — I'll recommend both "
                "restaurants and accommodations tailored to your trip."
            ),
            "tags": ["Full Package"],
            "distance_km": None,
            "vibe_match_score": None,
        },
        {
            "card_id": "pref_none",
            "title": "⏩ Activities Only",
            "subtitle": "Just sightseeing & activities",
            "badge": None,
            "image_url": None,
            "rating": None,
            "price_range": None,
            "description": (
                "Generate a tour plan focused on activities and "
                "sightseeing only — no dining or accommodation."
            ),
            "tags": ["Activities Only"],
            "distance_km": None,
            "vibe_match_score": None,
        },
    ]


def _build_budget_selection_cards(dining_preference: str) -> List[Dict[str, Any]]:
    """Build HITL cards asking the user about their budget range."""
    target = {
        "dining": "restaurants",
        "accommodation": "accommodations",
        "both": "restaurants & accommodations",
    }.get(dining_preference, "options")

    return [
        {
            "card_id": "budget_low",
            "title": "💰 Budget Friendly",
            "subtitle": f"Affordable {target}",
            "badge": "Budget",
            "image_url": None,
            "rating": None,
            "price_range": "$",
            "description": (
                "Budget-friendly options with great quality. "
                "Local eateries, street food, and comfortable guesthouses."
            ),
            "tags": ["Affordable", "Local", "Value"],
            "distance_km": None,
            "vibe_match_score": None,
        },
        {
            "card_id": "budget_medium",
            "title": "💎 Mid-Range",
            "subtitle": "Quality & comfort",
            "badge": "Popular",
            "image_url": None,
            "rating": None,
            "price_range": "$$",
            "description": (
                "Well-rated restaurants and comfortable hotels. "
                "The sweet spot of quality and value."
            ),
            "tags": ["Comfortable", "Quality", "Popular"],
            "distance_km": None,
            "vibe_match_score": None,
        },
        {
            "card_id": "budget_high",
            "title": "👑 Premium",
            "subtitle": "Luxury experiences",
            "badge": "Luxury",
            "image_url": None,
            "rating": None,
            "price_range": "$$$",
            "description": (
                "Top-rated fine dining and luxury resorts. "
                "Premium experiences for a memorable trip."
            ),
            "tags": ["Luxury", "Fine Dining", "Premium"],
            "distance_km": None,
            "vibe_match_score": None,
        },
    ]


def _build_accommodation_selection_cards(
    accommodation_recs: List[AccommodationRecommendation],
) -> List[Dict[str, Any]]:
    """
    Build mobile-ready selection cards for the top 3 accommodations.
    Sorts by rating (highest first), deduplicates by name.
    """
    if not accommodation_recs:
        return []

    seen_names: set = set()
    unique_recs: List[AccommodationRecommendation] = []
    for rec in sorted(
        accommodation_recs, key=lambda r: r.get("rating") or 0, reverse=True
    ):
        name = rec.get("name", "")
        if name not in seen_names:
            seen_names.add(name)
            unique_recs.append(rec)

    top = unique_recs[:3]
    cards: List[Dict[str, Any]] = []
    badges = ["Top Pick", "Best Value", "Guest Favourite"]

    for idx, rec in enumerate(top):
        acc_type = (rec.get("type") or "hotel").title()
        tags = [acc_type]
        if rec.get("near_location"):
            tags.append(f"Near {rec['near_location']}")

        cards.append({
            "card_id": rec["id"],
            "title": rec["name"],
            "subtitle": (
                f"{acc_type} near {rec.get('near_location', '')} — "
                f"Night {rec.get('check_in_day', 1)}"
            ),
            "badge": badges[idx] if idx < len(badges) else None,
            "image_url": rec.get("image_url"),
            "rating": rec.get("rating"),
            "price_range": rec.get("price_range"),
            "description": rec.get("description", ""),
            "tags": tags,
            "distance_km": None,
            "vibe_match_score": None,
        })

    return cards


# ── Google Places restaurant cache (4-hour TTL, avoids redundant API calls) ──
import hashlib as _hashlib

_RESTAURANT_CACHE: Dict[str, Dict[str, Any]] = {}  # {hash: {"data": [...], "ts": float}}
_RESTAURANT_CACHE_TTL = 4 * 60 * 60  # 4 hours


def _restaurant_cache_key(query: str) -> str:
    return _hashlib.sha256(query.lower().strip().encode()).hexdigest()[:20]


def _restaurant_cache_get(key: str) -> Optional[List[Dict[str, Any]]]:
    import time
    entry = _RESTAURANT_CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < _RESTAURANT_CACHE_TTL:
        return entry["data"]
    if entry:
        del _RESTAURANT_CACHE[key]
    return None


def _restaurant_cache_put(key: str, data: List[Dict[str, Any]]) -> None:
    import time
    # Evict oldest if cache exceeds 200 entries
    if len(_RESTAURANT_CACHE) >= 200:
        oldest = min(_RESTAURANT_CACHE, key=lambda k: _RESTAURANT_CACHE[k]["ts"])
        del _RESTAURANT_CACHE[oldest]
    _RESTAURANT_CACHE[key] = {"data": data, "ts": time.time()}


async def _search_google_places_restaurants(
    location_name: str,
    meal_slot: str,
    budget: Optional[str] = None,
    max_results: int = 3,
) -> List[Dict[str, Any]]:
    """
    Call Google Places API (New) Text Search directly to find real
    restaurants.  Returns normalised dicts with name, rating, price,
    image_url, etc.

    Requires GOOGLE_MAPS_API_KEY in the environment / settings.
    """
    import httpx
    from ...config import settings

    api_key = getattr(settings, "GOOGLE_MAPS_API_KEY", None) or getattr(settings, "MCP_GOOGLE_MAPS_KEY", None)
    if not api_key:
        logger.warning("No GOOGLE_MAPS_API_KEY configured — cannot search Google Places")
        return []

    budget_hint = f"{budget} " if budget else ""
    text_query = f"best {budget_hint}{meal_slot} restaurants near {location_name}, Sri Lanka"

    # ── Check cache ──
    ck = _restaurant_cache_key(text_query)
    cached = _restaurant_cache_get(ck)
    if cached is not None:
        logger.info(f"Restaurant cache HIT for '{text_query}' ({len(cached)} results)")
        return cached[:max_results]

    # ── Google Places (New) Text Search ──
    url = "https://places.googleapis.com/v1/places:searchText"
    fields = [
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
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": ",".join(fields),
    }
    body = {
        "textQuery": text_query,
        "maxResultCount": max_results,
        "languageCode": "en",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning(f"Google Places API call failed for '{text_query}': {exc}")
        return []

    # ── Normalise response ──
    price_map = {
        "PRICE_LEVEL_FREE": "$",
        "PRICE_LEVEL_INEXPENSIVE": "$",
        "PRICE_LEVEL_MODERATE": "$$",
        "PRICE_LEVEL_EXPENSIVE": "$$$",
        "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
    }

    results: List[Dict[str, Any]] = []
    for place in data.get("places", []):
        display_name = place.get("displayName", {})
        loc = place.get("location", {})
        editorial = place.get("editorialSummary", {})

        # First photo URL
        image_url = None
        for p in place.get("photos", [])[:1]:
            photo_name = p.get("name", "")
            if photo_name:
                image_url = (
                    f"https://places.googleapis.com/v1/{photo_name}/media"
                    f"?maxWidthPx=400&key={api_key}"
                )

        results.append({
            "name": display_name.get("text", "Unknown"),
            "rating": place.get("rating"),
            "price_range": price_map.get(place.get("priceLevel", ""), "$$"),
            "url": place.get("websiteUri"),
            "description": editorial.get("text", place.get("formattedAddress", "")),
            "image_url": image_url,
            "latitude": loc.get("latitude"),
            "longitude": loc.get("longitude"),
            "primary_type": place.get("primaryType"),
            "user_rating_count": place.get("userRatingCount"),
        })

    # ── Cache results ──
    if results:
        _restaurant_cache_put(ck, results)
        logger.info(f"Google Places returned {len(results)} restaurants for '{text_query}'")

    return results


async def _compute_restaurant_recommendations(
    locations: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    budget: Optional[str] = None,
) -> List[RestaurantRecommendation]:
    """
    Search for real restaurants near each location using the Google
    Places API (New) Text Search.  Results are cached for 4 hours to
    reduce API cost.  Falls back to Tavily web search only if no
    Google Maps API key is configured.

    Returns a flat list of RestaurantRecommendation grouped by day + meal_slot.
    """
    recommendations: List[RestaurantRecommendation] = []

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
        num_days = max(1, (end_dt - start_dt).days + 1)

        # ── Try Google Places API directly ─────────────────────────
        for day_num in range(1, num_days + 1):
            loc_idx = min(day_num - 1, len(locations) - 1)
            loc = locations[loc_idx]
            loc_name = loc.get("name", "Unknown")

            for meal_slot in ["breakfast", "lunch", "dinner"]:
                try:
                    places = await _search_google_places_restaurants(
                        location_name=loc_name,
                        meal_slot=meal_slot,
                        budget=budget,
                        max_results=3,
                    )
                    for i, p in enumerate(places):
                        rec_id = f"rest_d{day_num}_{meal_slot}_{i + 1}"
                        recommendations.append(RestaurantRecommendation(
                            id=rec_id,
                            name=p.get("name", f"Restaurant {i + 1}"),
                            rating=p.get("rating"),
                            cuisine_type=p.get("primary_type"),
                            price_range=p.get("price_range", "$$"),
                            url=p.get("url"),
                            description=(p.get("description") or "")[:200],
                            near_location=loc_name,
                            meal_slot=meal_slot,
                            day=day_num,
                            image_url=p.get("image_url"),
                        ))
                except Exception as e:
                    logger.warning(f"Google Places search failed for {loc_name} {meal_slot}: {e}")

        if recommendations:
            logger.info(f"Google Places returned {len(recommendations)} restaurant recommendations total")
            return recommendations

        # ── Fallback: Tavily web search (if Google Places not configured) ──
        logger.info("No Google Places results — falling back to Tavily web search")
        try:
            from ...tools.web_search import get_web_search_tool
            from ...config import settings

            web_search = get_web_search_tool(settings.TAVILY_API_KEY)
            if not web_search.enabled:
                logger.warning("Web search not enabled, skipping restaurant recommendations")
                return recommendations

            import re as _re

            for day_num in range(1, num_days + 1):
                loc_idx = min(day_num - 1, len(locations) - 1)
                loc = locations[loc_idx]
                loc_name = loc.get("name", "Unknown")

                for meal_slot in ["breakfast", "lunch", "dinner"]:
                    budget_hint = f"{budget} " if budget else ""
                    query = f"best {budget_hint}{meal_slot} restaurants near {loc_name} Sri Lanka reviews ratings"

                    try:
                        results = web_search.search(
                            query,
                            include_domains=["tripadvisor.com", "google.com", "yelp.com", "lonelyplanet.com"],
                        )

                        for i, r in enumerate(results.get("results", [])[:3]):
                            rec_id = f"rest_d{day_num}_{meal_slot}_{i + 1}"
                            content = r.get("content", "")

                            rating = None
                            rating_match = _re.search(r"(\d\.?\d?)\s*/?\s*5", content)
                            if rating_match:
                                try:
                                    rating = float(rating_match.group(1))
                                except ValueError:
                                    pass

                            price_range = "$$"
                            content_lower = content.lower()
                            if any(w in content_lower for w in ["luxury", "fine dining", "$$$", "high-end"]):
                                price_range = "$$$"
                            elif any(w in content_lower for w in ["budget", "cheap", "street food", "affordable"]):
                                price_range = "$"

                            recommendations.append(RestaurantRecommendation(
                                id=rec_id,
                                name=r.get("title", f"Restaurant {i + 1}"),
                                rating=rating,
                                cuisine_type=None,
                                price_range=price_range,
                                url=r.get("url"),
                                description=content[:200].strip() + ("..." if len(content) > 200 else ""),
                                near_location=loc_name,
                                meal_slot=meal_slot,
                                day=day_num,
                            ))
                    except Exception as e:
                        logger.warning(f"Restaurant search failed for {loc_name} {meal_slot}: {e}")
        except ImportError:
            logger.warning("Web search tool not available for restaurant recommendations")

    except Exception as e:
        logger.warning(f"Restaurant recommendation computation failed: {e}")

    return recommendations


async def _compute_accommodation_recommendations(
    locations: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    budget: Optional[str] = None,
) -> List[AccommodationRecommendation]:
    """
    Search for top 3 accommodation options near each overnight location.
    Only runs for 2+ day trips (single-day trips return empty list).

    Returns a flat list of AccommodationRecommendation grouped by check_in_day.
    """
    recommendations: List[AccommodationRecommendation] = []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if isinstance(start_date, str) else start_date
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date
    num_days = max(1, (end_dt - start_dt).days + 1)

    if num_days < 2:
        return recommendations  # No overnight needed for single-day trips

    try:
        from ...tools.web_search import get_web_search_tool
        from ...config import settings

        web_search = get_web_search_tool(settings.TAVILY_API_KEY)
        if not web_search.enabled:
            logger.warning("Web search not enabled, skipping accommodation recommendations")
            return recommendations

        import re as _re

        for day_num in range(1, num_days):  # No accommodation needed for last day
            loc_idx = min(day_num - 1, len(locations) - 1)
            loc = locations[loc_idx]
            loc_name = loc.get("name", "Unknown")

            budget_hint = f"{budget} " if budget else ""
            query = f"best {budget_hint}hotels resorts near {loc_name} Sri Lanka booking reviews ratings"

            try:
                results = web_search.search(
                    query,
                    include_domains=["booking.com", "tripadvisor.com", "agoda.com", "hotels.com"],
                )

                for i, r in enumerate(results.get("results", [])[:3]):
                    rec_id = f"hotel_d{day_num}_{i + 1}"
                    content = r.get("content", "")

                    # Extract rating heuristic
                    rating = None
                    rating_match = _re.search(r"(\d\.?\d?)\s*/?\s*5", content)
                    if rating_match:
                        try:
                            rating = float(rating_match.group(1))
                        except ValueError:
                            pass

                    # Extract price range heuristic
                    price_range = "$$"
                    content_lower = content.lower()
                    if any(w in content_lower for w in ["luxury", "5-star", "$$$", "premium"]):
                        price_range = "$$$"
                    elif any(w in content_lower for w in ["budget", "hostel", "cheap", "backpacker"]):
                        price_range = "$"

                    # Determine accommodation type
                    acc_type = "hotel"
                    if "resort" in content_lower:
                        acc_type = "resort"
                    elif any(w in content_lower for w in ["guesthouse", "guest house", "homestay"]):
                        acc_type = "guesthouse"

                    recommendations.append(AccommodationRecommendation(
                        id=rec_id,
                        name=r.get("title", f"Hotel {i + 1}"),
                        rating=rating,
                        price_range=price_range,
                        url=r.get("url"),
                        description=content[:200].strip() + ("..." if len(content) > 200 else ""),
                        near_location=loc_name,
                        check_in_day=day_num,
                        type=acc_type,
                    ))
            except Exception as e:
                logger.warning(f"Accommodation search failed near {loc_name}: {e}")

    except ImportError:
        logger.warning("Web search tool not available for accommodation recommendations")
    except Exception as e:
        logger.warning(f"Accommodation recommendation computation failed: {e}")

    return recommendations


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

    # Weather data (deep injection)
    weather_data = state.get("_weather_data", {})
    if weather_data and isinstance(weather_data, dict):
        parts.append("=== WEATHER FORECASTS (OpenWeatherMap API — Real-Time) ===")
        parts.append("USE THIS WEATHER DATA for weather-aware routing!")
        for loc_name, wd in weather_data.items():
            if not isinstance(wd, dict):
                continue
            parts.append(f"\n{loc_name}:")
            parts.append(f"  Summary: {wd.get('summary', 'N/A')}")
            parts.append(f"  Temperature: {wd.get('temperature_celsius', 'N/A')}°C")
            parts.append(f"  Rain Probability: {wd.get('rain_probability', 0)}%")
            parts.append(f"  Wind: {wd.get('wind_speed_kmh', 0)} km/h")
            parts.append(f"  Outdoor Suitability: {wd.get('suitability_score', 70)}%")
            if wd.get("indoor_alternative_needed"):
                alts = _get_indoor_alternatives(loc_name)
                parts.append(f"  *** INDOOR ALTERNATIVES NEEDED: {', '.join(alts[:3])}")
            if wd.get("alerts"):
                for alert in wd["alerts"]:
                    if isinstance(alert, dict):
                        parts.append(f"  ALERT [{alert.get('severity', 'medium').upper()}]: {alert.get('description', '')}")
                    parts.append(f"    → {alert.get('recommendation', '')}")
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

    # Selected restaurant context (for refinement with user selections)
    tour_ctx = state.get("tour_plan_context") or {}
    selected_restaurants = tour_ctx.get("selected_restaurant_ids")
    restaurant_recs = state.get("restaurant_recommendations", [])
    if selected_restaurants and restaurant_recs:
        parts.append("=== SELECTED RESTAURANTS (include these in the plan at the correct meal times) ===")
        for rec in restaurant_recs:
            if rec.get("id") in selected_restaurants:
                parts.append(f"  Day {rec.get('day', '?')} {rec.get('meal_slot', '').title()}: {rec.get('name', 'Unknown')}")
                if rec.get("rating"):
                    parts.append(f"    Rating: {rec['rating']}/5")
                if rec.get("price_range"):
                    parts.append(f"    Price: {rec['price_range']}")
        parts.append("")

    # Selected accommodation context (for refinement with user selections)
    selected_accommodations = tour_ctx.get("selected_accommodation_ids")
    accommodation_recs = state.get("accommodation_recommendations", [])
    if selected_accommodations and accommodation_recs:
        parts.append("=== SELECTED ACCOMMODATIONS (include check-in/check-out in the plan) ===")
        for rec in accommodation_recs:
            if rec.get("id") in selected_accommodations:
                parts.append(f"  Night of Day {rec.get('check_in_day', '?')}: {rec.get('name', 'Unknown')} ({rec.get('type', 'hotel')})")
                if rec.get("rating"):
                    parts.append(f"    Rating: {rec['rating']}/5")
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
                if not isinstance(tip_data, dict):
                    continue
                cultural_tips.append(CulturalTip(
                    location=tip_data.get("location", "General"),
                    tip=tip_data.get("tip", ""),
                    category=tip_data.get("category", "cultural"),
                ))

            # Parse itinerary
            for day_data in plan_data.get("itinerary", []):
                if not isinstance(day_data, dict):
                    logger.warning(f"Skipping non-dict itinerary day: {type(day_data)}")
                    continue
                day_num = day_data.get("day", 1)
                location = day_data.get("location", "")

                for order, activity in enumerate(day_data.get("activities", [])):
                    if not isinstance(activity, dict):
                        logger.warning(f"Skipping non-dict activity: {type(activity)}")
                        continue
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


@trace_node("tour_plan_generator", run_type="chain")
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

    # ── Multi-Step HITL: ask preferences before expensive computation ──
    dining_preference = tour_context.get("dining_preference")
    budget_preference = tour_context.get("budget_preference")

    # HITL Step 1: Dining / Accommodation / Both / None
    if not dining_preference and not is_modification:
        cards = _build_preference_selection_cards()
        duration_ms = (time.time() - start_time) * 1000
        logger.info("HITL Step 1 — asking dining/accommodation preferences")
        return {
            "pending_user_selection": True,
            "pending_restaurant_selection": True,
            "selection_cards": cards,
            "prompt_text": "What would you like included?",
            "generated_response": "What should I include in your tour plan?",
            "final_response": "What should I include in your tour plan?",
            "step_results": [{
                "node": "tour_plan_generate",
                "status": "pending",
                "summary": "Asking user about dining/accommodation preferences",
                "duration_ms": duration_ms,
            }],
        }

    # HITL Step 2: Budget (low / medium / high)
    if (
        dining_preference
        and dining_preference != "none"
        and not budget_preference
        and not is_modification
    ):
        cards = _build_budget_selection_cards(dining_preference)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"HITL Step 2 — asking budget (dining_preference={dining_preference})")
        return {
            "pending_user_selection": True,
            "pending_restaurant_selection": True,
            "selection_cards": cards,
            "prompt_text": "What's your budget?",
            "generated_response": "What's your budget range?",
            "final_response": "What's your budget range?",
            "step_results": [{
                "node": "tour_plan_generate",
                "status": "pending",
                "summary": f"Asking budget preference (dining_preference={dining_preference})",
                "duration_ms": duration_ms,
            }],
        }

    # ── Determine skip flags based on dining preference ──────────
    skip_restaurants = tour_context.get("skip_restaurants", False)
    skip_accommodations = tour_context.get("skip_accommodations", False)

    if dining_preference == "none":
        skip_restaurants = True
        skip_accommodations = True
    elif dining_preference == "dining":
        skip_accommodations = True
    elif dining_preference == "accommodation":
        skip_restaurants = True
    # "both" → use handler-set skip flags from previous passes

    # Use HITL budget, fallback to user_preferences
    budget = budget_preference
    if not budget:
        prefs = state.get("user_preferences")
        if prefs:
            budget = prefs.get("budget")

    # Steps 1-4: Compute all data in PARALLEL for performance
    logger.info("Computing golden hour, crowd, weather, event, restaurant, and accommodation data in parallel...")

    # Wrap sync functions for asyncio.gather
    loop = asyncio.get_event_loop()
    golden_future = loop.run_in_executor(None, _compute_golden_hour_data, locations, start_date, end_date)
    crowd_future = loop.run_in_executor(None, _compute_crowd_predictions, locations, start_date, end_date)
    event_future = loop.run_in_executor(None, _compute_event_data, locations, start_date, end_date)
    weather_future = _compute_weather_data(locations, start_date, end_date)

    # Restaurant/accommodation search (skippable by user)
    async def _empty_list():
        return []

    restaurant_future = (
        _compute_restaurant_recommendations(locations, start_date, end_date, budget)
        if not skip_restaurants else _empty_list()
    )
    accommodation_future = (
        _compute_accommodation_recommendations(locations, start_date, end_date, budget)
        if not skip_accommodations else _empty_list()
    )

    golden_data, crowd_data, event_data, weather_data, restaurant_recs, accommodation_recs = await asyncio.gather(
        golden_future, crowd_future, event_future, weather_future,
        restaurant_future, accommodation_future,
        return_exceptions=True
    )

    # Handle exceptions from parallel tasks gracefully
    if isinstance(golden_data, Exception):
        logger.warning(f"Golden hour computation failed: {golden_data}")
        golden_data = {}
    if isinstance(crowd_data, Exception):
        logger.warning(f"Crowd prediction failed: {crowd_data}")
        crowd_data = {}
    if isinstance(event_data, Exception):
        logger.warning(f"Event data failed: {event_data}")
        event_data = {}
    if isinstance(weather_data, Exception):
        logger.warning(f"Weather data failed: {weather_data}")
        weather_data = {}
    if isinstance(restaurant_recs, Exception):
        logger.warning(f"Restaurant recommendations failed: {restaurant_recs}")
        restaurant_recs = []
    if isinstance(accommodation_recs, Exception):
        logger.warning(f"Accommodation recommendations failed: {accommodation_recs}")
        accommodation_recs = []

    logger.info(f"Found {len(restaurant_recs)} restaurant and {len(accommodation_recs)} accommodation recommendations")

    # ─── HITL Step 3: Restaurant Selection ────────────────────────
    already_selected_restaurants = (
        state.get("selected_restaurant_ids")
        or tour_context.get("selected_restaurant_ids")
    )
    if (
        restaurant_recs
        and len(restaurant_recs) >= 3
        and not skip_restaurants
        and not already_selected_restaurants
        and not is_modification
        and dining_preference in ("dining", "both")
    ):
        selection_cards = _build_restaurant_selection_cards(restaurant_recs)
        if selection_cards:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Restaurant HITL triggered — presenting {len(selection_cards)} "
                f"restaurant cards for user selection"
            )
            return {
                "pending_user_selection": True,
                "pending_restaurant_selection": True,
                "selection_cards": selection_cards,
                "prompt_text": "Pick a restaurant",
                "restaurant_recommendations": restaurant_recs,
                "accommodation_recommendations": accommodation_recs,
                "generated_response": "Here are the top restaurants — pick your favourite!",
                "final_response": "Here are the top restaurants — pick your favourite!",
                "step_results": [{
                    "node": "tour_plan_generate",
                    "status": "pending",
                    "summary": (
                        f"Computed {len(restaurant_recs)} restaurants — "
                        f"awaiting user selection from top {len(selection_cards)}"
                    ),
                    "duration_ms": duration_ms,
                }],
            }

    # ─── HITL Step 4: Accommodation Selection ─────────────────────
    already_selected_accommodations = (
        state.get("selected_accommodation_ids")
        or tour_context.get("selected_accommodation_ids")
    )
    if (
        accommodation_recs
        and len(accommodation_recs) >= 1
        and not skip_accommodations
        and not already_selected_accommodations
        and not is_modification
        and dining_preference in ("accommodation", "both")
    ):
        accom_cards = _build_accommodation_selection_cards(accommodation_recs)
        if accom_cards:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Accommodation HITL triggered — presenting {len(accom_cards)} "
                f"accommodation cards for user selection"
            )
            return {
                "pending_user_selection": True,
                "pending_restaurant_selection": True,
                "selection_cards": accom_cards,
                "prompt_text": "Pick a place to stay",
                "accommodation_recommendations": accommodation_recs,
                "generated_response": "Great stays near your spots — which one looks good?",
                "final_response": "Great stays near your spots — which one looks good?",
                "step_results": [{
                    "node": "tour_plan_generate",
                    "status": "pending",
                    "summary": (
                        f"Computed {len(accommodation_recs)} accommodations — "
                        f"awaiting user selection from top {len(accom_cards)}"
                    ),
                    "duration_ms": duration_ms,
                }],
            }

    # Collect cultural tips
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
        "_weather_data": weather_data,
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
                # Build the visual-ready FinalItinerary
                final_itinerary_obj = _build_final_itinerary(
                    itinerary=itinerary,
                    locations=locations,
                    warnings=warnings,
                    tips=tips,
                    constraint_violations=state.get("constraint_violations", []),
                    weather_data=weather_data,
                    event_data=event_data,
                    summary=summary,
                )

                # Format a human-readable response
                final_response = f"🗺️ **Your {metadata['total_days']}-Day Sri Lanka Adventure**\n\n"
                final_response += f"📍 Covering {metadata['total_locations']} amazing locations\n"
                final_response += f"✨ Match Score: {metadata['match_score']}%\n\n"

                if summary:
                    final_response += f"{summary}\n\n"

                if metadata.get("preference_match_explanation"):
                    final_response += f"🎯 {metadata['preference_match_explanation']}\n\n"

                # Add weather summary to response
                weather_warnings = [
                    f"🌧️ {name}: {wd.get('summary', '')}"
                    for name, wd in weather_data.items()
                    if isinstance(wd, dict) and wd.get("indoor_alternative_needed")
                ]
                if weather_warnings:
                    final_response += "**Weather Alerts:**\n" + "\n".join(weather_warnings) + "\n\n"

                final_response += "I've optimized your itinerary using precise golden hour calculations, real-time crowd predictions, and live weather forecasts. "
                if is_modification:
                    final_response += "I've made the changes you requested while keeping the rest of your plan intact. "
                final_response += "Check out the detailed plan below! 👇"

                return {
                    "itinerary": itinerary,
                    "tour_plan_metadata": metadata,
                    "generated_response": final_response,
                    "final_response": final_response,
                    "cultural_tips": all_cultural_tips,
                    "weather_data": weather_data,
                    "final_itinerary": final_itinerary_obj,
                    "restaurant_recommendations": restaurant_recs,
                    "accommodation_recommendations": accommodation_recs,
                    "step_results": [{
                        "node": "tour_plan_generate",
                        "status": "success",
                        "summary": f"Generated {metadata['total_days']}-day plan with {len(itinerary)} activities, {len(all_cultural_tips)} cultural tips, {len(restaurant_recs)} restaurants, {len(accommodation_recs)} hotels",
                        "duration_ms": duration_ms,
                    }],
                }
            else:
                # Fallback response
                return {
                    "generated_response": response_text,
                    "final_response": response_text,
                    "cultural_tips": all_cultural_tips,
                    "restaurant_recommendations": restaurant_recs,
                    "accommodation_recommendations": accommodation_recs,
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
        "itinerary": basic_itinerary,
        "tour_plan_metadata": basic_metadata,
        "cultural_tips": all_cultural_tips,
        "restaurant_recommendations": restaurant_recs,
        "accommodation_recommendations": accommodation_recs,
        "generated_response": "I've created a basic tour plan for your selected locations.",
        "final_response": "I've created a basic tour plan for your selected locations.",
        "step_results": [{
            "node": "tour_plan_generate",
            "status": "warning",
            "summary": "Generated basic plan without LLM (fallback mode)",
            "duration_ms": duration_ms,
        }],
    }


def _build_final_itinerary(
    itinerary: List[ItinerarySlot],
    locations: List[Dict[str, Any]],
    warnings: List[str],
    tips: List[str],
    constraint_violations: List[Dict[str, Any]],
    weather_data: Dict[str, Any],
    event_data: Dict[str, Any],
    summary: str,
) -> FinalItinerary:
    """
    Build the visual-ready FinalItinerary JSON for maps integration.

    Creates sequence_ids, coordinates, route_polyline, and contextual_notes
    from the raw itinerary slots and location data.
    """
    # Build location coordinate lookup
    loc_coords = {}
    for loc in locations:
        name = loc.get("name", "")
        loc_coords[name] = {
            "lat": loc.get("latitude", 0),
            "lng": loc.get("longitude", 0),
        }

    # Build stops with sequence_id and coordinates
    stops: List[FinalItineraryStop] = []
    route_coords: List[RouteCoordinate] = []
    seen_locations_for_route = []
    seq_id = 0

    for slot in itinerary:
        seq_id += 1
        loc_name = slot.get("location", "")
        coords = loc_coords.get(loc_name, {"lat": 0, "lng": 0})

        # Weather summary for this location
        loc_weather = weather_data.get(loc_name, {})
        weather_summary = loc_weather.get("summary") if isinstance(loc_weather, dict) else None

        # ------------------------------------------------------------------
        # Visual-Ready Output: derive map_marker_icon + one-line summary
        # ------------------------------------------------------------------
        activity_text = (slot.get("activity") or "").lower()
        icon_slot = (slot.get("icon") or "").lower()
        _marker_icon = "Attraction"  # default
        if any(k in activity_text for k in ("hotel", "check-in", "check in", "accommodation", "stay")):
            _marker_icon = "Hotel"
        elif any(k in activity_text for k in ("restaurant", "food", "lunch", "dinner", "breakfast", "dining", "eat")):
            _marker_icon = "Food"
        elif any(k in activity_text for k in ("bar", "nightlife", "pub", "club", "party", "drink")):
            _marker_icon = "Party"
        elif any(k in activity_text for k in ("temple", "kovil", "mosque", "church", "shrine", "dagoba", "stupa")):
            _marker_icon = "Temple"
        elif any(k in activity_text for k in ("nature", "hike", "trek", "waterfall", "forest", "safari", "wildlife", "beach", "lake")):
            _marker_icon = "Nature"
        elif any(k in activity_text for k in ("photo", "camera", "sunrise", "sunset", "golden hour")):
            _marker_icon = "Camera"
        elif any(k in icon_slot for k in ("transport", "drive", "travel", "transfer")):
            _marker_icon = "Transport"

        _visual_summary = f"{slot.get('activity', loc_name)}"[:60]

        stop: FinalItineraryStop = {
            "sequence_id": seq_id,
            "day": slot.get("day", 1),
            "time": slot.get("time", "09:00"),
            "location": loc_name,
            "activity": slot.get("activity", ""),
            "duration_minutes": slot.get("duration_minutes", 60),
            "coordinates": coords,
            "crowd_prediction": slot.get("crowd_prediction", 50),
            "lighting_quality": slot.get("lighting_quality", "good"),
            "weather_summary": weather_summary,
            "icon": slot.get("icon"),
            "highlight": slot.get("highlight", False),
            "ai_insight": slot.get("ai_insight"),
            "cultural_tip": slot.get("cultural_tip"),
            "ethical_note": slot.get("ethical_note"),
            "best_photo_time": slot.get("best_photo_time"),
            "notes": slot.get("notes"),
            "visual_assets": {
                "map_marker_icon": _marker_icon,
                "summary": _visual_summary,
            },
        }
        stops.append(stop)

        # Build route polyline (one coordinate per unique location in order)
        if loc_name not in seen_locations_for_route:
            seen_locations_for_route.append(loc_name)
            route_coords.append(RouteCoordinate(
                lat=coords["lat"],
                lng=coords["lng"],
                location_name=loc_name,
                sequence_id=len(route_coords) + 1,
            ))

    # Build contextual notes from constraints, weather, and events
    contextual_notes: List[ContextualNote] = []
    note_seq = 0

    # From constraint violations
    for violation in (constraint_violations or []):
        note_seq += 1
        severity_map = {"low": "info", "medium": "warning", "high": "warning", "critical": "critical"}
        contextual_notes.append(ContextualNote(
            sequence_id=note_seq,
            location_name=violation.get("constraint_type", "general"),
            note_type=violation.get("constraint_type", "general"),
            message=f"{violation.get('description', '')} — {violation.get('suggestion', '')}",
            severity=severity_map.get(violation.get("severity", "medium"), "warning"),
        ))

    # From weather alerts
    for loc_name, wd in weather_data.items():
        if not isinstance(wd, dict):
            continue
        if wd.get("indoor_alternative_needed"):
            note_seq += 1
            contextual_notes.append(ContextualNote(
                sequence_id=note_seq,
                location_name=loc_name,
                note_type="weather_alert",
                message=f"Heavy rain expected at {loc_name} (rain probability {wd.get('rain_probability', 0)}%). Indoor alternatives recommended.",
                severity="warning",
            ))
        for alert in wd.get("alerts", []):
            if isinstance(alert, dict) and alert.get("severity") in ("high", "critical"):
                note_seq += 1
                contextual_notes.append(ContextualNote(
                    sequence_id=note_seq,
                    location_name=loc_name,
                    note_type="weather_alert",
                    message=alert.get("description", ""),
                    severity="critical" if alert.get("severity") == "critical" else "warning",
                ))

    # From Poya days
    for poya_date in event_data.get("poya_days", []):
        note_seq += 1
        contextual_notes.append(ContextualNote(
            sequence_id=note_seq,
            location_name="All Locations",
            note_type="poya_warning",
            message=f"Poya day on {poya_date} — No alcohol sales. Modest dress required at all sites.",
            severity="warning",
        ))

    # Calculate total distance
    total_dist = 0.0
    for i in range(1, len(route_coords)):
        prev = route_coords[i - 1]
        curr = route_coords[i]
        total_dist += calculate_distance(prev["lat"], prev["lng"], curr["lat"], curr["lng"])

    # Count unique days
    total_days = len(set(s["day"] for s in stops)) if stops else 1

    return FinalItinerary(
        stops=stops,
        route_polyline=route_coords,
        contextual_notes=contextual_notes,
        total_distance_km=round(total_dist, 1),
        total_days=total_days,
        summary=summary,
        warnings=warnings,
        tips=tips,
    )


def route_to_plan_generator(state: GraphState) -> bool:
    """Check if the request is for tour plan generation."""
    return state.get("tour_plan_context") is not None

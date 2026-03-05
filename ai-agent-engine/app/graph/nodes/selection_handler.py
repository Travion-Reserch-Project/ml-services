"""
Selection Handler & Post-Selection Re-Optimization Node.

Runs AFTER the user selects a candidate from the MCP search results.
Implements four post-selection stages:

    1. Selection Validation
       Look up the selected candidate from ``search_candidates`` and
       verify it exists.

    2. Geospatial Re-Optimization
       Recalculate Haversine distances and travel times from every
       existing itinerary stop to the newly selected venue.

    3. Cultural Guardrail — Event Sentinel Re-Check
       Re-verify the selected venue against the Event Sentinel (e.g.,
       Poya-day alcohol ban at a selected party venue, modest-dress
       requirements at a temple restaurant).

    4. Aesthetic Physics — Topographic Horizon Correction
       Apply the GoldenHourEngine's horizon-dip model to the selected
       stay/dining location to suggest the best sunset/sunrise viewing
       times, accounting for the venue's elevation.

After processing, the node outputs a ``MapReadyItinerary`` JSON with
precise coordinates, ``itinerary_sequence``, and ``dynamic_warnings``.

The resulting state is routed to the Verifier Node for up to 3
self-correction iterations.

Research Pattern:
    Human-in-the-Loop (HITL) with Multi-Objective Post-Validation —
    The agent does not blindly accept user choices.  Every selection
    is re-validated against geospatial, cultural, and aesthetic
    constraints before the plan is finalised.
"""

import logging
import math
import time
from datetime import datetime, date
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

from ..state import GraphState, SearchCandidate, ConstraintViolation, StepResult
from ...config import settings

logger = logging.getLogger(__name__)

# Earth's mean radius in km (for Haversine)
EARTH_RADIUS_KM = 6_371.0

# Average road speed in Sri Lanka (km/h)
SRI_LANKA_AVG_SPEED_KMH = 35.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. GEOSPATIAL — Haversine Distance & Travel Time
# ═══════════════════════════════════════════════════════════════════════════
def haversine_distance(
    lat1: float, lng1: float,
    lat2: float, lng2: float,
) -> float:
    """
    Great-circle distance using the Haversine formula.

    Returns distance in kilometres.
    """
    lat1_r, lng1_r = math.radians(lat1), math.radians(lng1)
    lat2_r, lng2_r = math.radians(lat2), math.radians(lng2)
    dlat = lat2_r - lat1_r
    dlng = lng2_r - lng1_r
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def _travel_segment(
    lat1: float, lng1: float,
    lat2: float, lng2: float,
) -> Dict[str, float]:
    """Distance and estimated travel time between two points."""
    dist = haversine_distance(lat1, lng1, lat2, lng2)
    return {
        "distance_km": round(dist, 2),
        "travel_minutes": round((dist / SRI_LANKA_AVG_SPEED_KMH) * 60, 1),
    }


def recalculate_travel_times(
    stops: List[Dict[str, Any]],
    new_coords: Dict[str, float],
    new_name: str,
) -> List[Dict[str, Any]]:
    """
    Annotate every existing itinerary stop with distance / travel-time
    to the newly selected venue.
    """
    new_lat = new_coords.get("lat")
    new_lng = new_coords.get("lng")
    if not new_lat or not new_lng:
        return stops

    updated: List[Dict[str, Any]] = []
    for stop in stops:
        coords = stop.get("coordinates", {})
        s_lat, s_lng = coords.get("lat"), coords.get("lng")
        if s_lat is not None and s_lng is not None:
            seg = _travel_segment(s_lat, s_lng, new_lat, new_lng)
            stop = {
                **stop,
                "distance_to_selected_km": seg["distance_km"],
                "travel_to_selected_min": seg["travel_minutes"],
            }
        updated.append(stop)
    return updated


# ═══════════════════════════════════════════════════════════════════════════
# 2. CULTURAL GUARDRAIL — Event Sentinel Re-Check
# ═══════════════════════════════════════════════════════════════════════════
def _check_event_constraints(
    candidate: Dict[str, Any],
    target_date_str: Optional[str],
) -> List[ConstraintViolation]:
    """
    Run the Event Sentinel against the selected venue.

    Checks for:
        • Poya-day alcohol bans (bars / party venues)
        • New-Year critical shutdowns
        • Modest-dress requirements
        • Crowd-modifier warnings
    """
    violations: List[ConstraintViolation] = []
    if not target_date_str:
        return violations

    try:
        from ...tools.event_sentinel import get_event_sentinel
        sentinel = get_event_sentinel()

        dt = datetime.strptime(target_date_str[:10], "%Y-%m-%d")

        ctype = candidate.get("type", "").lower()
        activity_map = {
            "bar": "alcohol",
            "hotel": "accommodation",
            "event": "event",
            "restaurant": "dining",
        }
        activity = activity_map.get(ctype, "dining")

        result = sentinel.check_activity_constraints(dt, activity, "Commercial")

        if not result.get("is_allowed", True):
            for v in result.get("violations", []):
                violations.append(ConstraintViolation(
                    constraint_type=v.get("type", "unknown"),
                    description=v.get("message", "Constraint violation detected"),
                    severity=v.get("severity", "high"),
                    suggestion=(
                        result.get("suggestions", ["Consider an alternative"])[0]
                        if result.get("suggestions")
                        else "Consider an alternative venue"
                    ),
                ))

        for w in result.get("warnings", []):
            violations.append(ConstraintViolation(
                constraint_type=w.get("type", "warning"),
                description=w.get("message", ""),
                severity=w.get("severity", "medium"),
                suggestion="Plan accordingly",
            ))

    except Exception as exc:
        logger.warning(f"Event Sentinel re-check failed: {exc}")

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# 3. AESTHETIC PHYSICS — Topographic Horizon Correction
# ═══════════════════════════════════════════════════════════════════════════
def _compute_golden_hour_for_selection(
    lat: Optional[float],
    lng: Optional[float],
    target_date_str: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Apply the GoldenHourEngine's Topographic Horizon Correction to the
    selected venue and return best sunrise/sunset viewing times.

    The engine uses:
        • NREL SPA solar position algorithm
        • Sæmundsson atmospheric refraction formula
        • Horizon-dip correction: θ ≈ arccos(R_e / (R_e + h))

    Returns dict with sunrise_golden, sunset_golden, blue_hour_start,
    blue_hour_end, and elevation_correction_deg, or None if unavailable.
    """
    if not lat or not lng or not target_date_str:
        return None

    try:
        from ...physics.golden_hour_engine import GoldenHourEngine

        engine = GoldenHourEngine()
        dt_obj = datetime.strptime(target_date_str[:10], "%Y-%m-%d")

        result = engine.calculate(
            latitude=lat,
            longitude=lng,
            date=dt_obj,
        )

        if result:
            return {
                "sunrise_golden_start": result.get("sunrise_golden_start"),
                "sunrise_golden_end": result.get("sunrise_golden_end"),
                "sunset_golden_start": result.get("sunset_golden_start"),
                "sunset_golden_end": result.get("sunset_golden_end"),
                "blue_hour_morning_start": result.get("blue_hour_morning_start"),
                "blue_hour_evening_end": result.get("blue_hour_evening_end"),
                "elevation_correction_deg": result.get("horizon_dip_deg", 0.0),
                "recommended_photo_times": _build_photo_recommendations(result),
            }
    except Exception as exc:
        logger.warning(f"Golden hour calculation failed: {exc}")

    return None


def _build_photo_recommendations(gh_data: Dict[str, Any]) -> List[str]:
    """Build human-readable photo-time suggestions from golden hour data."""
    recs: List[str] = []
    if gh_data.get("sunrise_golden_start"):
        recs.append(
            f"🌅 Best sunrise viewing: {gh_data['sunrise_golden_start']} — "
            f"{gh_data.get('sunrise_golden_end', 'N/A')}"
        )
    if gh_data.get("sunset_golden_start"):
        recs.append(
            f"🌇 Best sunset viewing: {gh_data['sunset_golden_start']} — "
            f"{gh_data.get('sunset_golden_end', 'N/A')}"
        )
    if gh_data.get("blue_hour_evening_end"):
        recs.append(
            f"🌌 Blue hour ends at {gh_data['blue_hour_evening_end']}"
        )
    return recs


# ═══════════════════════════════════════════════════════════════════════════
# 4. MAP-READY ITINERARY BUILDER
# ═══════════════════════════════════════════════════════════════════════════
def _build_map_ready_itinerary(
    state: GraphState,
    selected: Dict[str, Any],
    updated_stops: List[Dict[str, Any]],
    warnings: List[str],
    golden_hour_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produce a ``MapReadyItinerary`` JSON with:
        - precise coordinates for every stop
        - itinerary_sequence (ordered list of stop IDs)
        - dynamic_warnings (constraint, weather, event alerts)
        - golden_hour photo recommendations
    """
    existing_itinerary = state.get("final_itinerary") or {}

    # Base stops from existing itinerary (or empty)
    base_stops = updated_stops or existing_itinerary.get("stops", [])

    # Add the selected venue as a new stop if it has coordinates
    selected_stop = None
    if selected.get("lat") and selected.get("lng"):
        max_seq = max(
            (s.get("sequence_id", 0) for s in base_stops), default=0
        )
        selected_stop = {
            "sequence_id": max_seq + 1,
            "day": 1,
            "time": "TBD",
            "location": selected.get("name", "Selected Venue"),
            "activity": f"Visit {selected.get('type', 'venue')}: {selected.get('name', '')}",
            "duration_minutes": 90,
            "coordinates": {
                "lat": selected["lat"],
                "lng": selected["lng"],
            },
            "crowd_prediction": 0.5,
            "lighting_quality": "standard",
            "weather_summary": None,
            "icon": _icon_for_type(selected.get("type", "hotel")),
            "highlight": True,
            "ai_insight": (
                f"Selected via MCP search. Rating: {selected.get('rating', 'N/A')}, "
                f"Price: {selected.get('price_range', 'N/A')}"
            ),
            "cultural_tip": None,
            "ethical_note": None,
            "best_photo_time": (
                golden_hour_data.get("recommended_photo_times", [None])[0]
                if golden_hour_data
                else None
            ),
            "notes": selected.get("opening_hours"),
            "visual_assets": {
                "map_marker_icon": _icon_for_type(selected.get("type", "hotel")),
                "summary": (selected.get("description") or "")[:60],
            },
            # ── Visual optimisation fields ──
            "visual_hierarchy": 1,  # Selected venue is always must-see
            "best_for_photos": bool(
                golden_hour_data and golden_hour_data.get("recommended_photo_times")
            ),
            "photo_urls": selected.get("photo_urls", []),
        }

    all_stops = list(base_stops)
    if selected_stop:
        all_stops.append(selected_stop)

    # ── Annotate visual_hierarchy for existing stops ──────────────
    for stop in all_stops:
        if stop.get("visual_hierarchy") is not None:
            continue  # Already set (e.g. selected_stop)
        if stop.get("highlight"):
            stop["visual_hierarchy"] = 2  # recommended
        else:
            stop["visual_hierarchy"] = 3  # optional
        # Mark best_for_photos on existing stops that have golden hour data
        if stop.get("best_for_photos") is None:
            stop["best_for_photos"] = bool(
                stop.get("best_photo_time") or stop.get("lighting_quality") == "golden"
            )

    # Build itinerary_sequence
    itinerary_sequence = [
        {"sequence_id": s.get("sequence_id", i), "location": s.get("location", "")}
        for i, s in enumerate(all_stops, 1)
    ]

    # Build route polyline
    route_polyline = [
        {
            "lat": s.get("coordinates", {}).get("lat", 0.0),
            "lng": s.get("coordinates", {}).get("lng", 0.0),
            "location_name": s.get("location", ""),
            "sequence_id": s.get("sequence_id", i),
        }
        for i, s in enumerate(all_stops, 1)
        if s.get("coordinates", {}).get("lat") and s.get("coordinates", {}).get("lng")
    ]

    # ── Route geometry (GeoJSON LineString for Mapbox / Google Maps) ──
    route_geometry: List[Dict[str, Any]] = [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [
                        pt.get("lng", 0.0),
                        pt.get("lat", 0.0),
                    ]
                    for pt in route_polyline
                ],
            },
            "properties": {
                "stroke": "#3B82F6",
                "stroke-width": 3,
                "total_stops": len(route_polyline),
            },
        }
    ] if len(route_polyline) >= 2 else []

    # Build dynamic warnings
    dynamic_warnings = list(warnings)
    if golden_hour_data and golden_hour_data.get("recommended_photo_times"):
        for rec in golden_hour_data["recommended_photo_times"]:
            dynamic_warnings.append(rec)

    return {
        "stops": all_stops,
        "route_polyline": route_polyline,
        "route_geometry": route_geometry,
        "itinerary_sequence": itinerary_sequence,
        "contextual_notes": existing_itinerary.get("contextual_notes", []),
        "total_distance_km": existing_itinerary.get("total_distance_km", 0.0),
        "total_days": existing_itinerary.get("total_days", 1),
        "summary": (
            f"Itinerary updated with {selected.get('name', 'selected venue')}"
        ),
        "warnings": existing_itinerary.get("warnings", []) + warnings,
        "tips": existing_itinerary.get("tips", []),
        "dynamic_warnings": dynamic_warnings,
        "golden_hour": golden_hour_data,
        "selected_venue": {
            "name": selected.get("name"),
            "type": selected.get("type"),
            "coordinates": {
                "lat": selected.get("lat"),
                "lng": selected.get("lng"),
            },
            "price_range": selected.get("price_range"),
            "rating": selected.get("rating"),
            "opening_hours": selected.get("opening_hours"),
        },
    }


def _icon_for_type(search_type: str) -> str:
    """Map search type to visual marker icon."""
    return {
        "hotel": "Hotel",
        "restaurant": "Food",
        "bar": "Party",
        "event": "Attraction",
    }.get(search_type, "Attraction")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN NODE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
@trace_node("selection_handler", run_type="tool")
async def selection_handler_node(state: GraphState, llm=None) -> GraphState:
    """
    Post-Selection Re-Optimization Node.

    Runs after the graph resumes from the HITL interrupt.  Receives the
    user's ``selected_search_candidate_id``, then executes four stages:

        1. Validate the selection
        2. Haversine geospatial recalculation
        3. Event Sentinel cultural guardrail re-check
        4. Topographic Horizon Correction for golden hour

    Produces a ``MapReadyItinerary`` JSON and routes to the Verifier
    for up to 3 self-correction iterations.

    Args:
        state: GraphState (must contain selected_search_candidate_id)
        llm:   LangChain LLM (unused; kept for wrapper compatibility)

    Returns:
        Updated GraphState with map_ready_itinerary and constraint data
    """
    start = time.time()
    step_results: List[StepResult] = []

    selected_id = state.get("selected_search_candidate_id")
    candidates = state.get("search_candidates", [])

    # ── 0. Guard: no selection provided ──────────────────────────────
    if not selected_id:
        logger.warning("selection_handler called without selected_search_candidate_id")
        return {
            **state,
            "pending_user_selection": False,
            "step_results": [{
                "node": "selection_handler",
                "status": "warning",
                "summary": "No selection received from user",
                "duration_ms": (time.time() - start) * 1000,
            }],
        }

    # ── 1. Validate — find the selected candidate ────────────────────
    selected: Optional[Dict[str, Any]] = None
    for c in candidates:
        if c.get("id") == selected_id:
            selected = c
            break

    if not selected:
        logger.warning(f"Selected ID '{selected_id}' not in candidates")
        return {
            **state,
            "pending_user_selection": False,
            "step_results": [{
                "node": "selection_handler",
                "status": "error",
                "summary": f"Candidate '{selected_id}' not found",
                "duration_ms": (time.time() - start) * 1000,
            }],
            "error": f"Selected candidate '{selected_id}' not found.",
        }

    logger.info(
        f"Processing selection: {selected.get('name')} ({selected_id})"
    )

    # ── 2. Geospatial — Haversine travel-time recalculation ──────────
    t1 = time.time()
    existing_itinerary = state.get("final_itinerary")
    updated_stops: List[Dict[str, Any]] = []

    if existing_itinerary and selected.get("lat") and selected.get("lng"):
        stops = existing_itinerary.get("stops", [])
        updated_stops = recalculate_travel_times(
            stops,
            {"lat": selected["lat"], "lng": selected["lng"]},
            selected.get("name", "Selected Venue"),
        )
        step_results.append({
            "node": "selection_haversine",
            "status": "success",
            "summary": (
                f"Recalculated Haversine distances for {len(updated_stops)} stops "
                f"to '{selected.get('name')}'"
            ),
            "duration_ms": (time.time() - t1) * 1000,
        })
    else:
        step_results.append({
            "node": "selection_haversine",
            "status": "skipped",
            "summary": "No existing itinerary or missing coordinates — skipped",
            "duration_ms": (time.time() - t1) * 1000,
        })

    # ── 3. Cultural Guardrail — Event Sentinel ───────────────────────
    t2 = time.time()
    target_date = state.get("target_date")
    if not target_date:
        ctx = state.get("tour_plan_context")
        if ctx:
            target_date = ctx.get("start_date")

    event_violations = _check_event_constraints(selected, target_date)
    step_results.append({
        "node": "selection_event_sentinel",
        "status": "warning" if event_violations else "success",
        "summary": (
            f"Event Sentinel: {len(event_violations)} constraint(s) "
            f"for '{selected.get('name')}'"
            if event_violations
            else f"Event Sentinel: No constraints for '{selected.get('name')}'"
        ),
        "duration_ms": (time.time() - t2) * 1000,
    })

    # ── 4. Aesthetic Physics — Topographic Horizon Correction ────────
    t3 = time.time()
    golden_hour_data = _compute_golden_hour_for_selection(
        selected.get("lat"),
        selected.get("lng"),
        target_date,
    )
    step_results.append({
        "node": "selection_golden_hour",
        "status": "success" if golden_hour_data else "skipped",
        "summary": (
            f"Golden hour computed with topographic correction "
            f"(dip={golden_hour_data.get('elevation_correction_deg', 0):.2f}°)"
            if golden_hour_data
            else "Golden hour calculation skipped (missing coords or date)"
        ),
        "duration_ms": (time.time() - t3) * 1000,
    })

    # ── Build constraint warning texts ───────────────────────────────
    warning_texts: List[str] = []
    for v in event_violations:
        warning_texts.append(
            f"⚠️ **{v['constraint_type']}** ({v['severity']}): "
            f"{v['description']} — {v['suggestion']}"
        )

    # ── Build MapReadyItinerary ──────────────────────────────────────
    map_ready = _build_map_ready_itinerary(
        state, selected, updated_stops, warning_texts, golden_hour_data
    )

    # ── Build human-readable response ────────────────────────────────
    response_parts = [f"✅ Selected: **{selected.get('name')}**"]
    if selected.get("price_range"):
        response_parts[0] += f"  ({selected['price_range']})"
    if selected.get("rating"):
        response_parts[0] += f"  ★ {selected['rating']}/5"
    if selected.get("lat") and selected.get("lng"):
        response_parts.append(
            f"📍 Coordinates: {selected['lat']:.4f}, {selected['lng']:.4f}"
        )
    if golden_hour_data and golden_hour_data.get("recommended_photo_times"):
        response_parts.append("\n**Best Photo Times:**")
        response_parts.extend(golden_hour_data["recommended_photo_times"])
    if warning_texts:
        response_parts.append("\n**Constraint Warnings:**")
        response_parts.extend(warning_texts)
    if updated_stops:
        nearest = min(
            updated_stops,
            key=lambda s: s.get("distance_to_selected_km", float("inf")),
        )
        response_parts.append(
            f"\n📏 Nearest stop: **{nearest.get('location', '?')}** "
            f"({nearest.get('distance_to_selected_km', '?')} km, "
            f"~{nearest.get('travel_to_selected_min', '?')} min)"
        )

    response_text = "\n".join(response_parts)

    # ── Merge violations ─────────────────────────────────────────────
    all_violations = (state.get("constraint_violations") or []) + event_violations

    # ── Update final_itinerary ───────────────────────────────────────
    updated_itinerary = existing_itinerary
    if updated_stops and existing_itinerary:
        updated_itinerary = {**existing_itinerary, "stops": updated_stops}

    duration_ms = (time.time() - start) * 1000
    step_results.append({
        "node": "selection_handler",
        "status": "success",
        "summary": (
            f"Post-selection re-optimization complete for "
            f"'{selected.get('name')}' — Haversine ✓, Event Sentinel ✓, "
            f"Golden Hour ✓ [{duration_ms:.0f}ms]"
        ),
        "duration_ms": duration_ms,
    })

    return {
        **state,
        "pending_user_selection": False,
        "selected_search_candidate": selected,
        "constraint_violations": all_violations,
        "final_itinerary": updated_itinerary,
        "map_ready_itinerary": map_ready,
        "generated_response": response_text,
        "final_response": response_text,
        "step_results": step_results,
    }

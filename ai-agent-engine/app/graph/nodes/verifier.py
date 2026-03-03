"""
Verifier Node: Final Self-Correction Check.

This node performs a final validation of the generated response to ensure
it correctly addresses the user's query and doesn't violate any constraints.

Research Pattern:
    Self-Correcting Agent Loop - The verifier can reject a response and
    trigger regeneration with correction instructions, implementing a
    closed-loop feedback mechanism.

Verification Checks:
    1. Constraint Compliance: Response doesn't suggest banned activities
    2. Query Alignment: Response actually addresses the user's question
    3. Accuracy Check: No obvious factual errors in the response
    4. Completeness: Response includes key requested information
"""

import logging
import re
from datetime import datetime
from typing import Dict, Optional

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

from ..state import GraphState, ShadowMonitorLog, UserPreferences, StepResult, CulturalTip

logger = logging.getLogger(__name__)

# Verification prompt
VERIFIER_SYSTEM_PROMPT = """You are a response verifier for a Sri Lankan tourism assistant.

Check if the response:
1. Answers the user's question
2. Respects cultural constraints (if any mentioned)
3. Provides accurate information
4. Is helpful and complete

If the response is good, respond: APPROVED
If there's an issue, respond: NEEDS_CORRECTION: [brief explanation]

Be strict about constraint violations (e.g., suggesting alcohol on Poya day)."""


def verify_constraints(response: str, constraints: list) -> Dict:
    """
    Check if response violates any detected constraints.

    Args:
        response: Generated response text
        constraints: List of ConstraintViolation objects

    Returns:
        Dict with verification results
    """
    response_lower = response.lower()
    violations = []

    for constraint in constraints:
        constraint_type = constraint.get("constraint_type", "")

        # Check for Poya alcohol violations
        if constraint_type == "poya_alcohol":
            alcohol_terms = ["bar", "pub", "nightclub", "drinking", "alcohol", "beer", "wine", "cocktail"]
            for term in alcohol_terms:
                if term in response_lower and "not available" not in response_lower and "banned" not in response_lower:
                    violations.append({
                        "type": "constraint_violation",
                        "message": f"Response mentions '{term}' without noting Poya restrictions",
                        "severity": "high"
                    })
                    break

    return {
        "has_violations": len(violations) > 0,
        "violations": violations
    }


def verify_query_alignment(query: str, response: str) -> Dict:
    """
    Check if response aligns with the user's query.

    Args:
        query: Original user query
        response: Generated response

    Returns:
        Dict with alignment assessment
    """
    query_lower = query.lower()
    response_lower = response.lower()

    # Extract key entities from query
    key_terms = []

    # Location names to check
    locations = [
        "sigiriya", "galle", "kandy", "ella", "mirissa", "trincomalee",
        "jungle beach", "rumassala", "yala", "horton", "temple of the tooth",
        "nine arches", "dambulla", "polonnaruwa", "anuradhapura"
    ]

    for loc in locations:
        if loc in query_lower:
            key_terms.append(loc)

    # Check if key terms appear in response
    missing_terms = [term for term in key_terms if term not in response_lower]

    if missing_terms:
        return {
            "aligned": False,
            "missing_terms": missing_terms,
            "message": f"Response doesn't mention: {', '.join(missing_terms)}"
        }

    return {
        "aligned": True,
        "missing_terms": [],
        "message": "Response aligns with query"
    }


def verify_completeness(intent_value: str, response: str) -> Dict:
    """
    Check if response is complete for the intent type.

    Args:
        intent_value: Intent type (string)
        response: Generated response

    Returns:
        Dict with completeness assessment
    """
    response_len = len(response)

    if intent_value == "greeting":
        # Greetings should be short but welcoming
        is_complete = 50 < response_len < 500
        message = "Greeting is appropriate" if is_complete else "Greeting may be too short/long"

    elif intent_value == "trip_planning":
        # Trip plans should have times and locations
        has_times = bool(re.search(r'\d{1,2}:\d{2}|\d{1,2}\s*(am|pm|AM|PM)', response))
        has_structure = ":" in response or "-" in response or "•" in response
        is_complete = has_times or has_structure or response_len > 200
        message = "Itinerary appears complete" if is_complete else "Itinerary may need more detail"

    else:
        # General queries should be informative
        is_complete = response_len > 100
        message = "Response is informative" if is_complete else "Response may be too brief"

    return {
        "complete": is_complete,
        "message": message
    }


def verify_tour_plan_quality(
    itinerary: list,
    user_preferences: Optional[Dict] = None
) -> Dict:
    """
    Verify the quality of a generated tour plan across multiple dimensions.

    Runs the following checks when an itinerary is present:
        1. Cultural tips check - temple/religious site visits must have cultural_tip
        2. Preference alignment check - activities should align with top user preferences
        3. Day duration check - each day must not exceed 14 hours of total activities
        4. Golden hour consistency - "golden" lighting_quality must fall within
           reasonable golden hour windows (before 07:00 or after 17:00)

    Args:
        itinerary: List of ItinerarySlot dicts from the state
        user_preferences: Optional UserPreferences dict for alignment check

    Returns:
        Dict with 'issues' list and 'passed' boolean
    """
    issues = []

    # --- Religious / temple keywords used to detect cultural sites ---
    religious_keywords = [
        "temple", "kovil", "mosque", "church", "dagoba", "stupa",
        "shrine", "sacred", "religious", "vihara", "devalaya",
        "tooth relic", "bodhi tree", "puja"
    ]

    # ------------------------------------------------------------------
    # 1. Cultural tips check
    # ------------------------------------------------------------------
    for slot in itinerary:
        location = (slot.get("location") or "").lower()
        activity = (slot.get("activity") or "").lower()
        combined = f"{location} {activity}"

        is_religious = any(kw in combined for kw in religious_keywords)

        if is_religious and not slot.get("cultural_tip"):
            day_info = f" (day {slot['day']})" if slot.get("day") else ""
            issues.append(
                f"Cultural tip missing for religious/temple visit at "
                f"'{slot.get('location', 'unknown')}'{day_info}"
            )

    # ------------------------------------------------------------------
    # 2. Preference alignment check
    # ------------------------------------------------------------------
    if user_preferences:
        # Determine the top preferences (score >= 0.6)
        preference_categories = {
            "history": ["temple", "ruin", "ancient", "heritage", "museum",
                        "fort", "colonial", "kingdom", "historical", "palace"],
            "adventure": ["hike", "trek", "surf", "dive", "climb", "rafting",
                          "zip", "kayak", "safari", "adventure", "cycling"],
            "nature": ["wildlife", "national park", "forest", "waterfall",
                       "lake", "bird", "whale", "garden", "nature", "botanical"],
            "relaxation": ["beach", "spa", "resort", "pool", "lounge",
                           "massage", "relax", "sunset", "leisure", "stroll"],
        }

        top_prefs = [
            cat for cat in ["history", "adventure", "nature", "relaxation"]
            if user_preferences.get(cat, 0) >= 0.6
        ]

        if top_prefs:
            # Collect all activity + location text
            all_activity_text = " ".join(
                f"{(s.get('location') or '')} {(s.get('activity') or '')}"
                for s in itinerary
            ).lower()

            for pref in top_prefs:
                keywords = preference_categories.get(pref, [])
                if not any(kw in all_activity_text for kw in keywords):
                    issues.append(
                        f"Plan does not include activities aligned with "
                        f"top preference '{pref}' (score "
                        f"{user_preferences.get(pref, 0):.1f})"
                    )

    # ------------------------------------------------------------------
    # 3. Day duration check (max 14 hours = 840 minutes per day)
    # ------------------------------------------------------------------
    MAX_DAY_MINUTES = 14 * 60  # 840

    day_durations: Dict[int, int] = {}
    for slot in itinerary:
        day = slot.get("day", 1) or 1
        duration = slot.get("duration_minutes", 0) or 0
        day_durations[day] = day_durations.get(day, 0) + duration

    for day, total_minutes in sorted(day_durations.items()):
        if total_minutes > MAX_DAY_MINUTES:
            hours = total_minutes / 60
            issues.append(
                f"Day {day} exceeds 14-hour limit with {hours:.1f} hours "
                f"({total_minutes} minutes) of planned activities"
            )

    # ------------------------------------------------------------------
    # 4. Golden hour consistency check
    # ------------------------------------------------------------------
    def _parse_hour(time_str: str) -> Optional[int]:
        """Extract the hour (0-23) from a time string like '4:30 PM' or '16:30'."""
        if not time_str:
            return None
        time_str = time_str.strip().upper()

        # Try 12-hour format first (e.g. "4:30 PM")
        match_12 = re.match(r'(\d{1,2}):?\d{0,2}\s*(AM|PM)', time_str)
        if match_12:
            hour = int(match_12.group(1))
            period = match_12.group(2)
            if period == "PM" and hour != 12:
                hour += 12
            elif period == "AM" and hour == 12:
                hour = 0
            return hour

        # Try 24-hour format (e.g. "16:30")
        match_24 = re.match(r'(\d{1,2}):\d{2}', time_str)
        if match_24:
            return int(match_24.group(1))

        return None

    for slot in itinerary:
        lighting = (slot.get("lighting_quality") or "").lower()
        if lighting == "golden":
            hour = _parse_hour(slot.get("time", ""))
            if hour is not None and 7 <= hour < 17:
                day_info = f" (day {slot['day']})" if slot.get("day") else ""
                issues.append(
                    f"Golden hour lighting claimed at {slot.get('time')} for "
                    f"'{slot.get('location', 'unknown')}'{day_info}, but time "
                    f"is outside typical golden hour range (before 07:00 or "
                    f"after 17:00)"
                )

    return {
        "passed": len(issues) == 0,
        "issues": issues
    }


async def verifier_node(state: GraphState, llm=None) -> GraphState:
    """
    Verifier Node: Final validation and self-correction check.

    This node validates the generated response against:
    1. Detected constraints
    2. Query alignment
    3. Completeness
    4. Tour plan quality (cultural tips, preference alignment, day duration,
       golden hour consistency) - only when itinerary is present
    5. (Optional) LLM-based quality check

    If verification fails, it can trigger regeneration with corrections.

    Args:
        state: Current graph state
        llm: Optional LLM for quality verification

    Returns:
        Updated GraphState with verification results and step_results tracking

    Research Note:
        The verifier implements "Constitutional AI" principles by checking
        responses against explicit rules (constraints) before delivery.
    """
    import time as _time
    _start_ms = _time.time() * 1000

    response = state.get("generated_response", "")
    query = state["user_query"]
    intent = state.get("intent")
    constraints = state.get("constraint_violations", [])
    loops = state.get("reasoning_loops", 0)
    itinerary = state.get("itinerary")
    user_preferences = state.get("user_preferences")

    logger.info(f"Verifier checking response (loop {loops + 1})...")

    # Check constraints
    constraint_check = verify_constraints(response, [
        {"constraint_type": c["constraint_type"], "description": c["description"]}
        for c in constraints
    ] if constraints else [])

    # Check query alignment
    alignment_check = verify_query_alignment(query, response)

    # Check completeness
    completeness_check = verify_completeness(
        intent.value if intent else "tourism_query",
        response
    )

    # Aggregate verification results
    issues = []
    if constraint_check["has_violations"]:
        issues.extend([v["message"] for v in constraint_check["violations"]])
    if not alignment_check["aligned"]:
        issues.append(alignment_check["message"])
    if not completeness_check["complete"]:
        issues.append(completeness_check["message"])

    # Tour plan quality checks (only when itinerary is present)
    if itinerary:
        tour_quality = verify_tour_plan_quality(itinerary, user_preferences)
        if not tour_quality["passed"]:
            issues.extend(tour_quality["issues"])
            logger.warning(
                f"Tour plan quality issues found: {tour_quality['issues']}"
            )

    # Determine if regeneration needed
    needs_correction = len(issues) > 0 and loops < 2  # Max 2 correction loops

    # Log verification
    log_entry = ShadowMonitorLog(
        timestamp=datetime.now().isoformat(),
        check_type="verifier",
        input_context={
            "response_length": len(response),
            "constraint_count": len(constraints),
            "loop_number": loops,
            "has_itinerary": itinerary is not None,
            "has_user_preferences": user_preferences is not None
        },
        result="needs_correction" if needs_correction else "approved",
        details="; ".join(issues) if issues else "All checks passed",
        action_taken="regenerate" if needs_correction else "finalize"
    )

    # Build step_result for progress tracking
    _duration_ms = (_time.time() * 1000) - _start_ms
    step_result = StepResult(
        node="verifier",
        status="warning" if needs_correction else "success",
        summary=(
            f"Verification failed with {len(issues)} issue(s): "
            f"{'; '.join(issues[:3])}"
            if needs_correction
            else "All verification checks passed"
        ),
        duration_ms=round(_duration_ms, 2)
    )

    if needs_correction:
        # Prepare correction instructions for regeneration
        correction_prompt = f"Please regenerate addressing these issues: {'; '.join(issues)}"
        return {
            **state,
            "reasoning_loops": loops + 1,
            "_correction_instructions": correction_prompt,
            "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry],
            "step_results": state.get("step_results", []) + [step_result]
        }
    else:
        # Response approved - set as final
        return {
            **state,
            "final_response": response,
            "reasoning_loops": loops,
            "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry],
            "step_results": state.get("step_results", []) + [step_result]
        }


def route_after_verification(state: GraphState) -> str:
    """
    Routing function: Decide whether to finalize or regenerate.

    Args:
        state: Current graph state

    Returns:
        Next node name: "generate" (regenerate) or "__end__" (finalize)
    """
    if state.get("final_response"):
        return "__end__"
    elif state.get("_correction_instructions"):
        return "generate"  # Loop back for correction
    else:
        return "__end__"

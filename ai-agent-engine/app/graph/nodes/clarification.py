"""
Clarification Node: Interactive Agent Questioning for Tour Plan Generation.

This node checks if critical information is missing or ambiguous before
generating a tour plan. When it detects gaps, it creates a structured
question with selectable options that the mobile app displays to the user.

Research Pattern:
    Human-in-the-Loop with Structured Questioning - Instead of generating
    suboptimal plans due to missing information, the agent proactively asks
    the user for clarification, similar to how a real tour guide would ask
    questions before planning.

Triggers:
    - Missing or ambiguous dates
    - Too many locations for the date range (>3 per day)
    - Too few locations for the date range (<1 per day)
    - No user preferences available
    - Conflicting activity types
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..state import GraphState, ClarificationQuestion

logger = logging.getLogger(__name__)


def _calculate_days(start_date: str, end_date: str) -> int:
    """Calculate number of trip days."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return max((end - start).days + 1, 1)
    except (ValueError, TypeError):
        return 0


def _check_date_issues(tour_context: Dict[str, Any]) -> Optional[ClarificationQuestion]:
    """Check for date-related issues."""
    start_date = tour_context.get("start_date")
    end_date = tour_context.get("end_date")

    if not start_date or not end_date:
        return ClarificationQuestion(
            question="When are you planning to visit Sri Lanka?",
            options=[
                {"label": "This week", "description": "Plan for the upcoming 3-4 days", "recommended": False},
                {"label": "Next week", "description": "Plan for next week's schedule", "recommended": True},
                {"label": "Next month", "description": "Plan for a trip next month", "recommended": False},
                {"label": "Flexible dates", "description": "I'm flexible — suggest the best dates", "recommended": False},
            ],
            context="I need your travel dates to calculate precise golden hour times and crowd predictions for each location.",
            type="single_select",
        )

    # Check if dates are in the past
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        if start.date() < datetime.now().date():
            return ClarificationQuestion(
                question="Your start date appears to be in the past. Would you like to update it?",
                options=[
                    {"label": "Start tomorrow", "description": "Begin the trip from tomorrow", "recommended": True},
                    {"label": "Keep dates", "description": "These dates are correct (planning for reference)", "recommended": False},
                ],
                context="Planning with future dates lets me provide accurate crowd predictions and weather data.",
                type="single_select",
            )
    except ValueError:
        pass

    return None


def _check_location_day_balance(tour_context: Dict[str, Any]) -> Optional[ClarificationQuestion]:
    """Check if the number of locations is reasonable for the date range."""
    locations = tour_context.get("selected_locations", [])
    start_date = tour_context.get("start_date", "")
    end_date = tour_context.get("end_date", "")

    num_locations = len(locations)
    num_days = _calculate_days(start_date, end_date)

    if num_days == 0 or num_locations == 0:
        return None

    locations_per_day = num_locations / num_days

    if locations_per_day > 3:
        location_names = ", ".join(loc.get("name", "?") for loc in locations[:5])
        return ClarificationQuestion(
            question=f"You have {num_locations} locations for {num_days} day(s). That's quite packed! How would you like to proceed?",
            options=[
                {"label": "Highlights only", "description": f"I'll pick the top {num_days * 2} must-see locations and create a relaxed itinerary", "recommended": True},
                {"label": "Quick visits", "description": "Keep all locations but with shorter visits (30-60 min each)", "recommended": False},
                {"label": "Keep all", "description": "I want to see everything, even if it means a busy schedule", "recommended": False},
            ],
            context=f"A comfortable pace is 1-2 major locations per day. You selected: {location_names}{'...' if num_locations > 5 else ''}",
            type="single_select",
        )

    return None


def _check_preference_availability(
    user_preferences: Optional[Dict[str, Any]],
    locations: List[Dict[str, Any]]
) -> Optional[ClarificationQuestion]:
    """Check if user preferences are available for personalization."""
    if user_preferences:
        # Check if all scores are at default (0.5) — suggests no real preferences set
        scores = [
            user_preferences.get("history", 0.5),
            user_preferences.get("adventure", 0.5),
            user_preferences.get("nature", 0.5),
            user_preferences.get("relaxation", 0.5),
        ]
        if all(s == 0.5 for s in scores):
            return ClarificationQuestion(
                question="What type of experience are you looking for on this trip?",
                options=[
                    {"label": "Culture & History", "description": "Ancient temples, heritage sites, museums, local traditions", "recommended": False},
                    {"label": "Adventure", "description": "Hiking, water sports, wildlife safaris, rock climbing", "recommended": False},
                    {"label": "Nature & Scenic", "description": "Waterfalls, national parks, botanical gardens, scenic trains", "recommended": False},
                    {"label": "Relaxation", "description": "Beaches, spas, gentle walks, sunset watching", "recommended": False},
                ],
                context="This helps me tailor the activities, timing, and pace to match what you enjoy most.",
                type="multi_select",
            )
        return None

    # No preferences at all
    return ClarificationQuestion(
        question="What type of experience are you most interested in?",
        options=[
            {"label": "Culture & History", "description": "Temples, ruins, museums — learn about Sri Lanka's 2500-year history", "recommended": False},
            {"label": "Adventure & Outdoors", "description": "Hiking, safaris, water sports — active exploration", "recommended": False},
            {"label": "Nature & Photography", "description": "Scenic landscapes, wildlife, golden hour photo spots", "recommended": False},
            {"label": "Mix of everything", "description": "A balanced itinerary with variety each day (Recommended)", "recommended": True},
        ],
        context="Knowing your interests lets me schedule the right activities at the best times for you.",
        type="single_select",
    )


async def clarification_node(state: GraphState, llm=None) -> GraphState:
    """
    Clarification Node: Check for missing info and ask the user if needed.

    This node runs BEFORE tour plan generation and checks for:
    1. Missing or invalid dates
    2. Location-to-day imbalance
    3. Missing user preferences

    If clarification is needed, it sets clarification_needed=True and
    returns a structured question. The graph will short-circuit and
    return the question to the user instead of generating a plan.

    Args:
        state: Current graph state
        llm: LangChain LLM instance (not used in this node)

    Returns:
        Updated GraphState with clarification status
    """
    import time
    start_time = time.time()
    logger.info("Clarification node checking for missing information...")

    tour_context = state.get("tour_plan_context")
    if not tour_context:
        # No tour plan context — nothing to clarify
        return {
            **state,
            "clarification_needed": False,
            "clarification_question": None,
            "step_results": [{"node": "clarification", "status": "success", "summary": "No tour plan context — skipping", "duration_ms": 0}],
        }

    # Run checks in priority order
    # Note: Preference check removed — preferences are auto-fetched from user profile
    checks = [
        ("dates", lambda: _check_date_issues(tour_context)),
        ("location_balance", lambda: _check_location_day_balance(tour_context)),
    ]

    for check_name, check_fn in checks:
        question = check_fn()
        if question:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Clarification needed: {check_name} — {question['question']}")
            return {
                **state,
                "clarification_needed": True,
                "clarification_question": question,
                "step_results": [{
                    "node": "clarification",
                    "status": "needs_input",
                    "summary": f"Asking user about {check_name}: {question['question'][:80]}",
                    "duration_ms": duration_ms,
                }],
            }

    # All checks passed — proceed to generation
    duration_ms = (time.time() - start_time) * 1000
    logger.info("All clarification checks passed — ready for plan generation")
    return {
        **state,
        "clarification_needed": False,
        "clarification_question": None,
        "step_results": [{
            "node": "clarification",
            "status": "success",
            "summary": "All information available — proceeding to plan generation",
            "duration_ms": duration_ms,
        }],
    }


def route_after_clarification(state: GraphState) -> str:
    """Route after clarification: return to user or proceed to plan generation."""
    if state.get("clarification_needed"):
        return "end_with_clarification"
    return "tour_plan_generate"

"""
Restaurant Selection Handler Node.

Handles the Human-in-the-Loop (HITL) flow for restaurant selection
during tour plan generation.

Flow:
    1. ``tour_plan_generate`` finds restaurants, builds selection cards,
       sets ``pending_user_selection=True`` and pauses the graph.
    2. The mobile app shows the cards; the user picks one.
    3. ``resume_with_selection()`` injects ``selected_search_candidate_id``.
    4. This node runs: maps the selected ID to a restaurant, stores
       ``selected_restaurant_ids`` in tour_plan_context, and clears the
       pending flags so the graph can loop back to ``tour_plan_generate``
       for the full plan with the chosen restaurant baked in.
"""

import logging
from typing import Any, Dict, List

from ..state import GraphState, RestaurantRecommendation

logger = logging.getLogger(__name__)


async def restaurant_selection_handler_node(
    state: GraphState,
    llm=None,
) -> GraphState:
    """
    Process the user's restaurant selection and prepare state for
    the second pass of tour_plan_generate.

    Args:
        state: Current graph state with selected_search_candidate_id
        llm: LLM instance (unused — kept for wrapper signature parity)

    Returns:
        Updated GraphState ready for tour_plan_generate re-entry
    """
    selected_id = state.get("selected_search_candidate_id")
    restaurant_recs: List[RestaurantRecommendation] = state.get(
        "restaurant_recommendations", []
    )

    logger.info(
        f"Restaurant selection handler — selected_id={selected_id}, "
        f"available_recs={len(restaurant_recs)}"
    )

    # ── User explicitly skipped restaurant selection ──────────────
    if selected_id == "__SKIP__":
        logger.info("User skipped restaurant selection")
        tour_context = dict(state.get("tour_plan_context") or {})
        tour_context["skip_restaurants"] = True
        # Return ONLY the fields being updated — never spread **state
        # into the return dict.  GraphState uses operator.add reducers on
        # messages, shadow_monitor_logs, and step_results; spreading the
        # full state would cause those lists to double on every node.
        return {
            "tour_plan_context": tour_context,
            "pending_user_selection": False,
            "pending_restaurant_selection": False,
            "selection_cards": None,
            "selected_search_candidate_id": None,
            "skip_restaurants": True,
            "restaurant_recommendations": [],
            "step_results": [{
                "node": "restaurant_selection_handler",
                "status": "skipped",
                "summary": "User skipped restaurant selection — generating plan without dining recommendations",
                "duration_ms": 0,
            }],
        }

    # Locate the chosen restaurant
    selected_restaurant = None
    for rec in restaurant_recs:
        if rec.get("id") == selected_id:
            selected_restaurant = rec
            break

    if not selected_restaurant:
        logger.warning(
            f"Selected restaurant ID '{selected_id}' not found in "
            f"{len(restaurant_recs)} recommendations — proceeding without selection"
        )
        # Clear flags and let plan generate without a specific restaurant
        tour_context = dict(state.get("tour_plan_context") or {})
        tour_context["skip_restaurants"] = True
        return {
            "tour_plan_context": tour_context,
            "pending_user_selection": False,
            "pending_restaurant_selection": False,
            "selection_cards": None,
            "selected_search_candidate_id": None,
            "skip_restaurants": True,
            "step_results": [{
                "node": "restaurant_selection_handler",
                "status": "warning",
                "summary": f"Restaurant '{selected_id}' not found — skipping restaurant selection",
                "duration_ms": 0,
            }],
        }

    logger.info(
        f"User selected restaurant: {selected_restaurant.get('name')} "
        f"(day {selected_restaurant.get('day')}, "
        f"{selected_restaurant.get('meal_slot')})"
    )

    # Persist the selection inside tour_plan_context so
    # build_plan_context injects the restaurant into the LLM prompt.
    tour_context = dict(state.get("tour_plan_context") or {})
    tour_context["selected_restaurant_ids"] = [selected_id]
    tour_context["skip_restaurants"] = True  # don't re-search on second pass

    return {
        "tour_plan_context": tour_context,
        "pending_user_selection": False,
        "pending_restaurant_selection": False,
        "selection_cards": None,
        "selected_search_candidate_id": None,
        "selected_restaurant_ids": [selected_id],
        "skip_restaurants": True,
        "step_results": [{
            "node": "restaurant_selection_handler",
            "status": "success",
            "summary": (
                f"User selected '{selected_restaurant.get('name')}' for "
                f"Day {selected_restaurant.get('day')} "
                f"{(selected_restaurant.get('meal_slot') or '').title()}"
            ),
            "duration_ms": 0,
        }],
    }

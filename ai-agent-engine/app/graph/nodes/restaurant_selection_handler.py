"""
Multi-Step HITL Selection Handler Node.

Handles every selection card the user interacts with during the
multi-step tour plan generation flow:

    Step 1 — Dining/Accommodation preference  (card_id prefix: pref_)
    Step 2 — Budget preference                (card_id prefix: budget_)
    Step 3 — Restaurant pick                  (card_id prefix: rest_)
    Step 4 — Accommodation pick               (card_id prefix: hotel_)
    Skip   — ``__SKIP__`` sentinel            (context-sensitive)

After processing the user's choice the node clears the HITL flags and
loops back to ``tour_plan_generate`` for the next step.
"""

import logging
from typing import Any, Dict, List

from ..state import (
    GraphState,
    RestaurantRecommendation,
    AccommodationRecommendation,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────

def _step_result(status: str, summary: str) -> List[Dict[str, Any]]:
    """Return a tiny step_results list for the handler."""
    return [{
        "node": "restaurant_selection_handler",
        "status": status,
        "summary": summary,
        "duration_ms": 0,
    }]


def _clear_hitl_flags() -> Dict[str, Any]:
    """Fields that must be reset after every HITL interaction."""
    return {
        "pending_user_selection": False,
        "pending_restaurant_selection": False,
        "selection_cards": None,
        "selected_search_candidate_id": None,
    }


# ── Main handler ──────────────────────────────────────────────────

async def restaurant_selection_handler_node(
    state: GraphState,
    llm=None,
) -> GraphState:
    """
    Process the user's HITL selection and prepare state for the next
    pass of ``tour_plan_generate``.

    Routing logic is based on the ``selected_search_candidate_id``
    value injected by ``resume_with_selection()``.
    """
    selected_id: str = state.get("selected_search_candidate_id") or ""
    tour_context = dict(state.get("tour_plan_context") or {})

    logger.info(
        f"Selection handler — selected_id={selected_id!r}, "
        f"dining_pref={tour_context.get('dining_preference')!r}, "
        f"budget_pref={tour_context.get('budget_preference')!r}"
    )

    # ── 1. __SKIP__ — context-sensitive skip ─────────────────────
    if selected_id == "__SKIP__":
        return _handle_skip(tour_context)

    # ── 2. Preference cards (pref_dining / pref_accommodation / …)
    if selected_id.startswith("pref_"):
        return _handle_preference(selected_id, tour_context)

    # ── 3. Budget cards (budget_low / budget_medium / budget_high)
    if selected_id.startswith("budget_"):
        return _handle_budget(selected_id, tour_context)

    # ── 4. Restaurant pick (rest_d1_lunch_2 etc.)
    if selected_id.startswith("rest_"):
        return _handle_restaurant(selected_id, tour_context, state)

    # ── 5. Accommodation pick (hotel_d1_2 etc.)
    if selected_id.startswith("hotel_"):
        return _handle_accommodation(selected_id, tour_context, state)

    # ── Fallback — unknown card id ────────────────────────────────
    logger.warning(f"Unknown selection card_id: {selected_id!r} — skipping")
    return {
        **_clear_hitl_flags(),
        "tour_plan_context": tour_context,
        "step_results": _step_result(
            "warning",
            f"Unknown card '{selected_id}' — skipping selection step",
        ),
    }


# ── Individual handlers ───────────────────────────────────────────

def _handle_skip(tour_context: dict) -> Dict[str, Any]:
    """
    Determine what the user is skipping from the current HITL state
    and set sensible defaults so tour_plan_generate can continue.
    """
    dining_pref = tour_context.get("dining_preference")
    budget_pref = tour_context.get("budget_preference")
    skip_rest = tour_context.get("skip_restaurants", False)

    if not dining_pref:
        # Skipping the preference question → treat as "none"
        tour_context["dining_preference"] = "none"
        tour_context["skip_restaurants"] = True
        tour_context["skip_accommodations"] = True
        summary = "User skipped preferences — generating activities-only plan"
    elif not budget_pref:
        # Skipping budget → default to medium
        tour_context["budget_preference"] = "medium"
        summary = "User skipped budget — defaulting to mid-range"
    elif not skip_rest and dining_pref in ("dining", "both"):
        # Skipping restaurant selection
        tour_context["skip_restaurants"] = True
        summary = "User skipped restaurant selection"
    else:
        # Skipping accommodation selection
        tour_context["skip_accommodations"] = True
        summary = "User skipped accommodation selection"

    logger.info(summary)
    return {
        **_clear_hitl_flags(),
        "tour_plan_context": tour_context,
        "skip_restaurants": tour_context.get("skip_restaurants", False),
        "skip_accommodations": tour_context.get("skip_accommodations", False),
        "step_results": _step_result("skipped", summary),
    }


def _handle_preference(
    selected_id: str,
    tour_context: dict,
) -> Dict[str, Any]:
    """Store the dining/accommodation preference."""
    choice = selected_id.replace("pref_", "")  # dining|accommodation|both|none

    tour_context["dining_preference"] = choice
    if choice == "none":
        tour_context["skip_restaurants"] = True
        tour_context["skip_accommodations"] = True

    logger.info(f"User selected dining preference: {choice}")
    return {
        **_clear_hitl_flags(),
        "tour_plan_context": tour_context,
        "skip_restaurants": tour_context.get("skip_restaurants", False),
        "skip_accommodations": tour_context.get("skip_accommodations", False),
        "step_results": _step_result(
            "success",
            f"User chose '{choice}' for dining/accommodation preference",
        ),
    }


def _handle_budget(
    selected_id: str,
    tour_context: dict,
) -> Dict[str, Any]:
    """Store the budget preference."""
    choice = selected_id.replace("budget_", "")  # low|medium|high

    tour_context["budget_preference"] = choice
    logger.info(f"User selected budget: {choice}")
    return {
        **_clear_hitl_flags(),
        "tour_plan_context": tour_context,
        "step_results": _step_result(
            "success",
            f"User chose '{choice}' budget",
        ),
    }


def _handle_restaurant(
    selected_id: str,
    tour_context: dict,
    state: GraphState,
) -> Dict[str, Any]:
    """Find the chosen restaurant in recommendations and store it."""
    restaurant_recs: List[RestaurantRecommendation] = state.get(
        "restaurant_recommendations", []
    )

    selected = None
    for rec in restaurant_recs:
        if rec.get("id") == selected_id:
            selected = rec
            break

    if not selected:
        logger.warning(
            f"Restaurant '{selected_id}' not in {len(restaurant_recs)} recs — skipping"
        )
        tour_context["skip_restaurants"] = True
        return {
            **_clear_hitl_flags(),
            "tour_plan_context": tour_context,
            "skip_restaurants": True,
            "step_results": _step_result(
                "warning",
                f"Restaurant '{selected_id}' not found — skipping",
            ),
        }

    logger.info(
        f"User selected restaurant: {selected.get('name')} "
        f"(day {selected.get('day')}, {selected.get('meal_slot')})"
    )

    tour_context["selected_restaurant_ids"] = [selected_id]
    tour_context["skip_restaurants"] = True

    return {
        **_clear_hitl_flags(),
        "tour_plan_context": tour_context,
        "selected_restaurant_ids": [selected_id],
        "skip_restaurants": True,
        "step_results": _step_result(
            "success",
            (
                f"User selected '{selected.get('name')}' for "
                f"Day {selected.get('day')} "
                f"{(selected.get('meal_slot') or '').title()}"
            ),
        ),
    }


def _handle_accommodation(
    selected_id: str,
    tour_context: dict,
    state: GraphState,
) -> Dict[str, Any]:
    """Find the chosen accommodation in recommendations and store it."""
    accommodation_recs: List[AccommodationRecommendation] = state.get(
        "accommodation_recommendations", []
    )

    selected = None
    for rec in accommodation_recs:
        if rec.get("id") == selected_id:
            selected = rec
            break

    if not selected:
        logger.warning(
            f"Accommodation '{selected_id}' not in "
            f"{len(accommodation_recs)} recs — skipping"
        )
        tour_context["skip_accommodations"] = True
        return {
            **_clear_hitl_flags(),
            "tour_plan_context": tour_context,
            "skip_accommodations": True,
            "step_results": _step_result(
                "warning",
                f"Accommodation '{selected_id}' not found — skipping",
            ),
        }

    logger.info(
        f"User selected accommodation: {selected.get('name')} "
        f"(night {selected.get('check_in_day')}, {selected.get('type')})"
    )

    tour_context["selected_accommodation_ids"] = [selected_id]
    tour_context["skip_accommodations"] = True

    return {
        **_clear_hitl_flags(),
        "tour_plan_context": tour_context,
        "selected_accommodation_ids": [selected_id],
        "skip_accommodations": True,
        "step_results": _step_result(
            "success",
            (
                f"User selected '{selected.get('name')}' "
                f"({selected.get('type', 'hotel')}) for "
                f"Night {selected.get('check_in_day')}"
            ),
        ),
    }

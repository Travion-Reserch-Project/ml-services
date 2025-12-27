"""
Router Node: Intent Classification for Query Routing.

This node analyzes the user's query and classifies it into one of several
intent categories to determine the appropriate processing path.

Research Pattern:
    Semantic Router - Uses LLM classification to route queries through
    different processing pipelines based on intent.

Routing Logic:
    GREETING -> generate (direct response)
    TOURISM_QUERY -> retrieve (knowledge lookup)
    TRIP_PLANNING -> retrieve + shadow_monitor (multi-step planning)
    REAL_TIME_INFO -> web_search (external lookup)
    OFF_TOPIC -> generate (polite redirect)
"""

import logging
import re
from typing import Dict, Literal
from datetime import datetime

from ..state import GraphState, IntentType

logger = logging.getLogger(__name__)

# Intent classification prompts
ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a Sri Lankan tourism chatbot.

Classify the user's message into exactly one of these categories:

1. GREETING - Casual greetings, hellos, how are you, etc.
2. TOURISM_QUERY - Questions about Sri Lankan destinations, attractions, history, culture
3. TRIP_PLANNING - Requests for itineraries, trip plans, schedules, "plan a trip"
4. REAL_TIME_INFO - Questions about current weather, prices, availability, recent events
5. OFF_TOPIC - Questions unrelated to Sri Lankan tourism

Respond with ONLY the category name, nothing else.

Examples:
- "Hello!" -> GREETING
- "What can I see at Sigiriya?" -> TOURISM_QUERY
- "Plan a 3-day trip to Ella" -> TRIP_PLANNING
- "What's the weather like today?" -> REAL_TIME_INFO
- "What's the capital of France?" -> OFF_TOPIC
"""


def extract_entities(query: str) -> Dict:
    """
    Extract key entities from the user query.

    This function identifies:
    - Location names (fuzzy matching against known locations)
    - Dates and temporal expressions
    - Activity preferences

    Args:
        query: User's query string

    Returns:
        Dict with extracted entities
    """
    query_lower = query.lower()

    # Known Sri Lankan locations for matching
    LOCATIONS = {
        "sigiriya": "Sigiriya Lion Rock",
        "galle": "Galle Fort",
        "kandy": "Kandy",
        "ella": "Ella",
        "mirissa": "Mirissa",
        "arugam bay": "Arugam Bay",
        "trincomalee": "Trincomalee",
        "nuwara eliya": "Nuwara Eliya",
        "anuradhapura": "Anuradhapura",
        "polonnaruwa": "Polonnaruwa",
        "yala": "Yala National Park",
        "udawalawe": "Udawalawe National Park",
        "horton plains": "Horton Plains",
        "temple of the tooth": "Temple of the Tooth",
        "nine arches bridge": "Nine Arches Bridge",
        "jungle beach": "Jungle Beach (Rumassala)",
        "rumassala": "Jungle Beach (Rumassala)",
        "unawatuna": "Unawatuna Beach",
        "hikkaduwa": "Hikkaduwa",
        "bentota": "Bentota",
        "negombo": "Negombo",
        "jaffna": "Jaffna",
        "dambulla": "Dambulla Cave Temple",
        "colombo": "Colombo",
    }

    # Date patterns
    DATE_PATTERNS = [
        (r"next full moon", "next_poya"),
        (r"next poya", "next_poya"),
        (r"vesak", "vesak"),
        (r"poson", "poson"),
        (r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", "specific_date"),
        (r"tomorrow", "tomorrow"),
        (r"next week", "next_week"),
        (r"this weekend", "this_weekend"),
        (r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})?", "month"),
    ]

    # Activity patterns
    ACTIVITY_PATTERNS = {
        "sunrise": ["sunrise", "early morning", "dawn"],
        "sunset": ["sunset", "evening", "golden hour"],
        "photography": ["photo", "photography", "pictures", "camera"],
        "hiking": ["hike", "hiking", "trek", "trekking", "walk"],
        "beach": ["beach", "swimming", "snorkeling", "diving"],
        "wildlife": ["safari", "wildlife", "animals", "elephant", "leopard"],
        "temple": ["temple", "religious", "buddhist", "hindu"],
        "cultural": ["cultural", "heritage", "history", "historical"],
    }

    entities = {
        "locations": [],
        "date_reference": None,
        "activities": [],
        "preferences": {}
    }

    # Extract locations
    for key, value in LOCATIONS.items():
        if key in query_lower:
            entities["locations"].append(value)

    # Extract date references
    for pattern, date_type in DATE_PATTERNS:
        if re.search(pattern, query_lower):
            entities["date_reference"] = date_type
            break

    # Extract activities
    for activity, keywords in ACTIVITY_PATTERNS.items():
        for keyword in keywords:
            if keyword in query_lower:
                entities["activities"].append(activity)
                break

    return entities


def classify_intent_heuristic(query: str) -> IntentType:
    """
    Heuristic-based intent classification (fallback when LLM unavailable).

    Args:
        query: User's query string

    Returns:
        IntentType: Classified intent
    """
    query_lower = query.lower().strip()

    # Greeting patterns
    greeting_patterns = [
        r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)",
        r"^(how are you|what's up|howdy)",
        r"^(thanks|thank you|bye|goodbye)",
    ]
    for pattern in greeting_patterns:
        if re.match(pattern, query_lower):
            return IntentType.GREETING

    # Trip planning patterns
    planning_patterns = [
        r"plan (a |my )?(trip|visit|tour|itinerary)",
        r"(create|make|suggest) (a |an )?(itinerary|schedule|plan)",
        r"(\d+)[\s-]?(day|week) (trip|tour|itinerary)",
        r"what (should|can) i (do|see|visit)",
    ]
    for pattern in planning_patterns:
        if re.search(pattern, query_lower):
            return IntentType.TRIP_PLANNING

    # Real-time info patterns
    realtime_patterns = [
        r"(current|today|now|right now)",
        r"weather (today|now|current|forecast)",
        r"(price|cost|fee|ticket) (\d{4}|today|current)",
        r"(open|closed) (today|now)",
        r"(available|availability)",
    ]
    for pattern in realtime_patterns:
        if re.search(pattern, query_lower):
            return IntentType.REAL_TIME_INFO

    # Off-topic patterns (non-Sri Lanka)
    offtopic_patterns = [
        r"(france|paris|london|new york|tokyo|india|thailand)",
        r"(code|programming|python|javascript)",
        r"(recipe|cooking|food) (?!sri lanka)",
        r"(stock|crypto|bitcoin|investment)",
    ]
    for pattern in offtopic_patterns:
        if re.search(pattern, query_lower):
            return IntentType.OFF_TOPIC

    # Default to tourism query
    return IntentType.TOURISM_QUERY


async def router_node(state: GraphState, llm=None) -> GraphState:
    """
    Router Node: Classify intent and extract entities from user query.

    This is the entry point of the agentic reasoning loop. It:
    1. Classifies the user's intent (using fast heuristics)
    2. Extracts key entities (locations, dates, activities)
    3. Sets the routing path for subsequent nodes

    Args:
        state: Current graph state
        llm: Optional LLM (NOT used for router - heuristics are faster and sufficient)

    Returns:
        Updated GraphState with intent and entities

    Research Note:
        The router uses heuristic classification for speed (instant vs 5+ min LLM).
        Heuristics are sufficient for intent routing - LLM is saved for generation.
    """
    query = state["user_query"]
    logger.info(f"Router processing: {query[:50]}...")

    # Extract entities first (always useful)
    entities = extract_entities(query)

    # Use fast heuristic classification (LLM is too slow for routing)
    # LLM classification took 5+ minutes, heuristics are instant and accurate enough
    intent = classify_intent_heuristic(query)

    logger.info(f"Classified intent: {intent.value} (heuristic)")

    # Update state
    return {
        **state,
        "intent": intent,
        "target_location": entities["locations"][0] if entities["locations"] else None,
        "target_date": entities["date_reference"],
        "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [{
            "timestamp": datetime.now().isoformat(),
            "check_type": "router",
            "input_context": {"query": query},
            "result": "classified",
            "details": f"Intent: {intent.value}, Locations: {entities['locations']}, Date: {entities['date_reference']}",
            "action_taken": None
        }]
    }


def route_by_intent(state: GraphState) -> Literal["generate", "retrieve", "web_search"]:
    """
    Routing function for LangGraph conditional edges.

    This function determines the next node based on classified intent.

    Args:
        state: Current graph state

    Returns:
        Next node name as string literal
    """
    intent = state.get("intent")

    if intent == IntentType.GREETING:
        return "generate"
    elif intent == IntentType.OFF_TOPIC:
        return "generate"
    elif intent == IntentType.REAL_TIME_INFO:
        return "web_search"
    else:
        # TOURISM_QUERY and TRIP_PLANNING go to retrieve
        return "retrieve"

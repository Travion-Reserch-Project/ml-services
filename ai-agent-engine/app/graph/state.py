"""
GraphState: State Management for Agentic RAG System.

This module defines the TypedDict that flows through the LangGraph workflow.
It tracks conversation history, retrieved context, reasoning flags, and
monitoring logs for the self-correcting agent loop.

Research Note:
    The state design follows the "Blackboard Architecture" pattern where
    multiple specialist nodes read from and write to a shared state object,
    enabling complex multi-step reasoning with observable intermediate states.
"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated
from datetime import datetime
from enum import Enum
import operator


class IntentType(str, Enum):
    """
    Classification of user intent for routing decisions.

    GREETING: Casual conversation (hi, hello, how are you)
    TOURISM_QUERY: Questions about Sri Lankan destinations
    TRIP_PLANNING: Request for itinerary generation
    REAL_TIME_INFO: Weather, crowd, current events
    OFF_TOPIC: Queries outside the tourism domain
    """
    GREETING = "greeting"
    TOURISM_QUERY = "tourism_query"
    TRIP_PLANNING = "trip_planning"
    REAL_TIME_INFO = "real_time_info"
    OFF_TOPIC = "off_topic"


class DocumentRelevance(str, Enum):
    """
    Grader assessment of retrieved document quality.

    RELEVANT: Documents directly answer the query
    PARTIAL: Documents contain some useful information
    IRRELEVANT: Documents do not address the query
    INSUFFICIENT: Not enough documents retrieved
    """
    RELEVANT = "relevant"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"
    INSUFFICIENT = "insufficient"


class ConstraintViolation(TypedDict):
    """
    Record of a constraint violation detected by Shadow Monitor.

    Attributes:
        constraint_type: Category (poya_alcohol, crowd_warning, weather_alert)
        description: Human-readable explanation
        severity: low, medium, high, critical
        suggestion: Corrective action recommendation
    """
    constraint_type: str
    description: str
    severity: str
    suggestion: str


class RetrievedDocument(TypedDict):
    """
    Structure for documents fetched from ChromaDB.

    Attributes:
        content: The text content of the document
        metadata: Location name, type, aspect, etc.
        relevance_score: Similarity score from vector search
        source: Origin identifier (chromadb, web_search)
    """
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str


class ShadowMonitorLog(TypedDict):
    """
    Log entry from Shadow Monitor reasoning step.

    Attributes:
        timestamp: When the check was performed
        check_type: event_sentinel, crowdcast, golden_hour, weather
        input_context: What was being evaluated
        result: Pass/Fail/Warning
        details: Specific findings
        action_taken: What the agent decided to do
    """
    timestamp: str
    check_type: str
    input_context: Dict[str, Any]
    result: str
    details: str
    action_taken: Optional[str]


class ItinerarySlot(TypedDict):
    """
    Single time slot in a generated itinerary.

    Attributes:
        time: Suggested time (e.g., "4:30 PM")
        location: Destination name
        activity: What to do there
        duration_minutes: Suggested duration
        crowd_prediction: Expected crowd level (0-100)
        lighting_quality: Golden hour assessment
        notes: Special considerations
        day: Day number in multi-day trip (1-indexed)
        order: Order within the day
        icon: Icon name for UI display
        highlight: Whether this is a highlighted activity
        ai_insight: AI-generated insight for this activity
        cultural_tip: Cultural etiquette tip for the activity/location
        ethical_note: Ethical consideration or restriction
        best_photo_time: Exact recommended photography time window
    """
    time: str
    location: str
    activity: str
    duration_minutes: int
    crowd_prediction: float
    lighting_quality: str
    notes: Optional[str]
    day: Optional[int]
    order: Optional[int]
    icon: Optional[str]
    highlight: Optional[bool]
    ai_insight: Optional[str]
    cultural_tip: Optional[str]
    ethical_note: Optional[str]
    best_photo_time: Optional[str]


class TourPlanContext(TypedDict):
    """
    Context for tour plan generation requests.

    Attributes:
        selected_locations: List of locations to include in the plan
        start_date: Trip start date (ISO format)
        end_date: Trip end date (ISO format)
        preferences: User preferences (photography, adventure, etc.)
        constraints: User constraints (avoid crowds, etc.)
    """
    selected_locations: List[Dict[str, Any]]
    start_date: str
    end_date: str
    preferences: Optional[List[str]]
    constraints: Optional[List[str]]


class TourPlanMetadata(TypedDict):
    """
    Metadata for a generated tour plan.

    Attributes:
        match_score: Overall match score (0-100)
        total_days: Number of days in the plan
        total_locations: Number of locations covered
        golden_hour_optimized: Whether golden hour is optimized
        crowd_optimized: Whether crowd levels are optimized
        event_aware: Whether events/holidays are considered
        preference_match_explanation: Why this plan matches user interests
    """
    match_score: int
    total_days: int
    total_locations: int
    golden_hour_optimized: bool
    crowd_optimized: bool
    event_aware: bool
    preference_match_explanation: Optional[str]


class UserPreferences(TypedDict):
    """
    User's travel preference profile for personalized tour planning.

    Attributes:
        history: Interest in historical/cultural sites (0-1)
        adventure: Interest in adventure activities (0-1)
        nature: Interest in nature/wildlife (0-1)
        relaxation: Interest in relaxation/leisure (0-1)
        pace: Travel pace preference
        budget: Budget range
        group_size: Group size type
        dietary: Dietary restrictions
        accessibility: Accessibility needs
    """
    history: float
    adventure: float
    nature: float
    relaxation: float
    pace: Optional[str]
    budget: Optional[str]
    group_size: Optional[str]
    dietary: Optional[List[str]]
    accessibility: Optional[bool]


class ClarificationQuestion(TypedDict):
    """
    Structured question the agent asks the user during tour plan generation.

    Attributes:
        question: The question text
        options: List of selectable options
        context: Why this clarification is needed
        type: single_select or multi_select
    """
    question: str
    options: List[Dict[str, Any]]
    context: str
    type: str


class StepResult(TypedDict):
    """
    Result from a single node execution for progress tracking.

    Attributes:
        node: Node name (router, retrieval, shadow_monitor, etc.)
        status: success, warning, or error
        summary: Human-readable summary of what happened
        duration_ms: Execution time in milliseconds
    """
    node: str
    status: str
    summary: str
    duration_ms: float


class CulturalTip(TypedDict):
    """
    Cultural or ethical tip for a location.

    Attributes:
        location: Location name
        tip: The cultural/ethical tip text
        category: cultural, ethical, safety, or etiquette
    """
    location: str
    tip: str
    category: str


class GraphState(TypedDict):
    """
    Central state object for the Agentic RAG workflow.

    This TypedDict flows through all nodes in the LangGraph, accumulating
    information and decisions at each step. The design enables:

    1. **Observability**: Every reasoning step is logged
    2. **Self-Correction**: Flags trigger re-routing decisions
    3. **Multi-Objective Optimization**: Shadow Monitor integrates constraints

    Research Architecture:
        The state implements a "Reflective Agent" pattern where:
        - `retrieved_documents` captures perception (what the agent knows)
        - `document_relevance` enables reflection (is this good enough?)
        - `shadow_monitor_logs` records reasoning (what constraints apply?)
        - `constraint_violations` triggers correction (what went wrong?)
        - `reasoning_loops` prevents infinite recursion (safety limit)

    Attributes:
        messages: Conversation history (user + assistant messages)
        user_query: Current user input being processed
        intent: Classified intent type

        retrieved_documents: Documents fetched from ChromaDB
        document_relevance: Grader's assessment
        needs_web_search: Flag to trigger Tavily fallback
        web_search_results: External search findings

        shadow_monitor_logs: Audit trail of constraint checks
        constraint_violations: Detected issues requiring correction

        target_date: Extracted date from query (for Poya checks)
        target_location: Extracted location name
        target_coordinates: GPS for physics calculations

        generated_response: LLM output before verification
        final_response: Verified and corrected response
        itinerary: Structured trip plan (if requested)

        tour_plan_context: Context for tour plan generation
        tour_plan_metadata: Metadata about the generated plan

        reasoning_loops: Counter to prevent infinite loops
        error: Any error encountered during processing
    """

    # Conversation Context
    messages: Annotated[List[Dict[str, str]], operator.add]
    user_query: str
    intent: Optional[IntentType]

    # Retrieval State
    retrieved_documents: List[RetrievedDocument]
    document_relevance: Optional[DocumentRelevance]
    needs_web_search: bool
    web_search_results: List[Dict[str, Any]]

    # Shadow Monitor State
    shadow_monitor_logs: Annotated[List[ShadowMonitorLog], operator.add]
    constraint_violations: List[ConstraintViolation]

    # Extracted Entities
    target_date: Optional[str]
    target_location: Optional[str]
    target_coordinates: Optional[Dict[str, float]]

    # Response State
    generated_response: Optional[str]
    final_response: Optional[str]
    itinerary: Optional[List[ItinerarySlot]]

    # Tour Plan State
    tour_plan_context: Optional[TourPlanContext]
    tour_plan_metadata: Optional[TourPlanMetadata]

    # User Preferences (for personalized planning)
    user_preferences: Optional[UserPreferences]

    # Agent Clarification (interactive questioning)
    clarification_needed: bool
    clarification_question: Optional[ClarificationQuestion]

    # Step-by-step execution tracking (for LangSmith + mobile progress)
    step_results: Annotated[List[StepResult], operator.add]

    # Cultural & ethical tips
    cultural_tips: List[CulturalTip]

    # Control Flow
    reasoning_loops: int
    error: Optional[str]


def create_initial_state(
    user_query: str,
    target_location: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    tour_plan_context: Optional[TourPlanContext] = None,
    user_preferences: Optional[UserPreferences] = None
) -> GraphState:
    """
    Factory function to create a fresh GraphState for a new query.

    Args:
        user_query: The user's input message
        target_location: Optional location to focus retrieval on (for location-specific chats)
        conversation_history: Optional list of previous messages for conversation context
        tour_plan_context: Optional context for tour plan generation
        user_preferences: Optional user preference profile for personalized planning

    Returns:
        GraphState: Initialized state ready for graph execution

    Example:
        >>> state = create_initial_state("Plan a trip to Jungle Beach next full moon")
        >>> state["user_query"]
        'Plan a trip to Jungle Beach next full moon'

        >>> state = create_initial_state("What's the best time to visit?", target_location="Sigiriya")
        >>> state["target_location"]
        'Sigiriya'

        >>> prefs = {"history": 0.8, "adventure": 0.3, "nature": 0.6, "relaxation": 0.5}
        >>> state = create_initial_state("Plan my trip", user_preferences=prefs)
        >>> state["user_preferences"]["history"]
        0.8
    """
    # Start with conversation history if provided, otherwise empty list
    messages = []
    if conversation_history:
        messages.extend(conversation_history)

    # Add the current user query
    messages.append({"role": "user", "content": user_query})

    # Return as a dict (TypedDict is just for type hints, instantiate as regular dict)
    return {
        "messages": messages,
        "user_query": user_query,
        "intent": None,
        "retrieved_documents": [],
        "document_relevance": None,
        "needs_web_search": False,
        "web_search_results": [],
        "shadow_monitor_logs": [],
        "constraint_violations": [],
        "target_date": None,
        "target_location": target_location,
        "target_coordinates": None,
        "generated_response": None,
        "final_response": None,
        "itinerary": None,
        "tour_plan_context": tour_plan_context,
        "tour_plan_metadata": None,
        "user_preferences": user_preferences,
        "clarification_needed": False,
        "clarification_question": None,
        "step_results": [],
        "cultural_tips": [],
        "reasoning_loops": 0,
        "error": None,
    }

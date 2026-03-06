"""
Travion AI Engine: FastAPI Application Entry Point.

This module defines the FastAPI application for the Agentic Tour Guide,
including all endpoints for chat, planning, and tool access.

Architecture:
    The API exposes the LangGraph workflow through REST endpoints while
    providing direct access to individual tools (CrowdCast, EventSentinel,
    GoldenHour) for testing and integration.

Endpoints:
    - POST /api/v1/chat: Main conversational endpoint
    - POST /api/v1/plan: Trip planning with optimization
    - POST /api/v1/crowd: Crowd prediction
    - POST /api/v1/events: Event/holiday checking
    - POST /api/v1/golden-hour: Photography timing
    - GET /api/v1/health: Service health check
    - GET /api/v1/graph: Graph visualization
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager

# Fix "too many file descriptors in select()" on Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

from .config import settings
from .schemas import (
    ChatRequest,
    LocationChatRequest,
    PlanRequest,
    TourPlanGenerateRequest,
    CrowdPredictionRequest,
    EventCheckRequest,
    GoldenHourRequest,
    EventImpactRequest,
    PhysicsGoldenHourRequest,
    ClearChatHistoryRequest,
    ChatResponse,
    TourPlanResponse,
    TourPlanMetadataResponse,
    CrowdPredictionResponse,
    EventCheckResponse,
    GoldenHourResponse,
    HealthResponse,
    ErrorResponse,
    ItinerarySlotResponse,
    ConstraintViolationResponse,
    ShadowMonitorLogResponse,
    StepResultResponse,
    ClarificationQuestionResponse,
    ClarificationOptionResponse,
    CulturalTipResponse,
    EventInfoResponse,
    # Event Sentinel schemas (Temporal-Spatial Correlation)
    ConstraintInfo,
    BridgeDayInfo,
    TemporalIndexEntry,
    LocationSensitivity,
    EventImpactResponse,
    # Physics Golden Hour schemas
    TimeWindowResponse,
    LocationInfo,
    CalculationMetadata,
    SolarPositionResponse,
    PhysicsGoldenHourResponse,
    # Hotel/Restaurant search
    HotelSearchResultResponse,
    HotelSearchResponse,
    # Advanced Multi-Step Search (HITL)
    SearchCandidateResponse,
    AdvancedSearchResponse,
    SelectionRequest,
    WeatherResumeRequest,
    # Restaurant/Accommodation recommendations
    RestaurantRecommendationResponse,
    AccommodationRecommendationResponse,
)
from .schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedLocation,
    ExplanationRequest,
    ExplanationResponse,
    GeoLocation,
    ConstraintCheck,
    PreferenceConfidence,
)
from .graph import get_agent, invoke_agent, resume_agent_with_selection, resume_agent_with_weather_choice
from .graph.nodes.shadow_monitor import get_shadow_monitor
from .tools import (
    get_crowdcast,
    get_event_sentinel,
    get_golden_hour_agent,
)
from .physics import get_golden_hour_engine
from .core.recommender import get_recommender
from .agents.ranker import get_ranker_agent
from .utils.service_health import get_health_monitor, ServiceType

# Import tracing utilities
try:
    from .utils.tracing import init_langsmith, log_rag_metrics
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    def init_langsmith():
        return False
    def log_rag_metrics(*args, **kwargs):
        pass

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup/shutdown events with comprehensive health monitoring.

    Startup:
        - Initialize service health monitoring
        - Initialize LangSmith tracing
        - Initialize LangGraph agent
        - Initialize Active Guardian services
        - Report comprehensive service status

    Shutdown:
        - Cleanup resources
    """
    # Startup
    logger.info("=" * 80)
    logger.info("🚀 Starting Travion AI Engine...")
    logger.info("=" * 80)

    # Initialize health monitor FIRST
    health_monitor = get_health_monitor()
    for service_type in ServiceType:
        health_monitor.register_service(
            service_type,
            circuit_breaker_threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
            circuit_breaker_timeout=settings.CIRCUIT_BREAKER_TIMEOUT
        )
    logger.info("🏥 Service health monitor initialized")

    # Initialize LangSmith tracing
    if TRACING_AVAILABLE and settings.LANGCHAIN_API_KEY:
        tracing_enabled = init_langsmith(
            api_key=settings.LANGCHAIN_API_KEY,
            project_name=settings.LANGCHAIN_PROJECT,
            endpoint=settings.LANGCHAIN_ENDPOINT
        )
        if tracing_enabled:
            logger.info(f"✅ LangSmith tracing ENABLED")
            logger.info(f"   Project: {settings.LANGCHAIN_PROJECT}")
            logger.info(f"   Dashboard: https://smith.langchain.com/projects")
        else:
            logger.error("❌ LangSmith tracing FAILED to initialize")
            logger.error("   Check LANGCHAIN_API_KEY configuration")
    else:
        if not settings.LANGCHAIN_API_KEY:
            logger.warning("⚠️ LangSmith disabled: LANGCHAIN_API_KEY not set")
        else:
            logger.warning("⚠️ LangSmith unavailable: Install with `pip install langsmith`")

    # Initialize LLM and Agent
    try:
        agent = get_agent()
        logger.info(f"✅ Agent initialized: {type(agent.llm).__name__ if agent.llm else 'No LLM'}")
        health_monitor.report_success(ServiceType.LLM)
    except Exception as e:
        logger.error(f"❌ Agent initialization FAILED: {e}")
        health_monitor.report_failure(ServiceType.LLM, str(e))
        if settings.STRICT_VALIDATION:
            raise

    # Initialize Active Guardian Services
    logger.info("\n📡 Initializing Active Guardian Services...")

    # Event Sentinel (Critical)
    try:
        sentinel = get_event_sentinel()
        logger.info("✅ Event Sentinel: AVAILABLE")
        health_monitor.report_success(ServiceType.EVENT_SENTINEL)
    except Exception as e:
        logger.error(f"❌ Event Sentinel: FAILED - {e}")
        health_monitor.report_failure(ServiceType.EVENT_SENTINEL, str(e))

    # CrowdCast (Critical)
    try:
        crowdcast = get_crowdcast()
        logger.info("✅ CrowdCast: AVAILABLE")
        health_monitor.report_success(ServiceType.CROWDCAST)
    except Exception as e:
        logger.error(f"❌ CrowdCast: FAILED - {e}")
        health_monitor.report_failure(ServiceType.CROWDCAST, str(e))

    # Golden Hour (Critical)
    try:
        golden_hour = get_golden_hour_agent()
        logger.info("✅ Golden Hour: AVAILABLE")
        health_monitor.report_success(ServiceType.GOLDEN_HOUR)
    except Exception as e:
        logger.error(f"❌ Golden Hour: FAILED - {e}")
        health_monitor.report_failure(ServiceType.GOLDEN_HOUR, str(e))

    # Shadow Monitor and Active Guardian APIs
    shadow_monitor = get_shadow_monitor()

    # Weather API (Optional but important)
    if shadow_monitor.weather_tool and shadow_monitor.weather_tool.is_configured():
        logger.info("✅ Weather API: CONFIGURED")
    else:
        weather_health = health_monitor.get_health(ServiceType.WEATHER_API)
        if weather_health.error_message:
            logger.warning(f"⚠️ Weather API: NOT CONFIGURED")
            logger.warning(f"   └─ {weather_health.error_message}")
            logger.warning("   └─ Set OPENWEATHER_API_KEY in .env to enable weather validation")

    # News API (Optional - GDELT fallback available)
    if shadow_monitor.news_alert_tool:
        if os.getenv("NEWS_API_KEY"):
            logger.info("✅ News API: CONFIGURED (NewsAPI + GDELT)")
        else:
            logger.info("✅ News API: GDELT FALLBACK (no NewsAPI key - using free GDELT)")
    else:
        news_health = health_monitor.get_health(ServiceType.NEWS_API)
        if news_health.error_message:
            logger.warning(f"⚠️ News API: UNAVAILABLE")
            logger.warning(f"   └─ {news_health.error_message}")

    # ChromaDB (Vector Store)
    try:
        from .graph.nodes.retrieval import get_vectordb_service
        vectordb = get_vectordb_service()
        if vectordb.enabled:
            doc_count = vectordb.collection.count() if vectordb.collection else 0
            logger.info(f"✅ ChromaDB: AVAILABLE ({doc_count} documents)")
            health_monitor.report_success(ServiceType.CHROMADB)
        else:
            logger.warning("⚠️ ChromaDB: DISABLED (initialization failed)")
            health_monitor.report_failure(ServiceType.CHROMADB, "VectorDB not enabled")
    except Exception as e:
        logger.error(f"❌ ChromaDB: FAILED - {e}")
        health_monitor.report_failure(ServiceType.CHROMADB, str(e))

    # Initialize recommendation components
    try:
        recommender = get_recommender()
        loc_count = len(recommender.locations_df) if recommender.locations_df is not None else 0
        logger.info(f"✅ Recommender initialized: {loc_count} locations")
        ranker = get_ranker_agent()
        logger.info("✅ Ranker agent initialized")
    except Exception as e:
        logger.warning(f"⚠️ Recommendation components failed to initialize: {e}")

    # Print comprehensive service health summary
    logger.info("\n📊 Service Health Summary:")
    logger.info("─" * 80)
    all_health = health_monitor.get_all_health()
    for service_type, health in all_health.items():
        status_icon = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unavailable": "❌",
            "unknown": "❓"
        }.get(health.status.value, "❓")

        logger.info(f"   {status_icon} {service_type.value:<20} {health.status.value}")
        if health.error_message and health.status.value != "healthy":
            logger.info(f"      └─ {health.error_message}")

    # Print summary statistics
    summary = health_monitor.get_summary()
    logger.info("\n" + "─" * 80)
    logger.info(f"📈 Overall Status: {summary['overall_status'].upper()}")
    logger.info(f"   Total Services: {summary['total_services']}")
    logger.info(f"   Healthy: {summary['healthy_count']}")
    logger.info(f"   Degraded: {summary['degraded_count']}")
    logger.info(f"   Unavailable: {summary['unavailable_count']}")
    logger.info("=" * 80)
    logger.info(f"✅ Travion AI Engine ready on port {settings.PORT}")
    logger.info(f"   Environment: {settings.ENVIRONMENT}")
    logger.info(f"   Strict Validation: {settings.STRICT_VALIDATION}")
    logger.info("=" * 80)

    yield

    # Shutdown
    logger.info("🛑 Shutting down Travion AI Engine...")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
    ## Travion AI Engine

    An Agentic RAG system for Sri Lankan tourism with:

    - **Self-Correcting Reasoning Loop**: Router → Retrieval → Grader → Generator → Verifier
    - **Hybrid Recommendation Engine**: Cosine Similarity + Haversine Distance + LLM Re-ranking
    - **Shadow Monitoring**: Event Sentinel + CrowdCast + Golden Hour
    - **Multi-Objective Optimization**: Crowd, lighting, cultural constraints

    ### Key Features

    - Intelligent intent classification
    - Context-aware responses from ChromaDB
    - **Two-Stage Recommendations**: Mathematical retrieval + Agentic re-ranking
    - Poya day and holiday awareness
    - Crowd prediction and time optimization
    - Golden hour photography scheduling
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware with configurable origins
cors_origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Configurable via CORS_ORIGINS env var
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPER: User-scoped thread ID
# =============================================================================

def _build_user_thread_id(user_id: Optional[str], thread_id: Optional[str]) -> Optional[str]:
    """
    Build a user-scoped thread ID to ensure chat history isolation per user.
    If user_id is provided, prefix the thread_id with user_id.
    """
    if not thread_id:
        return None
    if user_id:
        return f"{user_id}_{thread_id}"
    return thread_id


# =============================================================================
# MAIN CHAT ENDPOINT
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Main conversational endpoint",
    description="""
    Process a user message through the agentic reasoning loop.

    The agent will:
    1. Classify intent (greeting, tourism query, trip planning)
    2. Retrieve relevant context from ChromaDB
    3. Check constraints (Poya days, crowds, lighting)
    4. Generate an optimized response
    5. Verify and self-correct if needed
    """
)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for conversational AI interaction.

    Args:
        request: ChatRequest with user message

    Returns:
        ChatResponse with agent's response and metadata
    """
    try:
        # Build user-scoped thread_id for chat history isolation
        scoped_thread_id = _build_user_thread_id(request.user_id, request.thread_id)

        # Invoke the agent
        result = await invoke_agent(
            query=request.message,
            thread_id=scoped_thread_id
        )

        # Check for errors
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Build response
        return ChatResponse(
            query=result["query"],
            intent=result.get("intent"),
            response=result.get("final_response", "I couldn't generate a response."),
            itinerary=[
                ItinerarySlotResponse(**slot)
                for slot in (result.get("itinerary") or [])
            ] if result.get("itinerary") else None,
            constraints=[
                ConstraintViolationResponse(
                    constraint_type=c.get("constraint_type", "unknown"),
                    description=c.get("description", ""),
                    severity=c.get("severity", "medium"),
                    suggestion=c.get("suggestion", "")
                )
                for c in (result.get("constraint_violations") or [])
            ] if result.get("constraint_violations") else None,
            reasoning_logs=[
                ShadowMonitorLogResponse(
                    timestamp=log.get("timestamp", ""),
                    check_type=log.get("check_type", ""),
                    result=log.get("result", ""),
                    details=log.get("details", "")
                )
                for log in (result.get("shadow_monitor_logs") or [])
            ] if result.get("shadow_monitor_logs") else None,
            metadata={
                "reasoning_loops": result.get("reasoning_loops", 0),
                "documents_retrieved": result.get("documents_retrieved", 0),
                "web_search_used": result.get("web_search_used", False)
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    f"{settings.API_V1_PREFIX}/chat/location",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Location-specific chat endpoint",
    description="""
    Process a user message with location-focused context.

    This endpoint is designed for location-specific chats where:
    1. The user is viewing a specific location (e.g., Sigiriya, Yala National Park)
    2. All queries should be answered in the context of that location
    3. Retrieval is filtered to prioritize location-relevant documents

    The agent will:
    1. Focus retrieval on the specified location
    2. Personalize responses based on user preference scores (if provided)
    3. Maintain conversation context via thread_id
    4. Apply the full reasoning loop (retrieve → grade → generate → verify)

    Use Cases:
    - "What's the best time to visit?" → Answered for the specific location
    - "Tell me about the history" → Location's historical significance
    - "What should I wear?" → Location-specific dress code advice
    """
)
async def location_chat(request: LocationChatRequest):
    """
    Location-specific chat endpoint for focused AI responses.

    Args:
        request: LocationChatRequest with message, location_name, and optional preferences

    Returns:
        ChatResponse with agent's location-focused response
    """
    try:
        # Build the query with location context for better routing
        # The target_location is passed separately to guide retrieval
        enriched_query = request.message

        # If user preferences are provided, we can use them for personalization
        # This info is logged but personalization happens in the generator node
        if request.user_preferences:
            logger.info(
                f"Location chat for {request.location_name} with preferences: "
                f"history={request.user_preferences.history}, "
                f"adventure={request.user_preferences.adventure}, "
                f"nature={request.user_preferences.nature}, "
                f"relaxation={request.user_preferences.relaxation}"
            )

        # Convert conversation history to the format expected by the agent
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]
            logger.info(f"Location chat includes {len(conversation_history)} previous messages")

        # Build user-scoped thread_id for chat history isolation
        scoped_thread_id = _build_user_thread_id(request.user_id, request.thread_id)

        # Invoke the agent with location focus and conversation history
        result = await invoke_agent(
            query=enriched_query,
            thread_id=scoped_thread_id,
            target_location=request.location_name,
            conversation_history=conversation_history
        )

        # Check for errors
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Build response
        return ChatResponse(
            query=result["query"],
            intent=result.get("intent"),
            response=result.get("final_response", "I couldn't generate a response."),
            itinerary=[
                ItinerarySlotResponse(**slot)
                for slot in (result.get("itinerary") or [])
            ] if result.get("itinerary") else None,
            constraints=[
                ConstraintViolationResponse(
                    constraint_type=c.get("constraint_type", "unknown"),
                    description=c.get("description", ""),
                    severity=c.get("severity", "medium"),
                    suggestion=c.get("suggestion", "")
                )
                for c in (result.get("constraint_violations") or [])
            ] if result.get("constraint_violations") else None,
            reasoning_logs=[
                ShadowMonitorLogResponse(
                    timestamp=log.get("timestamp", ""),
                    check_type=log.get("check_type", ""),
                    result=log.get("result", ""),
                    details=log.get("details", "")
                )
                for log in (result.get("shadow_monitor_logs") or [])
            ] if result.get("shadow_monitor_logs") else None,
            metadata={
                "reasoning_loops": result.get("reasoning_loops", 0),
                "documents_retrieved": result.get("documents_retrieved", 0),
                "web_search_used": result.get("web_search_used", False),
                "target_location": request.location_name
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Location chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TOUR PLAN RESPONSE BUILDER (Shared by generate, refine, resume-weather)
# =============================================================================

def _build_tour_plan_response(result: dict, thread_id: str) -> TourPlanResponse:
    """
    Convert the raw LangGraph result dict into a TourPlanResponse model.
    Re-used by generate, refine, and resume-weather endpoints.
    """
    # Build itinerary response
    itinerary = []
    for slot in (result.get("itinerary") or []):
        itinerary.append(ItinerarySlotResponse(
            time=slot.get("time", "09:00"),
            location=slot.get("location", ""),
            activity=slot.get("activity", ""),
            duration_minutes=slot.get("duration_minutes", 60),
            crowd_prediction=slot.get("crowd_prediction", 50),
            lighting_quality=slot.get("lighting_quality", "good"),
            notes=slot.get("notes"),
            day=slot.get("day"),
            order=slot.get("order"),
            icon=slot.get("icon"),
            highlight=slot.get("highlight", False),
            ai_insight=slot.get("ai_insight"),
            cultural_tip=slot.get("cultural_tip"),
            ethical_note=slot.get("ethical_note"),
            best_photo_time=slot.get("best_photo_time"),
        ))

    # Build metadata response
    plan_metadata = result.get("tour_plan_metadata") or {}
    metadata = TourPlanMetadataResponse(
        match_score=plan_metadata.get("match_score", 85),
        total_days=plan_metadata.get("total_days", 1),
        total_locations=plan_metadata.get("total_locations", 1),
        golden_hour_optimized=plan_metadata.get("golden_hour_optimized", True),
        crowd_optimized=plan_metadata.get("crowd_optimized", True),
        event_aware=plan_metadata.get("event_aware", True),
        preference_match_explanation=plan_metadata.get("preference_match_explanation"),
    )

    # Build constraint violations
    constraints = None
    if result.get("constraint_violations"):
        constraints = [
            ConstraintViolationResponse(
                constraint_type=c.get("constraint_type", "unknown"),
                description=c.get("description", ""),
                severity=c.get("severity", "medium"),
                suggestion=c.get("suggestion", "")
            )
            for c in result.get("constraint_violations", [])
        ]

    # Build reasoning logs
    reasoning_logs = None
    if result.get("shadow_monitor_logs"):
        reasoning_logs = [
            ShadowMonitorLogResponse(
                timestamp=log.get("timestamp", ""),
                check_type=log.get("check_type", ""),
                result=log.get("result", ""),
                details=log.get("details", "")
            )
            for log in result.get("shadow_monitor_logs", [])
        ]

    # Build step results
    step_results = None
    if result.get("step_results"):
        step_results = [
            StepResultResponse(
                node=s.get("node", ""),
                status=s.get("status", ""),
                summary=s.get("summary", ""),
                duration_ms=s.get("duration_ms", 0),
            )
            for s in result.get("step_results", [])
        ]

    # Build clarification question
    clarification_question = None
    clarification_data = result.get("clarification_question")
    if clarification_data and result.get("clarification_needed"):
        clarification_question = ClarificationQuestionResponse(
            question=clarification_data.get("question", ""),
            options=[
                ClarificationOptionResponse(
                    label=opt.get("label", ""),
                    description=opt.get("description", ""),
                    recommended=opt.get("recommended", False),
                )
                for opt in clarification_data.get("options", [])
            ],
            context=clarification_data.get("context", ""),
            type=clarification_data.get("type", "single_select"),
        )

    # Build cultural tips
    cultural_tips = None
    if result.get("cultural_tips"):
        cultural_tips = [
            CulturalTipResponse(
                location=tip.get("location", ""),
                tip=tip.get("tip", ""),
                category=tip.get("category", "cultural"),
            )
            for tip in result.get("cultural_tips", [])
        ]

    # Extract warnings and tips
    warnings = result.get("warnings") or None
    tips = result.get("tips") or None

    # Build restaurant recommendations
    restaurant_recommendations = None
    raw_restaurant_recs = result.get("restaurant_recommendations")
    if raw_restaurant_recs:
        restaurant_recommendations = [
            RestaurantRecommendationResponse(
                id=r.get("id", ""),
                name=r.get("name", ""),
                rating=r.get("rating"),
                cuisine_type=r.get("cuisine_type"),
                price_range=r.get("price_range"),
                url=r.get("url"),
                description=r.get("description", ""),
                near_location=r.get("near_location", ""),
                meal_slot=r.get("meal_slot", "lunch"),
                day=r.get("day", 1),
            )
            for r in raw_restaurant_recs
        ]

    # Build accommodation recommendations
    accommodation_recommendations = None
    raw_accommodation_recs = result.get("accommodation_recommendations")
    if raw_accommodation_recs:
        accommodation_recommendations = [
            AccommodationRecommendationResponse(
                id=r.get("id", ""),
                name=r.get("name", ""),
                rating=r.get("rating"),
                price_range=r.get("price_range"),
                url=r.get("url"),
                description=r.get("description", ""),
                near_location=r.get("near_location", ""),
                check_in_day=r.get("check_in_day", 1),
                type=r.get("type", "hotel"),
            )
            for r in raw_accommodation_recs
        ]

    return TourPlanResponse(
        success=True,
        thread_id=thread_id,
        response=result.get("final_response") or "Tour plan generated successfully!",
        itinerary=itinerary,
        metadata=metadata,
        constraints=constraints,
        reasoning_logs=reasoning_logs,
        warnings=warnings,
        tips=tips,
        step_results=step_results,
        clarification_question=clarification_question,
        cultural_tips=cultural_tips,
        events=None,  # Events built inline by generate endpoint if needed
        final_itinerary=result.get("final_itinerary"),
        weather_data=result.get("weather_data"),
        interrupt_reason=result.get("interrupt_reason"),
        weather_interrupt=result.get("weather_interrupt", False),
        weather_prompt_message=result.get("weather_prompt_message"),
        weather_prompt_options=result.get("weather_prompt_options"),
        pending_user_selection=result.get("pending_user_selection", False),
        selection_cards=result.get("selection_cards"),
        prompt_text=result.get("prompt_text"),
        search_candidates=result.get("search_candidates"),
        mcp_search_metadata=result.get("mcp_search_metadata"),
        restaurant_recommendations=restaurant_recommendations,
        accommodation_recommendations=accommodation_recommendations,
    )


# =============================================================================
# TOUR PLAN GENERATION ENDPOINT
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/tour-plan/generate",
    response_model=TourPlanResponse,
    tags=["Tour Planning"],
    summary="Generate an optimized tour plan",
    description="""
    Generate a comprehensive multi-day tour plan using the agentic RAG workflow.

    The agent will:
    1. Analyze selected locations and date range
    2. Retrieve relevant knowledge for each location
    3. Optimize visit order based on distance and logistics
    4. Check constraints (Poya days, crowds, weather)
    5. Generate optimized itinerary with golden hour timing
    6. Provide AI insights for each activity

    Features:
    - Multi-day itinerary generation
    - Crowd prediction and avoidance
    - Golden hour optimization for photography
    - Event/holiday awareness
    - Session-based conversation continuity for plan refinement
    """
)
async def generate_tour_plan(request: TourPlanGenerateRequest):
    """
    Tour plan generation endpoint.

    Args:
        request: TourPlanGenerateRequest with locations, dates, and preferences

    Returns:
        TourPlanResponse with optimized itinerary and metadata
    """
    import uuid

    try:
        # Generate or use provided thread_id, scoped to user
        raw_thread_id = request.thread_id or f"tour_{uuid.uuid4().hex[:12]}"
        thread_id = _build_user_thread_id(request.user_id, raw_thread_id) or raw_thread_id

        # Build tour plan context
        tour_plan_context = {
            "selected_locations": [
                {
                    "name": loc.name,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "image_url": loc.image_url,
                    "distance_km": loc.distance_km,
                }
                for loc in request.selected_locations
            ],
            "start_date": request.start_date,
            "end_date": request.end_date,
            "preferences": request.preferences,
            "constraints": None,
            # Restaurant/accommodation selection fields (for refine flow)
            "selected_restaurant_ids": request.selected_restaurant_ids,
            "selected_accommodation_ids": request.selected_accommodation_ids,
            "skip_restaurants": request.skip_restaurants,
            "skip_accommodations": request.skip_accommodations,
        }
        
        # Build the query message
        location_names = [loc.name for loc in request.selected_locations]
        query = request.message or f"Generate an optimized tour plan for {', '.join(location_names)} from {request.start_date} to {request.end_date}"
        
        logger.info(f"Generating tour plan for {len(location_names)} locations: {location_names}")
        
        # Invoke the agent with tour plan context
        result = await invoke_agent(
            query=query,
            thread_id=thread_id,
            target_location=location_names[0] if location_names else None,
            tour_plan_context=tour_plan_context
        )
        
        # Check for errors
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Build itinerary response
        itinerary = []
        for slot in (result.get("itinerary") or []):
            itinerary.append(ItinerarySlotResponse(
                time=slot.get("time", "09:00"),
                location=slot.get("location", ""),
                activity=slot.get("activity", ""),
                duration_minutes=slot.get("duration_minutes", 60),
                crowd_prediction=slot.get("crowd_prediction", 50),
                lighting_quality=slot.get("lighting_quality", "good"),
                notes=slot.get("notes"),
                day=slot.get("day"),
                order=slot.get("order"),
                icon=slot.get("icon"),
                highlight=slot.get("highlight", False),
                ai_insight=slot.get("ai_insight"),
                cultural_tip=slot.get("cultural_tip"),
                ethical_note=slot.get("ethical_note"),
                best_photo_time=slot.get("best_photo_time"),
            ))
        
        # Build metadata response
        plan_metadata = result.get("tour_plan_metadata") or {}
        metadata = TourPlanMetadataResponse(
            match_score=plan_metadata.get("match_score", 85),
            total_days=plan_metadata.get("total_days", 1),
            total_locations=plan_metadata.get("total_locations", len(location_names)),
            golden_hour_optimized=plan_metadata.get("golden_hour_optimized", True),
            crowd_optimized=plan_metadata.get("crowd_optimized", True),
            event_aware=plan_metadata.get("event_aware", True),
            preference_match_explanation=plan_metadata.get("preference_match_explanation"),
        )
        
        # Build constraint violations
        constraints = None
        if result.get("constraint_violations"):
            constraints = [
                ConstraintViolationResponse(
                    constraint_type=c.get("constraint_type", "unknown"),
                    description=c.get("description", ""),
                    severity=c.get("severity", "medium"),
                    suggestion=c.get("suggestion", "")
                )
                for c in result.get("constraint_violations", [])
            ]
        
        # Build reasoning logs
        reasoning_logs = None
        if result.get("shadow_monitor_logs"):
            reasoning_logs = [
                ShadowMonitorLogResponse(
                    timestamp=log.get("timestamp", ""),
                    check_type=log.get("check_type", ""),
                    result=log.get("result", ""),
                    details=log.get("details", "")
                )
                for log in result.get("shadow_monitor_logs", [])
            ]
        
        # Build step results
        step_results = None
        if result.get("step_results"):
            step_results = [
                StepResultResponse(
                    node=s.get("node", ""),
                    status=s.get("status", ""),
                    summary=s.get("summary", ""),
                    duration_ms=s.get("duration_ms", 0),
                )
                for s in result.get("step_results", [])
            ]

        # Build clarification question
        clarification_question = None
        clarification_data = result.get("clarification_question")
        if clarification_data and result.get("clarification_needed"):
            clarification_question = ClarificationQuestionResponse(
                question=clarification_data.get("question", ""),
                options=[
                    ClarificationOptionResponse(
                        label=opt.get("label", ""),
                        description=opt.get("description", ""),
                        recommended=opt.get("recommended", False),
                    )
                    for opt in clarification_data.get("options", [])
                ],
                context=clarification_data.get("context", ""),
                type=clarification_data.get("type", "single_select"),
            )

        # Build cultural tips
        cultural_tips = None
        if result.get("cultural_tips"):
            cultural_tips = [
                CulturalTipResponse(
                    location=tip.get("location", ""),
                    tip=tip.get("tip", ""),
                    category=tip.get("category", "cultural"),
                )
                for tip in result.get("cultural_tips", [])
            ]

        # Extract warnings and tips
        warnings = result.get("warnings") or None
        tips = result.get("tips") or None

        # Build events info from event data
        events = None
        event_data = result.get("event_data") or {}
        if event_data:
            events_list = []
            # Poya days
            for poya in event_data.get("poya_days", []):
                if isinstance(poya, dict):
                    events_list.append(EventInfoResponse(
                        date=poya.get("date", ""),
                        name=poya.get("name", "Poya Day"),
                        type="poya",
                        impact=f"Crowd modifier: {poya.get('crowd_modifier', 1.0)}x",
                        warnings=poya.get("warnings", []),
                    ))
                elif isinstance(poya, str):
                    events_list.append(EventInfoResponse(
                        date=poya,
                        name="Poya Full Moon Day",
                        type="poya",
                        impact="Alcohol sales banned. Expect higher crowds at religious sites.",
                        warnings=["Alcohol sales banned island-wide on Poya days"],
                    ))
            # Holidays
            for holiday in event_data.get("holidays", []):
                if isinstance(holiday, dict):
                    events_list.append(EventInfoResponse(
                        date=holiday.get("date", ""),
                        name=holiday.get("name", "Holiday"),
                        type="holiday",
                        impact=holiday.get("impact", ""),
                        warnings=holiday.get("warnings", []),
                    ))
                elif isinstance(holiday, str):
                    events_list.append(EventInfoResponse(
                        date="",
                        name=holiday,
                        type="holiday",
                        impact="Public holiday — some businesses may be closed.",
                        warnings=[],
                    ))
            if events_list:
                events = events_list

        # Build restaurant recommendations
        restaurant_recommendations = None
        raw_restaurant_recs = result.get("restaurant_recommendations")
        if raw_restaurant_recs:
            restaurant_recommendations = [
                RestaurantRecommendationResponse(
                    id=r.get("id", ""),
                    name=r.get("name", ""),
                    rating=r.get("rating"),
                    cuisine_type=r.get("cuisine_type"),
                    price_range=r.get("price_range"),
                    url=r.get("url"),
                    description=r.get("description", ""),
                    near_location=r.get("near_location", ""),
                    meal_slot=r.get("meal_slot", "lunch"),
                    day=r.get("day", 1),
                )
                for r in raw_restaurant_recs
            ]

        # Build accommodation recommendations
        accommodation_recommendations = None
        raw_accommodation_recs = result.get("accommodation_recommendations")
        if raw_accommodation_recs:
            accommodation_recommendations = [
                AccommodationRecommendationResponse(
                    id=r.get("id", ""),
                    name=r.get("name", ""),
                    rating=r.get("rating"),
                    price_range=r.get("price_range"),
                    url=r.get("url"),
                    description=r.get("description", ""),
                    near_location=r.get("near_location", ""),
                    check_in_day=r.get("check_in_day", 1),
                    type=r.get("type", "hotel"),
                )
                for r in raw_accommodation_recs
            ]

        return TourPlanResponse(
            success=True,
            thread_id=thread_id,
            response=result.get("final_response") or ("I need a bit more information to create the best plan for you." if result.get("clarification_needed") else "Tour plan generated successfully!"),
            itinerary=itinerary,
            metadata=metadata,
            constraints=constraints,
            reasoning_logs=reasoning_logs,
            warnings=warnings,
            tips=tips,
            step_results=step_results,
            clarification_question=clarification_question,
            cultural_tips=cultural_tips,
            events=events,
            final_itinerary=result.get("final_itinerary"),
            weather_data=result.get("weather_data"),
            interrupt_reason=result.get("interrupt_reason"),
            restaurant_recommendations=restaurant_recommendations,
            accommodation_recommendations=accommodation_recommendations,
            # HITL selection fields (restaurant selection during plan generation)
            pending_user_selection=result.get("pending_user_selection", False),
            selection_cards=result.get("selection_cards"),
            prompt_text=result.get("prompt_text"),
            search_candidates=None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Tour plan generation error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    f"{settings.API_V1_PREFIX}/tour-plan/refine",
    response_model=TourPlanResponse,
    tags=["Tour Planning"],
    summary="Refine an existing tour plan",
    description="""
    Refine an existing tour plan based on user feedback.
    
    Uses the session thread_id to maintain conversation context
    and apply modifications to the previously generated plan.
    """
)
async def refine_tour_plan(request: TourPlanGenerateRequest):
    """
    Tour plan refinement endpoint.
    
    Requires thread_id from a previous generate call.
    """
    if not request.thread_id:
        raise HTTPException(
            status_code=400, 
            detail="thread_id is required for plan refinement. Generate a plan first."
        )
    
    if not request.message:
        raise HTTPException(
            status_code=400,
            detail="message is required for plan refinement. Specify what changes you want."
        )
    
    # Use the same generate logic with the existing thread_id
    return await generate_tour_plan(request)


# =============================================================================
# HOTEL / RESTAURANT SEARCH ENDPOINT
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/hotel-search",
    response_model=HotelSearchResponse,
    tags=["Tour Planning"],
    summary="Search for hotels, restaurants, or activities near a location",
    description="""
    Search for accommodation, dining, or nightlife options near a specified location
    using real-time web search (Tavily).

    Returns 3-5 structured results with name, price range, rating, and booking links.

    Search types are auto-detected from the query:
    - Hotels/accommodation: "hotels near Sigiriya", "where to stay in Galle"
    - Restaurants/dining: "restaurants in Colombo", "best food near Ella"
    - Bars/nightlife: "nightlife in Mirissa", "bars near Unawatuna"
    """
)
async def hotel_search(query: str, location: Optional[str] = None):
    """
    Hotel/Restaurant search endpoint.

    Args:
        query: Natural language search query (e.g., "best hotels near Sigiriya")
        location: Optional explicit location name (overrides extraction from query)

    Returns:
        HotelSearchResponse with structured search results
    """
    try:
        result = await invoke_agent(
            query=query,
            target_location=location,
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        hotel_results = result.get("hotel_search_results") or []

        results_response = [
            HotelSearchResultResponse(
                name=r.get("name", ""),
                type=r.get("type", "hotel"),
                price_range=r.get("price_range"),
                rating=r.get("rating"),
                url=r.get("url"),
                description=r.get("description", ""),
                distance_from_location=r.get("distance_from_location"),
                location_name=r.get("location_name", ""),
            )
            for r in hotel_results
        ]

        # Determine search type from query
        from .graph.nodes.hotel_search import _determine_search_type
        search_type = _determine_search_type(query)

        return HotelSearchResponse(
            success=True,
            query=query,
            search_type=search_type,
            location=location or result.get("target_location", "Sri Lanka"),
            results=results_response,
            total_results=len(results_response),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hotel search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ADVANCED SEARCH SELECTION (HITL RESUME) ENDPOINT
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/select",
    tags=["Tour Planning"],
    summary="Submit user selection to resume the paused search graph",
    description="""
    **Human-in-the-Loop (HITL) resume endpoint.**

    After the tour-plan or hotel-search flow returns candidates with
    `pending_user_selection=True`, the mobile app calls this endpoint
    with the `thread_id` and the chosen `selected_candidate_id`.

    The paused LangGraph will:
    1. Inject the selection into state
    2. Run the **Selection Handler** node
       - Event Sentinel constraint re-check
       - Haversine travel-time recalculation
    3. Continue the remaining graph (verify → tour plan)
    4. Return the re-optimised result
    """
)
async def submit_selection(request: SelectionRequest):
    """
    Resume the paused LangGraph after user selects a search candidate.

    Works for both:
    - Advanced search HITL (hotel/restaurant discovery)
    - Restaurant selection HITL (during tour plan generation)

    Args:
        request: SelectionRequest with thread_id and selected_candidate_id

    Returns:
        Full tour plan response when the graph completes a plan,
        or AdvancedSearchResponse for simple search selections.
    """
    try:
        # The thread_id from the client is already user-scoped (returned by
        # the generate endpoint).  Do NOT re-scope — that would create a
        # double-prefixed ID that doesn't match the checkpoint.
        scoped_thread_id = request.thread_id

        logger.info(
            f"HITL selection received — thread={scoped_thread_id}, "
            f"candidate={request.selected_candidate_id}"
        )

        # Resume the graph
        result = await resume_agent_with_selection(
            thread_id=scoped_thread_id,
            selected_candidate_id=request.selected_candidate_id,
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        # --- If the resumed graph produced a tour plan OR needs another HITL
        #     selection step, return TourPlanResponse ---
        if result.get("itinerary") or result.get("final_itinerary") or result.get("pending_user_selection"):
            itinerary = []
            for slot in (result.get("itinerary") or []):
                itinerary.append(ItinerarySlotResponse(
                    time=slot.get("time", "09:00"),
                    location=slot.get("location", ""),
                    activity=slot.get("activity", ""),
                    duration_minutes=slot.get("duration_minutes", 60),
                    crowd_prediction=slot.get("crowd_prediction", 50),
                    lighting_quality=slot.get("lighting_quality", "good"),
                    notes=slot.get("notes"),
                    day=slot.get("day"),
                    order=slot.get("order"),
                    icon=slot.get("icon"),
                    highlight=slot.get("highlight", False),
                    ai_insight=slot.get("ai_insight"),
                    cultural_tip=slot.get("cultural_tip"),
                    ethical_note=slot.get("ethical_note"),
                    best_photo_time=slot.get("best_photo_time"),
                ))

            plan_metadata = result.get("tour_plan_metadata") or {}
            metadata = TourPlanMetadataResponse(
                match_score=plan_metadata.get("match_score", 85),
                total_days=plan_metadata.get("total_days", 1),
                total_locations=plan_metadata.get("total_locations", 1),
                golden_hour_optimized=plan_metadata.get("golden_hour_optimized", True),
                crowd_optimized=plan_metadata.get("crowd_optimized", True),
                event_aware=plan_metadata.get("event_aware", True),
                preference_match_explanation=plan_metadata.get("preference_match_explanation"),
            )

            restaurant_recommendations = None
            raw_recs = result.get("restaurant_recommendations")
            if raw_recs:
                # Field is now List[Dict] — convert any non-dict items to plain dicts
                restaurant_recommendations = [
                    dict(r) if isinstance(r, dict) else
                    r.model_dump() if hasattr(r, "model_dump") else
                    dict(r)
                    for r in raw_recs
                ]

            cultural_tips = None
            if result.get("cultural_tips"):
                cultural_tips = [
                    CulturalTipResponse(
                        location=tip.get("location", ""),
                        tip=tip.get("tip", ""),
                        category=tip.get("category", "cultural"),
                    )
                    for tip in result.get("cultural_tips", [])
                ]

            step_results = None
            if result.get("step_results"):
                step_results = [
                    StepResultResponse(
                        node=s.get("node", ""),
                        status=s.get("status", ""),
                        summary=s.get("summary", ""),
                        duration_ms=s.get("duration_ms", 0),
                    )
                    for s in result.get("step_results", [])
                ]

            return TourPlanResponse(
                success=True,
                thread_id=scoped_thread_id,
                response=result.get("final_response") or "Tour plan generated with your restaurant selection!",
                itinerary=itinerary,
                metadata=metadata,
                step_results=step_results,
                cultural_tips=cultural_tips,
                final_itinerary=result.get("final_itinerary"),
                weather_data=result.get("weather_data"),
                restaurant_recommendations=restaurant_recommendations,
                pending_user_selection=result.get("pending_user_selection", False),
                selection_cards=result.get("selection_cards"),
                prompt_text=result.get("prompt_text"),
                weather_interrupt=result.get("weather_interrupt", False),
                weather_prompt_message=result.get("weather_prompt_message"),
                weather_prompt_options=result.get("weather_prompt_options"),
            )

        # --- Otherwise, return AdvancedSearchResponse (hotel/restaurant discovery) ---
        # Build candidate responses
        candidates = [
            SearchCandidateResponse(
                id=c.get("id", ""),
                name=c.get("name", ""),
                type=c.get("type", "hotel"),
                description=c.get("description", ""),
                price_range=c.get("price_range"),
                rating=c.get("rating"),
                opening_hours=c.get("opening_hours"),
                lat=c.get("lat"),
                lng=c.get("lng"),
                url=c.get("url"),
                location_name=c.get("location_name", ""),
                vibe_match_score=c.get("vibe_match_score"),
            )
            for c in (result.get("search_candidates") or [])
        ]

        selected = result.get("selected_search_candidate") or {}

        return AdvancedSearchResponse(
            success=True,
            query=result.get("query", ""),
            search_type=selected.get("type", "hotel"),
            location=selected.get("location_name", ""),
            vibe="",
            candidates=candidates,
            total_candidates=len(candidates),
            pending_user_selection=result.get("pending_user_selection", False),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Selection endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WEATHER INTERRUPT RESUME (HITL) ENDPOINT
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/resume-weather",
    response_model=TourPlanResponse,
    tags=["Tour Planning"],
    summary="Resume the paused graph after a weather interrupt decision",
    description="""
    **Human-in-the-Loop (HITL) weather resume endpoint.**

    After the Shadow Monitor detects severe weather and the tour-plan
    response returns `weather_interrupt=True` with a prompt message and
    options, the mobile app calls this endpoint with the `thread_id`
    and the user's chosen `user_weather_choice`.

    Valid choices:
    - `switch_indoor` — Replace outdoor stops with indoor alternatives
    - `reschedule` — Move affected stops to a different time/day
    - `keep` — Keep the original plan despite weather warnings

    The paused LangGraph will:
    1. Inject the weather choice into state
    2. Continue from the shadow_monitor node
    3. Re-run Event Sentinel + Haversine constraint checks
    4. Return the re-optimised tour plan
    """
)
async def resume_weather(request: WeatherResumeRequest):
    """
    Resume the paused LangGraph after user decides on a weather action.

    Args:
        request: WeatherResumeRequest with thread_id and user_weather_choice

    Returns:
        TourPlanResponse with the re-optimised itinerary
    """
    try:
        # The thread_id from the client is already user-scoped (returned by
        # the generate endpoint).  Do NOT re-scope — that would create a
        # double-prefixed ID that doesn't match the checkpoint.
        scoped_thread_id = request.thread_id

        logger.info(
            f"HITL weather resume — thread={scoped_thread_id}, "
            f"choice={request.user_weather_choice}"
        )

        # Validate the choice
        valid_choices = {"switch_indoor", "reschedule", "keep"}
        if request.user_weather_choice not in valid_choices:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid weather choice '{request.user_weather_choice}'. "
                    f"Must be one of: {', '.join(sorted(valid_choices))}"
                ),
            )

        # Resume the graph with the weather choice
        result = await resume_agent_with_weather_choice(
            thread_id=scoped_thread_id,
            user_choice=request.user_weather_choice,
        )

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        # Build standard tour plan response
        return _build_tour_plan_response(result, scoped_thread_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Weather resume endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CLEAR CHAT HISTORY ENDPOINT
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/chat/clear-history",
    tags=["Chat"],
    summary="Clear chat history for a thread",
    description="""
    Clear the LangGraph checkpoint memory for a specific thread.
    This removes the conversation history stored in the agent's memory
    for the given user and thread combination.
    """
)
async def clear_chat_history(request: ClearChatHistoryRequest):
    """
    Clear chat history endpoint.

    Args:
        request: ClearChatHistoryRequest with thread_id and user_id

    Returns:
        Success status
    """
    try:
        # Build user-scoped thread_id
        scoped_thread_id = _build_user_thread_id(request.user_id, request.thread_id)
        if not scoped_thread_id:
            scoped_thread_id = request.thread_id

        # Clear the memory for this thread
        agent = get_agent()
        if agent.memory and hasattr(agent.memory, 'storage'):
            # MemorySaver stores checkpoints in a dict keyed by thread_id
            # Remove all checkpoints for this thread
            keys_to_remove = [
                key for key in agent.memory.storage.keys()
                if isinstance(key, tuple) and len(key) >= 1 and key[0] == scoped_thread_id
            ]
            for key in keys_to_remove:
                del agent.memory.storage[key]

            logger.info(f"Cleared {len(keys_to_remove)} checkpoint(s) for thread: {scoped_thread_id}")

        return {
            "success": True,
            "message": f"Chat history cleared for thread",
            "thread_id": request.thread_id,
        }

    except Exception as e:
        logger.error(f"Clear chat history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RECOMMENDATION ENDPOINTS
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/recommend",
    response_model=RecommendationResponse,
    tags=["Recommendations"],
    summary="Get personalized location recommendations",
    description="""
    Get top-K location recommendations using the Hybrid Recommendation Engine.

    **Stage 1: Candidate Generation**
    - Content-Based Filtering with Cosine Similarity
    - Geospatial Filtering with Haversine Distance

    **Stage 2: Agentic Re-ranking**
    - Constraint checking (crowds, weather, holidays)
    - LLM-powered reasoning generation
    - Self-correction loop for blocked locations
    """
)
async def get_recommendations(request: RecommendationRequest):
    """
    Hybrid Recommendation Endpoint.

    Combines mathematical retrieval with LLM-based contextual re-ranking.
    Enhanced with confidence weights, category boosting, and personalization.

    Args:
        request: RecommendationRequest with user preferences and location

    Returns:
        RecommendationResponse with ranked recommendations and reasoning
    """
    import time
    start_time = time.time()

    try:
        # Stage 1: Candidate Generation
        recommender = get_recommender()
        user_prefs = request.preferences.to_vector()

        # Get confidence weights if provided
        confidence_weights = None
        if request.preference_confidence:
            confidence_weights = request.preference_confidence.to_vector()

        candidates = recommender.get_candidates(
            user_preferences=user_prefs,
            user_lat=request.current_lat,
            user_lng=request.current_lng,
            top_k=request.top_k * 3,  # Get 3x for re-ranking buffer
            max_distance_km=request.max_distance_km,  # Use request's max distance (default 20km)
            outdoor_only=request.outdoor_only,
            exclude_locations=request.exclude_locations,
            visited_locations=request.visited_locations,
            favorite_categories=request.favorite_categories,
            avoid_categories=request.avoid_categories,
            confidence_weights=confidence_weights,
            search_history_boost=request.search_history
        )

        if not candidates:
            return RecommendationResponse(
                success=False,
                user_id=request.user_id,
                request_location=GeoLocation(
                    latitude=request.current_lat,
                    longitude=request.current_lng
                ),
                target_datetime=request.target_datetime,
                recommendations=[],
                metadata={"error": "No candidates found within range"},
                reasoning_summary="No locations found matching your criteria."
            )

        # Stage 2: Agentic Re-ranking
        ranker = get_ranker_agent()
        rerank_result = await ranker.rerank(
            candidates=candidates,
            user_lat=request.current_lat,
            user_lng=request.current_lng,
            user_preferences=user_prefs,
            target_datetime=request.target_datetime
        )

        # Build response
        recommendations = []
        for rec in rerank_result.get("recommendations", [])[:request.top_k]:
            # Convert constraint checks to Pydantic models
            constraint_checks = []
            for ctype, cdata in rec.get("constraint_checks", {}).items():
                if cdata:
                    constraint_checks.append(ConstraintCheck(
                        constraint_type=ctype,
                        status=cdata.get("status", "ok"),
                        value=cdata.get("value"),
                        message=cdata.get("message", "")
                    ))

            recommendations.append(RecommendedLocation(
                rank=rec.get("rank", len(recommendations) + 1),
                name=rec.get("name", "Unknown"),
                latitude=rec.get("lat", 0.0),
                longitude=rec.get("lng", 0.0),
                similarity_score=rec.get("similarity_score", 0.0),
                distance_km=rec.get("distance_km", 0.0),
                combined_score=rec.get("combined_score", 0.0),
                preference_scores=rec.get("preference_scores", {}),
                is_outdoor=rec.get("is_outdoor", True),
                constraint_checks=constraint_checks,
                reasoning=rec.get("reasoning", ""),
                optimal_visit_time=rec.get("optimal_visit_time"),
                warnings=rec.get("warnings", [])
            ))

        processing_time_ms = int((time.time() - start_time) * 1000)

        return RecommendationResponse(
            success=True,
            user_id=request.user_id,
            request_location=GeoLocation(
                latitude=request.current_lat,
                longitude=request.current_lng
            ),
            target_datetime=request.target_datetime,
            recommendations=recommendations,
            metadata={
                "candidates_evaluated": len(candidates),
                "processing_time_ms": processing_time_ms,
                "max_distance_km": request.max_distance_km,
                "self_corrections": rerank_result.get("self_corrections", 0),
                "constraints_checked": ["crowd", "lighting", "holiday"]
            },
            reasoning_summary=rerank_result.get(
                "overall_reasoning",
                "Recommendations based on your preferences and location."
            )
        )

    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    f"{settings.API_V1_PREFIX}/explain/{{location_name}}",
    response_model=ExplanationResponse,
    tags=["Recommendations"],
    summary="Get detailed recommendation explanation",
    description="""
    Get deep reasoning for why a specific location is recommended.

    Provides:
    - Preference match analysis
    - Constraint analysis (crowds, weather, etc.)
    - Similar location suggestions
    - Visit tips and optimal times
    """
)
async def explain_recommendation(
    location_name: str,
    user_lat: Optional[float] = None,
    user_lng: Optional[float] = None
):
    """
    Explanation Endpoint for recommendation reasoning.

    Args:
        location_name: Name of the location to explain
        user_lat: Optional user latitude for distance context
        user_lng: Optional user longitude for distance context

    Returns:
        ExplanationResponse with detailed analysis
    """
    try:
        recommender = get_recommender()
        ranker = get_ranker_agent()

        # Get location info
        location = recommender.get_location_info(location_name)

        if not location:
            return ExplanationResponse(
                location_name=location_name,
                found=False,
                detailed_reasoning=f"Location '{location_name}' not found in our database."
            )

        # Get similar locations
        similar = recommender.find_similar_locations(location_name, top_k=3)
        similar_names = [s.name for s in similar]

        # Build preference analysis
        pref_scores = location.preference_scores
        preference_analysis = {}

        categories = {
            "history": "Historical/Cultural significance",
            "adventure": "Adventure activities",
            "nature": "Nature and wildlife",
            "relaxation": "Relaxation and leisure"
        }

        for cat, desc in categories.items():
            score = pref_scores.get(cat, 0.0)
            if score >= 0.8:
                level = "Excellent"
            elif score >= 0.6:
                level = "Good"
            elif score >= 0.4:
                level = "Moderate"
            else:
                level = "Low"
            preference_analysis[f"{cat}_match"] = f"{level} ({score:.1f}) - {desc}"

        # Build constraint analysis
        constraint_analysis = {
            "typical_crowds": "Variable - check specific time",
            "weather_sensitivity": "High - outdoor location" if location.is_outdoor else "Low - indoor location",
            "poya_impact": "May be busier on Poya days for religious sites"
        }

        # Determine best times
        best_times = ["07:00-09:00", "15:00-17:00"] if location.is_outdoor else ["Any time during opening hours"]

        # Generate LLM reasoning if available
        detailed_reasoning = ""
        if ranker.llm:
            try:
                prompt = f"""Provide a detailed explanation for visiting {location.name} in Sri Lanka.

Location Details:
- Type: {"Outdoor" if location.is_outdoor else "Indoor"}
- Preference Scores: History={pref_scores.get('history', 0):.1f}, Adventure={pref_scores.get('adventure', 0):.1f}, Nature={pref_scores.get('nature', 0):.1f}, Relaxation={pref_scores.get('relaxation', 0):.1f}

Similar locations: {', '.join(similar_names)}

Write 3-4 sentences explaining what makes this location special and key tips for visitors."""

                response = await ranker.llm.ainvoke([
                    {"role": "system", "content": "You are Travion, an expert Sri Lankan tour guide. Provide helpful, accurate information."},
                    {"role": "user", "content": prompt}
                ])
                detailed_reasoning = response.content.strip()
            except Exception as e:
                logger.warning(f"LLM explanation failed: {e}")
                detailed_reasoning = f"{location.name} offers a unique experience in Sri Lanka."
        else:
            detailed_reasoning = f"{location.name} is a popular destination offering experiences across multiple categories."

        # Build tips
        tips = []
        if location.is_outdoor:
            tips.extend([
                "Arrive early to avoid crowds and heat",
                "Bring water and sun protection",
                "Check weather forecast before visiting"
            ])
        else:
            tips.extend([
                "Check opening hours before visiting",
                "Respect local customs and dress codes"
            ])

        return ExplanationResponse(
            location_name=location.name,
            found=True,
            location_info={
                "latitude": location.lat,
                "longitude": location.lng,
                "is_outdoor": location.is_outdoor,
                "preference_scores": pref_scores
            },
            preference_analysis=preference_analysis,
            constraint_analysis=constraint_analysis,
            similar_locations=similar_names,
            detailed_reasoning=detailed_reasoning,
            best_times=best_times,
            tips=tips
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    f"{settings.API_V1_PREFIX}/locations/nearby",
    tags=["Recommendations"],
    summary="Get nearby locations",
    description="Get locations sorted by distance from a given point."
)
async def get_nearby_locations(
    lat: float,
    lng: float,
    top_k: int = 5,
    max_distance_km: float = 50.0
):
    """
    Get nearby locations without preference filtering.

    Args:
        lat: User's latitude
        lng: User's longitude
        top_k: Number of locations to return
        max_distance_km: Maximum distance to consider

    Returns:
        List of nearby locations with distances
    """
    try:
        recommender = get_recommender()
        nearby = recommender.get_nearest_locations(
            user_lat=lat,
            user_lng=lng,
            top_k=top_k,
            max_distance_km=max_distance_km
        )

        return {
            "success": True,
            "request_location": {"latitude": lat, "longitude": lng},
            "locations": [
                {
                    "name": loc.name,
                    "latitude": loc.lat,
                    "longitude": loc.lng,
                    "distance_km": round(loc.distance_km, 2),
                    "is_outdoor": loc.is_outdoor,
                    "preference_scores": loc.preference_scores
                }
                for loc in nearby
            ],
            "total_found": len(nearby)
        }

    except Exception as e:
        logger.error(f"Nearby locations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TOOL ENDPOINTS
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/crowd",
    response_model=CrowdPredictionResponse,
    tags=["Tools"],
    summary="Predict crowd levels",
    description="Get crowd density prediction for a location and time."
)
async def predict_crowd(request: CrowdPredictionRequest):
    """Crowd prediction endpoint using CrowdCast model."""
    try:
        crowdcast = get_crowdcast()
        event_sentinel = get_event_sentinel()

        # Check if date is Poya
        event_info = event_sentinel.get_event_info(request.target_datetime)

        # Get prediction
        prediction = crowdcast.predict(
            location_type=request.location_type,
            target_datetime=request.target_datetime,
            is_poya=event_info["is_poya"],
            is_school_holiday=event_info["is_school_holiday"]
        )

        # Get optimal times
        optimal_times = crowdcast.find_optimal_time(
            request.location_type,
            request.target_datetime,
            is_poya=event_info["is_poya"],
            preference="low_crowd"
        )

        return CrowdPredictionResponse(
            location=request.location,
            datetime=request.target_datetime.isoformat(),
            crowd_level=prediction["crowd_level"],
            crowd_percentage=prediction["crowd_percentage"],
            crowd_status=prediction["crowd_status"],
            recommendation=prediction["recommendation"],
            optimal_times=optimal_times[:3]
        )

    except Exception as e:
        logger.error(f"Crowd prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    f"{settings.API_V1_PREFIX}/events",
    response_model=EventCheckResponse,
    tags=["Tools"],
    summary="Check events and holidays",
    description="Check if a date has Poya, holidays, or cultural events."
)
async def check_events(request: EventCheckRequest):
    """Event checking endpoint using Event Sentinel."""
    try:
        event_sentinel = get_event_sentinel()
        event_info = event_sentinel.get_event_info(request.date)

        # Check activity constraints if provided
        if request.activity and request.location_type:
            constraint_check = event_sentinel.check_activity_constraints(
                request.date,
                request.activity,
                request.location_type
            )
            if constraint_check.get("violations"):
                event_info["warnings"].extend([
                    v["message"] for v in constraint_check["violations"]
                ])

        return EventCheckResponse(
            date=request.date.strftime("%Y-%m-%d"),
            is_poya=event_info["is_poya"],
            is_school_holiday=event_info["is_school_holiday"],
            special_event=event_info.get("special_event"),
            alcohol_allowed=event_info["alcohol_allowed"],
            crowd_impact=event_info["crowd_impact"],
            warnings=event_info.get("warnings", []),
            recommendations=event_info.get("recommendations", [])
        )

    except Exception as e:
        logger.error(f"Event check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# EVENT SENTINEL v2 ENDPOINT (Temporal-Spatial Correlation)
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/events/impact",
    response_model=EventImpactResponse,
    tags=["Event Sentinel"],
    summary="Temporal-Spatial Correlation Impact Assessment",
    description="""
    Research-grade event impact analysis using Temporal-Spatial Correlation.

    **Research Features:**

    1. **High-Precision Temporal Indexing**
       - Bridge day detection (Tuesday/Thursday holidays = 4-day weekends)
       - Weekday adjacency analysis for crowd pattern prediction
       - Holiday category classification (Poya, Bank, Mercantile, Public)

    2. **Socio-Cultural Constraint Logic**
       - HARD_CONSTRAINT: Legal prohibitions (Poya alcohol ban)
       - CRITICAL_SHUTDOWN: Complete closure periods (April 13-14 New Year)
       - SOFT_CONSTRAINT: Strong advisories (modest dress at temples)

    3. **Location-Specific Sensitivity Engine**
       - Cross-references 80+ locations with thematic scores
       - `l_rel > 0.7`: EXTREME_CROWD_RISK on Poya days
       - `l_nat > 0.8`: DOMESTIC_TOURISM_PEAK on Mercantile holidays
       - Fuzzy matching for typo-tolerant location names

    **Output:**
    - `is_legal_conflict`: Boolean for hard constraint violations
    - `predicted_crowd_modifier`: Float multiplier (1.0=normal, 2.5=2.5x crowd)
    - `travel_advice_strings`: Actionable recommendations

    **Example Use Cases:**
    - Planning temple visit on Vesak (expect 3x crowds, alcohol ban)
    - Checking if April 13 activities are viable (CRITICAL_SHUTDOWN)
    - Assessing long weekend impact at nature sites
    """
)
async def get_event_impact(request: EventImpactRequest):
    """
    Event Sentinel v2: Temporal-Spatial Correlation Engine.

    Args:
        request: EventImpactRequest with location, date, and optional activity

    Returns:
        EventImpactResponse with comprehensive impact assessment
    """
    try:
        event_sentinel = get_event_sentinel()

        # Call the research-grade get_impact method
        impact_result = event_sentinel.get_impact(
            location_name=request.location_name,
            target_date=request.target_date,
            activity_type=request.activity_type
        )

        # Check for errors
        if impact_result.get("error"):
            raise HTTPException(
                status_code=400,
                detail=impact_result.get("message", "Unknown error")
            )

        # Build location sensitivity response
        loc_sens = impact_result["location_sensitivity"]
        location_sensitivity = LocationSensitivity(
            location_name=loc_sens["location_name"],
            match_confidence=loc_sens["match_confidence"],
            l_rel=loc_sens["l_rel"],
            l_nat=loc_sens["l_nat"],
            l_hist=loc_sens["l_hist"],
            l_adv=loc_sens["l_adv"],
            sensitivity_flags=loc_sens["sensitivity_flags"]
        )

        # Build temporal context response (if present)
        temporal_context = None
        if impact_result.get("temporal_context"):
            tc = impact_result["temporal_context"]
            bridge = tc["bridge_info"]
            temporal_context = TemporalIndexEntry(
                uid=tc["uid"],
                name=tc["name"],
                date=tc["date"],
                day_of_week=tc["day_of_week"],
                day_number=tc["day_number"],
                categories=tc["categories"],
                is_poya=tc["is_poya"],
                is_mercantile=tc["is_mercantile"],
                bridge_info=BridgeDayInfo(
                    is_bridge_day=bridge["is_bridge_day"],
                    bridge_type=bridge["bridge_type"],
                    potential_long_weekend_days=bridge["potential_long_weekend_days"],
                    adjacent_dates=bridge["adjacent_dates"]
                )
            )

        # Build constraints response
        constraints = [
            ConstraintInfo(
                constraint_type=c["constraint_type"],
                code=c["code"],
                severity=c["severity"],
                message=c["message"],
                affected_activities=c["affected_activities"]
            )
            for c in impact_result.get("constraints", [])
        ]

        return EventImpactResponse(
            is_legal_conflict=impact_result["is_legal_conflict"],
            predicted_crowd_modifier=impact_result["predicted_crowd_modifier"],
            travel_advice_strings=impact_result["travel_advice_strings"],
            location_sensitivity=location_sensitivity,
            temporal_context=temporal_context,
            constraints=constraints,
            is_poya_day=impact_result["is_poya_day"],
            is_new_year_shutdown=impact_result["is_new_year_shutdown"],
            is_weekend=impact_result["is_weekend"],
            is_long_weekend=impact_result["is_long_weekend"],
            activity_allowed=impact_result.get("activity_allowed"),
            activity_warnings=impact_result.get("activity_warnings", []),
            calculation_timestamp=impact_result["calculation_timestamp"],
            engine_version=impact_result["engine_version"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Event impact error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    f"{settings.API_V1_PREFIX}/golden-hour",
    response_model=GoldenHourResponse,
    tags=["Tools"],
    summary="Calculate golden hour times",
    description="Get optimal photography times for a location and date."
)
async def get_golden_hour(request: GoldenHourRequest):
    """Golden hour calculation endpoint."""
    try:
        golden_hour = get_golden_hour_agent()

        # Get optimal photo times
        photo_times = golden_hour.get_optimal_photo_times(
            request.location,
            request.date.date()
        )

        return GoldenHourResponse(
            location=request.location,
            date=request.date.strftime("%Y-%m-%d"),
            sunrise=photo_times["sun_times"]["sunrise"],
            sunset=photo_times["sun_times"]["sunset"],
            golden_hour_morning=photo_times["sun_times"]["golden_hour_morning"],
            golden_hour_evening=photo_times["sun_times"]["golden_hour_evening"],
            recommended_time=photo_times["recommended_time"],
            tips=photo_times["tips"]
        )

    except Exception as e:
        logger.error(f"Golden hour error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PHYSICS GOLDEN HOUR ENDPOINT (Research-Grade)
# =============================================================================

@app.post(
    f"{settings.API_V1_PREFIX}/physics/golden-hour",
    response_model=PhysicsGoldenHourResponse,
    tags=["Physics"],
    summary="Research-grade golden hour calculation",
    description="""
    Calculate golden hour and blue hour windows using physics-based sun elevation.

    **Research-Grade Algorithm:**
    This endpoint uses actual sun elevation angles rather than static time offsets.
    It implements the SAMP (Solar Azimuth and Magnitude Position) algorithm from
    the `astral` library, with NREL SPA algorithm fallback via `pysolar` for
    high-precision calculations in mountainous terrain.

    **Physical Definitions:**
    - **Golden Hour**: Sun elevation between -4° and +6°
      - Soft, warm light with reduced contrast
      - Long shadows without harsh edges
      - Optimal for landscape and portrait photography

    - **Blue Hour**: Sun elevation between -6° and -4°
      - Deep blue sky with residual warm horizon
      - City lights become visible
      - Ideal for architectural and twilight photography

    **Topographic Correction:**
    For elevated locations (e.g., Nuwara Eliya at 1868m), the geometric horizon
    is depressed. The dip angle shifts sunrise/sunset times by 2-4 minutes.

    **Validation:**
    Results can be validated against NOAA Solar Calculator:
    https://gml.noaa.gov/grad/solcalc/
    """
)
async def calculate_physics_golden_hour(request: PhysicsGoldenHourRequest):
    """
    Physics-based golden hour calculation endpoint.

    Uses the GoldenHourEngine with actual sun elevation calculations
    for research-grade accuracy.

    Args:
        request: PhysicsGoldenHourRequest with coordinates and date

    Returns:
        PhysicsGoldenHourResponse with complete solar timing data
    """
    from datetime import date as date_type

    try:
        # Get the physics engine
        engine = get_golden_hour_engine()

        # Parse date
        try:
            target_date = date_type.fromisoformat(request.date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {request.date}. Use YYYY-MM-DD."
            )

        # Calculate golden hour data
        result = engine.calculate(
            latitude=request.latitude,
            longitude=request.longitude,
            target_date=target_date,
            elevation_m=request.elevation_m,
            location_name=request.location_name or "Custom Location"
        )

        # Build response
        def build_time_window(tw) -> Optional[TimeWindowResponse]:
            if tw is None:
                return None
            return TimeWindowResponse(
                start=tw.start.isoformat(),
                end=tw.end.isoformat(),
                start_local=tw.start_local,
                end_local=tw.end_local,
                duration_minutes=round(tw.duration_minutes, 1),
                elevation_at_start_deg=round(tw.elevation_at_start, 2),
                elevation_at_end_deg=round(tw.elevation_at_end, 2)
            )

        response = PhysicsGoldenHourResponse(
            location=LocationInfo(
                name=result.location_name,
                latitude=result.latitude,
                longitude=result.longitude,
                elevation_m=result.elevation_m
            ),
            date=result.date,
            timezone=result.timezone,
            morning_golden_hour=build_time_window(result.morning_golden_hour),
            evening_golden_hour=build_time_window(result.evening_golden_hour),
            morning_blue_hour=build_time_window(result.morning_blue_hour),
            evening_blue_hour=build_time_window(result.evening_blue_hour),
            solar_noon=result.solar_noon,
            solar_noon_elevation_deg=round(result.solar_noon_elevation_deg, 2) if result.solar_noon_elevation_deg else None,
            sunrise=result.sunrise,
            sunset=result.sunset,
            day_length_hours=round(result.day_length_hours, 2) if result.day_length_hours else None,
            metadata=CalculationMetadata(
                topographic_correction_minutes=round(result.topographic_correction_minutes, 1),
                calculation_method=result.calculation_method,
                precision_estimate_deg=result.precision_estimate_deg
            ),
            warnings=result.warnings
        )

        # Add current position if requested
        if request.include_current_position:
            current = engine.get_current_solar_position(
                latitude=request.latitude,
                longitude=request.longitude,
                elevation_m=request.elevation_m
            )
            response.current_position = SolarPositionResponse(
                timestamp=current.timestamp.isoformat(),
                local_time=current.local_time,
                elevation_deg=round(current.elevation_deg, 2),
                azimuth_deg=round(current.azimuth_deg, 2),
                atmospheric_refraction_deg=round(current.atmospheric_refraction_deg, 4),
                is_daylight=current.is_daylight,
                light_quality=current.light_quality,
                calculation_method=current.calculation_method
            )

        return response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Physics golden hour calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    f"{settings.API_V1_PREFIX}/physics/golden-hour/{{location_name}}",
    response_model=PhysicsGoldenHourResponse,
    tags=["Physics"],
    summary="Get golden hour by location name",
    description="""
    Calculate golden hour for a known Sri Lankan location by name.

    Coordinates and elevation are automatically looked up from the locations database.
    """
)
async def get_physics_golden_hour_by_name(
    location_name: str,
    date: str,
    include_current_position: bool = False
):
    """
    Get physics-based golden hour for a named location.

    Args:
        location_name: Name of the location (e.g., "Ella", "Sigiriya")
        date: Target date in YYYY-MM-DD format
        include_current_position: Include current sun position

    Returns:
        PhysicsGoldenHourResponse
    """
    from datetime import date as date_type

    try:
        engine = get_golden_hour_engine()

        # Parse date
        try:
            target_date = date_type.fromisoformat(date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {date}. Use YYYY-MM-DD."
            )

        # Calculate using location name
        try:
            result = engine.calculate_for_location(location_name, target_date)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Build response (same as POST endpoint)
        def build_time_window(tw) -> Optional[TimeWindowResponse]:
            if tw is None:
                return None
            return TimeWindowResponse(
                start=tw.start.isoformat(),
                end=tw.end.isoformat(),
                start_local=tw.start_local,
                end_local=tw.end_local,
                duration_minutes=round(tw.duration_minutes, 1),
                elevation_at_start_deg=round(tw.elevation_at_start, 2),
                elevation_at_end_deg=round(tw.elevation_at_end, 2)
            )

        response = PhysicsGoldenHourResponse(
            location=LocationInfo(
                name=result.location_name,
                latitude=result.latitude,
                longitude=result.longitude,
                elevation_m=result.elevation_m
            ),
            date=result.date,
            timezone=result.timezone,
            morning_golden_hour=build_time_window(result.morning_golden_hour),
            evening_golden_hour=build_time_window(result.evening_golden_hour),
            morning_blue_hour=build_time_window(result.morning_blue_hour),
            evening_blue_hour=build_time_window(result.evening_blue_hour),
            solar_noon=result.solar_noon,
            solar_noon_elevation_deg=round(result.solar_noon_elevation_deg, 2) if result.solar_noon_elevation_deg else None,
            sunrise=result.sunrise,
            sunset=result.sunset,
            day_length_hours=round(result.day_length_hours, 2) if result.day_length_hours else None,
            metadata=CalculationMetadata(
                topographic_correction_minutes=round(result.topographic_correction_minutes, 1),
                calculation_method=result.calculation_method,
                precision_estimate_deg=result.precision_estimate_deg
            ),
            warnings=result.warnings
        )

        # Add current position if requested
        if include_current_position:
            current = engine.get_current_solar_position(
                latitude=result.latitude,
                longitude=result.longitude,
                elevation_m=result.elevation_m
            )
            response.current_position = SolarPositionResponse(
                timestamp=current.timestamp.isoformat(),
                local_time=current.local_time,
                elevation_deg=round(current.elevation_deg, 2),
                azimuth_deg=round(current.azimuth_deg, 2),
                atmospheric_refraction_deg=round(current.atmospheric_refraction_deg, 4),
                is_daylight=current.is_daylight,
                light_quality=current.light_quality,
                calculation_method=current.calculation_method
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Physics golden hour lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    f"{settings.API_V1_PREFIX}/physics/sun-position",
    response_model=SolarPositionResponse,
    tags=["Physics"],
    summary="Get current sun position",
    description="Get the current sun position and light quality for any location."
)
async def get_current_sun_position(
    latitude: float,
    longitude: float,
    elevation_m: float = 0.0
):
    """
    Get current sun position for a location.

    Args:
        latitude: GPS latitude (-90 to 90)
        longitude: GPS longitude (-180 to 180)
        elevation_m: Observer elevation in meters

    Returns:
        SolarPositionResponse with current sun data
    """
    try:
        engine = get_golden_hour_engine()

        current = engine.get_current_solar_position(
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation_m
        )

        return SolarPositionResponse(
            timestamp=current.timestamp.isoformat(),
            local_time=current.local_time,
            elevation_deg=round(current.elevation_deg, 2),
            azimuth_deg=round(current.azimuth_deg, 2),
            atmospheric_refraction_deg=round(current.atmospheric_refraction_deg, 4),
            is_daylight=current.is_daylight,
            light_quality=current.light_quality,
            calculation_method=current.calculation_method
        )

    except Exception as e:
        logger.error(f"Sun position calculation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SYSTEM ENDPOINTS
# =============================================================================

@app.get(
    f"{settings.API_V1_PREFIX}/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Comprehensive health check with service monitoring",
    description="Check the health of all system components including Active Guardian services."
)
async def health_check():
    """
    Enhanced health check endpoint with service health monitoring.

    Returns detailed status for:
    - LLM and graph compilation
    - Active Guardian services (Weather API, News API, Event Sentinel, CrowdCast, Golden Hour)
    - Circuit breaker status
    - Recommendation components
    """
    components = {}
    health_monitor = get_health_monitor()

    # Check agent/LLM
    try:
        agent = get_agent()
        components["llm"] = "connected" if agent.llm else "not_configured"
        components["graph"] = "compiled" if agent.graph else "not_compiled"
        if agent.llm:
            health_monitor.report_success(ServiceType.LLM)
    except Exception as e:
        components["llm"] = f"error: {str(e)}"
        components["graph"] = "error"
        health_monitor.report_failure(ServiceType.LLM, str(e))

    # Check Active Guardian Services with health monitoring
    shadow_monitor = get_shadow_monitor()

    # CrowdCast (Critical)
    try:
        crowdcast = get_crowdcast()
        components["crowdcast"] = "available"
        health_monitor.report_success(ServiceType.CROWDCAST)
    except Exception as e:
        components["crowdcast"] = f"error: {str(e)}"
        health_monitor.report_failure(ServiceType.CROWDCAST, str(e))

    # Event Sentinel (Critical)
    try:
        event_sentinel = get_event_sentinel()
        components["event_sentinel"] = "available"
        health_monitor.report_success(ServiceType.EVENT_SENTINEL)
    except Exception as e:
        components["event_sentinel"] = f"error: {str(e)}"
        health_monitor.report_failure(ServiceType.EVENT_SENTINEL, str(e))

    # Golden Hour (Critical)
    try:
        golden_hour = get_golden_hour_agent()
        components["golden_hour"] = "available"
        health_monitor.report_success(ServiceType.GOLDEN_HOUR)
    except Exception as e:
        components["golden_hour"] = f"error: {str(e)}"
        health_monitor.report_failure(ServiceType.GOLDEN_HOUR, str(e))

    # Weather API (Optional but important) - Check via Shadow Monitor
    weather_health = health_monitor.get_health(ServiceType.WEATHER_API)
    if shadow_monitor.weather_tool and shadow_monitor.weather_tool.is_configured():
        components["weather_api"] = "available"
    elif weather_health.is_unavailable():
        components["weather_api"] = f"unavailable: {weather_health.error_message}"
    elif weather_health.is_degraded():
        components["weather_api"] = f"degraded: {weather_health.error_message}"
    else:
        components["weather_api"] = "not_configured (set OPENWEATHER_API_KEY)"

    # News API (Optional - GDELT fallback available)
    news_health = health_monitor.get_health(ServiceType.NEWS_API)
    if shadow_monitor.news_alert_tool:
        components["news_api"] = "available"
    elif news_health.is_unavailable():
        components["news_api"] = f"unavailable: {news_health.error_message}"
    elif news_health.is_degraded():
        components["news_api"] = f"degraded: {news_health.error_message}"
    else:
        components["news_api"] = "not_configured (optional)"

    # ChromaDB (Vector Store)
    try:
        from .graph.nodes.retrieval import get_vectordb_service
        vectordb = get_vectordb_service()
        if vectordb.enabled:
            doc_count = vectordb.collection.count() if vectordb.collection else 0
            components["chromadb"] = f"available ({doc_count} documents)"
            health_monitor.report_success(ServiceType.CHROMADB)
        else:
            components["chromadb"] = "disabled"
            health_monitor.report_failure(ServiceType.CHROMADB, "VectorDB not enabled")
    except Exception as e:
        components["chromadb"] = f"error: {str(e)}"
        health_monitor.report_failure(ServiceType.CHROMADB, str(e))

    # Physics Engine
    try:
        engine = get_golden_hour_engine()
        components["physics_engine"] = f"available ({engine.primary_method})"
    except Exception:
        components["physics_engine"] = "error"

    # Recommendation components
    try:
        recommender = get_recommender()
        loc_count = len(recommender.locations_df) if recommender.locations_df is not None else 0
        components["recommender"] = f"available ({loc_count} locations)"
    except Exception:
        components["recommender"] = "error"

    try:
        ranker = get_ranker_agent()
        components["ranker_agent"] = "available" if ranker.graph else "graph_not_compiled"
        components["ranker_llm"] = "connected" if ranker.llm else "not_configured"
    except Exception:
        components["ranker_agent"] = "error"

    # Determine overall status based on critical services
    critical_services = [ServiceType.LLM, ServiceType.EVENT_SENTINEL, ServiceType.CROWDCAST, ServiceType.GOLDEN_HOUR]
    critical_failures = [
        svc.value for svc in critical_services
        if health_monitor.get_health(svc).is_unavailable()
    ]

    degraded_services = [
        svc.value for svc in [ServiceType.WEATHER_API, ServiceType.NEWS_API]
        if health_monitor.get_health(svc).is_degraded() or health_monitor.get_health(svc).is_unavailable()
    ]

    # Overall status determination
    if critical_failures:
        status = "unhealthy"
    elif degraded_services:
        status = "degraded"
    else:
        # Also check component errors
        errors = [v for v in components.values() if "error" in str(v).lower()]
        status = "healthy" if not errors else "degraded"

    # Build metadata with circuit breaker status
    metadata = {
        "critical_failures": critical_failures,
        "degraded_services": degraded_services,
        "strict_validation": settings.STRICT_VALIDATION,
        "circuit_breakers": {
            svc.value: {
                "is_open": health_monitor.get_circuit_breaker(svc).is_open()
                if health_monitor.get_circuit_breaker(svc) else False,
                "failure_count": health_monitor.get_circuit_breaker(svc).failure_count
                if health_monitor.get_circuit_breaker(svc) else 0
            }
            for svc in ServiceType
        },
        "service_summary": health_monitor.get_summary()
    }

    return HealthResponse(
        status=status,
        version=settings.APP_VERSION,
        components=components,
        metadata=metadata
    )


@app.get(
    f"{settings.API_V1_PREFIX}/graph",
    tags=["System"],
    summary="Get graph visualization",
    description="Returns Mermaid diagram of the agent workflow."
)
async def get_graph():
    """Return the LangGraph workflow diagram."""
    try:
        agent = get_agent()
        diagram = agent.get_graph_visualization()
        return {"diagram": diagram}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "endpoints": {
            "health": f"{settings.API_V1_PREFIX}/health",
            "chat": f"{settings.API_V1_PREFIX}/chat",
            "recommend": f"{settings.API_V1_PREFIX}/recommend",
            "explain": f"{settings.API_V1_PREFIX}/explain/{{location_name}}",
            "nearby": f"{settings.API_V1_PREFIX}/locations/nearby",
            "event_sentinel": {
                "impact": f"{settings.API_V1_PREFIX}/events/impact",
                "legacy": f"{settings.API_V1_PREFIX}/events"
            },
            "physics": {
                "golden_hour": f"{settings.API_V1_PREFIX}/physics/golden-hour",
                "golden_hour_by_name": f"{settings.API_V1_PREFIX}/physics/golden-hour/{{location_name}}",
                "sun_position": f"{settings.API_V1_PREFIX}/physics/sun-position"
            }
        }
    }


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}")
    return ErrorResponse(
        error="internal_error",
        message="An unexpected error occurred",
        details={"exception": str(exc)} if settings.DEBUG else None
    )


# =============================================================================
# SIMPLE API ENDPOINTS (Current Day Predictions)
# =============================================================================

from .simple_api_endpoints import (
    get_location_type_from_scores,
    PREFERENCE_CONTENT,
    SCORE_BASED_CONTENT,
    get_highlights_for_preference,
    get_primary_focus_from_scores,
    get_content_from_scores,
    get_highlights_from_scores,
)
from .schemas import (
    SimpleCrowdPredictionRequest,
    SimpleGoldenHourRequest,
    LocationDescriptionRequest,
    SimpleRecommendationRequest,
    SimpleCrowdPredictionResponse,
    SimpleGoldenHourResponse,
    LocationDescriptionResponse,
    SimpleRecommendationResponse,
    SimpleRecommendationLocation,
)
from difflib import get_close_matches


@app.post(
    f"{settings.API_V1_PREFIX}/simple/crowd",
    response_model=SimpleCrowdPredictionResponse,
    tags=["Simple APIs"],
    summary="Current day crowd prediction by location name",
    description="""
    Get crowd prediction for the current day by passing just the location name.
    
    The API automatically:
    - Looks up the location in the database
    - Determines the location type (Heritage, Beach, Nature, etc.)
    - Checks if today is a Poya day
    - Returns current crowd prediction and optimal times
    """
)
async def simple_crowd_prediction(request: SimpleCrowdPredictionRequest):
    """Simple crowd prediction - pass location name only."""
    try:
        now = datetime.now()
        today = now.date()
        
        recommender = get_recommender()
        location = recommender.get_location_info(request.location_name)
        
        if not location:
            all_locations = recommender.locations_df["Location_Name"].tolist() if not recommender.locations_df.empty else []
            matches = get_close_matches(request.location_name, all_locations, n=1, cutoff=0.6)
            if matches:
                location = recommender.get_location_info(matches[0])
        
        if not location:
            raise HTTPException(status_code=404, detail=f"Location not found: {request.location_name}")
        
        location_type = get_location_type_from_scores(location.preference_scores)
        
        event_sentinel = get_event_sentinel()
        event_info = event_sentinel.get_event_info(now)
        is_poya = event_info.get("is_poya", False)
        is_school_holiday = event_info.get("is_school_holiday", False)
        
        crowdcast = get_crowdcast()
        prediction = crowdcast.predict(
            location_type=location_type,
            target_datetime=now,
            is_poya=is_poya,
            is_school_holiday=is_school_holiday
        )
        
        optimal_times = crowdcast.find_optimal_time(
            location_type, now, is_poya=is_poya,
            is_school_holiday=is_school_holiday, preference="low_crowd"
        )
        
        return SimpleCrowdPredictionResponse(
            location_name=location.name,
            location_type=location_type,
            date=today.isoformat(),
            current_time=now.strftime("%H:%M"),
            crowd_level=prediction["crowd_level"],
            crowd_percentage=prediction["crowd_percentage"],
            crowd_status=prediction["crowd_status"],
            recommendation=prediction["recommendation"],
            optimal_times=optimal_times[:3],
            is_poya_day=is_poya,
            metadata={"model_type": prediction.get("model_type", "ml"),
                     "is_school_holiday": is_school_holiday,
                     "coordinates": {"lat": location.lat, "lng": location.lng}}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple crowd prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    f"{settings.API_V1_PREFIX}/simple/golden-hour",
    response_model=SimpleGoldenHourResponse,
    tags=["Simple APIs"],
    summary="Current day golden hour by location name",
    description="""
    Get golden hour times for the current day by passing just the location name.
    
    The API automatically:
    - Looks up the location coordinates
    - Calculates sunrise, sunset, and golden hour windows
    - Assesses current lighting quality
    - Provides photography tips for the location
    """
)
async def simple_golden_hour(request: SimpleGoldenHourRequest):
    """Simple golden hour - pass location name only."""
    try:
        now = datetime.now()
        today = now.date()
        
        golden_hour = get_golden_hour_agent()
        spot = golden_hour.PHOTOGRAPHY_SPOTS.get(request.location_name)
        lat, lng = None, None
        location_name = request.location_name
        
        if spot:
            lat, lng = spot["lat"], spot["lng"]
        else:
            recommender = get_recommender()
            location = recommender.get_location_info(request.location_name)
            
            if not location:
                all_locations = recommender.locations_df["Location_Name"].tolist() if not recommender.locations_df.empty else []
                all_spots = list(golden_hour.PHOTOGRAPHY_SPOTS.keys())
                all_options = all_locations + all_spots
                matches = get_close_matches(request.location_name, all_options, n=1, cutoff=0.6)
                
                if matches:
                    if matches[0] in golden_hour.PHOTOGRAPHY_SPOTS:
                        spot = golden_hour.PHOTOGRAPHY_SPOTS[matches[0]]
                        lat, lng = spot["lat"], spot["lng"]
                        location_name = matches[0]
                    else:
                        location = recommender.get_location_info(matches[0])
                        location_name = matches[0]
            
            if location:
                lat, lng = location.lat, location.lng
        
        if lat is None or lng is None:
            raise HTTPException(status_code=404, detail=f"Location not found: {request.location_name}")
        
        sun_times = golden_hour.get_sun_times(today, lat, lng, location_name)
        lighting = golden_hour.get_lighting_quality(now, lat, lng)
        photo_times = golden_hour.get_optimal_photo_times(location_name, today)
        
        return SimpleGoldenHourResponse(
            location_name=location_name,
            date=today.isoformat(),
            coordinates={"lat": lat, "lng": lng},
            sunrise=sun_times["sunrise"],
            sunset=sun_times["sunset"],
            golden_hour_morning=sun_times["golden_hour_morning"],
            golden_hour_evening=sun_times["golden_hour_evening"],
            current_lighting=lighting["quality"],
            recommended_time=photo_times["recommended_time"],
            tips=photo_times.get("tips", [])
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple golden hour error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    f"{settings.API_V1_PREFIX}/simple/description",
    response_model=LocationDescriptionResponse,
    tags=["Simple APIs"],
    summary="Generate personalized location description",
    description="""
    Generate a description of a location tailored to the user's preference scores.

    Pass preference scores (0.0 - 1.0) for:
    - history: Interest in historical/cultural sites
    - adventure: Interest in adventure activities
    - nature: Interest in nature/wildlife
    - relaxation: Interest in relaxation/spiritual experiences

    The API uses these scores to determine what aspects of the location
    to emphasize in the generated description.

    Example: High nature score (0.9) for Sigiriya focuses on flora, fauna, ecosystems.
    """
)
async def generate_location_description(request: LocationDescriptionRequest):
    """Generate personalized location description based on preference scores."""
    try:
        recommender = get_recommender()
        location = recommender.get_location_info(request.location_name)

        if not location:
            all_locations = recommender.locations_df["Location_Name"].tolist() if not recommender.locations_df.empty else []
            matches = get_close_matches(request.location_name, all_locations, n=1, cutoff=0.6)
            if matches:
                location = recommender.get_location_info(matches[0])

        if not location:
            raise HTTPException(status_code=404, detail=f"Location not found: {request.location_name}")

        # Extract preference scores from request
        pref_scores = {
            "history": request.preference.history,
            "adventure": request.preference.adventure,
            "nature": request.preference.nature,
            "relaxation": request.preference.relaxation
        }

        # Get primary focus and content based on scores
        primary_focus = get_primary_focus_from_scores(pref_scores)
        pref_content = get_content_from_scores(pref_scores)

        ranker = get_ranker_agent()
        description = ""

        if ranker.llm:
            try:
                focus_text = ", ".join(pref_content["focus"])

                # Build preference description for prompt
                pref_desc = []
                if pref_scores["nature"] >= 0.6:
                    pref_desc.append("nature and wildlife enthusiast")
                if pref_scores["history"] >= 0.6:
                    pref_desc.append("history and culture lover")
                if pref_scores["adventure"] >= 0.6:
                    pref_desc.append("adventure seeker")
                if pref_scores["relaxation"] >= 0.6:
                    pref_desc.append("seeking relaxation and peace")

                user_type = " who is also ".join(pref_desc) if pref_desc else f"{primary_focus} enthusiast"

                prompt = f"""Generate a personalized description of {location.name} in Sri Lanka for a traveler who is a {user_type}.

User Preference Scores:
- Nature interest: {pref_scores['nature']:.1f}
- History interest: {pref_scores['history']:.1f}
- Adventure interest: {pref_scores['adventure']:.1f}
- Relaxation interest: {pref_scores['relaxation']:.1f}

Primary focus should be on: {focus_text}

Write 3-4 sentences that blend these interests, emphasizing the primary focus ({primary_focus}). Be specific and evocative about {location.name}."""

                response = await ranker.llm.ainvoke([
                    {"role": "system", "content": "You are Travion, an expert Sri Lankan tour guide. Generate engaging, personalized descriptions that match user preferences."},
                    {"role": "user", "content": prompt}
                ])
                description = response.content.strip()
            except Exception as llm_err:
                logger.warning(f"LLM description failed: {llm_err}")

        if not description:
            focus_text = ", ".join(pref_content["focus"][:3])
            description = f"{location.name} offers wonderful experiences focusing on {focus_text}. This destination in Sri Lanka is perfect for those interested in {primary_focus}-related activities."

        highlights = get_highlights_from_scores(pref_scores)
        tips = [
            f"Best experienced during {pref_content['best_time'].lower()}",
            "Bring comfortable shoes for walking",
            "Carry water and sun protection"
        ]

        return LocationDescriptionResponse(
            location_name=location.name,
            preference_scores=pref_scores,
            primary_focus=primary_focus,
            description=description,
            highlights=highlights,
            best_time_to_visit=pref_content["best_time"],
            tips=tips,
            related_activities=pref_content["activities"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Location description error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    f"{settings.API_V1_PREFIX}/simple/recommend",
    response_model=SimpleRecommendationResponse,
    tags=["Simple APIs"],
    summary="Get simple location recommendations",
    description="""
    Get location recommendations by passing user location and preferences.

    The API:
    - Accepts user's current coordinates (lat, lng)
    - Accepts preference scores (history, adventure, nature, relaxation)
    - Accepts maximum distance in km
    - Returns a ranked list of recommended locations

    This is a simplified version of the full recommendation API that doesn't
    require authentication and uses content-based filtering.
    """
)
async def simple_recommend(request: SimpleRecommendationRequest):
    """Simple recommendation - pass location, preferences, and max distance."""
    try:
        recommender = get_recommender()

        # Convert preferences to 4D vector format [history, adventure, nature, relaxation]
        # The get_candidates method expects List[float], not a dict
        user_prefs = [
            request.preferences.history,
            request.preferences.adventure,
            request.preferences.nature,
            request.preferences.relaxation
        ]

        # Get candidates using the recommender
        candidates = recommender.get_candidates(
            user_preferences=user_prefs,
            user_lat=request.latitude,
            user_lng=request.longitude,
            top_k=request.top_k,
            max_distance_km=request.max_distance_km,
            outdoor_only=request.outdoor_only,  # Use request's outdoor_only filter
            exclude_locations=[]
        )

        if not candidates:
            return SimpleRecommendationResponse(
                success=True,
                user_location={"lat": request.latitude, "lng": request.longitude},
                max_distance_km=request.max_distance_km,
                total_found=0,
                recommendations=[]
            )

        # Build response with optional min_match_score filtering
        recommendations = []
        for idx, candidate in enumerate(candidates, start=1):
            # Skip candidates below minimum match score threshold
            if candidate.similarity_score < request.min_match_score:
                continue
            recommendations.append(SimpleRecommendationLocation(
                rank=idx,
                name=candidate.name,
                latitude=candidate.lat,
                longitude=candidate.lng,
                distance_km=round(candidate.distance_km, 2),
                similarity_score=round(candidate.similarity_score, 3),
                preference_scores=candidate.preference_scores,
                is_outdoor=candidate.is_outdoor,
                description=None
            ))

        # Re-rank after filtering to ensure consecutive ranks
        for idx, rec in enumerate(recommendations, start=1):
            rec.rank = idx

        return SimpleRecommendationResponse(
            success=True,
            user_location={"lat": request.latitude, "lng": request.longitude},
            max_distance_km=request.max_distance_km,
            total_found=len(recommendations),
            recommendations=recommendations
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

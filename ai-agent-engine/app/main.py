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

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json

from .config import settings
from .schemas import (
    ChatRequest,
    PlanRequest,
    CrowdPredictionRequest,
    EventCheckRequest,
    GoldenHourRequest,
    EventImpactRequest,
    PhysicsGoldenHourRequest,
    ChatResponse,
    CrowdPredictionResponse,
    EventCheckResponse,
    GoldenHourResponse,
    HealthResponse,
    ErrorResponse,
    ItinerarySlotResponse,
    ConstraintViolationResponse,
    ShadowMonitorLogResponse,
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
)
from .schemas.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendedLocation,
    ExplanationRequest,
    ExplanationResponse,
    GeoLocation,
    ConstraintCheck,
)
from .graph import get_agent, invoke_agent
from .tools import (
    get_crowdcast,
    get_event_sentinel,
    get_golden_hour_agent,
)
from .physics import get_golden_hour_engine
from .core.recommender import get_recommender
from .agents.ranker import get_ranker_agent

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup/shutdown events.

    Startup:
        - Initialize LangGraph agent
        - Connect to ChromaDB
        - Load ML models

    Shutdown:
        - Cleanup resources
    """
    # Startup
    logger.info("Starting Travion AI Engine...")

    # Initialize agent (preloads models)
    try:
        agent = get_agent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")

    # Initialize tools
    try:
        get_crowdcast()
        get_event_sentinel()
        get_golden_hour_agent()
        logger.info("Tools initialized successfully")
    except Exception as e:
        logger.warning(f"Some tools failed to initialize: {e}")

    # Initialize recommendation components
    try:
        recommender = get_recommender()
        logger.info(f"Recommender initialized: {len(recommender.locations_df)} locations")
        ranker = get_ranker_agent()
        logger.info("Ranker agent initialized")
    except Exception as e:
        logger.warning(f"Recommendation components failed to initialize: {e}")

    logger.info(f"Travion AI Engine ready on port {settings.PORT}")

    yield

    # Shutdown
    logger.info("Shutting down Travion AI Engine...")


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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        # Invoke the agent
        result = await invoke_agent(
            query=request.message,
            thread_id=request.thread_id
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

        candidates = recommender.get_candidates(
            user_preferences=user_prefs,
            user_lat=request.current_lat,
            user_lng=request.current_lng,
            top_k=request.top_k * 3,  # Get 3x for re-ranking buffer
            max_distance_km=request.max_distance_km,  # Use request's max distance (default 20km)
            outdoor_only=request.outdoor_only,
            exclude_locations=request.exclude_locations
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
    summary="Health check",
    description="Check the health of all system components."
)
async def health_check():
    """Health check endpoint."""
    components = {}

    # Check agent/LLM
    try:
        agent = get_agent()
        components["llm"] = "connected" if agent.llm else "not_configured"
        components["graph"] = "compiled" if agent.graph else "not_compiled"
    except Exception as e:
        components["llm"] = f"error: {str(e)}"
        components["graph"] = "error"

    # Check tools
    try:
        get_crowdcast()
        components["crowdcast"] = "available"
    except Exception:
        components["crowdcast"] = "error"

    try:
        get_event_sentinel()
        components["event_sentinel"] = "available"
    except Exception:
        components["event_sentinel"] = "error"

    try:
        get_golden_hour_agent()
        components["golden_hour"] = "available"
    except Exception:
        components["golden_hour"] = "error"

    try:
        engine = get_golden_hour_engine()
        components["physics_engine"] = f"available ({engine.primary_method})"
    except Exception:
        components["physics_engine"] = "error"

    # Check recommendation components
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

    # Overall status
    errors = [v for v in components.values() if "error" in str(v).lower()]
    status = "healthy" if not errors else "degraded"

    return HealthResponse(
        status=status,
        version=settings.APP_VERSION,
        components=components
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

"""
Response Schemas for Travion AI Engine API.

This module defines Pydantic models for API responses.
All responses follow a consistent structure for client consumption.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ItinerarySlotResponse(BaseModel):
    """
    Response model for a single itinerary time slot.

    Attributes:
        time: Suggested time (e.g., "16:30")
        location: Destination name
        activity: What to do
        duration_minutes: Suggested duration
        crowd_prediction: Expected crowd percentage
        lighting_quality: Golden hour assessment
        notes: Special considerations
        day: Day number in multi-day trip
        order: Order within the day
        icon: Icon name for UI display
        highlight: Whether this is a highlighted activity
        ai_insight: AI-generated insight for this activity
    """
    time: str
    location: str
    activity: str
    duration_minutes: int
    crowd_prediction: float
    lighting_quality: str
    notes: Optional[str] = None
    day: Optional[int] = None
    order: Optional[int] = None
    icon: Optional[str] = None
    highlight: Optional[bool] = False
    ai_insight: Optional[str] = None


class ConstraintViolationResponse(BaseModel):
    """
    Response model for constraint violations.

    Attributes:
        constraint_type: Type of constraint (poya_alcohol, etc.)
        description: Human-readable explanation
        severity: low, medium, high, critical
        suggestion: Corrective action
    """
    constraint_type: str
    description: str
    severity: str
    suggestion: str


class ShadowMonitorLogResponse(BaseModel):
    """
    Response model for shadow monitor log entries.

    Attributes:
        timestamp: When the check was performed
        check_type: Type of check
        result: Pass/Fail/Warning
        details: Specific findings
    """
    timestamp: str
    check_type: str
    result: str
    details: str


class ChatResponse(BaseModel):
    """
    Response model for the main chat endpoint.

    Attributes:
        query: Original user query
        intent: Classified intent
        response: Generated response text
        itinerary: Structured itinerary (if trip planning)
        constraints: Any constraint violations detected
        reasoning_logs: Shadow monitor audit trail
        metadata: Additional response metadata
    """
    query: str
    intent: Optional[str] = None
    response: str
    itinerary: Optional[List[ItinerarySlotResponse]] = None
    constraints: Optional[List[ConstraintViolationResponse]] = None
    reasoning_logs: Optional[List[ShadowMonitorLogResponse]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Plan a trip to Jungle Beach next full moon",
                "intent": "trip_planning",
                "response": "Here's an optimized plan for visiting Jungle Beach...",
                "itinerary": [
                    {
                        "time": "16:30",
                        "location": "Jungle Beach (Rumassala)",
                        "activity": "Beach visit and sunset photography",
                        "duration_minutes": 120,
                        "crowd_prediction": 45.0,
                        "lighting_quality": "golden",
                        "notes": "Poya day - no alcohol available"
                    }
                ],
                "constraints": [],
                "metadata": {
                    "reasoning_loops": 1,
                    "documents_retrieved": 5,
                    "web_search_used": False
                }
            }
        }


class TourPlanMetadataResponse(BaseModel):
    """
    Metadata for a generated tour plan.

    Attributes:
        match_score: Overall match score (0-100)
        total_days: Number of days in the plan
        total_locations: Number of locations covered
        golden_hour_optimized: Whether golden hour is optimized
        crowd_optimized: Whether crowd levels are optimized
        event_aware: Whether events/holidays are considered
    """
    match_score: int = Field(default=85, ge=0, le=100)
    total_days: int = Field(default=1, ge=1)
    total_locations: int = Field(default=1, ge=1)
    golden_hour_optimized: bool = True
    crowd_optimized: bool = True
    event_aware: bool = True


class TourPlanResponse(BaseModel):
    """
    Response model for tour plan generation endpoint.

    Attributes:
        success: Whether plan generation was successful
        thread_id: Session ID for conversation continuity
        response: Generated response text summary
        itinerary: List of itinerary slots organized by day
        metadata: Plan metadata with scores and optimization info
        constraints: Any constraint violations detected
        reasoning_logs: Shadow monitor audit trail
        warnings: List of warnings for the plan
        tips: Helpful tips for the trip
    """
    success: bool = True
    thread_id: str
    response: str
    itinerary: List[ItinerarySlotResponse] = Field(default_factory=list)
    metadata: TourPlanMetadataResponse = Field(default_factory=TourPlanMetadataResponse)
    constraints: Optional[List[ConstraintViolationResponse]] = None
    reasoning_logs: Optional[List[ShadowMonitorLogResponse]] = None
    warnings: Optional[List[str]] = None
    tips: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "thread_id": "tour_abc123",
                "response": "🗺️ Your 2-Day Sri Lanka Adventure covering 2 amazing locations!",
                "itinerary": [
                    {
                        "time": "06:30",
                        "location": "Sigiriya Rock Fortress",
                        "activity": "Arrival & Tickets",
                        "duration_minutes": 30,
                        "crowd_prediction": 15.0,
                        "lighting_quality": "golden",
                        "notes": "Collect tickets early to avoid the 8 AM rush",
                        "day": 1,
                        "order": 0,
                        "icon": "ticket",
                        "highlight": False,
                        "ai_insight": None
                    },
                    {
                        "time": "07:00",
                        "location": "Sigiriya Rock Fortress",
                        "activity": "Water Gardens",
                        "duration_minutes": 45,
                        "crowd_prediction": 20.0,
                        "lighting_quality": "golden",
                        "notes": "Best reflection shots of the rock fortress",
                        "day": 1,
                        "order": 1,
                        "icon": "water",
                        "highlight": True,
                        "ai_insight": "Golden Hour Alert: Best reflection shots of the rock fortress."
                    }
                ],
                "metadata": {
                    "match_score": 98,
                    "total_days": 2,
                    "total_locations": 2,
                    "golden_hour_optimized": True,
                    "crowd_optimized": True,
                    "event_aware": True
                },
                "warnings": ["Poya day on Jan 6 - modest dress required"],
                "tips": ["Start early to avoid crowds", "Bring water bottles"]
            }
        }


class CrowdPredictionResponse(BaseModel):
    """
    Response model for crowd prediction endpoint.

    Attributes:
        location: Location name
        datetime: Predicted datetime
        crowd_level: Crowd level (0.0 - 1.0)
        crowd_percentage: Crowd percentage (0 - 100)
        crowd_status: Status label (MINIMAL, LOW, MODERATE, HIGH, EXTREME)
        recommendation: Suggested action
        optimal_times: Best times to visit
    """
    location: str
    datetime: str
    crowd_level: float
    crowd_percentage: float
    crowd_status: str
    recommendation: str
    optimal_times: Optional[List[Dict[str, Any]]] = None


class EventCheckResponse(BaseModel):
    """
    Response model for event/holiday check endpoint.

    Attributes:
        date: Checked date
        is_poya: Whether it's a Poya day
        is_school_holiday: Whether it's a school holiday
        special_event: Name of special event (if any)
        alcohol_allowed: Whether alcohol is available
        crowd_impact: Expected crowd impact
        warnings: List of warnings
        recommendations: Suggested actions
    """
    date: str
    is_poya: bool
    is_school_holiday: bool
    special_event: Optional[str] = None
    alcohol_allowed: bool
    crowd_impact: str
    warnings: List[str]
    recommendations: List[str]


class GoldenHourResponse(BaseModel):
    """
    Response model for golden hour calculation endpoint.

    Attributes:
        location: Location name
        date: Calculation date
        sunrise: Sunrise time
        sunset: Sunset time
        golden_hour_morning: Morning golden hour window
        golden_hour_evening: Evening golden hour window
        recommended_time: Best photography time
        tips: Photography tips for the location
    """
    location: str
    date: str
    sunrise: str
    sunset: str
    golden_hour_morning: Dict[str, Any]
    golden_hour_evening: Dict[str, Any]
    recommended_time: str
    tips: List[str]


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.

    Attributes:
        status: Service health status
        version: API version
        components: Status of individual components
    """
    status: str = Field(default="healthy")
    version: str
    components: Dict[str, str] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "llm": "connected",
                    "vectordb": "connected",
                    "crowdcast": "available",
                    "event_sentinel": "available",
                    "golden_hour": "available"
                }
            }
        }


class ErrorResponse(BaseModel):
    """
    Response model for error responses.

    Attributes:
        error: Error type
        message: Human-readable error message
        details: Additional error details
    """
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# EVENT SENTINEL RESPONSE SCHEMAS (Temporal-Spatial Correlation)
# =============================================================================

class ConstraintInfo(BaseModel):
    """
    Individual constraint information.

    Represents a single constraint (hard or soft) affecting travel plans.
    """
    constraint_type: str = Field(
        ...,
        description="Type: HARD_CONSTRAINT, SOFT_CONSTRAINT, WARNING"
    )
    code: str = Field(
        ...,
        description="Machine-readable code (e.g., POYA_ALCOHOL_BAN)"
    )
    severity: str = Field(
        ...,
        description="Severity level: CRITICAL, HIGH, MEDIUM, LOW"
    )
    message: str = Field(
        ...,
        description="Human-readable constraint explanation"
    )
    affected_activities: List[str] = Field(
        default_factory=list,
        description="List of activities affected by this constraint"
    )


class BridgeDayInfo(BaseModel):
    """
    Bridge day detection information.

    Identifies potential long weekends when holidays fall near weekends.
    """
    is_bridge_day: bool = Field(
        ...,
        description="True if this holiday creates a potential long weekend"
    )
    bridge_type: Optional[str] = Field(
        None,
        description="Type: MONDAY_BRIDGE, FRIDAY_BRIDGE, DOUBLE_BRIDGE"
    )
    potential_long_weekend_days: int = Field(
        default=0,
        description="Number of consecutive off-days possible (including weekend)"
    )
    adjacent_dates: List[str] = Field(
        default_factory=list,
        description="Adjacent dates that form the long weekend"
    )


class TemporalIndexEntry(BaseModel):
    """
    High-precision temporal index for a holiday.

    Research Feature: Temporal indexing with weekday adjacency detection
    for predicting extended crowd patterns.
    """
    uid: str = Field(..., description="Unique holiday identifier")
    name: str = Field(..., description="Holiday name")
    date: str = Field(..., description="Holiday date (YYYY-MM-DD)")
    day_of_week: str = Field(..., description="Day name (Monday-Sunday)")
    day_number: int = Field(..., description="ISO weekday (1=Mon, 7=Sun)")
    categories: List[str] = Field(..., description="Holiday categories (Poya, Bank, etc.)")
    is_poya: bool = Field(..., description="Whether this is a Poya day")
    is_mercantile: bool = Field(..., description="Whether banks/offices closed")
    bridge_info: BridgeDayInfo


class LocationSensitivity(BaseModel):
    """
    Location-specific sensitivity analysis.

    Research Feature: Cross-references location thematic scores with
    calendar events to predict context-specific crowd patterns.

    Thresholds:
        - l_rel > 0.7: EXTREME_CROWD_RISK on Poya days
        - l_nat > 0.8: DOMESTIC_TOURISM_PEAK on Mercantile holidays
    """
    location_name: str = Field(..., description="Resolved location name")
    match_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fuzzy match confidence (1.0 = exact match)"
    )
    l_rel: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Religious significance score"
    )
    l_nat: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nature/outdoor score"
    )
    l_hist: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Historical significance score"
    )
    l_adv: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Adventure activity score"
    )
    sensitivity_flags: List[str] = Field(
        default_factory=list,
        description="Active sensitivity flags for this date"
    )


class EventImpactResponse(BaseModel):
    """
    Complete Event Sentinel impact assessment response.

    Research Feature: Temporal-Spatial Correlation Engine

    This response combines:
    1. High-precision temporal indexing (bridge detection, weekday analysis)
    2. Socio-cultural constraint logic (Poya rules, New Year shutdown)
    3. Location-specific sensitivity (thematic score cross-reference)

    Output Fields (per user specification):
        - is_legal_conflict: Boolean for hard constraint violations
        - predicted_crowd_modifier: Float multiplier for crowd predictions
        - travel_advice_strings: List of actionable recommendations
    """
    # Core Impact Assessment (per user requirements)
    is_legal_conflict: bool = Field(
        ...,
        description="True if activity would violate legal/cultural constraints"
    )
    predicted_crowd_modifier: float = Field(
        ...,
        ge=0.0,
        le=5.0,
        description="Crowd multiplier (1.0=normal, 2.0=2x crowd, etc.)"
    )
    travel_advice_strings: List[str] = Field(
        ...,
        description="Actionable travel recommendations"
    )

    # Detailed Analysis
    location_sensitivity: LocationSensitivity
    temporal_context: TemporalIndexEntry = Field(
        None,
        description="Holiday info if date is a holiday (null otherwise)"
    )
    constraints: List[ConstraintInfo] = Field(
        default_factory=list,
        description="All applicable constraints"
    )

    # Flags
    is_poya_day: bool = Field(..., description="Is this a Poya day?")
    is_new_year_shutdown: bool = Field(
        ...,
        description="Is this April 13-14 (CRITICAL_SHUTDOWN)?"
    )
    is_weekend: bool = Field(..., description="Is this a weekend?")
    is_long_weekend: bool = Field(
        ...,
        description="Is this part of a long weekend?"
    )

    # Activity-Specific (if activity_type provided)
    activity_allowed: Optional[bool] = Field(
        None,
        description="Whether the specified activity is permitted"
    )
    activity_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings specific to the planned activity"
    )

    # Metadata
    calculation_timestamp: str = Field(
        ...,
        description="When this assessment was generated (ISO format)"
    )
    engine_version: str = Field(
        default="2.0.0",
        description="Event Sentinel engine version"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "is_legal_conflict": True,
                "predicted_crowd_modifier": 2.5,
                "travel_advice_strings": [
                    "⚠️ POYA DAY: Alcohol sales banned island-wide",
                    "🕌 Temple of the Tooth expects 3-5x normal crowds on Vesak",
                    "📸 Arrive before 5:30 AM for photography with minimal crowds",
                    "👔 Modest dress required (cover shoulders and knees)"
                ],
                "location_sensitivity": {
                    "location_name": "Temple of the Tooth",
                    "match_confidence": 1.0,
                    "l_rel": 0.6,
                    "l_nat": 0.2,
                    "l_hist": 1.0,
                    "l_adv": 0.1,
                    "sensitivity_flags": ["HIGH_RELIGIOUS_SITE", "POYA_EXTREME_CROWD"]
                },
                "temporal_context": {
                    "uid": "sl_139",
                    "name": "Vesak Full Moon Poya Day",
                    "date": "2026-05-01",
                    "day_of_week": "Friday",
                    "day_number": 5,
                    "categories": ["Public", "Bank", "Poya"],
                    "is_poya": True,
                    "is_mercantile": False,
                    "bridge_info": {
                        "is_bridge_day": True,
                        "bridge_type": "FRIDAY_BRIDGE",
                        "potential_long_weekend_days": 4,
                        "adjacent_dates": ["2026-05-01", "2026-05-02", "2026-05-03"]
                    }
                },
                "constraints": [
                    {
                        "constraint_type": "HARD_CONSTRAINT",
                        "code": "POYA_ALCOHOL_BAN",
                        "severity": "CRITICAL",
                        "message": "Alcohol sales prohibited island-wide on Poya days",
                        "affected_activities": ["nightlife", "bar", "pub", "wine_tasting"]
                    }
                ],
                "is_poya_day": True,
                "is_new_year_shutdown": False,
                "is_weekend": False,
                "is_long_weekend": True,
                "activity_allowed": True,
                "activity_warnings": ["Expect extreme crowds at religious sites"],
                "calculation_timestamp": "2026-03-15T10:30:00+05:30",
                "engine_version": "2.0.0"
            }
        }


# =============================================================================
# PHYSICS GOLDEN HOUR RESPONSE SCHEMAS
# =============================================================================

class TimeWindowResponse(BaseModel):
    """
    Response model for a time window (golden hour, blue hour, etc.).

    Represents a period when the sun is at a specific elevation range.

    Attributes:
        start: Start time (ISO format)
        end: End time (ISO format)
        start_local: Local time string (HH:MM:SS, Asia/Colombo)
        end_local: Local time string (HH:MM:SS, Asia/Colombo)
        duration_minutes: Window duration in minutes
        elevation_at_start_deg: Sun elevation at window start
        elevation_at_end_deg: Sun elevation at window end
    """
    start: str = Field(..., description="Start time in ISO format")
    end: str = Field(..., description="End time in ISO format")
    start_local: str = Field(..., description="Local start time (HH:MM:SS)")
    end_local: str = Field(..., description="Local end time (HH:MM:SS)")
    duration_minutes: float = Field(..., description="Duration in minutes")
    elevation_at_start_deg: float = Field(..., description="Sun elevation at start (degrees)")
    elevation_at_end_deg: float = Field(..., description="Sun elevation at end (degrees)")


class LocationInfo(BaseModel):
    """Location information for physics calculation."""
    name: str
    latitude: float
    longitude: float
    elevation_m: float


class CalculationMetadata(BaseModel):
    """Metadata about the physics calculation."""
    topographic_correction_minutes: float = Field(
        ...,
        description="Time adjustment due to observer elevation"
    )
    calculation_method: str = Field(
        ...,
        description="Algorithm used (astral/pysolar/fallback)"
    )
    precision_estimate_deg: float = Field(
        ...,
        description="Estimated precision in degrees"
    )


class SolarPositionResponse(BaseModel):
    """
    Response model for current sun position.

    Provides real-time sun position data with light quality assessment.

    Attributes:
        timestamp: UTC timestamp
        local_time: Local time string (Asia/Colombo)
        elevation_deg: Sun elevation above horizon (degrees)
        azimuth_deg: Sun compass bearing (0=N, 90=E, 180=S, 270=W)
        atmospheric_refraction_deg: Refraction correction applied
        is_daylight: Whether sun is above geometric horizon
        light_quality: Current lighting classification
        calculation_method: Algorithm used for calculation
    """
    timestamp: str = Field(..., description="UTC timestamp (ISO format)")
    local_time: str = Field(..., description="Local time (HH:MM:SS)")
    elevation_deg: float = Field(..., description="Sun elevation (degrees, negative = below horizon)")
    azimuth_deg: float = Field(..., description="Sun compass bearing (degrees)")
    atmospheric_refraction_deg: float = Field(..., description="Atmospheric refraction correction")
    is_daylight: bool = Field(..., description="True if sun above horizon")
    light_quality: str = Field(
        ...,
        description="Lighting quality: golden, blue, harsh, good, dark, transitional"
    )
    calculation_method: str = Field(..., description="Algorithm used")


class PhysicsGoldenHourResponse(BaseModel):
    """
    Response model for physics-based golden hour calculation.

    This is the complete output from the GoldenHourEngine, providing
    research-grade solar timing data based on actual sun elevation angles.

    Research Definitions:
        - Golden Hour: Sun elevation -4° to +6° (soft, warm light)
        - Blue Hour: Sun elevation -6° to -4° (deep blue sky)
        - Civil Twilight: Sun elevation -6° to 0°

    Attributes:
        location: Location details
        date: Calculation date
        timezone: Timezone used (always Asia/Colombo)
        morning_golden_hour: Morning golden hour window
        evening_golden_hour: Evening golden hour window
        morning_blue_hour: Morning blue hour window
        evening_blue_hour: Evening blue hour window
        solar_noon: Time of solar noon
        solar_noon_elevation_deg: Sun elevation at solar noon
        sunrise: Geometric sunrise time
        sunset: Geometric sunset time
        day_length_hours: Total daylight duration
        current_position: Current sun position (if requested)
        metadata: Calculation metadata
        warnings: Any calculation warnings
    """
    location: LocationInfo
    date: str = Field(..., description="Calculation date (YYYY-MM-DD)")
    timezone: str = Field(default="Asia/Colombo", description="Timezone used")

    # Golden Hour Windows
    morning_golden_hour: Optional[TimeWindowResponse] = Field(
        None,
        description="Morning golden hour: sun elevation -4° to +6° (ascending)"
    )
    evening_golden_hour: Optional[TimeWindowResponse] = Field(
        None,
        description="Evening golden hour: sun elevation +6° to -4° (descending)"
    )

    # Blue Hour Windows
    morning_blue_hour: Optional[TimeWindowResponse] = Field(
        None,
        description="Morning blue hour: sun elevation -6° to -4° (ascending)"
    )
    evening_blue_hour: Optional[TimeWindowResponse] = Field(
        None,
        description="Evening blue hour: sun elevation -4° to -6° (descending)"
    )

    # Key Solar Events
    solar_noon: Optional[str] = Field(None, description="Solar noon time (HH:MM:SS)")
    solar_noon_elevation_deg: Optional[float] = Field(
        None,
        description="Maximum sun elevation at solar noon"
    )
    sunrise: Optional[str] = Field(None, description="Sunrise time (HH:MM:SS)")
    sunset: Optional[str] = Field(None, description="Sunset time (HH:MM:SS)")
    day_length_hours: Optional[float] = Field(None, description="Daylight duration (hours)")

    # Current Position (optional)
    current_position: Optional[SolarPositionResponse] = Field(
        None,
        description="Current sun position (if requested)"
    )

    # Metadata
    metadata: CalculationMetadata
    warnings: List[str] = Field(default_factory=list, description="Calculation warnings")

    class Config:
        json_schema_extra = {
            "example": {
                "location": {
                    "name": "Ella",
                    "latitude": 6.8667,
                    "longitude": 81.0667,
                    "elevation_m": 1041.0
                },
                "date": "2026-03-21",
                "timezone": "Asia/Colombo",
                "morning_golden_hour": {
                    "start": "2026-03-21T00:17:30+00:00",
                    "end": "2026-03-21T00:55:00+00:00",
                    "start_local": "05:47:30",
                    "end_local": "06:25:00",
                    "duration_minutes": 37.5,
                    "elevation_at_start_deg": -4.0,
                    "elevation_at_end_deg": 6.0
                },
                "evening_golden_hour": {
                    "start": "2026-03-21T11:53:00+00:00",
                    "end": "2026-03-21T12:30:30+00:00",
                    "start_local": "17:23:00",
                    "end_local": "18:00:30",
                    "duration_minutes": 37.5,
                    "elevation_at_start_deg": 6.0,
                    "elevation_at_end_deg": -4.0
                },
                "solar_noon": "12:12:15",
                "solar_noon_elevation_deg": 83.2,
                "sunrise": "06:08:45",
                "sunset": "18:15:30",
                "day_length_hours": 12.11,
                "metadata": {
                    "topographic_correction_minutes": 3.2,
                    "calculation_method": "astral",
                    "precision_estimate_deg": 0.5
                },
                "warnings": []
            }
        }


# =============================================================================
# SIMPLE API RESPONSE SCHEMAS
# =============================================================================

class SimpleCrowdPredictionResponse(BaseModel):
    """
    Response for simple crowd prediction API.

    Returns current day crowd prediction for a location.
    """
    location_name: str = Field(..., description="Name of the location")
    location_type: str = Field(..., description="Inferred location type")
    date: str = Field(..., description="Current date (YYYY-MM-DD)")
    current_time: str = Field(..., description="Current time (HH:MM)")
    crowd_level: float = Field(..., description="Predicted crowd level (0.0 - 1.0)")
    crowd_percentage: float = Field(..., description="Crowd percentage (0 - 100)")
    crowd_status: str = Field(..., description="Status: MINIMAL, LOW, MODERATE, HIGH, EXTREME")
    recommendation: str = Field(..., description="Visit recommendation")
    optimal_times: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Best times to visit today"
    )
    is_poya_day: bool = Field(default=False, description="Is today a Poya day")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "location_name": "Sigiriya",
                "location_type": "Heritage",
                "date": "2026-05-01",
                "current_time": "10:30",
                "crowd_level": 0.65,
                "crowd_percentage": 65.0,
                "crowd_status": "HIGH",
                "recommendation": "Expect queues. Book tickets in advance if possible.",
                "optimal_times": [
                    {"time": "06:00", "crowd_level": 0.25, "status": "LOW"},
                    {"time": "16:00", "crowd_level": 0.35, "status": "LOW"}
                ],
                "is_poya_day": True,
                "metadata": {"model_type": "ml"}
            }
        }


class SimpleGoldenHourResponse(BaseModel):
    """
    Response for simple golden hour API.

    Returns current day golden hour times for a location.
    """
    location_name: str = Field(..., description="Name of the location")
    date: str = Field(..., description="Current date (YYYY-MM-DD)")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Location coordinates")
    sunrise: str = Field(..., description="Sunrise time (HH:MM)")
    sunset: str = Field(..., description="Sunset time (HH:MM)")
    golden_hour_morning: Dict[str, Any] = Field(..., description="Morning golden hour window")
    golden_hour_evening: Dict[str, Any] = Field(..., description="Evening golden hour window")
    current_lighting: str = Field(..., description="Current lighting quality")
    recommended_time: str = Field(..., description="Best photography time")
    tips: List[str] = Field(default_factory=list, description="Photography tips")

    class Config:
        json_schema_extra = {
            "example": {
                "location_name": "Sigiriya",
                "date": "2026-05-01",
                "coordinates": {"lat": 7.957, "lng": 80.760},
                "sunrise": "06:05",
                "sunset": "18:20",
                "golden_hour_morning": {
                    "start": "05:45",
                    "end": "06:45"
                },
                "golden_hour_evening": {
                    "start": "17:35",
                    "end": "18:35"
                },
                "current_lighting": "harsh",
                "recommended_time": "05:45 - 06:45 (Sunrise golden hour)",
                "tips": [
                    "Climb early to reach top before sunrise",
                    "The lion's paws are lit beautifully at dawn"
                ]
            }
        }


class LocationDescriptionResponse(BaseModel):
    """
    Response for personalized location description API.

    Returns a description tailored to user's preference scores.
    """
    location_name: str = Field(..., description="Name of the location")
    preference_scores: Dict[str, float] = Field(..., description="User preference scores used")
    primary_focus: str = Field(..., description="Primary focus based on highest preference score")
    description: str = Field(..., description="Personalized description")
    highlights: List[str] = Field(default_factory=list, description="Key highlights for this preference")
    best_time_to_visit: Optional[str] = Field(None, description="Best time based on preference")
    tips: List[str] = Field(default_factory=list, description="Tips based on preference")
    related_activities: List[str] = Field(default_factory=list, description="Suggested activities")

    class Config:
        json_schema_extra = {
            "example": {
                "location_name": "Sigiriya",
                "preference_scores": {
                    "history": 0.2,
                    "adventure": 0.4,
                    "nature": 0.9,
                    "relaxation": 0.4
                },
                "primary_focus": "nature",
                "description": "Sigiriya is a natural wonderland rising 200 meters above the surrounding plains. Beyond its famous frescoes, the rock fortress is home to diverse ecosystems including rare orchids, medicinal plants, and over 100 bird species. The water gardens at the base showcase ancient hydraulic engineering that still supports lush vegetation today.",
                "highlights": [
                    "Diverse bird species including Sri Lanka Blue Magpie",
                    "Ancient water gardens with lotus pools",
                    "Rare orchids and medicinal plants",
                    "Panoramic views of jungle canopy"
                ],
                "best_time_to_visit": "Early morning (6-8 AM) for wildlife activity and cooler temperatures",
                "tips": [
                    "Bring binoculars for bird watching",
                    "Visit the water gardens at dawn when lotus flowers bloom",
                    "Look for giant squirrels in the surrounding trees"
                ],
                "related_activities": [
                    "Bird watching",
                    "Nature photography",
                    "Botanical exploration",
                    "Wildlife spotting"
                ]
            }
        }


# =============================================================================
# SIMPLE RECOMMENDATION API SCHEMAS
# =============================================================================

class SimpleRecommendationLocation(BaseModel):
    """A simplified recommended location."""
    rank: int = Field(..., description="Recommendation rank (1 = best)")
    name: str = Field(..., description="Location name")
    latitude: float = Field(..., description="Location latitude")
    longitude: float = Field(..., description="Location longitude")
    distance_km: float = Field(..., description="Distance from user in km")
    similarity_score: float = Field(..., description="Match score with preferences (0-1)")
    preference_scores: Dict[str, float] = Field(default_factory=dict, description="Location preference scores")
    is_outdoor: bool = Field(default=True, description="Whether location is outdoors")
    description: Optional[str] = Field(None, description="Brief description of the location")


class SimpleRecommendationResponse(BaseModel):
    """
    Response for simple recommendation API.
    
    Returns a list of recommended locations based on user preferences and location.
    """
    success: bool = Field(default=True, description="Request success status")
    user_location: Dict[str, float] = Field(..., description="User's coordinates")
    max_distance_km: float = Field(..., description="Maximum distance searched")
    total_found: int = Field(..., description="Total locations found")
    recommendations: List[SimpleRecommendationLocation] = Field(
        default_factory=list,
        description="List of recommended locations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "user_location": {"lat": 7.2906, "lng": 80.6337},
                "max_distance_km": 50.0,
                "total_found": 5,
                "recommendations": [
                    {
                        "rank": 1,
                        "name": "Sigiriya",
                        "latitude": 7.957,
                        "longitude": 80.760,
                        "distance_km": 45.3,
                        "similarity_score": 0.92,
                        "preference_scores": {
                            "history": 1.0,
                            "adventure": 0.4,
                            "nature": 0.5,
                            "relaxation": 0.1
                        },
                        "is_outdoor": True,
                        "description": "Ancient rock fortress with stunning views"
                    }
                ]
            }
        }

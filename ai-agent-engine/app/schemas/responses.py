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
    """
    time: str
    location: str
    activity: str
    duration_minutes: int
    crowd_prediction: float
    lighting_quality: str
    notes: Optional[str] = None


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
    golden_hour_morning: Dict[str, str]
    golden_hour_evening: Dict[str, str]
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

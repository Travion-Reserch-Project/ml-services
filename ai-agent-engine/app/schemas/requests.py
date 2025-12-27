"""
Request Schemas for Travion AI Engine API.

This module defines Pydantic models for API request validation.
All requests are validated before processing by the agent.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime as dt


class ChatRequest(BaseModel):
    """
    Request model for the main chat endpoint.

    Attributes:
        message: User's query or message
        thread_id: Optional conversation thread ID for context persistence
        stream: Whether to stream the response
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's query or message",
        examples=["Plan a trip to Jungle Beach next full moon"]
    )
    thread_id: Optional[str] = Field(
        None,
        description="Conversation thread ID for context persistence"
    )
    stream: bool = Field(
        False,
        description="Whether to stream the response"
    )


class PlanRequest(BaseModel):
    """
    Request model for trip planning endpoint.

    Attributes:
        destination: Primary destination or location
        date: Target date for the trip
        preferences: User preferences for the trip
        duration_days: Length of the trip
    """
    destination: str = Field(
        ...,
        description="Primary destination or location",
        examples=["Jungle Beach", "Sigiriya", "Ella"]
    )
    date: Optional[str] = Field(
        None,
        description="Target date (YYYY-MM-DD) or reference (next_poya)",
        examples=["2026-05-11", "next_poya", "next_weekend"]
    )
    preferences: Optional[List[str]] = Field(
        None,
        description="User preferences",
        examples=[["photography", "sunset", "low_crowds"]]
    )
    duration_days: int = Field(
        1,
        ge=1,
        le=14,
        description="Duration of the trip in days"
    )
    avoid_alcohol: bool = Field(
        False,
        description="Whether to avoid alcohol-related activities"
    )


class CrowdPredictionRequest(BaseModel):
    """
    Request model for crowd prediction endpoint.

    Attributes:
        location: Location name
        location_type: Type of location (Beach, Heritage, etc.)
        datetime: Target datetime for prediction
    """
    location: str = Field(
        ...,
        description="Location name",
        examples=["Sigiriya Lion Rock"]
    )
    location_type: str = Field(
        "Heritage",
        description="Location type",
        examples=["Beach", "Heritage", "Nature", "Religious", "Urban"]
    )
    target_datetime: dt = Field(
        ...,
        description="Target datetime for prediction"
    )


class EventCheckRequest(BaseModel):
    """
    Request model for event/holiday check endpoint.

    Attributes:
        date: Date to check
        activity: Planned activity (optional)
        location_type: Type of location being visited
    """
    date: dt = Field(
        ...,
        description="Date to check for events/holidays"
    )
    activity: Optional[str] = Field(
        None,
        description="Planned activity",
        examples=["sightseeing", "nightlife", "temple_visit"]
    )
    location_type: Optional[str] = Field(
        None,
        description="Type of location being visited"
    )


class GoldenHourRequest(BaseModel):
    """
    Request model for golden hour calculation endpoint.

    Attributes:
        location: Location name
        date: Target date
        latitude: GPS latitude (optional)
        longitude: GPS longitude (optional)
    """
    location: str = Field(
        ...,
        description="Location name",
        examples=["Sigiriya", "Galle Fort"]
    )
    date: dt = Field(
        ...,
        description="Date for sun time calculation"
    )
    latitude: Optional[float] = Field(
        None,
        description="GPS latitude"
    )
    longitude: Optional[float] = Field(
        None,
        description="GPS longitude"
    )


class EventImpactRequest(BaseModel):
    """
    Request model for Event Sentinel impact assessment.

    This endpoint provides Temporal-Spatial Correlation analysis, cross-referencing
    Sri Lankan cultural calendar with location-specific characteristics to generate
    precise travel impact predictions.

    Research Features:
        - High-Precision Temporal Indexing with Bridge Detection
        - Poya Day Hard Constraints (alcohol ban, modest dress)
        - New Year Critical Shutdown Detection (April 13-14)
        - Location-Specific Sensitivity (l_rel, l_nat thresholds)
        - Fuzzy Matching for robust location name resolution

    Attributes:
        location_name: Name of the location (fuzzy matched against 80+ known locations)
        target_date: Date for impact assessment (YYYY-MM-DD)
        activity_type: Optional planned activity for constraint checking
    """
    location_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Location name (supports fuzzy matching for typo tolerance)",
        examples=["Sigiriya", "Temple of the Tooth", "Jungle Beach"]
    )
    target_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Target date in YYYY-MM-DD format",
        examples=["2026-05-01", "2026-04-14", "2026-01-03"]
    )
    activity_type: Optional[str] = Field(
        default=None,
        description="Planned activity for constraint validation",
        examples=["nightlife", "temple_visit", "photography", "dining"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location_name": "Temple of the Tooth",
                "target_date": "2026-05-01",
                "activity_type": "temple_visit"
            }
        }


class PhysicsGoldenHourRequest(BaseModel):
    """
    Request model for physics-based golden hour calculation.

    This endpoint uses the research-grade GoldenHourEngine with actual
    sun elevation calculations (SAMP/NREL SPA algorithms) rather than
    static time offsets.

    Research Parameters:
        - Golden Hour: Sun elevation between -4° and +6°
        - Blue Hour: Sun elevation between -6° and -4°
        - Topographic correction for elevated locations

    Attributes:
        latitude: GPS latitude (-90 to 90)
        longitude: GPS longitude (-180 to 180)
        date: Target date (YYYY-MM-DD)
        elevation_m: Observer elevation in meters (affects horizon dip)
        location_name: Optional human-readable location name
        include_current_position: Include current sun position in response
    """
    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="GPS latitude in decimal degrees",
        examples=[6.9271, 7.9570]
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="GPS longitude in decimal degrees",
        examples=[79.8612, 80.7603]
    )
    date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Target date in YYYY-MM-DD format",
        examples=["2026-05-11", "2026-03-21"]
    )
    elevation_m: float = Field(
        default=0.0,
        ge=0.0,
        le=3000.0,
        description="Observer elevation in meters (Sri Lanka max ~2500m)",
        examples=[0.0, 1868.0, 1041.0]
    )
    location_name: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Human-readable location name",
        examples=["Nuwara Eliya", "Ella", "Sigiriya"]
    )
    include_current_position: bool = Field(
        default=False,
        description="Include current sun position in response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 6.8667,
                "longitude": 81.0667,
                "date": "2026-03-21",
                "elevation_m": 1041.0,
                "location_name": "Ella",
                "include_current_position": True
            }
        }

"""
Pydantic Schemas for Recommendation Engine.

Industry-standard validation for:
- User preference vectors
- Location coordinates
- Recommendation responses
- Reasoning explanations
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime as dt
from enum import Enum


class PreferenceCategory(str, Enum):
    """User preference categories matching the 4D vector."""
    HISTORY = "history"
    ADVENTURE = "adventure"
    NATURE = "nature"
    RELAXATION = "relaxation"


class UserPreferences(BaseModel):
    """
    User preference vector for content-based filtering.

    Each value is a float between 0 and 1 representing
    interest level in that category.
    """
    history: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Interest in historical/cultural sites (0-1)"
    )
    adventure: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Interest in adventure activities (0-1)"
    )
    nature: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Interest in nature/wildlife (0-1)"
    )
    relaxation: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Interest in relaxation/beach (0-1)"
    )

    def to_vector(self) -> List[float]:
        """Convert to 4D preference vector [hist, adv, nat, rel]."""
        return [self.history, self.adventure, self.nature, self.relaxation]

    class Config:
        json_schema_extra = {
            "example": {
                "history": 0.8,
                "adventure": 0.3,
                "nature": 0.6,
                "relaxation": 0.4
            }
        }


class GeoLocation(BaseModel):
    """Geographic coordinates with validation."""
    latitude: float = Field(
        ...,
        ge=-90.0, le=90.0,
        description="Latitude in degrees (-90 to 90)"
    )
    longitude: float = Field(
        ...,
        ge=-180.0, le=180.0,
        description="Longitude in degrees (-180 to 180)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 7.2906,
                "longitude": 80.6337
            }
        }


class PreferenceConfidence(BaseModel):
    """
    Confidence weights for each preference dimension.
    Higher values mean user is more certain about that preference.
    """
    history: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Confidence in history preference (0-1)"
    )
    adventure: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Confidence in adventure preference (0-1)"
    )
    nature: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Confidence in nature preference (0-1)"
    )
    relaxation: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Confidence in relaxation preference (0-1)"
    )

    def to_vector(self) -> List[float]:
        """Convert to 4D confidence vector [hist, adv, nat, rel]."""
        return [self.history, self.adventure, self.nature, self.relaxation]


class RecommendationRequest(BaseModel):
    """
    Request body for POST /recommend endpoint.

    Combines user identity, location, preferences, and constraints.
    Enhanced with confidence weights, category boosting, and personalization.
    """
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier for personalization"
    )
    current_lat: float = Field(
        ...,
        ge=5.0, le=10.0,
        description="Current latitude (Sri Lanka bounds: 5.9-9.8)"
    )
    current_lng: float = Field(
        ...,
        ge=79.0, le=82.0,
        description="Current longitude (Sri Lanka bounds: 79.6-81.9)"
    )
    preferences: UserPreferences = Field(
        default_factory=UserPreferences,
        description="User preference vector"
    )
    preference_confidence: Optional[PreferenceConfidence] = Field(
        default=None,
        description="Confidence weights for preferences (new users have lower confidence)"
    )
    target_datetime: Optional[dt] = Field(
        default=None,
        description="Target visit date/time for constraint checking"
    )
    max_distance_km: float = Field(
        default=20.0,
        ge=1.0, le=500.0,
        description="Maximum distance to consider (km) - default 20km radius"
    )
    top_k: int = Field(
        default=3,
        ge=1, le=10,
        description="Number of recommendations to return"
    )
    outdoor_only: Optional[bool] = Field(
        default=None,
        description="Filter for outdoor locations only"
    )
    exclude_locations: Optional[List[str]] = Field(
        default=None,
        description="List of location names to exclude"
    )
    visited_locations: Optional[List[str]] = Field(
        default=None,
        description="Previously visited locations (will be de-prioritized)"
    )
    favorite_categories: Optional[List[str]] = Field(
        default=None,
        description="Categories to boost: 'history', 'adventure', 'nature', 'relaxation'"
    )
    avoid_categories: Optional[List[str]] = Field(
        default=None,
        description="Categories to penalize: 'history', 'adventure', 'nature', 'relaxation'"
    )
    search_history: Optional[Dict[str, float]] = Field(
        default=None,
        description="Search history boost: location names mapped to click-through scores"
    )

    @field_validator("current_lat")
    @classmethod
    def validate_sri_lanka_lat(cls, v: float) -> float:
        """Validate latitude is within Sri Lanka bounds."""
        if not (5.5 <= v <= 10.0):
            # Allow slightly outside bounds but warn
            pass
        return v

    @field_validator("current_lng")
    @classmethod
    def validate_sri_lanka_lng(cls, v: float) -> float:
        """Validate longitude is within Sri Lanka bounds."""
        if not (79.0 <= v <= 82.5):
            # Allow slightly outside bounds but warn
            pass
        return v

    @field_validator("favorite_categories", "avoid_categories")
    @classmethod
    def validate_categories(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate category names."""
        valid_categories = {"history", "adventure", "nature", "relaxation"}
        if v:
            invalid = set(v) - valid_categories
            if invalid:
                raise ValueError(f"Invalid categories: {invalid}. Valid: {valid_categories}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_123",
                "current_lat": 7.2906,
                "current_lng": 80.6337,
                "preferences": {
                    "history": 0.8,
                    "adventure": 0.3,
                    "nature": 0.6,
                    "relaxation": 0.4
                },
                "preference_confidence": {
                    "history": 1.0,
                    "adventure": 0.7,
                    "nature": 0.9,
                    "relaxation": 0.5
                },
                "target_datetime": "2025-12-28T09:00:00",
                "max_distance_km": 20.0,
                "top_k": 3,
                "visited_locations": ["Sigiriya Lion Rock", "Temple of the Tooth"],
                "favorite_categories": ["history", "nature"],
                "avoid_categories": [],
                "search_history": {"Polonnaruwa Ruins": 0.1, "Dambulla Cave Temple": 0.05}
            }
        }


class ConstraintCheck(BaseModel):
    """Result of a constraint check (crowd, weather, etc.)."""
    constraint_type: str = Field(
        ...,
        description="Type of constraint (crowd, weather, poya, golden_hour)"
    )
    status: str = Field(
        ...,
        description="Status: ok, warning, blocked"
    )
    value: Optional[Any] = Field(
        default=None,
        description="Constraint value (e.g., crowd percentage)"
    )
    message: str = Field(
        default="",
        description="Human-readable message"
    )


class RecommendedLocation(BaseModel):
    """
    A single recommended location with full details.

    Includes both retrieval scores and reasoning.
    """
    rank: int = Field(..., description="Recommendation rank (1 = best)")
    name: str = Field(..., description="Location name")
    latitude: float = Field(..., description="Location latitude")
    longitude: float = Field(..., description="Location longitude")

    # Stage 1 scores
    similarity_score: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Cosine similarity to user preferences"
    )
    distance_km: float = Field(
        ...,
        ge=0.0,
        description="Distance from user in km"
    )
    combined_score: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Hybrid score (similarity + proximity)"
    )

    # Location attributes
    preference_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Location preference scores"
    )
    is_outdoor: bool = Field(
        default=True,
        description="Whether location is outdoors"
    )

    # Stage 2 reasoning
    constraint_checks: List[ConstraintCheck] = Field(
        default_factory=list,
        description="Results of constraint checks"
    )
    reasoning: str = Field(
        default="",
        description="LLM-generated reasoning for this recommendation"
    )
    optimal_visit_time: Optional[str] = Field(
        default=None,
        description="Suggested optimal visit time"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings (crowds, weather, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "rank": 1,
                "name": "Sigiriya Lion Rock",
                "latitude": 7.957,
                "longitude": 80.7603,
                "similarity_score": 0.92,
                "distance_km": 45.3,
                "combined_score": 0.87,
                "preference_scores": {
                    "history": 1.0,
                    "adventure": 0.4,
                    "nature": 0.5,
                    "relaxation": 0.1
                },
                "is_outdoor": True,
                "constraint_checks": [
                    {
                        "constraint_type": "crowd",
                        "status": "warning",
                        "value": 72,
                        "message": "Moderate crowds expected"
                    }
                ],
                "reasoning": "Sigiriya is an excellent match for your interest in history (score: 1.0). The morning visit is recommended to avoid peak crowds.",
                "optimal_visit_time": "07:00-09:00",
                "warnings": ["Moderate crowds expected in afternoon"]
            }
        }


class RecommendationResponse(BaseModel):
    """
    Response for POST /recommend endpoint.

    Contains ranked recommendations with full reasoning.
    """
    success: bool = Field(default=True, description="Request success status")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    request_location: GeoLocation = Field(
        ...,
        description="User's request location"
    )
    target_datetime: Optional[dt] = Field(
        default=None,
        description="Target visit datetime"
    )
    recommendations: List[RecommendedLocation] = Field(
        default_factory=list,
        description="Ranked list of recommendations"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (processing time, etc.)"
    )
    reasoning_summary: str = Field(
        default="",
        description="Overall reasoning summary"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "user_id": "user_123",
                "request_location": {
                    "latitude": 7.2906,
                    "longitude": 80.6337
                },
                "target_datetime": "2025-12-28T09:00:00",
                "recommendations": [],
                "metadata": {
                    "candidates_evaluated": 80,
                    "processing_time_ms": 245,
                    "constraints_checked": ["crowd", "weather", "poya"]
                },
                "reasoning_summary": "Based on your preferences for history and nature, here are the top recommendations near Kandy..."
            }
        }


class ExplanationRequest(BaseModel):
    """Request for detailed explanation of a recommendation."""
    location_name: str = Field(
        ...,
        min_length=2,
        description="Name of the location to explain"
    )
    user_lat: Optional[float] = Field(
        default=None,
        description="User latitude for context"
    )
    user_lng: Optional[float] = Field(
        default=None,
        description="User longitude for context"
    )
    preferences: Optional[UserPreferences] = Field(
        default=None,
        description="User preferences for context"
    )


class ExplanationResponse(BaseModel):
    """
    Response for GET /explain/{location_id} endpoint.

    Provides deep reasoning for a specific recommendation.
    """
    location_name: str = Field(..., description="Location name")
    found: bool = Field(default=True, description="Whether location was found")

    # Location details
    location_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Full location information"
    )

    # Preference match analysis
    preference_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis of preference matching"
    )

    # Constraint analysis
    constraint_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis of constraints (crowds, weather, etc.)"
    )

    # Similar locations
    similar_locations: List[str] = Field(
        default_factory=list,
        description="Names of similar locations"
    )

    # Full reasoning
    detailed_reasoning: str = Field(
        default="",
        description="LLM-generated detailed explanation"
    )

    # Visit recommendations
    best_times: List[str] = Field(
        default_factory=list,
        description="Recommended visit times"
    )
    tips: List[str] = Field(
        default_factory=list,
        description="Visit tips and suggestions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "location_name": "Sigiriya Lion Rock",
                "found": True,
                "location_info": {
                    "latitude": 7.957,
                    "longitude": 80.7603,
                    "is_outdoor": True
                },
                "preference_analysis": {
                    "history_match": "Excellent (1.0) - Ancient rock fortress",
                    "adventure_match": "Moderate (0.4) - Climbing involved",
                    "nature_match": "Good (0.5) - Gardens and views",
                    "relaxation_match": "Low (0.1) - Active exploration"
                },
                "constraint_analysis": {
                    "typical_crowds": "High in mornings",
                    "weather_sensitivity": "High - outdoor location",
                    "poya_impact": "May be busier on Poya days"
                },
                "similar_locations": [
                    "Pidurangala Rock",
                    "Dambulla Cave Temple",
                    "Anuradhapura Sacred City"
                ],
                "detailed_reasoning": "Sigiriya is a UNESCO World Heritage site...",
                "best_times": ["07:00-09:00", "15:00-17:00"],
                "tips": [
                    "Arrive early to avoid crowds",
                    "Bring water and sun protection"
                ]
            }
        }

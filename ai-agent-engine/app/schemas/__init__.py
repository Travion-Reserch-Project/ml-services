"""
Pydantic Schemas Package for Travion AI Engine.

This package contains all request and response models for the API.
"""

from .requests import (
    ChatRequest,
    LocationChatRequest,
    ConversationMessage,
    PlanRequest,
    TourPlanGenerateRequest,
    SelectedLocationInput,
    CrowdPredictionRequest,
    EventCheckRequest,
    GoldenHourRequest,
    EventImpactRequest,
    PhysicsGoldenHourRequest,
    ClearChatHistoryRequest,
    # Advanced Search selection (HITL)
    SelectionRequest,
    # Weather Interrupt resume (HITL)
    WeatherResumeRequest,
    # Simple API requests
    SimpleCrowdPredictionRequest,
    SimpleGoldenHourRequest,
    UserPreferenceScores,
    LocationDescriptionRequest,
    SimpleRecommendationRequest,
    # Image Search & Validation
    ImageSearchRequest,
    ImageUploadSearchRequest,
    ImageValidateRequest,
)
from .responses import (
    ChatResponse,
    ItinerarySlotResponse,
    ConstraintViolationResponse,
    ShadowMonitorLogResponse,
    TourPlanResponse,
    TourPlanMetadataResponse,
    StepResultResponse,
    ClarificationQuestionResponse,
    ClarificationOptionResponse,
    CulturalTipResponse,
    EventInfoResponse,
    CrowdPredictionResponse,
    EventCheckResponse,
    GoldenHourResponse,
    HealthResponse,
    ErrorResponse,
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
    # Vision / Image Search
    ImageSearchResultResponse,
    ImageSearchResponse,
    ImageValidateResponse,
    # Hotel/Restaurant search
    HotelSearchResultResponse,
    HotelSearchResponse,
    # Advanced Multi-Step Search
    VisualAssetResponse,
    SearchCandidateResponse,
    AdvancedSearchResponse,
    # Restaurant/Accommodation recommendations
    RestaurantRecommendationResponse,
    AccommodationRecommendationResponse,
    # Simple API responses
    SimpleCrowdPredictionResponse,
    SimpleGoldenHourResponse,
    LocationDescriptionResponse,
    SimpleRecommendationLocation,
    SimpleRecommendationResponse,
)

__all__ = [
    # Requests
    "ChatRequest",
    "LocationChatRequest",
    "ConversationMessage",
    "PlanRequest",
    "TourPlanGenerateRequest",
    "SelectedLocationInput",
    "CrowdPredictionRequest",
    "EventCheckRequest",
    "GoldenHourRequest",
    "EventImpactRequest",
    "PhysicsGoldenHourRequest",
    "ClearChatHistoryRequest",
    "SelectionRequest",
    "WeatherResumeRequest",
    # Simple API requests
    "SimpleCrowdPredictionRequest",
    "SimpleGoldenHourRequest",
    "UserPreferenceScores",
    "LocationDescriptionRequest",
    "SimpleRecommendationRequest",
    # Image Search & Validation requests
    "ImageSearchRequest",
    "ImageUploadSearchRequest",
    "ImageValidateRequest",
    # Responses
    "ChatResponse",
    "ItinerarySlotResponse",
    "ConstraintViolationResponse",
    "ShadowMonitorLogResponse",
    "TourPlanResponse",
    "TourPlanMetadataResponse",
    "StepResultResponse",
    "ClarificationQuestionResponse",
    "ClarificationOptionResponse",
    "CulturalTipResponse",
    "EventInfoResponse",
    "CrowdPredictionResponse",
    "EventCheckResponse",
    "GoldenHourResponse",
    "HealthResponse",
    "ErrorResponse",
    # Event Sentinel (Temporal-Spatial Correlation)
    "ConstraintInfo",
    "BridgeDayInfo",
    "TemporalIndexEntry",
    "LocationSensitivity",
    "EventImpactResponse",
    # Physics Golden Hour
    "TimeWindowResponse",
    "LocationInfo",
    "CalculationMetadata",
    "SolarPositionResponse",
    "PhysicsGoldenHourResponse",
    # Vision / Image Search
    "ImageSearchResultResponse",
    "ImageSearchResponse",
    "ImageValidateResponse",
    # Hotel/Restaurant search
    "HotelSearchResultResponse",
    "HotelSearchResponse",
    # Advanced Multi-Step Search
    "VisualAssetResponse",
    "SearchCandidateResponse",
    "AdvancedSearchResponse",
    # Restaurant/Accommodation recommendations
    "RestaurantRecommendationResponse",
    "AccommodationRecommendationResponse",
    # Simple API responses
    "SimpleCrowdPredictionResponse",
    "SimpleGoldenHourResponse",
    "LocationDescriptionResponse",
    "SimpleRecommendationLocation",
    "SimpleRecommendationResponse",
]

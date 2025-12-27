"""
Pydantic Schemas Package for Travion AI Engine.

This package contains all request and response models for the API.
"""

from .requests import (
    ChatRequest,
    PlanRequest,
    CrowdPredictionRequest,
    EventCheckRequest,
    GoldenHourRequest,
    EventImpactRequest,
    PhysicsGoldenHourRequest,
)
from .responses import (
    ChatResponse,
    ItinerarySlotResponse,
    ConstraintViolationResponse,
    ShadowMonitorLogResponse,
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
)

__all__ = [
    # Requests
    "ChatRequest",
    "PlanRequest",
    "CrowdPredictionRequest",
    "EventCheckRequest",
    "GoldenHourRequest",
    "EventImpactRequest",
    "PhysicsGoldenHourRequest",
    # Responses
    "ChatResponse",
    "ItinerarySlotResponse",
    "ConstraintViolationResponse",
    "ShadowMonitorLogResponse",
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
]

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class TransportPredictionRequest(BaseModel):
    """Request schema for transport prediction"""
    distance_km: float = Field(..., ge=2, le=120, description="Distance in kilometers (2-120)")
    weather: str = Field(..., description="Weather condition: sunny, rain, or storm")
    is_friday: int = Field(0, ge=0, le=1, description="1 if Friday, 0 otherwise")
    is_poya_day: int = Field(0, ge=0, le=1, description="1 if Poya day, 0 otherwise")
    traffic_score: float = Field(0.3, ge=0, le=1, description="Traffic intensity (0-1)")
    area_type: str = Field("urban", description="Area type: urban or village")
    is_peak_hours: int = Field(0, ge=0, le=1, description="1 if peak hours, 0 otherwise")
    is_weekend: int = Field(0, ge=0, le=1, description="1 if weekend, 0 otherwise")
    is_long_weekend: int = Field(0, ge=0, le=1, description="1 if long weekend, 0 otherwise")
    trip_hour: Optional[int] = Field(
        None,
        ge=0,
        le=23,
        description="Optional trip hour in 24h format (0-23) used for time-based overrides",
    )


class TransportPredictionResponse(BaseModel):
    """Response schema for transport prediction"""
    prediction: str
    confidence: float
    all_scores: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    status: str
    total: int
    predictions: List[dict]

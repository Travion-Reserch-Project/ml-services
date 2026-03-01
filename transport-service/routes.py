"""
Application routes registration module

This module centralizes all route imports and registrations.
It acts as a single entry point for managing all API routes.
"""

from fastapi import FastAPI, APIRouter, HTTPException
from app.schemas import (
    TransportPredictionRequest,
    TransportPredictionResponse,
    BatchPredictionResponse,
)
from app.controllers.prediction import (
    predict_best_transport,
    batch_predictions,
    health_check,
    model_info,
)

# Create router for transport routes
transport_router = APIRouter(prefix="/api/transport", tags=["transport"])


@transport_router.post("/predict", response_model=TransportPredictionResponse)
async def predict_route(request: TransportPredictionRequest):
    """
    Get the best transport recommendation for a single trip
    
    Returns:
        - prediction: Recommended transport type
        - confidence: Confidence score (0-1)
        - all_scores: Scores for all transport types
    """
    try:
        return await predict_best_transport(request)
    except ValueError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Models not available: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@transport_router.post("/batch", response_model=BatchPredictionResponse)
async def batch_route(requests: list[TransportPredictionRequest]):
    """
    Get transport recommendations for multiple trips in one request
    
    Args:
        requests: List of TransportPredictionRequest objects
        
    Returns:
        - status: "success" or "partial"
        - total: Number of predictions processed
        - predictions: List of prediction results
    """
    try:
        return await batch_predictions(requests)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@transport_router.get("/health")
async def health_route():
    """Health check endpoint"""
    return await health_check()


@transport_router.get("/models/info")
async def model_info_route():
    """Get information about loaded models"""
    try:
        return await model_info()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


def register_routes(app: FastAPI) -> None:
    """
    Register all application routes
    
    Args:
        app: FastAPI application instance
    """
    app.include_router(transport_router)

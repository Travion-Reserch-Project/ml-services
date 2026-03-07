import logging
from typing import List

from app.schemas import (
    TransportPredictionRequest,
    TransportPredictionResponse,
    BatchPredictionResponse,
)
from app.services import PredictionService, ModelService

logger = logging.getLogger(__name__)


async def predict_best_transport(request: TransportPredictionRequest):
    """
    Get the best transport recommendation for a single trip
    
    Args:
        request: TransportPredictionRequest with trip details
        
    Returns:
        TransportPredictionResponse with prediction and confidence scores
    """
    try:
        result = PredictionService.predict(request)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise ValueError(f"Models not available: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise Exception(f"Prediction failed: {str(e)}")


async def batch_predictions(requests: List[TransportPredictionRequest]):
    """
    Get transport recommendations for multiple trips in one request
    
    Args:
        requests: List of TransportPredictionRequest objects
        
    Returns:
        Dictionary with status, total count, and predictions list
    """
    if not requests:
        raise ValueError("Empty request list")
    
    if len(requests) > 1000:
        raise ValueError("Maximum 1000 predictions per batch")
    
    results = []
    errors = 0
    
    for idx, req in enumerate(requests):
        try:
            result = PredictionService.predict(req)
            results.append(result.dict())
        except Exception as e:
            logger.error(f"Batch prediction error at index {idx}: {e}")
            results.append({
                "error": str(e),
                "index": idx
            })
            errors += 1
    
    return {
        "status": "success" if errors == 0 else "partial",
        "total": len(results),
        "predictions": results
    }


async def health_check():
    """
    Health check endpoint
    
    Returns:
        Dictionary with status and models_loaded flag
    """
    models_loaded = ModelService.are_models_loaded()
    return {
        "status": "healthy",
        "models_loaded": models_loaded
    }


async def model_info():
    """
    Get information about loaded models
    
    Returns:
        Dictionary with model information and available classes
    """
    try:
        info = ModelService.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise Exception(str(e))

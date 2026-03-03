import logging
import numpy as np
from .model_service import ModelService
from app.schemas import TransportPredictionRequest, TransportPredictionResponse

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making transport predictions"""
    
    @staticmethod
    def predict(request: TransportPredictionRequest) -> TransportPredictionResponse:
        """
        Make a transport prediction based on input features
        
        Args:
            request: TransportPredictionRequest with all required features
            
        Returns:
            TransportPredictionResponse with prediction and confidence scores
            
        Raises:
            ValueError: If models are not loaded or input validation fails
        """
        if not ModelService.are_models_loaded():
            raise ValueError("ML models not loaded. Models directory may be missing.")
        
        try:
            # Encode categorical inputs
            weather_enc = ModelService.encode_weather(request.weather)
            area_enc = ModelService.encode_area(request.area_type)
            
            # Create feature vector in correct order
            X = np.array([[
                request.distance_km,
                weather_enc,
                request.is_weekend,
                request.is_long_weekend,
                request.is_friday,
                request.is_poya_day,
                request.traffic_score,
                area_enc,
                request.is_peak_hours
            ]])
            
            # Get prediction
            pred_idx, probs = ModelService.predict(X)
            
            prediction_class = ModelService.get_transport_class(pred_idx)
            all_scores = ModelService.get_all_scores(probs)
            
            return TransportPredictionResponse(
                prediction=prediction_class,
                confidence=float(probs[pred_idx]),
                all_scores=all_scores
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

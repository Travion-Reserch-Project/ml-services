import logging
import numpy as np
from typing import Optional
from .model_service import ModelService
from app.schemas import TransportPredictionRequest, TransportPredictionResponse

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for making transport predictions"""

    @staticmethod
    def _is_night_time(hour: Optional[int]) -> bool:
        if hour is None:
            return False
        return hour >= 22 or hour <= 5

    @staticmethod
    def _resolve_car_label(all_scores: dict) -> Optional[str]:
        for label in all_scores.keys():
            if str(label).lower() == "car":
                return label
        return None
    
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
            logger.info("Predict request payload: %s", request.dict())

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

            # Business override: late-night rides under 40 km should prefer car.
            if PredictionService._is_night_time(request.trip_hour) and request.distance_km < 40:
                car_label = PredictionService._resolve_car_label(all_scores)
                if car_label is not None:
                    for label in list(all_scores.keys()):
                        all_scores[label] = 1.0 if label == car_label else 0.0
                    prediction_class = car_label
                    confidence = 1.0
                else:
                    logger.warning("Night override active, but 'car' label not found in model classes")
                    confidence = float(probs[pred_idx])
            else:
                confidence = float(probs[pred_idx])
            
            return TransportPredictionResponse(
                prediction=prediction_class,
                confidence=confidence,
                all_scores=all_scores
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

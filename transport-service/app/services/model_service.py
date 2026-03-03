import logging
import os
import joblib
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Global model variables
MODEL_PATH = "ml-models"
TRANSPORT_CLASSIFIER_PATH = os.path.join(MODEL_PATH, "transport_classifier.pkl")
LABEL_ENCODERS_WEATHER_PATH = os.path.join(MODEL_PATH, "label_encoders.pkl")
LABEL_ENCODERS_AREA_PATH = os.path.join(MODEL_PATH, "label_encoders_area.pkl")
LABEL_ENCODERS_TARGET_PATH = os.path.join(MODEL_PATH, "label_encoders_target.pkl")


class ModelService:
    """Service for managing ML models and predictions"""
    
    _instance = None
    _model = None
    _le_weather = None
    _le_area = None
    _le_target = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load_models(cls) -> bool:
        """
        Load all required models and label encoders
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            logger.info("Loading ML models...")
            
            if not os.path.exists(MODEL_PATH):
                logger.warning(f"Model directory not found: {MODEL_PATH}")
                return False
            
            if not os.path.exists(TRANSPORT_CLASSIFIER_PATH):
                logger.warning(f"Transport classifier not found: {TRANSPORT_CLASSIFIER_PATH}")
                return False
            
            cls._model = joblib.load(TRANSPORT_CLASSIFIER_PATH)
            
            # Load label encoders - these are optional but log if missing
            try:
                cls._le_weather = joblib.load(LABEL_ENCODERS_WEATHER_PATH)
            except FileNotFoundError:
                logger.warning(f"Weather label encoder not found: {LABEL_ENCODERS_WEATHER_PATH}")
            
            try:
                cls._le_area = joblib.load(LABEL_ENCODERS_AREA_PATH)
            except FileNotFoundError:
                logger.warning(f"Area label encoder not found: {LABEL_ENCODERS_AREA_PATH}")
            
            try:
                cls._le_target = joblib.load(LABEL_ENCODERS_TARGET_PATH)
            except FileNotFoundError:
                logger.warning(f"Target label encoder not found: {LABEL_ENCODERS_TARGET_PATH}")
            
            logger.info("✓ Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    @classmethod
    def are_models_loaded(cls) -> bool:
        """Check if models are loaded"""
        return cls._model is not None
    
    @classmethod
    def predict(cls, features: np.ndarray) -> tuple:
        """
        Make a prediction using the loaded model
        
        Args:
            features: Input feature vector
            
        Returns:
            tuple: (predicted_index, probabilities)
            
        Raises:
            ValueError: If models are not loaded
        """
        if cls._model is None:
            raise ValueError("Model not loaded. Please load models first.")
        
        pred_idx = cls._model.predict(features)[0]
        probs = cls._model.predict_proba(features)[0]
        
        return pred_idx, probs
    
    @classmethod
    def encode_weather(cls, weather: str) -> Optional[int]:
        """Encode weather string to integer"""
        if cls._le_weather is None:
            raise ValueError("Weather label encoder not loaded")
        return cls._le_weather.transform([weather.lower()])[0]
    
    @classmethod
    def encode_area(cls, area: str) -> Optional[int]:
        """Encode area type string to integer"""
        if cls._le_area is None:
            raise ValueError("Area label encoder not loaded")
        return cls._le_area.transform([area.lower()])[0]
    
    @classmethod
    def get_transport_class(cls, idx: int) -> str:
        """Get transport class name from index"""
        if cls._le_target is None:
            raise ValueError("Target label encoder not loaded")
        return cls._le_target.classes_[idx]
    
    @classmethod
    def get_all_scores(cls, probs: np.ndarray) -> dict:
        """Convert probability array to dict mapping class names to scores"""
        if cls._le_target is None:
            raise ValueError("Target label encoder not loaded")
        
        return {
            cls._le_target.classes_[i]: float(probs[i])
            for i in range(len(cls._le_target.classes_))
        }
    
    @classmethod
    def get_model_info(cls) -> dict:
        """Get information about loaded models"""
        info = {
            "model_loaded": cls._model is not None,
            "label_encoders_loaded": all([
                cls._le_weather,
                cls._le_area,
                cls._le_target
            ]),
        }
        
        if cls._le_target is not None:
            info["transport_types"] = list(cls._le_target.classes_)
        
        if cls._le_weather is not None:
            info["weather_types"] = list(cls._le_weather.classes_)
        
        if cls._le_area is not None:
            info["area_types"] = list(cls._le_area.classes_)
        
        return info

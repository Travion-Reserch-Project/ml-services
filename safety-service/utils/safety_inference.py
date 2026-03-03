# Standard library imports
import os
import json

# Third-party imports for model handling and data processing
import joblib  # For loading serialized model artifacts
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature columns expected by the model
# Includes location data, place characteristics, temporal features, and security indicators
FEATURE_COLS = [
    'lat', 'lon', 'area_cluster', 'is_beach',
    'is_crowded', 'is_tourist_place', 'is_transit',
    'hour', 'day_of_week', 'is_weekend', 'police_nearby'
]

# Target risk types that the model predicts
# Each represents a specific type of safety concern for tourists
RISK_TARGETS = [
    'risk_harassment',
    'risk_pickpocket',
    'risk_scam',
    'risk_bag_snatching',
    'risk_theft',
    'risk_extortion',
    'risk_money_theft'
]

# Risk level labels used for classification
DEFAULT_LABELS = ["Low", "Medium", "High"]

#This class encapsulates all inference logic so it can be reused by FastAPI endpoints, batch prediction, or future backend integrations
class SafetyService: 
    """
    Main service class for safety risk predictions.
    Handles loading model artifacts, preprocessing, prediction, and postprocessing.
    """
    
    def __init__(self, artifact_path: str):
        """Initialize the safety service with model artifacts.
        
        Args:
            artifact_path: Path to the serialized model artifact file
        """
        # Initialize model components as None until loaded
        self.model = None
        self.scaler = None
        self.target_encoders = None
        
        # Metadata containing feature information and target labels
        self.meta: Dict[str, Any] = {
            "feature_cols": FEATURE_COLS,
            "targets": RISK_TARGETS,
            "labels": DEFAULT_LABELS
        }
        
        # Load all model artifacts from disk
        self._load_artifacts(artifact_path)

    def _load_artifacts(self, artifact_path: str):
        """Load model artifacts from disk or MLflow.
        
        Supports two formats:
        1. Unified pickle file containing model, scaler, encoders, and metadata
        2. Separate files for model, scaler, and metadata
        
        Args:
            artifact_path: Path to the main artifact file
        """
        try:
            artifact = Path(artifact_path)
            if artifact.exists():
                # Primary format: single pickle with dict containing all components
                obj = joblib.load(artifact)
                
                if isinstance(obj, dict):
                    # Extract all components from unified artifact
                    self.model = obj.get("model")
                    self.scaler = obj.get("scaler")
                    self.target_encoders = obj.get("target_encoders")
                    meta = obj.get("meta")
                    if isinstance(meta, dict):
                        self.meta.update(meta)
                    logger.info("✓ Loaded unified safety artifact")
                else:
                    # Fallback: load model and look for companion files in same directory
                    self.model = obj
                    cache_dir = artifact.parent
                    scaler_fp = cache_dir / "safety_scaler.pkl"
                    meta_fp = cache_dir / "safety_meta.json"
                    if scaler_fp.exists():
                        self.scaler = joblib.load(scaler_fp)
                        logger.info("✓ Loaded safety scaler")
                    if meta_fp.exists():
                        with open(meta_fp, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            if isinstance(meta, dict):
                                self.meta.update(meta)
                        logger.info("✓ Loaded safety metadata")
                    logger.info("✓ Loaded safety model")
            else:
                logger.warning(f"⚠️ Safety model artifact not found at {artifact}")
        except Exception as e:
            logger.error(f"✗ Failed to load safety artifacts: {e}")

    def status(self) -> Dict[str, Any]:
        """Return service status including loaded components and configuration.
        
        Returns:
            Dictionary containing loading status and model configuration
        """
        return {
            "model": "loaded" if self.model is not None else "not_loaded",
            "scaler": "loaded" if self.scaler is not None else "not_loaded",
            "feature_count": len(self.meta.get("feature_cols", FEATURE_COLS)),
            "targets": self.meta.get("targets", RISK_TARGETS),
        }

    def _to_dataframe(self, items: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert input data to DataFrame with proper column order.
        
        Ensures all required features are present, filling missing ones with 0.
        
        Args:
            items: List of feature dictionaries
            
        Returns:
            DataFrame with features in correct order
        """
        df = pd.DataFrame(items)
        # Ensure all expected features are present (missing features filled with 0)
        for col in self.meta.get("feature_cols", FEATURE_COLS):
            if col not in df.columns:
                df[col] = 0
        # Return DataFrame with columns in the correct order
        return df[self.meta.get("feature_cols", FEATURE_COLS)]

    def _preprocess(self, X_df: pd.DataFrame) -> np.ndarray:
        """Apply feature scaling if scaler is available.
        
        Args:
            X_df: DataFrame containing feature values
            
        Returns:
            Numpy array of processed features (scaled or raw)
        """
        X = X_df.values
        if self.scaler is not None:
            try:
                # Apply standardization/normalization using fitted scaler
                return self.scaler.transform(X)
            except Exception as e:
                logger.warning(f"Scaler transform failed: {e}, using raw features")
                return X
        # Return raw features if no scaler is available
        return X

    def _postprocess(self, preds: np.ndarray) -> List[Dict[str, Any]]:
        """Convert model predictions to labeled format.
        
        Decodes numerical predictions back to human-readable labels (Low/Medium/High).
        
        Args:
            preds: Numpy array of model predictions
            
        Returns:
            List of dictionaries mapping risk types to predicted labels
        """
        results: List[Dict[str, Any]] = []
        for prediction in preds:
            decoded_output: Dict[str, Any] = {}
            targets = self.meta.get("targets", RISK_TARGETS)
            
            # Decode each risk type prediction using its respective encoder
            for i, col in enumerate(targets):
                decoded_output[col] = self.target_encoders[col].inverse_transform(
                    [prediction[i]]
                )[0]

            results.append(decoded_output)

        return results

    def predict_one(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict risk levels for a single location.
        
        Args:
            features: Dictionary containing location features
            
        Returns:
            Dictionary mapping risk types to predicted levels
        """
        # Leverage batch prediction for single item
        batch = self.predict_batch([features])
        return batch[0] if batch else {}

    def predict_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict risk levels for multiple locations.
        
        Args:
            items: List of feature dictionaries
            
        Returns:
            List of dictionaries with risk predictions for each location
        """
        # Convert input to DataFrame with proper feature ordering
        X_df = self._to_dataframe(items)

        # Apply preprocessing (scaling)
        Xp = self._preprocess(X_df)

        if self.model is None:
            # Fallback mode: use simple heuristic if model isn't loaded
            # Calculates risk based on location characteristics and time
            logger.warning("⚠️ Model not loaded, using heuristic baseline")
            results = []
            for i in range(len(X_df)):
                row = X_df.iloc[i]
                # Base risk score calculated from various factors
                base = 0
                # Add risk points for each risk factor
                base += int(row.get('is_crowded', 0))  # Crowded areas
                base += int(row.get('is_tourist_place', 0))  # Tourist hotspots
                base += int(row.get('is_beach', 0))  # Beach locations
                
                # Evening/night hours (17:00-23:59 or 00:00-05:00) increase risk
                hour = int(row.get('hour', 12))
                if hour >= 17 or hour <= 5:
                    base += 1
                
                # Weekend increases risk
                if int(row.get('is_weekend', 0)) == 1:
                    base += 1
                
                # Absence of police increases risk
                if int(row.get('police_nearby', 0)) == 0:
                    base += 1
                
                # Map accumulated risk score to categorical label
                # 0-1: Low, 2-3: Medium, 4+: High
                if base <= 1:
                    label = "Low"
                elif base <= 3:
                    label = "Medium"
                else:
                    label = "High"
                
                # Apply same label to all risk types in heuristic mode
                out = {t: label for t in self.meta.get("targets", RISK_TARGETS)}
                results.append(out)
            return results

        # Normal mode: use trained ML model for predictions
        try:
            # Generate predictions using loaded model
            y_pred = self.model.predict(Xp)
            
            # Normalize prediction format to numpy array
            if isinstance(y_pred, list):
                y_pred = np.column_stack(y_pred)
            elif isinstance(y_pred, pd.DataFrame):
                y_pred = y_pred.values
            
            # Convert predictions to human-readable format
            return self._postprocess(y_pred)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Initialize the safety service with model path
    service = SafetyService(
        artifact_path='model/safety_risk_model.pkl'
    )
    
    # Define test features for a sample location
    # This represents a crowded tourist beach at night without police presence
    test_features = {
        'lat': 6.9271,  # Latitude (Colombo, Sri Lanka)
        'lon': 79.8612,  # Longitude
        'area_cluster': 0,  # Area clustering identifier
        'is_beach': 1,  # Is a beach location
        'is_crowded': 1,  # Crowded area
        'is_tourist_place': 1,  # Tourist destination
        'is_transit': 0,  # Not a transit hub
        'hour': 22,  # 10 PM (night time)
        'day_of_week': 5,  # Friday
        'is_weekend': 1,  # Weekend
        'police_nearby': 0  # No police presence
    }
    
    # Run prediction and display results
    print("\n🧪 Testing Safety Service:\n")
    result = service.predict_one(test_features)
    print(f"Risk Predictions for location ({test_features['lat']}, {test_features['lon']}):")
    for risk_type, level in result.items():
        print(f"   {risk_type}: {level}")
